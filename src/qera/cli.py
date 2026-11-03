"""命令列工具。

    python -m qera.cli convert <benchmark> [adapter 參數] --out data/full/<benchmark>.jsonl
    python -m qera.cli validate data/*.jsonl
    python -m qera.cli stats data/*.jsonl
    python -m qera.cli sample data/full/x.jsonl --per-class 250 --seed 0 --out data/x.balanced.jsonl
    python -m qera.cli show data/x.jsonl --index 0 --condition QEA
    python -m qera.cli audit data/x.jsonl --n 10 --out audit/x.md
    python -m qera.cli retokenize data/x.jsonl --out data/x.qwen.jsonl   # 先設 QERA_TOKENIZER
    python -m qera.cli apply-overrides data/x.jsonl overrides/kai.jsonl --out data/x.v2.jsonl

結束代碼（exit code）：0 = 成功；1 = 失敗；3 = 沒有可用標籤而略過。
批次腳本 convert_all.sh 依這些代碼判斷要不要繼續。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import tokens
from .render import CONDITIONS, build_verifier_input, e_candidates, render_segment
from .schema import SCHEMA_VERSION, Role, Trajectory

ADAPTERS = {
    "tau2": "qera.adapters.tau2",
    "swebench": "qera.adapters.swebench",
    "agentrewardbench": "qera.adapters.agentrewardbench",
}
EXIT_NO_LABELS = 3


def _parse_trajectory(line: str) -> Trajectory:
    # 舊版 token 欄位的語意不同，不能套用新版預設值後默默繼續。
    version = json.loads(line).get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version={version!r}，目前版本為 {SCHEMA_VERSION}；請重跑 scripts/convert_all.sh"
        )
    return Trajectory.model_validate_json(line)


def read_jsonl(paths: Iterable[str]) -> Iterator[Trajectory]:
    for p in paths:
        with open(p, encoding="utf-8") as f:  # 檔案迭代只依 \n 切行
            for ln, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield _parse_trajectory(line)
                    except Exception as e:
                        raise ValueError(f"{p} 第 {ln} 行無法解析：{e}") from e


def write_jsonl(trajs: Iterable[Trajectory], out: str) -> int:
    """先寫暫存檔，全部成功才替換正式檔。

    為什麼：之前直接覆寫，轉到一半出錯會留下一個「只有一半」的正式檔，
    下游拿到也看不出來。現在失敗時正式檔維持原狀（或不存在）。
    """
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_name(out_p.name + ".tmp")
    n = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for t in trajs:
                f.write(t.model_dump_json(exclude_none=False) + "\n")
                n += 1
        os.replace(tmp, out_p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return n


def code_version() -> str:
    """qera 程式碼的指紋，記在 manifest 裡，事後才能知道資料是哪一版程式產生的。"""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _get(obj: Any, dotted: str) -> Any:
    """取巢狀欄位，例如 solver.model、meta.tau2_domain。"""
    cur: Any = obj
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


# ---------------- 子命令 ----------------


def cmd_convert(args, extra) -> None:
    mod = importlib.import_module(ADAPTERS[args.benchmark])
    p = argparse.ArgumentParser(prog=f"qera convert {args.benchmark}")
    mod.add_args(p)
    p.add_argument("--out", required=True)
    a = p.parse_args(extra)
    n = write_jsonl(mod.run(a), a.out)
    print(
        f"[convert] {args.benchmark}: 寫出 {n} 筆 → {a.out}（tokenizer={tokens.name()}）"
    )


def check_trajectory(t: Trajectory, include_warnings: bool = True) -> list[str]:
    """schema 以外的語意檢查。回傳問題清單；警告以「[警告]」開頭。

    錯誤 = 資料本身有問題，下游不能用；警告 = 資料合法但這筆對某些分析沒有用（抽樣時會自動排除）。
    """
    problems = []
    if include_warnings and not any(s.has_evidence() for s in t.segments):
        problems.append("[警告] 沒有 E（QEA 與 QA 會一樣；抽樣時自動排除）")
    if t.label.correct is None and t.label.source not in (
        "pending",
        "human_annotation",
    ):
        # human_annotation 的 None = 標註者沒有過半數共識或 Unsure，是資料本身的性質（抽樣時會自動排除）
        problems.append(
            "label.correct 是 None 但 source 不是 pending / human_annotation"
        )
    for s in t.segments:
        if not s.text and s.call is None:
            problems.append(f"seg {s.seg_id} 是空的")
    # 標準答案不應出現在 Q（避免洩漏）。A 本來就可能等於答案，E 也可能合法地包含答案，所以只檢查 Q。
    g = t.hidden.get("gold_answer")
    if isinstance(g, str) and len(g) > 3:
        q_text = "\n".join(
            txt for s in t.segments for r, txt, _ in s.pieces() if r == Role.Q
        ).lower()
        if g.lower() in q_text:
            problems.append("gold 答案出現在 Q")
    return problems


def cmd_validate(args, _) -> None:
    bad = warn = total = 0
    for p in args.files:
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    t = _parse_trajectory(line)
                    problems = check_trajectory(t)
                    errors = [x for x in problems if not x.startswith("[警告]")]
                    if errors or (args.strict and problems):
                        bad += 1
                    elif problems:
                        warn += 1
                    if problems:
                        print(f"{p}:{ln} {t.traj_id}: " + "; ".join(problems))
                except Exception as e:  # noqa: BLE001 - 逐筆回報錯誤，不在第一筆就中斷
                    bad += 1
                    print(f"{p}:{ln}: schema 錯誤：{e}")
    print(f"[validate] {total} 筆：錯誤 {bad} 筆、警告 {warn} 筆")
    sys.exit(1 if bad else 0)


def cmd_stats(args, _) -> None:
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for t in read_jsonl(args.files):
        groups[f"{t.benchmark} | {t.solver.model}"].append(t)
    rows = []
    for k, ts in sorted(groups.items()):
        lab = Counter(t.label.correct for t in ts)
        tbr = [t.tokens_by_role() for t in ts]
        rows.append(
            {
                "group": k,
                "n": len(ts),
                "correct": lab.get(True, 0),
                "incorrect": lab.get(False, 0),
                "unlabeled": lab.get(None, 0),
                **{
                    f"med_tok_{r.value}": int(
                        statistics.median(x[r.value] for x in tbr)
                    )
                    for r in Role
                },
                "med_total_tok": int(statistics.median(sum(x.values()) for x in tbr)),
                "med_num_E": int(
                    statistics.median(
                        sum(s.has_evidence() for s in t.segments) for t in ts
                    )
                ),
                # R 是研究問題 (1) 的核心，要看「沒有原生推理」的比例，而不是只看中位數
                "no_native_R_rate": round(
                    sum(1 for t in ts if not t.meta.get("native_reasoning_tokens"))
                    / len(ts),
                    3,
                ),
                "incomplete": sum(1 for t in ts if t.meta.get("incomplete")),
                "call_order_changed": sum(
                    1 for t in ts if t.meta.get("call_order_changed", 0) > 0
                ),
                "ambiguous_seg_rate": round(
                    sum(s.ambiguous for t in ts for s in t.segments)
                    / max(1, sum(len(t.segments) for t in ts)),
                    3,
                ),
                "native_reasoning": dict(
                    Counter(t.solver.native_reasoning for t in ts)
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return
    cols = [
        "group",
        "n",
        "correct",
        "incorrect",
        "unlabeled",
        "med_tok_Q",
        "med_tok_E",
        "med_tok_R",
        "med_tok_A",
        "med_total_tok",
        "med_num_E",
        "no_native_R_rate",
        "incomplete",
        "call_order_changed",
        "ambiguous_seg_rate",
        "native_reasoning",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def cmd_sample(args, _) -> None:
    """抽正確 / 錯誤各半的平衡子集，並寫出 manifest 記錄每一步排除了多少、為什麼。

    步驟（順序很重要）：
    1. 篩選：有標籤、有 E、生成完整（incomplete 預設排除）、長度上限、指定的 split
    2. 去重：同一個 (task, solver) 只留一條，**在看標籤之前**隨機選
       （之前是邊平衡邊去重，某類滿額後同題的另一類 trial 還能進來，
        會偏向「有時成功、有時失敗」的題目，估計對象變得難以解釋）
    3. 平衡：正確、錯誤各取一樣多；有 --stratify-by 時在每一層內各自平衡，再輪流取到 per_class
    """
    rng = random.Random(args.seed)
    reasons: Counter = Counter()
    pool: list[Trajectory] = []
    n_input = 0
    splits_seen: Counter = Counter()
    for t in read_jsonl([args.file]):
        n_input += 1
        splits_seen[t.split] += 1
        if t.label.correct is None:
            reasons["no_label"] += 1
        elif not any(s.has_evidence() for s in t.segments):
            reasons["no_E"] += 1
        elif t.meta.get("incomplete") and not args.include_incomplete:
            reasons["incomplete"] += 1
        elif (
            args.max_total_tokens
            and sum(t.tokens_by_role().values()) > args.max_total_tokens
        ):
            reasons["too_long"] += 1
        elif args.split and t.split not in args.split:
            reasons["split_filtered"] += 1
        else:
            pool.append(t)

    # 步驟 2：先去重（與標籤無關）
    groups: dict[tuple, list[Trajectory]] = defaultdict(list)
    for t in pool:
        groups[(t.task_id, t.solver.model)].append(t)
    deduped = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda x: x.traj_id)
        deduped.append(rng.choice(g))
        reasons["duplicate_trial"] += len(g) - 1

    # 步驟 3：平衡（可分層）
    strata: dict[Any, dict[bool, list[Trajectory]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    for t in deduped:
        key = tuple(_get(t, f) for f in args.stratify_by) if args.stratify_by else "all"
        strata[key][t.label.correct].append(t)
    pairs_by_stratum = {}
    for key, d in strata.items():
        rng.shuffle(d[True])
        rng.shuffle(d[False])
        pairs_by_stratum[key] = list(zip(d[True], d[False]))
    chosen: list[Trajectory] = []
    keys = sorted(pairs_by_stratum, key=str)
    idx = 0
    while len(chosen) // 2 < args.per_class and any(pairs_by_stratum[k] for k in keys):
        k = keys[idx % len(keys)]
        if pairs_by_stratum[k]:
            chosen.extend(pairs_by_stratum[k].pop())
        idx += 1
    k_pairs = len(chosen) // 2

    out_p = Path(args.out)
    manifest_p = out_p.with_name(out_p.name.replace(".jsonl", "") + ".manifest.json")
    if k_pairs == 0:
        # 舊的平衡檔要刪掉，否則下游可能拿到過期的資料還以為是這次產生的
        for p in (out_p, manifest_p):
            if p.exists():
                p.unlink()
        print(
            f"[sample] {args.file}：沒有可用的正確 / 錯誤配對，略過（排除原因：{dict(reasons)}）。"
        )
        sys.exit(EXIT_NO_LABELS)
    if k_pairs < args.per_class:
        print(
            f"  注意：只湊得到 {k_pairs} 對（要求 {args.per_class}），兩類都取 {k_pairs}"
        )

    chosen.sort(key=lambda t: t.traj_id)
    n = write_jsonl(chosen, args.out)
    st = Path(args.file).stat()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "qera_code_version": code_version(),
        "input": {
            "path": str(args.file),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "n": n_input,
            "splits": {str(k): v for k, v in splits_seen.items()},
        },
        "params": {
            "per_class": args.per_class,
            "seed": args.seed,
            "split": args.split,
            "stratify_by": args.stratify_by,
            "max_total_tokens": args.max_total_tokens,
            "include_incomplete": args.include_incomplete,
        },
        "excluded": dict(reasons),
        "output": {
            "path": str(args.out),
            "n": n,
            "per_class": k_pairs,
            "by_stratum": {
                str(key): sum(
                    1
                    for t in chosen
                    if (
                        tuple(_get(t, f) for f in args.stratify_by)
                        if args.stratify_by
                        else "all"
                    )
                    == key
                )
                // 2
                for key in strata
            },
            "by_solver": dict(Counter(t.solver.model for t in chosen)),
            "by_split": {
                str(k): v for k, v in Counter(t.split for t in chosen).items()
            },
        },
        "note": "平衡樣本的正確率是 50%，不代表自然分布；用於計算 BA / EDR / FRR。",
    }
    manifest_p.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"[sample] 正確 {k_pairs}、錯誤 {k_pairs}，共 {n} 筆 → {args.out}（排除：{dict(reasons)}）"
    )


def cmd_show(args, _) -> None:
    ts = list(read_jsonl([args.file]))
    t = ts[args.index]
    sel = None
    if args.condition in ("QEsA", "QEJevA"):
        sel = [c["seg_id"] for c in e_candidates(t)][: args.first_k_e]
    vi = build_verifier_input(
        t, args.condition, selected_e=sel, max_chars_per_segment=args.max_chars
    )
    print(
        f"# {t.traj_id}  label={t.label.correct}  condition={args.condition}  "
        f"rendered_tokens={vi.n_tokens}  content_tokens={vi.n_tokens_content} {vi.tokens_by_role}\n"
    )
    print(vi.text)


def cmd_audit(args, _) -> None:
    """產生人工稽核用的 markdown。

    預設「盲化」：不顯示 trajectory 的對錯標籤。稽核者知道這條是錯的，
    可能會不自覺把可疑內容歸到 R，影響切分的客觀性。真的需要時用 --show-label。
    """
    rng = random.Random(args.seed)
    ts = list(read_jsonl([args.file]))
    pick = rng.sample(ts, min(args.n, len(ts)))
    max_chars = None if args.full else 300
    lines = [
        f"# E/R 切分稽核：{Path(args.file).name}\n",
        "請確認每個 segment 的角色。需要修改時，把結果寫成 JSONL（一行一筆），再用 `qera.cli apply-overrides` 寫回：",
        "",
        "```",
        '{"traj_id": "...", "seg_id": 12, "role": "E", "source": "manual"}                       # 整段改角色',
        '{"traj_id": "...", "seg_id": 12, "call_role": "R", "source": "manual"}                  # 只改呼叫參數的角色',
        '{"traj_id": "...", "seg_id": 12, "spans": [[0, 40, "Q"], [40, 95, "E"]], "source": "manual"}  # 一段切成多種角色',
        "```",
        "",
        f"（{'顯示' if args.show_label else '已隱藏'}對錯標籤；{'全文' if args.full else '每段只顯示前 300 字，需要全文請加 --full'}）\n",
    ]
    for t in pick:
        head = (
            f"\n## {t.traj_id}"
            + (f"（label={t.label.correct}）" if args.show_label else "")
            + "\n"
        )
        lines.append(head)
        lines.append("| seg | 角色 | 呼叫角色 | kind | 來源 | ⚠ | 內容 | 稽核 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in t.segments:
            body, _ = render_segment(s, None, max_chars)
            body = body.split("\n", 1)[-1].replace("|", "\\|").replace("\n", " ⏎ ")
            role = "+".join(sorted(r.value for r in s.text_roles()))
            call_role = s.call_role.value if s.call else ""
            lines.append(
                f"| {s.seg_id} | {role} | {call_role} | {s.kind} | {s.source_id or ''} | "
                f"{'⚠️' if s.ambiguous else ''} | {body} | |"
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] {len(pick)} 筆 → {args.out}")


def retokenize_segment(s) -> None:
    """重算一個 segment 的所有 token 數，計算方式和 builder 完全相同。"""
    s.n_tokens = tokens.count(s.text)
    s.call_n_tokens = tokens.count(s.call.render()) if s.call else None
    for sp in s.spans or []:
        sp.n_tokens = tokens.count(s.text[sp.start : sp.end])


def cmd_retokenize(args, _) -> None:
    def gen():
        for t in read_jsonl([args.file]):
            for s in t.segments:
                retokenize_segment(s)
            t.tokenizer = tokens.name()
            yield t

    n = write_jsonl(gen(), args.out)
    print(f"[retokenize] {n} 筆，tokenizer={tokens.name()} → {args.out}")


def cmd_apply_overrides(args, _) -> None:
    from .overrides import apply, load_overrides

    ov = load_overrides(args.overrides)
    stats: Counter = Counter()

    def checked():
        yield from apply(read_jsonl([args.file]), ov, stats)
        unmatched = len(ov) - stats["applied"] - stats["lower_priority"]
        if unmatched:
            # generator 尚未結束時正式輸出仍未替換；同檔輸入輸出也能安全失敗。
            print(
                f"  ❌ 有 {unmatched} 筆覆寫找不到對應的 segment（traj_id / seg_id 錯誤或資料檔不對）"
            )
            raise SystemExit(1)

    n = write_jsonl(checked(), args.out)
    print(
        f"[apply-overrides] 載入 {len(ov)} 筆覆寫，套用 {stats['applied']} 筆，寫出 {n} 筆 → {args.out}"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="qera")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("benchmark", choices=list(ADAPTERS))
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="+")
    v.add_argument("--strict", action="store_true", help="警告也當成錯誤")
    s = sub.add_parser("stats")
    s.add_argument("files", nargs="+")
    s.add_argument("--json", action="store_true")
    sm = sub.add_parser("sample")
    sm.add_argument("file")
    sm.add_argument("--per-class", type=int, default=250)
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--out", required=True)
    sm.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="排除超過這個長度的 trajectory（控制成本）",
    )
    sm.add_argument(
        "--split",
        nargs="+",
        default=None,
        help="只保留這些 split，例如 --split test。調 Jev 用的資料和報結果用的資料要分開",
    )
    sm.add_argument(
        "--stratify-by",
        nargs="+",
        default=None,
        help="分層平衡的欄位，例如 --stratify-by solver.model meta.tau2_domain",
    )
    sm.add_argument(
        "--include-incomplete",
        action="store_true",
        help="納入生成被截斷的 trajectory（預設排除）",
    )
    sh = sub.add_parser("show")
    sh.add_argument("file")
    sh.add_argument("--index", type=int, default=0)
    sh.add_argument("--condition", default="QERA", choices=list(CONDITIONS))
    sh.add_argument("--first-k-e", type=int, default=3)
    sh.add_argument("--max-chars", type=int, default=1500)
    au = sub.add_parser("audit")
    au.add_argument("file")
    au.add_argument("--n", type=int, default=10)
    au.add_argument("--seed", type=int, default=0)
    au.add_argument("--out", required=True)
    au.add_argument("--full", action="store_true", help="顯示全文（預設每段 300 字）")
    au.add_argument(
        "--show-label",
        action="store_true",
        help="顯示對錯標籤（預設隱藏，避免影響稽核判斷）",
    )
    ao = sub.add_parser("apply-overrides")
    ao.add_argument("file")
    ao.add_argument("overrides", nargs="+")
    ao.add_argument("--out", required=True)
    rt = sub.add_parser("retokenize")
    rt.add_argument("file")
    rt.add_argument("--out", required=True)

    args, extra = ap.parse_known_args(argv)
    fn = {
        "convert": cmd_convert,
        "validate": cmd_validate,
        "stats": cmd_stats,
        "sample": cmd_sample,
        "show": cmd_show,
        "audit": cmd_audit,
        "retokenize": cmd_retokenize,
        "apply-overrides": cmd_apply_overrides,
    }[args.cmd]
    if extra and args.cmd != "convert":
        ap.error(f"無法辨識的參數：{extra}")
    fn(args, extra)


if __name__ == "__main__":
    main()
