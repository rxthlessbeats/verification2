"""BrowseComp-Plus adapter。

輸入：
1. 解密後的資料 browsecomp_plus_decrypted.jsonl（query_id, query, answer, ...）
   產生方式：在 BrowseComp-Plus repo 執行
   python scripts_build_index/decrypt_dataset.py --output data/browsecomp_plus_decrypted.jsonl
2. run 檔（JSONL 或資料夾內的 run_*.json），格式為官方 search_agent 的輸出：
   {query_id, metadata:{model,...}, result:[{type: reasoning|tool_call|output_text, tool_name, arguments, output}]}
   官方釋出的 gpt5 / o3 trajectory：bash scripts_build_index/download_run_files.sh
3. （選用）judge 結果 JSONL（由 scripts/judge_bcp.py 產生）。BrowseComp-Plus 的對錯要用 LLM judge
   判（官方用 Qwen3-32B），沒有提供時 label.correct=None、source=pending。
   每筆判決都綁定 (run_tag, query_id, candidate_hash)：同一題在不同 run（gpt5@bm25、o3@bm25…）
   的答案不同，只用 query_id 對應會把 A 的判決貼到 B 的答案上。載入時會重新計算答案的 hash，
   對不上就不採用（標籤設 None），避免「格式正確但標籤錯配」這種很難察覺的錯誤。
4. （選用）topics-qrels/qrel_evidence.txt、qrel_golds.txt：標記哪些檢索結果是真正的 evidence。

Q/E/R/A 規則（SPEC.md §4.2）：
- Q : query
- E : 每次 search / get_document 的呼叫 + 回傳
- R : reasoning（OpenAI 的 reasoning summary）、非最後一次的文字輸出
- A : 最後一次 output_text

注意：BrowseComp 系列要求資料「不得以明文出現在網路上」。轉換後的 JSONL 含明文問題與
答案，不要推到公開 GitHub。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from .. import jsonl
from ..builder import TrajectoryBuilder, to_text
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

BENCHMARK = "browsecomp_plus"
DOMAIN = "multihop_qa"


def _iter_runs(path: str) -> Iterator[dict]:
    p = Path(path)
    files = sorted(p.rglob("*.json*")) if p.is_dir() else [p]
    for f in files:
        if f.suffix == ".jsonl":
            yield from jsonl.read(f)
        else:
            yield json.loads(f.read_text(encoding="utf-8"))


def _load_qrels(path: str | None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    if path and Path(path).exists():
        for line in jsonl.lines(path):
            parts = line.split()
            if len(parts) >= 4 and parts[3] != "0":
                out[parts[0]].add(parts[2])
    return out


def _docids(output) -> list[str]:
    parsed = output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except ValueError:  # 不是 JSON
            return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [
        str(x["docid"]) for x in parsed or [] if isinstance(x, dict) and "docid" in x
    ]


def _reasoning_text(output) -> str:
    if isinstance(output, list):  # summary: [{"type":"summary_text","text":...}]
        return "\n".join(
            x.get("text", "") if isinstance(x, dict) else str(x) for x in output
        )
    return to_text(output)


def final_output_text(run: dict) -> str:
    """這條 run 最後交出的答案文字（= trajectory 的 A）。judge 和 adapter 共用，確保判的是同一段文字。"""
    outs = [
        it.get("output")
        for it in run.get("result", [])
        if it.get("type") == "output_text"
    ]
    return to_text(outs[-1]) if outs else ""


def candidate_hash(text: str) -> str:
    """答案文字的指紋（去掉頭尾空白後的 SHA-256 前 16 碼）。"""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def run_tag_for(runs_path: str, run: dict | None = None) -> str:
    """run 的識別名稱：有 metadata.model 用它，否則用「檔名@上層資料夾」，例如 bm25/o3.jsonl → o3@bm25。"""
    if run and (run.get("metadata") or {}).get("model"):
        return str(run["metadata"]["model"])
    rp = Path(runs_path)
    return f"{rp.stem}@{rp.parent.name}" if rp.is_file() or rp.suffix else rp.name


def load_judgments(path: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    legacy = 0
    for j in jsonl.read(path):
        if "run_tag" not in j or "candidate_hash" not in j:
            legacy += 1  # 舊格式（只有 query_id），無法確認是哪條答案的判決，不採用
            continue
        out[(str(j["run_tag"]), str(j["query_id"]))] = (
            j  # 同一鍵出現多次（重判）時以最後一筆為準
        )
    if legacy:
        print(
            f"[browsecomp_plus] 警告：{legacy} 筆判決缺少 run_tag / candidate_hash，不採用；請用新版 judge_bcp.py 重跑"
        )
    return out


def load_queries(path: str) -> dict[str, dict]:
    """只保留 query_id / query / answer。

    解密後的資料檔約 2 GB（每題都附上 evidence、gold、negative 文件全文），全部載入很吃記憶體，
    所以逐行讀、只留需要的欄位。
    """
    out = {}
    for q in jsonl.read(path):
        out[str(q["query_id"])] = {
            "query_id": q["query_id"],
            "query": q["query"],
            "answer": q.get("answer"),
        }
    return out


def _label_from_judgment(j: dict | None, answer_text: str) -> tuple[Label, str | None]:
    """回傳 (標籤, 問題原因)。"""
    if j is None:
        return Label(correct=None, source="pending"), None
    detail = {
        "judge_model": j.get("judge_model"),
        "judge_status": j.get("status"),
        "judge_params": j.get("judge_params"),
    }
    if j.get("candidate_hash") != candidate_hash(answer_text):
        return Label(
            correct=None, source="pending", detail=detail | {"mismatch": True}
        ), "hash_mismatch"
    if j.get("status") != "ok" or not isinstance(j.get("correct"), bool):
        return Label(
            correct=None, source="pending", detail=detail
        ), f"status={j.get('status')}"
    return Label(correct=j["correct"], source="llm_judge", detail=detail), None


def convert_one(
    q: dict,
    run: dict,
    judgment: dict | None,
    ev: set[str],
    gold: set[str],
    native: str,
    model_override: str | None = None,
) -> Trajectory:
    b = TrajectoryBuilder()
    b.add(Role.Q, "user", "question", q["query"], source_id="query")
    items = run.get("result", [])
    last_out = max(
        (i for i, it in enumerate(items) if it.get("type") == "output_text"), default=-1
    )
    for i, it in enumerate(items):
        t = it.get("type")
        if t == "reasoning":
            b.add(
                Role.R,
                "solver",
                "reasoning",
                _reasoning_text(it.get("output")),
                step=i,
                source_id=f"result:{i}/reasoning",
            )
        elif t == "tool_call":
            args = it.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:  # 不是 JSON 就保留原字串
                    pass
            ids = _docids(it.get("output"))
            b.add(
                Role.E,
                "environment",
                "search_result",
                to_text(it.get("output")),
                step=i,
                source_id=f"result:{i}/tool_call",
                call=ToolCall(
                    name=it.get("tool_name") or "search",
                    arguments=args,
                    requestor="assistant",
                ),
                hidden={
                    "docids": ids,
                    "evidence_hits": sorted(set(ids) & ev),
                    "gold_hits": sorted(set(ids) & gold),
                },
            )
        elif t == "output_text":
            if i == last_out:
                b.add(
                    Role.A,
                    "solver",
                    "final_answer",
                    to_text(it.get("output")),
                    step=i,
                    source_id=f"result:{i}/output_text",
                )
            else:
                b.add(
                    Role.R,
                    "solver",
                    "intermediate_output",
                    to_text(it.get("output")),
                    step=i,
                    source_id=f"result:{i}/output_text",
                )
    if last_out < 0:
        b.add(Role.A, "solver", "final_answer", "(no answer)", skip_empty=False)

    # 官方釋出的 run 檔沒有 metadata，模型名稱要從檔案路徑（例如 bm25/gpt5.jsonl）帶進來
    model = model_override or (run.get("metadata") or {}).get("model") or "unknown"
    qid = str(q["query_id"])
    label, problem = _label_from_judgment(judgment, final_output_text(run))
    label_problem = problem
    return b.build(
        traj_id=f"bcp.{qid}.{model.replace('/', '_')}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=qid,
        split="test",
        solver=SolverInfo(
            model=model,
            agent="browsecomp_plus_search_agent",
            origin="official_release",
            native_reasoning=native,
        ),
        label=label,
        meta={
            "num_search_calls": sum(1 for it in items if it.get("type") == "tool_call"),
            "status": run.get("status"),
            "candidate_hash": candidate_hash(final_output_text(run)),
        },
        hidden={
            "gold_answer": q.get("answer"),
            "evidence_docids": sorted(ev),
            "gold_docids": sorted(gold),
            **({"label_problem": label_problem} if label_problem else {}),
        },
    )


def add_args(p) -> None:
    p.add_argument("--queries", required=True, help="browsecomp_plus_decrypted.jsonl")
    p.add_argument("--runs", required=True, help="run JSONL 或 run_*.json 資料夾")
    p.add_argument(
        "--judgments", default=None, help="scripts/judge_bcp.py 產生的判決 JSONL"
    )
    p.add_argument("--qrels-evidence", default=None)
    p.add_argument("--qrels-gold", default=None)
    p.add_argument(
        "--native-reasoning",
        default="summary",
        choices=["full", "summary", "hidden", "none"],
    )
    p.add_argument(
        "--model",
        default=None,
        help="solver 名稱；官方 run 檔沒記錄，預設用「檔名@上層資料夾」，例如 gpt5@bm25",
    )


def run(args) -> Iterator[Trajectory]:
    qs = load_queries(args.queries)
    judged = load_judgments(args.judgments) if args.judgments else {}
    ev, gold = _load_qrels(args.qrels_evidence), _load_qrels(args.qrels_gold)
    override = getattr(args, "model", None)
    problems: dict[str, int] = defaultdict(int)
    for r in _iter_runs(args.runs):
        qid = str(r.get("query_id"))
        if qid not in qs:
            continue
        tag = override or run_tag_for(args.runs, r)
        t = convert_one(
            qs[qid],
            r,
            judged.get((tag, qid)),
            ev.get(qid, set()),
            gold.get(qid, set()),
            args.native_reasoning,
            tag,
        )
        if t.hidden.get("label_problem"):
            problems[t.hidden["label_problem"].split("=")[0]] += 1
        yield t
    if problems:
        print(
            f"[browsecomp_plus] 有標籤問題、未採用的判決：{dict(problems)}（hash_mismatch = 判決對應的答案和這條 run 不同）"
        )
