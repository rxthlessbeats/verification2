"""MuSiQue-Ans adapter。

需要兩個輸入：
1. 官方資料 musique_ans_v1.0_dev.jsonl（或 train）。欄位：id, paragraphs[{idx,title,paragraph_text,is_supporting}],
   question, question_decomposition, answer, answer_aliases, answerable
2. solver 輸出（JSONL，每行一題），兩種格式擇一：
   (a) {"id": ..., "model": ..., "messages": [OpenAI chat messages，含 assistant 回覆]}
   (b) {"id": ..., "model": ..., "reasoning": "...", "answer": "..."}

另外提供 `make_solver_prompt()`，讓 Magnus 跑千問時用同一個 prompt，保證 E 就是
「給 solver 看過的那 20 段」。

Q/E/R/A 規則（SPEC.md §4.1）：
- E : 20 個 paragraph，各一個 segment（is_supporting 只放 hidden，不給 verifier / Jev，
      但可以事後用來評估 Jev 有沒有選到真正的支持段落）
- Q : question
  （順序和 solver 看到的 prompt 一致：段落在前、問題在後。proposal 要求保持原始順序。）
- R : 推理欄位 / <think> / "Answer:" 之前的文字（規則統一在 reasoning.py）
- A : "Answer:" 之後的文字（找不到就整段）
- 生成被截斷（finish_reason=length 或 <think> 沒收尾）：不產生 A 的答案文字，
  trajectory 標 meta.incomplete，抽樣時預設排除。「沒交答案」和「答錯」是兩回事。
- 標籤：EM（對 answer + aliases 取最大值）。F1 記在 detail。
  EM 只比對字串，同義但寫法不同的答案會被判錯，所以 EM=0 但「F1>0 或答案與標準答案互相包含」
  的標成 needs_review，建議人工抽查（MuSiQue 的標籤品質問題要在論文中說明）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from .. import jsonl, reasoning
from ..builder import TrajectoryBuilder
from ..chat import has_answer_line, split_answer
from ..grading import best_over_aliases, normalize_answer
from ..schema import Label, Role, SolverInfo, Trajectory

BENCHMARK = "musique_ans"
DOMAIN = "multihop_qa"

SOLVER_PROMPT = """Answer the question using the paragraphs below. Think step by step, then give a short final answer \
(an entity, date, number, or short phrase) on the last line in the form "Answer: <answer>".

{paragraphs}

Question: {question}"""


def hop_count(ex: dict) -> int:
    m = re.match(r"(\d)hop", ex.get("id", ""))
    return int(m.group(1)) if m else len(ex.get("question_decomposition", []))


def format_paragraph(p: dict) -> str:
    return f"Title: {p['title']}\n{p['paragraph_text']}"


def make_solver_prompt(ex: dict) -> str:
    paras = "\n\n".join(
        f"Paragraph {i + 1}: {format_paragraph(p)}"
        for i, p in enumerate(ex["paragraphs"])
    )
    return SOLVER_PROMPT.format(paragraphs=paras, question=ex["question"])


def _solver_parts(run: dict) -> tuple[str, str, dict]:
    """回傳 (推理, 答案, 生成狀態)。"""
    meta = run.get("meta") or {}
    status = {
        "finish_reason": meta.get("finish_reason"),
        "has_answer_line": None,
        "unclosed_think": False,
    }
    if "messages" in run:
        last = next(
            (m for m in reversed(run["messages"]) if m.get("role") == "assistant"), {}
        )
        think, content, unclosed = reasoning.split_think(last.get("content") or "")
        status["unclosed_think"] = unclosed
        status["has_answer_line"] = has_answer_line(content)
        pre, ans = split_answer(content) if not unclosed else (content, "")
        r = "\n\n".join(x for x in (reasoning.from_message(last), think, pre) if x)
        return r, ans, status
    return run.get("reasoning", ""), run.get("answer", ""), status


def _needs_review(answer: str, golds: list[str], em: float, f1: float) -> bool:
    if em:
        return False
    a = normalize_answer(answer)
    return f1 > 0 or any(
        a and g and (a in g or g in a) for g in map(normalize_answer, golds)
    )


def convert_one(ex: dict, run: dict, split: str, native: str) -> Trajectory:
    b = TrajectoryBuilder()
    for p in ex["paragraphs"]:
        b.add(
            Role.E,
            "environment",
            "passage",
            format_paragraph(p),
            source_id=f"prompt/paragraph:{p.get('idx')}",
            hidden={
                "is_supporting": bool(p.get("is_supporting")),
                "paragraph_idx": p.get("idx"),
            },
        )
    b.add(Role.Q, "user", "question", ex["question"], source_id="prompt/question")
    r_text, answer, status = _solver_parts(run)
    b.add(Role.R, "solver", "reasoning", r_text, source_id="response/reasoning")

    truncated = status["finish_reason"] == "length" or status["unclosed_think"]
    if truncated:
        b.mark_incomplete(
            "truncated",
            f"finish_reason={status['finish_reason']}, unclosed_think={status['unclosed_think']}",
        )
        answer = ""
    b.add(
        Role.A,
        "solver",
        "final_answer",
        answer or "(no answer)",
        skip_empty=False,
        source_id="response/answer",
    )

    golds = [ex["answer"], *ex.get("answer_aliases", [])]
    em, f1 = best_over_aliases(answer, golds)
    model = run.get("model", "unknown")
    return b.build(
        traj_id=f"musique.{ex['id']}.{model.replace('/', '_')}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=ex["id"],
        split=split,
        solver=SolverInfo(
            model=model, origin="self_generated", native_reasoning=native
        ),
        label=Label(
            correct=bool(em),
            source="exact_match",
            score=em,
            detail={
                "f1": round(f1, 4),
                "needs_review": _needs_review(answer, golds, em, f1),
                "format_ok": status["has_answer_line"],
            },
        ),
        meta={
            "hop_count": hop_count(ex),
            "num_paragraphs": len(ex["paragraphs"]),
            "generation": {k: v for k, v in status.items() if v is not None},
            "solver_gen_params": (run.get("meta") or {}).get("gen_params"),
        },
        hidden={
            "gold_answer": ex["answer"],
            "answer_aliases": ex.get("answer_aliases", []),
            "question_decomposition": ex.get("question_decomposition"),
        },
    )


def add_args(p) -> None:
    p.add_argument("--data", required=True, help="musique_ans_v1.0_dev.jsonl")
    p.add_argument("--runs", default=None, help="solver 輸出 JSONL（見模組說明）")
    p.add_argument("--split", default="dev")
    p.add_argument(
        "--native-reasoning",
        default="full",
        choices=["full", "summary", "hidden", "none"],
    )
    p.add_argument(
        "--export-prompts",
        default=None,
        help="不轉換，而是輸出給 solver 用的 prompt JSONL（{id, prompt}），交給跑模型的人",
    )


def run(args) -> Iterator[Trajectory]:
    exs = list(jsonl.read(args.data))
    if args.export_prompts:
        with open(args.export_prompts, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(
                    {"id": ex["id"], "prompt": make_solver_prompt(ex)},
                    ensure_ascii=False,
                )
                + "\n"
                for ex in exs
            )
        print(f"[musique] 已輸出 {len(exs)} 個 prompt → {args.export_prompts}")
        return
    if not args.runs:
        raise SystemExit("需要 --runs（或用 --export-prompts 先產生 prompt）")
    by_id = {e["id"]: e for e in exs}
    latest: dict[str, dict] = {}
    for r in jsonl.read(args.runs):  # 同一題出現多次（重跑）時，以最後一筆為準
        latest[r["id"]] = r
    for rid, r in latest.items():
        if rid not in by_id:
            print(f"[musique] 警告：{rid} 不在資料中，略過")
            continue
        yield convert_one(by_id[rid], r, args.split, args.native_reasoning)
