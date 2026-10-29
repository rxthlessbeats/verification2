"""BrowseComp-Plus 的 LLM judge：判斷每條 run 的最終答案是否正確，產生 adapter 需要的標籤檔。

────────────────────────────────────────────────────────────────────────
用法（CARC 上，先用 vLLM 開 judge 模型）
────────────────────────────────────────────────────────────────────────

    vllm serve Qwen/Qwen3-32B --port 8000
    python scripts/judge_bcp.py \
        --queries raw/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
        --runs raw/BrowseComp-Plus/data/decrypted_run_files/bm25/gpt5.jsonl \
        --out judgments/bcp_bm25_gpt5.jsonl \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B

輸出檔名請用 judgments/bcp_<retriever>_<模型>.jsonl，convert_all.sh 會自動找到。

────────────────────────────────────────────────────────────────────────
設定：完全照官方 evaluate_run.py
────────────────────────────────────────────────────────────────────────

官方 judge（scripts_evaluation/evaluate_run.py）的設定是：
  模型 Qwen3-32B、關閉 thinking（enable_thinking=False）、temperature 0.7、top_p 0.8、
  top_k 20、最多 4096 tokens。這裡的預設值完全相同，評分 prompt 也逐字沿用。
  為什麼照官方：BrowseComp-Plus 公布的正確率是用這組設定算的，我們的標籤要能和官方數字對照。
  judge 用哪個模型、要不要改設定，請和負責架模型的人（Magnus）確認後再改，並在論文中註明。

注意：judge 模型不要和 verifier 是同一個，否則 judge 的偏誤會和 verifier 的偏誤相關。

────────────────────────────────────────────────────────────────────────
輸出格式（每行一筆）
────────────────────────────────────────────────────────────────────────

    {"run_tag": "gpt5@bm25", "query_id": "769", "candidate_hash": "3f2a...",
     "status": "ok", "correct": true, "judge_model": "Qwen/Qwen3-32B",
     "judge_params": {...}, "finish_reason": "stop", "raw": "<judge 的完整回覆>"}

- run_tag + query_id + candidate_hash 一起綁定「這個判決是給哪條答案的」。
  同一題在不同 run 的答案不同，只用 query_id 會貼錯。adapter 載入時會重算 hash 比對。
- status：ok / parse_error（找不到唯一的判決欄位）/ truncated（judge 輸出被截斷）。
  只有 ok 會被當成標籤；其他狀態重跑同一行指令時會自動重判。
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# 讓這支腳本不論從哪個資料夾執行，都能 import 專案裡的 qera
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 沒裝套件也能直接跑

from qera import jsonl
from qera.adapters.browsecomp_plus import (
    candidate_hash,
    final_output_text,
    load_queries,
    run_tag_for,
)

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

# 判決欄位必須是「獨立的一行」，以 correct: 開頭（前面可有 markdown 粗體或清單符號）。
# 這樣 "incorrect: yes" 不會被誤判，reasoning 段落中間引用的 "correct: yes" 也不會被抓到。
VERDICT_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?\**\s*correct\s*\**\s*:\s*\**\s*(yes|no)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_verdict(text: str) -> tuple[str, bool | None]:
    """回傳 (status, correct)。

    要求「恰好一個」判決行；找不到或出現多個互相矛盾的判決，都視為 parse_error，
    寧可不給標籤，也不要給錯的標籤（標籤直接決定 EDR / FRR / BA）。
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    found = {m.group(1).lower() for m in VERDICT_LINE.finditer(text)}
    if len(found) != 1:
        return "parse_error", None
    return "ok", found.pop() == "yes"


def repair_output(out: Path) -> None:
    """續寫前恢復完整 JSON 行，避免半行或缺換行的尾端和新判決黏在一起。"""
    if not out.exists():
        return
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out.parent,
            prefix=out.name + ".",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp = Path(f.name)
            for i, line in enumerate(jsonl.lines(out), 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  警告：{out} 第 {i} 行不完整，移除後重判", file=sys.stderr)
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 讀取與寫入都成功才提交，修復失敗時保留原檔供追查。
        os.replace(tmp, out)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def load_done(out: Path) -> dict[tuple[str, str], dict]:
    """已完成的判決。只有 status=ok 且答案 hash 相同的才算完成，其他會重判。

    檔案最後一行如果不完整（例如程式被中途砍掉），會被略過並警告，不會讓整個續跑失敗。
    """
    done: dict[tuple[str, str], dict] = {}
    if not out.exists():
        return done
    lines = list(jsonl.lines(out))
    for i, line in enumerate(lines):
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            print(
                f"  警告：{out} 第 {i + 1} 行不是完整的 JSON（可能是上次中斷），略過",
                file=sys.stderr,
            )
            continue
        if j.get("status") == "ok":
            done[(j["run_tag"], str(j["query_id"]))] = j
    return done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="BrowseComp-Plus LLM judge（官方設定）")
    ap.add_argument("--queries", required=True, help="browsecomp_plus_decrypted.jsonl")
    ap.add_argument(
        "--runs",
        required=True,
        help="一個 run 檔，例如 decrypted_run_files/bm25/gpt5.jsonl",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--run-tag",
        default=None,
        help="預設「檔名@上層資料夾」，例如 gpt5@bm25；要和轉換時一致",
    )
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--api-key", default="EMPTY")
    # 以下預設值 = 官方 evaluate_run.py
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="官方關閉 thinking；除非有理由，不要開",
    )
    ap.add_argument("--max-retries", type=int, default=5)
    a = ap.parse_args(argv)

    from openai import OpenAI

    client = OpenAI(base_url=a.base_url, api_key=a.api_key)
    params = {
        "temperature": a.temperature,
        "top_p": a.top_p,
        "top_k": a.top_k,
        "max_tokens": a.max_tokens,
        "enable_thinking": a.enable_thinking,
    }
    gt = load_queries(a.queries)  # 只留需要的欄位，原檔約 2 GB
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repair_output(out)
    done = load_done(out)
    stats = {"ok": 0, "skipped": 0, "parse_error": 0, "truncated": 0, "failed": 0}

    with open(out, "a", encoding="utf-8") as f:
        for r in jsonl.read(a.runs):
            qid = str(r["query_id"])
            if qid not in gt:
                continue
            tag = a.run_tag or run_tag_for(a.runs, r)
            answer = final_output_text(r)
            h = candidate_hash(answer)
            prev = done.get((tag, qid))
            if prev and prev.get("candidate_hash") == h:
                stats["skipped"] += 1
                continue
            prompt = GRADER_TEMPLATE.format(
                question=gt[qid]["query"],
                response=answer,
                correct_answer=gt[qid]["answer"],
            )
            resp = None
            for attempt in range(a.max_retries):
                try:
                    resp = client.chat.completions.create(
                        model=a.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=a.temperature,
                        top_p=a.top_p,
                        max_tokens=a.max_tokens,
                        extra_body={
                            "top_k": a.top_k,
                            "chat_template_kwargs": {
                                "enable_thinking": a.enable_thinking
                            },
                        },
                    )
                    break
                except Exception as e:  # noqa: BLE001 - 網路或伺服器的任何錯誤都要重試
                    wait = min(60, 2**attempt)
                    print(
                        f"  [重試 {attempt + 1}/{a.max_retries}] {qid}: {e}（{wait} 秒後）",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
            if resp is None:
                stats["failed"] += 1
                continue
            choice = resp.choices[0]
            text = choice.message.content or ""
            if choice.finish_reason == "length":
                status, correct = "truncated", None
            else:
                status, correct = parse_verdict(text)
            stats[status] += 1
            f.write(
                json.dumps(
                    {
                        "run_tag": tag,
                        "query_id": qid,
                        "candidate_hash": h,
                        "status": status,
                        "correct": correct,
                        "judge_model": a.model,
                        "judge_params": params,
                        "finish_reason": choice.finish_reason,
                        "raw": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

    print(f"[judge] {stats}")
    if stats["parse_error"] or stats["truncated"] or stats["failed"]:
        print(
            "  有判決未完成；重跑同一行指令會自動重判這些題目。若一直 parse_error，請抽幾筆 raw 看 judge 的輸出格式。"
        )


if __name__ == "__main__":
    main()
