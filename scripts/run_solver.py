"""用千問（或任何 OpenAI 相容端點上的模型）當 solver，批次產生 trajectory。

────────────────────────────────────────────────────────────────────────
這支程式在整個流程中的位置
────────────────────────────────────────────────────────────────────────

    musique adapter --export-prompts        ← 產生統一的 solver prompt
            │
            ▼
    runs/musique_dev_prompts.jsonl          ← 輸入：{"id", "prompt"}
            │
            ▼
    scripts/run_solver.py（本程式）           ← 呼叫 vLLM 上的千問
            │
            ▼
    runs/musique.<模型名>.jsonl              ← 輸出：OpenAI chat 格式（見下方）
            │
            ▼
    scripts/convert_all.sh                  ← 自動找到 runs/musique.*.jsonl 並轉成 QERA 格式

────────────────────────────────────────────────────────────────────────
輸入格式（每行一題）
────────────────────────────────────────────────────────────────────────

    {"id": "2hop__13548_13529", "prompt": "Answer the question using ..."}

────────────────────────────────────────────────────────────────────────
輸出格式（每行一題，這是 qera/chat.py 和 musique adapter 認得的格式）
────────────────────────────────────────────────────────────────────────

    {
      "id": "2hop__13548_13529",
      "model": "Qwen/Qwen3-32B",
      "messages": [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant",
         "reasoning_content": "<千問的思考過程>",     ← 這就是研究要的 R，一定要保留
         "content": "...\nAnswer: <答案>"}
      ],
      "meta": {...}                                  ← 生成參數、token 用量、是否被截斷（分析用）
    }

────────────────────────────────────────────────────────────────────────
怎麼用（在 CARC 上）
────────────────────────────────────────────────────────────────────────

1. 開 vLLM（另一個終端機或 Slurm job，見 scripts/carc_solver.slurm）：

       vllm serve Qwen/Qwen3-32B --reasoning-parser qwen3 --max-model-len 32768 --port 8000

   ⚠ 一定要加 --reasoning-parser，vLLM 才會把思考內容放到獨立欄位。
     沒加也不會壞（思考會以 <think>...</think> 留在 content 裡，adapter 也認得），
     但分開放比較乾淨，也比較不容易出錯。

2. 先跑 5 題測試，確認格式正確：

       python scripts/run_solver.py --prompts runs/musique_dev_prompts.jsonl \
           --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B --limit 5

3. 正式跑（可以中斷，重跑同一行指令會從斷掉的地方接著跑）：

       python scripts/run_solver.py --prompts runs/musique_dev_prompts.jsonl \
           --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B \
           --sample 1000 --seed 0 --workers 16

────────────────────────────────────────────────────────────────────────
為什麼這樣設計
────────────────────────────────────────────────────────────────────────

- 輸出用 OpenAI chat 格式：vLLM 本來就回傳這種格式，不需要額外轉換；
  而且 qera/chat.py 已經會解析它，以後其他 benchmark 自己跑 solver 也共用同一套。
- 可以中斷續跑：MuSiQue 有 2,417 題，CARC 的 job 有時間上限，斷掉不用從頭來。
- 每完成一題就立刻寫入並 flush：job 被砍掉時，已完成的部分不會遺失。
- 平行送出請求（--workers）：vLLM 會自動把同時進來的請求合併成 batch，
  一次只送一題會讓 GPU 大部分時間閒著。
- 記錄 finish_reason：思考太長被 max_tokens 截斷的題目沒有答案，
  adapter 會把它判成「錯」，但那其實是設定問題、不是模型答錯，所以要另外標出來。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 讓這支腳本不論從哪個資料夾執行，都能 import 專案裡的 qera
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 沒裝套件也能直接跑

from qera import jsonl, reasoning

# ════════════════════════════════════════════════════════════════════
# 1. 呼叫模型
# ════════════════════════════════════════════════════════════════════


def extract_reasoning(message) -> str:
    """從 vLLM 回傳的 message 取出思考內容（規則統一在 qera/reasoning.py，所有 adapter 共用）。

    不同版本的 vLLM 欄位名稱不同：舊版叫 reasoning_content，新版改叫 reasoning；兩個都會試。
    """
    return reasoning.from_object(message)


def call_model(
    client, model: str, prompt: str, gen: dict, thinking: bool, max_retries: int = 5
) -> dict:
    """送出一題，回傳整理好的 assistant 訊息與中繼資料。失敗會自動重試（指數退避）。"""
    extra_body = {
        # Qwen3 系列的思考開關要透過 chat template 參數傳入；對不支援的模型無害
        "chat_template_kwargs": {"enable_thinking": thinking},
        "top_k": gen["top_k"],
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                max_tokens=gen["max_tokens"],
                seed=gen["seed"],
                extra_body=extra_body,
            )
            latency = time.time() - t0
            choice = resp.choices[0]
            msg = choice.message
            assistant = {"role": "assistant", "content": msg.content or ""}
            reasoning = extract_reasoning(msg)
            if reasoning:
                assistant["reasoning_content"] = reasoning
            usage = getattr(resp, "usage", None)
            return {
                "assistant": assistant,
                "finish_reason": choice.finish_reason,
                "latency_s": round(latency, 3),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
                if usage
                else None,
            }
        except Exception as e:  # noqa: BLE001 - 網路中斷、vLLM 暫時滿載等都要重試
            last_err = e
            wait = min(60, 2**attempt)
            print(
                f"  [重試 {attempt + 1}/{max_retries}] {type(e).__name__}: {e}（{wait} 秒後重試）",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"重試 {max_retries} 次仍失敗：{last_err}")


# ════════════════════════════════════════════════════════════════════
# 2. 檢查輸出
# ════════════════════════════════════════════════════════════════════

ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:final answer|answer)(?:\*\*)?\s*[:：]", re.IGNORECASE
)


def check_output(result: dict) -> list[str]:
    """回傳這題的警告。警告不會讓程式停下來，只會記錄在 meta 並在最後統計。"""
    warnings = []
    a = result["assistant"]
    if result["finish_reason"] == "length":
        # 思考還沒結束就被截斷：沒有答案，adapter 會判成錯，但其實是 max_tokens 設太小
        warnings.append("truncated")
    if not ANSWER_RE.search(a.get("content", "")):
        # 沒有照 prompt 要求寫 "Answer:"；adapter 會把整段 content 當答案，EM 幾乎一定是 0
        warnings.append("no_answer_line")
    if not a.get("reasoning_content") and "<think>" not in a.get("content", ""):
        # 沒拿到思考內容：R 會是空的。最常見原因是 vLLM 沒加 --reasoning-parser，或 thinking 被關掉
        warnings.append("no_reasoning")
    return warnings


# ════════════════════════════════════════════════════════════════════
# 3. 主流程
# ════════════════════════════════════════════════════════════════════


def model_tag(model: str) -> str:
    """把模型名稱轉成可以放進檔名的字串：Qwen/Qwen3-32B → Qwen3-32B"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def load_prompts(
    path: str, sample: int | None, seed: int, limit: int | None, ids_file: str | None
) -> list[dict]:
    items = list(jsonl.read(path))
    if ids_file:  # 只跑指定的題目（例如重跑出問題的題）
        keep = set(Path(ids_file).read_text().split())
        items = [x for x in items if x["id"] in keep]
    if sample and sample < len(items):
        # 固定 seed 的隨機抽樣，確保組員重跑時抽到同一批題目
        items = random.Random(seed).sample(items, sample)
    if limit:
        items = items[:limit]
    return items


def config_hash(model: str, gen: dict, thinking: bool, base_prompt_file: str) -> str:
    """這次實驗設定的指紋：模型、生成參數（max_tokens 除外）、thinking、prompt 檔名。

    為什麼需要：之前續跑只看題號。如果換了模型或參數卻沿用同一個輸出檔，
    已經跑過的題目會被直接跳過，結果檔裡混了兩種設定的輸出，而且看不出來。
    """
    # max_tokens 不列入：它只決定會不會被截斷，「調大上限、重跑被截斷的題」是正常流程。
    # 每一筆實際用的 max_tokens 仍記在 meta.gen_params。
    gen = {k: v for k, v in gen.items() if k != "max_tokens"}
    payload = json.dumps(
        {
            "model": model,
            "gen": gen,
            "thinking": thinking,
            "prompts": os.path.basename(base_prompt_file),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def scan_output(out_path: Path, cfg: str) -> tuple[dict[str, dict], int]:
    """讀取既有輸出，回傳 ({題號: 最後一筆}, 壞掉的行數)。

    - 設定指紋不同 → 直接停止：不同實驗不能混寫在同一個檔案
    - 最後一行不完整（程式被中途砍掉）→ 略過並警告，該題會重跑
    """
    rows: dict[str, dict] = {}
    bad = 0
    if not out_path.exists():
        return rows, bad
    for i, line in enumerate(jsonl.lines(out_path)):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            print(
                f"  警告：{out_path} 第 {i + 1} 行不是完整的 JSON（可能是上次中斷），該題會重跑",
                file=sys.stderr,
            )
            continue
        other = (row.get("meta") or {}).get("config_hash")
        if other != cfg:
            raise SystemExit(
                f"❌ {out_path} 裡有不同設定（config_hash={other}，這次是 {cfg}）的結果。\n"
                "   換了模型或參數時請用新的 --out 檔名，不要混在同一個檔案。"
            )
        rows[row["id"]] = row  # 同題多筆時以最後一筆為準
    return rows, bad


def is_complete(row: dict) -> bool:
    w = set((row.get("meta") or {}).get("warnings") or [])
    return not ({"truncated", "no_answer_line"} & w)


def rewrite_without(out_path: Path, drop_ids: set[str]) -> None:
    """把要重跑的題目從輸出檔移除（先寫暫存檔再替換，過程中斷也不會毀掉原檔）。"""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in jsonl.lines(out_path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["id"] not in drop_ids:
                f.write(line if line.endswith("\n") else line + "\n")
    os.replace(tmp, out_path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="用 OpenAI 相容端點上的模型批次跑 solver")
    ap.add_argument("--prompts", required=True, help="輸入 JSONL：每行 {id, prompt}")
    ap.add_argument(
        "--out", default=None, help="輸出 JSONL；預設 runs/musique.<模型名>.jsonl"
    )
    ap.add_argument(
        "--base-url", default="http://localhost:8000/v1", help="vLLM 的位址"
    )
    ap.add_argument(
        "--model", required=True, help="vLLM serve 時用的模型名稱，例如 Qwen/Qwen3-32B"
    )
    ap.add_argument("--api-key", default="EMPTY", help="vLLM 預設不檢查，填什麼都可以")

    # 抽樣
    ap.add_argument(
        "--sample", type=int, default=None, help="隨機抽幾題（不給就全部跑）"
    )
    ap.add_argument("--seed", type=int, default=0, help="抽樣與生成的隨機種子")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 題（測試用）")
    ap.add_argument("--ids", default=None, help="只跑檔案中列出的 id（一行一個）")

    # 生成參數：預設值是 Qwen3 官方建議的 thinking mode 設定。
    # 不建議用 temperature=0：Qwen 官方指出 thinking mode 下 greedy decoding 容易無限重複。
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="思考 + 回答的總上限；被截斷的比例太高就調大",
    )
    ap.add_argument(
        "--no-thinking",
        action="store_true",
        help="關閉思考模式（會讓 R 變空，本研究通常不要用）",
    )

    ap.add_argument("--workers", type=int, default=8, help="同時送出幾個請求")
    ap.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="重跑之前被截斷或沒寫 Answer: 的題目，通常搭配調大 --max-tokens 使用",
    )
    a = ap.parse_args(argv)

    from openai import OpenAI  # 放在這裡：只看 --help 時不需要安裝 openai

    out_path = Path(a.out or f"runs/musique.{model_tag(a.model)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gen = {
        "temperature": a.temperature,
        "top_p": a.top_p,
        "top_k": a.top_k,
        "max_tokens": a.max_tokens,
        "seed": a.seed,
    }
    cfg = config_hash(a.model, gen, not a.no_thinking, a.prompts)

    items = load_prompts(a.prompts, a.sample, a.seed, a.limit, a.ids)
    existing, bad_lines = scan_output(out_path, cfg)

    # 需要重跑的：--ids 明確指定的、--retry-incomplete 時的未完成題、prompt 內容改過的
    redo = set()
    if a.ids:
        redo |= {x["id"] for x in items if x["id"] in existing}
    if a.retry_incomplete:
        # 局部試跑只能重寫本次選中的題目，否則會刪掉未送回模型的舊紀錄。
        selected_ids = {x["id"] for x in items}
        redo |= {
            i
            for i, row in existing.items()
            if i in selected_ids and not is_complete(row)
        }
    redo |= {
        x["id"]
        for x in items
        if x["id"] in existing
        and (existing[x["id"]].get("meta") or {}).get("prompt_hash")
        != prompt_hash(x["prompt"])
    }
    if redo or bad_lines:
        rewrite_without(out_path, redo)
        for i in redo:
            existing.pop(i, None)

    todo = [x for x in items if x["id"] not in existing]
    print(
        f"[solver] 共 {len(items)} 題，已完成 {len(items) - len(todo)} 題，本次要跑 {len(todo)} 題 → {out_path}"
        f"（設定指紋 {cfg}）"
    )
    if not todo:
        return

    client = OpenAI(base_url=a.base_url, api_key=a.api_key)

    # 多個執行緒同時寫檔，要用鎖避免兩行資料交錯寫在一起
    lock = threading.Lock()
    stats = {
        "ok": 0,
        "failed": 0,
        "truncated": 0,
        "no_answer_line": 0,
        "no_reasoning": 0,
    }

    def work(item: dict) -> None:
        result = call_model(
            client, a.model, item["prompt"], gen, thinking=not a.no_thinking
        )
        warnings = check_output(result)
        row = {
            "id": item["id"],
            "model": a.model,
            "messages": [
                {"role": "user", "content": item["prompt"]},
                result["assistant"],
            ],
            "meta": {
                "config_hash": cfg,
                "prompt_hash": prompt_hash(item["prompt"]),
                "gen_params": {**gen, "thinking": not a.no_thinking},
                "finish_reason": result["finish_reason"],
                "latency_s": result["latency_s"],
                "usage": result["usage"],
                "warnings": warnings,
            },
        }
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            stats["ok"] += 1
            for w in warnings:
                stats[w] += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(work, x): x["id"] for x in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - 一題失敗不能讓整批停下來
                stats["failed"] += 1
                print(f"  ❌ {futures[fut]} 失敗：{e}", file=sys.stderr)
            if i % 50 == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t0)
                print(f"  進度 {i}/{len(todo)}（{rate:.2f} 題/秒）")

    # 最後的總結：這幾個數字決定要不要調參數重跑
    n = max(1, stats["ok"])
    print("\n[solver] 完成")
    print(
        f"  成功 {stats['ok']}、失敗 {stats['failed']}（失敗的題目重跑同一行指令即可補上）"
    )
    print(
        f"  被截斷 {stats['truncated']}（{stats['truncated'] / n:.1%}）  → 超過 5% 請調大 --max-tokens"
    )
    print(
        f"  沒寫 Answer: {stats['no_answer_line']}（{stats['no_answer_line'] / n:.1%}）"
    )
    print(
        f"  沒有思考內容 {stats['no_reasoning']}（{stats['no_reasoning'] / n:.1%}）  "
        f"→ 不是 0 的話，檢查 vLLM 有沒有加 --reasoning-parser"
    )

    if stats["failed"] > 0:
        # 成功紀錄已逐筆寫入；非零退出讓批次流程停止，續跑則只補尚未完成的題目。
        print("  有題目失敗；重跑同一行指令即可補上失敗的題目")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
