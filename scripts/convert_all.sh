#!/usr/bin/env bash
# 把 raw/ 裡的資料轉成統一格式（data/full/），驗證，再抽平衡子集（data/*.balanced.jsonl）。
# 先跑 scripts/download_data.sh。MuSiQue 若要用千問當 solver，見 README「自己跑 solver」。
#
#   bash scripts/convert_all.sh                          # 全部
#   bash scripts/convert_all.sh agentrewardbench tau2     # 只跑指定的 benchmark
#   PER_CLASS=150 bash scripts/convert_all.sh             # 每類抽 150 筆
#
# 可指定的名稱：tau2 swebench agentrewardbench browsecomp_plus appworld musique
#
# 失敗處理（為什麼這樣設計）：
#   之前任何一步失敗都只印一行 ❌ 然後繼續，最後仍然抽樣、印統計，看起來像整批成功，
#   下游可能拿到殘缺或過期的資料。現在：
#   - 任何一個轉換失敗 → 不做驗證和抽樣，直接以失敗結束
#   - 驗證有「錯誤」→ 停止（「警告」不會停，例如少數沒有 E 的 trajectory，抽樣時會自動排除）
#   - 抽樣時某個 benchmark 沒有標籤（例如 BrowseComp-Plus 還沒跑 judge）→ 明確標示「略過」，
#     並刪掉舊的平衡檔，不會留下過期資料
#   - 每個輸出都是先寫暫存檔、成功才替換，中途失敗不會留下半個檔案

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/raw"
PER_CLASS="${PER_CLASS:-250}"
cd "$ROOT"
Q="python -m qera.cli"
mkdir -p data/full

if [ "$#" -gt 0 ]; then TARGETS=" $* "; else TARGETS=" tau2 swebench agentrewardbench browsecomp_plus appworld musique "; fi
want() { case "$TARGETS" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

FAILED=()
CONVERTED=()
run() {
  local name="$1" out="$2"; shift 2
  echo "── $name"
  if "$@"; then CONVERTED+=("$out"); else echo "❌ $name 轉換失敗"; FAILED+=("$name"); fi
}

# τ²-Bench：官方 trajectory（GPT-4.1 / Claude 3.7 / o4-mini；airline, retail, telecom；default 模式）
T="$RAW/tau2-bench/data/tau2/results/final"
if want tau2 && [ -d "$T" ]; then
  TAU=()   # 路徑有空白（Fall 2026 等），用陣列保留完整檔名
  for f in "$T"/*_default_*.json; do case "$f" in *workflow*) ;; *) TAU+=("$f") ;; esac; done
  run tau2 data/full/tau2_bench.jsonl \
    $Q convert tau2 --tau2-root "$RAW/tau2-bench" --out data/full/tau2_bench.jsonl --input "${TAU[@]}"
fi

# SWE-bench Verified：Gemini 3.5 Flash（mini-swe-agent）
S="$RAW/swe-gemini-3-5-flash/trajs"
if want swebench && [ -d "$S" ]; then
  run swebench data/full/swebench_verified.jsonl \
    $Q convert swebench --trajs "$S" --model gemini-3-5-flash \
    --labels "$RAW/swe-experiments/evaluation/verified/20260901_mini-v2.4.2_gemini-3-5-flash/per_instance_details.json" \
    --out data/full/swebench_verified.jsonl
fi

# AgentRewardBench：官方 trajectory + 專家標註（labels 已內附）；預設排除 VisualWebArena（需要看圖）
A="$RAW/agent-reward-bench/trajectories/cleaned"
if want agentrewardbench && [ -d "$A" ]; then
  run agentrewardbench data/full/agentrewardbench.jsonl \
    $Q convert agentrewardbench --cleaned "$A" --out data/full/agentrewardbench.jsonl
fi

# BrowseComp-Plus：官方 gpt5 / o3 trajectory；標籤要先跑 scripts/judge_bcp.py
#   judge 輸出請命名為 judgments/bcp_<retriever>_<模型>.jsonl，例如 judgments/bcp_bm25_gpt5.jsonl
B="$RAW/BrowseComp-Plus"
if want browsecomp_plus && [ -d "$B/data/decrypted_run_files" ]; then
  for f in "$B"/data/decrypted_run_files/*/*.jsonl; do
    tag="$(basename "$(dirname "$f")")_$(basename "$f" .jsonl)"
    J="judgments/bcp_${tag}.jsonl"
    JARGS=(); [ -f "$J" ] && JARGS=(--judgments "$J")
    run "bcp $tag" "data/full/browsecomp_plus.$tag.jsonl" \
      $Q convert browsecomp_plus --queries "$B/data/browsecomp_plus_decrypted.jsonl" --runs "$f" \
      --qrels-evidence "$B/topics-qrels/qrel_evidence.txt" --qrels-gold "$B/topics-qrels/qrel_golds.txt" \
      "${JARGS[@]+"${JARGS[@]}"}" --out "data/full/browsecomp_plus.$tag.jsonl"
  done
fi

# AppWorld：官方 baseline 的實驗輸出只有「送出的 API 請求」和資料庫變更，
# 沒有 API 回傳（E）也沒有 LM 對話（R），不能用於本研究。
# 自己跑 agent 時（會產生 logs/lm_calls.jsonl），把實驗資料夾放在 runs/appworld/<exp>/ 再轉換。
if want appworld; then
  for exp in runs/appworld/*/; do
    [ -d "$exp" ] || continue
    name="$(basename "$exp")"
    run "appworld $name" "data/full/appworld.$name.jsonl" \
      $Q convert appworld --experiment "$exp" --dataset test_normal \
      --data-root "$RAW/appworld/data" --out "data/full/appworld.$name.jsonl"
  done
fi

# MuSiQue：沒有官方 trajectory，需要自己跑 solver（scripts/run_solver.py），有 runs 檔才轉換
M="$RAW/musique/data/musique_ans_v1.0_dev.jsonl"
if want musique; then
  for r in runs/musique.*.jsonl; do
    [ -f "$r" ] && [ -f "$M" ] || continue
    name="$(basename "$r" .jsonl | cut -d. -f2-)"
    run "musique $name" "data/full/musique_ans.$name.jsonl" \
      $Q convert musique --data "$M" --runs "$r" --out "data/full/musique_ans.$name.jsonl"
  done
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "❌ 以下轉換失敗，停止（不做驗證和抽樣）：${FAILED[*]}"
  exit 1
fi
if [ "${#CONVERTED[@]}" -eq 0 ]; then
  echo "沒有找到可轉換的資料（raw/ 裡沒有對應的 benchmark？）"
  exit 1
fi

echo "── 驗證"
if ! $Q validate "${CONVERTED[@]}" > data/validate_log.txt 2>&1; then
  tail -40 data/validate_log.txt
  echo "❌ 驗證有錯誤（完整紀錄：data/validate_log.txt），停止，不抽樣"
  exit 1
fi
tail -1 data/validate_log.txt

echo "── 抽平衡子集（每類 ${PER_CLASS}）"
SAMPLED=(); SKIPPED=()
for f in "${CONVERTED[@]}"; do
  out="data/$(basename "$f" .jsonl).balanced.jsonl"
  $Q sample "$f" --per-class "$PER_CLASS" --out "$out"
  rc=$?
  if [ "$rc" -eq 0 ]; then SAMPLED+=("$out")
  elif [ "$rc" -eq 3 ]; then SKIPPED+=("$(basename "$f")")
  else echo "❌ 抽樣失敗：$f"; exit 1
  fi
done
[ "${#SKIPPED[@]}" -gt 0 ] && echo "⏭  沒有標籤而略過抽樣：${SKIPPED[*]}"

[ "${#SAMPLED[@]}" -gt 0 ] && $Q stats "${SAMPLED[@]}"
echo "✅ 完成。每個平衡檔旁邊都有 .manifest.json，記錄排除原因與參數。"
