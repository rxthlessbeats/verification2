#!/usr/bin/env bash
# 從 Jev 快取產出每個 benchmark、每個預算的 selections 檔。
# 分數已經在 .jev_cache/ 裡，所以換預算不會再打 Jev（selection_cost_usd 會是 0）。
# 第一次對某個 benchmark 打分請先跑：
#   uv run python -m verification2.jev select --input data/<b>.balanced.jsonl --budget 2048 --out selections/full/<b>_jev_2048.jsonl
#
#   bash scripts/jev_select_all.sh                       # 全部 benchmark、預設預算
#   BUDGETS="1024 4096" bash scripts/jev_select_all.sh   # 自訂固定預算
#   RATIOS="0.25" bash scripts/jev_select_all.sh tau2_bench
set -euo pipefail
cd "$(dirname "$0")/.."
BUDGETS="${BUDGETS:-512 1024 2048 4096}"
RATIOS="${RATIOS:-0.25 0.5}"
if [ "$#" -gt 0 ]; then BENCHES="$*"; else BENCHES="tau2_bench swebench_verified agentrewardbench"; fi
mkdir -p selections
for b in $BENCHES; do
  for budget in $BUDGETS; do
    uv run python -m verification2.jev select --input "data/$b.balanced.jsonl" \
      --budget "$budget" --run "select-$b" --out "selections/${b}_jev_b${budget}.jsonl" --overwrite 2>/dev/null
  done
  for ratio in $RATIOS; do
    uv run python -m verification2.jev select --input "data/$b.balanced.jsonl" \
      --budget-ratio "$ratio" --run "select-$b" --out "selections/${b}_jev_r${ratio}.jsonl" --overwrite 2>/dev/null
  done
done
echo "selections/ 內容："; ls selections/*.jsonl | grep -v scores
