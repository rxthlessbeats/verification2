#!/usr/bin/env bash
# 從 Google Drive 下載資料到 data/ 與 runs/。已下載且大小正確的檔案會略過。
# 用法：bash scripts/get_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data runs

# 檔案 ID | 存放位置 | 正確大小（bytes）
FILES="
1ZQWnbZ3nstTmFiJw45pMR5ROqi_rxWam data/tau2_bench.balanced.jsonl 21305014
1mgGGrSrKvsFAMYrrFqG3ujypwuO3QitY data/swebench_verified.balanced.jsonl 16840909
1NRqyuEredqz19MCZSwHfuNhOXcOHUflw data/agentrewardbench.balanced.jsonl 75226900
1168GyW_ptIFJFAJYnTaxZMURa0y6DyA5 runs/musique_dev_prompts.jsonl 27468551
"

size() { [ -f "$1" ] && wc -c < "$1" | tr -d ' ' || echo 0; }

echo "$FILES" | while read -r id path bytes; do
  [ -z "$id" ] && continue
  if [ "$(size "$path")" = "$bytes" ]; then
    echo "✅ 已存在：$path"
    continue
  fi
  echo "⬇️  下載：$path"
  uvx gdown "$id" -O "$path" --quiet || true
  if [ "$(size "$path")" != "$bytes" ]; then
    echo "❌ $path 大小不對（$(size "$path") bytes，應為 $bytes）。請確認 Drive 的分享設定是「知道連結的任何人：檢視者」。"
    exit 1
  fi
  echo "✅ 完成：$path"
done

uv run qera validate data/*.balanced.jsonl
