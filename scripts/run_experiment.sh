#!/usr/bin/env bash
# 跑完整實驗：每個 benchmark 的 QERA / QA / QEA 各跑一次（跟 selector、預算無關），
# QEsA 則對每個預算 × 每個 selector（jev / random / recency / relevance）各跑一次。
# 這樣 verifier 不會對同一個條件重複呼叫。結果用 scripts/collect_runs.py 彙整。
#
# 需要的環境變數：
#   VERIFIER_ENDPOINT   完整的 chat completions URL，例如 http://<magnus-host>:8000/v1/chat/completions
#   VERIFIER_MODEL      vLLM 裡的模型名稱，例如 Qwen/Qwen3-27B
#   VERIFIER_API_KEY    （可選）Bearer token
# 可選：
#   BUDGETS="512 1024 2048 4096"   RATIOS=""   LIMIT=""   OUT=runs   TIMEOUT=300   MAX_TOKENS=512
#
#   VERIFIER_ENDPOINT=... VERIFIER_MODEL=... bash scripts/run_experiment.sh                   # 全部
#   LIMIT=20 VERIFIER_ENDPOINT=... VERIFIER_MODEL=... bash scripts/run_experiment.sh tau2_bench  # 先試 20 條
#   VERIFIER_ENDPOINT=... VERIFIER_MODEL=... BACKEND=dry-run bash scripts/run_experiment.sh     # 只組 prompt 不呼叫
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VERIFIER_ENDPOINT:?set VERIFIER_ENDPOINT}"; : "${VERIFIER_MODEL:?set VERIFIER_MODEL}"
BACKEND="${BACKEND:-http}"
BUDGETS="${BUDGETS:-512 1024 2048 4096}"
RATIOS="${RATIOS:-}"
OUT="${OUT:-runs}"
LIMIT_ARG=(); [ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")
if [ "$#" -gt 0 ]; then BENCHES="$*"; else BENCHES="tau2_bench swebench_verified agentrewardbench"; fi
# urllib 需要 certifi 的憑證才能連 https（vLLM 本機 http 不需要，但設了無害）
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(uv run python -c 'import certifi; print(certifi.where())')}"
COMMON=(--backend "$BACKEND" --endpoint "$VERIFIER_ENDPOINT" --model "$VERIFIER_MODEL"
        --timeout "${TIMEOUT:-300}" --max-tokens "${MAX_TOKENS:-512}" "${LIMIT_ARG[@]}")

run() {  # run <output dir> <args...>; 已存在的輸出目錄會略過，所以中斷後重跑會接著做
  local dir="$1"; shift
  if [ -d "$dir" ]; then echo "skip (exists): $dir"; return; fi
  echo "== $dir"
  uv run verification2 run --output "$dir" "$@" "${COMMON[@]}" || true   # 有 error 的格會回傳 1，照樣繼續
  # 把 selections 檔放進輸出目錄，collect_runs.py 靠它把 Jev 的 selection 成本和延遲併進 C_c
  local i; for ((i = 1; i <= $#; i++)); do
    if [ "${!i}" = "--selections" ]; then local j=$((i + 1)); cp "${!j}" "$dir/selections.jsonl"; fi
  done
}

for b in $BENCHES; do
  data="data/$b.balanced.jsonl"
  run "$OUT/$b/full" --input "$data" --conditions QERA QA QEA --budget 0
  for budget in $BUDGETS; do
    sel="selections/${b}_jev_b${budget}.jsonl"
    [ -f "$sel" ] || { echo "missing $sel (run scripts/jev_select_all.sh first)"; exit 1; }
    run "$OUT/$b/b${budget}_jev" --input "$data" --conditions QEsA --budget "$budget" --selector file --selections "$sel"
    for s in random recency relevance; do
      run "$OUT/$b/b${budget}_$s" --input "$data" --conditions QEsA --budget "$budget" --selector "$s"
    done
  done
  for ratio in $RATIOS; do
    # 比例預算：pipeline 的 --budget 是全域值，這裡放一個夠大的上限，實際每條的預算在 selections 檔裡；
    # baseline 沒有對應的比例版本，所以只跑 jev。
    sel="selections/${b}_jev_r${ratio}.jsonl"
    [ -f "$sel" ] || { echo "missing $sel"; exit 1; }
    run "$OUT/$b/r${ratio}_jev" --input "$data" --conditions QEsA --budget 1000000 --selector file --selections "$sel"
  done
done
echo "done. 彙整：uv run python scripts/collect_runs.py $OUT"
