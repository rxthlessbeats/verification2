#!/usr/bin/env bash
# 下載 benchmark 的原始資料與現成 trajectory。
# 請在「自己的終端機」執行（需要能連 HuggingFace、Google Drive、AWS S3、GitHub）。
#
#   bash scripts/download_data.sh                  # τ²、SWE-bench、AgentRewardBench
#   bash scripts/download_data.sh tau2             # 只下載指定的
#
# 下載到 ./raw/。每個 benchmark 各自獨立，一個失敗不影響其他。
# 需要：git、python3 ≥ 3.10、pip。HuggingFace 的部分可能需要先 `huggingface-cli login`。

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/raw"
mkdir -p "$RAW"
cd "$RAW"

ok()   { echo "✅ $1"; }
fail() { echo "❌ $1 失敗：$2"; }

# 用 uv 建的虛擬環境裡沒有 pip，所以有 uv 就用 uv pip
if command -v uv >/dev/null 2>&1; then PIP="uv pip"; else PIP="pip"; fi

tau2() {
  [ -d tau2-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau2-bench.git && \
  ok "τ²-Bench → raw/tau2-bench/data/tau2/results/final/（官方 trajectory 已在 repo 內）" || fail τ²-Bench "git clone"
}

swebench() {
  # SWE-bench/experiments 只放提交的 metadata 與每題結果；trajectory 在提交者自己的 repo 或 S3
  [ -d swe-experiments ] || git clone -q --depth 1 --filter=blob:none --sparse https://github.com/SWE-bench/experiments.git swe-experiments
  cd swe-experiments && git sparse-checkout set evaluation/verified/20260901_mini-v2.4.2_gemini-3-5-flash && cd "$RAW"
  [ -d swe-gemini-3-5-flash ] || git clone -q --depth 1 https://github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash.git swe-gemini-3-5-flash && \
  ok "SWE-bench Verified（Gemini 3.5 Flash, mini-swe-agent）→ raw/swe-gemini-3-5-flash/trajs" || fail SWE-bench "git clone"
  # 其他提交（2026 年中以前）在公開 S3，例如推理可見的 DeepSeek-V3.2-Reasoner：
  #   aws s3 sync --no-sign-request s3://swe-bench-submissions/bash-only/20251201_mini-v1.17.1_deepseek-v3.2-reasoner/trajs raw/swe-deepseek/trajs
  # 實際路徑以該提交 metadata.yaml 的 assets.trajs 為準。
}

agentrewardbench() {
  $PIP install -q huggingface_hub && \
  python - <<'PY' && ok "AgentRewardBench → raw/agent-reward-bench/trajectories/cleaned/" || fail AgentRewardBench "需要 HuggingFace 存取"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="McGill-NLP/agent-reward-bench", repo_type="dataset",
                  local_dir="agent-reward-bench/trajectories", allow_patterns=["cleaned/**"])
PY
}

ALL=(tau2 swebench agentrewardbench)
TARGETS=("${@:-${ALL[@]}}")
for t in "${TARGETS[@]}"; do echo "── $t"; "$t"; done
