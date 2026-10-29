#!/usr/bin/env bash
# 下載 6 個 benchmark 的原始資料與現成 trajectory。
# 請在「自己的終端機」執行（需要能連 HuggingFace、Google Drive、AWS S3、GitHub）。
#
#   bash scripts/download_data.sh            # 全部
#   bash scripts/download_data.sh musique tau2   # 只下載指定的
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

musique() {
  $PIP install -q gdown && \
  mkdir -p musique && cd musique && \
  gdown 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h -O musique_v1.0.zip && \
  unzip -o -q musique_v1.0.zip && rm -f musique_v1.0.zip && rm -rf __MACOSX && cd "$RAW" && \
  ok "MuSiQue → raw/musique/data/musique_ans_v1.0_dev.jsonl" || { cd "$RAW"; fail MuSiQue "Google Drive 下載不了時，請手動開 https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h 下載後解壓到 raw/musique/"; }
}

browsecomp_plus() {
  [ -d BrowseComp-Plus ] || git clone -q --depth 1 https://github.com/texttron/BrowseComp-Plus.git
  cd BrowseComp-Plus && $PIP install -q datasets huggingface_hub && \
  python scripts_build_index/decrypt_dataset.py --output data/browsecomp_plus_decrypted.jsonl --generate-tsv topics-qrels/queries.tsv && \
  bash scripts_build_index/download_run_files.sh && cd "$RAW" && \
  ok "BrowseComp-Plus → raw/BrowseComp-Plus/data/（含 gpt5 / o3 的 trajectory）" || { cd "$RAW"; fail BrowseComp-Plus "需要 HuggingFace 存取；可先 huggingface-cli login"; }
  echo "   ⚠ BrowseComp 資料不得以明文放上網路，raw/ 與轉換後的 JSONL 都不要推到公開 repo"
}

tau2() {
  [ -d tau2-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau2-bench.git && \
  ok "τ²-Bench → raw/tau2-bench/data/tau2/results/final/（官方 trajectory 已在 repo 內）" || fail τ²-Bench "git clone"
}

appworld() {
  # AppWorld 依賴 pydantic 1.x，和 qera（pydantic 2）衝突，所以裝在自己獨立的虛擬環境 raw/appworld/.venv-appworld
  mkdir -p appworld && cd appworld && \
  { [ -x .venv-appworld/bin/appworld ] || { python3 -m venv .venv-appworld && .venv-appworld/bin/pip install -q appworld; }; } && \
  .venv-appworld/bin/appworld install && .venv-appworld/bin/appworld download data && \
  .venv-appworld/bin/appworld download experiment-outputs && cd "$RAW" && \
  ok "AppWorld → raw/appworld/data、raw/appworld/experiments/outputs/" || { cd "$RAW"; fail AppWorld "見 https://github.com/StonyBrookNLP/appworld 的安裝說明"; }
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

ALL=(musique browsecomp_plus tau2 appworld swebench agentrewardbench)
TARGETS=("${@:-${ALL[@]}}")
for t in "${TARGETS[@]}"; do echo "── $t"; "$t"; done
