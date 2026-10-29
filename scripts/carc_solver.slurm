#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# 在 CARC Discovery 上：開 vLLM（千問）→ 跑 solver → 關掉 vLLM
#
# 用法（在 repo 根目錄，且已經 source .venv/bin/activate）：
#     sbatch scripts/carc_solver.slurm
#
# 送出前請確認下面標 ★ 的地方。
# ════════════════════════════════════════════════════════════════════

#SBATCH --job-name=qera-solver
#SBATCH --account=rashidin_1753          # ★ 專案帳號
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                # ★ 27B 的 bf16 權重約 54 GB，需要 80 GB 的 A100；
                                         #   只有 40/48 GB 的卡就改成 gpu:a40:2，並把下面 TP 改成 2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00                  # 中斷也沒關係：重新 sbatch 會從斷掉的地方接著跑
#SBATCH --output=logs/solver_%j.out

set -euo pipefail
mkdir -p logs

# ★ 模型：請跟 Magnus 確認實際使用的千問版本與名稱
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TP="${TP:-1}"                            # tensor parallel：用幾張 GPU 分攤一個模型
MAX_LEN="${MAX_LEN:-32768}"              # context 長度上限
PORT="${PORT:-8000}"
SAMPLE="${SAMPLE:-1000}"                 # 從 2,417 題 MuSiQue dev 抽幾題

# ★ 啟用裝有 vllm 和 openai 的環境（依你們實際的環境調整）
# module load conda && conda activate vllm
# source ~/venvs/vllm/bin/activate

# ── 1. 背景啟動 vLLM ─────────────────────────────────────────────
# --reasoning-parser qwen3：把千問的思考內容放進獨立欄位（reasoning_content），這就是研究要的 R
vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --reasoning-parser qwen3 \
    --port "$PORT" > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT   # job 結束時一定關掉 vLLM

# ── 2. 等 vLLM 載完模型（最多 20 分鐘）──────────────────────────────
echo "等待 vLLM 啟動..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then
        echo "vLLM 已就緒（約 $((i * 10)) 秒）"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM 啟動失敗，請看 logs/vllm_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 10
done

# ── 3. 跑 solver ────────────────────────────────────────────────
# 輸出檔名在這裡明確指定，step 4 用同一個變數，避免兩邊推導方式不同而對不上
OUT="runs/musique.$(basename "$MODEL" | sed 's/[^A-Za-z0-9._-]/_/g').jsonl"   # 與 run_solver.model_tag 相同的規則
python scripts/run_solver.py \
    --prompts runs/musique_dev_prompts.jsonl \
    --base-url "http://localhost:${PORT}/v1" \
    --model "$MODEL" \
    --out "$OUT" \
    --sample "$SAMPLE" --seed 0 \
    --workers 16
# run_solver 有題目失敗時會以非 0 結束；set -e 會讓 job 停在這裡，不會拿不完整的結果去轉換

# ── 4. 轉成 QERA 格式（只轉 MuSiQue，其他 benchmark 不受影響）──────────
python -m qera.cli convert musique \
    --data raw/musique/data/musique_ans_v1.0_dev.jsonl \
    --runs "$OUT" \
    --out "data/full/musique_ans.$(basename "$OUT" .jsonl | cut -d. -f2-).jsonl"
