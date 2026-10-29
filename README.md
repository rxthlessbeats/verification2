# verification2

USC EE599 Final Project：What Context Does a Verification Agent Really Need?

```bash
uv sync                       # 建環境、裝套件
uv run pre-commit install     # commit 前自動跑 ruff 檢查
uv run pytest -q              # 56 個測試應全部通過
```

## 專案結構

```
src/
  verification2/   pipeline（verifier、實驗流程）
  qera/            benchmark 統一接口：把 6 個 benchmark 的 trajectory 轉成 Q/E/R/A 格式，
                   並組出四種 verifier 輸入條件（QERA / QA / QEA / QEJevA）
    schema.py        Trajectory / Segment / Label 定義
    render.py        四種條件的 verifier 輸入、selector_view
    chat.py          OpenAI chat 格式 → segments（自己跑 solver 用）
    builder.py       adapter 共用工具
    reasoning.py     各種格式的推理內容（reasoning_content、<think> 等）
    tokens.py        token 計數（Qwen BPE）
    grading.py       EM / F1
    overrides.py     Jev 分類 / 人工稽核的角色覆寫
    cli.py           命令列（uv run qera ...）
    adapters/        6 個 benchmark 的 adapter
    resources/       τ² 工具類型表、τ² policy、AgentRewardBench 標註
scripts/           下載、批次轉換、solver（run_solver.py、carc_solver.slurm）、BrowseComp-Plus judge
tests/             測試與測試資料
docs/              SPEC.md（格式規格、QERA 切分規則）、BENCHMARKS.md（6 個 benchmark 的狀態與統計）
audit/             人工稽核表（盲化）
data/              只有 manifest 在 git 裡；資料檔請從 Google Drive 下載（見下方）
```

## 資料（不在 git 裡）

資料檔太大（最大 72 MB），pre-commit 和 GitHub 都會擋，所以放在 Google Drive：[benchmark 資料](https://drive.google.com/drive/folders/1-Ahl5K9TkI4Ir_7RcduF9zjQ_TaF-GyK?usp=sharing)。只要下載雲端上 `data/` 和 `runs/` 裡的 `.jsonl` 檔，放到 repo 的對應位置；程式碼一律以 repo 為準。

| 檔案 | 放到 | 內容 |
|---|---|---|
| `tau2_bench.balanced.jsonl` | `data/` | τ²-Bench，正確 / 錯誤各 250（3 個 solver） |
| `swebench_verified.balanced.jsonl` | `data/` | SWE-bench Verified，正確 / 錯誤各 82（Gemini 3.5 Flash） |
| `agentrewardbench.balanced.jsonl` | `data/` | AgentRewardBench，正確 / 錯誤各 250（4 個 solver；不含 VisualWebArena） |
| `musique_dev_prompts.jsonl` | `runs/` | MuSiQue 的 solver prompt（2,417 題） |

`data/*.balanced.manifest.json`（在 git 裡）記錄每個平衡檔的抽樣參數、排除原因與程式版本。
BrowseComp-Plus 已轉換但還沒有標籤（要先跑 judge）；MuSiQue 要先跑 solver。詳見 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)。

⚠ **BrowseComp-Plus 的資料不得以明文公開**：`raw/`、`judgments/`、`audit/browsecomp_plus.md` 已在 `.gitignore` 排除，不要強制加入。

指令可以用 `uv run ...`，或先 `source .venv/bin/activate` 再直接執行（CARC 上的 `carc_solver.slurm` 用後者）。

## 在 pipeline 裡使用

```python
from qera.cli import read_jsonl
from qera.render import build_verifier_input, selector_view

for traj in read_jsonl(["data/tau2_bench.balanced.jsonl"]):
    y = traj.label.correct                                   # True / False
    for cond in ["QERA", "QA", "QEA"]:
        vi = build_verifier_input(traj, cond)
        prompt = VERIFIER_INSTRUCTION + "\n\n" + vi.text      # 交給 Qwen 27B
        # vi.n_tokens = 實際輸入內容的 token；完整 prompt 與輸出 token、延遲請用 API 回傳的 usage

    # Jev / baseline：只能看 selector_view（不含標準答案）
    view = selector_view(traj)                                # question、answer、candidates
    chosen = my_selector(view, budget=B)                      # 回傳 seg_id 清單
    vi = build_verifier_input(traj, "QEJevA", selected_e=chosen, budget=B)  # 會檢查 ID 與預算
```

## 命令列

```bash
uv run qera convert <benchmark> [參數] --out data/full/x.jsonl   # benchmark: musique browsecomp_plus tau2 appworld swebench agentrewardbench
uv run qera convert tau2 --help                                  # 看某個 adapter 的參數
uv run qera validate data/*.jsonl                                # schema + 語意檢查
uv run qera stats data/*.balanced.jsonl                          # 每個 benchmark 的筆數、token 分佈
uv run qera sample data/full/x.jsonl --per-class 250 --out data/x.balanced.jsonl
uv run qera show data/x.jsonl --index 0 --condition QEA          # 看 verifier 實際會收到什麼
uv run qera audit data/x.jsonl --n 10 --out audit/x.md           # 產生人工稽核表
uv run qera apply-overrides data/x.jsonl overrides/kai.jsonl --out data/x.v2.jsonl
uv run qera retokenize data/x.jsonl --out data/x.qwen.jsonl      # 先 export QERA_TOKENIZER=Qwen/...
```

## 下載其他 benchmark

HuggingFace、Google Drive、S3 上的資料要在你自己的電腦下載：

```bash
uv run bash scripts/download_data.sh                 # 全部，或指定：uv run bash scripts/download_data.sh musique appworld
uv run bash scripts/convert_all.sh                   # 全部轉換 + 驗證 + 抽平衡子集 + 統計
```

BrowseComp-Plus 的標籤要先跑 LLM judge（在 CARC 上用 vLLM 開 Qwen3-32B）：

```bash
uv run python scripts/judge_bcp.py --queries raw/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
  --runs raw/BrowseComp-Plus/data/decrypted_run_files/bm25/gpt5.jsonl \
  --out judgments/bcp_bm25_gpt5.jsonl --base-url http://<node>:8000/v1 --model Qwen/Qwen3-32B
```

## 自己跑 solver（MuSiQue、或想要完整 R 的 τ²）

prompt 已經產生好：`runs/musique_dev_prompts.jsonl`（2,417 題）。

**最簡單的方式（CARC 上一行搞定）：**
```bash
sbatch scripts/carc_solver.slurm          # 開 vLLM → 跑 solver → 轉成 QERA 格式
```
送出前請先打開 `carc_solver.slurm`，確認標 ★ 的地方（模型名稱、GPU、環境）。

**手動執行（已經有 vLLM 在跑時）：**
```bash
# vLLM 一定要加 --reasoning-parser，思考內容（R）才會放在獨立欄位
vllm serve Qwen/Qwen3-32B --reasoning-parser qwen3 --max-model-len 32768 --port 8000

# 先跑 5 題確認格式
uv run python scripts/run_solver.py --prompts runs/musique_dev_prompts.jsonl --model Qwen/Qwen3-32B --limit 5
# 正式跑；中斷後重跑同一行會接著跑
uv run python scripts/run_solver.py --prompts runs/musique_dev_prompts.jsonl --model Qwen/Qwen3-32B --sample 1000 --seed 0 --workers 16
```
輸出在 `runs/musique.<模型名>.jsonl`，`uv run bash scripts/convert_all.sh` 會自動轉換。
跑完會印出「被截斷」「沒寫 Answer」「沒有思考內容」的比例，不是 0 的話請看 `run_solver.py` 開頭的說明。

τ²：`tau2 run --domain airline --agent-llm hosted_vllm/<model> ...`，產出的 results JSON 直接丟給 `convert tau2`；adapter 會從 `raw_data` 抽出 `reasoning_content` 當 R。

## 資料來源與授權

τ²-Bench（MIT, Sierra Research）、SWE-bench experiments 與提交者的 trajectory repo（未附授權檔）、AgentRewardBench 標註（McGill-NLP；repo 未附授權檔，對外發表前請確認 HF 資料集頁面）、BrowseComp-Plus judge prompt（MIT, texttron）。BrowseComp-Plus 的資料不得以明文公開。
