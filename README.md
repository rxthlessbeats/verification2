# verification2

USC EE599 Final Project：What Context Does a Verification Agent Really Need?

## 安裝

```bash
git clone https://github.com/rxthlessbeats/verification2
cd verification2
uv sync
uv run pre-commit install
bash scripts/get_data.sh
uv run pytest -q
```

`get_data.sh` 會把資料下載到 `data/`、`runs/`，最後應顯示「錯誤 0 筆」；`pytest` 應顯示 56 passed。

資料來源：[Google Drive](https://drive.google.com/drive/folders/1-Ahl5K9TkI4Ir_7RcduF9zjQ_TaF-GyK?usp=sharing)（不進 git）。

## qera：benchmark 與統一接口

### 資料

| 檔案 | 內容 |
|---|---|
| `data/tau2_bench.balanced.jsonl` | τ²-Bench，正確 / 錯誤各 250 |
| `data/swebench_verified.balanced.jsonl` | SWE-bench Verified，各 82 |
| `data/agentrewardbench.balanced.jsonl` | AgentRewardBench，各 250 |
| `runs/musique_dev_prompts.jsonl` | MuSiQue solver prompt，2,417 題 |

BrowseComp-Plus 等 judge 產生標籤、MuSiQue 等 solver。各 benchmark 說明見 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)，QERA 切分規則見 [`docs/SPEC.md`](docs/SPEC.md)。

### 接口用法

```python
from qera.cli import read_jsonl
from qera.render import build_verifier_input, selector_view

for traj in read_jsonl(["data/tau2_bench.balanced.jsonl"]):
    traj.label.correct                                        # 標籤：True / False
    build_verifier_input(traj, "QEA").text                    # 條件：QERA / QA / QEA
    selector_view(traj)                                       # 給 Jev 的輸入（不含答案）
    build_verifier_input(traj, "QEJevA", selected_e=seg_ids)  # Jev 選出的 E（seg_id 清單）
```

### 常用指令

| 指令 | 用途 |
|---|---|
| `uv run qera stats data/*.balanced.jsonl` | 筆數、token 分佈 |
| `uv run qera show data/tau2_bench.balanced.jsonl --index 0 --condition QEA` | 看 verifier 收到什麼 |
| `uv run qera validate data/*.jsonl` | 檢查格式 |
| `uv run qera apply-overrides data/x.jsonl overrides/kai.jsonl --out data/x.v2.jsonl` | 套用 Jev 分類 |
| `uv run qera sample data/full/x.jsonl --per-class 250 --out data/x.balanced.jsonl` | 重新抽樣 |

### Solver / Judge 的輸出格式

`scripts/run_solver.py`、`scripts/judge_bcp.py` 會把模型輸出存成統一格式。
模型需透過 vLLM 的 OpenAI 相容接口提供，並開啟 `--reasoning-parser`，R 才會被記錄。
`scripts/carc_solver.slurm` 是 CARC 上的參考範本。

### 從頭重建資料（通常不需要）

```bash
uv run bash scripts/download_data.sh
uv run bash scripts/convert_all.sh
```

第一行下載官方原始資料到 `raw/`，第二行轉換、驗證並抽樣。

### 注意

- 資料檔不要 commit（`.gitignore` 已排除）。
- BrowseComp-Plus 資料規定不得公開明文，`raw/`、`judgments/` 不要強制加入。
