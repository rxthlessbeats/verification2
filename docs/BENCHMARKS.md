# Benchmark 確認表

> **2026-10-28 組上決定：只使用 τ²-Bench、SWE-bench Verified、AgentRewardBench 3 個 benchmark。** MuSiQue、BrowseComp-Plus、AppWorld 已從 repo 移除（需要時可從 git 歷史找回）。

> 最後更新：2026-10-28（程式 v1.1）

配置：工具使用、軟體工程、網頁操作各 1 個，E 的來源分別是 API 回傳、程式執行結果、網頁內容。

## 總表

| 領域 | Benchmark | 授權 | 下載來源 | 現成 trajectory | 標籤來源 | 狀態 |
|---|---|---|---|---|---|---|
| 工具使用 | **τ²-Bench** | MIT | GitHub（trajectory 就在 repo 內） | ✅ GPT-4.1、Claude 3.7、o4-mini × airline / retail / telecom × 4 trials | 環境 reward（DB 狀態 + 動作 + 告知資訊） | ✅ **可用**（平衡樣本 500） |
| 軟體工程 | **SWE-bench Verified** | SWE-bench 本體 MIT；experiments 與 trajectory repo 未附授權檔 | GitHub（2026 年中後的提交）或公開 S3 | ✅ Gemini 3.5 Flash（mini-swe-agent），441 條 | 測試是否通過（resolved） | ✅ **可用**（平衡樣本 164） |
| 網頁操作 | **AgentRewardBench** | repo 未附 LICENSE，使用前請看 HF 資料集頁面的授權 | HuggingFace `McGill-NLP/agent-reward-bench` | ✅ 1,302 條（4 個 LLM × 5 個網頁 benchmark）；排除 VisualWebArena 後 1,002 條 | 專家標註（已內附） | ✅ **可用**（平衡樣本 500） |

**目前可以直接跑實驗的：τ²-Bench、SWE-bench Verified、AgentRewardBench。**

## 已轉換的真實資料（平衡樣本）

| Benchmark | Solver | 筆數（正確 / 錯誤） | Q 中位數 | E 中位數 | R 中位數 | A 中位數 | E 段數中位數 | 原生推理 |
|---|---|---|---|---|---|---|---|---|
| τ²-Bench | Claude 3.7 Sonnet | 173（89 / 84） | 1,953 | 2,300 | 179（旁白） | 1,179 | 11 | 無 |
| τ²-Bench | GPT-4.1 | 160（77 / 83） | 1,978 | 1,771 | 0 | 847 | 11 | 無 |
| τ²-Bench | o4-mini | 167（84 / 83） | 1,915 | 1,968 | 0 | 762 | 8 | 隱藏 |
| SWE-bench Verified | Gemini 3.5 Flash | 164（82 / 82） | 287 | 13,028 | 920（旁白） | 313 | 44 | 隱藏（加密） |
| AgentRewardBench | GPT-4o | 120（64 / 56） | 35 | 20,836 | 780 | 151 | 14 | ✅ 完整 |
| AgentRewardBench | Qwen2.5-VL-72B | 110（54 / 56） | 37 | 24,545 | 1,354 | 151 | 17 | ✅ 完整 |
| AgentRewardBench | Llama-3.3-70B | 119（44 / 75） | 34 | 30,180 | 2,524 | 161 | 31 | ✅ 完整 |
| AgentRewardBench | Claude 3.7 Sonnet | 151（88 / 63） | 35 | 15,877 | 0 | 194 | 13 | 無（只輸出 `<action>`） |

**總長度（Q+E+R+A，不含 verifier 指令）**

| Benchmark | p50 | p90 | 最大 |
|---|---|---|---|
| τ²-Bench | 7.3k | 9.9k | 14.3k |
| SWE-bench Verified | 14.8k | 29.8k | 51.3k |
| AgentRewardBench | 24.7k | **101.7k** | **175.5k** |

單位：token（Qwen BPE，以 dashscope 內附的詞表計算；正式實驗請用 verifier 實際的 tokenizer 跑 `qera.cli retokenize`）。

抽樣說明：
- 每個 (題目, 模型) 只取一條，在看標籤之前隨機選；之後正確、錯誤各取一樣多。每個平衡檔旁邊的 `.manifest.json` 記錄排除原因與參數。
- τ²：排除 2 條沒有 E 的（agent 沒呼叫任何工具）。
- SWE-bench：Gemini 3.5 Flash 的 resolve 率高，441 題裡只有 82 題失敗，所以最多 82 對。
- AgentRewardBench：排除 13 條標註者沒有過半數共識的。**整體正確 / 錯誤各 250，但各模型內部不平衡**（例如 Llama 44 / 75）；要分模型分析時，請用 `--stratify-by solver.model` 重新抽樣。split：test 416、dev 84。
- 工具回傳順序被調換（`call_order_changed`）：全部資料都是 0。

## 研究問題 (1)「拿掉 R 會不會失去資訊」目前能用的資料

| 資料 | 條數（平衡樣本中） | 有原生推理 |
|---|---|---|
| AgentRewardBench：GPT-4o、Qwen2.5-VL-72B、Llama-3.3-70B | **349**（正確 162 / 錯誤 187） | ✅ |
| AgentRewardBench：Claude 3.7 Sonnet | 151 | ❌ |
| τ²-Bench（三個模型） | 500 | ❌ |
| SWE-bench Verified | 164 | ❌ |

**目前研究問題 (1) 的證據只有 ARB 的 349 條。** 要擴充，可以用推理可見的 solver（例如千問 thinking mode）自己跑 τ²（`tau2 run`）。沒有原生推理的資料仍可用於研究問題 (2)、(3)。

## 各 benchmark 的風險與注意事項

**τ²-Bench**
- ⚠ 官方 trajectory **完全沒有原生推理**：GPT-4.1 呼叫工具時通常不說話；o4-mini 的推理被 API 隱藏；Claude 3.7 只有簡短旁白。
- telecom 的 policy 很長（約 5k tokens），佔 Q 的大部分；可以考慮只用 airline + retail，或在分析時把 policy 長度當控制變數。
- 約 20% 的 segment 標 ambiguous（中途對使用者的回覆），是人工稽核的重點。

**SWE-bench Verified**
- E 很長，是測 Jev 壓縮效益的好場景，也是成本較高的場景。
- Gemini 的推理是加密的（`thought_signature`），R 只有旁白。想要完整 R，可以找推理可見的提交（例如 S3 上的 DeepSeek-V3.2-Reasoner，要先確認 traj 裡有沒有存推理內容）。

**AgentRewardBench**
- R 取自模型原始輸出的 `<think>`。**Claude 3.7 Sonnet 完全沒有推理**（prompt 有要求，但模型只輸出 `<action>`），其他三個模型都有完整原生推理。
- 預設排除 VisualWebArena（需要看圖，verifier 只吃文字）。
- **最長的 benchmark**：p90 約 10 萬 tokens、最長約 17.5 萬，主要來自每一步的 accessibility tree。要確認 verifier 的 context 長度，或在抽樣時用 `--max-total-tokens` 排除過長的。
- 有官方 dev / test：調 Jev 與報告結果最好用不同的 split（`--split`），切法由全組決定。

## 下一步

| 事項 | 負責 |
|---|---|
| 把平衡樣本交給組員，pipeline 先用 τ² 與 ARB 跑通 | 我 → pipeline 負責人 |
| 決定 verifier 的 context 長度（ARB p90 約 10 萬、SWE 最長約 5 萬 tokens） | Magnus |
| （可選）用千問 thinking mode 跑 τ²，擴充有原生 R 的資料 | 全組決定 |
| 決定 ARB 的 Claude 3.7 排除或分層、tuning / held-out 的切法 | 全組 |
| 人工稽核 E/R 切分（`audit/`，盲化版本） | 我 |
