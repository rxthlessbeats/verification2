# 6 個 Benchmark 確認表

> 最後更新：2026-10-25（程式 v1.1，資料已用新版重新轉換）

配置：3 個領域 × 每領域 2 個，每個結論都能在同類任務上用第二個 benchmark 複驗。

## 總表

| 領域 | Benchmark | 授權 | 下載來源 | 現成 trajectory | 標籤來源 | 狀態 |
|---|---|---|---|---|---|---|
| 多跳問答 | **MuSiQue-Ans** | CC BY 4.0 | Google Drive（官方 repo 的 `download_data.sh`） | ❌ 需自己跑 solver | EM / F1（自動） | 資料已下載、adapter 已在真實資料上驗證；**等 solver**（prompt 已備好：`runs/musique_dev_prompts.jsonl`，2,417 題） |
| 多跳問答 | **BrowseComp-Plus** | MIT（程式碼）；資料有加密，不得公開明文 | HuggingFace `Tevatron/browsecomp-plus`、`-runs` | ✅ gpt5、o3 × BM25 / Qwen3-Embed（4 個 run，各 830 題） | LLM judge（官方 Qwen3-32B） | 已轉換 4 × 830 條；**等 judge 產生標籤** |
| 工具使用 | **τ²-Bench** | MIT | GitHub（trajectory 就在 repo 內） | ✅ GPT-4.1、Claude 3.7、o4-mini × airline / retail / telecom × 4 trials | 環境 reward（DB 狀態 + 動作 + 告知資訊） | ✅ **可用**（平衡樣本 500） |
| 工具使用 | **AppWorld** | Apache 2.0 | AWS S3（`appworld download`） | ⚠ 官方輸出只有 API 請求，沒有回傳與 LM 對話，**不能用** | state-based 單元測試 | 需自己跑 agent；待組上決定 |
| 軟體 / 網頁 | **SWE-bench Verified** | SWE-bench 本體 MIT；experiments 與 trajectory repo 未附授權檔 | GitHub（2026 年中後的提交）或公開 S3 | ✅ Gemini 3.5 Flash（mini-swe-agent），441 條 | 測試是否通過（resolved） | ✅ **可用**（平衡樣本 164） |
| 軟體 / 網頁 | **AgentRewardBench** | repo 未附 LICENSE，使用前請看 HF 資料集頁面的授權 | HuggingFace `McGill-NLP/agent-reward-bench` | ✅ 1,302 條（4 個 LLM × 5 個網頁 benchmark）；排除 VisualWebArena 後 1,002 條 | 專家標註（已內附） | ✅ **可用**（平衡樣本 500） |

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

**目前研究問題 (1) 的證據只有 ARB 的 349 條。** 要擴充，需要用推理可見的 solver（例如千問 thinking mode）自己跑 MuSiQue 和 τ²（`scripts/run_solver.py`、`tau2 run`）。沒有原生推理的資料仍可用於研究問題 (2)、(3)。

## 各 benchmark 的風險與注意事項

**MuSiQue-Ans**
- 答案多是維基百科知識，verifier 可能不看 E 也知道答案 → 建議額外跑一個只給 Q 的 closed-book 測試，標出 verifier 本來就會的題目，分析時排除或分層。
- EM 只比字串，同義不同寫法會被判錯；`label.detail.needs_review` 的樣本建議人工抽查。
- 生成被截斷的題目會標 `incomplete` 並在抽樣時排除；被截斷比例高時，調大 `--max-tokens` 後用 `--retry-incomplete` 重跑。

**BrowseComp-Plus**
- 題目很難（GPT-5 約 55–70%），錯誤樣本多，正好補足 MuSiQue。
- **非常長**：E 的中位數約 7.2 萬 tokens，p90 約 10.6 萬。四種條件裡 QERA / QEA 可能超過 verifier 的 context 長度，要先和 Magnus 確認。
- 標籤要先跑 `scripts/judge_bcp.py`（設定與官方相同）；judge 模型不要和 verifier 同一個。
- 資料不得以明文公開，`raw/`、`data/full/browsecomp_plus*`、`judgments/`、`audit/browsecomp_plus.md` 都不要推到公開 repo。

**τ²-Bench**
- ⚠ 官方 trajectory **完全沒有原生推理**：GPT-4.1 呼叫工具時通常不說話；o4-mini 的推理被 API 隱藏；Claude 3.7 只有簡短旁白。
- telecom 的 policy 很長（約 5k tokens），佔 Q 的大部分；可以考慮只用 airline + retail，或在分析時把 policy 長度當控制變數。
- 約 20% 的 segment 標 ambiguous（中途對使用者的回覆），是人工稽核的重點。

**AppWorld**
- 官方實驗輸出不能用。若要納入，需自己用 AppWorld 的 agent 框架接千問跑；adapter 還有已知問題（見 SPEC §7.4），決定要用之後再修。

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
| 把平衡樣本交給組員，pipeline 先用 τ² 與 ARB 跑通 | 我 → 子豪 |
| 決定 verifier 的 context 長度（ARB p90 約 10 萬、SWE 最長約 5 萬 tokens） | Magnus |
| 用千問 thinking mode 跑 MuSiQue（及 τ²）的 solver | Magnus |
| 在 CARC 跑 BrowseComp-Plus 的 judge | 我 / Magnus |
| 決定 ARB 的 Claude 3.7 排除或分層、tuning / held-out 的切法 | 全組 |
| AppWorld：自己跑、換掉，或減成 5 個 benchmark | 全組 |
| 人工稽核 E/R 切分（`audit/`，盲化版本） | 我 |
