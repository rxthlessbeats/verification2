# QERA 統一資料格式規格（v1.1）

> v1.1 第二次更新（2026-10-25）：讀檔拒絕舊版資料、覆寫逐欄位合併、選擇器的 A 依原始順序、
> 「有沒有 E」的判斷統一、續跑修正、偵測工具回傳順序。標記為 🆕²。
>
> v1.1 更新（2026-10-24，依外部審查修正）：新增 spans / call_role / source_id；R 的來源改由實際內容判定；
> 截斷的生成會被標出並排除；judge 標籤綁定答案；選擇器拿不到標準答案；批次流程失敗即停止。詳見各節的 🆕。

> EE599 Final Project — *What Context Does a Verification Agent Really Need?*
> 負責人：Howard（Benchmark search、統一接口、統一輸出格式）
> 這份文件是組內的「資料合約」：pipeline、Jev、verifier 都只依賴這裡定義的格式。

---

## 1. 為什麼需要統一格式

六個 benchmark 的原始格式完全不同（τ² 是對話 JSON、SWE-bench 是 agent log、AgentRewardBench 是網頁 step、MuSiQue 是題目加段落……）。如果每個人各自解析原始資料：

- 子豪的 pipeline 要寫六套讀檔邏輯；
- Kai 的 Jev 要針對六種格式各自找出「哪些是 E」；
- 同一個 benchmark 的 E/R 切法可能在不同人手上不一致，實驗結果就不可比。

所以由我先把所有 benchmark 轉成**同一種格式**，之後所有人只讀這一種。

## 2. 核心設計原則

| # | 原則 | 理由 |
|---|---|---|
| 1 | **以 segment 為單位，Q/E/R/A 只是標籤** | 角色可以事後改（Kai 的分類、人工稽核），不用重新轉檔。 |
| 2 | **保留原始順序** | proposal 規定 replay 時保持原本的訊息順序；改順序本身就會影響 verifier。 |
| 3 | **四種條件不存四份資料，用一個函式過濾** | 保證四種條件只差在「放了哪些 segment」，沒有其他變因。 |
| 4 | **觀測一定附上產生它的呼叫** | 「`cat foo.py` 的輸出」只有在知道指令是 `cat foo.py` 時才看得懂。呼叫內容是 E 的出處，不是推理。 |
| 5 | **gold 資訊只放 `hidden` / `label`，render 永遠不讀** | 防止答案或 gold evidence 洩漏給 verifier。 |
| 6 | **不確定就標 `ambiguous`，不要硬猜** | proposal 要求人工稽核 E/R 切分，ambiguous 的 segment 優先稽核。 |
| 7 | **記錄 solver 的推理通道（`native_reasoning`），而且依實際內容判定** | R 的多寡取決於 solver 會不會吐出推理；分析時一定要分層，否則「拿掉 R 沒差」可能只是因為 R 本來就是空的。🆕 不能只看欄位名稱（AgentRewardBench 的 `reasoning` 欄位有時只是動作文字）。 |
| 8 | 🆕 **一段可以有多種角色；呼叫參數可以單獨改判** | 「請取消訂單，畫面顯示錯誤 404」前半是需求、後半是觀測；bash 指令、搜尋字串是 solver 寫的，可能夾帶推理。格式要能表達這些情況，**怎麼切則由 Kai 和人工稽核決定**。 |
| 9 | 🆕 **沒完成的生成不混進錯誤樣本** | 被截斷、沒交答案和答錯是兩回事。標 `meta.incomplete`，抽樣預設排除。 |
| 10 | 🆕 **失敗要大聲** | 轉換、驗證、抽樣任何一步失敗就停止；輸出先寫暫存檔，成功才替換；選到不存在的 ID、覆寫對不到資料都會報錯。 |

## 3. 檔案格式

- 一個 benchmark × 一個 solver 一個檔案，**JSONL**，一行一條 trajectory。
- 命名：`data/<benchmark>.balanced.jsonl`（平衡子集，實驗用）、`data/full/<benchmark>.jsonl`（全部）。

### 3.1 Trajectory

| 欄位 | 型別 | 說明 |
|---|---|---|
| `schema_version` | str | 目前 `"1.1"`。🆕² 讀檔與 validate 遇到其他版本會直接報錯（1.0 的 `n_tokens` 含呼叫、語意不同），請重跑 `scripts/convert_all.sh`，不做自動遷移 |
| `traj_id` | str | 全域唯一，例如 `tau2.airline.3.trial0.gpt-4.1-2025-04-14` |
| `benchmark` | str | `musique_ans` / `browsecomp_plus` / `tau2_bench` / `appworld` / `swebench_verified` / `agentrewardbench` |
| `domain` | str | `multihop_qa` / `tool_use` / `software_web` |
| `task_id` | str | 原始 benchmark 的題號 |
| `split` | str? | dev / test / τ² 的模式 |
| `solver` | object | `model`、`agent`、`origin`（official_release / self_generated）、`native_reasoning` |
| `label` | object | 見 §3.3 |
| `segments` | list | 見 §3.2，依原始順序 |
| `tokenizer` | str | 算 `n_tokens` 用的 tokenizer |
| `meta` | dict | 可公開的分析資訊（hop 數、步數…），**不會**放進 verifier 輸入。🆕 `incomplete`（生成未完成）、`native_reasoning_tokens`、`native_reasoning_declared`（原本宣稱的推理通道）。🆕² `call_order_changed`（工具回傳順序與呼叫順序不同的次數，見 §7.6） |
| `hidden` | dict | gold 答案、評分細節，**絕不**放進 verifier 輸入 |

### 3.2 Segment

| 欄位 | 說明 |
|---|---|
| `seg_id` | 0..n-1，依原始順序 |
| `role_default` | adapter 用規則判斷的角色（Q/E/R/A） |
| `role_override` | Kai 的分類器或人工稽核覆寫的角色；**實際角色一律讀 `segment.role`**（override 優先） |
| `role_source` | `rule` / `classifier` / `manual`（各欄位來源中優先權最高的） |
| 🆕² `override_sources` | 逐欄位記錄 `role` / `spans` / `call_role` 各自來自哪個來源，覆寫合併時用來比較優先權 |
| `speaker` | `user` / `solver` / `environment` / `system` |
| `kind` | 細分類型（見下表），用於錯誤分析 |
| `step` | 原始 trajectory 的回合編號 |
| 🆕 `source_id` | 對應原始資料的位置，例如 `msg:12/tool_result`、`step:3/raw_output/think`，稽核時用來對照原始檔 |
| `text` | 內容 |
| `call` | 產生這段觀測或動作的呼叫：`name`、`arguments`、`requestor`、`tool_type` |
| 🆕 `call_role_override` | 呼叫參數的角色（預設跟 segment 相同）。判為 R 且條件不含 R 時，verifier 只看到工具名稱 |
| 🆕 `spans` | 一段含多種角色時，切成 `[start, end)` 區間各自標角色；必須依序、連續、覆蓋整段 |
| `n_tokens` | 🆕 只算 `text` 的 token 數（v1.0 包含呼叫，已改） |
| 🆕 `call_n_tokens` | 呼叫的 token 數，算的是 verifier 實際看到的呼叫文字（`ToolCall.render()`） |
| `ambiguous` | 規則不確定，需要稽核 |
| `note` | 為什麼這樣分類 |
| `hidden` | 這個 segment 的 gold 資訊（例如 `is_supporting`），不給 verifier |

`kind` 一覽：

| 角色 | kind |
|---|---|
| Q | `question` `goal` `issue` `instruction` `policy` `user_utterance` |
| E | `passage` `tool_result` `search_result` `execution_output` `observation` `user_observation` |
| R | `reasoning` `narration` `intermediate_output` |
| A | `final_answer` `agent_reply` `action` `patch` `state_diff` |

### 3.3 Label

| 欄位 | 說明 |
|---|---|
| `correct` | `true` / `false` / `null`。🆕 null 的情況：尚未評分、標註者沒有過半數共識、標籤型別異常（例如 SWE 的 resolved 不是布林值）、judge 判決與答案對不上。抽樣一律排除 |
| `source` | `exact_match` / `llm_judge` / `unit_tests` / `env_reward` / `human_annotation` / `pending` |
| `score` | 原始分數（EM、reward…） |
| `detail` | 其他評分資訊（F1、`needs_review`、標註者票數、judge 設定…） |

`correct` 統一成布林值，pipeline 不需要知道每個 benchmark 怎麼評分就能算 EDR / FRR / BA。

### 3.4 `native_reasoning`

| 值 | 意義 | 例子 |
|---|---|---|
| `full` | 推理完整可見 | Qwen3 thinking 的 `reasoning_content`、AgentLab GenericAgent 的 `<think>` |
| `summary` | 只有推理摘要 | OpenAI Responses API 的 reasoning summary（BrowseComp-Plus 的 gpt5 / o3） |
| `hidden` | 推理模型，但思考被隱藏或加密，R 只剩旁白 | o4-mini、Gemini 3.5 Flash（`thought_signature` 是加密的） |
| `none` | 這條 trajectory 實際上沒有原生推理 | GPT-4.1、τ² 的 Claude 3.7、🆕 AgentRewardBench 的 Claude 3.7（只輸出 `<action>`） |

🆕 這個欄位由 `builder` 依「實際有沒有 kind=reasoning 的 R」判定：宣稱 full / summary 卻沒有推理內容 → 改成 none；宣稱 none / hidden 卻有 → 改成 full。原本的宣稱值保留在 `meta.native_reasoning_declared`。🆕² 套用覆寫後會依新的角色重新計算（例如唯一的 R 被改成 E，就會變成 none），判斷基準仍是最初的宣稱值。所有 adapter 取推理的規則統一在 `qera/reasoning.py`（支援 `reasoning_content`、`reasoning`、`thinking_blocks`、`<think>`，包含沒收尾的 `<think>`）。

## 4. 各 benchmark 的 Q/E/R/A 規則

### 4.1 MuSiQue-Ans

| 角色 | 內容 |
|---|---|
| E | 給 solver 的 20 段 paragraph，**每段一個 segment**（Jev 以段落為單位挑選） |
| Q | question（🆕 放在段落之後，和 solver 看到的 prompt 順序一致） |
| R | 推理欄位、`<think>`、"Answer:" 之前的文字 |
| A | "Answer:" 之後的文字；🆕 生成被截斷時沒有答案，標 `meta.incomplete` |
| 標籤 | EM（對 answer 與所有 alias 取最大）。🆕 EM=0 但 F1>0 或答案互相包含 → `needs_review` |

- 🆕 EM 只比字串，同義不同寫法會被判錯；`needs_review` 的樣本建議人工抽查，並在論文說明標籤的限制。

- `is_supporting` 只放在 `segment.hidden`，符合 proposal「E 不含 gold supporting-passage 標註」。但保留下來可以**評估 Jev 有沒有挑到真正的支持段落**（evidence recall），這是 Jev 分析的額外指標。
- `meta.hop_count` 用於 proposal 的 hop 數分層分析。
- 沒有官方 trajectory，要自己跑 solver：`--export-prompts` 會輸出統一的 solver prompt。

### 4.2 BrowseComp-Plus

| 角色 | 內容 |
|---|---|
| Q | query |
| E | 每次 `search` / `get_document` 的呼叫與回傳 |
| R | reasoning summary、非最後一次的文字輸出 |
| A | 最後一次文字輸出 |
| 標籤 | LLM judge（官方 prompt 與官方設定，`scripts/judge_bcp.py`）；沒跑 judge 前為 `pending` |

- 🆕 judge 設定完全照官方 `evaluate_run.py`：Qwen3-32B、**關閉 thinking**、temperature 0.7、top_p 0.8、top_k 20、最多 4096 tokens。
- 🆕 判決解析只接受「恰好一個」獨立的 `correct: yes/no` 行；找不到或互相矛盾 → `parse_error`，重跑時自動重判。
- 🆕 每個判決綁定 `(run_tag, query_id, candidate_hash)`。同一題在 4 個 run 的答案不同，只用題號會把判決貼錯；載入時重算答案 hash，對不上就不採用。
- 檢索到的 docid 與官方 qrels 比對，結果記在 `hidden.evidence_hits` / `gold_hits`，同樣可評估 Jev。
- ⚠ BrowseComp 要求資料不得以明文出現在網路上，**轉換後的檔案不要推到公開 repo**。

### 4.3 τ²-Bench

| 角色 | 內容 |
|---|---|
| Q | domain policy、所有使用者說的話 |
| E | assistant 的 READ / GENERIC 工具呼叫與回傳；WRITE 工具的回傳；使用者端裝置的工具呼叫與回傳（telecom） |
| R | 伴隨工具呼叫的旁白 |
| A | WRITE 工具呼叫、`transfer_to_human_agents`、所有對使用者說的話（中途回覆標 ambiguous） |
| 標籤 | `reward_info.reward == 1`（官方 pass^1） |

做這些決定的理由：
- **policy 算 Q**：它是 agent 必須遵守的「需求」，verifier 不看 policy 就無法判斷動作是否合規。
- **使用者發言預設算 Q**：多輪對話的需求是逐步揭露的（例如之後才說「我其實是 Gold 會員」）。🆕 但外部審查指出，有些發言其實是觀測（例如「重開機後還是沒訊號」），整句算 Q 會讓 QA 條件偷偷保留證據。格式已支援用 `spans` 把一句話切成 Q 和 E；**實際怎麼切由 Kai 的分類器或人工稽核決定**，切太多也會讓 QA 幾乎沒有資訊，這個取捨要在組上討論。
- **工具類型用官方標記**：τ² 原始碼用 `@is_tool(ToolType.READ/WRITE/GENERIC)` 標了每個工具，已整理成 `qera/resources/tau2_tool_types.json`。WRITE 會改變資料庫，等於「產出物」，所以是 A；READ 只是取得資訊，所以呼叫加回傳是 E。
- **`transfer_to_human_agents` 是例外**：官方標為 GENERIC，但它是終止對話的動作，會影響評分，所以歸 A。
- **對使用者的回覆是 A（從真實資料發現的）**：τ² 的評分包含 `communicate_checks`，也就是 agent 有沒有把特定資訊告訴使用者。實際資料中，這個資訊常出現在**中途**的回覆（例如 airline 第 3 題，答案「4 件行李」在第 8 回合就說了，最後一則回覆在講別的事）。如果中途回覆算 R，QEA 條件下 verifier 根本看不到被評分的內容。所以全部歸 A，中途的標 `ambiguous`，因為裡面也夾帶詮釋。
- 開場白 "Hi! How can I help you today?" 是框架寫死的，不是 solver 產生的，所以略過。
- user simulator 的控制符號（`###STOP###`）會被移除。
- 模擬使用者的劇本（`user_scenario`）和評分標準絕不放進 Q。

### 4.4 AppWorld

| 角色 | 內容 |
|---|---|
| Q | task instruction（含 supervisor 姓名，因為任務常以「我」指稱 supervisor） |
| E | 每次程式執行的輸出，程式碼當作呼叫標頭 |
| R | LM 回覆中程式碼以外的文字、`reasoning_content` |
| A | `complete_task(answer=...)` 的答案，加上 `dbs/` 的資料庫變更摘要 |
| 標籤 | `evaluations/<dataset>.json` 的 `success`（state-based 單元測試） |

- AppWorld 用資料庫最終狀態評分，所以真正的「產出物」是狀態變更，A 一定要包含它。
- 程式碼註解可能夾帶推理，含 `#` 的程式碼標 ambiguous。
- ⚠ 這個 adapter 是依 AppWorld 原始碼推導的格式寫成，**還沒用官方實驗輸出實測過**。

### 4.5 SWE-bench Verified（mini-swe-agent 格式）

| 角色 | 內容 |
|---|---|
| Q | `<pr_description>` 裡的 issue（agent 的操作說明模板不算，verifier 用不到） |
| E | 每個 bash 指令 + 執行輸出 |
| R | assistant 的文字、`reasoning_content`、`<think>` |
| A | 最後提交的 patch |
| 標籤 | `per_instance_details.json` 的 `resolved`（測試是否通過） |

- 提交那一步（`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`）的輸出就是 patch 本身，放進 E 會讓 A 重複出現，所以略過。
- 🆕 標籤嚴格檢查型別：`resolved` 必須是布林值，缺值、null、字串一律設為 `null`（之前會被轉成 false，等於把「沒有標籤」當成「答錯」）。
- 🆕 舊版 mini-swe-agent 把指令寫在 ```` ```bash ```` 區塊裡；指令已經以呼叫形式出現在對應的 E 上，旁白（R）裡不再重複一次。

### 4.6 AgentRewardBench

| 角色 | 內容 |
|---|---|
| Q | goal |
| E | 每一步的觀測：URL、上一動作的錯誤訊息、pruned accessibility tree |
| R | 🆕 每一步模型**原始輸出**（`chat_messages` 最後一則 assistant）裡的 `<think>` |
| A | 每一步的動作；最後一個動作（通常是 `send_msg_to_user`）標 `final_answer` |
| 標籤 | 專家標註 `trajectory_success`；🆕 多位標註者要**過半數**才採用，否則為 `null` |

- 🆕 **不再直接用 `reasoning` 欄位**：模型沒寫 `<think>` 時，AgentLab 會把 `<action>…</action>` 放進這個欄位。實際檢查：**Claude 3.7 Sonnet 的 trajectory 全部只有 `<action>`、沒有推理**（prompt 有要求寫 `<think>`，但模型沒照做），其他三個模型 100% 有原生推理。Claude 的 trajectory 現在標 `native_reasoning=none`。
- 🆕 **預設排除 VisualWebArena**：任務常需要看圖，verifier 只吃文字，失敗可能是「資訊沒給」而不是「E 沒用」。
- 🆕 題號的 `.resized` / `.improved` 變體標記會先拿掉再查 split，之前有 113 條因此查不到 split。
- `chat_messages` 是每一步送給 LLM 的完整 prompt，不是使用者的後續對話；這幾個網頁任務沒有後續使用者訊息。

- 網頁任務的產出是「這串動作造成的狀態改變 + 最後回覆」，所以動作屬於 A。這也和 AgentRewardBench 論文裡的 judge 輸入一致（action history + final state + response）。
- 截圖路徑保留在 `hidden.screenshot_path`；目前 verifier 只吃文字。

## 5. 實驗條件與 verifier 輸入

```python
from qera.render import build_verifier_input, selector_view

vi = build_verifier_input(traj, "QEA")
vi.text              # 給 verifier 的內容（不含 verifier 自己的指令）
vi.included_seg_ids  # 放了哪些 segment
vi.n_tokens          # 🆕 實際渲染文字的 token 數（含標頭、截斷後）
vi.n_tokens_content  # 🆕 被納入內容的 token 數（未截斷、不含標頭）：預算比較用這個
vi.tokens_by_role    # {"Q":..., "E":..., "R":..., "A":...}（content 的拆分）
vi.truncated         # 🆕 有沒有被 max_chars_per_segment 截斷

# Jev / baseline：只能用 selector_view，裡面沒有 hidden、label、meta
view = selector_view(traj)          # {"question", "answer", "candidates": [{seg_id, text, call, n_tokens, ...}]}
chosen = my_selector(view, budget)  # 選擇器自己的邏輯
vi = build_verifier_input(traj, "QEJevA", selected_e=chosen, budget=budget)  # 🆕 會檢查 ID 與預算
```

🆕 過濾的單位是「角色」：有 spans 的 segment 只顯示條件允許的區間；呼叫參數判為 R 而條件不含 R 時只顯示工具名稱。

🆕² `selected_e` 和 `budget` **只能**用在 `QEsA` / `QEJevA`。在 QERA / QA / QEA 傳入（即使是空清單）會直接報錯：之前會默默把 E 過濾掉，QEA 實際上變成沒有 E，而且看不出來。

🆕² 「這條 trajectory 有沒有 E」在所有地方用同一個判斷（`Segment.has_evidence()`：正文含 E，或呼叫角色是 E）。`e_candidates`、`sample`、`validate`、`stats` 都用它，不會再出現「選擇器有候選、抽樣卻排除」的情況。注意 `Trajectory.by_role("E")` 只看正文，不含「只有呼叫是 E」的 segment，行為未改變。

⚠ `vi.n_tokens` 只是 verifier 輸入的「內容」部分；完整 prompt（含 verifier 指令、chat template）的 token 數、verifier 輸出 token 數、延遲與成本，要在 pipeline 用實際 API 回傳的 usage 計算（子豪）。

| 條件 | 內容 |
|---|---|
| `QERA` | 全部 |
| `QA` | Q + A |
| `QEA` | 拿掉 R |
| `QEJevA`（= `QEsA`） | Q + A + 被選中的 E；`selected_e` 傳入 Jev 或 baseline 選出的 `seg_id` |

**Jev 與三個 baseline（random / recency / relevance）共用同一個 `QEsA` 渲染器**，差別只在 `selected_e`。共用渲染器只保證「呈現方式相同」；預算是否相同由 `budget` 參數檢查，選取紀錄（方法、seed、預算、實際用量、選中 ID、耗時）要由實驗層保存。

## 6. 給各組員的接口

🆕 **分工邊界**：本接口負責「資料正確、格式一致、不洩漏、失敗會報錯」。下列屬於其他人的範圍，接口只提供支援：

| 事項 | 負責 | 接口提供的支援 |
|---|---|---|
| E/R 該怎麼切（含使用者混合發言、呼叫參數是否算推理） | Kai | `spans`、`call_role` 覆寫與驗證 |
| Jev 與 baseline 的預算如何訂、選取紀錄 | Kai／全組 | `selector_view`、`budget` 檢查、ID 驗證 |
| 完整 prompt token、verifier 輸出 token、延遲、成本、四條件配對結果 | 子豪 | `n_tokens`、`n_tokens_content`、`tokens_by_role` |
| solver / judge 用哪個模型與部署設定 | Magnus | `run_solver.py`、`judge_bcp.py`、`carc_solver.slurm` 範本 |
| tuning 與 held-out 怎麼切 | 全組 | `sample --split`、`--stratify-by`、manifest |

### 子豪（pipeline）
- 讀檔：`qera.cli.read_jsonl(paths)` → `Trajectory` 物件。
- 組輸入：`build_verifier_input(traj, condition, selected_e=...)`。
- 算指標：`traj.label.correct` 是布林值，`None` 的要排除。
- 🆕 每個平衡檔旁邊有 `.manifest.json`，記錄輸入、參數、排除原因與程式版本；報告結果時請一併保存。
- 長度（Q+E+R+A，不含 verifier 指令；2026-10-25 的平衡樣本）：τ² p50 7.3k / p90 9.9k / 最長 14.3k；SWE-bench 14.8k / 29.8k / 51.3k；**AgentRewardBench 24.7k / 101.7k / 175.5k**；BrowseComp-Plus 約 75k / 106k / 182k。請 Magnus 確認 vLLM 上 verifier 的 `max_model_len`；ARB 與 BrowseComp-Plus 有相當比例會超過 32k，可用 `qera.cli sample --max-total-tokens` 排除過長的，或用 `max_chars_per_segment` 截斷單段（會記在 `vi.truncated`）。cost / latency 指標要分 benchmark 報告。

### Kai（Jev、E/R 分類）
- 🆕 選擇器的輸入**一律用 `selector_view(traj)`**：只含 Q、A 文字與 E 候選（每段有 `n_tokens` = 選了會花掉的預算），不含標準答案。舊的 `e_candidates` 曾回傳帶 `hidden` 的完整 segment，已改成乾淨的 dict。
- 預算：SWE-bench 的 E 中位數約 13.6k tokens、τ² 約 2k、BrowseComp-Plus 約 72k，同一個固定數字在各 benchmark 意義不同。用固定值或比例都可以，**重點是在跑實驗前寫進實驗設定**，由 Kai 和全組決定。
- 分類結果寫成覆寫檔（JSONL），用 `qera.cli apply-overrides` 寫回，不用改原始資料。🆕 三種寫法：
  - `{"traj_id", "seg_id", "role": "E", "source": "classifier"}`：整段改角色
  - `{"traj_id", "seg_id", "call_role": "R", "source": "classifier"}`：只改呼叫參數的角色
  - `{"traj_id", "seg_id", "spans": [[0, 40, "Q"], [40, 95, "E"]], "source": "classifier"}`：一段切成多種角色
  - 覆寫對不到資料、切分不合法、改完沒有 Q 或 A，都會報錯；🆕² 檢查全部通過才會寫出檔案，失敗時原檔不變（輸入輸出可以是同一個檔案）。
  - 🆕² **合併規則（逐欄位）**：`role`、`spans`、`call_role` 各自取優先權最高的來源（manual > classifier；同優先權時，後讀到的檔案優先）。例如 classifier 給的 `call_role=R` 和人工給的 `role=E` 會同時生效，不會互相蓋掉。
  - 🆕² `role` 和 `spans` 互斥：都是在描述「正文的角色」，設定其中一個會清掉另一個；來自不同來源時取優先權高的。
- 🆕² `selector_view` 的 `answer` 依 trajectory 原始順序收集 A（同一段內呼叫在前、文字在後），和 verifier 看到的時間線一致。
- `segment.hidden` 裡的 `is_supporting`（MuSiQue）和 `evidence_hits`（BrowseComp-Plus）可以**事後**用來評估 Jev 選得好不好，**但 Jev 本身不能讀**（proposal：不使用標籤）。

### Magnus（模型）
- 跑 solver：`scripts/run_solver.py`（或 `sbatch scripts/carc_solver.slurm`）。輸出是 OpenAI chat 格式並保留推理內容。🆕 每筆記錄設定指紋；換了模型或參數卻寫進同一個檔案會被拒絕；被截斷的題目可用 `--retry-incomplete` 搭配較大的 `--max-tokens` 重跑。
  - 🆕² 有題目在重試後仍失敗時，以非 0 的 exit code 結束（已成功的紀錄保留），重跑同一行指令即可補上；Slurm 腳本因此不會在有缺漏時繼續轉換。
  - 🆕² 用 `--ids`、`--sample`、`--limit` 只跑部分題目時，重跑與刪除只會作用在本次選中的題目，其他紀錄不會被動到。
- 🆕² judge（`scripts/judge_bcp.py`）續跑前會先把輸出檔修復成「只有完整的行」，上次中斷留下的半行會移除並重判。
- MuSiQue 的 solver prompt 已產生：`runs/musique_dev_prompts.jsonl`。
- **solver 請用推理可見的模型**（例如 Qwen3 的 thinking mode），理由見 §7.1。vLLM 請加對應的 `--reasoning-parser`。
- 🆕 正式實驗的 token 數要用 verifier 實際的 tokenizer：`export QERA_TOKENIZER=<模型> QERA_TOKENIZER_REVISION=<版本>` 後跑 `qera.cli retokenize`；指定的 tokenizer 載入失敗會直接報錯，不會偷偷改用近似值。

## 7. 已知限制與待討論

### 7.1 ⚠ 官方 trajectory 的 R 幾乎是空的（最重要）
| 資料 | native_reasoning | R 的 token 中位數 | R = 0 的比例 |
|---|---|---|---|
| τ² GPT-4.1 | none | **0** | — |
| τ² o4-mini | hidden | **0** | — |
| τ² Claude 3.7 | none | 157 | — |
| τ² 三個模型合計 | | 0 | **59%** |
| SWE-bench Gemini 3.5 Flash | hidden | 973（全是旁白） | 12% |
| 🆕 AgentRewardBench Claude 3.7 | none | 0 | **100%**（只輸出 `<action>`） |
| 🆕 AgentRewardBench 其他三個模型 | full | — | 0% |

GPT-4.1 和 o4-mini 呼叫工具時通常不說話，o4-mini 和 Gemini 的思考被 API 隱藏或加密。結果是 **QERA 和 QEA 在這些資料上幾乎一樣**，研究問題 (1)「拿掉 R 會不會失去資訊」測不出來。研究問題 (1) 的主要證據應該來自 `native_reasoning=full` 的 trajectory，其他的分層報告（`qera.cli sample --stratify-by solver.native_reasoning`）。

建議：τ² 和 MuSiQue 用**推理可見的 solver 自己生成 trajectory**（例如 Qwen3 thinking mode）。τ² 官方 CLI `tau2 run --agent-llm ...` 透過 LiteLLM 可以接任何模型（包括 vLLM 上的千問），而且會把完整回應存在 `raw_data`；τ² adapter 已經會從 `raw_data` 抽出 `reasoning_content` 放進 R（有測試），自己跑出來的結果直接轉換即可。SWE-bench 可以改用推理可見的提交（例如 S3 上的 DeepSeek-V3.2-Reasoner，需要先確認它的 traj 有沒有存 reasoning_content）。官方 trajectory 仍然很適合先拿來把 pipeline 跑通。

### 7.2 不同 benchmark 的 E 規模差很多
SWE-bench 的 E 中位數約 13k tokens（44 段），τ² 約 2k（8–11 段），AgentRewardBench 約 16k–30k（13–31 段），BrowseComp-Plus 約 72k，MuSiQue 固定 20 段。Jev 的預算設計要考慮這點（見 §6）。

### 7.3 ambiguous 的 segment
τ² 約 20% 的 segment 標了 ambiguous（中途回覆）。這是 E/R 稽核的重點；稽核表在 `audit/`。

### 7.4 adapter 的實測狀態
🆕 τ²、SWE-bench、AgentRewardBench、BrowseComp-Plus、MuSiQue（資料部分）都已在真實資料上轉換與驗證。**AppWorld 尚未**：官方實驗輸出沒有 API 回傳與 LM 對話，不能用於本研究；adapter 針對自己跑 agent 產生的 `lm_calls.jsonl`，還有幾個已知問題（答案可能抓到變數名稱、資料庫變更摘要會截斷、只用最後一次 LM 呼叫重建歷史），等組上決定要不要自己跑 AppWorld 再修。

### 7.5 🆕 抽樣
- 先篩選（無標籤、無 E、未完成、過長、split）→ 再以 (task, solver) 去重（**在看標籤之前**隨機選一條）→ 最後平衡。
- 平衡樣本的正確率固定 50%，不代表自然分布。
- 用同一份資料調 Jev 又報告結果會造成評估污染；AgentRewardBench 有官方 dev / test，可用 `--split test` 只取 test。tuning / held-out 的切法請全組決定。
- 每個平衡檔旁的 `.manifest.json` 記錄排除原因、各 solver / split 的數量、seed 與程式版本。
- 🆕² 「沒有 E」的排除條件和選擇器使用同一個判斷（見 §5）。

### 7.6 🆕² 工具回傳順序（只偵測、不修正）
- 格式把工具呼叫附在它的回傳上一起顯示。如果原始紀錄是「呼叫 X、呼叫 Y、回傳 Y、回傳 X」，渲染後 Y 的呼叫會出現在 X 之前。
- 目前只**偵測並標記**：相關 segment 標 `ambiguous` 並寫明原因，trajectory 的 `meta.call_order_changed` 記錄次數，`stats` 有一欄統計受影響的 trajectory 數。
- 真實資料檢查（SWE-bench 120 條、τ² 3,336 條）：0 條受影響。
- ⚠ 已知限制：如果某個呼叫一直沒有收到回傳（例如格式錯誤提示），或同一則訊息有多個 inline 指令共用同一個 key，之後正常的回傳也可能被誤標。重新轉換後請看 `stats` 的 `call_order_changed` 欄；若不是 0，先抽查再決定要不要改格式（例如把呼叫拆成獨立的 segment，這會影響所有組員的接口，要先在組上討論）。
