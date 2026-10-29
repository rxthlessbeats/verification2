"""統一的 trajectory 資料格式（QERA schema）。

每個 benchmark 的 adapter 都輸出同一種 `Trajectory`，下游（pipeline、Jev、verifier）
只需要認得這一種格式。

設計重點：
- 以「segment」為最小單位，保留原始順序。Q/E/R/A 只是每個 segment 上的標籤，
  所以可以事後改標（Kai 的分類、人工稽核）而不用重新轉檔。
- `role_default` 是 adapter 用規則判斷的結果；`role_override` 留給 Kai / 人工稽核。
  實際使用的角色一律讀 `segment.role`（override 優先）。
- 一段文字可能同時含有兩種角色（例如使用者說「請取消訂單，畫面顯示錯誤 404」，
  前半是需求 Q、後半是觀測 E）。這種情況用 `spans` 把一段切成幾個區間，各自標角色。
  切分本身是 Kai（E/R 分類）或人工稽核的工作；schema 只負責「能表達」。
- 工具呼叫的參數（搜尋字串、bash 指令、程式碼）由 solver 產生，可能夾帶推理。
  所以呼叫有自己的角色 `call_role`，可以和回傳結果分開判斷（預設跟 segment 同角色）。
- 標準答案、gold evidence 等「verifier 不能看到」的資訊，只放在 `label` 和
  `hidden` 欄位；render.py 永遠不會把它們放進 verifier 輸入，
  給 Jev 用的候選清單（render.selector_view）也會把它們拿掉。

版本紀錄：
- 1.0：初版
- 1.1：新增 spans、call_role_override、source_id；R 的來源改由資料實際內容判定
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.1"


class Role(str, Enum):
    Q = "Q"  # 問題、需求、任務指示
    E = "E"  # 外部文件、工具輸出、環境觀測
    R = "R"  # solver 產生的推理、計畫、假設、詮釋
    A = "A"  # 最終候選答案或產出物


class Speaker(str, Enum):
    user = "user"  # 提出需求的一方（真人或 user simulator）
    solver = "solver"  # 被驗證的 agent / 模型
    environment = "environment"  # 工具、檢索器、程式執行環境、網頁
    system = "system"  # benchmark 本身提供的固定內容（例如 policy 文件）


# segment 的細分類型。用來做錯誤分析，也讓 Kai 的分類器有更細的特徵可用。
Kind = Literal[
    # Q
    "question",  # 單一問題（MuSiQue、BrowseComp-Plus）
    "goal",  # 網頁 agent 的任務目標
    "issue",  # SWE-bench 的 issue 描述
    "instruction",  # AppWorld 的任務指示
    "policy",  # τ² 的 domain policy（屬於「需求」）
    "user_utterance",  # 使用者在多輪對話中說的話
    # E
    "passage",  # 給 solver 的文件段落
    "tool_result",  # 工具呼叫 + 回傳（呼叫內容作為來源標頭）
    "search_result",  # 檢索工具的回傳
    "execution_output",  # 程式 / 指令的執行結果
    "observation",  # 網頁觀測（URL、accessibility tree、錯誤訊息）
    "user_observation",  # 使用者端裝置的工具結果（τ² telecom）
    # R
    "reasoning",  # 原生推理內容（reasoning_content、think、reasoning summary）
    "narration",  # 伴隨動作的簡短說明（「我先查一下訂位紀錄」）
    "intermediate_output",  # 不是最後一次的文字輸出
    # A
    "final_answer",  # 最終答案文字
    "agent_reply",  # 對話中途對使用者說的話（τ²）：已告知使用者的資訊也是產出的一部分
    "action",  # 改變狀態的動作（寫入型工具、網頁操作）
    "patch",  # SWE-bench 的 diff
    "state_diff",  # AppWorld 的資料庫變更
]


class ToolCall(BaseModel):
    """產生這段觀測（或這個動作）的呼叫。"""

    name: str
    arguments: Any = None
    requestor: str | None = None  # "assistant" 或 "user"（τ² telecom）
    tool_type: str | None = None  # READ / WRITE / GENERIC（若 benchmark 有提供）

    def render(self, include_args: bool = True) -> str:
        """呼叫在 verifier 輸入中的呈現方式。

        token 計數（builder）、重算 token（retokenize）、渲染（render）都用這一個函式，
        確保「算的」和「實際給 verifier 看的」是同一段文字。
        include_args=False 時只留工具名稱：當呼叫參數被判定為 R（solver 的推理）、
        而目前的實驗條件不含 R 時使用。
        """
        who = f"{self.requestor} " if self.requestor else ""
        if not include_args:
            return f"{who}call: {self.name}(…)"
        args = self.arguments
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        return f"{who}call: {self.name}({args})"


class Span(BaseModel):
    """segment 內的一個區間 text[start:end]，帶自己的角色。"""

    start: int
    end: int
    role: Role
    n_tokens: int | None = None


class Segment(BaseModel):
    seg_id: int
    role_default: Role
    role_override: Role | None = None
    role_source: Literal["rule", "classifier", "manual"] = "rule"
    override_sources: dict[str, Literal["rule", "classifier", "manual"]] = Field(
        default_factory=dict
    )
    # 逐欄位保存優先權，避免人工改正文後阻擋獨立的呼叫角色覆寫。
    speaker: Speaker
    kind: Kind
    step: int | None = None  # 原始 trajectory 中的回合 / step 編號
    source_id: str | None = (
        None  # 對應到原始資料的位置，例如 "msg:12" "msg:12/tool:0" "step:3/think"
    )
    text: str
    call: ToolCall | None = None
    call_role_override: Role | None = None  # 呼叫參數的角色；None = 跟 segment 同角色
    spans: list[Span] | None = None  # 一段內含多種角色時使用；None = 整段同一角色
    n_tokens: int | None = None  # text 的 token 數（不含呼叫）
    call_n_tokens: int | None = None  # 呼叫（ToolCall.render()）的 token 數
    ambiguous: bool = False  # 規則不確定時標 True，優先進人工稽核
    note: str | None = None  # 為什麼這樣分類（給稽核者看）
    hidden: dict[str, Any] = Field(default_factory=dict)  # 永遠不會給 verifier / Jev 看

    @property
    def role(self) -> Role:
        return self.role_override or self.role_default

    @property
    def call_role(self) -> Role:
        return self.call_role_override or self.role

    def text_roles(self) -> set[Role]:
        """這段文字裡出現的所有角色（有 spans 時可能不只一種）。"""
        return {sp.role for sp in self.spans} if self.spans else {self.role}

    def has_evidence(self) -> bool:
        # 正文與呼叫可獨立改標；各入口必須共用此判斷，否則同一筆會被抽樣誤排除。
        return any(
            role == Role.E and bool(text) for role, text, _ in self.pieces()
        ) or (self.call is not None and self.call_role == Role.E)

    def pieces(self) -> list[tuple[Role, str, int]]:
        """把文字依角色切成 (角色, 文字, token 數)。沒有 spans 時就是整段。"""
        if not self.spans:
            return [(self.role, self.text, self.n_tokens or 0)]
        return [
            (sp.role, self.text[sp.start : sp.end], sp.n_tokens or 0)
            for sp in self.spans
        ]

    @model_validator(mode="after")
    def _check_spans(self) -> Segment:
        if self.spans:
            pos = 0
            for sp in self.spans:
                if sp.start != pos or sp.end <= sp.start:
                    raise ValueError(
                        f"seg {self.seg_id}: spans 必須依序、不重疊、連續覆蓋整段文字"
                    )
                pos = sp.end
            if pos != len(self.text):
                raise ValueError(
                    f"seg {self.seg_id}: spans 沒有覆蓋到文字結尾（{pos} / {len(self.text)}）"
                )
        return self


class Label(BaseModel):
    """A 是否正確。correct=None 代表沒有可用的標籤（尚未評分、標註者意見不一、型別異常）。"""

    correct: bool | None
    source: Literal[
        "exact_match",
        "llm_judge",
        "unit_tests",
        "env_reward",
        "human_annotation",
        "pending",
    ]
    score: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class SolverInfo(BaseModel):
    model: str
    agent: str | None = None  # agent 框架，例如 mini-swe-agent、GenericAgent
    origin: Literal["official_release", "self_generated"] = "self_generated"
    # solver 的推理通道：決定 R 有多少內容，分析時要分層
    #   full    = 推理完整可見（reasoning_content、<think>、AgentLab 的 think）
    #   summary = 只有推理摘要（OpenAI reasoning summary）
    #   hidden  = 推理模型，但思考被隱藏或加密（o4-mini、Gemini thought_signature）→ R 只剩旁白
    #   none    = 這條 trajectory 實際上沒有推理內容
    # ⚠ 這個欄位是依「這條 trajectory 實際有沒有推理內容」判定的（builder.build 會檢查），
    #   不是依模型名稱或欄位名稱宣稱的。宣稱值記在 trajectory.meta["native_reasoning_declared"]。
    native_reasoning: Literal["full", "summary", "hidden", "none"] = "none"


class Trajectory(BaseModel):
    schema_version: str = SCHEMA_VERSION
    traj_id: str
    benchmark: str
    domain: Literal["multihop_qa", "tool_use", "software_web"]
    task_id: str
    split: str | None = None
    solver: SolverInfo
    label: Label
    segments: list[Segment]
    tokenizer: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)  # 可公開的分析用資訊（hop 數等）
    hidden: dict[str, Any] = Field(default_factory=dict)  # gold 答案等，永不渲染

    @model_validator(mode="after")
    def _check(self) -> Trajectory:
        ids = [s.seg_id for s in self.segments]
        if ids != list(range(len(ids))):
            raise ValueError(
                f"{self.traj_id}: seg_id 必須是 0..n-1 連號且依原始順序排列"
            )
        roles = (
            set().union(*(s.text_roles() for s in self.segments))
            if self.segments
            else set()
        )
        if Role.Q not in roles:
            raise ValueError(f"{self.traj_id}: 缺少 Q")
        if Role.A not in roles:
            raise ValueError(f"{self.traj_id}: 缺少 A")
        return self

    def revalidate(self) -> Trajectory:
        """改過 role / spans 之後重新驗證（pydantic 不會自動重跑 validator）。"""
        return Trajectory.model_validate(self.model_dump())

    # ---- 便利函式 ----
    def by_role(self, role: Role | str) -> list[Segment]:
        """文字中含有這個角色的 segment（有 spans 時，只要其中一段是這個角色就算）。"""
        role = Role(role)
        return [s for s in self.segments if role in s.text_roles()]

    def tokens_by_role(self) -> dict[str, int]:
        """各角色的 token 數。呼叫的 token 算在 call_role 底下。"""
        out = {r.value: 0 for r in Role}
        for s in self.segments:
            for role, _, n in s.pieces():
                out[role.value] += n
            if s.call is not None:
                out[s.call_role.value] += s.call_n_tokens or 0
        return out
