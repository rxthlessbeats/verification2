"""把 Trajectory 組成 verifier 的輸入，以及提供給 Jev / baseline 的候選清單。

四種實驗條件不需要存四份資料，只要依條件過濾內容、保留原始順序：

    QERA          : 全部
    QA            : 只留 Q、A
    QEA           : 拿掉 R
    QEJevA (QEsA) : 只留被選中的 E（Jev 或 random / recency / relevance baseline），拿掉 R

過濾的單位是「角色」而不是整個 segment：
- 有 spans 的 segment（一段裡同時有 Q 和 E），只保留條件允許的那幾段文字
- 工具呼叫有自己的角色（call_role）。呼叫參數被判為 R、而條件不含 R 時，
  只顯示工具名稱，不顯示參數（因為參數裡可能有 solver 的推理）

Jev 與三個 baseline 共用同一個渲染器，差異只來自「選了哪些 E」。
這裡負責的是「接口層」的保證：
- selector_view() 給選擇器的資料不含 hidden、label、meta（沒有標準答案可以偷看）
- build_verifier_input() 會檢查選出來的 seg_id 是否存在、是否真的含有 E
- 傳入 budget 時會檢查選出的 E 是否超過預算
預算怎麼訂、選取紀錄怎麼存，屬於實驗層（Jev / pipeline），不在這裡決定。

render 只讀 text / call / spans / kind / step，絕不讀 hidden 和 label。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from . import tokens
from .schema import Role, Segment, Trajectory

CONDITIONS = {
    "QERA": {Role.Q, Role.E, Role.R, Role.A},
    "QA": {Role.Q, Role.A},
    "QEA": {Role.Q, Role.E, Role.A},
    "QEsA": {Role.Q, Role.E, Role.A},  # E 只保留 selected_e
}
CONDITIONS["QEJevA"] = CONDITIONS["QEsA"]  # proposal 裡的名稱
SELECTIVE = {"QEsA", "QEJevA"}

_HEADERS = {
    Role.Q: "TASK",
    Role.E: "EVIDENCE",
    Role.R: "SOLVER REASONING",
    Role.A: "CANDIDATE ANSWER",
}


@dataclass
class VerifierInput:
    text: str
    included_seg_ids: list[int]
    n_tokens: (
        int  # 實際渲染出來的文字的 token 數（含標頭、截斷後），不含 verifier 自己的指令
    )
    n_tokens_content: int  # 被納入內容的 token 數（未截斷、不含標頭）：預算比較用這個
    tokens_by_role: dict[str, int] = field(
        default_factory=dict
    )  # n_tokens_content 依角色拆分
    truncated: bool = False


def _truncate(text: str, max_chars: int | None) -> tuple[str, bool]:
    if max_chars is None or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]", True


def _segment_parts(
    seg: Segment, keep: set[Role], e_allowed: bool
) -> tuple[list[tuple[Role, str, int]], str | None, int]:
    """回傳這個 segment 在此條件下要顯示的 (文字片段, 呼叫文字, 呼叫 token 數)。"""

    def ok(role: Role) -> bool:
        return role in keep and (role != Role.E or e_allowed)

    pieces = [(r, t, n) for r, t, n in seg.pieces() if ok(r) and t]
    call_txt, call_n = None, 0
    if seg.call is not None:
        if ok(seg.call_role):
            call_txt, call_n = seg.call.render(), seg.call_n_tokens or 0
        elif pieces:
            # 呼叫參數被排除，但它的回傳結果要顯示：只留工具名稱，讓回傳結果仍有出處
            call_txt, call_n = seg.call.render(include_args=False), 0
    return pieces, call_txt, call_n


def render_segment(
    seg: Segment,
    keep: set[Role] | None = None,
    max_chars: int | None = None,
    e_allowed: bool = True,
) -> tuple[str, bool]:
    keep = keep or set(Role)
    pieces, call_txt, _ = _segment_parts(seg, keep, e_allowed)
    roles = sorted({r.value for r, _, _ in pieces}) or [seg.call_role.value]
    head = (
        "["
        + " + ".join(_HEADERS[Role(r)] for r in roles)
        + f" | {seg.kind}"
        + (f" | step {seg.step}" if seg.step is not None else "")
        + "]"
    )
    body, truncated = _truncate("\n".join(t for _, t, _ in pieces), max_chars)
    lines = [head] + ([call_txt] if call_txt else []) + ([body] if body else [])
    return "\n".join(lines), truncated


def _validate_selection(
    traj: Trajectory, selected: set[int], budget: int | None
) -> None:
    by_id = {s.seg_id: s for s in traj.segments}
    unknown = sorted(i for i in selected if i not in by_id)
    if unknown:
        raise ValueError(
            f"{traj.traj_id}: selected_e 含有不存在的 seg_id {unknown[:10]}"
        )
    not_e = sorted(
        i
        for i in selected
        if Role.E not in by_id[i].text_roles() and by_id[i].call_role != Role.E
    )
    if not_e:
        raise ValueError(
            f"{traj.traj_id}: selected_e 含有不是 E 的 segment {not_e[:10]}"
        )
    if budget is not None:
        used = sum(c["n_tokens"] for c in e_candidates(traj) if c["seg_id"] in selected)
        if used > budget:
            raise ValueError(
                f"{traj.traj_id}: 選出的 E 共 {used} tokens，超過預算 {budget}"
            )


def build_verifier_input(
    traj: Trajectory,
    condition: str,
    selected_e: Iterable[int] | None = None,
    max_chars_per_segment: int | None = None,
    budget: int | None = None,
    count_rendered_tokens: bool = True,
) -> VerifierInput:
    if condition not in CONDITIONS:
        raise ValueError(f"未知條件 {condition}，可用：{list(CONDITIONS)}")
    if condition not in SELECTIVE:
        # 誤傳選取參數會讓完整證據條件悄悄變質，必須在渲染前明確拒絕。
        if selected_e is not None:
            raise ValueError("selected_e 只能用於 QEsA / QEJevA")
        if budget is not None:
            raise ValueError("budget 只能用於 QEsA / QEJevA")
    keep = CONDITIONS[condition]
    sel = set(selected_e) if selected_e is not None else None
    if condition in SELECTIVE:
        if sel is None:
            raise ValueError(
                f"{condition} 需要 selected_e（Jev 或 baseline 選出的 E seg_id）"
            )
        _validate_selection(traj, sel, budget)

    parts, ids, any_trunc = [], [], False
    by_role = {r.value: 0 for r in Role}
    for s in traj.segments:
        e_allowed = sel is None or s.seg_id in sel
        pieces, call_txt, call_n = _segment_parts(s, keep, e_allowed)
        if not pieces and call_txt is None:
            continue
        txt, trunc = render_segment(s, keep, max_chars_per_segment, e_allowed)
        any_trunc |= trunc
        parts.append(txt)
        ids.append(s.seg_id)
        for r, _, n in pieces:
            by_role[r.value] += n
        if call_n:
            by_role[s.call_role.value] += call_n

    text = "\n\n".join(parts)
    return VerifierInput(
        text=text,
        included_seg_ids=ids,
        n_tokens=tokens.count(text) if count_rendered_tokens else -1,
        n_tokens_content=sum(by_role.values()),
        tokens_by_role=by_role,
        truncated=any_trunc,
    )


# ════════════════════════════════════════════════════════════════════
# 給 Jev / baseline 的資料：只含選擇器可以看的東西
# ════════════════════════════════════════════════════════════════════


def e_candidates(traj: Trajectory) -> list[dict[str, Any]]:
    """可被選取的 E（依原始順序）。回傳的是乾淨的 dict，不含 hidden / label。

    ⚠ 舊版回傳完整的 Segment 物件，裡面帶有 hidden（例如 MuSiQue 的 is_supporting、
    BrowseComp-Plus 的 evidence_hits）。選擇器如果讀到，就等於偷看答案。現在一律用這個函式。
    n_tokens 是「選了這段會花掉的預算」：E 文字 + 呼叫（呼叫角色是 E 時）。
    """
    out = []
    for s in traj.segments:
        e_text = "\n".join(t for r, t, _ in s.pieces() if r == Role.E)
        e_tok = sum(n for r, _, n in s.pieces() if r == Role.E)
        call_is_e = s.call is not None and s.call_role == Role.E
        if not s.has_evidence():
            continue
        out.append(
            {
                "seg_id": s.seg_id,
                "kind": s.kind,
                "step": s.step,
                "call": s.call.render() if call_is_e else None,
                "text": e_text,
                "n_tokens": e_tok + ((s.call_n_tokens or 0) if call_is_e else 0),
            }
        )
    return out


def selector_view(traj: Trajectory) -> dict[str, Any]:
    """選擇器（Jev、random、recency、relevance）需要的全部資訊，而且只有這些。

    Jev 要判斷每段 E 對「支持或反駁 A」有多大用處，所以需要 Q 和 A 的文字；
    但不能看到標籤、標準答案、gold evidence 標註，也不需要 R。
    """
    q = [t for s in traj.segments for r, t, _ in s.pieces() if r == Role.Q and t]
    a = []
    # Jev 與 verifier 必須看到相同的動作時間線；同段呼叫在正文之前。
    for s in traj.segments:
        if s.call is not None and s.call_role == Role.A:
            a.append(s.call.render())
        a.extend(t for r, t, _ in s.pieces() if r == Role.A and t)
    return {
        "traj_id": traj.traj_id,
        "benchmark": traj.benchmark,
        "question": "\n\n".join(q),
        "answer": "\n\n".join(a),
        "candidates": e_candidates(traj),
    }
