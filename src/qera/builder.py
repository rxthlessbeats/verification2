"""Adapter 共用的小工具：依序累積 segment、自動編號、自動算 token、檢查 R 的來源。"""

from __future__ import annotations

import json
from typing import Any

from . import tokens
from .schema import Label, Role, Segment, SolverInfo, Speaker, ToolCall, Trajectory


def to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, indent=None)


class TrajectoryBuilder:
    def __init__(self) -> None:
        self.segments: list[Segment] = []
        # 生成沒有正常結束的原因（例如被截斷）。有任何一筆，trajectory 就標成 incomplete，
        # 抽樣時預設排除：「沒交出答案」和「交出錯的答案」是兩回事，不能混在錯誤類裡。
        self.incomplete: dict[str, str] = {}
        self._pending_call_order: list[str] = []
        self._call_segments: dict[str, list[Segment]] = {}
        self._reordered_calls: set[str] = set()
        self.call_order_changed = 0

    def record_call(self, key: str, segment: Segment | None = None) -> None:
        self._pending_call_order.append(key)
        self._call_segments[key] = [segment] if segment is not None else []

    def record_result(self, key: str, segment: Segment | None = None) -> None:
        if key not in self._pending_call_order:
            return
        if segment is not None:
            self._call_segments[key].append(segment)
        index = self._pending_call_order.index(key)
        if index:
            # 只標記被跨越的呼叫與回傳，不改原始順序或切分規則。
            self.call_order_changed += 1
            self._reordered_calls.update(self._pending_call_order[: index + 1])
        self._pending_call_order.remove(key)
        for affected in self._reordered_calls:
            for related in self._call_segments.get(affected, []):
                related.ambiguous = True
                note = "工具回傳順序與呼叫順序不同；保留原紀錄位置，需稽核順序影響"
                if note not in (related.note or ""):
                    related.note = (related.note + "；" if related.note else "") + note

    def mark_incomplete(self, reason: str, detail: str = "") -> None:
        self.incomplete[reason] = detail

    def add(
        self,
        role: Role | str,
        speaker: Speaker | str,
        kind: str,
        text: Any,
        *,
        step: int | None = None,
        source_id: str | None = None,
        call: ToolCall | dict | None = None,
        ambiguous: bool = False,
        note: str | None = None,
        hidden: dict | None = None,
        skip_empty: bool = True,
    ) -> Segment | None:
        text = to_text(text).strip()
        if skip_empty and not text and call is None:
            return None
        if isinstance(call, dict):
            call = ToolCall(**call)
        seg = Segment(
            seg_id=len(self.segments),
            role_default=Role(role),
            speaker=Speaker(speaker),
            kind=kind,  # type: ignore[arg-type]
            step=step,
            source_id=source_id,
            text=text,
            call=call,
            n_tokens=tokens.count(text),
            # 和 render 實際顯示的呼叫文字用同一個函式，確保 token 數一致
            call_n_tokens=tokens.count(call.render()) if call else None,
            ambiguous=ambiguous,
            note=note,
            hidden=hidden or {},
        )
        self.segments.append(seg)
        return seg

    def build(
        self,
        *,
        traj_id: str,
        benchmark: str,
        domain: str,
        task_id: str,
        solver: SolverInfo,
        label: Label,
        split: str | None = None,
        meta: dict | None = None,
        hidden: dict | None = None,
    ) -> Trajectory:
        meta = dict(meta or {})
        meta["call_order_changed"] = self.call_order_changed
        _check_reasoning_provenance(self.segments, solver, meta)
        if self.incomplete:
            meta["incomplete"] = True
            meta["incomplete_reasons"] = self.incomplete
        return Trajectory(
            traj_id=traj_id,
            benchmark=benchmark,
            domain=domain,  # type: ignore[arg-type]
            task_id=str(task_id),
            split=split,
            solver=solver,
            label=label,
            segments=self.segments,
            tokenizer=tokens.name(),
            meta=meta,
            hidden=hidden or {},
        )


def _check_reasoning_provenance(
    segments: list[Segment], solver: SolverInfo, meta: dict
) -> None:
    """依「這條 trajectory 實際有沒有原生推理內容」修正 native_reasoning。

    為什麼需要：之前 AgentRewardBench 依欄位名稱 reasoning 一律標成 full，
    但實際有 28% 的 trajectory 那個欄位裡只有 <action>…</action>，根本沒有推理。
    native_reasoning 是分析研究問題 (1) 時用來分層的依據，不能只看宣稱。

    規則：
    - 宣稱 full / summary，但沒有任何 kind=reasoning 的 R → 改成 none
    - 宣稱 none / hidden，但實際有 kind=reasoning 的 R（例如自己跑 τ² 時抽到 reasoning_content）→ 改成 full
    原本宣稱的值保留在 meta["native_reasoning_declared"]，並記錄原生推理的 token 數。
    """
    native = [
        (text, n)
        for s in segments
        if s.kind == "reasoning"
        for role, text, n in s.pieces()
        if role == Role.R and text.strip()
    ]
    # 重算時仍以最初宣稱為準，否則移除再恢復 R 會把 summary 誤升為 full。
    declared = meta.setdefault("native_reasoning_declared", solver.native_reasoning)
    actual = declared
    if declared in ("full", "summary") and not native:
        actual = "none"
    elif declared in ("none", "hidden") and native:
        actual = "full"
    solver.native_reasoning = actual  # type: ignore[assignment]
    meta["native_reasoning_tokens"] = sum(n for _, n in native)
