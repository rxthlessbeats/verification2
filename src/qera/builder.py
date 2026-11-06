"""Build ordered segments and track incomplete output and reasoning provenance."""

from __future__ import annotations

import json
from typing import Any

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
            self.call_order_changed += 1
            self._reordered_calls.update(self._pending_call_order[: index + 1])
        self._pending_call_order.remove(key)
        for affected in self._reordered_calls:
            for related in self._call_segments.get(affected, []):
                related.ambiguous = True
                note = (
                    "Tool results arrived out of call order; original order preserved."
                )
                if note not in (related.note or ""):
                    related.note = (related.note + "; " if related.note else "") + note

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
            meta=meta,
            hidden=hidden or {},
        )


def _check_reasoning_provenance(
    segments: list[Segment], solver: SolverInfo, meta: dict
) -> None:
    """Derive reasoning availability from recorded R, preserving the original declaration."""
    native = any(
        role == Role.R and text.strip()
        for s in segments
        if s.kind == "reasoning"
        for role, text in s.pieces()
    )
    # Preserve summaries when labels are removed and restored.
    declared = meta.setdefault("native_reasoning_declared", solver.native_reasoning)
    actual = declared
    if declared in ("full", "summary") and not native:
        actual = "none"
    elif declared in ("none", "hidden") and native:
        actual = "full"
    solver.native_reasoning = actual  # type: ignore[assignment]
