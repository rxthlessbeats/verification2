"""Render ordered evidence views without labels or hidden fields."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .schema import Index, Role, Trajectory

CONDITIONS = {
    "QERA": {Role.Q, Role.E, Role.R, Role.A},
    "QA": {Role.Q, Role.A},
    "QEA": {Role.Q, Role.E, Role.A},
    "QRA": {Role.Q, Role.R, Role.A},
}


class VerifierInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    included_seg_ids: list[Index]


def build_verifier_input(traj: Trajectory, condition: str) -> VerifierInput:
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition}; choose from {list(CONDITIONS)}"
        )
    keep = CONDITIONS[condition]
    parts, ids = [], []
    for s in traj.segments:
        pieces = [t for r, t in s.pieces() if r in keep and t]
        call = None
        if s.call is not None:
            if s.call_role in keep:
                call = s.call.render()
            elif pieces:
                call = s.call.render(include_args=False)
        if not pieces and call is None:
            continue
        head = (
            f"[segment {s.seg_id} | {s.speaker.value}"
            + (f" | step {s.step}" if s.step is not None else "")
            + "]"
        )
        body = "\n".join(pieces)
        parts.append(
            "\n".join([head] + ([call] if call else []) + ([body] if body else []))
        )
        ids.append(s.seg_id)
    return VerifierInput(text="\n\n".join(parts), included_seg_ids=ids)
