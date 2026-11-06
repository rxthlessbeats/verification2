"""Apply validated overrides with per-field precedence: manual > classifier > rule."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator

from .builder import _check_reasoning_provenance
from .schema import Role, Span, Trajectory

_PRIORITY = {"rule": 0, "classifier": 1, "manual": 2}


def apply(
    trajs: Iterable[Trajectory],
    overrides: dict[tuple[str, int], dict],
    stats: Counter | None = None,
) -> Iterator[Trajectory]:
    stats = stats if stats is not None else Counter()
    for t in trajs:
        changed = False
        for s in t.segments:
            o = overrides.get((t.traj_id, s.seg_id))
            if not o:
                continue
            applied = False
            for field in ("role", "spans", "call_role"):
                if field not in o:
                    continue
                src = o.get("_sources", {}).get(field, o.get("source", "manual"))
                text_field = "spans" if s.spans else "role"
                previous = s.override_sources.get(
                    "call_role" if field == "call_role" else text_field,
                    s.role_source
                    if (
                        s.call_role_override is not None
                        if field == "call_role"
                        else s.spans is not None or s.role_override is not None
                    )
                    else "rule",
                )
                if _PRIORITY[src] < _PRIORITY[previous]:
                    continue
                if field == "role":
                    s.role_override = Role(o[field])
                    s.spans = None
                    s.override_sources.pop("spans", None)
                elif field == "spans":
                    s.spans = [
                        Span(
                            start=a,
                            end=b,
                            role=Role(r),
                        )
                        for a, b, r in o[field]
                    ]
                    s.role_override = None
                    s.override_sources.pop("role", None)
                else:
                    s.call_role_override = Role(o[field])
                s.override_sources[field] = src
                applied = True
            if not applied:
                stats["lower_priority"] += 1
                continue
            s.role_source = max(s.override_sources.values(), key=_PRIORITY.get)
            if o.get("note"):
                s.note = o["note"]
            stats["applied"] += 1
            changed = True
        if changed:
            try:
                _check_reasoning_provenance(t.segments, t.solver, t.meta)
                t = t.revalidate()
            except Exception as e:
                raise ValueError(f"{t.traj_id} invalid after overrides: {e}") from e
        yield t
