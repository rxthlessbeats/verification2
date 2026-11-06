"""Read and write QERA JSONL using LF boundaries; Unicode separators remain inside records."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import SCHEMA_VERSION, Role, Trajectory


def lines(path) -> Iterator[str]:
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            if line.strip():
                yield line


def read(path) -> Iterator[dict]:
    for line in lines(path):
        yield json.loads(line)


def _parse_trajectory(line: str) -> Trajectory:
    # The role/span contract must match; retired optional count fields are ignored.
    version = json.loads(line).get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"schema_version={version!r}; expected {SCHEMA_VERSION}")
    return Trajectory.model_validate_json(line)


def read_jsonl(paths: Iterable[str]) -> Iterator[Trajectory]:
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield _parse_trajectory(line)
                    except Exception as e:
                        raise ValueError(f"{p}:{ln}: invalid record: {e}") from e


def write_jsonl(trajs: Iterable[Trajectory], out: str) -> int:
    """Validate all records before atomically replacing the output."""
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_name(out_p.name + ".tmp")
    n = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for t in trajs:
                t = t.revalidate()
                f.write(t.model_dump_json(exclude_none=False) + "\n")
                n += 1
        os.replace(tmp, out_p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return n


def check_trajectory(t: Trajectory, include_warnings: bool = True) -> list[str]:
    """Return semantic errors and optional [warning] messages."""
    problems = []
    if include_warnings and not any(s.has_evidence() for s in t.segments):
        problems.append("[warning] No evidence; QEA equals QA")
    if t.label.correct is None and t.label.source not in (
        "pending",
        "human_annotation",
    ):
        problems.append("Unknown label requires pending or human_annotation source")
    for s in t.segments:
        if not s.text and s.call is None:
            problems.append(f"seg {s.seg_id} is empty")
    g = t.hidden.get("gold_answer")
    if isinstance(g, str) and len(g) > 3:
        q_text = "\n".join(
            txt for s in t.segments for r, txt in s.pieces() if r == Role.Q
        ).lower()
        if g.lower() in q_text:
            problems.append("Gold answer appears in Q")
    return problems
