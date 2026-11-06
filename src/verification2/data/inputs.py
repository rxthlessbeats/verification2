"""Validate a complete input batch and retain a record of unusable rows."""

import hashlib
from pathlib import Path

from pydantic import Field

from qera.jsonl import _parse_trajectory, check_trajectory

from ..core.config import Name, Settings


class QuarantinedRow(Settings):
    source: str
    line: int = Field(gt=0)
    sha256: str
    error: str
    traj_id: Name | None = None


def read_inputs(paths, unique_ids=False):
    trajectories, quarantined, seen = [], [], set()
    for name in paths:
        path = Path(name)
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                trajectory = None
                try:
                    trajectory = _parse_trajectory(line)
                    problems = check_trajectory(trajectory, include_warnings=False)
                    if problems:
                        raise ValueError("; ".join(problems))
                    key = (
                        trajectory.traj_id
                        if unique_ids
                        else (trajectory.benchmark, trajectory.traj_id)
                    )
                    if key in seen:
                        raise ValueError("Duplicate trajectory ID")
                    seen.add(key)
                    trajectories.append(trajectory)
                except ValueError as exc:
                    quarantined.append(
                        QuarantinedRow(
                            source=str(path),
                            line=number,
                            sha256=hashlib.sha256(line.encode()).hexdigest(),
                            error=str(exc),
                            traj_id=trajectory.traj_id if trajectory else None,
                        )
                    )
    return trajectories, quarantined


def write_quarantine(output, rows):
    (output / "quarantine.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
