"""Frozen solver trajectories, collected once per task and solver model and reused by later runs.

Layout: data/trajectory/<benchmark>/<model>/<task>-<identity digest>/ with entry.json,
trajectory.json and the native episode artifacts.
"""

import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from qera.schema import Trajectory

from ..benchmarks.doctor import LOCKS, NEEDS
from ..benchmarks.setup import ROOT, SOURCES
from ..data.split import BENCHMARKS, digest, file_hash
from .summary import episode_usage

RUN_META = ("split_id", "cohort_id")
# The run's labelled copy of trajectory.json and its result row.
RUN_FILES = ("frozen.json", "result.json")


def default_store():
    return Path(os.environ.get("TRAJECTORY_STORE") or ROOT / "data/trajectory")


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80]


def identity(task, model, seed, solver_prompt, runtime_hashes, reasoning_effort=None):
    """Inputs that determine a native trajectory; unrelated code changes do not invalidate it."""
    needs = NEEDS[task.benchmark]
    return {
        "benchmark": task.benchmark,
        "task": task.key,
        "source_row_hash": task.source_row_hash,
        "solver_model": model,
        "solver_prompt_hash": digest(solver_prompt),
        "seed": seed,
        "native_sources": {
            n: SOURCES[n]["revision"] for n in sorted(needs & SOURCES.keys())
        },
        "runtime_hashes": {
            LOCKS[n]: runtime_hashes.get(LOCKS[n], "missing")
            for n in sorted(needs & LOCKS.keys())
        },
        # Omitted at the provider default so entries stored before efforts keep their key.
        **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
    }


def location(store, key):
    name = key["task"].removeprefix(key["benchmark"] + "/")
    return (
        Path(store)
        / key["benchmark"]
        / slug(key["solver_model"])
        / f"{slug(name)}-{digest(key)[:12]}"
    )


def load(store, key):
    directory = location(store, key)
    if not (directory / "entry.json").is_file():
        return None
    entry = json.loads((directory / "entry.json").read_text())
    trajectory = directory / "trajectory.json"
    if entry["identity"] != key or file_hash(trajectory) != entry["trajectory_sha256"]:
        raise ValueError(
            "Stored trajectory does not match its entry: " + str(directory)
        )
    return directory, entry, Trajectory.model_validate_json(trajectory.read_text())


def unlabelled(trajectory):
    """Train/test membership comes only from data/splits, applied when a run reuses it."""
    meta = {k: v for k, v in trajectory.meta.items() if k not in RUN_META}
    return trajectory.model_copy(update={"split": None, "meta": meta})


def save(store, key, trajectory, episode, replace=False, **details):
    """Publish atomically; an existing entry is replaced only on an explicit rerun."""
    directory = location(store, key)
    content = unlabelled(trajectory).model_dump_json() + "\n"
    if (directory / "entry.json").is_file() and (
        not replace or (directory / "trajectory.json").read_text() == content
    ):
        return directory
    staging = directory.with_name(directory.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(
        episode, staging / "episode", ignore=shutil.ignore_patterns(*RUN_FILES)
    )
    (staging / "trajectory.json").write_text(content)
    entry = {
        "identity": key,
        "trajectory_sha256": file_hash(staging / "trajectory.json"),
        "outcome": {
            "correct": trajectory.label.correct,
            "score": trajectory.label.score,
        },
        "segments": len(trajectory.segments),
        "usage": episode_usage(episode),
        **details,
    }
    (staging / "entry.json").write_text(json.dumps(entry, indent=2) + "\n")
    if directory.exists():
        retired = directory.with_name(directory.name + ".replaced")
        shutil.rmtree(retired, ignore_errors=True)
        directory.rename(retired)
        staging.rename(directory)
        shutil.rmtree(retired)
    else:
        staging.rename(directory)
    return directory


def add_rerun(parser):
    parser.add_argument(
        "--rerun",
        nargs="+",
        default=[],
        choices=[*BENCHMARKS, "all"],
        help="Solve these benchmarks (or all) again and replace their stored trajectories",
    )


def rerun_benchmarks(values):
    return set(BENCHMARKS) if "all" in values else set(values)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")
