"""Freeze each benchmark's train/test split before solver outcomes or prompt development."""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field, model_validator

from qera.jsonl import read as read_rows

from ..benchmarks.setup import (
    ROOT,
    file_hash,  # noqa: F401 - retained for existing split module callers.
)
from ..core.config import Benchmark, Domain, Name, Settings

BENCHMARKS = get_args(Benchmark)
DOMAINS = get_args(Domain)
SEED = 42  # Orders seeded splits and seeds every native solver episode.
SCIENCEWORLD_TASKS = 500  # Fixed train and test size for ScienceWorld.


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class Task(Settings):
    benchmark: Benchmark
    task_id: Name
    domain: Name | None = None  # tau2 domain or AgentDojo suite; optional elsewhere
    source: Path
    source_row_hash: Name
    partition: Literal["train", "test", "pilot"] | None = None

    @model_validator(mode="after")
    def native_identifier(self):
        if any(
            c in self.task_id for c in ("/", "\\", "\x00", "\n", "\r")
        ) or self.task_id in (".", ".."):
            raise ValueError("Native task ID must not be a path")
        if self.benchmark == "swebench_verified" and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.task_id
        ):
            raise ValueError("SWE instance ID must be a safe native filename")
        return self

    @property
    def key(self):
        return "/".join(filter(None, (self.benchmark, self.domain, self.task_id)))


class Split(Settings):
    """One benchmark's eligible tasks, each frozen as train or test."""

    version: Literal[1] = 1
    benchmark: Benchmark
    method: str
    tasks: list[Task] = Field(min_length=2)
    excluded: list[dict[str, str]] = Field(default_factory=list)

    @property
    def split_id(self):
        return digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def check_partition(self):
        if len({t.key for t in self.tasks}) != len(self.tasks):
            raise ValueError("Split tasks must be unique")
        if any(t.benchmark != self.benchmark for t in self.tasks):
            raise ValueError("Split tasks must belong to its benchmark")
        if {t.partition for t in self.tasks} != {"train", "test"}:
            raise ValueError("A split needs train and test tasks only")
        return self


def task_pool(benchmark):
    """Eligible native tasks; tau2/AppWorld carry their own train/test split."""
    result, excluded = [], []
    if benchmark == "tau2_bench":
        for domain in DOMAINS:
            directory = ROOT / "raw/tau2-bench/data/tau2/domains" / domain
            native = json.loads((directory / "split_tasks.json").read_text())
            partition = {str(i): p for p in ("train", "test") for i in native[p]}
            path = directory / "tasks.json"
            for row in json.loads(path.read_text()):
                if str(row["id"]) in map(str, native["base"]):
                    result.append(
                        Task(
                            benchmark=benchmark,
                            domain=domain,
                            task_id=str(row["id"]),
                            source=path.relative_to(ROOT),
                            source_row_hash=digest(row),
                            partition=partition[str(row["id"])],
                        )
                    )
        return result, excluded
    if benchmark == "appworld":
        path = ROOT / "raw/tasks/appworld.jsonl"
        if not path.exists():
            raise ValueError(f"Native task inputs missing: {path}; run make setup")
        for row in read_rows(path):
            partition = row.get("partition")
            if partition not in ("train", "test"):
                excluded.append(
                    {
                        "task_id": row["task_id"],
                        "reason": "excluded AppWorld split (e.g. test_challenge)",
                    }
                )
                continue
            result.append(
                Task(
                    benchmark=benchmark,
                    task_id=row["task_id"],
                    source=path.relative_to(ROOT),
                    source_row_hash=digest(row),
                    partition=partition,
                )
            )
        return result, excluded
    path = ROOT / "raw/tasks" / (benchmark + ".jsonl")
    if not path.exists():
        raise ValueError(f"Native task inputs missing: {path}; run make setup")
    for row in read_rows(path):
        if benchmark == "swebench_verified":
            identifier = row.get("instance_id")
            domain = None
        elif benchmark == "agentdojo":
            identifier = row["task_id"]
            domain = row["suite"]
        elif benchmark in ("scienceworld", "bird"):
            identifier = row["task_id"]
            domain = row.get("domain")
        else:
            raise ValueError("No native task pool for " + benchmark)
        result.append(
            Task(
                benchmark=benchmark,
                task_id=identifier,
                domain=domain,
                source=path.relative_to(ROOT),
                source_row_hash=digest(row),
                partition=row.get("partition"),
            )
        )
    return result, excluded


def load_task(task):
    path = task.source if task.source.is_absolute() else ROOT / task.source
    rows = (
        json.loads(path.read_text())
        if task.benchmark == "tau2_bench"
        else read_rows(path)
    )
    for row in rows:
        if task.benchmark == "tau2_bench":
            identifier = str(row.get("id"))
        elif task.benchmark == "swebench_verified":
            identifier = row.get("instance_id")
        elif task.benchmark == "agentdojo":
            # The same user_task_N exists in every suite.
            if row.get("suite") != task.domain:
                continue
            identifier = row.get("task_id")
        elif task.benchmark == "appworld":
            identifier = row.get("task_id")
        elif task.benchmark in ("scienceworld", "bird"):
            if row.get("domain") != task.domain:
                continue
            identifier = row.get("task_id")
        else:
            raise ValueError("No native task id for " + task.benchmark)
        if identifier == task.task_id:
            if digest(row) != task.source_row_hash:
                raise ValueError(
                    "Native task data changed after the split was frozen: " + task.key
                )
            return row
    raise ValueError("Frozen native task is missing: " + task.key)


def make_split(benchmark, pool, excluded=()):
    """tau2 keeps its native split; other benchmarks alternate train/test in seeded order.

    Tasks are listed by domain in seeded order, so the first N of a partition are a
    random sample.
    """
    strata = defaultdict(list)
    for task in sorted(pool, key=lambda t: t.key):
        strata[task.domain or "all"].append(task)
    tasks, odd = [], 0
    for key in sorted(strata):
        rows = strata[key]
        random.Random(f"{SEED}:{benchmark}:{key}").shuffle(rows)
        if any(task.partition is None for task in rows):
            # Odd domains take turns starting with test, so train is half, rounded down.
            first = 1 if len(rows) % 2 and odd % 2 == 0 else 0
            odd += len(rows) % 2
            rows = [
                t.model_copy(update={"partition": ("train", "test")[(i + first) % 2]})
                for i, t in enumerate(rows)
            ]
        tasks.extend(sorted(rows, key=lambda t: t.partition != "train"))
    tasks, dropped = limit_fixed_splits(benchmark, tasks)
    method = (
        "native tau2 split_tasks.json train/test"
        if benchmark == "tau2_bench"
        else "native AppWorld train+dev / test_normal"
        if benchmark == "appworld"
        else f"native ScienceWorld train+dev / test variations; seed-42 sample of {SCIENCEWORLD_TASKS} per partition"
        if benchmark == "scienceworld"
        else "native BIRD train / Mini-Dev; train sample matches the Mini-Dev count"
        if benchmark == "bird"
        else "seeded 50/50 within domain"
    )
    return Split(
        benchmark=benchmark,
        method=f"{method}; task order seeded with {SEED}",
        tasks=tasks,
        excluded=[*excluded, *dropped],
    )


def sample_partition(tasks, partition, count):
    """Seeded domain round-robin. Every domain is used before any domain is repeated."""
    pool = [task for task in tasks if task.partition == partition]
    if len(pool) <= count:
        return {task.key for task in pool}
    strata = defaultdict(list)
    for task in pool:
        strata[task.domain or "all"].append(task)
    picked = []
    queues = [strata[key] for key in sorted(strata)]
    for index in range(max(map(len, queues), default=0)):
        for queue in queues:
            if index < len(queue) and len(picked) < count:
                picked.append(queue[index])
    if len(picked) != count:
        raise ValueError(f"Could not sample {count} {partition} tasks")
    return {task.key for task in picked}


def limit_fixed_splits(benchmark, tasks):
    """BIRD matches its train sample to Mini-Dev. ScienceWorld keeps 500 on each side."""
    if benchmark == "bird":
        chosen = sample_partition(
            tasks, "train", sum(task.partition == "test" for task in tasks)
        )
        chosen.update(task.key for task in tasks if task.partition == "test")
        reason = "outside the seed-42 train sample matched to the test count"
    elif benchmark == "scienceworld":
        chosen = sample_partition(
            tasks, "train", SCIENCEWORLD_TASKS
        ) | sample_partition(tasks, "test", SCIENCEWORLD_TASKS)
        reason = f"outside the seed-42 sample of {SCIENCEWORLD_TASKS} per partition"
    else:
        return tasks, []
    dropped = [
        {
            "task_id": task.task_id,
            "domain": task.domain or "",
            "reason": reason,
        }
        for task in tasks
        if task.key not in chosen
    ]
    return [task for task in tasks if task.key in chosen], dropped


def load_split(benchmark):
    """Read data/splits/<benchmark>.json, freezing it on first use; native rows must not change."""
    path = ROOT / "data/splits" / (benchmark + ".json")
    pool, excluded = task_pool(benchmark)
    if not path.exists():
        split = make_split(benchmark, pool, excluded)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(split.model_dump_json(indent=2) + "\n")
        counts = [sum(t.partition == p for t in split.tasks) for p in ("train", "test")]
        print(f"Froze {benchmark} split: {counts[0]} train, {counts[1]} test -> {path}")
        return split
    split = Split.model_validate_json(path.read_text())
    rows = {t.key: t.source_row_hash for t in pool}
    changed = [t.key for t in split.tasks if rows.get(t.key) != t.source_row_hash]
    if changed:
        raise ValueError(
            f"{len(changed)} tasks in {path} are missing or changed in the native data, "
            f"for example {changed[0]}"
        )
    return split


def split_id(splits):
    return digest({s.benchmark: s.split_id for s in splits})


def select(splits, partition, limit=None):
    """Tasks per benchmark, round-robin over domains; the pilot is each first train task."""
    chosen = []
    for split in splits:
        part = "train" if partition == "pilot" else partition
        strata = defaultdict(list)
        for task in split.tasks:
            if task.partition == part:
                strata[task.domain or "all"].append(task)
        queues = [strata[key] for key in sorted(strata)]
        count = 1 if partition == "pilot" else limit
        picked = []
        for i in range(max(map(len, queues), default=0)):
            for queue in queues:
                if i < len(queue) and (count is None or len(picked) < count):
                    picked.append(queue[i].key)
        chosen.extend(
            t.model_copy(update={"partition": partition})
            for t in split.tasks
            if t.partition == part and t.key in picked
        )
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS)
    )
    for benchmark in parser.parse_args(argv).benchmarks:
        split = load_split(benchmark)
        counts = [sum(t.partition == p for t in split.tasks) for p in ("train", "test")]
        print(f"{benchmark}: {counts[0]} train, {counts[1]} test ({split.method})")


if __name__ == "__main__":
    main()
