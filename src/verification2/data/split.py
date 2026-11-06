"""Freeze each benchmark's train/test split before solver outcomes or prompt development."""

import argparse
import ast
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field, model_validator

from ..benchmarks.setup import ROOT
from ..core.config import Benchmark, Domain, Name, Settings

BENCHMARKS = get_args(Benchmark)
DOMAINS = get_args(Domain)
SEED = 42  # Orders seeded splits and seeds every native solver episode.


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Task(Settings):
    benchmark: Benchmark
    task_id: Name
    domain: Domain | None = None
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


def model_free_web_task(config):
    """Accept only audited native evaluator paths; unknown paths are ineligible."""
    evaluation = config.get("eval", {})
    kinds = evaluation.get("eval_types", [])
    if not kinds or not set(kinds) <= {"string_match", "url_match", "program_html"}:
        return False
    answers = evaluation.get("reference_answers") or {}
    if not set(answers) <= {"exact_match", "must_include"}:
        return (
            False  # Includes fuzzy matching and conditional unachievable-task judging.
        )
    helpers = {
        "shopping_get_latest_order_url",
        "shopping_get_sku_latest_review_author",
        "shopping_get_sku_latest_review_rating",
        "reddit_get_post_url",
        "gitlab_get_project_memeber_role",
    }
    for item in evaluation.get("program_html", []):
        if not set(item.get("required_contents", {})) <= {
            "exact_match",
            "must_include",
        }:
            return False
        for value in (item.get("url", ""), item.get("locator", "")):
            if isinstance(value, str) and value.startswith("func:"):
                try:
                    tree = ast.parse(value[5:], mode="eval")
                except SyntaxError:
                    return False
                if not isinstance(tree.body, ast.Call) or any(
                    not isinstance(
                        node,
                        (
                            ast.Expression,
                            ast.Call,
                            ast.Name,
                            ast.Load,
                            ast.Constant,
                            ast.keyword,
                        ),
                    )
                    for node in ast.walk(tree)
                ):
                    return False
                if any(
                    isinstance(node, ast.Call)
                    and (
                        not isinstance(node.func, ast.Name)
                        or node.func.id not in helpers
                    )
                    for node in ast.walk(tree)
                ):
                    return False
    return True


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def task_pool(benchmark):
    """Eligible native tasks; tau2 tasks carry the benchmark's own train/test split."""
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
    path = ROOT / "raw/tasks" / (benchmark + ".jsonl")
    if not path.exists():
        raise ValueError(f"Native task inputs missing: {path}; run make setup")
    for row in read_rows(path):
        identifier = (
            row.get("instance_id")
            if benchmark == "swebench_verified"
            else "webarena." + str(row["task_id"])
        )
        if benchmark == "agentrewardbench" and not model_free_web_task(row):
            excluded.append(
                {
                    "task_id": identifier,
                    "reason": "native scorer requires a model or an unaudited evaluator path",
                }
            )
            continue
        result.append(
            Task(
                benchmark=benchmark,
                task_id=identifier,
                source=path.relative_to(ROOT),
                source_row_hash=digest(row),
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
        identifier = (
            str(row.get("id"))
            if task.benchmark == "tau2_bench"
            else row.get("instance_id")
            if task.benchmark == "swebench_verified"
            else "webarena." + str(row["task_id"])
        )
        if identifier == task.task_id:
            if digest(row) != task.source_row_hash:
                raise ValueError(
                    "Native task data changed after the split was frozen: " + task.key
                )
            if task.benchmark == "agentrewardbench" and not model_free_web_task(row):
                raise ValueError(
                    "Model-based/unknown native web scoring is not eligible"
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
        if benchmark != "tau2_bench":
            # Odd domains take turns starting with test, so train is half, rounded down.
            first = 1 if len(rows) % 2 and odd % 2 == 0 else 0
            odd += len(rows) % 2
            rows = [
                t.model_copy(update={"partition": ("train", "test")[(i + first) % 2]})
                for i, t in enumerate(rows)
            ]
        tasks.extend(sorted(rows, key=lambda t: t.partition != "train"))
    method = (
        "native tau2 split_tasks.json train/test"
        if benchmark == "tau2_bench"
        else "seeded 50/50 within domain"
    )
    return Split(
        benchmark=benchmark,
        method=f"{method}; task order seeded with {SEED}",
        tasks=tasks,
        excluded=list(excluded),
    )


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
