"""Paired class-conditional error rates, coverage and task-bootstrap uncertainty."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from ..agents.verifier import digest, metrics
from ..core.models import VerifierResult
from ..data.split import BENCHMARKS, load_split, select, split_id

RATES = ("false_rejection_rate", "false_acceptance_rate")
CONTRASTS = {
    "add_E_without_R": ("QEA", "QA"),
    "add_R_without_E": ("QRA", "QA"),
    "add_R_with_E": ("QERA", "QEA"),
    "add_E_with_R": ("QERA", "QRA"),
}


def interval(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None, None

    def percentile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return percentile(0.025), percentile(0.975)


def read_runs(paths):
    selected, events, conditions, configuration = {}, [], None, None
    for path in dict.fromkeys(Path(p).resolve() for p in paths):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        signature = [
            manifest.get("model"),
            manifest.get("configuration", {}).get("endpoint"),
            manifest.get("prompt_hash"),
            manifest.get("max_tokens"),
            manifest["conditions"],
            manifest.get("source_hash"),
        ]
        if configuration is not None and configuration != signature:
            raise ValueError(
                "Report inputs have different model/prompt/decoding/conditions; report them separately"
            )
        configuration, conditions = signature, manifest["conditions"]
        for line in (path / "results.jsonl").read_text().splitlines():
            row = VerifierResult.model_validate_json(line).model_dump(mode="json")
            events.append(row)
            key = row["benchmark"], row["traj_id"], row["condition"]
            previous = selected.get(key)
            if previous and any(
                previous[k] != row[k]
                for k in (
                    "split_id",
                    "native_task_key",
                    "partition",
                    "trajectory_hash",
                    "label_correct",
                    "visible_r",
                )
            ):
                raise ValueError(
                    "Report inputs describe different frozen trajectories or labels"
                )
            if (
                previous
                and previous["status"] == row["status"] == "ok"
                and digest(previous["raw_response"]) != digest(row["raw_response"])
            ):
                raise ValueError(
                    "Conflicting valid responses for one frozen comparison"
                )
            if previous is None or previous["status"] != "ok" and row["status"] == "ok":
                selected[key] = row
    return list(selected.values()), events, conditions, configuration


def paired(rows, conditions):
    groups = defaultdict(list)
    for row in rows:
        groups[row["benchmark"], row["traj_id"]].append(row)
    return [
        r
        for group in groups.values()
        if len(group) == len(conditions)
        and {r["condition"] for r in group} == set(conditions)
        and all(r["status"] == "ok" and type(r["label_correct"]) is bool for r in group)
        for r in group
    ]


def rates(rows):
    good = [r for r in rows if r["label_correct"]]
    bad = [r for r in rows if not r["label_correct"]]
    return {
        RATES[0]: sum(r["verdict"] == "incorrect" for r in good) / len(good)
        if good
        else None,
        RATES[1]: sum(r["verdict"] == "correct" for r in bad) / len(bad)
        if bad
        else None,
    }


def analyze(rows, conditions, planned=None, iterations=2000, seed=42):
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    planned = planned or {
        b: len({r["traj_id"] for r in rows if r["benchmark"] == b})
        for b in {r["benchmark"] for r in rows}
    }
    planned = dict(sorted(planned.items()))
    summaries, comparisons = [], []
    for subgroup in ("all", "visible_R"):
        subset = [r for r in rows if subgroup == "all" or r.get("visible_r")]
        valid = paired(subset, conditions)
        base = {
            (r["benchmark"], r["condition"]): r for r in metrics(subset, conditions)
        }
        task_groups = defaultdict(lambda: defaultdict(list))
        for row in valid:
            task_groups[row["benchmark"]][row.get("task_id") or row["traj_id"]].append(
                row
            )
        distributions = defaultdict(list)
        randomizer = random.Random(f"{seed}:{subgroup}")
        for _ in range(iterations):
            sampled = {}
            for benchmark in planned:
                groups = task_groups[benchmark]
                keys = sorted(groups)
                sample = [r for _ in keys for r in groups[randomizer.choice(keys)]]
                for condition in conditions:
                    sampled[benchmark, condition] = rates(
                        [r for r in sample if r["condition"] == condition]
                    )
            for condition in conditions:
                sampled["macro_equal", condition] = {
                    rate: sum(values) / len(values)
                    if values and all(v is not None for v in values)
                    else None
                    for rate in RATES
                    for values in [[sampled[b, condition][rate] for b in planned]]
                }
            for (benchmark, condition), result in sampled.items():
                for rate in RATES:
                    distributions[benchmark, condition, rate].append(result[rate])
            for benchmark in (*planned, "macro_equal"):
                for name, (after, before) in CONTRASTS.items():
                    if after in conditions and before in conditions:
                        for rate in RATES:
                            a, b = (
                                sampled[benchmark, after][rate],
                                sampled[benchmark, before][rate],
                            )
                            distributions[benchmark, name, rate].append(
                                a - b if a is not None and b is not None else None
                            )
        points = {}
        for benchmark in planned:
            for condition in conditions:
                result = base.get(
                    (benchmark, condition),
                    {
                        "benchmark": benchmark,
                        "condition": condition,
                        "attempted": 0,
                        "errors": 0,
                        "paired_labelled_n": 0,
                        "incorrect_n": 0,
                        "correct_n": 0,
                        "false_rejections": 0,
                        "false_acceptances": 0,
                        RATES[0]: None,
                        RATES[1]: None,
                    },
                )
                points[benchmark, condition] = result
                n = (
                    planned[benchmark]
                    if subgroup == "all"
                    else len(
                        {r["traj_id"] for r in subset if r["benchmark"] == benchmark}
                    )
                )
                record = {
                    "subgroup": subgroup,
                    **result,
                    "planned_n": n,
                    "verifier_coverage": result["paired_labelled_n"]
                    / result["attempted"]
                    if result["attempted"]
                    else None,
                    "planned_coverage": result["paired_labelled_n"] / n if n else None,
                }
                for rate in RATES:
                    record[rate + "_low"], record[rate + "_high"] = interval(
                        distributions[benchmark, condition, rate]
                    )
                summaries.append(record)
        for condition in conditions:
            point = {
                rate: sum(values) / len(values)
                if values and all(v is not None for v in values)
                else None
                for rate in RATES
                for values in [[points[b, condition][rate] for b in planned]]
            }
            points["macro_equal", condition] = point
            record = {
                "subgroup": subgroup,
                "benchmark": "macro_equal",
                "condition": condition,
                **point,
            }
            for rate in RATES:
                record[rate + "_low"], record[rate + "_high"] = interval(
                    distributions["macro_equal", condition, rate]
                )
            summaries.append(record)
        for benchmark in (*planned, "macro_equal"):
            for name, (after, before) in CONTRASTS.items():
                if after in conditions and before in conditions:
                    for rate in RATES:
                        a, b = (
                            points[benchmark, after][rate],
                            points[benchmark, before][rate],
                        )
                        low, high = interval(distributions[benchmark, name, rate])
                        comparisons.append(
                            {
                                "subgroup": subgroup,
                                "benchmark": benchmark,
                                "comparison": name,
                                "metric": rate,
                                "change": a - b
                                if a is not None and b is not None
                                else None,
                                "low": low,
                                "high": high,
                            }
                        )
    return summaries, comparisons


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS)
    parser.add_argument(
        "--partition", choices=["pilot", "train", "test"], default="train"
    )
    parser.add_argument("--limit", type=int, help="Tasks per benchmark that were run")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Report output already exists; use a new directory")
    rows, events, conditions, configuration = read_runs(args.input)
    planned = None
    run_split = None
    if args.benchmarks:
        splits = [load_split(b) for b in args.benchmarks]
        run_split = split_id(splits)
        tasks = select(splits, args.partition, args.limit)
        selected = {t.key for t in tasks}
        trajectories = {}
        for row in rows:
            if (
                row["split_id"] != run_split
                or row["partition"] != args.partition
                or row["native_task_key"] not in selected
                or row["trajectory_hash"] is None
            ):
                raise ValueError(
                    "Verifier results do not belong to this frozen split/partition"
                )
            key = row["native_task_key"]
            identity = (
                row["traj_id"],
                row["trajectory_hash"],
                row["label_correct"],
                row["visible_r"],
            )
            if key in trajectories and trajectories[key] != identity:
                raise ValueError(
                    "Formal reports require one frozen trajectory per native task"
                )
            trajectories[key] = identity
        planned = {b: sum(t.benchmark == b for t in tasks) for b in args.benchmarks}
    summaries, comparisons = analyze(
        rows, conditions, planned, args.bootstrap, args.seed
    )
    args.output.mkdir(parents=True)
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "comparisons.csv", comparisons)
    report = {
        "configuration": configuration,
        "split_id": run_split,
        "inputs": [str(p.resolve()) for p in args.input],
        "partition": args.partition,
        "limit": args.limit,
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_unit": "canonical task; paired conditions resampled together",
        "seed": args.seed,
        "interval": "95% percentile bootstrap; zero-denominator draws omitted; missing class remains null",
        "quality_counts": "only valid common paired trajectories",
        "negative_change": "lower error rate",
        "non_cached_result_records": sum(
            r["status"] in ("ok", "error") and not r["cache_hit"] for r in events
        ),
        "cache_hits": sum(r["cache_hit"] for r in events),
        "error_result_records": sum(r["status"] == "error" for r in events),
        "unknown_usage_records": sum(r["usage"] is None for r in events),
        "summary": summaries,
        "comparisons": comparisons,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Report:", args.output)


if __name__ == "__main__":
    main()
