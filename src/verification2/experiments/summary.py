"""Readable run-log lines; the JSON artifacts each line summarises stay authoritative."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from qera.jsonl import read_jsonl


def duration(seconds):
    if seconds is None:
        return "unknown"
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours}h{minutes:02d}m{seconds:02d}s"
        if hours
        else f"{minutes}m{seconds:02d}s"
    )


def tokens(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def token_pair(usage):
    usage = usage if isinstance(usage, dict) else {}
    return (
        usage.get("input_tokens", usage.get("prompt_tokens")) or 0,
        usage.get("output_tokens", usage.get("completion_tokens")) or 0,
    )


def episode_usage(directory):
    """Sum provider-reported usage over every native model call recorded in an episode."""
    total = {"model_calls": 0, "cached_calls": 0, "input_tokens": 0, "output_tokens": 0}
    for path in sorted(Path(directory).rglob("calls.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            response = record.get("response") or record.get("raw_response")
            usage = record.get("usage") or (
                response.get("usage") if isinstance(response, dict) else None
            )
            used_in, used_out = token_pair(usage)
            total["model_calls"] += 1
            total["cached_calls"] += bool(record.get("cache_hit"))
            total["input_tokens"] += used_in
            total["output_tokens"] += used_out
    return total


def native_cause(directory):
    """Last exception line from a native worker log, which names the real failure."""
    log = Path(directory) / "native.log"
    if not log.is_file():
        return None
    errors = [
        line.strip()
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if re.search(r"\w+(Error|Exception)\b", line)
    ]
    return errors[-1][:300] if errors else None


def usage_text(usage):
    if not usage:
        return None
    text = f"{usage['model_calls']} calls, {tokens(usage['input_tokens'])} in / {tokens(usage['output_tokens'])} out tokens"
    return text + (
        f" ({usage['cached_calls']} cached)" if usage["cached_calls"] else ""
    )


def outcome(trajectory):
    label = trajectory.label
    verdict = {True: "correct", False: "incorrect", None: "unlabelled"}[label.correct]
    return f"native {verdict}" + (
        f" (score {label.score:g})" if label.score is not None else ""
    )


def solver_task(task, row, trajectory=None):
    parts = [f"[{task.benchmark}] {task.key}: {row['status']}"]
    if row.get("reused") == "store":
        parts.append(f"stored, originally solved in {duration(row.get('duration_s'))}")
    elif row.get("reused"):
        parts.append("resumed from this collection")
    else:
        parts.append(f"took {duration(row.get('duration_s'))}")
    if trajectory is not None:
        parts += [outcome(trajectory), f"{len(trajectory.segments)} segments"]
    parts += [usage_text(row.get("usage")), row.get("error"), row.get("cause")]
    if row.get("stored"):
        parts.append("store " + row["stored"])
    return " | ".join(p for p in parts if p)


def solver(results):
    """Per-benchmark totals over (result row, trajectory or None) pairs."""
    groups = defaultdict(list)
    for row, trajectory in results:
        groups[row["task"]["benchmark"]].append((row, trajectory))
    for benchmark, group in groups.items():
        ok = [(r, t) for r, t in group if r["status"] == "ok" and t is not None]
        correct = sum(t.label.correct is True for _, t in ok)
        timed = [r["duration_s"] for r, _ in ok if r.get("duration_s") is not None]
        usage = [r["usage"] for r, _ in ok if r.get("usage")]
        print(
            f"[solver] {benchmark}: {len(ok)}/{len(group)} ok"
            f" ({sum(r.get('reused') == 'store' for r, _ in ok)} from store)"
            f" | native {correct} correct / {len(ok) - correct} incorrect"
            f" | solve time {duration(sum(timed))} over {len(timed)} timed tasks"
            + (f", mean {duration(sum(timed) / len(timed))}" if timed else "")
            + (
                f" | {tokens(sum(u['input_tokens'] for u in usage))} in /"
                f" {tokens(sum(u['output_tokens'] for u in usage))} out tokens"
                if usage
                else ""
            ),
            flush=True,
        )


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def jev(directory):
    directory = Path(directory)
    classified = directory / "trajectories.jsonl"
    trajectories = (
        {t.traj_id: t for t in read_jsonl([classified])} if classified.exists() else {}
    )
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows(directory / "results.jsonl"):
        used_in, used_out = token_pair(row.get("usage"))
        total = totals[row["benchmark"]]
        total["n"] += 1
        total["errors"] += row["status"] != "ok"
        total["in"] += used_in
        total["out"] += used_out
        parts = [f"[jev] {row['benchmark']} {row['task_id']}: {row['status']}"]
        if isinstance(row.get("raw_response"), dict):
            parts.append(f"{len(row['raw_response'])} pieces")
        trajectory = trajectories.get(row["traj_id"])
        if trajectory is not None:
            parts.append(
                f"E in {len(trajectory.by_role('E'))} / R in {len(trajectory.by_role('R'))} segments"
            )
        parts.append(f"{tokens(used_in)} in / {tokens(used_out)} out tokens")
        if row.get("classification_ms") is not None:
            parts.append(duration(row["classification_ms"] / 1000))
        if row.get("error"):
            parts.append(str(row["error"]).splitlines()[0][:300])
        print(" | ".join(parts), flush=True)
    for benchmark, total in totals.items():
        print(
            f"[jev] {benchmark}: {total['n'] - total['errors']}/{total['n']} ok"
            f" | {tokens(total['in'])} in / {tokens(total['out'])} out tokens",
            flush=True,
        )


def verifier(directory):
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows(Path(directory) / "results.jsonl"):
        used_in, used_out = token_pair(row.get("usage"))
        total = totals[(row["benchmark"], row["condition"])]
        total["n"] += 1
        total["in"] += used_in
        total["out"] += used_out
        parts = [
            f"[verifier] {row['benchmark']} {row['task_id']} {row['condition']}: {row['status']}"
        ]
        if row["status"] == "ok":
            gold = {True: "correct", False: "incorrect"}.get(row.get("label_correct"))
            agree = gold is not None and row.get("verdict") == gold
            total["agree"] += agree
            parts.append(
                f"verdict {row.get('verdict')}, native {gold or 'unlabelled'}"
                f" -> {'agree' if agree else 'DISAGREE'}"
            )
        else:
            total["errors"] += 1
            parts.append(str(row.get("error")).splitlines()[0][:300])
        parts.append(f"{tokens(used_in)} in / {tokens(used_out)} out tokens")
        if row.get("total_ms") is not None:
            parts.append(duration(row["total_ms"] / 1000))
        print(" | ".join(parts), flush=True)
    for (benchmark, condition), total in totals.items():
        print(
            f"[verifier] {benchmark} {condition}: {total['agree']}/{total['n']} agree with native,"
            f" {total['errors']} errors | {tokens(total['in'])} in / {tokens(total['out'])} out tokens",
            flush=True,
        )


def rate(row, name):
    value = row.get(name + "_rate")
    if not value:
        return "n/a"
    low, high = row.get(name + "_rate_low"), row.get(name + "_rate_high")
    interval = f" [{float(low):.1%}, {float(high):.1%}]" if low and high else ""
    return f"{float(value):.1%}{interval}"


def report(analysis):
    summary = Path(analysis) / "summary.csv"
    if not summary.exists():
        return
    with summary.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["subgroup"] != "all" or not row["attempted"]:
                continue
            print(
                f"[report] {row['benchmark']} {row['condition']}: {row['attempted']} attempted,"
                f" {row['errors']} errors, coverage {float(row['verifier_coverage'] or 0):.0%}"
                f" | native {row['correct_n']} correct / {row['incorrect_n']} incorrect"
                f" | false rejection {row['false_rejections']} = {rate(row, 'false_rejection')}"
                f" | false acceptance {row['false_acceptances']} = {rate(row, 'false_acceptance')}",
                flush=True,
            )
