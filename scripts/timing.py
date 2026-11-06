"""Wall-clock timing from a timestamped run.log: steps, stages and per-benchmark solve times."""

import itertools
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")
STEPS = ("doctor", "pilot", "train", "freeze", "test")
PARTITIONS = ("pilot", "train", "test")
STAGES = (
    ("jev", "[jev] started"),
    ("verifier", "[verifier] started"),
    ("report", "Report:"),
)
TASK = re.compile(
    r"^\[(\w+)\] .+?: (\w+) \| (took|stored, originally solved in|resumed)"
)
SPAN = re.compile(r"(?:(\d+)h)?(\d+)m(\d+)s")


def clock(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours}h{minutes:02d}m{seconds:02d}s"
        if hours
        else f"{minutes}m{seconds:02d}s"
    )


def span(text):
    match = SPAN.search(text)
    return (
        int(match[1] or 0) * 3600 + int(match[2]) * 60 + int(match[3])
        if match
        else None
    )


def stages(lines, start, end):
    marks = [("solver", start)]
    for name, prefix in STAGES:
        marks += [(name, t) for t, text in lines if text.startswith(prefix)][:1]
    marks.append((None, end))
    return [
        f"{name} {clock((after - at).total_seconds())}"
        for (name, at), (_, after) in itertools.pairwise(marks)
    ]


def solving(lines):
    groups = defaultdict(
        lambda: {"status": Counter(), "fresh": [], "stored": [], "resumed": 0}
    )
    for _, text in lines:
        if not (match := TASK.match(text)):
            continue
        group = groups[match[1]]
        group["status"][match[2]] += 1
        seconds = span(text[match.end() :])
        if match[3] == "took" and seconds is not None:
            group["fresh"].append(seconds)
        elif match[3] == "resumed":
            group["resumed"] += 1
        elif seconds is not None:
            group["stored"].append(seconds)
    for benchmark, group in groups.items():
        fresh, stored = group["fresh"], group["stored"]
        parts = [
            f"  {benchmark}: {sum(group['status'].values())} tasks ("
            + ", ".join(f"{n} {s}" for s, n in sorted(group["status"].items()))
            + ")"
        ]
        if fresh:
            parts.append(
                f"solved now {len(fresh)} in {clock(sum(fresh))}, "
                f"mean {clock(sum(fresh) / len(fresh))}, max {clock(max(fresh))}"
            )
        if stored:
            parts.append(f"from store {len(stored)}, originally {clock(sum(stored))}")
        if group["resumed"]:
            parts.append(f"resumed {group['resumed']}")
        yield " | ".join(parts)


def main(path, now=False):
    """With now, the open step of a running log lasts until the current time."""
    events = []
    with open(path, encoding="utf-8") as log:
        for raw in log:
            if match := LINE.match(raw.rstrip("\n")):
                events.append((datetime.fromisoformat(match[1]), match[2]))
    if not events:
        sys.exit("No timestamped lines in " + path)
    if now:
        events.append((datetime.now().replace(microsecond=0), ""))
    bounds = [
        (i, text)
        for i, (_, text) in enumerate(events)
        if text in STEPS or text.startswith("done (")
    ]
    bounds.append((len(events), None))
    first, last = events[0][0], events[-1][0]
    done = next((text for _, text in events if text.startswith("done (")), "running")
    print(
        f"total {clock((last - first).total_seconds())} ({first:%F %T} -> {last:%T}), {done}"
    )
    for (i, name), (j, _) in itertools.pairwise(bounds):
        if name not in STEPS:
            continue
        start, end = events[i][0], events[min(j, len(events) - 1)][0]
        lines = events[i:j]
        line = f"{name} {clock((end - start).total_seconds())}"
        if name in PARTITIONS:
            line += " | " + " | ".join(stages(lines, start, end))
        print(line)
        for row in solving(lines):
            print(row)


if __name__ == "__main__":
    main(sys.argv[-1], now="--now" in sys.argv[1:-1])
