"""Collect every verification2 run under a directory into one table.

    uv run python scripts/collect_runs.py runs            # prints table, writes runs/collected.csv
    uv run python scripts/collect_runs.py runs --no-pair  # metrics per run instead of on the common paired set

One row per (benchmark, condition, selector, budget). Quality metrics follow
pipeline.metrics(): recall / FRR / balanced accuracy over trajectories whose
label is a bool. By default they are computed on the set of trajectories that
succeeded in EVERY run of that benchmark, so conditions are compared on the
same samples even when they were produced by separate runs.

Cost and latency follow the proposal's C_c = sum(s_i + v_i) and mean(l_i):
  verifier side   from results.jsonl (input_tokens, usage.cost if the endpoint
                  reports it, verification_ms)
  selection side  from <run>/selections.jsonl, which run_experiment.sh copies
                  next to the results for `--selector file` runs (Jev writes
                  selection_cost_usd and selection_latency_s per trajectory);
                  0 for the built-in selectors, whose selection time is the
                  pipeline's own selection_ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_runs(root: Path):
    runs = []
    for manifest in sorted(root.rglob("manifest.json")):
        d = manifest.parent
        results = d / "results.jsonl"
        if not results.exists():
            continue
        m = json.loads(manifest.read_text())
        with results.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        sel_cost, sel_lat = {}, {}
        # run_experiment.sh copies the selections file into the run dir; the manifest only stores its hash
        sel_path = d / "selections.jsonl"
        if sel_path.exists():
            with sel_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        sel_cost[r["traj_id"]] = r.get("selection_cost_usd", 0.0)
                        sel_lat[r["traj_id"]] = r.get("selection_latency_s", 0.0)
        runs.append(
            {
                "dir": str(d.relative_to(root)),
                "manifest": m,
                "rows": rows,
                "sel_cost": sel_cost,
                "sel_lat": sel_lat,
            }
        )
    return runs


def collect(runs, pair=True):
    by_bench = defaultdict(list)
    for run in runs:
        for r in run["rows"]:
            by_bench[r["benchmark"]].append((run, r))
    out = []
    for bench in sorted(by_bench):
        ok_per_run = defaultdict(set)
        all_traj = set()
        for run, r in by_bench[bench]:
            all_traj.add(r["traj_id"])
            if r["status"] == "ok":
                ok_per_run[(run["dir"], r["condition"])].add(r["traj_id"])
        paired = (
            set.intersection(*ok_per_run.values()) if pair and ok_per_run else all_traj
        )
        groups = defaultdict(list)
        for run, r in by_bench[bench]:
            m = run["manifest"]
            selector = m["selector"] if r["condition"] == "QEsA" else "-"
            if selector == "file":
                selector = "jev"
            budget = m["budget"] if r["condition"] == "QEsA" else "-"
            groups[(r["condition"], selector, budget)].append((run, r))
        for (cond, selector, budget), items in sorted(
            groups.items(), key=lambda kv: (kv[0][0], str(kv[0][2]), kv[0][1])
        ):
            # the same (condition, selector, budget) may appear in several runs; count each trajectory once
            seen, unique = set(), []
            for run, r in items:
                if r["traj_id"] not in seen:
                    seen.add(r["traj_id"])
                    unique.append((run, r))
            items = unique
            valid = [
                (run, r)
                for run, r in items
                if r["status"] == "ok"
                and r["traj_id"] in paired
                and isinstance(r["label_correct"], bool)
            ]
            bad = [r for _, r in valid if not r["label_correct"]]
            good = [r for _, r in valid if r["label_correct"]]
            recall = (
                sum(r["verdict"] == "incorrect" for r in bad) / len(bad)
                if bad
                else None
            )
            frr = (
                sum(r["verdict"] == "incorrect" for r in good) / len(good)
                if good
                else None
            )
            ver_cost = [((r.get("usage") or {}).get("cost")) for _, r in valid]
            ver_cost = [c for c in ver_cost if isinstance(c, (int, float))]
            sel_cost = [run["sel_cost"].get(r["traj_id"], 0.0) for run, r in valid]
            sel_lat = [
                run["sel_lat"].get(r["traj_id"], r["selection_ms"] / 1000)
                for run, r in valid
            ]
            n = len(valid)
            out.append(
                {
                    "benchmark": bench,
                    "condition": cond,
                    "selector": selector,
                    "budget": budget,
                    "attempted": len(items),
                    "errors": sum(r["status"] == "error" for _, r in items),
                    "n": n,
                    "incorrect_n": len(bad),
                    "correct_n": len(good),
                    "recall": recall,
                    "frr": frr,
                    "balanced_acc": (recall + 1 - frr) / 2
                    if recall is not None and frr is not None
                    else None,
                    "mean_input_tokens": sum(r["input_tokens"] or 0 for _, r in valid)
                    / n
                    if n
                    else None,
                    "mean_latency_s": (
                        sum(sel_lat)
                        + sum(r["verification_ms"] for _, r in valid) / 1000
                    )
                    / n
                    if n
                    else None,
                    "selection_cost_usd": sum(sel_cost),
                    "verifier_cost_usd": sum(ver_cost) if ver_cost else None,
                    "total_cost_usd": sum(ver_cost) + sum(sel_cost)
                    if ver_cost
                    else None,
                }
            )
    return out


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path)
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--out", type=Path, help="CSV path (default <root>/collected.csv)")
    a = ap.parse_args(argv)
    runs = load_runs(a.root)
    if not runs:
        print(f"no runs under {a.root}", file=sys.stderr)
        return 1
    table = collect(runs, pair=not a.no_pair)
    cols = [
        "benchmark",
        "condition",
        "selector",
        "budget",
        "n",
        "errors",
        "recall",
        "frr",
        "balanced_acc",
        "mean_input_tokens",
        "mean_latency_s",
        "selection_cost_usd",
        "verifier_cost_usd",
        "total_cost_usd",
    ]
    widths = {
        c: max(
            len(c),
            *(
                len(
                    fmt(
                        r[c],
                        0
                        if c == "mean_input_tokens"
                        else 3
                        if c
                        not in (
                            "selection_cost_usd",
                            "verifier_cost_usd",
                            "total_cost_usd",
                        )
                        else 4,
                    )
                )
                for r in table
            ),
        )
        for c in cols
    }
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in table:
        print(
            "  ".join(
                fmt(
                    r[c],
                    0
                    if c == "mean_input_tokens"
                    else 3
                    if c
                    not in ("selection_cost_usd", "verifier_cost_usd", "total_cost_usd")
                    else 4,
                ).ljust(widths[c])
                for c in cols
            )
        )
    out = a.out or a.root / "collected.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"\n{len(runs)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
