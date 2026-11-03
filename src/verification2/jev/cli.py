"""CLI: produce Jev selections for `verification2 run --selector file`, and show usage.

uv run python -m verification2.jev select --input data/tau2_bench.balanced.jsonl --budget 2048 --out selections/tau2_jev_2048.jsonl
uv run python -m verification2.jev select ... --budget-ratio 0.5     # per-trajectory budget = 50% of its E tokens
uv run python -m verification2.jev usage [--by run|day|model]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qera.cli import read_jsonl, retokenize_segment
from qera.render import selector_view
from verification2.jev.client import JevClient
from verification2.jev.scorer import JevScorer
from verification2.jev.select import select_for_view
from verification2.jev.usage import UsageLedger, format_table, summarize


def cmd_select(args: argparse.Namespace) -> int:
    if (args.budget is None) == (args.budget_ratio is None):
        raise SystemExit("give exactly one of --budget or --budget-ratio")
    if args.budget_ratio is not None and not 0 < args.budget_ratio <= 1:
        raise SystemExit("--budget-ratio must be in (0, 1]")
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    scores_path = out.with_name(out.stem + ".scores.jsonl")

    client = JevClient(run=args.run or out.stem)
    scorer = JevScorer(
        client,
        cache_dir=None if args.no_cache else args.cache_dir,
        workers=args.workers,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_sel, tmp_sc = out.with_suffix(".tmp"), scores_path.with_suffix(".tmp")
    n = total_tokens = 0
    total_cost = 0.0
    started = time.perf_counter()
    try:
        with (
            tmp_sel.open("w", encoding="utf-8") as sel_f,
            tmp_sc.open("w", encoding="utf-8") as sc_f,
        ):
            for traj in read_jsonl(args.input):
                if args.limit and n >= args.limit:
                    break
                for (
                    seg
                ) in traj.segments:  # same tokenizer as the pipeline uses at run time
                    retokenize_segment(seg)
                view = selector_view(traj)
                e_total = sum(c["n_tokens"] for c in view["candidates"])
                budget = (
                    args.budget
                    if args.budget is not None
                    else int(e_total * args.budget_ratio)
                )
                result = select_for_view(view, budget, scorer)
                sel_f.write(json.dumps(result.row(), ensure_ascii=False) + "\n")
                for row in result.score_rows():
                    sc_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                total_tokens += result.selection_input_tokens
                total_cost += result.selection_cost_usd
                print(
                    f"[{n}] {traj.traj_id}: {len(result.selected_e)}/{len(view['candidates'])} E, "
                    f"{result.used_tokens}/{budget} tok, jev {result.selection_input_tokens} tok",
                    file=sys.stderr,
                )
    except BaseException:
        tmp_sel.unlink(missing_ok=True)
        tmp_sc.unlink(missing_ok=True)
        raise
    finally:
        client.close()
    tmp_sel.replace(out)
    tmp_sc.replace(scores_path)
    print(
        f"{n} trajectories -> {out} (scores: {scores_path}); jev {total_tokens:,} input tokens, "
        f"${total_cost:.5f}, {time.perf_counter() - started:.1f}s"
    )
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    ledger = UsageLedger(args.log)
    total, groups = summarize(ledger.read(), group_by=args.by)
    if total.calls == 0:
        print(f"no records in {ledger.path}")
        return 0
    print(f"ledger: {ledger.path}\n")
    print(format_table(total, groups, args.by))
    print(
        f"\nspent so far: ${total.cost_usd:.5f} billed by gateway ({total.input_tokens:,} input tokens)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verification2.jev", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="write a selections JSONL for --selector file")
    s.add_argument("--input", nargs="+", required=True, help="QERA JSONL file(s)")
    s.add_argument(
        "--out",
        required=True,
        help="selections JSONL; scores go next to it as *.scores.jsonl",
    )
    s.add_argument("--budget", type=int, help="fixed E token budget per trajectory")
    s.add_argument(
        "--budget-ratio",
        type=float,
        help="budget as a fraction of each trajectory's E tokens",
    )
    s.add_argument("--limit", type=int, help="only the first N trajectories")
    s.add_argument("--run", help="ledger tag (default: output file stem)")
    s.add_argument(
        "--workers", type=int, default=8, help="parallel Jev calls per trajectory"
    )
    s.add_argument("--cache-dir", default=".jev_cache")
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_select)

    u = sub.add_parser("usage", help="show Jev token / cost ledger")
    u.add_argument("--by", choices=["run", "day", "model"], default="run")
    u.add_argument("--log", default=None)
    u.set_defaults(func=cmd_usage)

    args = p.parse_args(argv)
    return args.func(args)
