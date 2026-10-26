"""Show Jev token usage and cost from the local ledger.

uv run python scripts/usage.py              # grouped by experiment tag
uv run python scripts/usage.py --by day     # grouped by UTC day
uv run python scripts/usage.py --by model
uv run python scripts/usage.py --log other.jsonl
"""

import argparse

from verification2.jev.usage import UsageLedger, format_table, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["run", "day", "model"], default="run")
    ap.add_argument("--log", default=None, help="ledger path (default .jev_usage.jsonl or $JEV_USAGE_LOG)")
    args = ap.parse_args()

    ledger = UsageLedger(args.log)
    total, groups = summarize(ledger.read(), group_by=args.by)
    if total.calls == 0:
        print(f"no records in {ledger.path}")
        return
    print(f"ledger: {ledger.path}\n")
    print(format_table(total, groups, args.by))
    print(
        f"\nspent so far: ${total.input_tokens / 1e6 * 0.042:.5f} at list price ($0.042/M input tokens), "
        f"${total.cost_usd:.5f} billed by gateway"
    )


if __name__ == "__main__":
    main()
