"""Append-only ledger of every Jev call, plus a summary reader.

Each call appends one JSON line to the ledger (default: .jev_usage.jsonl, or
JEV_USAGE_LOG). Use `uv run python scripts/usage.py` to see totals.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LOG = ".jev_usage.jsonl"


@dataclass(frozen=True)
class UsageRecord:
    ts: str  # ISO-8601 UTC
    model: str
    run: str  # experiment tag, e.g. "musique-smoke"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    status: int  # HTTP status; 200 on success


class UsageLedger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.environ.get("JEV_USAGE_LOG", DEFAULT_LOG))

    def record(
        self,
        model: str,
        run: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_s: float,
        status: int = 200,
    ) -> UsageRecord:
        rec = UsageRecord(
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            model=model,
            run=run,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_s=latency_s,
            status=status,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def read(self) -> Iterator[UsageRecord]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield UsageRecord(**json.loads(line))


@dataclass
class UsageTotals:
    calls: int = 0
    ok: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def add(self, r: UsageRecord) -> None:
        self.calls += 1
        self.ok += r.status == 200
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cost_usd += r.cost_usd
        self.latency_s += r.latency_s

    @property
    def mean_latency_s(self) -> float:
        return self.latency_s / self.ok if self.ok else 0.0


def summarize(records: Iterator[UsageRecord], group_by: str = "run") -> tuple[UsageTotals, dict[str, UsageTotals]]:
    """Return overall totals and totals grouped by `run`, `day`, or `model`."""
    total = UsageTotals()
    groups: dict[str, UsageTotals] = defaultdict(UsageTotals)
    for r in records:
        total.add(r)
        key = r.ts[:10] if group_by == "day" else getattr(r, group_by)
        groups[key].add(r)
    return total, dict(groups)


def format_table(total: UsageTotals, groups: dict[str, UsageTotals], group_by: str) -> str:
    rows = sorted(groups.items())
    w = max([len(group_by)] + [len(k) for k, _ in rows] + [5])
    head = f"{group_by:<{w}}  {'calls':>6} {'ok':>5} {'in_tok':>10} {'out_tok':>8} {'cost_usd':>10} {'lat_ms':>7}"
    lines = [head, "-" * len(head)]
    for k, t in rows:
        lines.append(
            f"{k:<{w}}  {t.calls:>6} {t.ok:>5} {t.input_tokens:>10,} {t.output_tokens:>8,} {t.cost_usd:>10.5f} {t.mean_latency_s * 1000:>7.0f}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<{w}}  {total.calls:>6} {total.ok:>5} {total.input_tokens:>10,} {total.output_tokens:>8,} {total.cost_usd:>10.5f} {total.mean_latency_s * 1000:>7.0f}"
    )
    return "\n".join(lines)
