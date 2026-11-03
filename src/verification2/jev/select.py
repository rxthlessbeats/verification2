"""Pack the most useful E candidates into a token budget.

Rank by usefulness (ties broken by fewer tokens), take greedily while the
budget allows, then return seg_ids sorted (the renderer restores the original
timeline). Ranking by usefulness-per-token was tried first and rejected: it
fills the budget with tiny low-score segments and crowds out the decisive
ones. Exact duplicate candidates are scored once and only the first copy can
be chosen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from verification2.jev.scorer import CandidateScore, JevScorer, evidence_text


@dataclass(frozen=True)
class SelectionResult:
    traj_id: str
    benchmark: str
    budget: int
    selected_e: list[int]
    used_tokens: int
    scores: list[CandidateScore] = field(default_factory=list)

    @property
    def selection_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.scores)

    @property
    def selection_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.scores)

    @property
    def selection_latency_s(self) -> float:
        return sum(s.latency_s for s in self.scores)

    def row(self) -> dict[str, Any]:
        """One line of the selections JSONL (pipeline reads traj_id and selected_e)."""
        return {
            "traj_id": self.traj_id,
            "benchmark": self.benchmark,
            "selected_e": self.selected_e,
            "method": "jev",
            "budget": self.budget,
            "used_tokens": self.used_tokens,
            "n_candidates": len(self.scores),
            "selection_input_tokens": self.selection_input_tokens,
            "selection_cost_usd": round(self.selection_cost_usd, 8),
            "selection_latency_s": round(self.selection_latency_s, 3),
        }

    def score_rows(self) -> list[dict[str, Any]]:
        chosen = set(self.selected_e)
        return [
            {
                "traj_id": self.traj_id,
                "seg_id": s.seg_id,
                "supports": s.supports,
                "contradicts": s.contradicts,
                "usefulness": s.usefulness,
                "stance": s.stance,
                "selected": s.seg_id in chosen,
                "input_tokens": s.input_tokens,
                "cost_usd": s.cost_usd,
                "truncated": s.truncated,
                "cached": s.cached,
            }
            for s in self.scores
        ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def pack(
    candidates: list[dict[str, Any]], scores: list[CandidateScore], budget: int
) -> tuple[list[int], int]:
    """Greedy fill by usefulness, then by size; returns (sorted seg_ids, tokens used)."""
    by_id = {s.seg_id: s for s in scores}
    seen: set[str] = set()
    unique = []
    for c in candidates:
        key = _norm(evidence_text(c))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    ranked = sorted(
        unique,
        key=lambda c: (-by_id[c["seg_id"]].usefulness, c["n_tokens"], c["seg_id"]),
    )
    chosen, used = [], 0
    for c in ranked:
        if used + c["n_tokens"] <= budget:
            chosen.append(c["seg_id"])
            used += c["n_tokens"]
    return sorted(chosen), used


def select_for_view(
    view: dict[str, Any], budget: int, scorer: JevScorer
) -> SelectionResult:
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    scores = scorer.score_view(view)
    selected, used = pack(view["candidates"], scores, budget)
    return SelectionResult(
        traj_id=view["traj_id"],
        benchmark=view["benchmark"],
        budget=budget,
        selected_e=selected,
        used_tokens=used,
        scores=scores,
    )
