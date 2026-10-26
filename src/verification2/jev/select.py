"""Budgeted evidence selection: Jev plus the three baselines.

Every selector has the same signature so they are interchangeable in the
experiment loop:

    select(question, candidate_answer, units, budget_tokens, ...) -> list[EvidenceUnit]

The result is always a subset of `units` in their ORIGINAL order, whose total
token count is <= budget_tokens. None of the selectors see labels.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from verification2.jev.scorer import EvidenceScore, JevScorer

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    text: str
    tokens: int
    position: int  # index in the original trajectory order


def make_units(texts: Sequence[str], count_tokens: TokenCounter, prefix: str = "e") -> list[EvidenceUnit]:
    return [EvidenceUnit(id=f"{prefix}{i}", text=t, tokens=count_tokens(t), position=i) for i, t in enumerate(texts)]


def _pack(ranked: Sequence[EvidenceUnit], budget_tokens: int) -> list[EvidenceUnit]:
    """Greedily take units in `ranked` order while they fit, then restore original order."""
    chosen: list[EvidenceUnit] = []
    used = 0
    for u in ranked:
        if used + u.tokens <= budget_tokens:
            chosen.append(u)
            used += u.tokens
    return sorted(chosen, key=lambda u: u.position)


def _dedupe(units: Sequence[EvidenceUnit]) -> list[EvidenceUnit]:
    seen: set[str] = set()
    out: list[EvidenceUnit] = []
    for u in units:
        key = re.sub(r"\s+", " ", u.text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


# ---------------------------------------------------------------- baselines


def random_select(
    question: str, candidate_answer: str, units: Sequence[EvidenceUnit], budget_tokens: int, seed: int = 0
) -> list[EvidenceUnit]:
    rng = random.Random(seed)
    order = list(_dedupe(units))
    rng.shuffle(order)
    return _pack(order, budget_tokens)


def recency_select(
    question: str, candidate_answer: str, units: Sequence[EvidenceUnit], budget_tokens: int
) -> list[EvidenceUnit]:
    order = sorted(_dedupe(units), key=lambda u: u.position, reverse=True)
    return _pack(order, budget_tokens)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def relevance_select(
    question: str, candidate_answer: str, units: Sequence[EvidenceUnit], budget_tokens: int
) -> list[EvidenceUnit]:
    """BM25 similarity to the question only (does not look at the answer)."""
    deduped = _dedupe(units)
    if not deduped:
        return []
    bm25 = BM25Okapi([_tokenize(u.text) for u in deduped])
    scores = bm25.get_scores(_tokenize(question))
    order = [u for _, u in sorted(zip(scores, deduped), key=lambda p: p[0], reverse=True)]
    return _pack(order, budget_tokens)


# ---------------------------------------------------------------------- Jev


@dataclass(frozen=True)
class JevSelection:
    selected: list[EvidenceUnit]
    scores: dict[str, EvidenceScore]

    @property
    def selection_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.scores.values())

    @property
    def selection_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.scores.values())


def jev_select(
    question: str, candidate_answer: str, units: Sequence[EvidenceUnit], budget_tokens: int, scorer: JevScorer
) -> JevSelection:
    """Rank by usefulness per token, pack greedily, restore original order."""
    deduped = _dedupe(units)
    scores = {s.unit_id: s for s in scorer.score_many(question, candidate_answer, [(u.id, u.text) for u in deduped])}
    order = sorted(
        deduped,
        key=lambda u: scores[u.id].usefulness / max(u.tokens, 1),
        reverse=True,
    )
    return JevSelection(selected=_pack(order, budget_tokens), scores=scores)
