from verification2.jev.scorer import EvidenceScore
from verification2.jev.select import (
    EvidenceUnit,
    jev_select,
    random_select,
    recency_select,
    relevance_select,
)


def units():
    return [
        EvidenceUnit("e0", "The Eiffel Tower is in Paris.", 10, 0),
        EvidenceUnit("e1", "Bananas are yellow fruit.", 10, 1),
        EvidenceUnit("e2", "Paris is the capital of France.", 10, 2),
        EvidenceUnit("e3", "Paris is the capital of France.", 10, 3),  # duplicate
    ]


class FakeScorer:
    """Deterministic stand-in for JevScorer."""

    def __init__(self, table):
        self.table = table

    def score_many(self, q, a, pairs):
        return [
            EvidenceScore(uid, *self.table[uid], input_tokens=100, cost_usd=0.0000042, latency_s=0.0)
            for uid, _ in pairs
        ]


def test_recency_keeps_latest_within_budget_and_restores_order():
    out = recency_select("q", "a", units(), budget_tokens=20)
    assert [u.id for u in out] == ["e1", "e2"]  # e3 deduped; newest two are e2, e1; restored to original order


def test_random_is_deterministic_given_seed():
    a = random_select("q", "a", units(), 20, seed=1)
    b = random_select("q", "a", units(), 20, seed=1)
    assert [u.id for u in a] == [u.id for u in b]
    assert sum(u.tokens for u in a) <= 20


def test_relevance_prefers_question_overlap():
    out = relevance_select("What is the capital of France?", "Paris", units(), 10)
    assert [u.id for u in out] == ["e2"]


def test_jev_select_ranks_by_usefulness_density_and_tracks_cost():
    scorer = FakeScorer({"e0": (0.2, 0.1), "e1": (0.05, 0.05), "e2": (0.9, 0.05)})
    sel = jev_select("What is the capital of France?", "Paris", units(), 20, scorer)
    assert [u.id for u in sel.selected] == ["e0", "e2"]
    assert sel.selection_input_tokens == 300
    assert sel.scores["e2"].stance == "supports"
    assert sel.scores["e1"].stance == "neutral"


def test_jev_select_respects_budget_and_never_exceeds():
    scorer = FakeScorer({"e0": (0.9, 0.0), "e1": (0.9, 0.0), "e2": (0.9, 0.0)})
    sel = jev_select("q", "a", units(), 25, scorer)
    assert sum(u.tokens for u in sel.selected) <= 25
    assert len(sel.selected) == 2
