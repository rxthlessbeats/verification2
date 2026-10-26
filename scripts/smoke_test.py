"""Hit Jev once through the gateway on a toy MuSiQue-style example.

Run:  uv run python scripts/smoke_test.py
Expected: the supporting passage scores high on `supports`, the contradicting
passage scores high on `contradicts`, and the irrelevant one is low on both.
"""

from verification2.jev import JevClient, JevScorer, jev_select
from verification2.jev.select import make_units, random_select, recency_select, relevance_select
from verification2.jev.tokens import count_tokens

QUESTION = "In which year was the university that employs the discoverer of penicillin's biographer founded?"
CANDIDATE = "1826"

PASSAGES = [
    # irrelevant
    "The Amazon River is the largest river by discharge volume of water in the world, flowing through Brazil, Peru, and Colombia.",
    # supports
    "Gwyn Macfarlane, author of the biography 'Alexander Fleming: The Man and the Myth', was a professor at University College London, which was founded in 1826.",
    # contradicts
    "University College London was established in 1836 following a royal charter, making it the third university founded in England.",
    # weakly relevant
    "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital in London.",
]


def main() -> None:
    client = JevClient(run="smoke-test")
    scorer = JevScorer(client, cache_dir=None)
    units = make_units(PASSAGES, count_tokens)

    print(f"model: {client.model}\n")
    sel = jev_select(QUESTION, CANDIDATE, units, budget_tokens=60, scorer=scorer)
    print(f"{'id':<4}{'tok':>5}{'supports':>10}{'contra':>8}{'useful':>8}  stance       text")
    for u in units:
        s = sel.scores[u.id]
        print(
            f"{u.id:<4}{u.tokens:>5}{s.supports:>10.3f}{s.contradicts:>8.3f}{s.usefulness:>8.3f}  {s.stance:<12} {u.text[:60]}..."
        )

    print(f"\nJev selected (budget 60 tok): {[u.id for u in sel.selected]}")
    print(f"selection cost: {sel.selection_input_tokens} input tokens, ${sel.selection_cost_usd:.6f}")
    print(f"random:    {[u.id for u in random_select(QUESTION, CANDIDATE, units, 60)]}")
    print(f"recency:   {[u.id for u in recency_select(QUESTION, CANDIDATE, units, 60)]}")
    print(f"relevance: {[u.id for u in relevance_select(QUESTION, CANDIDATE, units, 60)]}")
    client.close()


if __name__ == "__main__":
    main()
