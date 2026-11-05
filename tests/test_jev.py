import json

from qera.builder import TrajectoryBuilder
from qera.cli import write_jsonl
from qera.render import build_verifier_input, selector_view
from qera.schema import Label, SolverInfo
from verification2.jev import cli as jev_cli
from verification2.jev.client import JevAnswer, JevResult
from verification2.jev.scorer import (
    MAX_STATE_TOKENS,
    CandidateScore,
    JevScorer,
    build_state,
)
from verification2.jev.select import pack, select_for_view
from verification2.pipeline import load_selections


def make_traj(traj_id="t0", benchmark="tau2_bench", evidence=None):
    b = TrajectoryBuilder()
    b.add("Q", "user", "question", "What colour is Unit 7 registered as?")
    for i, text in enumerate(
        evidence
        or ["Registry: Unit 7 is blue.", "Weather: sunny.", "Registry: Unit 7 is blue."]
    ):
        b.add(
            "E",
            "environment",
            "tool_result",
            text,
            call={"name": "lookup", "arguments": {"unit": 7}},
        )
    b.add("R", "solver", "reasoning", "I think it is red.")
    b.add("A", "solver", "final_answer", "red")
    return b.build(
        traj_id=traj_id,
        benchmark=benchmark,
        domain="tool_use",
        task_id="7",
        solver=SolverInfo(model="synthetic"),
        label=Label(correct=False, source="exact_match"),
        hidden={"gold_answer": "DO_NOT_SEND"},
    )


class FakeScorer:
    """Deterministic stand-in: usefulness by keyword, no API."""

    def score_view(self, view):
        out = []
        for c in view["candidates"]:
            contra = 0.9 if "blue" in c["text"] else 0.05
            out.append(
                CandidateScore(
                    c["seg_id"],
                    0.1,
                    contra,
                    input_tokens=100,
                    cost_usd=4.2e-6,
                    latency_s=0.1,
                    irrelevant=0.85 - contra if contra < 0.5 else 0.0,
                )
            )
        return out


def test_select_respects_budget_dedupes_and_sorts():
    traj = make_traj()
    view = selector_view(traj)
    big = sum(c["n_tokens"] for c in view["candidates"])
    res = select_for_view(view, big, FakeScorer())
    # two "blue" candidates are exact duplicates: only the first copy is eligible
    assert len(res.selected_e) == 2
    assert res.selected_e == sorted(res.selected_e)
    assert res.used_tokens <= big
    assert res.selection_input_tokens == 300
    assert res.scores[0].stance == "contradicts"


def test_select_prefers_useful_evidence_under_tight_budget():
    traj = make_traj()
    view = selector_view(traj)
    blue = next(c for c in view["candidates"] if "blue" in c["text"])
    res = select_for_view(view, blue["n_tokens"], FakeScorer())
    assert res.selected_e == [blue["seg_id"]]


def test_zero_budget_selects_nothing():
    res = select_for_view(selector_view(make_traj()), 0, FakeScorer())
    assert res.selected_e == [] and res.used_tokens == 0


def test_pack_never_exceeds_budget():
    cands = [
        {"seg_id": i, "text": f"e{i}", "call": None, "n_tokens": 10} for i in range(5)
    ]
    scores = [CandidateScore(i, 0.9, 0.0, 0, 0.0, 0.0) for i in range(5)]
    ids, used = pack(cands, scores, 25)
    assert len(ids) == 2 and used == 20


def test_output_row_is_accepted_by_pipeline_and_renderer(tmp_path):
    traj = make_traj()
    view = selector_view(traj)
    res = select_for_view(view, 40, FakeScorer())
    path = tmp_path / "sel.jsonl"
    path.write_text(json.dumps(res.row()) + "\n")
    selections = load_selections(path)
    assert selections[traj.traj_id] == res.selected_e
    vi = build_verifier_input(
        traj, "QEsA", selected_e=selections[traj.traj_id], budget=40
    )
    assert "DO_NOT_SEND" not in vi.text
    assert "I think it is red" not in vi.text


def test_build_state_truncates_oversized_evidence():
    traj = make_traj(evidence=["word " * (MAX_STATE_TOKENS + 5000)])
    view = selector_view(traj)
    state, truncated = build_state(view, view["candidates"][0])
    assert truncated and state["evidence"].endswith("...]")
    assert set(state) == {"task_type", "task", "candidate_output", "evidence"}


def test_build_state_includes_call_header_and_benchmark_hint():
    view = selector_view(make_traj(benchmark="swebench_verified"))
    state, truncated = build_state(view, view["candidates"][0])
    assert not truncated
    assert state["evidence"].startswith("call: lookup(")
    assert "patch" in state["task_type"]


class FakeClient:
    model = "fake/jev"
    calls = 0

    def __init__(self, *a, **k):
        pass

    def decide(self, state, questions):
        FakeClient.calls += 1
        contra = 0.9 if "blue" in state["evidence"] else 0.05
        return JevResult(
            model=self.model,
            answers={
                "role": JevAnswer(
                    "choice",
                    choice="refutes" if contra > 0.5 else "irrelevant",
                    probabilities={
                        "irrelevant": 0.85 - contra if contra < 0.5 else 0.0,
                        "context": 0.05,
                        "confirms": 0.1,
                        "refutes": contra,
                    },
                ),
            },
            input_tokens=100,
            output_tokens=0,
            cost_usd=4.2e-6,
            latency_s=0.01,
            raw={},
        )

    def close(self):
        pass


def test_scorer_cache_avoids_repeat_calls(tmp_path):
    FakeClient.calls = 0
    scorer = JevScorer(FakeClient(), cache_dir=tmp_path / "cache", workers=2)
    view = selector_view(make_traj())
    first = scorer.score_view(view)
    second = scorer.score_view(view)
    # 3 candidates but two are identical -> 2 distinct states; second pass is all cache hits
    assert FakeClient.calls == 2
    assert all(s.cached for s in second) and not any(s.cached for s in first[:1])
    assert [s.contradicts for s in first] == [s.contradicts for s in second]


def test_cli_select_writes_selections_and_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(jev_cli, "JevClient", FakeClient)
    data = tmp_path / "in.jsonl"
    write_jsonl([make_traj("a"), make_traj("b")], str(data))
    out = tmp_path / "sel" / "x.jsonl"
    jev_cli.main(
        [
            "select",
            "--input",
            str(data),
            "--out",
            str(out),
            "--budget-ratio",
            "0.5",
            "--no-cache",
        ]
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["traj_id"] for r in rows] == ["a", "b"]
    assert all(r["used_tokens"] <= r["budget"] for r in rows)
    scores = [
        json.loads(line)
        for line in (tmp_path / "sel" / "x.scores.jsonl").read_text().splitlines()
    ]
    assert len(scores) == 6 and {s["stance"] for s in scores} == {
        "contradicts",
        "neutral",
    }
    assert load_selections(out)["a"] == rows[0]["selected_e"]
