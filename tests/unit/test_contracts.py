import pytest

from qera import reasoning
from qera.builder import TrajectoryBuilder
from qera.overrides import apply
from qera.render import CONDITIONS, build_verifier_input
from qera.schema import Label, Role, SolverInfo
from tests.support import SENTINEL, _toy


@pytest.mark.unit
def test_no_hidden_or_label_reaches_verifier():
    t = _toy()
    for cond in CONDITIONS:
        assert SENTINEL not in build_verifier_input(t, cond).text, cond


@pytest.mark.unit
def test_unclosed_think_is_reasoning_not_answer():
    think, rest, unclosed = reasoning.split_think(
        "<think>unfinished internal reasoning"
    )
    assert think == "unfinished internal reasoning" and rest == "" and unclosed


@pytest.mark.unit
def test_declared_reasoning_is_checked_against_content():
    b = TrajectoryBuilder()
    b.add(Role.Q, "user", "goal", "g")
    b.add(Role.E, "environment", "observation", "o")
    b.add(Role.A, "solver", "final_answer", "a")
    t = b.build(
        traj_id="x",
        benchmark="b",
        domain="software_web",
        task_id="1",
        solver=SolverInfo(model="m", native_reasoning="full"),
        label=Label(correct=True, source="unit_tests"),
    )
    assert (
        t.solver.native_reasoning == "none"
        and t.meta["native_reasoning_declared"] == "full"
    )


@pytest.mark.unit
def test_b_role_and_spans_replace_each_other():
    t = _toy()
    key = (t.traj_id, 1)
    text = t.segments[1].text
    (t,) = apply([t], {key: {"spans": [[0, len(text), "E"]], "source": "classifier"}})
    (t,) = apply([t], {key: {"role": "R", "source": "manual"}})
    assert t.segments[1].spans is None
    assert text not in build_verifier_input(t, "QA").text
    (t,) = apply([t], {key: {"spans": [[0, len(text), "Q"]], "source": "manual"}})
    assert t.segments[1].role_override is None
    assert text in build_verifier_input(t, "QA").text


@pytest.mark.unit
def test_b_reasoning_metadata_recomputed_after_override():
    t = _toy()
    t.solver.native_reasoning = "summary"
    t.meta["native_reasoning_declared"] = "summary"
    key = (t.traj_id, 3)
    (t,) = apply([t], {key: {"role": "E", "source": "manual"}})
    assert t.solver.native_reasoning == "none"
    assert t.meta["native_reasoning_declared"] == "summary"
    text = t.segments[3].text
    (t,) = apply([t], {key: {"spans": [[0, len(text), "R"]], "source": "manual"}})
    assert t.solver.native_reasoning == "summary"
    assert t.meta["native_reasoning_declared"] == "summary"


@pytest.mark.unit
def test_i_qea_without_selection_preserves_all_evidence():
    t = _toy()
    vi = build_verifier_input(t, "QEA")
    for s in t.by_role("E"):
        assert s.seg_id in vi.included_seg_ids and s.text in vi.text
        if s.call is not None:
            assert s.call.render() in vi.text
