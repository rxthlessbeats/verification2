import json

import pytest

from qera.adapters import swebench, tau2
from qera.builder import TrajectoryBuilder
from qera.jsonl import check_trajectory, read_jsonl, write_jsonl
from qera.overrides import apply
from qera.render import build_verifier_input
from qera.schema import Label, Role, SolverInfo
from tests.support import FX, _toy, overrides


@pytest.mark.integration
def test_spans_split_mixed_utterance(tmp_path):
    t = _toy()
    text = t.segments[1].text
    cut = text.index("The screen")
    ov = tmp_path / "ov.jsonl"
    ov.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": 1,
                "spans": [[0, cut, "Q"], [cut, len(text), "E"]],
                "source": "manual",
            }
        )
        + "\n"
    )
    (t2,) = list(apply([t], overrides(ov)))
    qa = build_verifier_input(t2, "QA").text
    qea = build_verifier_input(t2, "QEA").text
    assert "Please check." in qa and "error 404" not in qa
    assert "error 404" in qea
    assert 1 in build_verifier_input(t2, "QEA").included_seg_ids


@pytest.mark.integration
def test_call_role_hides_solver_written_arguments(tmp_path):
    t = _toy()
    ov = tmp_path / "ov.jsonl"
    ov.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": 2,
                "call_role": "R",
                "source": "classifier",
            }
        )
        + "\n"
    )
    (t2,) = list(apply([t], overrides(ov)))
    assert "I bet it is Ada" in build_verifier_input(t2, "QERA").text
    qea = build_verifier_input(t2, "QEA").text
    assert "I bet it is Ada" not in qea and "call: search(…)" in qea
    assert "Lumen was founded by Ada." in qea


@pytest.mark.integration
def test_override_errors_are_not_silent(tmp_path):
    t = _toy()
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": 1,
                "spans": [[0, 5, "Q"]],
                "source": "manual",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="cover"):
        list(apply([t], overrides(bad)))
    no_a = tmp_path / "no_a.jsonl"
    no_a.write_text(
        json.dumps({"traj_id": t.traj_id, "seg_id": 4, "role": "R", "source": "manual"})
        + "\n"
    )
    with pytest.raises(ValueError, match="missing A"):
        list(apply([_toy()], overrides(no_a)))


@pytest.mark.integration
def test_tau2_inline_think_becomes_reasoning(tmp_path):
    d = json.loads(
        (
            FX / "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"
        ).read_text()
    )
    sim = d["simulations"][0]
    m = next(
        mm
        for mm in sim["messages"]
        if mm["role"] == "assistant" and mm.get("tool_calls")
    )
    m["content"] = "<think>check line first</think>Let me check."
    d["simulations"] = [sim]
    p = tmp_path / "qwen_telecom_default_x_1trials.json"
    p.write_text(json.dumps(d))
    (t,) = list(tau2.convert_file(str(p)))
    assert any(
        s.kind == "reasoning" and s.text == "check line first" for s in t.by_role("R")
    )
    assert all("<think>" not in s.text for s in t.segments)


@pytest.mark.integration
def test_write_is_atomic(tmp_path):
    out = tmp_path / "x.jsonl"
    write_jsonl([_toy()], str(out))
    before = out.read_text()

    def broken():
        yield _toy()
        raise RuntimeError("adapter crashed")

    with pytest.raises(RuntimeError):
        write_jsonl(broken(), str(out))
    assert out.read_text() == before
    assert not (tmp_path / "x.jsonl.tmp").exists()


@pytest.mark.integration
def test_write_revalidates_nested_mutations_before_replacing_data(tmp_path):
    out = tmp_path / "x.jsonl"
    write_jsonl([_toy()], str(out))
    before = out.read_bytes()
    invalid = _toy()
    invalid.label.correct = "false"
    with pytest.raises(ValueError):
        write_jsonl([_toy(), invalid], str(out))
    assert out.read_bytes() == before
    assert not out.with_name("x.jsonl.tmp").exists()


@pytest.mark.integration
def test_a_old_schema_rejected_on_read(tmp_path):
    row = _toy().model_dump()
    row["schema_version"] = "1.0"
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="expected"):
        list(read_jsonl([str(p)]))


@pytest.mark.integration
def test_b_overrides_respect_priority_per_field():
    key = ("toy.1", 2)
    text = _toy().segments[2].text
    (t,) = apply([_toy()], {key: {"call_role": "R", "source": "classifier"}})
    (t,) = apply([t], {key: {"spans": [[0, len(text), "E"]], "source": "manual"}})
    (t,) = apply([t], {key: {"role": "R", "source": "classifier"}})
    assert t.segments[2].call_role == Role.R
    assert t.segments[2].text_roles() == {Role.E}
    (t,) = apply([t.revalidate()], {key: {"role": "R", "source": "manual"}})
    assert t.segments[2].text_roles() == {Role.R} and t.segments[2].spans is None


@pytest.mark.integration
def test_c_call_only_e_is_a_valid_candidate():
    for i, ok in enumerate((True, False)):
        t = _toy(ok)
        t.traj_id = f"call-e.{i}"
        t.task_id = str(i)
        t.segments[2].role_override = Role.R
        t.segments[2].call_role_override = Role.E
        assert t.by_role("E") == []
        assert t.segments[2].call.render() in build_verifier_input(t, "QEA").text
        assert check_trajectory(t) == []


@pytest.mark.integration
@pytest.mark.parametrize("adapter", ["swebench", "tau2"])
def test_h_reversed_tool_results_are_detected(adapter, tmp_path):
    if adapter == "tau2":
        messages = [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "x", "name": "read_x", "arguments": {}},
                    {"id": "y", "name": "read_y", "arguments": {}},
                ],
            },
            {"role": "tool", "id": "y", "content": "result Y"},
            {"role": "tool", "id": "x", "content": "result X"},
            {"role": "assistant", "content": "done"},
        ]
        t = tau2._convert_sim(
            {"task_id": "1", "messages": messages, "reward_info": {"reward": 1}},
            "mock",
            "m",
            {},
            {"mock": {"assistant": {"read_x": "WRITE", "read_y": "WRITE"}}},
            None,
            "none",
            "self_generated",
        )
    else:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "x",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"read X"}',
                        },
                    },
                    {
                        "id": "y",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"read Y"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "y", "content": "result Y"},
            {"role": "tool", "tool_call_id": "x", "content": "result X"},
            {"role": "assistant", "content": "done"},
        ]
        p = tmp_path / "order.traj.json"
        p.write_text(
            json.dumps(
                {
                    "messages": [{"role": "user", "content": "task"}]
                    + messages
                    + [{"role": "exit", "extra": {"submission": "patch"}}]
                }
            )
        )
        t = swebench.convert_traj(str(p), True, "m", "none")
    results = [s for s in t.segments if s.text in ("result Y", "result X")]
    assert [s.text for s in results] == ["result Y", "result X"]
    assert t.meta["call_order_changed"] == 1
    assert all(s.ambiguous and "out of call order" in s.note for s in results)
    assert all(s.ambiguous for s in t.segments if s.call is not None)


@pytest.mark.integration
@pytest.mark.parametrize("order, expected", [("xyz", 0), ("zyx", 2)])
def test_h_order_counter_counts_reordered_results(order, expected):
    b = TrajectoryBuilder()
    b.add("Q", "user", "question", "task")
    for key in "xyz":
        b.record_call(key)
    for key in order:
        result = b.add("E", "environment", "tool_result", key)
        b.record_result(key, result)
    b.add("A", "solver", "final_answer", "answer")
    t = b.build(
        traj_id="counter",
        benchmark="counter",
        domain="tool_use",
        task_id="1",
        solver=SolverInfo(model="m"),
        label=Label(correct=True, source="unit_tests"),
    )
    assert t.meta["call_order_changed"] == expected
    assert all(s.ambiguous == bool(expected) for s in t.by_role("E"))


@pytest.mark.integration
def test_retired_counts_are_ignored_without_changing_frozen_content(tmp_path):
    original = _toy()
    legacy = original.model_dump(mode="json")
    legacy["tokenizer"] = "retired-tokenizer"
    legacy["meta"]["native_reasoning_tokens"] = 999999
    legacy["segments"][2].update(n_tokens=999999, call_n_tokens=999999)
    text = legacy["segments"][3]["text"]
    legacy["segments"][3]["spans"] = [
        {"start": 0, "end": len(text), "role": "R", "n_tokens": 999999}
    ]
    path = tmp_path / "frozen.jsonl"
    source = json.dumps(legacy) + "\n"
    path.write_text(source)
    (loaded,) = read_jsonl([path])
    output = loaded.model_dump(mode="json")
    assert path.read_text() == source
    assert "tokenizer" not in output and "native_reasoning_tokens" not in output["meta"]
    assert (
        "n_tokens" not in output["segments"][2]
        and "call_n_tokens" not in output["segments"][2]
    )
    assert "n_tokens" not in output["segments"][3]["spans"][0]
    for condition in ("QA", "QEA", "QERA"):
        assert (
            build_verifier_input(loaded, condition).text
            == build_verifier_input(original, condition).text
        )
    assert loaded.label == original.label and loaded.hidden == original.hidden
