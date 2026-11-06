import json
from pathlib import Path

import pytest

from qera.adapters import swebench, tau2
from qera.jsonl import check_trajectory
from qera.render import build_verifier_input
from qera.schema import Role, Trajectory

FX = Path(__file__).parents[1] / "fixtures"
TAU2 = FX / "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"


def _roundtrip(t: Trajectory) -> Trajectory:
    t2 = Trajectory.model_validate_json(t.model_dump_json())
    assert t2 == t
    return t2


def _no_leak(t: Trajectory) -> None:
    gold = t.hidden.get("gold_answer")
    full = build_verifier_input(t, "QERA").text
    assert "is_supporting" not in full and "evidence_hits" not in full
    if gold:
        for s in t.segments:
            if s.role == Role.Q:
                assert gold.lower() not in s.text.lower()


def _conditions_ok(t: Trajectory) -> None:
    full = build_verifier_input(t, "QERA")
    qa = build_verifier_input(t, "QA")
    qea = build_verifier_input(t, "QEA")
    assert len(qa.text) <= len(qea.text) <= len(full.text)
    assert (
        set(qa.included_seg_ids)
        <= set(qea.included_seg_ids)
        <= set(full.included_seg_ids)
    )
    assert full.included_seg_ids == sorted(full.included_seg_ids)


def _check(ts):
    assert ts
    for t in ts:
        t = _roundtrip(t)
        assert check_trajectory(t) == []
        _no_leak(t)
        _conditions_ok(t)
    return ts


@pytest.mark.integration
def test_tau2_real_sample():
    ts = _check(list(tau2.convert_file(str(TAU2))))
    t = ts[0]
    assert t.segments[0].kind == "policy" and "<main_policy>" in t.segments[0].text
    kinds = {s.kind for s in t.segments}
    assert "user_observation" in kinds
    for s in t.segments:
        if s.call and s.role == Role.A:
            assert (
                s.call.tool_type == "WRITE" or s.call.name == "transfer_to_human_agents"
            )
        if s.role == Role.Q:
            assert "###STOP###" not in s.text


@pytest.mark.integration
def test_swebench_real_sample():
    (t,) = _check(
        [
            swebench.convert_traj(
                str(FX / "swe_trajs/sympy__sympy-13372.traj.json"),
                True,
                "gemini-3-5-flash",
                "hidden",
            )
        ]
    )
    assert t.label.correct is True
    assert t.segments[0].kind == "issue" and "<instructions>" not in t.segments[0].text
    (a,) = t.by_role("A")
    assert a.kind == "patch" and a.text.startswith("diff --git")
    assert all(
        "COMPLETE_TASK_AND_SUBMIT" not in json.dumps(s.call.arguments)
        for s in t.by_role("E")
    )


@pytest.mark.integration
def test_overrides_and_greeting(tmp_path):
    from qera.overrides import apply
    from tests.support import overrides

    ts = list(tau2.convert_file(str(TAU2)))
    t = ts[0]
    assert all(s.text != tau2.GREETING for s in t.segments)
    target = next(s for s in t.segments if s.kind == "agent_reply")
    ov = tmp_path / "ov.jsonl"
    ov.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": target.seg_id,
                "role": "R",
                "source": "classifier",
            }
        )
        + "\n"
        + json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": target.seg_id,
                "role": "E",
                "source": "manual",
            }
        )
        + "\n"
    )
    (t2,) = list(apply([t], overrides(ov)))
    s = t2.segments[target.seg_id]
    assert s.role == Role.E and s.role_default == Role.A and s.role_source == "manual"
    assert target.seg_id not in build_verifier_input(t2, "QA").included_seg_ids


@pytest.mark.integration
def test_tau2_reasoning_from_raw_data(tmp_path):
    src = FX / "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"
    d = json.loads(src.read_text())
    sim = d["simulations"][0]
    m = next(
        mm
        for mm in sim["messages"]
        if mm["role"] == "assistant" and mm.get("tool_calls")
    )
    m["raw_data"] = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "User reports no service; check line status first."
                }
            }
        ]
    }
    d["simulations"] = [sim]
    dst = tmp_path / "qwen3-thinking_telecom_default_gpt-4.1-2025-04-14_1trials.json"
    dst.write_text(json.dumps(d))
    (t,) = list(tau2.convert_file(str(dst)))
    rs = [s for s in t.by_role("R") if s.kind == "reasoning"]
    assert rs and "check line status" in rs[0].text
    assert t.solver.native_reasoning == "full" and t.solver.origin == "self_generated"


@pytest.mark.integration
def test_jsonl_unicode_line_separator(tmp_path):
    from qera import jsonl

    p = tmp_path / "x.jsonl"
    rec = {"output": "line one\u2028line two\x85end"}
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert "\u2028" in p.read_text(encoding="utf-8")
    assert list(jsonl.read(p)) == [rec]
