import sys
from types import SimpleNamespace

import pytest

from qera.adapters.appworld import convert_episode, protect_python_actions
from qera.render import build_verifier_input
from qera.schema import Role, Trajectory
from verification2.benchmarks import scienceworld_worker, text_loop


def test_appworld_python_action_survives_every_view_and_cached_conversion():
    code = "print(apis.example.lookup())"
    episode = {
        "task_id": "example",
        "instruction": "Look up a price.",
        "success": True,
        "steps": [{"code": code, "output": "Price is 12."}],
    }
    trajectory = convert_episode(episode, "test-model", "test")
    segment = next(s for s in trajectory.segments if s.call)
    assert segment.call_role == Role.A
    segment.call_role_override = None
    protect_python_actions(trajectory)
    for view in ("QA", "QEA", "QRA", "QERA"):
        rendered = build_verifier_input(trajectory, view).text
        assert code in rendered
        assert ("Price is 12." in rendered) == ("E" in view)


def test_scienceworld_native_goal_is_shared_and_room_is_evidence(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        envStepLimit=100,
        load=lambda *a, **kw: None,
        reset=lambda: (
            "Room contains a kettle.\nScore: 0",
            {"taskDesc": "Boil water."},
        ),
        server=SimpleNamespace(getCompleted=lambda: True),
    )
    monkeypatch.setitem(
        sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=lambda: fake)
    )
    monkeypatch.setattr(scienceworld_worker, "enable_java", lambda: None)

    def run(job, messages, tools, handle, limit):
        assert "Boil water." in messages[1]["content"]
        assert "Room contains a kettle." in messages[1]["content"]
        assert limit == 101

    monkeypatch.setattr(scienceworld_worker, "run", run)
    job = SimpleNamespace(
        task=SimpleNamespace(task_id="1", domain="boil"),
        model="test-model",
        output=tmp_path,
    )
    scienceworld_worker.collect(job)
    trajectory = Trajectory.model_validate_json(
        (tmp_path / "trajectory.json").read_text()
    )
    for view in ("QA", "QEA", "QRA", "QERA"):
        rendered = build_verifier_input(trajectory, view).text
        assert "Boil water." in rendered and "Score: 0" not in rendered
        assert ("Room contains a kettle." in rendered) == ("E" in view)


def test_budget_exhaustion_is_unscored(monkeypatch):
    monkeypatch.setattr(
        text_loop,
        "_complete",
        lambda *a: {"choices": [{"message": {"content": "look"}}]},
    )
    with pytest.raises(text_loop.EpisodeLimitReached):
        text_loop.run(
            None, [], [text_loop.tool("act", "Act.")], lambda *a: ("Room.", False), 1
        )
