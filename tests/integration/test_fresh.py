import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from qera.jsonl import read_jsonl, write_jsonl
from tests.support import CHAT, fresh_splits, registry, sample, solve
from tests.support import fake_native_episode as fresh
from verification2.agents.classification import classification_view
from verification2.agents.verifier import Backend, main
from verification2.benchmarks import web_setup
from verification2.core.prompts import load_prompts
from verification2.core.registry import resolve
from verification2.data.split import BENCHMARKS, select, split_id
from verification2.experiments import collection, protocol


@pytest.mark.integration
def test_response_reuse_is_endpoint_scoped_atomic_and_revalidates_the_answer(tmp_path):
    raw = {
        "choices": [{"message": {"content": '{"verdict":"correct","reason":"saved"}'}}],
        "usage": {"cost": "0.01", "input_tokens": 13},
    }
    backend = Backend(
        mode="http",
        endpoint="https://example.invalid/chat/completions",
        cache_dir=tmp_path,
    )
    request = {"model": "m", "messages": [{"role": "user", "content": "frozen"}]}
    with patch(
        "urllib.request.urlopen",
        side_effect=lambda *a, **kw: BytesIO(json.dumps(raw).encode()),
    ) as wire:
        first = backend.call(request)
        second = backend.call(request)
        assert wire.call_count == 1
        assert (
            not first["cache_hit"]
            and second["cache_hit"]
            and first["cache_key"] == second["cache_key"]
        )
        assert first["usage"] == second["usage"] == raw["usage"]
        backend.call(request | {"model": "different"})
        assert wire.call_count == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.integration
def test_failed_answers_are_not_cached_and_concurrent_identical_calls_are_reused(
    tmp_path,
):
    request = {"model": "m"}
    backend = Backend(
        mode="http",
        endpoint="https://example.invalid/chat/completions",
        cache_dir=tmp_path,
    )
    invalid = {
        "choices": [{"message": {"content": "bad answer"}}],
        "usage": {"output_tokens": 17},
    }
    with patch(
        "urllib.request.urlopen",
        side_effect=lambda *a, **kw: BytesIO(json.dumps(invalid).encode()),
    ):
        assert backend.call(request)["status"] == "error"
    assert not list(tmp_path.glob("*.json"))
    valid = {
        "choices": [
            {"message": {"content": '{"verdict":"incorrect","reason":"valid"}'}}
        ]
    }
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **kw: BytesIO(json.dumps(valid).encode()),
        ) as wire,
        ThreadPoolExecutor(max_workers=2) as workers,
    ):
        results = list(workers.map(lambda _: backend.call(request), range(2)))
    assert wire.call_count == 1 and sum(r["cache_hit"] for r in results) == 1


@pytest.mark.integration
def test_quarantine_does_not_hide_usable_records_or_send_bad_rows_to_a_model(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl([sample()], str(source))
    source.write_text(
        source.read_text() + '{"schema_version":"1.1","label":{"correct":"false"}}\n'
    )
    output = tmp_path / "verified"
    with pytest.raises(SystemExit) as status:
        main(
            [
                "verify",
                "--input",
                str(source),
                "--output",
                str(output),
                "--backend",
                "mock",
            ]
        )
    assert status.value.code == 1
    assert len((output / "results.jsonl").read_text().splitlines()) == 1
    quarantined = json.loads((output / "quarantine.jsonl").read_text())
    assert quarantined["line"] == 2 and quarantined["sha256"]
    assert json.loads((output / "manifest.json").read_text())["quarantined_n"] == 1
    assert '"correct":"false"' not in (output / "requests.jsonl").read_text()


@pytest.mark.integration
def test_completed_solver_episodes_are_reused_and_modified_tasks_fail_before_calls(
    tmp_path, monkeypatch
):
    splits = fresh_splits(tmp_path, monkeypatch)
    runner = Mock(side_effect=fresh)
    monkeypatch.setattr(collection, "collect_job", runner)
    output = tmp_path / "collection"
    assert solve(splits, output, backend="http") == 0
    before = (output / "trajectories.jsonl").read_bytes()
    assert solve(splits, output, backend="http") == 0
    assert runner.call_count == 3
    assert (output / "trajectories.jsonl").read_bytes() == before
    assert all(
        t.solver.origin == "self_generated"
        for t in read_jsonl([output / "trajectories.jsonl"])
    )
    task = select(splits, "pilot")[0]
    (tmp_path / task.source).write_text("[]")
    with pytest.raises(ValueError, match="missing"):
        solve(splits, tmp_path / "changed", backend="http")
    assert runner.call_count == 3


@pytest.mark.integration
def test_failed_native_episodes_never_become_incorrect_labels_or_retry_automatically(
    tmp_path, monkeypatch
):
    splits = fresh_splits(tmp_path, monkeypatch)
    runner = Mock(side_effect=RuntimeError("native infrastructure failed"))
    monkeypatch.setattr(collection, "collect_job", runner)
    output = tmp_path / "collection"
    assert solve(splits, output, backend="http") == 1
    assert not (output / "trajectories.jsonl").exists()
    assert solve(splits, output, backend="http") == 1
    assert runner.call_count == 3
    monkeypatch.setattr(collection, "collect_job", fresh)
    assert solve(splits, output, backend="http", retry_failed=True) == 0
    assert solve(splits, output, backend="http") == 0
    assert len(list((output / "episodes").glob("*.attempt1"))) == 3


@pytest.mark.integration
def test_heldout_requires_frozen_protocol_and_rejects_prompt_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        protocol, "runtime_hashes", lambda **kwargs: {"native": "test-lock"}
    )
    splits = fresh_splits(tmp_path, monkeypatch)
    output = tmp_path / "protocol.json"
    endpoint = "https://example.invalid/chat/completions"
    models = {
        "test-model": {"chat_completions": CHAT},
        "verify": {"chat_completions": endpoint},
        "typesafe-ai/jev": {"systemone": "https://example.invalid/v1/systemone"},
    }
    monkeypatch.setenv("MODELS_FILE", str(registry(tmp_path, models)))
    monkeypatch.setenv("VERIFIER_MODEL", "verify")
    monkeypatch.setenv("JEV_MODEL", "typesafe-ai/jev")
    solver, prompts = resolve("solver", "test-model"), load_prompts()
    with pytest.raises(ValueError, match="requires --protocol"):
        protocol.check_heldout(splits, "test", None, solver, prompts)
    protocol.main(["--benchmarks", *BENCHMARKS, "--output", str(output)])
    assert protocol.check_heldout(splits, "test", output, solver, prompts)
    edited = prompts.model_copy(deep=True)
    edited.benchmarks["tau2_bench"].solver_instruction = "Edited after freeze."
    with pytest.raises(ValueError, match="Solver model, reasoning effort or prompt"):
        protocol.check_heldout(splits, "test", output, solver, edited)
    with pytest.raises(ValueError, match="frozen split"):
        protocol.check_heldout(splits[:1], "test", output, solver, prompts)
    trajectory = sample()
    trajectory.split = "test"
    trajectory.meta["split_id"] = split_id(splits)
    args = SimpleNamespace(
        protocol=output,
        model="verify",
        endpoint=endpoint,
        reasoning_effort=None,
        conditions=["QA", "QEA", "QRA", "QERA"],
        max_tokens=None,
    )
    protocol.check_stage([trajectory], args, load_prompts(), "verifier")
    changed = load_prompts().model_copy(deep=True)
    changed.verifier.instruction += " edited after freeze"
    with pytest.raises(ValueError, match="differ"):
        protocol.check_stage([trajectory], args, changed, "verifier")
    args.max_tokens = 10
    with pytest.raises(ValueError, match="output maximum"):
        protocol.check_stage([trajectory], args, load_prompts(), "verifier")


@pytest.mark.integration
def test_jev_benchmark_guidance_is_resolved_without_adapter_label_hints(tmp_path):
    trajectory = sample()
    trajectory.benchmark = "swebench_verified"
    view = classification_view(trajectory)
    assert all(
        "kind" not in segment and "role_default" not in segment
        for segment in view["segments"]
    )
    prompts = load_prompts()
    assert (
        prompts.classification("swebench_verified").criteria
        == prompts.classification("agentrewardbench").criteria
    )
    assert (
        prompts.classification("swebench_verified").instruction
        != prompts.classification("agentrewardbench").instruction
    )


@pytest.mark.integration
def test_website_reset_preserves_foreign_containers_and_records_native_failures(
    tmp_path, monkeypatch
):
    state = tmp_path / "websites"
    state.mkdir()
    actual = {"shopping": "foreign-id"}
    monkeypatch.setattr(web_setup, "inspect_containers", lambda: dict(actual))
    native = Mock(side_effect=RuntimeError("native setup failed"))
    monkeypatch.setattr(web_setup, "run", native)
    with pytest.raises(ValueError, match="ownership changed"):
        web_setup.recreate(state, {"containers": {"shopping": "owned-id"}})
    native.assert_not_called()
    with pytest.raises(RuntimeError, match="native setup failed"):
        web_setup.recreate(state, {"containers": actual})
    saved = json.loads((state / "setup.json").read_text())
    assert saved["status"] == "failed" and saved["containers"] == actual


@pytest.mark.integration
def test_formal_collection_requires_the_current_reasoning_aware_pilot(
    tmp_path, monkeypatch
):
    splits = fresh_splits(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pilot-report"):
        protocol.check_pilot(None, splits)
    path = tmp_path / "pilot.json"
    data = {
        "status": "passed",
        "split_id": split_id(splits),
        "source_hash": protocol.code_hash(),
        "runtime_hashes": protocol.runtime_hashes(),
        "readable_reasoning": {b: True for b in collection.BENCHMARKS},
    }
    path.write_text(json.dumps(data))
    protocol.check_pilot(path, splits)
    data["readable_reasoning"]["swebench_verified"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reasoning-aware pilot"):
        protocol.check_pilot(path, splits)
    data["readable_reasoning"] = {"tau2_bench": True}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reasoning-aware pilot"):
        protocol.check_pilot(path, splits[:1])
    data["split_id"] = split_id(splits[:1])
    path.write_text(json.dumps(data))
    protocol.check_pilot(path, splits[:1])


@pytest.mark.integration
def test_doctor_checks_only_what_the_selected_benchmarks_need():
    from verification2.benchmarks.doctor import check

    names = {r["name"] for r in check(benchmarks=["tau2_bench"])["checks"]}
    assert "runtime:tau2" in names and "api:VERIFIER" in names
    assert not names & {"docker", "docker-compose", "chromium", "runtime:swe"}
    assert not any(n.startswith(("web:", "source:arb", "tasks:")) for n in names)
    names = {r["name"] for r in check(benchmarks=["swebench_verified"])["checks"]}
    assert {"docker", "runtime:swe", "tasks:swebench_verified"} <= names
    assert "docker-compose" not in names and "chromium" not in names


@pytest.mark.integration
def test_native_swe_format_feedback_does_not_drop_the_failed_model_response(tmp_path):
    from qera.adapters.swebench import convert_traj

    path = tmp_path / "sample__repo-1.traj.json"
    raw = {
        "messages": [
            {
                "role": "user",
                "content": "<pr_description>Fix the failure.</pr_description>",
            },
            {
                "role": "user",
                "content": "Unknown tool. Use bash.",
                "extra": {
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "content": "I tried this command.",
                                    "reasoning_content": "Recorded reasoning before the formatting error.",
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "wrong_tool",
                                                "arguments": '{"command":"cat a.py"}',
                                            }
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                },
            },
            {"role": "exit", "content": "LimitsExceeded", "extra": {"submission": ""}},
        ]
    }
    path.write_text(json.dumps(raw))
    trajectory = convert_traj(str(path), False, "solver", "full")
    assert any(s.text == "Unknown tool. Use bash." for s in trajectory.by_role("E"))
    assert any("before the formatting error" in s.text for s in trajectory.by_role("R"))
    assert any(s.call and s.call.name == "wrong_tool" for s in trajectory.by_role("R"))


@pytest.mark.integration
def test_swe_grader_requires_explicit_native_outcomes_and_excludes_infrastructure(
    tmp_path,
):
    from verification2.benchmarks.swe_worker import native_score

    directory = tmp_path / "logs/evaluation/native"
    report = directory / "verification2/sample__repo-1/report.json"
    report.parent.mkdir(parents=True)
    task = "sample__repo-1"
    summary = {
        "submitted_ids": [task],
        "empty_patch_ids": [],
        "completed_ids": [task],
        "resolved_ids": [],
        "unresolved_ids": [task],
        "error_ids": [],
        "infra_failure_ids": [],
        "ambiguous_failure_ids": [],
    }
    (directory / "results.json").write_text(json.dumps(summary))
    with pytest.raises(FileNotFoundError):
        native_score(tmp_path, task, "patch")
    report.write_text(json.dumps({task: {"resolved": False, "infra_failure": False}}))
    assert native_score(tmp_path, task, "patch")[0] is False
    for field in ("infra_failure_ids", "ambiguous_failure_ids", "error_ids"):
        (directory / "results.json").write_text(json.dumps(summary | {field: [task]}))
        with pytest.raises(RuntimeError, match="no solver label"):
            native_score(tmp_path, task, "patch")
    (directory / "results.json").write_text(json.dumps(summary))
    report.write_text(json.dumps({task: {"resolved": False, "infra_failure": True}}))
    with pytest.raises(RuntimeError, match="infrastructure failure"):
        native_score(tmp_path, task, "patch")
    report.unlink()
    (directory / "results.json").write_text(
        json.dumps(
            summary
            | {"empty_patch_ids": [task], "completed_ids": [], "unresolved_ids": []}
        )
    )
    correct, detail = native_score(tmp_path, task, "")
    assert correct is False and detail["native_outcome"] == "empty_patch"
