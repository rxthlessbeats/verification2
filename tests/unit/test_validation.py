import json
from io import BytesIO
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from qera.schema import Label, Override, Trajectory
from tests.support import sample
from verification2.agents.verifier import Backend, main
from verification2.core.config import (
    ClassificationConfig,
    NativeRunConfig,
    SolverConfig,
    VerifierConfig,
)
from verification2.core.models import VerifierResult


@pytest.mark.unit
@pytest.mark.parametrize(
    "flags",
    [
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--timeout", "0"],
        ["--max-tokens", "0"],
        ["--conditions", "QERA", "QERA"],
    ],
)
def test_invalid_settings_fail_before_files_or_api_calls(tmp_path, flags):
    output = tmp_path / "invalid"
    with patch("urllib.request.urlopen") as transport, pytest.raises(ValidationError):
        main(["verify", "--input", "missing.jsonl", "--output", str(output), *flags])
    transport.assert_not_called()
    assert not output.exists()


@pytest.mark.unit
def test_settings_reject_unknown_fields_and_keep_optional_limits_unset(tmp_path):
    args = {"input": ["sample.jsonl"], "output": tmp_path}
    config = VerifierConfig(**args)
    assert config.conditions == ["QERA"] and config.timeout is config.max_tokens is None
    assert config.model_dump(mode="json")["input"] == ["sample.jsonl"]
    for change in ({"max_token": 10}, {"max_tokens": True}, {"timeout": True}):
        with pytest.raises(ValidationError):
            VerifierConfig(**args, **change)
    solver = {
        "model": "solver",
        "endpoint": "https://example.invalid/chat/completions",
        "domain": "retail",
        "task_ids": ["0"],
        "output": tmp_path,
    }
    assert SolverConfig(**solver).max_steps is None
    for change in ({"api_key_env": "SOLVER_API_KEY"}, {"endpoint": "file:///tmp/api"}):
        with pytest.raises(ValidationError):
            SolverConfig(**(solver | change))
    ClassificationConfig(
        **args, model="typesafe-ai/jev", endpoint="https://example.invalid/systemone"
    )
    for change in ({"task_ids": ["0", "0"]}, {"num_trials": 0}, {"max_steps": -1}):
        with pytest.raises(ValidationError):
            SolverConfig(**(solver | change))


@pytest.mark.unit
def test_backend_secrets_are_excluded_from_repr_and_serialization():
    secret = "never-record-this-key"
    backend = Backend(
        mode="http", endpoint="https://example.invalid/chat/completions", api_key=secret
    )
    assert secret not in repr(backend) + backend.model_dump_json()
    assert "api_key" not in backend.model_dump()
    assert backend.api_key.get_secret_value() == secret


@pytest.mark.unit
@pytest.mark.parametrize("label", ["false", "true", 0, 1])
def test_labels_cannot_be_coerced_to_boolean(label):
    with pytest.raises(ValidationError):
        Label(correct=label, source="env_reward")


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value", [("seg_id", True), ("seg_id", "0"), ("step", -1)]
)
def test_trajectory_indices_are_strict_nonnegative_integers(field, value):
    record = sample().model_dump(mode="json")
    record["segments"][0][field] = value
    with pytest.raises(ValidationError):
        Trajectory.model_validate(record)


@pytest.mark.unit
@pytest.mark.parametrize(
    "change",
    [
        {"seg_id": "1"},
        {"seg_id": True},
        {"role": "B"},
        {"spans": []},
        {"role": None},
        {"spans": [[0, True, "E"]]},
        {"role": "E", "spans": [[0, 3, "E"]]},
        {"source": "automatic"},
        {"unused": "typo"},
    ],
)
def test_overrides_validate_ids_roles_and_spans(change):
    record = {"traj_id": "sample", "seg_id": 1, "role": "E"} | change
    with pytest.raises(ValidationError):
        Override.model_validate(record)


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        {"choices": []},
        {"choices": [{"message": {"content": 42}}]},
        {"choices": [{"message": {"content": None}}]},
        {"content": [{"type": "text"}]},
    ],
)
def test_malformed_provider_responses_preserve_usage_and_raw_diagnostics(raw):
    raw["usage"] = {
        "custom_input_count": 17,
        "cost": "0.123",
        "provider_detail": {"cached": 2},
    }
    endpoint = "https://example.invalid/" + (
        "messages" if "content" in raw else "chat/completions"
    )
    with patch(
        "urllib.request.urlopen", return_value=BytesIO(json.dumps(raw).encode())
    ):
        result = Backend(mode="http", endpoint=endpoint).call({"model": "test"})
    assert (
        result["status"] == "error"
        and result["usage"] == raw["usage"]
        and result["raw"] == raw
    )


@pytest.mark.unit
def test_successful_results_require_a_verdict_and_preserve_provider_metadata():
    values = {
        "benchmark": "tau2",
        "traj_id": "t",
        "condition": "QERA",
        "label_correct": False,
        "status": "ok",
        "usage": {"tokens": 17, "cost": "0.0002"},
    }
    with pytest.raises(ValidationError):
        VerifierResult(**values)
    result = VerifierResult(**values, verdict="incorrect", reason="Wrong artifact")
    assert result.model_dump()["usage"] == values["usage"]
    with pytest.raises(ValidationError):
        VerifierResult(**values, verdict="incorrect", reason="x", total_ms=float("nan"))


@pytest.mark.unit
def test_native_manifest_is_validated_before_importing_the_benchmark(tmp_path):
    from verification2.benchmarks.tau2_worker import collect

    config = {
        "tau2_version": "1.0.1",
        "output": tmp_path,
        "run_id": "test",
        "model": "solver",
        "user_model": "solver",
        "judge_model": "solver",
        "domain": "retail",
        "endpoint": "https://example.invalid/chat/completions",
        "task_ids": ["0"],
        "num_trials": 1,
        "seed": 42,
        "complete": False,
    }
    assert NativeRunConfig.model_validate(config).output == tmp_path
    with patch("importlib.metadata.version") as version, pytest.raises(ValidationError):
        collect(config | {"task_ids": ["0", "0"]})
    version.assert_not_called()
    assert not (tmp_path / "native").exists()
