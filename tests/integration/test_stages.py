import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qera.adapters.tau2 import convert_file
from qera.jsonl import read_jsonl, write_jsonl
from tests.support import CHAT, decisions, registry, sample
from verification2.agents.classification import classification_view
from verification2.agents.verifier import main
from verification2.benchmarks.tau2_worker import collect


@pytest.mark.integration
def test_failed_or_dry_classification_never_publishes_a_dataset(tmp_path, monkeypatch):
    endpoints = {"systemone": "https://example.invalid/v1/systemone"}
    monkeypatch.setenv("MODELS_FILE", str(registry(tmp_path, {"jev-1.13": endpoints})))
    source = tmp_path / "source.jsonl"
    write_jsonl([sample()], str(source))
    for backend in ("dry-run", "http"):
        output = tmp_path / backend
        raw = {"answers": {}, "usage": {"input_tokens": 100, "output_tokens": 1}}
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=lambda *a, raw=raw, **kw: BytesIO(json.dumps(raw).encode()),
            ),
            pytest.raises(SystemExit) as status,
        ):
            main(
                [
                    "classify",
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--backend",
                    backend,
                    "--model",
                    "jev-1.13",
                ]
            )
        assert status.value.code == (1 if backend == "http" else 0)
        assert not (output / "trajectories.jsonl").exists()
        assert not json.loads((output / "manifest.json").read_text())["complete"]
        if backend == "http":
            result = json.loads((output / "results.jsonl").read_text())
            assert result["raw_response"]["s1_call"] == raw
            assert result["usage"]["input_tokens"] == 100 * len(result["raw_response"])
            assert result["status"] == "error" and "question ID" in result["error"]


@pytest.mark.integration
def test_registry_keys_are_shared_by_provider_without_cross_stage_model_defaults(
    tmp_path,
):
    source = tmp_path / "source.jsonl"
    write_jsonl([sample()], str(source))
    endpoints = {"systemone": "https://opencode.ai/inference/systemone/v1/systemone"}
    models = {
        "jev-1.13": {"api_key_env": "OPENCODE_API_KEY", "endpoints": endpoints},
        "jev-custom": {"api_key_env": "CUSTOM_API_KEY", "endpoints": endpoints},
    }
    path = tmp_path / "models.yaml"
    path.write_text(json.dumps({"models": models}))
    with patch.dict(
        "os.environ",
        {
            "MODELS_FILE": str(path),
            "OPENCODE_API_KEY": "shared",
            "JEV_MODEL": "jev-1.13",
            "VERIFIER_MODEL": "verify-only",
        },
        clear=True,
    ):
        with pytest.raises(ValueError, match="CUSTOM_API_KEY"):
            main(
                [
                    "classify",
                    "--input",
                    "missing.jsonl",
                    "--output",
                    str(tmp_path / "missing"),
                    "--backend",
                    "http",
                    "--model",
                    "jev-custom",
                ]
            )
        view = classification_view(sample())
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=lambda wire, timeout: BytesIO(
                    json.dumps(
                        decisions(view, json.loads(wire.data)["questions"])
                    ).encode()
                ),
            ) as transport,
            pytest.raises(SystemExit),
        ):
            main(
                [
                    "classify",
                    "--input",
                    str(source),
                    "--output",
                    str(tmp_path / "shared"),
                    "--backend",
                    "http",
                ]
            )
        assert (
            transport.call_args.args[0].get_header("Authorization") == "Bearer shared"
        )
        assert json.loads(transport.call_args.args[0].data)["model"] == "jev-1.13"


@pytest.mark.integration
@pytest.mark.parametrize("explicit_limits", [False, True])
def test_native_worker_routes_solver_user_and_judge_without_saving_keys(
    tmp_path, explicit_limits
):
    requests = []

    def completion(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"usage": {"prompt_tokens": 10}})

    llm = SimpleNamespace(
        completion=completion, litellm=SimpleNamespace(disable_cache=lambda: None)
    )

    def run_domain(config):
        assert config.max_concurrency == 1 and config.max_retries == 0
        assert hasattr(config, "max_steps") is explicit_limits
        for model in (config.llm_agent, config.llm_user, "gpt-4.1"):
            llm.completion(
                model=model, messages=[{"role": "user", "content": "native input"}]
            )

    environment = SimpleNamespace(tools="assistant", user_tools=None)
    modules = {
        "tau2": SimpleNamespace(),
        "tau2.data_model.simulation": SimpleNamespace(
            TextRunConfig=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
        "tau2.environment.toolkit": SimpleNamespace(
            get_tool_types=lambda tools: {"cancel": SimpleNamespace(value="write")}
        ),
        "tau2.registry": SimpleNamespace(
            registry=SimpleNamespace(
                get_env_constructor=lambda domain: lambda: environment
            )
        ),
        "tau2.run": SimpleNamespace(run_domain=run_domain),
        "tau2.utils": SimpleNamespace(llm_utils=llm),
    }
    config = {
        "tau2_version": "1.0.1",
        "output": str(tmp_path),
        "model": "solver-m",
        "user_model": "user-m",
        "judge_model": "judge-m",
        "endpoint": "https://opencode.ai/inference/go/openai/v1/chat/completions",
        "run_id": "native-run",
        "domain": "retail",
        "task_ids": ["0"],
        "num_trials": 1,
        "max_steps": 100 if explicit_limits else None,
        "seed": 42,
        "timeout": 60 if explicit_limits else None,
        "max_tokens": 4096 if explicit_limits else None,
    }
    with (
        patch.dict(sys.modules, modules),
        patch("importlib.metadata.version", return_value="1.0.1"),
        patch.dict("os.environ", {"SOLVER_API_KEY": "solver-only-secret"}),
    ):
        collect(config)
    assert [r["model"] for r in requests] == [
        "openai/solver-m",
        "openai/user-m",
        "openai/judge-m",
    ]
    assert all(r["api_key"] == "solver-only-secret" for r in requests)
    assert all(r["api_base"].endswith("/openai/v1") for r in requests)
    assert all(
        ("max_tokens" in r) is explicit_limits and ("timeout" in r) is explicit_limits
        for r in requests
    )
    assert llm.completion is completion
    logs = (tmp_path / "native/calls.jsonl").read_text()
    assert "solver-only-secret" not in logs
    assert [json.loads(line)["purpose"] for line in logs.splitlines()] == [
        "conversation",
        "conversation",
        "benchmark_judge",
    ]
    assert (
        json.loads((tmp_path / "native/tool_types.json").read_text())["retail"][
            "assistant"
        ]["cancel"]
        == "WRITE"
    )


@pytest.mark.integration
def test_solver_dry_run_is_independent_and_never_publishes_fake_trajectories(tmp_path):
    with (
        patch.dict(
            "os.environ",
            {
                "SOLVER_MODEL": "solver-only",
                "MODELS_FILE": str(
                    registry(
                        tmp_path,
                        {"solver-only": {"chat_completions": CHAT}},
                    )
                ),
                "JEV_MODEL": "jev-only",
                "VERIFIER_MODEL": "verifier-only",
            },
            clear=True,
        ),
        patch("verification2.agents.solver.subprocess.run") as native,
        pytest.raises(SystemExit) as status,
    ):
        main(
            [
                "solve",
                "--domain",
                "retail",
                "--task-ids",
                "0",
                "--output",
                str(tmp_path / "solver"),
            ]
        )
    assert status.value.code == 0
    native.assert_not_called()
    manifest = json.loads((tmp_path / "solver/manifest.json").read_text())
    assert manifest["model"] == "solver-only" and manifest["complete"] is False
    assert manifest["user_model"] == manifest["judge_model"] == manifest["model"]
    assert not (tmp_path / "solver/trajectories.jsonl").exists()


@pytest.mark.integration
def test_fresh_tau_uses_recorded_policy_and_native_tool_types(tmp_path):
    raw = {
        "info": {
            "agent_info": {"llm": "solver-m"},
            "environment_info": {
                "domain_name": "retail",
                "policy": "The exact policy used in this run.",
            },
        },
        "simulations": [
            {
                "task_id": "0",
                "trial": 0,
                "reward_info": {"reward": 1},
                "messages": [
                    {"role": "user", "content": "Cancel the order."},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "c", "name": "new_cancel", "arguments": {}}
                        ],
                    },
                    {"role": "tool", "id": "c", "content": "Canceled."},
                    {"role": "assistant", "content": "Done."},
                ],
            }
        ],
    }
    path = tmp_path / "native.json"
    path.write_text(json.dumps(raw))
    (trajectory,) = convert_file(
        str(path), tool_types={"retail": {"assistant": {"new_cancel": "WRITE"}}}
    )
    assert trajectory.segments[0].text == raw["info"]["environment_info"]["policy"]
    assert any(
        s.kind == "action" and s.call.name == "new_cancel" for s in trajectory.segments
    )
    assert trajectory.solver.origin == "self_generated"


@pytest.mark.integration
def test_solver_converts_native_results_into_a_separate_frozen_artifact(tmp_path):
    root = tmp_path / "tau2"
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    output = tmp_path / "collected"
    fixture = (
        Path(__file__).parents[1]
        / "fixtures/gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"
    )
    raw = json.loads(fixture.read_text())
    raw["simulations"] = raw["simulations"][:1]
    raw["simulations"][0]["trial"] = 0
    task_id = raw["simulations"][0]["task_id"]
    raw["info"]["agent_info"]["llm"] = "openai/fresh-solver"

    def native_call(command, **kwargs):
        if command[0] != "git":
            assert command[1:3] == ["-m", "verification2.benchmarks.tau2_worker"]
            assert kwargs["env"]["SOLVER_API_KEY"] == "native-test-key"
            assert "JEV_API_KEY" not in kwargs["env"]
            assert "AI_GATEWAY_API_KEY" not in kwargs["env"]
            native = output / "native"
            native.mkdir()
            (native / "results.json").write_text(json.dumps(raw))
            from qera.adapters.tau2 import _load_tool_types

            (native / "tool_types.json").write_text(json.dumps(_load_tool_types()))
        return subprocess.CompletedProcess(command, 0)

    models = registry(
        tmp_path, {"fresh-solver": {"chat_completions": CHAT}}, "OPENCODE_API_KEY"
    )
    settings = {
        "MODELS_FILE": str(models),
        "SOLVER_MODEL": "fresh-solver",
        "OPENCODE_API_KEY": "native-test-key",
        "JEV_API_KEY": "other-key",
        "AI_GATEWAY_API_KEY": "jev-gateway-secret",
    }
    with (
        patch.dict("os.environ", settings, clear=True),
        patch(
            "verification2.agents.solver.subprocess.check_output",
            return_value="pinned-revision",
        ),
        patch("verification2.agents.solver.subprocess.run", side_effect=native_call),
        pytest.raises(SystemExit) as status,
    ):
        main(
            [
                "solve",
                "--domain",
                "telecom",
                "--task-ids",
                str(task_id),
                "--output",
                str(output),
                "--backend",
                "http",
                "--tau2-root",
                str(root),
            ]
        )
    assert status.value.code == 0
    (final,) = read_jsonl([output / "trajectories.jsonl"])
    assert final.solver.model == "openai/fresh-solver"
    assert ".run_" in final.traj_id
    assert final.meta["solver_collection"]["revision"] == "pinned-revision"
    assert json.loads((output / "manifest.json").read_text())["complete"]
    assert "native-test-key" not in (output / "manifest.json").read_text()


@pytest.mark.integration
def test_anthropic_verifier_logs_exact_blinded_wire_requests(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl([sample()], str(source))
    captured = []

    def response(wire, timeout):
        assert wire.full_url == "https://opencode.ai/inference/go/anthropic/v1/messages"
        captured.append(json.loads(wire.data))
        assert "NEVER_SEND_THE_GOLD" not in wire.data.decode()
        return BytesIO(
            json.dumps(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"verdict":"incorrect","reason":"test"}',
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 8},
                }
            ).encode()
        )

    output = tmp_path / "verified"
    with (
        patch.dict(
            "os.environ",
            {"OPENCODE_API_KEY": "test-key"},
            clear=True,
        ),
        patch("urllib.request.urlopen", side_effect=response),
        pytest.raises(SystemExit) as status,
    ):
        main(
            [
                "verify",
                "--input",
                str(source),
                "--output",
                str(output),
                "--backend",
                "http",
                "--model",
                "qwen3.8-flash",
            ]
        )
    assert status.value.code == 0
    recorded = [
        json.loads(line)["request"]
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    assert recorded == captured and len(recorded) == 1
    assert all(request["max_tokens"] == 131072 for request in recorded)
    rows = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert all(row["usage"] == {"input_tokens": 10, "output_tokens": 8} for row in rows)
    assert all(
        "input_tokens" not in row and "selected_e_tokens" not in row for row in rows
    )
    assert (
        json.loads((output / "manifest.json").read_text())["api_format"]
        == "anthropic_messages"
    )
