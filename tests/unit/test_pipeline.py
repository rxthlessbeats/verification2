import json
from unittest.mock import patch

import pytest

from tests.support import registry
from verification2.agents.verifier import (
    Backend,
    main,
    metrics,
    parse_response,
)


@pytest.mark.unit
def test_response_parser():
    assert (
        parse_response('```json\n{"verdict":"incorrect","reason":"bad"}\n```')[
            "verdict"
        ]
        == "incorrect"
    )
    for bad in ["correct", '{"verdict":true,"reason":"x"}', '{"verdict":"correct"}']:
        with pytest.raises((ValueError, TypeError)):
            parse_response(bad)


@pytest.mark.unit
def test_paired_metrics_exclude_failed_pair():
    rows = []
    for i, label in enumerate([True, False, True]):
        for condition in ("QA", "QEA"):
            rows.append(
                {
                    "benchmark": "b",
                    "traj_id": str(i),
                    "condition": condition,
                    "label_correct": label,
                    "status": "error" if i == 2 and condition == "QA" else "ok",
                    "verdict": "correct" if label else "incorrect",
                }
            )
    summaries = metrics(rows, ("QA", "QEA"))
    assert all(s["paired_labelled_n"] == 2 for s in summaries)
    assert all(
        s["false_rejection_rate"] == 0 and s["false_acceptance_rate"] == 0
        for s in summaries
    )


@pytest.mark.unit
def test_verifier_error_rates_use_the_matching_solver_class_and_exclude_failures():
    rows = []
    for index, (label, verdict, status) in enumerate(
        [
            (True, "correct", "ok"),
            (True, "incorrect", "ok"),
            (False, "correct", "ok"),
            (False, "incorrect", "ok"),
            (False, "incorrect", "ok"),
            (True, None, "error"),
            (None, "correct", "ok"),
        ]
    ):
        rows.append(
            {
                "benchmark": "test",
                "traj_id": str(index),
                "condition": "QERA",
                "label_correct": label,
                "verdict": verdict,
                "status": status,
            }
        )
    (summary,) = metrics(rows, ("QERA",))
    assert (
        summary["attempted"] == 7
        and summary["errors"] == 1
        and summary["paired_labelled_n"] == 5
    )
    assert summary["correct_n"] == 2 and summary["incorrect_n"] == 3
    assert summary["false_rejections"] == summary["false_acceptances"] == 1
    assert summary["false_rejection_rate"] == 1 / 2
    assert summary["false_acceptance_rate"] == 1 / 3
    for label, absent_rate in (
        (True, "false_acceptance_rate"),
        (False, "false_rejection_rate"),
    ):
        (one_class,) = metrics(
            [row for row in rows if row["label_correct"] is label], ("QERA",)
        )
        assert one_class[absent_rate] is None


@pytest.mark.unit
def test_opencode_headers_identify_client_without_sending_labels():
    raw = {
        "choices": [{"message": {"content": '{"verdict":"correct","reason":"test"}'}}]
    }
    request = {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": "data"}],
    }

    def response(wire, timeout):
        from io import BytesIO

        assert wire.get_header("User-agent") == "verification2/0.1"
        assert wire.get_header("Authorization") == "Bearer test-only-key"
        assert wire.get_header("X-opencode-session").startswith("verification2-")
        assert json.loads(wire.data) == request
        return BytesIO(json.dumps(raw).encode())

    with patch("urllib.request.urlopen", side_effect=response):
        backend = Backend(
            mode="http",
            endpoint="https://opencode.ai/inference/go/openai/v1/chat/completions",
            api_key="test-only-key",
        )
        assert backend.call(request)["verdict"] == "correct"


@pytest.mark.unit
def test_missing_registry_key_fails_before_reading_data_or_creating_output(tmp_path):
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="qwen3.8-flash needs an API key in OPENCODE"),
    ):
        main(
            [
                "run",
                "--input",
                "missing.jsonl",
                "--output",
                str(tmp_path / "run"),
                "--backend",
                "http",
                "--model",
                "qwen3.8-flash",
            ]
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.unit
def test_model_comes_from_environment_or_cli_and_endpoint_from_registry(tmp_path):
    environment = "https://example.invalid/environment/v1/messages"
    explicit = "https://example.invalid/explicit/v1/chat/completions"
    models = registry(
        tmp_path,
        {
            "environment-model": {"messages": environment},
            "explicit-model": {"chat_completions": explicit},
        },
    )
    defaults = {"VERIFIER_MODEL": "environment-model", "MODELS_FILE": str(models)}
    cases = [
        ([], "environment-model", environment),
        (["--model", "explicit-model"], "explicit-model", explicit),
    ]
    with patch.dict("os.environ", defaults, clear=True):
        for flags, model, endpoint in cases:
            with (
                patch("verification2.agents.verifier.run", return_value=0) as run,
                pytest.raises(SystemExit),
            ):
                main(
                    [
                        "run",
                        "--input",
                        "unused.jsonl",
                        "--output",
                        str(tmp_path / "unused"),
                        *flags,
                    ]
                )
            args = run.call_args.args[0]
            assert args.model == model
            assert args.endpoint == endpoint
