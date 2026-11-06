import json
from io import BytesIO
from unittest.mock import patch

import pytest

from qera.overrides import apply
from qera.render import build_verifier_input
from qera.schema import Role, Span
from tests.support import SYSTEMONE, decisions, registry, sample
from verification2.agents.classification import (
    classification_request,
    classification_view,
    parse_classification,
)
from verification2.agents.verifier import Backend, main, model_request
from verification2.core.prompts import load_prompts


@pytest.mark.unit
def test_classification_repairs_roles_without_changing_text_actions_or_labels():
    original = sample()
    view = classification_view(original)
    assert "NEVER_SEND_THE_GOLD" not in json.dumps(view)
    assert "label" not in view and "hidden" not in view
    parsed = parse_classification(decisions(view), original, view)
    overrides = {(o["traj_id"], o["seg_id"]): o for o in parsed["overrides"]}
    (corrected,) = apply([original.model_copy(deep=True)], overrides)
    assert corrected.segments[1].role.value == "E"
    assert corrected.segments[2].role.value == "R"
    assert [s.text for s in corrected.segments] == [s.text for s in original.segments]
    assert corrected.label == original.label
    assert corrected.solver.model == "original-solver"
    assert (
        build_verifier_input(corrected, "QA").text
        == build_verifier_input(original, "QA").text
    )
    qea = build_verifier_input(corrected, "QEA").text
    assert "Registry: blue." in qea and "I suspect red." not in qea


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "fixed",
        "wrong_type",
        "wrong_choice",
        "not_an_answer",
        "not_answers",
    ],
)
def test_classification_rejects_partial_or_unsafe_decisions(case):
    trajectory = sample()
    view = classification_view(trajectory)
    bad = decisions(view)
    answers = bad["answers"]
    if case == "missing":
        answers.pop("s1_call")
    elif case in ("extra", "fixed"):
        answers["s0_text_0" if case == "fixed" else "unknown"] = {
            "type": "choice",
            "choice": "E",
        }
    elif case == "wrong_type":
        answers["s1_call"]["type"] = "noul"
    elif case == "wrong_choice":
        answers["s1_call"]["choice"] = "A"
    elif case == "not_an_answer":
        answers["s1_call"] = None
    else:
        bad["answers"] = []
    with pytest.raises(ValueError):
        parse_classification(bad, trajectory, view)


@pytest.mark.unit
def test_mixed_spans_protect_task_anchors_and_support_call_only_classification():
    trajectory = sample()
    segment = trajectory.segments[1]
    segment.text = "Task. Evidence."
    segment.spans = [Span(start=0, end=6, role="Q"), Span(start=6, end=15, role="E")]
    view = classification_view(trajectory)
    answer = decisions(view)
    assert "s1_text_0" not in answer["answers"]
    answer["answers"]["s1_text_1"]["choice"] = "R"
    parsed = parse_classification(answer, trajectory, view)
    assert parsed["overrides"][0]["spans"] == [[0, 6, "Q"], [6, 15, "R"]]
    request = classification_request("jev-1.13", view, load_prompts().jev, "s1_text_1")
    assert list(request["questions"]) == ["s1_text_1"]
    instructions = request["questions"]["s1_text_1"]["instructions"]
    assert (instructions["content"], instructions["start"], instructions["end"]) == (
        "Evidence.",
        6,
        15,
    )
    assert {"seg_id": 1, "speaker": segment.speaker.value, "text": "Task. "} in request[
        "state"
    ]["task"]
    segment.spans = None
    segment.role_override = Role.Q
    segment.call_role_override = Role.R
    view = classification_view(trajectory)
    answer = decisions(view)
    assert "s1_text_0" not in answer["answers"] and "s1_call" in answer["answers"]
    parsed = parse_classification(answer, trajectory, view)
    assert parsed["overrides"][0] == {
        "traj_id": "sample",
        "seg_id": 1,
        "call_role": "E",
        "source": "classifier",
    }


@pytest.mark.unit
def test_manual_annotations_keep_priority_over_jev():
    trajectory = sample()
    segment = trajectory.segments[2]
    segment.role_override = Role.E
    segment.role_source = "manual"
    view = classification_view(trajectory)
    parsed = parse_classification(decisions(view), trajectory, view)
    (corrected,) = apply(
        [trajectory], {(o["traj_id"], o["seg_id"]): o for o in parsed["overrides"]}
    )
    assert corrected.segments[2].role.value == "E"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "endpoints"),
    [
        ("glm-5.3-flash", {"systemone": "https://example.invalid/v1/systemone"}),
        ("qwen3.8-flash", {"systemone": "https://example.invalid/v1/systemone"}),
        (
            "jev-1.13",
            {"chat_completions": "https://example.invalid/v1/chat/completions"},
        ),
    ],
)
def test_classification_rejects_other_models_and_chat_protocol_before_reading_data(
    tmp_path, monkeypatch, model, endpoints
):
    monkeypatch.setenv("MODELS_FILE", str(registry(tmp_path, {model: endpoints})))
    with pytest.raises(ValueError, match="jev-.*systemone"):
        main(
            [
                "classify",
                "--input",
                "missing.jsonl",
                "--output",
                str(tmp_path / "invalid"),
                "--model",
                model,
            ]
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.unit
@pytest.mark.parametrize("custom_key", [False, True])
def test_keys_come_only_from_the_variable_the_registry_names(tmp_path, custom_key):
    settings = {
        "JEV_MODEL": "typesafe-ai/jev",
        "OPENCODE_API_KEY": "opencode-only",
        "AI_GATEWAY_API_KEY": "ignored-gateway-key",
        "VERIFIER_API_KEY": "opencode-verifier",
    }
    if custom_key:
        models = {"typesafe-ai/jev": {"systemone": SYSTEMONE}}
        settings["MODELS_FILE"] = str(registry(tmp_path, models, "CUSTOM_GATEWAY_KEY"))
        settings["JEV_API_KEY"] = "must-not-override-custom-key"
    with (
        patch.dict("os.environ", settings, clear=True),
        patch("urllib.request.urlopen") as transport,
        pytest.raises(
            ValueError, match="CUSTOM_GATEWAY_KEY" if custom_key else "JEV_API_KEY"
        ),
    ):
        main(
            [
                "classify",
                "--input",
                "missing.jsonl",
                "--output",
                str(tmp_path / "missing"),
                "--backend",
                "http",
            ]
        )
    transport.assert_not_called()
    assert not (tmp_path / "missing").exists()


@pytest.mark.unit
def test_anthropic_transport_uses_native_payload_headers_and_only_text_blocks():
    endpoint = "https://opencode.ai/inference/go/anthropic/v1/messages"
    request = model_request(
        "qwen3.8-flash", "Fixed instruction", "Frozen candidate", 512, endpoint
    )
    assert request["system"] == "Fixed instruction"
    assert request["messages"] == [
        {
            "role": "user",
            "content": "<trajectory>\nFrozen candidate\n</trajectory>\n\nFixed instruction",
        }
    ]
    raw = {
        "content": [
            {"type": "thinking", "thinking": "not a verdict"},
            {
                "type": "text",
                "text": '{"verdict":"incorrect","reason":"Evidence contradicts it."}',
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }

    def response(wire, timeout):
        assert wire.get_header("X-api-key") == "anthropic-test-key"
        assert wire.get_header("Authorization") is None
        assert wire.get_header("Anthropic-version") == "2023-06-01"
        assert wire.get_header("User-agent") == "verification2/0.1"
        assert json.loads(wire.data) == request
        return BytesIO(json.dumps(raw).encode())

    with patch("urllib.request.urlopen", side_effect=response):
        result = Backend(
            mode="http", endpoint=endpoint, api_key="anthropic-test-key"
        ).call(request)
    assert result["verdict"] == "incorrect" and result["usage"] == raw["usage"]


@pytest.mark.unit
@pytest.mark.parametrize("anthropic", [False, True])
@pytest.mark.parametrize("has_verdict", [False, True])
def test_truncated_verifier_output_keeps_diagnostics_without_accepting_a_verdict(
    anthropic, has_verdict
):
    text = '{"verdict":"correct","reason":"Incomplete response"}' if has_verdict else ""
    if anthropic:
        endpoint = "https://example.invalid/v1/messages"
        content = [{"type": "thinking", "thinking": "Still evaluating the evidence."}]
        if text:
            content.append({"type": "text", "text": text})
        raw = {
            "content": content,
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 100, "output_tokens": 2048},
        }
    else:
        endpoint = "https://example.invalid/v1/chat/completions"
        raw = {
            "choices": [
                {"finish_reason": "length", "message": {"content": text or None}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2048},
        }
    request = model_request(
        "test-model", "Return a verdict", "Frozen input", 2048, endpoint
    )
    with patch(
        "urllib.request.urlopen", return_value=BytesIO(json.dumps(raw).encode())
    ):
        result = Backend(mode="http", endpoint=endpoint).call(request)
    assert result["status"] == "error" and result.get("verdict") is None
    assert "max_tokens=2048" in result["error"] and "truncated" in result["error"]
    assert result["raw"] == raw and result["usage"] == raw["usage"]
