import json

import pytest
from pydantic import ValidationError

from tests.support import CHAT, registry
from verification2.core.registry import load_registry, max_output_tokens, resolve


@pytest.mark.unit
def test_packaged_registry_gives_each_role_the_api_it_needs(monkeypatch):
    monkeypatch.delenv("MODELS_FILE")
    solver = resolve("solver", "qwen3.8-flash")
    verifier = resolve("verifier", "qwen3.8-flash")
    jev = resolve("jev", "typesafe-ai/jev")
    assert solver.endpoint.endswith("/chat/completions")
    assert (verifier.api, jev.api) == ("messages", "systemone")
    assert solver.spec.api_key_env == "OPENCODE_API_KEY"
    assert jev.spec.api_key_env == "JEV_API_KEY"
    assert max_output_tokens("qwen3.8-flash") == 131072


@pytest.mark.unit
def test_reasoning_effort_comes_from_env_and_must_be_listed_for_the_model(
    monkeypatch,
):
    monkeypatch.delenv("MODELS_FILE")
    assert resolve("solver", "glm-5.3-flash").reasoning_effort == "medium"
    monkeypatch.setenv("VERIFIER_REASONING_EFFORT", "none")
    assert resolve("verifier", "glm-5.3-flash").reasoning_effort == "none"
    monkeypatch.setenv("SOLVER_REASONING_EFFORT", "max")
    monkeypatch.setenv("VERIFIER_REASONING_EFFORT", "xhigh")
    assert resolve("solver", "glm-5.3-flash").reasoning_effort == "max"
    assert resolve("verifier", "qwen3.8-flash").reasoning_effort == "xhigh"
    assert resolve("jev", "typesafe-ai/jev").reasoning_effort is None
    with pytest.raises(
        ValueError, match=r"SOLVER_REASONING_EFFORT=max .*qwen3.8-flash"
    ):
        resolve("solver", "qwen3.8-flash")


@pytest.mark.unit
def test_verifier_requests_carry_effort_only_when_set():
    from verification2.agents.verifier import model_request

    chat = "https://example.invalid/v1/chat/completions"
    messages = "https://example.invalid/v1/messages"
    plain = model_request("m", "prompt", "text", 10, chat)
    assert "reasoning_effort" not in plain and "output_config" not in plain
    assert model_request("m", "p", "t", 10, chat, "low")["reasoning_effort"] == "low"
    request = model_request("m", "p", "t", 10, messages, "low")
    assert request["output_config"] == {"effort": "low"}
    assert "reasoning_effort" not in request


@pytest.mark.unit
def test_models_without_an_entry_or_a_usable_api_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "MODELS_FILE", str(registry(tmp_path, {"chat": {"chat_completions": CHAT}}))
    )
    assert resolve("verifier", "chat").api == "chat_completions"
    with pytest.raises(ValueError, match="systemone endpoint, which the jev needs"):
        resolve("jev", "chat")
    with pytest.raises(ValueError, match="not in the model registry"):
        resolve("solver", "unlisted")
    with pytest.raises(ValueError, match="Set SOLVER_MODEL"):
        resolve("solver", None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        {"api_key_env": "K", "endpoints": {"messages": CHAT}},
        {
            "api_key_env": "K",
            "endpoints": {"chat_completions": "file:///tmp/v1/chat/completions"},
        },
        {
            "api_key_env": "K",
            "endpoints": {"chat_completions": "https:///v1/chat/completions"},
        },
        {
            "api_key_env": "K",
            "endpoints": {
                "chat_completions": "https://user:secret@example.invalid/v1/chat/completions"
            },
        },
        {"api_key_env": "K", "endpoints": {}},
        {
            "api_key_env": "K",
            "endpoints": {"chat_completions": CHAT},
            "max_output_tokens": 0,
        },
        {
            "api_key_env": "K",
            "endpoints": {"chat_completions": CHAT},
            "reasoning_efforts": ["low"],
            "default_reasoning_effort": "high",
        },
    ],
)
def test_registry_entries_are_validated(tmp_path, entry):
    path = tmp_path / "models.yaml"
    path.write_text(json.dumps({"models": {"m": entry}}))
    with pytest.raises(ValidationError):
        load_registry(path)


@pytest.mark.unit
def test_messages_needs_a_listed_output_maximum(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "m": {"api_key_env": "K", "endpoints": {"chat_completions": CHAT}}
                }
            }
        )
    )
    monkeypatch.setenv("MODELS_FILE", str(path))
    with pytest.raises(ValueError, match="no max_output_tokens for m"):
        max_output_tokens("m")
