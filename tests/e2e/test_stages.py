import json
from io import BytesIO
from unittest.mock import patch

import pytest

from qera.jsonl import read_jsonl, write_jsonl
from tests.support import decisions, sample
from verification2.agents.classification import classification_view
from verification2.agents.verifier import main


@pytest.mark.e2e
@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("gateway", [False, True])
def test_jev_and_verifier_use_independent_models_keys_and_saved_datasets(
    tmp_path, correct, gateway
):
    source = tmp_path / "source.jsonl"
    original = sample()
    if correct:
        original.label.correct = True
        original.segments[-1].text = "blue"
    write_jsonl([original], str(source))
    endpoint = (
        "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
        if gateway
        else "https://opencode.ai/inference/systemone/v1/systemone"
    )
    jev_model = "typesafe-ai/jev" if gateway else "jev-1.13"
    messages = "https://opencode.ai/inference/go/anthropic/v1/messages"
    models = tmp_path / "models.yaml"
    models.write_text(
        json.dumps(
            {
                "models": {
                    jev_model: {
                        "api_key_env": "JEV_API_KEY",
                        "endpoints": {"systemone": endpoint},
                    },
                    "verifier-model": {
                        "api_key_env": "VERIFIER_API_KEY",
                        "endpoints": {"messages": messages},
                    },
                }
            }
        )
    )
    settings = {
        "MODELS_FILE": str(models),
        "JEV_MODEL": jev_model,
        "JEV_API_KEY": "jev-test-key",
        "VERIFIER_MODEL": "verifier-model",
        "VERIFIER_API_KEY": "verify-test-key",
    }
    if gateway:
        settings["AI_GATEWAY_API_KEY"] = "ignored-gateway-key"
    captured = []

    def response(wire, timeout):
        request = json.loads(wire.data)
        captured.append(
            (
                request["model"],
                wire.get_header("Authorization") or wire.get_header("X-api-key"),
            )
        )
        assert "NEVER_SEND_THE_GOLD" not in wire.data.decode()
        if request["model"] == jev_model:
            assert wire.full_url == endpoint
            assert set(request) == {"model", "state", "questions"}
            assert set(request["state"]) == {"benchmark", "task"}
            ((name, question),) = request["questions"].items()
            assert name in {"s1_text_0", "s1_call", "s2_text_0"}
            assert question["type"] == "choice"
            assert set(question["criteria"]) == {"E", "R"}
            raw = decisions(classification_view(original), {name})
            raw["model"] = jev_model
            if gateway:
                raw["provider_metadata"] = {"gateway": {"cost": "0.00001"}}
        else:
            payload = {
                "verdict": "correct" if correct else "incorrect",
                "reason": "Scripted transport test response.",
            }
            raw = {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        return BytesIO(json.dumps(raw).encode())

    with (
        patch.dict("os.environ", settings, clear=True),
        patch("urllib.request.urlopen", side_effect=response),
    ):
        classified = tmp_path / "classified"
        with pytest.raises(SystemExit) as status:
            main(
                [
                    "classify",
                    "--input",
                    str(source),
                    "--output",
                    str(classified),
                    "--backend",
                    "http",
                ]
            )
        assert status.value.code == 0
        frozen = classified / "trajectories.jsonl"
        original_bytes = frozen.read_bytes()
        with pytest.raises(SystemExit) as status:
            main(
                [
                    "verify",
                    "--input",
                    str(frozen),
                    "--output",
                    str(tmp_path / "verified"),
                    "--backend",
                    "http",
                    "--max-tokens",
                    "131072",
                ]
            )
        assert status.value.code == 0
        assert frozen.read_bytes() == original_bytes
    assert captured == [(jev_model, "Bearer jev-test-key")] * 3 + [
        ("verifier-model", "verify-test-key"),
    ]
    (final,) = read_jsonl([frozen])
    assert final.meta["jev_classification"]["model"] == jev_model
    assert final.solver.native_reasoning == "full"
    assert final.label.correct is correct and final.solver.model == "original-solver"
    assert json.loads((classified / "manifest.json").read_text())["complete"]
    logged = "".join(p.read_text(encoding="utf-8-sig") for p in classified.iterdir())
    assert all(
        key not in logged
        for key in ("jev-test-key", "verify-test-key", "ignored-gateway-key")
    )
    result = json.loads((classified / "results.jsonl").read_text())
    assert {
        q: raw["answers"][q] for q, raw in result["raw_response"].items()
    } == decisions(classification_view(original))["answers"]
    assert result["usage"] == {"input_tokens": 300, "output_tokens": 60}
    if gateway:
        assert all(
            raw["provider_metadata"]["gateway"]["cost"] == "0.00001"
            for raw in result["raw_response"].values()
        )
    assert source.read_text().count("NEVER_SEND_THE_GOLD") == 1
