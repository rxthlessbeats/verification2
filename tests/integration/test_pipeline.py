import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from qera.jsonl import read_jsonl, write_jsonl
from tests.support import registry, synthetic
from verification2.agents.verifier import Backend, main, metrics
from verification2.core.prompts import load_prompts


@pytest.mark.integration
def test_mock_no_label_leak(tmp_path):
    source = tmp_path / "sample.jsonl"
    synthetic(source)
    output = tmp_path / "mock"
    with pytest.raises(SystemExit) as result:
        main(
            [
                "run",
                "--input",
                str(source),
                "--output",
                str(output),
                "--backend",
                "mock",
            ]
        )
    assert result.value.code == 0
    requests = (output / "requests.jsonl").read_text()
    assert "DO_NOT_SEND_HIDDEN" not in requests
    assert "label_correct" not in requests and "hidden" not in requests
    results = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert len(results) == 4 and all(r["status"] == "mock" for r in results)
    assert all(
        s["false_rejection_rate"] is None and s["false_acceptance_rate"] is None
        for s in metrics(results, ("QERA",))
    )


@pytest.mark.integration
def test_evidence_has_no_length_cap(tmp_path):
    source = tmp_path / "sample.jsonl"
    synthetic(source)
    trajectory = next(read_jsonl([source]))
    trajectory.segments[1].text = "Full evidence. " * 10000
    write_jsonl([trajectory], str(source))
    output = tmp_path / "evidence"
    with pytest.raises(SystemExit) as status:
        main(
            [
                "verify",
                "--input",
                str(source),
                "--output",
                str(output),
                "--conditions",
                "QEA",
                "--backend",
                "mock",
            ]
        )
    assert status.value.code == 0
    request = json.loads((output / "requests.jsonl").read_text())["request"]
    assert trajectory.segments[1].text in request["messages"][1]["content"]


@pytest.mark.integration
def test_http_wire_contract():
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            response = {
                "choices": [
                    {"message": {"content": '{"verdict":"incorrect","reason":"test"}'}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_args):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("This environment prohibits creating network sockets")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = {
            "model": "local-test",
            "messages": [{"role": "user", "content": "test"}],
        }
        response = Backend(
            mode="http",
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        ).call(request)
        assert response["verdict"] == "incorrect"
        assert response["usage"]["prompt_tokens"] == 10
        assert captured == [request]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.integration
def test_existing_output_not_overwritten(tmp_path):
    source = tmp_path / "sample.jsonl"
    synthetic(source)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        main(["run", "--input", str(source), "--output", str(output)])


@pytest.mark.integration
def test_prompt_file_is_used_and_recorded(tmp_path):
    source = tmp_path / "sample.jsonl"
    synthetic(source)
    prompt = tmp_path / "prompts.yaml"
    bundle = load_prompts().model_dump()
    instruction = "Custom verifier instruction: return verdict and reason.\n"
    bundle["verifier"]["instruction"] = instruction
    prompt.write_text(yaml.safe_dump(bundle))
    output = tmp_path / "prepared"
    with pytest.raises(SystemExit) as result:
        main(
            [
                "run",
                "--input",
                str(source),
                "--output",
                str(output),
                "--prompt-file",
                str(prompt),
                "--conditions",
                "QA",
            ]
        )
    assert result.value.code == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["instruction"] == instruction
    assert manifest["prompt"] == bundle["verifier"]
    assert manifest["configuration"]["prompt_file"] == str(prompt)
    for line in (output / "requests.jsonl").read_text().splitlines():
        assert json.loads(line)["request"]["messages"][0]["content"] == instruction
    prompt.write_text(" \n")
    with pytest.raises(ValueError, match="must not be empty"):
        main(
            [
                "run",
                "--input",
                str(source),
                "--output",
                str(tmp_path / "empty"),
                "--prompt-file",
                str(prompt),
            ]
        )
    assert not (tmp_path / "empty").exists()


@pytest.mark.integration
def test_uv_env_file_loads_configuration_without_recording_key(tmp_path):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is needed for its native --env-file integration test")
    source = tmp_path / "sample.jsonl"
    synthetic(source)
    env_file = tmp_path / ".env"
    secret = "test-only-dotenv-key"
    endpoints = {"chat_completions": "https://example.invalid/chat/completions"}
    models = registry(tmp_path, {"dotenv-model": endpoints})
    env_file.write_text(
        f"TEST_API_KEY={secret}\nVERIFIER_MODEL=dotenv-model\nMODELS_FILE={models}\n"
    )
    env = os.environ.copy()
    for name in [
        "TEST_API_KEY",
        "VERIFIER_MODEL",
        "MODELS_FILE",
        "UV_ENV_FILE",
        "UV_NO_ENV_FILE",
    ]:
        env.pop(name, None)
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            uv,
            "run",
            "--no-cache",
            "--no-sync",
            "--env-file",
            str(env_file),
            "python",
            "-m",
            "verification2.agents.verifier",
            "run",
            "--input",
            str(source),
            "--output",
            str(output),
            "--backend",
            "dry-run",
            "--conditions",
            "QA",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"] == "dotenv-model"
    assert manifest["endpoint_configured"]
    requests = [
        json.loads(line)["request"]
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 4 and all(
        request["model"] == "dotenv-model" for request in requests
    )
    recorded = result.stdout + result.stderr
    recorded += "".join(
        path.read_text(encoding="utf-8-sig") for path in output.iterdir()
    )
    assert secret not in recorded
