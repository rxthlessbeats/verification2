import json
from io import BytesIO
from pathlib import Path

import pytest
import yaml

from qera.jsonl import read_jsonl
from tests.support import configure, fake_native_episode, fresh_splits
from verification2.core.prompts import load_prompts
from verification2.data.split import BENCHMARKS, select, split_id
from verification2.experiments import collection, protocol
from verification2.experiments import report as reporting
from verification2.experiments import run as experiment


@pytest.mark.e2e
def test_fresh_benchmark_pilot_resume_and_frozen_heldout_matrix(tmp_path, monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(
        protocol, "runtime_hashes", lambda **kwargs: {"native": "test-lock"}
    )
    monkeypatch.setattr(
        experiment, "runtime_hashes", lambda **kwargs: {"native": "test-lock"}
    )
    splits = fresh_splits(tmp_path, monkeypatch)
    benchmarks = ["--benchmarks", *BENCHMARKS]
    native_calls, api_calls = [], []

    def native(job, *args):
        native_calls.append(job.task.key)
        return fake_native_episode(job)

    def response(wire, timeout):
        request = json.loads(wire.data)
        api_calls.append(request)
        assert "label_correct" not in wire.data.decode()
        if "questions" in request:
            assert set(request["state"]) == {"benchmark", "task"}
            assert len(request["questions"]) == 1
            raw = {
                "answers": {
                    name: {
                        "type": "choice",
                        "choice": "R"
                        if q["instructions"]["speaker"] == "solver"
                        else "E",
                    }
                    for name, q in request["questions"].items()
                },
                "usage": {"input_tokens": 19, "output_tokens": 3},
            }
        else:
            assert "SOLVER REASONING |" not in wire.data.decode()
            raw = {
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict":"correct","reason":"scripted transport test"}'
                        }
                    }
                ],
                "usage": {"input_tokens": 23, "output_tokens": 5},
            }
        return BytesIO(json.dumps(raw).encode())

    monkeypatch.setattr(collection, "collect_job", native)
    monkeypatch.setattr(
        experiment,
        "check",
        lambda **kwargs: {"ready": True, "checks": [], "paid_calls": 0},
    )
    monkeypatch.setattr("urllib.request.urlopen", response)
    pilot = tmp_path / "pilot"
    flags = [*benchmarks, "--output", str(pilot)]
    assert experiment.main([*flags, "--stage", "verifier"]) == 1
    assert not native_calls and not api_calls
    assert experiment.main([*flags, "--stage", "solver"]) == 0
    assert len(native_calls) == len(BENCHMARKS) and not api_calls
    assert experiment.main([*flags, "--stage", "verifier"]) == 1
    assert not api_calls
    assert experiment.main([*flags, "--stage", "jev"]) == 0
    assert api_calls and all("questions" in r for r in api_calls)
    after_jev = len(api_calls)
    assert (
        experiment.main([*flags, "--stage", "verifier", "--conditions", "QA", "QRA"])
        == 0
    )
    assert len(native_calls) == len(BENCHMARKS)
    assert len(api_calls[after_jev:]) == 2 * len(BENCHMARKS)
    assert all("questions" not in r for r in api_calls[after_jev:])
    verdicts = [
        json.loads(line)
        for line in (pilot / "verifier/results.jsonl").read_text().splitlines()
    ]
    assert {v["condition"] for v in verdicts} == {"QA", "QRA"}
    assert experiment.main(flags) == 0
    assert len(native_calls) == len(BENCHMARKS)
    before = len(api_calls)
    assert experiment.main(flags) == 0
    assert len(native_calls) == len(BENCHMARKS) and len(api_calls) == before
    report = json.loads((pilot / "report.json").read_text())
    assert report["status"] == "passed" and all(report["readable_reasoning"].values())
    train = tmp_path / "train"
    training = [
        *benchmarks,
        "--output",
        str(train),
        "--partition",
        "train",
    ]
    assert experiment.main(training) == 0
    native_count, api_count = len(native_calls), len(api_calls)
    changed = tmp_path / "verifier_v2.yaml"
    bundle = load_prompts()
    bundle.verifier.instruction += " Check each conclusion against the supplied record."
    changed.write_text(yaml.safe_dump(bundle.model_dump(mode="json")))
    assert experiment.main([*training, "--prompt-file", str(changed)]) == 0
    assert len(native_calls) == native_count
    assert len(api_calls) == api_count + len(BENCHMARKS) and all(
        "questions" not in r for r in api_calls[api_count:]
    )
    final_protocol = tmp_path / "protocol.json"
    protocol.main([*benchmarks, "--output", str(final_protocol)])
    heldout = tmp_path / "test"
    test_flags = [
        *benchmarks,
        "--output",
        str(heldout),
        "--partition",
        "test",
        "--protocol",
        str(final_protocol),
    ]
    assert experiment.main([*test_flags, "--stage", "solver"]) == 0
    after_solver = len(native_calls)
    assert experiment.main([*test_flags, "--stage", "jev"]) == 0
    assert (
        experiment.main(
            [*test_flags, "--stage", "verifier", "--conditions", "QA", "QEA"]
        )
        == 0
    )
    assert len(native_calls) == after_solver
    assert (
        experiment.main(
            [
                *benchmarks,
                "--output",
                str(heldout),
                "--partition",
                "test",
                "--protocol",
                str(final_protocol),
                "--conditions",
                "QA",
                "QEA",
                "QRA",
                "QERA",
                "--jobs",
                "3",
            ]
        )
        == 0
    )
    records = list(read_jsonl([heldout / "solver/trajectories.jsonl"]))
    order = [t.key for t in select(splits, "test")]
    assert [t.meta["native_task_key"] for t in records] == order
    assert {t.benchmark for t in records} == set(BENCHMARKS)
    assert all(
        t.split == "test" and t.solver.origin == "self_generated" for t in records
    )
    assert set(native_calls[: len(BENCHMARKS)]).isdisjoint(
        native_calls[len(BENCHMARKS) :]
    )
    full_report = json.loads((heldout / "report.json").read_text())
    verified = Path(full_report["stages"]["verifier"]["artifact"])
    results = [json.loads(line) for line in verified.read_text().splitlines()]
    assert [(r["native_task_key"], r["condition"]) for r in results] == [
        (key, c) for key in order for c in ("QA", "QEA", "QRA", "QERA")
    ]
    assert all(r["usage"] == {"input_tokens": 23, "output_tokens": 5} for r in results)
    assert "smoke-test-secret" not in "".join(
        p.read_text(encoding="utf-8-sig") for p in heldout.rglob("*.json")
    )
    analysis = json.loads((Path(full_report["analysis"]) / "report.json").read_text())
    assert (
        analysis["split_id"] == split_id(splits)
        and analysis["error_result_records"] == 0
    )
    results[0]["split_id"] = "another-split"
    verified.write_text("".join(json.dumps(r) + "\n" for r in results))
    with pytest.raises(ValueError, match="split/partition"):
        reporting.main(
            [
                "--input",
                str(verified.parent),
                *benchmarks,
                "--partition",
                "test",
                "--output",
                str(tmp_path / "wrong_report"),
            ]
        )


@pytest.mark.e2e
def test_readiness_failure_stops_before_any_solver_or_paid_provider(
    tmp_path, monkeypatch
):
    fresh_splits(tmp_path, monkeypatch)
    monkeypatch.setattr(
        experiment,
        "check",
        lambda **kwargs: {
            "ready": False,
            "checks": [{"name": "docker", "ok": False}],
            "paid_calls": 0,
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A blocked preflight must not start native/API work")

    monkeypatch.setattr(collection, "collect_job", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    output = tmp_path / "pilot"
    flags = ["--benchmarks", *BENCHMARKS, "--output", str(output)]
    assert experiment.main(flags) == 2
    assert json.loads((output / "report.json").read_text())["status"] == "blocked"
