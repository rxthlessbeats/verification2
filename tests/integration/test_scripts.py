import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

STAMPED = r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] "
EXPERIMENT_STUB = """import json, os, sys, time
from pathlib import Path
with open(os.environ['SCRIPT_CALLS'], 'a') as log:
    log.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')
args = sys.argv[1:]
if Path(sys.argv[0]).name == 'docker':
    if args[0] == 'ps':
        print('abc')
    sys.exit()
module = args[args.index('-m') + 1].split('.')[-1]
step = args[args.index('--partition') + 1] if '--partition' in args else module
print('working on', step)
if step == os.environ.get('SLOW_STEP'):
    time.sleep(60)
if step == os.environ.get('FAIL_STEP'):
    if step == 'doctor':
        checks = [{'name': 'docker', 'ok': False}, {'name': 'api:SOLVER', 'ok': True}]
        Path(args[args.index('--output') + 1]).write_text(json.dumps({'checks': checks}))
    sys.exit(3)
"""


def script(tmp_path, name, monkeypatch):
    directory = tmp_path / "scripts"
    directory.mkdir()
    target = directory / name
    shutil.copyfile(Path(__file__).parents[2] / "scripts" / name, target)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SCRIPT_CALLS", str(tmp_path / "calls.jsonl"))
    return target


def command(tmp_path, name, body):
    path = tmp_path / "bin" / name
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o700)


@pytest.mark.integration
def test_timing_summarizes_steps_stages_and_solve_times_of_a_running_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "[2026-01-01 10:00:00] experiment outputs/c1 | steps doctor train\n"
        "[2026-01-01 10:00:00] doctor\n"
        "[2026-01-01 10:00:05] doctor: ready\n"
        "[2026-01-01 10:00:05] train\n"
        "[2026-01-01 10:01:05] [tau2_bench] tau2_bench/a/1: ok | took 1m00s | correct\n"
        "[2026-01-01 10:01:06] [tau2_bench] tau2_bench/a/2: ok | stored, originally "
        "solved in 2m30s | correct\n"
        "[2026-01-01 10:01:06] [tau2_bench] tau2_bench/a/3: ok | resumed | correct\n"
        "untimed native output\n"
        "[2026-01-01 11:01:06] [swebench_verified] swebench_verified/x: failed | "
        "took 1h00m00s | error\n"
    )
    result = subprocess.run(
        [sys.executable, Path(__file__).parents[2] / "scripts/timing.py", log],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "total 1h01m06s (2026-01-01 10:00:00 -> 11:01:06), running",
        "doctor 0m05s",
        "train 1h01m01s | solver 1h01m01s",
        (
            "  tau2_bench: 3 tasks (3 ok) | solved now 1 in 1m00s, mean 1m00s, max 1m00s"
            " | from store 1, originally 2m30s | resumed 1"
        ),
        (
            "  swebench_verified: 1 tasks (1 failed) | solved now 1 in 1h00m00s,"
            " mean 1h00m00s, max 1h00m00s"
        ),
    ]


def experiment(tmp_path, monkeypatch):
    target = script(tmp_path, "experiment.sh", monkeypatch)
    shutil.copyfile(
        Path(__file__).parents[2] / "scripts/timing.py", target.with_name("timing.py")
    )
    command(tmp_path, "uv", EXPERIMENT_STUB)
    command(tmp_path, "docker", EXPERIMENT_STUB)
    (tmp_path / ".env").write_text("")
    saved = tmp_path / "outputs/c1/benchmarks"
    saved.parent.mkdir(parents=True)
    saved.write_text("tau2_bench swebench_verified\n")

    def run(*args):
        return subprocess.run(
            ["bash", str(target), "c1", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    return run


def calls(tmp_path, program):
    lines = (tmp_path / "calls.jsonl").read_text().splitlines()
    return [call[1:] for call in map(json.loads, lines) if call[0] == program]


def after(args, flag, count=1):
    index = args.index(flag) + 1
    return args[index : index + count]


@pytest.mark.integration
def test_experiment_runs_default_steps_with_a_timestamped_log_and_timing(
    tmp_path, monkeypatch
):
    result = experiment(tmp_path, monkeypatch)()
    assert result.returncode == 0, result.stderr
    doctor, pilot, train = calls(tmp_path, "uv")
    assert after(doctor, "-m") == ["verification2.benchmarks.doctor"]
    assert after(doctor, "--benchmarks", 2) == ["tau2_bench", "swebench_verified"]
    assert after(doctor, "--output") == ["outputs/c1/readiness.json"]
    for args, partition in ((pilot, "pilot"), (train, "train")):
        assert after(args, "--partition") == [partition]
        assert after(args, "--output") == [f"outputs/c1/{partition}"]
        assert after(args, "--cache-dir") == ["outputs/c1/api_cache"]
    assert "--conditions" not in pilot and after(train, "--conditions", 2)[0] == "QERA"

    log = (tmp_path / "outputs/c1/run.log").read_text()
    assert result.stdout == log
    for line in ("doctor: ready", "working on train", "done \\(exit 0\\)"):
        assert re.search(STAMPED + line + "$", log, re.MULTILINE)
    timing = (tmp_path / "outputs/c1/timing.txt").read_text().splitlines()
    assert timing[0].startswith("total ") and timing[0].endswith("done (exit 0)")
    assert [line.split()[0] for line in timing[1:]] == ["doctor", "pilot", "train"]


@pytest.mark.integration
def test_experiment_passes_options_and_stops_at_the_failed_step(tmp_path, monkeypatch):
    monkeypatch.setenv("FAIL_STEP", "test")
    run = experiment(tmp_path, monkeypatch)
    result = run(
        "freeze", "test", "train", "--output", "outputs/c1_retry",
        "--benchmarks", "tau2_bench  swebench_verified",
        "--conditions", "QA QERA", "--rerun", "all", "--retry-failed",
        "--train", "all", "--test", "5", "--jobs", "3",
    )  # fmt: skip
    assert result.returncode == 3
    freeze, test = calls(tmp_path, "uv")
    assert after(freeze, "--benchmarks", 2) == ["tau2_bench", "swebench_verified"]
    assert after(freeze, "--output") == ["outputs/c1_retry/protocol.json"]
    assert after(test, "--benchmarks", 3)[:2] == ["tau2_bench", "swebench_verified"]
    assert test[test.index("--conditions") :] == [
        "--conditions", "QA", "QERA", "--protocol", "outputs/c1_retry/protocol.json",
        "--limit", "5", "--jobs", "3", "--retry-failed", "--rerun", "all",
    ]  # fmt: skip
    log = (tmp_path / "outputs/c1_retry/run.log").read_text()
    assert re.search(STAMPED + "done \\(exit 3\\)$", log, re.MULTILINE)
    rejected = run("train", "--train", "0")
    assert rejected.returncode == 2 and "positive number or all" in rejected.stderr
    other = run("train", "--output", "outputs/c1_retry", "--benchmarks", "tau2_bench")
    assert other.returncode == 2 and "uses --benchmarks" in other.stderr
    new = run("train", "--output", "outputs/c2")
    assert new.returncode == 2 and "needs --benchmarks" in new.stderr


@pytest.mark.integration
def test_failed_doctor_names_failing_checks_and_runs_nothing_paid(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FAIL_STEP", "doctor")
    result = experiment(tmp_path, monkeypatch)()
    assert result.returncode == 1
    assert len(calls(tmp_path, "uv")) == 1
    assert "doctor: not ready (outputs/c1/readiness.json); failing: docker" in (
        result.stdout
    )


@pytest.mark.integration
def test_experiment_refuses_a_second_run_in_the_same_folder(tmp_path, monkeypatch):
    run = experiment(tmp_path, monkeypatch)
    output = tmp_path / "outputs/c1"
    log = output / "run.log"
    log.write_text("[2026-01-01 10:00:00] pilot\n")
    with open(output / "run.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refused = run("--detach")
        status = run("status")
    assert refused.returncode == 2 and "already active" in refused.stderr
    assert log.read_text() == "[2026-01-01 10:00:00] pilot\n"
    assert not (tmp_path / "calls.jsonl").exists()
    running, total, step = status.stdout.splitlines()[:3]
    assert running == "outputs/c1: running" and total.endswith("running")
    assert not total.startswith("total 0m00s") and step.startswith("pilot ")


@pytest.mark.integration
def test_detached_run_reports_status_and_stops_with_its_containers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLOW_STEP", "pilot")
    run = experiment(tmp_path, monkeypatch)
    started = run("--detach")
    assert started.returncode == 0 and "Running in the background" in started.stdout
    log = tmp_path / "outputs/c1/run.log"
    deadline = time.monotonic() + 30
    while "working on pilot" not in log.read_text():
        assert time.monotonic() < deadline, log.read_text()
        time.sleep(0.1)
    assert run("status").stdout.startswith("outputs/c1: running")

    stopped = run("stop")
    assert stopped.returncode == 0 and "Stopped outputs/c1" in stopped.stdout
    assert not (tmp_path / "outputs/c1/run.pid").exists()
    assert re.search(STAMPED + "done \\(stopped\\)$", log.read_text(), re.MULTILINE)
    assert [after(args, "--partition") for args in calls(tmp_path, "uv")[1:]] == [
        ["pilot"]
    ]
    docker = calls(tmp_path, "docker")
    assert ["ps", "-q", "--filter", "name=^minisweagent-"] in docker
    assert docker[-1] == ["rm", "-f", "abc"]
    assert run("status").stdout.startswith("outputs/c1: not running")
    assert run("stop").returncode == 2
