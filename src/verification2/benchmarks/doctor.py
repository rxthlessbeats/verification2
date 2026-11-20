"""Readiness checks; never starts a solver or makes paid model calls."""

import argparse
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from ..agents.verifier import backend_from_args
from ..core.registry import resolve
from ..data.split import BENCHMARKS
from .setup import ROOT, SOURCES, source

# Source names, runtimes and host tools each benchmark needs.
NEEDS = {
    "tau2_bench": {"tau2"},
    "swebench_verified": {"mini", "swe", "docker"},
    "agentdojo": {"agentdojo"},
    "appworld": {"appworld"},
    "scienceworld": {"scienceworld"},
    "bird": {"bird"},
}
LOCKS = {
    "tau2": "raw/tau2-bench/uv.lock",
    "swe": "raw/runtimes/swe/requirements.lock",
    "agentdojo": "raw/runtimes/agentdojo/requirements.lock",
    "appworld": "raw/runtimes/appworld/requirements.lock",
    "scienceworld": "raw/runtimes/scienceworld/requirements.lock",
    "bird": "raw/runtimes/bird/requirements.lock",
}


def check(network=False, benchmarks=BENCHMARKS, roles=("SOLVER", "JEV", "VERIFIER")):
    from ..experiments.collection import child_environment
    from ..experiments.protocol import runtime_hashes

    needs = set().union(*(NEEDS[b] for b in benchmarks))
    rows = []
    environment = child_environment(None)

    def probe(argv):
        try:
            return (
                subprocess.run(
                    argv,
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

    def record(name, ok, detail):
        rows.append({"name": name, "ok": bool(ok), "detail": detail})

    for name, spec in SOURCES.items():
        if name not in needs:
            continue
        path = ROOT / "raw" / spec["directory"]
        try:
            if not path.exists():
                raise ValueError("missing; run make setup")
            source(name)
            record("source:" + name, True, spec["revision"])
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            record("source:" + name, False, str(exc))
    for name, relative in {
        "tau2": "raw/tau2-bench/.venv/bin/python",
        "swe": "raw/runtimes/swe/.venv/bin/python",
        "agentdojo": "raw/runtimes/agentdojo/.venv/bin/python",
        "appworld": "raw/runtimes/appworld/.venv/bin/python",
        "scienceworld": "raw/runtimes/scienceworld/.venv/bin/python",
        "bird": "raw/runtimes/bird/.venv/bin/python",
    }.items():
        if name not in needs:
            continue
        python = ROOT / relative
        module = {
            "tau2": "tau2",
            "swe": "swebench,minisweagent",
            "agentdojo": "agentdojo",
            "appworld": "appworld",
            "scienceworld": "scienceworld,jdk4py",
            "bird": "sqlite3",
        }[name]
        ready = python.exists() and probe([str(python), "-c", "import " + module])
        record(
            "runtime:" + name,
            ready,
            "ready" if ready else "missing/incomplete; run make setup",
        )
    docker = shutil.which("docker")
    if "docker" in needs:
        try:
            result = (
                subprocess.run(
                    [docker, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                if docker
                else None
            )
        except subprocess.TimeoutExpired:
            result = None
        # docker info exits 0 with an empty version when the socket denies access.
        ready = result is not None and result.returncode == 0 and result.stdout.strip()
        record(
            "docker",
            ready,
            "ready"
            if ready
            else "Docker CLI/daemon unavailable; run make setup-system on the execution host",
        )
    for benchmark in benchmarks:
        if benchmark == "tau2_bench":
            continue
        path = ROOT / "raw/tasks" / (benchmark + ".jsonl")
        record("tasks:" + benchmark, path.is_file(), str(path))
    if "bird" in benchmarks:
        databases = ROOT / "raw/bird/databases"
        ready = databases.is_dir() and any(databases.rglob("*.sqlite"))
        record(
            "bird-databases",
            ready,
            "ready" if ready else "missing; run make setup",
        )
    if "appworld" in needs:
        data = ROOT / "raw/appworld-root/data/datasets/train.txt"
        record(
            "appworld-data",
            data.is_file(),
            "ready" if data.is_file() else "missing; run make setup",
        )
    hashes = runtime_hashes()
    for name, path in LOCKS.items():
        if name in needs:
            record(
                "lock:" + path,
                hashes[path] != "missing",
                "present" if hashes[path] != "missing" else "missing; run make setup",
            )
    for role in roles:
        model = os.environ.get(role + "_MODEL")
        try:
            resolved = resolve(role.lower(), model)
            args = SimpleNamespace(
                backend="http",
                model=model,
                endpoint=resolved.endpoint,
                timeout=None,
                cache_dir=None,
            )
            backend_from_args(args)
            host = urlsplit(resolved.endpoint).hostname
            record(
                "api:" + role,
                True,
                {
                    "model": model,
                    "api": resolved.api,
                    "host": host,
                    "key_configured": True,
                }
                | (
                    {}
                    if role == "JEV"
                    else {
                        "reasoning_effort": resolved.reasoning_effort
                        or "provider default"
                    }
                ),
            )
        except (ValueError, OSError) as exc:
            record("api:" + role, False, type(exc).__name__ + ": " + str(exc))
            continue
        if network:
            try:
                socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
                record("dns:" + role, True, host)
            except OSError as exc:
                record("dns:" + role, False, type(exc).__name__ + ": " + str(exc))
    return {
        "ready": all(row["ok"] for row in rows),
        "benchmarks": list(benchmarks),
        "checks": rows,
        "paid_calls": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        action="store_true",
        help="Also check public API hostname resolution",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS)
    )
    args = parser.parse_args(argv)
    report = check(args.network, args.benchmarks)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
