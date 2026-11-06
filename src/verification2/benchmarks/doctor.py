"""Readiness checks; never starts a solver or makes paid model calls."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
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
    "agentrewardbench": {"arb", "browser", "web", "docker"},
}
LOCKS = {
    "tau2": "raw/tau2-bench/uv.lock",
    "swe": "raw/runtimes/swe/requirements.lock",
    "web": "raw/runtimes/web/requirements.lock",
}

WEB_ENV = (
    "WA_HOMEPAGE",
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_FULL_RESET",
)


def check(network=False, benchmarks=BENCHMARKS):
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
        "web": "raw/runtimes/web/.venv/bin/python",
    }.items():
        if name not in needs:
            continue
        python = ROOT / relative
        module = {
            "tau2": "tau2",
            "swe": "swebench,minisweagent",
            "web": "browsergym,agent_reward_bench",
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
    if "web" in needs:
        record(
            "docker-compose",
            bool(docker) and probe([docker, "compose", "version"]),
            "Docker Compose v2 required for native websites",
        )
    for benchmark in ("swebench_verified", "agentrewardbench"):
        if benchmark in benchmarks:
            path = ROOT / "raw/tasks" / (benchmark + ".jsonl")
            record("tasks:" + benchmark, path.is_file(), str(path))
    hashes = runtime_hashes()
    for name, path in LOCKS.items():
        if name in needs:
            record(
                "lock:" + path,
                hashes[path] != "missing",
                "present" if hashes[path] != "missing" else "missing; run make setup",
            )
    for name in WEB_ENV if "web" in needs else ():
        value = os.environ.get(name)
        record(
            "web:" + name,
            bool(value),
            "configured" if value else "missing; see .env.example and make web-setup",
        )
        if network and value:
            try:
                with urllib.request.urlopen(
                    value.rstrip("/") + "/status" if name == "WA_FULL_RESET" else value,
                    timeout=5,
                ) as response:
                    ready = response.status == 200 and (
                        name != "WA_FULL_RESET"
                        or response.read().strip() == b"Ready for duty!"
                    )
                record(
                    "reachable:" + name,
                    ready,
                    "responding" if ready else "native service is not ready",
                )
            except (ValueError, OSError) as exc:
                record("reachable:" + name, False, type(exc).__name__)
    for role in ("SOLVER", "JEV", "VERIFIER"):
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
    if "web" in needs:
        browsers = ROOT / "raw/browsers"
        python = ROOT / "raw/runtimes/web/.venv/bin/python"
        ready = (
            python.is_file()
            and browsers.exists()
            and probe(
                [
                    str(python),
                    "-c",
                    "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(headless=True); b.close(); p.stop()",
                ]
            )
        )
        record(
            "chromium",
            ready,
            "headless launch passed"
            if ready
            else "missing browser/runtime or system libraries; run make setup and make setup-system",
        )
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
