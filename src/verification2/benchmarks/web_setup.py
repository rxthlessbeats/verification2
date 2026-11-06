"""Repeatable WebArena website setup using the pinned upstream deployment scripts."""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .setup import ROOT, run, source

WEBSITE_SOURCE = {
    "url": "https://github.com/gasse/webarena-setup.git",
    "revision": "8d0e0a20d50e4a8c02d1520195182e19fc3c080f",
    "directory": "webarena-setup",
    "archive_sha256": "f73c47cc7fd0d1e90b563fd6b68562bf1377e607db92ee0ac061eb1893658fd7",
}
ARCHIVES = (
    "shopping_final_0712.tar",
    "shopping_admin_final_0719.tar",
    "postmill-populated-exposed-withimg.tar",
    "gitlab-populated-final-port8023.tar",
    "openstreetmap-website-db.tar.gz",
    "openstreetmap-website-web.tar.gz",
    "openstreetmap-website.tar.gz",
    "wikipedia_en_all_maxi_2022-05.zim",
)
CONTAINERS = (
    "shopping",
    "shopping_admin",
    "forum",
    "gitlab",
    "wikipedia",
    "openstreetmap-website-db-1",
    "openstreetmap-website-web-1",
)


def inspect_containers():
    result = {}
    for name in CONTAINERS:
        p = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            result[name] = p.stdout.strip()
    return result


def save_state(state, receipt):
    receipt["containers"] = inspect_containers()
    temporary = state / "setup.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(state / "setup.json")


def recreate(state, receipt):
    current = inspect_containers()
    if current != receipt["containers"]:
        raise ValueError(
            "Website container ownership changed; existing containers were preserved"
        )
    receipt["status"] = "preparing"
    save_state(state, receipt)
    try:
        if current:
            run(["docker", "rm", "-f", *current])
        for script in (
            "03_docker_create_containers.sh",
            "04_docker_start_containers.sh",
            "05_docker_patch_containers.sh",
        ):
            run(["bash", script], cwd=state)
        if set(inspect_containers()) != set(CONTAINERS):
            raise ValueError(
                "Native setup did not create every expected website container"
            )
        receipt["status"] = "ready"
    except BaseException:
        receipt["status"] = "failed"
        raise
    finally:
        save_state(state, receipt)


def prepare(archives, hostname):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", hostname):
        raise ValueError("Use a hostname or IPv4 address without a URL scheme/path")
    if not shutil.which("docker"):
        raise ValueError(
            "Docker is required; run make setup-system on the execution host"
        )
    run(["docker", "info", "--format", "{{.ServerVersion}}"])
    run(["docker", "compose", "version"])
    missing = [name for name in ARCHIVES if not (archives / name).is_file()]
    if missing:
        raise ValueError(
            "Native website assets missing from "
            + str(archives)
            + ": "
            + ", ".join(missing)
            + ". Download the original benchmark archives using docs/FRESH_RUN.md; existing trajectory JSONL files are preserved."
        )
    upstream = source("websites", WEBSITE_SOURCE) / "webarena"
    state = ROOT / "raw/websites"
    receipt = state / "setup.json"
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    existing = inspect_containers()
    if existing and (previous is None or previous.get("containers") != existing):
        raise ValueError(
            "Existing Docker containers conflict with this setup; they were preserved. Use a dedicated benchmark host or resolve the named conflicts."
        )
    if previous and (
        previous.get("hostname") != hostname or previous.get("source") != WEBSITE_SOURCE
    ):
        raise ValueError(
            "Existing website setup uses a different hostname/source; it was preserved"
        )
    if not state.exists():
        shutil.copytree(upstream, state)
    variables = (upstream / "00_vars.sh").read_text()
    variables = variables.replace(
        'PUBLIC_HOSTNAME="YOUR_HOSTNAME_HERE"', f'PUBLIC_HOSTNAME="{hostname}"'
    )
    variables = variables.replace(
        'ARCHIVES_LOCATION="./"',
        "ARCHIVES_LOCATION=" + shlex.quote(str(archives.resolve())),
    )
    variables = (
        variables.replace("MAP_PORT=443", "MAP_PORT=3000")
        .replace("HOMEPAGE_PORT=80", "HOMEPAGE_PORT=4399")
        .replace("GITLAB_PORT=9001", "GITLAB_PORT=8023")
    )
    (state / "00_vars.sh").write_text(variables)
    if not previous or previous.get("status") != "ready":
        run(["bash", "01_docker_load_images.sh"], cwd=state)
    values = {
        "WA_HOMEPAGE": f"http://{hostname}:4399",
        "WA_SHOPPING": f"http://{hostname}:8082",
        "WA_SHOPPING_ADMIN": f"http://{hostname}:8083/admin",
        "WA_REDDIT": f"http://{hostname}:8080",
        "WA_GITLAB": f"http://{hostname}:8023",
        "WA_MAP": f"http://{hostname}:3000",
        "WA_WIKIPEDIA": f"http://{hostname}:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "WA_FULL_RESET": f"http://{hostname}:7565",
    }
    (state / ".env").write_text(
        "".join(f"{name}={value}\n" for name, value in values.items())
    )
    configuration = {
        "source": WEBSITE_SOURCE,
        "hostname": hostname,
        "containers": existing,
        "urls": values,
        "status": "preparing",
    }
    # Native reset semantics, with an ownership check before removing the named benchmark containers.
    (state / "reset_server/reset.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nexec "
        + shlex.quote(sys.executable)
        + " -m verification2.benchmarks.web_setup --reset\n"
    )
    server = state / "reset_server/server.py"
    server.write_text(
        server.read_text().replace("(('', args.port)", "(('127.0.0.1', args.port)")
    )
    if not previous or previous.get("status") != "ready":
        recreate(state, configuration)
    else:
        for script in (
            "04_docker_start_containers.sh",
            "05_docker_patch_containers.sh",
        ):
            run(["bash", script], cwd=state)
        configuration["status"] = "ready"
        save_state(state, configuration)
    print(
        "Website state ready. Start make web-serve in another terminal; then scripts/experiment.sh NAME doctor."
    )


def serve():
    state = ROOT / "raw/websites"
    receipt = json.loads((state / "setup.json").read_text())
    if (
        receipt.get("status") != "ready"
        or receipt["containers"] != inspect_containers()
    ):
        raise ValueError(
            "Website containers changed after setup; inspect before resetting"
        )
    python = ROOT / "raw/runtimes/web/.venv/bin/python"
    home = state / "webarena-homepage"
    template = (home / "templates/index.backup").read_text()
    for placeholder, key in {
        "SHOPPING_URL": "WA_SHOPPING",
        "SHOPPING_ADMIN_URL": "WA_SHOPPING_ADMIN",
        "GITLAB_URL": "WA_GITLAB",
        "REDDIT_URL": "WA_REDDIT",
        "MAP_URL": "WA_MAP",
        "WIKIPEDIA_URL": "WA_WIKIPEDIA",
    }.items():
        template = template.replace(placeholder, receipt["urls"][key])
    (home / "templates/index.html").write_text(template)
    commands = [
        (
            [
                str(python),
                "-m",
                "flask",
                "--app",
                "app",
                "run",
                "--host=127.0.0.1",
                "--port=4399",
            ],
            home,
        ),
        ([str(python), "server.py", "--port", "7565"], state / "reset_server"),
    ]
    processes = []
    from ..experiments.collection import child_environment

    try:
        for command, directory in commands:
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=directory,
                    env=child_environment(None),
                    start_new_session=True,
                )
            )
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Native website service stopped")
            time.sleep(0.25)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives", type=Path, default=ROOT / "raw/downloads/webarena"
    )
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(argv)
    if args.reset:
        state = ROOT / "raw/websites"
        recreate(state, json.loads((state / "setup.json").read_text()))
    elif args.serve:
        serve()
    else:
        prepare(args.archives, args.hostname)


if __name__ == "__main__":
    main()
