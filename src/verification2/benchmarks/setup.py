"""Install isolated native runtimes and reuse matching source/data downloads."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ..core.config import Name, Settings

_checkout = Path(__file__).resolve().parents[3]
ROOT = _checkout if (_checkout / "src/verification2").is_dir() else Path.cwd()
SOURCES = json.loads(Path(__file__).with_name("native_sources.json").read_text())
# The pinned v5 harness needs image/log_parser/eval_script fields, absent from princeton-nlp copies.
SWE_TASK_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"


class Source(Settings):
    url: Name
    revision: Name
    directory: Name
    archive_sha256: Name | None = None


def run(argv, **kwargs):
    kwargs.setdefault("cwd", ROOT)
    kwargs.setdefault(
        "env",
        {
            **os.environ,
            "HF_HOME": os.environ.get("HF_HOME", str(ROOT / "raw/huggingface")),
        },
    )
    subprocess.run([str(a) for a in argv], check=True, **kwargs)


def source(name, specification=None):
    spec = Source.model_validate(specification or SOURCES[name])
    path = ROOT / "raw" / spec.directory
    if path.exists():
        stamp = path / ".verification2-source.json"
        if stamp.exists():
            recorded = json.loads(stamp.read_text())
            if recorded["revision"] != spec.revision or any(
                not (path / filename).is_file()
                or hashlib.sha256((path / filename).read_bytes()).hexdigest()
                != expected
                for filename, expected in recorded["files"].items()
                if not (name == "tau2" and filename == "uv.lock")
            ):
                raise ValueError(
                    f"{path} differs from the pinned source; existing files were preserved"
                )
            return path
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        # uv repairs tau2's stale local package version in its lock; runtime hashes track it separately.
        paths = ["--", ".", ":(exclude)uv.lock"] if name == "tau2" else []
        dirty = subprocess.run(
            ["git", "-C", str(path), "diff", "--quiet", "HEAD", *paths], check=False
        ).returncode
        if revision != spec.revision or dirty:
            raise ValueError(
                f"{path} differs from the pinned source; existing files were preserved"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".checkout-") as temporary:
        archive = ROOT / "raw/downloads" / (spec.revision + ".tar.gz")
        archive.parent.mkdir(parents=True, exist_ok=True)
        repository = spec.url.removeprefix("https://github.com/").removesuffix(".git")
        if not archive.exists():
            download = Path(temporary) / "download.tar.gz"
            run(
                [
                    "curl",
                    "-fsSL",
                    f"https://codeload.github.com/{repository}/tar.gz/{spec.revision}",
                    "-o",
                    download,
                ]
            )
            download.replace(archive)
        if (
            spec.archive_sha256
            and hashlib.sha256(archive.read_bytes()).hexdigest() != spec.archive_sha256
        ):
            raise ValueError("Source archive checksum mismatch: " + str(archive))
        with tarfile.open(archive) as package:
            package.extractall(temporary, filter="data")
        (checkout,) = [p for p in Path(temporary).iterdir() if p.is_dir()]
        recorded = {
            "revision": spec.revision,
            "url": spec.url,
            "files": {
                str(p.relative_to(checkout)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in checkout.rglob("*")
                if p.is_file()
            },
        }
        (checkout / ".verification2-source.json").write_text(
            json.dumps(recorded, indent=2) + "\n"
        )
        checkout.rename(path)
    return path


def install_runtime(name, requirements):
    root = ROOT / "raw" / "runtimes" / name
    python = root / ".venv/bin/python"
    root.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(requirements) + "\n"
    specification, lock = root / "requirements.in", root / "requirements.lock"
    identity = hashlib.sha256(contents.encode()).hexdigest()
    previous = root / "specification.sha256"
    if not python.exists():
        run(["uv", "venv", "--no-cache", "--python", "3.12", python.parents[1]])
    if (
        not lock.exists()
        or not previous.exists()
        or previous.read_text().strip() != identity
    ):
        specification.write_text(contents)
        run(
            [
                "uv",
                "pip",
                "compile",
                "--no-cache",
                "--python-version",
                "3.12",
                specification,
                "--output-file",
                lock,
            ]
        )
        previous.write_text(identity + "\n")
    run(["uv", "pip", "sync", "--no-cache", "--python", python, lock])
    return python


def export_tasks(python, benchmark):
    target = ROOT / "raw/tasks" / (benchmark + ".jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Reusing task data: {target}", flush=True)
        return
    code = (
        "from datasets import load_dataset; import sys; "
        f"load_dataset('SWE-bench/SWE-bench_Verified', split='test', revision='{SWE_TASK_REVISION}').to_json(sys.argv[1])"
        if benchmark == "swebench_verified"
        else "import importlib.util,json,sys; from pathlib import Path; "
        "p=Path(importlib.util.find_spec('webarena').origin).parent/'test.raw.json'; "
        "rows=json.loads(p.read_text()); "
        "Path(sys.argv[1]).write_text(''.join(json.dumps(r)+'\\n' for r in rows))"
    )
    temporary = target.with_suffix(".tmp")
    try:
        run([python, "-c", code, temporary])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=["tau2", "swe", "web", "all"], default="all"
    )
    args = parser.parse_args(argv)
    if not shutil.which("uv"):
        raise ValueError(
            "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
        )
    common = ["pydantic==2.14.1", "pyyaml==6.0.3"]
    if args.benchmark in ("tau2", "all"):
        root = source("tau2")
        run(
            [
                "uv",
                "sync",
                "--no-cache",
                "--frozen",
                "--project",
                root,
                "--no-default-groups",
            ]
        )
    if args.benchmark in ("swe", "all"):
        mini, swe = source("mini"), source("swe")
        python = install_runtime("swe", [*common, f"-e {mini}", f"-e {swe}"])
        export_tasks(python, "swebench_verified")
    if args.benchmark in ("web", "all"):
        arb, browser = source("arb"), source("browser")
        python = install_runtime(
            "web",
            [
                *common,
                "setuptools==80.9.0",
                "openai>=1.66.0",
                "flask>=3",
                "browsergym==0.13.1",
                f"-e {arb}",
            ],
        )
        run(
            [python, "-m", "playwright", "install", "chromium"],
            env={**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(ROOT / "raw/browsers")},
        )
        export_tasks(python, "agentrewardbench")
        assert (browser / "demo_agent/agent.py").is_file()
    print(
        "Native runtimes prepared. Run scripts/experiment.sh NAME doctor before collecting trajectories."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
