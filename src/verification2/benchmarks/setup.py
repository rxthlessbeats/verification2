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
import zipfile
from pathlib import Path

from qera.jsonl import read as read_rows

from ..core.config import Name, Settings

_checkout = Path(__file__).resolve().parents[3]
ROOT = _checkout if (_checkout / "src/verification2").is_dir() else Path.cwd()
SOURCES = json.loads(Path(__file__).with_name("native_sources.json").read_text())
# The pinned v5 harness needs image/log_parser/eval_script fields, absent from princeton-nlp copies.
SWE_TASK_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
APPWORLD_DATA_URL = "https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle"
APPWORLD_DATA_SHA256 = (
    "fd9f9608c2ec71ed0ac25c3633a738b9129a318a129e31230425b9188e508250"
)
AGENTDOJO_VERSION = "v1.2.2"
BIRD_TRAIN_ZIP = 8919543554
BIRD_DEV_ZIP = 346207293
BIRD_ANNOTATIONS = {
    "train.jsonl": (
        "https://huggingface.co/datasets/birdsql/bird23-train-filtered/resolve/main/data/train-00000-of-00001.jsonl",
        "2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8",
    ),
    "mini_dev_sqlite.jsonl": (
        "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite.jsonl",
        "926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606",
    ),
}


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


def export_tasks(python, benchmark, env=None):
    target = ROOT / "raw/tasks" / (benchmark + ".jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Reusing task data: {target}", flush=True)
        return
    if benchmark == "swebench_verified":
        code = (
            "from datasets import load_dataset; import sys; "
            f"load_dataset('SWE-bench/SWE-bench_Verified', split='test', revision='{SWE_TASK_REVISION}').to_json(sys.argv[1])"
        )
    elif benchmark == "agentdojo":
        code = (
            "import json,sys; from pathlib import Path; "
            f"from agentdojo.task_suite.load_suites import get_suites; "
            f"suites=get_suites('{AGENTDOJO_VERSION}'); "
            "rows=["
            "{'suite': name, 'task_id': task_id, 'prompt': task.PROMPT} "
            "for name, suite in sorted(suites.items()) "
            "for task_id, task in sorted(suite.user_tasks.items())]; "
            "Path(sys.argv[1]).write_text(''.join(json.dumps(r)+'\\n' for r in rows))"
        )
    elif benchmark == "appworld":
        code = (
            "import json,sys; from pathlib import Path; "
            "from appworld.task import load_task_ids; "
            "rows=[]; "
            "rows += [{'task_id': i, 'dataset': 'train', 'partition': 'train'} "
            "for i in load_task_ids('train')]; "
            "rows += [{'task_id': i, 'dataset': 'dev', 'partition': 'train'} "
            "for i in load_task_ids('dev')]; "
            "rows += [{'task_id': i, 'dataset': 'test_normal', 'partition': 'test'} "
            "for i in load_task_ids('test_normal')]; "
            "rows += [{'task_id': i, 'dataset': 'test_challenge', 'partition': 'excluded'} "
            "for i in load_task_ids('test_challenge')]; "
            "Path(sys.argv[1]).write_text(''.join(json.dumps(r)+'\\n' for r in rows))"
        )
    elif benchmark == "scienceworld":
        temporary = target.with_suffix(".tmp")
        try:
            run(
                [
                    python,
                    Path(__file__).with_name("export_scienceworld.py"),
                    temporary,
                ]
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return
    elif benchmark == "bird":
        _export_bird(target)
        return
    else:
        raise ValueError("Unknown task export: " + benchmark)
    temporary = target.with_suffix(".tmp")
    try:
        run([python, "-c", code, temporary], env=env)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_appworld_data(python):
    root = ROOT / "raw/appworld-root"
    data = root / "data"
    if data.is_dir() and (data / "datasets" / "train.txt").is_file():
        print(f"Reusing AppWorld data: {data}", flush=True)
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "raw/downloads/data-0.1.0.bundle"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        download = archive.with_suffix(".tmp")
        run(["curl", "-fsSL", APPWORLD_DATA_URL, "-o", download])
        download.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != APPWORLD_DATA_SHA256:
        raise ValueError("AppWorld data bundle checksum mismatch: " + str(archive))
    run(
        [
            python,
            "-c",
            (
                "import os, shutil, sys; "
                "from pathlib import Path; "
                "from appworld.common.constants import PASSWORD, SALT; "
                "from appworld.common.utils import unpack_bundle; "
                "root=Path(sys.argv[1]); data=root/'data'; "
                "shutil.rmtree(data, ignore_errors=True); "
                "os.environ['APPWORLD_ROOT']=str(root); "
                "unpack_bundle(sys.argv[2], str(root), PASSWORD, SALT); "
                "assert (data/'datasets'/'train.txt').is_file()"
            ),
            root,
            archive,
        ],
        env={**os.environ, "APPWORLD_ROOT": str(root)},
    )
    return root


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    run(["curl", "-fL", "--retry", "3", "-o", temporary, url])
    temporary.replace(path)


def _export_bird(target):
    directory = ROOT / "raw/bird/annotations"
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in BIRD_ANNOTATIONS.items():
        path = directory / name
        _download(url, path)
        if file_hash(path) != checksum:
            raise ValueError("BIRD annotation checksum mismatch: " + str(path))
    rows = []
    for partition, filename in (
        ("train", "train.jsonl"),
        ("test", "mini_dev_sqlite.jsonl"),
    ):
        for index, row in enumerate(read_rows(directory / filename)):
            task_id = (
                f"train-{index}"
                if partition == "train"
                else f"mini-{row['question_id']}"
            )
            rows.append(
                {
                    "task_id": task_id,
                    "domain": row["db_id"],
                    "partition": partition,
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "evidence": row.get("evidence") or "",
                    "SQL": row["SQL"],
                }
            )
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"exported {len(rows)} bird tasks", flush=True)


def prepare_bird_databases():
    """Unpack the official train and dev SQLite files. Mini-Dev uses the dev files."""
    root = ROOT / "raw/bird/databases"
    if root.is_dir() and any(root.rglob("*.sqlite")):
        print(f"Reusing BIRD databases: {root}", flush=True)
        return
    downloads = ROOT / "raw/downloads"
    archives = {
        "bird-train.zip": (
            BIRD_TRAIN_ZIP,
            "train/train_databases.zip",
            "bird-train-databases.zip",
        ),
        "bird-dev.zip": (
            BIRD_DEV_ZIP,
            "dev_20240627/dev_databases.zip",
            "bird-dev-databases.zip",
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, (size, member, inner_name) in archives.items():
        outer = downloads / name
        if not outer.is_file() or outer.stat().st_size != size:
            raise ValueError(
                f"Missing {outer} ({size} bytes). "
                "Train: https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip "
                "Dev: https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
            )
        inner = downloads / inner_name
        if not inner.is_file():
            with (
                zipfile.ZipFile(outer) as package,
                package.open(member) as src,
                inner.open("wb") as out,
            ):
                shutil.copyfileobj(src, out)
        with zipfile.ZipFile(inner) as package:
            members = [
                item
                for item in package.namelist()
                if not item.startswith("__MACOSX") and "/._" not in item
            ]
            package.extractall(root, members)
    if not any(root.rglob("*.sqlite")):
        raise ValueError("BIRD database archives contained no sqlite files")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=[
            "tau2",
            "swe",
            "agentdojo",
            "appworld",
            "scienceworld",
            "bird",
            "all",
        ],
        default="all",
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
    if args.benchmark in ("agentdojo", "all"):
        agentdojo = source("agentdojo")
        python = install_runtime("agentdojo", [*common, f"-e {agentdojo}"])
        export_tasks(python, "agentdojo")
    if args.benchmark in ("appworld", "all"):
        appworld = source("appworld")
        # AppWorld pins pydantic v1; keep this runtime isolated from the v2 common set.
        python = install_runtime(
            "appworld",
            [f"-e {appworld}", "jinja2==3.1.6", "openai==1.66.0"],
        )
        run(
            [
                python,
                "-c",
                "from appworld.install import install_package; install_package()",
            ]
        )
        root = prepare_appworld_data(python)
        export_tasks(
            python,
            "appworld",
            env={**os.environ, "APPWORLD_ROOT": str(root)},
        )
    if args.benchmark in ("scienceworld", "all"):
        # ponytail: the published wheel and a bundled JDK, not the Scala checkout.
        python = install_runtime(
            "scienceworld", [*common, "scienceworld==1.2.3", "jdk4py"]
        )
        export_tasks(python, "scienceworld")
    if args.benchmark in ("bird", "all"):
        python = install_runtime("bird", common)
        prepare_bird_databases()
        export_tasks(python, "bird")
    print(
        "Native runtimes prepared. Run scripts/experiment.sh NAME doctor before collecting trajectories."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
