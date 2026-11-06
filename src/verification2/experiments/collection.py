"""Collect each frozen native task once, retaining completed episodes on restart."""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from pydantic import Field

from qera.jsonl import read_jsonl, write_jsonl
from qera.schema import Trajectory

from ..agents.verifier import backend_from_args
from ..benchmarks.setup import ROOT, source
from ..core.config import Endpoint, Name, Settings
from ..core.prompts import load_prompts
from ..data.split import (
    BENCHMARKS,
    SEED,
    Task,
    digest,
    file_hash,
    load_task,
    select,
    split_id,
)
from . import summary
from .store import (
    default_store,
    identity,
    load,
    location,
    now,
    save,
)


class Job(Settings):
    task: Task
    seed: int = Field(strict=True)
    model: Name
    endpoint: Endpoint
    output: Path
    cache_dir: Path
    solver_instruction: str | None = None
    reasoning_effort: Name | None = None


def child_environment(key):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY")
        and k not in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    }
    env["SOLVER_API_KEY"] = key.get_secret_value() if key else ""
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / "raw/browsers")
    env["HF_HOME"] = os.environ.get("HF_HOME", str(ROOT / "raw/huggingface"))
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["MSWEA_GLOBAL_CONFIG_DIR"] = str(ROOT / "raw/config/mini-swe-agent")
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    return env


def collect_job(job, key, prompt_file=None):
    load_task(job.task)
    if job.task.benchmark == "tau2_bench":
        from ..agents.solver import run

        args = SimpleNamespace(
            backend="http",
            model=job.model,
            endpoint=job.endpoint,
            output=job.output,
            domain=job.task.domain,
            task_ids=[job.task.task_id],
            tau2_root=ROOT / "raw/tau2-bench",
            tau2_python=None,
            user_model=None,
            judge_model=None,
            num_trials=1,
            max_steps=None,
            max_tokens=None,
            timeout=None,
            seed=job.seed,
            reasoning_effort=job.reasoning_effort,
            cache_dir=job.cache_dir,
            prompt_file=prompt_file,
        )
        if run(args):
            raise RuntimeError("Native tau2 collection failed; raw artifacts preserved")
        (trajectory,) = read_jsonl([job.output / "trajectories.jsonl"])
        return trajectory
    runtime = "swe" if job.task.benchmark == "swebench_verified" else "web"
    for name in ("mini", "swe") if runtime == "swe" else ("arb", "browser"):
        source(name)  # Existing pinned files must still match before any paid call.
    python = ROOT / "raw/runtimes" / runtime / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("Native runtime missing; run make setup")
    job.output.mkdir(parents=True)
    configuration = job.output / "job.json"
    configuration.write_text(job.model_dump_json(indent=2) + "\n")
    module = (
        "verification2.benchmarks.swe_worker"
        if runtime == "swe"
        else "verification2.benchmarks.web_worker"
    )
    with (job.output / "native.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(python), "-m", module, str(configuration)],
            cwd=ROOT,
            env=child_environment(key),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            "Native worker failed; inspect " + str(job.output / "native.log")
        )
    return Trajectory.model_validate_json((job.output / "trajectory.json").read_text())


def add_limits(parser):
    parser.add_argument(
        "--limit",
        type=int,
        help="Tasks per benchmark from the partition, balanced over domains (default all)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="Tasks solved and checks verified at once"
    )


def assign(trajectory, split, task, partition, prompt_hash):
    """Attach this run's provenance; stored trajectories may come from another run."""
    trajectory.split = partition
    provenance = {
        "split_id": split,
        "native_task_key": task.key,
        "native_source_row_hash": task.source_row_hash,
        "solver_prompt_hash": prompt_hash,
    }
    trajectory.meta = {
        k: v for k, v in trajectory.meta.items() if k not in provenance
    } | provenance
    return trajectory


def collect(
    splits,
    output,
    partition,
    model,
    endpoint,
    backend="dry-run",
    cache_dir=None,
    prompt_file=None,
    retry_failed=False,
    store=None,
    rerun=(),
    limit=None,
    jobs=1,
    reasoning_effort=None,
):
    """Reuse stored trajectories except for benchmarks in rerun, which are solved again."""
    from .protocol import code_hash, runtime_hashes

    prompts = load_prompts(prompt_file)
    run_split = split_id(splits)
    prompt_hash = digest({b: prompts.solver(b) for b in BENCHMARKS})
    tasks = select(splits, partition, limit)
    if not tasks:
        raise ValueError("No tasks match these benchmarks/partition")
    output = Path(output).resolve()
    cache_dir = Path(cache_dir or output / "api_cache").resolve()
    settings = SimpleNamespace(
        backend=backend,
        model=model,
        endpoint=endpoint,
        timeout=None,
        cache_dir=cache_dir,
    )
    transport = backend_from_args(settings)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    configuration = {
        "split_id": run_split,
        "partition": partition,
        "mode": backend,
        "solver_model": model,
        "solver_endpoint": endpoint,
        "solver_reasoning_effort": reasoning_effort,
        "source_hash": code_hash(),
        "runtime_hashes": runtime_hashes(),
        "rerun": sorted(rerun),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("configuration") != configuration:
            raise ValueError(
                "Existing collection has a different configuration (code, solver, "
                "runtimes or --rerun); use a new output directory"
            )
    manifest = {
        "stage": "collection",
        "configuration": configuration,
        "planned_n": len(tasks),
        "complete": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    store = Path(store or default_store())
    runtime = runtime_hashes()

    def publish(key, trajectory, episode, row, replace):
        entry = location(store, key) / "entry.json"
        before = entry.read_text() if entry.is_file() else None
        directory = save(
            store,
            key,
            trajectory,
            episode,
            replace=replace,
            duration_s=row.get("duration_s"),
            solver_endpoint=endpoint,
            source_hash=code_hash(),
            collected_at=now(),
        )
        row["stored"] = str(
            directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory
        ) + (" (replaced)" if before not in (None, entry.read_text()) else "")

    def solve(task):
        load_task(task)
        fingerprint = digest([run_split, task.key])
        key = identity(
            task,
            model,
            SEED,
            prompts.solver(task.benchmark),
            runtime,
            reasoning_effort,
        )
        base = {"task": task.model_dump(mode="json"), "fingerprint": fingerprint}
        rerunning = task.benchmark in rerun
        if backend == "http" and not rerunning and (hit := load(store, key)):
            stored, entry, trajectory = hit
            return (
                base
                | {
                    "status": "ok",
                    "reused": "store",
                    "trajectory_sha256": entry["trajectory_sha256"],
                    "output": str(stored),
                    "duration_s": entry.get("duration_s"),
                    "usage": entry.get("usage"),
                },
                assign(trajectory, run_split, task, partition, prompt_hash),
            )
        directory = output / "episodes" / fingerprint
        result = directory / "result.json"
        row = base | {"status": "prepared", "reused": False}
        if result.exists():
            row = json.loads(result.read_text())
            if row["status"] == "ok":
                completed = Path(row["output"]).resolve()
                if not completed.is_relative_to(output / "episodes"):
                    raise ValueError("Episode result points outside this collection")
                frozen = completed / "frozen.json"
                if file_hash(frozen) != row["trajectory_sha256"]:
                    raise ValueError(
                        "Completed episode was modified: " + str(directory)
                    )
                trajectory = Trajectory.model_validate_json(frozen.read_text())
                row = row | {"reused": True}
                publish(key, trajectory, completed, row, rerunning)
                return row, trajectory
            if not retry_failed:
                return row, None
        if directory.exists():
            if not retry_failed:
                return row | {
                    "status": "error",
                    "error": "Previous attempt has no completed result; inspect it before --retry-failed",
                }, None
            # Never overwrite diagnostics or an unknown in-flight outcome.
            attempt = 1
            while directory.with_name(fingerprint + f".attempt{attempt}").exists():
                attempt += 1
            directory = directory.with_name(fingerprint + f".attempt{attempt}")
        job = Job(
            task=task.model_copy(update={"partition": None}),
            seed=SEED,
            model=model,
            endpoint=endpoint,
            output=directory,
            cache_dir=cache_dir,
            solver_instruction=prompts.solver(task.benchmark),
            reasoning_effort=reasoning_effort,
        )
        row = base | {"status": "prepared", "reused": False}
        trajectory = None
        if backend == "http":
            started = time.monotonic()
            try:
                trajectory = collect_job(job, transport.api_key, prompt_file)
                if (
                    trajectory.label.correct is None
                    or trajectory.solver.origin != "self_generated"
                ):
                    raise ValueError(
                        "Fresh collection requires a new native outcome and self_generated provenance"
                    )
                expected_id = (
                    f"{task.domain}/{task.task_id}"
                    if task.benchmark == "tau2_bench"
                    else task.task_id
                )
                expected_label = (
                    "unit_tests"
                    if task.benchmark == "swebench_verified"
                    else "env_reward"
                )
                if (
                    trajectory.benchmark != task.benchmark
                    or trajectory.task_id != expected_id
                    or trajectory.label.source != expected_label
                ):
                    raise ValueError(
                        "Native trajectory task/benchmark/grader differs from the selected task"
                    )
                assign(trajectory, run_split, task, partition, prompt_hash)
                frozen = directory / "frozen.json"
                frozen.write_text(trajectory.revalidate().model_dump_json() + "\n")
                row.update(
                    status="ok",
                    trajectory_sha256=file_hash(frozen),
                    output=str(directory),
                    duration_s=round(time.monotonic() - started, 1),
                    usage=summary.episode_usage(directory),
                )
            except Exception as exc:  # noqa: BLE001 - Infrastructure errors never become incorrect solver labels.
                trajectory = None
                directory.mkdir(parents=True, exist_ok=True)
                row.update(
                    status="error",
                    error=type(exc).__name__ + ": " + str(exc),
                    output=str(directory),
                    duration_s=round(time.monotonic() - started, 1),
                    cause=summary.native_cause(directory),
                )
            (directory / "result.json").write_text(json.dumps(row, indent=2) + "\n")
            # The stable pointer preserves the first completed attempt and every failed attempt.
            canonical = output / "episodes" / fingerprint
            if directory != canonical:
                (canonical / "result.json").write_text(json.dumps(row, indent=2) + "\n")
            if trajectory is not None:
                publish(key, trajectory, directory, row, rerunning)
        return row, trajectory

    results = [None] * len(tasks)
    with ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(solve, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            print(summary.solver_task(tasks[i], *results[i]), flush=True)
    rows = [row for row, _ in results]
    trajectories = [t for _, t in results if t is not None]
    if backend == "http":
        summary.solver(results)
    if trajectories:
        write_jsonl(trajectories, str(output / "trajectories.jsonl"))
    (output / "task_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    manifest.update(
        complete=backend == "http" and len(trajectories) == len(tasks),
        completed_n=len(trajectories),
        failed_n=sum(r["status"] == "error" for r in rows),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if backend == "dry-run" or manifest["complete"] else 1
