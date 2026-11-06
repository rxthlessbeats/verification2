"""Collect native tau2 runs in an isolated environment, then adapt their logs."""

import json
import os
import subprocess
import uuid
from collections import Counter
from pathlib import Path

from qera.adapters.tau2 import convert_file
from qera.jsonl import write_jsonl

from ..core.config import NativeRunConfig, SolverConfig
from ..core.prompts import load_prompts
from ..data.split import file_hash
from .verifier import backend_from_args, model_args

TAU2_VERSION = "1.0.1"


def add_args(parser):
    parser.add_argument(
        "--domain", choices=["airline", "retail", "telecom"], required=True
    )
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["dry-run", "http"], default="dry-run")
    parser.add_argument(
        "--tau2-root",
        type=Path,
        default=Path(os.environ.get("TAU2_ROOT", "raw/tau2-bench")),
    )
    parser.add_argument("--tau2-python", type=Path)
    parser.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL"))
    parser.add_argument("--judge-model", default=os.environ.get("TAU2_JUDGE_MODEL"))
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional override of the native benchmark horizon",
    )
    parser.add_argument("--seed", type=int, default=42)
    model_args(parser, "SOLVER", prompt=False)
    parser.add_argument("--prompt-file", type=Path)


def run(args):
    args = SolverConfig.model_validate(vars(args))
    backend = backend_from_args(args)
    root = args.tau2_root.resolve()
    python = (args.tau2_python or root / ".venv/bin/python").absolute()
    revision = None
    if args.backend == "http":
        if not python.is_file():
            raise ValueError("Native tau2 environment missing; see docs/FRESH_RUN.md")
        archive = root / ".verification2-source.json"
        if archive.exists():
            from ..benchmarks.setup import ROOT, source

            if root != ROOT / "raw/tau2-bench":
                raise ValueError(
                    "Archive-based native tau2 must use the pinned raw/tau2-bench path"
                )
            source("tau2")
            revision = json.loads(archive.read_text())["revision"]
        else:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            pinned = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "v" + TAU2_VERSION + "^{commit}"],
                text=True,
            ).strip()
            if (
                revision != pinned
                or subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        "diff",
                        "--quiet",
                        "HEAD",
                        "--",
                        "src",
                        "data",
                    ],
                    check=False,
                ).returncode
            ):
                raise ValueError(
                    "Native checkout must match clean tau2 v" + TAU2_VERSION
                )
    if args.output.exists():
        raise ValueError("Output already exists; use a new output directory")
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    manifest = {
        "stage": "solver",
        "mode": args.backend,
        "run_id": uuid.uuid4().hex[:12],
        "model": args.model,
        "user_model": args.user_model or args.model,
        "judge_model": args.judge_model or args.model,
        "endpoint": args.endpoint,
        "tau2_version": TAU2_VERSION,
        "tau2_revision": revision,
        "tau2_root": str(root),
        "domain": args.domain,
        "task_ids": args.task_ids,
        "num_trials": args.num_trials,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "temperature": 0,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "output": str(output),
        "complete": False,
        "configuration": args.model_dump(mode="json"),
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "solver_instruction": load_prompts(args.prompt_file).solver("tau2_bench"),
    }
    NativeRunConfig.model_validate(manifest)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.backend == "dry-run":
        print(
            f"Solver prepared: {len(args.task_ids) * args.num_trials} native simulations -> {output}"
        )
        return 0
    from ..experiments.collection import child_environment

    env = child_environment(backend.api_key)
    env["TAU2_DATA_DIR"] = str(root / "data")
    with (output / "native.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(python), "-m", "verification2.benchmarks.tau2_worker", str(path)],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        manifest["error"] = (
            "Native tau2 runner failed; inspect native.log and preserved native artifacts"
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(manifest["error"])
        return 1
    native = output / "native/results.json"
    types = json.loads((native.parent / "tool_types.json").read_text(encoding="utf-8"))
    trajectories = list(
        convert_file(str(native), tau2_root=str(root), tool_types=types)
    )
    expected = Counter(
        {args.domain + "/" + task: args.num_trials for task in args.task_ids}
    )
    trials = {
        (args.domain + "/" + task, trial)
        for task in args.task_ids
        for trial in range(args.num_trials)
    }
    if (
        Counter(t.task_id for t in trajectories) != expected
        or {(t.task_id, t.meta["trial"]) for t in trajectories} != trials
        or any(t.label.correct is None for t in trajectories)
    ):
        raise ValueError(
            "Native output has missing/duplicate tasks or missing evaluation labels; artifacts preserved"
        )
    for trajectory in trajectories:
        trajectory.traj_id += ".run_" + manifest["run_id"]
        trajectory.meta["solver_collection"] = {
            "run_id": manifest["run_id"],
            "tau2_version": TAU2_VERSION,
            "revision": revision,
            "user_model": manifest["user_model"],
            "judge_model": manifest["judge_model"],
        }
    target = output / "trajectories.jsonl"
    write_jsonl(trajectories, str(target))
    manifest.update(
        complete=True,
        n_trajectories=len(trajectories),
        native_sha256=file_hash(native),
        trajectories_sha256=file_hash(target),
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Solver: {len(trajectories)} frozen trajectories -> {target}")
    return 0
