"""Run independent fresh stages with durable receipts and a reasoning-aware pilot."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from qera.jsonl import read_jsonl

from ..agents import classification, selection
from ..agents import verifier as pipeline
from ..benchmarks.doctor import check
from ..core.prompts import load_prompts
from ..core.registry import resolve
from ..data.split import BENCHMARKS, digest, file_hash, load_split, split_id
from . import summary
from .collection import add_limits, collect
from .protocol import check_heldout, check_pilot, code_hash, runtime_hashes
from .report import main as report
from .store import add_rerun, rerun_benchmarks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=["pilot", "train", "test"], default="pilot"
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["QERA"])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    add_limits(parser)
    add_rerun(parser)
    args = parser.parse_args(argv)
    rerun = rerun_benchmarks(args.rerun)
    splits = [load_split(b) for b in args.benchmarks]
    prompts = load_prompts(args.prompt_file)
    solver = resolve("solver", os.environ.get("SOLVER_MODEL"))
    check_heldout(splits, args.partition, args.protocol, solver, prompts)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    result = {
        "status": "running",
        "split_id": split_id(splits),
        "partition": args.partition,
        "source_hash": code_hash(),
        "runtime_hashes": runtime_hashes(),
        "rerun": sorted(rerun),
        "limit": args.limit,
        "stages": {},
    }
    receipts_file = args.output / "receipts.json"
    receipts = json.loads(receipts_file.read_text()) if receipts_file.exists() else {}
    cache = args.cache_dir or args.output.parent / "api_cache"

    def save():
        report_path.write_text(json.dumps(result, indent=2) + "\n")

    def stage(name, fingerprint, invoke, artifact):
        previous = receipts.get(fingerprint)
        if previous:
            path = Path(previous["artifact"])
            if previous["sha256"] is not None and not path.is_file():
                raise ValueError("Saved stage artifact is missing: " + str(path))
            if path.exists() and file_hash(path) != previous["sha256"]:
                raise ValueError("Saved stage artifact was modified: " + str(path))
            if previous["code"] == 0 or not args.retry_failed:
                result["stages"][name] = previous
                return path if path.exists() else None
        suffix = 0
        output = args.output / name
        while output.exists():
            suffix += 1
            output = args.output / f"{name}.attempt{suffix}"
        print(f"[{name}] started -> {output}", flush=True)
        code = invoke(output)
        path = output / artifact
        receipt = {
            "output": str(output),
            "artifact": str(path),
            "sha256": file_hash(path) if path.exists() else None,
            "code": code,
        }
        receipts[fingerprint] = receipt
        receipts_file.write_text(json.dumps(receipts, indent=2) + "\n")
        result["stages"][name] = receipt
        save()
        return path if path.exists() else None

    save()
    try:
        readiness = check(network=True, benchmarks=args.benchmarks)
        (args.output / "readiness.json").write_text(
            json.dumps(readiness, indent=2) + "\n"
        )
        if not readiness["ready"]:
            result.update(
                status="blocked",
                reason="Native/API readiness checks failed; no model requests were attempted",
            )
            return 2
        if args.partition != "pilot":
            check_pilot(args.output.parent / "pilot/report.json", splits)
        solver_dir = args.output / "solver"
        collected = collect(
            splits,
            solver_dir,
            args.partition,
            solver.model,
            solver.endpoint,
            backend="http",
            cache_dir=cache,
            prompt_file=args.prompt_file,
            retry_failed=args.retry_failed,
            rerun=rerun,
            limit=args.limit,
            jobs=args.jobs,
            reasoning_effort=solver.reasoning_effort,
        )
        source = solver_dir / "trajectories.jsonl"
        if not source.exists():
            raise ValueError(
                "No successfully scored native episodes; raw failures were preserved"
            )
        trajectories = list(read_jsonl([source]))
        if args.partition == "pilot":
            expected = set(args.benchmarks)
            captured = {
                b: any(
                    t.benchmark == b
                    and any(
                        s.kind == "reasoning" and s.text.strip() for s in t.by_role("R")
                    )
                    for t in trajectories
                )
                for b in expected
            }
            result["readable_reasoning"] = captured
            if not all(captured.values()) or collected:
                raise ValueError(
                    "Pilot needs a completed native episode with captured readable reasoning for every benchmark"
                )
        jev_model = os.environ.get("JEV_MODEL")
        jev_endpoint = resolve("jev", jev_model).endpoint
        classified = stage(
            "jev",
            digest(
                [
                    "jev",
                    file_hash(source),
                    jev_model,
                    jev_endpoint,
                    {
                        b: prompts.classification(b).model_dump()
                        for b in {t.benchmark for t in trajectories}
                    },
                    file_hash(args.protocol) if args.protocol else None,
                    code_hash(),
                ]
            ),
            lambda out: classification.run(
                SimpleNamespace(
                    input=[source],
                    output=out,
                    backend="http",
                    model=jev_model,
                    endpoint=jev_endpoint,
                    timeout=None,
                    cache_dir=cache,
                    prompt_file=args.prompt_file,
                    protocol=args.protocol,
                )
            ),
            "trajectories.jsonl",
        )
        if classified is None:
            raise ValueError("No validated Jev classifications were published")
        summary.jev(classified.parent)
        if "QJevA" in args.conditions:
            classified = stage(
                "selector",
                digest(
                    [
                        "selector",
                        file_hash(classified),
                        jev_model,
                        jev_endpoint,
                        {
                            b: prompts.selection(b).model_dump()
                            for b in {t.benchmark for t in trajectories}
                        },
                        file_hash(args.protocol) if args.protocol else None,
                        code_hash(),
                    ]
                ),
                lambda out: selection.run(
                    SimpleNamespace(
                        input=[classified],
                        output=out,
                        backend="http",
                        model=jev_model,
                        endpoint=jev_endpoint,
                        timeout=None,
                        cache_dir=cache,
                        prompt_file=args.prompt_file,
                        protocol=args.protocol,
                    )
                ),
                "trajectories.jsonl",
            )
            if classified is None:
                raise ValueError("No validated Jev selections were published")
        verifier = resolve("verifier", os.environ.get("VERIFIER_MODEL"))
        model, endpoint = verifier.model, verifier.endpoint
        verified = stage(
            "verifier",
            digest(
                [
                    "verifier",
                    file_hash(classified),
                    model,
                    endpoint,
                    verifier.reasoning_effort,
                    prompts.verifier.model_dump(),
                    args.conditions,
                    file_hash(args.protocol) if args.protocol else None,
                    code_hash(),
                ]
            ),
            lambda out: pipeline.run(
                SimpleNamespace(
                    input=[classified],
                    output=out,
                    backend="http",
                    model=model,
                    endpoint=endpoint,
                    reasoning_effort=verifier.reasoning_effort,
                    timeout=None,
                    max_tokens=None,
                    cache_dir=cache,
                    prompt_file=args.prompt_file,
                    conditions=args.conditions,
                    protocol=args.protocol,
                    jobs=args.jobs,
                )
            ),
            "results.jsonl",
        )
        if verified is None:
            raise ValueError("Verifier produced no result records")
        summary.verifier(verified.parent)
        analysis = args.output / "analysis"
        counter = 0
        while analysis.exists():
            counter += 1
            analysis = args.output / f"analysis.attempt{counter}"
        report(
            [
                "--input",
                str(verified.parent),
                "--output",
                str(analysis),
                "--benchmarks",
                *args.benchmarks,
                "--partition",
                args.partition,
                *(["--limit", str(args.limit)] if args.limit else []),
            ]
        )
        summary.report(analysis)
        failures = collected or any(s["code"] for s in result["stages"].values())
        result.update(
            status="partial" if failures else "passed",
            analysis=str(analysis),
            note="Pilot validates the workflow; it is not a benchmark accuracy claim."
            if args.partition == "pilot"
            else "Quality, coverage and exclusions must be interpreted together.",
        )
        return 1 if failures else 0
    except Exception as exc:  # noqa: BLE001 - Preserve an explicit terminal status and all stage diagnostics.
        result.update(status="error", error=type(exc).__name__ + ": " + str(exc))
        return 1
    except KeyboardInterrupt:
        result.update(
            status="interrupted",
            error="Interrupted; inspect partial native attempts before --retry-failed",
        )
        return 130
    finally:
        save()
        print(f"Experiment {result['status']}: {report_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
