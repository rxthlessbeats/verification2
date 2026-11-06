"""Jev usefulness labels for E/R pieces; QJevA shows only the useful ones."""

import json
import time

from qera.jsonl import write_jsonl

from ..core.config import ClassificationConfig
from ..core.models import SelectionResult
from ..core.prompts import SelectorCriteria, load_prompts
from ..data.inputs import read_inputs, write_quarantine
from ..data.split import file_hash
from ..experiments.protocol import check_stage, code_hash
from .classification import (
    ask_jev,
    classification_request,
    classification_view,
    jev_backend,
    total_usage,
)
from .verifier import digest

CHOICES = tuple(SelectorCriteria.model_fields)
KEEP = "useful"


def run(args):
    args = ClassificationConfig.model_validate(vars(args))
    backend = jev_backend(args)
    bundle = load_prompts(args.prompt_file)
    if args.output.exists():
        raise ValueError("Output already exists; use a new output directory")
    trajectories, quarantine = read_inputs(args.input, unique_ids=True)
    check_stage(trajectories, args, bundle, "selector")
    if not trajectories and not quarantine:
        raise ValueError("No trajectories found")
    prompts = {t.benchmark: bundle.selection(t.benchmark) for t in trajectories}
    args.output.mkdir(parents=True)
    write_quarantine(args.output, quarantine)
    manifest = {
        "stage": "selector",
        "source_hash": code_hash(),
        "mode": args.backend,
        "model": args.model,
        "configuration": args.model_dump(mode="json"),
        "benchmark_prompts": {b: p.model_dump(mode="json") for b, p in prompts.items()},
        "api_format": "systemone",
        "question_type": "choice",
        "choices": CHOICES,
        "keep": KEEP,
        "n_trajectories": len(trajectories),
        "quarantined_n": len(quarantine),
        "complete": False,
        "input_hash": digest([t.model_dump(mode="json") for t in trajectories]),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    errors, accepted = 0, []
    with (
        (args.output / "requests.jsonl").open("w", encoding="utf-8") as requests,
        (args.output / "results.jsonl").open("w", encoding="utf-8") as results,
    ):
        for trajectory in trajectories:
            prompt = prompts[trajectory.benchmark]
            view = classification_view(trajectory)
            row = SelectionResult(
                traj_id=trajectory.traj_id,
                benchmark=trajectory.benchmark,
                task_id=trajectory.task_id,
            )
            start = time.perf_counter()
            try:
                calls = {
                    q: classification_request(args.model, view, prompt, q, answer=True)
                    for q in view["targets"]
                }
                ids = {q: digest(c) for q, c in calls.items()}
                row.request_id = digest(list(ids.values()))
                for q, call in calls.items():
                    requests.write(
                        json.dumps({"request_id": ids[q], "request": call}) + "\n"
                    )
                requests.flush()
                responses = ask_jev(backend, calls, CHOICES) if calls else {}
                failed = [r for r in responses.values() if r["status"] == "error"]
                pending = [r for r in responses.values() if r["status"] != "ok"]
                row.usage = total_usage([r["usage"] for r in responses.values()])
                row.raw_response = {q: r["raw"] for q, r in responses.items()}
                row.cache_hit = all(r.get("cache_hit") for r in responses.values())
                if failed:
                    row.status = "error"
                    row.error = f"{len(failed)}/{len(calls)} Jev calls failed; first: {failed[0]['error']}"
                    errors += 1
                elif pending:
                    row.status = pending[0]["status"]
                else:
                    row.status = "ok"
                    row.labels = {
                        q: r["answer"]["choice"] for q, r in responses.items()
                    }
            except Exception as exc:  # noqa: BLE001 - Preserve diagnostics without publishing partial data.
                errors += 1
                row.status = "error"
                row.error = type(exc).__name__ + ": " + str(exc)
            if row.status == "ok":
                trajectory.meta["jev_selection"] = {
                    "model": args.model,
                    "prompt_hash": digest(prompt.model_dump(mode="json")),
                    "keep": [q for q, label in row.labels.items() if label == KEEP],
                }
                accepted.append(trajectory)
            row.selection_ms = (time.perf_counter() - start) * 1000
            record = SelectionResult.model_validate(row.model_dump())
            results.write(record.model_dump_json() + "\n")
            results.flush()
    if args.backend == "http" and accepted:
        target = args.output / "trajectories.jsonl"
        write_jsonl(accepted, str(target))
        manifest.update(
            complete=not errors and not quarantine,
            published_n=len(accepted),
            trajectories_sha256=file_hash(target),
        )
    manifest["errors"] = errors
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Jev selection {args.backend}: {len(trajectories)} trajectories, {errors} errors -> {args.output}"
    )
    return 1 if errors or quarantine else 0
