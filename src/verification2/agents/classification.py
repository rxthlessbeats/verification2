"""Jev E/R classification, separate from evidence selection and verification."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

from qera.jsonl import write_jsonl
from qera.overrides import apply
from qera.schema import Override, Role, Span

from ..core.config import ClassificationConfig
from ..core.models import ClassificationResult, JevResponse
from ..core.prompts import JevPrompt, load_prompts
from ..data.inputs import read_inputs, write_quarantine
from ..data.split import file_hash
from ..experiments.protocol import check_stage, code_hash
from .verifier import backend_from_args, digest

JEV_WORKERS = 8


def classification_view(trajectory):
    segments, targets = [], {}
    for segment in trajectory.segments:
        spans = segment.spans or [
            Span(start=0, end=len(segment.text), role=segment.role)
        ]
        segments.append(
            {
                "seg_id": segment.seg_id,
                "speaker": segment.speaker.value,
                "step": segment.step,
                "text": segment.text,
                "call": segment.call.model_dump(mode="json") if segment.call else None,
                "protected_spans": [
                    [s.start, s.end, s.role.value]
                    for s in spans
                    if s.role in (Role.Q, Role.A)
                ],
            }
        )
        for index, span in enumerate(spans):
            if span.role in (Role.E, Role.R) and segment.text[span.start : span.end]:
                targets[f"s{segment.seg_id}_text_{index}"] = {
                    "seg_id": segment.seg_id,
                    "field": "text",
                    "span_index": index,
                    "start": span.start,
                    "end": span.end,
                }
        if segment.call is not None and segment.call_role in (Role.E, Role.R):
            targets[f"s{segment.seg_id}_call"] = {
                "seg_id": segment.seg_id,
                "field": "call",
            }
    return {
        "traj_id": trajectory.traj_id,
        "benchmark": trajectory.benchmark,
        "segments": segments,
        "targets": targets,
    }


def task_context(view, wanted="Q"):
    return [
        {"seg_id": s["seg_id"], "speaker": s["speaker"], "text": s["text"][start:end]}
        for s in view["segments"]
        for start, end, role in s["protected_spans"]
        if role == wanted
    ]


def classification_request(model, view, prompt: JevPrompt, question_id, answer=False):
    """One E/R piece per call with the task (Q) as context, within Jev's 32k window."""
    target = view["targets"][question_id]
    segment = view["segments"][target["seg_id"]]
    content = (
        segment["text"][target["start"] : target["end"]]
        if target["field"] == "text"
        else segment["call"]
    )
    # Question IDs are opaque to Jev, so identify the target in instructions.
    question = {
        "type": "choice",
        "instructions": {
            "task": prompt.instruction,
            **target,
            "speaker": segment["speaker"],
            "content": content,
        },
        "criteria": prompt.criteria.model_dump(),
    }
    return {
        "model": model,
        "state": {"benchmark": view["benchmark"], "task": task_context(view)}
        | ({"answer": task_context(view, "A")} if answer else {}),
        "questions": {question_id: question},
    }


def parse_answer(response, question_id, choices=("E", "R")):
    answers = JevResponse.model_validate(response).answers
    if set(answers) != {question_id}:
        raise ValueError("Jev must answer exactly the supplied question ID")
    if answers[question_id].choice not in choices:
        raise ValueError("Jev chose an unknown option")
    return {"answer": answers[question_id].model_dump()}


def ask_jev(backend, calls, choices=("E", "R")):
    def ask(question_id):
        return backend.call(
            calls[question_id],
            parser=lambda raw: parse_answer(raw, question_id, choices),
        )

    with ThreadPoolExecutor(JEV_WORKERS) as pool:
        return dict(zip(calls, pool.map(ask, calls)))


def total_usage(usages):
    if any(not isinstance(u, dict) for u in usages):
        return None  # Unknown usage stays unknown.
    keys = {k for u in usages for k, v in u.items() if type(v) is int}
    return {k: sum(u.get(k, 0) for u in usages) for k in sorted(keys)}


def parse_classification(response, trajectory, view):
    answers = JevResponse.model_validate(response).answers
    if set(answers) != set(view["targets"]):
        raise ValueError("Jev must answer exactly the supplied E/R question IDs")
    overrides = {}
    for question_id, target in view["targets"].items():
        answer = answers[question_id]
        role, seg_id = answer.choice, target["seg_id"]
        if role not in ("E", "R"):
            raise ValueError("Jev must classify each piece as E or R")
        row = overrides.setdefault(
            (trajectory.traj_id, seg_id),
            {"traj_id": trajectory.traj_id, "seg_id": seg_id, "source": "classifier"},
        )
        segment = trajectory.segments[seg_id]
        if target["field"] == "call":
            row["call_role"] = role
        elif segment.spans:
            spans = row.setdefault(
                "spans", [[s.start, s.end, s.role.value] for s in segment.spans]
            )
            spans[target["span_index"]][2] = role
        else:
            row["role"] = role
    # Existing apply() preserves manual overrides and checks reasoning provenance.
    list(apply([trajectory.model_copy(deep=True)], overrides))
    return {"overrides": list(overrides.values())}


def jev_backend(args):
    backend = backend_from_args(args)
    model = args.model or ""
    if not backend.systemone or not (
        model == "typesafe-ai/jev" or model.startswith("jev-")
    ):
        raise ValueError(
            "Jev classification requires typesafe-ai/jev or a jev-* model and /systemone endpoint"
        )
    return backend


def run(args):
    args = ClassificationConfig.model_validate(vars(args))
    backend = jev_backend(args)
    bundle = load_prompts(args.prompt_file)
    if args.output.exists():
        raise ValueError("Output already exists; use a new output directory")
    trajectories, quarantine = read_inputs(args.input, unique_ids=True)
    check_stage(trajectories, args, bundle, "jev")
    if not trajectories and not quarantine:
        raise ValueError("No trajectories found")
    resolved = {t.benchmark: bundle.classification(t.benchmark) for t in trajectories}
    prompt = next(iter(resolved.values())) if len(resolved) == 1 else bundle.jev
    prompt_snapshot = prompt.model_dump(mode="json")
    args.output.mkdir(parents=True)
    write_quarantine(args.output, quarantine)
    model = args.model
    manifest = {
        "stage": "jev",
        "source_hash": code_hash(),
        "mode": args.backend,
        "model": model,
        "instruction": prompt.instruction,
        "configuration": args.model_dump(mode="json"),
        "prompt": prompt_snapshot,
        "benchmark_prompts": {
            b: p.model_dump(mode="json") for b, p in resolved.items()
        },
        "prompt_hash": digest(prompt_snapshot),
        "api_format": "systemone",
        "question_type": "choice",
        "choices": ["E", "R"],
        "timeout": args.timeout,
        "n_trajectories": len(trajectories),
        "quarantined_n": len(quarantine),
        "complete": False,
        "input_hash": digest([t.model_dump(mode="json") for t in trajectories]),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_overrides, errors, accepted = {}, 0, []
    with (
        (args.output / "requests.jsonl").open("w", encoding="utf-8") as requests,
        (args.output / "results.jsonl").open("w", encoding="utf-8") as results,
    ):
        for trajectory in trajectories:
            prompt = resolved[trajectory.benchmark]
            view = classification_view(trajectory)
            row = ClassificationResult(
                traj_id=trajectory.traj_id,
                benchmark=trajectory.benchmark,
                task_id=trajectory.task_id,
            )
            start = time.perf_counter()
            try:
                if view["targets"]:
                    calls = {
                        q: classification_request(model, view, prompt, q)
                        for q in view["targets"]
                    }
                    ids = {q: digest(c) for q, c in calls.items()}
                    row.request_id = digest(list(ids.values()))
                    for q, call in calls.items():
                        requests.write(
                            json.dumps(
                                {"request_id": ids[q], "request": call},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    requests.flush()
                    responses = ask_jev(backend, calls)
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
                        answers = {q: r["answer"] for q, r in responses.items()}
                        row.status = "ok"
                        row.overrides = [
                            Override.model_validate(o)
                            for o in parse_classification(
                                {"answers": answers}, trajectory, view
                            )["overrides"]
                        ]
                    for override in row.overrides:
                        all_overrides[trajectory.traj_id, override.seg_id] = (
                            override.model_dump(mode="json", exclude_none=True)
                        )
                else:
                    row.status = "unchanged"
            except Exception as exc:  # noqa: BLE001 - Preserve diagnostics without publishing partial data.
                errors += 1
                row.status = "error"
                row.error = type(exc).__name__ + ": " + str(exc)
            if row.status in ("ok", "unchanged"):
                accepted.append(trajectory)
            row.classification_ms = (time.perf_counter() - start) * 1000
            record = ClassificationResult.model_validate(row.model_dump()).model_dump(
                mode="json"
            )
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
    if args.backend == "http" and accepted:
        out = args.output / "overrides.jsonl"
        out.write_text(
            "".join(
                json.dumps(o, ensure_ascii=False) + "\n" for o in all_overrides.values()
            ),
            encoding="utf-8",
        )
        overrides_hash = file_hash(out)

        def classified():
            for trajectory in apply(accepted, all_overrides):
                prompt = resolved[trajectory.benchmark]
                trajectory.meta["jev_classification"] = {
                    "model": model,
                    "instruction_hash": digest(prompt.instruction),
                    "prompt_hash": digest(prompt.model_dump(mode="json")),
                    "source_hash": manifest["input_hash"],
                    "overrides_sha256": overrides_hash,
                }
                yield trajectory

        target = args.output / "trajectories.jsonl"
        write_jsonl(classified(), str(target))
        manifest.update(
            complete=not errors and not quarantine,
            published_n=len(accepted),
            trajectories_sha256=file_hash(target),
            overrides_sha256=overrides_hash,
        )
    manifest["errors"] = errors
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Jev {args.backend}: {len(trajectories)} trajectories, {errors} errors -> {args.output}"
    )
    return 1 if errors or quarantine else 0
