"""Run frozen QERA trajectories through a stateless verification backend."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from qera.render import build_verifier_input

from ..core.cache import cached_response
from ..core.config import (
    Condition,
    Endpoint,
    Mode,
    ProviderConfig,
    Timeout,
    VerifierConfig,
)
from ..core.models import (
    ChatResponse,
    MessagesResponse,
    MetricSummary,
    VerifierAnswer,
    VerifierRequest,
    VerifierResult,
)
from ..core.prompts import load_prompts
from ..core.registry import max_output_tokens, resolve, spec
from ..data.inputs import read_inputs, write_quarantine
from ..data.split import digest
from ..experiments.protocol import check_stage, code_hash

CONDITIONS = get_args(Condition)


def model_request(model, prompt, content, max_tokens, endpoint, effort=None):
    # A trajectory ending mid-dialogue otherwise invites the model to reply as the solver.
    content = f"<trajectory>\n{content}\n</trajectory>\n\n{prompt}"
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if urllib.parse.urlsplit(endpoint or "").path.rstrip("/").endswith("/messages"):
        request["system"] = prompt
        request["messages"] = request["messages"][1:]
        if effort:
            request["output_config"] = {"effort": effort}
    elif effort:
        request["reasoning_effort"] = effort
    return VerifierRequest.model_validate(request).model_dump(
        mode="json", exclude_none=True
    )


def parse_response(text):
    """Accept plain JSON or one fenced JSON object; never guess a verdict."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    return VerifierAnswer.model_validate_json(text).model_dump()


class Backend(BaseModel):
    """Send validated requests to a complete API endpoint; keep credentials out of logs."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    mode: Mode
    endpoint: Endpoint | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    timeout: Timeout | None = None
    cache_dir: Path | None = None

    @model_validator(mode="after")
    def require_endpoint(self):
        if self.mode == "http" and not self.endpoint:
            raise ValueError("http mode requires an endpoint")
        return self

    @property
    def anthropic(self):
        return (
            urllib.parse.urlsplit(self.endpoint or "")
            .path.rstrip("/")
            .endswith("/messages")
        )

    @property
    def systemone(self):
        return (
            urllib.parse.urlsplit(self.endpoint or "")
            .path.rstrip("/")
            .endswith("/systemone")
        )

    def _send(self, request):
        headers = {"Content-Type": "application/json"}
        if urllib.parse.urlsplit(self.endpoint).hostname == "opencode.ai":
            headers["User-Agent"] = "verification2/0.1"
            headers["x-opencode-session"] = "verification2-" + digest(request)
        if self.api_key:
            if self.anthropic:
                headers["x-api-key"] = self.api_key.get_secret_value()
            else:
                headers["Authorization"] = "Bearer " + self.api_key.get_secret_value()
        if self.anthropic:
            headers["anthropic-version"] = "2023-06-01"
        wire = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(wire, timeout=self.timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not persist server error bodies, credentials or signed URLs.
            raise RuntimeError(
                "Model endpoint returned HTTP " + str(exc.code)
            ) from None
        return raw

    def completion(self, request):
        with cached_response(self.cache_dir, self.endpoint, request) as state:
            raw = state["raw"] if state["hit"] else self._send(request)
            ChatResponse.model_validate(raw)
            state.update(raw=raw, accepted=True)
            return raw, state["hit"], state["key"]

    def call(self, request, parser=parse_response):
        if self.mode == "dry-run":
            return {
                "status": "prepared",
                "verdict": None,
                "reason": None,
                "usage": None,
                "raw": None,
            }
        if self.mode == "mock":
            # Constant response validates transport only; never consult labels.
            return {
                "status": "mock",
                "verdict": "correct",
                "reason": "MOCK: fixed acceptance; not a model judgment.",
                "usage": None,
                "raw": None,
            }
        with cached_response(self.cache_dir, self.endpoint, request) as state:
            raw = state["raw"] if state["hit"] else self._send(request)
            result = self._parse(raw, request, parser)
            if result["status"] == "ok":
                state.update(raw=raw, accepted=True)
            return {**result, "cache_hit": state["hit"], "cache_key": state["key"]}

    def _parse(self, raw, request, parser):
        usage = raw.get("usage") if isinstance(raw, dict) else None
        try:
            if not isinstance(raw, dict):
                raise TypeError("Expected a model response object")
            finish_reason = None
            if self.systemone:
                content = raw
            elif self.anthropic:
                envelope = MessagesResponse.model_validate(raw)
                finish_reason = envelope.stop_reason
                content = "\n".join(
                    block.text for block in envelope.content if block.type == "text"
                )
            else:
                choice = ChatResponse.model_validate(raw).choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content
            if finish_reason in ("max_tokens", "length"):
                raise ValueError(
                    f"Model output was truncated at max_tokens={request.get('max_tokens')}; "
                    "no complete verdict was accepted."
                )
            if not self.systemone and not isinstance(content, str):
                raise ValueError("Expected model response text")
            parsed = parser(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "error": type(exc).__name__ + ": " + str(exc),
                "usage": usage,
                "raw": raw,
            }
        return {"status": "ok", **parsed, "usage": usage, "raw": raw}


def model_args(parser, role, prompt=True, generation=True):
    parser.add_argument(
        "--model",
        default=os.environ.get(role + "_MODEL"),
        help="Model name; its endpoint and API key come from models.yaml",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Optional request timeout; no local timeout by default",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=os.environ.get("VERIFICATION_CACHE_DIR"),
        help="Reuse first valid responses for identical endpoint/request pairs",
    )
    if generation:
        parser.add_argument(
            "--max-tokens",
            type=int,
            help="Optional explicit output limit; otherwise use provider defaults/maximum",
        )
    if prompt:
        parser.add_argument(
            "--prompt-file",
            type=Path,
            help="Validated YAML prompt bundle; defaults to the packaged prompts.yaml",
        )


def with_endpoint(args, role):
    resolved = resolve(role, args.model) if args.model else None
    args.endpoint = resolved and resolved.endpoint
    args.reasoning_effort = resolved and resolved.reasoning_effort
    return args


def backend_from_args(args):
    args = ProviderConfig.model_validate(
        {
            key: getattr(args, key)
            for key in ProviderConfig.model_fields
            if hasattr(args, key)
        }
    )
    api_key = None
    if args.backend == "http":
        api_key_env = spec(args.model).api_key_env
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{args.model} needs an API key in {api_key_env}")
    return Backend(
        mode=args.backend,
        endpoint=args.endpoint,
        api_key=api_key,
        timeout=args.timeout,
        cache_dir=args.cache_dir,
    )


def metrics(rows, conditions):
    """Only real valid model responses are scored, on common paired samples."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["traj_id"])].append(row)
    paired = {
        key
        for key, group in groups.items()
        if len(group) == len(conditions) and all(r["status"] == "ok" for r in group)
    }
    result = []
    for benchmark in sorted({r["benchmark"] for r in rows}):
        for condition in conditions:
            group = [
                r
                for r in rows
                if r["benchmark"] == benchmark and r["condition"] == condition
            ]
            valid = [
                r
                for r in group
                if (benchmark, r["traj_id"]) in paired
                and type(r["label_correct"]) is bool
            ]
            bad = [r for r in valid if not r["label_correct"]]
            good = [r for r in valid if r["label_correct"]]
            rejected_correct = sum(r["verdict"] == "incorrect" for r in good)
            accepted_incorrect = sum(r["verdict"] == "correct" for r in bad)
            result.append(
                MetricSummary(
                    benchmark=benchmark,
                    condition=condition,
                    attempted=len(group),
                    errors=sum(r["status"] == "error" for r in group),
                    paired_labelled_n=len(valid),
                    incorrect_n=len(bad),
                    correct_n=len(good),
                    false_rejections=rejected_correct,
                    false_acceptances=accepted_incorrect,
                    false_rejection_rate=rejected_correct / len(good) if good else None,
                    false_acceptance_rate=accepted_incorrect / len(bad)
                    if bad
                    else None,
                ).model_dump(mode="json")
            )
    return result


def run(args):
    args = VerifierConfig.model_validate(vars(args))
    backend = backend_from_args(args)
    bundle = load_prompts(args.prompt_file)
    prompt_config = bundle.verifier
    prompt = prompt_config.instruction
    conditions = tuple(args.conditions)
    if args.output.exists():
        raise ValueError("Output already exists; use a new output directory")
    trajectories, quarantine = read_inputs(args.input)
    protocol = check_stage(trajectories, args, bundle, "verifier")
    if not trajectories and not quarantine:
        raise ValueError("No trajectories found")
    maximum = protocol.verifier_max_tokens if protocol else args.max_tokens
    if (
        protocol is None
        and trajectories
        and maximum is None
        and backend.anthropic
        and args.backend == "http"
    ):
        maximum = max_output_tokens(args.model)
    args.output.mkdir(parents=True)
    write_quarantine(args.output, quarantine)
    model = args.model or "offline-transport-mock"
    manifest = {
        "stage": "verifier",
        "source_hash": code_hash(),
        "mode": args.backend,
        "model": model,
        "conditions": conditions,
        "api_format": "anthropic_messages" if backend.anthropic else "chat_completions",
        "temperature": 0,
        "max_tokens": maximum,
        "reasoning_effort": args.reasoning_effort,
        "instruction": prompt,
        "input_hash": digest([t.model_dump(mode="json") for t in trajectories]),
        "configuration": args.model_dump(mode="json"),
        "prompt": prompt_config.model_dump(mode="json"),
        "prompt_hash": digest(prompt_config.model_dump(mode="json")),
        "n_trajectories": len(trajectories),
        "quarantined_n": len(quarantine),
        "endpoint_configured": bool(args.endpoint),
        "note": "Tokens/cost come only from API usage/raw_response. No evidence length budget. Messages output maximum comes from models.yaml on live runs; dry-run requests may omit it. Mock/dry-run quality metrics are null.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rows = []

    def check(trajectory, condition):
        request_line = None
        row = VerifierResult(
            benchmark=trajectory.benchmark,
            traj_id=trajectory.traj_id,
            task_id=trajectory.task_id,
            split_id=trajectory.meta.get("split_id"),
            native_task_key=trajectory.meta.get("native_task_key"),
            partition=trajectory.split if trajectory.meta.get("split_id") else None,
            trajectory_hash=digest(trajectory.model_dump(mode="json")),
            condition=condition,
            label_correct=trajectory.label.correct,
            visible_r=any(
                (role.value == "R" and bool(text.strip()))
                for segment in trajectory.segments
                for role, text in segment.pieces()
            )
            or any(
                segment.call is not None and segment.call_role.value == "R"
                for segment in trajectory.segments
            ),
        )
        condition_start = time.perf_counter()
        start = condition_start
        try:
            rendered = build_verifier_input(trajectory, condition)
            request = model_request(
                model,
                prompt,
                rendered.text,
                maximum,
                args.endpoint,
                args.reasoning_effort,
            )
            request_id = digest(
                [trajectory.benchmark, trajectory.traj_id, condition, request]
            )
            # This line is safe for model transport: no labels or hidden fields.
            request_line = json.dumps(
                {"request_id": request_id, "request": request}, ensure_ascii=False
            )
            row.request_id = request_id
            row.included_seg_ids = rendered.included_seg_ids
            start = time.perf_counter()
            try:
                response = backend.call(request)
            finally:
                row.verification_ms = (time.perf_counter() - start) * 1000
            row.status = response["status"]
            row.verdict, row.reason = response.get("verdict"), response.get("reason")
            row.error, row.usage = response.get("error"), response["usage"]
            row.raw_response = response["raw"]
            row.cache_hit, row.cache_key = (
                response.get("cache_hit", False),
                response.get("cache_key"),
            )
        except Exception as exc:  # noqa: BLE001 - Preserve other conditions when a model call fails.
            row.error = type(exc).__name__ + ": " + str(exc)
        row.total_ms = (time.perf_counter() - condition_start) * 1000
        record = VerifierResult.model_validate(row.model_dump()).model_dump(mode="json")
        return request_line, record

    pairs = [(t, c) for t in trajectories for c in conditions]
    with (
        (args.output / "requests.jsonl").open("w", encoding="utf-8") as requests,
        (args.output / "results.jsonl").open("w", encoding="utf-8") as results,
        ThreadPoolExecutor(args.jobs) as pool,
    ):
        for request_line, record in pool.map(lambda pair: check(*pair), pairs):
            if request_line is not None:
                requests.write(request_line + "\n")
                requests.flush()
            rows.append(record)
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
    summary = metrics(rows, conditions)
    with (args.output / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(MetricSummary.model_fields))
        writer.writeheader()
        writer.writerows(summary)
    errors = sum(r["status"] == "error" for r in rows)
    print(
        f"{args.backend}: {len(trajectories)} trajectories, {len(rows)} checks, {errors} errors -> {args.output}"
    )
    return 1 if errors or quarantine else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from .classification import run as classify
    from .selection import run as select
    from .solver import add_args as solver_args
    from .solver import run as solve

    solver = commands.add_parser(
        "solve", help="Collect fresh tau2 trajectories in its native environment"
    )
    solver_args(solver)
    classifier = commands.add_parser(
        "classify", help="Classify E/R with Jev and freeze a new dataset"
    )
    classifier.add_argument("--input", nargs="+", required=True)
    classifier.add_argument("--output", type=Path, required=True)
    classifier.add_argument("--backend", choices=["dry-run", "http"], default="dry-run")
    classifier.add_argument("--protocol", type=Path)
    model_args(classifier, "JEV", generation=False)
    selector = commands.add_parser(
        "select", help="Label E/R pieces useful, misleading or irrelevant with Jev"
    )
    selector.add_argument("--input", nargs="+", required=True)
    selector.add_argument("--output", type=Path, required=True)
    selector.add_argument("--backend", choices=["dry-run", "http"], default="dry-run")
    selector.add_argument("--protocol", type=Path)
    model_args(selector, "JEV", generation=False)
    runner = commands.add_parser(
        "run", aliases=["verify"], help="Verify frozen classified trajectories"
    )
    runner.add_argument("--input", nargs="+", required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument(
        "--backend", choices=["dry-run", "mock", "http"], default="dry-run"
    )
    runner.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["QERA"])
    runner.add_argument("--protocol", type=Path)
    runner.add_argument("--jobs", type=int, default=1, help="Checks sent at once")
    model_args(runner, "VERIFIER")
    args = parser.parse_args(argv)
    if args.command == "solve":
        raise SystemExit(solve(with_endpoint(args, "solver")))
    if args.command == "classify":
        raise SystemExit(classify(with_endpoint(args, "jev")))
    if args.command == "select":
        raise SystemExit(select(with_endpoint(args, "jev")))
    raise SystemExit(run(with_endpoint(args, "verifier")))


if __name__ == "__main__":
    main()
