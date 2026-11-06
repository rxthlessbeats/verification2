"""Executed by tau2's Python; only the shared Pydantic config is imported."""

import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

from ..core.cache import cached_response
from ..core.config import NativeRunConfig


def collect(config):
    config = NativeRunConfig.model_validate(config)
    if importlib.metadata.version("tau2") != config.tau2_version:
        raise ValueError("Native tau2 version differs from the frozen configuration")
    from tau2.data_model.simulation import TextRunConfig
    from tau2.environment.toolkit import get_tool_types
    from tau2.registry import registry
    from tau2.run import run_domain
    from tau2.utils import llm_utils

    original = llm_utils.completion
    original_conversion = getattr(llm_utils, "to_litellm_messages", None)
    original_cost = getattr(llm_utils, "get_response_cost", None)
    if original_cost:

        def provider_cost(response):
            value = (response.to_dict().get("usage") or {}).get("cost")
            try:
                return (
                    float(value)
                    if not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0
                    else None
                )
            except (TypeError, ValueError):
                return None

        llm_utils.get_response_cost = provider_cost
    if original_conversion:

        def preserve_reasoning(messages):
            converted = []
            for message in messages:
                parts = original_conversion([message])
                raw = getattr(message, "raw_data", None) or {}
                choices = raw.get("choices") or []
                recorded = choices[0].get("message", {}) if choices else {}
                for part in parts:
                    if part["role"] == "assistant":
                        for field in (
                            "reasoning_content",
                            "reasoning",
                            "reasoning_text",
                        ):
                            if recorded.get(field):
                                part[field] = recorded[field]
                converted.extend(parts)
            return converted

        llm_utils.to_litellm_messages = preserve_reasoning
    original_instruction = None
    if config.solver_instruction:
        from tau2.agent import llm_agent

        original_instruction = llm_agent.AGENT_INSTRUCTION
        llm_agent.AGENT_INSTRUCTION = config.solver_instruction
    native = config.output / "native"
    native.mkdir()
    environment = registry.get_env_constructor(config.domain)()
    types = {
        side: {
            name: kind.value.upper() for name, kind in get_tool_types(toolkit).items()
        }
        if toolkit is not None
        else {}
        for side, toolkit in (
            ("assistant", environment.tools),
            ("user", environment.user_tools),
        )
    }
    (native / "tool_types.json").write_text(
        json.dumps({config.domain: types}, indent=2), encoding="utf-8"
    )
    models = {config.model, config.user_model}
    llm_utils.litellm.disable_cache()
    with (native / "calls.jsonl").open("w", encoding="utf-8") as calls:

        def completion(*, model, **kwargs):
            kwargs.pop("api_key", None)
            requested = model.removeprefix("openai/")
            conversation = requested in models
            actual = requested if conversation else config.judge_model
            # Inject credentials at transport time: native info/debug logs must not store them.
            request = {
                **kwargs,
                "model": "openai/" + actual,
                "api_base": config.endpoint.removesuffix("/chat/completions"),
                "num_retries": 0,
                "extra_headers": {
                    "User-Agent": "verification2/0.1",
                    "x-opencode-session": "verification2-solver-" + config.run_id,
                },
            }
            for option in ("timeout", "max_tokens"):
                if option not in request and getattr(config, option) is not None:
                    request[option] = getattr(config, option)
            row = {
                "purpose": "conversation" if conversation else "benchmark_judge",
                "requested_model": model,
                "request": request,
                "status": "error",
            }
            start = time.perf_counter()
            try:
                identity = {k: v for k, v in request.items() if k != "extra_headers"}
                with cached_response(
                    config.cache_dir, config.endpoint, identity
                ) as cached:
                    response = (
                        llm_utils.litellm.ModelResponse(**cached["raw"])
                        if cached["hit"]
                        else original(
                            **request, api_key=os.environ.get("SOLVER_API_KEY")
                        )
                    )
                    cached.update(raw=response.to_dict(), accepted=True)
                    row.update(
                        status="ok",
                        response=response.to_dict(),
                        cache_hit=cached["hit"],
                        cache_key=cached["key"],
                    )
                return response
            except Exception as exc:  # noqa: BLE001 - Keep credential-bearing transport errors out of logs.
                row["error"] = type(exc).__name__
                raise RuntimeError(
                    "Native completion failed: " + type(exc).__name__
                ) from None
            finally:
                row["duration_ms"] = (time.perf_counter() - start) * 1000
                calls.write(json.dumps(row, ensure_ascii=False) + "\n")
                calls.flush()

        llm_utils.completion = completion
        try:
            run_domain(
                TextRunConfig(
                    domain=config.domain,
                    task_ids=config.task_ids,
                    task_split_name="base",
                    agent="llm_agent",
                    user="user_simulator",
                    llm_agent="openai/" + config.model,
                    llm_user="openai/" + config.user_model,
                    llm_args_agent={"temperature": 0}
                    | (
                        # litellm rejects reasoning_effort for models it does not know.
                        {"extra_body": {"reasoning_effort": config.reasoning_effort}}
                        if config.reasoning_effort
                        else {}
                    ),
                    llm_args_user={"temperature": 0},
                    num_trials=config.num_trials,
                    **(
                        {"max_steps": config.max_steps}
                        if config.max_steps is not None
                        else {}
                    ),
                    max_concurrency=1,
                    seed=config.seed,
                    save_to=str(native),
                    max_retries=0,
                    hallucination_retries=0,
                    auto_review=False,
                    verbose_logs=False,
                )
            )
        finally:
            llm_utils.completion = original
            if original_conversion:
                llm_utils.to_litellm_messages = original_conversion
            if original_cost:
                llm_utils.get_response_cost = original_cost
            if original_instruction is not None:
                llm_agent.AGENT_INSTRUCTION = original_instruction


if __name__ == "__main__":
    collect(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
