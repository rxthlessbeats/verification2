"""Native solver API calls retain provider reasoning and usage without local estimates."""

import json
import os
import time

from ..agents.verifier import Backend


def _wire_call(call):
    """The trajectory stores {id, function, args}. The endpoint expects OpenAI tool calls."""
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name") or "unknown"
        arguments = function.get("arguments")
    else:
        name = function or call.get("name") or "unknown"
        arguments = call.get("args")
        if arguments is None:
            arguments = call.get("arguments") or {}
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call.get("id") or "call",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _wire_message(message):
    permitted = {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "reasoning_content",
        "reasoning",
        "reasoning_text",
    }
    kept = {k: v for k, v in message.items() if k in permitted and v is not None}
    if kept.get("tool_calls"):
        kept["tool_calls"] = [_wire_call(call) for call in kept["tool_calls"]]
        if kept.get("content") == "":
            kept["content"] = None
    return kept


def completion(job, messages, tools=None):
    payload = {
        "model": job.model,
        "temperature": 0,
        "messages": [_wire_message(message) for message in messages],
    }
    if job.solver_instruction:
        payload["messages"] = [
            dict(m, content=job.solver_instruction) if m["role"] == "system" else m
            for m in payload["messages"]
        ]
    if tools:
        payload["tools"] = tools
    if job.reasoning_effort:
        payload["reasoning_effort"] = job.reasoning_effort
    backend = Backend(
        mode="http",
        endpoint=job.endpoint,
        api_key=os.environ["SOLVER_API_KEY"],
        cache_dir=job.cache_dir,
    )
    row = {"status": "error", "request": payload}
    start = time.perf_counter()
    try:
        raw, hit, key = backend.completion(payload)
        row.update(
            status="ok",
            raw_response=raw,
            usage=raw.get("usage"),
            cache_hit=hit,
            cache_key=key,
        )
        return raw
    except Exception as exc:  # noqa: BLE001 - Native SDK logs must never receive credential-bearing exceptions.
        detail = str(exc)
        # Keep only the status line. Server bodies can contain request echoes.
        safe = (
            detail
            if detail.startswith("Model endpoint returned HTTP ") and len(detail) < 80
            else type(exc).__name__
        )
        row["error"] = safe
        raise RuntimeError("Native API request failed: " + safe) from None
    finally:
        row["duration_ms"] = (time.perf_counter() - start) * 1000
        with (job.output / "calls.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
