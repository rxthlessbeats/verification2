"""Native solver API calls retain provider reasoning and usage without local estimates."""

import json
import os
import time

from ..agents.verifier import Backend


def completion(job, messages, tools=None):
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
    payload = {
        "model": job.model,
        "temperature": 0,
        "messages": [
            {k: v for k, v in message.items() if k in permitted and v is not None}
            for message in messages
        ],
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
        row["error"] = type(exc).__name__
        raise RuntimeError("Native API request failed: " + type(exc).__name__) from None
    finally:
        row["duration_ms"] = (time.perf_counter() - start) * 1000
        with (job.output / "calls.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
