"""AppWorld ReAct collection in the pydantic-v1 native runtime.

Writes episode.json only; the parent process converts it with qera (pydantic v2).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

COMPLETE_RE = re.compile(r"apis\.supervisor\.complete_task\(")
MAX_INTERACTIONS = 50


def request_hash(endpoint, request):
    wire = json.dumps([endpoint, request], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(wire.encode()).hexdigest()


def call_llm(job, messages, calls):
    payload = {
        "model": job["model"],
        "temperature": 0,
        "messages": messages,
    }
    if job.get("reasoning_effort"):
        payload["reasoning_effort"] = job["reasoning_effort"]
    row = {"status": "error", "request": {k: v for k, v in payload.items()}}
    start = time.perf_counter()
    try:
        endpoint = job["endpoint"]
        api_key = os.environ["SOLVER_API_KEY"]
        cache_dir = job.get("cache_dir")
        key = request_hash(endpoint, payload)
        raw = None
        hit = False
        lock_handle = None
        path = None
        if cache_dir:
            root = Path(cache_dir)
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = root / (key + ".json")
            lock_handle = (root / (key + ".lock")).open("a")
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("request_hash") != key:
                    raise ValueError(
                        "Cached response fingerprint does not match its filename"
                    )
                raw, hit = record["raw"], True
        if raw is None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
                "User-Agent": "verification2/0.1",
            }
            if urlsplit(endpoint).hostname == "opencode.ai":
                headers["x-opencode-session"] = "verification2-solver-" + key[:32]
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                method="POST",
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    raw = json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                row["error"] = "HTTP " + str(exc.code)
                raise RuntimeError(
                    "Native API request failed: HTTP " + str(exc.code)
                ) from None
            except Exception as exc:  # noqa: BLE001
                row["error"] = type(exc).__name__
                raise RuntimeError(
                    "Native API request failed: " + type(exc).__name__
                ) from None
            if path is not None:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", dir=path.parent, delete=False, encoding="utf-8"
                    ) as target:
                        temporary = Path(target.name)
                        target.write(
                            json.dumps({"version": 1, "request_hash": key, "raw": raw})
                            + "\n"
                        )
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, path)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
        if lock_handle is not None:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
        row.update(
            status="ok",
            raw_response=raw,
            usage=raw.get("usage"),
            cache_hit=hit,
            cache_key=key,
        )
        return raw
    finally:
        row["duration_ms"] = (time.perf_counter() - start) * 1000
        calls.write(json.dumps(row, ensure_ascii=False) + "\n")
        calls.flush()


def prompt_messages(task, template):
    from jinja2 import Template

    prompt = Template(template).render(
        {"supervisor": task.supervisor, "instruction": task.instruction}
    )
    messages = []
    last_start = 0
    for match in re.finditer(r"(USER|ASSISTANT|SYSTEM):\n", prompt):
        last_end = match.span()[0]
        if not messages:
            if last_end != 0:
                raise ValueError("Start of the prompt has no assigned role")
        else:
            messages[-1]["content"] = prompt[last_start:last_end]
        messages.append({"role": match.group(1).lower(), "content": None})
        last_start = match.span()[1]
    messages[-1]["content"] = prompt[last_start:]
    return messages


def collect(job):
    root = Path(job["appworld_root"])
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld
    from appworld.common.path_store import path_store

    path_store.update_root(str(root))
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)

    template = Path(job["prompt_path"]).read_text(encoding="utf-8")
    if job.get("solver_instruction"):
        template = "SYSTEM:\n" + job["solver_instruction"].strip() + "\n\n" + template

    task_id = job["task"]["task_id"]
    # Isolate native experiment artifacts under this episode directory.
    experiment_name = "run_" + output.name
    steps = []
    with (
        open(output / "calls.jsonl", "a", encoding="utf-8") as calls,
        AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            raise_on_unsafe_syntax=True,
            null_patch_unsafe_execution=True,
        ) as world,
    ):
        history = prompt_messages(world.task, template)
        output_text = None
        for _ in range(MAX_INTERACTIONS):
            if output_text is not None:
                history.append({"role": "user", "content": output_text})
            raw = call_llm(job, history, calls)
            text = ""
            if raw.get("choices"):
                text = raw["choices"][0].get("message", {}).get("content") or ""
            history.append({"role": "assistant", "content": text})
            code = text
            executed = world.execute(code)
            steps.append(
                {
                    "model_text": text,
                    "code": code,
                    "output": executed,
                    "raw_response": raw,
                }
            )
            output_text = executed
            if world.task_completed() or COMPLETE_RE.search(code or ""):
                break
        tracker = world.evaluate(suppress_errors=True)
        success = bool(tracker.success)
        completed = bool(world.task_completed())
        instruction = world.task.instruction
        supervisor = dict(world.task.supervisor)

    episode = {
        "task_id": task_id,
        "dataset": job["task"].get("dataset"),
        "instruction": instruction,
        "supervisor": supervisor,
        "steps": steps,
        "success": success,
        "completed": completed,
        "model": job["model"],
    }
    (output / "episode.json").write_text(
        json.dumps(episode, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    collect(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
