"""One tool-calling loop for the text environments."""

import json
import os
import time


class EpisodeLimitReached(RuntimeError):
    """The solver exhausted its tool turns without reaching a graded terminal state."""


def enable_java():
    """Point Java at a full JDK when one is configured, otherwise the runtime's JDK."""
    home = os.environ.get("VERIFICATION2_JAVA_HOME")
    if not home:
        from jdk4py import JAVA_HOME

        home = str(JAVA_HOME)
    os.environ["JAVA_HOME"] = home
    os.environ["PATH"] = home + "/bin:" + os.environ.get("PATH", "")


def tool(name, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }


def _arguments(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"command": raw}
    return parsed if isinstance(parsed, dict) else {"command": raw}


def _choice(message, default, step):
    calls = message.get("tool_calls") or []
    if calls:
        call = calls[0]
        function = call.get("function") or {}
        if isinstance(function, str):
            name, raw = function, call.get("arguments")
        else:
            name = function.get("name") or default
            raw = function.get("arguments")
        arguments = _arguments(raw)
        command = str(arguments.get("command") or "").strip()
        call_id = call.get("id") or f"call-{step}"
    else:
        command = (message.get("content") or "").strip()
        name, arguments, call_id = default, {"command": command}, f"call-{step}"
    return name, command, {"id": call_id, "function": name, "args": arguments}


def run(job, messages, tools, handle, limit):
    """handle(tool_name, command) -> (observation, done). Observations are already public."""
    default = tools[0]["function"]["name"]
    for step in range(limit):
        raw = _complete(job, messages, tools)
        message = raw["choices"][0]["message"]
        name, command, call = _choice(message, default, step)
        assistant = {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": [call],
        }
        for key in ("reasoning_content", "reasoning", "reasoning_text", "thinking"):
            if message.get(key):
                assistant[key] = message[key]
        messages.append(assistant)
        observation, done = handle(name, command)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": observation or "(no observation)",
            }
        )
        if done:
            return messages
    raise EpisodeLimitReached(
        f"No terminal action within {limit} solver turns; episode unscored"
    )


def _complete(job, messages, tools):
    from verification2.benchmarks.native_api import completion

    delay = 20
    for attempt in range(6):
        try:
            return completion(job, messages, tools)
        except RuntimeError as exc:
            if "HTTP 429" not in str(exc) or attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2
