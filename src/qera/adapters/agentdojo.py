"""Convert AgentDojo chat messages and utility labels to QERA."""

from __future__ import annotations

import json
from typing import Any

from .. import reasoning as rsn
from ..builder import TrajectoryBuilder
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

BENCHMARK = "agentdojo"
DOMAIN = "tool_use"
VERSION = "v1.2.2"


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("content") or block.get("text") or ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return json.dumps(content, ensure_ascii=False)


def _tool(tool_call: dict) -> tuple[ToolCall, str | None]:
    # ponytail: worker always writes {function, args, id}. OpenAI nests and live objects if a second caller appears.
    name = tool_call.get("function") or tool_call.get("name") or "unknown"
    return ToolCall(
        name=str(name),
        arguments=tool_call.get("args") or tool_call.get("arguments"),
        requestor="assistant",
    ), tool_call.get("id")


def convert_messages(
    messages: list[dict],
    *,
    suite: str,
    task_id: str,
    model: str,
    utility: bool,
    run_id: str,
    prompt: str | None = None,
) -> Trajectory:
    """Map AgentDojo messages: system/user → Q, tool results → E, thoughts → R, calls/final → A."""
    builder = TrajectoryBuilder()
    if prompt:
        builder.add(
            Role.Q,
            "user",
            "instruction",
            prompt,
            source_id="native/user_task_prompt",
        )
    pending: dict[str, ToolCall] = {}
    last_assistant = max(
        (
            i
            for i, message in enumerate(messages)
            if message.get("role") == "assistant"
            and not message.get("tool_calls")
            and _text(message.get("content")).strip()
        ),
        default=-1,
    )
    has_action = False
    for index, message in enumerate(messages):
        role = message.get("role")
        content = _text(message.get("content"))
        if role == "system":
            builder.add(
                Role.Q,
                "system",
                "instruction",
                content,
                step=index,
                source_id=f"native/msg:{index}/system",
            )
        elif role == "user":
            if prompt and content.strip() == prompt.strip():
                continue
            builder.add(
                Role.Q,
                "user",
                "user_utterance",
                content,
                step=index,
                source_id=f"native/msg:{index}/user",
            )
        elif role == "assistant":
            private = rsn.from_message(message)
            think, rest, unclosed = rsn.split_think(content)
            if unclosed:
                builder.mark_incomplete("unclosed_think", f"msg:{index}")
            for text, source in ((private, "reasoning_field"), (think, "think")):
                if text:
                    builder.add(
                        Role.R,
                        "solver",
                        "reasoning",
                        text,
                        step=index,
                        source_id=f"native/msg:{index}/{source}",
                    )
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                builder.add(
                    Role.R,
                    "solver",
                    "narration",
                    rest,
                    step=index,
                    source_id=f"native/msg:{index}/content",
                )
                for call_index, tool_call in enumerate(tool_calls):
                    call, call_id = _tool(tool_call)
                    segment = builder.add(
                        Role.A,
                        "solver",
                        "action",
                        "",
                        step=index,
                        call=call,
                        source_id=f"native/msg:{index}/tool_call:{call_index}",
                    )
                    has_action = True
                    if call_id:
                        pending[str(call_id)] = call
                        builder.record_call(str(call_id), segment)
            elif rest.strip():
                kind = "final_answer" if index == last_assistant else "agent_reply"
                builder.add(
                    Role.A,
                    "solver",
                    kind,
                    rest,
                    step=index,
                    source_id=f"native/msg:{index}/content",
                    ambiguous=kind == "agent_reply",
                )
                has_action = True
        elif role == "tool":
            call_id = message.get("tool_call_id")
            call = pending.pop(str(call_id), None) if call_id else None
            if call is None and message.get("tool_call"):
                call, _ = _tool(message["tool_call"])
            error = message.get("error")
            body = error or content
            segment = builder.add(
                Role.E,
                "environment",
                "tool_result",
                body,
                step=index,
                call=call,
                source_id=f"native/msg:{index}/tool_result",
            )
            if call_id:
                builder.record_result(str(call_id), segment)
    if not has_action:
        builder.add(
            Role.A,
            "solver",
            "final_answer",
            "(no action)",
            source_id="native/termination",
            skip_empty=False,
        )
    model_tag = model.replace("/", "_")
    return builder.build(
        traj_id=f"agentdojo.{suite}.{task_id}.{model_tag}.run_{run_id}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=f"{suite}/{task_id}",
        solver=SolverInfo(
            model=model,
            agent="agentdojo_tool_pipeline",
            origin="self_generated",
            native_reasoning="full",
        ),
        label=Label(
            correct=utility,
            source="env_reward",
            score=float(utility),
            detail={
                "utility": utility,
                "security": True,  # ponytail: no injections in this benchmark
                "attack": "none",
                "benchmark_version": VERSION,
            },
        ),
        meta={"agentdojo_suite": suite, "agentdojo_version": VERSION},
    )
