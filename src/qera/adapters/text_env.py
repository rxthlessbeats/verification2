"""Convert a text-environment tool loop into QERA."""

from __future__ import annotations

import json
from typing import Any

from .. import reasoning as rsn
from ..builder import TrajectoryBuilder
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

DOMAIN = "tool_use"


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _call(tool_call: dict) -> tuple[ToolCall, str | None]:
    name = tool_call.get("function") or tool_call.get("name") or "unknown"
    arguments = tool_call.get("args")
    if arguments is None:
        arguments = tool_call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"command": arguments}
    return ToolCall(
        name=str(name), arguments=arguments or {}, requestor="assistant"
    ), tool_call.get("id")


def convert_messages(
    messages: list[dict],
    *,
    benchmark: str,
    task_domain: str | None,
    task_id: str,
    model: str,
    run_id: str,
    correct: bool,
    score: float | None,
    detail: dict,
    agent: str,
    label_source: str,
) -> Trajectory:
    """System and user text are Q, tool results are E, thoughts are R, calls are A."""
    builder = TrajectoryBuilder()
    pending: dict[str, ToolCall] = {}
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
            builder.add(
                Role.Q,
                "user",
                "instruction",
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
            if rest.strip() and tool_calls:
                builder.add(
                    Role.R,
                    "solver",
                    "narration",
                    rest,
                    step=index,
                    source_id=f"native/msg:{index}/content",
                )
            for call_index, tool_call in enumerate(tool_calls):
                call, call_id = _call(tool_call)
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
        elif role == "tool":
            call_id = message.get("tool_call_id")
            call = pending.pop(str(call_id), None) if call_id else None
            segment = builder.add(
                Role.E,
                "environment",
                "tool_result",
                content,
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
    full_id = f"{task_domain}/{task_id}" if task_domain else task_id
    model_tag = model.replace("/", "_")
    return builder.build(
        traj_id=f"{benchmark}.{full_id}.{model_tag}.run_{run_id}",
        benchmark=benchmark,
        domain=DOMAIN,
        task_id=full_id,
        solver=SolverInfo(
            model=model,
            agent=agent,
            origin="self_generated",
            native_reasoning="full",
        ),
        label=Label(
            correct=correct,
            source=label_source,
            score=score,
            detail=detail,
        ),
    )
