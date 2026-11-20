"""Convert AppWorld ReAct episodes to QERA."""

from __future__ import annotations

import re
from typing import Any

from .. import reasoning as rsn
from ..builder import TrajectoryBuilder, to_text
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

BENCHMARK = "appworld"
DOMAIN = "tool_use"
COMPLETE_RE = re.compile(
    r"apis\.supervisor\.complete_task\((.*?)\)\s*$", re.DOTALL | re.MULTILINE
)
ANSWER_ARG_RE = re.compile(r"answer\s*=\s*(.+)", re.DOTALL)


def protect_python_actions(trajectory: Trajectory) -> Trajectory:
    """Keep executed code visible across views, including episodes stored by the original adapter."""
    for segment in trajectory.segments:
        if (
            segment.kind == "execution_output"
            and segment.call
            and segment.call.name == "python"
        ):
            segment.call_role_override = Role.A
            segment.override_sources["call_role"] = "rule"
    return trajectory


def _answer_from_code(code: str) -> str | None:
    match = COMPLETE_RE.search(code)
    if not match:
        return None
    answer = ANSWER_ARG_RE.search(match.group(1))
    return (
        answer.group(1).strip().rstrip(",") if answer else "(completed without answer)"
    )


def convert_episode(episode: dict[str, Any], model: str, run_id: str) -> Trajectory:
    """Build a trajectory from a self-generated AppWorld episode record."""
    builder = TrajectoryBuilder()
    task_id = str(episode["task_id"])
    supervisor = episode.get("supervisor") or {}
    who = (
        f"{supervisor.get('first_name', '')} {supervisor.get('last_name', '')}".strip()
    )
    instruction = episode.get("instruction") or ""
    question = (
        f"{instruction}\n\n(Supervisor: {who}, {supervisor.get('email', '')})"
        if who
        else instruction
    )
    builder.add(
        Role.Q,
        "user",
        "instruction",
        question,
        source_id="native/specs.instruction",
    )
    answer = None
    for index, step in enumerate(episode.get("steps") or []):
        raw = step.get("raw_response") or {}
        message = ((raw.get("choices") or [{}])[0].get("message")) or {}
        private = rsn.from_message(message)
        text = to_text(step.get("model_text") or message.get("content") or "")
        think, rest, unclosed = rsn.split_think(text)
        if unclosed:
            builder.mark_incomplete("unclosed_think", f"step:{index}")
        for block, source in ((private, "reasoning_field"), (think, "think")):
            if block:
                builder.add(
                    Role.R,
                    "solver",
                    "reasoning",
                    block,
                    step=index,
                    source_id=f"native/step:{index}/{source}",
                )
        code = (step.get("code") or "").strip()
        narration = rest
        if code and code in narration:
            narration = narration.replace(code, "").strip()
        builder.add(
            Role.R,
            "solver",
            "narration",
            narration,
            step=index,
            source_id=f"native/step:{index}/content",
            ambiguous="#" in code,
            note="Code comments may contain solver reasoning" if "#" in code else None,
        )
        if code:
            extracted = _answer_from_code(code)
            if extracted is not None:
                answer = extracted
                builder.add(
                    Role.A,
                    "solver",
                    "final_answer",
                    extracted,
                    step=index,
                    source_id=f"native/step:{index}/complete_task",
                )
            builder.add(
                Role.E,
                "environment",
                "execution_output",
                step.get("output") or "",
                step=index,
                call=ToolCall(
                    name="python",
                    arguments={"code": code},
                    requestor="assistant",
                ),
                source_id=f"native/step:{index}/execution",
                ambiguous="#" in code,
            )
    if answer is None:
        builder.add(
            Role.A,
            "solver",
            "final_answer",
            "(task not completed)",
            source_id="native/termination",
            skip_empty=False,
        )
    success = episode.get("success")
    model_tag = model.replace("/", "_")
    trajectory = builder.build(
        traj_id=f"appworld.{task_id}.{model_tag}.run_{run_id}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=task_id,
        solver=SolverInfo(
            model=model,
            agent="appworld_minimal_react",
            origin="self_generated",
            native_reasoning="full",
        ),
        label=Label(
            correct=None if success is None else bool(success),
            source="unit_tests",
            score=None if success is None else float(bool(success)),
            detail={
                "completed": episode.get("completed"),
                "dataset": episode.get("dataset"),
                "num_steps": len(episode.get("steps") or []),
            },
        ),
        meta={"appworld_dataset": episode.get("dataset")},
    )
    return protect_python_actions(trajectory)
