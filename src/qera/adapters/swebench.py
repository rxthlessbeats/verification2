"""Convert mini-SWE-agent histories; keep the issue, tool outputs, reasoning and patch."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import reasoning
from ..builder import TrajectoryBuilder
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

BENCHMARK = "swebench_verified"
DOMAIN = "software_web"
PR_RE = re.compile(
    r"<pr_description>\s*(?:Consider the following PR description:\s*)?(.*?)\s*</pr_description>",
    re.DOTALL,
)
SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
INLINE_CMD = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
_OUT_RE = re.compile(
    r"<returncode>(-?\d+)</returncode>\s*(?:<output>\n?(.*?)\n?</output>)?", re.DOTALL
)


def _text(c) -> str:
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _command(tc: dict) -> str:
    fn = tc.get("function", {})
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args
    return (args or {}).get("command", json.dumps(args, ensure_ascii=False))


def convert_traj(
    path: str, resolved: bool | None, model: str, native: str
) -> Trajectory:
    d = json.loads(Path(path).read_text())
    iid = d.get("instance_id") or Path(path).name.split(".traj")[0]
    msgs = d["messages"]
    b = TrajectoryBuilder()

    first_user = next((m for m in msgs if m["role"] == "user"), None)
    m_pr = PR_RE.search(_text(first_user["content"])) if first_user else None
    b.add(
        Role.Q,
        "user",
        "issue",
        m_pr.group(1) if m_pr else _text(first_user["content"]) if first_user else "",
        note=None if m_pr else "No pr_description; using the full prompt as Q.",
        ambiguous=m_pr is None,
        source_id="msg:first_user/pr_description",
    )

    pending: dict[str, tuple[str, bool]] = {}
    submission = None
    for i, m in enumerate(msgs):
        role = m["role"]
        if role == "assistant":
            think, content, unclosed = reasoning.split_think(_text(m.get("content")))
            if unclosed:
                b.mark_incomplete("unclosed_think", f"msg:{i}")
            b.add(
                Role.R,
                "solver",
                "reasoning",
                reasoning.from_message(m),
                step=i,
                source_id=f"msg:{i}/reasoning_field",
            )
            b.add(
                Role.R, "solver", "reasoning", think, step=i, source_id=f"msg:{i}/think"
            )
            tcs = m.get("tool_calls") or []
            if not tcs:
                for cmd in INLINE_CMD.findall(content):
                    pending[f"inline-{i}"] = (cmd.strip(), SUBMIT_MARK in cmd)
                    b.record_call(f"inline-{i}")
                content = INLINE_CMD.sub("", content).strip()
            b.add(
                Role.R,
                "solver",
                "narration",
                content,
                step=i,
                source_id=f"msg:{i}/content",
            )
            for tc in tcs:
                cmd = _command(tc)
                key = tc.get("id", f"{i}")
                pending[key] = (cmd, SUBMIT_MARK in cmd)
                b.record_call(key)
        elif role in ("tool", "user") and m is not first_user:
            failed = (m.get("extra") or {}).get("response") or {}
            if isinstance(failed, dict) and failed.get("choices"):
                original = failed["choices"][0].get("message") or {}
                think, narration, _ = reasoning.split_think(
                    _text(original.get("content"))
                )
                b.add(
                    Role.R,
                    "solver",
                    "reasoning",
                    "\n\n".join(
                        filter(None, (reasoning.from_message(original), think))
                    ),
                    step=i,
                    source_id=f"msg:{i}/failed_response/reasoning",
                )
                b.add(
                    Role.R,
                    "solver",
                    "narration",
                    narration,
                    step=i,
                    source_id=f"msg:{i}/failed_response/content",
                )
                for index, tc in enumerate(original.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    b.add(
                        Role.R,
                        "solver",
                        "intermediate_output",
                        "",
                        step=i,
                        call=ToolCall(
                            name=fn.get("name") or "unknown",
                            arguments=fn.get("arguments"),
                            requestor="assistant",
                        ),
                        source_id=f"msg:{i}/failed_response/call:{index}",
                    )
            key = m.get("tool_call_id") or next(
                (k for k in pending if k.startswith("inline-")), None
            )
            if key is None or key not in pending:
                b.add(
                    Role.E,
                    "environment",
                    "observation",
                    _text(m.get("content")),
                    step=i,
                    source_id=f"msg:{i}/protocol_feedback",
                )
                continue
            cmd, is_submit = pending.pop(key)
            if is_submit:
                # Submission output is already A; repeating it as E changes the comparison.
                b.record_result(key)
                continue
            out = _text(m.get("content"))
            mo = _OUT_RE.search(out)
            rc = int(mo.group(1)) if mo else None
            body = (mo.group(2) or "") if mo else out
            result_segment = b.add(
                Role.E,
                "environment",
                "execution_output",
                body if body.strip() else "(no output)",
                step=i,
                call=ToolCall(
                    name="bash", arguments={"command": cmd}, requestor="assistant"
                ),
                hidden={"returncode": rc},
                source_id=f"msg:{i}/tool_result",
            )
            b.record_result(key, result_segment)
        elif role == "exit":
            submission = (m.get("extra") or {}).get("submission") or _text(
                m.get("content")
            )

    if submission is None:
        submission = (d.get("info") or {}).get("submission", "")
    b.add(
        Role.A,
        "solver",
        "patch",
        submission or "(empty patch)",
        skip_empty=False,
        source_id="exit/submission",
    )

    return b.build(
        traj_id=f"swebench.{iid}.{model}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=iid,
        solver=SolverInfo(
            model=model,
            agent="mini-swe-agent",
            origin="official_release",
            native_reasoning=native,
        ),
        label=Label(
            correct=resolved, source="unit_tests" if resolved is not None else "pending"
        ),
        meta={
            "repo": iid.rsplit("-", 1)[0].replace("__", "/"),
            "api_calls": ((d.get("info") or {}).get("model_stats") or {}).get(
                "api_calls"
            ),
            "exit_status": (d.get("info") or {}).get("exit_status"),
            "trajectory_format": d.get("trajectory_format"),
        },
    )
