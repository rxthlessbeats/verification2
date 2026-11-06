"""Convert native tau2 conversations, tools and rewards to QERA."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

from .. import reasoning as rsn
from ..builder import TrajectoryBuilder
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

BENCHMARK = "tau2_bench"
DOMAIN = "tool_use"
TERMINAL_GENERIC = {"transfer_to_human_agents"}
GREETING = "Hi! How can I help you today?"
CONTROL_RE = re.compile(r"###(?:STOP|TRANSFER|OUT-OF-SCOPE)###")

FNAME_RE = re.compile(
    r"^(?P<agent>.+?)_(?P<domain>airline|retail|telecom-workflow|telecom|mock|banking_knowledge|intake)_(?P<mode>[a-z\-]+)_(?P<user>.+?)_(?P<n>\d+)trials\.json$"
)

_REASONING_MODELS = ("o1", "o3", "o4", "gpt-5", "deepseek-r", "qwq")


def _load_tool_types() -> dict:
    with resources.files("qera.resources").joinpath("tau2_tool_types.json").open() as f:
        return json.load(f)


def _load_policy(domain: str, tau2_root: str | None) -> str | None:
    if tau2_root:
        d = Path(tau2_root) / "data" / "tau2" / "domains"
        p = d / domain / "policy.md"
        if p.exists():
            return p.read_text()
    try:
        return (
            resources.files("qera.resources")
            .joinpath(f"tau2_policies/{domain}.md")
            .read_text()
        )
    except FileNotFoundError:
        return None


def _tool_type(types: dict, domain: str, side: str, name: str) -> str | None:
    base = domain.split("-")[0]
    return types.get(base, {}).get(side, {}).get(name)


def convert_file(
    path: str,
    tau2_root: str | None = None,
    include_policy: bool = True,
    only_modes: tuple[str, ...] = ("default",),
    tool_types: dict | None = None,
) -> Iterator[Trajectory]:
    path_ = Path(path)
    m = FNAME_RE.match(path_.name)
    file_domain = m.group("domain") if m else None
    mode = m.group("mode") if m else "default"
    if only_modes and mode not in only_modes:
        return
    data = json.loads(path_.read_text())
    info = data.get("info", {})
    agent_llm = (info.get("agent_info") or {}).get("llm") or (
        m.group("agent") if m else "unknown"
    )
    domain = file_domain or (info.get("environment_info") or {}).get(
        "domain_name", "unknown"
    )
    types = tool_types if tool_types is not None else _load_tool_types()
    # Fresh runs must retain the policy actually used, independent of bundled snapshots.
    policy = (
        (
            (info.get("environment_info") or {}).get("policy")
            or _load_policy(domain, tau2_root)
        )
        if include_policy
        else None
    )
    native = (
        "hidden" if any(k in agent_llm.lower() for k in _REASONING_MODELS) else "none"
    )
    origin = "official_release" if "results/final" in str(path_) else "self_generated"

    for sim in data["simulations"]:
        yield _convert_sim(sim, domain, agent_llm, info, types, policy, native, origin)


def _raw_reasoning(mm: dict) -> str:
    """Read reasoning from the message and its raw API response."""
    raw = mm.get("raw_data") or {}
    msgs = [mm, raw.get("message") or {}] + [
        c.get("message") or {} for c in raw.get("choices") or []
    ]
    parts, seen = [], set()
    for m in msgs:
        r = rsn.from_message(m)
        if r and r not in seen:
            seen.add(r)
            parts.append(r)
    return "\n\n".join(parts)


def _convert_sim(
    sim, domain, agent_llm, info, types, policy, native, origin
) -> Trajectory:
    b = TrajectoryBuilder()
    if policy:
        b.add(
            Role.Q,
            "system",
            "policy",
            policy,
            note="Domain policy is part of the task.",
            source_id="policy",
        )

    msgs = sim["messages"]
    last_reply = max(
        (
            i
            for i, mm in enumerate(msgs)
            if mm["role"] == "assistant"
            and not mm.get("tool_calls")
            and (mm.get("content") or "").strip()
        ),
        default=-1,
    )
    pending: dict[str, ToolCall] = {}
    has_reasoning = [False]

    for i, mm in enumerate(msgs):
        role, content = mm["role"], (mm.get("content") or "")
        tcs = mm.get("tool_calls") or []
        if role == "user":
            text = CONTROL_RE.sub("", content).strip()
            if text:
                b.add(
                    Role.Q,
                    "user",
                    "user_utterance",
                    text,
                    step=i,
                    source_id=f"msg:{i}/content",
                    note="User message may mix requirements and observations; use spans to separate them.",
                )
            for tc in tcs:
                pending[tc["id"]] = ToolCall(
                    name=tc["name"],
                    arguments=tc.get("arguments"),
                    requestor="user",
                    tool_type=_tool_type(types, domain, "user", tc["name"]),
                )
                b.record_call(tc["id"])
        elif role == "assistant":
            think, content, unclosed = rsn.split_think(content)
            if unclosed:
                b.mark_incomplete("unclosed_think", f"msg:{i}")
            for r_text, src in (
                (_raw_reasoning(mm), "reasoning_field"),
                (think, "think"),
            ):
                if r_text:
                    b.add(
                        Role.R,
                        "solver",
                        "reasoning",
                        r_text,
                        step=i,
                        source_id=f"msg:{i}/{src}",
                    )
                    has_reasoning[0] = True
            if tcs:
                b.add(
                    Role.R,
                    "solver",
                    "narration",
                    content,
                    step=i,
                    source_id=f"msg:{i}/content",
                )
                for k, tc in enumerate(tcs):
                    tt = _tool_type(types, domain, "assistant", tc["name"])
                    call = ToolCall(
                        name=tc["name"],
                        arguments=tc.get("arguments"),
                        requestor="assistant",
                        tool_type=tt,
                    )
                    action_segment = None
                    if tt == "WRITE" or tc["name"] in TERMINAL_GENERIC:
                        action_segment = b.add(
                            Role.A,
                            "solver",
                            "action",
                            "",
                            step=i,
                            call=call,
                            source_id=f"msg:{i}/tool_call:{k}",
                            note="Write tool changes task state and is part of the candidate output.",
                        )
                    pending[tc["id"]] = call
                    b.record_call(tc["id"], action_segment)
            elif content.strip():
                if i == 0 and content.strip() == GREETING:
                    continue
                if i == last_reply:
                    b.add(
                        Role.A,
                        "solver",
                        "final_answer",
                        content,
                        step=i,
                        source_id=f"msg:{i}/content",
                    )
                else:
                    b.add(
                        Role.A,
                        "solver",
                        "agent_reply",
                        content,
                        step=i,
                        ambiguous=True,
                        source_id=f"msg:{i}/content",
                        note="Intermediate replies count toward native communicate_checks; "
                        "keep them as A unless explicitly split.",
                    )
        elif role == "tool":
            call = pending.pop(mm.get("id", ""), None)
            requestor = mm.get("requestor") or (call.requestor if call else "assistant")
            kind = "user_observation" if requestor == "user" else "tool_result"
            result_segment = b.add(
                Role.E,
                "environment",
                kind,
                content,
                step=i,
                call=call,
                source_id=f"msg:{i}/tool_result",
            )
            b.record_result(mm.get("id", ""), result_segment)

    ri = sim.get("reward_info") or {}
    reward = ri.get("reward")
    label = Label(
        correct=None if reward is None else bool(reward >= 1.0 - 1e-9),
        source="env_reward",
        score=reward,
        detail={
            "db_match": (ri.get("db_check") or {}).get("db_match"),
            "reward_basis": ri.get("reward_basis"),
            "termination_reason": sim.get("termination_reason"),
        },
    )
    model_tag = agent_llm.replace("/", "_")
    traj_id = f"tau2.{domain}.{sim['task_id']}.trial{sim.get('trial', 0)}.{model_tag}"
    return b.build(
        traj_id=traj_id,
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=f"{domain}/{sim['task_id']}",
        solver=SolverInfo(
            model=agent_llm,
            agent="tau2_llm_agent",
            origin=origin,
            native_reasoning="full" if has_reasoning[0] else native,
        ),
        label=label,
        meta={
            "tau2_domain": domain,
            "trial": sim.get("trial"),
            "num_messages": len(msgs),
            "user_llm": (info.get("user_info") or {}).get("llm"),
        },
        hidden={
            "action_checks": ri.get("action_checks"),
            "nl_assertions": ri.get("nl_assertions"),
            "communicate_checks": ri.get("communicate_checks"),
        },
    )
