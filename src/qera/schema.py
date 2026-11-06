"""Validated QERA records; labels and hidden fields never enter model input."""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StringConstraints,
    model_validator,
)

SCHEMA_VERSION = "1.1"
Identifier = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
Index = Annotated[int, Field(strict=True, ge=0)]


class DataModel(BaseModel):
    # Archived records contain retired optional fields, which remain readable.
    model_config = ConfigDict(hide_input_in_errors=True)


class Role(str, Enum):
    Q = "Q"
    E = "E"
    R = "R"
    A = "A"


class Speaker(str, Enum):
    user = "user"
    solver = "solver"
    environment = "environment"
    system = "system"


Kind = Literal[
    # Q
    "question",
    "goal",
    "issue",
    "instruction",
    "policy",
    "user_utterance",
    # E
    "passage",
    "tool_result",
    "search_result",
    "execution_output",
    "observation",
    "user_observation",
    # R
    "reasoning",
    "narration",
    "intermediate_output",
    # A
    "final_answer",
    "agent_reply",
    "action",
    "patch",
    "state_diff",
]


class ToolCall(DataModel):
    """Tool name and arguments associated with an observation or action."""

    name: Identifier
    arguments: JsonValue = None
    requestor: str | None = None
    tool_type: str | None = None

    def render(self, include_args: bool = True) -> str:
        """Render the call, omitting arguments when their role is excluded."""
        who = f"{self.requestor} " if self.requestor else ""
        if not include_args:
            return f"{who}call: {self.name}(…)"
        args = self.arguments
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        return f"{who}call: {self.name}({args})"


class Span(DataModel):
    """A role-labelled character interval: text[start:end]."""

    start: Index
    end: Index
    role: Role


class Override(DataModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    traj_id: Identifier
    seg_id: Index
    role: Role | None = None
    call_role: Role | None = None
    spans: list[tuple[Index, Index, Role]] | None = Field(default=None, min_length=1)
    source: Literal["classifier", "manual"] = "manual"
    note: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.role is None and self.call_role is None and self.spans is None:
            raise ValueError("Override requires role, call_role or spans")
        if self.role is not None and self.spans is not None:
            raise ValueError("Override role and spans are mutually exclusive")
        if self.spans and any(end <= start for start, end, _ in self.spans):
            raise ValueError("Override spans require end > start")
        return self


class Segment(DataModel):
    seg_id: Index
    role_default: Role
    role_override: Role | None = None
    role_source: Literal["rule", "classifier", "manual"] = "rule"
    override_sources: dict[str, Literal["rule", "classifier", "manual"]] = Field(
        default_factory=dict
    )
    speaker: Speaker
    kind: Kind
    step: Index | None = None
    source_id: str | None = None
    text: str
    call: ToolCall | None = None
    call_role_override: Role | None = None
    spans: list[Span] | None = None
    ambiguous: StrictBool = False
    note: str | None = None
    hidden: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def role(self) -> Role:
        return self.role_override or self.role_default

    @property
    def call_role(self) -> Role:
        return self.call_role_override or self.role

    def text_roles(self) -> set[Role]:
        """Return all roles present in this segment."""
        return {sp.role for sp in self.spans} if self.spans else {self.role}

    def has_evidence(self) -> bool:
        return any(role == Role.E and bool(text) for role, text in self.pieces()) or (
            self.call is not None and self.call_role == Role.E
        )

    def pieces(self) -> list[tuple[Role, str]]:
        """Return text pieces with their effective roles."""
        if not self.spans:
            return [(self.role, self.text)]
        return [(sp.role, self.text[sp.start : sp.end]) for sp in self.spans]

    @model_validator(mode="after")
    def _check_spans(self) -> Segment:
        if self.spans:
            pos = 0
            for sp in self.spans:
                if sp.start != pos or sp.end <= sp.start:
                    raise ValueError(
                        f"seg {self.seg_id}: spans must cover the text in order without gaps or overlaps"
                    )
                pos = sp.end
            if pos != len(self.text):
                raise ValueError(
                    f"seg {self.seg_id}: spans do not cover the text end ({pos}/{len(self.text)})"
                )
        return self


class Label(DataModel):
    """Native outcome; None means no usable label."""

    correct: StrictBool | None
    source: Literal[
        "exact_match",
        "llm_judge",
        "unit_tests",
        "env_reward",
        "human_annotation",
        "pending",
    ]
    score: float | None = Field(default=None, allow_inf_nan=False)
    detail: dict[str, JsonValue] = Field(default_factory=dict)


class SolverInfo(DataModel):
    model: Identifier
    agent: str | None = None
    origin: Literal["official_release", "self_generated"] = "self_generated"
    native_reasoning: Literal["full", "summary", "hidden", "none"] = "none"


class Trajectory(DataModel):
    schema_version: Literal["1.1"] = SCHEMA_VERSION
    traj_id: Identifier
    benchmark: Identifier
    domain: Literal["multihop_qa", "tool_use", "software_web"]
    task_id: Identifier
    split: str | None = None
    solver: SolverInfo
    label: Label
    segments: list[Segment]
    meta: dict[str, JsonValue] = Field(default_factory=dict)
    hidden: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Trajectory:
        # Old frozen files remain readable; retired count metadata is not propagated.
        self.meta.pop("native_reasoning_tokens", None)
        ids = [s.seg_id for s in self.segments]
        if ids != list(range(len(ids))):
            raise ValueError(
                f"{self.traj_id}: seg_id must be consecutive in source order: 0..n-1"
            )
        roles = (
            set().union(*(s.text_roles() for s in self.segments))
            if self.segments
            else set()
        )
        if Role.Q not in roles:
            raise ValueError(f"{self.traj_id}: missing Q")
        if Role.A not in roles:
            raise ValueError(f"{self.traj_id}: missing A")
        return self

    def revalidate(self) -> Trajectory:
        """Validate nested edits to roles and spans."""
        return Trajectory.model_validate(self.model_dump(warnings=False))

    def by_role(self, role: Role | str) -> list[Segment]:
        """Return segments containing this text role."""
        role = Role(role)
        return [s for s in self.segments if role in s.text_roles()]
