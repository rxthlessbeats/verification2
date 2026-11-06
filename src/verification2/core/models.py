"""Schemas for model answers and experiment records; provider metadata stays raw."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictStr,
    model_validator,
)

from qera.schema import Override

from .config import Condition, Name, PositiveInt, Settings

SegmentId = Annotated[int, Field(strict=True, ge=0)]
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Verdict = Literal["correct", "incorrect"]
Status = Literal["ok", "error", "prepared", "mock", "unchanged"]


class VerifierAnswer(Settings):
    verdict: Verdict
    reason: StrictStr


class RequestMessage(Settings):
    role: Literal["system", "user"]
    content: StrictStr


class OutputConfig(Settings):
    effort: Name


class VerifierRequest(Settings):
    model: Name
    messages: list[RequestMessage] = Field(min_length=1)
    temperature: Literal[0] = 0
    max_tokens: PositiveInt | None = None
    system: StrictStr | None = None
    reasoning_effort: Name | None = None
    output_config: OutputConfig | None = None


class ProviderResponse(BaseModel):
    # Only consumed fields are parsed; all provider metadata stays in raw_response.
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)


class JevChoice(ProviderResponse):
    type: Literal["choice"]
    choice: Name


class JevResponse(ProviderResponse):
    answers: dict[str, JevChoice]


class ChatMessage(ProviderResponse):
    content: StrictStr | None


class ChatChoice(ProviderResponse):
    message: ChatMessage
    finish_reason: StrictStr | None = None


class ChatResponse(ProviderResponse):
    choices: list[ChatChoice] = Field(min_length=1)


class MessageBlock(ProviderResponse):
    type: StrictStr
    text: StrictStr | None = None

    @model_validator(mode="after")
    def require_text(self):
        if self.type == "text" and self.text is None:
            raise ValueError("A text block must contain text")
        return self


class MessagesResponse(ProviderResponse):
    content: list[MessageBlock]
    stop_reason: StrictStr | None = None


class Result(Settings):
    benchmark: Name
    traj_id: Name
    task_id: Name | None = None
    status: Status = "error"
    usage: JsonValue = None
    error: StrictStr | None = None
    raw_response: JsonValue = None
    request_id: StrictStr | None = None
    cache_hit: StrictBool = False
    cache_key: StrictStr | None = None


class VerifierResult(Result):
    split_id: Name | None = None
    native_task_key: Name | None = None
    partition: Literal["pilot", "train", "test"] | None = None
    trajectory_hash: Name | None = None
    condition: Condition
    label_correct: StrictBool | None
    verdict: Verdict | None = None
    reason: StrictStr | None = None
    included_seg_ids: list[SegmentId] = Field(default_factory=list)
    verification_ms: Milliseconds = 0
    total_ms: Milliseconds = 0
    visible_r: StrictBool = False

    @model_validator(mode="after")
    def require_verdict(self):
        if self.status == "ok" and (self.verdict is None or self.reason is None):
            raise ValueError("A successful verifier result requires verdict and reason")
        return self


class ClassificationResult(Result):
    overrides: list[Override] = Field(default_factory=list)
    classification_ms: Milliseconds = 0


class SelectionResult(Result):
    labels: dict[str, Name] = Field(default_factory=dict)
    selection_ms: Milliseconds = 0


class MetricSummary(Settings):
    benchmark: Name
    condition: Condition
    attempted: SegmentId
    errors: SegmentId
    paired_labelled_n: SegmentId
    incorrect_n: SegmentId
    correct_n: SegmentId
    false_rejections: SegmentId
    false_acceptances: SegmentId
    false_rejection_rate: (
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    )
    false_acceptance_rate: (
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    )
