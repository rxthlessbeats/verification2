"""Validated settings for CLI stages and native workers."""

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

Name = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Timeout = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Condition = Literal["QERA", "QA", "QEA", "QRA"]
Mode = Literal["dry-run", "mock", "http"]
Domain = Literal["airline", "retail", "telecom"]
Benchmark = Literal["tau2_bench", "swebench_verified", "agentrewardbench"]


def http_endpoint(value: str) -> str:
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError("Endpoint must be an absolute HTTP(S) URL")
    if url.username or url.password or url.fragment:
        raise ValueError("Endpoint must not contain credentials or a fragment")
    _ = url.port  # Validate malformed/out-of-range ports too.
    return value


Endpoint = Annotated[Name, AfterValidator(http_endpoint)]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ProviderConfig(Settings):
    backend: Mode = "dry-run"
    model: Name | None = None
    endpoint: Endpoint | None = None
    reasoning_effort: Name | None = None
    timeout: Timeout | None = None
    cache_dir: Path | None = None

    @model_validator(mode="after")
    def require_http_settings(self):
        if self.backend == "http" and (not self.endpoint or not self.model):
            raise ValueError("http mode requires a --model listed in models.yaml")
        return self


class VerifierConfig(ProviderConfig):
    command: Literal["run", "verify"] = "verify"
    input: list[Path] = Field(min_length=1)
    output: Path
    prompt_file: Path | None = None
    max_tokens: PositiveInt | None = None
    conditions: list[Condition] = Field(default_factory=lambda: ["QERA"], min_length=1)
    protocol: Path | None = None
    jobs: PositiveInt = 1

    @model_validator(mode="after")
    def check_conditions(self):
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("Duplicate conditions")
        return self


class ClassificationConfig(ProviderConfig):
    command: Literal["classify"] = "classify"
    backend: Literal["dry-run", "http"] = "dry-run"
    input: list[Path] = Field(min_length=1)
    output: Path
    prompt_file: Path | None = None
    protocol: Path | None = None

    @model_validator(mode="after")
    def require_jev(self):
        path = urlsplit(self.endpoint or "").path.rstrip("/")
        if not path.endswith("/systemone") or not (
            self.model == "typesafe-ai/jev" or (self.model or "").startswith("jev-")
        ):
            raise ValueError(
                "Jev classification requires typesafe-ai/jev or a jev-* model and /systemone endpoint"
            )
        return self


class SolverConfig(ProviderConfig):
    command: Literal["solve"] = "solve"
    backend: Literal["dry-run", "http"] = "dry-run"
    model: Name
    endpoint: Endpoint
    output: Path
    domain: Domain
    task_ids: list[Name] = Field(min_length=1)
    tau2_root: Path = Path("raw/tau2-bench")
    tau2_python: Path | None = None
    user_model: Name | None = None
    judge_model: Name | None = None
    num_trials: PositiveInt = 1
    max_steps: PositiveInt | None = None
    max_tokens: PositiveInt | None = None
    seed: Annotated[int, Field(strict=True)] = 42
    prompt_file: Path | None = None

    @model_validator(mode="after")
    def check_native_settings(self):
        if not urlsplit(self.endpoint).path.endswith("/chat/completions"):
            raise ValueError(
                "Solver requires a model and complete /chat/completions endpoint"
            )
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Task IDs must be unique")
        return self


class NativeRunConfig(Settings):
    # The solver manifest also contains provenance and completion fields.
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    tau2_version: Literal["1.0.1"]
    output: Path
    run_id: Name
    model: Name
    user_model: Name
    judge_model: Name
    endpoint: Endpoint
    domain: Domain
    task_ids: list[Name] = Field(min_length=1)
    num_trials: PositiveInt
    max_steps: PositiveInt | None = None
    max_tokens: PositiveInt | None = None
    timeout: Timeout | None = None
    seed: Annotated[int, Field(strict=True)]
    cache_dir: Path | None = None
    solver_instruction: str | None = None
    reasoning_effort: Name | None = None

    @model_validator(mode="after")
    def check_native_settings(self):
        if not urlsplit(self.endpoint).path.endswith("/chat/completions"):
            raise ValueError("Native runner requires a /chat/completions endpoint")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Task IDs must be unique")
        return self
