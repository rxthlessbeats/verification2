"""Model registry: endpoints, credentials and limits for every model a role can use."""

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import Field, model_validator

from .config import Endpoint, Name, PositiveInt, Settings

DEFAULT_REGISTRY = Path(__file__).with_name("models.yaml")
Api = Literal["chat_completions", "messages", "systemone"]
Role = Literal["solver", "jev", "verifier"]
# Native runners speak only Chat Completions and Jev only System One.
APIS = {
    "solver": ("chat_completions",),
    "jev": ("systemone",),
    "verifier": ("messages", "chat_completions"),
}
PATHS = {
    "chat_completions": "/chat/completions",
    "messages": "/messages",
    "systemone": "/systemone",
}


class ModelSpec(Settings):
    api_key_env: Name
    endpoints: dict[Api, Endpoint] = Field(min_length=1)
    max_output_tokens: PositiveInt | None = None
    reasoning_efforts: list[Name] = Field(default_factory=list)
    default_reasoning_effort: Name | None = None

    @model_validator(mode="after")
    def check_endpoints(self):
        for api, endpoint in self.endpoints.items():
            if not urlsplit(endpoint).path.rstrip("/").endswith(PATHS[api]):
                raise ValueError(f"The {api} endpoint must end with {PATHS[api]}")
        default = self.default_reasoning_effort
        if default is not None and default not in self.reasoning_efforts:
            raise ValueError(f"default_reasoning_effort {default} is not listed")
        return self


class Registry(Settings):
    models: dict[Name, ModelSpec] = Field(min_length=1)


class Resolved(Settings):
    role: Role
    model: Name
    api: Api
    endpoint: Endpoint
    spec: ModelSpec
    reasoning_effort: Name | None = None


def load_registry(path=None):
    path = Path(path or os.environ.get("MODELS_FILE") or DEFAULT_REGISTRY)
    try:
        contents = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid model registry {path}: {exc}") from exc
    return Registry.model_validate(contents)


def spec(model):
    found = load_registry().models.get(model)
    if found is None:
        raise ValueError(f"{model} is not in the model registry; add it to models.yaml")
    return found


def resolve(role, model):
    if not model:
        raise ValueError(f"Set {role.upper()}_MODEL in .env or pass --model")
    found = spec(model)
    api = next((a for a in APIS[role] if a in found.endpoints), None)
    if api is None:
        raise ValueError(
            f"{model} lists no {' or '.join(APIS[role])} endpoint, which the {role} needs"
        )
    return Resolved(
        role=role,
        model=model,
        api=api,
        endpoint=found.endpoints[api],
        spec=found,
        reasoning_effort=reasoning_effort(role, model, found),
    )


def reasoning_effort(role, model, found):
    """Empty uses the model's default; None (or "provider") means requests omit the field."""
    if role == "jev":
        return None
    variable = role.upper() + "_REASONING_EFFORT"
    effort = os.environ.get(variable) or found.default_reasoning_effort
    if effort == "provider":
        return None
    if effort and effort not in found.reasoning_efforts:
        allowed = ", ".join(found.reasoning_efforts) or "none in models.yaml"
        raise ValueError(f"{variable}={effort} is not supported by {model} ({allowed})")
    return effort


def max_output_tokens(model):
    maximum = spec(model).max_output_tokens
    if maximum is None:
        raise ValueError(
            f"models.yaml lists no max_output_tokens for {model}; Messages requires one"
        )
    return maximum
