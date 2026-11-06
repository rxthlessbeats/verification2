"""Load a complete, validated prompt bundle without executing YAML objects."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import Field, StringConstraints

from .config import Benchmark, Settings

Instruction = Annotated[str, StringConstraints(strict=True, pattern=r"\S")]
DEFAULT_PROMPTS = Path(__file__).with_name("prompts.yaml")


class Prompt(Settings):
    instruction: Instruction


class Criteria(Settings):
    E: Instruction
    R: Instruction


class JevPrompt(Prompt):
    criteria: Criteria


class SelectorCriteria(Settings):
    useful: Instruction
    misleading: Instruction
    irrelevant: Instruction


class SelectorPrompt(Prompt):
    criteria: SelectorCriteria


class BenchmarkPrompts(Settings):
    jev_guidance: Instruction
    selector_guidance: Instruction | None = None
    solver_instruction: Instruction | None = None


class Prompts(Settings):
    version: Annotated[int, Field(strict=True, ge=1, le=1)]
    verifier: Prompt
    jev: JevPrompt
    selector: SelectorPrompt
    benchmarks: dict[Benchmark, BenchmarkPrompts] = Field(default_factory=dict)

    def classification(self, benchmark: str) -> JevPrompt:
        guidance = self.benchmarks.get(benchmark)
        if guidance is None:
            return self.jev
        return JevPrompt(
            instruction=self.jev.instruction + "\n\n" + guidance.jev_guidance,
            criteria=self.jev.criteria,
        )

    def selection(self, benchmark: str) -> SelectorPrompt:
        guidance = self.benchmarks.get(benchmark)
        if guidance is None or guidance.selector_guidance is None:
            return self.selector
        return SelectorPrompt(
            instruction=self.selector.instruction + "\n\n" + guidance.selector_guidance,
            criteria=self.selector.criteria,
        )

    def solver(self, benchmark: str) -> str | None:
        prompt = self.benchmarks.get(benchmark)
        return prompt.solver_instruction if prompt else None


class PromptLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(not isinstance(key, str) for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Prompt YAML requires unique string keys")
        return super().construct_mapping(node, deep=deep)


def load_prompts(path: Path | None = None) -> Prompts:
    try:
        contents = yaml.load(
            (path or DEFAULT_PROMPTS).read_text(encoding="utf-8"), Loader=PromptLoader
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid prompt YAML: {exc}") from exc
    if contents is None:
        raise ValueError("Prompt YAML must not be empty")
    return Prompts.model_validate(contents)
