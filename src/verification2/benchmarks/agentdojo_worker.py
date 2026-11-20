"""Pinned AgentDojo collection without prompt injections; utility is the native label."""

import json
import sys
from pathlib import Path

from openai.types.chat import ChatCompletionMessage

from qera.adapters.agentdojo import VERSION, convert_messages

from ..experiments.collection import Job
from .native_api import completion


def collect(job):
    from agentdojo.agent_pipeline.agent_pipeline import (
        AgentPipeline,
        load_system_message,
    )
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.llms.openai_llm import (
        _function_to_openai,
        _message_to_openai,
        _openai_to_assistant_message,
    )
    from agentdojo.agent_pipeline.tool_execution import (
        ToolsExecutionLoop,
        ToolsExecutor,
    )
    from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.task_suite.task_suite import (
        functions_stack_trace_from_messages,
        model_output_from_messages,
    )
    from agentdojo.types import ChatMessage

    class CachedLLM(BasePipelineElement):
        """OpenAI-compatible transport through the shared cached completion helper."""

        def __init__(self, current):
            self.job = current
            self.model = current.model
            self.name = current.model.replace("/", "_")

        def query(
            self,
            query: str,
            runtime: FunctionsRuntime,
            env: Env | None = None,
            messages: list[ChatMessage] | tuple = (),
            extra_args: dict | None = None,
        ):
            if env is None:
                env = EmptyEnv()
            openai_messages = [
                _message_to_openai(message, self.model) for message in messages
            ]
            tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
            payload_messages = []
            for message in openai_messages:
                payload_messages.append(
                    message
                    if isinstance(message, dict)
                    else message.model_dump(exclude_none=True)
                )
            payload_tools = [
                tool if isinstance(tool, dict) else tool.model_dump(exclude_none=True)
                for tool in tools
            ]
            raw = completion(self.job, payload_messages, payload_tools or None)
            choice = raw["choices"][0]["message"]
            output = _openai_to_assistant_message(
                ChatCompletionMessage.model_validate(choice)
            )
            for field in ("reasoning_content", "reasoning", "reasoning_text"):
                if choice.get(field):
                    output[field] = choice[field]
            return query, runtime, env, [*messages, output], extra_args or {}

    suite_name = job.task.domain
    if not suite_name:
        raise ValueError("AgentDojo tasks require a suite domain")
    suite = get_suite(VERSION, suite_name)
    user_task = suite.get_user_task_by_id(job.task.task_id)
    system = job.solver_instruction or load_system_message(None)
    llm = CachedLLM(job)
    tools_loop = ToolsExecutionLoop([ToolsExecutor(), llm])
    pipeline = AgentPipeline([SystemMessage(system), InitQuery(), llm, tools_loop])
    pipeline.name = llm.name

    environment = suite.load_and_inject_default_environment({})
    task_environment = user_task.init_environment(environment)
    pre_environment = task_environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)
    messages: list[ChatMessage] = []
    for _ in range(3):
        _, _, task_environment, messages, _ = pipeline.query(
            user_task.PROMPT, runtime, task_environment
        )
        if model_output_from_messages(messages) is not None:
            break
    model_output = model_output_from_messages(messages) or []
    utility = suite._check_task_result(
        user_task,
        model_output,
        pre_environment,
        task_environment,
        functions_stack_trace_from_messages(messages),
    )
    serializable = []
    for message in messages:
        row = dict(message)
        if row.get("tool_calls"):
            row["tool_calls"] = [
                {
                    "id": getattr(call, "id", None),
                    "function": getattr(call, "function", None),
                    "args": getattr(call, "args", None),
                }
                for call in row["tool_calls"]
            ]
        if row.get("tool_call") is not None:
            call = row["tool_call"]
            row["tool_call"] = {
                "id": getattr(call, "id", None),
                "function": getattr(call, "function", None),
                "args": getattr(call, "args", None),
            }
        serializable.append(row)
    (job.output / "messages.json").write_text(
        json.dumps(serializable, indent=2, default=str) + "\n"
    )
    trajectory = convert_messages(
        serializable,
        suite=suite_name,
        task_id=job.task.task_id,
        model=job.model,
        utility=bool(utility),
        run_id=job.output.name,
        prompt=user_task.PROMPT,
    )
    (job.output / "trajectory.json").write_text(
        trajectory.revalidate().model_dump_json() + "\n"
    )


if __name__ == "__main__":
    collect(Job.model_validate_json(Path(sys.argv[1]).read_text()))
