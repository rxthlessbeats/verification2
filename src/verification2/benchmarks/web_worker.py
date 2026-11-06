"""Fresh ARB/WebArena episodes using the pinned BrowserGym native demo scaffold."""

import ast
import dataclasses
import importlib.resources
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from qera import reasoning
from qera.builder import TrajectoryBuilder
from qera.schema import Label, SolverInfo

from ..data.split import digest, load_task
from ..experiments.collection import Job
from .native_api import completion
from .setup import ROOT


def api_only_stats(step):
    t = step.profiling
    step.stats = {
        "step_elapsed": t.env_stop - t.env_start,
        "agent_elapsed": t.agent_stop - t.agent_start,
    }


def forbid_model_grading(*args, **kwargs):
    raise RuntimeError("The frozen split forbids model-based native web grading")


def guard_navigation(code):
    for node in ast.walk(ast.parse(code)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "goto"
        ):
            if not node.args:
                raise ValueError("goto requires an HTTP(S) URL")
            target = ast.literal_eval(node.args[0])
            if not isinstance(target, str):
                raise ValueError("goto requires an HTTP(S) URL")
            if target != "about:blank" and urlsplit(target).scheme not in (
                "http",
                "https",
            ):
                raise ValueError("Browser navigation requires an HTTP(S) URL")


def narration(content, spans):
    parts, start = [], 0
    for left, right in spans:
        parts.append(content[start:left])
        start = right
    parts.append(content[start:])
    return re.sub(r"(?m)^\s*```(?:python)?\s*$", "", "".join(parts)).strip()


def make_agent(job):
    sys.path.insert(0, str(ROOT / "raw/BrowserGym"))
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.core.action.parsers import highlevel_action_parser
    from demo_agent.agent import DemoAgent

    class WebActions(HighLevelActionSet):
        def to_python_code(self, action):
            code = super().to_python_code(action)
            guard_navigation(code.removeprefix(self.python_includes))
            return code

    class ApiAgent(DemoAgent):
        def __init__(self):
            # Same pinned native prompt/action scaffold, with the API transport injected.
            self.model_name, self.chat_mode = job.model, False
            self.use_html, self.use_axtree, self.use_screenshot = False, True, False
            self.action_set = WebActions(
                subsets=["webarena"], strict=False, multiaction=False, demo_mode="off"
            )
            self.action_history = []
            self.last_raw = None
            self.openai_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
            )

        def create(self, model, messages, **kwargs):
            self.last_raw = completion(job, messages)
            answer = self.last_raw["choices"][0]["message"].get("content")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
            )

        def get_action(self, obs):
            action, info = super().get_action(obs)
            info["raw_response"] = self.last_raw
            info["native_action"] = action
            private = reasoning.from_message(self.last_raw["choices"][0]["message"])
            if private:
                self.action_history[-1] = private + "\n\n" + (action or "")
            if action:
                try:
                    executed = self.action_set.to_python_code(action)
                    info["executed_action"] = executed.removeprefix(
                        self.action_set.python_includes
                    ).strip()
                    spans = [
                        (start, stop)
                        for _, start, stop in highlevel_action_parser.scan_string(
                            action
                        )
                    ]
                    info["narration"] = narration(action, spans)
                except (ValueError, NameError):
                    info["executed_action"] = (
                        action  # The environment records invalid-action feedback.
                    )
            return action, info

    return ApiAgent()


@dataclasses.dataclass
class AgentArgs:
    job_file: str
    agent_name: str = "verification2-browsergym-native-demo"

    def make_agent(self):
        return make_agent(Job.model_validate_json(Path(self.job_file).read_text()))


def convert_steps(job, steps, score, summary_path):
    builder = TrajectoryBuilder()
    initial = steps[0].obs
    goal = "\n".join(
        part["text"] for part in initial["goal_object"] if part.get("type") == "text"
    )
    builder.add("Q", "user", "goal", goal, source_id="native/goal_object")
    has_action = False
    for step in steps:
        obs = step.obs or {}
        titles = obs.get("open_pages_titles", [])
        active = obs.get("active_page_index")
        observations = [
            f"Tab {i}{' (active tab)' if i == active else ''}\nTitle: {titles[i] if i < len(titles) else ''}\nURL: {url}"
            for i, url in enumerate(obs.get("open_pages_urls", []))
        ]
        if obs.get("axtree_txt"):
            observations.append(obs["axtree_txt"])
        if obs.get("last_action_error"):
            observations.append("Last action error: " + obs["last_action_error"])
        builder.add(
            "E",
            "environment",
            "observation",
            "\n".join(observations),
            step=step.step,
            source_id=f"native/step:{step.step}/observation",
            hidden={
                "screenshot_path": str(
                    summary_path.parent / f"screenshot_step_{step.step}.png"
                )
            },
        )
        raw = (step.agent_info or {}).get("raw_response")
        if raw:
            message = raw["choices"][0]["message"]
            private = reasoning.from_message(message)
            think, _, _ = reasoning.split_think(message.get("content") or "")
            builder.add(
                "R",
                "solver",
                "reasoning",
                "\n\n".join(filter(None, (private, think))),
                step=step.step,
                source_id=f"native/step:{step.step}/api_reasoning",
            )
            _, rest, _ = reasoning.split_think(
                (step.agent_info or {}).get("narration", "")
            )
            builder.add(
                "R",
                "solver",
                "narration",
                rest,
                step=step.step,
                source_id=f"native/step:{step.step}/content",
            )
        action = (step.agent_info or {}).get("executed_action") or step.action
        if action:
            has_action = True
            builder.add(
                "A",
                "solver",
                "action",
                action,
                step=step.step,
                source_id=f"native/step:{step.step}/action",
            )
    if not has_action:
        builder.add(
            "A", "solver", "final_answer", "(no action)", source_id="native/termination"
        )
    return builder.build(
        traj_id=f"arb.{job.task.task_id}.{job.model}.run_{job.output.name}",
        benchmark="agentrewardbench",
        domain="software_web",
        task_id=job.task.task_id,
        solver=SolverInfo(
            model=job.model,
            agent="BrowserGym native text-only DemoAgent via ARB",
            origin="self_generated",
            native_reasoning="full",
        ),
        label=Label(
            correct=score >= 1 - 1e-9,
            source="env_reward",
            score=score,
            detail={"native_summary": str(summary_path), "human_annotations": False},
        ),
        meta={"web_benchmark": "webarena", "text_only": True},
    )


def collect(job):
    import webarena
    from agent_reward_bench.benchmarks import get_webarena_100_benchmark
    from browsergym.experiments import loop
    from webarena.evaluation_harness import evaluators, helper_functions

    expected = load_task(job.task)
    configs = json.loads(
        importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
    )
    (actual,) = [
        r for r in configs if "webarena." + str(r["task_id"]) == job.task.task_id
    ]
    if digest(actual) != digest(expected):
        raise ValueError(
            "Installed native WebArena task differs from the frozen source"
        )
    for module in (evaluators, helper_functions):
        module.llm_fuzzy_match = forbid_model_grading
        module.llm_ua_match = forbid_model_grading
    loop.StepInfo.make_stats = api_only_stats  # Preserve profiling while disabling native local token estimates.
    benchmark = get_webarena_100_benchmark()
    benchmark.prepare_backends()
    native_env = dataclasses.replace(
        benchmark.env_args_list[0], task_name=job.task.task_id, task_seed=job.seed
    )
    experiment = loop.ExpArgs(
        agent_args=AgentArgs(str(job.output / "job.json")),
        env_args=native_env,
        save_screenshot=True,
    )
    experiment.prepare(job.output / "browsergym")
    experiment.run()
    result = loop.ExpResult(experiment.exp_dir)
    summary = result.summary_info
    if summary.get("err_msg"):
        raise RuntimeError(
            "Native browser episode failed; native diagnostics preserved"
        )
    score = summary.get("cum_reward")
    if type(score) not in (int, float):
        raise ValueError("Native browser evaluator did not produce a numeric reward")
    trajectory = convert_steps(
        job, result.steps_info, score, Path(experiment.exp_dir) / "summary_info.json"
    )
    (job.output / "trajectory.json").write_text(trajectory.model_dump_json() + "\n")


if __name__ == "__main__":
    # Pickled AgentArgs must resolve to its importable package module after the worker exits.
    from .web_worker import collect as run

    run(Job.model_validate_json(Path(sys.argv[1]).read_text()))
