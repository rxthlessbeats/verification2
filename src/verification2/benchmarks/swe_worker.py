"""Pinned mini-SWE-agent collection followed by the independent SWE-bench harness."""

import json
import subprocess
import sys
import time
from pathlib import Path

from pydantic import StrictBool

from ..core.models import ProviderResponse
from ..data.split import load_task
from ..experiments.collection import Job
from .native_api import completion
from .setup import ROOT


class NativeTaskScore(ProviderResponse):
    resolved: StrictBool
    infra_failure: StrictBool


class NativeRunScore(ProviderResponse):
    submitted_ids: list[str]
    empty_patch_ids: list[str]
    completed_ids: list[str]
    resolved_ids: list[str]
    unresolved_ids: list[str]
    error_ids: list[str]
    infra_failure_ids: list[str]
    ambiguous_failure_ids: list[str]


def native_score(output, task_id, patch):
    # Pinned harness RUN_EVALUATION_LOG_DIR plus the --run_id below.
    directory = output / "logs/evaluation/native"
    summary_path = directory / "results.json"
    summary = NativeRunScore.model_validate_json(summary_path.read_text())
    if set(summary.submitted_ids) != {task_id}:
        raise ValueError("Native scorer report has unexpected submitted tasks")
    if any(
        task_id in rows
        for rows in (
            summary.error_ids,
            summary.infra_failure_ids,
            summary.ambiguous_failure_ids,
        )
    ):
        raise RuntimeError(
            "Native scorer failed or classified an infrastructure/ambiguous outcome; no solver label assigned"
        )
    if patch == "" and task_id in summary.empty_patch_ids:
        return False, {
            "native_summary": str(summary_path),
            "native_outcome": "empty_patch",
        }
    if task_id not in summary.completed_ids or task_id in summary.empty_patch_ids:
        raise ValueError("Native scorer did not complete this task")
    report_path = directory / "verification2" / task_id / "report.json"
    report = json.loads(report_path.read_text())
    result = NativeTaskScore.model_validate(report.get(task_id))
    if result.infra_failure:
        raise RuntimeError(
            "Native scorer reported an infrastructure failure; no solver label assigned"
        )
    if result.resolved != (task_id in summary.resolved_ids) or (
        not result.resolved and task_id not in summary.unresolved_ids
    ):
        raise ValueError(
            "Native task result disagrees with the aggregate grader report"
        )
    return result.resolved, {
        "native_report": str(report_path),
        "native_summary": str(summary_path),
        "native_outcome": "resolved" if result.resolved else "unresolved",
    }


def collect(job):
    import litellm
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir
    from minisweagent.exceptions import FormatError
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    class ApiModel(LitellmModel):
        def query(self, messages, **kwargs):
            raw = completion(job, self._prepare_messages_for_api(messages), [BASH_TOOL])
            response = litellm.ModelResponse(**raw)
            try:
                actions = self._parse_actions(response)
            except FormatError as exc:
                exc.messages[0].setdefault("extra", {})["response"] = raw
                raise
            message = response.choices[0].message.model_dump()
            message["extra"] = {
                "actions": actions,
                "response": raw,
                "timestamp": time.time(),
            }
            return message

    class Agent(DefaultAgent):
        def serialize(self, *extra):
            data = super().serialize(*extra)
            data["info"]["model_stats"].pop("instance_cost", None)
            data["info"]["cost_source"] = (
                "provider responses only; missing cost is unknown"
            )
            return data

    instance = load_task(job.task)
    config = yaml.safe_load(
        (builtin_config_dir / "benchmarks/swebench.yaml").read_text()
    )
    config["agent"]["cost_limit"] = (
        0  # The native scaffold's dollar cap is unrelated to API-reported billing.
    )
    config["agent"]["output_path"] = job.output / (job.task.task_id + ".traj.json")
    # Full test suites fan out one worker per host CPU; uncapped, they exhaust WSL and kill Docker.
    config["environment"]["run_args"] = ["--rm", "--memory=4g", "--memory-swap=4g"]
    if job.solver_instruction:
        config["agent"]["system_template"] = job.solver_instruction
    model = ApiModel(
        model_name=job.model,
        **{k: v for k, v in config.get("model", {}).items() if k != "model_name"},
    )
    environment = get_sb_environment(config, instance)
    agent = Agent(model, environment, **config["agent"])
    try:
        outcome = agent.run(instance["problem_statement"])
        if outcome.get("exit_status") not in (
            "Submitted",
            "LimitsExceeded",
            "RepeatedFormatError",
        ):
            raise ValueError(
                "Solver did not finish a native episode: "
                + str(outcome.get("exit_status"))
            )
        agent.save(config["agent"]["output_path"], {"instance_id": job.task.task_id})
    finally:
        if hasattr(environment, "cleanup"):
            environment.cleanup()
    prediction = job.output / "predictions.jsonl"
    prediction.write_text(
        json.dumps(
            {
                "instance_id": job.task.task_id,
                "model_name_or_path": "verification2",
                "model_patch": outcome.get("submission", ""),
            }
        )
        + "\n"
    )
    # Native test timeout/image behavior stays with the pinned harness.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            str(
                job.task.source
                if job.task.source.is_absolute()
                else ROOT / job.task.source
            ),
            "--split",
            "test",
            "--predictions_path",
            str(prediction),
            "--instance_ids",
            job.task.task_id,
            "--max_workers",
            "1",
            "--run_id",
            "native",
        ],
        cwd=job.output,
        check=True,
    )
    resolved, score_detail = native_score(
        job.output, job.task.task_id, outcome.get("submission", "")
    )
    from qera.adapters.swebench import convert_traj

    trajectory = convert_traj(
        str(config["agent"]["output_path"]), resolved, job.model, "full"
    )
    trajectory.solver.origin = "self_generated"
    trajectory.solver.agent = "mini-swe-agent (pinned native scaffold)"
    trajectory.traj_id += ".run_" + job.output.name
    trajectory.label.detail = score_detail
    (job.output / "trajectory.json").write_text(
        trajectory.revalidate().model_dump_json() + "\n"
    )


if __name__ == "__main__":
    collect(Job.model_validate_json(Path(sys.argv[1]).read_text()))
