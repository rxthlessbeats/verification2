"""ScienceWorld episodes. The label is task completion, not the 0–100 score."""

import sys
from pathlib import Path

from qera.adapters.text_env import convert_messages
from verification2.benchmarks.text_loop import enable_java, run, tool

SYSTEM = (
    "You control a science-lab simulation. Each turn, call act with one command. "
    "Look around, check your inventory, move, pick things up, and use the objects you see. "
    "The task ends when the environment says it is complete."
)


def _flag(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _public(text):
    # The simulator score is a label, so it never enters the transcript.
    lines = [line for line in str(text).splitlines() if "score" not in line.lower()]
    return "\n".join(lines).strip()


def collect(job):
    enable_java()
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv()
    try:
        variation = int(job.task.task_id)
        env.load(
            job.task.domain,
            variation,
            simplificationStr="easy",
            generateGoldPath=False,
        )
        observation, info = env.reset()
        question = info["taskDesc"]
        room = _public(observation)
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": question + "\n\n" + room,
            },
        ]

        def handle(_name, command):
            if not command:
                return "No command was given.", False
            step = env.step(command)
            done = bool(step[2]) if len(step) > 2 else False
            return _public(step[0]), done

        run(
            job,
            messages,
            [tool("act", "One ScienceWorld command.")],
            handle,
            env.envStepLimit + 1,
        )
        completed = _flag(env.server.getCompleted())
    finally:
        closer = getattr(env, "close", None)
        if closer:
            closer()
    trajectory = convert_messages(
        [
            messages[0],
            {"role": "user", "content": question},
            {"role": "tool", "content": room},
            *messages[2:],
        ],
        benchmark="scienceworld",
        task_domain=job.task.domain,
        task_id=job.task.task_id,
        model=job.model,
        run_id=job.output.name,
        correct=completed,
        score=1.0 if completed else 0.0,
        detail={"completed": completed, "simplification": "easy"},
        agent="scienceworld_text",
        label_source="env_reward",
    )
    (job.output / "trajectory.json").write_text(
        trajectory.revalidate().model_dump_json() + "\n"
    )


if __name__ == "__main__":
    from verification2.experiments.collection import Job

    collect(Job.model_validate_json(Path(sys.argv[1]).read_text()))
