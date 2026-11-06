import json
from pathlib import Path

from qera.builder import TrajectoryBuilder
from qera.jsonl import write_jsonl
from qera.schema import Label, Role, SolverInfo

FX = Path(__file__).parent / "fixtures"

SENTINEL = "ZZ_GOLD_SENTINEL_42"


def _toy(label=True):
    b = TrajectoryBuilder()
    b.add(Role.Q, "user", "question", "Who founded Lumen?")
    b.add(Role.Q, "user", "user_utterance", "Please check. The screen shows error 404.")
    b.add(
        Role.E,
        "environment",
        "search_result",
        "Lumen was founded by Ada.",
        call={"name": "search", "arguments": {"query": "I bet it is Ada"}},
        hidden={"is_supporting": SENTINEL},
    )
    b.add(Role.R, "solver", "reasoning", "The doc says Ada.")
    b.add(Role.A, "solver", "final_answer", "Ada")
    t = b.build(
        traj_id="toy.1",
        benchmark="toy",
        domain="multihop_qa",
        task_id="1",
        solver=SolverInfo(model="m", native_reasoning="full"),
        label=Label(correct=label, source="exact_match", detail={"secret": SENTINEL}),
        hidden={"gold_answer": SENTINEL},
    )
    return t


CHAT = "https://opencode.ai/inference/go/openai/v1/chat/completions"
SYSTEMONE = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"


def registry(directory, models, key="TEST_API_KEY"):
    """Write a model registry; models maps name -> {api: endpoint}."""
    path = Path(directory) / "models.yaml"
    path.write_text(
        json.dumps(
            {
                "models": {
                    name: {
                        "api_key_env": key,
                        "endpoints": endpoints,
                        "max_output_tokens": 4096,
                    }
                    for name, endpoints in models.items()
                }
            }
        )
    )
    return path


def configure(monkeypatch):
    for role in ("SOLVER", "VERIFIER"):
        monkeypatch.setenv(role + "_MODEL", "test-model")
    monkeypatch.setenv("JEV_MODEL", "typesafe-ai/jev")


def sample():
    builder = TrajectoryBuilder()
    builder.add("Q", "user", "question", "Report the registered color.")
    # Deliberately wrong initial E/R labels: Jev must classify the actual content.
    builder.add(
        "R",
        "environment",
        "tool_result",
        "Registry: blue.",
        call={"name": "lookup", "arguments": {"unit": 1}},
    )
    builder.add("E", "solver", "reasoning", "I suspect red.")
    builder.add("A", "solver", "final_answer", "red")
    return builder.build(
        traj_id="sample",
        benchmark="test",
        domain="tool_use",
        task_id="1",
        solver=SolverInfo(model="original-solver"),
        label=Label(correct=False, source="env_reward"),
        hidden={"gold": "NEVER_SEND_THE_GOLD"},
    )


def decisions(view, only=None):
    answers = {}
    for question_id, target in view["targets"].items():
        if only is not None and question_id not in only:
            continue
        segment = view["segments"][target["seg_id"]]
        choice = (
            "E"
            if target["field"] == "call" or segment["speaker"] == "environment"
            else "R"
        )
        answers[question_id] = {
            "type": "choice",
            "choice": choice,
            "confidence": 1.0,
            "probabilities": {choice: 1.0, "R" if choice == "E" else "E": 0.0},
        }
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def fresh_splits(tmp_path, monkeypatch):
    """A tiny native task tree: one train and one test task per benchmark."""
    from verification2.data import split

    tau = tmp_path / "raw/tau2-bench/data/tau2/domains"
    for domain, partition in (
        ("airline", "train"),
        ("retail", "test"),
        ("telecom", None),
    ):
        directory = tau / domain
        directory.mkdir(parents=True)
        ids = [domain + "0"] if partition else []
        (directory / "tasks.json").write_text(json.dumps([{"id": i} for i in ids]))
        native = {"base": ids, "train": [], "test": []}
        if partition:
            native[partition] = ids
        (directory / "split_tasks.json").write_text(json.dumps(native))
    tasks = tmp_path / "raw/tasks"
    tasks.mkdir(parents=True)
    rows = [
        {"instance_id": str(i), "task_id": i, "eval": {"eval_types": ["url_match"]}}
        for i in range(2)
    ]
    for benchmark in ("swebench_verified", "agentrewardbench"):
        (tasks / (benchmark + ".jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    monkeypatch.setattr(split, "ROOT", tmp_path)
    monkeypatch.setenv("SOLVER_MODEL", "test-model")
    return [split.load_split(b) for b in split.BENCHMARKS]


def solve(splits, output, partition="pilot", **options):
    from verification2.experiments import collection

    return collection.collect(splits, output, partition, "test-model", CHAT, **options)


def fake_native_episode(job, *args):
    job.output.mkdir(parents=True)
    builder = TrajectoryBuilder()
    builder.add("Q", "user", "question", "Complete task " + job.task.task_id)
    builder.add("E", "environment", "observation", "Observed tool result.")
    builder.add(
        "R",
        "solver",
        "reasoning",
        "Recorded API reasoning.",
        source_id="api/reasoning_content",
    )
    builder.add("A", "solver", "final_answer", "Candidate output.")
    return builder.build(
        traj_id=job.task.key,
        benchmark=job.task.benchmark,
        domain="tool_use" if job.task.benchmark == "tau2_bench" else "software_web",
        task_id=f"{job.task.domain}/{job.task.task_id}"
        if job.task.benchmark == "tau2_bench"
        else job.task.task_id,
        solver=SolverInfo(
            model=job.model, origin="self_generated", native_reasoning="full"
        ),
        label=Label(
            correct=True,
            source="unit_tests"
            if job.task.benchmark == "swebench_verified"
            else "env_reward",
        ),
    )


def overrides(*paths):
    rows = (
        json.loads(line) for p in paths for line in Path(p).read_text().splitlines()
    )
    return {(o["traj_id"], o["seg_id"]): o for o in rows}


def synthetic(path):
    """Four labelled toy trajectories; hidden fields must never reach a model."""
    trajectories = []
    for index in range(4):
        b = TrajectoryBuilder()
        b.add("Q", "user", "question", f"What is the registered color of Unit {index}?")
        b.add(
            "E",
            "environment",
            "tool_result",
            f"Registry: Unit {index} is blue.",
            call={"name": "registry_lookup", "arguments": {"unit": index}},
        )
        b.add("R", "solver", "reasoning", "I think the color is red.")
        b.add("E", "environment", "observation", "Weather: sunny.")
        b.add("A", "solver", "final_answer", "blue" if index % 2 == 0 else "red")
        trajectories.append(
            b.build(
                traj_id=f"synthetic-{index}",
                benchmark="synthetic",
                domain="tool_use",
                task_id=str(index),
                solver=SolverInfo(model="synthetic-solver"),
                label=Label(correct=index % 2 == 0, source="exact_match"),
                hidden={"gold_answer": "DO_NOT_SEND_HIDDEN"},
            )
        )
    write_jsonl(trajectories, str(path))
