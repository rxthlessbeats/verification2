import json
from unittest.mock import Mock

import pytest

from tests.support import fake_native_episode as fresh
from tests.support import fresh_splits, solve
from verification2.core.prompts import load_prompts
from verification2.data.split import BENCHMARKS, SEED, select, split_id
from verification2.experiments import collection, store
from verification2.experiments.protocol import runtime_hashes


@pytest.mark.integration
def test_each_task_is_solved_once_per_model_and_reused_by_later_collections(
    tmp_path, monkeypatch, trajectory_store
):
    splits = fresh_splits(tmp_path, monkeypatch)
    runner = Mock(side_effect=fresh)
    monkeypatch.setattr(collection, "collect_job", runner)
    assert solve(splits, tmp_path / "first", backend="http") == 0
    assert solve(splits, tmp_path / "second", backend="http") == 0
    assert runner.call_count == 3
    assert (tmp_path / "first/trajectories.jsonl").read_bytes() == (
        tmp_path / "second/trajectories.jsonl"
    ).read_bytes()
    entries = sorted(trajectory_store.glob("*/*/*/entry.json"))
    assert {p.relative_to(trajectory_store).parts[:2] for p in entries} == {
        (b, "test-model") for b in BENCHMARKS
    }
    entry = json.loads(entries[0].read_text())
    assert entry["duration_s"] is not None and entry["outcome"]["correct"] is True
    rows = (tmp_path / "second/task_results.jsonl").read_text().splitlines()
    assert {json.loads(r)["reused"] for r in rows} == {"store"}


@pytest.mark.integration
def test_stored_trajectories_are_unlabelled_and_runs_label_them_from_the_split(
    tmp_path, monkeypatch, trajectory_store
):
    splits = fresh_splits(tmp_path, monkeypatch)
    runner = Mock(side_effect=fresh)
    monkeypatch.setattr(collection, "collect_job", runner)
    assert solve(splits, tmp_path / "pilot", backend="http") == 0
    assert {c.args[0].task.partition for c in runner.call_args_list} == {None}
    for directory in {p.parent for p in trajectory_store.glob("*/*/*/entry.json")}:
        stored = json.loads((directory / "trajectory.json").read_text())
        assert stored["split"] is None and "split_id" not in stored["meta"]
        assert not {"frozen.json", "result.json"} & {
            p.name for p in (directory / "episode").iterdir()
        }
    assert solve(splits, tmp_path / "train", "train", backend="http") == 0
    assert runner.call_count == len(BENCHMARKS)
    rows = (tmp_path / "train/trajectories.jsonl").read_text().splitlines()
    assert {json.loads(r)["split"] for r in rows} == {"train"}
    assert {json.loads(r)["meta"]["split_id"] for r in rows} == {split_id(splits)}


@pytest.mark.integration
def test_each_reasoning_effort_is_solved_and_stored_separately(
    tmp_path, monkeypatch, trajectory_store
):
    splits = fresh_splits(tmp_path, monkeypatch)
    runner = Mock(side_effect=fresh)
    monkeypatch.setattr(collection, "collect_job", runner)
    assert solve(splits, tmp_path / "default", backend="http") == 0
    assert solve(splits, tmp_path / "low", backend="http", reasoning_effort="low") == 0
    assert (
        solve(splits, tmp_path / "again", backend="http", reasoning_effort="low") == 0
    )
    efforts = [c.args[0].reasoning_effort for c in runner.call_args_list]
    assert efforts == [None] * len(BENCHMARKS) + ["low"] * len(BENCHMARKS)
    assert len(list(trajectory_store.glob("*/*/*/entry.json"))) == 2 * len(BENCHMARKS)


@pytest.mark.integration
def test_store_identity_tracks_solver_inputs_and_rejects_modified_trajectories(
    tmp_path, monkeypatch, trajectory_store
):
    splits = fresh_splits(tmp_path, monkeypatch)
    monkeypatch.setattr(collection, "collect_job", fresh)
    assert solve(splits, tmp_path / "first", backend="http") == 0
    task = select(splits, "pilot")[0]
    prompt = load_prompts().solver(task.benchmark)
    runtime = runtime_hashes()
    key = store.identity(task, "test-model", SEED, prompt, runtime)
    directory, _, _ = store.load(trajectory_store, key)
    for changed in (
        store.identity(task, "other-model", SEED, prompt, runtime),
        store.identity(task, "test-model", SEED + 1, prompt, runtime),
        store.identity(task, "test-model", SEED, "other prompt", runtime),
    ):
        assert store.load(trajectory_store, changed) is None
    frozen = directory / "trajectory.json"
    frozen.write_text(frozen.read_text().replace("Candidate output.", "Edited."))
    with pytest.raises(ValueError, match="does not match"):
        solve(splits, tmp_path / "second", backend="http")


@pytest.mark.integration
def test_rerun_solves_chosen_benchmarks_again_and_replaces_only_on_success(
    tmp_path, monkeypatch, trajectory_store
):
    splits = fresh_splits(tmp_path, monkeypatch)
    monkeypatch.setattr(collection, "collect_job", fresh)
    assert solve(splits, tmp_path / "first", backend="http") == 0
    entries = {p.parts[-4]: p for p in trajectory_store.glob("*/*/*/entry.json")}
    before = {b: p.read_text() for b, p in entries.items()}

    def resolved(job, *args):
        trajectory = fresh(job, *args)
        trajectory.meta["attempt"] = 2
        return trajectory

    runner = Mock(side_effect=resolved)
    monkeypatch.setattr(collection, "collect_job", runner)
    rerun = {"swebench_verified"}
    with pytest.raises(ValueError, match="new output directory"):
        solve(splits, tmp_path / "first", backend="http", rerun=rerun)
    assert solve(splits, tmp_path / "second", backend="http", rerun=rerun) == 0
    assert [c.args[0].task.benchmark for c in runner.call_args_list] == list(rerun)
    assert {b for b, p in entries.items() if p.read_text() != before[b]} == rerun
    assert not list(trajectory_store.glob("*/*/*.replaced"))
    rows = [
        json.loads(r)
        for r in (tmp_path / "second/task_results.jsonl").read_text().splitlines()
    ]
    assert sorted(map(str, (r["reused"] for r in rows))) == ["False", "store", "store"]
    assert next(r for r in rows if not r["reused"])["stored"].endswith("(replaced)")
    replaced = entries["swebench_verified"].read_text()

    monkeypatch.setattr(
        collection, "collect_job", Mock(side_effect=RuntimeError("docker"))
    )
    assert solve(splits, tmp_path / "third", backend="http", rerun=rerun) == 1
    assert entries["swebench_verified"].read_text() == replaced
