import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qera.render import build_verifier_input
from tests.support import CHAT, _toy, configure
from verification2.core.prompts import load_prompts
from verification2.core.registry import resolve
from verification2.data import split
from verification2.data.split import (
    BENCHMARKS,
    Split,
    Task,
    digest,
    load_split,
    load_task,
    make_split,
    select,
    split_id,
)
from verification2.experiments import protocol
from verification2.experiments.report import analyze


def pool():
    return [
        Task(
            benchmark=b,
            domain=domain,
            task_id=str(i),
            source=Path("native.json"),
            source_row_hash=digest([b, domain, i]),
            partition=(
                ("test" if i < 4 else "train")
                if b in ("tau2_bench", "appworld")
                else None
            ),
        )
        for b in BENCHMARKS
        for domain in (
            ("airline", "retail", "telecom")
            if b == "tau2_bench"
            else ("workspace", "banking")
            if b == "agentdojo"
            else (None,)
        )
        for i in range(12)
    ]


def frozen_splits(tmp_path, monkeypatch, benchmarks=BENCHMARKS):
    monkeypatch.setattr(split, "ROOT", tmp_path)
    monkeypatch.setattr(
        split, "task_pool", lambda b: ([t for t in pool() if t.benchmark == b], [])
    )
    return [load_split(b) for b in benchmarks]


@pytest.mark.unit
def test_heldout_only_requires_selected_benchmark_runtime(tmp_path, monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    lock = tmp_path / "raw/tau2-bench/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("pinned tau2 runtime")
    splits = frozen_splits(tmp_path, monkeypatch, ["tau2_bench"])
    output = tmp_path / "protocol.json"
    protocol.main(["--benchmarks", "tau2_bench", "--output", str(output)])
    solver, prompts = resolve("solver", "test-model"), load_prompts()
    assert protocol.check_heldout(splits, "test", output, solver, prompts)
    trajectory = _toy()
    trajectory.benchmark = "tau2_bench"
    trajectory.split = "test"
    trajectory.meta["split_id"] = split_id(splits)
    args = SimpleNamespace(
        protocol=output,
        model="test-model",
        endpoint=CHAT,
        reasoning_effort=None,
        conditions=["QA", "QEA", "QRA", "QERA"],
        max_tokens=None,
    )
    assert protocol.check_stage([trajectory], args, prompts, "verifier")
    with pytest.raises(ValueError, match="incomplete"):
        protocol.runtime_hashes(required=True)
    lock.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        protocol.check_heldout(splits, "test", output, solver, prompts)
    with pytest.raises(ValueError, match="incomplete"):
        protocol.check_stage([trajectory], args, prompts, "verifier")


@pytest.mark.unit
def test_four_views_are_neutral_ordered_and_keep_the_same_candidate():
    trajectory = _toy()
    trajectory.segments[3].source_id = "private/reasoning_field"
    views = {
        c: build_verifier_input(trajectory, c) for c in ["QA", "QEA", "QRA", "QERA"]
    }
    reasoning, evidence = trajectory.segments[3].text, trajectory.segments[2].text
    assert reasoning in views["QRA"].text and evidence not in views["QRA"].text
    assert evidence in views["QEA"].text and reasoning not in views["QEA"].text
    for view in views.values():
        assert trajectory.segments[-1].text in view.text
        assert view.included_seg_ids == sorted(view.included_seg_ids)
        headers = [
            line for line in view.text.splitlines() if line.startswith("[segment")
        ]
        assert all(
            "reasoning" not in h and "EVIDENCE" not in h and "TASK" not in h
            for h in headers
        )
        assert "private/reasoning_field" not in view.text
    trajectory.segments[3].role_override = "E"
    assert (
        build_verifier_input(trajectory.revalidate(), "QA").text
        == build_verifier_input(trajectory.revalidate(), "QRA").text
    )


@pytest.mark.unit
def test_splits_freeze_native_or_seeded_halves_and_select_balanced_prefixes(
    tmp_path, monkeypatch
):
    splits = frozen_splits(tmp_path, monkeypatch)
    assert frozen_splits(tmp_path, monkeypatch) == splits
    assert len(list((tmp_path / "data/splits").glob("*.json"))) == len(BENCHMARKS)
    tau = {t.key: t.partition for t in splits[0].tasks}
    assert tau == {t.key: t.partition for t in pool() if t.benchmark == "tau2_bench"}
    for frozen in splits[1:]:
        rows = [t for t in pool() if t.benchmark == frozen.benchmark]
        if frozen.benchmark == "appworld":
            assert len(frozen.tasks) == 12
            assert sum(t.partition == "train" for t in frozen.tasks) == 8
            assert make_split(frozen.benchmark, rows[::-1]) == frozen
            continue
        assert len(frozen.tasks) == len(rows)
        assert sum(t.partition == "train" for t in frozen.tasks) == len(rows) // 2
        assert make_split(frozen.benchmark, rows[::-1]) == frozen
    chosen = select(splits[:1], "train", 5)
    domains = [t.domain for t in chosen]
    assert sorted(map(domains.count, set(domains))) == [1, 2, 2]
    assert {t.key for t in select(splits[:1], "train", 4)} < {t.key for t in chosen}
    pilots = select(splits, "pilot")
    assert [t.benchmark for t in pilots] == list(BENCHMARKS)
    assert all(t.partition == "pilot" for t in pilots)
    assert {t.key for t in pilots} <= {t.key for t in select(splits, "train")}
    tests = [t for s in splits for t in s.tasks if t.partition == "test"]
    assert select(splits, "test") == tests
    assert split_id(splits) != split_id(splits[:1])
    changed = [t.model_copy(update={"source_row_hash": "x"}) for t in pool()]
    monkeypatch.setattr(split, "task_pool", lambda b: (changed, []))
    with pytest.raises(ValueError, match="missing or changed"):
        load_split("tau2_bench")
    with pytest.raises(ValueError, match="train and test"):
        Split(benchmark="tau2_bench", method="m", tasks=select(splits[:1], "train"))
    with pytest.raises(ValueError, match="belong"):
        Split(benchmark="swebench_verified", method="m", tasks=splits[0].tasks)


@pytest.mark.unit
def test_agentdojo_load_uses_the_suite_when_task_ids_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "ROOT", tmp_path)
    path = tmp_path / "raw/tasks/agentdojo.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"suite": "banking", "task_id": "user_task_15", "prompt": "banking bill"},
        {"suite": "slack", "task_id": "user_task_15", "prompt": "slack channel"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    task = Task(
        benchmark="agentdojo",
        domain="slack",
        task_id="user_task_15",
        source=Path("raw/tasks/agentdojo.jsonl"),
        source_row_hash=digest(rows[1]),
    )
    assert load_task(task)["prompt"] == "slack channel"


@pytest.mark.unit
def test_report_preserves_paired_tradeoffs_coverage_and_bootstrap():
    rows = []
    for i, correct in enumerate([True, False, True]):
        for condition in ("QA", "QEA", "QRA", "QERA"):
            verdict = (
                "incorrect"
                if condition == "QA"
                else "correct"
                if condition == "QRA" or correct
                else "incorrect"
            )
            rows.append(
                {
                    "benchmark": "b",
                    "traj_id": str(i),
                    "task_id": str(i),
                    "condition": condition,
                    "label_correct": correct,
                    "status": "error" if i == 2 and condition == "QA" else "ok",
                    "verdict": verdict,
                    "visible_r": True,
                }
            )
    conditions = ["QA", "QEA", "QRA", "QERA"]
    summary, comparisons = analyze(rows, conditions, {"b": 4}, iterations=80)
    baseline = next(
        r
        for r in summary
        if r["subgroup"] == "all" and r["benchmark"] == "b" and r["condition"] == "QA"
    )
    assert baseline["paired_labelled_n"] == 2 and baseline["planned_coverage"] == 0.5
    assert (
        baseline["false_rejection_rate"] == 1 and baseline["false_acceptance_rate"] == 0
    )
    tradeoff = {
        r["metric"]: r
        for r in comparisons
        if r["subgroup"] == "all"
        and r["benchmark"] == "b"
        and r["comparison"] == "add_R_without_E"
    }
    assert tradeoff["false_rejection_rate"]["change"] == -1
    assert tradeoff["false_acceptance_rate"]["change"] == 1
    assert analyze(rows, conditions, {"b": 4}, iterations=80) == (summary, comparisons)


@pytest.mark.unit
def test_macro_weights_benchmarks_equally_and_keeps_absent_classes_null():
    rows = []
    for benchmark, count in (("large", 20), ("small", 1)):
        for i in range(count):
            for label in (True, False):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "traj_id": f"{i}-{label}",
                        "condition": "QERA",
                        "status": "ok",
                        "label_correct": label,
                        "verdict": ("correct" if label else "incorrect")
                        if benchmark == "large"
                        else ("incorrect" if label else "correct"),
                    }
                )
    summary, _ = analyze(rows, ["QERA"], iterations=50)
    macro = next(
        r for r in summary if r["subgroup"] == "all" and r["benchmark"] == "macro_equal"
    )
    assert macro["false_rejection_rate"] == macro["false_acceptance_rate"] == 0.5
    summary, _ = analyze(
        [r for r in rows if r["label_correct"]], ["QERA"], iterations=20
    )
    assert all(r["false_acceptance_rate"] is None for r in summary)


@pytest.mark.unit
def test_native_task_ids_cannot_become_paths():
    native = "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"
    assert (
        Task(
            benchmark="tau2_bench",
            domain="telecom",
            task_id=native,
            source=Path("native.json"),
            source_row_hash=digest(1),
        ).task_id
        == native
    )
    with pytest.raises(ValueError):
        Task(
            benchmark="swebench_verified",
            task_id="../../secret",
            source=Path("native.jsonl"),
            source_row_hash=digest(1),
        )


@pytest.mark.unit
def test_protocol_fingerprint_covers_every_subpackage(tmp_path, monkeypatch):
    package = tmp_path / "src/verification2"
    sources = [
        package / name
        for name in (
            "agents/verifier.py",
            "benchmarks/native_sources.json",
            "core/config.py",
            "data/split.py",
            "experiments/protocol.py",
        )
    ]
    for path in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    monkeypatch.setattr(protocol, "__file__", str(package / "experiments/protocol.py"))
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    original = protocol.code_hash()
    for path in sources:
        path.write_text("{}\n\n")
        assert protocol.code_hash() != original
        path.write_text("{}\n")
        assert protocol.code_hash() == original
