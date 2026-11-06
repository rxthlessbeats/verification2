from pathlib import Path
from types import SimpleNamespace

import pytest

from qera.render import build_verifier_input
from tests.support import CHAT, _toy, configure
from verification2.benchmarks.web_worker import (
    api_only_stats,
    convert_steps,
    guard_navigation,
    narration,
)
from verification2.core.prompts import load_prompts
from verification2.core.registry import resolve
from verification2.data import split
from verification2.data.split import (
    BENCHMARKS,
    Split,
    Task,
    digest,
    load_split,
    make_split,
    model_free_web_task,
    select,
    split_id,
)
from verification2.experiments import protocol
from verification2.experiments.collection import Job
from verification2.experiments.report import analyze


def pool():
    return [
        Task(
            benchmark=b,
            domain=domain,
            task_id=str(i),
            source=Path("native.json"),
            source_row_hash=digest([b, domain, i]),
            partition=("test" if i < 4 else "train") if b == "tau2_bench" else None,
        )
        for b in BENCHMARKS
        for domain in (
            ("airline", "retail", "telecom") if b == "tau2_bench" else (None,)
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
    assert len(list((tmp_path / "data/splits").glob("*.json"))) == 3
    tau = {t.key: t.partition for t in splits[0].tasks}
    assert tau == {t.key: t.partition for t in pool() if t.benchmark == "tau2_bench"}
    for frozen in splits[1:]:
        assert len(frozen.tasks) == 12
        assert sum(t.partition == "train" for t in frozen.tasks) == 6
        rows = [t for t in pool() if t.benchmark == frozen.benchmark]
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
@pytest.mark.parametrize(
    "evaluation",
    [
        {"eval_types": ["unknown"]},
        {"eval_types": ["string_match"], "reference_answers": {"fuzzy_match": "N/A"}},
        {
            "eval_types": ["program_html"],
            "program_html": [{"url": "func:__import__('os')"}],
        },
        {"eval_types": ["program_html"], "program_html": [{"url": "func:(lambda: 1)"}]},
        {
            "eval_types": ["program_html"],
            "program_html": [{"required_contents": {"fuzzy_match": "x"}}],
        },
    ],
)
def test_native_web_eligibility_rejects_model_based_and_unknown_paths(evaluation):
    assert not model_free_web_task({"eval": evaluation})


@pytest.mark.unit
def test_known_native_web_scorers_and_timing_do_not_invent_token_counts():
    assert model_free_web_task(
        {"eval": {"eval_types": ["url_match"], "reference_url": "http://benchmark/"}}
    )
    assert model_free_web_task(
        {
            "eval": {
                "eval_types": ["string_match"],
                "reference_answers": {"exact_match": "answer"},
            }
        }
    )
    step = SimpleNamespace(
        profiling=SimpleNamespace(env_start=1, env_stop=4, agent_start=5, agent_stop=7)
    )
    api_only_stats(step)
    assert step.stats == {"step_elapsed": 3, "agent_elapsed": 2}


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
def test_web_capture_keeps_the_observed_reasoning_narration_tabs_and_candidate(
    tmp_path,
):
    task = Task(
        benchmark="agentrewardbench",
        task_id="webarena.1",
        source=Path("native.jsonl"),
        source_row_hash=digest(1),
    )
    job = Job(
        task=task,
        seed=42,
        model="solver",
        endpoint="https://example.invalid/chat/completions",
        output=tmp_path,
        cache_dir=tmp_path / "cache",
    )
    content = "I will inspect the result.\n```\nclick('12')\n```"
    left = content.index("click")
    rest = narration(content, [(left, left + len("click('12')"))])
    assert rest == "I will inspect the result."
    obs = {
        "goal_object": [{"type": "text", "text": "Find the item."}],
        "open_pages_urls": ["http://shop/item"],
        "open_pages_titles": ["Item title"],
        "active_page_index": 0,
        "axtree_txt": "Item: blue",
    }
    first = SimpleNamespace(
        step=0,
        obs=obs,
        action=content,
        agent_info={
            "executed_action": "click('12')",
            "narration": rest,
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "content": content,
                            "reasoning_text": "API reasoning.",
                        }
                    }
                ]
            },
        },
    )
    final = SimpleNamespace(
        step=1, obs=obs | {"axtree_txt": "Result page."}, action=None, agent_info={}
    )
    trajectory = convert_steps(job, [first, final], 1, tmp_path / "summary_info.json")
    assert trajectory.label.correct and trajectory.solver.native_reasoning == "full"
    assert [s.text for s in trajectory.by_role("R")] == ["API reasoning.", rest]
    assert [s.text for s in trajectory.by_role("A")] == ["click('12')"]
    assert (
        "Title: Item title" in trajectory.by_role("E")[0].text
        and "(active tab)" in trajectory.by_role("E")[0].text
    )
    guard_navigation("goto('https://example.invalid/')")
    for malformed in ("goto()", "goto(7)"):
        with pytest.raises(ValueError, match="HTTP"):
            guard_navigation(malformed)
    for target in (
        "file:///etc/passwd",
        "data:text/plain,secret",
        "javascript:alert(1)",
    ):
        with pytest.raises(ValueError, match="HTTP"):
            guard_navigation(f"goto({target!r})")


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
