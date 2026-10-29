"""每個 adapter 的端到端測試，加上「hidden 欄位不會洩漏給 verifier」的檢查。

執行：python -m pytest -q tests
"""

import json
from pathlib import Path
from types import SimpleNamespace as NS

from qera.adapters import (
    agentrewardbench,
    appworld,
    browsecomp_plus,
    musique,
    swebench,
    tau2,
)
from qera.cli import check_trajectory
from qera.render import build_verifier_input
from qera.schema import Role, Trajectory

FX = Path(__file__).parent / "fixtures"


def _roundtrip(t: Trajectory) -> Trajectory:
    t2 = Trajectory.model_validate_json(t.model_dump_json())
    assert t2 == t
    return t2


def _no_leak(t: Trajectory) -> None:
    """gold 答案只能存在 hidden；Q / E / R 都不能出現（A 本來就可能等於答案）。"""
    gold = t.hidden.get("gold_answer")
    full = build_verifier_input(t, "QERA").text
    assert "is_supporting" not in full and "evidence_hits" not in full
    if gold:
        for s in t.segments:
            if s.role == Role.Q:
                assert gold.lower() not in s.text.lower()


def _conditions_ok(t: Trajectory) -> None:
    full = build_verifier_input(t, "QERA")
    qa = build_verifier_input(t, "QA")
    qea = build_verifier_input(t, "QEA")
    e_ids = [s.seg_id for s in t.by_role(Role.E)]
    sel = build_verifier_input(t, "QEJevA", selected_e=e_ids[:1])
    assert qa.n_tokens <= qea.n_tokens <= full.n_tokens
    assert sel.n_tokens <= qea.n_tokens
    assert qea.tokens_by_role["R"] == 0 and qa.tokens_by_role["E"] == 0
    # 原始順序保留
    assert full.included_seg_ids == sorted(full.included_seg_ids)


def _check(ts):
    assert ts
    for t in ts:
        t = _roundtrip(t)
        assert check_trajectory(t) == []  # 無錯誤也無警告
        _no_leak(t)
        _conditions_ok(t)
    return ts


def test_musique():
    ts = _check(
        list(
            musique.run(
                NS(
                    data=str(FX / "musique_dev.jsonl"),
                    runs=str(FX / "musique_runs.jsonl"),
                    split="dev",
                    native_reasoning="full",
                    export_prompts=None,
                )
            )
        )
    )
    a, b = ts
    assert a.label.correct is True and b.label.correct is False
    assert a.meta["hop_count"] == 2 and b.meta["hop_count"] == 3
    assert a.by_role("A")[0].text == "Ada Brightwater"
    assert "Leeds?" in b.by_role("R")[0].text  # <think> 進 R
    assert [s.hidden["is_supporting"] for s in a.by_role("E")] == [True, True, False]


def test_musique_export_prompts(tmp_path):
    out = tmp_path / "p.jsonl"
    list(
        musique.run(
            NS(
                data=str(FX / "musique_dev.jsonl"),
                runs=None,
                split="dev",
                native_reasoning="full",
                export_prompts=str(out),
            )
        )
    )
    p = json.loads(out.read_text().splitlines()[0])
    assert (
        "Paragraph 1: Title: Zephyr kettle" in p["prompt"] and "Answer:" in p["prompt"]
    )
    assert "is_supporting" not in p["prompt"]


def test_browsecomp_plus():
    (t,) = _check(
        list(
            browsecomp_plus.run(
                NS(
                    queries=str(FX / "bcp_queries.jsonl"),
                    runs=str(FX / "bcp_run.jsonl"),
                    judgments=str(FX / "bcp_judgments.jsonl"),
                    qrels_evidence=str(FX / "bcp_qrel_evidence.txt"),
                    qrels_gold=None,
                    native_reasoning="summary",
                )
            )
        )
    )
    assert t.label.correct is True and t.label.source == "llm_judge"
    (e,) = t.by_role("E")
    assert e.call.name == "search" and e.hidden["evidence_hits"] == ["111"]
    assert "Exact Answer: Leeds" in t.by_role("A")[0].text


def test_browsecomp_plus_pending_label():
    (t,) = list(
        browsecomp_plus.run(
            NS(
                queries=str(FX / "bcp_queries.jsonl"),
                runs=str(FX / "bcp_run.jsonl"),
                judgments=None,
                qrels_evidence=None,
                qrels_gold=None,
                native_reasoning="summary",
            )
        )
    )
    assert t.label.correct is None and t.label.source == "pending"


def test_tau2_real_sample():
    ts = _check(
        list(
            tau2.run(
                NS(
                    input=[
                        str(
                            FX
                            / "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"
                        )
                    ],
                    tau2_root=None,
                    modes="default",
                    no_policy=False,
                )
            )
        )
    )
    t = ts[0]
    assert t.segments[0].kind == "policy" and "<main_policy>" in t.segments[0].text
    kinds = {s.kind for s in t.segments}
    assert "user_observation" in kinds  # telecom 使用者端裝置操作 → E
    for s in t.segments:
        if s.call and s.role == Role.A:
            assert (
                s.call.tool_type == "WRITE" or s.call.name == "transfer_to_human_agents"
            )
        if s.role == Role.Q:
            assert "###STOP###" not in s.text


def test_swebench_real_sample():
    (t,) = _check(
        list(
            swebench.run(
                NS(
                    trajs=str(FX / "swe_trajs"),
                    labels=str(FX / "swe_labels.json"),
                    model="gemini-3-5-flash",
                    native_reasoning="hidden",
                    ids=None,
                )
            )
        )
    )
    assert t.label.correct is True
    assert t.segments[0].kind == "issue" and "<instructions>" not in t.segments[0].text
    (a,) = t.by_role("A")
    assert a.kind == "patch" and a.text.startswith("diff --git")
    assert all(
        "COMPLETE_TASK_AND_SUBMIT" not in json.dumps(s.call.arguments)
        for s in t.by_role("E")
    )


def test_agentrewardbench():
    (t,) = _check(
        list(
            agentrewardbench.run(
                NS(
                    cleaned=str(FX / "arb_cleaned"),
                    annotations=None,
                    splits=None,
                    web_benchmarks="webarena",
                    split=None,
                )
            )
        )
    )
    assert (
        t.task_id == "webarena.377"
        and t.label.correct is False
        and t.label.source == "human_annotation"
    )
    acts = t.by_role("A")
    assert [s.kind for s in acts] == ["action", "final_answer"]
    assert t.by_role("R")[0].text.startswith("I should search")
    assert "searchbox" in t.by_role("E")[0].text


def test_appworld_code_agent():
    (t,) = _check(
        list(
            appworld.run(
                NS(
                    experiment=str(FX / "appworld_exp" / "fixture_agent"),
                    dataset="dev",
                    data_root=str(FX / "appworld_data"),
                    model="fixture-model",
                    native_reasoning="none",
                )
            )
        )
    )
    assert t.label.correct is True
    assert (
        t.segments[0].text.startswith("Pay back Alex")
        and "Supervisor: Sam Lee" in t.segments[0].text
    )
    es = t.by_role("E")
    assert len(es) == 2 and es[0].call.name == "python" and es[0].ambiguous  # 有註解
    assert {s.kind for s in t.by_role("A")} == {"final_answer", "state_diff"}
    assert any("First I'll look up Alex" in s.text for s in t.by_role("R"))


def test_overrides_and_greeting(tmp_path):
    from qera.overrides import apply, load_overrides

    ts = list(
        tau2.run(
            NS(
                input=[
                    str(
                        FX
                        / "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"
                    )
                ],
                tau2_root=None,
                modes="default",
                no_policy=False,
            )
        )
    )
    t = ts[0]
    assert all(
        s.text != tau2.GREETING for s in t.segments
    )  # 框架開場白不算 solver 產出
    target = next(s for s in t.segments if s.kind == "agent_reply")
    ov = tmp_path / "ov.jsonl"
    ov.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": target.seg_id,
                "role": "R",
                "source": "classifier",
            }
        )
        + "\n"
        + json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": target.seg_id,
                "role": "E",
                "source": "manual",
            }
        )
        + "\n"
    )
    (t2,) = list(apply([t], load_overrides([str(ov)])))
    s = t2.segments[target.seg_id]
    assert s.role == Role.E and s.role_default == Role.A and s.role_source == "manual"
    # 改標後 render 立即反映
    assert target.seg_id not in build_verifier_input(t2, "QA").included_seg_ids


def test_tau2_reasoning_from_raw_data(tmp_path):
    """自己用 Qwen thinking 跑 τ² 時，推理在 raw_data 裡，要被抽成 R。"""
    src = FX / "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"
    d = json.loads(src.read_text())
    sim = d["simulations"][0]
    m = next(
        mm
        for mm in sim["messages"]
        if mm["role"] == "assistant" and mm.get("tool_calls")
    )
    m["raw_data"] = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "User reports no service; check line status first."
                }
            }
        ]
    }
    d["simulations"] = [sim]
    dst = tmp_path / "qwen3-thinking_telecom_default_gpt-4.1-2025-04-14_1trials.json"
    dst.write_text(json.dumps(d))
    (t,) = list(
        tau2.run(NS(input=[str(dst)], tau2_root=None, modes="default", no_policy=False))
    )
    rs = [s for s in t.by_role("R") if s.kind == "reasoning"]
    assert rs and "check line status" in rs[0].text
    assert t.solver.native_reasoning == "full" and t.solver.origin == "self_generated"


def test_jsonl_unicode_line_separator(tmp_path):
    """模型輸出含 U+2028 時，JSONL 不能被切成兩行（BrowseComp-Plus o3 run 檔的真實情況）。"""
    run = json.loads((FX / "bcp_run.jsonl").read_text())
    run["result"][-1]["output"] = (
        "Explanation: line one line two end\x85x\nExact Answer: Leeds"
    )
    p = tmp_path / "o3.jsonl"
    p.write_text(json.dumps(run, ensure_ascii=False) + "\n", encoding="utf-8")
    assert " " in p.read_text(encoding="utf-8")
    (t,) = list(
        browsecomp_plus.run(
            NS(
                queries=str(FX / "bcp_queries.jsonl"),
                runs=str(p),
                judgments=None,
                qrels_evidence=None,
                qrels_gold=None,
                native_reasoning="summary",
            )
        )
    )
    assert "line two" in t.by_role("A")[0].text
    assert t.solver.model == "gpt-5"  # 有 metadata 時用 metadata


def test_bcp_model_from_path(tmp_path):
    run = json.loads((FX / "bcp_run.jsonl").read_text())
    run.pop("metadata")
    d = tmp_path / "bm25"
    d.mkdir()
    (d / "o3.jsonl").write_text(json.dumps(run) + "\n")
    (t,) = list(
        browsecomp_plus.run(
            NS(
                queries=str(FX / "bcp_queries.jsonl"),
                runs=str(d / "o3.jsonl"),
                judgments=None,
                qrels_evidence=None,
                qrels_gold=None,
                native_reasoning="summary",
            )
        )
    )
    assert t.solver.model == "o3@bm25"
