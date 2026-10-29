"""研究契約的反例測試。

test_adapters.py 驗證的是「轉換結果的格式」；這裡驗證的是「實驗不會因為資料處理而失效」：
- 標準答案不會從任何管道洩漏給 verifier 或選擇器（Jev）
- 四種條件只差在「放了哪些角色」，切分、呼叫角色的改判都會正確反映
- 截斷、沒有推理、標籤錯配、標註者沒有共識等情況會被標出來，而不是混進一般樣本
- 失敗不會被吞掉、不會留下半個檔案
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from qera import cli, reasoning
from qera.adapters import agentrewardbench, browsecomp_plus, musique, swebench, tau2
from qera.builder import TrajectoryBuilder
from qera.overrides import apply, load_overrides
from qera.render import CONDITIONS, build_verifier_input, e_candidates, selector_view
from qera.schema import Label, Role, SolverInfo

FX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
SENTINEL = "ZZ_GOLD_SENTINEL_42"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy(label=True, with_spans=False):
    """小型 trajectory：Q、使用者混合發言、搜尋（呼叫 + 結果）、推理、答案；hidden 與 label 都放了 sentinel。"""
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


# ════════════════════════════════════════════════════════════════════
# 洩漏
# ════════════════════════════════════════════════════════════════════


def test_no_hidden_or_label_reaches_verifier_or_selector():
    t = _toy()
    for cond in CONDITIONS:
        sel = (
            [c["seg_id"] for c in e_candidates(t)]
            if cond in ("QEsA", "QEJevA")
            else None
        )
        assert SENTINEL not in build_verifier_input(t, cond, selected_e=sel).text, cond
    view = json.dumps(selector_view(t))
    assert SENTINEL not in view
    assert "hidden" not in view and "label" not in view
    assert all(
        set(c) == {"seg_id", "kind", "step", "call", "text", "n_tokens"}
        for c in e_candidates(t)
    )


# ════════════════════════════════════════════════════════════════════
# 條件與切分
# ════════════════════════════════════════════════════════════════════


def test_selection_is_validated():
    t = _toy()
    with pytest.raises(ValueError, match="不存在"):
        build_verifier_input(t, "QEJevA", selected_e=[999])
    with pytest.raises(ValueError, match="不是 E"):
        build_verifier_input(t, "QEJevA", selected_e=[0])  # seg 0 是 Q
    e_id = e_candidates(t)[0]["seg_id"]
    cost = e_candidates(t)[0]["n_tokens"]
    with pytest.raises(ValueError, match="超過預算"):
        build_verifier_input(t, "QEJevA", selected_e=[e_id], budget=cost - 1)
    assert build_verifier_input(
        t, "QEJevA", selected_e=[e_id], budget=cost
    ).included_seg_ids


def test_spans_split_mixed_utterance(tmp_path):
    t = _toy()
    e_before = t.tokens_by_role()["E"]
    text = t.segments[1].text
    cut = text.index("The screen")
    ov = tmp_path / "ov.jsonl"
    ov.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": 1,
                "spans": [[0, cut, "Q"], [cut, len(text), "E"]],
                "source": "manual",
            }
        )
        + "\n"
    )
    (t2,) = list(apply([t], load_overrides([str(ov)])))
    qa = build_verifier_input(t2, "QA").text
    qea = build_verifier_input(t2, "QEA").text
    assert "Please check." in qa and "error 404" not in qa  # 觀測那半句不屬於 QA
    assert "error 404" in qea
    assert any(c["seg_id"] == 1 for c in e_candidates(t2))  # 觀測那半句可以被 Jev 選
    assert t2.tokens_by_role()["E"] > e_before  # 切出來的 E 有算進 token


def test_call_role_hides_solver_written_arguments(tmp_path):
    t = _toy()
    ov = tmp_path / "ov.jsonl"
    ov.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": 2,
                "call_role": "R",
                "source": "classifier",
            }
        )
        + "\n"
    )
    (t2,) = list(apply([t], load_overrides([str(ov)])))
    assert "I bet it is Ada" in build_verifier_input(t2, "QERA").text
    qea = build_verifier_input(t2, "QEA").text
    assert "I bet it is Ada" not in qea and "call: search(…)" in qea  # 只剩工具名稱
    assert "Lumen was founded by Ada." in qea  # 回傳結果仍保留


def test_override_errors_are_not_silent(tmp_path):
    t = _toy()
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "traj_id": t.traj_id,
                "seg_id": 1,
                "spans": [[0, 5, "Q"]],
                "source": "manual",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="覆蓋"):
        list(apply([t], load_overrides([str(bad)])))
    # 改到沒有 A → 必須報錯
    no_a = tmp_path / "no_a.jsonl"
    no_a.write_text(
        json.dumps({"traj_id": t.traj_id, "seg_id": 4, "role": "R", "source": "manual"})
        + "\n"
    )
    with pytest.raises(ValueError, match="缺少 A"):
        list(apply([_toy()], load_overrides([str(no_a)])))
    # 對不到任何 segment → cli 失敗結束
    data = tmp_path / "d.jsonl"
    cli.write_jsonl([_toy()], str(data))
    miss = tmp_path / "miss.jsonl"
    miss.write_text(
        json.dumps({"traj_id": "nope", "seg_id": 0, "role": "E", "source": "manual"})
        + "\n"
    )
    with pytest.raises(SystemExit) as e:
        cli.main(
            [
                "apply-overrides",
                str(data),
                str(miss),
                "--out",
                str(tmp_path / "o.jsonl"),
            ]
        )
    assert e.value.code == 1


# ════════════════════════════════════════════════════════════════════
# R 的來源與生成狀態
# ════════════════════════════════════════════════════════════════════


def test_unclosed_think_is_reasoning_not_answer():
    think, rest, unclosed = reasoning.split_think(
        "<think>unfinished internal reasoning"
    )
    assert think == "unfinished internal reasoning" and rest == "" and unclosed
    ex = json.loads((FX / "musique_dev.jsonl").read_text().splitlines()[0])
    run = {
        "id": ex["id"],
        "model": "q",
        "meta": {"finish_reason": "length"},
        "messages": [
            {
                "role": "assistant",
                "content": "<think>the founder might be Ada Brightwater",
            }
        ],
    }
    t = musique.convert_one(ex, run, "dev", "full")
    assert t.meta["incomplete"] and t.by_role("A")[0].text == "(no answer)"
    assert "Ada Brightwater" in t.by_role("R")[0].text


def test_declared_reasoning_is_checked_against_content():
    b = TrajectoryBuilder()
    b.add(Role.Q, "user", "goal", "g")
    b.add(Role.E, "environment", "observation", "o")
    b.add(Role.A, "solver", "final_answer", "a")
    t = b.build(
        traj_id="x",
        benchmark="b",
        domain="software_web",
        task_id="1",
        solver=SolverInfo(model="m", native_reasoning="full"),
        label=Label(correct=True, source="unit_tests"),
    )
    assert (
        t.solver.native_reasoning == "none"
        and t.meta["native_reasoning_declared"] == "full"
    )


def test_arb_action_text_is_not_reasoning(tmp_path):
    """真實資料的情況：模型沒寫 <think>，AgentLab 把 <action> 放進 reasoning 欄位。"""
    d = tmp_path / "cleaned/webarena/GenericAgent-gpt-4o-2024-11-20/exp"
    d.mkdir(parents=True)
    traj = {
        "goal": "find price",
        "steps": [
            {
                "num": 0,
                "reasoning": "<action>\nclick('1')\n</action>",
                "action": "click('1')",
                "url": "u",
                "axtree_pruned": "x",
                "chat_messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "<action>\nclick('1')\n</action>"},
                ],
            },
            {
                "num": 1,
                "reasoning": "I see the price.",
                "action": "send_msg_to_user('$9')",
                "url": "u",
                "axtree_pruned": "y",
                "chat_messages": [
                    {
                        "role": "assistant",
                        "content": "<think>I see the price.</think><action>send_msg_to_user('$9')</action>",
                    }
                ],
            },
        ],
    }
    (d / "webarena.377.json").write_text(json.dumps(traj))
    (t,) = list(
        agentrewardbench.run(
            NS(
                cleaned=str(tmp_path / "cleaned"),
                annotations=None,
                splits=None,
                web_benchmarks="webarena",
                split=None,
            )
        )
    )
    rs = t.by_role("R")
    assert [s.text for s in rs] == ["I see the price."]  # <action> 不算推理
    assert all("<action>" not in s.text for s in rs)


def test_arb_split_and_majority(tmp_path):
    assert (
        agentrewardbench.normalize_task_id("visualwebarena.resized.398")
        == "visualwebarena.398"
    )
    ann = tmp_path / "a.csv"
    rows = [
        (
            "annotator_name,benchmark,task_id,model_name,exp_name,trajectory_success,trajectory_side_effect,"
            "trajectory_optimality,trajectory_looping"
        )
    ]
    for who, ok in [("A", "Successful"), ("B", "Unsuccessful")]:  # 1:1 沒有過半
        rows.append(f"{who},webarena,webarena.1,M,e,{ok},No,1,No")
    for who, ok in [
        ("A", "Successful"),
        ("B", "Successful"),
        ("C", "Unsuccessful"),
    ]:  # 2:1 過半
        rows.append(f"{who},webarena,webarena.2,M,e,{ok},No,1,No")
    for who, ok in [
        ("A", "Successful"),
        ("B", "Unsure"),
        ("C", "Unsuccessful"),
        ("D", "Unsure"),
    ]:  # 沒有過半
        rows.append(f"{who},webarena,webarena.3,M,e,{ok},No,1,No")
    ann.write_text("\n".join(rows) + "\n")
    lab = agentrewardbench.load_labels(str(ann))
    assert lab[("M", "webarena.1")]["correct"] is None
    assert lab[("M", "webarena.2")]["correct"] is True
    assert lab[("M", "webarena.3")]["correct"] is None


def test_swe_labels_require_real_booleans(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(
        json.dumps(
            {
                "a": {"resolved": True},
                "b": {"resolved": "false"},
                "c": {"resolved": None},
                "d": {},
            }
        )
    )
    lab = swebench.load_labels(str(p))
    assert lab == {"a": True, "b": None, "c": None, "d": None}


def test_tau2_inline_think_becomes_reasoning(tmp_path):
    d = json.loads(
        (
            FX / "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json"
        ).read_text()
    )
    sim = d["simulations"][0]
    m = next(
        mm
        for mm in sim["messages"]
        if mm["role"] == "assistant" and mm.get("tool_calls")
    )
    m["content"] = "<think>check line first</think>Let me check."
    d["simulations"] = [sim]
    p = tmp_path / "qwen_telecom_default_x_1trials.json"
    p.write_text(json.dumps(d))
    (t,) = list(
        tau2.run(NS(input=[str(p)], tau2_root=None, modes="default", no_policy=False))
    )
    assert any(
        s.kind == "reasoning" and s.text == "check line first" for s in t.by_role("R")
    )
    assert all("<think>" not in s.text for s in t.segments)


def test_musique_preserves_prompt_order():
    ex = json.loads((FX / "musique_dev.jsonl").read_text().splitlines()[0])
    t = musique.convert_one(ex, {"id": ex["id"], "answer": "x"}, "dev", "full")
    kinds = [s.kind for s in t.segments]
    assert kinds.index("question") > max(
        i for i, k in enumerate(kinds) if k == "passage"
    )  # 段落在前、問題在後


# ════════════════════════════════════════════════════════════════════
# judge 標籤
# ════════════════════════════════════════════════════════════════════


def test_judge_verdict_parsing():
    j = _load_script("judge_bcp")
    assert j.parse_verdict("incorrect: yes\ncorrect: no") == ("ok", False)
    assert j.parse_verdict("reasoning: it says correct: yes\ncorrect: no") == (
        "ok",
        False,
    )
    assert j.parse_verdict("**correct:** yes") == ("ok", True)
    assert j.parse_verdict("correct: yes\ncorrect: no")[0] == "parse_error"  # 互相矛盾
    assert j.parse_verdict("no verdict here")[0] == "parse_error"
    assert j.parse_verdict("<think>correct: yes</think>\ncorrect: no") == ("ok", False)


def test_judgment_must_match_answer(tmp_path):
    run = json.loads((FX / "bcp_run.jsonl").read_text())
    good = {
        "run_tag": "gpt-5",
        "query_id": "9001",
        "status": "ok",
        "correct": True,
        "candidate_hash": browsecomp_plus.candidate_hash(
            browsecomp_plus.final_output_text(run)
        ),
    }
    stale = dict(good, candidate_hash="0000000000000000")  # 判決對應的是另一個答案
    other_run = dict(good, run_tag="o3@bm25")  # 判決屬於另一條 run
    for j, expect in [(good, True), (stale, None), (other_run, None)]:
        p = tmp_path / "j.jsonl"
        p.write_text(json.dumps(j) + "\n")
        (t,) = list(
            browsecomp_plus.run(
                NS(
                    queries=str(FX / "bcp_queries.jsonl"),
                    runs=str(FX / "bcp_run.jsonl"),
                    judgments=str(p),
                    qrels_evidence=None,
                    qrels_gold=None,
                    native_reasoning="summary",
                )
            )
        )
        assert t.label.correct is expect, j


# ════════════════════════════════════════════════════════════════════
# 抽樣與批次流程
# ════════════════════════════════════════════════════════════════════


def _many(tmp_path):
    ts = []
    for task in range(30):
        for trial, ok in enumerate([True, False, True]):  # 同題多個 trial，標籤不同
            t = _toy(label=ok)
            t.traj_id, t.task_id = f"toy.{task}.{trial}", str(task)
            ts.append(t)
    p = tmp_path / "full.jsonl"
    cli.write_jsonl(ts, str(p))
    return p


def test_sample_dedups_before_balancing_and_writes_manifest(tmp_path):
    p = _many(tmp_path)
    out = tmp_path / "bal.jsonl"
    cli.main(["sample", str(p), "--per-class", "100", "--out", str(out)])
    got = list(cli.read_jsonl([str(out)]))
    assert len({t.task_id for t in got}) == len(got)  # 每題最多一條
    c = Counter(t.label.correct for t in got)
    assert c[True] == c[False]
    m = json.loads((tmp_path / "bal.manifest.json").read_text())
    assert (
        m["excluded"]["duplicate_trial"] == 60 and m["output"]["per_class"] == c[True]
    )


def test_sample_without_labels_exits_3_and_removes_stale(tmp_path):
    t = _toy(label=None)
    t.label = Label(correct=None, source="pending")
    p = tmp_path / "full.jsonl"
    cli.write_jsonl([t], str(p))
    out = tmp_path / "bal.jsonl"
    out.write_text("stale\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["sample", str(p), "--out", str(out)])
    assert e.value.code == 3 and not out.exists()


def test_write_is_atomic(tmp_path):
    out = tmp_path / "x.jsonl"
    cli.write_jsonl([_toy()], str(out))
    before = out.read_text()

    def broken():
        yield _toy()
        raise RuntimeError("adapter crashed")

    with pytest.raises(RuntimeError):
        cli.write_jsonl(broken(), str(out))
    assert out.read_text() == before  # 原檔不變
    assert not (tmp_path / "x.jsonl.tmp").exists()  # 不留暫存檔


# ════════════════════════════════════════════════════════════════════
# solver 續跑
# ════════════════════════════════════════════════════════════════════


def test_solver_resume_identity_and_corrupt_tail(tmp_path, monkeypatch):
    rs = _load_script("run_solver")
    from tests.test_run_solver import FakeClient, _run

    client = FakeClient()
    out, _ = _run(tmp_path, monkeypatch, client)
    # 模擬中斷：最後一行寫到一半
    lines = out.read_text().splitlines()
    out.write_text(lines[0] + "\n" + lines[1][:30])
    c2 = FakeClient()
    _run(tmp_path, monkeypatch, c2)
    assert c2.calls == 1  # 只補跑壞掉的那題
    assert all(
        json.loads(l) for l in out.read_text().splitlines()
    )  # 檔案恢復成全部合法
    # 換了生成參數卻用同一個輸出檔 → 停止
    with pytest.raises(SystemExit, match="不同設定"):
        _run(tmp_path, monkeypatch, FakeClient(), extra=("--temperature", "0.9"))
    del rs


def test_a_old_schema_rejected_by_read_and_validate(tmp_path, capsys):
    row = _toy().model_dump()
    row["schema_version"] = "1.0"
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="請重跑 scripts/convert_all.sh"):
        list(cli.read_jsonl([str(p)]))
    with pytest.raises(SystemExit) as e:
        cli.main(["validate", str(p)])
    assert e.value.code == 1
    assert "請重跑 scripts/convert_all.sh" in capsys.readouterr().out


def test_b_role_and_spans_replace_each_other():
    t = _toy()
    key = (t.traj_id, 1)
    text = t.segments[1].text
    (t,) = apply([t], {key: {"spans": [[0, len(text), "E"]], "source": "classifier"}})
    (t,) = apply([t], {key: {"role": "R", "source": "manual"}})
    assert t.segments[1].spans is None
    assert text not in build_verifier_input(t, "QA").text
    (t,) = apply([t], {key: {"spans": [[0, len(text), "Q"]], "source": "manual"}})
    assert t.segments[1].role_override is None
    assert text in build_verifier_input(t, "QA").text


def test_b_overrides_merge_per_field_and_priority(tmp_path):
    key = ("toy.1", 2)
    text = _toy().segments[2].text
    records = [
        {"call_role": "R", "source": "classifier"},
        {"role": "R", "source": "classifier"},
        {"spans": [[0, len(text), "E"]], "source": "manual"},
        {"role": "R", "source": "classifier"},
    ]
    paths = []
    for i, row in enumerate(records):
        p = tmp_path / f"{i}.jsonl"
        p.write_text(json.dumps(dict(traj_id=key[0], seg_id=key[1], **row)) + "\n")
        paths.append(str(p))
    (t,) = apply([_toy()], load_overrides(paths))
    assert t.segments[2].call_role == Role.R
    assert t.segments[2].text_roles() == {Role.E}
    # 人工正文判斷不能阻擋之後對獨立呼叫欄位的分類。
    t = t.revalidate()
    (t,) = apply([t], {key: {"call_role": "A", "source": "classifier"}})
    assert t.segments[2].call_role == Role.A
    # 同優先權後讀檔案勝出；較低優先權仍不能改掉人工正文。
    p = tmp_path / "last.jsonl"
    p.write_text(
        json.dumps(
            {"traj_id": key[0], "seg_id": key[1], "role": "R", "source": "manual"}
        )
        + "\n"
    )
    (t,) = apply([_toy()], load_overrides(paths + [str(p)]))
    assert t.segments[2].text_roles() == {Role.R} and t.segments[2].spans is None


def test_b_reasoning_metadata_recomputed_after_override():
    t = _toy()
    t.solver.native_reasoning = "summary"
    t.meta["native_reasoning_declared"] = "summary"
    key = (t.traj_id, 3)
    original_tokens = t.segments[3].n_tokens
    (t,) = apply([t], {key: {"role": "E", "source": "manual"}})
    assert t.meta["native_reasoning_tokens"] == 0
    assert t.solver.native_reasoning == "none"
    assert t.meta["native_reasoning_declared"] == "summary"
    text = t.segments[3].text
    (t,) = apply([t], {key: {"spans": [[0, len(text), "R"]], "source": "manual"}})
    assert t.meta["native_reasoning_tokens"] == original_tokens
    assert t.solver.native_reasoning == "summary"
    assert t.meta["native_reasoning_declared"] == "summary"


def test_c_call_only_e_consistent_across_interfaces(tmp_path, capsys):
    ts = []
    for i, ok in enumerate((True, False)):
        t = _toy(ok)
        t.traj_id = f"call-e.{i}"
        t.task_id = str(i)
        t.segments[2].role_override = Role.R
        t.segments[2].call_role_override = Role.E
        assert t.by_role("E") == []  # 既有 by_role 只看正文，不能改變此契約。
        assert len(e_candidates(t)) == 1
        assert cli.check_trajectory(t) == []
        ts.append(t)
    p = tmp_path / "calls.jsonl"
    cli.write_jsonl(ts, str(p))
    with pytest.raises(SystemExit) as e:
        cli.main(["validate", str(p)])
    assert e.value.code == 0
    capsys.readouterr()
    cli.main(["stats", str(p), "--json"])
    assert json.loads(capsys.readouterr().out)[0]["med_num_E"] == 1
    out = tmp_path / "balanced.jsonl"
    cli.main(["sample", str(p), "--per-class", "1", "--out", str(out)])
    assert len(list(cli.read_jsonl([str(out)]))) == 2


def test_d_selector_answer_preserves_calls_and_text_order():
    t = _toy()
    s = t.segments[2]
    s.role_override = Role.A
    s.call_role_override = Role.A
    s.call.name = "write_first"
    s.text = "same_segment_text_after_call"
    t.segments[3].role_override = Role.E
    t.segments[3].kind = "tool_result"
    t.segments[3].text = "tool_result_between_action_and_reply"
    t.segments[-1].text = "last_reply"
    answer = selector_view(t)["answer"]
    assert (
        answer.index("write_first")
        < answer.index("same_segment_text_after_call")
        < answer.index("last_reply")
    )
    assert "tool_result_between_action_and_reply" not in answer


@pytest.mark.parametrize("tail", ['\n{"run_tag": "partial', ""])
def test_e_judge_resume_repairs_tail_before_append(tmp_path, monkeypatch, tail):
    j = _load_script("judge_bcp")
    old = {
        "run_tag": "old",
        "query_id": "old",
        "status": "ok",
        "correct": False,
        "candidate_hash": "x",
    }
    out = tmp_path / "judgments.jsonl"
    out.write_text(json.dumps(old) + tail)
    import openai

    client = NS(
        chat=NS(
            completions=NS(
                create=lambda **kw: NS(
                    choices=[
                        NS(message=NS(content="correct: yes"), finish_reason="stop")
                    ]
                )
            )
        )
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    j.main(
        [
            "--queries",
            str(FX / "bcp_queries.jsonl"),
            "--runs",
            str(FX / "bcp_run.jsonl"),
            "--out",
            str(out),
        ]
    )
    labels = browsecomp_plus.load_judgments(str(out))
    assert len(labels) == 2
    assert labels[("old", "old")]["correct"] is False
    assert out.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_f_solver_retry_preserves_records_outside_items(tmp_path, monkeypatch):
    from tests.test_run_solver import FakeClient, _run

    out, _ = _run(tmp_path, monkeypatch, FakeClient())
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    rows[1]["meta"]["warnings"] = ["truncated"]
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ids = tmp_path / "ids.txt"
    ids.write_text(rows[0]["id"] + "\n")
    client = FakeClient()
    _run(tmp_path, monkeypatch, client, extra=("--ids", str(ids), "--retry-incomplete"))
    after = {row["id"]: row for row in map(json.loads, out.read_text().splitlines())}
    assert len(after) == 2
    assert after[rows[1]["id"]] == rows[1]
    assert client.calls == 1


def test_g_unmatched_override_preserves_same_input_output(tmp_path):
    p = tmp_path / "same.jsonl"
    cli.write_jsonl([_toy()], str(p))
    before = p.read_bytes()
    ov = tmp_path / "overrides.jsonl"
    ov.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"traj_id": "toy.1", "seg_id": 1, "role": "E", "source": "manual"},
                {"traj_id": "absent", "seg_id": 0, "role": "E", "source": "manual"},
            ]
        )
        + "\n"
    )
    with pytest.raises(SystemExit) as e:
        cli.main(["apply-overrides", str(p), str(ov), "--out", str(p)])
    assert e.value.code == 1
    assert p.read_bytes() == before
    assert not (tmp_path / "same.jsonl.tmp").exists()


@pytest.mark.parametrize("adapter", ["chat", "swebench", "tau2"])
def test_h_reversed_tool_results_are_detected(adapter, tmp_path, capsys):
    if adapter == "tau2":
        messages = [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "x", "name": "read_x", "arguments": {}},
                    {"id": "y", "name": "read_y", "arguments": {}},
                ],
            },
            {"role": "tool", "id": "y", "content": "result Y"},
            {"role": "tool", "id": "x", "content": "result X"},
            {"role": "assistant", "content": "done"},
        ]
        t = tau2._convert_sim(
            {"task_id": "1", "messages": messages, "reward_info": {"reward": 1}},
            "mock",
            "default",
            "m",
            {},
            {"mock": {"assistant": {"read_x": "WRITE", "read_y": "WRITE"}}},
            None,
            "none",
            "self_generated",
        )
    else:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "x",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"read X"}',
                        },
                    },
                    {
                        "id": "y",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"read Y"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "y", "content": "result Y"},
            {"role": "tool", "tool_call_id": "x", "content": "result X"},
            {"role": "assistant", "content": "done"},
        ]
        if adapter == "chat":
            from qera.chat import add_chat_messages

            b = TrajectoryBuilder()
            b.add("Q", "user", "question", "task")
            add_chat_messages(b, messages, is_final_action=lambda name, args: True)
            t = b.build(
                traj_id="chat.order",
                benchmark="chat",
                domain="tool_use",
                task_id="1",
                solver=SolverInfo(model="m"),
                label=Label(correct=True, source="unit_tests"),
            )
        else:
            p = tmp_path / "order.traj.json"
            p.write_text(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "task"}]
                        + messages
                        + [{"role": "exit", "extra": {"submission": "patch"}}]
                    }
                )
            )
            t = swebench.convert_traj(str(p), True, "m", "none")
    results = [s for s in t.segments if s.text in ("result Y", "result X")]
    assert [s.text for s in results] == ["result Y", "result X"]  # 只偵測，不重排。
    assert t.meta["call_order_changed"] == 1
    assert all(s.ambiguous and "回傳順序" in s.note for s in results)
    assert all(s.ambiguous for s in t.segments if s.call is not None)
    p = tmp_path / "stats.jsonl"
    cli.write_jsonl([t], str(p))
    cli.main(["stats", str(p), "--json"])
    assert json.loads(capsys.readouterr().out)[0]["call_order_changed"] == 1


@pytest.mark.parametrize("order, expected", [("xyz", 0), ("zyx", 2)])
def test_h_order_counter_and_stats_count_trajectories(
    tmp_path, capsys, order, expected
):
    b = TrajectoryBuilder()
    b.add("Q", "user", "question", "task")
    for key in "xyz":
        b.record_call(key)
    for key in order:
        result = b.add("E", "environment", "tool_result", key)
        b.record_result(key, result)
    b.add("A", "solver", "final_answer", "answer")
    t = b.build(
        traj_id="counter",
        benchmark="counter",
        domain="tool_use",
        task_id="1",
        solver=SolverInfo(model="m"),
        label=Label(correct=True, source="unit_tests"),
    )
    assert t.meta["call_order_changed"] == expected
    assert all(s.ambiguous == bool(expected) for s in t.by_role("E"))
    p = tmp_path / "counter.jsonl"
    cli.write_jsonl([t], str(p))
    cli.main(["stats", str(p), "--json"])
    assert json.loads(capsys.readouterr().out)[0]["call_order_changed"] == int(
        expected > 0
    )


@pytest.mark.parametrize("condition", ["QERA", "QA", "QEA"])
@pytest.mark.parametrize("argument", ["selected_e", "budget"])
def test_i_nonselective_conditions_reject_selection_arguments(condition, argument):
    kwargs = {argument: [] if argument == "selected_e" else 0}
    with pytest.raises(ValueError, match=f"{argument} 只能用於 QEsA / QEJevA"):
        build_verifier_input(_toy(), condition, **kwargs)


def test_i_qea_without_selection_preserves_all_evidence():
    t = _toy()
    vi = build_verifier_input(t, "QEA")
    for candidate in e_candidates(t):
        assert candidate["seg_id"] in vi.included_seg_ids
        assert candidate["text"] in vi.text
        assert candidate["call"] in vi.text
    assert vi.tokens_by_role["E"] == t.tokens_by_role()["E"]


def test_ii_solver_failure_exits_preserves_success_and_resumes(
    tmp_path, monkeypatch, capsys
):
    from tests.test_run_solver import FakeClient, _run, run_solver

    exs = [
        json.loads(line) for line in (FX / "musique_dev.jsonl").read_text().splitlines()
    ]
    bad_prompt = musique.make_solver_prompt(exs[1])

    class PartlyFailingClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.failed_calls = 0

        def _create(self, **kw):
            if kw["messages"][0]["content"] == bad_prompt:
                self.failed_calls += 1
                raise RuntimeError("persistent fixture failure")
            return super()._create(**kw)

    monkeypatch.setattr(run_solver.time, "sleep", lambda seconds: None)
    client = PartlyFailingClient()
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, monkeypatch, client)
    assert e.value.code not in (None, 0)
    assert "重跑同一行指令即可補上失敗的題目" in capsys.readouterr().out
    out = tmp_path / "runs" / "musique.fake.jsonl"
    successful = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(successful) == 1 and successful[0]["id"] == exs[0]["id"]
    assert client.calls == 1 and client.failed_calls == 5
    recovered = FakeClient()
    _run(tmp_path, monkeypatch, recovered)
    rows = {row["id"]: row for row in map(json.loads, out.read_text().splitlines())}
    assert set(rows) == {ex["id"] for ex in exs}
    assert rows[exs[0]["id"]] == successful[0]
    assert recovered.calls == 1
