"""run_solver.py 的測試。不需要真的 vLLM：用假的 client 模擬回傳。

測三件事：
1. 輸出格式正確，而且 musique adapter 能直接讀（整條流程接得起來）
2. 中斷後重跑，不會重複跑已完成的題目
3. 新舊版 vLLM 的思考欄位（reasoning_content / reasoning）都抓得到
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

from qera.adapters import musique

ROOT = Path(__file__).resolve().parent.parent
FX = Path(__file__).parent / "fixtures"

# scripts/ 不是 package，用路徑載入
_spec = importlib.util.spec_from_file_location(
    "run_solver", ROOT / "scripts" / "run_solver.py"
)
run_solver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_solver)


class FakeClient:
    """模擬 openai.OpenAI：記錄收到幾次請求，回傳固定格式的答案。"""

    def __init__(self, reasoning_field="reasoning_content", finish="stop"):
        self.calls = 0
        self.reasoning_field = reasoning_field
        self.finish = finish
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        msg = NS(
            content="The kettle maker was founded by Ada.\nAnswer: Ada Brightwater",
            model_extra={},
        )
        setattr(
            msg, self.reasoning_field, "Zephyr is made by Lumen; Lumen founded by Ada."
        )
        return NS(
            choices=[NS(message=msg, finish_reason=self.finish)],
            usage=NS(prompt_tokens=100, completion_tokens=40),
        )


def _run(tmp_path, monkeypatch, client, extra=()):
    prompts = tmp_path / "prompts.jsonl"
    exs = [json.loads(l) for l in (FX / "musique_dev.jsonl").read_text().splitlines()]
    prompts.write_text(
        "".join(
            json.dumps({"id": e["id"], "prompt": musique.make_solver_prompt(e)}) + "\n"
            for e in exs
        )
    )
    out = tmp_path / "runs" / "musique.fake.jsonl"
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    run_solver.main(
        [
            "--prompts",
            str(prompts),
            "--model",
            "Qwen/fake",
            "--out",
            str(out),
            "--workers",
            "2",
            *extra,
        ]
    )
    return out, exs


def test_output_format_and_adapter_roundtrip(tmp_path, monkeypatch):
    client = FakeClient()
    out, exs = _run(tmp_path, monkeypatch, client)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2 and client.calls == 2
    r = rows[0]
    assert (
        r["messages"][0]["role"] == "user" and r["messages"][1]["role"] == "assistant"
    )
    assert r["messages"][1]["reasoning_content"].startswith("Zephyr")
    assert r["meta"]["warnings"] == [] and r["meta"]["gen_params"]["thinking"] is True

    # 整條流程：solver 輸出 → musique adapter → QERA trajectory
    by_id = {e["id"]: e for e in exs}
    t = musique.convert_one(by_id[r["id"]], r, "dev", "full")
    assert t.by_role("R")[0].text.startswith("Zephyr")  # 思考內容成為 R
    assert any(s.text == "Ada Brightwater" for s in t.by_role("A"))


def test_resume_skips_done(tmp_path, monkeypatch):
    client = FakeClient()
    out, _ = _run(tmp_path, monkeypatch, client)
    assert client.calls == 2
    client2 = FakeClient()
    _run(tmp_path, monkeypatch, client2)  # 同樣指令再跑一次
    assert client2.calls == 0  # 全部已完成，不會重送
    assert len(out.read_text().splitlines()) == 2  # 也不會重複寫入


def test_new_vllm_reasoning_field_and_truncation_warning(tmp_path, monkeypatch):
    client = FakeClient(reasoning_field="reasoning", finish="length")
    out, _ = _run(tmp_path, monkeypatch, client)
    r = json.loads(out.read_text().splitlines()[0])
    assert r["messages"][1]["reasoning_content"].startswith(
        "Zephyr"
    )  # 新欄位名也抓得到
    assert "truncated" in r["meta"]["warnings"]  # 被截斷要標出來
