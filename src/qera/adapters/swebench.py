"""SWE-bench Verified adapter（mini-swe-agent 的 .traj.json 格式）。

資料來源：SWE-bench/experiments 的 evaluation/verified/<submission>/
- metadata.yaml 的 assets.trajs 指向 trajectory（2026 年中後的提交放在提交者自己的 GitHub）
- per_instance_details.json 有每題的 resolved（= 標籤）

Q/E/R/A 規則（理由見 SPEC.md §4.5）：
- Q : <pr_description> 裡的 issue 內容（agent 用的操作說明模板不算 Q，verifier 用不到）
- E : 每個 bash 指令 + 執行輸出（指令當作來源標頭）
- R : assistant 的文字（旁白 / 思考）；若有 reasoning_content 也放 R
- A : 最後提交的 patch（exit 訊息的 submission）
- 提交那一步（echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch）的輸出就是 patch 本身，
  放進 E 會讓 A 在 QA 條件以外又出現一次，所以略過。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from .. import reasoning
from ..builder import TrajectoryBuilder
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

BENCHMARK = "swebench_verified"
DOMAIN = "software_web"
PR_RE = re.compile(
    r"<pr_description>\s*(?:Consider the following PR description:\s*)?(.*?)\s*</pr_description>",
    re.DOTALL,
)
SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
INLINE_CMD = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
_OUT_RE = re.compile(
    r"<returncode>(-?\d+)</returncode>\s*(?:<output>\n?(.*?)\n?</output>)?", re.DOTALL
)


def _text(c) -> str:
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _command(tc: dict) -> str:
    fn = tc.get("function", {})
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:  # 不是 JSON
            return args
    return (args or {}).get("command", json.dumps(args, ensure_ascii=False))


def convert_traj(
    path: str, resolved: bool | None, model: str, native: str, split: str = "test"
) -> Trajectory:
    d = json.loads(Path(path).read_text())
    iid = d.get("instance_id") or Path(path).name.split(".traj")[0]
    msgs = d["messages"]
    b = TrajectoryBuilder()

    first_user = next((m for m in msgs if m["role"] == "user"), None)
    m_pr = PR_RE.search(_text(first_user["content"])) if first_user else None
    b.add(
        Role.Q,
        "user",
        "issue",
        m_pr.group(1) if m_pr else _text(first_user["content"]) if first_user else "",
        note=None if m_pr else "找不到 <pr_description>，整段 prompt 當 Q",
        ambiguous=m_pr is None,
        source_id="msg:first_user/pr_description",
    )

    pending: dict[str, tuple[str, bool]] = {}
    submission = None
    for i, m in enumerate(msgs):
        role = m["role"]
        if role == "assistant":
            think, content, unclosed = reasoning.split_think(_text(m.get("content")))
            if unclosed:
                b.mark_incomplete("unclosed_think", f"msg:{i}")
            b.add(
                Role.R,
                "solver",
                "reasoning",
                reasoning.from_message(m),
                step=i,
                source_id=f"msg:{i}/reasoning_field",
            )
            b.add(
                Role.R, "solver", "reasoning", think, step=i, source_id=f"msg:{i}/think"
            )
            tcs = m.get("tool_calls") or []
            if not tcs:  # 舊版 mini-swe-agent：指令寫在 ```bash 區塊裡
                for cmd in INLINE_CMD.findall(content):
                    pending[f"inline-{i}"] = (cmd.strip(), SUBMIT_MARK in cmd)
                    b.record_call(f"inline-{i}")
                # 指令會以呼叫的形式出現在對應的執行結果（E）上，旁白裡不再重複一次
                content = INLINE_CMD.sub("", content).strip()
            b.add(
                Role.R,
                "solver",
                "narration",
                content,
                step=i,
                source_id=f"msg:{i}/content",
            )
            for tc in tcs:
                cmd = _command(tc)
                key = tc.get("id", f"{i}")
                pending[key] = (cmd, SUBMIT_MARK in cmd)
                b.record_call(key)
        elif role in ("tool", "user") and m is not first_user:
            key = m.get("tool_call_id") or next(
                (k for k in pending if k.startswith("inline-")), None
            )
            if key is None or key not in pending:
                continue  # 格式錯誤提示等非執行結果
            cmd, is_submit = pending.pop(key)
            if is_submit:
                b.record_result(key)
                continue
            out = _text(m.get("content"))
            mo = _OUT_RE.search(out)
            rc = int(mo.group(1)) if mo else None
            body = (mo.group(2) or "") if mo else out
            result_segment = b.add(
                Role.E,
                "environment",
                "execution_output",
                body if body.strip() else "(no output)",
                step=i,
                call=ToolCall(
                    name="bash", arguments={"command": cmd}, requestor="assistant"
                ),
                hidden={"returncode": rc},
                source_id=f"msg:{i}/tool_result",
            )
            b.record_result(key, result_segment)
        elif role == "exit":
            submission = (m.get("extra") or {}).get("submission") or _text(
                m.get("content")
            )

    if submission is None:
        submission = (d.get("info") or {}).get("submission", "")
    b.add(
        Role.A,
        "solver",
        "patch",
        submission or "(empty patch)",
        skip_empty=False,
        source_id="exit/submission",
    )

    return b.build(
        traj_id=f"swebench.{iid}.{model}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=iid,
        split=split,
        solver=SolverInfo(
            model=model,
            agent="mini-swe-agent",
            origin="official_release",
            native_reasoning=native,
        ),
        label=Label(
            correct=resolved, source="unit_tests" if resolved is not None else "pending"
        ),
        meta={
            "repo": iid.rsplit("-", 1)[0].replace("__", "/"),
            "api_calls": ((d.get("info") or {}).get("model_stats") or {}).get(
                "api_calls"
            ),
            "exit_status": (d.get("info") or {}).get("exit_status"),
            "trajectory_format": d.get("trajectory_format"),
        },
    )


def load_labels(path: str) -> dict[str, bool | None]:
    """讀取每題是否 resolved。

    型別要嚴格檢查：resolved 必須是真正的布林值。之前用 bool(...) 轉換，
    缺值 / null 會變成 False（被當成「答錯」），字串 "false" 會變成 True。
    現在不是布林值的一律設為 None（沒有可用標籤，抽樣時排除），並印出警告。
    """
    d = json.loads(Path(path).read_text())
    if "resolved" in d and isinstance(d["resolved"], list):  # results.json 格式
        return {i: True for i in d["resolved"]} | {
            i: False for i in d.get("unresolved", []) + d.get("no_generation", [])
        }
    out: dict[str, bool | None] = {}
    bad = []
    for k, v in d.items():  # per_instance_details.json
        r = v.get("resolved") if isinstance(v, dict) else None
        if isinstance(r, bool):
            out[k] = r
        else:
            out[k] = None
            bad.append(k)
    if bad:
        print(
            f"[swebench] 警告：{len(bad)} 題的 resolved 不是布林值，標籤設為 None：{bad[:5]}"
        )
    return out


def add_args(p) -> None:
    p.add_argument("--trajs", required=True, help="放 *.traj.json 的資料夾")
    p.add_argument(
        "--labels", required=True, help="per_instance_details.json 或 results.json"
    )
    p.add_argument("--model", required=True, help="solver 名稱（寫進 traj_id）")
    p.add_argument(
        "--native-reasoning",
        default="hidden",
        choices=["full", "summary", "hidden", "none"],
        help="推理是否可見；Gemini / OpenAI 推理模型的思考是加密或隱藏的 → hidden",
    )
    p.add_argument("--ids", default=None, help="只轉換這些 instance（檔案，一行一個）")


def run(args) -> Iterator[Trajectory]:
    labels = load_labels(args.labels)
    keep = set(Path(args.ids).read_text().split()) if args.ids else None
    for f in sorted(Path(args.trajs).rglob("*.traj.json")):
        iid = f.name.split(".traj")[0]
        if keep and iid not in keep:
            continue
        yield convert_traj(str(f), labels.get(iid), args.model, args.native_reasoning)
