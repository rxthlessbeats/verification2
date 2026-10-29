"""AppWorld adapter。

輸入（AppWorld 的實驗輸出資料夾，官方 baseline 可用 `appworld download experiment-outputs` 取得）：
    experiments/outputs/<exp>/tasks/<task_id>/logs/lm_calls.jsonl     每行 {id, input:{messages,...}, output:{...}}
    experiments/outputs/<exp>/tasks/<task_id>/logs/environment_io.md  （lm_calls 不存在時的備援）
    experiments/outputs/<exp>/tasks/<task_id>/dbs/*.jsonl              資料庫變更（= 實際產出）
    experiments/outputs/<exp>/evaluations/<dataset>.json              individual.<task_id>.success
另外需要 data/tasks/<task_id>/specs.json 取得 instruction（不給時從第一則 user 訊息擷取）。

Q/E/R/A 規則（SPEC.md §4.4）：
- Q : 任務 instruction（含 supervisor 名字，因為任務常以「我」指稱 supervisor）
- E : 每次程式執行（或 tool call）的輸出，執行的程式碼當作來源標頭
- R : LM 回覆中程式碼以外的文字、reasoning_content
- A : apis.supervisor.complete_task(answer=...) 的答案 ＋ dbs/ 的資料庫變更摘要
      （AppWorld 用資料庫狀態評分，所以真正的產出物是狀態變更）
- 標籤：evaluations/<dataset>.json 的 success（state-based 單元測試全過）

⚠ 本 adapter 依 AppWorld 原始碼推導格式，尚未用官方實驗輸出實測；第一次轉換後請跑
  `qera audit` 抽查。程式碼裡的註解可能夾帶 solver 的推理，這部分標 ambiguous。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from .. import jsonl
from .. import reasoning as rsn
from ..builder import TrajectoryBuilder, to_text
from ..chat import add_chat_messages
from ..schema import Label, Role, SolverInfo, ToolCall, Trajectory

BENCHMARK = "appworld"
DOMAIN = "tool_use"
CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
COMPLETE_RE = re.compile(
    r"apis\.supervisor\.complete_task\((.*?)\)\s*$", re.DOTALL | re.MULTILINE
)
ANSWER_ARG_RE = re.compile(r"answer\s*=\s*(.+)", re.DOTALL)
EXEC_HDR = re.compile(r"#+ (Execution|Environment Interaction) (\d+(?:\.\d+)?)")
IO_BLOCK = re.compile(r"```python\n(.*?)\n```\n\n```\n(.*?)\n```", re.DOTALL)


def parse_environment_io(text: str) -> list[dict]:
    """與 AppWorld._parse_environment_io_log 相同的格式（較寬鬆的實作）。"""
    out = []
    for m in IO_BLOCK.finditer(text):
        out.append({"input": m.group(1), "output": m.group(2)})
    return out


def _answer_from_code(code: str) -> str | None:
    m = COMPLETE_RE.search(code)
    if not m:
        return None
    a = ANSWER_ARG_RE.search(m.group(1))
    return a.group(1).strip().rstrip(",") if a else "(completed without answer)"


def _db_diff_summary(task_dir: Path, max_lines: int = 200) -> str:
    dbs = task_dir / "dbs"
    if not dbs.exists():
        return ""
    lines = []
    for f in sorted(dbs.glob("*.jsonl")):
        rows = [l.rstrip("\n") for l in jsonl.lines(f)]
        if rows:
            lines.append(f"## {f.stem} ({len(rows)} changes)")
            lines.extend(rows[: max_lines // max(1, len(list(dbs.glob("*.jsonl"))))])
    return "\n".join(lines)


def _instruction(task_dir: Path, data_root: str | None, task_id: str) -> str | None:
    if data_root:
        spec = Path(data_root) / "tasks" / task_id / "specs.json"
        if spec.exists():
            s = json.loads(spec.read_text())
            sup = s.get("supervisor", {})
            who = f"{sup.get('first_name', '')} {sup.get('last_name', '')}".strip()
            return (
                f"{s['instruction']}\n\n(Supervisor: {who}, {sup.get('email', '')})"
                if who
                else s["instruction"]
            )
    return None


def convert_task(
    task_dir: Path, success: bool | None, data_root: str | None, model: str, native: str
) -> Trajectory:
    task_id = task_dir.name
    b = TrajectoryBuilder()
    instr = _instruction(task_dir, data_root, task_id)
    lm_log = task_dir / "logs" / "lm_calls.jsonl"
    answer: str | None = None

    if lm_log.exists():
        calls = list(jsonl.read(lm_log))
        last = calls[-1]
        msgs = list(last["input"]["messages"]) + [
            last["output"] | {"role": "assistant"}
        ]
        # 前置 prompt（system / few-shot / 任務說明）不屬於 solver 的作答，跳過，只取任務本身當 Q
        first_user = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), 0)
        if instr is None:
            instr = to_text(msgs[first_user].get("content"))
        b.add(
            Role.Q,
            "user",
            "instruction",
            instr,
            ambiguous=data_root is None,
            note=None
            if data_root
            else "沒有 specs.json，Q 取自第一則 user 訊息（可能含 prompt 模板）",
        )
        body = msgs[first_user + 1 :]
        if any(m.get("tool_calls") for m in body):  # function-calling agent
            add_chat_messages(
                b,
                body,
                tool_kind="execution_output",
                final_answer_from_tool=lambda n, a: (
                    (to_text((a or {}).get("answer")) or "(completed)")
                    if n.endswith("complete_task")
                    else None
                ),
            )
            answer = "set"
        else:  # code agent：assistant 寫程式碼，下一則 user 訊息是執行輸出
            for i, m in enumerate(body):
                if m.get("role") == "assistant":
                    think, content, _ = rsn.split_think(to_text(m.get("content")))
                    b.add(Role.R, "solver", "reasoning", rsn.from_message(m), step=i)
                    b.add(Role.R, "solver", "reasoning", think, step=i)
                    codes = CODE_RE.findall(content)
                    b.add(
                        Role.R, "solver", "narration", CODE_RE.sub("", content), step=i
                    )
                    pending_code = "\n".join(codes)
                    ans = _answer_from_code(pending_code)
                    if ans is not None:
                        answer = ans
                        b.add(Role.A, "solver", "final_answer", ans, step=i)
                    body[i]["_code"] = pending_code
                elif m.get("role") == "user" and i > 0 and body[i - 1].get("_code"):
                    code = body[i - 1]["_code"]
                    b.add(
                        Role.E,
                        "environment",
                        "execution_output",
                        to_text(m.get("content")),
                        step=i,
                        call=ToolCall(
                            name="python",
                            arguments={"code": code},
                            requestor="assistant",
                        ),
                        ambiguous="#" in code,
                        note="程式碼註解可能含推理" if "#" in code else None,
                    )
    else:
        env_io = task_dir / "logs" / "environment_io.md"
        b.add(
            Role.Q,
            "user",
            "instruction",
            instr or "(instruction unavailable: 請提供 --data-root)",
            ambiguous=instr is None,
        )
        for i, io in enumerate(
            parse_environment_io(env_io.read_text(encoding="utf-8"))
            if env_io.exists()
            else []
        ):
            b.add(
                Role.E,
                "environment",
                "execution_output",
                io["output"],
                step=i,
                call=ToolCall(
                    name="python",
                    arguments={"code": io["input"]},
                    requestor="assistant",
                ),
            )
            ans = _answer_from_code(io["input"])
            if ans is not None:
                answer = ans
                b.add(Role.A, "solver", "final_answer", ans, step=i)

    diff = _db_diff_summary(task_dir)
    if diff:
        b.add(
            Role.A,
            "environment",
            "state_diff",
            diff,
            note="資料庫變更是 solver 動作的結果，屬於產出物",
        )
    if answer is None and not diff:
        b.add(
            Role.A, "solver", "final_answer", "(task not completed)", skip_empty=False
        )

    return b.build(
        traj_id=f"appworld.{task_id}.{model.replace('/', '_')}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=task_id,
        solver=SolverInfo(
            model=model, origin="official_release", native_reasoning=native
        ),
        label=Label(
            correct=success, source="unit_tests" if success is not None else "pending"
        ),
        meta={"experiment": task_dir.parent.parent.name},
    )


def add_args(p) -> None:
    p.add_argument(
        "--experiment", required=True, help="experiments/outputs/<exp> 資料夾"
    )
    p.add_argument(
        "--dataset",
        default="dev",
        help="evaluations/<dataset>.json 的名稱（dev / test_normal ...）",
    )
    p.add_argument(
        "--data-root", default=None, help="AppWorld 的 data 資料夾（讀 specs.json）"
    )
    p.add_argument("--model", default=None, help="solver 名稱（預設用資料夾名稱）")
    p.add_argument(
        "--native-reasoning",
        default="none",
        choices=["full", "summary", "hidden", "none"],
    )


def run(args) -> Iterator[Trajectory]:
    exp = Path(args.experiment)
    ev_path = exp / "evaluations" / f"{args.dataset}.json"
    ev = json.loads(ev_path.read_text())["individual"] if ev_path.exists() else {}
    model = args.model or exp.name
    for task_dir in sorted((exp / "tasks").iterdir()):
        if task_dir.is_dir():
            succ = ev.get(task_dir.name, {}).get("success")
            yield convert_task(
                task_dir, succ, args.data_root, model, args.native_reasoning
            )
