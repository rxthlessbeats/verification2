"""AgentRewardBench adapter。

輸入：
1. HF 資料集 McGill-NLP/agent-reward-bench 的 trajectories/cleaned/ 資料夾
   結構：cleaned/<benchmark>/<GenericAgent-LLM>/<exp>/<task_id>.json
   每個檔案：{benchmark, agent, model, goal, steps:[{num, reasoning, action, url, axtree_pruned,
             last_action_error, screenshot_path, ...}], ...}（欄位見 repo 的 trajectories.py）
2. 標籤：repo 內 agent_reward_bench/data/annotations.csv（專家標註，已附在本專案 resources/）
3. 分割：agent_reward_bench/data/splits.csv（dev / test）

Q/E/R/A 規則（SPEC.md §4.6）：
- Q : goal
- E : 每一步的觀測（URL、錯誤訊息、pruned accessibility tree）——是 agent 做決定前看到的頁面
- R : 每一步模型原始輸出中的 <think>…</think>（取自 step.chat_messages 最後一則 assistant 訊息）。
      ⚠ 不直接用 step.reasoning 欄位：模型沒寫 <think> 時，AgentLab 會把 <action>…</action>
        放進這個欄位，實際檢查有 28% 的 trajectory 的 R 全部都是這種動作文字，不是推理。
        原始輸出拿不到時才退回 reasoning 欄位，而且會先去掉 <action> 區塊。
- A : 每一步的動作（click / fill / goto / send_msg_to_user ...）。網頁任務的「產出」就是
      這串動作造成的狀態改變加上最後回覆給使用者的訊息；最後一個動作標 final_answer
- 標籤：trajectory_success == "Successful"。多位標註者時要「過半數」同意才採用；
        沒過半、或多數是 Unsure → None（抽樣時自動排除），票數記在 label.detail

範圍：
- 預設排除 VisualWebArena：它的任務常需要看圖才能判斷，verifier 只吃文字，
  失敗可能是「資訊沒給」而不是「E 沒用」，會污染結論。要納入請用 --web-benchmarks 指定。
- AgentLab 的 chat_messages 是每一步送給 LLM 的完整 prompt，不是使用者的後續對話；
  這幾個網頁 benchmark 的任務在開始時就給定，沒有後續使用者訊息。

截圖路徑放在 hidden.screenshot_path，目前只用文字；之後若 verifier 改用多模態再加。
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

from .. import reasoning
from ..builder import TrajectoryBuilder
from ..schema import Label, Role, SolverInfo, Trajectory

BENCHMARK = "agentrewardbench"
DOMAIN = "software_web"
FINAL_ACTIONS = ("send_msg_to_user", "report_infeasible", "stop")
DEFAULT_WEB_BENCHMARKS = (
    "webarena,workarena,assistantbench"  # 排除 visualwebarena（需要看圖）
)
ACTION_BLOCK = re.compile(r"<action>.*?</action>", re.DOTALL)
VARIANT_RE = re.compile(
    r"\.(resized|improved)(?=\.)"
)  # 題號的變體標記，splits.csv 裡沒有


def normalize_task_id(task_id: str) -> str:
    """visualwebarena.resized.398 → visualwebarena.398（splits.csv 用的是沒有變體標記的題號）"""
    return VARIANT_RE.sub("", task_id)


def _read_csv(path: str | None, default_name: str) -> list[dict]:
    if path:
        return list(csv.DictReader(open(path, encoding="utf-8")))
    with (
        resources.files("qera.resources")
        .joinpath(default_name)
        .open(encoding="utf-8") as f
    ):
        return list(csv.DictReader(f))


def load_labels(path: str | None = None) -> dict[tuple[str, str], dict]:
    votes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in _read_csv(path, "arb_annotations.csv"):
        votes[(r["model_name"].strip(), r["task_id"].strip())].append(r)
    out = {}
    for k, rs in votes.items():
        c = Counter(r["trajectory_success"].strip() for r in rs)
        top, n = c.most_common(1)[0]
        # 要過半數才採用；相對多數（例如 2:1:1）不夠
        majority = n * 2 > len(rs)
        correct = (
            top == "Successful"
            if majority and top in ("Successful", "Unsuccessful")
            else None
        )
        out[k] = {
            "correct": correct,
            "votes": dict(c),
            "n_annotators": len(rs),
            "side_effect": Counter(r["trajectory_side_effect"] for r in rs).most_common(
                1
            )[0][0],
            "looping": Counter(r["trajectory_looping"] for r in rs).most_common(1)[0][
                0
            ],
            "optimality": Counter(r["trajectory_optimality"] for r in rs).most_common(
                1
            )[0][0],
        }
    return out


def _observation(step: dict) -> str:
    parts = []
    if step.get("url"):
        parts.append(f"URL: {step['url']}")
    if step.get("last_action_error"):
        parts.append(f"Last action error: {step['last_action_error']}")
    ax = step.get("axtree_pruned") or step.get("axtree") or ""
    if ax:
        parts.append(f"Accessibility tree:\n{ax}")
    return "\n".join(parts)


def _raw_assistant_output(step: dict) -> str | None:
    """這一步模型的原始輸出：chat_messages 裡最後一則 assistant 訊息。"""
    for m in reversed(step.get("chat_messages") or []):
        if m.get("role") == "assistant":
            c = m.get("content", m.get("message"))
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            return c if isinstance(c, str) else None
    return None


def _step_reasoning(step: dict) -> tuple[str, str]:
    """回傳 (推理文字, 來源)。來源用於追溯與稽核。"""
    raw = _raw_assistant_output(step)
    if raw is not None:
        think, _, _ = reasoning.split_think(raw)
        return think, "raw_output/think"
    fallback = ACTION_BLOCK.sub("", step.get("reasoning") or "").strip()
    return fallback, "reasoning_field(action_removed)"


def convert_file(path: Path, labels: dict, splits: dict) -> Trajectory | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    model_dir = path.parent.parent.name  # GenericAgent-<LLM>
    task_id = path.stem
    lab = labels.get((model_dir, task_id))
    b = TrajectoryBuilder()
    b.add(Role.Q, "user", "goal", d.get("goal", ""), source_id="goal")
    steps = d.get("steps", [])
    last_action_idx = max(
        (i for i, s in enumerate(steps) if s.get("action")), default=-1
    )
    r_sources: Counter = Counter()
    for i, s in enumerate(steps):
        num = s.get("num", i)
        b.add(
            Role.E,
            "environment",
            "observation",
            _observation(s),
            step=num,
            source_id=f"step:{num}/observation",
            hidden={"screenshot_path": s.get("screenshot_path")},
        )
        r_text, r_src = _step_reasoning(s)
        if r_text:
            r_sources[r_src] += 1
        b.add(
            Role.R,
            "solver",
            "reasoning",
            r_text,
            step=num,
            source_id=f"step:{num}/{r_src}",
        )
        act = s.get("action")
        if act:
            final = i == last_action_idx or str(act).startswith(FINAL_ACTIONS)
            b.add(
                Role.A,
                "solver",
                "final_answer" if final else "action",
                act,
                step=num,
                source_id=f"step:{num}/action",
            )
    if last_action_idx < 0:
        b.add(Role.A, "solver", "final_answer", "(no action)", skip_empty=False)

    llm = model_dir.replace("GenericAgent-", "")
    return b.build(
        traj_id=f"arb.{task_id}.{llm}",
        benchmark=BENCHMARK,
        domain=DOMAIN,
        task_id=task_id,
        split=splits.get(normalize_task_id(task_id)),
        solver=SolverInfo(
            model=llm,
            agent="GenericAgent(AgentLab)",
            origin="official_release",
            native_reasoning="full",
        ),
        label=Label(
            correct=lab["correct"] if lab else None,
            source="human_annotation" if lab else "pending",
            detail={k: v for k, v in (lab or {}).items() if k != "correct"},
        ),
        meta={
            "web_benchmark": d.get("benchmark") or path.parts[-4],
            "num_steps": len(steps),
            "text_only": True,
            "reasoning_sources": dict(r_sources),
        },
    )


def add_args(p) -> None:
    p.add_argument("--cleaned", required=True, help="trajectories/cleaned 資料夾")
    p.add_argument(
        "--annotations", default=None, help="annotations.csv（預設用內附副本）"
    )
    p.add_argument("--splits", default=None, help="splits.csv（預設用內附副本）")
    p.add_argument(
        "--web-benchmarks",
        default=DEFAULT_WEB_BENCHMARKS,
        help="要納入的網頁 benchmark；預設排除 visualwebarena（任務需要看圖）",
    )
    p.add_argument("--split", default=None, help="只保留 dev 或 test")


def run(args) -> Iterator[Trajectory]:
    labels = load_labels(args.annotations)
    splits = {
        r["task_id"]: r["split"] for r in _read_csv(args.splits, "arb_splits.csv")
    }
    keep = set(args.web_benchmarks.split(","))
    for f in sorted(Path(args.cleaned).rglob("*.json")):
        if f.parts[-4] not in keep:
            continue
        t = convert_file(f, labels, splits)
        if t and (args.split is None or t.split == args.split):
            yield t
