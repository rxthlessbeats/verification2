"""把 OpenAI chat 格式的 solver 對話轉成 segments。

組上自己跑 solver（例如千問 / vLLM）產生的 trajectory，請一律存成 OpenAI chat
messages。這樣每個 benchmark 的 adapter 只需要處理「Q 怎麼來、怎麼評分」，
不用各自寫一套解析 solver 輸出的程式。

規則（與 SPEC.md 一致）：
- assistant 的推理欄位、<think>...</think>         → R / reasoning（規則統一在 reasoning.py）
- assistant 伴隨 tool call 的文字                  → R / narration
- tool call + 對應的 tool 回傳                    → E（呼叫內容當作來源標頭，呼叫可另外改判角色）
- 最後一則 assistant 文字                          → A / final_answer
  （若含 "Answer:"，只取其後的部分當 A，前面的當 R）
- 中途其他 assistant 文字                          → R / intermediate_output
- 最後一則訊息有沒收尾的 <think>（被截斷）          → 全部當 R，**不產生 A**，並標記 incomplete
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from . import reasoning
from .builder import TrajectoryBuilder, to_text
from .schema import Role, ToolCall

ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:final answer|answer)(?:\*\*)?\s*[:：]\s*", re.IGNORECASE
)


def split_think(content: str) -> tuple[str, str]:
    """相容舊介面：回傳 (推理, 其餘內容)。沒收尾的 <think> 之後全部算推理。"""
    think, rest, _ = reasoning.split_think(content)
    return think, rest


def split_answer(text: str) -> tuple[str, str]:
    """回傳 (答案之前的推理, 答案)。找不到 "Answer:" 時整段當答案。"""
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return "", text.strip()
    m = matches[-1]
    return text[: m.start()].strip(), text[m.end() :].strip()


def has_answer_line(text: str) -> bool:
    return bool(ANSWER_RE.search(text or ""))


def _content_text(content: Any) -> str:
    if isinstance(content, list):  # OpenAI multi-part content
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return to_text(content)


def _parse_args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:  # 不是 JSON
            return raw
    return raw


def add_chat_messages(
    b: TrajectoryBuilder,
    messages: list[dict],
    *,
    tool_kind: str = "tool_result",
    is_final_action: Callable[[str, Any], bool] | None = None,
    final_answer_from_tool: Callable[[str, Any], str | None] | None = None,
    index_offset: int = 0,
) -> None:
    """處理 solver 開始作答之後的訊息（system / 任務 prompt 請由 adapter 自己處理成 Q）。

    is_final_action(name, args)        : 回傳 True 的 tool call 視為 A / action
    final_answer_from_tool(name, args) : 若某個 tool call 本身就是交答案（例如 AppWorld 的
                                          complete_task(answer=...)），回傳答案字串
    index_offset                        : messages 在原始對話中的起始位置（寫進 source_id）
    """
    pending: dict[str, ToolCall] = {}
    last_text_idx = max(
        (
            i
            for i, m in enumerate(messages)
            if m.get("role") == "assistant" and _content_text(m.get("content")).strip()
        ),
        default=-1,
    )
    answered_by_tool = False

    for i, m in enumerate(messages):
        src = f"msg:{i + index_offset}"
        role = m.get("role")
        if role == "assistant":
            think_inline, content, unclosed = reasoning.split_think(
                _content_text(m.get("content"))
            )
            b.add(
                Role.R,
                "solver",
                "reasoning",
                reasoning.from_message(m),
                step=i,
                source_id=f"{src}/reasoning_field",
            )
            b.add(
                Role.R,
                "solver",
                "reasoning",
                think_inline,
                step=i,
                source_id=f"{src}/think",
            )
            if unclosed:
                b.mark_incomplete(
                    "unclosed_think", f"{src} 的 <think> 沒有收尾（多半是生成被截斷）"
                )

            tool_calls = m.get("tool_calls") or []
            if content:
                if tool_calls:
                    b.add(
                        Role.R,
                        "solver",
                        "narration",
                        content,
                        step=i,
                        source_id=f"{src}/content",
                    )
                elif i == last_text_idx and not answered_by_tool and not unclosed:
                    pre, ans = split_answer(content)
                    b.add(
                        Role.R,
                        "solver",
                        "reasoning",
                        pre,
                        step=i,
                        source_id=f"{src}/content:before_answer",
                    )
                    b.add(
                        Role.A,
                        "solver",
                        "final_answer",
                        ans,
                        step=i,
                        source_id=f"{src}/content:answer",
                    )
                else:
                    b.add(
                        Role.R,
                        "solver",
                        "intermediate_output",
                        content,
                        step=i,
                        source_id=f"{src}/content",
                    )

            for k, tc in enumerate(tool_calls):
                fn = tc.get("function", tc)
                name = fn.get("name", "?")
                args = _parse_args(fn.get("arguments"))
                call = ToolCall(name=name, arguments=args, requestor="assistant")
                ans = (
                    final_answer_from_tool(name, args)
                    if final_answer_from_tool
                    else None
                )
                action_segment = None
                if ans is not None:
                    action_segment = b.add(
                        Role.A,
                        "solver",
                        "final_answer",
                        ans,
                        step=i,
                        call=call,
                        source_id=f"{src}/tool_call:{k}",
                    )
                    answered_by_tool = True
                elif is_final_action and is_final_action(name, args):
                    action_segment = b.add(
                        Role.A,
                        "solver",
                        "action",
                        "",
                        step=i,
                        call=call,
                        source_id=f"{src}/tool_call:{k}",
                    )
                key = tc.get("id", f"{i}:{k}")
                pending[key] = call
                b.record_call(key, action_segment)

        elif role == "tool":
            key = m.get("tool_call_id", "")
            call = pending.pop(key, None)
            result_segment = b.add(
                Role.E,
                "environment",
                tool_kind,
                _content_text(m.get("content")),
                step=i,
                call=call,
                source_id=f"{src}/tool_result",
            )
            b.record_result(key, result_segment)
