"""從模型輸出取出原生推理（R）的共用規則。

為什麼要集中在這裡：
之前每個 adapter 各自寫一套，有的只認 reasoning_content、有的會處理 <think>、
有的不會，結果「同一種回應」在不同 benchmark 會得到不同的 R。R 是研究問題 (1) 的核心，
規則必須一致，所以全部 adapter 和 run_solver.py 都改用這個模組。

支援的來源（依常見程度）：
1. 訊息欄位 reasoning_content（vLLM 舊版、DeepSeek、LiteLLM）
2. 訊息欄位 reasoning（vLLM 新版）
3. 訊息欄位 thinking / thinking_blocks（Anthropic）
4. 內文中的 <think>...</think>（沒有開 reasoning parser 時，Qwen 會這樣輸出）
"""

from __future__ import annotations

import re
from typing import Any

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_think(content: str) -> tuple[str, str, bool]:
    """把內文拆成 (推理, 其餘內容, 是否有沒收尾的 <think>)。

    沒收尾的 <think>（只有開頭沒有結尾）通常代表生成被 max_tokens 截斷：
    模型還在思考就被切掉了。這時 <think> 之後的全部都是推理，**不能**當成答案。
    """
    content = content or ""
    thinks = [t.strip() for t in THINK_BLOCK.findall(content)]
    rest = THINK_BLOCK.sub("", content)
    unclosed = False

    if "<think>" in rest:  # 有開頭、沒結尾：截斷
        before, after = rest.split("<think>", 1)
        thinks.append(after.strip())
        rest = before
        unclosed = True
    elif (
        "</think>" in rest
    ):  # 有些 chat template 把 <think> 放在 prompt 裡，輸出只剩結尾標記
        head, rest = rest.split("</think>", 1)
        thinks.insert(0, head.strip())

    return "\n\n".join(t for t in thinks if t), rest.strip(), unclosed


def from_message(msg: dict[str, Any]) -> str:
    """從一則 assistant 訊息（dict）的獨立欄位取出推理，不含內文中的 <think>。"""
    parts: list[str] = []
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list):  # 例如 OpenAI reasoning summary：[{"text": ...}]
            parts.extend(
                str(x.get("text", "")).strip()
                for x in val
                if isinstance(x, dict) and x.get("text")
            )
    for tb in msg.get("thinking_blocks") or []:
        if isinstance(tb, dict) and tb.get("thinking"):
            parts.append(str(tb["thinking"]).strip())
    seen, out = set(), []
    for p in parts:  # 同一段推理可能同時出現在兩個欄位，去重
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n\n".join(out)


def from_object(msg: Any) -> str:
    """同 from_message，但輸入是 openai 套件回傳的物件（run_solver 用）。"""
    d: dict[str, Any] = {}
    for key in ("reasoning_content", "reasoning", "thinking", "thinking_blocks"):
        val = getattr(msg, key, None)
        if val is None:
            val = (getattr(msg, "model_extra", None) or {}).get(key)
        if val is not None:
            d[key] = val
    return from_message(d)
