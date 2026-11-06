"""Extract recorded reasoning from API fields and think blocks."""

from __future__ import annotations

import re
from typing import Any

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_think(content: str) -> tuple[str, str, bool]:
    """Return reasoning, remaining content and whether a think block is unclosed."""
    content = content or ""
    thinks = [t.strip() for t in THINK_BLOCK.findall(content)]
    rest = THINK_BLOCK.sub("", content)
    unclosed = False

    if "<think>" in rest:
        before, after = rest.split("<think>", 1)
        thinks.append(after.strip())
        rest = before
        unclosed = True
    elif "</think>" in rest:
        head, rest = rest.split("</think>", 1)
        thinks.insert(0, head.strip())

    return "\n\n".join(t for t in thinks if t), rest.strip(), unclosed


def from_message(msg: dict[str, Any]) -> str:
    """Read separate reasoning fields; content think blocks are handled elsewhere."""
    parts: list[str] = []
    for key in ("reasoning_content", "reasoning_text", "reasoning", "thinking"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list):
            parts.extend(
                str(x.get("text", "")).strip()
                for x in val
                if isinstance(x, dict) and x.get("text")
            )
    for tb in msg.get("thinking_blocks") or []:
        if isinstance(tb, dict) and tb.get("thinking"):
            parts.append(str(tb["thinking"]).strip())
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n\n".join(out)
