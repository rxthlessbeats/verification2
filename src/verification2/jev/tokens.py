"""Token counting for budgets.

The budget is measured in the VERIFIER's tokens (Qwen), not Jev's. Until the
Qwen tokenizer is wired in, cl100k_base is a close-enough stand-in; swap
`count_tokens` for a HF tokenizer call when the verifier is set up.
"""

from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))
