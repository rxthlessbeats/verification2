"""Token 計數。

Jev 和其他 selection baseline 都要在「相同 token 預算」下比較，所以每個 segment 的
token 數要用 verifier 的 tokenizer（Qwen）來算才準。

優先順序：
1. 環境變數 QERA_TOKENIZER 指向的 HuggingFace tokenizer（本機路徑或 repo 名稱），
   例如 export QERA_TOKENIZER=Qwen/Qwen3-32B（可再用 QERA_TOKENIZER_REVISION 固定版本）
   ⚠ 有設定時，載入失敗會直接報錯，不會退回下面的備援方式
2. dashscope 套件內附的 Qwen BPE 詞表（與 Qwen2/3 的基礎詞表相同，誤差很小）
3. tiktoken 的 o200k_base（只適合粗估）
4. 字元數 / 4 的近似值

使用的 tokenizer 名稱會記在 Trajectory.tokenizer，之後換 tokenizer 只要跑
`python -m qera.cli retokenize` 重算即可。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache


@lru_cache(maxsize=1)
def get_counter() -> tuple[str, Callable[[str], int]]:
    name = os.environ.get("QERA_TOKENIZER")
    if name:
        # 有指定 tokenizer 時，載入失敗就直接停止，不默默改用其他 tokenizer：
        # 正式實驗的 token 預算必須用 verifier 實際的 tokenizer 計算，
        # 如果悄悄退回近似值，不同人跑出的「相同預算」其實不相同。
        # 只想粗估時，不要設定 QERA_TOKENIZER 即可。
        try:
            from transformers import AutoTokenizer  # type: ignore

            revision = os.environ.get(
                "QERA_TOKENIZER_REVISION"
            )  # 固定版本，避免模型更新後 token 數改變
            tok = AutoTokenizer.from_pretrained(name, revision=revision)
            tag = f"{name}@{revision}" if revision else name
            return tag, lambda s: len(tok.encode(s, add_special_tokens=False))
        except Exception as e:  # pragma: no cover - 依環境而定
            raise RuntimeError(
                f"[qera.tokens] 指定的 tokenizer {name} 載入失敗：{e}。"
                "正式實驗不能用替代 tokenizer；請修正環境，或取消 QERA_TOKENIZER 改用近似計數。"
            ) from e
    try:  # dashscope（pip install dashscope）附帶 Qwen 的 BPE 詞表，不需要連 HuggingFace
        import importlib.util

        import tiktoken  # type: ignore
        from tiktoken.load import load_tiktoken_bpe  # type: ignore

        spec = importlib.util.find_spec("dashscope")
        if spec and spec.submodule_search_locations:
            path = os.path.join(
                next(iter(spec.submodule_search_locations)),
                "resources",
                "qwen.tiktoken",
            )
            if os.path.exists(path):
                ranks = load_tiktoken_bpe(path)
                pat = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                enc = tiktoken.Encoding(
                    "qwen", pat_str=pat, mergeable_ranks=ranks, special_tokens={}
                )
                return "qwen-bpe(dashscope)", lambda s: len(
                    enc.encode(s, disallowed_special=())
                )
    except Exception:  # noqa: BLE001, S110 - 載入失敗就換下一個備援 tokenizer
        pass
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("o200k_base")
        return "tiktoken:o200k_base", lambda s: len(
            enc.encode(s, disallowed_special=())
        )
    except Exception:  # noqa: BLE001, S110 - 載入失敗就換下一個備援 tokenizer
        pass
    return "approx:chars/4", lambda s: max(1, (len(s) + 3) // 4) if s else 0


def count(text: str) -> int:
    return get_counter()[1](text)


def name() -> str:
    return get_counter()[0]
