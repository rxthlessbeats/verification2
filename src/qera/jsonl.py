"""JSONL 讀取。

不要用 str.splitlines()：它會把 U+2028、U+2029、\x85 等 Unicode 字元也當成換行，
模型輸出裡常有這些字元，會把一筆資料切成兩半。
JSONL 的分隔只有 "\n"。
"""

from __future__ import annotations

import json
from collections.abc import Iterator


def lines(path) -> Iterator[str]:
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:  # 檔案迭代只依 \n 切分
            if line.strip():
                yield line


def read(path) -> Iterator[dict]:
    for line in lines(path):
        yield json.loads(line)
