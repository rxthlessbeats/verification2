"""短答案評分（MuSiQue 官方 EM/F1 的正規化方式，與 SQuAD 相同）。"""

from __future__ import annotations

import collections
import re
import string


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def best_over_aliases(pred: str, golds: list[str]) -> tuple[float, float]:
    golds = [g for g in golds if g] or [""]
    return max(exact_match(pred, g) for g in golds), max(f1(pred, g) for g in golds)
