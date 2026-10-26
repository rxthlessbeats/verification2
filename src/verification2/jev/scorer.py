"""Score evidence units by how useful they are for verifying a candidate answer.

For each evidence unit we ask Jev two boolean questions in one request:
  supports    - does this evidence support the candidate answer?
  contradicts - does this evidence contradict the candidate answer?
The unit's usefulness is max(P(supports), P(contradicts)): evidence that
strongly bears on the answer in either direction is worth keeping; evidence
that does neither is noise. Labels are never used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from verification2.jev.client import JevClient, JevResult

SUPPORTS_Q: dict[str, Any] = {
    "type": "boolean",
    "instructions": (
        "Does this evidence provide information that supports the candidate answer to the question being correct?"
    ),
    "criteria": {
        "true": "the evidence contains facts that make the candidate answer more likely to be correct",
        "false": "the evidence is irrelevant to the candidate answer, or does not bear on its correctness",
    },
}

CONTRADICTS_Q: dict[str, Any] = {
    "type": "boolean",
    "instructions": (
        "Does this evidence provide information that contradicts the candidate "
        "answer to the question, or suggests it is incorrect?"
    ),
    "criteria": {
        "true": "the evidence contains facts that conflict with the candidate answer or point to a different answer",
        "false": "the evidence is irrelevant to the candidate answer, or does not conflict with it",
    },
}


@dataclass(frozen=True)
class EvidenceScore:
    unit_id: str
    supports: float
    contradicts: float
    input_tokens: int
    cost_usd: float
    latency_s: float

    @property
    def usefulness(self) -> float:
        return max(self.supports, self.contradicts)

    @property
    def stance(self) -> str:
        if self.usefulness < 0.5:
            return "neutral"
        return "supports" if self.supports >= self.contradicts else "contradicts"


class JevScorer:
    """Scores (question, candidate_answer, evidence) triples with Jev.

    Scores are cached on disk keyed by the content hash, so sweeping token
    budgets later never re-queries the API. Selection cost (s_i,c in the
    proposal) is the sum of input_tokens / cost_usd over the calls actually made.
    """

    def __init__(self, client: JevClient, cache_dir: str | Path | None = ".jev_cache") -> None:
        self.client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def score(self, question: str, candidate_answer: str, unit_id: str, evidence: str) -> EvidenceScore:
        key = self._cache_key(question, candidate_answer, evidence)
        cached = self._read_cache(key)
        if cached is not None:
            return EvidenceScore(**{**cached, "unit_id": unit_id, "cost_usd": 0.0, "latency_s": 0.0})

        state = {
            "question": question,
            "candidate_answer": candidate_answer,
            "evidence": evidence,
        }
        result: JevResult = self.client.decide(state, {"supports": SUPPORTS_Q, "contradicts": CONTRADICTS_Q})
        es = EvidenceScore(
            unit_id=unit_id,
            supports=float(result.answers["supports"].probability or 0.0),
            contradicts=float(result.answers["contradicts"].probability or 0.0),
            input_tokens=result.input_tokens,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
        )
        self._write_cache(key, es)
        return es

    def score_many(self, question: str, candidate_answer: str, units: list[tuple[str, str]]) -> list[EvidenceScore]:
        """units: list of (unit_id, evidence_text) in original order."""
        return [self.score(question, candidate_answer, uid, text) for uid, text in units]

    @staticmethod
    def _cache_key(question: str, candidate_answer: str, evidence: str) -> str:
        payload = json.dumps(
            {"q": question, "a": candidate_answer, "e": evidence, "sq": SUPPORTS_Q, "cq": CONTRADICTS_Q},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def _write_cache(self, key: str, es: EvidenceScore) -> None:
        if not self.cache_dir:
            return
        (self.cache_dir / f"{key}.json").write_text(json.dumps(asdict(es)))
