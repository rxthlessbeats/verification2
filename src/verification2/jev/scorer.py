"""Score one E candidate by how useful it is for verifying the candidate output.

For each candidate we ask Jev two boolean questions in a single request:
  supports    - does this evidence show the candidate output is correct?
  contradicts - does this evidence show the candidate output is wrong?
usefulness = max(P(supports), P(contradicts)). Evidence that bears on the
verdict in either direction is worth keeping; evidence that does neither is
noise. The scorer only ever sees question, answer and the candidate text.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qera import tokens
from verification2.jev.client import JevClient

# Jev's hard limit is 32k tokens per question (state + question). Keep a
# margin for the instructions and for tokenizer mismatch (qera counts with the
# verifier's tokenizer, Jev with its own).
MAX_STATE_TOKENS = 24_000

# The candidate output means different things per benchmark; tell Jev what it
# is looking at so "supports" and "contradicts" are interpreted correctly.
TASK_HINTS: dict[str, str] = {
    "tau2_bench": (
        "A customer-service agent handled a user's request by calling tools and replying. "
        "The candidate output is the agent's replies and actions. It is correct if it follows "
        "the policy and resolves the request; it is wrong if it violates policy, acts on wrong "
        "data, or fails to resolve the request."
    ),
    "swebench_verified": (
        "A coding agent was asked to fix an issue in a repository. The candidate output is the "
        "final patch (a diff). It is correct if the patch resolves the issue without breaking "
        "other behaviour; it is wrong otherwise."
    ),
    "agentrewardbench": (
        "A web agent was asked to complete a task in a browser. The candidate output is the "
        "sequence of actions and the final response. It is correct if the task was actually "
        "completed as asked; it is wrong if the agent failed, did the wrong thing, or claimed "
        "success without achieving it."
    ),
}
DEFAULT_HINT = (
    "A solver produced a candidate output for the task. It is correct if it satisfies the task; "
    "it is wrong otherwise."
)

QUESTIONS: dict[str, dict[str, Any]] = {
    "supports": {
        "type": "boolean",
        "instructions": (
            "Does this piece of evidence provide information showing that the candidate output "
            "is correct for the task?"
        ),
        "criteria": {
            "true": "the evidence contains facts that make the candidate output more likely to be correct",
            "false": "the evidence is irrelevant to the candidate output, or does not bear on its correctness",
        },
    },
    "contradicts": {
        "type": "boolean",
        "instructions": (
            "Does this piece of evidence provide information showing that the candidate output "
            "is wrong, incomplete, or violates the task requirements?"
        ),
        "criteria": {
            "true": "the evidence contains facts that conflict with the candidate output or reveal an error in it",
            "false": "the evidence is irrelevant to the candidate output, or does not conflict with it",
        },
    },
}


@dataclass(frozen=True)
class CandidateScore:
    seg_id: int
    supports: float
    contradicts: float
    input_tokens: int  # Jev tokens billed for this call (0 on cache hit)
    cost_usd: float
    latency_s: float
    truncated: bool = False  # evidence was cut to fit Jev's context limit
    cached: bool = False

    @property
    def usefulness(self) -> float:
        return max(self.supports, self.contradicts)

    @property
    def stance(self) -> str:
        if self.usefulness < 0.5:
            return "neutral"
        return "supports" if self.supports >= self.contradicts else "contradicts"


def evidence_text(candidate: dict[str, Any]) -> str:
    """Tool call header + result, exactly what the verifier would see for this E."""
    if candidate.get("call"):
        return f"{candidate['call']}\n{candidate['text']}"
    return candidate["text"]


def build_state(
    view: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, str], bool]:
    """State for one candidate; evidence is truncated if the whole thing would exceed Jev's limit."""
    hint = TASK_HINTS.get(view["benchmark"], DEFAULT_HINT)
    fixed = (
        tokens.count(hint)
        + tokens.count(view["question"])
        + tokens.count(view["answer"])
    )
    evidence = evidence_text(candidate)
    room = MAX_STATE_TOKENS - fixed
    if room <= 0:
        raise ValueError(
            f"{view['traj_id']}: question + answer alone exceed Jev's context budget"
        )
    truncated = False
    if tokens.count(evidence) > room:
        # Cut by characters proportionally, then verify. Token/char ratio is ~4 for English text.
        keep = int(len(evidence) * room / tokens.count(evidence) * 0.9)
        evidence = evidence[:keep] + "\n[... truncated for selector context limit ...]"
        truncated = True
    return {
        "task_type": hint,
        "task": view["question"],
        "candidate_output": view["answer"],
        "evidence": evidence,
    }, truncated


class JevScorer:
    """Scores candidates with Jev. Scores are cached on disk by content hash so
    budget sweeps never re-query. Selection cost is the sum of input_tokens /
    cost_usd over calls actually made."""

    def __init__(
        self,
        client: JevClient,
        cache_dir: str | Path | None = ".jev_cache",
        workers: int = 8,
    ) -> None:
        self.client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.workers = max(1, workers)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def score_view(self, view: dict[str, Any]) -> list[CandidateScore]:
        """Score every candidate of a selector_view, in original order."""
        candidates = view["candidates"]
        if not candidates:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda c: self.score_candidate(view, c), candidates))

    def score_candidate(
        self, view: dict[str, Any], candidate: dict[str, Any]
    ) -> CandidateScore:
        state, truncated = build_state(view, candidate)
        key = self._cache_key(state)
        cached = self._read_cache(key)
        if cached is not None:
            return CandidateScore(
                seg_id=candidate["seg_id"],
                supports=cached["supports"],
                contradicts=cached["contradicts"],
                input_tokens=0,
                cost_usd=0.0,
                latency_s=0.0,
                truncated=truncated,
                cached=True,
            )
        result = self.client.decide(state, QUESTIONS)
        score = CandidateScore(
            seg_id=candidate["seg_id"],
            supports=float(result.answers["supports"].probability or 0.0),
            contradicts=float(result.answers["contradicts"].probability or 0.0),
            input_tokens=result.input_tokens,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
            truncated=truncated,
        )
        self._write_cache(key, score)
        return score

    @staticmethod
    def _cache_key(state: dict[str, str]) -> str:
        payload = json.dumps(
            {"state": state, "questions": QUESTIONS}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _write_cache(self, key: str, score: CandidateScore) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(asdict(score)))
