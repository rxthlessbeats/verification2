"""Score one E candidate by how useful it is for verifying the candidate output.

For each candidate we ask Jev one four-way choice question: what role does
this single piece of evidence play for a verifier deciding whether the
candidate output is correct? irrelevant / context / confirms / refutes.
usefulness = P(confirms) + P(refutes). Evidence that bears on the verdict in
either direction is worth keeping; exploration and background are noise.

Two boolean questions (supports? contradicts?) were tried first and rejected:
on tau2 nearly every tool result scored > 0.5 on "contradicts" because any
record whose details differ from the one the agent acted on looked like a
conflict. A single choice question forces the probability mass to be shared,
and naming "other records the agent retrieved but did not use" as irrelevant
fixed the ranking (probe: 20 tau2 + 10 SWE trajectories, 2026-11-04).

The scorer only ever sees question, answer and the candidate text.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
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
        "A customer-service agent handled a user's request under the written policy included in "
        "the task. The candidate output is the agent's replies and tool calls. It is correct when "
        "every action and statement complies with the policy and matches the data returned by the "
        "tools; declining a request or transferring to a human is correct whenever the policy "
        "requires it. It is wrong only if the agent broke a policy rule, performed an action the "
        "policy did not allow, acted on data that conflicts with the tool results, or told the user "
        "something false."
    ),
    "swebench_verified": (
        "A coding agent was asked to fix an issue in a repository. The candidate output is the "
        "final patch (a diff). It is correct if the patch resolves the issue without breaking "
        "other behaviour; it is wrong otherwise."
    ),
    "agentrewardbench": (
        "A web agent was asked to complete a task in a browser. The candidate output is the "
        "sequence of actions and the final response. It is correct if the task was actually "
        "completed as asked and the final response is accurate; it is wrong if the agent did the "
        "wrong thing or claimed something the pages do not support. Each piece of evidence is one "
        "page the agent saw. Search-engine home pages, blocked or CAPTCHA pages, loading pages and "
        "other intermediate pages the output does not rely on are irrelevant, not refutations; a "
        "page refutes the output only when its content disagrees with a specific claim in the "
        "output or shows the task's final state was not reached."
    ),
}
DEFAULT_HINT = (
    "A solver produced a candidate output for the task. It is correct if it satisfies the task; "
    "it is wrong otherwise."
)

QUESTIONS: dict[str, dict[str, Any]] = {
    "role": {
        "type": "choice",
        "instructions": (
            "A verifier must decide whether the candidate output is correct for the task. "
            "What role does this single piece of evidence play for that decision?"
        ),
        "criteria": {
            "irrelevant": (
                "unrelated to the candidate output, or an exploratory lookup, search, file read, or "
                "record the output does not depend on and does not act on; other records the agent "
                "retrieved while searching but then did not use belong here even if their details "
                "differ from the record the output acted on"
            ),
            "context": (
                "related background that helps understand the task or the output, but neither "
                "confirms nor refutes any specific claim or action in the output"
            ),
            "confirms": (
                "directly confirms a specific fact, result, state, or action that the output relies "
                "on or claims, such as a successful test, a tool result the output used correctly, "
                "or a final state matching what the output says"
            ),
            "refutes": (
                "directly shows a specific claim or action in the output is wrong, failed, violated "
                "a rule, or disagrees with this evidence about the same record, file, or state; "
                "describing a different record than the one the output acted on is not a refutation"
            ),
        },
    },
}


@dataclass(frozen=True)
class CandidateScore:
    seg_id: int
    supports: float  # P(confirms)
    contradicts: float  # P(refutes)
    input_tokens: int  # Jev tokens billed for this call (0 on cache hit)
    cost_usd: float
    latency_s: float
    irrelevant: float = 0.0
    context: float = 0.0
    truncated: bool = False  # evidence was cut to fit Jev's context limit
    cached: bool = False

    @property
    def usefulness(self) -> float:
        return min(1.0, self.supports + self.contradicts)

    @property
    def stance(self) -> str:
        """Most likely role; irrelevant and context both read as neutral."""
        if max(self.irrelevant, self.context) >= max(self.supports, self.contradicts):
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
                irrelevant=cached.get("irrelevant", 0.0),
                context=cached.get("context", 0.0),
                input_tokens=0,
                cost_usd=0.0,
                latency_s=0.0,
                truncated=truncated,
                cached=True,
            )
        result = self.client.decide(state, QUESTIONS)
        p = result.answers["role"].probabilities
        score = CandidateScore(
            seg_id=candidate["seg_id"],
            supports=float(p.get("confirms", 0.0)),
            contradicts=float(p.get("refutes", 0.0)),
            irrelevant=float(p.get("irrelevant", 0.0)),
            context=float(p.get("context", 0.0)),
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
        try:
            return json.loads(p.read_text())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            # Partial or corrupt entry (e.g. interrupted write): treat as a miss and rescore.
            return None

    def _write_cache(self, key: str, score: CandidateScore) -> None:
        if not self.cache_dir:
            return
        # Atomic replace: identical candidates in one trajectory are scored concurrently,
        # and a reader must never see a half-written file.
        final = self.cache_dir / f"{key}.json"
        tmp = self.cache_dir / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(asdict(score)))
        os.replace(tmp, final)
