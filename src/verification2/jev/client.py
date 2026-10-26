"""Thin HTTP client for Jev via Vercel AI Gateway.

Endpoint: POST https://ai-gateway.vercel.sh/v1/evaluate
Docs:     https://vercel.com/docs/ai-gateway/modalities/decision
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from verification2.jev.usage import UsageLedger

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
DEFAULT_MODEL = "typesafe-ai/jev"

# Hard limits from TypeSafe docs; exceeding them returns an error.
MAX_REQUEST_TOKENS = 64_000
MAX_QUESTION_TOKENS = 32_000


class JevError(RuntimeError):
    """Raised when the gateway returns a non-retryable error."""


@dataclass(frozen=True)
class JevAnswer:
    type: str
    probability: float | None = None  # boolean
    choice: str | None = None  # choice
    score: float | None = None  # score
    probabilities: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> JevAnswer:
        return cls(
            type=d["type"],
            probability=d.get("probability"),
            choice=d.get("choice"),
            score=d.get("score"),
            probabilities=d.get("probabilities", {}),
        )


@dataclass(frozen=True)
class JevResult:
    model: str
    answers: dict[str, JevAnswer]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    raw: dict[str, Any]


class JevClient:
    """Synchronous client with retry on 429 / 5xx.

    Reads AI_GATEWAY_API_KEY from the environment (or .env) unless api_key is given.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        url: str = GATEWAY_URL,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        run: str = "default",
        ledger: UsageLedger | None = None,
    ) -> None:
        load_dotenv()
        self.run = run
        self.ledger = ledger if ledger is not None else UsageLedger()
        self.api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY")
        if not self.api_key:
            raise JevError("AI_GATEWAY_API_KEY is not set (put it in .env)")
        self.model = model
        self.url = url
        self.max_retries = max_retries
        self._http = httpx.Client(
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> JevResult:
        """Ask one or more typed questions about `state`.

        `questions` uses the gateway schema, e.g.
            {"supports": {"type": "boolean", "instructions": "...", "criteria": {...}}}
        """
        body = {"model": self.model, "state": state, "questions": questions}
        backoff = 1.0
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            resp = self._http.post(self.url, json=body)
            latency = time.perf_counter() - t0
            if resp.status_code == 200:
                result = self._parse(resp.json(), latency)
                self.ledger.record(
                    result.model or self.model,
                    self.run,
                    result.input_tokens,
                    result.output_tokens,
                    result.cost_usd,
                    latency,
                    200,
                )
                return result
            self.ledger.record(self.model, self.run, 0, 0, 0.0, latency, resp.status_code)
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == self.max_retries:
                raise JevError(f"gateway {resp.status_code}: {resp.text[:500]}")
            retry_after = resp.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after else backoff)
            backoff = min(backoff * 2, 30.0)
        raise AssertionError("unreachable")

    @staticmethod
    def _parse(data: dict[str, Any], latency: float) -> JevResult:
        usage = data.get("usage", {})
        gateway = data.get("providerMetadata", {}).get("gateway", {})
        return JevResult(
            model=data.get("model", ""),
            answers={k: JevAnswer.from_json(v) for k, v in data["answers"].items()},
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            cost_usd=float(gateway.get("cost", 0.0)),
            latency_s=latency,
            raw=data,
        )

    def close(self) -> None:
        self._http.close()
