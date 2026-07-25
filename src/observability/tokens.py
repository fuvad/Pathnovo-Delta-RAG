"""
Token Usage Tracker — tracks LLM token consumption and estimates cost.

Stores per-call and cumulative token usage across the session.
Cost estimation uses published per-token pricing for supported models.
"""

from dataclasses import dataclass, field
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cost per 1M tokens (USD) — update as pricing changes
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # Groq models (per 1M tokens)
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "mixtral-8x7b-32768": {"prompt": 0.24, "completion": 0.24},
    "gemma2-9b-it": {"prompt": 0.20, "completion": 0.20},
    # Ollama models — local, no cost
    "llama3.2": {"prompt": 0.0, "completion": 0.0},
    "llama3.1": {"prompt": 0.0, "completion": 0.0},
    "mistral": {"prompt": 0.0, "completion": 0.0},
}


# ---------------------------------------------------------------------------
# Single LLM call record
# ---------------------------------------------------------------------------

@dataclass
class TokenRecord:
    """One LLM call's token usage and cost."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Token Usage Tracker
# ---------------------------------------------------------------------------

class TokenTracker:
    """Tracks cumulative LLM token usage and cost across a session."""

    def __init__(self):
        self.records: list[TokenRecord] = []

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> TokenRecord:
        """Record one LLM call and estimate its cost.

        Args:
            provider: "groq" or "ollama"
            model: Model name.
            prompt_tokens: Input tokens.
            completion_tokens: Output tokens.
            total_tokens: Total (if not provided, computed from prompt + completion).

        Returns:
            The recorded TokenRecord with estimated cost.
        """
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

        rec = TokenRecord(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )
        self.records.append(rec)

        logger.info(
            "token_usage_recorded",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
        )

        return rec

    # -------------------------------------------------------------------
    # Cumulative stats
    # -------------------------------------------------------------------

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.estimated_cost_usd for r in self.records)

    @property
    def call_count(self) -> int:
        return len(self.records)

    def summary(self) -> dict:
        """Cumulative usage summary across all calls."""
        return {
            "total_calls": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_estimated_cost_usd": round(self.total_cost_usd, 6),
            "calls": [r.to_dict() for r in self.records],
        }

    # -------------------------------------------------------------------
    # Cost estimation
    # -------------------------------------------------------------------

    @staticmethod
    def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD based on model pricing.

        Returns 0.0 for unknown models or local (Ollama) models.
        """
        pricing = PRICING.get(model)
        if not pricing:
            return 0.0  # unknown model or local — no cost

        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

        return prompt_cost + completion_cost
