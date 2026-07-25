"""
Request Tracer — structured end-to-end tracing for every pipeline request.

Every request gets:
    - A unique request_id
    - Per-stage spans (ingest, delta, retrieve, llm, answer) with latency
    - Status (success / error) and error details
    - LLM telemetry (tokens, cost)
    - JSON trace file saved to logs/

Usage:
    trace = RequestTrace()
    with trace.span("ingest"):
        ... do ingestion ...
    with trace.span("delta"):
        ... compute delta ...
    trace.save()
"""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager
from dataclasses import dataclass, field
from src.config.settings import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Span — one stage in the pipeline
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """One stage in a traced request (e.g., ingest, delta, llm)."""
    stage: str
    request_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    start_iso: str = ""              # human-readable start
    end_iso: str = ""                # human-readable end
    status: str = "pending"          # pending → running → success / error
    error: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        if self.end_time and self.start_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0.0

    def to_dict(self) -> dict:
        result = {
            "stage": self.stage,
            "request_id": self.request_id,
            "start_time": self.start_iso,
            "end_time": self.end_iso,
            "latency_ms": self.latency_ms,
            "status": self.status,
        }
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# ---------------------------------------------------------------------------
# Request Trace — the full end-to-end record of one request
# ---------------------------------------------------------------------------

class RequestTrace:
    """End-to-end trace of one pipeline request.

    Tracks all stages with timing, status, errors, and LLM telemetry.
    Saves the complete trace as a JSON file for inspection.
    """

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or uuid4().hex[:16]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.start_time = time.time()
        self.end_time: float = 0.0
        self.spans: list[Span] = []
        self.llm_telemetry: dict = {}
        self.status = "running"
        self.error: str = ""
        self._current_span: Span | None = None

        logger.info(
            "trace_started",
            request_id=self.request_id,
        )

    @contextmanager
    def span(self, stage: str, **metadata):
        """Context manager to trace one pipeline stage.

        Usage:
            with trace.span("ingest", pid="pid_a"):
                ... do ingestion ...
        """
        s = Span(
            stage=stage,
            request_id=self.request_id,
            start_time=time.time(),
            start_iso=datetime.now(timezone.utc).isoformat(),
            status="running",
            metadata=metadata,
        )
        self._current_span = s
        self.spans.append(s)

        logger.info(
            "span_started",
            request_id=self.request_id,
            stage=stage,
            **metadata,
        )

        try:
            yield s
            s.end_time = time.time()
            s.end_iso = datetime.now(timezone.utc).isoformat()
            s.status = "success"

            logger.info(
                "span_completed",
                request_id=self.request_id,
                stage=stage,
                latency_ms=s.latency_ms,
                status="success",
            )

        except Exception as e:
            s.end_time = time.time()
            s.end_iso = datetime.now(timezone.utc).isoformat()
            s.status = "error"
            s.error = f"{type(e).__name__}: {str(e)}"

            logger.error(
                "span_failed",
                request_id=self.request_id,
                stage=stage,
                latency_ms=s.latency_ms,
                error=s.error,
                traceback=traceback.format_exc(),
            )
            raise  # re-raise so the caller sees it

        finally:
            self._current_span = None

    def record_llm(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> None:
        """Record LLM telemetry — tokens, model, provider, cost."""
        self.llm_telemetry = {
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
        }

        logger.info(
            "llm_telemetry",
            request_id=self.request_id,
            **self.llm_telemetry,
        )

    def finish(self, status: str = "success", error: str = "") -> None:
        """Mark the trace as complete."""
        self.end_time = time.time()
        self.status = status
        self.error = error

        logger.info(
            "trace_completed",
            request_id=self.request_id,
            status=self.status,
            total_latency_ms=self.total_latency_ms,
            stages=len(self.spans),
        )

    @property
    def total_latency_ms(self) -> float:
        if self.end_time and self.start_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0.0

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Full trace as a JSON-serializable dict."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "total_latency_ms": self.total_latency_ms,
            "latency_breakdown": self._latency_breakdown(),
            "error": self.error,
            "stages": [s.to_dict() for s in self.spans],
            "llm_telemetry": self.llm_telemetry,
        }

    def _latency_breakdown(self) -> dict:
        """Extract per-stage latencies for quick inspection."""
        breakdown = {}
        for s in self.spans:
            breakdown[f"{s.stage}_ms"] = s.latency_ms
        return breakdown

    def save(self, output_dir: Path | None = None) -> Path:
        """Save the trace as a JSON file."""
        if output_dir is None:
            output_dir = get_settings().LOGS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"trace_{self.request_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(
            "trace_saved",
            request_id=self.request_id,
            path=str(path),
        )

        return path
