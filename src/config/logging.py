"""
Structured logging configuration using structlog.

Produces JSON logs with correlation/request IDs for observability.
All modules should use `get_logger(__name__)` to get a bound logger.

Request ID binding:
    Call `bind_request_id(request_id)` at the start of a pipeline run.
    Every subsequent log line from any module will carry that request_id
    automatically via structlog's contextvars integration.
"""

import logging
import sys
import structlog
from src.config.settings import get_settings


def setup_logging() -> None:
    """Configure structlog with JSON or console output based on settings."""
    settings = get_settings()

    # Shared processors applied to every log entry
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,     # picks up bound request_id
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog's formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound with the module name."""
    return structlog.get_logger(name)


def bind_request_id(request_id: str) -> None:
    """Bind a request_id to all subsequent log lines in this context.

    Call this at the start of a pipeline run. Every log line from every
    module will automatically include this request_id until cleared.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """Clear the bound request context (call at end of request)."""
    structlog.contextvars.clear_contextvars()
