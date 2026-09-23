"""Job-scoped structured logging.

Every log record carries the ``job_id`` so a single video's stages can be traced
across modules. Set ``LOG_JSON=1`` for one-JSON-object-per-line output, which is
what production log collectors expect.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, MutableMapping

__all__ = ["setup_logging", "job_logger", "JobLogger"]

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class _JsonFormatter(logging.Formatter):
    """Render records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Human-friendly console output including the job id when present."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        job_id = getattr(record, "job_id", None)
        prefix = f"[{job_id}] " if job_id else ""
        base = f"{record.levelname:<7} {prefix}{record.getMessage()}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


class JobLogger(logging.LoggerAdapter):
    """LoggerAdapter that stamps every record with a ``job_id``."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra.setdefault("job_id", self.extra["job_id"] if self.extra else None)
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **extra: Any) -> "JobLogger":
        """Return a new adapter with additional context fields."""
        merged = dict(self.extra or {})
        merged.update(extra)
        return JobLogger(self.logger, merged)


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure the root logger exactly once."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if json_output else _ConsoleFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def job_logger(name: str, job_id: str, **context: Any) -> JobLogger:
    """Build a logger bound to a job id (and any extra context)."""
    return JobLogger(logging.getLogger(name), {"job_id": job_id, **context})
