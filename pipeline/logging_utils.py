"""Job-scoped structured logging.

Every log record carries the ``job_id`` so a single video's stages can be traced
across modules. Set ``LOG_JSON=1`` for one-JSON-object-per-line output, which is
what production log collectors expect.

Records always go to stderr. Passing ``log_file`` keeps a copy on disk as well:
dubbing a long video can outlive a terminal, a scrollback buffer or an impatient
Ctrl-C, and the only account of what the run did would otherwise go with it. The
file is **appended** to, never replaced - one input is normally dubbed many times
while its options are tuned, and the log of the run that produced the artifacts
now in the folder is exactly the one worth keeping. Each run opens with a
``Run started`` line, so the file can still be read one run at a time. Once that
file passes ``DEFAULT_MAX_LOG_BYTES`` it is rolled aside to a timestamped name and
a fresh one is begun, so "keep every run" cannot come to mean "grow forever".
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, MutableMapping

__all__ = [
    "setup_logging",
    "job_logger",
    "timed",
    "quiet_dependencies",
    "JobLogger",
    "DEFAULT_MAX_LOG_BYTES",
]

#: Size at which the run log is archived rather than appended to. Roughly twenty
#: ordinary 11-minute runs, or two at ``--verbose``.
DEFAULT_MAX_LOG_BYTES = 2 * 1024 * 1024

#: Dependencies that are extremely chatty at DEBUG (HTTP downloads, model
#: loading). ``-v`` is about this pipeline's steps, not their request logs, so
#: these stay at WARNING even when the root level is DEBUG.
_QUIET_LOGGERS = (
    "urllib3",
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "openai",
    "huggingface_hub",
    "hf_transfer",
    "modelscope",
    "modelscope_hub",
    "modelscope_hub.download",
    "modelscope.utils.logger",
    "transformers",
    "filelock",
    "asyncio",
    "matplotlib",
    "numba",
    "requests",
    "urllib3.connectionpool",
)


def quiet_dependencies() -> None:
    """Keep third-party loggers at WARNING even when the root level is DEBUG.

    ``-v`` is about this pipeline's steps, not about every HTTP request a
    dependency makes. Applied by :func:`setup_logging` in the parent and by each
    worker after it configures its own logging.
    """
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

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


def _console_line(record: logging.LogRecord, formatter: logging.Formatter) -> str:
    """``LEVEL [job_id] message`` - the shape stderr and the log file share."""
    job_id = getattr(record, "job_id", None)
    prefix = f"[{job_id}] " if job_id else ""
    base = f"{record.levelname:<7} {prefix}{record.getMessage()}"
    if record.exc_info:
        base = f"{base}\n{formatter.formatException(record.exc_info)}"
    return base


class _ConsoleFormatter(logging.Formatter):
    """Human-friendly console output including the job id when present."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        return _console_line(record, self)


class _FileFormatter(logging.Formatter):
    """Console formatting plus a wall clock, for a log that outlives the run.

    stderr needs no timestamp because it is live; a file read tomorrow does, and
    the stamp is what lets a line be matched against ``stage_seconds`` in
    ``report.json``.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        stamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{stamp} {_console_line(record, self)}"


def _human_bytes(size: int) -> str:
    """``0.6 KiB`` / ``2.4 MiB``: a roll note is read at a glance, not audited."""
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / 1024**2:.1f} MiB"


def _archived_path(path: Path) -> Path:
    """``run-20260929-070308.log``: named for when the log it holds was written."""
    try:
        when = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:  # pragma: no cover - the caller has just stat-ed the same file
        when = datetime.now()
    suffix = path.suffix or ".log"
    stamped = f"{when:%Y%m%d-%H%M%S}"
    candidate = path.with_name(f"{path.stem}-{stamped}{suffix}")
    counter = 1
    while candidate.exists():  # two rolls inside one second still land separately
        candidate = path.with_name(f"{path.stem}-{stamped}-{counter}{suffix}")
        counter += 1
    return candidate


def _archive_oversized(path: Path, max_bytes: int) -> tuple[Path, int] | None:
    """Move a full log aside under a timestamped name.

    Rolling happens as a run *starts*, never in the middle of one, so a run's lines
    are not split across two files: every boundary stays a ``Run started`` line. A
    run that is itself larger than the cap stays whole, and the run after it begins
    the new file.

    Returns:
        ``(where it went, how big it was)`` when a file was moved, else ``None``.
        Failing to tidy up is silent by design - housekeeping must never become
        the reason someone loses a run.
    """
    if max_bytes <= 0 or not path.is_file():
        return None
    try:
        size = path.stat().st_size
        if size < max_bytes:
            return None
        return path.replace(_archived_path(path)), size
    except OSError:
        return None


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


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    *,
    log_file: Path | str | None = None,
    max_bytes: int | None = None,
) -> Path | None:
    """Configure the root logger exactly once.

    Args:
        level: Root log level name, e.g. ``INFO`` or ``DEBUG``.
        json_output: Emit one JSON object per line instead of the console format.
            Applies to the log file too, so a collector can read either.
        log_file: Also write every record here, creating the parent directory and
            appending to whatever is already there. ``None`` keeps stderr as the
            only destination.
        max_bytes: Roll an existing log aside once it reaches this many bytes, so
            the live file stays bounded. ``None`` applies
            ``DEFAULT_MAX_LOG_BYTES``; ``0`` never rolls.

    Returns:
        The log file that was opened, or ``None`` when nothing was written to
        disk - which includes a path that could not be opened, reported as a
        warning rather than raised. A log file is a convenience; it must never be
        the reason a job fails.
    """
    if json_output:
        formatter, file_formatter = _JsonFormatter(), _JsonFormatter()
    else:
        formatter, file_formatter = _ConsoleFormatter(), _FileFormatter()

    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    handlers.append(stream)

    resolved: Path | None = None
    failure: str | None = None
    archived: tuple[Path, int] | None = None
    if log_file is not None:
        candidate = Path(log_file)
        limit = DEFAULT_MAX_LOG_BYTES if max_bytes is None else int(max_bytes)
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            archived = _archive_oversized(candidate, limit)
            file_handler = logging.FileHandler(candidate, mode="a", encoding="utf-8")
        except OSError as exc:
            failure = f"Could not write the log file {candidate}: {exc}"
        else:
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
            resolved = candidate

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
        if isinstance(existing, logging.FileHandler):
            # Only a file handler holds an OS resource, and an abandoned handle
            # would keep the path open for the life of the process.
            existing.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    quiet_dependencies()

    # Announced through the logger itself, so the line lands in the file it names
    # and doubles as the boundary between one run's lines and the next's.
    if resolved is not None:
        logging.getLogger(__name__).info("Run started: appending to %s", resolved)
    if archived is not None:
        rolled, size = archived
        logging.getLogger(__name__).info(
            "The log had reached %s, so it is now %s",
            _human_bytes(size),
            rolled.name,
        )
    if failure is not None:
        logging.getLogger(__name__).warning("%s. Logging to stderr only.", failure)
    return resolved


def job_logger(name: str, job_id: str, **context: Any) -> JobLogger:
    """Build a logger bound to a job id (and any extra context)."""
    return JobLogger(logging.getLogger(name), {"job_id": job_id, **context})


@contextmanager
def timed(
    logger: logging.Logger | logging.LoggerAdapter,
    message: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Time a block and log the result, on success and on failure.

    The yielded mapping is merged into the record, so a caller can attach facts
    it only learns inside the block (line counts, devices, sizes) to the same
    line that carries the duration. ``seconds`` is always present, which is what
    makes a JSON log queryable for slow steps.

    Args:
        logger: Logger or adapter to emit through.
        message: What the block does, without a duration ("Transcribed audio").
        level: Level to log at; DEBUG for per-line or per-batch detail.
        **fields: Extra record fields known before the block starts.
    """
    start = time.perf_counter()
    context: dict[str, Any] = {}
    try:
        yield context
    except BaseException:
        elapsed = time.perf_counter() - start
        logger.log(
            level,
            "%s failed after %.2fs",
            message,
            elapsed,
            extra={**fields, **context, "seconds": round(elapsed, 3)},
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.log(
            level,
            "%s in %.2fs",
            message,
            elapsed,
            extra={**fields, **context, "seconds": round(elapsed, 3)},
        )
