"""Job execution boundary.

Production runs each job in a spawned process. Tests (and, later, a remote
worker) can supply the same small protocol without changing the state machine.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
from typing import Any, Callable, Protocol

__all__ = ["JobExecutor", "ProcessExecutor"]

OnItem = Callable[[dict[str, Any]], None]


class JobExecutor(Protocol):
    """Runs one payload and reports items to ``on_item``."""

    def execute(self, payload: dict[str, Any], on_item: OnItem) -> int:
        """Block until the job finishes and return its exit code."""

    def cancel(self) -> None:
        """Best-effort termination of the running job."""


class ProcessExecutor:
    """Run one job at a time in a spawned child process."""

    def __init__(self) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._process: multiprocessing.Process | None = None

    def execute(self, payload: dict[str, Any], on_item: OnItem) -> int:
        events = self._ctx.Queue()
        process = self._ctx.Process(
            target=_worker_entry,
            args=(payload, events),
            name=f"dubsmith-{str(payload.get('job_id', ''))[:8]}",
        )
        with self._lock:
            self._process = process
        process.start()
        try:
            while True:
                try:
                    item = events.get(timeout=0.25)
                except queue.Empty:
                    if not process.is_alive():
                        break
                    continue
                on_item(item)
            process.join(timeout=10)
            while True:
                try:
                    on_item(events.get_nowait())
                except queue.Empty:
                    break
            return int(process.exitcode or 0)
        finally:
            try:
                events.close()
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass
            with self._lock:
                self._process = None

    def cancel(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.is_alive():
            process.terminate()


def _worker_entry(payload: dict[str, Any], events: Any) -> None:
    from .worker import run_job

    run_job(payload, events)
