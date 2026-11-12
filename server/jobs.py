"""Job state machine and single-slot process executor.

The API must never block on a model-heavy run. Jobs are accepted immediately,
persisted as ``QUEUED``, and executed one at a time in a spawned process. That is
the same resource bound the CLI has today (one model residency), and it leaves the
door open for a remote executor behind the same interface later.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable

from .executor import JobExecutor, ProcessExecutor
from .store import Store

_LOGGER = logging.getLogger(__name__)

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}


class QueueFullError(Exception):
    """Raised when the bounded local queue cannot accept more work."""


class JobManager:
    """Accept jobs, run one at a time, and stream their events into the store."""

    def __init__(
        self,
        store: Store,
        *,
        settings_provider: Callable[[], dict[str, Any]],
        max_queue: int = 32,
        executor: JobExecutor | None = None,
    ) -> None:
        self.store = store
        self._settings_provider = settings_provider
        self._executor = executor or ProcessExecutor()
        self._max_queue = max(1, int(max_queue))
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._payloads: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._current: str | None = None
        self._cancelled: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._recover()
        self._thread = threading.Thread(
            target=self._dispatcher, name="dubsmith-jobs", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._executor.cancel()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _recover(self) -> None:
        """Never silently lose accepted work across a restart."""
        for row in self.store.list_jobs(limit=200):
            if row["state"] == "PROCESSING":
                self.store.transition(
                    row["id"],
                    "FAILED",
                    error="The server restarted while this job was running. Retry it.",
                    finished=time.time(),
                )
                self.store.add_event(
                    row["id"], "error", message="Interrupted by a server restart"
                )
            elif row["state"] == "QUEUED":
                params = row.get("params") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        params = {}
                payload = dict(params)
                payload.update(
                    {
                        "kind": row["kind"],
                        "job_id": row["id"],
                        "settings": self._settings_provider(),
                        "models_dir": payload.get("models_dir")
                        or self._settings_provider().get("models_dir"),
                        "voices_root": payload.get("voices_root")
                        or self._settings_provider().get("voices_dir"),
                    }
                )
                self._payloads[row["id"]] = payload
                self._queue.put(row["id"])

    # -- public API -------------------------------------------------------- #

    def enqueue(
        self,
        kind: str,
        params: dict[str, Any],
        *,
        input_path: str | None = None,
        job_dir: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        if self.store.count_active_jobs() >= self._max_queue:
            raise QueueFullError(
                "The local job queue is full. Wait for a running job to finish."
            )
        job_id = uuid.uuid4().hex
        stored = {k: v for k, v in params.items() if k != "settings"}
        self.store.create_job(
            job_id,
            kind,
            stored,
            input_path=input_path,
            job_dir=job_dir,
        )
        payload = {
            **stored,
            "kind": kind,
            "job_id": job_id,
            "settings": self._settings_provider(),
            "models_dir": stored.get("models_dir")
            or self._settings_provider().get("models_dir"),
            "voices_root": stored.get("voices_root")
            or self._settings_provider().get("voices_dir"),
            **(extra or {}),
        }
        with self._lock:
            self._payloads[job_id] = payload
        self._queue.put(job_id)
        return job_id

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        row = self.store.get_job(job_id)
        if row is None:
            return None
        if row["state"] in _TERMINAL:
            return row
        self._cancelled.add(job_id)
        self.store.transition(job_id, "CANCELLED", finished=time.time())
        if self.is_running(job_id):
            self._executor.cancel()
        self.store.add_event(job_id, "state", message="Cancelled")
        return self.store.get_job(job_id)

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return self._current == job_id

    # -- dispatcher -------------------------------------------------------- #

    def _dispatcher(self) -> None:
        while not self._stopping.is_set():
            job_id = self._queue.get()
            if job_id is None:
                break
            row = self.store.get_job(job_id)
            if row is None or row["state"] in _TERMINAL:
                continue
            with self._lock:
                payload = self._payloads.pop(job_id, None)
            if payload is None:
                payload = self._payload_from_row(row)
            try:
                self._execute(job_id, payload)
            except Exception:  # noqa: BLE001 - one job must not kill the queue
                _LOGGER.exception("Job %s failed in the dispatcher", job_id)
                self.store.transition(
                    job_id,
                    "FAILED",
                    error="The job could not be started.",
                    finished=time.time(),
                )

    def _payload_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        params = row.get("params") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        settings = self._settings_provider()
        return {
            **params,
            "kind": row["kind"],
            "job_id": row["id"],
            "settings": settings,
            "models_dir": params.get("models_dir") or settings.get("models_dir"),
            "voices_root": params.get("voices_root") or settings.get("voices_dir"),
        }

    def _execute(self, job_id: str, payload: dict[str, Any]) -> None:
        # A cancel may have won the race with the dispatcher.
        row = self.store.get_job(job_id)
        if row is None or row["state"] in _TERMINAL:
            return
        with self._lock:
            self._current = job_id
        try:
            self.store.transition(
                job_id,
                "PROCESSING",
                started=time.time(),
                stage="starting",
                attempts=int(self.store.get_job(job_id).get("attempts") or 0) + 1,
            )
            self.store.add_event(job_id, "state", message="Processing")
            exitcode = self._executor.execute(
                payload, lambda item: self._handle_item(job_id, item)
            )
        finally:
            with self._lock:
                self._current = None

        row = self.store.get_job(job_id)
        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            if row is not None and row["state"] not in _TERMINAL:
                self.store.transition(
                    job_id, "CANCELLED", finished=time.time()
                )
            return
        if row is None or row["state"] in _TERMINAL:
            return
        if row.get("error") or exitcode != 0:
            self.store.transition(
                job_id,
                "FAILED",
                error=row.get("error")
                or f"The job process exited with code {exitcode}.",
                finished=time.time(),
            )
            self.store.add_event(job_id, "state", message="Failed")
        else:
            self.store.transition(
                job_id, "COMPLETED", finished=time.time(), stage="done"
            )
            self.store.add_event(job_id, "state", message="Completed")

    def _handle_item(self, job_id: str, item: dict[str, Any]) -> None:
        kind = item.get("type")
        if kind == "log":
            self.store.add_event(
                job_id,
                "log",
                message=str(item.get("message") or ""),
                data={
                    "level": item.get("level"),
                    "stage": item.get("stage"),
                },
            )
            if item.get("stage"):
                self.store.update_job(job_id, stage=str(item["stage"]))
            return
        if kind == "event":
            event_kind = str(item.get("kind") or "event")
            data = item.get("data") or {}
            if data.get("stage"):
                self.store.update_job(job_id, stage=str(data["stage"]))
            self.store.add_event(
                job_id, event_kind, message=str(data.get("stage") or ""), data=data
            )
            return
        if kind == "result":
            self.store.update_job(job_id, result=item.get("data") or {})
            return
        if kind == "error":
            message = str(item.get("message") or "The job failed.")
            self.store.update_job(job_id, error=message)
            self.store.add_event(job_id, "error", message=message)


