"""Durable local state for the web API.

Single-user, SQLite-only, no ORM. Every user-owned row already carries a
``user_id`` so adding authentication later is additive rather than a migration of
the whole data model.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["Store"]

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "QUEUED": {"PROCESSING", "CANCELLED", "FAILED"},
    "PROCESSING": {"COMPLETED", "FAILED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    stage TEXT,
    params TEXT NOT NULL,
    result TEXT,
    error TEXT,
    user_id TEXT NOT NULL DEFAULT 'local',
    input_path TEXT,
    job_dir TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    started REAL,
    finished REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created DESC);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    message TEXT,
    data TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    kind TEXT NOT NULL,
    job_id TEXT,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_created ON files(created DESC);
"""

class Store:
    """Thread-safe SQLite access shared by request handlers and the job worker."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

    # -- jobs -------------------------------------------------------------- #

    def create_job(
        self,
        job_id: str,
        kind: str,
        params: dict[str, Any],
        *,
        input_path: str | None = None,
        job_dir: str | None = None,
        user_id: str = "local",
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, state, params, user_id, input_path,"
                " job_dir, created) VALUES (?, ?, 'QUEUED', ?, ?, ?, ?, ?)",
                (
                    job_id,
                    kind,
                    json.dumps(params, default=str),
                    user_id,
                    input_path,
                    job_dir,
                    now,
                ),
            )
        self.add_event(job_id, "state", message="Queued")

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "state",
            "stage",
            "result",
            "error",
            "started",
            "finished",
            "attempts",
        }
        values = {k: v for k, v in fields.items() if k in allowed}
        if not values:
            return
        if "result" in values and not isinstance(values["result"], str):
            values["result"] = json.dumps(values["result"], default=str)
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock:
            # ``assignments`` is built only from the fixed allow-list above.
            self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",  # noqa: S608
                (*values.values(), job_id),
            )

    def transition(self, job_id: str, state: str, **fields: Any) -> None:
        """Move a job to ``state`` if the transition is valid.

        Raises:
            ValueError: If the current state cannot reach ``state``.
        """
        row = self.get_job(job_id)
        if row is None:
            return
        current = str(row["state"])
        if state != current and state not in _ALLOWED_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid job transition {current} -> {state}")
        self.update_job(job_id, state=state, **fields)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_row(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def count_active_jobs(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('QUEUED', 'PROCESSING')"
            ).fetchone()
        return int(row[0]) if row else 0

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        for key in ("params", "result"):
            if payload.get(key):
                try:
                    payload[key] = json.loads(payload[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return payload

    # -- events ------------------------------------------------------------ #

    def add_event(
        self,
        job_id: str,
        kind: str,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO job_events (job_id, ts, kind, message, data)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    job_id,
                    time.time(),
                    kind,
                    message,
                    json.dumps(data, default=str) if data else None,
                ),
            )
            return int(cursor.lastrowid)

    def events_after(
        self, job_id: str, after_id: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? AND id > ?"
                " ORDER BY id ASC LIMIT ?",
                (job_id, after_id, limit),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if item.get("data"):
                try:
                    item["data"] = json.loads(item["data"])
                except (TypeError, json.JSONDecodeError):
                    pass
            events.append(item)
        return events

    def tail_events(self, job_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in reversed(rows):
            item = dict(row)
            if item.get("data"):
                try:
                    item["data"] = json.loads(item["data"])
                except (TypeError, json.JSONDecodeError):
                    pass
            events.append(item)
        return events

    # -- settings ---------------------------------------------------------- #

    def get_settings(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (TypeError, json.JSONDecodeError):
                result[row["key"]] = row["value"]
        return result

    def patch_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for key, value in values.items():
                self._conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(value, default=str)),
                )
        return self.get_settings()

    # -- files ------------------------------------------------------------- #

    def add_file(
        self,
        file_id: str,
        original_name: str,
        stored_name: str,
        path: str,
        size: int,
        kind: str,
        *,
        job_id: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO files (id, original_name, stored_name, path, size,"
                " kind, job_id, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    original_name,
                    stored_name,
                    path,
                    size,
                    kind,
                    job_id,
                    time.time(),
                ),
            )

    def list_files(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM files ORDER BY created DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_file(self, file_id: str) -> dict[str, Any] | None:
        item = self.get_file(file_id)
        if item is None:
            return None
        with self._lock:
            self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        return item

    def close(self) -> None:
        with self._lock:
            self._conn.close()
