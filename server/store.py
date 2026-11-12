"""Durable local state for the web API.

Single-user, SQLite-only, no ORM. Every user-owned row already carries a
``user_id`` so adding authentication later is additive rather than a migration of
the whole data model.
"""

from __future__ import annotations

import json
import re
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

#: Human-readable description for each job kind, so the activity list never shows
#: a raw operation name.
_KIND_TITLES: dict[str, str] = {
    "dub": "Dub video",
    "subtitles": "Generate subtitles",
    "say": "Generate speech",
    "voice_design": "Design voice",
    "voice_clone": "Clone voice",
    "voice_add": "Register voice",
    "model_download": "Download models",
    "anchor_generate": "Generate anchors",
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
CREATE TABLE IF NOT EXISTS presets (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_presets_kind ON presets(kind, created DESC);
CREATE TABLE IF NOT EXISTS favorites (
    path TEXT PRIMARY KEY,
    label TEXT,
    kind TEXT NOT NULL,
    created REAL NOT NULL
);
"""


def _clip(value: str, limit: int = 60) -> str:
    text = str(value).strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def _input_label(row: dict[str, Any]) -> str | None:
    """What the job operated on: a file name, a spoken line, or a description."""
    path = row.get("input_path")
    if path:
        name = Path(path).name
        # Uploads are stored as "<uuid>__<original>"; show the original name.
        return re.sub(r"^[0-9a-f]{32}__", "", name)
    params = row.get("params") or {}
    if not isinstance(params, dict):
        return None
    spec = params.get("spec")
    if isinstance(spec, dict) and spec.get("text"):
        return _clip(str(spec["text"]))
    nested = params.get("params")
    if isinstance(nested, dict):
        for key in ("audio", "prompt", "label"):
            value = nested.get(key)
            if value:
                return Path(str(value)).name if key == "audio" else _clip(str(value))
    return None


def _output_label(result: Any) -> str | None:
    """The primary artifact a completed job produced, as a file name."""
    if not isinstance(result, dict):
        return None
    for key in ("dubbed_video", "output_video", "dub_track", "target_srt", "output"):
        value = result.get(key)
        if isinstance(value, str) and value:
            name = Path(value).name
            if name:
                return name
    return None


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

    def list_jobs(
        self,
        limit: int = 50,
        *,
        offset: int = 0,
        state: str | None = None,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if q:
            clauses.append("(input_path LIKE ? OR id LIKE ? OR kind LIKE ?)")
            needle = f"%{q}%"
            params.extend([needle, needle, needle])
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY created DESC LIMIT ? OFFSET ?",  # noqa: S608
                (*params, limit, offset),
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def count_jobs(self, state: str | None = None, q: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if q:
            clauses.append("(input_path LIKE ? OR id LIKE ? OR kind LIKE ?)")
            needle = f"%{q}%"
            params.extend([needle, needle, needle])
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM jobs{where}", params  # noqa: S608
            ).fetchone()
        return int(row[0]) if row else 0

    def delete_jobs(self, job_ids: list[str]) -> int:
        """Delete terminal jobs and their events; active jobs are left untouched."""
        removed = 0
        with self._lock:
            for job_id in job_ids:
                row = self._conn.execute(
                    "SELECT state FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None or row["state"] in ("QUEUED", "PROCESSING"):
                    continue
                self._conn.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))
                self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                removed += 1
        return removed

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
        payload["title"] = _KIND_TITLES.get(str(payload.get("kind")), payload.get("kind"))
        payload["input_label"] = _input_label(payload)
        payload["output_label"] = _output_label(payload.get("result"))
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

    def list_files(self, limit: int = 100, *, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM files ORDER BY created DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_files(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()
        return int(row[0]) if row else 0

    def delete_files(self, file_ids: list[str]) -> list[dict[str, Any]]:
        """Remove file rows, returning them so the caller can unlink their bytes."""
        removed: list[dict[str, Any]] = []
        with self._lock:
            for file_id in file_ids:
                item = self.get_file(file_id)
                if item is None:
                    continue
                self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                removed.append(item)
        return removed

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

    # -- presets ----------------------------------------------------------- #

    def list_presets(self, kind: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if kind:
                rows = self._conn.execute(
                    "SELECT * FROM presets WHERE kind = ? ORDER BY created DESC",
                    (kind,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM presets ORDER BY created DESC"
                ).fetchall()
        return [self._preset_row(row) for row in rows]

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM presets WHERE id = ?", (preset_id,)
            ).fetchone()
        return self._preset_row(row) if row else None

    def add_preset(
        self, preset_id: str, kind: str, name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            self._conn.execute(
                "INSERT INTO presets (id, kind, name, payload, created)"
                " VALUES (?, ?, ?, ?, ?)",
                (preset_id, kind, name, json.dumps(payload, default=str), time.time()),
            )
        row = self.get_preset(preset_id)
        assert row is not None
        return row

    def delete_preset(self, preset_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM presets WHERE id = ?", (preset_id,)
            )
        return cursor.rowcount > 0

    def _preset_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        try:
            payload["payload"] = json.loads(payload["payload"])
        except (TypeError, json.JSONDecodeError):
            payload["payload"] = {}
        return payload

    # -- favorites --------------------------------------------------------- #

    def list_favorites(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM favorites ORDER BY created DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def add_favorite(self, path: str, label: str | None, kind: str) -> dict[str, Any]:
        with self._lock:
            self._conn.execute(
                "INSERT INTO favorites (path, label, kind, created) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(path) DO UPDATE SET label = excluded.label,"
                " kind = excluded.kind",
                (path, label, kind, time.time()),
            )
            row = self._conn.execute(
                "SELECT * FROM favorites WHERE path = ?", (path,)
            ).fetchone()
        return dict(row) if row else {"path": path, "label": label, "kind": kind}

    def delete_favorite(self, path: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM favorites WHERE path = ?", (path,)
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
