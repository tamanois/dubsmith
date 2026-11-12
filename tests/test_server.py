"""API and job-state tests. No GPU, models, network or ffmpeg."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline.voices import save_voice
from server.app import create_app
from server.jobs import JobManager, QueueFullError
from server.store import Store


class FakeExecutor:
    """Synchronous executor that records payloads and emits scripted items."""

    def __init__(
        self, items: list[dict[str, Any]] | None = None, code: int = 0
    ) -> None:
        self.items = items or []
        self.code = code
        self.cancelled = False
        self.payloads: list[dict[str, Any]] = []

    def execute(self, payload: dict[str, Any], on_item: Any) -> int:
        self.payloads.append(payload)
        for item in self.items:
            on_item(item)
        return self.code

    def cancel(self) -> None:
        self.cancelled = True


def _wait(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = client.get(f"/api/jobs/{job_id}").json()
        if row["state"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return row
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def _upload(client: TestClient, name: str = "clip.wav") -> str:
    response = client.post(
        "/api/files/upload",
        files={"file": (name, b"RIFF0000WAVEfmt ", "audio/wav")},
    )
    assert response.status_code == 200, response.text
    return response.json()["path"]


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DUB_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("VOICES_DIR", str(tmp_path / "voices"))
    executor = FakeExecutor(
        items=[
            {"type": "event", "kind": "stage_start", "data": {"stage": "transcribe"}},
            {
                "type": "event",
                "kind": "stage_end",
                "data": {"stage": "transcribe", "seconds": 0.1},
            },
            {"type": "result", "data": {"job_dir": str(tmp_path)}},
        ]
    )
    app = create_app(data_dir=tmp_path / "data", executor=executor)
    with TestClient(app) as test_client:
        test_client.executor = executor  # type: ignore[attr-defined]
        yield test_client


def test_read_endpoints(client: TestClient) -> None:
    for path in (
        "/api/health",
        "/api/devices",
        "/api/models",
        "/api/settings",
        "/api/voices",
        "/api/files",
        "/api/packages",
        "/api/anchors",
        "/api/info",
    ):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)


def test_upload_rejects_unknown_extension(client: TestClient) -> None:
    response = client.post(
        "/api/files/upload", files={"file": ("evil.exe", b"x", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_upload_and_delete(client: TestClient) -> None:
    response = client.post(
        "/api/files/upload",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
    )
    assert response.status_code == 200
    item = response.json()
    assert Path(item["path"]).is_file()
    assert client.delete(f"/api/files/{item['id']}").status_code == 200
    assert not Path(item["path"]).exists()


def test_dub_job_uses_a_unique_run_folder(
    client: TestClient, tmp_path: Path
) -> None:
    path = _upload(client)
    body = {
        "config": {"audio": path, "target_language": "French"},
        "stretcher": "auto",
    }
    first = client.post("/api/jobs/dub", json=body)
    second = client.post("/api/jobs/dub", json=body)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    one = _wait(client, first.json()["job_id"])
    two = _wait(client, second.json()["job_id"])
    assert one["state"] == "COMPLETED"
    assert two["state"] == "COMPLETED"
    one_dir = Path((one["params"]["config"])["job_dir_override"])
    two_dir = Path((two["params"]["config"])["job_dir_override"])
    assert one_dir != two_dir
    assert one["result"]["job_dir"] == str(tmp_path)


def test_dub_rejects_a_path_outside_the_app(client: TestClient) -> None:
    body = {"config": {"audio": "/etc/passwd", "target_language": "French"}}
    response = client.post("/api/jobs/dub", json=body)
    assert response.status_code == 400
    assert "Only files uploaded" in response.json()["detail"]


def test_saved_voice_is_resolved_into_reference_audio(
    client: TestClient, tmp_path: Path
) -> None:
    voices = tmp_path / "voices"
    save_voice(
        voices,
        "warm-fr",
        np.zeros(24000, dtype=np.float32),
        24000,
        language="French",
        source="design",
        ref_text="Bonjour.",
        instruct="warm",
    )
    path = _upload(client)
    body = {
        "config": {"audio": path, "target_language": "French"},
        "voice": "warm-fr",
    }
    response = client.post("/api/jobs/dub", json=body)
    assert response.status_code == 200, response.text
    job = _wait(client, response.json()["job_id"])
    config = job["params"]["config"]
    assert Path(config["ref_audio"]).name == "voice.wav"
    assert config["accent_mode"] == "preserve"


def test_settings_patch_and_validation(client: TestClient) -> None:
    response = client.patch("/api/settings", json={"values": {"duck_db": 9.5}})
    assert response.status_code == 200
    assert response.json()["duck_db"] == 9.5
    assert client.patch(
        "/api/settings", json={"values": {"llm_api_key": "secret"}}
    ).status_code == 400
    assert client.patch(
        "/api/settings", json={"values": {"duck_db": 99}}
    ).status_code == 400
    device = client.patch(
        "/api/settings",
        json={"values": {"default_device": "cpu", "default_dtype": "fp32"}},
    )
    assert device.status_code == 200
    assert device.json()["default_device"] == "cpu"
    assert device.json()["default_dtype"] == "fp32"
    assert client.patch(
        "/api/settings", json={"values": {"default_dtype": "nope"}}
    ).status_code == 400


def test_outputs_are_listed_and_deletable(
    client: TestClient, tmp_path: Path
) -> None:
    target = tmp_path / "output" / "run-one" / "dub.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wave")
    listing = client.get("/api/outputs")
    assert listing.status_code == 200
    names = {item["relative"] for item in listing.json()["files"]}
    assert "run-one/dub.wav" in names
    deleted = client.delete("/api/outputs", params={"path": str(target)})
    assert deleted.status_code == 200
    assert not target.exists()


def test_invalid_transition_is_rejected(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite")
    try:
        store.create_job("one", "dub", {})
        with pytest.raises(ValueError):
            store.transition("one", "COMPLETED")
    finally:
        store.close()


def test_job_events_are_persisted(client: TestClient) -> None:
    path = _upload(client)
    response = client.post(
        "/api/jobs/dub",
        json={"config": {"audio": path, "target_language": "French"}},
    )
    job = _wait(client, response.json()["job_id"])
    kinds = [event["kind"] for event in job["events"]]
    assert "stage_start" in kinds
    assert "stage_end" in kinds


def test_failed_executor_marks_the_job_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUB_OUTPUT_DIR", str(tmp_path / "output"))
    app = create_app(
        data_dir=tmp_path / "data",
        executor=FakeExecutor(
            items=[{"type": "error", "message": "boom"}], code=1
        ),
    )
    with TestClient(app) as client:
        path = _upload(client)
        response = client.post(
            "/api/jobs/dub",
            json={"config": {"audio": path, "target_language": "French"}},
        )
        job = _wait(client, response.json()["job_id"])
        assert job["state"] == "FAILED"
        assert job["error"] == "boom"


class _BlockingExecutor(FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.release = False

    def execute(self, payload: dict[str, Any], on_item: Any) -> int:
        self.started = True
        while not self.release:
            time.sleep(0.01)
        return 0


def _manager(tmp_path: Path, executor: Any, max_queue: int = 4) -> JobManager:
    store = Store(tmp_path / "db.sqlite")
    manager = JobManager(
        store,
        settings_provider=lambda: {},
        executor=executor,
        max_queue=max_queue,
    )
    manager.start()
    return manager


def test_queue_is_bounded(tmp_path: Path) -> None:
    blocking = _BlockingExecutor()
    manager = _manager(tmp_path, blocking, max_queue=1)
    try:
        manager.enqueue("dub", {"config": {}})
        deadline = time.time() + 2
        while not blocking.started and time.time() < deadline:
            time.sleep(0.01)
        with pytest.raises(QueueFullError):
            manager.enqueue("dub", {"config": {}})
    finally:
        blocking.release = True
        manager.stop()


def test_cancel_running_job(tmp_path: Path) -> None:
    blocking = _BlockingExecutor()
    manager = _manager(tmp_path, blocking, max_queue=2)
    try:
        job_id = manager.enqueue("dub", {"config": {}})
        deadline = time.time() + 2
        while not blocking.started and time.time() < deadline:
            time.sleep(0.01)
        row = manager.cancel(job_id)
        assert row is not None and row["state"] == "CANCELLED"
    finally:
        blocking.release = True
        manager.stop()
