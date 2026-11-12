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
        "/api/usage",
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


def test_theme_is_a_validated_ui_preference(client: TestClient) -> None:
    response = client.patch("/api/settings", json={"values": {"theme": "light"}})
    assert response.status_code == 200
    assert response.json()["theme"] == "light"
    assert (
        client.patch("/api/settings", json={"values": {"theme": "neon"}}).status_code
        == 400
    )


def test_settings_export_and_import_round_trip(client: TestClient) -> None:
    client.patch(
        "/api/settings", json={"values": {"duck_db": 4.5, "theme": "dark"}}
    )
    exported = client.get("/api/settings/export")
    assert exported.status_code == 200
    document = exported.json()
    assert document["version"] == 1
    assert document["settings"]["duck_db"] == 4.5
    assert "llm_api_key" not in document["settings"]

    document["settings"]["duck_db"] = 7.0
    imported = client.post("/api/settings/import", json=document)
    assert imported.status_code == 200
    assert imported.json()["duck_db"] == 7.0

    document["settings"]["nope"] = 1
    assert client.post("/api/settings/import", json=document).status_code == 400
    assert client.post("/api/settings/import", json={}).status_code == 422


def test_presets_crud(client: TestClient) -> None:
    created = client.post(
        "/api/presets",
        json={
            "kind": "dub",
            "name": "Fast French",
            "payload": {"target_language": "French", "device": "cpu"},
        },
    )
    assert created.status_code == 200, created.text
    preset = created.json()["preset"]
    assert preset["name"] == "Fast French"
    assert preset["payload"]["target_language"] == "French"

    listing = client.get("/api/presets", params={"kind": "dub"})
    assert len(listing.json()["presets"]) == 1
    assert client.get("/api/presets", params={"kind": "say"}).json()["presets"] == []

    assert client.delete(f"/api/presets/{preset['id']}").status_code == 200
    assert client.delete(f"/api/presets/{preset['id']}").status_code == 404

    assert (
        client.post("/api/presets", json={"kind": "dub", "name": "x"}).status_code
        == 422
    )


def test_favorites_crud(client: TestClient) -> None:
    created = client.post(
        "/api/favorites",
        json={"path": "input/talk.mp4", "label": "talk", "kind": "input"},
    )
    assert created.status_code == 200, created.text
    assert len(client.get("/api/favorites").json()["favorites"]) == 1

    assert (
        client.delete("/api/favorites", params={"path": "input/talk.mp4"}).status_code
        == 200
    )
    assert client.get("/api/favorites").json()["favorites"] == []
    assert (
        client.delete("/api/favorites", params={"path": "/nope"}).status_code == 404
    )


def test_downloads_can_be_inline(client: TestClient, tmp_path: Path) -> None:
    target = tmp_path / "output" / "run" / "dub.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"RIFF0000WAVE")

    inline = client.get(
        "/api/outputs/download", params={"path": str(target), "inline": True}
    )
    assert inline.status_code == 200
    assert inline.headers["content-disposition"].startswith("inline")

    attachment = client.get(
        "/api/outputs/download", params={"path": str(target)}
    )
    assert attachment.headers["content-disposition"].startswith("attachment")


def test_outputs_are_one_row_per_run_and_deletable(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "output" / "run-one"
    folder.mkdir(parents=True)
    (folder / "dub.wav").write_bytes(b"wave")
    (folder / "report.json").write_text("{}", encoding="utf-8")
    (folder / "04_dubbed").mkdir()
    (folder / "04_dubbed" / "seg_0000.wav").write_bytes(b"line")

    listing = client.get("/api/outputs")
    assert listing.status_code == 200
    payload = listing.json()
    # One entry for the run, not one per intermediate artifact.
    assert payload["total"] == 1
    run = payload["runs"][0]
    assert run["primary"]["name"] == "dub.wav"
    assert run["file_count"] == 3

    details = client.get("/api/outputs/files", params={"job_dir": str(folder)})
    assert details.status_code == 200
    names = {item["relative"] for item in details.json()["files"]}
    assert names == {"dub.wav", "report.json", "04_dubbed/seg_0000.wav"}

    deleted = client.delete("/api/outputs", params={"path": run["primary"]["path"]})
    assert deleted.status_code == 200
    assert not (folder / "dub.wav").exists()


def test_a_run_with_a_plausible_output_lists_it(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "output" / "odd-run"
    folder.mkdir(parents=True)
    (folder / "speech.wav").write_bytes(b"wave")

    payload = client.get("/api/outputs").json()
    assert payload["total"] == 1
    assert payload["runs"][0]["primary"]["name"] == "speech.wav"


def test_a_run_with_only_internal_artifacts_is_hidden(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "output" / "partial-run"
    folder.mkdir(parents=True)
    (folder / "01_audio.wav").write_bytes(b"wave")
    (folder / "report.json").write_text("{}", encoding="utf-8")

    payload = client.get("/api/outputs").json()
    assert payload["total"] == 0


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


def test_model_delete_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline.models import REGISTRY

    models = tmp_path / "models"
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("DUB_OUTPUT_DIR", str(tmp_path / "output"))
    app = create_app(data_dir=tmp_path / "data", executor=FakeExecutor())
    with TestClient(app) as client:
        # A model that is not present: 404 is wrong here — the registry knows it.
        target = models / REGISTRY["tts_custom_voice"].dir_name
        target.mkdir(parents=True)
        (target / "model.safetensors").write_bytes(b"x" * 8)

        response = client.delete("/api/models/tts_custom_voice")
        assert response.status_code == 200, response.text
        assert response.json()["freed_bytes"] >= 8
        assert not target.exists()

        # Unknown names are rejected rather than touching the filesystem.
        assert client.delete("/api/models/nope").status_code == 404

        usage = client.get("/api/usage")
        assert usage.status_code == 200
        assert set(usage.json()["sizes"]) == {
            "models_dir",
            "voices_dir",
            "output",
            "uploads",
        }


def test_jobs_are_paginated_with_labels_and_filters(client: TestClient) -> None:
    path = _upload(client, "talk.mp4")
    for target in ("French", "German"):
        response = client.post(
            "/api/jobs/dub",
            json={"config": {"audio": path, "target_language": target}},
        )
        assert response.status_code == 200, response.text
        _wait(client, response.json()["job_id"])

    listing = client.get("/api/jobs", params={"page": 1, "page_size": 1})
    payload = listing.json()
    assert payload["page_size"] == 1
    assert len(payload["jobs"]) == 1
    assert payload["total"] == 2
    job = payload["jobs"][0]
    assert job["title"] == "Dub video"
    assert job["input_label"] == "talk.mp4"

    filtered = client.get("/api/jobs", params={"state": "COMPLETED"})
    assert filtered.json()["total"] == 2
    assert client.get("/api/jobs", params={"q": "nope"}).json()["total"] == 0


def test_clear_jobs_leaves_active_work_alone(client: TestClient) -> None:
    path = _upload(client)
    response = client.post(
        "/api/jobs/dub",
        json={"config": {"audio": path, "target_language": "French"}},
    )
    job_id = response.json()["job_id"]
    _wait(client, job_id)

    cleared = client.post("/api/jobs/clear", json={"ids": [job_id]})
    assert cleared.status_code == 200
    assert cleared.json()["removed"] == 1
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_retry_creates_a_new_job_in_a_fresh_folder(client: TestClient) -> None:
    path = _upload(client, "talk.mp4")
    first_id = client.post(
        "/api/jobs/dub",
        json={"config": {"audio": path, "target_language": "French"}},
    ).json()["job_id"]
    first = _wait(client, first_id)

    response = client.post(f"/api/jobs/{first_id}/retry")
    assert response.status_code == 200, response.text
    second_id = response.json()["job_id"]
    assert second_id != first_id

    second = _wait(client, second_id)
    assert second["kind"] == "dub"
    assert second["job_dir"] != first["job_dir"]
    # The original remains untouched and queryable.
    assert client.get(f"/api/jobs/{first_id}").status_code == 200


def test_retry_rejects_a_missing_input(client: TestClient) -> None:
    path = _upload(client)
    job_id = client.post(
        "/api/jobs/dub",
        json={"config": {"audio": path, "target_language": "French"}},
    ).json()["job_id"]
    _wait(client, job_id)
    Path(path).unlink()

    response = client.post(f"/api/jobs/{job_id}/retry")
    assert response.status_code == 400
    assert "Input not found" in response.json()["detail"]


def test_retry_unknown_job_is_404(client: TestClient) -> None:
    assert client.post("/api/jobs/nope/retry").status_code == 404


def test_job_input_streams_the_original_file(client: TestClient) -> None:
    path = _upload(client, "talk.mp4")
    job_id = client.post(
        "/api/jobs/dub",
        json={"config": {"audio": path, "target_language": "French"}},
    ).json()["job_id"]
    _wait(client, job_id)

    response = client.get(f"/api/jobs/{job_id}/input")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("inline")
    assert client.get("/api/jobs/nope/input").status_code == 404
