"""Local FastAPI application for the Dubsmith web UI.

The API is a thin adapter over ``pipeline/``: it validates inputs, persists jobs,
and streams their events. It never reimplements dubbing, TTS or translation.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from pipeline.anchors import list_anchors
from pipeline.config import (
    SUPPORTED_LANGUAGES,
    SubtitleJobConfig,
    normalize_device_request,
)
from pipeline.errors import ConfigError, DubbingError
from pipeline.info import devices_report, environment_report
from pipeline.models import (
    MVP_REQUIRED,
    REGISTRY,
    anchors_dir,
    registry_status,
)
from pipeline.voice_ops import list_voice_summaries, remove_voice

from .jobs import JobManager, QueueFullError
from .schemas import (
    AnchorGenerateRequest,
    DubRequest,
    JobSubmitResponse,
    ModelDownloadRequest,
    SayRequest,
    SettingsPatch,
    VoiceAddRequest,
    VoiceCloneRequest,
    VoiceDesignRequest,
)
from .settings import EDITABLE_FIELDS, effective_settings, settings_payload
from .store import Store

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
_ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".m4v",
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
}
_MAX_UPLOAD_BYTES = 8 * 1024**3

#: UI-only preferences that are not part of pipeline ``Settings``.
_UI_FIELDS = {"default_device", "default_dtype"}
_DTYPES = {"auto", "bf16", "fp16", "fp32"}


def _slug(value: str, limit: int = 60) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return (cleaned or "job")[:limit]


def _safe_name(value: str) -> str:
    return _slug(Path(value).name, limit=120)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def create_app(
    *,
    data_dir: Path | str | None = None,
    executor: Any | None = None,
) -> FastAPI:
    data = Path(data_dir or os.environ.get("DUB_DATA_DIR", "data")).expanduser()
    data = data.resolve()
    uploads = data / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    output_root = Path(os.environ.get("DUB_OUTPUT_DIR", "output")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = Path("input").expanduser().resolve()

    store = Store(data / "dubsmith.db")
    manager = JobManager(
        store,
        settings_provider=lambda: effective_settings(store).model_dump(mode="json"),
        executor=executor,
    )

    app = FastAPI(
        title="Dubsmith",
        version="0.2.0",
        description="Local dubbing and voice generation API.",
    )
    app.state.store = store
    app.state.manager = manager
    app.state.data_dir = data
    app.state.uploads_dir = uploads
    app.state.output_root = output_root
    app.state.input_root = input_root

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        manager.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        manager.stop()
        store.close()

    # -- helpers ----------------------------------------------------------- #

    def state(request: Request) -> Any:
        return request.app.state

    def allowed_input_roots(request: Request) -> list[Path]:
        app_state = state(request)
        return [
            app_state.uploads_dir,
            app_state.output_root,
            app_state.input_root,
            app_state.data_dir,
        ]

    def validate_input(request: Request, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"Input not found: {value}")
        if not any(_is_under(path, root) for root in allowed_input_roots(request)):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Only files uploaded through the app (or under input/output) "
                    "can be used as input."
                ),
            )
        return path

    def validate_output(request: Request, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if not _is_under(path, state(request).output_root):
            raise HTTPException(
                status_code=400,
                detail="Output paths must stay inside the app's output directory.",
            )
        return path

    def run_dir(request: Request, stem: str, kind: str, language: str = "") -> Path:
        token = uuid.uuid4().hex[:8]
        parts = [_slug(stem), kind]
        if language:
            parts.append(_slug(language))
        parts.append(token)
        folder = state(request).output_root / "__".join(parts)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def job_or_404(request: Request, job_id: str) -> dict[str, Any]:
        row = state(request).store.get_job(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return row

    # -- errors ------------------------------------------------------------ #

    @app.exception_handler(DubbingError)
    async def _domain_error(_request: Request, exc: DubbingError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(QueueFullError)
    async def _queue_full(_request: Request, exc: QueueFullError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    # -- health / environment --------------------------------------------- #

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "time": time.time()}

    @app.get("/api/devices")
    def devices() -> dict[str, Any]:
        return devices_report()

    @app.get("/api/info")
    def info(
        request: Request,
        device: str = "auto",
        dtype: str = "auto",
        devices: str | None = None,
    ) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        fanout = [part.strip() for part in devices.split(",") if part.strip()] if devices else None
        return environment_report(
            settings.models_dir,
            device=device,
            dtype=dtype,
            devices=fanout,
        )

    @app.get("/api/packages")
    def packages(request: Request) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        report = environment_report(settings.models_dir)
        return {
            "optional_packages": report["optional_packages"],
            "system_binaries": report["system_binaries"],
        }

    # -- models ------------------------------------------------------------ #

    @app.get("/api/models")
    def models(request: Request) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        return {
            "models_dir": str(Path(settings.models_dir).resolve()),
            "models": [
                {
                    "name": spec.name,
                    "repo_id": spec.repo_id,
                    "description": spec.description,
                    "license": spec.license,
                    "required": spec.name in MVP_REQUIRED,
                    "present": present,
                }
                for spec, present in registry_status(settings.models_dir)
            ],
        }

    @app.post("/api/models/download", response_model=JobSubmitResponse)
    def download_models(
        request: Request, body: ModelDownloadRequest
    ) -> JobSubmitResponse:
        names: list[str] = []
        for name in body.names:
            key = name.strip().lower()
            if key in {"all", "mvp"}:
                names.extend(MVP_REQUIRED)
            elif key == "everything":
                names.extend(REGISTRY)
            elif key in REGISTRY:
                names.append(key)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unknown model {name!r}"
                )
        names = list(dict.fromkeys(names))
        if not names:
            raise HTTPException(status_code=400, detail="No models requested")
        job_id = state(request).manager.enqueue(
            "model_download",
            {"names": names, "force": body.force},
            job_dir=str(state(request).data_dir / "downloads"),
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    # -- voices ------------------------------------------------------------ #

    @app.get("/api/voices")
    def voices(request: Request) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        root = Path(settings.voices_dir)
        return {"voices_dir": str(root.resolve()), "voices": list_voice_summaries(root)}

    @app.post("/api/voices/design", response_model=JobSubmitResponse)
    def voice_design(
        request: Request, body: VoiceDesignRequest
    ) -> JobSubmitResponse:
        folder = run_dir(request, body.label or "voice-design", "voice-design")
        params = body.model_dump()
        params["output_root"] = str(folder)
        if params.get("output"):
            params["output"] = str(validate_output(request, params["output"]))
        job_id = state(request).manager.enqueue(
            "voice_design", {"params": params}, job_dir=str(folder)
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    @app.post("/api/voices/clone", response_model=JobSubmitResponse)
    def voice_clone(request: Request, body: VoiceCloneRequest) -> JobSubmitResponse:
        audio = validate_input(request, body.audio)
        folder = run_dir(request, body.label or audio.stem, "voice-clone")
        params = body.model_dump()
        params["audio"] = str(audio)
        params["output_root"] = str(folder)
        if params.get("output"):
            params["output"] = str(validate_output(request, params["output"]))
        job_id = state(request).manager.enqueue(
            "voice_clone", {"params": params}, job_dir=str(folder)
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    @app.post("/api/voices/add", response_model=JobSubmitResponse)
    def voice_add(request: Request, body: VoiceAddRequest) -> JobSubmitResponse:
        audio = validate_input(request, body.audio)
        folder = run_dir(request, body.label or audio.stem, "voice-add")
        params = body.model_dump()
        params["audio"] = str(audio)
        job_id = state(request).manager.enqueue(
            "voice_add", {"params": params}, job_dir=str(folder)
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    @app.delete("/api/voices/{label}")
    def voice_delete(request: Request, label: str) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        removed = remove_voice(Path(settings.voices_dir), label)
        return {"removed": label, "path": str(removed)}

    # -- anchors ----------------------------------------------------------- #

    @app.get("/api/anchors")
    def anchors(request: Request) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        root = anchors_dir(settings.models_dir)
        return {
            "anchors_dir": str(root.resolve()),
            "anchors": [
                {"key": key, "path": str(path), "seconds": seconds}
                for key, path, seconds in list_anchors(root)
            ],
        }

    @app.post("/api/anchors/generate", response_model=JobSubmitResponse)
    def anchor_generate(
        request: Request, body: AnchorGenerateRequest
    ) -> JobSubmitResponse:
        languages = body.languages or [
            name for name in sorted(SUPPORTED_LANGUAGES) if name != "Auto"
        ]
        folder = run_dir(request, "anchors", "anchors")
        job_id = state(request).manager.enqueue(
            "anchor_generate",
            {
                "languages": languages,
                "device": body.device,
                "dtype": body.dtype,
                "force": body.force,
            },
            job_dir=str(folder),
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    # -- jobs -------------------------------------------------------------- #

    @app.post("/api/jobs/dub", response_model=JobSubmitResponse)
    def submit_dub(request: Request, body: DubRequest) -> JobSubmitResponse:
        from pipeline.job_ops import apply_accent_voice, apply_saved_voice

        job = body.config
        source = job.input_path
        validate_input(request, source)
        if job.ref_audio is not None:
            validate_input(request, job.ref_audio)
        if job.accent_audio is not None:
            validate_input(request, job.accent_audio)
        settings = effective_settings(state(request).store)
        if body.voice:
            job = apply_saved_voice(
                job,
                body.voice,
                voices_root=Path(settings.voices_dir),
                models_dir=Path(settings.models_dir),
            )
        if body.accent_voice:
            job = apply_accent_voice(
                job, body.accent_voice, voices_root=Path(settings.voices_dir)
            )
        folder = run_dir(request, source.stem, "dub", job.target_language)
        job = job.model_copy(update={"job_dir_override": folder})
        job_id = state(request).manager.enqueue(
            "dub",
            {"config": job.model_dump(mode="json")},
            input_path=str(source),
            job_dir=str(folder),
            extra={"stretcher": body.stretcher},
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    @app.post("/api/jobs/subtitles", response_model=JobSubmitResponse)
    def submit_subtitles(
        request: Request, body: SubtitleJobConfig
    ) -> JobSubmitResponse:
        source = body.input_path
        validate_input(request, source)
        folder = run_dir(
            request, source.stem, "subtitles", body.target_language
        )
        job = body.model_copy(update={"job_dir_override": folder})
        job_id = state(request).manager.enqueue(
            "subtitles",
            {"config": job.model_dump(mode="json")},
            input_path=str(source),
            job_dir=str(folder),
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    @app.post("/api/jobs/say", response_model=JobSubmitResponse)
    def submit_say(request: Request, body: SayRequest) -> JobSubmitResponse:
        raw = body.model_dump()
        if raw.get("clone"):
            raw["clone"] = str(validate_input(request, raw["clone"]))
        if raw.get("output"):
            validate_output(request, raw["output"])
        folder = run_dir(
            request,
            raw.get("voice") or raw.get("save_as") or "say",
            "say",
            raw.get("backend", "qwen"),
        )
        raw["output_root"] = str(folder)
        raw["output"] = None
        job_id = state(request).manager.enqueue(
            "say", {"spec": raw}, job_dir=str(folder)
        )
        return JobSubmitResponse(job_id=job_id, state="QUEUED")

    @app.get("/api/jobs")
    def list_jobs(request: Request, limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
        return {"jobs": state(request).store.list_jobs(limit)}

    @app.get("/api/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict[str, Any]:
        row = job_or_404(request, job_id)
        row["events"] = state(request).store.tail_events(job_id, limit=100)
        return row

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
        job_or_404(request, job_id)
        row = state(request).manager.cancel(job_id)
        return row or {}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(
        request: Request, job_id: str, after: int = 0
    ) -> StreamingResponse:
        job_or_404(request, job_id)
        store_ref = state(request).store

        async def stream() -> AsyncIterator[str]:
            last = max(0, int(after))
            while True:
                events = store_ref.events_after(job_id, after_id=last)
                for event in events:
                    last = int(event["id"])
                    payload = json.dumps(event, default=str)
                    yield f"event: {event['kind']}\ndata: {payload}\n\n"
                row = store_ref.get_job(job_id)
                if row and row["state"] in _TERMINAL:
                    yield (
                        "event: job\ndata: "
                        + json.dumps(row, default=str)
                        + "\n\n"
                    )
                    break
                if await request.is_disconnected():
                    break
                yield ": keepalive\n\n"
                await asyncio.sleep(0.6)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -- files ------------------------------------------------------------- #

    @app.get("/api/files")
    def files(request: Request) -> dict[str, Any]:
        return {"files": state(request).store.list_files()}

    @app.post("/api/files/upload")
    def upload(
        request: Request, file: Annotated[UploadFile, File()]
    ) -> dict[str, Any]:
        original = _safe_name(file.filename or "upload")
        suffix = Path(original).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type {suffix or '(none)'}.",
            )
        file_id = uuid.uuid4().hex
        stored_name = f"{file_id}__{original}"
        target = state(request).uploads_dir / stored_name
        size = 0
        with target.open("wb") as handle:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413, detail="File is too large."
                    )
                handle.write(chunk)
        state(request).store.add_file(
            file_id,
            original,
            stored_name,
            str(target),
            size,
            "input",
        )
        return {
            "id": file_id,
            "name": original,
            "path": str(target),
            "size": size,
        }

    @app.get("/api/outputs")
    def outputs(request: Request) -> dict[str, Any]:
        root = state(request).output_root
        files: list[dict[str, Any]] = []
        if root.is_dir():
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if any(part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files.append(
                    {
                        "path": str(path),
                        "relative": str(path.relative_to(root)),
                        "name": path.name,
                        "size": stat.st_size,
                        "created": stat.st_mtime,
                    }
                )
        files.sort(key=lambda item: item["created"], reverse=True)
        return {"output_root": str(root), "files": files[:1000]}

    @app.get("/api/outputs/download")
    def download_output(request: Request, path: str) -> FileResponse:
        target = validate_output(request, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Output not found")
        return FileResponse(
            target,
            filename=target.name,
            media_type=mimetypes.guess_type(target.name)[0]
            or "application/octet-stream",
        )

    @app.delete("/api/outputs")
    def delete_output(request: Request, path: str) -> dict[str, Any]:
        target = validate_output(request, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Output not found")
        target.unlink()
        return {"deleted": str(target)}

    @app.get("/api/files/{file_id}/download")
    def download(request: Request, file_id: str) -> FileResponse:
        item = state(request).store.get_file(file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="File not found")
        path = Path(item["path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File is missing on disk")
        return FileResponse(
            path,
            filename=item["original_name"],
            media_type=mimetypes.guess_type(item["original_name"])[0]
            or "application/octet-stream",
        )

    @app.delete("/api/files/{file_id}")
    def delete_file(request: Request, file_id: str) -> dict[str, Any]:
        item = state(request).store.get_file(file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="File not found")
        path = Path(item["path"])
        roots = [
            state(request).uploads_dir,
            state(request).output_root,
            state(request).data_dir,
        ]
        if not any(_is_under(path, root) for root in roots):
            raise HTTPException(status_code=400, detail="Refusing to delete that path.")
        state(request).store.delete_file(file_id)
        if path.is_file():
            path.unlink()
        return {"deleted": file_id}

    # -- settings ---------------------------------------------------------- #

    @app.get("/api/settings")
    def get_settings(request: Request) -> dict[str, Any]:
        payload = settings_payload(effective_settings(state(request).store))
        stored = state(request).store.get_settings()
        payload["default_device"] = stored.get("default_device", "auto")
        payload["default_dtype"] = stored.get("default_dtype", "auto")
        return payload

    @app.patch("/api/settings")
    def patch_settings(request: Request, body: SettingsPatch) -> dict[str, Any]:
        unknown = sorted(set(body.values) - set(EDITABLE_FIELDS) - _UI_FIELDS)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Settings not editable: {', '.join(unknown)}",
            )
        pipeline_values = {
            key: value for key, value in body.values.items() if key not in _UI_FIELDS
        }
        current = effective_settings(state(request).store).model_dump()
        try:
            from pipeline.config import Settings

            Settings.model_validate({**current, **pipeline_values})
            if "default_device" in body.values:
                normalize_device_request(str(body.values["default_device"]))
            if (
                "default_dtype" in body.values
                and body.values["default_dtype"] not in _DTYPES
            ):
                raise ValueError("default_dtype must be auto, bf16, fp16 or fp32")
        except (ValidationError, ConfigError, ValueError) as exc:
            message = (
                exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            )
            raise HTTPException(
                status_code=400, detail=f"Invalid setting: {message}"
            ) from None
        state(request).store.patch_settings(body.values)
        return get_settings(request)

    # -- static frontend --------------------------------------------------- #

    candidates = [
        Path.cwd() / "web" / "dist",
        Path(__file__).resolve().parent.parent / "web" / "dist",
    ]
    web_dist = next(
        (path for path in candidates if (path / "index.html").is_file()),
        candidates[-1],
    )
    if (web_dist / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=web_dist / "assets"),
            name="assets",
        )

    @app.get("/{full_path:path}")
    def spa(request: Request, full_path: str) -> Any:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if web_dist.is_dir():
            candidate = web_dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            index = web_dist / "index.html"
            if index.is_file():
                return FileResponse(index)
        raise HTTPException(
            status_code=404,
            detail="The web UI is not built. Run 'npm run build' in web/.",
        )

    return app

