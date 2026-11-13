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
from pipeline.capabilities import capabilities_payload, say_prosody_conflict
from pipeline.config import (
    SUPPORTED_LANGUAGES,
    SubtitleJobConfig,
    normalize_device_request,
)
from pipeline.errors import ConfigError, DubbingError, ModelNotFoundError
from pipeline.info import devices_report, environment_report, usage_report
from pipeline.models import (
    MVP_REQUIRED,
    REGISTRY,
    anchors_dir,
    registry_status,
    remove_model,
)
from pipeline.voice_ops import list_voice_summaries, remove_voice

from .jobs import JobManager, QueueFullError
from .schemas import (
    AnchorGenerateRequest,
    DubRequest,
    FavoriteRequest,
    IdListRequest,
    JobSubmitResponse,
    ModelDownloadRequest,
    PathListRequest,
    PresetRequest,
    SayRequest,
    SettingsImport,
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
_UI_FIELDS = {"default_device", "default_dtype", "theme"}
_DTYPES = {"auto", "bf16", "fp16", "fp32"}
_THEMES = {"system", "light", "dark"}


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

    @app.get("/api/backends")
    def backends() -> dict[str, list[dict[str, object]]]:
        """What each backend can honour, so the forms can disable the rest."""
        return capabilities_payload()

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

    @app.delete("/api/models/{name}")
    def delete_model(request: Request, name: str) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        if name not in REGISTRY:
            raise HTTPException(status_code=404, detail=f"Unknown model {name!r}")
        try:
            return remove_model(name, settings.models_dir)
        except (ConfigError, ModelNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    # -- usage ------------------------------------------------------------- #

    @app.get("/api/usage")
    def usage(request: Request) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        app_state = state(request)
        return usage_report(
            models_dir=settings.models_dir,
            voices_dir=settings.voices_dir,
            output_root=app_state.output_root,
            uploads_dir=app_state.uploads_dir,
        )

    # -- voices ------------------------------------------------------------ #

    @app.get("/api/voices")
    def voices(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        settings = effective_settings(state(request).store)
        root = Path(settings.voices_dir)
        all_voices = list_voice_summaries(root)
        total = len(all_voices)
        offset = (page - 1) * page_size
        return {
            "voices_dir": str(root.resolve()),
            "voices": all_voices[offset : offset + page_size],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

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

    @app.get("/api/voices/{label}/audio")
    def voice_audio(request: Request, label: str) -> FileResponse:
        settings = effective_settings(state(request).store)
        root = Path(settings.voices_dir).expanduser().resolve()
        path = (root / label / "voice.wav").resolve()
        if not _is_under(path, root) or not path.is_file():
            raise HTTPException(status_code=404, detail="Voice audio not found")
        return FileResponse(
            path, media_type="audio/wav", content_disposition_type="inline"
        )

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
        # Reject an impossible backend/option pair here instead of enqueuing a
        # job that is guaranteed to fail in the worker.
        conflict = say_prosody_conflict(
            body.backend,
            prosody=body.prosody,
            emotions=body.emotions,
            prosody_plan=body.prosody_plan,
        )
        if conflict:
            raise HTTPException(status_code=422, detail=conflict)
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
    def list_jobs(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
        state: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        # ``state`` here is the query parameter, so reach the app state directly
        # rather than through the same-named helper above.
        store_ref = request.app.state.store
        offset = (page - 1) * page_size
        return {
            "jobs": store_ref.list_jobs(
                page_size, offset=offset, state=state, q=q
            ),
            "total": store_ref.count_jobs(state=state, q=q),
            "page": page,
            "page_size": page_size,
        }

    @app.post("/api/jobs/clear")
    def clear_jobs(request: Request, body: IdListRequest) -> dict[str, Any]:
        removed = state(request).store.delete_jobs(body.ids)
        return {"removed": removed}

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

    @app.post("/api/jobs/{job_id}/retry", response_model=JobSubmitResponse)
    def retry_job(request: Request, job_id: str) -> JobSubmitResponse:
        """Re-run a finished job's stored configuration as a new job.

        The original row and folder are never touched: a retry gets its own run
        directory, so history stays immutable and nothing is overwritten. The
        original inputs are re-validated, because an upload may have been deleted
        since the first attempt.
        """
        row = job_or_404(request, job_id)
        kind = str(row["kind"])
        raw = row.get("params") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        payload: dict[str, Any] = json.loads(json.dumps(raw)) if isinstance(raw, dict) else {}
        folder_tokens = {
            "voice_design": "voice-design",
            "voice_clone": "voice-clone",
            "voice_add": "voice-add",
        }

        if kind in {"dub", "subtitles"}:
            config = payload.setdefault("config", {})
            source = config.get("video") or config.get("audio") or row.get("input_path")
            if not source:
                raise HTTPException(
                    status_code=400, detail="This job has no input to retry."
                )
            source = validate_input(request, source)
            if config.get("ref_audio"):
                config["ref_audio"] = str(validate_input(request, config["ref_audio"]))
            if config.get("accent_audio"):
                config["accent_audio"] = str(
                    validate_input(request, config["accent_audio"])
                )
            folder = run_dir(
                request, source.stem, kind, str(config.get("target_language") or "")
            )
            config["job_dir_override"] = str(folder)
            new_id = state(request).manager.enqueue(
                kind, payload, input_path=str(source), job_dir=str(folder)
            )
        elif kind == "say":
            spec = payload.setdefault("spec", {})
            if spec.get("clone"):
                spec["clone"] = str(validate_input(request, spec["clone"]))
            folder = run_dir(
                request,
                str(spec.get("voice") or spec.get("save_as") or "say"),
                "say",
                str(spec.get("backend") or "qwen"),
            )
            spec["output_root"] = str(folder)
            spec["output"] = None
            new_id = state(request).manager.enqueue(
                "say", payload, job_dir=str(folder)
            )
        elif kind in folder_tokens:
            params = payload.setdefault("params", {})
            if params.get("audio"):
                params["audio"] = str(validate_input(request, params["audio"]))
            folder = run_dir(
                request, str(params.get("label") or "voice"), folder_tokens[kind]
            )
            if kind in {"voice_design", "voice_clone"}:
                params["output_root"] = str(folder)
                params["output"] = None
            new_id = state(request).manager.enqueue(
                kind, payload, job_dir=str(folder)
            )
        elif kind == "model_download":
            new_id = state(request).manager.enqueue(
                "model_download",
                {
                    "names": payload.get("names") or [],
                    "force": bool(payload.get("force")),
                },
                job_dir=str(state(request).data_dir / "downloads"),
            )
        elif kind == "anchor_generate":
            folder = run_dir(request, "anchors", "anchors")
            new_id = state(request).manager.enqueue(
                "anchor_generate",
                {
                    "languages": payload.get("languages") or [],
                    "device": payload.get("device") or "auto",
                    "dtype": payload.get("dtype") or "auto",
                    "force": bool(payload.get("force")),
                },
                job_dir=str(folder),
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Cannot retry job kind {kind!r}"
            )

        return JobSubmitResponse(job_id=new_id, state="QUEUED")

    @app.get("/api/jobs/{job_id}/input")
    def job_input(request: Request, job_id: str) -> FileResponse:
        """Stream the file a job operated on, for the compare view."""
        row = job_or_404(request, job_id)
        raw = row.get("input_path")
        if not raw:
            raise HTTPException(
                status_code=404, detail="This job has no input file"
            )
        path = validate_input(request, raw)
        return FileResponse(
            path,
            filename=path.name,
            media_type=mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
            content_disposition_type="inline",
        )

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
    def files(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        store_ref = state(request).store
        offset = (page - 1) * page_size
        return {
            "files": store_ref.list_files(page_size, offset=offset),
            "total": store_ref.count_files(),
            "page": page,
            "page_size": page_size,
        }

    @app.post("/api/files/delete")
    def delete_files(request: Request, body: IdListRequest) -> dict[str, Any]:
        roots = [
            state(request).uploads_dir,
            state(request).output_root,
            state(request).data_dir,
        ]
        removed: list[str] = []
        for item in state(request).store.delete_files(body.ids):
            path = Path(item["path"])
            if any(_is_under(path, root) for root in roots) and path.is_file():
                path.unlink()
            removed.append(item["id"])
        return {"removed": removed}

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

    #: Filenames the pipeline treats as a run's main deliverable. These are the
    #: legacy generic names, kept so older folders still resolve their primary.
    _PRIMARY_NAMES = (
        "video_dubbed.mp4",
        "video_burned.mp4",
        "video_subtitled.mp4",
        "dubbed_video.mp4",
        "dub.wav",
        "subs_target.srt",
    )
    #: Current runs name their deliverables after the input (``talk_dubbed.mp4``),
    #: so match by suffix when the exact legacy name is not present.
    _PRIMARY_SUFFIXES = (
        "_dubbed.mp4",
        "_burned.mp4",
        "_subtitled.mp4",
        "_dubbed.wav",
    )
    _MEDIA_SUFFIXES = {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".mp4",
        ".mkv",
        ".mov",
        ".webm",
        ".srt",
    }

    def _primary_from_result(result: dict[str, Any]) -> Path | None:
        """The main artifact a job reported, in preference order."""
        for key in (
            "dubbed_video",
            "burned_video",
            "output_video",
            "dub_track",
            "target_srt",
            "output",
        ):
            value = result.get(key)
            if isinstance(value, str) and value:
                candidate = Path(value)
                if candidate.is_file():
                    return candidate
        return None

    def _visible_files(folder: Path) -> list[Path]:
        return [
            path
            for path in folder.rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") for part in path.relative_to(folder).parts
            )
        ]

    def _pick_primary(
        folder: Path, files: list[Path], known: Path | None
    ) -> Path | None:
        """Choose the one file a run should show in the output list.

        A finished job's own ``result`` is authoritative. Older or orphaned run
        folders fall back to the pipeline's well-known artifact names, then to the
        largest top-level media file. Intermediate artifacts (``01_audio.wav``,
        ``04_dubbed/…``) are never promoted: a run that produced no deliverable is
        hidden rather than listed with a stage file.
        """
        if known is not None and known.is_file():
            return known

        def internal(name: str) -> bool:
            return bool(re.match(r"^\d+[a-z]?_", name))

        top = [
            path
            for path in files
            if path.parent == folder and not internal(path.name)
        ]
        by_name = {path.name: path for path in top}
        for name in _PRIMARY_NAMES:
            if name in by_name:
                return by_name[name]
        for suffix in _PRIMARY_SUFFIXES:
            for name, path in by_name.items():
                if name.endswith(suffix):
                    return path
        media = sorted(
            (
                path
                for path in top
                if path.suffix.lower() in _MEDIA_SUFFIXES
            ),
            key=lambda path: path.stat().st_size,
            reverse=True,
        )
        return media[0] if media else None

    @app.get("/api/outputs")
    def outputs(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        """List one entry per run, with its main artifact.

        Scanning every file made the list unusable: a single dub writes dozens of
        intermediate artifacts. Each run therefore reports its primary output and
        a file count; the full list is available from ``/api/outputs/files``.
        """
        root = state(request).output_root
        store_ref = state(request).store
        known: dict[str, Path] = {}
        meta: dict[str, dict[str, Any]] = {}
        for row in store_ref.list_jobs(limit=5000):
            job_dir = row.get("job_dir")
            if not job_dir:
                continue
            key = str(Path(job_dir).resolve())
            meta.setdefault(key, row)
            result = row.get("result") or {}
            if isinstance(result, dict):
                primary = _primary_from_result(result)
                if primary is not None:
                    known[key] = primary

        runs: list[dict[str, Any]] = []
        if root.is_dir():
            for entry in root.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    folder, files = entry, _visible_files(entry)
                elif entry.is_file():
                    folder, files = entry, [entry]
                else:
                    continue
                if not files:
                    continue
                key = str(folder.resolve())
                primary = _pick_primary(folder, files, known.get(key))
                if primary is None:
                    continue
                try:
                    stat = primary.stat()
                except OSError:
                    continue
                row = meta.get(key) or {}
                runs.append(
                    {
                        "job_dir": str(folder),
                        "job_id": row.get("id"),
                        "title": row.get("title") or folder.name,
                        "input_label": row.get("input_label"),
                        "created": stat.st_mtime,
                        "primary": {
                            "path": str(primary),
                            "name": primary.name,
                            "size": stat.st_size,
                            "created": stat.st_mtime,
                        },
                        "file_count": len(files),
                    }
                )
        runs.sort(key=lambda item: item["created"], reverse=True)
        total = len(runs)
        offset = (page - 1) * page_size
        return {
            "output_root": str(root),
            "runs": runs[offset : offset + page_size],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    @app.get("/api/outputs/files")
    def output_files(request: Request, job_dir: str) -> dict[str, Any]:
        """Every artifact in one run folder, for the details view."""
        folder = validate_output(request, job_dir)
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail="Run folder not found")
        files: list[dict[str, Any]] = []
        for path in _visible_files(folder):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "path": str(path),
                    "relative": str(path.relative_to(folder)),
                    "name": path.name,
                    "size": stat.st_size,
                    "created": stat.st_mtime,
                }
            )
        files.sort(key=lambda item: item["relative"])
        return {"job_dir": str(folder), "files": files}

    @app.post("/api/outputs/delete")
    def delete_outputs(request: Request, body: PathListRequest) -> dict[str, Any]:
        removed: list[str] = []
        for value in body.paths:
            target = validate_output(request, value)
            if target.is_file():
                target.unlink()
                removed.append(str(target))
        return {"removed": removed}

    @app.get("/api/outputs/download")
    def download_output(
        request: Request, path: str, inline: bool = False
    ) -> FileResponse:
        target = validate_output(request, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Output not found")
        return FileResponse(
            target,
            filename=target.name,
            media_type=mimetypes.guess_type(target.name)[0]
            or "application/octet-stream",
            content_disposition_type="inline" if inline else "attachment",
        )

    @app.delete("/api/outputs")
    def delete_output(request: Request, path: str) -> dict[str, Any]:
        target = validate_output(request, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Output not found")
        target.unlink()
        return {"deleted": str(target)}

    @app.get("/api/files/{file_id}/download")
    def download(request: Request, file_id: str, inline: bool = False) -> FileResponse:
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
            content_disposition_type="inline" if inline else "attachment",
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

    def apply_settings(request: Request, values: dict[str, Any]) -> None:
        """Validate a settings patch and persist it, or reject it wholesale."""
        unknown = sorted(set(values) - set(EDITABLE_FIELDS) - _UI_FIELDS)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Settings not editable: {', '.join(unknown)}",
            )
        pipeline_values = {
            key: value for key, value in values.items() if key not in _UI_FIELDS
        }
        current = effective_settings(state(request).store).model_dump()
        try:
            from pipeline.config import Settings

            Settings.model_validate({**current, **pipeline_values})
            if "default_device" in values:
                normalize_device_request(str(values["default_device"]))
            if "default_dtype" in values and values["default_dtype"] not in _DTYPES:
                raise ValueError("default_dtype must be auto, bf16, fp16 or fp32")
            if "theme" in values and values["theme"] not in _THEMES:
                raise ValueError("theme must be system, light or dark")
        except (ValidationError, ConfigError, ValueError) as exc:
            message = (
                exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            )
            raise HTTPException(
                status_code=400, detail=f"Invalid setting: {message}"
            ) from None
        state(request).store.patch_settings(values)

    @app.get("/api/settings")
    def get_settings(request: Request) -> dict[str, Any]:
        payload = settings_payload(effective_settings(state(request).store))
        stored = state(request).store.get_settings()
        payload["default_device"] = stored.get("default_device", "auto")
        payload["default_dtype"] = stored.get("default_dtype", "auto")
        payload["theme"] = stored.get("theme", "system")
        return payload

    @app.patch("/api/settings")
    def patch_settings(request: Request, body: SettingsPatch) -> dict[str, Any]:
        apply_settings(request, body.values)
        return get_settings(request)

    @app.get("/api/settings/export")
    def export_settings(request: Request) -> dict[str, Any]:
        payload = get_settings(request)
        keys = [*EDITABLE_FIELDS, *_UI_FIELDS]
        return {
            "version": 1,
            "settings": {key: payload[key] for key in keys if key in payload},
        }

    @app.post("/api/settings/import")
    def import_settings(request: Request, body: SettingsImport) -> dict[str, Any]:
        apply_settings(request, body.settings)
        return get_settings(request)

    # -- presets and favorites --------------------------------------------- #

    @app.get("/api/presets")
    def presets(request: Request, kind: str | None = None) -> dict[str, Any]:
        return {"presets": state(request).store.list_presets(kind)}

    @app.post("/api/presets")
    def create_preset(request: Request, body: PresetRequest) -> dict[str, Any]:
        preset = state(request).store.add_preset(
            uuid.uuid4().hex, body.kind, body.name, body.payload
        )
        return {"preset": preset}

    @app.delete("/api/presets/{preset_id}")
    def remove_preset(request: Request, preset_id: str) -> dict[str, Any]:
        if not state(request).store.delete_preset(preset_id):
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"deleted": preset_id}

    @app.get("/api/favorites")
    def favorites(request: Request) -> dict[str, Any]:
        return {"favorites": state(request).store.list_favorites()}

    @app.post("/api/favorites")
    def add_favorite(request: Request, body: FavoriteRequest) -> dict[str, Any]:
        item = state(request).store.add_favorite(body.path, body.label, body.kind)
        return {"favorite": item}

    @app.delete("/api/favorites")
    def remove_favorite(request: Request, path: str) -> dict[str, Any]:
        if not state(request).store.delete_favorite(path):
            raise HTTPException(status_code=404, detail="Favorite not found")
        return {"deleted": path}

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

