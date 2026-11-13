"""Background worker entry point.

Each job runs in its own spawned process so a model-heavy run never blocks the API
process and its GPU memory is returned when the process exits. The worker reports
structured stage events and log records through a queue; the API persists them and
streams them to the browser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipeline.config import JobConfig, Settings, SubtitleJobConfig
from pipeline.errors import DubbingError
from pipeline.logging_utils import quiet_dependencies
from pipeline.models import anchors_dir, download_many

__all__ = ["run_job"]


class _QueueLogHandler(logging.Handler):
    """Forward records as plain dictionaries the API can persist and stream."""

    def __init__(self, queue: Any) -> None:
        super().__init__()
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put(
                {
                    "type": "log",
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "stage": getattr(record, "stage", None),
                    "time": record.created,
                }
            )
        except Exception:  # noqa: BLE001 - logging must never break a job
            pass


def _settings(payload: dict[str, Any]) -> Settings:
    raw = payload.get("settings")
    if raw:
        return Settings.model_validate(raw)
    return Settings.from_env()


def _dispatch(payload: dict[str, Any], emit: Any) -> dict[str, Any]:
    kind = payload["kind"]
    settings = _settings(payload)
    models_dir = Path(payload.get("models_dir") or settings.models_dir)
    voices_root = Path(payload.get("voices_root") or settings.voices_dir)

    if kind == "dub":
        from pipeline.orchestrate import Pipeline

        job = JobConfig(**payload["config"])
        result = Pipeline(
            job,
            settings,
            event=emit,
            stretcher_preference=payload.get("stretcher", "auto"),
        ).run()
        return {
            "job_dir": str(result.job_dir),
            "dub_track": str(result.dub_track),
            "dubbed_video": str(result.dubbed_video) if result.dubbed_video else None,
            "burned_video": str(result.burned_video) if result.burned_video else None,
            "source_srt": str(result.source_srt) if result.source_srt else None,
            "target_srt": str(result.target_srt) if result.target_srt else None,
            "report": result.report,
            "stage_seconds": result.stage_seconds,
        }

    if kind == "subtitles":
        from pipeline.subtitle_pipeline import SubtitlesPipeline

        job = SubtitleJobConfig(**payload["config"])
        result = SubtitlesPipeline(job, settings, event=emit).run()
        return {
            "job_dir": str(result.job_dir),
            "target_srt": str(result.target_srt),
            "output_video": str(result.output_video) if result.output_video else None,
            "report": result.report,
        }

    if kind == "say":
        from pipeline.say import SaySpec, render_speech

        raw = dict(payload["spec"])
        for key in ("clone", "output", "output_root"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        spec = SaySpec(**raw)
        result = render_speech(
            spec,
            settings,
            models_dir=models_dir,
            voices_root=voices_root,
            event=emit,
        )
        return result.to_dict()

    if kind == "voice_design":
        from pipeline.voice_ops import design_voice

        raw = dict(payload["params"])
        for key in ("output", "output_root"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        result = design_voice(
            voices_root, models_dir, event=emit, **raw
        )
        return result.to_dict()

    if kind == "voice_clone":
        from pipeline.voice_ops import clone_voice

        raw = dict(payload["params"])
        raw["audio"] = Path(raw["audio"])
        for key in ("output", "output_root"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        result = clone_voice(voices_root, models_dir, event=emit, **raw)
        return result.to_dict()

    if kind == "voice_add":
        from pipeline.voice_ops import register_voice

        raw = dict(payload["params"])
        raw["audio"] = Path(raw["audio"])
        result = register_voice(voices_root, models_dir, event=emit, **raw)
        return result.to_dict()

    if kind == "model_download":
        names = list(payload.get("names") or [])
        paths = download_many(
            names,
            payload.get("models_dir") or settings.models_dir,
            force=bool(payload.get("force")),
            progress=lambda name: emit("download", {"model": name}),
        )
        return {name: str(path) for name, path in paths.items()}

    if kind == "anchor_generate":
        from pipeline.anchors import generate_anchors

        result = generate_anchors(
            anchors_dir(models_dir),
            models_dir,
            list(payload.get("languages") or []),
            device=payload.get("device") or "auto",
            dtype=payload.get("dtype") or "auto",
            force=bool(payload.get("force")),
            event=emit,
        )
        return {"anchors": result}

    raise DubbingError(f"Unknown job kind {kind!r}")


def run_job(payload: dict[str, Any], queue: Any) -> None:
    """Execute one job and report its result; called in a spawned process."""
    logging.getLogger().handlers.clear()
    handler = _QueueLogHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if payload.get("verbose") else logging.INFO)
    quiet_dependencies()

    def emit(kind: str, data: dict[str, Any]) -> None:
        queue.put({"type": "event", "kind": kind, "data": data})

    code = 0
    try:
        result = _dispatch(payload, emit)
        queue.put({"type": "result", "data": result})
    except DubbingError as exc:
        queue.put({"type": "error", "message": str(exc)})
        code = 1
    except Exception:  # noqa: BLE001 - translated to a safe user message
        logging.getLogger(__name__).exception("Unexpected job failure")
        queue.put(
            {
                "type": "error",
                "message": "Unexpected error; expand the log for details.",
            }
        )
        code = 1
    finally:
        try:
            queue.close()
            queue.join_thread()
        except Exception:  # noqa: BLE001 - process is exiting anyway
            pass
    raise SystemExit(code)
