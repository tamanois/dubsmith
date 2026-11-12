"""Subtitles-only pipeline: recognition (and optionally translation) -> SRT -> video.

This is independent of dubbing. It reuses the ASR and translation modules so the
transcript and wording cannot disagree with a dub built from the same input, but
it never loads a TTS model, a stretcher or a separator. The output is a subtitle
file, plus — when the input is a video — that video with the subtitles embedded:

* ``soft`` (default) muxes a toggleable track, copying every original stream;
* ``burn`` renders the text into the frames, re-encoding the video;
* ``off`` writes the sidecar SRT only.

The stage cache is a transcript sidecar: a second run with the same settings
skips recognition, exactly as the dubbing pipeline's manifest does. A
subtitles-only job lives under ``<output>/<stem>-subtitles`` so it cannot
overwrite a dub's artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import media, subtitles
from .asr import Transcript, load_transcript, save_transcript, transcribe
from .config import Settings, SubtitleJobConfig
from .errors import MediaError
from .logging_utils import job_logger
from .models import resolve_model
from .translate import OpenAICompatTranslator, TranslationRequest, Translator

__all__ = ["SubtitlesPipeline", "SubtitleResult"]

_AUDIO_NAME = "01_audio.wav"
_TRANSCRIPT_NAME = "02_transcript.json"
_SUBTITLE_NAME = "subs_target.srt"
_SOFT_VIDEO_NAME = "video_subtitled.mp4"
_BURNED_VIDEO_NAME = "video_burned.mp4"
_REPORT_NAME = "report.json"

#: Transcription signature. Injected in tests so no model is ever loaded.
Transcriber = Callable[[Path], Transcript]


def _whisper_registry_name(value: str) -> str:
    """Map the CLI's ``--whisper-model`` spelling onto a registry key.

    The dubbing CLI and the docs say ``large-v3`` while the model registry calls
    it ``whisper_large_v3``; accepting both here keeps the two commands
    interchangeable.
    """
    key = str(value).strip().lower().replace("-", "_")
    if key in {"large_v3", "largev3", "large", "whisper_large_v3"}:
        return "whisper_large_v3"
    return str(value)


@dataclass
class SubtitleResult:
    """Where a finished subtitles-only job left its outputs."""

    job_dir: Path
    target_srt: Path
    output_video: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable multi-line summary for the CLI."""
        lines = [f"Output directory: {self.job_dir}"]
        lines.append(f"  subtitles    : {self.target_srt.name}")
        if self.output_video is not None:
            lines.append(f"  video        : {self.output_video.name}")
        segments = self.report.get("segment_count")
        if segments is not None:
            lines.append(f"  segments     : {segments}")
        cues = self.report.get("cue_count")
        if cues is not None:
            lines.append(f"  cues         : {cues}")
        lines.append(
            "  translated   : "
            + ("yes" if self.report.get("translated") else "no (target == source)")
        )
        return "\n".join(lines)


class SubtitlesPipeline:
    """Runs the subtitles-only stages for one :class:`SubtitleJobConfig`."""

    def __init__(
        self,
        job: SubtitleJobConfig,
        settings: Settings | None = None,
        *,
        transcriber: Transcriber | None = None,
        translator: Translator | None = None,
        event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.job = job
        self.settings = settings or Settings.from_env()
        self._event = event
        self.input_path = job.input_path
        self.job_dir = job.job_dir
        self.job_id = f"{self.input_path.stem}-subtitles-{job.fingerprint()}"
        self.log = job_logger("subtitles", self.job_id)
        self._transcriber = transcriber
        self._translator = translator

    # -- public API -------------------------------------------------------- #

    def run(self) -> SubtitleResult:
        """Recognise, translate, write the SRT and embed it in the video."""
        if not self.input_path.exists():
            raise MediaError(f"Input file not found: {self.input_path}")

        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.log.info("Starting subtitle job for %s", self.input_path.name)

        audio_path = self._stage("audio", lambda: self._audio_path())
        transcript = self._stage(
            "transcribe", lambda: self._transcribe(audio_path)
        )
        texts, translated = self._stage(
            "translate", lambda: self._texts(transcript)
        )

        def write_subtitles() -> tuple[list[Any], Path]:
            cues = subtitles.cues_from_segments(
                transcript.segments, texts, max_chars=self.job.max_chars
            )
            return cues, subtitles.write_srt(cues, self.job_dir / _SUBTITLE_NAME)

        cues, target_srt = self._stage("subtitles", write_subtitles)

        output_video = None
        if self.job.is_video and self.job.subtitles_mode != "off":
            output_video = self._stage(
                "embed", lambda: self._embed(target_srt)
            )

        report = {
            "mode": "subtitles",
            "input": str(self.input_path),
            "is_video": self.job.is_video,
            "source_language": transcript.language,
            "target_language": self.job.target_language,
            "translated": translated,
            "segment_count": len(transcript.segments),
            "cue_count": len(cues),
            "subtitles_mode": self.job.subtitles_mode,
            "subtitle_file": target_srt.name,
            "output_video": output_video.name if output_video is not None else None,
        }
        (self.job_dir / _REPORT_NAME).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self.log.info("Subtitles complete: %s", self.job_dir)
        return SubtitleResult(
            job_dir=self.job_dir,
            target_srt=target_srt,
            output_video=output_video,
            report=report,
        )

    # -- stages ------------------------------------------------------------ #

    def _emit(self, kind: str, **payload: Any) -> None:
        """Notify the optional observer, never letting it break the run."""
        if self._event is None:
            return
        try:
            self._event(kind, payload)
        except Exception:  # noqa: BLE001 - an observer must not fail a job
            self.log.debug("Stage observer failed for %s", kind, exc_info=True)

    def _stage(self, name: str, func: Callable[[], Any]) -> Any:
        """Run one stage, emitting its measured start/end and duration."""
        self._emit("stage_start", stage=name)
        started = time.perf_counter()
        try:
            result = func()
        except BaseException:
            self._emit(
                "stage_failed",
                stage=name,
                seconds=round(time.perf_counter() - started, 3),
            )
            raise
        self._emit(
            "stage_end",
            stage=name,
            seconds=round(time.perf_counter() - started, 3),
        )
        return result

    def _audio_path(self) -> Path:
        """Audio to recognise: the input itself, or an extracted mono WAV."""
        if not self.job.is_video:
            return self.input_path
        audio = self.job_dir / _AUDIO_NAME
        if audio.is_file() and not self.job.force:
            return audio
        info = media.probe(self.input_path)
        if not info.has_audio:
            raise MediaError(
                f"{self.input_path.name} has no audio stream to transcribe."
            )
        return media.extract_audio(self.input_path, audio)

    def _transcribe(self, audio_path: Path) -> Transcript:
        """Recognise speech, reusing a cached transcript unless ``--force``."""
        cached = self.job_dir / _TRANSCRIPT_NAME
        if cached.is_file() and not self.job.force:
            self.log.info("Reusing transcript %s", cached.name)
            return load_transcript(cached)

        if self._transcriber is not None:
            transcript = self._transcriber(audio_path)
        else:
            model_dir = resolve_model(
                _whisper_registry_name(self.job.whisper_model),
                self.settings.models_dir,
            )
            transcript = transcribe(
                audio_path,
                model_dir,
                device=self.job.device,
                language=self.job.source_code,
                no_speech_threshold=self.job.no_speech_threshold,
            )
        save_transcript(transcript, cached)
        return transcript

    def _texts(self, transcript: Transcript) -> tuple[dict[int, str], bool]:
        """Target-language text per segment id, and whether an LLM was used."""
        source = transcript.language
        target = self.job.target_code
        if target is None or (source is not None and source == target):
            # No translation needed: keep the recognised words.
            return {}, False

        requests = [
            TranslationRequest(id=segment.id, text=segment.text, duration=segment.duration)
            for segment in transcript.segments
            if segment.text.strip()
        ]
        if not requests:
            return {}, False

        translator = self._translator_instance()
        translate_subtitles = getattr(translator, "translate_subtitles", None)
        if callable(translate_subtitles):
            translated = translate_subtitles(
                requests,
                source_language=source,
                target_language=self.job.target_language,
            )
        else:  # A translator that predates the subtitle prompt still works.
            translated = translator.translate(
                requests,
                source_language=source,
                target_language=self.job.target_language,
            )
        return translated, True

    def _translator_instance(self) -> Translator:
        """Build the configured OpenAI-compatible translator on first use."""
        if self._translator is None:
            self._translator = OpenAICompatTranslator(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.job.llm_model or self.settings.llm_model,
                timeout_s=self.settings.llm_timeout_s,
                chunk_size=self.settings.llm_chunk_size,
            )
        return self._translator

    def _embed(self, target_srt: Path) -> Path:
        """Put the SRT into the video, softly or burned in."""
        if self.job.subtitles_mode == "burn":
            output = self.job_dir / _BURNED_VIDEO_NAME
            return media.burn_subtitles(
                self.input_path, target_srt, output, language=self.job.target_language
            )
        output = self.job_dir / _SOFT_VIDEO_NAME
        return media.add_subtitle_track(
            self.input_path, target_srt, output, language=self.job.target_language
        )
