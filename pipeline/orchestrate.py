"""Pipeline orchestration: stages, artifacts, caching and reporting.

Each stage writes its output into the job directory and records completion in
``manifest.json`` keyed by a fingerprint of the job settings. Re-running a job
therefore reuses finished stages, which is what makes iterating on translation or
accent settings cheap: ASR and diarization are never repeated needlessly.

Heavy components (transcriber, translator, TTS engine, stretcher) are injected
through the constructor, so the whole flow can be exercised in tests with fakes
and no GPU, models, or binaries.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import media, subtitles
from .asr import Segment, Transcript, load_transcript, save_transcript, transcribe
from .config import (
    JobConfig,
    Settings,
    language_code_for_name,
    normalize_loudness_policy,
    release_gpu_memory,
)
from .errors import ASRError, DubbingError, MediaError, TimingError
from .logging_utils import job_logger
from .models import resolve_model
from .mixing import duck_and_mix
from .separation import NullSeparator, detect_separator
from .timing import Placement, Timeline, build_timeline, detect_stretcher, fit_line
from .translate import (
    OpenAICompatTranslator,
    ShortenRequest,
    TranslationRequest,
    Translator,
)
from .tts import TTSEngine, select_donor

__all__ = ["Pipeline", "JobResult", "Manifest"]

_AUDIO_NAME = "01_audio.wav"
_SPEECH_NAME = "01b_speech.wav"
_BACKGROUND_NAME = "01c_background.wav"
_SEPARATION_MARKER = "01b_separation.done"
_TRANSCRIPT_NAME = "02_transcript.json"
_SCRIPT_NAME = "03_script.json"
#: Lines the length repair rewrote, and what it achieved. Written next to the
#: script so a cached run can still report them.
_SPOKEN_NAME = "03b_spoken.json"
_REFINE_NAME = "03c_refine.json"
_DUBBED_DIR = "04_dubbed"
_DONE_MARKER = ".done"
_DONOR_NAME = "04_dubbed/donor.json"
_TIMING_NAME = "timing.json"
_MIX_NAME = "05_mix.wav"
_DUB_TRACK_NAME = "dub.wav"
_REPORT_NAME = "report.json"
_MANIFEST_NAME = "manifest.json"
_TMP_DIR = "tmp"

#: Peak ceiling applied to each synthesised line (-3 dBFS). Taming lines before
#: they are mixed keeps their relative levels consistent and stops a few isolated
#: transient peaks from forcing the whole program several dB too quiet.
_LINE_CEILING = 10.0 ** (-3.0 / 20.0)


def _load_placements(path: Path) -> list[Placement]:
    """Re-read placement records so a cached mix still produces a full report."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [
        Placement(
            segment_id=item["id"],
            start=item["placed_start"],
            end=item["placed_end"],
            ratio=item["ratio"],
            source_start=item["source_start"],
            shifted=item.get("shifted", False),
            overrun=item.get("overrun", 0.0),
            trimmed=item.get("trimmed", 0.0),
            synth_seconds=item.get("synth_seconds", 0.0),
            deficit=item.get("deficit", 0.0),
        )
        for item in payload
    ]


@dataclass
class JobResult:
    """Where a completed job left its outputs."""

    job_dir: Path
    dub_track: Path
    dubbed_video: Path | None = None
    source_srt: Path | None = None
    target_srt: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable multi-line summary for the CLI."""
        lines = [f"Output directory: {self.job_dir}"]
        if self.dubbed_video is not None:
            lines.append(f"  dubbed video : {self.dubbed_video.name}")
        lines.append(f"  dub audio    : {self.dub_track.name}")
        if self.source_srt is not None:
            lines.append(f"  source subs  : {self.source_srt.name}")
        if self.target_srt is not None:
            lines.append(f"  target subs  : {self.target_srt.name}")
        segments = self.report.get("segment_count")
        if segments is not None:
            lines.append(f"  segments     : {segments}")
        if self.stage_seconds:
            stages = ", ".join(f"{k} {v:.1f}s" for k, v in self.stage_seconds.items())
            lines.append(f"  stages       : {stages}")
        return "\n".join(lines)


class Manifest:
    """Tracks which stages already completed for a given job fingerprint."""

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self._data: dict[str, Any] = {"fingerprint": fingerprint, "stages": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        # A different fingerprint means the settings changed: start fresh.
        if payload.get("fingerprint") == self.fingerprint:
            self._data = payload

    def is_done(self, stage: str, outputs: Sequence[Path]) -> bool:
        """True when the stage completed and all of its outputs still exist."""
        if stage not in self._data.get("stages", {}):
            return False
        return all(Path(output).exists() for output in outputs)

    def mark(self, stage: str, **meta: Any) -> None:
        """Record a stage as complete and persist the manifest."""
        self._data.setdefault("stages", {})[stage] = {"at": time.time(), **meta}
        self.save()

    def save(self) -> None:
        """Write the manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")


class Pipeline:
    """Runs the dubbing stages for one :class:`JobConfig`."""

    def __init__(
        self,
        job: JobConfig,
        settings: Settings | None = None,
        *,
        transcriber: Callable[[Path], Transcript] | None = None,
        translator: Translator | None = None,
        tts_engine: TTSEngine | None = None,
        stretcher: Any | None = None,
        separator: Any | None = None,
        stretcher_preference: str = "auto",
    ) -> None:
        self.job = job
        self.settings = settings or Settings.from_env()
        self.input_path = job.input_path
        self.stem = self.input_path.stem
        self.job_dir = job.output_dir / self.stem
        self.job_id = f"{self.stem}-{job.fingerprint()}"
        self.log = job_logger("pipeline", self.job_id)

        self._transcriber = transcriber
        self._translator = translator
        self._tts = tts_engine
        self._stretcher = stretcher
        self._separator = separator
        self._stretcher_preference = stretcher_preference
        self._stage_seconds: dict[str, float] = {}
        #: Segments of the run currently being refined (set by the length repair).
        self._segments_for_refine: list[Any] = []

        self.manifest = Manifest(self.job_dir / _MANIFEST_NAME, job.fingerprint())

    # -- public API -------------------------------------------------------- #

    def run(self) -> JobResult:
        """Execute every stage and return the resulting artifact paths."""
        if not self.input_path.exists():
            raise MediaError(f"Input file not found: {self.input_path}")

        self.job_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.job_dir / _TMP_DIR
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self.log.info("Starting job for %s", self.input_path.name)
        try:
            audio_path = self._stage_audio()
            speech_path = self._stage_separate(audio_path)
            # ASR and donor selection run on the speech stem when we have one:
            # music and ambience only degrade both.
            source_audio = speech_path or audio_path
            transcript = self._stage_transcribe(source_audio, tmp_dir)
            # Free the ASR model's cache before TTS loads its own weights;
            # otherwise the two multi-GB models contend for the same GPU.
            release_gpu_memory()
            script = self._stage_translate(transcript)
            dubbed = self._stage_synthesize(source_audio, transcript, script, tmp_dir)
            release_gpu_memory()
            timeline = self._stage_mix(audio_path, transcript, dubbed, tmp_dir)
            return self._stage_deliver(timeline, transcript, script)
        finally:
            # Never leak scratch space, even on failure.
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- stage plumbing ---------------------------------------------------- #

    def _stage(self, name: str, outputs: Sequence[Path], func: Callable[[], Any]) -> Any:
        """Run a stage unless it is already cached; record timing either way."""
        if self.manifest.is_done(name, outputs) and not self.job.force:
            self.log.info("Stage %s: cached", name)
            return None

        self.log.info("Stage %s: running", name)
        started = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - started
        self._stage_seconds[name] = elapsed
        self.manifest.mark(name, seconds=round(elapsed, 2))
        self.log.info("Stage %s: done in %.1fs", name, elapsed)
        return result

    def _resolve_stretcher(self, tmp_dir: Path) -> Any:
        if self._stretcher is not None:
            return self._stretcher
        self._stretcher = detect_stretcher(self._stretcher_preference, work_dir=tmp_dir)
        self.log.debug("Using %s time-stretcher", self._stretcher.name)
        return self._stretcher

    def _resolve_translator(self) -> Translator:
        if self._translator is None:
            self._translator = OpenAICompatTranslator(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.job.llm_model or self.settings.llm_model,
                timeout_s=self.settings.llm_timeout_s,
                chunk_size=self.settings.llm_chunk_size,
            )
        return self._translator

    def _resolve_tts(self) -> TTSEngine:
        if self._tts is None:
            model_dir = resolve_model("tts_base", self.settings.models_dir)
            self._tts = TTSEngine(
                model_dir,
                device=self.job.device,
                x_vector_only=self.job.accent_mode == "speaker-only",
            )
        return self._tts

    def _resolve_separator(self) -> Any:
        if self._separator is None:
            self._separator = detect_separator(self.job.separator, device=self.job.device)
        return self._separator

    # -- stages ------------------------------------------------------------ #

    def _stage_audio(self) -> Path:
        """Produce a 48 kHz mono WAV from the input, caching the result."""
        out = self.job_dir / _AUDIO_NAME

        def build() -> Path:
            if self.job.is_video:
                info = media.probe(self.input_path)
                if not info.has_audio:
                    raise MediaError(
                        f"{self.input_path.name} contains no audio stream, so there is "
                        "nothing to transcribe or dub. Check with:\n"
                        f'    ffprobe -v error -show_entries stream=codec_type "{self.input_path.name}"'
                    )
                return media.extract_audio(self.input_path, out)

            # Audio-only input: no ffmpeg needed at all.
            samples, sample_rate = media.load_audio(
                self.input_path, sample_rate=media.DEFAULT_SAMPLE_RATE
            )
            if samples.size == 0:
                raise MediaError(f"{self.input_path.name} contains no audio samples.")
            return media.write_audio(out, samples, sample_rate)

        cached = self._stage("audio", [out], build)
        return out if cached is None else cached

    def _stage_separate(self, audio_path: Path) -> Path | None:
        """Split the source into speech and background stems.

        Returns the speech stem's path, or ``None`` when background preservation
        is disabled or no separator is available — in which case the pipeline
        behaves exactly as it did before this stage existed.

        The speech stem improves ASR and donor quality; the background stem is
        mixed back under the dub so music and ambience survive.
        """
        if not self.job.keep_background:
            return None

        if isinstance(self._resolve_separator(), NullSeparator):
            return None

        speech_path = self.job_dir / _SPEECH_NAME
        background_path = self.job_dir / _BACKGROUND_NAME
        marker = self.job_dir / _SEPARATION_MARKER

        def build() -> Path:
            samples, sample_rate = media.load_audio(audio_path)
            speech, background = self._resolve_separator().separate(samples, sample_rate)
            media.write_audio(speech_path, speech, sample_rate)
            media.write_audio(background_path, background, sample_rate)
            marker.write_text(getattr(self._resolve_separator(), "name", "unknown"), encoding="utf-8")
            return speech_path

        result = self._stage("separate", [marker], build)
        if result is not None:
            return result
        return speech_path if speech_path.is_file() else None

    def _background_path(self) -> Path | None:
        """Path to the background stem, when one was produced."""
        candidate = self.job_dir / _BACKGROUND_NAME
        return candidate if candidate.is_file() else None

    def _donor_info(self) -> dict[str, Any] | None:
        """Which reference voice was used: an explicit file, or an auto segment.

        The synthesise stage writes ``donor.json``; when it is absent (an older job
        directory, or a stage served from cache) an explicit reference is still
        known from the job config, so the report stays useful either way.
        """
        path = self.job_dir / _DONOR_NAME
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        if self.job.ref_audio is not None:
            info: dict[str, Any] = {
                "source": "reference-audio",
                "path": str(self.job.ref_audio),
                "transcript_provided": bool((self.job.ref_text or "").strip()),
                "accent_mode": self.job.accent_mode,
            }
        elif self.job.accent_mode == "reference-accent":
            # Timbre came from the video, so it is unknown without donor.json; the
            # accent clip is still known from the config.
            info = {"accent_mode": self.job.accent_mode}
        else:
            return None

        if self.job.accent_audio is not None:
            info["accent"] = {
                "source": "reference-audio",
                "path": str(self.job.accent_audio),
                "transcript_provided": bool((self.job.accent_text or "").strip()),
            }
        return info

    def _spoken_overrides(self) -> dict[int, str]:
        """Lines the length repair rewrote, so subtitles match what is heard."""
        path = self.job_dir / _SPOKEN_NAME
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        overrides: dict[int, str] = {}
        for key, value in payload.items():
            try:
                overrides[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return overrides

    def _refine_stats(self) -> dict[str, Any]:
        """What the length repair achieved, or zeros when it did not run."""
        default: dict[str, Any] = {
            "attempted": False,
            "offenders": 0,
            "improved": 0,
            "seconds_recovered": 0.0,
            "llm_calls": 0,
        }
        path = self.job_dir / _REFINE_NAME
        if not path.is_file():
            return default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
        return {**default, **payload} if isinstance(payload, dict) else default

    def _stage_transcribe(self, audio_path: Path, tmp_dir: Path) -> Transcript:
        """Transcribe the audio, caching the transcript."""
        out = self.job_dir / _TRANSCRIPT_NAME

        def build() -> Transcript:
            if self._transcriber is not None:
                transcript = self._transcriber(audio_path)
            else:
                model_dir = resolve_model("whisper_large_v3", self.settings.models_dir)
                transcript = transcribe(
                    audio_path,
                    model_dir,
                    device=self.job.device,
                    language=language_code_for_name(self.job.source_language),
                    no_speech_threshold=self.job.no_speech_threshold,
                )
            save_transcript(transcript, out)
            return transcript

        result = self._stage("transcribe", [out], build)
        return load_transcript(out) if result is None else result

    def _reference_speech_rate(self) -> float | None:
        """Words per second of the target language, measured from a reference clip.

        A model cannot judge how long a sentence takes to say, but it can count
        words, so converting the slot into a word ceiling makes the budget
        actionable. The rate has to come from *target-language* speech: the
        auto-selected donor is the source speaker, whose rate describes the source
        language and would size the translation wrongly.
        """
        if self.job.length_budget != "words":
            return None

        if self.job.accent_mode == "reference-accent":
            clip, text = self.job.accent_audio, self.job.accent_text
        elif self.job.ref_audio is not None and self.job.ref_text:
            clip, text = self.job.ref_audio, self.job.ref_text
        else:
            self.log.info(
                "No target-language reference clip, so no word budget can be "
                "derived; translating with duration hints only."
            )
            return None

        if clip is None or not text or not text.strip():
            return None
        try:
            samples, sample_rate = media.load_audio(clip)
        except Exception as exc:  # noqa: BLE001 - a missing rate is not fatal
            self.log.warning("Could not measure the reference clip's rate: %s", exc)
            return None

        seconds = len(np.asarray(samples).reshape(-1)) / float(sample_rate or 1)
        rate = len(text.split()) / seconds if seconds > 0 else 0.0
        if rate <= 0:
            return None
        self.log.info("Reference speaking rate: %.2f words/s (from %s)", rate, clip)
        return rate

    def _stage_translate(self, transcript: Transcript) -> dict[int, str]:
        """Translate every segment to the target language."""
        out = self.job_dir / _SCRIPT_NAME

        def build() -> dict[int, str]:
            rate = self._reference_speech_rate()
            requests = [
                TranslationRequest(
                    id=segment.id,
                    text=segment.text,
                    duration=segment.duration,
                    # Only set when a rate is known; leaving it None keeps the
                    # prompt identical to a run made before budgets existed.
                    max_words=(
                        max(1, int(round(segment.duration * rate))) if rate else None
                    ),
                )
                for segment in transcript.segments
            ]
            translations = self._resolve_translator().translate(
                requests,
                source_language=transcript.language,
                target_language=self.job.target_language,
            )
            missing = [r.id for r in requests if r.id not in translations]
            if missing:
                raise ASRError(f"Translation missing for segment ids {missing}")
            out.write_text(
                json.dumps(
                    {
                        "target_language": self.job.target_language,
                        "source_language": transcript.language,
                        "lines": [
                            {
                                "id": segment.id,
                                "start": segment.start,
                                "end": segment.end,
                                "source": segment.text,
                                "target": translations[segment.id],
                            }
                            for segment in transcript.segments
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return translations

        result = self._stage("translate", [out], build)
        if result is not None:
            return result
        payload = json.loads(out.read_text(encoding="utf-8"))
        return {line["id"]: line["target"] for line in payload["lines"]}

    def _stage_synthesize(
        self,
        audio_path: Path,
        transcript: Transcript,
        script: dict[int, str],
        tmp_dir: Path,
    ) -> dict[int, Path]:
        """Clone the donor voice and synthesise one file per segment."""
        out_dir = self.job_dir / _DUBBED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / _DONE_MARKER
        expected = {segment.id: out_dir / f"seg_{segment.id:04d}.wav" for segment in transcript.segments}

        def build() -> dict[int, Path]:
            engine = self._resolve_tts()

            # Resolve the timbre source once: an explicit reference file, else the
            # best segment of the source speech. The accent mode decides how the
            # prompt is then built, not where the voice comes from.
            donor = None
            if self.job.ref_audio is not None:
                self.log.info("Using reference voice %s", self.job.ref_audio)
                donor_info: dict[str, Any] = {
                    "source": "reference-audio",
                    "path": str(self.job.ref_audio),
                    "transcript_provided": bool((self.job.ref_text or "").strip()),
                }
            else:
                samples, sample_rate = media.load_audio(audio_path)
                donor = select_donor(samples, sample_rate, transcript.segments)
                self.log.info(
                    "Auto-selected donor: segment %d (%.1fs)",
                    donor.segment_id,
                    donor.duration,
                )
                donor_info = {
                    "source": "auto",
                    "segment_id": donor.segment_id,
                    "start": round(donor.start, 2),
                    "end": round(donor.end, 2),
                    "duration": round(donor.duration, 2),
                    "transcript": donor.text,
                }

            if self.job.accent_mode == "reference-accent":
                assert self.job.accent_audio is not None  # guaranteed by JobConfig
                self.log.info(
                    "Hybrid voice: timbre from %s, accent from %s",
                    donor_info.get("path") or f"segment {donor_info['segment_id']}",
                    self.job.accent_audio,
                )
                engine.prepare_hybrid_prompt(
                    accent_audio=self.job.accent_audio,
                    accent_text=self.job.accent_text or "",
                    timbre=donor,
                    timbre_audio=self.job.ref_audio,
                )
                donor_info["accent"] = {
                    "source": "reference-audio",
                    "path": str(self.job.accent_audio),
                    "transcript_provided": bool((self.job.accent_text or "").strip()),
                }
            elif self.job.ref_audio is not None:
                engine.prepare_prompt_from_file(self.job.ref_audio, self.job.ref_text or "")
            else:
                assert donor is not None  # set in the branch above
                engine.prepare_prompt(donor)

            donor_info["accent_mode"] = self.job.accent_mode
            # Persisted so the report can state the donor even when the stage is
            # served from cache on a later run.
            (self.job_dir / _DONOR_NAME).write_text(
                json.dumps(donor_info, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            texts = [script.get(segment.id, "") for segment in transcript.segments]
            wavs = engine.synthesize(texts, language=self.job.target_language)
            wavs = self._refine_overlong(
                engine, transcript, script, wavs, tmp_dir
            )

            written: dict[int, Path] = {}
            for segment, wav in zip(transcript.segments, wavs):
                path = expected[segment.id]
                if wav is None:
                    # Blank line: leave no file so the timing stage emits silence.
                    path.unlink(missing_ok=True)
                    continue
                # Limit (never boost) so a line's dynamic range matches its peers
                # and cannot dominate the mix with a transient peak.
                media.write_audio(
                    path,
                    media.limit_peaks(wav, engine.sample_rate, ceiling=_LINE_CEILING),
                    engine.sample_rate,
                )
                written[segment.id] = path
            # A marker keeps the cache independent of how many lines were blank.
            marker.write_text(str(len(written)), encoding="utf-8")
            return written

        result = self._stage("synthesize", [marker], build)
        if result is not None:
            return result
        return {sid: path for sid, path in expected.items() if path.exists()}

    # -- length repair ----------------------------------------------------- #

    def _overflow_seconds(self, raw_seconds: float, index: int) -> float:
        """How far a line of ``raw_seconds`` would overflow its window.

        Mirrors the mix stage exactly: the stretch clamp is applied first, then
        the line may spill into the pause before the next utterance. Only what
        exceeds *that* has to be cut, so only that is worth repairing. The final
        line has no successor to protect, so it may run freely.
        """
        segments = self._segments_for_refine
        slot = segments[index].duration
        if slot <= 0 or raw_seconds <= 0:
            return 0.0
        ratio = min(self.job.max_stretch, max(self.job.min_stretch, slot / raw_seconds))
        stretched = raw_seconds * ratio
        if index + 1 >= len(segments):
            return 0.0
        allowed = slot + max(0.0, segments[index + 1].start - segments[index].end)
        return stretched - allowed

    def _refine_overlong(
        self,
        engine: TTSEngine,
        transcript: Transcript,
        script: dict[int, str],
        wavs: list[np.ndarray | None],
        tmp_dir: Path,
    ) -> list[np.ndarray | None]:
        """Rewrite and re-synthesise only the lines that would lose words.

        Translated speech is routinely longer than its slot, and the stretch
        clamp cannot always absorb the difference — what is left over gets cut.
        Rather than deleting words, ask the translator to say the same thing in
        fewer, then synthesise just that line again.

        Two properties make this safe to leave on by default:

        * lines that already fit are never touched, so their audio is bit-for-bit
          what it was before;
        * a rewrite is adopted **only** if it measurably reduces the overflow, so
          a line can never end up worse than it was.
        """
        stats: dict[str, Any] = {
            "attempted": False,
            "offenders": 0,
            "improved": 0,
            "seconds_recovered": 0.0,
            "llm_calls": 0,
        }
        (self.job_dir / _REFINE_NAME).write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )

        if not self.job.refine or self.job.refine_max_passes < 1:
            return wavs

        translator = self._resolve_translator()
        if not callable(getattr(translator, "shorten", None)):
            # A custom translator may only implement translate(); repairing is
            # then impossible rather than silently lossy.
            self.log.info(
                "Translator %s cannot rewrite lines, so the length repair is skipped.",
                type(translator).__name__,
            )
            return wavs

        segments = transcript.segments
        self._segments_for_refine = segments
        sample_rate = engine.sample_rate
        spoken: dict[int, str] = {}

        for _pass in range(self.job.refine_max_passes):
            offenders: list[tuple[int, float]] = []
            for index, wav in enumerate(wavs):
                if wav is None or not np.asarray(wav).size:
                    continue
                raw = len(np.asarray(wav).reshape(-1)) / float(sample_rate)
                overflow = self._overflow_seconds(raw, index)
                if overflow > self.job.refine_threshold:
                    offenders.append((index, overflow))

            if not offenders:
                break

            stats["attempted"] = True
            stats["offenders"] = max(stats["offenders"], len(offenders))

            requests = []
            for index, _overflow in offenders:
                segment = segments[index]
                # Always start from the original translation so repeated runs
                # cannot shorten an already-shortened line.
                current = spoken.get(segment.id) or script.get(segment.id, "")
                words = max(1, len(current.split()))
                raw = len(np.asarray(wavs[index]).reshape(-1)) / float(sample_rate)
                ratio = min(
                    self.job.max_stretch, max(self.job.min_stretch, segment.duration / raw)
                )
                allowed = max(0.0, raw * ratio - self._overflow_seconds(raw, index))
                ceiling = max(1, int(words * (allowed / (raw * ratio))))
                requests.append(
                    ShortenRequest(
                        id=segment.id,
                        text=current,
                        source=segment.text,
                        spoken_seconds=round(raw * ratio, 2),
                        available_seconds=round(allowed, 2),
                        max_words=ceiling,
                    )
                )

            try:
                rewritten = translator.shorten(
                    requests, target_language=self.job.target_language
                )
                stats["llm_calls"] += 1
            except Exception as exc:  # noqa: BLE001 - repair is best-effort
                self.log.warning(
                    "Could not rewrite %d overlong line(s): %s. Keeping the "
                    "existing audio, so the result matches a run without refinement.",
                    len(requests),
                    exc,
                )
                break

            texts = [
                rewritten.get(request.id) or request.text for request in requests
            ]
            try:
                retried = engine.synthesize(texts, language=self.job.target_language)
            except Exception as exc:  # noqa: BLE001 - repair is best-effort
                self.log.warning("Re-synthesis failed: %s. Keeping existing audio.", exc)
                break

            for (index, overflow), request, text, wav in zip(
                offenders, requests, texts, retried
            ):
                if wav is None or not np.asarray(wav).size:
                    continue
                new_raw = len(np.asarray(wav).reshape(-1)) / float(sample_rate)
                new_overflow = self._overflow_seconds(new_raw, index)
                if new_overflow >= overflow - 1e-3:
                    # No measurable gain: keep what we already had.
                    continue
                wavs[index] = wav
                spoken[segments[index].id] = text
                stats["improved"] += 1
                stats["seconds_recovered"] += max(0.0, overflow) - max(0.0, new_overflow)

        stats["seconds_recovered"] = round(stats["seconds_recovered"], 3)
        (self.job_dir / _REFINE_NAME).write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        if spoken:
            (self.job_dir / _SPOKEN_NAME).write_text(
                json.dumps({str(k): v for k, v in spoken.items()}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.log.info(
                "Refined %d of %d line(s), recovering %.2fs of speech.",
                stats["improved"],
                stats["offenders"],
                stats["seconds_recovered"],
            )
        return wavs

    def _stage_mix(
        self,
        audio_path: Path,
        transcript: Transcript,
        dubbed: dict[int, Path],
        tmp_dir: Path,
    ) -> Timeline:
        """Fit every line into its slot and assemble the dub track."""
        out = self.job_dir / _MIX_NAME

        def build() -> Timeline:
            stretcher = self._resolve_stretcher(tmp_dir)
            original, sample_rate = media.load_audio(audio_path)
            total_duration = len(original) / float(sample_rate)

            items: list[tuple[int, float, np.ndarray]] = []
            ratios: dict[int, float] = {}
            overruns: dict[int, float] = {}
            trimmed: dict[int, float] = {}
            synth_seconds: dict[int, float] = {}
            deficits: dict[int, float] = {}
            segments = transcript.segments

            for index, segment in enumerate(segments):
                path = dubbed.get(segment.id)
                synth: np.ndarray | None = None
                if path is not None:
                    samples, sr = media.load_audio(path)
                    if sr != sample_rate:
                        samples = media.resample(samples, sr, sample_rate)
                    synth = samples

                # A line may spill into the silence before the next utterance:
                # that costs nothing (nobody is talking) and avoids cutting
                # words. Anything beyond it would push the next line late, so the
                # residue is trimmed. The last line may run freely.
                if index + 1 < len(segments):
                    gap = max(0.0, segments[index + 1].start - segment.end)
                    max_overrun: float | None = gap
                else:
                    max_overrun = None

                fitted, ratio = fit_line(
                    synth,
                    sample_rate,
                    segment.duration,
                    stretcher=stretcher,
                    min_ratio=self.job.min_stretch,
                    max_ratio=self.job.max_stretch,
                    max_overrun=max_overrun,
                )
                ratios[segment.id] = ratio
                kept = len(fitted) / float(sample_rate)
                overruns[segment.id] = max(0.0, kept - segment.duration)

                if synth is not None and np.asarray(synth).size:
                    stretched = len(synth) / float(sample_rate) * ratio
                    trimmed[segment.id] = max(
                        0.0, stretched - max(segment.duration, kept)
                    )
                    synth_seconds[segment.id] = len(synth) / float(sample_rate)
                    deficits[segment.id] = max(0.0, segment.duration - stretched)
                else:
                    trimmed[segment.id] = 0.0
                    synth_seconds[segment.id] = 0.0
                    deficits[segment.id] = 0.0

                items.append((segment.id, segment.start, fitted))

            timeline = build_timeline(
                items, sample_rate=sample_rate, total_duration=total_duration
            )
            for placement in timeline.placements:
                placement.ratio = ratios.get(placement.segment_id, 1.0)
                placement.overrun = overruns.get(placement.segment_id, 0.0)
                placement.trimmed = trimmed.get(placement.segment_id, 0.0)
                placement.synth_seconds = synth_seconds.get(placement.segment_id, 0.0)
                placement.deficit = deficits.get(placement.segment_id, 0.0)

            notable = sum(1 for value in overruns.values() if value > 0.01)
            if notable:
                self.log.info(
                    "%d line(s) used the gap before the next utterance (%.1fs); "
                    "%.1fs trimmed where it would have desynchronised the video.",
                    notable,
                    sum(overruns.values()),
                    sum(trimmed.values()),
                )

            # Lay the dub over the background so music and ambience survive. The
            # bed is ducked exactly where dialogue plays, which is what keeps the
            # voice intelligible without muting the soundtrack.
            background = self._background_path()
            if background is not None:
                regions = [(p.start, p.end) for p in timeline.placements]
                bed, bed_rate = media.load_audio(background)
                if bed_rate != sample_rate:
                    bed = media.resample(bed, bed_rate, sample_rate)
                timeline = Timeline(
                    samples=duck_and_mix(
                        timeline.samples,
                        bed,
                        regions,
                        sample_rate,
                        duck_db=self.job.duck_db,
                        background_gain_db=self.job.background_gain_db,
                    ),
                    sample_rate=sample_rate,
                    placements=timeline.placements,
                )

            media.write_audio(out, timeline.samples, sample_rate)
            (self.job_dir / _TIMING_NAME).write_text(
                json.dumps(
                    [
                        {
                            "id": p.segment_id,
                            "source_start": round(p.source_start, 3),
                            "placed_start": round(p.start, 3),
                            "placed_end": round(p.end, 3),
                            "ratio": round(p.ratio, 4),
                            "overrun": round(p.overrun, 3),
                            "trimmed": round(p.trimmed, 3),
                            "synth_seconds": round(p.synth_seconds, 3),
                            "deficit": round(p.deficit, 3),
                            "shifted": p.shifted,
                        }
                        for p in timeline.placements
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            return timeline

        result = self._stage("mix", [out], build)
        if result is not None:
            return result
        samples, sample_rate = media.load_audio(out)
        return Timeline(
            samples=samples,
            sample_rate=sample_rate,
            placements=_load_placements(self.job_dir / _TIMING_NAME),
        )

    def _stage_deliver(
        self,
        timeline: Timeline,
        transcript: Transcript,
        script: dict[int, str],
    ) -> JobResult:
        """Normalise loudness, write subtitles, mux the video and report."""
        dub_track = self.job_dir / _DUB_TRACK_NAME
        # Normalise here (rather than via save_normalized) so the achieved
        # loudness can be reported accurately instead of re-measuring later.
        target_lufs = self._resolve_loudness_target()
        loudness = media.normalize_loudness_detailed(
            timeline.samples, timeline.sample_rate, target_lufs=target_lufs
        )
        media.write_audio(dub_track, loudness.samples, timeline.sample_rate)

        source_srt = target_srt = None
        if self.job.keep_subtitles:
            # Subtitles follow the text that was actually spoken, so a line the
            # length repair re-wrote does not display words the viewer cannot hear.
            spoken = self._spoken_overrides()
            source_srt = subtitles.write_srt(
                [
                    subtitles.Cue(s.start, s.end, s.text)
                    for s in transcript.segments
                ],
                self.job_dir / "subs_source.srt",
            )
            target_srt = subtitles.write_srt(
                [
                    subtitles.Cue(
                        s.start, s.end, spoken.get(s.id) or script.get(s.id, "")
                    )
                    for s in transcript.segments
                ],
                self.job_dir / "subs_target.srt",
            )

        dubbed_video = None
        if self.job.is_video:
            dubbed_video = media.mux_audio(
                self.input_path, dub_track, self.job_dir / "video_dubbed.mp4"
            )

        report = self._build_report(timeline, transcript, script, dubbed_video, loudness)
        (self.job_dir / _REPORT_NAME).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self.log.info("Job complete: %s", self.job_dir)
        return JobResult(
            job_dir=self.job_dir,
            dub_track=dub_track,
            dubbed_video=dubbed_video,
            source_srt=source_srt,
            target_srt=target_srt,
            report=report,
            stage_seconds=dict(self._stage_seconds),
        )

    def _source_loudness(self) -> float | None:
        """Integrated loudness of the extracted source audio, when measurable."""
        audio = self.job_dir / _AUDIO_NAME
        if not audio.is_file():
            return None
        samples, sample_rate = media.load_audio(audio)
        return media.measure_loudness(samples, sample_rate)

    def _loudness_policy(self) -> str:
        """Effective loudness policy (job override, else settings)."""
        return normalize_loudness_policy(
            self.job.loudness_target or self.settings.loudness_target
        )

    def _resolve_loudness_target(self) -> float:
        """LUFS the delivered dub should sit at.

        Matching the source is the sensible default for a replacement track: the
        viewer keeps the volume they expected from that video. It also sidesteps
        the case where a fixed broadcast target is unreachable for peaky speech.
        Falls back to the configured absolute target when the source is
        unmeasurable or the policy asks for one.
        """
        fallback = (
            self.job.target_lufs
            if self.job.target_lufs is not None
            else self.settings.target_lufs
        )
        if self._loudness_policy() != "match-source":
            return float(fallback)

        source = self._source_loudness()
        if source is None:
            self.log.warning(
                "Source loudness could not be measured; using %.1f LUFS instead.",
                fallback,
            )
            return float(fallback)
        return float(source)

    def _build_report(
        self,
        timeline: Timeline,
        transcript: Transcript,
        script: dict[int, str],
        dubbed_video: Path | None,
        loudness: media.LoudnessResult,
    ) -> dict[str, Any]:
        """Collect cheap quality signals so a run can be judged without listening."""
        source_lufs = self._source_loudness()
        ratios = [p.ratio for p in timeline.placements]
        shifted = sum(1 for p in timeline.placements if p.shifted)
        overruns = [p.overrun for p in timeline.placements]
        trimmed = [p.trimmed for p in timeline.placements]
        deficits = [p.deficit for p in timeline.placements]

        # A line's translation was too long for its slot when it overran, and too
        # short when it left silence. The two are very different failures — an
        # overlong line loses words, a short one only pauses — so they are
        # reported separately rather than lumped into one "clamped" count.
        def spoken_seconds(p: Placement) -> float:
            return p.synth_seconds * p.ratio

        deleted_lines = [
            p for p in timeline.placements if p.trimmed > 0.01 and spoken_seconds(p) > 0
        ]
        worst_deleted = max(
            (p.trimmed / spoken_seconds(p) for p in deleted_lines), default=0.0
        )
        words = sum(
            len((self._spoken_overrides().get(p.segment_id) or script.get(p.segment_id) or "").split())
            for p in timeline.placements
        )
        spoken_total = sum(p.synth_seconds for p in timeline.placements)
        return {
            "job_id": self.job_id,
            "input": str(self.input_path),
            "is_video": self.job.is_video,
            "source_language": transcript.language,
            "target_language": self.job.target_language,
            "segment_count": len(transcript.segments),
            "transcript_seconds": round(transcript.duration_s, 2),
            "dub_seconds": round(len(timeline.samples) / timeline.sample_rate, 2),
            # Loudness of the delivered file, plus what happened on the way there.
            "loudness_dub_lufs": (
                round(loudness.achieved_lufs, 2) if loudness.achieved_lufs is not None else None
            ),
            "loudness_source_lufs": round(source_lufs, 2) if source_lufs is not None else None,
            "loudness_target_lufs": round(loudness.target_lufs, 2),
            "loudness_policy": self._loudness_policy(),
            # Positive when the dub is louder than the track it replaced.
            "loudness_delta_vs_source": (
                round(loudness.achieved_lufs - source_lufs, 2)
                if loudness.achieved_lufs is not None and source_lufs is not None
                else None
            ),
            "loudness_gain_db": round(loudness.gain_db, 2),
            # True when the limiter engaged: peaks were reduced so the target
            # loudness could be reached without clipping.
            "loudness_peak_limited": loudness.peak_limited,
            "background_preserved": self._background_path() is not None,
            "duck_db": self.job.duck_db if self._background_path() is not None else None,
            "accent_mode": self.job.accent_mode,
            "donor": self._donor_info(),
            #: What the length repair did: which lines were re-worded to avoid
            #: cutting them, and how much speech that saved.
            "refine": self._refine_stats(),
            "stretch_ratio_min": round(min(ratios), 3) if ratios else None,
            "stretch_ratio_max": round(max(ratios), 3) if ratios else None,
            "clamped_segments": sum(
                1
                for r in ratios
                if r <= self.job.min_stretch + 1e-6 or r >= self.job.max_stretch - 1e-6
            ),
            #: Split of ``clamped_segments`` by direction. A high clamp means the
            #: line could not be slowed enough to fill its slot (harmless: it
            #: leaves silence). A low clamp means it could not be compressed
            #: enough (harmful: the excess may have to be cut).
            "clamped_long_segments": sum(
                1 for r in ratios if r <= self.job.min_stretch + 1e-6
            ),
            "clamped_short_segments": sum(
                1 for r in ratios if r >= self.job.max_stretch - 1e-6
            ),
            #: Lines that left silence in their slot, and how much in total.
            "deficit_segments": sum(1 for value in deficits if value > 0.01),
            "deficit_seconds_total": round(sum(deficits), 2),
            #: Lines whose speech was longer than slot + following gap.
            "surplus_seconds_total": round(sum(overruns) + sum(trimmed), 2),
            #: The metric that matters most: the largest share of a single line
            #: that had to be deleted. An aggregate can look harmless while a
            #: couple of lines lose a quarter of their words.
            "worst_line_deleted_ratio": round(worst_deleted, 3),
            "lines_over_10pct_deleted": sum(
                1 for p in deleted_lines if p.trimmed / spoken_seconds(p) > 0.10
            ),
            #: Words per second the cloned voice actually delivered. Lets the
            #: translation budget be sized from measurement, not guesswork.
            "speech_rate_wps": (
                round(words / spoken_total, 2) if spoken_total > 0 else None
            ),
            "shifted_segments": shifted,
            # Lines that used the gap before the next utterance instead of being
            # cut. Speech is only dropped when it would have desynchronised the
            # video, which `trimmed_seconds_total` measures.
            "overrun_segments": sum(1 for value in overruns if value > 0.01),
            "overrun_seconds_total": round(sum(overruns), 2),
            "overrun_seconds_max": round(max(overruns), 2) if overruns else 0.0,
            "trimmed_segments": sum(1 for value in trimmed if value > 0.01),
            "trimmed_seconds_total": round(sum(trimmed), 2),
            "outputs": {
                "dub_track": self.job_dir.name + "/" + _DUB_TRACK_NAME,
                "video": dubbed_video.name if dubbed_video else None,
            },
            "stage_seconds": self._stage_seconds,
        }
