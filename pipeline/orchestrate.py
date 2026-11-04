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
import logging
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import media, subtitles
from .anchors import (
    ANCHOR_TEXT,
    AnchorSpec,
    ensure_anchor,
    measure_gender,
    measure_rate,
    rate_band,
)
from .asr import (
    Segment,
    Transcript,
    asr_device,
    load_transcript,
    resolve_compute_type,
    save_transcript,
    transcribe,
    transcribe_many,
)
from .config import (
    JobConfig,
    Settings,
    configure_allocator,
    language_code_for_name,
    normalize_loudness_policy,
    probe_device,
    release_gpu_memory,
)
from .diarize import NullDiarizer, assign_streams, detect_diarizer, single_speaker_labels
from .errors import ASRError, DubbingError, MediaError, TimingError
from .logging_utils import job_logger
from .models import anchors_dir, resolve_model
from .mixing import apply_room, duck_and_mix, room_tone_floor
from .overlap_separation import NullSpeakerSeparator, detect_speaker_separator
from .parallel import (
    BLEND,
    PROBE,
    SYNTH,
    InlineRunner,
    ProcessRunner,
    Runner,
    Task,
    WorkerConfig,
    merge_tts_stats,
    split_evenly,
)
from .separation import NullSeparator, detect_separator
from .timing import (
    Placement,
    Timeline,
    build_timeline,
    detect_stretcher,
    fit_line,
    fit_ratio,
    redistribute_packed,
    trim_silence,
)
from .translate import (
    OpenAICompatTranslator,
    ShortenRequest,
    TranslationRequest,
    Translator,
)
from .tts import (
    MAX_DECODE_SECONDS,
    DonorClip,
    TTSEngine,
    VoiceDesignEngine,
    resolve_dtype,
    select_donor,
)
from .voice import VoiceSpec

__all__ = ["Pipeline", "JobResult", "Manifest"]

_AUDIO_NAME = "01_audio.wav"
_SPEECH_NAME = "01b_speech.wav"
_BACKGROUND_NAME = "01c_background.wav"
_SEPARATION_MARKER = "01b_separation.done"
_TRANSCRIPT_NAME = "02_transcript.json"
_SCRIPT_NAME = "03_script.json"
#: Words per second the cloned voice measured, per speaker. Written before the
#: script because the word budget every translation is given comes from it.
_RATE_NAME = "03a_rate.json"
#: Words of accent text the probe reads. A sentence or two is enough to measure a
#: rate; reading a whole narration would cost seconds of GPU for no more accuracy.
_RATE_PROBE_WORDS = 40
#: Lines the length repair rewrote, and what it achieved. Written next to the
#: script so a cached run can still report them.
_SPOKEN_NAME = "03b_spoken.json"
_REFINE_NAME = "03c_refine.json"
_DUBBED_DIR = "04_dubbed"
#: Lines the pre-synthesis word check had to reword, and what it achieved. Written
#: before synthesis so a cached run can still report what was spoken.
_PREFIT_NAME = "03d_prefit.json"
_DONE_MARKER = ".done"
_DONOR_NAME = "04_dubbed/donor.json"
#: The one-off clip the two-stage ``reference-accent`` path clones every line
#: from, and how it was chosen. Both live together because the JSON describes
#: this exact WAV; a stale pair would misreport the voice being cloned.
_BLEND_NAME = "04_dubbed/reference_blend.wav"
_BLEND_INFO_NAME = "04_dubbed/blend.json"
#: Which speaker each segment belongs to. A sidecar rather than a field in the
#: transcript, so ASR output stays pure recognition and changing the speaker
#: threshold does not force a re-transcribe.
_SPEAKERS_NAME = "02b_speakers.json"
#: Which segments carry a second voice, and how the run handled them. A sidecar
#: for the same reason as the labels: it depends on the audio and the threshold,
#: not on the translation or the chosen voice.
_OVERLAP_NAME = "02c_overlap.json"
#: The lines a flagged segment was split into, one per voice, and the speaker each
#: belongs to. Written by the overlap-separation stage and consumed when the lines
#: to dub are assembled, so the handling is cached independently of the flags.
_OVERLAP_LINES_NAME = "02d_overlap_segments.json"
#: Re-synthesises a chosen subset of lines, each in its own speaker's voice, and
#: returns one waveform per line in the order given (``None`` for a blank line).
Resynthesize = Callable[[Sequence[Segment], Sequence[str]], list[np.ndarray | None]]

#: Used whenever there is one voice: diarization off, unavailable, or finding only
#: a single speaker. Keeps every downstream lookup total.
_SINGLE_SPEAKER = "SPK_0"
#: Per-speaker record keys that must not be copied to the top level of the donor
#: record: they describe one voice, not the job.
_SPEAKER_ONLY_KEYS = frozenset({"segments", "speech_seconds"})
_TIMING_NAME = "timing.json"
_MIX_NAME = "05_mix.wav"
#: What the room-matching pass did, beside the mix it describes. A sidecar so a
#: cached mix still reports it, exactly as ``blend.json`` does for the blend.
_ROOM_NAME = "05_room.json"
_DUB_TRACK_NAME = "dub.wav"
_REPORT_NAME = "report.json"
_MANIFEST_NAME = "manifest.json"
_TMP_DIR = "tmp"

#: Peak ceiling applied to each synthesised line (-3 dBFS). Taming lines before
#: they are mixed keeps their relative levels consistent and stops a few isolated
#: transient peaks from forcing the whole program several dB too quiet.
_LINE_CEILING = 10.0 ** (-3.0 / 20.0)


def _json_similarity(value: Any) -> float | None:
    """A similarity as a JSON-safe number, or ``None`` when it was not scored.

    JSON has no NaN or Infinity, and writing them makes the report unreadable to
    strict parsers such as ``jq``, so an unscored take becomes ``null``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if np.isfinite(number) else None


def _blend_paths(job_dir: Path, speakers: Sequence[str]) -> dict[str, Path]:
    """Where each speaker's blended reference lives.

    A single-speaker job keeps the original filename, so existing artifacts, the
    docs and `audit_timbre.py` go on pointing at the same file.
    """
    if len(speakers) == 1:
        return {speakers[0]: job_dir / _BLEND_NAME}
    return {
        name: job_dir / _DUBBED_DIR / f"reference_blend_{name}.wav" for name in speakers
    }


def _gap_seconds(segments: Sequence[Segment], index: int) -> float:
    """Silence between a line and the next one; 0 when they touch or overlap.

    An overlay line shares its slot with the dominant line and the last line has
    nothing after it to protect, so neither may use a gap.
    """
    segment = segments[index]
    if segment.overlay or index + 1 >= len(segments):
        return 0.0
    return max(0.0, segments[index + 1].start - segment.end)


def _available_seconds(segments: Sequence[Segment], index: int) -> float:
    """Time a line may actually use: its slot plus the gap before the next one.

    The single definition of "how long is this line allowed to be". The mix stage,
    the length repair and the report all go through it so they cannot disagree
    about which lines are overlong.
    """
    return segments[index].duration + _gap_seconds(segments, index)


def _budget_words(available: float, words_per_second: float, speedup: float) -> int:
    """Most words a line may be given, for a window and a measured delivery rate.

    ``speedup`` is the tempo the budget is willing to spend on every line, so a
    line the model fills exactly still fits with room to spare. Never 0: an
    empty line is a missing line.
    """
    if available <= 0.0 or words_per_second <= 0.0 or speedup <= 0.0:
        return 0
    return max(1, int(round(available * words_per_second / speedup)))


def _probe_text(accent: tuple[Any, str] | None, language: str) -> str:
    """Text to measure a voice's delivery rate with: the accent's own, capped.

    Reusing the accent text measures the exact material the voice borrows its
    rhythm from - the generated anchor in ``native`` mode, the supplied clip's
    transcript in ``reference-accent`` - so the probe adds no new content to a
    job. The per-language anchor text is the fallback when no accent text exists.
    """
    text = (accent[1] if accent is not None else "") or ""
    words = text.split() or ANCHOR_TEXT.get(language, "").split()
    return " ".join(words[:_RATE_PROBE_WORDS])


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
            needed_ratio=item.get("needed_ratio", 0.0),
            silence_trimmed=item.get("silence_trimmed", 0.0),
        )
        for item in payload
    ]


@dataclass
class JobResult:
    """Where a completed job left its outputs."""

    job_dir: Path
    dub_track: Path
    dubbed_video: Path | None = None
    burned_video: Path | None = None
    source_srt: Path | None = None
    target_srt: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable multi-line summary for the CLI."""
        lines = [f"Output directory: {self.job_dir}"]
        if self.dubbed_video is not None:
            lines.append(f"  dubbed video : {self.dubbed_video.name}")
        if self.burned_video is not None:
            lines.append(f"  burned video : {self.burned_video.name}")
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
        anchor_engine: Any | None = None,
        stretcher: Any | None = None,
        separator: Any | None = None,
        diarizer: Any | None = None,
        speaker_separator: Any | None = None,
        stream_transcriber: Callable[[Path], Transcript] | None = None,
        stretcher_preference: str = "auto",
    ) -> None:
        self.job = job
        self.settings = settings or Settings.from_env()
        self.input_path = job.input_path
        self.stem = self.input_path.stem
        self.job_dir = job.job_dir
        self.job_id = f"{self.stem}-{job.fingerprint()}"
        self.log = job_logger("pipeline", self.job_id)

        self._transcriber = transcriber
        self._translator = translator
        self._tts = tts_engine
        #: VoiceDesign engine used to generate accent anchors. Injected in tests;
        #: ``None`` resolves the real checkpoint lazily, and only on a cache miss.
        self._anchor_engine = anchor_engine
        self.__anchor_engine_resolved: Any | None = None
        self._stretcher = stretcher
        self._separator = separator
        self._diarizer = diarizer
        self._speaker_separator = speaker_separator
        #: Re-transcribes a separated stream. Injected in tests; ``None`` falls back
        #: to the transcriber above, or to a batched real Whisper pass.
        self._stream_transcriber = stream_transcriber
        #: Cached :meth:`detect_diarizer` result. ``detect_diarizer`` never returns
        #: ``None``, so ``None`` here reliably means "not probed yet".
        self.__diarizer_resolved: Any | None = None
        #: Cached :meth:`detect_speaker_separator` result, same reasoning.
        self.__speaker_separator_resolved: Any | None = None
        self._stretcher_preference = stretcher_preference
        self._stage_seconds: dict[str, float] = {}
        #: Segments of the run currently being refined (set by the length repair).
        self._segments_for_refine: list[Any] = []
        #: Accent clip per speaker (path/array + its transcript), resolved once by
        #: the accent stage and consumed by the blend and synthesis stages.
        self._accents: dict[str, tuple[Any, str]] = {}
        #: Per-speaker accent provenance for the report, mirroring :attr:`_accents`.
        self._accent_records: dict[str, dict[str, Any]] = {}
        #: The anchor whose pace sizes the translator's word budget, when one exists.
        self._primary_accent: tuple[Any, str] | None = None
        #: Name of the speaker that :attr:`_primary_accent` belongs to.
        self._primary_speaker_name: str | None = None
        #: Execution backend for the fan-out stages. Built once per run so the
        #: blend stage and the synthesis that follows it share their workers, and
        #: therefore load each device's checkpoint once between them rather than
        #: once per stage.
        self._runner_instance: Runner | None = None
        #: Runs of lines whose ASR timing was impossible, and what the repair did.
        self._packed_runs: list[dict[str, Any]] = []
        #: Donor per speaker and its report record, resolved once and shared by the
        #: blend and synthesis stages. ``None`` until either of them needs it.
        self._donors_resolved: (
            dict[str, tuple[DonorClip | None, dict[str, Any]]] | None
        ) = None

        self.manifest = Manifest(self.job_dir / _MANIFEST_NAME, job.fingerprint())

    # -- public API -------------------------------------------------------- #

    def run(self) -> JobResult:
        """Execute every stage and return the resulting artifact paths."""
        if not self.input_path.exists():
            raise MediaError(f"Input file not found: {self.input_path}")

        # Before any model loads: torch reads the allocator setting at its first
        # allocation, so applying it later does nothing and says nothing.
        configure_allocator(self.job.primary_device)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.job_dir / _TMP_DIR
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self.log.info("Starting job for %s", self.input_path.name)
        run_started = time.perf_counter()
        try:
            audio_path = self._stage_audio()
            speech_path = self._stage_separate(audio_path)
            # ASR and donor selection run on the speech stem when we have one:
            # music and ambience only degrade both.
            source_audio = speech_path or audio_path
            transcript = self._stage_transcribe(source_audio, tmp_dir)
            # Who speaks each line. Annotates the segments in place, and is a no-op
            # beyond one speaker when diarization is off or unavailable.
            self._stage_diarize(source_audio, transcript)
            # Which lines carry a second voice. Its own cached stage: it re-embeds
            # short windows rather than whole segments, so sharing the diarization
            # stage would couple two independent answers to one cache entry.
            self._stage_overlap(source_audio, transcript)
            # Split the flagged segments into one stream per voice, re-transcribe
            # each and match it to a speaker. Runs before `release_gpu_memory`
            # because it needs both the ASR model (re-transcription) and the
            # speaker embedder (stream-to-speaker matching).
            self._stage_overlap_separate(source_audio, transcript)
            # The lines actually dubbed: the ASR segments, with each flagged overlap
            # segment replaced by one line per voice. Donor selection and the source
            # subtitles keep using the original ASR segments.
            lines = self._dub_lines(transcript)
            # Whisper's boundaries occasionally squeeze consecutive lines into a
            # fraction of the time their words need (measured: nine words in 0.9 s,
            # four of them in 0.16 s). The text is right and the timing is not, so
            # the slots are re-timed here - before the word budget is derived from
            # them - and never past the next line's start, so nothing downstream
            # moves and no sync is traded away.
            self._packed_runs = redistribute_packed(lines)
            self.log.info(
                "Source: %s, %.1fs, %d line(s), %d speaker(s)",
                self.input_path.name,
                transcript.duration_s,
                len(lines),
                len(self._speaker_order(transcript.segments)),
            )
            if self._packed_runs:
                self.log.info(
                    "Re-timed %d run(s) of impossible ASR timing (%d line(s), worst "
                    "%.1f words/s): %s.",
                    len(self._packed_runs),
                    int(sum(run["lines"] for run in self._packed_runs)),
                    max(run["worst_wps"] for run in self._packed_runs),
                    ", ".join(
                        f"{int(run['from_id'])}-{int(run['to_id'])}"
                        for run in self._packed_runs[:5]
                    ),
                )
            # Free the ASR model's cache before TTS loads its own weights;
            # otherwise the two multi-GB models contend for the same GPU.
            release_gpu_memory()
            # Which accent clip each voice borrows pronunciation from. Resolved (and
            # generated) here, before the base model loads, so the VoiceDesign and
            # cloning checkpoints are never resident together.
            accents = self._stage_accent_anchor(source_audio, transcript)
            release_gpu_memory()
            # Rendered before synthesis so every line is cloned from one clip
            # whose speech codes and speaker embedding agree - one clip per voice.
            blends = self._stage_blend(source_audio, transcript, accents)
            # How fast the cloned voice actually speaks. Measured from the voice
            # itself, because every translation is sized by it: the accent clip's
            # own rate reads ~20% below what the clone delivers (2.97 against 3.65
            # measured), and a ceiling that tight is one the model ignores. The
            # blend and the probe both need the base checkpoint, so they run back
            # to back and it loads once.
            rates = self._stage_rate_probe(source_audio, transcript, accents, blends)
            script = self._stage_translate(transcript, lines, rates)
            # What will actually be spoken: the script, minus the lines whose
            # translation cannot fit even at the compression floor. Checked here
            # rather than after synthesis, because a line that has to be reworded
            # should never be synthesised twice.
            fitted = self._stage_prefit(lines, script, rates)
            dubbed = self._stage_synthesize(
                source_audio, transcript, lines, fitted, tmp_dir, blends
            )
            release_gpu_memory()
            timeline = self._stage_mix(audio_path, transcript, lines, dubbed, tmp_dir)
            return self._stage_deliver(timeline, transcript, lines, script)
        finally:
            self._close_runner()
            elapsed = time.perf_counter() - run_started
            self.log.info(
                "Job finished in %.1fs",
                elapsed,
                extra={"seconds": round(elapsed, 3)},
            )
            # Never leak scratch space, even on failure.
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- stage plumbing ---------------------------------------------------- #

    def _stage(self, name: str, outputs: Sequence[Path], func: Callable[[], Any]) -> Any:
        """Run a stage unless it is already cached; record timing either way."""
        if self.manifest.is_done(name, outputs) and not self.job.force:
            self.log.info("Stage %s: cached", name, extra={"stage": name})
            return None

        self.log.info("Stage %s: running", name, extra={"stage": name})
        started = time.perf_counter()
        try:
            result = func()
        except BaseException:
            elapsed = time.perf_counter() - started
            self.log.error(
                "Stage %s: failed after %.1fs",
                name,
                elapsed,
                extra={"stage": name, "seconds": round(elapsed, 3)},
            )
            raise
        elapsed = time.perf_counter() - started
        self._stage_seconds[name] = elapsed
        self.manifest.mark(name, seconds=round(elapsed, 2))
        self.log.info(
            "Stage %s: done in %.1fs",
            name,
            elapsed,
            extra={"stage": name, "seconds": round(elapsed, 3)},
        )
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
            info = probe_device(self.job.primary_device)
            dtype = resolve_dtype(self.job.primary_device, self.job.dtype)
            # Logged because "which device did it actually use?" is the first
            # question on unfamiliar hardware, and the answer decides the runtime.
            self.log.info(
                "Synthesising on %s (%s) with %s.",
                info.label,
                info.name or info.kind,
                str(dtype).replace("torch.", ""),
            )
            self._tts = TTSEngine(
                model_dir,
                device=self.job.primary_device,
                dtype=dtype,
                batch_size=self.job.tts_batch,
                x_vector_only=self.job.accent_mode == "speaker-only",
            )
        return self._tts

    # -- execution backend -------------------------------------------------- #

    def _runner(self) -> Runner:
        """The backend the fan-out stages hand their work to.

        One device - or an injected engine, which is how tests drive the pipeline
        - means :class:`InlineRunner`: the work runs here, on the engine this
        process already resolved. That is not a fallback but the definition of the
        single-device path, so a ``--device`` run synthesises exactly what it did
        before ``--devices`` existed, and the fan-out *planning* stays fully
        testable with a fake engine and no subprocess.

        Several devices means one worker process each. A worker builds its own
        engine on its own card, which is why the parent never loads a second copy
        of the checkpoint.
        """
        if self._runner_instance is not None:
            return self._runner_instance
        devices = self.job.resolved_devices
        if len(devices) <= 1 or self._tts is not None:
            self._runner_instance = InlineRunner(self._resolve_tts(), self.log)
        else:
            self._runner_instance = ProcessRunner(
                [self._worker_config(device) for device in devices], log=self.log
            )
            self.log.info(
                "Blend and synthesis are spread over %d devices: %s",
                len(devices),
                ", ".join(devices),
            )
        return self._runner_instance

    def _worker_config(self, device: str) -> WorkerConfig:
        """What one worker needs to build its own copy of the checkpoint."""
        return WorkerConfig(
            model_dir=resolve_model("tts_base", self.settings.models_dir),
            device=device,
            dtype=self.job.dtype,
            batch_size=self.job.tts_batch,
            x_vector_only=self.job.accent_mode == "speaker-only",
            verbose=self.log.isEnabledFor(logging.DEBUG),
        )

    def _close_runner(self) -> None:
        """Stop the workers, if any were started.

        Called from ``run``'s ``finally`` so a failed job cannot leave several
        processes holding several GB of VRAM each.
        """
        if self._runner_instance is not None:
            self._runner_instance.close()
            self._runner_instance = None

    def _resolve_anchor_engine(self) -> Any:
        """The VoiceDesign engine used to build accent anchors.

        Resolved lazily - and only on a cache miss - so a run whose anchors are
        already generated never loads this checkpoint, let alone alongside the
        cloning one. A missing model raises the actionable download error from
        :func:`resolve_model`, which the CLI prints as-is.
        """
        if self._anchor_engine is not None:
            return self._anchor_engine
        if self.__anchor_engine_resolved is None:
            model_dir = resolve_model("tts_voice_design", self.settings.models_dir)
            self.log.info("Generating an accent anchor with the VoiceDesign model.")
            self.__anchor_engine_resolved = VoiceDesignEngine(
                model_dir,
                device=self.job.primary_device,
                dtype=resolve_dtype(self.job.primary_device, self.job.dtype),
            )
        return self.__anchor_engine_resolved

    def _resolve_separator(self) -> Any:
        if self._separator is None:
            self._separator = detect_separator(
                self.job.separator, device=self.job.primary_device
            )
        return self._separator

    def _resolve_speaker_separator(self) -> Any:
        """The overlap separator for this job (never ``None``).

        Degrades to :class:`NullSpeakerSeparator` when the optional backend is
        absent, so ``--overlap separate`` on a machine without it leaves the
        dominant-only behaviour rather than failing the job.
        """
        if self._speaker_separator is not None:
            return self._speaker_separator
        if self.__speaker_separator_resolved is None:
            self.__speaker_separator_resolved = detect_speaker_separator(
                self.job.overlap_separator, device=self.job.primary_device
            )
        return self.__speaker_separator_resolved

    def _resolve_diarizer(self) -> Any | None:
        """The speaker labeler for this job, or ``None`` for a single voice.

        ``--diarizer none`` wins over everything: it is an explicit request to dub
        every line in one voice, so it must not be second-guessed. Otherwise an
        injected labeler is used, which is how tests avoid the optional model.
        """
        if not self.job.multi_speaker:
            return None
        if self._diarizer is not None:
            return self._diarizer
        if self.__diarizer_resolved is None:
            self.__diarizer_resolved = detect_diarizer(
                self.job.diarizer,
                device=self.job.primary_device,
                threshold=self.job.resolved_speaker_threshold,
                min_seconds=self.job.min_speaker_seconds,
            )
        return self.__diarizer_resolved

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

    def _speech_path(self) -> Path | None:
        """Path to the speech stem, when one was produced.

        Room matching reads this stem rather than the mix: it is the only material
        that carries the room (the voice, its reverb tail, and the room tone the
        voice was masking).
        """
        candidate = self.job_dir / _SPEECH_NAME
        return candidate if candidate.is_file() else None

    def _load_bed(
        self, sample_rate: int, tone: np.ndarray | None
    ) -> np.ndarray | None:
        """The bed the dub is laid over: music/ambience, plus room tone if any.

        The static background offset is applied here rather than inside
        :func:`duck_and_mix`, because ``tone`` already carries the gain it was
        built with and a shared offset would lift it too.
        """
        bed: np.ndarray | None = None
        background = self._background_path()
        if background is not None:
            bed, bed_rate = media.load_audio(background)
            if bed_rate != sample_rate:
                bed = media.resample(bed, bed_rate, sample_rate)
            gain = 10.0 ** (self.job.background_gain_db / 20.0)
            bed = (np.asarray(bed, dtype=np.float32) * gain).astype(np.float32)

        if tone is None:
            return bed
        tone = np.asarray(tone, dtype=np.float32).reshape(-1)
        if bed is None:
            return tone

        combined = np.zeros(max(bed.size, tone.size), dtype=np.float32)
        combined[: bed.size] += bed
        combined[: tone.size] += tone
        return combined

    def _match_room(
        self,
        speech: np.ndarray,
        regions: Sequence[tuple[float, float]],
        sample_rate: int,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
        """Place the dry dub in the source's room, when room matching is enabled.

        Returns ``(stats, speech, tone)``: the record written to ``05_room.json``,
        the speech (possibly convolved with a reverb tail), and the room-tone floor
        to fold into the bed (``None`` when there is none).

        Room matching needs the separated speech stem, so with the background
        disabled the dub is delivered dry and the report says ``applied: false``
        rather than inventing a space.
        """
        stats: dict[str, Any] = {
            "mode": self.job.room,
            "applied": False,
            "rt60_s": None,
            #: ``wet_db`` is 0 when the reverb tail is off, which is the default.
            "wet_db": None,
            "reverb": self.job.room_wet_db < 0.0,
            "tone_gain_db": None,
            "duck_db": self.job.duck_db,
        }
        if not self.job.room_matching:
            return stats, speech, None

        reference_path = self._speech_path()
        if reference_path is None:
            self.log.warning(
                "Room matching needs the separated speech stem, which only exists "
                "when the background is kept. Delivering a dry dub instead: drop "
                "--room match, or keep the background (the default)."
            )
            return stats, speech, None

        reference, reference_rate = media.load_audio(reference_path)
        if reference_rate != sample_rate:
            reference = media.resample(reference, reference_rate, sample_rate)

        wet, rt60 = apply_room(
            speech,
            reference,
            sample_rate,
            wet_db=self.job.room_wet_db,
            rt60=self.job.room_rt60,
        )
        tone = room_tone_floor(
            reference, sample_rate, regions, gain_db=self.job.room_tone_gain_db
        )
        stats.update(
            {
                "applied": True,
                "rt60_s": round(rt60, 3),
                "wet_db": round(self.job.room_wet_db, 2),
                "reverb": self.job.room_wet_db < 0.0,
                "tone_gain_db": round(self.job.room_tone_gain_db, 2),
                # Reported because it is the depth actually used, not --duck-db.
                "duck_db": round(self.job.room_duck_db, 2),
            }
        )
        self.log.info(
            "Room matching: RT60 %.2fs measured, room tone %.1f dB, duck %.1f dB, "
            "reverb %s",
            rt60,
            self.job.room_tone_gain_db,
            self.job.room_duck_db,
            "off" if self.job.room_wet_db >= 0.0 else f"{self.job.room_wet_db:.1f} dB",
        )
        return stats, wet, tone

    def _donor_info(self) -> dict[str, Any] | None:
        """Which reference voice was used: an explicit file, or an auto segment.

        The synthesise stage writes ``donor.json``; when it is absent (an older job
        directory, or a stage served from cache) an explicit reference is still
        known from the job config, so the report stays useful either way.
        """
        path = self.job_dir / _DONOR_NAME
        if path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                stored = None
            if isinstance(stored, dict):
                # The blend record travels with the donor one; a directory written
                # before it existed re-derives the strategy from the config.
                blend_stats = self._blend_stats()
                if blend_stats is not None:
                    stored.setdefault("blend", blend_stats)
                return stored

        if self.job.ref_audio is not None:
            info: dict[str, Any] = {
                "source": "reference-audio",
                "path": str(self.job.ref_audio),
                "transcript_provided": bool((self.job.ref_text or "").strip()),
                "accent_mode": self.job.accent_mode,
            }
        elif self.job.accent_mode in ("native", "reference-accent"):
            # Timbre came from the video, so it is unknown without donor.json; the
            # accent source is still known from config or the resolved anchors.
            info = {"accent_mode": self.job.accent_mode}
        else:
            return None

        if self.job.accent_audio is not None:
            info["accent"] = {
                "source": "reference-audio",
                "path": str(self.job.accent_audio),
                "transcript_provided": bool((self.job.accent_text or "").strip()),
            }
        elif self._accent_records:
            primary = self._primary_speaker_name
            info["accent"] = self._accent_records.get(primary) or next(
                iter(self._accent_records.values())
            )
        blend_stats = self._blend_stats()
        if blend_stats is not None:
            info["blend"] = blend_stats
        return info

    def _speakers_summary(self) -> dict[str, Any] | None:
        """Per-speaker summary, or ``None`` when the video has a single voice.

        ``None`` rather than a one-entry map, so single-speaker reports stay
        exactly as they were before diarization existed.
        """
        donor = self._donor_info() or {}
        speakers = donor.get("speakers")
        if not isinstance(speakers, dict) or len(speakers) < 2:
            return None
        return {
            name: {
                "segments": record.get("segments"),
                "speech_seconds": record.get("speech_seconds"),
                "donor_segment": record.get("segment_id"),
            }
            for name, record in speakers.items()
        }

    def _overlap_stats(self) -> dict[str, Any] | None:
        """What the overlap stage found, or ``None`` when it did not run.

        Read from disk rather than held in memory, so a run served from cache still
        reports it; ``None`` keeps the field out of one-voice reports entirely,
        exactly as ``speakers`` stays ``null`` for them.
        """
        path = self.job_dir / _OVERLAP_NAME
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        # When separation ran, the flagged lines were split into two voices rather
        # than dubbed in one: record what actually happened, so the report does not
        # say "dominant" when the second voice was recovered.
        lines = self._overlap_lines()
        if lines:
            payload["mode"] = "separate"
            payload["separated"] = len(
                {record.get("parent_id") for record in lines}
            )
            payload["lines"] = len(lines)
        return payload

    def _room_stats(self) -> dict[str, Any] | None:
        """What room matching did, or ``None`` when it was not enabled.

        Read from disk rather than held in memory, so a run served from cache still
        reports it. ``None`` rather than a null field keeps reports from dry runs —
        and from runs made before this feature existed — byte-for-byte unchanged.
        """
        if not self.job.room_matching:
            return None
        fallback: dict[str, Any] = {"mode": self.job.room, "applied": False}
        path = self.job_dir / _ROOM_NAME
        if not path.is_file():
            return fallback
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return fallback
        return payload if isinstance(payload, dict) else fallback

    def _overlapped_ids(self) -> list[int]:
        """Segment ids the overlap stage flagged, tolerant of a malformed file."""
        stats = self._overlap_stats() or {}
        ids: list[int] = []
        for value in stats.get("overlapped_ids") or []:
            try:
                ids.append(int(value))
            except (TypeError, ValueError):
                continue
        return ids

    @staticmethod
    def _read_id_text(path: Path, *, nested: str | None = None) -> dict[int, str]:
        """Read a ``{id: text}`` sidecar, optionally nested under one key."""
        if not path.is_file():
            return {}
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if nested is not None:
            payload = payload.get(nested) if isinstance(payload, dict) else None
        if not isinstance(payload, dict):
            return {}
        overrides: dict[int, str] = {}
        for key, value in payload.items():
            try:
                overrides[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return overrides

    def _prefit_overrides(self) -> dict[int, str]:
        """The pre-synthesis rewrites, i.e. the wording that was actually spoken."""
        return self._read_id_text(self.job_dir / _PREFIT_NAME, nested="lines")

    def _spoken_overrides(self) -> dict[int, str]:
        """Lines that were rewritten, so subtitles match what is heard.

        Two passes can rewrite a line: the pre-synthesis word check and the measured
        repair that follows it. The repair wins where both touched the same line,
        because it was written against real audio rather than a prediction.
        """
        overrides = self._prefit_overrides()
        overrides.update(self._read_id_text(self.job_dir / _SPOKEN_NAME))
        return overrides

    def _prefit_stats(self) -> dict[str, Any]:
        """What the pre-synthesis word check did, or zeros when it could not run."""
        default: dict[str, Any] = {
            "attempted": False,
            "offenders": 0,
            "improved": 0,
            "llm_calls": 0,
            "words_before": 0,
            "words_after": 0,
            "words_kept": 1.0,
        }
        path = self.job_dir / _PREFIT_NAME
        if not path.is_file():
            return default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
        if not isinstance(payload, dict):
            return default
        stats = {**default, **payload}
        before = stats.get("words_before") or 0
        stats["words_kept"] = (
            round(stats.get("words_after", 0) / before, 3) if before else 1.0
        )
        stats.pop("lines", None)
        return stats

    def _packed_stats(self) -> dict[str, Any] | None:
        """What the ASR-timing repair changed, or ``None`` when nothing needed it.

        Absent rather than empty for the usual case, so a job whose boundaries were
        already plausible reports exactly what it did before the repair existed.
        """
        if not self._packed_runs:
            return None
        return {
            "runs": len(self._packed_runs),
            "lines": int(sum(run["lines"] for run in self._packed_runs)),
            "worst_wps": max(run["worst_wps"] for run in self._packed_runs),
            "seconds_added": round(
                sum(run["now_seconds"] - run["was_seconds"] for run in self._packed_runs),
                2,
            ),
            "detail": self._packed_runs,
        }

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
                    device=self.job.primary_device,
                    language=language_code_for_name(self.job.source_language),
                    no_speech_threshold=self.job.no_speech_threshold,
                )
            save_transcript(transcript, out)
            return transcript

        result = self._stage("transcribe", [out], build)
        return load_transcript(out) if result is None else result

    # -- speakers ---------------------------------------------------------- #

    @staticmethod
    def _speaker_of(segment: Segment) -> str:
        """The segment's speaker, defaulting to the single-voice label."""
        return segment.speaker or _SINGLE_SPEAKER

    def _speaker_order(self, segments: Sequence[Segment]) -> list[str]:
        """Speakers in order of first appearance, so ids and reports are stable."""
        order: list[str] = []
        for segment in segments:
            name = self._speaker_of(segment)
            if name not in order:
                order.append(name)
        return order

    def _stage_diarize(
        self, audio_path: Path, transcript: Transcript
    ) -> dict[int, str]:
        """Assign a speaker to every segment, then annotate the transcript.

        Cached on its own: the labels depend on the audio and the threshold, not on
        the translation or the chosen voice, so a rerun that only changes a blend
        setting reuses them.
        """
        out = self.job_dir / _SPEAKERS_NAME

        def build() -> dict[int, str]:
            segment_ids = [segment.id for segment in transcript.segments]
            labeler = self._resolve_diarizer()
            if labeler is None or isinstance(labeler, NullDiarizer):
                # No real labeler: one voice, and no need to read the audio at all.
                labels = single_speaker_labels(segment_ids)
            else:
                samples, sample_rate = media.load_audio(audio_path)
                labels = labeler.label(
                    segment_ids,
                    [(segment.start, segment.end) for segment in transcript.segments],
                    samples,
                    sample_rate,
                )

            counts: dict[str, int] = {}
            for name in labels.values():
                counts[name] = counts.get(name, 0) + 1
            self.log.info(
                "Diarization (%s): %d speaker(s) over %d segment(s) - %s",
                getattr(labeler, "name", "none"),
                len(counts),
                len(labels),
                ", ".join(f"{name} {count}" for name, count in sorted(counts.items())),
            )
            out.write_text(
                json.dumps({str(k): v for k, v in labels.items()}, indent=2),
                encoding="utf-8",
            )
            return labels

        result = self._stage("diarize", [out], build)
        if result is None:
            payload = json.loads(out.read_text(encoding="utf-8"))
            labels = {int(key): str(value) for key, value in payload.items()}
        else:
            labels = result

        # Annotated here so both the fresh and the cached path leave the transcript
        # in the same state for every later stage.
        for segment in transcript.segments:
            segment.speaker = labels.get(segment.id, _SINGLE_SPEAKER)
        return labels

    def _stage_overlap(
        self, audio_path: Path, transcript: Transcript
    ) -> dict[int, bool]:
        """Flag the segments that carry more than one voice.

        Only meaningful with a real labeler: with diarization off - or unavailable,
        which also collapses every line to one voice - there is no second voice to
        find, and the stage is skipped rather than writing an empty file that would
        make single-voice reports look different.

        Best-effort by construction. A segment whose windows cannot be embedded is
        left unflagged, and a flagged one keeps the dominant-speaker treatment it
        would have had anyway, so a miss costs nothing that was not already lost.
        """
        labeler = self._resolve_diarizer()
        if labeler is None or isinstance(labeler, NullDiarizer):
            return {}
        detect = getattr(labeler, "overlaps", None)
        if detect is None:
            # A labeler that cannot detect overlap leaves the segments unflagged.
            return {}

        out = self.job_dir / _OVERLAP_NAME

        def build() -> dict[int, bool]:
            samples, sample_rate = media.load_audio(audio_path)
            flags = detect(
                [segment.id for segment in transcript.segments],
                [(segment.start, segment.end) for segment in transcript.segments],
                # The labels decide who the speakers are; overlap only asks
                # whether a window fits a different one of them better.
                {
                    segment.id: self._speaker_of(segment)
                    for segment in transcript.segments
                },
                samples,
                sample_rate,
            )
            overlapped = [
                segment.id for segment in transcript.segments if flags.get(segment.id)
            ]
            seconds = sum(
                segment.duration
                for segment in transcript.segments
                if flags.get(segment.id)
            )
            total = sum(segment.duration for segment in transcript.segments)
            if overlapped:
                self.log.warning(
                    "Overlapping speech detected in %d of %d segment(s): %s. Each is "
                    "dubbed in the voice of its dominant speaker, so the second voice "
                    "is not separately dubbed (see report.json -> overlap).",
                    len(overlapped),
                    len(transcript.segments),
                    ", ".join(f"#{segment_id}" for segment_id in overlapped),
                )
            out.write_text(
                json.dumps(
                    {
                        "overlapped_ids": overlapped,
                        "segments": len(overlapped),
                        "seconds": round(seconds, 2),
                        "speech_ratio": round(seconds / total, 4) if total > 0 else 0.0,
                        "mode": "dominant",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return flags

        result = self._stage("overlap", [out], build)
        if result is not None:
            flags = result
        else:
            flags = {segment_id: True for segment_id in self._overlapped_ids()}

        # Annotated here so both the fresh and the cached path leave the transcript
        # in the same state, exactly as the diarization stage does for speakers.
        for segment in transcript.segments:
            segment.overlapped = bool(flags.get(segment.id, False))
        return flags

    # -- overlapping speech ------------------------------------------------ #

    def _overlap_lines(self) -> list[dict[str, Any]]:
        """The sub-lines the overlap stage split out, or ``[]`` when it did not run."""
        path = self.job_dir / _OVERLAP_LINES_NAME
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        records = payload.get("segments") if isinstance(payload, dict) else None
        return records if isinstance(records, list) else []

    def _dub_lines(self, transcript: Transcript) -> list[Segment]:
        """The lines to dub: ASR segments with flagged overlaps split per voice.

        The dominant voice's line reuses its parent's id, so donor, timing and
        report keys stay stable; the second voice becomes an overlay line with a
        fresh id. Donor selection and the source subtitles keep using the original
        ASR segments, so a donor is never taken from a separated - and therefore
        filtered - stream.
        """
        records = self._overlap_lines()
        if not records:
            return list(transcript.segments)

        by_parent: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            try:
                by_parent.setdefault(int(record["parent_id"]), []).append(record)
            except (KeyError, TypeError, ValueError):
                continue
        if not by_parent:
            return list(transcript.segments)

        lines: list[Segment] = []
        next_id = max((segment.id for segment in transcript.segments), default=-1) + 1
        for segment in transcript.segments:
            owned = by_parent.get(segment.id)
            if not owned:
                lines.append(segment)
                continue
            for index, record in enumerate(
                sorted(owned, key=lambda item: item.get("start", 0.0))
            ):
                line_id = segment.id if index == 0 else next_id
                if index > 0:
                    next_id += 1
                lines.append(
                    Segment(
                        id=line_id,
                        start=float(record.get("start", segment.start)),
                        end=float(record.get("end", segment.end)),
                        text=str(record.get("text", "")),
                        speaker=str(
                            record.get("speaker") or segment.speaker or _SINGLE_SPEAKER
                        ),
                        parent_id=segment.id,
                        overlay=index > 0,
                    )
                )
        return lines

    @staticmethod
    def _embed_stream(labeler: Any, stream: np.ndarray, sample_rate: int) -> Any:
        """Embed one separated stream, or ``None`` when it cannot be embedded."""
        embedder = getattr(labeler, "embedder", None)
        if embedder is None:
            return None
        data = np.asarray(stream, dtype=np.float32).reshape(-1)
        duration = data.size / float(sample_rate or 1)
        if duration <= 0:
            return None
        vectors = embedder.embed(data, sample_rate, [(0.0, duration)])
        return vectors[0] if vectors else None

    def _transcribe_streams(self, paths: Sequence[Path]) -> list[Transcript]:
        """Re-transcribe separated streams, loading the ASR model once."""
        if self._stream_transcriber is not None:
            return [self._stream_transcriber(path) for path in paths]
        if self._transcriber is not None:
            return [self._transcriber(path) for path in paths]
        model_dir = resolve_model("whisper_large_v3", self.settings.models_dir)
        return transcribe_many(
            paths,
            model_dir,
            device=self.job.primary_device,
            language=language_code_for_name(self.job.source_language),
            no_speech_threshold=self.job.no_speech_threshold,
        )

    def _stage_overlap_separate(
        self, audio_path: Path, transcript: Transcript
    ) -> list[dict[str, Any]]:
        """Split each flagged segment into one stream per voice and re-transcribe it.

        Gated on the overlap flags, so separation only runs on the handful of
        segments that need it. Each stream is matched to a speaker by comparing its
        embedding to the diarization centroids; a segment whose streams cannot be
        told apart keeps its dominant-only treatment. The second voice's words are
        not in Whisper's one-utterance output, which is why each stream is
        re-transcribed rather than reusing the original text.

        Best-effort by construction: a segment that cannot be separated, matched or
        re-transcribed is left exactly as it was, so this stage cannot make a job
        worse than the dominant-only path.
        """
        labeler = self._resolve_diarizer()
        if (
            not self.job.overlap_separation
            or labeler is None
            or isinstance(labeler, NullDiarizer)
            or getattr(labeler, "overlaps", None) is None
            # Stream-to-speaker matching needs the embedder behind the labels; a
            # labeler that only produces labels cannot separate.
            or getattr(labeler, "centroids", None) is None
        ):
            return []
        flagged = [segment for segment in transcript.segments if segment.overlapped]
        if not flagged:
            return []
        separator = self._resolve_speaker_separator()
        if isinstance(separator, NullSpeakerSeparator):
            return []

        out = self.job_dir / _OVERLAP_LINES_NAME

        def build() -> list[dict[str, Any]]:
            samples, sample_rate = media.load_audio(audio_path)
            centroids = labeler.centroids(
                [segment.id for segment in transcript.segments],
                [(segment.start, segment.end) for segment in transcript.segments],
                {
                    segment.id: self._speaker_of(segment)
                    for segment in transcript.segments
                },
                samples,
                sample_rate,
            )
            if len(centroids) < 2:
                # Nothing to match a second stream against, so every assignment
                # below is ``None`` and the artifact records an empty split.
                self.log.info(
                    "Overlap separation: fewer than two speakers could be embedded, "
                    "so no line can be split."
                )

            tmp_dir = self.job_dir / _TMP_DIR
            tmp_dir.mkdir(parents=True, exist_ok=True)
            pending: list[tuple[Segment, str, Path]] = []
            for segment in flagged:
                first = max(0, int(round(segment.start * sample_rate)))
                last = min(samples.size, int(round(segment.end * sample_rate)))
                if last - first <= 0:
                    continue
                try:
                    streams = separator.separate(
                        samples[first:last], sample_rate, num_speakers=2
                    )
                    if len(streams) != 2 or any(
                        np.asarray(stream).size == 0 for stream in streams
                    ):
                        continue
                    embeddings = [
                        self._embed_stream(labeler, stream, sample_rate)
                        for stream in streams
                    ]
                    assignment = assign_streams(
                        embeddings, self._speaker_of(segment), centroids
                    )
                    if assignment is None:
                        # The two streams cannot be told apart: most likely a
                        # false-positive flag, so keep the dominant-only treatment.
                        continue
                    for index, (stream, speaker) in enumerate(zip(streams, assignment)):
                        path = tmp_dir / f"overlap_{segment.id:04d}_{index}.wav"
                        media.write_audio(path, stream, sample_rate)
                        pending.append((segment, speaker, path))
                except Exception as exc:  # noqa: BLE001 - one bad segment must not fail a job
                    self.log.warning(
                        "Could not separate overlapping speech in segment %d: %s. "
                        "Keeping the dominant-speaker treatment for it.",
                        segment.id,
                        exc,
                    )

            grouped: dict[int, list[dict[str, Any]]] = {}
            if pending:
                transcripts = self._transcribe_streams(
                    [path for _segment, _speaker, path in pending]
                )
                for (segment, speaker, _path), sub in zip(pending, transcripts):
                    if sub.is_empty():
                        # A stream that recovered nothing cannot supply a line.
                        continue
                    start = segment.start + min(item.start for item in sub.segments)
                    end = segment.start + max(item.end for item in sub.segments)
                    text = " ".join(
                        item.text.strip() for item in sub.segments if item.text.strip()
                    )
                    grouped.setdefault(segment.id, []).append(
                        {
                            "parent_id": segment.id,
                            "speaker": speaker,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "text": text,
                        }
                    )

            records: list[dict[str, Any]] = []
            for segment in flagged:
                lines = grouped.get(segment.id)
                # Both voices must have been recovered; a single recovered voice is
                # what the dominant-only path already delivers.
                if lines is not None and len(lines) == 2:
                    records.extend(sorted(lines, key=lambda item: item["start"]))
            records.sort(key=lambda item: item["start"])

            out.write_text(
                json.dumps(
                    {
                        "mode": "separate",
                        "segments": records,
                        "separated": len({item["parent_id"] for item in records}),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.log.info(
                "Overlap separation: %d flagged segment(s) split into %d line(s).",
                len({item["parent_id"] for item in records}),
                len(records),
            )
            return records

        result = self._stage("overlap-separate", [out], build)
        return self._overlap_lines() if result is None else result

    def _probe_rates(self) -> dict[str, float]:
        """Re-read the delivery-rate probe, so a cached run reports its own numbers."""
        path = self.job_dir / _RATE_NAME
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(name): float(value) for name, value in payload.items()}

    def _stage_rate_probe(
        self,
        audio_path: Path,
        transcript: Transcript,
        accents: Mapping[str, tuple[Any, str]],
        blends: Mapping[str, Path],
    ) -> dict[str, float]:
        """Measure how fast the cloned voice speaks, in words per second.

        The word budget decides whether a translation can fit at all, so it has to
        be sized from the voice that will speak it. The accent clip's own rate is
        not that number: measured on a real job the clip read 2.97 words/s while
        the clone delivered 3.65, and a ceiling a third too tight is one the model
        simply ignores. One probe render per speaker buys the measurement; it is a
        single line of the accent text, so it is noise beside the job it sizes.
        """
        out = self.job_dir / _RATE_NAME
        speakers = self._speaker_order(transcript.segments)

        def build() -> dict[str, float]:
            runner = self._runner()
            donors = self._resolve_donors(audio_path, transcript)
            measured: dict[str, float] = {}
            #: One prompt is one voice, so one measurement serves every speaker who
            #: shares it - a pinned reference voice is measured once, not once per
            #: speaker.
            by_prompt: dict[str, float] = {}
            tasks: list[Task] = []
            owners: list[tuple[str, str]] = []
            for name in speakers:
                spec = self._voice_spec(name, donors, blends)
                if spec.key in by_prompt:
                    measured[name] = by_prompt[spec.key]
                    continue
                text = _probe_text(accents.get(name), self.job.target_language)
                if not text:
                    self.log.warning(
                        "No %s text to probe the delivery rate with; the word "
                        "budget falls back to the accent clip's own rate.",
                        self.job.target_language,
                    )
                    continue
                # The prompt *is* the voice, so the probe has to be rendered
                # through the same one the lines will be. The runner guarantees
                # that: a worker that already holds this voice reuses it.
                tasks.append(
                    Task(
                        kind=PROBE,
                        spec=spec,
                        language=self.job.target_language,
                        text=text,
                    )
                )
                owners.append((name, text))
            for (name, text), result in zip(owners, runner.run(tasks)):
                if result.rate <= 0:
                    self.log.warning("Rate probe for %s produced no audio.", name)
                    continue
                by_prompt[result.key] = measured[name] = round(result.rate, 3)
                self.log.info(
                    "Delivery rate for %s: %.2f words/s (%.1fs of probe audio).",
                    name,
                    result.rate,
                    len(text.split()) / result.rate,
                )
            out.write_text(json.dumps(measured, indent=2), encoding="utf-8")
            return measured

        result = self._stage("rate-probe", [out], build)
        return self._probe_rates() if result is None else result

    def _delivery_rate(
        self, rates: Mapping[str, float], speaker: str | None = None
    ) -> float | None:
        """Words per second the target voice speaks, for one line's budget.

        Prefers the probe's measurement for the speaker who owns the line. A
        speaker the probe skipped takes the slowest measured voice, so a line is
        never budgeted for a faster delivery than its own. Only when no probe ran
        at all does it fall back to the accent clip's rate, which is what this
        used to rely on and is a measured ~20% under-estimate.
        """
        if self.job.length_budget != "words":
            return None
        if speaker is not None and rates.get(speaker):
            return float(rates[speaker])
        if rates:
            return float(min(rates.values()))
        return self._clip_speech_rate()

    def _clip_speech_rate(self) -> float | None:
        """Words per second read off a reference clip, the pre-probe fallback.

        Cheap (no synthesis) but it describes the *clip*, not the clone: the model
        reads a passage at its own pace, so the two disagree. Kept as the fallback
        for the one case the probe cannot cover - no target-language text to read.
        """
        if self.job.accent_mode == "native":
            # The generated anchor is target-language speech, so it sizes the
            # budget exactly as a supplied accent clip would.
            if self._primary_accent is None:
                self.log.info(
                    "No accent anchor is available yet, so no word budget can be "
                    "derived; translating with duration hints only."
                )
                return None
            clip, text = self._primary_accent
        elif self.job.accent_mode == "reference-accent":
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
        self.log.info("Clip speaking rate: %.2f words/s (from %s)", rate, clip)
        return rate

    def _stage_translate(
        self,
        transcript: Transcript,
        lines: Sequence[Segment],
        rates: Mapping[str, float] | None = None,
    ) -> dict[int, str]:
        """Translate every line to the target language.

        ``lines`` may include separated overlap lines, so the second voice's words
        are translated too. ``rates`` is the delivery-rate probe, which is what
        turns a line's window into a word ceiling its slot can actually hold.
        """
        out = self.job_dir / _SCRIPT_NAME

        def build() -> dict[int, str]:
            requests = []
            for index, segment in enumerate(lines):
                rate = self._delivery_rate(rates or {}, self._speaker_of(segment))
                requests.append(
                    TranslationRequest(
                        id=segment.id,
                        text=segment.text,
                        duration=segment.duration,
                        # The ceiling covers the gap before the next line as well
                        # as the slot, because the mixer may spill into it. Left
                        # ``None`` when no rate is known, which keeps the prompt
                        # identical to a run made before budgets existed.
                        max_words=(
                            _budget_words(
                                _available_seconds(lines, index),
                                rate,
                                self.job.budget_speedup,
                            )
                            if rate
                            else None
                        ),
                    )
                )
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
                            for segment in lines
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

    def _stage_prefit(
        self,
        lines: Sequence[Segment],
        script: dict[int, str],
        rates: Mapping[str, float],
    ) -> dict[int, str]:
        """Rewrite the lines that cannot fit their window, before synthesising them.

        The mixer may compress a line to the floor and spill it into the gap before
        the next one, and no further: past that, words are deleted. The only honest
        answer left is a shorter sentence, and asking for it here - with the word
        ceiling and the predicted spoken length, no audio needed - is what keeps the
        mixer from ever having to choose. The measured repair after synthesis stays
        as the second pass, because it can see what was really spoken.

        Returns:
            The script with the accepted rewrites applied.
        """
        out = self.job_dir / _PREFIT_NAME

        def build() -> dict[int, str]:
            # A copy: ``03_script.json`` is the translation as delivered by the
            # model, and the caller keeps using that dict for provenance.
            fitted = dict(script)
            stats: dict[str, Any] = {
                "attempted": False,
                "offenders": 0,
                "improved": 0,
                "llm_calls": 0,
                "words_before": 0,
                "words_after": 0,
            }
            translator = self._resolve_translator()
            requests: list[ShortenRequest] = []
            for index, segment in enumerate(lines):
                rate = self._delivery_rate(rates, self._speaker_of(segment))
                if not rate:
                    continue
                window = _available_seconds(lines, index)
                text = fitted.get(segment.id, "")
                words = len(text.split())
                budget = _budget_words(window, rate, self.job.budget_speedup)
                if not text.strip() or words <= budget:
                    continue
                requests.append(
                    ShortenRequest(
                        id=segment.id,
                        text=text,
                        source=segment.text,
                        # Predicted, not measured: nothing has been spoken yet.
                        spoken_seconds=round(words / rate, 2),
                        available_seconds=round(window, 2),
                        max_words=budget,
                    )
                )
            stats["offenders"] = len(requests)
            stats["words_before"] = sum(len(r.text.split()) for r in requests)
            stats["words_after"] = stats["words_before"]

            if requests and callable(getattr(translator, "shorten", None)):
                stats["attempted"] = True
                try:
                    rewritten = translator.shorten(
                        requests, target_language=self.job.target_language
                    )
                    stats["llm_calls"] = 1
                except Exception as exc:  # noqa: BLE001 - best effort, as after synth
                    self.log.warning(
                        "Could not reword %d overlong line(s) before synthesis "
                        "(%s). They are synthesised as translated and repaired "
                        "afterwards if possible.",
                        len(requests),
                        exc,
                    )
                    rewritten = {}
                for request in requests:
                    candidate = str(rewritten.get(request.id) or "").strip()
                    before = len(request.text.split())
                    after = len(candidate.split())
                    # A rewrite is only ever adopted when it measurably helps: a
                    # "shorter" line that is not shorter is a worse translation.
                    if not candidate or after >= before:
                        continue
                    fitted[request.id] = candidate
                    stats["improved"] += 1
                    stats["words_after"] -= before - after
            elif requests:
                self.log.info(
                    "Translator %s cannot reword lines, so the pre-synthesis check "
                    "is skipped; %d line(s) may lose their ending.",
                    type(translator).__name__,
                    len(requests),
                )

            payload = dict(stats)
            payload["lines"] = {
                str(request.id): fitted[request.id]
                for request in requests
                if fitted.get(request.id) != request.text
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if stats["attempted"]:
                self.log.info(
                    "Pre-fit: reworded %d of %d overlong line(s) (%d of %d words "
                    "kept) before synthesis.",
                    stats["improved"],
                    stats["offenders"],
                    stats["words_after"],
                    stats["words_before"],
                )
            return fitted

        result = self._stage("prefit", [out], build)
        if result is not None:
            return result
        fitted = dict(script)
        fitted.update(self._prefit_overrides())
        return fitted

    # -- accent source ----------------------------------------------------- #

    def _speaker_rate_bands(self, transcript: Transcript) -> dict[str, str]:
        """Rate band per speaker, measured from that speaker's own lines."""
        bands: dict[str, str] = {}
        for name in self._speaker_order(transcript.segments):
            owned = [
                segment
                for segment in transcript.segments
                if self._speaker_of(segment) == name
            ]
            rate = measure_rate(owned)
            bands[name] = rate_band(rate)
            self.log.info(
                "Measured %s speaking rate: %.2f words/s (%s)",
                name,
                rate,
                bands[name],
            )
        return bands

    def _speaker_genders(
        self, audio_path: Path, transcript: Transcript
    ) -> dict[str, str]:
        """Coarse gender per speaker, from the donor clip when there is one.

        The donor is a clean single-speaker window, so it is a better sample than
        the whole track; when a reference file pins the voice there is no donor and
        the speaker's own segments are used instead.
        """
        donors = self._resolve_donors(audio_path, transcript)
        genders: dict[str, str] = {}
        samples: np.ndarray | None = None
        sample_rate = 0
        for name in self._speaker_order(transcript.segments):
            donor, _info = donors.get(name, (None, {}))
            if donor is not None:
                genders[name] = measure_gender(donor.samples, donor.sample_rate)
                self.log.info("Estimated %s voice gender: %s", name, genders[name])
                continue
            if samples is None:
                samples, sample_rate = media.load_audio(audio_path)
            owned = [
                segment
                for segment in transcript.segments
                if self._speaker_of(segment) == name
            ]
            chunks = [
                samples[max(0, int(segment.start * sample_rate)) : int(segment.end * sample_rate)]
                for segment in owned
            ]
            joined = (
                np.concatenate(chunks)
                if chunks
                else np.zeros(0, dtype=np.float32)
            )
            genders[name] = measure_gender(joined, sample_rate)
            self.log.info("Estimated %s voice gender: %s", name, genders[name])
        return genders

    def _primary_speaker(self, transcript: Transcript) -> str | None:
        """The speaker with the most speech, or ``None`` for an empty transcript."""
        speakers = self._speaker_order(transcript.segments)
        if not speakers:
            return None
        return max(
            speakers,
            key=lambda name: sum(
                segment.duration
                for segment in transcript.segments
                if self._speaker_of(segment) == name
            ),
        )

    def _stage_accent_anchor(
        self, audio_path: Path, transcript: Transcript
    ) -> dict[str, tuple[Any, str]]:
        """Resolve the accent clip each voice borrows pronunciation from.

        ``reference-accent`` uses the supplied clip for every speaker. ``native``
        uses a generated per-language anchor, or - with ``--accent-match`` - a
        per-speaker anchor whose delivery fits the source voice. The other modes
        borrow no accent and resolve to nothing.

        The anchor is generated only on a cache miss, so a warm cache never loads
        the VoiceDesign checkpoint at all.
        """
        speakers = self._speaker_order(transcript.segments)

        if self.job.accent_mode == "reference-accent":
            assert self.job.accent_audio is not None  # guaranteed by JobConfig
            text = self.job.accent_text or ""
            self._accents = {name: (self.job.accent_audio, text) for name in speakers}
            self._accent_records = {
                name: {
                    "source": "reference-audio",
                    "path": str(self.job.accent_audio),
                    "transcript_provided": bool(text.strip()),
                }
                for name in speakers
            }
            self._primary_accent = (self.job.accent_audio, text) if speakers else None
            self._primary_speaker_name = self._primary_speaker(transcript)
            return dict(self._accents)

        if self.job.accent_mode != "native":
            self._accents, self._accent_records, self._primary_accent = {}, {}, None
            self._primary_speaker_name = None
            return {}

        anchor_root = anchors_dir(self.settings.models_dir)
        matched = self.job.accent_match != "off"
        bands = self._speaker_rate_bands(transcript) if matched else {}
        genders = (
            self._speaker_genders(audio_path, transcript)
            if self.job.accent_match == "gender"
            else {}
        )

        accents: dict[str, tuple[Any, str]] = {}
        records: dict[str, dict[str, Any]] = {}
        for name in speakers:
            spec = AnchorSpec(
                self.job.target_language,
                rate_band=bands.get(name) if matched else None,
                gender=genders.get(name) if matched else None,
            )
            wav, txt = ensure_anchor(anchor_root, spec, self._resolve_anchor_engine)
            accents[name] = (wav, txt.read_text(encoding="utf-8"))
            records[name] = {
                "source": "anchor-matched" if matched else "anchor",
                "path": str(wav),
                "language": spec.language,
                "rate_band": spec.rate_band,
                "gender": spec.gender,
                "transcript_provided": True,
            }
            self.log.info(
                "Accent anchor for %s: %s%s",
                name,
                wav.name,
                (
                    f" (rate {spec.rate_band}, gender {spec.gender or 'u'})"
                    if matched
                    else ""
                ),
            )

        self._accents = accents
        self._accent_records = records
        primary = self._primary_speaker(transcript)
        self._primary_speaker_name = primary
        self._primary_accent = accents.get(primary) if primary is not None else None
        return dict(accents)

    def _blend_stats(self) -> dict[str, Any] | None:
        """How the blended reference was chosen, or ``None`` when it was not used.

        Re-derived from ``blend.json`` rather than held in memory, so a run served
        from cache still reports the strategy behind the voice it is reusing.
        """
        if self.job.accent_mode not in ("native", "reference-accent"):
            return None
        if self.job.accent_blend_mode != "two-stage":
            return {"strategy": "single-stage"}
        strategy: dict[str, Any] = {"strategy": "two-stage"}
        path = self.job_dir / _BLEND_INFO_NAME
        if not path.is_file():
            return strategy
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return strategy
        return payload if isinstance(payload, dict) else strategy

    def _resolve_donors(
        self, audio_path: Path, transcript: Transcript
    ) -> dict[str, tuple[DonorClip | None, dict[str, Any]]]:
        """Resolve a timbre source per speaker.

        Each voice gets its own donor - the best segment *that speaker* delivered -
        so a line is cloned from the person who actually said it rather than from
        whoever spoke longest in the video. Resolved once because both the blend and
        the synthesis stages need it, and the speech stem is not free to load twice.
        """
        if self._donors_resolved is None:
            resolved: dict[str, tuple[DonorClip | None, dict[str, Any]]] = {}
            samples: np.ndarray | None = None
            sample_rate = 0
            has_text = bool((self.job.ref_text or "").strip())
            for name in self._speaker_order(transcript.segments):
                if self.job.ref_audio is not None:
                    # An explicit reference voice replaces the video's speaker for
                    # every line, so there is nothing per-speaker to choose.
                    resolved[name] = (
                        None,
                        {
                            "speaker": name,
                            "source": "reference-audio",
                            "path": str(self.job.ref_audio),
                            "transcript_provided": has_text,
                        },
                    )
                    continue
                owned = [
                    segment
                    for segment in transcript.segments
                    if self._speaker_of(segment) == name
                ]
                if samples is None:
                    samples, sample_rate = media.load_audio(audio_path)
                donor = select_donor(samples, sample_rate, owned)
                self.log.info(
                    "Auto-selected donor for %s: segment %d (%.1fs)",
                    name,
                    donor.segment_id,
                    donor.duration,
                )
                resolved[name] = (
                    donor,
                    {
                        "speaker": name,
                        "source": "auto",
                        "segment_id": donor.segment_id,
                        "start": round(donor.start, 2),
                        "end": round(donor.end, 2),
                        "duration": round(donor.duration, 2),
                        "transcript": donor.text,
                    },
                )
            self._donors_resolved = resolved
        return {
            name: (donor, dict(info))
            for name, (donor, info) in self._donors_resolved.items()
        }

    def _stage_blend(
        self,
        audio_path: Path,
        transcript: Transcript,
        accents: Mapping[str, tuple[Any, str]] | None = None,
    ) -> dict[str, Path]:
        """Render a blended reference per speaker, for the two-stage accent path.

        Returns speaker to clip path, empty when the configured mode needs none.
        Each voice is reconciled with its own accent clip but its own embedding, so
        codes and speaker embedding agree per voice. Cached as a set, because these
        clips fix the voice of the whole video: regenerating them would make an
        otherwise identical rerun sound like different people.
        """
        if self.job.accent_mode not in ("native", "reference-accent"):
            return {}
        if self.job.accent_blend_mode != "two-stage":
            return {}

        resolved = dict(accents) if accents is not None else dict(self._accents)
        speakers = self._speaker_order(transcript.segments)
        outputs = _blend_paths(self.job_dir, speakers)
        info_path = self.job_dir / _BLEND_INFO_NAME

        def build() -> dict[str, Path]:
            runner = self._runner()
            donors = self._resolve_donors(audio_path, transcript)
            target_ratio = self.job.resolved_blend_ratio
            # One voice per task. The passes for different speakers share nothing,
            # so this is the parallel axis that needs no splitting at all - and on
            # the job where the blend costs most (a voice apiece for five people)
            # it is also the one that pays.
            tasks = [
                Task(
                    kind=BLEND,
                    spec=self._voice_spec(name, donors, {}, resolved),
                    language=self.job.target_language,
                    takes=self.job.resolved_blend_takes,
                    iters=self.job.resolved_blend_iters,
                    ratio=target_ratio,
                )
                for name in speakers
            ]
            rendered: dict[str, Path] = {}
            stats: dict[str, Any] = {}
            for name, result in zip(speakers, runner.run(tasks)):
                blended = result.blend
                out = outputs[name]
                out.parent.mkdir(parents=True, exist_ok=True)
                media.write_audio(out, blended.samples, blended.sample_rate)
                rendered[name] = out
                stats[name] = {
                    "strategy": "two-stage",
                    "takes": blended.takes,
                    "iters": blended.iters,
                    "target_ratio": target_ratio,
                    "duration": round(blended.duration, 2),
                    "timbre_similarity": _json_similarity(blended.timbre_similarity),
                    "accent_similarity": _json_similarity(blended.accent_similarity),
                    "donor_ratio": _json_similarity(blended.donor_ratio),
                    "margin": _json_similarity(blended.margin),
                }
                self.log.info(
                    "Blended reference for %s: %s (%.1fs, %d take(s), %d pass(es), "
                    "donor ratio %.2f)",
                    name,
                    out.name,
                    blended.duration,
                    blended.takes,
                    blended.iters,
                    blended.donor_ratio,
                )
            # Written beside the WAVs it describes, so the pair can never drift.
            info_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            return rendered

        result = self._stage("blend", [info_path, *outputs.values()], build)
        if result is not None:
            return result
        return {name: path for name, path in outputs.items() if path.is_file()}

    def _stage_synthesize(
        self,
        audio_path: Path,
        transcript: Transcript,
        lines: Sequence[Segment],
        script: dict[int, str],
        tmp_dir: Path,
        blends: Mapping[str, Path] | None = None,
    ) -> dict[int, Path]:
        """Clone each speaker's voice and synthesise one file per line.

        ``blends`` maps speaker to the stage-1 clip from :meth:`_stage_blend`. Where
        a speaker has one, that speaker's lines are cloned from it, so the clip's
        codes and its speaker embedding cannot disagree. ``lines`` may include
        separated overlap lines, each dubbed in the speaker it was matched to.
        """
        blends = blends or {}
        out_dir = self.job_dir / _DUBBED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / _DONE_MARKER
        expected = {segment.id: out_dir / f"seg_{segment.id:04d}.wav" for segment in lines}

        def build() -> dict[int, Path]:
            # Resolved once and shared with the blend stage, so both necessarily
            # agree about which voice belongs to which speaker.
            donors = self._resolve_donors(audio_path, transcript)
            self._write_voices(transcript, donors, blends)

            produced, sample_rate = self._synthesize_all(lines, script, donors, blends)
            if sample_rate > 0:
                wavs = self._refine_overlong(
                    sample_rate,
                    lines,
                    script,
                    [produced.get(segment.id) for segment in lines],
                    tmp_dir,
                    resynthesize=lambda owned, texts: self._retry_lines(
                        owned, texts, donors, blends
                    ),
                )
            else:
                # Nothing was synthesised (every line blank), so there is nothing
                # to measure an overflow against.
                wavs = [produced.get(segment.id) for segment in lines]

            written: dict[int, Path] = {}
            for segment, wav in zip(lines, wavs):
                path = expected[segment.id]
                if wav is None:
                    # Blank line: leave no file so the timing stage emits silence.
                    path.unlink(missing_ok=True)
                    continue
                # Limit (never boost) so a line's dynamic range matches its peers
                # and cannot dominate the mix with a transient peak.
                media.write_audio(
                    path,
                    media.limit_peaks(wav, sample_rate, ceiling=_LINE_CEILING),
                    sample_rate,
                )
                written[segment.id] = path
            # An earlier run of this input may have produced more lines, since ASR
            # re-segmentation changes the count, and nothing removed the clips it
            # left behind. Those orphans get picked up by anything walking the
            # folder, so drop whatever this run did not write.
            for stale in out_dir.glob("seg_*.wav"):
                if stale not in expected.values():
                    stale.unlink()
            # A marker keeps the cache independent of how many lines were blank.
            marker.write_text(str(len(written)), encoding="utf-8")
            return written

        result = self._stage("synthesize", [marker], build)
        if result is not None:
            return result
        return {sid: path for sid, path in expected.items() if path.exists()}

    # -- per-speaker synthesis --------------------------------------------- #

    def _voice_spec(
        self,
        name: str,
        donors: Mapping[str, tuple[DonorClip | None, dict[str, Any]]],
        blends: Mapping[str, Path],
        accents: Mapping[str, tuple[Any, str]] | None = None,
    ) -> VoiceSpec:
        """The facts that build one speaker's clone prompt.

        The rule itself lives in :mod:`pipeline.voice`, because a worker process
        has to build the very same prompt from the very same facts and the two
        must not be able to drift apart. This only gathers them from wherever the
        pipeline happens to keep them.
        """
        donor, info = donors.get(name, (None, {}))
        resolved = dict(accents) if accents is not None else self._accents
        # The accent source is resolved once by the accent stage; a speaker's own
        # matched anchor (or the shared supplied clip) lives in ``_accents``.
        accent_audio, accent_text = resolved.get(
            name, (self.job.accent_audio, self.job.accent_text or "")
        )
        label = ""
        if info:
            label = str(info.get("path") or f"segment {info.get('segment_id')}")
        return VoiceSpec(
            speaker=name,
            accent_mode=self.job.accent_mode,
            accent_audio=accent_audio,
            accent_text=accent_text or "",
            blend=blends.get(name),
            ref_audio=self.job.ref_audio,
            ref_text=self.job.ref_text or "",
            prompt=self.job.ref_prompt,
            prompt_seconds=self.job.ref_prompt_seconds,
            donor=donor,
            timbre_label=label,
            shared=self._prompt_is_shared(blends),
        )

    def _prompt_is_shared(self, blends: Mapping[str, Path]) -> bool:
        """True when one prompt serves every speaker, so it need not be rebuilt.

        A pinned voice with nothing per-speaker feeding the prompt is the same for
        everyone - re-encoding the reference audio once per group would be pure
        waste. A blend clip, by contrast, *is* the per-speaker part.
        """
        return self.job.ref_audio is not None and not blends

    def _synthesize_all(
        self,
        segments: Sequence[Segment],
        texts: Mapping[int, str],
        donors: Mapping[str, tuple[DonorClip | None, dict[str, Any]]],
        blends: Mapping[str, Path],
    ) -> tuple[dict[int, np.ndarray], int]:
        """Synthesise every line, spread over the run's devices.

        Work is grouped by speaker because the prompt *is* the voice: building it
        once per speaker rather than once per line is what makes multi-speaker
        cloning affordable, and it is also the natural unit for the fan-out, since
        one voice's lines cannot influence another's.

        A single-voice video would leave every card but one idle under that split,
        so when there are fewer speakers than devices each speaker's lines are cut
        into pieces as well. The pieces stay contiguous because they are line
        ranges: a worker then reads its text in order, which keeps its log
        readable and one line's overrun predictable.

        Returns:
            ``(segment_id -> samples, sample_rate)``. The rate is handed back
            because the caller writes the files and must not load an engine - on
            another card - just to learn it.
        """
        runner = self._runner()
        grouped = [
            (
                name,
                [segment for segment in segments if self._speaker_of(segment) == name],
            )
            for name in self._speaker_order(segments)
        ]
        width = len(self.job.resolved_devices)
        per_speaker = max(1, math.ceil(width / max(1, len(grouped))))
        tasks: list[Task] = []
        for name, owned in grouped:
            spec = self._voice_spec(name, donors, blends)
            items = [(segment.id, texts.get(segment.id, "")) for segment in owned]
            for piece in split_evenly(items, per_speaker):
                tasks.append(
                    Task(
                        kind=SYNTH,
                        spec=spec,
                        language=self.job.target_language,
                        lines=tuple(piece),
                    )
                )

        produced: dict[int, np.ndarray] = {}
        sample_rate = 0
        for result in runner.run(tasks):
            sample_rate = sample_rate or result.sample_rate
            for segment_id, wav in result.wavs:
                if wav is not None and np.asarray(wav).size:
                    produced[segment_id] = np.asarray(wav, dtype=np.float32).reshape(-1)
        return produced, sample_rate

    def _retry_lines(
        self,
        segments: Sequence[Segment],
        texts: Sequence[str],
        donors: Mapping[str, tuple[DonorClip | None, dict[str, Any]]],
        blends: Mapping[str, Path],
    ) -> list[np.ndarray | None]:
        """Re-synthesise rewritten lines, each in its own speaker's voice."""
        by_id = {segment.id: text for segment, text in zip(segments, texts)}
        produced, _rate = self._synthesize_all(segments, by_id, donors, blends)
        return [produced.get(segment.id) for segment in segments]

    def _write_voices(
        self,
        transcript: Transcript,
        donors: Mapping[str, tuple[DonorClip | None, dict[str, Any]]],
        blends: Mapping[str, Path],
    ) -> None:
        """Record which voice was cloned for each speaker, and from what.

        The per-speaker map is this job's voice bible. The dominant speaker's own
        fields are also copied to the top level, so single-speaker jobs - and
        everything that already reads ``donor.accent`` or ``donor.blend`` - keep
        working unchanged.
        """
        blend_info = self._blend_stats()
        speakers: dict[str, Any] = {}
        for name in self._speaker_order(transcript.segments):
            _donor, info = donors.get(name, (None, {}))
            record: dict[str, Any] = dict(info)
            record["accent_mode"] = self.job.accent_mode
            accent_record = self._accent_records.get(name)
            if accent_record is not None:
                record["accent"] = accent_record
            if isinstance(blend_info, dict):
                # Two-stage records statistics per speaker; single-stage only knows
                # the strategy, which is the same for everyone.
                record["blend"] = blend_info.get(name, blend_info)
            owned = [
                segment
                for segment in transcript.segments
                if self._speaker_of(segment) == name
            ]
            record["segments"] = len(owned)
            record["speech_seconds"] = round(
                sum(segment.duration for segment in owned), 2
            )
            speakers[name] = record

        payload: dict[str, Any] = {
            "speaker_count": len(speakers),
            "speakers": speakers,
        }
        if speakers:
            primary = max(speakers, key=lambda key: speakers[key]["speech_seconds"])
            payload["primary_speaker"] = primary
            payload.update(
                {
                    key: value
                    for key, value in speakers[primary].items()
                    if key not in _SPEAKER_ONLY_KEYS
                }
            )
        # Persisted so the report can state the voices even when the stage is
        # served from cache on a later run.
        (self.job_dir / _DONOR_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- length repair ----------------------------------------------------- #

    def _overflow_seconds(self, raw_seconds: float, index: int) -> float:
        """How far a line of ``raw_seconds`` would overflow its window.

        Mirrors the mix stage - both go through ``fit_ratio`` - so what this predicts
        is close to what the mixer will have to trim. It assumes the full overrun
        allowance is available, which is optimistic for a line the timeline has
        already pushed late; that only means the repair fires slightly less often
        than it could, never that it makes a line worse. The final line has no
        successor to protect, so it may run freely.
        """
        segments = self._segments_for_refine
        segment = segments[index]
        slot = segment.duration
        if slot <= 0 or raw_seconds <= 0:
            return 0.0
        if index + 1 >= len(segments) and not segment.overlay:
            return 0.0
        allowance = 0.0 if segment.overlay else self.job.max_overrun
        window = _available_seconds(segments, index) + allowance
        ratio = fit_ratio(
            slot,
            window,
            raw_seconds,
            min_ratio=self.job.min_stretch,
            max_ratio=self.job.max_stretch,
        )
        return raw_seconds * ratio - window

    def _refine_overlong(
        self,
        sample_rate: int,
        segments: Sequence[Segment],
        script: dict[int, str],
        wavs: list[np.ndarray | None],
        tmp_dir: Path,
        resynthesize: Resynthesize,
    ) -> list[np.ndarray | None]:
        """Rewrite and re-synthesise only the lines that would lose words.

        ``sample_rate`` is the rate the lines were produced at, passed in rather
        than read off an engine: with several workers behind the stage there is no
        single engine here to ask, and the caller already has the number.

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

        segments = list(segments)
        self._segments_for_refine = segments
        spoken: dict[int, str] = {}

        def spoken_seconds_of(wav: np.ndarray | None) -> float:
            """Duration the mixer will see: silence at the ends is not speech."""
            if wav is None or not np.asarray(wav).size:
                return 0.0
            return len(trim_silence(wav, sample_rate)) / float(sample_rate)

        for _pass in range(self.job.refine_max_passes):
            offenders: list[tuple[int, float]] = []
            for index, wav in enumerate(wavs):
                raw = spoken_seconds_of(wav)
                if raw <= 0.0:
                    continue
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
                raw = spoken_seconds_of(wavs[index])
                ratio = fit_ratio(
                    segment.duration,
                    _available_seconds(segments, index)
                    + (0.0 if segment.overlay else self.job.max_overrun),
                    raw,
                    min_ratio=self.job.min_stretch,
                    max_ratio=self.job.max_stretch,
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
            retry_segments = [segments[index] for index, _ in offenders]
            try:
                # Rebuilt per speaker: a rewritten line has to come back in its own
                # voice, which means its own clone prompt - and, when the job has
                # more than one card, its own task on whichever worker is free.
                retried = resynthesize(retry_segments, texts)
            except Exception as exc:  # noqa: BLE001 - repair is best-effort
                self.log.warning("Re-synthesis failed: %s. Keeping existing audio.", exc)
                break

            for (index, overflow), request, text, wav in zip(
                offenders, requests, texts, retried
            ):
                if wav is None or not np.asarray(wav).size:
                    continue
                new_raw = spoken_seconds_of(wav)
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
        lines: Sequence[Segment],
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
            needed: dict[int, float] = {}
            silence_trimmed: dict[int, float] = {}
            segments = lines

            # Lateness is bounded in *total*, not per line. ``build_timeline`` places
            # each line after the previous one's audio, so an uninterrupted run of
            # tight lines would otherwise push the stream steadily later - measured
            # on a real job, 27 lines in a row with the last one 1.9s late. A line
            # that is already late therefore loses its allowance and has its tail
            # cut instead: the sync budget belongs to the line, not to the chain.
            # The cursor mirrors ``build_timeline`` exactly, which is why it is kept
            # here rather than derived afterwards.
            cursor = 0.0
            for index, segment in enumerate(segments):
                path = dubbed.get(segment.id)
                synth: np.ndarray | None = None
                if path is not None:
                    samples, sr = media.load_audio(path)
                    if sr != sample_rate:
                        samples = media.resample(samples, sr, sample_rate)
                    # Dead air at the ends is not speech: it must not be paid for
                    # with tempo (or with reworded text), and left in place it is
                    # silence inside a slot whose first word should be at its start.
                    synth = trim_silence(samples, sample_rate)
                    silence_trimmed[segment.id] = max(
                        0.0,
                        (len(np.asarray(samples).reshape(-1)) - len(synth))
                        / float(sample_rate),
                    )

                start = segment.start if segment.overlay else max(segment.start, cursor)
                allowance = max(
                    0.0, self.job.max_overrun - max(0.0, start - segment.start)
                )
                # An overlay line shares its slot with the dominant line, so it may
                # not spill at all. The last line has nothing after it to protect, so
                # it may run freely - which also means it cannot push anything.
                window = _available_seconds(segments, index)
                if segment.overlay:
                    max_overrun: float | None = 0.0
                elif index + 1 < len(segments):
                    max_overrun = _gap_seconds(segments, index) + allowance
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
                    available=window + (0.0 if segment.overlay else allowance),
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
                    needed[segment.id] = (
                        window / synth_seconds[segment.id]
                        if synth_seconds[segment.id] > 0
                        else 1.0
                    )
                else:
                    trimmed[segment.id] = 0.0
                    synth_seconds[segment.id] = 0.0
                    deficits[segment.id] = 0.0
                    needed[segment.id] = 1.0

                items.append((segment.id, segment.start, fitted))
                if not segment.overlay:
                    # Exactly what build_timeline will place next: a padded line is as
                    # long as its slot, a compressed one as long as it came out.
                    cursor = start + len(fitted) / float(sample_rate)

            overlay_ids = {segment.id for segment in segments if segment.overlay}
            timeline = build_timeline(
                items,
                sample_rate=sample_rate,
                total_duration=total_duration,
                overlay_ids=overlay_ids,
            )
            for placement in timeline.placements:
                placement.ratio = ratios.get(placement.segment_id, 1.0)
                placement.overrun = overruns.get(placement.segment_id, 0.0)
                placement.trimmed = trimmed.get(placement.segment_id, 0.0)
                placement.synth_seconds = synth_seconds.get(placement.segment_id, 0.0)
                placement.deficit = deficits.get(placement.segment_id, 0.0)
                placement.needed_ratio = needed.get(placement.segment_id, 0.0)
                placement.silence_trimmed = silence_trimmed.get(
                    placement.segment_id, 0.0
                )

            notable = sum(1 for value in overruns.values() if value > 0.01)
            if notable:
                self.log.info(
                    "%d line(s) used the gap before the next utterance (%.1fs); "
                    "%.1fs trimmed where it would have desynchronised the video.",
                    notable,
                    sum(overruns.values()),
                    sum(trimmed.values()),
                )

            # Speech is only deleted when nothing else can absorb it, so when it
            # happens it is worth naming: the ids lead straight to the lines whose
            # translation is longer than the floor allows.
            dropped = sorted(
                (p for p in timeline.placements if p.trimmed > 0.01),
                key=lambda p: p.trimmed,
                reverse=True,
            )
            if dropped:
                self.log.warning(
                    "%d line(s) lost speech (floor %.2f): %s",
                    len(dropped),
                    self.job.min_stretch,
                    ", ".join(
                        f"id {p.segment_id} -{p.trimmed:.2f}s "
                        f"(needed {p.needed_ratio:.2f})"
                        for p in dropped[:5]
                    ),
                )

            regions = [(p.start, p.end) for p in timeline.placements]

            # Room matching: put the dub *in* the source's space instead of on top
            # of it. Both the reverb and the room-tone floor come from the separated
            # speech stem, the only material that carries the room.
            room_stats, speech, tone = self._match_room(
                timeline.samples, regions, sample_rate
            )
            if self.job.room_matching:
                # Only written when the feature was asked for, so a dry job leaves
                # no artifact that would make it look like something happened.
                (self.job_dir / _ROOM_NAME).write_text(
                    json.dumps(room_stats, indent=2), encoding="utf-8"
                )

            # Lay the dub over the bed so music and ambience survive. The bed is
            # ducked exactly where dialogue plays, which is what keeps the voice
            # intelligible without muting the soundtrack - but shallower once the
            # dub has been placed in the room, because the room is part of the voice.
            timeline = Timeline(
                samples=duck_and_mix(
                    speech,
                    self._load_bed(sample_rate, tone),
                    regions,
                    sample_rate,
                    duck_db=(
                        self.job.room_duck_db
                        if room_stats["applied"]
                        else self.job.duck_db
                    ),
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
                            "needed_ratio": round(p.needed_ratio, 4),
                            "silence_trimmed": round(p.silence_trimmed, 3),
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
        lines: Sequence[Segment],
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
                    for s in lines
                ],
                self.job_dir / "subs_target.srt",
            )

        dubbed_video = None
        burned_video = None
        if self.job.is_video:
            mode = self.job.subtitles_mode
            dubbed_video = media.mux_audio(
                self.input_path,
                dub_track,
                self.job_dir / "video_dubbed.mp4",
                # Embed the target subtitles softly by default. ``burn`` is applied
                # in a second pass, after the dub's audio is in place, so the
                # re-encoding pass has a finished file to draw over.
                subtitle_srt=(
                    target_srt if mode == "soft" and target_srt is not None else None
                ),
                subtitle_language=self.job.target_language,
            )
            if mode == "burn" and target_srt is not None:
                burned_video = media.burn_subtitles(
                    dubbed_video,
                    target_srt,
                    self.job_dir / "video_burned.mp4",
                    language=self.job.target_language,
                )

        report = self._build_report(timeline, transcript, script, dubbed_video, loudness)
        report["subtitles_mode"] = self.job.subtitles_mode
        (self.job_dir / _REPORT_NAME).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self.log.info("Job complete: %s", self.job_dir)
        return JobResult(
            job_dir=self.job_dir,
            dub_track=dub_track,
            dubbed_video=dubbed_video,
            burned_video=burned_video,
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

    def _device_info(self) -> dict[str, Any]:
        """Which device and precision this run actually used.

        The dtype is resolved exactly as the engine resolves it, so the report
        cannot claim a precision the synthesis did not use. Recorded because a
        run on unfamiliar hardware has to be diagnosable from the report alone.
        """
        info = probe_device(self.job.primary_device)
        return {
            "requested": self.job.primary_device,
            "resolved": info.label,
            "kind": info.kind,
            "name": info.name,
            "total_memory": info.total_memory,
            "capability": list(info.capability) if info.capability else None,
            "dtype": str(
                resolve_dtype(self.job.primary_device, self.job.dtype)
            ).replace("torch.", ""),
            # Speech recognition runs on CTranslate2, which accepts CUDA and CPU
            # only, whatever device the torch stages were given.
            "asr_compute_type": resolve_compute_type(
                asr_device(self.job.primary_device)
            ),
        }

    def _tts_stats(self) -> dict[str, Any]:
        """How synthesis batched, and how close to the memory ceiling it ran.

        The peak comes from the device counter rather than the engine, so a job
        served partly from cache still reports the number that matters. A fully
        cached job reports zero batches — the honest answer, since nothing was
        synthesised this time. Recorded because a thrashing run leaves artifacts
        that otherwise look clean: the allocator recovers internally, so no error
        ever reaches the pipeline.
        """
        stats: dict[str, Any] = {
            "batch_size": self.job.tts_batch,
            "max_decode_seconds": round(MAX_DECODE_SECONDS, 2),
            "reference_seconds": None,
            "seconds_per_char": None,
            "batches": 0,
            "peak_batch_seconds": 0.0,
            "oom_retries": 0,
            "cache_releases": 0,
            "ceiling_shrinks": 0,
            "peak_reserved_bytes": None,
        }
        reports: list[dict[str, Any]] = []
        if self._runner_instance is not None:
            reports = self._runner_instance.stats()
        elif self._tts is not None:
            reports = [self._tts.synthesis_stats()]
        merged = merge_tts_stats(reports)
        if merged:
            stats.update(merged)
        if self.job.multi_device:
            # The cards the stage actually used. Kept beside the merged peak,
            # because "9.3 GiB on one card and 2 GiB on the other" and "5.6 GiB on
            # each" share a maximum and are very different runs.
            stats["devices"] = list(self.job.resolved_devices)
        return stats

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
        room = self._room_stats()
        has_background = self._background_path() is not None
        # Room matching ducks the bed more gently, so report the depth actually
        # used (``room_duck_db``) rather than the flag the user passed.
        if has_background and room and room.get("applied"):
            effective_duck: float | None = room.get("duck_db", self.job.duck_db)
        elif has_background:
            effective_duck = self.job.duck_db
        else:
            effective_duck = None
        ratios = [p.ratio for p in timeline.placements]
        shifted = sum(1 for p in timeline.placements if p.shifted)
        #: How late each line actually starts, which is what the eye notices. The
        #: mixer may borrow `max_overrun` seconds from the next line's time, so a
        #: run with a non-zero count is expected now; the amount is what is judged.
        lates = [max(0.0, p.start - p.source_start) for p in timeline.placements]
        overruns = [p.overrun for p in timeline.placements]
        trimmed = [p.trimmed for p in timeline.placements]
        deficits = [p.deficit for p in timeline.placements]
        silence = [p.silence_trimmed for p in timeline.placements]
        rates = self._probe_rates()
        speedup = self.job.budget_speedup

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
        spoken_total = sum(spoken_seconds(p) for p in timeline.placements)
        report = {
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
            "background_preserved": has_background,
            "duck_db": effective_duck,
            "accent_mode": self.job.accent_mode,
            #: Which device and precision the models ran on.
            "device": self._device_info(),
            #: How synthesis batched, and the memory it actually reached. A run
            #: that quietly retried around an out-of-memory error shows up here
            #: rather than only in the allocator's own log lines.
            "tts": self._tts_stats(),
            #: How fast each cloned voice actually spoke, measured by the rate
            #: probe, and the ceiling that measurement produced. This pair decides
            #: whether a translation can fit at all; both are empty when there was
            #: no target-language text to probe with.
            "delivery_wps": rates,
            "budget_wps": {
                name: round(rate / speedup, 2) for name, rate in rates.items()
            },
            "budget_speedup": speedup,
            "donor": self._donor_info(),
            #: Which voices were found and how much each of them said. ``None``
            #: when the video has one speaker, so single-speaker reports are
            #: unchanged by diarization.
            "speakers": self._speakers_summary(),
            #: What the length repair did: which lines were re-worded to avoid
            #: cutting them, and how much speech that saved.
            "refine": self._refine_stats(),
            #: What the word check did *before* synthesising anything: the lines it
            #: had to reword so the mixer would not have to delete speech, and how
            #: much of the wording survived (``words_kept``).
            "prefit": self._prefit_stats(),
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
            #: The compression floor in force, so a report always says which
            #: limit its own "needed" numbers should be read against.
            "compress_floor": self.job.min_stretch,
            #: Lines the mixer had to speed up at all.
            "accelerated_segments": sum(1 for r in ratios if r < 0.98),
            #: Lines whose *translation* was longer than the floor allows, i.e. the
            #: ones the length repair can help. This is the number to drive to
            #: zero: while it is above zero, words are being deleted.
            "needed_below_floor_segments": sum(
                1
                for p in timeline.placements
                if 0.0 < p.needed_ratio < self.job.min_stretch
            ),
            "needed_ratio_min": (
                round(min(p.needed_ratio for p in timeline.placements), 3)
                if timeline.placements
                else None
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
            "shifted_seconds_total": round(sum(lates), 2),
            "shifted_seconds_max": round(max(lates), 3) if lates else 0.0,
            #: The lateness policy in force, so a report states the limit its own
            #: shift numbers should be read against.
            "max_overrun": self.job.max_overrun,
            # Lines that used the gap before the next utterance instead of being
            # cut. Speech is only dropped when it would have desynchronised the
            # video, which `trimmed_seconds_total` measures.
            "overrun_segments": sum(1 for value in overruns if value > 0.01),
            "overrun_seconds_total": round(sum(overruns), 2),
            "overrun_seconds_max": round(max(overruns), 2) if overruns else 0.0,
            "trimmed_segments": sum(1 for value in trimmed if value > 0.01),
            "trimmed_seconds_total": round(sum(trimmed), 2),
            #: Dead air removed from the ends of synthesised lines before fitting.
            #: Silent, so nothing is lost - but leaving it in would spend the
            #: compression budget (and, at the floor, the wording) on a pause.
            "silence_trimmed_segments": sum(1 for value in silence if value > 0.01),
            "silence_trimmed_seconds_total": round(sum(silence), 2),
            #: The worst offenders by name, so the ids in a failure report lead
            #: straight to the lines to listen to.
            "worst_trimmed_lines": [
                {
                    "id": p.segment_id,
                    "trimmed": round(p.trimmed, 3),
                    "needed_ratio": round(p.needed_ratio, 3),
                }
                for p in sorted(
                    (p for p in timeline.placements if p.trimmed > 0.01),
                    key=lambda p: p.trimmed,
                    reverse=True,
                )[:5]
            ],
            "outputs": {
                "dub_track": self.job_dir.name + "/" + _DUB_TRACK_NAME,
                "video": dubbed_video.name if dubbed_video else None,
            },
            "stage_seconds": self._stage_seconds,
        }
        # Present only when overlap detection actually ran, so a one-voice job
        # reports exactly what it did before this stage existed. A line carrying a
        # second voice is dubbed in the dominant speaker's voice and counted here
        # rather than silently attributed to one person.
        overlap = self._overlap_stats()
        if overlap is not None:
            report["overlap"] = overlap
        #: Present only when ASR timing had to be repaired, so an ordinary report is
        #: unchanged by this feature.
        packed = self._packed_stats()
        if packed is not None:
            report["packed"] = packed
        # Present only when room matching was requested, so a dry run's report
        # stays byte-for-byte what it was before this feature existed.
        if room is not None:
            report["room"] = room
        return report
