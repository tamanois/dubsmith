"""Speech recognition and segmentation.

faster-whisper ships an ONNX Silero VAD, so ``vad_filter=True`` gives us speech
gating without adding a separate silero-vad dependency. Segments the model
itself distrusts (``no_speech_prob`` above threshold, or empty text) are dropped
before they can reach translation or synthesis.

The conversion from raw decoder objects to our models lives in
:func:`build_transcript` so it can be unit tested without loading a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

from .errors import ASRError, DependencyError

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "Word",
    "Segment",
    "Transcript",
    "asr_device",
    "resolve_compute_type",
    "build_transcript",
    "transcribe",
    "transcribe_many",
    "load_transcript",
    "save_transcript",
]


class Word(BaseModel):
    """A single recognised word with timings, used for future alignment work."""

    word: str
    start: float
    end: float


class Segment(BaseModel):
    """One utterance on the timeline."""

    id: int = 0
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    words: list[Word] = Field(default_factory=list)
    #: Which speaker this utterance belongs to, filled in by the diarization
    #: stage. ``None`` means "not assigned" - transcripts written before
    #: diarization existed, and every run with ``--diarizer none``, which the rest
    #: of the pipeline treats as a single voice.
    speaker: str | None = None
    #: True when the overlap stage found a second voice inside this utterance. A
    #: flagged line is still dubbed in its dominant speaker's voice; the flag is
    #: what makes the second voice visible instead of silently attributed.
    overlapped: bool = False
    #: The ASR segment a separated sub-line came from. ``None`` for an ordinary
    #: segment; set on the per-speaker lines produced by overlap separation.
    parent_id: int | None = None
    #: True for the *second* voice's line in a separated segment. Overlay lines are
    #: summed onto the track at their true times instead of being pushed later,
    #: because the overlap is genuine simultaneous speech.
    overlay: bool = False

    @property
    def duration(self) -> float:
        """Slot length in seconds (never negative)."""
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    """Full transcription result for one input file."""

    language: str | None = None
    language_probability: float | None = None
    model: str = ""
    segments: list[Segment] = Field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """End of the last segment, 0.0 when empty."""
        return max((segment.end for segment in self.segments), default=0.0)

    def is_empty(self) -> bool:
        """True when nothing usable was recognised."""
        return not self.segments


def asr_device(device: str) -> str:
    """Map a device request to one CTranslate2 understands.

    CTranslate2, which faster-whisper runs on, offers CUDA and CPU only, and wants
    the bare ``"cuda"`` rather than ``"cuda:0"``. Metal and XPU therefore degrade
    to CPU instead of failing after the weights have loaded, which is what an
    explicit ``--device mps`` used to do.
    """
    from .config import resolve_device

    resolved = str(resolve_device(device)).lower()
    return "cuda" if resolved.startswith(("cuda", "rocm")) else "cpu"


def resolve_compute_type(device: str) -> str:
    """Pick a CTranslate2 compute type appropriate for the device."""
    return "float16" if str(device).lower().startswith("cuda") else "int8"


def build_transcript(
    raw_segments: Iterable[Any],
    *,
    language: str | None,
    language_probability: float | None = None,
    model: str = "",
    no_speech_threshold: float = 0.6,
    allow_empty: bool = False,
) -> Transcript:
    """Convert raw faster-whisper segments into a :class:`Transcript`.

    Args:
        raw_segments: Objects exposing ``start``, ``end``, ``text``,
            ``no_speech_prob`` and optionally ``words``.
        language: Detected (or forced) language code.
        language_probability: Confidence reported by the model, when available.
        model: Identifier recorded for provenance.
        no_speech_threshold: Segments at or above this probability are dropped as
            non-speech hallucinations.
        allow_empty: Return a transcript with no segments instead of raising when
            nothing usable survives. Used for separated streams, which are often
            silent for the weaker voice.

    Returns:
        Transcript with gaps removed and contiguous ids assigned.

    Raises:
        ASRError: If no usable segment survives filtering and ``allow_empty`` is
            ``False``.
    """
    segments: list[Segment] = []
    for raw in raw_segments:
        text = str(getattr(raw, "text", "") or "").strip()
        if not text:
            continue

        no_speech = float(getattr(raw, "no_speech_prob", 0.0) or 0.0)
        if no_speech >= no_speech_threshold:
            continue

        start = max(0.0, float(getattr(raw, "start", 0.0) or 0.0))
        end = float(getattr(raw, "end", start) or start)
        if end <= start:
            # A zero-length slot cannot be dubbed or subtitled.
            continue

        words: list[Word] = []
        for word in getattr(raw, "words", None) or []:
            if word is None:
                continue
            words.append(
                Word(
                    word=str(getattr(word, "word", "") or "").strip(),
                    start=float(getattr(word, "start", start) or start),
                    end=float(getattr(word, "end", end) or end),
                )
            )

        segments.append(
            Segment(
                id=len(segments),
                start=start,
                end=end,
                text=text,
                no_speech_prob=no_speech,
                words=words,
            )
        )

    if not segments and not allow_empty:
        raise ASRError(
            "No speech detected in the input. Check that the file contains audible "
            "speech and that the correct audio track was selected."
        )

    return Transcript(
        language=language,
        language_probability=language_probability,
        model=model,
        segments=segments,
    )


def _looks_like_cuda_library_failure(exc: BaseException) -> bool:
    """Heuristic: did CUDA fail because the runtime libraries are unusable?

    CTranslate2 links against a specific CUDA major version (currently 12), while
    torch may ship a different one (e.g. CUDA 13 wheels provide libcublas.so.13).
    The failure surfaces as a ``RuntimeError`` about a missing ``libcublas`` /
    ``libcudnn`` shared object, which is recoverable by running on CPU.
    """
    message = str(exc).lower()
    markers = (
        "libcublas",
        "libcudnn",
        "libcuda",
        "cannot be loaded",
        "cuda error",
        "no kernel image",
        "not compiled with cuda",
    )
    return any(marker in message for marker in markers)


def _cuda_library_dirs() -> list[Path]:
    """Directories holding CUDA runtime libraries shipped inside the environment.

    torch wheels vendor their CUDA runtime under ``site-packages/nvidia/*/lib``.
    When torch and CTranslate2 target different CUDA majors, the libs
    CTranslate2 needs are usually still present there (e.g. via the
    ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` wheels).
    """
    import sysconfig

    roots: list[Path] = []
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        roots.append(Path(purelib))

    directories: list[Path] = []
    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for package in sorted(nvidia.iterdir()):
            lib_dir = package / "lib"
            if lib_dir.is_dir():
                directories.append(lib_dir)
    return directories


_PRELOADED_CUDA_LIBS: int | None = None


def preload_cuda_libraries() -> int:
    """Load bundled CUDA runtime libraries into this process.

    Returns the number of libraries loaded (0 when none are available), and is
    idempotent. Libraries are loaded by absolute path with ``RTLD_GLOBAL`` so
    their sonames land in the process link map; a later ``dlopen("libcublas.so.12")``
    by CTranslate2 then resolves to the already-loaded copy, which is what lets GPU
    inference work without the user exporting ``LD_LIBRARY_PATH``.

    Load order matters (``libcublasLt`` before ``libcublas``, cuDNN sub-libraries
    before the umbrella lib), so loading is retried until no further progress is
    possible instead of hard-coding a dependency order.
    """
    global _PRELOADED_CUDA_LIBS
    if _PRELOADED_CUDA_LIBS is not None:
        return _PRELOADED_CUDA_LIBS

    import ctypes

    pending = [lib for directory in _cuda_library_dirs() for lib in sorted(directory.glob("*.so*"))]
    if not pending:
        _PRELOADED_CUDA_LIBS = 0
        return 0

    loaded = 0
    while pending:
        progressed = False
        for library in list(pending):
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            pending.remove(library)
            loaded += 1
            progressed = True
        if not progressed:
            break

    _PRELOADED_CUDA_LIBS = loaded
    return loaded


def _attempt_plan(ct2_device: str, allow_cpu_fallback: bool) -> list[tuple[str, bool]]:
    """Devices to try, in order, as ``(device, preload_cuda_libs)`` pairs."""
    plan: list[tuple[str, bool]] = [(ct2_device, False)]
    if ct2_device == "cuda" and allow_cpu_fallback:
        # Second chance: preload vendored CUDA libs, then last resort CPU.
        plan.append(("cuda", True))
        plan.append(("cpu", False))
    return plan


def _load_whisper(model_dir: Path | str, device: str, compute_type: str | None) -> Any:
    """Load a faster-whisper model on ``device``."""
    from faster_whisper import WhisperModel

    compute = compute_type or resolve_compute_type(device)
    return WhisperModel(str(model_dir), device=device, compute_type=compute)


def _decode_transcript(
    model: Any,
    audio_path: Path | str,
    *,
    language: str | None,
    vad_filter: bool,
    beam_size: int,
    no_speech_threshold: float,
    model_id: str,
    allow_empty: bool = False,
) -> Transcript:
    """Decode one file with an already-loaded model."""
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
    )
    # Materialise the generator here so decoding errors surface inside this call
    # rather than later, when a device fallback would no longer be available.
    return build_transcript(
        list(segments),
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        model=str(model_id),
        no_speech_threshold=no_speech_threshold,
        allow_empty=allow_empty,
    )


def transcribe(
    audio_path: Path | str,
    model_dir: Path | str,
    *,
    device: str = "auto",
    compute_type: str | None = None,
    language: str | None = None,
    vad_filter: bool = True,
    no_speech_threshold: float = 0.6,
    beam_size: int = 5,
    allow_cpu_fallback: bool = True,
) -> Transcript:
    """Transcribe ``audio_path`` with a local faster-whisper model.

    Args:
        audio_path: Mono WAV to transcribe.
        model_dir: Local directory produced by ``python dub.py models --download``.
        device: Torch-style device; ``auto`` prefers CUDA.
        compute_type: CTranslate2 compute type; inferred from the device if omitted.
        language: Force a source language code; ``None`` auto-detects.
        vad_filter: Enable the bundled Silero VAD gate.
        no_speech_threshold: Drop segments at or above this no-speech probability.
        beam_size: Decoder beam width.
        allow_cpu_fallback: When CUDA is unusable (typically a cuBLAS/cuDNN version
            mismatch, since CTranslate2 pins a CUDA major while torch may ship a
            newer one), retry first after preloading vendored CUDA libraries and
            finally on CPU. Without this, a GPU job would abort even though CPU
            inference produces identical output.

    Returns:
        Transcript with contiguous segment ids.

    Raises:
        DependencyError: If faster-whisper is not installed.
        ASRError: If no usable speech is found, or every device fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised via CLI paths
        raise DependencyError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    # CTranslate2 understands CUDA and CPU only, and wants the bare "cuda".
    ct2_device = asr_device(device)

    def run(target_device: str) -> Transcript:
        model = _load_whisper(model_dir, target_device, compute_type)
        return _decode_transcript(
            model,
            audio_path,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            no_speech_threshold=no_speech_threshold,
            model_id=str(model_dir),
        )

    last_error: RuntimeError | None = None
    for target_device, should_preload in _attempt_plan(ct2_device, allow_cpu_fallback):
        if should_preload:
            count = preload_cuda_libraries()
            if not count:
                continue  # Nothing to preload; skip straight to the next attempt.
            _LOGGER.info("Preloaded %d bundled CUDA libraries; retrying on %s.", count, target_device)

        try:
            return run(target_device)
        except RuntimeError as exc:
            if not _looks_like_cuda_library_failure(exc):
                raise ASRError(f"Speech recognition failed: {exc}") from exc
            last_error = exc
            _LOGGER.warning(
                "CUDA is unusable for speech recognition on %s (%s).",
                target_device,
                exc,
            )

    raise ASRError(
        "Speech recognition failed on every available device "
        f"(last error: {last_error}). Try --device cpu."
    )


def transcribe_many(
    audio_paths: Sequence[Path | str],
    model_dir: Path | str,
    *,
    device: str = "auto",
    compute_type: str | None = None,
    language: str | None = None,
    vad_filter: bool = True,
    no_speech_threshold: float = 0.6,
    beam_size: int = 5,
    allow_cpu_fallback: bool = True,
    allow_empty: bool = True,
) -> list[Transcript]:
    """Transcribe several clips, loading the Whisper model once.

    Overlap separation re-transcribes each separated stream, and loading a
    multi-GB model per stream would dominate the stage. Device fallback mirrors
    :func:`transcribe`: a CUDA library failure retries the whole batch on the next
    device, because the model - not the clip - is what failed.

    Args:
        audio_paths: Clips to transcribe.
        model_dir: Local faster-whisper model directory.
        allow_empty: Return an empty transcript instead of raising when a clip has
            no speech. A separated stream is often silent for the weaker voice, so
            this defaults to ``True`` here, unlike :func:`transcribe`.

    Other args are as in :func:`transcribe`.

    Returns:
        One transcript per input path, in the same order.

    Raises:
        DependencyError: If faster-whisper is not installed.
        ASRError: If every device fails.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via CLI paths
        raise DependencyError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    ct2_device = asr_device(device)
    paths = [Path(path) for path in audio_paths]

    def run(target_device: str) -> list[Transcript]:
        model = _load_whisper(model_dir, target_device, compute_type)
        return [
            _decode_transcript(
                model,
                path,
                language=language,
                vad_filter=vad_filter,
                beam_size=beam_size,
                no_speech_threshold=no_speech_threshold,
                model_id=str(model_dir),
                allow_empty=allow_empty,
            )
            for path in paths
        ]

    last_error: RuntimeError | None = None
    for target_device, should_preload in _attempt_plan(ct2_device, allow_cpu_fallback):
        if should_preload:
            count = preload_cuda_libraries()
            if not count:
                continue  # Nothing to preload; skip straight to the next attempt.
            _LOGGER.info(
                "Preloaded %d bundled CUDA libraries; retrying on %s.",
                count,
                target_device,
            )

        try:
            return run(target_device)
        except RuntimeError as exc:
            if not _looks_like_cuda_library_failure(exc):
                raise ASRError(f"Speech recognition failed: {exc}") from exc
            last_error = exc
            _LOGGER.warning(
                "CUDA is unusable for speech recognition on %s (%s).",
                target_device,
                exc,
            )

    raise ASRError(
        "Speech recognition failed on every available device "
        f"(last error: {last_error}). Try --device cpu."
    )


def save_transcript(transcript: Transcript, path: Path | str) -> Path:
    """Serialise a transcript to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_transcript(path: Path | str) -> Transcript:
    """Load a transcript previously written by :func:`save_transcript`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Transcript.model_validate(payload)
