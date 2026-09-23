"""Duration fitting and timeline assembly (isochrony).

Translated speech rarely matches the original slot length, so each synthesised
line is time-stretched to fit and then placed at its original start time. Errors
are bounded *per segment* — never accumulated — which is what keeps a dub from
drifting out of sync.

Three interchangeable backends are supported, chosen automatically:

======  ============================================================
Backend Requirement                        Notes
======  ============================================================
rubber  ``rubberband`` binary              best quality, formant-preserving
sox     ``sox`` binary                     good, widely available
librosa ``librosa`` Python package         pure Python fallback, no binaries
======  ============================================================
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import soundfile as sf

from .errors import DependencyError, TimingError
from .media import run_command, write_audio

__all__ = [
    "TimeStretcher",
    "IdentityStretcher",
    "RubberbandStretcher",
    "SoxStretcher",
    "LibrosaStretcher",
    "detect_stretcher",
    "compute_ratio",
    "fit_line",
    "Placement",
    "build_timeline",
    "MIN_RATIO",
    "MAX_RATIO",
]
#: Stretch bounds. Beyond roughly ±15–20% phase-vocoder and transient artefacts
#: become audible, so it is better to accept a small timing miss.
MIN_RATIO = 0.85
MAX_RATIO = 1.18

#: Ratios closer than this to 1.0 are not worth processing.
_NEUTRAL_EPSILON = 0.02


class TimeStretcher(Protocol):
    """Changes the duration of audio without changing its pitch."""

    name: str

    def stretch(self, samples: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
        """Return audio whose duration is ``ratio`` times the input's."""
        ...


class IdentityStretcher:
    """No-op backend, useful for tests and for ``--no-stretch`` runs."""

    name = "identity"

    def stretch(self, samples: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
        """Return the input unchanged."""
        return np.asarray(samples, dtype=np.float32)


class _BinaryStretcher:
    """Base class for backends that shell out to an external binary."""

    name = "binary"
    executable = ""

    def __init__(self, work_dir: Path | str | None = None) -> None:
        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd() / ".stretch_tmp"

    def _temp_paths(self) -> tuple[Path, Path]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:12]
        return self.work_dir / f"in_{token}.wav", self.work_dir / f"out_{token}.wav"

    def _command(self, in_path: Path, out_path: Path, ratio: float) -> list[str]:
        raise NotImplementedError

    def stretch(self, samples: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
        """Stretch via the external binary, cleaning up temp files afterwards."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0 or abs(ratio - 1.0) < 1e-6:
            return data

        in_path, out_path = self._temp_paths()
        try:
            write_audio(in_path, data, sample_rate)
            run_command(self._command(in_path, out_path, ratio), timeout=300)
            if not out_path.is_file():
                raise TimingError(f"{self.name} produced no output for ratio {ratio:.4f}")
            stretched, out_sr = sf.read(str(out_path), dtype="float32", always_2d=False)
            if int(out_sr) != int(sample_rate):
                raise TimingError(
                    f"{self.name} changed the sample rate ({sample_rate} -> {out_sr})"
                )
            return np.ascontiguousarray(np.asarray(stretched, dtype=np.float32).reshape(-1))
        finally:
            for path in (in_path, out_path):
                path.unlink(missing_ok=True)


class RubberbandStretcher(_BinaryStretcher):
    """Rubber Band backend: ``--time`` stretches to N times the original duration."""

    name = "rubberband"
    executable = "rubberband"

    def _command(self, in_path: Path, out_path: Path, ratio: float) -> list[str]:
        return [self.executable, "--time", f"{ratio:.6f}", str(in_path), str(out_path)]


class SoxStretcher(_BinaryStretcher):
    """SoX backend using the ``tempo`` effect."""

    name = "sox"
    executable = "sox"

    def _command(self, in_path: Path, out_path: Path, ratio: float) -> list[str]:
        return [self.executable, str(in_path), str(out_path), "tempo", f"{ratio:.6f}"]


class LibrosaStretcher:
    """Pure-Python phase-vocoder fallback; needs no external binaries."""

    name = "librosa"

    def stretch(self, samples: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
        """Stretch with ``librosa.effects.time_stretch``."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0 or abs(ratio - 1.0) < 1e-6:
            return data
        try:
            import librosa
        except ImportError as exc:  # pragma: no cover - librosa is a hard dep
            raise DependencyError("librosa is required for the fallback stretcher.") from exc
        # librosa's `rate` is a speed factor, so duration scales by 1/rate.
        return np.ascontiguousarray(
            librosa.effects.time_stretch(y=data, rate=1.0 / float(ratio)).astype(np.float32)
        )


def detect_stretcher(
    preference: str = "auto",
    *,
    work_dir: Path | str | None = None,
) -> TimeStretcher:
    """Select a stretching backend.

    Args:
        preference: ``auto`` (best available), ``none``/``identity`` (disable),
            or an explicit backend name.
        work_dir: Scratch directory for the binary backends.

    Returns:
        A ready-to-use stretcher.

    Raises:
        DependencyError: If an explicitly requested backend is unavailable.
    """
    import shutil

    preference = (preference or "auto").lower()

    if preference in {"none", "identity", "off"}:
        return IdentityStretcher()

    available: dict[str, type] = {
        "rubberband": RubberbandStretcher,
        "sox": SoxStretcher,
        "librosa": LibrosaStretcher,
    }

    if preference != "auto":
        backend = available.get(preference)
        if backend is None:
            raise DependencyError(
                f"Unknown stretcher {preference!r}. Choose from: auto, none, "
                f"{', '.join(sorted(available))}"
            )
        if preference in {"rubberband", "sox"} and shutil.which(preference) is None:
            raise DependencyError(
                f"Stretcher {preference!r} was requested but the binary is not on PATH."
            )
        return backend(work_dir) if backend in {RubberbandStretcher, SoxStretcher} else backend()  # type: ignore[call-arg]

    for name, cls in (("rubberband", RubberbandStretcher), ("sox", SoxStretcher)):
        if shutil.which(name) is not None:
            return cls(work_dir)
    return LibrosaStretcher()


def compute_ratio(
    source_duration: float,
    synth_duration: float,
    *,
    min_ratio: float = MIN_RATIO,
    max_ratio: float = MAX_RATIO,
) -> float:
    """Ratio that makes ``synth_duration`` fill ``source_duration``, clamped.

    Returns 1.0 when either duration is unusable, so a caller can treat the
    result as "leave it alone".
    """
    if source_duration <= 0 or synth_duration <= 0:
        return 1.0
    return float(min(max_ratio, max(min_ratio, source_duration / synth_duration)))


def _fit_length(
    samples: np.ndarray,
    target_samples: int,
    sample_rate: int,
    *,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Force audio to exactly ``target_samples``, trimming if necessary.

    Trimming applies a short fade-out first so a cut mid-waveform does not click.
    Prefer :func:`_pad_to_length` for dubbing: trimming discards speech.
    """
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    if data.size == target_samples:
        return data
    if data.size < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[: data.size] = data
        return padded

    fade_len = min(int(sample_rate * fade_ms / 1000.0), target_samples)
    trimmed = data[:target_samples].copy()
    if fade_len > 1:
        trimmed[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return trimmed


def _pad_to_length(samples: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with silence up to ``target_samples``, but never trim."""
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if data.size >= target_samples:
        return data
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: data.size] = data
    return padded


def fit_line(
    synth: np.ndarray | None,
    sample_rate: int,
    target_duration: float,
    *,
    stretcher: TimeStretcher,
    min_ratio: float = MIN_RATIO,
    max_ratio: float = MAX_RATIO,
    max_overrun: float | None = None,
) -> tuple[np.ndarray, float]:
    """Fit one synthesised line into ``target_duration`` seconds.

    A line shorter than its slot is padded with silence, preserving the original
    rhythm. A line that is still *longer* after stretching is the interesting
    case: translations are routinely longer than the source (French runs notably
    longer than English) and the stretch clamp prevents compressing arbitrarily
    far.

    Trimming the remainder deletes speech, but letting it run as long as it likes
    desynchronises the dub from the video. ``max_overrun`` is the compromise: the
    line may spill into the *gap* before the next utterance (which costs nothing,
    because nobody is talking) and only the residue beyond that is trimmed.

    Args:
        synth: Synthesised audio, or ``None`` for a blank line.
        sample_rate: Sample rate of both input and output.
        target_duration: Slot length taken from the source segment.
        stretcher: Backend used when stretching is needed.
        min_ratio: Lower clamp for the stretch ratio.
        max_ratio: Upper clamp for the stretch ratio.
        max_overrun: Extra seconds the line may exceed its slot — pass the gap to
            the next utterance. ``None`` never trims (may drift); ``0.0`` forces
            the exact slot length.

    Returns:
        ``(audio, ratio)`` where ``ratio`` is the stretch that was applied.
    """
    if target_duration <= 0:
        raise TimingError("target_duration must be positive")

    target_samples = int(round(target_duration * sample_rate))
    if synth is None or np.asarray(synth).size == 0:
        return np.zeros(target_samples, dtype=np.float32), 1.0

    data = np.asarray(synth, dtype=np.float32).reshape(-1)
    synth_duration = data.size / float(sample_rate)
    ratio = compute_ratio(
        target_duration, synth_duration, min_ratio=min_ratio, max_ratio=max_ratio
    )

    if abs(ratio - 1.0) > _NEUTRAL_EPSILON:
        data = stretcher.stretch(data, sample_rate, ratio)

    if max_overrun is not None:
        limit = int(round((target_duration + max(0.0, max_overrun)) * sample_rate))
        if data.size > limit:
            data = _fit_length(data, max(limit, 1), sample_rate)

    return _pad_to_length(data, target_samples), ratio


@dataclass
class Placement:
    """Where a fitted line ended up on the timeline."""

    segment_id: int
    start: float
    end: float
    ratio: float
    source_start: float
    shifted: bool = False
    #: Seconds the line exceeds its original slot (0.0 when it fits).
    overrun: float = 0.0
    #: Seconds dropped because the line exceeded its slot *and* the following gap.
    trimmed: float = 0.0


@dataclass
class Timeline:
    """Assembled dub track plus per-line placement records."""

    samples: np.ndarray
    sample_rate: int
    placements: list[Placement] = field(default_factory=list)


def build_timeline(
    items: Sequence[tuple[int, float, np.ndarray]],
    *,
    sample_rate: int,
    total_duration: float | None = None,
) -> Timeline:
    """Place fitted lines on a silent track at their original start times.

    Overlaps are resolved by pushing a line later: dubbing two voices on top of
    each other is worse than a small timing shift, and diarization already flags
    genuinely simultaneous speech for later manual review.

    Args:
        items: ``(segment_id, start_seconds, audio)`` triples.
        sample_rate: Sample rate of every audio array.
        total_duration: Minimum track length; the result grows if needed.

    Returns:
        A :class:`Timeline` with the assembled audio and placement records.
    """
    if sample_rate <= 0:
        raise TimingError("sample_rate must be positive")

    positions: list[tuple[int, int, np.ndarray, float, bool]] = []
    cursor = 0
    for segment_id, start, audio in items:
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if data.size == 0:
            continue
        start_sample = max(0, int(round(float(start) * sample_rate)))
        shifted = False
        if start_sample < cursor:
            start_sample, shifted = cursor, True
        positions.append((segment_id, start_sample, data, float(start), shifted))
        cursor = start_sample + data.size

    required = max(cursor, int(round((total_duration or 0.0) * sample_rate)))
    track = np.zeros(required, dtype=np.float32)

    placements: list[Placement] = []
    for segment_id, start_sample, data, source_start, shifted in positions:
        end_sample = start_sample + data.size
        track[start_sample:end_sample] += data
        placements.append(
            Placement(
                segment_id=segment_id,
                start=start_sample / sample_rate,
                end=end_sample / sample_rate,
                ratio=1.0,
                source_start=source_start,
                shifted=shifted,
            )
        )

    # Guard against summing overlapping lines into clipping.
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 1.0:
        track = (track / peak).astype(np.float32)

    return Timeline(samples=track, sample_rate=sample_rate, placements=placements)
