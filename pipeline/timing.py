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
from typing import Collection, Protocol, Sequence

import numpy as np
import soundfile as sf

from .errors import DependencyError, TimingError
from .media import limit_peaks, run_command, write_audio

__all__ = [
    "TimeStretcher",
    "IdentityStretcher",
    "RubberbandStretcher",
    "SoxStretcher",
    "LibrosaStretcher",
    "detect_stretcher",
    "compute_ratio",
    "fit_ratio",
    "fit_line",
    "trim_silence",
    "Placement",
    "build_timeline",
    "MIN_RATIO",
    "MAX_RATIO",
    "SHIFT_FLAG_GUARD_S",
]
#: Stretch bounds for callers that pass none. The pipeline is stricter about
#: slowing down (speech that lags its slot reads as sloppy) than about speeding
#: up, and its compression floor is ``JobConfig.min_stretch`` (0.75): past that
#: the wording has to change, because the alternative is deleting words.
MIN_RATIO = 0.85
MAX_RATIO = 1.18

#: Ratios closer than this to 1.0 are not worth processing.
_NEUTRAL_EPSILON = 0.02

#: A line pushed later than this counts as "shifted" in the report. Below it, the
#: move is rounding: a line fitted to end exactly at the next one's start can land
#: a sample or two over it, and calling that a desync would make the QC gate fail
#: on arithmetic rather than on sync.
SHIFT_FLAG_GUARD_S = 0.02

#: Level, relative to a clip's own peak, below which it counts as silent. Relative
#: because a quiet line must not be trimmed away, and -40 dB is far under speech.
_SILENCE_FLOOR_DB = -40.0
#: Silence shorter than this is kept, so a click or a plosive that dips the whole
#: clip for a few frames is not cut into.
_SILENCE_GUARD_S = 0.05
#: Give up on a clip that shrinks past this share of itself: the "peak" was
#: probably a transient and the rest of the line is not what was promised.
_SILENCE_MIN_KEEP = 0.25


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


def fit_ratio(
    slot: float,
    window: float,
    synth_seconds: float,
    *,
    min_ratio: float = MIN_RATIO,
    max_ratio: float = MAX_RATIO,
) -> float:
    """Ratio the mix stage applies to a ``synth_seconds``-long line.

    Compression aims at the *slot* first: speech should stop when the source
    speaker stopped. When that would need more than ``min_ratio``, the gap before
    the next line - ``window - slot`` - is spent before more tempo is, because a
    pause nobody talks over costs nothing while tempo always costs quality. A line
    that still does not fit the window is the only one that gets trimmed.

    Args:
        slot: The segment's own duration, in seconds.
        window: The time the line may use at most: the slot plus the gap before it.
        synth_seconds: Duration of the synthesised line, before stretching.
        min_ratio: Compression floor; the caller's policy limit.
        max_ratio: Slow-down ceiling.

    Returns:
        The ratio to apply, clamped to ``[min_ratio, max_ratio]``.
    """
    if slot <= 0.0 or synth_seconds <= 0.0:
        return 1.0
    ratio = min(max_ratio, max(min_ratio, slot / synth_seconds))
    needed = (window if window > 0.0 else slot) / synth_seconds
    # Compress less when the following pause can pay for the difference.
    if ratio < needed < 1.0:
        ratio = needed
    return float(ratio)


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    *,
    floor_db: float = _SILENCE_FLOOR_DB,
    guard_s: float = _SILENCE_GUARD_S,
) -> np.ndarray:
    """Drop leading and trailing silence from a synthesised line.

    A cloned line routinely opens with a tenth to a third of a second of digital
    silence (measured: 0.13 s and 0.35 s on a two-line job). Placed at its slot
    start that is audible dead air *and* it is charged to the compression budget,
    so the mixer speeds real speech up to pay for a pause nobody asked for - on the
    measured line, 0.5 s of silence forced a 0.82 compression where the speech
    alone needed 0.97.

    Args:
        samples: The synthesised line.
        sample_rate: Its sample rate.
        floor_db: Level relative to the clip's own peak that counts as silence.
        guard_s: Silence kept on each side, so an attack is never cut into.

    Returns:
        The trimmed clip, or the input unchanged when trimming is not safe.
    """
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return data
    peak = float(np.abs(data).max())
    if peak <= 0.0:
        return data

    loud = np.flatnonzero(np.abs(data) > peak * (10.0 ** (floor_db / 20.0)))
    if loud.size == 0:
        return data
    guard = max(0, int(round(guard_s * sample_rate)))
    start = max(0, int(loud[0]) - guard)
    stop = min(data.size, int(loud[-1]) + 1 + guard)
    if stop - start < _SILENCE_MIN_KEEP * data.size:
        # Almost everything measured as silence: the peak was a transient, and
        # returning a stub would turn a long line into a stretched click.
        return data
    return np.ascontiguousarray(data[start:stop])


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
    available: float | None = None,
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
        available: Total seconds the line may use, i.e. ``target_duration`` plus
            ``max_overrun``. Only changes how far the line is compressed: the gap
            is spent before tempo. ``None`` means "the slot only".

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
    ratio = fit_ratio(
        target_duration,
        available if available is not None else target_duration,
        synth_duration,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
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
    #: Duration of the synthesised line before stretching, in seconds.
    synth_seconds: float = 0.0
    #: Seconds of the slot still silent after stretching (0.0 when it fills).
    deficit: float = 0.0
    #: Compression the line needed to fit its slot plus the following gap, before
    #: the floor was applied. Below ``min_stretch`` means the wording, not the
    #: tempo, has to give - which is what the length repair is for.
    needed_ratio: float = 0.0
    #: Dead air dropped from the clip's ends before it was fitted.
    silence_trimmed: float = 0.0


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
    overlay_ids: Collection[int] = (),
) -> Timeline:
    """Place fitted lines on a silent track at their original start times.

    Overlaps are resolved by pushing a line later - dubbing two voices on top of
    each other is normally worse than a small timing shift. The exception is
    genuinely simultaneous speech: a line whose id is in ``overlay_ids`` is the
    second voice of a separated overlap segment, so it is summed at its true time
    rather than shifted, because that simultaneity is exactly what separation
    recovered.

    Args:
        items: ``(segment_id, start_seconds, audio)`` triples.
        sample_rate: Sample rate of every audio array.
        total_duration: Minimum track length; the result grows if needed.
        overlay_ids: Segment ids placed at their true time instead of being pushed
            later. An overlay line never advances the cursor, so it cannot shift a
            following line.

    Returns:
        A :class:`Timeline` with the assembled audio and placement records.
    """
    if sample_rate <= 0:
        raise TimingError("sample_rate must be positive")

    overlays = set(overlay_ids)
    guard = int(round(SHIFT_FLAG_GUARD_S * sample_rate))
    positions: list[tuple[int, int, np.ndarray, float, bool]] = []
    cursor = 0
    required = 0
    for segment_id, start, audio in items:
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if data.size == 0:
            continue
        start_sample = max(0, int(round(float(start) * sample_rate)))
        overlay = segment_id in overlays
        shifted = False
        if not overlay and start_sample < cursor:
            # Moved, but only a flag when the move is big enough to hear: see
            # ``SHIFT_FLAG_GUARD_S``.
            shifted = cursor - start_sample > guard
            start_sample = cursor
        positions.append((segment_id, start_sample, data, float(start), shifted))
        # An overlay line is simultaneous with the line it accompanies, so it must
        # not advance the cursor even though it lengthens the track.
        if not overlay:
            cursor = start_sample + data.size
        required = max(required, start_sample + data.size)

    required = max(required, int(round((total_duration or 0.0) * sample_rate)))
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

    # Guard against summing overlapping lines into clipping. A limiter is used
    # rather than scaling the whole track: scaling would drop the level of every
    # line to accommodate a single transient.
    track = limit_peaks(track, sample_rate)

    return Timeline(samples=track, sample_rate=sample_rate, placements=placements)
