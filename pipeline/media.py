"""Audio/video I/O: ffmpeg wrappers, probing, and loudness utilities.

Binary invocation always passes an argument list (never a shell string) to avoid
command injection, per the project's security guidelines. Loudness handling is
implemented with numpy + pyloudnorm so it works identically for video and
audio-only inputs and never needs a second ffmpeg pass.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from .errors import DependencyError, MediaError

__all__ = [
    "MediaInfo",
    "LoudnessResult",
    "require_binary",
    "run_command",
    "probe",
    "extract_audio",
    "load_audio",
    "write_audio",
    "resample",
    "mux_audio",
    "measure_loudness",
    "normalize_loudness",
    "normalize_loudness_detailed",
    "save_normalized",
    "DEFAULT_SAMPLE_RATE",
    "TARGET_LUFS",
]

DEFAULT_SAMPLE_RATE = 48_000

#: Broadcast-style loudness target (EBU R128 / streaming convention, ~-16 LUFS).
TARGET_LUFS = -16.0
_TARGET_LUFS = TARGET_LUFS


def require_binary(name: str, *, hint: str | None = None) -> str:
    """Return the resolved path of an external binary.

    Raises:
        DependencyError: With installation guidance when the binary is missing.
    """
    path = shutil.which(name)
    if path is None:
        message = f"Required binary {name!r} was not found on PATH."
        if hint:
            message = f"{message} {hint}"
        raise DependencyError(message)
    return path


def run_command(cmd: Sequence[str], *, timeout: float | None = None) -> str:
    """Run a command with ``shell=False`` and return its stdout.

    Raises:
        DependencyError: If the executable itself is missing.
        MediaError: On a non-zero exit status, including stderr context.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - argument list, shell disabled
            list(cmd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"Executable {cmd[0]!r} not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - environment dependent
        raise MediaError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-8:])
        raise MediaError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n{tail}"
        )
    return completed.stdout


@dataclass(frozen=True)
class MediaInfo:
    """Subset of ffprobe output the pipeline cares about."""

    duration_s: float
    has_audio: bool
    has_video: bool
    sample_rate: int | None = None
    channels: int | None = None


@dataclass(frozen=True)
class LoudnessResult:
    """Outcome of a loudness normalisation pass.

    Exposed so the job report can state the loudness that was actually
    delivered rather than the raw mix's, and flag when the target was
    unreachable without clipping.
    """

    samples: np.ndarray
    target_lufs: float
    measured_lufs: float | None
    achieved_lufs: float | None
    gain_db: float
    peak_limited: bool


def probe(path: Path | str) -> MediaInfo:
    """Inspect a media file with ffprobe."""
    require_binary("ffprobe", hint="Install ffmpeg (it ships ffprobe).")
    output = run_command(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed ffprobe output
        raise MediaError(f"Could not parse ffprobe output for {path}") from exc

    streams = payload.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)

    duration = payload.get("format", {}).get("duration")
    if duration in (None, "N/A"):
        # Streams can carry the duration when the container does not.
        duration = (audio or video or {}).get("duration")
    try:
        duration_s = float(duration)
    except (TypeError, ValueError):
        duration_s = 0.0

    return MediaInfo(
        duration_s=duration_s,
        has_audio=audio is not None,
        has_video=video is not None,
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        channels=int(audio["channels"]) if audio and audio.get("channels") else None,
    )


def extract_audio(
    video: Path | str,
    out: Path | str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Extract the first audio stream to a mono PCM WAV."""
    require_binary("ffmpeg", hint="Install ffmpeg.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(video),
            "-vn", "-map", "0:a:0",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out),
        ]
    )
    return out


def load_audio(
    path: Path | str,
    *,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio as mono float32, optionally resampling.

    Uses soundfile + librosa so audio-only inputs never require ffmpeg.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if sample_rate is not None and sr != sample_rate:
        import librosa

        data = librosa.resample(data, orig_sr=int(sr), target_sr=int(sample_rate))
        sr = sample_rate
    return data, int(sr)


def write_audio(path: Path | str, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a mono float array to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32), int(sample_rate))
    return path


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample in memory when the rates differ."""
    if orig_sr == target_sr:
        return np.asarray(samples, dtype=np.float32)
    import librosa

    return librosa.resample(
        np.asarray(samples, dtype=np.float32), orig_sr=int(orig_sr), target_sr=int(target_sr)
    )


def mux_audio(video: Path | str, audio: Path | str, out: Path | str) -> Path:
    """Replace the video's audio track, copying the video stream untouched.

    ``-shortest`` prevents an audio track longer than the video from extending
    the container, and the ``-map`` flags avoid accidentally carrying over the
    original audio or subtitle streams.
    """
    require_binary("ffmpeg", hint="Install ffmpeg.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out),
        ]
    )
    return out


def measure_loudness(samples: np.ndarray, sample_rate: int) -> float | None:
    """Integrated loudness in LUFS, or ``None`` when it cannot be measured.

    pyloudnorm rejects clips shorter than the 400 ms gating block and silent
    input, both of which happen in practice; those return ``None`` rather than
    raising so the caller can skip normalisation.
    """
    try:
        import pyloudnorm
    except ImportError:  # pragma: no cover - optional dependency
        return None

    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0:
        return None
    peak = float(np.max(np.abs(data)))
    if peak < 1e-6:
        return None
    try:
        meter = pyloudnorm.Meter(int(sample_rate))
        value = float(meter.integrated_loudness(data))
    except (ValueError, ZeroDivisionError):
        return None
    if not np.isfinite(value):
        return None
    return value


def normalize_loudness(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float = _TARGET_LUFS,
) -> np.ndarray:
    """Apply a gain so the clip sits at ``target_lufs``.

    The gain is limited so the result cannot clip, and the input is returned
    unchanged when loudness is unmeasurable. See
    :func:`normalize_loudness_detailed` for the numbers behind the result.
    """
    return normalize_loudness_detailed(
        samples, sample_rate, target_lufs=target_lufs
    ).samples


def normalize_loudness_detailed(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float = _TARGET_LUFS,
    headroom_db: float = 0.1,
) -> "LoudnessResult":
    """Normalise loudness and report what actually happened.

    Speech often has a high crest factor, so reaching a broadcast target such as
    -16 LUFS can require more gain than the peak allows. Rather than clipping (or
    quietly under-delivering), the gain is capped at the largest value that keeps
    the signal below full scale, and ``peak_limited`` records that this happened
    so the shortfall is visible instead of silent.

    Returns:
        A :class:`LoudnessResult` carrying the processed samples plus the
        measured, target and achieved loudness.
    """
    data = np.asarray(samples, dtype=np.float32)
    measured = measure_loudness(data, sample_rate)
    if measured is None:
        return LoudnessResult(
            samples=data,
            target_lufs=target_lufs,
            measured_lufs=None,
            achieved_lufs=None,
            gain_db=0.0,
            peak_limited=False,
        )

    desired_gain_db = target_lufs - measured
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    max_gain_db = (
        -20.0 * math.log10(peak) - headroom_db if 0.0 < peak < 1.0 else 0.0
    )
    gain_db = min(desired_gain_db, max_gain_db)
    processed = (data * (10.0 ** (gain_db / 20.0))).astype(np.float32)

    return LoudnessResult(
        samples=processed,
        target_lufs=target_lufs,
        measured_lufs=measured,
        achieved_lufs=measure_loudness(processed, sample_rate),
        gain_db=gain_db,
        peak_limited=gain_db < desired_gain_db - 1e-6,
    )


def save_normalized(
    samples: np.ndarray,
    sample_rate: int,
    path: Path | str,
    *,
    target_lufs: float = _TARGET_LUFS,
) -> Path:
    """Normalise loudness and write the result to ``path``."""
    return write_audio(path, normalize_loudness(samples, sample_rate, target_lufs=target_lufs), sample_rate)
