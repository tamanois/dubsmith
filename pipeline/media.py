"""Audio/video I/O: ffmpeg wrappers, probing, and loudness utilities.

Binary invocation always passes an argument list (never a shell string) to avoid
command injection, per the project's security guidelines. Loudness handling is
implemented with numpy + pyloudnorm so it works identically for video and
audio-only inputs and never needs a second ffmpeg pass.
"""

from __future__ import annotations

import json
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
    "SubtitleStream",
    "require_binary",
    "run_command",
    "probe",
    "probe_subtitle_streams",
    "extract_audio",
    "load_audio",
    "write_audio",
    "resample",
    "mux_audio",
    "add_subtitle_track",
    "burn_subtitles",
    "language_tag",
    "language_code_for_metadata",
    "measure_loudness",
    "normalize_loudness",
    "normalize_loudness_detailed",
    "limit_peaks",
    "moving_average",
    "save_normalized",
    "DEFAULT_SAMPLE_RATE",
    "TARGET_LUFS",
    "LIMIT_CEILING",
]

DEFAULT_SAMPLE_RATE = 48_000

#: Broadcast-style loudness target (EBU R128 / streaming convention, ~-16 LUFS).
#: Used only when the source track's loudness cannot be measured.
TARGET_LUFS = -16.0
_TARGET_LUFS = TARGET_LUFS

#: Peak ceiling the limiter enforces, just under full scale.
LIMIT_CEILING = 0.98


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
class SubtitleStream:
    """One subtitle stream in a container and the language it declares."""

    #: Absolute stream index, which is what ffmpeg ``-map 0:<index>`` expects.
    index: int
    language: str | None = None


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


#: ISO-639-1 codes plus the ISO-639-2/B and English spellings ffmpeg and ffprobe
#: commonly report, mapped to the two-letter tag we compare and write.
_LANGUAGE_TAGS: dict[str, str] = {
    "zh": "zh", "chi": "zh", "zho": "zh", "chinese": "zh",
    "en": "en", "eng": "en", "english": "en",
    "de": "de", "deu": "de", "ger": "de", "german": "de",
    "it": "it", "ita": "it", "italian": "it",
    "pt": "pt", "por": "pt", "portuguese": "pt",
    "es": "es", "spa": "es", "spanish": "es",
    "ja": "ja", "jpn": "ja", "japanese": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr",
    "ru": "ru", "rus": "ru", "russian": "ru",
}


def language_tag(value: str | None) -> str | None:
    """Normalise a subtitle language tag to an ISO-639-1 code when possible.

    ``fra``/``fre``/``French`` all become ``fr`` so comparing a container's
    tags against the job's target language does not depend on which spelling the
    muxer happened to use. Unknown tags are shortened to their first two letters.
    """
    if not value:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return _LANGUAGE_TAGS.get(key, key[:2] if len(key) >= 2 else key)


#: ISO-639-1 -> ISO-639-2/B. MP4's ``mov_text`` stores the three-letter form;
#: writing ``en`` is accepted by ffmpeg but silently leaves the track untagged.
_LANGUAGE_TAGS_3: dict[str, str] = {
    "zh": "chi", "en": "eng", "de": "ger", "it": "ita", "pt": "por",
    "es": "spa", "ja": "jpn", "ko": "kor", "fr": "fra", "ru": "rus",
}


def language_code_for_metadata(value: str | None) -> str | None:
    """The three-letter language code a muxed subtitle track should carry."""
    tag = language_tag(value)
    if tag is None:
        return None
    return _LANGUAGE_TAGS_3.get(tag, tag)


def probe_subtitle_streams(path: Path | str) -> list[SubtitleStream]:
    """Return the subtitle streams of ``path`` and their declared languages.

    Stream indices are the container's absolute indices, matching what ffmpeg's
    ``-map 0:<index>`` and ``-map -0:<index>`` expect.
    """
    require_binary("ffprobe", hint="Install ffmpeg (it ships ffprobe).")
    output = run_command(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)]
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed ffprobe output
        raise MediaError(f"Could not parse ffprobe output for {path}") from exc

    streams: list[SubtitleStream] = []
    for stream in payload.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue
        try:
            index = int(stream["index"])
        except (KeyError, TypeError, ValueError):
            continue
        tags = stream.get("tags") or {}
        streams.append(SubtitleStream(index=index, language=tags.get("language")))
    return streams


def _build_subtitle_map_args(
    existing: Sequence[SubtitleStream],
    language: str | None,
    *,
    replace: bool = True,
) -> tuple[list[str], int]:
    """ffmpeg ``-map`` arguments that keep every subtitle except the target's.

    Returns the arguments and how many preserved subtitle streams precede the
    newly added one — the index its ``-metadata:s:s:N`` and ``-c:s:N`` options
    need.

    Args:
        existing: The input's subtitle streams, as probed.
        language: The language of the track being added.
        replace: When true, drop an existing stream already tagged with
            ``language`` so the new one replaces it instead of duplicating it.

    Returns:
        ``(args, preserved)`` where ``args`` maps all subtitle streams and
        removes the target-language ones, and ``preserved`` counts what remains.
    """
    args = ["-map", "0:s?"]
    target = language_tag(language) if replace else None
    preserved = 0
    for stream in existing:
        if target is not None and language_tag(stream.language) == target:
            args.extend(["-map", f"-0:{stream.index}"])
        else:
            preserved += 1
    return args, preserved


def _escape_filter_path(path: Path | str) -> str:
    """Quote a path for ffmpeg's ``subtitles=filename=`` filter option."""
    text = (
        str(path)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
    )
    return f"'{text}'"


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
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as exc:
        # soundfile raises LibsndfileError (a RuntimeError) for unrecognised
        # containers, and OSError for unreadable paths. Both are expected user
        # mistakes, so turn them into the friendly error surface.
        raise MediaError(
            f"Could not read audio from {path}: {exc} "
            "If this is a video file, pass it with --video instead of --audio."
        ) from exc
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


def mux_audio(
    video: Path | str,
    audio: Path | str,
    out: Path | str,
    *,
    subtitle_srt: Path | str | None = None,
    subtitle_language: str | None = None,
    replace_subtitle_language: bool = True,
) -> Path:
    """Replace the video's audio track, copying the video stream untouched.

    The output is capped at the video's own duration, which stops a longer dub
    from extending the container. ``-shortest`` cannot be used for that: the SRT
    demuxer reports the *start* of the last cue as the end of its stream, so
    ffmpeg would truncate the video and audio to the last subtitle. The ``-map``
    flags avoid accidentally carrying over the original audio. When
    ``subtitle_srt`` is given, that track is added to the result (replacing an
    existing stream in the same language); subtitle streams in other languages
    are preserved.
    """
    require_binary("ffmpeg", hint="Install ffmpeg.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    subtitle_args: list[str] = []
    new_index = 0
    command = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(video),
        "-i", str(audio),
    ]
    if subtitle_srt is not None:
        command.extend(["-i", str(subtitle_srt)])
        existing = probe_subtitle_streams(video)
        subtitle_args, new_index = _build_subtitle_map_args(
            existing, subtitle_language, replace=replace_subtitle_language
        )
    else:
        # Preserve whatever subtitle tracks the source already carries, even when
        # no new one is being added: dropping another language would silently
        # remove information the viewer may still want.
        subtitle_args = ["-map", "0:s?"]

    command.extend([
        "-map", "0:v:0",
        "-map", "1:a:0",
        *subtitle_args,
    ])
    if subtitle_srt is not None:
        command.extend(["-map", "2:s:0"])
    command.extend([
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        # mov_text is the only subtitle codec MP4 carries. Existing tracks in
        # other languages are converted rather than dropped, so a source that
        # used SubRip or ASS still keeps its extra languages.
        "-c:s", "mov_text",
    ])
    if subtitle_srt is not None:
        tag = language_code_for_metadata(subtitle_language)
        if tag:
            command.extend([f"-metadata:s:s:{new_index}", f"language={tag}"])
    # Cap at the video's duration rather than using `-shortest`: the SRT
    # demuxer ends at the start of the last cue, which would cut the video and
    # audio short whenever a subtitle track is muxed in the same pass.
    duration = probe(video).duration_s
    if duration > 0:
        command.extend(["-t", f"{duration:.3f}"])
    else:
        command.append("-shortest")
    command.append(str(out))
    run_command(command)
    return out


def add_subtitle_track(
    video: Path | str,
    subtitle_srt: Path | str,
    out: Path | str,
    *,
    language: str | None = None,
    replace: bool = True,
) -> Path:
    """Embed an SRT as a soft, toggleable subtitle track.

    Every input stream is copied: the video, the original audio, and any
    subtitle streams in other languages. An existing stream already tagged with
    ``language`` is dropped and replaced, so re-running does not stack duplicate
    tracks.
    """
    require_binary("ffmpeg", hint="Install ffmpeg.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = probe_subtitle_streams(video)
    subtitle_args, new_index = _build_subtitle_map_args(
        existing, language, replace=replace
    )
    command = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(video),
        "-i", str(subtitle_srt),
        "-map", "0:v?",
        "-map", "0:a?",
        *subtitle_args,
        "-map", "1:s:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",
    ]
    tag = language_code_for_metadata(language)
    if tag:
        command.extend([f"-metadata:s:s:{new_index}", f"language={tag}"])
    command.append(str(out))
    run_command(command)
    return out


def burn_subtitles(
    video: Path | str,
    subtitle_srt: Path | str,
    out: Path | str,
    *,
    language: str | None = None,
) -> Path:
    """Render subtitles into the video frames, re-encoding the video.

    The audio and any non-target subtitle streams are copied. An existing
    subtitle stream already tagged with ``language`` is dropped, because its text
    is now part of the picture; streams in other languages stay toggleable.
    """
    require_binary("ffmpeg", hint="Install ffmpeg.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = probe_subtitle_streams(video)
    subtitle_args, _ = _build_subtitle_map_args(existing, language, replace=True)
    run_command(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(video),
            "-i", str(subtitle_srt),
            "-map", "0:v:0",
            "-map", "0:a?",
            *subtitle_args,
            "-vf", f"subtitles=filename={_escape_filter_path(subtitle_srt)}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-c:s", "mov_text",
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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average via cumulative sums (O(n), not O(n·window)).

    ``np.convolve`` would be prohibitively slow on a multi-minute timeline with a
    multi-thousand-sample window, which is the size the limiter needs.
    """
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1 or data.size == 0:
        return data.astype(np.float32)

    window = min(window, data.size)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(data, (pad_left, pad_right), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    return averaged.astype(np.float32)


def limit_peaks(
    samples: np.ndarray,
    sample_rate: int,
    *,
    ceiling: float = LIMIT_CEILING,
    release_ms: float = 150.0,
) -> np.ndarray:
    """Reduce peaks to ``ceiling`` without lowering the body of the signal.

    Synthesised speech is peakier than mastered audio (measured crest factor
    ~23 dB against the source's ~19 dB), and a few isolated transients would
    otherwise force the whole program to sit several dB too quiet: normalising
    for them costs the level of everything else.

    Gain is derived from an envelope that is the *maximum* of the instantaneous
    magnitude and a release-smoothed version of it. Because the envelope is never
    below the instantaneous magnitude, the resulting gain can never let a sample
    exceed the ceiling — no clipping by construction.

    Args:
        samples: Input audio.
        sample_rate: Sample rate in Hz.
        ceiling: Maximum absolute output level (1.0 = full scale).
        release_ms: Envelope release time; longer is smoother but reduces more.

    Returns:
        Limited audio (the input unchanged when it already fits).
    """
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0 or ceiling <= 0:
        return data

    magnitude = np.abs(data)
    peak = float(magnitude.max())
    if peak <= ceiling:
        return data

    window = max(1, int(sample_rate * release_ms / 1000.0))
    envelope = np.maximum(magnitude, moving_average(magnitude, window))
    gain = np.minimum(1.0, ceiling / np.maximum(envelope, 1e-12))
    return (data * gain).astype(np.float32)


def normalize_loudness(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float = _TARGET_LUFS,
) -> np.ndarray:
    """Apply a gain so the clip sits at ``target_lufs``.

    Peak growth is controlled by :func:`limit_peaks` rather than by refusing the
    gain, so the target is actually reachable on speech. See
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
    ceiling: float = LIMIT_CEILING,
    max_passes: int = 3,
    tolerance_lu: float = 0.3,
) -> "LoudnessResult":
    """Normalise loudness to ``target_lufs`` and report what happened.

    Unlike a plain peak-safe gain, the requested gain is *applied* and peaks are
    then limited, so speech (which has a high crest factor) can actually reach a
    broadcast target. Limiting changes loudness non-linearly, so the pass is
    repeated until the result is within ``tolerance_lu`` or ``max_passes`` is
    reached.

    Returns:
        A :class:`LoudnessResult` with the processed samples, the measured and
        achieved loudness, the total gain applied, and whether limiting engaged.
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

    current = data
    total_gain_db = 0.0
    limited = False

    for _ in range(max(1, max_passes)):
        present = measure_loudness(current, sample_rate)
        if present is None:
            break
        delta_db = target_lufs - present
        if abs(delta_db) <= tolerance_lu:
            break

        amplified = current * (10.0 ** (delta_db / 20.0))
        if float(np.max(np.abs(amplified))) > ceiling:
            limited = True
        current = limit_peaks(amplified, sample_rate, ceiling=ceiling)
        total_gain_db += delta_db

    return LoudnessResult(
        samples=current.astype(np.float32),
        target_lufs=target_lufs,
        measured_lufs=measured,
        achieved_lufs=measure_loudness(current, sample_rate),
        gain_db=total_gain_db,
        peak_limited=limited,
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
