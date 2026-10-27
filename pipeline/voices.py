"""Saved, labeled voice prints: the design/clone voice library.

A *voice print* is one short WAV plus a sidecar manifest. It is the reusable form
of a voice: the Base checkpoint clones from the WAV (with its exact transcript),
so the print can be used for as many lines, languages and jobs as the user wants
without regenerating it.

Two sources, one representation:

* ``design`` -- the VoiceDesign checkpoint renders a neutral, phonetically varied
  passage in the voice a text description asks for.
* ``clone`` -- the Base checkpoint renders that same passage in a voice cloned
  from a donated clip (and its transcript). Storing the *rendered passage* rather
  than the donor keeps every print a consistent length and format.

The module holds only pure data handling and the cache. It never imports a model:
the synthesiser is supplied by the caller, so every function here is testable
without a GPU or downloaded weights. This mirrors :mod:`pipeline.anchors`, which
caches the per-language accent clips the same way.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .anchors import measure_gender, rate_band
from .errors import TTSError, VoiceNotFoundError

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "VoicePrint",
    "default_label",
    "delete_voice",
    "list_voices",
    "load_voice",
    "save_voice",
    "slugify_label",
    "unique_label",
    "voice_paths",
]

#: Names inside a voice's own directory. One directory per voice keeps the audio,
#: the manifest and any future artifact (an embedding, a preview) together, so a
#: voice can be copied or deleted as a unit.
WAV_NAME = "voice.wav"
MANIFEST_NAME = "voice.json"

#: Manifest schema version, so a future change can migrate rather than guess.
MANIFEST_VERSION = 1

#: Where a voice print came from. Anything else is a programming error.
_SOURCES = ("design", "clone")

#: Characters a label may keep. Everything else becomes a separator, which keeps
#: labels safe as directory names and immune to path traversal.
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")

#: Longest auto-generated label. Long enough to stay descriptive, short enough to
#: type and to keep paths tidy.
_MAX_LABEL = 48


def slugify_label(value: str) -> str:
    """Normalise a voice label into a safe, filesystem-friendly key.

    Labels are the lookup key *and* the directory name, so this is also the guard
    against path traversal: only ``[a-z0-9]`` survives, and runs of anything else
    collapse to a single hyphen. Case is folded so ``Warm-FR`` and ``warm-fr``
    are the same voice.

    Args:
        value: A user-supplied or derived label.

    Returns:
        A non-empty lowercase slug.

    Examples:
        >>> slugify_label("Warm FR ")
        'warm-fr'
        >>> slugify_label("../etc/passwd")
        'etc-passwd'
    """
    slug = _SLUG_SEPARATOR.sub("-", str(value).strip().lower()).strip("-")
    return (slug or "voice")[:_MAX_LABEL].rstrip("-") or "voice"


def unique_label(preferred: str, root: Path | str) -> str:
    """A label that is not already taken under ``root``.

    Appends ``-2``, ``-3``, ... to the slugified preference, so an auto-named
    voice never silently overwrites an existing one. The suffix is counted inside
    the length limit, because ``save_voice`` slugifies the label again: a
    suffixed name that overflowed would be truncated back onto the original and
    collide.
    """
    base = slugify_label(preferred)
    directory = Path(root)
    if not (directory / base).exists():
        return base
    suffix = 2
    while True:
        suffix_text = f"-{suffix}"
        stem = base[: _MAX_LABEL - len(suffix_text)].rstrip("-") or "voice"
        candidate = f"{stem}{suffix_text}"
        if not (directory / candidate).exists():
            return candidate
        suffix += 1


def default_label(
    *,
    instruct: str | None = None,
    audio: Path | str | None = None,
) -> str:
    """Derive a label when the user did not name the voice.

    A description makes a better name than a timestamp: it says what the voice
    *is*. A clone falls back to the donated clip's filename. The caller is still
    responsible for de-duplicating with :func:`unique_label`.
    """
    if instruct and instruct.strip():
        return slugify_label(instruct)
    if audio is not None:
        return slugify_label(Path(audio).stem)
    return "voice"


def voice_paths(root: Path | str, label: str) -> tuple[Path, Path, Path]:
    """The ``(directory, wav, manifest)`` paths for a voice, existing or not.

    Pure path arithmetic; nothing is created or read.
    """
    directory = Path(root) / slugify_label(label)
    return directory, directory / WAV_NAME, directory / MANIFEST_NAME


@dataclass(frozen=True)
class VoicePrint:
    """A saved voice: its reference clip, transcript and provenance."""

    #: Lookup key and directory name (already slugified when saved).
    label: str
    #: Canonical language name the passage was rendered in. It is the voice's
    #: *home* language: a print is language-agnostic for cloning, but this is what
    #: the design render used and what ``say`` defaults to.
    language: str
    #: ``design`` or ``clone``.
    source: str
    #: The reference WAV the Base checkpoint clones from.
    wav: Path
    #: Exact transcript of :attr:`wav`. ICL cloning requires it, and storing it
    #: here keeps every print self-contained.
    ref_text: str
    #: The description the voice was designed from, when ``source == "design"``.
    instruct: str | None = None
    #: ISO-8601 UTC creation time.
    created: str = ""
    #: Measured pitch class: ``f``, ``m`` or ``u`` (unknown).
    gender: str = "u"
    #: Measured pace band: ``slow``, ``normal`` or ``fast``.
    rate_band: str = "normal"
    #: Length of the reference clip in seconds.
    seconds: float = 0.0

    @property
    def directory(self) -> Path:
        """The voice's own folder."""
        return self.wav.parent

    @property
    def manifest_path(self) -> Path:
        """Path of the sidecar manifest."""
        return self.directory / MANIFEST_NAME

    def to_manifest(self) -> dict[str, Any]:
        """Serialisable form, written as ``voice.json``."""
        return {
            "version": MANIFEST_VERSION,
            "label": self.label,
            "language": self.language,
            "source": self.source,
            "instruct": self.instruct,
            "ref_text": self.ref_text,
            "created": self.created,
            "gender": self.gender,
            "rate_band": self.rate_band,
            "seconds": round(float(self.seconds), 2),
        }

    def summary(self) -> str:
        """One-line human description for ``voices --list``."""
        origin = (
            f'design: "{self.instruct}"'
            if self.source == "design" and self.instruct
            else self.source
        )
        return (
            f"{self.label:<24}{self.language:<12}{self.seconds:6.1f}s  "
            f"{self.gender}  {self.rate_band:<7}{origin}"
        )


def _manifest_from_dict(payload: dict[str, Any], *, wav: Path) -> VoicePrint:
    """Build a :class:`VoicePrint` from parsed JSON, filling optional fields."""
    manifest = wav.parent / MANIFEST_NAME
    try:
        label = str(payload["label"])
        language = str(payload["language"])
    except KeyError as exc:
        raise VoiceNotFoundError(
            f"Voice manifest at {manifest} is missing {exc.args[0]!r}."
        ) from exc
    source = str(payload.get("source", "design"))
    if source not in _SOURCES:
        raise VoiceNotFoundError(
            f"Voice {label!r} has an unknown source {source!r} in {manifest}."
        )
    ref_text = str(payload.get("ref_text") or "")
    if not ref_text.strip():
        raise VoiceNotFoundError(
            f"Voice {label!r} has no transcript in {manifest}; it cannot be used to clone."
        )
    instruct_raw = payload.get("instruct")
    instruct = (
        str(instruct_raw)
        if isinstance(instruct_raw, str) and instruct_raw.strip()
        else None
    )
    try:
        seconds = float(payload.get("seconds") or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    return VoicePrint(
        label=label,
        language=language,
        source=source,
        wav=wav,
        ref_text=ref_text,
        instruct=instruct,
        created=str(payload.get("created") or ""),
        gender=str(payload.get("gender") or "u"),
        rate_band=str(payload.get("rate_band") or "normal"),
        seconds=seconds,
    )


def save_voice(
    root: Path | str,
    label: str,
    samples: np.ndarray,
    sample_rate: int,
    *,
    language: str,
    source: str,
    ref_text: str,
    instruct: str | None = None,
    force: bool = False,
) -> VoicePrint:
    """Write a voice print to ``root/<label>/``.

    Both files are written to temporaries in the destination directory and then
    renamed into place, so an interrupted save cannot leave a half-written voice
    that later loads as valid.

    Args:
        root: Voice-library directory.
        label: User-supplied or derived name; slugified before use.
        samples: The reference clip (mono float32).
        sample_rate: Its sample rate in Hz.
        language: Canonical language the clip speaks.
        source: ``design`` or ``clone``.
        ref_text: Exact transcript of ``samples``.
        instruct: The description, for designed voices.
        force: Overwrite an existing voice with the same label.

    Returns:
        The saved :class:`VoicePrint`.

    Raises:
        TTSError: If the clip is empty, the source is unknown, or the label exists
            and ``force`` is not set.
    """
    if source not in _SOURCES:
        raise TTSError(
            f"Unknown voice source {source!r}. Use one of: {', '.join(_SOURCES)}"
        )
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0:
        raise TTSError("Cannot save a voice: the reference clip is empty.")
    if int(sample_rate) <= 0:
        raise TTSError("Cannot save a voice: the sample rate must be positive.")
    if not ref_text or not ref_text.strip():
        raise TTSError(
            "Cannot save a voice without its transcript: the clone prompt needs it."
        )

    directory, wav, manifest = voice_paths(root, label)
    exists = wav.exists() or manifest.exists()
    if exists and not force:
        raise TTSError(
            f"Voice {slugify_label(label)!r} already exists at {directory}. "
            "Pass --force to overwrite it, or choose another label."
        )
    directory.mkdir(parents=True, exist_ok=True)

    seconds = data.size / float(sample_rate)
    words = len(ref_text.split())
    wps = words / seconds if seconds > 0.0 else 0.0

    print_ = VoicePrint(
        label=slugify_label(label),
        language=language,
        source=source,
        wav=wav,
        ref_text=ref_text,
        instruct=instruct,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        gender=measure_gender(data, int(sample_rate)),
        rate_band=rate_band(wps),
        seconds=seconds,
    )

    tmp_wav = directory / f".pending-{WAV_NAME}"
    tmp_json = directory / f".pending-{MANIFEST_NAME}"
    try:
        from .media import write_audio

        write_audio(tmp_wav, data, int(sample_rate))
        os.replace(tmp_wav, wav)
        tmp_json.write_text(
            json.dumps(print_.to_manifest(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_json, manifest)
    finally:
        # A failed save must not leave scratch files behind.
        tmp_wav.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)

    _LOGGER.info(
        "Saved %s voice %r (%s, %.1fs) to %s",
        print_.source,
        print_.label,
        print_.language,
        print_.seconds,
        directory,
    )
    return print_


def load_voice(root: Path | str, label: str) -> VoicePrint:
    """Read a saved voice print by label.

    Args:
        root: Voice-library directory.
        label: Label as shown by ``voices --list`` (case-insensitive).

    Returns:
        The stored :class:`VoicePrint`.

    Raises:
        VoiceNotFoundError: If the voice, its clip or its manifest is missing, or
            the manifest is unreadable.
    """
    directory, wav, manifest = voice_paths(root, label)
    if not wav.is_file() or not manifest.is_file():
        available = [item.label for item in list_voices(root)]
        hint = (
            f" Available: {', '.join(available)}."
            if available
            else " The library is empty."
        )
        raise VoiceNotFoundError(
            f"No saved voice named {slugify_label(label)!r} in {Path(root)}.{hint} "
            "Create one with 'python dub.py voices --design ...' or "
            "'python dub.py voices --clone ...', or list them with "
            "'python dub.py voices --list'."
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VoiceNotFoundError(
            f"Voice {slugify_label(label)!r} has an unreadable manifest at {manifest}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise VoiceNotFoundError(
            f"Voice {slugify_label(label)!r} has an invalid manifest at {manifest}."
        )
    return _manifest_from_dict(payload, wav=wav)


def list_voices(root: Path | str) -> list[VoicePrint]:
    """Every readable voice print under ``root``, sorted by label.

    A directory with a missing or corrupt manifest is skipped with a debug log
    rather than failing the listing: one bad voice must not hide the others.
    """
    directory = Path(root)
    if not directory.is_dir():
        return []
    found: list[VoicePrint] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        wav = child / WAV_NAME
        manifest = child / MANIFEST_NAME
        if not wav.is_file() or not manifest.is_file():
            continue
        try:
            found.append(load_voice(directory, child.name))
        except VoiceNotFoundError as exc:
            _LOGGER.debug("Skipping unreadable voice %s: %s", child, exc)
    return sorted(found, key=lambda item: item.label)


def delete_voice(root: Path | str, label: str) -> Path:
    """Remove a saved voice and return the directory that was deleted.

    Raises:
        VoiceNotFoundError: If the voice does not exist.
    """
    directory, wav, manifest = voice_paths(root, label)
    if not wav.is_file() and not manifest.is_file():
        raise VoiceNotFoundError(
            f"No saved voice named {slugify_label(label)!r} in {Path(root)}."
        )
    import shutil

    shutil.rmtree(directory)
    _LOGGER.info("Deleted voice %r from %s", slugify_label(label), directory)
    return directory
