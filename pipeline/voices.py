"""Saved, labeled voices: the design/clone/import voice library.

A *voice* is a directory holding one short WAV, a sidecar manifest and, for each
synthesiser that has encoded it, one *card* under ``prompts/``. The WAV is the
voice itself and works with every backend; a card is derived data that lets one
specific backend skip encoding the clip again.

Cards are named after the backend and the model configuration that produced them
(``prompts/qwen-d77fb851b8be.npz``), so a card is not portable between backends -
Qwen's speech codes mean nothing to Fish - and switching checkpoints keeps both
cards instead of rebuilding the first one later.

Three sources, one representation:

* ``design`` -- the VoiceDesign checkpoint renders a neutral, phonetically varied
  passage in the voice a text description asks for.
* ``clone`` -- a checkpoint encodes a donated clip (and its exact transcript)
  once, and the donor audio is kept beside the card, so the voice can still be
  heard, re-encoded by another model, or rebuilt from scratch.
* ``import`` -- an existing clip is registered verbatim, with no model involved.
  Its transcript is optional: without one the voice clones from the speaker
  embedding alone (x-vector mode), which is less faithful but needs no text.

The module holds only pure data handling and the cache. It never imports a model:
the synthesiser is supplied by the caller, so every function here is testable
without a GPU or downloaded weights. This mirrors :mod:`pipeline.anchors`, which
caches the per-language accent clips the same way.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .anchors import measure_gender, rate_band
from .errors import TTSError, VoiceNotFoundError

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "PromptData",
    "VoicePrint",
    "cards_in",
    "default_label",
    "delete_voice",
    "find_prompt",
    "import_voice",
    "list_voices",
    "load_prompt",
    "load_voice",
    "prompt_path",
    "save_prompt",
    "save_voice",
    "slugify_label",
    "unique_label",
    "voice_content",
    "voice_paths",
]

#: Names inside a voice's own directory. One directory per voice keeps the audio,
#: the manifest, the encodings and any future artifact (a preview) together, so a
#: voice can be copied or deleted as a unit.
WAV_NAME = "voice.wav"
MANIFEST_NAME = "voice.json"
#: Encodings live in their own folder, one file per backend and model. They are
#: derived data, and several may coexist: switching checkpoints keeps both cards.
PROMPT_DIR = "prompts"
#: Where an encoding written before cards were namespaced may still live. Read as
#: a Qwen card; never written to again.
LEGACY_PROMPT_NAME = "prompt.npz"

#: Manifest schema version, so a future change can migrate rather than guess.
#: v2 never listed the encodings; a v1 manifest loads unchanged.
MANIFEST_VERSION = 2

#: Where a voice came from. Anything else is a programming error.
_SOURCES = ("design", "clone", "import")

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


#: Version of the card payload, independent of the manifest version: the encoding
#: is produced by a different piece of code (the checkpoint wrapper) and may need
#: to change on its own schedule.
PROMPT_VERSION = 2


def _card_slug(value: str) -> str:
    """A path-safe fragment of a card's name.

    Backends and fingerprints are internal values, but a card is a file in a
    folder the user can edit, so both are reduced before they reach a path.
    """
    slug = _SLUG_SEPARATOR.sub("-", str(value).strip().lower()).strip("-")
    return slug[:32] or "unknown"


def voice_content(wav: Path | str, ref_text: str = "") -> str:
    """Short hash of what a card was derived from: the clip and its transcript.

    A card is only valid for the exact audio and text it was encoded from, so
    re-registering a voice with a different clip or transcript must not reuse it.
    The wav is read once (a megabyte or so) against an encode pass that costs
    seconds, and whitespace is folded so cosmetic edits do not invalidate a card.
    """
    digest = hashlib.sha256()
    try:
        digest.update(Path(wav).read_bytes())
    except OSError:
        digest.update(b"<missing>")
    digest.update(b"\x00")
    digest.update(" ".join(str(ref_text or "").split()).encode("utf-8"))
    return digest.hexdigest()[:12]


def prompt_path(
    directory: Path | str, backend: str, fingerprint: str
) -> Path:
    """Where the card for this backend and model configuration lives.

    The name is derived, so a card is found without opening anything:
    ``prompts/qwen-d77fb851b8be.npz``. One file per (backend, model), which is
    what lets two checkpoints keep their own card side by side.
    """
    return (
        Path(directory)
        / PROMPT_DIR
        / f"{_card_slug(backend)}-{_card_slug(fingerprint)}.npz"
    )


@dataclass(frozen=True)
class PromptData:
    """A clone prompt as plain data: what one checkpoint encodes a voice into.

    ``ref_code`` is the reference's speech codes (ICL mode only) and
    ``ref_spk_embedding`` is the speaker embedding. Both are numpy arrays, so
    this module stays torch-free and the file stays portable; the checkpoint
    moves them to its own device - and casts the embedding to the talker dtype -
    when it consumes them, which is what lets one stored card serve any card.

    The four provenance fields say *when the card may be used*: a card belongs to
    one backend (Qwen's codes mean nothing to Fish), one model configuration, and
    one specific clip + transcript. Anything else is a wrong voice, not a slow
    run.
    """

    #: Speech codes, shape ``(frames, quantizers)`` or ``(frames,)``. ``None`` in
    #: x-vector mode, where the codes are deliberately unused.
    ref_code: np.ndarray | None
    #: The speaker embedding, shape ``(dim,)``.
    ref_spk_embedding: np.ndarray
    #: True when only the embedding is used, so no transcript is needed.
    x_vector_only: bool = False
    #: Exact transcript the codes were aligned against; empty in x-vector mode.
    ref_text: str = ""
    #: Which synthesiser produced the encoding. Not portable between backends.
    backend: str = "qwen"
    #: Registry name of the checkpoint that produced it, for messages.
    model: str = "tts_base"
    #: Hash of the checkpoint configuration that produced the encoding.
    fingerprint: str = ""
    #: Hash of the clip and transcript it was derived from (:func:`voice_content`).
    content: str = ""


def save_prompt(directory: Path | str, data: PromptData) -> Path:
    """Write one card atomically and return its path.

    The destination is :func:`prompt_path`, so a second model never overwrites the
    first one's card. npz rather than a pickled tensor file: a voice directory is
    meant to be copied between machines, and loading a shared file must not be
    able to run code. Each card is a few tens of kilobytes.
    """
    directory = Path(directory)
    target = prompt_path(directory, data.backend, data.fingerprint)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".pending-{target.name}")
    content = data.content or voice_content(directory / WAV_NAME, data.ref_text)
    meta = {
        "version": PROMPT_VERSION,
        "backend": data.backend,
        "model": data.model,
        "fingerprint": data.fingerprint,
        "content": content,
        "x_vector_only": bool(data.x_vector_only),
        "ref_text": data.ref_text,
        "codes_shape": (
            None if data.ref_code is None else list(np.shape(data.ref_code))
        ),
        "embedding_dim": int(np.size(data.ref_spk_embedding)),
    }
    payload: dict[str, Any] = {
        "meta": np.array(json.dumps(meta, ensure_ascii=False)),
        "ref_spk_embedding": np.asarray(
            data.ref_spk_embedding, dtype=np.float32
        ),
    }
    if data.ref_code is not None:
        # Keep the tokenizer's integer dtype: the decoder concatenates these
        # codes with freshly generated ones, and a dtype change would break it.
        payload["ref_code"] = np.asarray(data.ref_code)
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, target)
    finally:
        # A failed save must not leave scratch files behind.
        tmp.unlink(missing_ok=True)
    _LOGGER.info("Saved the %s card to %s", data.backend, target)
    return target


def load_prompt(path: Path | str) -> PromptData:
    """Read a stored card.

    Raises:
        VoiceNotFoundError: If the file is missing, unreadable, or incomplete.
    """
    source = Path(path)
    if not source.is_file():
        raise VoiceNotFoundError(f"No voice card at {source}.")
    try:
        with np.load(source, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            embedding = np.asarray(archive["ref_spk_embedding"], dtype=np.float32)
            codes = (
                np.asarray(archive["ref_code"])
                if "ref_code" in archive.files
                else None
            )
    except (OSError, ValueError, KeyError) as exc:
        raise VoiceNotFoundError(
            f"Voice card at {source} is unreadable: {exc}"
        ) from exc
    if not isinstance(meta, dict):
        raise VoiceNotFoundError(f"Voice card at {source} has invalid metadata.")
    if embedding.size == 0:
        raise VoiceNotFoundError(f"Voice card at {source} has no speaker embedding.")
    x_vector_only = bool(meta.get("x_vector_only", False))
    if codes is None and not x_vector_only:
        raise VoiceNotFoundError(
            f"Voice card at {source} has no speech codes for ICL mode."
        )
    return PromptData(
        ref_code=codes,
        ref_spk_embedding=embedding,
        x_vector_only=x_vector_only,
        ref_text=str(meta.get("ref_text") or ""),
        backend=str(meta.get("backend") or "qwen"),
        model=str(meta.get("model") or meta.get("base_model") or "tts_base"),
        fingerprint=str(meta.get("fingerprint") or ""),
        content=str(meta.get("content") or ""),
    )


def cards_in(directory: Path | str) -> dict[tuple[str, str], Path]:
    """Every card stored in a voice folder, keyed by ``(backend, fingerprint)``.

    Best effort: an unreadable card is skipped rather than hiding the others, and
    a legacy root ``prompt.npz`` counts as whatever it declares itself to be.
    Derived data, so an empty result is normal.
    """
    directory = Path(directory)
    found: dict[tuple[str, str], Path] = {}
    for path in sorted((directory / PROMPT_DIR).glob("*.npz")):
        if path.name.startswith("."):
            continue
        try:
            data = load_prompt(path)
        except VoiceNotFoundError as exc:
            _LOGGER.debug("Skipping unreadable card %s: %s", path, exc)
            continue
        found.setdefault((data.backend, data.fingerprint), path)
    legacy = directory / LEGACY_PROMPT_NAME
    if legacy.is_file():
        try:
            data = load_prompt(legacy)
        except VoiceNotFoundError as exc:
            _LOGGER.debug("Skipping unreadable card %s: %s", legacy, exc)
        else:
            found.setdefault((data.backend, data.fingerprint), legacy)
    return found


def find_prompt(
    directory: Path | str,
    backend: str,
    fingerprint: str,
    *,
    ref_text: str = "",
) -> Path | None:
    """The card for exactly this backend, model and voice content, if any.

    An exact match only. Cards for other checkpoints are *kept* beside it, so
    switching models is instant in both directions, and a card whose clip or
    transcript no longer matches is treated as absent rather than trusted: it
    would be a different voice.
    """
    directory = Path(directory)
    expected = voice_content(directory / WAV_NAME, ref_text)
    candidate = prompt_path(directory, backend, fingerprint)
    options = [candidate]
    legacy = directory / LEGACY_PROMPT_NAME
    if legacy.is_file():
        options.append(legacy)
    for path in options:
        if not path.is_file():
            continue
        try:
            data = load_prompt(path)
        except VoiceNotFoundError as exc:
            _LOGGER.debug("Ignoring unreadable card %s: %s", path, exc)
            continue
        if data.backend != backend or data.fingerprint != fingerprint:
            continue
        # A card written before contents were recorded cannot be proven current,
        # so it is not used; the next encode replaces it in the new location.
        if data.content and data.content == expected:
            return path
    return None


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
    #: Stored encodings that belong to this voice, keyed by
    #: ``(backend, fingerprint)``. Derived data: any of them may be deleted, and
    #: several coexist because each model keeps its own card.
    cards: dict[tuple[str, str], Path] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        """The voice's own folder."""
        return self.wav.parent

    @property
    def manifest_path(self) -> Path:
        """Path of the sidecar manifest."""
        return self.directory / MANIFEST_NAME

    def card_for(self, backend: str, fingerprint: str) -> Path | None:
        """The stored encoding for one backend and model, if this voice has it."""
        return self.cards.get((backend, fingerprint))

    def to_manifest(self) -> dict[str, Any]:
        """Serialisable form, written as ``voice.json``.

        Encodings are deliberately absent: they are discovered on disk, so a card
        can be copied in or deleted without editing the manifest, and no entry
        can point at a file that no longer exists.
        """
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
        if self.source == "design" and self.instruct:
            origin = f'design: "{self.instruct}"'
        elif self.source == "import":
            origin = "import" if self.ref_text.strip() else "import (no transcript)"
        elif self.cards:
            backends = ", ".join(sorted({backend for backend, _ in self.cards}))
            origin = f"clone + {backends} card"
            if len(self.cards) > 1:
                origin += f" ×{len(self.cards)}"
        else:
            origin = "clone"
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
    # A designed or cloned voice is backed by audio whose transcript is known, and
    # ICL cloning needs it. An imported clip may legitimately have none: the voice
    # then clones from the speaker embedding alone.
    if source != "import" and not ref_text.strip():
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
        cards=_cards(wav),
    )


def _cards(wav: Path) -> dict[tuple[str, str], Path]:
    """The stored encodings beside ``wav``, discovered rather than declared.

    A card is derived data, so the manifest does not list them: one can be copied
    in, deleted, or belong to a model that is not installed right now, and none of
    that should touch ``voice.json``.
    """
    return cards_in(wav.parent)


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
    prompt: PromptData | None = None,
    provenance: dict[str, Any] | None = None,
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
        prompt: The checkpoint's encoding of ``samples``, when the caller already
            produced one. Stored under ``prompts/`` for that backend and model.
        provenance: Extra fields to record in the voice's timestamped provenance
            artifact (model, fingerprint, source clip, transcript origin).
        force: Overwrite an existing voice with the same label.

    Returns:
        The saved :class:`VoicePrint`.

    Raises:
        TTSError: If the clip is empty, the source is unknown, the label exists
            and ``force`` is not set, or a designed/cloned voice has no
            transcript.
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
    # A designed or cloned voice is rendered/encoded from a passage whose
    # transcript is known, and ICL cloning needs it. An imported clip may have
    # none, in which case it clones from the speaker embedding alone.
    if source != "import" and (not ref_text or not ref_text.strip()):
        raise TTSError(
            "Cannot save a voice without its transcript: the clone prompt needs it. "
            "Only an imported clip may be saved without one."
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
        if prompt is not None:
            # Written *after* the audio because the card records a hash of it.
            save_prompt(directory, prompt)
        tmp_json.write_text(
            json.dumps(print_.to_manifest(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_json, manifest)
    finally:
        # A failed save must not leave scratch files behind.
        tmp_wav.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)

    # Cards are never deleted here, not even on an overwrite: one written by a
    # different model stays valid for that model, and one for the *old* content is
    # rejected at lookup because it records the clip it was derived from.
    saved = replace(print_, cards=cards_in(directory))
    if provenance is not None:
        from .provenance import ProvenanceRecorder, timed_provenance_path

        recorder = ProvenanceRecorder(
            "voice",
            str(provenance.get("backend") or source),
            language=language,
            plan_source=str(provenance.get("plan_source") or "off"),
        )
        recorder.record_voice(
            mode=source,
            label=saved.label,
            prompt=instruct,
            instruct=instruct,
            ref_audio=provenance.get("ref_audio"),
            ref_text=ref_text,
            text_source=provenance.get("text_source"),
            model=provenance.get("model"),
            fingerprint=provenance.get("fingerprint"),
            verify=provenance.get("verify"),
            detected_language=provenance.get("detected_language"),
            seconds=round(seconds, 2),
            gender=saved.gender,
            rate_band=saved.rate_band,
        )
        recorder.write(timed_provenance_path(directory / "provenance"))
    _LOGGER.info(
        "Saved %s voice %r (%s, %.1fs)%s to %s",
        saved.source,
        saved.label,
        saved.language,
        saved.seconds,
        f" with {len(saved.cards)} card(s)" if saved.cards else "",
        directory,
    )
    return saved


def import_voice(
    root: Path | str,
    label: str,
    audio: Path | str | tuple[np.ndarray, int],
    *,
    language: str,
    ref_text: str = "",
    provenance: dict[str, Any] | None = None,
    force: bool = False,
) -> VoicePrint:
    """Register an existing clip as a voice, with no model work at all.

    Nothing is rendered or encoded: the clip is copied into the library (mono,
    at its own rate) and the transcript is stored exactly as given. The first
    consumer that needs an encoding pays for it once, and can then cache it with
    :func:`save_prompt`.

    Args:
        root: Voice-library directory.
        label: Name to save under; slugified before use.
        audio: The clip's path, or an ``(samples, sample_rate)`` pair when the
            caller already decoded it (to transcribe it, for instance).
        language: Language spoken in the clip.
        ref_text: Its exact transcript. Optional: without one the voice clones
            from the speaker embedding alone, which is less faithful.
        force: Overwrite an existing voice with the same label.

    Returns:
        The saved :class:`VoicePrint`.

    Raises:
        TTSError: If the clip cannot be read or holds no audio.
    """
    if isinstance(audio, tuple):
        samples, sample_rate = audio
    else:
        from .media import load_audio

        try:
            samples, sample_rate = load_audio(audio)
        except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing error
            raise TTSError(f"Could not read {audio}: {exc}") from exc
    return save_voice(
        root,
        label,
        samples,
        sample_rate,
        language=language,
        source="import",
        ref_text=ref_text,
        provenance=provenance,
        force=force,
    )


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
