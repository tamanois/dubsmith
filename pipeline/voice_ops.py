"""High-level voice-library operations shared by the CLI and the HTTP API.

The CLI's ``voices`` command and the web UI must create, list and remove voices
the same way. This module owns that workflow; ``dub.py`` and the API are thin
adapters over it.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .anchors import ANCHOR_TEXT
from .config import language_code_for_name, normalize_language, pick_dtype
from .errors import ConfigError
from .media import load_audio, write_audio
from .models import model_fingerprint, resolve_model
from .voices import (
    default_label,
    delete_voice,
    import_voice,
    list_voices,
    save_voice,
    slugify_label,
    unique_label,
)

__all__ = [
    "VoiceOpResult",
    "clone_voice",
    "design_voice",
    "list_voice_summaries",
    "register_voice",
    "remove_voice",
]

_LOGGER = logging.getLogger(__name__)

Event = Callable[[str, dict[str, Any]], None]
Log = logging.Logger | logging.LoggerAdapter


@dataclass
class VoiceOpResult:
    """What a voice operation produced."""

    label: str
    language: str
    source: str
    seconds: float
    directory: Path
    ref_text: str = ""
    cards: int = 0
    output: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["directory"] = str(self.directory)
        payload["output"] = str(self.output) if self.output is not None else None
        payload["has_transcript"] = bool(self.ref_text.strip())
        payload.pop("ref_text", None)
        return payload


def _emit(event: Event | None, kind: str, **payload: Any) -> None:
    if event is None:
        return
    try:
        event(kind, payload)
    except Exception:  # noqa: BLE001 - an observer must not fail the operation
        _LOGGER.debug("Voice observer failed for %s", kind, exc_info=True)


def _anchor_text(language: str) -> str:
    """The canonical audition passage for a language."""
    try:
        return ANCHOR_TEXT[language]
    except KeyError:
        raise ConfigError(
            f"No built-in passage for {language!r}; pick a supported language."
        ) from None


def _resolve_language(
    explicit: str | None,
    detected: str | None,
    *,
    fallback: str | None,
    log: Log = _LOGGER,
) -> str:
    """The language a voice is recorded in: explicit, else detected, else default."""
    if explicit:
        language = normalize_language(explicit)
        if language == "Auto":
            raise ConfigError("--lang must be an explicit language, not 'auto'.")
        if detected and normalize_language(detected) != language:
            log.warning(
                "The requested language is %s but the reference clip sounds like "
                "%s. The language is what 'say --voice' defaults to later.",
                language,
                normalize_language(detected),
            )
        return language
    if detected:
        return normalize_language(detected)
    if fallback is not None:
        return normalize_language(fallback)
    raise ConfigError(
        "The voice needs a language, or transcription to detect it from the audio."
    )


def _word_overlap(left: str, right: str) -> float:
    """Jaccard overlap of two transcripts' words, ``0.0``-``1.0``."""

    def words(value: str) -> set[str]:
        return set(re.findall(r"\w+", value.lower()))

    first, second = words(left), words(right)
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _transcribe_clip(
    samples: np.ndarray,
    sample_rate: int,
    models_dir: Path,
    *,
    language: str | None,
    device: str,
) -> tuple[str, str | None]:
    """Transcribe a reference clip with the local ASR."""
    from .asr import transcribe

    model_dir = resolve_model("whisper_large_v3", models_dir)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_audio(Path(tmp) / "reference.wav", samples, int(sample_rate))
        result = transcribe(
            path,
            model_dir,
            device=device,
            language=language_code_for_name(language),
        )
    text = " ".join(segment.text.strip() for segment in result.segments).strip()
    return text, result.language


def _reference_text(
    samples: np.ndarray,
    sample_rate: int,
    models_dir: Path,
    *,
    clone_text: str | None,
    transcribe: bool,
    verify: bool,
    language: str | None,
    device: str,
    required: bool,
    log: Log = _LOGGER,
) -> tuple[str, str | None]:
    """Resolve a reference clip's transcript for clone/add."""
    given = (clone_text or "").strip()
    if transcribe and given:
        raise ConfigError(
            "transcribe and clone_text are alternatives: pass one, not both."
        )
    heard, detected = "", None
    if transcribe or verify:
        heard, detected = _transcribe_clip(
            samples, sample_rate, models_dir, language=language, device=device
        )
        if not heard:
            raise ConfigError(
                "No speech was recognised in the reference clip; pass its "
                "transcript instead of asking for transcription."
            )
    if verify and given and heard:
        overlap = _word_overlap(given, heard)
        if overlap < 0.6:
            log.warning(
                "The transcript shares only %d%% of its words with what was heard. "
                "ICL cloning aligns the codes against this text, so this is likely "
                "to produce the wrong voice.",
                round(overlap * 100),
            )
    text = heard if transcribe else given
    if not text and required:
        raise ConfigError(
            "--clone needs the donated clip's transcript: pass --clone-text, or "
            "--transcribe to derive it from the audio."
        )
    return text, detected


def _write_speech(
    samples: np.ndarray,
    sample_rate: int,
    *,
    name: str,
    output: Path | None,
    output_root: Path,
) -> Path:
    if output is not None:
        target = output
    else:
        target = output_root / "say" / f"{slugify_label(name)}.wav"
    return write_audio(target, samples, int(sample_rate))


def design_voice(
    root: Path,
    models_dir: Path,
    *,
    prompt: str,
    language: str,
    label: str | None = None,
    speak: str | None = None,
    output: Path | None = None,
    output_root: Path = Path("output"),
    force: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    event: Event | None = None,
    log: Log = _LOGGER,
) -> VoiceOpResult:
    """Create a voice from a text description (VoiceDesign checkpoint)."""
    from .tts import VoiceDesignEngine

    if not prompt or not prompt.strip():
        raise ConfigError(
            "--design requires --prompt (a description of the voice)."
        )
    output_root = Path(output_root)
    output = Path(output) if output is not None else None
    canonical = _resolve_language(language, None, fallback="fr", log=log)
    anchor_text = _anchor_text(canonical)
    _emit(event, "stage_start", stage="design")
    engine = VoiceDesignEngine(
        resolve_model("tts_voice_design", models_dir),
        device=device,
        dtype=pick_dtype(device, dtype),
    )
    try:
        samples, sample_rate = engine.generate(anchor_text, prompt, canonical)
        chosen = label or unique_label(default_label(instruct=prompt), root)
        saved = save_voice(
            root,
            chosen,
            samples,
            sample_rate,
            language=canonical,
            source="design",
            ref_text=anchor_text,
            instruct=prompt,
            force=force,
            provenance={
                "backend": "voice-design",
                "model": str(resolve_model("tts_voice_design", models_dir)),
                "text_source": "anchor",
                "plan_source": "manual",
            },
        )
        out = None
        if speak:
            speech, speech_rate = engine.generate(speak, prompt, canonical)
            out = _write_speech(
                speech,
                speech_rate,
                name=saved.label,
                output=output,
                output_root=output_root,
            )
    finally:
        from .config import release_gpu_memory

        release_gpu_memory()
    _emit(event, "stage_end", stage="design", label=saved.label)
    return VoiceOpResult(
        label=saved.label,
        language=saved.language,
        source=saved.source,
        seconds=saved.seconds,
        directory=saved.directory,
        ref_text=saved.ref_text,
        cards=len(saved.cards),
        output=out,
    )


def clone_voice(
    root: Path,
    models_dir: Path,
    *,
    audio: Path,
    clone_text: str | None = None,
    transcribe: bool = False,
    verify: bool = False,
    language: str | None = None,
    label: str | None = None,
    speak: str | None = None,
    output: Path | None = None,
    output_root: Path = Path("output"),
    force: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    event: Event | None = None,
    log: Log = _LOGGER,
) -> VoiceOpResult:
    """Create a voice by encoding a donated clip once."""
    from .tts import TTSEngine

    output_root = Path(output_root)
    output = Path(output) if output is not None else None
    # Checked before the clip is read so a missing transcript is reported as such,
    # not as a decode error on a path that was fine all along.
    if not transcribe and not (clone_text or "").strip():
        raise ConfigError(
            "--clone needs the donated clip's transcript: pass --clone-text, or "
            "--transcribe to derive it from the audio."
        )
    try:
        samples, sample_rate = load_audio(audio)
    except Exception as exc:  # noqa: BLE001 - surfaced as a user error
        raise ConfigError(f"Could not read {audio}: {exc}") from exc
    ref_text, detected = _reference_text(
        samples,
        sample_rate,
        models_dir,
        clone_text=clone_text,
        transcribe=transcribe,
        verify=verify,
        language=language,
        device=device,
        required=True,
        log=log,
    )
    canonical = _resolve_language(language, detected, fallback="fr", log=log)
    _emit(event, "stage_start", stage="clone")
    engine = TTSEngine(
        resolve_model("tts_base", models_dir),
        device=device,
        dtype=pick_dtype(device, dtype),
    )
    try:
        engine.prepare_prompt_from_file(audio, ref_text)
        prompt = engine.prompt_to_data(
            fingerprint=model_fingerprint("tts_base", models_dir)
        )
        chosen = label or unique_label(default_label(audio=audio), root)
        saved = save_voice(
            root,
            chosen,
            samples,
            sample_rate,
            language=canonical,
            source="clone",
            ref_text=ref_text,
            prompt=prompt,
            force=force,
            provenance={
                "backend": "voice-clone",
                "model": str(resolve_model("tts_base", models_dir)),
                "ref_audio": str(audio),
                "text_source": "transcribe" if transcribe else "clone-text",
                "detected_language": detected,
                "fingerprint": model_fingerprint("tts_base", models_dir),
                "plan_source": "manual",
            },
        )
        out = None
        if speak:
            rendered = engine.synthesize([speak], language=canonical)
            speech = rendered[0] if rendered else None
            if speech is None:
                from .errors import TTSError

                raise TTSError("The cloning model returned no audio.")
            out = _write_speech(
                speech,
                engine.sample_rate,
                name=saved.label,
                output=output,
                output_root=output_root,
            )
    finally:
        from .config import release_gpu_memory

        release_gpu_memory()
    _emit(event, "stage_end", stage="clone", label=saved.label)
    return VoiceOpResult(
        label=saved.label,
        language=saved.language,
        source=saved.source,
        seconds=saved.seconds,
        directory=saved.directory,
        ref_text=saved.ref_text,
        cards=len(saved.cards),
        output=out,
    )


def register_voice(
    root: Path,
    models_dir: Path,
    *,
    audio: Path,
    clone_text: str | None = None,
    transcribe: bool = False,
    verify: bool = False,
    language: str | None = None,
    label: str | None = None,
    force: bool = False,
    device: str = "auto",
    event: Event | None = None,
    log: Log = _LOGGER,
) -> VoiceOpResult:
    """Register an existing clip as a voice with no model load."""
    if not audio.is_file():
        raise ConfigError(f"No such audio file: {audio}")
    try:
        samples, sample_rate = load_audio(audio)
    except Exception as exc:  # noqa: BLE001 - surfaced as a user error
        raise ConfigError(f"Could not read {audio}: {exc}") from exc
    if samples.size == 0:
        raise ConfigError(f"{audio} contains no audio.")
    warnings: list[str] = []
    ref_text, detected = _reference_text(
        samples,
        sample_rate,
        models_dir,
        clone_text=clone_text,
        transcribe=transcribe,
        verify=verify,
        language=language,
        device=device,
        required=False,
        log=log,
    )
    if language is None and detected is None:
        raise ConfigError(
            "--add needs --lang (the language spoken in the clip), or "
            "--transcribe to detect it from the audio."
        )
    canonical = _resolve_language(language, detected, fallback=None, log=log)
    seconds = float(samples.size) / float(sample_rate or 1)
    if seconds < 3.0:
        warnings.append(
            "The clip is only %.1fs long; a speaker embedding needs a few seconds "
            "of clean speech to be stable." % seconds
        )
    elif seconds > 30.0:
        warnings.append(
            "The clip is %.0fs long. All of it conditions the prompt, so a trimmed "
            "8-15 s excerpt usually clones better and faster." % seconds
        )
    for warning in warnings:
        log.warning(warning)
    chosen = label or unique_label(default_label(audio=audio), root)
    _emit(event, "stage_start", stage="register")
    saved = import_voice(
        root,
        chosen,
        (samples, sample_rate),
        language=canonical,
        ref_text=ref_text,
        force=force,
        provenance={
            "backend": "voice-add",
            "ref_audio": str(audio),
            "text_source": (
                "transcribe"
                if transcribe
                else ("clone-text" if (clone_text or "").strip() else "none")
            ),
            "detected_language": detected,
            "plan_source": "manual",
        },
    )
    _emit(event, "stage_end", stage="register", label=saved.label)
    return VoiceOpResult(
        label=saved.label,
        language=saved.language,
        source=saved.source,
        seconds=saved.seconds,
        directory=saved.directory,
        ref_text=saved.ref_text,
        cards=len(saved.cards),
        warnings=warnings,
    )


def list_voice_summaries(root: Path) -> list[dict[str, Any]]:
    """List saved voices without exposing their biometric transcript."""
    result: list[dict[str, Any]] = []
    for voice in list_voices(root):
        result.append(
            {
                "label": voice.label,
                "language": voice.language,
                "source": voice.source,
                "seconds": round(float(voice.seconds), 2),
                "gender": voice.gender,
                "rate_band": voice.rate_band,
                "instruct": voice.instruct,
                "has_transcript": bool(voice.ref_text.strip()),
                "cards": len(voice.cards),
                "directory": str(voice.directory),
            }
        )
    return result


def remove_voice(root: Path, label: str) -> Path:
    """Delete a saved voice and return where it used to live."""
    return delete_voice(root, label)
