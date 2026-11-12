"""Shared job mutations that turn a saved voice label into a ``JobConfig``.

The CLI's ``job_config_from_args`` and the HTTP API both need the same mapping:
a saved voice is a reference clip whose transcript (and optionally its stored
encoding) is already known, and its language decides whether the target accent
should be taken from the voice itself.
"""

from __future__ import annotations

from pathlib import Path

from .config import JobConfig, normalize_language
from .errors import ConfigError
from .models import model_fingerprint
from .voices import find_prompt, load_voice

__all__ = ["apply_accent_voice", "apply_saved_voice"]


def apply_saved_voice(
    job: JobConfig,
    label: str,
    *,
    voices_root: Path,
    models_dir: Path,
) -> JobConfig:
    """Resolve ``--voice`` into reference audio, transcript and optional card."""
    if job.ref_audio is not None:
        raise ConfigError(
            "A saved voice cannot be combined with a reference clip: a saved "
            "voice already is one. Pick one."
        )
    saved = load_voice(voices_root, label)
    card = find_prompt(
        saved.directory,
        "qwen",
        model_fingerprint("tts_base", models_dir),
        ref_text=saved.ref_text,
    )
    updates: dict[str, object] = {
        "ref_audio": saved.wav,
        "ref_text": saved.ref_text,
        "ref_prompt": card,
        "ref_prompt_seconds": saved.seconds,
    }
    # Only infer when the caller left the default in place. An explicit mode is
    # always honoured.
    if job.accent_mode == "native":
        if not saved.ref_text.strip():
            updates["accent_mode"] = "speaker-only"
        elif normalize_language(saved.language) == job.target_language:
            updates["accent_mode"] = "preserve"
    return job.model_copy(update=updates)


def apply_accent_voice(
    job: JobConfig,
    label: str,
    *,
    voices_root: Path,
) -> JobConfig:
    """Resolve ``--accent-voice`` into the accent clip and its transcript."""
    if job.accent_audio is not None or (job.accent_text or "").strip():
        raise ConfigError(
            "A saved accent voice already supplies both the clip and its "
            "transcript; do not pass them separately."
        )
    saved = load_voice(voices_root, label)
    if not (saved.ref_text or "").strip():
        raise ConfigError(
            f"Voice {label!r} has no transcript, so its speech codes cannot be "
            "aligned. Add the transcript, or supply the accent clip directly."
        )
    if normalize_language(saved.language) != job.target_language:
        raise ConfigError(
            f"Voice {label!r} is a {saved.language} voice, so it cannot supply a "
            f"native {job.target_language} accent."
        )
    return job.model_copy(
        update={
            "accent_audio": saved.wav,
            "accent_text": saved.ref_text,
            "accent_mode": "reference-accent",
        }
    )
