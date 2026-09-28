"""Job configuration, environment settings and language normalisation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .diarize import DEFAULT_THRESHOLD, MIN_SPEAKER_S
from .errors import ConfigError
from .mixing import (
    DEFAULT_ROOM_DUCK_DB,
    DEFAULT_ROOM_TONE_GAIN_DB,
    DEFAULT_ROOM_WET_DB,
)

__all__ = [
    "ACCELERATORS",
    "DTYPE_ALIASES",
    "DeviceInfo",
    "LANGUAGE_ALIASES",
    "SUPPORTED_LANGUAGES",
    "available_accelerators",
    "normalize_dtype",
    "normalize_language",
    "normalize_loudness_policy",
    "pick_dtype",
    "preferred_dtype",
    "probe_device",
    "release_gpu_memory",
    "resolve_device",
    "supports_bf16",
    "supports_fp16",
    "JobConfig",
    "Settings",
    "read_env_file",
]


# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #

#: Language names accepted by the Qwen3-TTS checkpoints. "Auto" lets the model
#: adapt to the text; it is only valid for the *target* language.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "Auto",
        "Chinese",
        "English",
        "German",
        "Italian",
        "Portuguese",
        "Spanish",
        "Japanese",
        "Korean",
        "French",
        "Russian",
    }
)

#: Convenience aliases so users can pass ``--target fr`` instead of ``French``.
LANGUAGE_ALIASES: dict[str, str] = {
    "auto": "Auto",
    "zh": "Chinese", "cn": "Chinese", "chi": "Chinese", "chinese": "Chinese",
    "en": "English", "eng": "English", "english": "English",
    "de": "German", "deu": "German", "ger": "German", "german": "German",
    "it": "Italian", "ita": "Italian", "italian": "Italian",
    "pt": "Portuguese", "por": "Portuguese", "portuguese": "Portuguese",
    "es": "Spanish", "spa": "Spanish", "spanish": "Spanish",
    "ja": "Japanese", "jp": "Japanese", "jpn": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "kor": "Korean", "korean": "Korean",
    "fr": "French", "fra": "French", "fre": "French", "french": "French",
    "ru": "Russian", "rus": "Russian", "russian": "Russian",
}

#: Human-readable language names keyed by ISO-639-1 code, used to describe the
#: language a Whisper model detected so the LLM prompt can name it.
CODE_TO_NAME: dict[str, str] = {
    "zh": "Chinese", "en": "English", "de": "German", "it": "Italian",
    "pt": "Portuguese", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "ru": "Russian",
}

#: Inverse of :data:`CODE_TO_NAME`; Whisper expects a bare code, not a name.
NAME_TO_CODE: dict[str, str] = {name: code for code, name in CODE_TO_NAME.items()}


def normalize_language(value: str) -> str:
    """Map a user-supplied language (``fr``, ``french``, ``French``) to canonical form.

    Args:
        value: Language code, alias, or full name.

    Returns:
        One of :data:`SUPPORTED_LANGUAGES`.

    Raises:
        ConfigError: If the language is unknown.
    """
    key = str(value).strip().lower()
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    # Allow exact canonical names that are not also aliases (e.g. "Auto").
    for canonical in SUPPORTED_LANGUAGES:
        if canonical.lower() == key:
            return canonical
    supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
    raise ConfigError(f"Unsupported language {value!r}. Supported: {supported}")


def language_name_for_code(code: str | None) -> str | None:
    """Best-effort mapping of a Whisper language code to a display name."""
    if not code:
        return None
    return CODE_TO_NAME.get(code.strip().lower(), code)


def language_code_for_name(name: str | None) -> str | None:
    """Map a canonical language name to the code Whisper expects.

    Returns ``None`` (auto-detect) when the name is unknown or unset, which is
    safer than passing a full name Whisper would reject.
    """
    if not name:
        return None
    key = str(name).strip()
    if key.lower() == "auto":
        return None
    if key.lower() in CODE_TO_NAME:
        return key.lower()
    return NAME_TO_CODE.get(key, None)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

ENV_PREFIX = ""

#: Accepted spellings for the loudness policy, normalised to two canonical forms.
_LOUDNESS_POLICIES: dict[str, str] = {
    "match-source": "match-source",
    "match_source": "match-source",
    "match": "match-source",
    "source": "match-source",
    "lufs": "lufs",
    "fixed": "lufs",
    "absolute": "lufs",
}


def normalize_loudness_policy(value: str) -> str:
    """Map a user-supplied loudness policy onto ``match-source`` or ``lufs``."""
    key = str(value).strip().lower()
    try:
        return _LOUDNESS_POLICIES[key]
    except KeyError:
        supported = ", ".join(sorted(set(_LOUDNESS_POLICIES.values())))
        raise ConfigError(f"Unknown loudness target {value!r}. Use: {supported}") from None


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` dotenv file.

    Supports ``export`` prefixes, ``#`` comments, blank lines, and single/double
    quoted values. Deliberately dependency-free: this project only needs the
    tiny subset of dotenv semantics and adding a package for it would be
    disproportionate.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


class Settings(BaseModel):
    """Runtime settings, sourced from environment variables and ``.env``."""

    model_config = ConfigDict(extra="forbid")

    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "local-model"
    llm_timeout_s: float = 180.0
    llm_chunk_size: int = Field(default=40, ge=1)

    # Loudness policy. "match-source" puts the dub at the level of the track it
    # replaces; "lufs" forces the absolute target below.
    loudness_target: str = "match-source"
    target_lufs: float = Field(default=-16.0, ge=-60.0, le=0.0)

    # Background preservation.
    keep_background: bool = True
    separator: str = "auto"
    duck_db: float = Field(default=6.0, ge=0.0, le=30.0)

    models_dir: Path = Path("models")
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("loudness_target")
    @classmethod
    def _check_loudness_policy(cls, value: str) -> str:
        return normalize_loudness_policy(value)

    @classmethod
    def from_env(
        cls,
        env_file: Path | str = ".env",
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Build settings from ``.env`` then the process environment.

        Real environment variables take precedence over the file, so a user can
        override a checked-in default without editing anything.
        """
        source = dict(environ if environ is not None else os.environ)
        file_values = read_env_file(Path(env_file))
        merged: dict[str, str] = {**file_values, **source}

        def pick(name: str, default: Any) -> Any:
            raw = merged.get(f"{ENV_PREFIX}{name}")
            if raw is None:
                return default
            if isinstance(default, bool):
                return str(raw).strip().lower() in {"1", "true", "yes", "on"}
            return raw

        payload = {
            "llm_base_url": pick("LLM_BASE_URL", cls.model_fields["llm_base_url"].default),
            "llm_api_key": pick("LLM_API_KEY", cls.model_fields["llm_api_key"].default),
            "llm_model": pick("LLM_MODEL", cls.model_fields["llm_model"].default),
            "llm_timeout_s": pick("LLM_TIMEOUT_S", cls.model_fields["llm_timeout_s"].default),
            "llm_chunk_size": pick("LLM_CHUNK_SIZE", cls.model_fields["llm_chunk_size"].default),
            "loudness_target": pick(
                "LOUDNESS_TARGET", cls.model_fields["loudness_target"].default
            ),
            "target_lufs": pick("LOUDNESS_LUFS", cls.model_fields["target_lufs"].default),
            "keep_background": pick(
                "KEEP_BACKGROUND", cls.model_fields["keep_background"].default
            ),
            "separator": pick("BACKGROUND_SEPARATOR", cls.model_fields["separator"].default),
            "duck_db": pick("DUCK_DB", cls.model_fields["duck_db"].default),
            "models_dir": pick("MODELS_DIR", cls.model_fields["models_dir"].default),
            "log_level": pick("LOG_LEVEL", cls.model_fields["log_level"].default),
            "log_json": pick("LOG_JSON", cls.model_fields["log_json"].default),
        }
        return cls.model_validate(payload)


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #


class JobConfig(BaseModel):
    """Everything needed to dub one video (or audio file)."""

    model_config = ConfigDict(extra="forbid")

    # Input: exactly one of these.
    video: Path | None = None
    audio: Path | None = None

    # Canonical name, not a code: pydantic validators do not run on defaults, so a
    # default of "fr" would bypass normalisation and leak a raw code downstream.
    target_language: str = "French"
    source_language: str | None = None  # None -> auto-detect

    # Voice donor: if omitted, the longest clean speech segment is used.
    ref_audio: Path | None = None
    ref_text: str | None = None

    output_dir: Path = Path("output")
    device: str = "auto"
    #: torch dtype for the TTS checkpoint. ``auto`` reads the device's capability:
    #: bfloat16 on Ampere and newer, float16 on older accelerators, float32 on
    #: CPU. An explicit value is honoured unless the device cannot run it.
    dtype: str = "auto"
    whisper_model: str = "large-v3"
    llm_model: str | None = None  # overrides Settings.llm_model

    # Duration fitting.
    min_stretch: float = Field(default=0.85, gt=0.05, le=1.0)
    max_stretch: float = Field(default=1.18, ge=1.0, le=3.0)

    # Length repair. Translations routinely come out longer than their slot;
    # re-wording the offending lines fits them without deleting speech.
    #: Rewrite the lines that would otherwise have words cut, then re-synthesise
    #: only those. Lines that already fit are never touched.
    refine: bool = True
    #: How many times a line may be rewritten before falling back to trimming.
    refine_max_passes: int = Field(default=1, ge=0, le=5)
    #: Seconds of overrun to tolerate before a line is worth rewriting.
    refine_threshold: float = Field(default=0.05, ge=0.0, le=5.0)

    # How the translator is told to size each line.
    #   "off"   -> only the slot duration is given (the long-standing behaviour)
    #   "words" -> also give a word ceiling derived from the measured speaking
    #              rate, which a language model can actually act on
    length_budget: str = "off"

    # ASR gating: segments the model itself distrusts are dropped.
    no_speech_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # Loudness policy; None inherits from Settings.
    #   "match-source" -> sit at the level of the track being replaced
    #   "lufs"         -> force target_lufs
    loudness_target: str | None = None
    target_lufs: float | None = Field(default=None, ge=-60.0, le=0.0)

    # Background preservation (music, ambience, room tone).
    keep_background: bool = True
    #: Separator backend: auto | demucs | none.
    separator: str = "auto"
    #: Attenuation applied to the background while dialogue plays.
    duck_db: float = Field(default=6.0, ge=0.0, le=30.0)
    #: Static offset applied to the background stem (lift or lower the bed).
    background_gain_db: float = Field(default=0.0, ge=-30.0, le=30.0)

    # Room matching. A dry, close voice-over is the *correct* choice for narrated
    # content, so it stays the default; "match" places the dub in the source's own
    # acoustic space instead of laying it on top of the picture.
    #   "off"   -> dry dub over the ducked bed (the long-standing behaviour)
    #   "match" -> reverb, a continuous room-tone floor and a softer duck, all
    #              derived from the separated speech stem
    room: str = "off"
    #: RT60 in seconds to use instead of measuring it from the speech stem. An
    #: escape hatch for a source whose decay cannot be estimated (or for A/B tests).
    room_rt60: float | None = Field(default=None, ge=0.05, le=5.0)
    #: Level of the reverb tail relative to the dry voice. ``0`` (the default) means
    #: **no reverb**: the convolutive stage is opt-in, because it can only be judged
    #: by ear and the first default was rejected on a real listen as an audible echo.
    room_wet_db: float = Field(default=DEFAULT_ROOM_WET_DB, ge=-60.0, le=0.0)
    #: How far the bed is pulled down under dialogue when matching a room. Shallower
    #: than ``duck_db``: the room is part of the voice, so a deep duck fights it.
    room_duck_db: float = Field(default=DEFAULT_ROOM_DUCK_DB, ge=0.0, le=30.0)
    #: Level of the room-tone floor taken from the speech stem's non-speech parts.
    room_tone_gain_db: float = Field(
        default=DEFAULT_ROOM_TONE_GAIN_DB, ge=-60.0, le=0.0
    )

    # Voice accent.
    #   "preserve"         -> clone the speaker's voice AND accent (ICL mode)
    #   "speaker-only"     -> keep only the speaker embedding, letting the model
    #                         use its own phonetics for the target language
    #   "reference-accent" -> keep the speaker's timbre but take pronunciation and
    #                         prosody from --accent-audio/--accent-text
    accent_mode: str = "preserve"
    #: Native target-language clip whose phonetics the dub should adopt.
    accent_audio: Path | None = None
    #: Exact transcript of ``accent_audio`` (required: it aligns the speech codes).
    accent_text: str | None = None
    #: How ``reference-accent`` builds the voice it clones. ``None`` means the
    #: default. See :attr:`accent_blend_mode`.
    accent_blend: str | None = None
    #: Takes rendered for the one-off blended reference (default 3). The blend
    #: fixes the voice of *every* line, so it is worth sampling more than once and
    #: keeping the take that kept the donor's timbre. Ignored unless
    #: ``accent_mode`` is ``reference-accent``.
    blend_takes: int | None = None
    #: Passes used to build the blended reference (default 4). Each pass re-encodes
    #: the previous blend and re-injects the speaker's embedding, walking the voice
    #: back toward the source speaker while the accent transcript keeps
    #: pronunciation anchored. ``1`` reproduces the single-pass behaviour.
    blend_iters: int | None = None
    #: How far toward the speaker's own voice the blend should reach, from ``0.0``
    #: (the accent clip's voice) to ``1.0`` (a French utterance in the speaker's own
    #: voice). Iteration stops as soon as it is met; ``0`` disables the search.
    blend_ratio: float | None = None

    # Speaker separation.
    #   "auto" -> cluster per-segment speaker embeddings, so each voice is cloned
    #             from its own donor (needs the optional funasr dependency)
    #   "none" -> treat the whole video as one speaker, as before
    diarizer: str = "auto"
    #: Cosine distance at which two voices count as different speakers. ``None``
    #: uses the calibrated default; raise it for similar voices, lower it to split
    #: a speaker the clustering merged.
    speaker_threshold: float | None = None
    #: Clusters shorter than this are folded into their nearest neighbour, because
    #: a voice needs a few seconds of speech before it can be cloned.
    min_speaker_seconds: float = Field(default=MIN_SPEAKER_S, ge=0.0, le=120.0)

    # Overlapping speech: a segment that carries two voices at once.
    #   "dominant" -> dub the whole line in its dominant speaker's voice (v1)
    #   "separate" -> split the line, re-transcribe each stream and dub both voices
    overlap_mode: str = "dominant"
    #: Backend used to split overlapping speech: auto | sepformer | none.
    overlap_separator: str = "auto"

    keep_subtitles: bool = True
    force: bool = False

    @field_validator("loudness_target")
    @classmethod
    def _check_loudness(cls, value: str | None) -> str | None:
        return None if value is None else normalize_loudness_policy(value)

    @field_validator("accent_mode")
    @classmethod
    def _check_accent(cls, value: str) -> str:
        key = str(value).strip().lower().replace("_", "-")
        aliases = {
            "preserve": "preserve",
            "icl": "preserve",
            "source": "preserve",
            "speaker-only": "speaker-only",
            "xvector": "speaker-only",
            "x-vector": "speaker-only",
            "speaker": "speaker-only",
            "reference-accent": "reference-accent",
            "ref-accent": "reference-accent",
            "hybrid": "reference-accent",
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(
                f"Unknown accent mode {value!r}. Use: preserve, speaker-only or "
                "reference-accent"
            ) from None

    @field_validator("accent_blend")
    @classmethod
    def _check_accent_blend(cls, value: str | None) -> str | None:
        if value is None:
            return None
        key = str(value).strip().lower().replace("_", "-")
        aliases = {
            "two-stage": "two-stage",
            "two": "two-stage",
            "blend": "two-stage",
            "single-stage": "single-stage",
            "single": "single-stage",
            "direct": "single-stage",
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(
                f"Unknown accent blend {value!r}. Use: two-stage or single-stage"
            ) from None

    @field_validator("blend_takes")
    @classmethod
    def _check_blend_takes(cls, value: int | None) -> int | None:
        if value is None:
            return None
        count = int(value)
        if count < 1:
            raise ConfigError("--blend-takes must be at least 1")
        return count

    @field_validator("blend_iters")
    @classmethod
    def _check_blend_iters(cls, value: int | None) -> int | None:
        if value is None:
            return None
        count = int(value)
        if count < 1:
            raise ConfigError("--blend-iters must be at least 1")
        return count

    @field_validator("blend_ratio")
    @classmethod
    def _check_blend_ratio(cls, value: float | None) -> float | None:
        if value is None:
            return None
        ratio = float(value)
        if not 0.0 <= ratio <= 1.0:
            raise ConfigError("--blend-ratio must be between 0 and 1")
        return ratio

    @field_validator("diarizer")
    @classmethod
    def _check_diarizer(cls, value: str) -> str:
        key = str(value).strip().lower().replace("_", "-")
        aliases = {
            "auto": "auto",
            "on": "auto",
            "true": "auto",
            "embedding": "auto",
            "none": "none",
            "off": "none",
            "false": "none",
            "single": "none",
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(
                f"Unknown diarizer {value!r}. Use: auto or none"
            ) from None

    @field_validator("speaker_threshold")
    @classmethod
    def _check_speaker_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return None
        distance = float(value)
        if not 0.0 < distance <= 2.0:
            raise ConfigError(
                "--speaker-threshold must be greater than 0 and at most 2"
            )
        return distance

    @field_validator("overlap_mode")
    @classmethod
    def _check_overlap_mode(cls, value: str) -> str:
        key = str(value).strip().lower().replace("_", "-")
        aliases = {
            "dominant": "dominant",
            "keep": "dominant",
            "off": "dominant",
            "separate": "separate",
            "split": "separate",
            "on": "separate",
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(
                f"Unknown overlap mode {value!r}. Use: dominant or separate"
            ) from None

    @field_validator("overlap_separator")
    @classmethod
    def _check_overlap_separator(cls, value: str) -> str:
        key = str(value).strip().lower().replace("_", "-")
        aliases = {
            "auto": "auto",
            "sepformer": "sepformer",
            "speechbrain": "sepformer",
            "none": "none",
            "off": "none",
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(
                f"Unknown overlap separator {value!r}. Use: auto, sepformer or none"
            ) from None

    @field_validator("length_budget")
    @classmethod
    def _check_length_budget(cls, value: str) -> str:
        key = str(value).strip().lower().replace("_", "-")
        aliases = {
            "off": "off",
            "none": "off",
            "seconds": "off",
            "words": "words",
            "word": "words",
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(
                f"Unknown length budget {value!r}. Use: off or words"
            ) from None

    @field_validator("room")
    @classmethod
    def _check_room(cls, value: str) -> str:
        key = str(value).strip().lower().replace("_", "-")
        aliases = {
            "off": "off",
            "none": "off",
            "dry": "off",
            "false": "off",
            "match": "match",
            "on": "match",
            "true": "match",
            "reverb": "match",
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(
                f"Unknown room mode {value!r}. Use: off or match"
            ) from None

    @field_validator("dtype")
    @classmethod
    def _check_dtype(cls, value: str) -> str:
        return normalize_dtype(value)

    @field_validator("target_language")
    @classmethod
    def _check_target(cls, value: str) -> str:
        return normalize_language(value)

    @field_validator("source_language")
    @classmethod
    def _check_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_language(value)
        if normalized == "Auto":
            raise ConfigError("source_language must be an explicit language, not 'auto'")
        return normalized

    @field_validator("video", "audio", "ref_audio", "accent_audio", "output_dir")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @model_validator(mode="after")
    def _validate(self) -> "JobConfig":
        if (self.video is None) == (self.audio is None):
            raise ConfigError("Provide exactly one input: either --video or --audio")

        if self.source_language is not None and self.source_language == self.target_language:
            raise ConfigError(
                f"source_language and target_language are both {self.target_language!r}"
            )

        if self.ref_audio is not None:
            # ICL cloning conditions on the reference transcript, so it is required
            # there. The other modes ignore it: speaker-only drops the codes
            # entirely, and reference-accent takes them from --accent-audio.
            needs_transcript = self.accent_mode == "preserve"
            if needs_transcript and (self.ref_text is None or not self.ref_text.strip()):
                raise ConfigError(
                    "--ref-audio requires --ref-text (the exact transcript of the "
                    "reference clip). Omit both to auto-select a donor from the video."
                )

        if self.accent_mode == "reference-accent":
            if self.accent_audio is None:
                raise ConfigError(
                    "--accent-mode reference-accent requires --accent-audio: a clip "
                    "of a native target-language speaker, used for pronunciation only."
                )
            if self.accent_text is None or not self.accent_text.strip():
                raise ConfigError(
                    "--accent-audio requires --accent-text (its exact transcript), "
                    "which aligns the accent clip's speech codes."
                )
        elif self.accent_audio is not None:
            raise ConfigError(
                "--accent-audio only applies to --accent-mode reference-accent; add "
                "that flag or drop --accent-audio."
            )

        # The blend knobs only mean anything in reference-accent mode, and a
        # silently ignored knob is worse than a rejected one. "Asked for" is
        # decided by the *value*, not by ``model_fields_set``: the CLI passes every
        # flag explicitly, so a default of ``None`` would otherwise look like a
        # request and reject the plain `dub.py --video in.mp4 --target fr`.
        if self.accent_mode != "reference-accent":
            for field_name, flag in (
                ("accent_blend", "--accent-blend"),
                ("blend_takes", "--blend-takes"),
                ("blend_iters", "--blend-iters"),
                ("blend_ratio", "--blend-ratio"),
            ):
                if getattr(self, field_name) is not None:
                    raise ConfigError(
                        f"{flag} only applies to --accent-mode reference-accent; add "
                        f"that flag or drop {flag}."
                    )
        return self

    @property
    def accent_blend_mode(self) -> str:
        """Resolved blend strategy: ``two-stage`` unless the user opted out.

        ``two-stage`` renders the accent transcript once in the donor's voice and
        clones every line from that clip, so the codes and the speaker embedding
        always describe the same speaker. ``single-stage`` conditions each line on
        the accent codes and the donor embedding simultaneously, which leaves a
        line free to follow either signal.
        """
        return self.accent_blend or "two-stage"

    @property
    def resolved_blend_takes(self) -> int:
        """Takes rendered for the one-off blend (default 3)."""
        return self.blend_takes if self.blend_takes is not None else 3

    @property
    def resolved_blend_iters(self) -> int:
        """Passes used to build the blend before giving up on the target (default 4)."""
        return self.blend_iters if self.blend_iters is not None else 4

    @property
    def resolved_blend_ratio(self) -> float:
        """Donor ratio that ends iteration early (default 0.5, ``0`` disables)."""
        return self.blend_ratio if self.blend_ratio is not None else 0.5

    @property
    def resolved_speaker_threshold(self) -> float:
        """Cosine distance separating two voices (the calibrated default)."""
        return (
            self.speaker_threshold
            if self.speaker_threshold is not None
            else DEFAULT_THRESHOLD
        )

    @property
    def overlap_separation(self) -> bool:
        """True when overlapping speech should be split into two voices."""
        return self.overlap_mode == "separate"

    @property
    def multi_speaker(self) -> bool:
        """True when the job asked for speaker separation."""
        return self.diarizer != "none"

    @property
    def room_matching(self) -> bool:
        """True when the dub should be placed in the source's acoustic space."""
        return self.room == "match"

    @property
    def input_path(self) -> Path:
        """The single input path (video or audio)."""
        path = self.video if self.video is not None else self.audio
        assert path is not None  # guaranteed by the model validator
        return path

    @property
    def is_video(self) -> bool:
        """True when the input is a video file."""
        return self.video is not None

    def fingerprint(self) -> str:
        """Stable hash of the settings that affect pipeline output.

        Used for cache invalidation: changing the target language, whisper
        model, donor, or duration bounds must re-run the affected stages.
        """
        import hashlib
        import json

        payload = self.model_dump(mode="json")
        for volatile in ("output_dir", "force"):
            payload.pop(volatile, None)
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #

_LOGGER = logging.getLogger(__name__)

#: Accelerator families this pipeline can talk to.
ACCELERATORS: tuple[str, ...] = ("cuda", "rocm", "xpu", "mps")

#: Where bfloat16 starts being supported natively. Below these, float16 is the
#: right choice and float32 the safe fallback: bfloat16 on a Pascal or Volta card
#: is unsupported or emulated at a fraction of the speed, which is exactly the
#: "works on my GPU, not on yours" failure this table exists to prevent.
_BF16_MIN_CAPABILITY: dict[str, tuple[int, int]] = {"cuda": (8, 0), "rocm": (9, 4)}
_FP16_MIN_CAPABILITY: dict[str, tuple[int, int]] = {"cuda": (7, 0), "rocm": (7, 0)}

#: Accepted spellings of a dtype, for ``--dtype`` and for ``JobConfig.dtype``.
DTYPE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "full": "fp32",
}

#: Canonical dtype name -> ``torch`` attribute.
_DTYPE_ATTRIBUTES: dict[str, str] = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


@dataclass(frozen=True)
class DeviceInfo:
    """What a device request resolves to, and what that hardware supports."""

    kind: str
    index: int | None = None
    name: str = ""
    total_memory: int | None = None
    capability: tuple[int, int] | None = None
    requested: str = "auto"

    @property
    def is_accelerated(self) -> bool:
        return self.kind in ACCELERATORS

    @property
    def label(self) -> str:
        """The string to hand to a backend, e.g. ``cuda``, ``cuda:1``, ``cpu``.

        ROCm is reported as ``cuda`` because AMD cards answer to the CUDA API;
        ``torch.device("rocm")`` does not exist.
        """
        if self.kind == "rocm":
            return "cuda" if self.index is None else f"cuda:{self.index}"
        if self.kind in {"cuda", "xpu"}:
            return self.kind if self.index is None else f"{self.kind}:{self.index}"
        return self.kind

    @property
    def supports_bf16(self) -> bool:
        return supports_bf16(self)

    @property
    def supports_fp16(self) -> bool:
        return supports_fp16(self)


def _import_torch() -> Any | None:
    """Import torch, or ``None`` when it is missing.

    Isolated so tests can substitute a fake module and exercise every hardware
    profile - Pascal, Ampere, ROCm, Metal, XPU - without owning any of it.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard runtime dep
        return None
    return torch


def _auto_kind(torch: Any | None) -> str:
    """The best device *every* stage can use: CUDA/ROCm, otherwise CPU.

    Metal and XPU are deliberately not chosen here. Speech recognition runs on
    CTranslate2, which offers only CUDA and CPU, so an ``auto`` that picked Metal
    would break a Mac that works today. ``--device mps`` is still honoured when
    asked for, and :func:`available_accelerators` reports what is present.
    """
    if torch is not None and torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return "cpu"


def _split_requested(requested: str, torch: Any | None) -> tuple[str, int | None]:
    """Split ``cuda:1`` into ``("cuda", 1)``; unknown names pass through."""
    text = (requested or "auto").strip().lower()
    base, _, suffix = text.partition(":")
    index: int | None = None
    if suffix:
        if not suffix.isdigit():
            return text, None
        index = int(suffix)
    if base == "auto":
        return _auto_kind(torch), None
    if base == "cuda" and torch is not None and getattr(torch.version, "hip", None):
        return "rocm", index
    return base, index


def _describe(
    torch: Any | None, kind: str, index: int | None, requested: str
) -> DeviceInfo:
    """Fill in the name, memory and compute capability for ``kind``."""
    if torch is None:
        return DeviceInfo(kind="cpu", requested=requested)

    probe_index = 0 if index is None else index
    name = ""
    memory: int | None = None
    capability: tuple[int, int] | None = None

    if kind in {"cuda", "rocm"}:
        try:
            name = str(torch.cuda.get_device_name(probe_index))
            memory = int(torch.cuda.get_device_properties(probe_index).total_memory)
            raw = torch.cuda.get_device_capability(probe_index)
            capability = tuple(int(part) for part in raw) if raw else None
        except Exception:  # a probe must never fail the run it describes
            _LOGGER.debug("Could not describe %s:%s", kind, probe_index, exc_info=True)
    elif kind == "xpu":
        try:
            name = str(torch.xpu.get_device_name(probe_index))
            properties = torch.xpu.get_device_properties(probe_index)
            memory = int(getattr(properties, "total_memory", 0)) or None
        except Exception:
            _LOGGER.debug("Could not describe xpu:%s", probe_index, exc_info=True)
    elif kind == "mps":
        name = "Apple Silicon (Metal)"
    elif kind == "cpu":
        import platform

        name = platform.processor() or platform.machine()

    return DeviceInfo(
        kind=kind,
        index=index,
        name=name,
        total_memory=memory,
        capability=capability,
        requested=requested,
    )


def probe_device(requested: str = "auto") -> DeviceInfo:
    """Resolve ``requested`` and describe the hardware behind it.

    Never raises: an unparseable request or an absent device produces a report
    that says so, because this is used for diagnostics as well as dispatch.
    """
    torch = _import_torch()
    if torch is None:
        return DeviceInfo(kind="cpu", requested=requested)
    kind, index = _split_requested(requested, torch)
    return _describe(torch, kind, index, requested)


def available_accelerators() -> tuple[str, ...]:
    """Every accelerator torch can use right now, most capable first."""
    torch = _import_torch()
    if torch is None:
        return ()
    found: list[str] = []
    if torch.cuda.is_available():
        found.append("rocm" if getattr(torch.version, "hip", None) else "cuda")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        found.append("xpu")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        found.append("mps")
    return tuple(found)


def supports_bf16(info: DeviceInfo) -> bool:
    """Whether the hardware does bfloat16 natively rather than emulated."""
    if info.kind == "xpu":
        return True
    if info.kind == "mps":
        # Supported on recent macOS only; float16 is the safe choice everywhere.
        return False
    minimum = _BF16_MIN_CAPABILITY.get(info.kind)
    if minimum is None or info.capability is None:
        return False
    return info.capability >= minimum


def supports_fp16(info: DeviceInfo) -> bool:
    """Whether float16 is both supported and worth using on this device."""
    if info.kind in {"xpu", "mps"}:
        return True
    minimum = _FP16_MIN_CAPABILITY.get(info.kind)
    if minimum is None or info.capability is None:
        return False
    return info.capability >= minimum


def preferred_dtype(info: DeviceInfo) -> str:
    """The canonical dtype name to use on ``info`` when the user has no opinion."""
    if supports_bf16(info):
        return "bf16"
    if supports_fp16(info):
        return "fp16"
    return "fp32"


def normalize_dtype(value: str | None) -> str:
    """Canonicalise a dtype spelling, raising ``ConfigError`` for nonsense."""
    key = str(value or "auto").strip().lower()
    try:
        return DTYPE_ALIASES[key]
    except KeyError:
        raise ConfigError(
            f"Unknown dtype {value!r}. Use: auto, bf16, fp16 or fp32"
        ) from None


def _dtype_usable(info: DeviceInfo, name: str) -> bool:
    """Whether an *explicitly requested* dtype can run on ``info``."""
    if not info.is_accelerated:
        return False  # CPU half-precision is slow and patchy; no reason to allow it
    if name == "bf16":
        return supports_bf16(info)
    return True  # float16 is available on every accelerator family


def pick_dtype(device: str = "auto", requested: str = "auto") -> Any:
    """The torch dtype to load weights with on ``device``.

    Capability-aware by default. An explicit ``requested`` dtype is honoured
    unless the device cannot run it (bfloat16 on a pre-Ampere card, or any
    half-precision on CPU), in which case it degrades to the best the device does
    support and says so in the log rather than failing the job.
    """
    torch = _import_torch()
    if torch is None:  # pragma: no cover - torch is a hard runtime dep
        raise ConfigError("torch is not installed, so no dtype can be selected")

    want = normalize_dtype(requested)
    info = probe_device(device)
    if want == "auto":
        want = preferred_dtype(info)
    elif want != "fp32" and not _dtype_usable(info, want):
        fallback = preferred_dtype(info)
        _LOGGER.warning(
            "dtype %s is not usable on %s (%s); using %s instead.",
            want,
            info.label,
            info.kind,
            fallback,
        )
        want = fallback
    return getattr(torch, _DTYPE_ATTRIBUTES[want])


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string for the model backends.

    ``auto`` picks a device *every* stage supports - CUDA (including ROCm) when
    one is present, otherwise CPU. Metal and XPU are reported by
    :func:`probe_device` and usable through an explicit ``--device``, but are not
    chosen automatically because CTranslate2, which speech recognition runs on,
    offers only CUDA and CPU.

    An explicit request is returned unchanged, so ``cuda:1`` stays ``cuda:1``
    instead of being silently narrowed to the default device.
    """
    if requested and requested != "auto":
        return requested
    return probe_device("auto").label


def release_gpu_memory() -> bool:
    """Return cached accelerator memory to the driver; ``True`` if anything was.

    Each heavy stage loads its own multi-GB model (Whisper ~3 GB, Qwen3-TTS
    ~3.5 GB). When a stage's model goes out of scope, PyTorch keeps the blocks in
    its caching allocator, so the next stage can hit an out-of-memory error even
    though the previous model is already gone. Calling this between stages makes
    that memory available again.

    Cheap and safe to call anywhere: a no-op when no accelerator is present.
    """
    torch = _import_torch()
    if torch is None:  # pragma: no cover - torch is a hard runtime dep
        return False

    import gc

    gc.collect()
    released = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        released = True

    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        empty = getattr(xpu, "empty_cache", None)
        if callable(empty):
            empty()
            released = True

    mps = getattr(torch, "mps", None)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps_backend is not None and mps_backend.is_available():
        empty = getattr(mps, "empty_cache", None)
        if callable(empty):
            empty()
            released = True

    return released
