"""Job configuration, environment settings and language normalisation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .diarize import DEFAULT_THRESHOLD, MIN_SPEAKER_S
from .errors import ConfigError

__all__ = [
    "LANGUAGE_ALIASES",
    "SUPPORTED_LANGUAGES",
    "normalize_language",
    "normalize_loudness_policy",
    "JobConfig",
    "Settings",
    "read_env_file",
    "resolve_device",
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
        # silently ignored knob is worse than a rejected one. `model_fields_set`
        # distinguishes "left at its default" from "the user asked for this".
        if self.accent_mode != "reference-accent":
            for field_name, flag in (
                ("accent_blend", "--accent-blend"),
                ("blend_takes", "--blend-takes"),
                ("blend_iters", "--blend-iters"),
                ("blend_ratio", "--blend-ratio"),
            ):
                if field_name in self.model_fields_set:
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
    def multi_speaker(self) -> bool:
        """True when the job asked for speaker separation."""
        return self.diarizer != "none"

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


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to the best available torch device.

    Kept dependency-light by importing torch lazily; returns ``"cpu"`` when torch
    is unavailable or reports no CUDA device.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard runtime dep
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def release_gpu_memory() -> bool:
    """Return cached GPU memory to the driver; ``True`` if anything was released.

    Each heavy stage loads its own multi-GB model (Whisper ~3 GB, Qwen3-TTS
    ~3.5 GB). When a stage's model goes out of scope, PyTorch keeps the blocks in
    its caching allocator, so the next stage can hit an out-of-memory error even
    though the previous model is already gone. Calling this between stages makes
    that memory available again.

    Cheap and safe to call anywhere: it is a no-op without CUDA.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard runtime dep
        return False
    if not torch.cuda.is_available():
        return False

    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return True
