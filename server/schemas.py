"""Request/response models for the local HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline.config import JobConfig
from pipeline.errors import ConfigError

_BACKENDS = Literal["qwen", "customvoice", "cosyvoice3", "chatterbox", "fish"]


class DubRequest(BaseModel):
    """A dubbing job plus the saved-voice labels the CLI resolves earlier."""

    model_config = ConfigDict(extra="forbid")

    config: JobConfig
    voice: str | None = None
    accent_voice: str | None = None
    stretcher: str = "auto"

    @model_validator(mode="before")
    @classmethod
    def _accent_voice_supplies_the_clip(cls, data: Any) -> Any:
        """Allow ``accent_voice`` to stand in for the accent clip it provides.

        ``JobConfig`` requires ``accent_audio`` for ``reference-accent``, but a
        saved voice *is* that clip plus its transcript. Validate such a request as
        if no mode were chosen yet; the route then re-derives the clip, the
        transcript and the mode from the voice itself
        (:func:`pipeline.job_ops.apply_accent_voice`), exactly as
        ``--accent-voice`` does on the command line. Naming another mode is
        rejected rather than silently overridden.
        """
        if not isinstance(data, dict) or not data.get("accent_voice"):
            return data
        config = data.get("config")
        if not isinstance(config, dict):
            return data
        mode = config.get("accent_mode")
        if mode is None:
            return data
        if mode != "reference-accent":
            raise ConfigError(
                "--accent-voice implies --accent-mode reference-accent; drop "
                f"--accent-mode {mode!r} or the saved accent voice."
            )
        if config.get("accent_audio") or (config.get("accent_text") or "").strip():
            return data
        return {**data, "config": {**config, "accent_mode": "native"}}


class SayRequest(BaseModel):
    """All standalone speech options, mirroring ``dub.py say``."""

    model_config = ConfigDict(extra="forbid")

    text: str
    backend: _BACKENDS = "qwen"
    voice: str | None = None
    design: str | None = None
    clone: str | None = None
    clone_text: str | None = None
    speaker: str | None = None
    language: str | None = None
    save_as: str | None = None
    output: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    force: bool = False
    provenance: bool = True
    prosody: list[str] | None = None
    emotions: list[str] | None = None
    prosody_plan: Literal["off", "llm"] = "off"
    style_instructs: list[str] | None = None
    style_plan: Literal["off", "llm"] = "off"
    style_plan_scope: Literal["per-line", "whole"] = "per-line"
    plan_chunk: int | None = Field(default=None, ge=1)
    chatterbox_exaggeration: float | None = Field(default=None, ge=0.0, le=1.0)
    chatterbox_cfg_weight: float | None = None
    chatterbox_temperature: float | None = None
    fish_temperature: float | None = None
    fish_top_p: float | None = None
    fish_quantize: Literal["int8", "off"] | None = None
    fish_tag_mode: Literal["keep", "llm"] = "keep"


class VoiceDesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    language: str = "fr"
    label: str | None = None
    speak: str | None = None
    output: str | None = None
    force: bool = False
    device: str = "auto"
    dtype: str = "auto"


class VoiceCloneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: str
    clone_text: str | None = None
    transcribe: bool = False
    verify: bool = False
    language: str | None = None
    label: str | None = None
    speak: str | None = None
    output: str | None = None
    force: bool = False
    device: str = "auto"
    dtype: str = "auto"


class VoiceAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: str
    clone_text: str | None = None
    transcribe: bool = False
    verify: bool = False
    language: str | None = None
    label: str | None = None
    force: bool = False
    device: str = "auto"


class ModelDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(min_length=1)
    force: bool = False


class AnchorGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    languages: list[str] = Field(default_factory=list)
    device: str = "auto"
    dtype: str = "auto"
    force: bool = False


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, Any]


class SettingsImport(BaseModel):
    """A settings export file, re-applied by the import endpoint."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    settings: dict[str, Any]


class IdListRequest(BaseModel):
    """A batch of identifiers, for bulk delete and clearing."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1)


class PathListRequest(BaseModel):
    """A batch of output paths, for bulk delete."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(min_length=1)


class PresetRequest(BaseModel):
    """A saved form configuration (a template for Dub or Say)."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any]


class FavoriteRequest(BaseModel):
    """An input path the user wants to keep within reach."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    label: str | None = None
    kind: str = "input"


class JobSubmitResponse(BaseModel):
    job_id: str
    state: str
