"""Request/response models for the local HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import JobConfig

_BACKENDS = Literal["qwen", "customvoice", "cosyvoice3", "chatterbox", "fish"]


class DubRequest(BaseModel):
    """A dubbing job plus the saved-voice labels the CLI resolves earlier."""

    model_config = ConfigDict(extra="forbid")

    config: JobConfig
    voice: str | None = None
    accent_voice: str | None = None
    stretcher: str = "auto"


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


class JobSubmitResponse(BaseModel):
    job_id: str
    state: str
