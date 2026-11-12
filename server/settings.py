"""Effective settings for the web app.

Secrets stay in the environment (or ``.env``) and are never written to SQLite.
Stored values are user preferences and only fill fields that the environment has
not already set, so a deployment can still override the UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pipeline.config import Settings, read_env_file

from .store import Store

__all__ = [
    "EDITABLE_FIELDS",
    "effective_settings",
    "settings_payload",
]

#: Fields the UI may persist. Secrets are deliberately absent.
EDITABLE_FIELDS: tuple[str, ...] = (
    "llm_base_url",
    "llm_model",
    "llm_timeout_s",
    "llm_chunk_size",
    "llm_plan_chunk_size",
    "loudness_target",
    "target_lufs",
    "keep_background",
    "separator",
    "duck_db",
    "models_dir",
    "voices_dir",
    "log_level",
    "log_json",
)

_FIELD_ENV = {
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "llm_timeout_s": "LLM_TIMEOUT_S",
    "llm_chunk_size": "LLM_CHUNK_SIZE",
    "llm_plan_chunk_size": "LLM_PLAN_CHUNK_SIZE",
    "loudness_target": "LOUDNESS_TARGET",
    "target_lufs": "LOUDNESS_LUFS",
    "keep_background": "KEEP_BACKGROUND",
    "separator": "BACKGROUND_SEPARATOR",
    "duck_db": "DUCK_DB",
    "models_dir": "MODELS_DIR",
    "voices_dir": "VOICES_DIR",
    "log_level": "LOG_LEVEL",
    "log_json": "LOG_JSON",
}


def effective_settings(store: Store) -> Settings:
    """Environment first, then ``.env``, then stored UI preferences."""
    base = Settings.from_env()
    stored = store.get_settings()
    env_keys = set(os.environ)
    file_keys = set(read_env_file(Path(".env")))
    payload = base.model_dump()
    for field, value in stored.items():
        if field not in Settings.model_fields or field not in EDITABLE_FIELDS:
            continue
        env_name = _FIELD_ENV.get(field, field.upper())
        if env_name in env_keys or env_name in file_keys:
            continue
        payload[field] = value
    return Settings.model_validate(payload)


def settings_payload(settings: Settings) -> dict[str, Any]:
    """Settings safe to return to the browser (no secret values)."""
    payload = settings.model_dump(mode="json")
    api_key = payload.pop("llm_api_key", None)
    payload["llm_api_key_set"] = bool(api_key and api_key != "not-needed")
    payload["editable"] = list(EDITABLE_FIELDS)
    return payload
