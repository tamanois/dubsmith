"""Model registry, download and local path resolution.

Models are intentionally *not* committed to the repository (``models/`` is
gitignored). Every component resolves its weights through :func:`resolve_model`
so no path is hard-coded in two places, and a missing model produces an
actionable message instead of an obscure loader crash.

Download everything the MVP needs with::

    python dub.py models --download all
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import ModelNotFoundError

__all__ = [
    "ModelSpec",
    "REGISTRY",
    "MVP_REQUIRED",
    "download_model",
    "download_many",
    "resolve_model",
    "is_available",
    "registry_status",
    "download_hint",
]


@dataclass(frozen=True)
class ModelSpec:
    """A model this pipeline can use."""

    name: str
    repo_id: str
    dir_name: str
    description: str
    license: str


REGISTRY: dict[str, ModelSpec] = {
    "whisper_large_v3": ModelSpec(
        name="whisper_large_v3",
        repo_id="Systran/faster-whisper-large-v3",
        dir_name="faster-whisper-large-v3",
        description="Speech recognition (CTranslate2 Whisper large-v3)",
        license="MIT",
    ),
    "tts_base": ModelSpec(
        name="tts_base",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dir_name="Qwen3-TTS-12Hz-1.7B-Base",
        description="Voice cloning base model (bundles its speech tokenizer)",
        license="Apache-2.0",
    ),
    # Optional extras.
    "tts_custom_voice": ModelSpec(
        name="tts_custom_voice",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dir_name="Qwen3-TTS-12Hz-1.7B-CustomVoice",
        description="Named-speaker voices with instruction control",
        license="Apache-2.0",
    ),
    "tts_voice_design": ModelSpec(
        name="tts_voice_design",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        dir_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        description="Voice design from text descriptions (P2 accent anchor)",
        license="Apache-2.0",
    ),
    "tts_tokenizer": ModelSpec(
        name="tts_tokenizer",
        repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        dir_name="Qwen3-TTS-Tokenizer-12Hz",
        description="Standalone speech tokenizer (optional; bundled with tts_base)",
        license="Apache-2.0",
    ),
}

#: Models the MVP cannot run without.
MVP_REQUIRED: tuple[str, ...] = ("whisper_large_v3", "tts_base")


def download_hint(name: str = "all") -> str:
    """Shell command that fetches ``name`` (or everything)."""
    return f"python dub.py models --download {name}"


def _spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(REGISTRY)
        raise ModelNotFoundError(
            f"Unknown model {name!r}. Known models: {known}"
        ) from None


def is_available(name: str, models_dir: Path | str = "models") -> bool:
    """True when the model directory exists and holds at least one file.

    A manually populated directory (e.g. copied over from another machine) counts
    as available, so the downloader is a convenience rather than a requirement.
    """
    spec = _spec(name)
    root = Path(models_dir) / spec.dir_name
    if not root.is_dir():
        return False
    return any(child.is_file() for child in root.iterdir())


def resolve_model(name: str, models_dir: Path | str = "models") -> Path:
    """Return the local directory for ``name``.

    Raises:
        ModelNotFoundError: With the exact download command as a remedy.
    """
    spec = _spec(name)
    root = Path(models_dir) / spec.dir_name
    if not is_available(name, models_dir):
        raise ModelNotFoundError(
            f"Model {name!r} ({spec.repo_id}) not found at {root}",
            remedy=download_hint(name),
        )
    return root


def download_model(
    name: str,
    models_dir: Path | str = "models",
    *,
    force: bool = False,
) -> Path:
    """Download ``name`` into ``models_dir`` and return the directory.

    Uses ``huggingface_hub.snapshot_download`` (already a dependency of
    transformers, so no extra package is required).
    """
    from huggingface_hub import snapshot_download

    spec = _spec(name)
    root = Path(models_dir) / spec.dir_name
    if is_available(name, models_dir) and not force:
        return root

    root.mkdir(parents=True, exist_ok=True)
    # Downloads resume automatically when a partial file is present.
    # (`local_dir_use_symlinks` / `resume_download` were deprecated in
    # huggingface_hub 0.3x and removed later, so they are deliberately omitted.)
    snapshot_download(repo_id=spec.repo_id, local_dir=str(root))
    return root


def download_many(
    names: list[str] | tuple[str, ...],
    models_dir: Path | str = "models",
    *,
    force: bool = False,
    progress: object | None = None,
) -> dict[str, Path]:
    """Download several models, returning ``{name: directory}``."""
    resolved: dict[str, Path] = {}
    for name in names:
        if progress is not None and callable(progress):  # pragma: no cover - UI hook
            progress(name)
        resolved[name] = download_model(name, models_dir, force=force)
    return resolved


def registry_status(models_dir: Path | str = "models") -> list[tuple[ModelSpec, bool]]:
    """Report availability for every known model, for ``models --list``."""
    return [(spec, is_available(spec.name, models_dir)) for spec in REGISTRY.values()]
