"""Model registry, download and local path resolution.

Models are intentionally *not* committed to the repository (``models/`` is
gitignored). Every component resolves its weights through :func:`resolve_model`
so no path is hard-coded in two places, and a missing model produces an
actionable message instead of an obscure loader crash.

Download everything the MVP needs with::

    python dub.py models --download all
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelNotFoundError

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "ModelSpec",
    "REGISTRY",
    "MVP_REQUIRED",
    "anchors_dir",
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
    #: How the weights are obtained: "huggingface" or "demucs" (torch hub, and
    #: fetched lazily by the package, so it is only pre-warmed here).
    source: str = "huggingface"
    #: Files to fetch from the repo. ``None`` means everything. Pinning the
    #: handful of files a checkpoint actually loads keeps a download to the
    #: model rather than every alternate weight, RL variant and export the repo
    #: happens to carry (CosyVoice3: ~5.4 GB of 9.75 GB; Chatterbox: ~3.2 GB of
    #: 13.9 GB).
    allow_patterns: tuple[str, ...] | None = None


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
    "tts_cosyvoice": ModelSpec(
        name="tts_cosyvoice",
        repo_id="FunAudioLLM/CosyVoice2-0.5B",
        dir_name="CosyVoice2-0.5B",
        description="CosyVoice2: cloned voice + per-utterance emotion instruction",
        license="Apache-2.0",
    ),
    "tts_cosyvoice3": ModelSpec(
        name="tts_cosyvoice3",
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        dir_name="Fun-CosyVoice3-0.5B",
        description=(
            "CosyVoice3: cloned voice + instruct (emotion/speed/volume) and "
            "paralinguistic tokens such as [breath] and [laughter]"
        ),
        license="Apache-2.0",
        # The checkpoint loads llm.pt/flow.pt/hift.pt, the FP32 speech tokenizer,
        # campplus, and the bundled Qwen tokenizer. The RL weights, the batch
        # tokenizer and the TensorRT ONNX export are alternative paths this
        # pipeline does not use.
        allow_patterns=(
            "config.json",
            "configuration.json",
            "cosyvoice3.yaml",
            "llm.pt",
            "flow.pt",
            "hift.pt",
            "campplus.onnx",
            "speech_tokenizer_v3.onnx",
            "CosyVoice-BlankEN/*",
        ),
    ),
    "tts_chatterbox": ModelSpec(
        name="tts_chatterbox",
        repo_id="ResembleAI/chatterbox",
        dir_name="chatterbox",
        description=(
            "Chatterbox Multilingual: cloned voice + emotion exaggeration "
            "(MIT weights; the chatterbox-tts package loads them)"
        ),
        license="MIT",
        # Exactly the files ChatterboxMultilingualTTS.from_local loads; the repo
        # also carries the English-only and older multilingual variants.
        allow_patterns=(
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ),
    ),
    "tts_fish": ModelSpec(
        name="tts_fish",
        repo_id="fishaudio/s2-pro",
        dir_name="fish-speech-s2-pro",
        description=(
            "Fish Speech S2-Pro: cloned voice + inline emotion/paralinguistic "
            "tags (needs ~24 GB VRAM)"
        ),
        license="Fish Audio Research License (non-commercial)",
    ),
    "tts_index_tts2": ModelSpec(
        name="tts_index_tts2",
        repo_id="IndexTeam/IndexTTS-2",
        dir_name="IndexTTS2",
        description=(
            "IndexTTS2: cloned voice + a separate emotion reference; output is "
            "time-fitted by the pipeline's stretcher"
        ),
        license="Bilibili IndexTTS License (non-commercial)",
    ),
    "tts_tokenizer": ModelSpec(
        name="tts_tokenizer",
        repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        dir_name="Qwen3-TTS-Tokenizer-12Hz",
        description="Standalone speech tokenizer (optional; bundled with tts_base)",
        license="Apache-2.0",
    ),
    "demucs": ModelSpec(
        name="demucs",
        repo_id="htdemucs (torch hub)",
        dir_name="demucs",
        description="Speech/background separation for keeping music and ambience",
        license="MIT",
        source="demucs",
    ),
}

#: Models the MVP cannot run without. The default ``native`` accent mode needs
#: the VoiceDesign checkpoint to generate its per-language accent anchor, so it
#: is required rather than optional.
MVP_REQUIRED: tuple[str, ...] = (
    "whisper_large_v3",
    "tts_base",
    "tts_voice_design",
)

#: Subdirectory of the models directory holding generated accent anchors.
ANCHORS_DIR_NAME = "anchors"


def anchors_dir(models_dir: Path | str = "models") -> Path:
    """Where generated accent anchors are cached.

    Kept beside the weights because it is derived data from a model and is
    similarly excluded from version control.
    """
    return Path(models_dir) / ANCHORS_DIR_NAME


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
    """True when the model can be used without downloading anything now.

    A manually populated directory (e.g. copied over from another machine) counts
    as available, so the downloader is a convenience rather than a requirement.
    Package-backed models (Demucs) report availability when the package is
    importable; their weights are fetched by the package on first use, or
    pre-warmed by :func:`download_model`.
    """
    spec = _spec(name)

    if spec.source == "demucs":
        import importlib.util

        return importlib.util.find_spec("demucs") is not None

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
        remedy = (
            "pip install demucs" if spec.source == "demucs" else download_hint(name)
        )
        raise ModelNotFoundError(
            f"Model {name!r} ({spec.repo_id}) not found at {root}",
            remedy=remedy,
        )
    _LOGGER.debug("Resolved model %s -> %s", name, root)
    return root


def _directory_size(root: Path) -> int:
    """Bytes under ``root``; ``0`` when it cannot be measured."""
    try:
        return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    except OSError:
        return 0


def download_model(
    name: str,
    models_dir: Path | str = "models",
    *,
    force: bool = False,
) -> Path:
    """Download ``name`` into ``models_dir`` and return the directory.

    Hugging Face models are fetched with ``huggingface_hub.snapshot_download``
    (already a dependency of transformers, so no extra package is required).
    Package-backed models such as Demucs are pre-warmed instead: their weights
    live in the torch hub cache and are downloaded by the package itself.

    Raises:
        DependencyError: If a package-backed model's package is not installed.
    """
    from huggingface_hub import snapshot_download

    spec = _spec(name)
    root = Path(models_dir) / spec.dir_name

    if spec.source == "demucs":
        # Fetch the weights into the torch hub cache so the first job does not
        # pay for the download.
        from demucs.pretrained import get_model

        _LOGGER.info("Pre-warming the Demucs weights")
        started = time.perf_counter()
        get_model("htdemucs")
        root.mkdir(parents=True, exist_ok=True)
        _LOGGER.info(
            "Demucs weights ready in %.1fs", time.perf_counter() - started
        )
        return root

    if is_available(name, models_dir) and not force:
        _LOGGER.info("Model %s is already present at %s", name, root)
        return root

    _LOGGER.info(
        "Downloading %s from %s%s",
        name,
        spec.repo_id,
        (
            f" ({len(spec.allow_patterns)} file pattern(s))"
            if spec.allow_patterns
            else ""
        ),
    )
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    # Downloads resume automatically when a partial file is present.
    # (`local_dir_use_symlinks` / `resume_download` were deprecated in
    # huggingface_hub 0.3x and removed later, so they are deliberately omitted.)
    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(root),
        allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
    )
    elapsed = time.perf_counter() - started
    _LOGGER.info(
        "Downloaded %s to %s in %.1fs (%.2f GB)",
        name,
        root,
        elapsed,
        _directory_size(root) / 1e9,
    )
    return root


def download_many(
    names: list[str] | tuple[str, ...],
    models_dir: Path | str = "models",
    *,
    force: bool = False,
    progress: object | None = None,
) -> dict[str, Path]:
    """Download several models, returning ``{name: directory}``."""
    if names:
        _LOGGER.info(
            "Downloading %d model(s): %s", len(names), ", ".join(names)
        )
    resolved: dict[str, Path] = {}
    for name in names:
        if progress is not None and callable(progress):  # pragma: no cover - UI hook
            progress(name)
        resolved[name] = download_model(name, models_dir, force=force)
    return resolved


def registry_status(models_dir: Path | str = "models") -> list[tuple[ModelSpec, bool]]:
    """Report availability for every known model, for ``models --list``."""
    return [(spec, is_available(spec.name, models_dir)) for spec in REGISTRY.values()]
