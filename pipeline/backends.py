"""Build the engine for one expressive synthesis backend.

The ``say``/``clone`` command and the dubbing pipeline both need to turn a
backend name plus a set of tuning knobs into a ready engine. Keeping that in one
place means the two call sites cannot drift apart about which checkpoint a
backend maps to or which knob it receives.

"Expressive" is the set of backends that render a :class:`~pipeline.prosody.ProsodySpec`
(or inline tags) rather than the plain Qwen Base clone. They all clone a speaker
from a single reference clip, which is what makes them usable for dubbing too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .capabilities import EXPRESSIVE_BACKENDS
from .errors import ConfigError, ModelNotFoundError
from .models import resolve_model

__all__ = ["EXPRESSIVE_BACKENDS", "build_expressive_engine"]


def build_expressive_engine(
    backend: str,
    models_dir: Path | str,
    device: str,
    *,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    temperature: float | None = None,
    fish_temperature: float | None = None,
    fish_top_p: float | None = None,
    fish_quantize: str | None = None,
    resolve: Callable[[str, Path | str], Path] | None = None,
) -> Any:
    """Construct the engine for ``backend``.

    Args:
        backend: One of :data:`EXPRESSIVE_BACKENDS`.
        models_dir: Root the local checkpoints are resolved under.
        device: Torch device string; each engine resolves it to what it can run.
        exaggeration: Chatterbox's fixed style scalar, if the caller set one.
        cfg_weight: Chatterbox classifier-free guidance weight.
        temperature: Chatterbox sampling temperature.
        fish_temperature: Fish sampling temperature.
        fish_top_p: Fish nucleus sampling threshold.
        fish_quantize: Fish quantization mode (e.g. ``int8``).
        resolve: Model resolver to use; defaults to :func:`pipeline.models.resolve_model`.
            Callers pass their own module-level resolver so an existing monkeypatch
            on that name keeps working.

    Raises:
        ConfigError: If ``backend`` is not an expressive backend.
        ModelNotFoundError: If the backend's checkpoint is missing and the backend
            cannot fetch its own weights (Chatterbox is exempt, see below).
    """
    resolve_fn = resolve or resolve_model
    if backend == "cosyvoice3":
        from .cosyvoice import CosyVoiceEngine

        return CosyVoiceEngine(
            resolve_fn("tts_cosyvoice3", models_dir),
            device=device,
        )
    if backend == "chatterbox":
        from .chatterbox import ChatterboxEngine

        try:
            model_dir: Path | None = resolve_fn("tts_chatterbox", models_dir)
        except ModelNotFoundError:
            # Chatterbox fetches its own weights by repo id; a missing local copy
            # is not an error.
            model_dir = None
        return ChatterboxEngine(
            model_dir,
            device=device,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
    if backend == "fish":
        from .fish import FishEngine

        return FishEngine(
            resolve_fn("tts_fish", models_dir),
            device=device,
            temperature=fish_temperature,
            top_p=fish_top_p,
            quantize=fish_quantize,
        )
    raise ConfigError(
        f"Unknown expressive backend {backend!r}. Use: "
        + ", ".join(EXPRESSIVE_BACKENDS)
    )
