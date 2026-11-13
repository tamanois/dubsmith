"""Speech / background separation.

The MVP replaced the video's whole audio track with synthesised speech, which
deleted music and ambience and made the result feel oddly sterile. This stage
splits the source into a *speech* stem and a *background* stem so the new dub can
be mixed back over the original soundtrack.

`htdemucs` is the workhorse: it is stereo-native and operates at 44.1 kHz, so
mono audio is duplicated across both channels and everything is resampled around
the model. Separation is cheap relative to the rest of the pipeline (~20x
realtime on a mid-range GPU), and `apply_model(split=True)` already chunks
internally with overlap, so no manual windowing is needed.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

import numpy as np

from . import options
from .errors import DependencyError, MediaError
from .media import resample

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "Separator",
    "NullSeparator",
    "DemucsSeparator",
    "detect_separator",
    "DEMUCS_SAMPLE_RATE",
]

#: htdemucs was trained at 44.1 kHz and expects that rate.
DEMUCS_SAMPLE_RATE = 44_100

#: Stems demucs produces for music; speech lands in "vocals".
DEFAULT_SPEECH_SOURCES = ("vocals",)
DEFAULT_BACKGROUND_SOURCES = ("drums", "bass", "other")


class Separator(Protocol):
    """Splits audio into a speech stem and a background stem."""

    name: str

    def separate(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(speech, background)``, both mono at ``sample_rate``."""
        ...


class NullSeparator:
    """No-op separator: treats everything as speech and yields silence.

    Used when separation is disabled or unavailable, so the rest of the pipeline
    is unchanged and the MVP behaviour remains available.
    """

    name = options.Separator.NONE.value

    def separate(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the input as speech and an equally long silent background."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        return data, np.zeros_like(data)


class DemucsSeparator:
    """Demucs-backed separator (``htdemucs`` by default)."""

    name = options.Separator.DEMUCS.value

    def __init__(
        self,
        *,
        model_name: str = "htdemucs",
        device: str = "auto",
        speech_sources: Sequence[str] = DEFAULT_SPEECH_SOURCES,
        background_sources: Sequence[str] = DEFAULT_BACKGROUND_SOURCES,
        overlap: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.speech_sources = tuple(speech_sources)
        self.background_sources = tuple(background_sources)
        self.overlap = overlap
        self._model: Any | None = None

    # -- lifecycle --------------------------------------------------------- #

    def _load(self) -> Any:
        """Load the demucs model once and keep it resident."""
        if self._model is not None:
            return self._model

        try:
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise DependencyError(
                "demucs is not installed but background preservation is enabled. "
                "Install it with: pip install demucs   (or pass --no-background)"
            ) from exc

        model = get_model(self.model_name)
        model.eval()
        self._model = model
        return model

    @property
    def model_sample_rate(self) -> int:
        """Sample rate the model expects (44.1 kHz for htdemucs)."""
        model = self._load()
        return int(getattr(model, "samplerate", DEMUCS_SAMPLE_RATE))

    # -- separation -------------------------------------------------------- #

    def separate(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate ``samples`` into speech and background stems.

        Args:
            samples: Mono audio.
            sample_rate: Rate of ``samples``.

        Returns:
            ``(speech, background)`` as mono float32 at ``sample_rate``, the same
            length as the input.

        Raises:
            DependencyError: If demucs is not installed.
            MediaError: If the model does not expose the expected stems.
        """
        import torch
        from demucs.apply import apply_model

        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return data, data

        model = self._load()
        sources = list(getattr(model, "sources", []))
        missing = [
            stem
            for stem in (*self.speech_sources, *self.background_sources)
            if stem not in sources
        ]
        if missing:
            raise MediaError(
                f"demucs model {self.model_name!r} does not provide stems {missing}; "
                f"available: {sources}"
            )

        # The model is stereo-native and rate-specific: resample to its rate and
        # duplicate the mono channel, then undo both on the way out.
        model_rate = self.model_sample_rate
        working = data if sample_rate == model_rate else resample(data, sample_rate, model_rate)

        mono = torch.from_numpy(np.ascontiguousarray(working, dtype=np.float32))
        batch = mono.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # (1, channels, time)

        with torch.no_grad():
            estimate = apply_model(
                model,
                batch,
                device=self._device(),
                split=True,
                overlap=self.overlap,
            )[0]

        def stem(name: str) -> np.ndarray:
            tensor = estimate[sources.index(name)]
            return tensor.mean(dim=0).cpu().numpy().astype(np.float32)

        speech = sum(stem(name) for name in self.speech_sources)
        background = sum(stem(name) for name in self.background_sources)

        if model_rate != sample_rate:
            speech = resample(speech, model_rate, sample_rate)
            background = resample(background, model_rate, sample_rate)

        # Resampling can shift the length by a sample or two; keep the timeline
        # aligned with the original audio.
        speech = _fit_length(speech, data.size)
        background = _fit_length(background, data.size)
        return speech, background

    def _device(self) -> str:
        """The torch device for Demucs, with the GPU index preserved.

        Narrowing ``cuda:1`` to ``cuda`` would put the model on the first GPU,
        colliding with whatever else is resident there - which is the whole point
        of pinning a device on a multi-GPU box.
        """
        if self.device and self.device != "auto":
            return str(self.device)
        # Imported locally: config.py imports this module for its defaults, so a
        # top-level import would cycle.
        from .config import resolve_device

        return resolve_device("auto")


def _fit_length(values: np.ndarray, size: int) -> np.ndarray:
    """Pad or trim ``values`` to exactly ``size`` samples."""
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    if data.size == size:
        return data
    if data.size > size:
        return data[:size]
    padded = np.zeros(size, dtype=np.float32)
    padded[: data.size] = data
    return padded


def detect_separator(
    preference: str = options.Defaults.SEPARATOR,
    *,
    device: str = "auto",
    model_name: str = "htdemucs",
) -> Separator:
    """Select a separator backend.

    Args:
        preference: ``auto`` (demucs when importable), ``none``/``off`` to
            disable, or ``demucs`` to require it.
        device: Preferred torch device.
        model_name: Demucs model name.

    Returns:
        A ready-to-use separator.

    Raises:
        DependencyError: If ``demucs`` was explicitly requested but is missing.
    """
    import importlib.util

    preference = (preference or options.Defaults.SEPARATOR).lower()
    canonical = options.SEPARATOR_ALIASES.get(preference)
    if canonical == options.Separator.NONE:
        return NullSeparator()

    available = importlib.util.find_spec("demucs") is not None

    if canonical == options.Separator.DEMUCS:
        if not available:
            raise DependencyError(
                "Background separation requested but demucs is not installed. "
                "Install it with: pip install demucs"
            )
        return DemucsSeparator(model_name=model_name, device=device)

    if canonical != options.Separator.AUTO:
        raise DependencyError(
            f"Unknown separator {preference!r}. Choose from: "
            f"{', '.join(options.SEPARATORS)}"
        )

    if available:
        return DemucsSeparator(model_name=model_name, device=device)
    _LOGGER.warning(
        "demucs is not installed, so background music and ambience cannot be "
        "preserved. Install it with: pip install demucs"
    )
    return NullSeparator()
