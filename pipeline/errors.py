"""Domain exception hierarchy.

Every failure raised by the pipeline derives from :class:`DubbingError` so the
CLI can present a single, friendly error surface instead of leaking tracebacks
for expected problems (missing model, bad input, missing binary, ...).
"""

from __future__ import annotations

__all__ = [
    "DubbingError",
    "ConfigError",
    "DependencyError",
    "MediaError",
    "ModelNotFoundError",
    "ASRError",
    "TranslationError",
    "TTSError",
    "TimingError",
]


class DubbingError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(DubbingError):
    """Invalid job configuration or unsupported option."""


class DependencyError(DubbingError):
    """A required external binary or Python package is unavailable."""


class MediaError(DubbingError):
    """Audio/video probing, extraction, or muxing failed."""


class ModelNotFoundError(DubbingError):
    """A required model is not present locally.

    Carries the shell command that downloads it so the message is directly
    actionable for the user.
    """

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        super().__init__(message)
        self.remedy = remedy


class ASRError(DubbingError):
    """Speech recognition or segmentation failed."""


class TranslationError(DubbingError):
    """Translation failed or returned an unusable response."""


class TTSError(DubbingError):
    """Speech synthesis failed."""


class TimingError(DubbingError):
    """Duration fitting or timeline assembly failed."""
