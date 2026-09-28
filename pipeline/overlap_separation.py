"""Separating two speakers inside one overlapping segment.

`pipeline/separation.py` splits *music and ambience* from dialogue, so the bed can
survive under the dub. This module solves a different problem: a single ASR
segment that carries **two voices at once** (the pipeline is per-segment, so such a
line is otherwise attributed to whoever dominates it and the second voice is
cloned in the wrong timbre).

The backend is pluggable and optional, exactly like Demucs and funasr. The default
is SpeechBrain's SepFormer at 16 kHz: it is pip-installable, self-contained, and
emits two time-aligned streams, so the result needs no extra resampling beyond the
pipeline's 48 kHz working rate. funasr was the original plan, but the installed
funasr 1.4.x exposes **no** registered separation model (and no
``ss_inference_launch``), and ModelScope only ships 8 kHz checkpoints - so neither
can be used without a new or repaired dependency.

Streams are used to recover the *second voice's words* and to assign each stream a
speaker. They are never mixed into the delivered dub: that is always synthesised
TTS in the matched speaker's own voice.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

import numpy as np

from .errors import DependencyError, MediaError
from .media import resample

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SEPARATION_MODEL",
    "SEPARATOR_RATE",
    "NullSpeakerSeparator",
    "SepFormerSpeakerSeparator",
    "SpeakerSeparator",
    "detect_speaker_separator",
]

#: SepFormer checkpoints operate at 16 kHz.
SEPARATOR_RATE = 16_000

#: SpeechBrain's 16 kHz SepFormer, trained on WHAMR! (noisy, reverberant, two
#: speakers). The 8 kHz `sepformer-wsj02mix` is deliberately *not* the default:
#: re-transcribing 8 kHz speech for a second voice loses consonants, which is the
#: one thing this stage exists to recover.
DEFAULT_SEPARATION_MODEL = "speechbrain/sepformer-whamr16k"

#: SepFormer is a fixed two-speaker model.
SUPPORTED_SPEAKERS = 2


class SpeakerSeparator(Protocol):
    """Splits one mixed region into one stream per speaker."""

    name: str

    def separate(
        self, samples: np.ndarray, sample_rate: int, num_speakers: int = SUPPORTED_SPEAKERS
    ) -> list[np.ndarray]:
        """Return one mono stream per speaker at ``sample_rate``.

        Every stream is the same length as the input, so a caller can align them
        to the original timeline sample-for-sample.
        """
        ...


class NullSpeakerSeparator:
    """No-op separator: the input is returned as a single stream.

    One stream (rather than two) is the honest answer - there is nothing to
    separate - and it lets the caller detect that separation did not happen
    without a separate "is this enabled" check.
    """

    name = "none"

    def separate(
        self, samples: np.ndarray, sample_rate: int, num_speakers: int = SUPPORTED_SPEAKERS
    ) -> list[np.ndarray]:
        """Return the input unchanged, as a single stream."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        return [data]


class SepFormerSpeakerSeparator:
    """SpeechBrain SepFormer-backed separator (16 kHz, two speakers)."""

    name = "sepformer"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SEPARATION_MODEL,
        device: str = "auto",
        savedir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        #: Where SpeechBrain caches the downloaded checkpoint. ``None`` uses its
        #: own default (~/.cache/speechbrain), which is fine for a CLI.
        self.savedir = savedir
        self._model: Any | None = None

    # -- lifecycle --------------------------------------------------------- #

    @property
    def model(self) -> Any:
        """Load the checkpoint on first use and keep it resident."""
        if self._model is not None:
            return self._model

        separator_cls = _sepformer_class()
        # Imported locally: config.py imports this module for its defaults, so a
        # top-level import would cycle.
        from .config import resolve_device

        kwargs: dict[str, Any] = {}
        if self.savedir is not None:
            kwargs["savedir"] = str(self.savedir)
        self._model = separator_cls.from_hparams(
            source=self.model_id,
            run_opts={"device": resolve_device(self.device)},
            **kwargs,
        )
        return self._model

    def _model_rate(self, model: Any) -> int:
        """The model's own sample rate, falling back to SepFormer's 16 kHz."""
        rate = getattr(model, "sample_rate", None)
        if rate is None:
            rate = getattr(getattr(model, "hparams", None), "sample_rate", None)
        return int(rate or SEPARATOR_RATE)

    # -- separation -------------------------------------------------------- #

    def separate(
        self, samples: np.ndarray, sample_rate: int, num_speakers: int = SUPPORTED_SPEAKERS
    ) -> list[np.ndarray]:
        """Separate a mixed region into two speaker streams.

        Args:
            samples: Mono audio of the mixed region.
            sample_rate: Rate of ``samples``.
            num_speakers: Must be 2; SepFormer models a fixed two-speaker mixture.

        Returns:
            Two mono float32 arrays at ``sample_rate``, each the same length as the
            input.

        Raises:
            DependencyError: If SpeechBrain is not installed.
            MediaError: If a number of speakers other than two is requested, or the
                model does not return per-speaker output.
        """
        if int(num_speakers) != SUPPORTED_SPEAKERS:
            raise MediaError(
                f"SepFormer separates exactly {SUPPORTED_SPEAKERS} speakers, "
                f"not {num_speakers}."
            )

        import torch

        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return [data, data]

        model = self.model
        rate = self._model_rate(model)
        working = data if sample_rate == rate else resample(data, sample_rate, rate)

        # `separate_batch` takes (batch, time) and returns (batch, time, nsrc).
        tensor = torch.from_numpy(np.ascontiguousarray(working, dtype=np.float32))
        with torch.no_grad():
            estimate = model.separate_batch(tensor.unsqueeze(0))

        streams = np.asarray(estimate[0].detach().cpu().numpy(), dtype=np.float32)
        if streams.ndim != 2 or streams.shape[-1] < SUPPORTED_SPEAKERS:
            raise MediaError(
                f"SepFormer returned {streams.shape}; expected one stream per "
                f"speaker ({SUPPORTED_SPEAKERS})."
            )

        outputs: list[np.ndarray] = []
        for index in range(SUPPORTED_SPEAKERS):
            stream = streams[:, index]
            if rate != sample_rate:
                stream = resample(stream, rate, sample_rate)
            outputs.append(_fit_length(stream, data.size))
        return outputs


def _sepformer_class() -> Any:
    """The SpeechBrain SepFormer inference class, from either import path.

    SpeechBrain moved its inference modules (``speechbrain.pretrained`` ->
    ``speechbrain.inference``) between releases, so both are tried rather than
    pinning a version.
    """
    try:
        from speechbrain.inference.separation import SepformerSeparation

        return SepformerSeparation
    except ImportError:
        pass
    try:
        from speechbrain.pretrained import SepformerSeparation

        return SepformerSeparation
    except ImportError as exc:
        raise DependencyError(
            "speechbrain is not installed, so overlapping speech cannot be "
            "separated. Install it with: pip install speechbrain, or pass "
            "--overlap dominant to dub each flagged line in one voice."
        ) from exc


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


def detect_speaker_separator(
    preference: str = "auto",
    *,
    device: str = "auto",
    model_id: str = DEFAULT_SEPARATION_MODEL,
    savedir: str | None = None,
) -> SpeakerSeparator:
    """Select a speech-separation backend, degrading rather than failing.

    Args:
        preference: ``auto`` (SepFormer when SpeechBrain is importable), ``none``/
            ``off`` to disable, or ``sepformer`` to require it.
        device: Preferred torch device.
        model_id: SpeechBrain checkpoint id.
        savedir: Where SpeechBrain caches the checkpoint.

    Returns:
        A ready-to-use separator; :class:`NullSpeakerSeparator` when disabled or
        unavailable, so the surrounding job keeps running.

    Raises:
        DependencyError: If ``sepformer`` was explicitly requested but SpeechBrain
            is missing.
    """
    import importlib.util

    key = (preference or "auto").strip().lower()
    if key in {"none", "off", "identity"}:
        return NullSpeakerSeparator()

    available = importlib.util.find_spec("speechbrain") is not None

    if key == "sepformer":
        if not available:
            raise DependencyError(
                "Overlapping-speech separation was requested but speechbrain is "
                "not installed. Install it with: pip install speechbrain"
            )
        return SepFormerSpeakerSeparator(model_id=model_id, device=device, savedir=savedir)

    if key != "auto":
        raise DependencyError(
            f"Unknown speaker separator {preference!r}. Choose from: auto, none, "
            "sepformer"
        )

    if available:
        return SepFormerSpeakerSeparator(model_id=model_id, device=device, savedir=savedir)
    _LOGGER.warning(
        "speechbrain is not installed, so overlapping speech cannot be separated; "
        "each flagged line is dubbed in its dominant speaker's voice. Install it "
        "with: pip install speechbrain"
    )
    return NullSpeakerSeparator()
