"""Shared test fixtures.

Every test must run without a GPU, without downloaded models, and without
ffmpeg: heavy components are injected as fakes and external binaries are mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from pipeline.asr import Segment, Transcript, Word

SAMPLE_RATE = 24_000


def speech_like(duration: float, *, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.2) -> np.ndarray:
    """Deterministic pseudo-speech: a modulated tone, non-silent and cheap."""
    t = np.linspace(0.0, duration, int(duration * sample_rate), endpoint=False)
    carrier = np.sin(2 * np.pi * 220.0 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (amplitude * carrier * envelope).astype(np.float32)


@pytest.fixture
def make_audio():
    """Return a factory producing deterministic audio arrays."""
    return speech_like


@pytest.fixture
def simple_transcript() -> Transcript:
    """A three-segment transcript with a blank-confidence profile."""
    return Transcript(
        language="en",
        language_probability=0.99,
        model="fake",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Hello there.", words=[Word(word="Hello", start=0.0, end=1.0)]),
            Segment(id=1, start=2.5, end=4.5, text="How are you?", no_speech_prob=0.05),
            Segment(id=2, start=5.0, end=8.0, text="I am fine, thanks for asking."),
        ],
    )


@pytest.fixture(autouse=True)
def _never_load_demucs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite fast and isolated by disabling real separation.

    The orchestrator would otherwise pick up the installed Demucs model on every
    pipeline test. Tests that need a background stem inject their own fake
    separator, so the real path stays covered without loading weights.
    """
    from pipeline.separation import NullSeparator

    monkeypatch.setattr(
        "pipeline.orchestrate.detect_separator", lambda *args, **kwargs: NullSeparator()
    )


@pytest.fixture(autouse=True)
def _never_load_diarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite fast and isolated by disabling real diarization.

    ``funasr`` is an optional dependency, so without this the orchestrator would
    load a speaker-embedding model on every pipeline test. Tests that need several
    speakers inject their own diarizer, keeping the real path covered without
    downloading or loading weights here.
    """
    from pipeline.diarize import NullDiarizer

    monkeypatch.setattr(
        "pipeline.orchestrate.detect_diarizer", lambda *args, **kwargs: NullDiarizer()
    )


@pytest.fixture(autouse=True)
def _never_load_speaker_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite isolated from the real SepFormer backend.

    ``speechbrain`` is an optional dependency: without this, a test that turns on
    ``--overlap separate`` would download and load the SepFormer checkpoint. Tests
    that need separation inject their own separator, so the real path stays
    covered without a model here.
    """
    from pipeline.overlap_separation import NullSpeakerSeparator

    monkeypatch.setattr(
        "pipeline.orchestrate.detect_speaker_separator",
        lambda *args, **kwargs: NullSpeakerSeparator(),
    )


# --------------------------------------------------------------------------- #
# Device profiles
# --------------------------------------------------------------------------- #


def make_fake_torch(
    *,
    kind: str = "cpu",
    capability: tuple[int, int] | None = (8, 6),
    name: str = "NVIDIA GeForce RTX 3060",
    memory: int = 12 * 1024**3,
) -> SimpleNamespace:
    """A stand-in for the ``torch`` module covering one hardware profile.

    ``kind`` is ``cpu``, ``cuda``, ``rocm``, ``xpu`` or ``mps``. Only the
    attributes the pipeline actually reads are provided, so a profile that lacks a
    capability raises rather than silently pretending to have it.
    """
    if kind not in {"cpu", "cuda", "rocm", "xpu", "mps"}:
        raise ValueError(f"unknown profile {kind!r}")

    torch = SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        version=SimpleNamespace(hip="5.7.1" if kind == "rocm" else None),
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    if kind in {"cuda", "rocm"}:
        torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index=0: name,
            get_device_properties=lambda index=0: SimpleNamespace(total_memory=memory),
            get_device_capability=lambda index=0: capability,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            # Half of the card free, a quarter ever reserved: enough for the
            # pipeline's memory-pressure checks to have something to read.
            mem_get_info=lambda index=None: (memory // 2, memory),
            max_memory_reserved=lambda index=0: memory // 4,
        )
    elif kind == "xpu":
        torch.xpu = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index=0: name,
            get_device_properties=lambda index=0: SimpleNamespace(total_memory=memory),
            empty_cache=lambda: None,
        )
    elif kind == "mps":
        torch.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
        )
        torch.mps = SimpleNamespace(empty_cache=lambda: None)
    return torch


@pytest.fixture
def torch_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., SimpleNamespace]:
    """Install a fake ``torch`` for the device layer; returns the installer.

    ``pipeline.config._import_torch`` is the single seam every device query goes
    through, so patching it makes hardware this machine does not have - Pascal,
    Volta, Ampere, CDNA, Metal, XPU - testable anyway.
    """

    def install(**kwargs: Any) -> SimpleNamespace:
        module = make_fake_torch(**kwargs)
        monkeypatch.setattr("pipeline.config._import_torch", lambda: module)
        return module

    return install
