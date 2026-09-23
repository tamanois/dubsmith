"""Shared test fixtures.

Every test must run without a GPU, without downloaded models, and without
ffmpeg: heavy components are injected as fakes and external binaries are mocked.
"""

from __future__ import annotations

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
