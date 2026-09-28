"""Tests for the overlapping-speech separator backend.

SpeechBrain is an optional dependency, so nothing here loads it: the SepFormer
wrapper is exercised through a fake model and a stubbed import, which is the same
seam the pipeline tests use.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from pipeline.errors import DependencyError, MediaError
from pipeline.overlap_separation import (
    DEFAULT_SEPARATION_MODEL,
    SEPARATOR_RATE,
    NullSpeakerSeparator,
    SepFormerSpeakerSeparator,
    detect_speaker_separator,
)

SR = 48_000


def tone(seconds: float, rate: int = SR, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * rate), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


class FakeSepFormerModel:
    """Mimics SpeechBrain's ``SepformerSeparation`` surface."""

    def __init__(self, sources: int = 2, rate: int = SEPARATOR_RATE) -> None:
        self.sample_rate = rate
        self.sources = sources
        self.batches: list[tuple[int, ...]] = []

    def separate_batch(self, tensor: Any) -> Any:
        import torch

        self.batches.append(tuple(tensor.shape))
        time = tensor.shape[-1]
        out = torch.zeros(tensor.shape[0], time, self.sources)
        for index in range(self.sources):
            out[:, :, index] = (index + 1) * 0.1
        return out


def separator_with(model: Any) -> SepFormerSpeakerSeparator:
    """A separator whose model is injected, so no checkpoint is loaded."""
    separator = SepFormerSpeakerSeparator()
    separator._model = model
    return separator


def hide_speechbrain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``find_spec("speechbrain")`` report the package as absent."""
    import importlib.util

    real = importlib.util.find_spec

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "speechbrain":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


class TestNullSeparator:
    """The no-op backend, used when separation is disabled or unavailable."""

    def test_the_input_is_returned_as_one_stream(self) -> None:
        data = tone(1.0)
        streams = NullSpeakerSeparator().separate(data, SR)
        # One stream, not two: there is nothing to separate, and the caller detects
        # that by the length rather than a separate enabled/disabled check.
        assert len(streams) == 1
        assert np.array_equal(streams[0], data)


class TestDetect:
    """Choosing a backend, and degrading instead of failing."""

    @pytest.mark.parametrize("preference", ["none", "off", "identity", "NONE", " none "])
    def test_none_aliases_return_the_null_separator(self, preference: str) -> None:
        assert isinstance(detect_speaker_separator(preference), NullSpeakerSeparator)

    def test_auto_degrades_when_speechbrain_is_absent(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hide_speechbrain(monkeypatch)
        with caplog.at_level(logging.WARNING):
            separator = detect_speaker_separator("auto")
        assert isinstance(separator, NullSpeakerSeparator)
        assert "speechbrain is not installed" in caplog.text

    def test_sepformer_requires_speechbrain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hide_speechbrain(monkeypatch)
        with pytest.raises(DependencyError, match="speechbrain"):
            detect_speaker_separator("sepformer")

    def test_an_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(DependencyError, match="Unknown speaker separator"):
            detect_speaker_separator("magic")


class TestSepFormerSeparator:
    """The wrapper's audio handling, with the model stubbed out."""

    def test_it_returns_one_stream_per_speaker_at_the_input_rate(self) -> None:
        model = FakeSepFormerModel()
        separator = separator_with(model)
        data = tone(2.0, SR)

        streams = separator.separate(data, SR, num_speakers=2)

        assert len(streams) == 2
        for stream in streams:
            assert stream.dtype == np.float32
            # Same length as the input, so the caller can align to the timeline.
            assert stream.size == data.size
        # One batch of (1, time): the mono region is run as a single mixture.
        assert model.batches[0][0] == 1

    def test_the_default_checkpoint_is_the_16k_sepformer(self) -> None:
        assert DEFAULT_SEPARATION_MODEL == "speechbrain/sepformer-whamr16k"
        assert SepFormerSpeakerSeparator().model_id == DEFAULT_SEPARATION_MODEL

    def test_a_number_of_speakers_other_than_two_is_rejected(self) -> None:
        with pytest.raises(MediaError, match="exactly 2"):
            separator_with(FakeSepFormerModel()).separate(tone(1.0), SR, num_speakers=3)

    def test_empty_audio_yields_two_empty_streams(self) -> None:
        streams = separator_with(FakeSepFormerModel()).separate(
            np.zeros(0, dtype=np.float32), SR
        )
        assert len(streams) == 2
        assert all(stream.size == 0 for stream in streams)

    def test_a_model_with_too_few_sources_is_rejected(self) -> None:
        with pytest.raises(MediaError, match="one stream per speaker"):
            separator_with(FakeSepFormerModel(sources=1)).separate(tone(1.0), SR)

    def test_a_model_returning_a_bad_shape_is_rejected(self) -> None:
        import torch

        class BadModel:
            sample_rate = SEPARATOR_RATE

            def separate_batch(self, tensor: Any) -> Any:
                return torch.zeros(tensor.shape[0], 10)

        with pytest.raises(MediaError, match="SepFormer returned"):
            separator_with(BadModel()).separate(tone(1.0), SR)

    def test_the_import_error_names_the_install_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".")[0] == "speechbrain":
                raise ImportError("speechbrain is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(DependencyError, match="pip install speechbrain"):
            SepFormerSpeakerSeparator().model
