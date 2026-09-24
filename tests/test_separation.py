"""Tests for speech/background separation.

The real Demucs model is never loaded: a fake demucs module stands in so the
wrapping logic (channel duplication, resampling around the model's rate, stem
aggregation, length alignment, backend selection) can be verified quickly.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pipeline.errors import DependencyError, MediaError
from pipeline.separation import (
    DEMUCS_SAMPLE_RATE,
    DemucsSeparator,
    NullSeparator,
    detect_separator,
)

SR = 48_000


class FakeDemucsModel:
    """Stands in for a pretrained demucs model."""

    sources = ["drums", "bass", "other", "vocals"]
    samplerate = DEMUCS_SAMPLE_RATE

    def eval(self) -> "FakeDemucsModel":
        return self


def install_fake_demucs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shape_log: list[tuple[int, ...]] | None = None,
    samplerate: int = DEMUCS_SAMPLE_RATE,
) -> None:
    """Register a fake ``demucs`` package that returns constant stems."""

    class Model(FakeDemucsModel):
        pass

    Model.samplerate = samplerate

    def fake_apply(model: Any, batch: Any, **kwargs: Any) -> Any:
        import torch

        if shape_log is not None:
            shape_log.append(tuple(batch.shape))
        samples = int(batch.shape[-1])
        # 4 stems: vocals loud, the rest quiet.
        stems = torch.zeros(1, 4, 2, samples)
        stems[:, 3] = 0.4  # vocals
        stems[:, 0] = 0.01
        stems[:, 1] = 0.01
        stems[:, 2] = 0.01
        return stems

    monkeypatch.setitem(
        sys.modules,
        "demucs",
        SimpleNamespace(__path__=["demucs"]),
    )
    monkeypatch.setitem(sys.modules, "demucs.pretrained", SimpleNamespace(get_model=lambda name: Model()))
    monkeypatch.setitem(sys.modules, "demucs.apply", SimpleNamespace(apply_model=fake_apply))


class TestNullSeparator:
    def test_passes_audio_through_as_speech(self) -> None:
        samples = np.full(1000, 0.5, dtype=np.float32)
        speech, background = NullSeparator().separate(samples, SR)
        assert np.array_equal(speech, samples)
        assert np.array_equal(background, np.zeros(1000, dtype=np.float32))

    def test_name_is_stable(self) -> None:
        assert NullSeparator().name == "none"


class TestDetectSeparator:
    def test_none_returns_null(self) -> None:
        assert isinstance(detect_separator("none"), NullSeparator)

    def test_auto_returns_null_when_demucs_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        assert isinstance(detect_separator("auto"), NullSeparator)

    def test_auto_prefers_demucs_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
        assert isinstance(detect_separator("auto"), DemucsSeparator)

    def test_explicit_demucs_missing_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        with pytest.raises(DependencyError, match="demucs is not installed"):
            detect_separator("demucs")

    def test_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(DependencyError, match="Unknown separator"):
            detect_separator("magic")


class TestDemucsSeparator:
    def test_model_is_loaded_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setitem(sys.modules, "demucs", SimpleNamespace(__path__=["demucs"]))
        monkeypatch.setitem(
            sys.modules,
            "demucs.pretrained",
            SimpleNamespace(get_model=lambda name: (calls.append(name), FakeDemucsModel())[1]),
        )
        monkeypatch.setitem(sys.modules, "demucs.apply", SimpleNamespace(apply_model=lambda *a, **k: None))

        separator = DemucsSeparator(device="cpu")
        separator._load()
        separator._load()
        assert calls == ["htdemucs"]

    def test_mono_is_duplicated_to_stereo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shapes: list[tuple[int, ...]] = []
        install_fake_demucs(monkeypatch, shape_log=shapes)

        separator = DemucsSeparator(device="cpu")
        separator.separate(np.zeros(SR, dtype=np.float32), SR)

        assert shapes and shapes[0][1] == 2  # (batch, channels, time)

    def test_input_is_resampled_to_the_model_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shapes: list[tuple[int, ...]] = []
        install_fake_demucs(monkeypatch, shape_log=shapes, samplerate=DEMUCS_SAMPLE_RATE)

        separator = DemucsSeparator(device="cpu")
        separator.separate(np.zeros(SR * 2, dtype=np.float32), SR)  # 2s at 48k

        expected = int(2 * DEMUCS_SAMPLE_RATE)
        assert abs(shapes[0][-1] - expected) < 100  # ~44.1k samples, resampling slack

    def test_output_length_matches_the_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_demucs(monkeypatch)
        separator = DemucsSeparator(device="cpu")
        speech, background = separator.separate(np.zeros(SR * 3, dtype=np.float32), SR)
        assert speech.size == SR * 3
        assert background.size == SR * 3

    def test_vocals_go_to_speech_and_the_rest_to_background(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_demucs(monkeypatch)
        separator = DemucsSeparator(device="cpu")
        speech, background = separator.separate(np.zeros(SR, dtype=np.float32), SR)

        rms = lambda values: float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        assert rms(speech) > rms(background)

    def test_unknown_stems_are_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class NarrowModel(FakeDemucsModel):
            sources = ["vocals"]  # no drums/bass/other

        monkeypatch.setitem(sys.modules, "demucs", SimpleNamespace(__path__=["demucs"]))
        monkeypatch.setitem(
            sys.modules, "demucs.pretrained", SimpleNamespace(get_model=lambda name: NarrowModel())
        )
        monkeypatch.setitem(sys.modules, "demucs.apply", SimpleNamespace(apply_model=lambda *a, **k: None))

        separator = DemucsSeparator(device="cpu")
        with pytest.raises(MediaError, match="does not provide stems"):
            separator.separate(np.zeros(SR, dtype=np.float32), SR)

    def test_empty_input_is_safe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_demucs(monkeypatch)
        separator = DemucsSeparator(device="cpu")
        speech, background = separator.separate(np.zeros(0, dtype=np.float32), SR)
        assert speech.size == 0 and background.size == 0

    def test_custom_stem_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_demucs(monkeypatch)
        separator = DemucsSeparator(device="cpu", speech_sources=("other",))
        speech, _ = separator.separate(np.zeros(SR, dtype=np.float32), SR)
        rms = float(np.sqrt(np.mean(np.square(speech, dtype=np.float64))))
        assert rms == pytest.approx(0.01, abs=0.005)  # picked the quiet stem
