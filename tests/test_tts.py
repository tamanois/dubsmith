"""Tests for donor selection and the voice-cloning engine wrapper."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pipeline.asr import Segment
from pipeline.errors import TTSError
from pipeline.tts import (
    DonorClip,
    TTSEngine,
    resolve_dtype,
    score_candidate,
    select_donor,
)

SR = 24_000


def audio(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    """Deterministic non-silent signal."""
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


class TestDonorClip:
    def test_duration_and_rms(self) -> None:
        clip = DonorClip(
            samples=audio(2.0, 0.5), sample_rate=SR, text="hi", start=0.0, end=2.0, segment_id=0
        )
        assert clip.duration == pytest.approx(2.0, abs=0.01)
        assert clip.rms == pytest.approx(0.354, abs=0.02)

    def test_empty_samples_have_zero_rms(self) -> None:
        clip = DonorClip(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=SR,
            text="",
            start=0.0,
            end=0.0,
            segment_id=0,
        )
        assert clip.rms == 0.0


class TestScoreCandidate:
    def test_ideal_length_beats_a_too_short_segment(self) -> None:
        samples = audio(20.0)
        ideal = Segment(id=0, start=0.0, end=8.0, text="a")
        short = Segment(id=1, start=10.0, end=11.0, text="b")
        assert score_candidate(ideal, samples, SR) > score_candidate(short, samples, SR)

    def test_louder_segment_wins_at_equal_length(self) -> None:
        quiet = audio(10.0, amplitude=0.001)
        loud = audio(10.0, amplitude=0.5)
        segment = Segment(id=0, start=0.0, end=5.0, text="a")
        assert score_candidate(segment, loud, SR) > score_candidate(segment, quiet, SR)

    def test_high_no_speech_probability_is_penalised(self) -> None:
        samples = audio(10.0)
        clean = Segment(id=0, start=0.0, end=5.0, text="a", no_speech_prob=0.0)
        noisy = Segment(id=0, start=0.0, end=5.0, text="a", no_speech_prob=0.9)
        assert score_candidate(clean, samples, SR) > score_candidate(noisy, samples, SR)

    def test_zero_length_scores_negative_infinity(self) -> None:
        segment = Segment(id=0, start=1.0, end=1.0, text="a")
        assert score_candidate(segment, audio(2.0), SR) == float("-inf")


class TestSelectDonor:
    def test_prefers_a_segment_inside_the_window(self) -> None:
        samples = audio(20.0)
        segments = [
            Segment(id=0, start=0.0, end=1.5, text="too short"),
            Segment(id=1, start=4.0, end=12.0, text="ideal"),
        ]
        donor = select_donor(samples, SR, segments)
        assert donor.segment_id == 1
        assert donor.text == "ideal"
        assert donor.duration == pytest.approx(8.0, abs=0.01)

    def test_falls_back_to_longest_when_none_are_in_range(self) -> None:
        samples = audio(10.0)
        segments = [
            Segment(id=0, start=0.0, end=0.5, text="tiny"),
            Segment(id=1, start=1.0, end=2.5, text="longer"),
        ]
        donor = select_donor(samples, SR, segments)
        assert donor.segment_id == 1

    def test_slices_the_requested_time_range(self) -> None:
        samples = audio(10.0)
        segments = [Segment(id=0, start=2.0, end=6.0, text="clip")]
        donor = select_donor(samples, SR, segments)
        assert donor.start == 2.0
        assert len(donor.samples) == pytest.approx(4.0 * SR, abs=2)

    def test_raises_without_segments(self) -> None:
        with pytest.raises(TTSError, match="no segments"):
            select_donor(audio(1.0), SR, [])

    def test_raises_when_segment_has_no_samples(self) -> None:
        segments = [Segment(id=0, start=100.0, end=110.0, text="beyond audio")]
        with pytest.raises(TTSError, match="no audio samples"):
            select_donor(audio(1.0), SR, segments)


class FakeQwenModel:
    """Stands in for ``Qwen3TTSModel``."""

    def __init__(self, *, sample_rate: int = SR, spoken: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.spoken = spoken
        self.prompts: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.model = SimpleNamespace(tts_model_type="base")

    def create_voice_clone_prompt(self, **kwargs: Any) -> str:
        self.prompts.append(kwargs)
        return "prompt-token"

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.calls.append(kwargs)
        texts = kwargs["text"]
        count = self.spoken if self.spoken is not None else len(texts)
        return [audio(0.4) for _ in range(count)], self.sample_rate


def engine_with(model: FakeQwenModel) -> TTSEngine:
    engine = TTSEngine("models/fake", device="cpu")
    engine._model = model  # inject, bypassing the heavy loader
    return engine


class TestTTSEngine:
    def test_synthesize_requires_a_prompt_first(self) -> None:
        engine = engine_with(FakeQwenModel())
        with pytest.raises(TTSError, match="prepare_prompt"):
            engine.synthesize(["bonjour"], language="French")

    def test_prepare_prompt_from_donor_passes_arrays(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt(
            DonorClip(samples=audio(3.0), sample_rate=SR, text="hello", start=0, end=3, segment_id=0)
        )
        kwargs = model.prompts[0]
        assert kwargs["ref_text"] == "hello"
        assert kwargs["x_vector_only_mode"] is False
        assert isinstance(kwargs["ref_audio"], tuple)

    def test_prepare_prompt_from_file_requires_text(self) -> None:
        engine = engine_with(FakeQwenModel())
        with pytest.raises(TTSError, match="ref_text is required"):
            engine.prepare_prompt_from_file("ref.wav", "   ")

    def test_blank_lines_map_to_none(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        result = engine.synthesize(["un", "   ", "trois"], language="French")
        assert result[1] is None
        assert result[0] is not None and result[2] is not None
        # Only real lines are sent to the model.
        assert model.calls[0]["text"] == ["un", "trois"]

    def test_all_blank_lines_skip_the_model(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        assert engine.synthesize(["", "  "], language="French") == [None, None]
        assert model.calls == []

    def test_empty_input_returns_empty(self) -> None:
        engine = engine_with(FakeQwenModel())
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        assert engine.synthesize([], language="French") == []

    def test_batches_requests(self) -> None:
        model = FakeQwenModel()
        engine = TTSEngine("models/fake", device="cpu", batch_size=2)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        result = engine.synthesize(["a", "b", "c"], language="French")
        assert len(model.calls) == 2
        assert model.calls[0]["text"] == ["a", "b"]
        assert model.calls[1]["text"] == ["c"]
        assert all(item is not None for item in result)

    def test_language_is_forwarded(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        engine.synthesize(["a"], language="French")
        assert model.calls[0]["language"] == ["French"]

    def test_wrong_output_count_raises(self) -> None:
        model = FakeQwenModel(spoken=1)
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        with pytest.raises(TTSError, match="returned 1 clips"):
            engine.synthesize(["a", "b"], language="French")

    def test_sample_rate_available_after_synthesis(self) -> None:
        model = FakeQwenModel(sample_rate=16_000)
        engine = engine_with(model)
        with pytest.raises(TTSError):
            _ = engine.sample_rate  # unknown before the first call
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        engine.synthesize(["a"], language="French")
        assert engine.sample_rate == 16_000

    def test_output_is_mono_float32(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        result = engine.synthesize(["a"], language="French")
        assert result[0] is not None
        assert result[0].ndim == 1
        assert result[0].dtype == np.float32


class TestModelTypeGuard:
    def test_rejects_non_base_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class WrongCheckpoint:
            def __init__(self) -> None:
                self.model = SimpleNamespace(tts_model_type="custom_voice")

            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> "WrongCheckpoint":
                return cls()

        fake_module = SimpleNamespace(Qwen3TTSModel=WrongCheckpoint)
        monkeypatch.setitem(sys.modules, "qwen_tts", fake_module)

        engine = TTSEngine("models/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cpu")
        with pytest.raises(TTSError, match="does not support voice cloning"):
            _ = engine.model

    def test_accepts_base_checkpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class BaseCheckpoint:
            def __init__(self) -> None:
                self.model = SimpleNamespace(tts_model_type="base")

            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> "BaseCheckpoint":
                return cls()

        monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=BaseCheckpoint))
        engine = TTSEngine("models/Qwen3-TTS-12Hz-1.7B-Base", device="cpu")
        assert engine.model.model.tts_model_type == "base"


class TestResolveDtype:
    def test_cuda_uses_bfloat16(self) -> None:
        import torch

        assert resolve_dtype("cuda:0") == torch.bfloat16

    def test_cpu_uses_float32(self) -> None:
        import torch

        assert resolve_dtype("cpu") == torch.float32


class TestOutOfMemoryHandling:
    """A batch too large for the GPU must be split, not fail the job."""

    class OomModel(FakeQwenModel):
        def __init__(self, *, fail_above: int) -> None:
            super().__init__()
            self.fail_above = fail_above
            self.attempts: list[int] = []

        def create_voice_clone_prompt(self, **kwargs: Any) -> str:
            return "prompt-token"

        def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
            texts = kwargs["text"]
            self.attempts.append(len(texts))
            if len(texts) > self.fail_above:
                raise RuntimeError(
                    "CUDA out of memory. Tried to allocate 1.46 GiB on device 0"
                )
            return [audio(0.4) for _ in texts], SR

    def test_batch_is_halved_until_it_fits(self) -> None:
        model = self.OomModel(fail_above=1)
        engine = TTSEngine("models/fake", device="cpu", batch_size=4)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        result = engine.synthesize(["a", "b", "c", "d"], language="French")

        assert all(item is not None for item in result)
        assert len(result) == 4
        assert model.attempts[0] == 4  # started with the full batch
        assert model.attempts[-1] == 1  # ended up splitting to singles

    def test_no_oom_means_a_single_attempt(self) -> None:
        model = self.OomModel(fail_above=99)
        engine = TTSEngine("models/fake", device="cpu", batch_size=4)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        engine.synthesize(["a", "b", "c", "d"], language="French")
        assert model.attempts == [4]

    def test_oom_on_a_single_line_is_reported(self) -> None:
        model = self.OomModel(fail_above=0)
        engine = TTSEngine("models/fake", device="cpu", batch_size=1)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        with pytest.raises(TTSError, match="Out of GPU memory"):
            engine.synthesize(["a"], language="French")

    def test_unrelated_errors_are_not_retried(self) -> None:
        class BrokenModel(FakeQwenModel):
            def create_voice_clone_prompt(self, **kwargs: Any) -> str:
                return "prompt-token"

            def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
                raise RuntimeError("model file is corrupt")

        engine = TTSEngine("models/fake", device="cpu", batch_size=2)
        engine._model = BrokenModel()
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        with pytest.raises(RuntimeError, match="corrupt"):
            engine.synthesize(["a", "b"], language="French")


class TestKnnSpeakerSelection:
    def test_reference_audio_requires_transcript_text(self) -> None:
        engine = engine_with(FakeQwenModel())
        with pytest.raises(TTSError, match="ref_text is required"):
            engine.prepare_prompt_from_file("ref.wav", "")
