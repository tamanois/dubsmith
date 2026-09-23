"""Tests for speech recognition post-processing and transcript persistence."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.asr import (
    Segment,
    Transcript,
    _attempt_plan,
    _looks_like_cuda_library_failure,
    build_transcript,
    load_transcript,
    resolve_compute_type,
    save_transcript,
    transcribe,
)


def raw(
    start: float,
    end: float,
    text: str,
    *,
    no_speech_prob: float = 0.0,
    words: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Build a duck-typed stand-in for a faster-whisper segment."""
    return SimpleNamespace(
        start=start, end=end, text=text, no_speech_prob=no_speech_prob, words=words or []
    )


class TestBuildTranscript:
    def test_keeps_clean_segments_and_assigns_ids(self) -> None:
        transcript = build_transcript(
            [raw(0, 1, "one"), raw(1, 2, "two")], language="en"
        )
        assert [s.id for s in transcript.segments] == [0, 1]
        assert transcript.language == "en"

    def test_blank_text_is_dropped(self) -> None:
        transcript = build_transcript(
            [raw(0, 1, "  "), raw(1, 2, "real")], language="en"
        )
        assert [s.text for s in transcript.segments] == ["real"]

    def test_high_no_speech_probability_is_dropped(self) -> None:
        transcript = build_transcript(
            [raw(0, 1, "hallucination", no_speech_prob=0.95), raw(1, 2, "real")],
            language="en",
            no_speech_threshold=0.6,
        )
        assert [s.text for s in transcript.segments] == ["real"]

    def test_threshold_is_inclusive(self) -> None:
        transcript = build_transcript(
            [
                raw(1, 2, "borderline", no_speech_prob=0.6),
                raw(2, 3, "kept", no_speech_prob=0.59),
            ],
            language="en",
            no_speech_threshold=0.6,
        )
        # Exactly at the threshold is dropped; just below it is kept.
        assert [s.text for s in transcript.segments] == ["kept"]

    def test_zero_length_segments_are_dropped(self) -> None:
        transcript = build_transcript(
            [raw(1, 1, "empty"), raw(2, 3, "real")], language="en"
        )
        assert [s.text for s in transcript.segments] == ["real"]

    def test_negative_start_is_clamped(self) -> None:
        transcript = build_transcript([raw(-2, 1, "clamped")], language="en")
        assert transcript.segments[0].start == 0.0

    def test_ids_stay_contiguous_after_filtering(self) -> None:
        transcript = build_transcript(
            [
                raw(0, 1, "a"),
                raw(1, 2, ""),
                raw(2, 3, "b", no_speech_prob=0.9),
                raw(3, 4, "c"),
            ],
            language="en",
        )
        assert [s.id for s in transcript.segments] == [0, 1]

    def test_words_are_parsed(self) -> None:
        words = [SimpleNamespace(word=" hi ", start=0.0, end=0.5)]
        transcript = build_transcript([raw(0, 1, "hi", words=words)], language="en")
        assert transcript.segments[0].words[0].word == "hi"
        assert transcript.segments[0].words[0].end == 0.5

    def test_none_words_becomes_empty_list(self) -> None:
        transcript = build_transcript([raw(0, 1, "hi", words=[None])], language="en")
        assert transcript.segments[0].words == []

    def test_raises_when_nothing_is_usable(self) -> None:
        with pytest.raises(Exception, match="No speech detected"):
            build_transcript([raw(0, 1, "")], language="en")

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(Exception, match="No speech detected"):
            build_transcript([], language="en")

    def test_duration_and_end_of_transcript(self) -> None:
        transcript = build_transcript([raw(0, 2, "a"), raw(2, 5, "b")], language="en")
        assert transcript.duration_s == 5.0
        assert transcript.segments[0].duration == 2.0


class TestResolveComputeType:
    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "CUDA"])
    def test_cuda_uses_float16(self, device: str) -> None:
        assert resolve_compute_type(device) == "float16"

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_cpu_uses_int8(self, device: str) -> None:
        assert resolve_compute_type(device) == "int8"


class TestPersistence:
    def test_roundtrip(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        path = save_transcript(simple_transcript, tmp_path / "t.json")
        restored = load_transcript(path)
        assert restored == simple_transcript

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        transcript = Transcript(language="en", segments=[Segment(start=0, end=1, text="x")])
        path = save_transcript(transcript, tmp_path / "a" / "b" / "t.json")
        assert path.is_file()

    def test_empty_transcript_is_reported(self) -> None:
        assert Transcript().is_empty() is True


def install_fake_whisper(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: Exception | None = None,
    fail_devices: tuple[str, ...] = ("cuda",),
    preload_result: int = 0,
) -> list[str]:
    """Patch faster_whisper with a fake that fails on the chosen devices.

    Returns the list of devices the pipeline attempted, in order, so the retry
    behaviour can be asserted.
    """
    attempts: list[str] = []

    class FakeWhisperModel:
        def __init__(self, path: str, *, device: str, compute_type: str) -> None:
            self.device = device
            attempts.append(device)

        def transcribe(self, *args: object, **kwargs: object):
            if self.device in fail_devices:
                raise error or RuntimeError(
                    "Library libcublas.so.12 is not found or cannot be loaded"
                )
            return iter([raw(0.0, 1.0, "hello there")]), SimpleNamespace(
                language="en", language_probability=0.9
            )

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(
        "pipeline.config.resolve_device",
        lambda requested="auto": "cuda" if requested == "auto" else requested,
    )
    # Default to "no vendored CUDA libraries available" so the retry chain is
    # deterministic regardless of which wheels happen to be installed.
    monkeypatch.setattr("pipeline.asr.preload_cuda_libraries", lambda: preload_result)
    return attempts


class TestCudaLibraryFailureDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "Library libcublas.so.12 is not found or cannot be loaded",
            "libcudnn.so.9: cannot open shared object file",
            "CUDA error: no kernel image is available",
            "ctranslate2 is not compiled with CUDA support",
        ],
    )
    def test_recognises_library_failures(self, message: str) -> None:
        assert _looks_like_cuda_library_failure(RuntimeError(message)) is True

    def test_ignores_unrelated_errors(self) -> None:
        assert _looks_like_cuda_library_failure(ValueError("bad audio")) is False


class TestTranscribeFallback:
    def test_falls_back_to_cpu_when_cuda_libraries_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = install_fake_whisper(monkeypatch)
        transcript = transcribe(tmp_path / "a.wav", tmp_path / "model")

        assert attempts == ["cuda", "cpu"]
        assert transcript.language == "en"
        assert len(transcript.segments) == 1

    def test_unrelated_runtime_error_becomes_domain_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_whisper(
            monkeypatch, error=RuntimeError("something else entirely"), fail_devices=("cuda", "cpu")
        )
        with pytest.raises(Exception, match="Speech recognition failed"):
            transcribe(tmp_path / "a.wav", tmp_path / "model")

    def test_fallback_can_be_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = install_fake_whisper(monkeypatch)
        with pytest.raises(Exception, match="Speech recognition failed"):
            transcribe(tmp_path / "a.wav", tmp_path / "model", allow_cpu_fallback=False)
        assert attempts == ["cuda"]

    def test_explicit_cpu_device_does_not_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = install_fake_whisper(monkeypatch, fail_devices=("cpu",))
        with pytest.raises(Exception, match="Speech recognition failed"):
            transcribe(tmp_path / "a.wav", tmp_path / "model", device="cpu")
        assert attempts == ["cpu"]

    def test_cuda_success_does_not_touch_cpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = install_fake_whisper(monkeypatch, fail_devices=())
        transcribe(tmp_path / "a.wav", tmp_path / "model")
        assert attempts == ["cuda"]

    def test_compute_type_follows_the_device_that_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, str]] = []

        class RecordingModel:
            def __init__(self, path: str, *, device: str, compute_type: str) -> None:
                seen.append((device, compute_type))

            def transcribe(self, *args: object, **kwargs: object):
                if seen[-1][0] == "cuda":
                    raise RuntimeError("Library libcublas.so.12 is not found")
                return iter([raw(0.0, 1.0, "hi")]), SimpleNamespace(language="en")

        monkeypatch.setitem(
            sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=RecordingModel)
        )
        monkeypatch.setattr("pipeline.config.resolve_device", lambda requested="auto": "cuda")
        monkeypatch.setattr("pipeline.asr.preload_cuda_libraries", lambda: 0)
        transcribe(tmp_path / "a.wav", tmp_path / "model")

        assert seen == [("cuda", "float16"), ("cpu", "int8")]


class TestAttemptPlan:
    def test_cuda_with_fallback_tries_preload_then_cpu(self) -> None:
        assert _attempt_plan("cuda", True) == [("cuda", False), ("cuda", True), ("cpu", False)]

    def test_cuda_without_fallback_tries_once(self) -> None:
        assert _attempt_plan("cuda", False) == [("cuda", False)]

    def test_cpu_only_tries_once(self) -> None:
        assert _attempt_plan("cpu", True) == [("cpu", False)]


class TestCudaPreload:
    def test_retries_cuda_after_successful_preload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The headline case: CUDA fails, preloading fixes it, CPU is never used."""
        state = {"cuda_ready": False}
        attempts: list[str] = []

        class Model:
            def __init__(self, path: str, *, device: str, compute_type: str) -> None:
                self.device = device
                attempts.append(device)

            def transcribe(self, *args: object, **kwargs: object):
                if self.device == "cuda" and not state["cuda_ready"]:
                    raise RuntimeError("Library libcublas.so.12 is not found")
                return iter([raw(0.0, 1.0, "hi")]), SimpleNamespace(language="en")

        def fake_preload() -> int:
            state["cuda_ready"] = True
            return 13

        monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
        monkeypatch.setattr("pipeline.config.resolve_device", lambda requested="auto": "cuda")
        monkeypatch.setattr("pipeline.asr.preload_cuda_libraries", fake_preload)

        transcript = transcribe(tmp_path / "a.wav", tmp_path / "model")
        assert attempts == ["cuda", "cuda"]  # never fell back to CPU
        assert transcript.language == "en"

    def test_skips_cuda_retry_when_nothing_to_preload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = install_fake_whisper(monkeypatch, preload_result=0)
        transcribe(tmp_path / "a.wav", tmp_path / "model")
        assert attempts == ["cuda", "cpu"]

    def test_raises_when_every_device_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = install_fake_whisper(monkeypatch, fail_devices=("cuda", "cpu"), preload_result=5)
        with pytest.raises(Exception, match="every available device"):
            transcribe(tmp_path / "a.wav", tmp_path / "model")
        assert attempts == ["cuda", "cuda", "cpu"]

    def test_library_directories_exist(self) -> None:
        from pipeline.asr import _cuda_library_dirs

        for directory in _cuda_library_dirs():
            assert directory.is_dir()
