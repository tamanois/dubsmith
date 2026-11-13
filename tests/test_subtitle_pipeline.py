"""Tests for the subtitles-only pipeline (no GPU, models or ffmpeg)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipeline.asr import Segment, Transcript
from pipeline.config import Settings, SubtitleJobConfig
from pipeline.subtitle_pipeline import SubtitlesPipeline

SR = 24_000


class FakeTranscriber:
    """Returns a fixed transcript and counts invocations."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.calls = 0

    def __call__(self, audio_path: Path) -> Transcript:
        self.calls += 1
        return self.transcript


class FakeTranslator:
    """Deterministic subtitle translation; records the requests it was given."""

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[Any] = []

    def translate_subtitles(
        self,
        requests: Any,
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        self.calls += 1
        self.requests = list(requests)
        return {
            request.id: f"[{target_language}] {request.text}" for request in requests
        }

    def translate(
        self,
        requests: Any,
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        # Fallback only: the pipeline prefers translate_subtitles when present.
        self.calls += 1
        return {request.id: request.text for request in requests}


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Hello there."),
            Segment(id=1, start=2.5, end=5.0, text="How are you?"),
        ],
    )


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "movie.mp4"
    path.write_bytes(b"not a real video")
    return path


def _patch_audio_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake probing and extraction so the pipeline never needs ffmpeg."""
    import soundfile as sf

    from pipeline import media

    monkeypatch.setattr(
        "pipeline.subtitle_pipeline.media.probe",
        lambda path: media.MediaInfo(
            duration_s=5.0, has_audio=True, has_video=True, sample_rate=SR, channels=1
        ),
    )

    def fake_extract(src: Path | str, out: Path | str, **kwargs: Any) -> Path:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0.0, 5.0, int(5.0 * SR), endpoint=False)
        sf.write(
            str(target),
            (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
            SR,
        )
        return target

    monkeypatch.setattr("pipeline.subtitle_pipeline.media.extract_audio", fake_extract)


def make_pipeline(
    tmp_path: Path,
    transcript: Transcript,
    *,
    translator: FakeTranslator | None = None,
    **job_kwargs: Any,
) -> tuple[SubtitlesPipeline, FakeTranscriber, FakeTranslator]:
    job_kwargs.setdefault("output_dir", tmp_path / "output")
    job = SubtitleJobConfig(**job_kwargs)
    transcriber = FakeTranscriber(transcript)
    translator = translator or FakeTranslator()
    pipeline = SubtitlesPipeline(
        job,
        Settings(models_dir=tmp_path / "models"),
        transcriber=transcriber,
        translator=translator,
    )
    return pipeline, transcriber, translator


class TestVideoSubtitles:
    def test_writes_translated_srt_and_embeds_soft_by_default(
        self, tmp_path: Path, transcript: Transcript, video: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_audio_inputs(monkeypatch)
        captured: list[dict[str, Any]] = []

        def fake_add(video_path: Path, srt: Path, out: Path, **kwargs: Any) -> Path:
            captured.append({"video": Path(video_path), "srt": Path(srt), **kwargs})
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"muxed")
            return Path(out)

        monkeypatch.setattr("pipeline.subtitle_pipeline.media.add_subtitle_track", fake_add)

        pipeline, _, translator = make_pipeline(
            tmp_path, transcript, video=video, target_language="French"
        )
        result = pipeline.run()

        assert translator.calls == 1
        assert result.target_srt.is_file()
        text = result.target_srt.read_text(encoding="utf-8")
        assert "[French] Hello there." in text
        assert result.output_video is not None
        assert result.output_video.name == f"{video.stem}_subtitled.mp4"
        assert captured[-1]["language"] == "French"
        assert result.report["subtitles_mode"] == "soft"

    def test_burn_mode_uses_burn_subtitles(
        self, tmp_path: Path, transcript: Transcript, video: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_audio_inputs(monkeypatch)
        captured: list[dict[str, Any]] = []

        def fake_burn(video_path: Path, srt: Path, out: Path, **kwargs: Any) -> Path:
            captured.append({"video": Path(video_path), "srt": Path(srt), **kwargs})
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"burned")
            return Path(out)

        monkeypatch.setattr("pipeline.subtitle_pipeline.media.burn_subtitles", fake_burn)

        pipeline, _, _ = make_pipeline(
            tmp_path,
            transcript,
            video=video,
            target_language="French",
            subtitles_mode="burn",
        )
        result = pipeline.run()

        assert result.output_video is not None
        assert result.output_video.name == f"{video.stem}_burned.mp4"
        assert captured[-1]["language"] == "French"

    def test_off_mode_writes_the_srt_only(
        self, tmp_path: Path, transcript: Transcript, video: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_audio_inputs(monkeypatch)
        pipeline, _, _ = make_pipeline(
            tmp_path,
            transcript,
            video=video,
            target_language="French",
            subtitles_mode="off",
        )
        result = pipeline.run()
        assert result.target_srt.is_file()
        assert result.output_video is None

    def test_target_equal_to_source_skips_translation(
        self, tmp_path: Path, transcript: Transcript, video: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_audio_inputs(monkeypatch)
        pipeline, _, translator = make_pipeline(
            tmp_path,
            transcript,
            video=video,
            target_language="English",
            subtitles_mode="off",
        )
        result = pipeline.run()

        assert translator.calls == 0
        assert "Hello there." in result.target_srt.read_text(encoding="utf-8")
        assert result.report["translated"] is False

    def test_a_long_line_becomes_several_cues(
        self, tmp_path: Path, video: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_audio_inputs(monkeypatch)
        long_transcript = Transcript(
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=6.0,
                    text="one two three four five six seven eight",
                )
            ],
        )
        pipeline, _, _ = make_pipeline(
            tmp_path,
            long_transcript,
            video=video,
            target_language="English",
            subtitles_mode="off",
            max_chars=15,
        )
        result = pipeline.run()
        assert result.report["cue_count"] > result.report["segment_count"]


class TestAudioAndCache:
    def test_audio_only_writes_no_video(self, tmp_path: Path, transcript: Transcript) -> None:
        import soundfile as sf

        audio = tmp_path / "clip.wav"
        t = np.linspace(0.0, 5.0, int(5.0 * SR), endpoint=False)
        sf.write(str(audio), (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), SR)

        pipeline, _, _ = make_pipeline(
            tmp_path, transcript, audio=audio, target_language="English"
        )
        result = pipeline.run()
        assert result.target_srt.is_file()
        assert result.output_video is None
        assert result.report["is_video"] is False

    def test_a_second_run_reuses_the_cached_transcript(
        self, tmp_path: Path, transcript: Transcript, video: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_audio_inputs(monkeypatch)
        pipeline, transcriber, _ = make_pipeline(
            tmp_path,
            transcript,
            video=video,
            target_language="English",
            subtitles_mode="off",
        )
        pipeline.run()
        pipeline.run()
        assert transcriber.calls == 1
