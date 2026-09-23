"""End-to-end pipeline tests using injected fakes.

No GPU, no downloaded models and no ffmpeg are involved: the transcriber,
translator, TTS engine and stretcher are all fakes, and the media layer is
patched where a video would be probed or muxed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import soundfile as sf

from pipeline import media
from pipeline.asr import Transcript
from pipeline.config import JobConfig, Settings
from pipeline.errors import MediaError
from pipeline.orchestrate import Pipeline
from pipeline.timing import IdentityStretcher
from pipeline.translate import TranslationRequest

SR = 48_000


def write_input(tmp_path: Path, seconds: float = 10.0, name: str = "input.wav") -> Path:
    """Write a real WAV file so the media layer has something to read."""
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / name
    sf.write(str(path), samples, SR)
    return path


class FakeTranscriber:
    """Returns a fixed transcript and counts invocations."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.calls = 0

    def __call__(self, audio_path: Path) -> Transcript:
        self.calls += 1
        return self.transcript


class FakeTranslator:
    """Deterministic 'translation' with optional blank lines."""

    def __init__(self, blank_ids: Sequence[int] = ()) -> None:
        self.blank_ids = set(blank_ids)
        self.calls = 0

    def translate(
        self,
        requests: Sequence[TranslationRequest],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        self.calls += 1
        return {
            request.id: ("" if request.id in self.blank_ids else f"[{target_language}] {request.text}")
            for request in requests
        }


class FakeTTSEngine:
    """Mimics TTSEngine's surface without loading a model."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self.prompt_kind: str | None = None
        self.prompt_calls = 0
        self.synth_calls = 0

    def prepare_prompt(self, donor: Any) -> None:
        self.prompt_kind = "donor"
        self.prompt_calls += 1

    def prepare_prompt_from_file(self, ref_audio: Path | str, ref_text: str) -> None:
        self.prompt_kind = "file"
        self.prompt_calls += 1

    def synthesize(self, texts: Sequence[str], *, language: str) -> list[np.ndarray | None]:
        self.synth_calls += 1
        out: list[np.ndarray | None] = []
        for index, text in enumerate(texts):
            if not text.strip():
                out.append(None)
                continue
            # Vary the length so the timing stage has work to do.
            seconds = 1.0 + 0.3 * index
            t = np.linspace(0.0, seconds, int(seconds * self.sample_rate), endpoint=False)
            out.append((0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
        return out


def make_pipeline(
    tmp_path: Path,
    transcript: Transcript,
    *,
    translator: FakeTranslator | None = None,
    tts: FakeTTSEngine | None = None,
    **job_kwargs: Any,
) -> tuple[Pipeline, FakeTranscriber, FakeTranslator, FakeTTSEngine]:
    audio = job_kwargs.pop("input_path", None) or write_input(tmp_path)
    job = JobConfig(audio=audio, output_dir=tmp_path / "output", **job_kwargs)
    transcriber = FakeTranscriber(transcript)
    translator = translator or FakeTranslator()
    tts = tts or FakeTTSEngine()
    pipeline = Pipeline(
        job,
        Settings(models_dir=tmp_path / "models"),
        transcriber=transcriber,
        translator=translator,
        tts_engine=tts,
        stretcher=IdentityStretcher(),
    )
    return pipeline, transcriber, translator, tts


class TestAudioJob:
    def test_produces_every_artifact(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()

        assert result.job_dir.is_dir()
        assert result.dub_track.is_file()
        assert (result.job_dir / "01_audio.wav").is_file()
        assert (result.job_dir / "02_transcript.json").is_file()
        assert (result.job_dir / "03_script.json").is_file()
        assert (result.job_dir / "04_dubbed" / ".done").is_file()
        assert (result.job_dir / "05_mix.wav").is_file()
        assert (result.job_dir / "manifest.json").is_file()
        assert (result.job_dir / "report.json").is_file()
        assert result.source_srt is not None and result.source_srt.is_file()
        assert result.target_srt is not None and result.target_srt.is_file()

    def test_audio_only_input_has_no_video(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()
        assert result.dubbed_video is None
        assert not (result.job_dir / "video_dubbed.mp4").exists()

    def test_dub_track_length_matches_the_source(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()
        samples, sample_rate = media.load_audio(result.dub_track)
        assert len(samples) / sample_rate == pytest.approx(10.0, abs=0.1)

    def test_dub_track_is_not_silent(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()
        samples, _ = media.load_audio(result.dub_track)
        assert float(np.max(np.abs(samples))) > 0.01

    def test_report_captures_quality_signals(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        report = pipeline.run().report
        assert report["segment_count"] == 3
        assert report["source_language"] == "en"
        assert report["target_language"] == "French"
        assert report["is_video"] is False
        assert report["clamped_segments"] >= 0

    def test_report_loudness_describes_the_delivered_file(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """The reported loudness must match dub.wav, not the pre-normalisation mix."""
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()
        report = result.report

        assert report["loudness_target_lufs"] == media.TARGET_LUFS
        assert set(report) >= {
            "loudness_dub_lufs",
            "loudness_source_lufs",
            "loudness_target_lufs",
            "loudness_gain_db",
            "loudness_peak_limited",
        }

        samples, sample_rate = media.load_audio(result.dub_track)
        actual = media.measure_loudness(samples, sample_rate)
        assert actual is not None
        if report["loudness_dub_lufs"] is not None:
            assert report["loudness_dub_lufs"] == pytest.approx(actual, abs=0.05)

    def test_subtitles_can_be_disabled(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, keep_subtitles=False
        )
        result = pipeline.run()
        assert result.source_srt is None
        assert not (result.job_dir / "subs_source.srt").exists()

    def test_translated_subtitles_contain_target_text(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()
        assert result.target_srt is not None
        content = result.target_srt.read_text(encoding="utf-8")
        assert "[French] Hello there." in content


class TestDonorHandling:
    def test_auto_selects_a_donor_from_the_video(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, tts = make_pipeline(tmp_path, simple_transcript)
        pipeline.run()
        assert tts.prompt_kind == "donor"

    def test_explicit_reference_audio_is_used(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        ref = write_input(tmp_path, seconds=4.0, name="ref.wav")
        pipeline, _, _, tts = make_pipeline(
            tmp_path, simple_transcript, ref_audio=ref, ref_text="some transcript"
        )
        pipeline.run()
        assert tts.prompt_kind == "file"


class TestEdgeCases:
    def test_blank_translation_becomes_silence_without_crashing(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        translator = FakeTranslator(blank_ids=[1])
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, translator=translator)
        result = pipeline.run()
        assert result.dub_track.is_file()
        # Segment 1 is blank, so its dub file must not exist.
        assert not (result.job_dir / "04_dubbed" / "seg_0001.wav").exists()

    def test_missing_input_file_raises(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        job = JobConfig(audio=tmp_path / "ghost.wav", output_dir=tmp_path / "output")
        pipeline = Pipeline(
            job,
            Settings(models_dir=tmp_path / "models"),
            transcriber=FakeTranscriber(simple_transcript),
            translator=FakeTranslator(),
            tts_engine=FakeTTSEngine(),
            stretcher=IdentityStretcher(),
        )
        with pytest.raises(MediaError, match="not found"):
            pipeline.run()

    def test_temp_directory_is_cleaned_up(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()
        assert not (result.job_dir / "tmp").exists()

    def test_short_single_segment_transcript(self, tmp_path: Path) -> None:
        from pipeline.asr import Segment

        transcript = Transcript(
            language="en", segments=[Segment(id=0, start=0.0, end=1.2, text="Short.")]
        )
        pipeline, _, _, _ = make_pipeline(tmp_path, transcript)
        assert pipeline.run().report["segment_count"] == 1


class TestCaching:
    def test_second_run_reuses_every_stage(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, transcriber, translator, tts = make_pipeline(tmp_path, simple_transcript)
        pipeline.run()
        assert transcriber.calls == 1

        # A fresh Pipeline over the same job directory must reuse the manifest.
        second, transcriber2, translator2, tts2 = make_pipeline(
            tmp_path, simple_transcript, input_path=pipeline.input_path
        )
        # Same transcriber instances are not reused, so assert on stage timings.
        result = second.run()
        assert result.stage_seconds == {}
        assert transcriber2.calls == 0
        assert translator2.calls == 0
        assert tts2.synth_calls == 0

    def test_force_reruns_stages(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        pipeline.run()

        forced, transcriber, _, _ = make_pipeline(
            tmp_path, simple_transcript, input_path=pipeline.input_path, force=True
        )
        forced.run()
        assert transcriber.calls == 1

    def test_changing_target_language_invalidates_the_cache(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        pipeline.run()

        changed, transcriber, _, _ = make_pipeline(
            tmp_path, simple_transcript, input_path=pipeline.input_path, target_language="de"
        )
        changed.run()
        assert transcriber.calls == 1

    def test_manifest_records_stage_timings(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        pipeline.run()
        manifest = json.loads((pipeline.job_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["fingerprint"] == pipeline.job.fingerprint()
        assert {"audio", "transcribe", "translate", "synthesize", "mix"} <= set(
            manifest["stages"]
        )


class TestLongTranslationUsesTheGap:
    """A verbose translation may spill into silence, but never desynchronise."""

    @staticmethod
    def long_tone(seconds: float, sample_rate: int) -> np.ndarray:
        t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    class LongTTS(FakeTTSEngine):
        """Always returns audio far longer than the slot (as EN->FR often does)."""

        def synthesize(self, texts: Sequence[str], *, language: str) -> list[np.ndarray | None]:
            self.synth_calls += 1
            return [
                TestLongTranslationUsesTheGap.long_tone(6.0, self.sample_rate)
                if text.strip()
                else None
                for text in texts
            ]

    def placements(self, pipeline: Any) -> list[dict[str, Any]]:
        return json.loads((pipeline.job_dir / "timing.json").read_text(encoding="utf-8"))

    def test_gap_is_used_and_excess_is_reported(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, tts=self.LongTTS())
        result = pipeline.run()
        report = result.report

        # Lines used the silence before the next utterance...
        assert report["overrun_segments"] >= 1
        assert report["overrun_seconds_total"] > 0
        # ...and the residue that would have desynchronised the video was trimmed.
        assert report["trimmed_segments"] >= 1
        assert report["trimmed_seconds_total"] > 0

    def test_no_timing_drift_is_introduced(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """Every line still starts exactly where the speaker did."""
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, tts=self.LongTTS())
        pipeline.run()
        source_starts = {s.id: s.start for s in simple_transcript.segments}
        for placement in self.placements(pipeline):
            assert placement["placed_start"] == pytest.approx(
                source_starts[placement["id"]], abs=1e-3
            )

    def test_lines_never_overlap(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, tts=self.LongTTS())
        pipeline.run()
        placements = self.placements(pipeline)
        for previous, following in zip(placements, placements[1:]):
            assert following["placed_start"] >= previous["placed_end"] - 1e-6

    def test_final_line_may_run_to_the_end(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """Nothing follows the last line, so it is not trimmed to its slot."""
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, tts=self.LongTTS())
        pipeline.run()
        last = self.placements(pipeline)[-1]
        assert last["overrun"] > 1.0
        assert last["trimmed"] == 0.0


class TestVideoJob:
    @pytest.fixture
    def video_job(self, tmp_path: Path, simple_transcript: Transcript, monkeypatch: pytest.MonkeyPatch):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"not a real video")

        monkeypatch.setattr(
            media,
            "probe",
            lambda path: media.MediaInfo(
                duration_s=10.0, has_audio=True, has_video=True, sample_rate=SR, channels=2
            ),
        )

        def fake_extract(src: Path | str, out: Path | str, **kwargs: Any) -> Path:
            """Write a real WAV so downstream stages have audio to read."""
            target = Path(out)
            t = np.linspace(0.0, 10.0, int(10.0 * SR), endpoint=False)
            samples = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
            sf.write(str(target), samples, SR)
            return target

        monkeypatch.setattr(media, "extract_audio", fake_extract)

        def fake_mux(video_path: Path | str, audio: Path | str, out: Path | str) -> Path:
            Path(out).write_bytes(b"muxed")
            return Path(out)

        monkeypatch.setattr(media, "mux_audio", fake_mux)

        job = JobConfig(video=video, output_dir=tmp_path / "output")
        pipeline = Pipeline(
            job,
            Settings(models_dir=tmp_path / "models"),
            transcriber=FakeTranscriber(simple_transcript),
            translator=FakeTranslator(),
            tts_engine=FakeTTSEngine(),
            stretcher=IdentityStretcher(),
        )
        return pipeline

    def test_video_input_produces_a_dubbed_video(self, video_job: Pipeline) -> None:
        result = video_job.run()
        assert result.dubbed_video is not None
        assert result.dubbed_video.exists()
        assert result.report["is_video"] is True

    def test_video_without_audio_stream_is_rejected(
        self, tmp_path: Path, simple_transcript: Transcript, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = tmp_path / "silent.mp4"
        video.write_bytes(b"x")
        monkeypatch.setattr(
            media,
            "probe",
            lambda path: media.MediaInfo(duration_s=5.0, has_audio=False, has_video=True),
        )
        job = JobConfig(video=video, output_dir=tmp_path / "output")
        pipeline = Pipeline(
            job,
            Settings(models_dir=tmp_path / "models"),
            transcriber=FakeTranscriber(simple_transcript),
            translator=FakeTranslator(),
            tts_engine=FakeTTSEngine(),
            stretcher=IdentityStretcher(),
        )
        with pytest.raises(MediaError, match="no audio stream"):
            pipeline.run()