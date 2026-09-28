"""End-to-end pipeline tests using injected fakes.

No GPU, no downloaded models and no ffmpeg are involved: the transcriber,
translator, TTS engine and stretcher are all fakes, and the media layer is
patched where a video would be probed or muxed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import soundfile as sf

from pipeline import media
from pipeline.asr import Segment, Transcript
from pipeline.config import JobConfig, Settings
from pipeline.errors import MediaError, TranslationError
from pipeline.mixing import RT60_BOUNDS
from pipeline.orchestrate import Pipeline
from pipeline.timing import IdentityStretcher
from pipeline.translate import ShortenRequest, TranslationRequest
from pipeline.tts import BlendedReference

SR = 48_000


def write_input(tmp_path: Path, seconds: float = 10.0, name: str = "input.wav") -> Path:
    """Write a real WAV file so the media layer has something to read."""
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / name
    sf.write(str(path), samples, SR)
    return path


def rms(samples: np.ndarray, window: slice) -> float:
    """RMS of one slice, as a plain float."""
    data = np.asarray(samples[window], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(data))))


class FakeTranscriber:
    """Returns a fixed transcript and counts invocations."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.calls = 0
        self.paths: list[Path] = []

    def __call__(self, audio_path: Path) -> Transcript:
        self.calls += 1
        self.paths.append(Path(audio_path))
        return self.transcript


class FakeSeparator:
    """Splits audio into the same speech plus a constant background tone."""

    name = "fake"

    def __init__(self, background_amplitude: float = 0.05) -> None:
        self.background_amplitude = background_amplitude
        self.calls = 0

    def separate(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        t = np.arange(data.size, dtype=np.float64) / float(sample_rate)
        background = (
            self.background_amplitude * np.sin(2 * np.pi * 440.0 * t)
        ).astype(np.float32)
        return data, background


class RoomToneSeparator:
    """A separator whose speech stem has real quiet gaps, like a real recording.

    ``FakeSeparator`` hands back the input unchanged as the speech stem, i.e. a
    constant tone. The room-tone floor gates on level, so a stem with no quiet
    frames correctly yields nothing — right, but useless for testing the floor.
    This models the structure the floor is built from: loud utterances with
    near-silent room tone between them, and a silent bed by default so a test can
    measure the floor without the music masking it.
    """

    name = "roomtone"

    #: Where the fake transcript speaks (matches ``simple_transcript``).
    SPEECH = ((0.0, 2.0), (2.5, 4.5), (5.0, 8.0))

    def __init__(
        self, room_tone_amplitude: float = 0.001, background_amplitude: float = 0.0
    ) -> None:
        # ~-43 dB below the speech burst. Real room tone sits 40-60 dB down, which
        # is why the level gate keeps it and rejects everything louder.
        self.room_tone_amplitude = room_tone_amplitude
        self.background_amplitude = background_amplitude
        self.calls = 0

    def separate(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        t = np.arange(data.size, dtype=np.float64) / float(sample_rate)
        rng = np.random.default_rng(0)
        room = self.room_tone_amplitude * rng.standard_normal(data.size)
        bursts = np.zeros(data.size, dtype=np.float32)
        for start, end in self.SPEECH:
            bursts[int(start * sample_rate) : int(end * sample_rate)] = 1.0
        stem = (room + data * bursts).astype(np.float32)
        background = (
            self.background_amplitude * np.sin(2 * np.pi * 440.0 * t)
        ).astype(np.float32)
        return stem, background


class FakeDiarizer:
    """Assigns scripted speakers, so per-speaker voices are assertable."""

    name = "fake"

    def __init__(self, labels: Sequence[str], overlap_ids: Sequence[int] = ()) -> None:
        self.labels = list(labels)
        #: Segments the fake reports as carrying a second voice.
        self.overlap_ids = set(overlap_ids)
        self.calls = 0
        self.overlap_calls = 0

    def label(
        self,
        segment_ids: Sequence[int],
        regions: Sequence[tuple[float, float]],
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[int, str]:
        self.calls += 1
        # Cycled rather than zipped, so a test only has to state the pattern.
        return {
            segment_id: self.labels[index % len(self.labels)]
            for index, segment_id in enumerate(segment_ids)
        }

    def overlaps(
        self,
        segment_ids: Sequence[int],
        regions: Sequence[tuple[float, float]],
        labels: dict[int, str],
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[int, bool]:
        self.overlap_calls += 1
        return {
            segment_id: segment_id in self.overlap_ids
            for segment_id in segment_ids
        }


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
        self.hybrid: dict[str, Any] | None = None
        #: Every prompt built, in order, so the two-stage sequence is assertable.
        self.prompt_history: list[str] = []
        self.file_prompt: tuple[Any, str] | None = None
        #: Every file prompt built, in order, so per-speaker voices are assertable.
        self.file_prompts: list[tuple[Any, str]] = []
        self.blend_calls = 0
        self.blend_takes: int | None = None
        self.blend_iters: int | None = None
        self.blend_target_ratio: float | None = None

    def prepare_prompt(self, donor: Any) -> None:
        self.prompt_kind = "donor"
        self.prompt_calls += 1
        self.prompt_history.append("donor")

    def prepare_prompt_from_file(self, ref_audio: Path | str, ref_text: str) -> None:
        self.prompt_kind = "file"
        self.prompt_calls += 1
        self.prompt_history.append("file")
        self.file_prompt = (ref_audio, ref_text)
        self.file_prompts.append((ref_audio, ref_text))

    def prepare_hybrid_prompt(
        self,
        *,
        accent_audio: Path | str,
        accent_text: str,
        timbre: Any = None,
        timbre_audio: Path | str | None = None,
    ) -> None:
        self.prompt_kind = "hybrid"
        self.prompt_calls += 1
        self.prompt_history.append("hybrid")
        self.hybrid = {
            "accent_audio": accent_audio,
            "accent_text": accent_text,
            "timbre": timbre,
            "timbre_audio": timbre_audio,
        }

    def build_blended_reference(
        self,
        *,
        accent_audio: Path | str,
        accent_text: str,
        language: str,
        timbre: Any = None,
        timbre_audio: Path | str | None = None,
        takes: int = 1,
        iters: int = 1,
        ratio: float = 0.0,
    ) -> BlendedReference:
        """Stage 1: goes through the hybrid prompt, then returns one far-too-short clip."""
        self.prepare_hybrid_prompt(
            accent_audio=accent_audio,
            accent_text=accent_text,
            timbre=timbre,
            timbre_audio=timbre_audio,
        )
        self.blend_calls += 1
        self.blend_takes = takes
        self.blend_iters = iters
        self.blend_target_ratio = ratio
        return BlendedReference(
            samples=np.zeros(int(0.5 * self.sample_rate), dtype=np.float32),
            sample_rate=self.sample_rate,
            text=accent_text,
            takes=takes,
            iters=iters,
            donor_ratio=0.8,
            margin=0.25,
            timbre_similarity=0.95,
            accent_similarity=0.70,
        )

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
    separator: Any | None = None,
    labeler: Any | None = None,
    speaker_separator: Any | None = None,
    stream_transcriber: Any | None = None,
    **job_kwargs: Any,
) -> tuple[Pipeline, FakeTranscriber, FakeTranslator, FakeTTSEngine]:
    audio = job_kwargs.pop("input_path", None) or write_input(tmp_path)
    output_dir = job_kwargs.pop("output_dir", None) or (tmp_path / "output")
    job = JobConfig(audio=audio, output_dir=output_dir, **job_kwargs)
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
        separator=separator,
        # Named apart from JobConfig.diarizer, which is the *preference* string.
        diarizer=labeler,
        speaker_separator=speaker_separator,
        stream_transcriber=stream_transcriber,
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

    def test_report_loudness_matches_the_source_by_default(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """The dub should sit at the level of the track it replaces."""
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        result = pipeline.run()
        report = result.report

        assert report["loudness_policy"] == "match-source"
        assert report["loudness_source_lufs"] is not None
        # Target is the source's own measured level.
        assert report["loudness_target_lufs"] == pytest.approx(
            report["loudness_source_lufs"], abs=0.01
        )

        samples, sample_rate = media.load_audio(result.dub_track)
        actual = media.measure_loudness(samples, sample_rate)
        assert actual is not None
        # The reported figure describes the delivered file...
        assert report["loudness_dub_lufs"] == pytest.approx(actual, abs=0.05)
        # ...and the delivered file is not quieter than the source.
        assert report["loudness_delta_vs_source"] == pytest.approx(0.0, abs=1.0)

    def test_absolute_loudness_policy_is_honoured(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            loudness_target="lufs",
            target_lufs=-30.0,
        )
        report = pipeline.run().report

        assert report["loudness_policy"] == "lufs"
        assert report["loudness_target_lufs"] == -30.0
        assert report["loudness_dub_lufs"] == pytest.approx(-30.0, abs=1.0)

    def test_dub_does_not_clip(self, tmp_path: Path, simple_transcript: Transcript) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        samples, _ = media.load_audio(pipeline.run().dub_track)
        assert float(np.max(np.abs(samples))) <= 1.0

    def test_accent_mode_is_reported_and_forwarded(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        tts = FakeTTSEngine()
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, tts=tts, accent_mode="speaker-only"
        )
        report = pipeline.run().report
        assert report["accent_mode"] == "speaker-only"

    def test_default_accent_mode_preserves_the_source_accent(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        assert pipeline.run().report["accent_mode"] == "preserve"

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


class TestReferenceAccentMode:
    """Timbre and accent come from different clips in this mode."""

    def build(self, tmp_path: Path, transcript: Transcript, **kwargs: Any):
        accent = write_input(tmp_path, seconds=3.0, name="accent.wav")
        pipeline, _, _, tts = make_pipeline(
            tmp_path,
            transcript,
            accent_mode="reference-accent",
            accent_audio=accent,
            accent_text="Bonjour, je suis Joel.",
            **kwargs,
        )
        return pipeline, tts, accent

    def test_hybrid_prompt_is_stage_one_and_the_blend_is_the_final_reference(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, tts, accent = self.build(tmp_path, simple_transcript)
        pipeline.run()

        # Stage 1 builds the hybrid prompt; stage 2 clones from the clip it produced.
        assert tts.prompt_history == ["hybrid", "file"]
        assert tts.hybrid is not None
        assert tts.hybrid["accent_audio"] == accent
        assert tts.hybrid["accent_text"] == "Bonjour, je suis Joel."
        # Timbre fell back to an auto-selected donor from the source speech.
        assert tts.hybrid["timbre"] is not None
        assert tts.hybrid["timbre_audio"] is None

        assert tts.file_prompt is not None
        blend_path, blend_text = tts.file_prompt
        assert Path(blend_path).name == "reference_blend.wav"
        assert Path(blend_path).is_file()
        # The blend's transcript is the accent transcript, so codes and embedding
        # are derived from the same audio.
        assert blend_text == "Bonjour, je suis Joel."

    def test_explicit_reference_audio_becomes_the_timbre_source(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        ref = write_input(tmp_path, seconds=4.0, name="ref.wav")
        pipeline, tts, _ = self.build(tmp_path, simple_transcript, ref_audio=ref)
        pipeline.run()

        assert tts.prompt_history == ["hybrid", "file"]
        assert tts.hybrid is not None
        # No donor selection happened, so no clip object, just the file path.
        assert tts.hybrid["timbre"] is None
        assert tts.hybrid["timbre_audio"] == ref

    def test_report_names_both_clips(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, accent = self.build(tmp_path, simple_transcript)
        report = pipeline.run().report

        assert report["accent_mode"] == "reference-accent"
        assert report["donor"]["source"] == "auto"
        assert report["donor"]["accent"]["path"] == str(accent)
        assert report["donor"]["accent"]["transcript_provided"] is True

    def test_report_names_the_timbre_clip_when_given(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        ref = write_input(tmp_path, seconds=4.0, name="ref.wav")
        pipeline, _, accent = self.build(tmp_path, simple_transcript, ref_audio=ref)
        report = pipeline.run().report

        assert report["donor"]["source"] == "reference-audio"
        assert report["donor"]["path"] == str(ref)
        assert report["donor"]["accent"]["path"] == str(accent)

    def test_cached_run_still_reports_the_accent_clip(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """A second run reuses 04_dubbed, so donor.json is not rewritten."""
        pipeline, _, accent = self.build(tmp_path, simple_transcript)
        pipeline.run()
        rerun_report = pipeline.run().report

        assert rerun_report["donor"]["accent"]["path"] == str(accent)
        # The blend record is cached with the audio it describes, so it survives too.
        assert rerun_report["donor"]["blend"]["takes"] == 3

    def test_blend_is_rendered_once_and_reused_from_cache(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, tts, _ = self.build(tmp_path, simple_transcript)
        pipeline.run()
        blend = pipeline.job_dir / "04_dubbed" / "reference_blend.wav"
        first = blend.read_bytes()

        pipeline.run()  # every stage should now be served from the manifest

        assert tts.blend_calls == 1
        assert blend.read_bytes() == first

    def test_blend_takes_are_forwarded_to_stage_one(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, tts, _ = self.build(tmp_path, simple_transcript, blend_takes=5)
        pipeline.run()

        assert tts.blend_takes == 5

    def test_blend_metadata_is_recorded(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _ = self.build(tmp_path, simple_transcript)
        report = pipeline.run().report

        blend = report["donor"]["blend"]
        assert blend["strategy"] == "two-stage"
        assert blend["takes"] == 3  # the default
        assert blend["iters"] == 4
        assert blend["target_ratio"] == 0.5
        assert blend["donor_ratio"] == 0.8
        assert blend["timbre_similarity"] == 0.95
        assert blend["accent_similarity"] == 0.70
        assert blend["margin"] == 0.25

    def test_iteration_knobs_reach_the_engine(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, tts, _ = self.build(
            tmp_path, simple_transcript, blend_iters=2, blend_ratio=0.7
        )
        report = pipeline.run().report

        assert tts.blend_iters == 2
        assert tts.blend_target_ratio == 0.7
        assert report["donor"]["blend"]["iters"] == 2
        assert report["donor"]["blend"]["target_ratio"] == 0.7

    def test_single_stage_conditions_every_line_on_both_sources(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, tts, _ = self.build(
            tmp_path, simple_transcript, accent_blend="single-stage"
        )
        report = pipeline.run().report

        # No blend clip at all: the hybrid prompt is the only prompt built.
        assert tts.prompt_history == ["hybrid"]
        assert tts.blend_calls == 0
        assert not (pipeline.job_dir / "04_dubbed" / "reference_blend.wav").exists()
        assert report["donor"]["blend"] == {"strategy": "single-stage"}

    def test_other_modes_keep_a_flat_donor_record(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        donor = pipeline.run().report["donor"]
        assert "accent" not in donor


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


class TestStaleSegmentClips:
    """A rerun must not leave clips from an earlier, differently-split run."""

    def test_orphaned_clips_are_removed(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        dub_dir = pipeline.job_dir / "04_dubbed"
        dub_dir.mkdir(parents=True, exist_ok=True)
        # As if a previous run had segmented the input into more lines.
        orphan = dub_dir / "seg_0099.wav"
        orphan.write_bytes(b"stale")

        pipeline.run()

        assert not orphan.exists()
        assert (dub_dir / "seg_0000.wav").is_file()


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


class TestBackgroundPreservation:
    """The original music and ambience must survive the dub."""

    def test_stems_are_written_and_reported(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        separator = FakeSeparator()
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, separator=separator)
        report = pipeline.run().report

        assert separator.calls == 1
        assert (pipeline.job_dir / "01b_speech.wav").is_file()
        assert (pipeline.job_dir / "01c_background.wav").is_file()
        assert report["background_preserved"] is True
        assert report["duck_db"] == 6.0

    def test_asr_and_donor_use_the_speech_stem(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, transcriber, _, _ = make_pipeline(
            tmp_path, simple_transcript, separator=FakeSeparator()
        )
        pipeline.run()
        assert transcriber.paths[-1].name == "01b_speech.wav"

    def test_background_audio_reaches_the_output(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """Compare with and without a bed: the gaps must differ.

        The two runs use separate output directories; sharing one would let the
        second reuse the first's cached mix (caching is keyed on the job
        fingerprint, which does not include an injected separator).
        """
        silent_pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, output_dir=tmp_path / "out_silent"
        )
        silent_result = silent_pipeline.run()
        silent_samples, sample_rate = media.load_audio(silent_result.dub_track)

        bedded_pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            separator=FakeSeparator(background_amplitude=0.05),
            output_dir=tmp_path / "out_bed",
        )
        bedded_result = bedded_pipeline.run()
        bedded_samples, _ = media.load_audio(bedded_result.dub_track)

        # After the last segment (8s) and before the end of the 10s input.
        window = slice(int(8.6 * sample_rate), int(9.6 * sample_rate))
        quiet = float(np.sqrt(np.mean(np.square(silent_samples[window], dtype=np.float64))))
        with_bed = float(np.sqrt(np.mean(np.square(bedded_samples[window], dtype=np.float64))))

        assert quiet < 1e-4  # nothing there without a bed
        assert with_bed > quiet * 5  # audible ambience with one

    def test_disabled_background_keeps_the_previous_behaviour(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, transcriber, _, _ = make_pipeline(
            tmp_path, simple_transcript, keep_background=False, separator=FakeSeparator()
        )
        report = pipeline.run().report

        assert report["background_preserved"] is False
        assert report["duck_db"] is None
        assert not (pipeline.job_dir / "01c_background.wav").exists()
        # Without separation, ASR reads the plain extracted audio.
        assert transcriber.paths[-1].name == "01_audio.wav"

    def test_duck_depth_is_reported(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, separator=FakeSeparator(), duck_db=12.0
        )
        assert pipeline.run().report["duck_db"] == 12.0

    def test_background_gain_lifts_the_bed(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """The static offset still reaches the output after the bed was reworked.

        The gain is applied to the bed when it is assembled (so a room-tone floor
        keeps its own level), which is a different code path from passing it
        through ``duck_and_mix``.
        """

        def gap_rms(gain_db: float, out: Path) -> float:
            pipeline, _, _, _ = make_pipeline(
                tmp_path,
                simple_transcript,
                separator=FakeSeparator(),
                background_gain_db=gain_db,
                output_dir=out,
            )
            samples, sample_rate = media.load_audio(pipeline.run().dub_track)
            return rms(samples, slice(int(8.6 * sample_rate), int(9.6 * sample_rate)))

        quiet = gap_rms(0.0, tmp_path / "out_gain0")
        loud = gap_rms(6.0, tmp_path / "out_gain6")

        assert quiet > 0.0
        assert loud > quiet * 1.5

    def test_output_still_does_not_clip(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, separator=FakeSeparator(background_amplitude=0.9)
        )
        samples, _ = media.load_audio(pipeline.run().dub_track)
        assert float(np.max(np.abs(samples))) <= 1.0

    def test_output_length_is_unchanged_by_the_bed(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, separator=FakeSeparator()
        )
        samples, sample_rate = media.load_audio(pipeline.run().dub_track)
        assert len(samples) / sample_rate == pytest.approx(10.0, abs=0.1)


class TestRoomMatching:
    """The dub can be placed in the source's own acoustic space (P3d)."""

    def test_off_is_the_default_and_leaves_the_report_unchanged(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, separator=FakeSeparator()
        )
        report = pipeline.run().report

        assert "room" not in report  # dry runs report exactly as they did before
        assert report["duck_db"] == 6.0
        assert not (pipeline.job_dir / "05_room.json").exists()

    def test_match_applies_and_is_reported(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, separator=FakeSeparator(), room="match"
        )
        report = pipeline.run().report

        room = report["room"]
        assert room["mode"] == "match"
        assert room["applied"] is True
        assert room["reverb"] is False  # opt-in, and off unless asked for
        assert RT60_BOUNDS[0] <= room["rt60_s"] <= RT60_BOUNDS[1]
        assert (pipeline.job_dir / "05_room.json").is_file()

    def test_reverb_is_opt_in(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """Two room-matched runs differing only in wet level: the tail is the delta."""

        def run(wet_db: float, out: Path) -> tuple[float, bool]:
            pipeline, _, _, _ = make_pipeline(
                tmp_path,
                simple_transcript,
                separator=RoomToneSeparator(),
                room="match",
                room_wet_db=wet_db,
                # No room-tone floor in either run, so only the reverb can differ.
                room_tone_gain_db=-60.0,
                # A constant-tone stem has no measurable decay, so pin the RT60 to
                # something long enough to still be ringing in the measured pause.
                room_rt60=2.0,
                output_dir=out,
            )
            result = pipeline.run()
            samples, sample_rate = media.load_audio(result.dub_track)
            pause = rms(samples, slice(int(8.6 * sample_rate), int(9.6 * sample_rate)))
            return pause, result.report["room"]["reverb"]

        dry_pause, dry_flag = run(0.0, tmp_path / "out_wet0")
        wet_pause, wet_flag = run(-6.0, tmp_path / "out_wet6")

        assert dry_flag is False
        assert wet_flag is True
        assert wet_pause > dry_pause

    def test_match_ducks_the_bed_more_gently(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            separator=FakeSeparator(),
            room="match",
            duck_db=12.0,
        )
        report = pipeline.run().report

        # The report states the depth actually used, not the flag that was passed.
        assert report["duck_db"] == 3.0
        assert report["room"]["duck_db"] == 3.0

    def test_explicit_rt60_overrides_the_measurement(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            separator=FakeSeparator(),
            room="match",
            room_rt60=1.5,
        )
        assert pipeline.run().report["room"]["rt60_s"] == pytest.approx(1.5)

    def test_match_without_a_speech_stem_degrades_to_dry(
        self,
        tmp_path: Path,
        simple_transcript: Transcript,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No separation means no stem, so the dub is delivered dry - and says so."""
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, room="match")
        with caplog.at_level(logging.WARNING):
            result = pipeline.run()
        report = result.report

        assert report["room"]["applied"] is False
        assert report["room"]["rt60_s"] is None
        assert report["duck_db"] is None  # no bed either
        assert not (pipeline.job_dir / "01b_speech.wav").exists()
        assert "Room matching needs the separated speech stem" in caplog.text
        # Degrading must still deliver a usable dub.
        samples, _ = media.load_audio(result.dub_track)
        assert float(np.max(np.abs(samples))) > 0.01

    def test_disabling_the_background_also_disables_room_matching(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, keep_background=False, room="match"
        )
        assert pipeline.run().report["room"]["applied"] is False

    def test_output_length_is_unchanged(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, separator=FakeSeparator(), room="match"
        )
        samples, sample_rate = media.load_audio(pipeline.run().dub_track)
        assert len(samples) / sample_rate == pytest.approx(10.0, abs=0.1)

    def test_output_still_does_not_clip(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            separator=FakeSeparator(background_amplitude=0.9),
            room="match",
            room_wet_db=-6.0,
        )
        samples, _ = media.load_audio(pipeline.run().dub_track)
        assert float(np.max(np.abs(samples))) <= 1.0

    def test_room_tone_floor_reaches_the_output(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """The floor comes from the stem's quiet gaps, so it must be audible there.

        Both runs use the same separator and a silent bed, so the only difference
        is the room-tone floor itself - not the bed, which is what the previous
        version of this test accidentally measured.
        """
        dry_pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            separator=RoomToneSeparator(),
            output_dir=tmp_path / "out_tone_dry",
        )
        dry_samples, sample_rate = media.load_audio(dry_pipeline.run().dub_track)

        roomed_pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            separator=RoomToneSeparator(),
            room="match",
            output_dir=tmp_path / "out_tone_roomed",
        )
        roomed_samples, _ = media.load_audio(roomed_pipeline.run().dub_track)

        # After the last segment (8s) and before the end of the 10s input.
        window = slice(int(8.6 * sample_rate), int(9.6 * sample_rate))
        quiet = rms(dry_samples, window)
        roomed = rms(roomed_samples, window)

        assert quiet < 1e-5  # nothing there without room matching
        assert roomed > quiet * 5  # the floor is placed there
        assert roomed > 1e-6  # ...and it is not numerical noise

        # After the last segment (8s) and before the end of the 10s input.
        window = slice(int(8.6 * sample_rate), int(9.6 * sample_rate))
        quiet = rms(dry_samples, window)
        roomed = rms(roomed_samples, window)
        assert quiet < 1e-4
        assert roomed > quiet * 5

    def test_the_dub_carries_a_reverb_tail(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """With the wet level up, the track rings on after a line ends."""
        def pause(out: Path, wet_db: float) -> float:
            pipeline, _, _, _ = make_pipeline(
                tmp_path,
                simple_transcript,
                separator=RoomToneSeparator(),
                room="match",
                room_wet_db=wet_db,
                room_rt60=2.0,
                room_tone_gain_db=-60.0,
                output_dir=out,
            )
            samples, sample_rate = media.load_audio(pipeline.run().dub_track)
            return rms(samples, slice(int(8.2 * sample_rate), int(9.5 * sample_rate)))

        # Both runs keep the (very quiet, -60 dB) floor, so the difference is the
        # reverb tail: without it the pause holds nothing else.
        off = pause(tmp_path / "out_tail_off", 0.0)
        on = pause(tmp_path / "out_tail_on", -6.0)

        assert on > off * 3


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

# --------------------------------------------------------------------------- #
# Length repair (speech that a translation makes too long for its slot)
# --------------------------------------------------------------------------- #


def refine_transcript() -> Transcript:
    """Line 0 is far too long for its slot; line 1 is the final line.

    Segment 0 spans 2.0s and is followed by a 0.5s pause, so it may occupy 2.5s
    before it would desynchronise the next line at 2.5s.
    """
    return Transcript(
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="A long English sentence."),
            Segment(id=1, start=2.5, end=4.5, text="Short."),
        ],
    )


class ScriptedTTSEngine(FakeTTSEngine):
    """Returns one scripted row of durations per ``synthesize`` call.

    Row 1 is the initial pass over every line; later rows serve the re-synthesis
    of just the lines that needed rewriting.
    """

    def __init__(self, rows: Sequence[Sequence[float]], *, sample_rate: int = 24_000) -> None:
        super().__init__(sample_rate)
        self._rows = [list(row) for row in rows]
        self._row = 0
        self.text_batches: list[list[str]] = []

    def _tone(self, seconds: float) -> np.ndarray:
        count = max(1, int(seconds * self.sample_rate))
        t = np.linspace(0.0, seconds, count, endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    def synthesize(self, texts: Sequence[str], *, language: str) -> list[np.ndarray | None]:
        self.synth_calls += 1
        self.text_batches.append(list(texts))
        row = self._rows[min(self._row, len(self._rows) - 1)]
        self._row += 1
        return [self._tone(seconds) for seconds in row[: len(texts)]]


class ShorteningTranslator(FakeTranslator):
    """A translator that can also rewrite lines that proved too long."""

    def __init__(
        self,
        *,
        shortenings: dict[int, str] | None = None,
        fail: bool = False,
        blank_ids: Sequence[int] = (),
    ) -> None:
        super().__init__(blank_ids)
        self.shortenings = shortenings or {}
        self.fail = fail
        self.shorten_calls = 0
        self.requested: list[ShortenRequest] = []

    def shorten(
        self, requests: Sequence[ShortenRequest], *, target_language: str
    ) -> dict[int, str]:
        self.shorten_calls += 1
        self.requested = list(requests)
        if self.fail:
            raise TranslationError("translator unreachable")
        return {
            request.id: self.shortenings.get(request.id, "Bref.")
            for request in requests
        }


def refine_pipeline(
    tmp_path: Path,
    *,
    rows: Sequence[Sequence[float]],
    translator: Any | None = None,
    refine: bool = True,
    output_dir: Path | None = None,
) -> tuple[Pipeline, Any, ScriptedTTSEngine, ShorteningTranslator]:
    """Build a pipeline whose TTS durations are scripted, so overflow is exact."""
    tts = ScriptedTTSEngine(rows)
    translator = translator or ShorteningTranslator(shortenings={0: "Bref."})
    pipeline, _, _, _ = make_pipeline(
        tmp_path,
        refine_transcript(),
        translator=translator,
        tts=tts,
        refine=refine,
        output_dir=output_dir or (tmp_path / "output"),
    )
    return pipeline, translator, tts, translator


class TestLengthRepair:
    """Only overlong lines are re-worded; everything else is left alone."""

    def test_only_the_overlong_line_is_rewritten(self, tmp_path: Path) -> None:
        # Line 0 at 3.5s needs 2.98s after the clamp but has 2.5s; line 1 is last
        # and unconstrained, so it must never be sent for rewriting.
        pipeline, translator, tts, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [1.8]])
        report = pipeline.run().report

        assert translator.shorten_calls == 1
        assert [r.id for r in translator.requested] == [0]
        assert report["refine"]["attempted"] is True
        assert report["refine"]["offenders"] == 1
        assert report["refine"]["improved"] == 1
        assert report["refine"]["llm_calls"] == 1
        assert tts.synth_calls == 2

    def test_the_second_pass_only_resynthesises_the_offender(self, tmp_path: Path) -> None:
        pipeline, _, tts, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [1.8]])
        pipeline.run()
        assert tts.text_batches[0] == [
            "[French] A long English sentence.",
            "[French] Short.",
        ]
        assert tts.text_batches[1] == ["Bref."]

    def test_the_fitted_result_loses_no_speech(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [1.8]])
        report = pipeline.run().report
        assert report["trimmed_seconds_total"] == 0.0
        assert report["refine"]["seconds_recovered"] > 0.4

    def test_without_refine_the_line_is_trimmed_as_before(self, tmp_path: Path) -> None:
        pipeline, translator, tts, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], refine=False
        )
        report = pipeline.run().report

        assert translator.shorten_calls == 0
        assert tts.synth_calls == 1
        assert report["refine"]["attempted"] is False
        # The old behaviour: the surplus beyond slot + gap is cut.
        assert report["trimmed_seconds_total"] == pytest.approx(0.475, abs=0.01)

    def test_lines_that_fit_are_byte_identical_with_and_without_refine(
        self, tmp_path: Path
    ) -> None:
        """The core safety property: repairing one line cannot disturb another."""
        rows = [[3.5, 1.0], [1.8]]
        plain, _, _, _ = refine_pipeline(
            tmp_path, rows=rows, refine=False, output_dir=tmp_path / "plain"
        )
        plain.run()
        repaired, _, _, _ = refine_pipeline(
            tmp_path, rows=rows, refine=True, output_dir=tmp_path / "repaired"
        )
        repaired.run()

        for job in ("plain", "repaired"):
            assert (tmp_path / job / "input" / "04_dubbed" / "seg_0001.wav").is_file()
        before = (tmp_path / "plain" / "input" / "04_dubbed" / "seg_0001.wav").read_bytes()
        after = (tmp_path / "repaired" / "input" / "04_dubbed" / "seg_0001.wav").read_bytes()
        assert before == after

    def test_a_rewrite_that_does_not_help_is_discarded(self, tmp_path: Path) -> None:
        # The retry comes back longer than what we already had.
        pipeline, translator, _, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [4.0]])
        report = pipeline.run().report

        assert translator.shorten_calls == 1
        assert report["refine"]["improved"] == 0
        samples, rate = media.load_audio(
            pipeline.job_dir / "04_dubbed" / "seg_0000.wav"
        )
        assert len(samples) / rate == pytest.approx(3.5, abs=0.02)
        assert report["trimmed_seconds_total"] == pytest.approx(0.475, abs=0.01)

    def test_a_partial_improvement_is_kept(self, tmp_path: Path) -> None:
        # 3.5s -> 3.0s still overflows (2.55s > 2.5s) but by less, so it is kept.
        pipeline, _, _, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [3.0]])
        report = pipeline.run().report

        assert report["refine"]["improved"] == 1
        assert 0.0 < report["trimmed_seconds_total"] < 0.475

    def test_a_failing_rewrite_is_not_fatal(self, tmp_path: Path) -> None:
        translator = ShorteningTranslator(fail=True)
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], translator=translator
        )
        report = pipeline.run().report

        assert report["refine"]["attempted"] is True
        assert report["refine"]["improved"] == 0
        assert report["trimmed_seconds_total"] == pytest.approx(0.475, abs=0.01)

    def test_a_translator_without_rewriting_is_handled(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], translator=FakeTranslator()
        )
        report = pipeline.run().report
        assert report["refine"]["attempted"] is False

    def test_the_original_translation_is_kept_for_idempotency(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [1.8]])
        pipeline.run()
        payload = json.loads(
            (pipeline.job_dir / "03_script.json").read_text(encoding="utf-8")
        )
        assert payload["lines"][0]["target"] == "[French] A long English sentence."

    def test_the_rewrite_is_recorded_and_reaches_the_subtitles(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [1.8]])
        result = pipeline.run()

        spoken = json.loads(
            (result.job_dir / "03b_spoken.json").read_text(encoding="utf-8")
        )
        assert spoken == {"0": "Bref."}
        assert result.target_srt is not None
        assert "Bref." in result.target_srt.read_text(encoding="utf-8")

    def test_the_request_carries_a_ceiling_derived_from_measurement(
        self, tmp_path: Path
    ) -> None:
        pipeline, translator, _, _ = refine_pipeline(tmp_path, rows=[[3.5, 1.0], [1.8]])
        pipeline.run()

        request = translator.requested[0]
        assert request.spoken_seconds == pytest.approx(2.98, abs=0.02)
        assert request.available_seconds == pytest.approx(2.5, abs=0.02)
        # 7 words in "[French] A long English sentence." scaled by 2.5/2.98
        assert 0 < request.max_words <= 7


class TestTimingDirectionSplit:
    """The report must separate 'too long' from 'too short'."""

    def test_the_two_directions_are_reported_separately(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], refine=False
        )
        report = pipeline.run().report

        assert report["clamped_segments"] == 2
        assert report["clamped_long_segments"] == 1  # line 0 could not compress
        assert report["clamped_short_segments"] == 1  # line 1 could not be slowed
        assert report["deficit_seconds_total"] > 0.5
        assert report["surplus_seconds_total"] > 0.4

    def test_worst_line_deletion_is_surfaced(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], refine=False
        )
        report = pipeline.run().report
        # 0.475s cut from a 2.98s line.
        assert report["worst_line_deleted_ratio"] == pytest.approx(0.16, abs=0.02)
        assert report["lines_over_10pct_deleted"] == 1

    def test_a_clean_run_reports_no_damage(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(tmp_path, rows=[[1.9, 2.0], [1.8]])
        report = pipeline.run().report
        assert report["trimmed_seconds_total"] == 0.0
        assert report["worst_line_deleted_ratio"] == 0.0
        assert report["lines_over_10pct_deleted"] == 0

    def test_measured_speaking_rate_is_reported(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(tmp_path, rows=[[1.9, 2.0], [1.8]])
        report = pipeline.run().report
        # 7 target words (the fake translator prefixes "[French]") over 3.9s of
        # speech.
        assert report["speech_rate_wps"] == pytest.approx(1.79, abs=0.05)

    def test_timing_json_carries_the_new_fields(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], refine=False
        )
        pipeline.run()
        payload = json.loads(
            (pipeline.job_dir / "timing.json").read_text(encoding="utf-8")
        )
        assert payload[0]["synth_seconds"] == pytest.approx(3.5, abs=0.02)
        assert payload[0]["deficit"] == 0.0
        assert payload[1]["deficit"] > 0.5
