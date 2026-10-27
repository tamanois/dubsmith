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
from pipeline.errors import MediaError, ModelNotFoundError, TranslationError
from pipeline.mixing import RT60_BOUNDS
from pipeline.orchestrate import Pipeline
from pipeline.timing import IdentityStretcher
from pipeline.translate import ShortenRequest, TranslationRequest
from pipeline.tts import MAX_DECODE_SECONDS, BlendedReference

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
        #: The most recent batch of requests, so budgets can be asserted.
        self.requests: list[TranslationRequest] = []

    def translate(
        self,
        requests: Sequence[TranslationRequest],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        self.calls += 1
        self.requests = list(requests)
        return {
            request.id: ("" if request.id in self.blank_ids else f"[{target_language}] {request.text}")
            for request in requests
        }


class FakeTTSEngine:
    """Mimics TTSEngine's surface without loading a model."""

    #: Words per second the delivery-rate probe reports. Scripted rather than
    #: derived, so a probe never consumes a row of the synth script.
    delivery_rate: float = 4.0

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self.batch_size = 8
        self.prompt_kind: str | None = None
        self.prompt_calls = 0
        self.synth_calls = 0
        self.rate_calls = 0
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

    def measure_delivery_rate(self, text: str, *, language: str) -> float:
        self.rate_calls += 1
        return self.delivery_rate

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

    def synthesis_stats(self) -> dict[str, Any]:
        """The real counters, mirrored: the report block is exercised too."""
        return {
            "batch_size": self.batch_size,
            "max_decode_seconds": MAX_DECODE_SECONDS,
            "reference_seconds": None,
            "seconds_per_char": None,
            "batches": self.synth_calls,
            "peak_batch_seconds": 0.0,
            "oom_retries": 0,
            "cache_releases": 0,
            "ceiling_shrinks": 0,
            "peak_reserved_bytes": None,
        }


class FakeVoiceDesignEngine:
    """Mimics VoiceDesignEngine: deterministic audio from a description."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        #: Every (text, instruct, language) requested, so the anchor is assertable.
        self.calls: list[tuple[str, str, str]] = []

    def generate(
        self, text: str, instruct: str, language: str
    ) -> tuple[np.ndarray, int]:
        self.calls.append((text, instruct, language))
        seconds = 2.0
        t = np.linspace(0.0, seconds, int(seconds * self.sample_rate), endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), self.sample_rate


def make_pipeline(
    tmp_path: Path,
    transcript: Transcript,
    *,
    translator: FakeTranslator | None = None,
    tts: FakeTTSEngine | None = None,
    anchor_engine: Any | None = None,
    separator: Any | None = None,
    labeler: Any | None = None,
    speaker_separator: Any | None = None,
    stream_transcriber: Any | None = None,
    **job_kwargs: Any,
) -> tuple[Pipeline, FakeTranscriber, FakeTranslator, FakeTTSEngine]:
    audio = job_kwargs.pop("input_path", None) or write_input(tmp_path)
    output_dir = job_kwargs.pop("output_dir", None) or (tmp_path / "output")
    # Most pipeline tests exercise the long-standing preserve path; the native
    # default is covered by its own tests, which supply an anchor engine.
    job_kwargs.setdefault("accent_mode", "preserve")
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
        anchor_engine=anchor_engine,
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

    def test_preserve_mode_is_reported(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, accent_mode="preserve"
        )
        assert pipeline.run().report["accent_mode"] == "preserve"

    def test_the_device_and_precision_are_recorded(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        """A run on unfamiliar hardware must be diagnosable from the report alone."""
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript, device="cpu")
        device = pipeline.run().report["device"]

        assert set(device) == {
            "requested",
            "resolved",
            "kind",
            "name",
            "total_memory",
            "capability",
            "dtype",
            "asr_compute_type",
        }
        assert device["requested"] == "cpu"
        assert device["resolved"] == "cpu"
        # CPU always runs float32, whatever precision was asked for.
        assert device["dtype"] == "float32"
        assert device["asr_compute_type"] == "int8"

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
        assert blend["iters"] == 5
        assert blend["target_ratio"] == 0.85
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


class TestNativeAccentMode:
    """The default mode borrows a generated anchor's accent, keeping the timbre."""

    def build(self, tmp_path: Path, transcript: Transcript, **kwargs: Any):
        anchor = FakeVoiceDesignEngine()
        pipeline, _, translator, tts = make_pipeline(
            tmp_path,
            transcript,
            accent_mode="native",
            anchor_engine=anchor,
            **kwargs,
        )
        return pipeline, translator, tts, anchor

    def test_a_generated_anchor_supplies_the_accent(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, tts, anchor = self.build(tmp_path, simple_transcript)
        report = pipeline.run().report

        assert report["accent_mode"] == "native"
        assert report["donor"]["accent"]["source"] == "anchor"
        # Generated once, in the target language, from the language's passage.
        assert len(anchor.calls) == 1
        text, _instruct, language = anchor.calls[0]
        assert language == "French"
        assert text.strip()
        # Stage 1 reconciles the anchor with the donor; stage 2 clones from it.
        assert tts.prompt_history == ["hybrid", "file"]
        assert tts.hybrid is not None
        assert Path(tts.hybrid["accent_audio"]).name == "fr.wav"

    def test_the_anchor_is_cached_between_runs(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, anchor = self.build(tmp_path, simple_transcript)
        pipeline.run()
        pipeline.run()
        # The second run reuses the cached anchor instead of regenerating it.
        assert len(anchor.calls) == 1
        assert (pipeline.settings.models_dir / "anchors" / "fr.wav").is_file()

    def test_a_missing_voice_design_model_is_an_actionable_error(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        # No anchor engine injected: the real resolver must fail with a remedy.
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, accent_mode="native"
        )
        with pytest.raises(ModelNotFoundError, match="tts_voice_design"):
            pipeline.run()

    def test_accent_match_shapes_the_generated_anchor(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, anchor = self.build(
            tmp_path, simple_transcript, accent_match="gender"
        )
        report = pipeline.run().report

        assert anchor.calls
        instruct = anchor.calls[0][1]
        assert "voice" in instruct  # the gender phrase reached the description
        record = report["donor"]["accent"]
        assert record["source"] == "anchor-matched"
        assert record["rate_band"] in {"slow", "normal", "fast"}

    def test_the_word_budget_is_sized_from_the_anchor(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, translator, _, _ = self.build(tmp_path, simple_transcript)
        pipeline.run()

        assert translator.requests
        assert all(request.max_words is not None for request in translator.requests)
        assert all(request.max_words >= 1 for request in translator.requests)


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
    """A verbose translation may spill into silence, but never desynchronise.

    With ``max_overrun=0``, which is what "exact sync" means now: the default buys
    a fraction of a second of lateness instead of cutting. See
    :class:`TestOverrunAllowance` for that side of the trade.
    """

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
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, tts=self.LongTTS(), max_overrun=0.0
        )
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
        """With no allowance, every line still starts where the speaker did."""
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, tts=self.LongTTS(), max_overrun=0.0
        )
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

        mux_calls: list[dict[str, Any]] = []

        def fake_mux(
            video_path: Path | str,
            audio: Path | str,
            out: Path | str,
            **kwargs: Any,
        ) -> Path:
            mux_calls.append({"video": Path(video_path), "audio": Path(audio), "out": Path(out), **kwargs})
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"muxed")
            return Path(out)

        monkeypatch.setattr(media, "mux_audio", fake_mux)

        job = JobConfig(video=video, output_dir=tmp_path / "output", accent_mode="preserve")
        pipeline = Pipeline(
            job,
            Settings(models_dir=tmp_path / "models"),
            transcriber=FakeTranscriber(simple_transcript),
            translator=FakeTranslator(),
            tts_engine=FakeTTSEngine(),
            stretcher=IdentityStretcher(),
        )
        return pipeline, mux_calls

    def test_video_input_produces_a_dubbed_video(self, video_job: tuple[Pipeline, list[dict[str, Any]]]) -> None:
        pipeline, _ = video_job
        result = pipeline.run()
        assert result.dubbed_video is not None
        assert result.dubbed_video.exists()
        assert result.report["is_video"] is True

    def test_soft_subtitles_are_embedded_by_default(
        self, video_job: tuple[Pipeline, list[dict[str, Any]]]
    ) -> None:
        pipeline, mux_calls = video_job
        result = pipeline.run()
        assert result.target_srt is not None
        assert mux_calls[-1]["subtitle_srt"] == result.target_srt
        assert mux_calls[-1]["subtitle_language"] == "French"

    def test_burn_mode_produces_a_burned_video(
        self, video_job: tuple[Pipeline, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline, _ = video_job
        pipeline.job.subtitles_mode = "burn"
        burned: list[dict[str, Any]] = []

        def fake_burn(video: Path | str, srt: Path | str, out: Path | str, **kwargs: Any) -> Path:
            burned.append({"video": Path(video), "srt": Path(srt), "out": Path(out), **kwargs})
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"burned")
            return Path(out)

        monkeypatch.setattr(media, "burn_subtitles", fake_burn)
        result = pipeline.run()
        assert result.burned_video is not None and result.burned_video.exists()
        assert burned[-1]["language"] == "French"

    def test_off_mode_does_not_embed(
        self, video_job: tuple[Pipeline, list[dict[str, Any]]]
    ) -> None:
        pipeline, mux_calls = video_job
        pipeline.job.subtitles_mode = "off"
        pipeline.run()
        assert mux_calls[-1]["subtitle_srt"] is None

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
    **job_kwargs: Any,
) -> tuple[Pipeline, Any, ScriptedTTSEngine, ShorteningTranslator]:
    """Build a pipeline whose TTS durations are scripted, so overflow is exact.

    Defaults to ``max_overrun=0`` - no lateness allowance - because the tests built
    on it were written against "fit the window or cut the residue". The allowance
    has its own tests, which pass it explicitly.
    """
    tts = ScriptedTTSEngine(rows)
    translator = translator or ShorteningTranslator(shortenings={0: "Bref."})
    job_kwargs.setdefault("max_overrun", 0.0)
    pipeline, _, _, _ = make_pipeline(
        tmp_path,
        refine_transcript(),
        translator=translator,
        tts=tts,
        refine=refine,
        output_dir=output_dir or (tmp_path / "output"),
        **job_kwargs,
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
        assert report["refine"]["seconds_recovered"] > 0.1

    def test_without_refine_the_residue_beyond_the_floor_is_trimmed(
        self, tmp_path: Path
    ) -> None:
        pipeline, translator, tts, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], refine=False
        )
        report = pipeline.run().report

        assert translator.shorten_calls == 0
        assert tts.synth_calls == 1
        assert report["refine"]["attempted"] is False
        # 3.5s compressed to the 0.75 floor is 2.625s, and the line may use 2.5s:
        # only the 0.125s beyond that is cut. The floor is what limits the damage.
        assert report["trimmed_seconds_total"] == pytest.approx(0.125, abs=0.01)
        assert report["worst_line_deleted_ratio"] == pytest.approx(0.048, abs=0.01)

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
        assert report["trimmed_seconds_total"] == pytest.approx(0.125, abs=0.01)

    def test_a_partial_improvement_is_kept(self, tmp_path: Path) -> None:
        # 4.0s -> 3.6s still overflows (2.7s > 2.5s) but by less, so it is kept.
        pipeline, _, _, _ = refine_pipeline(tmp_path, rows=[[4.0, 1.0], [3.6]])
        report = pipeline.run().report

        assert report["refine"]["improved"] == 1
        assert 0.0 < report["trimmed_seconds_total"] < 0.5

    def test_a_failing_rewrite_is_not_fatal(self, tmp_path: Path) -> None:
        translator = ShorteningTranslator(fail=True)
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], translator=translator
        )
        report = pipeline.run().report

        assert report["refine"]["attempted"] is True
        assert report["refine"]["improved"] == 0
        assert report["trimmed_seconds_total"] == pytest.approx(0.125, abs=0.01)

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
        # The slot would need 2.0/3.5 = 0.57, under the floor, so the floor is what
        # the line gets, and the gap has no room to help: 3.5 * 0.75 = 2.625s.
        assert request.spoken_seconds == pytest.approx(2.625, abs=0.02)
        assert request.available_seconds == pytest.approx(2.5, abs=0.02)
        # 7 words in "[French] A long English sentence." scaled by 2.5/2.625
        assert 0 < request.max_words <= 6


class TestImpossibleAsrTiming:
    """A stretch Whisper squeezed into less time than its words need."""

    @staticmethod
    def packed_transcript() -> Transcript:
        return Transcript(
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=0.20, text="Okay."),
                Segment(id=1, start=0.24, end=0.40, text="What's the next snack?"),
                Segment(id=2, start=0.42, end=0.90, text="I'm getting off track."),
                Segment(id=3, start=2.90, end=4.00, text="And then a normal line."),
            ],
        )

    def test_the_run_is_retimed_and_reported(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, self.packed_transcript())
        report = pipeline.run().report

        assert report["packed"]["runs"] == 1
        assert report["packed"]["lines"] == 2
        assert report["packed"]["worst_wps"] == pytest.approx(25.0, abs=0.1)
        assert report["packed"]["seconds_added"] > 1.0
        # The line after the run keeps its own time: the repair only redistributes
        # the stretch that was impossible, it never shifts a line out of its place.
        timing = json.loads((pipeline.job_dir / "timing.json").read_text())
        assert timing[3]["source_start"] == pytest.approx(2.90, abs=0.01)

    def test_the_wider_slots_reach_the_word_budget(self, tmp_path: Path) -> None:
        """A line that held one word now holds a sentence, which is the point."""
        pipeline, _, translator, _ = make_pipeline(tmp_path, self.packed_transcript())
        pipeline.run()

        budgets = {request.id: request.max_words for request in translator.requests}
        # 0.16s of slot was one word at any rate; ~1.3s is a sentence.
        assert budgets[1] >= 5
        assert budgets[2] >= 5
        # The untouched line before the run keeps its own, smaller ceiling.
        assert budgets[0] < budgets[1]

    def test_an_ordinary_timeline_reports_no_repair(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(tmp_path, simple_transcript)
        assert "packed" not in pipeline.run().report


class TestOverrunAllowance:
    """A line that needs a fraction of a second more keeps its words, and runs late.

    Measured on a real 11-minute job: at a zero allowance 52 lines lost 6.74 s of
    speech; at 0.25 s, four lines lost 0.70 s and the lines that used it started a
    mean 0.12 s late. Lateness is a smaller defect than a deleted word.
    """

    def test_a_stubborn_line_keeps_its_words_and_runs_late(
        self, tmp_path: Path
    ) -> None:
        # 3.6s of speech in a 2.0s slot with a 0.5s pause: at the 0.75 floor the
        # line needs 2.70s, and the allowance lets it use 2.75s, so it is played at
        # 0.76 painlessly rather than losing 0.20s of its tail.
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.6, 1.0], [1.8]], refine=False, max_overrun=0.25
        )
        report = pipeline.run().report

        assert report["max_overrun"] == pytest.approx(0.25)
        assert report["trimmed_seconds_total"] == 0.0
        assert report["shifted_segments"] == 1
        assert report["shifted_seconds_max"] == pytest.approx(0.25, abs=0.02)
        timing = json.loads((pipeline.job_dir / "timing.json").read_text())
        # The line that pays is the *next* one, pushed to the end of the spill.
        assert timing[1]["placed_start"] == pytest.approx(2.75, abs=0.02)

    def test_the_allowance_is_bounded(self, tmp_path: Path) -> None:
        # 4.5s of speech needs 3.375s at the floor: 0.625s past a window it may
        # stretch by only 0.25s, so the rest is still cut.
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[4.5, 1.0], [1.8]], refine=False, max_overrun=0.25
        )
        report = pipeline.run().report

        assert report["trimmed_seconds_total"] == pytest.approx(0.625, abs=0.02)
        assert report["shifted_seconds_max"] == pytest.approx(0.25, abs=0.02)

    def test_zero_allowance_trims_instead(self, tmp_path: Path) -> None:
        """The escape hatch: exact sync, paid for with the tail of the line."""
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.6, 1.0], [1.8]], refine=False, max_overrun=0.0
        )
        report = pipeline.run().report

        assert report["trimmed_seconds_total"] == pytest.approx(0.2, abs=0.02)
        assert report["shifted_segments"] == 0
        assert report["shifted_seconds_max"] == 0.0

    def test_a_line_inside_the_allowance_is_not_an_offender(
        self, tmp_path: Path
    ) -> None:
        """The repair must predict what the mixer will do, not what it used to."""
        pipeline, translator, _, _ = refine_pipeline(
            tmp_path, rows=[[3.6, 1.0], [1.8]], max_overrun=0.25
        )
        report = pipeline.run().report

        assert translator.shorten_calls == 0
        assert report["refine"]["attempted"] is False


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
        # 0.125s cut from a 2.625s line, and no line loses a tenth of itself.
        assert report["worst_line_deleted_ratio"] == pytest.approx(0.048, abs=0.01)
        assert report["lines_over_10pct_deleted"] == 0
        # The floor is what could not absorb it, so the text, not the tempo, owes.
        assert report["needed_below_floor_segments"] == 1
        assert report["compress_floor"] == pytest.approx(0.75)

    def test_the_lower_floor_absorbs_what_used_to_be_cut(self, tmp_path: Path) -> None:
        """2.5s of speech in a 2.0s slot + 0.5s gap needs 0.80: inside the floor."""
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[2.5, 1.0], [1.8]], refine=False
        )
        report = pipeline.run().report

        assert report["trimmed_seconds_total"] == 0.0
        assert report["accelerated_segments"] == 1
        assert report["needed_below_floor_segments"] == 0

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
        # 2.5s available for 3.5s of speech: the floor is applied, not 0.71.
        assert payload[0]["needed_ratio"] == pytest.approx(0.714, abs=0.01)
        assert payload[0]["ratio"] == pytest.approx(0.75, abs=0.001)
        assert "silence_trimmed" in payload[0]
        assert payload[1]["deficit"] > 0.5
