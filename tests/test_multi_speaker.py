"""Multi-speaker dubbing tests: diarization, donor selection, per-speaker voices.

Both the labeler and the TTS engine are injected, so the whole per-speaker path is
exercised without a model, a GPU, or the optional funasr dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from pipeline.asr import Segment, Transcript

from .test_pipeline import FakeDiarizer, FakeTranslator, make_pipeline, write_input

#: Two voices alternating, so each group has more than one member.
ALTERNATING = ("SPK_0", "SPK_1")


def two_speaker_transcript() -> Transcript:
    """Four segments: the odd ones from one voice, the even ones from another."""
    return Transcript(
        language="en",
        language_probability=0.99,
        model="fake",
        segments=[
            Segment(id=0, start=0.0, end=3.0, text="Hello from the first voice."),
            Segment(id=1, start=3.0, end=6.0, text="And a reply from the second."),
            Segment(id=2, start=6.0, end=9.0, text="Back to the first speaker again."),
            Segment(id=3, start=9.0, end=12.0, text="One more from the second voice."),
        ],
    )


def build(
    tmp_path: Path,
    transcript: Transcript | None = None,
    labels: Sequence[str] = ALTERNATING,
    labeler: Any | None = None,
    **job_kwargs: Any,
):
    """A pipeline with a scripted labeler, plus the transcript for assertions.

    The transcriber hands back the transcript object itself and the diarization
    stage annotates it in place, so the returned transcript is what the run saw.
    """
    transcript = transcript if transcript is not None else two_speaker_transcript()
    diarizer = labeler if labeler is not None else FakeDiarizer(labels)
    pipeline, _, _, tts = make_pipeline(
        tmp_path, transcript, labeler=diarizer, **job_kwargs
    )
    return pipeline, diarizer, tts, transcript


class TestDiarizationStage:
    """Assigning and caching speakers."""

    def test_every_segment_is_annotated_with_its_speaker(self, tmp_path: Path) -> None:
        pipeline, _, _, transcript = build(tmp_path)
        pipeline.run()

        assert [segment.speaker for segment in transcript.segments] == [
            "SPK_0",
            "SPK_1",
            "SPK_0",
            "SPK_1",
        ]

    def test_the_labels_are_persisted_for_reuse(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path)
        pipeline.run()

        payload = json.loads(
            (pipeline.job_dir / "02b_speakers.json").read_text(encoding="utf-8")
        )
        assert payload == {"0": "SPK_0", "1": "SPK_1", "2": "SPK_0", "3": "SPK_1"}

    def test_a_second_run_reuses_the_cached_labels(self, tmp_path: Path) -> None:
        pipeline, diarizer, _, _ = build(tmp_path)
        pipeline.run()
        assert diarizer.calls == 1

        pipeline.run()
        # Diarization does not depend on the voice or the translation, so a rerun
        # must not pay for it again.
        assert diarizer.calls == 1

    def test_the_cached_path_still_annotates_the_transcript(self, tmp_path: Path) -> None:
        first, _, _, _ = build(tmp_path)
        first.run()

        # Drop only the transcription, so the run has to label a fresh transcript
        # object and the only labels available are the cached ones. This fails if
        # annotation ever moves inside the diarization stage's build step.
        (first.job_dir / "02_transcript.json").unlink()

        fresh = two_speaker_transcript()
        second, diarizer, _, _ = build(tmp_path, fresh)
        second.run()

        assert diarizer.calls == 0
        assert [segment.speaker for segment in fresh.segments] == [
            "SPK_0",
            "SPK_1",
            "SPK_0",
            "SPK_1",
        ]

    def test_diarization_off_puts_everything_in_one_voice(self, tmp_path: Path) -> None:
        pipeline, diarizer, tts, _ = build(tmp_path, diarizer="none")
        report = pipeline.run().report

        assert diarizer.calls == 0  # an explicit "none" is never overridden
        assert tts.prompt_calls == 1
        assert report["speakers"] is None
        assert report["donor"]["speaker_count"] == 1

    def test_one_speaker_does_not_rebuild_the_prompt(self, tmp_path: Path) -> None:
        pipeline, _, tts, _ = build(tmp_path, labels=("SPK_0",))
        report = pipeline.run().report

        assert tts.prompt_calls == 1
        assert tts.synth_calls == 1
        assert report["speakers"] is None


class TestPerSpeakerDonors:
    """Each voice is cloned from its own segment of the source."""

    def test_each_speaker_gets_its_own_donor(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path)
        report = pipeline.run().report

        speakers = report["donor"]["speakers"]
        assert set(speakers) == {"SPK_0", "SPK_1"}
        # Each donor must come from a segment that speaker actually delivered.
        assert speakers["SPK_0"]["segment_id"] in {0, 2}
        assert speakers["SPK_1"]["segment_id"] in {1, 3}

    def test_the_voice_bible_counts_each_speakers_speech(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path)
        report = pipeline.run().report

        speakers = report["donor"]["speakers"]
        assert speakers["SPK_0"]["segments"] == 2
        assert speakers["SPK_1"]["segments"] == 2
        assert speakers["SPK_0"]["speech_seconds"] == pytest.approx(6.0)

    def test_the_dominant_speaker_is_mirrored_to_the_top_level(
        self, tmp_path: Path
    ) -> None:
        # Everything that already reads donor.accent, donor.source or donor.blend
        # keeps working: the flat record names the speaker who talks most.
        labels = ("SPK_0", "SPK_0", "SPK_0", "SPK_1")
        pipeline, _, _, _ = build(tmp_path, labels=labels)
        report = pipeline.run().report

        assert report["donor"]["primary_speaker"] == "SPK_0"
        assert report["donor"]["segment_id"] == (
            report["donor"]["speakers"]["SPK_0"]["segment_id"]
        )
        assert report["donor"]["source"] == "auto"

    def test_an_explicit_reference_voice_replaces_every_speaker(
        self, tmp_path: Path
    ) -> None:
        ref = write_input(tmp_path, seconds=4.0, name="ref.wav")
        pipeline, _, tts, _ = build(tmp_path, ref_audio=ref, ref_text="some transcript")
        report = pipeline.run().report

        # One prompt, because the same voice is cloned for every line.
        assert tts.prompt_kind == "file"
        assert tts.prompt_calls == 1
        for record in report["donor"]["speakers"].values():
            assert record["source"] == "reference-audio"
            assert record["path"] == str(ref)


class TestPerSpeakerVoices:
    """Grouped synthesis with one blend clip per voice."""

    def accent_pipeline(self, tmp_path: Path, **job_kwargs: Any):
        accent = write_input(tmp_path, seconds=3.0, name="accent.wav")
        return build(
            tmp_path,
            accent_mode="reference-accent",
            accent_audio=accent,
            accent_text="Bonjour, je suis Joel.",
            **job_kwargs,
        )

    def test_one_blend_clip_is_rendered_per_speaker(self, tmp_path: Path) -> None:
        pipeline, _, tts, _ = self.accent_pipeline(tmp_path)
        pipeline.run()

        dubbed = pipeline.job_dir / "04_dubbed"
        assert tts.blend_calls == 2
        assert (dubbed / "reference_blend_SPK_0.wav").is_file()
        assert (dubbed / "reference_blend_SPK_1.wav").is_file()
        # The single-voice filename is not used once the voices are split.
        assert not (dubbed / "reference_blend.wav").exists()

    def test_each_voice_is_cloned_from_its_own_clip(self, tmp_path: Path) -> None:
        pipeline, _, tts, _ = self.accent_pipeline(tmp_path)
        pipeline.run()

        cloned = [Path(path).name for path, _ in tts.file_prompts]
        assert cloned == ["reference_blend_SPK_0.wav", "reference_blend_SPK_1.wav"]

    def test_the_prompt_is_rebuilt_once_per_speaker(self, tmp_path: Path) -> None:
        pipeline, _, tts, _ = self.accent_pipeline(tmp_path)
        pipeline.run()

        # Stage 1 builds a hybrid prompt per speaker, then each speaker's lines are
        # cloned from a file: two voices means four prompts but only two batches.
        assert tts.prompt_history == ["hybrid", "hybrid", "file", "file"]
        assert tts.synth_calls == 2

    def test_consecutive_lines_of_one_speaker_share_a_batch(self, tmp_path: Path) -> None:
        labels = ("SPK_0", "SPK_0", "SPK_1", "SPK_1")
        pipeline, _, tts, _ = build(tmp_path, labels=labels)
        pipeline.run()

        # Two voices, two batches - not four, and not one for the whole video.
        assert tts.synth_calls == 2

    def test_every_segment_still_produces_its_own_clip(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = self.accent_pipeline(tmp_path)
        pipeline.run()

        for segment_id in range(4):
            clip = pipeline.job_dir / "04_dubbed" / f"seg_{segment_id:04d}.wav"
            assert clip.is_file()

    def test_the_flat_blend_record_mirrors_the_primary_speaker(
        self, tmp_path: Path
    ) -> None:
        # Single-speaker jobs read donor.blend.donor_ratio, so the mirror keeps
        # working; the whole per-speaker map lives in blend.json.
        pipeline, _, _, _ = self.accent_pipeline(tmp_path)
        report = pipeline.run().report

        blend = report["donor"]["blend"]
        assert blend["strategy"] == "two-stage"
        assert blend["donor_ratio"] == 0.8

        on_disk = json.loads(
            (pipeline.job_dir / "04_dubbed" / "blend.json").read_text(encoding="utf-8")
        )
        assert set(on_disk) == {"SPK_0", "SPK_1"}
        assert on_disk["SPK_1"]["donor_ratio"] == 0.8

    def test_each_speaker_record_names_its_blend_and_accent(
        self, tmp_path: Path
    ) -> None:
        pipeline, _, _, _ = self.accent_pipeline(tmp_path)
        report = pipeline.run().report

        for name in ("SPK_0", "SPK_1"):
            record = report["donor"]["speakers"][name]
            assert record["accent_mode"] == "reference-accent"
            assert record["accent"]["source"] == "reference-audio"
            assert record["blend"]["strategy"] == "two-stage"

    def test_single_stage_builds_one_hybrid_prompt_per_speaker(
        self, tmp_path: Path
    ) -> None:
        pipeline, _, tts, _ = self.accent_pipeline(tmp_path, accent_blend="single-stage")
        pipeline.run()

        assert tts.blend_calls == 0
        assert tts.prompt_history == ["hybrid", "hybrid"]
        assert not (pipeline.job_dir / "04_dubbed" / "reference_blend.wav").exists()

    def test_a_blank_line_in_a_group_does_not_stop_the_others(
        self, tmp_path: Path
    ) -> None:
        pipeline, _, tts, _ = build(tmp_path, translator=FakeTranslator(blank_ids={1}))
        pipeline.run()

        assert tts.synth_calls == 2
        assert not (pipeline.job_dir / "04_dubbed" / "seg_0001.wav").exists()
        assert (pipeline.job_dir / "04_dubbed" / "seg_0003.wav").is_file()


class TestSpeakerReport:
    """The summary the CLI and the docs read."""

    def test_the_report_lists_each_speaker_and_its_share(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path)
        report = pipeline.run().report

        speakers = report["speakers"]
        assert set(speakers) == {"SPK_0", "SPK_1"}
        assert speakers["SPK_0"]["segments"] == 2
        assert speakers["SPK_1"]["speech_seconds"] == pytest.approx(6.0)
        assert speakers["SPK_0"]["donor_segment"] in {0, 2}

    def test_a_single_voice_reports_no_speakers_block(self, tmp_path: Path) -> None:
        # None rather than a one-entry map, so single-speaker reports stay what
        # they were before diarization existed.
        pipeline, _, _, _ = build(tmp_path, labels=("SPK_0",))
        assert pipeline.run().report["speakers"] is None

    def test_the_report_is_written_to_disk(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path)
        pipeline.run()

        report = json.loads(
            (pipeline.job_dir / "report.json").read_text(encoding="utf-8")
        )
        assert set(report["speakers"]) == {"SPK_0", "SPK_1"}


class TestMultiSpeakerTiming:
    """Grouping must not disturb the timeline."""

    def test_every_segment_still_gets_a_cue(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path)
        result = pipeline.run()

        assert result.target_srt.read_text(encoding="utf-8").count("-->") == 4

    def test_the_timing_stage_sees_all_four_lines(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path)
        pipeline.run()

        timing = json.loads(
            (pipeline.job_dir / "timing.json").read_text(encoding="utf-8")
        )
        assert len(timing) == 4


class TestBackwardCompatibility:
    """A one-voice video must behave exactly as it did before diarization."""

    def test_a_one_voice_run_keeps_a_flat_donor_record(self, tmp_path: Path) -> None:
        pipeline, _, _, _ = build(tmp_path, labels=("SPK_0",))
        report = pipeline.run().report

        assert report["speakers"] is None
        assert report["donor"]["source"] == "auto"
        assert report["donor"]["speaker_count"] == 1
        assert "accent" not in report["donor"]

    def test_one_voice_keeps_the_blend_filename(self, tmp_path: Path) -> None:
        accent = write_input(tmp_path, seconds=3.0, name="accent.wav")
        pipeline, _, _, _ = build(
            tmp_path,
            labels=("SPK_0",),
            accent_mode="reference-accent",
            accent_audio=accent,
            accent_text="Bonjour, je suis Joel.",
        )
        pipeline.run()

        # audit_timbre.py and the docs both point at this name.
        dubbed = pipeline.job_dir / "04_dubbed"
        assert (dubbed / "reference_blend.wav").is_file()
        assert not (dubbed / "reference_blend_SPK_0.wav").exists()


class TestLabelerSelection:
    """Which labeler a run ends up using."""

    def test_an_injected_labeler_is_preferred_over_detection(
        self, tmp_path: Path
    ) -> None:
        # The autouse fixture makes detect_diarizer return a NullDiarizer, so the
        # fake being consulted proves injection takes precedence.
        pipeline, diarizer, _, _ = build(tmp_path, diarizer="auto")
        pipeline.run()
        assert diarizer.calls == 1

    def test_injection_does_not_override_an_explicit_off(self, tmp_path: Path) -> None:
        pipeline, diarizer, _, _ = build(tmp_path, diarizer="none")
        pipeline.run()
        assert diarizer.calls == 0

    def test_a_null_labeler_yields_a_single_voice(self, tmp_path: Path) -> None:
        from pipeline.diarize import NullDiarizer

        pipeline, _, _, _ = build(tmp_path, labeler=NullDiarizer())
        report = pipeline.run().report
        assert report["donor"]["speaker_count"] == 1


class TestUnexpectedSpeakers:
    """Edge cases in the labels a labeler can return."""

    def test_a_rare_voice_still_gets_its_own_batch(self, tmp_path: Path) -> None:
        labels = ("SPK_0", "SPK_0", "SPK_0", "SPK_1")
        pipeline, _, tts, _ = build(tmp_path, labels=labels)
        pipeline.run()

        # The rare voice is a speaker like any other, not a fallback.
        assert tts.synth_calls == 2

    def test_a_large_number_of_voices_gets_one_batch_each(
        self, tmp_path: Path
    ) -> None:
        labels = ("SPK_0", "SPK_1", "SPK_2", "SPK_3")
        pipeline, _, tts, _ = build(tmp_path, labels=labels)
        report = pipeline.run().report

        assert tts.synth_calls == 4
        assert len(report["speakers"]) == 4

    def test_interleaved_speakers_are_counted_once_each(self, tmp_path: Path) -> None:
        labels = ("SPK_1", "SPK_0", "SPK_1", "SPK_0")
        pipeline, _, _, _ = build(tmp_path, labels=labels)
        report = pipeline.run().report

        # Four alternating lines still make exactly two voices of two lines each.
        shares = {
            name: record["segments"] for name, record in report["speakers"].items()
        }
        assert shares == {"SPK_0": 2, "SPK_1": 2}
