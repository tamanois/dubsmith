"""End-to-end overlap separation through the pipeline, with every heavy part faked.

The labeler reports one segment as carrying two voices, a fake separator returns
two streams, and a fake stream transcriber recovers a different line from each.
What is asserted is the wiring: both voices are translated, synthesised in their
matched speaker's voice, placed *simultaneously* on the timeline, and reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from pipeline.asr import Segment, Transcript

from .test_pipeline import make_pipeline

SR = 48_000

VOICE_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
VOICE_B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def two_speaker_transcript() -> Transcript:
    """Segment 0 is flagged as overlapped; segment 1 is a clean reply."""
    return Transcript(
        language="en",
        language_probability=0.99,
        model="fake",
        segments=[
            Segment(id=0, start=0.0, end=6.0, text="Overlapping start."),
            Segment(id=1, start=6.0, end=10.0, text="A clean reply."),
        ],
    )


class StreamEmbedder:
    """Returns scripted embeddings in call order, one per stream."""

    def __init__(self, vectors: Sequence[np.ndarray]) -> None:
        self._vectors = list(vectors)
        self.calls = 0

    def embed(
        self,
        samples: np.ndarray,
        sample_rate: int,
        regions: Sequence[tuple[float, float]],
    ) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        for _region in regions:
            out.append(self._vectors[self.calls % len(self._vectors)])
            self.calls += 1
        return out


class OverlapLabeler:
    """Labels two speakers, flags segment 0, and supplies centroids + an embedder."""

    name = "overlap-fake"

    def __init__(self, overlap_ids: Sequence[int] = (0,), stream_vectors: Sequence[np.ndarray] = (VOICE_A, VOICE_B)) -> None:
        self.overlap_ids = set(overlap_ids)
        self.embedder = StreamEmbedder(stream_vectors)
        self.centroids_calls = 0
        self.overlap_calls = 0

    def label(
        self,
        segment_ids: Sequence[int],
        regions: Sequence[tuple[float, float]],
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[int, str]:
        return {
            segment_id: "SPK_0" if index % 2 == 0 else "SPK_1"
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
        return {segment_id: segment_id in self.overlap_ids for segment_id in segment_ids}

    def centroids(
        self,
        segment_ids: Sequence[int],
        regions: Sequence[tuple[float, float]],
        labels: dict[int, str],
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[str, np.ndarray]:
        self.centroids_calls += 1
        return {"SPK_0": VOICE_A, "SPK_1": VOICE_B}


class FakeSpeakerSeparator:
    """Splits a region into two streams: the input and a quieter copy."""

    name = "fake-sepformer"

    def __init__(self) -> None:
        self.calls = 0

    def separate(
        self, samples: np.ndarray, sample_rate: int, num_speakers: int = 2
    ) -> list[np.ndarray]:
        self.calls += 1
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        return [data, (data * 0.5).astype(np.float32)]


class FakeStreamTranscriber:
    """Recovers a different line from each separated stream."""

    def __init__(self, mapping: dict[str, Transcript]) -> None:
        self.mapping = mapping
        self.paths: list[Path] = []

    def __call__(self, audio_path: Path) -> Transcript:
        self.paths.append(Path(audio_path))
        return self.mapping[Path(audio_path).name]


def stream_transcriber() -> FakeStreamTranscriber:
    return FakeStreamTranscriber(
        {
            "overlap_0000_0.wav": Transcript(
                language="en",
                segments=[Segment(id=0, start=0.0, end=2.0, text="Overlapping start.")],
            ),
            "overlap_0000_1.wav": Transcript(
                language="en",
                segments=[Segment(id=0, start=1.0, end=4.0, text="And I cut in.")],
            ),
        }
    )


def build(
    tmp_path: Path,
    *,
    labeler: Any = "default",
    stream: FakeStreamTranscriber | None = None,
    speaker_separator: Any | None = None,
    **job_kwargs: Any,
):
    transcript = two_speaker_transcript()
    # A sentinel rather than ``None``, so a test can pass ``labeler=None`` to mean
    # "no real labeler" while the default still builds the overlap one.
    labeler = OverlapLabeler() if labeler == "default" else labeler
    stream = stream or stream_transcriber()
    separator = speaker_separator or FakeSpeakerSeparator()
    job_kwargs.setdefault("overlap_mode", "separate")
    pipeline, _, translator, tts = make_pipeline(
        tmp_path,
        transcript,
        labeler=labeler,
        speaker_separator=separator,
        stream_transcriber=stream,
        **job_kwargs,
    )
    return pipeline, transcript, translator, tts, separator, stream


class TestOverlapSeparationStage:
    """Splitting a flagged segment into one line per voice."""

    def test_the_flagged_segment_is_split_into_two_lines(self, tmp_path: Path) -> None:
        pipeline, *_ = build(tmp_path)
        pipeline.run()

        payload = json.loads(
            (pipeline.job_dir / "02d_overlap_segments.json").read_text(encoding="utf-8")
        )
        assert payload["mode"] == "separate"
        assert payload["separated"] == 1
        records = payload["segments"]
        assert [record["speaker"] for record in records] == ["SPK_0", "SPK_1"]
        assert {record["text"] for record in records} == {
            "Overlapping start.",
            "And I cut in.",
        }

    def test_both_voices_are_translated(self, tmp_path: Path) -> None:
        pipeline, *_ = build(tmp_path)
        result = pipeline.run()

        script = json.loads(
            (result.job_dir / "03_script.json").read_text(encoding="utf-8")
        )
        targets = [line["target"] for line in script["lines"]]
        # Three lines: the two voices of the split segment, plus the clean reply.
        assert len(targets) == 3
        assert "[French] And I cut in." in targets

    def test_both_voices_are_placed_simultaneously(self, tmp_path: Path) -> None:
        pipeline, *_ = build(tmp_path)
        result = pipeline.run()

        timing = json.loads(
            (result.job_dir / "timing.json").read_text(encoding="utf-8")
        )
        assert len(timing) == 3
        # The second voice is an overlay line: at its own time, never pushed late.
        overlay = timing[1]
        assert overlay["source_start"] == pytest.approx(1.0)
        assert overlay["shifted"] is False
        assert result.report["shifted_segments"] == 0

    def test_the_report_records_the_separation(self, tmp_path: Path) -> None:
        pipeline, *_ = build(tmp_path)
        report = pipeline.run().report

        assert report["overlap"]["mode"] == "separate"
        assert report["overlap"]["separated"] == 1
        assert report["overlap"]["lines"] == 2

    def test_the_target_subtitles_include_the_second_voice(self, tmp_path: Path) -> None:
        pipeline, *_ = build(tmp_path)
        result = pipeline.run()

        assert result.target_srt is not None
        text = result.target_srt.read_text(encoding="utf-8")
        assert "And I cut in." in text

    def test_the_second_voice_is_dubbed_by_its_own_speaker(self, tmp_path: Path) -> None:
        pipeline, _transcript, _translator, tts, _separator, _stream = build(tmp_path)
        pipeline.run()

        # Both voices come from the same file, but the prompt *is* the voice: the
        # second line must have been synthesised after its speaker's prompt was
        # (re)built. Two speakers => two distinct prompts.
        assert tts.prompt_calls == 2


class TestSeparationFallsBackSafely:
    """A segment that cannot be split keeps the dominant-only treatment."""

    def test_dominant_mode_leaves_the_line_whole(self, tmp_path: Path) -> None:
        pipeline, *_ = build(tmp_path, overlap_mode="dominant")
        result = pipeline.run()

        assert not (result.job_dir / "02d_overlap_segments.json").exists()
        timing = json.loads(
            (result.job_dir / "timing.json").read_text(encoding="utf-8")
        )
        assert len(timing) == 2
        assert result.report["overlap"]["mode"] == "dominant"

    def test_a_segment_whose_streams_match_one_voice_is_not_split(
        self, tmp_path: Path
    ) -> None:
        # Both streams embed as the same speaker: a likely false-positive flag, so
        # the line is left alone rather than split into two voices.
        labeler = OverlapLabeler(stream_vectors=(VOICE_A, VOICE_A))
        pipeline, *_ = build(tmp_path, labeler=labeler)
        result = pipeline.run()

        payload = json.loads(
            (result.job_dir / "02d_overlap_segments.json").read_text(encoding="utf-8")
        )
        assert payload["segments"] == []
        assert result.report["overlap"]["mode"] == "dominant"

    def test_a_silent_stream_does_not_split_the_segment(self, tmp_path: Path) -> None:
        stream = FakeStreamTranscriber(
            {
                "overlap_0000_0.wav": Transcript(
                    language="en",
                    segments=[
                        Segment(id=0, start=0.0, end=2.0, text="Overlapping start.")
                    ],
                ),
                "overlap_0000_1.wav": Transcript(language="en", segments=[]),
            }
        )
        pipeline, *_ = build(tmp_path, stream=stream)
        result = pipeline.run()

        payload = json.loads(
            (result.job_dir / "02d_overlap_segments.json").read_text(encoding="utf-8")
        )
        assert payload["segments"] == []

    def test_separation_is_skipped_without_a_real_labeler(self, tmp_path: Path) -> None:
        pipeline, *_ = build(tmp_path, labeler=None)
        result = pipeline.run()

        assert not (result.job_dir / "02d_overlap_segments.json").exists()
        # No real labeler means no overlap stage either, so the report omits it.
        assert "overlap" not in result.report

    def test_a_failing_separator_keeps_the_dominant_line(self, tmp_path: Path) -> None:
        class ExplodingSeparator:
            name = "boom"

            def separate(
                self, samples: np.ndarray, sample_rate: int, num_speakers: int = 2
            ) -> list[np.ndarray]:
                raise RuntimeError("model exploded")

        pipeline, *_ = build(tmp_path, speaker_separator=ExplodingSeparator())
        result = pipeline.run()

        # Best-effort: a separator failure must not fail the job, and the line is
        # left as the dominant-only path would have produced it.
        payload = json.loads(
            (result.job_dir / "02d_overlap_segments.json").read_text(encoding="utf-8")
        )
        assert payload["segments"] == []
        assert result.report["overlap"]["mode"] == "dominant"

    def test_a_labeler_that_cannot_match_streams_is_skipped(self, tmp_path: Path) -> None:
        class LabelsOnly:
            """Flags overlap but has no embedder, so streams cannot be matched."""

            name = "labels-only"

            def label(self, segment_ids, regions, samples, sample_rate):
                return {
                    segment_id: ("SPK_0" if index % 2 == 0 else "SPK_1")
                    for index, segment_id in enumerate(segment_ids)
                }

            def overlaps(self, segment_ids, regions, labels, samples, sample_rate):
                return {segment_id: segment_id == 0 for segment_id in segment_ids}

        pipeline, *_ = build(tmp_path, labeler=LabelsOnly())
        result = pipeline.run()

        assert not (result.job_dir / "02d_overlap_segments.json").exists()
        assert result.report["overlap"]["mode"] == "dominant"
