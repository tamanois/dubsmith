"""Diarization tests: clustering, label repair, and embedder selection.

The clustering is a pure function over embeddings, so the whole decision surface
is exercised here with synthetic vectors - no audio, no model, no GPU. Only the
funasr wrapper is left to integration.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pytest

from pipeline.diarize import (
    DEFAULT_THRESHOLD,
    MIN_EMBED_S,
    MIN_OVERLAP_S,
    MIN_SPEAKER_S,
    OVERLAP_HOP_S,
    OVERLAP_WINDOW_S,
    SPEAKER_RATE,
    EmbeddingDiarizer,
    FunASREmbedder,
    NullDiarizer,
    _cosine,
    _longest_other_span,
    _window_bounds,
    assign_streams,
    cluster_speakers,
    detect_diarizer,
    single_speaker_labels,
)

#: Unit vectors far apart on the sphere, used as stand-in speaker identities.
VOICE_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
VOICE_B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def jitter(voice: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """A recording of the same voice: close to ``voice``, not identical to it."""
    noise = np.ones_like(voice) * amount
    return voice + noise


class TestClusterSpeakers:
    """Assigning embeddings to speakers."""

    def test_two_voices_are_split(self) -> None:
        labels = cluster_speakers(
            [jitter(VOICE_A), jitter(VOICE_B), jitter(VOICE_A), jitter(VOICE_B)],
            [5.0, 5.0, 5.0, 5.0],
        )
        assert labels[0] == labels[2]
        assert labels[1] == labels[3]
        assert labels[0] != labels[1]
    def test_the_first_voice_is_named_first(self) -> None:
        # SPK_0 must be whoever speaks first, so a rerun numbers the same way.
        labels = cluster_speakers(
            [jitter(VOICE_B), jitter(VOICE_A), jitter(VOICE_B)], [5.0, 5.0, 5.0]
        )
        assert labels == ["SPK_0", "SPK_1", "SPK_0"]

    def test_one_recording_of_one_voice_stays_one_speaker(self) -> None:
        labels = cluster_speakers([jitter(VOICE_A, 0.01) for _ in range(5)], [4.0] * 5)
        assert labels == ["SPK_0"] * 5

    def test_a_single_segment_is_one_speaker(self) -> None:
        assert cluster_speakers([VOICE_A], [10.0]) == ["SPK_0"]

    def test_no_segments_produces_no_labels(self) -> None:
        assert cluster_speakers([], []) == []

    def test_only_unusable_rows_are_blank(self) -> None:
        assert cluster_speakers([None, None], [5.0, 5.0]) == ["", ""]

    @pytest.mark.parametrize(
        "unusable",
        [
            pytest.param(None, id="none"),
            pytest.param(np.zeros(4), id="zero-norm"),
            pytest.param(np.array([1.0, np.nan, 0.0, 0.0]), id="nan"),
            pytest.param(np.array([1.0, np.inf, 0.0, 0.0]), id="inf"),
            pytest.param(np.array([]), id="empty"),
            pytest.param("not a vector", id="wrong-type"),
        ],
    )
    def test_an_unusable_embedding_is_blank(self, unusable: Any) -> None:
        assert cluster_speakers([unusable, VOICE_A], [5.0, 5.0]) == ["", "SPK_0"]

    def test_a_segment_too_short_to_embed_is_blank(self) -> None:
        # Below MIN_EMBED_S the window is mostly noise, so it is not trusted.
        labels = cluster_speakers([VOICE_A, VOICE_A], [5.0, MIN_EMBED_S / 2])
        assert labels == ["SPK_0", ""]

    def test_a_cluster_too_short_to_clone_is_folded_in(self) -> None:
        # A 1 s stray recording is not a third voice: it joins its nearest match,
        # because a speaker with no usable donor window has no voice to clone.
        labels = cluster_speakers(
            [jitter(VOICE_A), jitter(VOICE_B), jitter(VOICE_B)],
            [10.0, 10.0, MIN_SPEAKER_S / 2],
        )
        assert labels == ["SPK_0", "SPK_1", "SPK_1"]

    def test_a_short_cluster_merges_leftwards_without_losing_a_voice(self) -> None:
        # Regression: merging into a lower-indexed neighbour must not drop or
        # duplicate a cluster. The short speaker sits between two long ones.
        labels = cluster_speakers(
            [jitter(VOICE_A), jitter(VOICE_B), jitter(VOICE_A)],
            [10.0, MIN_SPEAKER_S / 2, 10.0],
        )
        assert labels == ["SPK_0", "SPK_0", "SPK_0"]

    def test_a_short_first_cluster_is_absorbed_by_the_next(self) -> None:
        labels = cluster_speakers(
            [jitter(VOICE_B), jitter(VOICE_A), jitter(VOICE_A)],
            [1.0, 10.0, 10.0],
        )
        assert labels == ["SPK_0", "SPK_0", "SPK_0"]

    def test_a_zero_threshold_splits_every_recording(self) -> None:
        labels = cluster_speakers(
            [jitter(VOICE_A, 0.01), jitter(VOICE_A, 0.02)], [10.0, 10.0], threshold=0.0
        )
        assert labels == ["SPK_0", "SPK_1"]

    def test_a_high_threshold_merges_distinct_voices(self) -> None:
        labels = cluster_speakers(
            [jitter(VOICE_A), jitter(VOICE_B)], [10.0, 10.0], threshold=1.0
        )
        assert labels == ["SPK_0", "SPK_0"]

    def test_the_same_embeddings_always_give_the_same_labels(self) -> None:
        embeddings = [jitter(VOICE_A), jitter(VOICE_B)] * 3
        assert cluster_speakers(embeddings, [4.0] * 6) == cluster_speakers(
            embeddings, [4.0] * 6
        )

    def test_labels_are_dense_and_start_at_zero(self) -> None:
        labels = cluster_speakers(
            [jitter(VOICE_A), jitter(VOICE_B), jitter(VOICE_A), jitter(VOICE_B)],
            [5.0] * 4,
        )
        assert sorted(set(labels)) == ["SPK_0", "SPK_1"]

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cluster_speakers([VOICE_A, VOICE_B], [5.0])

    def test_only_clusters_below_the_minimum_are_folded(self) -> None:
        # A zero threshold keeps every recording separate, so each duration forms
        # its own cluster: the 4 s one is a speaker, the 1 s one has no usable
        # donor window and joins it.
        labels = cluster_speakers(
            [jitter(VOICE_A, 0.01), jitter(VOICE_A, 0.02), jitter(VOICE_A, 0.03)],
            [10.0, 4.0, 1.0],
            threshold=0.0,
        )
        assert labels == ["SPK_0", "SPK_1", "SPK_1"]

    def test_defaults_match_the_calibrated_constants(self) -> None:
        assert DEFAULT_THRESHOLD == 0.40
        assert MIN_SPEAKER_S == 3.0


class TestSingleSpeakerLabels:
    """The no-diarization answer."""

    def test_every_segment_gets_the_same_speaker(self) -> None:
        assert single_speaker_labels([3, 1, 2]) == {3: "SPK_0", 1: "SPK_0", 2: "SPK_0"}

    def test_no_segments_produces_no_labels(self) -> None:
        assert single_speaker_labels([]) == {}


class FakeEmbedder:
    """Returns scripted embeddings, so label repair can be tested on its own."""

    def __init__(self, script: Sequence[np.ndarray | None]) -> None:
        self.script = list(script)
        self.calls: list[tuple[int, Sequence[tuple[float, float]]]] = []

    def embed(
        self,
        samples: np.ndarray,
        sample_rate: int,
        regions: Sequence[tuple[float, float]],
    ) -> list[np.ndarray | None]:
        self.calls.append((sample_rate, regions))
        return self.script[: len(regions)]


def diarizer(script: Sequence[np.ndarray | None], **kwargs: Any) -> EmbeddingDiarizer:
    return EmbeddingDiarizer(embedder=FakeEmbedder(script), **kwargs)


REGIONS = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


class TestEmbeddingDiarizer:
    """Turning embeddings into a label per segment."""

    def test_two_voices_are_labelled(self) -> None:
        labels = diarizer([jitter(VOICE_A), jitter(VOICE_B), jitter(VOICE_A)]).label(
            [0, 1, 2], REGIONS, np.zeros(10), 16_000
        )
        assert labels == {0: "SPK_0", 1: "SPK_1", 2: "SPK_0"}

    def test_ids_are_paired_with_regions_in_order(self) -> None:
        labels = diarizer([jitter(VOICE_A), jitter(VOICE_B)]).label(
            [7, 3], [(0.0, 5.0), (5.0, 10.0)], np.zeros(10), 16_000
        )
        assert labels == {7: "SPK_0", 3: "SPK_1"}

    def test_a_failed_embedding_inherits_the_previous_speaker(self) -> None:
        # One bad window in the middle of an exchange must not invent a voice.
        labels = diarizer([jitter(VOICE_A), None, jitter(VOICE_B)]).label(
            [0, 1, 2], REGIONS, np.zeros(10), 16_000
        )
        assert labels == {0: "SPK_0", 1: "SPK_0", 2: "SPK_1"}

    def test_a_leading_failure_inherits_the_first_assigned_speaker(self) -> None:
        # Labels come from the first *embeddable* row, so the leading failure and
        # the voice after it are the same speaker.
        labels = diarizer([None, jitter(VOICE_B), jitter(VOICE_A)]).label(
            [0, 1, 2], REGIONS, np.zeros(10), 16_000
        )
        assert labels == {0: "SPK_0", 1: "SPK_0", 2: "SPK_1"}

    def test_every_embedding_failing_falls_back_to_one_speaker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            labels = diarizer([None, None, None]).label(
                [0, 1, 2], REGIONS, np.zeros(10), 16_000
            )
        assert labels == {0: "SPK_0", 1: "SPK_0", 2: "SPK_0"}
        assert "one speaker" in caplog.text

    def test_no_segments_is_an_empty_mapping(self) -> None:
        assert diarizer([]).label([], [], np.zeros(10), 16_000) == {}

    def test_the_threshold_reaches_the_clustering(self) -> None:
        # Two distinct voices at a threshold of 1.0 collapse into one.
        labels = diarizer([jitter(VOICE_A), jitter(VOICE_B)], threshold=1.0).label(
            [0, 1], [(0.0, 5.0), (5.0, 10.0)], np.zeros(10), 16_000
        )
        assert labels == {0: "SPK_0", 1: "SPK_0"}

    def test_a_short_cluster_is_folded_into_its_neighbour(self) -> None:
        labels = diarizer(
            [jitter(VOICE_A), jitter(VOICE_B), jitter(VOICE_B)], min_seconds=3.0
        ).label([0, 1, 2], [(0.0, 10.0), (10.0, 20.0), (20.0, 21.0)], np.zeros(10), 16_000)
        assert labels == {0: "SPK_0", 1: "SPK_1", 2: "SPK_1"}

    def test_defaults_follow_the_calibrated_constants(self) -> None:
        assert diarizer([VOICE_A]).threshold == DEFAULT_THRESHOLD
        assert diarizer([VOICE_A]).min_seconds == MIN_SPEAKER_S


class TestNullDiarizer:
    """Diarization switched off."""

    def test_every_segment_gets_the_same_speaker(self) -> None:
        assert NullDiarizer().label([0, 1, 2]) == {0: "SPK_0", 1: "SPK_0", 2: "SPK_0"}

    def test_audio_arguments_are_ignored(self) -> None:
        # The orchestrator calls this uniformly, so the extra positional args of
        # the embedding implementation must be harmless here.
        labels = NullDiarizer().label([0], [(0.0, 1.0)], np.zeros(10), 16_000)
        assert labels == {0: "SPK_0"}


class TestDetectDiarizer:
    """Choosing a backend, and degrading instead of failing."""

    @pytest.mark.parametrize("preference", ["none", "off", "false", "NONE", " none "])
    def test_off_aliases_return_the_null_diarizer(self, preference: str) -> None:
        assert isinstance(detect_diarizer(preference), NullDiarizer)

    def test_auto_returns_a_configured_embedding_diarizer(self) -> None:
        # funasr is the optional extra; the model is only loaded on first embed,
        # so this stays cheap even though it does import funasr.
        pytest.importorskip("funasr")
        resolved = detect_diarizer("auto", threshold=0.3, min_seconds=2.0)
        assert isinstance(resolved, EmbeddingDiarizer)
        assert resolved.threshold == 0.3
        assert resolved.min_seconds == 2.0

    def test_a_missing_dependency_degrades_to_one_speaker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulated by hiding funasr, which is what an uninstalled optional
        # dependency looks like from funasr's own import machinery.
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "funasr":
                raise ImportError("funasr is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert isinstance(detect_diarizer("auto"), NullDiarizer)


class TestFunASREmbedder:
    """The funasr wrapper's audio handling, without loading the model."""

    @pytest.mark.parametrize(
        "region",
        [pytest.param((0.0, 0.1), id="too-short"), pytest.param((5.0, 6.0), id="past-end")],
    )
    def test_a_window_with_no_audio_is_not_embedded(self, region: tuple[float, float]) -> None:
        embedder = FunASREmbedder()
        embedder._model = _ExplodingModel()
        assert embedder.embed(np.zeros(16_000, dtype=np.float32), 16_000, [region]) == [None]

    def test_a_usable_window_is_embedded_at_the_speaker_rate(self) -> None:
        # ERes2NetV2 is a 16 kHz model while the pipeline works at 48 kHz.
        embedder = FunASREmbedder()
        model = _RecordingModel()
        embedder._model = model
        result = embedder.embed(np.zeros(48_000, dtype=np.float32), 48_000, [(0.0, 1.0)])
        assert model.rates == [SPEAKER_RATE]
        assert result[0] is not None
        assert result[0].shape == (2,)

    def test_a_failing_window_does_not_fail_the_run(self) -> None:
        embedder = FunASREmbedder()
        embedder._model = _ExplodingModel()
        assert embedder.embed(np.zeros(16_000, dtype=np.float32), 16_000, [(0.0, 1.0)]) == [None]

    def test_a_failing_window_is_reported_at_warning_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The first failure is loud on purpose: every window failing silently looks
        # identical to a video that genuinely has one speaker, which cost a whole
        # debugging session on a real run.
        embedder = FunASREmbedder()
        embedder._model = _ExplodingModel()
        regions = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

        with caplog.at_level(logging.WARNING):
            embedder.embed(np.zeros(3 * 16_000, dtype=np.float32), 16_000, regions)

        warnings = [item for item in caplog.records if item.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Could not embed" in warnings[0].getMessage()

    def test_the_model_is_loaded_once_for_many_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        funasr = pytest.importorskip("funasr")
        loads: list[int] = []

        def counted(**kwargs: Any) -> Any:
            loads.append(1)
            return _RecordingModel()

        monkeypatch.setattr(funasr, "AutoModel", counted)
        embedder = FunASREmbedder()
        regions = [(float(index), float(index) + 1.0) for index in range(5)]

        embedder.embed(np.zeros(5 * 16_000, dtype=np.float32), 16_000, regions)

        # Reloading per window would make diarization unusably slow, and a raised
        # load is what a silent failure looked like on a real run.
        assert len(loads) == 1

    @pytest.mark.parametrize("requested", ["auto", "cpu", "cuda:0"])
    def test_the_device_is_never_left_as_auto(self, requested: str) -> None:
        # funasr hands the string to torch, which has no "auto" device, and it
        # fails *after* loading the weights - so every embedding comes back None.
        assert FunASREmbedder(device=requested).device != "auto"

    def test_a_failure_to_load_the_model_is_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A broken install must not be mistaken for a one-voice video.
        funasr = pytest.importorskip("funasr")

        def explode(**kwargs: Any) -> Any:
            raise RuntimeError("weights are corrupt")

        monkeypatch.setattr(funasr, "AutoModel", explode)
        embedder = FunASREmbedder()
        with pytest.raises(RuntimeError, match="corrupt"):
            embedder.embed(np.zeros(16_000, dtype=np.float32), 16_000, [(0.0, 1.0)])


class _ExplodingModel:
    def generate(self, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


class TimeVoiceEmbedder:
    """Returns a voice's vector for each window, chosen by the window midpoint.

    The overlap detector generates its windows internally, so the script is a
    function of time rather than a fixed list: each ``(start, end, vector)`` entry
    says which voice occupies that span, and a window is labelled by its midpoint.
    """

    name = "scripted"

    def __init__(self, timeline: Sequence[tuple[float, float, np.ndarray]]) -> None:
        self.timeline = list(timeline)
        self.calls = 0
        self.regions: list[tuple[float, float]] = []

    def embed(
        self,
        samples: np.ndarray,
        sample_rate: int,
        regions: Sequence[tuple[float, float]],
    ) -> list[np.ndarray | None]:
        self.calls += 1
        results: list[np.ndarray | None] = []
        for start, end in regions:
            self.regions.append((start, end))
            midpoint = (start + end) / 2.0
            vector: np.ndarray | None = None
            for low, high, voice in self.timeline:
                if low <= midpoint < high:
                    vector = np.asarray(voice, dtype=np.float64)
                    break
            results.append(vector)
        return results


class TestWindowBounds:
    """Cutting a segment into overlap windows."""

    def test_a_long_segment_is_covered_without_gaps(self) -> None:
        assert _window_bounds(0.0, 5.0) == [(0.0, 2.5), (1.25, 3.75), (2.5, 5.0)]

    def test_a_segment_shorter_than_a_window_is_one_window(self) -> None:
        assert _window_bounds(2.0, 4.0) == [(2.0, 4.0)]

    def test_the_final_window_is_flush_with_the_end(self) -> None:
        bounds = _window_bounds(0.0, 6.0)
        assert bounds[-1][1] == 6.0
        assert all(start < end for start, end in bounds)

    def test_an_empty_region_has_no_windows(self) -> None:
        assert _window_bounds(3.0, 3.0) == []

    def test_the_constants_match_the_measured_calibration(self) -> None:
        assert OVERLAP_WINDOW_S == 2.5
        assert OVERLAP_HOP_S == 1.25
        assert MIN_OVERLAP_S == 0.6


class TestLongestOtherSpan:
    """The consecutive-run measure behind the overlap flag."""

    def test_a_run_is_measured_from_its_first_start_to_its_last_end(self) -> None:
        windows = [
            (0.0, 2.5, True),
            (1.25, 3.75, False),
            (2.5, 5.0, False),
            (3.75, 6.25, True),
        ]
        assert _longest_other_span(windows) == 3.75

    def test_non_consecutive_windows_are_not_one_run(self) -> None:
        windows = [(0.0, 2.5, False), (1.25, 3.75, True), (2.5, 5.0, False)]
        assert _longest_other_span(windows) == 2.5

    def test_an_unscored_window_ends_a_run(self) -> None:
        windows = [(0.0, 2.5, False), (1.25, 3.75, None), (2.5, 5.0, False)]
        assert _longest_other_span(windows) == 2.5

    def test_no_other_windows_is_zero(self) -> None:
        assert _longest_other_span([(0.0, 2.5, True)]) == 0.0


class TestOverlapDetection:
    """Flagging segments that carry a second voice."""

    def test_a_one_voice_job_is_never_flagged(self) -> None:
        embedder = TimeVoiceEmbedder([(0.0, 10.0, VOICE_A)])
        diarizer = EmbeddingDiarizer(embedder=embedder)
        flags = diarizer.overlaps(
            [0, 1],
            [(0.0, 5.0), (5.0, 10.0)],
            {0: "SPK_0", 1: "SPK_0"},
            np.zeros(10),
            16_000,
        )
        assert flags == {0: False, 1: False}
        # No second centroid means no window can match one, so the audio is never
        # even read - a noisy solo recording cannot be mistaken for an overlap.
        assert embedder.calls == 0

    def test_a_clean_exchange_is_not_flagged(self) -> None:
        embedder = TimeVoiceEmbedder([(0.0, 5.0, VOICE_A), (5.0, 10.0, VOICE_B)])
        diarizer = EmbeddingDiarizer(embedder=embedder)
        flags = diarizer.overlaps(
            [0, 1],
            [(0.0, 5.0), (5.0, 10.0)],
            {0: "SPK_0", 1: "SPK_1"},
            np.zeros(10),
            16_000,
        )
        assert flags == {0: False, 1: False}

    def test_a_second_voice_inside_a_segment_is_flagged(self) -> None:
        embedder = TimeVoiceEmbedder(
            [(0.0, 2.0, VOICE_A), (2.0, 3.0, VOICE_B), (3.0, 5.0, VOICE_A), (5.0, 10.0, VOICE_B)]
        )
        diarizer = EmbeddingDiarizer(embedder=embedder)
        flags = diarizer.overlaps(
            [0, 1],
            [(0.0, 5.0), (5.0, 10.0)],
            {0: "SPK_0", 1: "SPK_1"},
            np.zeros(10),
            16_000,
        )
        assert flags == {0: True, 1: False}

    def test_a_segment_owned_by_another_voice_is_not_an_overlap(self) -> None:
        # Two of three windows match the other speaker: that is a labelling
        # problem (a mislabelled segment), so it is not reported as an overlap.
        embedder = TimeVoiceEmbedder(
            [
                (0.0, 5.0, VOICE_A),
                (5.0, 10.0, VOICE_B),
                (10.0, 11.25, VOICE_B),
                (11.25, 13.75, VOICE_B),
                (13.75, 15.0, VOICE_A),
            ]
        )
        diarizer = EmbeddingDiarizer(embedder=embedder)
        flags = diarizer.overlaps(
            [0, 1, 2],
            [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)],
            {0: "SPK_0", 1: "SPK_1", 2: "SPK_0"},
            np.zeros(10),
            16_000,
        )
        assert flags[2] is False

    def test_an_unscored_window_is_skipped_rather_than_guessed(self) -> None:
        # The middle window cannot be embedded (no voice covers its midpoint), so
        # it neither flags the segment nor breaks the run accounting.
        embedder = TimeVoiceEmbedder(
            [(0.0, 2.0, VOICE_A), (3.0, 5.0, VOICE_A), (5.0, 10.0, VOICE_B)]
        )
        diarizer = EmbeddingDiarizer(embedder=embedder)
        flags = diarizer.overlaps(
            [0, 1],
            [(0.0, 5.0), (5.0, 10.0)],
            {0: "SPK_0", 1: "SPK_1"},
            np.zeros(10),
            16_000,
        )
        assert flags == {0: False, 1: False}

    def test_mismatched_lengths_are_rejected(self) -> None:
        diarizer = EmbeddingDiarizer(embedder=TimeVoiceEmbedder([]))
        with pytest.raises(ValueError, match="same length"):
            diarizer.overlaps([0, 1], [(0.0, 5.0)], {}, np.zeros(10), 16_000)

    def test_an_empty_run_is_an_empty_mapping(self) -> None:
        diarizer = EmbeddingDiarizer(embedder=TimeVoiceEmbedder([]))
        assert diarizer.overlaps([], [], {}, np.zeros(10), 16_000) == {}


class _RecordingModel:
    """Records the rate it was asked for, and returns a fixed embedding."""

    def __init__(self) -> None:
        self.rates: list[int] = []

    def generate(self, *, input: Any, fs: int) -> Any:
        self.rates.append(fs)
        return [{"spk_embedding": np.array([1.0, 0.0], dtype=np.float32)}]


class TestAssignStreams:
    """Matching separated streams back to the speakers that produced them."""

    CENTROIDS = {"SPK_0": VOICE_A, "SPK_1": VOICE_B}

    def test_two_streams_are_matched_to_their_speakers(self) -> None:
        assert assign_streams(
            [jitter(VOICE_A), jitter(VOICE_B)], "SPK_0", self.CENTROIDS
        ) == ["SPK_0", "SPK_1"]

    def test_the_stream_order_does_not_matter(self) -> None:
        # The dominant voice happens to be the *second* stream here.
        assert assign_streams(
            [jitter(VOICE_B), jitter(VOICE_A)], "SPK_0", self.CENTROIDS
        ) == ["SPK_1", "SPK_0"]

    def test_a_false_positive_flag_is_not_split(self) -> None:
        # Both streams are the same voice, so the segment was never overlapped.
        assert assign_streams(
            [jitter(VOICE_A), jitter(VOICE_A)], "SPK_0", self.CENTROIDS
        ) is None

    def test_one_speaker_cannot_be_assigned(self) -> None:
        assert assign_streams(
            [jitter(VOICE_A), jitter(VOICE_B)], "SPK_0", {"SPK_0": VOICE_A}
        ) is None

    def test_an_unembeddable_stream_is_rejected(self) -> None:
        assert assign_streams([None, jitter(VOICE_B)], "SPK_0", self.CENTROIDS) is None

    def test_an_unknown_dominant_speaker_is_rejected(self) -> None:
        assert assign_streams(
            [jitter(VOICE_A), jitter(VOICE_B)], "SPK_9", self.CENTROIDS
        ) is None


class TestCentroids:
    """One mean embedding per speaker, used to match a separated stream."""

    def test_one_centroid_per_speaker(self) -> None:
        engine = diarizer([jitter(VOICE_A), jitter(VOICE_B), jitter(VOICE_A)])
        result = engine.centroids(
            [0, 1, 2],
            REGIONS,
            {0: "SPK_0", 1: "SPK_1", 2: "SPK_0"},
            np.zeros(10),
            16_000,
        )
        assert set(result) == {"SPK_0", "SPK_1"}
        # SPK_0's centroid averages its two recordings, so it sits nearer VOICE_A.
        assert _cosine(result["SPK_0"], VOICE_A) > _cosine(result["SPK_0"], VOICE_B)

    def test_a_speaker_with_no_usable_segment_is_omitted(self) -> None:
        engine = diarizer([None, jitter(VOICE_B)])
        result = engine.centroids(
            [0, 1],
            [(0.0, 5.0), (5.0, 10.0)],
            {0: "SPK_0", 1: "SPK_1"},
            np.zeros(10),
            16_000,
        )
        assert set(result) == {"SPK_1"}

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            diarizer([]).centroids([0, 1], [(0.0, 5.0)], {}, np.zeros(10), 16_000)
