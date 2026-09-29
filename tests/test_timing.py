"""Tests for duration fitting, stretching backends and timeline assembly."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import DependencyError
from pipeline.timing import (
    IdentityStretcher,
    LibrosaStretcher,
    RubberbandStretcher,
    SoxStretcher,
    _fit_length,
    build_timeline,
    compute_ratio,
    detect_stretcher,
    fit_line,
    fit_ratio,
    trim_silence,
)

SR = 24_000


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    """RMS of an array, as a plain float."""
    data = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(data))))


class SpyStretcher:
    """Records stretch requests and returns the audio unchanged."""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def stretch(self, samples: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
        self.calls.append((len(samples), ratio))
        return np.asarray(samples, dtype=np.float32)


class TestComputeRatio:
    def test_exact_fit_is_one(self) -> None:
        assert compute_ratio(2.0, 2.0) == pytest.approx(1.0)

    def test_slower_stretch_when_synthesis_is_too_long(self) -> None:
        assert compute_ratio(2.0, 4.0) == pytest.approx(0.5, abs=0.5)

    def test_clamped_at_both_ends(self) -> None:
        assert compute_ratio(10.0, 1.0) == pytest.approx(1.18)  # needs 10x, clamped up
        assert compute_ratio(1.0, 10.0) == pytest.approx(0.85)  # needs 0.1x, clamped down

    def test_custom_bounds_are_respected(self) -> None:
        assert compute_ratio(10.0, 1.0, min_ratio=0.5, max_ratio=2.0) == pytest.approx(2.0)

    @pytest.mark.parametrize(("source", "synth"), [(0.0, 2.0), (2.0, 0.0), (-1.0, 1.0)])
    def test_unusable_durations_are_neutral(self, source: float, synth: float) -> None:
        assert compute_ratio(source, synth) == 1.0


class TestFitRatio:
    """The floor and the following gap decide how a line is fitted."""

    def test_a_line_that_exactly_fits_is_untouched(self) -> None:
        assert fit_ratio(2.0, 2.0, 2.0) == pytest.approx(1.0)

    def test_compression_aims_at_the_slot(self) -> None:
        # Speech should stop when the source speaker stopped, so the slot is the
        # target even when the gap alone would have covered the difference.
        assert fit_ratio(2.0, 2.4, 2.4, min_ratio=0.75) == pytest.approx(2.0 / 2.4)

    def test_a_short_line_is_slowed_to_fill_its_slot(self) -> None:
        assert fit_ratio(2.0, 2.0, 1.0) == pytest.approx(1.18)

    def test_the_floor_limits_compression(self) -> None:
        # 4.0s of speech in a 2.0s slot needs 0.5, far below the floor.
        assert fit_ratio(2.0, 2.0, 4.0, min_ratio=0.75) == pytest.approx(0.75)

    def test_the_gap_is_spent_before_going_below_the_floor(self) -> None:
        # The slot alone would need 0.5, under the floor; the 1.4s gap pays for
        # the difference, so only 0.85 is applied and no quality is spent twice.
        assert fit_ratio(2.0, 3.4, 4.0, min_ratio=0.75) == pytest.approx(0.85)

    def test_the_gap_cannot_justify_more_compression(self) -> None:
        # A gap smaller than the excess still leaves the floor as the answer.
        assert fit_ratio(2.0, 2.4, 4.0, min_ratio=0.75) == pytest.approx(0.75)

    def test_a_zero_window_falls_back_to_the_slot(self) -> None:
        assert fit_ratio(2.0, 0.0, 2.5, min_ratio=0.75) == pytest.approx(0.8)

    @pytest.mark.parametrize(
        ("slot", "window", "synth"),
        [(0.0, 2.0, 3.0), (2.0, 2.0, 0.0), (-1.0, 2.0, 3.0)],
    )
    def test_unusable_durations_are_neutral(
        self, slot: float, window: float, synth: float
    ) -> None:
        assert fit_ratio(slot, window, synth) == 1.0


class TestTrimSilence:
    """Dead air at the ends of a synthesised line is not speech."""

    @staticmethod
    def padded(lead: float, speech: float, trail: float) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(int(lead * SR), dtype=np.float32),
                tone(speech),
                np.zeros(int(trail * SR), dtype=np.float32),
            ]
        )

    def test_leading_and_trailing_silence_are_dropped(self) -> None:
        trimmed = trim_silence(self.padded(0.35, 1.0, 0.2), SR)
        # 1.0s of speech plus the 50 ms guard on each side: 0.30s and 0.15s of
        # genuine silence are gone.
        assert len(trimmed) / SR == pytest.approx(1.1, abs=0.02)

    def test_a_clip_without_silence_is_unchanged(self) -> None:
        source = tone(1.0)
        assert trim_silence(source, SR).size == source.size

    def test_the_guard_bounds_the_start(self) -> None:
        """Silence nearer than the guard survives, so an attack is not cut."""
        source = self.padded(0.02, 1.0, 0.0)
        assert trim_silence(source, SR).size == source.size

    def test_a_quiet_line_is_measured_against_its_own_peak(self) -> None:
        quiet = self.padded(0.3, 1.0, 0.3) * 0.01
        trimmed = trim_silence(quiet, SR)
        assert len(trimmed) < quiet.size
        assert len(trimmed) / SR > 1.0

    def test_silence_alone_is_returned_untouched(self) -> None:
        source = np.zeros(2400, dtype=np.float32)
        assert trim_silence(source, SR).size == source.size

    def test_a_transient_does_not_shrink_a_line_to_a_click(self) -> None:
        """A single spike would otherwise make everything else "silence"."""
        source = np.concatenate(
            [np.zeros(int(2.0 * SR), dtype=np.float32), np.array([0.9], dtype=np.float32)]
        )
        assert trim_silence(source, SR).size == source.size

    def test_empty_input_is_returned_unchanged(self) -> None:
        assert trim_silence(np.zeros(0, dtype=np.float32), SR).size == 0


class TestFitLength:
    def test_pads_with_silence_when_short(self) -> None:
        result = _fit_length(tone(0.5), int(1.0 * SR), SR)
        assert result.size == int(1.0 * SR)
        assert np.allclose(result[int(0.6 * SR) :], 0.0)

    def test_trims_when_long(self) -> None:
        result = _fit_length(tone(2.0), int(1.0 * SR), SR)
        assert result.size == int(1.0 * SR)

    def test_trim_applies_a_fade_out(self) -> None:
        result = _fit_length(tone(2.0), int(1.0 * SR), SR, fade_ms=10)
        tail = result[-int(0.01 * SR) :]
        # The very end must decay towards silence to avoid a click.
        assert abs(float(tail[-1])) < abs(float(tail[0])) + 1e-6
        assert abs(float(result[-1])) < 1e-3

    def test_exact_length_is_returned_unchanged(self) -> None:
        samples = tone(1.0)
        assert np.array_equal(_fit_length(samples, samples.size, SR), samples)

    def test_zero_target_returns_empty(self) -> None:
        assert _fit_length(tone(1.0), 0, SR).size == 0


class TestFitLine:
    def test_none_becomes_exact_silence(self) -> None:
        result, ratio = fit_line(None, SR, 1.5, stretcher=SpyStretcher())
        assert result.size == int(1.5 * SR)
        assert np.allclose(result, 0.0)
        assert ratio == 1.0

    def test_empty_array_becomes_silence(self) -> None:
        result, _ = fit_line(np.zeros(0, dtype=np.float32), SR, 1.0, stretcher=SpyStretcher())
        assert result.size == int(SR)

    def test_no_stretch_when_already_close(self) -> None:
        stretcher = SpyStretcher()
        result, ratio = fit_line(tone(2.0), SR, 2.02, stretcher=stretcher)
        assert stretcher.calls == []
        assert ratio == pytest.approx(1.01, abs=0.01)
        assert result.size == int(2.02 * SR)

    def test_stretches_when_meaningfully_off(self) -> None:
        stretcher = SpyStretcher()
        _, ratio = fit_line(tone(1.0), SR, 2.0, stretcher=stretcher)
        assert len(stretcher.calls) == 1
        assert ratio == pytest.approx(1.18)  # clamped, not 2.0

    def test_pads_to_the_slot_when_shorter(self) -> None:
        for target in (1.5, 3.25):
            result, _ = fit_line(tone(1.0), SR, target, stretcher=SpyStretcher())
            assert result.size == int(round(target * SR))

    def test_never_trims_by_default(self) -> None:
        """Without a budget the whole clip survives, even if it overruns."""
        source = tone(2.0)
        result, ratio = fit_line(source, SR, 0.5, stretcher=SpyStretcher())
        assert result.size == source.size
        assert ratio == pytest.approx(0.85)  # compression clamped

    def test_zero_overrun_forces_the_exact_slot(self) -> None:
        result, _ = fit_line(
            tone(2.0), SR, 0.5, stretcher=SpyStretcher(), max_overrun=0.0
        )
        assert result.size == int(round(0.5 * SR))

    def test_overrun_budget_is_respected(self) -> None:
        """A line may use its slot plus the stated budget, and no more."""
        result, _ = fit_line(
            tone(2.0), SR, 0.5, stretcher=SpyStretcher(), max_overrun=0.25
        )
        assert result.size == int(round(0.75 * SR))

    def test_budget_larger_than_the_clip_keeps_everything(self) -> None:
        source = tone(1.0)
        result, _ = fit_line(
            source, SR, 0.5, stretcher=SpyStretcher(), max_overrun=5.0
        )
        assert result.size == source.size

    def test_zero_target_is_rejected(self) -> None:
        with pytest.raises(Exception, match="target_duration"):
            fit_line(tone(1.0), SR, 0.0, stretcher=SpyStretcher())
    def test_the_available_window_softens_the_compression(self) -> None:
        """A gap the line may spill into is spent before more tempo."""
        _, ratio = fit_line(
            tone(4.0), SR, 2.0, stretcher=SpyStretcher(), min_ratio=0.75, available=3.4
        )
        assert ratio == pytest.approx(0.85)

    def test_the_available_window_never_adds_compression(self) -> None:
        # A line that already fits its slot is compressed to the slot regardless.
        _, ratio = fit_line(
            tone(2.4), SR, 2.0, stretcher=SpyStretcher(), min_ratio=0.75, available=3.4
        )
        assert ratio == pytest.approx(2.0 / 2.4)

    def test_the_floor_decides_when_the_window_is_too_small(self) -> None:
        result, ratio = fit_line(
            tone(4.0),
            SR,
            2.0,
            stretcher=SpyStretcher(),
            min_ratio=0.75,
            max_overrun=0.4,
            available=2.4,
        )
        assert ratio == pytest.approx(0.75)
        assert result.size == int(round(2.4 * SR))


class TestBuildTimeline:
    def test_places_clips_at_their_start_times(self) -> None:
        timeline = build_timeline(
            [(0, 0.0, tone(1.0)), (1, 2.0, tone(1.0))], sample_rate=SR, total_duration=4.0
        )
        assert len(timeline.placements) == 2
        assert timeline.placements[0].start == pytest.approx(0.0)
        assert timeline.placements[1].start == pytest.approx(2.0)
        # The gap between the clips stays silent.
        assert np.allclose(timeline.samples[int(1.2 * SR) : int(1.8 * SR)], 0.0)

    def test_overlaps_are_pushed_later(self) -> None:
        timeline = build_timeline(
            [(0, 0.0, tone(2.0)), (1, 0.5, tone(1.0))], sample_rate=SR, total_duration=4.0
        )
        second = timeline.placements[1]
        assert second.shifted is True
        assert second.start == pytest.approx(2.0)

    def test_a_rounding_level_move_is_not_a_shift(self) -> None:
        """A line fitted to end on the next start can land a sample past it.

        Lines are now compressed to fill their window, so this is the common case,
        and calling it a desync would fail the QC gate on arithmetic.
        """
        previous = tone(1.0)
        over = previous.size + 2  # two samples past the next line's start
        timeline = build_timeline(
            [(0, 0.0, tone(over / SR)), (1, 1.0, tone(0.5))],
            sample_rate=SR,
            total_duration=4.0,
        )
        second = timeline.placements[1]
        # Still placed after the first clip, but not reported as a desync.
        assert second.start == pytest.approx(over / SR, abs=1 / SR)
        assert second.shifted is False

    def test_a_perceptible_move_is_still_a_shift(self) -> None:
        timeline = build_timeline(
            [(0, 0.0, tone(1.0)), (1, 0.95, tone(0.5))],
            sample_rate=SR,
            total_duration=4.0,
        )
        second = timeline.placements[1]
        assert second.start == pytest.approx(1.0)
        assert second.shifted is True

    def test_empty_audio_is_skipped(self) -> None:
        timeline = build_timeline(
            [(0, 0.0, np.zeros(0, dtype=np.float32))], sample_rate=SR, total_duration=1.0
        )
        assert timeline.placements == []

    def test_track_grows_to_fit_late_audio(self) -> None:
        timeline = build_timeline(
            [(0, 5.0, tone(1.0))], sample_rate=SR, total_duration=1.0
        )
        assert len(timeline.samples) >= int(6.0 * SR)

    def test_track_respects_total_duration(self) -> None:
        timeline = build_timeline([(0, 0.0, tone(0.5))], sample_rate=SR, total_duration=3.0)
        assert len(timeline.samples) == int(3.0 * SR)

    def test_overlapping_sum_never_clips(self) -> None:
        loud = tone(1.0, amplitude=0.9)
        timeline = build_timeline(
            [(0, 0.0, loud), (1, 0.0, loud)], sample_rate=SR, total_duration=2.0
        )
        assert float(np.max(np.abs(timeline.samples))) <= 1.0

    def test_overlay_lines_are_summed_at_their_true_time(self) -> None:
        # A separated second voice is genuinely simultaneous, so it is placed at
        # its own start instead of being pushed behind the dominant line.
        first = tone(1.0, amplitude=0.2)
        second = tone(0.5, amplitude=0.2)
        timeline = build_timeline(
            [(0, 0.0, first), (1, 0.0, second)],
            sample_rate=SR,
            total_duration=2.0,
            overlay_ids={1},
        )
        overlay = timeline.placements[1]
        assert overlay.shifted is False
        assert overlay.start == pytest.approx(0.0)
        # Summed, not replaced: the overlay's span is louder than either alone.
        window = slice(0, int(0.5 * SR))
        assert rms(timeline.samples[window]) > rms(first[window])

    def test_an_overlay_does_not_shift_a_following_line(self) -> None:
        timeline = build_timeline(
            [(0, 0.0, tone(1.0)), (1, 0.0, tone(2.0)), (2, 1.0, tone(0.5))],
            sample_rate=SR,
            total_duration=3.0,
            overlay_ids={1},
        )
        # The cursor advances only for the dominant line, so line 2 starts where
        # its own timeline says rather than after the (simultaneous) overlay.
        assert timeline.placements[2].shifted is False
        assert timeline.placements[2].start == pytest.approx(1.0)

    def test_a_long_overlay_still_grows_the_track(self) -> None:
        timeline = build_timeline(
            [(0, 0.0, tone(0.5)), (1, 0.0, tone(3.0))],
            sample_rate=SR,
            total_duration=1.0,
            overlay_ids={1},
        )
        assert len(timeline.samples) >= int(3.0 * SR)

    def test_negative_start_is_clamped_to_zero(self) -> None:
        timeline = build_timeline(
            [(0, -1.0, tone(0.5))], sample_rate=SR, total_duration=1.0
        )
        assert timeline.placements[0].start == pytest.approx(0.0)

    def test_invalid_sample_rate_is_rejected(self) -> None:
        with pytest.raises(Exception, match="sample_rate"):
            build_timeline([], sample_rate=0)


class TestDetectStretcher:
    def test_explicit_none_returns_identity(self) -> None:
        assert isinstance(detect_stretcher("none"), IdentityStretcher)

    def test_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(DependencyError, match="Unknown stretcher"):
            detect_stretcher("magic")

    def test_explicit_missing_binary_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(DependencyError, match="not on PATH"):
            detect_stretcher("rubberband")

    def test_auto_prefers_rubberband(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert isinstance(detect_stretcher("auto"), RubberbandStretcher)

    def test_auto_falls_back_to_sox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "shutil.which", lambda name: "/usr/bin/sox" if name == "sox" else None
        )
        assert isinstance(detect_stretcher("auto"), SoxStretcher)

    def test_auto_falls_back_to_librosa_without_binaries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert isinstance(detect_stretcher("auto"), LibrosaStretcher)

    def test_explicit_librosa_always_available(self) -> None:
        assert isinstance(detect_stretcher("librosa"), LibrosaStretcher)


class TestStretcherCommands:
    def test_rubberband_uses_time_ratio(self, tmp_path) -> None:
        stretcher = RubberbandStretcher(tmp_path)
        command = stretcher._command(tmp_path / "i.wav", tmp_path / "o.wav", 1.1)
        assert command[1] == "--time"
        assert command[2] == "1.100000"

    def test_sox_uses_tempo(self, tmp_path) -> None:
        stretcher = SoxStretcher(tmp_path)
        command = stretcher._command(tmp_path / "i.wav", tmp_path / "o.wav", 0.9)
        assert command[-2] == "tempo"
        assert command[-1] == "0.900000"


class TestIdentityStretcher:
    def test_returns_input_unchanged(self) -> None:
        samples = tone(1.0)
        assert np.array_equal(IdentityStretcher().stretch(samples, SR, 1.5), samples)


class TestLibrosaStretcher:
    def test_changes_duration(self) -> None:
        result = LibrosaStretcher().stretch(tone(1.0), SR, 1.15)
        assert result.size > int(1.0 * SR)

    def test_neutral_ratio_is_a_noop(self) -> None:
        samples = tone(1.0)
        assert np.array_equal(LibrosaStretcher().stretch(samples, SR, 1.0), samples)
