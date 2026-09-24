"""Tests for background ducking and mixing."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import media
from pipeline.mixing import DEFAULT_DUCK_DB, duck_and_mix, duck_envelope

SR = 48_000


def tone(seconds: float, amplitude: float = 0.3, frequency: float = 300.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


class TestDuckEnvelope:
    def test_unity_outside_dialogue(self) -> None:
        envelope = duck_envelope(SR * 10, SR, [(3.0, 5.0)])
        assert envelope[0] == pytest.approx(1.0, abs=1e-3)
        assert envelope[-1] == pytest.approx(1.0, abs=1e-3)

    def test_reaches_the_floor_inside_dialogue(self) -> None:
        envelope = duck_envelope(SR * 10, SR, [(3.0, 5.0)], duck_db=6.0, ramp_ms=150)
        middle = envelope[int(4.0 * SR)]
        assert middle == pytest.approx(10 ** (-6.0 / 20.0), abs=1e-3)

    def test_ramps_are_gradual(self) -> None:
        """A hard gate would click; the transition must span many samples."""
        envelope = duck_envelope(SR * 10, SR, [(3.0, 5.0)], ramp_ms=200)
        window = envelope[int(2.9 * SR) : int(3.2 * SR)]
        # More than a handful of distinct values means a real ramp, not a step.
        assert len(np.unique(np.round(window, 4))) > 20

    def test_never_exceeds_unity_or_goes_negative(self) -> None:
        envelope = duck_envelope(SR * 4, SR, [(0.0, 4.0)], duck_db=12.0)
        assert float(envelope.max()) <= 1.0
        assert float(envelope.min()) > 0.0

    def test_zero_duck_is_silent_no_op(self) -> None:
        envelope = duck_envelope(SR, SR, [(0.0, 1.0)], duck_db=0.0)
        assert np.allclose(envelope, 1.0, atol=1e-3)

    def test_empty_length_is_safe(self) -> None:
        assert duck_envelope(0, SR, [(0.0, 1.0)]).size == 0

    def test_regions_beyond_the_end_are_clipped(self) -> None:
        envelope = duck_envelope(SR, SR, [(0.5, 99.0)])
        assert envelope.size == SR
        # Fully ducked from the mid-point onwards: the region is clamped to the
        # buffer rather than raising.
        assert envelope[-1] == pytest.approx(10 ** (-6.0 / 20.0), abs=1e-3)

    def test_multiple_regions(self) -> None:
        envelope = duck_envelope(SR * 10, SR, [(1.0, 2.0), (6.0, 7.0)])
        assert envelope[int(1.5 * SR)] < 0.6
        assert envelope[int(6.5 * SR)] < 0.6
        assert envelope[int(4.0 * SR)] == pytest.approx(1.0, abs=1e-3)


class TestDuckAndMix:
    def test_background_is_kept_in_the_gaps(self) -> None:
        dubbed = np.zeros(SR * 8, dtype=np.float32)
        dubbed[int(3.0 * SR) : int(5.0 * SR)] = tone(2.0)  # speech in the middle
        bed = tone(8.0, amplitude=0.2)

        mixed = duck_and_mix(dubbed, bed, [(3.0, 5.0)], SR)

        gap = mixed[int(6.0 * SR) : int(8.0 * SR)]
        assert rms(gap) > 0.01  # music survives

    def test_background_is_reduced_under_dialogue(self) -> None:
        dubbed = np.zeros(SR * 8, dtype=np.float32)
        dubbed[int(3.0 * SR) : int(5.0 * SR)] = tone(2.0, amplitude=0.5)
        bed = tone(8.0, amplitude=0.2)

        mixed = duck_and_mix(dubbed, bed, [(3.0, 5.0)], SR, duck_db=6.0)

        # Compare the bed-only region with the ducked region's non-speech floor by
        # mixing a silent dub in, so only the bed is present.
        bed_only = duck_and_mix(np.zeros(SR * 8, dtype=np.float32), bed, [(3.0, 5.0)], SR, duck_db=6.0)
        ducked_level = rms(bed_only[int(3.5 * SR) : int(4.5 * SR)])
        open_level = rms(bed_only[int(6.0 * SR) : int(7.0 * SR)])
        assert ducked_level < open_level * 0.65  # ~6 dB down

    def test_no_background_returns_speech_only(self) -> None:
        dubbed = tone(1.0)
        mixed = duck_and_mix(dubbed, None, [], SR)
        assert np.allclose(mixed, dubbed, atol=1e-6)

    def test_empty_background_returns_speech_only(self) -> None:
        dubbed = tone(1.0)
        mixed = duck_and_mix(dubbed, np.zeros(0, dtype=np.float32), [], SR)
        assert np.allclose(mixed, dubbed, atol=1e-6)

    def test_output_length_covers_both_inputs(self) -> None:
        dubbed = tone(2.0)
        bed = tone(5.0)
        assert duck_and_mix(dubbed, bed, [], SR).size == bed.size

    def test_never_clips(self) -> None:
        dubbed = tone(4.0, amplitude=0.9)
        bed = tone(4.0, amplitude=0.9)
        mixed = duck_and_mix(dubbed, bed, [(0.0, 4.0)], SR, duck_db=0.0)
        assert float(np.max(np.abs(mixed))) <= media.LIMIT_CEILING + 1e-6

    def test_background_gain_offset_is_applied(self) -> None:
        dubbed = np.zeros(SR * 2, dtype=np.float32)
        bed = tone(2.0, amplitude=0.05)
        louder = duck_and_mix(dubbed, bed, [], SR, background_gain_db=6.0)
        assert rms(louder) > rms(bed) * 1.5

    def test_default_duck_is_six_db(self) -> None:
        assert DEFAULT_DUCK_DB == 6.0
