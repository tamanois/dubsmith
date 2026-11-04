"""Tests for background ducking and mixing."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import media
from pipeline.mixing import (
    DEFAULT_DUCK_DB,
    DEFAULT_ROOM_DUCK_DB,
    DEFAULT_ROOM_TONE_GAIN_DB,
    DEFAULT_ROOM_WET_DB,
    ROOM_DAMPING_HZ,
    ROOM_TONE_GATE_DB,
    RT60_BOUNDS,
    apply_room,
    convolve_reverb,
    duck_and_mix,
    duck_envelope,
    estimate_rt60,
    room_tone_floor,
    synthetic_ir,
)

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

        duck_and_mix(dubbed, bed, [(3.0, 5.0)], SR, duck_db=6.0)

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


def decaying_noise(rt60: float, seconds: float, *, seed: int = 0) -> np.ndarray:
    """Noise under an exponential decay, i.e. a room with a known RT60."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    envelope = 10.0 ** (-3.0 * t / rt60)
    return (0.4 * rng.standard_normal(t.size) * envelope).astype(np.float32)


def room_tone_stem(seconds: float = 4.0) -> np.ndarray:
    """A stem shaped like real separated speech: loud utterances over quiet room tone.

    A constant tone is not a usable stand-in any more: the floor now gates on
    level, so a stem with no quiet frames correctly yields nothing.
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    room = 0.001 * rng.standard_normal(t.size)  # ~-60 dBFS ambience
    voice = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    bursts = np.zeros(t.size)
    for start, end in ((0.5, 1.5), (2.0, 3.0)):
        bursts[int(start * SR) : int(end * SR)] = 1.0
    return (room + voice * bursts).astype(np.float32)


class TestRoomToneFloor:
    def test_sits_at_the_requested_level_relative_to_the_stem(self) -> None:
        speech = room_tone_stem()
        floor = room_tone_floor(speech, SR, [], gain_db=-30.0)
        assert rms(floor) / rms(speech) == pytest.approx(10 ** (-30.0 / 20.0), rel=0.2)

    def test_the_voice_is_not_laid_under_the_dub(self) -> None:
        """The floor must be ambience, not the speaker: their level must not track."""
        speech = room_tone_stem()
        floor = room_tone_floor(speech, SR, [], gain_db=-30.0)

        in_burst = slice(int(0.8 * SR), int(1.2 * SR))
        in_pause = slice(int(1.7 * SR), int(1.9 * SR))
        assert rms(floor[in_burst]) < rms(speech[in_burst]) * 0.1
        # ...and the pauses are what survives, so the floor is not silent either.
        assert rms(floor[in_pause]) > 0.0

    def test_dialogue_regions_are_suppressed(self) -> None:
        speech = room_tone_stem()
        with_regions = room_tone_floor(speech, SR, [(0.5, 1.5)], gain_db=-30.0)
        without = room_tone_floor(speech, SR, [], gain_db=-30.0)
        assert rms(with_regions) < rms(without)

    def test_a_stem_with_no_quiet_frames_fabricates_nothing(self) -> None:
        """Amplifying residue is capped, so a wall of speech yields no fake room."""
        loud = tone(3.0, amplitude=0.3)
        floor = room_tone_floor(loud, SR, [], gain_db=0.0)
        assert rms(floor) < rms(loud) * 0.1

    def test_gain_db_offsets_the_floor(self) -> None:
        speech = room_tone_stem()
        loud = room_tone_floor(speech, SR, [], gain_db=-10.0)
        quiet = room_tone_floor(speech, SR, [], gain_db=-30.0)
        assert rms(quiet) / rms(loud) == pytest.approx(0.1, rel=0.3)

    def test_length_is_preserved(self) -> None:
        speech = room_tone_stem(2.0)
        assert room_tone_floor(speech, SR, [(0.5, 1.0)]).size == speech.size

    def test_empty_input_is_safe(self) -> None:
        assert room_tone_floor(np.zeros(0, dtype=np.float32), SR, []).size == 0


class TestEstimateRt60:
    @pytest.mark.parametrize("rt60", [0.5, 0.8, 1.2])
    def test_recovers_a_known_decay(self, rt60: float) -> None:
        measured = estimate_rt60(decaying_noise(rt60, 4.0), SR)
        assert measured == pytest.approx(rt60, abs=0.2)

    def test_longer_decays_measure_longer(self) -> None:
        short = estimate_rt60(decaying_noise(0.4, 4.0), SR)
        long = estimate_rt60(decaying_noise(1.0, 4.0), SR)
        assert long > short

    def test_is_clamped_to_the_plausible_range(self) -> None:
        low, high = RT60_BOUNDS
        assert estimate_rt60(decaying_noise(10.0, 6.0), SR) == pytest.approx(high)
        assert estimate_rt60(decaying_noise(0.02, 2.0), SR) == pytest.approx(low)

    def test_a_decay_too_slow_to_see_reads_as_long_not_none(self) -> None:
        """A decay slower than the measurement window must not read as 'no room'."""
        slow = estimate_rt60(decaying_noise(10.0, 6.0), SR)
        assert slow > estimate_rt60(decaying_noise(0.5, 6.0), SR)
        assert slow != pytest.approx(RT60_BOUNDS[0])

    def test_white_noise_has_no_decay_to_measure(self) -> None:
        rng = np.random.default_rng(0)
        noise = (0.3 * rng.standard_normal(SR * 3)).astype(np.float32)
        assert estimate_rt60(noise, SR) == pytest.approx(RT60_BOUNDS[0])

    def test_silence_falls_back_to_the_lower_bound(self) -> None:
        assert estimate_rt60(np.zeros(SR, dtype=np.float32), SR) == RT60_BOUNDS[0]

    def test_empty_input_is_safe(self) -> None:
        assert estimate_rt60(np.zeros(0, dtype=np.float32), SR) == RT60_BOUNDS[0]

    def test_short_clip_is_safe(self) -> None:
        assert estimate_rt60(tone(0.01), SR) == RT60_BOUNDS[0]


class TestSyntheticIr:
    def test_length_matches_the_decay_time(self) -> None:
        assert synthetic_ir(0.5, SR).size == int(0.5 * SR)

    def test_has_unit_energy(self) -> None:
        impulse = synthetic_ir(0.8, SR)
        assert float(np.sum(np.square(impulse, dtype=np.float64))) == pytest.approx(
            1.0, abs=1e-4
        )

    def test_decays_over_time(self) -> None:
        impulse = synthetic_ir(1.0, SR)
        head = rms(impulse[: SR // 10])
        tail = rms(impulse[-SR // 10 :])
        assert tail < head * 0.1

    def test_is_deterministic_for_a_seed(self) -> None:
        assert np.array_equal(
            synthetic_ir(0.5, SR, seed=3), synthetic_ir(0.5, SR, seed=3)
        )

    def test_different_seeds_differ(self) -> None:
        assert not np.array_equal(
            synthetic_ir(0.5, SR, seed=1), synthetic_ir(0.5, SR, seed=2)
        )

    def test_zero_rt60_returns_one_sample(self) -> None:
        assert synthetic_ir(0.0, SR).size == 1

    def test_damping_rolls_off_the_top_end(self) -> None:
        """An undamped burst is flat to Nyquist and reads as a metallic hiss."""
        freqs = np.fft.rfftfreq(int(SR), 1.0 / SR)

        def band_db(impulse: np.ndarray, lo: float, hi: float) -> float:
            spectrum = np.abs(np.fft.rfft(impulse))
            mask = (freqs >= lo) & (freqs < hi)
            return float(20 * np.log10(np.sqrt(np.mean(spectrum[mask] ** 2))))

        def tilt(impulse: np.ndarray) -> float:
            """High band relative to low: normalisation-independent, both are unit-energy."""
            return band_db(impulse, 9000, 11000) - band_db(impulse, 200, 1000)

        damped = synthetic_ir(1.0, SR)
        flat = synthetic_ir(1.0, SR, damping_hz=0.0)

        assert flat.size == damped.size
        assert tilt(flat) > -2.0  # essentially flat to the top end
        assert tilt(damped) < tilt(flat) - 6.0


class TestConvolveReverb:
    def test_no_reverb_at_or_above_zero_db(self) -> None:
        """Reverb is opt-in: the default must leave the audio untouched."""
        dry = tone(1.0, amplitude=0.3)
        impulse = synthetic_ir(1.0, SR)
        assert np.array_equal(convolve_reverb(dry, impulse), dry)
        assert np.array_equal(convolve_reverb(dry, impulse, wet_db=0.0), dry)
        assert np.array_equal(convolve_reverb(dry, impulse, wet_db=6.0), dry)

    def test_output_length_is_unchanged(self) -> None:
        dry = tone(2.0)
        assert convolve_reverb(dry, synthetic_ir(1.0, SR), wet_db=-6.0).size == dry.size

    def test_leaves_a_tail_after_the_dry_signal(self) -> None:
        """A short burst followed by silence must ring out under reverb."""
        dry = np.zeros(int(3.0 * SR), dtype=np.float32)
        dry[: int(0.1 * SR)] = tone(0.1, amplitude=0.5)

        wet = convolve_reverb(dry, synthetic_ir(1.0, SR), wet_db=-6.0)
        tail = np.zeros_like(dry)
        tail[int(0.2 * SR) : int(1.0 * SR)] = wet[int(0.2 * SR) : int(1.0 * SR)]

        assert rms(dry[int(0.2 * SR) : int(1.0 * SR)]) < 1e-6
        assert rms(tail) > 0.001

    def test_wet_level_is_controllable(self) -> None:
        dry = np.zeros(int(2.0 * SR), dtype=np.float32)
        dry[: int(0.2 * SR)] = tone(0.2, amplitude=0.5)
        impulse = synthetic_ir(1.0, SR)

        loud = convolve_reverb(dry, impulse, wet_db=-6.0)
        quiet = convolve_reverb(dry, impulse, wet_db=-26.0)
        window = slice(int(0.3 * SR), int(1.0 * SR))

        assert rms(quiet[window]) < rms(loud[window]) * 0.5

    def test_dry_signal_survives(self) -> None:
        dry = tone(1.0, amplitude=0.3)
        wet = convolve_reverb(dry, synthetic_ir(0.5, SR), wet_db=-60.0)
        assert np.allclose(wet, dry, atol=1e-2)

    def test_empty_inputs_are_safe(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        assert convolve_reverb(empty, synthetic_ir(0.5, SR)).size == 0
        assert convolve_reverb(tone(1.0), empty).size == SR


class TestApplyRoom:
    def test_the_default_adds_no_reverb(self) -> None:
        dry = tone(1.0, amplitude=0.3)
        wet, _ = apply_room(dry, decaying_noise(0.9, 3.0), SR)
        assert np.array_equal(wet, dry)

    def test_preserves_length_and_reports_the_decay(self) -> None:
        reference = decaying_noise(0.9, 4.0)
        wet, rt60 = apply_room(tone(2.0), reference, SR, wet_db=-6.0)
        assert wet.size == int(2.0 * SR)
        assert rt60 == pytest.approx(0.9, abs=0.2)

    def test_a_long_tail_is_actually_added_when_asked_for(self) -> None:
        dry = np.zeros(int(3.0 * SR), dtype=np.float32)
        dry[: int(0.2 * SR)] = tone(0.2, amplitude=0.5)
        wet, _ = apply_room(dry, decaying_noise(1.0, 4.0), SR, wet_db=-6.0)
        window = slice(int(0.4 * SR), int(1.2 * SR))
        assert rms(wet[window]) > rms(dry[window]) + 1e-4

    def test_explicit_rt60_overrides_the_measurement(self) -> None:
        _, rt60 = apply_room(tone(1.0), decaying_noise(0.4, 3.0), SR, rt60=1.5)
        assert rt60 == pytest.approx(1.5)

    def test_explicit_rt60_is_still_clamped(self) -> None:
        _, rt60 = apply_room(tone(1.0), tone(1.0), SR, rt60=99.0)
        assert rt60 == pytest.approx(RT60_BOUNDS[1])

    def test_reference_shorter_than_the_decay_is_usable(self) -> None:
        wet, rt60 = apply_room(tone(0.2), tone(0.2), SR)
        assert wet.size == int(0.2 * SR)
        assert RT60_BOUNDS[0] <= rt60 <= RT60_BOUNDS[1]

    def test_output_is_finite(self) -> None:
        wet, _ = apply_room(tone(1.0, amplitude=0.9), decaying_noise(0.6, 2.0), SR, wet_db=-6.0)
        assert np.all(np.isfinite(wet))


class TestRoomDefaults:
    def test_reverb_is_off_and_the_duck_is_shallow(self) -> None:
        assert DEFAULT_ROOM_WET_DB == 0.0
        assert DEFAULT_ROOM_DUCK_DB == 3.0
        assert DEFAULT_ROOM_DUCK_DB < DEFAULT_DUCK_DB
        assert DEFAULT_ROOM_TONE_GAIN_DB == -30.0
        assert ROOM_TONE_GATE_DB < 0.0
        assert ROOM_DAMPING_HZ > 0.0
