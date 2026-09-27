"""Mixing the dubbed dialogue back over the original background.

The dubbed track contains only synthesised speech; the background stem holds the
music, ambience and room tone that the MVP used to discard. Adding the two back
together is not enough on its own — a bed left at full level fights the dialogue
and hurts intelligibility, so it is *ducked* under the new speech and restored in
the gaps, following standard dubbing practice.

A ducked bed is still not the same as *being there*: an anechoic synthesis over a
soundtrack reads as a voice laid over the picture. The room-matching helpers below
close that gap without touching the voice cloning — a continuous room-tone floor
lifted from the speech stem's own pauses, and a convolutive reverb whose decay is
measured from that same stem (see :func:`estimate_rt60` and
:func:`convolve_reverb`).

Everything here is pure numpy — with at most an FFT — so it can be tested without
any media tooling.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .media import limit_peaks, moving_average

__all__ = [
    "duck_envelope",
    "duck_and_mix",
    "room_tone_floor",
    "estimate_rt60",
    "synthetic_ir",
    "convolve_reverb",
    "apply_room",
    "DEFAULT_DUCK_DB",
    "DEFAULT_RAMP_MS",
    "DEFAULT_ROOM_WET_DB",
    "DEFAULT_ROOM_DUCK_DB",
    "DEFAULT_ROOM_TONE_GAIN_DB",
    "ROOM_TONE_GATE_DB",
    "ROOM_TONE_MAX_BOOST_DB",
    "ROOM_DAMPING_HZ",
    "RT60_BOUNDS",
]

#: How far the background is pulled down while dialogue plays. Enough to hear the
#: voice clearly without the music audibly pumping.
DEFAULT_DUCK_DB = 6.0

#: Ramp time in and out of ducking; too short sounds like a gate, too long lets
#: the music cover the first word.
DEFAULT_RAMP_MS = 150.0

#: Default reverb level relative to the dry voice. **``0`` means no reverb**: the
#: convolutive stage is opt-in, because hearing it is the only way to judge it and
#: the first default (-12 dB) was rejected on a real listen as an audible echo.
#: A negative value adds a tail at that level.
DEFAULT_ROOM_WET_DB = 0.0

#: How far the bed is ducked under dialogue when matching a room. Shallower than
#: :data:`DEFAULT_DUCK_DB`: the dub now shares the source's space, and a deep duck
#: pulls the ambience out from under it.
DEFAULT_ROOM_DUCK_DB = 3.0

#: Level of the room-tone floor. Low enough to be felt rather than heard.
DEFAULT_ROOM_TONE_GAIN_DB = -30.0

#: How far below the loudest part of a stem a frame must sit to count as room tone.
#: Frames louder than this are the speaker (or their decay), and laying those under
#: the dub is what puts a ghost of the original voice behind the new one.
ROOM_TONE_GATE_DB = -30.0

#: Most the surviving room tone may be amplified back up to its target level. A
#: stem with no quiet frames leaves only residue, and amplifying that would invent
#: a room out of noise.
ROOM_TONE_MAX_BOOST_DB = 24.0

#: Corner frequency of the impulse response's high-frequency damping. A real room
#: absorbs treble; an undamped noise burst does not, and reads as a metallic hiss
#: rather than a space.
ROOM_DAMPING_HZ = 4000.0

#: Plausible RT60 range, in seconds. Clamped so a bad estimate cannot turn a
#: living room into a cathedral or a phone booth.
RT60_BOUNDS: tuple[float, float] = (0.2, 3.0)

#: Attenuation used to isolate the speech stem's non-speech parts (room tone and
#: reverb tail) from the voice itself. Deep, but the moving-average ramps keep the
#: crossfade smooth instead of gating the stem abruptly.
_ROOM_TONE_ISOLATION_DB = 45.0


def duck_envelope(
    length: int,
    sample_rate: int,
    regions: Sequence[tuple[float, float]],
    *,
    duck_db: float = DEFAULT_DUCK_DB,
    ramp_ms: float = DEFAULT_RAMP_MS,
) -> np.ndarray:
    """Gain curve to apply to the background bed.

    The curve is at unity between utterances and at ``duck_db`` below unity
    while speech plays, with a moving average producing smooth ramps rather than
    a hard gate (which would be audible as clicking/pumping).

    Args:
        length: Number of samples required.
        sample_rate: Sample rate in Hz.
        regions: ``(start, end)`` seconds of dialogue.
        duck_db: Attenuation while dialogue plays (positive number of dB).
        ramp_ms: Transition time in milliseconds.

    Returns:
        A float32 gain array of ``length`` samples, in ``(0, 1]``.
    """
    if length <= 0:
        return np.zeros(0, dtype=np.float32)

    floor = 10.0 ** (-abs(duck_db) / 20.0)
    gain = np.ones(length, dtype=np.float32)

    for start, end in regions:
        first = max(0, int(round(start * sample_rate)))
        last = min(length, int(round(end * sample_rate)))
        if last > first:
            gain[first:last] = floor

    if ramp_ms > 0:
        window = max(3, int(sample_rate * ramp_ms / 1000.0))
        gain = moving_average(gain, window)

    return np.clip(gain, floor, 1.0).astype(np.float32)


def duck_and_mix(
    dubbed: np.ndarray,
    background: np.ndarray | None,
    regions: Sequence[tuple[float, float]],
    sample_rate: int,
    *,
    duck_db: float = DEFAULT_DUCK_DB,
    ramp_ms: float = DEFAULT_RAMP_MS,
    background_gain_db: float = 0.0,
) -> np.ndarray:
    """Lay the dubbed dialogue over a ducked background bed.

    Args:
        dubbed: The synthesised speech timeline.
        background: The original background stem, or ``None`` to skip it.
        regions: ``(start, end)`` seconds where dialogue plays.
        sample_rate: Sample rate in Hz.
        duck_db: Attenuation applied to the bed under dialogue.
        ramp_ms: Ducking ramp time.
        background_gain_db: Static offset for the bed (e.g. to lift quiet music).

    Returns:
        The combined mono track, limited so it cannot clip.
    """
    speech = np.asarray(dubbed, dtype=np.float32).reshape(-1)
    if background is None:
        return limit_peaks(speech, sample_rate)

    bed = np.asarray(background, dtype=np.float32).reshape(-1)
    if bed.size == 0:
        return limit_peaks(speech, sample_rate)

    length = max(speech.size, bed.size)
    mixed = np.zeros(length, dtype=np.float32)
    mixed[: speech.size] = speech

    bed_padded = np.zeros(length, dtype=np.float32)
    bed_padded[: bed.size] = bed
    bed_padded *= 10.0 ** (background_gain_db / 20.0)

    envelope = duck_envelope(
        length, sample_rate, regions, duck_db=duck_db, ramp_ms=ramp_ms
    )
    mixed += bed_padded * envelope

    return limit_peaks(mixed, sample_rate)


def room_tone_floor(
    speech: np.ndarray,
    sample_rate: int,
    regions: Sequence[tuple[float, float]],
    *,
    gain_db: float = DEFAULT_ROOM_TONE_GAIN_DB,
    ramp_ms: float = DEFAULT_RAMP_MS,
    isolation_db: float = _ROOM_TONE_ISOLATION_DB,
    gate_db: float = ROOM_TONE_GATE_DB,
    max_boost_db: float = ROOM_TONE_MAX_BOOST_DB,
) -> np.ndarray:
    """Continuous ambience bed taken from the *gaps* of the source speech stem.

    The separated speech stem is the only material that carries the room: the
    voice, its reverb tail, and the room tone the voice was masking. Laying its
    non-speech parts under the dub is nearly free, needs no reverb algorithm, and
    glues a dry synthesis into the original space surprisingly well.

    Two things are removed before anything is mixed in, because leaving either
    behind puts a ghost of the *original* voice under the new one:

    * the dialogue itself, via :func:`duck_envelope` at a very deep attenuation —
      the transcript's regions are accurate to a segment, and a line may overrun
      its slot, so regions alone are not enough;
    * anything *loud* anywhere, via a level gate relative to the stem's own peak,
      which is what catches speech the transcription never covered (breaths,
      dropped utterances, talk over music) and the speaker's own decay.

    What survives both is genuine room tone: quiet, and not under dialogue.

    Args:
        speech: The source's speech stem.
        sample_rate: Sample rate in Hz.
        regions: ``(start, end)`` seconds of dialogue.
        gain_db: Level of the resulting floor **relative to the stem's own level**,
            so the knob means the same thing on a quiet and a loud recording. The
            gate below only keeps quiet frames, so an absolute gain would compound
            with that attenuation and bury the room entirely.
        ramp_ms: Crossfade time around each dialogue region.
        isolation_db: How far the voice is attenuated inside dialogue.
        gate_db: How far below the stem's peak a frame must sit to count as room
            tone.

    Returns:
        A float32 bed the same length as ``speech``.
    """
    data = np.asarray(speech, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return data

    keep = duck_envelope(
        data.size, sample_rate, regions, duck_db=isolation_db, ramp_ms=ramp_ms
    )

    # Level gate: a short-window envelope of the stem, thresholded relative to its
    # own peak, so the gate adapts to any recording level.
    window = max(3, int(sample_rate * 0.02))
    envelope = moving_average(np.abs(data), window)
    reference = float(np.percentile(envelope, 95.0))
    if reference > 0.0:
        quiet = (envelope <= reference * 10.0 ** (gate_db / 20.0)).astype(np.float32)
        keep = keep * moving_average(quiet, max(3, int(sample_rate * 0.05)))

    room = np.asarray(data * keep, dtype=np.float64)
    # Renormalise: what survives the gate is quiet by construction, so scale it back
    # up to the requested level relative to the stem as a whole. The boost is capped,
    # because a stem with no quiet frames leaves nothing but numerical residue —
    # scaling *that* to the target would fabricate a room out of noise.
    stem_level = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    room_level = float(np.sqrt(np.mean(np.square(room, dtype=np.float64))))
    if room_level > 0.0 and stem_level > 0.0:
        boost = (stem_level / room_level) * 10.0 ** (gain_db / 20.0)
        boost = min(boost, 10.0 ** (max_boost_db / 20.0))
        room *= boost
    return room.astype(np.float32)


def _frame_energy_db(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Short-time energy in dB, plus the hop in seconds.

    Computed from a cumulative sum rather than a strided window matrix, so a long
    clip costs O(n) memory instead of O(frames × frame).
    """
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    if data.size < frame:
        return np.zeros(0, dtype=np.float64), hop / float(sample_rate)

    count = 1 + (data.size - frame) // hop
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(data))))
    starts = np.arange(count) * hop
    mean_square = (cumulative[starts + frame] - cumulative[starts]) / frame
    # 10*log10 of the mean square: an exponential *amplitude* decay therefore
    # falls at 60/RT60 dB per second, which is the definition RT60 is built on.
    return 10.0 * np.log10(np.maximum(mean_square, 1e-20)), hop / float(sample_rate)


def estimate_rt60(
    samples: np.ndarray,
    sample_rate: int,
    *,
    bounds: tuple[float, float] = RT60_BOUNDS,
    window_s: float = 0.5,
    min_drop_db: float = 5.0,
    max_seconds: float = 20.0,
) -> float:
    """Reverberation time (RT60) estimated from the decay of a speech stem, in
    seconds.

    Measures how fast the energy envelope falls after speech offsets, then takes
    the **median** decay rate: the room's decay, not the fastest transient and not
    the slow drift of a pause.

    The classic Schroeder backward integral is *deliberately not used here*, and
    the reason is measured: applied to a whole speech stem it integrates speech
    activity rather than reverberation, because energy keeps arriving for as long
    as the speaker talks. On this repo's real stems (60 s and 22 s of Demucs
    `vocals`) it saturated at the upper clamp every time — a cathedral for every
    video. A windowed envelope decay does not have that failure mode, because each
    window only ever sees one decay.

    Args:
        samples: Audio that contains a room (the separated speech stem).
        sample_rate: Sample rate in Hz.
        bounds: ``(min, max)`` seconds the result is clamped to.
        window_s: Span over which each decay is measured.
        min_drop_db: A window counts as a decay only if the envelope fell this far.
        max_seconds: Analyse at most this many seconds from the start.

    Returns:
        RT60 in seconds, clamped to ``bounds``. ``bounds[0]`` when no decay could
        be measured, which degrades to "almost no reverb" rather than to a
        cathedral.
    """
    low, high = bounds
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    if data.size == 0 or sample_rate <= 0:
        return float(low)

    data = data[: max(1, int(max_seconds * sample_rate))]
    envelope_db, hop_s = _frame_energy_db(data, sample_rate)
    if envelope_db.size == 0:
        return float(low)

    span = max(1, int(round(window_s / hop_s)))
    if envelope_db.size <= span:
        return float(low)

    starts = envelope_db[:-span]
    drops = starts - envelope_db[span:]
    # A window that begins in the noise floor has no decay to measure, and would
    # drag the estimate towards zero (i.e. towards an unmeasurably long RT60).
    audible = starts > float(envelope_db.max()) - 45.0
    decays = drops[audible & (drops >= min_drop_db)]
    if decays.size == 0:
        # Nothing fell fast enough for a single window to see it. If the envelope
        # still spans a wide dynamic range, the decay is simply slower than the
        # window: the honest answer is "long", not "none". Answering "none" here
        # would silently strip the reverb from the most reverberant sources.
        if float(envelope_db.max()) - float(envelope_db.min()) >= 20.0:
            return float(high)
        return float(low)

    # The *room* decay is the fast, sustained one. Taking the median instead folds
    # in the shallow dips between words — the voice continues through those — and
    # overestimates badly: measured on this repo's own stems it saturated at the
    # upper clamp (2.4–3.0 s) for ordinary talking-head footage. The 75th
    # percentile is still a robust statistic, but it tracks the decays that follow
    # speech offsets rather than the ones inside a sentence.
    rate_db_per_s = float(np.percentile(decays, 75.0)) / window_s
    if rate_db_per_s <= 0.0:
        return float(low)
    return float(min(high, max(low, 60.0 / rate_db_per_s)))


def synthetic_ir(
    rt60: float,
    sample_rate: int,
    *,
    seed: int = 0,
    floor_db: float = -60.0,
    damping_hz: float = ROOM_DAMPING_HZ,
) -> np.ndarray:
    """A synthetic room impulse response: damped noise under an exponential decay.

    The checkpoint cannot synthesise reverberant speech, so the room has to be
    added afterwards. A decaying noise burst is the standard stand-in for a
    measured impulse response: no measurement step, no extra dependency, and RT60
    becomes a knob.

    The noise is **low-passed** before it decays. Real rooms absorb treble, and an
    undamped burst is spectrally flat up to Nyquist, which does not sound like a
    room at all — it sounds like a metallic hiss laid over the voice.

    The response is normalised to unit energy, which makes the ``wet_db`` argument
    of :func:`convolve_reverb` a stable level rather than one that depends on how
    long the tail happens to be.

    Args:
        rt60: Reverberation time in seconds.
        sample_rate: Sample rate in Hz.
        seed: Fixed by default, so a rerun sounds identical and tests are stable.
        floor_db: Level the envelope has decayed to at ``rt60``.
        damping_hz: Corner frequency of the high-frequency damping. ``0`` disables
            it (flat noise, kept for callers that want the raw response).

    Returns:
        A float32 impulse response.
    """
    rt60 = max(0.0, float(rt60))
    length = max(1, int(round(rt60 * sample_rate)))
    time = np.arange(length, dtype=np.float64) / float(sample_rate)
    envelope = np.power(10.0, (floor_db / 20.0) * (time / max(rt60, 1e-6)))

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(length)
    if damping_hz and 0.0 < damping_hz < sample_rate / 2.0:
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(length, 1.0 / sample_rate)
        # A gentle first-order roll-off: -3 dB at the corner, 6 dB/octave above it.
        noise = np.fft.irfft(spectrum / np.sqrt(1.0 + (freqs / damping_hz) ** 2), length)

    impulse = (noise * envelope).astype(np.float32)
    energy = float(np.sum(np.square(impulse, dtype=np.float64)))
    if energy > 0.0:
        impulse = (impulse / np.sqrt(energy)).astype(np.float32)
    return impulse


def convolve_reverb(
    samples: np.ndarray,
    impulse: np.ndarray,
    *,
    wet_db: float = DEFAULT_ROOM_WET_DB,
) -> np.ndarray:
    """Add a reverb tail to ``samples``, returning the same number of samples.

    Uses an FFT convolution, which is O(n log n) and therefore cheap enough to run
    once over a whole timeline rather than once per line. Convolving the assembled
    track is also more correct than convolving line by line: the tail decays across
    the pauses between utterances, exactly as it would in the room.

    The output is deliberately kept the same length as the input, so the timeline
    geometry — and the dub's duration — is unchanged; only the very last tail is
    cut, and only when the video ends exactly on a word.

    Args:
        samples: Dry audio.
        impulse: Room impulse response (see :func:`synthetic_ir`).
        wet_db: Level of the reverb relative to the dry signal. ``0`` or positive
            means **no reverb**: the stage is opt-in, so a run that did not ask for
            it complicates nothing. Pass a negative value to add a tail.

    Returns:
        Dry audio plus the wet tail, same length as ``samples``.
    """
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    ir = np.asarray(impulse, dtype=np.float32).reshape(-1)
    if data.size == 0 or ir.size == 0 or wet_db >= 0.0:
        return data

    length = data.size + ir.size - 1
    size = 1 << max(0, length - 1).bit_length()
    wet = np.fft.irfft(
        np.fft.rfft(data, size) * np.fft.rfft(ir, size), size
    )[:length].astype(np.float32)

    mixed = np.zeros(length, dtype=np.float32)
    mixed[: data.size] = data
    mixed += (10.0 ** (wet_db / 20.0)) * wet
    return mixed[: data.size].astype(np.float32)


def apply_room(
    samples: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    wet_db: float = DEFAULT_ROOM_WET_DB,
    rt60: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Convolve ``samples`` with the room measured from ``reference``.

    Args:
        samples: Dry audio to place in the room (the assembled dub).
        reference: Audio that contains the room (the separated speech stem).
        sample_rate: Sample rate in Hz.
        wet_db: Level of the reverb tail relative to the dry voice.
        rt60: Use this decay instead of measuring one from ``reference``.
        seed: Seed for the synthetic impulse response.

    Returns:
        ``(wet_samples, rt60_used)``, so the caller can report what it measured.
    """
    low, high = RT60_BOUNDS
    measured = (
        float(rt60)
        if rt60 is not None
        else estimate_rt60(reference, sample_rate)
    )
    measured = float(min(high, max(low, measured)))
    if wet_db >= 0.0:
        # Reverb is opt-in. The decay is still measured, because the report states
        # it and a user may turn the tail on from it - but no IR is built and no
        # convolution runs, so an un-reverbed job costs nothing.
        return np.asarray(samples, dtype=np.float32).reshape(-1), measured
    impulse = synthetic_ir(measured, sample_rate, seed=seed)
    return convolve_reverb(samples, impulse, wet_db=wet_db), measured
