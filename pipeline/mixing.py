"""Mixing the dubbed dialogue back over the original background.

The dubbed track contains only synthesised speech; the background stem holds the
music, ambience and room tone that the MVP used to discard. Adding the two back
together is not enough on its own — a bed left at full level fights the dialogue
and hurts intelligibility, so it is *ducked* under the new speech and restored in
the gaps, following standard dubbing practice.

Both functions here are pure numpy so they can be tested without any media
tooling.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .media import limit_peaks, moving_average

__all__ = ["duck_envelope", "duck_and_mix", "DEFAULT_DUCK_DB", "DEFAULT_RAMP_MS"]

#: How far the background is pulled down while dialogue plays. Enough to hear the
#: voice clearly without the music audibly pumping.
DEFAULT_DUCK_DB = 6.0

#: Ramp time in and out of ducking; too short sounds like a gate, too long lets
#: the music cover the first word.
DEFAULT_RAMP_MS = 150.0


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
