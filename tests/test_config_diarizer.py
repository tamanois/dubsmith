"""Config tests for multi-speaker dubbing options."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import DEFAULT_MIN_SPEAKER_S, JobConfig
from pipeline.diarize import DEFAULT_THRESHOLD
from pipeline.errors import ConfigError


def job(**overrides: object) -> JobConfig:
    payload: dict[str, object] = {"audio": Path("a.wav")}
    payload.update(overrides)
    return JobConfig(**payload)  # type: ignore[arg-type]


class TestDiarizerOption:
    """`--diarizer` decides whether the video is split into voices."""

    def test_auto_is_the_default(self) -> None:
        assert job().diarizer == "auto"

    def test_the_default_threshold_and_minimum_are_the_calibrated_ones(self) -> None:
        resolved = job()
        assert resolved.resolved_speaker_threshold == DEFAULT_THRESHOLD
        assert resolved.min_speaker_seconds == DEFAULT_MIN_SPEAKER_S

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("auto", "auto"),
            ("embedding", "auto"),
            ("on", "auto"),
            ("true", "auto"),
            ("AUTO", "auto"),
            ("none", "none"),
            ("off", "none"),
            ("false", "none"),
            ("single", "none"),
            ("NONE", "none"),
        ],
    )
    def test_aliases(self, raw: str, expected: str) -> None:
        assert job(diarizer=raw).diarizer == expected

    def test_an_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown diarizer"):
            job(diarizer="telepathy")

    def test_diarization_is_on_by_default(self) -> None:
        assert job().multi_speaker is True

    def test_off_switches_multi_speaker_off(self) -> None:
        assert job(diarizer="none").multi_speaker is False

    def test_threshold_is_honoured(self) -> None:
        assert job(speaker_threshold=0.55).resolved_speaker_threshold == 0.55

    @pytest.mark.parametrize("bad", [0.0, -0.1, 2.5])
    def test_a_nonsensical_threshold_is_rejected(self, bad: float) -> None:
        with pytest.raises(ConfigError, match="--speaker-threshold"):
            job(speaker_threshold=bad)

    def test_minimum_speaker_length_is_honoured(self) -> None:
        assert job(min_speaker_seconds=5.0).min_speaker_seconds == 5.0

    def test_zero_minimum_is_allowed(self) -> None:
        # Meaningful: keep every cluster, even ones too short to clone from.
        assert job(min_speaker_seconds=0.0).min_speaker_seconds == 0.0

    @pytest.mark.parametrize("bad", [-1.0, 200.0])
    def test_an_impossible_minimum_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError):
            job(min_speaker_seconds=bad)


class TestDiarizerFingerprint:
    """Changing the voices must not reuse another setting's audio."""

    def test_diarization_settings_change_the_fingerprint(self) -> None:
        base = job().fingerprint()
        assert base != job(diarizer="none").fingerprint()
        assert base != job(speaker_threshold=0.55).fingerprint()
        assert base != job(min_speaker_seconds=5.0).fingerprint()

    def test_an_explicit_threshold_hashes_by_its_raw_value(self) -> None:
        # fingerprint() hashes the raw fields, so restating a default is treated as
        # a different setting. Conservative on purpose: a needless rerun costs time,
        # a stale cache costs correctness.
        assert (
            job().fingerprint()
            != job(speaker_threshold=DEFAULT_THRESHOLD).fingerprint()
        )


class TestOverlapOption:
    """`--overlap` decides what happens to a segment carrying two voices."""

    def test_dominant_is_the_default_and_means_no_separation(self) -> None:
        resolved = job()
        assert resolved.overlap_mode == "dominant"
        assert resolved.overlap_separation is False

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("dominant", "dominant"),
            ("keep", "dominant"),
            ("off", "dominant"),
            ("separate", "separate"),
            ("split", "separate"),
            ("on", "separate"),
            ("SEPARATE", "separate"),
        ],
    )
    def test_mode_aliases(self, raw: str, expected: str) -> None:
        assert job(overlap_mode=raw).overlap_mode == expected

    def test_separate_enables_separation(self) -> None:
        assert job(overlap_mode="separate").overlap_separation is True

    def test_an_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown overlap mode"):
            job(overlap_mode="telepathy")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("auto", "auto"),
            ("sepformer", "sepformer"),
            ("speechbrain", "sepformer"),
            ("none", "none"),
            ("off", "none"),
        ],
    )
    def test_separator_aliases(self, raw: str, expected: str) -> None:
        assert job(overlap_separator=raw).overlap_separator == expected

    def test_an_unknown_separator_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown overlap separator"):
            job(overlap_separator="magic")

    def test_overlap_settings_change_the_fingerprint(self) -> None:
        base = job().fingerprint()
        assert base != job(overlap_mode="separate").fingerprint()
        assert base != job(overlap_separator="none").fingerprint()
