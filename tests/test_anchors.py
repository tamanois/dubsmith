"""Tests for the accent-anchor analysis and cache.

No model is loaded: the analysis is pure and the synthesiser is faked, so these
cover the boundaries (empty input, unknown languages, cache hits/misses) that a
generated anchor depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.anchors import (
    ANCHOR_TEXT,
    AnchorSpec,
    anchor_paths,
    ensure_anchor,
    instruct_for,
    list_anchors,
    measure_gender,
    measure_rate,
    rate_band,
)
from pipeline.asr import Segment, Word
from pipeline.errors import TTSError

SAMPLE_RATE = 24_000


class FakeDesignEngine:
    """Records its calls and returns a short deterministic clip."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str, str]] = []

    def generate(
        self, text: str, instruct: str, language: str
    ) -> tuple[np.ndarray, int]:
        self.calls.append((text, instruct, language))
        seconds = 1.5
        t = np.linspace(0.0, seconds, int(seconds * self.sample_rate), endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), self.sample_rate


class TestRateBand:
    @pytest.mark.parametrize(
        ("wps", "expected"),
        [
            (0.0, "normal"),  # not measured
            (2.0, "slow"),
            (3.0, "normal"),  # the boundary is inclusive
            (3.9, "normal"),
            (4.5, "normal"),
            (5.5, "fast"),
        ],
    )
    def test_bands(self, wps: float, expected: str) -> None:
        assert rate_band(wps) == expected


class TestMeasureRate:
    def test_uses_word_timestamps_when_present(self) -> None:
        # Two words over one second of speech -> 2.0 words/s, ignoring the pause.
        segment = Segment(
            id=0,
            start=0.0,
            end=3.0,
            text="hello world",
            words=[
                Word(word="hello", start=0.0, end=0.5),
                Word(word="world", start=0.5, end=1.0),
            ],
        )
        assert measure_rate([segment]) == pytest.approx(2.0)

    def test_falls_back_to_segment_duration(self) -> None:
        segment = Segment(id=0, start=0.0, end=2.0, text="one two three four")
        assert measure_rate([segment]) == pytest.approx(2.0)

    def test_empty_input_has_no_opinion(self) -> None:
        assert measure_rate([]) == 0.0
        assert measure_rate([Segment(id=0, start=1.0, end=1.0, text="x")]) == 0.0


class TestMeasureGender:
    def test_low_pitch_reads_male(self) -> None:
        t = np.linspace(0.0, 1.0, SAMPLE_RATE, endpoint=False)
        clip = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
        assert measure_gender(clip, SAMPLE_RATE) == "m"

    def test_high_pitch_reads_female(self) -> None:
        t = np.linspace(0.0, 1.0, SAMPLE_RATE, endpoint=False)
        clip = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        assert measure_gender(clip, SAMPLE_RATE) == "f"

    def test_short_clip_is_unknown(self) -> None:
        assert measure_gender(np.zeros(100, dtype=np.float32), SAMPLE_RATE) == "u"

    def test_silence_is_unknown_or_a_value(self) -> None:
        # Never raises, whatever it decides.
        assert measure_gender(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE) in {
            "f",
            "m",
            "u",
        }


class TestInstruct:
    def test_plain_instruct_names_the_language(self) -> None:
        text = instruct_for("French")
        assert "French" in text
        assert "voice" not in text  # no gender hint
        assert "pace" not in text  # no rate hint

    def test_gender_and_rate_are_composed(self) -> None:
        text = instruct_for("German", rate_band="fast", gender="f")
        assert "German" in text
        assert "female" in text
        assert "brisk" in text

    def test_unknown_gender_is_ignored(self) -> None:
        assert instruct_for("French", gender="u") == instruct_for("French")


class TestAnchorSpec:
    def test_plain_key_is_the_language_code(self) -> None:
        assert AnchorSpec("French").key == "fr"

    def test_matched_key_carries_the_descriptors(self) -> None:
        assert AnchorSpec("French", rate_band="slow", gender="m").key == "fr_slow_m"

    def test_unknown_gender_is_not_in_the_key(self) -> None:
        assert AnchorSpec("French", rate_band="fast", gender="u").key == "fr_fast"

    def test_text_is_the_language_passage(self) -> None:
        assert AnchorSpec("French").text == ANCHOR_TEXT["French"]

    def test_unknown_language_is_rejected(self) -> None:
        with pytest.raises(TTSError, match="anchor text"):
            _ = AnchorSpec("Klingon").text


class TestEnsureAnchor:
    def test_generates_and_caches_on_a_miss(self, tmp_path: Path) -> None:
        engine = FakeDesignEngine()
        wav, txt = ensure_anchor(tmp_path, AnchorSpec("French"), lambda: engine)

        assert len(engine.calls) == 1
        assert wav.is_file() and txt.is_file()
        assert txt.read_text(encoding="utf-8") == ANCHOR_TEXT["French"]
        metadata = json.loads((tmp_path / "fr.json").read_text(encoding="utf-8"))
        assert metadata["language"] == "French"
        assert metadata["seconds"] > 0

    def test_a_hit_never_builds_the_engine(self, tmp_path: Path) -> None:
        engine = FakeDesignEngine()
        ensure_anchor(tmp_path, AnchorSpec("French"), lambda: engine)

        def boom() -> FakeDesignEngine:  # pragma: no cover - must not run
            raise AssertionError("engine should not be built on a cache hit")

        wav, _txt = ensure_anchor(tmp_path, AnchorSpec("French"), boom)
        assert wav.is_file()
        assert len(engine.calls) == 1

    def test_force_regenerates(self, tmp_path: Path) -> None:
        engine = FakeDesignEngine()
        ensure_anchor(tmp_path, AnchorSpec("French"), lambda: engine)
        ensure_anchor(tmp_path, AnchorSpec("French"), lambda: engine, force=True)
        assert len(engine.calls) == 2

    def test_matched_and_plain_anchors_coexist(self, tmp_path: Path) -> None:
        engine = FakeDesignEngine()
        plain, _ = ensure_anchor(tmp_path, AnchorSpec("French"), lambda: engine)
        matched, _ = ensure_anchor(
            tmp_path, AnchorSpec("French", rate_band="fast"), lambda: engine
        )
        assert plain != matched
        assert list_anchors(tmp_path)


class TestListAnchors:
    def test_empty_when_missing(self, tmp_path: Path) -> None:
        assert list_anchors(tmp_path / "nope") == []

    def test_reports_seconds(self, tmp_path: Path) -> None:
        ensure_anchor(tmp_path, AnchorSpec("German"), lambda: FakeDesignEngine())
        entries = list_anchors(tmp_path)
        assert [key for key, _path, _seconds in entries] == ["de"]
        assert entries[0][2] == pytest.approx(1.5, abs=0.05)


def test_anchor_paths_do_not_create_files(tmp_path: Path) -> None:
    wav, txt = anchor_paths(tmp_path, AnchorSpec("Spanish"))
    assert not wav.exists() and not txt.exists()
    assert wav.name == "es.wav"
