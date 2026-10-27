"""Tests for SRT rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.asr import Segment
from pipeline.subtitles import (
    Cue,
    build_srt,
    clean_text,
    cues_from_segments,
    format_timestamp,
    split_cue_text,
    write_srt,
)


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.25, "00:01:01,250"),
            (3661.007, "01:01:01,007"),
            (100 * 3600, "100:00:00,000"),
        ],
    )
    def test_formats_expected_values(self, seconds: float, expected: str) -> None:
        assert format_timestamp(seconds) == expected

    def test_clamps_negative_values(self) -> None:
        assert format_timestamp(-5.0) == "00:00:00,000"

    def test_carries_rounding_into_the_next_second(self) -> None:
        # 59.9996s must not render as ",1000".
        assert format_timestamp(59.9996) == "00:01:00,000"
        assert format_timestamp(0.9996) == "00:00:01,000"

    def test_rounds_half_millisecond(self) -> None:
        assert format_timestamp(1.0004) == "00:00:01,000"
        assert format_timestamp(1.0006) == "00:00:01,001"


class TestCleanText:
    def test_collapses_newlines_and_whitespace(self) -> None:
        assert clean_text("hello\n\n  world\t!") == "hello world !"

    def test_strips_surrounding_space(self) -> None:
        assert clean_text("   spaced   ") == "spaced"

    def test_empty_stays_empty(self) -> None:
        assert clean_text("   \n ") == ""


class TestBuildSrt:
    def test_numbers_cues_from_one(self) -> None:
        srt = build_srt([Cue(0, 1, "first"), Cue(1, 2, "second")])
        assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\nfirst\n")
        assert "2\n00:00:01,000 --> 00:00:02,000\nsecond\n" in srt

    def test_skips_empty_cues_without_breaking_numbering(self) -> None:
        srt = build_srt([Cue(0, 1, "a"), Cue(1, 2, "   "), Cue(2, 3, "c")])
        assert "1\n" in srt and "2\n" in srt
        assert "3\n" not in srt
        assert srt.count("-->") == 2

    def test_can_keep_empty_cues_when_asked(self) -> None:
        srt = build_srt([Cue(0, 1, "   ")], skip_empty=False)
        assert "1\n00:00:00,000 --> 00:00:01,000\n\n" in srt

    def test_repairs_inverted_and_negative_timestamps(self) -> None:
        srt = build_srt([Cue(-3, -1, "text")])
        assert srt.split("\n")[1] == "00:00:00,000 --> 00:00:00,000"

    def test_returns_empty_string_for_no_cues(self) -> None:
        assert build_srt([]) == ""

    def test_multiline_cue_text_is_flattened(self) -> None:
        srt = build_srt([Cue(0, 1, "line one\nline two")])
        assert "line one line two" in srt


class TestWriteSrt:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper" / "subs.srt"
        result = write_srt([Cue(0, 1, "hello")], target)
        assert result == target
        assert target.read_text(encoding="utf-8").startswith("1\n")

    def test_writes_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "fr.srt"
        write_srt([Cue(0, 1, "voilà, c'est écrit")], target)
        assert "voilà" in target.read_text(encoding="utf-8")


class TestSplitCueText:
    def test_splits_on_word_boundaries(self) -> None:
        assert split_cue_text("one two three four five", max_chars=9) == [
            "one two",
            "three",
            "four five",
        ]

    def test_empty_text_yields_no_chunks(self) -> None:
        assert split_cue_text("   ") == []

    def test_a_single_long_word_is_not_cut(self) -> None:
        assert split_cue_text("supercalifragilistic", max_chars=5) == [
            "supercalifragilistic"
        ]


class TestCuesFromSegments:
    def test_long_segment_is_split_inside_its_span(self) -> None:
        segment = Segment(id=0, start=0.0, end=6.0, text="one two three four five six")
        cues = cues_from_segments([segment], max_chars=10)

        assert len(cues) == 3
        assert cues[0].start == pytest.approx(0.0)
        assert cues[-1].end == pytest.approx(6.0)
        assert all(cue.start <= cue.end for cue in cues)

    def test_replacement_text_is_used(self) -> None:
        segment = Segment(id=7, start=0.0, end=2.0, text="hello")
        cues = cues_from_segments([segment], {7: "bonjour"})
        assert [cue.text for cue in cues] == ["bonjour"]

    def test_empty_translation_produces_no_cue(self) -> None:
        segment = Segment(id=1, start=0.0, end=2.0, text="hello")
        assert cues_from_segments([segment], {1: "   "}) == []
