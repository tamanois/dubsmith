"""The pre-synthesis word check: reword before spending GPU, not after.

A translation that cannot fit its window means the mixer must delete words, so the
only answers are a shorter sentence or a lost ending. Asking for the shorter
sentence *before* synthesising is the difference between one synthesis pass and
two, and it is what lets the fill/trim report stay at zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from pipeline.asr import Segment
from pipeline.translate import TranslationRequest

from .test_pipeline import ShorteningTranslator, refine_pipeline

#: Sixteen words against the twelve the widest window in these tests can hold.
LONG = " ".join(f"mot{n}" for n in range(1, 17))
SHORT = "Court."

#: Words per second the fake engine reports for the delivery-rate probe, so the
#: budget arithmetic below is exact.
WPS = 4.0
#: Segment 0 spans 2.0s with a 0.5s pause after it, so it may occupy 2.5s. At
#: 4 words/s and the default 0.85 budget tempo that is 11.76 -> 12 words.
BUDGET_LINE_0 = 12


class LongTranslator(ShorteningTranslator):
    """Translates the first line far past its budget, and the rest briefly."""

    def __init__(self, *, every_line: bool = False) -> None:
        super().__init__(shortenings={0: "Bref.", 1: "Bref."})
        self.every_line = every_line
        self.translate_calls = 0

    def translate(
        self,
        requests: Sequence[TranslationRequest],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        self.translate_calls += 1
        return {
            request.id: (LONG if self.every_line or request.id == 0 else SHORT)
            for request in requests
        }


class RaisingTranslator(LongTranslator):
    """A translator whose rewrite endpoint is down."""

    def shorten(self, requests: Sequence[Any], *, target_language: str) -> dict[int, str]:
        self.shorten_calls += 1
        self.requested = list(requests)
        raise RuntimeError("translator unreachable")


class NoRewriteTranslator(LongTranslator):
    """Went through the protocol's ``translate`` only."""

    shorten = None  # type: ignore[assignment]


def prefit_pipeline(tmp_path: Path, translator: Any, **kwargs: Any):
    """A pipeline whose lines are short enough that only the check can matter."""
    return refine_pipeline(
        tmp_path, rows=[[1.0, 1.0]], translator=translator, **kwargs
    )


class TestPreFitCheck:
    def test_an_overlong_line_is_reworded_before_synthesis(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        pipeline, _, tts, _ = prefit_pipeline(tmp_path, translator)
        report = pipeline.run().report

        assert translator.shorten_calls == 1
        assert [request.id for request in translator.requested] == [0]
        assert report["prefit"]["attempted"] is True
        assert report["prefit"]["offenders"] == 1
        assert report["prefit"]["improved"] == 1
        assert report["prefit"]["llm_calls"] == 1
        # One synthesis pass, already carrying the short wording: no line was
        # spoken twice, which is the whole point of checking before synthesis.
        assert tts.synth_calls == 1
        assert tts.text_batches[0] == ["Bref.", SHORT]
        assert report["trimmed_seconds_total"] == 0.0

    def test_the_ceiling_and_the_window_come_from_the_probe(self, tmp_path: Path) -> None:
        translator = LongTranslator(every_line=True)
        pipeline, _, _, _ = prefit_pipeline(tmp_path, translator)
        pipeline.run()

        first, last = translator.requested
        # line 0: 2.0s slot + 0.5s pause; line 1: the last line, so no pause after
        # it may be borrowed.
        assert first.available_seconds == pytest.approx(2.5)
        assert last.available_seconds == pytest.approx(2.0)
        assert first.max_words == BUDGET_LINE_0
        assert last.max_words == 9
        # Predicted, not measured: nothing has been spoken yet.
        assert first.spoken_seconds == pytest.approx(len(LONG.split()) / WPS)
        assert first.source == "A long English sentence."

    def test_a_line_inside_its_budget_is_left_alone(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        translator.translate = lambda requests, *, source_language, target_language: {
            request.id: SHORT for request in requests
        }
        pipeline, _, tts, _ = prefit_pipeline(tmp_path, translator)
        report = pipeline.run().report

        assert translator.shorten_calls == 0
        assert report["prefit"]["attempted"] is False
        assert report["prefit"]["offenders"] == 0
        assert report["prefit"]["words_kept"] == 1.0
        assert tts.synth_calls == 1

    def test_a_rewrite_that_is_not_shorter_is_discarded(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        translator.shortenings = {0: LONG + " encore"}
        pipeline, _, tts, _ = prefit_pipeline(tmp_path, translator)
        report = pipeline.run().report

        assert translator.shorten_calls == 1
        assert report["prefit"]["improved"] == 0
        assert report["prefit"]["words_kept"] == 1.0
        # The original wording is what gets spoken.
        assert tts.text_batches[0][0] == LONG

    def test_a_blank_rewrite_is_discarded(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        translator.shortenings = {0: "   "}
        pipeline, _, _, _ = prefit_pipeline(tmp_path, translator)
        report = pipeline.run().report

        assert report["prefit"]["improved"] == 0
        assert report["prefit"]["words_kept"] == 1.0

    def test_a_failing_rewrite_is_not_fatal(self, tmp_path: Path) -> None:
        translator = RaisingTranslator()
        pipeline, _, tts, _ = prefit_pipeline(tmp_path, translator)
        report = pipeline.run().report

        assert report["prefit"]["attempted"] is True
        assert report["prefit"]["improved"] == 0
        assert report["prefit"]["llm_calls"] == 0
        # The run continues with the translation it has, rather than failing.
        assert tts.synth_calls == 1
        assert tts.text_batches[0][0] == LONG

    def test_a_translator_without_rewriting_is_skipped(self, tmp_path: Path) -> None:
        translator = NoRewriteTranslator()
        pipeline, _, tts, _ = prefit_pipeline(tmp_path, translator)
        report = pipeline.run().report

        assert report["prefit"]["attempted"] is False
        assert report["prefit"]["offenders"] == 1
        assert tts.text_batches[0][0] == LONG

    def test_the_reworded_line_reaches_the_subtitles(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        pipeline, _, _, _ = prefit_pipeline(tmp_path, translator)
        result = pipeline.run()

        assert result.target_srt is not None
        text = result.target_srt.read_text(encoding="utf-8")
        assert "Bref." in text
        assert LONG not in text

    def test_the_raw_translation_is_kept_for_provenance(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        pipeline, _, _, _ = prefit_pipeline(tmp_path, translator)
        pipeline.run()

        payload = json.loads(
            (pipeline.job_dir / "03_script.json").read_text(encoding="utf-8")
        )
        # What the model said stays on disk; what was spoken is recorded beside it.
        assert payload["lines"][0]["target"] == LONG
        prefit = json.loads(
            (pipeline.job_dir / "03d_prefit.json").read_text(encoding="utf-8")
        )
        assert prefit["lines"] == {"0": "Bref."}

    def test_a_second_run_reuses_the_recorded_rewrites(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        pipeline, _, tts, _ = prefit_pipeline(tmp_path, translator)
        first = pipeline.run().report

        second = pipeline.run().report

        # Nothing re-derived: the rewrite is on disk, and the translation and the
        # synthesis that used it are both cached.
        assert translator.shorten_calls == 1
        assert tts.synth_calls == 1
        assert second["prefit"] == first["prefit"]
        assert second["prefit"]["improved"] == 1


class TestPreFitAndRepair:
    """The two rewriting passes must not fight over the same line."""

    def test_the_measured_repair_wins_over_the_prediction(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        # 3.5s of audio in a 2.0s slot, so the measured pass still has work to do
        # after the pre-fit check has already reworded the line.
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], translator=translator
        )
        pipeline.run()

        spoken = json.loads(
            (pipeline.job_dir / "03b_spoken.json").read_text(encoding="utf-8")
        )
        assert spoken["0"] == "Bref."
        prefit = json.loads(
            (pipeline.job_dir / "03d_prefit.json").read_text(encoding="utf-8")
        )
        assert prefit["lines"]["0"] == "Bref."

    def test_a_repair_only_line_still_reaches_the_subtitles(self, tmp_path: Path) -> None:
        translator = LongTranslator()
        translator.translate = lambda requests, *, source_language, target_language: {
            request.id: SHORT for request in requests
        }
        # Both lines are inside budget, so only the measured pass rewrites.
        translator.shortenings = {0: "Encore plus bref."}
        pipeline, _, _, _ = refine_pipeline(
            tmp_path, rows=[[3.5, 1.0], [1.8]], translator=translator
        )
        result = pipeline.run()

        assert result.target_srt is not None
        assert "Encore plus bref." in result.target_srt.read_text(encoding="utf-8")


def test_the_probe_text_is_target_language_not_source() -> None:
    """The delivery rate is measured on target-language words, never the donor's."""
    from pipeline.orchestrate import _probe_text

    text = _probe_text(None, "French")
    assert text.strip()
    assert text.split()[:3] == ["Une", "parole", "claire"]
    assert len(text.split()) <= 40


def test_the_probe_text_prefers_the_accent_clip_transcript() -> None:
    from pipeline.orchestrate import _probe_text

    assert _probe_text((None, "Bonjour, je suis Joel."), "French") == (
        "Bonjour, je suis Joel."
    )
    assert _probe_text((None, ""), "French").startswith("Une parole")


def test_the_probe_text_is_capped() -> None:
    from pipeline.orchestrate import _probe_text

    long_text = " ".join(str(index) for index in range(200))
    assert len(_probe_text((None, long_text), "French").split()) == 40


class TestBudgetsFromWindows:
    """The window arithmetic, pinned against the mixer's own rule."""

    def test_the_budget_is_the_window_at_the_budget_tempo(self) -> None:
        from pipeline.orchestrate import _budget_words

        assert _budget_words(2.5, WPS, 0.85) == BUDGET_LINE_0
        assert _budget_words(2.0, WPS, 0.85) == 9

    def test_a_slower_voice_gets_fewer_words(self) -> None:
        from pipeline.orchestrate import _budget_words

        assert _budget_words(2.5, 3.0, 0.85) < _budget_words(2.5, 4.0, 0.85)

    def test_an_empty_line_is_never_the_ceiling(self) -> None:
        from pipeline.orchestrate import _budget_words

        assert _budget_words(0.1, WPS, 0.85) == 1

    def test_unusable_inputs_have_no_budget(self) -> None:
        from pipeline.orchestrate import _budget_words

        assert _budget_words(0.0, WPS, 0.85) == 0
        assert _budget_words(2.0, 0.0, 0.85) == 0


def test_a_window_covers_the_pause_before_the_next_line() -> None:
    from pipeline.orchestrate import _available_seconds

    lines = [
        Segment(id=0, start=0.0, end=2.0, text="a"),
        Segment(id=1, start=2.5, end=4.5, text="b"),
    ]
    assert _available_seconds(lines, 0) == pytest.approx(2.5)
    # The last line has nothing after it to borrow from.
    assert _available_seconds(lines, 1) == pytest.approx(2.0)


def test_an_overlay_line_may_not_use_a_gap() -> None:
    """An overlay shares its slot with the dominant line, so it cannot spill."""
    from pipeline.orchestrate import _available_seconds

    lines = [
        Segment(id=0, start=0.0, end=2.0, text="a"),
        Segment(id=1, start=2.5, end=4.5, text="b", overlay=True),
    ]
    assert _available_seconds(lines, 1) == pytest.approx(2.0)
