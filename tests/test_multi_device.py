"""Spreading one job's synthesis over several devices.

The engine is injected, so the *planning* - how many pieces a stage is cut into,
which voice each piece carries, what the report says about it - is exercised
end-to-end without a GPU. That an injected engine keeps the run inline is itself
part of the contract: a fake engine cannot cross a process boundary, and neither
can a prompt.

The pool's own process handling lives in ``test_parallel.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import orchestrate
from pipeline.asr import Transcript
from pipeline.parallel import InlineRunner

from .test_multi_speaker import two_speaker_transcript
from .test_pipeline import (
    FakeDiarizer,
    FakeTranslator,
    FakeVoiceDesignEngine,
    make_pipeline,
)

#: Two of the same device: the fan-out is about *workers*, not about owning a
#: second card, and this is how it is tested on a machine with one processor.
TWO = ["cpu", "cpu"]

#: Four segments, two voices alternating, so each group has more than one member.
ALTERNATING = ["SPK_0", "SPK_1", "SPK_0", "SPK_1"]


class TestSingleDeviceIsUnchanged:
    """``--device`` alone must behave exactly as it did before ``--devices``."""

    def test_one_group_is_one_task(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, tts = make_pipeline(tmp_path, simple_transcript, refine=False)
        report = pipeline.run().report
        assert tts.synth_calls == 1
        # Nothing to say about a fan-out that did not happen, so the block is
        # byte-for-byte what it was before the feature existed.
        assert "devices" not in report["tts"]

    def test_an_injected_engine_is_never_handed_to_a_worker(
        self,
        tmp_path: Path,
        simple_transcript: Transcript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        constructed: list[object] = []

        def _record(*args: object, **kwargs: object) -> None:
            constructed.append(args)

        monkeypatch.setattr(orchestrate, "ProcessRunner", _record)
        pipeline, _, _, tts = make_pipeline(
            tmp_path, simple_transcript, devices=TWO, refine=False
        )
        pipeline.run()

        # A fake engine cannot be pickled, and neither can a built clone prompt.
        # Asking for two devices with one injected is therefore still this
        # process's business - which is also what lets the tests below drive the
        # fan-out planning for real.
        assert constructed == []
        assert pipeline._tts is tts


class TestFanOutPlanning:
    """How the stage is cut up, which is the whole balancing policy."""

    def test_one_voice_is_cut_into_a_piece_per_device(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, tts = make_pipeline(
            tmp_path, simple_transcript, devices=TWO, refine=False
        )
        pipeline.run()
        # Three lines, two devices: two tasks. Without the split the second card
        # would sit idle for the whole stage, which is the single-voice case the
        # split exists for.
        assert tts.synth_calls == 2

    def test_two_voices_are_one_task_each(self, tmp_path: Path) -> None:
        pipeline, _, _, tts = make_pipeline(
            tmp_path,
            two_speaker_transcript(),
            labeler=FakeDiarizer(ALTERNATING),
            devices=TWO,
            refine=False,
        )
        pipeline.run()
        # One task per voice, not two: a speaker's lines are only cut up when there
        # are fewer speakers than devices.
        assert tts.synth_calls == 2

    def test_the_blend_stage_is_one_task_per_voice(self, tmp_path: Path) -> None:
        pipeline, _, _, tts = make_pipeline(
            tmp_path,
            two_speaker_transcript(),
            labeler=FakeDiarizer(ALTERNATING),
            anchor_engine=FakeVoiceDesignEngine(),
            accent_mode="native",
            devices=TWO,
            refine=False,
        )
        pipeline.run()
        # The blend is per voice and the voices are independent, so this needs no
        # splitting at all - and it is the stage that costs the most on a
        # many-voice job.
        assert tts.blend_calls == 2

    def test_one_speaker_is_one_probe(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, tts = make_pipeline(
            tmp_path, simple_transcript, devices=TWO, refine=False
        )
        pipeline.run()
        # The probe is per *prompt*, not per device: measuring the same voice twice
        # would buy nothing and would spend a render per card.
        assert tts.rate_calls == 1


class TestTheRunStillCompletes:
    """The artifacts are the point, whatever the work was spread over."""

    def test_every_artifact_appears_with_two_devices(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, tts = make_pipeline(tmp_path, simple_transcript, devices=TWO)
        result = pipeline.run()

        assert result.dub_track.is_file()
        assert (result.job_dir / "04_dubbed" / ".done").is_file()
        # Every line came back through the pool and was written by the parent.
        written = sorted((result.job_dir / "04_dubbed").glob("seg_*.wav"))
        assert written
        # The parent never held an engine, so the rate it wrote the files with has
        # to have come back from the tasks.
        assert tts.sample_rate == 24_000

    def test_the_report_names_the_devices(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, _ = make_pipeline(
            tmp_path, simple_transcript, devices=TWO, refine=False
        )
        report = pipeline.run().report

        assert report["tts"]["devices"] == TWO
        # The single-card stages took the first device, which is what the block
        # that describes precision and capability has always described.
        assert report["device"]["requested"] == "cpu"
        assert report["device"]["resolved"] == "cpu"

    def test_the_counters_cover_both_workers(
        self, tmp_path: Path, simple_transcript: Transcript
    ) -> None:
        pipeline, _, _, tts = make_pipeline(
            tmp_path, simple_transcript, devices=TWO, refine=False
        )
        stats = pipeline.run().report["tts"]
        # Two in-process workers appear as one engine here, so the counters are the
        # engine's: what matters is that the fan-out did not lose them.
        assert stats["batches"] == tts.synth_calls


class _ExplodingTranslator(FakeTranslator):
    """A translator that fails *after* the fan-out stage has been built."""

    def translate(
        self, requests, *, source_language, target_language
    ) -> dict[int, str]:
        raise RuntimeError("translation is down")


class TestTeardown:
    """A failed job must not leave workers holding several GB each."""

    def test_the_runner_is_closed_when_a_later_stage_fails(
        self,
        tmp_path: Path,
        simple_transcript: Transcript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed: list[bool] = []

        class _RecordingRunner(InlineRunner):
            def close(self) -> None:
                closed.append(True)
                super().close()

        monkeypatch.setattr(orchestrate, "InlineRunner", _RecordingRunner)
        pipeline, _, _, _ = make_pipeline(
            tmp_path,
            simple_transcript,
            translator=_ExplodingTranslator(),
            devices=TWO,
            refine=False,
        )
        with pytest.raises(RuntimeError):
            pipeline.run()
        # The rate probe built the runner; the translation that failed afterwards
        # must not leave it behind. ``run``'s ``finally`` is what guarantees it.
        assert closed == [True]
