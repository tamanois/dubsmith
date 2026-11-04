"""Tests for the expressive dubbing backends and LLM prosody planning.

These exercise the pieces added so a dub can render per-line emotion with
CosyVoice3, Chatterbox or Fish: the config surface, the planning fallback, the
per-speaker synthesis loop, and the guards that keep the Qwen-only stages from
loading the Base checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipeline.asr import Segment, Transcript
from pipeline.config import JobConfig, Settings
from pipeline.errors import ConfigError
from pipeline.orchestrate import Pipeline
from pipeline.prosody import ProsodySpec
from pipeline.tts import DonorClip

SAMPLE_RATE = 24_000


class FakeExpressiveEngine:
    """Mimics an expressive engine's ``synthesize`` without loading a model."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def synthesize(self, lines, *, reference, **kwargs: Any):
        self.calls.append(
            {"lines": list(lines), "reference": reference, **kwargs}
        )
        return [
            np.full(int(0.1 * SAMPLE_RATE), 0.1, dtype=np.float32) for _ in lines
        ]


def _job(tmp_path: Path, **kwargs: Any) -> JobConfig:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")  # existence is not checked until run()
    kwargs.setdefault("accent_mode", "preserve")
    if kwargs["accent_mode"] == "preserve":
        kwargs.setdefault("ref_text", "hello")
    return JobConfig(audio=audio, output_dir=tmp_path / "out", **kwargs)


def _pipeline(tmp_path: Path, engine: Any | None = None, **kwargs: Any) -> Pipeline:
    return Pipeline(
        _job(tmp_path, **kwargs),
        Settings(models_dir=tmp_path / "models"),
        expressive_engine=engine,
    )


def _donor(segment_id: int, text: str = "source line") -> DonorClip:
    return DonorClip(
        samples=np.full(SAMPLE_RATE, 0.2, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        text=text,
        start=0.0,
        end=1.0,
        segment_id=segment_id,
    )


def _transcript(*segments: Segment) -> Transcript:
    return Transcript(language="en", language_probability=0.99, model="fake", segments=list(segments))


# --------------------------------------------------------------------------- #
# Config surface
# --------------------------------------------------------------------------- #


class TestBackendConfig:
    def test_defaults_keep_the_qwen_path(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        assert job.tts_backend == "qwen"
        assert job.prosody_plan == "auto"

    def test_backend_aliases_are_normalised(self, tmp_path: Path) -> None:
        assert _job(tmp_path, tts_backend="fish-speech").tts_backend == "fish"
        assert _job(tmp_path, tts_backend="cosyvoice").tts_backend == "cosyvoice3"

    def test_unknown_backend_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            _job(tmp_path, tts_backend="magic")

    def test_prosody_plan_aliases_are_normalised(self, tmp_path: Path) -> None:
        assert _job(tmp_path, prosody_plan="none").prosody_plan == "off"
        assert _job(tmp_path, prosody_plan="LLM").prosody_plan == "llm"

    def test_backend_changes_the_fingerprint(self, tmp_path: Path) -> None:
        qwen = _job(tmp_path, tts_backend="qwen")
        chatterbox = _job(tmp_path, tts_backend="chatterbox")
        assert qwen.fingerprint() != chatterbox.fingerprint()

    def test_dub_parser_exposes_the_new_flags(self) -> None:
        from dub import build_dub_parser

        args = build_dub_parser().parse_args(
            ["--video", "in.mp4", "--backend", "chatterbox", "--prosody-plan", "llm"]
        )
        assert args.tts_backend == "chatterbox"
        assert args.prosody_plan == "llm"

    def test_job_config_from_args_maps_the_new_flags(self, tmp_path: Path) -> None:
        from dub import build_dub_parser, job_config_from_args

        args = build_dub_parser().parse_args(
            ["--audio", str(tmp_path / "in.wav"), "--backend", "fish"]
        )
        job = job_config_from_args(args)
        assert job.tts_backend == "fish"
        assert job.prosody_plan == "auto"


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


class TestPlanProsody:
    def test_qwen_returns_nothing(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="qwen")
        lines = [Segment(id=0, start=0.0, end=1.0, text="a")]
        assert pipeline._plan_prosody(lines, {0: "Bonjour"}) == {}

    def test_off_never_calls_the_llm(self, tmp_path: Path, monkeypatch) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="chatterbox", prosody_plan="off")

        class Boom:
            def __init__(self, **kwargs: Any) -> None:
                raise AssertionError("planner must not be built when plan is off")

        monkeypatch.setattr("pipeline.prosody.ProsodyPlanner", Boom)
        lines = [Segment(id=0, start=0.0, end=1.0, text="a")]
        assert pipeline._plan_prosody(lines, {0: "Bonjour"}) == {}

    def test_auto_falls_back_to_neutral_when_planning_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="chatterbox", prosody_plan="auto")

        class Broken:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def plan(self, texts):
                raise RuntimeError("server down")

        monkeypatch.setattr("pipeline.prosody.ProsodyPlanner", Broken)
        lines = [
            Segment(id=0, start=0.0, end=1.0, text="a"),
            Segment(id=1, start=1.0, end=2.0, text="b"),
        ]
        planned = pipeline._plan_prosody(lines, {0: "Bonjour", 1: "Salut"})
        assert set(planned) == {0, 1}
        assert all(spec.is_empty() for spec in planned.values())

    def test_llm_propagates_planning_failure(self, tmp_path: Path, monkeypatch) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="chatterbox", prosody_plan="llm")

        class Broken:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def plan(self, texts):
                raise RuntimeError("server down")

        monkeypatch.setattr("pipeline.prosody.ProsodyPlanner", Broken)
        lines = [Segment(id=0, start=0.0, end=1.0, text="a")]
        with pytest.raises(RuntimeError):
            pipeline._plan_prosody(lines, {0: "Bonjour"})

    def test_fish_auto_keeps_the_text_when_planning_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="fish", prosody_plan="auto")

        class Broken:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def plan_lines(self, texts, *, language=None):
                raise RuntimeError("server down")

        monkeypatch.setattr("pipeline.prosody.FishInlinePlanner", Broken)
        lines = [Segment(id=0, start=0.0, end=1.0, text="a")]
        assert pipeline._plan_prosody(lines, {0: "Bonjour"}) == {0: "Bonjour"}

    def test_blank_lines_are_skipped_but_keep_an_entry(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="chatterbox", prosody_plan="auto")
        seen: list[str] = []

        class Planner:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def plan(self, texts):
                seen.extend(texts)
                return [ProsodySpec(emotion="happy") for _ in texts]

        monkeypatch.setattr("pipeline.prosody.ProsodyPlanner", Planner)
        lines = [
            Segment(id=0, start=0.0, end=1.0, text="a"),
            Segment(id=1, start=1.0, end=2.0, text=""),
            Segment(id=2, start=2.0, end=3.0, text="c"),
        ]
        planned = pipeline._plan_prosody(lines, {0: "Bonjour", 1: "", 2: "Salut"})
        assert seen == ["Bonjour", "Salut"]
        assert set(planned) == {0, 1, 2}
        assert planned[1].is_empty()
        assert planned[0].emotion == "happy"


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


class TestSynthesizeExpressive:
    def test_groups_by_speaker_and_maps_segment_ids(self, tmp_path: Path) -> None:
        engine = FakeExpressiveEngine()
        pipeline = _pipeline(tmp_path, engine=engine, tts_backend="chatterbox")
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="a", speaker="A"),
            Segment(id=1, start=1.0, end=2.0, text="", speaker="A"),
            Segment(id=2, start=2.0, end=3.0, text="c", speaker="B"),
        ]
        donors = {"A": (_donor(0), {}), "B": (_donor(2), {})}
        prosody = {0: ProsodySpec(emotion="happy"), 2: ProsodySpec(emotion="sad")}
        produced, rate = pipeline._synthesize_expressive(
            segments, {0: "Bonjour", 1: "", 2: "Salut"}, donors, tmp_path, prosody
        )
        assert set(produced) == {0, 2}
        assert rate == SAMPLE_RATE
        # One call per speaker; the blank line is never sent.
        assert len(engine.calls) == 2
        assert engine.calls[0]["lines"] == ["Bonjour"]
        assert engine.calls[1]["lines"] == ["Salut"]
        assert [spec.emotion for spec in engine.calls[0]["prosody"]] == ["happy"]
        assert [spec.emotion for spec in engine.calls[1]["prosody"]] == ["sad"]

    def test_auto_donor_is_written_to_a_temp_clip(self, tmp_path: Path) -> None:
        engine = FakeExpressiveEngine()
        pipeline = _pipeline(tmp_path, engine=engine, tts_backend="chatterbox")
        segments = [Segment(id=0, start=0.0, end=1.0, text="a", speaker="A")]
        donors = {"A": (_donor(0, text="source words"), {})}
        pipeline._synthesize_expressive(
            segments, {0: "Bonjour"}, donors, tmp_path
        )
        reference = engine.calls[0]["reference"]
        assert isinstance(reference, Path)
        assert reference.parent == tmp_path
        assert reference.exists()

    def test_pinned_reference_is_used_for_every_speaker(self, tmp_path: Path) -> None:
        engine = FakeExpressiveEngine()
        ref = tmp_path / "pinned.wav"
        ref.write_bytes(b"RIFF")
        pipeline = _pipeline(
            tmp_path,
            engine=engine,
            tts_backend="cosyvoice3",
            ref_audio=ref,
            ref_text="reference words",
        )
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="a", speaker="A"),
            Segment(id=1, start=1.0, end=2.0, text="b", speaker="B"),
        ]
        pipeline._synthesize_expressive(
            segments, {0: "Bonjour", 1: "Salut"}, {"A": (None, {}), "B": (None, {})}, tmp_path
        )
        assert all(call["reference"] == ref for call in engine.calls)
        assert all(call["ref_text"] == "reference words" for call in engine.calls)

    def test_chatterbox_receives_the_language_code(self, tmp_path: Path) -> None:
        engine = FakeExpressiveEngine()
        pipeline = _pipeline(
            tmp_path, engine=engine, tts_backend="chatterbox", target_language="fr"
        )
        segments = [Segment(id=0, start=0.0, end=1.0, text="a", speaker="A")]
        pipeline._synthesize_expressive(
            segments, {0: "Bonjour"}, {"A": (_donor(0), {})}, tmp_path
        )
        assert engine.calls[0]["language"] == "fr"

    def test_fish_speaks_the_tagged_line(self, tmp_path: Path) -> None:
        engine = FakeExpressiveEngine()
        pipeline = _pipeline(tmp_path, engine=engine, tts_backend="fish")
        segments = [Segment(id=0, start=0.0, end=1.0, text="a", speaker="A")]
        pipeline._synthesize_expressive(
            segments,
            {0: "Bonjour"},
            {"A": (_donor(0), {})},
            tmp_path,
            {0: "[sad] Bonjour"},
        )
        assert engine.calls[0]["lines"] == ["[sad] Bonjour"]


# --------------------------------------------------------------------------- #
# Qwen-only stage guards
# --------------------------------------------------------------------------- #


class TestQwenOnlyGuards:
    def test_blend_is_skipped_for_expressive_backends(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="chatterbox")
        transcript = _transcript(Segment(id=0, start=0.0, end=1.0, text="a"))
        assert pipeline._stage_blend(tmp_path / "a.wav", transcript) == {}

    def test_rate_probe_is_skipped_for_expressive_backends(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="fish")
        transcript = _transcript(Segment(id=0, start=0.0, end=1.0, text="a"))
        assert pipeline._stage_rate_probe(tmp_path / "a.wav", transcript, {}, {}) == {}

    def test_accent_anchor_is_skipped_for_expressive_backends(
        self, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(tmp_path, tts_backend="chatterbox", accent_mode="native")
        transcript = _transcript(Segment(id=0, start=0.0, end=1.0, text="a"))
        assert pipeline._stage_accent_anchor(tmp_path / "a.wav", transcript) == {}


# --------------------------------------------------------------------------- #
# End-to-end wiring through Pipeline.run()
# --------------------------------------------------------------------------- #


class TestExpressiveJob:
    def test_run_plans_and_dubs_with_the_expressive_backend(
        self, tmp_path: Path, simple_transcript: Transcript, monkeypatch
    ) -> None:
        from tests.test_pipeline import make_pipeline

        engine = FakeExpressiveEngine()

        class Planner:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def plan(self, texts):
                return [ProsodySpec(emotion="happy") for _ in texts]

        monkeypatch.setattr("pipeline.prosody.ProsodyPlanner", Planner)
        pipeline, *_ = make_pipeline(
            tmp_path,
            simple_transcript,
            tts_backend="chatterbox",
            prosody_plan="llm",
            expressive_engine=engine,
        )

        pipeline.run()

        assert engine.calls, "the expressive engine was never asked to synthesise"
        assert all(
            spec.emotion == "happy"
            for call in engine.calls
            for spec in call["prosody"]
        )
