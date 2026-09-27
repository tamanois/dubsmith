"""Tests for the timbre measurement helpers and the automated quality gate.

No GPU and no models: the timbre tests use a fake encoder, and the gate tests are
pure functions over a report dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.audit import (
    ANCHOR_WINDOW_S,
    BLEND_NAME,
    DONOR_NAME,
    GateResult,
    QCThresholds,
    accent_path_for,
    audit_job,
    cosine,
    evaluate,
    qc,
    reference_for,
    segment_ids,
    speaker_labels,
    speaker_order,
    summarize,
)
from pipeline.errors import AuditError

SR = 24_000


def write_wav(path: Path, seconds: float = 0.5, amplitude: float = 0.3) -> Path:
    """Write a real WAV so the audit's media layer has something to read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    sf.write(
        str(path), (amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.float32), SR
    )
    return path


class FakeEncoder:
    """Embedding from a clip's mean magnitude, so clips are distinguishable.

    Real speaker embeddings need a checkpoint; this keeps the audit's control flow
    (which clip is compared with which) testable without one. Two clips with the
    same amplitude embed identically, which is exactly the property the tests use.
    """

    def __init__(self) -> None:
        self.calls = 0

    def speaker_embedding(self, samples: np.ndarray, rate: int) -> np.ndarray:
        self.calls += 1
        mean = float(np.mean(np.abs(np.asarray(samples, dtype=np.float32))))
        return np.asarray([mean, 1.0], dtype=np.float64)


def clean_report(**overrides: object) -> dict:
    """A report that passes every cheap gate, so a test can break one field."""
    report: dict = {
        "shifted_segments": 0,
        "overrun_seconds_total": 0.4,
        "trimmed_seconds_total": 0.05,
        "worst_line_deleted_ratio": 0.0,
        "lines_over_10pct_deleted": 0,
        "loudness_delta_vs_source": -0.03,
        "loudness_peak_limited": False,
    }
    report.update(overrides)
    return report


def gate(results: list[GateResult], name: str) -> GateResult:
    return next(result for result in results if result.name == name)


def write_report(job_dir: Path, **overrides: object) -> Path:
    """Write a ``report.json`` into a job folder, so the gate has something to read."""
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "report.json"
    path.write_text(json.dumps(clean_report(**overrides)), encoding="utf-8")
    return path


class TestCosine:
    def test_identical_vectors_are_one(self) -> None:
        vector = np.asarray([0.3, 0.7, 0.1])
        assert cosine(vector, vector) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self) -> None:
        left = np.asarray([1.0, 0.0])
        right = np.asarray([0.0, 1.0])
        assert cosine(left, right) == pytest.approx(0.0)

    def test_is_scale_invariant(self) -> None:
        left = np.asarray([0.2, 1.0])
        assert cosine(left, left * 4.0) == pytest.approx(1.0)


class TestJobArtifacts:
    def test_segment_ids_come_from_timing(self, tmp_path: Path) -> None:
        (tmp_path / "timing.json").write_text(
            json.dumps([{"id": 3}, {"id": 7}]), encoding="utf-8"
        )
        assert segment_ids(tmp_path) == {3, 7}

    def test_segment_ids_fall_back_to_the_transcript(self, tmp_path: Path) -> None:
        (tmp_path / "02_transcript.json").write_text(
            json.dumps({"segments": [{"id": 1}]}), encoding="utf-8"
        )
        assert segment_ids(tmp_path) == {1}

    def test_segment_ids_are_empty_without_artifacts(self, tmp_path: Path) -> None:
        assert segment_ids(tmp_path) == set()

    def test_missing_speakers_file_is_empty(self, tmp_path: Path) -> None:
        assert speaker_labels(tmp_path) == {}

    def test_speakers_are_read_and_ordered(self, tmp_path: Path) -> None:
        (tmp_path / "02b_speakers.json").write_text(
            json.dumps({"0": "SPK_1", "1": "SPK_0", "2": "SPK_1"}), encoding="utf-8"
        )
        labels = speaker_labels(tmp_path)

        assert labels == {0: "SPK_1", 1: "SPK_0", 2: "SPK_1"}
        assert speaker_order(labels) == ["SPK_1", "SPK_0"]  # by first appearance

    def test_malformed_speakers_file_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "02b_speakers.json").write_text("{not json", encoding="utf-8")
        assert speaker_labels(tmp_path) == {}

    def test_reference_prefers_the_per_speaker_blend(self, tmp_path: Path) -> None:
        per_speaker = tmp_path / "04_dubbed" / "reference_blend_SPK_1.wav"
        per_speaker.parent.mkdir(parents=True, exist_ok=True)
        per_speaker.write_bytes(b"x")

        assert reference_for(tmp_path, "SPK_1", None) == per_speaker
        assert reference_for(tmp_path, "SPK_0", None) == tmp_path / BLEND_NAME

    def test_reference_override_wins(self, tmp_path: Path) -> None:
        override = tmp_path / "stick.wav"
        assert reference_for(tmp_path, "SPK_0", override) == override

    def test_accent_path_is_read_from_donor_json(self, tmp_path: Path) -> None:
        (tmp_path / DONOR_NAME).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / DONOR_NAME).write_text(
            json.dumps({"accent": {"path": "input/accent.wav"}}), encoding="utf-8"
        )
        assert accent_path_for(tmp_path) == Path("input/accent.wav")

    def test_accent_path_is_none_without_donor_json(self, tmp_path: Path) -> None:
        assert accent_path_for(tmp_path) is None


class TestAuditJob:
    """The measurement path, end to end, over a fake encoder."""

    def build_job(self, tmp_path: Path) -> tuple[Path, Path]:
        """A one-voice job whose lines sit on either side of the scale."""
        accent = write_wav(tmp_path / "accent.wav", amplitude=0.2)
        write_wav(tmp_path / "04_dubbed" / "reference_blend.wav", amplitude=0.4)
        write_wav(tmp_path / "04_dubbed" / "seg_0000.wav", amplitude=0.41)
        write_wav(tmp_path / "04_dubbed" / "seg_0001.wav", amplitude=0.19)
        return tmp_path, accent

    def test_scores_lines_and_flags_the_collapsed_one(self, tmp_path: Path) -> None:
        job_dir, accent = self.build_job(tmp_path)
        results = audit_job(job_dir, FakeEncoder(), accent=accent)

        assert len(results) == 1
        result = results[0]
        assert len(result["rows"]) == 2
        assert result["collapsed"] == ["seg_0001"]
        assert result["max"] > 0.0 > result["min"]

    def test_missing_accent_clip_is_an_error_not_a_wrong_answer(
        self, tmp_path: Path
    ) -> None:
        job_dir, _ = self.build_job(tmp_path)
        with pytest.raises(AuditError, match="no accent clip"):
            audit_job(job_dir, FakeEncoder())

    def test_missing_reference_clip_is_an_error(self, tmp_path: Path) -> None:
        accent = write_wav(tmp_path / "accent.wav")
        (tmp_path / "04_dubbed").mkdir()
        write_wav(tmp_path / "04_dubbed" / "seg_0000.wav")
        with pytest.raises(AuditError, match="no reference clip"):
            audit_job(tmp_path, FakeEncoder(), accent=accent)

    def test_orphaned_clips_are_ignored(self, tmp_path: Path) -> None:
        job_dir, accent = self.build_job(tmp_path)
        # A line this run never produced, left behind by an older segmentation.
        write_wav(job_dir / "04_dubbed" / "seg_0099.wav", amplitude=0.2)
        (job_dir / "timing.json").write_text(
            json.dumps([{"id": 0}, {"id": 1}]), encoding="utf-8"
        )

        result = audit_job(job_dir, FakeEncoder(), accent=accent)[0]
        assert len(result["rows"]) == 2
        assert result["skipped"] == 1

    def test_a_short_accent_clip_still_yields_a_usable_anchor(
        self, tmp_path: Path
    ) -> None:
        """A few seconds of accent speech is normal; the anchor must not be NaN."""
        accent = write_wav(tmp_path / "accent.wav", seconds=0.4, amplitude=0.2)
        write_wav(tmp_path / "04_dubbed" / "reference_blend.wav", amplitude=0.4)
        write_wav(tmp_path / "04_dubbed" / "seg_0000.wav", amplitude=0.41)

        result = audit_job(tmp_path, FakeEncoder(), accent=accent)[0]

        assert np.isfinite(result["collapsed_anchor"])
        assert np.isfinite(result["target_anchor"])

    def test_multi_voice_jobs_are_audited_per_speaker(self, tmp_path: Path) -> None:
        accent = write_wav(tmp_path / "accent.wav", amplitude=0.2)
        write_wav(tmp_path / "04_dubbed" / "reference_blend_SPK_0.wav", amplitude=0.4)
        write_wav(tmp_path / "04_dubbed" / "reference_blend_SPK_1.wav", amplitude=0.4)
        write_wav(tmp_path / "04_dubbed" / "seg_0000.wav", amplitude=0.41)
        write_wav(tmp_path / "04_dubbed" / "seg_0001.wav", amplitude=0.41)
        (tmp_path / "02b_speakers.json").write_text(
            json.dumps({"0": "SPK_0", "1": "SPK_1"}), encoding="utf-8"
        )

        results = audit_job(tmp_path, FakeEncoder(), accent=accent)

        assert [result["speaker"] for result in results] == ["SPK_0", "SPK_1"]
        # Each voice sees only its own line, so neither is judged against the other.
        assert all(len(result["rows"]) == 1 for result in results)
        assert ANCHOR_WINDOW_S > 0  # the anchor window is part of the scale


class TestEvaluate:
    def test_a_clean_report_passes_every_cheap_gate(self) -> None:
        results = evaluate(clean_report())

        assert [result.name for result in results] == [
            "sync",
            "truncation",
            "loudness",
            "timbre",
        ]
        assert gate(results, "sync").status == "pass"
        assert gate(results, "truncation").status == "pass"
        assert gate(results, "loudness").status == "pass"
        assert gate(results, "timbre").status == "skip"  # not measured

    def test_shifted_lines_fail_the_sync_gate(self) -> None:
        results = evaluate(clean_report(shifted_segments=1))
        assert gate(results, "sync").status == "fail"

    def test_spilling_into_pauses_is_not_a_sync_failure(self) -> None:
        """Overrun is the designed behaviour, so it must not fail the gate."""
        results = evaluate(
            clean_report(overrun_seconds_total=3.5, trimmed_seconds_total=0.0)
        )
        assert gate(results, "sync").status == "pass"

    def test_truncation_fails_on_the_worst_line(self) -> None:
        results = evaluate(clean_report(worst_line_deleted_ratio=0.59))
        assert gate(results, "truncation").status == "fail"

    def test_truncation_fails_on_the_count_of_damaged_lines(self) -> None:
        results = evaluate(clean_report(lines_over_10pct_deleted=3))
        assert gate(results, "truncation").status == "fail"

    def test_loudness_failure_and_skip(self) -> None:
        loud = evaluate(clean_report(loudness_delta_vs_source=4.2))
        assert gate(loud, "loudness").status == "fail"

        unknown = evaluate(clean_report(loudness_delta_vs_source=None))
        assert gate(unknown, "loudness").status == "skip"

    def test_peak_limiting_is_reported_but_not_a_failure(self) -> None:
        results = evaluate(clean_report(loudness_peak_limited=True))
        result = gate(results, "loudness")
        assert result.status == "pass"
        assert "limited" in result.detail

    def test_thresholds_are_overridable(self) -> None:
        report = clean_report(shifted_segments=2, worst_line_deleted_ratio=0.2)
        limits = QCThresholds(
            max_shifted_segments=2, max_worst_line_deleted_ratio=0.25
        )
        results = evaluate(report, thresholds=limits)
        assert gate(results, "sync").status == "pass"
        assert gate(results, "truncation").status == "pass"

    def test_a_missing_field_is_treated_as_healthy(self) -> None:
        """An older report without a field should not fail inscrutably."""
        results = evaluate({})
        assert gate(results, "sync").status == "pass"
        assert gate(results, "truncation").status == "pass"
        assert gate(results, "loudness").status == "skip"

    def test_timbre_passes_when_no_line_followed_the_accent_voice(self) -> None:
        timbre = [{"speaker": None, "rows": [(0,) * 4], "collapsed": []}]
        results = evaluate(clean_report(), timbre=timbre)
        assert gate(results, "timbre").status == "pass"

    def test_timbre_fails_when_a_line_followed_the_accent_voice(self) -> None:
        timbre = [{"speaker": None, "rows": [(0,) * 4] * 10, "collapsed": ["seg_0001"]}]
        results = evaluate(clean_report(), timbre=timbre)

        result = gate(results, "timbre")
        assert result.status == "fail"
        assert "1 of 10" in result.detail

    def test_timbre_is_skipped_without_lines(self) -> None:
        timbre = [{"speaker": None, "rows": [], "collapsed": []}]
        results = evaluate(clean_report(), timbre=timbre)
        assert gate(results, "timbre").status == "skip"


class TestSummarize:
    def test_shape_and_pass_flag(self) -> None:
        payload = summarize(evaluate(clean_report()))

        assert payload["passed"] is True
        assert payload["failed"] == []
        assert payload["skipped"] == ["timbre"]
        assert len(payload["gates"]) == 4
        assert all(
            set(item) == {"name", "status", "detail"} for item in payload["gates"]
        )

    def test_names_the_failed_gates(self) -> None:
        payload = summarize(evaluate(clean_report(shifted_segments=3)))
        assert payload["passed"] is False
        assert payload["failed"] == ["sync"]

    def test_is_json_serializable(self) -> None:
        payload = summarize(evaluate(clean_report()))
        assert json.loads(json.dumps(payload)) == payload


class TestQc:
    def test_reads_the_report_and_skips_timbre_without_an_engine(
        self, tmp_path: Path
    ) -> None:
        write_report(tmp_path)
        payload = qc(tmp_path)

        assert payload["passed"] is True
        assert payload["skipped"] == ["timbre"]
        assert payload["job_dir"] == str(tmp_path)

    def test_measures_timbre_when_an_engine_is_supplied(self, tmp_path: Path) -> None:
        write_report(tmp_path)
        accent = write_wav(tmp_path / "accent.wav", amplitude=0.2)
        write_wav(tmp_path / "04_dubbed" / "reference_blend.wav", amplitude=0.4)
        write_wav(tmp_path / "04_dubbed" / "seg_0000.wav", amplitude=0.19)

        payload = qc(tmp_path, engine=FakeEncoder(), accent=accent)

        assert payload["passed"] is False
        assert payload["failed"] == ["timbre"]

    def test_missing_report_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(AuditError, match="has no report.json"):
            qc(tmp_path)

    def test_invalid_report_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "report.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(AuditError, match="not valid JSON"):
            qc(tmp_path)

    def test_non_object_report_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "report.json").write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(AuditError, match="does not contain a job report"):
            qc(tmp_path)


class TestQcCli:
    """The CLI's exit codes, which are what CI reads."""

    def test_passes_with_a_clean_report(self, tmp_path: Path, capsys) -> None:
        import audit_qc

        write_report(tmp_path)
        assert audit_qc.main([str(tmp_path)]) == 0
        assert "PASS" in capsys.readouterr().out

    def test_fails_on_a_damaged_report(self, tmp_path: Path, capsys) -> None:
        import audit_qc

        (tmp_path / "report.json").write_text(
            json.dumps(clean_report(worst_line_deleted_ratio=0.8)), encoding="utf-8"
        )
        assert audit_qc.main([str(tmp_path)]) == 1
        assert "truncation" in capsys.readouterr().out

    def test_json_mode_emits_only_json(self, tmp_path: Path, capsys) -> None:
        import audit_qc

        write_report(tmp_path)
        assert audit_qc.main([str(tmp_path), "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["passed"] is True
        assert payload["skipped"] == ["timbre"]

    def test_unusable_job_exits_two(self, tmp_path: Path, capsys) -> None:
        import audit_qc

        assert audit_qc.main([str(tmp_path)]) == 2
        assert "has no report.json" in capsys.readouterr().err

    def test_threshold_flags_are_applied(self, tmp_path: Path, capsys) -> None:
        import audit_qc

        (tmp_path / "report.json").write_text(
            json.dumps(clean_report(worst_line_deleted_ratio=0.2)), encoding="utf-8"
        )
        assert audit_qc.main([str(tmp_path)]) == 1
        assert audit_qc.main([str(tmp_path), "--max-deleted-ratio", "0.5"]) == 0
