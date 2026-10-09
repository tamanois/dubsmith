"""Speaker-timbre measurement and the automated quality gate.

Two jobs in one module, because they answer the same question from the same
evidence: *did this job deliver what it was supposed to?*

* :func:`audit` measures, per dubbed line, whether the speaker's timbre survived,
  using the checkpoint's own speaker encoder. Judging that by ear is unreliable:
  the artefact is subtle, it moves between runs, and it is invisible in an average.
* :func:`evaluate` turns the signals in ``report.json`` — and the timbre audit when
  a GPU is available — into pass/fail gates, so "the run was fine" is checked
  rather than assumed.

The measurement has to be **language-matched**. Comparing a French dubbed line to
the donor clip (English, in a typical job) folds a language difference into a
speaker measurement, and the encoder is sensitive to it. So each line is compared
against two clips that are both in the target language:

    reference : the voice the line was supposed to be in. In ``reference-accent``
                mode that is ``04_dubbed/reference_blend.wav``, the stage-1 blend.
    accent    : the clip the pronunciation was borrowed from.

    margin = cos(line, reference) - cos(line, accent)

Both references speak the target language, so the only thing separating them is
whose voice it is. Two anchors give the number a scale:

    collapsed anchor : the accent clip's own utterances  (should be negative)
    target anchor    : the reference against itself      (= 1 - cos(ref, accent))

A line near the collapsed anchor followed the accent voice. Anything at or above
zero at least resembles the intended voice more than the borrowed one.

The command-line wrappers live in ``audit_timbre.py`` and ``audit_qc.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from . import media
from .config import DEFAULT_MAX_OVERRUN
from .errors import AuditError

#: Slack on the lateness limit: the timeline flags a move only past its own guard
#: (``timing.SHIFT_FLAG_GUARD_S``), so a run sitting exactly on the limit must not
#: fail on the last sample of rounding.
SHIFT_TOLERANCE_S = 0.02

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module import-light
    from .tts import TTSEngine

__all__ = [
    "cosine",
    "segment_ids",
    "embed",
    "speaker_labels",
    "speaker_order",
    "reference_for",
    "margin_for",
    "collapse_anchor",
    "audit",
    "audit_job",
    "accent_path_for",
    "timbre_available",
    "GateResult",
    "QCThresholds",
    "evaluate",
    "summarize",
    "qc",
    "BLEND_NAME",
    "BLEND_PATTERN",
    "DONOR_NAME",
    "SPEAKERS_NAME",
    "REPORT_NAME",
    "ANCHOR_WINDOW_S",
]

#: Artifacts a job writes that the audit reads back.
BLEND_NAME = "04_dubbed/reference_blend.wav"
BLEND_PATTERN = "04_dubbed/reference_blend_{speaker}.wav"
DONOR_NAME = "04_dubbed/donor.json"
SPEAKERS_NAME = "02b_speakers.json"
TIMING_NAME = "timing.json"
TRANSCRIPT_NAME = "02_transcript.json"
REPORT_NAME = "report.json"

#: Window used to bound the collapsed anchor, in seconds.
ANCHOR_WINDOW_S = 7.0


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def segment_ids(job_dir: Path) -> set[int]:
    """Ids this job actually produced, so stale clips from older runs are ignored.

    Nothing used to clean ``04_dubbed/``, so a job re-run with a different
    segmentation can still hold orphaned clips from the previous one.
    """
    for name in (TIMING_NAME, TRANSCRIPT_NAME):
        path = job_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return {int(item["id"]) for item in payload}
        if isinstance(payload, dict) and "segments" in payload:
            return {int(item["id"]) for item in payload["segments"]}
    return set()


def embed(engine: TTSEngine, samples: np.ndarray, rate: int) -> np.ndarray:
    """Speaker embedding of one clip, via the checkpoint's own encoder."""
    return engine.speaker_embedding(np.asarray(samples, dtype=np.float32), rate)


def speaker_labels(job_dir: Path) -> dict[int, str]:
    """Segment id to speaker, for a job that recorded a diarization.

    Empty for a single-voice job, which is what makes the audit fall back to
    measuring every line against one clip.
    """
    path = job_dir / SPEAKERS_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    try:
        return {int(key): str(value) for key, value in payload.items()}
    except (TypeError, ValueError):
        return {}


def speaker_order(labels: dict[int, str]) -> list[str]:
    """Speakers in order of first appearance, as the pipeline numbers them."""
    order: list[str] = []
    for speaker in labels.values():
        if speaker not in order:
            order.append(speaker)
    return order


def reference_for(job_dir: Path, speaker: str, override: Path | None) -> Path:
    """The clip a speaker's lines were supposed to be cloned from.

    With several voices each speaker has their own blend clip, and a line must be
    measured against the voice *it* was cloned from: comparing it to another
    speaker's clip would report a voice change that never happened.
    """
    if override is not None:
        return override
    per_speaker = job_dir / BLEND_PATTERN.format(speaker=speaker)
    return per_speaker if per_speaker.is_file() else job_dir / BLEND_NAME


def margin_for(
    engine: TTSEngine,
    samples: np.ndarray,
    rate: int,
    reference: np.ndarray,
    accent: np.ndarray,
) -> tuple[float, float, float]:
    """Score one clip: ``(margin, similarity to reference, similarity to accent)``."""
    line = embed(engine, samples, rate)
    to_reference = cosine(line, reference)
    to_accent = cosine(line, accent)
    return (to_reference - to_accent, to_reference, to_accent)


def collapse_anchor(
    engine: TTSEngine,
    accent_samples: np.ndarray,
    accent_rate: int,
    reference: np.ndarray,
    accent: np.ndarray,
) -> float:
    """Where the accent speaker's *own* speech scores on this scale.

    A line at or below this followed the accent voice, so it bounds the useful
    range of the measurement rather than being an arbitrary threshold.

    Up to two windows are averaged to cancel the per-window noise of a single
    utterance. An accent clip shorter than one window — which is normal, since a
    few seconds of speech is all the cloning needs — is measured whole rather than
    averaged over nothing, which would make the anchor ``NaN`` and the whole scale
    meaningless.
    """
    if np.asarray(accent_samples).size == 0:
        return float("nan")

    window = int(ANCHOR_WINDOW_S * accent_rate)
    windows = [
        accent_samples[start : start + window]
        for start in (0, window)
        if accent_samples[start : start + window].size
    ]
    if not windows:
        windows = [accent_samples]

    return float(
        np.mean(
            [
                margin_for(engine, part, accent_rate, reference, accent)[0]
                for part in windows
            ]
        )
    )


def audit(
    job_dir: Path,
    engine: TTSEngine,
    reference: Path,
    accent: Path,
    *,
    speaker: str | None = None,
    only_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Measure one group of lines against one reference clip.

    Args:
        job_dir: The job folder to audit.
        engine: Loaded TTS engine, used for its speaker encoder.
        reference: The voice the lines were supposed to be cloned from.
        accent: The clip the pronunciation was borrowed from.
        speaker: Speaker this group belongs to, for the report; ``None`` when the
            job has a single voice.
        only_ids: Restrict the clips to these segment ids. ``None`` audits all of
            the job's own lines.

    Returns:
        A result dict with the per-line rows, the summary statistics, and the two
        anchors that put the margins on a scale.
    """
    ref_samples, ref_rate = media.load_audio(reference)
    accent_samples, accent_rate = media.load_audio(accent)
    ref_embed = embed(engine, ref_samples, ref_rate)
    accent_embed = embed(engine, accent_samples, accent_rate)

    known = segment_ids(job_dir)
    dub_dir = job_dir / "04_dubbed"
    rows: list[tuple[str, float, float, float]] = []
    skipped = 0
    for path in sorted(dub_dir.glob("seg_*.wav")):
        number = int(path.stem.split("_")[1])
        # Orphans from an older, differently segmented run: not this job's lines.
        if known and number not in known:
            skipped += 1
            continue
        if only_ids is not None and number not in only_ids:
            continue
        samples, rate = media.load_audio(path)
        if samples.size == 0:
            continue
        margin, to_reference, to_accent = margin_for(
            engine, samples, rate, ref_embed, accent_embed
        )
        rows.append((path.stem, to_reference, to_accent, margin))

    margins = np.asarray([row[3] for row in rows], dtype=np.float64)
    return {
        "job_dir": job_dir,
        "speaker": speaker,
        "reference": reference,
        "rows": rows,
        "skipped": skipped,
        "min": float(margins.min()) if margins.size else float("nan"),
        "median": float(np.median(margins)) if margins.size else float("nan"),
        "max": float(margins.max()) if margins.size else float("nan"),
        "collapsed": [row[0] for row in rows if row[3] < 0.0],
        "target_anchor": 1.0 - cosine(ref_embed, accent_embed),
        "collapsed_anchor": collapse_anchor(
            engine, accent_samples, accent_rate, ref_embed, accent_embed
        ),
    }


def accent_path_for(job_dir: Path) -> Path | None:
    """The accent clip this job borrowed pronunciation from."""
    path = job_dir / DONOR_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    accent = payload.get("accent") or {}
    if not isinstance(accent, dict) or not accent.get("path"):
        return None
    return Path(str(accent["path"]))


def audit_job(
    job_dir: Path,
    engine: TTSEngine,
    *,
    accent: Path | None = None,
    reference: Path | None = None,
) -> list[dict[str, Any]]:
    """Audit every voice in a job, each against the clip it was cloned from.

    A single-voice job is measured against one clip. A multi-voice job is measured
    speaker by speaker, because two voices have two different "pure accent" floors
    and a line must never be judged against another voice's reference.

    Args:
        job_dir: The job folder to audit.
        engine: Loaded TTS engine.
        accent: Accent clip; defaults to the one recorded in ``donor.json``.
        reference: Force a single measuring stick. Only useful for a deliberate
            A/B, where the same clip must judge every line.

    Returns:
        One result dict per voice, ready for :func:`report`-style printing or for
        :func:`evaluate`.

    Raises:
        AuditError: When the accent clip, or a required reference clip, is missing.
            That makes the measurement impossible rather than merely imprecise.
    """
    chosen_accent = accent or accent_path_for(job_dir)
    if chosen_accent is None or not Path(chosen_accent).is_file():
        raise AuditError(
            f"{job_dir}: no accent clip found. Pass --accent to name one "
            "(donor.json records it for a reference-accent job)."
        )

    labels = speaker_labels(job_dir)
    voices = speaker_order(labels)
    if reference is not None or len(voices) < 2:
        # One voice, or a measuring stick chosen by hand: every line is measured
        # against the same clip, as before diarization existed.
        chosen = reference or (job_dir / BLEND_NAME)
        if not chosen.is_file():
            raise AuditError(
                f"{job_dir}: no reference clip at {chosen}. A single-stage job "
                "produces none - pass --reference pointing at another job's "
                "reference_blend.wav."
            )
        return [audit(job_dir, engine, chosen, Path(chosen_accent))]

    results: list[dict[str, Any]] = []
    for name in voices:
        chosen = reference_for(job_dir, name, None)
        if not chosen.is_file():
            raise AuditError(f"{job_dir}: no blend clip for {name} at {chosen}.")
        owned = {sid for sid, who in labels.items() if who == name}
        results.append(
            audit(
                job_dir,
                engine,
                chosen,
                Path(chosen_accent),
                speaker=name,
                only_ids=owned,
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateResult:
    """One quality gate: what was checked, what was measured, and the verdict.

    ``status`` is one of ``pass``, ``fail`` or ``skip``. A skipped gate does not
    fail the run — it was not evaluated, and pretending otherwise would turn a
    missing optional check into a false alarm.
    """

    name: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        """True unless the gate explicitly failed."""
        return self.status != "fail"

    def to_dict(self) -> dict[str, str]:
        """JSON-safe representation."""
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class QCThresholds:
    """Limits the quality gate applies.

    The defaults are the levels the roadmap's measurements treated as acceptable: a
    line may start at most ``DEFAULT_MAX_OVERRUN`` seconds late, may lose at most a
    tenth of its words, and the delivered level stays within 1.5 dB of the track it
    replaced.
    """

    #: Seconds a line may start later than its source. The mixer deliberately buys
    #: a fraction of a second of lateness per line rather than deleting the last
    #: word of it, so this is how much lateness counts as inside policy. It is
    #: independent of what a run was configured with: a run that borrows more than
    #: the policy allows fails here, which is the point.
    max_shift_seconds: float = DEFAULT_MAX_OVERRUN
    #: Largest share of a single line that may be trimmed away.
    max_worst_line_deleted_ratio: float = 0.10
    #: Lines allowed to lose more than 10% of their speech.
    max_lines_over_10pct_deleted: int = 0
    #: Allowed difference between the delivered and source loudness, in dB.
    max_loudness_delta_db: float = 1.5
    #: Share of lines allowed to sit closer to the accent voice than the intended
    #: one. Zero, because a dub whose speaker drifts is a defect, not a gradient.
    max_accent_side_share: float = 0.0


def _sync_gate(report: Mapping[str, Any], limits: QCThresholds) -> GateResult:
    """Gate on how late the dub runs; report spilling into pauses as context.

    ``overrun`` is by design: a line may use the silence before the next utterance,
    which costs nothing. Lateness is the real sync risk, and it is gated on the
    *amount* rather than on a count of moved lines - a run that borrowed 0.1 s of a
    line's time is not the same defect as one that started a line a second late.
    """
    shifted = int(report.get("shifted_segments") or 0)
    worst = float(report.get("shifted_seconds_max") or 0.0)
    overrun = float(report.get("overrun_seconds_total") or 0.0)
    trimmed = float(report.get("trimmed_seconds_total") or 0.0)
    limit = limits.max_shift_seconds
    passed = worst <= limit + SHIFT_TOLERANCE_S
    return GateResult(
        "sync",
        "pass" if passed else "fail",
        f"worst line {worst:.2f}s late (limit {limit:.2f}s); {shifted} line(s) "
        f"moved; {overrun:.2f}s spilled into pauses; {trimmed:.2f}s trimmed",
    )


def _truncation_gate(report: Mapping[str, Any], limits: QCThresholds) -> GateResult:
    """Gate on deleted words, which is the failure that loses information."""
    worst = float(report.get("worst_line_deleted_ratio") or 0.0)
    over = int(report.get("lines_over_10pct_deleted") or 0)
    passed = (
        worst <= limits.max_worst_line_deleted_ratio
        and over <= limits.max_lines_over_10pct_deleted
    )
    return GateResult(
        "truncation",
        "pass" if passed else "fail",
        f"worst line lost {worst * 100:.1f}% of its speech (limit "
        f"{limits.max_worst_line_deleted_ratio * 100:.0f}%); {over} line(s) over 10% "
        f"(limit {limits.max_lines_over_10pct_deleted})",
    )


def _loudness_gate(report: Mapping[str, Any], limits: QCThresholds) -> GateResult:
    """Gate on the delivered level, when the source level was measurable."""
    delta = report.get("loudness_delta_vs_source")
    if delta is None:
        return GateResult(
            "loudness",
            "skip",
            "the source loudness could not be measured, so there is nothing to compare",
        )
    value = abs(float(delta))
    detail = (
        f"delivered {value:.2f} dB from the source (limit "
        f"{limits.max_loudness_delta_db:g} dB)"
    )
    if report.get("loudness_peak_limited"):
        detail += "; peaks were limited to reach the target"
    return GateResult(
        "loudness",
        "pass" if value <= limits.max_loudness_delta_db else "fail",
        detail,
    )


def timbre_available(job_dir: Path, accent: Path | None = None) -> bool:
    """Whether per-line timbre can be measured for this job at all.

    The measurement needs the clip the pronunciation was borrowed from. A
    ``preserve`` job never had one, so asking whether its lines followed the
    accent voice is not a failure - it is a question with no answer, which the
    gate reports as *skipped* rather than aborting the whole check.
    """
    chosen = accent or accent_path_for(job_dir)
    return chosen is not None and Path(chosen).is_file()


def _timbre_gate(
    timbre: Sequence[Mapping[str, Any]] | None,
    limits: QCThresholds,
    reason: str | None = None,
) -> GateResult:
    """Gate on lines that followed the accent voice instead of the speaker's."""
    if timbre is None:
        return GateResult(
            "timbre", "skip", reason or "not measured; pass --timbre to run it"
        )

    rows = 0
    collapsed = 0
    worst = 0.0
    parts: list[str] = []
    for result in timbre:
        count = len(result.get("rows") or [])
        missed = len(result.get("collapsed") or [])
        share = missed / count if count else 0.0
        rows += count
        collapsed += missed
        worst = max(worst, share)
        parts.append(f"{result.get('speaker') or 'voice'} {share * 100:.1f}%")

    if rows == 0:
        return GateResult("timbre", "skip", "no dubbed lines to measure")

    return GateResult(
        "timbre",
        "pass" if worst <= limits.max_accent_side_share else "fail",
        f"{collapsed} of {rows} line(s) closer to the accent voice than to the "
        f"intended one ({', '.join(parts)}); limit "
        f"{limits.max_accent_side_share * 100:.0f}%",
    )


def evaluate(
    report: Mapping[str, Any],
    *,
    timbre: Sequence[Mapping[str, Any]] | None = None,
    thresholds: QCThresholds | None = None,
    timbre_reason: str | None = None,
) -> list[GateResult]:
    """Check a job's report against the quality thresholds.

    Args:
        report: The parsed ``report.json``.
        timbre: Per-voice results from :func:`audit_job`, when the timbre gate was
            run. ``None`` means it was not, which is reported as *skipped* rather
            than as a pass.
        thresholds: Limits to apply; the defaults are the roadmap's acceptance
            levels.
        timbre_reason: Why the timbre gate did not run, when the caller knows - for
            instance a job with no accent clip to measure against. Without it the
            skip reads as "not requested", which would be wrong.

    Returns:
        One :class:`GateResult` per gate, in a stable order.
    """
    limits = thresholds or QCThresholds()
    return [
        _sync_gate(report, limits),
        _truncation_gate(report, limits),
        _loudness_gate(report, limits),
        _timbre_gate(timbre, limits, timbre_reason),
    ]


def summarize(results: Sequence[GateResult]) -> dict[str, Any]:
    """Machine-readable summary of a gate run, for ``--json`` and for CI."""
    failed = [result.name for result in results if result.status == "fail"]
    return {
        "passed": not failed,
        "failed": failed,
        "skipped": [result.name for result in results if result.status == "skip"],
        "gates": [result.to_dict() for result in results],
    }


def _read_report(job_dir: Path) -> dict[str, Any]:
    """Load a job's report, or explain why it cannot be checked."""
    path = job_dir / REPORT_NAME
    if not path.is_file():
        raise AuditError(
            f"{job_dir} has no {REPORT_NAME}; run the job first, or point at a "
            "completed job directory."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{path} does not contain a job report.")
    return payload


def qc(
    job_dir: Path,
    *,
    engine: TTSEngine | None = None,
    accent: Path | None = None,
    reference: Path | None = None,
    thresholds: QCThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate every applicable gate for one job folder.

    The cheap gates read ``report.json`` alone, so they need no model and no GPU.
    The timbre gate only runs when an ``engine`` is supplied, because it needs the
    checkpoint's speaker encoder; without one it is reported as skipped. Supplying
    an engine for a job that has no accent clip is also a skip, not an error: there
    is nothing to measure the lines against, so "skipped" is the honest answer and
    the other gates still report.

    Args:
        job_dir: The job folder to check.
        engine: Loaded TTS engine, which enables the timbre gate.
        accent: Accent clip for the timbre gate; defaults to ``donor.json``.
        reference: Force one measuring stick for every voice (deliberate A/B only).
        thresholds: Limits to apply.

    Returns:
        A summary dict: ``passed``, ``failed``, ``skipped`` and the per-gate list.

    Raises:
        AuditError: When the report or a required clip is missing.
    """
    report = _read_report(job_dir)
    timbre = None
    reason = None
    if engine is not None:
        if timbre_available(job_dir, accent):
            timbre = audit_job(job_dir, engine, accent=accent, reference=reference)
        else:
            reason = (
                "no accent clip to measure against; needs a reference-accent job "
                "or --accent"
            )
    payload = summarize(
        evaluate(report, timbre=timbre, thresholds=thresholds, timbre_reason=reason)
    )
    payload["job_dir"] = str(job_dir)
    return payload
