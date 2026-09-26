#!/usr/bin/env python3
"""Measure, per dubbed line, whether the speaker's timbre survived.

Answering "did the voice change?" by ear is unreliable: the artifact is subtle,
it moves between runs, and it is invisible in an average. This measures it line by
line with the checkpoint's own speaker encoder.

The measurement has to be **language-matched**. Comparing a French dubbed line to
the donor clip (English, in a typical job) folds a language difference into a
speaker measurement, and the encoder is sensitive to it. So each line is compared
against two clips that are both in the target language:

    reference : the voice the line was supposed to be in. In `reference-accent`
                mode that is `04_dubbed/reference_blend.wav`, the stage-1 blend.
    accent    : the clip the pronunciation was borrowed from.

    margin = cos(line, reference) - cos(line, accent)

Both references speak the target language, so the only thing separating them is
whose voice it is. Two anchors give the number a scale:

    collapsed anchor : the accent clip's own utterances  (should be negative)
    target anchor    : the reference against itself      (= 1 - cos(ref, accent))

A line near the collapsed anchor followed the accent voice. Anything at or above
zero at least resembles the intended voice more than the borrowed one.

Usage:
    python audit_timbre.py output/<job>

    # A job folder with no reference_blend.wav (single-stage, or an older run)
    # needs the measuring stick supplied: both clips, since a trimmed folder may
    # also have lost the donor.json that names the accent clip.
    python audit_timbre.py out_transcript_fix/<job> \\
        --reference output/<job>/04_dubbed/reference_blend.wav \\
        --accent input/<accent-clip>

To compare a `single-stage` job against a `two-stage` one, point `--reference`
at the two-stage job's blend clip: it then acts as the same measuring stick for
both, which is what makes the two comparable.

A job with several speakers is audited one speaker at a time, each against their
own `04_dubbed/reference_blend_<speaker>.wav`. Each group is a separate scale, so
the anchors are printed per group rather than once - two voices have two
different "pure accent" floors. `--reference` overrides that and measures
everyone against one clip, which is only useful for a deliberate A/B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pipeline import media
from pipeline.tts import TTSEngine

#: Where the base checkpoint lives in this repo.
DEFAULT_MODEL_DIR = Path("models/Qwen3-TTS-12Hz-1.7B-Base")
BLEND_NAME = "04_dubbed/reference_blend.wav"
BLEND_PATTERN = "04_dubbed/reference_blend_{speaker}.wav"
DONOR_NAME = "04_dubbed/donor.json"
SPEAKERS_NAME = "02b_speakers.json"
ANCHOR_WINDOW_S = 7.0


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def segment_ids(job_dir: Path) -> set[int]:
    """Ids this job actually produced, so stale clips from older runs are ignored.

    Nothing used to clean ``04_dubbed/``, so a job re-run with a different
    segmentation can still hold orphaned clips from the previous one.
    """
    for name in ("timing.json", "02_transcript.json"):
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
    """
    windows = [
        accent_samples[int(start * accent_rate) : int((start + ANCHOR_WINDOW_S) * accent_rate)]
        for start in (0.0, ANCHOR_WINDOW_S)
    ]
    return float(
        np.mean(
            [margin_for(engine, window, accent_rate, reference, accent)[0] for window in windows]
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
) -> dict:
    """Measure one group of lines against one reference clip.

    Args:
        speaker: Speaker this group belongs to, for the report; ``None`` when the
            job has a single voice.
        only_ids: Restrict the clips to these segment ids. ``None`` audits all of
            the job's own lines.
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
    accent = json.loads(path.read_text(encoding="utf-8")).get("accent", {})
    return Path(accent["path"]) if accent.get("path") else None


def report(result: dict) -> None:
    label = f"  [{result['speaker']}]" if result.get("speaker") else ""
    print("=" * 78)
    print(f"{result['job_dir']}{label}")
    if result.get("speaker"):
        print(f"  reference     : {result['reference']}")
    print(f"  lines audited : {len(result['rows'])}"
          + (f"   ({result['skipped']} orphaned clip(s) ignored)" if result["skipped"] else ""))
    print("  scale for this reference (language-matched):")
    print(f"    collapsed anchor {result['collapsed_anchor']:+.4f}   "
          f"target anchor {result['target_anchor']:+.4f}")
    print("  margin cos(line, reference) - cos(line, accent):")
    print(f"    min {result['min']:+.4f}   median {result['median']:+.4f}   max {result['max']:+.4f}")
    print(f"    closer to the accent voice than to the reference: {len(result['collapsed'])}")
    worst = sorted(result["rows"], key=lambda row: row[3])[:5]
    print("  worst 5 lines:")
    for name, to_ref, to_accent, margin in worst:
        print(f"    {name}  reference={to_ref:.4f}  accent={to_accent:.4f}  margin={margin:+.4f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-line speaker-timbre audit of a dubbed job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("jobs", nargs="+", help="Job folders to audit.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "The clip the lines were supposed to be cloned from. Defaults to each "
            "job's own 04_dubbed/reference_blend.wav; pass an explicit path to "
            "measure a single-stage job against the same stick."
        ),
    )
    parser.add_argument(
        "--accent",
        type=Path,
        default=None,
        help="Accent clip, when a job folder has no donor.json to read it from.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=f"Models directory. (default: {DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args(argv)

    model_dir = args.models_dir or DEFAULT_MODEL_DIR
    if not model_dir.is_dir():
        print(f"Model not found at {model_dir}. Pass --models-dir.", file=sys.stderr)
        return 2

    print(f"Loading {model_dir.name} ...", flush=True)
    engine = TTSEngine(model_dir, device="auto")

    results = []
    for job in args.jobs:
        job_dir = Path(job)
        accent = args.accent or accent_path_for(job_dir)
        if accent is None or not Path(accent).is_file():
            print(f"{job_dir}: no accent clip found; pass --accent.", file=sys.stderr)
            return 2

        labels = speaker_labels(job_dir)
        voices = speaker_order(labels)
        if args.reference is not None or len(voices) < 2:
            # One voice (or a measuring stick chosen by hand): every line is
            # measured against the same clip, as before.
            reference = args.reference or (job_dir / BLEND_NAME)
            if not reference.is_file():
                print(
                    f"{job_dir}: no reference clip at {reference}.\n"
                    "  A single-stage job produces none - pass --reference pointing at "
                    "another job's reference_blend.wav.",
                    file=sys.stderr,
                )
                return 2
            results.append(audit(job_dir, engine, reference, accent))
            continue

        # Several voices: each speaker is measured against their own blend clip,
        # so a line is never judged against another voice.
        for name in voices:
            reference = reference_for(job_dir, name, None)
            if not reference.is_file():
                print(
                    f"{job_dir}: no blend clip for {name} at {reference}.",
                    file=sys.stderr,
                )
                return 2
            owned = {sid for sid, who in labels.items() if who == name}
            results.append(
                audit(job_dir, engine, reference, accent, speaker=name, only_ids=owned)
            )

    for result in results:
        report(result)

    print("SUMMARY")
    for result in results:
        share = 100.0 * len(result["collapsed"]) / max(1, len(result["rows"]))
        label = f" {result['speaker']}" if result.get("speaker") else ""
        print(
            f"  min={result['min']:+.4f}  median={result['median']:+.4f}  "
            f"closer-to-accent={share:5.1f}%{label}  ({result['job_dir']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
