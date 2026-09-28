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
import sys
from pathlib import Path

from pipeline.audit import audit_job
from pipeline.errors import AuditError
from pipeline.tts import TTSEngine

#: Where the base checkpoint lives in this repo.
DEFAULT_MODEL_DIR = Path("models/Qwen3-TTS-12Hz-1.7B-Base")


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
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device for the speaker encoder. `auto` picks the GPU when one is "
            "available; `cpu` is slower but works when the GPUs are busy. "
            "(default: auto)"
        ),
    )
    args = parser.parse_args(argv)

    model_dir = args.models_dir or DEFAULT_MODEL_DIR
    if not model_dir.is_dir():
        print(f"Model not found at {model_dir}. Pass --models-dir.", file=sys.stderr)
        return 2

    print(f"Loading {model_dir.name} ...", flush=True)
    engine = TTSEngine(model_dir, device=args.device)

    results = []
    for job in args.jobs:
        try:
            results.extend(
                audit_job(
                    Path(job), engine, accent=args.accent, reference=args.reference
                )
            )
        except AuditError as exc:
            print(str(exc), file=sys.stderr)
            return 2

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
