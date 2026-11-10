#!/usr/bin/env python3
"""Automated quality gate for a dubbed job.

Dubbing writes artifacts and a ``report.json``; this checks them. It runs the
cheap signals — sync, truncation, loudness — straight from the report, and can
add a per-line speaker-timbre audit when a GPU is available. The point is that
"the run was fine" becomes a checked claim instead of a feeling.

The gate never blocks the pipeline: the dub is complete before this runs, so a
failure is a verdict on the delivery rather than a reason to lose the work. That
also makes it usable in CI, or by hand, on any completed job folder.

Usage:
    python audit_qc.py output/<job>

    # Also measure speaker timbre (needs the TTS checkpoint).
    python audit_qc.py output/<job> --timbre

    # Machine-readable, for a script or a bake-off harness.
    python audit_qc.py output/<job> --json

Exit status is 0 when every gate passes, 1 when one fails, and 2 when the job
could not be checked at all (no report, missing clip, model not found).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.audit import QCThresholds, qc
from pipeline.config import normalize_device_request
from pipeline.errors import ConfigError, DubbingError

#: Where the base checkpoint lives in this repo, for the timbre gate.
DEFAULT_MODEL_DIR = Path("models/Qwen3-TTS-12Hz-1.7B-Base")

_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_UNUSABLE = 2


def _device_arg(value: str) -> str:
    """Validate ``--device`` with the same rule as the main CLI."""
    try:
        return normalize_device_request(value)
    except ConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    """Parser for the quality-gate command."""
    parser = argparse.ArgumentParser(
        prog="audit_qc.py",
        description="Check a dubbed job against the project's quality gates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit status is 0 when every gate passes, 1 when one fails, and 2 when "
            "a job could not be checked at all."
        ),
    )
    parser.add_argument("jobs", nargs="+", help="Job folders to check.")
    parser.add_argument(
        "--timbre",
        action="store_true",
        help=(
            "Also measure speaker timbre per line. Needs the TTS checkpoint; "
            "without it the timbre gate is reported as skipped."
        ),
    )
    parser.add_argument(
        "--accent",
        type=Path,
        default=None,
        help="Accent clip, when the job folder has no donor.json to name it.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Measuring stick for the timbre audit. Defaults to each job's own blend "
            "clip; pass one explicitly to compare jobs on the same scale."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=f"Models directory. (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--device",
        type=_device_arg,
        default="auto",
        help=(
            "Device for the timbre measurement. `auto` picks the GPU when one is "
            "available; `cpu` measures on the CPU, which is slower but works when "
            "the GPUs are busy with something else. (default: auto)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable results and nothing else.",
    )
    parser.add_argument(
        "--max-shift",
        type=float,
        default=None,
        help=(
            "Seconds a line may start later than its source before the sync gate "
            "fails. (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--max-deleted-ratio",
        type=float,
        default=None,
        help="Maximum share of a single line that may be trimmed. (default: 0.10)",
    )
    parser.add_argument(
        "--max-deleted-lines",
        type=int,
        default=None,
        help="Maximum lines allowed to lose over 10%% of their speech. (default: 0)",
    )
    parser.add_argument(
        "--max-loudness-delta",
        type=float,
        default=None,
        help="Maximum dB between the delivered dub and the source. (default: 1.5)",
    )
    parser.add_argument(
        "--max-accent-share",
        type=float,
        default=None,
        help=(
            "Maximum share of lines allowed to sit closer to the accent voice. "
            "(default: 0)"
        ),
    )
    return parser


def thresholds_from(args: argparse.Namespace) -> QCThresholds:
    """Apply only the overrides the user passed, so the defaults stay in code."""
    defaults = QCThresholds()
    return QCThresholds(
        max_shift_seconds=(
            args.max_shift
            if args.max_shift is not None
            else defaults.max_shift_seconds
        ),
        max_worst_line_deleted_ratio=(
            args.max_deleted_ratio
            if args.max_deleted_ratio is not None
            else defaults.max_worst_line_deleted_ratio
        ),
        max_lines_over_10pct_deleted=(
            args.max_deleted_lines
            if args.max_deleted_lines is not None
            else defaults.max_lines_over_10pct_deleted
        ),
        max_loudness_delta_db=(
            args.max_loudness_delta
            if args.max_loudness_delta is not None
            else defaults.max_loudness_delta_db
        ),
        max_accent_side_share=(
            args.max_accent_share
            if args.max_accent_share is not None
            else defaults.max_accent_side_share
        ),
    )


def _load_engine(models_dir: Path | None, device: str = "auto"):
    """Load the TTS engine, imported lazily so a cheap run stays cheap."""
    from pipeline.config import configure_allocator
    from pipeline.tts import TTSEngine

    resolved = models_dir or DEFAULT_MODEL_DIR
    if not resolved.is_dir():
        raise DubbingError(
            f"Model not found at {resolved}. Pass --models-dir, or run without "
            "--timbre to check the cheap gates only."
        )
    print(f"Loading {resolved.name} ...", file=sys.stderr, flush=True)
    # Before the model loads: torch reads the allocator setting at the first
    # allocation, so a long audit would otherwise never see it.
    configure_allocator(device)
    return TTSEngine(resolved, device=device)


def print_report(payload: dict) -> None:
    """One human-readable block per job."""
    print("=" * 78)
    print(payload["job_dir"])
    for gate in payload["gates"]:
        print(f"  {gate['status'].upper():<4} {gate['name']:<11} {gate['detail']}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Entry point: check every job and return a shell-friendly status."""
    args = build_parser().parse_args(argv)
    thresholds = thresholds_from(args)

    engine = None
    if args.timbre:
        try:
            engine = _load_engine(args.models_dir, args.device)
        except DubbingError as exc:
            print(str(exc), file=sys.stderr)
            return _EXIT_UNUSABLE

    payloads: list[dict] = []
    for job in args.jobs:
        try:
            payloads.append(
                qc(
                    Path(job),
                    engine=engine,
                    accent=args.accent,
                    reference=args.reference,
                    thresholds=thresholds,
                )
            )
        except DubbingError as exc:
            print(str(exc), file=sys.stderr)
            return _EXIT_UNUSABLE

    if args.json:
        # A single job stays an object so a script can read it directly; several
        # jobs become a list. Either way the shape is stable for one input.
        document = payloads[0] if len(payloads) == 1 else payloads
        print(json.dumps(document, indent=2))
    else:
        for payload in payloads:
            print_report(payload)
        print("SUMMARY")
        for payload in payloads:
            status = "PASS" if payload["passed"] else "FAIL"
            print(
                f"  {status}  failed: {', '.join(payload['failed']) or 'none'}"
                f"   skipped: {', '.join(payload['skipped']) or 'none'}"
                f"   ({payload['job_dir']})"
            )

    return _EXIT_OK if all(payload["passed"] for payload in payloads) else _EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
