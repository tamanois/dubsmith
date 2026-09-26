#!/usr/bin/env python3
"""Command-line entry point for the local video dubbing pipeline.

Examples
--------
Fetch the models the MVP needs (they are gitignored, never committed)::

    python dub.py models --download all

Dub a video into French, cloning the speaker's own voice::

    python dub.py --video talk.mp4 --target fr

Dub using an explicit reference voice::

    python dub.py --audio clip.wav --target fr --ref-audio me.wav --ref-text "Bonjour…"

Run ``python dub.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pipeline.config import SUPPORTED_LANGUAGES, JobConfig, Settings
from pipeline.errors import DubbingError, ModelNotFoundError
from pipeline.logging_utils import setup_logging
from pipeline.models import MVP_REQUIRED, REGISTRY, download_many, registry_status
from pipeline.orchestrate import Pipeline

_PROG = "dub.py"
_STOP = 0


def build_models_parser() -> argparse.ArgumentParser:
    """Parser for the ``models`` subcommand."""
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} models",
        description="List or download the models used by the pipeline.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Show every known model and whether it is present."
    )
    parser.add_argument(
        "--download",
        metavar="NAME|all",
        help="Download one model by name, or 'all' for the MVP-required set.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even when already present."
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Override the models directory."
    )
    return parser


def build_dub_parser() -> argparse.ArgumentParser:
    """Parser for the default (dubbing) command."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Dub a single-speaker video (or audio file) into a target language using "
            "the speaker's own voice timbre."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Input video file.")
    source.add_argument("--audio", type=Path, help="Input audio file (no video muxing).")

    parser.add_argument(
        "--target",
        dest="target_language",
        default="fr",
        help=f"Target language ({', '.join(sorted(SUPPORTED_LANGUAGES))}).",
    )
    parser.add_argument(
        "--source",
        dest="source_language",
        default=None,
        help="Source language; omit to auto-detect.",
    )

    parser.add_argument("--ref-audio", type=Path, default=None, help="Reference voice clip.")
    parser.add_argument(
        "--ref-text",
        default=None,
        help=(
            "Exact transcript of --ref-audio. Required with it in 'preserve' mode "
            "(where the model conditions on both); ignored by the other modes."
        ),
    )

    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output root.")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu.")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model id.")
    parser.add_argument("--llm-model", default=None, help="Override the configured LLM model.")
    parser.add_argument(
        "--stretcher",
        default="auto",
        choices=["auto", "rubberband", "sox", "librosa", "none"],
        help="Time-stretch backend.",
    )
    parser.add_argument(
        "--min-stretch",
        type=float,
        default=0.85,
        help=(
            "Maximum compression applied to fit a line into its slot (lower = allow "
            "faster speech). Lines still too long are allowed to overrun rather "
            "than being cut."
        ),
    )
    parser.add_argument(
        "--max-stretch",
        type=float,
        default=1.18,
        help="Maximum slow-down applied to fill a slot.",
    )
    parser.add_argument(
        "--loudness",
        dest="loudness_target",
        default=None,
        choices=["match-source", "lufs"],
        help=(
            "Loudness policy: 'match-source' (default) sits the dub at the level "
            "of the original track; 'lufs' forces --lufs."
        ),
    )
    parser.add_argument(
        "--lufs",
        dest="target_lufs",
        type=float,
        default=None,
        help="Absolute loudness target in LUFS, used when --loudness lufs.",
    )
    parser.add_argument(
        "--no-background",
        dest="keep_background",
        action="store_false",
        default=True,
        help=(
            "Discard the original music and ambience instead of preserving it. "
            "By default the background is separated out and ducked under the dub."
        ),
    )
    parser.add_argument(
        "--separator",
        default="auto",
        choices=["auto", "demucs", "none"],
        help="Backend used to split speech from background.",
    )
    parser.add_argument(
        "--duck-db",
        type=float,
        default=6.0,
        help="How far to pull the background down while dialogue plays, in dB.",
    )
    parser.add_argument(
        "--accent-mode",
        dest="accent_mode",
        default="preserve",
        choices=["preserve", "speaker-only", "reference-accent"],
        help=(
            "'preserve' clones the speaker's accent along with their voice; "
            "'speaker-only' keeps only the timbre embedding so the model uses its "
            "own phonetics for the target language; 'reference-accent' keeps the "
            "speaker's timbre but takes pronunciation from --accent-audio."
        ),
    )
    parser.add_argument(
        "--accent-audio",
        type=Path,
        default=None,
        help=(
            "Clip of a native target-language speaker whose pronunciation the dub "
            "should adopt. Requires --accent-text and --accent-mode reference-accent."
        ),
    )
    parser.add_argument(
        "--accent-text",
        default=None,
        help="Exact transcript of --accent-audio (required with it).",
    )
    parser.add_argument(
        "--accent-blend",
        dest="accent_blend",
        default=None,
        choices=["two-stage", "single-stage"],
        help=(
            "'two-stage' (default) renders the accent transcript once in the "
            "speaker's own voice and clones every line from that clip, so the "
            "speech codes and the speaker embedding always agree; 'single-stage' "
            "conditions each line on both sources at once."
        ),
    )
    parser.add_argument(
        "--blend-takes",
        dest="blend_takes",
        type=int,
        default=None,
        help=(
            "Takes rendered for the one-off blended reference, keeping the take "
            "closest to the speaker's timbre and furthest from the accent clip. "
            "Only the blend is sampled this way. (default: 3)"
        ),
    )
    parser.add_argument(
        "--blend-iters",
        dest="blend_iters",
        type=int,
        default=None,
        help=(
            "Passes used to build the one-off blended reference. Each pass "
            "re-encodes the previous blend and re-injects the speaker's embedding, "
            "pulling the voice back toward the speaker; it stops early once "
            "--blend-ratio is reached. Raise it if the speaker still sounds too "
            "faint. 1 keeps the single-pass behaviour. (default: 4)"
        ),
    )
    parser.add_argument(
        "--blend-ratio",
        dest="blend_ratio",
        type=float,
        default=None,
        help=(
            "How far toward the speaker's own voice the blend should reach, from 0 "
            "(the accent clip's voice) to 1 (a French utterance in the speaker's own "
            "voice). Iteration stops there. 0 disables the search. (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        default=True,
        help=(
            "Rewrite and re-synthesise the lines that would otherwise have words "
            "cut, so a long translation is fitted by wording rather than trimming. "
            "Lines that already fit are left untouched. (default: on)"
        ),
    )
    parser.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Disable the length repair and trim overlong lines as before.",
    )
    parser.add_argument(
        "--refine-max-passes",
        dest="refine_max_passes",
        type=int,
        default=1,
        help="How many times a line may be rewritten before trimming. (default: 1)",
    )
    parser.add_argument(
        "--refine-threshold",
        dest="refine_threshold",
        type=float,
        default=0.05,
        help=(
            "Seconds a line must overrun its slot (and the following pause) "
            "before it is worth rewriting. (default: 0.05)"
        ),
    )
    parser.add_argument(
        "--length-budget",
        dest="length_budget",
        default="off",
        choices=["off", "words"],
        help=(
            "'words' also tells the translator how many words fit each slot, "
            "measured from a target-language reference clip; 'off' sends only the "
            "duration, as before. Needs --accent-audio or a target-language "
            "--ref-audio. (default: off)"
        ),
    )
    parser.add_argument(
        "--no-subtitles", action="store_true", help="Skip writing SRT files."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached stages.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_models(argv: Sequence[str]) -> int:
    """Handle ``dub.py models``."""
    args = build_models_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir

    if args.download:
        requested = args.download.strip().lower()
        if requested in {"all", "mvp"}:
            names: list[str] = list(MVP_REQUIRED)
        elif requested == "everything":
            names = list(REGISTRY)
        else:
            if requested not in REGISTRY:
                known = ", ".join(sorted(REGISTRY))
                print(f"Unknown model {requested!r}. Known models: {known}", file=sys.stderr)
                return 2
            names = [requested]

        for name in names:
            spec = REGISTRY[name]
            print(f"→ {name} ({spec.repo_id})", flush=True)
            path = download_many([name], models_dir, force=args.force)[name]
            print(f"  saved to {path}")
        print("\nAll requested models are ready.")
        return _STOP

    # Default and --list both show the registry table.
    print(f"Models directory: {Path(models_dir).resolve()}\n")
    print(f"{'name':<18}{'status':<12}{'repo':<44}description")
    for spec, present in registry_status(models_dir):
        status = "present" if present else "missing"
        print(f"{spec.name:<18}{status:<12}{spec.repo_id:<44}{spec.description}")
    print(
        "\nMVP requires: "
        + ", ".join(MVP_REQUIRED)
        + "\nDownload with: python dub.py models --download all"
    )
    return _STOP


def cmd_dub(argv: Sequence[str]) -> int:
    """Handle the default dubbing command."""
    args = build_dub_parser().parse_args(list(argv))
    settings = Settings.from_env()
    setup_logging("DEBUG" if args.verbose else settings.log_level, settings.log_json)

    job = JobConfig(
        video=args.video,
        audio=args.audio,
        target_language=args.target_language,
        source_language=args.source_language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_dir=args.output_dir,
        device=args.device,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        min_stretch=args.min_stretch,
        max_stretch=args.max_stretch,
        loudness_target=args.loudness_target,
        target_lufs=args.target_lufs,
        keep_background=args.keep_background,
        separator=args.separator,
        duck_db=args.duck_db,
        accent_mode=args.accent_mode,
        accent_audio=args.accent_audio,
        accent_text=args.accent_text,
        accent_blend=args.accent_blend,
        blend_takes=args.blend_takes,
        blend_iters=args.blend_iters,
        blend_ratio=args.blend_ratio,
        refine=args.refine,
        refine_max_passes=args.refine_max_passes,
        refine_threshold=args.refine_threshold,
        length_budget=args.length_budget,
        keep_subtitles=not args.no_subtitles,
        force=args.force,
    )

    result = Pipeline(job, settings, stretcher_preference=args.stretcher).run()
    print()
    print(result.summary())
    return _STOP


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; dispatches on the first argument."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in {"models", "download-models"}:
        handler = cmd_models
        payload = args[1:]
    else:
        handler = cmd_dub
        payload = args

    try:
        return handler(payload)
    except ModelNotFoundError as exc:
        print(f"\nMissing model: {exc}", file=sys.stderr)
        if exc.remedy:
            print(f"Fix it with:\n    {exc.remedy}", file=sys.stderr)
        return 1
    except DubbingError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
