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
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

from pipeline import __version__
from pipeline.asr import asr_device, resolve_compute_type
from pipeline.config import (
    DEFAULT_BUDGET_SPEEDUP,
    DEFAULT_MIN_SPEAKER_S,
    SUPPORTED_LANGUAGES,
    JobConfig,
    Settings,
    available_accelerators,
    normalize_language,
    pick_dtype,
    probe_device,
    release_gpu_memory,
)
from pipeline.errors import DubbingError, ModelNotFoundError
from pipeline.logging_utils import DEFAULT_MAX_LOG_BYTES, setup_logging
from pipeline.mixing import (
    DEFAULT_ROOM_DUCK_DB,
    DEFAULT_ROOM_TONE_GAIN_DB,
    DEFAULT_ROOM_WET_DB,
)
from pipeline.models import (
    MVP_REQUIRED,
    REGISTRY,
    download_many,
    registry_status,
    resolve_model,
)
from pipeline.orchestrate import Pipeline

_PROG = "dub.py"
_STOP = 0

#: Name of the run log kept inside a job folder when no path is given.
_LOG_NAME = "run.log"

#: Optional dependencies and what each one unlocks. Reported by ``dub.py info``
#: because a missing one degrades one feature rather than failing a run.
_OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("demucs", "keep the original music and ambience (--no-background disables)"),
    ("funasr", "multi-speaker dubbing (--diarizer none disables)"),
    ("speechbrain", "overlapping-speech separation (--overlap separate)"),
)

#: System binaries the media layer looks for. Only ffmpeg is ever required.
_SYSTEM_BINARIES: tuple[tuple[str, str], ...] = (
    ("ffmpeg", "required for video input and muxing"),
    ("ffprobe", "required for video input"),
    ("rubberband", "best-quality time stretching"),
    ("sox", "alternative time stretching"),
)


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


def build_anchors_parser() -> argparse.ArgumentParser:
    """Parser for the ``anchors`` subcommand."""
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} anchors",
        description=(
            "List or generate the native accent anchors used by the default "
            "'native' accent mode. An anchor is a short native-language narration "
            "synthesised by the VoiceDesign model; one per language is enough, and "
            "it is generated once and cached."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate the requested anchors (without it, list what is cached).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List cached anchors (the default)."
    )
    parser.add_argument(
        "--lang",
        action="append",
        default=None,
        metavar="LANG",
        help=(
            "Language to generate, by code or name (repeatable). Defaults to every "
            "supported language."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Precision for the VoiceDesign checkpoint (default: auto).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate anchors that already exist."
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
            "Dub a video (or audio file) into a target language using each "
            "speaker's own voice timbre. Run 'info' first on unfamiliar hardware."
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
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cuda:1, mps, or cpu.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help=(
            "Precision for the voice-cloning checkpoint. 'auto' reads the "
            "device's capability: bf16 on Ampere and newer, fp16 on older "
            "accelerators, fp32 on CPU. An explicit value is honoured unless "
            "the device cannot run it."
        ),
    )
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model id.")
    parser.add_argument("--llm-model", default=None, help="Override the configured LLM model.")
    parser.add_argument(
        "--tts-batch",
        type=int,
        default=8,
        help=(
            "Most lines synthesised in one model call. The engine also caps a "
            "batch by the audio it predicts, because memory tracks the total "
            "seconds in the request rather than the number of lines; this is the "
            "ceiling, and the escape hatch for a machine with less headroom. "
            "Lower it if a long video still runs out of GPU memory."
        ),
    )
    parser.add_argument(
        "--stretcher",
        default="auto",
        choices=["auto", "rubberband", "sox", "librosa", "none"],
        help="Time-stretch backend.",
    )
    parser.add_argument(
        "--min-stretch",
        type=float,
        default=0.75,
        help=(
            "Compression floor: how much faster than natural a line may be played "
            "to fit its slot (lower = allow faster speech). Past it the wording has "
            "to give instead, because the alternative is deleting words."
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
        "--room",
        default="off",
        choices=["off", "match"],
        help=(
            "'match' places the dub in the source's own acoustic space: a reverb "
            "tail, a continuous room-tone floor and a shallower duck, all derived "
            "from the separated speech stem. 'off' (default) keeps the dry "
            "voice-over, which is the right choice for narrated content. Needs the "
            "background to be kept (the default)."
        ),
    )
    parser.add_argument(
        "--room-rt60",
        dest="room_rt60",
        type=float,
        default=None,
        help=(
            "Reverberation time in seconds, instead of measuring it from the "
            "speech stem. Useful when the source's decay cannot be estimated."
        ),
    )
    parser.add_argument(
        "--room-wet-db",
        dest="room_wet_db",
        type=float,
        default=DEFAULT_ROOM_WET_DB,
        help=(
            "Reverb level relative to the dry voice. 0 (the default) adds NO "
            "reverb, which is what the ear test called for; a negative value such "
            "as -20 enables the tail. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--room-duck-db",
        dest="room_duck_db",
        type=float,
        default=DEFAULT_ROOM_DUCK_DB,
        help=(
            "How far to pull the bed down under dialogue when --room match. "
            "Shallower than --duck-db, because the room is part of the voice. "
            f"(default: {DEFAULT_ROOM_DUCK_DB:g})"
        ),
    )
    parser.add_argument(
        "--room-tone-db",
        dest="room_tone_gain_db",
        type=float,
        default=DEFAULT_ROOM_TONE_GAIN_DB,
        help=(
            "Level of the room-tone floor taken from the speech stem's non-speech "
            f"parts. (default: {DEFAULT_ROOM_TONE_GAIN_DB:g})"
        ),
    )
    parser.add_argument(
        "--accent-mode",
        dest="accent_mode",
        default="native",
        choices=["native", "preserve", "speaker-only", "reference-accent"],
        help=(
            "'native' (default) keeps the speaker's timbre and takes pronunciation "
            "from a generated native accent anchor; 'preserve' clones the speaker's "
            "accent along with their voice; 'speaker-only' keeps only the timbre "
            "embedding so the model uses its own phonetics for the target language; "
            "'reference-accent' keeps the speaker's timbre but takes pronunciation "
            "from --accent-audio."
        ),
    )
    parser.add_argument(
        "--accent-match",
        dest="accent_match",
        default="off",
        choices=["off", "rate", "gender"],
        help=(
            "With --accent-mode native, generate a per-speaker accent anchor that "
            "fits the source voice: 'rate' matches its speaking rate, 'gender' also "
            "matches its gender. (default: off)"
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
            "faint. 1 keeps the single-pass behaviour. (default: 5)"
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
            "voice). Iteration stops there. 0 disables the search. (default: 0.85)"
        ),
    )
    parser.add_argument(
        "--diarizer",
        dest="diarizer",
        default="auto",
        choices=["auto", "none"],
        help=(
            "'auto' separates the speakers and clones each voice from its own "
            "donor; 'none' treats the video as a single speaker, as before. "
            "Separating needs the optional funasr dependency."
        ),
    )
    parser.add_argument(
        "--speaker-threshold",
        dest="speaker_threshold",
        type=float,
        default=None,
        help=(
            "Cosine distance at which two voices count as different speakers "
            "(default: 0.7). Raise it if one speaker is split in two; lower it if "
            "two speakers are merged into one."
        ),
    )
    parser.add_argument(
        "--min-speaker-seconds",
        dest="min_speaker_seconds",
        type=float,
        default=DEFAULT_MIN_SPEAKER_S,
        help=(
            "A voice must own at least this much speech to be dubbed separately; "
            "a shorter cluster is folded into its nearest neighbour. Well above "
            "the 3 s needed to clone at all, because a donor of a second or two "
            "does not describe a voice and its blend stays near the accent. "
            "0 keeps every cluster."
        ),
    )
    parser.add_argument(
        "--overlap",
        dest="overlap_mode",
        default="dominant",
        choices=["dominant", "separate"],
        help=(
            "How to handle a segment that carries two voices at once. 'dominant' "
            "dubs the whole line in its dominant speaker's voice (the default); "
            "'separate' splits it, re-transcribes each stream and dubs both "
            "voices. Separating needs the optional speechbrain dependency."
        ),
    )
    parser.add_argument(
        "--overlap-separator",
        dest="overlap_separator",
        default="auto",
        choices=["auto", "none", "sepformer"],
        help=(
            "Backend for splitting overlapping speech: 'auto' uses SpeechBrain's "
            "SepFormer when installed, 'sepformer' requires it, 'none' disables "
            "it. (default: auto)"
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
        default="words",
        choices=["off", "words"],
        help=(
            "'words' (default) also tells the translator how many words fit each "
            "line, measured from the voice that will speak it (a probe render per "
            "speaker); 'off' sends only the duration and tends to lose the ends of "
            "sentences."
        ),
    )
    parser.add_argument(
        "--budget-speedup",
        dest="budget_speedup",
        type=float,
        default=DEFAULT_BUDGET_SPEEDUP,
        help=(
            "Tempo the word budget spends on every line: the ceiling becomes "
            "words a line may hold at speedup times natural pace. Keep it above "
            "--min-stretch so only the stubborn lines reach the floor."
        ),
    )
    parser.add_argument(
        "--no-subtitles", action="store_true", help="Skip writing SRT files."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached stages.")
    parser.add_argument(
        "--log-file",
        dest="log_file",
        type=Path,
        default=None,
        help=(
            "Also write the run log to this file, appended to across runs. "
            "Defaults to <job folder>/run.log; LOG_FILE sets the same default."
        ),
    )
    parser.add_argument(
        "--no-log-file",
        dest="log_file",
        action="store_false",
        help="Log to stderr only, writing no file.",
    )
    parser.add_argument(
        "--log-max-bytes",
        dest="log_max_bytes",
        type=int,
        default=DEFAULT_MAX_LOG_BYTES,
        help=(
            "Roll the log file aside to a timestamped name once it reaches this "
            "many bytes, so it cannot grow without bound; 0 keeps one file "
            "forever."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{_PROG} {__version__}",
        help="Show the program version and exit.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def build_info_parser() -> argparse.ArgumentParser:
    """Parser for the ``info`` subcommand."""
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} info",
        description=(
            "Report the environment: accelerators, the device and precision the "
            "pipeline would pick, optional packages, system binaries and models."
        ),
    )
    parser.add_argument("--device", default="auto", help="Device to report on.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="dtype to report on (default: auto).",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Override the models directory."
    )
    return parser


def _module_available(name: str) -> bool:
    """True when an optional dependency can be imported."""
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def _torch_version() -> str:
    """torch's version, tagged with the CUDA or ROCm build it was compiled for."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard runtime dep
        return "not installed"

    version = getattr(torch, "version", None)
    hip = getattr(version, "hip", None)
    cuda = getattr(version, "cuda", None)
    if hip:
        return f"{torch.__version__}  (ROCm {hip})"
    if cuda:
        return f"{torch.__version__}  (CUDA {cuda})"
    return f"{torch.__version__}  (CPU build)"


def _dtype_label(device: str, requested: str) -> str:
    """The dtype the TTS checkpoint would load with, as plain text."""
    name = str(pick_dtype(device, requested)).replace("torch.", "")
    return name if requested == "auto" else f"{name}  (requested: {requested})"


def render_environment(
    models_dir: Path | str, *, device: str = "auto", dtype: str = "auto"
) -> str:
    """The text printed by ``dub.py info``.

    Returned rather than printed so it is testable and can be pasted whole into a
    bug report. The first question on unfamiliar hardware is "what did it pick, and
    why?", and every answer belongs here.
    """
    import platform
    import shutil

    info = probe_device(device)
    lines: list[str] = [
        f"python           {platform.python_version()}  (>=3.10 required)",
        f"platform         {platform.system()} {platform.machine()}",
        f"torch            {_torch_version()}",
    ]

    accelerators = available_accelerators()
    lines.append(
        "accelerators     " + (", ".join(accelerators) if accelerators else "none")
    )

    if info.is_accelerated:
        memory = (
            f"{info.total_memory / 1024 ** 3:.1f} GiB"
            if info.total_memory
            else "memory unknown"
        )
        capability = ""
        if info.capability:
            capability = f"  sm_{info.capability[0]}{info.capability[1]}"
        lines.append(f"device           {info.label}  {info.name}  {memory}{capability}")
        lines.append(
            "                 bf16: "
            + ("yes" if info.supports_bf16 else "no")
            + "   fp16: "
            + ("yes" if info.supports_fp16 else "no")
        )
    else:
        lines.append(f"device           cpu  {info.name}")

    lines.append(f"resolved         {info.label}   (requested: {info.requested})")
    lines.append(f"tts dtype        {_dtype_label(device, dtype)}")
    lines.append(
        f"asr device       {asr_device(device)}  "
        f"(compute type {resolve_compute_type(asr_device(device))})"
    )

    lines.append("")
    lines.append("optional packages")
    for name, purpose in _OPTIONAL_DEPENDENCIES:
        mark = "yes" if _module_available(name) else "no "
        lines.append(f"  {name:<14}{mark}  {purpose}")

    lines.append("")
    lines.append("system binaries")
    for name, purpose in _SYSTEM_BINARIES:
        lines.append(f"  {name:<14}{shutil.which(name) or 'MISSING'}  ({purpose})")

    lines.append("")
    lines.append(f"models directory {Path(models_dir).resolve()}")
    for spec, present in registry_status(models_dir):
        required = "  (required)" if spec.name in MVP_REQUIRED else ""
        state = "present" if present else "missing"
        lines.append(f"  {spec.name:<18}{state}{required}")

    lines.append("")
    lines.append("Missing a model?  python dub.py models --download all")
    return "\n".join(lines)


def cmd_info(argv: Sequence[str]) -> int:
    """Handle ``dub.py info``."""
    args = build_info_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir
    print(render_environment(models_dir, device=args.device, dtype=args.dtype))
    return _STOP


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


def cmd_anchors(argv: Sequence[str]) -> int:
    """Handle ``dub.py anchors``."""
    from pipeline.anchors import AnchorSpec, ensure_anchor, list_anchors
    from pipeline.models import anchors_dir
    from pipeline.tts import VoiceDesignEngine

    args = build_anchors_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir
    root = anchors_dir(models_dir)

    if args.list or not args.generate:
        cached = list_anchors(root)
        print(f"Accent anchors: {root}\n")
        if not cached:
            print(
                "  (none yet - generate with: python dub.py anchors --generate)"
            )
            return _STOP
        for key, path, seconds in cached:
            print(f"  {key:<16}{seconds:6.1f}s  {path.name}")
        return _STOP

    requested = args.lang or [
        name for name in sorted(SUPPORTED_LANGUAGES) if name != "Auto"
    ]
    languages = [normalize_language(name) for name in requested]
    if any(language == "Auto" for language in languages):
        print("--lang must be an explicit language, not 'auto'.", file=sys.stderr)
        return 2

    model_dir = resolve_model("tts_voice_design", models_dir)
    engine = VoiceDesignEngine(
        model_dir, device=args.device, dtype=pick_dtype(args.device, args.dtype)
    )
    for language in languages:
        wav, _transcript = ensure_anchor(
            root, AnchorSpec(language), lambda: engine, force=args.force
        )
        print(f"  {language:<12}{wav.name}")
    release_gpu_memory()
    print(f"\nAnchors ready in {root}")
    return _STOP


def job_config_from_args(args: argparse.Namespace) -> JobConfig:
    """Build the :class:`JobConfig` for a parsed set of dub options.

    Extracted from :func:`cmd_dub` so the flag-to-config mapping can be tested
    without running the pipeline. That matters: this function passes *every* flag
    explicitly, including its defaults, so a validator that keys off
    ``model_fields_set`` instead of the value will reject an ordinary run.
    """
    return JobConfig(
        video=args.video,
        audio=args.audio,
        target_language=args.target_language,
        source_language=args.source_language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        tts_batch=args.tts_batch,
        min_stretch=args.min_stretch,
        max_stretch=args.max_stretch,
        loudness_target=args.loudness_target,
        target_lufs=args.target_lufs,
        keep_background=args.keep_background,
        separator=args.separator,
        duck_db=args.duck_db,
        room=args.room,
        room_rt60=args.room_rt60,
        room_wet_db=args.room_wet_db,
        room_duck_db=args.room_duck_db,
        room_tone_gain_db=args.room_tone_gain_db,
        accent_mode=args.accent_mode,
        accent_match=args.accent_match,
        accent_audio=args.accent_audio,
        accent_text=args.accent_text,
        accent_blend=args.accent_blend,
        blend_takes=args.blend_takes,
        blend_iters=args.blend_iters,
        blend_ratio=args.blend_ratio,
        diarizer=args.diarizer,
        speaker_threshold=args.speaker_threshold,
        min_speaker_seconds=args.min_speaker_seconds,
        overlap_mode=args.overlap_mode,
        overlap_separator=args.overlap_separator,
        refine=args.refine,
        refine_max_passes=args.refine_max_passes,
        refine_threshold=args.refine_threshold,
        length_budget=args.length_budget,
        budget_speedup=args.budget_speedup,
        keep_subtitles=not args.no_subtitles,
        force=args.force,
    )


def log_file_from(
    args: argparse.Namespace, settings: Settings, job: JobConfig
) -> Path | None:
    """Where this run's log goes, if anywhere.

    ``--log-file`` wins, then ``LOG_FILE``, then the job's own folder - which is
    where every other artifact of the run already lives, so the default is one
    place to look rather than a flag to remember. ``--no-log-file`` is the only
    way back to stderr alone.
    """
    if args.log_file is False:
        return None
    if isinstance(args.log_file, Path):
        return args.log_file
    if settings.log_file is not None:
        return settings.log_file
    return job.job_dir / _LOG_NAME


def cmd_dub(argv: Sequence[str]) -> int:
    """Handle the default dubbing command."""
    args = build_dub_parser().parse_args(list(argv))
    settings = Settings.from_env()
    # Built before logging starts because the job knows which folder the log
    # belongs in. Nothing in here logs, so the reordering loses no record.
    job = job_config_from_args(args)
    setup_logging(
        "DEBUG" if args.verbose else settings.log_level,
        settings.log_json,
        log_file=log_file_from(args, settings, job),
        max_bytes=args.log_max_bytes,
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
    elif args and args[0] == "anchors":
        handler = cmd_anchors
        payload = args[1:]
    elif args and args[0] == "info":
        handler = cmd_info
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
