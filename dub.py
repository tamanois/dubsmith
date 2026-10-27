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
from typing import Any, Sequence

from pipeline import __version__
from pipeline.asr import asr_device, resolve_compute_type
from pipeline.config import (
    DEFAULT_BUDGET_SPEEDUP,
    DEFAULT_MAX_OVERRUN,
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
from pipeline.errors import ConfigError, DubbingError, ModelNotFoundError, TTSError
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


def build_voices_parser() -> argparse.ArgumentParser:
    """Parser for the ``voices`` subcommand (the saved voice library)."""
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} voices",
        description=(
            "Create, list and remove saved voice prints. A voice print is a short "
            "reference clip plus its transcript, so the same voice can be reused by "
            "name in 'python dub.py say' or in a dubbing run with --voice. Without a "
            "mode, the library is listed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--design",
        action="store_true",
        help="Create a voice from a text description (VoiceDesign checkpoint).",
    )
    mode.add_argument(
        "--clone",
        action="store_true",
        help="Create a voice cloned from --audio and --clone-text (Base checkpoint).",
    )
    mode.add_argument(
        "--list", action="store_true", help="List saved voices (the default)."
    )
    mode.add_argument(
        "--remove", metavar="LABEL", help="Delete a saved voice by label."
    )

    parser.add_argument(
        "--prompt",
        "--instruct",
        dest="prompt",
        metavar="TEXT",
        default=None,
        help=(
            "Description of the voice to create, e.g. 'a warm female French voice, "
            "slow and clear'. Required with --design."
        ),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        metavar="FILE",
        default=None,
        help="Reference clip containing the voice to clone. Required with --clone.",
    )
    parser.add_argument(
        "--clone-text",
        dest="clone_text",
        metavar="TEXT",
        default=None,
        help="Exact transcript of --audio. Required with --clone.",
    )
    parser.add_argument(
        "--lang",
        default="fr",
        metavar="LANG",
        help=(
            "Language the voice print is recorded in. It is also the accent the "
            "design model aims for, so pick the language you will speak most."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        metavar="NAME",
        help=(
            "Name to save the voice under. Defaults to a slug of the description "
            "(design) or the audio filename (clone)."
        ),
    )
    parser.add_argument(
        "--speak",
        default=None,
        metavar="TEXT",
        help="Optionally render this text with the new voice and write it to --output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where --speak writes its WAV (default: output/say/<label>.wav).",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Precision for the checkpoint (default: auto).",
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=None,
        help="Override the voice-library directory (default: VOICES_DIR).",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Override the models directory."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing voice with the same label.",
    )
    return parser


def build_say_parser() -> argparse.ArgumentParser:
    """Parser for the ``say`` subcommand (standalone speech synthesis)."""
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} say",
        description=(
            "Speak a text with a saved voice, or with a voice described or cloned "
            "inline. This is the fast path for making an audio clip: no video, no "
            "translation, no pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--voice",
        metavar="LABEL",
        help="Saved voice to use (see 'python dub.py voices --list').",
    )
    source.add_argument(
        "--design",
        metavar="PROMPT",
        help=(
            "Describe a voice inline, e.g. 'a warm female French voice'. Not saved "
            "unless --save-as is given."
        ),
    )
    source.add_argument(
        "--clone",
        type=Path,
        metavar="AUDIO",
        help="Clone a voice inline from this clip, which --clone-text transcribes.",
    )
    parser.add_argument(
        "--clone-text",
        dest="clone_text",
        metavar="TEXT",
        default=None,
        help="Exact transcript of --clone. Required with it.",
    )
    parser.add_argument(
        "--text", required=True, metavar="TEXT", help="Text to speak."
    )
    parser.add_argument(
        "--lang",
        default=None,
        metavar="LANG",
        help=(
            "Target language. Defaults to the saved voice's own language; required "
            "for an inline --design or --clone."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV (default: output/say/<name>.wav).",
    )
    parser.add_argument(
        "--save-as",
        dest="save_as",
        metavar="LABEL",
        default=None,
        help="Save an inline --design or --clone as a reusable voice under this label.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Precision for the checkpoint (default: auto).",
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=None,
        help="Override the voice-library directory (default: VOICES_DIR).",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Override the models directory."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing voice when --save-as names one.",
    )
    return parser


def _device_list(value: str) -> list[str]:
    """Parse the comma-separated device list ``--devices`` takes.

    Cleaning and validation live in :class:`JobConfig`, so there is one rule for
    what a device list may contain rather than two that can disagree.
    """
    return [part.strip() for part in value.split(",")]


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
    parser.add_argument(
        "--voice",
        default=None,
        metavar="LABEL",
        help=(
            "Reuse a saved voice print (see 'python dub.py voices --list') as the "
            "voice for every speaker. It supplies the timbre; pronunciation still "
            "comes from --accent-mode (native by default), so the accent matches "
            "the target language. Cannot be combined with --ref-audio."
        ),
    )

    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output root.")
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=None,
        help="Override the saved-voice directory used by --voice (default: VOICES_DIR).",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cuda:1, mps, or cpu.")
    parser.add_argument(
        "--devices",
        type=_device_list,
        default=None,
        metavar="DEV[,DEV...]",
        help=(
            "Spread synthesis over several devices, one worker process per device, "
            "e.g. cuda:0,cuda:1. Stages still run one at a time; the heavy blend "
            "and synthesis stages hand each piece to whichever worker is free, and "
            "the first device carries the stages that cannot be split (ASR, "
            "diarization, separation, accent anchors). Repeating a name "
            "(cpu,cpu) is allowed and is how the fan-out is exercised on a "
            "single-processor machine. Omit it to keep --device as the whole "
            "answer."
        ),
    )
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
        "--max-overrun",
        dest="max_overrun",
        type=float,
        default=DEFAULT_MAX_OVERRUN,
        help=(
            "Seconds a line may run past its window - into the next line's time - "
            "before the excess is cut. A fraction of a second of lateness is "
            "cheaper than a deleted word; 0 restores exact sync and cuts instead."
        ),
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
        default=None,
        choices=["native", "preserve", "speaker-only", "reference-accent"],
        help=(
            "'native' (the default) keeps the speaker's timbre and takes "
            "pronunciation from a generated native accent anchor; 'preserve' "
            "clones the speaker's accent along with their voice; 'speaker-only' "
            "keeps only the timbre embedding so the model uses its own phonetics "
            "for the target language; 'reference-accent' keeps the speaker's "
            "timbre but takes pronunciation from --accent-audio. With --voice, a "
            "saved voice already in the target language clones directly unless "
            "this flag says otherwise."
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
        "--devices",
        type=_device_list,
        default=None,
        metavar="DEV[,DEV...]",
        help="Report the fan-out these devices would produce, e.g. cuda:0,cuda:1.",
    )
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
    models_dir: Path | str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    devices: Sequence[str] | None = None,
) -> str:
    """The text printed by ``dub.py info``.

    Returned rather than printed so it is testable and can be pasted whole into a
    bug report. The first question on unfamiliar hardware is "what did it pick, and
    why?", and every answer belongs here.

    ``devices`` is the ``--devices`` fan-out when one was asked for. The stages
    that cannot be spread over cards still run on a single device, so the
    capability lines below describe that one - the first of the list.
    """
    import platform
    import shutil

    requested = list(devices) if devices else []
    device = requested[0] if requested else device
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
    if len(requested) > 1:
        lines.append(
            f"synthesis        {len(requested)} workers, one per device: "
            + ", ".join(requested)
        )
        lines.append(
            "                 (blend and synthesis are spread over them; ASR, "
            "diarization and separation stay on the first)"
        )
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
    print(
        render_environment(
            models_dir, device=args.device, dtype=args.dtype, devices=args.devices
        )
    )
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


def _voices_root(args: argparse.Namespace, settings: Settings) -> Path:
    """The voice library this invocation uses: ``--voices-dir``, else settings."""
    return Path(args.voices_dir) if args.voices_dir is not None else settings.voices_dir


def _write_say_output(
    path: Path | None, name: str, samples: Any, sample_rate: int
) -> Path:
    """Write a ``say``/``--speak`` clip, defaulting to ``output/say/<name>.wav``."""
    from pipeline.media import write_audio
    from pipeline.voices import slugify_label

    target = (
        path
        if path is not None
        else Path("output") / "say" / f"{slugify_label(name)}.wav"
    )
    return write_audio(target, samples, int(sample_rate))


def _anchor_text(language: str) -> str:
    """The canonical audition passage for a language, or an actionable error.

    Reusing the accent-anchor passage keeps every voice print comparable: the
    same words, the same phonetic coverage, roughly the same length.
    """
    from pipeline.anchors import ANCHOR_TEXT

    try:
        return ANCHOR_TEXT[language]
    except KeyError:
        raise ConfigError(
            f"No built-in passage for {language!r}; add one to "
            "pipeline/anchors.ANCHOR_TEXT or pick a supported language."
        ) from None


def cmd_voices(argv: Sequence[str]) -> int:
    """Handle ``dub.py voices``: create, list and remove saved voice prints."""
    from pipeline.tts import TTSEngine, VoiceDesignEngine
    from pipeline.voices import (
        default_label,
        delete_voice,
        list_voices,
        save_voice,
        unique_label,
    )

    args = build_voices_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir
    root = _voices_root(args, settings)

    if args.remove:
        removed = delete_voice(root, args.remove)
        print(f"Removed {args.remove!r} from {removed}")
        return _STOP

    if args.list or not (args.design or args.clone):
        saved = list_voices(root)
        print(f"Saved voices: {root.resolve()}\n")
        if not saved:
            print(
                "  (none yet - create one with "
                '\'python dub.py voices --design --prompt "..." --lang fr\')'
            )
            return _STOP
        for item in saved:
            print(f"  {item.summary()}")
        return _STOP

    language = normalize_language(args.lang)
    if language == "Auto":
        raise ConfigError("--lang must be an explicit language, not 'auto'.")
    anchor_text = _anchor_text(language)

    if args.design:
        if not args.prompt or not args.prompt.strip():
            raise ConfigError("--design requires --prompt (a description of the voice).")
        engine = VoiceDesignEngine(
            resolve_model("tts_voice_design", models_dir),
            device=args.device,
            dtype=pick_dtype(args.device, args.dtype),
        )
        samples, sample_rate = engine.generate(anchor_text, args.prompt, language)
        label = args.label or unique_label(default_label(instruct=args.prompt), root)
        saved = save_voice(
            root,
            label,
            samples,
            sample_rate,
            language=language,
            source="design",
            ref_text=anchor_text,
            instruct=args.prompt,
            force=args.force,
        )
        if args.speak:
            speech, speech_rate = engine.generate(args.speak, args.prompt, language)
            out = _write_say_output(args.output, saved.label, speech, speech_rate)
            print(f"Wrote {out}")
        release_gpu_memory()
    else:
        if args.audio is None:
            raise ConfigError("--clone requires --audio (a clip of the voice to clone).")
        if not args.clone_text or not args.clone_text.strip():
            raise ConfigError(
                "--clone requires --clone-text (the exact transcript of --audio)."
            )
        engine = TTSEngine(
            resolve_model("tts_base", models_dir),
            device=args.device,
            dtype=pick_dtype(args.device, args.dtype),
        )
        engine.prepare_prompt_from_file(args.audio, args.clone_text)
        texts = [anchor_text] + ([args.speak] if args.speak else [])
        rendered = engine.synthesize(texts, language=language)
        anchor = rendered[0] if rendered else None
        if anchor is None:
            raise TTSError("The cloning model returned no audio for the voice print.")
        label = args.label or unique_label(default_label(audio=args.audio), root)
        saved = save_voice(
            root,
            label,
            anchor,
            engine.sample_rate,
            language=language,
            source="clone",
            ref_text=anchor_text,
            force=args.force,
        )
        if args.speak and len(rendered) > 1 and rendered[1] is not None:
            out = _write_say_output(
                args.output, saved.label, rendered[1], engine.sample_rate
            )
            print(f"Wrote {out}")
        release_gpu_memory()

    print(
        f"Saved voice {saved.label!r} ({saved.language}, {saved.seconds:.1f}s) "
        f"to {saved.directory}"
    )
    print(f'Use it with: python dub.py say --voice {saved.label} --text "..."')
    return _STOP


def cmd_say(argv: Sequence[str]) -> int:
    """Handle ``dub.py say``: speak a text with a saved, designed or cloned voice."""
    from pipeline.tts import TTSEngine, VoiceDesignEngine
    from pipeline.voices import load_voice, save_voice

    args = build_say_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir
    root = _voices_root(args, settings)

    if not args.text or not args.text.strip():
        raise ConfigError("--text cannot be empty.")

    if args.voice:
        saved = load_voice(root, args.voice)
        language = normalize_language(args.lang) if args.lang else saved.language
        engine = TTSEngine(
            resolve_model("tts_base", models_dir),
            device=args.device,
            dtype=pick_dtype(args.device, args.dtype),
        )
        engine.prepare_prompt_from_file(saved.wav, saved.ref_text)
        rendered = engine.synthesize([args.text], language=language)
        speech = rendered[0] if rendered else None
        if speech is None:
            raise TTSError("The model returned no audio.")
        out = _write_say_output(args.output, saved.label, speech, engine.sample_rate)
        release_gpu_memory()
        print(f"Wrote {out}")
        return _STOP

    if args.lang is None:
        raise ConfigError(
            "--lang is required when the voice is described or cloned inline."
        )
    language = normalize_language(args.lang)
    if language == "Auto":
        raise ConfigError("--lang must be an explicit language, not 'auto'.")
    anchor_text = _anchor_text(language)

    if args.design:
        engine = VoiceDesignEngine(
            resolve_model("tts_voice_design", models_dir),
            device=args.device,
            dtype=pick_dtype(args.device, args.dtype),
        )
        speech, sample_rate = engine.generate(args.text, args.design, language)
        if args.save_as:
            samples, anchor_rate = engine.generate(anchor_text, args.design, language)
            save_voice(
                root,
                args.save_as,
                samples,
                anchor_rate,
                language=language,
                source="design",
                ref_text=anchor_text,
                instruct=args.design,
                force=args.force,
            )
        out = _write_say_output(
            args.output, args.save_as or "design", speech, sample_rate
        )
        release_gpu_memory()
        print(f"Wrote {out}")
        return _STOP

    if not args.clone_text or not args.clone_text.strip():
        raise ConfigError(
            "--clone requires --clone-text (the exact transcript of the clip)."
        )
    engine = TTSEngine(
        resolve_model("tts_base", models_dir),
        device=args.device,
        dtype=pick_dtype(args.device, args.dtype),
    )
    engine.prepare_prompt_from_file(args.clone, args.clone_text)
    texts = [args.text] + ([anchor_text] if args.save_as else [])
    rendered = engine.synthesize(texts, language=language)
    speech = rendered[0] if rendered else None
    if speech is None:
        raise TTSError("The model returned no audio.")
    if args.save_as:
        anchor = rendered[1] if len(rendered) > 1 else None
        if anchor is None:
            raise TTSError("The model returned no audio for the saved voice print.")
        save_voice(
            root,
            args.save_as,
            anchor,
            engine.sample_rate,
            language=language,
            source="clone",
            ref_text=anchor_text,
            force=args.force,
        )
    out = _write_say_output(
        args.output, args.save_as or "clone", speech, engine.sample_rate
    )
    release_gpu_memory()
    print(f"Wrote {out}")
    return _STOP


def job_config_from_args(
    args: argparse.Namespace, *, settings: Settings | None = None
) -> JobConfig:
    """Build the :class:`JobConfig` for a parsed set of dub options.

    Extracted from :func:`cmd_dub` so the flag-to-config mapping can be tested
    without running the pipeline. That matters: this function passes *every* flag
    explicitly, including its defaults, so a validator that keys off
    ``model_fields_set`` instead of the value will reject an ordinary run.

    ``--voice`` is resolved here into ``ref_audio``/``ref_text``, which is the
    representation the pipeline already understands. Doing it at this boundary
    keeps every downstream stage (fingerprinting, donor selection, prompt
    building, fan-out workers) unchanged: a saved voice is simply a reference
    clip whose transcript is known.
    """
    ref_audio = args.ref_audio
    ref_text = args.ref_text
    # ``None`` means "not asked for", which lets a saved voice pick the mode that
    # fits it without overriding an explicit choice. The pipeline's own default
    # is still ``native``.
    accent_mode = args.accent_mode or "native"
    accent_requested = args.accent_mode is not None
    voice_label = getattr(args, "voice", None)
    if voice_label:
        if ref_audio is not None:
            raise ConfigError(
                "--voice cannot be combined with --ref-audio: a saved voice already "
                "is a reference clip. Pick one."
            )
        from pipeline.voices import load_voice

        voices_root = getattr(args, "voices_dir", None)
        if voices_root is None:
            voices_root = (
                settings.voices_dir
                if settings is not None
                else Settings.from_env().voices_dir
            )
        saved = load_voice(voices_root, voice_label)
        ref_audio = saved.wav
        ref_text = saved.ref_text
        # The print's own language is already the target language, so its speech
        # codes carry the right accent: cloning it directly is both the most
        # faithful and the cheapest path (no blend, no donor anchor). A print in
        # another language keeps the default ``native`` mode, which borrows the
        # target accent from a generated anchor.
        if not accent_requested and saved.language == normalize_language(
            args.target_language
        ):
            accent_mode = "preserve"

    return JobConfig(
        video=args.video,
        audio=args.audio,
        target_language=args.target_language,
        source_language=args.source_language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        output_dir=args.output_dir,
        device=args.device,
        devices=args.devices,
        dtype=args.dtype,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        tts_batch=args.tts_batch,
        min_stretch=args.min_stretch,
        max_stretch=args.max_stretch,
        max_overrun=args.max_overrun,
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
        accent_mode=accent_mode,
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
    job = job_config_from_args(args, settings=settings)
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
    elif args and args[0] in {"voices", "voice-library"}:
        handler = cmd_voices
        payload = args[1:]
    elif args and args[0] == "say":
        handler = cmd_say
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
