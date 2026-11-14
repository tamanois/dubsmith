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
import logging
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pipeline import __version__, options
from pipeline.asr import asr_device, resolve_compute_type
from pipeline.config import (
    DEFAULT_BUDGET_SPEEDUP,
    DEFAULT_MAX_OVERRUN,
    DEFAULT_MIN_SPEAKER_S,
    SUPPORTED_LANGUAGES,
    DeviceInfo,
    JobConfig,
    Settings,
    SubtitleJobConfig,
    available_accelerators,
    enumerate_devices,
    language_code_for_name,
    normalize_device_request,
    normalize_language,
    pick_dtype,
    preferred_dtype,
    probe_device,
    release_gpu_memory,
)
from pipeline.errors import (
    ConfigError,
    DubbingError,
    ModelNotFoundError,
    TTSError,
)
from pipeline.logging_utils import (
    DEFAULT_MAX_LOG_BYTES,
    job_logger,
    setup_logging,
    timed,
)
from pipeline.mixing import (
    DEFAULT_ROOM_DUCK_DB,
    DEFAULT_ROOM_TONE_GAIN_DB,
    DEFAULT_ROOM_WET_DB,
)
from pipeline.models import (
    MVP_REQUIRED,
    REGISTRY,
    download_many,
    model_fingerprint,
    registry_status,
    resolve_model,
)
from pipeline.orchestrate import Pipeline
from pipeline.subtitle_pipeline import SubtitlesPipeline

_PROG = "dub.py"
_STOP = 0

#: Warnings raised while building the voice library. ``voices`` does not set up
#: the job logger (there is no job), so this relies on Python's last-resort
#: handler to reach stderr, and joins the run log whenever one exists.
_LOGGER = logging.getLogger("dub.voices")

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
        metavar="NAME|all|everything",
        help=(
            "Download one model by name, 'all' for the MVP-required set, or "
            "'everything' to include optional backends."
        ),
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
    parser.add_argument(
        "--device",
        type=_device_arg,
        default="auto",
        help="auto, cpu, cuda:N, rocm[:N] (alias for cuda), xpu, or mps.",
    )
    parser.add_argument(
        "--dtype",
        default=options.Defaults.DTYPE,
        choices=options.DTYPES,
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
            "Create, list and remove saved voices. A voice is a reference clip plus "
            "its transcript, and optionally the checkpoint's encoding of it, so the "
            "same voice can be reused by name in 'python dub.py say' or in a "
            "dubbing run with --voice. Three ways to create one: --design (from a "
            "description), --clone (encode a donated clip once) and --add (register "
            "a clip verbatim, no model loaded). Without a mode, the library is "
            "listed."
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
        help=(
            "Create a voice by encoding --audio and --clone-text once (Base "
            "checkpoint). The encoding is stored, so 'say' never encodes the "
            "clip again."
        ),
    )
    mode.add_argument(
        "--add",
        metavar="AUDIO",
        type=Path,
        help=(
            "Register an existing clip as a voice without processing it. The clip "
            "is copied and --clone-text is stored as given; no model is loaded. "
            "Add --transcribe to derive the transcript (and the language)."
        ),
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
        help=(
            "Exact transcript of the reference clip. Required with --clone unless "
            "--transcribe is given; optional with --add."
        ),
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help=(
            "Derive the transcript with the local ASR instead of typing it, and "
            "use the detected language when --lang is omitted. Costs one Whisper "
            "pass over the reference clip."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Transcribe the reference and warn when --clone-text does not match "
            "what was heard. ICL cloning aligns the codes against the transcript, "
            "so a mismatch changes the voice rather than failing."
        ),
    )
    parser.add_argument(
        "--lang",
        default=None,
        metavar="LANG",
        help=(
            "Language the voice is recorded in; it is also the accent the design "
            "model aims for. Defaults to the language detected by --transcribe, "
            "else to fr for --design/--clone; required for --add otherwise."
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
        help=(
            "Optionally render this text with the new voice and write it to "
            "--output. Available for --design and --clone (which load a model), "
            "not for --add."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where --speak writes its WAV (default: output/say/<label>.wav).",
    )
    parser.add_argument(
        "--device",
        type=_device_arg,
        default="auto",
        help="auto, cpu, cuda:N, rocm[:N] (alias for cuda), xpu, or mps.",
    )
    parser.add_argument(
        "--dtype",
        default=options.Defaults.DTYPE,
        choices=options.DTYPES,
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
    source = parser.add_mutually_exclusive_group(required=False)
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
        "--speaker",
        metavar="NAME",
        default=None,
        help=(
            "Preset speaker for --backend customvoice (e.g. 'Vivian', 'Ryan'). "
            "Required with that backend; rejected by the others."
        ),
    )
    parser.add_argument(
        "--clone-text",
        dest="clone_text",
        metavar="TEXT",
        default=None,
        help=(
            "Exact transcript of --clone. Optional: without it 'qwen' clones from "
            "the speaker embedding alone (lower similarity), and the CosyVoice "
            "backends lose the reference's accent."
        ),
    )
    parser.add_argument(
        "--backend",
        default=options.Defaults.SAY_BACKEND,
        choices=options.SAY_BACKENDS,
        help=(
            "Synthesiser: 'qwen' (default) clones with the Qwen Base checkpoint; "
            "'customvoice' uses a preset Qwen CustomVoice speaker with a tone/"
            "emotion instruction (needs --speaker, no --voice/--clone/--design); "
            "'cosyvoice3' uses CosyVoice3, which adds instruct control (emotion, "
            "speed, volume) plus paralinguistic tokens such as [breath] and "
            "[laughter]; 'chatterbox' uses Chatterbox Multilingual, whose style "
            "control is an emotion exaggeration; 'fish' uses Fish Speech S2-Pro, "
            "which reads inline [sad]/[sigh]/[whisper] tags written in --text."
        ),
    )
    parser.add_argument(
        "--emotion",
        "--instruct",
        dest="emotions",
        action="append",
        default=None,
        metavar="TEXT",
        help=(
            "Per-line emotion/style, e.g. 'calm' or 'excited'. Repeat it or pass a "
            "comma-separated list; a single value applies to every line. "
            "CosyVoice3, Chatterbox and CustomVoice map it onto their own control "
            "(prefer --prosody). Rejected by 'qwen' and 'fish' (Fish takes its "
            "direction from tags in --text). Alias: --instruct, not "
            "--style-instruct."
        ),
    )
    parser.add_argument(
        "--style-instruct",
        "--raw-instruct",
        dest="style_instructs",
        action="append",
        default=None,
        metavar="TEXT",
        help=(
            "Verbatim delivery instruction for --backend customvoice, e.g. 'Speak "
            "with restrained anger, then fade to exhaustion.' It is sent to the "
            "checkpoint unchanged, which is the natural-language form it expects. "
            "Repeat it once per line, or pass one value to apply everywhere; it "
            "wins over --prosody/--emotion/--prosody-plan. (--instruct is an alias "
            "of --emotion, not of this flag.)"
        ),
    )
    parser.add_argument(
        "--prosody",
        dest="prosody",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Per-line prosody, as comma-separated 'key=value' pairs or bare flags: "
            "emotion=happy, pace=fast, volume=loud, breath, pause, laughter, "
            "exaggeration=0.8. Repeat it once per line, or pass one value to apply "
            "everywhere. 'exaggeration' is Chatterbox's own 0.0-1.0 style knob; "
            "this is the structured form of --emotion."
        ),
    )
    parser.add_argument(
        "--prosody-plan",
        dest="prosody_plan",
        default=options.Defaults.SAY_PROSODY_PLAN,
        choices=options.SAY_PROSODY_PLANS,
        help=(
            "Let the configured OpenAI-compatible model decide per-line prosody "
            "from the text. 'off' uses only --prosody/--emotion; 'llm' plans it "
            "first, with any explicit --prosody value overriding the plan."
        ),
    )
    parser.add_argument(
        "--style-plan",
        dest="style_plan",
        default=options.Defaults.STYLE_PLAN,
        choices=options.STYLE_PLANS,
        help=(
            "CustomVoice only: let the configured OpenAI-compatible model write "
            "the rich verbatim delivery instruction that --style-instruct takes "
            "by hand. 'llm' plans it first; an explicit --style-instruct still "
            "wins. Use --style-plan-scope to choose one per line or one for all."
        ),
    )
    parser.add_argument(
        "--style-plan-scope",
        dest="style_plan_scope",
        default=options.Defaults.STYLE_PLAN_SCOPE,
        choices=options.STYLE_PLAN_SCOPES,
        help=(
            "With --style-plan llm: 'per-line' writes one instruction per line "
            "(chunked, with a whole-passage overview when long); 'whole' writes "
            "a single instruction for the entire text."
        ),
    )
    parser.add_argument(
        "--plan-chunk",
        dest="plan_chunk",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Lines per LLM planning request before --style-plan llm, "
            "--prosody-plan llm and --fish-tag-mode llm switch to the overview + "
            "chunked strategy (default: LLM_PLAN_CHUNK_SIZE, 10)."
        ),
    )
    parser.add_argument(
        "--chatterbox-exaggeration",
        type=float,
        default=None,
        metavar="0..1",
        help=(
            "Chatterbox: fixed expression intensity for every line, overriding "
            "the emotion mapping (default: derive from emotion)."
        ),
    )
    parser.add_argument(
        "--chatterbox-cfg-weight",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Chatterbox: classifier-free guidance weight (package default 0.5).",
    )
    parser.add_argument(
        "--chatterbox-temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Chatterbox: sampling temperature (package default 0.8).",
    )
    parser.add_argument(
        "--fish-temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Fish Speech: sampling temperature (model default 0.7).",
    )
    parser.add_argument(
        "--fish-top-p",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Fish Speech: nucleus sampling probability (model default 0.7).",
    )
    parser.add_argument(
        "--fish-tag-mode",
        dest="fish_tag_mode",
        default=options.Defaults.FISH_TAG_MODE,
        choices=options.FISH_TAG_MODES,
        help=(
            "Fish Speech: tags are the checkpoint's only style channel. 'keep' "
            "(default) speaks --text exactly as written, tags included; 'llm' has "
            "the configured model insert the tags, guarded so the words cannot "
            "change."
        ),
    )
    parser.add_argument(
        "--fish-quantize",
        dest="fish_quantize",
        default=None,
        choices=options.FISH_QUANTIZES,
        help=(
            "Fish Speech weight format. 'int8' (default) loads 8-bit weight-only "
            "(~6 GB resident, ~11 GB peak) so the model fits a 12 GB card; "
            "'off' loads the bf16 weights Fish documents as needing ~24 GB. "
            "CPU runs always use bf16."
        ),
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
        help=(
            "Save an inline --design or --clone as a reusable voice under this "
            "label. A saved clone stores the clip and its encoding, so using the "
            "voice later costs no encode."
        ),
    )
    parser.add_argument(
        "--device",
        type=_device_arg,
        default="auto",
        help="auto, cpu, cuda:N, rocm[:N] (alias for cuda), xpu, or mps.",
    )
    parser.add_argument(
        "--dtype",
        default=options.Defaults.DTYPE,
        choices=options.DTYPES,
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    parser.add_argument(
        "--log-file",
        dest="log_file",
        type=Path,
        default=None,
        help=(
            "Also write the say log to this file, appended to across runs. "
            "Defaults to output/say/<name>/say.log for the expressive backends "
            "and output/say/<name>.log for Qwen; LOG_FILE sets the same default."
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
        "--no-provenance",
        dest="no_provenance",
        action="store_true",
        help=(
            "Do not write the timestamped provenance artifact (per-line prosody, "
            "the LLM reply and the text transformations sent to the backend)."
        ),
    )
    return parser


def _device_arg(value: str) -> str:
    """Canonicalise ``--device`` at parse time, so a bad name fails immediately.

    ``argparse.ArgumentTypeError`` keeps the message we wrote instead of the
    generic "invalid value" text, which is the difference between a user learning
    what to type and a user guessing.
    """
    try:
        return normalize_device_request(value)
    except ConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _device_list(value: str) -> list[str]:
    """Parse the comma-separated device list ``--devices`` takes.

    Each entry is canonicalised here as well, so ``--devices rocm:0,rocm:1``
    reaches :class:`JobConfig` as ``cuda:0,cuda:1``. Blank entries are dropped;
    ``JobConfig`` still rejects a list that ends up empty.
    """
    devices: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            devices.append(normalize_device_request(part))
        except ConfigError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None
    return devices


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
    parser.add_argument(
        "--device",
        type=_device_arg,
        default="auto",
        help=(
            "auto, cpu, cuda:N, rocm[:N] (alias for cuda), xpu, or mps. Run "
            "'python dub.py devices' to list every card and the exact flag to use."
        ),
    )
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
        default=options.Defaults.DTYPE,
        choices=options.DTYPES,
        help=(
            "Precision for the voice-cloning checkpoint. 'auto' reads the "
            "device's capability: bf16 on Ampere and newer, fp16 on older "
            "accelerators, fp32 on CPU. An explicit value is honoured unless "
            "the device cannot run it."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        default=options.Defaults.WHISPER_MODEL,
        help="Whisper model id.",
    )
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
        "--backend",
        dest="tts_backend",
        default=options.Defaults.TTS_BACKEND,
        choices=options.TTS_BACKENDS,
        help=(
            "Synthesiser for the dubbed lines. 'qwen' (default) clones with the "
            "Qwen Base checkpoint and has no prosody control; 'cosyvoice3', "
            "'chatterbox' and 'fish' additionally render per-line emotion (see "
            "--prosody-plan). All four clone each speaker from a clip of their "
            "own voice."
        ),
    )
    parser.add_argument(
        "--prosody-plan",
        dest="prosody_plan",
        default=options.Defaults.PROSODY_PLAN,
        choices=options.PROSODY_PLANS,
        help=(
            "Let the configured OpenAI-compatible model decide per-line delivery "
            "(emotion, pace, volume, breath, pause, laughter) for the expressive "
            "backends. 'auto' plans when the LLM is reachable and falls back to "
            "neutral otherwise; 'llm' fails if it is unreachable; 'off' speaks "
            "neutral. Ignored by --backend qwen, which has no prosody control."
        ),
    )
    parser.add_argument(
        "--chatterbox-exaggeration",
        type=float,
        default=None,
        metavar="0..1",
        help=(
            "Chatterbox: fixed expression intensity for every line, overriding "
            "the value derived from the planned emotion."
        ),
    )
    parser.add_argument(
        "--chatterbox-cfg-weight",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Chatterbox: classifier-free guidance weight (package default 0.5).",
    )
    parser.add_argument(
        "--chatterbox-temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Chatterbox: sampling temperature (package default 0.8).",
    )
    parser.add_argument(
        "--fish-temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Fish Speech: sampling temperature (model default 0.7).",
    )
    parser.add_argument(
        "--fish-top-p",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Fish Speech: nucleus sampling probability (model default 0.7).",
    )
    parser.add_argument(
        "--fish-quantize",
        dest="fish_quantize",
        default=None,
        choices=options.FISH_QUANTIZES,
        help=(
            "Fish Speech weight format. 'int8' loads 8-bit weight-only so the "
            "model fits a 12 GB card; 'off' loads the bf16 weights."
        ),
    )
    parser.add_argument(
        "--stretcher",
        default=options.Defaults.STRETCHER,
        choices=options.STRETCHERS,
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
        choices=options.LOUDNESS_POLICIES,
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
        default=options.Defaults.SEPARATOR,
        choices=options.SEPARATORS,
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
        default=options.Defaults.ROOM,
        choices=options.ROOMS,
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
        choices=options.ACCENT_MODES,
        help=(
            "'native' (the default) keeps the speaker's timbre and takes "
            "pronunciation from a generated native accent anchor; 'preserve' "
            "clones the speaker's accent along with their voice; 'speaker-only' "
            "keeps only the timbre embedding so the model uses its own phonetics "
            "for the target language; 'reference-accent' keeps the speaker's "
            "timbre but takes pronunciation from --accent-audio or --accent-voice. "
            "With --voice, a saved voice already in the target language clones "
            "directly unless this flag says otherwise."
        ),
    )
    parser.add_argument(
        "--accent-match",
        dest="accent_match",
        default=options.Defaults.ACCENT_MATCH,
        choices=options.ACCENT_MATCHES,
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
            "should adopt. Requires --accent-text and --accent-mode reference-accent. "
            "Use --accent-voice instead to take both from a saved voice."
        ),
    )
    parser.add_argument(
        "--accent-text",
        default=None,
        help="Exact transcript of --accent-audio (required with it).",
    )
    parser.add_argument(
        "--accent-voice",
        default=None,
        metavar="LABEL",
        help=(
            "Saved voice (see 'python dub.py voices --list') that supplies the "
            "reference accent: its clip and its exact transcript, so neither "
            "--accent-audio nor --accent-text is needed. Implies --accent-mode "
            "reference-accent; cannot be combined with --accent-audio or "
            "--accent-text."
        ),
    )
    parser.add_argument(
        "--accent-blend",
        dest="accent_blend",
        default=None,
        choices=options.ACCENT_BLENDS,
        help=(
            "'two-stage' (default) renders the accent transcript once in the "
            "speaker's own voice and clones every line from that clip, so the "
            "speech codes and the speaker embedding always agree; 'single-stage' "
            "conditions each line on both sources at once."
        ),
    )
    parser.add_argument(
        "--expressive-accent",
        dest="expressive_accent",
        default=options.Defaults.EXPRESSIVE_ACCENT,
        choices=options.EXPRESSIVE_ACCENTS,
        help=(
            "How an expressive backend (cosyvoice3, chatterbox, fish) gets its "
            "clone reference: 'donor' (default) clones the source-language "
            "donor, keeping the source accent; 'blend' builds the target-language "
            "blended reference the Qwen path uses and clones that, keeping the "
            "speaker's timbre with a native target accent. 'blend' loads Qwen "
            "Base for the one-off reference and requires --accent-mode native "
            "or reference-accent."
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
        default=options.Defaults.DIARIZER,
        choices=options.DIARIZERS,
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
        default=options.Defaults.OVERLAP_MODE,
        choices=options.OVERLAP_MODES,
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
        default=options.Defaults.OVERLAP_SEPARATOR,
        choices=options.OVERLAP_SEPARATORS,
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
        default=options.Defaults.LENGTH_BUDGET,
        choices=options.LENGTH_BUDGETS,
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
        "--subtitles-mode",
        dest="subtitles_mode",
        default=options.Defaults.SUBTITLES_MODE,
        choices=options.SUBTITLES_MODES,
        help=(
            "Embed the target subtitles in the video: 'soft' (default) muxes a "
            "toggleable track and copies every other stream; 'burn' draws them "
            "into the frames (re-encodes the video); 'off' writes the SRT files "
            "only."
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
    parser.add_argument(
        "--no-provenance",
        dest="no_provenance",
        action="store_true",
        help=(
            "Do not write the timestamped provenance artifact (per-line prosody, "
            "the LLM reply and the text transformations sent to the backend)."
        ),
    )
    return parser


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def build_subtitles_parser() -> argparse.ArgumentParser:
    """Parser for the ``subtitles`` subcommand (independent of dubbing)."""
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} subtitles",
        description=(
            "Add target-language subtitles to a video (or transcribe an audio "
            "file) without dubbing it. The subtitles are embedded softly by "
            "default; other subtitle languages already in the video are kept."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Input video file.")
    source.add_argument("--audio", type=Path, help="Input audio file (SRT only).")

    parser.add_argument(
        "--target",
        dest="target_language",
        default="fr",
        help=f"Subtitle language ({', '.join(sorted(SUPPORTED_LANGUAGES))}).",
    )
    parser.add_argument(
        "--source",
        dest="source_language",
        default=None,
        help="Source language; omit to auto-detect.",
    )
    parser.add_argument(
        "--subtitles-mode",
        dest="subtitles_mode",
        default=options.Defaults.SUBTITLES_MODE,
        choices=options.SUBTITLES_MODES,
        help=(
            "'soft' (default) muxes a toggleable subtitle track; 'burn' draws "
            "the text into the frames (re-encodes the video); 'off' writes the "
            "SRT file only."
        ),
    )
    parser.add_argument(
        "--max-chars",
        dest="max_chars",
        type=int,
        default=42,
        help="Longest a cue's text may be before it is split.",
    )
    parser.add_argument(
        "--whisper-model",
        default=options.Defaults.WHISPER_MODEL,
        help="Whisper model id.",
    )
    parser.add_argument("--llm-model", default=None, help="Override the configured LLM model.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output root.")
    parser.add_argument(
        "--device",
        type=_device_arg,
        default="auto",
        help="auto, cpu, cuda:N, rocm[:N] (alias for cuda), xpu, or mps.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore the cached transcript.")
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def build_info_parser() -> argparse.ArgumentParser:
    """Parser for the ``info`` subcommand."""
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} info",
        description=(
            "Report the environment: accelerators, the device and precision the "
            "pipeline would pick, optional packages, system binaries and models."
        ),
    )
    parser.add_argument(
        "--device",
        type=_device_arg,
        default="auto",
        help="Device to report on.",
    )
    parser.add_argument(
        "--devices",
        type=_device_list,
        default=None,
        metavar="DEV[,DEV...]",
        help="Report the fan-out these devices would produce, e.g. cuda:0,cuda:1.",
    )
    parser.add_argument(
        "--dtype",
        default=options.Defaults.DTYPE,
        choices=options.DTYPES,
        help="dtype to report on (default: auto).",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Override the models directory."
    )
    return parser


def build_devices_parser() -> argparse.ArgumentParser:
    """Parser for the ``devices`` subcommand."""
    return argparse.ArgumentParser(
        prog=f"{_PROG} devices",
        description=(
            "List every GPU this machine can use, the precision each one would "
            "run, and the exact flag to select it. Read-only: it loads no model "
            "and starts no stage."
        ),
    )


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
    lines.append("all devices      python dub.py devices  (one line per card)")

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


def _format_memory(info: DeviceInfo) -> str:
    """Human-readable VRAM, or ``-`` when torch could not report it."""
    if not info.total_memory:
        return "-"
    return f"{info.total_memory / 1024 ** 3:.1f} GiB"


#: One line per supported family, printed by ``dub.py devices``. The wording names
#: the PyTorch build each family needs: "install a GPU" is not actionable, and
#: "install the ROCm wheel" is.
_SUPPORTED_FAMILIES: tuple[tuple[str, str], ...] = (
    ("cuda", "NVIDIA, with the CUDA build of PyTorch"),
    ("rocm", "AMD, with the ROCm build of PyTorch (reported as cuda)"),
    ("xpu", "Intel Arc / integrated, with the XPU build of PyTorch"),
    ("mps", "Apple Silicon, via Metal"),
    ("cpu", "always available; slower, and the fallback for every other GPU"),
)


def render_devices() -> str:
    """The text printed by ``dub.py devices``.

    Returned rather than printed so it is testable and can be pasted whole into a
    bug report, the same contract as :func:`render_environment`.
    """
    devices = enumerate_devices()
    lines = [f"torch            {_torch_version()}", "", "Supported backends"]
    for name, description in _SUPPORTED_FAMILIES:
        lines.append(f"  {name:<8}{description}")

    lines.extend(
        [
            "",
            "Note: Vulkan is not a backend here. Speech recognition runs on",
            "      CTranslate2 (CUDA or CPU only) and the other stages on PyTorch,",
            "      which has no Vulkan device. AMD cards use ROCm and appear as",
            "      'cuda'; '--device vulkan0' is invalid.",
            "",
        ]
    )

    if not devices:
        lines.append("Detected devices  none - everything will run on the CPU.")
        lines.extend(
            [
                "",
                "Suggested command",
                "  python dub.py --video INPUT --target fr --device cpu",
            ]
        )
        return "\n".join(lines)

    lines.append("Detected devices")
    for info in devices:
        name = info.name or "unknown"
        family = " (ROCm)" if info.kind == "rocm" else ""
        lines.append(f"  {info.label:<9}{name:<32}{_format_memory(info):>9}{family}")
        lines.append(
            f"  {'':<9}bf16 {'yes' if info.supports_bf16 else 'no':<4}"
            f"fp16 {'yes' if info.supports_fp16 else 'no':<4}"
            f"asr {asr_device(info.label):<5} tts {preferred_dtype(info)}"
        )

    labels = [info.label for info in devices]
    lines.extend(["", "Suggested command"])
    lines.append(f"  python dub.py --video INPUT --target fr --device {labels[0]}")
    if len(labels) > 1:
        lines.append(
            "  python dub.py --video INPUT --target fr --devices " + ",".join(labels)
        )
    if any(info.kind in {"xpu", "mps"} for info in devices):
        lines.extend(
            [
                "",
                "ASR stays on the CPU for xpu/mps: CTranslate2 has no XPU or",
                "Metal backend, so only the TTS, separation and diarization",
                "stages use the GPU.",
            ]
        )
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


def cmd_devices(argv: Sequence[str]) -> int:
    """Handle ``dub.py devices``."""
    build_devices_parser().parse_args(list(argv))
    print(render_devices())
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
            if name == "tts_fish":
                print(
                    "  loads int8 weight-only by default (needs torchao in "
                    ".venv-fish); --fish-quantize off uses the bf16 weights, "
                    "which need ~24 GB of VRAM"
                )
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


def _voices_root(args: argparse.Namespace, settings: Settings | None) -> Path:
    """The voice library this invocation uses: ``--voices-dir``, else settings.

    ``settings`` may be ``None`` because :func:`job_config_from_args` is also
    called without a resolved :class:`Settings`; the environment answers then.
    """
    root = getattr(args, "voices_dir", None)
    if root is not None:
        return Path(root)
    if settings is None:
        return Settings.from_env().voices_dir
    return settings.voices_dir


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
    from pipeline import voice_ops
    from pipeline.voice_ops import (
        clone_voice,
        design_voice,
        register_voice,
        remove_voice,
    )
    from pipeline.voices import list_voices

    # See cmd_say: the CLI supplies its resolver to the shared operation.
    voice_ops.resolve_model = resolve_model
    args = build_voices_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir
    root = _voices_root(args, settings)

    if args.remove:
        removed = remove_voice(root, args.remove)
        print(f"Removed {args.remove!r} from {removed}")
        return _STOP

    if args.list or not (args.design or args.clone or args.add is not None):
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

    if args.add is not None:
        if args.speak:
            raise ConfigError(
                "--speak needs a synthesiser, which --add does not load. Register "
                "the voice first, then speak it with 'python dub.py say --voice ...'."
            )
        result = register_voice(
            root,
            models_dir,
            audio=Path(args.add),
            clone_text=args.clone_text,
            transcribe=args.transcribe,
            verify=args.verify,
            language=args.lang,
            label=args.label,
            force=args.force,
            device=args.device,
        )
    elif args.design:
        result = design_voice(
            root,
            models_dir,
            prompt=args.prompt or "",
            language=args.lang or "fr",
            label=args.label,
            speak=args.speak,
            output=args.output,
            force=args.force,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        if args.audio is None:
            raise ConfigError(
                "--clone requires --audio (a clip of the voice to clone)."
            )
        result = clone_voice(
            root,
            models_dir,
            audio=args.audio,
            clone_text=args.clone_text,
            transcribe=args.transcribe,
            verify=args.verify,
            language=args.lang,
            label=args.label,
            speak=args.speak,
            output=args.output,
            force=args.force,
            device=args.device,
            dtype=args.dtype,
        )

    if result.output is not None:
        print(f"Wrote {result.output}")
    encoded = f" + {result.cards} card(s)" if result.cards else ""
    verb = "Registered" if result.source == "import" else "Saved"
    print(
        f"{verb} voice {result.label!r} "
        f"({result.language}, {result.seconds:.1f}s{encoded}) to {result.directory}"
    )
    for warning in result.warnings:
        print(warning)
    print(f'Use it with: python dub.py say --voice {result.label} --text "..."')
    return _STOP


def _cmd_voices_legacy(argv: Sequence[str]) -> int:
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

    if args.list or not (args.design or args.clone or args.add is not None):
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

    if args.add is not None:
        return _cmd_voices_add(args, root, models_dir)

    if args.design:
        if not args.prompt or not args.prompt.strip():
            raise ConfigError("--design requires --prompt (a description of the voice).")
        language = _resolve_language(args, None, fallback="fr")
        anchor_text = _anchor_text(language)
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
            provenance={
                "backend": "voice-design",
                "model": str(resolve_model("tts_voice_design", models_dir)),
                "text_source": "anchor",
                "plan_source": "manual",
            },
        )
        if args.speak:
            speech, speech_rate = engine.generate(args.speak, args.prompt, language)
            out = _write_say_output(args.output, saved.label, speech, speech_rate)
            print(f"Wrote {out}")
        release_gpu_memory()
    else:
        if args.audio is None:
            raise ConfigError("--clone requires --audio (a clip of the voice to clone).")
        # Checked before the clip is read so a missing transcript is reported as
        # such, not as a decode error on a path that was fine all along.
        if not args.transcribe and not (args.clone_text or "").strip():
            raise ConfigError(
                "--clone needs the donated clip's transcript: pass --clone-text, or "
                "--transcribe to derive it from the audio."
            )
        from pipeline.media import load_audio

        try:
            samples, sample_rate = load_audio(args.audio)
        except Exception as exc:  # noqa: BLE001 - surfaced as a user error
            raise ConfigError(f"--clone: could not read {args.audio}: {exc}") from exc
        ref_text, detected = _reference_text(
            args, samples, sample_rate, models_dir, required=True
        )
        language = _resolve_language(args, detected, fallback="fr")
        engine = TTSEngine(
            resolve_model("tts_base", models_dir),
            device=args.device,
            dtype=pick_dtype(args.device, args.dtype),
        )
        # The encoding *is* the saved artifact. Rendering the canonical passage to
        # store that instead would put the voice through the codec twice: once to
        # synthesise it, once to encode it again on the next use.
        engine.prepare_prompt_from_file(args.audio, ref_text)
        prompt = engine.prompt_to_data(
            fingerprint=model_fingerprint("tts_base", models_dir)
        )
        label = args.label or unique_label(default_label(audio=args.audio), root)
        saved = save_voice(
            root,
            label,
            samples,
            sample_rate,
            language=language,
            source="clone",
            ref_text=ref_text,
            prompt=prompt,
            force=args.force,
            provenance={
                "backend": "voice-clone",
                "model": str(resolve_model("tts_base", models_dir)),
                "ref_audio": str(args.audio),
                "text_source": "transcribe" if args.transcribe else "clone-text",
                "detected_language": detected,
                "fingerprint": model_fingerprint("tts_base", models_dir),
                "plan_source": "manual",
            },
        )
        if args.speak:
            rendered = engine.synthesize([args.speak], language=language)
            speech = rendered[0] if rendered else None
            if speech is None:
                raise TTSError("The cloning model returned no audio.")
            out = _write_say_output(
                args.output, saved.label, speech, engine.sample_rate
            )
            print(f"Wrote {out}")
        release_gpu_memory()

    encoded = f" + {len(saved.cards)} card(s)" if saved.cards else ""
    print(
        f"Saved voice {saved.label!r} "
        f"({saved.language}, {saved.seconds:.1f}s{encoded}) to {saved.directory}"
    )
    print(f'Use it with: python dub.py say --voice {saved.label} --text "..."')
    return _STOP


def _cmd_voices_add(args: argparse.Namespace, root: Path, models_dir: Path) -> int:
    """Handle ``voices --add``: register a clip without loading a model."""
    from pipeline.media import load_audio
    from pipeline.voices import default_label, import_voice, unique_label

    audio = Path(args.add)
    if not audio.is_file():
        raise ConfigError(f"--add: no such audio file: {audio}")
    if args.speak:
        raise ConfigError(
            "--speak needs a synthesiser, which --add does not load. Register the "
            "voice first, then speak it with 'python dub.py say --voice ...'."
        )
    try:
        samples, sample_rate = load_audio(audio)
    except Exception as exc:  # noqa: BLE001 - surfaced as a user error
        raise ConfigError(f"--add: could not read {audio}: {exc}") from exc
    if samples.size == 0:
        raise ConfigError(f"--add: {audio} contains no audio.")

    ref_text, detected = _reference_text(
        args, samples, sample_rate, models_dir, required=False
    )
    language = _resolve_language(args, detected, fallback=None)
    seconds = float(samples.size) / float(sample_rate or 1)
    if seconds < 3.0:
        _LOGGER.warning(
            "The clip is only %.1fs long; a speaker embedding needs a few seconds "
            "of clean speech to be stable.",
            seconds,
        )
    elif seconds > 30.0:
        _LOGGER.warning(
            "The clip is %.0fs long. All of it conditions the prompt, so a trimmed "
            "8-15 s excerpt usually clones better and faster.",
            seconds,
        )

    label = args.label or unique_label(default_label(audio=audio), root)
    saved = import_voice(
        root,
        label,
        (samples, sample_rate),
        language=language,
        ref_text=ref_text,
        force=args.force,
        provenance={
            "backend": "voice-add",
            "ref_audio": str(audio),
            "text_source": (
                "transcribe"
                if args.transcribe
                else ("clone-text" if (args.clone_text or "").strip() else "none")
            ),
            "detected_language": detected,
            "plan_source": "manual",
        },
    )
    print(
        f"Registered voice {saved.label!r} ({saved.language}, {saved.seconds:.1f}s) "
        f"from {audio} -> {saved.directory}"
    )
    if not ref_text.strip():
        print(
            "No transcript stored, so this voice clones from the speaker embedding "
            "alone. Add --clone-text or --transcribe for a closer match."
        )
    print(f'Use it with: python dub.py say --voice {saved.label} --text "..."')
    return _STOP


def _resolve_language(
    args: argparse.Namespace, detected: str | None, *, fallback: str | None
) -> str:
    """The language a voice is recorded in: explicit, else detected, else default.

    ``--add`` has nothing to infer from, so it passes no fallback: a clip
    registered without ``--transcribe`` must say which language it is in.
    """
    if args.lang:
        language = normalize_language(args.lang)
        if language == "Auto":
            raise ConfigError("--lang must be an explicit language, not 'auto'.")
        if detected:
            heard = normalize_language(detected)
            if heard != language:
                _LOGGER.warning(
                    "--lang is %s but the reference clip sounds like %s. The "
                    "language is what 'say --voice' defaults to later.",
                    language,
                    heard,
                )
        return language
    if detected:
        return normalize_language(detected)
    if fallback is not None:
        return normalize_language(fallback)
    raise ConfigError(
        "--add needs --lang (the language spoken in the clip), or --transcribe to "
        "detect it from the audio."
    )


def _reference_text(
    args: argparse.Namespace,
    samples: np.ndarray,
    sample_rate: int,
    models_dir: Path,
    *,
    required: bool,
) -> tuple[str, str | None]:
    """Resolve a reference clip's transcript for ``--clone``/``--add``.

    ``--clone-text`` is used exactly as given. ``--transcribe`` derives the text
    instead, and is what makes ``--lang`` optional. ``--verify`` asks the ASR even
    when the text was typed, because ICL cloning aligns the speech codes against
    that text: a transcript that does not match the audio yields a wrong voice
    rather than an error, so a warning is the only useful signal.

    Returns:
        ``(transcript, detected_language)``; the language is ``None`` unless the
        ASR ran.
    """
    given = (args.clone_text or "").strip()
    if args.transcribe and given:
        raise ConfigError(
            "--transcribe and --clone-text are alternatives: pass one, not both."
        )
    heard, detected = "", None
    if args.transcribe or args.verify:
        heard, detected = _transcribe_clip(samples, sample_rate, models_dir, args)
        if not heard:
            raise ConfigError(
                f"No speech was recognised in {args.add or args.audio}; pass "
                "--clone-text instead of --transcribe."
            )
    if args.verify and given and heard:
        overlap = _word_overlap(given, heard)
        if overlap < 0.6:
            _LOGGER.warning(
                "The transcript shares only %d%% of its words with what was heard "
                "in %s. ICL cloning aligns the codes against this text, so this is "
                "likely to produce the wrong voice. Heard: %r",
                round(overlap * 100),
                args.add or args.audio,
                heard,
            )
    text = heard if args.transcribe else given
    if not text and required:
        raise ConfigError(
            "--clone needs the donated clip's transcript: pass --clone-text, or "
            "--transcribe to derive it from the audio."
        )
    return text, detected


def _transcribe_clip(
    samples: np.ndarray,
    sample_rate: int,
    models_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, str | None]:
    """Transcribe a reference clip with the local ASR.

    The samples are written to a temporary mono WAV first, so the transcript
    describes exactly the audio that gets stored rather than the original file's
    channel layout or rate.
    """
    import tempfile

    from pipeline.asr import transcribe
    from pipeline.media import write_audio

    model_dir = resolve_model("whisper_large_v3", models_dir)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_audio(Path(tmp) / "reference.wav", samples, int(sample_rate))
        result = transcribe(
            path,
            model_dir,
            device=args.device,
            language=language_code_for_name(args.lang),
        )
    text = " ".join(segment.text.strip() for segment in result.segments).strip()
    return text, result.language


def _word_overlap(left: str, right: str) -> float:
    """Jaccard overlap of two transcripts' words, ``0.0``-``1.0``.

    Set-based, so a re-segmented but identical transcript scores ``1.0`` no
    matter how the ASR punctuated or ordered it.
    """
    import re

    def words(value: str) -> set[str]:
        return set(re.findall(r"\w+", value.lower()))

    first, second = words(left), words(right)
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _say_lines(text: str) -> list[str]:
    """Split ``--text`` into lines, treating a literal ``\\n`` as a break.

    A shell does not expand ``\\n`` inside double quotes (and ``$"..."`` is
    locale quoting, not ANSI-C quoting), so ``--text "one\\ntwo"`` reaches us as
    the two characters backslash-n. Interpreting that as a line break makes the
    obvious invocation do the obvious thing; real newlines still work too.
    """
    normalized = str(text or "").replace("\\n", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _parse_emotions(values: Sequence[str] | None) -> list[str] | None:
    """Flatten repeatable ``--emotion``/``--instruct`` values into one list.

    A value may itself be comma-separated ("calm, excited"). Empty entries are
    dropped; ``None`` or an all-empty list becomes ``None`` ("no instruction").
    ``--style-instruct`` is a separate, verbatim flag and never passes through
    here.
    """
    if values is None:
        return None
    flat = [
        part.strip() for value in values for part in value.split(",") if part.strip()
    ]
    return flat or None


def _one_per_line(
    values: Sequence[str] | None, lines: Sequence[str], flag: str
) -> list[str] | None:
    """Broadcast a single value over every line, or validate one per line."""
    if values is None:
        return None
    items = list(values)
    if len(items) == 1:
        return items * len(lines)
    if len(items) != len(lines):
        raise ConfigError(
            f"{flag}: {len(items)} value(s) for {len(lines)} line(s); give one per "
            "line, or a single one to apply everywhere."
        )
    return items


def _say_plan_source(backend: str, args: argparse.Namespace) -> str:
    """Whether a ``say`` plan came from the model, the CLI, or nowhere."""
    tag_mode = getattr(args, "fish_tag_mode", options.Defaults.FISH_TAG_MODE)
    if backend == options.SayBackend.CUSTOMVOICE:
        style_plan = getattr(args, "style_plan", options.Defaults.STYLE_PLAN)
        if style_plan == options.StylePlan.LLM:
            return "llm"
        if (
            getattr(args, "style_instructs", None)
            or getattr(args, "prosody", None)
            or getattr(args, "emotions", None)
        ):
            return "manual"
        return "off"
    if tag_mode == "llm":
        return "llm"
    if getattr(args, "prosody_plan", "off") == "llm":
        return "llm"
    if getattr(args, "prosody", None) or getattr(args, "emotions", None):
        return "manual"
    if backend == options.SayBackend.FISH:
        return "manual"  # Fish takes its direction from tags in --text
    return "off"


def _write_say_provenance(recorder: Any, name: str) -> None:
    """Write a ``say`` run's provenance beside its lines, timestamped per run."""
    from pipeline.provenance import timed_provenance_path

    recorder.write(timed_provenance_path(Path("output") / "say" / name / "provenance"))


def _write_qwen_say_provenance(
    name: str, language: str | None, mode: str, *, enabled: bool = True
) -> None:
    """Record the absence of a prosody channel for a Qwen ``say`` run."""
    if not enabled:
        return
    from pipeline.provenance import ProvenanceRecorder, timed_provenance_path

    recorder = ProvenanceRecorder("say", "qwen", language=language, plan_source="off")
    recorder.record_output(mode=mode)
    recorder.write(
        timed_provenance_path(Path("output") / "say" / name / "provenance"),
        supported=False,
    )


def _resolve_prosody_specs(
    lines: Sequence[str],
    *,
    emotions: Sequence[str] | None,
    values: Sequence[str] | None,
    plan: str,
    settings: Settings,
    backend: str,
    chunk_size: int | None = None,
    recorder: Any | None = None,
) -> list[Any]:
    """Per-line :class:`ProsodySpec` from the CLI and, optionally, the LLM.

    Precedence is deliberate: the LLM plans first, an explicit ``--prosody``
    overrides it, and ``--emotion`` overrides the emotion field. A user who types
    a value always wins over a model's guess.
    """
    from pipeline.prosody import ProsodyPlanner, ProsodySpec, parse_prosody_spec

    specs = [ProsodySpec() for _ in lines]
    if plan == "llm":
        planner = ProsodyPlanner(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            backend=backend,
            chunk_size=chunk_size or settings.llm_plan_chunk_size,
            recorder=recorder,
        )
        specs = planner.plan(lines)

    if values:
        parsed = _one_per_line(
            [parse_prosody_spec(value) for value in values], lines, "--prosody"
        )
        assert parsed is not None
        specs = [base.merged_with(override) for base, override in zip(specs, parsed)]

    if emotions:
        chosen = _one_per_line(emotions, lines, "--emotion")
        assert chosen is not None
        specs = [
            base.merged_with(ProsodySpec(emotion=emotion))
            for base, emotion in zip(specs, chosen)
        ]
    return specs


def _neural_engine(
    backend: str,
    models_dir: Path,
    device: str,
    *,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    temperature: float | None = None,
    fish_temperature: float | None = None,
    fish_top_p: float | None = None,
    fish_quantize: str | None = None,
) -> Any:
    """Build the engine for one expressive backend.

    Delegates to :func:`pipeline.backends.build_expressive_engine` so the dubbing
    pipeline resolves exactly the same engine from the same knobs.
    """
    from pipeline.backends import build_expressive_engine

    return build_expressive_engine(
        backend,
        models_dir,
        device,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        fish_temperature=fish_temperature,
        fish_top_p=fish_top_p,
        fish_quantize=fish_quantize,
        resolve=resolve_model,
    )


def _say_name(args: argparse.Namespace) -> str:
    """The folder/file stem a ``say`` run writes under."""
    from pipeline.voices import slugify_label

    if getattr(args, "voice", None):
        return slugify_label(args.voice)
    if getattr(args, "clone", None) is not None:
        return slugify_label(Path(args.clone).stem)
    if getattr(args, "speaker", None):
        return slugify_label(f"customvoice-{args.speaker}")
    return slugify_label(args.save_as or "design")


def _say_log_file(
    args: argparse.Namespace, settings: Settings, name: str
) -> Path | None:
    """Where this ``say`` run keeps its log.

    Mirrors the dubbing command: an explicit path wins, then ``LOG_FILE``, then a
    default beside the output. The expressive backends already write a folder per
    voice, so their log lives inside it; Qwen writes a single WAV, so its log is
    its sibling.
    """
    if args.log_file is False:
        return None
    if isinstance(args.log_file, Path):
        return args.log_file
    if settings.log_file is not None:
        return settings.log_file
    base = Path("output") / "say"
    if args.backend == options.SayBackend.QWEN:
        return base / f"{name}.log"
    return base / name / "say.log"


def _cmd_say_neural(
    args: argparse.Namespace,
    models_dir: Path,
    root: Path,
    *,
    backend: str,
    emotions: list[str] | None,
    settings: Settings,
    log: Any,
) -> int:
    """The expressive-backend path of ``dub.py say``.

    ``qwen`` is handled separately in :func:`cmd_say`, because it cannot honour
    prosody and keeps the plain clone call.
    """
    from pipeline.config import NAME_TO_CODE
    from pipeline.media import write_audio
    from pipeline.voices import load_voice, slugify_label

    if args.design is not None:
        raise ConfigError(
            f"--backend {backend} cannot be combined with --design: pass --voice or "
            "--clone instead."
        )
    if args.save_as:
        raise ConfigError(f"--save-as is not supported with --backend {backend} yet.")

    if args.voice:
        saved = load_voice(root, args.voice)
        reference = saved.wav
        name = saved.label
        language = normalize_language(args.lang) if args.lang else saved.language
        ref_text = saved.ref_text or None
        if saved.cards:
            # Cards hold one checkpoint's encoding; this backend cannot read it.
            log.info(
                "%s: using voice.wav directly (stored cards are Qwen-only)",
                backend,
            )
    elif args.clone is not None:
        reference = args.clone
        name = slugify_label(args.clone.stem)
        # CosyVoice infers the language from the text; Chatterbox's multilingual
        # model needs the code, so only that backend requires --lang here.
        language = normalize_language(args.lang) if args.lang else None
        ref_text = (args.clone_text or "").strip() or None
        if backend == options.SayBackend.CHATTERBOX and language is None:
            raise ConfigError(
                "--lang is required with --backend chatterbox when the voice is "
                "cloned inline."
            )
    else:  # pragma: no cover - argparse requires one of the group
        raise ConfigError(f"--backend {backend} needs --voice or --clone.")

    lines = _say_lines(args.text)
    if not lines:
        raise ConfigError("--text contains no lines to speak.")

    from pipeline.provenance import (
        LineProvenance,
        NullRecorder,
        ProvenanceRecorder,
        describe_transformations,
    )

    recorder = (
        NullRecorder()
        if getattr(args, "no_provenance", False)
        else ProvenanceRecorder(
            "say",
            backend,
            language=language,
            plan_source=_say_plan_source(backend, args),
        )
    )
    trace: list[dict[str, Any]] = []
    specs: list[Any] | None = None
    tagged: list[str] | None = None

    log.info(
        "Backend %s: reference=%s language=%s lines=%d",
        backend,
        reference,
        language or "auto",
        len(lines),
    )
    with timed(log, f"Resolved {backend} engine", backend=backend):
        engine = _neural_engine(
            backend,
            models_dir,
            args.device,
            exaggeration=getattr(args, "chatterbox_exaggeration", None),
            cfg_weight=getattr(args, "chatterbox_cfg_weight", None),
            temperature=getattr(args, "chatterbox_temperature", None),
            fish_temperature=getattr(args, "fish_temperature", None),
            fish_top_p=getattr(args, "fish_top_p", None),
            fish_quantize=getattr(args, "fish_quantize", None) or "int8",
        )

    # Fish has no spec: its delivery lives in the text, as tags. The planner can
    # write them ('llm'), or the caller already did ('keep', the default). Every
    # other backend renders one spec per line.
    fish_tag_mode = getattr(args, "fish_tag_mode", options.Defaults.FISH_TAG_MODE)
    if backend == options.SayBackend.FISH and fish_tag_mode == options.FishTagMode.LLM:
        # Fish has no side channel, so the plan *is* the text: the model writes
        # the tags into each line and a guard rejects any change to the words.
        from pipeline.prosody import FishInlinePlanner

        planner = FishInlinePlanner(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            chunk_size=args.plan_chunk or settings.llm_plan_chunk_size,
            recorder=recorder,
        )
        tagged = planner.plan_lines(lines, language=language)
        for index, text in enumerate(tagged, start=1):
            log.debug("fish inline %d/%d: %r", index, len(tagged), text)
        with timed(
            log,
            f"Synthesised {len(lines)} line(s)",
            lines=len(lines),
            backend=backend,
        ):
            rendered = engine.synthesize(
                tagged,
                reference=reference,
                ref_text=ref_text,
                language=language,
                trace=trace,
            )
    elif backend == options.SayBackend.FISH:
        # The caller wrote the tags. Reserved <|...|> tokens were rejected in
        # validation, so this only reports the advisory problems and folds a
        # tag-only line into its neighbour, which would otherwise be an empty
        # utterance.
        from pipeline.prosody import (
            strip_fish_tags,
            validate_fish_inline_text,
        )

        checked: list[str] = []
        for index, text in enumerate(lines, start=1):
            for problem in validate_fish_inline_text(text):
                log.warning("line %d: %s", index, problem)
            if not strip_fish_tags(text):
                if not checked:
                    log.warning("dropping a first line that is only tags: %r", text)
                    continue
                log.warning(
                    "line %d is only tags; merging it into the line before.", index
                )
                checked[-1] = f"{checked[-1]} {text}"
                continue
            checked.append(text)
        if not checked:
            raise ConfigError("--text contains only Fish tags, nothing to speak.")
        lines = checked
        log.info("Fish: %d hand-tagged line(s) sent as written", len(lines))
        with timed(
            log,
            f"Synthesised {len(lines)} line(s)",
            lines=len(lines),
            backend=backend,
        ):
            rendered = engine.synthesize(
                lines,
                reference=reference,
                ref_text=ref_text,
                language=language,
                trace=trace,
            )
    else:
        # CosyVoice 3 and Chatterbox: one spec per line, rendered by the engine.
        specs = _resolve_prosody_specs(
            lines,
            emotions=emotions,
            values=args.prosody,
            plan=args.prosody_plan,
            settings=settings,
            backend=backend,
            chunk_size=args.plan_chunk,
            recorder=recorder,
        )
        log.info(
            "Prosody: %s for %d line(s)",
            "LLM plan"
            if args.prosody_plan == options.SayProsodyPlan.LLM
            else "explicit values",
            len(specs),
        )
        for index, spec in enumerate(specs, start=1):
            log.debug("prosody %d/%d: %s", index, len(specs), spec.model_dump())
        with timed(
            log,
            f"Synthesised {len(lines)} line(s)",
            lines=len(lines),
            backend=backend,
        ):
            if backend == options.SayBackend.CHATTERBOX:
                rendered = engine.synthesize(
                    lines,
                    reference=reference,
                    prosody=specs,
                    language=NAME_TO_CODE.get(language),
                    trace=trace,
                )
            else:
                rendered = engine.synthesize(
                    lines,
                    reference=reference,
                    prosody=specs,
                    ref_text=ref_text,
                    language=language,
                    trace=trace,
                )

    out_dir = Path("output") / "say" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with timed(log, "Wrote output", lines=len(rendered)):
        for index, wav in enumerate(rendered):
            write_audio(out_dir / f"line_{index:03d}.wav", wav, engine.sample_rate)
        combined = (
            np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float32)
        )
        combined_path = (
            args.output if args.output is not None else out_dir / "speech.wav"
        )
        write_audio(combined_path, combined, engine.sample_rate)

    spoken_lines = tagged if tagged is not None else lines
    for index, planned in enumerate(lines):
        record = trace[index] if index < len(trace) else None
        spoken = str(record.get("text")) if record else spoken_lines[index]
        spec = (
            specs[index].model_dump()
            if specs is not None and index < len(specs)
            else None
        )
        recorder.record_line(
            LineProvenance(
                key=index,
                text_planned=planned,
                text_spoken=spoken,
                overrides={"prosody": args.prosody, "emotion": emotions},
                spec=spec,
                rendered=record,
                transformations=describe_transformations(
                    planned,
                    spoken,
                    channel=str(record.get("channel")) if record else backend,
                    spec=spec,
                    instruct=record.get("instruct") if record else None,
                    exaggeration=record.get("exaggeration") if record else None,
                ),
            )
        )
    _write_say_provenance(recorder, name)
    release_gpu_memory()

    log.info("Output: %s (combined %s)", out_dir, combined_path)
    print(f"Wrote {len(rendered)} line(s) to {out_dir} (combined: {combined_path})")
    return _STOP


def _validate_say_source(args: argparse.Namespace) -> None:
    """Check the source/backend combination argparse cannot express.

    ``--speaker`` and the ``--voice/--design/--clone`` group are alternatives that
    depend on the backend, so the parser leaves the group unrequired and this
    enforces exactly one valid source per backend.
    """
    if args.style_instructs and args.backend != options.SayBackend.CUSTOMVOICE:
        raise ConfigError(
            "--style-instruct is only valid with --backend customvoice; the other "
            "backends take --prosody/--emotion (and --design already accepts a "
            "free-form prompt)."
        )
    if (
        args.style_plan != options.StylePlan.OFF
        and args.backend != options.SayBackend.CUSTOMVOICE
    ):
        raise ConfigError(
            "--style-plan is only valid with --backend customvoice; the other "
            "backends use --prosody-plan llm for structured prosody."
        )
    if (
        args.style_plan_scope != options.StylePlanScope.PER_LINE
        and args.style_plan != options.StylePlan.LLM
    ):
        raise ConfigError("--style-plan-scope only applies with --style-plan llm.")
    tag_mode = getattr(args, "fish_tag_mode", options.Defaults.FISH_TAG_MODE)
    if tag_mode != options.FishTagMode.KEEP and args.backend != options.SayBackend.FISH:
        raise ConfigError("--fish-tag-mode is only valid with --backend fish.")
    if (
        getattr(args, "fish_quantize", None) is not None
        and args.backend != options.SayBackend.FISH
    ):
        raise ConfigError("--fish-quantize is only valid with --backend fish.")
    if args.backend == options.SayBackend.FISH and (
        args.prosody or args.emotions or args.prosody_plan != options.SayProsodyPlan.OFF
    ):
        raise ConfigError(
            "--backend fish takes its direction from [tags] inside --text (or "
            "--fish-tag-mode llm); it has no --prosody/--emotion/--prosody-plan "
            "channel."
        )
    if args.backend == options.SayBackend.FISH and tag_mode == options.FishTagMode.KEEP:
        from pipeline.prosody import fish_structural_token

        for index, line in enumerate(_say_lines(args.text), start=1):
            token = fish_structural_token(line)
            if token:
                raise ConfigError(
                    f"line {index} of --text contains {token!r}, which Fish "
                    "reserves for its own prompt; remove it."
                )
    if args.fish_temperature is not None and args.backend != options.SayBackend.FISH:
        raise ConfigError("--fish-temperature is only valid with --backend fish.")
    if args.fish_top_p is not None and args.backend != options.SayBackend.FISH:
        raise ConfigError("--fish-top-p is only valid with --backend fish.")
    if args.backend == options.SayBackend.CUSTOMVOICE:
        if not args.speaker:
            raise ConfigError(
                "--backend customvoice needs --speaker (a preset voice, e.g. "
                "'Vivian' or 'Ryan')."
            )
        if args.voice or args.design is not None or args.clone is not None:
            raise ConfigError(
                "--backend customvoice uses a preset --speaker and does not take "
                "--voice/--design/--clone."
            )
        return
    if args.speaker:
        raise ConfigError("--speaker is only valid with --backend customvoice.")
    if not (args.voice or args.design is not None or args.clone is not None):
        raise ConfigError(
            "One of --voice, --design or --clone is required (unless --backend "
            "customvoice, which needs --speaker)."
        )


def _cmd_say_customvoice(
    args: argparse.Namespace,
    models_dir: Path,
    settings: Settings,
    *,
    emotions: list[str] | None,
    log: Any,
) -> int:
    """The preset-speaker CustomVoice path of ``dub.py say``.

    CustomVoice runs in-process like the Qwen clone backend, but takes a preset
    ``--speaker`` instead of a reference clip. A tone/emotion instruction is the
    only style control it accepts: either a verbatim ``--style-instruct`` or, when
    that is absent, the prosody spec rendered into one.
    """
    from pipeline.config import normalize_language
    from pipeline.media import write_audio
    from pipeline.prosody import StylePlanner, customvoice_instruct
    from pipeline.tts import CustomVoiceEngine

    if args.lang is None:
        raise ConfigError("--lang is required with --backend customvoice.")
    language = normalize_language(args.lang)
    if language == "Auto":
        raise ConfigError("--lang must be an explicit language, not 'auto'.")
    if args.save_as:
        raise ConfigError("--save-as is not supported with --backend customvoice.")

    lines = _say_lines(args.text)
    if not lines:
        raise ConfigError("--text contains no lines to speak.")

    from pipeline.provenance import (
        LineProvenance,
        NullRecorder,
        ProvenanceRecorder,
        describe_transformations,
    )

    recorder = (
        NullRecorder()
        if getattr(args, "no_provenance", False)
        else ProvenanceRecorder(
            "say",
            "customvoice",
            language=language,
            plan_source=_say_plan_source("customvoice", args),
        )
    )

    if args.style_instructs:
        # A verbatim instruction is the checkpoint's own input format, so it is
        # sent as typed; anything structural given alongside it is dropped.
        if args.style_plan != options.StylePlan.OFF:
            log.warning("--style-instruct overrides --style-plan for this run.")
        if args.prosody or emotions or args.prosody_plan != options.SayProsodyPlan.OFF:
            log.warning(
                "--style-instruct overrides --prosody/--emotion/--prosody-plan for "
                "this run."
            )
        instructs = _one_per_line(
            [value.strip() for value in args.style_instructs], lines, "--style-instruct"
        )
        assert instructs is not None
        specs = None
        mode = "verbatim"
        label = "style-instruct"
    elif args.style_plan == options.StylePlan.LLM:
        # The richer form of --prosody-plan: the model writes the verbatim
        # instruction the checkpoint was trained to read, not a rendered template.
        if args.prosody or emotions or args.prosody_plan != options.SayProsodyPlan.OFF:
            log.warning(
                "--style-plan overrides --prosody/--emotion/--prosody-plan for "
                "this run."
            )
        planner = StylePlanner(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            chunk_size=args.plan_chunk or settings.llm_plan_chunk_size,
            backend="customvoice",
            recorder=recorder,
        )
        if args.style_plan_scope == "whole":
            instruction = planner.plan_whole(
                lines, speaker=args.speaker, language=language
            )
            instructs = [instruction] * len(lines)
        else:
            instructs = planner.plan_lines(
                lines, speaker=args.speaker, language=language
            )
        specs = None
        mode = "verbatim-llm"
        label = "style-plan"
    else:
        specs = _resolve_prosody_specs(
            lines,
            emotions=emotions,
            values=args.prosody,
            plan=args.prosody_plan,
            settings=settings,
            backend="customvoice",
            chunk_size=args.plan_chunk,
            recorder=recorder,
        )
        instructs = [customvoice_instruct(spec) for spec in specs]
        mode = "rendered"
        label = "prosody"
    log.info(
        "CustomVoice: speaker=%s language=%s lines=%d instruction=%s",
        args.speaker,
        language,
        len(lines),
        mode,
    )
    for index, instruct in enumerate(instructs, start=1):
        if specs is None:
            log.debug("%s %d/%d: %r", label, index, len(instructs), instruct)
        else:
            log.debug(
                "prosody %d/%d: %s -> %r",
                index,
                len(specs),
                specs[index - 1].model_dump(),
                instruct,
            )

    engine = CustomVoiceEngine(
        resolve_model("tts_custom_voice", models_dir),
        device=args.device,
        dtype=pick_dtype(args.device, args.dtype),
    )
    speakers = engine.speakers()
    if speakers and args.speaker.lower() not in {name.lower() for name in speakers}:
        raise ConfigError(
            f"Unknown --speaker {args.speaker!r}. Available: " + ", ".join(speakers)
        )

    rendered: list[np.ndarray] = []
    sample_rate = 0
    with timed(
        log,
        f"Synthesised {len(lines)} line(s)",
        lines=len(lines),
        backend="customvoice",
    ):
        for index, (text, instruct) in enumerate(zip(lines, instructs), start=1):
            samples, sample_rate = engine.generate(
                text, args.speaker, instruct, language
            )
            rendered.append(samples)
            log.debug(
                "customvoice line %d/%d: %d sample(s), instruct=%r",
                index,
                len(lines),
                samples.size,
                instruct,
            )

    out_dir = Path("output") / "say" / _say_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    with timed(log, "Wrote output", lines=len(rendered)):
        for index, wav in enumerate(rendered):
            write_audio(out_dir / f"line_{index:03d}.wav", wav, sample_rate)
        combined = (
            np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float32)
        )
        combined_path = (
            args.output if args.output is not None else out_dir / "speech.wav"
        )
        write_audio(combined_path, combined, sample_rate)
    release_gpu_memory()

    for index, (text, instruct) in enumerate(zip(lines, instructs)):
        record = {
            "channel": "customvoice",
            "text": text,
            "instruct": instruct,
            "speaker": args.speaker,
        }
        spec = (
            specs[index].model_dump()
            if specs is not None and index < len(specs)
            else None
        )
        recorder.record_line(
            LineProvenance(
                key=index,
                text_planned=text,
                text_spoken=text,
                overrides={
                    "style_instruct": args.style_instructs,
                    "prosody": args.prosody,
                    "emotion": emotions,
                },
                spec=spec,
                rendered=record,
                transformations=describe_transformations(
                    text,
                    text,
                    channel="customvoice",
                    spec=spec,
                    instruct=instruct,
                ),
            )
        )
    _write_say_provenance(recorder, _say_name(args))

    log.info("Output: %s (combined %s)", out_dir, combined_path)
    print(f"Wrote {len(rendered)} line(s) to {out_dir} (combined: {combined_path})")
    return _STOP


def _install_saved_prompt(
    engine: Any,
    saved: Any,
    models_dir: Path,
    settings: Settings,
    log: Any,
) -> None:
    """Prepare ``engine`` from a saved voice, reusing its stored card.

    Encoding the reference is the cost this path exists to avoid, so the order is:
    a card for this exact backend, model and clip, else encode once, else cache
    the result. Cards for other models are left alone: switching checkpoints must
    not cost the first one its card.
    """
    from pipeline.voices import find_prompt, save_prompt

    fingerprint = model_fingerprint("tts_base", models_dir)
    path = find_prompt(
        saved.directory, "qwen", fingerprint, ref_text=saved.ref_text
    )
    if path is not None:
        from pipeline.voices import load_prompt

        log.info("Loaded the stored prompt for %s", saved.label)
        engine.prepare_prompt_from_data(load_prompt(path), seconds=saved.seconds)
        return

    others = len(saved.cards)
    if others:
        log.info(
            "No usable qwen card for this model; %d existing card(s) are kept.",
            others,
        )
    engine.prepare_prompt_from_file(saved.wav, saved.ref_text)
    if not settings.voices_prompt_cache:
        return
    try:
        cached = save_prompt(
            saved.directory, engine.prompt_to_data(fingerprint=fingerprint)
        )
    except (OSError, TTSError) as exc:
        log.warning("Could not cache the prompt for %s: %s", saved.label, exc)
        return
    log.info("Cached the prompt for %s at %s", saved.label, cached)


def cmd_say(argv: Sequence[str]) -> int:
    """Handle ``dub.py say``: speak a text with a saved, designed or cloned voice."""
    from pipeline import say as say_module
    from pipeline.say import SaySpec, render_speech

    # Share this module's resolver, so a caller that monkeypatches (or wraps)
    # ``dub.resolve_model`` changes what the shared operation uses too.
    say_module.resolve_model = resolve_model
    args = build_say_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir
    root = _voices_root(args, settings)
    say_name = _say_name(args)
    setup_logging(
        "DEBUG" if args.verbose else settings.log_level,
        settings.log_json,
        log_file=_say_log_file(args, settings, say_name),
        max_bytes=args.log_max_bytes,
    )
    log = job_logger("dub.say", f"say-{say_name}")
    spec = SaySpec(
        text=args.text,
        backend=args.backend,
        voice=args.voice,
        design=args.design,
        clone=args.clone,
        clone_text=args.clone_text,
        speaker=args.speaker,
        language=args.lang,
        save_as=args.save_as,
        output=args.output,
        device=args.device,
        dtype=args.dtype,
        force=args.force,
        provenance=not args.no_provenance,
        prosody=args.prosody,
        emotions=args.emotions,
        prosody_plan=args.prosody_plan,
        style_instructs=args.style_instructs,
        style_plan=args.style_plan,
        style_plan_scope=args.style_plan_scope,
        plan_chunk=args.plan_chunk,
        chatterbox_exaggeration=args.chatterbox_exaggeration,
        chatterbox_cfg_weight=args.chatterbox_cfg_weight,
        chatterbox_temperature=args.chatterbox_temperature,
        fish_temperature=args.fish_temperature,
        fish_top_p=args.fish_top_p,
        fish_quantize=args.fish_quantize,
        fish_tag_mode=args.fish_tag_mode,
    )
    result = render_speech(
        spec, settings, models_dir=models_dir, voices_root=root, log=log
    )
    print(result.summary())
    return _STOP


def _cmd_say_legacy(argv: Sequence[str]) -> int:
    """Handle ``dub.py say``: speak a text with a saved, designed or cloned voice."""
    from pipeline.tts import TTSEngine, VoiceDesignEngine
    from pipeline.voices import load_voice, save_voice

    args = build_say_parser().parse_args(list(argv))
    settings = Settings.from_env()
    models_dir = args.models_dir or settings.models_dir
    root = _voices_root(args, settings)

    if not args.text or not args.text.strip():
        raise ConfigError("--text cannot be empty.")

    _validate_say_source(args)

    emotions = _parse_emotions(args.emotions)
    if args.backend == options.SayBackend.QWEN and (
        emotions or args.prosody or args.prosody_plan != options.SayProsodyPlan.OFF
    ):
        raise ConfigError(
            "--emotion/--prosody/--prosody-plan only applies to --backend "
            "customvoice, cosyvoice3 or chatterbox; fish takes tags in --text."
        )

    say_name = _say_name(args)
    setup_logging(
        "DEBUG" if args.verbose else settings.log_level,
        settings.log_json,
        log_file=_say_log_file(args, settings, say_name),
        max_bytes=args.log_max_bytes,
    )
    log = job_logger("dub.say", f"say-{say_name}")
    lines = _say_lines(args.text)
    fish_tag_mode = getattr(args, "fish_tag_mode", "keep")
    if args.style_instructs:
        prosody_state = "verbatim"
    elif args.style_plan == options.StylePlan.LLM:
        prosody_state = "verbatim-llm"
    elif (
        args.backend == options.SayBackend.FISH
        and fish_tag_mode == options.FishTagMode.LLM
    ):
        prosody_state = "inline-llm"
    elif (
        args.backend == options.SayBackend.FISH
        and fish_tag_mode == options.FishTagMode.KEEP
    ):
        prosody_state = "inline"
    elif args.prosody_plan == options.SayProsodyPlan.LLM:
        prosody_state = "llm"
    elif args.prosody or emotions:
        prosody_state = "manual"
    else:
        prosody_state = "off"
    log.info(
        "say: backend=%s voice=%s lines=%d language=%s prosody=%s",
        args.backend,
        say_name,
        len(lines),
        args.lang or "auto",
        prosody_state,
    )

    if args.backend == options.SayBackend.CUSTOMVOICE:
        return _cmd_say_customvoice(
            args,
            models_dir,
            settings,
            emotions=emotions,
            log=log,
        )
    if args.backend != options.SayBackend.QWEN:
        return _cmd_say_neural(
            args,
            models_dir,
            root,
            backend=args.backend,
            emotions=emotions,
            settings=settings,
            log=log,
        )

    if args.voice:
        saved = load_voice(root, args.voice)
        language = normalize_language(args.lang) if args.lang else saved.language
        has_text = bool(saved.ref_text.strip())
        log.info("Qwen: cloning from saved voice %s (%s)", saved.label, language)
        engine = TTSEngine(
            resolve_model("tts_base", models_dir),
            device=args.device,
            dtype=pick_dtype(args.device, args.dtype),
            x_vector_only=not has_text,
        )
        with timed(log, "Prepared the clone prompt"):
            _install_saved_prompt(engine, saved, models_dir, settings, log)
        with timed(log, "Synthesised the line", lines=1, backend="qwen"):
            rendered = engine.synthesize([args.text], language=language)
        speech = rendered[0] if rendered else None
        if speech is None:
            raise TTSError("The model returned no audio.")
        out = _write_say_output(args.output, saved.label, speech, engine.sample_rate)
        release_gpu_memory()
        _write_qwen_say_provenance(
            _say_name(args),
            language,
            "voice",
            enabled=not getattr(args, "no_provenance", False),
        )
        log.info("Output: %s", out)
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
        log.info("Qwen: designing a voice from a description (%s)", language)
        engine = VoiceDesignEngine(
            resolve_model("tts_voice_design", models_dir),
            device=args.device,
            dtype=pick_dtype(args.device, args.dtype),
        )
        with timed(log, "Designed the voice", lines=1, backend="qwen"):
            speech, sample_rate = engine.generate(args.text, args.design, language)
        if args.save_as:
            with timed(log, "Rendered the voice-print passage"):
                samples, anchor_rate = engine.generate(
                    anchor_text, args.design, language
                )
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
                provenance={
                    "backend": "voice-design",
                    "model": str(resolve_model("tts_voice_design", models_dir)),
                    "text_source": "anchor",
                    "plan_source": "manual",
                },
            )
        out = _write_say_output(
            args.output, args.save_as or "design", speech, sample_rate
        )
        release_gpu_memory()
        _write_qwen_say_provenance(
            _say_name(args),
            language,
            "design",
            enabled=not getattr(args, "no_provenance", False),
        )
        log.info("Output: %s", out)
        print(f"Wrote {out}")
        return _STOP

    # ICL needs the clip's transcript; without one the voice still clones, from
    # the speaker embedding alone. That is the same rule a saved voice follows,
    # so a voice behaves identically whether it came from the library or from
    # --clone.
    clone_text = (args.clone_text or "").strip()
    engine = TTSEngine(
        resolve_model("tts_base", models_dir),
        device=args.device,
        dtype=pick_dtype(args.device, args.dtype),
        x_vector_only=not clone_text,
    )
    log.info("Qwen: cloning from %s (%s)", args.clone, language)
    if not clone_text:
        log.warning(
            "No --clone-text: cloning in speaker-only (x-vector) mode. Similarity "
            "is lower than ICL; pass the clip's transcript for a closer match."
        )
    with timed(log, "Prepared the clone prompt"):
        engine.prepare_prompt_from_file(args.clone, clone_text)
    texts = [args.text]
    with timed(
        log, f"Synthesised {len(texts)} line(s)", lines=len(texts), backend="qwen"
    ):
        rendered = engine.synthesize(texts, language=language)
    speech = rendered[0] if rendered else None
    if speech is None:
        raise TTSError("The model returned no audio.")
    if args.save_as:
        from pipeline.media import load_audio

        # The saved voice is the donor plus its encoding. Rendering a passage and
        # storing that instead would send the voice through the codec twice: once
        # here, once when the saved voice is encoded on its next use.
        samples, sample_rate = load_audio(args.clone)
        save_voice(
            root,
            args.save_as,
            samples,
            sample_rate,
            language=language,
            source="clone",
            ref_text=clone_text,
            prompt=engine.prompt_to_data(
                fingerprint=model_fingerprint("tts_base", models_dir)
            ),
            force=args.force,
            provenance={
                "backend": "voice-clone",
                "model": str(resolve_model("tts_base", models_dir)),
                "ref_audio": str(args.clone),
                "text_source": "clone-text" if clone_text else "none",
                "fingerprint": model_fingerprint("tts_base", models_dir),
                "plan_source": "manual",
            },
        )
    out = _write_say_output(
        args.output, args.save_as or "clone", speech, engine.sample_rate
    )
    release_gpu_memory()
    _write_qwen_say_provenance(
        _say_name(args),
        language,
        "clone",
        enabled=not getattr(args, "no_provenance", False),
    )
    log.info("Output: %s", out)
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

    ``--voice`` is resolved here into ``ref_audio``/``ref_text`` (plus the voice's
    stored ``ref_prompt`` when it has one), which is the representation the
    pipeline already understands. Doing it at this boundary keeps every
    downstream stage (fingerprinting, donor selection, prompt building, fan-out
    workers) unchanged: a saved voice is simply a reference clip whose transcript
    and encoding are known.

    ``--accent-voice`` is resolved the same way, into ``accent_audio`` and
    ``accent_text``: a saved voice already carries its clip and its exact
    transcript, so the reference accent can be named in one word instead of
    repeating both on the command line.
    """
    ref_audio = args.ref_audio
    ref_text = args.ref_text
    ref_prompt: Path | None = None
    ref_prompt_seconds = 0.0
    # ``None`` means "not asked for", which lets a saved voice pick the mode that
    # fits it without overriding an explicit choice. The pipeline's own default
    # is still ``native``.
    accent_mode = args.accent_mode or options.Defaults.ACCENT_MODE
    # ``--accent-voice`` is an explicit request for reference-accent, so it must
    # stop the ``--voice`` block below from inferring a mode of its own.
    accent_requested = args.accent_mode is not None or bool(
        getattr(args, "accent_voice", None)
    )
    voice_label = getattr(args, "voice", None)
    accent_voice_label = getattr(args, "accent_voice", None)
    # Both flags read the same library, so resolve it once, and only when one of
    # them is used: a plain run must not read the environment for it.
    voices_root = (
        _voices_root(args, settings) if voice_label or accent_voice_label else None
    )
    if voice_label:
        if ref_audio is not None:
            raise ConfigError(
                "--voice cannot be combined with --ref-audio: a saved voice already "
                "is a reference clip. Pick one."
            )
        from pipeline.voices import find_prompt, load_voice

        assert voices_root is not None  # set whenever voice_label is
        saved = load_voice(voices_root, voice_label)
        ref_audio = saved.wav
        ref_text = saved.ref_text
        has_text = bool((saved.ref_text or "").strip())
        models_root = getattr(args, "models_dir", None) or (
            settings.models_dir
            if settings is not None
            else Settings.from_env().models_dir
        )
        card = find_prompt(
            saved.directory,
            "qwen",
            model_fingerprint("tts_base", models_root),
            ref_text=saved.ref_text,
        )
        if card is not None:
            ref_prompt = card
            ref_prompt_seconds = saved.seconds
        # The voice's own language is already the target language, so its speech
        # codes carry the right accent: cloning it directly is both the most
        # faithful and the cheapest path (no blend, no donor anchor). A voice in
        # another language keeps the default ``native`` mode, which borrows the
        # target accent from a generated anchor. Without a transcript there is no
        # ICL alignment at all, so only the speaker embedding is usable.
        if not accent_requested:
            if not has_text:
                accent_mode = "speaker-only"
            elif saved.language == normalize_language(args.target_language):
                accent_mode = "preserve"

    accent_audio = args.accent_audio
    accent_text = args.accent_text
    if accent_voice_label:
        # A saved voice already knows its clip and its exact transcript, so the
        # accent anchor does not have to be repeated on the command line. Both
        # routes produce the same representation (audio + text), so no stage
        # below needs to know where the anchor came from.
        if accent_audio is not None or (accent_text or "").strip():
            raise ConfigError(
                "--accent-voice cannot be combined with --accent-audio or "
                "--accent-text: the saved voice already supplies both. Pick one."
            )
        if (
            args.accent_mode is not None
            and args.accent_mode != options.AccentMode.REFERENCE_ACCENT
        ):
            raise ConfigError(
                "--accent-voice implies --accent-mode reference-accent, so it "
                f"cannot be combined with --accent-mode {args.accent_mode!r}. "
                "Pick one."
            )
        from pipeline.voices import load_voice

        assert voices_root is not None  # set whenever accent_voice_label is
        accent_saved = load_voice(voices_root, accent_voice_label)
        if not (accent_saved.ref_text or "").strip():
            raise ConfigError(
                f"Voice {accent_voice_label!r} has no transcript in "
                f"{accent_saved.manifest_path}, so its speech codes cannot be "
                "aligned. Add the transcript, or pass --accent-audio and "
                "--accent-text instead."
            )
        target_language = normalize_language(args.target_language)
        if accent_saved.language != target_language:
            raise ConfigError(
                f"--accent-voice {accent_voice_label!r} is a "
                f"{accent_saved.language} voice, so it cannot supply a native "
                f"{target_language} accent. Use a {target_language} voice, or "
                "--accent-mode preserve to keep the speaker's own accent."
            )
        accent_audio = accent_saved.wav
        accent_text = accent_saved.ref_text
        accent_mode = options.AccentMode.REFERENCE_ACCENT.value

    return JobConfig(
        video=args.video,
        audio=args.audio,
        target_language=args.target_language,
        source_language=args.source_language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        ref_prompt=ref_prompt,
        ref_prompt_seconds=ref_prompt_seconds,
        output_dir=args.output_dir,
        device=args.device,
        devices=args.devices,
        dtype=args.dtype,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        tts_batch=args.tts_batch,
        tts_backend=getattr(args, "tts_backend", "qwen"),
        prosody_plan=getattr(args, "prosody_plan", "auto"),
        chatterbox_exaggeration=getattr(args, "chatterbox_exaggeration", None),
        chatterbox_cfg_weight=getattr(args, "chatterbox_cfg_weight", None),
        chatterbox_temperature=getattr(args, "chatterbox_temperature", None),
        fish_temperature=getattr(args, "fish_temperature", None),
        fish_top_p=getattr(args, "fish_top_p", None),
        fish_quantize=getattr(args, "fish_quantize", None),
        provenance=not getattr(args, "no_provenance", False),
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
        accent_audio=accent_audio,
        accent_text=accent_text,
        expressive_accent=args.expressive_accent,
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
        subtitles_mode="off" if args.no_subtitles else args.subtitles_mode,
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


def cmd_subtitles(argv: Sequence[str]) -> int:
    """Handle the subtitles-only command."""
    args = build_subtitles_parser().parse_args(list(argv))
    settings = Settings.from_env()
    job = SubtitleJobConfig(
        video=args.video,
        audio=args.audio,
        target_language=normalize_language(args.target_language),
        source_language=args.source_language,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        device=args.device,
        output_dir=args.output_dir,
        subtitles_mode=args.subtitles_mode,
        max_chars=args.max_chars,
        force=args.force,
    )
    setup_logging(
        "DEBUG" if args.verbose else settings.log_level,
        settings.log_json,
        log_file=log_file_from(args, settings, job),
        max_bytes=args.log_max_bytes,
    )

    result = SubtitlesPipeline(job, settings).run()
    print()
    print(result.summary())
    return _STOP


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
    elif args and args[0] in {"devices", "list-devices"}:
        handler = cmd_devices
        payload = args[1:]
    elif args and args[0] == "info":
        handler = cmd_info
        payload = args[1:]
    elif args and args[0] == "subtitles":
        handler = cmd_subtitles
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
