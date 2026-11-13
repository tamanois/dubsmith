"""Accepted values for every option the pipeline, CLI and HTTP API share.

This module is the one place that states which spellings an option understands.
The CLI hands the tuples to ``argparse`` as ``choices``, the pydantic models
validate against the alias tables built from them, and the API documents the
closed sets in its schema. Renaming a value here renames it everywhere: nothing
else is allowed to hardcode the literal.

Each vocabulary is an :class:`enum.Enum` (so a closed set has a name and a type)
paired with a tuple of its values in the order the CLI lists them, and an alias
table mapping every accepted spelling to its canonical value. The canonical
spellings are generated from the tuple, so they are never retyped.

Only stdlib is imported: ``pipeline.capabilities`` and the API process import
this module at startup and must not pull in torch.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

__all__ = [
    "ACCENT_BLENDS",
    "ACCENT_BLEND_ALIASES",
    "ACCENT_MATCHES",
    "ACCENT_MATCH_ALIASES",
    "ACCENT_MODES",
    "ACCENT_MODE_ALIASES",
    "DIARIZERS",
    "DIARIZER_ALIASES",
    "DTYPES",
    "DTYPE_ALIASES",
    "EXPRESSIVE_ACCENTS",
    "EXPRESSIVE_ACCENT_ALIASES",
    "FISH_QUANTIZES",
    "FISH_QUANTIZE_ALIASES",
    "FISH_TAG_MODES",
    "FISH_TAG_MODE_ALIASES",
    "LENGTH_BUDGETS",
    "LENGTH_BUDGET_ALIASES",
    "LOG_LEVELS",
    "LOUDNESS_POLICIES",
    "LOUDNESS_POLICY_ALIASES",
    "OVERLAP_MODES",
    "OVERLAP_MODE_ALIASES",
    "OVERLAP_SEPARATORS",
    "OVERLAP_SEPARATOR_ALIASES",
    "PROSODY_PLANS",
    "PROSODY_PLAN_ALIASES",
    "ROOMS",
    "ROOM_ALIASES",
    "SAY_BACKENDS",
    "SAY_PROSODY_PLANS",
    "SEPARATORS",
    "SEPARATOR_ALIASES",
    "STYLE_PLANS",
    "STYLE_PLAN_SCOPES",
    "STRETCHERS",
    "STRETCHER_ALIASES",
    "SUBTITLES_MODES",
    "SUBTITLES_MODE_ALIASES",
    "TTS_BACKENDS",
    "TTS_BACKEND_ALIASES",
    "AccentBlend",
    "AccentMatch",
    "AccentMode",
    "Defaults",
    "Diarizer",
    "Dtype",
    "ExpressiveAccent",
    "FishQuantize",
    "FishTagMode",
    "LengthBudget",
    "LoudnessPolicy",
    "OverlapMode",
    "OverlapSeparator",
    "ProsodyPlan",
    "Room",
    "SayBackend",
    "SayProsodyPlan",
    "Separator",
    "Stretcher",
    "StylePlan",
    "StylePlanScope",
    "SubtitlesMode",
    "TtsBackend",
]

# Words that mean "let the code pick". Defined once because several options share
# the spelling and a typo in one place would silently disable a default.
_AUTO = "auto"
_NONE = "none"
_OFF = "off"


def _values(enum: type[Enum]) -> tuple[str, ...]:
    """The canonical values of ``enum``, in declaration order."""
    return tuple(str(member.value) for member in enum)


def _aliases(
    values: tuple[str, ...],
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Alias table whose identity entries are derived from ``values``.

    Only the genuinely different spellings are written out; a canonical spelling
    is never retyped, so renaming a value cannot leave a stale identity entry
    behind that quietly maps the old name to itself.
    """
    table = {value: value for value in values}
    if extra:
        table.update(extra)
    return table


# --------------------------------------------------------------------------- #
# Synthesis backends
# --------------------------------------------------------------------------- #


class SayBackend(str, Enum):
    """Backends ``dub.py say`` can synthesise with."""

    QWEN = "qwen"
    CUSTOMVOICE = "customvoice"
    COSYVOICE3 = "cosyvoice3"
    CHATTERBOX = "chatterbox"
    FISH = "fish"


class TtsBackend(str, Enum):
    """Backends a dub can synthesise its lines with.

    CustomVoice is absent on purpose: it speaks a preset speaker, so it cannot
    carry a source speaker's timbre into another language.
    """

    QWEN = "qwen"
    COSYVOICE3 = "cosyvoice3"
    CHATTERBOX = "chatterbox"
    FISH = "fish"


SAY_BACKENDS: tuple[str, ...] = _values(SayBackend)
TTS_BACKENDS: tuple[str, ...] = _values(TtsBackend)

#: Accepted spellings for a dub backend. ``customvoice`` is not among them: it
#: cannot carry a source speaker's timbre into another language.
TTS_BACKEND_ALIASES: dict[str, str] = _aliases(
    TTS_BACKENDS,
    {"base": "qwen", "cosyvoice": "cosyvoice3", "fish-speech": "fish"},
)

#: Backends that render a :class:`~pipeline.prosody.ProsodySpec` (or inline tags)
#: instead of the plain Qwen Base clone.
EXPRESSIVE_BACKENDS: tuple[str, ...] = tuple(
    backend for backend in TTS_BACKENDS if backend != TtsBackend.QWEN.value
)


# --------------------------------------------------------------------------- #
# Hardware and runtime
# --------------------------------------------------------------------------- #


class Dtype(str, Enum):
    """Numeric precision to run a checkpoint in."""

    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


DTYPES: tuple[str, ...] = _values(Dtype)
DTYPE_ALIASES: dict[str, str] = _aliases(
    DTYPES,
    {
        "bfloat16": "bf16",
        "float16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "full": "fp32",
    },
)

#: Stretch backends for duration fitting. ``none`` disables stretching, so a line
#: can only overrun or be trimmed.
class Stretcher(str, Enum):
    """Time-stretching backend."""

    AUTO = "auto"
    RUBBERBAND = "rubberband"
    SOX = "sox"
    LIBROSA = "librosa"
    NONE = "none"


STRETCHERS: tuple[str, ...] = _values(Stretcher)
STRETCHER_ALIASES: dict[str, str] = _aliases(
    STRETCHERS,
    {"identity": "none", "off": "none", "rubber band": "rubberband"},
)


class Separator(str, Enum):
    """Background (music and ambience) separator."""

    AUTO = "auto"
    DEMUCS = "demucs"
    NONE = "none"


SEPARATORS: tuple[str, ...] = _values(Separator)
SEPARATOR_ALIASES: dict[str, str] = _aliases(
    SEPARATORS,
    {"identity": "none", "off": "none"},
)


class OverlapSeparator(str, Enum):
    """Backend used to split overlapping speech."""

    AUTO = "auto"
    SEPFORMER = "sepformer"
    NONE = "none"


OVERLAP_SEPARATORS: tuple[str, ...] = _values(OverlapSeparator)
OVERLAP_SEPARATOR_ALIASES: dict[str, str] = _aliases(
    OVERLAP_SEPARATORS,
    {"speechbrain": "sepformer", "off": "none", "identity": "none"},
)


# --------------------------------------------------------------------------- #
# Prosody, style and language generation
# --------------------------------------------------------------------------- #


class ProsodyPlan(str, Enum):
    """Who decides per-line emotion and pace."""

    OFF = "off"
    LLM = "llm"
    AUTO = "auto"


PROSODY_PLANS: tuple[str, ...] = _values(ProsodyPlan)
PROSODY_PLAN_ALIASES: dict[str, str] = _aliases(PROSODY_PLANS, {"none": "off"})


class SayProsodyPlan(str, Enum):
    """``say`` decides per-line prosody explicitly, so ``auto`` is absent."""

    OFF = "off"
    LLM = "llm"


SAY_PROSODY_PLANS: tuple[str, ...] = _values(SayProsodyPlan)


class StylePlan(str, Enum):
    """Whether CustomVoice style instructions are planned by an LLM."""

    OFF = "off"
    LLM = "llm"


STYLE_PLANS: tuple[str, ...] = _values(StylePlan)


class StylePlanScope(str, Enum):
    """How many style instructions a style plan may produce."""

    PER_LINE = "per-line"
    WHOLE = "whole"


STYLE_PLAN_SCOPES: tuple[str, ...] = _values(StylePlanScope)


class FishQuantize(str, Enum):
    """Fish checkpoint quantization mode."""

    INT8 = "int8"
    OFF = "off"


FISH_QUANTIZES: tuple[str, ...] = _values(FishQuantize)
FISH_QUANTIZE_ALIASES: dict[str, str] = _aliases(
    FISH_QUANTIZES,
    {"bf16": "off", "none": "off", "float16": "off"},
)


class FishTagMode(str, Enum):
    """Whether inline ``[tags]`` are sent as written or rewritten by the LLM."""

    KEEP = "keep"
    LLM = "llm"


FISH_TAG_MODES: tuple[str, ...] = _values(FishTagMode)
FISH_TAG_MODE_ALIASES: dict[str, str] = _aliases(FISH_TAG_MODES, {"off": "keep"})


# --------------------------------------------------------------------------- #
# Voice and accent
# --------------------------------------------------------------------------- #


class AccentMode(str, Enum):
    """Where the dubbed speech takes its pronunciation from."""

    NATIVE = "native"
    PRESERVE = "preserve"
    SPEAKER_ONLY = "speaker-only"
    REFERENCE_ACCENT = "reference-accent"


ACCENT_MODES: tuple[str, ...] = _values(AccentMode)
ACCENT_MODE_ALIASES: dict[str, str] = _aliases(
    ACCENT_MODES,
    {
        "auto": "native",
        "generated": "native",
        "anchor": "native",
        "icl": "preserve",
        "source": "preserve",
        "xvector": "speaker-only",
        "x-vector": "speaker-only",
        "speaker": "speaker-only",
        "ref-accent": "reference-accent",
        "hybrid": "reference-accent",
    },
)


class AccentMatch(str, Enum):
    """How the generated native anchor follows the source voice."""

    OFF = "off"
    RATE = "rate"
    GENDER = "gender"


ACCENT_MATCHES: tuple[str, ...] = _values(AccentMatch)
ACCENT_MATCH_ALIASES: dict[str, str] = _aliases(
    ACCENT_MATCHES,
    {
        "none": "off",
        "false": "off",
        "speed": "rate",
        "pace": "rate",
        "voice": "gender",
        "gender-rate": "gender",
        "gender,rate": "gender",
        "rate,gender": "gender",
    },
)


class AccentBlend(str, Enum):
    """How the reference-accent blend is built."""

    TWO_STAGE = "two-stage"
    SINGLE_STAGE = "single-stage"


ACCENT_BLENDS: tuple[str, ...] = _values(AccentBlend)
ACCENT_BLEND_ALIASES: dict[str, str] = _aliases(
    ACCENT_BLENDS,
    {
        "two": "two-stage",
        "blend": "two-stage",
        "single": "single-stage",
        "direct": "single-stage",
    },
)


class ExpressiveAccent(str, Enum):
    """Which clip an expressive backend clones for a dub."""

    DONOR = "donor"
    BLEND = "blend"


EXPRESSIVE_ACCENTS: tuple[str, ...] = _values(ExpressiveAccent)
EXPRESSIVE_ACCENT_ALIASES: dict[str, str] = _aliases(
    EXPRESSIVE_ACCENTS,
    {
        "preserve": "donor",
        "source": "donor",
        "native": "blend",
        "anchor": "blend",
    },
)


# --------------------------------------------------------------------------- #
# Audio bed and timing
# --------------------------------------------------------------------------- #


class LoudnessPolicy(str, Enum):
    """How the dub's level is chosen."""

    MATCH_SOURCE = "match-source"
    LUFS = "lufs"


LOUDNESS_POLICIES: tuple[str, ...] = _values(LoudnessPolicy)
LOUDNESS_POLICY_ALIASES: dict[str, str] = _aliases(
    LOUDNESS_POLICIES,
    {
        "match": "match-source",
        "source": "match-source",
        "fixed": "lufs",
        "absolute": "lufs",
    },
)


class Room(str, Enum):
    """Whether the dub is placed in the source's acoustic space."""

    OFF = "off"
    MATCH = "match"


ROOMS: tuple[str, ...] = _values(Room)
ROOM_ALIASES: dict[str, str] = _aliases(
    ROOMS,
    {
        "none": "off",
        "dry": "off",
        "false": "off",
        "on": "match",
        "true": "match",
        "reverb": "match",
    },
)


class LengthBudget(str, Enum):
    """How the translator is told to size each line."""

    OFF = "off"
    WORDS = "words"


LENGTH_BUDGETS: tuple[str, ...] = _values(LengthBudget)
LENGTH_BUDGET_ALIASES: dict[str, str] = _aliases(
    LENGTH_BUDGETS,
    {"none": "off", "seconds": "off", "word": "words"},
)


# --------------------------------------------------------------------------- #
# Speakers and subtitles
# --------------------------------------------------------------------------- #


class Diarizer(str, Enum):
    """Whether speakers are separated or the input is treated as one voice."""

    AUTO = "auto"
    NONE = "none"


DIARIZERS: tuple[str, ...] = _values(Diarizer)
DIARIZER_ALIASES: dict[str, str] = _aliases(
    DIARIZERS,
    {
        "on": "auto",
        "true": "auto",
        "embedding": "auto",
        "off": "none",
        "false": "none",
        "single": "none",
    },
)


class OverlapMode(str, Enum):
    """What happens to a segment that carries two voices at once."""

    DOMINANT = "dominant"
    SEPARATE = "separate"


OVERLAP_MODES: tuple[str, ...] = _values(OverlapMode)
OVERLAP_MODE_ALIASES: dict[str, str] = _aliases(
    OVERLAP_MODES,
    {"keep": "dominant", "off": "dominant", "split": "separate", "on": "separate"},
)


class SubtitlesMode(str, Enum):
    """How target subtitles reach the video."""

    SOFT = "soft"
    BURN = "burn"
    OFF = "off"


SUBTITLES_MODES: tuple[str, ...] = _values(SubtitlesMode)
SUBTITLES_MODE_ALIASES: dict[str, str] = _aliases(
    SUBTITLES_MODES,
    {
        "mux": "soft",
        "track": "soft",
        "embed": "soft",
        "burn-in": "burn",
        "burnin": "burn",
        "hard": "burn",
        "hardcode": "burn",
        "none": "off",
        "skip": "off",
    },
)


#: Log levels the CLI and the settings form accept, quietest last.
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


class Defaults:
    """The default each option takes when the caller does not supply one.

    Kept together so the CLI, the pydantic models, the API schemas and the web
    forms cannot drift about what "unset" means. They are values of the
    vocabularies above, never bare literals.
    """

    DEVICE = _AUTO
    DTYPE = Dtype.AUTO.value
    STRETCHER = Stretcher.AUTO.value
    LOG_LEVEL = "INFO"
    WHISPER_MODEL = "large-v3"

    SAY_BACKEND = SayBackend.QWEN.value
    TTS_BACKEND = TtsBackend.QWEN.value
    PROSODY_PLAN = ProsodyPlan.AUTO.value
    SAY_PROSODY_PLAN = SayProsodyPlan.OFF.value
    STYLE_PLAN = StylePlan.OFF.value
    STYLE_PLAN_SCOPE = StylePlanScope.PER_LINE.value
    FISH_QUANTIZE = FishQuantize.INT8.value
    FISH_TAG_MODE = FishTagMode.KEEP.value

    ACCENT_MODE = AccentMode.NATIVE.value
    ACCENT_MATCH = AccentMatch.OFF.value
    ACCENT_BLEND = AccentBlend.TWO_STAGE.value
    EXPRESSIVE_ACCENT = ExpressiveAccent.DONOR.value

    LOUDNESS_POLICY = LoudnessPolicy.MATCH_SOURCE.value
    ROOM = Room.OFF.value
    SEPARATOR = Separator.AUTO.value
    LENGTH_BUDGET = LengthBudget.WORDS.value

    DIARIZER = Diarizer.AUTO.value
    OVERLAP_MODE = OverlapMode.DOMINANT.value
    OVERLAP_SEPARATOR = OverlapSeparator.AUTO.value
    SUBTITLES_MODE = SubtitlesMode.SOFT.value
