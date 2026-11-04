"""Per-line prosody: a backend-neutral description of *how* a line is spoken.

Emotion is only one part of what makes speech sound human. The same line can be
delivered fast or slow, loud or soft, with a breath before the punchline, or
with a laugh. This module owns the one representation of those choices
(:class:`ProsodySpec`) and the rules that translate it into each backend's
native controls, so the CLI and the backends cannot drift apart.

Two paths produce a spec:

* :func:`parse_prosody_spec` reads the compact ``key=value`` grammar the CLI
  exposes (``emotion=happy, pace=fast, breath``);
* :class:`ProsodyPlanner` asks an OpenAI-compatible LLM to infer one per line
  from the text and its context. It is opt-in, because a local model call is not
  free and a user may prefer to direct the delivery themselves.

The spec is deliberately backend-neutral. CosyVoice expresses it as an instruct
sentence plus inline tokens such as ``[breath]``; Chatterbox has a single
``exaggeration`` scalar. Keeping the difference here means the rest of the code
never has to know which backend is running.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from .errors import ConfigError, DependencyError, TranslationError
from .logging_utils import timed
from .translate import chunked, extract_json_object

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "EMOTIONS",
    "PACES",
    "VOLUMES",
    "BACKEND_CAPABILITIES",
    "COSYVOICE3_SYSTEM_PROMPT",
    "STYLE_SYSTEM_PROMPT",
    "STYLE_WHOLE_SYSTEM_PROMPT",
    "FISH_TAGS",
    "FISH_INLINE_SYSTEM_PROMPT",
    "FISH_INLINE_OVERVIEW_SYSTEM_PROMPT",
    "FISH_MAX_CUES_PER_LINE",
    "FISH_PLAN_SPLIT_CHARS",
    "ProsodySpec",
    "ProsodyPlanner",
    "StylePlanner",
    "FishInlinePlanner",
    "apply_inline_tokens",
    "build_fish_inline_messages",
    "build_fish_inline_overview_messages",
    "build_prosody_messages",
    "build_prosody_overview_messages",
    "build_style_messages",
    "build_style_overview_messages",
    "chatterbox_exaggeration",
    "cosyvoice_instruct",
    "cosyvoice_mode",
    "customvoice_instruct",
    "fish_cue_tokens",
    "fish_structural_token",
    "parse_fish_inline_response",
    "parse_prosody_response",
    "parse_prosody_spec",
    "parse_style_response",
    "resolve_prosody",
    "sanitize_fish_inline_line",
    "split_fish_line",
    "strip_fish_tags",
    "trained_emotion",
    "validate_fish_inline_text",
]

#: Canonical values for the axes that map onto a backend control. ``emotion`` is
#: free-form: the known-good CosyVoice emotions are covered by
#: :data:`_EMOTION_CLAUSES`, and anything else is passed through as a description.
EMOTIONS: tuple[str, ...] = ("neutral", "happy", "sad", "angry", "excited", "calm")
PACES: tuple[str, ...] = ("normal", "slow", "fast")
VOLUMES: tuple[str, ...] = ("normal", "soft", "loud")

#: Fields a user may name in a ``--prosody`` spec.
_FIELDS = frozenset(
    {"emotion", "pace", "volume", "breath", "pause", "laughter", "exaggeration"}
)

#: Words accepted as bare flags.
_FLAGS = frozenset({"breath", "pause", "laughter"})

#: Spellings accepted for a field name.
_FIELD_ALIASES = {
    "emotion": "emotion",
    "tone": "emotion",
    "mood": "emotion",
    "pace": "pace",
    "rate": "pace",
    "speed": "pace",
    "volume": "volume",
    "loudness": "volume",
    "breath": "breath",
    "pause": "pause",
    "laughter": "laughter",
    "laugh": "laughter",
    "exaggeration": "exaggeration",
    "expressiveness": "exaggeration",
    "expression": "exaggeration",
}

#: ``key=value`` pairs, where a value may contain spaces and ends at a comma or
#: at the next ``key=``. This is what lets ``emotion=very calm, pace=slow`` and
#: ``emotion=happy pace=fast`` both work.
_PAIR_SPLIT_RE = re.compile(r"\s+(?=[A-Za-z_]+\s*=)")
_FLAG_SPLIT_RE = re.compile(r"\s+(?=(?:breath|pause|laughter)\b)", re.IGNORECASE)

_PACE_ALIASES = {
    "": "normal",
    "normal": "normal",
    "medium": "normal",
    "default": "normal",
    "slow": "slow",
    "slower": "slow",
    "fast": "fast",
    "faster": "fast",
    "quick": "fast",
    "quickly": "fast",
}

_VOLUME_ALIASES = {
    "": "normal",
    "normal": "normal",
    "medium": "normal",
    "default": "normal",
    "soft": "soft",
    "quiet": "soft",
    "quieter": "soft",
    "loud": "loud",
    "louder": "loud",
}


class ProsodySpec(BaseModel):
    """How one line should be delivered.

    Defaults describe an ordinary, unmarked delivery, so an empty spec is also
    the safe fallback when the planner is off or returns nothing useful.
    """

    model_config = ConfigDict(extra="ignore")

    #: Free-form emotion ("happy", "sad", "calm", ...). Empty or "neutral" means
    #: no emotional direction.
    emotion: str = ""
    #: One of :data:`PACES`.
    pace: str = "normal"
    #: One of :data:`VOLUMES`.
    volume: str = "normal"
    #: Ask for a breath where a speaker would naturally take one.
    breath: bool = False
    #: Ask for a short pause inside a long line.
    pause: bool = False
    #: Ask for a laugh (a non-word event, not a word).
    laughter: bool = False
    #: Explicit 0.0-1.0 expression intensity. Only Chatterbox honours it, which
    #: is the one style knob that backend exposes; ``None`` means "derive it from
    #: the emotion" (see :func:`chatterbox_exaggeration`).
    exaggeration: float | None = None

    @field_validator("emotion")
    @classmethod
    def _clean_emotion(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("exaggeration")
    @classmethod
    def _check_exaggeration(cls, value: float | None) -> float | None:
        if value is None:
            return None
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(
                f"exaggeration must be between 0.0 and 1.0 (got {number})"
            )
        return number

    @field_validator("pace")
    @classmethod
    def _check_pace(cls, value: str) -> str:
        key = str(value or "").strip().lower()
        if key not in _PACE_ALIASES:
            raise ValueError(f"unknown pace {value!r}; use normal, slow or fast")
        return _PACE_ALIASES[key]

    @field_validator("volume")
    @classmethod
    def _check_volume(cls, value: str) -> str:
        key = str(value or "").strip().lower()
        if key not in _VOLUME_ALIASES:
            raise ValueError(f"unknown volume {value!r}; use normal, soft or loud")
        return _VOLUME_ALIASES[key]

    def is_empty(self) -> bool:
        """True when the spec asks for nothing beyond a plain delivery."""
        return (
            (
                not self.emotion.strip()
                or self.emotion.strip().lower() == "neutral"
            )
            and self.pace == "normal"
            and self.volume == "normal"
            and not self.breath
            and not self.pause
            and not self.laughter
            and self.exaggeration is None
        )

    def merged_with(self, other: "ProsodySpec") -> "ProsodySpec":
        """``self`` with ``other``'s explicit choices applied on top.

        A default in ``other`` means "not asked for", so it never clears a value
        ``self`` already carried. That is what makes a manual ``--prosody`` value
        an override of an LLM plan rather than a reset of it.
        """
        data = self.model_dump()
        if other.emotion.strip():
            data["emotion"] = other.emotion
        if other.pace != "normal":
            data["pace"] = other.pace
        if other.volume != "normal":
            data["volume"] = other.volume
        for flag in ("breath", "pause", "laughter"):
            if getattr(other, flag):
                data[flag] = True
        if other.exaggeration is not None:
            data["exaggeration"] = other.exaggeration
        return ProsodySpec(**data)


def _spec_from_mapping(entry: Mapping[str, Any]) -> ProsodySpec:
    """Build a spec from loosely-typed data (an LLM reply, mostly).

    Invalid enum values fall back to their default instead of raising: a planner
    that says ``"pace": "medium-fast"`` should cost a normal delivery for that
    line, not the whole run.
    """
    def choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
        key = str(value or "").strip().lower()
        return key if key in allowed else default

    def flag(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def number(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            number_value = float(value)
        except (TypeError, ValueError):
            return None
        return number_value if 0.0 <= number_value <= 1.0 else None

    return ProsodySpec(
        emotion=str(entry.get("emotion") or "").strip(),
        pace=choice(entry.get("pace"), PACES, "normal"),
        volume=choice(entry.get("volume"), VOLUMES, "normal"),
        breath=flag(entry.get("breath")),
        pause=flag(entry.get("pause")),
        laughter=flag(entry.get("laughter")),
        exaggeration=number(entry.get("exaggeration")),
    )


def parse_prosody_spec(raw: str) -> ProsodySpec:
    """Parse one ``--prosody`` value.

    The grammar is forgiving on purpose::

        emotion=happy, pace=fast, breath
        emotion=very calm pace=slow
        happy, soft

    A bare word that is not a flag is treated as the emotion, which keeps the
    short form usable. Unknown field names raise :class:`ConfigError` so a typo
    is reported rather than silently ignored.

    Raises:
        ConfigError: On an unknown field name.
    """
    text = str(raw or "").strip()
    if not text:
        return ProsodySpec()

    # A new ``key=`` or a bare flag starts a new clause even without a comma, so
    # both the readable form ("pace=fast breath") and the explicit one work.
    text = _PAIR_SPLIT_RE.sub(", ", text)
    text = _FLAG_SPLIT_RE.sub(", ", text)
    parts = [part.strip() for part in text.split(",") if part.strip()]

    fields: dict[str, Any] = {}
    bare: list[str] = []
    for part in parts:
        if "=" in part:
            key, _, value = part.partition("=")
            name = _FIELD_ALIASES.get(key.strip().lower())
            if name is None:
                raise ConfigError(
                    f"Unknown prosody field {key.strip()!r}. Use: "
                    + ", ".join(sorted(_FIELDS))
                )
            if name in _FLAGS:
                fields[name] = True
            else:
                fields[name] = value.strip()
        elif part.lower() in _FLAGS:
            fields[part.lower()] = True
        else:
            bare.append(part)

    if "emotion" not in fields and bare:
        fields["emotion"] = " ".join(bare)

    try:
        spec = ProsodySpec(**fields)
    except ValueError as exc:
        raise ConfigError(f"Invalid prosody {raw!r}: {exc}") from None
    _LOGGER.debug("Parsed prosody %r -> %s", raw, spec.model_dump())
    return spec


def resolve_prosody(
    lines: Sequence[str], prosody: Sequence[ProsodySpec] | None
) -> list[ProsodySpec]:
    """One spec per line, broadcasting a single value like ``--emotion`` does."""
    _LOGGER.debug(
        "Resolve prosody: %d line(s), %d spec(s)",
        len(lines),
        len(prosody) if prosody is not None else 0,
    )
    if not lines:
        return []
    if prosody is None:
        return [ProsodySpec() for _ in lines]
    values = list(prosody)
    if len(values) == 1 and len(lines) > 1:
        return values * len(lines)
    if len(values) != len(lines):
        raise ConfigError(
            f"Got {len(lines)} line(s) but {len(values)} prosody value(s); give "
            "one per line, or a single one to apply everywhere."
        )
    return values


# --------------------------------------------------------------------------- #
# CosyVoice: instruct sentence + inline tokens
# --------------------------------------------------------------------------- #

#: The instruct sentences CosyVoice 3 was actually trained on (see the repo's
#: ``cosyvoice/utils/common.py``). Only these three emotions are supported;
#: anything else is silently ignored by the model, which is why the CLI maps
#: aliases onto them rather than inventing new sentences.
_EMOTION_CLAUSES = {
    "happy": "请非常开心地说一句话。",
    "sad": "请非常伤心地说一句话。",
    "angry": "请非常生气地说一句话。",
}

#: Emotions the CLI accepts that are not themselves trained instructs. Mapping
#: them onto a trained clause is what makes e.g. "excited" audible at all; an
#: unmapped word yields no emotion clause rather than an out-of-distribution
#: sentence the model would ignore.
_EMOTION_ALIASES = {
    "excited": "happy",
    "joy": "happy",
    "joyful": "happy",
    "happiness": "happy",
    "cheerful": "happy",
    "delighted": "happy",
    "glad": "happy",
    "proud": "happy",
    "thrilled": "happy",
    "upset": "sad",
    "sorrowful": "sad",
    "unhappy": "sad",
    "gloomy": "sad",
    "depressed": "sad",
    "miserable": "sad",
    "furious": "angry",
    "mad": "angry",
    "irritated": "angry",
    "annoyed": "angry",
    "frustrated": "angry",
    # No trained clause: a calm or neutral line simply carries no emotion
    # instruction, which is exactly what "no emotion" means to CosyVoice.
    "calm": "",
    "neutral": "",
    "none": "",
}

_PACE_CLAUSES = {
    "slow": "请用尽可能慢地语速说一句话。",
    "fast": "请用尽可能快地语速说一句话。",
}

_VOLUME_CLAUSES = {
    "loud": "Please say a sentence as loudly as possible.",
    "soft": "Please say a sentence in a very soft voice.",
}

#: Volume clauses for non-English text. The English wording above is from the
#: checkpoint's training list, but on a French line it pulls the LM toward English
#: and is sometimes read aloud, so non-English targets use Chinese wording.
_VOLUME_CLAUSES_ZH = {
    "loud": "请大声地说一句话。",
    "soft": "请小声地说一句话。",
}

#: Target-language clauses for CosyVoice 3. The instruct path carries no reference
#: speech tokens, so without a language hint the LM can drift to an English accent
#: on a non-English line. A clause placed last (closest to ``<|endofprompt|>``)
#: keeps the output in the target language. Chinese and English need none.
_LANGUAGE_CLAUSES = {
    "French": "请用法语说这句话。",
    "German": "请用德语说这句话。",
    "Spanish": "请用西班牙语说这句话。",
    "Italian": "请用意大利语说这句话。",
    "Portuguese": "请用葡萄牙语说这句话。",
    "Russian": "请用俄语说这句话。",
    "Japanese": "请用日语说这句话。",
    "Korean": "请用韩语说这句话。",
}

#: Prefix CosyVoice 3 expects in every prompt, and the marker it asserts on
#: before it will decode anything.
_PROMPT_PREFIX = "You are a helpful assistant."
_ENDOFPROMPT = "<|endofprompt|>"

#: The system prompt CosyVoice 3 is trained to see, used to prefix a zero-shot
#: clone's reference transcript and an inline-token request's text.
COSYVOICE3_SYSTEM_PROMPT = f"{_PROMPT_PREFIX}{_ENDOFPROMPT}"


def trained_emotion(emotion: str) -> str:
    """The trained CosyVoice emotion clause key for ``emotion``, or ``""``.

    ``happy``/``sad``/``angry`` pass through; the aliases in
    :data:`_EMOTION_ALIASES` map onto one of them; anything else is unmapped and
    yields no clause. ``calm``/``neutral``/``none`` deliberately yield ``""``.
    """
    key = str(emotion or "").strip().lower()
    if key in _EMOTION_CLAUSES:
        return key
    return _EMOTION_ALIASES.get(key, "")


def cosyvoice_instruct(
    spec: ProsodySpec,
    *,
    language: str | None = None,
) -> str:
    """Render a spec as the instruct text CosyVoice 3 conditions on.

    The checkpoint prefixes every instruction with ``You are a helpful
    assistant.`` and expects the result to end with ``<|endofprompt|>``.

    ``language`` is the canonical target language. The instruct path carries no
    reference speech tokens, so a non-English line needs a language hint to avoid
    drifting to an English accent; the English volume wording is also replaced on
    non-English lines, where it pulls the output toward English.
    """
    lang = str(language or "").strip()
    non_english = bool(lang) and lang.lower() not in {"english", "chinese"}
    volume_clauses = _VOLUME_CLAUSES_ZH if non_english else _VOLUME_CLAUSES

    clauses: list[str] = []
    emotion = trained_emotion(spec.emotion)
    if emotion:
        clauses.append(_EMOTION_CLAUSES[emotion])
    elif spec.emotion.strip() and spec.emotion.strip().lower() not in _EMOTION_ALIASES:
        _LOGGER.warning(
            "CosyVoice has no trained instruct for emotion %r; dropping it "
            "(supported: happy, sad, angry).",
            spec.emotion,
        )
    # A pace clause inside an instruct destabilises non-English speech (dropped
    # or garbled words), so pace is applied by time-stretching in the engine
    # instead. English and Chinese keep the trained pace instruct.
    if not non_english and spec.pace in _PACE_CLAUSES:
        clauses.append(_PACE_CLAUSES[spec.pace])
    if spec.volume in volume_clauses:
        clauses.append(volume_clauses[spec.volume])
    # Last, so it is the most recent instruction the LM sees.
    if lang in _LANGUAGE_CLAUSES:
        clauses.append(_LANGUAGE_CLAUSES[lang])

    body = f"{_PROMPT_PREFIX} {' '.join(clauses)}".strip()
    instruct = f"{body}{_ENDOFPROMPT}"
    _LOGGER.debug("CosyVoice instruct for %s: %s", spec.model_dump(), instruct)
    return instruct


def cosyvoice_mode(spec: ProsodySpec) -> str:
    """Which CosyVoice inference method renders ``spec``.

    The three methods are not interchangeable:

    * ``instruct2`` — emotion/speed/volume instruction. It drops the reference
      speech prompt, so it is only used when there is an instruction to justify
      that.
    * ``cross_lingual`` — inline paralinguistic tokens in the text (``[breath]``
      etc.). Used when only tokens are requested.
    * ``zero_shot`` — full clone using the reference transcript *and* its speech
      tokens, which is what preserves the reference's native accent. This is the
      default when the spec asks for nothing.
    """
    if (
        trained_emotion(spec.emotion)
        or spec.pace in _PACE_CLAUSES
        or spec.volume in _VOLUME_CLAUSES
    ):
        return "instruct2"
    if spec.breath or spec.pause or spec.laughter:
        return "cross_lingual"
    return "zero_shot"


_SENTENCE_END_RE = re.compile(r"([.!?…]+)\s*")
_TRAILING_BREATH_RE = re.compile(r"\s*\[breath\]\s*$")
#: Punctuation that already marks a pause. Inserting a comma when one is present
#: doubles it up ("gagné., Vous") without adding any audible pause.
_PAUSE_PUNCT_RE = re.compile(r"[.,;:!?…，。；：！？]")


def apply_inline_tokens(text: str, spec: ProsodySpec) -> str:
    """Insert CosyVoice's fine-grained tokens into the line itself.

    ``[breath]`` and ``[laughter]`` are part of the text stream the model reads,
    not part of the instruction, which is why they live here rather than in
    :func:`cosyvoice_instruct`.
    """
    out = str(text or "").strip()
    if not out:
        return out

    if spec.breath:
        if _SENTENCE_END_RE.search(out):
            out = _SENTENCE_END_RE.sub(r"\1 [breath] ", out).strip()
            # A breath after the final full stop only pads the end of the clip.
            out = _TRAILING_BREATH_RE.sub("", out)
        else:
            out = f"[breath] {out}"

    if spec.pause:
        words = out.split()
        # Only a long line with no punctuation already creating a pause needs
        # one inserted; otherwise a comma would land next to existing
        # punctuation ("gagné., Vous").
        if len(words) >= 8 and not _PAUSE_PUNCT_RE.search(out):
            middle = len(words) // 2
            out = " ".join(words[:middle]) + ", " + " ".join(words[middle:])

    if spec.laughter:
        out = f"{out} [laughter]"
    if out != text:
        _LOGGER.debug("Inline tokens: %r -> %r", text, out)
    return out


# --------------------------------------------------------------------------- #
# Chatterbox: a single style scalar
# --------------------------------------------------------------------------- #

_EMOTION_EXAGGERATION = {
    "neutral": 0.5,
    "calm": 0.4,
    "sad": 0.3,
    "happy": 0.8,
    "excited": 0.9,
    "angry": 0.85,
}


def chatterbox_exaggeration(spec: ProsodySpec) -> float:
    """Map a spec onto Chatterbox's ``exaggeration`` knob.

    ``exaggeration`` is Chatterbox's only style control: a single 0.0-1.0 scalar
    that amplifies the emotion already present in the reference clip. It does not
    select a discrete emotion, so pace and volume can only nudge it, and breath,
    pause and laughter have no equivalent. An explicit ``exaggeration`` in the
    spec wins over the emotion table, which is what lets a user set the exact
    value this backend exposes.
    """
    if spec.exaggeration is not None:
        value = float(spec.exaggeration)
    else:
        emotion = spec.emotion.strip().lower()
        # Keep the table's own words (it has "excited" and "calm"); only map
        # synonyms the table does not know, so e.g. "joy" reaches the happy
        # intensity instead of falling back to neutral.
        if emotion not in _EMOTION_EXAGGERATION:
            emotion = _EMOTION_ALIASES.get(emotion) or emotion
        value = _EMOTION_EXAGGERATION.get(emotion, 0.5)
        if spec.pace == "fast":
            value += 0.05
        elif spec.pace == "slow":
            value -= 0.05
        if spec.volume == "loud":
            value += 0.1
        elif spec.volume == "soft":
            value -= 0.1
    value = round(max(0.0, min(1.0, value)), 3)
    _LOGGER.debug("Chatterbox exaggeration for %s: %.3f", spec.model_dump(), value)
    return value


def customvoice_instruct(spec: ProsodySpec) -> str:
    """Render a spec as the natural-language instruction CustomVoice accepts.

    Unlike CosyVoice, ``Qwen3-TTS-CustomVoice`` has no fixed instruct strings: it
    reads a short natural-language instruction such as ``"Speak in an angry
    tone."``. Pace and volume are phrased as sentences; breath, pause and
    laughter have no equivalent on this checkpoint and are dropped.
    """
    parts: list[str] = []
    emotion = spec.emotion.strip()
    if emotion and emotion.lower() not in {"neutral", "none"}:
        article = "an" if emotion[:1].lower() in "aeiou" else "a"
        parts.append(f"Speak in {article} {emotion.lower()} tone.")
    if spec.pace == "slow":
        parts.append("Speak slowly.")
    elif spec.pace == "fast":
        parts.append("Speak quickly.")
    if spec.volume == "soft":
        parts.append("Speak softly.")
    elif spec.volume == "loud":
        parts.append("Speak loudly.")
    instruct = " ".join(parts)
    _LOGGER.debug("CustomVoice instruct for %s: %s", spec.model_dump(), instruct)
    return instruct


# --------------------------------------------------------------------------- #
# Fish Speech: natural-language tags inline in the text
# --------------------------------------------------------------------------- #

#: Documented common tags of the S2-Pro checkpoint
#: (``models/fish-speech-s2-pro/README.md``: "Common tags (15,000+ unique tags
#: supported)"). The card also states that the model accepts **free-form
#: descriptions** such as ``[whisper in small voice]``, so this tuple is the
#: reliable core the planner and the docs point at, not a closed vocabulary.
FISH_TAGS: tuple[str, ...] = (
    "pause",
    "emphasis",
    "laughing",
    "inhale",
    "chuckle",
    "tsk",
    "singing",
    "excited",
    "laughing tone",
    "interrupting",
    "chuckling",
    "excited tone",
    "volume up",
    "echo",
    "angry",
    "low volume",
    "sigh",
    "low voice",
    "whisper",
    "screaming",
    "shouting",
    "loud",
    "surprised",
    "short pause",
    "exhale",
    "delight",
    "panting",
    "audience laughter",
    "with strong accent",
    "volume down",
    "clearing throat",
    "sad",
    "moaning",
    "shocked",
)


def _fish_pause(text: str) -> str:
    """Insert ``[pause]`` after the first clause boundary, else append it."""
    match = re.search(r"[.!?;]\s+", text)
    if match:
        end = match.end()
        return f"{text[:end].rstrip()} [pause] {text[end:].lstrip()}"
    return f"{text} [pause]"


#: Lines longer than this are split into sentences before inline planning, so a
#: paragraph pasted without line breaks is not planned as one giant unit.
FISH_PLAN_SPLIT_CHARS = 240

#: Cues per line. Beyond this the delivery reads as clipped rather than acted.
FISH_MAX_CUES_PER_LINE = 3

#: A bracketed cue: ``[whisper]`` or the free-form ``[whisper in small voice]``.
_FISH_CUE_RE = re.compile(r"\[\s*([^\[\]\n]{1,80}?)\s*\]")
#: A paired span, e.g. ``<whisper>...</whisper>``. Legal content for the
#: checkpoint, but only ``[cue]`` is documented, so spans are converted.
_FISH_SPAN_RE = re.compile(r"<([A-Za-z][\w-]*)\b[^>]*>(.*?)</\1>", re.S)
#: Tokens Fish writes into its own prompt; ``validate_drama3_text`` raises when
#: caller text contains them, so they must be caught before the worker runs.
_FISH_STRUCTURAL_RE = re.compile(r"<\|[^|>]*\|>")
#: Punctuation with no space after it (``modestes.Et``). Text pasted from a
#: document often loses those spaces, and the planner re-inserts them; that is
#: typography, not a change to the words.
_PUNCT_BOUNDARY = re.compile(r"(?<=[.,;:!?\u2026])(?=\S)")


def fish_structural_token(text: str) -> str | None:
    """The first Fish prompt token in ``text``, which the checkpoint rejects."""
    match = _FISH_STRUCTURAL_RE.search(str(text or ""))
    return match.group(0) if match else None


def fish_cue_tokens(text: str) -> list[str]:
    """The bracketed cues in ``text``, normalised, in order."""
    return [
        " ".join(match.group(1).split()).lower()
        for match in _FISH_CUE_RE.finditer(str(text or ""))
    ]


def strip_fish_tags(text: str) -> str:
    """Remove inline cues and spans, for comparing tagged text to its source."""
    body = _FISH_SPAN_RE.sub(lambda match: f" {match.group(2)} ", str(text or ""))
    body = _FISH_CUE_RE.sub(" ", body)
    body = unicodedata.normalize("NFKC", body)
    return " ".join(body.split())


def _fish_words(text: str) -> list[str]:
    """``text`` as comparable words: tags stripped, case folded.

    A missing space after punctuation is a typographic glitch, not a word change:
    text pasted from a document often loses the space in "modestes. Et", and the
    planner re-inserts it. Rejecting the line for that would throw away every tag
    on it, so punctuation boundaries are split on both sides before comparing -
    while any actual difference in the words themselves is still rejected.
    """
    body = _PUNCT_BOUNDARY.sub(" ", strip_fish_tags(text))
    return body.casefold().split()


def validate_fish_inline_text(text: str) -> list[str]:
    """Advisory problems with hand-written inline tags.

    Unknown cues are not a problem: the checkpoint accepts free-form
    descriptions. The caller logs these and keeps going; the one fatal case, a
    ``<|...|>`` prompt token, is reported by :func:`fish_structural_token`.
    """
    body = str(text or "")
    problems: list[str] = []
    for cue in fish_cue_tokens(body):
        if len(cue.split()) > 8:
            problems.append(
                f"tag [{cue}] is a phrase, not a short direction; Fish may read "
                "it as text"
            )
    if not strip_fish_tags(body):
        problems.append("the line is only tags, so there is nothing to speak")
    if re.match(r"\s*\[\s*pause\s*\]", body, re.I):
        problems.append("[pause] at the start of a line adds silence, not acting")
    if re.search(r"\]\s*\[", body):
        problems.append("adjacent tags: Fish reads one direction per position")
    if len(fish_cue_tokens(body)) > FISH_MAX_CUES_PER_LINE:
        problems.append(
            f"more than {FISH_MAX_CUES_PER_LINE} tags on one line sounds clipped"
        )
    if _FISH_SPAN_RE.search(body):
        problems.append(
            "paired <tag>...</tag> spans are not the documented form; use [tag]"
        )
    return problems


def _normalise_fish_cues(text: str) -> str:
    """Tidy cue spacing: one space after a cue, none before punctuation."""
    body = _FISH_CUE_RE.sub(
        lambda match: f"[{' '.join(match.group(1).split()).lower()}]", text
    )
    body = re.sub(r"\]\s+(?=[.,;:!?\u2026])", "]", body)
    body = re.sub(r"\]\s*(?=\S)", "] ", body)
    body = re.sub(r"(\[[^\[\]]+\])(?:\s*\1)+", r"\1", body)
    return " ".join(body.split())


def sanitize_fish_inline_line(candidate: str) -> str:
    """Make a planner-written line safe for the checkpoint.

    The words are the planner's responsibility (:func:`parse_fish_inline_response`
    rejects any change to them), so this only fixes tag mechanics. Cues are kept
    as written - the checkpoint accepts free-form descriptions - except one long
    enough to be a sentence, which the model would read out. Spans become cues,
    prompt tokens are removed, a leading pause moves inward, duplicates collapse
    and the cue count is capped.
    """
    body = _FISH_SPAN_RE.sub(
        lambda match: (
            f"[{' '.join(match.group(1).split()).lower()}] {match.group(2)}"
        ),
        str(candidate or ""),
    )
    body = _FISH_STRUCTURAL_RE.sub(" ", body)

    def cue(match: re.Match[str]) -> str:
        name = " ".join(match.group(1).split()).lower()
        if len(name.split()) > 8:
            _LOGGER.warning(
                "Fish inline plan wrote a phrase as a tag [%s]; dropping it.", name
            )
            return " "
        return f"[{name}]"

    body = _FISH_CUE_RE.sub(cue, body)
    body = _normalise_fish_cues(body)
    if body.lower().startswith("[pause]"):
        body = _fish_pause(body[len("[pause]") :].strip())

    seen = 0

    def cap(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        return match.group(0) if seen <= FISH_MAX_CUES_PER_LINE else " "

    body = _FISH_CUE_RE.sub(cap, body)
    return " ".join(body.split())


def split_fish_line(
    text: str, *, max_chars: int = FISH_PLAN_SPLIT_CHARS
) -> list[str]:
    """Split an over-long line into sentence-sized planning units.

    A paragraph pasted without line breaks would otherwise reach the planner as
    one giant unit, which is where tag quality collapses. The split prefers
    sentence ends, then clause marks, then spaces; cues are never cut because
    neither pattern breaks inside ``[...]``.
    """
    body = str(text or "").strip()
    if not body:
        return []
    limit = max(1, int(max_chars))
    if len(body) <= limit:
        return [body]

    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?\u2026])\s+", body) if part.strip()
    ]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            units.append(sentence)
        else:
            units.extend(_split_fish_clauses(sentence, limit))

    packed: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}" if current else unit
        if current and len(candidate) > limit:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _split_fish_clauses(text: str, limit: int) -> list[str]:
    """Split one over-long sentence at clause marks, then at spaces."""
    parts: list[str] = []
    for clause in re.split(r"(?<=[,;:])\s+", text):
        if len(clause) <= limit:
            parts.append(clause.strip())
            continue
        current = ""
        for word in clause.split():
            candidate = f"{current} {word}" if current else word
            if current and len(candidate) > limit:
                parts.append(current)
                current = word
            else:
                current = candidate
        if current:
            parts.append(current)
    return [part for part in parts if part]


# --------------------------------------------------------------------------- #
# LLM planner
# --------------------------------------------------------------------------- #

PROSODY_SYSTEM_PROMPT = """You are a dubbing prosody director.

For each numbered line, decide how it should be spoken so that the whole passage \
sounds like a person talking rather than a machine reading it. Vary the delivery \
across lines: adjacent lines should rarely share the same emotion and pace.

Reply with ONLY a JSON object of the form:
{"prosody":[{"id":<int>,"emotion":"<word>","pace":"normal|slow|fast",\
"volume":"normal|soft|loud","breath":<bool>,"pause":<bool>,"laughter":<bool>}]}

Allowed values:
- "emotion": a short adjective. Only the words listed under the backend \
capabilities are guaranteed to be honoured; any other word is accepted but may be \
ignored, so prefer a listed word.
- "pace": normal, slow, fast.
- "volume": normal, soft, loud.
- "breath", "pause", "laughter": booleans, true only when the wording suggests one.

Rules:
1. Include every requested id exactly once, in the same order. Never invent ids.
2. Never propose a control the active backend cannot honour (see its capabilities).
3. Use "breath" or "pause" sparingly, not on every line.
4. Use "laughter" only when the line is actually a laugh, never for emphasis.
5. Never add notes, explanations, or the original text.
"""

#: What each backend can actually do with a plan. Given to the planner so it does
#: not ask Chatterbox for a breath it will drop, or Qwen for anything at all.
#: These strings are the contract between the plan and the renderers: keep them
#: in sync with ``_EMOTION_CLAUSES`` / ``_PACE_CLAUSES`` / ``_VOLUME_CLAUSES``.
BACKEND_CAPABILITIES = {
    "qwen": "No prosody controls; a plan is ignored.",
    "cosyvoice3": (
        "emotion: happy, sad or angry only; pace: slow or fast; volume: loud or "
        "soft; breath, pause and laughter as inline tokens."
    ),
    "chatterbox": (
        "a single 0.0-1.0 expression intensity ('exaggeration'); emotion words only "
        "shift that intensity, pace and volume nudge it, and breath, pause and "
        "laughter are ignored."
    ),
    "customvoice": (
        "a free-form tone/emotion instruction; pace and volume can be phrased in "
        "words; breath, pause and laughter are ignored."
    ),
}


def _backend_capability(backend: str | None) -> str:
    """The capability clause appended to a planner's system prompt."""
    if not backend:
        return ""
    capability = BACKEND_CAPABILITIES.get(
        backend,
        "unknown backend; use only emotion, pace, volume, breath, pause and "
        "laughter.",
    )
    return f"\nBackend capabilities ({backend}): {capability}"


PROSODY_OVERVIEW_SYSTEM_PROMPT = """You are a dubbing prosody director.

Read the whole passage below and describe, in two or three sentences, how it \
should be delivered overall: the dominant emotional tone, the arc, and any broad \
changes in pace, energy or volume. This note will guide the per-line plan written \
next, so do not plan individual lines here.

Reply with ONLY the note, as plain text. No JSON, no bullet points, no headings.
"""


def build_prosody_overview_messages(
    texts: Sequence[str], *, backend: str | None = None
) -> list[dict[str, str]]:
    """Build the whole-passage "director's note" request used before chunking."""
    parts = ["Passage:"]
    parts.extend(str(text) for text in texts)
    return [
        {
            "role": "system",
            "content": PROSODY_OVERVIEW_SYSTEM_PROMPT + _backend_capability(backend),
        },
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_prosody_messages(
    texts: Sequence[str],
    context: Sequence[str] = (),
    *,
    backend: str | None = None,
    overview: str | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for one prosody-planning request.

    ``backend`` is described to the model so its plan stays inside what the
    chosen engine can render; without it the model is free to propose anything.
    ``overview`` is the whole-passage note from
    :func:`build_prosody_overview_messages`, and ``chunk_index`` /
    ``chunk_count`` tell the model it is planning one piece of a longer passage.
    """
    system = PROSODY_SYSTEM_PROMPT + _backend_capability(backend)

    parts: list[str] = []
    if overview:
        parts.append(f"Overall direction for the whole passage:\n{overview}")
    if chunk_count and chunk_count > 1:
        parts.append(
            f"This is chunk {chunk_index} of {chunk_count} of a longer passage; "
            "plan only the lines below, in keeping with the overall direction."
        )
    if context:
        joined = "\n".join(f"- {line}" for line in context)
        parts.append(
            f"Earlier lines for context (already planned, do not repeat):\n{joined}"
        )
    parts.append("Lines to plan:")
    for index, text in enumerate(texts):
        parts.append(json.dumps({"id": index, "text": text}, ensure_ascii=False))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def parse_prosody_response(
    text: str, count: int, batch: Sequence[str] | None = None
) -> list[ProsodySpec]:
    """Decode a planner reply into one spec per line, defaulting on gaps.

    ``batch`` is accepted for the shared planner loop and unused: a spec is
    checked against the schema, not against the line it came from.
    """
    payload = extract_json_object(text) or {}
    entries = payload.get("prosody")
    if entries is None:
        entries = payload.get("lines")
    if not isinstance(entries, list):
        entries = []
    if not entries:
        _LOGGER.warning(
            "Prosody planner returned no usable entries; using neutral delivery "
            "for %d line(s).",
            count,
        )

    result = [ProsodySpec() for _ in range(count)]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            result[index] = _spec_from_mapping(entry)
    return result


# --------------------------------------------------------------------------- #
# Style planner: verbatim delivery instructions for CustomVoice
# --------------------------------------------------------------------------- #

STYLE_SYSTEM_PROMPT = """You are a dubbing director writing delivery instructions \
for a Qwen3-TTS CustomVoice model.

For each numbered line, write ONE short natural-language instruction (one to three \
sentences) describing how that line should be spoken: emotion, energy, pacing, \
volume, pauses, and which words to stress. Write for the actual line, and vary the \
delivery across lines so the passage sounds like a person speaking rather than a \
machine reading it.

The checkpoint reads these instructions as plain language. These examples show the \
style and level of detail expected; they are guides, not templates, and must not be \
copied verbatim:
- "Speak in an incredulous tone, but with a hint of panic beginning to creep into \
your voice."
- "Very happy."
- "用特别愤怒的语气说"
- "Speak with restrained anger. Begin quietly, then let the frustration build until \
the last sentence lands flat and exhausted."

Reply with ONLY a JSON object of the form:
{"instructions":[{"id":<int>,"text":"<instruction>"}]}

Rules:
1. Include every requested id exactly once, in the same order. Never invent ids.
2. Write each instruction in English; instructions in the line's own language are \
also accepted.
3. Keep each instruction specific to its line; do not repeat one instruction for \
every line.
4. Never add notes, explanations, or the line's own text.
5. For a line that needs no particular delivery, use an empty string.
"""

STYLE_WHOLE_SYSTEM_PROMPT = """You are a dubbing director writing ONE delivery \
instruction for a Qwen3-TTS CustomVoice model.

Read the whole passage and write a single natural-language instruction (one to \
three sentences) describing how the entire passage should be spoken: the dominant \
mood, the emotional arc, pacing, volume, pauses and which words to stress.

The checkpoint reads the instruction as plain language. These examples show the \
style and level of detail expected; they are guides, not templates:
- "Speak in an incredulous tone, but with a hint of panic beginning to creep into \
your voice."
- "Speak with restrained anger. Begin quietly, then let the frustration build until \
the last sentence lands flat and exhausted."

Reply with ONLY a JSON object with exactly one entry, using id 0:
{"instructions":[{"id":0,"text":"<instruction>"}]}

Never add notes, explanations, or the passage's own text.
"""

STYLE_OVERVIEW_SYSTEM_PROMPT = """You are a dubbing director.

Read the whole passage below and describe, in two or three sentences, how it should \
be spoken overall: the dominant mood, the emotional arc, and any pacing or energy \
changes from beginning to end. This note will guide the per-line directions written \
next, so keep it about the passage as a whole and do not plan individual lines.

Reply with ONLY the note, as plain text. No JSON, no bullet points, no headings.
"""


def _style_hints(speaker: str | None, language: str | None) -> list[str]:
    hints: list[str] = []
    if speaker:
        hints.append(f"The speaker is {speaker!r}.")
    if language:
        hints.append(f"The lines are in {language}.")
    return hints


def build_style_overview_messages(
    texts: Sequence[str],
    *,
    speaker: str | None = None,
    language: str | None = None,
) -> list[dict[str, str]]:
    """Build the whole-passage "director's note" request used before chunking."""
    parts = _style_hints(speaker, language)
    parts.append("Passage:")
    parts.extend(str(text) for text in texts)
    return [
        {"role": "system", "content": STYLE_OVERVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_style_messages(
    texts: Sequence[str],
    *,
    speaker: str | None = None,
    language: str | None = None,
    context: Sequence[str] = (),
    overview: str | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
    whole: bool = False,
) -> list[dict[str, str]]:
    """Build the chat messages for one style-instruction request.

    ``whole`` asks for a single instruction covering the entire passage; the
    per-line form numbers each line so the reply can be mapped back by id.
    """
    hints = _style_hints(speaker, language)
    base = STYLE_WHOLE_SYSTEM_PROMPT if whole else STYLE_SYSTEM_PROMPT
    system = base + (f"\n{' '.join(hints)}" if hints else "")

    parts: list[str] = []
    if overview:
        parts.append(f"Overall direction for the whole passage:\n{overview}")
    if chunk_count and chunk_count > 1:
        parts.append(
            f"This is chunk {chunk_index} of {chunk_count} of a longer passage; "
            "write directions only for the lines below, in keeping with the "
            "overall direction."
        )
    if context:
        joined = "\n".join(f"- {line}" for line in context)
        parts.append(
            f"Earlier lines for context (already planned, do not repeat):\n{joined}"
        )
    if whole:
        parts.append("Write ONE instruction for the whole passage below.")
        parts.append("Passage:")
        parts.extend(str(text) for text in texts)
    else:
        parts.append("Lines to plan:")
        for index, text in enumerate(texts):
            parts.append(json.dumps({"id": index, "text": text}, ensure_ascii=False))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def parse_style_response(
    text: str, count: int, batch: Sequence[str] | None = None
) -> list[str]:
    """Decode a style-planner reply into one instruction per line, defaulting ``""``.

    ``batch`` is accepted for the shared planner loop and unused.
    """
    payload = extract_json_object(text) or {}
    entries = payload.get("instructions")
    if entries is None:
        entries = payload.get("style")
    if not isinstance(entries, list):
        entries = []
    if not entries:
        _LOGGER.warning(
            "Style planner returned no usable entries; using an empty instruction "
            "for %d line(s).",
            count,
        )
    result = ["" for _ in range(count)]
    texts: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if text:
            texts.append(text)
        try:
            index = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            result[index] = text
    if count == 1 and not result[0] and texts:
        # A model may number from 1 (or ignore ids) despite the request. With a
        # single target there is only one sensible answer, so take the first.
        result[0] = texts[0]
    return result


class _LLMPlanner:
    """Shared OpenAI-compatible transport for the planning passes.

    Subclasses supply the message builders and the response parser. This class
    owns the client, the retry, the timed/debug logging, and the two-pass
    overview + chunked loop that keeps a long passage on one emotional arc.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.3,
        chunk_size: int = 10,
        backend: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.chunk_size = max(1, int(chunk_size))
        #: Backend whose capabilities the planner is told about, so it does not
        #: propose controls that engine will drop.
        self.backend = backend
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily construct the OpenAI client so importing this module is free."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - config dependent
                raise DependencyError(
                    "openai is not installed. Run: pip install -r requirements.txt"
                ) from exc
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s
            )
        return self._client

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        label: str,
        tag: str = "",
        json_mode: bool = True,
    ) -> str:
        """Send one completion and return its text, logging both directions.

        ``-v`` (DEBUG) logs the full message list before the call and the raw
        reply after it, which is what makes a planner run debuggable.
        """
        suffix = f" ({tag})" if tag else ""
        _LOGGER.debug(
            "%s request%s:\n%s",
            label,
            suffix,
            json.dumps(messages, ensure_ascii=False, indent=2),
        )
        with timed(_LOGGER, f"{label} plan{suffix}"):
            try:
                response = self._complete(messages, json_mode=json_mode)
            except Exception as exc:  # noqa: BLE001 - reported as a domain error
                raise TranslationError(
                    f"{label} planning failed against {self.base_url} "
                    f"(model {self.model!r}): {exc}"
                ) from exc
        reply = self._content(response)
        _LOGGER.debug("%s reply%s: %s", label, suffix, reply)
        return reply

    def _plan_with_overview(
        self,
        texts: Sequence[str],
        *,
        label: str,
        build_overview: Any,
        build_chunk: Any,
        parse: Any,
    ) -> list[Any]:
        """Plan one result per line, adding a whole-passage note when chunked.

        A passage that fits in one request is planned in one call. A longer one
        first asks for a free-text overview, then plans each chunk with that
        note and its position ("chunk k of N") in hand, so the chunks keep the
        meaning and arc of the whole text.
        """
        if not texts:
            return []
        batches = list(chunked(list(texts), self.chunk_size))
        overview: str | None = None
        if len(batches) > 1:
            reply = self._chat(
                build_overview(list(texts)),
                label=label,
                tag=f"overview, {len(texts)} line(s)",
                json_mode=False,
            )
            overview = str(reply or "").strip() or None
            _LOGGER.info("%s overview: %s", label, overview or "(empty)")

        results: list[Any] = []
        context: list[str] = []
        chunk_count = len(batches)
        for index, batch in enumerate(batches, start=1):
            messages = build_chunk(
                batch,
                context=context,
                overview=overview,
                chunk_index=index,
                chunk_count=chunk_count,
            )
            reply = self._chat(
                messages, label=label, tag=f"chunk {index}/{chunk_count}"
            )
            results.extend(parse(reply, len(batch), batch))
            context = ([*context, *batch])[-3:]
        return results

    def _complete(
        self, messages: list[dict[str, str]], *, json_mode: bool = True
    ) -> Any:
        """Send one chat completion, preferring strict JSON mode when asked.

        Some servers do not implement ``response_format``; the retry without it
        keeps the planner working against them. The parse step still tolerates
        prose, so this is a best-effort improvement rather than a requirement.
        Free-text calls (the overview) skip JSON mode entirely.
        """
        if not json_mode:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception:  # noqa: BLE001 - fall back to a plain completion
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

    @staticmethod
    def _content(response: Any) -> str:
        """Extract the assistant text from an OpenAI-style response object."""
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return ""
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content or "")


class ProsodyPlanner(_LLMPlanner):
    """Per-line prosody planning through an OpenAI-compatible endpoint."""

    def plan(self, texts: Sequence[str]) -> list[ProsodySpec]:
        """Plan one spec per line, batching to respect context windows."""
        if not texts:
            return []
        _LOGGER.info(
            "Planning prosody for %d line(s) with %s (backend=%s)",
            len(texts),
            self.model,
            self.backend or "unspecified",
        )
        specs = self._plan_with_overview(
            list(texts),
            label="Prosody",
            build_overview=lambda batch: build_prosody_overview_messages(
                batch, backend=self.backend
            ),
            build_chunk=lambda batch, **kwargs: build_prosody_messages(
                batch, backend=self.backend, **kwargs
            ),
            parse=parse_prosody_response,
        )
        _LOGGER.debug(
            "Prosody plan: %s", [spec.model_dump() for spec in specs]
        )
        _LOGGER.info("Prosody plan ready for %d line(s)", len(specs))
        return specs


class StylePlanner(_LLMPlanner):
    """Verbatim delivery-instruction planning for the CustomVoice checkpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.7,
        chunk_size: int = 10,
        backend: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            temperature=temperature,
            chunk_size=chunk_size,
            backend=backend,
            client=client,
        )

    def plan_lines(
        self,
        texts: Sequence[str],
        *,
        speaker: str | None = None,
        language: str | None = None,
    ) -> list[str]:
        """Write one instruction per line, with an overview pass when chunked."""
        if not texts:
            return []
        _LOGGER.info(
            "Planning style instructions for %d line(s) with %s",
            len(texts),
            self.model,
        )
        instructs = self._plan_with_overview(
            list(texts),
            label="Style",
            build_overview=lambda batch: build_style_overview_messages(
                batch, speaker=speaker, language=language
            ),
            build_chunk=lambda batch, **kwargs: build_style_messages(
                batch, speaker=speaker, language=language, **kwargs
            ),
            parse=parse_style_response,
        )
        _LOGGER.info("Style plan ready for %d line(s)", len(instructs))
        return instructs

    def plan_whole(
        self,
        texts: Sequence[str],
        *,
        speaker: str | None = None,
        language: str | None = None,
    ) -> str:
        """Write a single instruction covering the entire text."""
        if not texts:
            return ""
        messages = build_style_messages(
            list(texts), speaker=speaker, language=language, whole=True
        )
        reply = self._chat(messages, label="Style", tag="whole passage")
        instruction = parse_style_response(reply, 1)[0]
        _LOGGER.info("Style plan (whole): %r", instruction)
        return instruction


# --------------------------------------------------------------------------- #
# Fish inline planner: the model writes the tags into the text itself
# --------------------------------------------------------------------------- #

FISH_INLINE_SYSTEM_PROMPT = """You are a dubbing director preparing lines for \
Fish Speech, a model that reads delivery from tags placed inline in the text.

Fish has no separate style channel: everything is a bracketed tag inside the \
line, and the model accepts FREE-FORM descriptions, not just a fixed list. \
[whisper in small voice], [professional broadcast tone], [pitch up] and \
[laughing tone] are all valid; the model was trained on 15,000+ tags.

Your job is to make each line sound like a person saying it, not a model reading \
it. Use tags to shape how the words come out and how the speaker reacts to what \
is being said: relief after a confession, disbelief at a betrayal, a joke that \
lands badly, a warning that is trying not to frighten anyone.

For each numbered line, return the SAME line with tags inserted where the acting \
belongs. Copy every word exactly: do not translate, paraphrase, reorder, split, \
merge, shorten or extend the text, and keep its punctuation. Change only tag \
placement.

Common tags (a reliable core; any other short description also works):
[pause] [emphasis] [laughing] [inhale] [chuckle] [tsk] [singing] [excited] \
[laughing tone] [interrupting] [chuckling] [excited tone] [volume up] [echo] \
[angry] [low volume] [sigh] [low voice] [whisper] [screaming] [shouting] [loud] \
[surprised] [short pause] [exhale] [delight] [panting] [audience laughter] \
[with strong accent] [volume down] [clearing throat] [sad] [moaning] [shocked]

How to use them:
1. A tag applies to the words after it, so put it before the words it describes. \
A tag at the very start covers the whole line.
2. A line may turn mid-way. Put a second tag where the turn happens; that turn is \
why this format beats one style for a whole clip.
3. Reach for the tag that fits the moment: [sigh] for resignation, [chuckle] for \
nervous amusement, [whisper] for secrecy or fear, [delight] for good news, \
[surprised] for a reversal, [pause] for a beat.
4. If none of the common tags fits, describe the delivery in a few words: \
[impatiently], [reading aloud], [trying not to cry], [whisper in small voice].
5. Use [pause] and [short pause] between clauses, never at the start or the end \
of a line, and at most one beat per short line.
6. At most three tags per line, never nested, never two side by side.
7. Write tags in English even when the line is in another language. Never \
translate the line itself.
8. A line that needs nothing stays exactly as it is.

Examples: they show the level of detail expected, and must not be copied.
- "I can't believe we actually did it!" -> \
"[excited] I can't believe we actually did it!"
- "Wait... did you hear that?" -> "[whisper] Wait... [pause] did you hear that?"
- "I really thought things would be different." -> \
"[sigh] I really thought things would be different."
- "I thought things would be different... but here we are." -> \
"[sad] I thought things would be different... [angry] but here we are."
- "You're not supposed to know that." -> \
"[whisper in small voice] You're not supposed to know that."
- "Fine. Have it your way." -> "[sigh] Fine. [pause] Have it your way."
- "We won. We actually won." -> "[delight] We won. [chuckle] We actually won."

Never do this:
- Writing a whole sentence as a tag: it is read out as text.
- Dropping, adding or translating words to make a line more emotional.
- Starting or ending a line with [pause].
- Stacking tags: "[excited] [angry] ...".
- Writing <|im_end|>, <|voice|> or any <|...|> token.

Reply with ONLY a JSON object of the form:
{"lines":[{"id":<int>,"text":"<the line, with tags>"}]}

Rules:
1. Include every requested id exactly once, in the same order. Never invent ids.
2. Every word of the original line must appear, in order and unchanged.
3. Tags are short bracketed directions; never write <|...|> tokens.
"""

FISH_INLINE_OVERVIEW_SYSTEM_PROMPT = """You are a dubbing director.

Read the whole passage below and describe, in two or three sentences, how it should \
be performed overall: the dominant emotion, the arc, and how the speaker's \
reaction changes from beginning to end (energy, pauses, laughter, hesitation). \
This note guides the per-line tagging written next, so keep it about the passage \
as a whole and do not tag individual lines.

Reply with ONLY the note, as plain text. No JSON, no bullet points, no headings.
"""


def build_fish_inline_overview_messages(
    texts: Sequence[str],
    *,
    speaker: str | None = None,
    language: str | None = None,
) -> list[dict[str, str]]:
    """Build the whole-passage note request used before chunked tagging."""
    parts = _style_hints(speaker, language)
    parts.append("Passage:")
    parts.extend(str(text) for text in texts)
    return [
        {"role": "system", "content": FISH_INLINE_OVERVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_fish_inline_messages(
    texts: Sequence[str],
    *,
    speaker: str | None = None,
    language: str | None = None,
    context: Sequence[str] = (),
    overview: str | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for one inline-tagging request.

    ``context`` carries the *original* earlier lines, untagged, so the model
    follows the passage's arc without copying the tags it already wrote.
    """
    hints = _style_hints(speaker, language)
    system = FISH_INLINE_SYSTEM_PROMPT + (f"\n{' '.join(hints)}" if hints else "")

    parts: list[str] = []
    if overview:
        parts.append(f"Overall direction for the whole passage:\n{overview}")
    if chunk_count and chunk_count > 1:
        parts.append(
            f"This is chunk {chunk_index} of {chunk_count} of a longer passage; "
            "tag only the lines below, in keeping with the overall direction."
        )
    if context:
        joined = "\n".join(f"- {line}" for line in context)
        parts.append(
            f"Earlier lines for context (already directed, do not repeat):\n{joined}"
        )
    parts.append("Lines to direct:")
    for index, text in enumerate(texts):
        parts.append(json.dumps({"id": index, "text": text}, ensure_ascii=False))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def parse_fish_inline_response(
    reply: str, originals: Sequence[str]
) -> list[str]:
    """Decode a Fish inline reply, keeping only lines whose words survive.

    A planner that paraphrases, translates or drops a word is the one failure
    this backend cannot tolerate: the audio would stop matching the subtitles.
    Such a line falls back to its original text, so a bad plan degrades to "no
    tags" rather than to wrong speech.
    """
    sources = [str(text or "") for text in originals]
    count = len(sources)
    payload = extract_json_object(reply) or {}
    entries = payload.get("lines")
    if entries is None:
        entries = payload.get("text")
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        entries = []
    if not entries:
        _LOGGER.warning(
            "Fish inline planner returned no usable entries; keeping %d line(s) "
            "unchanged.",
            count,
        )
        return sources

    candidates: list[str | None] = [None] * count
    ordered: list[str] = []
    for position, entry in enumerate(entries):
        if isinstance(entry, dict):
            text = str(entry.get("text") or entry.get("line") or "").strip()
            try:
                index = int(entry.get("id"))
            except (TypeError, ValueError):
                index = position
        elif isinstance(entry, str):
            text = entry.strip()
            index = position
        else:
            continue
        if not text:
            continue
        ordered.append(text)
        if 0 <= index < count and candidates[index] is None:
            candidates[index] = text
    # A model that numbered from 1, or ignored ids, is aligned by position.
    for index in range(count):
        if candidates[index] is None and index < len(ordered):
            candidates[index] = ordered[index]

    result: list[str] = []
    fallbacks = 0
    for source, candidate in zip(sources, candidates):
        if candidate is None:
            result.append(source)
            fallbacks += 1
            continue
        cleaned = sanitize_fish_inline_line(candidate)
        if _fish_words(cleaned) != _fish_words(source):
            _LOGGER.warning(
                "Fish inline plan changed the words of a line; keeping the "
                "original. planned=%r original=%r",
                cleaned,
                source,
            )
            result.append(source)
            fallbacks += 1
            continue
        result.append(cleaned)
    if fallbacks:
        _LOGGER.info(
            "Fish inline plan: %d/%d line(s) kept verbatim", fallbacks, count
        )
    return result


class FishInlinePlanner(_LLMPlanner):
    """Inline-tag planning for Fish Speech: the model rewrites the text.

    Fish has no side channel, so unlike the other planners this one returns
    text rather than a spec. Every line is checked word for word against its
    source (:func:`parse_fish_inline_response`), so a paraphrase cannot be
    spoken. An over-long line is split into sentences for planning only and
    rejoined afterwards, keeping one clip per ``--text`` line.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.7,
        chunk_size: int = 10,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            temperature=temperature,
            chunk_size=chunk_size,
            backend="fish",
            client=client,
        )

    def plan_lines(
        self,
        texts: Sequence[str],
        *,
        speaker: str | None = None,
        language: str | None = None,
    ) -> list[str]:
        """Direct every line, splitting an over-long one for planning only."""
        if not texts:
            return []
        originals = [str(text) for text in texts]
        units: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in originals:
            start = len(units)
            units.extend(split_fish_line(text))
            spans.append((start, len(units)))
        _LOGGER.info(
            "Planning Fish inline tags for %d line(s) as %d unit(s) with %s",
            len(originals),
            len(units),
            self.model,
        )
        planned = self._plan_with_overview(
            units,
            label="Fish",
            build_overview=lambda batch: build_fish_inline_overview_messages(
                batch, speaker=speaker, language=language
            ),
            build_chunk=lambda batch, **kwargs: build_fish_inline_messages(
                batch, speaker=speaker, language=language, **kwargs
            ),
            parse=lambda reply, count, batch: parse_fish_inline_response(
                reply, batch
            ),
        )
        _LOGGER.info("Fish inline plan ready for %d line(s)", len(planned))
        return [" ".join(planned[start:stop]) for start, stop in spans]
