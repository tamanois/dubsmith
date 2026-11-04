"""Translation via an OpenAI-compatible chat endpoint.

The transport is deliberately thin: any server exposing ``/v1/chat/completions``
works (local llama.cpp/``llama-server``, vLLM, TGI, or a cloud provider). Prompt
construction, response parsing, and validation are separated from the HTTP call
so they can be unit tested and reused by other backends.

Dubbing-specific concerns handled here:
  * length budget per line, so the dub fits the original time slot;
  * batched, context-aware translation rather than sentence-by-sentence;
  * strict id round-tripping with one repair attempt.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Protocol, Sequence

from pydantic import BaseModel

from .config import language_name_for_code
from .errors import DependencyError, TranslationError
from .logging_utils import timed

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "TranslationRequest",
    "ShortenRequest",
    "Translator",
    "OpenAICompatTranslator",
    "build_messages",
    "build_subtitle_messages",
    "build_shorten_messages",
    "extract_json_object",
    "chunked",
    "LENGTH_TOLERANCE",
]

#: How much longer/shorter a dub line may be than the original, used **only** when
#: no word budget is available. It deliberately does not mirror the timing stage's
#: floor: that floor is a fraction of the line's own *spoken* length, whereas this
#: compares the translation against the source - which a source speaker talking at
#: 4.2 words/s makes unachievable for a French voice delivering 3.65.
LENGTH_TOLERANCE = 0.2

_CONTEXT_LINES = 3

SYSTEM_PROMPT = """You are a professional audiovisual translator and dubbing adaptor.

Translate each numbered line into {target}. {source_clause}

Rules:
1. Reply with ONLY a JSON object of the form \
{{"translations":[{{"id":<int>,"text":"<translation>"}}]}}.
2. Include every requested id exactly once, in the same order. Never invent ids.
3. {length_rule}
4. Preserve register and tone, and use natural spoken {target} rather than \
written style. Match the formality implied by the source.
5. Keep proper nouns, numbers, units, and acronyms unchanged.
6. Never add notes, explanations, transliterations, or the untranslated source.
"""

#: Rule 3 when no word budget is available: a proportional hint measured against
#: the original's own length. Only ever right when the source speaker and the
#: target voice speak at comparable rates.
LENGTH_RULE_PROPORTIONAL = (
    "Keep each translation close in spoken length to its original (within roughly "
    "{percent}%). Say the same thing in fewer words rather than translating "
    "literally."
)

#: Rule 3 when each line carries ``max_words``. The ceiling *replaces* the
#: proportional hint rather than joining it, because the two disagree - the ceiling
#: comes from how fast this voice speaks in this slot, the hint from the source's
#: length - and the model resolves that argument in favour of whichever it is told
#: first. Measured on a real job the translations came out 1.5-1.8x the ceiling.
LENGTH_RULE_BUDGETED = (
    'Respect each line\'s "max_words": the most words that fit its slot when '
    "spoken, and a hard ceiling rather than a target. It matters more than "
    "matching the original's length: if the meaning needs more words, drop the "
    "least important detail instead of exceeding it, and never pad a shorter line "
    "to reach it."
)


class TranslationRequest(BaseModel):
    """One line to translate, with the slot it must fit."""

    id: int
    text: str
    duration: float = 0.0
    #: Longest the translation may be, when a speaking rate is known.
    max_words: int | None = None


class ShortenRequest(BaseModel):
    """A line whose translation turned out too long to fit its slot."""

    id: int
    #: The translation that was tried and proved too long.
    text: str
    #: The source line, given so the rewrite cannot drop its meaning.
    source: str = ""
    #: How long the previous translation took to speak.
    spoken_seconds: float = 0.0
    #: How long it actually had.
    available_seconds: float = 0.0
    #: Ceiling for the rewrite, derived from the measured speaking rate.
    max_words: int = 0


class Translator(Protocol):
    """Backend-agnostic translation interface."""

    def translate(
        self,
        requests: Sequence[TranslationRequest],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        """Translate requests, returning ``{id: translated_text}``."""
        ...

    def shorten(
        self,
        requests: Sequence[ShortenRequest],
        *,
        target_language: str,
    ) -> dict[int, str]:
        """Rewrite lines that proved too long, returning ``{id: text}``."""
        ...


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield successive slices of at most ``size`` items."""
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _render_line(request: TranslationRequest) -> str:
    payload: dict[str, Any] = {
        "id": request.id,
        "seconds": round(request.duration, 2),
    }
    if request.max_words:
        payload["max_words"] = request.max_words
    payload["text"] = request.text
    return json.dumps(payload, ensure_ascii=False)


def build_messages(
    requests: Sequence[TranslationRequest],
    *,
    source_language: str | None,
    target_language: str,
    context: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the chat messages for one batch.

    Length is governed by exactly one rule: the per-line word ceiling when any
    request carries one, and a proportional hint against the original otherwise.
    Sending both was the bug - the model followed the hint and overran the ceiling
    by a wide margin, which the mixer could only answer by deleting words.

    Args:
        requests: Lines to translate in this batch.
        source_language: Source language name, or ``None`` when auto-detected.
        target_language: Canonical target language name.
        context: Immediately preceding source lines, provided so the model keeps
            discourse references (pronouns, definite articles) coherent across
            batch boundaries.

    Returns:
        OpenAI-style message list.
    """
    source_clause = (
        f"The source language is {source_language}." if source_language else
        "Detect the source language from the lines themselves."
    )
    budgeted = any(request.max_words for request in requests)
    length_rule = (
        LENGTH_RULE_BUDGETED
        if budgeted
        else LENGTH_RULE_PROPORTIONAL.format(percent=int(LENGTH_TOLERANCE * 100))
    )
    system = SYSTEM_PROMPT.format(
        target=target_language,
        source_clause=source_clause,
        length_rule=length_rule,
    )

    parts: list[str] = []
    if context:
        joined = "\n".join(f"- {line}" for line in context)
        parts.append(f"Earlier lines for context (already translated, do not repeat):\n{joined}")
    parts.append("Lines to translate:")
    parts.extend(_render_line(request) for request in requests)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


SUBTITLE_SYSTEM_PROMPT = """You are a professional subtitle translator.

Translate each numbered line into {target}. {source_clause}

Rules:
1. Reply with ONLY a JSON object of the form \
{{"translations":[{{"id":<int>,"text":"<translation>"}}]}}.
2. Include every requested id exactly once, in the same order. Never invent ids.
3. Translate each id separately: do not merge or split lines.
4. Write concise, readable subtitles in natural written {target}. Prefer short \
sentences and cut padding; a viewer reads faster than a listener hears, but a \
line that is too long still cannot be read in time.
5. Keep proper nouns, numbers, units, and acronyms unchanged.
6. Never add notes, explanations, transliterations, or the untranslated source.
"""


def build_subtitle_messages(
    requests: Sequence[TranslationRequest],
    *,
    source_language: str | None,
    target_language: str,
    context: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the chat messages for one subtitle translation batch.

    Deliberately separate from :func:`build_messages`: the dubbing prompt sizes
    every line from a measured speaking rate, while a subtitle is constrained by
    reading speed and by line breaks rather than by a time slot. The transport,
    chunking and JSON repair are shared.
    """
    source_clause = (
        f"The source language is {source_language}." if source_language else
        "Detect the source language from the lines themselves."
    )
    system = SUBTITLE_SYSTEM_PROMPT.format(
        target=target_language,
        source_clause=source_clause,
    )

    parts: list[str] = []
    if context:
        joined = "\n".join(f"- {line}" for line in context)
        parts.append(
            f"Earlier lines for context (already translated, do not repeat):\n{joined}"
        )
    parts.append("Lines to translate:")
    parts.extend(_render_line(request) for request in requests)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


SHORTEN_SYSTEM_PROMPT = """You are a professional audiovisual dubbing adaptor.

The lines below were already translated into {target}, but they came out too long: \
spoken aloud they overrun the time slot available for them. Rewrite each one so that \
it fits.

Rules:
1. Reply with ONLY a JSON object of the form \
{{"translations":[{{"id":<int>,"text":"<rewritten line>"}}]}}.
2. Include every requested id exactly once, in the same order. Never invent ids.
3. Each rewrite must be at most its "max_words" words. Fewer is better than more.
4. Preserve the meaning. Cut padding first: filler and hedges ("actually", \
"really", "you know", "in fact"), redundant connectives, and restated ideas.
5. Never drop a fact, number, unit, name or acronym. If the line cannot be made \
short enough without losing information, keep the information and cut everything else.
6. Keep proper nouns, numbers, units, and acronyms unchanged.
7. Use natural spoken {target}, matching the register of the previous version.
8. Never add notes, explanations, transliterations, or the untranslated source.
"""


def build_shorten_messages(
    requests: Sequence[ShortenRequest],
    *,
    target_language: str,
) -> list[dict[str, str]]:
    """Build the chat messages used to rewrite overlong lines.

    Args:
        requests: Lines that overran their slot, with the ceiling for each.
        target_language: Canonical target language name.

    Returns:
        OpenAI-style message list.
    """
    parts = ["Lines to shorten:"]
    for request in requests:
        parts.append(
            json.dumps(
                {
                    "id": request.id,
                    "max_words": request.max_words,
                    "spoken_seconds": round(request.spoken_seconds, 2),
                    "available_seconds": round(request.available_seconds, 2),
                    "previous": request.text,
                    "meaning_to_preserve": request.source,
                },
                ensure_ascii=False,
            )
        )
    return [
        {"role": "system", "content": SHORTEN_SYSTEM_PROMPT.format(target=target_language)},
        {"role": "user", "content": "\n".join(parts)},
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a model response.

    Handles bare JSON, fenced code blocks, and JSON embedded in prose. Returns
    ``None`` when nothing parseable is found.
    """
    if not text:
        return None
    candidate = text.strip()

    if candidate.startswith("```"):
        # ```json\n{...}\n``` (or plain ```)
        first_newline = candidate.find("\n")
        if first_newline != -1:
            candidate = candidate[first_newline + 1 :]
        fence_end = candidate.rfind("```")
        if fence_end != -1:
            candidate = candidate[:fence_end]
        candidate = candidate.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _parse_translations(payload: dict[str, Any] | None) -> dict[int, str]:
    """Normalise a decoded payload into ``{id: text}``."""
    if not payload:
        return {}
    entries = payload.get("translations")
    if entries is None:
        # Tolerate a bare {id: text} mapping.
        entries = [
            {"id": key, "text": value}
            for key, value in payload.items()
            if str(key).lstrip("-").isdigit() and isinstance(value, str)
        ]
    if not isinstance(entries, list):
        return {}

    result: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        text = entry.get("text")
        if raw_id is None or text is None:
            continue
        try:
            identifier = int(raw_id)
        except (TypeError, ValueError):
            continue
        cleaned = str(text).strip()
        if cleaned:
            result[identifier] = cleaned
    return result


class OpenAICompatTranslator:
    """Translator backed by any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 180.0,
        chunk_size: int = 40,
        temperature: float = 0.2,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.chunk_size = max(1, int(chunk_size))
        self.temperature = temperature
        self._client = client

    # -- client ------------------------------------------------------------ #

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

    # -- public API -------------------------------------------------------- #

    def translate(
        self,
        requests: Sequence[TranslationRequest],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        """Translate every request, batching to respect context windows."""
        if not requests:
            return {}

        source_name = language_name_for_code(source_language) if source_language else None
        results: dict[int, str] = {}
        context: list[str] = []

        batches = list(chunked(list(requests), self.chunk_size))
        with timed(
            _LOGGER,
            f"Translated {len(requests)} line(s)",
            lines=len(requests),
            batches=len(batches),
            target=target_language,
        ):
            for batch in batches:
                messages = build_messages(
                    batch,
                    source_language=source_name,
                    target_language=target_language,
                    context=context,
                )
                batch_result = self._translate_batch(batch, messages)
                results.update(batch_result)
                context = ([*context, *(r.text for r in batch)])[-_CONTEXT_LINES:]

        return results

    def translate_subtitles(
        self,
        requests: Sequence[TranslationRequest],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        """Translate subtitle lines with reading-oriented instructions.

        Same transport and validation as :meth:`translate`, but the prompt does
        not impose a per-line spoken-length ceiling.
        """
        if not requests:
            return {}

        source_name = language_name_for_code(source_language) if source_language else None
        results: dict[int, str] = {}
        context: list[str] = []

        batches = list(chunked(list(requests), self.chunk_size))
        with timed(
            _LOGGER,
            f"Translated {len(requests)} subtitle line(s)",
            lines=len(requests),
            batches=len(batches),
            target=target_language,
        ):
            for batch in batches:
                messages = build_subtitle_messages(
                    batch,
                    source_language=source_name,
                    target_language=target_language,
                    context=context,
                )
                results.update(self._run_batch({r.id for r in batch}, messages))
                context = ([*context, *(r.text for r in batch)])[-_CONTEXT_LINES:]

        return results

    def shorten(
        self,
        requests: Sequence[ShortenRequest],
        *,
        target_language: str,
    ) -> dict[int, str]:
        """Rewrite overlong lines so they fit their slot.

        Only ever called for lines that were measured as too long, so a failure
        here costs nothing: the caller keeps the audio it already has.
        """
        if not requests:
            return {}

        results: dict[int, str] = {}
        batches = list(chunked(list(requests), self.chunk_size))
        with timed(
            _LOGGER,
            f"Shortened {len(requests)} line(s)",
            lines=len(requests),
            batches=len(batches),
            target=target_language,
        ):
            for batch in batches:
                messages = build_shorten_messages(
                    batch, target_language=target_language
                )
                results.update(self._run_batch({r.id for r in batch}, messages))
        return results

    # -- internals --------------------------------------------------------- #

    def _translate_batch(
        self,
        batch: Sequence[TranslationRequest],
        messages: list[dict[str, str]],
    ) -> dict[int, str]:
        """Translate one batch, retrying once with an explicit repair instruction."""
        return self._run_batch({request.id for request in batch}, messages)

    def _run_batch(
        self,
        expected: set[int],
        messages: list[dict[str, str]],
    ) -> dict[int, str]:
        """Send one batch and return its ids, repairing a malformed reply once."""
        with timed(
            _LOGGER,
            f"Translation batch ({len(expected)} line(s))",
            level=logging.DEBUG,
            lines=len(expected),
        ):
            parsed = self._call(messages)
        result = _parse_translations(parsed)
        missing = expected - result.keys()
        if not missing:
            return {key: value for key, value in result.items() if key in expected}

        _LOGGER.warning(
            "Translation reply missed id(s) %s; retrying with a repair instruction.",
            sorted(missing),
        )
        repair = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Your reply was missing or malformed for ids: "
                    f"{sorted(missing)}. Reply again with ONLY the JSON object, "
                    "covering every requested id exactly once."
                ),
            },
        ]
        parsed = self._call(repair, temperature=0.0)
        result = _parse_translations(parsed)
        missing = expected - result.keys()
        if missing:
            raise TranslationError(
                f"Translation response missing ids {sorted(missing)} after retry"
            )
        return {key: value for key, value in result.items() if key in expected}

    def _call(
        self, messages: list[dict[str, str]], *, temperature: float | None = None
    ) -> dict[str, Any] | None:
        """Perform one chat completion and return the decoded JSON payload."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        _LOGGER.debug(
            "Chat request: model=%s messages=%d temperature=%s",
            self.model,
            len(messages),
            payload["temperature"],
        )

        try:
            with timed(_LOGGER, "Chat completion", level=logging.DEBUG):
                response = self._request(payload)
        except Exception as exc:  # noqa: BLE001 - normalised into a domain error
            raise TranslationError(
                f"Translation request to {self.base_url} failed: {exc}. "
                "Check that your OpenAI-compatible server is running."
            ) from exc

        content = self._content_from(response)
        parsed = extract_json_object(content)
        if parsed is None:
            raise TranslationError(
                "Translation response was not valid JSON. Increase LLM_CHUNK_SIZE to "
                "reduce batch size, or verify the model supports JSON output."
            )
        return parsed

    def _request(self, payload: dict[str, Any]) -> Any:
        """Issue the completion, preferring strict JSON mode but tolerating servers
        that do not implement ``response_format``."""
        try:
            return self.client.chat.completions.create(
                **payload, response_format={"type": "json_object"}
            )
        except Exception:  # noqa: BLE001 - retry without the optional parameter
            return self.client.chat.completions.create(**payload)

    @staticmethod
    def _content_from(response: Any) -> str:
        """Extract the assistant text from an OpenAI-style response object."""
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise TranslationError("Unexpected response shape from translation model") from exc
        if isinstance(content, list):
            # Some servers return content parts rather than a plain string.
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content or "")
