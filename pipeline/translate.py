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
from typing import Any, Iterable, Protocol, Sequence

from pydantic import BaseModel, Field

from .config import language_name_for_code
from .errors import DependencyError, TranslationError

__all__ = [
    "TranslationRequest",
    "Translator",
    "OpenAICompatTranslator",
    "build_messages",
    "extract_json_object",
    "chunked",
    "LENGTH_TOLERANCE",
]

#: How much longer/shorter a dub line may be than the original slot. Mirrors the
#: timing stage's clamp so the two stages agree on what "fits" means.
LENGTH_TOLERANCE = 0.2

_CONTEXT_LINES = 3

SYSTEM_PROMPT = """You are a professional audiovisual translator and dubbing adaptor.

Translate each numbered line into {target}. {source_clause}

Rules:
1. Reply with ONLY a JSON object of the form \
{{"translations":[{{"id":<int>,"text":"<translation>"}}]}}.
2. Include every requested id exactly once, in the same order. Never invent ids.
3. Keep each translation close in spoken length to its original (within roughly \
{percent}%). Say the same thing in fewer words rather than translating literally.
4. Preserve register and tone, and use natural spoken {target} rather than \
written style. Match the formality implied by the source.
5. Keep proper nouns, numbers, units, and acronyms unchanged.
6. Never add notes, explanations, transliterations, or the untranslated source.
"""


class TranslationRequest(BaseModel):
    """One line to translate, with the slot it must fit."""

    id: int
    text: str
    duration: float = 0.0


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


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield successive slices of at most ``size`` items."""
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _render_line(request: TranslationRequest) -> str:
    return json.dumps(
        {"id": request.id, "seconds": round(request.duration, 2), "text": request.text},
        ensure_ascii=False,
    )


def build_messages(
    requests: Sequence[TranslationRequest],
    *,
    source_language: str | None,
    target_language: str,
    context: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the chat messages for one batch.

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
    system = SYSTEM_PROMPT.format(
        target=target_language,
        source_clause=source_clause,
        percent=int(LENGTH_TOLERANCE * 100),
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

        for batch in chunked(list(requests), self.chunk_size):
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

    # -- internals --------------------------------------------------------- #

    def _translate_batch(
        self,
        batch: Sequence[TranslationRequest],
        messages: list[dict[str, str]],
    ) -> dict[int, str]:
        """Translate one batch, retrying once with an explicit repair instruction."""
        expected = {request.id for request in batch}

        parsed = self._call(messages)
        result = _parse_translations(parsed)
        missing = expected - result.keys()
        if not missing:
            return {key: value for key, value in result.items() if key in expected}

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

        try:
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
