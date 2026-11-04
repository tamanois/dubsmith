"""Provenance for prosody: what was asked, chosen and spoken.

A ``say``, dubbing or saved-voice run can write a small JSON artifact recording
the per-line delivery settings, the LLM reply that produced them and the exact
text transformations sent to the backend. It is an audit trail only: nothing in
the pipeline reads it back, so writing it can never change a render.

The recorder is deliberately backend-agnostic. Planners and engines push what
they already computed; this module owns the schema and the file layout, so the
CLI and the pipeline cannot drift apart about either.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "LineProvenance",
    "LLMCall",
    "NullRecorder",
    "PlannerConfig",
    "PlannerTrace",
    "ProvenanceRecorder",
    "ProvenanceReport",
    "Transformation",
    "describe_transformations",
    "timed_provenance_path",
]

#: Suffix a provenance file is written with. Kept stable so a report can point at
#: it without knowing the timestamp.
PROVENANCE_PREFIX = "provenance"


@dataclass
class PlannerConfig:
    """How the LLM was configured, independent of any one request."""

    model: str
    base_url: str
    temperature: float
    chunk_size: int
    backend: str | None = None


@dataclass
class LLMCall:
    """One completion: the reply and whether JSON mode had to be abandoned."""

    label: str
    response: str
    fallback: bool = False


@dataclass
class PlannerTrace:
    """Every LLM exchange that contributed to a plan."""

    config: PlannerConfig | None = None
    overview: str | None = None
    calls: list[LLMCall] = field(default_factory=list)
    error: str | None = None


@dataclass
class Transformation:
    """One change applied to a line before it reached the model."""

    kind: str
    before: str | None = None
    after: str | None = None
    detail: str | None = None


@dataclass
class LineProvenance:
    """The settings and transforms for one line."""

    #: Segment id for a dub, line index for ``say``.
    key: int | str
    speaker: str | None = None
    #: The text the plan or spec was produced from.
    text_planned: str | None = None
    #: The text actually sent to the backend.
    text_spoken: str | None = None
    #: Raw ``--prosody``/``--emotion`` values, before they were merged.
    overrides: dict[str, Any] | None = None
    #: The effective :class:`~pipeline.prosody.ProsodySpec`, as a dict.
    spec: dict[str, Any] | None = None
    #: What the renderer produced: channel, text, instruct, mode, scalar, cues.
    rendered: dict[str, Any] | None = None
    transformations: list[Transformation] = field(default_factory=list)


@dataclass
class ProvenanceReport:
    """The serialisable artifact."""

    kind: str
    generated_at: str
    backend: str
    version: int = 1
    language: str | None = None
    plan_source: str = "off"
    #: False when the backend has no prosody channel at all (Qwen Base).
    supported: bool = True
    planner: PlannerTrace | None = None
    lines: list[LineProvenance] = field(default_factory=list)
    #: Saved-voice details (mode, label, prompt, model, fingerprint, verify).
    voice: dict[str, Any] | None = None
    #: Optional artifact paths this report describes.
    output: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _LOGGER.info("Wrote provenance to %s", target)
        return target


class ProvenanceRecorder:
    """Accumulates a run's provenance, then writes it once."""

    def __init__(
        self,
        kind: str,
        backend: str,
        *,
        language: str | None = None,
        plan_source: str = "off",
    ) -> None:
        self.kind = kind
        self.backend = backend
        self.language = language
        self.plan_source = plan_source
        self._config: PlannerConfig | None = None
        self._overview: str | None = None
        self._calls: list[LLMCall] = []
        self._error: str | None = None
        #: Keyed by line key so a rewrite replaces its own entry rather than
        #: appending a second record for the same line.
        self._lines: dict[int | str, LineProvenance] = {}
        self._voice: dict[str, Any] | None = None
        self._output: dict[str, Any] | None = None

    def enabled(self) -> bool:
        return True

    def set_plan_source(self, source: str) -> None:
        self.plan_source = source

    def record_planner_config(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float,
        chunk_size: int,
        backend: str | None = None,
    ) -> None:
        if self._config is not None:
            return
        self._config = PlannerConfig(
            model=str(model),
            base_url=str(base_url),
            temperature=float(temperature),
            chunk_size=int(chunk_size),
            backend=backend,
        )

    def record_overview(self, text: str | None) -> None:
        value = str(text or "").strip()
        if value:
            self._overview = value

    def record_call(
        self,
        label: str,
        response: str,
        *,
        fallback: bool = False,
    ) -> None:
        self._calls.append(
            LLMCall(
                label=str(label),
                response=str(response or ""),
                fallback=bool(fallback),
            )
        )

    def record_error(self, exc: BaseException | str) -> None:
        self._error = str(exc)

    def record_line(self, line: LineProvenance) -> None:
        self._lines[line.key] = line

    def record_voice(self, **fields: Any) -> None:
        self._voice = dict(fields)

    def record_output(self, **fields: Any) -> None:
        self._output = dict(fields)

    def build(self, *, supported: bool = True) -> ProvenanceReport:
        planner: PlannerTrace | None = None
        if self._config is not None or self._calls or self._overview or self._error:
            planner = PlannerTrace(
                config=self._config,
                overview=self._overview,
                calls=list(self._calls),
                error=self._error,
            )
        return ProvenanceReport(
            kind=self.kind,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            backend=self.backend,
            language=self.language,
            plan_source=self.plan_source,
            supported=supported,
            planner=planner,
            lines=list(self._lines.values()),
            voice=self._voice,
            output=self._output,
        )

    def write(self, path: Path | str, *, supported: bool = True) -> Path:
        return self.build(supported=supported).write(path)


class NullRecorder:
    """A recorder that writes nothing, so ``--no-provenance`` costs no code path."""

    def enabled(self) -> bool:
        return False

    def set_plan_source(self, source: str) -> None:
        return

    def record_planner_config(self, **kwargs: Any) -> None:
        return

    def record_overview(self, text: str | None) -> None:
        return

    def record_call(self, *args: Any, **kwargs: Any) -> None:
        return

    def record_error(self, exc: BaseException | str) -> None:
        return

    def record_line(self, line: LineProvenance) -> None:
        return

    def record_voice(self, **fields: Any) -> None:
        return

    def record_output(self, **fields: Any) -> None:
        return

    def build(self, *, supported: bool = True) -> None:
        return None

    def write(self, path: Path | str, *, supported: bool = True) -> None:
        return None


def timed_provenance_path(
    directory: Path | str, *, prefix: str = PROVENANCE_PREFIX
) -> Path:
    """A timestamped path in ``directory``, suffixed if the same second is reused.

    The timestamp makes every run's artifact unique; the suffix keeps two runs
    started in the same second from overwriting each other.
    """
    directory = Path(directory)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = directory / f"{prefix}-{stamp}.json"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{prefix}-{stamp}-{counter}.json"
        counter += 1
    return candidate


def describe_transformations(
    planned: str,
    spoken: str,
    *,
    channel: str,
    spec: Mapping[str, Any] | None = None,
    instruct: str | None = None,
    exaggeration: float | None = None,
) -> list[Transformation]:
    """Name what changed between the planned text and the text sent to the model.

    ``spec`` is read for the pause flag, but the tokens themselves are detected
    from the two strings: a renderer that adds a cue it was not asked for is
    still recorded accurately.
    """
    before = str(planned or "")
    after = str(spoken or "")
    items: list[Transformation] = []

    if channel == "cosyvoice":
        # The cross-lingual path prefixes the system prompt; it is transport, not
        # a transformation of the line.
        try:
            from .prosody import COSYVOICE3_SYSTEM_PROMPT

            if after.startswith(COSYVOICE3_SYSTEM_PROMPT):
                after = after[len(COSYVOICE3_SYSTEM_PROMPT) :]
        except ImportError:  # pragma: no cover - prosody is a hard dependency
            pass
        for token in ("[breath]", "[laughter]"):
            if token in after and token not in before:
                items.append(
                    Transformation(
                        kind="inline_token",
                        before=before,
                        after=after,
                        detail=token,
                    )
                )
        if after != before and ", " in after and ", " not in before:
            items.append(
                Transformation(kind="pause", before=before, after=after)
            )
        if instruct:
            items.append(Transformation(kind="instruct", detail=str(instruct)))
    elif channel == "fish":
        try:
            from .prosody import fish_cue_tokens

            cues = fish_cue_tokens(after)
        except ImportError:  # pragma: no cover - prosody is a hard dependency
            cues = []
        if cues:
            items.append(
                Transformation(
                    kind="fish_tag",
                    before=before,
                    after=after,
                    detail=", ".join(cues),
                )
            )
        elif before != after:
            items.append(
                Transformation(kind="fish_tag", before=before, after=after)
            )
    elif channel == "chatterbox":
        if exaggeration is not None:
            items.append(
                Transformation(
                    kind="exaggeration",
                    detail=f"{float(exaggeration):.3f}",
                )
            )
    elif channel == "customvoice":
        if instruct:
            items.append(Transformation(kind="instruct", detail=str(instruct)))

    return items
