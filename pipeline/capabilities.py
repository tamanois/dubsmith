"""Declared capabilities of each synthesis backend.

This module is the one place that states what a backend can do. The CLI/API
validators consult it so they cannot disagree with each other, and the HTTP API
publishes it (``GET /api/backends``) so the web forms can disable an option the
selected backend cannot honour *before* the user submits it.

Only stdlib is imported here: the API process imports this module at startup and
must not pull in torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

__all__ = [
    "BackendCapabilities",
    "DUB_BACKENDS",
    "EXPRESSIVE_BACKENDS",
    "SAY_BACKENDS",
    "capabilities_payload",
    "say_prosody_conflict",
]

#: Backends that render a :class:`~pipeline.prosody.ProsodySpec` (or inline tags)
#: instead of the plain Qwen Base clone. They all clone from a reference clip.
EXPRESSIVE_BACKENDS: tuple[str, ...] = ("cosyvoice3", "chatterbox", "fish")

_SOURCES: tuple[str, ...] = ("voice", "design", "clone")


@dataclass(frozen=True)
class BackendCapabilities:
    """What one backend can honour, plus the reason for each gap.

    ``unsupported`` maps a feature name to a user-facing explanation. The web
    form shows it beside the disabled control; the same text is returned when the
    server rejects an option, so the two never drift.
    """

    id: str
    label: str
    prosody: bool = False
    style: bool = False
    inline_tags: bool = False
    fish_tuning: bool = False
    chatterbox_tuning: bool = False
    speaker: bool = False
    sources: tuple[str, ...] = _SOURCES
    save_as: bool = False
    expressive: bool = False
    unsupported: Mapping[str, str] = field(default_factory=dict)


_PROSYD_BASE = (
    "--prosody/--emotion/--prosody-plan only apply to CustomVoice, CosyVoice3 and "
    "Chatterbox; Qwen Base has no prosody controls — put the delivery cues in the "
    "text."
)
_PROSYD_FISH = (
    "--prosody/--emotion/--prosody-plan are unavailable with --backend fish; it "
    "has no --prosody/--emotion/--prosody-plan channel. Use [tags] inside --text "
    "or --fish-tag-mode llm."
)

SAY_BACKENDS: dict[str, BackendCapabilities] = {
    "qwen": BackendCapabilities(
        id="qwen",
        label="Qwen Base (cloning)",
        prosody=False,
        save_as=True,
        unsupported={
            "prosody": _PROSYD_BASE,
            "style": "Style instructions are only available with CustomVoice.",
        },
    ),
    "customvoice": BackendCapabilities(
        id="customvoice",
        label="Qwen CustomVoice (preset speaker)",
        prosody=True,
        style=True,
        speaker=True,
        sources=(),
        unsupported={
            "sources": (
                "CustomVoice uses a preset speaker; it does not take a saved, "
                "designed or cloned voice."
            ),
            "save_as": "Saving a new voice is only available with Qwen Base.",
        },
    ),
    "cosyvoice3": BackendCapabilities(
        id="cosyvoice3",
        label="CosyVoice3",
        prosody=True,
        expressive=True,
        unsupported={
            "save_as": "Saving a new voice is only available with Qwen Base."
        },
    ),
    "chatterbox": BackendCapabilities(
        id="chatterbox",
        label="Chatterbox",
        prosody=True,
        chatterbox_tuning=True,
        expressive=True,
        unsupported={
            "save_as": "Saving a new voice is only available with Qwen Base."
        },
    ),
    "fish": BackendCapabilities(
        id="fish",
        label="Fish S2-Pro",
        prosody=False,
        inline_tags=True,
        fish_tuning=True,
        expressive=True,
        unsupported={
            "prosody": _PROSYD_FISH,
            "fish_tuning": "Fish tuning is only available with the Fish backend.",
            "save_as": "Saving a new voice is only available with Qwen Base.",
        },
    ),
}

DUB_BACKENDS: dict[str, BackendCapabilities] = {
    "qwen": BackendCapabilities(
        id="qwen",
        label="Qwen Base (cloning, no prosody)",
        prosody=False,
        unsupported={
            "prosody": (
                "Qwen Base ignores prosody planning; it renders the clone "
                "directly."
            ),
            "expressive": (
                "Blending a native accent is only available with CosyVoice3, "
                "Chatterbox or Fish."
            ),
        },
    ),
    "cosyvoice3": BackendCapabilities(
        id="cosyvoice3",
        label="CosyVoice3 (emotion + tokens)",
        prosody=True,
        expressive=True,
    ),
    "chatterbox": BackendCapabilities(
        id="chatterbox",
        label="Chatterbox (expression intensity)",
        prosody=True,
        chatterbox_tuning=True,
        expressive=True,
    ),
    "fish": BackendCapabilities(
        id="fish",
        label="Fish S2-Pro (inline tags)",
        prosody=False,
        inline_tags=True,
        fish_tuning=True,
        expressive=True,
        unsupported={
            "prosody": "Fish takes its direction from [tags] inside the text."
        },
    ),
}


def capabilities_payload() -> dict[str, list[dict[str, object]]]:
    """The JSON shape ``GET /api/backends`` returns, one group per job kind."""
    return {
        "say": [asdict(caps) for caps in SAY_BACKENDS.values()],
        "dub": [asdict(caps) for caps in DUB_BACKENDS.values()],
    }


def say_prosody_conflict(
    backend: str,
    *,
    prosody: object = None,
    emotions: object = None,
    prosody_plan: str | None = "off",
) -> str | None:
    """Why this backend cannot take the given prosody fields, or ``None``.

    Shared by ``validate_say_spec`` and the API's submit-time check, so a web
    request is rejected before a job is enqueued instead of failing in the worker.
    """
    caps = SAY_BACKENDS.get(backend)
    if caps is None or caps.prosody:
        return None
    if not prosody and not emotions and (prosody_plan or "off") == "off":
        return None
    return caps.unsupported.get("prosody")
