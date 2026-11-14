"""One voice, and the rule that builds its clone prompt.

The clone prompt *is* the voice: it is built from a timbre source plus, in the
accent modes, a target-language clip that supplies pronunciation. Two places have
to build the very same prompt from the very same facts - the pipeline, in one
process, and a synthesis worker, in another - so the facts live here as
:class:`VoiceSpec` and the rule lives here as :func:`build_prompt`.

A spec is deliberately made of plain data: a prompt is bound to the device that
built it and cannot be handed to another process, so a worker builds its own from
the description. That is also what makes the spec hashable, and therefore what
lets a worker skip rebuilding a prompt it already holds (:attr:`VoiceSpec.key`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import TTSError
from .options import AccentMode
from .voices import load_prompt

if TYPE_CHECKING:  # pragma: no cover - annotation only, avoids an import cycle
    from .tts import TTSEngine

_LOGGER = logging.getLogger(__name__)

__all__ = ["VoiceSpec", "build_prompt"]


@dataclass(frozen=True)
class VoiceSpec:
    """Everything needed to build one clone prompt, and nothing that is not."""

    #: Speaker this voice belongs to, as the pipeline names it (``SPK_0``, ...).
    speaker: str
    #: One of ``native``, ``reference-accent``, ``preserve``, ``speaker-only``.
    accent_mode: str
    #: Target-language clip supplying pronunciation (anchor or supplied clip).
    accent_audio: Path | None = None
    #: Its exact transcript, which aligns the accent clip's speech codes.
    accent_text: str = ""
    #: The speaker's blended reference from stage 1, when the job has one. When it
    #: is set the blend clip *is* the voice: both signals come from it, so the
    #: accent fields above are not consulted.
    blend: Path | None = None
    #: Pinned reference voice, when the user supplied one.
    ref_audio: Path | None = None
    ref_text: str = ""
    #: The library's stored encoding of :attr:`ref_audio`, when it has one. A
    #: worker that installs it skips the encode pass entirely.
    prompt: Path | None = None
    #: Length of the clip :attr:`prompt` encodes, used only to size a batch.
    prompt_seconds: float = 0.0
    #: Auto-selected donor for this speaker (a :class:`pipeline.tts.DonorClip`).
    #: Untyped to keep this module free of the heavier import.
    donor: Any | None = None
    #: Human-readable description of the timbre source, for the log line.
    timbre_label: str = ""
    #: True when one prompt serves every speaker (a pinned voice, no blend), so
    #: the same reference is never encoded once per speaker.
    shared: bool = False

    @property
    def key(self) -> str:
        """Identity of the prompt this spec builds.

        Two speakers sharing one pinned reference voice are *one* prompt, so they
        must share a key - otherwise each of them re-encodes the same audio for
        nothing. Everything that can differ between two prompts is folded in,
        including the donor's segment, because a stale prompt would be a wrong
        voice rather than a slow run.
        """
        if self.shared:
            return "shared"
        donor_id = ""
        if self.donor is not None:
            donor_id = str(getattr(self.donor, "segment_id", ""))
        return "|".join(
            (
                self.speaker,
                self.accent_mode,
                str(self.ref_audio or ""),
                str(self.prompt or ""),
                str(self.blend or ""),
                str(self.accent_audio or ""),
                donor_id,
            )
        )


def build_prompt(engine: TTSEngine, spec: VoiceSpec) -> None:
    """Prepare ``engine``'s clone prompt for ``spec``.

    The one implementation of "which clip is this voice", shared by the pipeline
    and by a worker so the two cannot drift:

    * in an accent mode, a blended clip *is* the voice - its codes and its
      embedding cannot disagree, which is the whole point of stage 1;
    * failing that, pronunciation comes from the accent clip and timbre from the
      donor (or from a stored prompt's embedding), combined by the engine's
      hybrid prompt;
    * a pinned reference voice with a stored prompt is installed as-is: nothing
      is encoded, because the library already holds the encoding;
    * a pinned reference without one is cloned from its file;
    * otherwise the donor is cloned directly.

    Raises:
        TTSError: When a spec cannot produce a prompt at all (no donor, no pinned
            voice), which is a bug in the caller rather than a bad recording.
    """
    data = (
        load_prompt(spec.prompt)
        if spec.prompt is not None and spec.blend is None
        else None
    )
    if spec.accent_mode in (
        AccentMode.NATIVE,
        AccentMode.REFERENCE_ACCENT,
    ):
        if spec.blend is not None:
            engine.prepare_prompt_from_file(spec.blend, spec.accent_text)
            return
        _LOGGER.info(
            "Hybrid voice for %s: timbre from %s, accent from %s",
            spec.speaker,
            spec.timbre_label or "auto-selected donor",
            spec.accent_audio,
        )
        engine.prepare_hybrid_prompt(
            accent_audio=spec.accent_audio,
            accent_text=spec.accent_text,
            timbre=spec.donor,
            timbre_audio=spec.ref_audio,
            timbre_item=data,
        )
        return
    if data is not None:
        engine.prepare_prompt_from_data(data, seconds=spec.prompt_seconds)
        return
    if spec.ref_audio is not None:
        engine.prepare_prompt_from_file(spec.ref_audio, spec.ref_text)
        return
    if spec.donor is None:
        raise TTSError(
            f"No timbre source for {spec.speaker}: the job has neither a reference "
            "clip nor a donor segment."
        )
    engine.prepare_prompt(spec.donor)
