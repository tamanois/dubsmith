"""SubRip (SRT) rendering.

Pure, dependency-free string handling so it is cheap to unit test and cannot be
affected by media tooling versions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Cue", "format_timestamp", "build_srt", "write_srt", "clean_text"]

_WHITESPACE = re.compile(r"\s+")


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm``.

    Negative values are clamped to zero, and the value is rounded once in
    milliseconds so ``59.9996`` becomes ``01:00:00,000`` rather than
    ``00:00:59,1000``.
    """
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clean_text(text: str) -> str:
    """Collapse whitespace/newlines, which SRT cue text must not contain raw."""
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class Cue:
    """One subtitle cue."""

    start: float
    end: float
    text: str

    def normalized(self) -> "Cue":
        """Return a cue safe to render: ordered timestamps, clean, non-empty text."""
        start = max(0.0, float(self.start))
        end = max(start, float(self.end))
        return Cue(start=start, end=end, text=clean_text(self.text))


def build_srt(cues: list[Cue], *, skip_empty: bool = True) -> str:
    """Render cues to SRT text.

    Args:
        cues: Cues in presentation order; they are renumbered from 1.
        skip_empty: Drop cues whose text is empty after cleaning. Subtitle files
            reject empty cues, and the ASR stage can legitimately produce them.

    Returns:
        SRT document, empty string when nothing remains.
    """
    blocks: list[str] = []
    index = 0
    for cue in cues:
        normalized = cue.normalized()
        if skip_empty and not normalized.text:
            continue
        index += 1
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(normalized.start)} --> {format_timestamp(normalized.end)}\n"
            f"{normalized.text}\n"
        )
    return "\n".join(blocks)


def write_srt(cues: list[Cue], path: Path) -> Path:
    """Write cues to ``path`` (creating parents) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_srt(cues), encoding="utf-8")
    return path
