"""SubRip (SRT) rendering.

Pure, dependency-free string handling so it is cheap to unit test and cannot be
affected by media tooling versions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "Cue",
    "format_timestamp",
    "build_srt",
    "write_srt",
    "clean_text",
    "split_cue_text",
    "cues_from_segments",
]

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


def split_cue_text(text: str, *, max_chars: int = 42) -> list[str]:
    """Split one line into subtitle-sized chunks, preferring word boundaries.

    A chunk never exceeds ``max_chars`` unless a single word is longer than that
    limit, in which case the word is returned as its own chunk rather than cut.
    Whitespace-only input yields no chunks, so the caller can skip the cue.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if max_chars <= 0:
        return [cleaned]

    chunks: list[str] = []
    current = ""
    for word in cleaned.split(" "):
        if not word:
            continue
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_span(
    start: float, end: float, weights: Sequence[float]
) -> list[tuple[float, float]]:
    """Divide ``[start, end]`` into contiguous spans proportional to ``weights``.

    Kept private and dependency-free so ``subtitles.py`` stays pure. The last
    span always ends exactly at ``end``, so rounding cannot leave a gap.
    """
    span = max(0.0, float(end) - float(start))
    if not weights:
        return []
    total = float(sum(weights))
    if total <= 0.0:
        weights = [1.0 for _ in weights]
        total = float(len(weights))

    spans: list[tuple[float, float]] = []
    cursor = float(start)
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            stop = float(end)
        else:
            stop = cursor + span * (float(weight) / total)
        stop = max(cursor, min(float(end), stop))
        spans.append((cursor, stop))
        cursor = stop
    return spans


def cues_from_segments(
    segments: Sequence[Any],
    texts: Mapping[int, str] | None = None,
    *,
    max_chars: int = 42,
) -> list[Cue]:
    """Build cues from ASR segments, splitting long lines into readable chunks.

    Args:
        segments: Objects exposing ``id``, ``start``, ``end`` and ``text`` —
            normally :class:`pipeline.asr.Segment`.
        texts: Optional replacement text keyed by segment id, e.g. the
            translation. Missing ids fall back to the segment's own text.
        max_chars: Longest a cue's text may be before it is split.

    Returns:
        Cues in timeline order. Each segment's span is shared between its chunks
        in proportion to their word counts, so a split never leaves the original
        utterance.
    """
    cues: list[Cue] = []
    for segment in segments:
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        identifier = int(getattr(segment, "id", 0))
        text = (
            texts.get(identifier, getattr(segment, "text", ""))
            if texts is not None
            else getattr(segment, "text", "")
        )
        chunks = split_cue_text(text, max_chars=max_chars)
        if not chunks:
            continue
        weights = [max(1, len(chunk.split())) for chunk in chunks]
        for chunk, (chunk_start, chunk_end) in zip(
            chunks, _split_span(start, end, weights)
        ):
            cues.append(Cue(chunk_start, chunk_end, chunk))
    return cues
