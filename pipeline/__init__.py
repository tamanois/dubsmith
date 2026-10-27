"""Local video dubbing pipeline.

Turns a video into a dub in a target language, cloning each speaker's own voice
timbre. See ``PLAN.md`` for scope and roadmap.

Module map:
    config      -- job/settings models and language normalisation
    errors      -- domain exception hierarchy
    logging_utils -- job-scoped structured logging
    models      -- model registry, download and path resolution
    media       -- ffmpeg/ffprobe wrappers and loudness utilities
    subtitles   -- SRT rendering
    asr         -- speech recognition and segmentation
    translate   -- OpenAI-compatible translation
    tts         -- voice cloning with Qwen3-TTS
    voices      -- saved, labeled voice prints (design + clone library)
    timing      -- time-stretching and timeline placement
    orchestrate -- stage runner tying everything together
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
