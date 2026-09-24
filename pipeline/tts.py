"""Voice cloning with Qwen3-TTS.

Two responsibilities:

1. :func:`select_donor` picks the best reference clip from the source video when
   the user does not supply one. It is a pure function over samples + segments so
   it can be tested without a model.
2. :class:`TTSEngine` wraps the Base checkpoint, builds the clone prompt once, and
   synthesises all lines in batches.

The Base checkpoint is required: only ``tts_model_type == "base"`` implements
``generate_voice_clone``. Pointing this at CustomVoice or VoiceDesign raises a
clear error instead of the library's generic one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .asr import Segment
from .config import release_gpu_memory
from .errors import DependencyError, TTSError

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DonorClip",
    "select_donor",
    "score_candidate",
    "TTSEngine",
    "resolve_dtype",
]

#: Donor clip length window. Too short and the speaker embedding is unstable;
#: too long and prompt encoding wastes time for no benefit.
MIN_DONOR_S = 3.0
MAX_DONOR_S = 15.0
IDEAL_DONOR_S = 8.0


@dataclass(frozen=True)
class DonorClip:
    """A reference clip used to clone a voice."""

    samples: np.ndarray
    sample_rate: int
    text: str
    start: float
    end: float
    segment_id: int

    @property
    def duration(self) -> float:
        """Clip length in seconds."""
        return len(self.samples) / float(self.sample_rate)

    @property
    def rms(self) -> float:
        """Root-mean-square energy, used for candidate scoring."""
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))


def score_candidate(segment: Segment, samples: np.ndarray, sample_rate: int) -> float:
    """Score a segment as a donor candidate; higher is better.

    Balances three signals:
      * length closeness to :data:`IDEAL_DONOR_S` (a triangle peaking at the ideal),
      * loudness (louder, clearer speech gives a better embedding),
      * the model's own no-speech probability (punished when high).
    """
    duration = segment.duration
    if duration <= 0:
        return float("-inf")

    # Triangle score: 1.0 at the ideal length, falling to 0 at the bounds.
    if duration < IDEAL_DONOR_S:
        span = max(1e-6, IDEAL_DONOR_S - MIN_DONOR_S)
        length_score = max(0.0, (duration - MIN_DONOR_S) / span)
    else:
        span = max(1e-6, MAX_DONOR_S - IDEAL_DONOR_S)
        length_score = max(0.0, 1.0 - (duration - IDEAL_DONOR_S) / span)

    start = max(0, int(round(segment.start * sample_rate)))
    end = min(len(samples), int(round(segment.end * sample_rate)))
    if end <= start:
        return float("-inf")
    chunk = samples[start:end]
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    # Map a plausible speech RMS range onto 0..1 without hard failure on quiet audio.
    loudness_score = float(np.clip(rms / 0.05, 0.0, 1.0))

    speech_score = 1.0 - float(np.clip(segment.no_speech_prob, 0.0, 1.0))
    return (0.5 * length_score + 0.3 * loudness_score + 0.2 * speech_score)


def select_donor(
    samples: np.ndarray,
    sample_rate: int,
    segments: Sequence[Segment],
    *,
    min_s: float = MIN_DONOR_S,
    max_s: float = MAX_DONOR_S,
) -> DonorClip:
    """Choose the best reference clip from the source audio.

    Prefers segments inside ``[min_s, max_s]``. When none qualify (very short or
    unusually paced audio) the longest available segment is used instead, so a
    best-effort clone is still produced rather than failing the job.

    Raises:
        TTSError: If there are no segments to choose from.
    """
    if not segments:
        raise TTSError("Cannot select a donor voice: the transcript has no segments.")

    in_range = [s for s in segments if min_s <= s.duration <= max_s]
    pool = in_range or list(segments)

    if in_range:
        best = max(pool, key=lambda s: score_candidate(s, samples, sample_rate))
    else:
        best = max(pool, key=lambda s: s.duration)

    start = max(0, int(round(best.start * sample_rate)))
    end = min(len(samples), int(round(best.end * sample_rate)))
    if end <= start:
        raise TTSError(
            f"Donor segment {best.id} has no audio samples in range "
            f"({best.start:.2f}s–{best.end:.2f}s)."
        )

    return DonorClip(
        samples=np.ascontiguousarray(samples[start:end], dtype=np.float32),
        sample_rate=int(sample_rate),
        text=best.text,
        start=best.start,
        end=best.end,
        segment_id=best.id,
    )


def resolve_dtype(device: str) -> Any:
    """bfloat16 on CUDA, float32 on CPU (CPU bfloat16 is slow and patchy)."""
    import torch

    return torch.bfloat16 if str(device).lower().startswith("cuda") else torch.float32


def _is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory errors (the class or a plain RuntimeError)."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:  # pragma: no cover - torch is a hard runtime dep
        pass
    return "out of memory" in str(exc).lower()


class TTSEngine:
    """Voice-cloning synthesiser backed by a local Qwen3-TTS Base checkpoint."""

    #: Checkpoints that implement ``generate_voice_clone``.
    SUPPORTED_MODEL_TYPES = ("base",)

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "auto",
        dtype: Any | None = None,
        batch_size: int = 8,
        x_vector_only: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._dtype = dtype
        self.batch_size = max(1, int(batch_size))
        #: Use only the reference's speaker embedding, ignoring its speech codes.
        #: ICL mode (False) also conditions on the reference phonetics, which is
        #: what carries the speaker's *accent*; speaker-only mode drops those and
        #: lets the model use its own phonetics for the target language.
        self.x_vector_only = bool(x_vector_only)
        self._model: Any | None = None
        self._prompt: Any | None = None
        self._sample_rate: int | None = None

    # -- lifecycle --------------------------------------------------------- #

    @property
    def model(self) -> Any:
        """Load the checkpoint on first use and keep it resident."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:  # pragma: no cover - config dependent
            raise DependencyError(
                "qwen-tts is not installed. Run: pip install -r requirements.txt"
            ) from exc

        from .config import resolve_device

        device = resolve_device(self.device)
        self.device = device
        model = Qwen3TTSModel.from_pretrained(
            str(self.model_dir),
            device_map=device,
            dtype=self._dtype if self._dtype is not None else resolve_dtype(device),
            attn_implementation="sdpa",
            local_files_only=True,
        )

        model_type = getattr(model.model, "tts_model_type", None)
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise TTSError(
                f"Checkpoint at {self.model_dir} has tts_model_type={model_type!r}, "
                "which does not support voice cloning. Use the '-Base' checkpoint "
                "(download with: python dub.py models --download tts_base)."
            )
        return model

    @property
    def sample_rate(self) -> int:
        """Model output sample rate, discovered from the first synthesis."""
        if self._sample_rate is None:
            raise TTSError("Sample rate is unknown until the first synthesis call.")
        return self._sample_rate

    # -- prompt ------------------------------------------------------------ #

    def prepare_prompt(self, donor: DonorClip) -> None:
        """Build the reusable clone prompt from a donor clip.

        Built once and reused for every line, which avoids re-encoding the
        reference audio on each call.
        """
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=(np.asarray(donor.samples, dtype=np.float32), int(donor.sample_rate)),
            ref_text=donor.text,
            x_vector_only_mode=self.x_vector_only,
        )

    def prepare_prompt_from_file(self, ref_audio: Path | str, ref_text: str) -> None:
        """Build the clone prompt from an explicit reference file and transcript."""
        # ICL mode conditions on the transcript, so it is required there; in
        # speaker-only mode the text is ignored and may legitimately be absent.
        if not self.x_vector_only and (not ref_text or not ref_text.strip()):
            raise TTSError("ref_text is required to clone from a reference file (ICL mode).")
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=self.x_vector_only,
        )

    # -- synthesis --------------------------------------------------------- #

    def synthesize(self, texts: Sequence[str], *, language: str) -> list[np.ndarray | None]:
        """Synthesise each line, returning arrays aligned with ``texts``.

        Blank lines map to ``None`` so the timing stage can emit a matching
        stretch of silence instead of letting the model invent speech.

        Args:
            texts: Target-language lines, one per segment.
            language: Canonical language name (e.g. ``"French"``).

        Returns:
            One entry per input: a mono float32 array, or ``None`` for blanks.

        Raises:
            TTSError: If the clone prompt is missing or the model's output count
                does not match the request.
        """
        if self._prompt is None:
            raise TTSError("prepare_prompt() must be called before synthesize().")
        if not texts:
            return []

        results: list[np.ndarray | None] = [None] * len(texts)
        pending = [(index, text.strip()) for index, text in enumerate(texts) if text.strip()]
        if not pending:
            return results

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            wavs, sample_rate = self._generate(batch, language)
            if len(wavs) != len(batch):
                raise TTSError(
                    f"Model returned {len(wavs)} clips for a batch of {len(batch)} lines."
                )
            self._sample_rate = int(sample_rate)
            for (index, _), wav in zip(batch, wavs):
                results[index] = np.ascontiguousarray(
                    np.asarray(wav, dtype=np.float32).reshape(-1)
                )
        return results

    def _generate(
        self,
        batch: Sequence[tuple[int, str]],
        language: str,
    ) -> tuple[list[np.ndarray], int]:
        """Generate one batch, splitting it in half if the GPU runs out of memory.

        Long lines need a large key/value cache, so a batch that fits early in a
        video can fail on a later, wordier passage. Halving the batch (rather
        than failing the job) keeps memory bounded and still produces every line;
        recursion bottoms out at a single line, where the error is genuine.
        """
        try:
            return self.model.generate_voice_clone(
                text=[text for _, text in batch],
                language=[language] * len(batch),
                voice_clone_prompt=self._prompt,
            )
        except (RuntimeError, MemoryError) as exc:
            if not _is_cuda_oom(exc) or len(batch) == 1:
                if _is_cuda_oom(exc):
                    raise TTSError(
                        "Out of GPU memory synthesising a single line. Free the GPU "
                        "or retry with --device cpu."
                    ) from exc
                raise
            release_gpu_memory()
            _LOGGER.warning(
                "GPU out of memory on a batch of %d; retrying as two smaller batches.",
                len(batch),
            )
            midpoint = len(batch) // 2
            first_wavs, first_rate = self._generate(batch[:midpoint], language)
            second_wavs, second_rate = self._generate(batch[midpoint:], language)
            return [*first_wavs, *second_wavs], first_rate or second_rate
