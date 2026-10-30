"""Voice cloning with Qwen3-TTS.

Two responsibilities:

1. :func:`select_donor` picks the best reference clip from the source video when
   the user does not supply one. It is a pure function over samples + segments so
   it can be tested without a model.
2. :class:`TTSEngine` wraps the Base checkpoint, builds the clone prompt once, and
   synthesises all lines in batches.
3. For ``reference-accent``, :meth:`TTSEngine.build_blended_reference` renders the
   accent transcript once in the donor's voice, so every line can then be cloned
   from a single clip whose codes and speaker embedding agree.

The Base checkpoint is required: only ``tts_model_type == "base"`` implements
``generate_voice_clone``. Pointing this at CustomVoice or VoiceDesign raises a
clear error instead of the library's generic one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .asr import Segment
from .config import (
    device_free_bytes,
    peak_reserved_bytes,
    pick_dtype,
    probe_device,
    release_gpu_memory,
    reset_peak_reserved,
)
from .errors import DependencyError, TTSError
from .logging_utils import timed

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "BlendedReference",
    "CustomVoiceEngine",
    "DonorClip",
    "ReferenceAudio",
    "select_donor",
    "score_candidate",
    "TTSEngine",
    "VoiceDesignEngine",
    "resolve_dtype",
]

#: Donor clip length window. Too short and the speaker embedding is unstable;
#: too long and prompt encoding wastes time for no benefit.
MIN_DONOR_S = 3.0
MAX_DONOR_S = 15.0
IDEAL_DONOR_S = 8.0

#: Rate the checkpoint's speaker encoder expects. ``extract_speaker_embedding``
#: asserts on it, so any clip scored against a prompt embedding must be resampled.
SPEAKER_ENCODER_RATE = 24_000

#: Free-memory floor below which a batch hands cached blocks back to the driver.
#: Set from the failure, not by taste: on the 11-minute run a 0.8-2.4 GiB request
#: failed with free memory as low as 54 MB, i.e. the batch had started with barely
#: more than the old 1.5 GB floor. Releasing at 3 GB means a batch starts with room
#: for the transient tensors it is about to ask for. Costs milliseconds.
MEMORY_FLOOR_BYTES = 3_000_000_000

#: Total audio the decoder may be asked to process in one call, in seconds -
#: **including the clone prompt's reference, which is decoded once per line**.
#: ``generate_voice_clone`` concatenates ``ref_code`` onto every line's codes and
#: only trims the reference back off the waveform afterwards, so a batch of eight
#: lines cloned from a 21 s blend clip decodes ~170 s of audio however short the
#: lines are. Measured (RTX 3060, bfloat16, 2026-09-28): the peak grows by
#: ~0.65 GiB per line at that reference length, and this budget keeps a batch near
#: 85 s of decoded audio, which measured 6.7 GiB of an 11.87 GiB card.
MAX_DECODE_SECONDS = 85.0

#: Fraction of the card a batch may fill before the following ones are made
#: smaller. The peak is not predictable from the text - it depends on the
#: reference length, the dtype and the card - so it is measured per batch and this
#: is the one rule applied to the measurement. 0.7 of an 11.87 GiB card is ~8.3 GiB,
#: which left 3.5 GiB free for the largest transient requests observed (2.3 GiB).
MEMORY_CEILING_FRACTION = 0.7

#: Ceiling used when the card's own size cannot be read.
DEFAULT_MEMORY_CEILING_BYTES = 8_000_000_000

#: Seconds of output assumed per character before any batch has been measured.
#: Only a seed for the first batch of a run: the engine then uses what it actually
#: produced, so the estimate converges on the language and delivery at hand.
_SEED_SECONDS_PER_CHAR = 0.07

#: A reference clip as the checkpoint accepts it: a path, or raw samples and rate.
#: Iteration needs the second form, because the previous blend exists only in
#: memory until the run finishes.
ReferenceAudio = Path | str | tuple[np.ndarray, int]


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


@dataclass(frozen=True)
class BlendedReference:
    """A one-off clip of the donor's voice speaking the accent clip's transcript.

    The checkpoint conditions a line on two independent signals: ``ref_code``
    (pronunciation and prosody) and ``ref_spk_embedding`` (vocal identity). When
    they come from different clips they can disagree, and a sampled line may
    follow either one - which is heard as the speaker changing mid-video. This
    clip removes that choice, because both signals are derived from it.

    The similarity fields record how the winning take was chosen. ``margin`` is
    the difference between the two, so a negative value means the take that best
    kept the donor's timbre was still closer to the accent clip.
    """

    samples: np.ndarray
    sample_rate: int
    text: str
    #: Takes considered in the first pass (1 when scoring was unavailable).
    takes: int
    #: Passes run. ``1`` means the first pass already met the target, or no target
    #: was set; higher means the blend had to be pulled further toward the speaker.
    iters: int
    #: Where the kept clip sits on the scale from the accent voice (``0.0``) to a
    #: French utterance in the speaker's own voice (``1.0``). ``NaN`` when no donor
    #: anchor could be built. Roughly ``0.5`` is an even blend.
    donor_ratio: float
    #: ``timbre_similarity - accent_similarity`` of the winning take.
    #:
    #: This ranks takes against *each other*, and the ranking is sound because
    #: every take reads the same text, so any language penalty applies equally to
    #: all of them. The *absolute* value is not: a French take compared against an
    #: English donor is penalised for the language difference rather than the
    #: voice, so a negative margin does not by itself mean the voice collapsed.
    #: Measured on this repo's TED talk, the two reference voices sat 0.035 apart on
    #: the encoder's scale while a perfectly good take scored -0.011. Judging a
    #: clip in absolute terms needs a language-matched reference, which is what
    #: ``audit_timbre.py`` builds - not a threshold here.
    margin: float
    #: Cosine similarity of the winning take to the donor's speaker embedding.
    timbre_similarity: float
    #: Cosine similarity of the winning take to the accent clip's embedding.
    accent_similarity: float

    @property
    def duration(self) -> float:
        """Clip length in seconds."""
        return len(self.samples) / float(self.sample_rate)


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


def resolve_dtype(device: str, requested: str = "auto") -> Any:
    """The torch dtype to load the checkpoint with on ``device``.

    Capability-aware: bfloat16 only where the hardware has it (Ampere and newer
    on CUDA, CDNA on ROCm), float16 on older accelerators, float32 on CPU. The
    previous behaviour picked bfloat16 for *any* CUDA device, which is unsupported
    on Pascal and Volta cards - the most likely "other GPU" a user has.

    ``requested`` is an explicit ``--dtype``; it is honoured unless the device
    cannot run it, in which case it degrades and logs why.
    """
    return pick_dtype(device, requested)


def _is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory errors (the class or a plain RuntimeError)."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:  # pragma: no cover - torch is a hard runtime dep
        pass
    return "out of memory" in str(exc).lower()


def _embedding_vector(value: Any) -> np.ndarray | None:
    """Flatten a tensor/array embedding into a finite float vector.

    Returns ``None`` for anything that cannot be compared - an unavailable
    embedding, a placeholder, or one carrying NaN - so callers can treat
    "unscorable" differently from "scored zero". Tensors are detached and moved
    to the CPU first, which is why a duck-typed ``detach`` check is enough here.
    """
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    return vector


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors.

    Returns ``NaN`` when the vectors cannot be compared (different widths, or a
    zero-norm vector), which keeps "no opinion" distinct from "zero similarity".
    """
    if left.shape != right.shape:
        return float("nan")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def _format_similarity(value: float) -> str:
    """Readable similarity, or ``"n/a"`` when a take could not be scored."""
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _reference_audio(value: ReferenceAudio) -> Any:
    """A reference clip in the form the checkpoint expects.

    Paths become strings while ``(samples, rate)`` pairs pass through, which is
    what lets a blend be fed back in as the next pass's reference.
    """
    if isinstance(value, tuple):
        samples, rate = value
        return (np.asarray(samples, dtype=np.float32), int(rate))
    return str(value)


def _audio_seconds(value: ReferenceAudio) -> float:
    """Duration of a reference clip, or ``0.0`` when it cannot be measured.

    Only ever used to charge a batch for the audio the decoder will be handed, so
    an unmeasurable clip is charged nothing rather than failing: the measured
    memory ceiling below still catches a batch that turns out too large.
    """
    if isinstance(value, tuple):
        samples, rate = value
        return float(len(samples)) / float(rate or 1)
    try:
        return float(sf.info(str(value)).duration)
    except Exception:  # noqa: BLE001 - a reference clip may be a URL or a stream
        _LOGGER.debug("Could not measure the reference clip %s", value, exc_info=True)
        return 0.0


def _donor_ratio(
    candidate: np.ndarray,
    accent_embedding: np.ndarray | None,
    donor_embedding: np.ndarray | None,
) -> float:
    """Where ``candidate`` sits between the accent voice and the donor's own voice.

    Projects the candidate onto the line joining the two anchors, so ``0.0`` is
    the accent clip and ``1.0`` is a French utterance in the donor's voice. Both
    anchors are in the target language, which is the whole point: a raw similarity
    against an English donor measures the language gap as much as the voice, and
    its absolute value was therefore never safe to threshold.

    Returns ``NaN`` when either anchor is missing or the two coincide.
    """
    if accent_embedding is None or donor_embedding is None:
        return float("nan")
    if accent_embedding.shape != candidate.shape or donor_embedding.shape != candidate.shape:
        return float("nan")
    axis = donor_embedding - accent_embedding
    span = float(np.dot(axis, axis))
    if not np.isfinite(span) or span <= 0.0:
        return float("nan")
    position = float(np.dot(candidate - accent_embedding, axis) / span)
    if not np.isfinite(position):
        return float("nan")
    return float(np.clip(position, 0.0, 1.0))


#: How much shorter than the longest sibling a render may be and still be treated
#: as a real candidate for the blended reference.
_TRUNCATED_RENDER_FLOOR = 0.5


def _is_truncated(candidate: np.ndarray, reference: np.ndarray) -> bool:
    """True when ``candidate`` is far too short beside ``reference`` to be complete.

    A genuinely terser take is shorter than its siblings, but not by half; the model
    occasionally returns a fragment of the requested text instead.
    """
    return int(candidate.size) < _TRUNCATED_RENDER_FLOOR * int(reference.size)


def _drop_truncated(renders: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Drop renders that clearly stopped early, keeping the fullest ones.

    The model occasionally returns a fragment of the requested text - observed on a
    real run as a 1.4 s clip for a 14 s transcript. A fragment is a *failed* render,
    not a candidate voice: it covers only part of the text, yet the ratio search will
    happily prefer it, because a short clip can score well on its own. Keeping only
    renders within :data:`_TRUNCATED_RENDER_FLOOR` of the longest costs nothing,
    since they have already been generated.

    The longest render is within the floor of itself, so a non-empty input always
    yields at least one candidate.
    """
    if not renders:
        return []
    longest = max(renders, key=lambda wav: int(wav.size))
    return [wav for wav in renders if not _is_truncated(wav, longest)]


@dataclass(frozen=True)
class _BlendPass:
    """One rendered blend and how it scored on the language-matched scale."""

    samples: np.ndarray
    timbre_similarity: float
    accent_similarity: float
    donor_ratio: float

    @property
    def margin(self) -> float:
        """``timbre - accent``, used to rank *takes* within a pass.

        Only meaningful for ranking: every take reads the same text, so the
        language penalty in the timbre term cancels. It is not an absolute score -
        see :attr:`BlendedReference.margin`. Falls back to the timbre similarity
        when there is no accent side to move away from.
        """
        if not np.isfinite(self.timbre_similarity):
            return float("nan")
        if np.isfinite(self.accent_similarity):
            return self.timbre_similarity - self.accent_similarity
        return self.timbre_similarity


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
        max_decode_seconds: float = MAX_DECODE_SECONDS,
        x_vector_only: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._dtype = dtype
        self.batch_size = max(1, int(batch_size))
        #: The cap the user asked for, which the measured ceiling may shrink from.
        self.max_batch_size = self.batch_size
        #: Longest decoded audio one batch may ask for; see
        #: :data:`MAX_DECODE_SECONDS`. A single line over budget still goes out
        #: alone, because text cannot be split.
        self.max_decode_seconds = max(1.0, float(max_decode_seconds))
        #: Length of the clone prompt's reference, which the checkpoint decodes once
        #: per line. Set by the ``prepare_*`` calls, and ``0.0`` when the prompt
        #: carries no speech codes (x-vector mode), where nothing is decoded on the
        #: line's behalf.
        self.reference_seconds = 0.0
        #: Seconds of output per character of text: seeded, then measured from what
        #: the model actually produces. Only ever used to size a batch.
        self._seconds_per_char = _SEED_SECONDS_PER_CHAR
        #: How much of the card a batch may fill, resolved from the card on first
        #: use. ``None`` off CUDA, where nothing is measured.
        self.memory_ceiling: int | None = None
        #: How often the run hit the ceiling or the allocator, and how often it
        #: handed cached blocks back. A thrashing run used to leave artifacts that
        #: looked clean, so these travel to ``report.json`` rather than the log.
        self.oom_retries = 0
        self.cache_releases = 0
        self.ceiling_shrinks = 0
        self.batches = 0
        self.peak_batch_seconds = 0.0
        self.peak_reserved = 0
        #: Use only the reference's speaker embedding, ignoring its speech codes.
        #: ICL mode (False) also conditions on the reference phonetics, which is
        #: what carries the speaker's *accent*; speaker-only mode drops those and
        #: lets the model use its own phonetics for the target language.
        self.x_vector_only = bool(x_vector_only)
        self._model: Any | None = None
        self._prompt: Any | None = None
        self._sample_rate: int | None = None
        #: Speaker embeddings of the hybrid prompt's two sources, recorded when it
        #: is built so candidate blends can be scored against both without
        #: re-encoding either clip. ``None`` until ``prepare_hybrid_prompt`` runs.
        self._accent_spk_embedding: Any | None = None
        self._timbre_spk_embedding: Any | None = None

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
        with timed(
            _LOGGER, "Built the clone prompt", level=logging.DEBUG, mode="donor"
        ):
            self._prompt = self.model.create_voice_clone_prompt(
                ref_audio=(
                    np.asarray(donor.samples, dtype=np.float32),
                    int(donor.sample_rate),
                ),
                ref_text=donor.text,
                x_vector_only_mode=self.x_vector_only,
            )
        self.reference_seconds = 0.0 if self.x_vector_only else donor.duration

    def prepare_prompt_from_file(self, ref_audio: Path | str, ref_text: str) -> None:
        """Build the clone prompt from an explicit reference file and transcript."""
        # ICL mode conditions on the transcript, so it is required there; in
        # speaker-only mode the text is ignored and may legitimately be absent.
        if not self.x_vector_only and (not ref_text or not ref_text.strip()):
            raise TTSError("ref_text is required to clone from a reference file (ICL mode).")
        with timed(
            _LOGGER, "Built the clone prompt", level=logging.DEBUG, mode="file"
        ):
            self._prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                x_vector_only_mode=self.x_vector_only,
            )
        self.reference_seconds = 0.0 if self.x_vector_only else _audio_seconds(ref_audio)

    def prepare_hybrid_prompt(
        self,
        *,
        accent_audio: ReferenceAudio,
        accent_text: str,
        timbre: DonorClip | None = None,
        timbre_audio: Path | str | None = None,
    ) -> None:
        """Take the accent from one clip and the timbre from another.

        The checkpoint conditions on two independent signals: ``ref_code`` (the
        reference's speech codes, which drive pronunciation and prosody) and
        ``ref_spk_embedding`` (who the speaker is). Sourcing them from different
        clips yields native pronunciation in the original speaker's voice - the
        combination plain cloning cannot express, because it derives both from a
        single reference.

        Args:
            accent_audio: Native target-language clip supplying the phonetics. Also
                accepts ``(samples, rate)``, which is how iteration feeds the
                previous blend back in as the next pass's code source.
            accent_text: Its exact transcript; mandatory, because ICL mode aligns
                the speech codes against the reference text.
            timbre: Speaker to keep, taken from the source video.
            timbre_audio: Speaker to keep, taken from a file.

        Raises:
            TTSError: If the accent transcript is missing, neither timbre source is
                given, or the library's prompt item cannot be combined.
        """
        if not accent_text or not accent_text.strip():
            raise TTSError(
                "accent_text is required: the accent clip's transcript is what "
                "aligns its speech codes so they can condition pronunciation."
            )
        if timbre is None and timbre_audio is None:
            raise TTSError("prepare_hybrid_prompt needs a timbre source.")

        with timed(
            _LOGGER, "Built the hybrid prompt", level=logging.DEBUG, mode="hybrid"
        ):
            accent_items = self.model.create_voice_clone_prompt(
                ref_audio=_reference_audio(accent_audio),
                ref_text=accent_text,
                x_vector_only_mode=False,
            )

            # x-vector mode returns the speaker embedding without needing a
            # transcript (its speech codes are discarded anyway).
            if timbre is not None:
                timbre_items = self.model.create_voice_clone_prompt(
                    ref_audio=(
                        np.asarray(timbre.samples, dtype=np.float32),
                        int(timbre.sample_rate),
                    ),
                    ref_text=timbre.text,
                    x_vector_only_mode=True,
                )
            else:
                timbre_items = self.model.create_voice_clone_prompt(
                    ref_audio=str(timbre_audio),
                    ref_text="",
                    x_vector_only_mode=True,
                )

        # Keep the accent item's ref_code/ref_text/flags so `ref_ids` stay aligned
        # with the codes, and swap in the other clip's speaker embedding.
        try:
            combined = replace(
                accent_items[0], ref_spk_embedding=timbre_items[0].ref_spk_embedding
            )
        except (AttributeError, TypeError) as exc:  # pragma: no cover - library change
            raise TTSError(
                "Could not combine accent and timbre prompts; the installed qwen-tts "
                "version may have changed VoiceClonePromptItem."
            ) from exc
        self._prompt = [combined]
        # The codes the decoder will be handed once per line come from the accent
        # side, so that clip is what a batch has to be charged for - the timbre
        # side is an embedding and contributes none.
        self.reference_seconds = _audio_seconds(accent_audio)
        # Recorded from the items already in hand, so scoring candidate blends
        # against both sources costs no extra encoder pass. The accent item is
        # read after ``replace`` only because the two live in different items -
        # ``replace`` returns a copy and leaves the original untouched.
        self._accent_spk_embedding = getattr(accent_items[0], "ref_spk_embedding", None)
        self._timbre_spk_embedding = getattr(timbre_items[0], "ref_spk_embedding", None)

    # -- blended reference (two-stage) ------------------------------------- #

    def speaker_embedding(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Speaker embedding of a clip, resampled to the encoder's 24 kHz.

        This is the one objective handle on "whose voice is this", so it serves
        both the take selection below and an offline audit of a finished dub.

        Raises:
            TTSError: If the checkpoint returns nothing usable.
        """
        audio_in = np.asarray(samples, dtype=np.float32).reshape(-1)
        if int(sample_rate) != SPEAKER_ENCODER_RATE:
            import librosa

            audio_in = librosa.resample(
                audio_in, orig_sr=int(sample_rate), target_sr=SPEAKER_ENCODER_RATE
            )
        vector = _embedding_vector(
            self.model.model.extract_speaker_embedding(
                audio=audio_in, sr=SPEAKER_ENCODER_RATE
            )
        )
        if vector is None:
            raise TTSError("The checkpoint returned no usable speaker embedding.")
        return vector

    def _donor_anchor(
        self,
        *,
        accent_text: str,
        language: str,
        timbre: DonorClip | None,
        timbre_audio: Path | str | None,
    ) -> np.ndarray | None:
        """A target-language utterance in the donor's own voice, embedding only.

        Built in x-vector mode, so no accent codes take part: it is what the speaker
        sounds like without the borrowed pronunciation. That makes it the far end of
        a *language-matched* scale, and therefore something absolute measurements
        can finally be taken against - a raw similarity to an English donor only
        ever measured the language gap.

        Best-effort: a checkpoint without the speaker encoder yields ``None`` and the
        ratio is simply left unmeasured.

        Note:
            Replaces the current clone prompt, like every other ``prepare_*`` call.
        """
        if timbre is not None:
            ref_audio: Any = (
                np.asarray(timbre.samples, dtype=np.float32),
                int(timbre.sample_rate),
            )
        elif timbre_audio is not None:
            ref_audio = str(timbre_audio)
        else:
            return None

        try:
            self._prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text="",
                x_vector_only_mode=True,
            )
            # x-vector mode carries no speech codes, so nothing is decoded on the
            # line's behalf and the batch is charged for its own audio only.
            self.reference_seconds = 0.0
            wavs = self.synthesize([accent_text], language=language)
            anchor = wavs[0] if wavs else None
            if anchor is None or not np.asarray(anchor).size:
                return None
            return self.speaker_embedding(anchor, self.sample_rate)
        except Exception as exc:  # noqa: BLE001 - a missing anchor is not fatal
            _LOGGER.warning("Could not build a donor anchor clip: %s", exc)
            return None

    def _render(self, text: str, language: str, count: int) -> list[np.ndarray]:
        """Synthesise ``count`` takes of ``text``, dropping anything unusable."""
        wavs = self.synthesize([text] * count, language=language)
        return [
            np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
            for wav in wavs
            if wav is not None and np.asarray(wav).size
        ]

    def _measure_pass(
        self,
        samples: np.ndarray,
        timbre_embedding: np.ndarray | None,
        accent_embedding: np.ndarray | None,
        donor_anchor: np.ndarray | None,
    ) -> _BlendPass:
        """Score one rendered pass; anything unmeasurable stays ``NaN``."""
        try:
            candidate = self.speaker_embedding(samples, self.sample_rate)
        except Exception as exc:  # noqa: BLE001 - scoring must never break a run
            _LOGGER.warning("Could not score a blend pass: %s", exc)
            return _BlendPass(samples, float("nan"), float("nan"), float("nan"))
        return _BlendPass(
            samples=samples,
            timbre_similarity=(
                _cosine(candidate, timbre_embedding)
                if timbre_embedding is not None
                else float("nan")
            ),
            accent_similarity=(
                _cosine(candidate, accent_embedding)
                if accent_embedding is not None
                else float("nan")
            ),
            donor_ratio=_donor_ratio(candidate, accent_embedding, donor_anchor),
        )

    @staticmethod
    def _best_pass(passes: Sequence[_BlendPass]) -> int:
        """Index of the pass that moved furthest toward the speaker's own voice.

        Prefers the measured donor ratio, which is language-matched by construction.
        The accent similarity is the fallback when no anchor could be built - still
        comparable, because both of *its* sides are in the target language. Ties keep
        the earliest pass, so a rerun is reproducible.
        """
        ratios = [item.donor_ratio for item in passes]
        if any(np.isfinite(value) for value in ratios):
            return int(np.nanargmax(ratios))
        accents = [item.accent_similarity for item in passes]
        if any(np.isfinite(value) for value in accents):
            return int(np.nanargmin(accents))
        return 0

    def build_blended_reference(
        self,
        *,
        accent_audio: Path | str,
        accent_text: str,
        language: str,
        timbre: DonorClip | None = None,
        timbre_audio: Path | str | None = None,
        takes: int = 1,
        iters: int = 1,
        ratio: float = 0.0,
    ) -> BlendedReference:
        """Render the accent transcript in the donor's voice, iterating toward it.

        Stage 1 of the two-stage ``reference-accent`` path. The accent clip supplies
        the speech codes and the donor the speaker embedding, so the model reconciles
        the two sources *once* here - instead of once per line, where each line is
        free to pick a different winner.

        One pass is not neutral between them. The accent clip arrives as a full
        in-context example - its speech codes and its transcript - while the donor is
        a single embedding token, so the model reproduces the accent voice far more
        faithfully than the speaker's own. ``iters > 1`` re-encodes the previous
        blend and re-injects the donor's embedding, walking the voice back toward the
        speaker while the accent transcript keeps pronunciation anchored. Iteration
        stops as soon as the donor ``ratio`` is reached, and the *best* pass is kept
        rather than the last, because passes are stochastic and need not improve
        monotonically.

        Args:
            takes: Candidates rendered in the first pass; the best one is kept.
            iters: Maximum passes.
            ratio: Donor ratio to stop at; ``0`` runs every pass without measuring.

        Raises:
            TTSError: If a prompt cannot be built, or the model returns no audio.
        """
        count = max(1, int(takes))
        planned = max(1, int(iters))
        target = float(ratio)

        # Where the speaker's own voice sits on the scale. Skipped when nothing
        # would read it, since it costs a synthesis of its own.
        donor_anchor: np.ndarray | None = None
        if target > 0.0 or planned > 1:
            donor_anchor = self._donor_anchor(
                accent_text=accent_text,
                language=language,
                timbre=timbre,
                timbre_audio=timbre_audio,
            )

        # Pass 0 reconciles the two sources for the first time. Its accent embedding
        # is what every later pass is measured against, so it is captured now: each
        # later pass overwrites it with the blend it is handed.
        self.prepare_hybrid_prompt(
            accent_audio=accent_audio,
            accent_text=accent_text,
            timbre=timbre,
            timbre_audio=timbre_audio,
        )
        accent_embedding = _embedding_vector(self._accent_spk_embedding)
        timbre_embedding = _embedding_vector(self._timbre_spk_embedding)

        candidates = self._render(accent_text, language, count)
        if not candidates:
            raise TTSError("The model returned no audio for the blended reference.")
        # A take that stopped early is a failed render, not a voice: it covers only
        # part of the transcript, and it is exactly the kind of clip the ratio search
        # below would prefer. Filtered before scoring so it cannot be chosen.
        candidates = _drop_truncated(candidates)

        # Scoring is only paid for when it decides something: which take to keep, or
        # how far along the scale the blend ended up.
        measuring = count > 1 or target > 0.0 or planned > 1
        pool = [
            (
                self._measure_pass(wav, timbre_embedding, accent_embedding, donor_anchor)
                if measuring
                else _BlendPass(wav, float("nan"), float("nan"), float("nan"))
            )
            for wav in candidates
        ]
        margins = [item.margin for item in pool]
        first = 0
        if any(np.isfinite(value) for value in margins):
            # Ties keep the earliest take, so a rerun is reproducible.
            first = int(np.nanargmax(margins))
        made = [pool[first]]
        for _pass in range(1, planned):
            if (
                target > 0.0
                and np.isfinite(made[-1].donor_ratio)
                and made[-1].donor_ratio >= target
            ):
                break
            # Codes come from the previous blend, the embedding from the donor: the
            # pull back toward the speaker, applied once more.
            self.prepare_hybrid_prompt(
                accent_audio=(made[-1].samples, self.sample_rate),
                accent_text=accent_text,
                timbre=timbre,
                timbre_audio=timbre_audio,
            )
            rendered = self._render(accent_text, language, 1)
            if not rendered:
                break
            if _is_truncated(rendered[0], made[-1].samples):
                # A fragment, so there is nothing to measure: retry rather than
                # let a partial render compete with a full one. Bounded by `planned`.
                continue
            made.append(
                self._measure_pass(
                    rendered[0], timbre_embedding, accent_embedding, donor_anchor
                )
            )

        for number, item in enumerate(made):
            _LOGGER.debug(
                "Blend pass %d/%d: accent %s, timbre %s, donor ratio %s",
                number + 1,
                len(made),
                _format_similarity(item.accent_similarity),
                _format_similarity(item.timbre_similarity),
                _format_similarity(item.donor_ratio),
            )

        best = self._best_pass(made)
        kept = made[best]
        margin = kept.margin
        if target > 0.0 and np.isfinite(kept.donor_ratio) and kept.donor_ratio < target:
            _LOGGER.warning(
                "The blend reached a donor ratio of only %s after %d pass(es), short "
                "of the %.2f target. The dub is still built from it; raise "
                "--blend-iters to pull further toward the speaker's voice.",
                _format_similarity(kept.donor_ratio),
                len(made),
                target,
            )
        _LOGGER.info(
            "Blended reference: kept pass %d of %d (accent %s, donor ratio %s, "
            "margin %s)",
            best + 1,
            len(made),
            _format_similarity(kept.accent_similarity),
            _format_similarity(kept.donor_ratio),
            _format_similarity(margin),
        )
        return BlendedReference(
            samples=kept.samples,
            sample_rate=int(self.sample_rate),
            text=accent_text,
            takes=len(candidates),
            iters=len(made),
            donor_ratio=kept.donor_ratio,
            margin=margin,
            timbre_similarity=kept.timbre_similarity,
            accent_similarity=kept.accent_similarity,
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

        started = time.perf_counter()
        batches = 0
        while pending:
            batch = self._take_batch(pending)
            batches += 1
            self._release_when_tight()
            # Measured per batch, so the ceiling below reacts to what *this* call
            # asked for rather than to a high-water mark from earlier in the run.
            reset_peak_reserved(self.device)
            with timed(
                _LOGGER,
                f"TTS batch ({len(batch)} line(s))",
                level=logging.DEBUG,
                lines=len(batch),
            ):
                wavs, sample_rate = self._generate(batch, language)
            if len(wavs) != len(batch):
                raise TTSError(
                    f"Model returned {len(wavs)} clips for a batch of {len(batch)} lines."
                )
            self._sample_rate = int(sample_rate)
            self._observe(batch, wavs, sample_rate)
            self._adapt_batch_size(len(batch))
            for (index, _), wav in zip(batch, wavs):
                results[index] = np.ascontiguousarray(
                    np.asarray(wav, dtype=np.float32).reshape(-1)
                )
            pending = pending[len(batch) :]
        elapsed = time.perf_counter() - started
        _LOGGER.info(
            "Synthesised %d line(s) in %d batch(es) in %.2fs",
            len(texts),
            batches,
            elapsed,
        )
        return results

    def measure_delivery_rate(self, text: str, *, language: str) -> float:
        """Words per second this voice delivers ``text`` at, measured on the spot.

        The word budget that sizes every translation has to come from the voice
        that will speak it, and the honest way to get it is to render something
        and time it: the reference clip's *own* pace reads about 20% slower than
        the clone's delivery (2.97 against 3.65 words/s measured on a real job),
        which made the ceiling a third too tight and the model ignored it. One line
        of text is the whole cost.

        Args:
            text: Target-language text to read. The caller caps its length.
            language: Canonical language name, as for :meth:`synthesize`.

        Returns:
            Words per second, or ``0.0`` when nothing usable came back.
        """
        words = len(text.split())
        if words == 0:
            return 0.0
        with timed(_LOGGER, "Delivery-rate probe", lines=1):
            rendered = self.synthesize([text], language=language)
        samples = rendered[0] if rendered else None
        if samples is None or not np.asarray(samples).size:
            _LOGGER.warning("Delivery-rate probe returned no audio.")
            return 0.0
        seconds = len(np.asarray(samples).reshape(-1)) / float(self.sample_rate)
        rate = words / seconds if seconds > 0 else 0.0
        _LOGGER.info(
            "Delivery rate: %.2f words/s (%d word(s) in %.2fs)",
            rate,
            words,
            seconds,
        )
        return rate

    def _take_batch(
        self, pending: Sequence[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """The next batch: bounded by line count *and* by what will be decoded.

        The reference is charged **per line**, because that is how the checkpoint
        uses it: it concatenates the prompt's speech codes onto every line's codes
        and decodes the lot, trimming the reference off the waveform afterwards.
        A 21 s reference therefore costs ~21 s of decoding *per line*, which dwarfed
        the lines themselves on the 11-minute run and is why capping the lines' own
        duration alone changed nothing.

        Batches are taken one at a time rather than pre-computed, because the first
        batch's outcome is what sizes the rest of the run: both an out-of-memory
        error and the measured ceiling shrink ``batch_size``, and every later batch
        has to respect the smaller size. A single line over budget still goes out
        alone - text cannot be split.
        """
        batch: list[tuple[int, str]] = []
        predicted = 0.0
        for item in pending:
            seconds = self.reference_seconds + len(item[1]) * self._seconds_per_char
            if batch and (
                len(batch) >= self.batch_size
                or predicted + seconds > self.max_decode_seconds
            ):
                break
            batch.append(item)
            predicted += seconds
        return batch

    def _observe(
        self,
        batch: Sequence[tuple[int, str]],
        wavs: Sequence[np.ndarray],
        sample_rate: int,
    ) -> None:
        """Learn how long this text takes, from what was just produced.

        A running average of seconds per character, which is what lets the next
        batch be sized for the language and delivery actually in use instead of a
        constant chosen for one of them.
        """
        rate = max(1, int(sample_rate))
        produced = sum(int(np.asarray(wav).size) / rate for wav in wavs)
        self.batches += 1
        self.peak_batch_seconds = max(self.peak_batch_seconds, produced)
        characters = sum(len(text) for _index, text in batch)
        if characters <= 0 or produced <= 0.0:
            return
        observed = produced / characters
        self._seconds_per_char = (
            observed
            if self.batches == 1
            else ((self._seconds_per_char * (self.batches - 1)) + observed)
            / self.batches
        )

    def _release_when_tight(self) -> None:
        """Hand cached blocks back to the driver before they are needed.

        The allocator keeps freed blocks at the shape they were requested, so after
        a few hundred batches most of a card can be held in sizes nothing can use —
        measured as a 1.6 GiB request failing against 1.1 GiB of free memory.
        Releasing while there is still room is what stops that; waiting for the
        out-of-memory error is waiting too long.
        """
        free = device_free_bytes(self.device)
        if free is None or free >= MEMORY_FLOOR_BYTES:
            return
        if release_gpu_memory():
            self.cache_releases += 1

    def _resolve_ceiling(self) -> int | None:
        """How much of the card one batch may fill, or ``None`` off CUDA."""
        if self.memory_ceiling is None:
            info = probe_device(self.device)
            if info.kind not in {"cuda", "rocm"}:
                return None
            self.memory_ceiling = (
                int(MEMORY_CEILING_FRACTION * info.total_memory)
                if info.total_memory
                else DEFAULT_MEMORY_CEILING_BYTES
            )
        return self.memory_ceiling

    def _adapt_batch_size(self, taken: int) -> None:
        """Resize the next batch from the memory this one actually used.

        The peak cannot be predicted from the text: the checkpoint decodes the clone
        prompt's reference once per line, so a batch cloned from a 21 s clip asks for
        ~21 s of decoding per line however short the lines are. Measured on the
        11-minute job (RTX 3060, bfloat16): batches of eight lines peaked at
        9.55 GiB of an 11.87 GiB card and still failed 66 allocations, while batches
        of four peaked at 6.74 GiB and failed none. Rather than encode that in a
        constant that would be wrong on another card, dtype or reference length,
        measure it: a batch that filled more of the card than the ceiling allows
        halves the size of everything after it. It never grows back, so the run
        cannot oscillate around the edge.

        The halving applies to ``taken`` - the batch that actually did it - rather
        than to the configured cap, because the two differ whenever the decoded-audio
        budget is the binding constraint. On the 11-minute run the first overshoot
        was a three-line batch against a cap of eight; halving the cap would have
        left the next batch as large as the one that just overshot.
        """
        ceiling = self._resolve_ceiling()
        peak = peak_reserved_bytes(self.device) if ceiling is not None else None
        if ceiling is None or peak is None:
            return
        self.peak_reserved = max(self.peak_reserved, peak)
        if peak <= ceiling or self.batch_size <= 1:
            return
        self.batch_size = max(1, min(self.batch_size, taken) // 2)
        self.ceiling_shrinks += 1
        _LOGGER.warning(
            "A batch of %d line(s) filled %.1f GiB, past the %.1f GiB ceiling for "
            "this card; using %d line(s) per batch from here on.",
            taken,
            peak / 1024**3,
            ceiling / 1024**3,
            self.batch_size,
        )

    def synthesis_stats(self) -> dict[str, Any]:
        """What the batching did, for ``report.json``."""
        return {
            "batch_size": self.batch_size,
            "max_decode_seconds": round(self.max_decode_seconds, 2),
            "reference_seconds": round(self.reference_seconds, 2),
            "seconds_per_char": round(self._seconds_per_char, 4),
            "batches": self.batches,
            "peak_batch_seconds": round(self.peak_batch_seconds, 2),
            "oom_retries": self.oom_retries,
            "cache_releases": self.cache_releases,
            "ceiling_shrinks": self.ceiling_shrinks,
            "peak_reserved_bytes": self.peak_reserved or None,
        }

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
            self.oom_retries += 1
            # Remembered rather than re-derived: a batch that did not fit is
            # evidence about this job's budget, and without it every later batch
            # asks for the same thing again - which on a long video means paying
            # the failed allocation hundreds of times over.
            self.batch_size = max(1, len(batch) // 2)
            _LOGGER.warning(
                "GPU out of memory on a batch of %d; retrying as two smaller batches "
                "and using %d line(s) per batch from here on.",
                len(batch),
                self.batch_size,
            )
            midpoint = len(batch) // 2
            first_wavs, first_rate = self._generate(batch[:midpoint], language)
            second_wavs, second_rate = self._generate(batch[midpoint:], language)
            return [*first_wavs, *second_wavs], first_rate or second_rate


class VoiceDesignEngine:
    """Speech synthesiser backed by the Qwen3-TTS VoiceDesign checkpoint.

    Where :class:`TTSEngine` needs a reference recording to clone, this model
    generates speech from a natural-language *description* - which is what lets a
    native accent anchor exist for every supported language without shipping a
    voice per language. It is used only to produce anchors, never the dubbed
    lines, so the base checkpoint and this one are never needed at the same time.
    """

    #: Only the VoiceDesign checkpoint implements ``generate_voice_design``.
    SUPPORTED_MODEL_TYPES = ("voice_design",)

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "auto",
        dtype: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._dtype = dtype
        self._model: Any | None = None

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
                "which does not support voice design. Use the '-VoiceDesign' "
                "checkpoint (download with: python dub.py models --download "
                "tts_voice_design)."
            )
        return model

    def generate(
        self,
        text: str,
        instruct: str,
        language: str,
    ) -> tuple[np.ndarray, int]:
        """Synthesise ``text`` in the style ``instruct`` describes.

        Returns:
            ``(samples, sample_rate)`` for a single mono clip.

        Raises:
            TTSError: If the model returns nothing usable.
        """
        wavs, sample_rate = self.model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
            non_streaming_mode=True,
        )
        if not wavs:
            raise TTSError("The voice-design model returned no audio.")
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise TTSError("The voice-design model returned an empty clip.")
        return samples, int(sample_rate)


class CustomVoiceEngine:
    """Speech synthesiser backed by the Qwen3-TTS CustomVoice checkpoint.

    CustomVoice is a *named-speaker* model: it has nine preset voices and, on the
    1.7B checkpoint, accepts a natural-language ``instruct`` that controls tone,
    emotion and style. It is the one Qwen model that pairs a fixed voice with
    instruction-based delivery, which is what makes it usable as a ``say``
    backend without a reference clip. It does not clone or design a voice.

    Note:
        The 0.6B CustomVoice checkpoint ignores ``instruct``; this engine targets
        the 1.7B checkpoint the registry points at.
    """

    #: Only the CustomVoice checkpoint implements ``generate_custom_voice``.
    SUPPORTED_MODEL_TYPES = ("custom_voice",)

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "auto",
        dtype: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._dtype = dtype
        self._model: Any | None = None

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
                "which does not support custom voices. Download it with: "
                "python dub.py models --download tts_custom_voice"
            )
        return model

    def speakers(self) -> list[str]:
        """The preset speaker names this checkpoint exposes, if any."""
        try:
            names = self.model.get_supported_speakers()
        except Exception:  # noqa: BLE001 - a checkpoint may not expose the list
            return []
        return [str(name) for name in (names or [])]

    def generate(
        self,
        text: str,
        speaker: str,
        instruct: str,
        language: str,
    ) -> tuple[np.ndarray, int]:
        """Synthesise ``text`` in ``speaker``'s voice, styled by ``instruct``.

        Returns:
            ``(samples, sample_rate)`` for a single mono clip.

        Raises:
            TTSError: If the model returns nothing usable.
        """
        wavs, sample_rate = self.model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct or None,
            non_streaming_mode=True,
        )
        if not wavs:
            raise TTSError("The custom-voice model returned no audio.")
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise TTSError("The custom-voice model returned an empty clip.")
        return samples, int(sample_rate)
