"""Speaker diarization: who speaks when, without a speaker-labelled transcript.

The pipeline is per-segment already, so all this stage has to produce is a
*label* per segment. It works by embedding each ASR segment and clustering the
embeddings, which avoids a second segmentation pass and keeps the labels aligned
with the text by construction: a segment is never split, it is assigned.

The embedder is pluggable. ``funasr`` (ERes2NetV2) is the default because it is
the only open, pip-installable speaker-verification model that actually
discriminates here - measured on the repo's two-speaker clip, its pairwise
cosines span 0.34-0.87, where the Qwen3-TTS cloning encoder spans 0.96-0.99 and
cannot separate the two voices at all.

Everything except :meth:`FunASREmbedder.embed` is pure and unit-tested without a
model, so the clustering can be reasoned about on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .errors import DependencyError
from .media import resample

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_THRESHOLD",
    "MIN_SPEAKER_S",
    "SPEAKER_RATE",
    "NullDiarizer",
    "EmbeddingDiarizer",
    "FunASREmbedder",
    "SpeakerEmbedder",
    "cluster_speakers",
    "detect_diarizer",
    "single_speaker_labels",
]

#: Speaker models expect 16 kHz, whatever the pipeline's 48 kHz working rate is.
SPEAKER_RATE = 16_000

#: The only speaker-verification checkpoint registered by funasr 1.4.x.
DEFAULT_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

#: Cosine *distance* at which two cluster averages are considered different
#: voices, for average-linkage clustering. Calibrated on the repo's two-speaker
#: clip, where the merge profile was flat to 0.379 and then jumped to 0.468 - so
#: anything in that gap returns the same two speakers. That plateau is what makes
#: a constant defensible; ``--speaker-threshold`` exposes it for other material.
DEFAULT_THRESHOLD = 0.40

#: A cluster shorter than this cannot supply a usable clone prompt - the donor
#: window starts at 3 s - so it is folded into the nearest other cluster instead
#: of becoming a speaker with no voice.
MIN_SPEAKER_S = 3.0

#: Shortest segment worth embedding; below this the embedding is mostly noise.
MIN_EMBED_S = 0.4


class SpeakerEmbedder(Protocol):
    """Turns audio regions into speaker embeddings."""

    def embed(
        self,
        samples: np.ndarray,
        sample_rate: int,
        regions: Sequence[tuple[float, float]],
    ) -> list[np.ndarray | None]:
        """Embed each ``(start, end)`` second region; ``None`` where unusable."""
        ...


def single_speaker_labels(segment_ids: Sequence[int], speaker: str = "SPK_0") -> dict[int, str]:
    """Every segment belongs to one speaker - the no-diarization answer."""
    return {segment_id: speaker for segment_id in segment_ids}


def _as_vector(value: Any) -> np.ndarray | None:
    """Flatten an embedding to a finite float vector, or ``None`` if unusable."""
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
    if float(np.linalg.norm(vector)) == 0.0:
        return None
    return vector


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity, ``NaN`` when the vectors cannot be compared."""
    if left.shape != right.shape:
        return float("nan")
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _similarity_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Full pairwise cosine similarity; embeddings are not assumed normalised."""
    matrix = np.eye(len(vectors), dtype=np.float64)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            value = _cosine(vectors[i], vectors[j])
            matrix[i, j] = matrix[j, i] = value
    return matrix


def cluster_speakers(
    embeddings: Sequence[np.ndarray | None],
    durations: Sequence[float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_seconds: float = MIN_SPEAKER_S,
) -> list[str]:
    """Cluster per-segment embeddings into speakers.

    Average-linkage agglomerative clustering on cosine distance, then any cluster
    too short to clone is folded into its nearest neighbour, and finally labels are
    assigned in order of first appearance so they are stable run to run.

    Args:
        embeddings: One embedding per segment; ``None``/unusable rows get ``""``.
        durations: Segment length in seconds, used for the short-cluster merge.
        threshold: Cosine distance above which clusters stay separate.
        min_seconds: Clusters shorter than this are merged away.

    Returns:
        One label per input row, ``"SPK_0"`` upwards, or ``""`` for a row that
        could not be embedded (the caller fills those in).

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(embeddings) != len(durations):
        raise ValueError("embeddings and durations must be the same length")

    vectors: dict[int, np.ndarray] = {}
    for index, value in enumerate(embeddings):
        vector = _as_vector(value)
        if vector is not None and durations[index] >= MIN_EMBED_S:
            vectors[index] = vector

    labels = ["" for _ in embeddings]
    if not vectors:
        return labels

    rows = list(vectors)
    matrix = _similarity_matrix([vectors[index] for index in rows])

    # Average-linkage AHC. Segment counts here are in the hundreds at most, and
    # the O(n^3) cost is trivial next to embedding, so clarity wins over thrift.
    clusters: list[list[int]] = [[position] for position in range(len(rows))]
    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pairs = [matrix[a, b] for a in clusters[i] for b in clusters[j]]
                distance = 1.0 - float(np.mean(pairs))
                if best is None or distance < best[0]:
                    best = (distance, i, j)
        assert best is not None  # len(clusters) > 1
        distance, i, j = best
        if distance > threshold:
            break
        clusters = [
            cluster for position, cluster in enumerate(clusters) if position not in (i, j)
        ] + [clusters[i] + clusters[j]]

    # A voice needs enough speech to clone. Fold the shortest sub-threshold cluster
    # into whichever cluster is closest, repeatedly, until none are left.
    def seconds(cluster: Sequence[int]) -> float:
        return sum(durations[rows[position]] for position in cluster)

    while len(clusters) > 1:
        small = [k for k, cluster in enumerate(clusters) if seconds(cluster) < min_seconds]
        if not small:
            break
        k = small[0]
        others = [
            (1.0 - float(np.mean([matrix[a, b] for a in clusters[k] for b in clusters[o]])), o)
            for o in range(len(clusters))
            if o != k
        ]
        _, target = min(others)
        # Merged before either index is dropped: removing the lower index first
        # would shift the higher one, so the merged cluster would land in the
        # wrong slot (or off the end) whenever the neighbour sits before it.
        merged = clusters[k] + clusters[target]
        for position in sorted((k, target), reverse=True):
            clusters.pop(position)
        clusters.append(merged)

    # Label by first appearance, so SPK_0 is the speaker who talks first and the
    # numbering does not shuffle between runs.
    clusters.sort(key=min)
    for number, cluster in enumerate(clusters):
        for position in cluster:
            labels[rows[position]] = f"SPK_{number}"
    return labels


class NullDiarizer:
    """Everyone is one speaker - the behaviour when diarization is off."""

    name = "none"

    def label(self, segment_ids: Sequence[int], *_: Any, **__: Any) -> dict[int, str]:
        """See :meth:`EmbeddingDiarizer.label`."""
        return single_speaker_labels(segment_ids)


@dataclass
class EmbeddingDiarizer:
    """Labels segments by clustering their speaker embeddings."""

    embedder: SpeakerEmbedder
    threshold: float = DEFAULT_THRESHOLD
    min_seconds: float = MIN_SPEAKER_S
    name: str = "embedding"

    def label(
        self,
        segment_ids: Sequence[int],
        regions: Sequence[tuple[float, float]],
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[int, str]:
        """Assign a speaker to each segment.

        Args:
            segment_ids: Ids in the same order as ``regions``.
            regions: ``(start, end)`` seconds per segment.
            samples: The speech stem the regions index into.
            sample_rate: Its rate.

        Returns:
            Segment id to label. A segment whose embedding failed inherits the
            previous segment's speaker, which keeps a two-person exchange from
            sprouting a third voice out of one bad 2-second window.
        """
        embeddings = self.embedder.embed(samples, sample_rate, regions)
        durations = [max(0.0, end - start) for start, end in regions]
        labels = cluster_speakers(
            embeddings, durations, threshold=self.threshold, min_seconds=self.min_seconds
        )

        resolved: dict[int, str] = {}
        previous = ""
        for segment_id, label in zip(segment_ids, labels):
            if not label:
                label = previous
            if label:
                previous = label
            resolved[segment_id] = label

        # Nothing clustered (every embedding failed): one speaker is the honest
        # answer, and it keeps the rest of the pipeline on its normal path.
        if not any(resolved.values()):
            _LOGGER.warning(
                "No segment could be embedded, so the whole video is treated as one "
                "speaker. Check the speech stem is not silent, and see the warning "
                "above for the underlying error. Pass --diarizer none to silence this."
            )
            return single_speaker_labels(segment_ids)

        # A leading run of failures has nothing to inherit from: give it the first
        # speaker that was actually assigned.
        first = next(label for label in resolved.values() if label)
        return {segment_id: label or first for segment_id, label in resolved.items()}


class FunASREmbedder:
    """Speaker embeddings from funasr's ERes2NetV2 speaker-verification model.

    The weights come from ModelScope on first use (~30 MB) and are cached under
    ``~/.cache/modelscope``.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> None:
        # Imported locally because config.py imports this module for its defaults.
        from .config import resolve_device

        self.model_id = model_id
        #: Resolved here rather than passed through: funasr hands the string to
        #: torch, which has no "auto" device, and it fails *after* loading the
        #: weights - so the symptom is a silent total failure rather than an error.
        self.device = resolve_device(device)
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Load the model on first use and keep it resident."""
        if self._model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:  # pragma: no cover - config dependent
                raise DependencyError(
                    "funasr is not installed, so speaker diarization is unavailable. "
                    "Install it with: pip install -r requirements.txt, or pass "
                    "--diarizer none to dub every line in one voice."
                ) from exc
            self._model = AutoModel(
                model=self.model_id, device=self.device, disable_update=True
            )
        return self._model

    def embed(
        self,
        samples: np.ndarray,
        sample_rate: int,
        regions: Sequence[tuple[float, float]],
    ) -> list[np.ndarray | None]:
        """Embed each region; see :class:`SpeakerEmbedder`."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if int(sample_rate) != SPEAKER_RATE:
            audio = resample(audio, int(sample_rate), SPEAKER_RATE)

        # Built once, and outside the per-window handler: a model that fails to
        # load is a broken install, not a bad two-second window, and absorbing it
        # here would look exactly like "this video has one voice".
        model = self.model

        results: list[np.ndarray | None] = []
        failures = 0
        for start, end in regions:
            first = max(0, int(round(start * SPEAKER_RATE)))
            last = min(audio.size, int(round(end * SPEAKER_RATE)))
            if last - first < int(MIN_EMBED_S * SPEAKER_RATE):
                results.append(None)
                continue
            try:
                output = model.generate(input=audio[first:last], fs=SPEAKER_RATE)
                results.append(_as_vector(output[0].get("spk_embedding")))
            except Exception as exc:  # noqa: BLE001 - one bad window must not fail a job
                failures += 1
                # The first failure is reported at warning level: a silent, total
                # failure is indistinguishable from a one-voice video. The rest
                # stay at debug so a mostly-bad stem does not flood the log.
                log = _LOGGER.warning if failures == 1 else _LOGGER.debug
                log("Could not embed %.2f-%.2f s: %s", start, end, exc)
                results.append(None)
        return results


def detect_diarizer(
    preference: str = "auto",
    *,
    device: str = "auto",
    threshold: float = DEFAULT_THRESHOLD,
    min_seconds: float = MIN_SPEAKER_S,
) -> NullDiarizer | EmbeddingDiarizer:
    """Pick a diarizer, degrading to single-speaker rather than failing.

    Args:
        preference: ``auto`` tries the embedding model, ``none`` forces one speaker.
        device: Torch device for the embedding model.
        threshold: Cosine distance between speakers.
        min_seconds: Shortest cluster worth keeping as its own speaker.

    Returns:
        An :class:`EmbeddingDiarizer`, or :class:`NullDiarizer` when diarization is
        off or unavailable - so a missing optional dependency downgrades the result
        instead of failing the job.
    """
    if str(preference).strip().lower() in {"none", "off", "false"}:
        return NullDiarizer()
    try:
        import funasr  # noqa: F401  - availability probe only
    except ImportError as exc:
        _LOGGER.warning(
            "funasr is not installed, so every line will be dubbed in one voice. "
            "Install it (pip install -r requirements.txt) for multi-speaker dubbing: %s",
            exc,
        )
        return NullDiarizer()
    return EmbeddingDiarizer(
        embedder=FunASREmbedder(device=device),
        threshold=threshold,
        min_seconds=min_seconds,
    )
