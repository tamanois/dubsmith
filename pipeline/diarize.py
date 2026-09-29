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
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .errors import DependencyError
from .media import resample

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_THRESHOLD",
    "MIN_OVERLAP_S",
    "MIN_SPEAKER_S",
    "OVERLAP_HOP_S",
    "OVERLAP_WINDOW_S",
    "SPEAKER_RATE",
    "NullDiarizer",
    "EmbeddingDiarizer",
    "FunASREmbedder",
    "SpeakerEmbedder",
    "cluster_speakers",
    "detect_diarizer",
    "assign_streams",
    "single_speaker_labels",
]

#: Speaker models expect 16 kHz, whatever the pipeline's 48 kHz working rate is.
SPEAKER_RATE = 16_000

#: The only speaker-verification checkpoint registered by funasr 1.4.x.
DEFAULT_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

#: Cosine *distance* at which two cluster averages are considered different
#: voices, for average-linkage clustering. Measured, not assumed: on the 11-minute
#: job the previously calibrated 0.40 split the video into 42 voices, while 0.70
#: returned the 5 people actually in it - and a listener confirmed that as the
#: point where the extra voices stopped appearing. It holds up as a constant
#: because real material plateaus rather than slides (the repo's two-speaker clip
#: was flat to 0.379 and then jumped to 0.468), so a value inside the plateau is
#: no more arbitrary than one at its edge; ``--speaker-threshold`` exposes it for
#: other material.
DEFAULT_THRESHOLD = 0.7

#: A cluster shorter than this cannot supply a usable clone prompt - the donor
#: window starts at 3 s - so it is folded into the nearest other cluster instead
#: of becoming a speaker with no voice.
MIN_SPEAKER_S = 3.0

#: Shortest segment worth embedding; below this the embedding is mostly noise.
MIN_EMBED_S = 0.4

#: Window used to look for a second voice inside a segment. Short windows are
#: unusable here: on the repo's two-person clip a 1 s window splits one speaker's
#: *own* voice by up to 1.08 cosine distance, versus 0.34 between the two
#: speakers, so a short-window "split" reports embedding noise rather than a
#: second voice. Same-speaker spread falls with window length (0.88 at 1.5 s,
#: 0.85 at 2 s, 0.63 at 2.5 s) and so does the false-flip rate (6/13, 3/13,
#: 1/13), so 2.5 s is the shortest window at which noise stops driving the answer.
OVERLAP_WINDOW_S = 2.5

#: Hop between consecutive overlap windows.
OVERLAP_HOP_S = 1.25

#: A different voice must own at least this much of a segment, in one consecutive
#: run of windows, before the segment counts as overlapped. It is a floor for
#: short segments - a window is already longer than this - so an interruption
#: shorter than about a window is deliberately missed.
MIN_OVERLAP_S = 0.6


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


def _window_bounds(
    start: float,
    end: float,
    *,
    window: float = OVERLAP_WINDOW_S,
    hop: float = OVERLAP_HOP_S,
) -> list[tuple[float, float]]:
    """Cut ``[start, end]`` into overlapping windows for overlap detection.

    A region shorter than one window yields a single window covering it, so a
    short exchange is still compared rather than skipped; the final window is
    flush with the region's end so no speech is left unexamined.
    """
    length = end - start
    if length <= 0:
        return []
    if length <= window:
        return [(start, end)]
    bounds: list[tuple[float, float]] = []
    offset = start
    while offset + window < end:
        bounds.append((offset, offset + window))
        offset += hop
    bounds.append((end - window, end))
    return bounds


def _centroid(members: Sequence[np.ndarray]) -> np.ndarray | None:
    """Mean of a speaker's window embeddings (unit-normalised), or ``None``.

    Averaging many windows is the whole point: it cancels the per-window noise
    that makes comparing two short windows directly meaningless.
    """
    usable = [np.asarray(member, dtype=np.float64).reshape(-1) for member in members]
    usable = [member for member in usable if member.size and np.isfinite(member).all()]
    if not usable or len({member.size for member in usable}) != 1:
        return None
    centroid = np.mean(np.stack(usable, axis=0), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return None
    return centroid / norm


def _best_speaker(
    vector: np.ndarray, centroids: Mapping[str, np.ndarray | None]
) -> str | None:
    """The speaker whose centroid is closest to ``vector``, or ``None``."""
    best: str | None = None
    best_score = -np.inf
    for speaker, centroid in centroids.items():
        if centroid is None:
            continue
        score = _cosine(vector, centroid)
        if np.isfinite(score) and score > best_score:
            best, best_score = speaker, score
    return best


def _longest_other_span(
    windows: Sequence[tuple[float, float, bool | None]],
) -> float:
    """Longest consecutive run assigned to another speaker, in seconds.

    ``windows`` is ``(start, end, own)`` per window, where ``own`` is ``True``
    when the window fits its own speaker's centroid better, ``False`` when it
    fits another speaker's, and ``None`` when it could not be scored. The span is
    measured from the run's first window start to its last window end, so it
    counts the audio covered, not the number of windows.
    """
    longest = 0.0
    run_start: float | None = None
    for start, end, own in windows:
        if own is False:
            if run_start is None:
                run_start = start
            longest = max(longest, end - run_start)
        else:
            run_start = None
    return longest


def assign_streams(
    embeddings: Sequence[np.ndarray | None],
    dominant: str,
    centroids: Mapping[str, np.ndarray | None],
) -> list[str] | None:
    """Match two separated streams to speakers, or ``None`` when ambiguous.

    The dominant stream is the one closest to the segment's own speaker centroid;
    the other stream is matched to the best-scoring *different* speaker, and only
    accepted when it fits that speaker better than the dominant one. That last
    check is what stops a false-positive overlap flag - a segment one person
    actually said - from being split into two voices.

    Args:
        embeddings: One embedding per separated stream (exactly two).
        dominant: The speaker the diarization assigned the original segment.
        centroids: Mean embedding per speaker.

    Returns:
        One speaker per stream, aligned with ``embeddings``, or ``None`` when the
        streams cannot be told apart - in which case the caller keeps the
        dominant-only treatment rather than guessing.
    """
    vectors = [_as_vector(value) for value in embeddings]
    if len(vectors) != 2 or any(vector is None for vector in vectors):
        return None
    names = [name for name, centroid in centroids.items() if centroid is not None]
    if len(names) < 2 or dominant not in names:
        return None

    typed = [vector for vector in vectors if vector is not None]
    score = {
        (index, name): _cosine(typed[index], centroids[name])
        for index in range(2)
        for name in names
    }
    if not (np.isfinite(score[(0, dominant)]) and np.isfinite(score[(1, dominant)])):
        return None

    first, second = (0, 1)
    if score[(second, dominant)] > score[(first, dominant)]:
        first, second = second, first

    others = [name for name in names if name != dominant]
    match = max(others, key=lambda name: score[(second, name)])
    if not np.isfinite(score[(second, match)]):
        return None
    if score[(second, match)] <= score[(second, dominant)]:
        return None

    assignment = ["", ""]
    assignment[first] = dominant
    assignment[second] = match
    return assignment


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

    def centroids(
        self,
        segment_ids: Sequence[int],
        regions: Sequence[tuple[float, float]],
        labels: Mapping[int, str],
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[str, np.ndarray]:
        """Mean embedding per speaker, from the segments the labels assign.

        Used to match a separated stream back to a voice: a stream is compared to
        these centroids rather than to a single segment, so the same averaging that
        makes overlap detection stable applies here too.

        Args:
            segment_ids: Ids in the same order as ``regions``.
            regions: ``(start, end)`` seconds per segment.
            labels: Segment id to speaker.
            samples: The speech stem the regions index into.
            sample_rate: Its rate.

        Returns:
            Speaker to unit-normalised centroid, omitting speakers that could not
            be embedded.

        Raises:
            ValueError: If ``segment_ids`` and ``regions`` differ in length.
        """
        if len(segment_ids) != len(regions):
            raise ValueError("segment_ids and regions must be the same length")

        embeddings = self.embedder.embed(samples, sample_rate, regions)
        owned: dict[str, list[np.ndarray]] = {}
        for segment_id, value in zip(segment_ids, embeddings):
            vector = _as_vector(value)
            speaker = labels.get(segment_id)
            if vector is not None and speaker:
                owned.setdefault(str(speaker), []).append(vector)

        resolved: dict[str, np.ndarray] = {}
        for speaker, members in owned.items():
            centroid = _centroid(members)
            if centroid is not None:
                resolved[speaker] = centroid
        return resolved

    def overlaps(
        self,
        segment_ids: Sequence[int],
        regions: Sequence[tuple[float, float]],
        labels: Mapping[int, str],
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[int, bool]:
        """Flag the segments that carry a second voice.

        Supervised by the diarization labels: the labels already say who the
        speakers are, so each window is compared against a speaker *centroid*
        (the mean of that speaker's own windows) rather than against other
        windows. Averaging is what cancels the per-window noise that makes
        window-to-window comparison useless on this embedder. A segment is
        flagged only when a consecutive run of its windows fits *another*
        speaker's centroid better than its own, **and** its own speaker still
        owns the majority of the segment - a segment whose windows all match
        another voice is a mislabelled segment, which is the diarization stage's
        problem, not an overlap. A one-voice job cannot be flagged: with no
        second centroid, no window can match one.

        This is deliberately conservative. A false "overlap" sends a listener
        hunting a defect that is not there, and an interruption shorter than
        about a window is missed rather than guessed.

        Args:
            segment_ids: Ids in the same order as ``regions``.
            regions: ``(start, end)`` seconds per segment.
            labels: Segment id to the speaker the diarization assigned it.
            samples: The speech stem the regions index into.
            sample_rate: Its rate.

        Returns:
            Segment id to flag; a segment that cannot be scored is ``False``.

        Raises:
            ValueError: If ``segment_ids`` and ``regions`` differ in length.
        """
        if len(segment_ids) != len(regions):
            raise ValueError("segment_ids and regions must be the same length")

        flags = {segment_id: False for segment_id in segment_ids}
        speakers = {str(label) for label in labels.values() if label}
        if len(speakers) < 2:
            # No second voice to find, so nothing can be flagged. This is why a
            # noisy solo recording can never be mistaken for an overlap.
            return flags

        bounds: list[tuple[int, float, float]] = []
        for index, (start, end) in enumerate(regions):
            if not labels.get(segment_ids[index]):
                continue
            for window_start, window_end in _window_bounds(start, end):
                bounds.append((index, window_start, window_end))
        if not bounds:
            return flags

        embeddings = self.embedder.embed(
            samples, sample_rate, [(start, end) for _index, start, end in bounds]
        )
        if len(embeddings) != len(bounds):
            raise ValueError("embedder returned a different number of vectors")

        vectors = [_as_vector(value) for value in embeddings]

        # One centroid per speaker, from every window that speaker owns.
        owned: dict[str, list[np.ndarray]] = {}
        for (index, _start, _end), vector in zip(bounds, vectors):
            if vector is None:
                continue
            owned.setdefault(str(labels[segment_ids[index]]), []).append(vector)
        centroids: dict[str, np.ndarray | None] = {
            speaker: _centroid(members) for speaker, members in owned.items()
        }

        per_segment: dict[int, list[tuple[float, float, bool | None]]] = {}
        for position, (index, start, end) in enumerate(bounds):
            own = str(labels[segment_ids[index]])
            vector = vectors[position]
            if vector is None or centroids.get(own) is None:
                state: bool | None = None
            else:
                state = _best_speaker(vector, centroids) == own
            per_segment.setdefault(index, []).append((start, end, state))

        for index, windows in per_segment.items():
            scored = [own for _start, _end, own in windows if own is not None]
            if not scored:
                continue
            own_count = sum(1 for own in scored if own)
            # Another voice owns the majority: a labelling problem, not an overlap.
            if own_count * 2 <= len(scored):
                continue
            if _longest_other_span(windows) >= MIN_OVERLAP_S:
                flags[segment_ids[index]] = True
        return flags


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
