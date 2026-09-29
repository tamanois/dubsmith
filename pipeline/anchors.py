"""Generated accent anchors for the ``native`` accent mode.

Accent and timbre are stored separately by the checkpoint: pronunciation and
prosody ride in the reference *speech codes*, while vocal identity rides in the
*speaker embedding*. The ``native`` mode exploits that split by taking
pronunciation from a short native-language narration - the *anchor* - and
timbre from the video's own speaker.

An anchor is language-only, so it is generated once and cached under
``<models_dir>/anchors/`` and shared by every job. With ``--accent-match`` a job
can instead ask for a per-speaker anchor whose delivery (speaking rate, and
optionally the speaker's gender) fits the source voice; those live beside the
plain ones, keyed by the descriptor that was matched.

This module holds the pure analysis (rate and gender measurement, keying) and the
cache-aware generation. It never imports a model: the synthesiser is injected, so
every function here is testable without a GPU or downloaded weights.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .asr import Segment
from .config import NAME_TO_CODE
from .errors import TTSError

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "ANCHOR_TEXT",
    "AnchorSpec",
    "anchor_paths",
    "ensure_anchor",
    "instruct_for",
    "list_anchors",
    "measure_gender",
    "measure_rate",
    "rate_band",
]


# --------------------------------------------------------------------------- #
# Anchor content
# --------------------------------------------------------------------------- #

#: One neutral, phonetically varied passage per supported language. These are
#: *text*, not voices: the model synthesises them, which is what lets ten
#: languages share a single checkpoint instead of shipping ten recordings. Keep
#: each near 15 seconds of speech - long enough for stable codes, short enough
#: to generate quickly.
ANCHOR_TEXT: dict[str, str] = {
    "English": (
        "Clear speech carries an idea further than volume ever can. When we speak "
        "to someone, we share not only words but rhythm, emphasis and intent. A "
        "steady pace gives the listener time to follow, while a lively tone keeps "
        "attention from drifting. This short passage exists to sound natural in "
        "ordinary conversation."
    ),
    "French": (
        "Une parole claire porte une idée plus loin que la simple force de la voix. "
        "Lorsque nous parlons à quelqu'un, nous partageons non seulement des mots, "
        "mais aussi un rythme, des accents et une intention. Un débit régulier "
        "laisse à l'auditeur le temps de suivre, tandis qu'un ton vivant retient son "
        "attention. Ce court passage existe pour sonner naturel dans une "
        "conversation ordinaire."
    ),
    "German": (
        "Klare Sprache trägt eine Idee weiter als bloße Lautstärke. Wenn wir mit "
        "jemandem sprechen, teilen wir nicht nur Wörter, sondern auch Rhythmus, "
        "Betonung und Absicht. Ein ruhiges Tempo gibt dem Zuhörer Zeit zu folgen, "
        "während ein lebendiger Ton die Aufmerksamkeit wachhält. Dieser kurze "
        "Abschnitt soll in einem gewöhnlichen Gespräch natürlich klingen."
    ),
    "Spanish": (
        "El habla clara lleva una idea más lejos que el simple volumen. Cuando "
        "hablamos con alguien compartimos no solo palabras, sino también ritmo, "
        "énfasis e intención. Un ritmo constante da al oyente tiempo para seguir, "
        "mientras que un tono vivo mantiene su atención. Este breve texto existe "
        "para sonar natural en una conversación corriente."
    ),
    "Italian": (
        "Una voce chiara porta un'idea più lontano del semplice volume. Quando "
        "parliamo con qualcuno condividiamo non solo parole, ma anche ritmo, enfasi "
        "e intenzione. Un ritmo costante dà all'ascoltatore il tempo di seguire, "
        "mentre un tono vivace mantiene viva l'attenzione. Questo breve brano esiste "
        "per suonare naturale in una conversazione comune."
    ),
    "Portuguese": (
        "A fala clara leva uma ideia mais longe do que o simples volume. Quando "
        "falamos com alguém, compartilhamos não apenas palavras, mas também ritmo, "
        "ênfase e intenção. Um ritmo constante dá ao ouvinte tempo para acompanhar, "
        "enquanto um tom vivo mantém a atenção. Este pequeno texto existe para soar "
        "natural numa conversa comum."
    ),
    "Russian": (
        "Ясная речь доносит мысль дальше, чем одна только громкость. Когда мы "
        "говорим с кем-то, мы делимся не только словами, но и ритмом, ударением и "
        "намерением. Ровный темп даёт слушателю время следить за мыслью, а живая "
        "интонация удерживает внимание. Этот короткий отрывок нужен, чтобы звучать "
        "естественно в обычном разговоре."
    ),
    "Chinese": (
        "清晰的语言比单纯的音量更能传达想法。当我们与人交谈时，分享的不只是词句，"
        "还有节奏、重音和意图。平稳的语速让听者有时间跟上，而富有活力的语调能保持"
        "他们的注意。这段简短的文字，是为了在日常对话中听起来自然而写。"
    ),
    "Japanese": (
        "明確な話し方は、単なる声の大きさよりも考えを遠くまで届けます。誰かと話す"
        "とき、私たちは言葉だけでなく、リズムや強調、意図も分かち合っています。"
        "安定した速さは聞き手に追う時間を与え、生き生きとした抑揚は注意を引きつけ"
        "ます。この短い文章は、ふつうの会話で自然に聞こえるように書かれています。"
    ),
    "Korean": (
        "분명한 말은 단순한 목소리 크기보다 생각을 더 멀리 전합니다. 누군가와 "
        "이야기할 때 우리는 말뿐 아니라 리듬과 강세, 의도까지 나눕니다. 일정한 "
        "속도는 듣는 사람에게 따라갈 시간을 주고, 생기 있는 억양은 주의를 붙잡습니다. "
        "이 짧은 글은 평범한 대화에서 자연스럽게 들리도록 쓰였습니다."
    ),
}

#: Base style instruction. Kept in English because the instruct is a
#: natural-language style prompt rather than target-language text; the target
#: phonetics come from the ``language`` argument and the anchor text itself.
_BASE_INSTRUCT = (
    "a warm, clear native {language} speaker with a natural news-reader delivery"
)

#: Delivery phrases per measured rate band. The band, not a raw words-per-second
#: figure, is what the model can act on: it is a style, not a number.
_RATE_PHRASES: dict[str, str] = {
    "slow": "speaking slowly and deliberately",
    "normal": "at a natural conversational pace",
    "fast": "at a brisk, energetic pace",
}

_GENDER_PHRASES: dict[str, str] = {
    "f": "female voice",
    "m": "male voice",
}

#: Words per second below/above which the delivery counts as slow/fast. Initial
#: constants to validate on real clips, like the speaker threshold.
SLOW_WPS = 3.0
FAST_WPS = 4.5

#: Median F0 above which a voice reads as female. Coarse on purpose: this only
#: picks a delivery hint, so an uncertain answer stays ``"u"`` rather than
#: guessing.
GENDER_F0_HZ = 165.0
#: Minimum voiced frames (at the analysis rate) before a gender is committed to.
_MIN_VOICED = 20


# --------------------------------------------------------------------------- #
# Language-agnostic analysis
# --------------------------------------------------------------------------- #


def measure_rate(segments: Sequence[Segment]) -> float:
    """Speaking rate in words per second for a set of a speaker's segments.

    Prefers word timestamps, which exclude the pauses between words; falls back to
    segment durations when the transcript carries none. Returns ``0.0`` when there
    is nothing to measure, which callers treat as "no opinion" rather than a rate.
    """
    words = 0
    spoken = 0.0
    for segment in segments:
        if segment.words:
            words += len(segment.words)
            spoken += sum(max(0.0, w.end - w.start) for w in segment.words)
        else:
            words += len(segment.text.split())
            spoken += segment.duration
    if words <= 0 or spoken <= 0.0:
        return 0.0
    return words / spoken


def rate_band(wps: float) -> str:
    """Map a words-per-second figure onto ``slow``/``normal``/``fast``.

    A non-positive rate means "not measured", which is reported as ``normal`` so
    the anchor falls back to the plain per-language delivery.
    """
    if wps <= 0.0:
        return "normal"
    if wps < SLOW_WPS:
        return "slow"
    if wps > FAST_WPS:
        return "fast"
    return "normal"


def measure_gender(samples: np.ndarray, sample_rate: int) -> str:
    """Best-effort ``"f"``/``"m"``/``"u"`` for a speaker's audio.

    A coarse median-F0 estimate; ``"u"`` (unknown) whenever librosa is missing,
    the clip is too short, or too few frames are voiced. Only used as an optional
    delivery hint, so "unknown" is a perfectly good answer.
    """
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0 or sample_rate <= 0:
        return "u"
    if data.size < int(0.2 * sample_rate):
        return "u"
    try:
        import librosa

        f0 = librosa.yin(data, fmin=65.0, fmax=400.0, sr=int(sample_rate))
    except Exception as exc:  # noqa: BLE001 - a hint must never fail a job
        _LOGGER.debug("Could not estimate pitch: %s", exc)
        return "u"
    voiced = np.asarray(f0, dtype=np.float64)
    voiced = voiced[np.isfinite(voiced)]
    if voiced.size < _MIN_VOICED:
        return "u"
    return "m" if float(np.median(voiced)) < GENDER_F0_HZ else "f"


# --------------------------------------------------------------------------- #
# Anchor specification and cache
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnchorSpec:
    """Which anchor to use: a language, plus optional matched descriptors.

    ``rate_band`` and ``gender`` are ``None`` for the plain per-language anchor.
    """

    language: str
    rate_band: str | None = None
    gender: str | None = None

    @property
    def key(self) -> str:
        """Deterministic cache filename stem, e.g. ``fr`` or ``fr_normal_f``."""
        parts = [NAME_TO_CODE.get(self.language, self.language.lower()[:2])]
        if self.rate_band:
            parts.append(self.rate_band)
        if self.gender and self.gender != "u":
            parts.append(self.gender)
        return "_".join(parts)

    @property
    def text(self) -> str:
        """The passage to synthesise."""
        try:
            return ANCHOR_TEXT[self.language]
        except KeyError:
            raise TTSError(
                f"No accent anchor text for {self.language!r}. Add one to "
                "pipeline/anchors.ANCHOR_TEXT."
            ) from None

    @property
    def instruct(self) -> str:
        """The style instruction describing the anchor voice."""
        return instruct_for(self.language, self.rate_band, self.gender)


def instruct_for(
    language: str,
    rate_band: str | None = None,
    gender: str | None = None,
) -> str:
    """Compose the VoiceDesign instruction for a language and matched hints."""
    parts = [_BASE_INSTRUCT.format(language=language)]
    if gender in _GENDER_PHRASES:
        parts.append(_GENDER_PHRASES[gender])
    if rate_band in _RATE_PHRASES:
        parts.append(_RATE_PHRASES[rate_band])
    return ", ".join(parts)


def anchor_paths(anchor_dir: Path | str, spec: AnchorSpec) -> tuple[Path, Path]:
    """The ``(wav, txt)`` pair for an anchor, whether or not it exists yet."""
    root = Path(anchor_dir)
    return root / f"{spec.key}.wav", root / f"{spec.key}.txt"


def ensure_anchor(
    anchor_dir: Path | str,
    spec: AnchorSpec,
    engine_factory: Callable[[], Any],
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Return the anchor's ``(wav, txt)``, generating it once and caching it.

    ``engine_factory`` is called *only* on a cache miss, so a cached anchor never
    loads the VoiceDesign checkpoint. The factory may raise (e.g. a missing model);
    that propagates as the actionable error the CLI prints.
    """
    wav, txt = anchor_paths(anchor_dir, spec)
    if not force and wav.is_file() and txt.is_file():
        return wav, txt

    engine = engine_factory()
    samples, sample_rate = engine.generate(spec.text, spec.instruct, spec.language)
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise TTSError(
            "The voice-design model returned no audio for the accent anchor."
        )

    from .media import write_audio

    write_audio(wav, samples, sample_rate)
    txt.write_text(spec.text, encoding="utf-8")
    (Path(anchor_dir) / f"{spec.key}.json").write_text(
        json.dumps(
            {
                "language": spec.language,
                "rate_band": spec.rate_band,
                "gender": spec.gender,
                "instruct": spec.instruct,
                "seconds": round(samples.size / float(sample_rate or 1), 2),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _LOGGER.info(
        "Generated %s accent anchor (%s, %.1fs)",
        spec.language,
        spec.key,
        samples.size / float(sample_rate or 1),
    )
    return wav, txt


def list_anchors(anchor_dir: Path | str) -> list[tuple[str, Path, float]]:
    """Every cached anchor as ``(key, wav_path, seconds)``, sorted by key."""
    root = Path(anchor_dir)
    if not root.is_dir():
        return []
    found: list[tuple[str, Path, float]] = []
    for wav in sorted(root.glob("*.wav")):
        seconds = 0.0
        try:
            import soundfile as sf

            info = sf.info(str(wav))
            seconds = float(info.frames) / float(info.samplerate or 1)
        except Exception:  # noqa: BLE001 - listing must tolerate a bad file
            seconds = 0.0
        found.append((wav.stem, wav, seconds))
    return found
