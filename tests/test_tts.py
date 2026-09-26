"""Tests for donor selection and the voice-cloning engine wrapper."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pipeline.asr import Segment
from pipeline.errors import TTSError
from pipeline.tts import (
    BlendedReference,
    DonorClip,
    TTSEngine,
    _donor_ratio,
    _drop_truncated,
    resolve_dtype,
    score_candidate,
    select_donor,
)

SR = 24_000


def audio(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    """Deterministic non-silent signal."""
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


class TestDonorClip:
    def test_duration_and_rms(self) -> None:
        clip = DonorClip(
            samples=audio(2.0, 0.5), sample_rate=SR, text="hi", start=0.0, end=2.0, segment_id=0
        )
        assert clip.duration == pytest.approx(2.0, abs=0.01)
        assert clip.rms == pytest.approx(0.354, abs=0.02)

    def test_empty_samples_have_zero_rms(self) -> None:
        clip = DonorClip(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=SR,
            text="",
            start=0.0,
            end=0.0,
            segment_id=0,
        )
        assert clip.rms == 0.0


class TestScoreCandidate:
    def test_ideal_length_beats_a_too_short_segment(self) -> None:
        samples = audio(20.0)
        ideal = Segment(id=0, start=0.0, end=8.0, text="a")
        short = Segment(id=1, start=10.0, end=11.0, text="b")
        assert score_candidate(ideal, samples, SR) > score_candidate(short, samples, SR)

    def test_louder_segment_wins_at_equal_length(self) -> None:
        quiet = audio(10.0, amplitude=0.001)
        loud = audio(10.0, amplitude=0.5)
        segment = Segment(id=0, start=0.0, end=5.0, text="a")
        assert score_candidate(segment, loud, SR) > score_candidate(segment, quiet, SR)

    def test_high_no_speech_probability_is_penalised(self) -> None:
        samples = audio(10.0)
        clean = Segment(id=0, start=0.0, end=5.0, text="a", no_speech_prob=0.0)
        noisy = Segment(id=0, start=0.0, end=5.0, text="a", no_speech_prob=0.9)
        assert score_candidate(clean, samples, SR) > score_candidate(noisy, samples, SR)

    def test_zero_length_scores_negative_infinity(self) -> None:
        segment = Segment(id=0, start=1.0, end=1.0, text="a")
        assert score_candidate(segment, audio(2.0), SR) == float("-inf")


class TestSelectDonor:
    def test_prefers_a_segment_inside_the_window(self) -> None:
        samples = audio(20.0)
        segments = [
            Segment(id=0, start=0.0, end=1.5, text="too short"),
            Segment(id=1, start=4.0, end=12.0, text="ideal"),
        ]
        donor = select_donor(samples, SR, segments)
        assert donor.segment_id == 1
        assert donor.text == "ideal"
        assert donor.duration == pytest.approx(8.0, abs=0.01)

    def test_falls_back_to_longest_when_none_are_in_range(self) -> None:
        samples = audio(10.0)
        segments = [
            Segment(id=0, start=0.0, end=0.5, text="tiny"),
            Segment(id=1, start=1.0, end=2.5, text="longer"),
        ]
        donor = select_donor(samples, SR, segments)
        assert donor.segment_id == 1

    def test_slices_the_requested_time_range(self) -> None:
        samples = audio(10.0)
        segments = [Segment(id=0, start=2.0, end=6.0, text="clip")]
        donor = select_donor(samples, SR, segments)
        assert donor.start == 2.0
        assert len(donor.samples) == pytest.approx(4.0 * SR, abs=2)

    def test_raises_without_segments(self) -> None:
        with pytest.raises(TTSError, match="no segments"):
            select_donor(audio(1.0), SR, [])

    def test_raises_when_segment_has_no_samples(self) -> None:
        segments = [Segment(id=0, start=100.0, end=110.0, text="beyond audio")]
        with pytest.raises(TTSError, match="no audio samples"):
            select_donor(audio(1.0), SR, segments)


class FakeQwenModel:
    """Stands in for ``Qwen3TTSModel``."""

    def __init__(self, *, sample_rate: int = SR, spoken: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.spoken = spoken
        self.prompts: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.model = SimpleNamespace(tts_model_type="base")

    def create_voice_clone_prompt(self, **kwargs: Any) -> str:
        self.prompts.append(kwargs)
        return "prompt-token"

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.calls.append(kwargs)
        texts = kwargs["text"]
        count = self.spoken if self.spoken is not None else len(texts)
        return [audio(0.4) for _ in range(count)], self.sample_rate


def engine_with(model: FakeQwenModel) -> TTSEngine:
    engine = TTSEngine("models/fake", device="cpu")
    engine._model = model  # inject, bypassing the heavy loader
    return engine


class TestTTSEngine:
    def test_synthesize_requires_a_prompt_first(self) -> None:
        engine = engine_with(FakeQwenModel())
        with pytest.raises(TTSError, match="prepare_prompt"):
            engine.synthesize(["bonjour"], language="French")

    def test_prepare_prompt_from_donor_passes_arrays(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt(
            DonorClip(samples=audio(3.0), sample_rate=SR, text="hello", start=0, end=3, segment_id=0)
        )
        kwargs = model.prompts[0]
        assert kwargs["ref_text"] == "hello"
        assert kwargs["x_vector_only_mode"] is False
        assert isinstance(kwargs["ref_audio"], tuple)

    def test_prepare_prompt_from_file_requires_text(self) -> None:
        engine = engine_with(FakeQwenModel())
        with pytest.raises(TTSError, match="ref_text is required"):
            engine.prepare_prompt_from_file("ref.wav", "   ")

    def test_blank_lines_map_to_none(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        result = engine.synthesize(["un", "   ", "trois"], language="French")
        assert result[1] is None
        assert result[0] is not None and result[2] is not None
        # Only real lines are sent to the model.
        assert model.calls[0]["text"] == ["un", "trois"]

    def test_all_blank_lines_skip_the_model(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        assert engine.synthesize(["", "  "], language="French") == [None, None]
        assert model.calls == []

    def test_empty_input_returns_empty(self) -> None:
        engine = engine_with(FakeQwenModel())
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        assert engine.synthesize([], language="French") == []

    def test_batches_requests(self) -> None:
        model = FakeQwenModel()
        engine = TTSEngine("models/fake", device="cpu", batch_size=2)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        result = engine.synthesize(["a", "b", "c"], language="French")
        assert len(model.calls) == 2
        assert model.calls[0]["text"] == ["a", "b"]
        assert model.calls[1]["text"] == ["c"]
        assert all(item is not None for item in result)

    def test_language_is_forwarded(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        engine.synthesize(["a"], language="French")
        assert model.calls[0]["language"] == ["French"]

    def test_wrong_output_count_raises(self) -> None:
        model = FakeQwenModel(spoken=1)
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        with pytest.raises(TTSError, match="returned 1 clips"):
            engine.synthesize(["a", "b"], language="French")

    def test_sample_rate_available_after_synthesis(self) -> None:
        model = FakeQwenModel(sample_rate=16_000)
        engine = engine_with(model)
        with pytest.raises(TTSError):
            _ = engine.sample_rate  # unknown before the first call
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        engine.synthesize(["a"], language="French")
        assert engine.sample_rate == 16_000

    def test_output_is_mono_float32(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        result = engine.synthesize(["a"], language="French")
        assert result[0] is not None
        assert result[0].ndim == 1
        assert result[0].dtype == np.float32


class TestModelTypeGuard:
    def test_rejects_non_base_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class WrongCheckpoint:
            def __init__(self) -> None:
                self.model = SimpleNamespace(tts_model_type="custom_voice")

            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> "WrongCheckpoint":
                return cls()

        fake_module = SimpleNamespace(Qwen3TTSModel=WrongCheckpoint)
        monkeypatch.setitem(sys.modules, "qwen_tts", fake_module)

        engine = TTSEngine("models/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cpu")
        with pytest.raises(TTSError, match="does not support voice cloning"):
            _ = engine.model

    def test_accepts_base_checkpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class BaseCheckpoint:
            def __init__(self) -> None:
                self.model = SimpleNamespace(tts_model_type="base")

            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> "BaseCheckpoint":
                return cls()

        monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=BaseCheckpoint))
        engine = TTSEngine("models/Qwen3-TTS-12Hz-1.7B-Base", device="cpu")
        assert engine.model.model.tts_model_type == "base"


class TestAccentMode:
    """Speaker-only mode keeps the timbre embedding but drops the accent."""

    def test_icl_mode_is_the_default(self) -> None:
        model = FakeQwenModel()
        engine = engine_with(model)
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        assert model.prompts[0]["x_vector_only_mode"] is False

    def test_speaker_only_mode_is_forwarded(self) -> None:
        model = FakeQwenModel()
        engine = TTSEngine("models/fake", device="cpu", x_vector_only=True)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")
        assert model.prompts[0]["x_vector_only_mode"] is True

    def test_speaker_only_mode_does_not_require_a_transcript(self) -> None:
        model = FakeQwenModel()
        engine = TTSEngine("models/fake", device="cpu", x_vector_only=True)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "")  # must not raise
        assert model.prompts[0]["x_vector_only_mode"] is True

    def test_donor_prompt_honours_the_mode(self) -> None:
        model = FakeQwenModel()
        engine = TTSEngine("models/fake", device="cpu", x_vector_only=True)
        engine._model = model
        engine.prepare_prompt(
            DonorClip(samples=audio(3.0), sample_rate=SR, text="hi", start=0, end=3, segment_id=0)
        )
        assert model.prompts[0]["x_vector_only_mode"] is True


@dataclass
class FakePromptItem:
    '''Duck-typed stand-in for the checkpoint's VoiceClonePromptItem.

    It has to be a real dataclass because the hybrid path rebuilds it with
    dataclasses.replace.
    '''

    ref_code: Any = None
    ref_spk_embedding: Any = None
    x_vector_only_mode: bool = False
    icl_mode: bool = True
    ref_text: str | None = None


class HybridQwenModel(FakeQwenModel):
    '''Tags each prompt item with the clip it came from, so provenance is testable.'''

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[FakePromptItem]:
        self.prompts.append(kwargs)
        ref = kwargs['ref_audio']
        label = ref if isinstance(ref, str) else 'array'
        return [
            FakePromptItem(
                ref_code=f'code:{label}',
                ref_spk_embedding=f'embedding:{label}',
                x_vector_only_mode=kwargs['x_vector_only_mode'],
                icl_mode=not kwargs['x_vector_only_mode'],
                ref_text=kwargs['ref_text'],
            )
        ]


class TestHybridPrompt:
    '''reference-accent takes codes from the accent clip, timbre from the voice.'''

    def prepared(self, **kwargs: Any) -> tuple[TTSEngine, HybridQwenModel]:
        model = HybridQwenModel()
        engine = engine_with(model)  # type: ignore[arg-type]
        engine.prepare_hybrid_prompt(
            accent_audio=kwargs.pop('accent_audio', 'accent.wav'),
            accent_text=kwargs.pop('accent_text', 'bonjour'),
            **kwargs,
        )
        return engine, model

    def test_accent_clip_is_read_in_icl_mode(self) -> None:
        '''Codes only carry pronunciation when ICL mode is on.'''
        _, model = self.prepared(timbre_audio='donor.wav')
        assert model.prompts[0]['x_vector_only_mode'] is False
        assert model.prompts[0]['ref_audio'] == 'accent.wav'

    def test_timbre_clip_is_read_in_xvector_mode(self) -> None:
        '''Only an embedding is wanted, so no timbre transcript is needed.'''
        _, model = self.prepared(timbre_audio='donor.wav')
        assert model.prompts[1]['x_vector_only_mode'] is True
        assert model.prompts[1]['ref_text'] == ''

    def test_result_takes_codes_from_accent_and_embedding_from_timbre(self) -> None:
        engine, _ = self.prepared(timbre_audio='donor.wav')
        assert isinstance(engine._prompt, list) and len(engine._prompt) == 1
        item = engine._prompt[0]
        assert item.ref_code == 'code:accent.wav'
        assert item.ref_spk_embedding == 'embedding:donor.wav'
        assert item.ref_text == 'bonjour'
        assert item.icl_mode is True
        assert item.x_vector_only_mode is False

    def test_donor_clip_is_accepted_as_the_timbre_source(self) -> None:
        donor = DonorClip(
            samples=audio(3.0), sample_rate=SR, text='hello', start=0, end=3, segment_id=0
        )
        engine, model = self.prepared(timbre=donor)
        assert model.prompts[1]['x_vector_only_mode'] is True
        assert engine._prompt[0].ref_spk_embedding == 'embedding:array'

    def test_requires_an_accent_transcript(self) -> None:
        engine = engine_with(HybridQwenModel())  # type: ignore[arg-type]
        with pytest.raises(TTSError, match='accent_text is required'):
            engine.prepare_hybrid_prompt(
                accent_audio='accent.wav', accent_text='  ', timbre_audio='donor.wav'
            )

    def test_requires_a_timbre_source(self) -> None:
        engine = engine_with(HybridQwenModel())  # type: ignore[arg-type]
        with pytest.raises(TTSError, match='needs a timbre source'):
            engine.prepare_hybrid_prompt(accent_audio='accent.wav', accent_text='bonjour')

    def test_unusable_prompt_items_are_reported(self) -> None:
        '''A checkpoint returning objects without the field must fail loudly.'''

        class OpaqueModel(HybridQwenModel):
            def create_voice_clone_prompt(self, **kwargs: Any) -> list[Any]:
                return [SimpleNamespace(ref_code='x')]

        engine = engine_with(OpaqueModel())  # type: ignore[arg-type]
        with pytest.raises(TTSError, match='Could not combine'):
            engine.prepare_hybrid_prompt(
                accent_audio='accent.wav', accent_text='bonjour', timbre_audio='donor.wav'
            )

    def test_hybrid_prompt_reaches_generation(self) -> None:
        engine, model = self.prepared(timbre_audio='donor.wav')
        engine.synthesize(['bonjour'], language='French')
        assert model.calls[0]['voice_clone_prompt'][0].ref_code == 'code:accent.wav'

    def test_existing_prompts_are_untouched(self) -> None:
        '''Regression guard: the two original paths still produce one item each.'''
        from_file = engine_with(FakeQwenModel())
        from_file.prepare_prompt_from_file('ref.wav', 'bonjour')
        assert from_file._prompt == 'prompt-token'

        from_donor = engine_with(FakeQwenModel())
        from_donor.prepare_prompt(
            DonorClip(samples=audio(3.0), sample_rate=SR, text='hi', start=0, end=3, segment_id=0)
        )
        assert from_donor._prompt == 'prompt-token'


class ScoringQwenModel(FakeQwenModel):
    '''Hybrid model with real embedding vectors and a scoreable speaker encoder.

    ``take_embeddings`` supplies the embedding the encoder reports for each take,
    in order, so a test can state exactly which take sounds like the donor and
    which one collapsed onto the accent clip.
    '''

    def __init__(
        self,
        take_embeddings: list[list[float]],
        take_lengths: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.take_embeddings = [
            np.asarray(vector, dtype=np.float64) for vector in take_embeddings
        ]
        #: Sample counts per take, cycled. ``None`` renders every take the same
        #: length; a shorter entry is how a truncated render is simulated.
        self.take_lengths = take_lengths
        self._next_take = 0
        self._produced = 0
        #: Indices of the takes the encoder was asked about, in order.
        self.embedded: list[int] = []
        self.model = SimpleNamespace(
            tts_model_type='base',
            extract_speaker_embedding=self._extract,
        )

    def _extract(self, audio: Any, sr: int) -> np.ndarray:
        index = self._next_take % max(1, len(self.take_embeddings))
        self._next_take += 1
        self.embedded.append(index)
        return self.take_embeddings[index]

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[FakePromptItem]:
        self.prompts.append(kwargs)
        # Orthogonal directions, so a take's margin against one source says
        # nothing about the other and the score is unambiguous.
        accent_side = not kwargs['x_vector_only_mode']
        vector = np.asarray([1.0, 0.0]) if accent_side else np.asarray([0.0, 1.0])
        return [
            FakePromptItem(
                ref_code=f'code:{kwargs["ref_audio"]}',
                ref_spk_embedding=vector,
                x_vector_only_mode=kwargs['x_vector_only_mode'],
                icl_mode=not kwargs['x_vector_only_mode'],
                ref_text=kwargs['ref_text'],
            )
        ]

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.calls.append(kwargs)
        wavs = []
        for _ in kwargs['text']:
            # The first sample records which take this is, which makes the take
            # that was finally kept identifiable in the result.
            index = self._produced
            length = 16
            if self.take_lengths:
                length = self.take_lengths[index % len(self.take_lengths)]
            block = np.zeros(length, dtype=np.float32)
            block[0] = float(index)
            self._produced += 1
            wavs.append(block)
        return wavs, self.sample_rate


def build_blend(engine: TTSEngine, **kwargs: Any) -> Any:
    '''Build a blended reference with the two sources wired up.'''
    return engine.build_blended_reference(
        accent_audio='accent.wav',
        accent_text='bonjour',
        language='French',
        timbre_audio='donor.wav',
        **kwargs,
    )


class TestBlendedReference:
    '''Stage 1: one clip whose speech codes and speaker embedding agree.'''

    #: Mostly the accent clip's direction - the collapse this stage guards against.
    COLLAPSED = [0.9, 0.1]
    #: Mostly the donor's direction - the blend that is wanted.
    KEPT = [0.1, 0.9]
    #: Exactly between the two, so the margin is zero.
    MIDDLE = [0.5, 0.5]

    def scorer(self, take_embeddings: list[list[float]]) -> tuple[TTSEngine, ScoringQwenModel]:
        model = ScoringQwenModel(take_embeddings)
        return engine_with(model), model  # type: ignore[arg-type]

    def test_keeps_the_take_that_sounds_most_like_the_donor(self) -> None:
        engine, model = self.scorer([self.COLLAPSED, self.KEPT, self.MIDDLE])
        result = build_blend(engine, takes=3)

        assert result.takes == 3
        assert model.embedded == [0, 1, 2]  # every take was scored
        assert result.samples[0] == pytest.approx(1.0)  # take 1 was kept
        assert result.timbre_similarity == pytest.approx(0.9939, abs=0.001)
        assert result.accent_similarity == pytest.approx(0.1104, abs=0.001)
        assert result.margin == pytest.approx(0.8835, abs=0.001)

    def test_a_single_take_is_not_scored(self) -> None:
        '''One take is not a choice, so there is nothing to compare.'''
        engine, model = self.scorer([self.KEPT])
        result = build_blend(engine, takes=1)

        assert result.takes == 1
        assert model.embedded == []
        assert np.isnan(result.margin)
        assert result.samples[0] == pytest.approx(0.0)

    def test_ties_keep_the_earliest_take(self) -> None:
        engine, _ = self.scorer([self.MIDDLE, self.MIDDLE, self.MIDDLE])
        result = build_blend(engine, takes=3)

        assert result.samples[0] == pytest.approx(0.0)
        assert result.margin == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_embeddings_fall_back_to_the_first_take(self) -> None:
        '''A zero vector has no direction, so no take can be preferred.'''
        engine, _ = self.scorer([[0.0, 0.0], [0.0, 0.0]])
        result = build_blend(engine, takes=2)

        assert result.samples[0] == pytest.approx(0.0)
        assert np.isnan(result.margin)

    def test_a_checkpoint_without_an_encoder_still_produces_a_blend(self) -> None:
        '''Scoring is a preference; audio must still come out without it.'''
        engine = engine_with(HybridQwenModel())  # type: ignore[arg-type]
        result = build_blend(engine, takes=3)

        assert result.takes == 3
        assert np.isnan(result.margin)
        assert result.samples.size > 0

    def test_takes_below_one_render_a_single_candidate(self) -> None:
        engine, model = self.scorer([self.KEPT])
        result = build_blend(engine, takes=0)

        assert result.takes == 1
        assert model.calls[0]['text'] == ['bonjour']

    def test_stage_one_uses_the_hybrid_prompt_in_the_target_language(self) -> None:
        engine, model = self.scorer([self.KEPT, self.KEPT])
        build_blend(engine, takes=2)

        # Accent clip read in ICL mode, timbre clip in x-vector mode - the same
        # hybrid prompt the single-stage path uses, just consumed once.
        assert model.prompts[0]['ref_audio'] == 'accent.wav'
        assert model.prompts[0]['x_vector_only_mode'] is False
        assert model.prompts[1]['ref_audio'] == 'donor.wav'
        assert model.prompts[1]['x_vector_only_mode'] is True
        assert model.calls[0]['language'] == ['French', 'French']
        assert model.calls[0]['text'] == ['bonjour', 'bonjour']

    def test_requires_the_accent_transcript(self) -> None:
        engine = engine_with(HybridQwenModel())  # type: ignore[arg-type]
        with pytest.raises(TTSError, match='accent_text is required'):
            engine.build_blended_reference(
                accent_audio='accent.wav',
                accent_text='   ',
                language='French',
                timbre_audio='donor.wav',
            )

    def test_silent_output_is_reported(self) -> None:
        class SilentModel(HybridQwenModel):
            def generate_voice_clone(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                return (
                    [np.zeros(0, dtype=np.float32) for _ in kwargs['text']],
                    self.sample_rate,
                )

        engine = engine_with(SilentModel())  # type: ignore[arg-type]
        with pytest.raises(TTSError, match='no audio for the blended reference'):
            build_blend(engine, takes=2)


class TestTruncatedRenders:
    '''A render that stopped early is a failed render, not a candidate voice.

    Observed on a real run: a 1.4 s clip came back for a 14 s transcript and, being
    a fragment, scored well enough on the ratio search to be chosen as the voice the
    whole video would be cloned from.
    '''

    #: Mostly the donor's direction - the blend that is wanted.
    KEPT = [0.1, 0.9]

    def test_a_fragment_cannot_win_the_take_search(self) -> None:
        # Take 0 is a fragment that scores like the donor; take 1 is the full render.
        model = ScoringQwenModel(
            take_embeddings=[self.KEPT, self.KEPT], take_lengths=[16, 400]
        )
        engine = engine_with(model)  # type: ignore[arg-type]

        reference = build_blend(engine, takes=2)

        # The fragment was dropped before scoring, so the full render is kept.
        assert reference.samples.size == 400

    def test_takes_of_comparable_length_are_all_kept(self) -> None:
        kept = _drop_truncated([np.zeros(100), np.zeros(98), np.zeros(51)])

        # A genuinely terser take is still a take.
        assert [item.size for item in kept] == [100, 98, 51]

    def test_a_fragment_is_dropped(self) -> None:
        # The real case: a 1.4 s render beside a 14 s transcript.
        kept = _drop_truncated([np.zeros(140), np.zeros(49), np.zeros(14)])

        assert [item.size for item in kept] == [140]

    def test_no_renders_is_no_candidates(self) -> None:
        assert _drop_truncated([]) == []

    def test_a_single_render_is_always_usable(self) -> None:
        # However short, the only render there is must survive: something has to be
        # chosen, and the alternative is failing the job.
        assert [item.size for item in _drop_truncated([np.zeros(5)])] == [5]


class BlendModel(FakeQwenModel):
    '''Numeric embeddings, distinguishable audio, and a scripted speaker encoder.

    The accent clip and the donor sit at opposite ends of the embedding space, so a
    rendering's donor ratio is exactly its position between them. ``scripted`` gives
    the encoder's answer for each call in order, which is how a test can state where
    each rendering landed.
    '''

    ACCENT = np.asarray([1.0, 0.0])
    DONOR = np.asarray([0.0, 1.0])
    #: Handed to later passes' accent side, so scoring against the *current* clip
    #: instead of the original accent clip would change the reported number.
    DECOY = np.asarray([0.5, 0.5])

    def __init__(self, scripted: list[list[float]]) -> None:
        super().__init__()
        self.scripted = [np.asarray(vector, dtype=np.float64) for vector in scripted]
        #: Indices of ``scripted`` the encoder was asked for, in order.
        self.embedded: list[int] = []
        self.accent_prompts = 0
        self._produced = 0
        self._next = 0
        self.model = SimpleNamespace(
            tts_model_type='base', extract_speaker_embedding=self._extract
        )

    def _extract(self, audio: Any, sr: int) -> np.ndarray:
        index = min(self._next, len(self.scripted) - 1)
        self._next += 1
        self.embedded.append(index)
        return self.scripted[index]

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[FakePromptItem]:
        self.prompts.append(kwargs)
        if kwargs['x_vector_only_mode']:
            vector = self.DONOR
        else:
            self.accent_prompts += 1
            vector = self.ACCENT if self.accent_prompts == 1 else self.DECOY
        return [
            FakePromptItem(
                ref_code=f'code:{kwargs["ref_audio"]}',
                ref_spk_embedding=vector.copy(),
                x_vector_only_mode=kwargs['x_vector_only_mode'],
                icl_mode=not kwargs['x_vector_only_mode'],
                ref_text=kwargs['ref_text'],
            )
        ]

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.calls.append(kwargs)
        wavs = []
        for _ in kwargs['text']:
            # The first sample records which rendering this is, so the one that was
            # kept can be identified in the result.
            block = np.zeros(16, dtype=np.float32)
            block[0] = float(self._produced)
            self._produced += 1
            wavs.append(block)
        return wavs, self.sample_rate


class TestDonorRatio:
    '''The language-matched scale: 0 is the accent voice, 1 is the speaker's own.'''

    ACCENT = np.asarray([1.0, 0.0])
    DONOR = np.asarray([0.0, 1.0])

    def ratio(self, candidate: list[float]) -> float:
        return _donor_ratio(np.asarray(candidate, dtype=np.float64), self.ACCENT, self.DONOR)

    def test_ends_and_middle(self) -> None:
        assert self.ratio([1.0, 0.0]) == pytest.approx(0.0)
        assert self.ratio([0.0, 1.0]) == pytest.approx(1.0)
        assert self.ratio([0.5, 0.5]) == pytest.approx(0.5)

    def test_linear_along_the_axis(self) -> None:
        assert self.ratio([0.9, 0.1]) == pytest.approx(0.1)
        assert self.ratio([0.4, 0.6]) == pytest.approx(0.6)

    def test_positions_outside_the_ends_are_clamped(self) -> None:
        assert self.ratio([2.0, -1.0]) == pytest.approx(0.0)
        assert self.ratio([-1.0, 2.0]) == pytest.approx(1.0)

    def test_missing_or_degenerate_anchors_are_unmeasurable(self) -> None:
        candidate = np.asarray([0.5, 0.5], dtype=np.float64)
        assert np.isnan(_donor_ratio(candidate, None, self.DONOR))
        assert np.isnan(_donor_ratio(candidate, self.ACCENT, None))
        # Both anchors the same voice leaves no direction to measure along.
        assert np.isnan(_donor_ratio(candidate, self.ACCENT, self.ACCENT))
        assert np.isnan(_donor_ratio(np.asarray([1.0, 0.0, 0.0]), self.ACCENT, self.DONOR))


class TestBlendIteration:
    '''One pass favours the accent clip, so later passes pull toward the speaker.'''

    #: Ratio 0.1 - still practically the accent clip's voice.
    ACCENTISH = [0.9, 0.1]
    #: Ratio 0.6 - past the default 0.5 target.
    SPEAKERISH = [0.4, 0.6]

    def scorer(self, scripted: list[list[float]]) -> tuple[TTSEngine, BlendModel]:
        model = BlendModel(scripted)
        return engine_with(model), model  # type: ignore[arg-type]

    def test_a_clean_pass_needs_no_further_iteration(self) -> None:
        engine, model = self.scorer([BlendModel.DONOR.tolist(), self.SPEAKERISH])
        result = build_blend(engine, iters=4, ratio=0.5)

        assert result.iters == 1
        assert result.donor_ratio == pytest.approx(0.6)
        # One synthesis for the donor anchor, one for the single blend pass.
        assert len(model.calls) == 2

    def test_iteration_walks_toward_the_speaker_and_stops_at_the_target(self) -> None:
        engine, model = self.scorer(
            [BlendModel.DONOR.tolist(), self.ACCENTISH, self.SPEAKERISH]
        )
        result = build_blend(engine, iters=4, ratio=0.5)

        # Pass 0 was still accent-like, pass 1 cleared the target, so pass 2 never ran.
        assert result.iters == 2
        assert result.donor_ratio == pytest.approx(0.6)
        assert len(model.calls) == 3
        # The kept clip is pass 1's rendering, whose first sample records its index.
        assert result.samples[0] == pytest.approx(2.0)

    def test_iteration_re_encodes_the_previous_pass(self) -> None:
        engine, model = self.scorer(
            [BlendModel.DONOR.tolist(), self.ACCENTISH, self.SPEAKERISH]
        )
        build_blend(engine, iters=2, ratio=0.5)

        # The second pass reads its codes from the first clip, handed over as
        # samples rather than a path: it never exists as a file.
        from_samples = [p for p in model.prompts if isinstance(p['ref_audio'], tuple)]
        assert len(from_samples) == 1
        assert from_samples[0]['x_vector_only_mode'] is False  # the accent side
        samples, rate = from_samples[0]['ref_audio']
        assert samples[0] == pytest.approx(1.0)
        assert rate == SR

    def test_the_best_pass_wins_even_when_it_is_not_the_last(self) -> None:
        '''Passes are stochastic, so running longer must not mean getting worse.'''
        engine, model = self.scorer(
            [
                BlendModel.DONOR.tolist(),
                [1.0, 0.0],   # pass 0: ratio 0.0
                [0.1, 0.9],   # pass 1: ratio 0.9  <- the best
                [0.6, 0.4],   # pass 2: ratio 0.4
            ]
        )
        result = build_blend(engine, iters=3, ratio=0.0)

        assert result.iters == 3  # nothing to stop at, so every pass ran
        assert result.donor_ratio == pytest.approx(0.9)
        assert result.samples[0] == pytest.approx(2.0)

    def test_scoring_uses_the_original_accent_clip(self) -> None:
        '''Later passes hand over a blend as the accent; it must not become the yardstick.'''
        engine, _ = self.scorer(
            [BlendModel.DONOR.tolist(), self.ACCENTISH, self.SPEAKERISH]
        )
        result = build_blend(engine, iters=2, ratio=0.5)

        # Scoring is against the *original* accent embedding, so the accent
        # similarity is the normalised cosine to it - not the ~0.98 the decoy that
        # later passes carry would produce.
        expected = float(np.dot([0.4, 0.6], [1.0, 0.0]) / np.linalg.norm([0.4, 0.6]))
        assert result.accent_similarity == pytest.approx(expected)
        assert result.accent_similarity < 0.7
        assert result.donor_ratio == pytest.approx(0.6)

    def test_the_anchor_probe_is_skipped_when_nothing_reads_it(self) -> None:
        engine, model = self.scorer([BlendModel.DONOR.tolist()])
        result = build_blend(engine, takes=1, iters=1, ratio=0.0)

        # No anchor synthesis and no scoring: just the one blend rendering.
        assert len(model.calls) == 1
        assert model.embedded == []
        assert np.isnan(result.donor_ratio)
        assert result.iters == 1

    def test_an_unmeasurable_ratio_still_produces_a_blend(self) -> None:
        '''No speaker encoder means no ratio, not a failed job.'''
        async_model = HybridQwenModel()  # string embeddings, no encoder at all
        engine = engine_with(async_model)  # type: ignore[arg-type]
        result = build_blend(engine, iters=2, ratio=0.5)

        assert result.iters == 2
        assert np.isnan(result.donor_ratio)
        assert result.samples.size > 0

    def test_iters_below_one_is_clamped(self) -> None:
        engine, model = self.scorer([BlendModel.DONOR.tolist(), self.ACCENTISH])
        result = build_blend(engine, iters=0, ratio=0.5)

        # At most one pass - but the anchor is still measured, so a shortfall is
        # reported instead of silently ignored.
        assert result.iters == 1
        assert len(model.calls) == 2
        assert result.donor_ratio == pytest.approx(0.1)


class TestResolveDtype:
    def test_cuda_uses_bfloat16(self) -> None:
        import torch

        assert resolve_dtype('cuda:0') == torch.bfloat16

    def test_cpu_uses_float32(self) -> None:
        import torch

        assert resolve_dtype("cpu") == torch.float32


class TestOutOfMemoryHandling:
    """A batch too large for the GPU must be split, not fail the job."""

    class OomModel(FakeQwenModel):
        def __init__(self, *, fail_above: int) -> None:
            super().__init__()
            self.fail_above = fail_above
            self.attempts: list[int] = []

        def create_voice_clone_prompt(self, **kwargs: Any) -> str:
            return "prompt-token"

        def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
            texts = kwargs["text"]
            self.attempts.append(len(texts))
            if len(texts) > self.fail_above:
                raise RuntimeError(
                    "CUDA out of memory. Tried to allocate 1.46 GiB on device 0"
                )
            return [audio(0.4) for _ in texts], SR

    def test_batch_is_halved_until_it_fits(self) -> None:
        model = self.OomModel(fail_above=1)
        engine = TTSEngine("models/fake", device="cpu", batch_size=4)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        result = engine.synthesize(["a", "b", "c", "d"], language="French")

        assert all(item is not None for item in result)
        assert len(result) == 4
        assert model.attempts[0] == 4  # started with the full batch
        assert model.attempts[-1] == 1  # ended up splitting to singles

    def test_no_oom_means_a_single_attempt(self) -> None:
        model = self.OomModel(fail_above=99)
        engine = TTSEngine("models/fake", device="cpu", batch_size=4)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        engine.synthesize(["a", "b", "c", "d"], language="French")
        assert model.attempts == [4]

    def test_oom_on_a_single_line_is_reported(self) -> None:
        model = self.OomModel(fail_above=0)
        engine = TTSEngine("models/fake", device="cpu", batch_size=1)
        engine._model = model
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        with pytest.raises(TTSError, match="Out of GPU memory"):
            engine.synthesize(["a"], language="French")

    def test_unrelated_errors_are_not_retried(self) -> None:
        class BrokenModel(FakeQwenModel):
            def create_voice_clone_prompt(self, **kwargs: Any) -> str:
                return "prompt-token"

            def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
                raise RuntimeError("model file is corrupt")

        engine = TTSEngine("models/fake", device="cpu", batch_size=2)
        engine._model = BrokenModel()
        engine.prepare_prompt_from_file("ref.wav", "bonjour")

        with pytest.raises(RuntimeError, match="corrupt"):
            engine.synthesize(["a", "b"], language="French")


class TestKnnSpeakerSelection:
    def test_reference_audio_requires_transcript_text(self) -> None:
        engine = engine_with(FakeQwenModel())
        with pytest.raises(TTSError, match="ref_text is required"):
            engine.prepare_prompt_from_file("ref.wav", "")
