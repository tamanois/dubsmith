"""Tests for the saved voice-print library.

No model is loaded and no GPU is needed: the store is pure data handling, which
is exactly why it lives in its own module. The boundaries that matter are a
label's identity (slug, de-duplication), the save/load roundtrip, the atomic
write, and the errors a user can actually hit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.errors import TTSError, VoiceNotFoundError
from pipeline.voices import (
    LEGACY_PROMPT_NAME,
    MANIFEST_NAME,
    MANIFEST_VERSION,
    PROMPT_DIR,
    PromptData,
    VoicePrint,
    WAV_NAME,
    cards_in,
    default_label,
    delete_voice,
    find_prompt,
    import_voice,
    list_voices,
    load_prompt,
    load_voice,
    save_voice,
    slugify_label,
    unique_label,
    voice_content,
    voice_paths,
)

SAMPLE_RATE = 24_000
TRANSCRIPT = "Bonjour tout le monde."


def tone(seconds: float = 1.5, hz: float = 220.0) -> np.ndarray:
    """A deterministic clip with a known pitch and length."""
    t = np.linspace(0.0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def write_clip(path: Path, seconds: float = 1.5) -> Path:
    """A real WAV on disk, so the card's content hash has something to read."""
    from pipeline.media import write_audio

    return write_audio(path, tone(seconds), SAMPLE_RATE)


def make_voice(root: Path, label: str = "warm-fr", **overrides: object):
    """Save a voice with sensible defaults, overriding only what a test needs."""
    payload: dict[str, object] = {
        "language": "French",
        "source": "design",
        "ref_text": TRANSCRIPT,
    }
    payload.update(overrides)
    return save_voice(root, label, tone(), SAMPLE_RATE, **payload)  # type: ignore[arg-type]


class TestSlugify:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("warm-fr", "warm-fr"),
            ("Warm FR", "warm-fr"),
            ("  spaced  out  ", "spaced-out"),
            ("a_b.c", "a-b-c"),
            ("", "voice"),
            ("---", "voice"),
            ("../etc/passwd", "etc-passwd"),
            ("../../evil", "evil"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert slugify_label(raw) == expected

    def test_truncates_long_labels(self) -> None:
        slug = slugify_label("word " * 40)
        assert len(slug) <= 48
        assert not slug.endswith("-")


class TestUniqueLabel:
    def test_free_label_is_returned_as_is(self, tmp_path: Path) -> None:
        assert unique_label("Warm FR", tmp_path) == "warm-fr"

    def test_collisions_get_a_suffix(self, tmp_path: Path) -> None:
        make_voice(tmp_path, "warm-fr")
        assert unique_label("warm-fr", tmp_path) == "warm-fr-2"

    def test_suffix_skips_taken_numbers(self, tmp_path: Path) -> None:
        make_voice(tmp_path, "warm-fr")
        make_voice(tmp_path, "warm-fr-2")
        assert unique_label("warm-fr", tmp_path) == "warm-fr-3"

    def test_long_label_suffix_stays_within_the_limit(self, tmp_path: Path) -> None:
        # The suffixed name is slugified again on save, so if it overflowed the
        # limit it would be truncated back onto the original and collide.
        long = "word " * 40
        first = make_voice(tmp_path, long)
        candidate = unique_label(long, tmp_path)

        assert len(candidate) <= 48
        assert candidate != first.label
        second = make_voice(tmp_path, candidate)
        assert [item.label for item in list_voices(tmp_path)] == sorted(
            [first.label, second.label]
        )


class TestDefaultLabel:
    def test_description_names_the_voice(self) -> None:
        assert default_label(instruct="A warm female French voice") == (
            "a-warm-female-french-voice"
        )

    def test_audio_filename_is_the_fallback(self) -> None:
        assert default_label(audio=Path("input/Delphine-Charming.mp3")) == (
            "delphine-charming"
        )

    def test_no_hint_is_a_generic_name(self) -> None:
        assert default_label() == "voice"


class TestPaths:
    def test_paths_do_not_create_anything(self, tmp_path: Path) -> None:
        directory, wav, manifest = voice_paths(tmp_path, "Warm FR")

        assert directory == tmp_path / "warm-fr"
        assert wav == directory / WAV_NAME
        assert manifest == directory / MANIFEST_NAME
        assert not directory.exists()


class TestSave:
    def test_writes_clip_and_manifest(self, tmp_path: Path) -> None:
        print_ = make_voice(tmp_path, "Warm FR")

        assert print_.label == "warm-fr"
        assert print_.wav.is_file()
        assert print_.manifest_path.is_file()
        payload = json.loads(print_.manifest_path.read_text(encoding="utf-8"))
        assert payload["version"] == MANIFEST_VERSION
        assert payload["label"] == "warm-fr"
        assert payload["language"] == "French"
        assert payload["source"] == "design"
        assert payload["ref_text"] == TRANSCRIPT
        assert payload["seconds"] == pytest.approx(1.5, abs=0.01)

    def test_measures_pitch_and_pace(self, tmp_path: Path) -> None:
        print_ = make_voice(tmp_path)

        assert print_.gender == "f"  # 220 Hz
        # Four words in 1.5 s -> ~2.7 words/s, below the slow threshold.
        assert print_.rate_band == "slow"

    def test_leaves_no_temporary_files(self, tmp_path: Path) -> None:
        print_ = make_voice(tmp_path)
        names = sorted(item.name for item in print_.directory.iterdir())
        assert names == sorted([WAV_NAME, MANIFEST_NAME])

    def test_duplicate_label_is_refused(self, tmp_path: Path) -> None:
        make_voice(tmp_path)
        with pytest.raises(TTSError, match="already exists"):
            make_voice(tmp_path)

    def test_force_overwrites(self, tmp_path: Path) -> None:
        first = make_voice(tmp_path, "warm-fr")
        second = make_voice(tmp_path, "warm-fr", force=True, source="clone")
        assert second.source == "clone"
        assert load_voice(tmp_path, "warm-fr").source == "clone"
        assert first.created <= second.created

    def test_clone_source_is_kept(self, tmp_path: Path) -> None:
        print_ = make_voice(tmp_path, source="clone")
        assert load_voice(tmp_path, print_.label).source == "clone"

    def test_empty_clip_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TTSError, match="empty"):
            save_voice(
                tmp_path,
                "x",
                np.zeros(0, dtype=np.float32),
                SAMPLE_RATE,
                language="French",
                source="design",
                ref_text=TRANSCRIPT,
            )

    def test_missing_transcript_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TTSError, match="transcript"):
            save_voice(
                tmp_path,
                "x",
                tone(),
                SAMPLE_RATE,
                language="French",
                source="design",
                ref_text="  ",
            )

    def test_unknown_source_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TTSError, match="Unknown voice source"):
            save_voice(
                tmp_path,
                "x",
                tone(),
                SAMPLE_RATE,
                language="French",
                source="magic",
                ref_text=TRANSCRIPT,
            )


class TestLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        saved = make_voice(tmp_path, "Warm FR", instruct="a warm voice")
        loaded = load_voice(tmp_path, "warm fr")  # case-insensitive

        assert loaded.label == saved.label
        assert loaded.language == saved.language
        assert loaded.source == saved.source
        assert loaded.ref_text == saved.ref_text
        assert loaded.instruct == "a warm voice"
        assert loaded.wav == saved.wav

    def test_missing_voice_lists_what_exists(self, tmp_path: Path) -> None:
        make_voice(tmp_path, "warm-fr")
        with pytest.raises(VoiceNotFoundError) as info:
            load_voice(tmp_path, "nope")

        message = str(info.value)
        assert "nope" in message
        assert "warm-fr" in message
        assert "voices --list" in message

    def test_empty_library_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(VoiceNotFoundError, match="library is empty"):
            load_voice(tmp_path, "nope")

    def test_corrupt_manifest_is_reported(self, tmp_path: Path) -> None:
        directory, wav, manifest = voice_paths(tmp_path, "broken")
        directory.mkdir(parents=True)
        wav.write_bytes(b"not audio")
        manifest.write_text("{not json", encoding="utf-8")

        with pytest.raises(VoiceNotFoundError, match="unreadable manifest"):
            load_voice(tmp_path, "broken")

    def test_manifest_without_transcript_cannot_be_cloned(self, tmp_path: Path) -> None:
        directory, wav, manifest = voice_paths(tmp_path, "empty-text")
        directory.mkdir(parents=True)
        wav.write_bytes(b"not audio")
        manifest.write_text(
            json.dumps({"label": "empty-text", "language": "French", "source": "clone"}),
            encoding="utf-8",
        )

        with pytest.raises(VoiceNotFoundError, match="no transcript"):
            load_voice(tmp_path, "empty-text")


class TestList:
    def test_missing_directory_is_empty(self, tmp_path: Path) -> None:
        assert list_voices(tmp_path / "nope") == []

    def test_sorted_by_label(self, tmp_path: Path) -> None:
        make_voice(tmp_path, "zebra")
        make_voice(tmp_path, "alpha")
        assert [item.label for item in list_voices(tmp_path)] == ["alpha", "zebra"]

    def test_skips_unreadable_entries(self, tmp_path: Path) -> None:
        make_voice(tmp_path, "good")
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / WAV_NAME).write_bytes(b"x")
        (broken / MANIFEST_NAME).write_text("{}", encoding="utf-8")

        assert [item.label for item in list_voices(tmp_path)] == ["good"]


class TestDelete:
    def test_removes_the_directory(self, tmp_path: Path) -> None:
        print_ = make_voice(tmp_path)
        removed = delete_voice(tmp_path, print_.label)

        assert removed == print_.directory
        assert not print_.directory.exists()
        assert list_voices(tmp_path) == []

    def test_missing_voice_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(VoiceNotFoundError):
            delete_voice(tmp_path, "nope")


def test_summary_is_human_readable(tmp_path: Path) -> None:
    print_ = make_voice(tmp_path, "warm-fr", instruct="a warm voice")
    line = print_.summary()

    assert "warm-fr" in line
    assert "French" in line
    assert "a warm voice" in line


def prompt_data(
    *, x_vector: bool = False, fingerprint: str = "abc123def456"
) -> PromptData:
    """A deterministic encoding, shaped like the checkpoint's own."""
    return PromptData(
        ref_code=None if x_vector else np.arange(24, dtype=np.int64).reshape(6, 4),
        ref_spk_embedding=np.linspace(0.0, 1.0, 8, dtype=np.float32),
        x_vector_only=x_vector,
        ref_text="" if x_vector else TRANSCRIPT,
        fingerprint=fingerprint,
    )


def saved_with_prompt(
    root: Path,
    label: str = "v",
    *,
    fingerprint: str = "abc123def456",
    force: bool = False,
) -> VoicePrint:
    return save_voice(
        root,
        label,
        tone(),
        SAMPLE_RATE,
        language="French",
        source="clone",
        ref_text=TRANSCRIPT,
        prompt=prompt_data(fingerprint=fingerprint),
        force=force,
    )


class TestCards:
    def test_roundtrip(self, tmp_path: Path) -> None:
        print_ = saved_with_prompt(tmp_path)
        card = print_.card_for("qwen", "abc123def456")
        assert card is not None
        loaded = load_prompt(card)

        assert loaded.ref_code is not None
        assert loaded.ref_code.shape == (6, 4)
        # The tokenizer's integer dtype survives: the decoder concatenates these
        # codes with freshly generated ones.
        assert loaded.ref_code.dtype == np.int64
        assert loaded.ref_spk_embedding.dtype == np.float32
        assert loaded.x_vector_only is False
        assert loaded.ref_text == TRANSCRIPT
        assert loaded.backend == "qwen"
        assert loaded.model == "tts_base"
        assert loaded.fingerprint == "abc123def456"
        assert loaded.content == voice_content(print_.wav, TRANSCRIPT)

    def test_x_vector_card_has_no_codes(self, tmp_path: Path) -> None:
        save_voice(
            tmp_path,
            "v",
            tone(),
            SAMPLE_RATE,
            language="French",
            source="import",
            ref_text="",
            prompt=prompt_data(x_vector=True),
        )
        stored = load_voice(tmp_path, "v")

        loaded = load_prompt(stored.card_for("qwen", "abc123def456"))
        assert loaded.ref_code is None
        assert loaded.x_vector_only is True

    def test_named_after_backend_and_model(self, tmp_path: Path) -> None:
        print_ = saved_with_prompt(tmp_path)

        assert print_.cards == {
            ("qwen", "abc123def456"): print_.directory
            / PROMPT_DIR
            / "qwen-abc123def456.npz"
        }
        assert print_.card_for("qwen", "abc123def456").is_file()

    def test_two_models_keep_their_own_card(self, tmp_path: Path) -> None:
        first = saved_with_prompt(tmp_path, fingerprint="aaaaaaaaaaaa")
        second = saved_with_prompt(
            tmp_path, fingerprint="bbbbbbbbbbbb", force=True
        )

        # Re-saving for another model must not cost the first card.
        assert first.card_for("qwen", "aaaaaaaaaaaa").is_file()
        assert second.card_for("qwen", "bbbbbbbbbbbb").is_file()
        assert len(second.cards) == 2

    def test_lookup_is_exact(self, tmp_path: Path) -> None:
        print_ = saved_with_prompt(tmp_path)

        found = find_prompt(
            print_.directory, "qwen", "abc123def456", ref_text=TRANSCRIPT
        )
        assert found == print_.card_for("qwen", "abc123def456")
        assert (
            find_prompt(
                print_.directory, "qwen", "other", ref_text=TRANSCRIPT
            )
            is None
        )
        assert (
            find_prompt(
                print_.directory, "fish", "abc123def456", ref_text=TRANSCRIPT
            )
            is None
        )

    def test_a_card_from_another_clip_is_not_used(self, tmp_path: Path) -> None:
        print_ = saved_with_prompt(tmp_path)

        # Same model, but the voice's transcript changed: the card is stale.
        assert find_prompt(
            print_.directory, "qwen", "abc123def456", ref_text="Autre chose."
        ) is None
        # And so is a card whose audio was replaced.
        print_.wav.write_bytes(b"different audio")
        assert find_prompt(
            print_.directory, "qwen", "abc123def456", ref_text=TRANSCRIPT
        ) is None

    def test_a_voice_without_a_card_has_none(self, tmp_path: Path) -> None:
        saved = make_voice(tmp_path, "v")
        assert load_voice(tmp_path, saved.label).cards == {}

    def test_a_deleted_card_is_not_fatal(self, tmp_path: Path) -> None:
        print_ = saved_with_prompt(tmp_path)
        card = print_.card_for("qwen", "abc123def456")
        assert card is not None
        card.unlink()

        # Derived data: losing it costs one encode, not the voice.
        assert load_voice(tmp_path, print_.label).cards == {}

    def test_an_unreadable_card_is_reported(self, tmp_path: Path) -> None:
        print_ = saved_with_prompt(tmp_path)
        card = print_.card_for("qwen", "abc123def456")
        assert card is not None
        card.write_bytes(b"not an npz")

        with pytest.raises(VoiceNotFoundError, match="unreadable"):
            load_prompt(card)

    def test_the_manifest_does_not_list_cards(self, tmp_path: Path) -> None:
        print_ = saved_with_prompt(tmp_path)
        payload = json.loads(print_.manifest_path.read_text(encoding="utf-8"))

        assert payload["version"] == MANIFEST_VERSION
        assert "prompt" not in payload
        assert "prompts" not in payload

    def test_a_v1_manifest_still_loads(self, tmp_path: Path) -> None:
        directory, wav, manifest = voice_paths(tmp_path, "old")
        directory.mkdir(parents=True)
        wav.write_bytes(b"x")
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "label": "old",
                    "language": "French",
                    "source": "design",
                    "ref_text": TRANSCRIPT,
                }
            ),
            encoding="utf-8",
        )

        loaded = load_voice(tmp_path, "old")
        assert loaded.cards == {}
        assert loaded.ref_text == TRANSCRIPT

    def test_a_contentless_legacy_card_is_listed_but_not_reused(
        self, tmp_path: Path
    ) -> None:
        print_ = saved_with_prompt(tmp_path)
        card = print_.card_for("qwen", "abc123def456")
        assert card is not None
        # A card from before cards recorded what they were derived from.
        legacy = print_.directory / LEGACY_PROMPT_NAME
        np.savez(
            legacy,
            meta=np.array(
                json.dumps(
                    {
                        "version": 1,
                        "x_vector_only": False,
                        "ref_text": TRANSCRIPT,
                        "fingerprint": "abc123def456",
                        "base_model": "tts_base",
                    }
                )
            ),
            ref_code=np.zeros((2, 2), dtype=np.int64),
            ref_spk_embedding=np.zeros(4, dtype=np.float32),
        )

        assert ("qwen", "abc123def456") in cards_in(print_.directory)
        # Provenance cannot be checked, so it is not trusted; the good card is.
        assert find_prompt(
            print_.directory, "qwen", "abc123def456", ref_text=TRANSCRIPT
        ) == card


def test_voice_content_follows_the_clip_and_the_text(tmp_path: Path) -> None:
    first = voice_content(write_clip(tmp_path / "a.wav"), "Bonjour")
    same = voice_content(write_clip(tmp_path / "b.wav"), "Bonjour  ")
    other_text = voice_content(tmp_path / "a.wav", "Autre chose")

    assert first == same  # identical audio, whitespace folded
    assert first != other_text


def test_summary_marks_a_stored_card(tmp_path: Path) -> None:
    assert "clone + qwen card" in saved_with_prompt(tmp_path).summary()


class TestImport:
    def test_a_clip_can_be_registered_without_a_transcript(self, tmp_path: Path) -> None:
        saved = import_voice(
            tmp_path, "raw", (tone(), SAMPLE_RATE), language="French"
        )

        assert saved.source == "import"
        assert saved.ref_text == ""
        assert saved.wav.is_file()
        assert load_voice(tmp_path, "raw").cards == {}
        assert "no transcript" in saved.summary()

    def test_the_transcript_is_stored_as_given(self, tmp_path: Path) -> None:
        saved = import_voice(
            tmp_path,
            "raw",
            (tone(), SAMPLE_RATE),
            language="French",
            ref_text=TRANSCRIPT,
        )

        assert load_voice(tmp_path, saved.label).ref_text == TRANSCRIPT

    def test_designed_or_cloned_voices_still_need_a_transcript(
        self, tmp_path: Path
    ) -> None:
        for source in ("design", "clone"):
            with pytest.raises(TTSError, match="transcript"):
                save_voice(
                    tmp_path,
                    source,
                    tone(),
                    SAMPLE_RATE,
                    language="French",
                    source=source,
                    ref_text="",
                )
