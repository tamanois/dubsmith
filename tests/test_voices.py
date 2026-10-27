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
    MANIFEST_NAME,
    WAV_NAME,
    default_label,
    delete_voice,
    list_voices,
    load_voice,
    save_voice,
    slugify_label,
    unique_label,
    voice_paths,
)

SAMPLE_RATE = 24_000
TRANSCRIPT = "Bonjour tout le monde."


def tone(seconds: float = 1.5, hz: float = 220.0) -> np.ndarray:
    """A deterministic clip with a known pitch and length."""
    t = np.linspace(0.0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


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
        assert payload["version"] == 1
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
