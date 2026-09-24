"""Tests for job configuration, environment loading and language handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import (
    JobConfig,
    Settings,
    language_code_for_name,
    language_name_for_code,
    normalize_language,
    read_env_file,
    release_gpu_memory,
)
from pipeline.errors import ConfigError


class TestNormalizeLanguage:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("fr", "French"),
            ("FR", "French"),
            ("french", "French"),
            ("French", "French"),
            ("fra", "French"),
            ("en", "English"),
            ("zh", "Chinese"),
            ("auto", "Auto"),
            ("Auto", "Auto"),
        ],
    )
    def test_maps_aliases_and_names(self, raw: str, expected: str) -> None:
        assert normalize_language(raw) == expected

    def test_rejects_unknown_language(self) -> None:
        with pytest.raises(ConfigError, match="Unsupported language"):
            normalize_language("klingon")


class TestLanguageCodeRoundTrip:
    @pytest.mark.parametrize("code", ["en", "fr", "de", "ja"])
    def test_name_and_code_round_trip(self, code: str) -> None:
        assert language_code_for_name(language_name_for_code(code)) == code

    def test_auto_and_unknown_map_to_none(self) -> None:
        assert language_code_for_name("auto") is None
        assert language_code_for_name(None) is None
        assert language_code_for_name("Klingon") is None

    def test_code_passthrough_when_already_a_code(self) -> None:
        assert language_name_for_code("fi") == "fi"
        assert language_name_for_code(None) is None


class TestReadEnvFile:
    def test_parses_comments_quotes_and_export(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            "# comment\n"
            "\n"
            "LLM_MODEL='qwen local'\n"
            'LLM_BASE_URL="http://x/v1"\n'
            "export LLM_API_KEY=secret\n"
            "malformed_line\n"
            "EMPTY=\n",
            encoding="utf-8",
        )
        values = read_env_file(env)
        assert values["LLM_MODEL"] == "qwen local"
        assert values["LLM_BASE_URL"] == "http://x/v1"
        assert values["LLM_API_KEY"] == "secret"
        assert values["EMPTY"] == ""
        assert "malformed_line" not in values

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_env_file(tmp_path / "nope.env") == {}


class TestSettings:
    def test_defaults_when_nothing_is_set(self, tmp_path: Path) -> None:
        settings = Settings.from_env(env_file=tmp_path / "absent.env", environ={})
        assert settings.llm_base_url.startswith("http://")
        assert settings.models_dir == Path("models")

    def test_environment_overrides_file(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("LLM_MODEL=from_file\nLOG_JSON=0\n", encoding="utf-8")
        settings = Settings.from_env(
            env_file=env, environ={"LLM_MODEL": "from_env", "LOG_JSON": "true"}
        )
        assert settings.llm_model == "from_env"
        assert settings.log_json is True

    def test_booleans_accept_common_truthy_spellings(self, tmp_path: Path) -> None:
        for raw in ("1", "true", "YES", "on"):
            settings = Settings.from_env(
                env_file=tmp_path / "absent.env", environ={"LOG_JSON": raw}
            )
            assert settings.log_json is True


class TestJobConfigValidation:
    def test_requires_exactly_one_input(self) -> None:
        with pytest.raises(ConfigError, match="exactly one input"):
            JobConfig()
        with pytest.raises(ConfigError, match="exactly one input"):
            JobConfig(video=Path("a.mp4"), audio=Path("a.wav"))

    def test_accepts_video_or_audio(self) -> None:
        assert JobConfig(video=Path("a.mp4")).is_video is True
        assert JobConfig(audio=Path("a.wav")).is_video is False
        assert JobConfig(audio=Path("a.wav")).input_path == Path("a.wav")

    def test_ref_audio_requires_ref_text(self) -> None:
        with pytest.raises(ConfigError, match="ref-text"):
            JobConfig(audio=Path("a.wav"), ref_audio=Path("r.wav"))
        with pytest.raises(ConfigError, match="ref-text"):
            JobConfig(audio=Path("a.wav"), ref_audio=Path("r.wav"), ref_text="  ")

    def test_ref_audio_with_text_is_valid(self) -> None:
        job = JobConfig(audio=Path("a.wav"), ref_audio=Path("r.wav"), ref_text="Bonjour")
        assert job.ref_text == "Bonjour"

    def test_speaker_only_mode_does_not_need_a_reference_transcript(self) -> None:
        """The transcript is ignored in speaker-only mode, so it is not required."""
        job = JobConfig(audio=Path("a.wav"), ref_audio=Path("r.wav"), accent_mode="speaker-only")
        assert job.ref_audio == Path("r.wav")
        assert job.ref_text is None

    def test_preserve_mode_still_requires_the_transcript(self) -> None:
        with pytest.raises(ConfigError, match="ref-text"):
            JobConfig(audio=Path("a.wav"), ref_audio=Path("r.wav"), accent_mode="preserve")

    def test_rejects_identical_source_and_target(self) -> None:
        with pytest.raises(ConfigError, match="source_language and target_language"):
            JobConfig(audio=Path("a.wav"), source_language="en", target_language="en")

    def test_rejects_auto_source_language(self) -> None:
        with pytest.raises(ConfigError, match="explicit language"):
            JobConfig(audio=Path("a.wav"), source_language="auto")

    def test_normalises_languages(self) -> None:
        job = JobConfig(audio=Path("a.wav"), target_language="fr", source_language="en")
        assert job.target_language == "French"
        assert job.source_language == "English"

    def test_stretch_bounds_are_validated(self) -> None:
        with pytest.raises(Exception):
            JobConfig(audio=Path("a.wav"), min_stretch=1.5)
        with pytest.raises(Exception):
            JobConfig(audio=Path("a.wav"), max_stretch=0.5)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("preserve", "preserve"),
            ("icl", "preserve"),
            ("speaker-only", "speaker-only"),
            ("xvector", "speaker-only"),
            ("x-vector", "speaker-only"),
            ("Speaker_Only", "speaker-only"),
        ],
    )
    def test_accent_mode_aliases(self, raw: str, expected: str) -> None:
        assert JobConfig(audio=Path("a.wav"), accent_mode=raw).accent_mode == expected

    def test_unknown_accent_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown accent mode"):
            JobConfig(audio=Path("a.wav"), accent_mode="telepathy")

    def test_accent_mode_defaults_to_preserve(self) -> None:
        assert JobConfig(audio=Path("a.wav")).accent_mode == "preserve"

    def test_expands_user_home(self) -> None:
        job = JobConfig(audio=Path("~/clip.wav"))
        assert "~" not in str(job.input_path)


class TestFingerprint:
    def test_stable_for_equal_settings(self) -> None:
        a = JobConfig(audio=Path("a.wav"), target_language="fr")
        b = JobConfig(audio=Path("a.wav"), target_language="French")
        assert a.fingerprint() == b.fingerprint()

    def test_changes_with_target_language(self) -> None:
        a = JobConfig(audio=Path("a.wav"), target_language="fr")
        b = JobConfig(audio=Path("a.wav"), target_language="de")
        assert a.fingerprint() != b.fingerprint()

    def test_ignores_output_dir_and_force(self) -> None:
        a = JobConfig(audio=Path("a.wav"), output_dir=Path("out1"))
        b = JobConfig(audio=Path("a.wav"), output_dir=Path("out2"), force=True)
        assert a.fingerprint() == b.fingerprint()

    def test_changes_with_donor(self) -> None:
        a = JobConfig(audio=Path("a.wav"))
        b = JobConfig(audio=Path("a.wav"), ref_audio=Path("r.wav"), ref_text="hi")
        assert a.fingerprint() != b.fingerprint()


class TestReleaseGpuMemory:
    def test_returns_a_bool_and_never_raises(self) -> None:
        assert isinstance(release_gpu_memory(), bool)

    def test_is_safe_to_call_repeatedly(self) -> None:
        for _ in range(3):
            release_gpu_memory()
