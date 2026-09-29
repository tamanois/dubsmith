"""Tests for job configuration, environment loading and language handling."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.config import (
    ALLOCATOR_ENV_NAMES,
    DEFAULT_ALLOCATOR_CONF,
    JobConfig,
    Settings,
    configure_allocator,
    device_free_bytes,
    language_code_for_name,
    language_name_for_code,
    normalize_language,
    peak_reserved_bytes,
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

    def test_the_log_file_can_come_from_the_environment(self, tmp_path: Path) -> None:
        settings = Settings.from_env(
            env_file=tmp_path / "absent.env",
            environ={"LOG_FILE": str(tmp_path / "run.log")},
        )

        assert settings.log_file == tmp_path / "run.log"

    def test_without_the_variable_the_log_destination_is_undecided(
        self, tmp_path: Path
    ) -> None:
        # None is not "no log": dub.py reads it as "use the job folder".
        settings = Settings.from_env(env_file=tmp_path / "absent.env", environ={})

        assert settings.log_file is None


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

    def test_the_job_folder_is_one_per_input_stem(self) -> None:
        job = JobConfig(video=Path("input/talk.mp4"), output_dir=Path("out"))

        assert job.job_dir == Path("out/talk")

    def test_ref_audio_requires_ref_text(self) -> None:
        with pytest.raises(ConfigError, match="ref-text"):
            JobConfig(
                audio=Path("a.wav"), ref_audio=Path("r.wav"), accent_mode="preserve"
            )
        with pytest.raises(ConfigError, match="ref-text"):
            JobConfig(
                audio=Path("a.wav"),
                ref_audio=Path("r.wav"),
                accent_mode="preserve",
                ref_text="  ",
            )

    def test_native_mode_does_not_need_a_reference_transcript(self) -> None:
        """The default mode borrows an accent, so the donor needs no transcript."""
        job = JobConfig(audio=Path("a.wav"), ref_audio=Path("r.wav"))
        assert job.accent_mode == "native"
        assert job.ref_text is None

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
            ("native", "native"),
            ("auto", "native"),
            ("generated", "native"),
            ("ancHor", "native"),
            ("preserve", "preserve"),
            ("icl", "preserve"),
            ("speaker-only", "speaker-only"),
            ("xvector", "speaker-only"),
            ("x-vector", "speaker-only"),
            ("Speaker_Only", "speaker-only"),
            ("reference-accent", "reference-accent"),
            ("hybrid", "reference-accent"),
            ("ref-accent", "reference-accent"),
            ("Reference_Accent", "reference-accent"),
        ],
    )
    def test_accent_mode_aliases(self, raw: str, expected: str) -> None:
        # reference-accent additionally demands its two clips.
        extra = (
            {"accent_audio": Path("accent.wav"), "accent_text": "Bonjour"}
            if expected == "reference-accent"
            else {}
        )
        job = JobConfig(audio=Path("a.wav"), accent_mode=raw, **extra)  # type: ignore[arg-type]
        assert job.accent_mode == expected

    def test_unknown_accent_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown accent mode"):
            JobConfig(audio=Path("a.wav"), accent_mode="telepathy")

    def test_accent_mode_defaults_to_native(self) -> None:
        assert JobConfig(audio=Path("a.wav")).accent_mode == "native"


class TestReferenceAccentMode:
    """`reference-accent` splits phonetics from timbre across two clips."""

    @staticmethod
    def build(**overrides: object) -> JobConfig:
        payload: dict[str, object] = {
            "audio": Path("a.wav"),
            "accent_mode": "reference-accent",
            "accent_audio": Path("accent.wav"),
            "accent_text": "Bonjour",
        }
        payload.update(overrides)
        return JobConfig(**payload)  # type: ignore[arg-type]

    def test_accepts_an_accent_clip(self) -> None:
        job = self.build()
        assert job.accent_audio == Path("accent.wav")
        assert job.accent_text == "Bonjour"

    def test_requires_an_accent_clip(self) -> None:
        with pytest.raises(ConfigError, match="requires --accent-audio"):
            self.build(accent_audio=None)

    def test_requires_the_accent_transcript(self) -> None:
        with pytest.raises(ConfigError, match="requires --accent-text"):
            self.build(accent_text=None)
        with pytest.raises(ConfigError, match="requires --accent-text"):
            self.build(accent_text="   ")

    def test_accent_clip_without_the_mode_is_rejected(self) -> None:
        """Silently ignoring --accent-audio would be worse than failing."""
        with pytest.raises(ConfigError, match="only applies to --accent-mode"):
            JobConfig(audio=Path("a.wav"), accent_audio=Path("accent.wav"))

    def test_timbre_reference_does_not_need_a_transcript(self) -> None:
        """The donor clip supplies only an embedding here, so no transcript."""
        job = self.build(ref_audio=Path("donor.wav"))
        assert job.ref_text is None

    def test_auto_donor_needs_neither_reference_nor_transcript(self) -> None:
        job = self.build()
        assert job.ref_audio is None and job.ref_text is None

    def test_existing_modes_are_unaffected(self) -> None:
        assert JobConfig(audio=Path("a.wav")).accent_mode == "native"
        speaker_only = JobConfig(
            audio=Path("a.wav"), ref_audio=Path("r.wav"), accent_mode="speaker-only"
        )
        assert speaker_only.accent_audio is None
        assert speaker_only.ref_text is None

    def test_accent_clip_is_expanded(self) -> None:
        job = self.build(accent_audio=Path("~/accent.wav"))
        assert "~" not in str(job.accent_audio)

    def test_expands_user_home(self) -> None:
        job = JobConfig(audio=Path("~/clip.wav"))
        assert "~" not in str(job.input_path)


class TestAccentBlendOptions:
    """How `reference-accent` builds the voice it clones."""

    @staticmethod
    def build(**overrides: object) -> JobConfig:
        payload: dict[str, object] = {
            "audio": Path("a.wav"),
            "accent_mode": "reference-accent",
            "accent_audio": Path("accent.wav"),
            "accent_text": "Bonjour",
        }
        payload.update(overrides)
        return JobConfig(**payload)  # type: ignore[arg-type]

    def test_defaults_to_two_stage_with_three_takes(self) -> None:
        job = self.build()
        assert job.accent_blend_mode == "two-stage"
        assert job.resolved_blend_takes == 3
        assert job.resolved_blend_iters == 5
        assert job.resolved_blend_ratio == 0.85

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("two-stage", "two-stage"),
            ("two", "two-stage"),
            ("blend", "two-stage"),
            ("Two_Stage", "two-stage"),
            ("single-stage", "single-stage"),
            ("single", "single-stage"),
            ("direct", "single-stage"),
        ],
    )
    def test_blend_aliases(self, raw: str, expected: str) -> None:
        assert self.build(accent_blend=raw).accent_blend_mode == expected

    def test_unknown_blend_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown accent blend"):
            self.build(accent_blend="telepathy")

    def test_takes_are_honoured(self) -> None:
        assert self.build(blend_takes=7).resolved_blend_takes == 7

    def test_takes_below_one_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="at least 1"):
            self.build(blend_takes=0)

    def test_iteration_is_honoured(self) -> None:
        assert self.build(blend_iters=2).resolved_blend_iters == 2

    def test_iters_below_one_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="--blend-iters must be at least 1"):
            self.build(blend_iters=0)

    def test_ratio_is_honoured(self) -> None:
        assert self.build(blend_ratio=0.8).resolved_blend_ratio == 0.8
        # 0 is meaningful: it disables the search rather than falling back.
        assert self.build(blend_ratio=0).resolved_blend_ratio == 0.0

    def test_ratio_outside_zero_to_one_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="between 0 and 1"):
            self.build(blend_ratio=1.5)
        with pytest.raises(ConfigError, match="between 0 and 1"):
            self.build(blend_ratio=-0.1)

    def test_blend_without_the_mode_is_rejected(self) -> None:
        """A silently ignored knob is worse than a rejected one."""
        with pytest.raises(ConfigError, match="--accent-blend only applies"):
            JobConfig(
                audio=Path("a.wav"), accent_mode="preserve", accent_blend="single-stage"
            )

    def test_takes_without_the_mode_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="--blend-takes only applies"):
            JobConfig(audio=Path("a.wav"), accent_mode="preserve", blend_takes=4)

    def test_iters_without_the_mode_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="--blend-iters only applies"):
            JobConfig(audio=Path("a.wav"), accent_mode="preserve", blend_iters=2)

    def test_ratio_without_the_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="--blend-ratio only applies"):
            JobConfig(audio=Path("a.wav"), accent_mode="preserve", blend_ratio=0.5)

    def test_the_native_mode_accepts_the_blend_knobs(self) -> None:
        """`native` blends too, so the knobs must reach it."""
        job = JobConfig(audio=Path("a.wav"), accent_mode="native", blend_takes=5)
        assert job.resolved_blend_takes == 5

    def test_other_modes_keep_the_defaults(self) -> None:
        for mode in ("preserve", "speaker-only"):
            job = JobConfig(audio=Path("a.wav"), accent_mode=mode)
            assert job.accent_blend_mode == "two-stage"
            assert job.resolved_blend_takes == 3
            assert job.resolved_blend_iters == 5
            assert job.resolved_blend_ratio == 0.85

    def test_fingerprint_changes_with_the_blend_strategy(self) -> None:
        """Different strategies must not reuse each other's cached audio."""
        assert self.build().fingerprint() != self.build(
            accent_blend="single-stage"
        ).fingerprint()
        assert self.build().fingerprint() != self.build(blend_takes=5).fingerprint()
        assert self.build().fingerprint() != self.build(blend_iters=2).fingerprint()
        assert self.build().fingerprint() != self.build(blend_ratio=0.9).fingerprint()


class TestAccentMatch:
    """`--accent-match` shapes the generated native anchor."""

    def test_defaults_to_off(self) -> None:
        assert JobConfig(audio=Path("a.wav")).accent_match == "off"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("off", "off"),
            ("none", "off"),
            ("rate", "rate"),
            ("Speed", "rate"),
            ("gender", "gender"),
            ("gender,rate", "gender"),
        ],
    )
    def test_aliases(self, raw: str, expected: str) -> None:
        assert JobConfig(audio=Path("a.wav"), accent_match=raw).accent_match == expected

    def test_unknown_match_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown accent match"):
            JobConfig(audio=Path("a.wav"), accent_match="telepathy")

    def test_rejected_outside_native(self) -> None:
        with pytest.raises(ConfigError, match="--accent-match only applies"):
            JobConfig(audio=Path("a.wav"), accent_mode="preserve", accent_match="rate")
        with pytest.raises(ConfigError, match="--accent-match only applies"):
            JobConfig(
                audio=Path("a.wav"),
                accent_mode="reference-accent",
                accent_audio=Path("a.wav"),
                accent_text="Bonjour",
                accent_match="gender",
            )

    def test_native_rejects_a_supplied_accent_clip(self) -> None:
        with pytest.raises(ConfigError, match="only applies to --accent-mode"):
            JobConfig(
                audio=Path("a.wav"), accent_mode="native", accent_audio=Path("a.wav")
            )


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


class TestAllocatorConfiguration:
    """The caching allocator, configured before the first allocation or not at all.

    Measured failure (2026-09-28, 11-minute video, RTX 3060): a 1.53-1.61 GiB
    request failed against 1.13 GiB of free memory, hundreds of times in a row,
    because the allocator holds freed blocks at the size they were requested.
    """

    @staticmethod
    def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ALLOCATOR_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)

    def test_cuda_gets_expandable_segments(
        self, monkeypatch: pytest.MonkeyPatch, torch_profile
    ) -> None:
        self._clear(monkeypatch)
        torch_profile(kind="cuda")

        assert configure_allocator("cuda:0") == DEFAULT_ALLOCATOR_CONF
        assert os.environ["PYTORCH_ALLOC_CONF"] == DEFAULT_ALLOCATOR_CONF

    def test_a_value_the_user_set_always_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

        assert configure_allocator("cuda") is None
        assert "PYTORCH_ALLOC_CONF" not in os.environ
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"

    def test_cpu_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, torch_profile
    ) -> None:
        self._clear(monkeypatch)
        torch_profile(kind="cpu")

        assert configure_allocator("cpu") is None
        assert "PYTORCH_ALLOC_CONF" not in os.environ

    def test_rocm_keeps_its_defaults(
        self, monkeypatch: pytest.MonkeyPatch, torch_profile
    ) -> None:
        # The option is CUDA-specific, and ROCm answers to the CUDA API, so the
        # exclusion has to be by kind rather than by the requested spelling.
        self._clear(monkeypatch)
        torch_profile(kind="rocm")

        assert configure_allocator("cuda") is None
        assert "PYTORCH_ALLOC_CONF" not in os.environ

    def test_free_memory_is_unknown_where_there_is_none(self, torch_profile) -> None:
        torch_profile(kind="cpu")
        assert device_free_bytes("cpu") is None

    def test_free_memory_reads_the_device_counter(self, torch_profile) -> None:
        fake = torch_profile(kind="cuda")
        assert device_free_bytes("cuda") == fake.cuda.mem_get_info(0)[0]

    def test_peak_reserved_reads_the_device_counter(self, torch_profile) -> None:
        fake = torch_profile(kind="cuda")
        assert peak_reserved_bytes("cuda") == fake.cuda.max_memory_reserved(0)

    def test_a_missing_counter_is_not_an_error(self, torch_profile) -> None:
        '''An older torch, or a backend without the API, reports unknown.'''
        fake = torch_profile(kind="cuda")
        del fake.cuda.mem_get_info
        del fake.cuda.max_memory_reserved

        assert device_free_bytes("cuda") is None
        assert peak_reserved_bytes("cuda") is None


class TestTtsBatch:
    """The line-count ceiling on a synthesis batch."""

    def test_defaults_to_eight(self) -> None:
        assert JobConfig(audio=Path("a.wav")).tts_batch == 8

    def test_bounds_are_validated(self) -> None:
        with pytest.raises(ValidationError):
            JobConfig(audio=Path("a.wav"), tts_batch=0)
        with pytest.raises(ValidationError):
            JobConfig(audio=Path("a.wav"), tts_batch=1000)

    def test_changes_the_fingerprint(self) -> None:
        # The ceiling decides the audio a run produces, so a cached job must not
        # be reused when it changes.
        default = JobConfig(audio=Path("a.wav"))
        smaller = JobConfig(audio=Path("a.wav"), tts_batch=2)
        assert default.fingerprint() != smaller.fingerprint()


class TestLengthRepairOptions:
    """Options controlling the length repair and the word budget."""

    def test_repair_is_on_by_default(self) -> None:
        job = JobConfig(audio=Path("a.wav"))
        assert job.refine is True
        assert job.refine_max_passes == 1
        assert job.refine_threshold == pytest.approx(0.05)

    def test_repair_can_be_switched_off(self) -> None:
        assert JobConfig(audio=Path("a.wav"), refine=False).refine is False

    def test_zero_passes_disables_repair_effectively(self) -> None:
        assert JobConfig(audio=Path("a.wav"), refine_max_passes=0).refine_max_passes == 0

    def test_negative_passes_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JobConfig(audio=Path("a.wav"), refine_max_passes=-1)

    def test_length_budget_defaults_to_words(self) -> None:
        # Every accent mode now has a target-language reference (the accent anchor
        # or clip), so the budget is measured rather than skipped.
        assert JobConfig(audio=Path("a.wav")).length_budget == "words"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("off", "off"),
            ("none", "off"),
            ("seconds", "off"),
            ("words", "words"),
            ("Word", "words"),
            ("Words", "words"),
            ("words", "words"),
        ],
    )
    def test_length_budget_aliases(self, raw: str, expected: str) -> None:
        assert JobConfig(audio=Path("a.wav"), length_budget=raw).length_budget == expected

    def test_unknown_length_budget_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown length budget"):
            JobConfig(audio=Path("a.wav"), length_budget="syllables")


class TestRoomMatchingOptions:
    """Options controlling the room-matching pass (P3d)."""

    def test_dry_is_the_default(self) -> None:
        job = JobConfig(audio=Path("a.wav"))
        assert job.room == "off"
        assert job.room_matching is False
        assert job.room_rt60 is None
        # Reverb is opt-in: 0 dB means none, because the first default (-12 dB) was
        # rejected on a real listen as an audible echo.
        assert job.room_wet_db == pytest.approx(0.0)
        assert job.room_duck_db == pytest.approx(3.0)
        assert job.room_tone_gain_db == pytest.approx(-30.0)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("off", "off"),
            ("none", "off"),
            ("dry", "off"),
            ("False", "off"),
            ("match", "match"),
            ("on", "match"),
            ("True", "match"),
            ("reverb", "match"),
        ],
    )
    def test_room_aliases(self, raw: str, expected: str) -> None:
        assert JobConfig(audio=Path("a.wav"), room=raw).room == expected

    def test_unknown_room_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown room mode"):
            JobConfig(audio=Path("a.wav"), room="concert-hall")

    def test_match_enables_the_property(self) -> None:
        assert JobConfig(audio=Path("a.wav"), room="match").room_matching is True

    def test_room_matching_ducks_the_bed_more_gently_than_the_default(self) -> None:
        job = JobConfig(audio=Path("a.wav"), room="match")
        assert job.room_duck_db < job.duck_db


class TestCliJobConfig:
    """The flag-to-config mapping ``dub.py`` uses, exercised through the real parser.

    These tests exist because a validator that keys off ``model_fields_set`` rather
    than the value rejects an ordinary run: the CLI passes every flag explicitly,
    including its ``None`` defaults.
    """

    def test_the_default_invocation_is_accepted(self) -> None:
        """The Definition of Done's one-liner must build a valid config."""
        import dub

        args = dub.build_dub_parser().parse_args(["--video", "in.mp4", "--target", "fr"])
        job = dub.job_config_from_args(args)

        assert job.accent_mode == "native"
        assert job.room == "off"
        assert job.accent_blend_mode == "two-stage"  # the resolved default

    def test_the_batch_ceiling_is_wired(self) -> None:
        import dub

        args = dub.build_dub_parser().parse_args(
            ["--video", "in.mp4", "--target", "fr", "--tts-batch", "3"]
        )
        assert dub.job_config_from_args(args).tts_batch == 3

    def test_room_match_is_accepted_and_wired(self) -> None:
        import dub

        args = dub.build_dub_parser().parse_args(
            ["--video", "in.mp4", "--target", "fr", "--room", "match"]
        )
        job = dub.job_config_from_args(args)

        assert job.room_matching is True
        assert job.room_duck_db == pytest.approx(3.0)

    def test_overlap_defaults_to_dominant(self) -> None:
        import dub

        args = dub.build_dub_parser().parse_args(["--video", "in.mp4", "--target", "fr"])
        job = dub.job_config_from_args(args)

        assert job.overlap_mode == "dominant"
        assert job.overlap_separation is False
        assert job.overlap_separator == "auto"

    def test_overlap_separate_is_accepted_and_wired(self) -> None:
        import dub

        args = dub.build_dub_parser().parse_args(
            ["--video", "in.mp4", "--target", "fr", "--overlap", "separate"]
        )
        job = dub.job_config_from_args(args)

        assert job.overlap_separation is True

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--accent-blend", "two-stage"),
            ("--blend-takes", "2"),
            ("--blend-iters", "2"),
            ("--blend-ratio", "0.5"),
        ],
    )
    def test_a_blend_flag_without_the_accent_mode_is_still_rejected(
        self, flag: str, value: str
    ) -> None:
        """The guard still does its job for a knob the user actually asked for."""
        import dub

        args = dub.build_dub_parser().parse_args(
            ["--video", "in.mp4", "--target", "fr", "--accent-mode", "preserve", flag, value]
        )
        with pytest.raises(ConfigError, match="only applies to --accent-mode"):
            dub.job_config_from_args(args)

    def test_rt60_override_is_bounded(self) -> None:
        job = JobConfig(audio=Path("a.wav"), room_rt60=1.2)
        assert job.room_rt60 == pytest.approx(1.2)
        with pytest.raises(ValidationError):
            JobConfig(audio=Path("a.wav"), room_rt60=0.0)

    def test_out_of_range_levels_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JobConfig(audio=Path("a.wav"), room_wet_db=3.0)
        with pytest.raises(ValidationError):
            JobConfig(audio=Path("a.wav"), room_duck_db=-1.0)
        with pytest.raises(ValidationError):
            JobConfig(audio=Path("a.wav"), room_tone_gain_db=6.0)

    def test_fingerprint_changes_with_the_room_mode(self) -> None:
        dry = JobConfig(audio=Path("a.wav"))
        roomed = JobConfig(audio=Path("a.wav"), room="match")
        assert dry.fingerprint() != roomed.fingerprint()

    def test_fingerprint_changes_with_the_room_knobs(self) -> None:
        base = JobConfig(audio=Path("a.wav"), room="match")
        louder = JobConfig(audio=Path("a.wav"), room="match", room_wet_db=-6.0)
        assert base.fingerprint() != louder.fingerprint()
