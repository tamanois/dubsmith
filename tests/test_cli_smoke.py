"""CLI-surface smoke tests: no GPU, no models, no ffmpeg.

The unit suites cover the pipeline internals; these cover the command surface a
user actually types and the environment report they are told to paste into a bug
report. That matters because the CLI passes *every* flag explicitly - including
its defaults - so a flag can silently stop reaching the job config without any
pipeline test noticing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dub import (
    _parse_emotions,
    _say_lines,
    build_dub_parser,
    build_say_parser,
    build_voices_parser,
    job_config_from_args,
    log_file_from,
    main,
    render_devices,
    render_environment,
)
from pipeline import __version__
from pipeline.config import JobConfig, Settings
from pipeline.errors import ConfigError, VoiceNotFoundError
from pipeline.models import REGISTRY, model_fingerprint
from pipeline.voices import (
    PROMPT_DIR,
    PromptData,
    load_voice,
    prompt_path,
    save_voice,
)

SAMPLE_RATE = 24_000


def _saved_voice(root: Path, label: str = "warm-fr"):
    """A voice print with no model involvement, for CLI mapping tests."""
    t = np.linspace(0.0, 1.5, int(1.5 * SAMPLE_RATE), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return save_voice(
        root,
        label,
        samples,
        SAMPLE_RATE,
        language="French",
        source="design",
        ref_text="Bonjour tout le monde.",
        instruct="a warm voice",
    )


def _imported_voice(root: Path, label: str = "silent-fr"):
    """An imported clip saved without a transcript, for accent-anchor tests."""
    t = np.linspace(0.0, 1.5, int(1.5 * SAMPLE_RATE), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return save_voice(
        root,
        label,
        samples,
        SAMPLE_RATE,
        language="French",
        source="import",
        ref_text="",
    )


def test_models_list_never_downloads(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models", "--list", "--models-dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "whisper_large_v3" in out
    assert "missing" in out


def test_version_flag_reports_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_info_reports_the_environment_without_loading_a_model(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["info", "--models-dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    # The four answers a bug report needs before anything else.
    assert "resolved" in out
    assert "tts dtype" in out
    assert "asr device" in out
    assert "speechbrain" in out


def test_info_reports_what_an_unusable_dtype_will_really_do(tmp_path) -> None:
    # fp16 cannot run on CPU, so the report must show the dtype that will be used
    # rather than the one that was asked for.
    text = render_environment(tmp_path, device="cpu", dtype="fp16")

    assert "float32" in text
    assert "requested: fp16" in text


def test_render_environment_lists_every_registered_model(tmp_path) -> None:
    text = render_environment(tmp_path)

    for name in REGISTRY:
        assert name in text


def test_dtype_reaches_the_job_config() -> None:
    args = build_dub_parser().parse_args(["--video", "in.mp4", "--dtype", "fp16"])

    assert job_config_from_args(args).dtype == "fp16"


def test_devices_reach_the_job_config() -> None:
    args = build_dub_parser().parse_args(
        ["--video", "in.mp4", "--devices", "cuda:0,cuda:1"]
    )

    job = job_config_from_args(args)
    assert job.devices == ["cuda:0", "cuda:1"]
    assert job.primary_device == "cuda:0"


def test_devices_default_to_the_single_device_path() -> None:
    args = build_dub_parser().parse_args(["--video", "in.mp4"])

    job = job_config_from_args(args)
    assert job.devices is None
    assert job.resolved_devices == [job.device]


def test_devices_reject_a_mixed_auto() -> None:
    args = build_dub_parser().parse_args(
        ["--video", "in.mp4", "--devices", "auto,cuda:1"]
    )

    with pytest.raises(ConfigError, match="cannot mix 'auto'"):
        job_config_from_args(args)


def test_info_reports_the_fan_out(tmp_path) -> None:
    text = render_environment(tmp_path, devices=["cuda:0", "cuda:1"])

    assert "2 workers" in text
    assert "cuda:0, cuda:1" in text


def test_dtype_and_device_default_to_auto() -> None:
    args = build_dub_parser().parse_args(["--video", "in.mp4", "--target", "fr"])
    job = job_config_from_args(args)

    assert job.dtype == "auto"
    assert job.device == "auto"


# --------------------------------------------------------------------------- #
# Device listing and validation
# --------------------------------------------------------------------------- #


def test_devices_lists_the_supported_accelerators(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["devices"]) == 0

    out = capsys.readouterr().out
    assert "Supported backends" in out
    assert "Vulkan is not a backend" in out
    assert "Detected devices" in out
    assert "Suggested command" in out


def test_devices_prints_one_row_per_card(torch_profile) -> None:
    torch_profile(
        kind="cuda",
        devices=[
            {"name": "RTX 3060 A", "memory": 12 * 1024**3, "capability": (8, 6)},
            {"name": "RTX 3060 B", "memory": 8 * 1024**3, "capability": (7, 5)},
        ],
    )

    text = render_devices()

    assert "cuda:0" in text
    assert "cuda:1" in text
    assert "RTX 3060 A" in text
    assert "RTX 3060 B" in text
    # Each card gets its own row, and the fan-out suggestion names both.
    assert "--devices cuda:0,cuda:1" in text
    # Capability is per card: Ampere gets bf16, Turing does not.
    assert "bf16 yes" in text
    assert "bf16 no" in text
    assert "tts fp16" in text


def test_devices_marks_rocm_cards(torch_profile) -> None:
    torch_profile(
        kind="rocm",
        devices=[
            {"name": "RX 7900 XTX", "memory": 24 * 1024**3, "capability": (9, 4)}
        ],
    )

    text = render_devices()

    assert "cuda:0" in text
    assert "(ROCm)" in text
    assert "RX 7900 XTX" in text


def test_devices_falls_back_to_cpu_when_no_gpu(torch_profile) -> None:
    torch_profile(kind="cpu")

    text = render_devices()

    assert "none - everything will run on the CPU" in text
    assert "--device cpu" in text


def test_devices_explains_asr_fallback_for_metal(torch_profile) -> None:
    torch_profile(kind="mps")

    text = render_devices()

    assert "mps" in text
    assert "ASR stays on the CPU" in text


def test_info_points_at_the_device_listing(tmp_path) -> None:
    text = render_environment(tmp_path)

    assert "python dub.py devices" in text


@pytest.mark.parametrize("flag", ["--device", "--devices"])
def test_the_cli_rejects_vulkan_before_any_stage_runs(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_dub_parser().parse_args(["--video", "in.mp4", flag, "vulkan0"])

    assert exit_info.value.code == 2
    assert "Vulkan is not a supported backend" in capsys.readouterr().err


def test_devices_reject_vulkan_inside_a_list(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_dub_parser().parse_args(
            ["--video", "in.mp4", "--devices", "cuda:0,vulkan0"]
        )

    assert exit_info.value.code == 2
    assert "Vulkan is not a supported backend" in capsys.readouterr().err


def test_the_rocm_alias_reaches_the_config_as_cuda() -> None:
    args = build_dub_parser().parse_args(
        ["--video", "in.mp4", "--device", "rocm:1"]
    )

    assert job_config_from_args(args).device == "cuda:1"


# --------------------------------------------------------------------------- #
# Voice library
# --------------------------------------------------------------------------- #


def test_voices_list_never_loads_a_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["voices", "--list", "--voices-dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "Saved voices" in out
    assert "none yet" in out


def test_voices_parser_defaults_to_listing() -> None:
    args = build_voices_parser().parse_args([])
    assert not args.design and not args.clone and not args.list


def test_voice_flag_becomes_the_reference_clip(tmp_path: Path) -> None:
    saved = _saved_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--voice",
            "warm fr",  # case- and space-insensitive
            "--voices-dir",
            str(tmp_path),
        ]
    )

    job = job_config_from_args(args)

    assert job.ref_audio == saved.wav
    assert job.ref_text == saved.ref_text


def test_voice_flag_also_supplies_the_preserve_transcript(tmp_path: Path) -> None:
    _saved_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--voice",
            "warm-fr",
            "--accent-mode",
            "preserve",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    # No ConfigError: 'preserve' needs a transcript and the voice print has one.
    assert job_config_from_args(args).ref_text == "Bonjour tout le monde."


def test_voice_flag_rejects_an_unknown_label(tmp_path: Path) -> None:
    args = build_dub_parser().parse_args(
        ["--video", "in.mp4", "--voice", "nope", "--voices-dir", str(tmp_path)]
    )

    with pytest.raises(VoiceNotFoundError, match="voices --list"):
        job_config_from_args(args)


def test_voice_flag_conflicts_with_ref_audio(tmp_path: Path) -> None:
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--voice",
            "warm-fr",
            "--ref-audio",
            "me.wav",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ConfigError, match="cannot be combined"):
        job_config_from_args(args)


def test_voice_in_the_target_language_clones_directly(tmp_path: Path) -> None:
    _saved_voice(tmp_path)  # French

    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert job_config_from_args(args).accent_mode == "preserve"


def test_voice_in_another_language_keeps_the_native_accent(tmp_path: Path) -> None:
    _saved_voice(tmp_path)  # French

    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "en",
            "--voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert job_config_from_args(args).accent_mode == "native"


def test_explicit_accent_mode_wins_over_the_voice_default(tmp_path: Path) -> None:
    _saved_voice(tmp_path)

    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--voice",
            "warm-fr",
            "--accent-mode",
            "native",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert job_config_from_args(args).accent_mode == "native"


def test_without_voice_the_accent_default_is_unchanged() -> None:
    args = build_dub_parser().parse_args(["--video", "in.mp4", "--target", "fr"])

    assert job_config_from_args(args).accent_mode == "native"


# --------------------------------------------------------------------------- #
# Reference accent from a saved voice
# --------------------------------------------------------------------------- #


def test_accent_voice_supplies_the_reference_assets(tmp_path: Path) -> None:
    saved = _saved_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--accent-mode",
            "reference-accent",
            "--accent-voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    job = job_config_from_args(args)
    assert job.accent_audio == saved.wav
    assert job.accent_text == saved.ref_text


def test_accent_voice_implies_reference_accent(tmp_path: Path) -> None:
    _saved_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--accent-voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert job_config_from_args(args).accent_mode == "reference-accent"


def test_accent_voice_conflicts_with_the_explicit_accent_clip(tmp_path: Path) -> None:
    _saved_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--accent-mode",
            "reference-accent",
            "--accent-voice",
            "warm-fr",
            "--accent-audio",
            "nat.wav",
            "--accent-text",
            "Bonjour",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ConfigError, match="cannot be combined"):
        job_config_from_args(args)


def test_accent_voice_conflicts_with_another_accent_mode(tmp_path: Path) -> None:
    _saved_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--accent-mode",
            "native",
            "--accent-voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ConfigError, match="implies --accent-mode reference-accent"):
        job_config_from_args(args)


def test_accent_voice_rejects_an_unknown_label(tmp_path: Path) -> None:
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--accent-voice",
            "nope",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(VoiceNotFoundError, match="voices --list"):
        job_config_from_args(args)


def test_accent_voice_needs_a_transcript(tmp_path: Path) -> None:
    _imported_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--accent-voice",
            "silent-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ConfigError, match="no transcript"):
        job_config_from_args(args)


def test_accent_voice_must_speak_the_target_language(tmp_path: Path) -> None:
    _saved_voice(tmp_path)  # French
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "en",
            "--accent-voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ConfigError, match="native English accent"):
        job_config_from_args(args)


def test_voice_and_accent_voice_can_be_combined(tmp_path: Path) -> None:
    saved = _saved_voice(tmp_path)
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--voice",
            "warm-fr",
            "--accent-voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    job = job_config_from_args(args)
    assert job.ref_audio == saved.wav
    assert job.accent_mode == "reference-accent"
    assert job.accent_audio == saved.wav


def test_say_parser_accepts_a_saved_voice() -> None:
    args = build_say_parser().parse_args(["--voice", "warm-fr", "--text", "Bonjour"])

    assert args.voice == "warm-fr"
    assert args.text == "Bonjour"
    assert args.lang is None


def test_say_parser_accepts_an_inline_design() -> None:
    args = build_say_parser().parse_args(
        ["--design", "a warm voice", "--text", "Bonjour", "--lang", "fr"]
    )

    assert args.design == "a warm voice"
    assert args.lang == "fr"


def test_say_without_a_source_is_a_clear_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["say", "--text", "Bonjour"])

    assert code == 1
    assert "One of --voice, --design or --clone is required" in capsys.readouterr().err


def test_say_parser_accepts_customvoice_speaker() -> None:
    args = build_say_parser().parse_args(
        ["--backend", "customvoice", "--speaker", "Vivian", "--lang", "en", "--text", "Hi"]
    )

    assert args.backend == "customvoice"
    assert args.speaker == "Vivian"


def test_say_parser_accepts_a_verbatim_style_instruction() -> None:
    args = build_say_parser().parse_args(
        [
            "--backend",
            "customvoice",
            "--speaker",
            "Ryan",
            "--lang",
            "en",
            "--text",
            "Hi",
            "--style-instruct",
            "Speak with restrained anger.",
            "--style-instruct",
            "Speak calmly.",
        ]
    )

    assert args.style_instructs == [
        "Speak with restrained anger.",
        "Speak calmly.",
    ]


def test_style_instruct_is_rejected_for_other_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--style-instruct",
            "Speak slowly.",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "--style-instruct is only valid" in capsys.readouterr().err


def test_style_instruct_must_match_the_line_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--backend",
            "customvoice",
            "--speaker",
            "Vivian",
            "--lang",
            "en",
            "--text",
            "one\\ntwo",
            "--style-instruct",
            "a",
            "--style-instruct",
            "b",
            "--style-instruct",
            "c",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "--style-instruct: 3 value(s) for 2 line(s)" in capsys.readouterr().err


def test_style_instruct_overrides_prosody(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spoken: list[str] = []

    class FakeEngine:
        sample_rate = 24000

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def speakers(self) -> list[str]:
            return ["Ryan"]

        def generate(
            self, text: str, speaker: str, instruct: str, language: str
        ) -> tuple[np.ndarray, int]:
            spoken.append(instruct)
            return np.zeros(240, dtype=np.float32), 24000

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.tts.CustomVoiceEngine", FakeEngine)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: tmp_path)

    code = main(
        [
            "say",
            "--backend",
            "customvoice",
            "--speaker",
            "Ryan",
            "--lang",
            "en",
            "--text",
            "Hi",
            "--style-instruct",
            "Speak like a robot.",
            "--prosody",
            "emotion=happy",
        ]
    )

    assert code == 0
    assert spoken == ["Speak like a robot."]
    assert "--style-instruct overrides" in capsys.readouterr().err


def test_say_parser_accepts_style_plan_flags() -> None:
    args = build_say_parser().parse_args(
        [
            "--backend",
            "customvoice",
            "--speaker",
            "Ryan",
            "--lang",
            "en",
            "--text",
            "hi",
            "--style-plan",
            "llm",
            "--style-plan-scope",
            "whole",
            "--plan-chunk",
            "4",
        ]
    )

    assert args.style_plan == "llm"
    assert args.style_plan_scope == "whole"
    assert args.plan_chunk == 4


def test_style_plan_is_rejected_for_other_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--style-plan",
            "llm",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "--style-plan is only valid" in capsys.readouterr().err


def test_style_plan_scope_requires_style_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--backend",
            "customvoice",
            "--speaker",
            "Ryan",
            "--lang",
            "en",
            "--text",
            "hi",
            "--style-plan-scope",
            "whole",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "--style-plan-scope only applies" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("per-line", ["Direction for one", "Direction for two"]),
        ("whole", ["Whole direction.", "Whole direction."]),
    ],
)
def test_style_plan_writes_the_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scope: str,
    expected: list[str],
) -> None:
    spoken: list[str] = []

    class FakeEngine:
        sample_rate = 24000

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def speakers(self) -> list[str]:
            return ["Ryan"]

        def generate(
            self, text: str, speaker: str, instruct: str, language: str
        ) -> tuple[np.ndarray, int]:
            spoken.append(instruct)
            return np.zeros(240, dtype=np.float32), 24000

    class FakePlanner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def plan_lines(self, texts, *, speaker=None, language=None):
            return [f"Direction for {text}" for text in texts]

        def plan_whole(self, texts, *, speaker=None, language=None):
            return "Whole direction."

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.tts.CustomVoiceEngine", FakeEngine)
    monkeypatch.setattr("pipeline.prosody.StylePlanner", FakePlanner)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: tmp_path)

    code = main(
        [
            "say",
            "--backend",
            "customvoice",
            "--speaker",
            "Ryan",
            "--lang",
            "en",
            "--text",
            "one\\ntwo",
            "--style-plan",
            "llm",
            "--style-plan-scope",
            scope,
        ]
    )

    assert code == 0
    assert spoken == expected


def test_style_instruct_overrides_style_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spoken: list[str] = []

    class FakeEngine:
        sample_rate = 24000

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def speakers(self) -> list[str]:
            return ["Ryan"]

        def generate(
            self, text: str, speaker: str, instruct: str, language: str
        ) -> tuple[np.ndarray, int]:
            spoken.append(instruct)
            return np.zeros(240, dtype=np.float32), 24000

    class FakePlanner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def plan_lines(self, texts, **kwargs):
            return ["Should not run." for _ in texts]

        def plan_whole(self, texts, **kwargs):
            return "Should not run."

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.tts.CustomVoiceEngine", FakeEngine)
    monkeypatch.setattr("pipeline.prosody.StylePlanner", FakePlanner)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: tmp_path)

    code = main(
        [
            "say",
            "--backend",
            "customvoice",
            "--speaker",
            "Ryan",
            "--lang",
            "en",
            "--text",
            "Hi",
            "--style-instruct",
            "Speak like a robot.",
            "--style-plan",
            "llm",
        ]
    )

    assert code == 0
    assert spoken == ["Speak like a robot."]
    assert "--style-instruct overrides --style-plan" in capsys.readouterr().err


def test_say_text_treats_a_literal_backslash_n_as_a_line_break() -> None:
    assert _say_lines("one\\ntwo") == ["one", "two"]
    assert _say_lines("one\ntwo") == ["one", "two"]
    assert _say_lines("  one  \\n  two  ") == ["one", "two"]


def test_customvoice_requires_a_speaker(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["say", "--backend", "customvoice", "--lang", "en", "--text", "hi"])

    assert code == 1
    assert "--speaker" in capsys.readouterr().err


def test_speaker_is_rejected_for_other_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["say", "--voice", "warm-fr", "--speaker", "Vivian", "--text", "hi"])

    assert code == 1
    assert "--speaker is only valid" in capsys.readouterr().err


def test_say_parser_requires_text() -> None:
    with pytest.raises(SystemExit):
        build_say_parser().parse_args(["--voice", "warm-fr"])


def test_say_inline_needs_a_language(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["say", "--design", "a voice", "--text", "hi", "--voices-dir", str(tmp_path)]
    )

    assert code == 1
    assert "--lang is required" in capsys.readouterr().err


def test_voices_design_needs_a_prompt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["voices", "--design", "--lang", "fr", "--voices-dir", str(tmp_path)])

    assert code == 1
    assert "--prompt" in capsys.readouterr().err


def test_voices_clone_needs_clone_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "voices",
            "--clone",
            "--audio",
            "me.wav",
            "--lang",
            "fr",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "--clone-text" in capsys.readouterr().err


def _write_clip(path: Path, seconds: float = 4.0) -> Path:
    """A real, decodable clip so ``--add`` can read it back."""
    from pipeline.media import write_audio

    t = np.linspace(0.0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    return write_audio(path, samples, SAMPLE_RATE)


def test_voices_add_needs_a_language_or_a_transcription(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = _write_clip(tmp_path / "me.wav")

    code = main(["voices", "--add", str(clip), "--voices-dir", str(tmp_path / "v")])

    assert code == 1
    assert "--lang" in capsys.readouterr().err


def test_voices_add_registers_the_clip_without_building_an_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _write_clip(tmp_path / "me.wav")
    root = tmp_path / "voices"

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("--add must not build a synthesiser")

    monkeypatch.setattr("pipeline.tts.TTSEngine", explode)

    code = main(
        [
            "voices",
            "--add",
            str(clip),
            "--clone-text",
            "Bonjour tout le monde.",
            "--lang",
            "fr",
            "--label",
            "me",
            "--voices-dir",
            str(root),
        ]
    )

    assert code == 0
    saved = load_voice(root, "me")
    assert saved.source == "import"
    assert saved.ref_text == "Bonjour tout le monde."
    assert saved.cards == {}
    assert saved.wav.is_file()


def test_voices_add_rejects_transcribe_alongside_clone_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = _write_clip(tmp_path / "me.wav")

    code = main(
        [
            "voices",
            "--add",
            str(clip),
            "--transcribe",
            "--clone-text",
            "Bonjour.",
            "--lang",
            "fr",
            "--voices-dir",
            str(tmp_path / "voices"),
        ]
    )

    assert code == 1
    assert "alternatives" in capsys.readouterr().err


def test_voices_add_rejects_speak(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = _write_clip(tmp_path / "me.wav")

    code = main(
        [
            "voices",
            "--add",
            str(clip),
            "--lang",
            "fr",
            "--speak",
            "Bonjour.",
            "--voices-dir",
            str(tmp_path / "voices"),
        ]
    )

    assert code == 1
    assert "--speak" in capsys.readouterr().err


class _FakeCloneEngine:
    """Records which preparation path ``say --voice`` took."""

    sample_rate = 24_000

    instances: list["_FakeCloneEngine"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        type(self).instances.append(self)

    def prepare_prompt_from_data(self, data: object, *, seconds: float = 0.0) -> None:
        self.calls.append("stored")

    def prepare_prompt_from_file(self, audio: object, text: str) -> None:
        self.calls.append("encode")

    def prompt_to_data(self, **kwargs: object) -> object:
        return PromptData(
            ref_code=np.zeros((2, 2), dtype=np.int64),
            ref_spk_embedding=np.zeros(4, dtype=np.float32),
            ref_text="Bonjour.",
            fingerprint=str(kwargs.get("fingerprint") or ""),
        )

    def synthesize(
        self, texts: list[str], *, language: str
    ) -> list[np.ndarray]:
        self.calls.append("synth")
        return [np.zeros(240, dtype=np.float32) for _ in texts]


def _clone_voice(
    root: Path,
    *,
    with_prompt: bool,
    models_dir: Path,
    fingerprint: str | None = None,
):
    stamp = (
        fingerprint
        if fingerprint is not None
        else model_fingerprint("tts_base", models_dir)
    )
    t = np.linspace(0.0, 1.5, int(1.5 * SAMPLE_RATE), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return save_voice(
        root,
        "me",
        samples,
        SAMPLE_RATE,
        language="French",
        source="clone",
        ref_text="Bonjour.",
        prompt=(
            PromptData(
                ref_code=np.zeros((2, 2), dtype=np.int64),
                ref_spk_embedding=np.zeros(4, dtype=np.float32),
                ref_text="Bonjour.",
                fingerprint=stamp,
            )
            if with_prompt
            else None
        ),
    )


def test_say_voice_uses_the_stored_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "voices"
    models = tmp_path / "models"
    _clone_voice(root, with_prompt=True, models_dir=models)
    _FakeCloneEngine.instances.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.tts.TTSEngine", _FakeCloneEngine)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: models)

    code = main(
        [
            "say",
            "--voice",
            "me",
            "--text",
            "Bonjour.",
            "--voices-dir",
            str(root),
            "--models-dir",
            str(models),
        ]
    )

    assert code == 0
    engine = _FakeCloneEngine.instances[-1]
    # No encode: the library already holds the encoding, which is the whole point.
    assert "encode" not in engine.calls
    assert engine.calls[0] == "stored"


def test_say_voice_caches_the_encoding_on_first_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "voices"
    models = tmp_path / "models"
    saved = _clone_voice(root, with_prompt=False, models_dir=models)
    assert saved.cards == {}
    _FakeCloneEngine.instances.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.tts.TTSEngine", _FakeCloneEngine)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: models)

    code = main(
        [
            "say",
            "--voice",
            "me",
            "--text",
            "Bonjour.",
            "--voices-dir",
            str(root),
            "--models-dir",
            str(models),
        ]
    )

    assert code == 0
    engine = _FakeCloneEngine.instances[-1]
    assert engine.calls[:2] == ["encode", "synth"]
    # Encoded once, then written back: the next run needs no encode at all.
    card = prompt_path(
        saved.directory,
        "qwen",
        model_fingerprint("tts_base", models),
    )
    assert card.is_file()
    assert card.parent.name == PROMPT_DIR
    assert load_voice(root, "me").cards


def _voice_job(root: Path, models: Path) -> JobConfig:
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--voice",
            "me",
            "--voices-dir",
            str(root),
        ]
    )
    return job_config_from_args(
        args, settings=Settings(voices_dir=root, models_dir=models)
    )


def test_a_dub_reuses_the_voice_encoding(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    models = tmp_path / "models"
    _clone_voice(root, with_prompt=True, models_dir=models)

    job = _voice_job(root, models)

    assert job.ref_prompt == prompt_path(
        root / "me", "qwen", model_fingerprint("tts_base", models)
    )
    assert job.ref_prompt_seconds == pytest.approx(1.5, abs=0.05)
    # Same language as the target: clone the voice directly, no anchor needed.
    assert job.accent_mode == "preserve"


def test_a_card_for_another_model_is_not_used_and_is_kept(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    models = tmp_path / "models"
    saved = _clone_voice(
        root, with_prompt=True, models_dir=models, fingerprint="aaaaaaaaaaaa"
    )
    other = saved.card_for("qwen", "aaaaaaaaaaaa")
    assert other is not None and other.is_file()

    job = _voice_job(root, models)

    # Re-encoding is cheap and correct; another checkpoint's card is neither used
    # nor discarded, so switching models back is instant.
    assert job.ref_prompt is None
    assert job.ref_audio == saved.wav
    assert other.is_file()
    assert len(load_voice(root, "me").cards) == 1


def test_a_voice_without_a_transcript_uses_speaker_only(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    models = tmp_path / "models"
    t = np.linspace(0.0, 1.5, int(1.5 * SAMPLE_RATE), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    save_voice(
        root,
        "me",
        samples,
        SAMPLE_RATE,
        language="French",
        source="import",
        ref_text="",
    )

    job = _voice_job(root, models)

    assert job.accent_mode == "speaker-only"
    assert job.ref_text == ""


def test_native_is_the_default_accent_mode() -> None:
    args = build_dub_parser().parse_args(["--video", "in.mp4", "--target", "fr"])
    job = job_config_from_args(args)

    assert job.accent_mode == "native"
    assert job.accent_match == "off"


def test_accent_match_reaches_the_job_config() -> None:
    args = build_dub_parser().parse_args(
        ["--video", "in.mp4", "--target", "fr", "--accent-match", "gender"]
    )

    assert job_config_from_args(args).accent_match == "gender"


def test_accent_match_is_rejected_outside_native() -> None:
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--accent-mode",
            "preserve",
            "--accent-match",
            "rate",
        ]
    )
    with pytest.raises(ConfigError, match="--accent-match only applies"):
        job_config_from_args(args)


def _log_destination(
    extra: list[str], settings_file: Path | None = None
) -> Path | None:
    """Resolve a command line's run log the way ``cmd_dub`` does."""
    args = build_dub_parser().parse_args(["--video", "input/talk.mp4", *extra])
    settings = Settings(log_file=settings_file)
    return log_file_from(args, settings, job_config_from_args(args))


def test_the_run_log_lands_in_the_job_folder_by_default() -> None:
    # One place to look for everything a run produced, including what it said.
    assert _log_destination([]) == Path("output/talk/run.log")


def test_an_explicit_log_file_wins() -> None:
    assert _log_destination(["--log-file", "logs/nightly.log"]) == Path(
        "logs/nightly.log"
    )


def test_the_environment_path_is_used_when_the_flag_is_absent() -> None:
    assert _log_destination([], settings_file=Path("from-env.log")) == Path(
        "from-env.log"
    )


def test_no_log_file_beats_both_other_destinations() -> None:
    path = _log_destination(
        ["--log-file", "logs/nightly.log", "--no-log-file"],
        settings_file=Path("from-env.log"),
    )

    assert path is None


def test_anchors_lists_without_a_model(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["anchors", "--list", "--models-dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "Accent anchors" in out
    assert "none yet" in out


def test_anchors_generate_without_the_model_names_the_download(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["anchors", "--generate", "--lang", "fr", "--models-dir", str(tmp_path)]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "tts_voice_design" in err
    assert "dub.py models --download" in err


def test_an_unknown_dtype_is_rejected_by_the_cli() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_dub_parser().parse_args(["--video", "in.mp4", "--dtype", "float8"])

    assert exit_info.value.code == 2


def test_dtype_is_normalised_in_the_config() -> None:
    assert JobConfig(audio="a.wav", dtype="BFloat16").dtype == "bf16"


def test_expressive_accent_defaults_to_the_donor() -> None:
    args = build_dub_parser().parse_args(
        ["--video", "in.mp4", "--target", "fr", "--backend", "chatterbox"]
    )
    assert job_config_from_args(args).expressive_accent == "donor"


def test_expressive_accent_blend_reaches_the_job_config() -> None:
    args = build_dub_parser().parse_args(
        [
            "--video",
            "in.mp4",
            "--target",
            "fr",
            "--backend",
            "fish",
            "--expressive-accent",
            "blend",
        ]
    )
    assert job_config_from_args(args).expressive_accent == "blend"


def test_say_parser_accepts_cosyvoice3_and_emotion() -> None:
    args = build_say_parser().parse_args(
        [
            "--voice",
            "x",
            "--text",
            "hi",
            "--backend",
            "cosyvoice3",
            "--emotion",
            "calm",
        ]
    )

    assert args.backend == "cosyvoice3"
    assert args.emotions == ["calm"]


def test_say_parser_defaults_to_qwen() -> None:
    args = build_say_parser().parse_args(["--voice", "x", "--text", "hi"])

    assert args.backend == "qwen"
    assert args.emotions is None
    assert args.prosody is None
    assert args.prosody_plan == "off"


@pytest.mark.parametrize("backend", ["cosyvoice3", "chatterbox"])
def test_say_parser_accepts_the_new_backends(backend: str) -> None:
    args = build_say_parser().parse_args(
        ["--voice", "x", "--text", "hi", "--backend", backend]
    )

    assert args.backend == backend


def test_say_parser_accepts_logging_flags() -> None:
    args = build_say_parser().parse_args(
        ["--voice", "x", "--text", "hi", "-v", "--no-log-file"]
    )

    assert args.verbose is True
    assert args.log_file is False


def test_say_parser_accepts_prosody_and_plan() -> None:
    args = build_say_parser().parse_args(
        [
            "--voice",
            "x",
            "--text",
            "hi",
            "--backend",
            "cosyvoice3",
            "--prosody",
            "emotion=happy, pace=fast",
            "--prosody",
            "breath",
            "--prosody-plan",
            "llm",
        ]
    )

    assert args.prosody == ["emotion=happy, pace=fast", "breath"]
    assert args.prosody_plan == "llm"


def test_say_qwen_rejects_prosody(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--prosody",
            "emotion=happy",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "--prosody" in capsys.readouterr().err


def test_parse_emotions_flattens_repeatable_and_comma_values() -> None:
    assert _parse_emotions(["calm, excited"]) == ["calm", "excited"]
    assert _parse_emotions(["calm", "excited"]) == ["calm", "excited"]
    assert _parse_emotions(None) is None
    assert _parse_emotions([]) is None
    assert _parse_emotions(["  "]) is None


def test_say_cosyvoice3_rejects_design(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "say",
            "--design",
            "a voice",
            "--text",
            "hi",
            "--lang",
            "fr",
            "--backend",
            "cosyvoice3",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_say_qwen_rejects_emotion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--emotion",
            "calm",
            "--voices-dir",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "--emotion" in capsys.readouterr().err


def test_an_unknown_dtype_is_rejected_in_the_config() -> None:
    with pytest.raises(ConfigError, match="Unknown dtype"):
        JobConfig(audio="a.wav", dtype="float8")


def test_dtype_is_part_of_the_fingerprint() -> None:
    # A different precision is a different synthesis, so the cache must not be
    # reused across a dtype change - silently serving fp32 clips for an fp16 run
    # (or the reverse) would be invisible in the output.
    base = JobConfig(audio="a.wav")

    assert base.fingerprint() != JobConfig(audio="a.wav", dtype="fp32").fingerprint()


@pytest.mark.parametrize("backend", ["fish"])
def test_say_parser_accepts_the_fish_backend(backend: str) -> None:
    args = build_say_parser().parse_args(
        ["--voice", "x", "--text", "hi", "--backend", backend]
    )

    assert args.backend == backend


def test_fish_sampling_flags_are_backend_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--fish-temperature",
            "0.5",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "--fish-temperature is only valid" in capsys.readouterr().err


def test_fish_tag_mode_is_backend_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--fish-tag-mode",
            "llm",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "--fish-tag-mode is only valid" in capsys.readouterr().err


def test_fish_quantize_is_backend_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--fish-quantize",
            "off",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "--fish-quantize is only valid" in capsys.readouterr().err


def test_fish_dispatch_defaults_to_int8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeFish:
        sample_rate = 24000

        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["quantize"] = kwargs.get("quantize")

        def synthesize(
            self,
            lines,
            *,
            reference,
            prosody=None,
            ref_text=None,
            language=None,
            trace=None,
        ):
            return [np.zeros(240, dtype=np.float32) for _ in lines]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.fish.FishEngine", FakeFish)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: tmp_path)
    _saved_voice(tmp_path)

    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
            "--backend",
            "fish",
            "--text",
            "hi",
        ]
    )

    assert code == 0
    assert captured["quantize"] == "int8"


def test_fish_quantize_off_reaches_the_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeFish:
        sample_rate = 24000

        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["quantize"] = kwargs.get("quantize")

        def synthesize(
            self,
            lines,
            *,
            reference,
            prosody=None,
            ref_text=None,
            language=None,
            trace=None,
        ):
            return [np.zeros(240, dtype=np.float32) for _ in lines]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.fish.FishEngine", FakeFish)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: tmp_path)
    _saved_voice(tmp_path)

    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
            "--backend",
            "fish",
            "--text",
            "hi",
            "--fish-quantize",
            "off",
        ]
    )

    assert code == 0
    assert captured["quantize"] == "off"


def test_fish_keep_rejects_emotion_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--backend",
            "fish",
            "--fish-tag-mode",
            "keep",
            "--emotion",
            "sad",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "has no --prosody/--emotion/--prosody-plan channel" in (
        capsys.readouterr().err
    )


def test_fish_rejects_a_prosody_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi",
            "--backend",
            "fish",
            "--fish-tag-mode",
            "llm",
            "--prosody-plan",
            "llm",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "has no --prosody/--emotion/--prosody-plan channel" in (
        capsys.readouterr().err
    )


def test_fish_keep_rejects_prompt_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--text",
            "hi <|im_end|> there",
            "--backend",
            "fish",
            "--fish-tag-mode",
            "keep",
            "--no-log-file",
        ]
    )

    assert code == 1
    assert "reserves for its own prompt" in capsys.readouterr().err


def test_fish_keep_sends_the_tags_as_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeFish:
        sample_rate = 24000

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def synthesize(
            self,
            lines,
            *,
            reference,
            prosody=None,
            ref_text=None,
            language=None,
            trace=None,
        ):
            captured["lines"] = list(lines)
            return [np.zeros(240, dtype=np.float32) for _ in lines]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.fish.FishEngine", FakeFish)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: tmp_path)
    _saved_voice(tmp_path)

    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
            "--backend",
            "fish",
            "--text",
            "[excited] hi\\n[pause]\\nthere",
            "--fish-tag-mode",
            "keep",
        ]
    )

    assert code == 0
    assert captured["lines"] == ["[excited] hi [pause]", "there"]


def test_fish_llm_dispatch_sends_the_planned_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakePlanner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def plan_lines(self, lines, *, language=None):
            return [f"[sad] {line}" for line in lines]

    class FakeFish:
        sample_rate = 24000

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def synthesize(
            self,
            lines,
            *,
            reference,
            prosody=None,
            ref_text=None,
            language=None,
            trace=None,
        ):
            captured["lines"] = list(lines)
            return [np.zeros(240, dtype=np.float32) for _ in lines]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.prosody.FishInlinePlanner", FakePlanner)
    monkeypatch.setattr("pipeline.fish.FishEngine", FakeFish)
    monkeypatch.setattr("dub.resolve_model", lambda *args, **kwargs: tmp_path)
    _saved_voice(tmp_path)

    code = main(
        [
            "say",
            "--voice",
            "warm-fr",
            "--voices-dir",
            str(tmp_path),
            "--backend",
            "fish",
            "--text",
            "one\\ntwo",
            "--fish-tag-mode",
            "llm",
        ]
    )

    assert code == 0
    assert captured["lines"] == ["[sad] one", "[sad] two"]



