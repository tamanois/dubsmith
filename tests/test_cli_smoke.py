"""CLI-surface smoke tests: no GPU, no models, no ffmpeg.

The unit suites cover the pipeline internals; these cover the command surface a
user actually types and the environment report they are told to paste into a bug
report. That matters because the CLI passes *every* flag explicitly - including
its defaults - so a flag can silently stop reaching the job config without any
pipeline test noticing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dub import (
    build_dub_parser,
    job_config_from_args,
    log_file_from,
    main,
    render_environment,
)
from pipeline import __version__
from pipeline.config import JobConfig, Settings
from pipeline.errors import ConfigError
from pipeline.models import REGISTRY


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


def test_dtype_and_device_default_to_auto() -> None:
    args = build_dub_parser().parse_args(["--video", "in.mp4", "--target", "fr"])
    job = job_config_from_args(args)

    assert job.dtype == "auto"
    assert job.device == "auto"


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


def test_an_unknown_dtype_is_rejected_in_the_config() -> None:
    with pytest.raises(ConfigError, match="Unknown dtype"):
        JobConfig(audio="a.wav", dtype="float8")


def test_dtype_is_part_of_the_fingerprint() -> None:
    # A different precision is a different synthesis, so the cache must not be
    # reused across a dtype change - silently serving fp32 clips for an fp16 run
    # (or the reverse) would be invisible in the output.
    base = JobConfig(audio="a.wav")

    assert base.fingerprint() != JobConfig(audio="a.wav", dtype="fp32").fingerprint()
