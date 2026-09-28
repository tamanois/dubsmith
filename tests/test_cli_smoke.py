"""CLI-surface smoke tests: no GPU, no models, no ffmpeg.

The unit suites cover the pipeline internals; these cover the command surface a
user actually types and the environment report they are told to paste into a bug
report. That matters because the CLI passes *every* flag explicitly - including
its defaults - so a flag can silently stop reaching the job config without any
pipeline test noticing.
"""

from __future__ import annotations

import pytest

from dub import build_dub_parser, job_config_from_args, main, render_environment
from pipeline import __version__
from pipeline.config import JobConfig
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
