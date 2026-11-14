"""The option vocabularies are the one place each value is spelled.

A value renamed in ``pipeline/options.py`` but not in the CLI, the models or the
capability table is the failure mode this module guards. Every test compares the
declarations with each other, never with a copy of the literals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pytest

import dub
from pipeline import options
from pipeline.capabilities import DUB_BACKENDS, SAY_BACKENDS
from pipeline.config import (
    JobConfig,
    SubtitleJobConfig,
    normalize_dtype,
    normalize_loudness_policy,
    normalize_subtitles_mode,
)
from pipeline.errors import ConfigError
from pipeline.options import Defaults
from server.schemas import SayRequest

#: Every alias table paired with the values it is allowed to produce.
ALIAS_TABLES = [
    (options.DTYPE_ALIASES, options.DTYPES),
    (options.STRETCHER_ALIASES, options.STRETCHERS),
    (options.SEPARATOR_ALIASES, options.SEPARATORS),
    (options.OVERLAP_SEPARATOR_ALIASES, options.OVERLAP_SEPARATORS),
    (options.PROSODY_PLAN_ALIASES, options.PROSODY_PLANS),
    (options.FISH_QUANTIZE_ALIASES, options.FISH_QUANTIZES),
    (options.FISH_TAG_MODE_ALIASES, options.FISH_TAG_MODES),
    (options.ACCENT_MODE_ALIASES, options.ACCENT_MODES),
    (options.ACCENT_MATCH_ALIASES, options.ACCENT_MATCHES),
    (options.ACCENT_BLEND_ALIASES, options.ACCENT_BLENDS),
    (options.EXPRESSIVE_ACCENT_ALIASES, options.EXPRESSIVE_ACCENTS),
    (options.LOUDNESS_POLICY_ALIASES, options.LOUDNESS_POLICIES),
    (options.ROOM_ALIASES, options.ROOMS),
    (options.LENGTH_BUDGET_ALIASES, options.LENGTH_BUDGETS),
    (options.DIARIZER_ALIASES, options.DIARIZERS),
    (options.OVERLAP_MODE_ALIASES, options.OVERLAP_MODES),
    (options.SUBTITLES_MODE_ALIASES, options.SUBTITLES_MODES),
    (options.TTS_BACKEND_ALIASES, options.TTS_BACKENDS),
]

#: Pairs of (JobConfig field, accepted values) that need no extra constraints.
PLAIN_FIELDS = [
    ("tts_backend", options.TTS_BACKENDS),
    ("prosody_plan", options.PROSODY_PLANS),
    ("length_budget", options.LENGTH_BUDGETS),
    ("separator", options.SEPARATORS),
    ("room", options.ROOMS),
    ("accent_match", options.ACCENT_MATCHES),
    ("diarizer", options.DIARIZERS),
    ("overlap_mode", options.OVERLAP_MODES),
    ("overlap_separator", options.OVERLAP_SEPARATORS),
    ("subtitles_mode", options.SUBTITLES_MODES),
    ("dtype", options.DTYPES),
]

#: Defaults paired with the values they must belong to.
DEFAULTS = [
    (Defaults.DTYPE, options.DTYPES),
    (Defaults.STRETCHER, options.STRETCHERS),
    (Defaults.SEPARATOR, options.SEPARATORS),
    (Defaults.PROSODY_PLAN, options.PROSODY_PLANS),
    (Defaults.SAY_PROSODY_PLAN, options.SAY_PROSODY_PLANS),
    (Defaults.STYLE_PLAN, options.STYLE_PLANS),
    (Defaults.STYLE_PLAN_SCOPE, options.STYLE_PLAN_SCOPES),
    (Defaults.FISH_QUANTIZE, options.FISH_QUANTIZES),
    (Defaults.FISH_TAG_MODE, options.FISH_TAG_MODES),
    (Defaults.ACCENT_MODE, options.ACCENT_MODES),
    (Defaults.ACCENT_MATCH, options.ACCENT_MATCHES),
    (Defaults.ACCENT_BLEND, options.ACCENT_BLENDS),
    (Defaults.EXPRESSIVE_ACCENT, options.EXPRESSIVE_ACCENTS),
    (Defaults.LOUDNESS_POLICY, options.LOUDNESS_POLICIES),
    (Defaults.ROOM, options.ROOMS),
    (Defaults.LENGTH_BUDGET, options.LENGTH_BUDGETS),
    (Defaults.DIARIZER, options.DIARIZERS),
    (Defaults.OVERLAP_MODE, options.OVERLAP_MODES),
    (Defaults.OVERLAP_SEPARATOR, options.OVERLAP_SEPARATORS),
    (Defaults.SUBTITLES_MODE, options.SUBTITLES_MODES),
    (Defaults.TTS_BACKEND, options.TTS_BACKENDS),
    (Defaults.SAY_BACKEND, options.SAY_BACKENDS),
    (Defaults.LOG_LEVEL, options.LOG_LEVELS),
]


def _action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    """The parser action for ``dest``, so the CLI contract can be read back."""
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise AssertionError(f"{dest!r} is not an option of {parser.prog}")


@pytest.mark.parametrize(("aliases", "values"), ALIAS_TABLES)
def test_alias_tables_only_produce_declared_values(
    aliases: dict[str, str], values: tuple[str, ...]
) -> None:
    assert set(aliases.values()) <= set(values)
    # A canonical spelling must map to itself, or a user passing the documented
    # value would be rejected while an alias still worked.
    for value in values:
        assert aliases[value] == value


@pytest.mark.parametrize(("default", "values"), DEFAULTS)
def test_defaults_are_declared_values(default: str, values: tuple[str, ...]) -> None:
    assert default in values


def test_capability_ids_match_the_vocabulary() -> None:
    assert tuple(SAY_BACKENDS) == options.SAY_BACKENDS
    assert tuple(DUB_BACKENDS) == options.TTS_BACKENDS
    assert set(options.EXPRESSIVE_BACKENDS) <= set(options.TTS_BACKENDS)


def test_say_parser_choices_come_from_the_vocabulary() -> None:
    parser = dub.build_say_parser()

    assert tuple(_action(parser, "backend").choices) == options.SAY_BACKENDS
    assert _action(parser, "backend").default == Defaults.SAY_BACKEND
    assert tuple(_action(parser, "prosody_plan").choices) == options.SAY_PROSODY_PLANS
    assert tuple(_action(parser, "style_plan").choices) == options.STYLE_PLANS
    assert tuple(_action(parser, "style_plan_scope").choices) == (
        options.STYLE_PLAN_SCOPES
    )
    assert tuple(_action(parser, "fish_quantize").choices) == options.FISH_QUANTIZES
    assert tuple(_action(parser, "fish_tag_mode").choices) == options.FISH_TAG_MODES
    assert tuple(_action(parser, "dtype").choices) == options.DTYPES


def test_dub_parser_choices_come_from_the_vocabulary() -> None:
    parser = dub.build_dub_parser()

    assert tuple(_action(parser, "tts_backend").choices) == options.TTS_BACKENDS
    assert _action(parser, "tts_backend").default == Defaults.TTS_BACKEND
    assert tuple(_action(parser, "accent_mode").choices) == options.ACCENT_MODES
    assert tuple(_action(parser, "accent_match").choices) == options.ACCENT_MATCHES
    assert tuple(_action(parser, "accent_blend").choices) == options.ACCENT_BLENDS
    assert tuple(_action(parser, "expressive_accent").choices) == (
        options.EXPRESSIVE_ACCENTS
    )
    assert tuple(_action(parser, "prosody_plan").choices) == options.PROSODY_PLANS
    assert tuple(_action(parser, "stretcher").choices) == options.STRETCHERS
    assert tuple(_action(parser, "loudness_target").choices) == (
        options.LOUDNESS_POLICIES
    )
    assert tuple(_action(parser, "separator").choices) == options.SEPARATORS
    assert tuple(_action(parser, "room").choices) == options.ROOMS
    assert tuple(_action(parser, "diarizer").choices) == options.DIARIZERS
    assert tuple(_action(parser, "overlap_mode").choices) == options.OVERLAP_MODES
    assert tuple(_action(parser, "overlap_separator").choices) == (
        options.OVERLAP_SEPARATORS
    )
    assert tuple(_action(parser, "length_budget").choices) == options.LENGTH_BUDGETS
    assert tuple(_action(parser, "subtitles_mode").choices) == options.SUBTITLES_MODES
    assert tuple(_action(parser, "fish_quantize").choices) == options.FISH_QUANTIZES
    assert tuple(_action(parser, "dtype").choices) == options.DTYPES


def test_subtitles_parser_choices_come_from_the_vocabulary() -> None:
    parser = dub.build_subtitles_parser()

    assert tuple(_action(parser, "subtitles_mode").choices) == options.SUBTITLES_MODES


@pytest.mark.parametrize(
    "builder",
    [dub.build_anchors_parser, dub.build_voices_parser, dub.build_info_parser],
)
def test_device_parsers_share_the_dtype_vocabulary(
    builder: Callable[[], argparse.ArgumentParser],
) -> None:
    parser = builder()

    assert tuple(_action(parser, "dtype").choices) == options.DTYPES


@pytest.mark.parametrize(("field", "values"), PLAIN_FIELDS)
def test_job_config_accepts_every_declared_value(
    field: str, values: tuple[str, ...]
) -> None:
    for value in values:
        job = JobConfig(audio=Path("a.wav"), **{field: value})
        assert getattr(job, field) == value


@pytest.mark.parametrize("mode", options.ACCENT_MODES)
def test_job_config_accepts_every_accent_mode(mode: str) -> None:
    extra: dict[str, object] = {}
    if mode == options.AccentMode.REFERENCE_ACCENT.value:
        extra = {"accent_audio": Path("accent.wav"), "accent_text": "bonjour"}
    job = JobConfig(audio=Path("a.wav"), accent_mode=mode, **extra)

    assert job.accent_mode == mode


@pytest.mark.parametrize("mode", options.EXPRESSIVE_ACCENTS)
def test_expressive_accent_values_need_an_expressive_backend(mode: str) -> None:
    job = JobConfig(
        audio=Path("a.wav"),
        tts_backend=options.TtsBackend.COSYVOICE3,
        expressive_accent=mode,
    )

    assert job.expressive_accent == mode


@pytest.mark.parametrize("value", options.SUBTITLES_MODES)
def test_subtitle_config_accepts_every_mode(value: str) -> None:
    job = SubtitleJobConfig(audio=Path("a.wav"), subtitles_mode=value)

    assert job.subtitles_mode == value


def test_normalizers_accept_every_declared_value() -> None:
    for value in options.LOUDNESS_POLICIES:
        assert normalize_loudness_policy(value) == value
    for value in options.SUBTITLES_MODES:
        assert normalize_subtitles_mode(value) == value
    for value in options.DTYPES:
        assert normalize_dtype(value) == value


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("accent_mode", "xvector", "speaker-only"),
        ("tts_backend", "cosyvoice", "cosyvoice3"),
        ("length_budget", "seconds", "off"),
        ("overlap_mode", "split", "separate"),
        ("subtitles_mode", "hardcode", "burn"),
        ("loudness_target", "fixed", "lufs"),
    ],
)
def test_documented_aliases_still_resolve(
    field: str, value: str, expected: str
) -> None:
    job = JobConfig(audio=Path("a.wav"), **{field: value})

    assert getattr(job, field) == expected


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("accent_mode", "accent mode"),
        ("tts_backend", "TTS backend"),
        ("prosody_plan", "prosody plan"),
        ("room", "room mode"),
        ("overlap_separator", "overlap separator"),
    ],
)
def test_an_unknown_value_names_the_option(field: str, label: str) -> None:
    with pytest.raises(ConfigError, match=f"Unknown {label}"):
        JobConfig(audio=Path("a.wav"), **{field: "not-a-value"})


def test_say_schema_uses_the_say_backend_vocabulary() -> None:
    for backend in options.SAY_BACKENDS:
        request = SayRequest(text="hi", backend=backend)
        assert request.backend == backend
    # Serialised to JSON, an enum must stay the plain string the CLI and the
    # stored job payload use.
    assert '"backend":"fish"' in SayRequest(text="hi", backend="fish").model_dump_json()
