"""The capability table and the Say validator must agree.

If a future backend is added or a rule changes, this test fails rather than
letting the web form offer an option the pipeline will reject.
"""

from __future__ import annotations

import pytest

from pipeline.capabilities import (
    DUB_BACKENDS,
    SAY_BACKENDS,
    capabilities_payload,
    say_prosody_conflict,
)
from pipeline.errors import ConfigError
from pipeline.say import SaySpec


def _spec(backend: str, **overrides: object) -> SaySpec:
    caps = SAY_BACKENDS[backend]
    base: dict[str, object] = {"text": "hi", "backend": backend}
    if caps.speaker:
        base["speaker"] = "Vivian"
    else:
        base["voice"] = "warm-fr"
    base.update(overrides)
    return SaySpec(**base)


def test_payload_lists_both_groups() -> None:
    payload = capabilities_payload()

    assert [item["id"] for item in payload["say"]] == list(SAY_BACKENDS)
    assert [item["id"] for item in payload["dub"]] == list(DUB_BACKENDS)
    # Every gap has an explanation, so the UI never shows a mute disabled field.
    for item in payload["say"] + payload["dub"]:
        for reason in item["unsupported"].values():
            assert reason


@pytest.mark.parametrize("backend", list(SAY_BACKENDS))
def test_prosody_capability_matches_the_validator(backend: str) -> None:
    caps = SAY_BACKENDS[backend]
    conflict = say_prosody_conflict(backend, prosody=["emotion=happy"])

    if caps.prosody:
        assert conflict is None
        _spec(backend, prosody=["emotion=happy"])  # must not raise
    else:
        assert conflict
        with pytest.raises(ConfigError):
            _spec(backend, prosody=["emotion=happy"])


@pytest.mark.parametrize("backend", list(SAY_BACKENDS))
def test_an_off_prosody_plan_is_always_allowed(backend: str) -> None:
    assert say_prosody_conflict(backend, prosody_plan="off") is None
    _spec(backend)  # the default spec is valid for every backend
