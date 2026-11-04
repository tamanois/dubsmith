"""Tests for the provenance artifact and its recorder."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.provenance import (
    LineProvenance,
    NullRecorder,
    ProvenanceRecorder,
    describe_transformations,
    timed_provenance_path,
)
from pipeline.prosody import ProsodySpec


def test_timed_path_is_unique_within_a_second(tmp_path: Path) -> None:
    first = timed_provenance_path(tmp_path)
    first.write_text("{}", encoding="utf-8")
    second = timed_provenance_path(tmp_path)
    assert first != second
    assert first.parent == second.parent


def test_recorder_writes_the_schema(tmp_path: Path) -> None:
    recorder = ProvenanceRecorder("dub", "fish", language="French", plan_source="llm")
    recorder.record_planner_config(
        model="m",
        base_url="http://x",
        temperature=0.3,
        chunk_size=10,
        backend="fish",
    )
    recorder.record_overview("arc")
    recorder.record_call("chunk 1/1", "reply", fallback=True)
    recorder.record_line(
        LineProvenance(
            key=0,
            speaker="A",
            text_planned="a",
            text_spoken="[sad] a",
            spec=ProsodySpec().model_dump(),
            rendered={"channel": "fish", "text": "[sad] a"},
        )
    )

    payload = recorder.build().to_dict()

    assert payload["kind"] == "dub"
    assert payload["backend"] == "fish"
    assert payload["plan_source"] == "llm"
    assert payload["planner"]["config"]["model"] == "m"
    assert payload["planner"]["overview"] == "arc"
    assert payload["planner"]["calls"][0]["fallback"] is True
    assert payload["lines"][0]["text_spoken"] == "[sad] a"

    path = recorder.write(tmp_path / "nested" / "provenance.json")
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_null_recorder_writes_nothing(tmp_path: Path) -> None:
    recorder = NullRecorder()
    assert recorder.enabled() is False
    assert recorder.write(tmp_path / "provenance.json") is None
    assert not (tmp_path / "provenance.json").exists()


def test_recorder_replaces_a_retried_line() -> None:
    recorder = ProvenanceRecorder("dub", "fish")
    recorder.record_line(LineProvenance(key=0, text_spoken="first"))
    recorder.record_line(LineProvenance(key=0, text_spoken="second"))
    assert [line.text_spoken for line in recorder.build().lines] == ["second"]


def test_describe_transformations_fish() -> None:
    items = describe_transformations("hello", "[sad] hello", channel="fish")
    assert [item.kind for item in items] == ["fish_tag"]
    assert items[0].detail == "sad"


def test_describe_transformations_cosyvoice() -> None:
    items = describe_transformations(
        "hello",
        "hello [breath]",
        channel="cosyvoice",
        spec=ProsodySpec(breath=True).model_dump(),
        instruct="Speak softly.",
    )
    kinds = [item.kind for item in items]
    assert "inline_token" in kinds
    assert "instruct" in kinds


def test_artifact_never_contains_a_client_key() -> None:
    recorder = ProvenanceRecorder("say", "chatterbox")
    recorder.record_planner_config(
        model="m", base_url="http://x", temperature=0.2, chunk_size=10
    )
    recorder.record_call("chunk 1/1", "reply")
    blob = json.dumps(recorder.build().to_dict())
    assert "api_key" not in blob
