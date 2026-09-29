"""Tests for prompt construction, response parsing and the OpenAI-compatible client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from pipeline.errors import TranslationError
from pipeline.translate import (
    OpenAICompatTranslator,
    ShortenRequest,
    TranslationRequest,
    build_messages,
    build_shorten_messages,
    chunked,
    extract_json_object,
)


def response(content: str) -> SimpleNamespace:
    """Mimic an OpenAI chat completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeCompletions:
    """Scripted completions endpoint."""

    def __init__(self, contents: list[Any]) -> None:
        self._contents = list(contents)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._contents.pop(0) if self._contents else ""
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return response(item)
        return item


class FakeClient:
    def __init__(self, contents: list[Any]) -> None:
        self.completions = FakeCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def make_translator(contents: list[Any], **kwargs: Any) -> OpenAICompatTranslator:
    return OpenAICompatTranslator(
        base_url="http://localhost:8080/v1",
        api_key="x",
        model="local",
        client=FakeClient(contents),
        **kwargs,
    )


def requests_payload(count: int) -> list[TranslationRequest]:
    return [TranslationRequest(id=i, text=f"line {i}", duration=1.0) for i in range(count)]


class TestExtractJsonObject:
    def test_bare_object(self) -> None:
        assert extract_json_object('{"translations": []}') == {"translations": []}

    def test_fenced_block(self) -> None:
        raw = '```json\n{"translations": [{"id": 1, "text": "x"}]}\n```'
        assert extract_json_object(raw) == {"translations": [{"id": 1, "text": "x"}]}

    def test_object_embedded_in_prose(self) -> None:
        raw = 'Sure! Here you go:\n{"translations": []}\nHope that helps.'
        assert extract_json_object(raw) == {"translations": []}

    def test_fenced_without_language_tag(self) -> None:
        assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}

    @pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1, 2, 3]", "{}"])
    def test_unusable_payloads(self, raw: str) -> None:
        result = extract_json_object(raw)
        assert result is None or isinstance(result, dict)

    def test_top_level_array_is_rejected(self) -> None:
        assert extract_json_object('[{"id": 1}]') is None

    def test_malformed_object_returns_none(self) -> None:
        assert extract_json_object('{"translations": [}') is None


class TestChunked:
    def test_splits_evenly(self) -> None:
        assert [list(c) for c in chunked(list(range(5)), 2)] == [[0, 1], [2, 3], [4]]

    def test_single_chunk_when_size_exceeds_length(self) -> None:
        assert [list(c) for c in chunked([1, 2], 10)] == [[1, 2]]

    def test_empty_input_yields_nothing(self) -> None:
        assert list(chunked([], 5)) == []

    def test_rejects_non_positive_size(self) -> None:
        with pytest.raises(ValueError):
            list(chunked([1], 0))


class TestBuildMessages:
    def test_includes_target_language_and_lines(self) -> None:
        messages = build_messages(
            requests_payload(2), source_language="English", target_language="French"
        )
        assert messages[0]["role"] == "system"
        assert "French" in messages[0]["content"]
        assert "English" in messages[0]["content"]
        assert '"id": 0' in messages[1]["content"]

    def test_context_is_included_when_present(self) -> None:
        messages = build_messages(
            requests_payload(1),
            source_language="English",
            target_language="French",
            context=["earlier line"],
        )
        assert "earlier line" in messages[1]["content"]

    def test_prompts_auto_detection_without_source(self) -> None:
        messages = build_messages(
            requests_payload(1), source_language=None, target_language="French"
        )
        assert "Detect the source language" in messages[0]["content"]

    def test_length_tolerance_is_stated(self) -> None:
        messages = build_messages(
            requests_payload(1), source_language=None, target_language="French"
        )
        assert "20" in messages[0]["content"]


class TestTranslatorHappyPath:
    def test_returns_id_to_text_mapping(self) -> None:
        payload = json.dumps(
            {"translations": [{"id": 0, "text": "zéro"}, {"id": 1, "text": "un"}]}
        )
        translator = make_translator([payload])
        result = translator.translate(
            requests_payload(2), source_language="en", target_language="French"
        )
        assert result == {0: "zéro", 1: "un"}

    def test_empty_requests_skip_the_api(self) -> None:
        translator = make_translator([])
        assert translator.translate([], source_language="en", target_language="French") == {}
        assert translator.client.completions.calls == []

    def test_uses_language_name_in_prompt(self) -> None:
        payload = json.dumps({"translations": [{"id": 0, "text": "x"}]})
        translator = make_translator([payload])
        translator.translate(requests_payload(1), source_language="en", target_language="French")
        system = translator.client.completions.calls[0]["messages"][0]["content"]
        assert "English" in system

    def test_batches_large_requests(self) -> None:
        translator = make_translator(
            [
                json.dumps({"translations": [{"id": i, "text": "x"} for i in range(2)]}),
                json.dumps({"translations": [{"id": i, "text": "x"} for i in range(2, 4)]}),
            ],
            chunk_size=2,
        )
        result = translator.translate(
            requests_payload(4), source_language="en", target_language="French"
        )
        assert set(result) == {0, 1, 2, 3}
        assert len(translator.client.completions.calls) == 2

    def test_extra_ids_are_discarded(self) -> None:
        payload = json.dumps(
            {"translations": [{"id": 0, "text": "ok"}, {"id": 99, "text": "stray"}]}
        )
        translator = make_translator([payload])
        result = translator.translate(
            [TranslationRequest(id=0, text="a")], source_language="en", target_language="French"
        )
        assert result == {0: "ok"}

    def test_bare_mapping_payload_is_tolerated(self) -> None:
        translator = make_translator([json.dumps({"0": "zéro"})])
        result = translator.translate(
            [TranslationRequest(id=0, text="a")], source_language="en", target_language="French"
        )
        assert result == {0: "zéro"}

    def test_string_ids_are_coerced(self) -> None:
        payload = json.dumps({"translations": [{"id": "3", "text": "trois"}]})
        translator = make_translator([payload])
        result = translator.translate(
            [TranslationRequest(id=3, text="a")], source_language="en", target_language="French"
        )
        assert result == {3: "trois"}


class TestTranslatorRepair:
    def test_missing_id_triggers_one_repair(self) -> None:
        first = json.dumps({"translations": [{"id": 0, "text": "zéro"}]})
        second = json.dumps(
            {"translations": [{"id": 0, "text": "zéro"}, {"id": 1, "text": "un"}]}
        )
        translator = make_translator([first, second])
        result = translator.translate(
            requests_payload(2), source_language="en", target_language="French"
        )
        assert result == {0: "zéro", 1: "un"}
        assert len(translator.client.completions.calls) == 2
        # The repair call is deterministic.
        assert translator.client.completions.calls[1]["temperature"] == 0.0

    def test_repair_failure_raises(self) -> None:
        incomplete = json.dumps({"translations": [{"id": 0, "text": "zéro"}]})
        translator = make_translator([incomplete, incomplete])
        with pytest.raises(TranslationError, match="missing ids"):
            translator.translate(
                requests_payload(2), source_language="en", target_language="French"
            )

    def test_non_json_response_raises(self) -> None:
        translator = make_translator(["I cannot help with that."])
        with pytest.raises(TranslationError, match="not valid JSON"):
            translator.translate(
                requests_payload(1), source_language="en", target_language="French"
            )


class TestTranslatorTransport:
    def test_api_error_becomes_domain_error(self) -> None:
        translator = make_translator([RuntimeError("connection refused"), RuntimeError("again")])
        with pytest.raises(TranslationError, match="failed"):
            translator.translate(
                requests_payload(1), source_language="en", target_language="French"
            )

    def test_unsupported_response_format_is_retried(self) -> None:
        payload = json.dumps({"translations": [{"id": 0, "text": "x"}]})
        translator = make_translator([RuntimeError("response_format unsupported"), payload])
        result = translator.translate(
            requests_payload(1), source_language="en", target_language="French"
        )
        assert result == {0: "x"}

    def test_asks_for_json_object_format_first(self) -> None:
        payload = json.dumps({"translations": [{"id": 0, "text": "x"}]})
        translator = make_translator([payload])
        translator.translate(requests_payload(1), source_language="en", target_language="French")
        assert translator.client.completions.calls[0]["response_format"] == {"type": "json_object"}

    def test_content_parts_are_joined(self) -> None:
        payload = [{"text": '{"translations": [{"id": 0, "text": '}, {"text": '"x"}]}'}]
        translator = make_translator([response(payload)])  # type: ignore[arg-type]
        result = translator.translate(
            requests_payload(1), source_language="en", target_language="French"
        )
        assert result == {0: "x"}

    def test_unexpected_response_shape_raises(self) -> None:
        translator = make_translator([SimpleNamespace(nonsense=True)])
        with pytest.raises(TranslationError):
            translator.translate(
                requests_payload(1), source_language="en", target_language="French"
            )


class TestShortenMessages:
    """The rewrite prompt carries everything needed to shorten without losing meaning."""

    @staticmethod
    def request(**overrides: Any) -> ShortenRequest:
        payload: dict[str, Any] = {
            "id": 3,
            "text": "une phrase vraiment tres longue",
            "source": "a very long sentence",
            "spoken_seconds": 4.42,
            "available_seconds": 3.48,
            "max_words": 9,
        }
        payload.update(overrides)
        return ShortenRequest(**payload)

    def test_states_the_ceiling_and_the_pressures(self) -> None:
        messages = build_shorten_messages([self.request()], target_language="French")
        user = messages[1]["content"]
        assert '"id": 3' in user
        assert '"max_words": 9' in user
        assert '"spoken_seconds": 4.42' in user
        assert '"available_seconds": 3.48' in user

    def test_carries_the_previous_wording_and_the_source(self) -> None:
        messages = build_shorten_messages([self.request()], target_language="French")
        user = messages[1]["content"]
        assert "une phrase vraiment tres longue" in user
        assert "a very long sentence" in user

    def test_forbids_dropping_facts_and_names_the_target(self) -> None:
        system = build_shorten_messages([self.request()], target_language="French")[0]["content"]
        assert "French" in system
        assert "number" in system and "name" in system

    def test_every_line_becomes_one_json_object(self) -> None:
        messages = build_shorten_messages(
            [self.request(id=0), self.request(id=1)], target_language="German"
        )
        assert messages[1]["content"].count('"id":') == 2


class TestWordBudget:
    """A budget changes the prompt only when one is supplied."""

    def test_absent_when_no_budget_is_known(self) -> None:
        messages = build_messages(
            [TranslationRequest(id=0, text="a", duration=1.0)],
            source_language="en",
            target_language="French",
        )
        assert "max_words" not in messages[0]["content"]
        assert "max_words" not in messages[1]["content"]

    def test_present_when_a_budget_is_known(self) -> None:
        messages = build_messages(
            [TranslationRequest(id=0, text="a", duration=2.0, max_words=5)],
            source_language="en",
            target_language="French",
        )
        assert '"max_words": 5' in messages[1]["content"]
        assert "max_words" in messages[0]["content"]

    def test_the_duration_hint_is_kept(self) -> None:
        messages = build_messages(
            [TranslationRequest(id=0, text="a", duration=2.0, max_words=5)],
            source_language="en",
            target_language="French",
        )
        assert '"seconds": 2.0' in messages[1]["content"]

    def test_a_budget_replaces_the_proportional_hint(self) -> None:
        """The two length rules disagree, so only one of them may be sent."""
        unbudgeted = build_messages(
            [TranslationRequest(id=0, text="a", duration=2.0)],
            source_language="en",
            target_language="French",
        )
        budgeted = build_messages(
            [TranslationRequest(id=0, text="a", duration=2.0, max_words=5)],
            source_language="en",
            target_language="French",
        )
        assert "within roughly" in unbudgeted[0]["content"]
        assert "within roughly" not in budgeted[0]["content"]
        assert "hard ceiling" in budgeted[0]["content"]


class TestShortenCalls:
    """The shorten backend reuses the same transport and repair path."""

    def test_rewrites_are_returned_by_id(self) -> None:
        payload = json.dumps({"translations": [{"id": 0, "text": "Bref."}]})
        translator = make_translator([payload])
        result = translator.shorten(
            [ShortenRequest(id=0, text="long", max_words=2)], target_language="French"
        )
        assert result == {0: "Bref."}

    def test_empty_requests_skip_the_api(self) -> None:
        translator = make_translator([])
        assert translator.shorten([], target_language="French") == {}
        assert translator.client.completions.calls == []

    def test_missing_id_is_repaired_once(self) -> None:
        translator = make_translator(
            [
                json.dumps({"translations": [{"id": 0, "text": "a"}]}),
                json.dumps({"translations": [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}]}),
            ]
        )
        result = translator.shorten(
            [ShortenRequest(id=0, text="x", max_words=1), ShortenRequest(id=1, text="y", max_words=1)],
            target_language="French",
        )
        assert result == {0: "a", 1: "b"}
        assert len(translator.client.completions.calls) == 2

    def test_transport_failure_raises_a_domain_error(self) -> None:
        # Two failures: the client first retries without `response_format`, so a
        # single queued error would be masked by an empty second response.
        translator = make_translator(
            [RuntimeError("connection refused"), RuntimeError("connection refused")]
        )
        with pytest.raises(TranslationError, match="failed"):
            translator.shorten(
                [ShortenRequest(id=0, text="x", max_words=1)], target_language="French"
            )
