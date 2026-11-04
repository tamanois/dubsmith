"""Tests for the backend-neutral prosody model and its renderers."""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

import pytest

from pipeline.errors import ConfigError, TranslationError
from pipeline.prosody import (
    BACKEND_CAPABILITIES,
    FISH_MAX_CUES_PER_LINE,
    FISH_TAGS,
    FishInlinePlanner,
    ProsodyPlanner,
    ProsodySpec,
    StylePlanner,
    apply_inline_tokens,
    build_fish_inline_messages,
    build_prosody_messages,
    build_style_messages,
    chatterbox_exaggeration,
    cosyvoice_instruct,
    cosyvoice_mode,
    customvoice_instruct,
    fish_cue_tokens,
    fish_structural_token,
    parse_fish_inline_response,
    parse_prosody_response,
    parse_prosody_spec,
    parse_style_response,
    resolve_prosody,
    sanitize_fish_inline_line,
    split_fish_line,
    strip_fish_tags,
    trained_emotion,
    validate_fish_inline_text,
)


def test_parse_defaults_to_a_plain_delivery() -> None:
    spec = parse_prosody_spec("")
    assert spec.is_empty()
    assert spec.pace == "normal"
    assert spec.volume == "normal"


def test_parse_comma_separated_pairs() -> None:
    spec = parse_prosody_spec("emotion=happy, pace=fast, volume=loud")
    assert spec.emotion == "happy"
    assert spec.pace == "fast"
    assert spec.volume == "loud"


def test_parse_space_separated_pairs_without_commas() -> None:
    spec = parse_prosody_spec("emotion=happy pace=slow breath")
    assert spec.emotion == "happy"
    assert spec.pace == "slow"
    assert spec.breath is True


def test_parse_treats_a_bare_word_as_the_emotion() -> None:
    assert parse_prosody_spec("excited").emotion == "excited"


def test_parse_accepts_bare_flags() -> None:
    spec = parse_prosody_spec("breath, pause, laughter")
    assert spec.breath and spec.pause and spec.laughter


def test_parse_accepts_field_aliases() -> None:
    spec = parse_prosody_spec("rate=quick, loudness=soft")
    assert spec.pace == "fast"
    assert spec.volume == "soft"


def test_parse_rejects_an_unknown_field() -> None:
    with pytest.raises(ConfigError, match="Unknown prosody field"):
        parse_prosody_spec("tempo=fast")


def test_parse_rejects_an_unknown_pace() -> None:
    with pytest.raises(ConfigError, match="Invalid prosody"):
        parse_prosody_spec("pace=warp")


def test_resolve_broadcasts_one_value() -> None:
    specs = resolve_prosody(["a", "b"], [ProsodySpec(emotion="happy")])
    assert [spec.emotion for spec in specs] == ["happy", "happy"]


def test_resolve_keeps_one_per_line() -> None:
    specs = resolve_prosody(
        ["a", "b"], [ProsodySpec(emotion="happy"), ProsodySpec(emotion="sad")]
    )
    assert [spec.emotion for spec in specs] == ["happy", "sad"]


def test_resolve_rejects_a_mismatch() -> None:
    with pytest.raises(ConfigError, match="3 prosody value"):
        resolve_prosody(
            ["a", "b"],
            [ProsodySpec(), ProsodySpec(), ProsodySpec()],
        )


def test_merge_keeps_values_the_override_did_not_ask_for() -> None:
    base = ProsodySpec(emotion="happy", pace="fast", breath=True)
    override = ProsodySpec(emotion="sad")
    merged = base.merged_with(override)
    assert merged.emotion == "sad"
    assert merged.pace == "fast"
    assert merged.breath is True


def test_cosyvoice_instruct_is_neutral_but_marked_when_empty() -> None:
    instruct = cosyvoice_instruct(ProsodySpec())
    assert instruct == "You are a helpful assistant.<|endofprompt|>"


def test_cosyvoice_instruct_uses_the_trained_sentences() -> None:
    instruct = cosyvoice_instruct(
        ProsodySpec(emotion="happy", pace="fast", volume="loud")
    )
    assert instruct.startswith("You are a helpful assistant.")
    assert "请非常开心地说一句话" in instruct
    assert "请用尽可能快地语速说一句话" in instruct
    assert "as loudly as possible" in instruct
    assert instruct.endswith("<|endofprompt|>")


def test_inline_breath_lands_at_sentence_boundaries() -> None:
    text = apply_inline_tokens(
        "First sentence. Second sentence.", ProsodySpec(breath=True)
    )
    assert text == "First sentence. [breath] Second sentence."


def test_inline_breath_prepends_when_there_is_no_boundary() -> None:
    assert apply_inline_tokens("just one clause", ProsodySpec(breath=True)).startswith(
        "[breath]"
    )


def test_inline_laughter_is_appended() -> None:
    assert apply_inline_tokens("that was funny", ProsodySpec(laughter=True)).endswith(
        "[laughter]"
    )


def test_inline_pause_is_not_added_when_punctuation_already_pauses() -> None:
    text = "Félicitations! Vous avez gagné. Vous êtes les champions"
    assert apply_inline_tokens(text, ProsodySpec(pause=True)) == text


def test_inline_pause_is_added_to_a_long_unpunctuated_line() -> None:
    text = "one two three four five six seven eight nine"
    out = apply_inline_tokens(text, ProsodySpec(pause=True))
    assert ", " in out and out != text


def test_exaggeration_rises_with_emotion_and_falls_when_sad() -> None:
    assert chatterbox_exaggeration(ProsodySpec(emotion="excited")) > 0.5
    assert chatterbox_exaggeration(ProsodySpec(emotion="sad")) < 0.5
    assert chatterbox_exaggeration(ProsodySpec()) == 0.5


def test_joy_maps_to_the_happy_intensity_on_chatterbox() -> None:
    assert chatterbox_exaggeration(ProsodySpec(emotion="joy")) == 0.8


def test_build_messages_includes_every_line() -> None:
    messages = build_prosody_messages(["one", "two"])
    assert messages[0]["role"] == "system"
    body = messages[1]["content"]
    assert '"id": 0' in body and '"id": 1' in body


def test_build_messages_describes_backend_capabilities() -> None:
    messages = build_prosody_messages(["one"], backend="chatterbox")
    system = messages[0]["content"]
    assert "chatterbox" in system
    assert "exaggeration" in system
    assert "ignored" in system


def test_build_messages_limits_cosyvoice3_to_the_trained_emotions() -> None:
    system = build_prosody_messages(["one"], backend="cosyvoice3")[0]["content"]
    assert "happy, sad or angry" in system


def test_parse_response_defaults_missing_and_invalid_entries() -> None:
    reply = json.dumps(
        {
            "prosody": [
                {"id": 0, "emotion": "happy", "pace": "fast"},
                {"id": 1, "emotion": "sad", "pace": "warp"},
            ]
        }
    )
    specs = parse_prosody_response(reply, 3)
    assert specs[0].emotion == "happy" and specs[0].pace == "fast"
    assert specs[1].emotion == "sad" and specs[1].pace == "normal"
    assert specs[2].is_empty()


def test_parse_response_tolerates_prose_and_fences() -> None:
    reply = 'Here you go:\n```json\n{"prosody":[{"id":0,"emotion":"calm"}]}\n```'
    assert parse_prosody_response(reply, 1)[0].emotion == "calm"


class _FakeClient:
    """Minimal OpenAI-compatible client returning a fixed prosody reply."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)

                class Message:
                    content = '{"prosody":[{"id":0,"emotion":"happy"},{"id":1,"emotion":"sad"}]}'

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_planner_returns_one_spec_per_line() -> None:
    client = _FakeClient()
    planner = ProsodyPlanner(
        base_url="http://x", api_key="k", model="m", client=client, chunk_size=40
    )
    specs = planner.plan(["first", "second"])
    assert [spec.emotion for spec in specs] == ["happy", "sad"]
    assert client.calls[0]["model"] == "m"


def test_planner_logs_start_and_completion(caplog) -> None:
    planner = ProsodyPlanner(
        base_url="http://x",
        api_key="k",
        model="m",
        backend="cosyvoice3",
        client=_FakeClient(),
    )

    with caplog.at_level(logging.INFO, logger="pipeline.prosody"):
        planner.plan(["first", "second"])

    messages = [record.getMessage() for record in caplog.records]
    assert any("Planning prosody for 2 line(s)" in message for message in messages)
    assert any("Prosody plan ready for 2 line(s)" in message for message in messages)
    assert any("in " in message and "s" in message for message in messages)


def test_cosyvoice3_mode_selects_the_method() -> None:
    assert cosyvoice_mode(ProsodySpec()) == "zero_shot"
    assert cosyvoice_mode(ProsodySpec(breath=True)) == "cross_lingual"
    assert cosyvoice_mode(ProsodySpec(emotion="happy")) == "instruct2"
    assert cosyvoice_mode(ProsodySpec(pace="fast")) == "instruct2"
    assert cosyvoice_mode(ProsodySpec(volume="soft")) == "instruct2"


def test_cosyvoice_maps_excited_onto_the_trained_happy_clause() -> None:
    assert "请非常开心地说一句话" in cosyvoice_instruct(ProsodySpec(emotion="excited"))


def test_joy_maps_onto_the_trained_happy_clause() -> None:
    assert trained_emotion("joy") == "happy"


def test_cosyvoice_calm_has_no_emotion_clause() -> None:
    assert cosyvoice_instruct(ProsodySpec(emotion="calm")) == (
        "You are a helpful assistant.<|endofprompt|>"
    )


def test_cosyvoice_french_instruct_avoids_english_volume_and_adds_language() -> None:
    instruct = cosyvoice_instruct(
        ProsodySpec(emotion="happy", pace="fast", volume="loud"), language="French"
    )
    assert "as loudly as possible" not in instruct
    assert "请大声地说一句话" in instruct
    assert instruct.endswith("请用法语说这句话。<|endofprompt|>")


def test_cosyvoice_english_instruct_keeps_the_english_volume_clause() -> None:
    instruct = cosyvoice_instruct(ProsodySpec(volume="loud"), language="English")
    assert "as loudly as possible" in instruct
    assert "请用法语说这句话" not in instruct


def test_cosyvoice_french_instruct_omits_pace() -> None:
    instruct = cosyvoice_instruct(
        ProsodySpec(emotion="sad", pace="fast"), language="French"
    )
    assert "语速" not in instruct
    assert "请非常伤心地说一句话" in instruct


def test_customvoice_instruct_renders_a_sentence() -> None:
    instruct = customvoice_instruct(ProsodySpec(emotion="angry", pace="fast"))
    assert "angry" in instruct
    assert "quickly" in instruct


def test_explicit_exaggeration_wins_over_the_emotion_table() -> None:
    assert (
        chatterbox_exaggeration(ProsodySpec(emotion="sad", exaggeration=0.95))
        == 0.95
    )


def test_parse_reads_an_explicit_exaggeration() -> None:
    assert parse_prosody_spec("exaggeration=0.7").exaggeration == 0.7


def test_parse_rejects_an_out_of_range_exaggeration() -> None:
    with pytest.raises(ConfigError, match="Invalid prosody"):
        parse_prosody_spec("exaggeration=1.5")


def test_planner_logs_the_raw_request_and_reply(caplog) -> None:
    planner = ProsodyPlanner(
        base_url="http://x",
        api_key="k",
        model="m",
        backend="cosyvoice3",
        client=_FakeClient(),
    )

    with caplog.at_level(logging.DEBUG, logger="pipeline.prosody"):
        planner.plan(["first"])

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Prosody request" in text
    assert "Backend capabilities" in text
    assert "Prosody reply" in text


# --------------------------------------------------------------------------- #
# Whole-passage overview + chunking, and the verbatim style planner
# --------------------------------------------------------------------------- #


def _reply(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _requested_ids(user: str) -> list[int]:
    return [int(value) for value in re.findall(r'"id": (\d+)', user)]


class _PlanFakeClient:
    """Answers prosody/style requests, including the free-text overview pass."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail
        outer = self

        class Completions:
            def create(self, **kwargs):
                if outer.fail:
                    raise RuntimeError("boom")
                outer.calls.append(kwargs)
                system = kwargs["messages"][0]["content"]
                user = kwargs["messages"][1]["content"]
                if "delivered overall" in system or "spoken overall" in system:
                    content = "A calm opening that builds into quiet anger."
                elif "dubbing prosody director" in system:
                    entries = [
                        {"id": i, "emotion": "calm"} for i in _requested_ids(user)
                    ]
                    content = json.dumps({"prosody": entries})
                elif "whole passage below" in user:
                    content = json.dumps(
                        {"instructions": [{"id": 0, "text": "One direction."}]}
                    )
                else:
                    entries = [
                        {"id": i, "text": f"Direction {i}."}
                        for i in _requested_ids(user)
                    ]
                    content = json.dumps({"instructions": entries})
                return _reply(content)

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_style_planner_writes_one_instruction_per_line() -> None:
    client = _PlanFakeClient()
    planner = StylePlanner(
        base_url="http://x", api_key="k", model="m", client=client
    )

    assert planner.plan_lines(["first", "second"]) == [
        "Direction 0.",
        "Direction 1.",
    ]
    assert len(client.calls) == 1


def test_style_planner_defaults_temperature_to_zero_point_seven() -> None:
    client = _PlanFakeClient()
    planner = StylePlanner(
        base_url="http://x", api_key="k", model="m", client=client
    )

    assert planner.temperature == 0.7
    planner.plan_lines(["first"])
    assert client.calls[0]["temperature"] == 0.7


def test_style_planner_whole_writes_one_instruction() -> None:
    client = _PlanFakeClient()
    planner = StylePlanner(
        base_url="http://x", api_key="k", model="m", client=client
    )

    assert planner.plan_whole(["first", "second"]) == "One direction."
    assert len(client.calls) == 1
    assert "Passage:" in client.calls[0]["messages"][1]["content"]


def test_style_planner_chunks_with_a_whole_passage_overview() -> None:
    client = _PlanFakeClient()
    planner = StylePlanner(
        base_url="http://x", api_key="k", model="m", client=client, chunk_size=2
    )

    instructs = planner.plan_lines([f"line {i}" for i in range(5)])

    assert len(instructs) == 5
    # One overview call, then ceil(5 / 2) = 3 chunk calls.
    assert len(client.calls) == 4
    assert "Passage:" in client.calls[0]["messages"][1]["content"]
    chunk_users = [call["messages"][1]["content"] for call in client.calls[1:]]
    assert "chunk 1 of 3" in chunk_users[0]
    assert "chunk 3 of 3" in chunk_users[2]
    assert all("Overall direction" in user for user in chunk_users)
    assert "A calm opening" in chunk_users[0]


def test_style_planner_logs_each_phase(caplog) -> None:
    planner = StylePlanner(
        base_url="http://x",
        api_key="k",
        model="m",
        client=_PlanFakeClient(),
        chunk_size=2,
    )

    with caplog.at_level(logging.DEBUG, logger="pipeline.prosody"):
        planner.plan_lines([f"line {i}" for i in range(5)])

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Style request (overview" in text
    assert "Style reply (overview" in text
    assert "Style request (chunk 1/3)" in text


def test_style_planner_raises_on_a_failed_request() -> None:
    planner = StylePlanner(
        base_url="http://x",
        api_key="k",
        model="m",
        client=_PlanFakeClient(fail=True),
    )

    with pytest.raises(TranslationError, match="Style planning failed"):
        planner.plan_lines(["first"])


def test_prosody_planner_chunks_with_a_whole_passage_overview() -> None:
    client = _PlanFakeClient()
    planner = ProsodyPlanner(
        base_url="http://x", api_key="k", model="m", client=client, chunk_size=2
    )

    specs = planner.plan([f"line {i}" for i in range(5)])

    assert len(specs) == 5
    assert all(spec.emotion == "calm" for spec in specs)
    assert len(client.calls) == 4
    chunk_users = [call["messages"][1]["content"] for call in client.calls[1:]]
    assert "chunk 1 of 3" in chunk_users[0]
    assert all("Overall direction" in user for user in chunk_users)


def test_style_messages_omit_speaker_and_language_when_not_given() -> None:
    system = build_style_messages(["one"])[0]["content"]

    assert "The speaker" not in system
    assert "The lines are in" not in system


def test_style_messages_include_speaker_and_language_when_given() -> None:
    system = build_style_messages(
        ["one"], speaker="Ryan", language="English"
    )[0]["content"]

    assert "Ryan" in system
    assert "English" in system


def test_style_prompt_frames_its_examples_as_guides() -> None:
    system = build_style_messages(["one"])[0]["content"]

    assert "incredulous tone" in system
    assert "guides, not templates" in system


def test_parse_style_response_defaults_gaps_to_empty() -> None:
    reply = '{"instructions":[{"id":0,"text":"Speak softly."}]}'

    assert parse_style_response(reply, 3) == ["Speak softly.", "", ""]


def test_parse_style_response_takes_the_first_entry_for_a_single_target() -> None:
    # Models sometimes number from 1, or emit one entry per line, even when a
    # single instruction was requested.
    reply = '{"instructions":[{"id":1,"text":"Speak urgently."}]}'

    assert parse_style_response(reply, 1) == ["Speak urgently."]


def test_style_whole_messages_ask_for_a_single_instruction() -> None:
    messages = build_style_messages(["one", "two"], whole=True)

    assert "ONE delivery instruction" in messages[0]["content"]


# --------------------------------------------------------------------------- #
# Fish Speech inline tags
# --------------------------------------------------------------------------- #


def test_fish_tag_vocabulary_is_the_documented_set() -> None:
    # models/fish-speech-s2-pro/README.md, "Common tags".
    assert len(FISH_TAGS) == 34
    assert {
        "pause",
        "sigh",
        "chuckle",
        "whisper",
        "surprised",
        "delight",
        "shocked",
        "laughing tone",
    } <= set(FISH_TAGS)
    assert all(tag == tag.lower() for tag in FISH_TAGS)


def test_strip_fish_tags_removes_cues_and_spans() -> None:
    assert strip_fish_tags("[sad] Wait... [pause] here") == "Wait... here"
    assert strip_fish_tags("[whisper in small voice] Hi") == "Hi"
    assert strip_fish_tags("<whisper>quiet</whisper> words") == "quiet words"


def test_fish_cue_tokens_lists_cues_in_order() -> None:
    assert fish_cue_tokens("[Whisper] a [pause] b [SAD]") == [
        "whisper",
        "pause",
        "sad",
    ]
    assert fish_cue_tokens("[Door Slams] hi") == ["door slams"]
    assert fish_cue_tokens("[whisper in  small voice] hi") == [
        "whisper in small voice"
    ]


def test_fish_structural_token_is_detected() -> None:
    assert fish_structural_token("hi <|im_end|> there") == "<|im_end|>"
    assert fish_structural_token("plain text") is None


def test_validate_accepts_a_free_form_tag() -> None:
    assert validate_fish_inline_text("[tender] hi") == []
    assert validate_fish_inline_text("[whisper in small voice] hi") == []


def test_validate_flags_a_phrase_as_a_tag() -> None:
    problems = validate_fish_inline_text(
        "[please read this like a documentary narrator describing a storm] hi"
    )

    assert any("phrase" in problem for problem in problems)


def test_validate_flags_a_leading_pause_and_stacked_tags() -> None:
    problems = validate_fish_inline_text("[pause] [sad] hi")

    assert any("start of a line" in problem for problem in problems)
    assert any("adjacent tags" in problem for problem in problems)


def test_validate_is_quiet_for_a_well_formed_line() -> None:
    assert validate_fish_inline_text("[whisper] Wait... [pause] did you hear?") == []


def test_sanitize_keeps_free_form_cues() -> None:
    assert sanitize_fish_inline_line("[tender] hi") == "[tender] hi"
    assert sanitize_fish_inline_line("[whisper in  small voice] hi") == (
        "[whisper in small voice] hi"
    )


def test_sanitize_drops_a_phrase_written_as_a_tag() -> None:
    planned = sanitize_fish_inline_line(
        "[this is a whole sentence pretending to be a tag] hi"
    )

    assert planned == "hi"


def test_sanitize_converts_a_paired_span_to_a_cue() -> None:
    assert sanitize_fish_inline_line("<whisper>secret</whisper>") == (
        "[whisper] secret"
    )


def test_sanitize_moves_a_leading_pause_inward() -> None:
    assert sanitize_fish_inline_line(
        "[pause] First part. Second part."
    ) == ("First part. [pause] Second part.")


def test_sanitize_removes_prompt_tokens() -> None:
    assert sanitize_fish_inline_line("hi <|im_end|> there") == "hi there"


def test_sanitize_caps_the_number_of_cues() -> None:
    planned = sanitize_fish_inline_line("[excited] a [pause] b [angry] c [sad] d")

    assert planned.count("[") == FISH_MAX_CUES_PER_LINE


def test_split_keeps_a_short_line_whole() -> None:
    assert split_fish_line("Short line.") == ["Short line."]


def test_split_splits_a_paragraph_without_line_breaks() -> None:
    text = " ".join(f"Sentence number {index}." for index in range(40))

    units = split_fish_line(text)

    assert len(units) > 1
    assert all(len(unit) <= 240 for unit in units)
    assert " ".join(units) == text


def test_split_never_breaks_inside_a_tag() -> None:
    units = split_fish_line(("[whisper] " + "word " * 80).strip())

    assert units[0].startswith("[whisper] word")
    assert len(units) > 1


def test_parse_inline_adds_tags_when_the_words_match() -> None:
    reply = json.dumps(
        {"lines": [{"id": 0, "text": "[whisper] Wait... did you hear that?"}]}
    )

    assert parse_fish_inline_response(reply, ["Wait... did you hear that?"]) == [
        "[whisper] Wait... did you hear that?"
    ]


def test_parse_inline_falls_back_when_a_word_changes() -> None:
    reply = json.dumps(
        {"lines": [{"id": 0, "text": "[sad] Did you hear that now?"}]}
    )

    assert parse_fish_inline_response(reply, ["Did you hear that?"]) == [
        "Did you hear that?"
    ]


def test_parse_inline_keeps_tags_when_only_spacing_is_fixed() -> None:
    # Text pasted from a document loses the space after a full stop, and the
    # planner re-inserts it. That is typography, not a change to the words, so
    # the line keeps its tags instead of falling back to the untagged source.
    source = (
        "Le son est net, même sur des haut-parleurs modestes.Et pour finir… "
        "une phrase plus douce."
    )
    tagged = (
        "Le son est net, [short pause] même sur des haut-parleurs modestes. "
        "[low volume] Et pour finir… une phrase plus douce."
    )
    reply = json.dumps({"lines": [{"id": 0, "text": tagged}]})

    assert parse_fish_inline_response(reply, [source]) == [tagged]


def test_parse_inline_still_rejects_a_real_word_change() -> None:
    source = "Le son est net. Et pour finir, une phrase plus douce."
    reply = json.dumps(
        {
            "lines": [
                {
                    "id": 0,
                    "text": (
                        "[low volume] Le son est clair. Et pour finir, une "
                        "phrase plus douce."
                    ),
                }
            ]
        }
    )

    # "net" -> "clair" is a word change, not spacing: the tags are dropped.
    assert parse_fish_inline_response(reply, [source]) == [source]


def test_parse_inline_keeps_the_source_when_the_reply_is_empty() -> None:
    assert parse_fish_inline_response("no json here", ["one", "two"]) == [
        "one",
        "two",
    ]


def test_parse_inline_aligns_entries_numbered_from_one() -> None:
    reply = json.dumps({"lines": [{"id": 1, "text": "[sad] one"}]})

    assert parse_fish_inline_response(reply, ["one"]) == ["[sad] one"]


def test_parse_inline_defaults_a_missing_line_to_its_source() -> None:
    reply = json.dumps({"lines": [{"id": 0, "text": "[sad] one"}]})

    assert parse_fish_inline_response(reply, ["one", "two"]) == ["[sad] one", "two"]


def test_fish_inline_prompt_explains_free_form_and_human_delivery() -> None:
    system = build_fish_inline_messages(["one"])[0]["content"]

    assert "FREE-FORM" in system
    assert "[whisper in small voice]" in system
    assert "15,000+" in system
    assert "sound like a person saying it" in system
    assert "Copy every word exactly" in system
    assert "or any <|...|> token" in system


def _requested_texts(user: str) -> list[str]:
    body = user.split("Lines to direct:", 1)[-1]
    return [
        json.loads(line)["text"]
        for line in body.splitlines()
        if line.strip().startswith("{")
    ]


class _FishFakeClient:
    """Answers inline-tagging requests, prefixing every line with one tag."""

    def __init__(self, *, tag: str = "sad", forbidden: str | None = None):
        self.calls: list[dict] = []
        self.tag = tag
        self.forbidden = forbidden
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                system = kwargs["messages"][0]["content"]
                user = kwargs["messages"][1]["content"]
                if "delivered overall" in system:
                    return _reply("Quiet, then a turn.")
                entries = []
                for index, text in enumerate(_requested_texts(user)):
                    body = text
                    if outer.forbidden is not None:
                        body = outer.forbidden.format(text=text)
                    entries.append(
                        {"id": index, "text": f"[{outer.tag}] {body}"}
                    )
                return _reply(json.dumps({"lines": entries}))

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_fish_inline_planner_writes_tags_into_the_text() -> None:
    client = _FishFakeClient()
    planner = FishInlinePlanner(
        base_url="http://x", api_key="k", model="m", client=client
    )

    assert planner.plan_lines(["first", "second"]) == [
        "[sad] first",
        "[sad] second",
    ]
    assert client.calls[0]["temperature"] == 0.7


def test_fish_inline_planner_splits_and_rejoins_a_long_line() -> None:
    client = _FishFakeClient()
    planner = FishInlinePlanner(
        base_url="http://x", api_key="k", model="m", client=client, chunk_size=10
    )
    long_line = "First sentence here. " * 20

    planned = planner.plan_lines([long_line])

    assert len(planned) == 1
    assert planned[0].count("[sad]") > 1
    assert strip_fish_tags(planned[0]) == strip_fish_tags(long_line)


def test_fish_inline_planner_uses_an_overview_when_chunked() -> None:
    client = _FishFakeClient()
    planner = FishInlinePlanner(
        base_url="http://x", api_key="k", model="m", client=client, chunk_size=1
    )

    planner.plan_lines(["First sentence here. " * 20])

    assert len(client.calls) > 2
    assert "Passage:" in client.calls[0]["messages"][1]["content"]
    chunk_users = [call["messages"][1]["content"] for call in client.calls[1:]]
    assert all("Overall direction" in user for user in chunk_users)


def test_fish_inline_planner_degrades_to_the_source_text() -> None:
    client = _FishFakeClient(forbidden="Totally different words.")
    planner = FishInlinePlanner(
        base_url="http://x", api_key="k", model="m", client=client
    )

    assert planner.plan_lines(["keep these words"]) == ["keep these words"]


def test_backend_capabilities_cover_the_surviving_backends() -> None:
    assert set(BACKEND_CAPABILITIES) == {
        "qwen",
        "cosyvoice3",
        "chatterbox",
        "customvoice",
    }


def test_planner_records_config_and_reply() -> None:
    from pipeline.provenance import ProvenanceRecorder

    recorder = ProvenanceRecorder("say", "chatterbox", language="French")
    planner = ProsodyPlanner(
        base_url="http://x",
        api_key="secret-key",
        model="m",
        temperature=0.2,
        backend="chatterbox",
        client=_PlanFakeClient(),
        recorder=recorder,
        chunk_size=40,
    )

    planner.plan(["first", "second"])
    report = recorder.build()

    assert report.planner is not None
    assert report.planner.config is not None
    assert report.planner.config.model == "m"
    assert report.planner.config.backend == "chatterbox"
    assert report.planner.config.temperature == 0.2
    assert report.planner.calls
    assert "prosody" in report.planner.calls[0].response
    # The key must never reach the artifact.
    assert "secret-key" not in json.dumps(report.to_dict())


def test_planner_records_the_overview_when_chunked() -> None:
    from pipeline.provenance import ProvenanceRecorder

    recorder = ProvenanceRecorder("dub", "chatterbox", language="French")
    planner = ProsodyPlanner(
        base_url="http://x",
        api_key="k",
        model="m",
        backend="chatterbox",
        client=_PlanFakeClient(),
        recorder=recorder,
        chunk_size=1,
    )

    planner.plan(["first line", "second line"])
    report = recorder.build()

    assert report.planner is not None
    assert report.planner.overview is not None
    assert report.planner.overview.startswith("A calm opening")
    assert len(report.planner.calls) >= 2


def test_planner_records_a_failure_and_still_raises() -> None:
    from pipeline.provenance import ProvenanceRecorder

    recorder = ProvenanceRecorder("dub", "chatterbox", language="French")
    planner = ProsodyPlanner(
        base_url="http://x",
        api_key="k",
        model="m",
        backend="chatterbox",
        client=_PlanFakeClient(fail=True),
        recorder=recorder,
        chunk_size=40,
    )

    with pytest.raises(TranslationError):
        planner.plan(["first"])

    report = recorder.build()
    assert report.planner is not None
    assert report.planner.error is not None
    assert "boom" in report.planner.error
