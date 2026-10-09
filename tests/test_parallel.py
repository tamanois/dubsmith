"""Fan-out planning, task execution and the worker pool.

Two layers are covered separately, because they fail in different ways:

* the **task** layer - what a worker is asked to do, and what it does with the
  prompt - is exercised against the same fake engine the rest of the suite uses,
  so no model is needed;
* the **pool** layer - dispatch, retry, teardown - is exercised with a stub
  worker that answers without a model, and therefore runs in CI on a machine with
  one processor and no GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.errors import TTSError
from pipeline.parallel import (
    BLEND,
    PROBE,
    SYNTH,
    InlineRunner,
    ProcessRunner,
    Result,
    Task,
    TaskFailure,
    WorkerConfig,
    merge_tts_stats,
    split_evenly,
)
from pipeline.voice import VoiceSpec, build_prompt


def make_spec(speaker: str = "SPK_0", **kwargs: object) -> VoiceSpec:
    """A spec with the long-standing ``preserve`` shape, plus any overrides.

    The donor is a stand-in object: the engine is fake throughout, so what matters
    is *which* source a spec names, not what the source contains.
    """
    fields: dict[str, object] = {
        "speaker": speaker,
        "accent_mode": "preserve",
        "donor": "donor",
    }
    fields.update(kwargs)
    return VoiceSpec(**fields)  # type: ignore[arg-type]


def fake_engine():  # noqa: ANN201 - the fake lives in the pipeline test module
    """The suite's fake engine, imported lazily so a spawned child stays light."""
    from .test_pipeline import FakeTTSEngine

    return FakeTTSEngine()


class TestSplitEvenly:
    """The pieces a worker is handed, which is the whole balancing policy."""

    def test_no_items_needs_no_pieces(self) -> None:
        assert split_evenly([], 2) == []

    def test_one_part_keeps_the_order(self) -> None:
        assert split_evenly([1, 2, 3], 1) == [[1, 2, 3]]

    def test_pieces_are_equal_when_the_count_divides(self) -> None:
        assert split_evenly([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_the_remainder_goes_to_the_first_pieces(self) -> None:
        # 5 items over 3 workers: 2/2/1, never 1/1/3 - the largest piece is what
        # decides when a worker frees up.
        assert split_evenly([1, 2, 3, 4, 5], 3) == [[1, 2], [3, 4], [5]]

    def test_more_parts_than_items_gives_one_piece_each(self) -> None:
        assert split_evenly([1, 2], 8) == [[1], [2]]

    def test_pieces_stay_contiguous(self) -> None:
        pieces = split_evenly(list(range(10)), 4)
        assert [item for piece in pieces for item in piece] == list(range(10))


class TestMergeTtsStats:
    """The report block: one card is unchanged, several are folded."""

    def test_nothing_measured_is_an_empty_block(self) -> None:
        assert merge_tts_stats([]) == {}

    def test_one_report_passes_through_untouched(self) -> None:
        report = {"batches": 3, "peak_reserved_bytes": 10}
        assert merge_tts_stats([report]) == report

    def test_several_reports_are_summed_and_the_worst_peak_kept(self) -> None:
        merged = merge_tts_stats(
            [
                {
                    "batches": 3,
                    "oom_retries": 1,
                    "peak_reserved_bytes": 10,
                    "batch_size": 4,
                },
                {
                    "batches": 5,
                    "oom_retries": 0,
                    "peak_reserved_bytes": 30,
                    "batch_size": 2,
                },
            ]
        )
        assert merged["batches"] == 8
        assert merged["oom_retries"] == 1
        assert merged["peak_reserved_bytes"] == 30
        # The per-card detail survives: a shared maximum hides which card was tight.
        assert merged["batch_size"] == [4, 2]
        assert len(merged["per_device"]) == 2


class TestVoiceSpecKey:
    """The prompt cache is only sound if two different voices never share a key."""

    def test_a_shared_voice_is_one_key(self) -> None:
        first = make_spec("SPK_0", shared=True)
        second = make_spec("SPK_1", shared=True)
        assert first.key == second.key

    def test_speakers_differ(self) -> None:
        assert make_spec("SPK_0").key != make_spec("SPK_1").key

    def test_a_different_blend_is_a_different_voice(self) -> None:
        first = make_spec(blend=Path("a.wav"))
        second = make_spec(blend=Path("b.wav"))
        assert first.key != second.key

    def test_the_same_facts_are_stable(self) -> None:
        assert make_spec(donor=None).key == make_spec(donor=None).key


class TestBuildPrompt:
    """The four ways a voice is built, shared with every worker process."""

    def test_a_blend_clip_is_the_voice(self) -> None:
        engine = fake_engine()
        build_prompt(
            engine,
            make_spec(
                accent_mode="native",
                blend=Path("ref_blend.wav"),
                accent_text="bonjour",
            ),
        )
        assert engine.prompt_kind == "file"
        assert engine.file_prompt == (Path("ref_blend.wav"), "bonjour")

    def test_native_without_a_blend_combines_accent_and_timbre(self) -> None:
        engine = fake_engine()
        build_prompt(
            engine,
            make_spec(
                accent_mode="native",
                accent_audio=Path("anchor.wav"),
                accent_text="bonjour",
                donor="donor",
            ),
        )
        assert engine.prompt_kind == "hybrid"
        assert engine.hybrid is not None
        assert engine.hybrid["accent_audio"] == Path("anchor.wav")
        assert engine.hybrid["timbre"] == "donor"

    def test_a_pinned_reference_is_cloned_directly(self) -> None:
        engine = fake_engine()
        build_prompt(engine, make_spec(ref_audio=Path("ref.wav"), ref_text="hello"))
        assert engine.prompt_kind == "file"
        assert engine.file_prompt == (Path("ref.wav"), "hello")

    def test_a_donor_is_cloned_when_nothing_is_pinned(self) -> None:
        engine = fake_engine()
        build_prompt(engine, make_spec(donor="donor"))
        assert engine.prompt_kind == "donor"

    def test_no_timbre_source_is_an_error_rather_than_a_silent_voice(self) -> None:
        with pytest.raises(TTSError):
            build_prompt(fake_engine(), make_spec(donor=None))


class TestInlineRunner:
    """The single-device path: this is what every existing run does."""

    def test_tasks_run_on_the_engine_it_was_given(self) -> None:
        engine = fake_engine()
        runner = InlineRunner(engine)
        tasks = [
            Task(
                kind=SYNTH,
                spec=make_spec(),
                language="French",
                lines=((1, "bonjour"),),
            ),
            Task(kind=PROBE, spec=make_spec(), language="French", text="bonjour"),
        ]
        results = runner.run(tasks)
        assert [result.kind for result in results] == [SYNTH, PROBE]
        assert results[1].rate == engine.delivery_rate
        assert engine.synth_calls == 1

    def test_one_prompt_serves_every_task_that_shares_a_voice(self) -> None:
        engine = fake_engine()
        runner = InlineRunner(engine)
        tasks = [
            Task(kind=SYNTH, spec=make_spec(), language="French", lines=((1, "un"),)),
            Task(kind=SYNTH, spec=make_spec(), language="French", lines=((2, "deux"),)),
        ]
        runner.run(tasks)
        # Two tasks, one voice: the reference is encoded once, not once per task,
        # which is what makes cutting a single speaker's lines up affordable.
        assert engine.prompt_calls == 1

    def test_stats_mirror_the_engine(self) -> None:
        engine = fake_engine()
        runner = InlineRunner(engine)
        runner.run(
            [Task(kind=SYNTH, spec=make_spec(), language="French", lines=((1, "a"),))]
        )
        assert runner.stats()[0]["batches"] == engine.synth_calls

    def test_close_is_safe_to_call_twice(self) -> None:
        runner = InlineRunner(fake_engine())
        runner.close()
        runner.close()


class TestExecute:
    """What one task does, independent of who runs it."""

    def test_synthesis_keeps_a_blank_line_blank(self) -> None:
        engine = fake_engine()
        runner = InlineRunner(engine)
        result = runner.run(
            [
                Task(
                    kind=SYNTH,
                    spec=make_spec(),
                    language="French",
                    lines=((1, "bonjour"), (2, "   ")),
                )
            ]
        )[0]
        ids = [segment_id for segment_id, _wav in result.wavs]
        assert ids == [1, 2]
        assert result.wavs[1][1] is None
        assert result.sample_rate == engine.sample_rate

    def test_a_blend_invalidates_the_loaded_prompt(self) -> None:
        engine = fake_engine()
        runner = InlineRunner(engine)
        spec = make_spec(
            accent_mode="native", accent_audio=Path("anchor.wav"), donor="donor"
        )
        runner.run(
            [
                Task(
                    kind=BLEND,
                    spec=spec,
                    language="French",
                    takes=1,
                    iters=1,
                    ratio=0.0,
                ),
                Task(kind=SYNTH, spec=spec, language="French", lines=((1, "bonjour"),)),
            ]
        )
        # The blend leaves its own hybrid prompt loaded, so the synthesis after it
        # has to rebuild rather than trust the cache.
        assert engine.prompt_calls == 2

    def test_an_unknown_kind_is_an_error(self) -> None:
        with pytest.raises(TTSError):
            InlineRunner(fake_engine()).run(
                [Task(kind="nonsense", spec=make_spec(), language="French")]
            )


# --------------------------------------------------------------------------- #
# The pool itself: spawn, dispatch, retry, teardown
# --------------------------------------------------------------------------- #


def stub_worker(config: WorkerConfig, conn) -> None:  # noqa: ANN001 - a pipe
    """A worker with no model, so the pool can be tested without one.

    Answers every task, counts what it did, and refuses anything sent to a device
    named ``bad`` - which is how the retry-on-another-card path is provoked.
    """
    batches = 0
    while True:
        try:
            task = conn.recv()
        except EOFError:
            return
        if task is None:
            return
        if config.device == "bad":
            conn.send(TaskFailure(config.device, "no room on this card"))
            continue
        batches += 1
        conn.send(
            Result(
                kind=task.kind,
                key=task.spec.key,
                device=config.device,
                wavs=[
                    (segment_id, np.full(4, batches, dtype=np.float32))
                    for segment_id, _text in task.lines
                ],
                sample_rate=16_000,
                stats={"batches": batches, "peak_reserved_bytes": 1024 * batches},
            )
        )


def worker(device: str) -> WorkerConfig:
    """A worker config the stub target ignores except for its name."""
    return WorkerConfig(model_dir=Path("/nonexistent"), device=device)


def synth_task(line_id: int, speaker: str = "SPK_0") -> Task:
    return Task(
        kind=SYNTH,
        spec=make_spec(speaker),
        language="French",
        lines=((line_id, f"line {line_id}"),),
    )


class TestProcessRunner:
    """One worker per device, fed as each finishes."""

    def test_work_is_spread_over_every_device(self) -> None:
        runner = ProcessRunner([worker("good-0"), worker("good-1")], target=stub_worker)
        try:
            results = runner.run([synth_task(index) for index in range(4)])
        finally:
            runner.close()
        assert len(results) == 4
        # Dynamic dispatch means no guarantee about *which* task went where, but a
        # pool of two must have used both.
        assert {result.device for result in results} == {"good-0", "good-1"}

    def test_stats_are_kept_per_device(self) -> None:
        runner = ProcessRunner([worker("good-0"), worker("good-1")], target=stub_worker)
        try:
            runner.run([synth_task(1), synth_task(2)])
            stats = runner.stats()
        finally:
            runner.close()
        assert len(stats) == 2
        assert sum(report["batches"] for report in stats) == 2

    def test_two_workers_on_one_device_are_both_reported(self) -> None:
        # ``--devices cpu,cpu`` is how the fan-out is exercised on a machine with a
        # single card, so the counters are keyed by worker, not by device name.
        runner = ProcessRunner([worker("cpu"), worker("cpu")], target=stub_worker)
        try:
            runner.run([synth_task(1), synth_task(2)])
            stats = runner.stats()
        finally:
            runner.close()
        assert len(stats) == 2

    def test_a_failed_task_is_retried_on_another_card(self) -> None:
        runner = ProcessRunner(
            [worker("bad"), worker("good-0")], target=stub_worker, max_attempts=2
        )
        try:
            results = runner.run([synth_task(1)])
        finally:
            runner.close()
        # The first attempt went to the card that refused it; the retry did not go
        # back to it, because a card that just failed is not evidence about the line.
        assert [result.device for result in results] == ["good-0"]

    def test_a_task_no_card_can_run_fails_the_job(self) -> None:
        runner = ProcessRunner(
            [worker("bad"), worker("bad")], target=stub_worker, max_attempts=2
        )
        try:
            with pytest.raises(TTSError, match="no room on this card"):
                runner.run([synth_task(1)])
        finally:
            runner.close()

    def test_no_tasks_means_no_processes(self) -> None:
        runner = ProcessRunner([worker("good-0"), worker("good-1")], target=stub_worker)
        try:
            assert runner.run([]) == []
            assert runner.stats() == []
            # A job served entirely from cache must not pay for two workers.
            assert runner._workers == []
        finally:
            runner.close()

    def test_close_stops_the_workers(self) -> None:
        runner = ProcessRunner([worker("good-0"), worker("good-1")], target=stub_worker)
        runner.run([synth_task(1)])
        processes = [entry.process for entry in runner._workers]
        assert len(processes) == 2
        assert all(process.is_alive() for process in processes)
        runner.close()
        assert not any(process.is_alive() for process in processes)

    def test_a_card_that_cannot_open_its_engine_says_why(self) -> None:
        # The real worker entry point, with a dtype no device can run. A card that
        # cannot even build its engine must report the reason rather than vanish,
        # so the failure a user reads names the cause.
        configs = [
            WorkerConfig(model_dir=Path("/nonexistent"), device="cpu", dtype="nonsense")
        ]
        runner = ProcessRunner(configs, max_attempts=1)
        try:
            with pytest.raises(TTSError, match="could not load the checkpoint"):
                runner.run([synth_task(1)])
        finally:
            runner.close()

    def test_an_empty_config_list_is_refused(self) -> None:
        with pytest.raises(TTSError):
            ProcessRunner([], target=stub_worker)
