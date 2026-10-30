"""Spread one job's synthesis over several devices.

The pipeline runs one stage at a time and loads a multi-GB checkpoint per stage,
sizing each synthesis batch from the free memory of *its* card. That design is
kept: this module only lets several such processes work at once, one per device,
each taking a share of the same stage. Nothing is sharded inside a model - a
clone prompt is bound to the device that built it, and the checkpoint is small
enough that one copy per card is the honest trade.

Three properties everything else follows from:

* **The parent owns the artifacts.** A worker returns audio and counters, never a
  path it wrote, so the job folder, the stage cache and the manifest keep exactly
  one writer.
* **Processes, not threads.** The caching allocator's per-process state is what
  caused the documented long-video OOM, so isolation is the point rather than an
  implementation detail. The pool uses ``spawn``: a forked child inherits a CUDA
  context and dies the first time it uses it.
* **One device is the old code path.** :class:`InlineRunner` executes tasks in
  this process, on the engine the caller already resolved, so a run with a single
  ``--device`` synthesises exactly what it did before - and the whole test suite
  can exercise the fan-out planning without spawning anything.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .config import configure_allocator
from .errors import TTSError
from .logging_utils import quiet_dependencies
from .tts import TTSEngine, resolve_dtype
from .voice import VoiceSpec, build_prompt

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "InlineRunner",
    "ProcessRunner",
    "Result",
    "Runner",
    "Task",
    "TaskFailure",
    "WorkerConfig",
    "merge_tts_stats",
    "split_evenly",
    "worker_main",
]

#: How long a worker may take to stop before it is terminated. Long enough for a
#: final `synthesis_stats()` round trip, short enough not to hang a finished run.
SHUTDOWN_TIMEOUT_S = 15.0

#: Task kinds. A string keeps the wire format readable in a debugger and stable
#: against a re-ordered enum.
PROBE = "probe"
SYNTH = "synth"
BLEND = "blend"


@dataclass(frozen=True)
class Task:
    """One unit of synthesis work, and everything needed to do it anywhere.

    Deliberately self-contained: it carries the voice to build, not a built
    prompt, because the prompt belongs to the process that made it. The same
    object is executed by an inline runner in this process and pickled to a
    worker in another, so it may not hold anything device-bound.
    """

    kind: str
    spec: VoiceSpec
    language: str
    #: ``(segment_id, text)`` pairs, in order, for a ``synth`` task.
    lines: tuple[tuple[int, str], ...] = ()
    #: Text to read for a ``probe`` task.
    text: str = ""
    #: Blend knobs, used only by a ``blend`` task.
    takes: int = 1
    iters: int = 1
    ratio: float = 0.0


@dataclass
class Result:
    """What a task produced, in a form the parent can use without an engine."""

    kind: str
    #: The :attr:`VoiceSpec.key` of the prompt that produced it, so the caller can
    #: tell which speakers shared one prompt.
    key: str
    #: Device the work ran on, for the log and the report.
    device: str = ""
    #: ``(segment_id, samples)`` pairs; ``samples`` is ``None`` for a blank line.
    wavs: list[tuple[int, np.ndarray | None]] = field(default_factory=list)
    #: Words per second, for a ``probe`` task.
    rate: float = 0.0
    #: The rendered clip, for a ``blend`` task (a :class:`BlendedReference`).
    blend: Any | None = None
    #: Model output rate. Carried because the parent writes the per-line WAVs and
    #: must not have to load an engine just to learn it.
    sample_rate: int = 0
    #: How long the task took inside :func:`execute`, in seconds.
    seconds: float = 0.0
    #: The worker's live counters, refreshed with every result.
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskFailure:
    """A task that raised, sent back instead of killing the worker.

    The child stays up: one bad line is not a reason to lose a loaded checkpoint,
    and the parent may still be able to run that task on another card.
    """

    device: str
    message: str


@dataclass(frozen=True)
class WorkerConfig:
    """What a worker needs to build its own engine, and nothing more."""

    model_dir: Path
    device: str
    dtype: str = "auto"
    batch_size: int = 8
    x_vector_only: bool = False
    #: Mirror the parent's log level into the child, so `-v` also shows worker
    #: load and task detail.
    verbose: bool = False


class Runner(Protocol):
    """Executes synthesis tasks and reports what the devices did."""

    def run(self, tasks: Sequence[Task]) -> list[Result]:
        """Execute ``tasks``, returning one result per task, in order."""
        ...

    def stats(self) -> list[dict[str, Any]]:
        """The latest counters, one entry per device involved."""
        ...

    def close(self) -> None:
        """Release whatever the runner holds. Safe to call twice."""
        ...


# --------------------------------------------------------------------------- #
# Task execution - shared by the inline runner and the worker process
# --------------------------------------------------------------------------- #


def _sample_rate(engine: Any) -> int:
    """The engine's output rate, or ``0`` before it has produced anything.

    The real engine raises until the first synthesis, and a blank-lines-only job
    never gets there; a caller that only needs the rate to write a file has
    nothing to write in that case, so zero is a safe answer.
    """
    try:
        return int(engine.sample_rate)
    except Exception:  # noqa: BLE001 - an unknown rate is not an error here
        return 0


def _ensure_prompt(spec: VoiceSpec, engine: Any, cache: dict[str, Any]) -> None:
    """Build ``spec``'s prompt unless the engine already holds exactly that one."""
    if cache.get("key") == spec.key:
        return
    build_prompt(engine, spec)
    cache["key"] = spec.key


def execute(task: Task, engine: Any, cache: dict[str, Any]) -> Result:
    """Run one task on ``engine``, reusing the prompt ``cache`` describes.

    A worker sees a speaker's lines as several tasks when a single voice has to be
    spread over several cards, so the cache is what keeps that from re-encoding
    the reference once per task.
    """
    started = time.perf_counter()
    if task.kind == BLEND:
        blended = engine.build_blended_reference(
            accent_audio=task.spec.accent_audio,
            accent_text=task.spec.accent_text,
            language=task.language,
            timbre=task.spec.donor,
            timbre_audio=task.spec.ref_audio,
            takes=task.takes,
            iters=task.iters,
            ratio=task.ratio,
        )
        # The blend leaves its own hybrid prompt loaded, which is not the prompt
        # any later task asked for.
        cache["key"] = None
        return Result(
            kind=BLEND,
            key=task.spec.key,
            blend=blended,
            sample_rate=int(blended.sample_rate),
            seconds=round(time.perf_counter() - started, 3),
        )

    _ensure_prompt(task.spec, engine, cache)
    if task.kind == PROBE:
        return Result(
            kind=PROBE,
            key=task.spec.key,
            rate=float(engine.measure_delivery_rate(task.text, language=task.language)),
            sample_rate=_sample_rate(engine),
            seconds=round(time.perf_counter() - started, 3),
        )
    if task.kind == SYNTH:
        texts = [text for _segment_id, text in task.lines]
        produced = engine.synthesize(texts, language=task.language)
        wavs = [
            (segment_id, None if wav is None else np.asarray(wav, dtype=np.float32))
            for (segment_id, _text), wav in zip(task.lines, produced)
        ]
        return Result(
            kind=SYNTH,
            key=task.spec.key,
            wavs=wavs,
            sample_rate=_sample_rate(engine),
            seconds=round(time.perf_counter() - started, 3),
        )
    raise TTSError(f"Unknown synthesis task {task.kind!r}.")


def _open_engine(config: WorkerConfig) -> TTSEngine:
    """Build a worker's engine on its own device.

    The allocator option is applied here as well as in the parent because it is
    read at torch's first allocation: an inherited value is respected, and a
    missing one would otherwise cost a worker the fix that the parent applied for
    its own card.
    """
    configure_allocator(config.device)
    return TTSEngine(
        config.model_dir,
        device=config.device,
        dtype=resolve_dtype(config.device, config.dtype),
        batch_size=config.batch_size,
        x_vector_only=config.x_vector_only,
    )


def worker_main(config: WorkerConfig, conn: Any) -> None:
    """Child entry point: serve tasks until the parent stops us.

    The checkpoint is loaded on the **first** task rather than at startup, so a
    job that turns out to be fully cached never pays for a model load here.

    A card that cannot load its copy of the checkpoint keeps answering, with the
    real reason, instead of dying silently: the parent can then run the work on
    another card and still report *why* this one was dropped, rather than an
    unexplained "worker stopped".
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(processName)s] %(message)s",
    )
    quiet_dependencies()
    if config.verbose:
        _LOGGER.setLevel(logging.DEBUG)
    engine: TTSEngine | None = None
    open_error: str | None = None
    cache: dict[str, Any] = {}
    try:
        while True:
            try:
                task = conn.recv()
            except (EOFError, KeyboardInterrupt):
                return
            if task is None:
                return
            if engine is None and open_error is None:
                try:
                    load_started = time.perf_counter()
                    engine = _open_engine(config)
                    _LOGGER.info(
                        "Loaded the checkpoint on %s in %.2fs",
                        config.device,
                        time.perf_counter() - load_started,
                    )
                except BaseException as exc:  # noqa: BLE001 - reported, not raised
                    open_error = (
                        f"could not load the checkpoint on {config.device}: {exc}"
                    )
                    _LOGGER.exception("%s", open_error)
            if open_error is not None:
                try:
                    conn.send(TaskFailure(config.device, open_error))
                except Exception:  # noqa: BLE001 - the parent is gone
                    return
                continue
            try:
                result = execute(task, engine, cache)
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                _LOGGER.exception("Synthesis task failed on %s", config.device)
                failure = TaskFailure(config.device, f"{type(exc).__name__}: {exc}")
                try:
                    conn.send(failure)
                except Exception:  # noqa: BLE001 - the parent is gone
                    return
                continue
            result.device = config.device
            _LOGGER.debug(
                "Task %s on %s finished in %.2fs",
                task.kind,
                config.device,
                result.seconds,
            )
            try:
                result.stats = engine.synthesis_stats()
            except Exception:  # noqa: BLE001 - counters must never fail a task
                result.stats = {}
            try:
                conn.send(result)
            except Exception:  # noqa: BLE001 - the parent is gone
                return
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Inline runner - the single-device path
# --------------------------------------------------------------------------- #


class InlineRunner:
    """Runs tasks in this process, on the engine the caller already resolved.

    This is what a one-device run uses, which is why adding multi-device support
    did not change a single synthesis call for anyone who did not ask for it. It
    is also the runner every test uses: the fan-out *planning* is exercised
    end-to-end with a fake engine, and only the pool itself needs a subprocess.
    """

    def __init__(self, engine: Any, log: logging.Logger | None = None) -> None:
        self.engine = engine
        self.log = log or _LOGGER
        self._cache: dict[str, Any] = {}

    def run(self, tasks: Sequence[Task]) -> list[Result]:
        results: list[Result] = []
        started = time.perf_counter()
        for task in tasks:
            result = execute(task, self.engine, self._cache)
            result.device = str(getattr(self.engine, "device", "") or "")
            results.append(result)
            self.log.debug(
                "Task %s on %s in %.2fs (%d line(s))",
                task.kind,
                result.device or "inline",
                result.seconds,
                len(task.lines),
            )
        if tasks:
            self.log.info(
                "Ran %d synthesis task(s) in %.2fs",
                len(tasks),
                time.perf_counter() - started,
            )
        return results

    def stats(self) -> list[dict[str, Any]]:
        stats = getattr(self.engine, "synthesis_stats", None)
        return [stats()] if callable(stats) else []

    def close(self) -> None:
        """Nothing to release: the engine belongs to the caller."""
        return None


# --------------------------------------------------------------------------- #
# Process runner - one worker per device
# --------------------------------------------------------------------------- #


@dataclass
class _Worker:
    """Parent-side handle for one child process."""

    #: Position in the pool. The counters are keyed by it rather than by device
    #: name, because ``--devices cpu,cpu`` puts two workers on one name and both
    #: of them have something to report.
    slot: int
    device: str
    process: Any
    conn: Any
    served: int = 0
    dead: bool = False


class ProcessRunner:
    """One worker process per device, fed as each finishes its task.

    Dispatch is a pull, not a static split: a worker is handed the next task the
    moment it reports a result, so a speaker with unusually long lines cannot
    leave a card idle behind it. That also makes the number of tasks independent
    of the number of devices - the planner only has to produce *enough* pieces.
    """

    def __init__(
        self,
        configs: Sequence[WorkerConfig],
        *,
        log: logging.Logger | None = None,
        max_attempts: int = 2,
        target: Any = worker_main,
    ) -> None:
        if not configs:
            raise TTSError("A process runner needs at least one worker.")
        self._configs = list(configs)
        self._log = log or _LOGGER
        #: The callable a child runs. ``worker_main`` in production; tests inject
        #: their own so the pool's dispatch, retry and teardown can be exercised
        #: without spawning a checkpoint.
        self._target = target
        self._workers: list[_Worker] = []
        self._latest: dict[int, dict[str, Any]] = {}
        self._context = multiprocessing.get_context("spawn")
        #: Attempts allowed per task. Two is the useful number: it means "the other
        #: card gets one go at the line that beat this one", without turning a
        #: genuinely impossible line into a loop.
        self.max_attempts = max(1, int(max_attempts))

    # -- lifecycle --------------------------------------------------------- #

    def _start(self) -> None:
        """Spawn the workers. Deferred so a cached job never pays for them."""
        if self._workers:
            return
        for slot, config in enumerate(self._configs):
            parent_conn, child_conn = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=self._target,
                args=(config, child_conn),
                name=f"dub-tts-{config.device}",
                daemon=True,
            )
            process.start()
            # The child holds the only reference it needs; closing the parent's
            # copy is what lets a dead child look like EOF rather than a hang.
            child_conn.close()
            self._workers.append(_Worker(slot, config.device, process, parent_conn))
        self._log.info(
            "Synthesis workers started on %s.",
            ", ".join(config.device for config in self._configs),
        )

    def _live(self) -> list[_Worker]:
        return [worker for worker in self._workers if not worker.dead]

    def close(self) -> None:
        for worker in self._workers:
            if worker.dead:
                continue
            try:
                worker.conn.send(None)
            except Exception:  # noqa: BLE001 - already gone
                worker.dead = True
        for worker in self._workers:
            process = worker.process
            if process.is_alive():
                process.join(timeout=SHUTDOWN_TIMEOUT_S)
            if process.is_alive():  # pragma: no cover - a wedged worker
                self._log.warning(
                    "Worker on %s did not stop; terminating.", worker.device
                )
                process.terminate()
                process.join(timeout=SHUTDOWN_TIMEOUT_S)
            if process.exitcode not in (0, None):
                self._log.debug(
                    "Worker on %s exited with code %s.", worker.device, process.exitcode
                )
            try:
                worker.conn.close()
            except Exception:  # noqa: BLE001 - closing twice is fine
                pass
        self._workers.clear()

    def stats(self) -> list[dict[str, Any]]:
        return [self._latest[slot] for slot in sorted(self._latest)]

    # -- dispatch ---------------------------------------------------------- #

    def run(self, tasks: Sequence[Task]) -> list[Result]:
        if not tasks:
            return []
        started = time.perf_counter()
        self._start()
        pending: deque[int] = deque(range(len(tasks)))
        attempts: dict[int, int] = {}
        last_device: dict[int, str] = {}
        inflight: dict[Any, tuple[_Worker, int]] = {}
        results: dict[int, Result] = {}
        failures: dict[int, str] = {}

        while pending or inflight:
            while pending and len(inflight) < len(self._live()):
                worker = self._pick(self._live(), inflight, last_device, pending[0])
                if worker is None:
                    break
                index = pending.popleft()
                try:
                    worker.conn.send(tasks[index])
                except Exception as exc:  # noqa: BLE001 - the worker went away
                    worker.dead = True
                    self._log.warning(
                        "Could not reach the worker on %s (%s); retrying elsewhere.",
                        worker.device,
                        exc,
                    )
                    pending.appendleft(index)
                    continue
                inflight[worker.conn] = (worker, index)

            if not inflight:
                detail = failures or {"job": "every worker stopped before finishing"}
                raise TTSError(
                    "Every synthesis worker failed; no line could be produced. "
                    f"Last error: {next(iter(detail.values()))}"
                )

            for conn in multiprocessing.connection.wait(list(inflight)):
                worker, index = inflight.pop(conn)
                outcome = self._receive(worker)
                if isinstance(outcome, Result):
                    results[index] = outcome
                    worker.served += 1
                    last_device[index] = worker.device
                    self._log.debug(
                        "Task %d (%s) on %s in %.2fs",
                        index,
                        tasks[index].kind,
                        worker.device,
                        outcome.seconds,
                    )
                    if outcome.stats:
                        self._latest[worker.slot] = dict(outcome.stats)
                    continue
                # A failure: retry it, preferring a different card.
                message = outcome.message
                failures[index] = message
                attempts[index] = attempts.get(index, 0) + 1
                # Remembered as though it had succeeded there, so the retry does
                # not go straight back to the card that just rejected it.
                last_device[index] = worker.device
                if attempts[index] < self.max_attempts and self._live():
                    self._log.warning(
                        "Synthesis failed on %s (%s); retrying on another device.",
                        worker.device,
                        message,
                    )
                    pending.appendleft(index)
                    continue
                raise TTSError(
                    f"Synthesis failed on every attempted device. Last error: {message}"
                )

        self._log.info(
            "Synthesis fan-out: %d task(s) over %d device(s) in %.2fs",
            len(tasks),
            len(self._live()),
            time.perf_counter() - started,
        )
        return [results[index] for index in range(len(tasks))]

    def _pick(
        self,
        live: Sequence[_Worker],
        inflight: Mapping[Any, tuple[_Worker, int]],
        last_device: Mapping[int, str],
        index: int,
    ) -> _Worker | None:
        """The worker to give ``index`` to: idle first, and never the one that
        just failed on it unless there is no alternative."""
        busy = {entry[0].conn for entry in inflight.values()}
        idle = [worker for worker in live if worker.conn not in busy]
        if not idle:
            return None
        previous = last_device.get(index)
        if previous is not None:
            fresh = [worker for worker in idle if worker.device != previous]
            if fresh:
                return fresh[0]
        return idle[0]

    def _receive(self, worker: _Worker) -> Result | TaskFailure:
        """Read one reply, turning a dead worker into a failure to retry."""
        try:
            message = worker.conn.recv()
        except (EOFError, OSError) as exc:
            worker.dead = True
            worker.process.join(timeout=SHUTDOWN_TIMEOUT_S)
            exit_code = worker.process.exitcode
            reason = (
                f"worker on {worker.device} stopped unexpectedly (exit {exit_code})"
            )
            self._log.error("%s: %s", reason, exc)
            return TaskFailure(worker.device, reason)
        if isinstance(message, TaskFailure):
            return message
        return message


# --------------------------------------------------------------------------- #
# Planning and reporting helpers
# --------------------------------------------------------------------------- #


def split_evenly(items: Sequence[Any], parts: int) -> list[list[Any]]:
    """Cut ``items`` into at most ``parts`` contiguous, size-balanced pieces.

    Contiguous rather than round-robin because the pieces are line ranges of one
    speaker: keeping neighbouring lines together keeps the text in order, which is
    what makes a worker's log readable and a line's overrun predictable.

    Returns ``[]`` for no items and a single piece when there is nothing to gain,
    so a caller never has to special-case a one-device run.
    """
    total = len(items)
    if total == 0:
        return []
    count = max(1, min(int(parts), total))
    if count == 1:
        return [list(items)]
    base, extra = divmod(total, count)
    pieces: list[list[Any]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        pieces.append(list(items[start : start + size]))
        start += size
    return pieces


def merge_tts_stats(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold several devices' counters into the one block the report carries.

    A single report is returned untouched, so a one-device run's ``report.json``
    is exactly what it was before this feature existed. With several, sums are
    sums (batches, retries, releases), the peak is the worst card - and the
    per-device list is kept beside it, because "9.3 GiB on one card, 2 GiB on the
    other" and "5.6 GiB on each" share a maximum and are very different runs.
    """
    clean = [dict(report) for report in reports if report]
    if not clean:
        return {}
    if len(clean) == 1:
        return clean[0]

    def total(key: str) -> int:
        return sum(int(report.get(key) or 0) for report in clean)

    def largest(key: str) -> float | None:
        values = [report.get(key) for report in clean if report.get(key) is not None]
        return max(values) if values else None  # type: ignore[type-var]

    def mean(key: str) -> float | None:
        values = [report[key] for report in clean if report.get(key) is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    merged: dict[str, Any] = {
        "batch_size": [report.get("batch_size") for report in clean],
        "max_decode_seconds": largest("max_decode_seconds"),
        "reference_seconds": largest("reference_seconds"),
        "seconds_per_char": mean("seconds_per_char"),
        "batches": total("batches"),
        "peak_batch_seconds": largest("peak_batch_seconds") or 0.0,
        "oom_retries": total("oom_retries"),
        "cache_releases": total("cache_releases"),
        "ceiling_shrinks": total("ceiling_shrinks"),
        "peak_reserved_bytes": largest("peak_reserved_bytes"),
        "per_device": clean,
    }
    return merged
