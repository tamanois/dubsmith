"""The run log: stderr always, a file when one was asked for.

A dub can run for half an hour, so the file is the only durable account of what
the job did. What matters here is that it is really written, that a path which
cannot be opened does not take the run down, and that a second run adds to the
first instead of erasing it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pytest

from pipeline.logging_utils import (
    DEFAULT_MAX_LOG_BYTES,
    job_logger,
    setup_logging,
    timed,
)


@pytest.fixture(autouse=True)
def restore_logging():
    """Leave the root logger as it was found.

    ``setup_logging`` removes every handler by design, so without this a test that
    calls it would switch off pytest's own capture and the suites after it.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def say(message: str) -> None:
    """Emit a record the way the pipeline does: through a job-bound adapter."""
    job_logger("pipeline", "job-1").info(message)


def contents(path: Path) -> str:
    """The log file, read back the way a person would."""
    return path.read_text(encoding="utf-8")


def line_with(path: Path, needle: str) -> str:
    """The one line mentioning ``needle``."""
    matches = [line for line in contents(path).splitlines() if needle in line]
    assert len(matches) == 1, f"expected exactly one line with {needle!r}: {matches}"
    return matches[0]


class TestConsoleDestination:
    """stderr is always configured; the file is the addition."""

    def test_no_path_means_no_file(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)

        assert setup_logging() is None
        assert list(tmp_path.iterdir()) == []

    def test_lines_carry_the_job_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        setup_logging()
        say("hello")

        assert "[job-1] hello" in capsys.readouterr().err

    def test_setting_up_twice_does_not_double_every_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        setup_logging()
        setup_logging()
        say("once")

        assert capsys.readouterr().err.count("once") == 1

    def test_the_console_says_where_the_file_went(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "run.log"

        assert setup_logging(log_file=path) == path
        assert "Run started: appending to" in capsys.readouterr().err


class TestLogFile:
    def test_the_file_and_its_parents_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "output" / "talk" / "run.log"

        assert setup_logging(log_file=path) == path
        say("written")

        assert "written" in contents(path)

    def test_the_file_is_stamped_and_the_console_is_not(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "run.log"
        setup_logging(log_file=path)
        say("stamped")

        line = line_with(path, "[job-1] stamped")

        assert line.endswith("INFO    [job-1] stamped")
        assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", line)
        # Same record, two shapes: the console keeps the live view unchanged.
        assert "INFO    [job-1] stamped" in capsys.readouterr().err

    def test_every_run_keeps_its_own_lines(self, tmp_path: Path) -> None:
        # Re-running the same input is the normal way to work on this project -
        # every option tweak rebuilds the job - so the later run must not erase
        # the earlier one, and each must still be findable on its own.
        path = tmp_path / "run.log"
        setup_logging(log_file=path)
        say("first run")
        setup_logging(log_file=path)
        say("second run")

        text = contents(path)

        assert text.count("Run started: appending to") == 2
        assert text.index("first run") < text.index("second run")

    def test_json_output_writes_one_readable_object_per_line(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.jsonl"
        setup_logging(json_output=True, log_file=path)
        say("parsed")

        record = json.loads(line_with(path, '"message": "parsed"'))

        assert record["message"] == "parsed"
        assert record["job_id"] == "job-1"
        assert record["level"] == "INFO"
        assert "time" in record

    def test_the_level_applies_to_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        setup_logging(level="WARNING", log_file=path)
        say("too quiet")
        logging.getLogger("pipeline").warning("loud enough")

        text = contents(path)

        assert "loud enough" in text
        assert "too quiet" not in text

    def test_a_traceback_reaches_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        setup_logging(log_file=path)

        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("pipeline").exception("failed")

        text = contents(path)

        assert "failed" in text
        assert "ValueError: boom" in text

    def test_an_unusable_path_warns_instead_of_raising(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A directory is the cheapest stand-in for every "cannot open this" case
        # (permissions, disk full); the point is that the run survives all of them.
        assert setup_logging(log_file=tmp_path) is None
        say("still logged")

        err = capsys.readouterr().err

        assert "Could not write the log file" in err
        assert "still logged" in err


class TestTimed:
    """The one helper every timed step uses, so it is worth pinning down."""

    def test_it_logs_the_duration_and_the_fields(self, caplog) -> None:
        logger = logging.getLogger("pipeline")

        with caplog.at_level(logging.INFO, logger="pipeline"):
            with timed(logger, "Did work", lines=3):
                pass

        record = caplog.records[-1]
        assert record.getMessage().startswith("Did work in ")
        assert record.seconds >= 0.0
        assert record.lines == 3

    def test_it_accepts_fields_discovered_inside_the_block(self, caplog) -> None:
        logger = logging.getLogger("pipeline")

        with caplog.at_level(logging.INFO, logger="pipeline"):
            with timed(logger, "Did work") as context:
                context["lines"] = 7

        assert caplog.records[-1].lines == 7

    def test_it_logs_a_failure_and_re_raises(self, caplog) -> None:
        logger = logging.getLogger("pipeline")

        with caplog.at_level(logging.INFO, logger="pipeline"):
            with pytest.raises(RuntimeError, match="boom"):
                with timed(logger, "Doomed"):
                    raise RuntimeError("boom")

        record = caplog.records[-1]
        assert record.getMessage().startswith("Doomed failed after ")
        assert record.seconds >= 0.0

    def test_it_logs_at_the_level_it_is_given(self, caplog) -> None:
        logger = logging.getLogger("pipeline")

        with caplog.at_level(logging.DEBUG, logger="pipeline"):
            with timed(logger, "Quiet step", level=logging.DEBUG):
                pass

        assert caplog.records[-1].levelno == logging.DEBUG

    def test_verbose_does_not_enable_dependency_debug(self) -> None:
        setup_logging(level="DEBUG")

        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("modelscope").level == logging.WARNING


class TestRolling:
    """Keeping every run cannot mean growing one file forever."""

    #: Last-write time of the filled file, so the archive name is predictable.
    STAMP = datetime(2026, 9, 29, 7, 3, 8)
    ARCHIVE = "run-20260929-070308.log"

    def fill(self, path: Path, size: int) -> str:
        """Write a log file of ``size`` bytes and backdate its last write."""
        payload = "x" * size
        path.write_text(payload, encoding="utf-8")
        when = self.STAMP.timestamp()
        os.utime(path, (when, when))
        return payload

    def test_a_small_log_is_appended_to(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        path.write_text("old run\n", encoding="utf-8")

        assert setup_logging(log_file=path, max_bytes=1024) == path
        say("new run")

        assert "old run" in contents(path)
        assert list(tmp_path.glob("run-*")) == []

    def test_a_full_log_moves_aside_under_its_own_timestamp(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.log"
        payload = self.fill(path, 2048)

        assert setup_logging(log_file=path, max_bytes=1024) == path
        say("new run")

        archives = list(tmp_path.glob("run-*"))
        assert [archive.name for archive in archives] == [self.ARCHIVE]
        # The archive holds the whole old file; the live one starts empty.
        assert contents(archives[0]) == payload
        text = contents(path)
        assert "new run" in text
        assert payload not in text

    def test_the_roll_is_announced_with_the_archive_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "run.log"
        self.fill(path, 2048)
        setup_logging(log_file=path, max_bytes=1024)

        err = capsys.readouterr().err

        assert self.ARCHIVE in err
        assert "2.0 KiB" in err
        # In the new file too, so the archive can be found from either one.
        assert self.ARCHIVE in contents(path)

    def test_an_archive_name_already_in_use_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.log"
        payload = self.fill(path, 2048)
        taken = tmp_path / self.ARCHIVE
        taken.write_text("earlier archive", encoding="utf-8")

        setup_logging(log_file=path, max_bytes=1024)

        assert contents(taken) == "earlier archive"
        assert contents(tmp_path / "run-20260929-070308-1.log") == payload

    def test_a_suffixless_path_still_gets_a_log_name(self, tmp_path: Path) -> None:
        path = tmp_path / "nightly"
        self.fill(path, 2048)

        setup_logging(log_file=path, max_bytes=1024)

        assert [p.name for p in tmp_path.glob("nightly-*")] == [
            "nightly-20260929-070308.log"
        ]

    def test_zero_never_rolls(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        payload = self.fill(path, 2048)

        setup_logging(log_file=path, max_bytes=0)
        say("new run")

        assert payload in contents(path)
        assert list(tmp_path.glob("run-*")) == []

    def test_the_default_cap_is_the_documented_one(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        self.fill(path, DEFAULT_MAX_LOG_BYTES + 1)

        setup_logging(log_file=path)

        assert len(list(tmp_path.glob("run-*"))) == 1

    def test_a_directory_named_like_the_log_is_left_alone(self, tmp_path: Path) -> None:
        # Rolling must only ever move a regular file; a directory is somebody
        # else's data, and the log then simply fails to open as before.
        path = tmp_path / "run.log"
        path.mkdir()

        assert setup_logging(log_file=path, max_bytes=1) is None
        assert path.is_dir()
