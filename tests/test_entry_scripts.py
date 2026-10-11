"""Tests for the two files the entry point reaches through a mode.

watch_training.py follows the telemetry stream and never returns, and
modal_train.py bills a remote GPU, so neither can sit in the default chain.
Both are still reached from run_jupiter.bat, so both need to behave correctly
when the chain does use them: a bounded summary, and an availability check that
cannot fail the chain on a machine without the optional dependency.
"""

import subprocess
import sys
import unittest
from pathlib import Path

import modal_train
import watch_training
from telemetry import Telemetry

from tests.support import TempDirTestCase

PROJECT = Path(__file__).resolve().parent.parent


class TelemetrySummaryTests(TempDirTestCase):
    """Architecture 35: the run has to be observable while it happens."""

    def test_summary_returns_the_last_events(self):
        path = self.workdir / "telemetry.jsonl"
        telemetry = Telemetry(path)
        for index in range(4):
            telemetry.write("micro_update", step=index)
        self.assertEqual(watch_training.summary(path, limit=2), 2)

    def test_summary_of_a_missing_file_is_not_an_error(self):
        self.assertEqual(watch_training.summary(self.workdir / "absent.jsonl"), 0)

    def test_reading_appended_returns_only_what_is_new(self):
        path = self.workdir / "telemetry.jsonl"
        Telemetry(path).write("first", value=1)
        chunk, offset = watch_training.read_appended(path, 0)
        self.assertIn("first", chunk)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"event": "second"}\n')
        later, moved = watch_training.read_appended(path, offset)
        self.assertIn("second", later)
        self.assertNotIn("first", later)
        self.assertGreater(moved, offset)

    def test_reading_appended_from_a_missing_file_is_not_an_error(self):
        chunk, offset = watch_training.read_appended(self.workdir / "absent.jsonl", 0)
        self.assertEqual(chunk, "")
        self.assertEqual(offset, 0)


class RemoteAvailabilityTests(unittest.TestCase):
    def test_availability_always_answers(self):
        available, reason = modal_train.availability()
        self.assertIsInstance(available, bool)
        self.assertTrue(reason.strip())
        self.assertIn("modal", reason)

    def test_check_exits_cleanly_whether_or_not_modal_is_installed(self):
        # The default chain calls this, so a non-zero exit would fail a run that
        # was otherwise fine. modal being absent is a normal state, not a fault.
        result = subprocess.run(
            [sys.executable, "modal_train.py", "--check"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[remote]", result.stdout)


class SummaryCommandTests(TempDirTestCase):
    def test_summary_flag_exits_cleanly(self):
        path = self.workdir / "telemetry.jsonl"
        Telemetry(path).write("cycle_start", learning_cycle=1)
        result = subprocess.run(
            [
                sys.executable,
                "watch_training.py",
                "--summary",
                "--path",
                str(path),
                "--limit",
                "3",
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cycle_start", result.stdout)
