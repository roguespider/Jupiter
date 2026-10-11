"""Tests for the single-writer lock (Architecture 38).

The one that matters: a worker that is killed leaves its lock behind, and the
next run must not be blocked by a file nobody will ever delete. That happened:
a killed background worker left a lock, and every subsequent run failed with
FileExistsError while the error message blamed a process that was already dead.
"""

import json
import os

import lock as lock_module
from lock import LockHeld, acquire, read_lock, release

from tests.support import TempDirTestCase, make_config


def dead_pid():
    """A process id that is certainly not running."""
    # Spawning and reaping a process is the only portable way to be sure.
    import subprocess

    proc = subprocess.Popen([__import__("sys").executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class LockTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config()
        self.path = self.workdir / "worker.lock"

    def assert_lock_held(self, path):
        """Assert the lock at `path` is readable and names this process.

        read_lock returns None for a lock that is free, unreadable, or not a
        record, so it is narrowed once here rather than subscripted at each call
        site. Subscripting it directly worked until it did not, and the failure
        that reached was "None is not subscriptable" from a test about lock
        takeover, which says nothing about the lock.
        """
        record = read_lock(path)
        if record is None:
            self.fail(f"the lock at {path} was not readable after acquiring it")
        self.assertEqual(record["pid"], os.getpid())
        return record

    def test_acquiring_then_releasing_works(self):
        taken = acquire(self.path, self.config)
        self.assertTrue(taken.exists())
        record = read_lock(taken)
        # read_lock returns None for a free or unreadable lock, so the record is
        # narrowed here rather than subscripted on the assumption it exists. A
        # lock that could not be read is a different failure from a lock that
        # does not exist, and both are worth naming rather than crashing on.
        if record is None:
            self.fail(f"the lock at {taken} could not be read back")
        self.assertEqual(record["pid"], os.getpid())
        self.assertTrue(release(self.path))
        self.assertFalse(self.path.exists())

    def test_a_live_owner_is_refused(self):
        acquire(self.path, self.config)
        with self.assertRaises(LockHeld) as caught:
            acquire(self.path, self.config)
        self.assertIn("already owns", str(caught.exception))

    def test_releasing_a_lock_this_process_does_not_own_leaves_it(self):
        acquire(self.path, self.config)
        # Pretend someone else owns it.
        self.path.write_text(
            json.dumps({"pid": dead_pid(), "started": "2026-01-01 00:00:00"}),
            encoding="utf-8",
        )
        self.assertFalse(release(self.path))
        self.assertTrue(self.path.exists())

    def test_a_lock_left_by_a_dead_process_is_taken_over(self):
        # This is the case that blocked a real run. The owner is gone, so
        # keeping the file would block every future run until a person deleted
        # it by hand, which is what a lock exists to prevent.
        gone = dead_pid()
        self.path.write_text(
            json.dumps({"pid": gone, "started": "2026-01-01 00:00:00"}),
            encoding="utf-8",
        )
        taken = acquire(self.path, self.config)
        self.assert_lock_held(taken)

    def test_an_unreadable_lock_is_taken_over(self):
        self.path.write_text("{not json", encoding="utf-8")
        taken = acquire(self.path, self.config)
        self.assert_lock_held(taken)

    def test_an_empty_lock_is_taken_over(self):
        self.path.write_text("", encoding="utf-8")
        self.assertIsNone(read_lock(self.path))
        taken = acquire(self.path, self.config)
        self.assert_lock_held(taken)

    def test_acquiring_creates_the_parent_directory(self):
        nested = self.workdir / "a" / "b" / "worker.lock"
        taken = acquire(nested, self.config)
        self.assertTrue(taken.exists())

    def test_reading_a_missing_lock_is_none(self):
        self.assertIsNone(read_lock(self.workdir / "absent.lock"))

    def test_process_alive_recognises_this_process(self):
        self.assertTrue(lock_module.process_alive(os.getpid()))

    def test_process_alive_rejects_impossible_ids(self):
        self.assertFalse(lock_module.process_alive(-1))
        self.assertFalse(lock_module.process_alive(0))
