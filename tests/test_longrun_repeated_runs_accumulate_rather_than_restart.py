"""Repeated runs accumulate rather than restart.

One test per file, split out of test_long_repeated_learning_tests.py so the parallel runner can give each
its own process. Every one of these builds its own corpus, model and store and
shares nothing with any other, so the test is the unit that can run beside the
others rather than after them.

Measured: one cycle of this model is about seven seconds on the CPU, most of it
the per-skill probe pass, so a three-cycle run is over twenty seconds. Nine of
these in one process is over three minutes of wall clock and one file the whole
rest of the suite waits behind.

Split by copy rather than by inheritance so that each file carries exactly one
test and the runner's one-file-per-process rule maps onto the work directly. The
body is copied verbatim: a test rewritten during a refactor is a test that has
changed without being reviewed.
"""

import json
import unittest
from pathlib import Path

from tests.support_long_run import LongRunTestCase


class RepeatedLearningTests(LongRunTestCase):
    def test_repeated_runs_accumulate_rather_than_restart(self):
        # The headline property. Two separate runs, and the second must begin
        # where the first finished rather than at random initialization.
        c = self.config()
        self.curate_with(c)
        first = self.run_worker(c, 2)
        second = self.run_worker(self.config(), 2)
        self.assertIn("random initialization", first)
        self.assertIn("Resuming from", second)

if __name__ == "__main__":
    unittest.main()
