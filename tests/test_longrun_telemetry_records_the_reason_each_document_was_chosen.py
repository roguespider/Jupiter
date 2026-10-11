"""Telemetry records the reason each document was chosen.

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
    def test_telemetry_records_the_reason_each_document_was_chosen(self):
        # Without this a curriculum change is indistinguishable from a random
        # walk when the experiment log is read back later.
        c = self.config()
        self.curate_with(c)
        self.run_worker(c, 3)
        starts = [e for e in self.telemetry(c) if e["event"] == "worker_cycle_start"]
        self.assertTrue(starts)
        for event in starts:
            self.assertTrue(event.get("chosen_because"), event)

if __name__ == "__main__":
    unittest.main()
