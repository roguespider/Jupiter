"""Several cycles complete and leave a record each.

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
    def test_several_cycles_complete_and_leave_a_record_each(self):
        c = self.config()
        self.curate_with(c)
        self.run_worker(c, 3)
        rows = self.experiments(c)
        self.assertGreaterEqual(
            len(rows),
            1,
            "a multi-cycle run must record at least one experiment",
        )
        for row in rows:
            self.assertIn("decision", row)
            self.assertIn("before_metrics", row)
            self.assertIn("after_metrics", row)

if __name__ == "__main__":
    unittest.main()
