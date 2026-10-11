"""The run terminates and reports why.

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
    def test_the_run_terminates_and_reports_why(self):
        c = self.config()
        self.curate_with(c)
        self.run_worker(c, 2)
        state = json.loads(Path(c.worker_state).read_text(encoding="utf-8"))
        # The statuses decide_stop can return, plus the two the loop sets itself
        # when a cycle times out or destabilises. Anything other than "running"
        # means the run ended on a rule rather than falling off the end.
        self.assertNotEqual(
            state.get("status"),
            "running",
            "a run that hit its cycle budget must not still claim to be running",
        )
        self.assertTrue(
            state.get("reason"),
            "a finished run must say which rule ended it",
        )

if __name__ == "__main__":
    unittest.main()
