"""A rejected cycle restores the live model.

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
    def test_a_rejected_cycle_restores_the_live_model(self):
        # Architecture 32 wants a rejected update to be reversible. It used to be
        # left in place until the end of the run, so every following cycle
        # measured itself against damage that had already been dealt with.
        c = self.config()
        self.curate_with(c)
        self.run_worker(c, 4)
        events = self.telemetry(c)
        rejects = [r for r in self.experiments(c) if r["decision"] == "REJECT"]
        for event in events:
            if event["event"] != "worker_rejected_restore":
                continue
            self.assertTrue(
                event.get("restored_to"),
                "a restore must say what it restored to",
            )
            break
        # Either no cycle was rejected, or the restore was recorded for each.
        restores = [e for e in events if e["event"] == "worker_rejected_restore"]
        self.assertEqual(
            len(restores),
            len(rejects),
            "every REJECT must restore the live model, and say so",
        )

if __name__ == "__main__":
    unittest.main()
