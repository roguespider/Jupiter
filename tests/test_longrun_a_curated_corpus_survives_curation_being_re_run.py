"""A curated corpus survives curation being re run.

One test per file, split out of test_long_resume_across_runs_tests.py so the parallel runner can give each
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


class ResumeAcrossRunsTests(LongRunTestCase):
    def test_a_curated_corpus_survives_curation_being_re_run(self):
        # Curation owns data/curated and quarantines anything it would not
        # produce. Re-running it must be a no-op on a corpus it already accepted.
        c = self.config()
        first = self.curate_with(c)
        before = sorted(p.name for p in Path(c.curated_dir).glob("*.txt"))
        second = self.curate_with(self.config())
        after = sorted(p.name for p in Path(c.curated_dir).glob("*.txt"))
        self.assertEqual(before, after)
        self.assertEqual(first["quarantined"], 0)
        self.assertEqual(second["quarantined"], 0)

if __name__ == "__main__":
    unittest.main()
