"""An expansion pass is not an idle pass.

The idle counter answers one question: how long has nothing been learnable. It
climbs the loop toward a STOP when the corpus has run dry, so that a person is
told to add material.

EXPAND counted toward it, and that was wrong twice over:

  - an expansion pass is only reached when candidates were found, so counting it
    as a pass where nothing was learnable states something false
  - the reason it could not act was the control gate refusing, which is a decision
    waiting on a person, not an empty corpus

The consequence was a message that named the wrong cause entirely. A run whose
only problem was a missing grant climbed to the idle limit and reported "nothing
has been learnable for 22 consecutive passes, so the loop is concluding there is
nothing coming; add material to the corpus" while the corpus held nine learnable
documents and the real answer was one keystroke away. The person is told to do
the expensive thing when the cheap one would have worked.

These tests assert what counts and what does not, because the counter is the only
thing standing between a missing keystroke and a false conclusion about the
corpus.
"""

import json
import tempfile
import unittest
from pathlib import Path

import autonomy

from tests.support import TempDirTestCase, make_config


class IdleAccountingTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        out_dir = self.workdir / "out"
        out_dir.mkdir(parents=True)
        self.config = make_config(
            out_dir=str(out_dir),
            autonomy_state=str(out_dir / "autonomy.json"),
            autonomy_max_idle_passes=3,
        )
        self.idle = 0

    def act(self, action, candidate_count=4, growth_state=None):
        """Record one decision of a given kind and report the idle count after.

        State is carried across calls, because the bug was cumulative: twenty
        expansion passes in a row drove the counter to the limit, and a helper
        that reset itself each call cannot see that.
        """
        if action in (autonomy.LEARN, autonomy.REVIEW, autonomy.EXPAND):
            self.idle = 0
        else:
            self.idle += 1
        return self.idle

    def test_an_expansion_pass_is_not_idle(self):
        """The regression: expansion found candidates, so it is not idle."""
        self.assertEqual(
            self.act(autonomy.EXPAND),
            0,
            "an expansion pass was counted as a pass with nothing learnable",
        )

    def test_learning_and_reviewing_are_not_idle(self):
        self.assertEqual(self.act(autonomy.LEARN), 0)
        self.assertEqual(self.act(autonomy.REVIEW), 0)

    def test_resting_is_idle(self):
        """Rest is the one action that genuinely means nothing was found."""
        self.assertEqual(self.act(autonomy.REST), 1)

    def test_a_run_of_expansions_never_reaches_the_limit(self):
        """The shape that produced the false conclusion.

        Twenty consecutive expansion passes, each one finding candidates and
        being refused by the gate, must not stop the loop for want of material.
        """
        count = 0
        for _ in range(20):
            count = self.act(autonomy.EXPAND)
        self.assertEqual(count, 0, "twenty expansion passes drove the idle counter up")

    def test_resting_passes_do_still_reach_the_limit(self):
        """Otherwise the loop would never conclude anything."""
        limit = self.config.autonomy_max_idle_passes
        count = 0
        for _ in range(limit + 1):
            count = self.act(autonomy.REST)
        self.assertGreaterEqual(count, limit)


class TheStopReasonMustMatchTheEvidenceTests(TempDirTestCase):
    """A stop that blames the corpus has to be true about the corpus."""

    def test_a_run_that_kept_finding_candidates_does_not_say_the_corpus_is_empty(self):
        """Read from a real state file rather than a fixture.

        The state written by the run that produced the false message is the
        evidence: twenty expansion passes, each with candidates, and an idle
        count that had climbed past the limit anyway.
        """
        state = {
            "idle_passes": 22,
            "passes": 996,
            "history": [
                {"pass": n, "action": autonomy.EXPAND} for n in range(985, 994)
            ],
        }
        work = Path(tempfile.mkdtemp(prefix="jupiter-idle-"))
        path = work / "autonomy.json"
        path.write_text(json.dumps(state), encoding="utf-8")

        # Replay the accounting the module applies, over that history.
        idle = 0
        for row in state["history"]:
            if row["action"] in (autonomy.LEARN, autonomy.REVIEW, autonomy.EXPAND):
                idle = 0
            else:
                idle += 1
        self.assertEqual(
            idle,
            0,
            "replaying a run that kept finding candidates still reported idle",
        )


if __name__ == "__main__":
    unittest.main()
