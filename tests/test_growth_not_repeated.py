"""A growth that is already applied must not be chosen again.

The loop kept printing "the model is grown" on every pass and the model never
changed. Two things combined:

  decide() checked whether a growth was approved and, once it had been approved
  AND reviewed with no verdict either way, fell through to the capacity branch,
  which chose EXPAND again. approve_growth() then returned None, because the
  proposal had already been applied and nothing was waiting. act() reported that
  as `applied: False, denied: False`, which reads as a success rather than as
  nothing happening, and the worker printed the decision's reason rather than the
  outcome.

So a person watched EXPAND repeat indefinitely on a growth that had already been
applied, and was told each time that more material could not answer the shortage
because the model was being grown. It was not being grown. It had been, once, and
the approval that let it happen had already been consumed.

Three states are asserted here, because they need three different responses and
telling them apart is the fix:

  approved, unreviewed   -> LEARN, because the next cycles are the measurement
  approved and kept      -> LEARN, on the shape that earned it
  approved, unjudged     -> LEARN, because there is nothing left to approve

and approving when nothing is pending must say so rather than report a success.
"""

import os
import unittest

import autonomy
import control
import expansion

from tests.support import TempDirTestCase, make_config


class AlreadyAppliedGrowthTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        out_dir = self.workdir / "out"
        out_dir.mkdir(parents=True)
        self.config = make_config(
            out_dir=str(out_dir),
            expansion_state=str(out_dir / "expansion.json"),
        )
        self.shortage = {
            "candidate_count": 11,
            "capacity_shortages": 65,
            "idle_passes": 0,
        }

    def write_growth(self, **fields):
        proposal = {
            "source": "a.txt",
            "cause": "insufficient model capacity",
            "from_layers": 8,
            "to_layers": 9,
            "from_embd": 448,
            "to_embd": 448,
            "approved": True,
            "reviewed_at": 1.0,
            "kept": None,
        }
        proposal.update(fields)
        state = expansion.load(self.config)
        state["capacity_growth"] = proposal
        expansion.save(self.config, state)
        return proposal

    def test_an_unjudged_growth_does_not_choose_expand_again(self):
        """The regression: this returned EXPAND on every pass."""
        growth = self.write_growth(approved=True, reviewed_at=1.0, kept=None)
        decision = autonomy.decide(self.config, self.shortage, growth_state=growth)
        self.assertNotEqual(
            decision.action,
            autonomy.EXPAND,
            "an already-applied growth was chosen again: " + decision.reason,
        )
        self.assertEqual(decision.action, autonomy.LEARN)

    def test_the_reason_says_nothing_is_left_to_approve(self):
        growth = self.write_growth(approved=True, reviewed_at=1.0, kept=None)
        decision = autonomy.decide(self.config, self.shortage, growth_state=growth)
        self.assertIn(
            "nothing is left to approve",
            decision.reason,
            f"the reason does not say the growth was already applied: {decision.reason}",
        )

    def test_an_approved_unmeasured_growth_still_learns(self):
        growth = self.write_growth(approved=True, reviewed_at=None)
        decision = autonomy.decide(self.config, self.shortage, growth_state=growth)
        self.assertEqual(decision.action, autonomy.LEARN)
        self.assertIn("not been measured", decision.reason)

    def test_a_kept_growth_learns_on_the_shape_that_earned_it(self):
        growth = self.write_growth(approved=True, reviewed_at=1.0, kept=True)
        decision = autonomy.decide(self.config, self.shortage, growth_state=growth)
        self.assertEqual(decision.action, autonomy.LEARN)
        self.assertIn("kept", decision.reason)

    def test_a_reverted_growth_learns_rather_than_growing_again(self):
        growth = self.write_growth(approved=True, reviewed_at=1.0, kept=False)
        decision = autonomy.decide(self.config, self.shortage, growth_state=growth)
        self.assertEqual(decision.action, autonomy.LEARN)

    def test_approving_nothing_pending_is_reported_as_nothing_pending(self):
        """act() reported this as a success, which is how it looked like growth."""
        os.environ[control.GRANT_VARIABLE] = "expansion"
        self.write_growth(approved=True, reviewed_at=1.0, kept=None)
        decision = autonomy.Decision(
            autonomy.EXPAND, "chosen for the test", evidence={}
        )
        outcome = autonomy.act(self.config, decision)
        self.assertTrue(
            outcome.get("nothing_pending"),
            f"approving nothing was not reported as such: {outcome}",
        )
        self.assertFalse(outcome.get("applied"))
        self.assertIn(
            "no growth waiting",
            str(outcome.get("reason", "")),
        )


class GrowthDecisionsStillWorkTests(TempDirTestCase):
    """The fix must not stop a genuine growth from being proposed."""

    def setUp(self):
        super().setUp()
        out_dir = self.workdir / "out"
        out_dir.mkdir(parents=True)
        self.config = make_config(
            out_dir=str(out_dir),
            expansion_state=str(out_dir / "expansion.json"),
        )
        self.saved_environment = dict(os.environ)
        self.addCleanup(
            lambda: (
                os.environ.clear(),
                os.environ.update(self.saved_environment),
            )
        )

    def test_a_shortage_with_no_growth_on_file_still_expands(self):
        """The fix must not stop a genuine growth from being proposed.

        With the expansion grant held and nothing on file, a real shortage has to
        reach EXPAND. Without the grant it deliberately returns LEARN, which is
        the other case and is asserted separately.
        """
        os.environ[control.GRANT_VARIABLE] = "expansion"
        state = expansion.load(self.config)
        state.pop("capacity_growth", None)
        expansion.save(self.config, state)
        decision = autonomy.decide(
            self.config,
            {
                "candidate_count": 11,
                "capacity_shortages": 65,
                "idle_passes": 0,
                # decide() reads the grants the observation recorded rather than
                # reading the environment itself, which is what keeps a test from
                # depending on a developer's own shell.
                "granted": ["expansion"],
            },
            growth_state=None,
        )
        self.assertEqual(
            decision.action,
            autonomy.EXPAND,
            "a real shortage no longer produces a growth: " + decision.reason,
        )

    def test_a_shortage_without_the_grant_keeps_learning(self):
        """The refusal path is unchanged and must stay a refusal."""
        os.environ.pop(control.GRANT_VARIABLE, None)
        state = expansion.load(self.config)
        state.pop("capacity_growth", None)
        expansion.save(self.config, state)
        decision = autonomy.decide(
            self.config,
            {
                "candidate_count": 11,
                "capacity_shortages": 65,
                "idle_passes": 0,
                "granted": [],
            },
            growth_state=None,
        )
        self.assertEqual(decision.action, autonomy.LEARN)
        self.assertIn("not granted", decision.reason)


if __name__ == "__main__":
    unittest.main()
