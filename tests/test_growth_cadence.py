"""Tests for the growth cadence.

The cadence replaces a diagnosis with a count. That is a real design choice and
these tests pin down what it commits to: a growth is owed after a hundred accepted
cycles, a rejected cycle does not count toward it, a growth already paid for is
not paid for twice, and a model with no room is told so rather than being handed
a proposal the growth code cannot apply.
"""

import json
import unittest
from pathlib import Path

import expansion

from tests.support import TempDirTestCase, make_config


class CadenceCountingTests(TempDirTestCase):
    """When a growth is owed."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            curated_dir=str(self.workdir / "curated"),
            expansion_state=str(self.workdir / "out" / "expansion.json"),
            n_layer=5,
            n_embd=256,
            growth_cadence_cycles=100,
        )
        (self.workdir / "out").mkdir(parents=True, exist_ok=True)
        (self.workdir / "curated").mkdir(parents=True, exist_ok=True)

    def accept(self, count=1):
        """Record accepted cycles the way the worker does."""
        state, due = expansion.record_accepted(self.config)
        state = dict(state)
        state["accepted_cycles"] += count - 1
        expansion.save_cadence(self.config, state)
        return state, due

    def test_ninety_nine_accepted_cycles_earn_nothing(self):
        for _ in range(99):
            _, due = self.accept()
        self.assertFalse(due, "a hundredth cycle is what earns a growth")

    def test_the_hundredth_earns_a_growth(self):
        for _ in range(99):
            self.accept()
        _state, due = self.accept()
        self.assertTrue(due)

    def test_a_growth_is_paid_for_once_and_not_twice(self):
        for _ in range(100):
            self.accept()
        expansion.mark_growth_earned(self.config, 100)
        for _ in range(99):
            _state, due = self.accept()
        self.assertFalse(due, "the growth was already paid for")

    def test_the_next_hundred_earns_the_next_growth(self):
        for _ in range(100):
            self.accept()
        expansion.mark_growth_earned(self.config, 100)
        for _ in range(99):
            self.accept()
        _state, due = self.accept()
        self.assertTrue(due, "a second growth is earned, not lost")

    def test_the_count_survives_a_restart(self):
        for _ in range(100):
            self.accept()
        # A fresh load, as a new process would see.
        state = expansion.load_cadence(self.config)
        self.assertEqual(state["accepted_cycles"], 100)

    def test_a_corrupt_counter_starts_over_rather_than_raising(self):
        expansion.cadence_state_path(self.config).write_text(
            "{not json", encoding="utf-8"
        )
        state = expansion.load_cadence(self.config)
        self.assertEqual(state["accepted_cycles"], 0)


class CadenceAvailabilityTests(TempDirTestCase):
    """Whether there is room to grow."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            curated_dir=str(self.workdir / "curated"),
            expansion_state=str(self.workdir / "out" / "expansion.json"),
            n_layer=5,
            n_embd=256,
            growth_cadence_cycles=100,
        )
        (self.workdir / "out").mkdir(parents=True, exist_ok=True)
        (self.workdir / "curated").mkdir(parents=True, exist_ok=True)

    def test_a_small_model_can_grow(self):
        available, _why = expansion.growth_available(self.config)
        self.assertTrue(available)

    def test_a_model_at_the_layer_ceiling_cannot(self):
        self.config.n_layer = expansion.MAX_LAYERS
        (self.workdir / "out" / "Jupiter.architecture.json").write_text(
            json.dumps({"config": {"n_layer": expansion.MAX_LAYERS, "n_embd": 256}}),
            encoding="utf-8",
        )
        available, why = expansion.growth_available(self.config)
        self.assertFalse(available)
        self.assertIn("layer ceiling", why)

    def test_a_model_that_would_pass_the_width_ceiling_cannot(self):
        self.config.n_embd = expansion.MAX_EMBD
        (self.workdir / "out" / "Jupiter.architecture.json").write_text(
            json.dumps({"config": {"n_layer": 5, "n_embd": expansion.MAX_EMBD}}),
            encoding="utf-8",
        )
        available, why = expansion.growth_available(self.config)
        self.assertFalse(available)
        self.assertIn("width ceiling", why)

    def test_no_proposal_is_made_when_there_is_no_room(self):
        self.config.n_layer = expansion.MAX_LAYERS
        (self.workdir / "out" / "Jupiter.architecture.json").write_text(
            json.dumps({"config": {"n_layer": expansion.MAX_LAYERS, "n_embd": 256}}),
            encoding="utf-8",
        )
        proposal, why = expansion.propose_cadence_growth(self.config, 200)
        self.assertIsNone(
            proposal, "a proposal the growth code cannot apply is worse than none"
        )
        self.assertIn("layer ceiling", why)


class CadenceProposalTests(TempDirTestCase):
    """What a cadence growth says when it is proposed."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            curated_dir=str(self.workdir / "curated"),
            expansion_state=str(self.workdir / "out" / "expansion.json"),
            n_layer=5,
            n_embd=256,
            growth_cadence_cycles=100,
        )
        (self.workdir / "out").mkdir(parents=True, exist_ok=True)
        (self.workdir / "curated").mkdir(parents=True, exist_ok=True)
        # The gate reads the real environment, so granting expansion here is
        # what a person setting JUPITER_CONTROL_GRANTS would be doing. It is
        # restored afterwards so the grant does not leak into another test.
        import os

        previous = os.environ.get("JUPITER_CONTROL_GRANTS")
        os.environ["JUPITER_CONTROL_GRANTS"] = "expansion"

        def restore():
            if previous is None:
                os.environ.pop("JUPITER_CONTROL_GRANTS", None)
            else:
                os.environ["JUPITER_CONTROL_GRANTS"] = previous

        self.addCleanup(restore)
        for _ in range(100):
            expansion.record_accepted(self.config)
            expansion.save_cadence(self.config, expansion.load_cadence(self.config))

    def test_it_grows_by_one_layer(self):
        proposal, _why = expansion.propose_cadence_growth(self.config, 100)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["from_layers"], 5)
        self.assertEqual(proposal["to_layers"], 6)

    def test_it_records_why_it_was_proposed(self):
        # The cause is the count, not a diagnosis, and a proposal that did not say
        # so would be indistinguishable from one made because a document was hard.
        proposal, _why = expansion.propose_cadence_growth(self.config, 100)
        self.assertEqual(proposal["cause"], "cadence")
        self.assertIsNone(proposal["frustration_index"])

    def test_it_still_needs_a_human(self):
        # The cadence says when to try. Whether to keep is the gate's decision,
        # so a cadence proposal must arrive unapproved like any other.
        proposal, _why = expansion.propose_cadence_growth(self.config, 100)
        self.assertFalse(proposal["approved"])

    def test_it_is_not_proposed_twice_before_being_approved(self):
        expansion.propose_cadence_growth(self.config, 100)
        again, why = expansion.propose_cadence_growth(self.config, 101)
        self.assertIsNone(again, "the first is still waiting for a human")
        self.assertIn("waiting", why)

    def test_the_baseline_is_still_recorded_at_approval(self):
        # Kept or rolled back is still decided by comparing against the loss when
        # the growth was authorised, so a cadence growth is measured the same way
        # any other is.
        proposal, _why = expansion.propose_cadence_growth(self.config, 100)
        expansion.approve_growth(self.config, holdout_loss=4.2)
        recorded = expansion.load(self.config)["capacity_growth"]
        self.assertEqual(recorded["holdout_at_approval"], 4.2)


if __name__ == "__main__":
    unittest.main()
