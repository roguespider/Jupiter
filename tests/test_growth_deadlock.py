"""An approved growth must not block every growth that comes after it.

Two bugs met here and neither had a test.

The first was a deadlock. resolve_growth() is what judges whether a growth
earned its place, and the worker only called it when a run produced a holdout
loss. A run without one approved a growth, could not judge it, and left it in the
state file marked approved and unreviewed. Every later proposal then answered
"the approved growth is still being measured" and returned without proposing
anything.

Nothing was measuring it. There was no pending measurement and no process waiting
to make one; there was a record waiting for a verdict that nothing was going to
arrive with. So one run without a holdout stopped growth permanently, and the
symptom was a model that sat at one size no matter how long it learned.

The second was reachability. prompt_for_growth() returned early when the expansion
grant was absent, and the only way to have that grant was an environment variable
that run_jupiter.bat set and learning_worker.py did not. Running the script
directly therefore could never ask, and the run printed that a growth was waiting
for a human with no way to become that human.
"""

import os
import unittest

import control
import expansion
import learning_worker

from tests.support import TempDirTestCase, make_config


class UnreviewedGrowthDoesNotBlockTests(TempDirTestCase):
    """The deadlock: an approved, unjudged growth must not freeze growth."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            expansion_state=str(self.workdir / "out" / "expansion.json"),
            growth_cadence=100,
        )
        os.environ.pop(control.GRANT_VARIABLE, None)

    def _propose(self, cycle):
        """Propose a cadence growth, having cleared the waiting-proposal gate.

        The cadence records that it proposed and waits until that proposal is
        consumed. That is right: replacing a proposal a person is looking at would
        be worse than waiting. It is cleared here so each case starts from a state
        where a proposal is genuinely allowed, which is what the tests are about.
        """
        state = expansion.load(self.config)
        state.pop("capacity_gadence", None)
        expansion.save(self.config, state)
        return expansion.propose_cadence_growth(
            self.config, cycle=cycle, source="a.txt"
        )

    def test_an_unreviewed_growth_is_reported_not_left_open(self):
        proposal, reason = self._propose(10)
        self.assertIsNotNone(proposal, reason)
        # Approved but never judged: the state a run without a holdout leaves.
        state = expansion.load(self.config)
        state["capacity_growth"]["approved"] = True
        expansion.save(self.config, state)

        # Asking again must not report that it is still being measured.
        proposal, reason = self._propose(20)
        self.assertIsNotNone(
            proposal,
            "an unreviewed growth blocked the next one: " + str(reason),
        )

    def test_the_blocked_growth_is_replaced_not_left_open(self):
        """The unreviewed growth is closed out and a fresh one takes its place.

        The record is replaced rather than amended, so what is asserted is that
        the new proposal exists and is not the old one still marked approved and
        unreviewed. Asserting on the closed-out record would be asserting that
        the proposal was not made.
        """
        self._propose(10)
        state = expansion.load(self.config)
        first = dict(state["capacity_growth"])
        state["capacity_growth"]["approved"] = True
        expansion.save(self.config, state)

        proposal, _reason = self._propose(20)
        self.assertIsNotNone(proposal)

        recorded = expansion.load(self.config)["capacity_growth"]
        self.assertFalse(
            recorded.get("approved"),
            "the replacement proposal is already approved, so nothing can be asked",
        )
        self.assertEqual(
            recorded.get("proposed_at_cycle"),
            20,
            "the proposal on file is the blocked one, not a new one",
        )
        self.assertEqual(first.get("proposed_at_cycle"), 10)

    def test_a_second_attempt_also_is_not_blocked(self):
        """The same question twice must give the same answer.

        A gate that clears a deadlock once and then re-enters it is not a fix.
        """
        self._propose(10)
        state = expansion.load(self.config)
        state["capacity_growth"]["approved"] = True
        expansion.save(self.config, state)

        first, _ = self._propose(20)
        self.assertIsNotNone(first)

        state = expansion.load(self.config)
        state["capacity_growth"]["approved"] = True
        expansion.save(self.config, state)
        second, _ = self._propose(30)
        self.assertIsNotNone(second, "the deadlock came back on the next attempt")
        self.assertIsNotNone(second, "the deadlock came back on the next attempt")

    def test_a_judged_growth_still_blocks_nothing_either(self):
        """The normal path: reviewed growth is replaced rather than blocking."""
        self._propose(10)
        state = expansion.load(self.config)
        state["capacity_growth"]["approved"] = True
        expansion.save(self.config, state)
        expansion.resolve_growth(self.config, 5.0)

        proposal, reason = self._propose(20)
        self.assertIsNotNone(proposal, reason)

    def test_a_waiting_proposal_is_still_waiting(self):
        """Clearing a deadlock must not start replacing proposals under a reader.

        The cadence gate that keeps one proposal on file until it is consumed is
        the thing this fix must not have broken.
        """
        self._propose(10)
        state = expansion.load(self.config)
        state["capacity_gadence"] = {"proposed_at_cycle": 10, "consumed": False}
        expansion.save(self.config, state)

        proposal, reason = expansion.propose_cadence_growth(
            self.config, cycle=20, source="a.txt"
        )
        self.assertIsNone(proposal)
        self.assertIn("already been proposed", str(reason))


class PromptIsReachableTests(TempDirTestCase):
    """The prompt must be reachable from a plain `python learning_worker.py`."""

    def setUp(self):
        super().setUp()
        self.saved_environment = dict(os.environ)
        self.addCleanup(
            lambda: (
                os.environ.clear(),
                os.environ.update(self.saved_environment),
            )
        )
        self.saved_stdin = learning_worker.sys.stdin
        self.saved_refused = set(learning_worker._refused_domains)

        class Terminal:
            def isatty(self):
                return True

        learning_worker.sys.stdin = Terminal()

        def restore():
            learning_worker.sys.stdin = self.saved_stdin
            learning_worker._refused_domains = set(self.saved_refused)

        self.addCleanup(restore)
        os.environ.pop("JUPITER_NO_PROMPT", None)
        os.environ.pop(control.GRANT_VARIABLE, None)

    def test_a_terminal_is_enough_to_ask(self):
        """Without the grant, a person at the keyboard still has to be askable.

        This was hard coded to require expansion, which meant the only run that
        could ask was the one that already did not need to.
        """
        self.assertTrue(
            learning_worker._interactive(),
            "a terminal with no grant must still be able to ask",
        )

    def test_the_domain_is_only_required_when_named(self):
        self.assertFalse(
            learning_worker._interactive("expansion"),
            "a domain that has not been granted is correctly reported as absent",
        )
        os.environ[control.GRANT_VARIABLE] = "expansion"
        self.assertTrue(learning_worker._interactive("expansion"))

    def test_no_prompt_still_silences_everything(self):
        os.environ["JUPITER_NO_PROMPT"] = "1"
        self.assertFalse(learning_worker._interactive())

    def test_a_pipe_is_not_a_person(self):
        class Pipe:
            def isatty(self):
                return False

        learning_worker.sys.stdin = Pipe()
        self.assertFalse(learning_worker._interactive())


if __name__ == "__main__":
    unittest.main()
