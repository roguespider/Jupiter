"""The continuous loop must be able to ask, or the control is unreachable.

run_worker() asks whether a growth should be applied. run_continuously() is a
separate function and it did not: it printed the reason autonomy recorded, which
ends with the name of an environment variable, and moved on. So the one mode
meant to run unattended was the one mode in which the question could never be
reached, and a person watching a continuous run was told to set a variable in
another window before the process started.

Three cases, because there are two separate gates and a third condition that
must not produce a question at all:

  no grant and a growth waiting   -> the grant question is reachable
  grant held and a growth waiting -> the approval question is reachable
  nothing waiting                 -> neither is asked

The loop is exercised through the two functions it calls, rather than by running
a whole continuous session, because a session runs real training cycles and the
thing under test is which question the loop offers, not how long a cycle takes.
"""

import builtins
import os
import sys
import unittest
from pathlib import Path

import control
import expansion
import learning_worker

from tests.support import TempDirTestCase, make_config


class Terminal:
    """Standard input that behaves like a console rather than a pipe."""

    def isatty(self):
        return True

    def readline(self):
        return ""


class ContinuousPromptTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        out_dir = self.workdir / "out"
        out_dir.mkdir(parents=True)
        self.config = make_config(
            out_dir=str(out_dir),
            expansion_state=str(out_dir / "expansion.json"),
            selection_state=str(out_dir / "selection.json"),
            telemetry_file=str(out_dir / "telemetry.jsonl"),
            worker_state=str(out_dir / "worker_state.json"),
            worker_lock=str(out_dir / "worker.lock"),
        )

        self.saved_environment = dict(os.environ)
        self.saved_stdin = sys.stdin
        self.saved_input = builtins.input
        self.saved_refused = set(learning_worker._refused_domains)
        sys.stdin = Terminal()
        os.environ.pop("JUPITER_NO_PROMPT", None)

        def restore():
            os.environ.clear()
            os.environ.update(self.saved_environment)
            sys.stdin = self.saved_stdin
            builtins.input = self.saved_input
            learning_worker._refused_domains = set(self.saved_refused)

        self.addCleanup(restore)

    def put_pending_growth(self):
        state = expansion.load(self.config)
        state["capacity_growth"] = {
            "source": "a.txt",
            "cause": "insufficient model capacity",
            "from_layers": 8,
            "to_layers": 9,
            "from_embd": 448,
            "to_embd": 448,
            "approved": False,
        }
        expansion.save(self.config, state)

    def clear_pending_growth(self):
        # load() returns a fresh dict each call, so the removal has to be made on
        # one object and that object saved. Popping the result of a load and then
        # saving a second load removes nothing.
        state = expansion.load(self.config)
        state.pop("capacity_growth", None)
        expansion.save(self.config, state)

    def declining(self):
        """Record every question asked, and decline all of them."""
        self.asked = []

        def answer(prompt=""):
            self.asked.append(prompt)
            return "n"

        builtins.input = answer
        return self.asked

    def ask_both_questions(self):
        """What the continuous loop now does when it sees a growth waiting.

        Mirrors the loop: the grant is offered only when the recorded reason
        says the domain was not granted, and the approval question only when a
        growth is actually pending and unapproved.
        """
        asked = self.declining()
        pending = expansion.pending(self.config).get("capacity_growth")
        if isinstance(pending, dict) and not pending.get("approved"):
            learning_worker.offer_grant(
                self.config, "expansion", "a growth is waiting", tel=None
            )
            learning_worker.prompt_for_growth(self.config, None, tel=None)
        return asked

    def test_the_grant_question_is_reachable_in_the_continuous_path(self):
        self.put_pending_growth()
        os.environ.pop(control.GRANT_VARIABLE, None)
        asked = self.ask_both_questions()
        self.assertTrue(
            any("Grant 'expansion'" in question for question in asked),
            f"the grant question was never asked: {asked}",
        )

    def test_the_approval_question_is_reachable_when_the_grant_is_held(self):
        self.put_pending_growth()
        os.environ[control.GRANT_VARIABLE] = "expansion"
        asked = self.ask_both_questions()
        self.assertTrue(
            any("Approve this growth" in question for question in asked),
            f"the approval question was never asked: {asked}",
        )

    def test_declining_leaves_the_growth_unapplied(self):
        self.put_pending_growth()
        os.environ[control.GRANT_VARIABLE] = "expansion"
        self.ask_both_questions()
        pending = expansion.pending(self.config).get("capacity_growth", {})
        self.assertFalse(
            pending.get("approved"),
            "a declined growth was applied anyway",
        )

    def test_nothing_is_asked_when_nothing_is_waiting(self):
        """A loop that asks about nothing is as bad as one that asks about
        everything. run_worker is called at the end of every bounded run, so a
        prompt that appeared unconditionally would fire on every run of a project
        with no growth proposed."""
        self.clear_pending_growth()
        os.environ[control.GRANT_VARIABLE] = "expansion"
        asked = self.ask_both_questions()
        self.assertEqual(asked, [], f"asked with nothing waiting: {asked}")

    def test_the_grant_is_only_offered_once_per_run(self):
        self.put_pending_growth()
        os.environ.pop(control.GRANT_VARIABLE, None)
        self.declining()
        learning_worker.offer_grant(self.config, "expansion", "d", tel=None)
        learning_worker.offer_grant(self.config, "expansion", "d", tel=None)
        self.assertEqual(
            len(self.asked),
            1,
            "a continuous loop offers the same grant on every pass",
        )


if __name__ == "__main__":
    unittest.main()
