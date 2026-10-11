"""A control domain can be granted at the point of refusal, once, on a yes.

Architecture 38 keeps architecture changes behind a human. The gate was right and
the affordance was missing: the only way past a refusal was to know the name of an
environment variable and set it before the process started, which a person
watching a run cannot do. Every growth met that wall, and the run stopped with a
message naming a variable the reader was not looking at.

The prompt this adds asks at the moment the answer is needed. What makes it a
control rather than a convenience is the set of things it refuses to do, and each
of those is a test here:

  - it does not grant when there is nobody to ask, because granting to nobody is
    how a control stops being one, and a detached worker must keep refusing
  - it does not grant on anything that is not a clear yes, so a stray keystroke,
    an empty line, or a closed input cannot widen what Jupiter may do
  - it asks once per domain per run, so a run that is refused on every cycle does
    not ask on every cycle

The real environment is restored after each case, so a developer's own grants are
never disturbed by running these.
"""

import builtins
import io
import os
import unittest
from contextlib import redirect_stdout

import control
import learning_worker

from tests.support import TempDirTestCase, make_config


class GrantPromptTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.saved_environment = dict(os.environ)
        self.saved_interactive = learning_worker._interactive
        self.saved_refused = set(learning_worker._refused_domains)
        self.saved_input = builtins.input

        def restore():
            os.environ.clear()
            os.environ.update(self.saved_environment)
            learning_worker._interactive = self.saved_interactive
            learning_worker._refused_domains = set(self.saved_refused)
            builtins.input = self.saved_input

        self.addCleanup(restore)

        c = make_config(
            out_dir=str(self.workdir / "out"),
            control_log=str(self.workdir / "decisions.jsonl"),
        )
        self.config = c

    def answer(self, reply):
        """Make input() reply with a canned value, and record what it was asked."""
        self.asked = []

        def fake_input(prompt=""):
            self.asked.append(prompt)
            if isinstance(reply, BaseException):
                raise reply
            return reply

        builtins.input = fake_input

    def run_prompt(self, domain="architecture", detail="shape differs"):
        with redirect_stdout(io.StringIO()):
            return learning_worker.offer_grant(self.config, domain, detail, tel=None)

    def test_nobody_to_ask_is_not_granted(self):
        learning_worker._interactive = lambda: False
        self.assertFalse(self.run_prompt())

    def test_a_clear_yes_grants_and_the_gate_sees_it(self):
        learning_worker._interactive = lambda: True
        os.environ[control.GRANT_VARIABLE] = "expansion"
        self.answer("y")

        self.assertTrue(self.run_prompt())
        self.assertIn("architecture", control.granted_domains())
        self.assertEqual(len(self.asked), 1, "asked more than once")

    def test_yes_in_either_case_is_accepted(self):
        for reply in ("y", "Y", "yes", "YES"):
            learning_worker._refused_domains = set()
            learning_worker._interactive = lambda: True
            os.environ[control.GRANT_VARIABLE] = ""
            self.answer(reply)
            self.assertTrue(self.run_prompt(), f"{reply!r} should be a clear yes")
            self.assertIn("architecture", control.granted_domains())

    def test_anything_else_is_not_granted(self):
        """The refusal half is the important half.

        Every one of these is a person who did not say yes, or did not say
        anything at all. None of them may widen what Jupiter is allowed to do.
        """
        for reply in ("", "n", "N", "no", "maybe", "sure", "1", "why not"):
            with self.subTest(reply=reply):
                learning_worker._refused_domains = set()
                learning_worker._interactive = lambda: True
                os.environ[control.GRANT_VARIABLE] = "expansion"
                self.answer(reply)

                self.assertFalse(self.run_prompt())
                self.assertNotIn("architecture", control.granted_domains())

    def test_a_closed_input_is_not_a_yes(self):
        """EOF is what a pipe gives and what Ctrl+D gives.

        Treating it as anything but a refusal would grant a domain to a run that
        never answered.
        """
        for reply in (EOFError(), KeyboardInterrupt()):
            with self.subTest(reply=type(reply).__name__):
                learning_worker._refused_domains = set()
                learning_worker._interactive = lambda: True
                os.environ[control.GRANT_VARIABLE] = "expansion"
                self.answer(reply)

                self.assertFalse(self.run_prompt())
                self.assertNotIn("architecture", control.granted_domains())

    def test_it_asks_once_per_domain(self):
        learning_worker._refused_domains = set()
        learning_worker._interactive = lambda: True
        os.environ[control.GRANT_VARIABLE] = ""
        self.answer("n")

        self.run_prompt()
        self.run_prompt()
        self.run_prompt()
        self.assertEqual(
            len(self.asked), 1, "a run refused every cycle must not ask every cycle"
        )

    def test_a_grant_keeps_the_domains_already_given(self):
        learning_worker._interactive = lambda: True
        os.environ[control.GRANT_VARIABLE] = "expansion,memory"
        self.answer("y")

        self.assertTrue(self.run_prompt())
        granted = control.granted_domains()
        self.assertIn("architecture", granted)
        self.assertIn("expansion", granted, "an existing grant was dropped")
        self.assertIn("memory", granted, "an existing grant was dropped")

    def test_the_run_still_refuses_when_the_grant_is_declined(self):
        """Declining must leave the gate exactly as it was.

        The prompt is offered back to the caller; a declined prompt returns False
        and the caller re-raises, so the run stops rather than proceeding as
        though the question had never been asked.
        """
        learning_worker._interactive = lambda: True
        os.environ[control.GRANT_VARIABLE] = ""
        self.answer("n")

        self.assertFalse(self.run_prompt())
        with self.assertRaises(control.ControlDenied):
            control.check(
                "architecture",
                "change the recorded architecture of a model in use",
                c=self.config,
            )


if __name__ == "__main__":
    unittest.main()
