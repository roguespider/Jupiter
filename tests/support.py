"""Shared helpers for the Jupiter test suite.

Every test that persists state gets a private temporary directory, so running
the suite cannot disturb the real data/ and out/ trees.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import control

from config import Config


def make_config(**overrides):
    """A Config with test overrides applied.

    Tests use the real Config class and the real default thresholds, so a change
    to a production default is visible to the suite rather than hidden by a
    test-local copy of the settings.
    """
    config = Config()
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


class AmbientControlIsolation(unittest.TestCase):
    """Runs every test with no Architecture 38 grants in the environment.

    control.check reads JUPITER_CONTROL_GRANTS from the real process
    environment, and several tests assert that a gated action is refused. If a
    developer, a CI runner or a parent shell has that variable set, those
    assertions fail for a reason that has nothing to do with the code: the gate
    is genuinely granted, so it genuinely does not refuse. The failure is
    indistinguishable from a broken control unless you know the variable was set.

    It was set. Running the pipeline with expansion granted made five control
    tests fail in step 1/8 while the rest of the suite stayed green, which is
    exactly the shape of a bug that is not there. Clearing the variable for the
    duration of every test makes the refusal deterministic no matter how the
    suite was invoked, and costs nothing: a test that wants a grant says so
    explicitly, which is what a grant is supposed to be.
    """

    def setUp(self):
        super().setUp()
        previous = os.environ.pop(control.GRANT_VARIABLE, None)

        def restore():
            os.environ.pop(control.GRANT_VARIABLE, None)
            if previous is not None:
                os.environ[control.GRANT_VARIABLE] = previous

        self.addCleanup(restore)


class TempDirTestCase(AmbientControlIsolation):
    """A test case with a private temporary directory."""

    def setUp(self):
        super().setUp()
        self.workdir = Path(tempfile.mkdtemp(prefix="jupiter-test-"))
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)

    def write(self, relative, text):
        """Write a file under the temporary directory and return its path."""
        target = self.workdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target
