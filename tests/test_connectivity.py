"""Connectivity: every file must be reachable from the single entry point.

run_jupiter.bat is the entry point for the whole Jupiter project, so the project
has to behave as one interconnected system rather than a folder of modules where
some are used and some are not.

This test resolves the call graph that starts at run_jupiter.bat: the scripts it
invokes, plus everything they import transitively. A module no path reaches is
an orphan, and an orphan is a failure. There is no tolerated list and no pending
list: every module in the project is either driven by the entry point or it is
not finished, and this test says which.

Two files could not be in the default chain for structural reasons, and both are
solved by giving the entry point a mode rather than by excusing the file:

* watch_training.py follows the telemetry stream and never returns, so it runs
  through `run_jupiter.bat watch` and the chain uses its --summary mode.
* modal_train.py bills a remote A100 for up to an hour, so it runs through
  `run_jupiter.bat remote` and the chain only asks it whether it is available.

Architecture 40 lists one desired loop: initialize, run, learn, measure, save,
reload, verify. This test is the mechanical check that the code on disk can
perform that loop from a single command.

Print the current picture with:

    python -m tests.test_connectivity
"""

import ast
import re
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ENTRY_POINT = "run_jupiter.bat"
MODES = ("watch", "remote", "mcp")


def module_files():
    """Every production module filename in the project root."""
    return {path.name for path in PROJECT.glob("*.py")}


def module_stems():
    """The importable names of those modules, without the .py suffix."""
    return {Path(name).stem for name in module_files()}


def entry_scripts():
    """The scripts run_jupiter.bat invokes, in the order it invokes them.

    Interpreter flags are allowed before the filename because a script that has
    to print before it blocks is run as `python -u`, and a pattern that only
    matched a bare `python foo.py` would report that script as unreachable from
    the entry point. The test would then be enforcing a narrower idea of the
    entry point than the entry point actually is, which is the same class of
    failure this suite exists to catch: a green check over a stale assumption.

    Leading whitespace is allowed for the same reason. A script invoked inside a
    conditional block is still invoked by the entry point, and refusing to see it
    would report a script the chain runs as dead code.
    """
    text = (PROJECT / ENTRY_POINT).read_text(encoding="utf-8")
    return re.findall(
        r"^\s*python\s+(?:-\S+\s+)*([A-Za-z0-9_]+\.py)", text, re.MULTILINE
    )


def imports_of(stem):
    """Local module names one module imports, read from its source."""
    path = PROJECT / f"{stem}.py"
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            found.add(node.module.split(".")[0])
    return found & module_stems()


def reachable_from_entry():
    """Local module names reachable from the entry point, transitively."""
    reachable = set()
    frontier = [Path(name).stem for name in entry_scripts()]
    while frontier:
        stem = frontier.pop()
        if stem in reachable:
            continue
        reachable.add(stem)
        frontier.extend(imports_of(stem) - reachable)
    return reachable


def orphans():
    """Modules no path from the entry point reaches."""
    unreachable = module_stems() - reachable_from_entry()
    return {f"{stem}.py" for stem in unreachable}


class ConnectivityTests(unittest.TestCase):
    def test_the_entry_point_only_invokes_scripts_that_exist(self):
        for name in entry_scripts():
            self.assertTrue(
                (PROJECT / name).exists(),
                f"{ENTRY_POINT} invokes {name}, which does not exist",
            )

    def test_the_entry_point_learns_rather_than_only_fine_tuning(self):
        # Architecture 40 wants one loop: initialize, run, learn, measure,
        # save, reload, verify. A chain that only calls train.py reaches
        # none of the acceptance, confidence, regression or knowledge code,
        # so the loop the architecture describes is not the loop that runs.
        self.assertIn(
            "learning_worker.py",
            entry_scripts(),
            f"{ENTRY_POINT} does not invoke the learning controller, so "
            "update acceptance and regression detection never run",
        )

    def test_no_module_is_unreachable(self):
        self.assertEqual(
            orphans(),
            set(),
            f"these modules cannot be reached from {ENTRY_POINT}: {sorted(orphans())}",
        )

    def test_the_entry_point_offers_a_mode_for_every_non_default_file(self):
        text = (PROJECT / ENTRY_POINT).read_text(encoding="utf-8")
        for mode in MODES:
            self.assertIn(
                f'if /i "%~1"=="{mode}"',
                text,
                f"{ENTRY_POINT} has no {mode} mode",
            )

    def _default_chain(self):
        """The lines of the entry point that run when it is given no argument.

        Split on the label at the start of a line, not on any mention of it,
        because the mode dispatch near the top also says ":watch".
        """
        text = (PROJECT / ENTRY_POINT).read_text(encoding="utf-8")
        return text.split("\n:watch")[0]

    def _default_invocations(self, script):
        """Every line in the default chain that invokes `script`."""
        return [
            line.strip()
            for line in self._default_chain().splitlines()
            if script in line
        ]

    def test_remote_mode_is_not_in_the_default_chain(self):
        # A remote A100 run costs real money, so the default chain may only ask
        # whether remote training is available, never start it.
        invocations = self._default_invocations("modal_train.py")
        self.assertTrue(invocations, "the chain should still report availability")
        for line in invocations:
            self.assertIn("--check", line, f"default chain must not run {line}")

    def test_the_following_watcher_is_not_in_the_default_chain(self):
        # Following the stream never returns, so the default chain uses the
        # bounded summary mode instead.
        invocations = self._default_invocations("watch_training.py")
        self.assertTrue(invocations, "the chain should still show telemetry")
        for line in invocations:
            self.assertIn("--summary", line, f"default chain must not follow: {line}")

    def test_batch_files_use_windows_line_endings(self):
        # cmd.exe mis-parses labels in a batch file written with bare LF: goto
        # can fail to resolve a label, the script stops where it stands and
        # still exits 0, which silently hides a failing stage. Every stage in
        # the entry point is guarded by `if errorlevel 1 goto :error`, so the
        # line endings here are load-bearing, not cosmetic.
        for path in sorted(PROJECT.glob("*.bat")):
            raw = path.read_bytes()
            crlf = raw.count(b"\r\n")
            total = raw.count(b"\n")
            self.assertEqual(
                crlf,
                total,
                f"{path.name} must be CRLF throughout; found {crlf} CRLF "
                f"out of {total} lines",
            )


def _report():
    reachable = reachable_from_entry()
    everything = module_stems()
    print(f"entry point scripts : {', '.join(entry_scripts())}")
    print(f"reachable           : {len(reachable)}/{len(everything)} modules")
    print(f"orphans             : {', '.join(sorted(orphans())) or 'none'}")


if __name__ == "__main__":
    _report()
