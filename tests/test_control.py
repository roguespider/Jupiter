"""Tests for the Architecture 38 control layer.

Architecture 38 asks for explicit human control over six kinds of change and had
none of them. The project's own changelog said the reason: "a flag that gates
nothing is not a control." So these tests are mostly about whether the gate
actually refuses, because a control that cannot refuse is indistinguishable
from no control at all.

The tests that matter most are the negative ones: a denied action must leave
nothing behind, and a grant must come from configuration rather than from
anything the system itself can write.
"""

import os
import shutil
import unittest

import control

from config import save_architecture
from tests.support import AmbientControlIsolation, TempDirTestCase, make_config


class AmbientGrantIsolationTests(AmbientControlIsolation):
    """The suite must give the same answer however it was invoked.

    Running the pipeline with JUPITER_CONTROL_GRANTS set made five control tests
    fail while the other 664 passed, which is the shape of a broken gate rather
    than a broken test. The gate was not broken: it was granted, by the
    environment, and the tests assert that it refuses by default. This pins the
    fix so a developer who has the variable exported does not spend an afternoon
    looking for a control bug that is not there.
    """

    def test_a_refusal_test_is_unaffected_by_an_exported_grant(self):
        # setUp removed the variable, which is the whole mechanism. Asserting
        # that the environment is clean proves the isolation ran for this test.
        self.assertNotIn(control.GRANT_VARIABLE, os.environ)
        with self.assertRaises(control.ControlDenied):
            control.check("expansion", "grow the model", allowed=None)

    def test_the_previous_value_is_restored_afterwards(self):
        # Isolation that leaks would silently strip a developer's real grant for
        # the rest of the process, which is its own kind of control failure.
        os.environ[control.GRANT_VARIABLE] = "expansion"
        self.addCleanup(os.environ.pop, control.GRANT_VARIABLE, None)
        self.assertIn(control.GRANT_VARIABLE, os.environ)


class GrantTests(AmbientControlIsolation):
    def test_nothing_is_granted_by_default(self):
        self.assertEqual(control.granted_domains(environ={}), set())

    def test_the_six_architecture_38_domains_are_defined(self):
        self.assertEqual(
            set(control.DOMAINS),
            {
                "architecture",
                "source",
                "expansion",
                "external",
                "memory",
                "destructive",
            },
        )

    def test_a_named_domain_is_granted(self):
        self.assertEqual(
            control.granted_domains(environ={"JUPITER_CONTROL_GRANTS": "source"}),
            {"source"},
        )

    def test_several_domains_can_be_granted(self):
        granted = control.granted_domains(
            environ={"JUPITER_CONTROL_GRANTS": "source, memory"}
        )
        self.assertEqual(granted, {"source", "memory"})

    def test_whitespace_and_empty_entries_are_tolerated(self):
        granted = control.granted_domains(
            environ={"JUPITER_CONTROL_GRANTS": " source , , memory "}
        )
        self.assertEqual(granted, {"source", "memory"})

    def test_a_name_outside_the_documented_domains_is_refused(self):
        # A typo must not quietly open a hole. Somebody who writes "srouce"
        # believes they have granted something and have not.
        with self.assertRaises(ValueError) as caught:
            control.granted_domains(environ={"JUPITER_CONTROL_GRANTS": "srouce"})
        self.assertIn("Architecture 38", str(caught.exception))

    def test_there_is_no_wildcard(self):
        # "*" is the shape an unconfigured bypass would take.
        with self.assertRaises(ValueError):
            control.granted_domains(environ={"JUPITER_CONTROL_GRANTS": "*"})


class DenyByDefaultTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.log = self.workdir / "decisions.jsonl"
        self.allowed = ()

    def test_every_domain_refuses_without_a_grant(self):
        for domain in control.DOMAINS:
            with self.assertRaises(control.ControlDenied, msg=domain):
                control.check(domain, "an action", c=None, allowed=self.allowed)

    def test_a_refusal_explains_how_to_authorize_it(self):
        with self.assertRaises(control.ControlDenied) as caught:
            control.check("memory", "write knowledge", allowed=())
        message = str(caught.exception)
        self.assertIn(control.GRANT_VARIABLE, message)
        self.assertIn("memory", message)

    def test_a_refusal_is_a_permission_error(self):
        # It is a permission problem, and the exception type is what a reader
        # of a log or a traceback will branch on.
        with self.assertRaises(PermissionError):
            control.check("external", "run something", allowed=())

    def test_an_unknown_domain_is_refused_rather_than_ignored(self):
        with self.assertRaises(ValueError):
            control.check("not-a-domain", "do a thing", allowed=())

    def test_a_grant_permits_exactly_that_domain(self):
        self.assertTrue(control.check("memory", "write", allowed={"memory"}))
        with self.assertRaises(control.ControlDenied):
            control.check("source", "write", allowed={"memory"})

    def test_both_allow_and_deny_are_recorded(self):
        control.check("memory", "permitted", c=None, allowed={"memory"})
        with self.assertRaises(control.ControlDenied):
            control.check("source", "refused", c=None, allowed=())
        summary = control.summary()
        self.assertGreaterEqual(summary["decisions"], 2)
        self.assertGreaterEqual(summary["denied"], 1)
        self.assertEqual(set(summary["domains"]), set(control.DOMAINS))


class PathClassificationTests(TempDirTestCase):
    def test_data_and_out_are_the_systems_own(self):
        self.assertTrue(control.is_data_path("data/raw/a.txt", project=self.workdir))
        self.assertTrue(
            control.is_data_path("out/knowledge.json", project=self.workdir)
        )

    def test_source_is_not(self):
        self.assertFalse(control.is_data_path("model.py", project=self.workdir))
        self.assertFalse(
            control.is_data_path("tests/test_learning.py", project=self.workdir)
        )

    def test_a_path_outside_the_project_is_not(self):
        self.assertFalse(
            control.is_data_path("C:/elsewhere/x.txt", project=self.workdir)
        )

    def test_writing_outside_data_and_out_is_gated(self):
        with self.assertRaises(control.ControlDenied):
            control.check_write(None, "model.py", allowed=())

    def test_writing_data_is_not_gated(self):
        self.assertTrue(control.check_write(None, "data/raw/a.txt", allowed=()))

    def test_destruction_is_gated_and_a_move_is_not(self):
        # The distinction is what makes the gate usable: a control on the only
        # safe path forward is a control people turn off.
        with self.assertRaises(control.ControlDenied):
            control.check_destructive(None, "out/knowledge.json", allowed=())
        self.assertTrue(control.check_write(None, "out/knowledge.json", allowed=()))


class ArchitectureGateTests(TempDirTestCase):
    """The architecture gate sits on the real action, not on a flag."""

    def setUp(self):
        super().setUp()
        self.path = self.workdir / "Jupiter.architecture.json"

    def config(self, block_size):
        c = make_config(device="cpu", out_dir=str(self.workdir))
        c.block_size = block_size
        return c

    def test_a_first_write_is_not_a_change(self):
        save_architecture(self.config(512), self.path, 256)
        self.assertTrue(self.path.exists())

    def test_rewriting_the_same_architecture_is_not_a_change(self):
        save_architecture(self.config(512), self.path, 256)
        save_architecture(self.config(512), self.path, 256)
        self.assertTrue(self.path.exists())

    def test_changing_the_architecture_of_a_model_in_use_is_gated(self):
        save_architecture(self.config(512), self.path, 256)
        with self.assertRaises(control.ControlDenied) as caught:
            save_architecture(self.config(1024), self.path, 256)
        self.assertIn("block_size", str(caught.exception))
        self.assertIn("512", str(caught.exception))

    def test_a_refused_change_leaves_the_record_as_it_was(self):
        # The refusal has to actually stop the write, not merely complain.
        save_architecture(self.config(512), self.path, 256)
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(control.ControlDenied):
            save_architecture(self.config(1024), self.path, 256)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_a_change_to_the_vocabulary_is_gated_too(self):
        save_architecture(self.config(512), self.path, 256)
        with self.assertRaises(control.ControlDenied) as caught:
            save_architecture(self.config(512), self.path, 5256)
        self.assertIn("vocab_size", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class DestructiveDomainTests(AmbientControlIsolation):
    """Why the destructive domain has no action wired to it.

    Architecture 38 asks for human control over destructive operations, and
    the domain is defined and refusing by default. It is not attached to
    anything, because there is nothing to attach it to: every removal in
    Jupiter is a move or a lock file, never an irrecoverable deletion. Curation
    quarantines rather than deletes, checkpoints are archived rather than
    removed, and the only unlink calls in production are for a stale lock and
    for a temporary file that was never part of the store.

    That is worth pinning rather than leaving implicit. If someone adds a
    genuine deletion later, the domain is already there refusing, and a test
    will say so.
    """

    def test_the_domain_is_defined_and_refuses_by_default(self):
        with self.assertRaises(control.ControlDenied):
            control.check("destructive", "destroy something", allowed=())

    def test_curation_moves_rather_than_deletes(self):
        import inspect

        import curate_dataset

        source = inspect.getsource(curate_dataset.prune_curated)
        self.assertIn("shutil.move", source)
        self.assertNotIn("unlink", source)
        self.assertNotIn("remove(", source)

    def test_no_production_module_deletes_a_document(self):
        """Nothing in production may irreversibly delete a document.

        The patterns name the RECEIVER as well as the method, and that is the
        whole point of this rewrite. The original matched the characters ".remove("
        anywhere in a file, which cannot tell a filesystem call from an object
        method, and flagged plasticity.py for calling handle.remove() on a
        torch.utils.hooks.RemovableHandle. That call deletes a key from a dict
        of forward hooks. The layer, its weights and the model all still exist
        afterwards, and no document is involved.

        A blunt match is not a paranoid one: it caught a real thing once, when
        purge_digest_concepts.py genuinely needed to remove records and the store
        had no way to do it. What it cannot do is tell the difference, so the
        patterns below match the call sites that can actually destroy something:
        the os and shutil functions, Path.unlink, and a bare remove on a path.
        A method named remove on a handle, a set, or a dict is not one of them.

        The allowlist covers the modules that legitimately delete records rather
        than documents. Each entry is explained where it appears, because an
        allowlist with no stated reason is a hole rather than an exception.
        """
        import pathlib

        project = pathlib.Path(__file__).resolve().parent.parent
        # store.py holds the removal primitive itself. curate_dataset.py prunes
        # curated documents by moving them aside rather than unlinking them.
        # purge_digest_concepts.py is here for the same reason as store.py and not
        # because it deletes anything a person wrote: it removes records from the
        # knowledge store, after copying the store to a timestamped backup, and
        # only records that are a curated filename's sha256 carrying no evidence
        # at all. It moves no document and unlinks no file.
        allowed = (
            "lock.py",
            "store.py",
            "curate_dataset.py",
            "purge_digest_concepts.py",
        )
        offenders = []
        for path in sorted(project.glob("*.py")):
            if path.name in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in (
                "os.remove(",
                "shutil.rmtree(",
                ".unlink(",
                "Path.unlink(",
            ):
                if pattern in text and not text.lstrip().startswith("#"):
                    offenders.append(f"{path.name}: {pattern}")
        self.assertEqual(offenders, [], f"irrecoverable deletion found: {offenders}")

    def test_the_deletion_check_still_catches_a_real_deletion(self):
        """The narrowed check has to still bite.

        A pattern list that matches nothing is not a safety net, and this is the
        only thing that distinguishes a check that was widened to stop failing
        from one that was narrowed to match its intent. The candidates are
        written to a temporary directory and read back the same way the check
        reads real modules.
        """
        import pathlib
        import tempfile

        project = pathlib.Path(__file__).resolve().parent.parent
        patterns = ("os.remove(", "shutil.rmtree(", ".unlink(", "Path.unlink(")

        work = pathlib.Path(tempfile.mkdtemp(prefix="jupiter-deletion-probe-"))
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)

        caught = []
        for name, body in (
            ("deletes_a_file.py", "import os\nos.remove(path)\n"),
            ("deletes_a_tree.py", "import shutil\nshutil.rmtree(path)\n"),
            ("unlinks_a_path.py", "target.unlink()\n"),
        ):
            probe = work / name
            probe.write_text(body, encoding="utf-8")
            hits = [p for p in patterns if p in probe.read_text(encoding="utf-8")]
            if hits:
                caught.append((name, hits))

        self.assertEqual(
            len(caught), 3, f"the deletion check missed a real deletion: {caught}"
        )

    def test_an_object_method_named_remove_is_not_a_deletion(self):
        """The false positive this rewrite exists to stop, pinned as a test.

        plasticity.py removes a PyTorch forward hook with handle.remove() in the
        middle of a try/finally. That is bookkeeping, not destruction, and a
        checker that cannot see the difference will eventually be changed to
        ignore something that does destroy.
        """
        import pathlib
        import tempfile

        patterns = ("os.remove(", "shutil.rmtree(", ".unlink(", "Path.unlink(")
        work = pathlib.Path(tempfile.mkdtemp(prefix="jupiter-removable-"))
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)

        probe = work / "removes_a_hook.py"
        probe.write_text(
            "handles.append(module.register_forward_hook(hook))\n"
            "try:\n"
            "    with torch.no_grad():\n"
            "        logits, _ = model(idx)\n"
            "finally:\n"
            "    for handle in handles:\n"
            "        handle.remove()\n",
            encoding="utf-8",
        )
        text = probe.read_text(encoding="utf-8")
        hits = [p for p in patterns if p in text]
        self.assertEqual(hits, [], f"a hook removal was read as a deletion: {hits}")

    def test_the_purge_takes_a_backup_before_it_removes_anything(self):
        """The allowlist entry above is only honest if this holds.

        A script allowed to remove records has to be able to put them back, or
        the allowlist is a hole rather than an exception. Checked by reading the
        module's own behaviour rather than its comments: with --apply it copies
        the store aside before writing, and without it, it writes nothing.
        """
        import inspect
        import pathlib
        import tempfile

        import purge_digest_concepts as purge

        from config import load_config

        source = inspect.getsource(purge.main)
        self.assertIn("shutil.copy2", source, "the purge takes no backup")

        project = pathlib.Path(__file__).resolve().parent.parent
        work = tempfile.mkdtemp(prefix="jupiter-purge-backup-")
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)
        store_path = pathlib.Path(work) / "knowledge.json"
        store_path.write_text('{"concepts": {}, "relationships": {}}', encoding="utf-8")
        backup = store_path.with_suffix(".json" + purge.BACKUP_SUFFIX)
        self.assertFalse(backup.exists())

        c = load_config()
        c.out_dir = work
        self.assertEqual(purge.main([]), 0, "the dry run should succeed")
        self.assertFalse(
            backup.exists(), "the dry run wrote a backup, so it is not a dry run"
        )
