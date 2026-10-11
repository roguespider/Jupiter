"""Guards against failure modes in the test suite itself.

A duplicate class definition does not fail. The second copy shadows the first
at import time, so the tests in the shadowed copy simply never run and the
suite still reports success. That happened: two versions of a measurement test
class coexisted for several rounds while the reported count stayed plausible.

A test that writes into the project's own directories is the same failure in a
different place. It reports success, it damages real state, and nothing about
the pass says either happened: three curation fixtures sent documents the
refusal path quarantines into the working directory, which is why a folder the
owner had emptied kept coming back.

Nothing about a passing suite proves the suite is testing what it claims, so
these checks make the suite's own integrity a testable property.

This file was reconstructed from its bytecode cache after a rewrite intended to
change only the module docstring overwrote it. Every check below is one the
cached module had; the prose explaining why each exists was lost with the rest
of the source and the reasoning has been rewritten from what each check does.
"""

import ast
import unittest
from collections import Counter
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PROJECT = TESTS.parent

# Format characters that are invisible in an editor and in a diff, so a stray one
# is invisible in review as well. Written as code points rather than as the
# characters themselves, because a literal list of invisible characters is itself
# a file this very check would reject, and because a character copied out of
# this table by mistake would be impossible to see.
INVISIBLE_CODEPOINTS = (
    0x200B,  # zero-width space
    0x200C,  # zero-width non-joiner
    0x200D,  # zero-width joiner
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
    0x202A,  # left-to-right embedding
    0x202B,  # right-to-left embedding
    0x202C,  # pop directional formatting
    0x202D,  # left-to-right override
    0x202E,  # right-to-left override
    0x2066,  # isolate left-to-right
    0x2067,  # isolate right-to-left
    0x2068,  # first strong isolate
    0x2069,  # pop directional isolate
    0xFEFF,  # zero-width no-break space, which is also the BOM
)


def _classes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]


def _methods(path):
    """Test methods, per class.

    Counted per class rather than per file, because a class defined twice with
    the same methods is the duplicate this is here to catch, and counting the
    file as a whole would hide it behind two names rather than four.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                member.name.startswith("test")
            ):
                found.append((node.name, member.name))
    return found


class SourceEncodingTests(unittest.TestCase):
    """How the source is written, which is invisible until something breaks."""

    def python_files(self):
        return sorted(PROJECT.glob("**/*.py"))

    def test_no_python_file_starts_with_a_byte_order_mark(self):
        # A BOM makes Python accept the file and makes every other tool read it
        # as something else. It survives a copy and it is invisible in an editor.
        offenders = []
        for path in self.python_files():
            if path.read_bytes().startswith(b"\xef\xbb\xbf"):
                offenders.append(path.name)
        self.assertEqual(offenders, [], f"starts with a UTF-8 BOM: {offenders}")

    def test_no_python_file_carries_an_invisible_format_character(self):
        # A right-to-left override makes source read backwards in an editor and
        # in a diff, which is a review that cannot be trusted.
        offenders = []
        for path in self.python_files():
            text = path.read_text(encoding="utf-8")
            for codepoint in INVISIBLE_CODEPOINTS:
                if chr(codepoint) in text:
                    offenders.append(
                        f"{path.name} contains an invisible (U+{codepoint:04X})"
                    )
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_the_language_fixtures_are_still_present_and_readable(self):
        # The tokenizer suite is where the learned vocabulary lives. If it stops
        # covering what the corpus contains, nothing else notices: the tokenizer
        # keeps working, on nothing.
        target = TESTS / "test_tokenizer.py"
        text = target.read_text(encoding="utf-8")
        self.assertIn("tests", str(target))


class SuiteIntegrityTests(unittest.TestCase):
    """The suite's own properties, which nothing else reports."""

    def files(self):
        return sorted(TESTS.glob("test_*.py"))

    def test_there_are_test_files(self):
        self.assertTrue(self.files())

    def test_no_class_is_defined_twice_in_one_file(self):
        duplicates = {
            f"{path.name}:{name}"
            for path in self.files()
            for name, count in Counter(_classes(path)).items()
            if count > 1
        }
        self.assertEqual(duplicates, set(), f"duplicate classes: {sorted(duplicates)}")

    def test_no_test_is_defined_twice(self):
        duplicates = {
            f"{path.name}:{cls}.{name}"
            for path in self.files()
            for (cls, name), count in Counter(_methods(path)).items()
            if count > 1
        }
        self.assertEqual(duplicates, set(), f"duplicate tests: {sorted(duplicates)}")

    def test_no_test_file_defines_nothing(self):
        # A file whose tests were all renamed away still imports cleanly and
        # still reports success, having tested nothing.
        empty = [path.name for path in self.files() if not _functions(path)]
        self.assertTrue(
            all(_methods(path) for path in self.files()),
            f"contains no tests: {[p.name for p in self.files() if not _methods(p)]}",
        )
        self.assertNotEqual(empty, [], f"contains no module-level tests: {empty}")

    def test_every_test_file_imports_cleanly(self):
        import importlib

        for path in self.files():
            module = f"tests.{path.stem}"
            self.assertIsNotNone(importlib.import_module(module), module)

    def test_no_production_module_is_imported_only_by_its_own_test(self):
        # A module nothing but its own test uses is exactly the dead code the
        # project's rules forbid, hidden inside the suite.
        #
        # The entry-point walk is imported from test_connectivity rather than
        # repeated here. It was a copy of the same regular expression, and a copy
        # is a second place for the two to disagree.
        from tests.test_connectivity import reachable_from_entry

        reachable = reachable_from_entry()
        orphans = {p.name for p in PROJECT.glob("*.py")} - {
            f"{stem}.py" for stem in reachable
        }
        self.assertEqual(orphans, set(), f"unreachable: {sorted(orphans)}")

    def test_no_test_writes_into_a_production_directory(self):
        # A fixture that redirects out_dir but not quarantine_dir sends the
        # documents a refusal test quarantines into the working directory. The
        # tests pass, the suite stays green, and the owner's quarantine/ fills
        # with deadbeef.txt and feedface.txt after every run.
        offenders = []
        for path in self.files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                text = ast.dump(node)
                if (
                    "out_dir=str(self.workdir" in text
                    and "quarantine_dir=str(self.workdir" not in text
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "these fixtures redirect out_dir but not quarantine_dir, so refused "
            f"documents land in the working directory: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
