"""Tests for structured programming knowledge (Architecture 26).

Architecture 26's requirement is not that Jupiter knows about twelve programming
capabilities. It is that "programming knowledge should be treated as structured
knowledge rather than simply memorized text", and that one sentence decides what
is worth building. So these tests are mostly about whether the structure is real.

The distinction they hold hardest is between the language with a parser and the
languages read by pattern. Python is read with the standard library's own ast
module, which gives real names, real kinds, real parameter lists and real line
numbers. Anything read by pattern is a guess, and a guess about code becomes a
fabricated fact, so the tests below check that the two are never confused.
"""

import unittest

import code_knowledge
import store as store_module

from tests.support import TempDirTestCase

PYTHON = """import os
from pathlib import Path

MAX_SIZE = 1024


def parse(text):
    return text.strip()


class Reader:
    def __init__(self, path):
        self.path = path

    def read(self):
        return open(self.path).read()
"""

JAVASCRIPT = """const fs = require("fs");

function loadConfig(path) {
  return fs.readFileSync(path);
}

class Store {
  constructor(name) { this.name = name; }
}
"""


class PythonParsingTests(unittest.TestCase):
    def test_python_is_read_with_the_real_parser(self):
        parsed = code_knowledge.parse(PYTHON)
        self.assertEqual(parsed["language"], "python")
        self.assertEqual(parsed["read_with"], "ast")
        self.assertIsNone(parsed["parse_problem"])

    def test_a_class_method_is_not_also_a_free_function(self):
        # Walking a class and then recursing into it again at the bottom of the
        # loop recorded every method twice. A symbol table that claims a class
        # has two copies of its own methods is worse than no table.
        parsed = code_knowledge.parse(PYTHON)
        names = [s["name"] for s in parsed["symbols"]]
        self.assertEqual(len(names), len(set(names)), names)
        kinds = {s["name"]: s["kind"] for s in parsed["symbols"]}
        self.assertEqual(kinds["__init__"], "method")
        self.assertEqual(kinds["read"], "method")
        self.assertEqual(kinds["parse"], "function")

    def test_parameter_names_are_real(self):
        parsed = code_knowledge.parse(PYTHON)
        by_name = {s["name"]: s for s in parsed["symbols"]}
        self.assertEqual(by_name["parse"]["params"], ["text"])
        self.assertEqual(by_name["__init__"]["params"], ["self", "path"])

    def test_a_constant_is_told_from_a_variable(self):
        parsed = code_knowledge.parse(PYTHON)
        self.assertIn("constant", [s["kind"] for s in parsed["symbols"]])

    def test_imports_are_collected(self):
        self.assertEqual(
            sorted(code_knowledge.parse(PYTHON)["imports"]), ["os", "pathlib"]
        )

    def test_broken_python_is_reported_rather_than_raised(self):
        parsed = code_knowledge.parse("def broken(:\n  pass\n")
        self.assertEqual(parsed["language"], "python")
        self.assertIsNotNone(parsed["parse_problem"])
        # Not half-read. Reading a broken file with the weaker reader is how
        # wrong symbols get recorded.
        self.assertEqual(parsed["symbols"], [])

    def test_a_fragment_that_declares_python_is_still_recognised_as_python(self):
        # The declaration patterns do not require a trailing colon precisely so
        # that a file with an error in it is still recognised and reported.
        self.assertEqual(code_knowledge.detect_language("def broken(:"), "python")


class LanguageDetectionTests(unittest.TestCase):
    def test_common_languages_are_recognised(self):
        cases = {
            PYTHON: "python",
            JAVASCRIPT: "javascript",
            '#include <stdio.h>\nint main(void) { printf("x"); }': "c",
            "package main\nfunc run() int { return 1 }": "go",
            "SELECT id FROM users WHERE id = 1": "sql",
            "<!DOCTYPE html>\n<html><div>x</div></html>": "html",
        }
        for text, expected in cases.items():
            self.assertEqual(code_knowledge.detect_language(text), expected, expected)

    def test_prose_is_not_mistaken_for_code(self):
        self.assertIsNone(
            code_knowledge.detect_language("the wind moved over the valley. " * 5)
        )

    def test_an_unidentified_document_is_none_not_a_guess(self):
        # Saying "unknown" and saying "this is not a programme" are different
        # claims, and only the first is supported.
        self.assertIsNone(code_knowledge.detect_language(""))


class PatternReadingTests(unittest.TestCase):
    def test_other_languages_are_read_with_patterns_and_say_so(self):
        parsed = code_knowledge.parse(JAVASCRIPT)
        self.assertEqual(parsed["language"], "javascript")
        self.assertEqual(parsed["read_with"], "patterns")
        self.assertIsNone(parsed["parse_problem"])

    def test_javascript_declarations_are_found(self):
        names = {s["name"] for s in code_knowledge.parse(JAVASCRIPT)["symbols"]}
        self.assertIn("loadConfig", names)
        self.assertIn("Store", names)

    def test_javascript_requires_are_collected(self):
        self.assertIn("fs", code_knowledge.parse(JAVASCRIPT)["imports"])


class DiagnosticTests(unittest.TestCase):
    def test_errors_warnings_and_hints_are_told_apart(self):
        found = code_knowledge.read_diagnostics(
            "Traceback: most recent call last. Warning: deprecated API. hint: try this"
        )
        kinds = {d["kind"] for d in found}
        self.assertIn("error", kinds)
        self.assertIn("warning", kinds)
        self.assertIn("hint", kinds)

    def test_diagnostics_carry_a_line(self):
        found = code_knowledge.read_diagnostics("ok\nError: something\n")
        self.assertEqual(found[0]["line"], 2)

    def test_prose_with_no_diagnostics_yields_none(self):
        self.assertEqual(
            code_knowledge.read_diagnostics("the valley was quiet. " * 5), []
        )

    def test_reading_diagnostics_is_bounded(self):
        text = "Error: bad. " * 5000
        self.assertLessEqual(len(code_knowledge.read_diagnostics(text)), 200)


class StoreTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_a_recorded_document_becomes_a_structured_record(self):
        parsed = code_knowledge.record(self.store, "doc-a.py", PYTHON, cycle=1)
        self.assertEqual(parsed["source"], "doc-a.py")
        stored = self.store.collection(code_knowledge.SYMBOLS)["doc-a.py"]
        self.assertEqual(stored["language"], "python")
        self.assertTrue(stored["symbols"])

    def test_dependencies_are_recorded_against_their_module(self):
        code_knowledge.record(self.store, "doc-a.py", PYTHON, cycle=1)
        self.assertIn("os", self.store.collection(code_knowledge.DEPENDENCIES))

    def test_the_same_symbol_in_two_languages_is_paired(self):
        # Architecture 26 lists cross-language reasoning. The groundwork is this:
        # a name reached in two languages becomes two records that share a name,
        # so finding the pairing is a lookup rather than an inference.
        code_knowledge.record(self.store, "a.py", "def load():\n    pass\n", cycle=1)
        code_knowledge.record(self.store, "b.js", "function load() {\n}\n", cycle=2)
        pairs = code_knowledge.cross_language(self.store)
        self.assertTrue(pairs, pairs)
        self.assertEqual(pairs[0]["name"], "load")
        self.assertEqual(sorted(pairs[0]["languages"]), ["javascript", "python"])

    def test_a_symbol_in_one_language_is_not_paired(self):
        code_knowledge.record(self.store, "a.py", "def only():\n    pass\n", cycle=1)
        self.assertEqual(code_knowledge.cross_language(self.store), [])

    def test_languages_the_corpus_contains_are_counted(self):
        code_knowledge.record(self.store, "a.py", PYTHON, cycle=1)
        code_knowledge.record(self.store, "b.js", JAVASCRIPT, cycle=1)
        languages = self.store.collection(code_knowledge.LANGUAGES)
        self.assertEqual(sorted(languages), ["javascript", "python"])
        self.assertEqual(languages["python"]["documents"], 1)

    def test_the_census_says_what_it_is_not(self):
        # A layer that claims more than it does is worse than a smaller honest
        # one, so the census carries its own limits.
        note = code_knowledge.census(self.store)["note"].lower()
        self.assertIn("parser", note)
        self.assertIn("heuristic", note)
        self.assertIn("executes", note)


if __name__ == "__main__":
    unittest.main()
