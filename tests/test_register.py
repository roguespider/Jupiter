"""Tests for the register classifier (Architecture 27).

Architecture 27 names the initial language capability: English, programming
language text, technical documentation, informal text, common errors, slang and
malformed input. The last three are the ones a byte-level model actually has to
survive, and none of them were being recorded anywhere, so the knowledge layer
had no idea which of its material was a novel and which was a log file.

The tests are mostly about the two failure modes that matter. A classifier that
over-claims turns a gap in itself into a fact about the corpus, and one that
under-claims leaves the interesting cases invisible.
"""

import unittest

import register
import store as store_module

from tests.support import TempDirTestCase

PROSE = (
    "The observatory stood on the granite above the valley. "
    "Instruments recorded the river and the wind through the long dark winter nights. "
)
CODE = (
    "def parse(text):\n    if not text:\n        return None\n    return text.strip()\n"
)
DOCUMENTATION = (
    "The parse function returns None when the text is empty. "
    "Parameters: text, a string. Raises ValueError on invalid input. "
    "Usage: call parse before reading the file. "
)
TRACEBACK = (
    "Traceback (most recent call last): KeyError: 'missing' at line 42 in parse()"
)
SLANG = (
    "that rollout was absolutely fire, the model is goated and the eval is clean bro"
)


class MalformedTests(unittest.TestCase):
    def test_control_characters_are_not_text(self):
        self.assertTrue(register.is_malformed("ok\x00\x01\x02 data")[0])

    def test_a_replacement_character_means_a_failed_decode(self):
        found, reason = register.is_malformed("the river � the valley")
        self.assertTrue(found)
        self.assertIn("replacement", reason)

    def test_ordinary_text_is_not_malformed(self):
        self.assertFalse(register.is_malformed(PROSE)[0])

    def test_newlines_and_tabs_are_whitespace_not_malformed(self):
        # The control range deliberately excludes tab, newline and carriage
        # return. Treating a tab as corruption would condemn every code file.
        self.assertFalse(register.is_malformed("a\tb\nc\r\nd")[0])

    def test_empty_text_is_not_malformed(self):
        self.assertFalse(register.is_malformed("")[0])

    def test_malformed_beats_every_other_class(self):
        # A document that is not text has no language to classify, and calling
        # it prose is worse than saying nothing.
        self.assertEqual(
            register.classify(PROSE + "\x00\x01\x02\x03\x04")["register"], "malformed"
        )


class RegisterTests(unittest.TestCase):
    def test_english_prose_is_prose(self):
        self.assertEqual(register.classify(PROSE)["register"], "prose")

    def test_code_is_programming_language_text(self):
        self.assertEqual(
            register.classify(CODE)["register"], "programming_language_text"
        )

    def test_a_stack_trace_is_an_error_report(self):
        self.assertEqual(register.classify(TRACEBACK)["register"], "error_report")

    def test_documentation_about_errors_is_not_an_error_report(self):
        # This is the case the two-tier error markers exist for. A manual that
        # says "raises ValueError" is documentation, and filing it as an error
        # report puts a page of instructions in the wrong pile.
        self.assertEqual(
            register.classify(DOCUMENTATION)["register"], "technical_documentation"
        )

    def test_slang_is_not_plain_prose(self):
        # Checked before the English fallback, because a sentence with several
        # slang markers is informal English and reporting it as prose loses
        # exactly what Architecture 27 names.
        self.assertEqual(register.classify(SLANG)["register"], "slang")

    def test_an_empty_document_is_unknown_rather_than_prose(self):
        self.assertEqual(register.classify("")["register"], "unknown")

    def test_a_class_that_matches_nothing_is_left_unknown(self):
        # Forcing it would turn a gap in the classifier into a claim about the
        # corpus, which is the failure this design is meant to avoid.
        result = register.classify("qqq zzz vvv www eee rrr ttt yyy uuu")
        self.assertEqual(result["register"], "unknown")

    def test_every_result_carries_its_reason(self):
        for text in (PROSE, CODE, DOCUMENTATION, TRACEBACK, SLANG, "", "x\x00"):
            self.assertTrue(register.classify(text)["reason"], text)

    def test_every_class_named_by_the_chapter_exists(self):
        # Architecture 27's list, so a class cannot quietly go missing.
        for name in (
            "prose",
            "programming_language_text",
            "technical_documentation",
            "informal",
            "error_report",
            "slang",
            "malformed",
        ):
            self.assertIn(name, register.CLASSES, name)

    def test_the_result_reports_size(self):
        result = register.classify(PROSE)
        self.assertEqual(result["words"], len(PROSE.split()))
        self.assertGreater(result["characters"], 0)


class RegisterStoreTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_a_recorded_document_keeps_its_finding(self):
        register.record(self.store, "a.txt", PROSE, cycle=1)
        stored = self.store.collection(register.REGISTER)["a.txt"]
        self.assertEqual(stored["register"], "prose")
        self.assertEqual(stored["source"], "a.txt")

    def test_the_census_counts_registers_and_malformed_material(self):
        register.record(self.store, "a.txt", PROSE, cycle=1)
        register.record(self.store, "b.py", CODE, cycle=1)
        register.record(self.store, "c.log", "data \x00\x01\x02", cycle=1)
        census = register.census(self.store)
        self.assertEqual(census["documents"], 3)
        self.assertEqual(census["by_register"]["prose"], 1)
        self.assertEqual(census["malformed"], 1)

    def test_the_census_says_what_it_is_not(self):
        note = register.census(self.store)["note"]
        self.assertIn("does not", note)
        self.assertIn("unknown", note)


if __name__ == "__main__":
    unittest.main()
