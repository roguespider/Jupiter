"""Tests for the payload scrub (Architecture 25, cleaning stage).

The scrub is a net under the parser, and a net that catches real fish also
catches things that are not fish. These tests are mostly about what it must NOT
touch, because this project's corpus contains a page of LaTeX and a library of
programming material, and a scrub that quietly edits either of those is worse
than a scrub that does nothing.
"""

import unittest

import scrub


class EscapeRemovalTests(unittest.TestCase):
    """Escapes that arrived inside a string and were never unescaped."""

    def test_an_escaped_newline_becomes_a_newline(self):
        cleaned, counts = scrub.scrub("row one\\nrow two")
        self.assertEqual(cleaned, "row one\nrow two")
        self.assertEqual(counts["escaped_newlines"], 1)

    def test_an_escaped_quote_becomes_a_quote(self):
        cleaned, counts = scrub.scrub('say \\"hi\\"')
        self.assertEqual(cleaned, 'say "hi"')
        self.assertEqual(counts["escaped_quotes"], 2)

    def test_an_escaped_tab_before_a_space_is_a_tab(self):
        cleaned, counts = scrub.scrub("a\\t b")
        self.assertEqual(cleaned, "a  b")
        self.assertEqual(counts["escaped_whitespace"], 1)

    def test_an_escaped_carriage_return_at_the_end_is_removed(self):
        cleaned, counts = scrub.scrub("line\\r")
        self.assertEqual(cleaned, "line ")
        self.assertEqual(counts["escaped_whitespace"], 1)

    def test_a_json_string_with_escapes_is_cleaned(self):
        cleaned, counts = scrub.scrub('{"content": "row one\\nrow two"}')
        self.assertNotIn("\\n", cleaned)
        self.assertGreater(scrub.total(counts), 0)

    def test_empty_text_scrubs_to_nothing_without_raising(self):
        cleaned, counts = scrub.scrub("")
        self.assertEqual(cleaned, "")
        self.assertEqual(scrub.total(counts), 0)


class LaTeXIsNotContaminationTests(unittest.TestCase):
    """The corpus contains real mathematics, and it must survive untouched.

    This is the whole reason key removal is gated and tab removal is
    positional. A pattern that turned `\\to` into "to" would delete a page of
    algebra and report a clean run doing it.
    """

    UNTOUCHED = (
        ("inline math", "Let \\(a,b\\) satisfy the conditions"),
        ("display math", "$$f: \\mathbb{R} \\to \\mathbb{R}$$"),
        ("leq", "a \\leq b"),
        ("times", "a \\times b"),
        ("text commands", "\\textbf{bold} \\textit{it} \\text{word}"),
        ("theta", "x \\theta y"),
        ("tilde", "p \\tilde q"),
        ("top", "\\top"),
        ("quad", "x \\quad y"),
        ("matrix", "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}"),
    )

    def test_no_latex_is_ever_altered(self):
        for label, text in self.UNTOUCHED:
            with self.subTest(label):
                cleaned, _ = scrub.scrub(text)
                self.assertEqual(cleaned, text)

    def test_latex_survives_the_aggressive_path_too(self):
        # Aggressive mode removes keys; it must still not touch maths.
        for label, text in self.UNTOUCHED:
            with self.subTest(label):
                cleaned, _ = scrub.scrub(text, aggressive=True)
                self.assertEqual(cleaned, text)


class KeyRemovalIsGatedTests(unittest.TestCase):
    """Quoted keys are only removed from a document judged to be a payload."""

    PAYLOAD = (
        '{"input": [{"role": "user", "content": "what is an index"}], '
        '"output": "it speeds up lookups"}'
    )

    def test_a_judged_payload_loses_its_field_names(self):
        cleaned, counts = scrub.scrub(self.PAYLOAD, aggressive=True)
        self.assertGreater(counts["structural_keys"], 0)
        self.assertNotIn('"role"', cleaned)

    def test_the_same_text_keeps_its_keys_when_it_is_not_a_payload(self):
        # A book about JSON is a book. Removing its field names would damage the
        # corpus to fix a problem it does not have.
        cleaned, counts = scrub.scrub(self.PAYLOAD)
        self.assertEqual(cleaned, self.PAYLOAD)
        self.assertEqual(scrub.total(counts), 0)

    def test_real_source_code_is_never_judged_a_payload(self):
        code = 'def f(x):\n    return {"name": x, "value": 2}\n'
        self.assertFalse(scrub.looks_like_payload(code))

    def test_a_json_snippet_in_prose_is_not_judged_a_payload(self):
        prose = (
            'The file looked like this:\n\n{"name": "a", "value": 1}\n\n'
            "and that was all."
        )
        self.assertFalse(scrub.looks_like_payload(prose))

    def test_a_document_full_of_field_markers_is_judged_a_payload(self):
        self.assertTrue(scrub.looks_like_payload(self.PAYLOAD * 4))

    def test_a_single_field_mention_is_not_a_payload(self):
        # One occurrence in a document about JSON is prose that mentions one.
        self.assertFalse(scrub.looks_like_payload('He saw a "role": field once.'))


class ProseIsNotContaminationTests(unittest.TestCase):
    """Ordinary writing, including quoted speech, must survive."""

    def test_quoted_speech_is_untouched(self):
        text = 'He said "the answer is: forty-two" and left.'
        cleaned, _ = scrub.scrub(text)
        self.assertEqual(cleaned, text)

    def test_a_conversation_is_untouched(self):
        text = "Alice: hello there\nBob: hi\n"
        cleaned, _ = scrub.scrub(text)
        self.assertEqual(cleaned, text)

    def test_prose_with_braces_is_untouched(self):
        text = "A set is written {1, 2, 3} in the usual way."
        cleaned, _ = scrub.scrub(text)
        self.assertEqual(cleaned, text)


class BannerRemovalTests(unittest.TestCase):
    """A record banner is structure and not language."""

    def test_a_banner_line_is_removed(self):
        cleaned, counts = scrub.scrub("--- SAMPLE 0 ---\nthe text")
        self.assertEqual(cleaned, "the text")
        self.assertEqual(counts["banners"], 1)

    def test_every_banner_in_a_shard_is_removed(self):
        body = "--- SAMPLE 0 ---\na\n--- SAMPLE 1 ---\nb\n--- SAMPLE 2 ---\nc\n"
        cleaned, counts = scrub.scrub(body)
        self.assertNotIn("SAMPLE", cleaned)
        self.assertEqual(counts["banners"], 3)


class ScrubFileTests(unittest.TestCase):
    """The file-level entry point, which is what the entry point calls."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.folder, ignore_errors=True)
        )

    def write(self, name, text):
        target = self.folder / name
        target.write_text(text, encoding="utf-8")
        return target

    def test_a_mangled_payload_is_rewritten(self):
        path = self.write("bad.txt", '{"content": "a\\nb"}' * 6)
        result = scrub.scrub_file(path)
        self.assertGreater(result["removed"], 0)
        self.assertNotIn("\\n", path.read_text(encoding="utf-8"))

    def test_a_clean_file_is_left_byte_identical(self):
        # Rewriting a file that needed nothing changes its hash, which would
        # silently invalidate anything keyed on it.
        original = "The observatory stood on the granite above the valley.\n"
        path = self.write("clean.txt", original)
        result = scrub.scrub_file(path)
        self.assertEqual(result["removed"], 0)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_latex_is_left_byte_identical(self):
        original = "Let \\(a,b\\) satisfy \\leq with \\to and \\times.\n" * 20
        path = self.write("math.txt", original)
        scrub.scrub_file(path)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_an_unreadable_file_is_reported_rather_than_truncated(self):
        result = scrub.scrub_file(self.folder / "absent.txt")
        self.assertEqual(result["removed"], 0)
        self.assertIn("could not be read", result["note"])


class CensusTests(unittest.TestCase):
    """The dry run, which must not write anything."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.folder, ignore_errors=True)
        )

    def test_a_census_reports_without_writing(self):
        target = self.folder / "bad.txt"
        target.write_text('{"content": "a\\nb"}' * 6, encoding="utf-8")
        before = target.read_text(encoding="utf-8")
        rows = scrub.census([target])
        self.assertGreater(rows[0]["removed"], 0)
        self.assertEqual(target.read_text(encoding="utf-8"), before)

    def test_the_worst_offender_is_listed_first(self):
        heavy = self.folder / "heavy.txt"
        heavy.write_text('{"content": "a\\nb"}' * 20, encoding="utf-8")
        light = self.folder / "light.txt"
        light.write_text("just some prose about a valley.\n", encoding="utf-8")
        rows = scrub.census([light, heavy])
        self.assertEqual(rows[0]["path"], str(heavy))


if __name__ == "__main__":
    unittest.main()
