"""Tests for external knowledge and the retention decision (Architecture 28).

Architecture 28 is explicit that external information "should not automatically
become permanent internal knowledge" and that "the system should evaluate
whether information deserves retention". So the tests are mostly about the
refusals. A retriever that cannot say no is a way of filling a knowledge layer
with whatever it happened to be shown next.
"""

import unittest

import control
import external
import store as store_module

from tests.support import TempDirTestCase, make_config

HELD = {"widget", "gadget", "sprocket", "valve", "cog", "bearing"}


class TermsTests(unittest.TestCase):
    def test_significant_words_are_kept(self):
        self.assertIn("valve", external.terms("the valve was replaced"))

    def test_stopwords_and_short_words_are_dropped(self):
        found = external.terms("the a of and it was to a valve")
        self.assertIn("valve", found)
        self.assertNotIn("the", found)
        self.assertNotIn("and", found)


class FileSourceTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.docs = self.workdir / "docs"
        self.docs.mkdir()
        (self.docs / "valves.md").write_text(
            "A valve controls flow. The spring returns the valve to its seat.",
            encoding="utf-8",
        )
        (self.docs / "unrelated.txt").write_text(
            "Photosynthesis converts light into sugar inside a chloroplast.",
            encoding="utf-8",
        )
        self.source = external.FileSource("docs", self.docs)

    def test_a_matching_document_is_found(self):
        found = self.source.fetch("how does a valve control flow")
        self.assertTrue(found)
        self.assertIn("valve", found[0]["matched"])

    def test_a_partial_overlap_is_a_weak_hit_not_a_miss(self):
        # Lexical retrieval returning a partial match is correct behaviour, and
        # suppressing it would hide that the source answered partially. The
        # score says how weakly.
        found = self.source.fetch("photosynthesis and chloroplasts")
        self.assertEqual(len(found), 1)
        self.assertLess(found[0]["score"], 1.0)
        self.assertIn("photosynthesis", found[0]["matched"])

    def test_a_query_with_no_overlap_returns_nothing(self):
        self.assertEqual(self.source.fetch("zzzz qqqq wwww"), [])

    def test_a_result_carries_an_excerpt_that_can_be_judged(self):
        found = self.source.fetch("valve spring")
        self.assertTrue(found[0]["excerpt"].strip())
        self.assertIn("valve", found[0]["excerpt"].lower())

    def test_a_binary_file_is_skipped_rather_than_decoded(self):
        # Decoding something that is not text produces plausible nonsense, and a
        # retriever that hands that to a knowledge layer is worse than one that
        # returns nothing.
        (self.docs / "blob.md").write_bytes(b"\x00\x01\x02 not text at all")
        for result in self.source.fetch("valve flow control", limit=20):
            self.assertNotIn("blob", result["reference"])

    def test_a_missing_location_is_not_an_error(self):
        self.assertEqual(
            external.FileSource("gone", self.workdir / "nowhere").fetch("x"), []
        )


class RetentionTests(unittest.TestCase):
    def test_on_topic_material_is_worth_keeping(self):
        keep, reason = external.worth_keeping(
            {"matched": ["valve", "gadget", "cog"], "score": 0.8}, HELD
        )
        self.assertTrue(keep)
        self.assertIn("shared terms", reason)

    def test_material_unrelated_to_what_is_held_is_refused(self):
        # Interesting to a reader, and not a thing to become part of the model.
        keep, reason = external.worth_keeping(
            {"matched": ["photosynthesis", "chloroplast"], "score": 0.9}, HELD
        )
        self.assertFalse(keep)
        self.assertIn("already holds", reason)

    def test_a_single_shared_term_is_too_thin(self):
        keep, reason = external.worth_keeping(
            {"matched": ["valve"], "score": 0.9}, HELD
        )
        self.assertFalse(keep)

    def test_a_low_score_is_refused(self):
        keep, reason = external.worth_keeping(
            {"matched": ["valve", "gadget", "cog"], "score": 0.05}, HELD
        )
        self.assertFalse(keep)
        self.assertIn("floor", reason)

    def test_something_already_held_is_refused(self):
        keep, reason = external.worth_keeping(
            {"matched": ["valve", "gadget"], "score": 0.8, "reference": "x.md"},
            HELD,
            existing={"x.md"},
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "already held")

    def test_a_result_with_no_recorded_terms_cannot_be_judged(self):
        keep, _reason = external.worth_keeping({"matched": [], "score": 1.0}, HELD)
        self.assertFalse(keep)


class RetainTests(TempDirTestCase):
    def test_retained_material_is_filed_for_the_gate_not_written_as_knowledge(self):
        # Architecture 28: it must not become permanent internal knowledge.
        # It goes into the raw corpus, where curation and the acceptance gate
        # decide whether it is worth anything.
        store = store_module.Store(self.workdir / "knowledge.json")
        result = {"source": "docs", "reference": "docs/valves.md", "excerpt": "a valve"}
        target = external.retain(store, result, "on topic and new")
        self.assertTrue(target.exists())
        body = target.read_text(encoding="utf-8")
        self.assertIn("docs/valves.md", body)
        self.assertIn("a valve", body)
        self.assertEqual(store.all("concepts"), [])

    def test_two_retentions_from_one_source_do_not_collide(self):
        store = store_module.Store(self.workdir / "knowledge.json")
        result = {"source": "docs", "reference": "docs/valves.md", "excerpt": "x"}
        first = external.retain(store, result, "first")
        second = external.retain(store, result, "second")
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists() and second.exists())


class ConfigurationTests(TempDirTestCase):
    def test_nothing_is_configured_by_default(self):
        # Reaching outside the corpus is a decision, not a default.
        self.assertEqual(make_config().external_sources, ())

    def test_a_string_source_becomes_a_file_source(self):
        (self.workdir / "docs").mkdir()
        c = make_config(external_sources=(str(self.workdir / "docs"),))
        found = external.sources(c)
        self.assertEqual(len(found), 1)
        self.assertIsInstance(found[0], external.FileSource)

    def test_an_mcp_source_becomes_a_client(self):
        c = make_config(
            external_sources=(
                {"kind": "mcp", "name": "agent", "endpoint": "http://127.0.0.1:9/"},
            )
        )
        found = external.sources(c)
        self.assertIsInstance(found[0], external.McpClient)
        self.assertEqual(found[0].endpoint, "http://127.0.0.1:9/")

    def test_an_unreachable_source_does_not_fail_the_question(self):
        # One dead source is not a reason to abandon the question; it is simply
        # not there, and the caller is not given an error it cannot act on.
        #
        # The external grant is given because reading outside the process is
        # Architecture 38 external access and is refused without it. A refused
        # fetch raises rather than returning nothing, so this test can only
        # reach the behaviour it is about once the gate has been passed.
        c = make_config(
            external_sources=(
                {"kind": "mcp", "name": "dead", "endpoint": "http://127.0.0.1:9/"},
            )
        )
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"external"}
        self.addCleanup(setattr, control, "granted_domains", original)
        self.assertEqual(external.fetch(c, "anything"), [])

    def test_the_census_names_what_is_not_implemented(self):
        note = external.census(make_config())["note"]
        self.assertIn("not implemented", note)
        self.assertIn("retention", note)


if __name__ == "__main__":
    unittest.main()
