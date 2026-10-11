"""Tests for what a relationship endpoint is allowed to be (Architecture 12).

A relationship joins concepts. Architecture 14 and the assertion layer also join
sources, because a disagreement is between two documents. Both are correct, and
the two used to be indistinguishable: an edge whose endpoint was a source name
had a concept invented for it, and since curation names every curated file by the
sha256 of its contents, the concept table filled with sixty-eight character hex
strings.

Measured on the live store before this was fixed: 268 such concepts, all with
zero evidence, and 2,276 CONTRADICTS edges, 22% of the whole graph. They could
never be recalled, because the only terms a sha256 contains are the fragments the
tokenizer cut out of it.

These tests are therefore about the boundary. An edge between two documents is
recorded, and neither document becomes a concept. An edge between two terms is
recorded, and both terms are real. Neither case invents a concept to hold the
edge, because a concept is made when something is learned about a term and not
when an edge merely mentions a name.
"""

import unittest

import assertions
import concepts
import relationships as relationships_module
import store as store_module

from tests.support import TempDirTestCase

# Shaped like a curated filename, which is what curation actually produces:
# the sha256 of the document's contents, with a .txt suffix.
DIGEST_NAME = "00c203359a1bd7a005507ef0a3a677ad0d9aa645125035fd373e14455cccdabf.txt"
OTHER_DIGEST = "2dc00e2bb28a218d89c34a48917a2703b6e1abb0f2d39fcf8011bd2bd54f6900.txt"

ASSERTS = "the river is safe. the survey is not safe. "
DENIES = "the river is not safe. the survey is safe. "


def digest_shaped(name):
    """Whether a name is a curated filename rather than a term.

    Only used by these tests, to assert that a digest did not become a concept.
    Not a filter on the pipeline: the pipeline does not need it, because the
    pipeline never receives a filename as a term in the first place.
    """
    text = str(name).lower()
    if text.endswith(".txt"):
        text = text[:-4]
    return len(text) >= 32 and all(c in "0123456789abcdef" for c in text)


class SourceEndpointsAreNotConceptsTests(TempDirTestCase):
    """The regression: a conflict between documents must not create concepts."""

    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_an_edge_between_two_sources_invents_no_concept(self):
        relationships_module.observe(
            self.store,
            DIGEST_NAME,
            relationships_module.CO_OCCURS,
            OTHER_DIGEST,
            endpoints=relationships_module.SOURCE_ENDPOINTS,
        )
        held = concepts.all_concepts(self.store)
        self.assertEqual(
            [r for r in held if digest_shaped(r.get("identifier"))],
            [],
            "a source name became a concept",
        )

    def test_the_edge_itself_is_still_recorded(self):
        """Refusing to invent a concept must not lose the measurement."""
        relationships_module.observe(
            self.store,
            DIGEST_NAME,
            relationships_module.CO_OCCURS,
            OTHER_DIGEST,
            endpoints=relationships_module.SOURCE_ENDPOINTS,
        )
        edges = relationships_module.all_relationships(self.store)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["from"], DIGEST_NAME.lower())
        self.assertEqual(edges[0]["to"], OTHER_DIGEST.lower())

    def test_a_conflict_between_documents_invents_no_concept(self):
        """The assertion layer end to end, which is where the damage came from."""
        for source, text in ((DIGEST_NAME, ASSERTS), (OTHER_DIGEST, DENIES)):
            for record in assertions.extract(text):
                assertions.observe(self.store, source, {**record, "cycle": 1})

        linked = 0
        for claim in assertions.contradictions(self.store):
            if DIGEST_NAME in claim.get("supported_by", []) or DIGEST_NAME in (
                claim.get("contradicted_by", [])
            ):
                if assertions.link(self.store, claim):
                    linked += 1

        self.assertTrue(linked, "the two documents were never found to disagree")
        held = concepts.all_concepts(self.store)
        self.assertEqual(
            [r for r in held if digest_shaped(r.get("identifier"))],
            [],
            "a conflict between two documents created a concept for one of them",
        )

    def test_a_contradiction_is_visible_from_both_sides(self):
        """Both directions are still written, so the conflict is findable."""
        relationships_module.observe(
            self.store,
            DIGEST_NAME,
            relationships_module.CONTRADICTS,
            OTHER_DIGEST,
            supporting=False,
            endpoints=relationships_module.SOURCE_ENDPOINTS,
        )
        forward = relationships_module.all_relationships(self.store)
        keys = {(e["from"], e["relation"], e["to"]) for e in forward}
        self.assertIn(
            (
                DIGEST_NAME.lower(),
                relationships_module.CONTRADICTS,
                OTHER_DIGEST.lower(),
            ),
            keys,
        )
        self.assertIn(
            (
                OTHER_DIGEST.lower(),
                relationships_module.CONTRADICTS,
                DIGEST_NAME.lower(),
            ),
            keys,
        )


class TermEndpointsTests(TempDirTestCase):
    """Concept endpoints keep working, and only real concepts are touched."""

    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_co_occurrence_between_real_terms_still_registers_on_both(self):
        concepts.observe(self.store, "river", delta_evidence=1)
        concepts.observe(self.store, "flood", delta_evidence=1)
        relationships_module.record_co_occurrence(self.store, ["river", "flood"])

        for term in ("river", "flood"):
            record = self.store.get(concepts.CONCEPTS, term)
            self.assertIsNotNone(record)
            self.assertTrue(
                record["relationships"],
                f"{term} lost the edge that mentions it",
            )

    def test_an_edge_to_an_unknown_term_does_not_invent_it(self):
        """A name nobody has learned anything about is not yet a concept.

        Creating it here is what let a filename become a node. The concept is
        made by concepts.observe, when there is evidence for it.
        """
        relationships_module.observe(
            self.store,
            "river",
            relationships_module.CO_OCCURS,
            "neverseen",
        )
        self.assertIsNone(
            self.store.get(concepts.CONCEPTS, "neverseen"),
            "an edge endpoint with no evidence behind it became a concept",
        )

    def test_a_learned_term_is_still_registered_by_a_later_edge(self):
        concepts.observe(self.store, "river", delta_evidence=1)
        relationships_module.observe(
            self.store,
            "river",
            relationships_module.CO_OCCURS,
            "flood",
        )
        record = self.store.get(concepts.CONCEPTS, "river")
        self.assertTrue(record["relationships"])

    def test_prerequisites_are_still_recorded_on_a_real_concept(self):
        concepts.observe(self.store, "pump", delta_evidence=1)
        concepts.observe(self.store, "power", delta_evidence=1)
        relationships_module.observe(
            self.store,
            "pump",
            relationships_module.DEPENDS_ON,
            "power",
        )
        record = self.store.get(concepts.CONCEPTS, "pump")
        self.assertIn("power", record["prerequisites"])


class EndpointDeclarationTests(unittest.TestCase):
    """The declaration itself, so the two kinds cannot be confused by accident."""

    def test_the_two_endpoint_kinds_are_distinct_values(self):
        self.assertNotEqual(
            relationships_module.CONCEPT_ENDPOINTS,
            relationships_module.SOURCE_ENDPOINTS,
        )

    def test_concepts_are_the_default(self):
        """Derived relationships are about terms, so the default must be terms."""
        signature = relationships_module.observe.__defaults__
        self.assertIn(
            relationships_module.CONCEPT_ENDPOINTS,
            signature,
            "observe() should default to concept endpoints",
        )


if __name__ == "__main__":
    unittest.main()
