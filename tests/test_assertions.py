"""Tests for the assertion layer (Architecture 14 steps 6 and 7).

Step 6 was built earlier and is correct, but it never fired. Across twelve real
cycles every promoted term ended up both supporting and contradicted by the same
document, so the decision read it as a context difference every time, correctly.
The input was the problem: a contradiction was a bare count on a term, and term
co-occurrence cannot say that one source denies what another asserts.

These tests are therefore mostly about the representation. A disagreement has to
be expressible, two sources have to be found making it, and a source disagreeing
with itself must not be counted as two sources.

They are also mostly negative, for the same reason the expansion tests were. A
fabricated contradiction is worse than no contradiction, because it is a
fabricated disagreement between two real sources.
"""

import unittest

import assertions
import relationships as relationships_module
import store as store_module

from tests.support import TempDirTestCase

DENIES = "the river is not safe. the survey is safe. the pump requires power. "
ASSERTS = "the river is safe. the survey is not safe. the pump requires power. "


class ExtractionTests(unittest.TestCase):
    def test_a_negation_is_read_as_a_negation(self):
        found = assertions.extract(DENIES)
        denials = [r for r in found if r["polarity"] == assertions.NEGATIVE]
        self.assertTrue(denials, found)
        self.assertIn(
            ("river", "is not", "safe"),
            [(r["subject"], r["relation"], r["object"]) for r in denials],
        )

    def test_an_assertion_is_read_as_an_assertion(self):
        found = assertions.extract(ASSERTS)
        self.assertIn(
            ("river", "is", "safe"),
            [(r["subject"], r["relation"], r["object"]) for r in found],
        )

    def test_different_connectives_stay_different_claims(self):
        # "X requires Y" and "X is not Y" are different statements. Collapsing
        # them would invent disagreements nobody stated.
        found = assertions.extract("the pump requires power. ")
        self.assertIn("requires", [r["relation"] for r in found])

    def test_a_document_with_no_claims_yields_nothing(self):
        self.assertEqual(assertions.extract("the wind moved over the valley. "), [])

    def test_extraction_is_bounded(self):
        # A long document contains thousands of these shapes; reading all of
        # them is how a corpus fills with claims nobody made.
        text = "the river is not safe. " * 500
        self.assertLessEqual(len(assertions.extract(text)), assertions.MAX_PER_SOURCE)

    def test_a_sentence_making_no_claim_of_itself_is_dropped(self):
        found = assertions.extract("the river is the river. ")
        self.assertEqual(found, [])


class PropositionTests(unittest.TestCase):
    def test_a_negation_collides_with_its_assertion(self):
        # This collision is the entire module. Without it two sources can never
        # be found to disagree, however plainly they do.
        self.assertEqual(
            assertions.claim_key(
                {"subject": "river", "relation": "is", "object": "safe"}
            ),
            assertions.claim_key(
                {"subject": "river", "relation": "is not", "object": "safe"}
            ),
        )

    def test_different_propositions_stay_apart(self):
        self.assertNotEqual(
            assertions.claim_key(
                {"subject": "river", "relation": "is", "object": "safe"}
            ),
            assertions.claim_key(
                {"subject": "river", "relation": "requires", "object": "power"}
            ),
        )


class DisagreementTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def edges(self):
        return [
            record
            for record in self.store.collection("relationships").values()
            if record.get("relation") == relationships_module.CONTRADICTS
        ]

    def test_two_sources_making_the_same_claim_corroborate(self):
        assertions.harvest(
            self.store, "doc-a.txt", "the pump requires power. " * 8, cycle=1
        )
        result = assertions.harvest(
            self.store, "doc-b.txt", "the pump requires power. " * 8, cycle=2
        )
        self.assertEqual(result["counts"]["corroborated"], 1)
        self.assertEqual(result["counts"]["contradicted"], 0)
        self.assertEqual(assertions.contradictions(self.store), [])

    def test_two_sources_making_opposite_claims_contradict(self):
        assertions.harvest(self.store, "doc-a.txt", DENIES * 8, cycle=1)
        result = assertions.harvest(self.store, "doc-b.txt", ASSERTS * 8, cycle=2)
        self.assertEqual(result["counts"]["contradicted"], 2)
        disputes = assertions.contradictions(self.store)
        self.assertEqual(len(disputes), 2)
        for dispute in disputes:
            self.assertIn("doc-a.txt", dispute["supported_by"])
            self.assertIn("doc-b.txt", dispute["contradicted_by"])

    def test_a_dispute_opens_an_edge_in_both_directions(self):
        # Architecture 14 step 7. This path was unreachable before, because
        # nothing had established that a conflict was real.
        assertions.harvest(self.store, "doc-a.txt", DENIES * 8, cycle=1)
        assertions.harvest(self.store, "doc-b.txt", ASSERTS * 8, cycle=2)
        pairs = {frozenset((e["from"], e["to"])) for e in self.edges()}
        self.assertIn(frozenset(("doc-a.txt", "doc-b.txt")), pairs)
        for edge in self.edges():
            self.assertNotEqual(
                edge["from"],
                edge["to"],
                "an edge from a source to itself is not a disagreement",
            )

    def test_a_source_contradicting_itself_is_not_a_dispute(self):
        # Architecture 14: some apparent conflicts are context differences.
        assertions.harvest(self.store, "doc-a.txt", DENIES * 8, cycle=1)
        result = assertions.harvest(self.store, "doc-a.txt", ASSERTS * 8, cycle=2)
        self.assertEqual(result["counts"]["self_contradiction"], 2)
        self.assertEqual(result["counts"]["contradicted"], 0)
        self.assertEqual(assertions.contradictions(self.store), [])
        self.assertEqual(self.edges(), [])

    def test_a_denying_source_is_not_listed_as_asserting(self):
        # Putting it in both lists made every genuine disagreement read as a
        # source agreeing with itself.
        assertions.harvest(self.store, "doc-a.txt", DENIES * 8, cycle=1)
        assertions.harvest(self.store, "doc-b.txt", ASSERTS * 8, cycle=2)
        for record in self.store.collection(assertions.ASSERTIONS).values():
            if record.get("contradicted_by"):
                self.assertFalse(
                    set(record["contradicted_by"]) & set(record["sources"]),
                    "a source that denied a claim cannot also assert it",
                )

    def test_a_third_source_taking_a_side_is_counted(self):
        # Two sources denying the same claim is corroboration of each other, and
        # a stronger position than one denial met with silence. Before this the
        # two cases were indistinguishable.
        assertions.harvest(self.store, "doc-a.txt", DENIES * 8, cycle=1)
        assertions.harvest(self.store, "doc-b.txt", ASSERTS * 8, cycle=2)
        result = assertions.harvest(self.store, "doc-c.txt", ASSERTS * 8, cycle=3)
        self.assertEqual(result["counts"]["corroborated_denial"], 2)
        for record in self.store.collection(assertions.ASSERTIONS).values():
            if record.get("contradicted_by"):
                self.assertEqual(record.get("contradicting"), 2)


class CensusTests(TempDirTestCase):
    def test_census_counts_what_is_held(self):
        store = store_module.Store(self.workdir / "knowledge.json")
        self.assertEqual(assertions.census(store)["total"], 0)
        assertions.harvest(store, "doc-a.txt", DENIES * 8, cycle=1)
        assertions.harvest(store, "doc-b.txt", ASSERTS * 8, cycle=2)
        census = assertions.census(store)
        self.assertEqual(census["total"], 3)
        self.assertEqual(census["disputed"], 2)
        self.assertEqual(census["corroborated"], 1)

    def test_the_census_explains_itself(self):
        store = store_module.Store(self.workdir / "knowledge.json")
        self.assertIn("polarity", assertions.census(store)["note"].lower())


if __name__ == "__main__":
    unittest.main()
