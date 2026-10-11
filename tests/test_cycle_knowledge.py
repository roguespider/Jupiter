"""Integration tests for the cycle-to-knowledge step.

Architecture 46 asks an experience to say what was attempted, what was known,
what changed, how performance moved and whether the update helped, and
Architecture 10 lists usage frequency and supporting experiences as concept
fields. These tests check that a completed cycle actually writes all of that,
rather than only the fields that were convenient to record.
"""

import unittest
from pathlib import Path

import acceptance
import concepts
import confidence
import learning_worker
import store as store_module

import memory
from tests.support import TempDirTestCase, make_config

# Two terms repeated often enough to survive candidate_terms, with everything
# else in the sentence drawn from the stopword list, so the promoted set is
# predictable.
LEARNABLE_TEXT = "the database and the index and the database and the index " * 8


def routed_score(decision):
    """The score a routing decision was made on, for comparison in tests."""
    return decision["score"]


class ConfidenceRecoveryTests(TempDirTestCase):
    """Architecture 13: confidence updates as further evidence becomes available.

    The review floor and the confidence update had become mutually exclusive. A
    source below the floor is classified REVIEW, and REVIEW used to leave
    confidence exactly as it was, so the only verdict that could raise it --
    ACCEPT -- stayed out of reach for the rest of the run. Confidence stopped
    being a measurement of evidence and became a latch, and a source that had
    demonstrably stopped damaging the reserved document could never be trusted
    again.

    Measured on a real corpus: holdout loss fell by 1.5 to 4.3 against a defer
    band of 0.0001, and six cycles in a row were classified REVIEW because
    confidence sat at 0.258 against a floor of 0.35. No ACCEPT followed, so no
    published checkpoint was ever written and the conversation could not start.
    """

    SOURCE = "source-1"

    def setUp(self):
        super().setUp()
        self.config = make_config(
            confidence_state=str(self.workdir / "confidence.json")
        )

    def feed(self, verdict, times=1, starting=0.0, observed=0):
        c = self.config
        state = {self.SOURCE: {"confidence": starting, "observed": observed}}
        record = state[self.SOURCE]
        for _ in range(times):
            record = confidence.observe(c, state, self.SOURCE, verdict)
        return record

    def test_a_review_raises_confidence(self):
        record = self.feed(acceptance.REVIEW, times=3)
        self.assertGreater(record["confidence"], 0.0)

    def test_a_review_counts_as_evidence(self):
        record = self.feed(acceptance.REVIEW, times=3)
        self.assertEqual(record["evidence"], 3)
        self.assertEqual(record["observed"], 3)

    def test_a_review_is_weaker_evidence_than_an_accept(self):
        reviewed = self.feed(acceptance.REVIEW, times=3)["confidence"]
        accepted = self.feed(acceptance.ACCEPT, times=3)["confidence"]
        self.assertLess(reviewed, accepted)

    def test_a_source_below_the_floor_can_climb_back_over_it(self):
        # The defect itself. A source already under the floor, which is where a
        # rejection leaves it, has to be able to earn trust again.
        c = self.config
        record = self.feed(acceptance.REVIEW, times=40, starting=0.258, observed=17)
        self.assertGreaterEqual(
            record["confidence"],
            c.confidence_min,
            "a source reviewed repeatedly without a contradiction must eventually "
            "clear the review floor",
        )

    def test_climbing_back_over_the_floor_makes_acceptance_reachable(self):
        c = self.config
        verdicts = []
        state = {self.SOURCE: {"confidence": 0.258, "observed": 17, "accepted": 2}}
        for _ in range(40):
            record = confidence.observe(c, state, self.SOURCE, acceptance.REVIEW)
            verdict, _reason = acceptance.classify(
                c,
                improvement=0.5,
                regression_delta=0.0,
                regression_available=True,
                confidence=record["confidence"],
                observed=record["observed"],
            )
            verdicts.append(verdict)
        self.assertIn(
            acceptance.ACCEPT,
            verdicts,
            "a measured improvement must become ACCEPTable once the source has "
            "earned its way back over the floor",
        )

    def test_repeated_rejections_still_hold_a_source_down(self):
        # The floor is only escapable by clean observation. A source that keeps
        # damaging the reserved document is rejected, not reviewed, and a
        # rejection subtracts, so it must not be able to climb.
        c = self.config
        record = self.feed(acceptance.REJECT, times=30, starting=0.9, observed=10)
        self.assertLess(record["confidence"], c.confidence_min)

    def test_a_defer_is_still_not_evidence_either_way(self):
        # Nothing moved by more than the dead band, so there is no measurement
        # to record and confidence must not drift on its own. The evidence key
        # is never created, which is the point: a defer records nothing.
        record = self.feed(acceptance.DEFER, times=5)
        self.assertEqual(record["confidence"], 0.0)
        self.assertEqual(record.get("evidence", 0), 0)


class CycleKnowledgeTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(
            knowledge_store=str(self.workdir / "knowledge.json"),
            concept_terms_per_cycle=5,
        )
        self.knowledge = store_module.Store(self.config.knowledge_store)
        self.source_path = Path("data/curated/doc_a.txt")

    def cycle_result(self, before=2.0, after=1.0):
        return {
            "holdout_loss_before": before,
            "holdout_loss_after": after,
            "holdout_loss_improvement": before - after,
            "successful_steps": 12,
            "gradient_recoveries": 0,
        }

    def record(
        self,
        cycle=1,
        learned=True,
        verdict="ACCEPT",
        regression_delta=0.1,
        confidence=0.44,
    ):
        return learning_worker.record_cycle_knowledge(
            self.knowledge,
            self.config,
            cycle,
            self.source_path,
            LEARNABLE_TEXT,
            self.cycle_result(),
            holdout_before=2.0,
            regression_delta=regression_delta,
            confidence=confidence,
            learning_rate=3e-4,
            learned=learned,
            verdict=verdict,
            verdict_reason="because the measurement said so",
        )

    def test_the_experience_names_the_concepts_the_cycle_exercised(self):
        promoted, experience = self.record()
        self.assertIn("database", promoted)
        self.assertIn("index", promoted)
        self.assertEqual(experience["concepts"], promoted)

    def test_the_experience_records_the_measured_values(self):
        _, experience = self.record()
        self.assertEqual(experience["attempted"], str(self.source_path))
        self.assertEqual(experience["holdout_loss_before"], 2.0)
        self.assertEqual(experience["holdout_loss_after"], 1.0)
        self.assertEqual(experience["performance_change"], 1.0)
        self.assertEqual(experience["regression_change"], 0.1)
        self.assertEqual(experience["confidence"], 0.44)
        self.assertEqual(experience["update"]["successful_steps"], 12)
        self.assertEqual(experience["update"]["gradient_recoveries"], 0)
        self.assertEqual(experience["update"]["learning_rate"], 3e-4)
        self.assertTrue(experience["helped"])
        self.assertEqual(experience["decision"], "ACCEPT")
        self.assertEqual(experience["reason"], "because the measurement said so")

    def test_a_rejected_cycle_records_that_it_did_not_help(self):
        promoted, experience = self.record(learned=False, verdict="REJECT")
        self.assertFalse(experience["helped"])
        self.assertEqual(experience["decision"], "REJECT")
        for identifier in promoted:
            record = concepts.get(self.knowledge, identifier)
            self.assertEqual(record["evidence"], 0)
            self.assertEqual(record["contradictions"], 1)
            self.assertEqual(record["confidence"], 0.0)

    def test_concepts_are_no_longer_empty_shells(self):
        promoted, _ = self.record()
        for identifier in promoted:
            record = concepts.get(self.knowledge, identifier)
            self.assertNotEqual(record["description"], "")
            self.assertEqual(record["usage_frequency"], 1)
            self.assertEqual(record["supporting_experiences"], ["cycle-1"])
            self.assertEqual(record["sources"], ["doc_a.txt"])
            self.assertEqual(record["first_cycle"], 1)

    def test_a_description_says_where_the_evidence_came_from(self):
        self.record()
        description = concepts.get(self.knowledge, "database")["description"]
        self.assertIn("database", description)
        self.assertIn("doc_a.txt", description)
        self.assertIn("cycle 1", description)
        self.assertIn("not what the term means", description)

    def test_repeat_learning_counts_usage_and_keeps_the_first_cycle(self):
        first, _ = self.record(cycle=1)
        second, _ = self.record(cycle=2)
        self.assertEqual(first, second)
        record = concepts.get(self.knowledge, "database")
        self.assertEqual(record["usage_frequency"], 2)
        self.assertEqual(record["first_cycle"], 1)
        self.assertEqual(record["supporting_experiences"], ["cycle-1", "cycle-2"])
        self.assertEqual(record["sources"], ["doc_a.txt"])
        self.assertEqual(record["evidence"], 2)

    def test_census_does_not_call_unrelated_concepts_duplicates(self):
        promoted, _ = self.record()
        report = concepts.census(self.knowledge)
        self.assertEqual(report["total"], len(promoted))
        self.assertEqual(
            report["duplicate"],
            0,
            "provenance descriptions must stay distinct per concept",
        )

    def test_the_experience_records_the_architecture_16_routing_decision(self):
        _, experience = self.record()
        routing = experience["routing"]
        self.assertIn("destination", routing)
        self.assertIn("reasons", routing)
        self.assertIn(routing["destination"], memory.DESTINATIONS)

    def test_routing_keeps_a_source_external_when_nothing_argues_otherwise(self):
        # Three of the nine Architecture 16 factors are measurable today. The
        # rest are zero, which means no evidence, so the biased policy answer
        # is external -- and the missing factors are visible in the record.
        routing = memory.route_for_source(observed=3, accepted=3, confidence=0.4)
        self.assertEqual(routing["destination"], memory.DEST_EXTERNAL)
        self.assertTrue(routing["reasons"])

    def test_a_contradicting_source_is_routed_external_harder(self):
        clean = memory.route_for_source(observed=4, accepted=4, contradictions=0)
        contradicted = memory.route_for_source(observed=4, accepted=1, contradictions=3)
        self.assertLess(routed_score(contradicted), routed_score(clean))
        self.assertEqual(contradicted["destination"], memory.DEST_EXTERNAL)

    def test_working_memory_holds_the_active_document_then_is_cleared(self):
        # Architecture 15: working memory is what the current task needs, so it
        # has a lifecycle. The durable record of the cycle is the experience,
        # not the working copy.
        self.record()
        counts = memory.census(self.knowledge)
        self.assertEqual(counts[memory.WORKING], 0, "working memory is task-scoped")
        self.assertEqual(counts[memory.LONG_TERM], 1)
        self.assertEqual(counts[memory.EXPERIENCE], 1)

    def test_long_term_memory_records_the_verdict(self):
        self.record()
        stored = memory.long_term(self.knowledge)
        self.assertEqual(len(stored), 1)
        entry = stored[0]
        self.assertIn("doc_a.txt", entry["content"])
        self.assertIn("ACCEPT", entry["content"])
        self.assertEqual(entry["confidence"], 0.44)

    def test_the_census_states_that_count_is_not_a_score(self):
        # Architecture 11 treats concept count as a measurement, and the census
        # has to say so in its own output rather than only in a docstring.
        report = concepts.census(self.knowledge)
        self.assertIn("not a score", report["note"])
        self.assertEqual(
            report["strongly_connected"],
            [],
            "no relationship exists yet, so strong connectivity is empty",
        )


if __name__ == "__main__":
    unittest.main()
