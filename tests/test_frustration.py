"""Tests for learning frustration (Architecture 21).

Architecture 21 says the signal must feed architectural analysis, so the tests
check the two things that matter: that a source which keeps failing scores
high, and that a source which never fails does not. A frustration measure that
rises for everything is not a measure.
"""

import unittest

import frustration
from frustration import HIGH, LOW, MEDIUM

from tests.support import TempDirTestCase, make_config


class IndexTests(unittest.TestCase):
    def test_an_unjudged_source_is_not_frustrated(self):
        self.assertEqual(frustration.index_for(0, 0, 0, 0.0), 0.0)
        self.assertEqual(frustration.band(0.0), LOW)

    def test_frequent_failure_scores_high(self):
        self.assertEqual(frustration.band(frustration.index_for(10, 1, 9, 0.0)), HIGH)

    def test_consistent_success_is_not_frustrated(self):
        self.assertEqual(frustration.band(frustration.index_for(10, 10, 0, 1.0)), LOW)

    def test_improvement_discounts_the_error_rate(self):
        # Same failure count, but one case gained a lot when it worked.
        poor = frustration.index_for(10, 1, 9, 0.0)
        productive = frustration.index_for(10, 1, 9, 5.0)
        self.assertLess(productive, poor)

    def test_the_index_is_bounded(self):
        for attempts in (1, 5, 100):
            for errors in range(attempts + 1):
                value = frustration.index_for(attempts, attempts - errors, errors, 0.0)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_bands_are_ordered(self):
        self.assertEqual(frustration.band(frustration.MEDIUM_AT - 0.01), LOW)
        self.assertEqual(frustration.band(frustration.MEDIUM_AT), MEDIUM)
        self.assertEqual(frustration.band(frustration.HIGH_AT), HIGH)


class SourceTests(unittest.TestCase):
    def test_an_unknown_source_is_reported_as_unjudged(self):
        result = frustration.for_source({}, "missing.txt")
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["band"], LOW)
        self.assertIn("not been judged", result["note"])

    def test_counts_come_from_the_confidence_record(self):
        state = {
            "doc.txt": {
                "observed": 8,
                "accepted": 2,
                "contradictions": 6,
                "confidence": 0.1,
            }
        }
        result = frustration.for_source(state, "doc.txt", improvement=0.0)
        self.assertEqual(result["attempts"], 8)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["errors"], 6)
        self.assertEqual(result["band"], HIGH)

    def test_counts_can_be_supplied_directly_for_a_live_cycle(self):
        result = frustration.from_counts(9, 1, 8, improvement=0.0, label="doc.txt")
        self.assertEqual(result["label"], "doc.txt")
        self.assertEqual(result["attempts"], 9)
        self.assertEqual(result["band"], HIGH)

    def test_a_reliable_source_is_not_frustrated(self):
        state = {
            "doc.txt": {
                "observed": 8,
                "accepted": 8,
                "contradictions": 0,
                "confidence": 0.9,
            }
        }
        self.assertEqual(frustration.for_source(state, "doc.txt")["band"], LOW)

    def test_a_malformed_state_does_not_crash(self):
        self.assertEqual(
            frustration.for_source({"doc": "nonsense"}, "doc")["attempts"], 0
        )


class ConceptTests(unittest.TestCase):
    def test_a_seen_and_understood_concept_is_not_frustrated(self):
        record = {
            "identifier": "x",
            "evidence": 6,
            "contradictions": 0,
            "confidence": 0.9,
        }
        self.assertEqual(frustration.for_concept(record)["band"], LOW)

    def test_a_contradicted_concept_scores_high(self):
        record = {
            "identifier": "x",
            "evidence": 1,
            "contradictions": 7,
            "confidence": 0.1,
        }
        self.assertEqual(frustration.for_concept(record)["band"], HIGH)
        self.assertAlmostEqual(
            frustration.for_concept(record)["index"], 0.7875, places=3
        )

    def test_the_index_is_reconstructible_from_the_record(self):
        record = {
            "identifier": "x",
            "evidence": 3,
            "contradictions": 1,
            "confidence": 0.4,
        }
        self.assertAlmostEqual(frustration.for_concept(record)["index"], 0.15, places=4)

    def test_a_seen_but_unlearned_concept_is_not_frustrated(self):
        # Seen often, never contradicted, still low confidence. That is not
        # repeated difficulty, it is not learned yet, and the concept's own
        # confidence is what reports it. Conflating the two would send the
        # expansion logic after capacity when the answer is more data.
        record = {
            "identifier": "x",
            "evidence": 5,
            "contradictions": 0,
            "confidence": 0.2,
        }
        self.assertEqual(frustration.for_concept(record)["band"], LOW)


class CauseTests(unittest.TestCase):
    def test_no_attempts_means_no_cause_is_claimed(self):
        self.assertEqual(
            frustration.likely_cause({"attempts": 0, "errors": 0}),
            "not enough evidence to say",
        )

    def test_every_cause_the_system_produces_is_a_declared_cause(self):
        # The report advertised three causes no run could ever emit and omitted
        # one that every new source produced, because the tuple and the branches
        # were written at different times and nothing compared them. Sweeping
        # the inputs is what makes the mismatch visible rather than theoretical.
        produced = set()
        for attempts in range(0, 12):
            for errors in range(0, attempts + 1):
                for improvement in (-1.0, 0.0, 1.0):
                    produced.add(
                        frustration.likely_cause(
                            {
                                "attempts": attempts,
                                "errors": errors,
                                "improvement": improvement,
                            }
                        )
                    )
        self.assertEqual(
            produced - set(frustration.CAUSES),
            set(),
            "a cause the system returns must be one it declared",
        )

    def test_the_no_evidence_cause_is_declared(self):
        # Returned for every source that has never been judged, and absent from
        # the declared list, so causes_considered under-reported what the system
        # actually does.
        self.assertIn(frustration.NO_EVIDENCE, frustration.CAUSES)

    def test_causes_the_system_cannot_evidence_are_named_as_such(self):
        # Rather than inventing a measurement to make the list look complete: an
        # expansion proposal has to name a cause it can evidence.
        self.assertTrue(frustration.UNEVIDENCED_CAUSES)
        for cause in frustration.UNEVIDENCED_CAUSES:
            self.assertIn(cause, frustration.CAUSES)

    def test_a_source_curation_refused_is_insufficient_data(self):
        # It was one of three declared causes nothing could produce, and the
        # evidence was already there: the curation report names the files it
        # refused as too short, duplicated or too large to read.
        self.assertEqual(
            frustration.likely_cause({"starved": True, "attempts": 0, "errors": 0}),
            "insufficient data",
        )
        self.assertNotIn("insufficient data", frustration.UNEVIDENCED_CAUSES)

    def test_starvation_outranks_the_count_based_hypotheses(self):
        # A source that never entered the corpus has not been attempted, so no
        # error rate describes it and the count-based branch is meaningless.
        self.assertEqual(
            frustration.likely_cause({"starved": True, "attempts": 40, "errors": 39}),
            "insufficient data",
        )

    def test_a_healthy_source_is_not_called_starved(self):
        self.assertEqual(
            frustration.likely_cause({"starved": False, "attempts": 10, "errors": 0}),
            "inadequate context",
        )

    def test_a_reliable_source_is_not_blamed_on_capacity(self):
        self.assertEqual(
            frustration.likely_cause({"attempts": 6, "errors": 0}),
            "inadequate context",
        )

    def test_heavy_failure_points_at_capacity(self):
        self.assertEqual(
            frustration.likely_cause({"attempts": 10, "errors": 9}),
            "insufficient model capacity",
        )

    def test_moderate_failure_points_at_representation(self):
        self.assertEqual(
            frustration.likely_cause({"attempts": 10, "errors": 5}),
            "poor representation",
        )


class ImprovementTests(TempDirTestCase):
    def test_improvement_is_averaged_per_source(self):
        import experiment

        self.config = make_config(
            experiment_log=str(self.workdir / "experiments.jsonl")
        )
        for before, after in ((3.0, 1.0), (2.0, 1.5)):
            experiment.write_experiment(
                self.config,
                cycle=1,
                source="doc.txt",
                cycle_result={
                    "holdout_loss_before": before,
                    "holdout_loss_after": after,
                    "final_learning_rate": 3e-4,
                    "micro_steps": 4,
                    "gradient_clip": 1.0,
                    "gradient_stop": 25.0,
                    "successful_steps": 4,
                    "gradient_recoveries": 0,
                },
                verdict="ACCEPT",
                reason="measured",
                regression_before=1.0,
                regression_after=1.0,
                confidence=0.5,
                model_cycle=1,
                resources={},
            )
        improvements = frustration.improvement_by_source(self.config)
        self.assertAlmostEqual(improvements["doc.txt"], 1.25)

    def test_an_empty_log_reports_nothing(self):
        self.config = make_config(experiment_log=str(self.workdir / "absent.jsonl"))
        self.assertEqual(frustration.improvement_by_source(self.config), {})


class ReportTests(TempDirTestCase):
    def test_the_report_lists_sources_and_concepts_with_causes(self):
        self.config = make_config(
            knowledge_store=str(self.workdir / "knowledge.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
        )
        state = {
            "doc.txt": {
                "observed": 9,
                "accepted": 1,
                "contradictions": 8,
                "confidence": 0.05,
            }
        }
        result = frustration.report(self.config, state=state)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["band"], HIGH)
        self.assertIn("likely_cause", result["sources"][0])
        self.assertEqual(len(result["high_frustration"]), 1)
        self.assertIn("not a diagnosis", result["note"])
        self.assertIn("insufficient model capacity", result["causes_considered"])
