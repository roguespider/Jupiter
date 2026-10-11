"""Unit tests for the decision layer.

Covers Architecture 20 (update acceptance), 13 (confidence), 31 (experiment
records and review) and 33 (regression detection).
"""

import json
import unittest

import acceptance
import confidence as confidence_module
import experiment
import regression

from config import ARCHITECTURE_VERSION
from tests.support import TempDirTestCase, make_config


class AcceptanceTests(unittest.TestCase):
    """Architecture 20: a proposed update is not automatically accepted."""

    def setUp(self):
        self.config = make_config(
            regression_tolerance=0.01, defer_band=1e-4, confidence_min=0.35
        )

    def classify(self, **overrides):
        arguments = {
            "improvement": 0.5,
            "regression_delta": 0.0,
            "regression_available": True,
            "confidence": 0.8,
            "observed": 5,
        }
        arguments.update(overrides)
        return acceptance.classify(self.config, **arguments)

    def test_clear_improvement_is_accepted(self):
        verdict, reason = self.classify()
        self.assertEqual(verdict, acceptance.ACCEPT)
        self.assertIn("improved", reason)

    def test_forgetting_beyond_tolerance_is_rejected(self):
        verdict, reason = self.classify(regression_delta=-0.4)
        self.assertEqual(verdict, acceptance.REJECT)
        self.assertIn("regressed", reason)

    def test_forgetting_inside_tolerance_still_accepts(self):
        verdict, _ = self.classify(regression_delta=-0.005)
        self.assertEqual(verdict, acceptance.ACCEPT)

    def test_worsening_holdout_is_rejected(self):
        verdict, _ = self.classify(improvement=-0.5)
        self.assertEqual(verdict, acceptance.REJECT)

    def test_movement_inside_the_dead_band_is_deferred_not_guessed(self):
        verdict, reason = self.classify(improvement=1e-6)
        self.assertEqual(verdict, acceptance.DEFER)
        self.assertIn("dead band", reason)

    def test_missing_yardstick_is_review_not_accept(self):
        verdict, reason = self.classify(regression_available=False)
        self.assertEqual(verdict, acceptance.REVIEW)
        self.assertIn("yardstick", reason)

    def test_low_confidence_after_repetition_is_review(self):
        # The observation count has to clear the grace period, which is what
        # stops a source being judged on a record too short to mean anything.
        # At two this was circular: a source held at REVIEW never accrued the
        # accepts it needed to clear the floor.
        verdict, reason = self.classify(confidence=0.1, observed=14)
        self.assertEqual(verdict, acceptance.REVIEW)
        self.assertIn("confidence", reason)

    def test_a_source_within_its_grace_period_is_judged_on_this_cycle(self):
        # The point of the grace period. A source with two observations and no
        # confidence has no history to be unconfident about, so holding it at
        # REVIEW is holding it for the thing it is meant to be measuring.
        verdict, reason = self.classify(confidence=0.0, observed=2)
        self.assertEqual(verdict, acceptance.ACCEPT)
        self.assertNotIn("confidence", reason)

    def test_measured_forgetting_outranks_a_low_confidence_floor(self):
        # The ordering this pins: forgetting was measured, so the update can be
        # judged. Reporting REVIEW would hide a measured regression behind a
        # request for more observation, and REVIEW does not roll the update
        # back the way REJECT does.
        verdict, reason = self.classify(
            regression_delta=-0.4, confidence=0.05, observed=9
        )
        self.assertEqual(verdict, acceptance.REJECT)
        self.assertIn("regressed", reason)

    def test_a_confident_update_with_forgetting_is_still_rejected(self):
        verdict, _ = self.classify(regression_delta=-0.4, confidence=0.95, observed=9)
        self.assertEqual(verdict, acceptance.REJECT)

    def test_low_confidence_without_forgetting_is_still_review(self):
        verdict, reason = self.classify(
            regression_delta=0.0, confidence=0.05, observed=14
        )
        self.assertEqual(verdict, acceptance.REVIEW)
        self.assertIn("confidence", reason)

    def test_every_verdict_the_module_declares_is_reachable(self):
        # ROLLBACK is produced by the worker's instability path, not by the
        # quality classifier, which is why it has no branch here.
        self.assertIn(acceptance.ROLLBACK, acceptance.VERDICTS)
        produced = {
            self.classify(**case)[0]
            for case in (
                {"improvement": 0.5},
                {"improvement": 0.5, "regression_delta": -0.4},
                {"improvement": 1e-6},
                {"regression_available": False},
                {"confidence": 0.05, "observed": 9},
            )
        }
        self.assertEqual(
            produced,
            {
                acceptance.ACCEPT,
                acceptance.REJECT,
                acceptance.DEFER,
                acceptance.REVIEW,
            },
        )

    def test_first_observations_are_graced(self):
        verdict, _ = self.classify(confidence=0.0, observed=1)
        self.assertEqual(verdict, acceptance.ACCEPT)


class ConfidenceTests(TempDirTestCase):
    """Architecture 13: confidence is evidence strength, not a truth value."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            confidence_state=str(self.workdir / "confidence.json"),
            confidence_gain=0.25,
            confidence_loss=0.20,
        )
        self.state = {}

    def test_a_blank_source_carries_no_evidence(self):
        blank = confidence_module.blank()
        self.assertEqual(blank["confidence"], 0.0)
        self.assertEqual(blank["evidence"], 0)
        self.assertIsNone(blank["last_verified"])

    def test_accept_raises_confidence_and_records_evidence(self):
        record = confidence_module.observe(
            self.config, self.state, "doc", acceptance.ACCEPT
        )
        self.assertAlmostEqual(record["confidence"], 0.25)
        self.assertEqual(record["evidence"], 1)
        self.assertEqual(record["accepted"], 1)
        self.assertEqual(record["observed"], 1)

    def test_reject_and_rollback_lower_confidence(self):
        # confidence_module.observe returns the live record inside the state
        # dict, so each value is read back fresh instead of being held in a
        # variable that later mutations would silently rewrite.
        confidence_module.observe(self.config, self.state, "doc", acceptance.ACCEPT)
        accepted = confidence_module.value_of(
            confidence_module.entry_for(self.state, "doc")
        )
        confidence_module.observe(self.config, self.state, "doc", acceptance.REJECT)
        rejected = confidence_module.value_of(
            confidence_module.entry_for(self.state, "doc")
        )
        confidence_module.observe(self.config, self.state, "doc", acceptance.ROLLBACK)
        rolled = confidence_module.entry_for(self.state, "doc")
        self.assertLess(rejected, accepted)
        self.assertLess(rolled["confidence"], rejected)
        self.assertEqual(rolled["contradictions"], 2)
        self.assertEqual(rolled["evidence"], 1)

    def test_a_defer_carries_no_evidence_either_way(self):
        # The update moved by less than the dead band, so nothing was measured
        # and there is no evidence to record in either direction.
        deferred = confidence_module.observe(
            self.config, self.state, "doc", acceptance.DEFER
        )
        self.assertEqual(deferred["confidence"], 0.0)
        self.assertEqual(deferred["evidence"], 0)
        self.assertEqual(deferred["observed"], 1)

    def test_a_review_is_evidence_weakly(self):
        # A review is an observation of the source, not a contradiction of it.
        # Architecture 13 says the system can update confidence as further
        # evidence becomes available, and the review floor withheld exactly the
        # verdict that used to raise confidence, so a review that recorded
        # nothing left a sub-floor source permanently untrustable.
        reviewed = confidence_module.observe(
            self.config, self.state, "doc", acceptance.REVIEW
        )
        self.assertGreater(reviewed["confidence"], 0.0)
        self.assertEqual(reviewed["evidence"], 1)

    def test_state_persists_and_reloads(self):
        confidence_module.observe(self.config, self.state, "doc", acceptance.ACCEPT)
        self.assertIn("doc", confidence_module.load(self.config))

    def test_corrupt_state_resets_instead_of_stopping_learning(self):
        (self.workdir / "confidence.json").write_text("{{{", encoding="utf-8")
        self.assertEqual(confidence_module.load(self.config), {})

    def test_value_of_is_defensive_about_bad_stored_data(self):
        self.assertEqual(confidence_module.value_of({"confidence": "nonsense"}), 0.0)
        self.assertEqual(confidence_module.value_of({"confidence": 5.0}), 1.0)


class RegressionTests(TempDirTestCase):
    """Architecture 33: the reserved sample is the fixed yardstick.

    It was one document, which is the shape Architecture 33 warns about in
    section 52: a single reserved document cannot detect forgetting that damages
    the reserved and learned documents equally. Measured over 200 cycles here,
    101 updates improved the holdout while moving that one document, and 79 were
    discarded for it. The reservation is now a stratified sample, so
    degradation is measured against a cross-section rather than a single file.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(
            regression_state=str(self.workdir / "regression_set.json"),
            regression_source="",
        )

    def pieces(self):
        return [
            (self.write("a.txt", "alpha text"), "alpha text"),
            (self.write("b.txt", "beta text"), "beta text"),
        ]

    def test_reserving_leaves_something_learnable(self):
        reserved, remaining = regression.reserve(self.config, self.pieces())
        self.assertIsInstance(reserved, list)
        self.assertGreaterEqual(len(reserved), 1)
        # With two documents the yardstick takes one and the other stays
        # learnable; reserving both would leave the worker nothing to do.
        self.assertEqual(len(reserved) + len(remaining), 2)
        self.assertGreaterEqual(len(remaining), 1)
        names = {path.name for path in reserved}
        self.assertNotIn(remaining[0][0].name, names)

    def test_a_corpus_of_many_documents_reserves_a_sample_not_one_file(self):
        pieces = [
            (
                self.write(f"doc{index:02d}.txt", f"text {index} " * (index + 1)),
                f"text {index}",
            )
            for index in range(9)
        ]
        reserved, remaining = regression.reserve(self.config, pieces)
        self.assertGreater(
            len(reserved),
            1,
            "one document out of nine is not a representative yardstick",
        )
        self.assertGreaterEqual(len(remaining), 1)

    def test_the_reserved_sample_spans_document_sizes(self):
        # Stratified by size, so the largest and smallest documents are both
        # represented. A yardstick drawn only from one size misses a class of
        # regression entirely.
        pieces = [
            (self.write(f"doc{index:02d}.txt", "x " * (index + 1) * 40), "x")
            for index in range(9)
        ]
        reserved, _ = regression.reserve(self.config, pieces)
        sizes = sorted(path.stat().st_size for path in reserved)
        self.assertGreater(len(sizes), 1)
        self.assertNotEqual(sizes[0], sizes[-1])

    def test_a_single_document_reservation_is_replaced_by_a_real_sample(self):
        # A project that already reserved one document must actually get the
        # fix, or the change would only ever apply to a fresh checkout. The
        # replaced document is not deleted; it goes back into the candidate
        # list, which is the whole difference between reserving and removing.
        single = self.write("a.txt", "alpha")
        self.write("regression_set.json", json.dumps({"reserved": str(single)}))
        # load_reserved returns None when it cannot read the state, so the
        # assertion is on the length of what it actually returned rather than
        # on the value being non-None by assumption.
        recorded = regression.load_reserved(self.config)
        # load_reserved returns None when it cannot read the state, so the
        # value is narrowed here rather than assumed. assertIsNotNone does not
        # narrow for a reader, and a later len() on the same call is exactly
        # the shape that breaks when the state file is unreadable.
        if recorded is None:
            self.fail("the single-document reservation was not readable")
        self.assertEqual(len(recorded), 1)

        pieces = [
            (single, "alpha"),
            (self.write("b.txt", "beta"), "beta"),
            (self.write("c.txt", "gamma"), "gamma"),
            (self.write("d.txt", "delta"), "delta"),
            (self.write("e.txt", "epsilon"), "epsilon"),
            (self.write("f.txt", "zeta"), "zeta"),
            (self.write("g.txt", "eta"), "eta"),
        ]
        reserved, remaining = regression.reserve(self.config, pieces)
        self.assertGreater(
            len(reserved),
            1,
            "a single-document reservation must be replaced by a sample",
        )
        # Nothing is deleted, whichever documents the sample settles on. The
        # corpus is the accumulated result of everything Jupiter has learned, and
        # a document removed from it is material no later run can recover.
        # Which document returns to learning is a sampling decision and is not
        # asserted here; the invariant is that every file survives untouched.
        before = {path: path.read_text(encoding="utf-8") for path, _ in pieces}
        for path, text in before.items():
            self.assertTrue(path.exists(), f"{path.name} was removed by reserving")
            self.assertEqual(path.read_text(encoding="utf-8"), text)
        self.assertGreater(
            len(remaining),
            1,
            "the yardstick must leave enough to keep selecting between",
        )

    def test_an_existing_stratified_sample_is_not_moved(self):
        # A yardstick that moves between runs stops being one, so an existing
        # sample of the right shape is honoured as it stands.
        pieces = [
            (self.write(f"doc{index:02d}.txt", "x " * (index + 1) * 40), "x")
            for index in range(9)
        ]
        first, _ = regression.reserve(self.config, pieces)
        second, _ = regression.reserve(self.config, pieces)
        self.assertEqual(first, second)

    def test_reservation_never_deletes_or_moves_a_document(self):
        # The corpus is the accumulated result of everything Jupiter has ever
        # learned. Reserving means withholding from the candidate list; the file
        # stays exactly where it is.
        pieces = self.pieces()
        before = {path: path.read_text(encoding="utf-8") for path, _ in pieces}
        regression.reserve(self.config, pieces)
        for path, text in before.items():
            self.assertTrue(path.exists(), f"{path} was removed by reserving")
            self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_the_reservation_persists_across_calls(self):
        first, _ = regression.reserve(self.config, self.pieces())
        second, _ = regression.reserve(self.config, self.pieces())
        self.assertEqual(first, second)
        self.assertEqual(regression.load_reserved(self.config), first)

    def test_a_configured_source_wins(self):
        self.config.regression_source = str(self.workdir / "b.txt")
        reserved, remaining = regression.reserve(self.config, self.pieces())
        self.assertEqual([path.name for path in reserved], ["b.txt"])
        self.assertEqual([path.name for path, _ in remaining], ["a.txt"])

    def test_a_missing_configured_source_is_refused(self):
        self.config.regression_source = str(self.workdir / "absent.txt")
        with self.assertRaises(FileNotFoundError):
            regression.reserve(self.config, self.pieces())

    def test_reserving_the_last_document_is_refused(self):
        single = [(self.write("only.txt", "text"), "text")]
        with self.assertRaises(RuntimeError):
            regression.reserve(self.config, single)

    def test_an_unreadable_reservation_reports_none_so_the_worker_re_reserves(self):
        (self.workdir / "regression_set.json").write_text("not json", encoding="utf-8")
        self.assertIsNone(regression.load_reserved(self.config))

    def test_a_reservation_whose_document_vanished_is_replaced(self):
        # A previous run reserved a document that has since been moved, renamed
        # or deleted. Honouring that reservation returned a path the worker
        # could not find among its candidates, and the worker then died on a
        # KeyError naming an internal detail rather than the document that was
        # missing. The yardstick has to be a document that is actually there.
        reserved, _ = regression.reserve(self.config, self.pieces())
        for path in reserved:
            path.unlink()
        chosen, _ = regression.reserve(self.config, self.pieces())
        self.assertTrue(
            all(path.exists() for path in chosen), "the yardstick must be readable"
        )
        self.assertTrue(all(path.name in ("a.txt", "b.txt") for path in chosen))

    def test_a_vanished_reservation_does_not_silently_keep_learning_everything(self):
        # The replaced reservation must still be held out of learning, or the
        # document being measured would also be the document being trained on.
        reserved, _ = regression.reserve(self.config, self.pieces())
        for path in reserved:
            path.unlink()
        pieces = self.pieces()
        chosen, remaining = regression.reserve(self.config, pieces)
        learned = {path.resolve() for path, _ in remaining}
        for path in chosen:
            self.assertNotIn(path.resolve(), learned)


class ExperimentRecordTests(TempDirTestCase):
    """Architecture 31: every cycle leaves a record someone else could read."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            experiment_log=str(self.workdir / "experiments.jsonl")
        )

    def record(self, verdict=acceptance.ACCEPT):
        return experiment.write_experiment(
            self.config,
            cycle=1,
            source="doc.txt",
            cycle_result={
                "holdout_loss_before": 2.0,
                "holdout_loss_after": 1.0,
                "final_learning_rate": 3e-4,
                "micro_steps": 10,
                "gradient_clip": 1.0,
                "gradient_stop": 25.0,
                "successful_steps": 10,
                "gradient_recoveries": 0,
            },
            verdict=verdict,
            reason="because",
            regression_before=3.0,
            regression_after=2.9,
            confidence=0.5,
            model_cycle=1,
            resources={"cpu": {}},
        )

    def test_the_architecture_version_is_a_version_and_not_a_filename(self):
        # This field held the string "Jupiter.architecture.json" for long
        # enough that nobody reading an experiment record could tell which
        # architecture produced it. The test above only ever checked that the
        # field existed, so the wrong value passed it every time. Architecture
        # 30 asks a saved instance to determine the architecture, and
        # Architecture 43 asks that the versions be separate dimensions.
        record = self.record()
        self.assertEqual(record["architecture_version"], ARCHITECTURE_VERSION)
        self.assertNotIn(".json", record["architecture_version"])

    def test_the_model_version_names_the_architecture_it_was_built_for(self):
        record = self.record()
        self.assertIn(ARCHITECTURE_VERSION, record["model_version"])
        self.assertNotEqual(
            record["model_version"],
            f"checkpoint-cycle-{record['learning_cycle']}",
            "a model version that is only a cycle count carries no version",
        )

    def test_a_tensor_version_is_recorded_and_changes_with_the_layout(self):
        # Architecture 30 names a tensor version among the things a saved
        # instance must be able to determine. Nothing recorded one at all.
        first = self.record()
        self.assertTrue(first["tensor_version"].startswith("tensors-"))
        wider = make_config(n_embd=self.config.n_embd + 64)
        self.assertNotEqual(
            experiment.tensor_version(self.config),
            experiment.tensor_version(wider),
            "a different tensor layout must produce a different tensor version",
        )

    def test_record_carries_the_architecture_31_fields(self):
        record = self.record()
        for field in (
            "experiment_id",
            "timestamp",
            "model_version",
            "architecture_version",
            "input",
            "learning_method",
            "hyperparameters",
            "before_metrics",
            "update",
            "after_metrics",
            "decision",
        ):
            self.assertIn(field, record)

    def test_record_measures_both_sides_of_the_update(self):
        record = self.record()
        self.assertEqual(record["before_metrics"]["holdout_loss"], 2.0)
        self.assertEqual(record["after_metrics"]["holdout_loss"], 1.0)
        self.assertEqual(record["before_metrics"]["regression_loss"], 3.0)
        self.assertEqual(record["after_metrics"]["regression_loss"], 2.9)

    def test_a_rejected_cycle_is_kept_as_readily_as_an_accepted_one(self):
        self.record(verdict=acceptance.REJECT)
        stored = experiment.read_experiments(self.config.experiment_log)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["decision"], acceptance.REJECT)

    def test_reading_skips_an_unreadable_line_instead_of_failing(self):
        self.record()
        with open(self.config.experiment_log, "a", encoding="utf-8") as handle:
            handle.write("{truncated\n")
        self.assertEqual(
            len(experiment.read_experiments(self.config.experiment_log)), 1
        )

    def test_reading_a_missing_log_is_empty_not_an_error(self):
        self.assertEqual(
            experiment.read_experiments(str(self.workdir / "absent.jsonl")), []
        )

    def test_review_counts_each_verdict(self):
        summary = experiment.review(
            [
                {"source": "a", "decision": acceptance.ACCEPT, "regressed": False},
                {"source": "a", "decision": acceptance.REJECT, "regressed": True},
                {"source": "b", "decision": acceptance.DEFER, "regressed": False},
                {"source": "c", "decision": acceptance.REVIEW, "regressed": False},
            ]
        )
        self.assertEqual(summary["cycles"], 4)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["deferred"], 1)
        self.assertEqual(summary["flagged_for_review"], 1)
        self.assertEqual(summary["cycles_with_regression"], 1)
        self.assertEqual(summary["distinct_sources"], 3)

    def test_describe_review_renders_every_count(self):
        text = experiment.describe_review(
            {
                "cycles": 4,
                "accepted": 1,
                "rejected": 1,
                "deferred": 1,
                "flagged_for_review": 1,
                "cycles_with_regression": 1,
                "distinct_sources": 3,
            }
        )
        self.assertIn("4 cycles", text)
        self.assertIn("3 sources", text)
