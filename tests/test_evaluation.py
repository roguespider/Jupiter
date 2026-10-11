"""Tests for the evaluation surface (Architecture 34).

These check two things: that the metrics compute what they claim to, and that
the categories Jupiter cannot measure are reported as unmeasured rather than
quietly counted as covered. A category listed with no measurement behind it is
worse than an absent one.
"""

import contextlib
import io
import json
import unittest
from pathlib import Path

import control
import evaluation
import expansion
import experiment
import growth
import language_eval
import store as store_module
import torch
import train
from model import Jupiter, ModelConfig

import memory
from tests.support import TempDirTestCase, make_config

DEGENERATE = "the the the the the the the the the the the the the the"
VARIED = "one two three four five six seven eight nine ten eleven twelve"


class ConsoleOutputTests(unittest.TestCase):
    """What the entry point actually prints (Architecture 35).

    The evaluation report is measured per concept, and the frustration census
    reaches several hundred rows once a corpus has been learned. Printing all
    of it produced around three thousand lines in a burst. On a Windows console
    a burst that size is enough to trip the terminal's selection mode, which
    pauses the display, so a run that had already finished looked like one that
    had hung. These tests hold the console bounded and the detail on disk.
    """

    def big_report(self, concepts=165, high=129):
        return {
            "coverage": dict(evaluation.COVERAGE),
            "measured_categories": list(evaluation.MEASURED),
            "language": {"samples": 0, "note": "no generated text to measure"},
            "knowledge": {"note": "lexical only"},
            "memory": {"note": "reopened store"},
            "learning": {"note": "negative is an improvement"},
            "stability": {"note": "negative means worse"},
            "frustration": {
                "concepts": [
                    {"concept": f"c{n}", "index": 0.9, "band": "HIGH"}
                    for n in range(concepts)
                ],
                "high_frustration": [{"concept": f"c{n}"} for n in range(high)],
                "sources": [],
                "note": "measured",
                "causes_considered": ["inadequate context"],
            },
        }

    def printed(self, result):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            evaluation.summarise(result, "data/evaluation_report.json")
        return buffer.getvalue()

    def test_the_summary_does_not_grow_with_the_report(self):
        # The property that matters: console size is bounded by the number of
        # categories, not by how many concepts have been learned.
        small = len(self.printed(self.big_report(concepts=1, high=1)).splitlines())
        large = len(self.printed(self.big_report()).splitlines())
        self.assertEqual(
            small,
            large,
            "a larger corpus must not produce a longer console summary",
        )

    def test_a_large_report_prints_far_less_than_it_contains(self):
        result = self.big_report()
        printed = self.printed(result)
        dumped = len(json.dumps(result, indent=2).splitlines())
        self.assertLess(
            len(printed.splitlines()),
            dumped / 10,
            f"printed {len(printed.splitlines())} lines for a {dumped}-line report",
        )

    def test_the_summary_states_which_categories_are_covered(self):
        printed = self.printed(self.big_report())
        self.assertIn("measured", printed)
        for name in evaluation.COVERAGE:
            self.assertIn(name, printed)

    def test_the_summary_reports_the_census_counts(self):
        printed = self.printed(self.big_report(concepts=165, high=129))
        self.assertIn("165 concepts", printed)
        self.assertIn("129 above the high band", printed)

    def test_the_summary_says_where_the_detail_was_written(self):
        printed = self.printed(self.big_report())
        self.assertIn("data/evaluation_report.json", printed)

    def test_a_report_with_no_frustration_section_still_summarises(self):
        result = self.big_report()
        result.pop("frustration")
        printed = self.printed(result)
        self.assertIn("Frustration", printed)


class TrendTests(unittest.TestCase):
    """Whether a number is growing, shrinking, or standing still.

    This is the measurement the report was missing. Everything else here
    describes one cycle or one probe; none of it answers whether the system is
    getting better, which is the question a person running a learning system
    opened the report to ask.
    """

    def test_a_falling_loss_is_improving(self):
        values = [5.0, 4.6, 4.8, 4.0, 3.4, 3.0, 2.6, 2.2]
        result = evaluation.series_trend(values, lower_is_better=True, window=3)
        self.assertEqual(result["trend"], "IMPROVING")
        self.assertLess(result["last"], result["first"])
        self.assertLess(result["change_percent"], 0)

    def test_a_rising_loss_is_a_regression(self):
        values = [2.0, 2.4, 2.2, 2.8, 3.2, 3.0, 3.6, 3.9]
        result = evaluation.series_trend(values, lower_is_better=True, window=3)
        self.assertEqual(result["trend"], "REGRESSION")

    def test_a_rising_accuracy_is_improving(self):
        # The direction depends on whether lower is better, and getting that
        # backwards would report every gain as a loss.
        values = [0.10, 0.14, 0.12, 0.20, 0.26, 0.30, 0.34, 0.38]
        result = evaluation.series_trend(values, lower_is_better=False, window=3)
        self.assertEqual(result["trend"], "IMPROVING")

    def test_noise_is_reported_as_flat_rather_than_as_progress(self):
        # A metric that wobbles by 8% cycle to cycle has not improved by 1%.
        # Calling that growth is how a system reports improving for a hundred
        # cycles while going precisely nowhere.
        values = [2.0, 2.2, 1.8, 2.3, 1.9, 2.1, 1.7, 2.2, 1.8, 2.05]
        result = evaluation.series_trend(values, lower_is_better=True, window=3)
        self.assertEqual(result["trend"], "FLAT")
        self.assertGreater(result["noise_floor_percent"], abs(result["change_percent"]))

    def test_a_genuine_downward_drift_is_not_dismissed_as_noise(self):
        # The noise floor must not swallow a real trend: a steady 20% fall over
        # 20 cycles is improvement even though each individual step is small.
        values = [5.0 - (0.2 * index) for index in range(20)]
        result = evaluation.series_trend(values, lower_is_better=True, window=5)
        self.assertEqual(result["trend"], "IMPROVING")

    def test_too_few_measurements_refuse_to_report_a_direction(self):
        for count in (0, 1, 2, 3):
            result = evaluation.series_trend(
                [3.0] * count, lower_is_better=True, window=3
            )
            self.assertIsNone(result["trend"])
            self.assertEqual(result["samples"], count)

    def test_none_values_are_dropped_rather_than_counted_as_zero(self):
        # A missing measurement is not a measurement of zero. Treating it as
        # one would manufacture a collapse in the series.
        result = evaluation.series_trend(
            [5.0, None, 4.0, None, 3.0, 2.0, 1.5, 1.0], lower_is_better=True, window=2
        )
        self.assertEqual(result["samples"], 6)

    def test_the_two_windows_are_the_same_size(self):
        # A change must not be able to come from one end having more samples.
        values = [float(index) for index in range(21)]
        result = evaluation.series_trend(values, lower_is_better=True, window=5)
        self.assertEqual(result["window"], 5)

    def test_a_flat_zero_series_does_not_divide_by_zero(self):
        result = evaluation.series_trend([0.0] * 8, lower_is_better=True, window=3)
        self.assertIn(result["trend"], ("FLAT", "UNKNOWN"))


class TrendReportTests(TempDirTestCase):
    """The trend section as the entry point prints it."""

    def printed(self, result):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            evaluation.summarise(result, "data/evaluation_report.json")
        return buffer.getvalue()

    def base(self):
        return {
            "coverage": dict(evaluation.COVERAGE),
            "measured_categories": list(evaluation.MEASURED),
            "language": {"samples": 0, "note": "none"},
            "knowledge": {"note": "lexical"},
            "memory": {"note": "store"},
            "learning": {"note": "loss"},
            "stability": {"note": "reserved"},
            "expansion": {"note": "growth"},
            "frustration": {"concepts": [], "high_frustration": [], "sources": []},
            "trends": {
                "holdout_loss": {
                    "samples": 200,
                    "window": 25,
                    "first": 3.3875,
                    "last": 3.3787,
                    "change_percent": -0.26,
                    "trend": "IMPROVING",
                }
            },
        }

    def test_the_summary_leads_with_a_direction(self):
        printed = self.printed(self.base())
        self.assertIn("GROWTH AND REGRESSION", printed)
        self.assertIn("IMPROVING", printed)

    def test_the_two_means_are_shown_with_the_verdict(self):
        # A verdict of IMPROVING on a change of 0.26% is a verdict about
        # noise, and the reader can only see that if the numbers are there.
        printed = self.printed(self.base())
        self.assertIn("3.3875", printed)
        self.assertIn("3.3787", printed)
        self.assertIn("200 cycles", printed)

    def test_a_report_with_no_cycles_says_so_rather_than_printing_nothing(self):
        result = self.base()
        result["trends"] = {}
        printed = self.printed(result)
        self.assertIn("no cycles recorded yet", printed)

    def test_the_redundant_measured_list_is_not_printed_three_times(self):
        # The category names used to appear as a headline list, as a "measured"
        # line and as a coverage table. Three copies of the same nine names
        # told the reader nothing about direction.
        printed = self.printed(self.base())
        self.assertNotIn("Architecture 34 coverage:", printed)
        self.assertNotIn("  measured     :", printed)

    def test_unmeasured_categories_are_still_named(self):
        # Dropping the redundant list must not drop the honest coverage claim.
        result = self.base()
        result["coverage"] = dict(evaluation.COVERAGE)
        result["coverage"]["Language"] = "NOT measured yet"
        printed = self.printed(result)
        self.assertIn("not measured", printed)
        self.assertIn("Language", printed)


class LanguageProbeTests(TempDirTestCase):
    """The Language category has to be generated to be measured.

    evaluate_language took a list of samples that no caller in the project ever
    passed, so the Architecture 34 Language line read "no generated text to
    measure" on every run while the coverage table still claimed the category
    was covered. These hold the probe closed against that.
    """

    def config(self):
        # The field is regression_state, not regression_set: overriding a name
        # the Config does not have leaves the real out/ directory in play, and
        # the test then reads the project's own reserved document instead of the
        # temporary one it just wrote.
        return make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            out_dir=str(self.workdir / "out"),
            knowledge_store=str(self.workdir / "out" / "knowledge.json"),
            evaluation_report=str(self.workdir / "out" / "evaluation.json"),
            language_samples=str(self.workdir / "out" / "language_samples.txt"),
            regression_state=str(self.workdir / "out" / "regression_set.json"),
        )

    def seed_reserved(self, text=None, relative="curated/reserved.txt"):
        """Reserve a document the probe can draw prompts from.

        The path recorded is absolute, which is how the worker writes it: a
        relative one would resolve against whatever directory the process
        happened to start in, and the probe would then read a document that is
        not the reserved one.

        The default text varies paragraph by paragraph. A body made of one
        sentence repeated yields identical spans at every offset, which would
        make a distinctness test fail for a reason that has nothing to do with
        the probe.
        """
        absolute = self.workdir / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        if text is None:
            body = " ".join(
                f"Paragraph {index} describes the observatory on the granite above "
                f"the valley, and records what was measured there during season "
                f"{index} of the survey."
                for index in range(120)
            )
        else:
            body = text
        absolute.write_text(body, encoding="utf-8")
        self.write(
            "out/regression_set.json",
            json.dumps({"reserved": str(absolute), "reason": "test"}),
        )
        return absolute

    def test_prompts_come_from_the_reserved_document(self):
        self.seed_reserved()
        prompts = language_eval.load_prompts(self.config(), count=4)
        self.assertEqual(len(prompts), 4)
        self.assertTrue(all(prompt.strip() for prompt in prompts))

    def test_prompts_are_distinct(self):
        # The same span eight times would measure one prompt, not eight.
        self.seed_reserved()
        prompts = language_eval.load_prompts(self.config(), count=8)
        self.assertEqual(len(set(prompts)), len(prompts))

    def test_no_reserved_document_is_reported_not_silently_scored_zero(self):
        # "Nothing was measured" and "measured as perfect" are different claims
        # and only one of them is honest.
        with self.assertRaises(language_eval.LanguageUnavailable) as caught:
            language_eval.load_prompts(self.config())
        self.assertIn("reserved", str(caught.exception))

    def test_a_reserved_document_that_does_not_exist_is_reported(self):
        self.write(
            "out/regression_set.json",
            json.dumps({"reserved": str(self.workdir / "curated" / "gone.txt")}),
        )
        with self.assertRaises(language_eval.LanguageUnavailable):
            language_eval.load_prompts(self.config())

    def test_a_document_too_short_to_prompt_with_is_reported(self):
        self.seed_reserved(text="short")
        with self.assertRaises(language_eval.LanguageUnavailable) as caught:
            language_eval.load_prompts(self.config(), count=8)
        self.assertIn("too short", str(caught.exception))

    def test_a_prompt_is_asked_to_continue_something(self):
        text = language_eval.continuation("The observatory stood on granite above")
        self.assertIn("Write a short paragraph", text)

    def test_generated_text_is_kept_so_the_score_can_be_checked(self):
        samples = ["first generated text here", "second generated text here"]
        path = language_eval.save(self.config(), samples)
        self.assertTrue(path.exists())
        body = path.read_text(encoding="utf-8")
        self.assertIn("first generated text here", body)
        self.assertIn("second generated text here", body)

    def test_kept_text_is_read_back_intact(self):
        samples = ["alpha beta gamma delta", "epsilon zeta eta theta"]
        language_eval.save(self.config(), samples)
        self.assertEqual(language_eval.load_samples(self.config()), samples)

    def test_no_kept_text_reads_as_none_rather_than_empty(self):
        # An empty list would score as a perfect zero repetition rate.
        self.assertIsNone(language_eval.load_samples(self.config()))

    def test_a_probe_that_cannot_run_is_reported_not_scored(self):
        result = evaluation.evaluate_language(self.config())
        self.assertEqual(result["samples"], 0)
        self.assertIn("did not run", result["note"])

    def test_supplied_samples_are_scored_without_touching_a_checkpoint(self):
        # The path the report used before, kept working: a caller with text in
        # hand should not have to load a model to score it.
        result = evaluation.evaluate_language(
            self.config(), samples=["the the the the the the the the"]
        )
        self.assertEqual(result["samples"], 1)
        self.assertGreater(result["mean_repetition_rate"], 0)

    def test_varied_samples_score_a_low_repetition_rate(self):
        result = evaluation.evaluate_language(
            self.config(), samples=[VARIED, "one two three four five six"]
        )
        self.assertEqual(result["mean_repetition_rate"], 0.0)


class ReportFileTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(
            knowledge_store=str(self.workdir / "knowledge.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
            evaluation_report=str(self.workdir / "evaluation.json"),
        )

    def test_the_report_path_is_configured(self):
        self.assertTrue(make_config().evaluation_report.endswith(".json"))

    def test_main_writes_the_full_report_to_the_configured_path(self):
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(c=self.config, argv=[])
        target = Path(self.config.evaluation_report)
        self.assertTrue(target.exists(), "main must persist the report it prints")
        written = json.loads(target.read_text(encoding="utf-8"))
        self.assertIn("coverage", written)
        self.assertIn("frustration", written)

    def test_the_full_flag_still_emits_the_whole_report(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            evaluation.main(c=self.config, argv=["--full"])
        printed = buffer.getvalue()
        self.assertIn('"coverage"', printed)
        self.assertIn('"frustration"', printed)

    def test_the_default_console_output_is_bounded(self):
        # The whole point: main() must not flood the entry-point console.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            evaluation.main(c=self.config, argv=[])
        printed = buffer.getvalue()
        self.assertLess(
            len(printed.splitlines()),
            80,
            f"evaluation printed {len(printed.splitlines())} lines to the console",
        )


class RepetitionTests(unittest.TestCase):
    def test_looping_text_scores_high(self):
        self.assertGreater(evaluation.repetition_rate(DEGENERATE), 0.9)

    def test_varied_text_scores_low(self):
        self.assertEqual(evaluation.repetition_rate(VARIED), 0.0)

    def test_short_text_has_no_measurable_repetition(self):
        self.assertEqual(evaluation.repetition_rate("hi"), 0.0)
        self.assertEqual(evaluation.repetition_rate(""), 0.0)

    def test_the_measure_is_bounded(self):
        for text in (DEGENERATE, VARIED, "a b a b a b a b"):
            value = evaluation.repetition_rate(text)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class CoverageTests(unittest.TestCase):
    def test_every_architecture_34_category_is_listed(self):
        for name in (
            "Language",
            "Programming",
            "Knowledge",
            "Reasoning",
            "Memory",
            "Learning",
            "Stability",
            "Expansion",
        ):
            self.assertIn(name, evaluation.COVERAGE)

    def test_unmeasured_categories_say_so(self):
        # Every Architecture 34 category now has a measurement behind it.
        # Programming and Reasoning were measured by holdout loss on the class
        # of material they name, and Expansion by the outcome of the growth
        # decision expansion made. The check is kept rather than deleted, because
        # it is the one that stops a category being quietly marked measured with
        # nothing computing it: if a category is ever added without a
        # measurement, this is the test that has to be made to fail first.
        for name in evaluation.COVERAGE:
            if name not in evaluation.MEASURED:
                self.assertTrue(
                    evaluation.COVERAGE[name].startswith("NOT MEASURED"),
                    f"{name} is unmeasured and must say so rather than looking "
                    "like a category that was forgotten",
                )

    def test_measured_categories_exclude_the_unmeasured_ones(self):
        # The invariant that matters: nothing is measured that has no
        # measurement, and nothing with a measurement is left out of the list.
        self.assertEqual(set(evaluation.MEASURED), set(evaluation.COVERAGE))

    def test_a_measured_category_says_what_it_measured(self):
        # A category marked measured with nothing behind it is the failure the
        # module exists to prevent, so the description has to be a description
        # rather than a label.
        for name in evaluation.MEASURED:
            self.assertFalse(
                evaluation.COVERAGE[name].startswith("NOT"),
                f"{name} is marked measured and must say what it measured",
            )
            self.assertIn(" ", evaluation.COVERAGE[name].strip())


class ExpansionMeasurementTests(TempDirTestCase):
    """Architecture 34 Expansion, which had nothing to measure until recently.

    The category was reported as unmeasured because dynamic expansion did not
    exist. It exists now, and the point of measuring it is not to count growths
    but to report what the growth decision was worth, because Architecture 22
    requires expansion to be measurable rather than merely available.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(
            expansion_state=str(self.workdir / "expansion.json"),
            out_dir=str(self.workdir / "out"),
            # The knowledge store has to be redirected too. A test that calls
            # evaluation.report() with only the expansion paths redirected reads
            # the developer's real store, which is twenty-five megabytes of
            # records and roughly twenty minutes of retrieval probing. That one
            # test was ninety-seven per cent of the whole suite's runtime, and it
            # looked like a slow suite rather than a test pointed at production
            # data. Every path a report reads belongs in a fixture.
            knowledge_store=str(self.workdir / "knowledge.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
            evaluation_report=str(self.workdir / "evaluation.json"),
            confidence_state=str(self.workdir / "confidence.json"),
            selection_state=str(self.workdir / "selection.json"),
            n_layer=2,
        )
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)

    def short(self, source="doc-a.txt"):
        return [
            {
                "source": source,
                "likely_cause": expansion.CAPACITY_CAUSE,
                "index": 0.9,
            }
        ]

    def test_a_project_that_never_grew_says_so_rather_than_scoring_zero(self):
        # Zero growths measured as a score of zero would be indistinguishable
        # from a growth that was made and did nothing. The absence is the finding.
        result = evaluation.evaluate_expansion(self.config)
        self.assertEqual(result["proposed"], 0)
        self.assertNotIn("helped", result)
        self.assertIn("no capacity growth", result["note"])

    def test_a_growth_awaiting_authorisation_is_reported_as_awaiting(self):
        expansion.propose_growth(self.config, self.short(), cycle=1)
        result = evaluation.evaluate_expansion(self.config)
        self.assertEqual(result["proposed"], 1)
        self.assertNotIn("helped", result, "an unapproved growth has no verdict")
        self.assertIn("waiting", result["note"])

    def test_a_growth_that_lowered_the_loss_is_reported_as_helping(self):
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        expansion.resolve_growth(self.config, 3.0)
        result = evaluation.evaluate_expansion(self.config)
        self.assertTrue(result["helped"])
        self.assertEqual(result["mean_holdout_change"], -1.0)

    def test_a_growth_that_raised_the_loss_is_reported_as_not_helping(self):
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        expansion.resolve_growth(self.config, 5.0)
        result = evaluation.evaluate_expansion(self.config)
        self.assertFalse(result["helped"])
        self.assertEqual(result["mean_holdout_change"], 1.0)

    def test_a_growth_with_no_baseline_is_not_given_a_verdict(self):
        # A single number with nothing to compare against is not a measurement.
        # Reporting a verdict here would be the exact failure this module exists
        # to prevent, one level up from the category list.
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        expansion.resolve_growth(self.config, 3.0)
        result = evaluation.evaluate_expansion(self.config)
        self.assertNotIn("helped", result)
        self.assertNotIn("mean_holdout_change", result)
        self.assertIn("without a baseline", result["note"])

    def test_the_report_names_the_layer_change_it_judged(self):
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        expansion.resolve_growth(self.config, 3.0)
        result = evaluation.evaluate_expansion(self.config)
        self.assertEqual(result["current"]["from_layers"], 2)
        self.assertEqual(result["current"]["to_layers"], 3)

    def test_the_whole_report_carries_the_category(self):
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        expansion.resolve_growth(self.config, 3.0)
        result = evaluation.report(self.config)
        self.assertIn("expansion", result)
        self.assertIn("Expansion", result["measured_categories"])


class ImprovementArithmeticTests(unittest.TestCase):
    """The numbers the report prints have to mean what they say.

    A percentage is only worth reporting if the denominator is the thing a
    reader would choose, and only worth trusting if the sign survives. These
    pin the arithmetic, because a report that is confidently wrong is worse than
    one that reports nothing: the whole reason to print a percentage rather than
    a raw delta is that it is comparable across categories.
    """

    def test_an_improvement_is_negative_because_the_loss_went_down(self):
        # Loss dropping from 10 to 7 is an improvement, and the number stays
        # negative. Flipping it to +30% would break the comparison against every
        # other loss figure in the report.
        self.assertEqual(evaluation.improvement(10, 7), -30.0)

    def test_a_worsening_is_positive(self):
        self.assertEqual(evaluation.improvement(10, 13), 30.0)

    def test_no_change_is_zero(self):
        self.assertEqual(evaluation.improvement(4, 4), 0.0)

    def test_the_denominator_is_the_starting_value_not_the_final_one(self):
        # The share of the starting error that was removed. Using the final
        # value would give a different number for the same measurement.
        self.assertEqual(evaluation.improvement(100, 50), -50.0)

    def test_a_zero_baseline_has_no_percentage_rather_than_an_infinite_one(self):
        # Dividing by zero is how a report ends up claiming a 100% improvement
        # from a rounding error. There is no honest number here.
        self.assertIsNone(evaluation.improvement(0, 5))

    def test_a_missing_value_has_no_percentage(self):
        self.assertIsNone(evaluation.improvement(None, 5))
        self.assertIsNone(evaluation.improvement(5, None))

    def test_a_rate_is_a_percentage(self):
        self.assertEqual(evaluation.percent(16, 20), 80.0)

    def test_a_rate_with_no_total_has_no_percentage(self):
        self.assertIsNone(evaluation.percent(0, 0))
        self.assertIsNone(evaluation.percent(5, None))

    def test_a_rate_keeps_its_denominator(self):
        # 80% of 4 probes and 80% of 200 probes are very different claims, and
        # the percentage alone cannot tell them apart.
        measured = evaluation.rate(16, 20)
        self.assertEqual(measured["count"], 16)
        self.assertEqual(measured["total"], 20)
        self.assertEqual(measured["percent"], 80.0)

    def test_a_rate_of_nothing_is_zero_not_missing(self):
        measured = evaluation.rate(0, 0)
        self.assertEqual(measured["total"], 0)
        self.assertIsNone(measured["percent"])


class HeadlineTests(TempDirTestCase):
    """The console leads with how much moved, not with what it means."""

    def test_a_loss_category_prints_a_percentage_and_its_denominator(self):
        lines = "\n".join(
            evaluation.headline(
                {
                    "learning": {
                        "measured": 10,
                        "loss_before": 6.4,
                        "loss_after": 4.4,
                        "mean_improvement_percent": -25.0,
                        "improvement_rate_percent": 80.0,
                    }
                }
            )
        )
        self.assertIn("-25.00%", lines)
        self.assertIn("80.00% of cycles improved", lines)
        self.assertIn("6.4 -> 4.4", lines)

    def test_an_improvement_rate_is_not_printed_as_a_signed_change(self):
        # "80% of cycles improved" and "it went up by 80%" are different claims,
        # and only the first one is true.
        lines = "\n".join(
            evaluation.headline(
                {
                    "learning": {
                        "measured": 10,
                        "loss_before": 6.4,
                        "loss_after": 4.4,
                        "mean_improvement_percent": -25.0,
                        "improvement_rate_percent": 80.0,
                    }
                }
            )
        )
        self.assertNotIn("+80.00%", lines)

    def test_an_unmeasured_category_says_so_rather_than_reporting_zero(self):
        # A measured zero is a result. An absent measurement is not, and printing
        # 0.0% for it would be the report claiming coverage it does not have.
        lines = "\n".join(evaluation.headline({"learning": {"measured": 0}}))
        self.assertIn("nothing measured yet", lines)
        self.assertNotIn("0.00%", lines)

    def test_expansion_prints_the_percentage_and_the_verdict(self):
        lines = "\n".join(
            evaluation.headline(
                {
                    "expansion": {
                        "proposed": 1,
                        "helped": True,
                        "improvement_percent": -3.37,
                        "loss_at_approval": 4.0,
                        "loss_at_review": 3.87,
                        "current": {"from_layers": 4, "to_layers": 5},
                    }
                }
            )
        )
        self.assertIn("-3.37%", lines)
        self.assertIn("helped", lines)
        self.assertIn("4 -> 5", lines)

    def test_an_unjudged_growth_does_not_invent_a_percentage(self):
        lines = "\n".join(
            evaluation.headline(
                {
                    "expansion": {
                        "proposed": 1,
                        "note": "not yet judged",
                        "current": {"from_layers": 4, "to_layers": 5},
                    }
                }
            )
        )
        self.assertIn("not yet judged", lines)
        self.assertNotIn("%", lines)

    def test_every_category_appears_even_with_no_data(self):
        # A category that silently vanishes from the summary is a category the
        # reader has to assume was forgotten.
        lines = "\n".join(evaluation.headline({}))
        for label in (
            "Language",
            "Programming",
            "Reasoning",
            "Learning",
            "Knowledge",
            "Memory",
            "Stability",
            "Expansion",
        ):
            self.assertIn(label, lines)


class WidthGrowthTests(TempDirTestCase):
    """Widening the hidden dimension, which was refused for most of the project.

    n_embd sat at 256 for the whole life of the model. Architecture 22 lists
    "more neurons" and "more hidden dimensions" as expansion mechanisms, and
    neither was available: growth refused to widen because the attention
    projection fuses Q, K and V into one tensor, and copying a prefix of its
    rows would put the learned Q rows where K belongs. That reasoning was
    correct about the prefix and wrong about the conclusion, because the tensor
    is three stacked blocks rather than one, and each can be copied as a block.
    """

    def setUp(self):
        super().setUp()
        self.work = self.workdir / "out"
        self.config = make_config(
            out_dir=str(self.work),
            device="cpu",
            n_layer=1,
            n_embd=16,
            n_head=2,
            block_size=32,
        )
        self.small = Jupiter(
            ModelConfig(block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=16)
        ).to("cpu")
        self.path = train.save_checkpoint(self.small, self.config, cycle=1)

    def wide(self, n_embd=32):
        return Jupiter(
            ModelConfig(
                block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=n_embd
            )
        ).to("cpu")

    def test_widening_is_no_longer_refused(self):
        verdict, _detail = growth.compatibility(
            {
                "block_size": 32,
                "vocab_size": 64,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
            },
            {
                "block_size": 32,
                "vocab_size": 64,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 32,
            },
        )
        self.assertEqual(verdict, "compatible")

    def test_each_of_q_k_and_v_keeps_its_own_learned_weights(self):
        # The defect the refusal was protecting against. If K's learned rows
        # landed in the Q slot the model would still run and still produce
        # numbers, so nothing downstream could detect it.
        model = self.wide(32)
        growth.load_growing(model, self.path)
        learned = self.small.state_dict()["blocks.0.attn.qkv.weight"]
        widened = model.state_dict()["blocks.0.attn.qkv.weight"]
        for group in range(3):
            start = group * 32
            self.assertTrue(
                torch.equal(
                    widened[start : start + 16, 0:16],
                    learned[group * 16 : (group + 1) * 16],
                ),
                f"group {group} must keep its own learned weights",
            )
        self.assertFalse(
            torch.equal(widened[0:16, 0:16], learned[16:32]),
            "K's weights must not have been written into the Q slot",
        )

    def test_the_four_mlp_groups_each_keep_their_own_weights(self):
        # The MLP up-projection stacks four copies of the hidden size, so it has
        # the same hazard as the attention projection and the same treatment.
        model = self.wide(32)
        growth.load_growing(model, self.path)
        learned = self.small.state_dict()["blocks.0.mlp.fc.weight"]
        widened = model.state_dict()["blocks.0.mlp.fc.weight"]
        for group in range(4):
            start = group * 32
            self.assertTrue(
                torch.equal(
                    widened[start : start + 16, 0:16],
                    learned[group * 16 : (group + 1) * 16],
                ),
                f"MLP group {group} must keep its own learned weights",
            )

    def test_every_learned_tensor_survives_a_widening(self):
        model = self.wide(32)
        growth.load_growing(model, self.path)
        learned = self.small.state_dict()
        widened = model.state_dict()
        for name in (
            "token_embedding.weight",
            "lm_head.weight",
            "position_embedding.weight",
        ):
            self.assertTrue(
                torch.equal(widened[name][:, 0:16], learned[name]),
                f"{name} must keep every learned row",
            )
        for name in (
            "blocks.0.ln1.weight",
            "blocks.0.ln1.bias",
        ):
            # One value per hidden unit, and not a stacked tensor, so the learned
            # values are simply the first entries of the widened vector.
            self.assertTrue(
                torch.equal(widened[name][:16], learned[name]),
                f"{name} must keep its learned values",
            )
        # The fused projections hold several stacked groups down their first
        # dimension, so each group keeps its own values at its own offset rather
        # than all of them at the front. The group count is read from the growth
        # rules rather than written out here, so this test cannot quietly disagree
        # with the code it is checking.
        for suffix, groups in growth.STACKED_ROW_MULTIPLIERS.items():
            for prefix in ("blocks.0.attn.", "blocks.0.mlp."):
                name = prefix + suffix
                if name not in learned:
                    continue
                for group in range(groups):
                    start = group * 32
                    if widened[name].dim() == 1:
                        got = widened[name][start : start + 16]
                        want = learned[name][group * 16 : (group + 1) * 16]
                    else:
                        # A weight also has a second axis to widen, and the
                        # learned values live in its leading block.
                        got = widened[name][start : start + 16, 0:16]
                        want = learned[name][group * 16 : (group + 1) * 16]
                    self.assertTrue(
                        torch.equal(got, want),
                        f"{name} group {group} must keep its learned values",
                    )
        # The attention output projection is square, so widening it is a
        # two-dimensional block: the learned rows and columns stay where they
        # were and the rest is fresh.
        self.assertTrue(
            torch.equal(
                widened["blocks.0.attn.proj.weight"][:16, :16],
                learned["blocks.0.attn.proj.weight"],
            ),
            "the attention output projection must keep its learned block",
        )
        # The MLP down-projection is [n_embd, 4*n_embd], so widening it grows
        # both axes and the learned input columns extend past the hidden size.
        # Checking only the leading n_embd columns would pass even if the rest of
        # the learned input had been dropped, which is the part that is easy to
        # lose when an axis is a multiple of another.
        self.assertTrue(
            torch.equal(
                widened["blocks.0.mlp.proj.weight"][:16, :64],
                learned["blocks.0.mlp.proj.weight"],
            ),
            "the MLP down-projection must keep every learned input column",
        )

    def test_a_widened_model_still_produces_finite_output(self):
        # Not an equality check: a wider model has more degrees of freedom and
        # is not required to reproduce the narrow one exactly. What it must not
        # do is produce NaN or Inf, which is what a misplaced weight would look
        # like a few steps later.
        model = self.wide(32)
        growth.load_growing(model, self.path)
        logits, _loss = model(torch.randint(0, 64, (1, 8)))
        self.assertTrue(
            torch.isfinite(logits).all(), "a widened model must stay finite"
        )

    def test_the_checkpoint_is_not_archived_by_a_widening(self):
        model = self.wide(32)
        growth.load_growing(model, self.path)
        self.assertTrue(self.path.exists())

    def test_a_preview_states_the_real_cost_in_parameters(self):
        # "n_embd 256 -> 320" does not convey that the model gains a million
        # parameters, and an operator asked to authorise a change is owed the
        # number that says what it costs.
        import growth as growth_module

        after = growth_module.preview(self.config, n_embd=64)
        self.assertGreater(after["parameters_after"], after["parameters_before"])
        self.assertEqual(
            after["parameters_after"] - after["parameters_before"],
            after["parameters_added"],
        )
        self.assertIsNotNone(after["percent_larger"])
        self.assertEqual(after["to"]["n_embd"], 64)

    def test_a_preview_changes_nothing(self):
        import growth as growth_module

        before = growth_module.preview(self.config)["parameters_after"]
        growth_module.preview(self.config, n_embd=512, n_layer=8)
        after = growth_module.preview(self.config)["parameters_after"]
        self.assertEqual(before, after, "a preview must not alter the model")

    def test_grow_builds_the_larger_model_the_preview_describes(self):
        # grow() is the builder the preview and the real load both stand on, so
        # what it produces is checked against what the operator was shown.
        from model import Jupiter as _Jupiter
        from model import ModelConfig as _ModelConfig

        base = _Jupiter(
            _ModelConfig(block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=16)
        )
        grown, touched = growth.grow(base, n_embd=32)
        self.assertEqual(grown.config.n_embd, 32)
        self.assertEqual(grown.config.n_layer, 1)
        self.assertIn("n_embd", touched)
        self.assertNotIn("n_layer", touched)

    def test_width_still_cannot_shrink(self):
        # Growth is one-directional. A narrower model genuinely does not fit a
        # wider checkpoint and must still be archived rather than loaded.
        verdict, _detail = growth.compatibility(
            {
                "block_size": 32,
                "vocab_size": 64,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 32,
            },
            {
                "block_size": 32,
                "vocab_size": 64,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
            },
        )
        self.assertEqual(verdict, "incompatible")


class ExpansionCommandLineTests(TempDirTestCase):
    """A human has to be able to authorise expansion without a Python prompt.

    The worker prints "then approve it" and approve_growth had no caller outside
    the test suite, so the instruction was unactionable from the entry point: a
    control whose application is unreachable is a control that is off.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(
            expansion_state=str(self.workdir / "expansion.json"),
            out_dir=str(self.workdir / "out"),
            n_layer=2,
            n_embd=64,
        )
        self.short = [
            {
                "source": "doc-a.txt",
                "likely_cause": expansion.CAPACITY_CAUSE,
                "index": 0.9,
            }
        ]

    def run_cli(self, argv, granted=()):
        original = control.granted_domains
        control.granted_domains = lambda environ=None: set(granted)
        self.addCleanup(setattr, control, "granted_domains", original)
        return expansion.main(self.config, argv)

    def test_status_reports_a_proposal(self):
        expansion.propose_growth(self.config, self.short, cycle=1)
        self.assertEqual(self.run_cli(["status"]), 0)

    def test_approving_without_the_grant_is_refused_and_exits_nonzero(self):
        expansion.propose_growth(self.config, self.short, cycle=1)
        self.assertEqual(self.run_cli(["approve-growth", "--loss", "4.0"]), 1)
        self.assertIsNone(
            expansion.pending_shape(self.config),
            "a refused approval must not become a shape change",
        )

    def test_approving_with_the_grant_applies_the_growth(self):
        expansion.propose_growth(self.config, self.short, cycle=1)
        self.assertEqual(
            self.run_cli(["approve-growth", "--loss", "4.0"], ["expansion"]), 0
        )
        pending = expansion.pending_shape(self.config)
        self.assertEqual(pending["n_layer"], 3)
        self.assertGreater(pending["n_embd"], self.config.n_embd)

    def test_the_whole_human_loop_runs_from_the_command_line(self):
        # propose -> approve -> judge, with no Python required, which is what
        # the worker's printed instruction tells the operator to do.
        expansion.propose_growth(self.config, self.short, cycle=1)
        self.assertEqual(
            self.run_cli(["approve-growth", "--loss", "4.0"], ["expansion"]), 0
        )
        self.assertEqual(
            self.run_cli(["review-growth", "--loss", "3.0"], ["expansion"]), 0
        )
        verdict = expansion.pending(self.config)["capacity_growth"]
        self.assertTrue(verdict["kept"], "a lower loss means the growth helped")
        self.assertEqual(verdict["change"], -1.0)

    def test_a_growth_can_be_judged_as_not_helping(self):
        expansion.propose_growth(self.config, self.short, cycle=1)
        self.run_cli(["approve-growth", "--loss", "4.0"], ["expansion"])
        self.assertEqual(
            self.run_cli(["review-growth", "--loss", "5.0"], ["expansion"]), 0
        )
        self.assertFalse(expansion.pending(self.config)["capacity_growth"]["kept"])

    def test_approving_nothing_is_reported_rather_than_crashing(self):
        self.assertEqual(self.run_cli(["approve-growth"], ["expansion"]), 1)

    def test_a_growth_with_no_baseline_is_reported_as_unjudgeable(self):
        # Approving without a loss is allowed, because refusing it would stop a
        # human making a decision they are entitled to make. It is reported as
        # unjudged rather than scored, which is the honest outcome.
        expansion.propose_growth(self.config, self.short, cycle=1)
        self.assertEqual(self.run_cli(["approve-growth"], ["expansion"]), 0)
        self.assertEqual(
            self.run_cli(["review-growth", "--loss", "3.0"], ["expansion"]), 1
        )


class ProgrammingAndReasoningTests(TempDirTestCase):
    """Architecture 34: two categories that had no measurement behind them.

    Both were reported as NOT MEASURED with the reasoning that "nothing exists
    to evaluate", which was not quite true: the corpus contains programming
    documents, and the learning loop already records a before and after holdout
    loss for every cycle. What was missing was the question, not the data.

    What these measure is whether bounded updates reduce loss on that class of
    material, which is a fact about the learning loop. It is deliberately not a
    capability claim: a low loss on a code document means its bytes are
    predicted, not that the program is correct.
    """

    CODE = (
        "def parse(text):\n"
        "    if not text:\n"
        "        return None\n"
        "    return text.strip()\n"
        "class Reader:\n"
        "    def read(self, path):\n"
        "        with open(path) as handle:\n"
        "            return handle.read()\n"
    )
    PROSE = (
        "The observatory stood on the granite above the valley. "
        "Instruments recorded the river through the long winter. "
    )

    ARGUMENT = (
        "The valve must be closed first because pressure builds while the "
        "pump runs. Therefore the operator checks the gauge, and because the "
        "gauge reads high the pump is stopped. Hence the line is isolated. "
    )

    def test_code_is_told_apart_from_prose(self):
        self.assertGreater(evaluation.code_share(self.CODE), 0.5)
        self.assertLess(evaluation.code_share(self.PROSE), 0.05)

    def test_prose_dividing_a_line_can_look_like_code(self):
        # "class A: pass" is a code line, but a sentence beginning with
        # "class:" is not code either, and a heuristic that claims it is would
        # quietly mislabel ordinary writing.
        self.assertEqual(evaluation.code_share(""), 0.0)
        self.assertEqual(evaluation.code_share("\n\n\n"), 0.0)

    def test_reasoning_connectives_are_counted_per_hundred_words(self):
        self.assertGreater(evaluation.reasoning_share(self.ARGUMENT), 1.0)
        self.assertLess(evaluation.reasoning_share(self.PROSE), 0.2)

    def test_the_measurements_exist_for_both_categories(self):
        c = make_config(experiment_log=str(self.workdir / "experiments.jsonl"))
        for result in (
            evaluation.evaluate_programming(c),
            evaluation.evaluate_reasoning(c),
        ):
            self.assertIn("measured", result)
            self.assertIn("note", result)

    def test_an_empty_experiment_log_says_so_rather_than_measuring_nothing(self):
        c = make_config(experiment_log=str(self.workdir / "experiments.jsonl"))
        result = evaluation.evaluate_programming(c)
        self.assertEqual(result["measured"], 0)
        self.assertIn("no comparable", result["note"])

    def test_a_code_source_contributes_its_cycle(self):
        work = self.workdir
        source = work / "code.txt"
        source.write_text(self.CODE * 8, encoding="utf-8")
        log = work / "experiments.jsonl"
        log.write_text(
            json.dumps(
                {
                    "input": str(source),
                    "before_metrics": {"holdout_loss": 4.0},
                    "after_metrics": {"holdout_loss": 3.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        c = make_config(experiment_log=str(log))
        result = evaluation.evaluate_programming(c)
        self.assertEqual(result["measured"], 1)
        self.assertEqual(result["cycles_that_improved"], 1)
        self.assertAlmostEqual(result["mean_holdout_change"], -1.0, places=5)

    def test_prose_only_material_leaves_programming_unmeasured(self):
        # Honesty matters more than coverage: a category with nothing behind it
        # should say so rather than borrow someone else's numbers.
        work = self.workdir
        source = work / "prose.txt"
        source.write_text(self.PROSE * 40, encoding="utf-8")
        log = work / "experiments.jsonl"
        log.write_text(
            json.dumps(
                {
                    "input": str(source),
                    "before_metrics": {"holdout_loss": 4.0},
                    "after_metrics": {"holdout_loss": 3.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        c = make_config(experiment_log=str(log))
        self.assertEqual(evaluation.evaluate_programming(c)["measured"], 0)

    def test_the_report_includes_both(self):
        c = make_config(
            knowledge_store=str(self.workdir / "knowledge.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
        )
        result = evaluation.report(c)
        self.assertIn("programming", result)
        self.assertIn("reasoning", result)
        self.assertNotIn("programming", result["coverage"]["Programming"].lower()[:4])


class MetricTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(
            knowledge_store=str(self.workdir / "knowledge.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
        )
        self.knowledge = store_module.Store(self.config.knowledge_store)

    def test_language_reports_when_there_is_nothing_to_measure(self):
        # The config is passed by keyword because the first positional argument
        # is the config, not the samples.
        result = evaluation.evaluate_language(self.config, samples=[])
        self.assertEqual(result["samples"], 0)
        self.assertIn("note", result)

    def test_language_summarises_samples(self):
        result = evaluation.evaluate_language(self.config, samples=[DEGENERATE, VARIED])
        self.assertEqual(result["samples"], 2)
        # 1 - unique/total approaches 1.0 as a loop gets longer; it is never
        # exactly 1.0 for finite text, so the bound is asserted, not the value.
        self.assertGreater(result["worst_repetition_rate"], 0.9)
        self.assertEqual(result["best_repetition_rate"], 0.0)

    def test_knowledge_reports_when_the_store_is_missing(self):
        result = evaluation.evaluate_knowledge(None)
        self.assertEqual(result["probes"], 0)

    def test_knowledge_finds_records_it_queries_with(self):
        memory.remember_external(
            self.knowledge,
            "neutrino",
            "the neutrino observatory records neutrino events every night",
            origin="doc.txt",
        )
        result = evaluation.evaluate_knowledge(self.knowledge)
        self.assertGreater(result["probes"], 0)
        self.assertEqual(result["hit_rate"], 1.0)

    def test_memory_counts_what_a_later_session_would_see(self):
        memory.remember_long_term(self.knowledge, "fact", "a stable fact")
        reopened = store_module.Store(self.config.knowledge_store)
        result = evaluation.evaluate_memory(self.config, reopened)
        self.assertEqual(result["collections"][memory.LONG_TERM], 1)
        self.assertEqual(result["retrievable"], 1)

    def test_learning_reports_an_improvement_as_negative_change(self):
        experiment.write_experiment(
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
            verdict="ACCEPT",
            reason="measured",
            regression_before=3.0,
            regression_after=2.9,
            confidence=0.5,
            model_cycle=1,
            resources={},
        )
        result = evaluation.evaluate_learning(self.config)
        self.assertEqual(result["cycles"], 1)
        self.assertEqual(result["cycles_that_improved"], 1)
        self.assertLess(result["mean_holdout_change"], 0.0)

    def test_stability_reports_when_there_is_no_yardstick_measurement(self):
        result = evaluation.evaluate_stability(self.config)
        self.assertEqual(result["cycles"], 0)
        self.assertIn("Architecture 33", result["note"])

    def test_stability_counts_a_regression(self):
        for before, after in ((3.0, 2.9), (2.9, 3.4)):
            experiment.write_experiment(
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
                verdict="ACCEPT",
                reason="measured",
                regression_before=before,
                regression_after=after,
                confidence=0.5,
                model_cycle=1,
                resources={},
            )
        result = evaluation.evaluate_stability(self.config)
        self.assertEqual(result["cycles"], 2)
        self.assertEqual(result["cycles_that_regressed"], 1)

    def test_the_full_report_is_serializable(self):
        memory.remember_long_term(self.knowledge, "fact", "a stable fact")
        result = evaluation.report(self.config, knowledge=self.knowledge)
        self.assertIn("coverage", result)
        self.assertIn("measured_categories", result)
        json.dumps(result)
