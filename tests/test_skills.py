"""Tests for skill-aware document selection.

The loop this closes is the one the corpus was sorted for: the evaluation report
measures named categories, and something has to act on which of them is weak.
These tests pin down that it acts in the right direction and, just as
importantly, that it stays a preference rather than becoming a policy.
"""

import unittest
from pathlib import Path

import selection

import skills


def make_report(**scores):
    """An evaluation report shaped like the one evaluation.py writes."""
    return {name: {"score": value} for name, value in scores.items()}


class CategoryScoreTests(unittest.TestCase):
    def test_a_score_is_inverted_into_a_weakness(self):
        # The report stores how well a category did. Weakness is the other
        # direction, and doing the inversion once is what stops every consumer
        # having to remember which way a number points.
        scores = skills.category_scores(make_report(conversation=0.25))
        self.assertEqual(scores["conversation"], 0.75)

    def test_a_perfect_score_is_no_weakness(self):
        self.assertEqual(
            skills.category_scores(make_report(conversation=1.0))["conversation"],
            0.0,
        )

    def test_a_category_with_nothing_measurable_contributes_nothing(self):
        # Inventing a score for a category nobody measured is the dishonesty
        # this project exists to avoid.
        report = {"conversation": {"note": "not measured yet"}}
        self.assertEqual(skills.category_scores(report), {})

    def test_a_score_outside_the_range_is_refused_not_clamped(self):
        # Clamping a number this function does not understand would turn a
        # mistake into a confident answer.
        for bad in (-0.5, 1.5, 42):
            self.assertNotIn(
                "conversation",
                skills.category_scores(make_report(conversation=bad)),
            )

    def test_a_missing_report_is_no_weakness_rather_than_an_error(self):
        self.assertEqual(skills.category_scores({}), {})
        self.assertEqual(skills.category_scores(None), {})


class FolderWeaknessTests(unittest.TestCase):
    def test_a_weak_conversation_weakens_the_conversation_folder(self):
        weakness = skills.folder_weakness({"conversation": 0.9})
        self.assertEqual(weakness["conversation"], 0.9)

    def test_a_folder_takes_the_worst_of_the_categories_that_speak_to_it(self):
        # A folder may answer to several categories, and the strongest of them
        # is the one that counts: ranking by the better of the two would
        # understate how urgently the material is needed.
        weakness = skills.folder_weakness({"language": 0.2, "conversation": 0.8})
        self.assertEqual(weakness["conversation"], 0.8)

    def test_the_wanted_cap_is_respected(self):
        # A category that scored badly must not be able to outrank a document
        # that is actively failing to learn.
        self.assertEqual(skills.bonus_for(1.0), skills.MAX_BONUS)
        self.assertEqual(skills.bonus_for(0.0), 0.0)
        self.assertLess(skills.MAX_BONUS, selection.need_for("HIGH", 0.0))


class FolderOfTests(unittest.TestCase):
    def test_a_filed_document_reports_its_folder(self):
        self.assertEqual(
            skills.folder_of(Path("data/raw/conversation/a.txt")), "conversation"
        )

    def test_an_unfiled_document_reports_nothing(self):
        # The folder is a statement the owner made by where they put the file.
        # Guessing from the contents would be a semantic decision curation says
        # it has no business making.
        self.assertIsNone(skills.folder_of(Path("data/raw/mystery.txt")))


class SelectionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.pieces = [
            (Path("data/raw/conversation/a.txt"), "chat"),
            (Path("data/raw/code/b.txt"), "code"),
        ]

    def test_a_weak_skill_moves_its_material_up_the_queue(self):
        weak = skills.folder_weakness({"conversation": 1.0})
        ordered = selection.rank(self.pieces, {}, {}, 1, weak)
        self.assertEqual(ordered[0][0].parent.name, "conversation")

    def test_a_strong_skill_does_not(self):
        strong = skills.folder_weakness({"conversation": 0.0})
        ordered = selection.rank(self.pieces, {}, {}, 1, strong)
        self.assertEqual(ordered[0][0].parent.name, "code")

    def test_no_evaluation_means_the_previous_behaviour(self):
        # Before there was a report there was no preference, and the age term
        # still decides. A first run must not behave differently because the
        # thing that feeds the preference has not been written yet.
        ordered = selection.rank(self.pieces, {}, {}, 1, {})
        self.assertEqual(len(ordered), 2)

    def test_the_recorded_reason_says_why_the_document_won(self):
        # A curriculum that changed because a capability was measured weak has
        # to be legible from the record, or it looks like a random walk.
        _path, _text, reason = selection.choose(
            self.pieces, {}, {}, 1, skills.folder_weakness({"conversation": 1.0})
        )
        self.assertIn("skill conversation is weak", reason)

    def test_a_raw_weakness_cannot_exceed_the_cap(self):
        # rank() is public and a caller may hand it a weakness rather than the
        # capped bonus, so the cap is enforced where the number is used.
        _path, _text, reason = selection.choose(
            self.pieces, {}, {}, 1, {"conversation": 1.0}
        )
        self.assertIn(f"+{skills.MAX_BONUS:.2f}", reason)
