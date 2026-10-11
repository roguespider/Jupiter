"""Tests for dialogue as training material and the conversation yardstick.

Two things are checked here, and they are the same request seen from two ends.

Dialogue used to be something curation could report and not learn from. A
document was measured as a conversation, counted in the report, and then written
out exactly as it arrived, with the speaker labels wherever they happened to
fall. A model trained on that learns the characters of a transcript rather than
the shape of taking a turn in one.

The conversation set exists because loss on text cannot see any of that. A model
can drive every loss number down and still answer each question with another
question, and the acceptance gate would have called that an improvement.
"""

import unittest

import acceptance
import conversation_eval as ce
import dialogue

from tests.support import make_config

CONVERSATION = (
    "Alice: what does a database index do\n"
    "Bob: it makes lookups faster by storing an ordering\n"
    "Alice: does it change the data\n"
    "Bob: no it only changes how fast you find a row\n"
    "Alice: so it is about speed\n"
    "Bob: exactly that\n"
)

PROSE = (
    "The river ran cold that winter and she walked beside it without speaking.\n"
    "The afternoon light went flat and the trees made no shadow at all.\n"
)


class TrainingMaterialTests(unittest.TestCase):
    """A conversation is rendered into the layout the model is spoken in."""

    def test_a_conversation_is_rendered_as_training_material(self):
        rendered = dialogue.render_for_training(CONVERSATION)
        self.assertIn(dialogue.HUMAN, rendered)
        self.assertIn(dialogue.JUPITER, rendered)
        self.assertNotIn("Alice", rendered)
        self.assertNotIn("Bob", rendered)

    def test_the_rendering_is_the_layout_the_model_is_spoken_in(self):
        # The whole point of rendering: the format trained on and the format
        # asked at have to be the same one, or the model is being set up to
        # fail at the moment it is used.
        rendered = dialogue.render_for_training(CONVERSATION)
        for line in rendered.splitlines():
            self.assertRegex(line, r"^(You|Jupiter): .+")

    def test_both_positions_appear(self):
        # Roles alternate by position. Assigning them by name produced a
        # transcript in which every turn was addressed to the same speaker,
        # because no name in a general corpus is on any list of assistant
        # names, and that teaches a monologue.
        rendered = dialogue.render_for_training(CONVERSATION)
        self.assertIn(f"{dialogue.HUMAN}:", rendered)
        self.assertIn(f"{dialogue.JUPITER}:", rendered)
        self.assertGreaterEqual(rendered.count(dialogue.HUMAN), 2)
        self.assertGreaterEqual(rendered.count(dialogue.JUPITER), 2)

    def test_prose_is_returned_untouched(self):
        # Guessing where a turn begins in a novel is the semantic decision
        # curation says it has no business making.
        self.assertEqual(dialogue.render_for_training(PROSE), PROSE)

    def test_every_turn_survives_the_rendering(self):
        rendered = dialogue.render_for_training(CONVERSATION)
        self.assertEqual(
            dialogue.training_turns(CONVERSATION),
            len([line for line in rendered.splitlines() if line.strip()]),
        )

    def test_prose_contributes_no_training_turns(self):
        self.assertEqual(dialogue.training_turns(PROSE), 0)

    def test_a_document_that_is_not_dialogue_reports_no_turns(self):
        self.assertEqual(dialogue.training_turns("just a sentence"), 0)


class ScoringTests(unittest.TestCase):
    """A reply passes when it says the thing, not when it sounds right."""

    def test_a_correct_answer_passes(self):
        self.assertTrue(
            ce.answer_contains(
                "It speeds up database lookup by letting you find rows faster.",
                ["speed", "lookup", "find"],
            )
        )

    def test_a_wrong_answer_fails(self):
        self.assertFalse(
            ce.answer_contains(
                "Databases are files that store things.",
                ["speed", "lookup", "find"],
            )
        )

    def test_a_partial_answer_does_not_pass(self):
        # Every expected term is required. An "any of" rule lets a reply that
        # says one right thing among nine wrong ones count as correct, which is
        # a check that flatters a model rather than measuring it.
        self.assertFalse(
            ce.answer_contains("It speeds up lookups.", ["speed", "lookup", "find"])
        )
        self.assertAlmostEqual(
            ce.score_reply("It speeds up lookups.", ["speed", "lookup", "find"]),
            0.6667,
            places=4,
        )

    def test_the_question_being_asked_does_not_count_as_an_answer(self):
        # A model that repeats the question has not answered it, and the
        # question's own words are in the prompt.
        self.assertFalse(
            ce.answer_contains(
                "What does a database index do? It is a thing.",
                ["speed", "lookup", "find"],
            )
        )

    def test_plural_and_singular_are_the_same_term(self):
        self.assertTrue(ce.answer_contains("It stores an index.", ["index"]))
        self.assertTrue(ce.answer_contains("It uses indexes.", ["index"]))

    def test_an_empty_expected_list_scores_zero_rather_than_passing(self):
        self.assertEqual(ce.score_reply("anything", []), 0.0)

    def test_the_set_is_not_empty(self):
        # A yardstick with no questions passes everything and measures nothing.
        self.assertTrue(ce.CONVERSATION_SET)

    def test_every_question_states_what_a_correct_answer_contains(self):
        for question, expected in ce.CONVERSATION_SET:
            self.assertTrue(question.strip(), question)
            self.assertTrue(expected, f"{question} has no expected answer")


class AcceptanceTests(unittest.TestCase):
    """The gate judges conversation, separately from loss."""

    def classify(self, improvement, conversation_change):
        c = make_config()
        return acceptance.classify(
            c,
            improvement=improvement,
            regression_delta=0.5,
            regression_available=True,
            confidence=0.9,
            observed=5,
            conversation_change=conversation_change,
        )

    def test_a_conversation_regression_is_rejected(self):
        # The case the measurement exists for: every loss number improved and
        # the model still got worse at the thing a conversation is for.
        verdict, reason = self.classify(improvement=0.9, conversation_change=-0.25)
        self.assertEqual(verdict, acceptance.REJECT)
        self.assertIn("conversation", reason)

    def test_unchanged_conversation_does_not_block_an_improvement(self):
        verdict, _reason = self.classify(improvement=0.9, conversation_change=0.0)
        self.assertEqual(verdict, acceptance.ACCEPT)

    def test_an_improvement_in_conversation_does_not_rescue_a_worse_one(self):
        verdict, _reason = self.classify(improvement=-0.9, conversation_change=0.5)
        self.assertEqual(verdict, acceptance.REJECT)

    def test_an_unmeasured_set_does_not_penalise_anything(self):
        # Not measured and not worse are different facts. Only one of them is a
        # pass, and the gate must not invent the other.
        verdict, _reason = self.classify(improvement=0.9, conversation_change=None)
        self.assertEqual(verdict, acceptance.ACCEPT)

    def test_the_tolerance_is_configurable(self):
        c = make_config(conversation_tolerance=0.5)
        verdict, _reason = acceptance.classify(
            c,
            improvement=0.9,
            regression_delta=0.5,
            regression_available=True,
            confidence=0.9,
            observed=5,
            conversation_change=-0.25,
        )
        self.assertEqual(
            verdict,
            acceptance.ACCEPT,
            "a fall inside the configured tolerance is not a regression",
        )


class ComparisonTests(unittest.TestCase):
    """Two runs of the same set, and what changed between them."""

    def before(self, rate):
        return {"pass_rate": rate, "questions": 8, "passed": int(8 * rate)}

    def test_a_fall_is_reported_as_a_regression(self):
        change = ce.compare(self.before(0.5), self.before(0.25))
        self.assertTrue(change["regressed"])
        self.assertEqual(change["change"], -0.25)

    def test_no_change_is_not_a_regression(self):
        change = ce.compare(self.before(0.5), self.before(0.5))
        self.assertFalse(change["regressed"])

    def test_a_rise_is_not_a_regression(self):
        change = ce.compare(self.before(0.25), self.before(0.75))
        self.assertFalse(change["regressed"])

    def test_comparing_with_nothing_returns_nothing(self):
        self.assertIsNone(ce.compare(None, self.before(0.5)))
        self.assertIsNone(ce.compare(self.before(0.5), None))
