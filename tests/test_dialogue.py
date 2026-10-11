"""Tests for dialogue structure (Architecture 25, 27).

The classifier is a counting heuristic and these tests pin down exactly what it
counts, including the case that matters most: a leaked chat template is
contamination, not dialogue, and must never be learned from as conversation.
"""

import unittest

import dialogue

CHAT_LOG = (
    "Alice: did the detector record anything tonight\n"
    "Bob: it recorded three events above threshold\n"
    "Alice: are you sure that is not noise\n"
    "Bob: the second trace was clean\n"
    "Alice: then we have a result\n"
    "Bob: we have a result worth writing down\n"
)

PROSE = (
    "The observatory stood on the granite above the valley, and the wind off "
    "the ice field made the door hard to close. Inside, the detector counted "
    "events in the dark. Nobody spoke for a long while.\n"
)

LEAKED_TEMPLATE = (
    "==================================================\n"
    "\n"
    "[SYSTEM]: reasoning language: Spanish\n"
    "\n"
    "System: Your response should be provided as a brief statement.\n"
    "System: Follow these instructions exactly.\n"
    "[USER]: what is the capital of France\n"
    "[ASSISTANT]: The capital of France is Paris.\n"
)


class TurnDetectionTests(unittest.TestCase):
    def test_speaker_labels_are_found(self):
        self.assertEqual(dialogue.turns("Alice: hello\nBob: hi\n"), ["Alice", "Bob"])

    def test_a_label_must_start_the_line(self):
        self.assertEqual(dialogue.turns("he said Alice: hello\n"), [])

    def test_ordinary_prose_has_no_turns(self):
        self.assertEqual(dialogue.turns(PROSE), [])

    def test_speakers_are_ordered_by_frequency(self):
        text = "Alice: a\nBob: b\nAlice: c\nAlice: d\n"
        self.assertEqual(dialogue.speakers(text), ["Alice", "Bob"])


class ConversationTests(unittest.TestCase):
    def test_a_chat_log_is_conversational(self):
        self.assertTrue(dialogue.is_conversational(CHAT_LOG))

    def test_prose_is_not_conversational(self):
        self.assertFalse(dialogue.is_conversational(PROSE))

    def test_a_few_stray_labels_in_a_long_document_are_not_dialogue(self):
        # The failure this guards against: a novel containing a handful of
        # "Name:" lines must not be classified as a conversation.
        text = PROSE * 200 + "\nAlice: hello\nBob: hi\n"
        self.assertGreater(dialogue.turn_count(text), 0)
        self.assertLess(dialogue.turn_share(text), dialogue.MIN_TURN_SHARE)
        self.assertFalse(dialogue.is_conversational(text))

    def test_one_speaker_alone_is_not_a_conversation(self):
        monologue = "".join(f"System: line {n}\n" for n in range(20))
        self.assertFalse(dialogue.is_conversational(monologue))

    def test_a_leaked_chat_template_is_never_conversational(self):
        # It is mostly speaker lines, so counting alone would pass it. It is
        # contamination and has to be rejected.
        self.assertGreater(dialogue.turn_count(LEAKED_TEMPLATE), 0)
        self.assertFalse(dialogue.is_conversational(LEAKED_TEMPLATE))

    def test_template_artifacts_are_counted(self):
        self.assertGreater(len(dialogue.template_artifacts(LEAKED_TEMPLATE)), 0)
        self.assertEqual(dialogue.template_artifacts(PROSE), [])

    def test_turn_share_is_reported(self):
        self.assertGreater(dialogue.turn_share(CHAT_LOG), 0.5)
        self.assertEqual(dialogue.turn_share(""), 0.0)
        self.assertEqual(dialogue.turn_share("   \n\n  \n"), 0.0)


class ReportTests(unittest.TestCase):
    def test_the_report_states_measurements(self):
        result = dialogue.report(CHAT_LOG)
        self.assertEqual(result["turns"], 6)
        self.assertEqual(result["speakers"], ["Alice", "Bob"])
        self.assertTrue(result["conversational"])
        self.assertEqual(result["template_artifacts"], 0)

    def test_the_report_says_which_layout_it_read(self):
        # A corpus can hold both layouts and the two are laid out differently,
        # so a reader comparing two documents needs to know where the number
        # came from.
        self.assertEqual(dialogue.report(CHAT_LOG)["layout"], "labelled")
        self.assertEqual(dialogue.report(STRUCTURED)["layout"], "structured")
        self.assertEqual(dialogue.report(PROSE)["layout"], "prose")

    def test_a_prose_report_is_honest_about_being_prose(self):
        result = dialogue.report(PROSE)
        self.assertFalse(result["conversational"])
        self.assertEqual(result["turns"], 0)


class TurnFormatTests(unittest.TestCase):
    def test_a_turn_is_a_label_and_the_text(self):
        self.assertEqual(dialogue.format_turn("Alice", "hello"), "Alice: hello\n")

    def test_the_conversation_uses_a_label_for_each_side(self):
        self.assertEqual(dialogue.HUMAN, "You")
        self.assertEqual(dialogue.JUPITER, "Jupiter")


# The corpus arrived holding instruction data as JSON records rather than as
# labelled transcripts. The line reader sees none of it, so 259 MB of the best
# conversation material in the corpus was reported as prose and rendered as raw
# JSON, which would have taught the model to emit braces instead of answers.
STRUCTURED = (
    '{"input": [{"role": "user", "content": "what is a variable"}], '
    '"output": "a name bound to a value", "category": "code"}\n'
    '{"input": [{"role": "user", "content": "what is a function"}], '
    '"output": "a callable that takes arguments", "category": "code"}\n'
)


class StructuredRecordTests(unittest.TestCase):
    def test_structured_records_are_conversation(self):
        # The regression that mattered: this returned False, so the corpus was
        # counted as prose and nothing about it was rendered as turns.
        self.assertTrue(dialogue.is_conversational(STRUCTURED))

    def test_both_sides_of_a_record_are_counted_as_turns(self):
        # A record always has one speaker in its input and its reply under a
        # different field, so counting only the role keys found one speaker and
        # every structured document looked like a monologue.
        result = dialogue.report(STRUCTURED)
        self.assertEqual(result["turns"], 4)
        self.assertEqual(sorted(result["speakers"]), ["assistant", "user"])
        self.assertGreater(result["structured_turn_share"], 0.0)

    def test_the_report_separates_who_is_addressed_from_who_speaks(self):
        # A labelled transcript names its speakers; a record only names the one
        # it addresses. Reporting just the addressed role would describe every
        # instruction document as a monologue.
        self.assertEqual(dialogue.report(STRUCTURED)["addressed_roles"], ["user"])
        self.assertEqual(dialogue.report(CHAT_LOG)["addressed_roles"], [])

    def test_a_record_renders_as_the_two_turns_the_model_is_spoken_in(self):
        rendered = dialogue.render_for_training(STRUCTURED)
        self.assertIn("You: what is a variable", rendered)
        self.assertIn("Jupiter: a name bound to a value", rendered)
        self.assertNotIn('"role"', rendered)
        self.assertNotIn('"output"', rendered)

    def test_the_share_is_measured_per_record_not_per_line(self):
        # A JSON object is often one enormous line, so a per-line share called a
        # document made entirely of conversation worth nothing. The line share is
        # still reported, clamped to a fraction because a share above one reads as
        # nonsense, and it is the record share the classifier actually consults.
        self.assertEqual(dialogue.turn_share(STRUCTURED), 1.0)
        self.assertEqual(dialogue.structured_turn_share(STRUCTURED), 1.0)

    def test_training_turns_counts_two_per_record(self):
        self.assertEqual(dialogue.training_turns(STRUCTURED), 4)

    def test_json_that_merely_mentions_a_role_is_not_a_conversation(self):
        # A permissive search would match any document discussing agents.
        mention = '{"note": "the role field is optional", "count": 1}\n' * 4
        self.assertFalse(dialogue.is_conversational(mention))

    def test_a_leaked_template_is_still_refused(self):
        # Structured reading must not become a way around contamination.
        self.assertFalse(dialogue.is_conversational(LEAKED_TEMPLATE))

    def test_a_labelled_transcript_is_unaffected(self):
        # The line layout has to behave exactly as it did.
        result = dialogue.report(CHAT_LOG)
        self.assertEqual(result["turns"], 6)
        self.assertEqual(result["speakers"], ["Alice", "Bob"])
        self.assertEqual(result["structured_turn_share"], 0.0)
        self.assertTrue(result["conversational"])

    def test_prose_is_still_prose(self):
        self.assertFalse(dialogue.is_conversational(PROSE))
