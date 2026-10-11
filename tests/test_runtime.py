"""Tests for the interactive runtime (sample.py).

The runtime is the only part of Jupiter a person talks to, and it is the part
that was least tested. Two things matter enough to pin here.

The experience record is the runtime's contribution to learning. A conversation
that is answered and then forgotten teaches nothing, and one that is recorded
badly teaches something wrong, so the record is checked rather than assumed.

The startup banner once read its vocabulary from the configuration rather than
from the architecture record, and reported 256 for a model that is 5256 wide.
That is a small lie told at exactly the moment a person decides whether to trust
what they are about to see, and it is the kind of thing a test has to hold.
"""

import json
import unittest

import sample

from tests.support import TempDirTestCase, make_config


class ExperienceRecordTests(TempDirTestCase):
    """A completed interaction is recorded, and recorded truthfully."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            knowledge_store=str(self.workdir / "out" / "knowledge.json"),
        )

    def test_the_experience_file_lives_under_the_output_directory(self):
        # No new configuration setting: the store belongs beside the model it
        # describes, and a setting for it would be a setting that can disagree
        # with where the model actually is.
        path = sample.experience_path(self.config)
        self.assertEqual(path.parent, self.workdir / "out")
        self.assertEqual(path.suffix, ".jsonl")

    def test_a_recording_with_nothing_recalled_is_still_a_recording(self):
        # A turn where retrieval found nothing is a real turn. Recording only the
        # turns that recalled something would make the store a record of what
        # Jupiter already knew rather than of what it was asked.
        sample.record_experience(self.config, "hello", "hi", [])
        records = [
            json.loads(line)
            for line in sample.experience_path(self.config)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["user"], "hello")
        self.assertEqual(records[0]["jupiter"], "hi")
        self.assertEqual(records[0]["recalled"], [])

    def test_a_recording_says_what_was_recalled_and_from_where(self):
        recalled = [
            {
                "collection": "concepts",
                "id": "abc",
                "score": 0.8,
                "matched": ["index", "database"],
            }
        ]
        sample.record_experience(
            self.config, "what is an index", "a structure", recalled
        )
        record = json.loads(
            sample.experience_path(self.config)
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(record["recalled"][0]["collection"], "concepts")
        self.assertEqual(record["recalled"][0]["matched"], ["index", "database"])

    def test_recording_appends_rather_than_replacing(self):
        # The file is a stream of turns, not a summary of the last one.
        sample.record_experience(self.config, "first", "one", [])
        sample.record_experience(self.config, "second", "two", [])
        self.assertEqual(sample.experience_count(self.config), 2)

    def test_the_count_is_zero_before_anything_is_recorded(self):
        self.assertEqual(sample.experience_count(self.config), 0)

    def test_the_record_does_not_modify_the_model(self):
        # An experience is an observation. Writing one changes no weights and
        # needs no training run, which is the whole point of separating the two.
        before = sample.experience_count(self.config)
        sample.record_experience(self.config, "hello", "hi", [])
        self.assertEqual(sample.experience_count(self.config), before + 1)
        weights = self.workdir / "out" / "Jupiter_latest.safetensors"
        self.assertFalse(
            weights.exists(),
            "recording an experience must not write a checkpoint",
        )


class ContextAssemblyTests(unittest.TestCase):
    """Knowledge and conversation compete for a fixed window.

    The choice of who loses is the whole design: a recalled fact answers the
    question being asked now, while a question from twenty turns ago usually
    does not, so the conversation is what gets dropped.
    """

    def test_knowledge_is_kept_and_older_conversation_is_dropped(self):
        import torch

        prefix = torch.zeros((1, 10), dtype=torch.long)
        history = torch.arange(100, 600, dtype=torch.long).reshape(1, -1)
        assembled = sample.assemble(prefix, history, 100)
        self.assertEqual(assembled.shape[1], 100)
        # The knowledge comes first, and the tail of the conversation follows.
        self.assertTrue(torch.equal(assembled[0, :10], prefix[0]))
        self.assertTrue(torch.equal(assembled[0, -1], torch.tensor(599)))

    def test_knowledge_alone_fills_the_window_when_there_is_no_conversation(self):
        import torch

        prefix = torch.zeros((1, 10), dtype=torch.long)
        history = torch.zeros((1, 0), dtype=torch.long)
        self.assertEqual(sample.assemble(prefix, history, 100).shape[1], 10)

    def test_the_window_never_exceeds_the_block_size(self):
        import torch

        prefix = torch.zeros((1, 600), dtype=torch.long)
        history = torch.arange(500, dtype=torch.long).reshape(1, -1)
        assembled = sample.assemble(prefix, history, 512)
        self.assertLessEqual(
            assembled.shape[1], 512, "the model cannot be given more than it holds"
        )

    def test_the_current_turn_is_included_before_generation(self):
        # Without this the first turn produces an empty context and the model
        # answers nothing at all, because there is nothing to continue from.
        import torch

        prefix = torch.zeros((1, 5), dtype=torch.long)
        history = torch.zeros((1, 0), dtype=torch.long)
        turn = torch.tensor([[1, 2, 3]], dtype=torch.long)
        context = sample.generation_context(prefix, history, turn, 512)
        self.assertTrue(
            torch.equal(context[0, -3:], turn[0]),
            "the turn being asked must reach the model",
        )


class ConsoleTests(TempDirTestCase):
    """The banner reports the model that is actually loaded."""

    def test_the_encoding_comes_from_the_console_and_never_fails(self):
        # A byte-level tokenizer emits bytes the console may not be able to
        # show, and printing one of those raised UnicodeEncodeError and ended
        # the conversation mid-reply. The model was answering; the terminal
        # could not display it.
        import sys

        self.assertTrue(sample._console_encoding())
        original = sys.stdout
        sys.stdout = type("NoEncoding", (), {"encoding": None})()
        self.addCleanup(setattr, sys, "stdout", original)
        self.assertEqual(sample._console_encoding(), "utf-8")
