"""Tests for retrieval, and for what Architecture 34 calls Memory evaluation.

The retention test is the one that matters: it writes knowledge through the
learning cycle, throws the store away, opens it again from scratch, and checks
that a question can still find what was recorded. That is the whole point of
Architecture 16, which says information should be able to live outside the
weights and still be recallable.
"""

import unittest

import concepts
import learning_worker
import sample
import store as store_module
import torch

import memory
import retrieval
from tests.support import TempDirTestCase, make_config

LEARNABLE_TEXT = (
    "the neutrino observatory records neutrino events. "
    "a neutrino observatory needs a detector. "
) * 8


class RetrievalTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.knowledge = store_module.Store(self.workdir / "knowledge.json")

    def test_query_terms_drop_stopwords(self):
        self.assertEqual(
            retrieval.query_terms("what is a neutrino observatory"),
            {"neutrino", "observatory", "neutrino observatory"},
        )

    def test_score_is_the_fraction_of_the_query_matched(self):
        terms = {"a", "b", "c", "d"}
        self.assertAlmostEqual(retrieval.score(terms, {"a", "b"}), 0.5)
        self.assertEqual(retrieval.score(terms, set()), 0.0)
        self.assertEqual(retrieval.score(set(), {"a"}), 0.0)

    def test_a_record_is_found_by_its_own_terms(self):
        memory.remember_external(
            self.knowledge, "src", LEARNABLE_TEXT, origin="doc.txt"
        )
        results = retrieval.retrieve(self.knowledge, "tell me about the observatory")
        self.assertTrue(results)
        self.assertEqual(results[0]["collection"], memory.EXTERNAL)
        self.assertIn("observatory", results[0]["matched"])

    def test_every_result_says_which_terms_matched(self):
        memory.remember_external(
            self.knowledge, "src", LEARNABLE_TEXT, origin="doc.txt"
        )
        for item in retrieval.retrieve(self.knowledge, "neutrino detector"):
            self.assertTrue(item["matched"], item)
            self.assertGreater(item["score"], 0.0)
            self.assertLessEqual(item["score"], 1.0)

    def test_every_result_says_how_it_was_matched(self):
        # Architecture 31 wants the method recorded with the result, and the
        # module docstring said so. It did not, and a score with no visible
        # basis is a score that overstates its evidence: a lexical match means
        # "these words appear together", not "this means the same thing".
        memory.remember_external(
            self.knowledge, "src", LEARNABLE_TEXT, origin="doc.txt"
        )
        results = retrieval.retrieve(self.knowledge, "neutrino detector")
        self.assertTrue(results)
        for item in results:
            self.assertEqual(item["matching"], retrieval.MATCHING)
            self.assertTrue(item["matching"].strip())

    def test_an_unrelated_question_returns_nothing(self):
        memory.remember_external(
            self.knowledge, "src", LEARNABLE_TEXT, origin="doc.txt"
        )
        self.assertEqual(
            retrieval.retrieve(self.knowledge, "photosynthesis chlorophyll"),
            [],
            "retrieval must not invent a match",
        )

    def test_an_empty_question_returns_nothing(self):
        memory.remember_external(self.knowledge, "src", "text", origin="doc.txt")
        self.assertEqual(retrieval.retrieve(self.knowledge, ""), [])

    def test_the_result_limit_is_respected(self):
        for index in range(6):
            memory.remember_external(
                self.knowledge, f"src{index}", LEARNABLE_TEXT, origin="doc.txt"
            )
        self.assertLessEqual(
            len(retrieval.retrieve(self.knowledge, "neutrino", limit=2)), 2
        )

    def test_context_block_names_the_source_of_each_recall(self):
        memory.remember_external(
            self.knowledge, "src", LEARNABLE_TEXT, origin="doc.txt"
        )
        block = retrieval.context_block(retrieval.retrieve(self.knowledge, "neutrino"))
        self.assertTrue(block.startswith("[recalled]"))
        self.assertIn("external_memory", block)
        self.assertIn("matched", block)

    def test_context_block_is_empty_when_nothing_was_recalled(self):
        self.assertEqual(retrieval.context_block([]), "")

    def test_browse_lists_the_store_without_a_query(self):
        memory.remember_external(self.knowledge, "src", LEARNABLE_TEXT, origin="d")
        memory.remember_long_term(self.knowledge, "lt", "a stable fact", confidence=0.8)
        found = retrieval.browse(self.knowledge)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["id"], "lt")
        self.assertEqual(found[0]["confidence"], 0.8)

    def test_an_empty_store_browses_to_nothing(self):
        self.assertEqual(retrieval.browse(self.knowledge), [])


class RetentionTests(TempDirTestCase):
    """Architecture 34 Memory, and Architecture 16 external knowledge."""

    def setUp(self):
        super().setUp()
        self.path = self.workdir / "knowledge.json"
        self.config = make_config(
            knowledge_store=str(self.path), concept_terms_per_cycle=5
        )

    def learn(self, learned=True, verdict="ACCEPT", cycle=1):
        learning_worker.record_cycle_knowledge(
            store_module.Store(self.path),
            self.config,
            cycle,
            "data/curated/neutrino.txt",
            LEARNABLE_TEXT,
            {
                "holdout_loss_before": 2.0,
                "holdout_loss_after": 1.0,
                "holdout_loss_improvement": 1.0,
                "successful_steps": 8,
                "gradient_recoveries": 0,
            },
            holdout_before=2.0,
            regression_delta=0.1,
            confidence=0.5,
            learning_rate=3e-4,
            learned=learned,
            verdict=verdict,
            verdict_reason="measured",
        )

    def test_knowledge_survives_the_process_that_learned_it(self):
        self.learn()
        # A brand new Store stands in for a later session or a restart.
        reopened = store_module.Store(self.path)
        results = retrieval.retrieve(reopened, "what does the observatory record")
        self.assertTrue(
            results,
            "a later session must be able to recall what an earlier one learned",
        )
        self.assertTrue(
            any(item["collection"] == memory.EXTERNAL for item in results),
            results,
        )

    def test_accepted_material_is_recallable_and_rejected_material_is_not(self):
        self.learn(learned=True, verdict="ACCEPT", cycle=1)
        self.learn(learned=False, verdict="REJECT", cycle=2)
        reopened = store_module.Store(self.path)
        external = [
            item["id"]
            for item in retrieval.browse(reopened)
            if item["collection"] == memory.EXTERNAL
        ]
        self.assertEqual(
            external,
            ["source-neutrino.txt"],
            "a rejected document must not be recorded as recallable knowledge",
        )

    def test_the_experience_itself_is_retrievable(self):
        self.learn()
        reopened = store_module.Store(self.path)
        results = retrieval.retrieve(reopened, "neutrino observatory")
        self.assertTrue(
            any(item["collection"] == memory.EXPERIENCE for item in results),
            results,
        )

    def test_concepts_are_retrievable(self):
        self.learn()
        reopened = store_module.Store(self.path)
        results = retrieval.retrieve(reopened, "neutrino observatory detector")
        self.assertTrue(
            any(item["collection"] == concepts.CONCEPTS for item in results),
            results,
        )


class ContextAssemblyTests(unittest.TestCase):
    def test_knowledge_is_kept_and_the_oldest_conversation_is_dropped(self):
        history = torch.arange(10).view(1, -1)
        prefix = torch.tensor([[100, 101, 102]])
        context = sample.assemble(prefix, history, block_size=8)
        self.assertEqual(context.shape[1], 8)
        self.assertTrue(torch.equal(context[0, :3], prefix[0]))
        self.assertTrue(torch.equal(context[0, 3:], torch.tensor([5, 6, 7, 8, 9])))

    def test_without_knowledge_the_conversation_uses_the_whole_window(self):
        history = torch.arange(10).view(1, -1)
        empty = torch.zeros((1, 0), dtype=torch.long)
        context = sample.assemble(empty, history, block_size=4)
        self.assertTrue(torch.equal(context[0], torch.tensor([6, 7, 8, 9])))

    def test_knowledge_larger_than_the_window_is_trimmed_not_rejected(self):
        # The model raises if the context exceeds block_size, so an oversized
        # recall has to be trimmed rather than passed through.
        history = torch.arange(5).view(1, -1)
        prefix = torch.zeros((1, 10), dtype=torch.long)
        context = sample.assemble(prefix, history, block_size=8)
        self.assertEqual(context.shape[1], 8)

    def test_the_stored_history_is_never_contaminated_by_knowledge(self):
        # assemble returns the generation context only; the caller keeps its own
        # history, so injected knowledge cannot accumulate in the transcript.
        history = torch.arange(4).view(1, -1)
        prefix = torch.tensor([[7, 7, 7]])
        sample.assemble(prefix, history, block_size=8)
        self.assertTrue(torch.equal(history, torch.arange(4).view(1, -1)))


class GenerationContextTests(unittest.TestCase):
    """The context a turn is answered from.

    A session's first turn arrives with an empty history, and when nothing was
    recalled the prefix is empty too. That produced a context of zero tokens,
    and generate() samples from the last token it is given, so the very first
    thing a person typed could end the session in an IndexError. These tests
    hold the context non-empty and the question inside it.
    """

    TURN = torch.tensor([[10, 11, 12, 13]])

    def test_the_first_turn_of_a_session_is_never_an_empty_context(self):
        empty_prefix = torch.zeros((1, 0), dtype=torch.long)
        empty_history = torch.zeros((1, 0), dtype=torch.long)
        context = sample.generation_context(
            empty_prefix, empty_history, self.TURN, block_size=32
        )
        self.assertGreater(context.shape[1], 0)
        self.assertTrue(torch.equal(context, self.TURN))

    def test_the_model_is_shown_the_turn_being_asked(self):
        # A model cannot answer a question it has not been given. The turn used
        # to be committed to the transcript only after generation, so the
        # answer was produced without ever seeing the question.
        empty_prefix = torch.zeros((1, 0), dtype=torch.long)
        empty_history = torch.zeros((1, 0), dtype=torch.long)
        context = sample.generation_context(
            empty_prefix, empty_history, self.TURN, block_size=32
        )
        self.assertTrue(
            torch.equal(context[0, -self.TURN.shape[1] :], self.TURN[0]),
            "the turn being asked must be the last thing the model reads",
        )

    def test_a_context_the_model_would_reject_is_never_built(self):
        empty_prefix = torch.zeros((1, 0), dtype=torch.long)
        history = torch.arange(64).view(1, -1)
        context = sample.generation_context(
            empty_prefix, history, self.TURN, block_size=16
        )
        self.assertLessEqual(context.shape[1], 16)

    def test_recalled_knowledge_still_comes_first(self):
        prefix = torch.tensor([[50, 51, 52]])
        history = torch.arange(4).view(1, -1)
        context = sample.generation_context(prefix, history, self.TURN, block_size=32)
        self.assertTrue(torch.equal(context[0, :3], prefix[0]))
        self.assertTrue(
            torch.equal(context[0, -self.TURN.shape[1] :], self.TURN[0]),
            "the turn being asked must survive even when knowledge is present",
        )

    def test_knowledge_large_enough_to_fill_the_window_still_leaves_a_context(self):
        # A recall that fills the whole window leaves no room for conversation.
        # The result is the knowledge alone, which is non-empty, so generation
        # still has something to continue from.
        prefix = torch.arange(10, 26).view(1, -1)
        context = sample.generation_context(
            prefix, torch.zeros((1, 0), dtype=torch.long), self.TURN, block_size=16
        )
        self.assertEqual(context.shape[1], 16)
        self.assertGreater(context.shape[1], 0)

    def test_the_context_is_never_empty_however_the_session_is_shaped(self):
        # The invariant, checked across the shapes a session actually takes:
        # first turn, a turn with recall, and a full window.
        empty_prefix = torch.zeros((1, 0), dtype=torch.long)
        for history in (
            torch.zeros((1, 0), dtype=torch.long),
            torch.arange(20).view(1, -1),
            torch.arange(512).view(1, -1),
        ):
            for prefix in (
                empty_prefix,
                torch.tensor([[1, 2, 3]]),
                torch.arange(600).view(1, -1),
            ):
                context = sample.generation_context(
                    prefix, history, self.TURN, block_size=512
                )
                self.assertGreater(
                    context.shape[1],
                    0,
                    f"empty context for history={history.shape} prefix={prefix.shape}",
                )
                self.assertLessEqual(context.shape[1], 512)
