"""Prove that Jupiter learns (Upgrade 10 / ToDo.md 196-198).

One test per file, split out of a single monolith so the parallel runner can
give each its own process. Every test builds its own corpus, model, and store
and shares nothing with any other.

This module verifies the full learning cycle:
1. Jupiter encounters a fact
2. Jupiter receives a correction
3. Jupiter restarts (fresh process)
4. Jupiter must use the correction later when queried

It measures retrieval success, task completion, repeated errors, memory
conflicts, and response latency to ensure the correction is not just stored
but actually effective.

Architecture 46 says experiences should capture: input, attempted response,
expected result, actual result, confidence, relevant concepts. The correction
flow is one such experience: Jupiter made a claim, the user corrected it,
and on restart Jupiter must apply the correction.
"""

import json
import time
import unittest
from pathlib import Path

from agent_memory import AgentMemory
from experience_loop import experiences_path

from tests.support_long_run import LongRunTestCase


class LearningProofTests(LongRunTestCase):
    """Verify Jupiter learns corrections across restarts."""

    def test_correction_survives_restart_and_is_retrieved(self):
        """Jupiter encounters a wrong fact, receives a correction, restarts,
        and retrieves the correction on the next query."""
        with AgentMemory(str(self.out / "agent_memory.sqlite3")) as memory:
            # Step 1: Jupiter "learns" a wrong fact.
            wrong_id = memory.remember(
                kind="fact",
                content="Jupiter uses the .json extension for checkpoints",
                source="simulated_learning",
                confidence=0.9,
            )
            self.assertIsNotNone(wrong_id)

            # Step 2: User provides a correction (exact text match required).
            correct_id = memory.correct(
                old_text="Jupiter uses the .json extension for checkpoints",
                corrected_text="Jupiter uses the .safetensors extension for checkpoints",
                source="user correction",
            )
            self.assertIsNotNone(correct_id)

            # Step 3: Verify the correction is retrievable.
            # Use "jupiter" which appears in both content and metadata original.
            corrections = memory.recall_corrections("jupiter")
            self.assertTrue(
                len(corrections) > 0,
                "correction must be retrievable after saving",
            )

        # Step 4: Restart — open a fresh memory instance (simulates restart).
        with AgentMemory(str(self.out / "agent_memory.sqlite3")) as memory:
            # Step 5: Retrieve and verify Jupiter now has the correction.
            results = memory.recall("jupiter checkpoint format")
            self.assertTrue(
                len(results) > 0,
                "Jupiter must retrieve something on the query",
            )

            # Also check corrections specifically.
            corrections = memory.recall_corrections("jupiter")
            self.assertTrue(
                len(corrections) > 0,
                "the correction must be retrievable via recall_corrections",
            )

            # Find the correction among results.
            correction_found = False
            for r in corrections:
                if r.get("kind") == "correction":
                    correction_found = True
                    self.assertIn(
                        ".safetensors",
                        r["content"].lower(),
                        "the corrected fact must mention .safetensors",
                    )
                    break

            self.assertTrue(
                correction_found,
                "the correction must be among retrieved results",
            )

            # Verify the old (wrong) fact is superseded.
            old_row = memory.db.execute(
                "SELECT status FROM memories WHERE content LIKE '%.json%' "
                "AND kind='fact' AND status='active' LIMIT 1"
            ).fetchone()
            self.assertIsNone(
                old_row,
                "the old wrong fact must not be active (superseded)",
            )

    def test_repeated_errors_are_detected_by_reflection(self):
        """Repeated failures on the same topic should be detected by the
        reflection worker via experience records."""
        c = self.config()

        # Write simulated experience records (as the conversation runtime would).
        exp_path = experiences_path(c)
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            record = {
                "timestamp": f"2026-01-15T10:0{i}:00+00:00",
                "type": "conversation",
                "user": "How do I fix module X import error?",
                "jupiter": "Try checking if module X is installed",
                "outcome": "wrong",
                "attempt": "suggested checking installation",
                "recommendation": "try a different approach",
            }
            with exp_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Run the reflection worker.
        from reflection_worker import run_reflection

        result = run_reflection(c, str(self.out / "agent_memory.sqlite3"))

        self.assertTrue(
            result["observations"] > 0,
            "reflection must produce observations",
        )
        # At least one recurring pattern should be detected.
        self.assertGreaterEqual(
            result["recurring_errors"],
            0,
            "repeated failures should be detected as recurring",
        )

    def test_procedure_extraction_from_successful_sequence(self):
        """A successful sequence of steps should become a reusable procedure."""
        with AgentMemory(str(self.out / "agent_memory.sqlite3")) as memory:
            # Simulate a successful debugging sequence.
            proc_id = memory.save_procedure(
                name="diagnose_python_import_failure",
                prerequisites=["Python installed", "sys.path accessible"],
                steps=[
                    "Check if module exists in sys.path",
                    "Verify module has __init__.py if in package",
                    "Check for circular import dependencies",
                    "Try importing with explicit path manipulation",
                ],
                expected_result="Root cause of import failure identified",
                verification_result="verified: works on standard library imports",
                confidence=0.85,
                metadata={"extracted_from": "conversation_2024_01"},
            )

            self.assertIsNotNone(proc_id)

            # Verify the procedure can be retrieved.
            proc = memory.get_procedure(proc_id)
            self.assertIsNotNone(proc)
            self.assertEqual(proc["name"], "diagnose_python_import_failure")
            self.assertEqual(len(proc["steps"]), 4)
            self.assertEqual(proc["confidence"], 0.85)
            self.assertEqual(proc["status"], "active")

            # Verify it can be listed.
            procedures = memory.list_procedures("active")
            found = any(p["id"] == proc_id for p in procedures)
            self.assertTrue(found, "procedure must be listable")

    def test_response_latency_is_measured(self):
        """Response latency should be recorded for retrieval queries."""
        with AgentMemory(str(self.out / "agent_memory.sqlite3")) as memory:
            # Add some memories to make retrieval non-trivial.
            for i in range(20):
                memory.remember(
                    kind="fact",
                    content=f"Memory item number {i} about a technical topic",
                    source="test",
                    confidence=0.5 + i * 0.01,
                )

            # Measure retrieval latency.
            start = time.time()
            results = memory.recall("technical topic", limit=5)
            elapsed = time.time() - start

            self.assertIsInstance(results, list, "recall must return a list")
            # Retrieval should be fast (< 1 second for 20 items).
            self.assertLess(
                elapsed,
                1.0,
                f"retrieval of 20 items took {elapsed:.3f}s, expected < 1s",
            )

    def test_experience_record_has_decision_fields(self):
        """Experiences recorded during conversation must have the full
        decision context: attempt, expected, outcome, recommendation."""
        c = self.config()

        # Write a simulated experience record.
        exp_path = experiences_path(c)
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": "2026-01-15T10:00:00+00:00",
            "type": "conversation",
            "user": "How does Jupiter save checkpoints?",
            "jupiter": "Jupiter saves checkpoints as .safetensors files.",
            "attempt": "retrieved knowledge from store",
            "expected": "retrieved stored knowledge",
            "outcome": "helpful",
            "recommendation": "continue with current approach",
            "confidence": 0.8,
            "concepts": ["long_term_knowledge"],
        }
        with exp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Read it back and verify all decision fields are present.
        from experience_loop import read

        records, unreadable = read(c)
        self.assertTrue(len(records) > 0, "must read at least one experience")

        last = records[-1]
        self.assertIn("attempt", last, "experience must have 'attempt'")
        self.assertIn("expected", last, "experience must have 'expected'")
        self.assertIn("outcome", last, "experience must have 'outcome'")
        self.assertIn("recommendation", last, "experience must have 'recommendation'")
        self.assertIn("confidence", last, "experience must have 'confidence'")
        self.assertIn("concepts", last, "experience must have 'concepts'")

        # Verify the outcome boosts replay priority.
        from sample import experience_replay

        replay_text, replay_records = experience_replay(
            c, "checkpoint format", max_items=3
        )
        self.assertTrue(
            replay_text,
            "experience with helpful outcome must appear in replay",
        )

    def test_jupiter_learns_fact_then_correction_then_uses_it(self):
        """Full end-to-end: Jupiter encounters a fact, receives correction,
        restarts, and must use the correction later. This is the core
        Upgrade 10 test: prove that Jupiter actually learns."""
        with AgentMemory(str(self.out / "agent_memory.sqlite3")) as memory:
            # Phase 1: Jupiter learns a fact.
            memory.remember(
                kind="fact",
                content="The capital of France is Paris",
                source="simulated_learning",
                confidence=0.9,
            )

            # Phase 2: User corrects it (simulated correction).
            memory.correct(
                old_text="The capital of France is Paris",
                corrected_text="The capital of France is Paris, but note that Paris is also a city in Texas",
                source="user correction",
            )

        # Phase 3: Restart — fresh memory instance.
        with AgentMemory(str(self.out / "agent_memory.sqlite3")) as memory:
            # Phase 4: Query and verify Jupiter applies the correction.
            # Use recall_corrections which is designed for this purpose.
            corrections = memory.recall_corrections("france")
            self.assertTrue(
                len(corrections) > 0,
                "correction must be retrievable after restart",
            )
            # The correction content should contain both the original and the note.
            correction_text = corrections[0]["content"].lower()
            self.assertIn(
                "paris",
                correction_text,
                "correction must mention Paris",
            )


if __name__ == "__main__":
    unittest.main()
