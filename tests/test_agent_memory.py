"""Smoke tests for the additive no-training memory layer."""

import tempfile
import unittest
from pathlib import Path

from agent_memory import AgentMemory


class AgentMemoryTests(unittest.TestCase):
    def test_remember_recall_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            with AgentMemory(path) as memory:
                key = memory.remember(
                    "fact", "Jupiter uses SafeTensors", source="test", confidence=0.9
                )
                self.assertTrue(key)
                results = memory.recall("what format does Jupiter use")
                self.assertTrue(results)
                self.assertEqual(results[0]["content"], "Jupiter uses SafeTensors")
            with AgentMemory(path) as memory:
                self.assertEqual(memory.stats()["active_memories"], 1)

    def test_tasks_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                task_id = memory.add_task("Wire memory into sample.py", priority=0.8)
                self.assertEqual(memory.open_tasks()[0]["id"], task_id)
                interaction_id = memory.log_interaction("question", "answer")
                memory.feedback(interaction_id, "helpful", "worked")
                self.assertEqual(memory.stats()["rated_interactions"], 1)

    def test_correction_supersedes_old_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                memory.remember(
                    "fact",
                    "Jupiter uses JSON files for its model weights",
                    source="test",
                )
                memory.correct(
                    "Jupiter uses JSON files for its model weights",
                    "Jupiter stores model weights in SafeTensors",
                )
                results = memory.recall("Jupiter model weights")
                self.assertTrue(
                    any("SafeTensors" in item["content"] for item in results)
                )
                self.assertEqual(memory.stats()["superseded_memories"], 1)

    def test_remember_with_deduplication(self):
        """Duplicate content reinforces confidence instead of creating a second row."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                key1 = memory.remember(
                    "fact", "Jupiter uses SafeTensors", source="test", confidence=0.6
                )
                key2 = memory.remember(
                    "fact", "Jupiter uses SafeTensors", source="test", confidence=0.8
                )
                self.assertEqual(key1, key2)
                results = memory.recall("what format does Jupiter use")
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0]["confidence"] > 0.6)

    def test_remember_validates_confidence_range(self):
        """remember() rejects confidence outside [0, 1]."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                with self.assertRaises(ValueError):
                    memory.remember("fact", "test", confidence=1.5)
                with self.assertRaises(ValueError):
                    memory.remember("fact", "test", confidence=-0.1)

    def test_remember_requires_non_empty_fields(self):
        """remember() raises ValueError for empty kind or content."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                with self.assertRaises(ValueError):
                    memory.remember("", "some content")
                with self.assertRaises(ValueError):
                    memory.remember("kind", "")

    def test_recall_empty_query_returns_empty(self):
        """recall() with no meaningful terms returns []."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                memory.remember("fact", "Jupiter uses SafeTensors", source="test")
                results = memory.recall("")
                self.assertEqual(results, [])

    def test_recall_with_kinds_filter(self):
        """recall() with kinds=[...] only returns matching kinds."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                memory.remember("fact", "Jupiter uses SafeTensors", source="test")
                memory.remember("opinion", "I think it works well", source="test")
                results = memory.recall("Jupiter", kinds=["fact"])
                self.assertTrue(all(r["kind"] == "fact" for r in results))

    def test_log_interaction_returns_key(self):
        """log_interaction() returns a non-empty string key."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                key = memory.log_interaction("hello", "hi there")
                self.assertTrue(key)
                self.assertGreater(len(key), 0)

    def test_feedback_on_unknown_interaction(self):
        """feedback() raises KeyError for unknown interaction_id."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                with self.assertRaises(KeyError):
                    memory.feedback("nonexistent", "helpful")

    def test_feedback_rejects_invalid_outcome(self):
        """feedback() raises ValueError for invalid outcome."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                key = memory.log_interaction("q", "a")
                with self.assertRaises(ValueError):
                    memory.feedback(key, "invalid_outcome")

    def test_open_tasks_sorts_by_priority(self):
        """open_tasks() returns tasks sorted by priority DESC."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                memory.add_task("Low priority", priority=0.2)
                memory.add_task("High priority", priority=0.9)
                tasks = memory.open_tasks()
                self.assertEqual(tasks[0]["priority"], 0.9)

    def test_add_task_validates_title(self):
        """add_task() raises ValueError for empty title."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                with self.assertRaises(ValueError):
                    memory.add_task("")

    def test_update_task_validates_status(self):
        """update_task() raises ValueError for invalid status."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                key = memory.add_task("test")
                with self.assertRaises(ValueError):
                    memory.update_task(key, status="invalid_status")

    def test_update_task_unknown_id(self):
        """update_task() raises KeyError for unknown task_id."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                with self.assertRaises(KeyError):
                    memory.update_task("nonexistent", status="done")

    def test_recent_events(self):
        """recent_events() returns a list of event dicts."""
        with tempfile.TemporaryDirectory() as tmp:
            with AgentMemory(Path(tmp) / "memory.sqlite3") as memory:
                memory.remember("fact", "test content", source="test")
                events = memory.recent_events(limit=10)
                self.assertIsInstance(events, list)
                self.assertGreater(len(events), 0)
                self.assertIn("event_type", events[0])


if __name__ == "__main__":
    unittest.main()
