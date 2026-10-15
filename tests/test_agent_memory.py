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
                key = memory.remember("fact", "Jupiter uses SafeTensors", source="test", confidence=0.9)
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
                memory.remember("fact", "Jupiter uses JSON files for its model weights", source="test")
                memory.correct("Jupiter uses JSON files for its model weights",
                               "Jupiter stores model weights in SafeTensors")
                results = memory.recall("Jupiter model weights")
                self.assertTrue(any("SafeTensors" in item["content"] for item in results))
                self.assertEqual(memory.stats()["superseded_memories"], 1)


if __name__ == "__main__":
    unittest.main()
