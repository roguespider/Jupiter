"""Jupiter no-training persistent agent memory.

Uses only Python's standard library. This stores facts, corrections, preferences,
procedures, tasks, and experiences in SQLite. It never loads or changes model
weights and never imports PyTorch.

Quick use:
    from agent_memory import AgentMemory
    with AgentMemory("out/agent_memory.sqlite3") as memory:
        memory.remember("fact", "Jupiter uses SafeTensors", source="user", confidence=0.95)
        print(memory.recall("what format does Jupiter use?"))
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

_WORDS = re.compile(r"[a-zA-Z0-9_+#.-]{2,}")
_STOP = frozenset(
    """
a an and are as at be been but by can did do does for from had has have he her
here hers him his how i if in into is it its me my of on or our ours she so than
that the their theirs them then there these they this those to us was we were what
when where which who why will with you your yours not
""".split()
)


def _terms(value: str) -> set[str]:
    return {w.lower() for w in _WORDS.findall(value) if w.lower() not in _STOP}


def _now() -> float:
    return time.time()


class AgentMemory:
    """Persistent, searchable memory with provenance and explicit feedback."""

    def __init__(self, path: str | Path = "out/agent_memory.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                normalized TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                created REAL NOT NULL,
                updated REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_access REAL,
                status TEXT NOT NULL DEFAULT 'active',
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                created REAL NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                outcome TEXT NOT NULL DEFAULT 'unrated',
                feedback TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                priority REAL NOT NULL DEFAULT 0.5,
                next_action TEXT NOT NULL DEFAULT '',
                created REAL NOT NULL,
                updated REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                event_type TEXT NOT NULL,
                detail TEXT NOT NULL
            )
        """)
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_kind_status ON memories(kind,status)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status_priority ON tasks(status,priority)"
        )
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prerequisites TEXT NOT NULL DEFAULT '[]',
                steps TEXT NOT NULL DEFAULT '[]',
                expected_result TEXT NOT NULL DEFAULT '',
                verification_result TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                created REAL NOT NULL,
                updated REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_procedures_name ON procedures(name)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_procedures_status ON procedures(status)"
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "AgentMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _normalize(content: str) -> str:
        return " ".join(content.casefold().split())

    def remember(
        self,
        kind: str,
        content: str,
        *,
        source: str = "conversation",
        confidence: float = 0.6,
        metadata: dict[str, Any] | None = None,
        deduplicate: bool = True,
    ) -> str:
        """Save a memory. Duplicate content updates evidence, not a second row."""
        kind = kind.strip().lower()
        content = " ".join(str(content).split())
        if not kind or not content:
            raise ValueError("kind and content must not be empty")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        normalized = self._normalize(content)
        now = _now()
        if deduplicate:
            row = self.db.execute(
                "SELECT id, confidence, metadata FROM memories "
                "WHERE kind=? AND normalized=? AND status='active' LIMIT 1",
                (kind, normalized),
            ).fetchone()
            if row:
                # Combine repeated evidence conservatively; never claim certainty.
                old = float(row["confidence"])
                combined = min(0.99, 1 - (1 - old) * (1 - confidence * 0.5))
                merged_meta = json.loads(row["metadata"])
                if metadata:
                    merged_meta.update(metadata)
                self.db.execute(
                    "UPDATE memories SET confidence=?, source=?, updated=?, metadata=? WHERE id=?",
                    (
                        combined,
                        source,
                        now,
                        json.dumps(merged_meta, ensure_ascii=False),
                        row["id"],
                    ),
                )
                self._event("memory_reinforced", f"{row['id']}:{kind}")
                self.db.commit()
                return str(row["id"])

        key = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO memories(id,kind,content,normalized,source,confidence,created,updated,metadata) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                key,
                kind,
                content,
                normalized,
                source,
                confidence,
                now,
                now,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self._event("memory_added", f"{key}:{kind}")
        self.db.commit()
        return key

    def correct(
        self, old_text: str, corrected_text: str, *, source: str = "user correction"
    ) -> str:
        """Retire a conflicting matching memory and save the correction explicitly."""
        old_terms = _terms(old_text)
        rows = self.db.execute(
            "SELECT id,content FROM memories WHERE status='active' ORDER BY updated DESC"
        ).fetchall()
        retired = []
        for row in rows:
            existing = _terms(row["content"])
            overlap = len(old_terms & existing) / max(1, len(old_terms | existing))
            if overlap >= 0.45 or self._normalize(row["content"]) == self._normalize(
                old_text
            ):
                retired.append(row["id"])
        with self.db:
            for key in retired:
                self.db.execute(
                    "UPDATE memories SET status='superseded',updated=? WHERE id=?",
                    (_now(), key),
                )
            new_id = self.remember(
                "correction",
                corrected_text,
                source=source,
                confidence=0.95,
                metadata={"corrects": old_text, "supersedes": retired},
                deduplicate=False,
            )
            self._event("correction_saved", f"retired={len(retired)};new={new_id}")
        return new_id

    def recall(
        self, query: str, *, limit: int = 8, kinds: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Lexical retrieval ranked by overlap, confidence, recency, and use."""
        query_terms = _terms(query)
        if not query_terms:
            return []
        params: list[Any] = []
        sql = "SELECT * FROM memories WHERE status='active'"
        if kinds:
            sql += " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            params.extend(kinds)
        rows = self.db.execute(sql, params).fetchall()
        now = _now()
        scored = []
        for row in rows:
            content_terms = _terms(row["content"])
            overlap = len(query_terms & content_terms) / max(1, len(query_terms))
            if overlap <= 0:
                continue
            age_days = max(0.0, (now - float(row["updated"])) / 86400)
            recency = 1.0 / (1.0 + age_days / 30.0)
            usage = min(1.0, float(row["access_count"]) / 10.0)
            score = overlap * (
                0.65 + 0.25 * float(row["confidence"]) + 0.07 * recency + 0.03 * usage
            )
            item = dict(row)
            item["metadata"] = json.loads(item["metadata"])
            item["score"] = round(score, 4)
            scored.append(item)
        scored.sort(key=lambda x: (-x["score"], -x["updated"]))
        chosen = scored[: max(0, int(limit))]
        with self.db:
            for item in chosen:
                self.db.execute(
                    "UPDATE memories SET access_count=access_count+1,last_access=? WHERE id=?",
                    (now, item["id"]),
                )
        return chosen

    def log_interaction(
        self,
        user_text: str,
        assistant_text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        key = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO interactions(id,created,user_text,assistant_text,metadata) VALUES(?,?,?,?,?)",
            (
                key,
                _now(),
                user_text,
                assistant_text,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self.db.commit()
        return key

    def feedback(self, interaction_id: str, outcome: str, feedback: str = "") -> None:
        if outcome not in {"helpful", "wrong", "partial", "uncertain"}:
            raise ValueError("outcome must be helpful, wrong, partial, or uncertain")
        row = self.db.execute(
            "SELECT id FROM interactions WHERE id=?", (interaction_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"unknown interaction: {interaction_id}")
        with self.db:
            self.db.execute(
                "UPDATE interactions SET outcome=?,feedback=? WHERE id=?",
                (outcome, feedback, interaction_id),
            )
            self._event("feedback", f"{interaction_id}:{outcome}:{feedback}")

    def add_task(
        self, title: str, *, priority: float = 0.5, next_action: str = ""
    ) -> str:
        title = " ".join(title.split())
        if not title:
            raise ValueError("task title must not be empty")
        key, now = uuid.uuid4().hex, _now()
        self.db.execute(
            "INSERT INTO tasks(id,title,status,priority,next_action,created,updated) VALUES(?,?,?,?,?,?,?)",
            (
                key,
                title,
                "open",
                max(0.0, min(1.0, float(priority))),
                next_action,
                now,
                now,
            ),
        )
        self.db.commit()
        self._event("task_added", f"{key}:{title}")
        return key

    def update_task(
        self, task_id: str, *, status: str | None = None, next_action: str | None = None
    ) -> None:
        allowed = {"open", "in_progress", "blocked", "done", "cancelled"}
        if status is not None and status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        row = self.db.execute("SELECT id FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown task: {task_id}")
        fields, values = [], []
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if next_action is not None:
            fields.append("next_action=?")
            values.append(next_action)
        fields.append("updated=?")
        values.append(_now())
        values.append(task_id)
        self.db.execute(f"UPDATE tasks SET {','.join(fields)} WHERE id=?", values)
        self.db.commit()

    def open_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM tasks WHERE status IN ('open','in_progress','blocked') "
            "ORDER BY priority DESC,updated ASC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        def count(table: str, where: str = "1=1") -> int:
            return int(
                self.db.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {where}"
                ).fetchone()[0]
            )

        return {
            "active_memories": count("memories", "status='active'"),
            "superseded_memories": count("memories", "status='superseded'"),
            "interactions": count("interactions"),
            "rated_interactions": count("interactions", "outcome!='unrated'"),
            "open_tasks": count("tasks", "status IN ('open','in_progress','blocked')"),
        }

    def _event(self, event_type: str, detail: str) -> None:
        self.db.execute(
            "INSERT INTO events(created,event_type,detail) VALUES(?,?,?)",
            (_now(), event_type, detail[:2000]),
        )

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT created,event_type,detail FROM events ORDER BY id DESC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_procedure(
        self,
        name: str,
        prerequisites: list[str] | None = None,
        steps: list[str] | None = None,
        expected_result: str = "",
        verification_result: str = "",
        confidence: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a reusable procedural skill with prerequisites, steps, expected and verification results."""
        name = " ".join(str(name).split())
        if not name:
            raise ValueError("procedure name must not be empty")
        prerequisites_json = json.dumps(prerequisites or [], ensure_ascii=False)
        steps_json = json.dumps(steps or [], ensure_ascii=False)
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        key = uuid.uuid4().hex
        now = _now()
        self.db.execute(
            "INSERT INTO procedures(id,name,prerequisites,steps,expected_result,verification_result,confidence,created,updated,status,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                name,
                prerequisites_json,
                steps_json,
                expected_result,
                verification_result,
                confidence,
                now,
                now,
                "active",
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self.db.commit()
        self._event("procedure_saved", f"{key}:{name}")
        return key

    def get_procedure(self, procedure_id: str) -> dict[str, Any] | None:
        """Retrieve a procedure by its ID."""
        row = self.db.execute(
            "SELECT * FROM procedures WHERE id=? AND status='active'", (procedure_id,)
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["prerequisites"] = json.loads(item["prerequisites"])
        item["steps"] = json.loads(item["steps"])
        item["metadata"] = json.loads(item["metadata"])
        return item

    def list_procedures(
        self, status: str = "active", limit: int = 20
    ) -> list[dict[str, Any]]:
        """List procedures, optionally filtered by status."""
        allowed = {"active", "deprecated", "verified", "rejected"}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        rows = self.db.execute(
            "SELECT * FROM procedures WHERE status=? ORDER BY updated DESC LIMIT ?",
            (status, max(0, int(limit))),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["prerequisites"] = json.loads(item["prerequisites"])
            item["steps"] = json.loads(item["steps"])
            item["metadata"] = json.loads(item["metadata"])
            result.append(item)
        return result

    def verify_procedure(
        self,
        procedure_id: str,
        verification_result: str,
        confidence: float | None = None,
    ) -> None:
        """Record verification result for a procedure and optionally update confidence."""
        row = self.db.execute(
            "SELECT id FROM procedures WHERE id=? AND status='active'", (procedure_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"unknown or inactive procedure: {procedure_id}")
        fields, values = ["verification_result=?"], [verification_result]
        if confidence is not None:
            confidence = float(confidence)
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")
            fields.append("confidence=?")
            values.append(confidence)
        fields.append("updated=?")
        values.append(_now())
        values.append(procedure_id)
        self.db.execute(f"UPDATE procedures SET {','.join(fields)} WHERE id=?", values)
        self.db.commit()
        self._event("procedure_verified", f"{procedure_id}:{verification_result}")

    def update_procedure(
        self,
        procedure_id: str,
        *,
        name: str | None = None,
        prerequisites: list[str] | None = None,
        steps: list[str] | None = None,
        expected_result: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update fields of an existing procedure."""
        allowed = {"active", "deprecated", "verified", "rejected"}
        if status is not None and status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        row = self.db.execute(
            "SELECT id FROM procedures WHERE id=? AND status='active'", (procedure_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"unknown or inactive procedure: {procedure_id}")
        fields, values = [], []
        if name is not None:
            name = " ".join(str(name).split())
            if not name:
                raise ValueError("procedure name must not be empty")
            fields.append("name=?")
            values.append(name)
        if prerequisites is not None:
            fields.append("prerequisites=?")
            values.append(json.dumps(prerequisites, ensure_ascii=False))
        if steps is not None:
            fields.append("steps=?")
            values.append(json.dumps(steps, ensure_ascii=False))
        if expected_result is not None:
            fields.append("expected_result=?")
            values.append(str(expected_result))
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if metadata is not None:
            fields.append("metadata=?")
            values.append(json.dumps(metadata, ensure_ascii=False))
        if fields:
            fields.append("updated=?")
            values.append(_now())
            values.append(procedure_id)
            self.db.execute(
                f"UPDATE procedures SET {','.join(fields)} WHERE id=?", values
            )
            self.db.commit()
            self._event("procedure_updated", f"{procedure_id}:{','.join(fields)}")
