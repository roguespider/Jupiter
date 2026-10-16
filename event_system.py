#!/usr/bin/env python3
"""Runtime event system for Jupiter.

Every important action produces a structured event with:
- unique event ID (reconnect-safe)
- timestamp
- session ID
- event type
- relevant data

Event types:
  message.received, context.retrieved, memory.recalled,
  concept.activated, relationship.proposed, response.token,
  feedback.received, learning.update, worker.status, runtime.error
"""

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    event_id: str
    timestamp: float
    session_id: str
    event_type: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "data": self.data,
        }


class EventStore:
    """In-memory event store with reconnect-safe IDs."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.events: List[Event] = []
        self.event_index: Dict[str, int] = {}  # event_id -> index

    def emit(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Event:
        event = Event(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            session_id=session_id or self.session_id,
            event_type=event_type,
            data=data or {},
        )
        self.events.append(event)
        self.event_index[event.event_id] = len(self.events) - 1
        return event

    def get_by_id(self, event_id: str) -> Optional[Event]:
        idx = self.event_index.get(event_id)
        if idx is not None and 0 <= idx < len(self.events):
            return self.events[idx]
        return None

    def get_since(self, event_id: str) -> List[Event]:
        """Return all events after the given event ID (reconnect-safe)."""
        idx = self.event_index.get(event_id)
        if idx is None:
            return list(self.events)
        return self.events[idx + 1 :]

    def get_all(self) -> List[Event]:
        return list(self.events)

    def get_last_n(self, n: int = 50) -> List[Event]:
        return self.events[-n:] if len(self.events) > n else list(self.events)

    def clear(self):
        self.events.clear()
        self.event_index.clear()


# Global event store for the runtime
_global_store: Optional[EventStore] = None


def get_store(session_id: Optional[str] = None) -> EventStore:
    global _global_store
    if _global_store is None:
        _global_store = EventStore(session_id=session_id)
    return _global_store


def reset_store(session_id: Optional[str] = None):
    global _global_store
    _global_store = EventStore(session_id=session_id)
