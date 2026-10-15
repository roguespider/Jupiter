"""Persistent goals and autonomous scheduling (Upgrade 7 / ToDo.md 191-193).

A durable goal queue that survives restarts, carries priorities and review
deadlines, records why each action was chosen, and lets the loop rest when
nothing useful remains. It operates independently of train.py.

Usage:
    from goals import GoalQueue
    with GoalQueue("out/goals.json") as goals:
        goals.add("Improve conversation evaluation", priority=0.9,
                  review_by=time.time() + 86400 * 7)
        for g in goals.active():
            print(g.title, g.priority, g.status)
"""

import json
import time
from pathlib import Path
from typing import Any


class Goal:
    """One durable objective with priority, deadline, status, and reason."""

    def __init__(
        self,
        title: str,
        priority: float = 0.5,
        review_by: float | None = None,
        status: str = "open",
        reason: str = "",
        created: float | None = None,
        updated: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not title or not title.strip():
            raise ValueError("goal title must not be empty")
        self.title = " ".join(str(title).split())
        self.priority = float(max(0.0, min(1.0, priority)))
        self.review_by = float(review_by) if review_by is not None else None
        allowed = {"open", "in_progress", "blocked", "done", "cancelled", "resting"}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        self.status = status
        self.reason = str(reason).strip()
        self.created = float(created) if created is not None else time.time()
        self.updated = float(updated) if updated is not None else self.created
        self.metadata = dict(metadata or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "priority": round(self.priority, 4),
            "review_by": self.review_by,
            "status": self.status,
            "reason": self.reason,
            "created": self.created,
            "updated": self.updated,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"Goal({self.title!r}, priority={self.priority}, status={self.status!r})"


class GoalQueue:
    """Persistent goal queue backed by a JSON file (no external dependencies)."""

    def __init__(self, path: str | Path = "out/goals.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._goals: list[Goal] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._goals = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._goals = []
            return
        if not isinstance(data, list):
            self._goals = []
            return
        loaded = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                loaded.append(
                    Goal(
                        title=item.get("title", ""),
                        priority=item.get("priority", 0.5),
                        review_by=item.get("review_by"),
                        status=item.get("status", "open"),
                        reason=item.get("reason", ""),
                        created=item.get("created"),
                        updated=item.get("updated"),
                        metadata=item.get("metadata", {}),
                    )
                )
            except ValueError:
                # Skip corrupt entries rather than crashing the loop.
                continue
        self._goals = loaded

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                [g.as_dict() for g in self._goals],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def add(
        self,
        title: str,
        *,
        priority: float = 0.5,
        review_by: float | None = None,
        status: str = "open",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Goal:
        """Add a new durable goal to the queue."""
        goal = Goal(
            title=title,
            priority=priority,
            review_by=review_by,
            status=status,
            reason=reason,
            metadata=metadata,
        )
        self._goals.append(goal)
        self._save()
        return goal

    def active(self, include_blocked: bool = False) -> list[Goal]:
        """Goals that are not done or cancelled, ordered by priority then review time."""
        allowed = {"open", "in_progress", "resting"}
        if include_blocked:
            allowed.add("blocked")
        filtered = [g for g in self._goals if g.status in allowed]
        # Sort: highest priority first, then earliest review deadline.
        filtered.sort(
            key=lambda g: (
                -g.priority,
                g.review_by if g.review_by is not None else float("inf"),
            )
        )
        return filtered

    def done(self) -> list[Goal]:
        return [g for g in self._goals if g.status == "done"]

    def open(self) -> list[Goal]:
        return [g for g in self._goals if g.status == "open"]

    def update(
        self,
        title: str,
        *,
        status: str | None = None,
        priority: float | None = None,
        review_by: float | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Goal | None:
        """Update the first matching open goal by title."""
        for g in self._goals:
            if g.title == title and g.status in (
                "open",
                "in_progress",
                "blocked",
                "resting",
            ):
                if status is not None:
                    g.status = status
                if priority is not None:
                    g.priority = float(max(0.0, min(1.0, priority)))
                if review_by is not None:
                    g.review_by = float(review_by)
                if reason is not None:
                    g.reason = str(reason).strip()
                if metadata is not None:
                    g.metadata.update(metadata)
                g.updated = time.time()
                self._save()
                return g
        return None

    def complete(self, title: str, reason: str = "") -> Goal | None:
        """Mark a goal as done with an optional closing reason."""
        return self.update(title, status="done", reason=reason or "completed")

    def rest(self, title: str, reason: str = "") -> Goal | None:
        """Mark a goal as resting when nothing useful remains."""
        return self.update(
            title, status="resting", reason=reason or "resting: nothing useful to do"
        )

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self._goals),
            "open": len(self.open()),
            "in_progress": sum(1 for g in self._goals if g.status == "in_progress"),
            "blocked": sum(1 for g in self._goals if g.status == "blocked"),
            "resting": sum(1 for g in self._goals if g.status == "resting"),
            "done": len(self.done()),
            "cancelled": sum(1 for g in self._goals if g.status == "cancelled"),
        }

    def __enter__(self) -> "GoalQueue":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._save()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="goals",
        description="Persistent goal queue for autonomous scheduling.",
    )
    parser.add_argument(
        "--add",
        metavar="TITLE",
        help="add a new goal",
    )
    parser.add_argument(
        "--priority",
        type=float,
        default=0.5,
        help="priority 0.0-1.0 (default 0.5)",
    )
    parser.add_argument(
        "--review-in",
        type=int,
        default=0,
        metavar="SECONDS",
        help="review deadline in seconds from now (0 = none)",
    )
    parser.add_argument(
        "--complete",
        metavar="TITLE",
        help="mark a goal as done",
    )
    parser.add_argument(
        "--rest",
        metavar="TITLE",
        help="mark a goal as resting",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list active goals",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="show queue statistics",
    )
    args = parser.parse_args()

    with GoalQueue() as goals:
        if args.add:
            review_by = time.time() + args.review_in if args.review_in > 0 else None
            g = goals.add(
                args.add,
                priority=args.priority,
                review_by=review_by,
                reason="added via CLI",
            )
            print(
                f"Added goal: {g.title} (priority={g.priority}, review_by={g.review_by})"
            )
        elif args.complete:
            g = goals.complete(args.complete, reason="completed via CLI")
            if g:
                print(f"Completed goal: {g.title}")
            else:
                print(f"No open goal found with title: {args.complete}")
        elif args.rest:
            g = goals.rest(args.rest, reason="resting via CLI")
            if g:
                print(f"Rested goal: {g.title}")
            else:
                print(f"No open goal found with title: {args.rest}")
        elif args.list:
            active = goals.active()
            if not active:
                print("No active goals.")
            for g in active:
                review_str = (
                    f"review in {int((g.review_by - time.time()) / 3600)}h"
                    if g.review_by
                    else "no deadline"
                )
                print(f"[{g.status}] {g.title} (priority={g.priority}, {review_str})")
        elif args.stats:
            s = goals.stats()
            for k, v in sorted(s.items()):
                print(f"{k}: {v}")
        else:
            parser.print_help()


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)
