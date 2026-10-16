"""Show past experiences grouped by outcome (Architecture 46).

Called as: python show_experiences.py [--limit N]

Displays experiences sorted by outcome (successes, failures, unrated)
so a developer can verify the experience replay system is recording
correctly.
"""

import json
import sys
from pathlib import Path

from experience_loop import experiences_path


def main():
    limit = 5
    if len(sys.argv) > 2 and sys.argv[1] == "--limit":
        try:
            limit = int(sys.argv[2])
        except ValueError:
            pass

    path = experiences_path(type("C", (), {"out_dir": "out"})())
    if not path.exists():
        print("No experiences recorded yet.")
        return

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type", "").lower() == "conversation":
                    records.append(rec)
            except (json.JSONDecodeError, ValueError):
                pass

    if not records:
        print("No experiences recorded yet.")
        return

    # Limit to most recent N records for display.
    records = records[-limit:]

    successes = [
        r for r in records if r.get("outcome", "").lower() in ("helpful", "success")
    ]
    failures = [
        r for r in records if r.get("outcome", "").lower() in ("wrong", "failure")
    ]
    unrated = [r for r in records if r not in successes and r not in failures]

    print(
        f"Experience replay ({len(records)} total, "
        f"{len(successes)} success, {len(failures)} failure, "
        f"{len(unrated)} unrated):"
    )
    print()

    if successes:
        print("[successes - these worked and can be reused]")
        for i, r in enumerate(successes, 1):
            user = str(r.get("user", "")).strip()[:120]
            attempt = str(r.get("attempt", "")).strip()[:120]
            expected = str(r.get("expected", "")).strip()[:120]
            outcome = r.get("outcome", "")
            rec = str(r.get("recommendation", "")).strip()[:120]
            print(f"  [{i}] user: {user}")
            if attempt:
                print(f"      tried: {attempt}")
            if expected:
                print(f"      expected: {expected}")
            if outcome:
                print(f"      outcome: {outcome}")
            if rec:
                print(f"      next: {rec}")
            print()

    if failures:
        print("[failures - cautionary evidence]")
        for i, r in enumerate(failures, 1):
            user = str(r.get("user", "")).strip()[:120]
            attempt = str(r.get("attempt", "")).strip()[:120]
            expected = str(r.get("expected", "")).strip()[:120]
            outcome = r.get("outcome", "")
            rec = str(r.get("recommendation", "")).strip()[:120]
            print(f"  [{i}] user: {user}")
            if attempt:
                print(f"      tried: {attempt}")
            if expected:
                print(f"      expected: {expected}")
            if outcome:
                print(f"      outcome: {outcome}")
            if rec:
                print(f"      next: {rec}")
            print()

    if unrated:
        print("[unrated - no outcome recorded]")
        for i, r in enumerate(unrated, 1):
            user = str(r.get("user", "")).strip()[:120]
            attempt = str(r.get("attempt", "")).strip()[:120]
            print(f"  [{i}] user: {user}")
            if attempt:
                print(f"      tried: {attempt}")
            print()


if __name__ == "__main__":
    main()
