"""Background reflection worker (Upgrade 6 / ToDo.md 188-190).

A separate, lightweight worker that reviews new experiences, detects recurring
errors, proposes memories, updates task status, and identifies missing knowledge.
It runs without calling train.py and operates independently of the neural
learning cycle.

Usage:
    python reflection_worker.py [--once]

The --once flag runs a single reflection pass and exits; without it the worker
loops continuously with a configurable interval.
"""

import argparse
import json
import time
from pathlib import Path

from agent_memory import AgentMemory
from experience_loop import census, experiences_path, read

DEFAULT_INTERVAL = 60.0  # seconds between reflection passes


def review_experiences(memory: AgentMemory, c) -> list[dict]:
    """Read recent conversation experiences and return structured observations."""
    records, unreadable = read(c)
    observations = []
    for record in records[-20:]:  # review the most recent 20 interactions
        user_text = str(record.get("user", "")).strip()
        jupiter_text = str(record.get("jupiter", "")).strip()
        if not user_text or not jupiter_text:
            observations.append(
                {
                    "interaction": record.get("id", "unknown"),
                    "status": "incomplete",
                    "reason": "missing user text or assistant reply",
                }
            )
            continue
        # Detect recurring errors: very short replies may indicate failure
        reply_len = len(jupiter_text)
        if reply_len < 20:
            observations.append(
                {
                    "interaction": record.get("id", "unknown"),
                    "status": "potential_error",
                    "reason": f"reply too short ({reply_len} chars) — possible generation failure",
                    "user_text": user_text[:120],
                    "assistant_text": jupiter_text[:120],
                }
            )
        else:
            observations.append(
                {
                    "interaction": record.get("id", "unknown"),
                    "status": "complete",
                    "user_text_preview": user_text[:120],
                    "assistant_text_preview": jupiter_text[:120],
                }
            )
    return observations


def detect_recurring_errors(observations: list[dict]) -> list[dict]:
    """Identify patterns of repeated failures from observations."""
    errors = [o for o in observations if o.get("status") == "potential_error"]
    if not errors:
        return []
    # Group by similar user text patterns (simplified lexical grouping)
    patterns = {}
    for error in errors:
        preview = error.get("user_text", "")
        # Use first 3 words as a rough grouping key
        key_words = " ".join(preview.split()[:3]).lower()
        patterns.setdefault(key_words, []).append(error)
    recurring = []
    for key, group in patterns.items():
        if len(group) >= 2:
            recurring.append(
                {
                    "pattern_key": key,
                    "count": len(group),
                    "example_user_text": group[0].get("user_text", ""),
                    "recommendation": "Consider proposing a memory or task to address this recurring failure.",
                }
            )
    return recurring


def propose_memories(
    memory: AgentMemory, observations: list[dict], recurring_errors: list[dict]
) -> list[str]:
    """Propose new memories based on observations and recurring errors."""
    proposed_ids = []
    # Propose a memory for each recurring error pattern
    for error in recurring_errors:
        recommendation = error.get("recommendation", "")
        if recommendation:
            new_id = memory.remember(
                kind="procedure_insight",
                content=f"Recurring issue detected: {error['pattern_key']}. Recommendation: {recommendation}",
                source="reflection_worker",
                confidence=0.75,
                metadata={
                    "pattern_key": error["pattern_key"],
                    "count": error.get("count", 0),
                },
            )
            proposed_ids.append(new_id)
    # Propose a memory summarizing the overall observation set
    complete_count = sum(1 for o in observations if o.get("status") == "complete")
    error_count = sum(1 for o in observations if o.get("status") == "potential_error")
    summary_content = (
        f"Reflection pass: {complete_count} complete interactions, "
        f"{error_count} potential errors observed."
    )
    summary_id = memory.remember(
        kind="reflection_summary",
        content=summary_content,
        source="reflection_worker",
        confidence=0.8,
        metadata={"complete": complete_count, "potential_errors": error_count},
    )
    proposed_ids.append(summary_id)
    return proposed_ids


def update_task_status(
    memory: AgentMemory, observations: list[dict], recurring_errors: list[dict]
) -> None:
    """Update open tasks based on reflection findings."""
    open_tasks = memory.open_tasks()
    for task in open_tasks:
        # If recurring errors relate to a task, update its status
        for error in recurring_errors:
            if error.get("pattern_key", "") in task.get("title", ""):
                memory.update_task(
                    task["id"],
                    status="blocked",
                    next_action=f"Address recurring issue: {error['pattern_key']}",
                )


def identify_missing_knowledge(
    memory: AgentMemory, observations: list[dict]
) -> list[str]:
    """Identify gaps based on incomplete or failed interactions."""
    gaps = []
    for obs in observations:
        if obs.get("status") == "potential_error":
            user_text = obs.get("user_text", "")
            # Extract significant terms to describe the gap
            from agent_memory import _terms

            terms = sorted(_terms(user_text))[:5]
            gap_description = f"Possible knowledge gap related to: {' '.join(terms)}"
            gaps.append(gap_description)
    # Deduplicate gaps
    unique_gaps = sorted(set(gaps))
    return unique_gaps


def run_reflection(c, memory_path: str = "out/agent_memory.sqlite3") -> dict:
    """One reflection pass: observe, detect, propose, update, identify gaps."""
    memory = AgentMemory(memory_path)
    observations = review_experiences(memory, c)
    recurring_errors = detect_recurring_errors(observations)
    proposed_memory_ids = propose_memories(memory, observations, recurring_errors)
    update_task_status(memory, observations, recurring_errors)
    missing_knowledge = identify_missing_knowledge(memory, observations)
    memory.close()
    return {
        "observations": len(observations),
        "potential_errors": len(
            [o for o in observations if o.get("status") == "potential_error"]
        ),
        "recurring_errors": len(recurring_errors),
        "proposed_memories": len(proposed_memory_ids),
        "missing_knowledge_gaps": len(missing_knowledge),
        "missing_knowledge": missing_knowledge[:5],
        "note": "Reflection completed without calling train.py.",
    }


def main():
    parser = argparse.ArgumentParser(
        prog="reflection_worker",
        description="Background reflection: review experiences, detect errors, propose memories, update tasks, identify gaps.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single reflection pass and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="seconds between passes when running continuously",
    )
    args = parser.parse_args()

    import config

    c = config.load_config()

    if args.once:
        result = run_reflection(c)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    # Continuous mode: loop until interrupted
    import signal
    import sys

    stop_flag = False

    def _handle_signal(signum, frame):
        nonlocal stop_flag
        stop_flag = True
        print("Reflection worker stopping...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(
        f"Reflection worker started. Interval: {args.interval}s. Press Ctrl+C to stop."
    )
    while not stop_flag:
        try:
            result = run_reflection(c)
            print(f"Reflection pass complete: {json.dumps(result)}")
        except Exception as exc:
            print(f"Reflection error: {exc}")
        # Wait with periodic checks for stop signal
        for _ in range(int(args.interval)):
            if stop_flag:
                break
            time.sleep(1)
    print("Reflection worker finished.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
