"""Watch or summarise Jupiter's learning telemetry.

Two modes, so the entry point can use this file without blocking on it:

    --summary   print the last N events and exit
    --follow    keep printing new events as they arrive (the default)

The follower tracks a byte offset instead of re-reading the file. Telemetry is
append-only and grows without bound -- a single sandboxed two-cycle run already
produced a 13MB checkpoint and a multi-megabyte event stream -- so re-reading
the whole file once a second is what made the previous version of this script
unusable as soon as a run got long.

    python watch_training.py                 follow out/telemetry.jsonl
    python watch_training.py --summary       print the last 20 events
    python watch_training.py --summary --limit 5
"""

import argparse
import json
import time
from pathlib import Path

from telemetry import read_recent

DEFAULT_PATH = "out/telemetry.jsonl"
FOLLOW_INTERVAL = 1.0
DEFAULT_LIMIT = 20
TAIL_CHUNK = 8192


def summary(path, limit=DEFAULT_LIMIT):
    """Print the most recent events and return how many were shown."""
    events = read_recent(path, limit=limit)
    if not events:
        print(f"[no telemetry yet at {path}]")
        return 0
    for event in events:
        print(json.dumps(event, ensure_ascii=False))
    return len(events)


def read_appended(path, offset):
    """Read whatever has been appended since `offset`.

    Returns the new text and the new offset. A missing file is not an error:
    the follower may start before the first cycle has written anything.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            chunk = handle.read()
            return chunk, handle.tell()
    except OSError:
        return "", offset


def follow(path, interval=FOLLOW_INTERVAL, stream=None):
    """Print new telemetry events until interrupted.

    `stream` is injectable so the loop can be exercised without waiting on
    real time passing.
    """
    out = print if stream is None else stream
    out(f"Watching {path}. Ctrl+C to stop.")
    offset = 0
    while True:
        if path.exists():
            chunk, offset = read_appended(path, offset)
            for raw in chunk.splitlines():
                try:
                    out(json.dumps(json.loads(raw), ensure_ascii=False))
                except json.JSONDecodeError:
                    continue
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Watch or summarise Jupiter's learning telemetry.",
    )
    parser.add_argument(
        "--path", default=DEFAULT_PATH, help=f"telemetry file (default {DEFAULT_PATH})"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print the most recent events and exit instead of following",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="how many events to show"
    )
    args = parser.parse_args()
    path = Path(args.path)
    if args.summary:
        summary(path, args.limit)
        return
    try:
        follow(path)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
