"""Small append-only telemetry and learning-cycle records."""

import json
import os
import time
from pathlib import Path

TAIL_CHUNK = 8192


class Telemetry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event, **values):
        record = {"time": time.time(), "event": event, **values}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def read_recent(path, limit=20):
    """Return the last `limit` records, reading only the tail of the file.

    The telemetry stream is append-only and grows without bound, so reading it
    end to end to show the last few events becomes the slowest thing in the
    project once a run is long. This walks backwards in chunks until it has
    enough newlines, which is a fixed cost no matter how large the file grows.

    Unreadable and half-written lines are skipped rather than raising: the
    writer appends without a lock, so a partially flushed final line is a
    normal thing to observe.
    """
    p = Path(path)
    if not p.exists():
        return []
    with p.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        data = b""
        while size > 0 and data.count(b"\n") <= limit:
            step = min(TAIL_CHUNK, size)
            size -= step
            handle.seek(size)
            data = handle.read(step) + data
    result = []
    for line in data.decode("utf-8", errors="replace").splitlines()[-limit:]:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def write_status(
    path,
    status,
    cycle,
    source,
    steps,
    best_holdout_loss,
    current_holdout_loss,
):
    """Write the small, human-readable learning status snapshot.

    This is a convenience view for a person or another tool that only needs to
    know whether Jupiter is learning and how well. It is deliberately a
    separate file from learning_state.json, which holds the full per-cycle
    record that train.py and learning_worker.py read back to resume numbering.

    Holdout loss is measured on the tail of the document being learned, so
    these numbers describe `source` and are NOT comparable across different
    documents. The worker only ever compares holdout before and after the same
    cycle, which is the comparison that stays valid.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "status": status,
                "cycle": cycle,
                "source": source,
                "steps": steps,
                "best_holdout_loss": best_holdout_loss,
                "current_holdout_loss": current_holdout_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
