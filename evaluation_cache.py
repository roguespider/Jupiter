"""Classify learning sources once and remember the answer.

Which recorded sources are code-bearing and which argue is a property of the
bytes in those files, not of the run that is asking. The answer was being
recomputed every time the evaluation report was built, and measured on the live
store that meant reading 888MB of documents to classify 126 of them, taking 89
seconds, twice per report because the programming and reasoning sections both
ask.

The answer is kept beside the other derived state, keyed by each file's size and
modification time rather than by its content, for the same reason the curation
cache uses those: comparing content means reading the file, and reading the file
is the whole cost this cache exists to avoid. A file edited without its timestamp
moving is a case the filesystem cannot detect, and it is the same trade every
build cache makes.

Deleting the cache is always safe. It is derived, it is rebuilt from files that
are still there, and a stale entry is detected rather than believed: the size and
timestamp are checked on every read, so a document that has changed is
reclassified instead of answered from a record of what it used to be.
"""

import hashlib
import json
import os
import time
from pathlib import Path

# Bumped when the classification itself changes, so an old record is not read as
# though it had been produced by the current rules. Deliberately its own counter
# rather than the architecture version: most architecture changes do not alter
# what a document is classified as, and invalidating on all of them would discard
# a cache for edits that changed nothing.
CACHE_VERSION = 1


def cache_path(out_dir):
    return Path(out_dir) / "source_classification.json"


def load(out_dir):
    """Every remembered classification, or an empty map if there is nothing usable."""
    path = cache_path(out_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        return {}
    entries = data.get("sources")
    return entries if isinstance(entries, dict) else {}


def save(out_dir, entries):
    """Write the remembered classifications, whole and atomically.

    A half-written cache would claim a document was classified when it was not,
    and the next run would answer from that claim. The payload goes to a
    temporary file and is renamed into place, so an interrupted write leaves the
    previous cache rather than a truncated one.
    """
    path = cache_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"version": CACHE_VERSION, "written": time.time(), "sources": entries},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def signature(path):
    """Size and modification time, or None if the file cannot be measured."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {"size": int(stat.st_size), "mtime": float(stat.st_mtime)}


def remembered(out_dir, path):
    """The stored classification for a path, or None if there is none to trust."""
    record = load(out_dir).get(str(path))
    if not isinstance(record, dict):
        return None
    current = signature(path)
    if current is None:
        return None
    if record.get("size") != current["size"] or record.get("mtime") != current["mtime"]:
        return None
    verdict = record.get("verdict")
    return verdict if isinstance(verdict, dict) else None


def remember(out_dir, path, verdict):
    """Record a classification against the file's current size and timestamp."""
    entries = load(out_dir)
    current = signature(path)
    if current is None:
        return entries
    entries[str(path)] = {**current, "verdict": verdict}
    return entries


def census(out_dir):
    """What the cache holds, so it can be inspected rather than trusted."""
    entries = load(out_dir)
    return {
        "path": str(cache_path(out_dir)),
        "version": CACHE_VERSION,
        "documents": len(entries),
        "bytes": sum(int(e.get("size", 0) or 0) for e in entries.values()),
        "note": (
            "size and timestamp keyed, derived, and safe to delete. A document "
            "whose size or timestamp no longer matches is reclassified rather "
            "than answered from this file."
        ),
    }


def digest_of(text):
    """A stable id for a document's text, used when reporting what was classified.

    Not a cache key. It exists so two documents that curate to the same content
    can be reported as one piece of material rather than as two, which is what
    the reader of the report is actually asking.
    """
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
