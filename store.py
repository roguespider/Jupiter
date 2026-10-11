"""Document store for concepts, relationships, memories and experiences.

Architecture 15 lists four memory classes and calls SQLite "a strong
candidate", but also states that the exact storage technology is
implementation-dependent. This store is a small JSON document store so that
nothing new is introduced as a dependency. Everything above it talks to this
interface, which is the seam where a SQL backend would go later: the only
operations callers use are put, get, all, update and delete over named
collections.

Two properties matter more than speed here. Writes are atomic, so an
interrupted process cannot leave a half-written store. And a corrupt store is
reported as corrupt rather than silently treated as empty, because silently
discarding a knowledge base would make Jupiter forget without ever recording
that it forgot.
"""

import json
import os
import tempfile
import time
from pathlib import Path

# A read that collides with a concurrent writer is retried rather than
# reported. The write path is atomic, so the collision window is one os.replace
# wide, and a handful of short attempts covers it without anyone having to
# think about it.
READ_ATTEMPTS = 5
READ_RETRY_SECONDS = 0.05

# A write that collides with a concurrent reader is retried for the same reason
# a read that collides with a writer is. On Windows a sharing violation is a
# WinError 5 from os.replace, and it arrives for the same reason the read path
# saw one: something else had the file open at the instant it was replaced.
#
# The read retry alone was not enough, because the write path is where the
# collision is actually lost. The learning cycle derives relationships in a
# loop, and every put replaced the whole store, so one cycle performed on the
# order of a thousand os.replace calls against the same path. Each one is a
# chance to lose a race with anything else reading the directory, and losing
# it raised PermissionError out of the learning worker and stopped the entry
# point. The store was never damaged by it: the temporary file is removed and
# the previous document is left in place, so this was an avoidable crash rather
# than data loss.
WRITE_ATTEMPTS = 5
WRITE_RETRY_SECONDS = 0.05


class StoreCorrupt(RuntimeError):
    """Raised when a store file exists but cannot be parsed.

    This is deliberately not caught internally. Silently starting from an
    empty store would let Jupiter lose its knowledge without recording the
    loss, which is the exact failure Architecture 32 warns about.
    """


class _Transaction:
    """The context manager Store.transaction returns.

    Holds no state of its own beyond the entry depth: everything staged lives on
    the store, so a nested block joining the outer one is the same collection
    and therefore the same single flush.
    """

    def __init__(self, store):
        self.store = store
        self.outermost = False

    def __enter__(self):
        self.outermost = self.store._depth == 0
        self.store._depth += 1
        return self.store

    def __exit__(self, exc_type, exc_value, traceback):
        self.store._depth -= 1
        # Only the outermost block flushes, and only on success. An exception
        # inside the block means the caller did not finish, and writing the
        # half of its work that happened to be staged would be worse than
        # writing none of it.
        if self.outermost and exc_type is None:
            self.store._flush()
        return False


class StoreBusy(RuntimeError):
    """Raised when a store file is held open by another process.

    Separate from StoreCorrupt because it means the opposite. The file is
    intact and will be readable as soon as the other process lets go, and
    telling someone their knowledge is corrupt when it is merely busy sends
    them looking for damage that is not there.
    """


class Store:
    """Named collections of records, persisted as one JSON document."""

    def __init__(self, path):
        self.path = Path(path)
        # The parsed document, kept only while the file on disk still matches
        # the stamp it was parsed from. Not a cache for its own sake: the
        # learning cycle re-reads an unchanged store constantly, and every read
        # was a full parse of the whole document.
        self._cache = None
        self._stamp_cached = None
        # Writes staged by an open transaction, and how deeply they nest.
        self._pending = {}
        self._depth = 0

    def _read_text(self):
        """Read the store, distinguishing "busy" from "broken".

        A read can fail because another process holds the file, and on Windows
        that is a sharing violation rather than damage. The write path is atomic,
        so the window is only as wide as one os.replace, but it is real: a
        learning worker running alongside anything that reads the store hit it
        often enough to be worth naming.

        That case was reported as StoreCorrupt, which is wrong twice over. It
        says the file is damaged when it is not, and it is raised out of a
        constructor with no recovery, so one concurrent writer took the entry
        point down with a message that sent whoever read it looking for a
        corrupt file instead of a running worker.

        A busy store is therefore retried briefly, and only reported as what it
        is if it stays busy. A genuine I/O failure still raises immediately:
        silently retrying a real fault would hide it.
        """
        last = None
        for attempt in range(READ_ATTEMPTS):
            try:
                return self.path.read_text(encoding="utf-8")
            except PermissionError as failure:
                last = failure
                if attempt + 1 < READ_ATTEMPTS:
                    time.sleep(READ_RETRY_SECONDS)
        raise StoreBusy(
            f"store {self.path} is held by another process and could not be "
            f"read after {READ_ATTEMPTS} attempts: {last}. If a learning worker "
            "is running, stop it first; the store is not damaged."
        )

    def _stamp(self):
        """The file's identity on disk, cheaply.

        Modification time and size together are enough to notice another
        process rewriting the store, and asking for them costs a fraction of
        parsing the document. That fraction is the whole point: the learning
        cycle reads the same store a hundred and sixty times per cycle, and
        re-parsing an unchanged file each time was most of its wall clock.
        """
        try:
            info = self.path.stat()
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size)

    def _read(self):
        if not self.path.exists():
            self._cache = None
            self._stamp_cached = None
            return {}
        stamp = self._stamp()
        if self._cache is not None and stamp == self._stamp_cached:
            return self._cache
        try:
            text = self._read_text()
        except StoreBusy:
            raise
        except OSError as failure:
            raise StoreCorrupt(f"cannot read store {self.path}: {failure}")
        if not text.strip():
            self._cache = {}
            self._stamp_cached = stamp
            return self._cache
        try:
            data = json.loads(text)
        except ValueError as failure:
            raise StoreCorrupt(
                f"store {self.path} is not valid JSON: {failure}. "
                f"Move it aside to start fresh; it was not deleted."
            )
        if not isinstance(data, dict):
            raise StoreCorrupt(
                f"store {self.path} must contain a JSON object, "
                f"found {type(data).__name__}"
            )
        self._cache = data
        self._stamp_cached = self._stamp()
        return data

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file in the same directory, then replace, so a
        # crash mid-write cannot truncate the existing store.
        handle, temp_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=self.path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            self._replace(temp_name)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        # The cache now describes exactly what is on disk. Leaving it stale
        # would make the next read return the previous document.
        self._cache = data
        self._stamp_cached = self._stamp()

    def _replace(self, temp_name):
        """Move the finished temporary file onto the store, retrying a collision.

        os.replace is atomic on both platforms, so the only thing that can go
        wrong here is that the destination is momentarily held open by another
        process, which Windows reports as PermissionError and which means the
        document is fine and the timing was not.

        Retrying is the correct response rather than propagating it, because the
        alternative is that a reader with no intention of writing can stop a
        learning cycle. The retry is bounded and then reported as what it is,
        so a genuinely unwritable store still surfaces instead of hanging.
        """
        last = None
        for attempt in range(WRITE_ATTEMPTS):
            try:
                os.replace(temp_name, self.path)
                return
            except PermissionError as failure:
                last = failure
                if attempt + 1 < WRITE_ATTEMPTS:
                    time.sleep(WRITE_RETRY_SECONDS)
        raise StoreBusy(
            f"store {self.path} is held by another process and could not be "
            f"written after {WRITE_ATTEMPTS} attempts: {last}. If another "
            "Jupiter process is running, stop it first; the store is not "
            "damaged and the previous contents are still in place."
        )

    def collection(self, name):
        """Return a collection, creating it when absent."""
        data = self._read()
        records = data.get(name)
        if records is None:
            records = {}
            data[name] = records
            self._write(data)
        if not isinstance(records, dict):
            raise StoreCorrupt(
                f"collection {name!r} in {self.path} must be an object, "
                f"found {type(records).__name__}"
            )
        return records

    def put(self, name, key, record):
        """Insert or replace a record and return it."""
        with self.transaction():
            self._stage(name, [(key, record)])
        return record

    def transaction(self):
        """Collect many writes and flush them as one.

        This is what a caller that makes thousands of writes in a row needs, and
        put_many is not it: put_many still writes once per call, and a deriver
        that calls it in a loop pays the full rewrite cost on every iteration.
        Measured on this corpus, deriving relationships wrote the store 582 times
        inside a three-cycle run and spent 414 of the 420 seconds doing it,
        against 0.1 seconds of actual training.

        Inside the block, puts are staged in memory and nothing touches the disk.
        The whole block is flushed once when it closes, through the same atomic
        replace every other write uses, so a process killed mid-derivation
        leaves the previous document intact rather than half of one.

        Nested blocks are supported and join the outermost one, so a helper that
        opens a transaction still writes once when the caller opened one too.
        Nothing here changes what is written or how: the same records land under
        the same keys with the same atomicity. Only the number of writes changes.
        """
        return _Transaction(self)

    @property
    def in_transaction(self):
        """Whether a write staged right now would be deferred rather than written.

        Public so a caller batching in its own way can decide whether to stage
        or write, instead of reaching into the private depth counter to find out.
        """
        return self._depth > 0

    def _stage(self, name, items):
        """Record a write for the open transaction, or write it now.

        Reads still go through to disk. A transaction defers writes and nothing
        else, so a caller reading what it just wrote sees the new value because
        the staged data is merged into the cache immediately.
        """
        merged = {}
        for key, record in items:
            merged[str(key)] = record
        if not merged:
            return []
        if self._depth:
            self._pending.setdefault(name, {}).update(merged)
            # Merged into the cache now so a read inside the transaction sees
            # what it just wrote. Without this a caller that wrote and then read
            # the same key back would get the previous document's answer.
            data = self._read()
            records = data.setdefault(name, {})
            if isinstance(records, dict):
                records.update(merged)
                self._cache = data
            return list(merged.values())
        data = self._read()
        records = data.setdefault(name, {})
        if not isinstance(records, dict):
            raise StoreCorrupt(f"collection {name!r} in {self.path} must be an object")
        records.update(merged)
        self._write(data)
        return list(merged.values())

    def _flush(self):
        """Write everything staged, in one pass."""
        if not self._pending:
            return
        pending = self._pending
        self._pending = {}
        data = self._read()
        for name, records in pending.items():
            existing = data.setdefault(name, {})
            if not isinstance(existing, dict):
                raise StoreCorrupt(
                    f"collection {name!r} in {self.path} must be an object"
                )
            existing.update(records)
        self._write(data)

    def put_many(self, name, items):
        """Insert or replace many records in one write, and return them.

        put() rewrites the entire store, which is the right trade for one record
        and a compounding one for a batch. Relationship derivation writes an edge
        at a time, and a store holding twelve hundred of them turned a single
        learning cycle into roughly twelve hundred full rewrites of a multi-megabyte
        document: minutes of disk traffic to persist a few hundred kilobytes of
        new edges, and twelve hundred chances to lose a race with any other
        process reading the directory.

        Batching does not change what is persisted or how it is persisted. The
        same records land under the same keys with the same atomic replace, so
        an interrupted batch still leaves the previous document intact rather
        than a partial one. Only the number of writes changes.

        Within one call, the items are merged by key so two writes to the same
        key cost one write rather than two. That matters because the relationship
        deriver touches the same concept from both of its endpoints, and a
        caller that is already batching would otherwise still pay for the
        duplicates it created itself.
        """
        return self._stage(name, items)

    def remove(self, name, keys):
        """Delete records by key, and return how many were actually there.

        put_many() merges, so it can add and replace but never remove: writing the
        survivors of a collection back through it left everything in place. That
        is why a cleanup that rewrote a collection without its unwanted records
        reported removing them and removed none.

        Removal is written whole and atomically like every other write, so an
        interrupted delete leaves the previous document rather than a partial one.
        A key that was not there is not an error and is not counted, because the
        caller asked for records to be gone and they are.
        """
        wanted = {str(key) for key in keys}
        if not wanted:
            return 0
        data = self._read()
        records = data.get(name)
        if not isinstance(records, dict):
            raise StoreCorrupt(f"collection {name!r} in {self.path} must be an object")
        removed = 0
        for key in wanted:
            if records.pop(key, None) is not None:
                removed += 1
        self._write(data)
        return removed

    def get(self, name, key):
        """Return one record, or None when it is absent."""
        return self.collection(name).get(str(key))

    def all(self, name):
        """Return every record in a collection, ordered by key."""
        return [self.collection(name)[key] for key in sorted(self.collection(name))]

    def clear(self, name):
        """Remove every record in a collection. Returns how many were removed.

        Used for task-scoped memory (Architecture 15), which is discarded when
        a task ends rather than retained.

        Inside a transaction the clear is staged rather than written, because a
        staged record and a written clear would fight: the clear would go to
        disk and the flush would put the records straight back. Working memory is
        held for the length of a cycle and cleared at the end of it, which is
        inside the same transaction that recorded the cycle.
        """
        data = self._read()
        records = data.setdefault(name, {})
        removed = len(records)
        data[name] = {}
        if self._depth:
            # Deferred, and the collection is emptied in the cache so a read
            # inside the transaction sees it as cleared.
            self._pending[name] = {}
            self._cache = data
            return removed
        self._write(data)
        return removed

    def counts(self):
        """Return the number of records in each collection."""
        data = self._read()
        return {
            name: len(records)
            for name, records in sorted(data.items())
            if isinstance(records, dict)
        }
