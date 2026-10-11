"""Single-writer lock for the learning worker (Architecture 38).

Two workers running at once would both write the same SafeTensors checkpoint
and the same learning state, which is a destructive operation nobody asked
for. This is the minimum guard against that.

The lock is a file created exclusively. It records the owning process id so a
lock left behind by a process that no longer exists can be reported instead of
being silently stolen.
"""

import json
import os
import time
from pathlib import Path


def process_alive(pid):
    """Best-effort check for whether a process id is still running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but belongs to another user.
        return True
    return True


def _alive_windows(pid):
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def read_lock(path):
    """Return the lock record, or None when the lock is free."""
    lock_path = Path(path)
    if not lock_path.exists():
        return None
    try:
        record = json.loads(lock_path.read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else None
    except (OSError, ValueError):
        return None


class LockHeld(RuntimeError):
    """Raised when another live worker already owns the lock."""


def acquire(path, c):
    """Take the lock, or raise LockHeld naming the current owner.

    A lock left behind by a process that no longer exists is stale, and it is
    removed and taken. The owner is provably gone, so leaving the file in place
    would block every future run until a person deleted it by hand, which is
    exactly the outcome a lock is supposed to prevent. The removal is reported
    rather than silent, because Architecture 38 asks for a governing decision
    to be visible instead of quiet.

    Only a live owner is ever refused, and the O_EXCL create below still
    decides a genuine race between two workers starting at once.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = read_lock(lock_path)
    if record is not None:
        owner = int(record.get("pid", -1))
        if process_alive(owner):
            raise LockHeld(
                f"another learning worker already owns {lock_path} "
                f"(pid {owner}, started {record.get('started')}). "
                f"Stop it first, or remove {lock_path} if you are certain it is dead."
            )
        reason = f"pid {owner} is no longer running"
    else:
        reason = "the lock could not be read"
    if lock_path.exists():
        lock_path.unlink(missing_ok=True)
        print(f"[lock] removed a stale lock at {lock_path}: {reason}")
    payload = json.dumps(
        {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=2,
    )
    try:
        # Exclusive creation is the atomic part: exactly one worker wins.
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = read_lock(lock_path)
        raise LockHeld(
            f"lost the race for {lock_path}; another worker took it "
            f"(pid {holder.get('pid') if holder else 'unknown'})"
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return lock_path


def release(path):
    """Remove the lock if this process owns it."""
    lock_path = Path(path)
    record = read_lock(lock_path)
    if record is not None and int(record.get("pid", -1)) != os.getpid():
        # Owned by someone else. Leave it alone.
        return False
    lock_path.unlink(missing_ok=True)
    return True
