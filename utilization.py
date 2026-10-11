"""Resource and learning observability (Architecture 35).

The goal is to avoid the experience of "something has been training for six
hours and I have no idea what it is doing". This samples the machine the
worker is running on, using only the standard library and torch, so no
dependency is added (Architecture 22).
"""

import os
import shutil
import time
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard requirement elsewhere
    torch = None


def cpu_snapshot():
    """Return CPU seconds used by this process, since it started."""
    times = os.times()
    return {
        "user_seconds": round(times.user, 2),
        "system_seconds": round(times.system, 2),
    }


def ram_snapshot():
    """Return resident memory in megabytes, or None where unavailable."""
    try:
        import resource
    except ImportError:
        return _ram_windows()
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    divisor = 1024.0 if os.uname().sysname == "Linux" else 1024.0 * 1024.0
    return {"max_rss_mb": round(usage / divisor, 1)}


def _ram_windows():
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {
        "memory_load_percent": int(status.memory_load),
        "available_physical_mb": round(status.available_physical / 1048576.0, 1),
        "total_physical_mb": round(status.total_physical / 1048576.0, 1),
    }


def gpu_snapshot():
    """Return GPU memory facts, or None when running on CPU."""
    if torch is None or not torch.cuda.is_available():
        return None
    return {
        "device": torch.cuda.get_device_name(0),
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576.0, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576.0, 1),
    }


def storage_snapshot(path: str | Path = "."):
    """Disk usage for the directory Jupiter writes to.

    Architecture 35 lists storage usage among the things the system should
    expose. A learning run appends to telemetry and rewrites a checkpoint on
    every accept, so the disk fills at a rate nobody would notice from CPU and
    GPU numbers alone. Returns None where the filesystem cannot be measured.

    Takes a str or a Path because every caller already has a Path in hand:
    c.out_dir is a configuration value, and converting it back to a string to
    satisfy an annotation would be the wrong direction.
    """
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    gigabyte = 1024.0**3
    return {
        "total_gb": round(usage.total / gigabyte, 2),
        "used_gb": round(usage.used / gigabyte, 2),
        "free_gb": round(usage.free / gigabyte, 2),
    }


def sample(path: str | Path = "."):
    """One combined observability sample.

    Takes a str or a Path for the same reason storage_snapshot does: the
    callers hold a Path already.
    """
    return {
        "time": time.time(),
        "cpu": cpu_snapshot(),
        "ram": ram_snapshot(),
        "gpu": gpu_snapshot(),
        "storage": storage_snapshot(path),
    }


def describe(sample_data):
    """Render a sample as a single readable line for logs."""
    parts = []
    gpu = sample_data.get("gpu")
    if gpu:
        parts.append(
            f"gpu={gpu['allocated_mb']}MB alloc/{gpu['reserved_mb']}MB reserved"
        )
    ram = sample_data.get("ram")
    if ram and "max_rss_mb" in ram:
        parts.append(f"ram={ram['max_rss_mb']}MB")
    elif ram and "memory_load_percent" in ram:
        parts.append(f"ram={ram['memory_load_percent']}%")
    storage = sample_data.get("storage")
    if storage:
        parts.append(f"disk={storage['free_gb']}GB free")
    cpu = sample_data.get("cpu")
    if cpu:
        parts.append(f"cpu={cpu['user_seconds']}s user/{cpu['system_seconds']}s sys")
    return " | ".join(parts) if parts else "no resource data"
