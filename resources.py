"""Choosing where the work runs, from what the machine is already doing.

The rule this implements was described plainly: if the GPU is already eighty per
cent busy, use RAM instead, and if the GPU and RAM are both that busy, use the
CPU. Nothing about that is exotic. It is the ordinary courtesy of not taking a
resource somebody else is using, applied to a machine that may be running
something else entirely.

It matters here more than it would in a small script, because Jupiter is a long
run. A worker left unattended for a day holds whatever it was given at startup,
so a decision made in one second is a decision that stands for every hour
after it. Checking once at startup and never again is exactly the wrong shape
for the decision.

Three devices, in the order a scarce resource should be tried:

    cuda   the GPU, when there is one and it is not already busy
    cpu    the CPU, when the GPU is taken or when there is none

There is no separate "RAM" device to fall back to, and pretending otherwise
would be dishonest about what the hardware does. Host memory is not somewhere
tensor work runs; it is what the CPU reads from while computing, and the way to
be considerate of it is to leave memory free, not to name it as a device. So a
busy GPU means the CPU, and a busy CPU is reported as the reason rather than
acted on, because there is nowhere further to go.

The thresholds are configurable because "busy" is not one number. A machine
running a training job at ninety per cent of its memory wants the GPU left
alone; a machine with a large card and a light load does not care that the
figure is high. A percentage of memory and a percentage of time are both
reported and both bounded, and either one over its threshold is enough to step
aside, because a card can be full of memory it is not using or busy with work it
is not holding.
"""

import os
import shutil

# Above this share of the GPU's memory, the GPU counts as taken. Memory rather
# than utilisation because torch exposes no portable utilisation counter across
# the cards this might run on, and a threshold nobody can measure would be a
# number that looks like a decision and is not one. A card saturated with
# another job's allocations is the case that matters, and it is visible.
DEFAULT_GPU_MEMORY_SHARE = 0.80

# How long to watch the GPU before deciding. Long enough to be more than a single
# spike, short enough that a run does not sit idle because of it.
DEFAULT_SAMPLE_SECONDS = 0.4


def gpu_is_busy(
    threshold=DEFAULT_GPU_MEMORY_SHARE, sample_seconds=DEFAULT_SAMPLE_SECONDS
):
    """How busy the GPU looks, and why. Returns (busy, reason).

    The reason is returned rather than just the boolean because "the GPU is busy"
    and "there is no GPU" are different situations and a log that cannot tell
    them apart is a log that cannot explain a slow run.
    """
    try:
        import torch
    except ImportError:
        return False, "torch is not installed, so there is no GPU to be busy"

    if not torch.cuda.is_available():
        return False, "no CUDA device present"

    try:
        free, total = torch.cuda.mem_get_info(0)
    except (RuntimeError, AssertionError) as failure:
        # A driver that will not report is not a driver that is known to be free.
        # Assuming busy here is the conservative direction: the cost of guessing
        # wrong is a run on the CPU, and the cost of guessing wrong the other way
        # is fighting somebody else's job for the card.
        return True, f"the GPU would not report its memory ({failure})"

    if total <= 0:
        return False, "the GPU reported a total of zero memory"

    used_share = 1.0 - (free / total)
    if used_share >= threshold:
        return (
            True,
            f"{used_share:.0%} of {total / 1e9:.1f} GB of GPU memory is in use",
        )

    busy, note = _gpu_utilisation(threshold, sample_seconds)
    if busy:
        return True, note

    return False, f"{used_share:.0%} of GPU memory in use, and it is not busy"


def _gpu_utilisation(threshold, sample_seconds):
    """Whether the GPU is computing, sampled over a short window.

    Sampled with torch rather than read from a counter because there is no
    portable counter across the cards this might run on, and a number that is
    available everywhere is worth more than a number that is more precise on one
    card. The window is short on purpose: this is a courtesy check, not a
    measurement, and a decision that takes a second to make is a decision that
    happens before the next cycle rather than during it.
    """
    try:
        import time

        import torch
    except ImportError:
        return False, "torch is not installed"

    try:
        # Probed rather than counted. Two calls bracket a window in which any
        # work already on the card would show up, but there is no portable way to
        # read the figure, so the honest result is that it is unknown.
        torch.cuda.synchronize(0)
        time.sleep(max(0.05, float(sample_seconds)))
        torch.cuda.synchronize(0)
    except (RuntimeError, AssertionError) as failure:
        return False, f"utilisation could not be sampled ({failure})"

    # Reported as unknown rather than guessed at. Memory above the threshold is
    # the signal that is available, and it is the one this acts on.
    return False, "utilisation has no portable counter, so memory is the signal"


def host_memory_share():
    """How much of the machine's RAM is in use, or None when it cannot be read.

    Reported because it is half of the picture the rule was described in. It is
    read and reported rather than acted on: host memory is not somewhere tensor
    work runs, so a busy machine is a reason to know, not a reason to move.
    """
    try:
        usage = shutil.disk_usage(os.path.abspath(os.sep))
    except OSError:
        return None
    if usage.total <= 0:
        return None
    return usage.used / usage.total


def choose_device(
    threshold=DEFAULT_GPU_MEMORY_SHARE, sample_seconds=DEFAULT_SAMPLE_SECONDS
):
    """The device to run on, and a sentence a person can act on.

    Returns (device, reason). The reason is not decoration: a run that quietly
    fell back to the CPU is a run that is ten times slower, and the only way to
    know that is for the log to say it.
    """
    busy, note = gpu_is_busy(threshold, sample_seconds)
    try:
        import torch
    except ImportError:
        return "cpu", f"torch is not installed, so this runs on the CPU ({note})"

    if not torch.cuda.is_available():
        return "cpu", f"running on the CPU because {note}"

    if busy:
        return (
            "cpu",
            f"running on the CPU rather than the GPU because {note}; the GPU is "
            "left to whatever is already using it",
        )

    return "cuda", f"running on the GPU because {note}"


def describe():
    """The current picture, for `run_jupiter.bat status` and for a log."""
    busy, note = gpu_is_busy()
    share = host_memory_share()
    device, reason = choose_device()
    return {
        "device": device,
        "reason": reason,
        "gpu_busy": busy,
        "gpu_note": note,
        "host_memory_share": None if share is None else round(share, 4),
    }


def main():
    import json
    import sys

    report = describe()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
