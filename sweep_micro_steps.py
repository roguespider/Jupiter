"""Sweep micro_steps and find the setting that learns most per unit of time.

micro_steps was set to 8 because 512 was being rejected 79% of the time: a
512-step push damaged the reserved document so often that the gain was restored
every cycle and nothing accumulated. Eight steps is accepted, but it was chosen
to stop the bleeding rather than measured for what it costs. It may be far too
timid, and a timid setting that is accepted is still not the right one.

Every setting in the sweep starts from the same checkpoint and reads the same
corpus, so the only variable is the number of steps. That is the whole design:
a sweep that lets each setting end where the previous one left it measures the
order the settings were run in rather than the settings themselves.

What is measured per setting:

    accepted        how often a cycle survived the acceptance gate
    gain_per_cycle  mean holdout improvement per cycle run, accepted or not,
                    which is the number that matters because a rejected cycle
                    costs its full wall clock and returns nothing
    gain_total      the sum of those, which is what a run actually buys
    regression      mean damage to the reserved document, because a setting that
                    learns fast by wrecking everything it knew is not a setting
                    that learns

wall_seconds is reported so a faster setting is not mistaken for a worse one for
having had less time to work.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SWEEP = PROJECT / "temp" / "sweep"
OUT = PROJECT / "out"
BASE_CHECKPOINT = SWEEP / "base.safetensors"
BASE_ARCHITECTURE = SWEEP / "base.json"

# 8 is what is configured now; the rest are guesses at the middle.
SETTINGS = (8, 16, 32, 64, 128)

CYCLES_PER_SETTING = 12


def restore_baseline():
    """Put the model back to the state the sweep starts from.

    Restoring the checkpoint AND the architecture record matters: leaving the
    record describing the previous setting's model would make the next run
    build the wrong shape and the comparison meaningless.
    """
    shutil.copy2(BASE_CHECKPOINT, OUT / "Jupiter_latest.safetensors")
    shutil.copy2(BASE_ARCHITECTURE, OUT / "Jupiter.architecture.json")


def read_cycle_number():
    """The cycle counter the next run will start from, read before restoring."""
    path = OUT / "learning_state.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("learning_cycle", 0))
    except (OSError, ValueError, TypeError):
        return 0


def run_setting(steps):
    """Run one setting from the baseline and measure what it bought."""
    start_cycle = read_cycle_number()
    restore_baseline()
    env = dict(os.environ)
    env["JUPITER_MICRO_STEPS"] = str(steps)
    env["JUPITER_WORKER_MAX_CYCLES"] = str(CYCLES_PER_SETTING)
    # The patience rule stops a run after two unaccepted cycles, which for a
    # sweep measures patience rather than the setting. It is raised well past
    # the cycle budget so the budget is what ends the run.
    env["JUPITER_WORKER_PATIENCE"] = "50"
    env["JUPITER_WORKER_IDLE_SECONDS"] = "2"
    env["JUPITER_WORKER_MAX_SECONDS"] = "600"
    env["JUPITER_NO_PROMPT"] = "1"

    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", "learning_worker.py"],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    wall = time.time() - started

    # Isolated by the cycle counter this run starts from, not by taking the
    # last N rows. The worker resumes its counter from disk, so every cycle in
    # a setting has a number above the one read before it started and nothing
    # from an earlier setting does. A tail slice looked simpler and was wrong:
    # it caught twice the intended rows, so the first sweep's numbers mixed two
    # settings together.
    with (OUT / "experiments.jsonl").open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    fresh = [
        r
        for r in records
        if r.get("after_metrics", {}).get("holdout_loss") is not None
        and r.get("learning_cycle", 0) > start_cycle
    ]

    accepted = [r for r in fresh if r["decision"] == "ACCEPT"]
    gains = [
        r["before_metrics"]["holdout_loss"] - r["after_metrics"]["holdout_loss"]
        for r in fresh
    ]
    regressions = [
        r["before_metrics"]["regression_loss"] - r["after_metrics"]["regression_loss"]
        for r in fresh
        if r.get("after_metrics", {}).get("regression_loss") is not None
    ]

    return {
        "micro_steps": steps,
        "cycles": len(fresh),
        "accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / len(fresh), 4) if fresh else 0.0,
        "gain_per_cycle": round(statistics.mean(gains), 4) if gains else 0.0,
        "gain_total": round(sum(gains), 4) if gains else 0.0,
        "regression_mean": round(statistics.mean(regressions), 4)
        if regressions
        else None,
        "wall_seconds": round(wall, 1),
        "gain_per_minute": round(sum(gains) / (wall / 60.0), 4)
        if gains and wall
        else 0.0,
        "exit_code": proc.returncode,
    }


def main():
    results = []
    for steps in SETTINGS:
        print(f"--- micro_steps={steps} ---", flush=True)
        result = run_setting(steps)
        results.append(result)
        print(
            f"    accepted {result['accepted']}/{result['cycles']} "
            f"({result['acceptance_rate'] * 100:.0f}%)  "
            f"gain/cycle {result['gain_per_cycle']:+.4f}  "
            f"total {result['gain_total']:+.3f}  "
            f"{result['wall_seconds']:.0f}s  "
            f"gain/min {result['gain_per_minute']:+.3f}",
            flush=True,
        )
    (SWEEP / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(
        f"{'steps':>6} {'accept':>8} {'gain/cyc':>10} {'total':>9} {'sec':>7} {'gain/min':>10}"
    )
    for row in results:
        print(
            f"{row['micro_steps']:>6} "
            f"{row['acceptance_rate'] * 100:>7.0f}% "
            f"{row['gain_per_cycle']:>+10.4f} "
            f"{row['gain_total']:>+9.3f} "
            f"{row['wall_seconds']:>7.0f} "
            f"{row['gain_per_minute']:>+10.3f}"
        )


if __name__ == "__main__":
    main()
