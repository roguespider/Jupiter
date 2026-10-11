"""Run the test suite across several processes and report one verdict.

The suite is a set of independent files that share nothing but the modules they
import, so the work divides cleanly: thirty-nine files over thirty-two cores is
one file each and nothing contended. That is why this is faster rather than
merely different - there is no shared fixture to serialise on.

Splitting by module rather than by test is deliberate. Several of these files
build a tokenizer, a corpus or a model in setUp, and an individual test is often
cheaper than that setup. Running one file per process keeps each setup paid for
exactly once and gives a failure the whole file it belongs to, which is what a
person needs in order to act on it.

The verdict is the same one `python -m unittest` would give: every test ran, and
the suite passes only if all of them passed. A file that fails to import is a
failure here too, which matters because a module that will not import is the
most expensive way for a suite to lie.

Nothing is written that `unittest` would not write. The per-process results are
reported as they arrive, and the exit code is the same.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

TESTS = Path(__file__).resolve().parent / "tests"
PROJECT = Path(__file__).resolve().parent


def _modules():
    return sorted(p.stem for p in TESTS.glob("test_*.py"))


def _run_one(stem):
    """Run one test module in this process. Returns (name, ok, seconds, tail).

    The thread limit is set here rather than left to torch. Torch defaults to one
    thread per core, so on a thirty-two core machine each of these child
    processes claimed twenty-four of them. Run one at a time that is harmless; run
    thirty-two at a time, which is what the pool does, thirty-two processes
    oversubscribe every core and each long test slowed from forty-six seconds to
    over two hundred.
    """
    started = time.time()
    environment = dict(os.environ)
    per_worker = max(1, (os.cpu_count() or 2) // max(1, workers_in_use()))
    environment["OMP_NUM_THREADS"] = str(per_worker)
    environment["MKL_NUM_THREADS"] = str(per_worker)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", f"tests.{stem}"],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=FILE_CEILING,
            env=environment,
        )
        code = result.returncode
        output = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired as expired:
        # A file that runs past its ceiling is reported as hung. Waiting forever
        # is the one behaviour a verification step must not have: a hung suite and
        # a passing suite look the same to somebody watching a window.
        code = 1
        output = (
            (expired.stdout or "")
            + (expired.stderr or "")
            + f"\nHUNG: no result within {FILE_CEILING:.0f}s\n"
        )
        if isinstance(output, bytes):  # noqa: F632 - both forms are real
            output = output.decode("utf-8", "replace")
    elapsed = time.time() - started
    tail = [
        line
        for line in output.splitlines()
        if line.startswith(("FAIL:", "ERROR:", "Ran ", "OK", "FAILED", "HUNG"))
    ]
    return stem, code == 0, elapsed, tail


# How many processes the pool is running, and how long one file may take before
# the runner reports it as hung rather than waiting forever. A file with no
# ceiling is a suite that hangs rather than fails, which is the one behaviour a
# verification step must not have.
_WORKERS = {"count": 1}
FILE_CEILING = float(os.environ.get("JUPITER_TEST_FILE_CEILING", "900"))


def workers_in_use():
    return _WORKERS["count"]


# How long each file took last time, so the next run can say how long to expect.
#
# Measured rather than guessed, because a guess would be wrong in whichever
# direction nobody checked. A file that has never run is absent from here and
# simply does not contribute to the estimate.
TIMINGS = Path(__file__).resolve().parent / "out" / "test_timings.json"
TIMINGS.parent.mkdir(parents=True, exist_ok=True)


def _load_timings():
    try:
        data = json.loads(TIMINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_timings(data):
    temporary = TIMINGS.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(TIMINGS)


def _known_durations():
    return _load_timings()


def _estimate_remaining(finished, total_files, elapsed, slowest):
    """How long this run probably has left, or None when it cannot be said.

    A parallel run finishes when its slowest file finishes, so the time still
    owed is the time the slowest file has left to run, not the average of what
    has completed. Every fast file finishing early is not progress against that,
    which is why the figure is driven by the slowest known file rather than by
    how many have passed.

    The longest time any file has ever taken on this machine is the best
    available bound, and it is what the opening line quotes. Subtracting the
    elapsed time accounts for how far into that longest file we already are.

    It is an estimate and is printed as one. A run that beats its history shows
    the count reaching the end early, which is not a fault; a run that does not
    keeps the same figure on screen, which is the honest picture.
    """
    if finished >= total_files:
        return 0.0
    history = _load_timings()
    longest = max(history.values()) if history else 0.0
    if not longest and slowest:
        longest = max(seconds for seconds, _ in slowest)
    if longest <= 0:
        return None
    return max(0.0, longest - elapsed)


def _record_timings(measured):
    """Remember how long each file took, for the next run's estimate.

    Only files that actually ran are recorded, so a file that was never reached
    does not enter the history and quietly becomes the estimate for something
    nobody has measured.
    """
    history = _load_timings()
    for name, seconds in measured.items():
        previous = history.get(name)
        # A slow outlier should not define the estimate for ever, and a fast run
        # should not erase a slow one either, so the history is the larger of the
        # two: what the next run has to budget for is the worst it has seen.
        history[name] = max(float(seconds), float(previous or 0.0))
    _save_timings(history)


def main():
    parser = argparse.ArgumentParser(
        prog="run_tests",
        description="Run the Jupiter test suite in parallel and report one verdict.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="processes to use; 0 means one per test file, capped at the cores",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failing file rather than running them all",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the final verdict and any failures",
    )
    parser.add_argument(
        "--skip-prefix",
        default="",
        help="run every test file whose name does not start with this",
    )
    parser.add_argument(
        "--only-prefix",
        default="",
        help="run only the test files whose name starts with this",
    )
    args = parser.parse_args()

    modules = _modules()
    if args.only_prefix:
        modules = [name for name in modules if name.startswith(args.only_prefix)]
    if args.skip_prefix:
        modules = [name for name in modules if not name.startswith(args.skip_prefix)]
    if not modules:
        print("no test files matched", file=sys.stderr)
        return 1

    cores = os.cpu_count() or 1
    workers = args.workers or min(len(modules), cores)
    _WORKERS["count"] = workers

    started = time.time()
    # The estimate is the honest one: how long the longest file has taken on this
    # machine, measured rather than guessed. It is the ceiling of the run, not the
    # average, because the runner cannot finish before its slowest file does.
    known = _known_durations()
    estimate = max(known.values()) if known else 60.0
    print(f"running {len(modules)} test files across {workers} processes")
    if not args.quiet:
        print(f"longest known file: {estimate:.0f}s, so expect roughly that long")
    print()

    failures = []
    slowest = []
    finished = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # as_completed rather than map. map yields in submission order, so the
        # first slow file holds back every result behind it and the progress line
        # stops moving for two minutes at a time, which is exactly the silence
        # this is meant to remove. as_completed reports each file the moment it
        # finishes, so the count advances while the slow ones are still running
        # and the estimate has something to be an estimate from.
        futures = {pool.submit(_run_one, name): name for name in modules}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, ok, elapsed, tail = future.result()
            except Exception as error:  # noqa: BLE001 - a pool failure is a failure
                ok, elapsed, tail = (
                    False,
                    0.0,
                    [f"ERROR: {type(error).__name__}: {error}"],
                )
            finished += 1
            slowest.append((elapsed, name))
            if not ok:
                failures.append((name, tail))
            if not args.quiet:
                done = time.time() - started
                left = _estimate_remaining(finished, len(modules), done, slowest)
                eta = f"eta {left:5.0f}s" if left is not None else "eta    ?"
                print(
                    f"  {'PASS' if ok else 'FAIL'}  {name}  "
                    f"({elapsed:5.1f}s)  {finished:>3}/{len(modules)}  {eta}",
                    flush=True,
                )
                if not ok:
                    for line in tail:
                        if line.startswith(("FAIL:", "ERROR:", "HUNG")):
                            print(f"          {line}", flush=True)
            elif not ok:
                for line in tail:
                    if line.startswith(("FAIL:", "ERROR:", "HUNG")):
                        print(f"          {line}", flush=True)

    total = time.time() - started
    _record_timings(dict(slowest and [(name, seconds) for seconds, name in slowest]))
    slowest.sort(reverse=True)
    print()
    print("slowest files:")
    for elapsed, name in slowest[:5]:
        print(f"  {elapsed:>6.1f}s  {name}")
    print()
    print(f"{len(modules)} files, {len(failures)} failed, {total:.1f}s wall clock")

    if failures:
        print()
        print("failures:")
        for name, tail in failures:
            print(f"  {name}")
            for line in tail:
                print(f"    {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
