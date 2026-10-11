"""Which acceptance configuration actually judges correctly?

Four configurations are run and measured against ground truth rather than against
each other, because "decides more often" is not a virtue. A gate that accepts
everything decides constantly and is wrong.

THE CONFIGURATIONS
    baseline            no smoothing, one window per measurement
    smoothed            a running mean over recent cycles
    windows             several windows per measurement
    smoothed+windows    both

WHAT GROUND TRUTH IS
    Between every cycle the model is scored on a fixed evaluation set that is
    never learned from. That score is the only honest answer to "did this update
    help", because it is measured on the same material every time regardless of
    which document the cycle happened to draw.

    The gate's verdict is then compared against it. A configuration is better
    when its ACCEPTs correspond to real improvements and its REJECTs correspond
    to real harm.

WHAT IS REPORTED
    accept rate      how often it accepts. High is not good on its own.
    precision        of the cycles accepted, the share that really improved
    recall           of the cycles that really improved, the share it accepted
    spurious reject  cycles that really improved and it threw away
    real gain        total improvement on material never learned from

    Spurious rejects matter most. Every one is real learning discarded, and a
    gate that rejects good work costs learning while looking busy.

WHY FOUR RUNS
    The configurations differ in how much they change and cost different amounts
    of time, so the wall clock is reported alongside the judgement. That is what
    stops a slower configuration being quietly declared best.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "out"
SWEEP = PROJECT / "temp" / "gate_compare"

CYCLES = int(os.environ.get("GATE_CYCLES", "8"))
EVAL_DOCS = int(os.environ.get("GATE_EVAL_DOCS", "6"))

# (label, smoothing, measurement windows). The update's own window count is held
# constant across all four, so only the judgement changes.
CONFIGS = [
    ("baseline", 1, 1),
    ("smoothed", 16, 1),
    ("windows", 1, 8),
    ("smoothed+windows", 16, 8),
]

SCORE_SCRIPT = """
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
import torch
import train
import learning_worker
from safetensors.torch import load_model
from config import load_config
c = load_config()
# The measurement runs on the CPU whatever the configured device is: the device
# is chosen from what is busy at the moment, so leaving it to configuration would
# make the score depend on what else happened to be running.
c.device = 'cpu'
tok = learning_worker.load_tokenizer(c)
model, _ = train.build_model(c, vocab_size=tok.vocab_size)
load_model(model, 'out/Jupiter_latest.safetensors', strict=True)
model.eval()
model.to('cpu')
total, n = 0.0, 0
for name in json.loads(r'''%s'''):
    p = Path(name)
    if not p.exists():
        continue
    text = p.read_text(encoding='utf-8', errors='replace')[:150000]
    tokens = tok.encode(text)
    if len(tokens) < c.block_size + 2:
        continue
    x = torch.tensor(tokens[:c.block_size], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(tokens[1:c.block_size + 1], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x, y)
        total += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    n += 1
print('SCORE', total / max(1, n), n)
"""

CYCLE_SCRIPT = """
import io, sys
from contextlib import redirect_stdout
sys.path.insert(0, '.')
import learning_worker
from config import load_config
from telemetry import Telemetry
c = load_config()
c.acceptance_smoothing = %d
c.measurement_windows = %d
try:
    with redirect_stdout(io.StringIO()):
        learning_worker.run_worker(c, Telemetry(c.telemetry_file), 1)
except SystemExit:
    pass
print('DONE')
"""


def snapshot():
    SWEEP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "Jupiter_latest.safetensors", SWEEP / "base.safetensors")
    shutil.copy2(OUT / "Jupiter.architecture.json", SWEEP / "base.json")


def restore():
    shutil.copy2(SWEEP / "base.safetensors", OUT / "Jupiter_latest.safetensors")
    shutil.copy2(SWEEP / "base.json", OUT / "Jupiter.architecture.json")


def eval_paths():
    docs = sorted(Path("data/curated").glob("*.txt"))
    if len(docs) <= EVAL_DOCS:
        return docs
    step = max(1, len(docs) // EVAL_DOCS)
    return docs[::step][:EVAL_DOCS]


def score_fixed_set(paths):
    """Loss on a fixed set, read identically every time."""
    script = SCORE_SCRIPT % json.dumps([str(p) for p in paths])
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SCORE"):
            parts = line.split()
            return float(parts[1]), int(parts[2])
    return None, 0


def experiments():
    path = OUT / "experiments.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def run_config(label, smoothing, windows, eval_set):
    """One configuration from the baseline, with ground truth per cycle.

    Cycles are run one at a time rather than as a block, because the fixed
    evaluation set has to be scored between them. That is the only way to know
    whether a verdict was right: a gate that accepts 25% of cycles has either
    accepted the quarter that helped or the quarter that did not, and only
    scoring between cycles tells those apart.
    """
    restore()

    env = dict(os.environ)
    env["JUPITER_NO_PROMPT"] = "1"
    env["JUPITER_CONTROL_GRANTS"] = "architecture"
    env["JUPITER_MEASUREMENT_WINDOWS"] = str(windows)
    env["JUPITER_WORKER_MAX_CYCLES"] = "1"
    env["JUPITER_WORKER_PATIENCE"] = "50"
    env["JUPITER_WORKER_IDLE_SECONDS"] = "1"
    env["JUPITER_WORKER_MAX_SECONDS"] = "300"

    verdicts = []
    real_gains = []
    started = time.time()
    ok = True
    for _step in range(CYCLES):
        before_rows = len(experiments())
        before_score, _n = score_fixed_set(eval_set)

        script = CYCLE_SCRIPT % (smoothing, windows)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT,
                env=env,
                capture_output=True,
                text=True,
                timeout=400,
            )
            if "DONE" not in proc.stdout:
                ok = False
        except subprocess.TimeoutExpired:
            ok = False
            break

        rows = [
            row
            for row in experiments()[before_rows:]
            if row.get("learning_cycle") is not None
        ]
        if not rows:
            continue
        verdicts.append(rows[-1].get("decision"))

        after_score, _n = score_fixed_set(eval_set)
        if before_score is not None and after_score is not None:
            real_gains.append(before_score - after_score)

    return {
        "label": label,
        "smoothing": smoothing,
        "windows": windows,
        "cycles": len(verdicts),
        "ran": ok,
        "seconds": time.time() - started,
        "verdicts": verdicts,
        # Positive means the model really did get better on material it had
        # never seen. This is the ground truth the verdict is scored against.
        "real_gains": real_gains,
    }


def judge(result):
    """How well the verdicts lined up with what actually happened.

    A verdict is RIGHT when accepting it coincided with the model really
    improving on the fixed set, and rejecting it coincided with the model really
    not improving. DEFER and REVIEW are not counted as wrong: they are the gate
    declining to claim knowledge it does not have, which is correct behaviour
    when the measurement cannot tell.

    Spurious rejects are the number that matters most. Every one is a real
    improvement discarded, and discarding improvements is how a run learns
    nothing while appearing busy.
    """
    pairs = list(zip(result["verdicts"], result["real_gains"]))
    if not pairs:
        return None
    accepted = [(v, g) for v, g in pairs if v == "ACCEPT"]
    rejected = [(v, g) for v, g in pairs if v == "REJECT"]
    improved = [g for _v, g in pairs if g > 0]
    return {
        "n": len(pairs),
        "accept": len(accepted),
        "reject": len(rejected),
        "defer": sum(1 for v, _ in pairs if v == "DEFER"),
        "review": sum(1 for v, _ in pairs if v == "REVIEW"),
        "precision": (
            sum(1 for _v, g in accepted if g > 0) / len(accepted) if accepted else None
        ),
        "recall": (
            sum(1 for v, g in pairs if v == "ACCEPT" and g > 0) / len(improved)
            if improved
            else None
        ),
        "spurious_rejects": sum(1 for _v, g in rejected if g > 0),
        "missed": sum(1 for v, g in pairs if v != "ACCEPT" and g > 0),
        "total_real_gain": sum(g for v, g in pairs),
        "accepted_real_gain": sum(g for _v, g in accepted),
    }


def _pct(value):
    return "n/a" if value is None else format(value, ".1%")


def main():
    print("=== gate configuration comparison ===")
    print(f"  cycles per configuration : {CYCLES}")
    print("  ground truth             : fixed evaluation set, scored each cycle")
    print()

    snapshot()
    eval_set = eval_paths()
    print(f"  evaluation documents : {len(eval_set)}")

    results = []
    for label, smoothing, windows in CONFIGS:
        print(
            f"--- {label} (smoothing={smoothing}, windows={windows}) ---",
            flush=True,
        )
        result = run_config(label, smoothing, windows, eval_set)
        counts = {}
        for verdict in result["verdicts"]:
            counts[verdict] = counts.get(verdict, 0) + 1
        print(
            f"  cycles {result['cycles']}  verdicts {counts}  {result['seconds']:.0f}s"
        )
        results.append(result)
        print()

    judged = {}
    for result in results:
        judged[result["label"]] = judge(result)

    print("=== RESULT: what the gate decided ===")
    header = (
        f"{'config':<20}{'cycles':>7}{'accept':>8}{'defer':>8}{'reject':>8}{'s':>7}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        j = judged[result["label"]] or {}
        n = max(1, j.get("n", 0))
        print(
            f"{result['label']:<20}{j.get('n', 0):>7}"
            f"{j.get('accept', 0) / n:>8.1%}"
            f"{j.get('defer', 0) / n:>8.1%}"
            f"{j.get('reject', 0) / n:>8.1%}{result['seconds']:>7.0f}"
        )

    print()
    print("=== RESULT: was the gate right? ===")
    header = (
        f"{'config':<20}{'precision':>11}{'recall':>9}"
        f"{'spurious':>10}{'real gain':>12}{'kept':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        j = judged[result["label"]]
        if not j or not j["n"]:
            print(f"{result['label']:<20}{'no data':>11}")
            continue
        kept = (
            j["accepted_real_gain"] / j["total_real_gain"]
            if j["total_real_gain"] > 0
            else None
        )
        print(
            f"{result['label']:<20}"
            f"{_pct(j['precision']):>11}{_pct(j['recall']):>9}"
            f"{j['spurious_rejects']:>10}{j['total_real_gain']:>+12.4f}"
            f"{_pct(kept):>9}"
        )

    print()
    print("  precision  of the cycles accepted, the share that really improved.")
    print("             A gate that accepts everything has no precision to report.")
    print("  recall     of the cycles that really improved, the share it kept.")
    print("  spurious   cycles that really improved and were rejected. Each is")
    print("             real learning thrown away.")
    print("  real gain  total improvement on documents never learned from.")
    print("  kept       the share of that improvement the gate let through.")

    ranked = sorted(
        (r for r in results if judged[r["label"]] and judged[r["label"]]["n"]),
        key=lambda r: (
            judged[r["label"]]["spurious_rejects"],
            -judged[r["label"]]["total_real_gain"],
        ),
    )
    if ranked:
        best = judged[ranked[0]["label"]]
        worst = judged[ranked[-1]["label"]]
        print()
        print("=== VERDICT ===")
        print(
            f"  best : {ranked[0]['label']} "
            f"({best['spurious_rejects']} spurious rejects, "
            f"real gain {best['total_real_gain']:+.4f})"
        )
        print(
            f"  worst: {ranked[-1]['label']} "
            f"({worst['spurious_rejects']} spurious rejects, "
            f"real gain {worst['total_real_gain']:+.4f})"
        )

    (OUT / "gate_config_comparison.json").write_text(
        json.dumps({"cycles": CYCLES, "results": results, "judged": judged}, indent=2),
        encoding="utf-8",
    )
    print()
    print("  written to out/gate_config_comparison.json")
    print("  out/ holds whichever configuration ran last.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
