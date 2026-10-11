"""Does reward-modulated Hebbian plasticity actually learn, and does it beat gradient?

Two questions, answered separately because they are different questions.

PART ONE - can the rule learn at all?
    A tiny controlled problem: one Linear layer, a known mapping, a known
    difficulty. If the rule cannot reduce loss on a problem it is set up to
    solve, then no result on a real corpus says anything, and there is no point
    running a comparison. This is the test that separates "the rule does not
    work" from "the rule does not work on this corpus".

PART TWO - does it beat gradient learning on the real corpus?
    The comparison run earlier did not answer this. It measured holdout loss on
    the document each cycle happened to draw, and the before-values spanned 2.3
    to 7.4 because every cycle saw a different document. A per-document loss is
    comparable only to itself, so most of what that run called gain was which
    document was drawn.

    So: one fixed evaluation set, scored before and after, identical for both
    modes, and never used for learning. Both modes start from the same weights.
    That is the only shape in which the answer means anything.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "out"
SWEEP = PROJECT / "temp" / "learn_compare"

CYCLES = int(os.environ.get("COMPARE_CYCLES", "8"))


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------- part one


def toy_can_the_rule_learn():
    """A single Linear layer, a known mapping, and the plasticity rule.

    The mapping is a fixed random rotation, which is learnable and is not
    something a frequency heuristic can shortcut. The reward is correctness of
    the argmax, which is the same signal the real cycle uses.
    """
    import plasticity as P
    import torch
    import torch.nn as nn

    vocab = 64
    torch.manual_seed(11)
    dim = 64
    layer = nn.Linear(dim, dim, bias=False)
    target = torch.linalg.qr(torch.randn(dim, dim))[0]

    idx = torch.randint(0, vocab, (1, 24))
    inputs = torch.randn(1, 24, dim)
    targets = (inputs @ target).argmax(dim=-1)

    class Net(nn.Module):
        """Wraps one Linear under a name the eligibility filter recognises.

        The projection is the only thing being learned, and it is named to match
        Jupiter's own blocks so the same filter selects it. A toy whose layer
        names did not match would select nothing and appear to learn nothing,
        which would say more about the filter than about the rule.
        """

        def __init__(self):
            super().__init__()
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.proj = layer
            self.blocks = nn.ModuleList([block])
            # The plasticity rule calls forward(idx, targets=None), so the toy
            # holds its own target rather than expecting one on the call.
            self.answer = targets

        def forward(self, ids, targets=None):
            answer = self.answer if targets is None else targets
            x = self.blocks[0].attn.proj(inputs)
            logits = torch.zeros(x.shape[0], x.shape[1], vocab)
            logits.scatter_(2, answer.unsqueeze(-1), x.sum(-1, keepdim=True))
            return logits, None

    def accuracy(m):
        with torch.no_grad():
            logits, _ = m(idx, targets)
            return float(logits.argmax(-1).eq(targets).float().mean().item())

    learner = P.RewardModulatedHebbian(
        P.PlasticityConfig(learning_rate=0.05, max_update_norm=0.05, max_positions=24)
    )

    model = Net()
    before = accuracy(model)
    for _ in range(12):
        learner.apply(model, idx, targets, feedback=1.0)
    after = accuracy(model)

    log("=== PART ONE: can the rule learn a known mapping? ===")
    log(f"  correct before                 : {before:.4f}")
    log(f"  correct after 12 applied events: {after:.4f}")
    log(f"  change                         : {after - before:+.4f}")
    log("")
    log("  Note: a 64-dim toy with a fixed target and no noise. It says the update")
    log("  rule is wired correctly and can move a metric. It says nothing about")
    log("  language.")
    log("")
    return after - before


# ---------------------------------------------------------------- part two


def snapshot_baseline():
    SWEEP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "Jupiter_latest.safetensors", SWEEP / "base.safetensors")
    shutil.copy2(OUT / "Jupiter.architecture.json", SWEEP / "base.json")


def restore_baseline():
    shutil.copy2(SWEEP / "base.safetensors", OUT / "Jupiter_latest.safetensors")
    shutil.copy2(SWEEP / "base.json", OUT / "Jupiter.architecture.json")


def fixed_eval_docs(count=8):
    """Documents held out of learning and used only to score both modes.

    Taken from data/curated and never offered to the worker, so no mode can learn
    from the thing it is scored on.
    """
    docs = sorted(Path("data/curated").glob("*.txt"))
    if len(docs) <= count:
        return docs
    step = max(1, len(docs) // count)
    return docs[::step][:count]


def score_eval_docs(mode, cycles, eval_paths):
    """Run one mode from the baseline and score it on the fixed set.

    The score is the mean loss over the fixed documents, measured with the same
    code for both modes, before and after the cycles.
    """
    restore_baseline()

    def score():
        script = (
            "import sys, json; sys.path.insert(0, '.')\n"
            "from pathlib import Path\n"
            "import train\n"
            "import learning_worker\n"
            "from config import load_config\n"
            "c = load_config()\n"
            "tok = learning_worker.load_tokenizer(c)\n"
            "model, _ = train.build_model(c, vocab_size=tok.vocab_size)\n"
            "import torch\n"
            "from safetensors.torch import load_model\n"
            "try:\n"
            "    load_model(model, 'out/Jupiter_latest.safetensors', strict=True)\n"
            "except Exception as e:\n"
            "    print('LOADFAIL', e)\n"
            "total = 0.0\n"
            "n = 0\n"
            "for name in json.loads(r'''%s'''):\n"
            "    p = Path(name)\n"
            "    if not p.exists():\n"
            "        continue\n"
            "    text = p.read_text(encoding='utf-8', errors='replace')[:200000]\n"
            "    tokens, _ = train.split_tokens(text, c.block_size, c.holdout_fraction, tok)\n"
            "    x, y = train.make_batch(tokens, c.block_size, c.device)\n"
            "    with torch.no_grad():\n"
            "        logits, _ = model(x, y)\n"
            "        loss = torch.nn.functional.cross_entropy(\n"
            "            logits.reshape(-1, logits.size(-1)), y.reshape(-1))\n"
            "    total += float(loss.item()); n += 1\n"
            "print('SCORE', total / max(1, n), n)\n"
        ) % json.dumps([str(p) for p in eval_paths])
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("SCORE"):
                parts = line.split()
                return float(parts[1]), int(parts[2])
        return None, 0

    env = dict(os.environ)
    env["JUPITER_LEARNING_MODE"] = mode
    env["JUPITER_WORKER_MAX_CYCLES"] = str(cycles)
    env["JUPITER_WORKER_PATIENCE"] = "50"
    env["JUPITER_WORKER_IDLE_SECONDS"] = "2"
    env["JUPITER_WORKER_MAX_SECONDS"] = "1200"
    env["JUPITER_NO_PROMPT"] = "1"

    before, n = score()

    started = time.time()
    subprocess.run(
        [sys.executable, "-u", "learning_worker.py"],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    wall = time.time() - started

    after, _ = score()

    return {
        "mode": mode,
        "eval_docs": n,
        "loss_before": before,
        "loss_after": after,
        "improvement": (before - after)
        if (before is not None and after is not None)
        else None,
        "seconds": wall,
        "seconds_per_cycle": wall / max(1, cycles),
    }


def main():
    log("=== PART TWO: fixed evaluation set, both modes, same starting weights ===")
    snapshot_baseline()
    eval_paths = fixed_eval_docs(8)
    log(f"  eval documents : {len(eval_paths)} (held out of learning)")
    log(f"  cycles per mode: {CYCLES}")
    log("")

    results = []
    for mode in ("gradient", "plasticity"):
        log(f"--- {mode} ---")
        r = score_eval_docs(mode, CYCLES, eval_paths)
        results.append(r)
        if r["improvement"] is None:
            log(f"  could not score ({r['loss_before']}, {r['loss_after']})")
        else:
            log(
                f"  fixed-set loss {r['loss_before']:.4f} -> {r['loss_after']:.4f} "
                f"= {r['improvement']:+.4f}"
            )
        log(f"  {r['seconds_per_cycle']:.1f}s per cycle")
        log("")

    log("=== RESULT (higher improvement is better) ===")
    header = f"{'mode':<14}{'loss before':>13}{'loss after':>12}{'improvement':>14}{'s/cycle':>10}"
    log(header)
    log("-" * len(header))
    for r in results:
        if r["improvement"] is None:
            log(
                f"{r['mode']:<14}{'n/a':>13}{'n/a':>12}{'n/a':>14}{r['seconds_per_cycle']:>10.1f}"
            )
        else:
            log(
                f"{r['mode']:<14}{r['loss_before']:>13.4f}{r['loss_after']:>12.4f}"
                f"{r['improvement']:>+14.4f}{r['seconds_per_cycle']:>10.1f}"
            )

    (OUT / "learning_mode_comparison.json").write_text(
        json.dumps({"cycles": CYCLES, "results": results}, indent=2), encoding="utf-8"
    )
    log("")
    log("  out/ holds whichever mode ran last. Run learning_worker.py to continue.")
    return 0


if __name__ == "__main__":
    toy = toy_can_the_rule_learn()
    sys.exit(main())
