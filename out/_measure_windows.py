"""How much does the measured loss move when you read a different window?

The gate judges one cycle by comparing a loss measured before the update to a
loss measured after it. Both are read from a window of the document, and a large
document has many windows. So the question that decides whether the spread is
in the measurement or in the update is:

    fixed weights, fixed document. Read W windows starting at different offsets.
    How much does the resulting estimate move?

If W windows move the estimate a lot and 16 move it little, then the spread the
gate sees is mostly the choice of window, and averaging more windows per
measurement is the fix. If they all move it equally, the window is not the
source and averaging will not help.

The weights are identical across every trial. Repeating the same measurement
gives exactly the same number, because there is no randomness in a forward pass,
so the trials vary WHICH material is read rather than how many times it is read.
That is the variation the gate actually experiences.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import learning_worker  # noqa: E402
import train  # noqa: E402
from tests.support import make_config  # noqa: E402


def measure_windows(tokens, model, c, block_size, count, offset_blocks):
    """Mean loss over `count` blocks starting `offset_blocks` in."""
    import torch

    total = 0.0
    counted = 0
    start = offset_blocks * block_size
    while counted < count:
        chunk = tokens[start : start + block_size + 1]
        if len(chunk) < block_size + 1:
            start = 0
            if counted:
                break
            continue
        x = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0)
        y = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x, y)
            total += float(
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                ).item()
            )
        counted += 1
        start += block_size
    return total / max(1, counted)


def main():
    document = sys.argv[1] if len(sys.argv) > 1 else None
    if document is None:
        candidates = sorted(Path("data/curated").glob("*.txt"))
        if not candidates:
            print("no curated documents")
            return 1
        document = max(candidates, key=lambda p: p.stat().st_size)
    document = Path(document)

    trials = int(os.environ.get("DIAG_TRIALS", "6"))
    window_counts = [1, 2, 4, 8, 16]

    work = Path(tempfile.mkdtemp(prefix="jupiter-windows-"))
    out_dir = work / "out"
    out_dir.mkdir(parents=True)
    c = make_config(
        device="cpu",
        dtype="float32",
        compile=False,
        block_size=512,
        n_layer=2,
        n_head=4,
        n_embd=128,
        micro_steps=4,
        out_dir=str(out_dir),
        knowledge_store=str(out_dir / "knowledge.json"),
        curated_dir=str(work / "curated"),
    )

    text = document.read_text(encoding="utf-8", errors="replace")
    tokenizer = learning_worker.load_tokenizer(c)
    tokens = tokenizer.encode(text[:2_000_000])

    # One model, used by every trial at every window count, so the only thing
    # that varies between them is which material was read.
    model, _ = train.build_model(c, vocab_size=tokenizer.vocab_size)
    model.eval()

    print("=== same weights, same document, different windows ===")
    print(f"  document  : {document.name}")
    print(f"  tokens    : {len(tokens):,}")
    print(f"  trials per window count : {trials}")
    print()

    results = {}
    for windows in window_counts:
        estimates = []
        for trial in range(trials):
            offset = trial * windows  # a different part of the document each time
            estimates.append(
                measure_windows(tokens, model, c, c.block_size, windows, offset)
            )
        mean = sum(estimates) / len(estimates)
        spread = max(estimates) - min(estimates)
        results[windows] = (mean, spread)
        print(
            f"  {windows:>3} window(s): mean {mean:.4f}   "
            f"range across windows {spread:.4f}"
        )

    print()
    print("=== does averaging windows shrink the range? ===")
    baseline = results[1][1]
    for windows in window_counts[1:]:
        spread = results[windows][1]
        ratio = baseline / spread if spread > 1e-9 else float("inf")
        print(
            f"  {windows:>3} windows vs 1: range {spread:.4f}  reduction {ratio:.2f}x"
        )
    print()
    print(f"  If the spread were in the measurement, 16 windows would cut it by")
    print(f"  roughly {16**0.5:.1f}x, because averaging n samples divides the")
    print(f"  standard error by the square root of n.")
    print()
    if results[window_counts[-1]][1] < baseline * 0.4:
        print("  RESULT: the spread is largely in the measurement. Reading more")
        print("  windows per measurement shrinks it, so that is where the gate's")
        print("  variance comes from and more evidence per judgment is the fix.")
    else:
        print("  RESULT: the spread does not shrink with more windows, so it is not")
        print("  the window choice. It is in the update itself, and averaging the")
        print("  measurement will not separate a step from a shuffle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
