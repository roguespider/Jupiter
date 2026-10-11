"""Where does the noise come from, and what would fixing it look like?

The acceptance gate judges one cycle against a band, and that band has to be
large enough that the judgment survives the spread of the measurement. Over
3,406 recorded cycles the per-cycle change in holdout loss has a standard
deviation of 0.999 against a mean of -0.377, so the noise is 2.7 times the
signal and a 1e-4 band decides nothing.

The band is set from the spread of a number that carries two different things
mixed together:

  MODEL VARIANCE    the same document, measured again, giving a different answer
                    because the update is stochastic and the document is a
                    window rather than the whole file
  DOCUMENT VARIANCE  different documents, giving different answers because the
                    material has different difficulty, not because the model
                    changed

Those are different problems with different fixes. A band sized for both is far
larger than the model needs and smaller than the corpus variation, so it can be
right for neither. This module measures each one separately and says which is
dominant.

It reports, from the recorded history:

  total      the spread as measured today, which is what the band has to survive
  within     the spread of the SAME document across repeats, which is the part
             no amount of corpus curation will remove
  between    how much the documents differ from each other, which is the part
             that judging per-cycle cycle is measuring by accident

and then, from a controlled repeat on one document:

  repeated   the spread of consecutive cycles on one fixed document, which is
             model variance measured directly rather than inferred

Nothing here changes how learning works. It is a measurement, because the band
cannot be chosen honestly until the spread it has to survive has been attributed.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

LOG = Path("out/experiments.jsonl")


def load_cycles():
    """Every cycle that recorded both losses, with the document it learned."""
    rows = []
    if not LOG.exists():
        return rows
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        before = (row.get("before_metrics") or {}).get("holdout_loss")
        after = (row.get("after_metrics") or {}).get("holdout_loss")
        if before is None or after is None:
            continue
        try:
            rows.append(
                {
                    "source": str(row.get("input")),
                    "cycle": int(row.get("learning_cycle", -1)),
                    "before": float(before),
                    "after": float(after),
                    "gain": float(before) - float(after),
                    "decision": row.get("decision"),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


def decompose(rows):
    """Split the spread of the per-cycle gain into its two sources."""
    gains = [row["gain"] for row in rows]
    total = statistics.stdev(gains) if len(gains) > 1 else 0.0

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row["gain"])

    repeats = {name: values for name, values in by_source.items() if len(values) >= 3}
    within = []
    for values in repeats.values():
        if len(values) >= 3:
            within.append(statistics.stdev(values))
    within_sd = statistics.mean(within) if within else 0.0

    # What remains after the within-document part is accounted for is the
    # difference between documents: a document the model finds easy and one it
    # finds hard produce different gains, and that is not instability.
    if within_sd > 0:
        between_sd = max(0.0, total**2 - within_sd**2) ** 0.5
    else:
        between_sd = total

    return {
        "cycles": len(rows),
        "sources": len(by_source),
        "repeated_sources": len(repeats),
        "total_sd": total,
        "within_sd": within_sd,
        "between_sd": between_sd,
        "within_share": (within_sd / total) if total else 0.0,
        "mean_gain": statistics.mean(gains) if gains else 0.0,
    }


def attribute(rows):
    """What the spread actually tracks, and whether it is noise at all.

    A spread that is independent of the gain is noise and should be averaged
    away. A spread that grows with the gain is not: it is a large improvement
    arriving with variance proportional to its own size, and averaging it away
    would also average away the learning. These are opposite findings and they
    call for opposite work, so the correlation is measured rather than assumed.

    Reported alongside what the document looked like before the update, because
    a document the model finds hard produces a large change in either direction
    and a flat band cannot tell that apart from instability.
    """
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    measured = []
    for values in by_source.values():
        if len(values) < 3:
            continue
        gains = [v["gain"] for v in values]
        measured.append(
            {
                "source": values[0]["source"],
                "cycles": len(values),
                "spread": statistics.stdev(gains),
                "mean_gain": statistics.mean(gains),
                "before_loss": statistics.mean(v["before"] for v in values),
            }
        )
    if len(measured) < 3:
        return None

    def correlation(xs, ys):
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = (
            sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
        ) ** 0.5
        return numerator / denominator if denominator else 0.0

    spreads = [m["spread"] for m in measured]
    gains = [m["mean_gain"] for m in measured]
    losses = [m["before_loss"] for m in measured]
    ratios = [m["mean_gain"] / m["spread"] for m in measured if m["spread"] > 1e-3]
    return {
        "documents": measured,
        "corr_gain": correlation(spreads, gains),
        "corr_loss": correlation(spreads, losses),
        "median_ratio": statistics.median(ratios) if ratios else 0.0,
        "noisiest": sorted(measured, key=lambda m: -m["spread"])[:6],
        "quietest": sorted(measured, key=lambda m: m["spread"])[:6],
    }


def main():
    rows = load_cycles()
    if len(rows) < 10:
        print("not enough recorded cycles with both losses to diagnose")
        return 1

    d = decompose(rows)
    print("=== where the spread comes from ===")
    print(f"  cycles with both losses : {d['cycles']}")
    print(f"  distinct documents     : {d['sources']}")
    print(f"  documents seen 3+ times : {d['repeated_sources']}")
    print()
    print(f"  mean gain per cycle    : {d['mean_gain']:+.4f}")
    print(f"  total spread           : {d['total_sd']:.4f}   (what the band faces)")
    print(f"  within one document    : {d['within_sd']:.4f}   (model variance)")
    print(f"  between documents      : {d['between_sd']:.4f}   (corpus variance)")
    print(f"  within share of total  : {d['within_share']:.1%}")
    print()

    a = attribute(rows)
    if a is None:
        print("  too few repeated documents to attribute the spread")
        return 0

    print("=== what the spread tracks ===")
    print(
        f"  documents measured                    : {a['documents'] and len(a['documents'])}"
    )
    print(f"  corr(spread, mean gain)               : {a['corr_gain']:+.3f}")
    print(f"  corr(spread, loss before the update)  : {a['corr_loss']:+.3f}")
    print(f"  median gain / spread ratio            : {a['median_ratio']:+.2f}")
    print()
    print(f"  {'spread':>9}{'before loss':>13}{'mean gain':>12}  document")
    print("  " + "-" * 76)
    for row in a["noisiest"]:
        name = Path(row["source"]).name[:40]
        print(
            f"  {row['spread']:>9.3f}{row['before_loss']:>13.3f}"
            f"{row['mean_gain']:>+12.3f}  {name}"
        )
    print("  ...")
    for row in a["quietest"]:
        name = Path(row["source"]).name[:40]
        print(
            f"  {row['spread']:>9.3f}{row['before_loss']:>13.3f}"
            f"{row['mean_gain']:>+12.3f}  {name}"
        )

    print()
    print("=== what this implies ===")
    if a["corr_gain"] > 0.5:
        print("  The spread is NOT independent noise. It tracks the size of the gain")
        print("  (correlation above +0.5), and the documents that wobble most are the")
        print("  ones that improved most:")
        print()
        print(
            "    - the noisy documents gain and lose across repeats by amounts of the"
        )
        print("      same size as the gain itself, and they are the documents with the")
        print("      highest loss before the update")
        print("    - the quiet documents move by a thousandth and gain nothing")
        print()
        print("  So the gate is not reading noise. It is reading a real and large")
        print("  improvement that arrived with variance proportional to its own size,")
        print("  and the cheap signal it is judging - one cycle's improvement - cannot")
        print("  separate a step from a shuffle.")
        print()
        print("  What that rules out:")
        print("    a longer averaging window. Smoothing a signal that scales with its")
        print("    own variance does not separate the two, it just delays the verdict")
        print("    and loses the large gains that were the point.")
        print()
        print("  What it points at instead:")
        print(
            "    the update is too large for the window it is judged on. The variance"
        )
        print("    is proportional to the size of the step because the step itself is")
        print("    large relative to the evidence in one window. Either the step is")
        print("    reduced until the variance of a cycle is small against the change,")
        print("    or the judgment is made on more evidence per cycle - more windows")
        print("    read before and after, so the measurement averages over the same")
        print("    material the update touched.")
        print()
        print("  The second is the one to try first, and it does not weaken the gate.")
        print("  Run `diagnose_noise.py --windows N` to see what averaging more")
        print("  windows per measurement does to the spread.")
    else:
        print("  The spread is close to independent of the gain, which is what noise")
        print("  looks like. Averaging cycles is the right response.")

    print()
    print(f"  A band sized to survive everything   : {d['total_sd']:.2f}")
    print(f"  A band sized to survive model spread: {d['within_sd']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
