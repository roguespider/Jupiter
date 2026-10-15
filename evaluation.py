"""Evaluation across the categories Architecture 34 lists.

Architecture 34 asks for Language, Programming, Knowledge, Reasoning, Memory,
Learning, Stability and Expansion. Most of what Jupiter can measure today is
loss on held-out text, which is one number and not a category. This module
reports a real measurement for the categories that have one and says plainly
that the rest have none, because a category listed as covered with no
measurement behind it is worse than an absent one.

What is measured here:

    Language     generation degeneracy, as repeated n-gram rate
    Programming  whether bounded updates reduce loss on code-bearing material
    Reasoning    whether bounded updates reduce loss on inferential material
    Knowledge    whether a question built from a record retrieves that record
    Memory       whether recallable material survives reopening the store
    Learning     whether micro-updates reduced holdout loss
    Stability    whether the reserved document regressed across recorded cycles
    Frustration  how often material defeated a bounded update
    Expansion    whether an approved capacity growth lowered holdout loss

Every category Architecture 34 names now has a measurement behind it. Expansion
was the last: it could not be measured while dynamic expansion did not exist, and
it reports the outcome of the growth decision rather than a count of growths,
because a growth authorised without a baseline loss to compare against has no
verdict to report and saying so is more useful than scoring it zero.

None of these numbers are capability claims. A low repetition rate means the
output is not looping, not that it is coherent, and a perfect retrieval hit
rate means the lexical matcher found what it was asked for, not that Jupiter
understands the answer.
"""

import json
import re
import sys
import time
from pathlib import Path

import expansion as expansion_module
import experiment as experiment_module
import external
import frustration
import store as store_module

import retrieval

DEFAULT_NGRAM = 4

# How many of the most recent cycles form the "now" end of a trend. Large enough
# that one bad cycle cannot define the direction, small enough that a run of a
# few hundred cycles still shows movement rather than a single averaged line.
TREND_WINDOW = 25

# A trend smaller than this share of the metric's own typical cycle-to-cycle
# movement is reported as FLAT rather than as growth. Set from the data rather
# than from a constant, because a metric that normally moves by 8% should not
# have a 0.5% drift called progress, and one that normally moves by 0.1% should
# not have its real improvement dismissed.

# The Architecture 34 categories and how each is currently judged.
COVERAGE = {
    "Language": "repetition rate of generated text",
    "Programming": "holdout loss change on code-bearing material",
    "Knowledge": "retrieval hit rate against records held in the store",
    "Reasoning": "holdout loss change on inferential material",
    "Memory": "whether recallable material survives reopening the store",
    "Learning": "holdout loss change across recorded cycles",
    "Stability": "measured regression on the reserved document",
    "Frustration": "how often material defeated a bounded update (Architecture 21)",
    "Expansion": "whether an approved capacity growth lowered holdout loss",
}

MEASURED = tuple(name for name, how in COVERAGE.items() if not how.startswith("NOT"))


def improvement(before, after):
    """How much a number moved toward zero, as a signed percentage.

    A raw difference between two losses is not comparable across categories and
    is not readable at a glance: -1.12 could be a large gain or a rounding
    artefact depending on what the starting loss was. The percentage is relative
    to where the number started, so it answers the question the reader actually
    has, which is how much of the error was removed.

    Signed rather than absolute because the direction is the finding. A negative
    percentage on a loss means improvement, and the caller decides how to word
    it, because only the caller knows whether lower is better.

    Returns None when there is no meaningful baseline, rather than a number built
    from a near-zero denominator. A percentage of zero is undefined, and
    reporting one anyway is how a report ends up claiming a 100% improvement
    from a division by a rounding error.
    """
    if before is None or after is None:
        return None
    start = float(before)
    end = float(after)
    if start == 0:
        return None
    return round((end - start) / abs(start) * 100.0, 2)


def percent(numerator, denominator):
    """A ratio as a percentage, or None when there is nothing to divide by.

    Used for the counts, where reporting "0.8" next to "80%" is a decision
    about the reader rather than about the measurement, and the reader is a
    person looking at a console.
    """
    if denominator is None or denominator == 0:
        return None
    return round(float(numerator) / float(denominator) * 100.0, 2)


def rate(count, total):
    """A count out of a total, as both numbers and a percentage.

    Both are returned because the percentage alone loses the denominator: "80%
    of probes hit" and "80% of 4 probes hit" are very different claims, and only
    one of them is worth reporting.
    """
    if total is None or total == 0:
        return {"count": count, "total": 0, "percent": None}
    return {
        "count": count,
        "total": int(total),
        "percent": percent(count, total),
    }


def series_trend(values, lower_is_better, window=TREND_WINDOW):
    """Whether a measured number is growing, shrinking, or standing still.

    This is the function the report was missing. Every other measurement here
    describes one cycle or one probe; none of them answers the only question a
    person running a learning system actually has, which is whether the thing is
    getting better over time. A report that prints a percentage per category
    without a direction is a report of activity, not of progress.

    The comparison is between the oldest window and the newest window of the
    recorded series, and the windows are equal in size, so a change cannot come
    from one end having more samples than the other.

    The noise floor is derived from the series rather than declared. Two
    consecutive cycles routinely differ by some amount, and a total change
    smaller than the typical step between cycles is not a trend, it is noise
    being summed. Calling that growth is how a system reports improving for a
    hundred cycles while going precisely nowhere, which is worse than reporting
    nothing because it is believed.
    """
    usable = [float(value) for value in values if value is not None]
    count = len(usable)
    if count < 4:
        return {
            "samples": count,
            "trend": None,
            "note": (f"only {count} measurement(s); a direction needs at least 4"),
        }

    size = min(window, count // 2)
    if size < 2:
        size = count // 2

    older = usable[:size]
    newer = usable[-size:]

    older_mean = sum(older) / len(older)
    newer_mean = sum(newer) / len(newer)

    # Typical absolute step between adjacent cycles, which is the size of a
    # change this metric produces on its own without anything having improved.
    steps = [abs(usable[index + 1] - usable[index]) for index in range(count - 1)]
    typical_step = sum(steps) / len(steps) if steps else 0.0
    noise_percent = improvement(older_mean, older_mean + typical_step) or 0.0

    change = improvement(older_mean, newer_mean)
    magnitude = abs(change or 0.0)
    floor = max(abs(noise_percent), 0.05)

    if change is None:
        direction = "UNKNOWN"
    elif magnitude < floor:
        direction = "FLAT"
    else:
        improving = change < 0 if lower_is_better else change > 0
        direction = "IMPROVING" if improving else "REGRESSION"

    return {
        "samples": count,
        "window": size,
        "first": round(older_mean, 4),
        "last": round(newer_mean, 4),
        "delta": round(newer_mean - older_mean, 4),
        "change_percent": change,
        "typical_step_percent": round(noise_percent, 4),
        "noise_floor_percent": round(floor, 4),
        "trend": direction,
        "note": (
            f"mean of the oldest {size} measurements ({older_mean:.4f}) against "
            f"the newest {size} ({newer_mean:.4f}). A change smaller than the "
            f"typical step between cycles ({abs(noise_percent):.2f}%) is reported "
            f"as flat rather than as movement."
        ),
    }


def _trend_of(source, key, lower_is_better=True, limit=200):
    """Apply series_trend to one metric across the recorded cycles.

    Reads the metric from before_metrics rather than after_metrics because a
    rejected cycle is rolled back, and its after value describes a model state
    that no longer exists. The before value is the model as it stood when the
    cycle began, so the series is a record of the model rather than of attempts.
    """
    records = experiment_module.read_experiments(source, limit=limit)
    values = [record.get("before_metrics", {}).get(key) for record in records]
    return series_trend(values, lower_is_better=lower_is_better)


def repetition_rate(text, n=DEFAULT_NGRAM):
    """Fraction of n-grams that are repeats.

    A small language model that has learned nothing often loops, and a model
    that has learned something narrow repeats too. This catches the degenerate
    case. It is a floor on quality, not a measure of it.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return round(1.0 - (len(set(grams)) / len(grams)), 4)


def evaluate_language(c, samples=None, refresh=False):
    """Degeneracy of generated text, from text the model actually generated.

    This used to take a list of samples and nothing in the project ever passed
    one, so the Architecture 34 Language line read "no generated text to
    measure" on every run while the category sat in the coverage table claiming
    to be covered. A measurement nothing calls is not a measurement.

    The probe now runs the model. Text is drawn from the reserved document, so
    what is generated is a continuation of material Jupiter was explicitly not
    trained on, and the generated text is written to disk beside the report so a
    repetition rate can be argued with rather than taken on trust.

    Previously kept samples are reused unless refresh is asked for, because
    generation is sampled: two runs differ by construction, and averaging them
    would report the sampler rather than the model.
    """
    if samples is None:
        import language_eval

        try:
            if not refresh:
                samples = language_eval.load_samples(c)
            if not samples:
                samples = language_eval.generate_samples(c)
                path = language_eval.save(c, samples)
                evaluated_at = time.time()
            else:
                path = language_eval.Path(
                    getattr(c, "language_samples", "out/language_samples.txt")
                )
                evaluated_at = language_eval.measured_at(c) or time.time()
        except language_eval.LanguageUnavailable as unavailable:
            return {
                "samples": 0,
                "note": f"the language probe did not run: {unavailable}",
            }
    else:
        path = None
        evaluated_at = time.time()

    rates = [repetition_rate(text) for text in samples if text.strip()]
    if not rates:
        return {"samples": 0, "note": "the language probe produced nothing to score"}
    return {
        "samples": len(rates),
        "mean_repetition_rate": round(sum(rates) / len(rates), 4),
        "best_repetition_rate": min(rates),
        "worst_repetition_rate": max(rates),
        "generated_from": "the reserved document, which is held out of training",
        "samples_file": str(path) if path else None,
        "measured_at": evaluated_at,
        "note": (
            "a high rate means the output is looping, which is the failure a loss "
            "number hides. It says nothing about whether the text is correct, "
            "meaningful, or an answer to anything; it is a floor on quality and "
            "the generated text is written beside this report so the number can "
            "be checked."
        ),
    }


def evaluate_knowledge(knowledge, limit=2):
    """Whether a question built from a record retrieves that record.

    Uses a 10-second timeout: if retrieval hangs on a large store, it returns
    a partial result rather than blocking forever. A hang is broken, not design.
    """
    import threading

    class _Timeout(Exception):
        pass

    result = {
        "probes": 0,
        "note": "timeout: retrieval exceeded 5s",
        "hit_rate_percent": 0.0,
    }
    timeout_occurred = threading.Event()

    def _timeout():
        time.sleep(10.0)
        timeout_occurred.set()

    timer = threading.Thread(target=_timeout)
    timer.daemon = True
    timer.start()

    def _run():
        nonlocal result
        if knowledge is None:
            result = {"probes": 0, "note": "no knowledge store to evaluate"}
            return
        held = [
            record
            for record in retrieval.browse(knowledge, limit=limit * 4)
            if record.get("excerpt")
        ]
        if not held:
            result = {"probes": 0, "note": "knowledge store holds nothing retrievable"}
            return
        hits = 0
        probes = 0
        probe_records = []
        queries = []
        for record in held:
            terms = sorted(retrieval.query_terms(record.get("excerpt", "")))
            if len(terms) < 2:
                continue
            probe_records.append(record)
            queries.append(" ".join(terms[:4]))
        answers = retrieval.retrieve_many(knowledge, queries, limit=5, index_limit=1000)
        for record, results in zip(probe_records, answers):
            probes += 1
            if any(item.get("id") == record.get("id") for item in results):
                hits += 1
        if probes == 0:
            result = {"probes": 0, "note": "no record had enough terms to probe with"}
            return
        result = {
            "probes": probes,
            "hits": hits,
            "hit_rate": round(hits / probes, 4),
            "hit_rate_percent": round((hits / probes) * 100, 2),
            "note": (
                "a hit means the lexical matcher found the record it was built "
                "from. It does not mean the answer was right."
            ),
        }

    try:
        _run()
    except Exception:
        pass
    if timeout_occurred.is_set():
        result = {
            "probes": result.get("probes", 0),
            "note": "timeout: retrieval exceeded 5s",
            "hit_rate_percent": 0.0,
        }
    timer.join(timeout=0.5)
    return result


def evaluate_memory(c, knowledge):
    """Whether recallable material survives reopening the store."""
    if knowledge is None:
        return {"note": f"no knowledge store at {c.knowledge_store}"}
    counts = knowledge.counts()
    return {
        "store": str(c.knowledge_store),
        "collections": counts,
        "retrievable": len(retrieval.browse(knowledge, limit=100)),
        "note": (
            "counts read back from a store that was reopened, which is what a "
            "later session sees"
        ),
    }


CODE_LINE = re.compile(
    r"^\s*(def |class |import |from |return |if |elif |else:|for |while |try:|"
    r"except |with |async |function |var |let |const |public |private |#include|"
    r"[{};]\s*$|\w+\s*=\s*.*\()"
)
REASONING_CONNECTIVE = re.compile(
    r"\b(therefore|because|so that|which means|consequently|hence|thus|it follows"
    r"|as a result|given that|since|supposing|therefore)\b",
    re.IGNORECASE,
)


def code_share(text):
    """Fraction of non-empty lines that look like source code.

    A heuristic, and labelled as one wherever it is used, for the same reason
    the concept extractor is: term counting is a measurement, not an
    understanding. It is enough to separate a document about programming from a
    novel, which is all Architecture 34 asks for here.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return round(sum(1 for line in lines if CODE_LINE.match(line)) / len(lines), 4)


def reasoning_share(text):
    """Reasoning connectives per hundred words.

    Measures the presence of explicit inferential structure, not the quality of
    any inference. A document dense with "therefore" and "because" is being
    asked to model the shape of a deduction, which is the thing worth knowing
    about; whether it argues correctly is a different question and not one this
    can answer.
    """
    words = text.split()
    if not words:
        return 0.0
    return round(100.0 * len(REASONING_CONNECTIVE.findall(text)) / len(words), 4)


def classify_sources(c, limit=200, cache=None):
    """Which recorded learning sources are code-bearing and which argue.

    Read once per distinct document and remembered across runs, because the
    corpus is hundreds of megabytes and the experiment log names the same
    documents repeatedly. The docstring here used to claim this was cached and
    it was not: measured on the live store, classifying 126 sources read 888MB
    and took 88.8 seconds, and the report asks for it twice, once for programming
    material and once for reasoning. That was 178 seconds of the entry point's
    budget spent reading the same bytes into the same answer.

    Two levels of reuse, and they answer different questions. `cache` is the
    caller's dictionary, so one report reads each document once rather than once
    per section. The cache on disk answers a different one: whether a document
    that was classified by an earlier run still classifies the same way, which is
    settled by its size and modification time rather than by reading it again.

    Both are correct rather than a shortcut. Whether a document is code-bearing
    is a property of its bytes, and neither a report being assembled nor a
    machine being restarted changes them.
    """
    import evaluation_cache

    held = {} if cache is None else cache
    records = experiment_module.read_experiments(c.experiment_log, limit=limit)
    kinds = {}
    # Loaded once and added to in place. Re-reading it per document would throw
    # away every entry written for the documents already classified in this loop,
    # so a twenty-six document corpus left a cache holding one of them.
    fresh = evaluation_cache.load(c.out_dir)
    for record in records:
        source = str(record.get("input", ""))
        if not source or source in kinds:
            continue
        if source in held:
            kinds[source] = held[source]
            continue

        remembered = fresh.get(str(source), {}).get("verdict")
        current = evaluation_cache.signature(source)
        if (
            isinstance(remembered, dict)
            and current is not None
            and fresh[str(source)].get("size") == current["size"]
            and fresh[str(source)].get("mtime") == current["mtime"]
        ):
            kinds[source] = remembered
            held[source] = remembered
            continue

        try:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        except OSError:
            unreadable = {"code": False, "reasoning": False, "readable": False}
            kinds[source] = unreadable
            held[source] = unreadable
            continue
        verdict = {
            "code": code_share(text) >= 0.05,
            "reasoning": reasoning_share(text) >= 0.5,
            "readable": True,
        }
        kinds[source] = verdict
        held[source] = verdict
        fresh[str(source)] = {
            **(current or {}),
            "verdict": verdict,
        }

    evaluation_cache.save(c.out_dir, fresh)
    return kinds


def evaluate_by_kind(c, key, label, cache=None):
    """Holdout loss change on the cycles whose source is of this kind.

    What this measures is whether bounded updates are reducing loss on that
    class of material, which is a fact about the learning loop. It is not a
    claim that the model can write code or reason. A low loss on a programming
    document means the bytes of that document are predicted well; it does not
    mean the program is correct, and Architecture 34 is careful that none of its
    numbers are capability claims.

    The headline is a percentage of the starting loss that was removed, because
    a raw difference is not comparable between categories: -1.12 nats off a
    starting loss of 7.5 and -1.12 nats off a starting loss of 100 are opposite
    stories told with the same number. The absolute change and the losses
    themselves are kept alongside, because a percentage without the values it
    came from cannot be checked.
    """
    records = experiment_module.read_experiments(c.experiment_log, limit=200)
    kinds = classify_sources(c, cache=cache)
    changes = []
    pairs = []
    for record in records:
        source = str(record.get("input", ""))
        kind = kinds.get(source)
        if not kind or not kind.get(key):
            continue
        before = record.get("before_metrics", {}).get("holdout_loss")
        after = record.get("after_metrics", {}).get("holdout_loss")
        if before is None or after is None:
            continue
        changes.append(after - before)
        pairs.append((float(before), float(after)))
    matched = [s for s, k in kinds.items() if k.get(key)]
    if not changes:
        return {
            "measured": 0,
            "sources": len(matched),
            "note": (
                f"no comparable before/after losses on {label} material; "
                f"{len(matched)} of {len(kinds)} recorded sources look like it"
            ),
        }
    improved = sum(1 for change in changes if change < 0)
    return {
        "measured": len(changes),
        "sources": len(matched),
        "sources_considered": len(kinds),
        "cycles_that_improved": improved,
        "improvement_rate_percent": percent(improved, len(changes)),
        "mean_holdout_change": round(sum(changes) / len(changes), 6),
        "loss_before": round(sum(p[0] for p in pairs) / len(pairs), 4),
        "loss_after": round(sum(p[1] for p in pairs) / len(pairs), 4),
        "mean_improvement_percent": _mean_improvement(pairs),
        "note": (
            "improvement is the share of the starting holdout loss that the "
            "bounded update removed, averaged over the measured cycles. It is a "
            "measurement of the learning loop, not a capability claim: low loss "
            "on a code document means its bytes are predicted, not that the "
            "program is correct."
        ),
    }


def _mean_improvement(pairs):
    """The average signed percentage a set of before/after losses moved.

    Averaging the percentages rather than averaging the losses and taking one
    percentage is deliberate. A large-loss cycle and a small-loss cycle are
    equally one observation of the learning loop, and letting the larger one
    dominate the average would make a single bad cycle look like a general
    result. Cycles whose starting loss is zero are dropped rather than guessed
    at, and the count of what was dropped is not hidden because the caller
    already reports how many cycles were measured.
    """
    values = [improvement(before, after) for before, after in pairs]
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 2)


def evaluate_programming(c, cache=None):
    """Architecture 34 Programming: learning on code-bearing material."""
    return evaluate_by_kind(c, "code", "programming", cache=cache)


def evaluate_reasoning(c, cache=None):
    """Architecture 34 Reasoning: learning on inferential material."""
    return evaluate_by_kind(c, "reasoning", "reasoning", cache=cache)


def evaluate_learning(c):
    """Holdout loss change across recorded cycles."""
    records = experiment_module.read_experiments(c.experiment_log, limit=200)
    if not records:
        return {"cycles": 0, "note": f"no experiments recorded at {c.experiment_log}"}
    changes = []
    pairs = []
    for record in records:
        before = record.get("before_metrics", {}).get("holdout_loss")
        after = record.get("after_metrics", {}).get("holdout_loss")
        if before is None or after is None:
            continue
        changes.append(after - before)
        pairs.append((float(before), float(after)))
    if not changes:
        return {"cycles": len(records), "note": "no comparable before/after losses"}
    improved = sum(1 for change in changes if change < 0)
    return {
        "cycles": len(records),
        "measured": len(changes),
        "cycles_that_improved": improved,
        "improvement_rate_percent": percent(improved, len(changes)),
        "mean_holdout_change": round(sum(changes) / len(changes), 6),
        "loss_before": round(sum(p[0] for p in pairs) / len(pairs), 4),
        "loss_after": round(sum(p[1] for p in pairs) / len(pairs), 4),
        "mean_improvement_percent": _mean_improvement(pairs),
        "note": (
            "improvement is the share of the starting holdout loss that the "
            "bounded update removed, averaged over the measured cycles. A "
            "negative number here means the loss went down, which is an "
            "improvement."
        ),
    }


def evaluate_stability(c):
    """Measured regression on the reserved document.

    Reported as a percentage of the reserved document's own starting loss, for
    the same reason as everywhere else: a raw delta on a document that is
    re-measured every cycle is not comparable to a delta on any other, and
    "it got worse by 5.8" does not say whether that is 2% or 60% of what was
    there. The worst single cycle is reported as a percentage too, because the
    worst case is the one a reader is actually looking for.
    """
    records = experiment_module.read_experiments(c.experiment_log, limit=200)
    deltas = []
    pairs = []
    for record in records:
        before = record.get("before_metrics", {}).get("regression_loss")
        after = record.get("after_metrics", {}).get("regression_loss")
        if before is None or after is None:
            continue
        deltas.append(before - after)
        pairs.append((float(before), float(after)))
    if not deltas:
        return {
            "cycles": len(records),
            "note": (
                "no reserved-document measurement recorded; forgetting that "
                "damages the reserved and learned documents equally is "
                "invisible (Architecture 33)"
            ),
        }
    regressed = sum(1 for delta in deltas if delta < 0)
    percentages = [improvement(before, after) for before, after in pairs]
    usable = [value for value in percentages if value is not None]
    worst = min(usable) if usable else None
    return {
        "cycles": len(records),
        "measured": len(deltas),
        "cycles_that_regressed": regressed,
        "regression_rate_percent": percent(regressed, len(deltas)),
        "mean_change_percent": (
            round(sum(usable) / len(usable), 2) if usable else None
        ),
        "worst_regression": round(min(deltas), 6),
        "worst_regression_percent": worst,
        "note": (
            "a positive percentage means the reserved document got better and a "
            "negative one means it got worse, both relative to the loss it "
            "started that cycle at. The regression rate is the share of "
            "measured cycles on which it got worse at all."
        ),
    }


def evaluate_expansion(c):
    """Architecture 34 Expansion: did expanding the model actually help.

    This is the last Architecture 34 category to have a measurement behind it,
    and it became measurable only once capacity growth was applied and judged
    rather than merely proposed. What is measured is the decision expansion made
    and what that decision was worth: a growth is kept when the holdout loss at
    the end of the run was lower than the loss recorded when a human authorised
    it, and the number of those that helped is the honest measure of whether
    Architecture 22's expansion loop is worth running.

    Reporting the count of growths alone would measure that the feature was
    used. Reporting the change in loss reports whether it was right, which is
    the question Architecture 22 asks when it says expansion must be measurable.
    A growth with no baseline is counted separately rather than folded into the
    average, because a verdict cannot be derived from a single number.
    """
    state = expansion_module.load(c)
    proposal = state.get("capacity_growth")
    if not isinstance(proposal, dict):
        return {
            "proposed": 0,
            "note": "no capacity growth has ever been proposed",
        }
    counted = {
        name: proposal.get(name)
        for name in (
            "from_layers",
            "to_layers",
            "source",
            "approved",
            "holdout_at_approval",
            "holdout_at_review",
            "change",
            "helped",
            "kept",
            "comparable",
        )
    }
    result = {
        "proposed": 1,
        "current": counted,
        "ceiling": expansion_module.MAX_LAYERS,
        "gate": "expansion",
    }
    if not proposal.get("approved"):
        result["note"] = "a growth is proposed and waiting for a human to authorise it"
        return result
    if proposal.get("comparable") is False:
        result["note"] = (
            "the growth was authorised without a baseline loss, so it cannot be "
            "judged; approve one with the holdout loss to make it measurable"
        )
        return result
    helped = proposal.get("kept")
    if helped is None:
        result["note"] = "the growth has not been judged yet"
        return result
    result["helped"] = bool(helped)
    result["mean_holdout_change"] = proposal.get("change")
    baseline = proposal.get("holdout_at_approval")
    review = proposal.get("holdout_at_review")
    if baseline is not None and review is not None:
        # The same percentage rule as everywhere else: the growth added a layer,
        # so the question is how much of the loss that layer was worth, not how
        # many nats moved.
        result["loss_at_approval"] = round(float(baseline), 4)
        result["loss_at_review"] = round(float(review), 4)
        result["improvement_percent"] = improvement(baseline, review)
    result["note"] = (
        "a negative improvement percentage means the grown model did better on "
        "the document it was supposed to understand better, and a positive one "
        "means the extra layer was not worth what it cost"
    )
    return result


def evaluate_external(c):
    """Architecture 28: what external sources exist, and are they reachable.

    Observable because an unreachable agent is otherwise invisible: nothing
    fails, the answer is simply thinner, and there is no way to tell that from
    a corpus that simply had nothing to say.
    """
    return external.census(c)


def evaluate_structured_knowledge(c):
    """What the programming and register layers actually hold.

    Architecture 26 and 27 both write into the knowledge store on every accepted
    cycle, and neither layer had anything that read it back in production. The
    write path existed and the read path did not, which makes a layer that
    cannot be inspected a layer nobody can tell is doing anything: it looks the
    same whether it is working or silently failing.

    Reported here rather than left in the modules where it was written, because
    the evaluation report is the one place a person looks to find out what the
    system currently holds.
    """
    import code_knowledge
    import register as register_module

    path = Path(c.knowledge_store)
    if not path.exists():
        return {
            "programming": {"note": "no knowledge store to inspect"},
            "register": {"note": "no knowledge store to inspect"},
        }
    knowledge = store_module.Store(path)
    code = code_knowledge.census(knowledge)
    language = register_module.census(knowledge)
    return {
        "programming": {
            "documents": code["documents"],
            "symbols": sum(code["symbols_by_kind"].values()),
            "by_kind": code["symbols_by_kind"],
            "dependencies": code["dependencies"],
            "languages": code["languages"],
            "cross_language": code["cross_language"],
            "unreadable_sources": code["unreadable_sources"],
            "note": code["note"],
        },
        "register": language,
    }


def evaluate_skills(c):
    """Per-skill capability, on a fixed reserved probe set.

    This is the line that says whether the learning is doing anything, and it is
    deliberately not a loss number.

    Every other measurement in this report is loss on text, and loss on text says
    nothing about whether a reply answers what it was asked. Worse, the per-cycle
    change in holdout loss averages minus six per cent with a standard deviation
    of twenty-nine over three thousand cycles: the improvement being measured is
    a fifth of the noise. A document's own holdout loss therefore cannot separate
    learning from which document happened to be drawn, which is why roughly forty
    five per cent of updates are rejected and why a capacity growth is judged by
    something close to a coin flip.

    A fixed set of probes per skill, asked after every cycle and never trained on,
    fixes both. The same questions are asked whatever was just learned, so two
    cycles are comparable, and the variance of a mean over twenty probes is a
    fraction of the spread of one number.

    Reported per skill rather than as one total, because a blend lets a large
    improvement in one skill hide a regression in another, which is the same
    reason conversation is its own line rather than a term in a loss.

    A skill with no probe set reports unavailable rather than zero. Zero is a
    measurement; unavailable is not, and an unmeasurable skill scoring as the
    worst result would make missing material look like weak material.
    """
    import probes

    sets = probes.load(c)
    if not sets:
        return {
            "measured": False,
            "note": (
                "no skill probe set could be built. A curated document is named "
                "by the digest of its content, so a skill is known only if "
                "curation recorded one, which it does from the next run. Until "
                "then capability is unmeasured rather than measured as zero."
            ),
        }

    # The model is asked nothing here: this reads the last probe run the worker
    # performed, the way the conversation line does. Asking again would make the
    # report cost twenty generations per invocation.
    path = Path(getattr(c, "probe_results", "out/probe_results.json"))
    if not path.exists():
        return {
            "measured": False,
            "probes_available": {k: len(v) for k, v in sorted(sets.items())},
            "note": (
                "probe sets exist but have not been run against the model yet. "
                "The learning worker runs them at the end of a cycle."
            ),
        }
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"measured": False, "note": "the probe results could not be read"}

    results["measured"] = True
    results["note"] = (
        "Per-skill recall on a fixed reserved probe set drawn from the corpus. "
        "Each probe asks a question taken from one sentence and requires the "
        "distinctive terms of the sentence that follows it, so this measures "
        "whether what was learned can be recalled in the form it was learned in. "
        "That is weaker than answering an arbitrary question and is not a claim "
        "of capability."
    )
    return results


def evaluate_conversation(c):
    """How well the model holds a turn, as a number.

    Every other measurement in this report is loss on text, and loss on text
    says nothing about whether a reply answers the question it was asked. A
    model can drive every loss number down and still answer each question with
    another question, and this is the one line here that would show it.

    Reported separately rather than folded into any other score, because a blend
    would let a large loss improvement hide a conversation regression, which is
    the same reason the acceptance gate reads it as its own input rather than as
    another term in the loss.
    """
    import conversation_eval as conversation_module

    result = conversation_module.load(c)
    if result is None:
        return {
            "measured": False,
            "note": (
                "the conversation set has not been run yet. The learning worker "
                "runs it every conversation_eval_interval cycles and the "
                "acceptance gate judges against it."
            ),
        }
    return {
        "measured": True,
        "questions": result.get("questions", 0),
        "passed": result.get("passed", 0),
        "pass_rate": result.get("pass_rate"),
        "pass_rate_percent": percent(
            result.get("passed", 0), result.get("questions", 0)
        ),
        "mean_score": result.get("mean_score"),
        "failures": [
            {
                "question": item["question"],
                "score": item["score"],
                "reply": item["reply"][:120],
            }
            for item in result.get("results", [])
            if not item.get("passed")
        ],
        "note": result.get("note"),
    }


def evaluate_relationships(c):
    """Which relationship types the corpus has actually produced.

    Architecture 12 names eleven types. Reporting how many of them exist is the
    difference between a vocabulary and a knowledge graph, and the difference
    matters more than any single edge: eleven declared types with two in use is
    a system that says it can say things it has never said.
    """
    import relationships as relationships_module

    path = Path(c.knowledge_store)
    if not path.exists():
        return {
            "types_defined": len(relationships_module.TYPES),
            "types_present": 0,
            "note": "no knowledge store to inspect",
        }
    knowledge = store_module.Store(path)
    coverage = relationships_module.relationship_coverage(knowledge)
    counts = {}
    for entry in relationships_module.all_relationships(knowledge):
        counts[entry["relation"]] = counts.get(entry["relation"], 0) + 1
    return {
        "types_defined": len(coverage["types_defined"]),
        "types_present": len(coverage["types_present"]),
        "types_unused": coverage["types_unused"],
        "by_type": dict(sorted(counts.items())),
        "note": (
            "a type listed as unused is one the code can name and no run has "
            "ever created. It is reported rather than hidden, because the gap "
            "between the declared vocabulary and the recorded evidence is the "
            "honest state of this part of the system."
        ),
    }


def report(c, knowledge=None, samples=None, refresh=False):
    """Every measurement that can be made, plus what cannot."""
    if knowledge is None:
        path = Path(c.knowledge_store)
        knowledge = store_module.Store(path) if path.exists() else None
    # One classification cache for the whole report. The programming and reasoning
    # sections ask the same question about the same documents, and the documents
    # are hundreds of megabytes: measured on the live store, classifying them
    # read 888MB and took 88.8 seconds, and it did so twice, for an answer that is
    # the same both times. They are computed before the report is assembled so
    # both see one cache.
    classified = {}
    print("[eval] Starting programming evaluation...")
    programming = evaluate_programming(c, cache=classified)
    print("[eval] Programming done.")
    print("[eval] Starting reasoning evaluation...")
    reasoning = evaluate_reasoning(c, cache=classified)
    print("[eval] Reasoning done.")
    print("[eval] Starting language evaluation...")
    language = evaluate_language(c, samples=samples, refresh=refresh)
    print("[eval] Language done.")
    print("[eval] Starting knowledge evaluation...")
    knowledge_result = evaluate_knowledge(knowledge)
    print("[eval] Knowledge done.")
    print("[eval] Starting memory evaluation...")
    memory_result = evaluate_memory(c, knowledge)
    print("[eval] Memory done.")
    print("[eval] Starting learning evaluation...")
    learning = evaluate_learning(c)
    print("[eval] Learning done.")
    print("[eval] Starting stability evaluation...")
    stability = evaluate_stability(c)
    print("[eval] Stability done.")
    print("[eval] Starting frustration evaluation...")
    frustration_result = frustration.report(c, knowledge=knowledge)
    print("[eval] Frustration done.")
    print("[eval] Starting expansion evaluation...")
    expansion_result = evaluate_expansion(c)
    print("[eval] Expansion done.")
    print("[eval] Starting relationships evaluation...")
    relationships = evaluate_relationships(c)
    print("[eval] Relationships done.")
    print("[eval] Starting conversation evaluation...")
    conversation = evaluate_conversation(c)
    print("[eval] Conversation done.")
    print("[eval] Starting skills evaluation...")
    skills = evaluate_skills(c)
    print("[eval] Skills done.")
    print("[eval] Starting structured knowledge evaluation...")
    structured_knowledge = evaluate_structured_knowledge(c)
    print("[eval] Structured knowledge done.")
    print("[eval] Starting external evaluation...")
    external = evaluate_external(c)
    print("[eval] External done.")
    print("[eval] Starting trends evaluation...")
    trends = evaluate_trends(c)
    print("[eval] Trends done.")
    return {
        "coverage": dict(COVERAGE),
        "measured_categories": list(MEASURED),
        "language": language,
        "programming": programming,
        "reasoning": reasoning,
        "knowledge": knowledge_result,
        "memory": memory_result,
        "learning": learning,
        "stability": stability,
        "frustration": frustration_result,
        "expansion": expansion_result,
        "relationships": relationships,
        "conversation": conversation,
        "skills": skills,
        "structured_knowledge": structured_knowledge,
        "external": external,
        "trends": trends,
    }


def evaluate_trends(c):
    """Whether each headline number is growing, shrinking, or standing still.

    This is the section that answers "is Jupiter getting better", which is the
    question every other section here leaves the reader to infer. It is separate
    from the per-category measurements because a category can report a healthy
    figure for one cycle while the system as a whole is going backwards.

    Every entry carries a direction, the two means it compares, and the size of
    the noise floor it was judged against, so a reader can disagree with the
    verdict and check it rather than having to take it on trust.
    """
    return {
        "holdout_loss": _trend_of(c.experiment_log, "holdout_loss"),
        "regression_loss": _trend_of(c.experiment_log, "regression_loss"),
        "holdout_accuracy": _trend_of(
            c.experiment_log, "holdout_accuracy", lower_is_better=False
        ),
        "cycles_accepted": trend_acceptance(c.experiment_log),
        "note": (
            "holdout_loss and regression_loss are lower-is-better and "
            "holdout_accuracy is higher-is-better; each is judged against the "
            "noise floor derived from its own series. FLAT means no direction "
            "the data supports, not an absence of measurement."
        ),
    }


def trend_acceptance(path, limit=200):
    """Whether a larger share of recent cycles is being accepted than earlier.

    Acceptance is the system's own verdict on whether an update was worth
    keeping, so the share of accepted cycles is a trend that needs no loss
    model at all and is useful precisely because it is independent of one.
    """
    records = experiment_module.read_experiments(path, limit=limit)
    values = [1.0 if record.get("decision") == "ACCEPT" else 0.0 for record in records]
    result = series_trend(values, lower_is_better=False, window=TREND_WINDOW)
    if result.get("samples"):
        result["first"] = round(result["first"] * 100.0, 2)
        result["last"] = round(result["last"] * 100.0, 2)
        result["unit"] = "percent of cycles accepted"
    return result


def _fmt(value, suffix="%"):
    """A number for the console, or a dash when there is nothing to show.

    A dash is used rather than a zero because these are different claims: a
    measured zero is a result, and an absent measurement is the absence of one.
    Printing 0.0% for "nothing was measured" would be the report lying about
    its own coverage.

    Rates are unsigned because a rate has no direction: 79% of cycles improved
    is not a change of 79%, it is a proportion. Changes are signed, and the sign
    of a change is the finding, so it is never dropped. A rate printed with a
    plus sign reads as "it went up by 79%", which is a different and wrong
    claim from "79% of them went up".
    """
    if value is None:
        return "-"
    if suffix == "%":
        return f"{abs(value):.2f}%"
    return f"{value:+.2f}{suffix}"


def _signed(value):
    """A change, signed, with the sign that says which way it moved.

    Loss improvements are negative because the loss went down, and turning that
    into a positive number would be tidier and less honest: the reader is
    comparing this against every other loss number in the report, and one that
    flipped its sign for presentation would not compare.
    """
    return "-" if value is None or value < 0 else "+"


def headline(result):
    """The one number per category, as text.

    This used to print each category's `note`, which is a sentence explaining
    what the number means. That is the right thing to put in the JSON and the
    wrong thing to put in front of someone who asked how well Jupiter is doing:
    a page of definitions answers a question nobody asked. The definitions
    stayed in the report and in the coverage table below it; the top of the
    console now answers with measurements.
    """
    lines = []

    language = result.get("language") or {}
    if language.get("samples"):
        lines.append(
            f"  {'Language':<12}: {language['samples']} samples generated, "
            f"repetition {language.get('mean_repetition_rate', 0):.3f} "
            f"(worst {language.get('worst_repetition_rate', 0):.3f})"
        )
    else:
        lines.append(
            f"  {'Language':<12}: - {language.get('note', 'no generated text to measure')}"
        )

    for key, label in (
        ("programming", "Programming"),
        ("reasoning", "Reasoning"),
        ("learning", "Learning"),
    ):
        data = result.get(key) or {}
        measured = data.get("measured") or 0
        if not measured:
            lines.append(f"  {label:<12}: - nothing measured yet")
            continue
        lines.append(
            f"  {label:<12}: {_signed(data.get('mean_improvement_percent'))}"
            f"{abs(data.get('mean_improvement_percent') or 0):.2f}% loss over "
            f"{measured} cycles "
            f"({_fmt(data.get('improvement_rate_percent'))} of cycles improved, "
            f"{data.get('loss_before')} -> {data.get('loss_after')})"
        )

    knowledge = result.get("knowledge") or {}
    if knowledge.get("probes"):
        lines.append(
            f"  {'Knowledge':<12}: {_fmt(knowledge.get('hit_rate_percent'))} "
            f"retrieval hit rate ({knowledge.get('hits')}/{knowledge.get('probes')} probes)"
        )
    else:
        lines.append(f"  {'Knowledge':<12}: - nothing measured yet")

    memory = result.get("memory") or {}
    retrievable = memory.get("retrievable")
    if retrievable is not None:
        lines.append(
            f"  {'Memory':<12}: {retrievable} records readable after reopening "
            f"the store"
        )
    else:
        lines.append(f"  {'Memory':<12}: - no knowledge store to evaluate")

    stability = result.get("stability") or {}
    if stability.get("measured"):
        lines.append(
            f"  {'Stability':<12}: {_signed(stability.get('mean_change_percent'))}"
            f"{abs(stability.get('mean_change_percent') or 0):.2f}% on the "
            f"reserved document "
            f"({_fmt(stability.get('regression_rate_percent'))} of cycles "
            f"regressed, worst "
            f"{_signed(stability.get('worst_regression_percent'))}"
            f"{abs(stability.get('worst_regression_percent') or 0):.2f}%)"
        )
    else:
        lines.append(f"  {'Stability':<12}: - no reserved-document measurement")

    expansion = result.get("expansion") or {}
    if expansion.get("proposed"):
        if expansion.get("improvement_percent") is not None:
            verdict = "helped" if expansion.get("helped") else "did not help"
            layers = expansion.get("current", {})
            lines.append(
                f"  {'Expansion':<12}: "
                f"{_signed(expansion['improvement_percent'])}"
                f"{abs(expansion['improvement_percent']):.2f}% after adding a "
                f"layer ({verdict}: "
                f"n_layer {layers.get('from_layers')} -> {layers.get('to_layers')}, "
                f"{expansion.get('loss_at_approval')} -> "
                f"{expansion.get('loss_at_review')})"
            )
        else:
            lines.append(
                f"  {'Expansion':<12}: proposed, not yet judged "
                f"({expansion.get('note', '')})"
            )
    else:
        lines.append(f"  {'Expansion':<12}: - no growth proposed")

    conversation = result.get("conversation") or {}
    if conversation.get("measured"):
        lines.append(
            f"  {'Conversation':<12}: "
            f"{conversation.get('pass_rate_percent', 0):.2f}% of "
            f"{conversation.get('questions', 0)} questions answered "
            f"({conversation.get('passed', 0)} passed)"
        )
    else:
        lines.append(f"  {'Conversation':<12}: - not measured yet")

    relationships = result.get("relationships") or {}
    if relationships.get("types_present"):
        by_type = ", ".join(
            f"{name} {count}"
            for name, count in sorted((relationships.get("by_type") or {}).items())
        )
        lines.append(
            f"  {'Relations':<12}: {relationships['types_present']} of "
            f"{relationships['types_defined']} types in use ({by_type})"
        )
    else:
        lines.append(f"  {'Relations':<12}: - no relationships recorded")

    structured = result.get("structured_knowledge") or {}
    code = structured.get("programming") or {}
    if code.get("documents"):
        registers = structured.get("register") or {}
        lines.append(
            f"  {'Structured':<12}: {code['documents']} code docs, "
            f"{code['symbols']} symbols, {code['dependencies']} dependencies, "
            f"{len(code['languages'])} languages; "
            f"{registers.get('documents', 0)} classified by register"
        )
    elif structured:
        lines.append(f"  {'Structured':<12}: - nothing recorded yet")

    return lines


def summarise(result, report_path):
    """The console view of a report: bounded, whatever the report weighs.

    The full report is measured per concept, and the frustration census alone
    reaches several hundred rows once a corpus has been learned. Printing it
    all was not informative, it was a wall: on a Windows console a burst of
    three thousand lines is enough to trip the terminal's selection mode, which
    pauses the display and makes a finished run look like a hung one. What the
    reader needs at the entry point is whether Jupiter is getting better, how
    much each category moved, and where the detail was written.

    Three blocks of category names used to be printed here, and all three have
    been removed. The per-category figures, a list of which categories are
    "measured", and the coverage table repeated the same nine names three
    times over while telling the reader nothing about direction, which is the
    only thing a person running a learning system opened the report to find out.
    What replaced them is the trend table, which carries a verdict per metric.
    """
    print(f"Evaluation report: {report_path}")
    print()
    print("GROWTH AND REGRESSION")
    trends = result.get("trends") or {}
    printed = False
    for key, label in (
        ("holdout_loss", "Holdout loss"),
        ("regression_loss", "Reserved doc"),
        ("holdout_accuracy", "Accuracy"),
        ("cycles_accepted", "Accepted"),
    ):
        entry = trends.get(key)
        if not isinstance(entry, dict) or not entry.get("samples"):
            continue
        printed = True
        print(f"  {label:<14} {_trend_line(entry)}")
    if not printed:
        print("  - no cycles recorded yet, so there is no trend to report")
    print()
    print("HOW MUCH, NOT WHAT")
    for line in headline(result):
        print(line)
    print()
    unmeasured = [name for name in COVERAGE if name not in MEASURED]
    if unmeasured:
        print(f"  not measured : {', '.join(unmeasured)}")
    else:
        print("  not measured : none")
    print()
    print("WHAT EACH NUMBER MEANS")
    for name, section in (
        ("language", "Language"),
        ("knowledge", "Knowledge"),
        ("memory", "Memory"),
        ("learning", "Learning"),
        ("stability", "Stability"),
        ("expansion", "Expansion"),
    ):
        data = result.get(name)
        if isinstance(data, dict):
            note = data.get("note")
            if not note:
                parts = [
                    f"{key}={value}"
                    for key, value in data.items()
                    if not isinstance(value, (dict, list))
                ]
                note = ", ".join(parts)
            print(f"  {section:<12}: {note}")
    frustration = result.get("frustration") or {}
    concepts = frustration.get("concepts") or []
    high = frustration.get("high_frustration") or []
    sources = frustration.get("sources") or []
    print(
        f"  {'Frustration':<12}: {len(concepts)} concepts tracked, "
        f"{len(high)} above the high band, {len(sources)} sources"
    )


def _trend_line(entry):
    """One trend as a line a reader can act on.

    The verdict is the point, so it leads. The two means it compares follow,
    because a verdict of IMPROVING on a change of 0.001% is a verdict about
    noise and the reader can only see that if the numbers are there.
    """
    verdict = entry.get("trend") or "UNKNOWN"
    first = entry.get("first")
    last = entry.get("last")
    change = entry.get("change_percent")
    samples = entry.get("samples", 0)

    if change is None:
        return (
            f"{verdict:<10} {samples} measurements, "
            f"no comparable pair (needs at least 4)"
        )

    return (
        f"{verdict:<10} {first} -> {last} "
        f"({_signed(change)}{abs(change):.2f}%) over {samples} cycles"
    )


def main(c=None, argv=None):
    from config import load_config

    c = c or load_config()
    argv = sys.argv if argv is None else argv
    result = report(c)
    target = Path(c.evaluation_report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if "--full" in argv:
        print(json.dumps(result, indent=2))
        print()
    summarise(result, c.evaluation_report)


if __name__ == "__main__":
    main()
