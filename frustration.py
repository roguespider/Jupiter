"""Learning frustration (Architecture 21).

Frustration is not an emotion here. It is a measurement: how often the same
material defeated a bounded update, and what that implies about the
representation rather than about the knowledge.

Architecture 21 gives the shape of it directly:

    Concept: recursive ownership pattern
    Attempts: 27
    Successful generalization: 4
    Repeated errors: 23
    Loss improvement: minimal
    Frustration Index: HIGH

Every one of those inputs is something the controller already records, so this
needs no new data collection. Attempts and outcomes come from the confidence
record, which counts how many cycles judged a source, how many were accepted
and how many regressed. Loss improvement comes from the experiment records.
Concepts carry the same counts directly.

The index is deliberately simple and deliberately explained, because a
frustration score that cannot be reconstructed from its inputs is worse than
none. It answers one question: what fraction of the attempts on this material
failed, discounted by how little the successful ones gained.

What the number is for is Architecture 21's last line, which is the reason it
matters: repeated difficulty feeds architectural analysis. It distinguishes
material Jupiter does not know from material Jupiter cannot represent, and
only the second kind is a reason to expand. A source that keeps failing because
the corpus does not contain the answer needs data, not neurons.
"""

import json
import statistics
from pathlib import Path

import concepts as concepts_module
import experiment as experiment_module
import relationships as relationships_module
import store as store_module

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"

# Bands chosen so that a source which has never been judged is not reported as
# frustrated, and one that mostly succeeds is not reported as struggling.
MEDIUM_AT = 0.25
HIGH_AT = 0.60

CAUSES = (
    "insufficient data",
    "poor representation",
    "insufficient model capacity",
    "inadequate context",
    "conflicting knowledge",
    "tokenizer limitations",
    "inappropriate architecture",
    "not enough evidence to say",
)

# The causes that mean Jupiter is being asked to do something it cannot, as
# opposed to not having been given enough to work with. Architecture 21 asks for
# exactly this separation, because the first group is a reason to fetch or write
# more material and the second is a reason to change the model, and treating
# them the same either starves the corpus or grows the model for no reason.
#
# Two of these are still named and never produced, and saying so is the honest
# state rather than inventing a measurement to fill the table. Nothing counts how
# often the tokenizer is the limit or whether the architecture is the wrong
# shape for the material, and an expansion proposal has to name a cause it can
# evidence. "Insufficient data" is no longer in this list: a source curation
# refused for being too short, duplicated or too large to read is evidence of
# exactly that, and it is now produced.
UNEVIDENCED_CAUSES = (
    "tokenizer limitations",
    "inappropriate architecture",
)

# The cause returned when a source has never been judged. It is in CAUSES and
# was not: the report advertised causes the system could not emit, and omitted
# one it emitted every time a new source was seen.
NO_EVIDENCE = "not enough evidence to say"


def band(index):
    """Name the band a score falls in."""
    if index >= HIGH_AT:
        return HIGH
    if index >= MEDIUM_AT:
        return MEDIUM
    return LOW


def index_for(attempts, successes, errors, mean_improvement):
    """The frustration index for one set of counts.

    error_rate is the fraction of attempts that failed. The gain factor
    discounts it by how much the successful attempts actually bought, so a
    source that fails often but improves a lot when it works is not treated the
    same as one that fails often and improves nothing. Both inputs are
    reported alongside the result so the score can be checked by hand.
    """
    if attempts <= 0:
        return 0.0
    error_rate = min(1.0, max(0.0, errors / attempts))
    gain = 1.0 / (1.0 + max(0.0, mean_improvement))
    return round(error_rate * gain, 4)


def improvement_by_source(c, limit=500):
    """Mean holdout improvement per source, from the experiment records."""
    totals = {}
    for record in experiment_module.read_experiments(c.experiment_log, limit=limit):
        source = str(record.get("input", ""))
        before = record.get("before_metrics", {}).get("holdout_loss")
        after = record.get("after_metrics", {}).get("holdout_loss")
        if before is None or after is None:
            continue
        totals.setdefault(source, []).append(before - after)
    return {
        source: round(statistics.fmean(values), 6)
        for source, values in totals.items()
        if values
    }


def from_counts(attempts, successes, errors, improvement=0.0, label=""):
    """Frustration from raw counts, without a confidence state.

    `improvement` is a single observed gain when called live during a cycle, and
    a mean over recorded experiments when called from report(). The key is named
    for the quantity rather than for the statistic so the two are not confused.
    """
    index = index_for(attempts, successes, errors, improvement)
    return {
        "label": label,
        "attempts": attempts,
        "successes": successes,
        "errors": errors,
        "improvement": round(float(improvement), 6),
        "index": index,
        "band": band(index),
    }


def for_source(state, source, improvement=0.0):
    """Frustration for one learning source, from its confidence record."""
    record = state.get(str(source)) if isinstance(state, dict) else None
    if not isinstance(record, dict):
        return {
            "source": str(source),
            "attempts": 0,
            "successes": 0,
            "errors": 0,
            "improvement": round(float(improvement), 6),
            "index": 0.0,
            "band": LOW,
            "note": "this source has not been judged yet",
        }
    measured = from_counts(
        int(record.get("observed", 0)),
        int(record.get("accepted", 0)),
        int(record.get("contradictions", 0)),
        improvement,
        label=str(source),
    )
    measured["source"] = str(source)
    return measured


def for_concept(record):
    """Frustration for one concept, from the counts it already carries.

    The discount is the concept's own confidence rather than an invented
    stand-in for missing loss data, so the score is reconstructible from the
    record: the fraction of observations that were contradictions, discounted by
    how little the concept is currently trusted.

    Note what this deliberately does not capture. A concept seen many times
    with no contradictions and low confidence is not frustrated, it is simply
    not learned yet, and that is a different signal which the concept's own
    confidence already reports. Architecture 21 is about repeated difficulty,
    not about being new. Conflating the two would send the expansion logic
    after capacity when the answer is more data.
    """
    evidence = int(record.get("evidence", 0))
    contradictions = int(record.get("contradictions", 0))
    attempts = evidence + contradictions
    confidence = float(record.get("confidence", 0.0))
    index = index_for(attempts, evidence, contradictions, 0.0)
    index = round(index * (1.0 - min(1.0, max(0.0, confidence))), 4)
    return {
        "concept": record.get("identifier"),
        "attempts": attempts,
        "successes": evidence,
        "errors": contradictions,
        "confidence": confidence,
        "index": index,
        "band": band(index),
    }


def likely_cause(measurement):
    """What repeated difficulty most plausibly means (Architecture 21).

    The order is deliberate. It starts from the cheapest explanations, because
    expanding a model to compensate for missing data is the most expensive way
    to be wrong, and Architecture 23 says expansion must be justified rather
    than automatic. The list is a ranked hypothesis, not a diagnosis: Jupiter
    cannot tell insufficient data from a bad representation, and saying so
    would overstate what a count can show.

    A measurement may carry `starved`, set when curation refused the material
    before it was ever learned. That is direct evidence of insufficient data and
    it outranks every count-based hypothesis, because a source that never made
    it into the corpus has not been attempted at all and no error rate describes
    it.
    """
    if measurement.get("starved"):
        return "insufficient data"
    attempts = measurement.get("attempts", 0)
    errors = measurement.get("errors", 0)
    if attempts == 0:
        return NO_EVIDENCE
    if errors == 0:
        return "conflicting knowledge" if attempts < 2 else "inadequate context"
    rate = errors / attempts
    if rate < 0.4:
        return "conflicting knowledge"
    if rate < 0.7:
        return "poor representation"
    return "insufficient model capacity"


def starved_sources(c):
    """Sources curation refused for want of usable material, by name.

    This is the evidence for "insufficient data", which was one of the three
    causes the system declared and could never produce. It could: the curation
    report already says which files were refused as too short, as duplicates or
    as too large to read, and a source that never made it into the corpus is not
    a source that was hard to learn, it is a source there was not enough of.

    Returning the names rather than a count is what makes the cause attributable.
    A count of refused files says the corpus had a problem; naming them says
    which document to go and write more of.
    """
    import json
    from pathlib import Path

    report_path = Path(getattr(c, "curation_report", "data/curation_report.json"))
    if not report_path.exists():
        return []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(report, dict):
        return []
    refused = []
    for entry in report.get("oversized_files", []) or []:
        if isinstance(entry, dict) and entry.get("source"):
            refused.append(Path(str(entry["source"])).name)
    for entry in report.get("unreadable_files", []) or []:
        if isinstance(entry, dict) and entry.get("source"):
            refused.append(Path(str(entry["source"])).name)
    return sorted(set(refused))


def for_sources(c, state=None):
    """The per-source frustration rows, each with its likely cause.

    This is the half of report() the learning controller needs. report() also
    walks every concept, which is the expensive part and which nothing in the
    loop acts on, so the controller asks for sources only.
    """
    import confidence as confidence_module

    if state is None:
        state = confidence_module.load(c)
    if not isinstance(state, dict):
        return []
    improvements = improvement_by_source(c)
    starved = set(starved_sources(c))
    rows = [
        for_source(state, source, improvements.get(source, 0.0)) for source in state
    ]
    for row in rows:
        row["improvement_basis"] = "mean over recorded experiments"
        row["starved"] = Path(row["source"]).name in starved
        row["likely_cause"] = likely_cause(row)
    rows.sort(key=lambda item: (-item["index"], item["source"]))
    return rows


def report(c, knowledge=None, state=None):
    """Frustration for every judged source and every concept."""
    if state is None:
        state = {}
    improvements = improvement_by_source(c)
    sources = [
        for_source(state, source, improvements.get(source, 0.0)) for source in state
    ]
    starved = set(starved_sources(c))
    for source in sources:
        source["improvement_basis"] = "mean over recorded experiments"
        source["starved"] = Path(source["source"]).name in starved
    sources.sort(key=lambda item: (-item["index"], item["source"]))
    if knowledge is None:
        path = Path(c.knowledge_store)
        knowledge = store_module.Store(path) if path.exists() else None
    concept_rows = []
    if knowledge is not None:
        for record in concepts_module.all_concepts(knowledge):
            concept_rows.append(for_concept(record))
    concept_rows.sort(key=lambda item: (-item["index"], item["concept"]))
    # Read once for the whole report. conflicts_of() re-reads the relationship
    # collection on every call, and asking it about every concept held meant
    # re-parsing the whole store once per concept: 150 seconds of a 166 second
    # evaluation, spent re-reading a document that had not changed. The map is
    # the same answers grouped by source.
    conflicts = (
        relationships_module.conflicts_map(knowledge) if knowledge is not None else {}
    )
    for row in sources + concept_rows:
        row["likely_cause"] = likely_cause(row)
        # A concept another source has already contradicted carries direct
        # evidence of conflicting knowledge, which is a stronger signal than
        # the error-rate heuristic produces. It is attached to the row here
        # because a conflict that is recorded and never read back is the same
        # as a conflict that was never recorded.
        row["contradicted_by"] = (
            conflicts.get(row["concept"], [])
            if knowledge is not None and row in concept_rows
            else []
        )
    return {
        "sources": sources,
        "concepts": concept_rows,
        "high_frustration": [
            row for row in sources + concept_rows if row["band"] == HIGH
        ],
        "conflicts": {
            "concepts_contradicted": sum(
                1 for row in concept_rows if row.get("contradicted_by")
            ),
            "note": (
                "a concept listed here is one that another source has been "
                "recorded as contradicting. It is direct evidence of "
                "conflicting knowledge rather than an inference from how often "
                "a bounded update failed."
            ),
        },
        "note": (
            "Frustration is a measurement, not a diagnosis. It says how often "
            "material defeated a bounded update, which Architecture 21 uses to "
            "separate not knowing something from being unable to represent it. "
            "Only the second is a reason to expand."
        ),
        "causes_considered": list(CAUSES),
        # Reported because a list of causes is a list of intentions until
        # something says which of them the system can actually produce. Three
        # are named in the document and cannot be evidenced by any measurement
        # Jupiter takes, and saying so is the honest state rather than letting
        # the list imply a diagnostic power that does not exist.
        "causes_unevidenced": list(UNEVIDENCED_CAUSES),
        "causes_produced": sorted(
            {
                row.get("likely_cause")
                for row in sources + concept_rows
                if row.get("likely_cause")
            }
        ),
    }


def main():
    import confidence as confidence_module

    from config import load_config

    c = load_config()
    print(json.dumps(report(c, state=confidence_module.load(c)), indent=2))


if __name__ == "__main__":
    main()
