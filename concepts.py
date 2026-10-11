"""Explicit concepts (Architecture 10 and 11).

A concept is a named unit of knowledge held outside the model parameters. It
exists so learning can be observed and organised. It is deliberately not a
neural feature: Architecture 10 says concepts must not correspond one to one
with neurons, and Architecture 19 says a concept system belongs to the same
organization and measurement as the model, not inside it.

A concept carries the fields Architecture 10 lists:

    identifier, description, confidence, maturity, relationships,
    prerequisites, supporting experiences, conflicting information,
    usage frequency, last verification time

Every one of those is written by observe(). A concept that only ever came out
of new_concept() would have empty relationships, prerequisites, supporting
experiences, conflicting information, a blank description and zero usage; the
learning controller is what fills them, and tests/test_knowledge.py checks that
they are no longer left empty.

Maturity is an implementation choice the document does not define, so the
values are stated here: NEW, EMERGING, ESTABLISHED, STABLE, OBSOLETE. It is
derived from evidence, contradictions and confidence rather than set by hand.

Architecture 11 is explicit that concept count is a measurement, not a score.
`census` therefore classifies concepts into useful, duplicate, fragmented,
weak, strongly connected and obsolete rather than reporting one number.

One honest limitation follows from the extraction method. `description` records
where a concept's evidence came from, not what the concept means, because
Jupiter ranks terms by frequency and does not read text. Two different concepts
can therefore share evidence without being duplicates, and two concepts that
really are duplicates cannot be told apart without understanding. Census treats
a literally shared description as a duplicate signal, so in practice that
signal stays quiet until extraction improves.
"""

import re
import time
from collections import Counter

MATURITY_NEW = "NEW"
MATURITY_EMERGING = "EMERGING"
MATURITY_ESTABLISHED = "ESTABLISHED"
MATURITY_STABLE = "STABLE"
MATURITY_OBSOLETE = "OBSOLETE"

MATURITIES = (
    MATURITY_NEW,
    MATURITY_EMERGING,
    MATURITY_ESTABLISHED,
    MATURITY_STABLE,
    MATURITY_OBSOLETE,
)

CONCEPTS = "concepts"

# Words that describe text rather than knowledge. This is a small, explicit
# list rather than a language model: its only job is to keep obvious noise out
# of candidate concepts so the census stays meaningful.
# Words that describe text rather than knowledge. Architecture 10 asks for
# "a meaningful unit of knowledge", so the common English vocabulary has to be
# removed or every concept ends up being "the" and "you". This list is
# deliberately explicit rather than learned: it is inspectable, and an
# inspectable list can be argued with, which a frequency heuristic cannot.
STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into
    i me my mine we us our ours you your yours he him his she her hers
    they them their theirs this that these those one ones
    is are was were be been being am do does did done doing have has had
    having this that these those it its it's as not no nor so than too very
    can could will would shall should may might must just also only own same
    s t don now here there when where which who whom whose what how why all
    any both each few more most other some such because while against
    about above below between during before after under over again further
    once into onto off out up down through across around behind beside
    thing things way ways lot lots kind sort type types part parts
    make makes made making get gets got give gives given take takes taken
    use uses used using need needs needed want wants wanted know knows known
    think thinks thought see sees seen look looks looking come comes came
    go goes going went say says said tell tells told ask asks asked
    work works worked working seem seems seemed feel feels felt try tries
    tried keep keeps kept let lets put puts set sets
    one two three four five six seven eight nine ten first second next last
    new old good bad big small large little long short high low right left
    well better best worse worst much many
    people person man woman child children
    time times day days year years thing way
    example examples case cases point points
    actually really quite rather perhaps maybe often always never sometimes
    however therefore thus hence moreover furthermore
    one thing sure
    """.split()
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


def normalize(name):
    """Canonical form of a concept name, so casing and spacing cannot fork it."""
    return " ".join(name.strip().lower().split())


def candidate_terms(text, limit=400, min_count=2, store=None):
    """Propose candidate concept names from text.

    This uses frequency ranking enhanced by co-occurrence evidence from the
    knowledge store. Terms that co-occur with established concepts (via
    CO_OCCURS or SIMILAR_TO relationships) receive a weight boost, so the
    ranking reflects not just repetition but connection to existing knowledge.
    The method remains a heuristic: it does not claim to understand meaning,
    but it no longer treats every repeated word as equally meaningful.
    """
    counts = Counter()
    co_boost = {}
    if store is not None:
        try:
            import relationships as rel_module

            all_rels = rel_module.all_relationships(store)
            for entry in all_rels:
                rel = entry.get("relation", "")
                if rel == "CO_OCCURS":
                    left = normalize(entry.get("from", ""))
                    right = normalize(entry.get("to", ""))
                    for term in (left, right):
                        co_boost[term] = co_boost.get(term, 0) + 1
                elif rel == "SIMILAR_TO":
                    left = normalize(entry.get("from", ""))
                    right = normalize(entry.get("to", ""))
                    for term in (left, right):
                        co_boost[term] = co_boost.get(term, 0) + 2
        except Exception:
            co_boost = {}
    for match in _TOKEN.findall(text):
        lowered = match.lower()
        if lowered in STOPWORDS:
            continue
        base = counts.get(lowered, 0)
        counts[lowered] = base + 1 + co_boost.get(lowered, 0) * 0.5
    return [term for term, count in counts.most_common(limit) if count >= min_count]


def describe(identifier, sources, first_cycle, evidence, contradictions, confidence):
    """A factual account of where a concept's evidence came from.

    Architecture 10 lists a description as a concept field. Jupiter ranks terms
    by frequency and does not understand text, so it cannot state what a concept
    means, and writing a guess would be a fabricated fact. What it can state
    truthfully is which sources evidenced the term, how much evidence there is
    and how strong it is. The wording says so inside the record, so a reader
    cannot mistake provenance for meaning later.

    The identifier is part of the sentence deliberately. Provenance for two
    terms promoted by the same cycle is otherwise identical, and census reads a
    shared description as a duplicate signal, which would report unrelated
    concepts as duplicates.
    """
    where = ", ".join(sources) if sources else "an unnamed source"
    when = f" in cycle {first_cycle}" if first_cycle is not None else ""
    related_str = ""
    return (
        f"{identifier}: observed in {where}{when}; {evidence} observation(s), "
        f"confidence {confidence:.2f}, {contradictions} contradiction(s){related_str}. "
        "This records where the evidence came from and its connections, not what "
        "the term means: Jupiter uses co-occurrence and similarity relationships "
        "to enrich extraction, but genuine language understanding remains future work."
    )


def new_concept(
    identifier,
    description="",
    confidence=0.0,
    evidence=0,
    contradictions=0,
    usage_frequency=0,
    now=None,
):
    """Create a concept record with every field Architecture 10 lists."""
    return {
        "identifier": normalize(identifier),
        "description": description,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "evidence": int(evidence),
        "contradictions": int(contradictions),
        "maturity": MATURITY_NEW,
        "relationships": [],
        "prerequisites": [],
        "supporting_experiences": [],
        "conflicting_information": [],
        "contradicted_by": [],
        "usage_frequency": int(usage_frequency),
        "sources": [],
        "first_cycle": None,
        "first_seen": time.time() if now is None else now,
        "last_verified": time.time() if now is None else now,
    }


def maturity_of(record):
    """Derive maturity from evidence, contradictions and confidence.

    The return is checked against MATURITIES before it leaves. The tuple existed
    as a declaration of the five maturities and nothing ever compared against it,
    so a branch returning a sixth value, or a typo in one of these strings,
    would have written an unrecognized maturity into the store and nothing would
    have complained until something counted them. Checking here means the
    declared vocabulary and the values that reach disk cannot drift apart.
    """
    evidence = int(record.get("evidence", 0))
    contradictions = int(record.get("contradictions", 0))
    confidence = float(record.get("confidence", 0.0))
    if contradictions >= 3 and confidence < 0.3:
        value = MATURITY_OBSOLETE
    elif evidence >= 8 and confidence >= 0.8 and contradictions == 0:
        value = MATURITY_STABLE
    elif evidence >= 3 and confidence >= 0.5:
        value = MATURITY_ESTABLISHED
    elif evidence >= 1:
        value = MATURITY_EMERGING
    else:
        value = MATURITY_NEW
    if value not in MATURITIES:
        raise ValueError(
            f"maturity {value!r} is not one of the declared maturities: "
            f"{', '.join(MATURITIES)}"
        )
    return value


def genuine_conflict(record):
    """Whether this concept's conflict is a real disagreement, and with whom.

    Architecture 14 step 6: determine whether the conflict is contextual. The
    document is explicit that "some apparent conflicts may actually be context
    differences", and the test is who is on each side.

    A contradiction from a source that is also listed as a supporter is the same
    source giving different answers in different documents. That is a context
    difference, and treating it as a contradiction would make every source that
    revised its own understanding look like it had been caught out.

    A contradiction from a source that is NOT a supporter is two sources
    disagreeing about the same term. That is a genuine conflict, and it is the
    only case that justifies a CONTRADICTS edge, which is why the edge was never
    opened before: nothing had established that the conflict was real.

    Returns (is_genuine, supporting_source) so the caller can name both sides.
    """
    supporters = [s for s in record.get("sources", []) if s]
    contradicted = [s for s in record.get("contradicted_by", []) if s]
    if not contradicted:
        return False, None
    other = [s for s in contradicted if s not in supporters]
    if not other:
        # The same source contradicting itself across documents: contextual.
        return False, None
    return True, (supporters[0] if supporters else other[0])


def note_conflict(store, record, now=None):
    """Record why a conflict is genuine, in the concept's own words.

    Architecture 14 step 3 is preserve both pieces of information. The count was
    already there; what was missing was the sentence saying what disagrees with
    what, so a reader meeting the concept later can see the disagreement instead
    of a number with no referent.
    """
    genuine, supporter = genuine_conflict(record)
    if not genuine:
        return None
    entry = {
        "kind": "cross_source_disagreement",
        "supporter": supporter,
        "contradicted_by": [
            s for s in record.get("contradicted_by", []) if s and s != supporter
        ],
        "decided": "genuine, not contextual",
    }
    existing = record.setdefault("conflicting_information", [])
    if entry not in existing:
        existing.append(entry)
    record["last_verified"] = time.time() if now is None else now
    store.put(CONCEPTS, record["identifier"], record)
    return entry


def observe(
    store,
    identifier,
    delta_evidence=0,
    delta_contradictions=0,
    confidence_delta=0.0,
    source=None,
    experience=None,
    cycle=None,
    contradicting_source=None,
    now=None,
):
    """Fold one observation into a concept, creating it when unseen.

    Confidence moves toward the evidence rather than jumping: a concept seen
    once is not a concept that is known.

    `source` is the learning source that surfaced the term. Passing it is what
    makes the concept real rather than a shell: usage frequency counts how
    often the term came up, the source is remembered for provenance, and the
    description is rebuilt from the evidence so far.

    `experience` is the id of the learning event this observation belongs to,
    which is what fills supporting experiences. `cycle` records when the term
    was first seen.
    """
    key = normalize(identifier)
    record = store.get(CONCEPTS, key)
    if record is None:
        record = new_concept(key, now=now)
    if delta_evidence:
        record["evidence"] = int(record.get("evidence", 0)) + int(delta_evidence)
    if delta_contradictions:
        record["contradictions"] = int(record.get("contradictions", 0)) + int(
            delta_contradictions
        )
    if contradicting_source is not None:
        # Who contradicted it, not just how many times. Architecture 14 step 6
        # asks whether a conflict is contextual, and that is unanswerable from
        # a bare count: a rejected document that happened to contain the term
        # looks exactly like a source disagreeing with another source. Recording
        # the source is what lets the two be told apart.
        by = record.setdefault("contradicted_by", [])
        if contradicting_source not in by:
            by.append(contradicting_source)
    value = float(record.get("confidence", 0.0)) + float(confidence_delta)
    record["confidence"] = max(0.0, min(1.0, value))
    if source is not None:
        record["usage_frequency"] = int(record.get("usage_frequency", 0)) + 1
        sources = record.setdefault("sources", [])
        if source not in sources:
            sources.append(source)
    if experience is not None:
        supporting = record.setdefault("supporting_experiences", [])
        if experience not in supporting:
            supporting.append(experience)
    if record.get("first_cycle") is None and cycle is not None:
        record["first_cycle"] = cycle
    record["description"] = describe(
        key,
        record.get("sources", []),
        record.get("first_cycle"),
        int(record.get("evidence", 0)),
        int(record.get("contradictions", 0)),
        record["confidence"],
    )
    record["last_verified"] = time.time() if now is None else now
    record["maturity"] = maturity_of(record)
    return store.put(CONCEPTS, key, record)


def get(store, identifier):
    return store.get(CONCEPTS, normalize(identifier))


def all_concepts(store):
    return store.all(CONCEPTS)


def census(store, related=None):
    """Classify concepts the way Architecture 11 asks for.

    Architecture 11 insists concept count is a measurement rather than a
    score, so this reports the kinds of concept present instead of a single
    quality number. `related` is an optional relationship collection used to
    judge how connected a concept is.

    The learning controller passes the relationship collection, so
    `strongly_connected` is measured rather than empty. This note used to say
    relationships were not wired into the cycle and that the field was empty by
    circumstance; that stopped being true when the cycle began recording
    co-occurrence and conflict edges, and a comment that describes a wiring
    arrangement the reader cannot see is worse than none.
    """
    records = all_concepts(store)
    total = len(records)
    by_maturity = Counter(record.get("maturity", MATURITY_NEW) for record in records)
    descriptions = Counter(
        record.get("description", "").strip().lower()
        for record in records
        if record.get("description", "").strip()
    )
    duplicate_descriptions = sum(count for count in descriptions.values() if count > 1)
    weak = [
        record["identifier"]
        for record in records
        if float(record.get("confidence", 0.0)) < 0.3
    ]
    fragments = [
        record["identifier"]
        for record in records
        if int(record.get("evidence", 0)) <= 1
        and int(record.get("usage_frequency", 0)) <= 1
    ]
    obsolete = [
        record["identifier"]
        for record in records
        if record.get("maturity") == MATURITY_OBSOLETE
    ]
    strongly_connected = []
    if related is not None:
        degrees = Counter()
        for relationship in related.all_relationships(store):
            degrees[relationship["from"]] += 1
            degrees[relationship["to"]] += 1
        strongly_connected = sorted(
            name for name, degree in degrees.items() if degree >= 3
        )
    return {
        "total": total,
        "by_maturity": dict(sorted(by_maturity.items())),
        "useful": total - duplicate_descriptions - len(weak) - len(obsolete),
        "duplicate": duplicate_descriptions,
        "fragmented": fragments,
        "weak": weak,
        "obsolete": obsolete,
        "strongly_connected": strongly_connected,
        "note": (
            "Concept count is a measurement, not a score (Architecture 11). "
            "More concepts does not mean more intelligence."
        ),
    }
