"""Relationships between concepts (Architecture 12).

Concepts are connected by typed edges:

    IS_A  PART_OF  DEPENDS_ON  USED_BY  SIMILAR_TO
    CONTRADICTS  CAUSES  PRECEDES  REQUIRES  IMPLEMENTS
    CO_OCCURS

CO_OCCURS was added under Architecture 53 and is the only type Jupiter
discovers for itself. Architecture 12's original list had no way to say "these
two terms were ranked highly by the same document" without overstating it:
SIMILAR_TO would claim a likeness that term frequency cannot support, and
IS_A or PART_OF would claim a structural relationship Jupiter has no evidence
for. Naming the measurement honestly is better than forcing it into a type
that means something else.

CO_OCCURS is a fact about the corpus, not a claim about meaning. Two terms
co-occur because one source surfaced both, and Jupiter still cannot say why
they appear together, let alone whether they are related. Every other type
above stays available for a future system that has real evidence to assert it.

Architecture 12 is explicit that a relationship is not automatically a fact.
Every relationship therefore carries its own confidence, and it moves on
evidence the same way concept confidence does, instead of being asserted once
and treated as permanently true.

CONTRADICTS is treated specially: it never merges or overwrites. It preserves
both sides, which is what Architecture 13 and 14 require when evidence
conflicts.
"""

import itertools
import time

import concepts as concepts_module

RELATIONSHIPS = "relationships"

IS_A = "IS_A"
PART_OF = "PART_OF"
DEPENDS_ON = "DEPENDS_ON"
USED_BY = "USED_BY"
SIMILAR_TO = "SIMILAR_TO"
CONTRADICTS = "CONTRADICTS"
CAUSES = "CAUSES"
PRECEDES = "PRECEDES"
REQUIRES = "REQUIRES"
IMPLEMENTS = "IMPLEMENTS"
CO_OCCURS = "CO_OCCURS"

TYPES = (
    IS_A,
    PART_OF,
    DEPENDS_ON,
    USED_BY,
    SIMILAR_TO,
    CONTRADICTS,
    CAUSES,
    PRECEDES,
    REQUIRES,
    IMPLEMENTS,
    CO_OCCURS,
)

# Confidence moves a relationship toward the truth when it is re-observed and
# away from it when it is contradicted. These are deliberately gentle, because
# one observation is weak evidence either way.
SUPPORT_GAIN = 0.20
CONTRADICT_LOSS = 0.35

# How similarly two terms must be distributed across sources before Jupiter says
# they are similar. Set high on purpose. Below this, two terms appearing in the
# same few documents is an accident of a small corpus rather than a fact about
# the terms, and recording SIMILAR_TO on that basis would fill the graph with
# confident edges that mean nothing.
SIMILAR_OVERLAP = 0.8

# How many sources must be involved before SIMILAR_TO is considered at all.
# One document containing two terms says the document is short, not that the
# terms are related, and on a corpus of a few documents almost every pair would
# clear a bare overlap threshold.
MIN_SIMILAR_SOURCES = 3

# How many sources must agree on an ordering before PRECEDES is recorded. One
# document is an accident; several documents saying the same order is a
# property of how the material is written.
ORDER_SUPPORT = 2


def _key(source, relation, target):
    return "|".join(
        (
            concepts_module.normalize(source),
            relation,
            concepts_module.normalize(target),
        )
    )


def record(
    source,
    relation,
    target,
    confidence=0.0,
    evidence=0,
    now=None,
):
    """Build a relationship record."""
    if relation not in TYPES:
        raise ValueError(
            f"unknown relationship type {relation!r}; "
            f"expected one of {', '.join(TYPES)}"
        )
    return {
        "from": concepts_module.normalize(source),
        "relation": relation,
        "to": concepts_module.normalize(target),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "evidence": int(evidence),
        "contradictions": 0,
        "first_seen": time.time() if now is None else now,
        "last_verified": time.time() if now is None else now,
    }


# What an edge endpoint names. Architecture 12 joins concepts, but Architecture
# 14 and the assertion layer join sources: a disagreement is between two
# documents, not between two words. Both are legitimate, and telling them apart
# by guessing at the shape of the name was how a sha256 filename became a
# concept with no evidence behind it.
CONCEPT_ENDPOINTS = "concepts"
SOURCE_ENDPOINTS = "sources"


def observe(
    store,
    source,
    relation,
    target,
    supporting=True,
    now=None,
    endpoints=CONCEPT_ENDPOINTS,
):
    """Record one observation of a relationship and return the record.

    A supporting observation raises confidence and evidence. A contradicting
    observation lowers confidence, counts a contradiction, and links both
    concepts through CONTRADICTS so the conflict stays visible instead of
    being resolved by overwriting.

    `endpoints` says what the two names are. The default is that they are
    concepts, which is what every derived relationship is. A caller holding a
    disagreement between two documents passes SOURCE_ENDPOINTS, and the edge is
    then recorded in the relationship collection without either document being
    written into the concept table, because a document is not a concept.
    """
    key = _key(source, relation, target)
    existing = store.get(RELATIONSHIPS, key)
    if existing is None:
        entry = record(
            source,
            relation,
            target,
            confidence=0.0 if supporting else CONTRADICT_LOSS,
            evidence=1 if supporting else 0,
            now=now,
        )
        if not supporting:
            entry["contradictions"] = 1
    else:
        entry = dict(existing)
        value = float(entry.get("confidence", 0.0))
        if supporting:
            value += (1.0 - value) * SUPPORT_GAIN
            entry["evidence"] = int(entry.get("evidence", 0)) + 1
        else:
            value -= value * CONTRADICT_LOSS
            entry["contradictions"] = int(entry.get("contradictions", 0)) + 1
        entry["confidence"] = max(0.0, min(1.0, value))
        entry["last_verified"] = time.time() if now is None else now
    stored = store.put(RELATIONSHIPS, key, entry)
    if not supporting:
        _link_conflict(store, source, target, now=now, endpoints=endpoints)
    _register_on_concepts(store, stored, now=now, endpoints=endpoints)
    return stored


def _link_conflict(store, source, target, now=None, endpoints=CONCEPT_ENDPOINTS):
    """Preserve a contradiction as a CONTRADICTS edge in both directions.

    The edges are registered on the concepts as well, so a conflict is visible
    from either side and not only by traversing relationships. When the two
    names are sources rather than concepts there is no concept to register on,
    and the bidirectional edge in the relationship collection is the whole of
    what the conflict consists of.
    """
    for left, right in (
        (source, target),
        (target, source),
    ):
        key = _key(left, CONTRADICTS, right)
        if store.get(RELATIONSHIPS, key) is None:
            entry = record(left, CONTRADICTS, right, confidence=0.5, now=now)
            store.put(RELATIONSHIPS, key, entry)
            _register_on_concepts(store, entry, now=now, endpoints=endpoints)


def _register_on_concepts(store, entry, now=None, endpoints=CONCEPT_ENDPOINTS):
    """Keep the denormalized edge lists on both concepts in step.

    Concepts are stored separately from relationships, so each concept keeps
    the ids of the edges touching it. That lets a concept be read without
    loading the whole relationship collection.

    Both endpoints are written in one call rather than two. This runs once per
    derived edge, and each put rewrites the whole store, so a derivation over a
    thousand edges was performing two thousand more full rewrites on top of the
    edge writes themselves.

    An endpoint that is not already a concept is not turned into one here. It
    used to be, and that is what filled the concept table with document digests:
    curation names every curated file by the sha256 of its contents, so an edge
    between two sources is an edge between two sixty-eight character names, and
    manufacturing a concept for each one produced a graph in which twenty-two
    per cent of the edges connected shell records that no question could ever
    recall. A concept is made when something is actually learned about a term,
    in concepts.observe. Not here, where the only thing known about the endpoint
    is that an edge mentioned it.
    """
    if endpoints != CONCEPT_ENDPOINTS:
        return
    edge_key = _key(entry["from"], entry["relation"], entry["to"])
    stamp = time.time() if now is None else now
    updates = []
    for endpoint in (entry["from"], entry["to"]):
        concept = store.get(concepts_module.CONCEPTS, endpoint)
        if concept is None:
            continue
        edges = concept.setdefault("relationships", [])
        if edge_key not in edges:
            edges.append(edge_key)
        if entry["relation"] in (DEPENDS_ON, REQUIRES):
            prerequisites = concept.setdefault("prerequisites", [])
            other = entry["to"] if endpoint == entry["from"] else entry["from"]
            if other not in prerequisites:
                prerequisites.append(other)
        if entry["relation"] == CONTRADICTS:
            conflicts = concept.setdefault("conflicting_information", [])
            other = entry["to"] if endpoint == entry["from"] else entry["from"]
            if other not in conflicts:
                conflicts.append(other)
        concept["last_verified"] = stamp
        updates.append((endpoint, concept))
    if updates:
        store.put_many(concepts_module.CONCEPTS, updates)


def record_co_occurrence(store, terms, now=None):
    """Record that one learning source surfaced these terms together.

    This is a measurement rather than an understanding. The terms co-occur
    because a single document ranked them all highly, which is a fact about the
    corpus. It is not a claim that the terms mean the same thing, that one
    depends on the other, or that Jupiter understands why they appear together.

    The pair is canonicalised so `a` with `b` is one edge rather than two
    directions of the same fact. Confidence then grows every time the pair is
    seen together, so terms that co-occur across several sources end up
    connected more strongly than terms seen together once, which is exactly
    the evidence-based movement Architecture 12 asks for.

    Every pair from one source is written inside a single transaction. A
    document surfacing thirty terms is 435 pairs, and each one updated two
    concepts: without the transaction that is 1,305 full rewrites of the store
    to record one measurement, and the loop spent longer persisting co-occurrence
    than learning anything.
    """
    names = sorted({concepts_module.normalize(term) for term in terms})
    recorded = []
    with store.transaction():
        for left, right in itertools.combinations(names, 2):
            recorded.append(observe(store, left, CO_OCCURS, right, now=now))
    return recorded


def derive_graded_relationships(store, now=None):
    """Derive the relationship types the corpus can actually justify.

    Architecture 12 names eleven types. Ten of them had no code path that ever
    created one, which made the enumeration a list of intentions rather than a
    description of the system, and it is the reason the concept graph was nine
    disconnected fragments of pure co-occurrence.

    What follows derives two of them, plus CO_OCCURS which is measured directly
    during learning rather than derived here. The standard is the same for all of
    them: a relationship is recorded only when the corpus contains evidence for
    that specific relationship and not merely for the two concepts existing.
    Nothing here guesses, and a type that cannot be evidenced on this corpus
    stays unrecorded rather than being invented to fill the table.

    SIMILAR_TO is the strongest claim available from term statistics alone. Two
    terms that appear in almost exactly the same documents are describing the
    same thing under different names, and the measurement that supports it is
    the Jaccard overlap of their source sets, so that is what is recorded and
    the score is the confidence.

    PRECEDES is directional, and direction has to come from somewhere or the
    arrow is a decoration. Order within a document is the one direction the
    corpus states: if A is always written before B and they are never written
    the other way round, the corpus has said which comes first, and that is a
    fact about how the material is written rather than a claim about meaning.

    USED_BY and PART_OF were named here for a while with no code behind them. The
    measurements that would support them exist, but the argument each one needed
    was not written down, and a dictionary key that is always empty is a claim
    the code does not make. They are listed in the return value so a caller can
    see they were considered and not derived, rather than being quietly absent.
    """
    edges = all_relationships(store)
    co_pairs = {}
    for entry in edges:
        if entry["relation"] == CO_OCCURS:
            key = tuple(sorted((entry["from"], entry["to"])))
            co_pairs.setdefault(key, []).append(entry)

        # Only the types this function actually derives appear as keys. USED_BY and
        # PART_OF were listed here once and were never appended to, so the returned
        # dictionary claimed four derivations while holding two, and a caller
        # checking which types ran had no way to tell an empty list from a type that
        # was never attempted.
        derived = {"SIMILAR_TO": [], "PRECEDES": []}

    # Edges are collected and appended to `pending` rather than stored as each
    # one is derived, and the whole derivation runs inside one transaction
    # opened by the caller. Every put rewrites the whole store, so a graph
    # holding twelve hundred edges turned one derivation into twelve hundred full
    # rewrites of a multi-megabyte document: minutes of disk traffic for a few
    # hundred kilobytes of new edges, and twelve hundred chances to collide with
    # anything else reading the directory. The records and the atomicity are
    # unchanged; only the number of writes is.
    pending = []

    def flush():
        """Hand the pending edges to whoever owns the write.

        Inside a transaction the edges are already staged and the transaction
        flushes once at the end, so this clears the list and writes nothing.
        Outside one there is nothing to join, so it writes. The two derivation
        loops call it to say they are finished without knowing which case they
        are in.
        """
        if store.in_transaction:
            pending.clear()
            return
        if pending:
            store.put_many(RELATIONSHIPS, list(pending))
            pending.clear()

    # The whole derivation is one transaction. Deriving edges touches each
    # concept from both of its endpoints, and each of those is a full rewrite of
    # a multi-megabyte store; measured, that was 582 writes and 414 seconds
    # inside a three-cycle run. One transaction makes the whole pass one write.
    with store.transaction():
        _derive_into(
            store,
            co_pairs,
            edges,
            derived,
            pending,
            flush,
            now=now,
        )
    return derived


def _derive_into(store, co_pairs, edges, derived, pending, flush, now=None):
    """Derive every relationship type the evidence supports.

    Split out of derive_graded_relationships so the body runs inside one
    transaction opened by the caller, rather than being a function that opens
    its own and cannot be composed with another writer.
    """

    # SIMILAR_TO, from how similarly two terms are distributed across sources.
    # Confidence rises with the overlap and is capped below certainty, because
    # similar distribution is evidence of similar meaning and not proof of it.
    for (left, right), entries in co_pairs.items():
        # The sources each endpoint has, kept apart. Both endpoints of an edge
        # were seen in the same documents by construction, so asking "which
        # documents contain left" by testing membership of the edge's own
        # endpoints answers the same question for both and makes every pair look
        # identical. The endpoint has to be matched to its own side.
        left_sources = set()
        right_sources = set()
        for entry in entries:
            for source in entry.get("sources", []):
                if entry["from"] == left:
                    left_sources.add(source)
                if entry["to"] == right:
                    right_sources.add(source)
        if not left_sources or not right_sources:
            continue
        if len(left_sources | right_sources) < MIN_SIMILAR_SOURCES:
            # One document is an accident of a small corpus, not a fact about
            # the terms, and recording SIMILAR_TO on that basis would fill the
            # graph with confident edges that mean nothing.
            continue
        union = left_sources | right_sources
        overlap = len(left_sources & right_sources) / len(union)
        if overlap >= SIMILAR_OVERLAP:
            edge = record(
                left,
                SIMILAR_TO,
                right,
                confidence=round(min(0.9, overlap), 4),
                now=now,
            )
            pending.append((_key(left, SIMILAR_TO, right), edge))
            _register_on_concepts(store, edge, now=now)
            derived["SIMILAR_TO"].append(edge)

    # PRECEDES, from consistent order within a source. Only a pair whose order
    # never reverses is recorded; a pair seen both ways is evidence that the
    # corpus does not state an order, and is left alone.
    order = {}
    for entry in edges:
        if entry["relation"] != CO_OCCURS:
            continue
        for source, written in (entry.get("orders") or {}).items():
            if len(written) != 2:
                continue
            first, second = written
            # Canonicalised for the lookup only, never for the value. The value
            # is which term was written first, which is the whole point.
            key = tuple(sorted((first, second)))
            order.setdefault(key, {})[source] = (first, second)

    for (left, right), placements in list(order.items()):
        forwards = set()
        backwards = set()
        for source, (first, second) in placements.items():
            if first == left:
                forwards.add(source)
            else:
                backwards.add(source)
        if forwards and backwards:
            # The material states the order both ways. Drawing an arrow either
            # way would be inventing a claim the corpus has declined to make.
            continue
        if len(forwards) < ORDER_SUPPORT and len(backwards) < ORDER_SUPPORT:
            continue
        if forwards:
            edge = record(
                left,
                PRECEDES,
                right,
                confidence=round(min(0.8, 0.4 + 0.1 * len(forwards)), 4),
                now=now,
            )
        else:
            edge = record(
                right,
                PRECEDES,
                left,
                confidence=round(min(0.8, 0.4 + 0.1 * len(backwards)), 4),
                now=now,
            )
        pending.append((_key(edge["from"], PRECEDES, edge["to"]), edge))
        _register_on_concepts(store, edge, now=now)
        derived["PRECEDES"].append(edge)

    flush()
    return derived


def relationship_coverage(store):
    """Which relationship types exist and which are merely named.

    Reported rather than assumed, because "eleven types" and "two types in use"
    are very different sentences and the project's own rule is that a feature
    listed as covered with nothing behind it is worse than an absent one.
    """
    present = {entry["relation"] for entry in all_relationships(store)}
    return {
        "types_defined": list(TYPES),
        "types_present": sorted(present),
        "types_unused": sorted(set(TYPES) - present),
        "note": (
            "a type in types_unused is one the code can name and no run has "
            "ever created. It is listed rather than hidden so the difference "
            "between the vocabulary and the evidence stays visible."
        ),
    }


def record_co_occurrence_from_source(store, terms, source, now=None):
    """Record co-occurrence and remember which source produced it.

    The source is kept on the edge because the relationship types derived from
    it need it: two terms that appear in the same documents as each other are
    similar, and two that always appear in the same order are ordered, and
    neither question can be asked without knowing which documents were involved.

    The sources accumulate rather than being replaced. observe() already folds a
    repeat sighting into a single edge and raises its confidence, so writing the
    source again would overwrite the one that was there and leave a pair that
    appeared in four documents looking like it had appeared in one. Every source
    that ever surfaced the pair is kept, which is the only thing that makes the
    count mean what it says.
    """
    # The order the caller passed the terms in is preserved, because it is the
    # only evidence of which term was written first. Deduplicated but NOT
    # sorted: sorting was how the order got lost, since combinations() over a
    # sorted list always yields the smaller name first and made a document
    # saying "two one" indistinguishable from one saying "one two".
    seen = set()
    names = []
    for term in terms:
        name = concepts_module.normalize(term)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    recorded = []
    for left, right in itertools.combinations(names, 2):
        entry = dict(observe(store, left, CO_OCCURS, right, now=now))
        sources = sorted(set(entry.get("sources", [])) | {str(source)})
        entry["sources"] = sources
        # The order the two terms were actually written in, kept per source.
        # The edge itself is canonicalised so "a with b" is one edge rather than
        # two directions of the same fact, and that canonicalisation is exactly
        # what destroys the order. Without it recorded here, a document saying
        # "one two" and another saying "two one" produce the same edge carrying
        # the same two sources, and the corpus's disagreement about which comes
        # first becomes invisible. PRECEDES is derived from this, so a pair whose
        # order reverses is left unlinked rather than given an arrow the
        # material does not support.
        written = (
            [left, right] if names.index(left) < names.index(right) else [right, left]
        )
        entry.setdefault("orders", {})[str(source)] = written
        # The most recent source is kept as well as the full set, so a reader
        # that only wants "where did this last show up" does not have to know
        # about the accumulated set.
        entry["from_source"] = str(source)
        entry["to_source"] = str(source)
        store.put(RELATIONSHIPS, _key(left, CO_OCCURS, right), entry)
        recorded.append(entry)
    return recorded


def get(store, source, relation, target):
    return store.get(RELATIONSHIPS, _key(source, relation, target))


def all_relationships(store):
    return store.all(RELATIONSHIPS)


def outgoing(store, name, relation=None):
    """Relationships where `name` is the source."""
    return [
        entry
        for entry in all_relationships(store)
        if entry["from"] == concepts_module.normalize(name)
        and (relation is None or entry["relation"] == relation)
    ]


def incoming(store, name, relation=None):
    """Relationships where `name` is the target."""
    return [
        entry
        for entry in all_relationships(store)
        if entry["to"] == concepts_module.normalize(name)
        and (relation is None or entry["relation"] == relation)
    ]


def prerequisites_of(store, name, _seen=None):
    """Concepts that must be understood before `name`, transitively.

    Cycles are possible in a knowledge graph, so traversal carries a visited
    set rather than trusting the data to be acyclic.
    """
    seen = set() if _seen is None else _seen
    key = concepts_module.normalize(name)
    if key in seen:
        return []
    seen.add(key)
    found = []
    for entry in outgoing(store, key):
        if entry["relation"] in (DEPENDS_ON, REQUIRES, PRECEDES):
            target = entry["to"]
            if target not in found:
                found.append(target)
            found.extend(prerequisites_of(store, target, seen))
    return found


def conflicts_of(store, name):
    """Concepts this one is recorded as contradicting."""
    return [entry["to"] for entry in outgoing(store, name, CONTRADICTS)]


def conflicts_map(store, relation=CONTRADICTS):
    """Every concept's conflicts, read from the edge collection once.

    conflicts_of() answers the same question for one name, and answering it
    re-reads and re-parses the whole relationship collection each time. Frustration
    asks it about every concept it holds, so on a store with 2,411 concepts and
    10,468 edges that was 2,411 full parses of a multi-megabyte document to answer
    a question the collection already answered.

    This reads the edges once and groups them by source, so a caller that needs
    every concept's conflicts pays for one pass rather than one per concept. The
    result is the same list conflicts_of() would have returned for each name, and
    an absent name maps to an empty list rather than being missing.
    """
    grouped = {}
    for entry in store.collection(RELATIONSHIPS).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("relation") != relation:
            continue
        grouped.setdefault(entry["from"], []).append(entry["to"])
    return grouped


def census(store):
    """Relationship counts by type, for observability."""
    counts = {relation: 0 for relation in TYPES}
    for entry in all_relationships(store):
        counts[entry["relation"]] = counts.get(entry["relation"], 0) + 1
    return {
        "total": sum(counts.values()),
        "by_type": counts,
        "low_confidence": sum(
            1
            for entry in all_relationships(store)
            if float(entry.get("confidence", 0.0)) < 0.4
        ),
    }
