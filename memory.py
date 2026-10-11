"""Memory classes and the memory-versus-weights decision
(Architecture 15 and 16).

Architecture 15 names four memory classes:

    working     information relevant to the active task
    long-term   stable knowledge retained for later
    experience  records of previous learning events and outcomes
    external    information held outside the weights and retrieved when needed

Architecture 16 asks for a mechanism that decides where a piece of
information belongs. The decision is made from the factors Architecture 16
lists, and the policy is deliberately biased: the document says external
knowledge should not automatically become permanent internal knowledge, and
Architecture 21 lists insufficient data and poor representation as reasons
repeated learning may fail. So the default answer is "keep it external" and
moving something into the weights has to be argued for.

The decision is recorded with its reasons, because an unexplained routing
decision cannot be reviewed later.
"""

import time

WORKING = "working_memory"
LONG_TERM = "long_term_knowledge"
EXPERIENCE = "experience_memory"
EXTERNAL = "external_memory"

CLASSES = (WORKING, LONG_TERM, EXPERIENCE, EXTERNAL)

# Architecture 16 destinations.
DEST_EXTERNAL = "external"
DEST_WEIGHTS = "weights"
DEST_CONCEPT = "concept"
DEST_RELATIONSHIP = "relationship"
DEST_EXPERIENCE = "experience"
DEST_MULTIPLE = "multiple"

DESTINATIONS = (
    DEST_EXTERNAL,
    DEST_WEIGHTS,
    DEST_CONCEPT,
    DEST_RELATIONSHIP,
    DEST_EXPERIENCE,
    DEST_MULTIPLE,
)

# Reference counts for the Architecture 16 factors that are measured rather
# than stated. They are here, named, and arguable on purpose: "important" has
# to mean something concrete for the number to mean anything, and a reader who
# disagrees with 20 concepts or 5 retrievals can change it here rather than
# finding out later that the policy quietly assumed otherwise.
CONCEPT_REFERENCE = 20.0
RETRIEVAL_REFERENCE = 5.0
RETRIEVAL_SECONDS_REFERENCE = 0.05
LEARNING_SECONDS_REFERENCE = 60.0
AGE_REFERENCE = 10.0


def _record(collection, key, payload, now=None):
    entry = {"id": str(key), "recorded": time.time() if now is None else now}
    entry.update(payload)
    return entry


def remember_working(store, key, content, now=None):
    """Hold something for the current task only."""
    return store.put(
        WORKING,
        key,
        _record(WORKING, key, {"content": content, "scope": "task"}, now=now),
    )


def remember_long_term(store, key, content, confidence=0.0, now=None):
    """Retain stable knowledge for later use."""
    return store.put(
        LONG_TERM,
        key,
        _record(
            LONG_TERM,
            key,
            {
                "content": content,
                "confidence": max(0.0, min(1.0, float(confidence))),
                "scope": "durable",
            },
            now=now,
        ),
    )


def remember_external(store, key, content, origin, now=None):
    """Hold retrievable information that is not yet Jupiter's own knowledge.

    Architecture 28 requires external information not to become permanent
    internal knowledge automatically, so an external memory always records
    where it came from.
    """
    return store.put(
        EXTERNAL,
        key,
        _record(
            EXTERNAL,
            key,
            {"content": content, "origin": str(origin), "scope": "retrievable"},
            now=now,
        ),
    )


def record_experience(store, key, payload, now=None):
    """Record what happened during a learning event (Architecture 46)."""
    return store.put(
        EXPERIENCE,
        key,
        _record(EXPERIENCE, key, dict(payload, scope="experience"), now=now),
    )


def working(store):
    return store.all(WORKING)


def long_term(store):
    return store.all(LONG_TERM)


def experiences(store, limit=None):
    """Experience records, oldest first, optionally the most recent N."""
    records = store.all(EXPERIENCE)
    return records[-limit:] if limit else records


def external(store):
    return store.all(EXTERNAL)


def clear_working(store):
    """Working memory is task-scoped, so ending a task clears it."""
    return store.clear(WORKING)




def census(store):
    """Counts per memory class, for the observability surface."""
    counts = {name: store.counts().get(name, 0) for name in CLASSES}
    # Verification: every memory class must have a non-negative count, and the
    # total must equal the sum of individual counts. A negative count or a
    # mismatch means the store's internal accounting is broken, which is the
    # same kind of silent failure the architecture asks the review step to catch.
    for name, count in counts.items():
        if count < 0:
            raise ValueError(f"memory class {name!r} has negative count {count}")
    total = sum(counts.values())
    store_total = store.counts().get("_total", total)
    if total != store_total:
        raise ValueError(
            f"memory census total ({total}) does not match store total "
            f"({store_total}); a record was lost or duplicated"
        )
    return counts


# --- Architecture 16: where does this information belong? ---


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def route_for_source(
    observed=0,
    accepted=0,
    contradictions=0,
    confidence=0.0,
    improvement=0.0,
    regression_gain=0.0,
    learning_cost=0.0,
    retrieval_cost=0.0,
    concepts=0,
    retrievals=0,
    age_cycles=0,
):
    """Decide where what a learning source taught belongs (Architecture 16).

    Architecture 16 names nine factors. An earlier version of this filled only
    three and passed the rest as zero, on the grounds that they were not
    measurable. That turned out to be wrong: every one of them can be observed
    from something the system already records, and each mapping below says
    exactly what was measured.

        frequency          cycles the source was accepted
        confidence         the confidence record's value
        contradiction_risk contradictions over attempts
        generalization     holdout gain on the reserved document, which is
                           exactly "did learning this help anything else"
        learning_cost      seconds the source actually took to learn
        retrieval_cost     seconds a real retrieval against it actually took
        importance         concepts it contributed and how often it is
                           retrieved, since importance is how much of the
                           system's attention the material attracts
        usefulness         how often it has been retrieved, which is the only
                           honest proxy for useful: not a claim, a count
        permanence         cycles it has been retained without regression

    Three of the nine are scaled against a reference count rather than being
    raw, because decide_destination expects a 0 to 1 factor and a raw count of
    3000 concepts is not comparable to a confidence of 0.8. The reference
    values are stated as constants below so the scaling can be argued with
    rather than discovered.
    """
    contradiction_risk = 0.0
    if observed > 0:
        contradiction_risk = min(1.0, contradictions / observed)
    # Generalization is what a source has actually demonstrated rather than what
    # it might: a large holdout gain on the reserved document means the learning
    # transferred, and a small one means it did not, even though the source was
    # accepted. Both halves are kept whole rather than averaged, so the
    # reserved-document measurement can still move the decision on its own. A
    # small gain stays a real measurement that does not yet count, which is a
    # different thing from not measuring it.
    generalization = max(_clamp01(regression_gain), _clamp01(improvement))
    return decide_destination(
        frequency=accepted,
        importance=0.5 * _clamp01(concepts / CONCEPT_REFERENCE)
        + 0.5 * _clamp01(retrievals / RETRIEVAL_REFERENCE),
        confidence=confidence,
        generalization=generalization,
        retrieval_cost=_clamp01(retrieval_cost / RETRIEVAL_SECONDS_REFERENCE),
        learning_cost=_clamp01(learning_cost / LEARNING_SECONDS_REFERENCE),
        permanence=_clamp01(age_cycles / AGE_REFERENCE),
        usefulness=_clamp01(retrievals / RETRIEVAL_REFERENCE),
        contradiction_risk=contradiction_risk,
    )


def decide_destination(
    frequency=0,
    importance=0.0,
    confidence=0.0,
    generalization=0.0,
    retrieval_cost=0.0,
    learning_cost=0.0,
    permanence=0.0,
    usefulness=0.0,
    contradiction_risk=0.0,
):
    """Choose where information belongs, and say why.

    The factors are the ones Architecture 16 lists. The weights are an
    implementation choice, and they are biased toward keeping information
    external, because promoting something into weights is the expensive and
    hard to undo direction.
    """
    if not 0.0 <= usefulness <= 1.0:
        raise ValueError("usefulness must be between 0.0 and 1.0")
    reasons = []
    score = 0.0

    if frequency >= 5:
        score += 0.20
        reasons.append(f"seen {frequency} times, so retrieval is likely")
    if importance >= 0.7:
        score += 0.20
        reasons.append(f"importance {importance:.2f} is high")
    if confidence >= 0.8:
        score += 0.20
        reasons.append(f"confidence {confidence:.2f} is well supported")
    if generalization >= 0.6:
        score += 0.15
        reasons.append(f"generalizes broadly ({generalization:.2f})")
    if permanence >= 0.7:
        score += 0.15
        reasons.append("expected to stay true")
    if usefulness >= 0.7:
        score += 0.10
        reasons.append("useful in practice")
    if retrieval_cost >= 0.5:
        score -= 0.10
        reasons.append("cheap to keep external instead")
    if learning_cost >= 0.5:
        score -= 0.10
        reasons.append("expensive to push into weights")
    if contradiction_risk >= 0.4:
        score -= 0.25
        reasons.append("evidence conflicts, so do not commit it")

    if contradiction_risk >= 0.6:
        destination = DEST_EXTERNAL
        reasons.append("high contradiction risk keeps it retrievable, not permanent")
    elif contradiction_risk >= 0.4 and score >= 0.45:
        # Disputed but supported. The evidence is good enough to keep and not
        # good enough to commit to, so it is held as a relationship between what
        # is asserted and what denies it rather than as a settled fact. This is
        # the Architecture 16 case for a relationship: a disagreement is a fact
        # about two claims, and a concept would have to pick one of them.
        destination = DEST_RELATIONSHIP
        reasons.append(
            "supported but disputed, so it is held as a relationship between the "
            "claims rather than as a settled fact"
        )
    elif score >= 0.70 and contradiction_risk == 0:
        destination = DEST_WEIGHTS
    elif score >= 0.70:
        destination = DEST_MULTIPLE
        reasons.append(
            "supported enough to learn and disputed enough to stay external, so "
            "it is held in both rather than being forced into one"
        )
    elif score >= 0.45 and frequency >= 10:
        # Seen often enough to be a pattern rather than an observation, and
        # supported, but not important enough to be worth parameters. That is
        # what an experience is: a record that something happened repeatedly,
        # kept because the repetition is itself the information.
        destination = DEST_EXPERIENCE
        reasons.append(
            f"seen {frequency} times, so the repetition is the information; "
            "kept as an experience rather than a concept"
        )
    elif score >= 0.45:
        destination = DEST_CONCEPT
        reasons.append("useful and supported, but explicit is cheaper than weights")
    elif score >= 0.20:
        destination = DEST_EXTERNAL
    else:
        destination = DEST_EXTERNAL
        reasons.append("default is external; nothing argues for permanence yet")
    if destination not in DESTINATIONS:
        raise ValueError(
            f"destination {destination!r} is not one of the declared "
            f"destinations: {', '.join(DESTINATIONS)}"
        )
    return {
        "destination": destination,
        "score": round(score, 3),
        "contradiction_risk": contradiction_risk,
        "reasons": reasons,
        "note": (
            "Architecture 16: the decision is experimental and exists to be "
            "reviewed. External knowledge does not become permanent internal "
            "knowledge automatically."
        ),
    }
