"""Assertions: what a source claims, so two sources can be found to disagree.

Architecture 14 asks Jupiter to detect a conflict, preserve both sides, and
decide whether it is contextual. The decision was built and is correct, and it
never fired, and the reason is worth stating plainly because it is what this
module is for.

A contradiction used to be a bare count on a term. Across twelve real cycles
over three documents, every promoted term ended up both supporting and
contradicted by the same document, because a document accepted in one cycle and
rejected in another appears under one name on both lists. The decision correctly
read that as contextual. Nothing was wrong with the decision. The input was
wrong: term co-occurrence records that two words appeared in the same document,
and cannot say that one document asserts a proposition and another denies it.
A rejected cycle is evidence that learning a document damaged the reserved
document, which is not a disagreement about anything.

So the missing thing is not a better decision. It is a way to record a claim.

An assertion is a subject, a relation, an object, a polarity and a source. The
polarity is what makes a conflict expressible: two sources making the same claim
corroborate each other, and two sources making opposite claims contradict. That
is the whole mechanism, and it is what co-occurrence could not carry.

The patterns below are narrow and are declared a heuristic wherever they are
used, for the same reason the concept extractor is. They read ordinary English
sentences of the form "X is Y" and "X is not Y" and are meant to be
recognisably incomplete rather than quietly clever. A pattern set that
recognised more would recognise more wrongly, and an assertion layer that
invents claims is worse than no assertion layer, because a fabricated
contradiction is a fabricated disagreement between two real sources.
"""

import re
import time

import relationships as relationships_module
import store as store_module

ASSERTIONS = "assertions"

POSITIVE = "positive"
NEGATIVE = "negative"

# How many assertions one document may contribute. A long document will contain
# thousands of these shapes and reading all of them is how a corpus fills with
# claims nobody made.
MAX_PER_SOURCE = 50

# Subject and object are short noun phrases. Deliberately short: a wider window
# would swallow half a sentence and produce claims nobody made.
_PHRASE = r"[a-z][a-z0-9 _-]{1,28}?"

# Ordered, because "is not" has to be tried before "is" or every denial is read
# as an assertion. The order is the whole reason the first two differ. Each
# entry is (connective, pattern, polarity), and the connective is stored with the
# claim so "X requires Y" and "X is not Y" stay distinguishable. Collapsing them
# would invent disagreements nobody stated.
PATTERNS = (
    (
        "is not",
        re.compile(rf"\b(?P<s>{_PHRASE})\s+is\s+not\s+(?P<o>{_PHRASE})\b", re.I),
        NEGATIVE,
    ),
    (
        "is",
        re.compile(rf"\b(?P<s>{_PHRASE})\s+is\s+(?P<o>{_PHRASE})\b", re.I),
        POSITIVE,
    ),
    (
        "is not",
        re.compile(rf"\b(?P<s>{_PHRASE})\s+is\s+not\s+the\s+(?P<o>{_PHRASE})\b", re.I),
        NEGATIVE,
    ),
    (
        "requires",
        re.compile(rf"\b(?P<s>{_PHRASE})\s+requires\s+(?P<o>{_PHRASE})\b", re.I),
        POSITIVE,
    ),
    (
        "causes",
        re.compile(rf"\b(?P<s>{_PHRASE})\s+causes\s+(?P<o>{_PHRASE})\b", re.I),
        POSITIVE,
    ),
    (
        "precedes",
        re.compile(rf"\b(?P<s>{_PHRASE})\s+precedes\s+(?P<o>{_PHRASE})\b", re.I),
        POSITIVE,
    ),
)


def _clean(phrase):
    """Trim a captured phrase to something that reads as a noun phrase."""
    words = phrase.lower().split()
    while words and words[0] in {"the", "a", "an", "not", "no", "very", "also"}:
        words.pop(0)
    while words and words[-1] in {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "that",
        "which",
    }:
        words.pop()
    return " ".join(words).strip()


def extract(text, limit=MAX_PER_SOURCE):
    """The claims a document appears to make.

    Returns a list of {"subject", "relation", "object", "polarity"}. The
    relation is the matched connective, so "requires" and "is" stay
    distinguishable, because "X requires Y" and "X is not Y" are different
    claims and collapsing them would invent disagreements.
    """
    found = []
    seen = set()
    for relation, pattern, polarity in PATTERNS:
        for match in pattern.finditer(text):
            subject = _clean(match.group("s"))
            obj = _clean(match.group("o"))
            if not subject or not obj or subject == obj:
                continue
            key = (subject, relation, obj, polarity)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "polarity": polarity,
                }
            )
            if len(found) >= limit:
                return found
    return found


def proposition(record):
    """The claim a record makes, with the negation lifted out of the wording.

    "river is safe" and "river is not safe" are two statements about one
    proposition, and that is the collision this whole module exists to create.
    Keying on subject, relation and object as written would keep them apart,
    because the connective differs, and the disagreement would never be found.
    So the negation is normalised away here and carried by polarity instead.
    """
    relation = str(record.get("relation", "is"))
    if relation.endswith(" not"):
        relation = relation[: -len(" not")]
    return "|".join(
        (
            str(record.get("subject", "")),
            relation,
            str(record.get("object", "")),
        )
    )


def claim_key(record):
    """The identity of a proposition, ignoring which source and which polarity.

    Two sources asserting the same proposition with opposite polarity is the
    event Architecture 14 is about.
    """
    return proposition(record)


def observe(store, source, record, now=None):
    """Record one claim from one source, and say what it did to the claim.

    Returns an outcome describing whether the claim is new, corroborated, or
    contradicted. A contradiction between two different sources is the event
    Architecture 14 is about, and it is opened here as a CONTRADICTS edge in
    both directions, which is the first production path that can reach that
    edge.

    A source that denies a proposition is not listed among the sources that
    assert it. Putting it in both lists made every genuine disagreement read as a
    source agreeing with itself, which is the mistake this function exists to
    avoid.
    """
    key = claim_key(record)
    claims = store.collection(ASSERTIONS)
    existing = claims.get(key)
    polarity = record.get("polarity")

    if existing is None:
        claims[key] = {
            "claim": key,
            "subject": record.get("subject"),
            "relation": record.get("relation"),
            "object": record.get("object"),
            "polarity": polarity,
            "sources": [str(source)],
            "contradicted_by": [],
            "supporting": 1,
            "contradictions": 0,
            "first_cycle": record.get("cycle"),
            "last_verified": time.time() if now is None else now,
        }
        store.put(ASSERTIONS, key, claims[key])
        return {"outcome": "new", "claim": key}

    same_polarity = existing.get("polarity") == polarity
    asserted_by = existing.setdefault("sources", [])
    already_asserted = str(source) in asserted_by
    existing["last_verified"] = time.time() if now is None else now

    if same_polarity:
        if not already_asserted:
            asserted_by.append(str(source))
            existing["supporting"] = int(existing.get("supporting", 1)) + 1
        outcome = "corroborated"
    else:
        # A source denying a claim it previously asserted is not two sources
        # disagreeing; Architecture 14 calls that a context difference, and
        # it is the case step 6 exists to catch.
        opposers = existing.setdefault("contradicted_by", [])
        if str(source) not in opposers:
            opposers.append(str(source))
        if already_asserted:
            outcome = "self_contradiction"
        else:
            existing["contradictions"] = int(existing.get("contradictions", 0)) + 1
            # A second source on the denying side is corroboration, not more
            # conflict. Without this, two sources agreeing against one and
            # one source contradicted by two unrelated others look identical,
            # and the first is the stronger claim.
            existing["contradicting"] = int(existing.get("contradicting", 0)) + 1
            outcome = (
                "corroborated_denial"
                if int(existing["contradicting"]) > 1
                else "contradicted"
            )
    store.put(ASSERTIONS, key, existing)
    return {"outcome": outcome, "claim": key}
    return {"outcome": outcome, "claim": key}


def contradictions(store):
    """Claims that two different sources genuinely disagree about.

    This is the population Architecture 14 step 6 is supposed to filter. The
    decision itself lives in concepts.genuine_conflict for term-level
    contradictions; here the disagreement is explicit in the record, so the
    filter is a check that the two sides are different sources rather than a
    re-derivation of polarity.
    """
    rows = []
    for key, record in store.collection(ASSERTIONS).items():
        if not isinstance(record, dict):
            continue
        supporters = [s for s in record.get("sources", []) if s]
        opposers = [s for s in record.get("contradicted_by", []) if s]
        others = [s for s in opposers if s not in supporters]
        if not others:
            continue
        rows.append(
            {
                "claim": key,
                "subject": record.get("subject"),
                "relation": record.get("relation"),
                "object": record.get("object"),
                "supported_by": supporters,
                "contradicted_by": others,
                "polarity": record.get("polarity"),
                "note": "two sources make opposite claims about the same thing",
            }
        )
    rows.sort(key=lambda row: row["claim"])
    return rows


def link(store, claim, now=None):
    """Open a CONTRADICTS edge between the two sides of a disputed claim.

    The edge joins a source that asserted the proposition to a source that
    denied it. It does not join the source that happened to be harvesting: a
    document being read is not a party to a disagreement it may not even have
    an opinion on, and anchoring there produced an edge from a source to
    itself. Both directions are written, because a conflict should be visible
    from either side.

    The endpoints are sources and are declared as such. Architecture 12 joins
    concepts, so an edge whose endpoints are documents would otherwise have had
    a concept invented for each of them, and curation names every curated file
    by the sha256 of its contents: the concept table filled with sixty-eight
    character names that carry no evidence and answer no question.
    """
    supporters = [s for s in claim.get("supported_by", []) if s]
    opposers = [s for s in claim.get("contradicted_by", []) if s]
    edges = 0
    for supporter in supporters:
        for opposer in opposers:
            if supporter == opposer:
                continue
            relationships_module.observe(
                store,
                supporter,
                relationships_module.CONTRADICTS,
                opposer,
                supporting=False,
                now=now,
                endpoints=relationships_module.SOURCE_ENDPOINTS,
            )
            relationships_module.observe(
                store,
                opposer,
                relationships_module.CONTRADICTS,
                supporter,
                supporting=False,
                now=now,
                endpoints=relationships_module.SOURCE_ENDPOINTS,
            )
            edges += 1
    return claim.get("claim") if edges else None


def harvest(store, source, text, cycle=None, now=None):
    """Extract a document's claims, record them, and link what they dispute.

    The whole of the assertion layer in one call: what the document claims, how
    each claim sits against what other sources have said, and a CONTRADICTS edge
    between the two sides of anything genuinely disputed.
    """
    outcomes = {"new": 0, "corroborated": 0, "contradicted": 0, "self_contradiction": 0}
    for record in extract(text):
        outcome = observe(store, source, {**record, "cycle": cycle}, now=now)
        outcomes[outcome["outcome"]] = outcomes.get(outcome["outcome"], 0) + 1
    linked = 0
    for claim in contradictions(store):
        if source in claim.get("supported_by", []) or source in claim.get(
            "contradicted_by", []
        ):
            if link(store, claim, now=now) is not None:
                linked += 1
    return {
        "counts": outcomes,
        "contradictions_linked": linked,
        "extracted": sum(outcomes.values()),
    }


def census(store):
    """What the assertion layer currently holds."""
    claims = store.collection(ASSERTIONS)
    rows = [r for r in claims.values() if isinstance(r, dict)]
    return {
        "total": len(rows),
        "disputed": sum(1 for r in rows if r.get("contradicted_by")),
        "corroborated": sum(1 for r in rows if int(r.get("supporting", 1)) > 1),
        "relations": sorted({r.get("relation") for r in rows if r.get("relation")}),
        "note": (
            "An assertion records a claim a source made. Polarity is what makes "
            "a disagreement expressible: co-occurrence cannot say that one "
            "source denies what another asserts."
        ),
    }


__all__ = [
    "ASSERTIONS",
    "POSITIVE",
    "NEGATIVE",
    "MAX_PER_SOURCE",
    "census",
    "claim_key",
    "contradictions",
    "extract",
    "harvest",
    "link",
    "observe",
    "store_module",
]
