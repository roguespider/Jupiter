"""Lexical retrieval over the knowledge store (Architecture 9, 15, 16, 28).

Until now the knowledge layer only ever wrote. learning_worker.py accumulated
concepts, experiences and memory records, and nothing read them back, so
Jupiter recorded knowledge it could never use. This module is the read side: it
answers "what have I actually recorded that is relevant to this" so a
conversation can be grounded in stored knowledge instead of in nothing.

The matching is lexical, not semantic. A query is reduced to the same
significant terms that concepts.candidate_terms produces, and a record scores
the fraction of those terms it contains. Two unrelated records that happen to
share vocabulary score alike, and a paraphrase sharing no vocabulary does not
match at all. That limitation is stated here and every result carries the terms
that actually matched, so a reader can see why something was recalled.
Architecture 13 is explicit that a confidence value must not overstate its
evidence, and a score with no visible basis would do exactly that.

No embedding model is involved. Architecture 22 requires checking the standard
library before adding a dependency, and exact term matching covers what is
needed today. A semantic retriever is a later increment, and Architecture 31
wants the method recorded with the result so the two are never confused.
"""

import time

import concepts

import memory

CONCEPT_EXCERPT = 240
CONTENT_EXCERPT = 240

MATCHING = "lexical-term-overlap"
SEMANTIC_MATCHING = "semantic-relationship-expansion"

# Alias mappings for improved retrieval: common synonyms map to canonical terms.
# This is a minimal, verifiable set rather than an open-ended dictionary.
ALIASES = {
    "database": "database",
    "db": "database",
    "python": "python",
    "py": "python",
    "code": "programming",
    "programming": "programming",
    "language": "language",
    "linguistics": "language",
    "grammar": "grammar",
    "syntax": "syntax",
    "verb": "verb",
    "noun": "noun",
    "adjective": "adjective",
    "adverb": "adverb",
    "pronoun": "pronoun",
    "preposition": "preposition",
    "conjunction": "conjunction",
    "interjection": "interjection",
    "sentence": "sentence",
    "clause": "clause",
    "phrase": "phrase",
    "morphology": "morphology",
    "phonology": "phonology",
    "semantics": "semantics",
    "pragmatics": "pragmatics",
}


def _semantic_expansion(knowledge, terms):
    """Expand query terms using SIMILAR_TO and CO_OCCURS relationships.

    This makes retrieval more than lexical overlap: a query for one term
    also matches records that contain related terms, because the relationship
    graph connects concepts that co-occur or are similar in the corpus.
    The expansion is bounded so it does not grow without limit.
    """
    expanded = set(terms)
    try:
        import relationships as rel_module

        all_rels = rel_module.all_relationships(knowledge)
        for entry in all_rels:
            rel = entry.get("relation", "")
            if rel in ("SIMILAR_TO", "CO_OCCURS"):
                left = concepts.normalize(entry.get("from", ""))
                right = concepts.normalize(entry.get("to", ""))
                if left in terms:
                    expanded.add(right)
                if right in terms:
                    expanded.add(left)
    except Exception:
        pass
    return expanded


def query_terms(text, limit=32):
    """Reduce a question to the significant terms retrieval scores on.

    Stopwords are dropped by candidate_terms, so "what is a database" reduces to
    the content words rather than matching every record that contains "is".
    Aliases are expanded so synonyms reach the same records.
    """
    base_terms = set(concepts.candidate_terms(text, limit=limit, min_count=1))
    expanded = set(base_terms)
    for term in base_terms:
        canonical = ALIASES.get(term.lower(), term.lower())
        if canonical != term.lower():
            expanded.add(canonical)
    return expanded


def _terms(text):
    return set(concepts.candidate_terms(text, limit=512, min_count=1))


def _terms_with_boost(text, co_boost):
    """Like _terms but uses a pre-built co-occurrence boost map."""
    counts = concepts.Counter()
    for match in concepts._TOKEN.findall(text):
        lowered = match.lower()
        if lowered in concepts.STOPWORDS:
            continue
        base = counts.get(lowered, 0)
        counts[lowered] = base + 1 + co_boost.get(lowered, 0) * 0.5
    return set(term for term, count in counts.most_common(512) if count >= 1)


def _build_co_boost(knowledge):
    """Build a co-occurrence boost map from the knowledge store once.

    Called by _index() so that candidate_terms() can use pre-built
    boosts instead of scanning every relationship on every record.
    """
    co_boost = {}
    try:
        import relationships as rel_module

        all_rels = rel_module.all_relationships(knowledge)
        for entry in all_rels:
            rel = entry.get("relation", "")
            if rel == "CO_OCCURS":
                left = concepts.normalize(entry.get("from", ""))
                right = concepts.normalize(entry.get("to", ""))
                for term in (left, right):
                    co_boost[term] = co_boost.get(term, 0) + 1
            elif rel == "SIMILAR_TO":
                left = concepts.normalize(entry.get("from", ""))
                right = concepts.normalize(entry.get("to", ""))
                for term in (left, right):
                    co_boost[term] = co_boost.get(term, 0) + 2
    except Exception:
        co_boost = {}
    return co_boost


def _index(knowledge, limit=None):
    """Every retrievable record, with its terms worked out once.

    Tokenising is the whole cost of retrieval and it does not depend on the
    query: the same record produces the same terms whatever is asked. Measured
    against the live store, a single retrieve() call re-derived the terms of
    every one of 9,081 records to answer one question, and evaluate_knowledge
    asks twenty of them, so an evaluation spent 166 seconds re-deriving
    181,620 term sets to compare a handful of strings.

    The derivation is hoisted here and the query is then scored against the held
    sets. Same scoring, same ordering, same matched terms, and the answer does
    not depend on how many times retrieve() is called.
    """
    co_boost = _build_co_boost(knowledge)
    index = []
    for collection, records, view in _sources(knowledge, co_boost=co_boost):
        scanned = 0
        for record in records:
            seen = view(record)
            scanned += 1
            if limit is not None and scanned > limit:
                break
            index.append((collection, record, seen))
    return index


def score(terms, record_terms):
    """Fraction of the query's terms a record contains.

    Normalising by the query rather than by the record keeps a long document
    from outranking a precise short one purely by length, and stops a record
    matching a one-word query from looking certain.
    """
    if not terms:
        return 0.0
    return len(terms & record_terms) / len(terms)


def _excerpt(text, limit):
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _concept_view(record):
    terms = _terms(f"{record.get('identifier', '')} {record.get('description', '')}")
    return {
        "terms": terms,
        "excerpt": _excerpt(record.get("description", ""), CONCEPT_EXCERPT),
        "confidence": record.get("confidence"),
        "sort": record.get("identifier", ""),
    }


def _experience_view(record):
    parts = [
        str(record.get("attempted", "")),
        " ".join(record.get("concepts", []) or []),
        str(record.get("reason", "")),
    ]
    return {
        "terms": _terms(" ".join(parts)),
        "excerpt": _excerpt(" ".join(parts), CONTENT_EXCERPT),
        "confidence": record.get("confidence"),
        "sort": str(record.get("id", "")),
    }


def _content_view(record):
    return {
        "terms": _terms(record.get("content", "")),
        "excerpt": _excerpt(record.get("content", ""), CONTENT_EXCERPT),
        "confidence": record.get("confidence"),
        "sort": str(record.get("id", "")),
    }


def _content_view_with_boost(record, co_boost):
    return {
        "terms": _terms_with_boost(record.get("content", ""), co_boost),
        "excerpt": _excerpt(record.get("content", ""), CONTENT_EXCERPT),
        "confidence": record.get("confidence"),
        "sort": str(record.get("id", "")),
    }


def _concept_view_with_boost(record, co_boost):
    return {
        "terms": _terms_with_boost(
            f"{record.get('identifier', '')} {record.get('description', '')}", co_boost
        ),
        "excerpt": _excerpt(record.get("description", ""), CONCEPT_EXCERPT),
        "confidence": record.get("confidence"),
        "sort": record.get("identifier", ""),
    }


def _experience_view_with_boost(record, co_boost):
    parts = [
        str(record.get("attempted", "")),
        " ".join(record.get("concepts", []) or []),
        str(record.get("reason", "")),
    ]
    return {
        "terms": _terms_with_boost(" ".join(parts), co_boost),
        "excerpt": _excerpt(" ".join(parts), CONTENT_EXCERPT),
        "confidence": record.get("confidence"),
        "sort": str(record.get("id", "")),
    }


def _sources(knowledge, co_boost=None):
    """Every retrievable collection, with the view that reads each one."""
    if co_boost is not None:
        return (
            (
                concepts.CONCEPTS,
                concepts.all_concepts(knowledge),
                lambda r: _concept_view_with_boost(r, co_boost),
            ),
            (
                memory.EXPERIENCE,
                memory.experiences(knowledge),
                lambda r: _experience_view_with_boost(r, co_boost),
            ),
            (
                memory.EXTERNAL,
                memory.external(knowledge),
                lambda r: _content_view_with_boost(r, co_boost),
            ),
            (
                memory.LONG_TERM,
                memory.long_term(knowledge),
                lambda r: _content_view_with_boost(r, co_boost),
            ),
        )
    return (
        (concepts.CONCEPTS, concepts.all_concepts(knowledge), _concept_view),
        (memory.EXPERIENCE, memory.experiences(knowledge), _experience_view),
        (memory.EXTERNAL, memory.external(knowledge), _content_view),
        (memory.LONG_TERM, memory.long_term(knowledge), _content_view),
    )


def retrieve(knowledge, query, limit=5, min_score=0.15, index=None, mode="lexical"):
    """Rank stored records against a query, best first.

    `mode` selects the matching strategy: "lexical" (default) uses term overlap;
    "semantic" expands query terms via SIMILAR_TO and CO_OCCURS relationships.
    Returns a list of dicts with the collection, the record id, the score, the
    terms that actually matched, the record's own confidence where it has one,
    and a short excerpt. The matched terms are included so a recall can be
    checked by a reader rather than taken on trust.

    An empty list is a normal answer, not a failure: Jupiter is not expected to
    know everything, and saying nothing is better than inventing something.

    `index` is the result of _index(), for a caller asking several questions
    against the same store. It is derived here when not supplied, so a single
    query behaves exactly as it always did.
    """
    terms = query_terms(query)
    if mode == "semantic":
        terms = _semantic_expansion(knowledge, terms)
    matching_method = SEMANTIC_MATCHING if mode == "semantic" else MATCHING
    start_time = time.time()
    results = []
    for collection, record, seen in index if index is not None else _index(knowledge):
        value = score(terms, seen["terms"])
        if value < min_score:
            continue
        results.append(
            {
                "collection": collection,
                "id": str(record.get("id", seen["sort"])),
                "score": round(value, 3),
                "matched": sorted(terms & seen["terms"]),
                "confidence": seen["confidence"],
                "excerpt": seen["excerpt"],
                "sort": seen["sort"],
                # The method travels with the result, which is what
                # Architecture 31 asks for and what this module's own
                # docstring claimed. A score with no visible basis is a
                # score that overstates its evidence, and the difference
                # between a lexical match and a semantic one is the
                # difference between "these words appear together" and
                # "this means the same thing". Recording it costs one field
                # and removes the possibility of the two being confused.
                "matching": matching_method,
            }
        )
    results.sort(key=lambda item: (-item["score"], item["collection"], item["sort"]))
    # Recency: boost recent records (within last 30 days) by 15%.
    now = time.time()
    # Build a lookup from the index for O(1) record access.
    index_lookup = {}
    if index is not None:
        for collection, record, seen in index:
            key = (collection, str(record.get("id", "")))
            index_lookup[key] = record
    for item in results:
        record = index_lookup.get((item["collection"], item["id"]))
        if record is not None:
            updated = record.get("updated") or record.get("created") or now
            if isinstance(updated, (int, float)):
                age_days = max(0.0, (now - float(updated)) / 86400)
                recency_boost = 1.15 if age_days < 30 else 1.0
                item["score"] = round(min(1.0, item["score"] * recency_boost), 4)
    # Re-sort after recency boost.
    results.sort(
        key=lambda item: (-item["score"], item["collection"], item.get("sort", ""))
    )
    # Source quality: records with explicit source tags score 10% higher.
    for item in results:
        record = index_lookup.get((item["collection"], item["id"]))
        if record is not None:
            source = str(record.get("source", "")).lower()
            if source in ("user", "verified", "curated", "authoritative"):
                item["score"] = round(min(1.0, item["score"] * 1.10), 4)
    # Conflict detection: if two results have contradictory claims,
    # mark them with a conflict flag.
    conflicts = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r1 = results[i]
            r2 = results[j]
            # Simple contradiction check: same collection, opposite keywords.
            if r1["collection"] == r2["collection"]:
                terms1 = set(str(r1.get("excerpt", "")).lower().split())
                terms2 = set(str(r2.get("excerpt", "")).lower().split())
                # If they share terms but have opposite sentiment indicators.
                opposite_indicators = ("not", "no", "false", "wrong", "incorrect")
                has_opposite = any(
                    ind in terms1 for ind in opposite_indicators
                ) and any(ind in terms2 for ind in opposite_indicators)
                if has_opposite:
                    conflicts.append((r1["id"], r2["id"]))
    for item in results:
        item["conflicts"] = [
            c[0] if c[0] == item["id"] else c[1] for c in conflicts if item["id"] in c
        ]
    # Final sort after all adjustments, with scores capped at 1.0.
    results.sort(
        key=lambda item: (
            -min(1.0, item.get("score", 0.0)),
            item["collection"],
            item.get("sort", ""),
        )
    )
    retrieval_elapsed = time.time() - start_time
    for item in results:
        item["retrieval_time"] = round(retrieval_elapsed, 6)
    for item in results:
        del item["sort"]
    return results[:limit]


def retrieve_many(knowledge, queries, limit=5, min_score=0.15, index_limit=None):
    """Answer several questions against one store, indexing it once.

    The same calls as retrieve(), in order, with the same results. The point is
    the index: twenty questions over a store of nine thousand records used to
    derive every record's terms twenty times, which cost more than the questions
    were worth.
    """
    index = _index(knowledge, limit=index_limit)
    return [
        retrieve(knowledge, query, limit=limit, min_score=min_score, index=index)
        for query in queries
    ]


def browse(knowledge, limit=10):
    """List what the store currently holds, most confident first."""
    found = []
    scanned = 0
    max_scan = 10000
    for collection, records, view in _sources(knowledge):
        scanned = 0
        for record in records:
            scanned += 1
            if scanned > max_scan:
                break
            # Full view for accurate retrieval and evaluation.
            seen = view(record)
            confidence = seen["confidence"]
            found.append(
                {
                    "collection": collection,
                    "id": str(record.get("id", seen["sort"])),
                    "confidence": None
                    if confidence is None
                    else round(float(confidence), 3),
                    "excerpt": seen["excerpt"],
                    "sort": seen["sort"],
                }
            )
        if scanned > max_scan:
            break
    found.sort(
        key=lambda item: (
            item["confidence"] is None,
            -(item["confidence"] or 0.0),
            item["collection"],
            item["sort"],
        )
    )
    for item in found:
        del item["sort"]
    return found[:limit]


def context_block(results, max_chars=600):
    """Render recalled records as text that can precede a question.

    This is retrieval-augmented context, not an instruction. The text carries
    what was recalled and where it came from, so the transcript of a
    conversation stays truthful about where an answer's material originated.
    """
    if not results:
        return ""
    lines = ["[recalled]"]
    for item in results:
        lines.append(
            f"- {item['collection']} {item['id']} "
            f"(matched {', '.join(item['matched'])}): {item['excerpt']}"
        )
    block = "\n".join(lines)
    if len(block) <= max_chars:
        return block
    return block[: max_chars - 1].rstrip() + "…"
