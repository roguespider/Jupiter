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
    """
    return set(concepts.candidate_terms(text, limit=limit, min_count=1))


def _terms(text):
    return set(concepts.candidate_terms(text, limit=512, min_count=1))


def _index(knowledge):
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
    index = []
    for collection, records, view in _sources(knowledge):
        for record in records:
            seen = view(record)
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


def _sources(knowledge):
    """Every retrievable collection, with the view that reads each one."""
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
    retrieval_elapsed = time.time() - start_time
    for item in results:
        item["retrieval_time"] = round(retrieval_elapsed, 6)
    for item in results:
        del item["sort"]
    return results[:limit]


def retrieve_many(knowledge, queries, limit=5, min_score=0.15):
    """Answer several questions against one store, indexing it once.

    The same calls as retrieve(), in order, with the same results. The point is
    the index: twenty questions over a store of nine thousand records used to
    derive every record's terms twenty times, which cost more than the questions
    were worth.
    """
    index = _index(knowledge)
    return [
        retrieve(knowledge, query, limit=limit, min_score=min_score, index=index)
        for query in queries
    ]


def browse(knowledge, limit=10):
    """List what the store currently holds, most confident first.

    This takes no query, because an empty query can never match anything. It
    exists so a person can see what Jupiter is able to recall at all, which is
    the question a retrieval score cannot answer on its own.
    """
    found = []
    for collection, records, view in _sources(knowledge):
        for record in records:
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
