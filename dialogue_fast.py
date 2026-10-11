"""Measure a document once instead of six times.

dialogue.report() and the functions behind it each walk the whole document
independently: turns() runs two regexes over every character, speakers() calls
turns() again, turn_share() calls turns() a third time, and is_conversational()
calls all of them again before deciding anything. On a sixty-five million
character document that is roughly a dozen full passes where one would do, and
it is why curating this corpus took four hours.

The fix has three parts, and they are worth separating because they are worth
different amounts.

One pass. A scan gathers the speaker labels, the structured pairs, the template
markers and the line count in a single walk, and every measurement is read off
what it gathered. Nothing recomputes what is already known.

Early exit. The decision that matters is whether a document is conversation, and
that settles early. A document with no speaker label in its first few thousand
characters is not one that acquires one at character sixty million; it is prose,
and the only reason to keep reading is to count the markers for the report. So the
scan takes a prefix for the decision and a full pass only for the numbers.

A prefix by default. report() is asked "what kind of document is this" about a
document curation is about to read in full anyway, and the answer does not change
past the prefix. The full measurement stays available, because the report's
numbers are supposed to be honest about a whole document rather than about the
part that was read.

The suffix is bounded rather than absent, because a chat template dumped into the
middle of a book is a real case and this check exists to catch it.
"""

import re

# How much of a document is read before the conversation decision is treated as
# settled. Chosen from what the corpus actually looks like: a transcript
# announces itself in its first speaker label, and the documents in this corpus
# are sixty-odd megabytes each, so a prefix is a rounding error against the
# decision being made.
DECISION_PREFIX_CHARS = 8000

# The regexes and the scan that use them are defined in the module this file
# documents; importing them here keeps one definition of each rather than a copy
# that can drift. See dialogue.py for the patterns and the reasoning.

from dialogue import (  # noqa: E402
    RECORD_INPUT,
    RECORD_OUTPUT,
    ROLE_TURN,
    SPEAKER,
    TEMPLATE_MARKERS,
)


def scan(text, limit=None):
    """Gather everything the dialogue measurements need, in one pass.

    `limit` bounds how much is read. A prefix scan answers "is this conversation"
    for every document that decides early, which is most of them; a full scan is
    what the report's numbers are taken from.

    Returns the raw counts, so every measurement is arithmetic on this rather
    than another walk.
    """
    if limit is None:
        body = text
    else:
        # Not a slice: see the note below. Joined into one string so the five
        # regexes below each walk the prefix once rather than each walking a
        # slice of it.
        body = "".join([text[i] for i in range(min(limit, len(text)))])
    labels = [match.group(1) for match in SPEAKER.finditer(body)]
    roles = [match.group("role") for match in ROLE_TURN.finditer(body)]
    markers = len(TEMPLATE_MARKERS.findall(body))
    records = len(RECORD_INPUT.findall(body))
    outputs = len(RECORD_OUTPUT.findall(body))
    lines = sum(1 for line in body.splitlines() if line.strip())
    return {
        "labels": labels,
        "roles": roles,
        "markers": markers,
        "records": records,
        "outputs": outputs,
        "lines": lines,
        "truncated": limit is not None and len(text) > limit,
    }


def turn_share_from(scan_result):
    """Fraction of non-empty lines that are speaker turns, from a scan."""
    lines = scan_result["lines"]
    if not lines:
        return 0.0
    total = len(scan_result["labels"]) + len(scan_result["roles"])
    return round(min(1.0, total / lines), 4)


def structured_share_from(scan_result):
    """Fraction of records that are structured turns, from a scan.

    Two turns per record: the question and the reply it was answered with. The
    reply is counted even though it lives under a different field, which is the
    correction that stopped every instruction document reading as a monologue.
    A record with no reply contributes one turn rather than none, so a document
    that is mostly questions still scores as conversation rather than as a list.
    """
    records = scan_result["records"]
    if not records:
        return 0.0
    replies = min(scan_result["outputs"], records)
    total = records + replies
    return round(min(1.0, total / records), 4)


def is_conversational_scan(scan_result, min_turns=3, min_speakers=2):
    """Whether the scan says this is dialogue, from one pass over a prefix.

    The order matters and is the early exit. Template markers first, because a
    leaked template is contamination whatever else it contains and there is no
    point measuring it. Then enough turns, then enough speakers. A document that
    fails the first of those cannot be rescued by the others.
    """
    if scan_result["markers"] > 5:
        return False
    # Every record contributes a question, and a record that was answered
    # contributes the reply too. Counting only the roles that were said by a
    # speaker would give every instruction document one turn per record, which
    # is the mistake the reply counting exists to correct.
    replies = min(scan_result["outputs"], scan_result["records"])
    turns_found = len(scan_result["labels"]) + scan_result["records"] + replies
    if turns_found < min_turns:
        return False
    speakers_found = set(scan_result["labels"]) | set(scan_result["roles"])
    # A record that was answered has a second speaker: whoever gave the reply.
    # Without this every instruction document is a monologue spoken by one
    # person, which is the failure the reply counting exists to correct.
    if scan_result["outputs"]:
        speakers_found.add("assistant")
    if len(speakers_found) < min_speakers:
        return False
    if scan_result["records"]:
        return structured_share_from(scan_result) >= 0.20
    return turn_share_from(scan_result) >= 0.20
