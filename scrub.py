"""Strip JSON punctuation from text that is about to become training material.

This is a safety net, not the fix. The fix is in json_source and curate_dataset:
a structured file is parsed and its text taken out of it. That works, and the
corpus proves it. But "works" is not the same as "cannot fail", and this corpus
found the gap twice. Shards cut at byte offsets open mid-record, so the first
thing the parser sees is `"output": "...` rather than a banner or a brace, and
one 67MB file spent weeks as raw punctuation before anything noticed.

So this is the second line. Whatever reaches curation, structured or not, is
measured for the fingerprints of a payload that was never taken apart, and the
ones that are actually payload are removed.

What counts as contamination, and why each test is the one it is:

An escaped newline, `\\n`, is a newline that survived being carried inside a
JSON string. A real newline in the text is a newline. The difference is one
backslash, and a model trained on both learns to emit the two-character sequence
as if it were a line break, which is exactly what the first batch of replies
looked like.

A structural key marker is a quoted name followed by a colon. Prose contains
quoted words, and this corpus contains LaTeX, code and quoted speech, so the
key has to be one a payload actually uses. The list is closed on purpose: a
regex that matched any `"word":` would strip quotes out of ordinary English.

A record separator, `--- SAMPLE n ---`, is a banner. It is structural and it is
not language.

A bare brace at the start of a line is weak evidence on its own, because code
and formatted text open lines with braces legitimately. It is only removed when
the document has already been judged a payload, which is what
looks_structured decides upstream.

What this never does, and the reason it is safe to run over everything:

It does not touch a legitimate brace, a legitimate quote, or a legitimate
backslash. Every removal is anchored to a payload fingerprint or to a payload
count. A book about programming keeps all of its braces, because it has no
payload fingerprints to remove them by.

It is not a parser. Nothing here reconstructs a record or guesses where one
ended. It removes marks that are unambiguously not language and leaves the words
alone, because the words are the only part worth training on.
"""

import re
from pathlib import Path

# A key that names a field in a payload this pipeline reads. Closed on purpose:
# a generic quoted-word-then-colon would match ordinary English and quoted
# speech, and stripping those would damage the corpus to fix a problem it does
# not have.
PAYLOAD_KEYS = (
    "role",
    "content",
    "output",
    "input",
    "text",
    "message",
    "messages",
    "conversation",
    "conversations",
    "speaker",
    "from",
    "value",
    "system_prompt",
    "category",
    "license",
    "generator",
    "used_in_training",
    "reasoning",
    "version",
    "completion",
    "response",
    "answer",
    "target",
    "instruction",
    "prompt",
    "query",
    "utterance",
    "body",
    "line",
    "name",
    "author",
    "sender",
)

# The quoted key markers of a payload, longest names first so "system_prompt" is
# not partially matched as "system".
_KEY_ALTERNATION = "|".join(
    re.escape(key) for key in sorted(PAYLOAD_KEYS, key=len, reverse=True)
)

# The field markers above which a document is judged to be a payload and has
# its quoted keys removed. Four is the smallest real record in this corpus:
# input, role, content, output. Every genuine payload here has thousands, and
# nothing real reaches four.
PAYLOAD_THRESHOLD = 4

# A payload field declaration, with or without whitespace after the colon.
STRUCTURAL_KEY = re.compile(rf'"?\b(?:{_KEY_ALTERNATION})\b"?\s*:\s*')

# A newline that arrived inside a JSON string and was never unescaped.
ESCAPED_NEWLINE = re.compile(r"\\n")

# An escaped carriage return or tab, which travel the same way as an escaped
# newline and corrupt a model the same way.
#
# Both are ambiguous with LaTeX, and this corpus contains a maths document where
# `\times`, `\to` and `\text` are real content. The rule is positional rather
# than by letter: a backslash-r or backslash-t that is immediately followed by a
# letter is a LaTeX command and is left alone, and one followed by anything else
# is an escape that arrived inside a string.
#
# `\times` is the case that makes positional the right rule rather than a
# blocklist. A blocklist of LaTeX commands starting with t has to be exhaustive
# to be safe, and an exhaustive list of LaTeX commands is a thing nobody has.
# `\times` is not on any plausible blocklist, which is how a naive pattern turns
# "a \times b" into "a  imes b" and quietly deletes a page of algebra.
ESCAPED_TAB = re.compile(r"\\t(?![A-Za-z])")
ESCAPED_CARRIAGE = re.compile(r"\\r(?![A-Za-z])")

# An escaped quote inside a payload string. Guarded against a following
# backslash, because `\\\\"` is JavaScript: this corpus contains a programming
# document whose Windows paths are written as `"\\\\Desktop\\\\"`, and unescaping
# that turns a working path literal into something that does not compile. An
# escaped quote is followed by a letter, a space or punctuation in a payload;
# the backslash case is a language of its own.
ESCAPED_QUOTE = re.compile(r'\\"(?!\\)')

# The banner that introduces a record in a sharded export.
SAMPLE_BANNER = re.compile(r"(?m)^--- SAMPLE \d+ ---[ \t]*$\n?")

# A doubled quote left behind once the keys are gone, and a colon with nothing
# before it. Both are what removing a key leaves behind, and both are punctuation
# rather than language.
ORPHAN_COLON = re.compile(r"(?m)^[ \t]*:[ \t]*$")
DOUBLE_QUOTE_RUN = re.compile(r'"{2,}')

# Three or more consecutive newlines, collapsed to two. Held here rather than
# written inline, because a pattern literal inside the scrub is looked up in the
# re module's global cache on every call and this one is applied to every
# document the pipeline accepts.
BLANK_RUN = re.compile(r"\n{3,}")


def looks_like_payload(text, threshold=PAYLOAD_THRESHOLD):
    """Whether a text carries enough payload marks to be judged one.

    A count and not a judgement. One `"role":` in a document about JSON is
    prose that happens to mention a field name; a run of them is a payload that
    was not taken apart.

    The threshold is 4 because the smallest real record in this corpus is four
    fields: input, role, content, output. At 8 that record was judged prose and
    kept its field names, which is the exact failure this exists to catch. Four
    was measured against the corpus rather than chosen: every real payload here
    has thousands of markers, and no piece of real code or prose in the corpus
    reaches four. LaTeX, source code, quoted speech and a book quoting a JSON
    snippet were all checked at this threshold and none of them is judged one.
    """
    if not text:
        return False
    # findall builds a list of every match in the document, and this only asks
    # whether there are enough of them. Counting stops once the threshold is
    # passed, so a payload carrying four thousand markers stops looking after
    # the fourth rather than collecting four thousand of them.
    found = 0
    for _match in STRUCTURAL_KEY.finditer(text):
        found += 1
        if found >= threshold:
            return True
    return False


def scrub(text, aggressive=False):
    """Remove payload punctuation from text, and report what was removed.

    Returns the cleaned text and a count of each kind, because a scrub that
    changed nothing and a scrub that is not running look identical otherwise,
    and the difference between them is the difference between a clean corpus and
    a quietly contaminated one.

    Payloads that carry a question and an answer are the case where scrubbing is
    destructive rather than corrective: removing the keys leaves the words, which
    is a conversation with the labels taken off. That is strictly better than
    training on the labels, and it is what this is for. The parser is what
    recovers the labels properly, and it runs first.

    aggressive=False removes only what cannot be anything but a payload: an
    escaped newline, an escaped quote, a record banner. A `{"name": ...}` in a
    book about programming is real code, and removing its keys would damage the
    corpus to fix a problem it does not have. Measured against this project's
    own corpus, that is not hypothetical: the maths and programming material is
    full of legitimate quoted keys.

    aggressive=True removes the structural keys too, and is only correct for a
    document that has already been judged a payload. It mangles code, which is
    why it is not the default and why scrub_file gates it on looks_like_payload.
    """
    if not text:
        return "", {
            "structural_keys": 0,
            "escaped_newlines": 0,
            "escaped_whitespace": 0,
            "escaped_quotes": 0,
            "banners": 0,
            "orphan_colons": 0,
        }

    # Every step below used to run each pattern twice: once through findall to
    # count the matches, then again through sub to remove them. On a sixty-six
    # megabyte payload document that was ten seconds, of which four and a half
    # was the counting pass alone, re-walking a string the substitution was about
    # to walk anyway. subn does both in one pass and returns the same count,
    # because it is the same scan by the same engine over the same text - findall
    # would have returned one entry per non-overlapping match, and subn reports
    # one substitution per non-overlapping match. The two counts are the same
    # number.
    cleaned, banners = SAMPLE_BANNER.subn("", text)

    # The escapes are removed first and unconditionally. A literal backslash-n is
    # not valid in ordinary prose and is not valid in ordinary code either; the
    # one place it belongs is inside a JSON string, which is the thing being
    # removed here.
    cleaned, newlines = ESCAPED_NEWLINE.subn("\n", cleaned)

    # Carriage return first, then tab. Order is immaterial to the counts: each
    # pattern matches a backslash and a letter, and replacing one leaves the
    # other's matches exactly where they were.
    cleaned, carries = ESCAPED_CARRIAGE.subn(" ", cleaned)
    cleaned, tabs = ESCAPED_TAB.subn(" ", cleaned)
    whitespace = carries + tabs

    cleaned, quotes = ESCAPED_QUOTE.subn('"', cleaned)

    keys = 0
    colons = 0
    runs = 0
    if aggressive:
        cleaned, keys = STRUCTURAL_KEY.subn("", cleaned)

        cleaned, colons = ORPHAN_COLON.subn("", cleaned)

        cleaned, runs = DOUBLE_QUOTE_RUN.subn('"', cleaned)

        # Collapsing the blank lines a removed key leaves behind, so a document
        # that was mostly field declarations does not become mostly whitespace.
        cleaned = BLANK_RUN.sub("\n\n", cleaned)

    return cleaned, {
        "structural_keys": keys,
        "escaped_newlines": newlines,
        "escaped_whitespace": whitespace,
        "escaped_quotes": quotes,
        "banners": banners,
        "orphan_colons": colons,
        "quote_runs": runs,
    }


def total(counts):
    """Every removal a scrub made, as one number."""
    return sum(value for value in counts.values() if isinstance(value, int))


def scrub_file(path, threshold=PAYLOAD_THRESHOLD, force=False):
    """Scrub a file in place, and report what changed.

    Reads defensively: a curated file is being rewritten by a pipeline that can
    be interrupted, and a scrub that cannot open its input must not truncate the
    corpus it was asked to clean.

    Key removal is gated on the document already looking like a payload. Below
    the threshold only the escapes are touched, because a book about
    programming has quoted keys in it that are real code.

    Nothing is written when nothing was removed. A rewrite that changes nothing
    is churn, and on a 57MB document it is also the slowest possible way to do
    nothing.
    """
    target = Path(path)
    try:
        original = target.read_text(encoding="utf-8", errors="replace")
    except OSError as failure:
        return {
            "path": str(target),
            "removed": 0,
            "counts": {},
            "note": f"could not be read: {failure}",
        }

    suspect = force or looks_like_payload(original, threshold)
    cleaned, counts = scrub(original, aggressive=suspect)
    removed = total(counts)

    if removed == 0:
        return {
            "path": str(target),
            "removed": 0,
            "counts": counts,
            "judged_payload": suspect,
            "note": (
                "nothing to remove"
                if suspect
                else (
                    "clean; no payload escapes found, and too few field markers "
                    "to judge it a payload, so quoted keys were left alone"
                )
            ),
        }

    target.write_text(cleaned, encoding="utf-8")
    return {
        "path": str(target),
        "removed": removed,
        "counts": {key: value for key, value in counts.items() if value},
        "judged_payload": suspect,
        "characters_before": len(original),
        "characters_after": len(cleaned),
        "note": (
            "payload punctuation removed; the words were kept, because the words "
            "are the part worth learning"
        ),
    }


def census(paths, threshold=PAYLOAD_THRESHOLD):
    """What every named file would lose, without writing anything."""
    rows = []
    for name in paths:
        target = Path(name)
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as failure:
            rows.append({"path": str(target), "error": str(failure)})
            continue
        suspect = looks_like_payload(text, threshold)
        _, counts = scrub(text, aggressive=suspect)
        rows.append(
            {
                "path": str(target),
                "removed": total(counts),
                "judged_payload": suspect,
                "counts": {key: value for key, value in counts.items() if value},
            }
        )
    rows.sort(key=lambda row: row.get("removed", 0), reverse=True)
    return rows


def main(c=None, argv=None):
    """Scrub the curated corpus from the command line.

    Exists because the entry point reaches this, and an inline python -c in a
    batch file is not a command line anyone can read, quote correctly, or run
    by hand. Curation already scrubs as documents pass through it; this reaches
    the files that were curated before that existed, because they are already
    on disk and nothing rewrites them.

    Nothing is written for a file with nothing to remove, so running it twice
    reports nothing the second time rather than churning the corpus.
    """
    import argparse

    from config import load_config

    parser = argparse.ArgumentParser(
        prog="scrub",
        description=(
            "Remove JSON payload punctuation from the curated corpus. A net "
            "under the parser in json_source, for files curated before it."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be removed without writing anything",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=PAYLOAD_THRESHOLD,
        help=(
            "field markers above which a document is judged a payload and its "
            "quoted keys are removed; below it only escapes are touched"
        ),
    )
    args = parser.parse_args(argv)

    c = c or load_config()
    paths = sorted(Path(c.curated_dir).glob("*.txt"))
    if not paths:
        print(f"  no curated documents in {c.curated_dir}")
        return 1

    if args.dry_run:
        rows = census(paths, threshold=args.threshold)
    else:
        rows = [scrub_file(path, threshold=args.threshold) for path in paths]

    rows.sort(key=lambda row: row.get("removed", 0), reverse=True)
    for row in rows:
        removed = row.get("removed", 0)
        if not removed:
            continue
        counts = row.get("counts") or {}
        breakdown = ", ".join(f"{name} {value}" for name, value in counts.items())
        verb = "would remove" if args.dry_run else "removed"
        print(f"  {verb} {removed:>7}  {Path(row['path']).name[:44]}  {breakdown}")

    touched = sum(1 for row in rows if row.get("removed"))
    total = sum(row.get("removed", 0) for row in rows)
    label = "would be removed" if args.dry_run else "removed"
    print(
        f"  {total} payload marks {label} from {touched} of {len(rows)} "
        f"curated documents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
