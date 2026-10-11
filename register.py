"""What kind of language a document is, and what is wrong with it (Architecture 27).

Architecture 27 asks for natural-language understanding and generation, and then
names the initial capability: English, programming language text, technical
documentation, informal text, common errors, slang, and malformed input. That
last three are the interesting ones, because they are the ones a byte-level
model actually has to survive and the ones nothing currently measures.

The tokenizer already round-trips malformed input, which is a claim about bytes
surviving. It says nothing about the system coping with a document that is
mostly a stack trace, or slang it has never seen, or text that is not text at
all. Those are properties of a corpus rather than of a tokenizer, and nothing
in the project was recording them, so the knowledge layer had no idea which of
its material was clean English and which was a log file.

So this module classifies. It does not generate and does not understand: it says
what kind of thing a document is, and the note on every result says so. A
classifier that claimed to understand would be claiming more than it can show.

Every signal here is a measurement with a stated method, not a judgement. The
thresholds are stated, the reasons are carried on the record, and a document
that matches nothing is labelled unknown rather than being forced into the
nearest class. Forcing it would turn a gap in the classifier into a fact about
the corpus.
"""

import re
import time

REGISTER = "register"

CLASSES = (
    "prose",
    "programming_language_text",
    "technical_documentation",
    "informal",
    "error_report",
    "slang",
    "malformed",
    "unknown",
)

# Markers per class. Narrow on purpose: a broad list turns a classifier into a
# guesser, and a document mislabelled as prose is a document nobody looks at.
# Error output, split by strength. A stack trace is an error report; a page of
# documentation that says "raises ValueError" is not, and treating the two the
# same files a manual under the wrong register.
STRONG_ERROR = re.compile(
    r"\b(traceback|stack trace|segmentation fault|core dumped|"
    r"panic:|fatal error|unhandled exception)\b",
    re.IGNORECASE,
)
WEAK_ERROR = re.compile(
    r"\b(exception|assertionerror|keyerror|valueerror|typeerror|nameerror|"
    r"nullpointer|referenceerror|syntaxerror|traceback)\b",
    re.IGNORECASE,
)
MARKERS = {
    "programming_language_text": re.compile(
        r"(^\s*(def|class|function|import|from|package|public|private|"
        r"#include|using|namespace)\s|=>|::|\{\s*$|;\s*$|\bself\.\w+)",
        re.MULTILINE,
    ),
    "technical_documentation": re.compile(
        r"\b(architecture|implementation|parameter|configuration|returns|"
        r"raises|example|usage|note|warning|api|version|specification|"
        r"requirement|interface|dependency)\b",
        re.IGNORECASE,
    ),
    "informal": re.compile(
        r"\b(gonna|wanna|gotta|kinda|sorta|nah|yeah|ok|okay|hey|"
        r"btw|imo|imho|tbh|lol|omg|btwn|cheers)\b",
        re.IGNORECASE,
    ),
    "slang": re.compile(
        r"\b(cool|awesome|dude|nah|bro|chill|hyped|legendary|bail|"
        r"ghosted|lowkey|highkey|mid|goated|fire|epic)\b",
        re.IGNORECASE,
    ),
}

# Structural English: articles, auxiliary verbs and pronouns. A document with
# almost none of them is not English prose whatever else it may be.
ENGLISH_FUNCTION_WORDS = re.compile(
    r"\b(the|a|an|is|are|was|were|be|been|being|and|or|but|if|then|"
    r"that|this|these|those|it|its|of|to|in|for|with|as|by|from|not|"
    r"have|has|had|do|does|did|can|could|would|should|will|shall|may)\b",
    re.IGNORECASE,
)

# Malformed input: control characters that are not whitespace, replacement
# characters, and unpaired surrogates encoded as text. These are what a
# tokenizer silently substitutes rather than failing on.
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPLACEMENT = re.compile(r"[�]")
# A high proportion of non-word, non-punctuation characters: binary content
# that happened to be decoded, rather than text.
NON_TEXT = re.compile(r"[^\w\s\.,;:!?'\"()\[\]{}<>/\\@#\$%\^&\*\+=~\-`\|]")

MALFORMED_CONTROL_RATE = 0.001
MALFORMED_NON_TEXT_RATE = 0.15
MALFORMED_REPLACEMENT_COUNT = 1


def statistics(text):
    """The raw counts every decision below is made from."""
    characters = len(text)
    words = text.split()
    return {
        "characters": characters,
        "words": len(words),
        "lines": len([line for line in text.splitlines() if line.strip()]),
        "control": len(CONTROL_CHARACTER.findall(text)),
        "replacement": len(REPLACEMENT.findall(text)),
        "function_words": len(ENGLISH_FUNCTION_WORDS.findall(text)),
    }


def is_malformed(text, counts=None):
    """Whether the text is not really text.

    Three signals, and each is stated rather than blended: control characters
    that are not whitespace, a replacement character from a failed decode, and a
    high share of characters that are neither word, punctuation nor space. A log
    file is not malformed - it is error output, and it has its own class. What
    lands here is content that was never text.
    """
    counts = counts or statistics(text)
    if counts["characters"] == 0:
        return False, "empty"
    if counts["replacement"] >= MALFORMED_REPLACEMENT_COUNT:
        return (
            True,
            f"{counts['replacement']} replacement characters from a failed decode",
        )
    if counts["control"] / counts["characters"] > MALFORMED_CONTROL_RATE:
        return True, f"{counts['control']} control characters"
    odd = len(NON_TEXT.findall(text))
    if odd / counts["characters"] > MALFORMED_NON_TEXT_RATE:
        return True, f"{odd} characters that are neither word, punctuation nor space"
    return False, None


def classify(text):
    """What kind of language a document is, with the evidence for the answer.

    Malformed is decided first, because a document that is not text has no
    language to classify, and answering "prose" about a fragment of binary would
    be worse than saying nothing.
    """
    counts = statistics(text)
    malformed, malformed_reason = is_malformed(text, counts)
    if malformed:
        return _result("malformed", text, counts, malformed_reason)

    if counts["characters"] == 0:
        return _result("unknown", text, counts, "empty document")

    # Marked classes, strongest first. A stack trace beats code, because a
    # traceback is code-shaped and is not a program to learn. A weak error
    # marker alone is not enough: documentation mentions exception classes more
    # often than error reports contain them, and filing a manual under the
    # wrong register helps nobody.
    strong = len(STRONG_ERROR.findall(text))
    if strong:
        return _result("error_report", text, counts, f"{strong} stack or fatal markers")
    for name in ("programming_language_text", "technical_documentation"):
        matches = len(MARKERS[name].findall(text))
        if matches:
            return _result(name, text, counts, f"{matches} {name} markers")
    weak = len(WEAK_ERROR.findall(text))
    if weak >= 3:
        return _result("error_report", text, counts, f"{weak} error class mentions")

    # Slang and informality are markers rather than structure, so they are
    # checked before the English fallback: a sentence carrying several of them
    # is informal English, and reporting it as plain prose loses exactly what
    # Architecture 27 names.
    for name in ("slang", "informal"):
        matches = len(MARKERS[name].findall(text))
        if matches >= 3:
            return _result(name, text, counts, f"{matches} {name} markers")

    # English, judged on structure rather than on vocabulary. A document with
    # almost no function words is not English prose whatever it is about.
    density = counts["function_words"] / max(1, counts["words"])
    if density >= 0.08:
        return _result("prose", text, counts, f"function word density {density:.3f}")

    # Slang and informality are both markers rather than structure, so they are
    # reported only when there is enough of one to be more than an accident.
    for name in ("slang", "informal"):
        matches = len(MARKERS[name].findall(text))
        if matches >= 3:
            return _result(name, text, counts, f"{matches} {name} markers")

    return _result(
        "unknown", text, counts, "no class marker was strong enough to claim"
    )


def _result(name, text, counts, reason):
    return {
        "register": name,
        "reason": reason,
        "characters": counts["characters"],
        "words": counts["words"],
        "lines": counts["lines"],
        "malformed": name == "malformed",
    }


def record(store, source, text, cycle=None, now=None):
    """Classify a document and write the finding into the knowledge store."""
    result = classify(text)
    result["source"] = str(source)
    result["cycle"] = cycle
    result["recorded"] = time.time() if now is None else now
    store.put(REGISTER, str(source), result)
    return result


def census(store):
    """What the corpus is made of, and how much of it is not language at all."""
    rows = [r for r in store.collection(REGISTER).values() if isinstance(r, dict)]
    by_class = {}
    for row in rows:
        name = row.get("register", "unknown")
        by_class[name] = by_class.get(name, 0) + 1
    return {
        "documents": len(rows),
        "by_register": by_class,
        "malformed": sum(1 for r in rows if r.get("malformed")),
        "unclassified": by_class.get("unknown", 0),
        "note": (
            "Registers are measured with stated markers and thresholds, not "
            "understood. This does not generate language and does not claim to "
            "understand it. A document matching nothing is left unknown rather "
            "than forced into the nearest class."
        ),
    }
