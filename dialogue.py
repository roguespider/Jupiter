"""Dialogue structure (Architecture 25, 27, 44).

Architecture 25 lists conversations as a legitimate origin for learning
material, and none of the current corpus is conversational: it is twelve
Asimov novels. Architecture 27 asks for natural-language understanding and
generation, and a model that has only ever seen prose has no way to learn that
a turn exists.

So this module defines the shape of a turn and recognises one. It does not
invent any. Supplying conversational text is a data question, and the honest
position is that the pipeline is ready for it and has none: the curation
report says how many conversational documents it found, and if that number is
zero then Jupiter has never seen a conversation and should not pretend
otherwise.

Speaker labels are kept as they are written rather than collapsed into a single
pair of roles. A model trained on ``Alice: ...`` and ``Bob: ...`` learns that a
turn is a label followed by text, which is the transferable part. Collapsing
everybody into User and Jupiter would throw away the variety that makes the
pattern learnable, and it would also be a semantic decision about who said
what, which curation has no business making.
"""

import re

# A speaker label at the start of a line: a capitalised name, then a colon.
# Deliberately simple and deliberately labelled a heuristic, in the same way
# concept extraction is.
SPEAKER = re.compile(r"^[ \t]*([A-Z][A-Za-z .'-]{0,24}):(?:[ \t]|$)", re.MULTILINE)

# Chat-template markers, which are not dialogue. A document that contains these
# has had a conversation template baked into it, and training on them teaches
# the model to emit instructions rather than to converse. They are counted and
# reported separately so the problem is visible instead of being learned from.
TEMPLATE_MARKERS = re.compile(
    r"^[ \t]*\[?(system|user|assistant|human)\]?[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)

HUMAN = "You"
JUPITER = "Jupiter"

# Below this a document is prose that happens to contain a colon, not dialogue.
MIN_TURNS = 3
MIN_SPEAKERS = 2

# A conversation is mostly turns. Prose that happens to contain a few labelled
# lines is not, and the ratio is what separates them: a novel with a handful of
# artifacts scores near zero, a chat log or a play script scores high.
MIN_TURN_SHARE = 0.20


# A conversation held as structured records rather than as labelled lines. The
# corpus arrived carrying instruction data as JSON objects of the shape
# {"input": [{"role": "user", "content": "..."}], "output": "..."} rather than
# as "Alice: ..." transcripts, and the line-based reader above sees none of it:
# every role lives inside a quoted JSON key, not at the start of a line. Two
# hundred and fifty-nine megabytes of the best conversation material in the
# corpus was therefore being reported as prose, and turn_count read zero.
#
# These patterns recognise the two shapes that actually occur. They are
# deliberately narrow. A permissive "role" search would match any JSON document
# that mentions the word, and the refusal this guards against is the expensive
# one to get wrong in the other direction.
ROLE_TURN = re.compile(
    r'"role"\s*:\s*"(?P<role>user|human|assistant|system|ai|bot|jupiter)"',
    re.IGNORECASE,
)
RECORD_INPUT = re.compile(r'"input"\s*:\s*\[(?P<body>.*?)\]', re.IGNORECASE | re.DOTALL)
RECORD_OUTPUT = re.compile(
    r'"output"\s*:\s*"(?P<body>(?:[^"\\]|\\.)*)"', re.IGNORECASE | re.DOTALL
)


def role_turns(text):
    """Roles named in the input side of structured records, in order.

    Reported separately from `turns` because it answers a narrower question:
    which speakers the corpus addresses, as opposed to how many turns a
    document holds. A record's reply is not in its input, so this sees one
    speaker per record even where the document is a conversation throughout.
    """
    return [match.group("role") for match in ROLE_TURN.finditer(text)]


def _structured_pairs(text):
    """(role, body) pairs from structured records, in document order.

    A pair is a message that was said together with who said it. The output
    field is attributed to Jupiter because it is the reply the model is being
    taught to produce, and attributing it to the role that happened to be last
    in the input would teach the model to continue its own question.
    """
    pairs = []
    for record_input in RECORD_INPUT.finditer(text):
        body = record_input.group("body")
        content = re.search(
            r'"content"\s*:\s*"(?P<body>(?:[^"\\]|\\.)*)"', body, re.DOTALL
        )
        # The role is taken from the first one named rather than by a search of
        # the whole body, so a message whose text happens to quote the word
        # "role" cannot claim to be the speaker.
        role = ROLE_TURN.search(body)
        if content is not None and role is not None:
            pairs.append((role.group("role"), content.group("body")))
    # An output belongs to the record it follows, so the whole document is
    # scanned once and each output is attributed to the last input seen before
    # it rather than to a record boundary this function would have to guess at.
    pending = list(pairs)
    produced = []
    for output in RECORD_OUTPUT.finditer(text):
        if pending:
            role, body = pending.pop(0)
            produced.append((role, body))
            produced.append(("assistant", output.group("body")))
    return produced


def structured_turn_share(text):
    """Fraction of records that are structured turns rather than raw text.

    Measured per record rather than per line, because a JSON document is often
    one enormous line and a per-line share would call a document made entirely
    of conversation worth zero.
    """
    records = len(RECORD_INPUT.findall(text))
    if not records:
        return 0.0
    return round(min(1.0, len(_structured_pairs(text)) / (2 * records)), 4)


def turns(text):
    """Every speaker label found, in order.

    Both layouts are read and reported together, because a corpus may hold one
    or the other and a report that only saw one of them undercounts the
    conversation material the project exists to learn from.

    The structured roles come from the parsed pairs rather than from a bare
    search for the word "role". A search finds only the roles that were said by
    a speaker, and a record always has exactly one such speaker while its reply
    is stored under a different field entirely, so every structured document
    looked like a monologue: one speaker, and therefore not a conversation at
    all. Reading the pairs counts the reply too, which is what makes the
    speaker test mean what it says.
    """
    found = [match.group(1) for match in SPEAKER.finditer(text)]
    found.extend(role for role, _body in _structured_pairs(text))
    return found


def turn_count(text):
    return len(turns(text))


def speakers(text):
    """Distinct speaker labels, most frequent first."""
    seen = {}
    for name in turns(text):
        seen[name] = seen.get(name, 0) + 1
    return sorted(seen, key=lambda name: (-seen[name], name))


def turn_share(text):
    """Fraction of non-empty lines that are speaker turns.

    This is the measurement that separates dialogue from prose. Counting
    speaker labels alone is not enough: a novel can contain dozens of labelled
    lines and still be a novel, and a document with a chat template dumped into
    it scores high on raw label count while being the opposite of dialogue.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    # Clamped, because the two layouts are counted together now and a structured
    # document is often one enormous line holding many turns. A share is a
    # fraction by definition, and a reported value of 2.0 would be read as a
    # document twice made of turns, which is not a thing.
    return round(min(1.0, len(turns(text)) / len(lines)), 4)


def template_artifacts(text):
    """Chat-template markers found, which are contamination rather than dialogue."""
    return [match.group(0).strip() for match in TEMPLATE_MARKERS.finditer(text)]


def template_share(text):
    """Fraction of non-empty lines that are chat-template markers.

    A raw count is not enough here for the same reason turn_share is not enough
    for dialogue: a document can carry a great many markers and still not be a
    leaked template. The word "user" followed by a colon is ordinary English in
    any writing about interfaces, permissions or agents, and a technical
    document set in a conversation about such things can carry dozens of the
    lines in a few million.

    What a dumped conversation template looks like is not a document that
    contains markers but a document that is made of them. So this measures the
    share, and the refusal test uses the share rather than the count, with the
    count kept as a floor so a handful of markers in a short document still
    counts.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return round(len(template_artifacts(text)) / len(lines), 6)


def is_conversational(text, min_turns=MIN_TURNS, min_speakers=MIN_SPEAKERS):
    """Whether a document is dialogue rather than prose.

    Three conditions, because each alone is fooled: enough turns to be more
    than a stray label, more than one speaker, and turns making up a real share
    of the document. All three are counting, not understanding, and this is
    used to report what the corpus contains rather than to decide what is true
    about the text.

    The share test has two forms because the two layouts do not share a unit. A
    labelled transcript is measured per line, since a turn is a line. Structured
    records are measured per record, since a record is a JSON object that may
    occupy one line or twenty. Using the line share on structured data reports
    every conversational record as worth nothing, which is how two hundred and
    fifty-nine megabytes of instruction data came to be counted as prose.
    """
    if template_artifacts(text):
        return False
    found = turns(text)
    if len(found) < min_turns or len(set(found)) < min_speakers:
        return False
    structured = structured_turn_share(text)
    if structured > 0.0:
        return structured >= MIN_TURN_SHARE
    return turn_share(text) >= MIN_TURN_SHARE


def format_turn(speaker, text):
    """One turn in the layout a conversation is trained and spoken in.

    The trailing newline is part of the contract rather than an accident: every
    caller that builds a conversation from turns writes them straight through, so
    a missing newline joins two turns onto one line and destroys the very
    boundary the layout exists to express.
    """
    body = " ".join(str(text).split())
    return f"{speaker}: {body}" + "\n"


# The two positions a conversation has, in the order they alternate. Training
# alternates them rather than detecting them by name, because a corpus names its
# speakers whatever it likes and the model is spoken with these two names only.
_ROLE_ORDER_DEFAULT = (HUMAN, JUPITER)


def render_for_training(
    text, human=HUMAN, jupiter=JUPITER, role_order=_ROLE_ORDER_DEFAULT
):
    """Lay a conversation out as the text the model is trained to produce.

    Curation was able to say a document was dialogue and then learned from it as
    if it were prose: the speaker labels survived into the byte stream and
    nothing marked where one turn ended and the next began. A model trained on
    that learns the characters of a transcript rather than the shape of taking a
    turn in one, which is why the conversation sample worked before the corpus
    did.

    The rendering is the same one sample.py speaks in, and that is the whole
    point: the two have to agree, or the model is trained on a format it is never
    asked to produce. A document that is not dialogue is returned unchanged,
    because guessing at speakers in prose is the semantic decision curation says
    it has no business making.
    """
    if not is_conversational(text):
        return text
    if structured_turn_share(text) > 0.0:
        # A structured record is laid out from its parsed pairs rather than by
        # scanning lines, because the roles are inside JSON and there is no line
        # to match against. This is the format the model is actually spoken in
        # once the corpus is in this shape, so training it on the raw object
        # would teach it to emit braces and quoted keys instead of an answer.
        lines = [
            format_turn(JUPITER if role == "assistant" else HUMAN, body).rstrip(chr(10))
            for role, body in _structured_pairs(text)
        ]
        return "\n".join(lines) if lines else text
    roles = (human, jupiter) if role_order is _ROLE_ORDER_DEFAULT else tuple(role_order)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = SPEAKER.match(line)
        if match is None:
            # A line inside a turn is that turn's text, not a new one.
            if lines:
                lines[-1] = f"{lines[-1]} {stripped}"
            continue
        body = stripped[match.end() :].strip()
        # Roles are assigned by position rather than by name, because a corpus
        # of transcripts names its speakers whatever it likes: Alice and Bob, or
        # Question and Answer, or a user handle and a company. Two positions is
        # what a conversation has, and the model is spoken with exactly those two
        # names.
        #
        # Matching on names instead produced a transcript in which every turn
        # was addressed to the same speaker, because none of the names in a
        # general corpus happen to be on the list of assistant names. Training
        # on that teaches a monologue, which is the opposite of what a
        # conversation is.
        position = roles[len(lines) % 2]
        # rstripped because the lines are joined below, and
        # format_turn already carries a newline for callers
        # that write turns straight through.
        lines.append(format_turn(position, body).rstrip(chr(10)))
    return "\n".join(lines)


def training_turns(text):
    """How many turns a conversation contributes once rendered.

    Reported rather than assumed, so the curation report can say how much
    conversational material the corpus actually holds rather than how many
    documents merely contained some labelled lines.
    """
    if not is_conversational(text):
        return 0
    if structured_turn_share(text) > 0.0:
        # Two turns per record: the question and the reply it was answered with.
        return 2 * len(RECORD_INPUT.findall(text))
    return sum(1 for line in text.splitlines() if SPEAKER.match(line) and line.strip())


def report(text):
    """What this module can say about a document, stated as measurement."""
    found = turns(text)
    addressed = role_turns(text)
    return {
        "turns": len(found),
        "speakers": speakers(text),
        # Who the document addresses, as opposed to who it counts as speaking.
        # For a labelled transcript the two are the same thing and this is
        # empty; for structured records the replies are turns that no speaker
        # label in the input ever named, and reporting only the addressed roles
        # would describe every instruction document as a monologue.
        "addressed_roles": sorted(set(addressed)),
        "turn_share": turn_share(text),
        "structured_turn_share": structured_turn_share(text),
        "template_artifacts": len(template_artifacts(text)),
        "template_share": template_share(text),
        "conversational": is_conversational(text),
        # Whether the turns came from labelled lines or from structured records.
        # Reported because a corpus can hold both and the two are laid out
        # differently, so a reader comparing two documents needs to know which
        # reading produced the number above.
        "layout": (
            "structured"
            if structured_turn_share(text) > 0.0
            else ("labelled" if turn_count(text) else "prose")
        ),
    }
