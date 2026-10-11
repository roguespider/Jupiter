"""Structured raw sources: the Parsing stage of Architecture 25.

Architecture 25 puts Parsing between Raw Data and Cleaning, and this is it for
the suffixes curation used to step over. A .json file dropped in data/raw was
invisible: not accepted, not refused, not counted, and not mentioned in the
report. The obvious workaround was worse than the silence. Renaming the file to
.txt got it read, but as raw punctuation -- braces, quotes and key names were
trained on verbatim, and the contamination check passed it, because a JSON
payload carries no [SYSTEM]: marker to count.

So the file is parsed and the text is taken out of it. What that text looks like
depends on the record shape, and the shapes worth supporting are few enough to
name:

    {"speaker": "Alice", "text": "..."}          one turn
    {"role": "user", "content": "..."}           one turn, role mapped to the
                                                label dialogue.py speaks in
    {"messages": [ ...turns... ]}                a conversation
    {"conversations": [{"from":..., "value":...}]}  a ShareGPT export
    ["first paragraph", "second paragraph"]      paragraphs

Anything else is not guessed at. A record whose fields this module does not
recognise is counted as unrecognized and reported, because a parser that
invents text for a shape it has not read is how a corpus fills with material
nobody wrote. The count is the useful signal: a shape worth supporting shows up
as unrecognized, and adding it is a deliberate change rather than a surprise
after training has already happened on it.

Turns are rendered through dialogue.format_turn, and each turn is flattened to
a single line. The flattening is load-bearing, not cosmetic: dialogue.turn_share
counts turns against non-empty lines, so a turn written across six lines scores
as six lines of prose and the document is filed as prose when it is not.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import dialogue

# The field a turn's words live in, in priority order. A record is read from the
# first of these that holds a non-empty string.
TEXT_KEYS = (
    "content",
    "text",
    "message",
    "value",
    "line",
    "utterance",
    "body",
    "prompt",
    "input",
    "question",
    "answer",
    "completion",
    "output",
    "response",
    "description",
    "summary",
)

# The field naming who is speaking, in priority order.
SPEAKER_KEYS = ("role", "speaker", "name", "author", "from", "sender")

# The field holding a list of turns, in priority order. OpenAI-style exports
# nest the conversation under one of these rather than using the top level.
TURN_LIST_KEYS = (
    "messages",
    "conversations",
    "conversation",
    "turns",
    "dialog",
    "dialogue",
    "chat",
    "exchanges",
)

# Role values mapped onto the labels dialogue.py already speaks in. This is the
# mapping sample.py uses at inference time, so a role that is mapped here
# produces the same label the model is trained to expect and asked to produce.
# A role that is not mapped keeps its own name, which is deliberate: a
# "system" turn stays "System", and a document full of those is recognised by
# the contamination check as a leaked conversation template rather than being
# quietly laundered into clean-looking dialogue.
ROLE_LABELS = {
    "user": dialogue.HUMAN,
    "human": dialogue.HUMAN,
    "you": dialogue.HUMAN,
    "assistant": dialogue.JUPITER,
    "gpt": dialogue.JUPITER,
    "bot": dialogue.JUPITER,
    "model": dialogue.JUPITER,
    "jupiter": dialogue.JUPITER,
}

LINE_BREAKS = re.compile(r"\s*\n\s*")
# Runs of spaces and tabs, collapsed the way curate_dataset.normalize collapses
# them, so a turn flattened here and a document normalized there come out the
# same and the two paths cannot disagree about what a document is.
RUNS = re.compile(r"[ \t]{2,}")

SUFFIXES = (".json", ".jsonl", ".ndjson")
LINE_SUFFIXES = {".jsonl", ".ndjson"}

# A record separator used by sharded instruction exports: one JSON object per
# sample, introduced by a banner line. 13,867 of these appear in a single
# 63MB file in this project's own corpus, so this is not a hypothetical shape.
SAMPLE_BANNER = re.compile(r"(?m)^--- SAMPLE \d+ ---[ \t]*$")

# Fields holding the prompt side of a prompt/completion record. A record shaped
# {"input": [{"role": ..., "content": ...}], "output": "..."} is a conversation
# with the answer already written, and reading only "output" trains the model to
# answer a question it was never shown.
PROMPT_KEYS = ("input", "instruction", "prompt", "question", "query")

# Fields holding the completion side of the same record.
COMPLETION_KEYS = ("output", "completion", "response", "answer", "target")


class StructuredSourceError(Exception):
    """A structured file could not be read as structured data at all."""


@dataclass
class Extraction:
    """What one structured file yielded, stated as counts rather than prose."""

    text: str = ""
    source_format: str = "json"
    container: str = "none"
    records: int = 0
    extracted: int = 0
    unrecognized: int = 0
    errors: list = field(default_factory=list)

    def summary(self):
        return {
            "format": self.source_format,
            "container": self.container,
            "records": self.records,
            "extracted": self.extracted,
            "unrecognized": self.unrecognized,
        }


def flatten(value) -> str:
    """One line of text, so a turn counts as one turn and not as six lines."""
    text = LINE_BREAKS.sub(" ", str(value)).replace("\t", " ")
    return RUNS.sub(" ", text).strip()


def is_supported(path) -> bool:
    """Whether this module reads a file with that name."""
    return Path(path).suffix.lower() in SUFFIXES


def pick_speaker(record):
    """Who is speaking, or None when the record does not say."""
    for key in SPEAKER_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            label = value.strip()
            return ROLE_LABELS.get(label.lower(), label)
    return None


def pick_text(record):
    """The words in a record, or None when this module cannot find them.

    A record with exactly one string value is read as prose even when the field
    name is unrecognised, because a single string in a record is the text far
    more often than it is a label. A record with several unrecognised strings
    is left alone: choosing between them would be a guess, and a guess here
    becomes training material.
    """
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return flatten(value)
    strings = [value for value in record.values() if isinstance(value, str)]
    if len(strings) == 1 and strings[0].strip():
        return flatten(strings[0])
    return None


def label_for(speaker: str) -> str:
    """A speaker label dialogue.SPEAKER can recognise.

    The recogniser requires a capital, so an exported "alice" would otherwise
    be filed as prose. Capitalising the first letter changes no content; it
    makes the label-then-text pattern the thing a model actually learns.
    """
    return speaker[:1].upper() + speaker[1:]


def render_record(record):
    """One record as a line of text, or None when it cannot be read."""
    if isinstance(record, str):
        return flatten(record) or None
    if not isinstance(record, dict):
        return None
    speaker = pick_speaker(record)
    text = pick_text(record)
    if text is None:
        return None
    if speaker is not None:
        return dialogue.format_turn(label_for(speaker), text).strip()
    return text


def records_from(payload):
    """The records inside a parsed payload, and the container they came from."""
    if isinstance(payload, list):
        return payload, "list"
    if isinstance(payload, dict):
        for key in TURN_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list) and value:
                return value, f"object/{key}"
        completion = pick_completion(payload)
        if completion is not None:
            prompt = pick_prompt_turns(payload)
            return [*prompt, completion], "object/prompt-completion"
        return [payload], "object"
    return [payload], "scalar"


def parse_lines(text: str, name: str):
    """One JSON value per line, keeping going past a line that will not parse.

    A single corrupt line in a ten-thousand-line export should not cost the
    other nine thousand nine hundred and ninety-nine, so a bad line is recorded
    and skipped rather than raised. The errors are reported, so nothing is lost
    quietly.
    """
    values = []
    errors = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as error:
            errors.append(
                {
                    "line": number,
                    "reason": f"{name} line {number} is not valid JSON: {error.msg}",
                }
            )
    return values, errors


def extract_text(records):
    """A block of records as text, with what was read and what was not.

    Returned as a triple rather than a string because the counts are the point.
    A caller that only kept the text would not be able to report that records
    were dropped, and a dropped record is exactly what needs saying out loud.
    """
    lines = []
    extracted = 0
    unrecognized = 0
    for record in records:
        line = render_record(record)
        if line is None:
            unrecognized += 1
            continue
        extracted += 1
        lines.append(line)
    return "\n".join(lines), extracted, unrecognized


def pick_completion(record):
    """The answer side of a prompt/completion record, as a turn, or None.

    Read separately from pick_text because these two fields hold different
    things. "output" is in TEXT_KEYS so that a record whose only content is an
    output is still read, but on a prompt/completion record it must be rendered
    as Jupiter's turn rather than as unattributed prose, or the model learns to
    emit answers that follow no question.
    """
    if not isinstance(record, dict):
        return None
    for key in COMPLETION_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return {
                "role": dialogue.JUPITER,
                "content": flatten(value),
            }
    return None


def pick_prompt_turns(record):
    """The question side of a prompt/completion record, as a list of turns.

    The prompt may be a list of already-shaped turns or a single string. Both
    are returned in the form render_record understands, so a prompt/completion
    record and a native turn list reach the renderer by the same road.
    """
    if not isinstance(record, dict):
        return []
    for key in PROMPT_KEYS:
        value = record.get(key)
        if isinstance(value, list) and value:
            return [turn for turn in value if render_record(turn) is not None]
        if isinstance(value, str) and value.strip():
            return [{"role": dialogue.HUMAN, "content": flatten(value)}]
    return []


def looks_structured(text):
    """Whether text is a JSON payload whatever the file is named.

    The suffix check was the whole of the previous rule, and it was wrong for
    this project's largest files: 63.6% of the corpus arrived as .txt holding
    sharded JSON instruction data, so it was read as raw punctuation and the
    model spent its training budget learning braces and key names.

    Three shapes qualify, because three shapes are actually in this corpus:

    A sample banner anywhere in the text, which is how a shard announces itself.

    A leading JSON object or array that parses as a whole document.

    A leading fragment of one. Shards are cut at byte offsets, so the first
    record of a shard is routinely truncated and the file begins mid-object with
    `"output": "...` rather than with a banner or a brace. Only a file that ends
    without a closer is excluded, and that test is deliberately weak: it is
    there to keep a novel that mentions JSON from being routed through a parser,
    not to prove the file is a payload. The parser then decides, and decides per
    record, so a wrong answer here costs one wasted read rather than a corpus
    trained on punctuation.
    """
    if not text:
        return False
    if SAMPLE_BANNER.search(text):
        return True
    stripped = text.lstrip()
    if not stripped:
        return False
    # Checked before the brace test, because a shard cut mid-record opens with a
    # quoted key rather than with a delimiter. Ordering these the other way round
    # rejected exactly the file this was written for.
    if _opens_like_a_fragment(stripped):
        return True
    if stripped[0] not in "{[":
        return False
    try:
        json.loads(text)
        return True
    except ValueError:
        pass
    return stripped.rstrip().endswith(("}", "]"))


# A truncated shard begins with a quoted key rather than a brace, because the
# opening brace was in the previous file. This is the shape that distinguishes
# "a shard of JSON" from "prose that happens to mention a field name".
_FRAGMENT_START = re.compile(r'^\s*"[^"]{1,64}"\s*:')


def split_banners(text):
    """The JSON chunks between sample banners.

    A shard that begins mid-record, because the previous file was cut at a
    different point, yields a first chunk that will not parse. That chunk is
    reported as an error and dropped rather than failing the whole file, which
    is what keeps 11,742 usable samples out of one 63MB file instead of
    refusing all of them over a partial first line.
    """
    chunks = [chunk for chunk in SAMPLE_BANNER.split(text) if chunk.strip()]
    return chunks


def parse_concatenated(raw, name):
    """Every JSON value in a stream of them, found by brace matching.

    Sharded exports are cut at byte offsets, so a shard opens mid-record and no
    amount of splitting on a delimiter recovers the boundary. Brace matching
    does: it walks the text counting delimiters outside string literals and
    yields each complete top-level value.

    The leading fragment is skipped rather than repaired, because a truncated
    JSON object cannot be reconstructed without inventing the part that was cut
    off. It is reported, so what was lost is visible.
    """
    values = []
    errors = []
    depth = 0
    start = 0
    first_start = 0
    in_string = False
    escaped = False

    for index, character in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "{[":
            if depth == 0:
                start = index
                if not values and not errors:
                    first_start = index
            depth += 1
        elif character in "}]":
            if depth == 0:
                # Stray closer, which is what the tail of a cut record looks
                # like. Skipped; it belongs to a value this shard no longer has.
                continue
            depth -= 1
            if depth == 0:
                chunk = raw[start : index + 1]
                try:
                    values.append(json.loads(chunk))
                except json.JSONDecodeError as error:
                    errors.append(
                        {
                            "offset": start,
                            "reason": f"{name} record at offset {start} is not "
                            f"valid JSON: {error.msg}",
                        }
                    )

    if depth != 0:
        # The file ends mid-record. Named rather than silently dropped, because
        # an unterminated value is the one thing that means data was cut.
        errors.append(
            {
                "reason": f"{name} ends inside an unterminated JSON value; the "
                "trailing fragment is incomplete and was not read"
            }
        )
    # Whether the file opened mid-record. The test is what precedes the first
    # value, not the offset of the last one: start is rewritten by every record
    # the walk finds, so it says where parsing ended up rather than whether
    # anything was skipped at the front.
    if first_start > 0 and raw[:first_start].strip():
        errors.insert(
            0,
            {
                "offset": 0,
                "reason": f"{name} opens mid-record; the leading fragment is "
                "incomplete and was not read",
            },
        )
    return values, errors


def parse_chunks(chunks, name):
    """Each chunk parsed as one JSON value, keeping going past a bad one."""
    values = []
    errors = []
    for number, chunk in enumerate(chunks, start=1):
        try:
            values.append(json.loads(chunk))
        except json.JSONDecodeError as error:
            errors.append(
                {
                    "sample": number,
                    "reason": (
                        f"{name} sample {number} is not valid JSON: {error.msg}"
                    ),
                }
            )
    return values, errors


def _opens_like_a_fragment(stripped):
    """Whether text opens as the truncated middle of a JSON record.

    Deliberately weak, and paired with the parser rather than trusted alone. A
    prose document that opens with a quoted label and a colon is rare but not
    impossible, and when it happens the parser finds no records and the caller
    reads the file as prose, which is where it would have gone anyway.
    """
    return bool(_FRAGMENT_START.match(stripped[:80]))


def load(path):
    """Read a structured file into (payloads, format, errors)."""
    target = Path(path)
    raw = target.read_text(encoding="utf-8", errors="replace")
    if target.suffix.lower() in LINE_SUFFIXES:
        values, errors = parse_lines(raw, target.name)
        return values, "jsonl", errors
    try:
        return [json.loads(raw)], "json", []
    except json.JSONDecodeError as error:
        # Not one JSON document. Before concluding the file is unreadable,
        # check whether it is a shard, which is the shape this project's own
        # corpus uses and which was being read as raw prose. Two shard shapes
        # exist and both have to be tried: one announced by banners, and one cut
        # mid-record so that the banner announcing its first record is in the
        # previous file.
        chunks = split_banners(raw)
        if len(chunks) > 1:
            values, errors = parse_chunks(chunks, target.name)
            if values:
                return values, "json-banners", errors
        values, errors = parse_concatenated(raw, target.name)
        if values:
            return values, "json-concatenated", errors
        raise StructuredSourceError(
            f"{target.name} is not valid JSON: {error.msg} "
            f"at line {error.lineno} column {error.colno}"
        ) from error


def extract_tolerant(path) -> Extraction:
    """Extract what can be read, and report the rest, without raising.

    Strict extraction is right for a file that claims a suffix it does not
    honour, and wrong for a file whose head looks structured and whose body is
    partly not. Jupiter's own generated material is the case that matters here:
    a synthesized file opens with the prompt it was seeded from, in the same
    banner-delimited JSON as the corpus it came from, and then continues into
    whatever the model produced. Some of those files parse as JSON and some do
    not, and refusing the unparseable ones throws away readable material on the
    strength of the tail rather than the head.

    Returning the partial result lets the caller decide, and the counts are what
    make that decision honest: records > 0 means something was read, records == 0
    with errors means the file really was unreadable, and neither means the file
    was never structured and should be read as prose.
    """
    target = Path(path)
    raw = target.read_text(encoding="utf-8", errors="replace")

    values, source_format, errors = [], "none", []
    if target.suffix.lower() in LINE_SUFFIXES:
        values, errors = parse_lines(raw, target.name)
        source_format = "jsonl"
    else:
        try:
            values, source_format, errors = [json.loads(raw)], "json", []
        except json.JSONDecodeError:
            chunks = split_banners(raw)
            if len(chunks) > 1:
                values, errors = parse_chunks(chunks, target.name)
                source_format = "json-banners"
            else:
                # A shard cut at a byte offset opens mid-record, so the banner
                # announcing its first record is in the previous file and
                # splitting on banners finds nothing. Reading the body as one
                # stream of concatenated objects recovers every intact record.
                values, errors = parse_concatenated(raw, target.name)
                source_format = "json-concatenated"
                if not values:
                    # Not structure at all despite appearances. Not an error:
                    # this is a file that only looked structured, and the caller
                    # is told so by an empty record count rather than by an
                    # exception.
                    return Extraction(source_format="none", container="none")

    result = Extraction(source_format=source_format, errors=list(errors))
    records = []
    for payload in values:
        found, container = records_from(payload)
        if container != "none":
            result.container = container
        records.extend(found)
    result.records = len(records)
    result.text, result.extracted, result.unrecognized = extract_text(records)
    return result


def extract(path) -> Extraction:
    """Every readable record in a structured file, as one block of text."""
    payloads, source_format, errors = load(path)
    result = Extraction(source_format=source_format, errors=list(errors))
    records = []
    for payload in payloads:
        found, container = records_from(payload)
        if container != "none":
            result.container = container
        records.extend(found)
    result.records = len(records)
    result.text, result.extracted, result.unrecognized = extract_text(records)
    return result
