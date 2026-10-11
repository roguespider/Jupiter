"""Conversation experience, turned into learning material.

Architecture 4 is titled "Experience Creates Learning", and the chain it draws is
conversation -> experience -> memory -> curation -> learning. Three of those four
seams existed. The fourth did not: sample.py wrote out/experiences.jsonl and
nothing in the project ever read it. A tree-wide search for that filename returns
exactly one file, the one that writes it.

So the loop was open. Every conversation Jupiter had was recorded, and every
recorded conversation was then invisible to the learning system that exists to
learn from it. That is not a small gap. It is the mechanism Architecture 51 names
when it says "preserve experience", and it was the reason the corpus only ever
grew from material a person placed in data/raw.

This module is the reader.

It is a reader and not a writer for a reason worth stating. Experience is
produced by the conversation runtime, which knows what was actually said and
what was actually answered. Learning is produced by the cycle, which knows what
was actually measured. Neither should invent the other's facts. This module reads
what the conversation recorded, renders it in the layout curation and inference
already agree on, and writes it where curation will find it as ordinary raw text.

Two properties are deliberate.

It does not bypass curation. Rendered experience is written into the raw
directory, so it is parsed, cleaned, classified and deduplicated exactly as a
human file is, and refused if it is contaminated or too short. A self-feeding
loop is the last place that should have a private path, and the acceptance gate
downstream still judges anything learned from it on its evidence.

It never writes a turn it was not given. A conversation record with an empty
reply is not a half-written turn: an assistant that said nothing has produced no
training signal, and padding it out would teach the model to produce nothing in
exchange for being asked. Those records are counted and reported rather than
rendered.
"""

import json
import time
from pathlib import Path

import dialogue

# Where the conversation runtime records what was said. Read rather than
# duplicated, so a second writer cannot create a second definition of where
# experience lives.
EXPERIENCES_NAME = "experiences.jsonl"

# Rendered experience is written here, which is the directory synthesize.py
# already writes generated material into. Sharing the directory is deliberate:
# both are text Jupiter wrote about itself, and curation treats them identically.
GENERATED_SUBDIR = "generated"

# A reply shorter than this is not language. Counting it as a turn would teach
# the model that a short non-answer is a correct shape of answer.
MIN_REPLY_CHARS = 12

# The marker that identifies a file this module rendered. Written into the file
# so a later run can tell its own output from material a person placed in the
# directory, which is what makes the rewrite safe.
GENERATED_MARKER = "generated-experience"


def experiences_path(c):
    """Where the conversation runtime records interactions."""
    return Path(getattr(c, "out_dir", "out")) / EXPERIENCES_NAME


def generated_dir(c):
    """Where rendered experience is written, for curation to pick up."""
    return Path(getattr(c, "raw_dir", "data/raw")) / GENERATED_SUBDIR


def read(c, limit=None):
    """Every recorded interaction, oldest first.

    Read defensively for the same reason autonomy.load is: this is a file
    another process appends to, and a loop that cannot start because its own
    memory has a torn last line is a loop that cannot recover from anything.

    A line that will not parse is counted and skipped rather than raised, and
    the count is returned so a caller can say how much of the file it did not
    understand instead of quietly learning from the part it could.
    """
    path = experiences_path(c)
    if not path.exists():
        return [], 0
    records = []
    unreadable = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            unreadable += 1
            continue
        if not isinstance(record, dict):
            unreadable += 1
            continue
        if str(record.get("type", "")).lower() != "conversation":
            continue
        records.append(record)
    if limit is not None:
        records = records[-int(limit) :]
    return records, unreadable


def render(record):
    """One interaction as the turns curation and inference both speak in.

    Returns None when the record cannot be rendered honestly, which is the case
    for a conversation that never got an answer. The alternative is to write the
    question with nothing after it, and a model trained on that learns to stop.
    """
    question = " ".join(str(record.get("user", "")).split())
    reply = " ".join(str(record.get("jupiter", "")).split())
    if not question or len(reply) < MIN_REPLY_CHARS:
        return None
    return dialogue.format_turn(dialogue.HUMAN, question) + dialogue.format_turn(
        dialogue.JUPITER, reply
    )


def render_all(records):
    """The renderable subset of a conversation log, and what was left out."""
    turns = []
    skipped = 0
    for record in records:
        line = render(record)
        if line is None:
            skipped += 1
            continue
        turns.append(line)
    return turns, skipped


def target(c):
    """The file rendered experience is written to."""
    directory = generated_dir(c)
    return directory / f"{GENERATED_MARKER}.txt"


def publish(c, limit=None):
    """Render recorded experience into the raw directory, for curation to read.

    Returns what was written and what was left out, so the caller can report it.
    Nothing is written when there is nothing renderable, because an empty file in
    the raw directory is a document curation has to classify and reject.

    The write is a whole-file replace rather than an append. The file is derived
    entirely from the conversation log, so rewriting it is idempotent: running
    this twice produces the same file rather than doubling every conversation.
    That matters because this is called on a schedule, and an append would turn
    a hundred runs into a hundred copies of the same experience.
    """
    records, unreadable = read(c)
    turns, skipped = render_all(records)
    path = target(c)
    if not turns:
        return {
            "written": 0,
            "skipped": skipped,
            "unreadable": unreadable,
            "path": str(path),
            "note": (
                "no interaction in the conversation log could be rendered, so "
                "nothing was written; a conversation with no reply is not "
                "learning material"
            ),
        }

    body = "\n".join(turns)
    document = (
        f"# {GENERATED_MARKER}: {len(turns)} conversation(s) recorded by "
        f"Jupiter's own runtime.\n"
        f"# written {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# {skipped} record(s) had no usable reply and were not rendered.\n"
        f"{body}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return {
        "written": len(turns),
        "skipped": skipped,
        "unreadable": unreadable,
        "path": str(path),
        "characters": len(body),
    }


def census(c):
    """What the conversation log currently holds, for a report to state."""
    records, unreadable = read(c)
    turns, skipped = render_all(records)
    return {
        "interactions": len(records),
        "renderable": len(turns),
        "no_reply": skipped,
        "unreadable_lines": unreadable,
        "source": str(experiences_path(c)),
        "rendered_to": str(target(c)),
        "note": (
            "an interaction counts as learning material only when it has both a "
            "question and a reply worth reading; the rest are counted here "
            "rather than padded into training text"
        ),
    }
