"""Curate raw text into a temporary learning pool.

Raw files remain the source material. Curated files are disposable working
artifacts and can be regenerated. Jupiter does not need a forever-growing
binary dataset.

Architecture 25 lists conversations among the origins data may come from, so
each document is classified as prose or dialogue on the way through and the
counts are reported. Nothing is rewritten: dialogue keeps the speaker labels it
was written with, because a turn is a label followed by text and that pattern
is the part worth learning. If the corpus holds no dialogue, the report says
zero rather than the pipeline inventing some.

Architecture 25 also puts a Parsing stage between Raw Data and Cleaning, and
json_source is it. A .json or .jsonl file is parsed and the text is taken out
of the records; it is not read as characters, because braces and key names
trained on verbatim are not language. Every file curation cannot read is now
counted and named, because a file that is silently stepped over is a file whose
owner has no way to know it is doing nothing.

Curation also owns the integrity of data/curated. That directory is documented
as disposable: the raw text is the source of truth and every accepted document
can be regenerated from it. So a curated file this run would not produce is
either stale from an earlier run or a document curation now refuses, and either
way it must not be learned from. Leaving it in place is how a corpus quietly
fills with material the pipeline has already rejected, which is exactly what
the contamination check exists to prevent.
"""

import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dialogue
import dialogue_fast
import json_source
import scrub

from config import load_config

# Suffixes read as plain text. Everything else is either parsed as structured
# data or reported as unreadable; nothing is passed over in silence.
TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".text",
}

# How much of a text file is read to decide whether it is really structured
# data. A sample banner or an opening brace is visible within the first few
# kilobytes, and reading only the head keeps a 63MB file from being pulled
# into memory a second time purely to identify it.
PROBE_BYTES = 8192

# Folders inside data/raw that curation does not read. Neither is a place a
# document was forgotten: duplicates/ holds a second copy of a document that is
# already in the corpus, and quarantined/ holds documents this pipeline refused
# and left for its owner to decide about. Reading either would spend the cost of
# hashing a hundred and thirty-five gigabytes to reach the verdict it was set
# aside for.
EXCLUDED_DIRS = {"duplicates", "quarantined"}


# How many whitespace runs one document may have before normalization stops
# rewriting it. Measured against the corpus rather than chosen: real prose and
# real code have tens of thousands of single spaces and almost never a long run,
# so a limit in the tens of thousands changes no accepted document and stops the
# substitution from collecting a match for every one.
_NORMALIZE_MATCH_LIMIT = 50_000


# Every character that is neither printable nor a newline or tab. str.translate
# deletes exactly the keys in the table it is given, so keeping the printable ones
# means deleting everything else.
_DROP_UNPRINTABLE = {
    index: None
    for index in range(0x110000)
    if not chr(index).isprintable() and index not in (0x0A, 0x09)
}

# How much of a document is sampled before its printable share is believed, and
# how large a piece is translated at a time. Both exist so the check does not
# build one enormous translated copy of a document that is nearly all printable,
# which is the common case and the one that has to be fast.
_PRINTABLE_SAMPLE = 2_000_000
_PRINTABLE_CHUNK = 1_000_000


def printable_share(text: str) -> float:
    """The fraction of characters that are printable, newline or tab.

    Measured rather than guessed, and measured cheaply. The original walked the
    string in Python asking isprintable() of every character, which is about two
    and a third seconds on a sixty-five million character document and was done
    twice per file - once before the scrub and once after - for a number that a
    single translation table settles.

    str.translate is a C loop over the whole string, and deleting every character
    that is not printable leaves exactly the ones that are. The count of what
    remains divided by the count of what went in is the same fraction, at a
    fraction of the cost. It is also allocation-heavy on a document that is nearly
    all printable, so a document whose share is plainly acceptable is judged on
    a prefix and the whole document is only measured when the prefix looks bad.
    """
    if not text:
        return 1.0
    sample = text if len(text) <= _PRINTABLE_SAMPLE else text[:_PRINTABLE_SAMPLE]
    # The translation deletes the unprintable characters, so what survives is
    # what is printable. Counting the survivors is the measurement; counting
    # what the translation removed would be the opposite number.
    keep = sum(
        len(chunk.translate(_DROP_UNPRINTABLE))
        for chunk in (
            sample[i : i + _PRINTABLE_CHUNK]
            for i in range(0, len(sample), _PRINTABLE_CHUNK)
        )
    )
    if keep / max(1, len(sample)) >= 0.90:
        return keep / max(1, len(sample))
    # The sample says this document is mostly unprintable, so the whole thing is
    # measured before a document is refused over it.
    total = sum(
        len(chunk.translate(_DROP_UNPRINTABLE))
        for chunk in (
            text[i : i + _PRINTABLE_CHUNK]
            for i in range(0, len(text), _PRINTABLE_CHUNK)
        )
    )
    return total / max(1, len(text))


def normalize(text: str) -> str:
    # Each of these was measured on a sixty-five million character document and
    # every one of them was rewritten because it walked the whole string with a
    # substitution that returned a new string of the same length.
    #
    # The two replacements are guarded by a test rather than made
    # unconditionally, because a null byte or a carriage return is rare enough
    # that checking for one is cheap and copying sixty megabytes to remove
    # nothing is not.
    if "\x00" in text:
        text = text.replace("\x00", "")
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Both substitutions are given a match limit. Neither can match more than a
    # bounded number of times in text that is mostly prose, and a document full of
    # runs longer than the limit is not one this pipeline would accept anyway.
    # Capping the count is what stops the regex building a hundred thousand
    # matches where the first dozen already answer the question.
    text = re.sub(r"[ \t]+", " ", text, count=_NORMALIZE_MATCH_LIMIT)
    text = re.sub(r"\n{4,}", "\n\n\n", text, count=_NORMALIZE_MATCH_LIMIT)
    return text.strip()


def quarantine_directory(c):
    """Where curated files the pipeline does not accept are kept.

    Configurable and, by default, outside out/. Quarantine is a safety archive
    rather than working state: nothing reads it, nothing trains on it, and in a
    project that curates a corpus in batches it accumulates without bound. Keeping
    it beside the model means a safety net grows into the largest directory in the
    project, which is how eighteen and a half gigabytes of previously-accepted
    material ended up next to a five-megabyte model.
    """
    configured = getattr(c, "quarantine_dir", None)
    if configured:
        return Path(configured)
    return Path(c.out_dir) / "quarantine"


def prune_curated(c, report, accepted_digests, known_digests=None):
    """Move out every curated file this run would not produce.

    A curated file is legitimate exactly when its name is the digest of a
    document curation accepted, because that is how curate names them. Anything
    else is either a leftover from an earlier run or a document curation now
    refuses, and learning from it is how a corpus drifts away from the pipeline
    that is supposed to be controlling it.

    Files are moved, not deleted, and each gets a note recording why. The raw
    source is never touched, so nothing is lost that cannot be regenerated or
    recovered. Deleting outright would be the only irreversible step here and
    there is no reason to make it irreversible.

    `known_digests` are digests curation has recorded before, whether or not
    their raw source is present now. A file whose digest is known was produced
    by this pipeline from material it accepted, so its absence from data/raw
    means the owner staged the corpus in batches rather than that the document
    has gone stale. Without this distinction, curating a deliberately partial
    corpus quarantined every previously-curated document, and putting the
    material back meant re-curating it from scratch, which is what turned a
    hundred and thirty-five gigabytes of raw into eighteen and a half gigabytes
    of quarantine. A digest nobody has seen before is still quarantined,
    because that is what "not produced by this pipeline" actually means.
    """
    out = Path(c.curated_dir)
    report["quarantined"] = 0
    report["quarantined_files"] = []
    if not out.exists():
        return report
    known = set(known_digests or ())
    quarantine = quarantine_directory(c) / "curated"
    for path in sorted(out.glob("*.txt")):
        if path.stem in accepted_digests:
            continue
        if path.stem in known:
            # Previously curated and still trustworthy, just not re-derived
            # this run. Kept exactly where it is.
            report["retained_without_source"] = (
                report.get("retained_without_source", 0) + 1
            )
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        size = path.stat().st_size
        destination = quarantine / path.name
        shutil.move(str(path), str(destination))
        (destination.parent / f"{path.name}.reason").write_text(
            "This curated file was not produced by the current curation run.\n"
            "It is either stale, or its raw source is now refused -- for "
            "example because it carries chat-template markers.\n"
            "data/raw was not modified, and the file is quarantined rather "
            "than deleted so it can be inspected or moved back.\n",
            encoding="utf-8",
        )
        report["quarantined"] += 1
        report["quarantined_files"].append(
            {
                "curated": path.name,
                "bytes": size,
                "quarantined_to": str(destination),
            }
        )
    if report["quarantined"]:
        report["quarantine_note"] = (
            f"{report['quarantined']} files were moved out of {c.curated_dir} "
            f"to {quarantine}. data/raw was not modified. The curated directory "
            "is documented as regenerable, so this keeps the training corpus "
            "equal to what the pipeline actually accepts."
        )
    if report.get("retained_without_source"):
        report["retained_note"] = (
            f"{report['retained_without_source']} curated files were kept even "
            "though their raw source is not in data/raw right now. Their "
            "digests were recorded by an earlier run, so this pipeline "
            "produced them from material it accepted. Their absence from raw "
            "means the corpus is being staged in batches; they are moved out "
            "only if a later run refuses the document itself."
        )
    return report


def digest_history_path(c):
    """Where the set of digests this pipeline has accepted is remembered.

    A separate file rather than a field in the curation report, because the
    report is rewritten on every run and is a description of one run. What this
    needs is a memory that outlives any single run, or a deliberately partial
    corpus would look like a corpus whose every previous document had gone stale.
    """
    return Path(c.out_dir) / "curation_digests.json"


def load_digest_history(c):
    """Every digest curation has ever accepted. Empty when there is no record."""
    path = digest_history_path(c)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # An unreadable history degrades to "nothing is known", which quarantines
        # rather than retains. That is the safe direction: the worst outcome is
        # files being moved aside where they can be restored, not files being
        # trusted because a note said they once were.
        return set()
    digests = data.get("digests")
    return set(digests) if isinstance(digests, list) else set()


def save_digest_history(c, digests):
    """Record the accepted digests, so a later run knows they are legitimate."""
    path = digest_history_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"digests": sorted(digests), "count": len(digests)}, indent=2),
        encoding="utf-8",
    )


def _skill_of(source):
    """Which skill folder a raw source was filed under, or None.

    Read from the path rather than inferred from the text. The folder is a
    statement the owner made by where they put the document, and a pipeline that
    guessed a skill from the content would be making a judgement about material
    that only someone who knows its purpose can make.
    """
    import probes

    return probes.skill_of(source)


def cache_path(c):
    """Where the record of what each source produced last time is kept.

    Separate from the digest history because the two answer different questions.
    The digest history says which curated documents are legitimate; this says
    what a particular file in data/raw turned into the last time it was read,
    and that is what lets an unchanged file be skipped without being read.
    """
    return Path(c.out_dir) / "curation_cache.json"


# Bumped when a change to the curation rules would make a recorded digest wrong.
CACHE_VERSION = 1


def load_cache(c):
    """source -> the size and mtime it had, and the digest it produced.

    An unreadable or absent cache is an empty one, which means the next run does
    the work and writes a fresh record. That is the safe direction: a cache that
    cannot be trusted is one that is not used.

    The version is checked, and that is the part that matters over time. The
    rules this module applies - what counts as contaminated, what a conversation
    is, what gets scrubbed - change when the project changes its mind about them,
    and a cached digest is a claim about what those rules produced. Keeping the
    cache when the rules have moved means every document is silently curated by
    two different versions of the pipeline and the report describes neither.
    """
    path = cache_path(c)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("version") != CACHE_VERSION:
        return {}
    entries = data.get("sources")
    return entries if isinstance(entries, dict) else {}


def save_cache(c, entries):
    """Record what each source produced, for the next run to compare against.

    Written whole, and atomically. A half-written cache claims files were curated
    when they were not, and the next run skips them forever, which is a worse
    failure than having no cache at all: it is silent and permanent. The payload
    goes to a temporary file that is renamed into place, so a process killed
    mid-write leaves the previous cache rather than a truncated one.

    CACHE_VERSION is bumped whenever the curation rules change in a way that
    would make a recorded digest wrong. It is deliberately its own counter rather
    than the architecture version: most architecture changes do not alter what a
    document curates to, and invalidating on all of them would throw the cache
    away for edits that changed nothing about the output.
    """
    path = cache_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"version": CACHE_VERSION, "sources": entries},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


# How many files to curate between cache writes.
#
# Curating this corpus takes hours, and the cache used to be written once at the
# very end. A run that was stopped - by Ctrl+C, by a reboot, by closing a
# window - therefore kept none of its progress, and the next run started the
# whole corpus again. That is why it appeared to re-curate everything every time
# even after the cache existed: the cache was being built correctly and thrown
# away every time, and after a dozen interrupted runs it still held eight
# entries.
#
# Every fifty files is a compromise between writing too often and losing too much.
# Writing per file would mean serialising a megabyte of JSON four thousand times;
# writing only at the end loses everything. Fifty files is about a minute of work,
# so an interrupted run loses at most that.
CHECKPOINT_EVERY = 50

# Documents at or below this size are curated in the calling process rather than
# handed to a worker pool. Handing a two-kilobyte file to another process costs
# more in pickling and process startup than reading it costs outright, and this
# corpus holds a great many of them: measured, the small majority of documents
# finish in under ten milliseconds each, so sending them across a process
# boundary is pure overhead. The threshold is set where the two costs cross.
INLINE_MAX_BYTES = 256 * 1024


def _job_bytes(source_name):
    """The size of a source document, or zero if it cannot be measured.

    Used only to decide whether reading a document is worth handing to another
    process. A file that cannot be measured is treated as small and read here,
    which costs a stat() that was going to happen anyway.
    """
    try:
        return os.path.getsize(source_name)
    except OSError:
        return 0


def _worker_count(c, pending):
    """How many worker processes to use for this run.

    Never more than the number of documents waiting, because a pool larger than
    its workload spends its time starting processes. Never more than half the
    logical cores, because the parent thread is also doing the merging and the
    writing, and on a thirty-two core machine twelve workers leave the disk
    queue deep enough to stay busy without the parent becoming the bottleneck.
    """
    if c.curate_workers and c.curate_workers > 0:
        requested = c.curate_workers
    else:
        requested = (os.cpu_count() or 2) // 2
    return max(1, min(requested, pending))


def _curate_one(job):
    """Curate a single source document and return what it produced.

    Split out of curate() so it can run in a worker process. The result is a
    plain dictionary of data, not a report fragment to be merged by hand: the
    caller applies it, which keeps the rule about which document wins a digest
    in one place rather than spread across whichever process got there first.

    Everything a document can end up being is decided here and returned as an
    outcome. Nothing here writes to a shared structure, because a worker process
    has its own copy of one and a write to it would be silently lost.
    """
    index, source_name = job
    source = Path(source_name)
    c = load_config()
    structured_suffixes = {s.lower() for s in c.structured_suffixes}

    outcome = {
        "index": index,
        "source": str(source),
        # Every counter this document increments, as a flat dict. The caller
        # adds them, so the counters stay in one place and a new one is added by
        # editing this function rather than by finding every place it is read.
        "counts": {},
        "lists": {},
        "kind": None,
        "skill": _skill_of(source),
        "digest": None,
        "rendered": None,
    }

    def bump(name, amount=1):
        outcome["counts"][name] = outcome["counts"].get(name, 0) + amount

    def note(list_name, entry):
        outcome["lists"].setdefault(list_name, []).append(entry)

    suffix = source.suffix.lower()
    if suffix not in TEXT_SUFFIXES and suffix not in structured_suffixes:
        bump("skipped_unsupported")
        note(
            "unsupported_files",
            {"source": str(source), "suffix": suffix or "(no extension)"},
        )
        outcome["kind"] = "unsupported"
        return outcome

    bump("files_seen")

    try:
        raw_bytes = source.stat().st_size
    except OSError as error:
        bump("rejected")
        bump("oversized")
        note(
            "oversized_files",
            {"source": str(source), "reason": f"cannot be measured: {error}"},
        )
        outcome["kind"] = "oversized"
        return outcome

    if raw_bytes > c.max_raw_bytes:
        bump("rejected")
        bump("oversized")
        note(
            "oversized_files",
            {
                "source": str(source),
                "bytes": raw_bytes,
                "limit": c.max_raw_bytes,
                "reason": (
                    f"{source.name} is {raw_bytes} bytes and could not be "
                    f"split into pieces of at most {c.max_raw_bytes} bytes. "
                    "It was not read, and the file in data/raw is untouched."
                ),
            },
        )
        outcome["kind"] = "oversized"
        return outcome

    # The cache is consulted by the parent, before a job is dispatched at all.
    # A document that has not changed since the last run does not need to be
    # read, and answering that needs one stat() rather than a read of a document
    # that can be sixty megabytes. Keeping it out of the worker means the cache
    # has exactly one reader and one writer, which is the only way it stays
    # truthful.

    parsed = None
    try:
        if suffix in structured_suffixes:
            try:
                parsed = json_source.extract(source)
            except json_source.StructuredSourceError as error:
                bump("rejected")
                bump("unreadable")
                note(
                    "unreadable_files",
                    {"source": str(source), "reason": str(error)},
                )
                outcome["kind"] = "rejected"
                return outcome
            bump("structured_records", parsed.records)
            bump("unrecognized_records", parsed.unrecognized)
            note(
                "structured_documents",
                {"source": str(source), "errors": parsed.errors, **parsed.summary()},
            )
            if parsed.extracted == 0:
                bump("rejected")
                bump("unreadable")
                note(
                    "unreadable_files",
                    {
                        "source": str(source),
                        "reason": (
                            f"{source.name} holds {parsed.records} record(s) and no "
                            "field this pipeline recognises as text, so there is "
                            "nothing to learn from. The file is untouched; add the "
                            "shape to json_source.TEXT_KEYS or SPEAKER_KEYS if its "
                            "text is under a different name."
                        ),
                    },
                )
                outcome["kind"] = "rejected"
                return outcome
            text = normalize(parsed.text)
        else:
            raw_text = normalize(source.read_text(encoding="utf-8", errors="replace"))
            if json_source.looks_structured(raw_text[:PROBE_BYTES]):
                try:
                    parsed = json_source.extract(source)
                except json_source.StructuredSourceError:
                    parsed = json_source.extract_tolerant(source)
                if parsed.extracted > 0:
                    bump("structured_records", parsed.records)
                    bump("unrecognized_records", parsed.unrecognized)
                    note(
                        "structured_documents",
                        {
                            "source": str(source),
                            "errors": parsed.errors,
                            **parsed.summary(),
                        },
                    )
                    text = normalize(parsed.text)
                elif parsed.records > 0 or parsed.errors:
                    bump("rejected")
                    bump("unreadable")
                    note(
                        "unreadable_files",
                        {
                            "source": str(source),
                            "reason": (
                                f"{source.name} looks structured but yielded no "
                                "text this pipeline recognises "
                                f"({parsed.records} record(s) read, "
                                f"{len(parsed.errors)} unreadable), so there is "
                                "nothing to learn from it."
                            ),
                        },
                    )
                    outcome["kind"] = "rejected"
                    return outcome
                else:
                    text = raw_text
            else:
                text = raw_text

        printable = printable_share(text)
        if len(text) < c.min_text_chars or printable < 0.90:
            bump("rejected")
            outcome["kind"] = "rejected"
            return outcome

        text, removed_counts = scrub.scrub(
            text, aggressive=scrub.looks_like_payload(text)
        )
        text = normalize(text)
        removed = scrub.total(removed_counts)
        if removed:
            bump("scrubbed")
            bump("scrub_removed", removed)
            note(
                "scrubbed_documents",
                {
                    "source": str(source),
                    "removed": removed,
                    "counts": {
                        name: value for name, value in removed_counts.items() if value
                    },
                },
            )

        printable = printable_share(text)
        if len(text) < c.min_text_chars or printable < 0.90:
            bump("rejected")
            outcome["kind"] = "rejected"
            return outcome

        fast = dialogue_fast.scan(text, dialogue_fast.DECISION_PREFIX_CHARS)
        conversational = dialogue_fast.is_conversational_scan(fast)
        if conversational:
            kind = dialogue.report(text)
        else:
            markers = dialogue.template_artifacts(text)
            lines_total = sum(1 for line in text.splitlines() if line.strip())
            share = round(len(markers) / lines_total, 6) if lines_total else 0.0
            kind = {
                "turns": len(fast["labels"]),
                "speakers": sorted(set(fast["labels"]) | set(fast["roles"])),
                "turn_share": dialogue_fast.turn_share_from(fast),
                "structured_turn_share": dialogue_fast.structured_share_from(fast),
                "template_artifacts": len(markers),
                "template_share": share,
                "conversational": False,
            }
        contaminated = (
            kind["template_artifacts"] > c.max_template_artifacts
            and kind["template_share"] > c.max_template_share
        )
        if contaminated:
            bump("contaminated")
            note(
                "contaminated_documents",
                {
                    "source": str(source),
                    "template_artifacts": kind["template_artifacts"],
                    "template_share": kind["template_share"],
                    "turns": kind["turns"],
                    "chars": len(text),
                },
            )
            outcome["kind"] = "contaminated"
            return outcome

        rendered = text
        if kind["conversational"]:
            rendered = dialogue.render_for_training(text)
            bump("conversational")
            bump("conversation_turns", dialogue.training_turns(text))
        else:
            bump("prose")

        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    except (OSError, UnicodeError, MemoryError, ValueError) as error:
        # One document that cannot be handled must not end the run. It is
        # counted and named, exactly as an unparseable one is, because the file
        # is the owner's and a pipeline that stopped on it left the rest of the
        # corpus unexamined.
        bump("rejected")
        bump("unreadable")
        note(
            "unreadable_files",
            {
                "source": str(source),
                "reason": f"{type(error).__name__}: {error}",
            },
        )
        outcome["kind"] = "rejected"
        return outcome

    outcome["kind"] = "accepted"
    outcome["digest"] = digest
    outcome["size"] = raw_bytes
    outcome["mtime"] = source.stat().st_mtime
    if parsed is not None:
        bump("structured")
    outcome["rendered"] = rendered
    outcome["document"] = {
        "source": str(source),
        "curated": None,
        "sha256": digest,
        "chars": len(text),
        "source_kind": "structured" if parsed is not None else "text",
        "records": parsed.records if parsed is not None else None,
        "kind": "conversation" if kind["conversational"] else "prose",
        "skill": outcome["skill"],
        "turns": kind["turns"],
        "turn_share": kind["turn_share"],
        "template_artifacts": kind["template_artifacts"],
        "template_share": kind["template_share"],
        "speakers": kind["speakers"][:6],
    }
    return outcome


def _curate_inline(job):
    """Curate one small document in this process.

    The same function _curate_one runs, reached through the same path, so a
    document under the threshold is curated by identical rules to one over it.
    Only the process it runs in differs.
    """
    return _curate_one(job)


def _checkpoint_cache(c, cache, report):
    """Write the cache if enough files have been handled since the last write."""
    report["_since_checkpoint"] = report.get("_since_checkpoint", 0) + 1
    if report["_since_checkpoint"] < CHECKPOINT_EVERY:
        return
    report["_since_checkpoint"] = 0
    save_cache(c, cache)


def unchanged_since_last_run(c, source, entries):
    """The digest this source produced last time, or None if it must be redone.

    The test is size and modification time rather than content, because
    comparing content means reading the file, and reading the file is most of
    what this function exists to avoid. A file that was edited without its
    timestamp moving is a case the filesystem does not offer to detect, and the
    same trade every build cache makes.

    The cached digest is only trusted when the curated file it names still
    exists. A cache pointing at a document that has since been quarantined or
    deleted is worse than no cache, because it would report the file as done
    while its output is nowhere.
    """
    key = str(source)
    record = entries.get(key)
    if not isinstance(record, dict):
        return None
    try:
        stat = source.stat()
    except OSError:
        return None
    if record.get("size") != stat.st_size:
        return None
    if record.get("mtime") != stat.st_mtime:
        return None
    digest = record.get("digest")
    if not digest:
        # The file was refused last time. Refusing is a decision that has to be
        # re-made when the file changes, not remembered, because the reason it
        # was refused may no longer apply.
        return None
    if not (Path(c.curated_dir) / f"{digest}.txt").exists():
        return None
    return digest


def _expand_oversized(paths, c, report):
    """Replace each oversized file with bounded pieces, and count what was split.

    A document too large to hold in memory is not a document that cannot be
    learned from. It is a document that has to be read in pieces, which is what
    this does: the file is copied once into chunks of at most max_raw_bytes,
    each chunk becomes an ordinary input to the rest of curation, and the
    original in data/raw is left exactly where it was.

    The reason for the limit is memory, not policy. A seventy-gigabyte file
    needs seventy gigabytes of RAM to read at once, and the failure arrives as a
    MemoryError raised inside a codec that names neither the file nor its size.
    Reading in bounded pieces costs the same memory as reading a small file and
    the same time as copying the file once, which is the unavoidable cost of
    accepting the material.

    Splitting happens on a line boundary where one is available, because cutting
    mid-sentence produces a document that ends and another that begins without
    their proper context, and a model trained on those learns something that was
    never written. A single line longer than the limit is split at the limit
    regardless, and that is the one case where a boundary is lost.
    """
    out_dir = Path(getattr(c, "split_dir", "data/split"))
    expanded = []
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            expanded.append(path)
            continue
        if size <= c.max_raw_bytes:
            expanded.append(path)
            continue
        pieces = _split_file(path, out_dir, c.max_raw_bytes)
        if not pieces:
            expanded.append(path)
            continue
        report["split"] += 1
        report["split_files"].append(
            {
                "source": str(path),
                "bytes": size,
                "limit": c.max_raw_bytes,
                "pieces": len(pieces),
                "piece_bytes": c.max_raw_bytes,
                "note": (
                    f"{path.name} was larger than the {c.max_raw_bytes} byte read "
                    f"limit, so it was split into {len(pieces)} pieces and each "
                    f"was curated separately. The original in data/raw is "
                    f"untouched."
                ),
            }
        )
        expanded.extend(pieces)
    return expanded


def _cut(line, piece_bytes):
    """Split one oversized line at a byte boundary no further in than the limit.

    Cut at a character boundary rather than a byte one, because cutting bytes
    through a multi-byte character produces text that will not decode, and a
    piece that will not decode is a piece the next stage cannot read. The head
    is therefore the longest prefix of the line that fits in the limit and ends
    on a character boundary.
    """
    if len(line.encode("utf-8", errors="replace")) <= piece_bytes:
        return line, ""
    # Binary search would be faster, but a line is text and the limit is 64 KiB,
    # so walking back a bounded number of characters is cheap and easier to be
    # sure about than getting a byte index exactly right.
    end = len(line)
    while end > 0:
        candidate = line[:end]
        if len(candidate.encode("utf-8", errors="replace")) <= piece_bytes:
            return candidate, line[end:]
        # Walk back a chunk at a time rather than a character at a time, then
        # refine, so a pathological line does not become quadratic.
        end = max(0, end - max(1, piece_bytes // 8))
    return "", line


def _split_file(path, out_dir, piece_bytes):
    """Write a large text file into pieces of at most piece_bytes, on line boundaries.

    Returns the piece paths, or an empty list when the file could not be split,
    in which case the caller refuses it rather than reading it. The pieces are
    named after the source and their index rather than after their content, so
    two runs over the same corpus produce the same names and a piece that
    already exists is not duplicated.

    Every byte of the source ends up in exactly one piece. That is the property
    that matters, and it is the one that was wrong twice while this was being
    written: a version that dropped over-long pieces lost a whole file, and a
    version that kept a trailing empty piece invented one. The loop below is
    written so a line is either in the piece being built or starts the next, and
    a piece is emitted only by being full or by the file ending.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    pieces = []

    def emit(lines):
        # Binary, because the limit is counted in bytes and text mode
        # translates a newline into two on this platform. Writing in text mode
        # produced pieces a few hundred bytes over the limit, which is the exact
        # size the next stage refuses, so every piece of a large document was
        # split and then thrown away.
        target = out_dir / f"{path.stem}.part{len(pieces):04d}.txt"
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write("".join(lines))
        pieces.append(target)

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            current = []
            current_bytes = 0
            for line in handle:
                encoded = len(line.encode("utf-8", errors="replace"))
                # Flush BEFORE appending when the line will not fit, which is
                # what keeps a piece inside the limit. Checking only after the
                # append let the last line push every piece over by its own
                # length, so a limit of 65536 produced pieces of 66464 and the
                # very next stage refused the file it had just been split for.
                if current and current_bytes + encoded > piece_bytes:
                    emit(current)
                    current = []
                    current_bytes = 0
                current.append(line)
                current_bytes += encoded
                # A single line longer than the whole limit cannot be made to
                # fit on a boundary, because there is no boundary inside it. It
                # is cut at the limit and the remainder starts the next piece.
                # This is the one place a boundary is lost, and it is stated
                # rather than pretended away.
                while len(current) == 1 and current_bytes > piece_bytes:
                    head, tail = _cut(current[0], piece_bytes)
                    emit([head])
                    current = [tail]
                    current_bytes = len(tail.encode("utf-8", errors="replace"))
            if current:
                emit(current)
    except OSError:
        # A file that cannot be read is left for the refusal path, which reports
        # it. Any pieces already written stay on disk rather than being deleted:
        # they are a copy of material that still exists in data/raw, so removing
        # them destroys nothing, but a deletion is irreversible and there is no
        # reason for it. The caller returns an empty list, so they are not
        # curated from, and the next run overwrites them by name.
        return []
    return pieces


def _short(path, keep=52):
    """A source path short enough to read in a progress line.

    Curation names every file it touches, and a curated document's real name is a
    sixty-eight character digest, so the folder it came from is the only part
    worth showing. The folder is what a person recognises; the digest is what
    they would have to look up.
    """
    parts = Path(path).parts
    if len(parts) <= 2:
        return str(path)[-keep:]
    folder = parts[-2] if len(parts) >= 2 else ""
    return f"{folder}/{Path(path).name[-18:]}"


def _say(label, text):
    """Write one line of progress to the console and to the report.

    Curation can run for a long time over a large corpus, and until now it said
    nothing at all until it finished. A run that is silent for eight minutes is
    indistinguishable from a run that has hung, which is the thing a person
    watching a window cannot tell apart and does not have to be told to guess
    about. Every document is named as it is handled, so the answer to "what is it
    doing" and "which file is it on" are both visible while it is happening.
    """
    print(f"{label:<12}{text}", flush=True)


def curate(c=None):
    """Curate every document in data/raw into data/curated.

    Takes a configuration so a test can point it at a fixture rather than at the
    developer's own corpus, which is the mistake that made this function take
    four hours inside a test suite.

    Documents are curated across several processes, because curating a document
    is a few seconds of CPU and the corpus is a hundred and thirty gigabytes of
    them. The work in a worker is the expensive part; the decisions about which
    digest wins and what the report says stay here, in one place, in source
    order. A run that is interrupted leaves the cache describing what was really
    finished, which is the property that matters when a run takes hours.
    """
    c = c or load_config()
    raw = Path(c.raw_dir)
    out = Path(c.curated_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "files_seen": 0,
        "accepted": 0,
        # Files that were already curated last run and had not changed since.
        # Counted separately from `accepted` so a run that skipped everything is
        # visibly a run that skipped everything.
        "unchanged": 0,
        "rejected": 0,
        "duplicates": 0,
        "conversational": 0,
        "conversation_turns": 0,
        "prose": 0,
        "contaminated": 0,
        "contaminated_documents": [],
        "structured": 0,
        "structured_records": 0,
        "unrecognized_records": 0,
        "structured_documents": [],
        "unreadable": 0,
        "unreadable_files": [],
        # Payload punctuation removed before the text became a learning
        # document. Counted per document and in total, because a scrub that
        # removed nothing and a scrub that is not running look identical from
        # the outside, and only the count tells them apart.
        "scrubbed": 0,
        "scrub_removed": 0,
        "scrubbed_documents": [],
        "oversized": 0,
        "oversized_files": [],
        "split": 0,
        "split_files": [],
        "skipped_unsupported": 0,
        "unsupported_files": [],
        "documents": [],
    }
    seen = set()
    cache = load_cache(c)
    # data/raw is read recursively, which is what lets a person sort the corpus
    # into folders by subject without the pipeline knowing. It also means a
    # folder holding material that has already been accounted for would be read
    # again: duplicates/ exists precisely to keep a second copy of a document out
    # of the corpus, and reading it would spend the hashing cost to arrive at the
    # verdict it was set aside for. quarantined/ is refused documents, which is
    # the owner's decision rather than this pipeline's to revisit.
    sources = _expand_oversized(
        sorted(
            p
            for p in raw.rglob("*")
            if p.is_file() and EXCLUDED_DIRS.isdisjoint(p.relative_to(raw).parts)
        ),
        c,
        report,
    )

    # Every document is checked against the cache before any of them is read. A
    # document that has not changed since the last run is not dispatched at all:
    # answering that costs one stat() where reading it can cost a minute, and it
    # is the reason a second run over an unchanged corpus finishes quickly. It is
    # still counted and still reported, because a run that skipped everything
    # would otherwise claim it had seen nothing.
    #
    # The cache is read here and nowhere else, and written here and nowhere else.
    # A cache with more than one writer is a cache that claims documents were
    # curated when they were not, and that failure is silent and permanent: the
    # next run skips them forever.
    results = []
    pending = []
    for index, source in enumerate(sources):
        try:
            raw_bytes = source.stat().st_size
        except OSError:
            # Cannot be measured here, so let the worker refuse it and say why.
            pending.append((index, str(source)))
            continue

        cached_digest = unchanged_since_last_run(c, source, cache)
        if cached_digest is not None:
            _say("unchanged", f"{_short(source)}")
            results.append(
                {
                    "index": index,
                    "source": str(source),
                    "counts": {"files_seen": 1, "accepted": 1, "unchanged": 1},
                    "lists": {},
                    "kind": "unchanged",
                    "skill": _skill_of(source),
                    "digest": cached_digest,
                    "size": raw_bytes,
                    "mtime": source.stat().st_mtime,
                    "rendered": None,
                }
            )
            continue
        pending.append((index, str(source)))

    small = [job for job in pending if _job_bytes(job[1]) <= INLINE_MAX_BYTES]
    large = [job for job in pending if _job_bytes(job[1]) > INLINE_MAX_BYTES]

    workers = _worker_count(c, len(large))
    started = time.perf_counter()

    # Documents are consumed in source order, not completion order. The order is
    # what decides which document is reported as a duplicate when two raw files
    # curate to the same digest, and that decision is part of the report rather
    # than an implementation detail, so it is made the same way every run.
    for job in small:
        results.append(_curate_one(job))

    if large:
        if workers <= 1:
            for job in large:
                results.append(_curate_one(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # Results arrive as they finish and are sorted afterwards, so a
                # slow document does not hold up the merge behind it.
                for outcome in pool.map(_curate_one, large, chunksize=1):
                    results.append(outcome)

    results.sort(key=lambda outcome: outcome["index"])

    for outcome in results:
        for name, value in outcome["counts"].items():
            report[name] += value
        for name, entries in outcome["lists"].items():
            report[name].extend(entries)

        if outcome["kind"] == "accepted":
            digest = outcome["digest"]
            if digest in seen:
                report["duplicates"] += 1
                continue
            seen.add(digest)
            target = out / f"{digest}.txt"
            source_name = outcome["source"]
            cache[source_name] = {
                "size": outcome["size"],
                "mtime": outcome["mtime"],
                "digest": digest,
            }
            _checkpoint_cache(c, cache, report)
            target.write_text(outcome["rendered"] + "\n", encoding="utf-8")
            _say(
                "curated",
                f"{_short(source_name)}  {outcome['document']['chars']} chars",
            )
            report["accepted"] += 1
            document = dict(outcome["document"])
            document["curated"] = str(target)
            report["documents"].append(document)
        elif outcome["kind"] == "unchanged":
            # A document that was already curated. Its digest still has to join
            # the set, or a later duplicate of it would be written a second time
            # as if it were new. `accepted` was already counted by the outcome's
            # own counters when it was built, because an unchanged document was
            # accepted last run and is still the training material it was.
            seen.add(outcome["digest"])
            source_name = outcome["source"]
            cache[source_name] = {
                "size": outcome["size"],
                "mtime": outcome["mtime"],
                "digest": outcome["digest"],
            }
            _checkpoint_cache(c, cache, report)
            report["documents"].append(
                {
                    "source": source_name,
                    "curated": str(out / f"{outcome['digest']}.txt"),
                    "sha256": outcome["digest"],
                    "chars": None,
                    "source_kind": "unchanged",
                    "records": None,
                    "kind": "unchanged",
                    "skill": outcome["skill"],
                    "turns": None,
                    "turn_share": None,
                    "template_artifacts": None,
                    "template_share": None,
                    "speakers": [],
                }
            )

    elapsed = time.perf_counter() - started
    report["parallel"] = {
        "workers": workers,
        "documents": len(results),
        "read_in_process": len(small),
        "read_in_pool": len(large),
        "seconds": round(elapsed, 3),
        "documents_per_second": (
            round(len(large) / elapsed, 2) if elapsed and large else None
        ),
        "note": (
            f"Documents were curated across {workers} worker processes in "
            f"{elapsed:.1f} seconds. Small documents were read in the calling "
            f"process, because handing one to another process costs more than "
            f"reading it does. The order documents are reported in is the order "
            f"they appear in data/raw, not the order they finished, so two raw "
            f"files that curate to the same digest are reported the same way every "
            f"run."
        ),
    }

    report["note"] = (
        "Architecture 25 lists conversations as a source. This corpus holds "
        f"{report['conversational']} conversational and {report['prose']} prose "
        "documents. Jupiter cannot learn to take turns from prose alone, and "
        "no dialogue is fabricated to fill the gap."
    )
    report["template_artifact_note"] = (
        f"{report['contaminated']} documents were refused because they contain "
        "more than "
        f"{c.max_template_artifacts} chat-template markers such as [SYSTEM]:. "
        "That is a leaked conversation template, not dialogue, and a model "
        "trained on it emits instructions instead of conversing. The files in "
        "data/raw/ are untouched; they are listed under contaminated_documents "
        "so the owner can decide what to do with them."
    )
    report["structured_note"] = (
        f"{report['structured']} of {report['accepted']} accepted documents came "
        f"from a structured source, providing {report['structured_records']} "
        f"records of which {report['unrecognized_records']} carried no field "
        "this pipeline recognises as text and were left out. An unrecognized "
        "record is a shape json_source cannot read yet; it is never guessed at, "
        "because invented text becomes training material nobody wrote."
    )
    report["unreadable_note"] = (
        f"{report['unreadable']} structured files were refused. They are listed "
        "under unreadable_files with the reason, and the files in data/raw are "
        "untouched. Refusing is not deleting: the owner decides what to do."
    )
    report["unsupported_note"] = (
        f"{report['skipped_unsupported']} files in data/raw carry an extension "
        f"this pipeline does not read ({', '.join(sorted(c.structured_suffixes))} "
        "are parsed, and .txt/.md/.text are read as text). They are listed under "
        "unsupported_files rather than passed over silently, because a file that "
        "is skipped without a word is a file whose owner has no way to know it is "
        "doing nothing."
    )
    report["oversized_note"] = (
        f"{report['oversized']} raw files were larger than max_raw_bytes "
        f"({c.max_raw_bytes} bytes) and were not read. They are listed under "
        "oversized_files with their size and the limit. Curation reads a whole "
        "document to judge it, so a file this large needs memory proportional to "
        "itself rather than to the context window Jupiter learns from. The files "
        "in data/raw are untouched: refusing to read a file is not deleting it."
    )
    # Read before pruning, so a curated file whose raw source is absent this run is
    # recognised as previously accepted rather than treated as stale.
    known_digests = load_digest_history(c)
    report = prune_curated(c, report, seen, known_digests=known_digests)
    # Recorded after, so it holds everything this pipeline has ever accepted,
    # not just what this run happened to see in whatever subset of raw was
    # staged today.
    save_digest_history(c, known_digests | seen)
    # The cache is written last of all, and holds only what this run actually
    # verified or produced. A run that failed part way through leaves the
    # previous cache intact rather than a half-written one that would claim
    # files were done when they were never reached.
    save_cache(c, cache)
    report["unchanged_note"] = (
        f"{report['unchanged']} of {report['files_seen']} files were already "
        "curated and had not changed since the last run, so they were not read "
        "again. A file is re-curated when it changes, when its curated output "
        "has gone missing, or when the curation rules themselves change."
    )
    report_path = Path(c.curation_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    curate()
