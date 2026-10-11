"""Organise data/raw into skill folders, and move duplicates aside.

Two jobs, and they are deliberately separate from each other.

Organising is a move, not a copy and not a deletion. Curation already reads
data/raw recursively, so a file in data/raw/math/ is curated exactly as it was
in data/raw/ and nothing about the pipeline has to know the folders exist. What
the folders buy is that a person can see what the corpus holds, and that the
skill the material teaches is visible in the path rather than only in a report.

Duplicates are moved rather than refused for a reason that is about cost rather
than judgement. Curation hashes every file it reads on every run, so a duplicate
in raw/ is re-read and re-hashed forever to arrive at the same verdict. Moving
it out of the scanned tree ends that cost. It is still a move: the bytes are
untouched, the file is still on disk, and putting it back restores the corpus
exactly.

Nothing is deleted. Every move is reported, and --apply is required, so the
whole thing can be inspected before it happens.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

# Where material goes, and the order matters: the first pattern that matches a
# filename wins, so a file named default_chat.part0000.txt lands in conversation
# rather than in whichever later rule would also have claimed it.
FOLDERS = (
    ("conversation", re.compile(r"chat|conversation|dialog|talk|instruction", re.I)),
    ("math", re.compile(r"math|latex|arith|algebra|geometry|calculus", re.I)),
    ("code", re.compile(r"code|program|debug|python|java|javascript|sql", re.I)),
    ("reasoning", re.compile(r"reason|think|chain_of_thought|cot", re.I)),
    ("science", re.compile(r"science|physics|chem|biology|safety", re.I)),
    ("prose", re.compile(r"asimov|novel|book|story|stories|english_text", re.I)),
    # A scanned book carries the djvu suffix every OCR text of one does, and
    # 225 of them say nothing else about their subject. Prose is the honest
    # default for a book rather than leaving it unclassified.
    ("prose", re.compile(r"_djvu$", re.I)),
)

# The corpus holds 1,873 files named "<Language>_sample_N", covering 663
# distinct programming languages from 1C Enterprise to Zig. None of them carries
# a word any of the patterns above match, so every one of them read as
# unclassified and the largest single body of programming material in the corpus
# would have been left in the root with nothing saying what it was. These are
# programming languages by the shape of the corpus rather than by their names,
# and classifying them from the name would mean maintaining a list of six
# hundred and sixty-three languages, which is a list that goes stale the day a
# language is added.
SAMPLE_FILE = re.compile(r"_sample_?\d+$", re.I)

# What each unrecognised kind of material actually opens with, and where it
# belongs. Each is anchored to a literal the file has to contain, so a document
# that merely discusses a subject is not filed by it.
CONTENT_RULES = (
    (
        "math",
        re.compile(
            r"\$\$|\\\[|\\frac|\\boxed|JEE Main|JEE Advanced|Problem Statement|"
            r"Input Question|\\to\\to",
            re.I,
        ),
    ),
    (
        "conversation",
        re.compile(
            r'"role"\s*:\s*"(user|assistant|human|system)"|'
            r"\bUser Prompt:|\bgenerated-experience:|"
            r"^\s*You:\s",
            re.I | re.MULTILINE,
        ),
    ),
    (
        "code",
        re.compile(
            r"Binary Object \d+|Malware:|Enable quiet mode|pre-commit hook|"
            # A syntax matrix, not a document that merely says which language
            # it is written in: the Gutenberg header of a translated Asimov
            # title reads "Language: English", and matching that filed a
            # nuclear-energy book as source code.
            r"\bLanguage:\s*[A-Za-z+#./ ]{1,24}\(?(?:ANSI|ISO|C\+\+|Java|Python)\)?|"
            r"Keywords:\s+\w+,\s+\w+",
            re.I,
        ),
    ),
    (
        "science",
        re.compile(r"Threat Log \d+|Threat Modeling|Hacking Group", re.I),
    ),
    (
        "prose",
        re.compile(
            r"Project Gutenberg|Once upon a time|little girl named|"
            r"for all the love, romance|Heestan waxaa|"
            r"^\s*Abandon:|\bJupiter:",
            re.I | re.MULTILINE,
        ),
    ),
)

# How much of a file is read to decide what it is. The decision only needs the
# opening, and reading a twelve-megabyte document in full to learn its subject
# would cost more than filing it was worth. It is also what makes a file named by
# hash fileable at all: thirty-one documents arrived named that way, and among
# them were the largest set of examination mathematics in the corpus and nine
# megabytes of programming instruction.
HEAD_BYTES = 4096

# The folders the layout is made of, derived from the classification rules so
# that adding a skill folder cannot leave it prunable by accident.
FOLDER_NAMES = tuple(folder for folder, _pattern in FOLDERS)

# Folders that are never scanned. duplicates/ holds material that is already in
# the corpus by another name, and re-reading it would spend the hashing cost
# again to reach the same verdict. quarantined/ holds documents curation refused,
# which is a decision about the owner and not about this script.
SKIP = {"duplicates", "quarantined", "organized"}

# The same names, kept as a set for the fast membership test on path parts.
EXCLUDED_DIRS = frozenset(SKIP)

# A part file is a slice of one document that was too large to keep whole, and
# the slices belong together. Splitting them across folders would be a category
# assignment per slice rather than per document, which is why the part suffix is
# stripped before the name is classified.
PART_SUFFIX = re.compile(r"\.part\d+$", re.I)


def classify(name):
    """Which skill folder a filename belongs to, or None when nothing matches.

    None is a real answer rather than a failure. Guessing a folder for material
    nothing recognises would put a document somewhere it does not belong and
    make the folder layout claim something untrue about the corpus.
    """
    stem = PART_SUFFIX.sub("", Path(name).stem)
    for folder, pattern in FOLDERS:
        if pattern.search(stem):
            return folder
    if SAMPLE_FILE.search(stem):
        return "code"
    return None


def classify_path(path):
    """Which folder a file belongs to, by name first and then by content.

    The name is tried first because it is free and because a name is a
    deliberate label. Content is only consulted for a file no name rule claimed,
    because thirty-one documents in this corpus carry a hash for a name and are
    otherwise perfectly ordinary material: without this they stayed in the root
    and the folder layout claimed to describe a corpus it had not classified.

    Every unclaimed file is read, not only the hash-named ones. Reading a
    seventeen-megabyte document to learn its subject is bounded and cheap, and
    the alternative is a corpus whose largest maths set and whose own generated
    material sit in the root because their filenames happened not to contain a
    word from a list. A name that says nothing useful is not a reason to stop
    looking.

    Returning None is still a real answer. A document nothing recognises is
    reported rather than guessed at, because a folder that means the wrong thing
    is worse than no folder.
    """
    by_name = classify(path.name)
    if by_name is not None:
        return by_name
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(HEAD_BYTES)
    except OSError:
        return None
    for folder, pattern in CONTENT_RULES:
        if pattern.search(head):
            return folder
    return None


def digest_of(path, chunk=1 << 20):
    """SHA-256 of a file, read in blocks.

    Blocked because the corpus contains files of sixty-odd megabytes and reading
    one whole to hash it is exactly the cost this script exists to avoid paying
    repeatedly. The hash is the same either way.

    Only files that already share a size get here. `fingerprint` below reads a
    few kilobytes per file to decide, and this reads the whole file only for the
    handful that survive it. On this corpus that is the difference between eight
    minutes and a few seconds.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def fingerprint(path, span=1 << 15):
    """A cheap identity for a file: its length plus the bytes at both ends.

    Grouping by length alone was not enough, and the reason is specific to this
    corpus: the large documents were split into equal-sized parts, so a thousand
    of them share one length and grouping by it re-read the entire hundred and
    thirty-five gigabytes. Reading the first and last thirty-two kilobytes of
    each file instead costs about a second across the whole corpus, and two
    different documents that agree on their length and on both of their ends
    are rare enough that the few that do get a full hash to settle it.

    The fingerprint decides what is worth hashing; it never decides the answer
    on its own.
    """
    size = path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        hasher.update(handle.read(span))
        if size > span * 2:
            handle.seek(-span, os.SEEK_END)
            hasher.update(handle.read(span))
    return size, hasher.hexdigest()


def cache_path(raw):
    """Where the classification and fingerprint cache is kept.

    Beside the corpus rather than inside it, so it is never itself a file this
    script tries to classify.
    """
    return Path(raw).parent / "organize_cache.json"


def load_cache(raw):
    """The cached classification and fingerprints, or an empty mapping."""
    path = cache_path(raw)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def cache_key(path):
    """What identifies a file's contents for caching: its path, size and mtime."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return f"{path}|{stat.st_size}|{stat.st_mtime}"


def save_cache(raw, entries):
    """Write the cache so the next run does not read the corpus again."""
    path = cache_path(raw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    except OSError:
        # A cache that cannot be written costs a slower run and nothing else.
        pass


def find_duplicates(paths, chunk=1 << 20):
    """Files whose contents match another file's, by content alone.

    Reads 64 KB per file rather than the whole file, and only hashes in full
    the handful whose cheap fingerprint collides. Hashing every byte of every
    file took over eight minutes on this corpus and produced its answer for a
    few dozen files; this produces the same answer in about a second.

    Both the fingerprint and the full hash are cached, so a second run over an
    unchanged corpus reads nothing at all. That matters because the plan and the
    apply each want this answer, and asking twice was half the cost of running
    twice.
    """
    cache = load_cache(CACHE_RAW[0]) if CACHE_RAW[0] else {}
    by_fingerprint = {}
    for path in paths:
        key = cache_key(path)
        if key is None:
            continue
        record = cache.get(key)
        if isinstance(record, dict) and "fingerprint" in record:
            cheap = tuple(record["fingerprint"])
        else:
            try:
                cheap = fingerprint(path)
            except OSError:
                continue
            entry = cache.setdefault(key, {})
            entry["fingerprint"] = list(cheap)
            entry.setdefault("folder", classify_path(path))
        by_fingerprint.setdefault(cheap, []).append(path)

    seen = {}
    duplicates = []
    for (size, _cheap), group in sorted(by_fingerprint.items()):
        if len(group) < 2:
            continue
        for path in group:
            key = cache_key(path)
            entry = cache.get(key) if key else None
            if isinstance(entry, dict) and "digest" in entry:
                digest = entry["digest"]
            else:
                digest = digest_of(path, chunk)
                if key is not None:
                    cache.setdefault(key, {})["digest"] = digest
            if digest in seen:
                duplicates.append((path, seen[digest], digest, size))
            else:
                seen[digest] = path

    if CACHE_RAW[0]:
        save_cache(CACHE_RAW[0], cache)
    return duplicates


# Set by plan() so the cache helpers above can find the corpus without every
# caller having to thread a path through. A one-element list rather than a
# module global so it can be rebound in a test.
CACHE_RAW = [None]


def walk_sources(raw):
    """Every file worth moving, skipping the folders this script owns.

    Cached between the plan and the apply. Both walk the whole corpus, and
    between them they were reading every unclassified file's head and every
    candidate's first and last kilobytes twice over, which on a corpus this size
    is minutes of disk traffic to produce the same answer twice. The cache is
    keyed on the path with the size and modification time alongside, so a file
    edited between the plan and the apply is re-read rather than trusted.
    """
    for path in sorted(raw.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(raw)
        if relative.parts and relative.parts[0] in SKIP:
            continue
        yield path


def plan(raw):
    """What would happen, without doing it.

    Returns the moves, the duplicates, and what stayed put. Nothing is written,
    so this is safe to run at any time and is how the layout is inspected before
    it is committed.
    """
    classified = []
    unclassified = []
    settled = 0
    CACHE_RAW[0] = raw
    cache = load_cache(raw)
    for path in walk_sources(raw):
        key = cache_key(path)
        entry = cache.get(key) if key else None
        if isinstance(entry, dict) and "folder" in entry:
            folder = entry["folder"]
        else:
            folder = classify_path(path)
            if key is not None:
                cache.setdefault(key, {})["folder"] = folder
        if folder is None:
            unclassified.append(path)
            continue
        if path.parent.name == folder and path.parent.parent == raw:
            # Already filed. Counted and left alone, so re-running is safe and
            # a part-finished run converges instead of nesting folders.
            settled += 1
            continue
        classified.append((path, raw / folder / path.name))

    moves = classified
    # Only the files that would be moved are candidates for duplication, plus
    # the files already filed, which are in their skill folder rather than at
    # the root. Files set aside in duplicates/ or quarantined/ are excluded: a
    # second run would otherwise find each duplicate against the copy filed
    # there and move it back out, so the folder empties itself one run at a
    # time.
    candidates = [source for source, _target in moves]
    candidates.extend(
        path
        for path in raw.glob("*/*")
        if path.is_file()
        and not EXCLUDED_DIRS.intersection(path.relative_to(raw).parts)
    )
    duplicates = find_duplicates(candidates)

    return moves, duplicates, unclassified, settled


def prune_empty_dirs(raw):
    """Remove directories that hold no files, deepest first.

    Filing a corpus leaves the shells of where its documents used to be: a book
    folder whose single document moved to prose/ keeps an empty prose/ behind it,
    and a thousand of those make the tree harder to read than the flat one it
    replaced. Only directories with no file anywhere beneath them are removed,
    and a directory that still holds a document is left exactly as it is, so
    nothing that was filed can be lost to tidying.

    The order matters. Removing deepest first means a directory emptied by the
    removal of its own subdirectories is caught in the same pass rather than
    surviving as a husk.
    """
    removed = 0
    # The folders the layout itself is made of. They are kept even when empty,
    # because removing one means the next run has to recreate it and the folder
    # a duplicate was moved into would vanish the moment it was emptied.
    keep = {raw.resolve()} | {
        (raw / name).resolve() for name in SKIP | set(FOLDER_NAMES)
    }
    # Deepest first, and re-checked until a pass removes nothing: a directory
    # emptied by the removal of its own subdirectories is only visible on the
    # next pass, so a single ordered walk leaves the parents behind.
    while True:
        removed_this_pass = 0
        for directory in sorted(
            (p for p in raw.rglob("*") if p.is_dir()),
            key=lambda p: -len(p.parts),
        ):
            if directory.resolve() in keep:
                continue
            try:
                if any(directory.iterdir()):
                    continue
                directory.rmdir()
                removed += 1
                removed_this_pass += 1
            except OSError:
                # A directory that could not be removed is left alone rather
                # than treated as a failure: it costs nothing to keep and
                # reporting a tidy-up as an error would be worse than the
                # untidiness.
                continue
        if not removed_this_pass:
            break
    return removed


def apply_plan(raw, moves, duplicates, move_files=True, prune=True):
    """Perform the moves, returning what was actually done.

    Duplicates are handled before the classification moves, not after. A
    duplicate's original is itself usually a classified file, so moving the
    classified files first renames the original out from under the duplicate
    that was going to be filed beside it, and the second step then fails on a
    path that no longer exists. Doing the duplicates first means both members of
    every pair are still where the scan found them.
    """
    done = {"moved": 0, "duplicates_moved": 0, "empty_dirs_removed": 0}
    # A source that the duplicate pass has already moved is not moved again.
    # The duplicates are handled first precisely so that both members of a pair
    # are still where the scan found them, and the member that was set aside
    # would otherwise be filed from a path that no longer exists.
    moved_as_duplicate = {path for path, _o, _d, _s in duplicates}

    for path, original, _digest, _size in duplicates:
        if not move_files:
            continue
        # A duplicate is kept beside the file it duplicates rather than beside
        # its own category, so the pair stays legible when someone goes looking
        # for why one of them was set aside.
        holding = raw / "duplicates" / original.name
        holding.parent.mkdir(parents=True, exist_ok=True)
        if path == holding:
            continue
        # Reinstating a file that is already set aside is left alone rather than
        # moved, for the same reason the scan skips the folder: the copy is
        # where it was put on purpose and there is nothing to gain by churning
        # it between two folders that are both outside the corpus.
        if holding.exists():
            continue
        path.replace(holding)
        done["duplicates_moved"] += 1

    for source, target in moves:
        if not move_files:
            continue
        if source in moved_as_duplicate:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        done["moved"] += 1
    if prune and move_files:
        done["empty_dirs_removed"] = prune_empty_dirs(raw)
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="organize_raw",
        description=(
            "Sort data/raw into skill folders and move duplicates aside. "
            "Nothing is deleted and nothing is copied: every file is moved."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually move the files. Without it, only the plan is printed.",
    )
    parser.add_argument(
        "--raw",
        default=str(PROJECT / "data" / "raw"),
        help="the raw directory to organise",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="write the plan as JSON to this path",
    )
    args = parser.parse_args(argv)

    raw = Path(args.raw)
    if not raw.is_dir():
        print(f"no such directory: {raw}", file=sys.stderr)
        return 1

    moves, duplicates, unclassified, settled = plan(raw)

    by_folder = {}
    for _source, target in moves:
        by_folder[target.parent.name] = by_folder.get(target.parent.name, 0) + 1

    summary = {
        "raw": str(raw),
        "files_to_move": len(moves),
        "already_filed": settled,
        "by_folder": dict(sorted(by_folder.items())),
        "duplicates_found": len(duplicates),
        "duplicates_sample": [
            str(p.relative_to(raw)) for p, _o, _d, _s in duplicates[:10]
        ],
        "duplicate_bytes": sum(size for _p, _o, _d, size in duplicates),
        "unclassified": len(unclassified),
        "unclassified_sample": [str(p.relative_to(raw)) for p in unclassified[:10]],
        "applied": bool(args.apply),
    }

    if args.report:
        Path(args.report).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.apply:
        done = apply_plan(raw, moves, duplicates)
        print(json.dumps(done, indent=2, sort_keys=True))
    else:
        print("\nnothing moved. Re-run with --apply to carry this out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
