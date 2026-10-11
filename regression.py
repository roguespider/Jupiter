"""Regression detection for repeated learning (Architecture 33).

An update that improves one capability but destroys another is not a success.
Holdout loss measured on the document currently being learned cannot see that,
because every document supplies its own holdout and the numbers are not
comparable across documents.

This module reserves exactly one document out of the learning candidates and
re-measures it on every cycle. It becomes the fixed yardstick that makes
"improved" mean "improved without forgetting".
"""

import json
from pathlib import Path

from train import loss_on, split_tokens


def load_reserved(c):
    """Return the reserved regression documents, or None if not set yet.

    Returns a list. It used to return a single path, and the state file named
    one document, so this is the same contract widened rather than a new idea:
    Architecture 33 needs the degradation to be measured against something that
    stands for the corpus, and one file out of thirty does not.
    """
    state_path = Path(c.regression_state)
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        reserved = state.get("documents")
        if reserved is None:
            single = state.get("reserved")
            reserved = [single] if single else []
        if not isinstance(reserved, list):
            return None
    except (OSError, ValueError, TypeError, AttributeError):
        # An unreadable reservation file is treated as "not reserved" so the
        # worker re-reserves rather than silently losing regression detection.
        return None
    paths = [Path(name) for name in reserved if name]
    return paths or None


def largest_of(paths):
    """The largest path, which is the one held back when the corpus is tiny."""
    return max(paths, key=lambda path: path.stat().st_size)


def choose_yardstick(c, pieces, max_share=0.25):
    """Pick documents to hold back that stand for the corpus rather than one file.

    This used to be the first candidate in sorted order, which made the yardstick
    one document out of thirty. The specification asks the system to monitor
    knowledge degradation, and one document cannot do that: it sees a change in
    itself and nothing else, so learning any new way of expressing something
    moves it and the update is discarded. Measured over 200 cycles on this
    corpus, 101 were a direct conflict in which the holdout loss improved while
    the reserved document got worse, and 79 updates were thrown away despite a
    mean 13.77% improvement. That is generalisation being read as damage, and it
    is why conversation would not stick however long the loop ran.

    Three strata by size: the smallest document, the largest, and one near the
    median. Size is used because it is the only property available without
    reading the corpus on every reservation, and because it correlates with the
    thing that matters here - a very large document behaves differently under a
    fixed context window than a small one does. A candidate knows its size
    already, so sizing a corpus is free now that candidates are not read.

    The sample is capped at a share of the corpus rather than a fixed count. A
    fixed count is wrong at both ends: three documents are not a yardstick, and
    reserving a quarter of thirty files would leave the worker almost nothing to
    learn from. A share keeps both ends honest.

    Representative is claimed, not achieved. Nothing here can know whether the
    corpus holds material unlike the sample, and the note written into the state
    file says so, because a yardstick that is quietly unrepresentative is worse
    than one that is admitted to be.
    """

    def size_of(piece):
        if isinstance(piece, tuple):
            return piece[0].stat().st_size
        return piece.bytes_on_disk

    def path_of(piece):
        return piece[0] if isinstance(piece, tuple) else piece.path

    # Sorted by the size the candidate already recorded rather than by a fresh
    # stat() of every document: a candidate knows its size because that is how
    # it was described, and asking the filesystem again for a thousand documents
    # is a thousand round trips to learn something already held.
    ordered = sorted(pieces, key=size_of)
    paths = [path_of(piece) for piece in ordered]
    if len(paths) <= 2:
        # Too few documents to sample. Everything but the largest is held back,
        # which leaves the worker something to learn and something to verify.
        return paths[:-1]

    # The sample is capped at a share of the corpus rather than a fixed count. A
    # fixed count is wrong at both ends: three documents are not a yardstick, and
    # reserving a quarter of thirty files would leave the worker almost nothing to
    # learn from. A share keeps both ends honest.
    #
    # The share is of a corpus expected to be larger than the one being measured,
    # so the floor is scaled rather than fixed. A corpus of nine documents gets
    # two; thirty gets seven. The sample must leave at least one document to
    # learn on, which is the one hard constraint: a yardstick that reserves
    # everything is a halt rather than a measurement.
    ceiling = max(2, int(len(paths) * max_share))
    ceiling = min(ceiling, max(1, len(paths) - 1))

    # With only a handful of documents, holding back two leaves one to learn
    # from and every cycle lands on it, which is the behaviour the selection
    # queue exists to avoid. Below five documents the share is the wrong shape:
    # reserve one and let the rest cycle.
    if len(paths) < 5:
        return [largest_of(paths)]

    smallest = paths[0]
    largest = paths[-1]
    median = paths[len(paths) // 2]

    chosen = []
    for candidate in (largest, median, smallest):
        if candidate not in chosen and len(chosen) < ceiling:
            chosen.append(candidate)

    # Never reserve the whole corpus. A yardstick that leaves nothing to learn
    # from is not a yardstick, it is a halt: the worker would have nothing to
    # select from and every cycle would end in REVIEW with no verdict.
    if len(chosen) >= len(paths):
        chosen = chosen[: len(paths) - 1]

    return chosen


def reserve(c, pieces):
    """Reserve documents as the regression yardstick.

    A document named in config wins, so a human can choose the yardstick.
    Otherwise a small stratified sample is reserved and persisted, which makes
    the choice stable across runs and representative rather than arbitrary.

    Returns (reserved_paths, remaining_pieces). Reserving everything is refused:
    that would leave the worker with nothing to learn from and no way to detect
    forgetting.

    Nothing is deleted and nothing is moved. Reserving means withholding from
    the candidate list; the document stays in data/curated exactly where it is,
    because the corpus is the accumulated result of everything Jupiter has ever
    learned and a document removed from it today is material no later run can
    recover.
    """
    if len(pieces) < 2:
        raise RuntimeError(
            f"Regression detection needs at least 2 learnable documents, "
            f"found {len(pieces)}. The worker cannot both learn and verify."
        )
    state_path = Path(c.regression_state)
    already = load_reserved(c)
    if c.regression_source:
        chosen = [Path(c.regression_source)]
        missing = [str(path) for path in chosen if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Configured regression source does not exist: {', '.join(missing)}"
            )
    elif already and len(already) > 1 and all(path.exists() for path in already):
        # An existing stratified sample is kept, so the yardstick does not move
        # between runs. A single document is NOT kept: that is the reservation
        # this replaces, and honouring it would mean the fix never takes effect
        # on a project that already had one. The document is not deleted either
        # way; it simply goes back into the candidate list.
        chosen = already
    else:
        # Either nothing was reserved before, or what was reserved has since
        # been moved, renamed or deleted. Choosing a document the worker cannot
        # read would return a path that is not among the candidates, and the
        # caller would then look up its text in a map that does not contain it.
        chosen = choose_yardstick(c, pieces)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "reserved": str(chosen[0]),
                "documents": [str(path) for path in chosen],
                "reason": (
                    "held out of learning candidates and re-measured every "
                    "cycle to detect catastrophic forgetting. A stratified "
                    "sample by document size, because one document cannot "
                    "detect a regression in material unlike itself. Not "
                    "deleted and not moved: withheld only."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    held = {path.resolve() for path in chosen}

    # Pieces arrive as train.Candidate objects, which know their path and not
    # their text, because the corpus is not read in order to be ranked. They are
    # passed through untouched: whatever was not held out is still a candidate,
    # and rebuilding it as a (path, text) pair here would mean reading the whole
    # corpus in the one function whose job is to hold a few documents out of it.
    def path_of(piece):
        return piece[0] if isinstance(piece, tuple) else piece.path

    remaining = [piece for piece in pieces if path_of(piece).resolve() not in held]
    return chosen, remaining


def tokens_for(c, text, tokenizer):
    """Build the fixed holdout tensor for the reserved document.

    Only the holdout half is kept. The reserved document is never trained on,
    so its training half serves no purpose and would only invite misuse.
    """
    _, holdout = split_tokens(text, c.block_size, c.holdout_fraction, tokenizer)
    return holdout


def measure(model, c, holdout_tokens):
    """Return the loss of the current model on the regression document."""
    return loss_on(holdout_tokens, model, c)
