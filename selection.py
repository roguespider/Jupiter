"""Choosing what to learn next, from what the last cycles actually showed.

The worker used to walk its documents in the order `sorted()` returned them,
which is alphabetical, and take the next one each cycle. That is a directory
listing, not a curriculum. Nothing in the project asked which material Jupiter
was worst at, and nothing acted on the answer even where one existed.

Architecture 21 measures repeated learning difficulty and names seven causes
for it. Two of them are things more data can fix: insufficient data and
inadequate context. The other five are not. A representation the model cannot
compress does not become compressible by being shown more of the same thing,
and Architecture 23 is explicit that expansion must be justified rather than
automatic, which is the same caution applied to the cheap remedy. So the causes
are not a ranking, they are a gate.

This module turns those measurements into an order. The ordering is
deliberately total and deterministic: Architecture 29 asks for experiments to be
reproducible, and a queue that reshuffles itself between runs makes two runs
incomparable for reasons that have nothing to do with learning.

Starvation is the failure mode a priority queue always has. A document the model
is permanently bad at would be chosen forever and the rest of the corpus would
never be revisited, so every document carries an age term that reaches its
maximum after one full rotation. Frustration decides who goes first within a
rotation; the age term guarantees everybody is seen at least once per rotation
however badly they are doing.
"""

import json
import time
from pathlib import Path

import skills

# How much a frustration band argues for revisiting. Chosen so that an unjudged
# or comfortable source is still learnable, just later.
BAND_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.2, "": 0.2}

# Confidence is Architecture 13's measure of how much evidence a source has
# earned. A source with little of it has not been shown to be learnable, and
# that is a reason to look again rather than a reason to write it off.
CONFIDENCE_WEIGHT = 0.5

# The causes Architecture 21 names that more of the same material can address.
# Everything else in that list is a representation or capacity problem, and
# showing a model more of what it already cannot compress is how a corpus gets
# reinforced rather than learned.
GENERATION_CAUSES = frozenset({"inadequate context", "insufficient data"})


def key_for(path) -> str:
    """The name every module identifies a learning source by.

    The worker records confidence, experiments and frustration against the bare
    filename, because that is what it has always recorded and changing it would
    orphan every existing record. Candidates arrive as full paths, so joining
    the two on the path string silently matches nothing, and a queue with no
    signals degrades to alphabetical order while looking healthy. Normalising
    once, here, is cheaper than discovering that twice.
    """
    return Path(path).name


def load(c):
    """Read the selection state, or an empty map when there is none."""
    path = Path(c.selection_state)
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return state if isinstance(state, dict) else {}


def save(c, state):
    """Persist the selection state, keeping only what is still learnable."""
    path = Path(c.selection_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def seen_cycle(state, source):
    """The last cycle this source was learned in, or None if never."""
    record = state.get(key_for(source))
    if not isinstance(record, dict):
        return None
    try:
        return int(record.get("cycle"))
    except (TypeError, ValueError):
        return None


def mark_seen(state, source, cycle):
    state[key_for(source)] = {"cycle": int(cycle), "seen": time.time()}


def band_weight(band):
    return BAND_WEIGHT.get(str(band), BAND_WEIGHT[""])


def need_for(band, confidence):
    """How strongly a source argues for being learned now.

    Deliberately not a probability and not a score of quality. It is a
    preference, and the age term in rank() is what stops it becoming a verdict.
    """
    weight = band_weight(band)
    try:
        held = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        held = 0.0
    return weight + (1.0 - held) * CONFIDENCE_WEIGHT


def age_term(seen, cycle, rotation):
    """0.0 for a source learned this cycle, 1.0 after a full rotation.

    This is the starvation guard. Without it a permanently difficult source
    would be chosen every cycle and nothing else would ever be revisited.
    """
    if seen is None:
        return 1.0
    age = max(0, int(cycle) - int(seen))
    return min(1.0, age / max(1, int(rotation)))


def rank(pieces, signals, state, cycle, folder_bonus=None):
    """Order candidates by need, oldest first among equals.

    `pieces` is the (path, text) list the worker already discovered. `signals`
    maps a source path to {"band": ..., "confidence": ...}. `folder_bonus` maps a
    skill folder to the priority its weakness earns, from skills.py. Sorting is
    by descending priority, and ties break on the path so the order is total and
    reproducible between runs.

    The folder bonus is added last and capped well below the need signal on
    purpose. The two answer different questions: need_for asks how hard this
    document has been to learn, and the bonus asks how much the capability it
    teaches is wanted. A document nobody can learn is still a document nobody
    can learn, and letting a category score talk its way past that would be
    learning the wrong thing confidently rather than the right thing slowly.
    """
    rotation = max(1, len(pieces))
    scored = []
    for piece in pieces:
        # A candidate may be a train.Candidate, which knows its path and not its
        # text, or a plain (path, text) pair from regression.reserve(). Ranking
        # needs the path alone, so both shapes are accepted and the piece is
        # carried through untouched for whichever caller needs it.
        if isinstance(piece, tuple):
            path = piece[0]
        else:
            path = piece.path
        signal = signals.get(key_for(path), {})
        band = signal.get("band", "")
        confidence = signal.get("confidence", 0.0)
        novelty = age_term(seen_cycle(state, path), cycle, rotation)
        priority = need_for(band, confidence) + novelty
        if folder_bonus:
            folder = skills.folder_of(path)
            if folder is not None:
                # Clamped again here because rank() is public and a caller may
                # hand it a raw weakness rather than the capped bonus. The cap
                # is the whole reason this is a preference, so it is enforced
                # where the number is used rather than trusted to the caller.
                priority += min(
                    skills.MAX_BONUS, max(0.0, float(folder_bonus.get(folder, 0.0)))
                )
        scored.append((-priority, str(path), path, piece))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(item[2], item[3]) for item in scored]


def choose(pieces, signals, state, cycle, folder_bonus=None):
    """The next document to learn, and why it was chosen.

    Returns (path, piece, reason) so the worker can record the reason. A
    recorded reason is what makes an experiment readable later: without it a
    curriculum change looks like a random walk.

    The second value is the piece itself rather than its text, because a
    candidate may not have been read yet and reading one here would put the
    corpus back in memory at the moment of ranking. The worker reads the winner
    once it has chosen it.

    The reason names the skill bonus as well as the need, so a curriculum that
    changed because a capability was measured to be weak is legible from the
    record rather than looking like the queue reshuffled on its own.
    """
    if not pieces:
        return None, None, "no candidate available"
    ordered = rank(pieces, signals, state, cycle, folder_bonus=folder_bonus)
    path, piece = ordered[0]
    signal = signals.get(key_for(path), {})
    reason = (
        f"need {need_for(signal.get('band', ''), signal.get('confidence', 0.0)):.2f} "
        f"from band {signal.get('band') or 'unjudged'} and confidence "
        f"{float(signal.get('confidence', 0.0)):.2f}, "
        f"age {age_term(seen_cycle(state, path), cycle, max(1, len(pieces))):.2f}"
    )
    if folder_bonus:
        folder = skills.folder_of(path)
        bonus = (
            min(skills.MAX_BONUS, max(0.0, float(folder_bonus.get(folder, 0.0))))
            if folder
            else 0.0
        )
        if bonus:
            reason += f", skill {folder} is weak so +{bonus:.2f}"
    return path, piece, reason


def worth_generating(cause):
    """Whether more material of this kind could plausibly help.

    A gate, not a ranking. Architecture 21 lists seven causes; only the two
    here are answered by showing more of the same thing. A source the model
    cannot represent is not improved by more examples of its own failure.
    """
    return str(cause) in GENERATION_CAUSES
