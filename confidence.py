"""Per-source confidence tracking (Architecture 13).

Confidence is the current strength of evidence behind what Jupiter has
learned from a source. It is deliberately not a truth value and not a
replacement for the knowledge itself.

Each learning source keeps:

    confidence     0.0 - 1.0
    evidence       cycles that produced a measured gain, or a review
    contradictions cycles that regressed or were rejected
    accepted       updates that were kept
    observed       how many times the source has been judged at all
    last_verified  unix time of the most recent measurement

Competing evidence is preserved rather than overwritten: a source that
regresses keeps its history and only has its confidence lowered. A DEFER carries
no evidence either way and leaves confidence alone, because the update moved by
less than the dead band and nothing was measured. A REVIEW does move it, weakly:
Architecture 13 treats accumulated observations as evidence and says the system
can update confidence as further evidence arrives, so a review that is not a
contradiction has to count for something. Treating it as nothing at all made the
review floor a latch rather than a floor.
"""

import json
import time
from pathlib import Path

# How much of the accept gain a reviewed cycle contributes. Below 1.0 so an
# accepted update stays far stronger evidence than a reviewed one, and so a
# source has to be reviewed many times, without damaging anything, before it is
# trusted enough to be published.
REVIEW_EVIDENCE_WEIGHT = 0.25


def load(c):
    """Read stored confidence state, or an empty map if there is none."""
    path = Path(c.confidence_state)
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        # Corrupt confidence state must not stop learning. It resets to empty
        # and is rebuilt from later evidence.
        return {}
    if not isinstance(state, dict):
        return {}
    return state


def save(c, state):
    path = Path(c.confidence_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def blank():
    """A source that has not been judged yet carries no evidence."""
    return {
        "confidence": 0.0,
        "evidence": 0,
        "contradictions": 0,
        "accepted": 0,
        "observed": 0,
        "last_verified": None,
    }


def entry_for(state, source):
    """Return the record for a source, creating an unobserved one if new."""
    key = str(source)
    record = state.get(key)
    if not isinstance(record, dict):
        record = blank()
        state[key] = record
    return record


def value_of(record):
    """Read the confidence value defensively, defaulting to no confidence."""
    try:
        return max(0.0, min(1.0, float(record.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def observed_of(record):
    """How many times this source has been judged, defaulting to zero."""
    try:
        return max(0, int(record.get("observed", 0)))
    except (TypeError, ValueError):
        return 0


def observe(c, state, source, verdict):
    """Fold one cycle's verdict into the confidence for this source."""
    record = entry_for(state, source)
    value = value_of(record)
    if verdict == "ACCEPT":
        value += (1.0 - value) * c.confidence_gain
        record["evidence"] = int(record.get("evidence", 0)) + 1
        record["accepted"] = int(record.get("accepted", 0)) + 1
    elif verdict == "REVIEW":
        # A review is an observation of the source, not a contradiction of it,
        # and Architecture 13 models confidence as evidence accumulated against
        # contradictions found: its own example carries 147 observations beside
        # 2 contradictions. It also states that the system "can later update
        # these values as additional evidence becomes available".
        #
        # Leaving a review confidence-neutral contradicted that. The only
        # verdict that raised confidence was ACCEPT, and the review floor is
        # precisely what withholds ACCEPT, so a source that fell below the floor
        # could never rise above it again for the rest of the run. Confidence
        # stopped being a measurement and became a latch. A source that has
        # since stopped damaging the reserved document was denied forever the
        # chance to be trusted again, which is the opposite of what the floor
        # was introduced to do.
        #
        # The weight is a fraction of the accept gain, so an accepted update
        # remains far stronger evidence than a reviewed one, and REJECT and
        # ROLLBACK still subtract. A source that keeps damaging the reserved
        # document is not being reviewed, so it still cannot climb.
        value += (1.0 - value) * c.confidence_gain * REVIEW_EVIDENCE_WEIGHT
        record["evidence"] = int(record.get("evidence", 0)) + 1
    elif verdict == "ROLLBACK":
        value -= value * c.confidence_loss
        record["contradictions"] = int(record.get("contradictions", 0)) + 1
    elif verdict == "REJECT":
        value -= value * c.confidence_loss * 0.5
        record["contradictions"] = int(record.get("contradictions", 0)) + 1
    # DEFER is left exactly as it was: the update moved by less than the dead
    # band, so there was no measurement in either direction to record.
    record["confidence"] = max(0.0, min(1.0, value))
    record["observed"] = observed_of(record) + 1
    record["last_verified"] = time.time()
    save(c, state)
    return record
