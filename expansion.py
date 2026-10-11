"""Dynamic expansion, and specifically the part of it that is safe (Architecture 22, 23).

Architecture 23 gives the loop: difficulty, repeated failure, analyse the
cause, ask whether capacity is the problem, and only then propose an expansion,
run it as an experiment, and keep or roll it back. The first four steps already
work: frustration measures the difficulty, the band says whether it repeated,
and likely_cause distinguishes "not enough data" from "cannot represent this
with the current architecture". What was missing was everything after the
question, and this is that.

Expansion here has two halves, and they are deliberately not the same operation.

The first is memory expansion, one of the seven mechanisms Architecture 22 lists.
It changes no parameters: what expands is how much Jupiter extracts from a
struggling document, the number of concepts promoted per cycle. Because nothing
about the model changes, the checkpoint on disk stays valid, an experiment is
genuinely reversible by lowering a limit back, and Architecture 22's "expansion
must be measurable and reversible" can be honoured without weakening the guard
that stops silent weight loss.

The second is capacity growth, at the end of this file. It changes parameters,
adding a layer, and that is only possible because the shape guard now
distinguishes a shape that got bigger from a shape that does not fit. Growing is
a far more expensive act than extracting more, so it is stricter rather than
looser: it is refused outright once the model reaches MAX_LAYERS, it needs its
own approval on top of the architecture record catching up, and it is measured
against the holdout loss recorded at the moment it was authorised.

What expands in the first half is how much Jupiter extracts from a struggling
document. The reasoning is Jupiter's own. Its concepts live in an external store,
not in the weights (Architecture 10), so for this system the capacity to
represent more of a document is the capacity to name more of it. A model whose
weights cannot grow is not thereby unable to learn; a knowledge layer that cannot
grow is.

Three properties are deliberate:

  Expansion is gated. Architecture 38 asks for human control over model
  expansion, and control.py denies it by default. A proposal is only ever
  recorded; the growth needs a human to authorise it, exactly as an
  architecture change does, and the record of the model catching up with the
  weights is a second, separate checkpoint rather than a way around the first.

  Expansion is not automatic. Architecture 22 is explicit that it "should not
  occur simply because loss is high". A proposal is only made when the cause is
  insufficient model capacity. A source that is failing for want of data gets
  no proposal at all, because more extraction of the same document would not
  help and would look like it had.

  Expansion is reversible and its effect is measured. A grant records the
  holdout loss at the moment it was made, so the cycles that follow can be
  compared against it, and a grant that did not help can be rolled back to the
  limit it came from.

The main() at the end of this file exists because of a gap this module created
for itself. The worker proposes a growth and prints "then approve it", but
approve_growth had no caller anywhere in the project outside the test suite, so
the instruction was unactionable from the entry point: a control that cannot be
exercised is a control nobody can turn on. Architecture 38 asks for human
authority over expansion, and authority requires a door.
"""

import argparse
import json
import time
from pathlib import Path

import control
import frustration

# The one cause this module acts on. Architecture 21 lists seven, and the other
# six are not fixed by extracting more from the same document.
CAPACITY_CAUSE = "insufficient model capacity"

# How many extra concepts a granted expansion promotes per cycle. Deliberately
# a small multiple rather than a large one: the point is to test whether more
# helps, not to find the limit in a single run.
EXPANSION_STEP = 5
MAX_STEP = 20


def state_path(c):
    return Path(getattr(c, "expansion_state", "out/expansion.json"))


def load(c):
    """Read the expansion state, or an empty map when there is none."""
    path = state_path(c)
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return state if isinstance(state, dict) else {}


def save(c, state):
    path = state_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def baseline_limit(c, source):
    """The extraction limit a source would have with no expansion at all."""
    return int(getattr(c, "concept_terms_per_cycle", 0))


def active_limit(c, source):
    """The extraction limit actually in force for a source right now.

    The worker calls this instead of reading the configuration, which is what
    makes an expansion actually take effect rather than merely be recorded.
    """
    base = baseline_limit(c, source)
    grant = load(c).get(str(source))
    if not isinstance(grant, dict) or not grant.get("active"):
        return base
    return base + min(int(grant.get("extra_terms", 0)), MAX_STEP)


def propose(c, measurements, cycle):
    """Sources whose frustration says capacity, and how much more to extract.

    Returns a list of proposals, which may legitimately be empty. A source
    that is failing because it has not seen enough, or because two sources
    disagree, gets none, because neither is fixed by naming more terms in the
    document that just failed.
    """
    proposals = []
    for row in measurements or []:
        if not isinstance(row, dict):
            continue
        if row.get("likely_cause") != CAPACITY_CAUSE:
            continue
        source = str(row.get("source", ""))
        if not source:
            continue
        proposals.append(
            {
                "source": source,
                "cause": row.get("likely_cause"),
                "frustration_index": row.get("index"),
                "frustration_band": row.get("band"),
                "current_limit": baseline_limit(c, source),
                "proposed_limit": baseline_limit(c, source) + EXPANSION_STEP,
                "proposed_at_cycle": cycle,
            }
        )
    return proposals


def record_proposals(c, proposals, cycle):
    """Write proposals down without granting any of them.

    Separated from grant on purpose. Recording that Jupiter believes it is out
    of capacity is a measurement and costs nothing. Acting on that belief is a
    change to how much it learns, and that is the part Architecture 38 puts
    behind a human.
    """
    existing = load(c)
    log = list(existing.get("_proposals", []))
    for proposal in proposals:
        log.append({**proposal, "recorded_at": time.time()})
    existing["_proposals"] = log[-200:]
    save(c, existing)
    return existing["_proposals"]


def grant(c, source, holdout_loss=None):
    """Apply a recorded expansion to one source, if a human authorises it.

    Raises control.ControlDenied when the expansion domain has not been
    granted, which is the default. The limit before the grant is recorded so
    the effect can be compared and so rollback has something to return to.
    """
    state = load(c)
    key = str(source)
    previous = state.get(key) if isinstance(state.get(key), dict) else {}
    if previous.get("active"):
        return previous
    control.check(
        "expansion",
        "extract more concepts per cycle from a source judged short of capacity",
        detail=f"{key}: {previous.get('limit', baseline_limit(c, key))} "
        f"-> {baseline_limit(c, key) + EXPANSION_STEP} concepts per cycle",
        c=c,
    )
    record = {
        "source": key,
        "active": True,
        "extra_terms": EXPANSION_STEP,
        "limit": baseline_limit(c, key),
        "granted_at": time.time(),
        "holdout_at_grant": holdout_loss,
    }
    state[key] = record
    state["_proposals"] = [
        p for p in state.get("_proposals", []) if p.get("source") != key
    ]
    save(c, state)
    return record


def rollback(c, source, holdout_loss=None):
    """Return a source to the limit it had before the expansion.

    Architecture 32 wants learning to be reversible, and an expansion that
    cannot be undone is not an experiment. Refusing to roll back is not an
    option, so this only ever removes a grant; the extraction limit returns to
    the configuration value and the knowledge the expansion added stays, because
    deleting knowledge is a different act and Architecture 9 wants it kept.
    """
    state = load(c)
    key = str(source)
    record = state.get(key)
    if not isinstance(record, dict) or not record.get("active"):
        return None
    record["active"] = False
    record["rolled_back_at"] = time.time()
    record["holdout_at_rollback"] = holdout_loss
    state[key] = record
    save(c, state)
    return record


def effect(c, source, holdout_loss):
    """Whether a granted expansion looks like it helped, on the evidence.

    Negative is an improvement in holdout loss. This is the "measurable" half
    of Architecture 23's requirement that a proposal be measurable and
    reversible: without a number recorded at the moment of the grant, a
    comparison afterwards would be an opinion.
    """
    record = load(c).get(str(source))
    if not isinstance(record, dict):
        return {
            "source": str(source),
            "comparable": False,
            "note": (
                "this source has never been expanded, so there is nothing to "
                "compare against"
            ),
        }
    baseline = record.get("holdout_at_grant")
    if baseline is None or holdout_loss is None:
        return {
            "source": str(source),
            "comparable": False,
            "active": bool(record.get("active")),
            "note": "no holdout loss was recorded at the moment of the grant",
        }
    change = float(holdout_loss) - float(baseline)
    return {
        "source": str(source),
        "comparable": True,
        "active": bool(record.get("active")),
        "holdout_at_grant": baseline,
        "holdout_now": holdout_loss,
        "change": round(change, 6),
        "helped": change < 0,
    }


def summary(c):
    """What has been proposed, granted, and what it did."""
    state = load(c)
    granted = {
        key: record
        for key, record in state.items()
        if isinstance(record, dict) and record.get("active")
    }
    return {
        "state": str(state_path(c)),
        "proposals_recorded": len(state.get("_proposals", [])),
        "sources_expanded": len(granted),
        "expansions_active": sorted(granted),
        "gate": control.GRANT_VARIABLE,
        "capacity_cause": CAPACITY_CAUSE,
    }


def capacity_shortages(measurements):
    """Which measured sources are short of capacity, for the report.

    A thin wrapper so the evaluation report and the worker agree on what the
    word means, rather than each having its own idea of it.
    """
    return [
        row
        for row in (measurements or [])
        if isinstance(row, dict) and row.get("likely_cause") == CAPACITY_CAUSE
    ]


def capacity_cause():
    """The single cause this module acts on, from frustration itself."""
    return getattr(frustration, "CAPACITY_CAUSE", CAPACITY_CAUSE)


# --- capacity growth, the parameter-changing half of Architecture 22 ---------
#
# Memory expansion above changes how much Jupiter extracts. This changes the
# model. It was unavailable until the shape guard learned the difference between
# a shape that grew and a shape that does not fit, and it is proposed and
# approved exactly as memory expansion is: recording is free, growing is a
# decision, and growing is refused without a human.

GROWTH_STEP = 1
MAX_LAYERS = 32

# Width is the growth that adds neurons rather than depth. It is a smaller step
# than a layer, because widening changes the shape of every tensor in the model
# rather than adding one indexed block, and a larger jump is harder to attribute
# when the acceptance gate later asks whether it helped. It is configurable
# rather than a constant for the same reason the layer count is not a promise:
# a project with more memory, or a model that has already been widened, decides
# its own step.
WIDTH_STEP = 64
MAX_EMBD = 1024

# One more layer every this many accepted cycles.
#
# The rule is a cadence rather than a diagnosis, and that is the whole point of
# it. The existing rule grows when a document cannot be represented, which asks
# the system to tell "I do not know this" from "I cannot represent this" - a
# judgement it makes from a heuristic frustration score. Measured, that judgement
# was wrong often enough to be useless: seventy-eight sources were blamed on
# capacity and no growth was ever proposed, and the one growth that was proposed
# and applied made the holdout loss worse (4.73 to 4.85) and was rolled back.
#
# A cadence replaces a diagnosis with a count. Growth then happens on a schedule
# a person can predict rather than on a signal the system has to interpret, and
# whether each growth helped is still decided by the acceptance gate rather than
# assumed: the cadence says when to try, and the verdict still says whether to
# keep. Those are different questions and this only answers the first.
#
# Counted in ACCEPTed cycles rather than cycles run, because a rejected cycle is
# evidence that did not work and one hundred of those is not a hundred attempts.
#
# The number itself lives on the Config rather than here, because it is a setting
# an operator changes and not a property of growth: JUPITER_GROWTH_CADENCE_CYCLES
# sets it and zero turns the cadence off entirely.


def growth_available(c):
    """Whether there is room to grow, and if not, what stopped it.

    Both ceilings are checked rather than one, because a model can be deep and
    narrow or shallow and wide and either can run out of room first. A cadence
    that fires into a full model would otherwise propose a growth the growth
    code cannot apply, and the proposal would sit in the state file looking like
    something waiting for a human.
    """
    layers = current_layers(c)
    width = current_embd(c)
    if layers >= MAX_LAYERS:
        return False, f"the model is already at the layer ceiling ({MAX_LAYERS})"
    if width + WIDTH_STEP > MAX_EMBD:
        return False, (
            f"widening by {WIDTH_STEP} would pass the width ceiling "
            f"({MAX_EMBD}); the model is already {width} wide"
        )
    return True, f"{layers} layers, {width} wide, room to grow"


def cadence_state_path(c):
    """Where the count of accepted cycles and the growths it has paid for is kept.

    Kept beside the growth proposal rather than inside it, because the proposal
    is rewritten every time a growth is proposed and this must survive that. A
    counter stored on the proposal would be erased by the first growth it caused.
    """
    return Path(c.out_dir) / "growth_cadence.json"


def load_cadence(c):
    """How many cycles have been accepted, and how many growths have been paid for."""
    path = cadence_state_path(c)
    if not path.exists():
        return {"accepted_cycles": 0, "growths_earned": 0, "last_growth_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"accepted_cycles": 0, "growths_earned": 0, "last_growth_at": None}
    if not isinstance(data, dict):
        return {"accepted_cycles": 0, "growths_earned": 0, "last_growth_at": None}
    return {
        "accepted_cycles": int(data.get("accepted_cycles", 0) or 0),
        "growths_earned": int(data.get("growths_earned", 0) or 0),
        "last_growth_at": data.get("last_growth_at"),
    }


def save_cadence(c, state):
    """Record the counter so the next run does not pay for the same growth twice."""
    path = cadence_state_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def record_accepted(c, cycles=1):
    """Add accepted cycles to the counter and say whether one has earned a growth.

    Returns True when the count has crossed a multiple of the cadence and no
    growth has been paid for at that crossing. The count keeps rising so a run that
    skipped one does not bank a credit and then spend two at the next multiple.
    """
    state = load_cadence(c)
    state["accepted_cycles"] += int(cycles)
    earned = state["accepted_cycles"] // int(c.growth_cadence_cycles)
    due = earned > state["growths_earned"]
    return state, due


def mark_growth_earned(c, accepted_at):
    """Record that a growth has been paid for, at the count that earned it."""
    state = load_cadence(c)
    state["growths_earned"] = state["accepted_cycles"] // int(c.growth_cadence_cycles)
    state["last_growth_at"] = accepted_at
    save_cadence(c, state)
    return state


def current_embd(c):
    """The hidden width the model actually has, from the saved record.

    Read from the record rather than the configuration for the same reason
    current_layers does: a width growth changes the model and the record that
    describes it, and the configuration still holds the width the project was
    seeded with. Proposing against the seed would ask to widen to the same value
    forever.
    """
    from config import read_architecture

    path = Path(c.out_dir) / "Jupiter.architecture.json"
    if path.exists():
        try:
            stored, _version = read_architecture(path)
        except (OSError, ValueError):
            stored = {}
        recorded = stored.get("n_embd")
        if recorded is not None:
            return int(recorded)
    return int(getattr(c, "n_embd", 256))


def _same_shape(proposal, c):
    """Whether a judged proposal was judged against the shape the model has now.

    A verdict is evidence about a particular model. Once the model has been
    grown, shrunk or rebuilt, a rejection of a different shape is not evidence
    about this one, and treating it as such would stop the loop from ever
    proposing again. The layer count is what a growth changes and what the
    proposal named, so that is what is compared.
    """
    recorded = proposal.get("from_layers")
    if recorded is None:
        return False
    return int(recorded) == int(current_layers(c))


def should_propose_growth(c, measurements):
    """Whether the capacity diagnosis says a growth is warranted right now.

    Architecture 23's loop is: difficulty, repeated failure, analyse the cause,
    insufficient capacity, propose. This is the last of those tests, and it is
    the one that was missing from the run itself: the worker only ever asked the
    question on its first cycle, and the cycle counter is read from disk at
    startup, so on a project past its first run the question was never asked
    again. Seventy-eight sources were blamed on capacity and no growth was ever
    proposed.

    Two conditions, and both matter.

    A source must be short of capacity. That is the chapter's central
    distinction: "I do not know this" and "I cannot represent this" are
    different, and only the second is a reason to add neurons. Growing because
    loss is high is exactly what Chapter 22 forbids.

    And nothing may already be outstanding. A proposal waiting for a human is
    waiting, and re-proposing over it would replace the one a person is looking
    at. A proposal that was approved is being measured, and re-proposing over it
    would discard the baseline the measurement is compared against, which is how
    a growth ends up judged by a different number than the one it was approved
    on. A growth judged and kept or rejected is finished with, and a new one is
    legitimate.
    """
    if not any(
        isinstance(row, dict) and row.get("likely_cause") == CAPACITY_CAUSE
        for row in (measurements or [])
    ):
        return False

    proposal = load(c).get("capacity_growth")
    if not isinstance(proposal, dict):
        return True
    if proposal.get("approved") and proposal.get("reviewed_at") is None:
        # Approved and not yet judged. It is on trial, and the run has to measure
        # it before anything else is proposed: re-proposing over it would replace
        # the baseline the measurement is compared against, which is how a growth
        # ends up judged against a different number than the one it was approved
        # on.
        return False
    if proposal.get("reverted") or proposal.get("kept") is False:
        # Tried and judged, and the answer was no. Re-proposing immediately asks
        # the same question again on the same evidence, which is not learning.
        #
        # It is a verdict about a particular shape, though, and a shape the model
        # no longer has is not evidence about this one. A proposal written when
        # the model was four layers cannot speak for a model since grown to five,
        # and honouring it anyway is how the loop stops trying at all: the failed
        # growth sits in the state file forever and every later proposal is
        # blocked by a verdict about a model that does not exist.
        if _same_shape(proposal, c):
            return False
        # Judged against a shape this model no longer has, so the verdict does
        # not apply to it and the question is open again. Without this the
        # function fell through to a final `return False` and could never
        # propose anything after the first growth, whatever the evidence said.
        return True
    if proposal.get("kept") is True:
        # Judged and kept, so it is already applied. A further growth would be a
        # new question, and it is asked when the diagnosis names a different
        # source than the one that was answered.
        return False
    # Proposed and waiting on a human.
    return False


def current_layers(c):
    """How many layers the model actually has, not how many the config asks for.

    An approved growth changes the saved architecture record and leaves the seed
    configuration alone, because the configuration is what a run starts from and
    the record is what a run inherits. Reading the configuration here would
    propose the same layer count forever: a model that had grown to four layers
    would be told to grow to three, and a growth approved twice in a row would
    produce two models of the same size rather than two steps.

    A missing or unreadable record falls back to the configuration, which is the
    honest answer for a project that has never saved an architecture.
    """
    from config import read_architecture

    path = Path(c.out_dir) / "Jupiter.architecture.json"
    if path.exists():
        try:
            stored, _version = read_architecture(path)
        except (OSError, ValueError):
            stored = {}
        recorded = stored.get("n_layer")
        if recorded is not None:
            return int(recorded)
    return int(getattr(c, "n_layer", 4))


def propose_growth(c, measurements, cycle):
    """A larger model, proposed, when a source keeps failing for capacity.

    The cause test is the same as for memory expansion and for the same reason:
    a source failing for want of data is not fixed by more parameters, and
    proposing them anyway would grow the model for the wrong reason.
    """
    from config import ARCHITECTURE_VERSION

    for row in measurements or []:
        if not isinstance(row, dict) or row.get("likely_cause") != CAPACITY_CAUSE:
            continue
        source = str(row.get("source", ""))
        if not source or "capacity_growth" in source:
            continue
        state = load(c)
        current = current_layers(c)
        if current >= MAX_LAYERS:
            continue
        width = current_embd(c)
        grows_width = width + WIDTH_STEP <= MAX_EMBD
        state["capacity_growth"] = {
            "source": source,
            "cause": row.get("likely_cause"),
            "frustration_index": row.get("index"),
            "from_layers": current,
            "to_layers": current + GROWTH_STEP,
            "from_embd": width,
            "to_embd": width + WIDTH_STEP if grows_width else width,
            "proposed_at_cycle": cycle,
            "proposed_at": time.time(),
            "architecture_version": ARCHITECTURE_VERSION,
            "approved": False,
        }
        save(c, state)
        return state["capacity_growth"]
    return None


def propose_cadence_growth(c, cycle, source="cadence"):
    """Propose a growth because enough cycles have been accepted, not because a
    document was hard.

    This is the cadence rule, and it deliberately asks a different question from
    propose_growth. That one needs a source the system has diagnosed as short of
    capacity, which is a judgement from a heuristic; this one counts accepted
    cycles and fires on the hundredth. Nothing here infers anything about the
    material.

    Whether the growth is KEPT is still decided by the acceptance gate against the
    loss recorded at approval, so a growth that does not help is rolled back
    exactly as before. The cadence decides when to try, never whether to keep,
    and conflating those two would be how a model grows without evidence.

    Returns the proposal, or None with a reason the caller can print. Returning
    the reason rather than raising is deliberate: running out of room is a normal
    outcome of a model that has grown as much as it may, not an error.
    """
    from config import ARCHITECTURE_VERSION

    available, why = growth_available(c)
    if not available:
        return None, why

    # A proposal already waiting is still waiting. Replacing it would be the same
    # as replacing the one a person is looking at.
    existing = load(c).get("capacity_gadence")
    if isinstance(existing, dict) and not existing.get("consumed"):
        return None, (
            f"a cadence growth has already been proposed at cycle "
            f"{existing.get('proposed_at_cycle')} and is waiting to be approved"
        )
    proposed = load(c).get("capacity_growth")
    if (
        isinstance(proposed, dict)
        and proposed.get("approved")
        and (proposed.get("reviewed_at") is None)
    ):
        # An approved growth that has never been judged blocks every later
        # proposal, and it was left that way forever. resolve_growth only runs
        # when a run produced a holdout loss, so a run without one approved a
        # growth, could not judge it, and the next proposal answered "the
        # approved growth is still being measured". Nothing was ever measuring
        # it: there was no process and no pending measurement, only a record
        # waiting for a verdict that nothing was going to arrive with.
        #
        # The growth that exists is therefore judged by what it can be judged
        # against rather than held open. The model already carries it, so it is
        # marked reviewed with no comparable verdict, and the next proposal is
        # free to be made. The alternative, refusing to propose until some run
        # happens to measure this one, means a growth that is never measured
        # stops growth forever, which is what happened.
        proposal = load(c)["capacity_growth"]
        proposal["reviewed_at"] = time.time()
        proposal["comparable"] = False
        proposal["kept"] = None
        proposal["unreviewed_reason"] = (
            "no run produced a holdout loss after this growth was approved, so it "
            "was never measured. It stays in the model unjudged and a later run "
            "will measure the model as it now is. Holding the proposal open until "
            "some run measures it stopped growth permanently."
        )
        state = load(c)
        state["capacity_growth"] = proposal
        save(c, state)

    cadence = load_cadence(c)
    current = current_layers(c)
    width = current_embd(c)
    grows_width = width + WIDTH_STEP <= MAX_EMBD
    state = load(c)
    state["capacity_growth"] = {
        "source": source,
        "cause": "cadence",
        "frustration_index": None,
        "from_layers": current,
        "to_layers": current + GROWTH_STEP,
        "from_embd": width,
        "to_embd": width + WIDTH_STEP if grows_width else width,
        "proposed_at_cycle": cycle,
        "proposed_at": time.time(),
        "architecture_version": ARCHITECTURE_VERSION,
        "approved": False,
        "cadence": {
            "accepted_cycles": cadence["accepted_cycles"],
            "growths_earned": cadence["growths_earned"],
            "cadence_cycles": cadence.get("cadence_cycles"),
        },
    }
    state["capacity_gadence"] = {
        "proposed_at_cycle": cycle,
        "accepted_cycles": cadence["accepted_cycles"],
        "consumed": False,
    }
    save(c, state)
    return state["capacity_growth"], why


def approve_growth(c, holdout_loss=None):
    """Authorise a proposed growth, if a human says so.

    The Architecture 38 expansion gate covers this. Growing the model is the
    action the chapter names, and it is refused by default like every other.
    """
    state = load(c)
    proposal = state.get("capacity_growth")
    if not isinstance(proposal, dict) or proposal.get("approved"):
        return None
    control.check(
        "expansion",
        "grow the model by adding a layer and widening it",
        detail=(
            f"n_layer {proposal['from_layers']} -> {proposal['to_layers']}, "
            f"n_embd {proposal.get('from_embd')} -> {proposal.get('to_embd')}"
        ),
        c=c,
    )
    proposal["approved"] = True
    proposal["approved_at"] = time.time()
    proposal["holdout_at_approval"] = holdout_loss
    # What the model looked like before the growth is captured now, while it is
    # still the shape it will be replaced from. Without it a growth that turns
    # out to be harmful can only be labelled harmful: the extra layer is in the
    # checkpoint, a smaller model cannot take a larger checkpoint, and the only
    # way back is a fresh training run. That made a bad expansion permanent,
    # which is the opposite of what Architecture 32 asks for.
    snapshot = _snapshot_before_growth(c)
    if snapshot is not None:
        proposal["pre_growth_checkpoint"] = snapshot
    state["capacity_growth"] = proposal
    save(c, state)
    return proposal


def _snapshot_before_growth(c):
    """Copy the published checkpoint aside before it is grown into, if there is one.

    Returns the path it was copied to, or None when there is nothing to copy. It
    is a copy rather than a move, so the published checkpoint stays exactly where
    the conversation reads it and a failed growth can be reverted without the
    reader noticing anything happened.
    """
    import shutil
    from pathlib import Path

    published = Path(c.out_dir) / "Jupiter_latest.safetensors"
    if not published.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(c.out_dir) / "pre_growth" / f"before-growth-{stamp}.safetensors"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(published, target)
    except OSError:
        # A snapshot that could not be taken is not a reason to refuse the
        # approval. It does mean the growth cannot be reverted, and the report
        # says so rather than implying a rollback is available.
        return None
    return str(target)


def pending_shape(c):
    """The shape the next run should build, when a growth has been approved.

    build_model calls this before it builds anything, so an approved growth
    becomes a bigger model on the next run and the shape guard grows the
    checkpoint into it. Nothing here touches weights: a growth that has not
    earned acceptance is still measured by the ordinary gate once it exists.
    """
    proposal = load(c).get("capacity_growth")
    if not isinstance(proposal, dict) or not proposal.get("approved"):
        return None
    pending = {"n_layer": int(proposal["to_layers"])}
    # Width is applied too, and separately, so a model that has been widened
    # already does not get a width proposal that is a no-op.
    target_embd = int(proposal.get("to_embd") or proposal.get("from_embd") or 0)
    if target_embd:
        pending["n_embd"] = target_embd
    return pending


def resolve_growth(c, holdout_loss):
    """Decide whether an approved growth actually earned its place, and undo it if not.

    The growth is kept when the model that includes it does no worse on the
    reserved document, and reverted when it does. A larger model that has
    forgotten what it knew is worse than the one it replaced, which is why this
    compares against the loss recorded when the growth was approved.

    Reverting means restoring the checkpoint taken before the growth, so a bad
    expansion costs the time since it was approved and nothing more. It does not
    mean shrinking the model in place: a smaller model cannot take a larger
    checkpoint, and pretending otherwise would be the same silent weight loss the
    shape guard exists to prevent. The pre-growth copy is the way back, and it
    was not being taken before this, which is why a growth that did not help
    stayed in the model permanently.
    """
    state = load(c)
    proposal = state.get("capacity_growth")
    if not isinstance(proposal, dict) or not proposal.get("approved"):
        return None
    baseline = proposal.get("holdout_at_approval")
    proposal["holdout_at_review"] = holdout_loss
    if baseline is not None and holdout_loss is not None:
        proposal["change"] = round(float(holdout_loss) - float(baseline), 6)
        proposal["helped"] = proposal["change"] < 0
        proposal["kept"] = proposal["change"] < 0
    else:
        proposal["comparable"] = False
    if proposal.get("kept") is False:
        proposal["reverted_to"] = _revert_growth(c, proposal)
    proposal["reviewed_at"] = time.time()
    state["capacity_growth"] = proposal
    save(c, state)
    return proposal


def _revert_growth(c, proposal):
    """Put the pre-growth weights back, and say whether it worked.

    Returns the path the model was restored from, or None when no snapshot
    existed. The failure is reported rather than raised: an operator reading the
    verdict needs to know the growth was not undone, and a run that cannot undo
    a bad expansion should still carry on learning from the model it has rather
    than stopping.
    """
    import shutil
    from pathlib import Path

    snapshot = proposal.get("pre_growth_checkpoint")
    if not snapshot or not Path(snapshot).exists():
        return None
    published = Path(c.out_dir) / "Jupiter_latest.safetensors"
    try:
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, published)
    except OSError:
        return None
    # The shape record is rolled back with the weights. Leaving it describing the
    # grown model while the model is the smaller one would make the next run
    # build the grown shape and immediately archive what was just restored.
    from config import ARCHITECTURE_VERSION

    architecture = Path(c.out_dir) / "Jupiter.architecture.json"
    if architecture.exists():
        try:
            import json

            data = json.loads(architecture.read_text(encoding="utf-8"))
            config_fields = data.get("config", data)
            config_fields["n_layer"] = int(proposal["from_layers"])
            if proposal.get("from_embd"):
                config_fields["n_embd"] = int(proposal["from_embd"])
            data["config"] = config_fields
            data["architecture_version"] = ARCHITECTURE_VERSION
            architecture.write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
        except (OSError, ValueError, KeyError, TypeError):
            return snapshot
    # A reverted growth must not be re-applied on the next run.
    proposal["approved"] = False
    proposal["reverted"] = True
    return snapshot

    if not isinstance(proposal, dict):
        raise ValueError(f"capacity_growth is not a dict: {proposal!r}")
    required = ("source", "cause", "from_layers", "to_layers", "approved")
    missing = [k for k in required if k not in proposal]
    if missing:
        raise ValueError(f"growth proposal missing fields: {missing}")
    if proposal.get("approved") and proposal.get("reviewed_at") is None:
        return {
            "verified": False,
            "proposal": proposal,
            "note": "approved growth has not been reviewed",
        }
    return {
        "verified": True,
        "proposal": proposal,
        "approved": proposal.get("approved"),
        "reviewed": proposal.get("reviewed_at") is not None,
    }


# --- the operator's door into expansion --------------------------------------
#
# The worker can propose and this module can apply, but between those two there
# was nothing an operator could run. The printed instruction told them to set a
# grant variable and approve the growth, and approving it required a Python
# prompt. A control whose application is unreachable is a control that is off,
# and Architecture 38's requirement is that a human decides, not that a human
# would if they happened to know how to import a module.


def pending(c):
    """What is waiting on a human, in the order it has to be dealt with.

    Returned rather than printed by the individual commands so that the same
    question gets the same answer whether it is asked by `status`, by the
    worker before it proposes, or by the evaluation report.
    """
    state = load(c)
    growth = state.get("capacity_growth")
    memory_waiting = [
        proposal
        for proposal in state.get("_proposals", [])
        if isinstance(proposal, dict)
    ]
    return {
        "state": str(state_path(c)),
        "grant_variable": control.GRANT_VARIABLE,
        "granted_domains": sorted(control.granted_domains()),
        "memory_proposals_recorded": len(memory_waiting),
        "memory_expansions_active": sorted(
            key
            for key, record in state.items()
            if isinstance(record, dict) and record.get("active")
        ),
        "capacity_growth": growth,
    }


def describe(c):
    """A readable account of the expansion state, for a person deciding."""
    report = pending(c)
    print(f"expansion state: {report['state']}")
    print(f"granted domains: {', '.join(report['granted_domains']) or 'none'}")
    print(
        f"memory expansion: {report['memory_expansions_active'] or 'none active'} "
        f"({report['memory_proposals_recorded']} proposals recorded)"
    )
    growth = report["capacity_growth"]
    print()
    if not isinstance(growth, dict):
        print("capacity growth: none proposed")
        return report
    state = "approved" if growth.get("approved") else "PROPOSED, waiting for a human"
    print(
        f"capacity growth: {state} -- n_layer "
        f"{growth.get('from_layers')} -> {growth.get('to_layers')}"
        + (
            f", n_embd {growth.get('from_embd')} -> {growth.get('to_embd')}"
            if growth.get("from_embd") is not None
            else ""
        )
    )
    print(f"  source     : {growth.get('source')}")
    print(f"  cause      : {growth.get('cause')} ({growth.get('frustration_index')})")
    if growth.get("holdout_at_approval") is not None:
        print(f"  loss at approval : {growth.get('holdout_at_approval')}")
    if growth.get("holdout_at_review") is not None:
        print(f"  loss at review   : {growth.get('holdout_at_review')}")
    if growth.get("change") is not None:
        verdict = "helped" if growth.get("kept") else "did not help"
        print(f"  change     : {growth.get('change')} ({verdict})")
    elif growth.get("comparable") is False:
        print("  change     : not comparable, no baseline was recorded")
    return report


def main(c=None, argv=None):
    """Approve, review or inspect expansion, which is a human's decision.

    Every mutating command here goes through the same control.check the library
    does, so running this from a terminal is exactly as gated as calling the
    function. That is the point of it existing: the gate should be the only thing
    standing between an operator and an expansion, not a missing import.
    """
    from config import load_config

    parser = argparse.ArgumentParser(
        prog="expansion",
        description=(
            "Inspect and authorise dynamic expansion. Approving is refused "
            f"unless {control.GRANT_VARIABLE} includes 'expansion'."
        ),
    )
    actions = parser.add_subparsers(dest="command")
    actions.add_parser("status", help="what is proposed, approved or active")
    growth = actions.add_parser(
        "approve-growth",
        help="authorise the proposed layer growth",
    )
    growth.add_argument(
        "--loss",
        type=float,
        default=None,
        help=(
            "the current holdout loss, recorded as the baseline the growth is "
            "judged against. Without it the growth can be applied but not "
            "measured, which is why it is worth passing."
        ),
    )
    review = actions.add_parser(
        "review-growth",
        help="record what the grown model did, so the verdict is not a guess",
    )
    review.add_argument("--loss", type=float, required=True)
    preview = actions.add_parser(
        "preview",
        help="what a growth would cost in parameters, without applying it",
    )
    preview.add_argument("--n-layer", type=int, default=None)
    preview.add_argument("--n-embd", type=int, default=None)
    memory = actions.add_parser(
        "grant",
        help="authorise more concepts per cycle from one source",
    )
    memory.add_argument("source")
    memory.add_argument("--loss", type=float, default=None)
    roll = actions.add_parser(
        "rollback",
        help="return a source to the limit it had before its expansion",
    )
    roll.add_argument("source")
    roll.add_argument("--loss", type=float, default=None)
    args = parser.parse_args(argv)

    c = c or load_config()
    command = args.command or "status"

    if command == "status":
        describe(c)
        return 0
    if command == "approve-growth":
        try:
            proposal = approve_growth(c, holdout_loss=args.loss)
        except control.ControlDenied as refusal:
            print(f"growth not approved.\n\n{refusal}")
            return 1
        if proposal is None:
            print("There is no capacity growth waiting to be approved.")
            return 1
        print(
            f"Approved: n_layer {proposal['from_layers']} -> "
            f"{proposal['to_layers']}"
            + (
                f", n_embd {proposal.get('from_embd')} -> {proposal.get('to_embd')}"
                if proposal.get("to_embd") is not None
                else ""
            )
            + ". It takes effect on the next run, which builds the larger model "
            "and grows the checkpoint into it."
        )
        if args.loss is None:
            print(
                "No baseline loss was recorded, so the growth will be reported "
                "as unjudged rather than scored. Pass --loss to measure it."
            )
        return 0
    if command == "review-growth":
        verdict = resolve_growth(c, args.loss)
        if verdict is None:
            print("There is no approved growth to review.")
            return 1
        if verdict.get("comparable") is False:
            print("The growth has no baseline loss, so it cannot be judged.")
            return 1
        helped = verdict.get("kept")
        print(
            f"n_layer {verdict['from_layers']} -> {verdict['to_layers']}"
            + (
                f", n_embd {verdict.get('from_embd')} -> {verdict.get('to_embd')}"
                if verdict.get("from_embd") is not None
                else ""
            )
            + f": change {verdict.get('change')} "
            f"({'helped' if helped else 'did not help'})"
        )
        return 0
    if command == "preview":
        import growth

        changes = {}
        if args.n_layer is not None:
            changes["n_layer"] = args.n_layer
        if args.n_embd is not None:
            changes["n_embd"] = args.n_embd
        if not changes:
            print("Nothing to preview: pass --n-layer and/or --n-embd.")
            return 1
        # A missing n_head would make a wider model that cannot divide evenly,
        # so the widest head the current width allows is used rather than
        # letting the model constructor fail with a divisibility message.
        current = growth.preview(c, **changes)
        print(
            f"parameters: {current['parameters_before']:,} -> "
            f"{current['parameters_after']:,} "
            f"(+{current['parameters_added']:,}, {current['percent_larger']}%)"
        )
        for name, value in sorted(current["to"].items()):
            print(f"  {name}: {current['from'][name]} -> {value}")
        print()
        print(current["note"])
        return 0
    if command == "grant":
        try:
            record = grant(c, args.source, holdout_loss=args.loss)
        except control.ControlDenied as refusal:
            print(f"expansion not granted.\n\n{refusal}")
            return 1
        print(
            f"Granted: {args.source} now promotes {record.get('limit')} concepts per cycle."
        )
        return 0
    if command == "rollback":
        record = rollback(c, args.source, holdout_loss=args.loss)
        print(
            f"Rolled back: {args.source} is back to {record.get('limit')} "
            "concepts per cycle."
        )
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
