"""The decision layer that makes the subsystems one agent (Architecture 37).

Architecture 37 describes a loop: Observe, Receive, Evaluate, Learn, Measure,
Store experience, Rest, Review, Continue. Every stage of it existed somewhere in
the project and none of them were connected. Frustration measured difficulty,
selection chose a source, growth could widen the model, acceptance judged the
result and rollback undid a bad update, and the worker ran them in an order
someone had written down in a batch file. What was missing was the thing that
decides, and that is what this module is.

The distinction matters. A pipeline runs because the order was written down; an
agent runs because it looked at its own state and chose. The difference is
visible in one place above all: a pipeline that finishes exits, and this decides
what to do next from the evidence it has rather than from a step counter.

Four things make it a decision layer rather than a list of ifs.

The decision is a value, not a side effect. observe() reads and returns;
decide() takes an observation and returns a Decision; act() carries it out. That
separation is what lets a decision be recorded, replayed and tested without
running a training cycle, which is the only way to test a thing that normally
takes twenty minutes to produce an answer.

The decision says why. Every Decision carries the evidence it was made from and
a plain sentence naming it, because a system that chooses without recording its
reasoning cannot be audited, and an unauditable autonomous system is the thing
Architecture 38 exists to prevent. The sentence is written for a reader, not for
a log parser.

Autonomy is bounded by the same gates as everything else. This module never
bypasses control.check and never decides that it is allowed to. A decision to
grow the model is a decision to ask a human; if the human has granted the
expansion domain the ask succeeds, and if not the denial is caught, recorded and
turned into a decision to keep going without growing. Refusing to expand is a
normal outcome of this loop, not a failure of it, and the loop is built to carry
on either way.

Rest is a stage, not an absence of one. Architecture 37 puts Rest between Store
and Review, and a loop that never rests is a loop that cannot notice it is
making things worse. Boredom, the urge to look at something else, is a
deliberate part of the loop here: a source that keeps winning gets deprioritised
rather than left to starve the corpus, which is the same rule selection already
applies and is stated here so the two agree.
"""

import json
import time
from pathlib import Path

import control
import experiment as experiment_module
import frustration as frustration_module
import goals

# Reachability: new subsystems are wired into the entry point through this import.
import reflection_worker
import resource_scheduler
import selection as selection_module
import store as store_module
import test_learning_restart
import tool_adapter

# Every action the loop can take. Named rather than numbered so a decision
# record reads as something a person can act on.
#
# This tuple is the whole vocabulary and decide() returns nothing outside it. It
# previously also listed OBSERVE, JUDGE, ROLLBACK, REST and STOP, none of which
# any branch could ever produce: observe is a stage the loop always passes
# through rather than a choice it can make, judge is what the acceptance gate
# does on every learn, and the other three were intentions. A vocabulary that
# names decisions the system cannot take is a list of things it wishes it did,
# and a caller checking a decision against it was being told those were
# possible.
LEARN = "learn"
EXPAND = "expand"
CONSULT = "consult"
REVIEW = "review"
REST = "rest"
STOP = "stop"

ACTIONS = (LEARN, EXPAND, CONSULT, REVIEW, REST, STOP)

# How long to wait between loop passes when there is nothing to learn, in
# seconds. Configuration rather than a constant because the right value depends
# entirely on what is being waited for: a corpus being written elsewhere wants a
# short wait, and a finished project wants a long one.
DEFAULT_IDLE_SECONDS = 30.0

# How many loop passes may end in "nothing to do" before the loop says it is
# finished. Without this a run with an empty corpus would spin forever looking
# for work that does not exist, which is the difference between resting and
# being stuck. Bounded and reported rather than open-ended.
DEFAULT_MAX_IDLE_PASSES = 20

# How many consecutive passes may consult the outside world before the loop
# goes back to resting on what it has. A world is an unbounded supply of
# documents, so a loop allowed to keep consulting it never concludes there is
# nothing coming and never reaches its own stopping rule. The bound is what
# keeps CONSULT an answer to a dry corpus rather than a reason to stop asking.
DEFAULT_MAX_CONSECUTIVE_CONSULTS = 3


class Decision:
    """One choice, with the evidence behind it and a reason a person can read.

    A plain object rather than a dict so a missing field is a crash at the point
    of construction instead of a KeyError in a log reader three hours later.
    """

    def __init__(self, action, reason, evidence=None, growth=None, holdout=None):
        if action not in ACTIONS:
            raise ValueError(
                f"unknown action {action!r}; the loop can only decide between "
                f"{', '.join(ACTIONS)}"
            )
        self.action = action
        self.reason = reason
        self.evidence = dict(evidence or {})
        self.growth = growth
        self.holdout = holdout
        self.at = time.time()

    def as_dict(self):
        """A record of the decision, for the state file and for a reader."""
        return {
            "action": self.action,
            "reason": self.reason,
            "evidence": self.evidence,
            "at": self.at,
        }

    def __repr__(self):
        return f"Decision({self.action!r}, {self.reason!r})"


def state_path(c):
    return Path(getattr(c, "autonomy_state", "out/autonomy.json"))


def load(c):
    """The loop's own state, or a fresh one when there is none.

    Read defensively on purpose. This is a state file written by a process that
    may have been killed mid-write, and a loop that cannot start because its own
    memory is corrupt would be a loop that cannot recover from anything.
    """
    import json

    path = state_path(c)
    if not path.exists():
        return {"passes": 0, "history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"passes": 0, "history": []}
    if not isinstance(data, dict):
        return {"passes": 0, "history": []}
    data.setdefault("passes", 0)
    data.setdefault("history", [])
    return data


def save(c, state):
    import json

    path = state_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The history is capped because this file is written every pass and a long
    # autonomous run would otherwise grow it without bound. The cap is on the
    # tail, so what survives is the recent past, which is what a reader wants.
    state["history"] = list(state["history"])[-200:]
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def observe(c, candidates=None, knowledge=None):
    """Everything the loop knows before it decides, gathered in one place.

    Reading the subsystems here rather than passing their outputs around is what
    makes the loop a supervisor instead of another participant. Each call is the
    same call the rest of the project already makes, so a decision is made from
    the system's real state and not from a summary that could drift.

    The loop's own recent decisions are read here too, because deciding whether
    to rest or to stop is a question about how long this has been going on and
    the answer lives in the state file rather than in any subsystem.
    """
    if knowledge is None:
        path = Path(c.knowledge_store)
        knowledge = store_module.Store(path) if path.exists() else None

    sources = frustration_module.for_sources(c)
    # The cause name comes from expansion, which owns the decision about what a
    # shortage of capacity means. Asking frustration for it would be asking the
    # module that measures difficulty to also decide what the measurement
    # implies, which is exactly the split the two modules exist to keep.
    import expansion as expansion_module

    short = expansion_module.capacity_shortages(sources)
    high_frustration = [
        row for row in sources if isinstance(row, dict) and row.get("band") == "HIGH"
    ]
    generation_needs = sorted(
        {
            cause
            for row in high_frustration
            if isinstance(row.get("likely_cause"), str)
            and (cause := row["likely_cause"])
            in {"insufficient data", "inadequate context"}
        }
    )
    experiments = experiment_module.read_experiments(c.experiment_log, limit=200)
    accepted = sum(1 for record in experiments if record.get("decision") == "ACCEPT")
    measured = [
        record
        for record in experiments
        if record.get("before_metrics", {}).get("holdout_loss") is not None
        and record.get("after_metrics", {}).get("holdout_loss") is not None
    ]
    improvements = [
        record["after_metrics"]["holdout_loss"]
        - record["before_metrics"]["holdout_loss"]
        for record in measured
    ]
    seen = selection_module.load(c)
    current = load(c)
    # Whether the world has anything to offer at all. sources() only constructs
    # source objects, so asking costs no I/O and tells the decision layer whether
    # consulting is even a thing this run can do. It is asked here rather than
    # inside decide() because decide() takes an observation and reads nothing
    # else, which is what makes a decision replayable without touching the world.
    import external as external_module

    # What Jupiter would ask the world, and only when it might have to ask.
    # Walking every concept is the expensive part of frustration.report(), so it
    # is done only on the passes where the corpus is dry, which are the only
    # passes CONSULT can be chosen on.
    #
    # The concepts are used rather than the sources, and deliberately: a curated
    # document is filed under a content hash, so its filename stem is a hex
    # string. Asking the world about "e77e6b148b0b695d" retrieves nothing while
    # looking exactly like a search that worked, which is the shape of a fake
    # capability. The concept layer is the only place Jupiter records what it
    # actually knows the name of.
    query_terms = []
    if not (candidates or []):
        import concepts as concepts_module

        if knowledge is not None:
            hardest = []
            for record in concepts_module.all_concepts(knowledge):
                row = frustration_module.for_concept(record)
                hardest.append((row["index"], row["concept"], record))
            hardest.sort(key=lambda item: (-item[0], item[1]))
            for _index, concept, record in hardest[:3]:
                # The name alone is too thin a question. Architecture 28's
                # retention test requires two shared terms, so a single-word
                # query can never satisfy it and every consult would be refused
                # for a reason that has nothing to do with the material. The
                # description is what makes the query specific enough to retrieve
                # something and to be judged against what Jupiter already holds.
                query_terms.append(concept)
                description = record.get("description")
                if isinstance(description, str):
                    query_terms.extend(sorted(external_module.terms(description))[:6])

    return {
        "candidates": list(candidates or []),
        "candidate_count": len(candidates or []),
        "external_configured": bool(external_module.sources(c)),
        "consecutive_consults": int(current.get("consecutive_consults", 0) or 0),
        "query_terms": query_terms,
        "sources_measured": len(sources),
        "capacity_shortages": len(short),
        "high_frustration_count": len(high_frustration),
        "generation_needs": generation_needs,
        "cycles_recorded": len(experiments),
        "cycles_accepted": accepted,
        "mean_improvement": (
            round(sum(improvements) / len(improvements), 6) if improvements else None
        ),
        "improving": bool(improvements) and (sum(improvements) / len(improvements)) < 0,
        "sources_seen": len(seen.get("seen", {})) if isinstance(seen, dict) else 0,
        "granted": sorted(control.granted_domains()),
        "passes": int(current.get("passes", 0)),
        "idle_passes": int(current.get("idle_passes", 0)),
    }


def decide(c, observation, growth_state=None):
    """Choose the next action from what was observed, and say why.

    The order of the tests is the policy, and it is deliberately defensive
    before it is ambitious. Reviewing past work comes before proposing new work,
    and rest comes before expansion, because a loop that expands while it is
    not learning is a loop making the model larger without making it better.

    Every branch names its reason in a sentence rather than a code, because the
    record of this decision is the only thing a person has when they ask why the
    agent did that. The actions returned are the whole of ACTIONS: judging an
    update is the acceptance gate's job on every learn rather than a separate
    decision, and rollback happens inside a learn when one is rejected, so
    neither is a choice this loop gets to make.
    """
    granted = set(observation.get("granted", []))

    # Consecutive passes with nothing to learn are what REST and STOP are for.
    # The check is before the candidate check because a loop that has concluded
    # there is nothing coming should say so and stop, rather than reporting IDLE
    # forever and relying on the caller to notice.
    idle_passes = int(observation.get("idle_passes", 0) or 0)
    idle_limit = int(getattr(c, "autonomy_max_idle_passes", DEFAULT_MAX_IDLE_PASSES))
    if idle_passes >= idle_limit:
        return Decision(
            STOP,
            f"nothing has been learnable for {idle_passes} consecutive passes, so "
            "the loop is concluding there is nothing coming rather than waiting "
            "forever; add material to the corpus or raise "
            "autonomy_max_idle_passes to wait longer",
            evidence={"idle_passes": idle_passes, "idle_limit": idle_limit},
        )

    if observation["candidate_count"] == 0:
        # A dry corpus is the one place where the world is the obvious answer,
        # because Architecture 28 says external knowledge exists to be consulted
        # and this was the branch that gave up. It consults only when a source is
        # configured and the run is not already inside its consult bound, so a
        # world that never yields anything becomes a bounded number of attempts
        # and then a REST like any other empty pass rather than an open-ended
        # search that never reaches the stopping rule above.
        if observation.get("external_configured"):
            consults = int(observation.get("consecutive_consults", 0) or 0)
            consult_limit = int(
                getattr(
                    c,
                    "autonomy_max_consecutive_consults",
                    DEFAULT_MAX_CONSECUTIVE_CONSULTS,
                )
            )
            if consults < consult_limit:
                if "external" not in granted:
                    # The same shape as the expansion refusal: the loop can see
                    # that the world would answer and cannot ask it, so it says
                    # so in the reason rather than resting as though nothing were
                    # there. Refusing is a normal outcome, not a failure.
                    return Decision(
                        CONSULT,
                        "the corpus is dry and external sources are configured, but "
                        "a human has not granted the external domain, so Jupiter "
                        f"cannot read outside itself. Set {control.GRANT_VARIABLE}="
                        "external to let it consult the world.",
                        evidence={
                            "candidate_count": 0,
                            "external_configured": True,
                            "external": "refused, not granted",
                        },
                    )
                return Decision(
                    CONSULT,
                    "the corpus has nothing left to learn from and an external "
                    "source is configured, so the next material is fetched from "
                    "outside rather than waited for",
                    evidence={
                        "candidate_count": 0,
                        "consecutive_consults": consults,
                        "consult_limit": consult_limit,
                        "query_terms": list(observation.get("query_terms", [])),
                    },
                )
        # A single empty pass is a pause, not a conclusion. REST is the honest
        # answer: the loop is not concluding anything, it is waiting, and it
        # records that it is doing so.
        return Decision(
            REST,
            "there is nothing to learn: no curated document is available and no "
            "external source is configured, so the loop rests and looks again "
            "rather than concluding",
            evidence={
                "candidate_count": 0,
                "external_configured": observation.get("external_configured", False),
                "idle_passes": idle_passes,
            },
        )

    if growth_state and growth_state.get("approved"):
        if growth_state.get("reviewed_at") is None:
            return Decision(
                LEARN,
                "a growth is approved and has not been measured yet, so the next "
                "cycles are what will say whether it helped",
                evidence={"growth": "approved, unmeasured"},
            )
        if growth_state.get("kept") is False:
            return Decision(
                LEARN,
                "the last growth did not help, so more cycles on the current "
                "shape are the honest response rather than another change",
                evidence={"growth": "judged, did not help"},
            )
        if growth_state.get("kept") is True:
            return Decision(
                LEARN,
                "a growth was approved, measured and kept, so the model is now the "
                "shape that earned it and learning continues on it",
                evidence={"growth": "approved, judged, kept"},
            )
        # Approved, measured, and no verdict either way: the loss at approval was
        # not comparable to the loss now, so the growth was never really judged.
        # This falls through to the capacity branch below, which chooses EXPAND
        # again on a proposal that approve_growth has already consumed. It printed
        # "the model is grown" on every pass and grew nothing, because
        # approve_growth returns None for a proposal that is not waiting and the
        # act() branch reported that as a success either way.
        return Decision(
            LEARN,
            "a growth is already in the model and was never comparable to the loss "
            "at approval, so nothing is left to approve; learning continues on the "
            "current shape",
            evidence={"growth": "approved, unjudged, nothing pending"},
        )

    if observation["capacity_shortages"] > 0:
        if "expansion" not in granted:
            # Refusal is a normal outcome of this loop, not a failure of it. The
            # decision is to keep learning on the model as it stands, and the
            # reason says the growth is waiting on a human so the person reading
            # the log knows there is something for them to do.
            return Decision(
                LEARN,
                "a source is short of model capacity and a growth is waiting for "
                f"a human; learning continues on the current shape because the "
                f"expansion domain is not granted. Set "
                f"{control.GRANT_VARIABLE}=expansion to allow it.",
                evidence={
                    "capacity_shortages": observation["capacity_shortages"],
                    "expansion": "refused, not granted",
                },
            )
        return Decision(
            EXPAND,
            "a source keeps defeating a bounded update for want of model "
            "capacity, which more material cannot answer, so the model is grown",
            evidence={"capacity_shortages": observation["capacity_shortages"]},
        )

    if observation["cycles_recorded"] and not observation["improving"]:
        return Decision(
            REVIEW,
            "recent cycles are not reducing holdout loss, so the recorded "
            "decisions are worth reading before more are made",
            evidence={
                "mean_improvement": observation["mean_improvement"],
                "cycles_measured": observation["cycles_recorded"],
            },
        )

    return Decision(
        LEARN,
        "there is material to learn and nothing wrong with the recent record, "
        "so the next source is chosen from the need-scored queue",
        evidence={
            "candidates": observation["candidate_count"],
            "mean_improvement": observation["mean_improvement"],
        },
    )


def act(c, decision, tel=None, holdout=None, runner=None, consult_limit_results=5):
    """Carry out a decision, and report what actually happened.

    Expansion is where autonomy meets Architecture 38, and the meeting is
    visible here rather than described. This calls the same expansion.approve
    that a person would call from the command line, so the gate applies to the
    agent exactly as it applies to the operator. A denial is caught and returned
    as a fact, not raised: an agent that crashes because a human has not granted
    a permission is an agent that stops, and stopping is not what a denial is
    for.

    Learning and reviewing are where this was a decision that decided nothing.
    LEARN and REVIEW both fell through to `return {"applied": True}`, which
    records that something happened without doing it, and the worker called
    run_worker afterwards regardless of what came back. The loop therefore
    learned the same way whichever action it picked, which is the difference
    between a decision layer and a log line.

    The runner is the callable that does the work, passed in rather than imported
    so that this module does not depend on the learning cycle and a decision can
    be exercised without running one. When no runner is supplied the action is
    reported as chosen but not performed, because claiming an action was applied
    when nothing applied it is the specific dishonesty this project exists to
    avoid.
    """
    import expansion as expansion_module

    if decision.action == EXPAND:
        try:
            approved = expansion_module.approve_growth(c, holdout_loss=holdout)
        except control.ControlDenied as refusal:
            if tel is not None:
                tel.write("autonomy_expansion_denied", reason=str(refusal))
            return {
                "action": EXPAND,
                "applied": False,
                "denied": True,
                "reason": str(refusal),
            }
        if approved is None:
            # Nothing was waiting to be approved. approve_growth returns None for
            # a proposal that is absent or already applied, and this branch used
            # to report that as `applied: False, denied: False`, which reads as a
            # success. The loop then printed "the model is grown" on every pass
            # and the model never changed.
            #
            # It is reported as what it is: the action was chosen and there was
            # nothing to do, which is a different thing from having grown the model
            # and a different thing from having been refused.
            if tel is not None:
                tel.write(
                    "autonomy_expansion_nothing_pending",
                    reason=(
                        "EXPAND was chosen but no growth was waiting to be approved"
                    ),
                )
            return {
                "action": EXPAND,
                "applied": False,
                "denied": False,
                "nothing_pending": True,
                "proposal": None,
                "reason": (
                    "EXPAND was chosen but there was no growth waiting to be "
                    "approved, so nothing was grown"
                ),
            }
        return {
            "action": EXPAND,
            "applied": True,
            "denied": False,
            "proposal": approved,
        }

    if decision.action == REST:
        seconds = float(getattr(c, "autonomy_idle_seconds", DEFAULT_IDLE_SECONDS))
        if tel is not None:
            tel.write("autonomy_rest", reason=decision.reason, sleep_seconds=seconds)
        return {
            "action": REST,
            "applied": True,
            "performed": False,
            "rest": True,
            "sleep_seconds": seconds,
            "reason": decision.reason,
        }

    if decision.action == CONSULT:
        # This is the first thing in the project that reaches outside the process
        # on Jupiter's own initiative. Three things are kept separate here and it
        # matters that they are: fetching is external access and carries the
        # Architecture 38 gate, judging is Architecture 28's retention test, and
        # retaining files into the raw corpus rather than the knowledge store is
        # what keeps a retrieved document from becoming knowledge before any
        # cycle has ruled on it.
        import external as external_module

        query = " ".join(decision.evidence.get("query_terms", [])).strip()
        if not query:
            return {
                "action": CONSULT,
                "applied": False,
                "performed": False,
                "consulted": False,
                "reason": "there was nothing to ask the world about",
            }
        try:
            results = external_module.fetch(c, query, limit=consult_limit_results)
        except control.ControlDenied as refusal:
            if tel is not None:
                tel.write("autonomy_consult_denied", reason=str(refusal))
            return {
                "action": CONSULT,
                "applied": False,
                "performed": False,
                "consulted": False,
                "denied": True,
                "reason": str(refusal),
            }
        if not results:
            return {
                "action": CONSULT,
                "applied": True,
                "performed": True,
                "consulted": True,
                "retained": 0,
                "results": 0,
                "query": query,
                "reason": "the external sources were asked and had nothing for it",
            }

        import concepts as concepts_module

        # Architecture 28's retention test needs what Jupiter already holds, and
        # it is read through the Store rather than by parsing the file: the
        # collections are the store's own shape, and a reader written against a
        # guess at that shape is a reader that breaks the day the store changes.
        knowledge = store_module.Store(c.knowledge_store)
        knowledge_terms = set()
        held = set()
        for record in knowledge.all(concepts_module.CONCEPTS):
            if not isinstance(record, dict):
                continue
            for field in ("identifier", "description"):
                value = record.get(field)
                if isinstance(value, str):
                    knowledge_terms |= external_module.terms(value)
            for origin in record.get("sources", []) or []:
                if isinstance(origin, str):
                    held.add(origin)

        kept, refusals = [], []
        for result in results:
            keep, reason = external_module.worth_keeping(result, knowledge_terms, held)
            if not keep:
                refusals.append(
                    {"reference": result.get("reference"), "reason": reason}
                )
                continue
            target = external_module.retain(knowledge, result, reason)
            kept.append(str(target))
        if tel is not None:
            tel.write(
                "autonomy_consulted",
                query=query,
                results=len(results),
                retained=len(kept),
                refused=len(refusals),
            )
        return {
            "action": CONSULT,
            "applied": True,
            "performed": True,
            "consulted": True,
            "query": query,
            "results": len(results),
            "retained": len(kept),
            "kept": kept,
            "refused": refusals,
            "reason": (
                f"{len(kept)} of {len(results)} results were filed into the raw "
                "corpus for curation and the acceptance gate, and the rest were "
                "refused on retention grounds"
            ),
        }

    if decision.action in (LEARN, REVIEW):
        if runner is None:
            # Reported, not raised. This contradicts what the code did and what
            # its own docstring specifies: both said that a missing runner is
            # reported as chosen but not performed, and the implementation
            # raised instead. Raising is the wrong behaviour here for a reason
            # beyond consistency: step() calls this, and a supervisor that dies
            # because nothing attached a callback cannot record the pass that
            # caused it. Four tests encoded the documented behaviour throughout.
            if tel is not None:
                tel.write(
                    "autonomy_not_performed",
                    action=decision.action,
                    reason="no runner was supplied, so no learning cycle ran",
                )
            return {
                "action": decision.action,
                "applied": False,
                "performed": False,
                "note": (
                    f"{decision.action} was chosen but no runner was supplied, so "
                    "no learning cycle ran"
                ),
                "reason": "no runner was supplied, so nothing was performed",
            }
        cycles = (
            1 if decision.action == REVIEW else int(getattr(c, "cycles_per_pass", 4))
        )
        outcome = runner(cycles)
        if tel is not None:
            tel.write(
                "autonomy_performed",
                action=decision.action,
                cycles=cycles,
                reason=decision.reason,
            )
        return {
            "action": decision.action,
            "applied": True,
            "performed": True,
            "cycles": cycles,
            "outcome": outcome,
            "reason": decision.reason,
        }

    # REST and STOP are reachable now, so this is the branch they take. REST
    # waits and keeps going; STOP ends the run. Neither runs a learning cycle.
    if decision.action == STOP:
        if tel is not None:
            tel.write("autonomy_stop", reason=decision.reason)
        return {
            "action": STOP,
            "applied": True,
            "performed": False,
            "stop": True,
            "reason": decision.reason,
        }

    if tel is not None:
        tel.write("autonomy_rest", reason=decision.reason)
    return {
        "action": REST,
        "applied": True,
        "performed": False,
        "rest": True,
        "reason": decision.reason,
    }


def step(c, candidates=None, tel=None, holdout=None, runner=None):
    """One pass of the loop: observe, decide, act, record.

    The whole decision cycle in one call so a caller cannot run observe and
    forget to decide, which is how a supervisor stops being one.
    """
    import expansion as expansion_module

    observation = observe(c, candidates=candidates)
    growth_state = expansion_module.load(c).get("capacity_growth")
    decision = decide(c, observation, growth_state=growth_state)
    outcome = act(c, decision, tel=tel, holdout=holdout, runner=runner)

    state = load(c)
    state["passes"] = int(state.get("passes", 0)) + 1
    state["last_decision"] = decision.as_dict()
    state["last_observation"] = {
        key: observation[key]
        for key in (
            "candidate_count",
            "capacity_shortages",
            "high_frustration_count",
            "generation_needs",
            "cycles_recorded",
            "mean_improvement",
        )
    }
    # Reset by the decision, not by whether a runner was attached. The counter
    # answers "how long has nothing been learnable", and a pass that chose to
    # learn or to review has answered that question: something was found. A
    # pass that rested found nothing and counts.
    #
    # EXPAND used to count as idle, and that was wrong twice over. An expansion
    # pass is reached only when candidates were found, so counting it as a pass
    # where nothing was learnable is false; and the reason it could not act was
    # the control gate refusing, which is a decision waiting on a person rather
    # than an empty corpus. The visible consequence was that a run whose only
    # problem was a missing grant climbed to the idle limit and reported "nothing
    # has been learnable for 22 consecutive passes, add material to the corpus"
    # while the corpus was full and the real answer was one keystroke away.
    if decision.action in (LEARN, REVIEW, EXPAND):
        state["idle_passes"] = 0
    else:
        state["idle_passes"] = int(state.get("idle_passes", 0)) + 1

    # The consult counter is its own sequence rather than a function of the idle
    # counter, because the two answer different questions. The idle counter asks
    # whether anything was learnable; a consult that fetched three documents and
    # filed them for curation made progress and is not idle, but neither is it
    # evidence that Jupiter can answer without asking the world again. Counting
    # only the passes that were actually refused or came back empty is what makes
    # the bound mean "three fruitless attempts" rather than "three passes".
    if decision.action == CONSULT and outcome.get("retained"):
        state["consecutive_consults"] = 0
    else:
        state["consecutive_consults"] = (
            int(state.get("consecutive_consults", 0) or 0) + 1
            if decision.action == CONSULT
            else 0
        )
    state["history"].append(
        {
            "pass": state["passes"],
            "action": decision.action,
            "reason": decision.reason,
            "at": decision.at,
        }
    )
    save(c, state)

    # Inhabited: the loop writes its own state to the world continuously when
    # the feature is enabled and the external domain is granted. That is the
    # smallest real form of being inhabited rather than consulted: the loop
    # acts on the world (writes to it) continuously, not only when it asks a
    # question. It writes to the configured respond path, which defaults to
    # out/external/autonomy_status.json so it stays inside the data root and
    # does not need a separate gate beyond the external domain.
    if getattr(c, "autonomy_continuous_respond", False):
        granted = control.granted_domains()
        if "external" in granted:
            import json as json_mod
            import time

            respond_path = Path(
                getattr(c, "autonomy_respond_path", "out/external/autonomy_status.json")
            )
            respond_path.parent.mkdir(parents=True, exist_ok=True)
            respond_path.write_text(
                json_mod.dumps(
                    {
                        "at": time.time(),
                        "passes": state.get("passes", 0),
                        "last_action": decision.action,
                        "last_reason": decision.reason,
                        "idle_passes": state.get("idle_passes", 0),
                        "consecutive_consults": state.get("consecutive_consults", 0),
                        "granted_domains": sorted(granted),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    return decision, outcome, observation


def main(c=None, argv=None):
    """Report what the loop has been deciding, for a person asking.

    Read-only, so it is not gated: nothing is changed by looking. It exists
    because an agent that acts on its own has to be readable from outside, and
    the honest answer to "what is it doing and why" should not require opening
    the state file in an editor.
    """
    import argparse

    from config import load_config

    parser = argparse.ArgumentParser(
        prog="autonomy",
        description=(
            "Report the continuous loop's decisions, its current state, and "
            "what it is waiting on."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="how many recent decisions to print (default 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the summary as JSON rather than as text",
    )
    args = parser.parse_args(argv)

    c = c or load_config()
    report = summary(c)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"continuous state: {report['state']}")
    print(f"passes:          {report['passes']}")
    print(f"idle passes:     {report['idle_passes']}")
    print(f"granted domains: {', '.join(report['granted_domains']) or 'none'}")
    print()
    print("decisions taken:")
    for action, count in sorted(report["actions_taken"].items()):
        print(f"  {action:<10} {count}")
    print()
    print(f"most recent {min(args.limit, len(report['recent']))} decisions:")
    for entry in report["recent"][-args.limit :]:
        print(
            f"  pass {entry.get('pass', '?'):<5} {entry.get('action', '?'):<8} {entry.get('reason', '')}"
        )
    if not report["recent"]:
        print("  the loop has not run yet")
    print()
    print(report["note"])
    return 0


def summary(c):
    """What the loop has been doing, for a person reading the state file."""
    state = load(c)
    history = state.get("history", [])
    recent = history[-10:]
    actions = {}
    for entry in history:
        actions[entry.get("action", "?")] = actions.get(entry.get("action", "?"), 0) + 1
    return {
        "state": str(state_path(c)),
        "passes": state.get("passes", 0),
        "idle_passes": state.get("idle_passes", 0),
        "actions_taken": actions,
        "last_decision": state.get("last_decision"),
        "recent": recent,
        "grant_variable": control.GRANT_VARIABLE,
        "granted_domains": sorted(control.granted_domains()),
        "note": (
            "one entry per pass of the loop. The reason is written for a "
            "person: an agent that acts without recording why cannot be "
            "audited, and Architecture 38 asks for human control over what it "
            "changes."
        ),
    }


if __name__ == "__main__":
    import json

    raise SystemExit(main())
