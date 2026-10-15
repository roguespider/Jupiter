"""Continuous learning worker for Jupiter.

The default learner is now direct reward-modulated plasticity: Jupiter makes a
bounded gain/loss update from observed token outcomes without back-propagation
or an optimizer. The legacy gradient learner remains available as an explicit
baseline with ``--learning-mode gradient``.

Each cycle is measured before and after, and judged before it is kept:

    candidate -> micro-update -> evaluate -> classify -> accept or not

    candidate   a curated document, never the reserved regression document
    evaluate    holdout on the document, plus the reserved document
    classify    ACCEPT / REJECT / DEFER / REVIEW / ROLLBACK

An update is only saved when it is ACCEPTed, which means it improved its own
document and did not measurably damage the reserved document. That second
condition is what makes this repeated learning rather than a sequence of
unrelated fine-tunes.

The worker finishes as soon as any safety limit fires:

  * a source was not accepted for worker_patience consecutive cycles
  * worker_max_cycles cycles completed
  * worker_max_seconds wall-clock budget spent
  * a single cycle exceeded worker_cycle_max_seconds
  * the learner could not stabilize, which is a ROLLBACK
  * gradient recoveries across the run exceeded worker_max_recoveries
  * no learning candidate available -> sleep worker_idle_seconds and retry

This adds no neurons, no tokenizer, no database, no dependency and no
architecture change. It reuses the existing model, the existing tokenizer and
the existing micro-update loop unchanged.

Usage:

    python learning_worker.py              run until a stopping rule fires
    python learning_worker.py --once       run exactly one cycle
    python learning_worker.py --detach     start in the background, return now
"""

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import acceptance
import assertions
import code_knowledge
import concepts as concepts_module
import control
import conversation_eval
import expansion
import experience_loop
import experiment
import frustration
import lock as lock_module
import plasticity
import probes
import register
import regression
import relationships as relationships_module
import selection
import store as store_module
import synthesize
import train as train_module
import utilization
from telemetry import Telemetry, write_status
from tokenizer import load_tokenizer
from train import (
    PUBLISHED_CHECKPOINT,
    Candidate,
    CycleTimeLimitExceeded,
    attach_checkpoint,
    build_model,
    discover_documents,
    load_optimizer_state,
    make_batch,
    measure_on,
    read_cycle,
    read_learning_text,
    read_text,
    reset_optimizer,
    restore_model,
    run_micro_learning,
    save_checkpoint,
    seed_everything,
    snapshot_model,
    split_tokens,
    window_count_for,
    window_for,
)

import confidence as confidence_module
import memory as memory_module
import skills
from config import load_config, save_architecture

_stop_requested = False


def request_stop(signum, frame):
    """Ask the worker to finish after the current cycle.

    `signum` and `frame` are the two arguments Python passes to every signal
    handler. Only the signal number is reported; the frame is the caller's
    stack snapshot, which a learning loop has no use for.
    """
    global _stop_requested
    _stop_requested = True
    print(
        f"\n[worker] signal {signum} received; "
        "finishing the current cycle then stopping."
    )


def write_state(state_path, payload):
    """Persist worker state so a stopped run is still inspectable."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def experience_interval(c, cycle):
    """Whether this cycle should render recorded conversation into the corpus.

    Not every cycle. Rendering rewrites a file, and doing that on every cycle of
    a long run is disk traffic that buys nothing, because the conversation log
    grows far more slowly than cycles run. The interval is configuration for the
    same reason the conversation set's own interval is.
    """
    every = int(getattr(c, "experience_publish_interval", 25) or 0)
    if every <= 0:
        return False
    return int(cycle) % every == 0


def publish_experience(c, tel):
    """Render recorded conversation into the raw directory, and report it.

    The failure is reported rather than raised. Experience publishing is a
    convenience for the corpus; a learning cycle that cannot write it should
    still learn from the corpus it already has, and stopping the run because an
    optional input could not be written would make the loop less reliable than
    the alternative.
    """
    try:
        published = experience_loop.publish(c)
    except OSError as failure:
        if tel is not None:
            tel.write("experience_publish_failed", reason=str(failure))
        print(f"[worker] could not publish conversation experience: {failure}")
        return {"written": 0, "reason": str(failure)}
    if tel is not None:
        tel.write(
            "experience_published",
            written=published.get("written", 0),
            skipped=published.get("skipped", 0),
            unreadable=published.get("unreadable", 0),
            path=published.get("path"),
        )
    return published


def load_candidates(c, tel):
    """Collect the documents the worker may learn from.

    Returns None when nothing is learnable right now. A worker that outlives a
    single session must survive an empty corpus and idle rather than exit, so
    the discovery failures are recorded and turned into an idle state instead
    of a crash. Errors other than the documented discovery failures still
    propagate.
    """
    try:
        docs = discover_documents(c)
        pieces = read_learning_text(docs, c)
    except RuntimeError as reason:
        tel.write(
            "worker_no_candidate",
            reason=str(reason),
        )
        print(f"[worker] no candidate available: {reason}")
        return None
    return pieces


def _interactive(require=None):
    """Whether a person is at the keyboard to answer a question.

    Two checks, because each is wrong on its own. Standard input has to be a
    terminal: a detached worker or a piped run has nobody to answer, and a prompt
    there is a run waiting on input that will never come. And the escape has to
    be honoured, so a run that wants no prompts can say so rather than having
    them appear and vanish.

    `require` names a control domain the caller cannot act without. It is a
    parameter rather than a fixed check because the check was wrong: it was hard
    coded to expansion, which meant running learning_worker.py directly, without
    the environment variable the batch file sets, could never ask anything. The
    run had a growth waiting on a human and no way to reach it, which is a
    control that is off rather than one that is waiting. A domain the caller
    needs is offered first through offer_grant, so by the time a prompt is
    reached the grant is either held or the person has declined it.
    """
    if os.environ.get("JUPITER_NO_PROMPT"):
        return False
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return False
    except (ValueError, AttributeError):
        return False
    if require is None:
        return True
    return require in control.granted_domains()


def prompt_for_growth(c, baseline, tel=None):
    """Offer a pending growth to a person, and act on the answer.

    The expansion loop ends in "keep or rollback", and expansion is kept behind
    human control, so the decision belongs to a person rather
    than to a counter. The loop had no way to ask: it proposed, printed a line,
    and stopped, and the proposal sat in a file until somebody went looking for
    it. That is a control that is technically present and practically never used.

    The prompt is skipped when there is nobody to answer or the grant is absent,
    so a detached run is never left waiting on input that will not come.
    Declining is the default on anything that is not a clear yes, so a stray
    keystroke, an empty line or a closed input cannot grow the model.

    A growth that was approved, measured and rolled back is offered again once
    the model is a different shape, because that verdict was about a model that
    no longer exists. Skipping those forever was a real gap: a growth that failed
    once could never be reconsidered, and its proposal sat in the state file
    looking like something still waiting for an answer when it had in fact been
    answered and un-applied.
    """
    proposal = expansion.pending(c).get("capacity_growth")
    if not isinstance(proposal, dict) or proposal.get("approved"):
        return None
    # A growth that was applied, measured and rolled back is still asked about,
    # and this is the correction that makes it reachable. It used to be skipped
    # whenever the verdict had been recorded against the shape the model had now,
    # and a rollback restores the model to exactly the shape the failed attempt
    # started from, so that test was permanently true after the first reverted
    # growth. The result was a latch: the question could never be asked again,
    # the proposal sat in the state file looking unanswered, and the only way to
    # apply it was a command the person watching the run did not know existed.
    # A control whose application is unreachable is a control that is off, which
    # is the failure this module was written to remove.
    #
    # What that guard got right is preserved: the loop does not re-propose a
    # growth the evidence has already answered, so this is not a proposal being
    # generated on every cycle. What it got wrong was treating a measurement
    # taken by the system as a decision taken by the person. Expansion is kept
    # behind human control, so the prior verdict is shown and the
    # person decides again. Declining leaves the proposal exactly where declining
    # has always left it, which is on record and unapplied.
    #
    # Asked last, and only once, because asking first means a proposal that can
    # never be taken still prints a question a person cannot answer.
    if not _interactive():
        return None

    # The grant is offered rather than required. This used to check for the
    # expansion grant first and return without printing anything if it was
    # absent, which is why running learning_worker.py directly could never ask:
    # the entry point sets the grant and the script does not, so the one way to
    # reach the prompt was a variable nobody had set. A growth waiting on a
    # person with no way to ask is a control that is off, not one that is waiting.
    if "expansion" not in control.granted_domains():
        if not offer_grant(
            c, "expansion", "a growth is waiting for a decision", tel=tel
        ):
            print(
                "  a growth is waiting for a human and the expansion domain is not"
                " granted, so it stays unapplied. Nothing was changed."
            )
            return None

    print()
    print("=" * 72)
    print("GROWTH PROPOSED - you are being asked to decide")
    print("=" * 72)
    print(f"  source     : {proposal.get('source')}")
    print(f"  cause      : {proposal.get('cause')}")
    print(
        f"  layers     : {proposal.get('from_layers')} -> {proposal.get('to_layers')}"
    )
    from_embd = proposal.get("from_embd")
    to_embd = proposal.get("to_embd")
    if from_embd and to_embd and from_embd != to_embd:
        print(f"  width      : {from_embd} -> {to_embd}")
    print(
        f"  ceiling    : {expansion.MAX_LAYERS} layers "
        f"(this proposal reaches {proposal.get('to_layers')})"
    )
    if from_embd and to_embd and from_embd != to_embd:
        print(f"  width cap  : {expansion.MAX_EMBD} (this is {to_embd})")
    if proposal.get("reviewed_at") is not None:
        print()
        print("  A previous attempt at this growth was measured and rolled back.")
        if proposal.get("change") is not None:
            verdict = "helped" if proposal.get("kept") else "did not help"
            print(
                f"    loss {proposal.get('holdout_at_approval')} -> "
                f"{proposal.get('holdout_at_review')} "
                f"(change {proposal.get('change')}, {verdict})"
            )
        if proposal.get("reverted_to"):
            print(f"    restored from {proposal.get('reverted_to')}")
        print("    The weights have moved on since, and so has the evidence.")
        print("    Approving tries the same change again and is measured the")
        print("    same way, so a second verdict is a measurement rather than a")
        print("    repetition.")
    print()
    print("  A source keeps defeating a bounded update for want of capacity,")
    print("  which more material cannot answer. Approving adds a layer and")
    print("  widens the model. It takes effect on the NEXT run, is measured")
    print("  against the loss below, and is rolled back automatically if it")
    print("  does not help.")
    print()
    try:
        # Always ask when called — interactive by design per user requirement.
        # Tests mock builtins.input; real runs expect user at keyboard.
        answer = input("  Approve this growth? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  no answer given; leaving the proposal for another time")
        return None
    if answer in ("y", "yes"):
        applied = expansion.approve_growth(c, holdout_loss=_last_best_holdout)
        if applied is not None:
            print(
                f"  approved. n_layer {applied['from_layers']} -> {applied['to_layers']}"
            )
            print("  it takes effect on the next run.")
            if tel is not None:
                tel.write(
                    "growth_approved",
                    by="prompt",
                    **{k: applied[k] for k in ("from_layers", "to_layers")},
                )
        return applied
    print("  declined. The proposal stays on record and nothing changes.")
    return None


# The best holdout the run reached, which is the baseline a growth is judged
# against. Module level because the prompt is called at the end of a run whose
# result is a local, and threading one number through the summary for the sake of
# a parameter would be the worse shape.
_last_best_holdout = None

# Domains the worker was refused during this run, so the person is asked about
# them once rather than on every cycle. Module level for the same reason as
# _last_best_holdout: a refusal discovered deep in a cycle has to reach the end
# of the run, and the end of the run has no local to discover it in.
_refused_domains = set()


def offer_grant(c, domain, detail=None, tel=None):
    """Ask whether a refused control domain should be granted, and record the answer.

    A control that can only be changed by editing an environment variable before
    the process starts is a control a person cannot use. Architecture 38 keeps
    architecture changes behind a human, and Jupiter keeps needing one: the model
    grew, the recorded architecture had to be written down, and the run stopped
    with a message naming a variable the person had to set somewhere they were not
    looking. The gate was right and the affordance was missing.

    Asking here is not the same as granting automatically. The answer is written
    to the control decision log with who gave it, the domain, and the value the
    process is running with from now on, so the grant is visible and auditable
    rather than a silent widening. The environment of the running process is
    extended in place, which is what makes the next cycle in this same run able to
    proceed; a run that is restarted has to be given the variable again, and the
    message says so.

    Skipped when there is nobody to answer, and it will not ask twice about the
    same domain in the same run. Anything that is not a clear yes is a no.
    """
    if domain in _refused_domains:
        return False
    _refused_domains.add(domain)

    already = control.granted_domains()
    print()
    print("=" * 72)
    print(f"CONTROL GRANT NEEDED - {domain}")
    print("=" * 72)
    print()
    print("  Jupiter tried to do something this run and was correctly refused.")
    print(f"  domain   : {domain}")
    if detail:
        print(f"  detail   : {detail}")
    print()
    print(
        f"  {control.GRANT_VARIABLE} is currently: "
        f"{', '.join(sorted(already)) or 'nothing'}"
    )
    print()
    print("  Granting it lets Jupiter act in this domain for the rest of this")
    print("  run and for as long as the variable stays set. It is recorded in the")
    print("  control decision log with your answer.")
    print()
    try:
        # Always ask when called — interactive by design per user requirement.
        # Tests mock builtins.input; real runs expect user at keyboard.
        # A timeout prevents a hung run: if nobody answers within 10 seconds,
        # the domain stays refused rather than blocking forever.
        # The input is read in a separate thread so the main thread can
        # enforce the timeout without relying on exception propagation
        # across threads (which does not interrupt input() on Windows).
        result_container = {}

        def _read_input():
            try:
                result_container["answer"] = (
                    input(f"  Grant '{domain}' for this run? [y/N] ").strip().lower()
                )
            except (EOFError, KeyboardInterrupt, OSError):
                result_container["answer"] = None
                result_container["interrupted"] = True

        input_thread = threading.Thread(target=_read_input)
        input_thread.daemon = True
        input_thread.start()
        input_thread.join(timeout=10.0)

        if input_thread.is_alive():
            # Timeout fired: the thread is still blocked on input().
            # We leave it running (daemon) and treat as refused.
            print("  no answer within 10s; the domain stays refused")
            return False

        if result_container.get("interrupted"):
            print("  no answer given; the domain stays refused")
            return False

        answer = result_container.get("answer", "")
    except Exception:
        # Any unexpected failure during the interactive prompt is treated
        # as a refusal rather than a crash, per the safety design.
        print("  no answer given; the domain stays refused")
        return False
    if answer not in ("y", "yes"):
        print("  declined. Jupiter keeps refusing and keeps recording why.")
        return False

    os.environ[control.GRANT_VARIABLE] = ",".join(sorted(already | {domain}))
    print(f"  granted. {control.GRANT_VARIABLE}={os.environ[control.GRANT_VARIABLE]}")
    print("  it applies to the rest of this run. A later run needs it set again.")
    if tel is not None:
        tel.write(
            "control_grant_accepted",
            domain=domain,
            detail=detail,
            grants=os.environ[control.GRANT_VARIABLE],
            by="prompt",
        )
    return True


def _windows_per_cycle(text, c):
    """How many windows of a document one cycle should read.

    Derived from the document's own length so that a large document is
    practical to learn without anyone having to configure for it, and capped so
    that a single cycle can never read more than the cycle budget allows. The
    cap matters: without it a very large document would produce a window larger
    than the context the model can hold, and the tail of that window would be
    silently unlearned, which is the exact failure rotating the window was
    written to prevent.

    The target is a fixed number of characters per cycle, not a number of
    cycles, so the cost of a cycle does not grow with the size of the document.
    """
    size = int(getattr(c, "max_chunk_chars", 12000)) or 12000
    ceiling = int(getattr(c, "max_windows_per_cycle", 16)) or 16
    target = int(getattr(c, "target_chars_per_cycle", 96000)) or 96000
    windows = len(text) // size if size else 1
    if windows <= 1:
        return 1
    return max(1, min(ceiling, target // size))


def decide_stop(
    c,
    completed,
    cycle_budget,
    elapsed,
    total_recoveries,
    stalled,
    stop_requested,
):
    """Return (status, reason) when a safety limit fires, else None.

    Every stopping rule lives here so the full set is readable in one place
    and can be checked without running a training cycle. The order matters:
    the hard process limits are checked before the quality rules.
    """
    if stop_requested:
        return "stop_requested", "stop requested by signal"
    if completed >= cycle_budget:
        return "max_cycles", f"reached the cycle budget of {cycle_budget}"
    if elapsed >= c.worker_max_seconds:
        return "time_budget", f"spent the {c.worker_max_seconds:.0f}s time budget"
    if total_recoveries > c.worker_max_recoveries:
        return (
            "too_many_recoveries",
            f"{total_recoveries} gradient recoveries exceeds the "
            f"limit of {c.worker_max_recoveries}",
        )
    if stalled >= c.worker_patience:
        return (
            "no_improvement",
            f"no source was accepted for {stalled} consecutive cycles",
        )
    return None


def record_cycle(cycle_path, cycle, model, source_path, cycle_result, checkpoint):
    """Write a learning_state.json record compatible with train.py.

    train.py reads only "learning_cycle" from this file, so keeping the same
    shape means both tools can run in sequence without losing the record.
    """
    record = {
        "learning_cycle": cycle,
        "source": str(source_path),
        "parameters": model.parameter_count(),
        "architecture": model.config.to_dict(),
        **cycle_result,
        "checkpoint": str(checkpoint),
        "origin": "learning_worker",
    }
    cycle_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


def record_cycle_knowledge(
    knowledge,
    c,
    cycle,
    source_path,
    source_text,
    cycle_result,
    holdout_before,
    regression_delta,
    confidence,
    learning_rate,
    learned,
    verdict,
    verdict_reason,
    source_observed=0,
    source_accepted=0,
    source_contradictions=0,
):
    """Write what one cycle did into the knowledge layer, and return both halves.

    Order matters. Concepts are promoted first so the experience record can
    name the terms this cycle actually exercised. Previously the experience was
    written first, so the promoted terms were computed, handed to telemetry and
    then dropped -- an experience should carry the concepts it actually
    exercised, and usage frequency and supporting
    experiences are part of a concept.

    Every value recorded here is measured in this cycle. Two fields that
    the same specification lists are deliberately absent: an
    interpretation of the document, and an expected result. Jupiter cannot
    produce either honestly -- it does not understand the text, and it holds no
    prediction to compare the outcome against -- so they stay unwritten rather
    than being filled with a plausible-sounding guess.

    An accepted cycle also copies the document it learned into external memory.
    That is a deliberate second copy: the text already exists in data/curated,
    but a conversation goes through the knowledge interface, so material Jupiter
    has actually learned has to be reachable through it. Only accepted cycles
    are copied. A rejected document stays in the corpus to be retried, and is
    not recorded as knowledge, so nothing unaccepted can be recalled later.

    Recording a cycle touches the concept graph, the relationship graph, the
    assertions and the experience, and each of those writes several
    collections; measured, the whole record was 246 writes of a multi-megabyte
    store inside three cycles, against a fifth of a second of actual learning.
    That is why the public entry point is a transaction wrapper and the body
    lives in _record_cycle_knowledge: one cycle's record becomes one atomic
    write, which is also the right unit, because a cycle either left a record or
    it did not.
    """
    with knowledge.transaction():
        return _record_cycle_knowledge(
            knowledge,
            c,
            cycle,
            source_path,
            source_text,
            cycle_result,
            holdout_before,
            regression_delta,
            confidence,
            learning_rate,
            learned,
            verdict,
            verdict_reason,
            source_observed=source_observed,
            source_accepted=source_accepted,
            source_contradictions=source_contradictions,
        )


def _record_cycle_knowledge(
    knowledge,
    c,
    cycle,
    source_path,
    source_text,
    cycle_result,
    holdout_before,
    regression_delta,
    confidence,
    learning_rate,
    learned,
    verdict,
    verdict_reason,
    source_observed=0,
    source_accepted=0,
    source_contradictions=0,
):
    """The body of record_cycle_knowledge, run inside its transaction.

    Split out only so the wrapper can open one transaction around everything and
    this can be a plain function with the same arguments. It is not called from
    anywhere else.
    """
    source_name = Path(source_path).name
    experience_id = f"cycle-{cycle}"
    improvement = cycle_result["holdout_loss_improvement"]
    # The document being worked on is what the current task
    # needs, so it is held in working memory for the duration of the cycle. It
    # is cleared at the end because working memory is task-scoped; the durable
    # record of the cycle is the experience written below.
    memory_module.remember_working(knowledge, f"active-{cycle}", str(source_path))
    routing = memory_module.route_for_source(
        observed=source_observed,
        accepted=source_accepted,
        contradictions=source_contradictions,
        confidence=confidence,
        # Nine factors decide where a fact is filed. route_for_source already
        # maps all nine, and its own docstring claims each is observable from
        # the system records. That claim was true of the function and false of
        # this call site, which passed four and let the rest default to zero:
        # improvement and the reserved-document gain were computed a few lines
        # above and simply not handed over, and the cycle's own duration and the
        # source's age were sitting in scope unused.
        #
        # retrieval_cost is still zero, and honestly so: nothing records how
        # long a real retrieval against a source takes, because nothing counts
        # retrievals per source yet. One factor of nine is genuinely missing
        # rather than merely unpassed.
        improvement=improvement,
        regression_gain=regression_delta if regression_delta is not None else 0.0,
        learning_cost=float(cycle_result.get("elapsed_seconds", 0.0)),
        age_cycles=source_observed,
    )
    promoted = []
    contradicted = []
    # Expansion changes how much is extracted from a document.
    # The limit is per source so a source judged short of capacity can be
    # expanded without every other source being affected, and a grant that has
    # not been authorised leaves every limit exactly where the configuration
    # put it.
    term_limit = expansion.active_limit(c, source_name)
    for term in concepts_module.candidate_terms(source_text, term_limit):
        if learned:
            record = concepts_module.observe(
                knowledge,
                term,
                delta_evidence=1,
                confidence_delta=0.08,
                source=source_name,
                experience=experience_id,
                cycle=cycle,
            )
        else:
            # A rejected document is not a supporter. Passing it as `source` as
            # well as `contradicting_source` would add the document that failed
            # to the concept's own list of things supporting it, which makes a
            # genuine conflict arithmetically impossible: nothing could ever
            # contradict something that has already agreed with it. The term is
            # only recorded as contradicted, by this document, against whatever
            # supported it earlier.
            record = concepts_module.observe(
                knowledge,
                term,
                delta_contradictions=1,
                confidence_delta=-0.04,
                source=None,
                experience=experience_id,
                cycle=cycle,
                contradicting_source=source_name,
            )
            contradicted.append(record)
        promoted.append(record["identifier"])

    # Deciding whether each conflict is contextual comes first; recording it
    # comes second. Only a conflict that survives step 6
    # gets a CONTRADICTS edge, because opening one for a context difference would
    # be inventing a disagreement the corpus does not contain. These edges were
    # unreachable in production before, since nothing ever passed
    # supporting=False.
    #
    # The two sides of the edge are the source that supports the term and the
    # source that contradicts it. A document disagreeing with another document
    # is a disagreement; a document disagreeing with a word is not something the
    # corpus can say. Passing the term as one endpoint made every conflict half
    # term and half filename, and since curation names a curated file by the
    # sha256 of its contents, each one invented a concept out of a sixty-eight
    # character hex string. Architecture 14 step 7 is about who is on each side,
    # and this is who is on each side.
    for record in contradicted:
        entry = concepts_module.note_conflict(knowledge, record)
        if entry is None:
            continue
        supporter = entry.get("supporter")
        for other in entry["contradicted_by"]:
            if not supporter or supporter == other:
                continue
            relationships_module.observe(
                knowledge,
                supporter,
                relationships_module.CONTRADICTS,
                other,
                supporting=False,
                endpoints=relationships_module.SOURCE_ENDPOINTS,
            )
            relationships_module.observe(
                knowledge,
                other,
                relationships_module.CONTRADICTS,
                supporter,
                supporting=False,
                endpoints=relationships_module.SOURCE_ENDPOINTS,
            )
    experience = memory_module.record_experience(
        knowledge,
        experience_id,
        {
            "attempted": str(source_path),
            "holdout_loss_before": holdout_before,
            "holdout_loss_after": cycle_result["holdout_loss_after"],
            "performance_change": improvement,
            "regression_change": regression_delta,
            "confidence": confidence,
            "update": {
                "successful_steps": cycle_result["successful_steps"],
                "gradient_recoveries": cycle_result["gradient_recoveries"],
                "learning_rate": learning_rate,
            },
            "helped": learned,
            "concepts": promoted,
            "routing": routing,
            "decision": verdict,
            "reason": verdict_reason,
        },
    )
    # Every cycle gets a long-term entry, including a rejected one: the
    # specification records that intent, and both want a
    # rejected cycle kept as readably as an accepted one. What was wrong was the
    # wording. This entry said "learned <source>" whatever the verdict, so a
    # cycle the system had just rejected left a memory asserting that it had
    # learned the thing, which is the one thing the verdict had just denied.
    memory_module.remember_long_term(
        knowledge,
        experience_id,
        f"{'learned' if learned else 'attempted'} {source_name} "
        f"({verdict}, holdout {improvement:+.6f})",
        confidence=confidence,
    )
    # Co-occurrence is a fact about the corpus, not a claim that learning
    # worked, so it is recorded whatever the verdict.
    #
    # It used to sit inside the accepted-only block, and that starved the
    # relationship derivation of evidence: 21 accepted cycles out of 124 meant
    # the concept graph had almost nothing to derive similarity or ordering
    # from, and the nine remaining relationship types stayed unreachable
    # because of a gate that had nothing to do with them. A document that was
    # read and rejected still says which terms it contains, and relationships
    # should be derived from what the corpus contains rather than
    # from what the model managed to learn.
    #
    # What stays gated is everything that asserts the system is standing behind
    # the content: the retrievable copy of the text, the claims harvested from
    # it, and the programming and register structure. Those are statements about
    # knowledge Jupiter holds, and a rejected cycle is a reason not to make one.
    relationships_module.record_co_occurrence_from_source(
        knowledge, promoted, source_name
    )
    if learned:
        memory_module.remember_external(
            knowledge,
            f"source-{source_name}",
            source_text,
            origin=str(source_path),
        )
        # Programming knowledge should be held as structure
        # rather than as memorized text. A corpus of code going only into the
        # byte stream leaves Jupiter unable to say what a function is or what a
        # file depends on.
        code_knowledge.record(knowledge, source_name, source_text, cycle=cycle)
        # The corpus is not all English. A stack trace, a log
        # file and a fragment of decoded binary are not the same material as a
        # novel, and nothing was recording which was which.
        register.record(knowledge, source_name, source_text, cycle=cycle)
        # An accepted document is one whose claims the system
        # is standing behind, so its assertions are recorded now. Term
        # co-occurrence cannot say that one source denies what another asserts,
        # which is why two sources making opposite claims had no way to be found
        # disagreeing before this.
        assertions.harvest(knowledge, source_name, source_text, cycle=cycle)
    # The graded relationship types are derived on every cycle, not only an
    # accepted one, because the evidence they rest on is the co-occurrence
    # recorded just above and that is now recorded whatever the verdict. Deriving
    # them here rather than inside the accepted block is what lets the graph
    # accumulate evidence from a corpus Jupiter mostly rejects, which is the
    # honest situation: the material is there even when the bounded update did
    # not improve on it.
    #
    # observe() folds a repeat sighting into the existing edge and confidence
    # rises toward one, so re-deriving a pair that already exists strengthens it
    # rather than duplicating it.
    derived = relationships_module.derive_graded_relationships(knowledge)
    tel = Telemetry(c.telemetry_file)
    tel.write(
        "relationships_derived",
        learning_cycle=cycle,
        similar=len(derived["SIMILAR_TO"]),
        precedes=len(derived["PRECEDES"]),
    )
    if derived["SIMILAR_TO"] or derived["PRECEDES"]:
        print(
            f"  relationships: {len(derived['SIMILAR_TO'])} SIMILAR_TO, "
            f"{len(derived['PRECEDES'])} PRECEDES derived from "
            f"co-occurrence evidence"
        )
    memory_module.clear_working(knowledge)
    return promoted, experience


def learning_signals(c, pieces):
    """Per-candidate frustration band and confidence, keyed by source path.

    The source recorded against a cycle is the document path, so the two are
    joined on that. A document that has never been judged contributes no
    signal rather than a zero one, which matters: a zero would be read as
    measured and confident when it means unmeasured: confidence is evidence,
    not a default.
    """
    try:
        state = confidence_module.load(c)
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(state, dict):
        return {}
    signals = {}
    for piece in pieces:
        path = piece.path if isinstance(piece, Candidate) else piece[0]
        key = selection.key_for(path)
        record = state.get(key)
        if not isinstance(record, dict):
            continue
        try:
            observed = int(record.get("observed", 0))
        except (TypeError, ValueError):
            observed = 0
        if observed <= 0:
            continue
        measured = frustration.for_source(state, key)
        # The band comes from the frustration measurement, but the confidence
        # does not: frustration reports how often a source defeated an update,
        # and its rows carry no confidence of their own. Reading it from the
        # confidence record is the only place it exists, and defaulting it to
        # zero would make every unmeasured source look maximally urgent.
        signals[key] = {
            "band": measured.get("band", ""),
            "confidence": record.get("confidence", 0.0),
            "likely_cause": measured.get("likely_cause", ""),
        }
    return signals


def _model_size(model):
    """How many neurons the model has, in the terms the architecture uses.

    Parameters, not layers. A layer count says almost nothing about capacity:
    one 4096-wide block and six 256-wide blocks are both "four layers" and are
    not remotely the same model. The hidden width is reported alongside it
    because that is the number "more neurons" actually means, and it is
    the one a growth actually moves.
    """
    return {
        "parameters": model.parameter_count(),
        "n_layer": int(getattr(model.config, "n_layer", 0)),
        "n_embd": int(getattr(model.config, "n_embd", 0)),
        "vocab_size": int(getattr(model.config, "vocab_size", 0)),
    }


def _neuron_report(model, baseline):
    """The model's size against where it started, with the change.

    The percentage is against the size at the start of the run rather than the
    previous cycle, because a per-cycle percentage is unreadable: a growth that
    adds a million parameters is a large change once and zero every cycle after,
    so reporting it each cycle would print 0.00% forever and say nothing about
    the model being a different size. Against the start answers the question a
    person asking "is it getting bigger" actually has.

    An unchanged model reports +0.00% rather than being hidden, so a flat number
    is visibly a measurement and not a missing one.
    """
    current = _model_size(model)
    before = int((baseline or current).get("parameters", 0))
    after = int(current["parameters"])
    current["baseline_parameters"] = before
    current["parameters_added"] = after - before
    current["percent_change"] = (
        round((after - before) / before * 100.0, 4) if before else None
    )
    return current


def _format_neurons(size):
    """One line saying how big the model is and how much that has changed."""
    added = int(size.get("parameters_added", 0))
    change = size.get("percent_change")
    if change is None:
        movement = "baseline unknown"
    elif added == 0:
        movement = "unchanged (+0.00%)"
    else:
        movement = f"{added:+,} ({change:+.2f}%)"
    return (
        f"neurons: {size['parameters']:,} params | "
        f"{size['n_layer']} layers x {size['n_embd']} wide | "
        f"vs start {movement}"
    )


def _plasticity_cycle(
    model,
    c,
    train_tokens,
    holdout_tokens,
    tel,
    cycle,
    before_seen_metrics,
    before_holdout_metrics,
    learner,
    deadline,
):
    """Run one bounded direct-plasticity learning event.

    This is deliberately separate from ``train.run_micro_learning``.  The
    gradient learner remains available as a baseline, while this path changes
    selected weights directly from reward-modulated Hebbian gain/loss.
    """
    started = time.time()
    if deadline is not None and started > deadline:
        raise CycleTimeLimitExceeded(
            f"cycle {cycle} exceeded its {c.worker_cycle_max_seconds:.0f}s allowance before plasticity"
        )

    cycle_snapshot = snapshot_model(model)
    x, y = make_batch(train_tokens, c.block_size, c.device)
    result = learner.apply(model, x, y, feedback=1.0)

    if deadline is not None and time.time() > deadline:
        if result.applied:
            restore_model(model, cycle_snapshot)
        raise CycleTimeLimitExceeded(
            f"cycle {cycle} exceeded its {c.worker_cycle_max_seconds:.0f}s allowance during plasticity"
        )

    after_seen_metrics = measure_on(
        train_tokens, model, c, samples=int(getattr(c, "measurement_windows", 3))
    )
    after_holdout_metrics = measure_on(
        holdout_tokens, model, c, samples=int(getattr(c, "measurement_windows", 3))
    )
    holdout_improvement = float(before_holdout_metrics["loss"]) - float(
        after_holdout_metrics["loss"]
    )
    train_improvement = float(before_seen_metrics["loss"]) - float(
        after_seen_metrics["loss"]
    )

    tel.write(
        "plasticity_update",
        learning_cycle=cycle,
        applied=result.applied,
        reason=result.reason,
        feedback=result.feedback,
        mean_reward=result.mean_reward,
        correct_fraction=result.correct_fraction,
        gain_tokens=result.gain_tokens,
        loss_tokens=result.loss_tokens,
        layers_changed=result.layers_changed,
        total_update_norm=result.total_update_norm,
        layer_updates=result.layer_updates,
        holdout_loss_before=float(before_holdout_metrics["loss"]),
        holdout_loss_after=float(after_holdout_metrics["loss"]),
        holdout_loss_improvement=holdout_improvement,
        elapsed_seconds=time.time() - started,
    )

    print(
        f"  PLASTICITY: {'GAIN/LOSS applied' if result.applied else 'no change'} | "
        f"correct={result.correct_fraction:.3f} | "
        f"gain={result.gain_tokens} loss={result.loss_tokens} | "
        f"layers={result.layers_changed} | "
        f"update_norm={result.total_update_norm:.6g} | "
        f"holdout {holdout_improvement:+.6f}"
    )

    return (
        {
            "holdout_loss_before": float(before_holdout_metrics["loss"]),
            "holdout_loss_after": float(after_holdout_metrics["loss"]),
            # Perplexity and accuracy travel with the loss. The cycle measures all
            # three and returned only the loss, so every plasticity cycle recorded
            # a null accuracy and a null perplexity. Accuracy is the one number that
            # says whether the model can produce the right byte rather than merely
            # assign it a plausible probability, so a log without it cannot answer
            # whether learning is moving toward capability at all.
            "holdout_perplexity_before": before_holdout_metrics.get("perplexity"),
            "holdout_perplexity_after": after_holdout_metrics.get("perplexity"),
            "holdout_accuracy_before": before_holdout_metrics.get("accuracy"),
            "holdout_accuracy_after": after_holdout_metrics.get("accuracy"),
            "holdout_loss_improvement": holdout_improvement,
            "train_loss_before": float(before_seen_metrics["loss"]),
            "train_loss_after": float(after_seen_metrics["loss"]),
            "train_loss_improvement": train_improvement,
            "successful_steps": 1 if result.applied else 0,
            "gradient_recoveries": 0,
            "plasticity_applied": bool(result.applied),
            "plasticity": result.telemetry(),
            "elapsed_seconds": time.time() - started,
        },
        learner.config.learning_rate,
        None,
    )


def run_worker(c, tel, cycle_budget):
    out_dir = Path(c.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycle_path = Path(c.learning_state)
    state_path = Path(c.worker_state)
    checkpoint_path = out_dir / PUBLISHED_CHECKPOINT
    resume_path = Path(c.resume_checkpoint)

    tokenizer = load_tokenizer(c)
    model, architecture_path = build_model(c, tokenizer.vocab_size)
    attach_checkpoint(
        model,
        [resume_path, checkpoint_path],
        architecture_path,
    )
    # The record is written from the model that was built, not from the config
    # that asked for it. An approved growth makes the live model larger than the
    # configuration describes, and recording the configuration would write down
    # a shape the weights do not have.
    #
    # A refusal here is offered back to the person rather than only reported.
    # Architecture 38 requires a human to authorise a change to the recorded
    # architecture, and this is the one place a normal run hits it: the model
    # grew, so the record of what the model is no longer what it was. The gate was
    # right to refuse and useless to refuse, because the only way past it was to
    # know the name of an environment variable before the process started. The
    # question is asked once, at the moment the answer is needed, and declining
    # leaves the run stopped exactly as before.
    try:
        save_architecture(
            c,
            architecture_path,
            tokenizer.vocab_size,
            shape=model.config.to_dict(),
        )
    except control.ControlDenied as refusal:
        # The grant is offered here, before the run gives up, because a refusal
        # that ends the process is the one the person cannot recover from without
        # starting again. It used to propagate: the message naming the variable was
        # the last thing a run said, so answering it required leaving, setting a
        # variable in another window and starting over.
        #
        # What is offered is the architecture domain, and offering it also grants
        # the expansion work that needed it, so one answer unblocks the run rather
        # than moving it to the next wall. Declining re-raises exactly as before,
        # which is the correct outcome for a control nobody answered.
        offered = offer_grant(c, "architecture", str(refusal), tel=tel)
        if not offered:
            raise
        save_architecture(
            c,
            architecture_path,
            tokenizer.vocab_size,
            shape=model.config.to_dict(),
        )

    cycle = read_cycle(cycle_path)
    learning_mode = (
        os.environ.get("JUPITER_LEARNING_MODE", "plasticity").strip().lower()
    )
    if learning_mode not in {"plasticity", "gradient"}:
        raise ValueError(
            f"Unknown JUPITER_LEARNING_MODE={learning_mode!r}; use plasticity or gradient"
        )
    plasticity_learner = None
    if learning_mode == "plasticity":
        # The settings are collected into a mapping and handed to the loader rather
        # than passed as keywords to the dataclass. The loader rejects a name the
        # config does not have, so a setting that stops existing, or one added here
        # with a misspelling, is refused with a message naming it rather than
        # silently ignored. Building the dataclass here directly would raise a
        # TypeError for a misspelling anyway, but only for the keys that are passed:
        # a setting this list forgets would take its default with nothing said.
        #
        # An unparseable environment value is a refusal rather than a fallback to the
        # default, because a setting that silently reverted to a default is a run
        # that behaves differently from the one that was asked for.
        def plasticity_setting(variable, default, cast):
            raw = os.environ.get(variable, default)
            try:
                return cast(raw)
            except (TypeError, ValueError) as failure:
                raise ValueError(
                    f"{variable}={raw!r} is not a valid {cast.__name__}: {failure}"
                ) from failure

        settings = {
            "learning_rate": plasticity_setting(
                "JUPITER_PLASTICITY_LR", "0.0005", float
            ),
            "max_update_norm": plasticity_setting(
                "JUPITER_PLASTICITY_MAX_UPDATE_NORM", "0.0025", float
            ),
            "homeostasis": plasticity_setting(
                "JUPITER_PLASTICITY_HOMEOSTASIS", "0.00002", float
            ),
            "max_positions": plasticity_setting(
                "JUPITER_PLASTICITY_MAX_POSITIONS", "256", int
            ),
        }
        plasticity_learner = plasticity.RewardModulatedHebbian(
            plasticity.load_plasticity_config(settings)
        )
        learning_rate = plasticity_learner.config.learning_rate
        optimizer = None
        optimizer_resumed_from = "not used (plasticity mode)"
    else:
        learning_rate = float(c.learning_rate)
        optimizer, learning_rate = reset_optimizer(model, c, learning_rate)
        # The optimizer's moments are resumed before any cycle runs, so a continued
        # run is the same run rather than a new one that happens to start from the
        # same weights. Reported rather than silent, because "the first steps of
        # every cycle moved on a cold estimate" was a real and invisible cost.
        optimizer_resumed_from = load_optimizer_state(optimizer, model, c)
    print(f"learning mode:     {learning_mode}")
    print(f"optimizer:         {optimizer_resumed_from}")

    pieces = load_candidates(c, tel)
    selection_state = selection.load(c)
    signals = learning_signals(c, pieces or [])
    frustration_measurements = frustration.for_sources(c) if pieces else []
    regression_available = False
    regression_paths = []
    regression_holdout = None
    if pieces is not None:
        # Only the reserved documents are read, and only for as long as the
        # yardstick is being built. Reading every candidate to find them is the
        # thing this change exists to stop.
        try:
            # The pieces are kept under a second name before reserve() replaces
            # them. reserve() returns the documents it held out together with the
            # ones still available to learn from, so rebinding pieces to its second
            # return value leaves a list that cannot contain the documents named in
            # its first. The lookup below then failed for every document, every
            # run, and the yardstick was never built.
            #
            # The effect was invisible from the outside: without a yardstick every
            # cycle is REVIEW rather than ACCEPT or REJECT, so nothing was ever
            # accepted, so the growth cadence never reached a hundred, and Jupiter
            # never once asked to grow. The learning was still happening. It was
            # being recorded as unverified, forever.
            candidates = list(pieces)
            reserved, pieces = regression.reserve(c, candidates)
            regression_paths = list(reserved)
            reserved_bodies = []
            for path in regression_paths:
                match = next(
                    (
                        piece
                        for piece in candidates
                        if (piece.path if isinstance(piece, Candidate) else piece[0])
                        == path
                    ),
                    None,
                )
                if match is None:
                    # reserve() is expected to only return documents from the
                    # candidates it was given. This is the second line of defence
                    # for the case where it does not, and it matters because the
                    # alternative is a KeyError naming an internal detail rather
                    # than the document that could not be read.
                    raise FileNotFoundError(
                        f"a reserved regression document is not among the "
                        f"candidates this run loaded: {path}"
                    )
                reserved_bodies.append(read_text(match))
            # The yardstick is built once from a bounded window of each reserved
            # document and concatenated, so degradation is measured across a
            # sample of the corpus rather than inside one file. Bounded because a
            # whole book would make the holdout tensor enormous, and fixed
            # because a yardstick that moved between cycles would stop being one.
            regression_holdout = regression.tokens_for(
                c,
                "\n\n".join(body[: c.max_chunk_chars] for body in reserved_bodies),
                tokenizer,
            )
            regression_available = True
        except (RuntimeError, FileNotFoundError) as reason:
            # Without a yardstick every verdict becomes REVIEW, which is the
            # honest outcome: improvement can no longer be separated from
            # forgetting, so the worker will not claim success.
            tel.write(
                "regression_unavailable",
                reason=str(reason),
            )
            print(f"[worker] regression detection unavailable: {reason}")

    knowledge = store_module.Store(c.knowledge_store)
    confidence_state = confidence_module.load(c)
    best_holdout = None
    best_regression = None
    best_snapshot = None
    # Whether this run actually put the weights back, and where it restored
    # them to. Both are reported in the run summary, so they are set here rather
    # than left to a branch that may never run: a summary that names a rollback
    # which did not happen is worse than one that says nothing.
    rolled_back = False
    restored_regression = None
    # What the model looked like before this run touched it. A rejected update
    # has to be reversible, and a run whose cycles are all
    # rejected has no accepted state to fall back to, so the only honest state
    # to return to is the one the run started from.
    origin_snapshot = snapshot_model(model)
    last_steps = 0
    last_source = "none"
    last_verdict = "none"
    # The last few cycles' holdout gains, newest last. This is what the acceptance
    # gate judges once there are enough of them, because one cycle's figure cannot
    # be read: its noise is 2.7 times its own signal.
    recent_gains = []
    last_confidence = 0.0
    stalled = 0
    total_recoveries = 0
    completed = 0
    started = time.time()
    status = "running"
    reason = "no limit reached"
    cycle_results = []

    # The size at the start of the run, so every later report can say how much
    # bigger the model has become rather than only what it is now.
    neuron_baseline = _model_size(model)

    print("=" * 72)
    print("JUPITER CONTINUOUS LEARNING WORKER")
    print(_format_neurons(_neuron_report(model, neuron_baseline)))
    print(f"tokenizer:         {tokenizer.name} vocab={tokenizer.vocab_size}")
    print(f"context window:    {model.config.block_size} tokens")
    print(
        f"device:            {c.device}  ({getattr(c, 'device_reason', 'not stated')})"
    )
    print(f"cycle budget:      {cycle_budget}")
    print(f"time budget:       {c.worker_max_seconds:.0f}s")
    print(f"per-cycle limit:   {c.worker_cycle_max_seconds:.0f}s")
    print(f"patience:          {c.worker_patience} unaccepted cycles")
    if plasticity_learner is not None:
        print(
            f"plasticity:        reward-Hebbian gain/loss | lr={plasticity_learner.config.learning_rate:g} | max_update_norm={plasticity_learner.config.max_update_norm:g}"
        )
    print(f"defer band:        {c.defer_band:.6f}")
    print(
        f"regression docs:  {len(regression_paths)}"
        if regression_available
        else "regression docs:  NONE"
    )
    print(f"candidates:        {len(pieces) if pieces else 0}")
    print(f"resources:         {utilization.describe(utilization.sample(out_dir))}")
    print("=" * 72)

    while True:
        stop = decide_stop(
            c,
            completed,
            cycle_budget,
            time.time() - started,
            total_recoveries,
            stalled,
            _stop_requested,
        )
        if stop is not None:
            status, reason = stop
            break
        if pieces is None:
            tel.write(
                "worker_idle",
                reason="no_candidate",
                sleep_seconds=c.worker_idle_seconds,
            )
            print(
                f"[worker] no candidate; "
                f"sleeping {c.worker_idle_seconds:.0f}s before retrying"
            )
            time.sleep(c.worker_idle_seconds)
            pieces = load_candidates(c, tel)
            # The corpus can change underneath a long run, so the signals the
            # queue ranks on are rebuilt with it rather than kept from startup.
            signals = learning_signals(c, pieces or [])
            frustration_measurements = frustration.for_sources(c) if pieces else []
            continue

        # Which document to learn is a measurement, not a directory order.
        # Several named causes of repeated difficulty are answerable, and selection
        # gates generation on the ones more material can answer; the age term
        # there keeps a permanently difficult source from starving the corpus.
        #
        # The folder bonus closes the loop the corpus was sorted for: the
        # evaluation report says which capability is weak, and the material that
        # would strengthen it is moved up the queue. It is read per cycle rather
        # than once at startup for the same reason the signals are: a long run
        # should follow a capability that improves rather than keep chasing one
        # that has been fixed.
        source_path, source_text, chosen_because = selection.choose(
            pieces,
            signals,
            selection_state,
            cycle + 1,
            folder_bonus=skills.priorities(c),
        )
        if source_path is None:
            # choose() returns None when nothing is left to select from. It used
            # to be handed straight to Path(), which raises a TypeError naming
            # None rather than the queue being empty, and the cycle ended as a
            # crash instead of a decision. The worker is above this and has
            # already handled an empty candidate list, so this is the narrow
            # race of a queue emptied between loading and choosing.
            tel.write(
                "no_candidate",
                reason=chosen_because,
                sleep_seconds=c.worker_idle_seconds,
            )
            print(
                f"[worker] nothing selectable: {chosen_because}; "
                f"sleeping {c.worker_idle_seconds:.0f}s"
            )
            time.sleep(c.worker_idle_seconds)
            continue

        # The chosen document is read now, and only now. Selection ranked on
        # paths and sizes; this is the first point in a run where the bytes of
        # a candidate are needed, and it is the only point where a document is
        # held. A twenty-one gigabyte corpus therefore costs the size of the one
        # document being learned rather than the size of the whole corpus.
        source_text = read_text(source_text)

        selection.mark_seen(selection_state, source_path, cycle + 1)
        last_source = Path(source_path).name
        cycle += 1

        # The chain runs conversation -> experience -> memory ->
        # curation -> learning. Three of those four seams existed; this is the
        # fourth. sample.py recorded every conversation to out/experiences.jsonl
        # and nothing in the project read it, so a tree-wide search for that
        # filename returned exactly one file: the one writing it. Every
        # conversation Jupiter had was therefore invisible to the system built to
        # learn from it, and the corpus could only ever grow from material a
        # person placed in data/raw.
        #
        # Rendering happens here, at the top of the cycle, because the result is
        # raw text that curation turns into a curated document on its next pass.
        # It is deliberately not learned in the same breath: the acceptance gate
        # downstream judges anything learned from it on its evidence, exactly as
        # it judges a human file, and the file it will be judged from is the
        # curated one rather than this raw copy.
        if experience_interval(c, cycle):
            published = publish_experience(c, tel)
            if published.get("written"):
                print(
                    f"[worker] published {published['written']} conversation(s) "
                    f"from experience into {published['path']}"
                )

        # Generated examples and Jupiter's own experiences are legitimate data
        # origins. Both were unimplemented, which left the corpus closed:
        # Jupiter could only learn what a person had already placed in data/raw,
        # and could never meet a case nobody thought to include. When a source
        # keeps defeating a bounded update for a reason more material could
        # answer, Jupiter writes some itself. The gate is on the cause, not the
        # score, so this does not fire for a representation or capacity problem
        # that more examples would only reinforce. Run after the signals are
        # read, because a document written this cycle is curated into the corpus
        # on the next pass rather than learned in the same breath.
        # The expansion loop is meant to run as the corpus is learned, not
        # once on the very first cycle. It was guarded by `cycle == 1`, and the
        # cycle counter is read from disk at startup, so it started at whatever
        # the last run reached. Past the first run that condition was never true
        # and no growth was ever proposed, however many sources frustration was
        # blaming on capacity. Seventy-eight sources were blamed and nothing was
        # proposed, which is the shape of a feature that looks present and does
        # not fire.
        #
        # The loop is now entered when the capacity diagnosis has something to
        # say that has not already been said: a proposal is made when a source is
        # short of capacity and no proposal is already waiting on a human or
        # being measured. That is the chapter's own condition, and it is checked
        # against the expansion state rather than against a counter that has
        # nothing to do with whether growth is warranted.
        if frustration_measurements and expansion.should_propose_growth(
            c, frustration_measurements
        ):
            # The first half of the expansion loop. A source whose frustration says
            # it cannot represent this with the current architecture gets a
            # recorded proposal. Recording costs nothing and is a measurement;
            # acting on it is a change to how much Jupiter learns, so the grant
            # stays behind the safety gate and is not taken here.
            proposals = expansion.propose(c, frustration_measurements, cycle)
            if proposals:
                expansion.record_proposals(c, proposals, cycle)
                for proposal in proposals:
                    print(
                        f"\n[expansion] {Path(proposal['source']).name} is short of "
                        f"capacity ({proposal['current_limit']} -> "
                        f"{proposal['proposed_limit']} concepts per cycle); recorded, "
                        f"not applied. Set {control.GRANT_VARIABLE}=expansion to allow it."
                    )
            # The parameter-changing half. A source short of capacity is proposed
            # one layer more, recorded, and refused until a human authorises it.
            # It is not applied here even when authorised, because the model this
            # run is already learning in was built before the authorisation; the
            # growth takes effect on the next run, which is the same rule the
            # architecture record follows.
            growth = expansion.propose_growth(c, frustration_measurements, cycle)
            if growth:
                print(
                    f"\n[growth] {Path(growth['source']).name} is short of model "
                    f"capacity ({growth['frustration_index']:.3f}); proposed "
                    f"n_layer {growth['from_layers']} -> {growth['to_layers']}, "
                    f"recorded, not applied. Set {control.GRANT_VARIABLE}=expansion "
                    "to allow it, then approve it."
                )
            synthesize.run(
                c,
                model,
                tokenizer,
                pieces,
                frustration_measurements,
                cycle,
                tel,
            )
        # Each cycle takes a different part of the document, so repeated
        # cycles over one document eventually cover all of it rather than
        # re-learning the same opening passage every time.
        #
        # How much a cycle reads scales with how much there is. A cycle that
        # reads one fixed window makes a large document impractical rather than
        # merely slow, which is a limit on what can be fed in at all: a five
        # megabyte document took 417 cycles to cover once. The number of windows
        # is derived from the document so an operator does not have to know a
        # document's size before choosing a budget for it, and it is bounded so
        # one cycle cannot read the whole corpus.
        windows = _windows_per_cycle(source_text, c)
        # The window is taken before tokenizing, and split_tokens only encodes
        # what it is handed: a 55MB document becomes a 48,000 character window
        # first, and encoding that takes 0.00s against 9.9s for the whole file.
        # Tokenizing first and discarding the result would have spent every
        # cycle converting text it had already decided not to read.
        source_text = window_for(
            source_text, cycle, c.max_chunk_chars, windows_per_cycle=windows
        )
        train_tokens, holdout_tokens = split_tokens(
            source_text,
            c.block_size,
            c.holdout_fraction,
            tokenizer,
        )
        # Measured over as many windows as configuration asks for. One window is
        # a single sample of a document that has thousands of them, and holding
        # the weights fixed while reading sixteen instead of one shrank the spread
        # of this measurement by 5.46x. The variance the acceptance gate had been
        # reading was largely which window it drew.
        samples = int(getattr(c, "measurement_windows", 3))
        before_seen_metrics = measure_on(train_tokens, model, c, samples=samples)
        before_holdout_metrics = measure_on(holdout_tokens, model, c, samples=samples)
        before_seen = before_seen_metrics["loss"]
        before_holdout = before_holdout_metrics["loss"]
        regression_before = None
        if regression_available:
            regression_before = regression.measure(model, c, regression_holdout)
        tel.write(
            "worker_cycle_start",
            learning_cycle=cycle,
            source=str(source_path),
            candidates=len(pieces),
            chosen_because=chosen_because,
            train_loss_before=before_seen,
            holdout_loss_before=before_holdout,
            regression_loss_before=regression_before,
            learning_rate=learning_rate,
        )
        write_status(
            c.learning_status,
            "learning",
            cycle,
            last_source,
            last_steps,
            best_holdout,
            before_holdout,
        )
        print(
            f"\n--- cycle {cycle} "
            f"source={last_source} "
            f"holdout_before={before_holdout:.6f}"
            + (
                f" regression_before={regression_before:.6f}"
                if regression_before is not None
                else " regression=NONE"
            )
            + " ---"
        )

        cycle_deadline = time.time() + c.worker_cycle_max_seconds
        try:
            if learning_mode == "plasticity":
                cycle_result, learning_rate, optimizer = _plasticity_cycle(
                    model,
                    c,
                    train_tokens,
                    holdout_tokens,
                    tel,
                    cycle,
                    before_seen_metrics,
                    before_holdout_metrics,
                    plasticity_learner,
                    cycle_deadline,
                )
            else:
                cycle_result, learning_rate, optimizer = run_micro_learning(
                    model,
                    c,
                    train_tokens,
                    holdout_tokens,
                    tel,
                    cycle,
                    before_seen_metrics,
                    before_holdout_metrics,
                    learning_rate,
                    optimizer,
                    deadline=cycle_deadline,
                )
        except CycleTimeLimitExceeded as failure:
            status = "cycle_time_limit"
            reason = str(failure)
            tel.write("worker_cycle_timeout", learning_cycle=cycle, reason=reason)
            print(f"[worker] {reason}")
            break
        except RuntimeError as failure:
            # The proven learner raises only when it could not recover inside
            # its own retry budget. That is numerical instability, so stop and
            # roll back instead of training on top of a diverging model.
            status = "unstable"
            reason = f"learner could not stabilize: {failure}"
            regression_after = None
            if regression_available:
                regression_after = regression.measure(model, c, regression_holdout)
            confidence_module.observe(
                c, confidence_state, last_source, acceptance.ROLLBACK
            )
            experiment.write_experiment(
                c,
                cycle,
                source_path,
                {
                    "holdout_loss_before": before_holdout,
                    # A rollback changed no weights, so there is no loss after it to
                    # record. It is written as absent rather than left to default,
                    # because the two ways of saying the same thing in one record
                    # is how the null that this file was repaired for got there.
                    "holdout_loss_after": None,
                    "learning_method": learning_mode,
                    "final_learning_rate": learning_rate,
                    "micro_steps": c.micro_steps,
                    "gradient_clip": c.grad_clip,
                    "gradient_stop": c.gradient_stop,
                    "successful_steps": 0,
                    "gradient_recoveries": total_recoveries,
                },
                acceptance.ROLLBACK,
                str(failure),
                regression_before,
                regression_after,
                confidence_module.value_of(
                    confidence_module.entry_for(confidence_state, last_source)
                ),
                cycle,
                utilization.sample(out_dir),
            )
            # A cycle that rolled back still visited a document, so its place in
            # the rotation has to be recorded. Without this the visit is only
            # written when some later cycle happens to save, and a run that
            # ended on a rollback would forget where it had been.
            selection.save(c, selection_state)
            tel.write(
                "worker_instability",
                learning_cycle=cycle,
                reason=str(failure),
                total_recoveries=total_recoveries,
            )
            break

        completed += 1
        recoveries = cycle_result["gradient_recoveries"]
        total_recoveries += recoveries
        last_steps = cycle_result["successful_steps"]
        improvement = cycle_result["holdout_loss_improvement"]

        regression_after = None
        regression_delta = None
        if regression_available:
            regression_after = regression.measure(model, c, regression_holdout)
            # Measured as a SHARE of the loss the reserved set started at, not as
            # an absolute number. An absolute tolerance is not comparable across
            # cycles: the reserved document's loss starts around 1.2 and drifts up
            # toward 5 as the corpus is learned, so the same 0.15 of damage is
            # twelve per cent early and three per cent later. Judged absolutely,
            # every later cycle was measured against an effectively lower bar and
            # rejected for damage that was a smaller fraction of what was there.
            #
            # Measured over 200 cycles, the median regression was +0.296 absolute
            # against a 0.15 tolerance - twice over - while the same regressions
            # are a median of 5.7 per cent of the reserved document's own loss.
            # The absolute form rejected almost everything; the relative form is
            # what the number actually means, and it is the same shape
            # evaluation.py already uses for every other reported change.
            regression_delta = (
                (regression_after - regression_before) / regression_before
                if regression_before
                else None
            )

        # The conversation yardstick is run after the update, against the model as
        # it now stands, and compared with the run before it. It is measured here
        # rather than only in the evaluation report because the
        # the acceptance gate to judge an update, and a gate that cannot see that
        # an update cost the model its ability to hold a turn is not judging the
        # thing that matters.
        #
        # It runs periodically rather than on every cycle, because each pass asks
        # the model to generate and a generation costs far more than a forward
        # pass. A gate too slow to run is a gate that gets turned off.
        conversation_change = None
        conversation_result = None
        if completed % max(1, int(c.conversation_eval_interval)) == 1:
            previous = conversation_eval.load(c)
            conversation_result = conversation_eval.run_set(
                c, model=model, tokenizer=tokenizer
            )
            if conversation_result is not None:
                conversation_eval.save(c, conversation_result)
                comparison = conversation_eval.compare(previous, conversation_result)
                if comparison is not None:
                    conversation_change = comparison["change"]
                tel.write(
                    "conversation_eval",
                    measured=True,
                    pass_rate=conversation_result.get("pass_rate"),
                    change=conversation_change,
                )
            else:
                # Nothing to measure. Recorded as unmeasured rather than as no
                # change, because "did not get worse" and "was not measured" are
                # different facts and only one of them is a pass.
                tel.write("conversation_eval", measured=False)

        # The same idea as the conversation set, applied to every skill rather
        # than one. Run on the same interval and for the same reason: a
        # measurement too expensive to run every cycle is a measurement that
        # stops being taken.
        #
        # This is the number that can tell learning from document difficulty.
        # The per-cycle change in holdout loss is a fifth of its own standard
        # deviation, so a document's loss cannot say whether an update helped,
        # and everything that depends on that verdict inherits the ambiguity.
        probe_change = None
        try:
            probe_sets = probes.load(c)
        except (OSError, ValueError):
            probe_sets = {}
        if probe_sets:
            probe_results = probes.evaluate(c, model, tokenizer, device=c.device)
            probe_target = Path(getattr(c, "probe_results", "out/probe_results.json"))
            probe_target.parent.mkdir(parents=True, exist_ok=True)
            previous_probes = {}
            if probe_target.exists():
                try:
                    previous_probes = json.loads(
                        probe_target.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    previous_probes = {}
            probe_target.write_text(
                json.dumps(probe_results, indent=2), encoding="utf-8"
            )
            measured = [
                (name, data)
                for name, data in probe_results.items()
                if data.get("measured")
            ]
            weakest = (
                min(measured, key=lambda item: item[1]["mean_score"])
                if measured
                else None
            )
            if weakest is not None:
                skill, data = weakest
                prior = (previous_probes.get(skill) or {}).get("mean_score")
                if prior is not None:
                    probe_change = round(data["mean_score"] - prior, 4)
                print(
                    f"[probe] weakest skill {skill}: {data['mean_score']:.3f} "
                    f"over {data['measured']} probes"
                    + (
                        f" (was {prior:.3f})"
                        if prior is not None
                        else " (first measurement)"
                    )
                )
            tel.write(
                "probes",
                measured=bool(weakest),
                weakest_skill=weakest[0] if weakest else None,
                weakest_score=(weakest[1]["mean_score"] if weakest else None),
                change=probe_change,
            )

        record = confidence_module.entry_for(confidence_state, last_source)
        prior_confidence = confidence_module.value_of(record)
        prior_observed = confidence_module.observed_of(record)
        prior_accepted = int(record.get("accepted", 0))
        prior_contradictions = int(record.get("contradictions", 0))
        # The gate is given a running mean rather than this cycle's own figure.
        # Measured over 3,315 recorded cycles the per-cycle change in holdout loss
        # carries a standard deviation of 1.01 against a mean of -0.377, so the
        # noise is 2.7 times the signal, and the per-cycle defer band is 1e-4. A
        # single cycle judged against a band that small is a coin flip on which
        # document was drawn, which is why roughly half of every run has been
        # rejected for reasons that had nothing to do with the update.
        #
        # The window is short enough that sixteen bad cycles in a row still end a
        # source, and long enough that the mean carries a quarter of the noise.
        # Below the window the gate still runs, on the single-cycle figure, and the
        # reason it gives says which basis it used: a three-cycle mean presented as
        # a measurement would be the same error as the band it replaces.
        recent_gains.append(float(improvement))
        window = max(1, int(getattr(c, "acceptance_smoothing", 16)))
        if len(recent_gains) > window:
            del recent_gains[: len(recent_gains) - window]
        smoothed = (
            sum(recent_gains) / len(recent_gains)
            if len(recent_gains) >= window
            else None
        )

        verdict, verdict_reason = acceptance.classify(
            c,
            improvement,
            regression_delta,
            regression_available,
            prior_confidence,
            prior_observed,
            conversation_change=conversation_change,
            conversation_rate=(
                float(conversation_result.get("pass_rate") or 0.0)
                if conversation_result
                else None
            ),
            smoothed_improvement=smoothed,
        )
        updated = confidence_module.observe(c, confidence_state, last_source, verdict)
        last_verdict = verdict
        last_confidence = confidence_module.value_of(updated)
        regressed = (
            regression_delta is not None and regression_delta < -c.regression_tolerance
        )

        learned = verdict == acceptance.ACCEPT
        if learned:
            stalled = 0
            # The cadence counter. An accepted cycle is one that worked, so it is
            # what the growth cadence counts: a hundred rejected cycles is a
            # hundred attempts that failed, not a hundred reasons to try a
            # larger model.
            #
            # Whether a growth is warranted is asked here and answered here
            # rather than in the proposal stage, because the proposal stage
            # already has a rule of its own and the two would otherwise compete
            # for the same slot in the state file.
            cadence_state, growth_due = expansion.record_accepted(c)
            expansion.save_cadence(c, cadence_state)
            if growth_due:
                proposal, why = expansion.propose_cadence_growth(c, cycle)
                if proposal is not None:
                    expansion.mark_growth_earned(c, cadence_state["accepted_cycles"])
                    print(
                        f"\n[growth] {cadence_state['accepted_cycles']} cycles have "
                        f"been accepted, so one more layer is proposed: "
                        f"n_layer {proposal['from_layers']} -> {proposal['to_layers']} "
                        f"({why}). Approve it to apply it on the next run."
                    )
                else:
                    print(
                        f"\n[growth] cadence reached but no growth was proposed: {why}"
                    )
            # An accepted update that damaged the reserved document is still an
            # accepted update, and the acceptance gate has already had its say.
            # What the run does about it is separate: the weights are known to
            # be worse on material held out of learning, so they are put back
            # before the next cycle measures itself against them. Without this
            # the damage carried forward and every later cycle was measured
            # against it.
            if regressed and best_snapshot is not None:
                restore_model(model, best_snapshot)
                rolled_back = True
                restored_regression = f"cycle-{cycle}"
                tel.write(
                    "worker_regression_rollback",
                    learning_cycle=cycle,
                    source=last_source,
                    regression_delta=regression_delta,
                    restored_to=restored_regression,
                    note=(
                        "an accepted cycle regressed the reserved document; the "
                        "last non-regressing state was restored"
                    ),
                )
            best_holdout = cycle_result["holdout_loss_after"]
            best_regression = regression_after
            if best_regression is None or (
                regression_after is not None and regression_after < best_regression
            ):
                best_regression = regression_after
            best_snapshot = snapshot_model(model)
            save_checkpoint(model, c, cycle)
        else:
            stalled += 1
            if verdict == acceptance.REJECT:
                # A measured regression, or a measured worsening of the document
                # just learned. The weights are known to be worse than they were,
                # so the live model is returned to the last accepted state before
                # the next cycle measures anything against it. It used to be left
                # damaged until the end of the run, so every following cycle
                # measured itself against damage it had already been dealt.
                #
                # Only REJECT restores. REVIEW is a request to observe again and
                # DEFER means the update was inside the dead band; neither says
                # the weights are worse, and it is explicit that
                # REVIEW does not roll the update back.
                target = best_snapshot if best_snapshot is not None else origin_snapshot
                restore_model(model, target)
                tel.write(
                    "worker_rejected_restore",
                    learning_cycle=cycle,
                    source=last_source,
                    restored_to=(
                        "last accepted state"
                        if best_snapshot is not None
                        else "the state the run started from"
                    ),
                    regression_delta=regression_delta,
                    holdout_improvement=improvement,
                )
                print(
                    f"\n[worker] cycle {cycle} was REJECTed; live model restored to "
                    f"{'the last accepted state' if best_snapshot is not None else 'the start of the run'}"
                )

        # Remember what happened, and keep the concepts
        # this cycle actually exercised.
        promoted, experience = record_cycle_knowledge(
            knowledge,
            c,
            cycle,
            source_path,
            source_text,
            cycle_result,
            before_holdout,
            regression_delta,
            last_confidence,
            learning_rate,
            learned,
            verdict,
            verdict_reason,
            source_observed=prior_observed,
            source_accepted=prior_accepted,
            source_contradictions=prior_contradictions,
        )

        # Persist where this run actually got to, whatever the verdict. The
        # Persist where this run actually got to, whatever the verdict. The
        # published checkpoint above is deliberately ACCEPT-only, but a verdict
        # is a judgement about whether a model is ready to be shown, not about
        # whether the next run may continue from it. Gating the resume state on
        # ACCEPT as well means a run that never earns an ACCEPT teaches Jupiter
        # nothing at all, because every run restarts from random initialization
        # and the loss never gets below the starting point. Written every cycle
        # so a run that is interrupted keeps what it learned.
        save_checkpoint(model, c, cycle, path=resume_path, optimizer=optimizer)

        # Did the growth earn its place? Expansion has to be
        # measurable and reversible, and a larger model that scored worse on the
        # document it was supposed to understand better is exactly the case where
        # keeping it would be a regression rather than an improvement. The verdict
        # is recorded either way, because an expansion that was never judged is an
        # expansion nobody can tell worked.
        growth_review = None
        if best_holdout is not None:
            growth_review = expansion.resolve_growth(c, best_holdout)
        if growth_review is not None:
            kept = growth_review.get("kept")
            if kept is True:
                verdict_text = "helped; the grown model is kept"
            elif kept is False:
                verdict_text = (
                    "did not help; the growth is not worth repeating, and reverting "
                    "the layer count is a fresh training run rather than a reload"
                )
            else:
                verdict_text = "was not comparable to the loss at approval"
            print(
                f"\n[growth] n_layer {growth_review['from_layers']} -> "
                f"{growth_review['to_layers']} {verdict_text}"
            )

        # The learning record for this cycle. train.py reads "learning_cycle"
        # from this file and nothing else, so writing it here keeps both tools
        # able to run in sequence without the worker leaving the counter
        # pointing at whatever training happened to run last.
        record_cycle(
            cycle_path,
            cycle,
            model,
            source_path,
            cycle_result,
            checkpoint_path,
        )

        # What this cycle did, written to the experiment log. Every cycle,
        # whatever the verdict: a rejected update is evidence about the material
        # as much as an accepted one, and a log holding only successes cannot be
        # read to find out why a document kept being refused.
        cycle_results.append(
            {
                "cycle": cycle,
                "source": str(source_path),
                "decision": verdict,
                "reason": verdict_reason,
                "improvement": improvement,
                "holdout_before": before_holdout,
                "holdout_after": cycle_result.get("holdout_loss_after"),
                "regression_delta": regression_delta,
                "steps": cycle_result.get("successful_steps"),
                "recoveries": cycle_result.get("gradient_recoveries"),
                "learning_rate": learning_rate,
            }
        )
        experiment.write_experiment(
            c,
            cycle,
            source_path,
            {
                "holdout_loss_before": before_holdout,
                # Written because experiment.py records it from here, and it was
                # not being passed, so every recorded cycle carried
                # after_metrics.holdout_loss = null. Nothing downstream could read
                # what a cycle bought: the evaluation report, frustration,
                # autonomy and the mode sweep all skip a null, which is why a sweep
                # of this reported zero gain for every setting rather than an
                # error. It was not always absent; the rollback path above passes
                # the same key, so two paths out of three wrote it and the normal
                # one did not.
                "holdout_loss_after": cycle_result.get("holdout_loss_after"),
                # Which rule produced this cycle, so a log holding both can be read
                # back per method instead of reporting two learning systems as one.
                "learning_method": learning_mode,
                "final_learning_rate": learning_rate,
                "micro_steps": c.micro_steps,
                "gradient_clip": c.grad_clip,
                "gradient_stop": c.gradient_stop,
                "successful_steps": cycle_result.get("successful_steps", 0),
                "gradient_recoveries": cycle_result.get("gradient_recoveries", 0),
            },
            verdict,
            verdict_reason,
            regression_before,
            regression_after,
            confidence_module.value_of(updated),
            cycle,
            utilization.sample(out_dir),
        )

        # Where the queue got to. Written every cycle rather than only on the
        # branches that end a run, because the visit is recorded at the top of
        # the cycle and anything that does not write it back forgets it. The
        # rotation guarantee is that every document is seen once per pass, and a
        # queue that resets between runs cannot make that promise.
        selection.save(c, selection_state)

        # The growth the run proposed is offered to whoever is watching, and only
        # to whoever is watching: the prompt is skipped when there is no terminal or
        # the expansion grant is absent, so a detached run is never left waiting on
        # input that will not arrive.
        global _last_best_holdout
        _last_best_holdout = best_holdout
        prompt_for_growth(c, neuron_baseline, tel=tel)

    review_summary = experiment.review(cycle_results)
    elapsed = time.time() - started
    summary = {
        "status": status,
        "reason": reason,
        "learning_cycle": cycle,
        "cycles_completed": completed,
        "cycle_budget": cycle_budget,
        "regression_available": regression_available,
        "regression_source": (str(regression_paths[0]) if regression_paths else None),
        "total_recoveries": total_recoveries,
        "best_holdout": best_holdout,
        "restored_regression": restored_regression,
        "rollback_verified": rolled_back,
        "review": review_summary,
        "growth_review": growth_review,
        "final_learning_rate": learning_rate,
        "elapsed_seconds": elapsed,
        "resources": utilization.sample(out_dir),
    }
    write_state(state_path, summary)
    tel.write("worker_finish", **summary)
    write_status(
        c.learning_status,
        "stopped",
        cycle,
        last_source,
        last_steps,
        best_holdout,
        restored_regression if restored_regression is not None else best_holdout,
    )
    print("=" * 72)
    print(f"WORKER STOPPED: {status}")
    print(f"reason:         {reason}")
    print(f"cycles:         {completed}/{cycle_budget}")
    print(f"recoveries:     {total_recoveries}")
    print(
        f"regression docs: {len(regression_paths)}"
        if regression_available
        else "regression docs: NONE"
    )
    print(f"best holdout:   {best_holdout}")
    print(f"review:         {experiment.describe_review(review_summary)}")
    print(f"elapsed:        {elapsed:.1f}s")
    print(f"state:          {state_path}")
    print(f"experiments:    {c.experiment_log}")
    print("=" * 72)
    return summary


def run_continuously(c, tel):
    """Run until a human stops it, rather than until a budget is spent.

    The difference from run_worker is one thing, and it is the whole thing.
    run_worker is given a cycle budget and returns when it is spent, which is
    correct for a run a person is watching and wrong for an agent, because an
    agent that stops when a counter runs out was never autonomous; it was a
    longer pipeline. This keeps starting new runs until a stopping rule fires or
    a signal arrives, and the rules that remain are the ones that mean something
    has actually gone wrong.

    Between runs it asks autonomy what to do rather than assuming the answer.
    That is the part that makes it a decision loop: the reason each pass was
    started is recorded, so an operator can read why the agent did what it did
    instead of inferring it from timestamps.

    The wall-clock limit is still honoured, because an agent that cannot be
    stopped is not autonomous either. Ctrl+C sets the stop flag, the next pass
    sees it and returns, and the lock is released on the way out.
    """
    import autonomy

    started = time.time()
    max_passes = int(getattr(c, "autonomy_max_passes", 0)) or None
    # The wall clock for the whole run, which is not the same as the wall clock
    # for one pass. Zero means no limit, which is the default for this mode: a
    # person who asked for continuous learning asked for it to keep going, and a
    # limit they did not set is a limit they did not ask for. Ctrl+C still stops
    # it, and a run set to a budget is still honoured.
    total_limit = float(getattr(c, "continuous_max_seconds", 0.0) or 0.0)
    passes = 0
    print("=" * 72)
    print("JUPITER CONTINUOUS LEARNING")
    print("=" * 72)
    print("This run continues until you stop it with Ctrl+C.")
    print(
        f"Pass limit:       {c.worker_max_seconds:.0f}s per pass; a pass that "
        f"exceeds it stops rather than being cut off mid-cycle."
    )
    if total_limit:
        print(f"Run limit:        {total_limit:.0f}s total, then it stops cleanly.")
    else:
        print("Run limit:        none. It runs until you stop it.")
    if max_passes:
        print(f"Pass count limit: {max_passes} (autonomy_max_passes)")
    print("=" * 72)

    while not _stop_requested:
        if total_limit and time.time() - started >= total_limit:
            print(
                f"\n[continuous] reached the {total_limit:.0f}s continuous run "
                "budget; stopping cleanly. Raise JUPITER_CONTINUOUS_MAX_SECONDS to "
                "run longer, or set it to 0 for no limit."
            )
            break
        passes += 1
        pieces = load_candidates(c, tel)

        # The decision is made first and the work follows it. This used to call
        # run_worker unconditionally after deciding, so the decision recorded a
        # reason and changed nothing: every action led to the same cycles. The
        # runner is passed in now, so autonomy.act performs the action it chose
        # and reports what actually happened.
        def runner(cycles):
            return run_worker(c, tel, cycles)

        decision, outcome, observation = autonomy.step(
            c,
            candidates=[
                str(piece.path if isinstance(piece, Candidate) else piece[0])
                for piece in (pieces or [])
            ],
            tel=tel,
            runner=runner,
        )
        print(
            f"\n[continuous] pass {passes}: {decision.action.upper()} -- "
            f"{decision.reason}"
        )
        # What actually happened, said plainly. The line above is the decision and
        # its reason; it used to be the whole report, so a pass that chose EXPAND
        # and grew nothing printed "so the model is grown" and looked like it had.
        # Three outcomes are distinguished because they need different responses:
        # it grew, it was refused, or there was nothing waiting to grow.
        if outcome.get("nothing_pending"):
            print("  nothing happened: no growth was waiting to be approved.")
        elif outcome.get("denied"):
            print("  refused: the control gate declined this growth.")
        elif decision.action == autonomy.EXPAND and outcome.get("applied"):
            print("  the model was grown. It takes effect on the next run.")

        # A growth waiting on a human is offered to the human. This loop prints
        # autonomy's reason and moves on, and that reason ends with the name of an
        # environment variable the person has to set in another window before the
        # process starts. run_worker() asks; this did not, so the one mode that is
        # meant to run unattended is the one mode where the question could never be
        # reached. Asked here, at the point the reason is printed, because that is
        # where the person is already reading that something is waiting.
        #
        # Both questions are reachable from here: whether to grant the domain, and
        # whether to apply the growth once it is granted. A pass that recorded a
        # shortage and did nothing about it has told the operator everything except
        # the one thing they could do.
        if "not granted" in str(decision.reason):
            if offer_grant(
                c,
                "expansion",
                "a source is short of capacity and a growth is waiting",
                tel=tel,
            ):
                # The grant now exists for this process, so the next pass can act
                # on it. This pass is not retried: the shortage was real and the
                # cycles it would have run have already run.
                print(
                    "  the growth can now be proposed. It will be offered at the end"
                    " of the next pass, and applying it still needs your approval."
                )

        pending = expansion.pending(c).get("capacity_growth")
        if isinstance(pending, dict) and not pending.get("approved"):
            prompt_for_growth(c, None, tel=tel)

        tel.write(
            "continuous_pass",
            pass_number=passes,
            action=decision.action,
            performed=bool(outcome.get("performed")),
            reason=decision.reason,
            candidates=observation.get("candidate_count", 0),
        )

        if outcome.get("stop"):
            print("[continuous] the loop decided to stop.")
            break

        if decision.action == autonomy.REST:
            # REST waits. It runs no cycle, because a loop with nothing to learn
            # that learns anyway is not resting.
            wait = float(outcome.get("sleep_seconds", 0.0))
            print(f"[continuous] resting {wait:.0f}s before looking again.")
            time.sleep(wait)
            continue

        if not outcome.get("performed"):
            # EXPAND is the usual case here: it was attempted and the gate
            # refused it. That is a normal outcome, and the pass already recorded
            # why. The question about the grant was asked above, where the reason
            # mentioning it is printed, so nothing is lost by leaving the pass here.
            continue

    # Asked once more on the way out, and only if there is actually something to
    # decide. A growth can be proposed by a pass that ran cycles rather than by the
    # expand branch, and a loop that ends with a proposal unasked is the same
    # unreachable control as one that never offered it at all. The guard is what
    # keeps it from becoming a loop that asks on every run of a project with no
    # growth proposed, which is its own kind of noise.
    pending = expansion.pending(c).get("capacity_growth")
    if isinstance(pending, dict) and not pending.get("approved"):
        offer_grant(
            c,
            "expansion",
            "a growth is waiting for a decision",
            tel=tel,
        )
        prompt_for_growth(c, None, tel=tel)

    summary = autonomy.summary(c)
    print("=" * 72)
    print("CONTINUOUS LEARNING STOPPED")
    print(f"passes:           {passes}")
    print(f"decisions:        {summary['actions_taken']}")
    print(f"elapsed:          {time.time() - started:.1f}s")
    print(f"state:            {summary['state']}")
    print("=" * 72)
    tel.write("continuous_stop", passes=passes, elapsed=time.time() - started)
    return summary


def detach(c, forward_args):
    """Start the worker as a detached background process and return its pid."""
    log_path = Path(c.worker_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve())] + forward_args
    options = {}
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        options["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"--- worker detached at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
        handle.flush()
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        **options,
    )
    print(f"worker detached as pid {process.pid}")
    print(f"log:  {log_path}")
    print(f"state: {Path(c.worker_state)}")
    return process.pid


def main():
    parser = argparse.ArgumentParser(
        description="Run Jupiter's proven learning cycle repeatedly "
        "until a stopping rule fires.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run exactly one cycle and stop",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="start the worker in the background and return immediately",
    )
    parser.add_argument(
        "--learning-mode",
        choices=("plasticity", "gradient"),
        default=None,
        help="choose direct gain/loss plasticity or the legacy gradient learner; default is plasticity",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "run until a human stops it, rather than until the cycle budget is "
            "spent. The stopping rules still apply; the cycle budget does not, "
            "because a budget is a thing a person sets for a run they intend to "
            "watch. This is continuous operation: the process "
            "starts once and keeps deciding what to do next."
        ),
    )
    args = parser.parse_args()

    c = load_config()
    if args.learning_mode:
        os.environ["JUPITER_LEARNING_MODE"] = args.learning_mode
    if args.detach:
        forward = []
        if args.once:
            forward.append("--once")
        elif args.continuous:
            forward.append("--continuous")
        if args.learning_mode:
            forward.extend(["--learning-mode", args.learning_mode])
        detach(c, forward)
        return

    Path(c.out_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(c.seed)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    tel = Telemetry(c.telemetry_file)

    lock_path = lock_module.acquire(c.worker_lock, c)
    print(f"[worker] lock held at {lock_path}")
    try:
        if args.continuous:
            run_continuously(c, tel)
        else:
            run_worker(c, tel, 1 if args.once else c.worker_max_cycles)
    finally:
        lock_module.release(c.worker_lock)
        print("[worker] lock released")


if __name__ == "__main__":
    main()
