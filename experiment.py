"""Experiment records (Architecture 31) and the learning review step (37).

Every meaningful learning cycle should leave a record detailed enough that the
result does not depend on anyone's memory. The record follows the field list in
Architecture 31.

This is also the "Store experience" and "Review" steps of the continuous
operation loop in Architecture 37: a rejected update is written down with the
same care as an accepted one, so a failed experiment is information rather than
something that disappears.
"""

import hashlib
import json
import time
from pathlib import Path

import acceptance

from config import (
    ARCHITECTURE_VERSION,
    LEARNING_SYSTEM_VERSION,
    MEMORY_SCHEMA_VERSION,
    architecture_dict,
)


def _next_id():
    """Return a timestamp-based id that is also readable when sorted."""
    return time.strftime("exp-%Y%m%dT%H%M%S")


def tensor_version(c):
    """A fingerprint of the tensor layout an experiment ran under.

    Architecture 30 asks a saved instance to be able to determine the tensor
    version, and Architecture 43 asks that the versions be tracked separately
    because they do not advance together. The layout is fully determined by the
    shape fields, so hashing those gives a real version rather than a label.

    Before this, nothing recorded one at all, and model_version was a cycle
    count, so two experiments with different tensor layouts were
    indistinguishable unless someone went and read the checkpoint.
    """
    payload = json.dumps(architecture_dict(c), sort_keys=True)
    return "tensors-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def write_experiment(
    c,
    cycle,
    source,
    cycle_result,
    verdict,
    reason,
    regression_before,
    regression_after,
    confidence,
    model_cycle,
    resources,
):
    """Append one experiment record and return it."""
    path = Path(c.experiment_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment_id": _next_id(),
        "timestamp": time.time(),
        "learning_cycle": cycle,
        "model_version": f"arch-{ARCHITECTURE_VERSION}-cycle-{model_cycle}",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_record": "Jupiter.architecture.json",
        "tensor_version": tensor_version(c),
        # Architecture 43 names six version domains and says they do not advance
        # together. These two were the ones with nowhere to be read from, so a
        # knowledge store written by a different learning system, or holding a
        # different shape, was indistinguishable from one this run produced.
        "learning_system_version": LEARNING_SYSTEM_VERSION,
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "input": str(source),
        # Which update rule produced this cycle. It was a fixed string naming
        # gradient clipping, so a cycle that changed weights by reward-modulated
        # Hebbian plasticity was recorded as a gradient cycle. The two are
        # different learning systems and the log is how a run is compared against
        # another, so mislabelled rows make plasticity and gradient look like the
        # same method and every comparison between them meaningless.
        "learning_method": cycle_result.get("learning_method")
        or "bounded micro-updates with gradient clipping",
        "hyperparameters": {
            "learning_rate": cycle_result.get("final_learning_rate"),
            "micro_steps": cycle_result.get("micro_steps"),
            "gradient_clip": cycle_result.get("gradient_clip"),
            "gradient_stop": cycle_result.get("gradient_stop"),
        },
        "before_metrics": {
            "holdout_loss": cycle_result.get("holdout_loss_before"),
            "holdout_perplexity": cycle_result.get("holdout_perplexity_before"),
            "holdout_accuracy": cycle_result.get("holdout_accuracy_before"),
            "regression_loss": regression_before,
            "confidence": confidence,
        },
        "update": {
            "successful_steps": cycle_result.get("successful_steps"),
            "gradient_recoveries": cycle_result.get("gradient_recoveries"),
            "magnitude": cycle_result.get("update_magnitude"),
        },
        "after_metrics": {
            "holdout_loss": cycle_result.get("holdout_loss_after"),
            "holdout_perplexity": cycle_result.get("holdout_perplexity_after"),
            "holdout_accuracy": cycle_result.get("holdout_accuracy_after"),
            "regression_loss": regression_after,
            "generalization_gap": cycle_result.get("generalization_gap"),
        },
        "decision": verdict,
        "reason": reason,
        "resources": resources,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_experiments(path, limit=None):
    """Read experiment records, newest last. Unreadable lines are skipped."""
    target = Path(path)
    if not target.exists():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    if limit is not None:
        records = records[-limit:]
    return records


def review(cycle_results):
    """Summarise a finished run for the Architecture 37 review step.

    This is deliberately a summary, not a decision. It states what happened so
    a person can judge the run without reading the raw log.

    The verdicts are named from acceptance.VERDICTS rather than written as
    literals. They were four bare strings here, and a fifth verdict, ROLLBACK,
    was silently absent from the summary: an Architecture 20 rollback is not an
    accepted update, and a review that could not see one would report a run as
    having done nothing when it had in fact undone something. Counting by
    iterating the tuple rather than by naming four cases means a new verdict
    cannot be forgotten here again.
    """
    counts = {verdict: 0 for verdict in acceptance.VERDICTS}
    for record in cycle_results:
        verdict = record.get("decision")
        if verdict in counts:
            counts[verdict] += 1
    regressing = [r for r in cycle_results if r.get("regressed")]
    return {
        "cycles": len(cycle_results),
        "accepted": counts[acceptance.ACCEPT],
        "rejected": counts[acceptance.REJECT],
        "deferred": counts[acceptance.DEFER],
        "flagged_for_review": counts[acceptance.REVIEW],
        "rolled_back": counts[acceptance.ROLLBACK],
        "cycles_with_regression": len(regressing),
        "distinct_sources": len({r["source"] for r in cycle_results}),
    }


def describe_review(summary):
    """Render the review summary for the log."""
    return (
        f"{summary['cycles']} cycles over {summary['distinct_sources']} sources: "
        f"{summary['accepted']} accepted, {summary['rejected']} rejected, "
        f"{summary['deferred']} deferred, "
        f"{summary['flagged_for_review']} flagged for review, "
        f"{summary['cycles_with_regression']} with measured regression"
    )
