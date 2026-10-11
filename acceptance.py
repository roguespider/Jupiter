"""Update acceptance (Architecture 20).

A proposed update is not automatically accepted. The controller classifies it
as one of:

    ACCEPT   the update helped and did not damage earlier learning
    REJECT   it did not help, or it damaged earlier learning
    DEFER    the change was inside the dead band, so neither verdict is honest
    REVIEW   it cannot be judged, so a human or a later check must look
    ROLLBACK the model destabilized and must be restored

The distinction between REJECT and DEFER matters: collapsing them would make
Jupiter look decisive about updates that simply did not move.
"""

ACCEPT = "ACCEPT"
REJECT = "REJECT"
DEFER = "DEFER"
REVIEW = "REVIEW"
ROLLBACK = "ROLLBACK"

VERDICTS = (ACCEPT, REJECT, DEFER, REVIEW, ROLLBACK)


def classify(
    c,
    improvement,
    regression_delta,
    regression_available,
    confidence,
    observed,
    conversation_change=None,
    conversation_rate=None,
    smoothed_improvement=None,
):
    """Classify one completed learning cycle.

    improvement           holdout gain on the document just learned
    regression_delta      holdout gain on the reserved document, or None
    regression_available  False when no yardstick exists to judge with
    confidence            current confidence for this source, 0.0 - 1.0
    observed              how many times this source has been judged before
    conversation_change   change in conversation pass rate, or None
        conversation_rate    the absolute conversation pass rate, or None
        smoothed_improvement mean improvement over recent cycles, or None. Used
            in place of the single-cycle figure when there is enough history for
            it to mean something; see below.

    Order matters and is deliberate. Measured forgetting is checked before the
    confidence floor, because the two answer different questions. A
    regression means the update was measured and it damaged earlier learning,
    which is a judgement Jupiter can make: Architecture 20 lists regression
    among the reasons to reject. A low confidence floor means the source has not
    yet earned trust, which is not a judgement about this update at all, it is a
    reason to look again. Reporting the second as REVIEW when the first also
    holds would hide a measured regression behind a request for more
    observation, and REVIEW does not roll the update back.

    The conversation set is checked at the same level as forgetting, and before
    the confidence floor, because it is the same kind of evidence: something
    measured that got worse. Loss on text says nothing about whether a model can
    hold a turn, so a cycle can improve every number this function was given and
    still damage the one ability a conversation exists for. It is a separate
    measurement rather than a term folded into the loss, because a blend would
    let a large loss improvement hide a conversation regression, which is the
    failure this check exists to prevent.

    The number of observations before the confidence floor applies is
    c.confidence_grace_observations, configuration rather than a constant here,
    because it is a policy decision that has been re-tuned against measured
    behaviour and belongs with the other acceptance thresholds. It was two, and
    at two the floor was circular: confidence is the share of a source's cycles
    that were accepted, so a source held at REVIEW never accrued the accepts it
    needed to clear the floor.
    """
    if not regression_available:
        # Without a fixed yardstick, an improvement on the current document
        # cannot be separated from forgetting. That is REVIEW, not ACCEPT.
        return (
            REVIEW,
            "no regression yardstick available, so forgetting cannot be ruled out",
        )
    if regression_delta is not None and regression_delta < -c.regression_tolerance:
        return (
            REJECT,
            f"reserved document regressed by {-regression_delta:.6f}, "
            f"beyond the {c.regression_tolerance:.6f} tolerance",
        )
    if (
        conversation_change is not None
        and conversation_change < -c.conversation_tolerance
    ):
        return (
            REJECT,
            f"conversation pass rate fell by {-conversation_change:.4f}, beyond "
            f"the {c.conversation_tolerance:.4f} tolerance: the update damaged "
            f"an ability the loss measurements cannot see",
        )
    if conversation_rate is not None and conversation_rate < c.conversation_min_rate:
        # The change check above can only fire once there is something to lose.
        # While the pass rate is zero, every run scores zero, every change is
        # zero, and nothing is ever rejected for conversation - so a model that
        # cannot hold a turn at all is accepted indefinitely on the strength of
        # its holdout loss. That is precisely the failure this gate was written
        # to catch: every loss number improving while the ability a
        # conversation exists for is absent.
        #
        # This is a floor on the absolute rate, not a change, so it applies from
        # the first measurement. It does not roll the update back, because an
        # update that improved learning is still an improvement; it marks the
        # cycle as unjudgeable, which is what REVIEW means.
        return (
            REVIEW,
            f"conversation pass rate is {conversation_rate:.4f}, below the "
            f"{c.conversation_min_rate:.4f} floor: the update cannot be judged "
            f"until the model can hold a turn at all",
        )
    if observed >= c.confidence_grace_observations and confidence < c.confidence_min:
        return (
            REVIEW,
            f"confidence {confidence:.3f} is below the review floor "
            f"{c.confidence_min:.3f} after {observed} observations",
        )
    # The decision below is made on a smoothed improvement rather than on this
    # cycle's own figure, because a single cycle cannot be read.
    #
    # Measured over 3,315 recorded cycles, the per-cycle change in holdout loss
    # has a mean of -0.377 and a standard deviation of 1.010: the noise is 2.7
    # times the signal. The defer band is 1e-4, which is four orders of magnitude
    # below that noise. Comparing a single cycle against a band that small
    # decides nothing at all: whether a cycle is accepted is close to a coin flip
    # on which document happened to be drawn, and that is why roughly half of every
    # run has been rejected for reasons that had nothing to do with the update.
    #
    # So the figure judged is the mean over recent cycles, which has a standard
    # error of sigma/sqrt(n) and falls below the noise once a handful have
    # accumulated. Until then the single-cycle figure is used and the verdict says
    # so, because pretending a three-cycle mean is a measurement would be the
    # same error wearing a smoother number.
    judged = improvement if smoothed_improvement is None else smoothed_improvement
    # The band depends on what is being judged. A single cycle is compared
    # against the per-cycle band, which is small enough that it does not fire on
    # noise only because noise that large is rarer than the window length. A mean
    # of several cycles carries less noise, so it is compared against a band the
    # size of that noise rather than against a precision the measurement cannot
    # deliver.
    band = (
        c.defer_band
        if smoothed_improvement is None
        else float(getattr(c, "smoothed_defer_band", c.defer_band))
    )
    basis = (
        "this cycle"
        if smoothed_improvement is None
        else f"the mean of {c.acceptance_smoothing} recent cycles"
    )

    if judged >= band:
        return (
            ACCEPT,
            f"holdout improved by {judged:.6f} across {basis} with no measured "
            f"forgetting",
        )
    if judged <= -band:
        return (
            REJECT,
            f"holdout worsened by {-judged:.6f} across {basis}",
        )
    return (
        DEFER,
        f"holdout moved by {judged:+.6f} across {basis}, inside the "
        f"{band:.6f} dead band",
    )
