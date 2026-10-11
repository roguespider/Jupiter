"""Tests for dynamic expansion, the part of it that can be built (Architecture 22, 23).

Architecture 23 specifies a loop: difficulty, repeated failure, analyse the
cause, ask whether capacity is the problem, propose, experiment, evaluate, keep
or roll back. The first four steps already worked. These cover the rest, and
they cover mostly the refusals, because the properties that matter here are the
ones that stop an expansion happening for the wrong reason or being impossible
to undo.
"""

import unittest
from pathlib import Path

import control
import expansion
import torch
import train
from model import Jupiter, ModelConfig

import config
from tests.support import TempDirTestCase, make_config

CAPACITY = "insufficient model capacity"


def measurement(source, cause, index=0.9, band="HIGH"):
    return {
        "source": source,
        "likely_cause": cause,
        "index": index,
        "band": band,
    }


class ProposalTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(
            expansion_state=str(self.workdir / "expansion.json"),
            concept_terms_per_cycle=5,
        )

    def test_a_source_short_of_capacity_gets_a_proposal(self):
        proposals = expansion.propose(
            self.config, [measurement("doc-a.txt", CAPACITY)], 1
        )
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["current_limit"], 5)
        self.assertEqual(proposals[0]["proposed_limit"], 10)

    def test_a_source_failing_for_another_reason_gets_none(self):
        # Architecture 22: expansion must not occur simply because loss is high.
        # Naming more terms in a document that lacked data, or that two sources
        # disagree about, does not address either.
        for cause in (
            "inadequate context",
            "insufficient data",
            "conflicting knowledge",
            "poor representation",
            "conflicting knowledge",
        ):
            self.assertEqual(
                expansion.propose(self.config, [measurement("doc-a.txt", cause)], 1),
                [],
                cause,
            )

    def test_no_proposals_is_the_normal_case(self):
        self.assertEqual(expansion.propose(self.config, [], 1), [])
        self.assertEqual(expansion.propose(self.config, None, 1), [])

    def test_proposals_are_recorded_without_being_applied(self):
        proposals = expansion.propose(
            self.config, [measurement("doc-a.txt", CAPACITY)], 1
        )
        expansion.record_proposals(self.config, proposals, 1)
        summary = expansion.summary(self.config)
        self.assertEqual(summary["proposals_recorded"], 1)
        # Recording is a measurement. It must not have changed any limit.
        self.assertEqual(expansion.active_limit(self.config, "doc-a.txt"), 5)


class GrantTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(
            expansion_state=str(self.workdir / "expansion.json"),
            concept_terms_per_cycle=5,
        )

    def test_a_grant_is_refused_without_human_authority(self):
        # Architecture 38 puts model expansion behind human control, and
        # control.py denies by default. The refusal is the point.
        with self.assertRaises(control.ControlDenied):
            expansion.grant(self.config, "doc-a.txt", holdout_loss=4.0)
        self.assertEqual(expansion.active_limit(self.config, "doc-a.txt"), 5)

    def test_an_authorised_grant_raises_the_limit(self):
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)
        expansion.grant(self.config, "doc-a.txt", holdout_loss=4.0)
        self.assertEqual(expansion.active_limit(self.config, "doc-a.txt"), 10)

    def test_granting_twice_does_not_stack(self):
        # A second grant must not double the expansion, or a repeated proposal
        # would grow the limit without limit.
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)
        expansion.grant(self.config, "doc-a.txt", holdout_loss=4.0)
        expansion.grant(self.config, "doc-a.txt", holdout_loss=4.0)
        self.assertEqual(expansion.active_limit(self.config, "doc-a.txt"), 10)

    def test_one_source_expanding_does_not_move_another(self):
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)
        expansion.grant(self.config, "doc-a.txt", holdout_loss=4.0)
        self.assertEqual(expansion.active_limit(self.config, "doc-b.txt"), 5)

    def test_an_ungranted_source_keeps_the_configured_limit(self):
        self.assertEqual(expansion.active_limit(self.config, "never-seen.txt"), 5)


class ReversibilityTests(TempDirTestCase):
    """Architecture 23: a proposal must be measurable and reversible."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            expansion_state=str(self.workdir / "expansion.json"),
            concept_terms_per_cycle=5,
        )
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)
        expansion.grant(self.config, "doc-a.txt", holdout_loss=4.0)

    def test_rollback_returns_the_limit(self):
        self.assertEqual(expansion.active_limit(self.config, "doc-a.txt"), 10)
        expansion.rollback(self.config, "doc-a.txt", holdout_loss=5.5)
        self.assertEqual(expansion.active_limit(self.config, "doc-a.txt"), 5)

    def test_rollback_is_recorded_with_what_it_measured(self):
        expansion.rollback(self.config, "doc-a.txt", holdout_loss=5.5)
        state = expansion.load(self.config)
        record = state["doc-a.txt"]
        self.assertFalse(record["active"])
        self.assertEqual(record["holdout_at_rollback"], 5.5)
        self.assertIn("rolled_back_at", record)

    def test_rolling_back_an_unexpanded_source_is_not_an_error(self):
        self.assertIsNone(expansion.rollback(self.config, "doc-z.txt"))

    def test_the_effect_of_a_grant_is_comparable(self):
        result = expansion.effect(self.config, "doc-a.txt", 2.5)
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["change"], -1.5, places=5)
        self.assertTrue(result["helped"])

    def test_a_worsened_source_does_not_look_helped(self):
        result = expansion.effect(self.config, "doc-a.txt", 5.5)
        self.assertFalse(result["helped"])
        self.assertAlmostEqual(result["change"], 1.5, places=5)

    def test_an_effect_with_no_baseline_says_so(self):
        # Without a loss recorded at the grant, a later comparison would be an
        # opinion rather than a measurement.
        other = make_config(
            expansion_state=str(self.workdir / "other.json"),
            concept_terms_per_cycle=5,
        )
        result = expansion.effect(other, "doc-a.txt", 2.5)
        self.assertFalse(result["comparable"])
        self.assertIn("note", result)

    def test_rollback_keeps_the_knowledge_the_expansion_added(self):
        # Architecture 9 wants experimental results preserved. An expansion that
        # learned something is not a reason to delete what it learned.
        expansion.rollback(self.config, "doc-a.txt", holdout_loss=5.5)
        state = expansion.load(self.config)
        self.assertIn("doc-a.txt", state)
        self.assertIn("holdout_at_grant", state["doc-a.txt"])


class CauseAgreementTests(unittest.TestCase):
    def test_expansion_and_frustration_name_the_same_cause(self):
        # Two modules deciding what "short of capacity" means is how the
        # proposal and the report end up disagreeing.
        import frustration

        self.assertEqual(expansion.capacity_cause(), CAPACITY)
        self.assertIn(CAPACITY, frustration.CAUSES)

    def test_capacity_shortages_filters_to_the_one_cause(self):
        rows = [
            measurement("a.txt", CAPACITY),
            measurement("b.txt", "inadequate context"),
        ]
        self.assertEqual(
            [r["source"] for r in expansion.capacity_shortages(rows)], ["a.txt"]
        )
        self.assertEqual(expansion.capacity_shortages([]), [])


class CapacityGrowthTests(TempDirTestCase):
    """Growing the model itself, rather than how much Jupiter extracts.

    Memory expansion above changes no parameters. This half adds a layer, which
    only became possible once the shape guard learned to tell a shape that got
    bigger from a shape that does not fit. The tests below cover the whole path
    a growth travels, because a proposal that is recorded and never applied is
    the failure mode that makes an expansion system look finished and do
    nothing.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(
            expansion_state=str(self.workdir / "expansion.json"),
            out_dir=str(self.workdir / "out"),
            device="cpu",
            n_layer=2,
        )

    def grant(self, *domains):
        original = control.granted_domains
        control.granted_domains = lambda environ=None: set(domains)
        self.addCleanup(setattr, control, "granted_domains", original)

    def short(self, source="doc-a.txt"):
        return [measurement(source, CAPACITY)]

    def architecture(self):
        return Path(self.config.out_dir) / "Jupiter.architecture.json"

    # -- proposing ---------------------------------------------------------

    def test_a_capacity_shortage_proposes_one_more_layer(self):
        proposal = expansion.propose_growth(self.config, self.short(), cycle=1)
        self.assertEqual(proposal["from_layers"], 2)
        self.assertEqual(proposal["to_layers"], 3)
        self.assertFalse(proposal["approved"], "a proposal is never self-approving")

    def test_a_source_failing_for_another_reason_gets_no_growth(self):
        # More parameters do not fix a source that is failing for want of data,
        # and proposing them anyway would grow the model for the wrong reason.
        rows = [measurement("doc-a.txt", "inadequate context")]
        self.assertIsNone(expansion.propose_growth(self.config, rows, cycle=1))

    def test_no_shortage_is_the_normal_case(self):
        self.assertIsNone(expansion.propose_growth(self.config, [], cycle=1))

    def test_a_model_at_the_ceiling_is_not_proposed_again(self):
        self.config.n_layer = expansion.MAX_LAYERS
        self.assertIsNone(expansion.propose_growth(self.config, self.short(), cycle=1))

    # -- authorising -------------------------------------------------------

    def test_growing_is_refused_without_human_authority(self):
        expansion.propose_growth(self.config, self.short(), cycle=1)
        with self.assertRaises(control.ControlDenied):
            expansion.approve_growth(self.config)
        self.assertIsNone(
            expansion.pending_shape(self.config),
            "a refused growth must not become a shape change",
        )

    def test_an_authorised_growth_becomes_pending(self):
        self.grant("expansion")
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        pending = expansion.pending_shape(self.config)
        # A growth is now two things, not one: a layer and a widening. The layer
        # is depth, the width is the neurons, and Architecture 22 lists them as
        # separate mechanisms.
        self.assertEqual(pending["n_layer"], 3)
        self.assertEqual(pending["n_embd"], self.config.n_embd + expansion.WIDTH_STEP)

    def test_a_widened_growth_adds_neurons_and_not_only_depth(self):
        # The point of the width step. Depth adds capability; width adds neurons,
        # and n_embd was previously frozen for the whole life of the project.
        self.grant("expansion")
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        self.assertGreater(
            expansion.pending_shape(self.config)["n_embd"],
            self.config.n_embd,
            "an approved growth must be able to widen the model",
        )

    def test_an_unapproved_proposal_is_never_pending(self):
        expansion.propose_growth(self.config, self.short(), cycle=1)
        self.assertIsNone(expansion.pending_shape(self.config))

    # -- applying ----------------------------------------------------------

    def build(self):
        model, architecture_path = train.build_model(self.config, vocab_size=256)
        config.save_architecture(
            self.config,
            architecture_path,
            256,
            shape=model.config.to_dict(),
        )
        return model, architecture_path

    def test_an_approved_growth_makes_the_next_model_larger(self):
        self.grant("expansion", "architecture")
        config.save_architecture(self.config, self.architecture(), 256)
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        model, _ = self.build()
        self.assertEqual(model.config.n_layer, 3)

    def test_the_record_follows_the_grown_model_rather_than_the_config(self):
        # The defect this guards against: the seed config still says two layers
        # after the model has grown to three, so recording the config would write
        # down a shape the weights do not have. The next run would then read that
        # record, build the smaller model, and archive the checkpoint it had just
        # produced, throwing away the growth and everything learned into it.
        self.grant("expansion", "architecture")
        config.save_architecture(self.config, self.architecture(), 256)
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        model, _ = self.build()
        self.assertEqual(self.config.n_layer, 2, "the config is not rewritten")
        recorded, _version = config.read_architecture(self.architecture())
        self.assertEqual(
            recorded["n_layer"],
            3,
            "the record must describe the model that was actually built",
        )

    def test_growing_twice_does_not_grow_again_or_crash(self):
        # The second run after an approval reads a record that already describes
        # the grown model. Applying the same growth again would be refused for
        # shrinking nothing, so the build has to recognise it as already applied.
        self.grant("expansion", "architecture")
        config.save_architecture(self.config, self.architecture(), 256)
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        self.build()
        again, _ = self.build()
        self.assertEqual(again.config.n_layer, 3, "a kept growth is not re-applied")

    def test_growths_accumulate_rather_than_replacing_each_other(self):
        # Reading the saved record before growing is what makes a second growth
        # cumulative. Growing from the configuration instead would build five
        # layers every time a second layer was approved.
        #
        # The record has to be written from the model, which is what the second
        # build does, or the next run would read two layers again and the
        # approved growth would oscillate between three and never accumulate.
        self.grant("expansion", "architecture")
        config.save_architecture(self.config, self.architecture(), 256)
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        first, _ = self.build()
        self.assertEqual(first.config.n_layer, 3)
        expansion.propose_growth(self.config, self.short("doc-b.txt"), cycle=2)
        expansion.approve_growth(self.config)
        model, _ = self.build()
        self.assertEqual(model.config.n_layer, 4)

    def test_a_grown_checkpoint_is_loaded_into_the_grown_model(self):
        # The point of the whole path. The model is larger than the checkpoint,
        # which used to be treated as an incompatible shape and archived. Now
        # the learned weights are carried over and only what is genuinely new is
        # fresh, so a growth does not cost what it learned.
        self.grant("expansion", "architecture")
        self.config.n_embd = 32
        self.config.n_head = 1
        self.config.block_size = 64
        config.save_architecture(self.config, self.architecture(), 256)
        small = Jupiter(
            ModelConfig(block_size=64, vocab_size=256, n_layer=2, n_head=1, n_embd=32)
        ).to("cpu")
        path = train.save_checkpoint(small, self.config, cycle=1)
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        grown, architecture_path = self.build()
        self.assertTrue(
            train.attach_checkpoint(grown, [path], architecture_path),
            "a smaller checkpoint must grow into the model, not be archived",
        )
        # Each of Q, K and V must still hold the weight it was trained with, in
        # its own group of the fused projection. Checking only the first block
        # would pass even if K and V had been dropped, and the whole point of
        # the block-wise widen is that they are not.
        groups = 3
        new_embd = grown.config.n_embd
        for index in range(2):
            learned = small.blocks[index].attn.qkv.weight
            for group in range(groups):
                start = group * new_embd
                self.assertTrue(
                    torch.equal(
                        grown.blocks[index].attn.qkv.weight[start : start + 32, 0:32],
                        learned[group * 32 : (group + 1) * 32],
                    ),
                    f"block {index} group {group} must keep its learned weights",
                )
        self.assertTrue(path.exists(), "a grown checkpoint must not be archived")

    # -- judging -----------------------------------------------------------

    def test_a_growth_that_lowered_the_loss_is_kept(self):
        self.grant("expansion")
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        review = expansion.resolve_growth(self.config, 3.0)
        self.assertTrue(review["kept"])
        self.assertEqual(review["change"], -1.0)

    def test_a_growth_that_raised_the_loss_is_not_kept(self):
        # Architecture 22 asks for expansion to be measurable, and a larger model
        # that scored worse is a regression rather than an improvement. The
        # verdict is recorded either way.
        self.grant("expansion")
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        review = expansion.resolve_growth(self.config, 5.0)
        self.assertFalse(review["kept"])
        self.assertEqual(review["change"], 1.0)

    def test_a_growth_with_no_baseline_says_it_is_not_comparable(self):
        # Without the loss at approval there is nothing to compare against, and
        # inventing a verdict from a single number would be a measurement of
        # nothing.
        self.grant("expansion")
        expansion.propose_growth(self.config, self.short(), cycle=1)
        expansion.approve_growth(self.config)
        review = expansion.resolve_growth(self.config, 3.0)
        self.assertFalse(review["comparable"])
        self.assertNotIn("kept", review)

    def test_judging_a_growth_that_was_never_proposed_is_not_an_error(self):
        self.assertIsNone(expansion.resolve_growth(self.config, 3.0))


if __name__ == "__main__":
    unittest.main()


class GrowthProposalGateTests(TempDirTestCase):
    """Whether Architecture 23's loop actually gets a chance to run.

    It did not, and the reason was a counter. propose_growth was guarded by
    `cycle == 1`, and the cycle counter is read from disk at startup, so it began
    at whatever the previous run had reached. Past the first run the condition
    was never true: 78 sources were blamed on insufficient model capacity and not
    one growth was ever proposed. That is the shape of a feature that looks
    present and never fires, and no test caught it because every test of
    propose_growth called it directly rather than through the worker.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(
            expansion_state=str(self.workdir / "expansion.json"),
            out_dir=str(self.workdir / "out"),
            n_layer=5,
        )
        self.short = [
            {
                "source": "doc-a.txt",
                "likely_cause": expansion.CAPACITY_CAUSE,
                "index": 0.9,
            }
        ]
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)

    def test_a_capacity_shortage_proposes_regardless_of_the_cycle_counter(self):
        # The regression itself. The proposal is offered on every pass of the
        # loop, and it is the state of the expansion, not the cycle number, that
        # decides whether one is warranted.
        self.assertTrue(expansion.should_propose_growth(self.config, self.short))
        expansion.propose_growth(self.config, self.short, 311)
        self.assertIn("capacity_growth", expansion.load(self.config))

    def test_no_capacity_shortage_means_no_proposal(self):
        other = [
            {"source": "x", "likely_cause": "poor representation", "index": 0.9}
        ]
        self.assertFalse(expansion.should_propose_growth(self.config, other))

    def test_a_proposal_waiting_on_a_human_is_not_replaced(self):
        expansion.propose_growth(self.config, self.short, 1)
        self.assertFalse(expansion.should_propose_growth(self.config, self.short))

    def test_an_approved_growth_on_trial_is_not_replaced(self):
        # Re-proposing over it would discard the baseline it is judged against.
        expansion.propose_growth(self.config, self.short, 1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        self.assertFalse(expansion.should_propose_growth(self.config, self.short))

    def test_a_growth_kept_on_this_shape_is_not_repeated(self):
        expansion.propose_growth(self.config, self.short, 1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        expansion.resolve_growth(self.config, 3.0)
        self.assertTrue(expansion.load(self.config)["capacity_growth"]["kept"])
        self.assertFalse(expansion.should_propose_growth(self.config, self.short))

    def test_a_verdict_about_a_different_shape_does_not_bind_this_one(self):
        # A rejection is evidence about a particular model. Once the model has
        # been grown, a verdict about the shape it used to have is not evidence
        # about this one, and honouring it stops the loop proposing at all.
        expansion.propose_growth(self.config, self.short, 1)
        expansion.approve_growth(self.config, holdout_loss=4.0)
        expansion.resolve_growth(self.config, 5.0)
        self.assertFalse(expansion.load(self.config)["capacity_growth"]["kept"])
        self.config.n_layer = 9
        self.assertTrue(
            expansion.should_propose_growth(self.config, self.short),
            "a model that has since grown needs a new verdict, not an old one",
        )
