"""Tests for the decision layer and continuous operation (Architecture 37).

Architecture 37 describes a loop: Observe, Receive, Evaluate, Learn, Measure,
Store experience, Rest, Review, Continue. Every stage existed somewhere in the
project and none of them were connected, so the tests below are mostly about
whether the connection makes decisions for the right reasons rather than
whether it runs. A loop that runs is easy; a loop that grows the model because
loss was high, or that stops when a counter runs out, is the failure Architecture
23 and Architecture 37 both warn about.
"""

import json
from pathlib import Path

import autonomy
import control
import expansion
import learning_worker
import torch
import train
from model import Jupiter, ModelConfig
from safetensors.torch import load_file

import config
from tests.support import TempDirTestCase, make_config


def write_experiment(path, before, after, decision="ACCEPT"):
    record = {
        "before_metrics": {"holdout_loss": before},
        "after_metrics": {"holdout_loss": after},
        "decision": decision,
        "input": "doc-a.txt",
        "learning_cycle": 1,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class DecisionTests(TempDirTestCase):
    """What the loop decides, and why, given a state."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            autonomy_state=str(self.workdir / "autonomy.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
            knowledge_store=str(self.workdir / "knowledge.json"),
            expansion_state=str(self.workdir / "expansion.json"),
            selection_state=str(self.workdir / "selection.json"),
        )

    def observation(self, **overrides):
        base = {
            "candidates": ["a.txt"],
            "candidate_count": 1,
            "sources_measured": 0,
            "capacity_shortages": 0,
            "cycles_recorded": 0,
            "cycles_accepted": 0,
            "mean_improvement": None,
            "improving": False,
            "sources_seen": 0,
            "granted": [],
            # How many consecutive passes have found nothing learnable. Read by
            # the policy to decide between resting and concluding, so it has to
            # be part of what an observation can say.
            "idle_passes": 0,
            # Whether the world has anything to offer. Read by the policy to
            # decide between resting on a dry corpus and consulting it, so it
            # belongs to what an observation can say alongside the rest.
            "external_configured": False,
            "consecutive_consults": 0,
            "query_terms": [],
        }
        base.update(overrides)
        return base

    def test_with_nothing_to_learn_the_loop_rests(self):
        decision = autonomy.decide(
            self.config, self.observation(candidates=[], candidate_count=0)
        )
        self.assertEqual(decision.action, autonomy.REST)
        self.assertIn("nothing to learn", decision.reason)

    def test_with_material_and_a_healthy_record_it_learns(self):
        decision = autonomy.decide(
            self.config,
            self.observation(cycles_recorded=5, mean_improvement=-0.4, improving=True),
        )
        self.assertEqual(decision.action, autonomy.LEARN)

    def test_a_capacity_shortage_proposes_growth_when_authorised(self):
        decision = autonomy.decide(
            self.config,
            self.observation(capacity_shortages=3, granted=["expansion"]),
        )
        self.assertEqual(decision.action, autonomy.EXPAND)
        self.assertEqual(decision.evidence["capacity_shortages"], 3)

    def test_a_capacity_shortage_keeps_learning_when_not_authorised(self):
        # This is the property that separates an agent from something that
        # either crashes on a denial or ignores the gate. The loop carries on
        # and says what it is waiting for.
        decision = autonomy.decide(
            self.config, self.observation(capacity_shortages=3, granted=[])
        )
        self.assertEqual(decision.action, autonomy.LEARN)
        self.assertIn(control.GRANT_VARIABLE, decision.reason)

    def test_cycles_that_are_not_improving_are_reviewed_before_more_are_made(self):
        # Architecture 23 asks for evaluation before expansion. A loop that only
        # ever adds capacity would make a failing model bigger.
        decision = autonomy.decide(
            self.config,
            self.observation(
                cycles_recorded=12,
                cycles_accepted=0,
                mean_improvement=0.6,
                improving=False,
                granted=["expansion"],
            ),
        )
        self.assertEqual(decision.action, autonomy.REVIEW)
        self.assertEqual(decision.evidence["mean_improvement"], 0.6)

    def test_an_unmeasured_growth_is_given_cycles_to_prove_itself(self):
        decision = autonomy.decide(
            self.config,
            self.observation(),
            growth_state={"approved": True, "reviewed_at": None},
        )
        self.assertEqual(decision.action, autonomy.LEARN)
        self.assertIn("not been measured", decision.reason)

    def test_a_growth_that_failed_is_not_repeated(self):
        decision = autonomy.decide(
            self.config,
            self.observation(),
            growth_state={"approved": True, "reviewed_at": 1.0, "kept": False},
        )
        self.assertEqual(decision.action, autonomy.LEARN)
        self.assertIn("did not help", decision.reason)

    def test_every_decision_carries_a_reason_a_person_can_read(self):
        # An agent that acts without recording why cannot be audited, and
        # Architecture 38 asks for human control over what it changes.
        for kwargs in (
            {"candidates": [], "candidate_count": 0},
            {"capacity_shortages": 1, "granted": ["expansion"]},
            {"capacity_shortages": 1, "granted": []},
            {"cycles_recorded": 3, "mean_improvement": 1.0, "improving": False},
        ):
            decision = autonomy.decide(self.config, self.observation(**kwargs))
            self.assertTrue(decision.reason.strip(), "a decision must say why")
            self.assertGreater(len(decision.reason.split()), 3)

    def test_an_unknown_action_is_refused_rather_than_accepted(self):
        # A Decision with an action the loop cannot carry out would be recorded
        # and then not acted on, which is a decision that lies.
        with self.assertRaises(ValueError):
            autonomy.Decision("invent-something", "because")

    def test_every_declared_action_is_one_the_loop_can_actually_choose(self):
        # The vocabulary listed five actions no branch could produce, which told
        # a caller checking a decision against it that taking them was possible.
        # The declared set is the reachable set, and this is what keeps the two
        # the same as branches are added.
        chosen = set()
        states = [
            # no material
            self.observation(candidates=[], candidate_count=0),
            # material, nothing wrong with the record
            self.observation(cycles_recorded=3, mean_improvement=-0.5, improving=True),
            # a growth waiting to be measured
            self.observation(),
            # a growth that failed
            self.observation(),
            # a shortage with authority
            self.observation(capacity_shortages=2, granted=["expansion"]),
            # a shortage without authority
            self.observation(capacity_shortages=2, granted=[]),
            # recent cycles not improving
            self.observation(cycles_recorded=9, mean_improvement=0.4, improving=False),
            # nothing learnable for long enough to conclude
            self.observation(candidates=[], candidate_count=0, idle_passes=99),
            # a dry corpus with a configured source and authority to read it
            self.observation(
                candidates=[],
                candidate_count=0,
                external_configured=True,
                granted=["external"],
            ),
        ]
        growth_states = [
            None,
            None,
            {"approved": True, "reviewed_at": None},
            {"approved": True, "reviewed_at": 1.0, "kept": False},
            None,
            None,
            None,
            None,
            None,
        ]
        for observation, growth in zip(states, growth_states):
            chosen.add(
                autonomy.decide(self.config, observation, growth_state=growth).action
            )
        self.assertEqual(
            chosen - set(autonomy.ACTIONS),
            set(),
            "a decision outside the declared vocabulary would be refused on write",
        )
        # Every declared action must be reachable. REST and STOP were declared
        # and produced by nothing until the loop was made to conclude as well as
        # to pause, which is the difference between waiting and giving up.
        for required in autonomy.ACTIONS:
            self.assertIn(
                required, chosen, f"{required} is declared but no branch produces it"
            )


class ApplyingTests(TempDirTestCase):
    """Carrying a decision out, and what happens when a gate says no."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            expansion_state=str(self.workdir / "expansion.json"),
        )
        expansion.propose_growth(
            self.config,
            [
                {
                    "source": "doc-a.txt",
                    "likely_cause": expansion.CAPACITY_CAUSE,
                    "index": 0.9,
                }
            ],
            cycle=1,
        )

    def grant(self, *domains):
        original = control.granted_domains
        control.granted_domains = lambda environ=None: set(domains)
        self.addCleanup(setattr, control, "granted_domains", original)

    def test_an_authorised_growth_is_applied(self):
        self.grant("expansion")
        decision = autonomy.Decision(autonomy.EXPAND, "short of capacity")
        outcome = autonomy.act(self.config, decision, holdout=4.0)
        self.assertTrue(outcome["applied"])
        self.assertFalse(outcome["denied"])
        self.assertTrue(expansion.pending(self.config)["capacity_growth"]["approved"])

    def test_a_refused_growth_does_not_stop_the_loop(self):
        # An agent that crashes because a human has not granted a permission is
        # an agent that stops. Refusing to expand is a normal outcome.
        self.grant()
        decision = autonomy.Decision(autonomy.EXPAND, "short of capacity")
        outcome = autonomy.act(self.config, decision, holdout=4.0)
        self.assertTrue(outcome["denied"])
        self.assertFalse(outcome["applied"])
        self.assertFalse(expansion.pending(self.config)["capacity_growth"]["approved"])
        self.assertIn(control.GRANT_VARIABLE, outcome["reason"])

    def test_the_growth_the_agent_applies_is_the_one_it_proposed(self):
        self.grant("expansion")
        proposed = expansion.pending(self.config)["capacity_growth"]
        autonomy.act(
            self.config,
            autonomy.Decision(autonomy.EXPAND, "short of capacity"),
            holdout=4.0,
        )
        pending = expansion.pending_shape(self.config)
        self.assertEqual(pending["n_layer"], proposed["to_layers"])
        self.assertEqual(pending["n_embd"], self.config.n_embd + expansion.WIDTH_STEP)

    def test_a_bad_optimizer_file_does_not_stop_the_growth(self):
        # The agent must not be able to crash itself by finding a state file
        # something else left behind. The file is all that matters here; no
        # model is built, because the point is what the agent does with a
        # broken file rather than how the file got there.
        path = train.optimizer_state_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a torch file at all")
        self.grant("expansion")
        outcome = autonomy.act(
            self.config,
            autonomy.Decision(autonomy.EXPAND, "short of capacity"),
            holdout=4.0,
        )
        self.assertTrue(outcome["applied"])


class GrowthRevertTests(TempDirTestCase):
    """A growth that did not help is undone, not merely labelled.

    It used to be recorded as a failure and then stay in the model forever. The
    extra layer was in the checkpoint, a smaller model cannot take a larger
    checkpoint, and nothing had been saved beforehand, so the honest options were
    a full retrain or accepting a model that had been made worse. That is the
    opposite of what Architecture 32 asks for, and it meant the acceptance gate
    could judge a growth and then do nothing about the judgement.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            expansion_state=str(self.workdir / "expansion.json"),
            device="cpu",
            n_layer=2,
            n_embd=16,
            n_head=2,
            block_size=32,
            byte_vocab_size=64,
        )
        self.small = Jupiter(
            ModelConfig(block_size=32, vocab_size=64, n_layer=2, n_head=2, n_embd=16)
        ).to("cpu")
        self.path = train.save_checkpoint(self.small, self.config, cycle=1)
        config.save_architecture(
            self.config, self.workdir / "out" / "Jupiter.architecture.json", 64
        )
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion", "architecture"}
        self.addCleanup(setattr, control, "granted_domains", original)

    def propose(self):
        expansion.propose_growth(
            self.config,
            [
                {
                    "source": "doc-a.txt",
                    "likely_cause": expansion.CAPACITY_CAUSE,
                    "index": 0.9,
                }
            ],
            cycle=1,
        )

    def grow(self):
        """Do what a run does once the growth is approved."""
        grown, architecture_path = train.build_model(self.config, vocab_size=64)
        train.attach_checkpoint(grown, [self.path], architecture_path)
        config.save_architecture(
            self.config, architecture_path, 64, shape=grown.config.to_dict()
        )
        train.save_checkpoint(grown, self.config, cycle=2)
        return grown, architecture_path

    def test_approving_a_growth_takes_a_snapshot_of_what_it_replaces(self):
        self.propose()
        proposal = expansion.approve_growth(self.config, holdout_loss=4.0)
        self.assertTrue(
            proposal.get("pre_growth_checkpoint"),
            "a growth with nothing to go back to cannot be undone",
        )
        self.assertTrue(Path(proposal["pre_growth_checkpoint"]).exists())

    def test_a_growth_that_did_not_help_puts_the_weights_back(self):
        self.propose()
        expansion.approve_growth(self.config, holdout_loss=4.0)
        grown, _path = self.grow()
        self.assertEqual(grown.config.n_layer, 3)

        review = expansion.resolve_growth(self.config, 5.0)
        self.assertFalse(review["kept"])
        self.assertTrue(review.get("reverted_to"), "the growth was not undone")
        restored = load_file(
            str(Path(self.config.out_dir) / "Jupiter_latest.safetensors")
        )
        self.assertTrue(
            torch.equal(
                restored["blocks.0.attn.qkv.weight"],
                self.small.blocks[0].attn.qkv.weight,
            ),
            "the weights before the growth must be back",
        )

    def test_a_reverted_growth_is_not_applied_again(self):
        self.propose()
        expansion.approve_growth(self.config, holdout_loss=4.0)
        self.grow()
        expansion.resolve_growth(self.config, 5.0)
        self.assertIsNone(
            expansion.pending_shape(self.config),
            "a growth that was undone must not grow the next model again",
        )

    def test_a_reverted_growth_rolls_the_shape_record_back(self):
        # Leaving the record describing the grown model while the weights are
        # the smaller ones would make the next run build the grown shape and
        # immediately archive what was just restored.
        self.propose()
        expansion.approve_growth(self.config, holdout_loss=4.0)
        _grown, architecture_path = self.grow()
        expansion.resolve_growth(self.config, 5.0)
        recorded, _version = config.read_architecture(architecture_path)
        self.assertEqual(recorded["n_layer"], 2)
        self.assertEqual(recorded["n_embd"], 16)

    def test_a_growth_that_helped_is_kept_and_nothing_is_restored(self):
        self.propose()
        expansion.approve_growth(self.config, holdout_loss=4.0)
        review = expansion.resolve_growth(self.config, 3.0)
        self.assertTrue(review["kept"])
        self.assertIsNone(review.get("reverted_to"))
        self.assertTrue(review["approved"], "a kept growth stays applied")
        self.assertIsNotNone(expansion.pending_shape(self.config))

    def test_a_growth_with_nothing_to_revert_to_says_so(self):
        # No published checkpoint means no snapshot, and the verdict still has to
        # be honest about what can and cannot be undone.
        c = make_config(
            out_dir=str(self.workdir / "empty"),
            expansion_state=str(self.workdir / "x.json"),
            n_layer=2,
        )
        expansion.propose_growth(
            c,
            [
                {
                    "source": "doc-a.txt",
                    "likely_cause": expansion.CAPACITY_CAUSE,
                    "index": 0.9,
                }
            ],
            cycle=1,
        )
        expansion.approve_growth(c, holdout_loss=4.0)
        review = expansion.resolve_growth(c, 5.0)
        self.assertFalse(review["kept"])
        self.assertIsNone(review.get("reverted_to"))


class LoopRecordTests(TempDirTestCase):
    """The loop's own memory, which is what makes it auditable."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            autonomy_state=str(self.workdir / "autonomy.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
            knowledge_store=str(self.workdir / "knowledge.json"),
            expansion_state=str(self.workdir / "expansion.json"),
            selection_state=str(self.workdir / "selection.json"),
        )

    def test_a_pass_is_recorded_with_its_reason(self):
        autonomy.step(self.config, candidates=["a.txt"])
        report = autonomy.summary(self.config)
        self.assertEqual(report["passes"], 1)
        self.assertIn("learn", report["actions_taken"])
        self.assertTrue(report["last_decision"]["reason"])

    def test_the_history_is_capped_so_a_long_run_cannot_grow_it_without_bound(self):
        # The state file is rewritten every pass. A continuous agent running for
        # days would otherwise produce a file nobody can open.
        state = {"passes": 0, "history": []}
        for index in range(500):
            state["history"].append({"pass": index, "action": "learn", "reason": "x"})
            autonomy.save(self.config, state)
        self.assertLessEqual(len(autonomy.load(self.config)["history"]), 200)

    def test_a_corrupt_state_file_does_not_stop_the_loop(self):
        # A loop that cannot start because its own memory is damaged cannot
        # recover from anything, including being killed mid-write.
        with open(self.config.autonomy_state, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertEqual(autonomy.load(self.config)["passes"], 0)
        autonomy.step(self.config, candidates=["a.txt"])
        self.assertEqual(autonomy.summary(self.config)["passes"], 1)

    def test_idle_passes_reset_once_there_is_something_to_do(self):
        # Otherwise a loop that idles twenty times and then finds work would
        # still be one pass from the idle limit and would stop immediately.
        autonomy.step(self.config, candidates=[])
        self.assertEqual(autonomy.load(self.config)["idle_passes"], 1)
        autonomy.step(self.config, candidates=["a.txt"])
        self.assertEqual(autonomy.load(self.config)["idle_passes"], 0)

    def test_the_report_can_be_printed_without_the_loop_having_run(self):
        report = autonomy.summary(
            make_config(autonomy_state=str(self.workdir / "n.json"))
        )
        self.assertEqual(report["passes"], 0)


class DecisionActsTests(TempDirTestCase):
    """A decision that decides nothing is a log line, not a decision layer.

    LEARN and REVIEW both used to fall through to `return {"applied": True}`,
    recording that something happened without doing it, and the worker ran its
    cycles afterwards whatever came back. These hold the action to performing
    the work it chose.
    """

    def config(self):
        return make_config(
            autonomy_state=str(self.workdir / "autonomy.json"),
            knowledge_store=str(self.workdir / "out" / "knowledge.json"),
        )

    def observing(self, **overrides):
        base = {
            "candidates": ["a.txt"],
            "candidate_count": 1,
            "sources_measured": 0,
            "capacity_shortages": 0,
            "cycles_recorded": 3,
            "cycles_accepted": 2,
            "mean_improvement": -0.5,
            "improving": True,
            "sources_seen": 0,
            "granted": [],
            "idle_passes": 0,
        }
        base.update(overrides)
        return base

    def test_learn_performs_a_cycle_when_a_runner_is_supplied(self):
        calls = []
        decision = autonomy.Decision(autonomy.LEARN, "there is material")
        outcome = autonomy.act(self.config(), decision, runner=calls.append)
        self.assertTrue(outcome["performed"])
        self.assertEqual(outcome["cycles"], 4)
        self.assertEqual(len(calls), 1)

    def test_review_performs_fewer_cycles_than_learn(self):
        # Reviewing past work is looking before making more, so it does one pass
        # rather than a full batch.
        calls = []
        decision = autonomy.Decision(autonomy.REVIEW, "recent record is worth reading")
        outcome = autonomy.act(self.config(), decision, runner=calls.append)
        self.assertTrue(outcome["performed"])
        self.assertEqual(outcome["cycles"], 1)
        self.assertEqual(calls, [1])

    def test_a_decision_with_no_runner_is_reported_as_not_performed(self):
        # Claiming an action was applied when nothing applied it is the specific
        # dishonesty this project exists to avoid.
        decision = autonomy.Decision(autonomy.LEARN, "there is material")
        outcome = autonomy.act(self.config(), decision)
        self.assertFalse(outcome["performed"])
        self.assertIn("no runner", outcome["note"])

    def test_rest_does_not_run_a_cycle(self):
        calls = []
        decision = autonomy.Decision(autonomy.REST, "nothing to learn")
        outcome = autonomy.act(self.config(), decision, runner=calls.append)
        self.assertFalse(outcome["performed"])
        self.assertTrue(outcome["rest"])
        self.assertEqual(calls, [])

    def test_stop_asks_the_caller_to_end_the_run(self):
        decision = autonomy.Decision(autonomy.STOP, "there is nothing coming")
        outcome = autonomy.act(self.config(), decision)
        self.assertTrue(outcome["stop"])

    def test_a_long_enough_idle_run_concludes_rather_than_waiting_forever(self):
        # The difference between resting and being stuck.
        config = self.config()
        decision = autonomy.decide(
            config, self.observing(candidates=[], candidate_count=0, idle_passes=99)
        )
        self.assertEqual(decision.action, autonomy.STOP)
        self.assertIn("nothing coming", decision.reason)

    def test_one_empty_pass_rests_rather_than_concluding(self):
        decision = autonomy.decide(
            self.config(), self.observing(candidates=[], candidate_count=0)
        )
        self.assertEqual(decision.action, autonomy.REST)


class ObserveTests(TempDirTestCase):
    """What the loop believes about itself, which everything else rests on."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            autonomy_state=str(self.workdir / "autonomy.json"),
            experiment_log=str(self.workdir / "experiments.jsonl"),
            knowledge_store=str(self.workdir / "knowledge.json"),
            expansion_state=str(self.workdir / "expansion.json"),
            selection_state=str(self.workdir / "selection.json"),
        )

    def test_an_empty_record_reports_no_cycles_rather_than_none_of_the_number(self):
        observed = autonomy.observe(self.config, candidates=[])
        self.assertEqual(observed["cycles_recorded"], 0)
        self.assertIsNone(observed["mean_improvement"])
        self.assertFalse(observed["improving"])

    def test_improving_cycles_are_reported_as_improving(self):
        write_experiment(self.config.experiment_log, 5.0, 3.0)
        observed = autonomy.observe(self.config, candidates=["a.txt"])
        self.assertTrue(observed["improving"])
        self.assertEqual(observed["mean_improvement"], -2.0)
        self.assertEqual(observed["cycles_accepted"], 1)

    def test_worsening_cycles_are_reported_as_not_improving(self):
        write_experiment(self.config.experiment_log, 3.0, 5.0)
        observed = autonomy.observe(self.config, candidates=["a.txt"])
        self.assertFalse(observed["improving"])
        self.assertEqual(observed["mean_improvement"], 2.0)

    def test_the_observation_carries_what_was_granted(self):
        # The loop's authority is part of what it observes about itself, because
        # it changes what it is allowed to decide.
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)
        self.assertEqual(autonomy.observe(self.config)["granted"], ["expansion"])


class OptimizerPersistenceTests(TempDirTestCase):
    """A resumed run is the same run, not a new one that starts from the same
    weights.

    The optimizer was rebuilt at the start of every run, so AdamW's moments were
    thrown away between cycles and the opening steps of a bounded micro-update
    moved on a cold estimate of every gradient. For an update made of a few
    dozen steps, that is most of what the update is.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(out_dir=str(self.workdir / "out"), device="cpu")

    def stepped_model(self, n_embd=16):
        model = Jupiter(
            ModelConfig(
                block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=n_embd
            )
        ).to("cpu")
        optimizer = model.configure_optimizer(1e-3, 0.01)
        tokens = torch.randint(0, 64, (1, 16))
        logits, _ = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1].reshape(-1, 64), tokens[0, 1:]
        )
        loss.backward()
        optimizer.step()
        return model, optimizer

    def test_moments_are_written_and_read_back(self):
        model, optimizer = self.stepped_model()
        saved = len(optimizer.state_dict()["state"])
        self.assertGreater(saved, 0, "a stepped optimizer must have moments")
        train.save_optimizer_state(optimizer, self.config, cycle=4)

        fresh = Jupiter(model.config).to("cpu")
        fresh_optimizer = fresh.configure_optimizer(1e-3, 0.01)
        self.assertEqual(len(fresh_optimizer.state_dict()["state"]), 0)
        report = train.load_optimizer_state(fresh_optimizer, fresh, self.config)
        self.assertEqual(len(fresh_optimizer.state_dict()["state"]), saved)
        self.assertIn("resumed", report)

    def test_a_model_that_has_grown_starts_cold_rather_than_loading_a_stale_state(self):
        # The moments describe parameters that no longer exist at the same
        # addresses. Loading them anyway would be worse than starting cold, and
        # torch.load_state_dict does not check: it matches by position and
        # succeeds, leaving most parameters with no moment at all.
        model, optimizer = self.stepped_model(n_embd=16)
        train.save_optimizer_state(optimizer, self.config, cycle=4)

        grown = Jupiter(
            ModelConfig(block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=32)
        ).to("cpu")
        grown_optimizer = grown.configure_optimizer(1e-3, 0.01)
        report = train.load_optimizer_state(grown_optimizer, grown, self.config)
        self.assertEqual(len(grown_optimizer.state_dict()["state"]), 0)
        self.assertIn("growth", report)

    def test_a_corrupt_optimizer_file_does_not_stop_a_run(self):
        model, optimizer = self.stepped_model()
        path = train.optimizer_state_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a torch file")
        fresh = Jupiter(model.config).to("cpu")
        fresh_optimizer = fresh.configure_optimizer(1e-3, 0.01)
        report = train.load_optimizer_state(fresh_optimizer, fresh, self.config)
        self.assertIn("could not be read", report)
        self.assertEqual(len(fresh_optimizer.state_dict()["state"]), 0)

    def test_no_state_file_is_reported_as_a_first_run_rather_than_a_failure(self):
        model = Jupiter(
            ModelConfig(block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=16)
        ).to("cpu")
        optimizer = model.configure_optimizer(1e-3, 0.01)
        report = train.load_optimizer_state(optimizer, model, self.config)
        self.assertIn("no optimizer state", report)

    def test_the_optimizer_state_sits_beside_the_resume_weights(self):
        # Beside rather than inside: the checkpoint's metadata answers "which
        # model is this", and a second kind of state inside it makes that harder
        # to answer for no benefit.
        model, optimizer = self.stepped_model()
        train.save_optimizer_state(optimizer, self.config, cycle=2)
        self.assertEqual(
            train.optimizer_state_path(self.config).parent,
            self.workdir / "out",
        )

    def test_the_published_checkpoint_carries_no_optimizer_state(self):
        # The conversation loads the published checkpoint and does not train, so
        # moments in it would be state nobody reads.
        model, optimizer = self.stepped_model()
        path = train.save_checkpoint(model, self.config, cycle=1, optimizer=optimizer)
        self.assertFalse(train.optimizer_state_path(self.config).exists())
        self.assertTrue(path.exists())


class GrowthPromptTests(TempDirTestCase):
    """A person is asked to decide about a growth, at the end of a run.

    Architecture 23's loop ends in "keep or rollback" and Architecture 38 puts
    expansion behind human control, so the decision belongs to a person. The
    loop had no way to ask: it proposed, printed a line, and stopped, and the
    proposal sat in a file until somebody went looking for it. A control that
    is present and practically never used is not a control.

    The tests below care most about the case that would be worst if it were
    wrong. A prompt on a detached worker is a run that hangs waiting for input
    that will never come, and that is worse than no prompt at all.
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
        self.grant_expansion()
        expansion.propose_growth(self.config, self.short, 1)

    def grant_expansion(self):
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"expansion"}
        self.addCleanup(setattr, control, "granted_domains", original)

    def answer(self, text):
        """Make input() return a fixed answer, the way a person would."""
        import builtins

        original = builtins.input
        builtins.input = lambda prompt="": text
        self.addCleanup(setattr, builtins, "input", original)

    def as_interactive(self):
        """Pretend there is a person at the keyboard."""
        original = learning_worker._interactive
        learning_worker._interactive = lambda: True
        self.addCleanup(setattr, learning_worker, "_interactive", original)

    def test_no_prompt_without_the_expansion_grant(self):
        # Offering an action the Architecture 38 gate would refuse is asking a
        # person to approve something the system will not do.
        original_interactive = learning_worker._interactive
        learning_worker._interactive = lambda: False
        self.addCleanup(setattr, learning_worker, "_interactive", original_interactive)
        self.answer("y")
        learning_worker._last_best_holdout = 2.44
        self.assertIsNone(learning_worker.prompt_for_growth(self.config, {}))
        self.assertFalse(
            expansion.load(self.config)["capacity_growth"]["approved"],
            "a prompt must not be able to bypass the gate",
        )

    def test_no_prompt_when_nobody_can_answer(self):
        # The detached case. A prompt here is a run that hangs on input which
        # will never arrive. The terminal check itself is stubbed rather than
        # asserted, because the test runner may or may not have a tty and the
        # property under test is what happens when it does not.
        import sys

        original = sys.stdin
        sys.stdin = None
        self.addCleanup(setattr, sys, "stdin", original)
        self.assertFalse(
            learning_worker._interactive(),
            "a run with no terminal must not be asked a question",
        )

    def test_the_escape_switches_the_prompt_off(self):
        import os

        os.environ["JUPITER_NO_PROMPT"] = "1"
        self.addCleanup(os.environ.pop, "JUPITER_NO_PROMPT", None)
        self.assertFalse(learning_worker._interactive())

    def test_saying_no_changes_nothing(self):
        self.as_interactive()
        self.answer("n")
        learning_worker._last_best_holdout = 2.44
        self.assertIsNone(learning_worker.prompt_for_growth(self.config, {}))
        self.assertFalse(expansion.load(self.config)["capacity_growth"]["approved"])

    def test_anything_that_is_not_a_yes_declines(self):
        # A stray keystroke, an empty line or a closed input must not grow the
        # model. Only an explicit yes may.
        for answer in ("", "maybe", "yep", "1", "no"):
            with self.subTest(answer=answer):
                expansion.load(self.config)["capacity_growth"]["approved"] = False
                expansion.save(self.config, expansion.load(self.config))
                self.as_interactive()
                self.answer(answer)
                learning_worker._last_best_holdout = 2.44
                learning_worker.prompt_for_growth(self.config, {})
                self.assertFalse(
                    expansion.load(self.config)["capacity_growth"]["approved"],
                    f"{answer!r} must not approve a growth",
                )

    def test_saying_yes_approves_and_records_the_baseline(self):
        # The baseline is the loss the growth is judged against, so the verdict
        # later is against a number rather than against a feeling.
        self.as_interactive()
        self.answer("yes")
        learning_worker._last_best_holdout = 2.44
        result = learning_worker.prompt_for_growth(self.config, {})
        self.assertIsNotNone(result)
        self.assertTrue(result["approved"])
        self.assertEqual(result["holdout_at_approval"], 2.44)
        self.assertEqual(
            expansion.pending_shape(self.config),
            {"n_layer": 6, "n_embd": 320},
            "the next run must build the bigger model",
        )

    def test_no_prompt_for_a_growth_already_approved(self):
        expansion.approve_growth(self.config, holdout_loss=4.0)
        self.as_interactive()
        self.answer("y")
        learning_worker._last_best_holdout = 2.44
        self.assertIsNone(learning_worker.prompt_for_growth(self.config, {}))

    def test_no_prompt_when_there_is_no_proposal(self):
        self.config2 = make_config(
            expansion_state=str(self.workdir / "empty.json"),
            out_dir=str(self.workdir / "out2"),
        )
        self.as_interactive()
        self.answer("y")
        learning_worker._last_best_holdout = 2.44
        self.assertIsNone(learning_worker.prompt_for_growth(self.config2, {}))

    def judge_and_roll_back(self):
        """Put the state in the position a real reverted growth leaves behind.

        A growth only becomes un-approved when there was a checkpoint to revert
        to, so this writes that state rather than pretending a failure alone
        produces it. A growth judged on a run that had nothing to restore stays
        approved, because there was no earlier model to go back to.
        """
        expansion.approve_growth(self.config, holdout_loss=4.0)
        expansion.resolve_growth(self.config, 5.0)
        state = expansion.load(self.config)
        state["capacity_growth"]["approved"] = False
        state["capacity_growth"]["kept"] = False
        expansion.save(self.config, state)
        return state["capacity_growth"]

    def test_a_reverted_growth_is_offered_again_on_the_same_shape(self):
        # A rollback restores the model to exactly the shape the failed attempt
        # started from, so a guard keyed on "the shape has not changed" is
        # permanently true after the first reverted growth. That is what made the
        # question unaskable: the verdict was the system's own measurement, and
        # Architecture 38 gives the decision to a person, so the person is asked
        # again and shown what happened last time.
        self.judge_and_roll_back()
        self.as_interactive()
        self.answer("y")
        learning_worker._last_best_holdout = 2.44
        result = learning_worker.prompt_for_growth(self.config, {})
        self.assertIsNotNone(
            result,
            "a reverted growth must be offered again: Architecture 38 puts this "
            "decision with a person, and a latch here is a control that is off",
        )
        self.assertTrue(result["approved"])

    def test_the_reoffer_names_the_previous_verdict(self):
        # Approving a second time is only an informed decision if the prompt says
        # what the first attempt did. The number is read off the proposal rather
        # than recomputed, so the figure shown is the figure that was recorded.
        self.judge_and_roll_back()
        self.as_interactive()
        self.answer("n")
        learning_worker._last_best_holdout = 2.44
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            learning_worker.prompt_for_growth(self.config, {})
        printed = buffer.getvalue()
        self.assertIn("rolled back", printed)
        self.assertIn("did not help", printed)

    def test_a_reverted_growth_is_offered_again_once_the_model_has_changed(self):
        # The user reported a run that never asked. The reason was this: a growth
        # that had been rolled back was treated as settled forever, so its
        # proposal sat in the state file looking un-answered when it had in fact
        # been answered about a model that no longer exists.
        self.judge_and_roll_back()
        self.config.n_layer = 9
        self.as_interactive()
        self.answer("y")
        learning_worker._last_best_holdout = 2.44
        result = learning_worker.prompt_for_growth(self.config, {})
        self.assertIsNotNone(result, "a new shape has not been judged, so ask")
        self.assertTrue(result["approved"])
