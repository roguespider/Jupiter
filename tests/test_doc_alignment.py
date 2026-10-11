"""Tests that the documentation and the code cannot drift apart.

Architecture 53 asks that architecture changes be recorded, and this project's
own rule is that a feature listed as covered with nothing behind it is worse
than an absent one. The recurring failure that these tests exist to prevent is
not a missing feature. It is a declaration: a constant, an enumeration, a
docstring or a document sentence that describes something the code cannot
produce, which reads as a description of the system and is a list of intentions.

Each class below pins one kind of declaration to something the code can actually
do. They are mostly about refusals, because that is where a declaration drifts:
an action nothing returns, a destination nothing routes to, a type nothing
creates. A test that only checked the happy path would pass against a system
that could do one thing out of six it claimed.
"""

import ast
import unittest
from pathlib import Path

import acceptance
import autonomy
import code_knowledge
import concepts
import frustration
import growth
import relationships
import tokenizer as tokenizer_module
import train

import memory
from tests.support import TempDirTestCase, make_config

PROJECT = Path(__file__).resolve().parent.parent

# Modules that legitimately have no production entry point, and why. A module
# that is only reachable from a test is dead code by Architecture 40's rule, and
# the connectivity suite already enforces reachability from the entry point.
# This list exists so that if a module is removed from the entry point, the
# connectivity test fails and names it, rather than someone adding it here.


class ActionVocabularyTests(TempDirTestCase):
    """The agent's declared actions against the ones it can actually choose.

    The vocabulary once listed five actions no branch could produce, so a caller
    checking a decision against it was being told they were possible. decide()
    needs a config, so this lives in a case that has one rather than reaching for
    a fake.
    """

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
            "idle_passes": 0,
            # Whether the world has anything to offer, and whether the run may
            # read it. Both are read by the policy, so both are part of what an
            # observation can say.
            "external_configured": False,
            "consecutive_consults": 0,
            "query_terms": [],
        }
        base.update(overrides)
        return base

    def test_every_action_the_policy_decides_is_declared(self):
        chosen = set()
        for observation, growth_state in (
            (self.observation(candidates=[], candidate_count=0), None),
            (self.observation(cycles_recorded=3, mean_improvement=-0.5), None),
            (self.observation(), {"approved": True, "reviewed_at": None}),
            (self.observation(), {"approved": True, "reviewed_at": 1.0, "kept": False}),
            (self.observation(capacity_shortages=2, granted=["expansion"]), None),
            (self.observation(capacity_shortages=2, granted=[]), None),
            (self.observation(cycles_recorded=9, mean_improvement=0.4), None),
            # Nothing learnable for long enough that the loop concludes rather
            # than resting. Without this branch STOP is in the vocabulary and
            # produced by nothing, which is the defect this test exists for.
            (
                self.observation(candidates=[], candidate_count=0, idle_passes=10_000),
                None,
            ),
            # A dry corpus with a source configured and the external domain
            # granted. Without this branch CONSULT is in the vocabulary and
            # produced by nothing, which is the same defect this test exists for.
            (
                self.observation(
                    candidates=[],
                    candidate_count=0,
                    external_configured=True,
                    granted=["external"],
                ),
                None,
            ),
        ):
            chosen.add(
                autonomy.decide(
                    self.make_config(), observation, growth_state=growth_state
                ).action
            )
        self.assertEqual(chosen - set(autonomy.ACTIONS), set())
        # Every declared action, not a hand-picked subset: REST and STOP were
        # both in the vocabulary and produced by no branch until the loop was
        # given a way to conclude as well as to pause.
        self.assertEqual(
            set(autonomy.ACTIONS) - chosen,
            set(),
            "a decision the policy depends on is no longer reachable",
        )

    def make_config(self):
        return make_config(autonomy_state=str(self.workdir / "autonomy.json"))


class DeclarationTests(unittest.TestCase):
    """Every declared vocabulary must be reachable or marked unreachable."""

    def test_every_relationship_type_is_created_or_reported_as_unused(self):
        # Ten of the eleven were named and never created, which is what left the
        # concept graph as nine fragments of co-occurrence. The remaining ones
        # are still unreachable; the point is that the system says which.
        coverage = relationships.relationship_coverage(_empty_store())
        self.assertEqual(
            set(coverage["types_present"]) | set(coverage["types_unused"]),
            set(relationships.TYPES),
        )
        self.assertIn("types_unused", coverage)
        self.assertIn("note", coverage)

    def test_the_document_and_the_code_agree_on_how_many_relationship_types_exist(self):
        document = (PROJECT / "JUPITER_ARCHITECTURE.md").read_text(encoding="utf-8")
        # The document says the remaining types are named and never created. If
        # a future change derives more of them, that sentence becomes false and
        # this fails rather than leaving the document quietly wrong.
        self.assertIn("still named and never created", document)
        self.assertIn("SIMILAR_TO", document)
        self.assertIn("PRECEDES", document)

    def test_every_memory_destination_is_reachable(self):
        # Three of the six were declared and had no code path that could produce
        # them, so Architecture 16's enumeration was three-quarters true.
        found = set()
        for frequency in (0, 3, 10):
            for importance in (0.0, 0.75, 0.95):
                for confidence in (0.0, 0.85, 0.95):
                    for generalization in (0.0, 0.65, 0.9):
                        for permanence in (0.0, 0.75, 0.95):
                            for usefulness in (0.0, 0.75, 0.95):
                                for risk in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
                                    found.add(
                                        memory.decide_destination(
                                            frequency=frequency,
                                            importance=importance,
                                            confidence=confidence,
                                            generalization=generalization,
                                            permanence=permanence,
                                            usefulness=usefulness,
                                            contradiction_risk=risk,
                                        )["destination"]
                                    )
        self.assertEqual(set(memory.DESTINATIONS) - found, set())

    def test_every_frustration_cause_produced_is_declared(self):
        produced = set()
        for attempts in range(0, 12):
            for errors in range(0, attempts + 1):
                produced.add(
                    frustration.likely_cause(
                        {"attempts": attempts, "errors": errors, "improvement": -1.0}
                    )
                )
        self.assertEqual(produced - set(frustration.CAUSES), set())

    def test_causes_the_system_cannot_evidence_say_so(self):
        # Rather than inventing a measurement to make the list look complete.
        self.assertTrue(frustration.UNEVIDENCED_CAUSES)
        for cause in frustration.UNEVIDENCED_CAUSES:
            self.assertIn(cause, frustration.CAUSES)

    def test_every_concept_maturity_the_code_can_produce_is_declared(self):
        for evidence, contradictions, confidence in (
            (0, 0, 0.0),
            (1, 0, 0.1),
            (3, 0, 0.6),
            (8, 0, 0.9),
            (9, 4, 0.1),
        ):
            value = concepts.maturity_of(
                {
                    "evidence": evidence,
                    "contradictions": contradictions,
                    "confidence": confidence,
                }
            )
            self.assertIn(value, concepts.MATURITIES)

    def test_every_verdict_the_run_review_counts_is_a_real_verdict(self):
        # ROLLBACK was missing from the review because it named four verdicts as
        # literals, so a run that had undone something was reported as a run that
        # had done nothing.
        summary = _experiment_module().review(
            [{"decision": verdict, "source": "s"} for verdict in acceptance.VERDICTS]
        )
        for verdict in acceptance.VERDICTS:
            self.assertIn(verdict, acceptance.VERDICTS)
        self.assertEqual(
            summary["accepted"]
            + summary["rejected"]
            + summary["deferred"]
            + summary["flagged_for_review"]
            + summary["rolled_back"],
            len(acceptance.VERDICTS),
        )

    def test_every_definition_kind_the_code_emits_is_declared(self):
        # A kind spelled differently from DEFINITION_KINDS would be written to
        # the store and would never appear in any count of what the corpus holds.
        python_kinds = {"function", "method", "class", "constant", "variable"}
        self.assertTrue(
            python_kinds <= set(code_knowledge.DEFINITION_KINDS),
            "a kind the Python reader emits is not a declared definition kind",
        )

    def test_growable_shape_fields_match_what_growth_actually_grows(self):
        # growth.compatibility compares these five to decide whether a saved
        # model can grow into a target. A field that could be grown but is not
        # compared, or compared but cannot be, is a hole in the guard.
        self.assertEqual(
            set(growth.GROWTH_FIELDS),
            {"block_size", "vocab_size", "n_layer", "n_head", "n_embd"},
        )
        self.assertEqual(
            set(growth.STACKED_ROW_MULTIPLIERS.values()),
            {3, 4},
            "the fused projections are three and four groups; a new stacked "
            "tensor has to be declared here or it is widened as a plain prefix "
            "and its learned rows land in the wrong place",
        )

    def test_no_module_declares_a_constant_nothing_reads(self):
        # The general form of the failure this file exists for. A module-level
        # constant that is declared and never read is a declaration about the
        # system that nothing enforces, and adding one is how the vocabulary
        # starts lying again.
        sources = {
            path.stem: path.read_text(encoding="utf-8") for path in PROJECT.glob("*.py")
        }
        unread = []
        for path in sorted(PROJECT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name) or not target.id.isupper():
                        continue
                    if sum(text.count(target.id) for text in sources.values()) <= 1:
                        unread.append(f"{path.name}:{node.lineno} {target.id}")
        self.assertEqual(
            unread,
            [],
            "constants nothing reads: " + ", ".join(unread),
        )

    def test_no_module_defines_a_function_nothing_calls(self):
        sources = list(PROJECT.glob("*.py")) + list(PROJECT.glob("tests/*.py"))
        dead = []
        for path in sorted(PROJECT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                if node.name in ("main", "setUp", "tearDown"):
                    continue
                uses = sum(
                    p.read_text(encoding="utf-8").count(node.name) for p in sources
                )
                if uses <= 1:
                    dead.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(dead, [], "functions nothing calls: " + ", ".join(dead))

    def test_the_tokenizer_does_not_declare_a_second_vocabulary_size(self):
        # It did, duplicating ByteTokenizer.vocab_size, and nothing in production
        # read it, so a reader could not tell which was authoritative.
        self.assertFalse(hasattr(tokenizer_module, "VOCAB_SIZE"))

    def test_the_learning_rate_accessor_reads_the_optimizer(self):
        # The number the worker reports must be the number learning used, not the
        # variable it computed.
        from model import Jupiter, ModelConfig

        model = Jupiter(
            ModelConfig(block_size=16, vocab_size=32, n_layer=1, n_head=2, n_embd=8)
        )
        optimizer = model.configure_optimizer(0.01, 0.0)
        self.assertEqual(train.get_learning_rate(optimizer), 0.01)

    def test_reset_optimizer_reports_the_rate_it_actually_applied(self):
        from model import Jupiter, ModelConfig

        model = Jupiter(
            ModelConfig(block_size=16, vocab_size=32, n_layer=1, n_head=2, n_embd=8)
        )

        class _Config:
            weight_decay = 0.0

        optimizer, applied = train.reset_optimizer(model, _Config(), 0.005)
        self.assertEqual(applied, 0.005)
        self.assertEqual(train.get_learning_rate(optimizer), 0.005)


def _empty_store():
    import tempfile

    import store as store_module

    return store_module.Store(str(Path(tempfile.mkdtemp()) / "knowledge.json"))


def _experiment_module():
    import experiment

    return experiment
