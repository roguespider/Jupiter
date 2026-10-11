"""Tests for consulting the outside world (Architectures 28, 37 and 38).

Three claims are tested here, and the third is the one that matters most.

The first is that the decision layer can choose to reach outside itself. Until
this existed, Architecture 28's external knowledge module was reachable from the
entry point but never called by anything that ran: evaluation imported it only
to report a census of sources that nothing consulted. Jupiter could describe a
world it never queried.

The second is that the choice is bounded. A world is an unbounded supply of
documents, so a loop allowed to consult it without limit never concludes there
is nothing coming and never reaches its own stopping rule. CONSULT is an answer
to a dry corpus, not a reason to stop asking.

The third is that reaching outside is gated on the Architecture 38 external
domain, and that a refusal is reported as a refusal. This is the half that makes
the first half safe: a module that reads arbitrary configured paths and opens
sockets to agents it did not start is external tool access whether or not anyone
called it that name, and it must not be reachable on Jupiter's own initiative
without a human having said so.
"""

import json
import unittest
from pathlib import Path

import autonomy
import concepts
import control
import external
import store as store_module

from tests.support import TempDirTestCase, make_config


class ConsultFixture(TempDirTestCase):
    """A dry corpus, one external source, and a concept worth consulting about."""

    def setUp(self):
        super().setUp()
        self.docs = self.workdir / "docs"
        self.docs.mkdir()
        (self.docs / "valves.md").write_text(
            "A valve controls flow through a pipe. The spring returns the valve "
            "to its seat.",
            encoding="utf-8",
        )
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            autonomy_state=str(self.workdir / "out" / "autonomy.json"),
            knowledge_store=str(self.workdir / "out" / "knowledge.json"),
            control_log=str(self.workdir / "out" / "control_decisions.jsonl"),
            external_sources=(
                {"kind": "file", "name": "docs", "location": str(self.docs)},
            ),
        )
        self.knowledge = store_module.Store(self.config.knowledge_store)
        self.knowledge.put(
            concepts.CONCEPTS,
            "valve",
            concepts.new_concept("valve", "a valve controls flow through a pipe"),
        )

    def grant(self, *domains):
        self.assertIn("external", domains)
        original = control.granted_domains
        control.granted_domains = lambda environ=None: set(domains)
        self.addCleanup(setattr, control, "granted_domains", original)


class ExternalGateTests(ConsultFixture):
    """Nothing in external.py leaves the process without the grant."""

    def test_fetching_without_the_grant_is_refused(self):
        with self.assertRaises(control.ControlDenied):
            external.fetch(self.config, "valve")

    def test_the_refusal_names_the_grant_a_human_has_to_make(self):
        # A control that refuses without saying what would permit it is a wall
        # rather than a decision, and the reader of the log is the person who
        # could unblock it.
        with self.assertRaises(control.ControlDenied) as caught:
            external.fetch(self.config, "valve")
        self.assertIn(control.GRANT_VARIABLE, str(caught.exception))
        self.assertIn("external", str(caught.exception))

    def test_probing_an_agent_without_the_grant_is_refused(self):
        client = external.McpClient("peer", "http://127.0.0.1:9/")
        with self.assertRaises(control.ControlDenied):
            client.available()

    def test_a_refusal_is_not_swallowed_into_an_empty_result(self):
        # ControlDenied is a PermissionError, which is an OSError, which the
        # per-source handler catches. Letting it through would report a refusal
        # as a world with nothing to say, and those are different facts.
        with self.assertRaises(control.ControlDenied):
            external.fetch(self.config, "valve")


class CensusHonestyTests(ConsultFixture):
    """The census must not report an unasked question as a bad answer."""

    def test_an_unprobed_agent_is_unknown_rather_than_unreachable(self):
        client = external.McpClient("peer", "http://127.0.0.1:9/")
        original = external.sources
        external.sources = lambda c: [client]
        self.addCleanup(setattr, external, "sources", original)
        report = external.census(self.config)
        row = report["sources"][0]
        self.assertIsNone(row["reachable"])
        self.assertIn("not attempted", row["probe"])

    def test_a_file_source_is_reported_without_asking_anything(self):
        report = external.census(self.config)
        row = report["sources"][0]
        self.assertEqual(row["documents"], 1)
        self.assertNotIn("probe", row)


class ConsultDecisionTests(ConsultFixture):
    """When the loop decides to reach outside, and when it does not."""

    def observe(self, **overrides):
        for key, value in overrides.items():
            setattr(self.config, key, value)
        return autonomy.observe(self.config, candidates=[])

    def test_a_dry_corpus_with_a_source_consults(self):
        self.grant("external")
        decision = autonomy.decide(self.config, self.observe())
        self.assertEqual(decision.action, autonomy.CONSULT)

    def test_a_dry_corpus_without_a_source_rests(self):
        self.grant("external")
        self.config.external_sources = ()
        decision = autonomy.decide(self.config, self.observe())
        self.assertEqual(decision.action, autonomy.REST)

    def test_a_corpus_with_material_learns_rather_than_consults(self):
        # The world is for a dry corpus. Consulting while there is material to
        # learn would spend an external access on a question Jupiter could have
        # answered from what it already has.
        self.grant("external")
        observation = autonomy.observe(
            self.config, candidates=[str(self.docs / "valves.md")]
        )
        self.assertEqual(
            autonomy.decide(self.config, observation).action, autonomy.LEARN
        )

    def test_an_ungranted_source_is_reported_as_refused_not_as_rest(self):
        # The loop can see the world would answer and cannot ask it. That is a
        # different fact from a world with nothing to say, and resting as though
        # nothing were there would hide a decision that is waiting on a person.
        decision = autonomy.decide(self.config, self.observe())
        self.assertEqual(decision.action, autonomy.CONSULT)
        self.assertEqual(decision.evidence["external"], "refused, not granted")
        self.assertIn(control.GRANT_VARIABLE, decision.reason)

    def test_the_consult_bound_stops_the_loop_asking_forever(self):
        self.grant("external")
        observation = self.observe()
        observation["consecutive_consults"] = (
            self.config.autonomy_max_consecutive_consults
        )
        decision = autonomy.decide(self.config, observation)
        self.assertEqual(decision.action, autonomy.REST)

    def test_the_query_is_built_from_concepts_and_not_from_content_hashes(self):
        # A curated document is filed under a content hash, so a query built from
        # a filename asks the world about a hex string and retrieves nothing
        # while looking exactly like a search that worked.
        self.grant("external")
        terms = self.observe()["query_terms"]
        self.assertIn("valve", terms)
        self.assertTrue(all(not term.isdigit() for term in terms))

    def test_the_query_is_wide_enough_to_pass_the_retention_test(self):
        # Architecture 28 requires two shared terms. A single-word query can
        # never satisfy it, so every consult would be refused for a reason that
        # has nothing to do with the material it found.
        self.grant("external")
        self.assertGreaterEqual(len(set(self.observe()["query_terms"])), 2)


class ConsultActionTests(ConsultFixture):
    """Carrying the decision out."""

    def test_a_consult_files_material_for_curation_rather_than_knowing_it(self):
        self.grant("external")
        observation = autonomy.observe(self.config, candidates=[])
        outcome = autonomy.act(self.config, autonomy.decide(self.config, observation))
        self.assertEqual(outcome["retained"], 1)
        filed = Path(outcome["kept"][0])
        self.assertTrue(filed.exists())
        self.assertIn("acceptance gate", filed.read_text(encoding="utf-8"))
        # Architecture 28: nothing external becomes permanent internal knowledge
        # on the strength of having been fetched.
        self.assertEqual(
            self.knowledge.all(concepts.CONCEPTS)[0]["identifier"], "valve"
        )

    def test_an_ungranted_consult_reports_the_refusal_and_fetches_nothing(self):
        outcome = autonomy.act(
            self.config,
            autonomy.Decision(
                autonomy.CONSULT, "test", evidence={"query_terms": ["valve"]}
            ),
        )
        self.assertTrue(outcome["denied"])
        self.assertFalse(outcome["consulted"])
        self.assertFalse((self.workdir / "out" / "external").exists())

    def test_a_world_with_nothing_to_say_is_reported_as_such(self):
        self.grant("external")
        self.config.external_sources = (
            {"kind": "file", "name": "docs", "location": str(self.workdir / "empty")},
        )
        (self.workdir / "empty").mkdir()
        observation = autonomy.observe(self.config, candidates=[])
        outcome = autonomy.act(self.config, autonomy.decide(self.config, observation))
        self.assertEqual(outcome["results"], 0)
        self.assertEqual(outcome["retained"], 0)
        self.assertTrue(outcome["consulted"])

    def test_the_consult_record_is_written(self):
        self.grant("external")
        _, decision, _observation = autonomy.step(
            self.config, candidates=[], tel=None, runner=None
        )
        state = autonomy.load(self.config)
        self.assertEqual(state["last_decision"]["action"], autonomy.CONSULT)
        self.assertEqual(len(state["history"]), 1)

    def test_a_consult_that_found_nothing_counts_against_the_bound(self):
        # The bound has to mean three fruitless attempts, not three passes. A
        # consult that filed material resets it, because that one worked.
        self.grant("external")
        self.config.external_sources = (
            {"kind": "file", "name": "docs", "location": str(self.workdir / "empty")},
        )
        (self.workdir / "empty").mkdir()
        for _ in range(2):
            autonomy.step(self.config, candidates=[], runner=None)
        self.assertEqual(autonomy.load(self.config)["consecutive_consults"], 2)

    def test_the_control_decision_is_recorded_in_the_audit_log(self):
        self.grant("external")
        autonomy.step(self.config, candidates=[], runner=None)
        log = Path(self.workdir / "out" / "control_decisions.jsonl")
        entries = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(any(entry["domain"] == "external" for entry in entries))
