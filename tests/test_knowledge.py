"""Unit tests for the knowledge layer.

Covers Architecture 10 (concepts), 11 (concept census), 12 (relationships),
14 (conflict preservation), 15 (memory classes) and 16 (memory versus weights).
"""

import unittest

import concepts
import evaluation
import relationships
import store as store_module

import memory
from tests.support import TempDirTestCase, make_config


class ConflictContextTests(TempDirTestCase):
    """Architecture 14 steps 6 and 7: is this conflict contextual or real?

    The document says some apparent conflicts are actually context differences,
    and that the system should determine which. It could not, because a
    contradiction was recorded as a bare count with no record of who caused it. A
    rejected document that happened to contain a term therefore looked identical
    to one source disagreeing with another, and since nothing could tell them
    apart, nothing was confident enough to open a CONTRADICTS edge. The edge
    machinery has existed and been tested the whole time, unreachable.

    The distinction is who is on each side. A source contradicting itself across
    two documents is a context difference. A source contradicting a different one
    is a genuine disagreement.
    """

    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def supported_then_contradicted(self, supporter, contradictor, term="widget"):
        concepts.observe(
            self.store,
            term,
            delta_evidence=1,
            confidence_delta=0.08,
            source=supporter,
            experience="e1",
            cycle=1,
        )
        concepts.observe(
            self.store,
            term,
            delta_contradictions=1,
            confidence_delta=-0.04,
            source=supporter,
            experience="e1",
            cycle=1,
            contradicting_source=contradictor,
        )
        return concepts.get(self.store, term)

    def test_two_sources_disagreeing_is_a_genuine_conflict(self):
        record = self.supported_then_contradicted("doc-a.txt", "doc-b.txt")
        genuine, supporter = concepts.genuine_conflict(record)
        self.assertTrue(genuine, "a source contradicting another is genuine")
        self.assertEqual(supporter, "doc-a.txt")

    def test_a_source_contradicting_itself_is_contextual(self):
        concepts.observe(
            self.store,
            "widget",
            delta_evidence=1,
            source="doc-a.txt",
            experience="e1",
            cycle=1,
        )
        concepts.observe(
            self.store,
            "widget",
            delta_contradictions=1,
            source="doc-a.txt",
            experience="e2",
            cycle=2,
            contradicting_source="doc-a.txt",
        )
        record = concepts.get(self.store, "widget")
        self.assertFalse(concepts.genuine_conflict(record)[0])

    def test_a_contradiction_with_no_other_source_is_not_a_conflict(self):
        concepts.observe(
            self.store,
            "widget",
            delta_contradictions=1,
            source="doc-b.txt",
            experience="e2",
            cycle=2,
            contradicting_source="doc-b.txt",
        )
        self.assertFalse(
            concepts.genuine_conflict(concepts.get(self.store, "widget"))[0]
        )

    def test_the_contradiction_count_is_still_recorded(self):
        # The count was always there and must not be lost; the name is an
        # addition to it, not a replacement for it.
        record = self.supported_then_contradicted("doc-a.txt", "doc-b.txt")
        self.assertEqual(record["contradictions"], 1)
        self.assertIn("doc-b.txt", record["contradicted_by"])

    def test_a_genuine_conflict_is_explained_in_the_concept(self):
        record = self.supported_then_contradicted("doc-a.txt", "doc-b.txt")
        entry = concepts.note_conflict(self.store, record)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["supporter"], "doc-a.txt")
        self.assertIn("doc-b.txt", entry["contradicted_by"])
        self.assertTrue(concepts.get(self.store, "widget")["conflicting_information"])

    def test_a_contextual_difference_is_not_recorded_as_a_conflict(self):
        concepts.observe(
            self.store,
            "widget",
            delta_evidence=1,
            source="doc-a.txt",
            experience="e1",
            cycle=1,
        )
        concepts.observe(
            self.store,
            "widget",
            delta_contradictions=1,
            source="doc-a.txt",
            experience="e2",
            cycle=2,
            contradicting_source="doc-a.txt",
        )
        record = concepts.get(self.store, "widget")
        self.assertIsNone(concepts.note_conflict(self.store, record))
        self.assertEqual(
            concepts.get(self.store, "widget")["conflicting_information"], []
        )

    def test_opening_a_contradicts_edge_is_now_reachable(self):
        # The edge machinery has always existed; nothing could reach it.
        edge = relationships.observe(
            self.store,
            "widget",
            relationships.CONTRADICTS,
            "doc-b.txt",
            supporting=False,
        )
        self.assertEqual(edge["relation"], relationships.CONTRADICTS)
        self.assertEqual(edge["contradictions"], 1)

    def test_a_new_concept_declares_the_field_the_decision_reads(self):
        self.assertIn("contradicted_by", concepts.new_concept("widget"))


class StoreBusyTests(TempDirTestCase):
    """A store held open by another process is busy, not corrupt.

    The write path is atomic, so on Windows a read can still collide with an
    in-flight os.replace. That was reported as StoreCorrupt, which is wrong
    twice: it says the knowledge is damaged when it is not, and it is raised
    with no recovery, so a running learning worker took the entry point down
    with a message that sent whoever read it hunting for corruption.
    """

    def setUp(self):
        super().setUp()
        self.path = self.workdir / "knowledge.json"
        self.path.write_text('{"memory": {"r1": {"text": "kept"}}}', encoding="utf-8")
        self.store = store_module.Store(self.path)

    def hold(self, error):
        """Make every read of this store raise `error`."""
        import pathlib

        original = pathlib.Path.read_text
        attempts = {"n": 0}
        # Captured outside the hook: once a function is assigned to the class
        # its own `self` is the Path, not the test case.
        name = self.path.name

        def locked(target, *args, **kwargs):
            if target.name == name:
                attempts["n"] += 1
                raise error
            return original(target, *args, **kwargs)

        pathlib.Path.read_text = locked
        self.addCleanup(setattr, pathlib.Path, "read_text", original)
        return attempts

    def test_a_held_store_reports_busy_rather_than_corrupt(self):
        self.hold(PermissionError(13, "file in use"))
        with self.assertRaises(store_module.StoreBusy) as caught:
            self.store.all("memory")
        self.assertNotIsInstance(caught.exception, store_module.StoreCorrupt)

    def test_a_busy_message_says_the_file_is_not_damaged(self):
        self.hold(PermissionError(13, "file in use"))
        with self.assertRaises(store_module.StoreBusy) as caught:
            self.store.all("memory")
        self.assertIn("not damaged", str(caught.exception))

    def test_a_busy_store_is_retried_before_it_is_reported(self):
        attempts = self.hold(PermissionError(13, "file in use"))
        with self.assertRaises(store_module.StoreBusy):
            self.store.all("memory")
        self.assertEqual(attempts["n"], store_module.READ_ATTEMPTS)

    def test_a_read_that_succeeds_on_retry_never_reports_busy(self):
        # The point of retrying: a collision lasting one os.replace is invisible.
        import pathlib

        original = pathlib.Path.read_text
        seen = {"n": 0}
        name = self.path.name

        def flaky(target, *args, **kwargs):
            if target.name == name:
                seen["n"] += 1
                if seen["n"] < 3:
                    raise PermissionError(13, "file in use")
            return original(target, *args, **kwargs)

        pathlib.Path.read_text = flaky
        self.addCleanup(setattr, pathlib.Path, "read_text", original)
        self.assertEqual(self.store.all("memory"), [{"text": "kept"}])

    def test_a_genuine_io_failure_is_still_reported_as_corruption(self):
        # Retrying a real fault would hide it, so only PermissionError is
        # retried. Everything else raises immediately.
        self.hold(OSError(5, "input/output error"))
        with self.assertRaises(store_module.StoreCorrupt):
            self.store.all("memory")

    def test_unparseable_json_is_still_corruption(self):
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(store_module.StoreCorrupt):
            self.store.all("memory")


class StoreWriteBusyTests(TempDirTestCase):
    """A write that collides with a reader is retried, for the same reason.

    The read path above was hardened for a Windows sharing violation and the
    write path was not. That asymmetry is what let a reader with no intention of
    writing stop a learning cycle: PermissionError out of os.replace, straight
    out of run_jupiter.bat, with the store intact and nothing to repair.
    """

    def setUp(self):
        super().setUp()
        self.path = self.workdir / "knowledge.json"
        self.path.write_text('{"memory": {"r1": {"text": "kept"}}}', encoding="utf-8")
        self.store = store_module.Store(self.path)

    def hold_replace(self, error, fail_times=None):
        """Make os.replace raise, optionally only for the first N attempts."""
        import os as os_module

        original = os_module.replace
        seen = {"n": 0}

        def flaky(source, target, *args, **kwargs):
            seen["n"] += 1
            blocked = PermissionError(13, "file in use") if error is None else error
            if fail_times is None or seen["n"] <= fail_times:
                raise blocked
            return original(source, target, *args, **kwargs)

        store_module.os.replace = flaky
        self.addCleanup(setattr, store_module.os, "replace", original)
        return seen

    def test_a_write_that_loses_the_race_once_still_succeeds(self):
        self.hold_replace(None, fail_times=1)
        self.store.put("memory", "r2", {"text": "new"})
        self.assertEqual(self.store.get("memory", "r2"), {"text": "new"})

    def test_a_write_blocked_for_every_attempt_reports_busy_not_corruption(self):
        self.hold_replace(None)
        with self.assertRaises(store_module.StoreBusy) as caught:
            self.store.put("memory", "r2", {"text": "new"})
        self.assertNotIsInstance(caught.exception, store_module.StoreCorrupt)
        self.assertIn("not damaged", str(caught.exception))

    def test_a_write_that_keeps_failing_leaves_the_previous_document_intact(self):
        # The whole point of the temporary file: a refused write must not cost
        # the store its previous contents.
        self.hold_replace(None)
        with self.assertRaises(store_module.StoreBusy):
            self.store.put("memory", "r2", {"text": "new"})
        self.assertIn("kept", self.path.read_text(encoding="utf-8"))

    def test_a_failed_write_leaves_no_temporary_file_behind(self):
        self.hold_replace(None)
        with self.assertRaises(store_module.StoreBusy):
            self.store.put("memory", "r2", {"text": "new"})
        leftovers = list(self.workdir.glob("knowledge.json.*.tmp"))
        self.assertEqual(leftovers, [], f"left behind: {leftovers}")

    def test_a_genuine_write_fault_is_not_retried_as_if_it_were_busy(self):
        import os as os_module

        original = os_module.replace
        seen = {"n": 0}

        def faulty(source, target, *args, **kwargs):
            seen["n"] += 1
            raise OSError(5, "input/output error")

        store_module.os.replace = faulty
        self.addCleanup(setattr, store_module.os, "replace", original)
        with self.assertRaises(OSError):
            self.store.put("memory", "r2", {"text": "new"})
        self.assertEqual(seen["n"], 1, "a real fault must not be retried")


class PutManyTests(TempDirTestCase):
    """Batching writes many records in one pass over the store.

    put() rewrites the entire document, which is right for one record and
    compounding for a batch. Relationship derivation writes an edge at a time,
    so a store holding twelve hundred of them turned one learning cycle into
    roughly twelve hundred full rewrites of a multi-megabyte file.
    """

    def setUp(self):
        super().setUp()
        self.path = self.workdir / "knowledge.json"
        self.store = store_module.Store(self.path)

    def test_a_batch_persists_every_record(self):
        items = [(f"k{index}", {"n": index}) for index in range(50)]
        self.store.put_many("memory", items)
        self.assertEqual(len(self.store.all("memory")), 50)
        for index in range(50):
            self.assertEqual(self.store.get("memory", f"k{index}"), {"n": index})

    def test_a_batch_returns_what_it_was_given(self):
        returned = self.store.put_many("memory", [("a", {"x": 1}), ("b", {"x": 2})])
        self.assertEqual(returned, [{"x": 1}, {"x": 2}])

    def test_an_empty_batch_writes_nothing(self):
        import os as os_module

        original = os_module.replace
        seen = {"n": 0}

        def counting(source, target, *args, **kwargs):
            seen["n"] += 1
            return original(source, target, *args, **kwargs)

        store_module.os.replace = counting
        self.addCleanup(setattr, store_module.os, "replace", original)
        self.assertEqual(self.store.put_many("memory", []), [])
        self.assertEqual(seen["n"], 0)

    def test_a_batch_survives_a_reopened_store(self):
        # What a later session sees is the only test that matters for a store.
        self.store.put_many("memory", [(f"k{i}", {"n": i}) for i in range(20)])
        reopened = store_module.Store(self.path)
        self.assertEqual(len(reopened.all("memory")), 20)

    def test_a_batch_replaces_rather_than_duplicates(self):
        self.store.put("memory", "k0", {"n": "first"})
        self.store.put_many("memory", [("k0", {"n": "second"})])
        self.assertEqual(self.store.get("memory", "k0"), {"n": "second"})

    def test_put_is_still_a_single_record_put(self):
        # put() delegates to put_many, so the two cannot disagree about what a
        # record looks like on disk.
        self.assertEqual(self.store.put("memory", "k", {"n": 1}), {"n": 1})
        self.assertEqual(self.store.get("memory", "k"), {"n": 1})

    def test_keys_are_coerced_to_strings_as_put_does(self):
        self.store.put_many("memory", [(7, {"n": 7})])
        self.assertEqual(self.store.get("memory", "7"), {"n": 7})

    def test_a_batch_writes_the_store_once_not_once_per_record(self):
        import os as os_module

        original = os_module.replace
        seen = {"n": 0}

        def counting(source, target, *args, **kwargs):
            seen["n"] += 1
            return original(source, target, *args, **kwargs)

        store_module.os.replace = counting
        self.addCleanup(setattr, store_module.os, "replace", original)
        self.store.put_many("memory", [(f"k{i}", {"n": i}) for i in range(100)])
        self.assertEqual(seen["n"], 1)


class RelationshipWriteVolumeTests(TempDirTestCase):
    """Derivation must not scale its disk traffic with the size of the graph."""

    def counting_replace(self):
        import os as os_module

        original = os_module.replace
        seen = {"n": 0}

        def counting(source, target, *args, **kwargs):
            seen["n"] += 1
            return original(source, target, *args, **kwargs)

        store_module.os.replace = counting
        self.addCleanup(setattr, store_module.os, "replace", original)
        return seen

    def test_deriving_edges_does_not_rewrite_the_store_per_edge(self):
        path = self.workdir / "knowledge.json"
        store = store_module.Store(path)
        terms = [f"term{index}" for index in range(12)]
        # Enough shared sources that SIMILAR_TO is evidenced rather than refused.
        for round_index in range(6):
            relationships.record_co_occurrence(store, terms, now=1000.0 + round_index)
        seen = self.counting_replace()
        relationships.derive_graded_relationships(store, now=2000.0)
        # The defect was one full rewrite per derived edge, plus two per edge for
        # the concept lists. A bound rather than an exact count, because the
        # derivation refuses edges the evidence does not support.
        self.assertLessEqual(
            seen["n"], 12, f"derivation performed {seen['n']} store writes"
        )


class StoreTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_put_get_round_trip(self):
        self.store.put("c", "k", {"value": 1})
        self.assertEqual(self.store.get("c", "k"), {"value": 1})

    def test_missing_record_reads_as_none(self):
        self.assertIsNone(self.store.get("c", "absent"))

    def test_all_is_ordered_by_key(self):
        for key in ("b", "a", "c"):
            self.store.put("c", key, {"key": key})
        self.assertEqual(
            [record["key"] for record in self.store.all("c")],
            ["a", "b", "c"],
        )

    def test_clear_reports_the_count_and_empties_the_collection(self):
        self.store.put("c", "a", {})
        self.store.put("c", "b", {})
        self.assertEqual(self.store.clear("c"), 2)
        self.assertEqual(self.store.all("c"), [])

    def test_counts_reports_every_collection(self):
        self.store.put("c", "a", {})
        self.store.put("d", "a", {})
        self.assertEqual(self.store.counts(), {"c": 1, "d": 1})

    def test_state_survives_reopening_the_file(self):
        self.store.put("c", "k", {"v": 1})
        reopened = store_module.Store(self.workdir / "knowledge.json")
        self.assertEqual(reopened.get("c", "k"), {"v": 1})

    def test_unreadable_json_raises_instead_of_starting_empty(self):
        (self.workdir / "knowledge.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(store_module.StoreCorrupt):
            self.store.get("c", "k")

    def test_non_object_document_raises(self):
        (self.workdir / "knowledge.json").write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(store_module.StoreCorrupt):
            self.store.get("c", "k")

    def test_non_object_collection_raises(self):
        (self.workdir / "knowledge.json").write_text('{"c": 5}', encoding="utf-8")
        with self.assertRaises(store_module.StoreCorrupt):
            self.store.get("c", "k")


class ConceptTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_new_concept_carries_every_architecture_10_field(self):
        record = concepts.new_concept("Rust")
        for field in (
            "identifier",
            "description",
            "confidence",
            "maturity",
            "relationships",
            "prerequisites",
            "supporting_experiences",
            "conflicting_information",
            "usage_frequency",
            "last_verified",
        ):
            self.assertIn(field, record)

    def test_identifier_is_normalized_so_casing_cannot_fork_it(self):
        self.assertEqual(
            concepts.new_concept("  Own  ErRor ")["identifier"],
            "own error",
        )

    def test_maturity_ladder(self):
        self.assertEqual(
            concepts.maturity_of({"evidence": 0}),
            concepts.MATURITY_NEW,
        )
        self.assertEqual(
            concepts.maturity_of({"evidence": 1, "confidence": 0.0}),
            concepts.MATURITY_EMERGING,
        )
        self.assertEqual(
            concepts.maturity_of({"evidence": 3, "confidence": 0.5}),
            concepts.MATURITY_ESTABLISHED,
        )
        self.assertEqual(
            concepts.maturity_of(
                {"evidence": 8, "confidence": 0.8, "contradictions": 0}
            ),
            concepts.MATURITY_STABLE,
        )
        self.assertEqual(
            concepts.maturity_of(
                {"evidence": 9, "confidence": 0.2, "contradictions": 3}
            ),
            concepts.MATURITY_OBSOLETE,
        )

    def test_observe_creates_then_accumulates_evidence(self):
        first = concepts.observe(
            self.store, "Recursion", delta_evidence=1, confidence_delta=0.2
        )
        self.assertEqual(first["evidence"], 1)
        second = concepts.observe(
            self.store, "Recursion", delta_evidence=1, confidence_delta=0.2
        )
        self.assertEqual(second["evidence"], 2)
        self.assertAlmostEqual(second["confidence"], 0.4)

    def test_observe_counts_contradictions(self):
        record = concepts.observe(
            self.store,
            "Gravity",
            delta_evidence=1,
            delta_contradictions=1,
            confidence_delta=0.1,
        )
        self.assertEqual(record["contradictions"], 1)

    def test_confidence_is_clamped_into_range(self):
        self.assertEqual(
            concepts.observe(self.store, "Clamp", confidence_delta=5.0)["confidence"],
            1.0,
        )
        self.assertEqual(
            concepts.observe(self.store, "Clamp", confidence_delta=-5.0)["confidence"],
            0.0,
        )

    def test_candidate_terms_drop_stopwords_and_single_sightings(self):
        terms = concepts.candidate_terms(
            "the database and the database matter but mystery", limit=10, min_count=2
        )
        self.assertEqual(terms, ["database"])

    def test_census_counts_useful_concepts(self):
        for name in ("alpha", "beta", "gamma"):
            concepts.observe(self.store, name, delta_evidence=4, confidence_delta=0.6)
        report = concepts.census(self.store)
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["useful"], 3)
        self.assertEqual(report["weak"], [])
        self.assertEqual(report["fragmented"], [])
        self.assertEqual(report["obsolete"], [])

    def test_census_flags_shared_descriptions_as_duplicates(self):
        for name in ("twin1", "twin2"):
            self.store.put(
                concepts.CONCEPTS,
                name,
                concepts.new_concept(
                    name,
                    description="the same idea",
                    evidence=4,
                    confidence=0.6,
                    now=1.0,
                ),
            )
        self.assertEqual(concepts.census(self.store)["duplicate"], 2)

    def test_census_reports_strong_connection_only_with_relationships(self):
        for name in ("hub", "a", "b", "c"):
            concepts.observe(self.store, name, delta_evidence=1, confidence_delta=0.1)
        for leaf in ("a", "b", "c"):
            relationships.observe(self.store, "hub", relationships.PART_OF, leaf)
        self.assertIn(
            "hub", concepts.census(self.store, relationships)["strongly_connected"]
        )


class RelationshipTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_unknown_relation_type_is_rejected(self):
        with self.assertRaises(ValueError):
            relationships.record("a", "INVENTED", "b")

    def test_support_raises_confidence_and_evidence(self):
        first = relationships.observe(
            self.store, "Rust", relationships.USED_BY, "ownership"
        )
        self.assertEqual(first["evidence"], 1)
        second = relationships.observe(
            self.store, "Rust", relationships.USED_BY, "ownership"
        )
        self.assertEqual(second["evidence"], 2)
        self.assertAlmostEqual(second["confidence"], relationships.SUPPORT_GAIN)

    def test_contradiction_lowers_confidence_and_is_counted(self):
        relationships.observe(self.store, "a", relationships.PART_OF, "b")
        entry = relationships.observe(
            self.store, "a", relationships.PART_OF, "b", supporting=False
        )
        self.assertEqual(entry["contradictions"], 1)
        self.assertLess(entry["confidence"], 1.0)

    def test_contradiction_preserves_both_sides(self):
        relationships.observe(
            self.store, "alpha", relationships.IS_A, "beta", supporting=False
        )
        self.assertIsNotNone(
            relationships.get(self.store, "alpha", relationships.CONTRADICTS, "beta")
        )
        self.assertIsNotNone(
            relationships.get(self.store, "beta", relationships.CONTRADICTS, "alpha")
        )

    def test_relationship_is_registered_on_both_concepts(self):
        # The concepts exist first, as they do in the learning cycle: a term is
        # observed, which creates it, and the relationship is derived afterwards.
        # An edge whose endpoint has never been learned does not conjure the
        # concept into being, which is what used to let a curated filename into
        # the concept table as a sixty-eight character node with no evidence.
        concepts.observe(self.store, "http", delta_evidence=1)
        concepts.observe(self.store, "tcp", delta_evidence=1)
        relationships.observe(self.store, "http", relationships.DEPENDS_ON, "tcp")
        self.assertIn("tcp", concepts.get(self.store, "http")["prerequisites"])
        self.assertIn("http", concepts.get(self.store, "tcp")["prerequisites"])

    def test_outgoing_and_incoming_can_filter_by_type(self):
        relationships.observe(self.store, "a", relationships.CAUSES, "b")
        self.assertEqual(len(relationships.outgoing(self.store, "a")), 1)
        self.assertEqual(len(relationships.incoming(self.store, "b")), 1)
        self.assertEqual(
            relationships.outgoing(self.store, "a", relationships.IS_A), []
        )

    def test_prerequisite_traversal_terminates_on_a_cycle(self):
        relationships.observe(self.store, "a", relationships.DEPENDS_ON, "b")
        relationships.observe(self.store, "b", relationships.DEPENDS_ON, "a")
        found = relationships.prerequisites_of(self.store, "a")
        self.assertIn("b", found)
        self.assertIn("a", found)

    def test_census_counts_by_type(self):
        relationships.observe(self.store, "a", relationships.CAUSES, "b")
        relationships.observe(self.store, "a", relationships.PRECEDES, "b")
        report = relationships.census(self.store)
        self.assertEqual(report["by_type"][relationships.CAUSES], 1)
        self.assertEqual(report["by_type"][relationships.PRECEDES], 1)
        self.assertEqual(report["by_type"][relationships.IS_A], 0)


class CoOccurrenceTests(TempDirTestCase):
    """The one relationship Jupiter derives for itself (Architecture 12/53)."""

    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_the_new_type_is_part_of_the_vocabulary(self):
        self.assertIn(relationships.CO_OCCURS, relationships.TYPES)

    def test_a_pair_becomes_one_edge_not_two_directions(self):
        recorded = relationships.record_co_occurrence(self.store, ["alpha", "beta"])
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["relation"], relationships.CO_OCCURS)
        self.assertIsNotNone(
            relationships.get(self.store, "alpha", relationships.CO_OCCURS, "beta")
        )
        self.assertIsNone(
            relationships.get(self.store, "beta", relationships.CO_OCCURS, "alpha")
        )

    def test_pair_order_does_not_matter(self):
        relationships.record_co_occurrence(self.store, ["beta", "alpha"])
        again = relationships.record_co_occurrence(self.store, ["alpha", "beta"])
        self.assertEqual(again[0]["evidence"], 2)

    def test_three_terms_give_three_pairs(self):
        self.assertEqual(
            len(relationships.record_co_occurrence(self.store, ["a", "b", "c"])), 3
        )

    def test_a_repeated_term_creates_no_self_edge(self):
        recorded = relationships.record_co_occurrence(self.store, ["a", "a", "b"])
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["from"], "a")
        self.assertEqual(recorded[0]["to"], "b")

    def test_repeated_co_occurrence_strengthens_the_relationship(self):
        first = relationships.record_co_occurrence(self.store, ["a", "b"])[0]
        second = relationships.record_co_occurrence(self.store, ["a", "b"])[0]
        self.assertEqual(second["evidence"], 2)
        self.assertGreater(second["confidence"], first["confidence"])

    def test_the_edge_is_registered_on_both_concepts(self):
        # Observed before the relationship is derived, which is the order the
        # learning cycle uses and the order in which both endpoints are real
        # concepts rather than names an edge happened to mention.
        concepts.observe(self.store, "alpha", delta_evidence=1)
        concepts.observe(self.store, "beta", delta_evidence=1)
        relationships.record_co_occurrence(self.store, ["alpha", "beta"])
        edge = f"alpha|{relationships.CO_OCCURS}|beta"
        self.assertIn(edge, concepts.get(self.store, "alpha")["relationships"])
        self.assertIn(edge, concepts.get(self.store, "beta")["relationships"])

    def test_it_appears_in_the_relationship_census(self):
        relationships.record_co_occurrence(self.store, ["a", "b"])
        report = relationships.census(self.store)
        self.assertEqual(report["by_type"][relationships.CO_OCCURS], 1)

    def test_it_makes_a_hub_concept_strongly_connected(self):
        terms = ["hub", "a", "b", "c"]
        relationships.record_co_occurrence(self.store, terms)
        for name in terms:
            concepts.observe(self.store, name, delta_evidence=1, confidence_delta=0.1)
        self.assertIn(
            "hub", concepts.census(self.store, relationships)["strongly_connected"]
        )


class DerivedRelationshipTests(TempDirTestCase):
    """The relationship types the corpus can justify, derived from evidence.

    Architecture 12 names eleven types and ten of them had no code path that
    ever created one. The graph was therefore nine disconnected fragments of
    "these two words appeared together", and the enumeration in the document
    read as a description of the system when it was a list of intentions.

    The standard for every type here is the same: a relationship is recorded only
    when the corpus contains evidence for that specific relationship and not
    merely for the two concepts existing. Most of these tests are about what is
    NOT recorded, because that is where an inference goes wrong.
    """

    def setUp(self):
        super().setUp()
        self.store = store_module.Store(str(self.workdir / "knowledge.json"))

    def test_a_pair_seen_in_one_document_is_not_similar(self):
        # One document containing two terms says the document is short, not
        # that the terms are related.
        relationships.record_co_occurrence_from_source(
            self.store, ["alpha", "beta"], "only.txt"
        )
        derived = relationships.derive_graded_relationships(self.store)
        self.assertEqual(derived["SIMILAR_TO"], [])

    def test_terms_shared_across_documents_are_recorded_as_similar(self):
        for index in range(4):
            relationships.record_co_occurrence_from_source(
                self.store, ["alpha", "beta"], f"doc{index}.txt"
            )
        derived = relationships.derive_graded_relationships(self.store)
        self.assertEqual(len(derived["SIMILAR_TO"]), 1)
        edge = derived["SIMILAR_TO"][0]
        self.assertEqual({edge["from"], edge["to"]}, {"alpha", "beta"})
        self.assertLess(edge["confidence"], 1.0, "certainty is not available here")

    def test_unrelated_pairs_are_not_recorded_as_similar(self):
        relationships.record_co_occurrence_from_source(self.store, ["x", "y"], "a.txt")
        relationships.record_co_occurrence_from_source(self.store, ["p", "q"], "b.txt")
        derived = relationships.derive_graded_relationships(self.store)
        self.assertEqual(derived["SIMILAR_TO"], [])

    def test_a_consistent_order_is_recorded_as_precedes(self):
        for index in range(4):
            relationships.record_co_occurrence_from_source(
                self.store, ["alpha", "beta"], f"doc{index}.txt"
            )
        derived = relationships.derive_graded_relationships(self.store)
        pairs = [(edge["from"], edge["to"]) for edge in derived["PRECEDES"]]
        self.assertIn(("alpha", "beta"), pairs)

    def test_the_console_reports_how_many_types_are_really_in_use(self):
        # The point of measuring this at all: "eleven types" and "two in use"
        # are very different sentences about the same system.
        relationships.record_co_occurrence(self.store, ["a", "b"])
        report = evaluation.evaluate_relationships(
            make_config(knowledge_store=str(self.workdir / "knowledge.json"))
        )
        self.assertEqual(report["types_defined"], len(relationships.TYPES))
        self.assertEqual(report["types_present"], 1)
        self.assertIn(relationships.IS_A, report["types_unused"])
        self.assertIn("CO_OCCURS", report["by_type"])

    def test_an_order_that_reverses_is_not_recorded_at_all(self):
        # The corpus stating the order both ways is evidence that it does not
        # state an order. Drawing an arrow either way would be inventing a claim.
        relationships.record_co_occurrence_from_source(
            self.store, ["one", "two"], "a.txt"
        )
        relationships.record_co_occurrence_from_source(
            self.store, ["two", "one"], "b.txt"
        )
        derived = relationships.derive_graded_relationships(self.store)
        self.assertEqual(derived["PRECEDES"], [])

    def test_order_is_kept_per_source_so_a_reversal_is_visible(self):
        relationships.record_co_occurrence_from_source(
            self.store, ["one", "two"], "a.txt"
        )
        relationships.record_co_occurrence_from_source(
            self.store, ["two", "one"], "b.txt"
        )
        orders = {}
        for edge in relationships.all_relationships(self.store):
            if edge["relation"] == relationships.CO_OCCURS:
                orders.update(edge.get("orders") or {})
        self.assertEqual(orders.get("a.txt"), ["one", "two"])
        self.assertEqual(orders.get("b.txt"), ["two", "one"])

    def test_one_document_does_not_fix_an_order(self):
        relationships.record_co_occurrence_from_source(
            self.store, ["alpha", "beta"], "only.txt"
        )
        derived = relationships.derive_graded_relationships(self.store)
        self.assertEqual(derived["PRECEDES"], [])

    def test_sources_accumulate_rather_than_being_overwritten(self):
        # observe() folds a repeat sighting into one edge, so writing the source
        # again would leave a pair seen in four documents looking like one.
        for index in range(3):
            relationships.record_co_occurrence_from_source(
                self.store, ["alpha", "beta"], f"doc{index}.txt"
            )
        edge = relationships.get(self.store, "alpha", relationships.CO_OCCURS, "beta")
        self.assertEqual(len(edge.get("sources", [])), 3)

    def test_coverage_reports_what_exists_rather_than_what_is_named(self):
        # "Eleven types" and "two in use" are very different sentences, and the
        # project's own rule is that a feature listed as covered with nothing
        # behind it is worse than an absent one.
        relationships.record_co_occurrence(self.store, ["a", "b"])
        coverage = relationships.relationship_coverage(self.store)
        self.assertIn(relationships.CO_OCCURS, coverage["types_present"])
        self.assertIn(relationships.IS_A, coverage["types_unused"])
        self.assertEqual(
            set(coverage["types_present"]) | set(coverage["types_unused"]),
            set(coverage["types_defined"]),
        )


class ConceptsTestShapeTests(TempDirTestCase):
    """The store exposes only what callers use.

    Architecture 40 forbids dead code, and a store that cannot forget is a
    store whose missing operations get re-added by whoever needs them next,
    without a test. Writing a record is a full replace, and forgetting a whole
    memory class is a clear, which is what the four memory classes need.
    """

    def test_the_store_offers_exactly_the_operations_callers_use(self):
        for name in ("put", "get", "all", "clear", "counts", "collection"):
            self.assertTrue(callable(getattr(store_module.Store, name)), name)
        for name in ("update", "delete", "stamp"):
            self.assertFalse(
                hasattr(store_module.Store, name),
                f"{name} is not used by any caller and should not exist",
            )

    def test_writing_replaces_rather_than_merges(self):
        store = store_module.Store(self.workdir / "k.json")
        store.put("c", "k", {"a": 1, "b": 2})
        store.put("c", "k", {"a": 9})
        self.assertEqual(store.get("c", "k"), {"a": 9})


class MeasuredFactorTests(unittest.TestCase):
    """All nine Architecture 16 factors, each from an observation.

    An earlier version passed six of them as zero, on the grounds that they
    were not measurable. These pin down that every one of them now comes from
    something the system actually records, so "not measurable" cannot creep
    back in as an unexamined assumption.
    """

    def test_every_factor_can_move_the_decision(self):
        # The policy in decide_destination gates each factor on a threshold, so
        # a factor only counts once it crosses one. Each value below clears the
        # threshold it is testing.
        base = dict(observed=8, accepted=8, confidence=0.85, improvement=0.3)
        baseline = memory.route_for_source(**base)["score"]
        self.assertGreater(baseline, 0.0)
        for name, value in (
            ("regression_gain", 0.8),
            ("age_cycles", 30),
            ("retrievals", 9),
        ):
            raised = memory.route_for_source(**base, **{name: value})
            self.assertGreater(raised["score"], baseline, f"{name} had no effect")

    def test_importance_needs_both_signals_because_it_is_their_average(self):
        # importance is defined as half concepts and half retrievals, so neither
        # alone reaches the 0.7 threshold. That is a property worth pinning: a
        # concept the system never looks up is not important however many there
        # are.
        base = dict(observed=8, accepted=8, confidence=0.85, improvement=0.3)
        concepts_only = memory.route_for_source(**base, concepts=40)
        both = memory.route_for_source(**base, concepts=40, retrievals=9)
        self.assertEqual(
            concepts_only["score"],
            memory.route_for_source(**base)["score"],
            "concepts alone must not clear the importance threshold",
        )
        self.assertGreater(both["score"], concepts_only["score"])

    def test_measured_evidence_can_reach_the_weights(self):
        # A source that is frequently accepted, confident, generalizes, is
        # cheap to learn and cheap to retrieve, has been retained a long time
        # and is actually being used has earned the strongest destination.
        decision = memory.route_for_source(
            observed=12,
            accepted=12,
            contradictions=0,
            confidence=0.95,
            improvement=0.5,
            regression_gain=0.5,
            learning_cost=0.0,
            retrieval_cost=0.0,
            concepts=40,
            retrievals=9,
            age_cycles=30,
        )
        self.assertEqual(decision["destination"], memory.DEST_WEIGHTS)

    def test_generalization_is_measured_from_the_reserved_document(self):
        base = dict(observed=6, accepted=6, confidence=0.8, improvement=0.3)
        without = memory.route_for_source(**base)
        with_gain = memory.route_for_source(**base, regression_gain=0.8)
        self.assertGreater(with_gain["score"], without["score"])
        # Below the policy threshold it is a real measurement that does not yet
        # count, which is different from not measuring it at all.
        small_gain = memory.route_for_source(**base, regression_gain=0.2)
        self.assertEqual(small_gain["score"], without["score"])

    def test_expensive_learning_pushes_back_against_the_weights(self):
        base = dict(
            observed=10,
            accepted=10,
            confidence=0.9,
            improvement=0.4,
            concepts=40,
            retrievals=9,
            age_cycles=30,
            regression_gain=0.5,
        )
        cheap = memory.route_for_source(**base, learning_cost=0.0)
        costly = memory.route_for_source(**base, learning_cost=120.0)
        self.assertLess(costly["score"], cheap["score"])

    def test_contradiction_still_forces_external_however_good_the_source_is(self):
        decision = memory.route_for_source(
            observed=12,
            accepted=12,
            contradictions=8,
            confidence=0.99,
            improvement=0.9,
            regression_gain=0.9,
            concepts=99,
            retrievals=99,
            age_cycles=99,
        )
        self.assertEqual(decision["destination"], memory.DEST_EXTERNAL)

    def test_the_reference_values_are_named_so_they_can_be_argued_with(self):
        for name in (
            "CONCEPT_REFERENCE",
            "RETRIEVAL_REFERENCE",
            "RETRIEVAL_SECONDS_REFERENCE",
            "LEARNING_SECONDS_REFERENCE",
            "AGE_REFERENCE",
        ):
            self.assertGreater(getattr(memory, name), 0.0, name)

    def test_nothing_unmeasured_is_left_at_zero_silently(self):
        # The decision must say which factors argued, so a reader can see that
        # a source stayed external because of measured evidence rather than
        # because nothing was passed in.
        decision = memory.route_for_source(observed=1, accepted=1, confidence=0.0)
        self.assertTrue(decision["reasons"])

    def test_improvement_moves_the_decision(self):
        # It was accepted as an argument and then discarded, so a source that
        # consistently taught nothing and one that taught a great deal routed
        # identically. That is the difference between a routing decision and a
        # popularity contest.
        base = dict(
            observed=8,
            accepted=8,
            confidence=0.6,
            concepts=20,
            retrievals=5,
            age_cycles=20,
            regression_gain=0.0,
        )
        learned_nothing = memory.route_for_source(**base, improvement=0.0)
        learned_a_lot = memory.route_for_source(**base, improvement=0.95)
        self.assertGreater(
            learned_a_lot["score"],
            learned_nothing["score"],
            "a source that taught the model more must score higher",
        )

    def test_every_declared_destination_is_reachable(self):
        # Three of the six were named in the document and had no code path that
        # could ever produce them, which made the enumeration a list of
        # intentions. A destination nothing can return is not a destination.
        #
        # The sweep varies every factor rather than only the contradiction risk,
        # because the destinations are not a ladder on one axis: a concept needs
        # a middling score and an experience needs a high frequency with a
        # middling one, so a sweep that held everything else maximal would find
        # only the two strongest and conclude the rest were unreachable.
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
        self.assertEqual(
            set(memory.DESTINATIONS) - found,
            set(),
            "a destination nothing can produce is a declaration, not a feature",
        )

    def test_a_disputed_but_supported_fact_becomes_a_relationship(self):
        # Architecture 16 lists becoming a relationship as a destination, and a
        # disagreement is the case that suits it: a concept would have to pick
        # one of the two claims, and a relationship records that both exist.
        decision = memory.decide_destination(
            frequency=10,
            importance=0.75,
            confidence=0.85,
            generalization=0.65,
            permanence=0.75,
            usefulness=0.0,
            contradiction_risk=0.5,
        )
        self.assertEqual(decision["destination"], memory.DEST_RELATIONSHIP)

    def test_supported_and_disputed_at_once_is_held_in_both_places(self):
        decision = memory.decide_destination(
            frequency=10,
            importance=0.95,
            confidence=0.95,
            generalization=0.9,
            permanence=0.95,
            usefulness=0.95,
            contradiction_risk=0.1,
        )
        self.assertEqual(decision["destination"], memory.DEST_MULTIPLE)


class MemoryTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = store_module.Store(self.workdir / "knowledge.json")

    def test_all_four_architecture_15_classes_are_usable(self):
        memory.remember_working(self.store, "w", "current task")
        memory.remember_long_term(self.store, "l", "stable fact", confidence=0.9)
        memory.remember_external(self.store, "e", "borrowed", origin="file.txt")
        memory.record_experience(self.store, "x", {"decision": "ACCEPT"})
        counts = memory.census(self.store)
        self.assertEqual(counts[memory.WORKING], 1)
        self.assertEqual(counts[memory.LONG_TERM], 1)
        self.assertEqual(counts[memory.EXTERNAL], 1)
        self.assertEqual(counts[memory.EXPERIENCE], 1)

    def test_external_memory_always_records_where_it_came_from(self):
        stored = memory.remember_external(self.store, "e", "text", origin="mcp")
        self.assertEqual(stored["origin"], "mcp")

    def test_long_term_confidence_is_clamped(self):
        self.assertEqual(
            memory.remember_long_term(self.store, "l", "x", confidence=4.0)[
                "confidence"
            ],
            1.0,
        )

    def test_clearing_working_memory_leaves_durable_memory_alone(self):
        memory.remember_working(self.store, "w", "x")
        memory.remember_long_term(self.store, "l", "y")
        self.assertEqual(memory.clear_working(self.store), 1)
        self.assertEqual(memory.working(self.store), [])
        self.assertEqual(len(memory.long_term(self.store)), 1)

    def test_experiences_can_be_limited_to_the_most_recent(self):
        for index in range(3):
            memory.record_experience(self.store, f"c{index}", {"index": index})
        self.assertEqual(len(memory.experiences(self.store)), 3)
        self.assertEqual(len(memory.experiences(self.store, limit=2)), 2)
        self.assertEqual(memory.experiences(self.store, limit=1)[0]["index"], 2)


class DestinationTests(unittest.TestCase):
    """Architecture 16: where should this information live?"""

    def test_default_keeps_information_external(self):
        decision = memory.decide_destination()
        self.assertEqual(decision["destination"], memory.DEST_EXTERNAL)
        self.assertTrue(
            any("external" in reason for reason in decision["reasons"]),
            decision["reasons"],
        )

    def test_high_contradiction_risk_forces_external(self):
        decision = memory.decide_destination(
            frequency=100,
            importance=1.0,
            confidence=1.0,
            generalization=1.0,
            permanence=1.0,
            usefulness=1.0,
            contradiction_risk=0.9,
        )
        self.assertEqual(decision["destination"], memory.DEST_EXTERNAL)

    def test_strong_evidence_reaches_the_weights(self):
        decision = memory.decide_destination(
            frequency=10,
            importance=0.9,
            confidence=0.9,
            generalization=0.8,
            permanence=0.9,
            usefulness=0.9,
        )
        self.assertEqual(decision["destination"], memory.DEST_WEIGHTS)

    def test_moderate_evidence_becomes_an_explicit_concept(self):
        decision = memory.decide_destination(
            importance=0.7, confidence=0.8, usefulness=0.7
        )
        self.assertEqual(decision["destination"], memory.DEST_CONCEPT)

    def test_usefulness_outside_the_unit_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            memory.decide_destination(usefulness=1.5)


class StoreCacheTests(TempDirTestCase):
    """The parsed document is reused, and only while the file still matches.

    The learning cycle reads the same store about a hundred and sixty times per
    cycle and every read was a full parse of the whole document, which is where
    most of a cycle's wall clock went. The cache is only safe if another
    process writing the store is noticed, so that is what these check.
    """

    def setUp(self):
        super().setUp()
        self.path = self.workdir / "knowledge.json"
        self.store = store_module.Store(self.path)

    def test_an_unchanged_file_is_parsed_once(self):
        self.store.put("memory", "r1", {"text": "kept"})
        first = self.store._cache
        self.store.collection("concepts")
        self.store.all("memory")
        self.assertIs(self.store._cache, first)

    def test_a_second_reader_sees_the_first_readers_writes(self):
        self.store.put("memory", "r1", {"text": "kept"})
        other = store_module.Store(self.path)
        self.assertEqual(other.get("memory", "r1"), {"text": "kept"})

    def test_a_write_by_another_process_is_noticed(self):
        # The whole reason the cache is keyed on a stat rather than trusted.
        self.store.put("memory", "r1", {"text": "first"})
        other = store_module.Store(self.path)
        other.put("memory", "r2", {"text": "second"})
        self.assertEqual(
            self.store.get("memory", "r2"),
            {"text": "second"},
            "a cache that cannot see another process's write is worse than no cache",
        )

    def test_a_deletion_by_another_process_is_noticed(self):
        self.store.put("memory", "r1", {"text": "kept"})
        self.path.write_text("{}", encoding="utf-8")
        self.assertEqual(self.store.all("memory"), [])

    def test_a_file_that_becomes_empty_reads_as_empty(self):
        self.store.put("memory", "r1", {"text": "kept"})
        self.path.write_text("", encoding="utf-8")
        self.assertEqual(self.store.all("memory"), [])

    def test_a_corrupt_file_is_still_reported(self):
        # A cache must not turn a parse failure into a stale success.
        self.store.put("memory", "r1", {"text": "kept"})
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(store_module.StoreCorrupt):
            self.store.all("memory")
