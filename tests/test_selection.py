"""Tests for choosing what to learn next, and for writing its own material.

The two properties worth protecting are the ones that fail quietly.

A priority queue can starve. If frustration alone decided the order, the one
document the model was worst at would be chosen forever and the rest of the
corpus would never be revisited, and nothing would say so: the run would look
busy. The age term is what prevents that, and it has to be tested by visiting
the corpus rather than by reading the formula.

A self-feeding corpus can reinforce itself. The gate is on the cause of the
frustration rather than on its size, because a model that cannot represent
something does not start representing it by being shown more of the same thing.
Getting that gate wrong is not a performance problem, it is a corpus that
quietly learns to be worse.
"""

import unittest
from pathlib import Path

import selection
import synthesize

from tests.support import TempDirTestCase, make_config

A = "doc-a.txt"
B = "doc-b.txt"
C = "doc-c.txt"

PIECES = [(A, "alpha"), (B, "beta"), (C, "gamma")]


def signal(band, confidence):
    return {"band": band, "confidence": confidence}


class NeedTests(unittest.TestCase):
    def test_higher_frustration_raises_need(self):
        self.assertGreater(
            selection.need_for("HIGH", 0.5),
            selection.need_for("LOW", 0.5),
        )

    def test_lower_confidence_raises_need(self):
        # Architecture 13: a source that has not earned much evidence has not
        # been shown to be learnable, which is a reason to look again.
        self.assertGreater(
            selection.need_for("MEDIUM", 0.1), selection.need_for("MEDIUM", 0.9)
        )

    def test_an_unusable_confidence_value_is_treated_as_none(self):
        self.assertEqual(
            selection.need_for("LOW", "not a number"),
            selection.need_for("LOW", 0.0),
        )

    def test_need_is_clamped_at_both_ends(self):
        self.assertGreaterEqual(selection.need_for("HIGH", -5.0), 0.0)
        self.assertLessEqual(
            selection.need_for("HIGH", 5.0), selection.need_for("HIGH", 0.0) + 1.0
        )


def selection_frustration_high():
    return "HIGH"


class SourceKeyTests(unittest.TestCase):
    """Sources are identified by bare filename, everywhere.

    The worker has always recorded confidence, experiments and frustration
    against `Path(source).name`, and candidates arrive as full paths. Joining
    the two on the path string matches nothing at all, and the failure is
    silent: the queue simply falls back to alphabetical order and keeps looking
    healthy. This is the exact bug that was live before it was caught.
    """

    def test_a_full_path_and_a_bare_name_are_the_same_source(self):
        self.assertEqual(
            selection.key_for("data/curated/doc-a.txt"),
            selection.key_for("doc-a.txt"),
        )

    def test_a_windows_path_and_a_posix_path_are_the_same_source(self):
        self.assertEqual(
            selection.key_for(r"data\curated\doc-a.txt"),
            selection.key_for("data/curated/doc-a.txt"),
        )

    def test_signals_keyed_by_name_match_candidates_keyed_by_path(self):
        pieces = [("data/curated/doc-a.txt", "a"), ("data/curated/doc-b.txt", "b")]
        signals = {"doc-b.txt": {"band": "HIGH", "confidence": 0.0}}
        ordered = [p for p, _t in selection.rank(pieces, signals, {}, 1)]
        self.assertEqual(
            ordered[0],
            "data/curated/doc-b.txt",
            "a signal recorded under the bare name must reach the candidate "
            "carrying the full path",
        )

    def test_visit_history_survives_the_same_mismatch(self):
        state = {}
        selection.mark_seen(state, "data/curated/doc-a.txt", 4)
        self.assertEqual(selection.seen_cycle(state, "doc-a.txt"), 4)


class RankingTests(unittest.TestCase):
    def test_an_unjudged_corpus_still_rotates(self):
        # No signals at all is the startup state, and it must not collapse the
        # queue onto one document.
        state = {}
        seen = []
        for cycle in range(1, 13):
            _path, _text, _why = selection.choose(PIECES, {}, state, cycle)
            selection.mark_seen(state, _path, cycle)
            seen.append(_path)
        self.assertEqual(len(set(seen)), 3, f"corpus did not rotate: {seen}")

    def test_a_difficult_source_is_preferred_within_a_rotation(self):
        signals = {
            A: signal("LOW", 0.9),
            B: signal("LOW", 0.9),
            C: signal("HIGH", 0.1),
        }
        state = {}
        chosen = []
        for cycle in range(1, 7):
            path, _text, _why = selection.choose(PIECES, signals, state, cycle)
            selection.mark_seen(state, path, cycle)
            chosen.append(path)
        self.assertEqual(
            chosen[0], C, f"the difficult source should go first: {chosen}"
        )
        self.assertEqual(chosen[1], C, f"and again before a full rotation: {chosen}")

    def test_a_difficult_source_does_not_starve_the_rest(self):
        # The failure this exists to prevent: one permanently bad document
        # chosen every cycle forever.
        signals = {
            A: signal("HIGH", 0.0),
            B: signal("HIGH", 0.0),
            C: signal("HIGH", 0.0),
        }
        state = {}
        chosen = []
        for cycle in range(1, 10):
            path, _text, _why = selection.choose(PIECES, signals, state, cycle)
            selection.mark_seen(state, path, cycle)
            chosen.append(path)
        self.assertEqual(
            len(set(chosen)),
            3,
            f"every source is equally difficult, so all must be visited: {chosen}",
        )

    def test_the_order_is_deterministic(self):
        # Architecture 29 asks for reproducible experiments. A queue that
        # reshuffles between identical runs makes two runs incomparable.
        signals = {A: signal("MEDIUM", 0.5), B: signal("LOW", 0.5)}
        first = [p for p, _t in selection.rank(PIECES, signals, {}, 1)]
        second = [p for p, _t in selection.rank(PIECES, signals, {}, 1)]
        self.assertEqual(first, second)

    def test_ties_break_on_the_path_so_the_order_is_total(self):
        signals = {A: signal("LOW", 0.5), B: signal("LOW", 0.5), C: signal("LOW", 0.5)}
        ordered = [p for p, _t in selection.rank(PIECES, signals, {}, 1)]
        self.assertEqual(ordered, [A, B, C])

    def test_choose_explains_itself(self):
        # An unrecorded curriculum is indistinguishable from a random walk when
        # someone reads the experiment log back.
        _path, _text, why = selection.choose(PIECES, {A: signal("HIGH", 0.1)}, {}, 1)
        self.assertIn("HIGH", why)
        self.assertIn("need", why)

    def test_choose_on_an_empty_corpus_says_so(self):
        path, text, why = selection.choose([], {}, {}, 1)
        self.assertIsNone(path)
        self.assertIsNone(text)
        self.assertIn("no candidate", why)

    def test_ranking_preserves_every_candidate(self):
        ordered = selection.rank(PIECES, {}, {}, 1)
        self.assertEqual(sorted(p for p, _t in ordered), sorted(p for p, _t in PIECES))


class ResumeStateTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(selection_state=str(self.workdir / "selection.json"))

    def test_state_round_trips(self):
        state = selection.load(self.config)
        selection.mark_seen(state, A, 7)
        selection.save(self.config, state)
        self.assertEqual(selection.seen_cycle(selection.load(self.config), A), 7)

    def test_an_unseen_source_reports_none(self):
        self.assertIsNone(selection.seen_cycle({}, A))

    def test_a_corrupt_state_file_does_not_stop_the_queue(self):
        Path(self.config.selection_state).write_text("{not json", encoding="utf-8")
        self.assertEqual(selection.load(self.config), {})

    def test_an_unreadable_cycle_value_is_treated_as_unseen(self):
        self.assertIsNone(selection.seen_cycle({A: {"cycle": "later"}}, A))


class GenerationGateTests(unittest.TestCase):
    def test_more_material_is_worth_writing_for_these_causes(self):
        # The two of Architecture 21's seven that more of the same thing can
        # actually answer.
        self.assertTrue(selection.worth_generating("inadequate context"))
        self.assertTrue(selection.worth_generating("insufficient data"))

    def test_more_material_is_not_worth_writing_for_these(self):
        # A representation the model cannot compress does not become
        # compressible by being shown more of it. Architecture 23's caution
        # about acting on a measurement applies to the cheap remedy too.
        for cause in (
            "poor representation",
            "insufficient model capacity",
            "conflicting knowledge",
            "tokenizer limitations",
            "inappropriate architecture",
        ):
            self.assertFalse(selection.worth_generating(cause), cause)

    def test_an_unknown_cause_is_not_worth_writing_for(self):
        self.assertFalse(selection.worth_generating("something new"))
        self.assertFalse(selection.worth_generating(""))


class TargetSelectionTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "source": "a",
                "likely_cause": "inadequate context",
                "index": 0.2,
            },
            {
                "source": "b",
                "likely_cause": "insufficient model capacity",
                "index": 0.9,
            },
            {
                "source": "c",
                "likely_cause": "inadequate context",
                "index": 0.7,
            },
        ]

    def test_only_eligible_causes_are_returned(self):
        chosen = synthesize.targets_for(self.rows())
        self.assertEqual([row["source"] for row in chosen], ["c", "a"])

    def test_most_frustrated_eligible_source_comes_first(self):
        chosen = synthesize.targets_for(self.rows())
        self.assertEqual(chosen[0]["source"], "c")

    def test_no_eligible_source_returns_nothing(self):
        self.assertEqual(
            synthesize.targets_for(
                [{"source": "b", "likely_cause": "poor representation", "index": 0.9}]
            ),
            [],
        )

    def test_an_empty_measurement_set_is_not_an_error(self):
        self.assertEqual(synthesize.targets_for([]), [])
        self.assertEqual(synthesize.targets_for(None), [])

    def test_the_cause_field_is_read_from_likely_cause(self):
        # frustration names the field likely_cause. Reading a field called
        # "cause" would find nothing and silently gate every source out, which
        # looks identical to a generator that is simply broken.
        rows = [{"source": "x", "cause": "inadequate context", "index": 0.5}]
        self.assertEqual(synthesize.targets_for(rows), [])


class UsabilityTests(unittest.TestCase):
    def test_long_printable_text_is_usable(self):
        self.assertTrue(synthesize.usable("a normal sentence. " * 40))

    def test_short_text_is_not_kept(self):
        # data/raw is never pruned, so junk written there has to be dealt with
        # by a person later. The check is on writing, not after.
        self.assertFalse(synthesize.usable("too short"))

    def test_mostly_unprintable_text_is_not_kept(self):
        self.assertFalse(synthesize.usable("\x00\x01\x02" * 200))

    def test_empty_text_is_not_usable(self):
        self.assertFalse(synthesize.usable(""))


class GeneratedNamingTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(synthesis_dir=str(self.workdir / "raw" / "generated"))

    def test_generated_documents_are_identifiable_by_name(self):
        self.assertTrue(
            synthesize.is_generated("generated-doc-a-c3-20260101T000000.txt")
        )
        self.assertFalse(synthesize.is_generated("doc-a.txt"))

    def test_a_written_document_records_its_seed_and_cycle(self):
        target = synthesize.write(self.config, "data/doc-a.txt", "body text", cycle=3)
        name = target.name
        self.assertTrue(name.startswith("generated-doc-a-c3-"))
        self.assertTrue(target.exists())

    def test_two_documents_from_one_seed_do_not_collide(self):
        first = synthesize.write(self.config, "data/doc-a.txt", "one", cycle=1)
        second = synthesize.write(self.config, "data/doc-a.txt", "two", cycle=1)
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_documents_land_where_curation_will_find_them(self):
        target = synthesize.write(self.config, "data/doc-a.txt", "body", cycle=1)
        self.assertEqual(
            target.parent,
            Path(self.config.synthesis_dir),
        )
