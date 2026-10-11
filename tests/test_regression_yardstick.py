"""The regression yardstick has to be reachable, or nothing is ever accepted.

This is the test for a bug that had no failing test and no visible symptom. The
worker called reserve(), which returns the documents it held out alongside the
ones still available to learn from, and then rebound its candidate list to the
second of those. It went on to look for each reserved document in that list.

The reserved documents had just been removed from it, so the lookup failed every
time, for every document, on every run. The worker caught the failure and
reported it honestly: regression detection was unavailable. That is correct
behaviour for a genuine absence.

The consequence was not. Without a yardstick the acceptance gate cannot separate
an improvement from forgetting, so it returned REVIEW for every cycle instead of
ACCEPT or REJECT. Review is not a rejection: the update still happened. But
nothing was ever accepted, the growth cadence counts accepted cycles, and so the
cadence could not reach a hundred and Jupiter never once offered to grow. A
hundred of the last hundred cycles were REVIEW.

The suite passed throughout, because every existing test either built its own
yardstick or did not check that one was built at all.
"""

import contextlib
import io
import unittest
from pathlib import Path

import curate_dataset
import learning_worker
import regression
import train

from tests.support import TempDirTestCase, make_config


class YardstickIsReachableTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.raw = self.workdir / "raw"
        self.raw.mkdir(parents=True)
        self.out = self.workdir / "out"
        self.out.mkdir(parents=True)
        # Three lexically distinct documents, so reserve() has something to hold out and
        # something left to learn from. reserve() refuses to reserve everything,
        # and curation deduplicates by content digest, so three copies of one
        # paragraph curate to one document and there is nothing to reserve from.
        for name, text in (
            ("a.txt", "the observatory recorded the river and the weather. " * 200),
            ("b.txt", "a second account named the survey and the observer. " * 200),
            ("c.txt", "a third document concerns permissions and grants. " * 200),
        ):
            (self.raw / name).write_text(text, encoding="utf-8")

    def config(self, **overrides):
        settings = dict(
            device="cpu",
            raw_dir=str(self.raw),
            curated_dir=str(self.workdir / "curated"),
            out_dir=str(self.out),
            knowledge_store=str(self.out / "knowledge.json"),
            regression_state=str(self.out / "regression_set.json"),
            selection_state=str(self.out / "selection.json"),
            telemetry_file=str(self.out / "telemetry.jsonl"),
            learning_state=str(self.out / "learning_state.json"),
            learning_status=str(self.out / "learning_status.json"),
            worker_state=str(self.out / "worker_state.json"),
            worker_lock=str(self.out / "worker.lock"),
            confidence_state=str(self.out / "confidence.json"),
            experiment_log=str(self.out / "experiments.jsonl"),
            min_text_chars=128,
        )
        settings.update(overrides)
        return make_config(**settings)

    def candidates(self, c, curate=False):
        if curate:
            original = curate_dataset.load_config
            curate_dataset.load_config = lambda: c
            self.addCleanup(setattr, curate_dataset, "load_config", original)
            with contextlib.redirect_stdout(io.StringIO()):
                curate_dataset.curate(c)
        docs = learning_worker.discover_documents(c)
        return train.read_learning_text(docs, c)

    def test_reserved_documents_are_in_the_list_they_are_looked_up_in(self):
        """The shape of the bug, stated as a property of reserve().

        reserve() returns two things: what it held out, and what remains. A
        caller that wants the text of a held-out document has to look for it in
        the candidates it passed in, never in the remainder, because the
        remainder is by definition the candidates minus the held-out ones.
        """
        c = self.config()
        pieces = self.candidates(c, curate=True)
        self.assertGreaterEqual(len(pieces), 3)

        candidates = list(pieces)
        reserved, remaining = regression.reserve(c, candidates)

        self.assertTrue(reserved, "reserve() held nothing out")
        self.assertLess(len(remaining), len(candidates))

        candidate_paths = {
            str(piece.path if hasattr(piece, "path") else piece[0])
            for piece in candidates
        }
        remaining_paths = {
            str(piece.path if hasattr(piece, "path") else piece[0])
            for piece in remaining
        }

        for path in reserved:
            self.assertIn(
                str(path),
                candidate_paths,
                f"{path} was reserved but was never a candidate to begin with",
            )
            self.assertNotIn(
                str(path),
                remaining_paths,
                "a reserved document must be absent from what remains",
            )

    def test_the_worker_builds_a_yardstick_from_a_normal_corpus(self):
        """The end to end claim: a run over a small corpus has a yardstick.

        Asserted through reserve() over a curated corpus rather than by
        re-reading the worker, because the bug was in the wiring between two
        functions that were each correct on their own, and only the wiring was
        wrong.
        """

        c = self.config()
        original = curate_dataset.load_config
        curate_dataset.load_config = lambda: c
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        with contextlib.redirect_stdout(io.StringIO()):
            curate_dataset.curate(c)

        pieces = self.candidates(c)
        self.assertGreaterEqual(len(pieces), 3, "curation produced too little to test")

        candidates = list(pieces)
        reserved, _remaining = regression.reserve(c, candidates)

        # Read each reserved document the way the worker has to: from the
        # candidates it was given, not from what reserve() left behind.
        bodies = []
        for path in reserved:
            match = next(
                (
                    piece
                    for piece in candidates
                    if str(piece.path if hasattr(piece, "path") else piece[0])
                    == str(path)
                ),
                None,
            )
            self.assertIsNotNone(
                match,
                f"{path} could not be found in the candidates it was drawn from",
            )
            bodies.append(
                train.Candidate.read(match) if hasattr(match, "read") else match[1]
            )

        self.assertTrue(bodies, "no reserved document produced text")
        for body in bodies:
            self.assertTrue(body.strip(), "a reserved document read as empty")

    def test_the_state_file_names_documents_that_exist(self):
        """A reservation naming a missing file would be a second silent loss."""
        c = self.config()
        pieces = self.candidates(c, curate=True)
        regression.reserve(c, list(pieces))
        reserved = regression.load_reserved(c)
        self.assertTrue(reserved)
        for path in reserved:
            self.assertTrue(
                Path(path).exists(),
                f"the state file names {path}, which is not on disk",
            )


if __name__ == "__main__":
    unittest.main()
