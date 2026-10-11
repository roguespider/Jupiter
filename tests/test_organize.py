"""Tests for organising the corpus by skill, and for tidying up afterwards.

Three things are pinned here, and each of them is a mistake this script has
actually made. Filing is only worth doing if a document lands in the folder
that describes it; tidying is only safe if it cannot remove a file; and a second
run has to leave the corpus alone, or a part-finished run quietly nests folders
inside each other.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import organize_raw


class ClassificationTests(unittest.TestCase):
    def test_a_name_that_says_what_it_is(self):
        self.assertEqual(organize_raw.classify("default_math.part0001.txt"), "math")
        self.assertEqual(
            organize_raw.classify("default_chat.part0000.txt"), "conversation"
        )
        self.assertEqual(
            organize_raw.classify("reasoning_corpus.part0003.txt"), "reasoning"
        )

    def test_a_programming_language_sample_is_code(self):
        # 1,873 files in this corpus are named "<Language>_sample_N" across 663
        # languages, and not one of them contains a word any pattern matched.
        # The shape is what identifies them, because a list of names would go
        # stale the day a language is added.
        self.assertEqual(organize_raw.classify("Zig_sample_2.txt"), "code")
        self.assertEqual(organize_raw.classify("ABAP CDS_sample_1.txt"), "code")

    def test_a_scanned_book_is_prose(self):
        self.assertEqual(organize_raw.classify("ahumanroots_djvu.txt"), "prose")

    def test_a_name_that_says_nothing_classifies_as_nothing(self):
        self.assertIsNone(organize_raw.classify("mystery-document.txt"))


class ContentClassificationTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, body):
        path = self.dir / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_a_hashed_maths_file_is_recognised(self):
        path = self.write(
            "a" * 64 + ".txt", "JEE Main 2022\nLet f : R $$to$$ R be continuous"
        )
        self.assertEqual(organize_raw.classify_path(path), "math")

    def test_a_hashed_instruction_file_is_conversation(self):
        path = self.write(
            "b" * 64 + ".txt", '{"input": [{"role": "user"}], "output": "x"}'
        )
        self.assertEqual(organize_raw.classify_path(path), "conversation")

    def test_a_book_header_is_not_source_code(self):
        # "Language: English" is in the Gutenberg header of every translated
        # Asimov title, and an earlier version of the code rule matched it and
        # filed a nuclear-energy book as code.
        path = self.write(
            "c" * 64 + ".txt",
            "The Project Gutenberg EBook of Worlds Within Worlds\nLanguage: English\n",
        )
        self.assertEqual(organize_raw.classify_path(path), "prose")

    def test_a_syntax_matrix_is_code(self):
        path = self.write(
            "d" * 64 + ".txt",
            "Language: C (ANSI (C89)\nKeywords: auto, break, case",
        )
        self.assertEqual(organize_raw.classify_path(path), "code")

    def test_material_nothing_recognises_is_left_alone(self):
        # A folder that means the wrong thing is worse than no folder, so an
        # unreadable file is reported rather than guessed at.
        path = self.write("e" * 64 + ".txt", "zzz nothing here matches zzz")
        self.assertIsNone(organize_raw.classify_path(path))


class PruneTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_an_empty_directory_is_removed(self):
        (self.dir / "book" / "prose").mkdir(parents=True)
        removed = organize_raw.prune_empty_dirs(self.dir)
        # Two, not one: removing the inner folder leaves its parent empty, and a
        # parent is only noticed on the next pass.
        self.assertEqual(removed, 2)
        self.assertFalse((self.dir / "book").exists())

    def test_a_directory_holding_a_document_survives(self):
        keep = self.dir / "book" / "prose"
        keep.mkdir(parents=True)
        (keep / "a.txt").write_text("x", encoding="utf-8")
        organize_raw.prune_empty_dirs(self.dir)
        self.assertTrue(keep.exists())
        self.assertTrue((keep / "a.txt").exists())

    def test_a_top_level_folder_is_kept_even_when_empty(self):
        # The skill folders are the layout. Removing an emptied one means the
        # next run has to recreate it, and the folder a duplicate was moved into
        # would vanish the moment it was emptied.
        for name in set(organize_raw.FOLDER_NAMES) | organize_raw.SKIP:
            (self.dir / name).mkdir()
        organize_raw.prune_empty_dirs(self.dir)
        for name in set(organize_raw.FOLDER_NAMES) | organize_raw.SKIP:
            self.assertTrue((self.dir / name).exists(), name)

    def test_a_parent_emptied_by_its_children_is_removed_too(self):
        # Deepest first is not enough on its own: a directory only becomes
        # visible as empty once its own subdirectories are gone, so the walk has
        # to repeat until a pass removes nothing.
        (self.dir / "a" / "b" / "c").mkdir(parents=True)
        removed = organize_raw.prune_empty_dirs(self.dir)
        self.assertEqual(removed, 3)
        self.assertFalse((self.dir / "a").exists())


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_running_twice_moves_nothing_the_second_time(self):
        # The move is not atomic, so a run that stops part way leaves files
        # already filed. Without this they get filed again into a folder nested
        # inside the first one, and the corpus ends up described by a path that
        # means nothing.
        (self.dir / "default_math.part1.txt").write_text("x = 1", encoding="utf-8")
        moves, _dupes, _un, _settled = organize_raw.plan(self.dir)
        organize_raw.apply_plan(self.dir, moves, _dupes)
        moves2, _d2, _u2, settled = organize_raw.plan(self.dir)
        self.assertEqual(moves2, [])
        self.assertEqual(settled, 1)
        self.assertFalse((self.dir / "math" / "math").exists())

    def test_a_duplicate_is_set_aside_rather_than_deleted(self):
        body = "identical content in two files"
        (self.dir / "default_math.a.txt").write_text(body, encoding="utf-8")
        (self.dir / "default_math.b.txt").write_text(body, encoding="utf-8")
        moves, dupes, _un, _s = organize_raw.plan(self.dir)
        organize_raw.apply_plan(self.dir, moves, dupes)
        holding = list((self.dir / "duplicates").glob("*"))
        self.assertEqual(len(holding), 1)
        self.assertEqual(holding[0].read_text(encoding="utf-8"), body)

    def test_a_set_aside_duplicate_is_not_moved_back_out(self):
        # A second run used to find each duplicate against the copy filed in
        # duplicates/ and move it out again, so the folder emptied itself one
        # run at a time.
        body = "identical content in two files"
        (self.dir / "default_math.a.txt").write_text(body, encoding="utf-8")
        (self.dir / "default_math.b.txt").write_text(body, encoding="utf-8")
        moves, dupes, _un, _s = organize_raw.plan(self.dir)
        organize_raw.apply_plan(self.dir, moves, dupes)
        moves2, dupes2, _u2, _s2 = organize_raw.plan(self.dir)
        organize_raw.apply_plan(self.dir, moves2, dupes2)
        self.assertEqual(len(list((self.dir / "duplicates").glob("*"))), 1)


class CacheTests(unittest.TestCase):
    """A second run over an unchanged corpus must not read the corpus again.

    The plan and the apply each want the same answers, and asking twice meant
    reading every unclassified file's head and every candidate's first and last
    kilobytes twice over. On a hundred-and-thirty-five gigabyte corpus that is
    minutes of disk traffic to produce one answer, and it is why this script
    appeared to hang.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.raw = self.dir / "raw"
        self.raw.mkdir()
        (self.raw / "default_math.a.txt").write_text("x = 1" * 200, encoding="utf-8")
        (self.raw / "default_chat.b.txt").write_text("hello there" * 200, encoding="utf-8")

    def test_a_second_run_reads_nothing(self):
        organize_raw.plan(self.raw)
        calls = []
        original = organize_raw.fingerprint

        def counting(path, span=1 << 15):
            calls.append(path.name)
            return original(path, span)

        organize_raw.fingerprint = counting
        try:
            organize_raw.plan(self.raw)
        finally:
            organize_raw.fingerprint = original
        self.assertEqual(calls, [], "an unchanged corpus must not be re-read")

    def test_a_changed_file_is_read_again(self):
        organize_raw.plan(self.raw)
        target = self.raw / "default_math.a.txt"
        target.write_text("completely different content" * 200, encoding="utf-8")
        calls = []
        original = organize_raw.fingerprint

        def counting(path, span=1 << 15):
            calls.append(path.name)
            return original(path, span)

        organize_raw.fingerprint = counting
        try:
            organize_raw.plan(self.raw)
        finally:
            organize_raw.fingerprint = original
        self.assertIn("default_math.a.txt", calls)
