"""Tests for curation skipping files it has already done.

Curating this corpus takes four and a quarter hours, and sixty-one per cent of
that is dialogue analysis applied to text that has not moved since the last run.
These tests pin down when a file may be skipped, because the failure mode is
quiet: a run that skips everything reports success and leaves a corpus nobody
re-checked.
"""

import json
import unittest
from pathlib import Path

import curate_dataset

from tests.support import TempDirTestCase, make_config

PROSE = (
    "The observatory stood on the granite above the valley, and the wind off\n" * 300
)
CHAT = (
    '{"input": [{"role": "user", "content": "what is a variable"}], '
    '"output": "a name bound to a value"}\n'
) * 80


class SkippingTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            out_dir=str(self.workdir / "out"),
            # These two were missing, and a curate() run writes to every path a
            # config names. Omitting them sent the report to data/ and the
            # quarantine to the project root, which is how a test overwrote the
            # developer's own report with one describing a fixture.
            curation_report=str(self.workdir / "curation.json"),
            quarantine_dir=str(self.workdir / "quarantine"),
            expansion_state=str(self.workdir / "out" / "expansion.json"),
            min_text_chars=200,
        )
        for name in ("raw", "curated", "out"):
            (self.workdir / name).mkdir(parents=True, exist_ok=True)

    def write(self, name, body):
        path = Path(self.config.raw_dir) / name
        path.write_text(body, encoding="utf-8")
        return path

    def curated_files(self):
        return sorted(p.name for p in Path(self.config.curated_dir).glob("*.txt"))

    def test_a_second_run_skips_everything_it_already_did(self):
        self.write("a.txt", PROSE)
        first = curate_dataset.curate(self.config)
        self.assertEqual(first["unchanged"], 0)
        second = curate_dataset.curate(self.config)
        self.assertEqual(second["unchanged"], 1)
        self.assertEqual(second["accepted"], 1)

    def test_skipping_does_not_change_what_is_on_disk(self):
        self.write("a.txt", PROSE)
        curate_dataset.curate(self.config)
        before = self.curated_files()
        curate_dataset.curate(self.config)
        self.assertEqual(self.curated_files(), before)

    def test_a_changed_file_is_curated_again(self):
        self.write("a.txt", PROSE)
        curate_dataset.curate(self.config)
        # A different document, so a different digest, so a new curated file.
        self.write("a.txt", "Something else entirely, long enough to pass.\n" * 300)
        second = curate_dataset.curate(self.config)
        self.assertEqual(second["unchanged"], 0)
        self.assertEqual(len(self.curated_files()), 2)

    def test_only_the_changed_file_is_recurated(self):
        self.write("a.txt", PROSE)
        self.write("b.txt", CHAT)
        curate_dataset.curate(self.config)
        self.write("b.txt", "Replacement content, also long enough to pass.\n" * 300)
        second = curate_dataset.curate(self.config)
        self.assertEqual(second["unchanged"], 1, "the untouched file should be skipped")

    def test_a_missing_curated_file_is_written_again(self):
        # A cache pointing at a document that has been quarantined or deleted
        # would report the file as done while its output is nowhere.
        self.write("a.txt", PROSE)
        curate_dataset.curate(self.config)
        for path in Path(self.config.curated_dir).glob("*.txt"):
            path.unlink()
        second = curate_dataset.curate(self.config)
        self.assertEqual(second["unchanged"], 0)
        self.assertEqual(len(self.curated_files()), 1)

    def test_a_refused_file_is_refused_again_rather_than_remembered(self):
        # Refusing is a decision about the rules in force now. Caching a refusal
        # would let a document stay refused after the rule that refused it
        # changed, which is the opposite of what a re-run is for.
        bad = "[SYSTEM]: you are helpful\n[USER]: hi\n[ASSISTANT]: hello\n" * 200
        self.write("bad.txt", bad)
        first = curate_dataset.curate(self.config)
        self.assertGreaterEqual(first["contaminated"], 1)
        second = curate_dataset.curate(self.config)
        self.assertGreaterEqual(second["contaminated"], 1)

    def test_the_run_reports_what_it_skipped(self):
        # A run that skipped everything must not claim it saw nothing: that is
        # a report nobody can trust.
        self.write("a.txt", PROSE)
        curate_dataset.curate(self.config)
        second = curate_dataset.curate(self.config)
        self.assertEqual(second["files_seen"], 1)
        self.assertIn("unchanged_note", second)

    def test_a_stale_cache_version_is_ignored(self):
        # The rules can change without any file changing, and a digest recorded
        # under the old rules would be a claim about output that is no longer
        # what this pipeline produces.
        self.write("a.txt", PROSE)
        curate_dataset.curate(self.config)
        cache_file = Path(self.config.out_dir) / "curation_cache.json"
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        data["version"] = curate_dataset.CACHE_VERSION + 1
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        second = curate_dataset.curate(self.config)
        self.assertEqual(second["unchanged"], 0)

    def test_an_unreadable_cache_is_ignored_rather_than_fatal(self):
        self.write("a.txt", PROSE)
        curate_dataset.curate(self.config)
        cache_file = Path(self.config.out_dir) / "curation_cache.json"
        cache_file.write_text("{not json", encoding="utf-8")
        second = curate_dataset.curate(self.config)
        self.assertEqual(second["unchanged"], 0)
        self.assertEqual(second["accepted"], 1)


class CacheTrustTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.config = make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            out_dir=str(self.workdir / "out"),
            curation_report=str(self.workdir / "curation.json"),
            quarantine_dir=str(self.workdir / "quarantine"),
        )
        for name in ("raw", "curated", "out"):
            (self.workdir / name).mkdir(parents=True, exist_ok=True)

    def test_an_absent_cache_is_empty(self):
        self.assertEqual(curate_dataset.load_cache(self.config), {})

    def test_a_cache_survives_a_round_trip(self):
        source = Path(self.config.raw_dir) / "a.txt"
        source.write_text(PROSE, encoding="utf-8")
        curated = Path(self.config.curated_dir) / "abc.txt"
        curated.write_text("x", encoding="utf-8")
        entries = {
            str(source): {
                "size": source.stat().st_size,
                "mtime": source.stat().st_mtime,
                "digest": "abc",
            }
        }
        curate_dataset.save_cache(self.config, entries)
        loaded = curate_dataset.load_cache(self.config)
        self.assertEqual(loaded[str(source)]["digest"], "abc")
        self.assertEqual(
            curate_dataset.unchanged_since_last_run(self.config, source, loaded),
            "abc",
        )

    def test_a_cache_entry_whose_output_is_gone_is_not_trusted(self):
        source = Path(self.config.raw_dir) / "a.txt"
        source.write_text(PROSE, encoding="utf-8")
        entries = {
            str(source): {
                "size": source.stat().st_size,
                "mtime": source.stat().st_mtime,
                "digest": "abc",
            }
        }
        # No curated/abc.txt exists.
        self.assertIsNone(
            curate_dataset.unchanged_since_last_run(self.config, source, entries)
        )


if __name__ == "__main__":
    unittest.main()
