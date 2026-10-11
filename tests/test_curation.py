"""Tests for curation (Architecture 25, 28).

The one that matters here is contamination. Two of the real corpus files were
synthetic pretraining dumps carrying thousands of chat-template markers, and
training on them teaches the model to emit instructions rather than converse.
Curation refuses them without touching the source, because deleting somebody
else's data is not curation's decision to make.
"""

import contextlib
import io
import json

import curate_dataset
import dialogue
import json_source

from config import Config
from tests.support import TempDirTestCase, make_config

LONG_ENOUGH = "The observatory stood on the granite above the valley. " * 12

LEAKED_TEMPLATE = (
    "[SYSTEM]: reasoning language: Spanish\nSystem: answer briefly.\n" * 40
) + LONG_ENOUGH

CLEAN_PROSE = LONG_ENOUGH

CHAT_LOG = "".join(
    f"Alice: line {n} of the exchange\nBob: reply {n} in the exchange\n"
    for n in range(6)
)

CHAT_LINE = "The observatory stood on the granite above the valley. "


class PruningTests(TempDirTestCase):
    """Curation owns data/curated, and that directory can drift.

    A curated file the pipeline would refuse still sits in the training corpus
    until something removes it, because refusing only stops it being written
    again. That is how 36 percent of a real corpus ended up being leaked LLM
    output while curation reported a refusal on every run.
    """

    def config(self):
        return make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            curation_report=str(self.workdir / "curation.json"),
            out_dir=str(self.workdir / "out"),
            quarantine_dir=str(self.workdir / "quarantine"),
            min_text_chars=128,
            max_template_artifacts=5,
        )

    def curate(self):
        original = curate_dataset.load_config
        chosen = self.config()
        curate_dataset.load_config = lambda: chosen
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        return curate_dataset.curate(chosen)

    def curated_files(self):
        return sorted(p.name for p in (self.workdir / "curated").glob("*.txt"))

    def test_a_stale_curated_file_is_removed(self):
        self.write("raw/book.txt", CLEAN_PROSE)
        stale = self.write("curated/deadbeef.txt", "left over from an old run")
        report = self.curate()
        self.assertFalse(stale.exists())
        self.assertEqual(report["quarantined"], 1)

    def test_a_file_whose_raw_source_is_now_refused_is_removed(self):
        # The case that actually mattered: this file was curated before the
        # contamination check existed, so refusing did nothing to it.
        self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        leftover = self.write("curated/" + "a" * 64 + ".txt", LEAKED_TEMPLATE)
        self.curate()
        self.assertFalse(leftover.exists())

    def test_the_raw_source_is_never_deleted(self):
        source = self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        self.write("curated/" + "a" * 64 + ".txt", LEAKED_TEMPLATE)
        self.curate()
        self.assertTrue(source.exists())
        self.assertGreater(source.stat().st_size, 0)

    def test_a_quarantined_file_is_kept_not_destroyed(self):
        stale = self.write("curated/deadbeef.txt", CLEAN_PROSE)
        size = stale.stat().st_size
        self.curate()
        kept = curate_dataset.quarantine_directory(self.config()) / "curated"
        moved = kept / stale.name
        self.assertTrue(moved.exists(), "quarantine must preserve the file")
        self.assertEqual(moved.stat().st_size, size)
        self.assertTrue(
            (kept / f"{stale.name}.reason").exists(),
            "a quarantined file must say why it was moved",
        )

    def test_accepted_files_are_never_touched(self):
        self.write("raw/book.txt", CLEAN_PROSE)
        self.curate()
        before = self.curated_files()
        self.curate()
        self.assertEqual(self.curated_files(), before)

    def test_the_curated_set_ends_up_exactly_what_was_accepted(self):
        self.write("raw/book.txt", CLEAN_PROSE)
        self.write("raw/chat.txt", CHAT_LOG)
        self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        self.write("curated/deadbeef.txt", CLEAN_PROSE)
        self.write("curated/feedface.txt", LEAKED_TEMPLATE)
        report = self.curate()
        self.assertEqual(len(self.curated_files()), report["accepted"])
        self.assertEqual(report["quarantined"], 2)

    def test_the_report_says_what_it_quarantined(self):
        self.write("raw/book.txt", CLEAN_PROSE)
        self.write("curated/deadbeef.txt", CLEAN_PROSE)
        self.curate()
        written = json.loads(
            (self.workdir / "curation.json").read_text(encoding="utf-8")
        )
        self.assertIn("data/raw was not modified", written["quarantine_note"])
        self.assertTrue(written["quarantined_files"][0]["bytes"] > 0)


class CurationTests(TempDirTestCase):
    def config(self):
        return make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            curation_report=str(self.workdir / "curation.json"),
            # out_dir and quarantine_dir were missing here, so every path
            # curation writes to outside data/curated went to the developer's
            # real out/ instead of the fixture. That is how a test run overwrote
            # the production curation report and cache with paths pointing at
            # temporary directories, and why the next real run started by
            # re-curating a corpus the cache said it had never seen.
            out_dir=str(self.workdir / "out"),
            quarantine_dir=str(self.workdir / "quarantine"),
            expansion_state=str(self.workdir / "out" / "expansion.json"),
            min_text_chars=128,
            max_template_artifacts=5,
        )

    def curate(self):
        """curate() reads the config from the environment, so patch it in."""
        original = curate_dataset.load_config
        chosen = self.config()
        curate_dataset.load_config = lambda: chosen
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        return curate_dataset.curate(chosen)

    def test_a_leaked_chat_template_is_refused(self):
        self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        report = self.curate()
        self.assertEqual(report["contaminated"], 1)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(len(report["contaminated_documents"]), 1)
        self.assertGreater(report["contaminated_documents"][0]["template_artifacts"], 5)

    def test_a_refused_document_is_left_in_data_raw(self):
        # Refusing is not deleting. The owner decides what to do with the file.
        source = self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        self.curate()
        self.assertTrue(source.exists())
        self.assertGreater(source.stat().st_size, 0)

    def test_a_refused_document_is_not_written_to_the_curated_set(self):
        self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        self.curate()
        self.assertEqual(list((self.workdir / "curated").glob("*.txt")), [])

    def test_ordinary_prose_is_accepted(self):
        self.write("raw/book.txt", CLEAN_PROSE)
        report = self.curate()
        self.assertEqual(report["contaminated"], 0)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["prose"], 1)

    def test_a_real_conversation_is_accepted_and_classified(self):
        self.write("raw/chat.txt", CHAT_LOG)
        report = self.curate()
        self.assertEqual(report["conversational"], 1)
        self.assertEqual(report["prose"], 0)
        document = report["documents"][0]
        self.assertEqual(document["kind"], "conversation")
        self.assertIn("Alice", document["speakers"])

    def test_a_stray_marker_does_not_condemn_a_document(self):
        # One "Note:" in a novel is not contamination. The threshold is a count
        # precisely so this stays possible.
        self.write("raw/book.txt", CLEAN_PROSE + "\nNote: see the appendix.\n")
        report = self.curate()
        self.assertEqual(report["contaminated"], 0)
        self.assertEqual(report["accepted"], 1)

    def test_the_report_says_what_it_refused_and_why(self):
        self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        self.curate()
        written = json.loads(
            (self.workdir / "curation.json").read_text(encoding="utf-8")
        )
        self.assertIn("chat-template", written["template_artifact_note"])
        self.assertIn("untouched", written["template_artifact_note"])

    def test_the_threshold_is_configuration_not_a_constant(self):
        config = Config()
        self.assertGreater(config.max_template_artifacts, 0)
        # report() exposes the raw count the threshold is compared against, so
        # the decision is inspectable rather than hidden inside curation.
        measured = dialogue.report(LEAKED_TEMPLATE)
        self.assertGreater(
            measured["template_artifacts"], config.max_template_artifacts
        )

    def test_a_document_too_short_is_rejected_as_before(self):
        self.write("raw/tiny.txt", "too short")
        report = self.curate()
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(report["accepted"], 0)


class OversizedSourceTests(TempDirTestCase):
    """A raw file larger than the machine can hold is refused, not attempted.

    The corpus this pipeline was run against held a 77 GB document. Curation
    reads a whole file to judge it, so the read raised a MemoryError from inside
    a codec, which named neither the file nor the size nor anything the owner
    could act on. The size is taken from the filesystem before any read, so an
    oversized document is refused and named, and the run continues.
    """

    def config(self):
        return make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            curation_report=str(self.workdir / "curation.json"),
            out_dir=str(self.workdir / "out"),
            quarantine_dir=str(self.workdir / "quarantine"),
            min_text_chars=128,
            max_template_artifacts=5,
        )

    def curate(self, config=None):
        # The config is passed to curate() rather than only patched into
        # load_config, so a test cannot reach the developer's own corpus even if
        # the patch is removed. Seven tests were curating real material because
        # curate() fell back to load_config() and nothing told it not to.
        chosen = config or self.config()
        original = curate_dataset.load_config
        curate_dataset.load_config = lambda: chosen
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        return curate_dataset.curate(chosen)

    def test_a_file_too_large_to_read_is_refused_before_it_is_read(self):
        c = self.config()
        c.max_raw_bytes = 1024
        self.write("raw/huge.txt", "x" * 2048)
        report = self.curate(c)
        self.assertEqual(report["split"], 1)
        self.assertEqual(report["rejected"], 0)
        entry = report["split_files"][0]
        self.assertEqual(entry["bytes"], 2048)
        self.assertEqual(entry["limit"], 1024)
        self.assertIn("huge.txt", entry["source"])
        self.assertIn("larger than", entry.get("note", entry.get("reason", "")))

    def test_the_raw_file_is_never_deleted_for_being_too_large(self):
        # Refusing to read a file is not permission to remove it. data/raw is the
        # source of truth, and what to do with a 77 GB document is the owner's
        # decision rather than curation's.
        c = self.config()
        c.max_raw_bytes = 1024
        target = self.write("raw/huge.txt", "x" * 2048)
        self.curate(c)
        self.assertTrue(target.exists(), "an oversized raw file must survive")
        self.assertEqual(target.stat().st_size, 2048)

    def test_the_limit_is_configuration_rather_than_a_constant(self):
        # A project on a larger machine, or with a corpus of genuinely large
        # documents, has to be able to raise this deliberately rather than by
        # editing source, which is the change Architecture 38 asks a human for.
        self.assertEqual(make_config().max_raw_bytes, 64 * 1024 * 1024)
        c = self.config()
        c.max_raw_bytes = 4096
        self.write("raw/big.txt", "x" * 2048)
        report = self.curate(c)
        self.assertEqual(report["oversized"], 0)
        self.assertEqual(report["accepted"], 1)

    def test_a_file_under_the_limit_is_unaffected(self):
        # The guard is a ceiling, not a filter. Ordinary documents are still read
        # and accepted exactly as before.
        c = self.config()
        self.write("raw/normal.txt", CLEAN_PROSE)
        report = self.curate(c)
        self.assertEqual(report["oversized"], 0)
        self.assertEqual(report["accepted"], 1)

    def test_the_report_says_it_refused_an_oversized_file_and_why(self):
        c = self.config()
        c.max_raw_bytes = 1024
        self.write("raw/huge.txt", "x" * 2048)
        report = self.curate(c)
        self.assertIn("oversized_note", report)
        self.assertIn("max_raw_bytes", report["oversized_note"])
        self.assertIn("untouched", report["oversized_note"])

    def test_a_corpus_with_no_oversized_file_reports_zero_rather_than_nothing(self):
        # A count of zero is different from a key that is absent, and the report
        # is read by people deciding whether curation is doing its job.
        c = self.config()
        self.write("raw/normal.txt", CLEAN_PROSE)
        report = self.curate(c)
        self.assertEqual(report["oversized"], 0)
        self.assertEqual(report["oversized_files"], [])


class ContaminationShareTests(TempDirTestCase):
    """A count of markers is not enough to spot a leaked template.

    The words user, system, human and assistant followed by a colon are ordinary
    English in any writing about interfaces, permissions or agents. A real
    corpus file in this project was refused for exactly that reason: it carried
    ten such lines in sixteen million characters and was not contamination at
    all, so a usable training document was being thrown away for nothing.

    What a dumped conversation template looks like is not a document containing
    markers but a document made of them, which is the same reasoning
    dialogue.turn_share already applies to speaker labels.
    """

    # The marked lines are a small minority of a long document, which is the
    # whole point: this is what writing that discusses users and systems
    # actually looks like. The real corpus file that was being refused had ten
    # such lines in sixteen million characters, a share of six hundred-
    # thousandths, so the fixture is built at that scale deliberately.
    MARKED = (
        "User: the operator must confirm before the change is applied.\n"
        "System: every request is recorded in the audit log.\n"
        "user: the session expires after a period of inactivity.\n"
        "System: a backup is taken before the upgrade proceeds.\n"
        "Human: an operator may override the default setting.\n"
        "Assistant: the summary is written to the incident report.\n"
    )

    ORDINARY_LINE = "The interface requires confirmation before a change is applied.\n"

    PROSE = ORDINARY_LINE * 400 + MARKED + ORDINARY_LINE * 400

    def config(self):
        return make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            curation_report=str(self.workdir / "curation.json"),
            out_dir=str(self.workdir / "out"),
            quarantine_dir=str(self.workdir / "quarantine"),
            min_text_chars=128,
            max_template_artifacts=5,
        )

    def curate(self):
        original = curate_dataset.load_config
        chosen = self.config()
        curate_dataset.load_config = lambda: chosen
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        with contextlib.redirect_stdout(io.StringIO()):
            return curate_dataset.curate(chosen)

    def test_prose_that_merely_mentions_users_is_not_contamination(self):
        # Six marker lines here, above the count of five, in a long document.
        self.write("raw/prose.txt", self.PROSE)
        report = self.curate()
        self.assertEqual(report["contaminated"], 0)
        self.assertEqual(report["accepted"], 1)

    def test_the_count_alone_would_have_condemned_it(self):
        # Pins that the count really is over the threshold, so the test above is
        # passing because of the share and not by accident.
        measured = dialogue.report(self.PROSE)
        self.assertGreater(
            measured["template_artifacts"], self.config().max_template_artifacts
        )
        self.assertLessEqual(
            measured["template_share"], self.config().max_template_share
        )

    def test_a_dumped_template_is_still_refused(self):
        self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        report = self.curate()
        self.assertEqual(report["contaminated"], 1)
        self.assertEqual(report["accepted"], 0)

    def test_both_thresholds_are_reported_for_a_refusal(self):
        self.write("raw/leaked.txt", LEAKED_TEMPLATE)
        self.curate()
        entry = json.loads(
            (self.workdir / "curation.json").read_text(encoding="utf-8")
        )["contaminated_documents"][0]
        self.assertIn("template_share", entry)
        self.assertGreater(entry["template_share"], self.config().max_template_share)

    def test_the_share_of_a_text_with_no_lines_is_zero(self):
        self.assertEqual(dialogue.template_share(""), 0.0)
        self.assertEqual(dialogue.template_share("\n\n  \n"), 0.0)


class StructuredSourceTests(TempDirTestCase):
    """A .json file in data/raw used to do nothing at all.

    It was not accepted, not refused and not counted: the suffix check in
    curate_dataset skipped it before the file counter moved, so a hundred of
    them could sit in the corpus while the report read exactly as it had
    before. The workaround of renaming one to .txt was worse, because the raw
    punctuation was then trained on verbatim and the contamination check passed
    it, a JSON payload having no [SYSTEM]: marker to count. These tests hold
    both of those closed.
    """

    def config(self):
        return make_config(
            raw_dir=str(self.workdir / "raw"),
            curated_dir=str(self.workdir / "curated"),
            curation_report=str(self.workdir / "curation.json"),
            out_dir=str(self.workdir / "out"),
            quarantine_dir=str(self.workdir / "quarantine"),
            min_text_chars=128,
            max_template_artifacts=5,
        )

    def curate(self):
        original = curate_dataset.load_config
        chosen = self.config()
        curate_dataset.load_config = lambda: chosen
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        with contextlib.redirect_stdout(io.StringIO()):
            return curate_dataset.curate(chosen)

    def write_json(self, name, payload):
        target = self.workdir / "raw" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def write_jsonl(self, name, rows):
        target = self.workdir / "raw" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return target

    def curated_texts(self):
        return [
            path.read_text(encoding="utf-8")
            for path in sorted((self.workdir / "curated").glob("*.txt"))
        ]

    def write_text(self, name, body):
        target = self.workdir / "raw" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return target

    def test_a_txt_file_holding_sharded_json_is_parsed_not_trained_as_punctuation(self):
        # The single highest-impact defect found by audit: 63.6% of the real
        # corpus arrived as .txt holding banner-delimited instruction JSON, so
        # the model spent its budget learning braces and key names. The suffix
        # decided it; the content has to be asked.
        body = "\n".join(
            f"--- SAMPLE {index} ---\n"
            + json.dumps(
                {
                    "input": [{"role": "user", "content": f"question {index} here"}],
                    "output": f"answer {index} here",
                    "category": "chat",
                }
            )
            for index in range(4)
        )
        self.write_text("default_chat.part0000.txt", body)
        report = self.curate()
        texts = self.curated_texts()
        self.assertEqual(len(texts), 1)
        self.assertNotIn('"role"', texts[0])
        self.assertNotIn('"output"', texts[0])
        self.assertIn("You: question 1 here", texts[0])
        self.assertIn("Jupiter: answer 2 here", texts[0])
        self.assertEqual(report["structured_documents"][0]["format"], "json-banners")
        self.assertEqual(report["documents"][0]["kind"], "conversation")

    def test_plain_prose_named_txt_is_unaffected(self):
        self.write_text("a_novel.txt", CLEAN_PROSE + "And the wind turned north.")
        report = self.curate()
        self.assertEqual(len(report["documents"]), 1)
        self.assertEqual(report["documents"][0]["kind"], "prose")
        self.assertEqual(report["structured_documents"], [])
        self.assertIn("observatory", self.curated_texts()[0])

    def test_prose_that_opens_with_a_brace_is_not_sent_to_the_parser(self):
        # A document about JSON is a document, not a payload.
        self.write_text(
            "notes.txt",
            '{ "note": "this is prose about JSON, not JSON itself" }\n' + CLEAN_PROSE,
        )
        report = self.curate()
        self.assertEqual(report["documents"][0]["kind"], "prose")
        self.assertEqual(report["structured_documents"], [])

    def test_a_banner_shard_whose_records_are_all_unreadable_is_refused(self):
        # It announces itself as structured and yields nothing. Refusing beats
        # training on the punctuation of a structure nobody could read, which is
        # the failure this check exists to prevent.
        self.write_text(
            "broken.txt",
            "--- SAMPLE 0 ---\n{ neither of these is JSON\n" * 2,
        )
        report = self.curate()
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(report["unreadable"], 1)
        self.assertEqual(self.curated_texts(), [])

    def test_an_unterminated_brace_is_prose_rather_than_a_broken_payload(self):
        # One opening brace with nothing closing it is a document that happens
        # to start with a brace, and it is read as the prose it is. The text is
        # padded past the minimum so that what is being tested is the routing
        # and not the length check.
        self.write_text(
            "opening.txt",
            "{ this opens like JSON but never closes it. " + CLEAN_PROSE,
        )
        report = self.curate()
        self.assertEqual(report["documents"][0]["kind"], "prose")
        self.assertEqual(report["structured_documents"], [])
        self.assertEqual(report["rejected"], 0)

    def test_a_generated_file_that_only_looks_structured_is_read_as_text(self):
        # Jupiter's own synthesized material: a banner-delimited prompt that
        # never closes, followed by whatever the model produced. Refusing it
        # would throw away readable material on the strength of a malformed
        # head, which is how a corpus loses its own voice.
        self.write_text(
            "generated-sample.txt",
            "--- SAMPLE 0 ---\n{ not closed\n"
            "The observatory stood on the granite above the valley. " * 4,
        )
        report = self.curate()
        self.assertEqual(report["rejected"], 0)
        self.assertEqual(len(report["documents"]), 1)
        self.assertEqual(report["structured_documents"], [])
        self.assertIn("observatory", self.curated_texts()[0])

    def test_a_json_conversation_is_accepted_and_classified_as_dialogue(self):
        self.write_json(
            "chat.json",
            [
                {"speaker": "Alice", "text": CHAT_LINE},
                {"speaker": "Bob", "text": CHAT_LINE},
                {"speaker": "Alice", "text": CHAT_LINE},
                {"speaker": "Bob", "text": CHAT_LINE},
            ],
        )
        report = self.curate()
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["conversational"], 1)
        document = report["documents"][0]
        self.assertEqual(document["source_kind"], "structured")
        self.assertEqual(document["records"], 4)
        self.assertIn("Alice", document["speakers"])

    def test_a_jsonl_stream_is_accepted(self):
        self.write_jsonl(
            "chat.jsonl",
            [
                {"from": "human", "value": CHAT_LINE},
                {"from": "gpt", "value": CHAT_LINE},
                {"from": "human", "value": CHAT_LINE},
                {"from": "gpt", "value": CHAT_LINE},
            ],
        )
        report = self.curate()
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["conversational"], 1)
        self.assertEqual(report["structured"], 1)
        self.assertEqual(report["structured_records"], 4)

    def test_structured_text_is_never_trained_on_as_raw_markup(self):
        # The regression. Curation used to have no way to read a .json, and the
        # only route into the corpus was renaming it, which put the braces and
        # the key names into the text.
        self.write_json("chat.json", [{"speaker": "Alice", "text": CHAT_LINE}] * 6)
        self.curate()
        joined = "\n".join(self.curated_texts())
        self.assertNotIn("{", joined)
        self.assertNotIn("}", joined)
        self.assertNotIn('"speaker"', joined)
        self.assertNotIn('"text"', joined)
        self.assertIn("Alice: ", joined)

    def test_a_json_file_is_counted_as_seen(self):
        self.write_json("chat.json", [{"speaker": "Alice", "text": CHAT_LINE}] * 6)
        self.write_json("odd.json", [{"alpha": 1}])
        report = self.curate()
        self.assertEqual(report["files_seen"], 2)

    def test_malformed_json_is_refused_without_stopping_the_run(self):
        self.write("raw/broken.json", "{not json at all")
        self.write("raw/book.txt", CLEAN_PROSE)
        report = self.curate()
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(report["unreadable"], 1)
        self.assertEqual(report["accepted"], 1)
        reasons = [item["reason"] for item in report["unreadable_files"]]
        self.assertTrue(any("broken.json" in reason for reason in reasons))

    def test_a_refused_structured_file_is_left_in_data_raw(self):
        source = self.write("raw/broken.json", "{not json at all")
        self.curate()
        self.assertTrue(source.exists())
        self.assertGreater(source.stat().st_size, 0)

    def test_records_of_an_unknown_shape_are_reported_not_invented(self):
        self.write_json("odd.json", [{"alpha": 1, "beta": [2, 3]}])
        report = self.curate()
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["unrecognized_records"], 1)
        self.assertEqual(report["unreadable"], 1)
        self.assertIn(
            "no field this pipeline recognises", report["unreadable_files"][0]["reason"]
        )

    def test_a_file_with_no_usable_records_at_all_is_refused(self):
        self.write_json("odd.json", [{"alpha": 1}, {"beta": 2}])
        report = self.curate()
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["unreadable"], 1)

    def test_an_unsupported_extension_is_now_visible(self):
        # The reason a .json file could go missing without trace was that the
        # skip was silent. Every unreadable extension is now counted and named.
        self.write("raw/table.csv", "a,b,c")
        report = self.curate()
        self.assertEqual(report["skipped_unsupported"], 1)
        self.assertEqual(report["files_seen"], 0)
        self.assertTrue(report["unsupported_files"])
        self.assertEqual(report["unsupported_files"][0]["suffix"], ".csv")

    def test_an_unsupported_file_is_left_exactly_where_it_was(self):
        source = self.write("raw/table.csv", "a,b,c")
        self.curate()
        self.assertTrue(source.exists())
        self.assertEqual(source.read_text(encoding="utf-8"), "a,b,c")

    def test_a_file_with_no_extension_is_reported_rather_than_ignored(self):
        self.write("raw/README", "some notes")
        report = self.curate()
        self.assertEqual(report["skipped_unsupported"], 1)
        self.assertEqual(report["unsupported_files"][0]["suffix"], "(no extension)")

    def test_text_and_structured_documents_deduplicate_against_each_other(self):
        # Both carry the same turns in the same order, so both resolve to one
        # digest and the corpus holds one document rather than two identical
        # ones. A parser that reordered them would silently double the corpus.
        self.write("raw/chat.txt", CHAT_LOG)
        self.write_json(
            "chat.json",
            [
                record
                for n in range(6)
                for record in (
                    {"speaker": "Alice", "text": f"line {n} of the exchange"},
                    {"speaker": "Bob", "text": f"reply {n} in the exchange"},
                )
            ],
        )
        report = self.curate()
        self.assertEqual(report["duplicates"], 1)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(len(self.curated_texts()), 1)

    def test_plain_text_behaviour_is_unchanged(self):
        self.write("raw/book.txt", CLEAN_PROSE)
        report = self.curate()
        self.assertEqual(report["files_seen"], 1)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["prose"], 1)
        self.assertEqual(report["structured"], 0)
        self.assertIsNone(report["documents"][0]["records"])
        self.assertEqual(report["documents"][0]["source_kind"], "text")

    def test_a_leaked_template_export_is_still_refused(self):
        # Mapping the roles must not launder a leaked conversation template
        # into something the contamination check cannot see.
        self.write_json(
            "leaked.json",
            [{"role": "system", "content": "reason in Spanish"}] * 8,
        )
        report = self.curate()
        self.assertEqual(report["contaminated"], 1)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(list((self.workdir / "curated").glob("*.txt")), [])

    def test_the_report_explains_what_the_structured_run_found(self):
        self.write_json("chat.json", [{"speaker": "Alice", "text": CHAT_LINE}] * 6)
        self.curate()
        written = json.loads(
            (self.workdir / "curation.json").read_text(encoding="utf-8")
        )
        self.assertIn("structured", written["structured_note"])
        self.assertIn("unreadable_files", written["unreadable_note"])
        self.assertIn("unsupported_files", written["unsupported_note"])

    def test_the_configured_suffixes_come_from_configuration(self):
        # Configuration rather than a constant in the parser, so the set of
        # readable formats is one declared decision.
        self.assertIn(".json", Config().structured_suffixes)
        self.assertIn(".jsonl", Config().structured_suffixes)
        self.assertEqual(tuple(Config().structured_suffixes), json_source.SUFFIXES)

    def test_the_configured_suffixes_actually_drive_curation(self):
        self.write_json("chat.json", [{"speaker": "Alice", "text": CHAT_LINE}] * 6)
        config = self.config()
        config.structured_suffixes = ()
        original = curate_dataset.load_config
        curate_dataset.load_config = lambda: config
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        with contextlib.redirect_stdout(io.StringIO()):
            # The config carrying structured_suffixes=() is the one that has to be
            # used. Passing a freshly built one here would test the default and
            # pass for the wrong reason.
            report = curate_dataset.curate(config)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["skipped_unsupported"], 1)
