"""Tests for the Architecture 25 Parsing stage of structured raw sources.

The behaviour worth protecting here is restraint. A structured file has to yield
its text, and a record whose fields are not recognised has to be counted and
left alone rather than guessed at. The second half matters more than the first:
a parser that invents text for a shape it has not read puts material into the
corpus that nobody wrote, and the report says nothing about it.
"""

import unittest

import curate_dataset
import dialogue
import json_source


class RenderTests(unittest.TestCase):
    def test_a_speaker_and_text_record_becomes_one_turn(self):
        line = json_source.render_record({"speaker": "Alice", "text": "hello there"})
        self.assertEqual(line, "Alice: hello there")

    def test_turns_use_the_layout_dialogue_already_speaks_in(self):
        # One definition of a turn, so training and inference cannot drift apart.
        line = json_source.render_record({"speaker": "Alice", "text": "hello"})
        self.assertEqual(line, dialogue.format_turn("Alice", "hello").strip())

    def test_roles_map_to_the_labels_sample_py_uses(self):
        self.assertEqual(
            json_source.render_record({"role": "user", "content": "hi"}), "You: hi"
        )
        self.assertEqual(
            json_source.render_record({"role": "assistant", "content": "hi"}),
            "Jupiter: hi",
        )

    def test_a_role_mapping_targets_the_constants_dialogue_defines(self):
        self.assertEqual(json_source.ROLE_LABELS["user"], dialogue.HUMAN)
        self.assertEqual(json_source.ROLE_LABELS["assistant"], dialogue.JUPITER)

    def test_an_unmapped_role_keeps_its_own_name(self):
        # A "system" turn must stay "System" so the contamination check still
        # sees a leaked template instead of it being laundered into dialogue.
        line = json_source.render_record({"role": "system", "content": "be brief"})
        self.assertEqual(line, "System: be brief")
        self.assertEqual(len(dialogue.template_artifacts(line)), 1)

    def test_a_lowercase_speaker_is_capitalised_so_the_label_is_visible(self):
        # dialogue.SPEAKER requires a capital; an unrecognised label is prose
        # the model is then told is not a turn.
        line = json_source.render_record({"speaker": "alice", "text": "hi"})
        self.assertEqual(dialogue.turn_count(line), 1)

    def test_sharegpt_from_and_value_is_read(self):
        line = json_source.render_record({"from": "human", "value": "hello"})
        self.assertEqual(line, "You: hello")

    def test_a_multi_line_turn_is_flattened_to_one_line(self):
        # Load-bearing: turn_share counts turns against non-empty lines, so a
        # turn written across three lines scores as prose it is not.
        line = json_source.render_record(
            {"speaker": "Alice", "text": "one\ntwo\nthree"}
        )
        self.assertEqual(line, "Alice: one two three")
        self.assertNotIn("\n", line)

    def test_a_flattened_conversation_still_scores_as_a_conversation(self):
        text, extracted, unrecognized = json_source.extract_text(
            [{"speaker": "Alice", "text": "one\ntwo\nthree"} for _ in range(6)]
        )
        self.assertEqual((extracted, unrecognized), (6, 0))
        self.assertEqual(dialogue.turn_share(text), 1.0)

    def test_a_plain_string_record_becomes_prose(self):
        self.assertEqual(
            json_source.render_record("just a sentence"), "just a sentence"
        )

    def test_a_record_with_no_recognisable_field_is_not_guessed_at(self):
        self.assertIsNone(json_source.render_record({"alpha": 1, "beta": [2, 3]}))

    def test_a_record_with_several_unknown_strings_is_not_guessed_at(self):
        # Choosing between them would be a coin toss that becomes training text.
        self.assertIsNone(json_source.render_record({"alpha": "one", "beta": "two"}))

    def test_a_lone_unrecognised_string_is_still_read(self):
        # One string in a record is the text far more often than it is a label.
        self.assertEqual(
            json_source.render_record({"document": "the text"}), "the text"
        )

    def test_a_record_with_no_speaker_is_rendered_without_a_label(self):
        self.assertEqual(json_source.render_record({"text": "just this"}), "just this")

    def test_a_non_record_is_not_rendered(self):
        self.assertIsNone(json_source.render_record(42))
        self.assertIsNone(json_source.render_record(None))


class ContainerTests(unittest.TestCase):
    def test_a_list_is_the_record_list(self):
        records, container = json_source.records_from([{"a": 1}])
        self.assertEqual(container, "list")
        self.assertEqual(records, [{"a": 1}])

    def test_a_nested_message_list_is_unwrapped(self):
        records, container = json_source.records_from(
            {"messages": [{"role": "user", "content": "hi"}], "id": 7}
        )
        self.assertEqual(container, "object/messages")
        self.assertEqual(records, [{"role": "user", "content": "hi"}])

    def test_a_sharegpt_conversation_list_is_unwrapped(self):
        records, container = json_source.records_from(
            {"conversations": [{"from": "human", "value": "hi"}]}
        )
        self.assertEqual(container, "object/conversations")
        self.assertEqual(len(records), 1)

    def test_a_bare_object_is_a_single_record(self):
        records, container = json_source.records_from({"text": "hi"})
        self.assertEqual(container, "object")
        self.assertEqual(records, [{"text": "hi"}])

    def test_an_empty_turn_list_is_not_treated_as_conversation_records(self):
        # "messages": [] carries no turns, so the object itself is the record.
        records, container = json_source.records_from({"messages": [], "text": "hi"})
        self.assertEqual(container, "object")

    def test_a_scalar_is_a_single_record(self):
        records, container = json_source.records_from("just text")
        self.assertEqual(container, "scalar")
        self.assertEqual(records, ["just text"])

    def test_a_prompt_completion_record_becomes_a_two_turn_conversation(self):
        # The shape 63.6% of this project's corpus arrived in. Read as plain
        # text it taught the model braces; read as a record it is a question
        # and an answer, which is the only thing a model can learn to do.
        records, container = json_source.records_from(
            {
                "input": [{"role": "user", "content": "what is an index"}],
                "output": "it speeds up lookups",
                "category": "chat",
            }
        )
        self.assertEqual(container, "object/prompt-completion")
        self.assertEqual(
            [json_source.render_record(record) for record in records],
            ["You: what is an index", "Jupiter: it speeds up lookups"],
        )

    def test_the_answer_is_spoken_by_jupiter_not_left_as_prose(self):
        # Otherwise the model learns to emit answers that follow no question.
        line = json_source.render_record(
            {"role": "assistant", "content": "it speeds up lookups"}
        )
        self.assertTrue(line.startswith(f"{dialogue.JUPITER}: "))

    def test_a_prompt_given_as_a_bare_string_is_still_a_question(self):
        records, container = json_source.records_from(
            {"prompt": "what is an index", "completion": "it speeds up lookups"}
        )
        self.assertEqual(container, "object/prompt-completion")
        self.assertEqual(
            [json_source.render_record(record) for record in records],
            ["You: what is an index", "Jupiter: it speeds up lookups"],
        )

    def test_a_multi_turn_prompt_keeps_every_turn(self):
        records, _ = json_source.records_from(
            {
                "input": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
                "output": "third",
            }
        )
        self.assertEqual(
            [json_source.render_record(record) for record in records],
            ["You: first", "Jupiter: second", "Jupiter: third"],
        )

    def test_a_completion_with_no_prompt_is_still_the_answer_turn(self):
        # Half a conversation is worth keeping; dropping it loses the answer.
        records, container = json_source.records_from({"output": "an answer alone"})
        self.assertEqual(container, "object/prompt-completion")
        self.assertEqual(
            [json_source.render_record(record) for record in records],
            ["Jupiter: an answer alone"],
        )

    def test_an_empty_output_is_not_read_as_a_completion(self):
        records, container = json_source.records_from(
            {"input": [{"role": "user", "content": "q"}], "output": "   "}
        )
        self.assertEqual(container, "object")
        self.assertEqual(
            records, [{"input": [{"role": "user", "content": "q"}], "output": "   "}]
        )

    def test_a_native_turn_list_still_wins_over_the_prompt_shape(self):
        # "messages" is a complete conversation already; nothing is appended.
        records, container = json_source.records_from(
            {"messages": [{"role": "user", "content": "hi"}], "output": "ignored"}
        )
        self.assertEqual(container, "object/messages")
        self.assertEqual(len(records), 1)


class ContentDetectionTests(unittest.TestCase):
    """Whether a file is structured is a question about its content, not its name."""

    def test_banner_delimited_json_is_recognised(self):
        text = '--- SAMPLE 0 ---\n{"input": [], "output": "x"}'
        self.assertTrue(json_source.looks_structured(text))

    def test_plain_prose_is_not_structured(self):
        text = "It was a bright cold day in April.\nThe clocks were striking thirteen."
        self.assertFalse(json_source.looks_structured(text))

    def test_prose_that_opens_with_a_brace_is_still_prose(self):
        # A single leading brace is not evidence. A book about JSON opens with
        # one and must not be routed through a parser.
        self.assertFalse(
            json_source.looks_structured('{ "note": "this document is about JSON"')
        )

    def test_empty_text_is_not_structured(self):
        self.assertFalse(json_source.looks_structured(""))
        self.assertFalse(json_source.looks_structured("   \n  "))

    def test_one_json_document_is_recognised(self):
        self.assertTrue(json_source.looks_structured('{"a": 1}'))

    def test_a_json_array_is_recognised(self):
        self.assertTrue(json_source.looks_structured('[{"a": 1}]'))

    def test_a_shard_opening_mid_record_is_recognised(self):
        # Shards are cut at byte offsets, so one opens with a quoted key and no
        # brace and no banner. This is 67MB of the corpus that stayed raw
        # punctuation because the probe only looked for an opening delimiter.
        self.assertTrue(
            json_source.looks_structured(
                '  "output": "Here are the instructions",\n  "category": "chat"\n'
            )
        )

    def test_prose_opening_with_a_labelled_line_is_not_structured(self):
        # The fragment test is a quoted key and a colon. A prose line that
        # happens to have that shape must not be routed through a parser.
        self.assertFalse(
            json_source.looks_structured(
                "Note: this paragraph is ordinary prose that runs on for a "
                "while so the file is long enough to be plausible.\n"
                "It continues here and mentions nothing structured at all.\n"
            )
        )

    def test_prose_starting_with_a_quote_is_not_structured(self):
        # A novel opening with a quotation mark is prose, not a payload.
        self.assertFalse(
            json_source.looks_structured(
                '"It was a bright cold day in April," he remembered well.'
            )
        )


class ConcatenatedShardTests(unittest.TestCase):
    """A shard with no banner in it is still a shard.

    When a file is cut mid-record the banner that would have announced the first
    record is in the previous file, so splitting on banners finds nothing and the
    whole shard would have been read as prose.
    """

    def _write(self, folder, body):
        path = folder / "shard.txt"
        path.write_text(body, encoding="utf-8")
        return path

    def _shard(self, count):
        return (
            '  "output": "orphaned tail",\n'
            + json_source.json.dumps(
                {
                    "input": [{"role": "user", "content": "q"}],
                    "output": "a",
                }
            )
            + "\n"
        ) * count

    def test_records_are_recovered_without_any_banner(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            result = json_source.extract(self._write(Path(raw), self._shard(3)))
        self.assertEqual(result.source_format, "json-concatenated")
        # Three prompt/completion documents are six turns, because a conversation
        # is counted in turns and each of these is a question and an answer. The
        # records field counts turns for the same reason everywhere else.
        self.assertEqual((result.records, result.extracted), (6, 6))
        self.assertIn("Jupiter: a", result.text)

    def test_the_leading_fragment_is_reported_rather_than_invented(self):
        # A truncated object cannot be reconstructed without inventing the part
        # that was cut off, so it is named instead of guessed at.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            result = json_source.extract(self._write(Path(raw), self._shard(2)))
        self.assertTrue(any("mid-record" in error["reason"] for error in result.errors))

    def test_a_trailing_fragment_is_reported(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "shard.txt"
            path.write_text('{"a": 1}\n{"b": ', encoding="utf-8")
            values, errors = json_source.parse_concatenated(
                path.read_text(encoding="utf-8"), path.name
            )
        # The complete record is kept and the cut one is named, rather than
        # either being silently dropped or silently repaired.
        self.assertEqual(len(values), 1)
        self.assertTrue(any("unterminated" in error["reason"] for error in errors))

    def test_braces_inside_strings_do_not_confuse_the_matching(self):
        # A payload full of code samples is full of braces that are not
        # structure. Counting them without tracking strings would truncate the
        # first record and lose everything after it.
        import tempfile
        from pathlib import Path

        body = (
            json_source.json.dumps(
                {
                    "input": [{"role": "user", "content": 'a } b { c " d'}],
                    "output": "answer",
                }
            )
            + "\n"
            + json_source.json.dumps({"output": "second"})
        )
        with tempfile.TemporaryDirectory() as raw:
            result = json_source.extract(self._write(Path(raw), body))
        # Two documents: the first a question and an answer, the second an answer
        # alone. Braces inside the quoted content must not have ended the first
        # record early, which would have truncated it and lost the second.
        self.assertEqual(result.records, 3)
        self.assertIn("second", result.text)

    def test_prose_with_braces_yields_nothing_rather_than_a_guess(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "novel.txt"
            path.write_text("He said { and then } the end.", encoding="utf-8")
            result = json_source.extract_tolerant(path)
        self.assertEqual(result.records, 0)


class BannerShardTests(unittest.TestCase):
    """A sharded export must not be refused over its first partial record."""

    def _write(self, folder, body):
        path = folder / "shard.txt"
        path.write_text(body, encoding="utf-8")
        return path

    def test_every_sample_between_banners_is_read(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            body = "\n".join(
                f"--- SAMPLE {index} ---\n"
                + json_source.json.dumps(
                    {
                        "input": [{"role": "user", "content": f"q{index}"}],
                        "output": f"a{index}",
                    }
                )
                for index in range(3)
            )
            result = json_source.extract(self._write(Path(raw), body))
        self.assertEqual(result.source_format, "json-banners")
        # Two turns per sample: the question and the answer. A conversation is
        # counted in turns, so three samples are six records.
        self.assertEqual((result.records, result.extracted), (6, 6))
        self.assertIn("You: q1", result.text)
        self.assertIn("Jupiter: a2", result.text)

    def test_a_shard_starting_mid_record_keeps_the_rest(self):
        # Shards are cut at byte offsets, so the first chunk is a fragment.
        # Losing 63MB over it would be the wrong trade.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            body = (
                '--- SAMPLE 0 ---\n  "output": "orphaned tail"\n'
                "--- SAMPLE 1 ---\n"
                + json_source.json.dumps(
                    {
                        "input": [{"role": "user", "content": "q"}],
                        "output": "a",
                    }
                )
            )
            result = json_source.extract(self._write(Path(raw), body))
        # The orphaned fragment is dropped and reported; the intact sample that
        # follows is still read, as both of its turns.
        self.assertEqual(result.extracted, 2)
        self.assertIn("Jupiter: a", result.text)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("sample 1", result.errors[0]["reason"])

    def test_prose_named_txt_is_still_refused_by_the_parser(self):
        # looks_structured says False for this, and extract must not invent
        # structure out of a novel.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "novel.txt"
            path.write_text("It was a bright cold day in April.", encoding="utf-8")
            with self.assertRaises(json_source.StructuredSourceError):
                json_source.extract(path)

    def test_a_shard_that_is_not_really_json_is_returned_as_a_zero_count(self):
        # Jupiter's own generated files are the case that matters: they open
        # with the banner-delimited prompt they were seeded from and continue
        # into whatever the model produced, so the head looks structured and
        # the body often does not parse. Returning zero records with zero
        # errors is what tells the caller to read the file as prose instead of
        # condemning it.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "generated.txt"
            path.write_text(
                "--- SAMPLE 0 ---\n{ not closed\nsome prose the model produced\n",
                encoding="utf-8",
            )
            result = json_source.extract_tolerant(path)
        self.assertEqual(result.records, 0)
        self.assertEqual(result.extracted, 0)
        self.assertEqual(result.errors, [])

    def test_the_tolerant_reader_keeps_what_does_parse(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "mixed.txt"
            path.write_text(
                "--- SAMPLE 0 ---\n{ broken\n"
                "--- SAMPLE 1 ---\n"
                + json_source.json.dumps(
                    {
                        "input": [{"role": "user", "content": "q"}],
                        "output": "a",
                    }
                ),
                encoding="utf-8",
            )
            result = json_source.extract_tolerant(path)
        self.assertEqual(result.extracted, 2)
        self.assertIn("Jupiter: a", result.text)
        self.assertEqual(len(result.errors), 1)


class SupportTests(unittest.TestCase):
    def test_the_configured_suffixes_are_the_ones_supported(self):
        for suffix in (".json", ".jsonl", ".ndjson"):
            self.assertTrue(json_source.is_supported("x" + suffix))
        for suffix in (".txt", ".md", ".csv", ""):
            self.assertFalse(json_source.is_supported("x" + suffix))

    def test_the_suffix_test_ignores_case(self):
        self.assertTrue(json_source.is_supported("X.JSON"))

    def test_flatten_collapses_runs_of_whitespace(self):
        self.assertEqual(json_source.flatten("a\n\n  b\tc  "), "a b c")

    def test_flatten_treats_a_tab_the_way_normalize_does(self):
        # A tab inside a one-line turn means nothing, and normalize() already
        # rewrites it to a space. If the two disagreed, the same turn would
        # hash differently depending on which path produced it.
        self.assertEqual(json_source.flatten("b\tc"), "b c")
        self.assertEqual(json_source.flatten("b\tc"), curate_dataset.normalize("b\tc"))


if __name__ == "__main__":
    unittest.main()
