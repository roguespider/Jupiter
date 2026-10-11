"""Parsing a structured file off disk.

These are the tests that cover the failure modes a file on disk actually has:
it is malformed, half of it is malformed, or it holds a shape nothing reads.
None of them may stop the run, and all of them have to be reported.
"""

import json
import unittest

import dialogue
import json_source

from tests.support import TempDirTestCase

LONG = "The observatory stood on the granite above the valley above the river. "


class StructuredFileTests(TempDirTestCase):
    def write_json(self, name, payload):
        return self.write(name, json.dumps(payload))

    def test_a_json_array_of_turns_is_parsed(self):
        path = self.write_json(
            "chat.json",
            [
                {"speaker": "Alice", "text": LONG},
                {"speaker": "Bob", "text": LONG},
            ],
        )
        result = json_source.extract(path)
        self.assertEqual(result.source_format, "json")
        self.assertEqual(result.container, "list")
        self.assertEqual(result.records, 2)
        self.assertEqual(result.extracted, 2)
        self.assertEqual(result.unrecognized, 0)
        self.assertIn("Alice: ", result.text)

    def test_a_nested_message_list_is_unwrapped(self):
        path = self.write_json(
            "openai.json",
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": LONG},
                    {"role": "assistant", "content": LONG},
                ],
            },
        )
        result = json_source.extract(path)
        self.assertEqual(result.container, "object/messages")
        self.assertEqual(result.records, 2)
        self.assertIn("You: ", result.text)
        self.assertIn("Jupiter: ", result.text)

    def test_a_jsonl_stream_is_parsed_line_by_line(self):
        path = self.write(
            "sharegpt.jsonl",
            "\n".join(
                json.dumps({"from": who, "value": LONG})
                for who in ["human", "gpt", "human", "gpt"]
            )
            + "\n",
        )
        result = json_source.extract(path)
        self.assertEqual(result.source_format, "jsonl")
        self.assertEqual(result.records, 4)
        self.assertEqual(result.unrecognized, 0)

    def test_an_ndjson_file_is_read_as_a_stream(self):
        path = self.write(
            "log.ndjson", json.dumps({"text": LONG}) + "\n" + json.dumps({"text": LONG})
        )
        result = json_source.extract(path)
        self.assertEqual(result.source_format, "jsonl")
        self.assertEqual(result.records, 2)

    def test_blank_lines_in_a_stream_are_not_records(self):
        path = self.write("sparse.jsonl", "\n\n" + json.dumps({"text": LONG}) + "\n\n")
        result = json_source.extract(path)
        self.assertEqual(result.records, 1)
        self.assertEqual(result.errors, [])

    def test_malformed_json_is_refused_with_the_reason(self):
        path = self.write("broken.json", "{not json at all")
        with self.assertRaises(json_source.StructuredSourceError) as caught:
            json_source.extract(path)
        # The message names the file, because a reason an owner cannot match
        # to a file is a reason they cannot act on.
        self.assertIn("broken.json", str(caught.exception))

    def test_an_empty_file_is_refused_as_invalid_json(self):
        # An empty file is not an empty document, it is a file that could not be
        # read. Saying so is more useful than reporting zero records and letting
        # the owner wonder whether the file was empty or ignored.
        path = self.write("empty.json", "")
        with self.assertRaises(json_source.StructuredSourceError) as caught:
            json_source.extract(path)
        self.assertIn("empty.json", str(caught.exception))

    def test_a_bad_line_does_not_cost_the_good_lines(self):
        # One corrupt record in a long export should not throw away the rest.
        path = self.write(
            "mixed.jsonl",
            "\n".join(
                [
                    json.dumps({"text": LONG}),
                    "{ this line is broken",
                    json.dumps({"text": LONG}),
                ]
            ),
        )
        result = json_source.extract(path)
        self.assertEqual(result.records, 2)
        self.assertEqual(result.extracted, 2)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["line"], 2)

    def test_a_record_of_an_unknown_shape_is_counted_not_invented(self):
        path = self.write_json("odd.json", [{"alpha": 1, "beta": [2, 3]}])
        result = json_source.extract(path)
        self.assertEqual(result.records, 1)
        self.assertEqual(result.extracted, 0)
        self.assertEqual(result.unrecognized, 1)
        self.assertEqual(result.text, "")

    def test_extraction_never_exceeds_the_records_it_was_given(self):
        path = self.write_json(
            "mixed.json",
            [{"text": LONG}, {"alpha": 1}, {"speaker": "Alice", "text": LONG}],
        )
        result = json_source.extract(path)
        self.assertLessEqual(result.extracted, result.records)
        self.assertEqual(result.extracted + result.unrecognized, result.records)

    def test_the_summary_reports_the_shape_it_found(self):
        path = self.write_json("s.json", [{"text": LONG}])
        summary = json_source.extract(path).summary()
        self.assertEqual(summary["format"], "json")
        self.assertEqual(summary["container"], "list")
        self.assertEqual(summary["records"], 1)

    def test_a_leaked_template_export_is_recognised_as_contamination(self):
        # Roles are mapped, but "system" is not one of them, so it survives as a
        # label the contamination check already knows how to count. Eight
        # records is deliberately more than the threshold of five.
        path = self.write_json(
            "leaked.json",
            [{"role": "system", "content": "reason in Spanish"}] * 8,
        )
        result = json_source.extract(path)
        self.assertEqual(result.extracted, 8)
        self.assertGreater(len(dialogue.template_artifacts(result.text)), 5)


if __name__ == "__main__":
    unittest.main()
