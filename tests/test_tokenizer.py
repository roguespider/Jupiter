"""Tests for both tokenizers (Architecture 8).

Architecture 8 requires the tokenizer to represent natural language, code,
symbols, malformed input, slang and typing errors. Every one of those tests
below is a round trip: encode then decode must return the original text. A
tokenizer that silently drops input it has not seen would fail the whole
project silently, so the round trip is the property that matters, not the
compression figure.
"""

import unittest

import tokenizer as tokenizer_module
from tokenizer_bpe import BASE_VOCAB, BpeTokenizer, TokenizerError, pretokenize

from tests.support import TempDirTestCase

CORPUS = (
    "the neutrino observatory records neutrino events every night. "
    "a neutrino observatory needs a detector underground. "
) * 12

SAMPLES = (
    "hello",
    "the quick brown fox",
    "naive cafe",
    "naïve café",
    "日本語のテキスト",
    "🦙 emoji survive",
    "def f(x):\n    return x + 1\n",
    "colour colour colour colour",
    "teh rapid typo",
    "tab\tseparated and double  spaces",
    "",
)


class PretokenizeTests(unittest.TestCase):
    def test_words_whitespace_and_punctuation_are_separate(self):
        pieces = pretokenize("hi there! ok?")
        self.assertEqual(pieces, ["hi", " ", "there", "!", " ", "ok", "?"])

    def test_pretokenizing_covers_the_whole_text(self):
        text = "a b\tc\nd,e"
        self.assertEqual("".join(pretokenize(text)), text)


class ByteTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tokenizer_module.ByteTokenizer()

    def test_round_trip_for_every_kind_of_text(self):
        for text in SAMPLES:
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_one_token_per_byte(self):
        self.assertEqual(self.tokenizer.compression("hello"), 1.0)
        self.assertEqual(len(self.tokenizer.encode("hello")), 5)

    def test_validate_accepts_every_byte_and_rejects_the_rest(self):
        self.tokenizer.validate(list(range(256)))
        for bad in (256, -1):
            with self.assertRaises(ValueError):
                self.tokenizer.validate([bad])

    def test_the_module_interface_is_the_byte_tokenizer(self):
        self.assertEqual(tokenizer_module.encode("ab"), [97, 98])
        self.assertEqual(tokenizer_module.decode([97, 98]), "ab")
        # Read from the tokenizer rather than from a module constant. There was
        # a VOCAB_SIZE here that duplicated this and nothing in production read
        # it, so a reader could not tell which was authoritative and neither was
        # checked against the other.
        self.assertEqual(
            tokenizer_module.ByteTokenizer.vocab_size,
            BASE_VOCAB,
        )
        self.assertFalse(
            hasattr(tokenizer_module, "VOCAB_SIZE"),
            "a second, unread copy of the vocabulary size is worse than none",
        )


class BpeTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BpeTokenizer([]).train([CORPUS], num_merges=120)

    def test_round_trip_for_every_kind_of_text(self):
        for text in SAMPLES:
            with self.subTest(text=text):
                self.assertEqual(
                    self.tokenizer.decode(self.tokenizer.encode(text)), text
                )

    def test_malformed_input_does_not_crash(self):
        self.assertIsInstance(
            self.tokenizer.decode(self.tokenizer.encode("\ud800")), str
        )

    def test_text_it_was_never_trained_on_still_encodes(self):
        # Architecture 8 requires malformed input and unfamiliar text to be
        # representable. The 256 byte tokens are always available, so nothing
        # can be unrepresentable.
        for text in ("zygote quokka", "🙂🙃", "x"):
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_vocabulary_is_the_base_plus_one_id_per_merge(self):
        # Training stops early when the corpus runs out of pairs to merge, so
        # the invariant is the relationship, not the requested count.
        self.assertLessEqual(len(self.tokenizer.merges), 120)
        self.assertGreater(len(self.tokenizer.merges), 0)
        self.assertEqual(
            self.tokenizer.vocab_size, BASE_VOCAB + len(self.tokenizer.merges)
        )

    def test_a_larger_corpus_learns_more_merges(self):
        bigger = BpeTokenizer([]).train(
            [CORPUS + ("zebra quartz " * 200)], num_merges=120
        )
        self.assertGreater(len(bigger.merges), len(self.tokenizer.merges))

    def test_every_id_decodes_to_bytes(self):
        for identifier in range(self.tokenizer.vocab_size):
            self.assertIsInstance(self.tokenizer.decode([identifier]), str)

    def test_repeated_text_compresses(self):
        # The whole point of the exercise. One byte per token gives 1.0.
        self.assertGreater(self.tokenizer.compression(CORPUS), 1.5)
        self.assertEqual(tokenizer_module.ByteTokenizer().compression(CORPUS), 1.0)

    def test_compression_is_reported_not_assumed(self):
        figure = self.tokenizer.compression(CORPUS)
        self.assertIsInstance(figure, float)
        self.assertGreater(figure, 0.0)

    def test_training_is_deterministic(self):
        again = BpeTokenizer([]).train([CORPUS], num_merges=120)
        self.assertEqual(again.merges, self.tokenizer.merges)
        self.assertEqual(again.encode(CORPUS), self.tokenizer.encode(CORPUS))

    def test_fewer_merges_means_a_smaller_vocabulary(self):
        small = BpeTokenizer([]).train([CORPUS], num_merges=10)
        self.assertLess(small.vocab_size, self.tokenizer.vocab_size)

    def test_validate_rejects_ids_outside_the_vocabulary(self):
        self.tokenizer.validate([0, self.tokenizer.vocab_size - 1])
        with self.assertRaises(ValueError):
            self.tokenizer.validate([self.tokenizer.vocab_size])

    def test_an_empty_merge_list_is_still_a_valid_byte_tokenizer(self):
        plain = BpeTokenizer([])
        self.assertEqual(plain.vocab_size, BASE_VOCAB)
        self.assertEqual(plain.decode(plain.encode("abc")), "abc")


class BpePersistenceTests(TempDirTestCase):
    def test_a_saved_vocabulary_reproduces_the_same_ids(self):
        trained = BpeTokenizer([]).train([CORPUS], num_merges=60)
        path = trained.save(self.workdir / "tokenizer.json")
        reloaded = BpeTokenizer.load(path)
        self.assertEqual(reloaded.merges, trained.merges)
        self.assertEqual(reloaded.vocab_size, trained.vocab_size)
        self.assertEqual(reloaded.encode(CORPUS), trained.encode(CORPUS))
        self.assertEqual(reloaded.version, trained.version)

    def test_a_missing_vocabulary_explains_what_to_do(self):
        with self.assertRaises(TokenizerError) as caught:
            BpeTokenizer.load(self.workdir / "absent.json")
        self.assertIn("train", str(caught.exception).lower())

    def test_a_file_that_is_not_a_bpe_vocabulary_is_refused(self):
        path = self.write("tokenizer.json", '{"kind": "something-else"}')
        with self.assertRaises(TokenizerError):
            BpeTokenizer.load(path)

    def test_the_saved_record_is_versioned(self):
        # Architecture 43: tokenizer version is tracked separately.
        trained = BpeTokenizer([]).train([CORPUS], num_merges=10)
        record = trained.to_dict()
        self.assertEqual(record["kind"], "byte-level-bpe")
        self.assertIn("tokenizer_version", record)
        self.assertEqual(record["base_vocab"], BASE_VOCAB)
        self.assertEqual(record["vocab_size"], trained.vocab_size)


class InterfaceTests(unittest.TestCase):
    """Both tokenizers must present the same surface.

    This exists because they did not: BpeTokenizer was missing `name`, and the
    worker only reached it when running with the BPE tokenizer selected, so the
    divergence survived until a real run. Checking the whole interface is
    cheaper than discovering the next missing attribute in production.
    """

    def setUp(self):
        self.byte = tokenizer_module.ByteTokenizer()
        self.bpe = BpeTokenizer([]).train([CORPUS], num_merges=20)

    def test_both_expose_the_same_members(self):
        for attribute in ("name", "version", "vocab_size"):
            self.assertTrue(
                hasattr(self.byte, attribute),
                f"byte tokenizer has no {attribute}",
            )
            self.assertTrue(
                hasattr(self.bpe, attribute),
                f"bpe tokenizer has no {attribute}",
            )

    def test_both_expose_the_same_methods(self):
        for method in ("encode", "decode", "validate", "compression"):
            self.assertTrue(callable(getattr(self.byte, method)), method)
            self.assertTrue(callable(getattr(self.bpe, method)), method)

    def test_both_encode_decode_and_validate_the_same_text(self):
        text = "the observatory records events"
        for tokenizer in (self.byte, self.bpe):
            ids = tokenizer.encode(text)
            tokenizer.validate(ids)
            self.assertEqual(tokenizer.decode(ids), text)

    def test_the_names_match_the_configuration_values(self):
        self.assertEqual(self.byte.name, tokenizer_module.BYTE)
        self.assertEqual(self.bpe.name, tokenizer_module.BPE)


class TokenizerSelectionTests(TempDirTestCase):
    def test_bpe_is_the_default(self):
        from config import Config

        self.assertIsInstance(
            tokenizer_module.load_tokenizer(Config()),
            BpeTokenizer,
        )

    def test_bpe_is_loaded_when_the_configuration_asks_for_it(self):
        from config import Config

        trained = BpeTokenizer([]).train([CORPUS], num_merges=20)
        config = Config()
        config.tokenizer = "bpe"
        config.tokenizer_vocab = str(trained.save(self.workdir / "tokenizer.json"))
        loaded = tokenizer_module.load_tokenizer(config)
        self.assertIsInstance(loaded, BpeTokenizer)
        self.assertEqual(loaded.vocab_size, trained.vocab_size)

    def test_bpe_without_a_vocabulary_raises_rather_than_falling_back(self):
        from config import Config

        config = Config()
        config.tokenizer = "bpe"
        config.tokenizer_vocab = str(self.workdir / "absent.json")
        with self.assertRaises(TokenizerError):
            tokenizer_module.load_tokenizer(config)

    def test_an_unknown_tokenizer_name_is_refused(self):
        from config import Config

        config = Config()
        config.tokenizer = "nonsense"
        with self.assertRaises(ValueError):
            tokenizer_module.load_tokenizer(config)
