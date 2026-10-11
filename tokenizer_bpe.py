"""Byte-level BPE tokenizer (Architecture 8, 30, 43).

The byte tokenizer in tokenizer.py is lossless and needs no vocabulary file,
which is why it was the right first choice. Its cost is one token per byte, so
a 512 token context holds 512 characters -- about eighty words. That is too
small to hold a question, some recalled knowledge and an answer.

This tokenizer learns merges over raw bytes, so a common word or fragment
becomes a single token. Nothing is lost: every token still decodes to bytes, and
unknown text still encodes, because the 256 single-byte tokens remain the base
of the vocabulary. That matters more than it sounds -- Jupiter's corpus is
ordinary prose with ordinary misspellings, and a tokenizer that cannot encode
something it has not seen would be a tokenizer that silently drops input.

Architecture 8 says tokenizer experiments must be measurable, so the module
reports its compression ratio rather than leaving it to be assumed. It also
says the tokenizer decision should be made before long-term model development
because changing it invalidates existing weights. This one is therefore a
migration, not a switch: switching tokenizers changes the vocabulary size, the
shape guard in train.py notices, and the old checkpoint is archived rather than
misused.

No external dependency. The merge table is trained here and stored as JSON.
"""

import json
import re
from collections import Counter
from pathlib import Path

TOKENIZER_VERSION = "0.1"
BASE_VOCAB = 256

# How much of each curated document the vocabulary is trained on.
#
# This is deliberately not c.max_chunk_chars, which is how much of a document one
# learning cycle reads. A 57MB document contributed its first 12,000 characters
# to the vocabulary and nothing else, so thirty documents trained it on 269,191
# characters and it managed 1.66 bytes per token on the corpus it came from,
# against 2.29 for the same merge count trained on 1.5MB of that corpus. A
# vocabulary is a summary of the corpus or it is a summary of whichever document
# sorted first.
#
# Bounded so that training does not read the whole 346MB corpus into memory,
# large enough that a large document is represented rather than sampled from its
# opening.
VOCAB_TRAIN_CHARS = 400_000
# The name this tokenizer is selected by in configuration, matching the values
# config.tokenizer accepts.
NAME = "bpe"

# Words, runs of whitespace, and single punctuation characters. Merges never
# span these boundaries, so a token can never straddle two words, which keeps
# encoding fast and stops the vocabulary filling with whitespace trivia.
PRETOKEN = re.compile(r"\s+|\w+|[^\s\w]")


class TokenizerError(RuntimeError):
    """Raised when a tokenizer file is missing or unusable."""


def pretokenize(text):
    """Split text into the units merges are learned over."""
    return PRETOKEN.findall(text)


class BpeTokenizer:
    """A learned byte-level BPE tokenizer.

    Token ids 0-255 are raw bytes. Each learned merge takes the next id, so the
    vocabulary is 256 + the number of merges and every id decodes to bytes.
    """

    name = NAME

    def __init__(self, merges, version=TOKENIZER_VERSION, trained_on=None):
        self.merges = [tuple(pair) for pair in merges]
        self.version = version
        self.trained_on = trained_on
        self.ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.vocab_size = BASE_VOCAB + len(self.merges)
        self._cache = {}

    # --- training ---

    def _word_pairs(self, word):
        return list(zip(word, word[1:]))

    def train(self, texts, num_merges=2000, progress=None):
        """Learn merges from raw text and return the tokenizer.

        The standard incremental approach: count every adjacent pair across all
        words, merge the most frequent, and only rewrite the words that
        actually contained it. Rescanning the whole corpus each round does not
        scale to Jupiter's documents.
        """
        words = Counter()
        for text in texts:
            for word in pretokenize(text):
                words[tuple(word.encode("utf-8", errors="replace"))] += 1
        words = {word: count for word, count in words.items() if len(word) > 1}

        pair_counts = Counter()
        pair_words = {}
        for word, count in words.items():
            for pair in set(self._word_pairs(word)):
                pair_counts[pair] += count
                pair_words.setdefault(pair, set()).add(word)

        merges = []
        for step in range(num_merges):
            if not pair_counts:
                break
            # Ties break on the pair itself so training is deterministic.
            pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merges.append(list(pair))
            new_id = BASE_VOCAB + len(merges) - 1
            for word in list(pair_words.get(pair, ())):
                self._rewrite(words, word, pair, new_id, pair_counts, pair_words)
            if progress is not None and (step + 1) % 200 == 0:
                progress(step + 1, num_merges)

        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.vocab_size = BASE_VOCAB + len(self.merges)
        self._cache = {}
        return self

    def _rewrite(self, words, word, pair, new_id, pair_counts, pair_words):
        """Apply one merge to a single word and repair the pair indexes."""
        count = words.get(word, 0)
        for old in set(self._word_pairs(word)):
            pair_counts[old] -= count
            if pair_counts[old] <= 0:
                del pair_counts[old]
            pair_words.get(old, set()).discard(word)
        merged = []
        index = 0
        while index < len(word):
            if index < len(word) - 1 and (word[index], word[index + 1]) == pair:
                merged.append(new_id)
                index += 2
            else:
                merged.append(word[index])
                index += 1
        updated = tuple(merged)
        if updated == word:
            return
        del words[word]
        words[updated] = words.get(updated, 0) + count
        for new in set(self._word_pairs(updated)):
            pair_counts[new] += count
            pair_words.setdefault(new, set()).add(updated)

    # --- use ---

    def _encode_word(self, word):
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word)
        while len(ids) > 1:
            best_rank = None
            best_at = None
            for index in range(len(ids) - 1):
                rank = self.ranks.get((ids[index], ids[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_at = index
            if best_at is None:
                break
            # best_rank is set whenever best_at is, because both are assigned
            # in the same branch and nothing clears either. The explicit
            # check is here because the arithmetic below cannot be read
            # correctly on its own: a reader cannot see that the None case
            # has already broken, and the token id it produces is silent
            # wrong rather than loud.
            if best_rank is None:
                break
            ids[best_at : best_at + 2] = [BASE_VOCAB + best_rank]
        self._cache[word] = ids
        return ids

    def encode(self, text):
        """Encode text to token ids. Never fails, and never drops input."""
        ids = []
        for word in pretokenize(text):
            ids.extend(self._encode_word(tuple(word.encode("utf-8", errors="replace"))))
        return ids

    def _token_bytes(self, value):
        """The bytes a single token stands for.

        A merge token stands for the concatenation of the two tokens it joined,
        recursively, so every id in the vocabulary expands back to bytes and
        nothing can be undecodable.
        """
        if 0 <= value < BASE_VOCAB:
            return bytes([value])
        merge_index = value - BASE_VOCAB
        if 0 <= merge_index < len(self.merges):
            left, right = self.merges[merge_index]
            return self._token_bytes(left) + self._token_bytes(right)
        return b""

    def decode(self, ids):
        """Decode token ids back to text."""
        pieces = bytearray()
        for token in ids:
            pieces.extend(self._token_bytes(int(token)))
        return bytes(pieces).decode("utf-8", errors="replace")

    def validate(self, ids):
        for token in ids:
            if not 0 <= int(token) < self.vocab_size:
                raise ValueError(
                    f"token {token} is outside the vocabulary of {self.vocab_size}"
                )

    def compression(self, text):
        """Bytes per token on this text. Architecture 8 asks for this number."""
        encoded = self.encode(text)
        if not encoded:
            return 0.0
        return round(len(text.encode("utf-8", errors="replace")) / len(encoded), 3)

    # --- persistence ---

    def to_dict(self):
        return {
            "tokenizer_version": self.version,
            "kind": "byte-level-bpe",
            "base_vocab": BASE_VOCAB,
            "merges": [list(pair) for pair in self.merges],
            "vocab_size": self.vocab_size,
            "trained_on": self.trained_on,
        }

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path):
        target = Path(path)
        if not target.exists():
            raise TokenizerError(
                f"no tokenizer vocabulary at {target}. Train one with "
                f"train_tokenizer.py first, or run with the byte tokenizer."
            )
        data = json.loads(target.read_text(encoding="utf-8"))
        if data.get("kind") != "byte-level-bpe":
            raise TokenizerError(
                f"{target} is not a byte-level BPE vocabulary: "
                f"kind={data.get('kind')!r}"
            )
        return cls(
            data.get("merges", []),
            version=data.get("tokenizer_version", "unknown"),
            trained_on=data.get("trained_on"),
        )


def train_tokenizer(c, num_merges=2000, limit=None, progress=None):
    """Train a tokenizer from the curated corpus and save it.

    How much of each document the vocabulary is trained on is a decision with a
    large measured consequence, and it was set by a constant meant for a
    different purpose. c.max_chunk_chars is how much of a document ONE LEARNING
    CYCLE reads, which is the right size for a cycle and far too small for a
    vocabulary: a 57MB document contributes its first 12,000 characters and
    nothing else. Thirty documents therefore trained the vocabulary on 269,191
    characters, and produced 1.66 bytes per token on the corpus that produced it,
    against 2.29 when the same merge count was trained on 1.5MB of that same
    corpus. A vocabulary is a summary of the whole corpus or it is a summary of
    whichever document happened to sort first.

    The cap is its own constant, large enough that a big document is represented
    rather than sampled from its opening, and still bounded so that training does
    not read 346MB into memory.
    """
    root = Path(c.curated_dir)
    documents = sorted(root.glob("*.txt"))
    if limit:
        documents = documents[:limit]
    if not documents:
        raise TokenizerError(
            f"no curated documents in {root}. Run curate_dataset.py first."
        )
    texts = [
        path.read_text(encoding="utf-8", errors="replace")[:VOCAB_TRAIN_CHARS]
        for path in documents
    ]
    tokenizer = BpeTokenizer([])
    tokenizer.trained_on = {
        "documents": len(documents),
        "characters": sum(len(text) for text in texts),
    }
    tokenizer.train(texts, num_merges=num_merges, progress=progress)
    return tokenizer
