"""Jupiter's tokenizers (Architecture 8).

Two tokenizers live here, behind one interface:

    ByteTokenizer   lossless UTF-8 bytes, no vocabulary file, one token per byte
    BpeTokenizer    learned byte-level BPE, several characters per token

Both are lossless. Every byte tokenizer id is a byte, and every BPE id decodes
back to bytes, so neither can silently drop input Jupiter has not seen. That
matters because Architecture 8 requires the tokenizer to represent malformed
input, slang and typing errors as well as clean text.

BPE is the default because the byte tokenizer, at one byte per token, makes a
512-token window hold only 512 characters, which is too small to carry a
question, some retrieved knowledge and an answer. The byte tokenizer remains
available and is still the lossless fallback: it needs nothing learned and
cannot be invalidated. See tokenizer_bpe.py.

Architecture 8 also warns that changing tokenization after substantial
training invalidates existing weights, so switching is a migration rather than
a preference. train.py's shape guard notices the vocabulary change and archives
the old checkpoint instead of misusing it.
"""

import tokenizer_bpe

BYTE = "bytes"
BPE = "bpe"
KINDS = (BYTE, BPE)


class ByteTokenizer:
    """UTF-8 bytes, one token each. The seed tokenizer."""

    name = BYTE
    version = "1.0"
    vocab_size = tokenizer_bpe.BASE_VOCAB

    def encode(self, text):
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids):
        values = bytes(int(i) % self.vocab_size for i in ids)
        return values.decode("utf-8", errors="replace")

    def validate(self, ids):
        for token in ids:
            if not 0 <= int(token) < self.vocab_size:
                raise ValueError(f"Invalid token id: {token}")

    def compression(self, text):
        """Bytes per token. Always 1, reported so both tokenizers compare."""
        encoded = self.encode(text)
        if not encoded:
            return 0.0
        return round(len(text.encode("utf-8", errors="replace")) / len(encoded), 3)


def load_tokenizer(c):
    """Return the tokenizer the configuration asks for.

    A BPE configuration with no vocabulary file raises rather than quietly
    falling back to bytes, because silently training a byte-level model against
    a configuration that asked for something else is exactly the kind of
    mismatch Architecture 8 warns about.
    """
    kind = getattr(c, "tokenizer", BYTE)
    if kind == BYTE:
        return ByteTokenizer()
    if kind == BPE:
        return tokenizer_bpe.BpeTokenizer.load(c.tokenizer_vocab)
    raise ValueError(f"unknown tokenizer {kind!r}; expected one of {', '.join(KINDS)}")


def compare(text, bpe):
    """Bytes per token for both tokenizers, which is Architecture 8's measure."""
    return {
        "characters": len(text),
        "byte_tokenizer": ByteTokenizer().compression(text),
        "bpe_tokenizer": bpe.compression(text),
    }


# The module-level interface below is the byte tokenizer, kept because the data
# pipeline and the existing tests use it directly.
#
# There is deliberately no VOCAB_SIZE constant here. There was one, it
# duplicated ByteTokenizer.vocab_size, and nothing read it, so a reader could
# not tell which of the two was authoritative and neither was checked against
# the other. A vocabulary size that is a property of the tokenizer is read from
# the tokenizer.
_BYTE = ByteTokenizer()


def encode(text: str) -> list[int]:
    return _BYTE.encode(text)


def decode(ids) -> str:
    return _BYTE.decode(ids)


def validate(ids) -> None:
    _BYTE.validate(ids)
