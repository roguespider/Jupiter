"""Inspect learning material and prepare the tokenizer.

There is deliberately no train.bin/val.bin conversion here. Text stays text,
and the learning controller chooses chunks for each micro-update.

This is also where the tokenizer is prepared, because Architecture 8 says a
tokenizer choice has to be measurable and not merely convenient. The report
compares bytes per token for the byte tokenizer and for the learned BPE
vocabulary, so the cost and the benefit of switching are numbers rather than an
opinion. The vocabulary is trained only when it is missing, which keeps a
tokenizer experiment to one variable at a time, as Architecture 42 requires.
"""

import json
from pathlib import Path

import tokenizer as tokenizer_module
import tokenizer_bpe

from config import load_config

COMPARISON_CHARACTERS = 4000


def ensure_bpe(c, force=False):
    """Train the BPE vocabulary when it is missing. Returns the tokenizer."""
    path = Path(c.tokenizer_vocab)
    if path.exists() and not force:
        return tokenizer_bpe.BpeTokenizer.load(path)
    trained = tokenizer_bpe.train_tokenizer(c, num_merges=c.tokenizer_merges)
    trained.save(path)
    return trained


def comparison_text(c):
    """A little curated text, enough to measure a tokenizer on."""
    documents = sorted(Path(c.curated_dir).glob("*.txt"))
    if not documents:
        return ""
    return documents[0].read_text(encoding="utf-8", errors="replace")[
        :COMPARISON_CHARACTERS
    ]


def inspect(c=None, train_vocabulary=True):
    """Report the learning material and the tokenizer comparison."""
    c = c or load_config()
    byte_tokenizer = tokenizer_module.ByteTokenizer()
    documents = []
    for path in sorted(Path(c.curated_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.append(
            {
                "path": str(path),
                "characters": len(text),
                "byte_tokens": len(byte_tokenizer.encode(text)),
            }
        )
    report = {
        "documents": len(documents),
        "total_characters": sum(item["characters"] for item in documents),
        "total_byte_tokens": sum(item["byte_tokens"] for item in documents),
        "documents_detail": documents,
        "storage_model": "source text + curated text; no permanent binary dataset",
        "tokenizer_in_use": c.tokenizer,
    }
    if train_vocabulary and documents:
        bpe = ensure_bpe(c)
        text = comparison_text(c)
        report["tokenizer_vocab"] = str(c.tokenizer_vocab)
        report["tokenizer_version"] = bpe.version
        report["tokenizer_comparison"] = tokenizer_module.compare(text, bpe)
        report["tokenizer_comparison"]["note"] = (
            "bytes per token. Higher means more text fits in the same "
            f"{c.block_size} token window. Architecture 8 requires this to be "
            "measured rather than assumed."
        )
    target = Path(c.dataset_report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    report = inspect()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
