"""Material Jupiter writes for itself, and the gate that decides it may.

Architecture 25 lists where data may come from. Curated datasets, documentation
and conversations were implemented; generated examples and Jupiter's own
experiences were listed and never built. That left the loop closed in the wrong
place: every training byte had been placed there by a person, so Jupiter could
only ever learn what it had already been handed, and could never encounter a
case somebody did not think to include.

This closes that. When a source keeps defeating a bounded update for a reason
that more material could address, Jupiter writes more of that material itself
by continuing the source through its own weights, and the result is written into
data/raw/generated/ as ordinary raw text.

Three properties are deliberate and load-bearing.

It does not bypass curation. Generated text is raw data like any other. It is
parsed, cleaned, classified and deduplicated by curate_dataset, and if it comes
out contaminated or too short it is refused exactly as a human file would be.
A self-feeding loop is the last place that should have a private path.

The generator is gated on the cause, not on the frustration score. Architecture
21 names seven causes for repeated difficulty. Only insufficient data and
inadequate context are answered by writing more material. A representation the
model cannot compress, or a capacity limit, is not, and Architecture 23's
caution about acting on a measurement applies here as much as it does to
expansion. Generating into those would reinforce the failure rather than fix it.

Whether the generated material helped is not a separate claim. It becomes a
normal candidate, so the same acceptance gate, the same reserved regression
document and the same pre/post measurement apply to it as to anything else. A
generated document that does not earn an ACCEPT is rejected on the evidence, the
same as any other, and that verdict is what measures it.
"""

import time
from pathlib import Path

import control
import selection
import torch

# Prefixed so a generated document is identifiable in the curation report, in
# the experiment log and in data/raw, without a separate manifest to fall out of
# step with the files.
PREFIX = "generated"

# A continuation that decodes to nothing, or to a fragment of one, is not
# material. The window below is generous because the byte tokenizer can emit
# very short runs before it settles into anything readable.
MIN_CHARS = 400
MIN_PRINTABLE = 0.9


def generated_dir(c):
    return Path(c.synthesis_dir)


def is_generated(path) -> bool:
    """Whether a document came from this module, by name."""
    return Path(path).name.startswith(PREFIX + "-")


def printable_ratio(text):
    if not text:
        return 0.0
    return sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)


def usable(text) -> bool:
    """Whether generated text is worth writing into the raw corpus at all.

    Deliberately a precondition on writing rather than a filter applied after,
    because data/raw is the source of truth and is never pruned. Junk written
    there would have to be dealt with by a human later. Curation still judges
    the result; this only stops obviously worthless output being filed.
    """
    return len(text) >= MIN_CHARS and printable_ratio(text) >= MIN_PRINTABLE


def continuation(model, c, tokenizer, seed):
    """Text the model itself produced, continuing `seed`.

    Jupiter is a byte-level continuation model rather than an instruction
    follower, so it is asked to continue rather than to answer. Pretending
    otherwise would generate text in a layout it has never been trained on and
    then measure the result, which is Architecture 41's unit test with the
    order reversed.
    """
    ids = tokenizer.encode(seed)
    if not ids:
        return None
    window = ids[-c.block_size :]
    start = torch.tensor([window], dtype=torch.long, device=c.device)
    with torch.no_grad():
        produced = model.generate(
            start,
            c.synthesis_max_new_tokens,
            c.temperature,
            c.top_k,
        )
    fresh = produced[0, start.shape[1] :]
    return tokenizer.decode(fresh.tolist())


def document_for(model, c, tokenizer, source_path, source_text):
    """One generated training document: the seed, then what the model added.

    The seed is kept, and it is kept short. A continuation without the passage
    that prompted it is a fragment with no context, and learning on fragments
    teaches the model to produce fragments. But a seed is a prompt, not
    material: seeding with an entire document produced a file that was almost
    entirely a copy of a file already in the corpus, which dedup cannot catch
    because the appended continuation changes the digest, and which then sits
    in data/raw forever because raw is never pruned. The seed is therefore
    windowed to the same order of size a cycle actually learns.
    """
    seed = source_text[: c.synthesis_seed_chars].rstrip()
    if not seed:
        return None
    made = continuation(model, c, tokenizer, seed)
    if made is None:
        return None
    body = f"{seed}\n{made.strip()}\n"
    if not usable(body):
        return None
    return body


def write(c, source_path, text, cycle):
    """File one generated document and report where it went.

    The name carries the seed it came from and the cycle that asked for it, so
    provenance survives in the filename. That matters more here than elsewhere:
    a self-feeding corpus is very hard to reason about if you cannot tell which
    parts of it Jupiter wrote.
    """
    directory = generated_dir(c)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem[:40]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    target = directory / f"{PREFIX}-{stem}-c{cycle}-{stamp}.txt"
    counter = 0
    while target.exists():
        counter += 1
        target = directory / f"{PREFIX}-{stem}-c{cycle}-{stamp}-{counter}.txt"
    # The output directory is configuration, and configuration can be wrong. A
    # generator that can be pointed at the source tree will eventually litter
    # it, so the path is checked against what Jupiter is allowed to write.
    control.check_write(c, target)
    target.write_text(text, encoding="utf-8")
    return target


def targets_for(measurements):
    """The sources worth writing more material for, most frustrated first.

    `measurements` is frustration's per-source output. Anything whose cause is
    not one more material can fix is dropped, so this returns a short list or
    nothing at all. Returning nothing is the normal case and is not a failure.
    """
    scored = []
    for record in measurements or []:
        if not isinstance(record, dict):
            continue
        # frustration names the field likely_cause; "cause" is not a field and
        # reading it would silently gate every source out.
        cause = record.get("likely_cause", "")
        if not selection.worth_generating(cause):
            continue
        try:
            index = float(record.get("index", 0.0))
        except (TypeError, ValueError):
            index = 0.0
        scored.append((-index, str(record.get("source", "")), record))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored]


def run(c, model, tokenizer, pieces, measurements, cycle, tel, limit=None):
    """Write up to `limit` generated documents, and report what was written.

    Returns the list of paths written, which may legitimately be empty. Every
    decision is recorded: a refused cause, an unusable continuation and a
    written document are all things a later reader needs to see, because a
    generator that silently does nothing looks exactly like a generator that is
    broken.
    """
    budget = c.synthesis_per_run if limit is None else int(limit)
    if budget <= 0:
        return []
    by_source = {selection.key_for(path): text for path, text in pieces}
    written = []
    for record in targets_for(measurements):
        if len(written) >= budget:
            break
        source = str(record.get("source", ""))
        seed = by_source.get(source)
        if not seed:
            tel.write(
                "synthesis_skipped",
                learning_cycle=cycle,
                source=source,
                reason="source is not in the current candidate set",
            )
            continue
        body = document_for(model, c, tokenizer, source, seed)
        if body is None:
            tel.write(
                "synthesis_skipped",
                learning_cycle=cycle,
                source=source,
                reason="the model produced too little usable text to keep",
            )
            continue
        target = write(c, source, body, cycle)
        written.append(target)
        tel.write(
            "synthesis_written",
            learning_cycle=cycle,
            source=source,
            cause=record.get("likely_cause"),
            frustration_index=record.get("index"),
            generated=str(target),
            characters=len(body),
        )
        print(
            f"\n[synthesis] wrote {target.name} ({len(body)} chars) "
            f"from {Path(source).name}; cause {record.get('likely_cause')}"
        )
    return written
