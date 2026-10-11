"""Remove digest-shaped concepts and the edges that only exist because of them.

Before this, a conflict between two documents had a concept invented for each
side. Curation names a curated file by the sha256 of its contents, so the concept
table accumulated sixty-eight character names, every one of them with zero
evidence and a confidence of zero, and the relationship graph filled with edges
between them: 268 concepts and 2,276 CONTRADICTS edges on the live store, 22% of
the graph.

Nothing here guesses at what a digest is. A name is treated as a filename if it
is a sha256 with an optional .txt suffix, which is the one shape curation
produces and the only one seen in the store.

What is removed:

  - the digest-shaped concept records themselves
  - edges where BOTH endpoints are digest-shaped, which only exist because the
    two shells were adjacent in the graph
  - the dangling references to those edges from the concepts that listed them

What is deliberately NOT removed:

  - an edge between a digest and a real word. Those name something real, and
    dropping the digest endpoint would silently discard a record about an actual
    term. They are reported so the owner can decide.
  - anything with evidence. A digest that somehow acquired evidence is not
    noise and is reported rather than deleted.

Dry run by default. --apply writes, and only after backing the store up.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import concepts
import relationships as relationships_module
import store as store_module

from config import load_config

BACKUP_SUFFIX = ".pre_digest_cleanup"


def digest_shaped(name):
    """Whether a name is a curated filename rather than a term.

    Curation writes data/curated/<sha256>.txt, so the shape to recognise is a
    lowercase hex run of at least 32 characters with an optional .txt suffix. A
    real term that happens to be written in hex is not a thing this corpus
    contains, and the length floor is what keeps short words like "beefed" and
    "facade" out of it.
    """
    text = str(name).strip().lower()
    if text.endswith(".txt"):
        text = text[:-4]
    if len(text) < 32:
        return False
    return all(character in "0123456789abcdef" for character in text)


def survey(store):
    """What the store holds, before anything is changed."""
    concepts_held = concepts.all_concepts(store)
    shells = []
    evidenced = []
    for record in concepts_held:
        identifier = str(record.get("identifier", ""))
        if not digest_shaped(identifier):
            continue
        evidence = int(record.get("evidence", 0) or 0)
        contradictions = int(record.get("contradictions", 0) or 0)
        confidence = float(record.get("confidence", 0.0) or 0.0)
        usage = int(record.get("usage_frequency", 0) or 0)
        if evidence or contradictions or confidence or usage:
            evidenced.append((identifier, evidence, contradictions, confidence))
        else:
            shells.append(identifier)

    edges = relationships_module.all_relationships(store)
    shell_set = set(shells)
    both_shell = []
    one_shell = []
    for edge in edges:
        left, right = str(edge["from"]), str(edge["to"])
        left_shell = left in shell_set
        right_shell = right in shell_set
        if left_shell and right_shell:
            both_shell.append(edge)
        elif left_shell or right_shell:
            one_shell.append(edge)

    # Which concepts still point at the edges that will go.
    doomed_keys = {
        relationships_module._key(e["from"], e["relation"], e["to"]) for e in both_shell
    }
    referencing = []
    for record in concepts_held:
        listed = record.get("relationships") or []
        hit = [key for key in listed if key in doomed_keys]
        if hit:
            referencing.append((str(record.get("identifier", "")), len(hit)))

    # Edges that no longer exist at all, from any cause. A concept listing an
    # edge id that is not in the collection is holding a name that resolves to
    # nothing, and a traversal that trusts it learns something that is not there.
    # The shell purge created these, and so did the older bug that invented the
    # shells, so they are repaired here rather than left for whoever reads them.
    live_keys = set(store.collection(relationships_module.RELATIONSHIPS).keys())
    orphaned = []
    for record in concepts_held:
        listed = record.get("relationships") or []
        missing = [key for key in listed if key not in live_keys]
        if missing:
            orphaned.append((str(record.get("identifier", "")), missing))

    return {
        "shells": shells,
        "evidenced": evidenced,
        "both_shell": both_shell,
        "one_shell": one_shell,
        "referencing": referencing,
        "doomed_keys": doomed_keys,
        "orphaned": orphaned,
        "concepts_total": len(concepts_held),
        "edges_total": len(edges),
    }


def apply_cleanup(store, findings):
    """Remove the shells and their exclusive edges, and repair the back-references.

    Edges with only one shell endpoint are left in place, so their real-word
    endpoint keeps whatever it was told, and the shell's own record goes away with
    the shell.

    Deletion is expressed as a rewrite of the surviving records rather than as a
    removal, because Store has no delete: it is a document of named collections
    and the way to drop a key is to write the document without it. put_many
    rewrites a whole collection, so each collection is written once inside the
    caller's transaction rather than once per key.
    """
    doomed_keys = findings["doomed_keys"]
    shells = set(findings["shells"])
    repaired = 0

    # The edges that survive are rewritten first, because the surviving concepts
    # have to be told which of their references no longer resolve, and that
    # answer depends on which edges went.
    removed_edges = store.remove(relationships_module.RELATIONSHIPS, doomed_keys)

    surviving_concepts = []
    for key, record in store.collection(concepts.CONCEPTS).items():
        if key in shells:
            continue
        listed = record.get("relationships") or []
        kept = [edge for edge in listed if edge not in doomed_keys]
        if len(kept) != len(listed):
            record["relationships"] = kept
            repaired += 1
        surviving_concepts.append((key, record))
    if repaired:
        store.put_many(concepts.CONCEPTS, surviving_concepts)

    # Then any reference to an edge that does not exist at all, whatever removed
    # it. This runs after the edge removal so the edges this purge is taking go
    # are included in what counts as missing.
    live_keys = set(store.collection(relationships_module.RELATIONSHIPS).keys())
    repaired_now = 0
    updates = []
    for key, record in store.collection(concepts.CONCEPTS).items():
        if key in shells:
            continue
        listed = record.get("relationships") or []
        kept = [edge for edge in listed if edge in live_keys]
        if len(kept) != len(listed):
            record["relationships"] = kept
            updates.append((key, record))
            repaired_now += 1
    if updates:
        store.put_many(concepts.CONCEPTS, updates)
        repaired += repaired_now

    # Deletion last, and through remove() rather than by writing the survivors
    # back. put_many merges, so writing the collection without the shells left
    # every one of them in place: the first version of this reported removing two
    # hundred and sixty-eight concepts and removed none of them.
    removed_concepts = store.remove(concepts.CONCEPTS, shells)
    return removed_edges, removed_concepts, repaired


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="purge_digest_concepts",
        description=(
            "Remove document-digest concepts and the edges that exist only "
            "because of them. Dry run unless --apply is given."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the changes, after backing the store up",
    )
    args = parser.parse_args(argv)

    c = load_config()
    path = Path(c.out_dir) / "knowledge.json"
    if not path.exists():
        print(f"  no store at {path}")
        return 1

    store = store_module.Store(path)
    findings = survey(store)

    print(f"=== survey of {path} ===")
    print(f"  concepts held      : {findings['concepts_total']}")
    print(f"  relationships held : {findings['edges_total']}")
    print()
    print(f"  digest-shaped concepts with no evidence : {len(findings['shells'])}")
    print(f"  digest-shaped concepts WITH evidence    : {len(findings['evidenced'])}")
    for identifier, evidence, contradictions, confidence in findings["evidenced"][:10]:
        print(
            f"    KEPT {identifier[:52]} evidence={evidence} "
            f"contradictions={contradictions} confidence={confidence}"
        )
    print()
    print(f"  edges with BOTH endpoints digest-shaped  : {len(findings['both_shell'])}")
    print(f"  edges with ONE digest endpoint           : {len(findings['one_shell'])}")
    print(f"    (left alone: their other endpoint names something real)")
    for edge in findings["one_shell"][:5]:
        left, right = str(edge["from"]), str(edge["to"])
        word = right if digest_shaped(left) else left
        print(f"      {word[:40]:<42} {edge['relation']}")
    print()
    print(
        f"  concepts holding a reference to a doomed edge: {len(findings['referencing'])}"
    )
    for identifier, count in findings["referencing"][:5]:
        print(f"    {identifier[:52]} -> {count} dangling")

    if not args.apply:
        print()
        print("  dry run. Nothing was written. Re-run with --apply to remove them.")
        return 0

    if findings["evidenced"]:
        print()
        print("  refusing to run: a digest concept carries evidence, and deleting")
        print("  something that was learned is not a cleanup. Investigate first.")
        return 1

    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    shutil.copy2(path, backup)
    print()
    print(f"  backed up to {backup.name}")

    with store.transaction():
        removed_edges, removed_concepts, repaired = apply_cleanup(store, findings)
    print(f"  removed {removed_concepts} digest concepts")
    print(f"  removed {removed_edges} shell-to-shell edges")
    print(f"  repaired {repaired} concepts holding dangling edge references")

    after = survey(store)
    print()
    print("=== after ===")
    print(f"  digest-shaped concepts with no evidence : {len(after['shells'])}")
    print(f"  edges with BOTH endpoints digest-shaped  : {len(after['both_shell'])}")
    print(f"  edges with ONE digest endpoint           : {len(after['one_shell'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
