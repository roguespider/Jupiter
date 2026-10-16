"""Jupiter interactive agent runtime.

Talk to Jupiter with what it has learned available to it.

This is a persistent session, not a single prompt. Jupiter is a byte-level
language model rather than an instruction-tuned assistant, so the conversation
is one growing text stream: every human turn is appended to the history and
the model continues from there.

On each turn:

1. Relevant stored knowledge is retrieved.
2. Knowledge and conversation are assembled into the model's context.
3. Jupiter generates a response.
4. The complete interaction is recorded as an experience.

The experience record is deliberately separate from the model weights.
Conversation therefore becomes learning material without requiring a training
run during the conversation itself.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import dialogue
import experience_loop
import store as store_module
import torch
from agent_memory import AgentMemory
from model import Jupiter, ModelConfig
from safetensors.torch import load_model
from tokenizer import load_tokenizer

import retrieval
from config import load_config, read_architecture

HELP = (
    "/exit     end the session\n"
    "/reset    forget the conversation context\n"
    "/history  show how much context is still active\n"
    "/recall   show what is currently retrievable\n"
    "/experiences show how many experiences have been recorded\n"
    "/replay   show past experiences for decision evidence\n"
    "/remember <kind> <text>  save a durable memory\n"
    "/correct <old> => <new>  record a correction\n"
    "/feedback <helpful|wrong|partial|uncertain> [details]  rate last interaction\n"
    "/task add <title>  create a task\n"
    "/task list  list open tasks\n"
    "/memory   report memory counts\n"
    "/corrections [query]  list corrections relevant to query\n"
    "/procedure save <name> <steps...>  store a reusable procedure\n"
    "/procedure list  list saved procedures\n"
    "/procedure get <id>  show a procedure\n"
    "/learn    show recent learning cycle summary\n"
    "/help     show this message"
)

RECALL_LINES = 3


def load_jupiter(c):
    """Load the current Jupiter SafeTensors checkpoint."""

    checkpoint = Path(c.out_dir) / "Jupiter_latest.safetensors"
    architecture = Path(c.out_dir) / "Jupiter.architecture.json"

    if not checkpoint.exists() or not architecture.exists():
        missing = [
            str(path) for path in (checkpoint, architecture) if not path.exists()
        ]

        raise FileNotFoundError(
            "Run run_jupiter.bat first so Jupiter has a SafeTensors "
            f"checkpoint. Missing: {', '.join(missing)}"
        )

    stored, _version = read_architecture(architecture)
    cfg = ModelConfig(**stored)

    model = Jupiter(cfg).to(c.device)
    load_model(model, str(checkpoint), strict=True)
    model.eval()

    return model, cfg


def load_knowledge(c):
    """Open the knowledge store, or None when it does not exist yet."""

    path = Path(c.knowledge_store)

    if not path.exists():
        return None

    return store_module.Store(path)


def recalled_context(knowledge, question, max_chars):
    """Retrieve relevant stored knowledge for the current question."""

    if knowledge is None:
        return "", []

    results = retrieval.retrieve(knowledge, question)

    return retrieval.context_block(
        results,
        max_chars=max_chars,
    ), results


# Minimal stopword set for experience replay word matching.
# These words are so common that they match almost everything and
# produce noise rather than signal when used for overlap scoring.
REPLAY_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "because",
    "but",
    "and",
    "or",
    "if",
    "while",
    "about",
}


def _query_words(text):
    """Extract meaningful words from text, filtering stopwords.

    Returns a set of lowercased words with punctuation stripped.
    Stopwords are excluded so overlap scores reflect substantive
    word matching rather than universal words like 'the' or 'is'.
    """
    words = set()
    for token in str(text).split():
        cleaned = token.lower().strip(".,;:!?\"'()[]{}")
        if cleaned and cleaned not in REPLAY_STOPWORDS:
            words.add(cleaned)
    return words


def experience_replay(c, question, max_items=3):
    """Retrieve relevant past experiences, prioritising successes.

    Experiences are scored by lexical overlap (filtered for stopwords)
    and outcome: successful interactions are preferred so the model can
    reuse what worked rather than repeating what failed. Failures are
    included at lower priority as cautionary evidence.

    Returns (formatted_text, relevant_records).
    """
    records, unreadable = experience_loop.read(c, limit=max_items * 2)
    if not records:
        return "", []
    question_words = _query_words(question)
    scored = []
    for record in records:
        user_text = str(record.get("user", ""))
        jupiter_text = str(record.get("jupiter", ""))
        outcome = record.get("outcome", "")
        attempt = record.get("attempt", "")
        recommendation = record.get("recommendation", "")
        expected = record.get("expected", "")
        overlap = len(question_words & _query_words(user_text))
        # Skip experiences with zero overlap when the problem is substantive.
        if overlap == 0 and len(user_text) >= 100:
            continue
        # Score: base overlap, boosted by success.
        score = overlap
        if outcome in ("helpful", "success"):
            score += 10
        elif outcome in ("wrong", "failure"):
            score += 2  # low priority but still useful as a caution
        scored.append((score, record))
    scored.sort(key=lambda x: -x[0])
    relevant = [r for _, r in scored[:max_items]]
    lines = []
    for r in relevant:
        outcome = r.get("outcome", "")
        attempt = r.get("attempt", "")
        recommendation = r.get("recommendation", "")
        expected = r.get("expected", "")
        user_text = str(r.get("user", "")).strip()[:100]
        jupiter_text = str(r.get("jupiter", "")).strip()[:100]
        parts = [f"[experience] user: {user_text}"]
        if attempt:
            parts.append(f"tried: {attempt}")
        if expected:
            parts.append(f"expected: {expected}")
        if outcome:
            parts.append(f"outcome: {outcome}")
        if recommendation:
            parts.append(f"next: {recommendation}")
        parts.append(f"jupiter: {jupiter_text}")
        lines.append(" | ".join(parts))
    return "\n".join(lines), relevant


def report_recall(results):
    """Show what knowledge was recalled for this turn."""

    if not results:
        print("[no recorded knowledge matched that]")
        return

    for item in results[:RECALL_LINES]:
        print(
            f"[recalled {item['collection']} {item['id']} "
            f"score={item['score']} on {', '.join(item['matched'])}]"
        )


def assemble(prefix_tokens, history, block_size):
    """Fit recalled knowledge and conversation into one model window.

    Knowledge is placed first and retained preferentially. Older conversation
    is removed when the window becomes full.
    """

    if prefix_tokens is None or prefix_tokens.numel() == 0:
        return history[:, -block_size:]

    prefix_tokens = prefix_tokens[:, -block_size:]

    room = max(
        0,
        block_size - prefix_tokens.shape[1],
    )

    kept = history[:, -room:] if room else history[:, :0]

    return torch.cat(
        (
            prefix_tokens,
            kept,
        ),
        dim=1,
    )


def generation_context(prefix_tokens, history, turn, block_size):
    """Build the exact context Jupiter should continue from.

    The current human turn must be included before generation. This prevents
    the first conversation turn from producing an empty generation context.
    """

    pending = torch.cat(
        (
            history[:, -block_size:],
            turn,
        ),
        dim=1,
    )

    return assemble(
        prefix_tokens,
        pending,
        block_size,
    )


def generate(model, cfg, c, tokenizer, context):
    """Generate Jupiter's response from the supplied context."""

    with torch.no_grad():
        result = model.generate(
            context,
            c.max_new_tokens,
            c.temperature,
            c.top_k,
        )

    fresh = result[0, context.shape[1] :]

    return tokenizer.decode(fresh.tolist())


def _console_encoding():
    """Return the encoding supported by the current console."""

    return (
        getattr(
            sys.stdout,
            "encoding",
            None,
        )
        or "utf-8"
    )


def experience_path(c):
    """Return the persistent JSONL experience file path.

    This intentionally does not require a new configuration setting. The
    experience store lives underneath Jupiter's existing output directory.
    """

    return Path(c.out_dir) / "experiences.jsonl"


def record_experience(
    c,
    user_text,
    jupiter_text,
    recalled,
    attempt=None,
    outcome=None,
    recommendation=None,
    expected=None,
    confidence=0.0,
    concepts=None,
):
    """Record one completed interaction as a durable experience.

    Experiences are observations, not automatic weight updates.

    Architecture 46 says experiences should capture: input, interpretation,
    attempted response, expected result, actual result, confidence, relevant
    concepts. This record includes those fields so the replay system can
    score and reuse experiences as evidence rather than repeating them blindly.

    Nothing here modifies the model. The future learning system can consume
    these records, curate them, score them, and decide whether an experience
    deserves a micro-update.
    """

    path = experience_path(c)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "conversation",
        "user": user_text,
        "jupiter": jupiter_text,
        "recalled": [
            {
                "collection": item.get("collection"),
                "id": item.get("id"),
                "score": item.get("score"),
                "matched": item.get("matched", []),
            }
            for item in recalled
        ],
        "attempt": attempt or "",
        "outcome": outcome or "",
        "recommendation": recommendation or "",
        "expected": expected or "",
        "confidence": confidence,
        "concepts": concepts or [],
    }

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def experience_count(c):
    """Return the number of recorded experiences."""

    path = experience_path(c)

    if not path.exists():
        return 0

    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                count += 1

    return count


def main():
    c = load_config()

    tokenizer = load_tokenizer(c)

    model, cfg = load_jupiter(c)

    knowledge = load_knowledge(c)

    memory = AgentMemory(str(Path(c.out_dir) / "agent_memory.sqlite3"))

    history = torch.zeros(
        (1, 0),
        dtype=torch.long,
        device=c.device,
    )

    # Keep knowledge from consuming the entire context window.
    recall_budget = max(
        32,
        cfg.block_size // 3,
    )

    print(
        f"Jupiter ready on {c.device} "
        f"(block_size={cfg.block_size}, "
        # The vocabulary comes from the architecture record rather than the
        # configuration. Config.vocab_size was renamed to byte_vocab_size
        # precisely because it was a seed default that read as the model's real
        # vocabulary, and printing it here reported 256 against a model that is
        # 5256 wide.
        f"vocab_size={cfg.vocab_size}, "
        f"tokenizer={tokenizer.name}, "
        f"parameters={model.parameter_count():,})."
    )

    if knowledge is None:
        print(
            "No knowledge store yet. Run run_jupiter.bat so Jupiter has "
            "something to recall, or ask anyway."
        )

    print(f"Experience store: {experience_path(c)}")

    print(f"Recorded experiences: {experience_count(c)}")

    print("Type /help for commands, /exit to finish.")

    interaction_id = None

    while True:
        try:
            line = input("you> ")

        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = line.strip().lower()

        if command in (
            "/exit",
            "/quit",
            "exit",
            "quit",
        ):
            memory.close()
            break

        if command in (
            "/help",
            "/?",
        ):
            print(HELP)
            continue

        if command == "/reset":
            history = torch.zeros(
                (1, 0),
                dtype=torch.long,
                device=c.device,
            )

            print("[context cleared]")
            continue

        if command == "/history":
            print(f"active context: {history.shape[1]}/{cfg.block_size} tokens")
            continue

        if command == "/experiences":
            print(f"recorded experiences: {experience_count(c)}")
            print(f"experience file: {experience_path(c)}")
            continue

        if command == "/learn":
            records, _ = experience_loop.read(c, limit=10)
            if not records:
                print("[no experiences recorded yet]")
            else:
                print(f"Recent learning cycle ({len(records)} experiences):")
                print()
                for i, rec in enumerate(records[-5:], 1):
                    user = str(rec.get("user", "")).strip()[:80]
                    jupiter = str(rec.get("jupiter", "")).strip()[:80]
                    outcome = rec.get("outcome", "unrated")
                    attempt = rec.get("attempt", "")
                    recommendation = rec.get("recommendation", "")
                    expected = rec.get("expected", "")
                    print(f"  [{i}] user: {user}")
                    print(f"      jupiter: {jupiter}")
                    print(f"      outcome: {outcome}")
                    if attempt:
                        print(f"      tried: {attempt[:120]}")
                    if expected:
                        print(f"      expected: {expected[:120]}")
                    if recommendation:
                        print(f"      next:  {recommendation[:120]}")
                    print()

        if command == "/replay":
            records, unreadable = experience_loop.read(c, limit=20)
            if not records:
                print("[no experiences recorded yet]")
                continue
            # Show experiences grouped by outcome: successes first, then
            # failures as cautionary evidence, then unrated.
            successes = []
            failures = []
            unrated = []
            for rec in records:
                outcome = str(rec.get("outcome", "")).strip().lower()
                user_text = str(rec.get("user", "")).strip()[:120]
                attempt = str(rec.get("attempt", "")).strip()[:120]
                expected = str(rec.get("expected", "")).strip()[:120]
                jupiter = str(rec.get("jupiter", "")).strip()[:120]
                recommendation = str(rec.get("recommendation", "")).strip()[:120]
                entry = f"  user: {user_text}\n  jupiter: {jupiter}"
                if attempt:
                    entry += f"\n  tried: {attempt}"
                if expected:
                    entry += f"\n  expected: {expected}"
                if recommendation:
                    entry += f"\n  next: {recommendation}"
                entry += f"\n  outcome: {outcome or 'unrated'}"
                if outcome in ("helpful", "success"):
                    successes.append(entry)
                elif outcome in ("wrong", "failure"):
                    failures.append(entry)
                else:
                    unrated.append(entry)
            print(
                f"Experience replay ({len(records)} total, "
                f"{len(successes)} success, {len(failures)} failure, "
                f"{len(unrated)} unrated):"
            )
            if successes:
                print("\n[successes - these worked and can be reused]")
                for i, entry in enumerate(successes, 1):
                    print(f"\n[success {i}]:")
                    print(entry)
            if failures:
                print("\n[failures - cautionary evidence]")
                for i, entry in enumerate(failures, 1):
                    print(f"\n[caution {i}]:")
                    print(entry)
            if unrated:
                print("\n[unrated - no outcome recorded]")
                for i, entry in enumerate(unrated, 1):
                    print(f"\n[unrated {i}]:")
                    print(entry)
            if unreadable:
                print(f"\n[skipped {unreadable} unreadable lines]")
            continue

        if command == "/recall":
            if knowledge is None:
                print(f"[no knowledge store at {c.knowledge_store}]")
                continue

            found = retrieval.browse(knowledge)

            if not found:
                print(f"[knowledge store {c.knowledge_store} is empty]")
                continue

            for item in found:
                print(
                    f"[{item['collection']} {item['id']} "
                    f"confidence={item['confidence']}]: "
                    f"{item['excerpt']}"
                )

            continue

        if command.startswith("/remember "):
            parts = line.replace("/remember ", "").split(" ", 1)
            kind = parts[0].strip() if parts else "fact"
            content = parts[1].strip() if len(parts) > 1 else ""
            if content:
                memory.remember(kind, content, source="user", confidence=0.95)
                print("[memory saved]")
            else:
                print("[usage: /remember <kind> <text>]")
            continue

        if command.startswith("/correct "):
            rest = line.replace("/correct ", "").strip()
            if "=>" in rest:
                old_text, corrected_text = rest.split("=>", 1)
                memory.correct(old_text.strip(), corrected_text.strip())
                print("[correction saved]")
            else:
                print("[usage: /correct <old> => <corrected>]")
            continue

        if command.startswith("/feedback "):
            feedback_text = line.replace("/feedback ", "").strip()
            outcome = "helpful"
            details = ""
            if feedback_text.startswith("wrong"):
                outcome = "wrong"
                details = feedback_text.replace("wrong", "").strip()
            elif feedback_text.startswith("partial"):
                outcome = "partial"
                details = feedback_text.replace("partial", "").strip()
            elif feedback_text.startswith("uncertain"):
                outcome = "uncertain"
                details = feedback_text.replace("uncertain", "").strip()
            try:
                if interaction_id is None:
                    raise KeyError(
                        "no interaction logged yet; generate a response first"
                    )
                memory.feedback(interaction_id, outcome, details)
                print("[feedback recorded]")
            except Exception as exc:
                print(f"[feedback error: {exc}]")
            continue

        if command.startswith("/task "):
            sub = line.replace("/task ", "").strip()
            if sub.startswith("add "):
                title = sub.replace("add ", "").strip()
                if title:
                    memory.add_task(title)
                    print("[task added]")
                else:
                    print("[usage: /task add <title>]")
            elif sub == "list":
                tasks = memory.open_tasks()
                if tasks:
                    for t in tasks:
                        print(
                            f"[task {t['id']}] {t['title']} (status={t['status']}, priority={t['priority']})"
                        )
                else:
                    print("[no open tasks]")
            else:
                print("[usage: /task add <title> | /task list]")
            continue

        if command == "/memory":
            stats = memory.stats()
            for k, v in stats.items():
                print(f"[memory {k}: {v}]")
            continue

        if command.startswith("/corrections"):
            query_part = line.replace("/corrections", "").strip()
            corrections = memory.recall_corrections(query_part if query_part else line)
            if corrections:
                for c_item in corrections:
                    original = c_item.get("metadata", {}).get(
                        "corrects", "(no original stored)"
                    )
                    print(
                        f"[correction] {c_item['content']} (corrects: {original[:100]})"
                    )
            else:
                print("[no stored corrections match this query]")
            continue

        if command.startswith("/procedure "):
            sub = line.replace("/procedure ", "").strip()
            if sub.startswith("save "):
                parts = sub.split(" ", 1)
                name = parts[0].replace("save ", "").strip()
                steps_text = parts[1].strip() if len(parts) > 1 else ""
                if name and steps_text:
                    steps = [s.strip() for s in steps_text.split(";")]
                    memory.save_procedure(
                        name=name,
                        steps=steps,
                        confidence=0.7,
                    )
                    print(f"[procedure '{name}' saved, {len(steps)} steps]")
                else:
                    print("[usage: /procedure save <name> <steps separated by ;>]")
            elif sub == "list":
                procedures = memory.list_procedures()
                if procedures:
                    for p in procedures:
                        print(
                            f"[procedure {p['id']}] {p['name']} "
                            f"(confidence={p['confidence']}, steps={len(p['steps'])})"
                        )
                else:
                    print("[no saved procedures]")
            elif sub.startswith("verify "):
                rest = sub.replace("verify ", "").strip()
                if " " in rest:
                    proc_id, result_text = rest.split(" ", 1)
                    proc_id = proc_id.strip()
                    result_text = result_text.strip()
                else:
                    proc_id = rest.strip()
                    result_text = ""
                if proc_id:
                    try:
                        memory.verify_procedure(proc_id, result_text or "verified")
                        print(
                            f"[procedure '{proc_id}' verified: {result_text or 'verified'}]"
                        )
                    except Exception as exc:
                        print(f"[verify error: {exc}]")
                else:
                    print("[usage: /procedure verify <id> [result]]")
            elif sub.startswith("update "):
                rest = sub.replace("update ", "").strip()
                if " prerequisites " in rest:
                    proc_id, prereqs_text = rest.split(" prerequisites ", 1)
                    proc_id = proc_id.strip()
                    prereqs = [p.strip() for p in prereqs_text.split(";") if p.strip()]
                    try:
                        memory.update_procedure(proc_id, prerequisites=prereqs)
                        print(f"[procedure '{proc_id}' prerequisites updated]")
                    except Exception as exc:
                        print(f"[update error: {exc}]")
                elif " expected " in rest:
                    proc_id, expected_text = rest.split(" expected ", 1)
                    proc_id = proc_id.strip()
                    try:
                        memory.update_procedure(proc_id, expected_result=expected_text)
                        print(f"[procedure '{proc_id}' expected result updated]")
                    except Exception as exc:
                        print(f"[update error: {exc}]")
                else:
                    print(
                        "[usage: /procedure update <id> prerequisites <list> | expected <text>]"
                    )
            elif sub.startswith("get "):
                proc_id = sub.replace("get ", "").strip()
                proc = memory.get_procedure(proc_id)
                if proc:
                    print(f"[procedure {proc['id']}] {proc['name']}")
                    print(f"  confidence: {proc['confidence']}")
                    print(
                        f"  prerequisites: {', '.join(proc['prerequisites']) or 'none'}"
                    )
                    for i, step in enumerate(proc["steps"], 1):
                        print(f"  {i}. {step}")
                    if proc.get("expected_result"):
                        print(f"  expected: {proc['expected_result']}")
                    if proc.get("verification_result"):
                        print(f"  verified: {proc['verification_result']}")
                else:
                    print(f"[procedure '{proc_id}' not found]")
            else:
                print("[usage: /procedure save|list|get <id>]")
            continue

        if not command:
            continue

        # Format the human turn exactly as the model expects.
        turn = torch.tensor(
            [
                tokenizer.encode(
                    dialogue.format_turn(
                        dialogue.HUMAN,
                        line,
                    )
                )
            ],
            dtype=torch.long,
            device=c.device,
        )

        # Retrieve corrections first — they influence the very next answer
        # without changing neural weights. Corrections are placed at the
        # front of the context so the model sees them before everything else.
        corrections = memory.recall_corrections(line)
        correction_text = ""
        if corrections:
            correction_lines = []
            for c_item in corrections:
                meta = c_item.get("metadata", {})
                original = meta.get("corrects", "")
                if original:
                    correction_lines.append(
                        f"[correction] Corrected: {original}\n         Answer: {c_item['content']}"
                    )
                else:
                    correction_lines.append(f"[correction] {c_item['content']}")
            correction_text = "\n\n".join(correction_lines)

        # Retrieve persistent agent memory before generation.
        mem_items = memory.recall(line, limit=4)
        memory_context = "\n".join(
            f"[memory {m['kind']}] {m['content']}" for m in mem_items
        )

        # Retrieve relevant past experiences before generation.
        replay_text, replay_records = experience_replay(c, line, max_items=3)

        # Retrieve relevant knowledge before generation.
        prefix, results = recalled_context(
            knowledge,
            line,
            recall_budget,
        )

        report_recall(results)

        # Build the full prefix with corrections at the very front.
        # Corrections are the highest-priority context because they influence
        # the very next answer without changing neural weights.
        # Experiences come next as evidence from past successes/failures.
        context_parts = []
        if correction_text:
            context_parts.append(correction_text)
        if memory_context:
            context_parts.append(memory_context)
        if replay_text:
            context_parts.append(replay_text)
        if prefix:
            context_parts.append(prefix)

        if context_parts:
            prefix = "\n\n".join(context_parts)

        if prefix:
            prefix_tokens = torch.tensor(
                [tokenizer.encode(prefix + "\n")],
                dtype=torch.long,
                device=c.device,
            )
        else:
            prefix_tokens = torch.zeros(
                (1, 0),
                dtype=torch.long,
                device=c.device,
            )

        # Build the complete generation context.
        context = generation_context(
            prefix_tokens,
            history,
            turn,
            cfg.block_size,
        )

        # Generate Jupiter's response.
        text = generate(
            model,
            cfg,
            c,
            tokenizer,
            context,
        )

        # Log the interaction to persistent agent memory.
        interaction_id = memory.log_interaction(
            line, text, metadata={"recalled": len(results)}
        )

        # Preserve the conversation in the active session.
        history = torch.cat(
            (
                history,
                turn,
            ),
            dim=1,
        )[:, -cfg.block_size :]

        jupiter_turn = torch.tensor(
            [
                tokenizer.encode(
                    dialogue.format_turn(
                        dialogue.JUPITER,
                        text,
                    )
                )
            ],
            dtype=torch.long,
            device=c.device,
        )

        history = torch.cat(
            (
                history,
                jupiter_turn,
            ),
            dim=1,
        )[:, -cfg.block_size :]

        # Make terminal output safe on Windows consoles.
        encoding = _console_encoding()

        printable = text.encode(
            encoding,
            errors="replace",
        ).decode(
            encoding,
            errors="replace",
        )

        print(f"jupiter> {printable}")

        # Record the completed interaction AFTER Jupiter answers.
        #
        # This is intentionally outside the generation path. If experience
        # recording fails, Jupiter's response has still been generated and
        # displayed rather than making the conversation itself fail.
        try:
            # Check if there is feedback for this interaction.
            interaction_record = memory.db.execute(
                "SELECT outcome, feedback FROM interactions WHERE id=?",
                (interaction_id,),
            ).fetchone()
            outcome = interaction_record["outcome"] if interaction_record else ""
            feedback_text = interaction_record["feedback"] if interaction_record else ""

            # What was attempted: describe the approach taken this turn.
            attempt_parts = ["generated response from conversation context"]
            if correction_text:
                attempt_parts.append("applied relevant corrections")
            if memory_context:
                attempt_parts.append("recalled agent memory")
            if results:
                attempt_parts.append("retrieved stored knowledge")
            attempt = "; ".join(attempt_parts)

            # What to try next: derive a recommendation from the outcome.
            recommendation = ""
            if outcome in ("wrong", "failure"):
                recommendation = (
                    "review corrections and recalled knowledge before answering"
                )
            elif outcome == "partial":
                recommendation = "verify with additional evidence; clarify with user"
            elif outcome == "uncertain":
                recommendation = "seek confirmation; check for conflicting corrections"
            elif outcome in ("helpful", "success"):
                recommendation = (
                    "continue with current retrieval and generation approach"
                )
            elif not outcome and feedback_text:
                recommendation = (
                    "monitor feedback and adjust retrieval if outcome is negative"
                )

            # What was expected: Jupiter's prediction before generating.
            expected = ""
            if results:
                expected = (
                    "retrieved stored knowledge and applied corrections"
                    if results and correction_text
                    else "retrieved stored knowledge"
                )

            record_experience(
                c,
                line,
                text,
                results,
                attempt=attempt,
                outcome=outcome,
                recommendation=recommendation,
                expected=expected,
                confidence=float(results[0]["confidence"] if results else 0.0),
                concepts=[r["collection"] for r in results if r.get("collection")],
            )

            print("[experience recorded]")

        except Exception as exc:
            print(f"[warning: experience could not be recorded: {exc}]")


if __name__ == "__main__":
    main()
