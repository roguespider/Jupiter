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
    "/remember <kind> <text>  save a durable memory\n"
    "/correct <old> => <new>  record a correction\n"
    "/feedback <helpful|wrong|partial|uncertain> [details]  rate last interaction\n"
    "/task add <title>  create a task\n"
    "/task list  list open tasks\n"
    "/memory   report memory counts\n"
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


def experience_replay(c, question, max_items=3):
    """Retrieve relevant past experiences for the current question."""
    records, unreadable = experience_loop.read(c, limit=max_items * 2)
    if not records:
        return "", []
    # Simple lexical overlap for relevance: match keywords in question.
    question_words = set(str(question).lower().split())
    relevant = []
    for record in records:
        user_text = str(record.get("user", ""))
        jupiter_text = str(record.get("jupiter", ""))
        overlap = len(question_words & set(user_text.lower().split()))
        if overlap > 0 or len(user_text) < 200:
            relevant.append(record)
    # Take the most recent relevant experiences.
    relevant = relevant[-max_items:]
    replay_text = "\n".join(
        f"[past experience] user: {str(r.get('user', '')).strip()[:80]} | jupiter: {str(r.get('jupiter', '')).strip()[:80]}"
        for r in relevant
    )
    return replay_text, relevant


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
):
    """Record one completed interaction as a durable experience.

    Experiences are observations, not automatic weight updates.

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

        if memory_context and prefix:
            prefix = memory_context + "\n" + prefix
        elif memory_context:
            prefix = memory_context

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
            record_experience(
                c,
                line,
                text,
                results,
            )

            print("[experience recorded]")

        except Exception as exc:
            print(f"[warning: experience could not be recorded: {exc}]")


if __name__ == "__main__":
    main()
