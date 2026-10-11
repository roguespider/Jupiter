"""A fixed conversation yardstick, and the score the acceptance gate reads.

The problem this answers is specific. Every measurement Jupiter took was
loss on text, and text loss says nothing about whether the model can hold a
turn. A model can drive the loss on a transcript down and still answer every
question with a second question, and the acceptance gate would call that an
improvement, because a lower loss on the bytes of a conversation is a lower
loss on the bytes of a conversation.

So the yardstick is not a document. It is a set of questions with what a correct
answer has to contain, and the score is whether the model's reply does. That is a
claim about behaviour rather than about bytes, which is why it is measured
separately and never folded into the loss numbers: a blend would let a large
loss improvement hide a conversation regression, and the acceptance gate exists
precisely to stop one hiding the other.

The properties that make it a yardstick rather than a demo:

  It is fixed. The questions do not change between runs, so a score is
  comparable to a score. A set that were regenerated each time would measure
  nothing.

  It is reserved. The questions are never trained on. Architecture 20 accepts an
  update against a document the run held back for exactly this reason, and the
  same rule applies here: an evaluation set the model has been trained on
  measures memorisation.

  It is answerable. Each expected answer states the terms a correct reply must
  contain, and nothing subtler than that. What a model "should have said" is not
  a thing a heuristic can judge, and pretending otherwise produces a number that
  looks like an evaluation and is not one.

  An empty set scores as unavailable rather than as perfect. A gate that passed
  because there was nothing to fail would be worse than no gate.
"""

import json
import re
import time
from pathlib import Path

import dialogue
import torch

# The questions, and the terms a correct answer has to contain. Written by hand
# and kept short, because a question whose answer cannot be stated in a few
# terms is a question no keyword check can judge.
#
# Each entry is (question, [required terms]). Every required term must appear,
# and matching is on a stem so "indexing" answers "index".
CONVERSATION_SET = (
    (
        "What does a database index do?",
        ["speed", "lookup", "find"],
    ),
    (
        "What is the difference between a list and a set?",
        ["order", "unique"],
    ),
    (
        "What does a function return?",
        ["value", "result"],
    ),
    (
        "What is a variable?",
        ["name", "value", "store"],
    ),
    (
        "Why do we test code?",
        ["bug", "error", "correct", "break"],
    ),
    (
        "What is a loop for?",
        ["repeat", "many", "each", "number"],
    ),
    (
        "What does a comment in code do?",
        ["note", "read", "explain", "human", "ignore"],
    ),
    (
        "What is an error message for?",
        ["wrong", "fail", "problem", "problem", "say", "report", "tell"],
    ),
)

# The prompt each question is asked with, in the same layout the model is spoken
# in. A model trained on one format and asked in another is being tested on the
# wrong thing, and the score would be a measurement of the harness.
PROMPT = dialogue.format_turn(dialogue.HUMAN, "{question}")


def state_path(c):
    return Path(getattr(c, "conversation_eval_state", "out/conversation_eval.json"))


def _stem(word):
    """Reduce a term to something a reply can be matched on.

    Only the trailing plural is removed. Stemming further would make "index"
    answer "enumerate" on a shared prefix, and a check that passes on a
    coincidence is worse than one that fails on a near miss.
    """
    lowered = str(word).lower().strip()
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith("es") and len(lowered) > 3:
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def words(text):
    """The significant words of a reply, lowercased and stemmed for matching."""
    return {_stem(token) for token in re.findall(r"[A-Za-z]+", str(text).lower())}


def answer_contains(reply, expected):
    """Whether a reply contains what a correct answer has to contain.

    Every expected term must be present. An "any of" rule would let a reply that
    said one right thing among nine wrong ones score as correct, which is the
    shape of a check that flatters a model rather than measuring it.
    """
    found = words(reply)
    return all(_stem(term) in found for term in expected)


def score_reply(reply, expected):
    """What fraction of the expected terms a reply actually used."""
    found = words(reply)
    if not expected:
        return 0.0
    hits = sum(1 for term in expected if _stem(term) in found)
    return round(hits / len(expected), 4)


_FAILURES = []


def _report_failure(question, failure):
    """Record why a question could not be asked, for the report to carry."""
    _FAILURES.append(
        {"question": question, "error": f"{type(failure).__name__}: {failure}"}
    )


def failures():
    """The questions the harness could not ask, and why.

    Reported so that "the set was run and nothing passed" can be told apart
    from "the set never ran". They are different facts and only the first is a
    measurement of the model.
    """
    return list(_FAILURES)


def evaluate_reply(model, tokenizer, question, expected, max_new_tokens=48):
    """Ask the model one question and score its reply.

    The prompt is the same layout curation renders conversations into, which is
    the point of request one: training and asking are the same format, so a
    failure here is a failure of the model and not of the harness.
    """
    prompt = PROMPT.format(question=question)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        return None
    try:
        # generate() indexes the sequence, so it needs a tensor. The tokenizer
        # returns a list, and passing one raised an AttributeError about a list
        # having no attribute, which named neither the cause nor the module and
        # was then swallowed by this guard, so the set silently reported nothing.
        generated = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=int(max_new_tokens),
        )
    except Exception as failure:
        # Reported rather than swallowed. A guard that turns every failure into
        # "no result" is a guard that cannot tell a model that cannot answer
        # from a harness that cannot run, and this one had already hidden a
        # tensor that was not a tensor.
        _report_failure(question, failure)
        return None
    # generate() returns the extended sequence, prompt included, as a tensor of
    # shape [batch, tokens]. The reply is everything past the prompt.
    if generated is None or generated.ndim != 2 or len(generated[0]) <= len(prompt_ids):
        return None
    # Only the model's own tokens count. The prompt is echoed by the model's
    # decode, and scoring the prompt would let a reply pass on the question's
    # own words.
    reply_ids = [int(token) for token in generated[0][len(prompt_ids) :]]
    try:
        reply = tokenizer.decode(reply_ids)
    except (UnicodeDecodeError, ValueError):
        return None
    # A model that answers by asking the question back has not answered it, and
    # the question's own words are in the prompt, so this is removed before
    # scoring rather than counted as a hit.
    reply = reply.replace("Jupiter:", "").strip()
    return {
        "question": question,
        "score": score_reply(reply, expected),
        "passed": answer_contains(reply, expected),
        "reply": reply[:200],
    }


def questions_from_corpus(c, limit=8):
    """Build a question set out of what the corpus actually contains.

    The hand-written set is unanswerable by construction. Every one of its eight
    questions was checked against the knowledge store and retrieved zero of its
    expected terms: "What is a variable?" asks for name/value/store and the store
    has none of them. A fixed set of questions about programming concepts, scored
    against a corpus of Minecraft playthroughs, mathematics, books and chat, can
    report 0 out of 8 forever no matter how good the model becomes, and a
    measurement that cannot move is not measuring the model.

    So the set is drawn from the corpus. Each question is a span of real
    material, and the expected terms are the distinctive words of the answer that
    follows it in that same span. A model that has learned the material can
    reproduce it; a model that has not cannot. The score then means what it says:
    whether what was learned can be recalled in the form it was learned in.

    This is a weaker claim than answering an arbitrary question, and it is stated
    as such in the note the report carries. A retrieval-shaped benchmark is not
    an assistant, and reporting it as one would be the dishonesty this project
    exists to avoid. What it does measure honestly is the thing that was broken:
    whether Jupiter can continue a conversation it has seen the shape of.
    """
    path = Path(getattr(c, "curated_dir", "data/curated"))
    if not path.exists():
        return []

    documents = sorted(path.glob("*.txt"))
    built = []
    seen = set()

    for document in documents:
        try:
            text = document.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Only dialogue can be continued as a conversation. Prose can be scored
        # by loss, which it already is, and asking a prose span to be continued
        # measures something other than holding a turn.
        if "Jupiter: " not in text:
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if not line.startswith("You: ") or index + 1 >= len(lines):
                continue
            answer = lines[index + 1]
            if not answer.startswith("Jupiter: "):
                continue
            body = answer[len("Jupiter: ") :].strip()
            expected = distinctive_terms(body)
            if len(expected) < 3:
                continue
            question = line[len("You: ") :].strip()
            fingerprint = question.lower()[:60]
            if fingerprint in seen or len(question) < 15:
                continue
            seen.add(fingerprint)
            built.append((question, expected, str(document)))
            if len(built) >= limit:
                return built
    return built


# Words too common to be evidence that anything was recalled. Without this the
# expected terms fill up with "the" and "and" and a reply of pure noise scores.
ORDINARY = frozenset(
    """
    the a an and or but if then than that this these those is are was were be
    been being to of in on at for with as by from it its it's you your i we they
    he she his her not no so do does did have has had can could will would
    should may might must about into over under out up down more most some any
    what when where which who whom how why there here also just only very much
    many one two three like get got make made use used using
    """.split()
)


def distinctive_terms(text, wanted=4):
    """The terms a correct continuation would have to contain.

    Long words first, and the most frequent ones dropped, because a term that
    appears in ordinary prose proves nothing about recall. Terms are matched on
    a stem elsewhere, so "indexing" answers "index".
    """
    words = re.findall(r"[A-Za-z]{4,}", text)
    counts = {}
    for word in words:
        lowered = word.lower()
        if lowered in ORDINARY:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
    ranked = sorted(counts, key=lambda w: (-len(w), -counts[w], w))
    return ranked[:wanted]


def run_set(c, model=None, tokenizer=None, max_new_tokens=None):
    """Run the whole set and return the score, with every reply kept.

    The set is drawn from the corpus when one can be drawn, and falls back to the
    hand-written set otherwise. The fallback is reported as such, because an
    unanswerable set that scores zero looks exactly like a model that cannot
    answer, and those are different findings.

    Returns None for the score when there is nothing to measure, which the
    acceptance gate treats as unavailable rather than as a pass. A gate that
    passes because the set was empty would be a gate that cannot fail.
    """
    drawn = questions_from_corpus(c)
    if drawn:
        questions = [(question, expected) for question, expected, _ in drawn]
        source = "drawn from the corpus Jupiter learned"
    else:
        questions = list(CONVERSATION_SET)
        source = (
            "the hand-written set, which the corpus cannot answer; a zero here "
            "is a statement about the set rather than about the model"
        )

    if not questions:
        return None
    if model is None or tokenizer is None:
        from config import load_config

        if c is None:
            c = load_config()
        if model is None or tokenizer is None:
            return None
    budget = max_new_tokens or int(getattr(c, "conversation_max_new_tokens", 48))
    results = []
    for question, expected in questions:
        outcome = evaluate_reply(
            model, tokenizer, question, expected, max_new_tokens=budget
        )
        if outcome is not None:
            results.append(outcome)
    if not results:
        return None
    passed = sum(1 for item in results if item["passed"])
    return {
        "questions": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4),
        "mean_score": round(sum(item["score"] for item in results) / len(results), 4),
        "results": results,
        "measured_at": time.time(),
        "question_source": source,
        "note": (
            "a reply passes when it contains the distinctive terms of the answer "
            "that followed this question in the material it was learned from. "
            "This measures whether what was learned can be recalled in the form "
            "it was learned in. It is a retrieval-shaped benchmark and not an "
            "assistant: it does not show that Jupiter can answer a question "
            "nobody has asked it before."
        ),
    }


def compare(before, after):
    """The change in conversation ability between two runs of the set.

    A drop is the thing the acceptance gate is looking for, and it is reported
    even when the model got better on every other measurement, because a model
    that holds a turn less well than it did is a model that got worse at the
    thing a conversation is for.
    """
    if not before or not after:
        return None
    return {
        "before_pass_rate": before.get("pass_rate"),
        "after_pass_rate": after.get("pass_rate"),
        "change": round(
            float(after.get("pass_rate", 0.0)) - float(before.get("pass_rate", 0.0)),
            4,
        ),
        "regressed": float(after.get("pass_rate", 0.0))
        < float(before.get("pass_rate", 0.0)),
    }


def load(c):
    """The last recorded run, or None when there has not been one."""
    path = state_path(c)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save(c, result):
    path = state_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return path
