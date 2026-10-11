"""A fixed yardstick per skill, reserved and never trained on.

Why this exists, in numbers rather than adjectives.

Over three thousand recorded cycles the per-cycle change in holdout loss averages
minus six per cent with a standard deviation of twenty-nine. The improvement
being measured is a fifth of the noise. Everything downstream inherits that:

* roughly forty-five per cent of updates are rejected, not because they were bad
  but because one document's holdout loss cannot separate learning from which
  document happened to be drawn
* a capacity growth is judged on a single number, so whether it helped is decided
  by a coin weighted by document difficulty
* "frustration" partly measures which document came up rather than what the model
  can do

conversation_eval.py already states the principle and proves it:

    "A model can drive the loss on a transcript down and still answer every
    question with a second question, and the acceptance gate would call that an
    improvement."

That reasoning is correct, and it is applied to exactly one category. Holdout
loss on the document that caused the cycle is still the yardstick for
Programming, Reasoning, Learning and Stability. This module generalises what
conversation_eval does: a fixed set of probes per skill, drawn from the corpus,
never trained on, scored by whether a correct answer's terms appear.

What it buys, concretely. The variance of a mean falls with the square root of
the number of probes, so a score taken over twenty probes is around a quarter of
the spread of one number. More importantly, document difficulty stops being the
verdict: the same twenty probes are asked after every cycle regardless of what
was learned, so two cycles are comparable and a growth is a measurement rather
than a guess.

What it is not. This is a retrieval-shaped benchmark. It asks whether what was
learned can be recalled in the form it was learned in. That is not the same as
answering an arbitrary question, and the report says so in the note it carries
rather than letting a percentage imply a capability it does not demonstrate.

The honesty rule, enforced rather than intended: a skill whose probe set cannot
be built reports as unavailable and contributes no signal. An empty set scoring
as a perfect score would make an unmeasurable skill look like the strongest one,
which is the specific dishonesty this project exists to avoid.
"""

import json
import random
import re
from pathlib import Path

# How many probes per skill. Twenty is enough for the spread to fall to roughly
# a fifth of a single measurement, and small enough that asking the model twenty
# questions stays affordable at the end of every cycle rather than becoming a
# cost that gets skipped.
DEFAULT_PROBES = 20

# The token a left-padded prompt is filled with while a batch of probes is
# generated together. It has to be a real vocabulary index or the embedding
# lookup refuses the row, and zero is always valid.
PAD_ID = 0

# A probe needs a question and an answer that are both substantial enough to be
# worth asking. Real prose sentences are short - a two-sentence span is the
# usual unit, and one sentence on its own rarely carries enough to make an answer
# with distinctive terms in it. This started at 400 characters per sentence and
# rejected every sentence in a real curated document, which is the failure a
# threshold invented without looking at the material looks like.
MIN_SPAN_CHARS = 60

# The probe set is fixed once built, because a benchmark whose questions change
# between runs cannot say whether the model improved. The seed is recorded with
# the set so a rebuild can reproduce the same one.
SEED = 20261003


def _sentences(text):
    """Split into sentences, keeping only those long enough to be a span."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SPAN_CHARS]


def _distinctive(words, limit=6):
    """The rarest-looking terms in a span, as the terms an answer must contain.

    Not a frequency ranking against the whole corpus, which would mean reading
    everything to build twenty questions. Instead the longest and rarest-looking
    tokens, which is enough to make a check that passes on a coincidence very
    unlikely: a reply has to use most of them.
    """
    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "had",
        "was",
        "were",
        "are",
        "but",
        "not",
        "you",
        "your",
        "they",
        "them",
        "their",
        "there",
        "here",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "why",
        "all",
        "any",
        "can",
        "will",
        "would",
        "should",
        "could",
        "into",
        "than",
        "then",
        "these",
        "those",
        "such",
        "some",
        "more",
        "most",
        "other",
        "been",
        "also",
        "about",
        "because",
        "while",
        "where",
        "after",
        "before",
        "over",
        "only",
        "very",
        "each",
        "both",
        "many",
        "much",
        "does",
        "did",
        "doing",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", " ".join(words))
    keep = [t for t in tokens if t.lower() not in stop]
    # Longest first, which favours the specific over the incidental, and stable
    # ordering so the same span always yields the same required terms.
    seen, ordered = set(), []
    for token in sorted(keep, key=lambda t: (-len(t), t)):
        low = token.lower()
        if low in seen:
            continue
        seen.add(low)
        ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


def skill_of(path):
    """Which skill a curated document teaches, from the folder it was filed in.

    Read from the path rather than from the content. The folder is a statement
    the owner made by where they put the document, and a pipeline that inferred a
    skill from the text would be making a judgement about material that only
    someone who knows its purpose can make.
    """
    for part in Path(path).parts:
        if part in SKILLS:
            return part
    return None


# The skills a probe set can be built for. Derived from the folders
# organize_raw creates rather than invented here, so a skill with no material is
# a skill that reports unavailable rather than a skill that scores zero.
SKILLS = ("conversation", "math", "code", "reasoning", "science", "prose")

# The prompt layout probes are asked in. The same one curation renders
# conversations into, because training and asking have to agree or a failure here
# is a failure of the harness rather than of the model.
PROMPT = "You: {question}\nJupiter:"


def state_path(c):
    """Where the built probe sets are kept."""
    return Path(getattr(c, "probe_state", "out/probes.json"))


def attribution_path(c):
    """Which curated document came from which skill, built from the last run.

    A curated document is named by the digest of its own content, so the folder
    it came from in data/raw is not recoverable from its path. Curation records
    it on every document it writes, and this reads those records back.

    Without it there is no per-skill measurement at all: every curated file looks
    unattributed, and a benchmark that cannot say which capability it is scoring
    is not one. Rebuilding from the report rather than guessing from the text is
    the whole reason the report records a skill.
    """
    report = Path(getattr(c, "curation_report", "data/curation_report.json"))
    if not report.exists():
        return {}
    try:
        documents = json.loads(report.read_text(encoding="utf-8")).get("documents", [])
    except (OSError, ValueError):
        return {}
    mapping = {}
    for entry in documents:
        if not isinstance(entry, dict):
            continue
        digest = entry.get("sha256")
        skill = entry.get("skill")
        if digest and skill:
            mapping[digest] = skill
    return mapping


def build_probes(c, per_skill=DEFAULT_PROBES, seed=SEED, attribution=None):
    """Build one probe set per skill from the curated corpus.

    Each probe is a question taken from one sentence of a document, and the terms
    an answer has to contain are the distinctive words of the sentence that
    follows it. A model that learned the material can reproduce it; one that did
    not cannot. This is a claim about recall in the form it was learned, which is
    what the note in the report says, and it is weaker than arbitrary question
    answering on purpose: a benchmark that can report 90 out of 100 forever no
    matter how good the model gets is not measuring anything.
    """
    curated = Path(getattr(c, "curated_dir", "data/curated"))
    if not curated.exists():
        return {}

    if attribution is None:
        attribution = attribution_path(c)

    by_skill = {}
    for path in curated.glob("*.txt"):
        skill = attribution.get(path.stem) or skill_of(path)
        if skill is None or skill not in SKILLS:
            continue
        by_skill.setdefault(skill, []).append(path)

    rng = random.Random(seed)
    sets = {}
    for skill, paths in sorted(by_skill.items()):
        rng.shuffle(paths)
        probes_for_skill = []
        for path in paths:
            if len(probes_for_skill) >= per_skill:
                break
            try:
                # A bounded read: the corpus holds documents of sixty-odd
                # megabytes and a probe set is twenty questions.
                text = path.read_text(encoding="utf-8", errors="replace")[:20000]
            except OSError:
                continue
            # Consecutive pairs, so the answer is the text that actually follows
            # the question in the material. A question answered by an unrelated
            # sentence elsewhere would measure the matcher rather than recall.
            sentences = _sentences(text)
            for index in range(len(sentences) - 1):
                if len(probes_for_skill) >= per_skill:
                    break
                question, answer = sentences[index], sentences[index + 1]
                expected = _distinctive(answer.split())
                if len(expected) < 3:
                    # Too few distinctive terms and the check passes on a
                    # coincidence, which flatters the model rather than
                    # measuring it.
                    continue
                probes_for_skill.append(
                    {
                        "question": question[:300],
                        "expected": expected,
                        "source": str(path),
                    }
                )
        if probes_for_skill:
            sets[skill] = probes_for_skill
    return sets


def load(c):
    """The probe sets, building them once and keeping them fixed afterwards.

    A benchmark whose questions change between runs cannot say whether the model
    improved, so the set is written to disk the first time and read from there
    every time after. Deleting the file rebuilds it deliberately, which is a
    decision rather than something that happens.
    """
    path = state_path(c)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sets = data.get("probes")
            if isinstance(sets, dict):
                return sets
        except (OSError, ValueError):
            pass
    sets = build_probes(c)
    if sets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"probes": sets, "seed": SEED, "per_skill": DEFAULT_PROBES},
                indent=2,
            ),
            encoding="utf-8",
        )
    return sets


def answer_score(reply, expected):
    """What fraction of the required terms a reply actually used."""
    from conversation_eval import _stem

    found = {_stem(token) for token in re.findall(r"[A-Za-z]+", str(reply).lower())}
    if not expected:
        return 0.0
    hits = sum(1 for term in expected if _stem(term) in found)
    return round(hits / len(expected), 4)


def ask_batch(model, tokenizer, probes, max_new_tokens=40, device=None):
    """Ask several probes in one generation pass, and score each reply.

    One generate() call per probe meant one forward pass per token per probe: a
    twenty-probe set cost eight hundred sequential passes on the CPU, which is
    about eight seconds of a learning cycle and two thirds of its whole cost. The
    probes are independent questions, so they are stacked into one batch and
    continued together. The number of passes falls from twenty per token to one
    per token, and the answer for each probe is the same sequence it would have
    produced on its own.

    Padding is on the left so every row's continuation starts at the same
    position, which is what lets the last position of the forward pass be read as
    the next token for every row at once. A prompt that would not encode, or a
    generate() that raised, yields None for that probe and leaves the others
    alone: a harness that cannot ask and a model that cannot answer have to stay
    distinguishable, or a broken measurement looks like a weak model.
    """
    import torch

    if not probes:
        return []

    encoded = []
    for probe in probes:
        prompt_ids = tokenizer.encode(PROMPT.format(question=probe["question"]))
        encoded.append(prompt_ids if prompt_ids else None)

    live = [index for index, ids in enumerate(encoded) if ids]
    results = [None] * len(probes)
    if not live:
        return results

    width = max(len(encoded[index]) for index in live)
    rows = []
    for index in live:
        ids = encoded[index]
        # Left padding, so the final column is a real token for every row.
        rows.append([PAD_ID] * (width - len(ids)) + ids)

    try:
        generated = model.generate(
            torch.tensor(rows, dtype=torch.long),
            max_new_tokens=int(max_new_tokens),
        )
    except Exception:
        return results

    if generated is None or getattr(generated, "ndim", 0) != 2:
        return results

    for row, index in enumerate(live):
        sequence = generated[row]
        if len(sequence) <= width:
            continue
        try:
            reply = tokenizer.decode([int(token) for token in sequence[width:]])
        except (UnicodeDecodeError, ValueError):
            continue
        # A model that asks the question back has not answered it, and the
        # question's own words are in the prompt.
        reply = reply.replace("Jupiter:", "").strip()
        probe = probes[index]
        score = answer_score(reply, probe["expected"])
        results[index] = {
            "question": probe["question"][:160],
            "score": score,
            "passed": score >= 0.6,
            "reply": reply[:200],
        }
    return results


def ask(model, tokenizer, probe, max_new_tokens=40, device=None):
    """Ask one probe and score the reply, or None when it could not be asked."""
    import torch

    prompt = PROMPT.format(question=probe["question"])
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        return None
    try:
        generated = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=int(max_new_tokens),
        )
    except Exception:
        # Reported as a missing score rather than a zero. A harness that cannot
        # ask and a model that cannot answer have to stay distinguishable, or a
        # broken measurement looks like a weak model.
        return None
    if generated is None or generated.ndim != 2 or len(generated[0]) <= len(prompt_ids):
        return None
    reply_ids = [int(token) for token in generated[0][len(prompt_ids) :]]
    try:
        reply = tokenizer.decode(reply_ids)
    except (UnicodeDecodeError, ValueError):
        return None
    # A model that asks the question back has not answered it, and the question's
    # own words are in the prompt.
    reply = reply.replace("Jupiter:", "").strip()
    return {
        "question": probe["question"][:160],
        "score": answer_score(reply, probe["expected"]),
        "passed": answer_score(reply, probe["expected"]) >= 0.6,
        "reply": reply[:200],
    }


def evaluate(c, model, tokenizer, device=None):
    """Score every skill's probe set. Returns a report, never a bare number.

    A skill with no probes reports as unavailable and contributes nothing. It does
    not score zero, because zero is a measurement and unavailable is not, and a
    set that could not be built scoring as the worst possible result would make
    missing material look like weak material.
    """
    sets = load(c)
    results = {}
    for skill, probes in sorted(sets.items()):
        # Asked as one batch per skill. Each probe was previously a generate() of
        # its own, so a twenty-probe set cost eight hundred sequential forward
        # passes; batched, it costs forty. Nothing about what is being asked
        # changes, only how many times the model is walked to answer it.
        scored = []
        for outcome in ask_batch(model, tokenizer, probes):
            if outcome is not None:
                scored.append(outcome)
        if not scored:
            results[skill] = {
                "measured": 0,
                "note": f"no probe could be asked for {skill}",
            }
            continue
        mean = round(sum(r["score"] for r in scored) / len(scored), 4)
        passed = sum(1 for r in scored if r["passed"])
        results[skill] = {
            "measured": len(scored),
            "mean_score": mean,
            "passed": passed,
            "pass_rate": round(passed / len(scored), 4),
            "note": (
                f"{passed} of {len(scored)} probes recalled the terms of the "
                "answer that followed the question in the material it was drawn "
                "from. This measures recall in the form it was learned, not the "
                "ability to answer a question nobody asked."
            ),
            "results": scored,
        }
    return results


def headline(results):
    """One sentence for a person reading a run, or the reason there is none."""
    measured = {k: v for k, v in results.items() if v.get("measured")}
    if not measured:
        return "no skill has a probe set yet, so capability is unmeasured"
    ordered = sorted(measured.items(), key=lambda item: item[1]["mean_score"])
    weakest = ordered[0]
    return (
        f"weakest skill: {weakest[0]} at {weakest[1]['mean_score']:.2f} "
        f"over {weakest[1]['measured']} probes; "
        f"{len(measured)} skill(s) measured"
    )


def main():
    import sys

    from config import load_config

    c = load_config()
    sets = load(c)
    report = {
        "state": str(state_path(c)),
        "skills": {k: len(v) for k, v in sorted(sets.items())},
    }
    print(json.dumps(report, indent=2))
    if not sets:
        print("no probe set could be built; capability is unmeasured")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
