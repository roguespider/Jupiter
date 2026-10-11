"""Generation quality as a measurement, produced by actually generating.

Architecture 34 lists Language first among its evaluation categories, and the
report has always carried a Language line. That line read "no generated text to
measure" on every run, because the only thing that ever fed it was a `samples`
argument and no caller in the project has ever passed one. evaluate_language
existed, worked, and was unreachable from the entry point.

That is the gap this closes. Language is the one Architecture 34 category that
cannot be measured on text Jupiter already holds: every other category reads the
corpus or the experiment log, but whether the model produces language is a
property of the model, so it has to be asked. Nothing measures it without
generating.

Two properties make the measurement worth reporting rather than a formality.

The prompts come from the corpus. They are drawn from documents Jupiter has
learned, so what is measured is the model being asked to continue material it
actually knows rather than material chosen by whoever wrote this file. A fixed
hand-written prompt set measures one narrow thing; asking about what was learned
measures the thing that was learned.

The prompts are held out. Every prompt is taken from a span that is not trained
on, in the same sense Architecture 20 holds back a document: a generation
quality figure computed from text the model was fitted on measures copying, not
language. The reserved-document regression set is the source, because it is
already excluded from every update for exactly this reason.

What is measured is degeneracy, not quality. Repeated n-grams catch a model
that has collapsed into a loop, which is the failure a loss number hides: a
model can drive loss down and emit the same four tokens forever. A low
repetition rate is a floor and not a ceiling, and saying so in the output is the
difference between a measurement and a compliment.
"""

import re
from pathlib import Path

# Prompts are prefixed so the model is asked to continue rather than to classify.
# The prefix is fixed for every probe: a set that varied its framing would
# measure the framing.
CONTINUE = "Write a short paragraph about {topic}.\n\n"

# How many prompts to draw, and how many tokens to generate for each. Eight
# prompts at forty tokens each is enough to expose a degenerate model and small
# enough that the evaluation does not cost more than the learning cycle it
# reports on.
DEFAULT_PROMPTS = 8
DEFAULT_NEW_TOKENS = 40

# Below this many characters a generation is treated as empty rather than scored,
# because an n-gram rate over two words is not a measurement of anything.
MIN_SAMPLE_CHARS = 12

# A run is not repeated for a trend: generation is sampled, so two runs differ by
# construction and differencing them would report the sampler as a change in the
# model. The report says how many probes ran instead.
SAMPLE_MARKER = re.compile(r"^## SAMPLE \d+ ##$", re.MULTILINE)


class LanguageUnavailable(RuntimeError):
    """The language probe could not run, and the reason is worth reporting."""


def _reserved_sources(c):
    """The documents held back from training, which are the regression yardstick.

    Read through regression.load_reserved rather than by opening the file
    directly, so the language probe and the regression gate cannot disagree
    about which documents are reserved. Two modules reading the same file with
    two different key names is how a probe ends up prompting on text that was in
    fact trained on.

    Returns every reserved document, so prompts are drawn from the same
    representative sample the acceptance gate measures against. Prompting on the
    yardstick alone would weight the probe toward the one stratum of the corpus
    that happens to be held back.
    """
    import regression

    reserved = regression.load_reserved(c)
    if not reserved:
        raise LanguageUnavailable(
            "nothing is reserved yet, so there is no untrained document to "
            "prompt with; run a learning cycle, which reserves some"
        )
    readable = []
    for path in reserved:
        if path.exists():
            readable.append(path)
    if not readable:
        named = ", ".join(str(path) for path in reserved)
        raise LanguageUnavailable(
            f"every reserved document is missing ({named}), so there is no "
            "untrained text to prompt with"
        )
    return readable


def load_prompts(c, count=DEFAULT_PROMPTS):
    """Draw prompts from the reserved documents, spread across the sample.

    Each reserved document contributes a share of the prompts, so the probe is
    not weighted toward whichever stratum happens to be first.

    The span is taken from the middle of each document rather than the start,
    because the first few hundred characters of a curated file are whatever the
    source happened to open with, which is rarely representative of what the
    document is about.
    """
    bodies = []
    for path in _reserved_sources(c):
        try:
            bodies.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as failure:
            raise LanguageUnavailable(f"cannot read {path}: {failure}")

    if not bodies:
        raise LanguageUnavailable("no reserved document could be read")

    total = sum(len(text) for text in bodies)
    if total < MIN_SAMPLE_CHARS * count:
        raise LanguageUnavailable(
            f"the reserved documents total {total} characters, too short to "
            f"draw {count} distinct prompts from"
        )

    per_document = max(1, count // len(bodies))
    prompts = []
    for text in bodies:
        span = max(MIN_SAMPLE_CHARS * per_document, 1)
        for index in range(per_document):
            # Spread across the document, offset by a quarter so the opening
            # lines are never the only thing measured.
            step = max(1, len(text) // per_document)
            start = min(len(text) - span - 1, step * index + span // 4)
            start = max(0, start)
            fragment = text[start : start + 240].strip()
            if len(fragment) >= MIN_SAMPLE_CHARS:
                prompts.append(fragment.replace("\n", " ").replace("\r", " "))

    if not prompts:
        raise LanguageUnavailable(
            "the reserved documents yielded no span long enough to prompt with"
        )
    return prompts[:count]


def continuation(prompt):
    """The prefix a span is asked to continue, so generation is not left to chance."""
    head = " ".join(prompt.split()[:12]).rstrip(",;:.")
    if not head:
        return prompt
    return CONTINUE.format(topic=head)


def generate_samples(
    c,
    model=None,
    tokenizer=None,
    count=DEFAULT_PROMPTS,
    max_new_tokens=DEFAULT_NEW_TOKENS,
):
    """Ask the model to continue each prompt, and return what it produced.

    The model and tokenizer are loaded here rather than passed in from the report
    so that the entry point can measure Language without a caller having to know
    that measuring it requires a checkpoint. Every failure is reported as
    LanguageUnavailable with its reason rather than as an empty result, because
    "the probe did not run" and "the model produced nothing" are different facts
    and only one of them is a measurement.
    """
    import torch

    if model is None or tokenizer is None:
        try:
            from sample import load_jupiter
            from tokenizer import load_tokenizer
        except ImportError as failure:  # pragma: no cover - import guard
            raise LanguageUnavailable(f"cannot load the model: {failure}")
        try:
            tokenizer = load_tokenizer(c)
            model, _cfg = load_jupiter(c)
        except Exception as failure:
            raise LanguageUnavailable(
                f"cannot load a checkpoint to measure generation: "
                f"{type(failure).__name__}: {failure}"
            )

    prompts = load_prompts(c, count=count)
    device = getattr(c, "device", "cpu")
    samples = []
    for prompt in prompts:
        text = continuation(prompt)
        ids = tokenizer.encode(text)
        if not ids:
            continue
        try:
            # The tensor has to be built on the model's own device. The model is
            # moved to the configured device when it loads, so a prompt left on
            # the CPU raises inside embedding with a message about index_select
            # that names neither the cause nor the module it belongs to.
            prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                produced = model.generate(
                    prompt_tensor,
                    int(max_new_tokens),
                    float(getattr(c, "temperature", 0.8)),
                    int(getattr(c, "top_k", 40)),
                )
        except Exception as failure:
            raise LanguageUnavailable(
                f"generation failed on a prompt: {type(failure).__name__}: {failure}"
            )
        if produced is None or produced.ndim != 2 or len(produced[0]) <= len(ids):
            continue
        fresh = [int(token) for token in produced[0][len(ids) :]]
        try:
            samples.append(tokenizer.decode(fresh))
        except (UnicodeDecodeError, ValueError) as failure:
            raise LanguageUnavailable(f"generated text could not be decoded: {failure}")

    if not samples:
        raise LanguageUnavailable(
            "every prompt produced nothing decodable, so there is no generated "
            "text to measure"
        )
    return samples


def save(c, samples, target=None):
    """Write the generated text, so a score can be inspected rather than trusted.

    The scores are a summary and the text is the evidence. A repetition rate with
    no text behind it cannot be argued with, which is the opposite of what a
    measurement is for.
    """
    path = Path(target or getattr(c, "language_samples", "out/language_samples.txt"))
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, text in enumerate(samples):
        body = text.strip()
        blocks.append(f"## SAMPLE {index} ##\n{body}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def load_samples(c, target=None):
    """Previously generated text, or None when none has been kept."""
    path = Path(target or getattr(c, "language_samples", "out/language_samples.txt"))
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    parts = SAMPLE_MARKER.split(text)
    # The split yields a leading empty string before the first marker.
    return [part.strip() for part in parts[1:] if part.strip()] or None


def measured_at(c, target=None):
    """When the kept samples were generated, or None."""
    path = Path(target or getattr(c, "language_samples", "out/language_samples.txt"))
    if not path.exists():
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None
