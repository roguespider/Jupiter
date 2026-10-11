"""Which skill is weakest, and which documents would strengthen it.

Evaluation measures Jupiter in named categories, and the data pipeline says what
it may learn from. Nothing connected the two. Selection already ranked
documents by how badly they were going (frustration band, confidence, novelty),
which is the right instinct and the wrong question: a document nobody can learn
is not the same as a capability Jupiter lacks, and a corpus can be full of the
first while the second goes unaddressed.

So this reads the evaluation report, turns each category into a weakness between
zero and one, and maps that onto the folders data/raw was sorted into. A low
conversation score raises the priority of everything in conversation/, which is
the loop the project wanted: measure, find what is weak, go and learn the
material for it.

Two things it deliberately does not do.

It does not invent a score. Every category carries a note saying what its
number measures, and a number whose meaning is unclear is not evidence of
weakness. A category with nothing measurable contributes no preference, and
that is reported rather than smoothed over.

And it does not decide anything on its own. It returns a priority per folder;
selection still ranks, the acceptance gate still judges, and a document that
turns out to be unlearnable is still rejected. This is a preference, not a
policy, and the human keeps the decisions that matter.
"""

import json
import re
from pathlib import Path

# Which evaluation category speaks to which folder. A folder may answer to more
# than one category, and the strongest of them is the one that counts: a document
# that teaches code is worth reading both when code is weak and when reasoning is
# weak, and ranking it by the worse of the two would overstate how urgently it
# is needed.
CATEGORY_FOLDERS = {
    "conversation": ("conversation",),
    "language": ("conversation", "prose"),
    "programming": ("code",),
    "reasoning": ("reasoning", "math"),
    "knowledge": ("science", "prose"),
    "math": ("math",),
    "science": ("science",),
}

# The folder a category implies when a document is not under a named folder.
# Nothing is inferred from file contents here: the folder layout is the
# declaration, and a document that was left unclassified says nothing about what
# it teaches, which is why organize_raw reports those separately rather than
# guessing for them.
FOLDER_ORDER = (
    "conversation",
    "reasoning",
    "math",
    "code",
    "science",
    "prose",
)

# How strongly a weakness is allowed to move a document up the queue. The cap
# matters: this is a preference expressed alongside a measured need, and a
# category that scored badly should not be able to outrank a document that is
# actively failing to learn. Selection's own signal is about how hard the
# material is; this one is about how much the capability is wanted.
MAX_BONUS = 0.5

# A measured number is read from these keys, in order, and only the first that
# is present and numeric is used. They are the shapes the evaluation report has
# actually used for each category, rather than one invented here.
SCORE_KEYS = ("score", "pass_rate", "mean_score", "value")


def _numeric(value):
    """A score in the range zero to one, or None when there is not one.

    Out-of-range and non-numeric values return None rather than being clamped.
    A number outside the range is a measurement this function does not
    understand, and clamping it would turn a mistake into a confident answer.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value < 0.0 or value > 1.0:  # NaN or out of range
        return None
    return float(value)


def category_scores(report):
    """Weakness per category, zero being strong and one being as weak as it goes.

    The report stores a score that means "how well it did", so it is inverted
    here. Doing that inversion once, in one place, is what keeps every consumer
    from having to remember which direction a number points.
    """
    if not isinstance(report, dict):
        return {}
    scores = {}
    for name in CATEGORY_FOLDERS:
        section = report.get(name)
        if not isinstance(section, dict):
            continue
        for key in SCORE_KEYS:
            score = _numeric(section.get(key))
            if score is not None:
                scores[name] = round(1.0 - score, 4)
                break
    return scores


def folder_weakness(scores):
    """The strongest weakness among the categories that speak to each folder."""
    out = {}
    for folder in FOLDER_ORDER:
        worst = None
        for category, folders in CATEGORY_FOLDERS.items():
            if folder not in folders:
                continue
            value = scores.get(category)
            if value is not None and (worst is None or value > worst):
                worst = value
        if worst is not None:
            out[folder] = worst
    return out


def bonus_for(weakness, weight=MAX_BONUS):
    """Turn a weakness into an addition to a document's priority.

    Linear and capped, because a preference that could outrank a measured need
    would be a policy rather than a preference, and this is meant to be the
    former.
    """
    if weakness is None:
        return 0.0
    return round(max(0.0, min(1.0, weakness)) * weight, 4)


def load_report(path):
    """The evaluation report, or an empty mapping when there is not one yet.

    A missing or unreadable report is not an error: on a first run there is no
    evaluation yet, and the right answer then is no preference rather than a
    guess about which capability is weakest.
    """
    target = Path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def weakness_from_report(c):
    """Per-folder weakness for a configuration, ready to hand to selection."""
    return folder_weakness(category_scores(load_report(c.evaluation_report)))


def describe(weakness):
    """A readable account of what is weak, for the person reading the run."""
    if not weakness:
        return (
            "no category weakness measured yet, so every document is ranked on "
            "how hard it is to learn and none on what it would improve"
        )
    ordered = sorted(weakness.items(), key=lambda item: (-item[1], item[0]))
    parts = ", ".join(f"{name} {value:.2f}" for name, value in ordered)
    return f"weakest first: {parts}"


def folder_of(path):
    """Which skill folder a source path sits in, or None when it is not filed.

    Read from the path rather than from the file's contents, because the folder
    is a statement the owner made by where they put the document, and a pipeline
    that second-guessed it from the text would be making a semantic decision
    curation says it has no business making.
    """
    parts = Path(path).parts
    for name in FOLDER_ORDER:
        if name in parts:
            return name
    return None


def priorities(c, weight=MAX_BONUS):
    """Priority addition per skill folder, ready for selection to add in."""
    return {
        folder: bonus_for(value, weight)
        for folder, value in weakness_from_report(c).items()
    }


def main(c=None, argv=None):
    """Print what is weak, so the report a run makes says what it is aiming at."""
    from config import load_config

    c = c or load_config()
    scores = category_scores(load_report(c.evaluation_report))
    weakness = folder_weakness(scores)
    print("category scores (0 strong, 1 weak)")
    for name, value in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name:<14} {value:.4f}")
    print()
    print(describe(weakness))
    print()
    print("document priority added")
    for folder, bonus in sorted(
        priorities(c).items(), key=lambda item: (-item[1], item[0])
    ):
        print(f"  {folder:<14} +{bonus:.4f}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
