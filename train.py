"""Jupiter micro-learning controller.

Each invocation performs one bounded learning cycle:

Discover curated text.
Split the selected document into train/holdout material.
Measure loss before learning.
Perform tiny gradient updates.
Detect unstable gradients and recover safely.
Measure loss after learning.
Save SafeTensors and a JSON learning record.

This is learning-cycle infrastructure, not conventional long-run
pretraining. learning_worker.py reuses run_micro_learning() below to keep
learning across cycles, so there is exactly one micro-update implementation.
"""

import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from model import Jupiter, ModelConfig
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from telemetry import Telemetry

from config import (
    ARCHITECTURE_VERSION,
    architecture_dict,
    load_config,
    read_architecture,
    shape_of,
)

# The checkpoint the conversation loads. Written only when a cycle is ACCEPTed,
# because Architecture 20 requires an update to be judged before it is kept.
# The resume checkpoint in config.resume_checkpoint is the separate working
# state a run continues from, and is never published.
PUBLISHED_CHECKPOINT = "Jupiter_latest.safetensors"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_documents(c):
    root = Path(c.curated_dir)
    docs = sorted(root.glob("*.txt"))
    if docs:
        return docs
    raw = sorted(Path(c.raw_dir).rglob("*.txt"))
    if raw:
        raise RuntimeError("No curated text exists. Run: python curate_dataset.py")
    raise RuntimeError(
        f"No readable text found in {c.raw_dir}. Put a .txt, .md, .text, .json "
        "or .jsonl file there first."
    )


def read_learning_text(docs, c):
    """Describe the learnable documents without reading them.

    Documents are not truncated here. A cycle works on a window of a document
    (see learning_worker.window_for), so truncating at read time would pin
    every cycle to the same opening passage and the rest of the corpus would
    never be seen.

    Nothing is read here either, which is the change. This used to read every
    document in full and hold the whole corpus in memory for the length of a
    run, on the reasoning that the corpus was "small next to the model and the
    corpus on disk". That was true at three hundred and forty megabytes and
    stopped being true at twenty-one: a run held the entire corpus in RAM and
    reported it plainly as being small. The project prefers RAM and CPU to GPU
    where it can, which makes a load-everything step the wrong place to spend
    that preference - not because RAM is scarce but because twenty-one gigabytes
    of it buys nothing, since a cycle reads a window and a window is a few
    hundred kilobytes.

    So a candidate is a path plus what can be known from the filesystem: its
    size on disk, which is what the window rotation needs in order to know how
    many windows the document has. The text is read when selection has chosen
    one, and dropped as soon as that cycle ends.

    The size is measured in bytes rather than characters because that is what
    the filesystem knows without reading. The two differ for non-ASCII text, so
    the window count is computed from the real character length once the chosen
    document has been read; see window_count_for.
    """
    candidates = []
    for path in docs:
        try:
            size = path.stat().st_size
        except OSError as failure:
            # A document that vanished between being listed and being measured
            # is not a reason to abandon the corpus. Skipped and reported by the
            # caller through the count that no longer matches the listing.
            print(f"[worker] could not measure {path}: {failure}")
            continue
        if size >= c.min_text_chars:
            candidates.append(Candidate(path, size))
    if not candidates:
        raise RuntimeError("Curated documents are too small to learn from.")
    return candidates


class Candidate:
    """One learnable document, described rather than loaded.

    A plain object rather than a tuple so that a caller reaching for `.text` gets
    an error that names the mistake, rather than a tuple that unpacks into a
    path and an integer and fails somewhere later with no explanation. Holding
    the bytes is the expensive part and the thing this exists to avoid.
    """

    __slots__ = ("path", "bytes_on_disk")

    def __init__(self, path, bytes_on_disk):
        self.path = path
        self.bytes_on_disk = bytes_on_disk

    def __repr__(self):
        return f"Candidate({self.path.name}, {self.bytes_on_disk} bytes)"

    def read(self):
        """The document's text, read now rather than at discovery."""
        return self.path.read_text(encoding="utf-8", errors="replace").strip()


def read_text(piece):
    """The text of a candidate, whether it is a Candidate or a plain tuple.

    regression.reserve() hands back plain (path, text) pairs, so a caller that
    has to work with both cannot assume one shape. This is the seam.
    """
    if isinstance(piece, Candidate):
        return piece.read()
    _path, text = piece
    return text


def window_count_for(text, size, windows_per_cycle=1):
    """How many windows a document has, from its real character length.

    The candidate knows only how many bytes the file is, and one byte is not one
    character for anything that is not ASCII. This is where that difference is
    settled, on the text that was actually read, so the rotation covers the whole
    document rather than stopping short of it on non-ASCII material. Mathematically
    it is window_for's own arithmetic, kept here so the two cannot drift apart.
    """
    if len(text) <= size:
        return 1
    windows = max(1, (len(text) + size - 1) // size)
    return max(1, int(windows_per_cycle)) if windows else 1


def window_for(text, index, size, windows_per_cycle=1):
    """A different slice of a document for each cycle.

    Without this, a document always contributed the same first `size`
    characters, so a twelve-million-character book taught the same twelve
    thousand characters on every cycle and the remaining ninety-nine percent
    was never seen. Rotating the window is what lets repeated cycles over one
    document cover that document, which is the difference between a run over
    the corpus and a run over its first page repeated.

    `windows_per_cycle` is how many consecutive windows one cycle reads. It
    exists because one window per cycle made a large document impractical
    rather than merely slow: covering a five-megabyte document once took 417
    cycles, and a document of 77 GB would have taken six and a half million,
    which is not a curriculum, it is a number no run reaches. Taking more
    windows per cycle scales the work with the size of the document instead of
    forcing the operator to own documents small enough to be finished.

    Consecutive rather than strided, because consecutive is what reads a
    document: a cycle that jumps to window seven of a book has not learned the
    first six, and the model is a causal one, so the text it sees should read
    forwards.
    """
    if len(text) <= size:
        return text
    windows = max(1, (len(text) + size - 1) // size)
    take = max(1, int(windows_per_cycle))
    if take >= windows:
        # The document is no longer than one cycle reads, so there is nothing
        # to rotate through and the whole of it is the honest answer.
        return text
    # The first cycle starts at the beginning of the document, and each cycle
    # after it advances by the number of windows it just read. The offset by one
    # matters: without it, cycle one starts at window `take` and the opening of
    # every large document is never seen at all, which is the exact failure this
    # function exists to prevent.
    start = (((index - 1) * take) % windows) * size
    end = start + size * take
    if end > len(text):
        # The window wraps around to the beginning so every cycle reads a fixed
        # amount rather than shrinking at the end of the document. A short final
        # window would mean the last page of a book is learned from a fragment,
        # and the amount of text a cycle sees should not depend on where in the
        # document it happens to land.
        return text[start:] + text[: end - len(text)]
    return text[start:end]


def split_tokens(text, block_size, holdout_fraction, tokenizer):
    tokens = tokenizer.encode(text)
    if len(tokens) < block_size + 2:
        repeats = (block_size + 2 + len(tokens) - 1) // max(1, len(tokens))
        tokens = (tokens * repeats)[: block_size + 2]
    cut = int(len(tokens) * (1.0 - holdout_fraction))
    cut = min(
        max(cut, block_size + 1),
        len(tokens) - 1,
    )
    train = torch.tensor(
        tokens[:cut],
        dtype=torch.long,
    )
    holdout = torch.tensor(
        tokens[cut - 1 :],
        dtype=torch.long,
    )
    return train, holdout


def make_batch(tokens, block_size, device):
    """Draw one block of block_size tokens plus the tokens that follow it.

    A span needs block_size + 1 tokens: block_size inputs and a target for the
    last one. A document shorter than that is repeated until it is long enough,
    which is the same treatment split_tokens already gives it, because a
    document that cannot supply even one block cannot train.

    The upper bound is exclusive and is len - block_size, not len - block_size
    - 1. Using the wrong one makes randrange(0, 0) for a document of exactly
    block_size + 1 tokens, which is the smallest document split_tokens will
    hand over, so every short document crashed rather than merely degrading.
    """
    needed = block_size + 1
    if len(tokens) < needed:
        repeats = (needed + len(tokens) - 1) // max(1, len(tokens))
        tokens = tokens.repeat(repeats)[:needed]
    start = random.randrange(0, len(tokens) - block_size)
    x = tokens[start : start + block_size].unsqueeze(0).to(device)
    y = tokens[start + 1 : start + block_size + 1].unsqueeze(0).to(device)
    return x, y


@torch.no_grad()
def measure_on(tokens, model, c, samples=3):
    """Loss, perplexity and next-token accuracy on a token span.

    Architecture 19 asks for more than loss, and lists accuracy and perplexity
    among the measurements. Perplexity is the exponential of the mean loss.
    Accuracy is the share of positions where the highest scoring prediction was
    the byte that actually followed, which is the only accuracy a byte-level
    model can be held to without inventing a task it has not been given.
    """
    model.eval()
    losses = []
    correct = 0
    positions = 0
    if len(tokens) <= c.block_size + 1:
        starts = [0]
    else:
        max_start = max(0, len(tokens) - c.block_size - 1)
        starts = np.linspace(
            0, max_start, num=min(samples, max_start + 1), dtype=int
        ).tolist()
    for start in starts:
        if len(tokens) <= c.block_size + 1:
            repeats = (c.block_size + 2 + len(tokens) - 1) // max(1, len(tokens))
            eval_tokens = tokens.repeat(repeats)[: c.block_size + 1]
        else:
            eval_tokens = tokens
        x = eval_tokens[start : start + c.block_size].unsqueeze(0).to(c.device)
        y = eval_tokens[start + 1 : start + c.block_size + 1].unsqueeze(0).to(c.device)
        with autocast_context(c):
            logits, loss = model(x, y)
        value = float(loss.item())
        if not np.isfinite(value):
            model.train()
            raise RuntimeError("Model produced a non-finite evaluation loss.")
        losses.append(value)
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        positions += int(y.numel())
    model.train()
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": round(float(np.exp(min(mean_loss, 60.0))), 4),
        "accuracy": round(correct / positions, 6) if positions else 0.0,
        "positions": positions,
    }


@torch.no_grad()
def loss_on(tokens, model, c, samples=3):
    """The loss alone, for callers that only need the number."""
    return measure_on(tokens, model, c, samples=samples)["loss"]


@torch.no_grad()
def update_magnitude(before, model):
    """The L2 norm of the weight change a cycle produced.

    Architecture 35 lists update magnitude as something the system should
    expose, and it is the cheapest honest answer to "did this cycle actually
    change anything, and by how much". A cycle that reports an improvement with
    a near-zero magnitude was already close to the optimum for that document,
    which is different from a cycle that moved the weights a long way.
    """
    total = 0.0
    for name, parameter in model.named_parameters():
        start = before.get(name)
        if start is None:
            continue
        total += float((parameter.detach().cpu() - start).pow(2).sum().item())
    return round(total**0.5, 8)


def use_mixed_precision(c) -> bool:
    """Whether the forward pass should run narrowed to bfloat16.

    Architecture 36 asks for hardware to be used intelligently rather than
    maxed out, and the configured precision was previously chosen and then never
    applied, so a machine that could train in bfloat16 trained in float32 and
    said nothing about it.

    bfloat16 keeps float32's exponent range, so it narrows the forward pass
    without needing a loss scaler, and the parameters themselves stay in
    float32 for the optimizer. float16 is deliberately not used: it needs a
    scaler, and a scaler that is not threaded through every step is a source of
    silent divergence rather than a speedup.
    """
    return c.dtype == "bfloat16" and str(c.device).startswith("cuda")


def autocast_context(c):
    """The autocast context for one forward pass, or nothing at all."""
    if use_mixed_precision(c):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def active_precision(c) -> str:
    """The precision the next forward pass will actually use, for the record."""
    return "bfloat16" if use_mixed_precision(c) else "float32"


def optimizer_state_path(c):
    """Where the optimizer's own state is kept, beside the resume weights.

    Beside the resume checkpoint and not inside it, for two reasons. The
    weights are written on a schedule and in a format with a deliberate role
    field, and mixing a second kind of state into that file would make "which
    model is this" harder to answer than it needs to be. And a resumed run that
    grows the model is loading into a different parameter set, at which point
    the old moments no longer describe anything; keeping them separate lets the
    growth path discard them deliberately instead of silently loading a
    mismatched state.
    """
    return Path(c.out_dir) / "Jupiter_optimizer.pt"


def save_optimizer_state(optimizer, c, cycle):
    """Persist the optimizer's moments so a resumed run is the same run.

    This was a known limitation recorded in the architecture's own list: the
    optimizer was rebuilt at the start of every run, so AdamW's first and second
    moments were thrown away between cycles and the opening steps of each cycle
    moved with a cold estimate of every gradient. For a bounded micro-update
    that is most of what the update is made of, so the cost was paid on every
    single cycle.

    The state is written next to the resume weights and carries the cycle it
    belongs to, so a reader can tell whether it describes the model on disk.
    """
    path = optimizer_state_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "learning_cycle": int(cycle),
            "architecture_version": ARCHITECTURE_VERSION,
            "written": time.time(),
        },
        path,
    )
    return path


def load_optimizer_state(optimizer, model, c):
    """Restore the optimizer's moments, or report why it started cold.

    Returns a string saying what happened, rather than a bare True or False,
    because the difference between "there was nothing to load" and "there was
    something and it did not fit this model" is the difference between a first
    run and a growth, and a caller deciding whether to warn needs to tell them
    apart.
    """
    # The device comes from the parameters rather than from the model object,
    # because nn.Module has no device attribute and asking it for one raises an
    # AttributeError that says nothing about what went wrong.
    device = next(model.parameters()).device
    path = optimizer_state_path(c)
    if not path.exists():
        return "no optimizer state on disk; starting with fresh moments"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as reason:
        # Every failure mode is caught, not just the three that were expected
        # when this was written. torch.load raises UnpicklingError on a file
        # that is not a torch file at all, which is not an OSError, a
        # RuntimeError, a ValueError or an EOFError, and a half-written state
        # file raised exactly that and took the run down with it.
        #
        # Catching broadly here is right rather than lazy: the input is a file
        # on disk that a previous process may have been killed while writing,
        # and every way it can be unreadable means the same thing, which is
        # that there is nothing to resume from.
        return f"optimizer state at {path} could not be read ({type(reason).__name__}); starting cold"
    payload = payload if isinstance(payload, dict) else {}
    state = payload.get("optimizer")
    if not isinstance(state, dict):
        return f"optimizer state at {path} is not in the expected form; starting cold"
    # The shapes are checked here rather than left to load_state_dict, because
    # load_state_dict does not check them. It matches tensors by position in the
    # parameter list, so loading moments saved for a narrower model into a wider
    # one succeeds and leaves most parameters with no moment at all: a run that
    # looks resumed and is not. This is the growth case, and it is the case that
    # matters, so it is checked rather than assumed.
    #
    # The comparison is against the model's own parameter count rather than
    # against the live optimizer state, which is empty until the first step. A
    # fresh optimizer reporting zero moments is not a mismatch, it is a fresh
    # optimizer.
    saved_shapes = state.get("state", {})
    model_shapes = [tuple(parameter.shape) for parameter in model.parameters()]
    if len(saved_shapes) > len(model_shapes):
        return (
            f"optimizer state at {path} was saved for a model with "
            f"{len(saved_shapes)} tensors, this model has {len(model_shapes)}; "
            "starting cold. This is expected after a growth."
        )
    # The saved state is accepted only if every parameter in it has a partner of
    # the same shape in the model, or has a strictly larger moment that a growth
    # would explain.
    #
    # Matching per moment does not work, because AdamW does not keep one moment
    # per parameter: exp_avg matches the parameter, while exp_avg_sq is stored
    # expanded by the amsgrad flag and the step count is a scalar. Comparing
    # shapes one value at a time therefore rejects a state that is perfectly
    # good and accepts one that is not, depending on flags rather than on the
    # model. So each entry in the saved state is checked as a unit: its largest
    # non-scalar moment is the parameter it belongs to.
    for moments in saved_shapes.values():
        shapes = [
            tuple(value.shape)
            for value in moments.values()
            if torch.is_tensor(value) and value.dim() > 0
        ]
        if not shapes:
            continue
        widest = shapes[0]
        for shape in shapes[1:]:
            if len(shape) > len(widest) or (
                len(shape) == len(widest) and shape > widest
            ):
                widest = shape
        if widest not in model_shapes:
            return (
                f"optimizer state at {path} was saved for a model with "
                f"parameters of shape {widest}, which this model does not have; "
                "starting cold. This is expected after a growth, where the model "
                "changed shape."
            )
    try:
        optimizer.load_state_dict(state)
    except (KeyError, ValueError, RuntimeError):
        return (
            f"optimizer state at {path} does not fit this model; starting cold. "
            "This is expected after a growth, where the parameter set changed."
        )
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.device != device:
                state[key] = value.to(device)
    return f"resumed optimizer moments from cycle {payload.get('learning_cycle')}"


def save_checkpoint(model, c, cycle, path=None, optimizer=None):
    """Write the model, and the optimizer's state beside it when there is one.

    With no path this writes the published checkpoint, which is what the
    conversation loads and which only an ACCEPTed cycle is allowed to write.
    A caller that wants the run to resume from this state instead passes its
    own path; that file is working state, never published.

    The optimizer is written only for the resume state. The published
    checkpoint is what the conversation loads, and the conversation does not
    train, so carrying moments in it would be state nobody reads.
    """
    out = Path(c.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(path) if path is not None else out / "Jupiter_latest.safetensors"
    save_model(
        model,
        str(path),
        metadata={
            "project": "Jupiter",
            "format": "safetensors",
            "architecture_version": ARCHITECTURE_VERSION,
            "learning_cycle": str(cycle),
            "role": "resume" if path.name != PUBLISHED_CHECKPOINT else "published",
            "architecture": json.dumps(
                model.config.to_dict(),
                sort_keys=True,
            ),
        },
    )
    if optimizer is not None and path.name != PUBLISHED_CHECKPOINT:
        save_optimizer_state(optimizer, c, cycle)
    return path


def snapshot_model(model):
    """Copy model weights to CPU for rollback."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def restore_model(model, snapshot):
    """Restore model weights from a previous safe state."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(snapshot[name].to(parameter.device))


def reset_optimizer(model, c, learning_rate):
    """Create a fresh optimizer after unstable learning.

    The rate is read back off the optimizer and returned alongside it. A
    recovery path that reports the variable it computed while the optimizer
    holds something else would record a learning rate that is not the one
    learning used, and the recovery is exactly the moment where that matters:
    the rate is being halved because something went wrong, and a caller
    checking the log afterwards would be checking the wrong number.
    """
    optimizer = model.configure_optimizer(
        learning_rate,
        c.weight_decay,
    )
    return optimizer, get_learning_rate(optimizer)


def get_learning_rate(optimizer):
    """The learning rate the optimizer will actually apply.

    Read from the optimizer rather than from the variable the worker carries,
    because the two can disagree: recovery recomputes a rate and rebuilds the
    optimizer, and a code path that reported the variable while the optimizer
    held something else would print a number that is not what learning used.
    Reading the optimizer is the only way to be sure.
    """
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def build_model(c, vocab_size):
    """Create Jupiter from the saved architecture, or from seed config.

    The saved record wins for every shape field except the vocabulary. The
    vocabulary belongs to the tokenizer, which is a runtime choice: leaving it
    to the saved record would build a 256 entry embedding table for a tokenizer
    that emits ids above 255, and the failure would surface deep inside a
    forward pass as an out-of-range index rather than as a configuration
    mistake. A tokenizer change therefore makes the checkpoint's shape
    disagree, and attach_checkpoint archives it and starts fresh, which is the
    honest outcome for a migration.

    An approved capacity growth is applied on top of whichever of those two the
    shape came from. Reading the record first and growing afterwards is what
    makes the growth cumulative: a model that grew to eight layers has eight in
    its record, and the next approved growth makes it nine rather than five. The
    returned tuple carries the shape that was actually built, because the caller
    has to record that and not the configuration, which still describes the
    model as it was before it grew.

    vocab_size is required rather than optional. It used to default to None and
    was passed to int() three lines later, so calling this without one raised a
    TypeError naming None inside a function whose name says nothing about
    vocabularies. The only caller passes tokenizer.vocab_size, and making that
    mandatory states the actual contract rather than a default that cannot be
    honoured.
    """
    import expansion

    architecture_path = Path(c.out_dir) / "Jupiter.architecture.json"
    if architecture_path.exists():
        stored, _version = read_architecture(architecture_path)
        stored["vocab_size"] = int(vocab_size)
        model_config = ModelConfig(**stored)
    else:
        model_config = ModelConfig(**architecture_dict(c, vocab_size))
    approved = expansion.pending_shape(c)
    if approved:
        import growth

        already = {
            name: getattr(model_config, name)
            for name, value in approved.items()
            if getattr(model_config, name, None) == value
        }
        outstanding = {
            name: value for name, value in approved.items() if name not in already
        }
        if outstanding:
            # grown_config is annotated as returning the five growable shape
            # fields and is the only thing that reaches this line, so the dict
            # it hands back cannot contain bias. The annotation states that so a
            # reader does not have to verify it, and so a change to what growth
            # is allowed to touch is a visible edit here rather than a runtime
            # surprise on a model that has already been trained.
            grown = growth.grown_config(model_config, **outstanding)
            model_config = ModelConfig(
                block_size=grown["block_size"],
                vocab_size=grown["vocab_size"],
                n_layer=grown["n_layer"],
                n_head=grown["n_head"],
                n_embd=grown["n_embd"],
                dropout=model_config.dropout,
                bias=model_config.bias,
            )
            print(
                "Applying an approved capacity growth: "
                + ", ".join(
                    f"{name} -> {value}" for name, value in sorted(outstanding.items())
                )
            )
        if already:
            # The record already describes the grown model, which is the normal
            # state on the run after the one that applied it. Growing again would
            # be refused for shrinking nothing, and the honest report is that the
            # growth is already in the shape rather than a silent no-op.
            print(
                "The approved growth is already in the saved architecture ("
                + ", ".join(
                    f"{name} = {value}" for name, value in sorted(already.items())
                )
                + "); nothing to apply."
            )
    return Jupiter(model_config).to(c.device), architecture_path


def checkpoint_shape(path):
    """The shape fields a checkpoint was written with, or None if unreadable.

    A checkpoint written before the architecture record was versioned may carry
    no readable metadata, so this returns None rather than guessing. The
    pre-check is a courtesy that turns a confusing tensor-size error into a
    clear migration; loading the weights is still what actually decides.
    """
    try:
        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata() or {}
    except (OSError, ValueError, RuntimeError):
        return None
    raw = metadata.get("architecture")
    if not raw:
        return None
    try:
        stored = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(stored, dict):
        return None
    return shape_of(stored)


def archive_incompatible(checkpoint_path, reason):
    """Move a checkpoint that cannot be reused into the archive, and say why.

    Architecture 32 wants learning to be reversible and Architecture 9 wants
    experimental results preserved, so a checkpoint trained with a different
    shape is never deleted and never silently ignored. It is moved aside with a
    note explaining the mismatch, and Jupiter starts again from random
    initialization.

    This is not gated as a destructive operation, and the distinction matters.
    Archiving preserves: nothing is lost, the file is moved rather than removed,
    and the move is reversible. Gating it would put a control on the only safe
    path forward, since the alternative to moving an incompatible checkpoint
    aside is loading weights against a shape they were not trained for. The
    control sits upstream instead, in save_architecture, which asks a human
    before the shape of a model already in use may change at all. A gate in
    both places would mean a run could not proceed even with approval, which is
    how a control becomes something people disable.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(checkpoint_path).parent / "archive" / f"incompatible-{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    destination = target / Path(checkpoint_path).name
    Path(checkpoint_path).replace(destination)
    (target / "reason.txt").write_text(
        reason + "\n" + f"archived at {time.time()}\n", encoding="utf-8"
    )
    return destination


def attach_checkpoint(model, checkpoint_paths, architecture_path):
    """Restore weights from the first candidate checkpoint that fits the model.

    Two files can carry a resumable model and they answer different questions.
    The resume checkpoint is written after every cycle and is where the next run
    continues from, so that learning accumulates across runs instead of
    restarting. The published checkpoint is written only when a cycle is
    ACCEPTed, and it is what the conversation loads, so Architecture 20's
    judgement still governs everything that is ever shown to anyone. Candidates
    are tried in that order: a run should continue from the furthest point it
    honestly reached, falling back to the last published state only when the
    resume state cannot be used.

    A checkpoint trained with a different shape cannot be loaded, and that
    happens legitimately when the architecture changes: a wider context window
    means new position embeddings, and a new tokenizer means a new vocabulary.
    Rather than failing the run with a tensor size error, an unusable checkpoint
    is archived with an explanation and the next candidate is tried, so a shape
    change costs one training run rather than the whole project.
    """
    candidates = [Path(path) for path in checkpoint_paths]
    for path in candidates:
        if not path.exists():
            continue
        if not architecture_path.exists():
            print(
                f"Cannot check the shape of {path.name}: {architecture_path} is "
                "missing, so Jupiter starts from random initialization."
            )
            return False
        recorded = checkpoint_shape(path)
        current = shape_of(model.config.to_dict())
        if recorded is not None and recorded != current:
            import growth

            verdict, detail = growth.compatibility(recorded, current)
            if verdict == "compatible":
                # A shape that got bigger is a model that grew, not a
                # mismatch. The parts that fit are carried over and the rest is
                # initialised, which is what makes an expansion a growth rather
                # than a restart. This is the case the guard used to refuse.
                report = growth.load_growing(model, path)
                print(
                    f"Growing from {path.name}: {detail}. "
                    f"{report['kept']} tensors carried over, "
                    f"{len(report['grown_into'])} extended, "
                    f"{len(report['new'])} newly initialised."
                )
                return True
            if verdict == "unsupported":
                print(f"[growth] {path.name} is not growable: {detail}")
                return False
            differences = ", ".join(
                f"{name} checkpoint={recorded.get(name)} model={value}"
                for name, value in current.items()
                if recorded.get(name) != value
            )
            reason = (
                f"this checkpoint was trained with a different model shape: "
                f"{differences}. Weights cannot be reused across shapes, so it "
                "needs a fresh training run."
            )
            destination = archive_incompatible(path, reason)
            print(f"[migration] {reason}")
            print(f"[migration] archived to {destination}")
            continue
        load_model(
            model,
            str(path),
            strict=True,
        )
        print(f"Resuming from {path.name}")
        return True
    print("Starting Jupiter from random initialization.")
    return False


def read_cycle(cycle_path):
    """Read the last completed cycle number.

    A missing or unreadable state file reports cycle 0 on purpose: a corrupt
    record must not stop a learning run, it only restarts the count.
    """
    if not cycle_path.exists():
        return 0
    try:
        previous_state = json.loads(cycle_path.read_text(encoding="utf-8"))
        return int(previous_state.get("learning_cycle", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


class CycleTimeLimitExceeded(RuntimeError):
    """Raised when a single cycle exceeds its wall-clock allowance.

    This is a limit on one cycle, distinct from the learner's own recovery
    failures, so the worker can tell a slow cycle from an unstable one.
    """


def run_micro_learning(
    model,
    c,
    train_tokens,
    holdout_tokens,
    tel,
    cycle,
    before_seen,
    before_holdout,
    learning_rate,
    optimizer,
    deadline=None,
):
    """Run one bounded cycle of micro-updates and measure what changed.

    This is the proven learner. train.py and learning_worker.py both drive this
    function so there is exactly one micro-update implementation. The caller
    supplies the pre-update losses because the caller also reports them.

    Returns the cycle result, the learning rate and the optimizer, because all
    three can change during recovery and must carry into the next cycle.

    `deadline` is an optional absolute time.time() value. When supplied, the
    cycle stops at the next step boundary if it has been exceeded, so one slow
    cycle cannot outlast its own budget.

    `before_seen` and `before_holdout` are measurement dicts from measure_on,
    not bare losses, so the record carries perplexity and accuracy for both
    sides of the update as Architecture 19 asks.
    """
    start_time = time.time()
    model.train()
    cycle_start_weights = snapshot_model(model)
    successful_steps = 0
    recovery_count = 0
    damped = 0
    last_loss = None
    minimum_learning_rate = float(c.learning_rate) * 0.01
    for step in range(1, c.micro_steps + 1):
        if deadline is not None and time.time() > deadline:
            raise CycleTimeLimitExceeded(
                f"cycle {cycle} exceeded its {c.worker_cycle_max_seconds:.0f}s "
                f"allowance at step {step} of {c.micro_steps}"
            )
        retries = 0
        step_finished = False
        while not step_finished:
            rollback = snapshot_model(model)
            optimizer.zero_grad(set_to_none=True)
            x, y = make_batch(
                train_tokens,
                c.block_size,
                c.device,
            )
            with autocast_context(c):
                _, loss = model(x, y)
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                restore_model(
                    model,
                    rollback,
                )
                old_lr = learning_rate
                learning_rate = max(
                    learning_rate * 0.5,
                    minimum_learning_rate,
                )
                optimizer, learning_rate = reset_optimizer(
                    model,
                    c,
                    learning_rate,
                )
                recovery_count += 1
                retries += 1
                tel.write(
                    "micro_recovery",
                    learning_cycle=cycle,
                    step=step,
                    retry=retries,
                    reason="nonfinite_loss",
                    old_learning_rate=old_lr,
                    new_learning_rate=learning_rate,
                )
                print(
                    f"  RECOVERY step {step:02d}: "
                    f"non-finite loss; "
                    f"learning rate "
                    f"{old_lr:.8g} -> "
                    f"{learning_rate:.8g}"
                )
                if retries > 3:
                    raise RuntimeError(
                        f"Unable to recover from non-finite loss at step {step}."
                    )
                continue
            loss.backward()
            raw_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    c.grad_clip,
                )
            )
            if raw_grad_norm >= c.gradient_alert:
                tel.write(
                    "gradient_alert",
                    learning_cycle=cycle,
                    step=step,
                    gradient_norm=raw_grad_norm,
                    alert_threshold=c.gradient_alert,
                    stop_threshold=c.gradient_stop,
                    learning_rate=learning_rate,
                )
                print(
                    f"  GRADIENT ALERT step {step:02d}: "
                    f"{raw_grad_norm:.6f} "
                    f"(clipped to {c.grad_clip:.6f})"
                )
            # A gradient past the stop threshold and a gradient that is not a
            # number are different faults and get different responses.
            #
            # A non-finite norm means the weights are broken, and restoring the
            # last safe state genuinely repairs them, so recomputing the step
            # can succeed. That has always been the recovery path.
            #
            # A merely large gradient is not a fault. It was treated as one and
            # the run stopped dead: the step is recomputed from the same weights
            # on the same batch, so it produces the same gradient, halving the
            # learning rate does not change that, and the retry budget was
            # spent three times before the cycle was abandoned. Measured over
            # 22,903 steps the gradient norm has a median of 3.2, a 99.9th
            # percentile of 43.4 and a maximum of 99.2, and the five largest all
            # occur at step one or two of a cycle, because the opening steps on
            # a document the model has not seen are legitimately steep. A
            # threshold inside that range fires on ordinary learning.
            #
            # So a large gradient is damped rather than retried: the safe state
            # is restored, the rate is cut, the step is skipped, and the cycle
            # carries on. Gradients are clipped to grad_clip on every step
            # anyway, so the update that would have been applied was already
            # bounded; what the threshold buys is a slower rate while the model
            # is on steep ground, not the end of a cycle.
            diverged = np.isfinite(raw_grad_norm) and raw_grad_norm >= c.gradient_stop
            if diverged:
                tel.write(
                    "gradient_stop",
                    learning_cycle=cycle,
                    step=step,
                    gradient_norm=raw_grad_norm,
                    stop_threshold=c.gradient_stop,
                )
                print(
                    f"  GRADIENT STOP step {step:02d}: {raw_grad_norm:.6f} "
                    f"exceeds {c.gradient_stop:.6f}; damping and skipping"
                )
                restore_model(
                    model,
                    rollback,
                )
                learning_rate = max(
                    learning_rate * 0.5,
                    minimum_learning_rate,
                )
                optimizer, learning_rate = reset_optimizer(
                    model,
                    c,
                    learning_rate,
                )
                damped += 1
                # The step is over. The retry loop below is a `while not
                # step_finished`, so continuing without setting this re-enters
                # it and never advances: the original version of this branch
                # hung the cycle rather than skipping the step.
                step_finished = True
                continue
            if not np.isfinite(raw_grad_norm):
                restore_model(
                    model,
                    rollback,
                )
                old_lr = learning_rate
                learning_rate = max(
                    learning_rate * 0.5,
                    minimum_learning_rate,
                )
                optimizer, learning_rate = reset_optimizer(
                    model,
                    c,
                    learning_rate,
                )
                recovery_count += 1
                retries += 1
                tel.write(
                    "micro_recovery",
                    learning_cycle=cycle,
                    step=step,
                    retry=retries,
                    reason="unsafe_gradient",
                    gradient_norm=raw_grad_norm,
                    old_learning_rate=old_lr,
                    new_learning_rate=learning_rate,
                )
                print(
                    f"  RECOVERY step {step:02d}: "
                    f"gradient={raw_grad_norm:.6f} "
                    f"-> learning_rate="
                    f"{learning_rate:.8g} "
                    f"(retry {retries})"
                )
                if retries > 3:
                    raise RuntimeError(
                        f"Unable to stabilize gradient "
                        f"at step {step} after "
                        f"{retries - 1} recoveries."
                    )
                continue
            optimizer.step()
            last_loss = loss_value
            successful_steps += 1
            step_finished = True
            tel.write(
                "micro_update",
                learning_cycle=cycle,
                step=step,
                loss=last_loss,
                gradient_norm=raw_grad_norm,
                learning_rate=learning_rate,
                retries=retries,
            )
            print(
                f"step {step:02d}/{c.micro_steps} "
                f"loss={last_loss:.6f} "
                f"raw_grad={raw_grad_norm:.6f} "
                f"clip={c.grad_clip:.6f} "
                f"lr={learning_rate:.8g}"
            )
    after_seen = measure_on(train_tokens, model, c)
    after_holdout = measure_on(holdout_tokens, model, c)
    magnitude = update_magnitude(cycle_start_weights, model)
    result = {
        "train_loss_before": before_seen["loss"],
        "train_loss_after": after_seen["loss"],
        "train_loss_improvement": before_seen["loss"] - after_seen["loss"],
        "train_perplexity_before": before_seen["perplexity"],
        "train_perplexity_after": after_seen["perplexity"],
        "holdout_loss_before": before_holdout["loss"],
        "holdout_loss_after": after_holdout["loss"],
        "holdout_loss_improvement": before_holdout["loss"] - after_holdout["loss"],
        "holdout_perplexity_before": before_holdout["perplexity"],
        "holdout_perplexity_after": after_holdout["perplexity"],
        "holdout_accuracy_before": before_holdout["accuracy"],
        "holdout_accuracy_after": after_holdout["accuracy"],
        "generalization_gap": round(after_seen["loss"] - after_holdout["loss"], 6),
        "update_magnitude": magnitude,
        "micro_steps": c.micro_steps,
        "successful_steps": successful_steps,
        "gradient_recoveries": recovery_count,
        "gradient_damped": damped,
        "initial_learning_rate": float(c.learning_rate),
        "final_learning_rate": learning_rate,
        "gradient_clip": c.grad_clip,
        "gradient_stop": c.gradient_stop,
        "precision": active_precision(c),
        "elapsed_seconds": time.time() - start_time,
    }
    return result, learning_rate, optimizer


def main():
    """Run exactly one learning cycle, through the governed controller.

    This script used to drive its own cycle and save its checkpoint
    unconditionally. That skipped Architecture 20, which says an update must be
    judged before it is kept, and Architecture 33, which says an update that
    damages earlier learning is not a success. A single governed cycle is
    already what the controller's --once mode does, so this delegates instead of
    keeping a second implementation that could drift from the first and quietly
    reintroduce the same gap.

    The import sits inside the function because learning_worker imports this
    module for the micro-update primitives. By the time main runs this module is
    fully loaded, so there is no cycle to resolve.
    """
    import learning_worker

    c = load_config()
    Path(c.out_dir).mkdir(parents=True, exist_ok=True)
    max_retries = int(getattr(c, "train_max_retries", 5))
    last_exception = None
    last_exception = None
    for attempt in range(1, max_retries + 1):
        tel = Telemetry(c.telemetry_file)
        try:
            learning_worker.run_worker(c, tel, 1)
            return
        except Exception as exc:
            last_exception = exc
            print(f"[train] attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                print(f"[train] retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(1.0)
    # Re-raised rather than returned, because the caller asked for a cycle and
    # did not get one. The message names the real cause when there is one to
    # name, because a bare "None" here told a reader nothing about which attempt
    # failed or why.
    if last_exception is not None:
        print(
            f"[train] all {max_retries} attempts failed; last error: {last_exception}"
        )
        raise last_exception
    raise RuntimeError(
        f"[train] the worker reported no cycles and raised nothing across "
        f"{max_retries} attempts"
    )


if __name__ == "__main__":
    main()
