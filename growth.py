"""Growing a model without throwing away what it has learned (Architecture 22).

Architecture 22 lists six ways to increase representational capacity: more
neurons, more hidden dimensions, more layers, specialised modules, expert
modules and auxiliary networks. Every one of them changes the shape of the
model, and until now that was the whole problem: the shape guard archives any
checkpoint whose shape disagrees, because loading weights against a shape they
were not trained for either raises a tensor error or, worse, silently pairs the
wrong numbers together.

So growth was not available at all. The guard was right to refuse, and the six
mechanisms remained unbuilt. This is the piece that makes them possible, and it
is a distinction the guard was missing rather than a weakening of it.

A shape that got BIGGER is not the same event as a shape that is incompatible.
A checkpoint with three blocks and a model with four is a model that grew, and
the three blocks are still exactly the weights they were. A checkpoint with a
512-token window and a model with 512 is a match. A checkpoint with 1024 tokens
and a model with 512 is a different model, and the honest response is to archive
it, exactly as before.

So the rule is one-directional: the saved model may be smaller in every
respect, never larger, and the parts that overlap are loaded and the rest is
freshly initialised. That is the "hooks the new neurons onto the old ones"
mechanism, and it is what makes an expansion a growth rather than a restart.

Widening the hidden dimension was refused here for most of the project's life,
and the reason was right: the attention projection fuses Q, K and V into one
tensor whose rows are Q, then K, then V, so copying a prefix of those rows puts
the learned Q weights where K belongs. The model still runs and still produces
numbers, so nothing downstream would catch it. What that reasoning missed is
that the tensor is three stacked blocks rather than one, and each block can be
copied as a block. The MLP up-projection stacks four the same way. So n_embd is
now a growable axis, and n_embd +1 layers is not the only way to add capacity.

What this does not do is prove the grown model is any good. Growth is proposed,
measured against the reserved document, and kept or rolled back by the same
acceptance gate as any other change, because a larger model that has forgotten
everything is worse than the one it replaced. A widened model is also not
required to reproduce the narrow one exactly: it has more degrees of freedom, so
what is guaranteed is that every learned weight survives in the slot it was
learned for, not that the outputs are identical.
"""

GROWTH_FIELDS = ("block_size", "vocab_size", "n_layer", "n_head", "n_embd")

# Tensors whose FIRST dimension is several stacked copies of the hidden
# dimension rather than the hidden dimension itself. Widening these by copying a
# prefix of the rows would place the learned Q rows where the K rows belong,
# because Q, K and V are fused into one tensor with Q first. Each group is
# therefore copied as its own block, which is the only arrangement that keeps
# every learned weight attached to the parameter it was learned for.
#
# The multiplier is how many copies of n_embd the first dimension holds:
# qkv fuses three (Q, K, V), the MLP up-projection holds four.
STACKED_ROW_MULTIPLIERS = {
    "qkv.weight": 3,
    "qkv.bias": 3,
    "mlp.fc.weight": 4,
    "mlp.fc.bias": 4,
}


def _stacked_key(name):
    """The stacked-tensor rule for a parameter name, or None if it has none.

    Matched on the suffix rather than the full path so it holds for every block
    in the model, present and future, without enumerating layer indices.
    """
    for suffix, multiplier in STACKED_ROW_MULTIPLIERS.items():
        if name.endswith(suffix):
            return multiplier
    return None


def compatibility(recorded, target):
    """Whether a saved model can grow into the target, and by how much.

    Returns (verdict, detail) where verdict is "compatible" or "incompatible".
    Compatible means every shape field of the saved model is less than or equal
    to the target's. A field that shrank means the saved weights genuinely do not
    fit, and that is the case the guard already handled correctly.
    """
    if not recorded:
        return "incompatible", "the checkpoint records no readable shape"
    smaller = []
    larger = []
    for name in GROWTH_FIELDS:
        was = recorded.get(name)
        now = target.get(name)
        if was is None or now is None:
            continue
        if was == now:
            continue
        if was < now:
            smaller.append(f"{name} {was} -> {now}")
        else:
            larger.append(f"{name} {now} -> {was}")
    if larger:
        return (
            "incompatible",
            "the saved model is larger than this one: " + ", ".join(larger),
        )
    if not smaller:
        return "compatible", "the shapes match"
    return "compatible", "growth: " + ", ".join(smaller)


def _slice_for(saved_shape, model_shape):
    """The part of a saved tensor that fits inside the model tensor.

    Returns None when the shapes cannot be aligned, which is different from
    both being equal and one being a prefix of the other.
    """
    if len(saved_shape) != len(model_shape):
        return None
    slices = []
    for was, now in zip(saved_shape, model_shape):
        if was > now:
            return None
        slices.append(slice(0, was))
    return tuple(slices)


def _recorded_embd(path):
    """The hidden width the checkpoint says it was trained at, or None.

    Read through train.checkpoint_shape so the metadata format stays owned by
    one module. It is imported inside the function because growth is imported by
    train, and a module-level import would be a cycle.
    """
    try:
        import train
    except ImportError:
        return None
    shape = train.checkpoint_shape(path)
    if isinstance(shape, dict):
        return shape.get("n_embd")
    return None


def _old_embd_for(name, saved_shape, reference):
    """The hidden width a saved tensor was built at, or None if unknowable.

    This cannot be read off "the last dimension" or "the first dimension"
    because the hidden size appears in different places in different tensors.
    The embedding is [vocab, n_embd], the attention output projection is
    [n_embd, n_embd], the MLP down-projection is [n_embd, 4*n_embd] and the
    fused QKV is [3*n_embd, n_embd]. Taking the last dimension of the MLP
    down-projection would report four times the true width and then place every
    learned weight at the wrong offset.

    `reference` is the checkpoint's own recorded width, which states it outright. It is the authority because it was written by the run that trained
    the model, rather than inferred from a tensor whose layout this function
    would then be guessing at. The inference below is only a fallback for a
    checkpoint whose recorded shape is unreadable, and it is deliberately
    conservative: it returns a number only when the stacked rule or an
    unambiguous square tensor pins it down.
    """
    if reference:
        return int(reference)
    multiplier = _stacked_key(name)
    if multiplier is not None:
        if saved_shape and saved_shape[0] % multiplier == 0:
            return saved_shape[0] // multiplier
        return None
    if len(saved_shape) == 2 and saved_shape[0] == saved_shape[1]:
        return saved_shape[0]
    if not saved_shape:
        return None
    # A two-dimensional tensor that is not square and not a known stack: the
    # hidden size is the smaller of the two, because the larger is either the
    # vocabulary or a multiple of the hidden size on the other axis.
    return min(saved_shape[0], saved_shape[-1])


def _widen_stacked(name, saved, target, old_embd, new_embd):
    """Copy a stacked-row tensor into a wider one, one group at a time.

    A fused QKV projection has Q, K and V stacked down its first dimension, so
    the saved tensor is three copies of [n_embd, n_embd] end to end. Copying the
    first new_embd rows would put every learned Q row where K belongs and leave
    V empty, and the model would then produce a number rather than an error: the
    failure is silent, which is why this is written out rather than left to the
    generic path.

    Each group is placed at its own offset in the wider tensor, so the learned
    weight for Q stays the Q weight, the K weight stays the K weight, and the
    new columns inside each group are freshly initialised. The result is a model
    with more capacity per group that still computes what the old one computed
    on the subspace it already knew.
    """
    multiplier = _stacked_key(name)
    if multiplier is None:
        return None
    if saved.shape[0] != multiplier * old_embd:
        # The saved tensor is not the stacked shape this rule assumes. Guessing
        # would silently misplace weights, so the caller falls back to refusing.
        return None
    merged = target.clone()
    for group in range(multiplier):
        old_block = saved[group * old_embd : (group + 1) * old_embd]
        new_block = merged[group * new_embd : (group + 1) * new_embd]
        if new_block.dim() == 1:
            # A bias: one value per hidden unit, so the group's own rows are the
            # whole tensor and there is no second axis to slice.
            new_block[:old_embd] = old_block
        else:
            new_block[:old_embd, :old_embd] = old_block
    return merged


def load_growing(model, path):
    """Load what a grown model can keep, and initialise the rest.

    Returns a report naming every tensor that was carried over, every tensor
    that was partly carried over, and every tensor that is new. A report that
    only said "loaded" would leave a caller unable to tell a genuine growth from
    a checkpoint that matched nothing.
    """
    from safetensors import safe_open

    target = model.state_dict()
    kept, partial = [], []
    payload = {}
    # Read from the model rather than being told, because whether a tensor needs
    # the stacked treatment depends on which axis moved, and the caller knows the
    # old and new hidden size but not which tensors those are.
    new_embd = int(getattr(model.config, "n_embd", 0) or 0)
    # The checkpoint's own recorded width, which is the authority for where a
    # learned weight belongs. Read before the loop so an unreadable record is
    # handled once rather than per tensor.
    recorded = _recorded_embd(path)

    with safe_open(str(path), framework="pt") as handle:
        available = set(handle.keys())
        for name in available:
            if name not in target:
                continue
            saved = handle.get_tensor(name)
            model_shape = list(target[name].shape)
            saved_shape = list(saved.shape)
            exact = all(was == now for was, now in zip(saved_shape, model_shape))
            if exact:
                kept.append(name)
                payload[name] = saved.clone()
                continue
            merged = None
            old_embd = _old_embd_for(name, saved_shape, recorded)
            if old_embd and new_embd and old_embd < new_embd:
                merged = _widen_stacked(name, saved, target[name], old_embd, new_embd)
            if merged is None:
                # The general case: any axis that grew is filled with the prefix
                # the saved tensor already has and freshly initialised beyond
                # it. This covers the context window, the vocabulary and the
                # hidden size for every tensor that is not a stacked group, and
                # it runs whether or not the width itself moved.
                slices = _slice_for(saved_shape, model_shape)
                if slices is not None:
                    merged = target[name].clone()
                    merged[slices] = saved
            if merged is None:
                continue
            partial.append(name)
            payload[name] = merged

    fresh = sorted(available - set(payload))
    model.load_state_dict(payload, strict=False)
    return {
        "kept": len(kept),
        "grown_into": partial,
        "new": fresh,
        "note": (
            "Tensors that fitted were carried over whole, tensors that grew were "
            "filled in the part that fits and freshly initialised beyond it, and "
            "anything the saved model did not have is new."
        ),
    }


def grown_config(config, **changes) -> dict:
    """A copy of a model configuration with some fields larger.

    Rejects a field that would shrink, because a model that is going to forget
    is not a growth and Architecture 32 wants learning to be reversible. Width is
    accepted here and refused at load time with its own reason, because refusing
    it in two places would leave the reason in the place a caller is least
    likely to be looking.

    Returns exactly the five growable shape fields, and never a bias or a
    dropout. Growth changes how big the model is; it does not change what kind
    of model it is. The return is a plain dict because the caller rebuilds a
    ModelConfig from it, and the shape of that dict is part of the contract:
    callers enumerate these five names rather than splatting it, so a new field
    appearing here is a deliberate change rather than a silent one.
    """
    values = {
        "block_size": config.block_size,
        "vocab_size": config.vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
    }
    for name, amount in changes.items():
        if name not in values:
            raise ValueError(f"{name} is not a growable shape field")
        if amount < values[name]:
            raise ValueError(
                f"{name} cannot shrink from {values[name]} to {amount}: a model "
                "that is going to forget is not a growth"
            )
        values[name] = amount
    return values


def grow(model, **changes):
    """A larger model, initialised fresh, ready to be loaded over an old one.

    Returns (new_model, grown_fields). The caller loads the old weights over it
    with load_growing; nothing here touches a checkpoint, so a growth can be
    proposed and measured before anything is persisted.

    This is the preview half of a growth, and preview_answer uses it so an
    operator can see what a change would cost before authorising it rather than
    after. Building the model is the expensive part of widening, because every
    tensor in it changes shape, and being told the parameter count first is more
    useful than being told afterwards.
    """
    from model import Jupiter, ModelConfig

    values = grown_config(model.config, **changes)
    grown = Jupiter(ModelConfig(**values)).to(model.lm_head.weight.device)
    touched = [
        name
        for name, amount in changes.items()
        if amount != getattr(model.config, name)
    ]
    return grown, touched


def parameter_count(model):
    """How many parameters a model has, counted from the tensors it owns.

    Counted rather than estimated, because the number a person is shown before
    authorising a growth has to be the number the model will actually have.
    """
    return int(sum(t.numel() for t in model.state_dict().values()))


def preview(c, **changes):
    """What a proposed growth would cost, without applying or persisting it.

    Read-only and ungated on purpose: nothing is changed, so there is nothing to
    authorise. It is the honest answer to the question a person asks before
    saying yes, which is how much bigger, because "n_embd 256 -> 320" does not
    convey that the model gains a third of a million parameters.
    """
    from model import Jupiter, ModelConfig

    current = {
        "n_layer": int(getattr(c, "n_layer", 4)),
        "n_embd": int(getattr(c, "n_embd", 256)),
        "n_head": int(getattr(c, "n_head", 4)),
        "block_size": int(getattr(c, "block_size", 512)),
        "vocab_size": int(getattr(c, "byte_vocab_size", 256)),
    }
    wanted = {**current, **changes}
    before = Jupiter(ModelConfig(**current))
    after = Jupiter(ModelConfig(**wanted))
    old_count = parameter_count(before)
    new_count = parameter_count(after)
    return {
        "from": {k: current[k] for k in changes},
        "to": {k: wanted[k] for k in changes},
        "parameters_before": old_count,
        "parameters_after": new_count,
        "parameters_added": new_count - old_count,
        "percent_larger": (
            round((new_count - old_count) / old_count * 100.0, 2) if old_count else None
        ),
        "note": (
            "computed from the models themselves, not estimated. Nothing is "
            "written and nothing is gated, because nothing is changed."
        ),
    }
