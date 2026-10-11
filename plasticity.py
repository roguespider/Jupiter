"""Jupiter's experimental reward-modulated plasticity learner.

This module is deliberately separate from ``train.py``.  It does NOT use
back-propagation, an optimizer, or an optimizer state.  Instead it implements a
small, bounded Hebbian-style learning rule with an explicit gain/loss signal:

    useful/correct activity  -> GAIN
    harmful/incorrect activity -> LOSS

The rule is reward-modulated Hebbian plasticity.  For a selected Linear layer,
its input activity (pre) and output activity (post) are recorded during a
normal forward pass.  A signed reward then strengthens or weakens the
correlation between those activities.

This is an EXPERIMENT, not a claim that this is the final way Jupiter should
learn language.  It is intentionally conservative:

* no gradients are calculated;
* no optimizer is touched;
* only selected Linear layers are plastic by default;
* every layer update has a hard norm bound;
* weights receive a small homeostatic pull toward their pre-update scale;
* the complete update can be reverted from the returned snapshot;
* non-finite updates are refused before they reach the model.

The surrounding learning worker can therefore use the same acceptance gate it
already has for gradient learning: propose -> evaluate -> keep or rollback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn


@dataclass
class PlasticityConfig:
    """Safety and learning parameters for one plasticity event."""

    # Base size of the signed Hebbian update.
    learning_rate: float = 0.0005
    # Prevent one layer from receiving an arbitrarily large update.
    max_update_norm: float = 0.0025
    # Tiny pull toward the old weight scale. This prevents repeated positive
    # correlation from silently inflating weights.
    homeostasis: float = 0.00002
    # Clamp the signed learning signal before it reaches the update.
    reward_clip: float = 1.0
    # Number of most-relevant token positions sampled from the sequence. Zero
    # means use every position. Sampling keeps long documents cheap.
    max_positions: int = 256
    # Only these kinds of layers are plastic by default. LayerNorm, embeddings,
    # biases and the tied language head are left alone until we have evidence
    # that a later experiment needs them.
    linear_name_fragments: tuple[str, ...] = (
        ".attn.proj",
        ".mlp.proj",
    )
    # A layer whose update is below this fraction of its weight norm is still
    # considered a real event and is retained in telemetry.
    min_effective_fraction: float = 1e-8

    def to_dict(self):
        return asdict(self)


@dataclass
class PlasticityResult:
    """Everything the caller needs to evaluate or undo a plasticity event."""

    applied: bool
    reason: str
    feedback: float
    mean_reward: float
    correct_fraction: float
    gain_tokens: int
    loss_tokens: int
    layers_changed: int
    total_update_norm: float
    layer_updates: Dict[str, dict]
    snapshot: Dict[str, torch.Tensor]

    def telemetry(self):
        """Return JSON-safe information, excluding the tensor snapshot."""
        return {
            "applied": self.applied,
            "reason": self.reason,
            "feedback": self.feedback,
            "mean_reward": self.mean_reward,
            "correct_fraction": self.correct_fraction,
            "gain_tokens": self.gain_tokens,
            "loss_tokens": self.loss_tokens,
            "layers_changed": self.layers_changed,
            "total_update_norm": self.total_update_norm,
            "layer_updates": self.layer_updates,
        }


class RewardModulatedHebbian:
    """Apply bounded gain/loss plasticity directly to a live Jupiter model."""

    def __init__(self, config: Optional[PlasticityConfig] = None):
        self.config = config or PlasticityConfig()

    def eligible_layers(self, model: nn.Module) -> Dict[str, nn.Linear]:
        """Return the Linear modules this learner is allowed to modify."""
        selected = {}
        fragments = self.config.linear_name_fragments
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name == "lm_head":
                # Jupiter ties lm_head.weight to token_embedding.weight.  Direct
                # plasticity here would also alter the input vocabulary matrix,
                # making the first experiment much harder to reason about.
                continue
            if fragments and not any(fragment in name for fragment in fragments):
                continue
            if module.weight.requires_grad:
                selected[name] = module
        return selected

    @staticmethod
    def _normalise_activity(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """RMS-normalise each activation vector without building a graph."""
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        return x / rms

    @staticmethod
    def _flatten_activity(x: torch.Tensor) -> torch.Tensor:
        """Flatten arbitrary leading dimensions into one activity-position axis."""
        return x.reshape(-1, x.shape[-1])

    def _positions(self, count: int, device: torch.device) -> torch.Tensor:
        limit = int(self.config.max_positions)
        if limit <= 0 or count <= limit:
            return torch.arange(count, device=device)
        # Evenly spaced positions are deterministic and cover the whole context
        # instead of always teaching from the first N tokens.
        return torch.linspace(0, count - 1, limit, device=device).long()

    def _reward_signal(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        feedback: float,
    ) -> tuple[torch.Tensor, float, float, int, int]:
        """Create a signed per-token gain/loss signal from observable outcomes."""
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets)
            reward = torch.where(correct, torch.ones_like(logits[..., 0]), -torch.ones_like(logits[..., 0]))
            reward = reward * float(feedback)
            reward = reward.clamp(-self.config.reward_clip, self.config.reward_clip)
            correct_fraction = float(correct.float().mean().item()) if correct.numel() else 0.0
            mean_reward = float(reward.mean().item()) if reward.numel() else 0.0
            gain_tokens = int((reward > 0).sum().item())
            loss_tokens = int((reward < 0).sum().item())
        return reward, correct_fraction, mean_reward, gain_tokens, loss_tokens

    def _collect(self, model: nn.Module, idx: torch.Tensor, targets: torch.Tensor):
        """Run one no-grad forward pass and retain local pre/post activity."""
        layers = self.eligible_layers(model)
        captured: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        handles = []

        def make_hook(name):
            def hook(module, inputs, output):
                if not inputs:
                    return
                pre = inputs[0].detach()
                post = output.detach()
                captured[name] = (pre, post)
            return hook

        for name, module in layers.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits, _ = model(idx, targets=None)
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                model.train()

        return logits.detach(), captured

    def _delta_for_layer(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        reward: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build a normalized reward-modulated Hebbian delta for one layer."""
        pre = self._flatten_activity(pre)
        post = self._flatten_activity(post)
        reward = reward.reshape(-1)

        count = min(pre.shape[0], post.shape[0], reward.shape[0])
        pre = pre[:count]
        post = post[:count]
        reward = reward[:count]
        positions = positions[positions < count]
        if positions.numel() == 0:
            return torch.zeros(
                post.shape[-1],
                pre.shape[-1],
                device=pre.device,
                dtype=pre.dtype,
            )

        pre = self._normalise_activity(pre[positions])
        post = self._normalise_activity(post[positions])
        reward = reward[positions].to(dtype=pre.dtype)

        # mean(reward * post^T @ pre) is the local Hebbian correlation.  The
        # reward is what turns plain association into gain/loss plasticity.
        weighted_post = post * reward.unsqueeze(-1)
        delta = weighted_post.transpose(0, 1) @ pre
        delta /= max(1, positions.numel())
        return delta

    @staticmethod
    def _bounded_delta(delta: torch.Tensor, weight: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Scale a candidate update to a hard Frobenius norm ceiling."""
        if not torch.isfinite(delta).all():
            raise FloatingPointError("plasticity produced a non-finite update")
        norm = float(torch.linalg.vector_norm(delta).item())
        if norm == 0.0:
            return delta
        ceiling = float(max_norm) * max(float(torch.linalg.vector_norm(weight).item()), 1.0)
        if not torch.isfinite(torch.tensor(ceiling)) or ceiling <= 0.0:
            raise FloatingPointError("plasticity update ceiling is invalid")
        if norm > ceiling:
            delta = delta * (ceiling / norm)
        return delta

    def apply(
        self,
        model: nn.Module,
        idx: torch.Tensor,
        targets: torch.Tensor,
        feedback: float = 1.0,
    ) -> PlasticityResult:
        """Apply one bounded gain/loss event to the model.

        ``feedback`` is an external teaching signal in [-1, 1].  Positive
        feedback means the observed outcome should be reinforced. Negative
        feedback reverses the gain/loss direction. A value of zero deliberately
        produces no weight change.
        """
        feedback = max(-1.0, min(1.0, float(feedback)))
        empty_snapshot: Dict[str, torch.Tensor] = {}
        if feedback == 0.0:
            return PlasticityResult(
                False, "zero_feedback", 0.0, 0.0, 0.0, 0, 0, 0, 0.0, {}, empty_snapshot
            )

        logits, captured = self._collect(model, idx, targets)
        reward, correct_fraction, mean_reward, gain_tokens, loss_tokens = self._reward_signal(
            logits, targets, feedback
        )
        positions = self._positions(reward.numel(), reward.device)
        layers = self.eligible_layers(model)

        snapshot = {
            name: module.weight.detach().clone()
            for name, module in layers.items()
        }
        layer_updates = {}
        total_norm = 0.0
        changed = 0

        try:
            with torch.no_grad():
                for name, module in layers.items():
                    activity = captured.get(name)
                    if activity is None:
                        continue
                    delta = self._delta_for_layer(*activity, reward, positions)
                    delta = delta.to(device=module.weight.device, dtype=module.weight.dtype)

                    # Homeostasis is deliberately tiny. It counteracts runaway
                    # magnitude without becoming an optimizer in disguise.
                    homeo = -float(self.config.homeostasis) * module.weight.detach()
                    delta = delta * float(self.config.learning_rate) + homeo
                    delta = self._bounded_delta(
                        delta, module.weight.detach(), self.config.max_update_norm
                    )

                    update_norm = float(torch.linalg.vector_norm(delta).item())
                    weight_norm = max(float(torch.linalg.vector_norm(module.weight).item()), 1e-12)
                    fraction = update_norm / weight_norm
                    if not torch.isfinite(delta).all():
                        raise FloatingPointError(f"non-finite delta for {name}")

                    module.weight.add_(delta)
                    if not torch.isfinite(module.weight).all():
                        raise FloatingPointError(f"non-finite weight after plasticity in {name}")

                    if fraction >= float(self.config.min_effective_fraction):
                        changed += 1
                    total_norm += update_norm
                    layer_updates[name] = {
                        "update_norm": update_norm,
                        "weight_norm": weight_norm,
                        "relative_update": fraction,
                        "gain": int((reward > 0).sum().item()),
                        "loss": int((reward < 0).sum().item()),
                    }
        except Exception:
            self.restore(model, snapshot)
            raise

        return PlasticityResult(
            True,
            "applied",
            feedback,
            mean_reward,
            correct_fraction,
            gain_tokens,
            loss_tokens,
            changed,
            total_norm,
            layer_updates,
            snapshot,
        )

    @staticmethod
    def restore(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
        """Restore the exact weights captured before a plasticity event."""
        with torch.no_grad():
            modules = dict(model.named_modules())
            for name, weight in snapshot.items():
                module = modules.get(name)
                if not isinstance(module, nn.Linear):
                    raise KeyError(f"cannot restore unknown plastic layer: {name}")
                module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))


def load_plasticity_config(values: Optional[Mapping[str, object]] = None) -> PlasticityConfig:
    """Create a config from a small mapping without silently accepting junk."""
    values = dict(values or {})
    allowed = set(PlasticityConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown plasticity settings: {', '.join(unknown)}")
    return PlasticityConfig(**values)
