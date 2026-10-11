"""Jupiter's seed neural model.

This is intentionally a small causal transformer. It is a starting organism,
not a claim that Jupiter must always be a transformer. The model config is
stored with every checkpoint so later experiments can change the architecture.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        if c.n_embd % c.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = c.n_head
        self.head_dim = c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.dropout = c.dropout
        self.resid_dropout = nn.Dropout(c.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(c.block_size, c.block_size)).view(
                1, 1, c.block_size, c.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        b, t, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            y = weights @ v
        y = y.transpose(1, 2).contiguous().view(b, t, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        hidden = 4 * c.n_embd
        self.fc = nn.Linear(c.n_embd, hidden, bias=c.bias)
        self.proj = nn.Linear(hidden, c.n_embd, bias=c.bias)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Jupiter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for module in self.modules():
            if (
                isinstance(module, nn.Linear)
                and module is not self.lm_head
                and module.out_features == config.n_embd
            ):
                nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        if t > self.config.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(t, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        """Continue from `idx` and return the extended sequence.

        At least one token of context is required. The next token is sampled
        from the last position of the forward pass, so a zero-length sequence
        has no position to read. That used to surface as an IndexError about
        dimension sizes, which names the symptom and not the mistake, and the
        caller that produced it was a long way from the message. It is refused
        here instead, where the reason can be stated.

        The training flag is restored in a finally rather than after the loop,
        so a failure part way through cannot leave the model silently switched
        to eval for whatever runs next.
        """
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError(
                "generate needs at least one token of context to continue "
                f"from, but was given a sequence of {tuple(idx.shape)}. A model "
                "samples the next token from the last token it was given, so an "
                "empty context has nothing to continue from."
            )
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = idx[:, -self.config.block_size :]
                logits, _ = self(context)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            if was_training:
                self.train()
        return idx

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(self, learning_rate, weight_decay):
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.ndim >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {}
        if (
            self.parameters
            and self.lm_head.weight.device.type == "cuda"
            and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        ):
            kwargs["fused"] = True
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), **kwargs)
