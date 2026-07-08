# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# The "understand" half of Labs 1-4: a minimal, readable, pure-PyTorch nano GPT.
#
# NO Tenstorrent dependencies. Runs on CPU anywhere. This is the reference the
# labs read alongside the TT-native expression (ttnn / TT-Lang kernels) so a
# reader can see the same math twice: once in plain PyTorch, once TT-native.
#
# The architecture and config mirror the ttml NanoGPT the training lab actually
# runs (tt-train/configs/model_configs/nanogpt.yaml): token + positional
# embeddings, a stack of pre-norm transformer blocks (multi-head self-attention
# + MLP), a final norm and an LM head. Kept deliberately small and comment-heavy.
#
# Normalization: RMSNorm, not LayerNorm — matching Lab 4's rmsnorm.py kernel
# and the ttml model, so the "same math twice" pedagogy (PyTorch here, TT-Lang
# / ttnn kernel there) actually holds for norm as well as attention.
# ============================================================================
"""Pure-PyTorch nano GPT reference model.

Smoke test (builds the model, one forward on random tokens):

    python content/templates/llm-from-scratch/reference_gpt.py --smoke

Expected: prints logits shape [batch, seq, vocab] and exits 0.
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.nn.RMSNorm landed in PyTorch 2.4. Fall back to a tiny pure-PyTorch
# implementation (matching Lab 4's rmsnorm.py: x / sqrt(mean(x^2) + eps),
# scaled by a learned per-feature weight) on older installs so this file has
# no hard version requirement.
if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:

    class RMSNorm(nn.Module):
        """out = x / sqrt(mean(x^2, dim=-1) + eps) * weight

        No mean-subtraction and no bias (the classic RMSNorm formulation) —
        just a learned per-feature rescale of the root-mean-square norm.
        """

        def __init__(self, normalized_shape: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(normalized_shape))

        def forward(self, x):
            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * rms * self.weight


# --- Nano baseline config (matches tt-train/configs/model_configs/nanogpt.yaml)
@dataclass
class GPTConfig:
    vocab_size: int = 96      # char-level Shakespeare rounds to ~96; GPT-2 uses 50257
    block_size: int = 256     # max sequence length
    n_embd: int = 384         # embedding / residual-stream width
    n_head: int = 6           # attention heads (n_embd must be divisible by n_head)
    n_layer: int = 6          # transformer blocks
    dropout: float = 0.2
    bias: bool = True


class CharTokenizer:
    """The simplest possible tokenizer: one integer per character.

    Mirrors ttml.common.data.CharTokenizer. A real LLM uses BPE (Lab 1 builds
    one), but char-level is enough to make the data pipeline concrete.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention: the heart of Lab 3.

    Q, K, V come from a single fused linear projection, are split into heads,
    then for each head:  softmax(Q Kᵀ / sqrt(d) + causal_mask) V.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # One matmul produces Q, K, V (3 * n_embd) — cheaper than three.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask (no token attends to the future).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = weights @ v                       # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, GELU, project back (Lab 4)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """A pre-norm transformer block: the residual stream threads through both
    the attention and the MLP sub-layers (Lab 2's residual stream, Lab 4's block).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual add around attention
        x = x + self.mlp(self.ln2(x))    # residual add around MLP
        return x


class NanoGPT(nn.Module):
    """The whole model: embeddings -> blocks -> norm -> LM head."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # residual stream
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                  # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def smoke() -> int:
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = NanoGPT(cfg)
    model.eval()

    batch, seq = 2, 32
    idx = torch.randint(0, cfg.vocab_size, (batch, seq))
    targets = torch.randint(0, cfg.vocab_size, (batch, seq))
    with torch.no_grad():
        logits, loss = model(idx, targets)

    print(f"config: n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"n_embd={cfg.n_embd} vocab={cfg.vocab_size}")
    print(f"parameters: {model.num_params():,}")
    print(f"logits shape: {list(logits.shape)}  (expected [{batch}, {seq}, {cfg.vocab_size}])")
    print(f"initial loss: {loss.item():.4f}  (~ln(vocab) = {math.log(cfg.vocab_size):.4f})")
    assert list(logits.shape) == [batch, seq, cfg.vocab_size]
    print("PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure-PyTorch nano GPT reference.")
    parser.add_argument("--smoke", action="store_true",
                        help="Build the model and run one forward pass on random tokens.")
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
