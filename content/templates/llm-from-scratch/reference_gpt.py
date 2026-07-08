# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# The "understand" half of Labs 1-4: a minimal, readable, pure-PyTorch nano
# LLM built from the MODERN Llama-3 component set.
#
# NO Tenstorrent dependencies. Runs on CPU anywhere. This is the reference the
# labs read alongside the TT-native expression (ttnn / TT-Lang kernels) so a
# reader can see the same math twice: once in plain PyTorch, once TT-native.
#
# The architecture and config mirror the ttml `nanollama3` model the training
# lab actually runs on Blackhole:
#   * tt-train/configs/model_configs/nanollama3_char.yaml
#       model_type=llama, embedding_dim 384, num_heads 6, num_groups 3 (GQA),
#       num_blocks 6, max_sequence_length 256, theta 500000, char tokenizer.
#   * tt-train/sources/ttml/ttml/models/llama/{transformer,gqattn}.py
#       RMSNorm pre-norm, RoPE on Q/K, Grouped-Query Attention, SwiGLU MLP.
#
# The four Llama-3 components this file builds from scratch (and where the arc
# teaches each TT-native):
#   * RoPE   (rotary position embeddings) — replaces learned positional
#            embeddings; precompute cos/sin from `theta`, apply rotate_half to
#            Q and K. TT-native: kernels/rope.py + ttml.ops.rope. (Lab 2)
#   * RMSNorm (pre-norm) — x / sqrt(mean(x^2)+eps) * weight. TT-native:
#            kernels/rmsnorm.py + ttml.ops.rmsnorm. (Lab 4)
#   * GQA    (grouped-query attention) — num_kv_groups (3) < num_heads (6): KV
#            heads are computed once per group and shared across the query heads
#            in that group (cheaper KV cache, same quality). (Lab 3)
#   * SwiGLU MLP — down(silu(gate(x)) * up(x)), three projections. Replaces the
#            GELU MLP. TT-native: PyTorch ref here + ttml.ops.swiglu. (Lab 4)
#
# This is a from-scratch mirror of `ttml.models.llama`, not a copy — kept
# deliberately small and comment-heavy. (GPT-2 with learned positional
# embeddings + a GELU MLP is the historical contrast the arc footnotes; this
# file is the modern path the hero run trains.)
# ============================================================================
"""Pure-PyTorch nano Llama-3 reference model (RoPE + RMSNorm + GQA + SwiGLU).

Smoke test (builds the model, one forward on random tokens):

    python content/templates/llm-from-scratch/reference_gpt.py --smoke

Expected: prints logits shape [batch, seq, vocab], param count, and exits 0.
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
# no hard version requirement. ttml's RMSNormLayer uses eps=1e-5.
if hasattr(nn, "RMSNorm"):

    def make_rmsnorm(dim: int, eps: float = 1e-5) -> nn.Module:
        return nn.RMSNorm(dim, eps=eps)

else:

    class _RMSNorm(nn.Module):
        """out = x / sqrt(mean(x^2, dim=-1) + eps) * weight

        No mean-subtraction and no bias (the classic RMSNorm formulation) —
        just a learned per-feature rescale of the root-mean-square norm. This
        is what modern LLMs (Llama-3) use in place of LayerNorm.
        """

        def __init__(self, normalized_shape: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(normalized_shape))

        def forward(self, x):
            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * rms * self.weight

    def make_rmsnorm(dim: int, eps: float = 1e-5) -> nn.Module:
        return _RMSNorm(dim, eps=eps)


# --- Nano Llama-3 config (matches nanollama3_char.yaml) ----------------------
@dataclass
class LlamaConfig:
    vocab_size: int = 96        # char-level Shakespeare rounds to ~96
    block_size: int = 256       # max_sequence_length
    n_embd: int = 384           # embedding_dim / residual-stream width
    n_head: int = 6             # attention (query) heads
    n_kv_groups: int = 3        # KV groups for GQA (num_groups); must divide n_head
    n_layer: int = 6            # num_blocks (transformer blocks)
    rope_theta: float = 500000.0  # RoPE base frequency (theta)
    dropout: float = 0.0        # nanollama3 uses dropout_prob 0.0
    norm_eps: float = 1e-5      # RMSNorm epsilon (ttml RMSNormLayer default)


def compute_swiglu_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """Llama SwiGLU intermediate size (mirrors ttml's helper of the same name).

    SwiGLU has three matrices (gate/up/down) vs two in a standard MLP, so the
    intermediate is scaled to 2/3 of 4*hidden = 8/3*hidden and rounded up to
    `multiple_of` for tile alignment. For hidden=384 this yields 1024.
    """
    unrounded = (4 * hidden_size * 2) // 3
    return ((unrounded + multiple_of - 1) // multiple_of) * multiple_of


class CharTokenizer:
    """The simplest possible tokenizer: one integer per character.

    Mirrors ttml.common.data.CharTokenizer. Production Llama / Mini-LLM use a
    SentencePiece BPE 32K vocab (Lab 1 frames it), but the verified nano hero
    run uses char-level tokenization — so that is what this reference keeps.
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


# --- RoPE: rotary position embeddings ---------------------------------------
def precompute_rope_cos_sin(head_dim: int, max_seq: int, theta: float):
    """Precompute the cos/sin tables RoPE rotates Q and K with.

    RoPE encodes position by *rotating* pairs of features by an angle that
    grows with position and shrinks with feature index — so relative position
    falls out of the Q·K dot product, with no learned positional table.

    inv_freq[i] = theta^(-2i/head_dim) for i in [0, head_dim/2).
    angle(pos, i) = pos * inv_freq[i]; we duplicate the half-width table so it
    lines up with rotate_half's [-x2, x1] convention (the Llama/HF layout).

    Returns (cos, sin), each shaped [max_seq, head_dim].
    """
    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()
    freqs = torch.outer(positions, inv_freq)      # [max_seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)       # [max_seq, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(-x2, x1): rotate the two halves of the last dim by 90 degrees."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x shaped [B, n_head, T, head_dim].

    x_rot = x * cos + rotate_half(x) * sin — the standard rotary formula. cos
    and sin are [T, head_dim] and broadcast over batch and head dims.
    """
    T = x.shape[-2]
    cos = cos[:T].to(x.dtype)          # [T, head_dim] -> broadcasts to [1,1,T,hd]
    sin = sin[:T].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    """Causal grouped-query attention with RoPE — the heart of Lab 3.

    GQA: `n_head` query heads share `n_kv_groups` key/value heads (each KV head
    serves `n_head // n_kv_groups` query heads). This shrinks the KV cache and
    KV projections while keeping full query resolution — the modern inference
    win over vanilla multi-head attention.

    Pedagogy note: ttml's GroupedQueryAttention fuses K and V into one
    `kv_linear` (concat_kv_dim) and creates heads with
    `ttml.ops.multi_head_utils.grouped_heads_creation`; here we keep separate
    q/k/v projections for readability. The math is identical.
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_groups == 0, (
            "n_head must be divisible by n_kv_groups"
        )
        self.n_head = cfg.n_head
        self.n_kv_groups = cfg.n_kv_groups
        self.head_dim = cfg.n_embd // cfg.n_head
        self.group_size = cfg.n_head // cfg.n_kv_groups  # query heads per KV head

        # Llama linears carry no bias. Q is full width; K/V are group-width.
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular causal mask (no token attends to the future).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        # Project, then split into heads. Q has n_head heads; K/V have
        # n_kv_groups heads (that is the whole GQA trick).
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # RoPE rotates Q and K (never V) — position lives in the Q·K product.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Share each KV head across its group of query heads. repeat_interleave
        # (not repeat) keeps heads of the same group adjacent, matching how the
        # query heads are laid out. (B, n_kv_groups, T, hd) -> (B, n_head, T, hd)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = weights @ v                        # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(out))


class SwiGLU(nn.Module):
    """Llama SwiGLU feed-forward: down(silu(gate(x)) * up(x)) — Lab 4.

    Three projections (no bias): `gate` (w1) and `up` (w3) both map n_embd ->
    intermediate; `down` (w2) maps back. SiLU-gates the up-projection. This
    replaces GPT-2's expand-4x + GELU MLP.
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        inter = compute_swiglu_intermediate_size(cfg.n_embd)
        self.gate = nn.Linear(cfg.n_embd, inter, bias=False)  # w1
        self.up = nn.Linear(cfg.n_embd, inter, bias=False)    # w3
        self.down = nn.Linear(inter, cfg.n_embd, bias=False)  # w2
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """A pre-norm Llama transformer block: RMSNorm -> GQA -> +residual,
    RMSNorm -> SwiGLU -> +residual (mirrors ttml's LlamaBlock).
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attention_norm = make_rmsnorm(cfg.n_embd, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = make_rmsnorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attention_norm(x), cos, sin)  # residual around attention
        x = x + self.mlp(self.mlp_norm(x))                   # residual around MLP
        return x


class NanoLlama(nn.Module):
    """The whole model: token embeddings -> Llama blocks -> RMSNorm -> LM head.

    No learned positional embedding table — position enters only through RoPE
    inside attention. cos/sin are precomputed once and threaded to every block.
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_rmsnorm(cfg.n_embd, cfg.norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # RoPE tables, sized to the max sequence length. head_dim = n_embd/n_head.
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope_cos_sin(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        x = self.drop(self.tok_emb(idx))          # residual stream (no pos-emb add)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)
        logits = self.head(x)                      # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Backwards-compatible aliases: earlier drafts of the arc used GPT-2 names.
GPTConfig = LlamaConfig
NanoGPT = NanoLlama


def smoke() -> int:
    torch.manual_seed(0)
    cfg = LlamaConfig()
    model = NanoLlama(cfg)
    model.eval()

    batch, seq = 2, 32
    idx = torch.randint(0, cfg.vocab_size, (batch, seq))
    targets = torch.randint(0, cfg.vocab_size, (batch, seq))
    with torch.no_grad():
        logits, loss = model(idx, targets)

    print("model: nano Llama-3 (RoPE + RMSNorm + GQA + SwiGLU), mirrors ttml nanollama3")
    print(f"config: n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"n_kv_groups={cfg.n_kv_groups} (GQA) n_embd={cfg.n_embd} "
          f"vocab={cfg.vocab_size} rope_theta={cfg.rope_theta:g}")
    print(f"swiglu intermediate: {compute_swiglu_intermediate_size(cfg.n_embd)}")
    print(f"parameters: {model.num_params():,}")
    print(f"logits shape: {list(logits.shape)}  (expected [{batch}, {seq}, {cfg.vocab_size}])")
    print(f"initial loss: {loss.item():.4f}  (~ln(vocab) = {math.log(cfg.vocab_size):.4f})")
    assert list(logits.shape) == [batch, seq, cfg.vocab_size]
    print("PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pure-PyTorch nano Llama-3 reference (RoPE/RMSNorm/GQA/SwiGLU)."
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Build the model and run one forward pass on random tokens.")
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
