"""
Nano-Trickster: A tiny transformer architecture (10-20M parameters)

This is a minimal but complete transformer implementation designed for:
- Training from scratch on N150 (30-60 minutes)
- Understanding transformer fundamentals
- Character-level language modeling

Architecture:
- vocab_size: 256 (character-level, simple!)
- hidden_dim: 256 (small but workable)
- num_layers: 6 (shallow, fast to train)
- num_heads: 8 (decent parallelism)
- mlp_dim: 768 (3× hidden_dim)
- max_seq_len: 512 (short context)
- Total params: ~11M

Based on: nanoGPT (Karpathy) + TinyLlama patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization: x / rms(x)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - better than learned positions"""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute theta for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute sin/cos for all positions
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE"""

    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask (autoregressive)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out


class SwiGLU(nn.Module):
    """SwiGLU activation (better than ReLU for transformers)"""

    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, mlp_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer block: Attention + FFN with pre-norm and residuals"""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        # Pre-norm for attention
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, max_seq_len, dropout)

        # Pre-norm for FFN
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention block with residual
        x = x + self.attn(self.attn_norm(x), mask)

        # FFN block with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class NanoTrickster(nn.Module):
    """
    Nano-Trickster: Tiny transformer for training from scratch

    Architecture:
    - Character-level tokenization (vocab_size=256)
    - 6 transformer layers
    - 256 hidden dimensions
    - 8 attention heads
    - 768 MLP dimensions (3× hidden)
    - Total: ~11M parameters
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 768,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embeddings (no learned positional - RoPE handles it)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(hidden_dim)

        # Output projection (weight-tied with embedding)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.output_proj.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Nano-Trickster initialized: {trainable_params:,} trainable params ({total_params:,} total)")

    def _init_weights(self, module):
        """Initialize weights with small values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_emb(input_ids)

        # Create causal mask for autoregressive generation
        mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).view(1, 1, seq_len, seq_len)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization
        x = self.final_norm(x)

        # Output projection (logits)
        logits = self.output_proj(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively"""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context if needed
            context = prompt if prompt.size(1) <= self.max_seq_len else prompt[:, -self.max_seq_len:]

            # Forward pass
            outputs = self(context)
            logits = outputs["logits"]

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            prompt = torch.cat([prompt, next_token], dim=1)

        return prompt


def count_parameters(model: nn.Module) -> dict:
    """Count parameters by component"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Breakdown by component
    embedding = sum(p.numel() for p in model.token_emb.parameters())
    blocks = sum(p.numel() for p in model.blocks.parameters())
    output = model.output_proj.weight.numel()  # Weight-tied, so same as embedding

    return {
        "total": total,
        "trainable": trainable,
        "embedding": embedding,
        "transformer_blocks": blocks,
        "output_layer": output,
        "per_block": blocks // len(model.blocks) if model.blocks else 0,
    }


if __name__ == "__main__":
    # Create model
    model = NanoTrickster(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_dim=768,
        max_seq_len=512,
    )

    # Count parameters
    params = count_parameters(model)
    print("\nParameter breakdown:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Embedding: {params['embedding']:,}")
    print(f"  Transformer blocks: {params['transformer_blocks']:,}")
    print(f"  Per block: {params['per_block']:,}")
    print(f"  Output layer: {params['output_layer']:,} (weight-tied)")

    # Test forward pass
    batch_size, seq_len = 4, 64
    input_ids = torch.randint(0, 256, (batch_size, seq_len))

    outputs = model(input_ids, labels=input_ids)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Test generation
    prompt = torch.randint(0, 256, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\nTest generation:")
    print(f"  Prompt shape: {prompt.shape}")
    print(f"  Generated shape: {generated.shape}")
