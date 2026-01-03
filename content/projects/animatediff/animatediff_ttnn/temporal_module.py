# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Temporal Attention Module for AnimateDiff Integration

This module adds temporal coherence across video frames by applying temporal
attention after spatial attention operations. Based on AnimateDiff architecture
(https://arxiv.org/abs/2307.04725).

Key Operation:
    - Spatial attention: Attention within each frame (handled by SD 3.5)
    - Temporal attention: Attention across frames (this module)

The temporal attention operates on reshaped tensors where the frame dimension
becomes the sequence dimension for cross-frame attention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import ttnn


@dataclass
class TemporalAttentionWeights:
    """Temporal attention weights loaded from AnimateDiff checkpoint.

    Attributes:
        to_q_weight: Query projection weight
        to_q_bias: Query projection bias (optional)
        to_k_weight: Key projection weight
        to_k_bias: Key projection bias (optional)
        to_v_weight: Value projection weight
        to_v_bias: Value projection bias (optional)
        to_out_weight: Output projection weight
        to_out_bias: Output projection bias (optional)
        pos_encoding: Positional encoding for frame indices [1, max_frames, dim]
    """
    to_q_weight: torch.Tensor
    to_q_bias: Optional[torch.Tensor]
    to_k_weight: torch.Tensor
    to_k_bias: Optional[torch.Tensor]
    to_v_weight: torch.Tensor
    to_v_bias: Optional[torch.Tensor]
    to_out_weight: torch.Tensor
    to_out_bias: Optional[torch.Tensor]
    pos_encoding: Optional[torch.Tensor]

    dim: int
    num_heads: int


def create_sinusoidal_positional_encoding(
    dim: int,
    max_len: int = 24,
) -> torch.Tensor:
    """Create sinusoidal positional encoding for frame indices.

    Args:
        dim: Embedding dimension
        max_len: Maximum sequence length (number of frames)

    Returns:
        Positional encoding tensor of shape [1, max_len, dim]
    """
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

    pe = torch.zeros(1, max_len, dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe


def temporal_attention_torch(
    hidden_states: torch.Tensor,
    weights: TemporalAttentionWeights,
    num_frames: int,
) -> torch.Tensor:
    """Apply temporal attention across video frames (PyTorch version for testing).

    Args:
        hidden_states: Input tensor of shape (batch*frames, spatial_tokens, channels)
        weights: Temporal attention weights
        num_frames: Number of frames in the video sequence

    Returns:
        Output tensor of same shape as input
    """
    batch_frames, seq_len, channels = hidden_states.shape

    if num_frames == 1:
        return hidden_states

    batch_size = batch_frames // num_frames
    num_heads = weights.num_heads
    head_dim = channels // num_heads

    # Reshape: (b*f, spatial, c) → (b*spatial, f, c)
    hidden_states = hidden_states.view(batch_size, num_frames, seq_len, channels)
    hidden_states = hidden_states.permute(0, 2, 1, 3)  # (b, spatial, f, c)
    hidden_states = hidden_states.reshape(batch_size * seq_len, num_frames, channels)

    # Add positional encoding
    if weights.pos_encoding is not None:
        pos_enc = weights.pos_encoding[:, :num_frames, :].to(hidden_states.device)
        hidden_states = hidden_states + pos_enc

    # Q, K, V projections
    query = torch.nn.functional.linear(hidden_states, weights.to_q_weight, weights.to_q_bias)
    key = torch.nn.functional.linear(hidden_states, weights.to_k_weight, weights.to_k_bias)
    value = torch.nn.functional.linear(hidden_states, weights.to_v_weight, weights.to_v_bias)

    # Reshape for multi-head attention
    query = query.view(batch_size * seq_len, num_frames, num_heads, head_dim)
    query = query.permute(0, 2, 1, 3)  # (b*spatial, heads, frames, head_dim)

    key = key.view(batch_size * seq_len, num_frames, num_heads, head_dim)
    key = key.permute(0, 2, 1, 3)

    value = value.view(batch_size * seq_len, num_frames, num_heads, head_dim)
    value = value.permute(0, 2, 1, 3)

    # Scaled dot-product attention
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_probs, value)

    # Concatenate heads
    attention_output = attention_output.permute(0, 2, 1, 3)
    attention_output = attention_output.reshape(batch_size * seq_len, num_frames, channels)

    # Output projection
    output = torch.nn.functional.linear(attention_output, weights.to_out_weight, weights.to_out_bias)

    # Reshape back: (b*spatial, f, c) → (b*f, spatial, c)
    output = output.view(batch_size, seq_len, num_frames, channels)
    output = output.permute(0, 2, 1, 3)  # (b, f, spatial, c)
    output = output.reshape(batch_frames, seq_len, channels)

    return output


def temporal_attention_ttnn(
    hidden_states: ttnn.Tensor,
    weights: TemporalAttentionWeights,
    num_frames: int,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """Apply temporal attention across video frames (TTNN version).

    Args:
        hidden_states: Input tensor of shape (batch*frames, spatial_tokens, channels)
        weights: Temporal attention weights
        num_frames: Number of frames in the video sequence
        device: TTNN device

    Returns:
        Output tensor of same shape as input
    """
    # Convert to torch for processing, then back to ttnn
    # This is a simplified version - full TTNN version would use all TTNN ops

    hidden_states_torch = ttnn.to_torch(hidden_states)
    output_torch = temporal_attention_torch(hidden_states_torch, weights, num_frames)

    output_ttnn = ttnn.from_torch(
        output_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return output_ttnn


def load_animatediff_weights(
    checkpoint_path: str,
    dim: int = 320,  # SD 1.5 default
    num_heads: int = 8,
    max_frames: int = 24,
    block_index: int = 0,
) -> TemporalAttentionWeights:
    """Load AnimateDiff temporal module weights from checkpoint.

    Args:
        checkpoint_path: Path to AnimateDiff checkpoint (.ckpt file)
        dim: Hidden dimension
        num_heads: Number of attention heads
        max_frames: Maximum number of frames for positional encoding
        block_index: Which transformer block to load weights for

    Returns:
        TemporalAttentionWeights instance with loaded weights

    Example:
        >>> weights = load_animatediff_weights(
        ...     "~/models/animatediff/mm_sd_v15_v2.ckpt",
        ...     dim=320,
        ...     num_heads=8,
        ... )
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # AnimateDiff checkpoint structure:
    # down_blocks.{i}.motion_modules.{j}.temporal_transformer.transformer_blocks.{k}.{layer}
    prefix = f"down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.{block_index}"

    # Extract attention weights
    def get_weight(key: str) -> Optional[torch.Tensor]:
        full_key = f"{prefix}.attention_blocks.0.{key}"
        return checkpoint.get(full_key)

    to_q_weight = get_weight("to_q.weight")
    to_q_bias = get_weight("to_q.bias")
    to_k_weight = get_weight("to_k.weight")
    to_k_bias = get_weight("to_k.bias")
    to_v_weight = get_weight("to_v.weight")
    to_v_bias = get_weight("to_v.bias")
    to_out_weight = get_weight("to_out.0.weight")
    to_out_bias = get_weight("to_out.0.bias")

    if to_q_weight is None:
        raise ValueError(f"Could not find weights in checkpoint at prefix: {prefix}")

    # Create positional encoding
    pos_encoding = create_sinusoidal_positional_encoding(dim, max_frames)

    return TemporalAttentionWeights(
        to_q_weight=to_q_weight,
        to_q_bias=to_q_bias,
        to_k_weight=to_k_weight,
        to_k_bias=to_k_bias,
        to_v_weight=to_v_weight,
        to_v_bias=to_v_bias,
        to_out_weight=to_out_weight,
        to_out_bias=to_out_bias,
        pos_encoding=pos_encoding,
        dim=dim,
        num_heads=num_heads,
    )
