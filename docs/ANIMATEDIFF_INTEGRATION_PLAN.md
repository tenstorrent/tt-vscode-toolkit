# AnimateDiff Integration with SD 3.5 on TT Hardware

**Date:** 2026-01-03
**Status:** Phase 1 & 2 Complete - Ready for Implementation
**Goal:** Enable native animated video generation on N150 (16-frame sequences at 512x512)

---

## Executive Summary

AnimateDiff adds temporal coherence to Stable Diffusion by injecting **temporal attention modules** between spatial attention and feed-forward layers. This allows generating actual animated video (not just frame-by-frame) with smooth motion.

**Key insight:** AnimateDiff doesn't modify base SD weights - it only adds temporal attention modules that learn motion patterns. Base SD 3.5 weights remain frozen.

---

## Phase 1 & 2 Findings

### AnimateDiff Architecture (SD 1.5 UNet-based)

**Components:**
- `VanillaTemporalModule`: Temporal attention wrapper
- `TemporalTransformer3DModel`: Processes 5D tensors (b, c, f, h, w)
- `VersatileAttention`: Temporal attention across frame dimension

**Key operation:**
```python
# Reshape to do attention across frames (not spatial)
hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

# Apply positional encoding for frame indices
hidden_states = pos_encoder(hidden_states)

# Standard Q, K, V attention across frame dimension
query = to_q(hidden_states)
key = to_k(hidden_states)
value = to_v(hidden_states)
attention_scores = matmul(query, key.transpose(-1, -2)) / sqrt(dim)
attention_probs = softmax(attention_scores, dim=-1)
output = matmul(attention_probs, value)

# Reshape back
output = rearrange(output, "(b d) f c -> (b f) d c", d=d)
```

**Injection pattern in AnimateDiff UNet:**
1. Spatial Self-Attention
2. **Temporal Attention** ← NEW
3. Cross-Attention (text conditioning)
4. Feed-Forward Network

**Config-driven injection:**
```python
use_motion_module = True
motion_module_resolutions = (1,2,4,8)  # Only at certain resolution levels
motion_module_type = "Vanilla"
motion_module_kwargs = {
    "num_attention_heads": 8,
    "num_transformer_block": 2,
    ...
}
```

---

### SD 3.5 Architecture (DiT Transformer-based)

**Critical difference:** SD 3.5 uses **DiT (Diffusion Transformer)** NOT UNet!

**Transformer block structure** (`fun_transformer_block.py`):
```python
def sd_transformer_block(spatial, prompt, time_embed, ...):
    # 1. Time conditioning
    t = silu(time_embed)
    spatial_time = linear(t)  # Gating parameters
    prompt_time = linear(t)

    # 2. Dual attention (joint spatial + prompt)
    spatial_normed = layer_norm(spatial, norm_1)
    prompt_normed = layer_norm(prompt, norm_1)
    spatial_attn, prompt_attn = dual_attn_block(spatial_normed, prompt_normed, ...)
    spatial += spatial_attn  # LINE 336 - RESIDUAL CONNECTION

    # ✅ PERFECT INJECTION POINT: Between LINE 336 and LINE 353

    # 3. Spatial feed-forward
    spatial_normed = layer_norm(spatial, norm_2)  # LINE 353
    spatial += gated_ff_block(spatial_normed, ...)

    # 4. Prompt feed-forward
    prompt += prompt_attn
    prompt += gated_ff_block(prompt_normed, ...)

    return spatial, prompt
```

**Injection point identified:**
- **After:** `spatial += spatial_attn` (line 336)
- **Before:** `spatial_normed = layer_norm(spatial, norm_2)` (line 353)
- **Pattern matches AnimateDiff:** Spatial attention → Temporal attention → Feed-forward

**Tensor shapes:**
```python
# Latents (pipeline.py line 394-399)
latents = (batch_size, height//8, width//8, num_channels)
# For 1024x1024: (1, 128, 128, 16)
# For 512x512: (1, 64, 64, 16)

# In transformer block (after patch embedding):
spatial = (batch*frames, spatial_tokens, channels)
# For 512x512, 16 frames: (16, 4096, 1536)
# spatial_tokens = (64//2) * (64//2) = 1024 (patch_size=2)

# For temporal attention, reshape:
# (b*f, spatial, channels) → (b*spatial, frames, channels)
# Do attention across frame dimension
# Then reshape back
```

**Key files:**
- `fun_transformer_block.py`: Transformer block with dual_attn + feed_forward
- `fun_pipeline.py`: Main pipeline, latent preparation
- `fun_attention.py`: TTNN-optimized attention implementation
- `fun_feed_forward.py`: TTNN-optimized feed-forward

---

## Implementation Strategy

### Option A: Direct Temporal Module Injection (Recommended)

**Approach:** Inject temporal attention directly into SD 3.5 transformer blocks

**Pros:**
- Minimal code changes
- Keeps SD 3.5 weights frozen
- Only temporal modules are new parameters
- Matches AnimateDiff's proven architecture

**Cons:**
- Need to port temporal attention to TTNN ops
- Need to adapt for SD 3.5's DiT (not UNet)

**Implementation:**

1. **Create `temporal_module.py`** with TTNN operations:
```python
class TtTemporalAttention:
    def __init__(self, dim, num_heads, device):
        self.to_q = ttnn_linear(dim, dim)
        self.to_k = ttnn_linear(dim, dim)
        self.to_v = ttnn_linear(dim, dim)
        self.to_out = ttnn_linear(dim, dim)
        self.pos_encoder = PositionalEncoding(dim, max_len=24)

    def __call__(self, hidden_states, num_frames):
        batch_frames, seq_len, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        # Reshape: (b*f, seq, c) → (b*seq, f, c)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, num_frames, seq_len, channels))
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1, 3))  # (b, seq, f, c)
        hidden_states = ttnn.reshape(hidden_states, (batch_size * seq_len, num_frames, channels))

        # Positional encoding for frame indices
        hidden_states = self.pos_encoder(hidden_states)

        # Standard attention across frame dimension
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        attention_scores = ttnn.matmul(query, ttnn.transpose(key, -1, -2))
        attention_scores = attention_scores / math.sqrt(channels)
        attention_probs = ttnn.softmax(attention_scores, dim=-1)
        output = ttnn.matmul(attention_probs, value)

        output = self.to_out(output)

        # Reshape back: (b*seq, f, c) → (b*f, seq, c)
        output = ttnn.reshape(output, (batch_size, seq_len, num_frames, channels))
        output = ttnn.permute(output, (0, 2, 1, 3))  # (b, f, seq, c)
        output = ttnn.reshape(output, (batch_frames, seq_len, channels))

        return output
```

2. **Modify `fun_transformer_block.py`:**
```python
def sd_transformer_block(
    spatial, prompt, time_embed, parameters,
    num_frames=1,  # NEW parameter
    temporal_module=None,  # NEW parameter
    ...):

    # ... existing dual_attn code ...
    spatial += spatial_attn  # LINE 336

    # NEW: Temporal attention injection
    if num_frames > 1 and temporal_module is not None:
        temporal_attn = temporal_module(spatial, num_frames)
        spatial += temporal_attn  # Residual connection

    # ... existing feed_forward code ...
    spatial_normed = layer_norm(spatial, norm_2)  # LINE 353
    spatial += gated_ff_block(spatial_normed, ...)

    return spatial, prompt
```

3. **Update pipeline to support multi-frame:**
```python
def __call__(self, prompt, num_frames=1, ...):
    # Prepare latents for multiple frames
    if num_frames > 1:
        latents = torch.randn(
            (batch_size, num_frames, height//8, width//8, channels)
        )
        # Reshape to (batch*frames, height//8, width//8, channels)
        latents = latents.reshape(batch_size * num_frames, height//8, width//8, channels)
    else:
        latents = torch.randn((batch_size, height//8, width//8, channels))

    # ... denoising loop ...
    for timestep in timesteps:
        noise_pred = transformer(
            latents,
            prompt_embeds,
            timestep,
            num_frames=num_frames,  # Pass through
            temporal_module=self.temporal_module if num_frames > 1 else None
        )
        latents = scheduler.step(noise_pred, timestep, latents)

    # Decode frames
    if num_frames > 1:
        # Decode each frame separately or in batches
        frames = []
        for i in range(num_frames):
            frame_latent = latents[i::num_frames]  # Extract frame i
            frame = vae_decode(frame_latent)
            frames.append(frame)
        return frames
    else:
        return vae_decode(latents)
```

4. **Load AnimateDiff motion module weights:**
```python
def load_temporal_weights(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path)  # mm_sd_v15_v2.ckpt

    # Extract temporal module weights
    temporal_weights = {}
    for key, value in ckpt.items():
        if 'temporal' in key or 'motion' in key:
            # Convert to TTNN
            temporal_weights[key] = ttnn.from_torch(value, device=device)

    return temporal_weights
```

---

## Next Steps (Phase 3: Implementation)

### Priority 1: Port Temporal Attention to TTNN

**Tasks:**
1. Create `temporal_module.py` with TTNN operations
2. Implement `TtTemporalAttention` class
3. Map PyTorch ops to TTNN equivalents:
   - `torch.nn.Linear` → `ttnn.linear`
   - `torch.matmul` → `ttnn.matmul`
   - `torch.softmax` → `ttnn.softmax`
   - Manual tensor reshaping with `ttnn.reshape`, `ttnn.permute`
4. Load AnimateDiff weights and convert to TTNN format

### Priority 2: Inject into SD 3.5 Transformer

**Tasks:**
1. Modify `fun_transformer_block.py` to accept `num_frames` and `temporal_module`
2. Add temporal attention call after line 336
3. Test with single frame first (should be no-op)
4. Test with 2 frames (minimum viable animation)

### Priority 3: Update Pipeline

**Tasks:**
1. Modify `fun_pipeline.py` to accept `num_frames` parameter
2. Expand latent generation for multi-frame
3. Update denoising loop to pass `num_frames` through
4. Implement frame-by-frame VAE decoding

### Priority 4: Test and Optimize

**Tasks:**
1. Test 2-frame sequence (baseline functionality)
2. Test 16-frame sequence (full animation)
3. Optimize memory usage (batch processing if needed)
4. Optimize performance (TTNN acceleration)

---

## Expected Results

### Minimum Success
- ✅ 2-frame sequence with temporal coherence
- ✅ Visual motion between frames
- ✅ Runs on N150 without crashes

### Full Success
- ✅ 16-frame animated sequences at 512x512
- ✅ Smooth motion (butterfly actually lands, wings move)
- ✅ ~2-3 minutes per 16-frame sequence on N150
- ✅ Export to MP4 video

---

## Technical Challenges

### Challenge 1: Tensor Shape Mismatches
**Problem:** SD 3.5 uses different tensor shapes than AnimateDiff (DiT vs UNet)
**Solution:** Adapt reshaping logic for SD 3.5's `(batch*frames, spatial_tokens, channels)` format

### Challenge 2: TTNN Operation Coverage
**Problem:** Some PyTorch ops may not have TTNN equivalents
**Solution:** Start with CPU fallback for temporal layers, gradually port to TTNN

### Challenge 3: Memory Constraints
**Problem:** 16 frames @ 512x512 may exceed N150 DRAM
**Solution:** Batch processing (4 frames at a time), lower resolution if needed

### Challenge 4: Weight Compatibility
**Problem:** AnimateDiff trained for SD 1.5, we're using SD 3.5
**Solution:** May need fine-tuning or finding SD 3.5-specific motion module

---

## Resources

**Files:**
- Motion module: `~/models/animatediff/mm_sd_v15_v2.ckpt` (1.7GB)
- AnimateDiff reference: `~/vendor/AnimateDiff/`
- SD 3.5 code: `~/tt-metal/models/experimental/stable_diffusion_35_large/tt/`

**Key references:**
- AnimateDiff paper: https://arxiv.org/abs/2307.04725
- AnimateDiff repo: https://github.com/guoyww/AnimateDiff
- SD 3.5 model: stabilityai/stable-diffusion-3.5-large

---

**Status:** Ready to begin Phase 3 (TTNN implementation)
