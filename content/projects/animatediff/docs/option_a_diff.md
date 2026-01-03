# Option A: Minimal tt-metal Modification (Optional Performance Enhancement)

This document describes an optional modification to `tt-metal` that provides a cleaner integration with AnimateDiff. **This modification is NOT required** - the standalone wrapper in `examples/generate_with_sd35.py` works without it.

## Why This Might Be Useful

The standalone wrapper (Option C) works by running the full pipeline and extracting latents before VAE decode. Option A provides a more elegant API by adding a `return_latents` parameter to the SD 3.5 pipeline.

**Benefits:**
- Cleaner code (no need to duplicate denoising logic)
- Better performance (avoids unnecessary computation)
- Useful for the community (could be submitted as a PR to tt-metal)

**Tradeoff:**
- Requires modifying `tt-metal` codebase
- Couples AnimateDiff implementation to tt-metal version

## The Modification

Add a `return_latents` parameter to `TtStableDiffusion3Pipeline.__call__()` that returns latents before VAE decode.

### File to Modify

`~/tt-metal/models/experimental/stable_diffusion_35_large/tt/fun_pipeline.py`

### Diff

```diff
--- a/fun_pipeline.py
+++ b/fun_pipeline.py
@@ -360,7 +360,8 @@ class TtStableDiffusion3Pipeline:
         num_images_per_prompt: int = 1,
         max_t5_sequence_length: int = 333,
         clip_skip: int = 0,
-        traced: bool = False,
+        traced: bool = False,
+        return_latents: bool = False,
     ) -> list[Image.Image] | torch.Tensor:
         """
         Generate images from text prompts using the Stable Diffusion 3.5 pipeline.
@@ -380,6 +381,7 @@ class TtStableDiffusion3Pipeline:
             max_t5_sequence_length: Maximum sequence length for T5 encoder
             clip_skip: Number of CLIP layers to skip
             traced: Whether to use Metal Trace for performance
+            return_latents: If True, return latents before VAE decode (for AnimateDiff)

         Returns:
             List of PIL images
@@ -555,6 +557,10 @@ class TtStableDiffusion3Pipeline:

                 torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
                 torch_latents = (torch_latents / self._torch_vae_scaling_factor) + self._torch_vae_shift_factor
+
+                # Return latents if requested (for AnimateDiff temporal attention)
+                if return_latents:
+                    return torch_latents

                 # HACK: reshape submesh device 0 from 2D to 1D
                 original_vae_device_shape = tuple(self.vae_parallel_manager.device.shape)
```

### Lines Changed

**Total: 3 lines added**

1. Line 364: Add `return_latents: bool = False` parameter
2. Line 365: Update return type hint to `list[Image.Image] | torch.Tensor`
3. Lines 560-562: Add early return if `return_latents=True`

### Backward Compatibility

✅ Fully backward compatible - existing code continues to work unchanged:
- Default `return_latents=False` behaves exactly as before
- Return type is union `list[Image.Image] | torch.Tensor`
- No breaking changes to existing functionality

## Usage with AnimateDiff

If you apply this modification, you can simplify the AnimateDiff wrapper:

```python
# Generate latents for each frame (with modification)
latents_list = []
for frame_idx in range(num_frames):
    frame_latents = self.sd35(
        prompt_1=[prompt],
        prompt_2=[prompt],
        prompt_3=[prompt],
        negative_prompt_1=[""],
        negative_prompt_2=[""],
        negative_prompt_3=[""],
        num_inference_steps=num_inference_steps,
        seed=base_seed + frame_idx,
        return_latents=True,  # ← Get latents instead of images
    )
    latents_list.append(frame_latents)

# Apply temporal attention
latents_stacked = torch.stack(latents_list, dim=1)
latents_coherent = animatediff.apply_temporal_coherence(latents_stacked, num_frames)

# Decode frames
frames = []
for frame_idx in range(num_frames):
    # Now we need to decode each frame
    # This requires calling VAE decoder directly (see below)
    frame_image = decode_with_vae(latents_coherent[0, frame_idx])
    frames.append(frame_image)
```

**Note:** You'd also need to expose the VAE decoder as a separate method for the decode step.

## Alternative: Keep It Standalone

**Recommended approach:** Use the standalone wrapper (Option C) unless you're actively contributing to tt-metal development.

**Reasons:**
1. No coupling to tt-metal versions
2. Works immediately without modifications
3. Easier to maintain independently
4. Clear separation of concerns

The standalone wrapper in `examples/generate_with_sd35.py` implements the full denoising loop manually, giving complete control without requiring tt-metal modifications.

## Submitting to tt-metal (Optional)

If you'd like to contribute this enhancement to the tt-metal community:

1. Test thoroughly on all hardware (N150, N300, T3K, P100)
2. Add unit tests for the new parameter
3. Update documentation
4. Submit PR with use case explanation (AnimateDiff integration)

This would be a useful addition for anyone working with temporal models or video generation!
