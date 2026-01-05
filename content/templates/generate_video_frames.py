#!/usr/bin/env python3
"""
Video Frame Generator with Stable Diffusion 3.5
Generates sequential frames for video creation on Tenstorrent hardware
"""
import sys
import os
from pathlib import Path

# Setup tt-metal Python environment
tt_metal_home = Path.home() / "tt-metal"
sys.path.insert(0, str(tt_metal_home))
os.environ["PYTHONPATH"] = str(tt_metal_home)

import ttnn
from models.experimental.stable_diffusion_35_large.tt.pipeline import TtStableDiffusion3Pipeline

def load_prompts(prompts_file):
    """Load prompts from text file, skipping comments and blank lines"""
    with open(prompts_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def model_location_generator(model_version, model_subdir="", download_if_ci_v2=False, ci_v2_timeout_in_s=300):
    """Simple model locator that downloads from HuggingFace"""
    return model_version

def find_existing_frames(output_dir):
    """Find already-generated frames to enable resume"""
    existing = []
    i = 0
    while True:
        frame_path = output_dir / f"frame_{i:03d}.png"
        if frame_path.exists():
            existing.append(i)
            i += 1
        else:
            break
    return existing

def main():
    # Configuration
    prompts_file = Path("~/tt-scratchpad/video_prompts.txt").expanduser()
    output_dir = Path("~/tt-scratchpad").expanduser()

    # Load your custom prompts
    prompts = load_prompts(prompts_file)

    # Check for existing frames (resume capability)
    existing_frames = find_existing_frames(output_dir)
    start_frame = len(existing_frames)

    if start_frame > 0:
        print(f"🔄 Resuming from frame {start_frame} ({start_frame} frames already exist)")
        print(f"⏱️  Estimated time: ~1.5 minutes per frame on N150 (~{(len(prompts) - start_frame) * 1.5:.0f} min remaining)\n")
    else:
        print(f"🎬 Generating {len(prompts)} video frames for Tenstorrent World's Fair!")
        print(f"⏱️  Estimated time: ~4 minutes per frame on N150 (~{len(prompts) * 4} min total)\n")

    # Initialize Tenstorrent hardware
    print("🔧 Initializing N150 Wormhole hardware...")
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1)
    )
    print("✓ Hardware ready\n")

    # Load Stable Diffusion 3.5 Large pipeline
    print("📥 Loading Stable Diffusion 3.5 Large (~10GB download on first run)...")
    pipeline = TtStableDiffusion3Pipeline(
        checkpoint_name="stabilityai/stable-diffusion-3.5-large",
        device=mesh_device,
        enable_t5_text_encoder=False,
        vae_cpu_fallback=True,
        guidance_cond=2,
        model_location_generator=model_location_generator,
    )

    pipeline.prepare(
        width=1024,
        height=1024,
        guidance_scale=3.5,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
    )
    print("✓ Pipeline loaded and ready\n")

    # Generate frames from your custom prompts (with resume support)
    import time
    total_frames = len(prompts)
    frames_to_generate = total_frames - start_frame

    if start_frame > 0:
        print(f"🎨 Resuming frame generation from frame {start_frame + 1}...\n")
    else:
        print("🎨 Starting frame generation...\n")

    start_time = time.time()
    for i, prompt in enumerate(prompts[start_frame:], start=start_frame):
        frames_done = i - start_frame
        print(f"━━━ Frame {i+1}/{total_frames} ━━━")
        print(f"📝 Prompt: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

        # Progress and ETA
        if frames_done > 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / frames_done
            remaining_frames = frames_to_generate - frames_done
            eta_seconds = avg_time * remaining_frames
            eta_minutes = eta_seconds / 60
            print(f"⏱️  Progress: {frames_done}/{frames_to_generate} | ETA: {eta_minutes:.1f} min")

        # Generate 1024x1024 image with your prompt (with retry on failure)
        negative_prompt = ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                images = pipeline(
                    prompt_1=[prompt],
                    prompt_2=[prompt],
                    prompt_3=[prompt],
                    negative_prompt_1=[negative_prompt],
                    negative_prompt_2=[negative_prompt],
                    negative_prompt_3=[negative_prompt],
                    num_inference_steps=28,
                    seed=0
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Generation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"❌ Generation failed after {max_retries} attempts: {e}")
                    print(f"   Tip: Try running 'tt-smi -r' to reset device, then rerun script")
                    raise

        # Save with video-ready sequential naming
        output_path = output_dir / f"frame_{i:03d}.png"
        images[0].save(str(output_path))
        print(f"✓ Saved: {output_path.name}")
        print(f"   (Frame {i+1} complete)\n")

    # Cleanup
    ttnn.close_mesh_device(mesh_device)

    # Show summary
    total_time = time.time() - start_time
    total_minutes = total_time / 60
    avg_per_frame = total_time / frames_to_generate if frames_to_generate > 0 else 0

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎉 All frames generated!")
    print(f"📁 Location: ~/tt-scratchpad/frame_*.png")
    print(f"📊 Total frames: {total_frames}")
    print(f"⏱️  Total time: {total_minutes:.1f} minutes ({avg_per_frame:.1f}s per frame)")
    print("\n▶️  Next: Create your video with ffmpeg")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if __name__ == "__main__":
    main()
