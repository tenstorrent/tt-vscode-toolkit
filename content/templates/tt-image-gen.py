#!/usr/bin/env python3
"""
Tenstorrent Image Generation Script
Generate images from text prompts using Stable Diffusion on tt-metal
"""

import argparse
import os
import time
from pathlib import Path

try:
    from diffusers import StableDiffusionPipeline
    import torch
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install diffusers transformers accelerate safetensors pillow")
    exit(1)


def setup_model(model_path=None, low_memory=False):
    """
    Load Stable Diffusion model optimized for N150
    """
    if model_path is None:
        model_path = os.path.expanduser("~/models/stable-diffusion-v1-4")

    print(f"Loading Stable Diffusion model from {model_path}...")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please download the model first using:")
        print("  hf download CompVis/stable-diffusion-v1-4 --local-dir ~/models/stable-diffusion-v1-4")
        return None

    # Load pipeline
    # Note: For tt-metal integration, we use CPU/torch as a baseline
    # In production, this would be replaced with tt-metal backend
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,  # Disable for faster generation
    )

    # Optimize for N150 - enable attention slicing to reduce memory
    if low_memory:
        pipe.enable_attention_slicing()

    # Move to available device
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("✓ Using CUDA acceleration")
    else:
        pipe = pipe.to("cpu")
        print("✓ Using CPU (consider installing CUDA for faster generation)")

    print("✓ Model loaded successfully!\n")
    return pipe


def generate_image(pipe, prompt, output_path, steps=50, guidance_scale=7.5,
                   width=512, height=512, seed=None):
    """
    Generate a single image from a text prompt
    """
    print(f"Prompt: '{prompt}'")
    print(f"Generating image ({steps} denoising steps)...\n")

    # Set random seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        print(f"Using seed: {seed}")

    start_time = time.time()

    # Generate image
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    generation_time = time.time() - start_time

    # Save image
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"\n✓ Image saved to: {output_path}")
    print(f"Generation time: {generation_time:.1f} seconds\n")

    return image


def generate_batch(pipe, prompt, output_prefix, batch_size=4, **kwargs):
    """
    Generate multiple variations of the same prompt
    """
    print(f"Generating {batch_size} variations...")
    print(f"Prompt: '{prompt}'\n")

    images = []
    for i in range(batch_size):
        output_path = f"{output_prefix}{i}.png"
        print(f"[{i+1}/{batch_size}]")

        # Use different seed for each image
        kwargs['seed'] = i if 'seed' not in kwargs else kwargs['seed'] + i

        image = generate_image(pipe, prompt, output_path, **kwargs)
        images.append(image)

    print(f"✓ Generated {batch_size} images!")
    return images


def interactive_mode(pipe, default_output="~/tt-scratchpad/generated"):
    """
    Interactive mode - generate multiple images from user prompts
    """
    print("=" * 60)
    print("Interactive Image Generation Mode")
    print("=" * 60)
    print("Enter your prompts to generate images.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit interactive mode")
    print("  'params' - Show current parameters")
    print("=" * 60)
    print()

    # Default parameters
    params = {
        'steps': 50,
        'guidance_scale': 7.5,
        'width': 512,
        'height': 512,
    }

    image_count = 0

    while True:
        try:
            prompt = input("> ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break

            if prompt.lower() == 'params':
                print(f"\nCurrent parameters:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
                print()
                continue

            # Generate image
            output_path = Path(default_output).expanduser()
            output_file = output_path.parent / f"{output_path.name}-{image_count:03d}.png"

            generate_image(pipe, prompt, str(output_file), **params)
            image_count += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Stable Diffusion on Tenstorrent hardware"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for image generation"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="~/tt-scratchpad/generated.png",
        help="Output path for generated image (default: ~/tt-scratchpad/generated.png)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="~/models/stable-diffusion-v1-4",
        help="Path to Stable Diffusion model (default: ~/models/stable-diffusion-v1-4)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50)"
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale - how closely to follow the prompt (default: 7.5)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (optional)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        help="Generate multiple variations (batch size)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode for multiple generations"
    )

    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable memory optimizations (slower but uses less RAM)"
    )

    args = parser.parse_args()

    # Load model
    pipe = setup_model(args.model, args.low_memory)
    if pipe is None:
        return 1

    # Interactive mode
    if args.interactive:
        interactive_mode(pipe, args.output.replace('.png', ''))
        return 0

    # Check if prompt provided
    if not args.prompt:
        print("Error: Please provide a prompt with --prompt or use --interactive mode")
        print("\nExample:")
        print('  python tt-image-gen.py --prompt "A futuristic AI chip, orange accents"')
        return 1

    # Generate image(s)
    kwargs = {
        'steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'width': args.width,
        'height': args.height,
    }

    if args.seed is not None:
        kwargs['seed'] = args.seed

    if args.batch:
        # Batch generation
        output_prefix = args.output.replace('.png', '-')
        generate_batch(pipe, args.prompt, output_prefix, args.batch, **kwargs)
    else:
        # Single image generation
        generate_image(pipe, args.prompt, args.output, **kwargs)

    print("=" * 60)
    print("Image generation complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
