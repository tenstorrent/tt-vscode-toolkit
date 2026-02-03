#!/usr/bin/env python3
"""
Interactive Chat with NanoGPT Model

Chat with a trained NanoGPT model (like the trickster model).
Supports .pkl checkpoints from train_nanogpt.py.

Usage:
    python chat_with_nanogpt.py --checkpoint path/to/model.pkl --temperature 0.7

Commands:
    - Type your prompt and press Enter
    - Type 'exit' or 'quit' to end conversation
    - Type 'help' for tips
"""

import argparse
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass

# Import from train_nanogpt.py location
import os
import sys

# Add tt-train to path
tt_metal_home = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal"))
tt_train_path = os.path.join(tt_metal_home, "tt-train/sources/examples/nano_gpt")
if tt_train_path not in sys.path:
    sys.path.insert(0, tt_train_path)

from train_nanogpt import load_model_from_checkpoint
import ttml
import ttnn
from ttml.common.utils import no_grad


# Configuration classes (needed for pickle deserialization)
@dataclass
class ModelConfig:
    """Model configuration - must match train_nanogpt.py for pickle loading."""
    model_type: str = "gpt2"
    model_path: str = ""
    vocab_size: int = 50304
    embedding_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 6
    dropout_prob: float = 0.2
    bias: bool = True
    max_sequence_length: int = 128
    block_size: int = 128  # Alias for max_sequence_length (used by NanoGPT)


# Minimal TrainingConfig for pickle (we don't actually use it, just need it for deserialization)
class TrainingConfig:
    """Minimal TrainingConfig for pickle deserialization."""
    def __init__(self, yaml_config=None):
        pass


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: NanoGPT model
        tokenizer: Character tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated text
    """
    model.eval()

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)

    # Generate
    generated_tokens = prompt_tokens[:]

    import numpy as np

    for _ in range(max_new_tokens):
        # Get model input (last block_size tokens)
        context = generated_tokens[-model.config.block_size:]

        # Pad to block_size if needed
        if len(context) < model.config.block_size:
            # Pad with zeros (or a pad token if your tokenizer has one)
            padded_context = [0] * (model.config.block_size - len(context)) + context
        else:
            padded_context = context

        # Shape: (1, block_size)
        context_np = np.array([padded_context], dtype=np.uint32)
        context_tensor = ttml.autograd.Tensor.from_numpy(
            context_np,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
        )

        # Forward pass
        with no_grad():
            logits = model(context_tensor)

        # Convert to numpy and get last token logits
        # logits shape: (batch_size, seq_len, vocab_size)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        # Get last position logits for first batch element (ensure 1D)
        last_logits = logits_np[0, -1, :].flatten()

        # Sample with temperature
        if temperature == 0.0:
            # Greedy
            next_token = int(np.argmax(last_logits))
        else:
            # Temperature sampling
            logits_temp = last_logits / temperature

            # Softmax
            exp_logits = np.exp(logits_temp - np.max(logits_temp))
            probs = exp_logits / exp_logits.sum()

            # Sample
            next_token = np.random.choice(len(probs), p=probs)

        # Stop if end token (if using)
        # For character-level, there's no specific end token
        # Just generate until max_new_tokens

        generated_tokens.append(int(next_token))

    # Decode
    generated_text = tokenizer.decode(generated_tokens)

    # Return only the new part (after prompt)
    return generated_text[len(prompt):]


def print_welcome():
    """Print welcome message."""
    print("=" * 70)
    print("🎭 Interactive NanoGPT Chat")
    print("=" * 70)
    print()
    print("Chat with your trained NanoGPT model!")
    print()
    print("Commands:")
    print("  - Type your prompt and press Enter")
    print("  - Type 'exit' or 'quit' to end conversation")
    print("  - Type 'help' for tips")
    print()
    print("=" * 70)
    print()


def print_help():
    """Print help message."""
    print()
    print("💡 Tips for Great Generation:")
    print("-" * 70)
    print("✅ Keep prompts short (model has limited context)")
    print("✅ Try different temperatures (0.3 = conservative, 0.9 = creative)")
    print("✅ Experiment with prompt phrasing")
    print()
    print("Example prompts:")
    print("  - Once upon a time")
    print("  - Q: What is")
    print("  - The meaning of life is")
    print("-" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with NanoGPT model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pkl file)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-1.0). Higher = more creative. Default: 0.7",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per response. Default: 100",
    )

    args = parser.parse_args()

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model
    print(f"📦 Loading model from {args.checkpoint}...")
    try:
        model, tokenizer, model_config, training_config, step = load_model_from_checkpoint(
            str(checkpoint_path)
        )
        print(f"✅ Model loaded! (trained for {step} steps)")
        print(f"   Vocabulary size: {model_config.vocab_size}")
        print(f"   Block size: {model_config.block_size}")
        print()
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        sys.exit(1)

    # Print welcome
    print_welcome()

    # Interactive loop
    conversation_count = 0
    try:
        while True:
            # Get prompt
            try:
                prompt = input("You: ").strip()
            except EOFError:
                # Handle Ctrl+D
                print("\n\n👋 Goodbye!")
                break

            # Check for exit commands
            if prompt.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Goodbye!")
                break

            # Check for help
            if prompt.lower() in ["help", "?"]:
                print_help()
                continue

            # Skip empty prompts
            if not prompt:
                continue

            # Generate response
            print("\n🎭 NanoGPT: ", end="", flush=True)
            try:
                response = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(response)
                print()  # Extra newline
                conversation_count += 1

            except Exception as e:
                print(f"\n❌ Error generating: {e}")
                print("Try a different prompt or type 'help' for tips.\n")
                continue

    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("\n\n👋 Goodbye!")

    # Summary
    print(f"\n📊 Chat statistics: {conversation_count} prompts answered")
    print("✨ Thanks for chatting!")


if __name__ == "__main__":
    main()
