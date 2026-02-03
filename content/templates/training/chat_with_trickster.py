#!/usr/bin/env python3
"""
Interactive Chat with Trickster

Have a conversation with your fine-tuned Trickster model!
Type questions and get creative, playful responses.

Usage:
    python chat_with_trickster.py --model-path ./output/final_model --config configs/trickster_n150.yaml

Commands:
    - Type your question and press Enter
    - Type 'exit' or 'quit' to end the conversation
    - Type 'help' for tips on getting good responses
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from transformers import AutoTokenizer
import ttml
from ttml.common.config import load_config
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import (
    round_up_to_tile,
    initialize_device,
    build_logits_mask,
    no_grad,
)
from ttml.common.data import build_causal_mask


def generate_response(
    model,
    tokenizer: AutoTokenizer,
    question: str,
    max_sequence_length: int,
    causal_mask: ttml.autograd.Tensor,
    logits_mask_tensor: ttml.autograd.Tensor,
    max_gen_tokens: int = 128,
    temperature: float = 0.7,
):
    """
    Generate a response to a question.

    Args:
        model: TT model
        tokenizer: Tokenizer
        question: Input question
        max_sequence_length: Maximum sequence length
        causal_mask: Causal attention mask
        logits_mask_tensor: Logits mask
        max_gen_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)

    Returns:
        Generated response text
    """
    model.eval()

    # Format prompt (matching training format)
    prompt_text = f"### Question:\n{question}\n\n### Answer:\n"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    pad_token_id = tokenizer.eos_token_id

    generated_tokens = []

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Preallocate buffer
    padded_prompt_tokens = np.full(
        (1, 1, 1, max_sequence_length), pad_token_id, dtype=np.uint32
    )

    with no_grad():
        for _ in range(max_gen_tokens):
            # Sliding window for long prompts
            if len(prompt_tokens) > max_sequence_length:
                start_idx = len(prompt_tokens) - max_sequence_length
                window = prompt_tokens[start_idx:]
            else:
                window = prompt_tokens

            # Fill buffer
            padded_prompt_tokens[...] = pad_token_id
            padded_prompt_tokens[0, 0, 0, : len(window)] = np.asarray(
                window, dtype=np.uint32
            )

            # Forward pass
            padded_prompt_tensor = ttml.autograd.Tensor.from_numpy(
                padded_prompt_tokens,
                ttml.Layout.ROW_MAJOR,
                ttml.autograd.DataType.UINT32,
            )
            logits = model(padded_prompt_tensor, causal_mask)

            # Sample with temperature
            next_token_tensor = ttml.ops.sample.sample_op(
                logits, temperature, np.random.randint(low=1e7), logits_mask_tensor
            )
            next_token_np = composer(next_token_tensor).numpy()
            next_token_id = int(next_token_np[0, 0, 0, len(window) - 1])

            # Stop if EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)
            prompt_tokens.append(next_token_id)

    # Decode response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


def print_welcome():
    """Print welcome message."""
    print("=" * 70)
    print("🎭 Interactive Trickster Chat")
    print("=" * 70)
    print()
    print("Welcome! You're chatting with your fine-tuned Trickster model.")
    print("Ask questions about machine learning, coding, or anything technical.")
    print()
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'exit' or 'quit' to end the conversation")
    print("  - Type 'help' for tips on getting good responses")
    print()
    print("=" * 70)
    print()


def print_help():
    """Print help message."""
    print()
    print("💡 Tips for Great Conversations:")
    print("-" * 70)
    print("✅ Ask clear, specific questions")
    print("✅ Focus on technical topics (ML, programming, algorithms)")
    print("✅ Try questions similar to your training data")
    print("✅ Experiment with different phrasings")
    print()
    print("Example questions:")
    print("  - What is a neural network?")
    print("  - How do I learn machine learning?")
    print("  - Explain backpropagation like I'm 5")
    print("  - What's the difference between AI and ML?")
    print("  - How does gradient descent work?")
    print("-" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with fine-tuned Trickster model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config YAML"
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
        default=128,
        help="Maximum tokens to generate per response. Default: 128",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    max_sequence_length = config.model_config.max_seq_len

    # Initialize device
    print("🔧 Initializing Tenstorrent device...")
    device = initialize_device(seed=config.training_config.seed)
    device_ctx = ttml.autograd.AutoContext(device=device)
    device_ctx.enter()

    # Load tokenizer
    print(f"📚 Loading tokenizer: {config.model_config.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.tokenizer_path, use_fast=True
    )

    # Create model
    print("🤖 Creating model...")
    model = TransformerModelFactory.create_model(config.model_config, device)

    # Load fine-tuned weights
    print(f"⚖️  Loading fine-tuned weights from {args.model_path}")
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        sys.exit(1)

    model.load_state_dict(model_path)
    print("✅ Model loaded successfully!")

    # Build masks
    print("🎭 Building attention masks...")
    padded_seq_len = round_up_to_tile(max_sequence_length)
    causal_mask = build_causal_mask(padded_seq_len, device)
    logits_mask = build_logits_mask(tokenizer.vocab_size)
    logits_mask_tensor = ttml.autograd.Tensor.from_numpy(
        logits_mask, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.FLOAT32
    )

    # Print welcome
    print_welcome()

    # Interactive chat loop
    conversation_count = 0
    try:
        while True:
            # Get user input
            try:
                question = input("You: ").strip()
            except EOFError:
                # Handle Ctrl+D
                print("\n\n👋 Goodbye! Keep learning!")
                break

            # Check for exit commands
            if question.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Goodbye! Keep learning!")
                break

            # Check for help
            if question.lower() in ["help", "?"]:
                print_help()
                continue

            # Skip empty questions
            if not question:
                continue

            # Generate response
            print("\n🎭 Trickster: ", end="", flush=True)
            try:
                response = generate_response(
                    model,
                    tokenizer,
                    question,
                    max_sequence_length,
                    causal_mask,
                    logits_mask_tensor,
                    max_gen_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(response)
                print()  # Extra newline for readability
                conversation_count += 1

            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Try rephrasing your question or type 'help' for tips.\n")
                continue

    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("\n\n👋 Goodbye! Keep learning!")

    # Cleanup
    print(f"\n📊 Chat statistics: {conversation_count} questions answered")
    print("✨ Thanks for chatting with Trickster!")
    device_ctx.exit()
    device.close()


if __name__ == "__main__":
    main()
