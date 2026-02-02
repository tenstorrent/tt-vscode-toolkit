#!/usr/bin/env python3
"""
Zen Master Inference Testing

Tests the fine-tuned Zen Master model by generating responses
to sample prompts. Compares base model vs fine-tuned responses.

Usage:
    # Test fine-tuned model
    python test_zen_model.py --model-path ./output/final_model --config configs/finetune_zen_n150.yaml

    # Compare base vs fine-tuned
    python test_zen_model.py --model-path ./output/final_model --config configs/finetune_zen_n150.yaml --compare
"""

import argparse
import numpy as np
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
    temperature: float = 0.0,
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
        temperature: Sampling temperature (0.0 for greedy)

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

            # Sample (temperature=0.0 means greedy)
            next_token_tensor = ttml.ops.sample.sample_op(
                logits, temperature, np.random.randint(low=1e7), logits_mask_tensor
            )

            # Extract token at last position
            next_token_idx = (
                max_sequence_length - 1
                if len(prompt_tokens) > max_sequence_length
                else len(window) - 1
            )
            next_token = int(
                next_token_tensor.to_numpy(composer=composer).reshape(-1, 1)[
                    next_token_idx
                ][0]
            )

            if next_token == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            prompt_tokens.append(next_token)

    model.train()

    return tokenizer.decode(generated_tokens).strip()


def test_model(args):
    """
    Test the Zen Master model on sample questions.
    """

    print("=" * 70)
    print("🧘 Zen Master Inference Test")
    print("=" * 70)
    print()

    # Load config
    print(f"Loading config: {args.config}")
    yaml_config = load_config(args.config)
    model_config = load_config(yaml_config["training_config"]["model_config"])

    device_config = ttml.common.config.DeviceConfig(yaml_config)

    # Initialize device (for multi-device)
    if device_config.total_devices() > 1:
        print(f"Initializing {device_config.total_devices()} devices...")
        initialize_device(yaml_config)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model factory
    orig_vocab_size = tokenizer.vocab_size
    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    max_sequence_length = tt_model_factory.transformer_config.max_sequence_length
    padded_vocab_size = round_up_to_tile(orig_vocab_size, 32)

    # Setup masks
    causal_mask = build_causal_mask(max_sequence_length)
    causal_mask = ttml.autograd.Tensor.from_numpy(
        causal_mask, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.BFLOAT16
    )
    logits_mask_tensor = build_logits_mask(orig_vocab_size, padded_vocab_size)

    # Test questions
    test_questions = [
        "What is enlightenment?",
        "How do I debug my code?",
        "What is the meaning of life?",
        "Why do we procrastinate?",
        "What is consciousness?",
        "How do I learn programming?",
        "What is time?",
        "Why do we need AI?",
        "What is success?",
        "How do I handle failure?",
    ]

    # Test base model (if requested)
    if args.compare:
        print("\n" + "=" * 70)
        print("📊 BASELINE: Testing base TinyLlama model (before fine-tuning)")
        print("=" * 70)
        print("\nCreating base model...")
        base_model = tt_model_factory.create_model()

        if args.base_weights:
            print(f"Loading base weights from {args.base_weights}")
            base_model.load_from_safetensors(args.base_weights)
        else:
            print("⚠️  No base weights specified - using random initialization")

        print("\nGenerating baseline responses...")
        print("-" * 70)

        for i, question in enumerate(test_questions[:5], 1):  # Test first 5
            print(f"\nQ{i}: {question}")
            response = generate_response(
                base_model,
                tokenizer,
                question,
                max_sequence_length,
                causal_mask,
                logits_mask_tensor,
                max_gen_tokens=128,
                temperature=args.temperature,
            )
            print(f"Base: {response}")

    # Test fine-tuned model
    print("\n" + "=" * 70)
    print("✨ ZEN MASTER: Testing fine-tuned model")
    print("=" * 70)
    print("\nCreating model...")
    zen_model = tt_model_factory.create_model()

    print(f"Loading fine-tuned weights from {args.model_path}")
    zen_model.load_from_safetensors(args.model_path)

    print("\nGenerating Zen Master responses...")
    print("-" * 70)

    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        response = generate_response(
            zen_model,
            tokenizer,
            question,
            max_sequence_length,
            causal_mask,
            logits_mask_tensor,
            max_gen_tokens=128,
            temperature=args.temperature,
        )
        print(f"Zen: {response}")
        results.append((question, response))

    # Save results
    if args.output:
        output_path = Path(args.output)
        print(f"\n\nSaving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Zen Master Inference Test Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Temperature: {args.temperature}\n\n")

            for i, (question, response) in enumerate(results, 1):
                f.write(f"Question {i}: {question}\n")
                f.write(f"Response: {response}\n\n")

        print(f"✅ Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("✅ Testing complete!")
    print(f"Tested {len(test_questions)} questions")
    print(f"Model: {args.model_path}")
    if args.compare:
        print("Comparison: Base vs Fine-tuned ✓")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Zen Master model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (safetensors directory)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (same as used for training)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        help="HuggingFace model name for tokenizer",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default=None,
        help="Path to base model weights (for comparison)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model vs fine-tuned model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy, higher for more randomness)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="zen_test_results.txt",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model path not found: {model_path}")
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        return 1

    test_model(args)
    return 0


if __name__ == "__main__":
    exit(main())
