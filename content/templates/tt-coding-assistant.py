#!/usr/bin/env python3
"""
Coding Assistant with Prompt Engineering for Tenstorrent Hardware

This script uses the Direct API with Llama 3.1 8B and a coding-focused system prompt
to create an AI coding assistant. The model loads once and stays in memory for fast
subsequent queries.

Key Concept: PROMPT ENGINEERING - Shape model behavior through system prompts
rather than model specialization. This often delivers 80%+ of the value with
zero compatibility issues!

Usage:
    export LLAMA_DIR=~/models/Llama-3.1-8B-Instruct/original
    export PYTHONPATH=~/tt-metal
    cd ~/tt-metal
    python3 ~/tt-scratchpad/tt-coding-assistant.py
"""

import sys
import os

# Check environment before importing heavy dependencies
llama_dir = os.environ.get('LLAMA_DIR')
pythonpath = os.environ.get('PYTHONPATH')

if not llama_dir:
    print("❌ LLAMA_DIR environment variable not set")
    print('   Run: export LLAMA_DIR=~/models/Llama-3.1-8B-Instruct/original')
    sys.exit(1)

if not pythonpath:
    print("❌ PYTHONPATH environment variable not set")
    print('   Run: export PYTHONPATH=~/tt-metal')
    sys.exit(1)

# Check if we're in the tt-metal directory
if not os.path.exists('models/tt_transformers'):
    print("❌ Not running from tt-metal directory")
    print('   Run: cd ~/tt-metal')
    sys.exit(1)

print("🔄 Importing tt-metal libraries (this may take a moment)...")

# Import tt-metal dependencies
import torch
import ttnn
from loguru import logger

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, SamplingParams, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision

logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Only show INFO and above

# ============================================================================
# SYSTEM PROMPT - The Key to Prompt Engineering
# ============================================================================

SYSTEM_PROMPT = """You are an expert coding assistant specializing in:
- Algorithm design and analysis
- Data structures (trees, graphs, heaps, hash tables)
- Code debugging and optimization
- Explaining complex programming concepts
- Writing clean, well-documented code
- Time and space complexity analysis

When answering:
- Provide clear, concise explanations
- Include code examples when relevant
- Explain your reasoning
- Consider edge cases
- Suggest optimizations where applicable
- Use appropriate programming language syntax

Focus on being helpful, accurate, and educational."""


def prepare_generator(
    mesh_device,
    max_batch_size=1,
    max_seq_len=2048,
    max_generated_tokens=256,
):
    """
    Prepare the Generator and all necessary components.
    This loads the model into memory once.
    """
    print("📥 Loading Llama 3.1 8B model (this will take 2-5 minutes on first run)...")

    # Use performance optimizations for speed
    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    # Create the model and KV cache (without paged attention for simplicity)
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=True,  # Use instruct mode for chat
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=None,  # Disable paged attention for single-user chat
        dtype=ttnn.bfloat8_b,
        state_dict=None,
    )

    # No page table needed when paged attention is disabled
    page_table = None

    # Create the generator
    generator = Generator(
        [model],
        [model_args],
        mesh_device,
        tokenizer=model_args.tokenizer,
    )

    print("✅ Model loaded and ready for coding assistance!")

    return generator, model_args, model, page_table, tt_kv_cache, max_generated_tokens


def generate_response(
    generator,
    model_args,
    model,
    page_table,
    tt_kv_cache,
    prompt,
    max_generated_tokens=256,
    temperature=0.7,
):
    """
    Generate a response for the given prompt using the loaded model.
    Uses temperature 0.7 for balanced determinism and creativity in coding tasks.
    """
    # Prepend system prompt to user prompt
    # This is the key to prompt engineering - shape behavior through instructions!
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

    # Preprocess the prompt
    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [full_prompt],  # Single prompt with system context
        model_args.tokenizer,
        [model_args],
        instruct=True,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=2048,
    )

    # Batch the input
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)

    # Prefill phase - process the prompt
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=[tt_kv_cache],  # Generator expects a list of kv_caches
        prompt_lens=decoding_pos,
    )

    # Get the first token
    prefilled_token = torch.argmax(logits, dim=-1)

    # Collect all output tokens (encoded_prompts is already a list)
    all_outputs = encoded_prompts[0][:prefill_lens[0]]
    all_outputs.append(int(prefilled_token[0].item()))

    # Decode phase - generate tokens one by one
    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = prefilled_token

    # Setup sampling with temperature for balanced responses
    device_sampling_params = None  # Use host sampling for temperature control

    for iteration in range(max_generated_tokens):
        # Generate next token
        logits = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=False,  # Tracing disabled for simplicity
            page_table=page_table,
            kv_cache=[tt_kv_cache],
            sampling_params=device_sampling_params,
        )

        # Sample the next token with temperature
        _, out_tok = sample_host(
            logits,
            temperature=temperature,
            top_p=0.9,
            on_host=True,
        )

        user_tok = out_tok[0].item()

        # Check for end of sequence
        if user_tok in model_args.tokenizer.stop_tokens:
            break

        all_outputs.append(user_tok)
        current_pos += 1

    # Decode tokens to text
    full_text = model_args.tokenizer.decode(all_outputs)

    # Extract just the response (remove the prompt including system prompt)
    # The model's tokenizer adds special tags, so we need to remove everything before "Assistant:"
    if "Assistant:" in full_text:
        response = full_text.split("Assistant:", 1)[1].strip()
    else:
        # Fallback: remove the full prompt
        response = full_text.replace(full_prompt, "", 1).strip()

    return response


def chat_loop():
    """Run the interactive coding assistant loop"""
    print("\n🤖 Coding Assistant on Tenstorrent (Powered by Prompt Engineering)")
    print("=" * 70)
    print("Using Llama 3.1 8B with coding-focused system prompt")
    print("Model loads once and stays in memory for fast responses!\n")

    # Get mesh device (handles N150, N300, T3K, etc.)
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(ttnn.get_device_ids())),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.ROW),
    )

    try:
        # Load the model (this is the slow part - only happens once!)
        generator, model_args, model, page_table, tt_kv_cache, max_gen_tokens = prepare_generator(
            mesh_device,
            max_batch_size=1,
            max_seq_len=2048,
            max_generated_tokens=256,  # Good for code examples
        )

        print("\nCommands:")
        print("  • Type your coding question and press ENTER")
        print("  • Type 'exit' or 'quit' to end")
        print("  • Press Ctrl+C to interrupt\n")
        print("Example prompts:")
        print("  - Write a Python function to find the longest palindrome")
        print("  - Explain how quicksort works with pseudocode")
        print("  - Debug this code: [paste code here]")
        print("  - What's the time complexity of this algorithm?")
        print("  - Suggest optimizations for: [paste code]\n")

        while True:
            try:
                # Get user input
                user_input = input("💻 > ").strip()

                # Check for exit
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break

                # Skip empty input
                if not user_input:
                    continue

                print("\n🤖 Generating response...")

                # Generate response (fast after model is loaded!)
                response = generate_response(
                    generator,
                    model_args,
                    model,
                    page_table,
                    tt_kv_cache,
                    user_input,
                    max_generated_tokens=256,  # Adjust for longer/shorter responses
                    temperature=0.7,  # Balanced creativity for coding
                )

                print(f"\n{response}\n")
                print("-" * 70 + "\n")

            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Try again or type 'exit' to quit\n")
                continue

        print("👋 Coding session ended")

    finally:
        # Cleanup
        ttnn.close_mesh_device(mesh_device)


def main():
    """Main entry point"""
    try:
        chat_loop()
    except Exception as e:
        print(f"❌ Failed to start coding assistant: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
