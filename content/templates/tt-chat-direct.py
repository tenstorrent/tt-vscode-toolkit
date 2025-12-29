#!/usr/bin/env python3
"""
Direct API Interactive Chat for Llama on Tenstorrent Hardware

This script uses the Generator API directly to keep the model loaded in memory,
providing much faster inference after the initial load.

Usage:
    export HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    export PYTHONPATH=~/tt-metal
    cd ~/tt-metal
    python3 ~/tt-chat-direct.py
"""

import sys
import os

# Check environment before importing heavy dependencies
hf_model = os.environ.get('HF_MODEL')
pythonpath = os.environ.get('PYTHONPATH')

if not hf_model:
    print("❌ HF_MODEL environment variable not set")
    print('   Run: export HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"')
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


def prepare_generator(
    mesh_device,
    max_batch_size=1,
    max_seq_len=2048,
    max_generated_tokens=128,
):
    """
    Prepare the Generator and all necessary components.
    This loads the model into memory once.
    """
    print("📥 Loading model (this will take 2-5 minutes on first run)...")

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

    print("✅ Model loaded and ready!")

    return generator, model_args, model, page_table, tt_kv_cache, max_generated_tokens


def generate_response(
    generator,
    model_args,
    model,
    page_table,
    tt_kv_cache,
    prompt,
    max_generated_tokens=128,
    temperature=0.0,
):
    """
    Generate a response for the given prompt using the loaded model.
    """
    # Preprocess the prompt
    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt],  # Single prompt
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

    # Setup sampling (temperature=0 means greedy/argmax)
    if temperature == 0.0:
        device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
    else:
        device_sampling_params = None

    for iteration in range(max_generated_tokens):
        # Generate next token
        logits = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=False,  # Tracing disabled for simplicity (requires trace region allocation)
            page_table=page_table,
            kv_cache=[tt_kv_cache],  # Generator expects a list of kv_caches
            sampling_params=device_sampling_params,
        )

        # Sample the next token
        if device_sampling_params is not None:
            out_tok = logits.unsqueeze(1)
        else:
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

    # Extract just the response (remove the prompt)
    prompt_including_tags = model_args.tokenizer.decode(
        model_args.encode_prompt(prompt, instruct=True)
    )
    response = full_text.replace(prompt_including_tags, "", 1).strip()

    return response


def chat_loop():
    """Run the interactive chat loop"""
    print("\n🤖 Direct API Chat with Llama on Tenstorrent")
    print("=" * 60)
    print("This version loads the model once and keeps it in memory.")
    print("After initial load, responses will be much faster!\n")

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
            max_generated_tokens=128,
        )

        print("\nCommands:")
        print("  • Type your prompt and press ENTER")
        print("  • Type 'exit' or 'quit' to end")
        print("  • Press Ctrl+C to interrupt\n")

        while True:
            try:
                # Get user input
                user_input = input("> ").strip()

                # Check for exit
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break

                # Skip empty input
                if not user_input:
                    continue

                print("\n🤖 Generating response...")

                # Generate response (this is fast after model is loaded!)
                response = generate_response(
                    generator,
                    model_args,
                    model,
                    page_table,
                    tt_kv_cache,
                    user_input,
                    max_generated_tokens=128,
                    temperature=0.0,  # Greedy decoding
                )

                print(f"\n{response}\n")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Try again or type 'exit' to quit\n")
                continue

        print("👋 Chat session ended")

    finally:
        # Cleanup
        ttnn.close_mesh_device(mesh_device)


def main():
    """Main entry point"""
    try:
        chat_loop()
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
