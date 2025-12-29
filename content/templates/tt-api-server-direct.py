#!/usr/bin/env python3
"""
Direct API HTTP Server for Llama on Tenstorrent Hardware

This Flask server uses the Generator API directly, keeping the model loaded
in memory for fast inference across multiple requests.

Usage:
    export HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    export PYTHONPATH=~/tt-metal
    cd ~/tt-metal
    python3 ~/tt-api-server-direct.py --port 8080
"""

import sys
import os
import time
import argparse

# Check Flask is installed
try:
    from flask import Flask, request, jsonify
except ImportError:
    print("❌ Flask not installed. Please run: pip install flask")
    sys.exit(1)

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

print("🔄 Importing tt-metal libraries and loading model...")
print("   This will take 2-5 minutes on first run...\n")

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
logger.add(sys.stderr, level="WARNING")  # Only show warnings and errors

# Global variables for model (loaded once on startup)
GENERATOR = None
MODEL_ARGS = None
MODEL = None
PAGE_TABLE = None
TT_KV_CACHE = None
MESH_DEVICE = None
MAX_GENERATED_TOKENS = 128


def initialize_model():
    """
    Initialize the model and generator.
    This is called once when the server starts.
    """
    global GENERATOR, MODEL_ARGS, MODEL, PAGE_TABLE, TT_KV_CACHE, MESH_DEVICE, MAX_GENERATED_TOKENS

    print("📥 Initializing Tenstorrent mesh device...")

    # Get mesh device
    MESH_DEVICE = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(ttnn.get_device_ids())),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.ROW),
    )

    print("📥 Loading model into memory...")

    # Use performance optimizations
    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    # Create the model (without paged attention for simplicity)
    MODEL_ARGS, MODEL, TT_KV_CACHE, state_dict = create_tt_model(
        MESH_DEVICE,
        instruct=True,
        max_batch_size=1,
        optimizations=optimizations,
        max_seq_len=2048,
        paged_attention_config=None,  # Disable paged attention for single-user server
        dtype=ttnn.bfloat8_b,
        state_dict=None,
    )

    # No page table needed when paged attention is disabled
    PAGE_TABLE = None

    # Create generator
    GENERATOR = Generator(
        [MODEL],
        [MODEL_ARGS],
        MESH_DEVICE,
        tokenizer=MODEL_ARGS.tokenizer,
    )

    print("✅ Model loaded successfully!")


def generate_response(prompt, max_tokens=128, temperature=0.0):
    """
    Generate a response for the given prompt.
    Uses the globally loaded model for fast inference.
    """
    # Preprocess the prompt
    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt],
        MODEL_ARGS.tokenizer,
        [MODEL_ARGS],
        instruct=True,
        max_generated_tokens=max_tokens,
        max_prefill_len=2048,
    )

    # Batch the input
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)

    # Prefill phase
    logits = GENERATOR.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=PAGE_TABLE,
        kv_cache=[TT_KV_CACHE],  # Generator expects a list of kv_caches
        prompt_lens=decoding_pos,
    )

    # Get first token
    prefilled_token = torch.argmax(logits, dim=-1)

    # Collect outputs (encoded_prompts is already a list)
    all_outputs = encoded_prompts[0][:prefill_lens[0]]
    all_outputs.append(int(prefilled_token[0].item()))

    # Decode phase
    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = prefilled_token

    # Setup sampling
    if temperature == 0.0:
        device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
    else:
        device_sampling_params = None

    tokens_generated = 0

    for iteration in range(max_tokens):
        # Generate next token
        logits = GENERATOR.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=False,  # Tracing disabled for simplicity (requires trace region allocation)
            page_table=PAGE_TABLE,
            kv_cache=[TT_KV_CACHE],  # Generator expects a list of kv_caches
            sampling_params=device_sampling_params,
        )

        # Sample
        if device_sampling_params is not None:
            out_tok = logits.unsqueeze(1)
        else:
            _, out_tok = sample_host(logits, temperature=temperature, top_p=0.9, on_host=True)

        user_tok = out_tok[0].item()

        # Check for end of sequence
        if user_tok in MODEL_ARGS.tokenizer.stop_tokens:
            break

        all_outputs.append(user_tok)
        current_pos += 1
        tokens_generated += 1

    # Decode to text
    full_text = MODEL_ARGS.tokenizer.decode(all_outputs)

    # Extract response
    prompt_including_tags = MODEL_ARGS.tokenizer.decode(
        MODEL_ARGS.encode_prompt(prompt, instruct=True)
    )
    response = full_text.replace(prompt_including_tags, "", 1).strip()

    return response, tokens_generated


# Create Flask app
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": os.environ.get('HF_MODEL'),
        "model_loaded": GENERATOR is not None,
        "note": "Model is loaded in memory for fast inference"
    })


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint - generates responses using the loaded model.

    Request JSON:
        {
            "prompt": "Your question here",
            "max_tokens": 128,      // optional
            "temperature": 0.0      // optional (0.0 = greedy)
        }

    Response JSON:
        {
            "prompt": "Your question here",
            "response": "Generated response...",
            "tokens_generated": 45,
            "time_seconds": 1.23
        }
    """
    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        max_tokens = data.get('max_tokens', 128)
        temperature = data.get('temperature', 0.0)

        # Validate parameters
        if max_tokens < 1 or max_tokens > 2048:
            return jsonify({"error": "max_tokens must be between 1 and 2048"}), 400

        if temperature < 0.0 or temperature > 2.0:
            return jsonify({"error": "temperature must be between 0.0 and 2.0"}), 400

        print(f"📝 Request: {prompt[:100]}...")

        # Generate response
        start_time = time.time()
        response, tokens_generated = generate_response(prompt, max_tokens, temperature)
        elapsed = time.time() - start_time

        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0

        print(f"✓ Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.1f} tok/s)")

        return jsonify({
            "prompt": prompt,
            "response": response,
            "tokens_generated": tokens_generated,
            "time_seconds": round(elapsed, 2),
            "tokens_per_second": round(tokens_per_second, 1)
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /chat"
        ]
    }), 404


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Llama API Server (Direct API)")
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()

    try:
        # Initialize model on startup (this is the slow part)
        initialize_model()

        print("\n🌐 Llama API Server (Direct API) on Tenstorrent")
        print("=" * 60)
        print(f"Model: {os.environ.get('HF_MODEL')}")
        print(f"\n🚀 Server ready on http://{args.host}:{args.port}")
        print("\nAvailable endpoints:")
        print(f"  • GET  http://{args.host}:{args.port}/health")
        print(f"  • POST http://{args.host}:{args.port}/chat")
        print("\nNote: Model is loaded in memory - inference is fast!")
        print("      No reloading between requests.\n")
        print("Press CTRL+C to stop the server\n")

        # Run Flask server
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=False,  # Single-threaded to avoid concurrent access to model
        )

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if MESH_DEVICE is not None:
            ttnn.close_mesh_device(MESH_DEVICE)


if __name__ == "__main__":
    main()
