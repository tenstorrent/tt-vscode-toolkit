#!/usr/bin/env python3
"""
HTTP API Server for Llama on Tenstorrent Hardware

This script provides a simple Flask-based REST API that wraps the tt-metal demo.
For production use, you'd want to load the model once and keep it in memory.

Usage:
    export LLAMA_DIR="~/models/Llama-3.1-8B-Instruct/original"
    export PYTHONPATH=~/tt-metal
    cd ~/tt-metal
    python3 ~/tt-api-server.py --port 8080
"""

import sys
import os
import subprocess
import time
import argparse
import json
import tempfile
from pathlib import Path

# Check Flask is installed
try:
    from flask import Flask, request, jsonify
except ImportError:
    print("❌ Flask not installed. Please run: pip install flask")
    sys.exit(1)

# Check environment
llama_dir = os.environ.get('LLAMA_DIR')
pythonpath = os.environ.get('PYTHONPATH')

if not llama_dir:
    print("❌ LLAMA_DIR not set")
    print('   Run: export LLAMA_DIR="~/models/Llama-3.1-8B-Instruct/original"')
    sys.exit(1)

if not pythonpath:
    print("❌ PYTHONPATH not set")
    print('   Run: export PYTHONPATH=~/tt-metal')
    sys.exit(1)

# Check if we're in the tt-metal directory
if not os.path.exists('models/tt_transformers/demo/simple_text_demo.py'):
    print("❌ Not running from tt-metal directory")
    print('   Run: cd ~/tt-metal')
    sys.exit(1)

# Create Flask app
app = Flask(__name__)

def create_prompt_json(prompt: str, temp_dir: Path) -> Path:
    """
    Create a temporary JSON file with the user's prompt.
    Format matches the demo's expected input structure.
    """
    prompt_file = temp_dir / "custom_prompt.json"

    # Format: list of objects with "prompt" key
    # This matches the format in models/tt_transformers/demo/sample_prompts/*.json
    prompt_data = [{"prompt": prompt}]

    with open(prompt_file, 'w') as f:
        json.dump(prompt_data, f, indent=2)

    return prompt_file

def run_inference_demo(prompt: str):
    """
    Run the tt-metal demo script with a custom prompt

    Note: This runs the full demo which loads the model each time.
    For production, you'd want to load the model once and reuse it.

    Args:
        prompt: The user's prompt text

    Returns:
        tuple: (success: bool, output: str, elapsed_time: float)
    """
    # Create temp directory for prompt JSON
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        prompt_file = create_prompt_json(prompt, temp_path)

        cmd = [
            'pytest',
            'models/tt_transformers/demo/simple_text_demo.py',
            '-k', 'performance-batch-1',
            '--input_prompts', str(prompt_file),
            '--max_seq_len', '1024',
            '--max_generated_tokens', '128',
            '--instruct', '1',  # Use instruct mode
            '-s',
        ]

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            elapsed = time.time() - start_time

            if result.returncode == 0:
                return True, result.stdout, elapsed
            else:
                return False, result.stderr, elapsed

        except subprocess.TimeoutExpired:
            return False, "Demo timed out after 5 minutes", 300
        except Exception as e:
            return False, str(e), 0

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "note": "Server is running. Send POST /chat with custom prompts.",
        "llama_dir": os.environ.get('LLAMA_DIR'),
    })

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint

    Accepts custom prompts and generates responses using the Llama model.

    Request JSON:
        {
            "prompt": "Your prompt here"
        }

    Response JSON:
        {
            "success": true,
            "prompt": "Your prompt here",
            "output": "Demo output...",
            "time_seconds": 45.2
        }
    """
    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']

        print(f"📝 Received prompt: {prompt}")
        print("⚙️  Running inference (this may take a few minutes on first run)...")

        success, output, elapsed = run_inference_demo(prompt)

        if success:
            print(f"✓ Inference completed in {elapsed:.1f}s")
            return jsonify({
                "success": True,
                "prompt": prompt,
                "output": output,
                "time_seconds": round(elapsed, 2)
            })
        else:
            print(f"❌ Inference failed: {output}")
            return jsonify({
                "success": False,
                "error": output,
                "time_seconds": round(elapsed, 2)
            }), 500

    except Exception as e:
        print(f"❌ Error: {e}")
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
    parser = argparse.ArgumentParser(description="Llama API Server")
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()

    print("\n🌐 Llama API Server on Tenstorrent Hardware")
    print("=" * 50)
    print(f"✓ LLAMA_DIR: {llama_dir}")
    print(f"✓ PYTHONPATH: {pythonpath}")
    print(f"✓ Demo script: models/tt_transformers/demo/simple_text_demo.py")
    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print("\nAvailable endpoints:")
    print(f"  • GET  http://{args.host}:{args.port}/health")
    print(f"  • POST http://{args.host}:{args.port}/chat")
    print("\nNote: Send custom prompts via POST /chat endpoint.")
    print("      Each request reloads the model (demo limitation).")
    print("      First request will take several minutes for kernel compilation.")
    print("\nPress CTRL+C to stop the server\n")

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=False,  # Single-threaded to avoid concurrent model loading
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")

if __name__ == "__main__":
    main()
