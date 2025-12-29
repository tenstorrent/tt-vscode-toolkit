#!/usr/bin/env python3
"""
Interactive Chat Script for Llama on Tenstorrent Hardware

This script provides a simple REPL wrapper around the tt-metal demo,
allowing you to chat with custom prompts by generating temporary JSON files.

Note: This approach reloads the model for each query. For production use,
you'd want to use the Generator API directly to keep the model loaded.

Usage:
    export LLAMA_DIR="~/models/Llama-3.1-8B-Instruct/original"
    export PYTHONPATH=~/tt-metal
    cd ~/tt-metal
    python3 ~/tt-chat.py
"""

import sys
import os
import subprocess
import json
import tempfile
from pathlib import Path

def check_environment():
    """Check that required environment variables are set"""
    llama_dir = os.environ.get('LLAMA_DIR')
    pythonpath = os.environ.get('PYTHONPATH')

    if not llama_dir:
        print("❌ LLAMA_DIR environment variable not set")
        print('   Run: export LLAMA_DIR="~/models/Llama-3.1-8B-Instruct/original"')
        return False

    if not pythonpath:
        print("❌ PYTHONPATH environment variable not set")
        print('   Run: export PYTHONPATH=~/tt-metal')
        return False

    # Check if we're in the tt-metal directory
    if not os.path.exists('models/tt_transformers/demo/simple_text_demo.py'):
        print("❌ Not running from tt-metal directory")
        print('   Run: cd ~/tt-metal')
        return False

    return True

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

def run_inference(prompt: str) -> bool:
    """
    Run inference with the given prompt by creating a temporary JSON file
    and running the demo with it.

    Note: This reloads the model each time. The first run will take several
    minutes for kernel compilation, subsequent runs will be faster.
    """
    # Create temp directory for prompt JSON
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        prompt_file = create_prompt_json(prompt, temp_path)

        cmd = [
            'pytest',
            'models/tt_transformers/demo/simple_text_demo.py',
            '-k', 'performance-batch-1',  # Use batch-1 configuration
            '--input_prompts', str(prompt_file),
            '--max_seq_len', '1024',
            '--max_generated_tokens', '128',
            '--instruct', '1',  # Use instruct mode
            '-s',  # Show stdout
        ]

        try:
            print("\n🤖 Generating response...\n")
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Inference failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print("❌ pytest not found. Make sure it's installed:")
            print("   pip install pytest")
            return False

def chat_loop():
    """Run the interactive chat REPL"""
    print("\n🤖 Llama Chat on Tenstorrent Hardware")
    print("=" * 50)

    if not check_environment():
        return

    print("\n✅ Environment configured correctly")
    print("\nNote: Each query reloads the model (demo limitation).")
    print("      The first run will take several minutes to compile kernels.")
    print("      Subsequent runs are faster due to caching.\n")
    print("Commands:")
    print("  • Type your prompt and press ENTER")
    print("  • Type 'exit' or 'quit' to end")
    print("  • Press Ctrl+C to interrupt\n")

    while True:
        try:
            # Read user input
            user_input = input("> ").strip()

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            # Skip empty input
            if not user_input:
                continue

            # Run inference with the user's prompt
            if not run_inference(user_input):
                print("\nTry again, or type 'exit' to quit")
                continue

            print("\n" + "-" * 50)
            print("Ready for next prompt (or 'exit' to quit)\n")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

    print("👋 Chat session ended")

def main():
    """Main entry point"""
    try:
        chat_loop()
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
