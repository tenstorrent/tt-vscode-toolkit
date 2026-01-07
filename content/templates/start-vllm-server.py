#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Production-ready vLLM server starter for Tenstorrent hardware
#
# This script registers TT-specific models with vLLM and starts the OpenAI-compatible API server.
# Works with ANY model that has a HuggingFace config.json (Llama, Qwen, Mistral, etc.)
# NO dependencies on examples/ directory - this is production code!
#
# AUTO-DETECTS:
# - HF_MODEL: HuggingFace model identifier based on your --model path
# - Hardware type: Runs tt-smi to detect N150/N300/T3K/P100/P150
# - Environment variables: MESH_DEVICE, TT_METAL_ARCH_NAME, TT_METAL_HOME

import runpy
import sys
import os
import json
import subprocess

def register_tt_models():
    """
    Register Tenstorrent model implementations with vLLM's ModelRegistry.

    This allows vLLM to find TT-optimized model implementations in tt-metal
    instead of falling back to slower CPU-based HuggingFace Transformers.

    Registered models run on Tenstorrent hardware using:
    - Specialized TT-Metal kernels
    - Hardware-aware optimizations
    - Direct DRAM access patterns

    Models are imported from: TT_METAL_HOME/models/tt_transformers/tt/generator_vllm.py

    Why this is needed:
    - vLLM doesn't automatically discover custom hardware implementations
    - Without registration, vLLM falls back to generic (slow) implementations
    - Must register before vLLM tries to load any models

    Supported model architectures:
    - Llama (Llama-3.1-8B, Llama-3.1-70B, Llama-3.2-3B-Instruct, etc.)
    - Gemma 3 (Gemma-3-1B-IT, Gemma-3-4B-IT, etc.) - uses Llama architecture
    - Qwen2 (Qwen2.5-Coder-1.5B, Qwen2.5-Coder-7B, etc.) - uses Llama architecture (explicitly registered)
    - Qwen3 (Qwen3-0.6B, Qwen3-8B, etc.) - uses Llama architecture
    - Mistral family - uses Llama architecture
    - Any Llama-compatible architecture
    """
    from vllm import ModelRegistry

    # Register Llama-based models for Tenstorrent hardware
    # Many models (Qwen, Mistral, etc.) use Llama architecture internally
    #
    # Format: ModelRegistry.register_model(
    #     model_name: str,           # Architecture name from config.json
    #     module_path: str           # Python path to TT implementation
    # )

    # Register TT Llama implementation
    ModelRegistry.register_model(
        "TTLlamaForCausalLM",
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    )

    # Explicitly register Qwen2 to use TT Llama implementation
    # Qwen2 uses Llama architecture internally but needs explicit mapping
    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    )

    # Note: Gemma, Mistral, and other Llama-compatible models
    # automatically use the TTLlamaForCausalLM implementation above!

    # Qwen3 (newer architecture) auto-detects, but Qwen2 needs explicit registration

    # Add more TT model architectures here as they become available:
    # ModelRegistry.register_model("TTMixtralForCausalLM", "...")

    print("✓ Registered Tenstorrent model implementations with vLLM")
    print("✓ Supported: Llama, Gemma, Qwen2, Qwen3, Mistral, and Llama-compatible architectures")


def detect_and_configure_hardware():
    """
    Auto-detect Tenstorrent hardware and set required environment variables.

    Runs tt-smi -s to detect hardware type and configures:
    - MESH_DEVICE: Hardware identifier (N150, N300, T3K, P100, P150, GALAXY)
    - TT_METAL_ARCH_NAME: Architecture name (blackhole for P100/P150, auto-detected for others)
    - TT_METAL_HOME: Path to tt-metal installation (defaults to ~/tt-metal)

    Why this is needed:
    - vLLM requires MESH_DEVICE to target the correct hardware
    - Blackhole chips (P100/P150) need explicit TT_METAL_ARCH_NAME=blackhole
    - TT_METAL_HOME must be set for model imports to work

    Hardware types:
    - N150: Single Wormhole chip (most common for development)
    - N300: Dual Wormhole chips
    - T3K: 8 Wormhole chips (Tenstorrent Galaxy)
    - P100: Single Blackhole chip
    - P150: Dual Blackhole chips
    - GALAXY: Multi-chip configurations
    """
    # Only auto-detect if not already set
    if 'MESH_DEVICE' in os.environ:
        print(f"✓ Using existing MESH_DEVICE={os.environ['MESH_DEVICE']}")
        return

    try:
        # Run tt-smi -s to get hardware info in JSON format
        result = subprocess.run(
            ['tt-smi', '-s'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"⚠️  Warning: tt-smi failed (exit code {result.returncode})")
            print("⚠️  Please set MESH_DEVICE manually: export MESH_DEVICE=N150")
            return

        # Parse JSON output
        output = result.stdout
        data = json.loads(output)

        # Extract board type from device_info
        if 'device_info' not in data or not data['device_info']:
            print("⚠️  Warning: No device_info in tt-smi output")
            print("⚠️  Please set MESH_DEVICE manually: export MESH_DEVICE=N150")
            return

        device = data['device_info'][0]
        if 'board_info' not in device or 'board_type' not in device['board_info']:
            print("⚠️  Warning: No board_type in tt-smi output")
            print("⚠️  Please set MESH_DEVICE manually: export MESH_DEVICE=N150")
            return

        board_type = device['board_info']['board_type'].upper()

        # Map board_type to MESH_DEVICE
        # Examples: "N150 L" -> N150, "n300" -> N300, "P100" -> P100
        if 'N150' in board_type:
            mesh_device = 'N150'
            arch_name = None  # Wormhole auto-detects
        elif 'N300' in board_type:
            mesh_device = 'N300'
            arch_name = None  # Wormhole auto-detects
        elif 'T3K' in board_type:
            mesh_device = 'T3K'
            arch_name = None  # Wormhole auto-detects
        elif 'P100' in board_type:
            mesh_device = 'P100'
            arch_name = 'blackhole'  # Blackhole requires explicit arch
        elif 'P150' in board_type:
            mesh_device = 'P150'
            arch_name = 'blackhole'  # Blackhole requires explicit arch
        elif 'GALAXY' in board_type:
            mesh_device = 'GALAXY'
            arch_name = None  # Auto-detect
        else:
            print(f"⚠️  Warning: Unknown board type '{board_type}'")
            print("⚠️  Please set MESH_DEVICE manually: export MESH_DEVICE=N150")
            return

        # Set MESH_DEVICE
        os.environ['MESH_DEVICE'] = mesh_device
        print(f"✓ Auto-detected hardware: {mesh_device}")

        # Set TT_METAL_ARCH_NAME for Blackhole chips
        if arch_name and 'TT_METAL_ARCH_NAME' not in os.environ:
            os.environ['TT_METAL_ARCH_NAME'] = arch_name
            print(f"✓ Auto-set TT_METAL_ARCH_NAME={arch_name}")

        # Set TT_METAL_HOME if not already set (required for imports)
        if 'TT_METAL_HOME' not in os.environ:
            tt_metal_home = os.path.expanduser('~/tt-metal')
            if os.path.exists(tt_metal_home):
                os.environ['TT_METAL_HOME'] = tt_metal_home
                print(f"✓ Auto-set TT_METAL_HOME={tt_metal_home}")
            else:
                print(f"⚠️  Warning: ~/tt-metal not found")
                print("⚠️  Please set TT_METAL_HOME: export TT_METAL_HOME=/path/to/tt-metal")

    except subprocess.TimeoutExpired:
        print("⚠️  Warning: tt-smi timed out")
        print("⚠️  Please set MESH_DEVICE manually: export MESH_DEVICE=N150")
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Failed to parse tt-smi output: {e}")
        print("⚠️  Please set MESH_DEVICE manually: export MESH_DEVICE=N150")
    except FileNotFoundError:
        print("⚠️  Warning: tt-smi not found in PATH")
        print("⚠️  Please install tt-smi or set MESH_DEVICE manually: export MESH_DEVICE=N150")
    except Exception as e:
        print(f"⚠️  Warning: Hardware detection failed: {e}")
        print("⚠️  Please set MESH_DEVICE manually: export MESH_DEVICE=N150")


def auto_detect_hf_model():
    """
    Auto-detect and set HF_MODEL environment variable from --model path.

    This is REQUIRED for non-Llama models to work correctly with vLLM on TT hardware.
    The HF_MODEL tells vLLM the HuggingFace model identifier, separate from the local path.

    Examples:
        --model ~/models/Qwen3-0.6B  →  HF_MODEL=Qwen/Qwen3-0.6B
        --model ~/models/gemma-3-1b-it  →  HF_MODEL=google/gemma-3-1b-it
        --model ~/models/Llama-3.1-8B-Instruct  →  No HF_MODEL needed (Llama auto-detects)
    """
    # Only auto-set if HF_MODEL is not already set
    if 'HF_MODEL' in os.environ:
        print(f"✓ Using existing HF_MODEL={os.environ['HF_MODEL']}")
        return

    # Find --model argument
    if '--model' not in sys.argv:
        return

    try:
        model_path = sys.argv[sys.argv.index('--model') + 1]
        model_name = os.path.basename(model_path.rstrip('/'))

        # Detect model type and set HF_MODEL accordingly
        if 'Qwen' in model_name or 'qwen' in model_name.lower():
            # Qwen models: Qwen/Qwen3-0.6B, Qwen/Qwen3-8B, etc.
            os.environ['HF_MODEL'] = f'Qwen/{model_name}'
            print(f"✓ Auto-detected HF_MODEL=Qwen/{model_name}")
        elif 'gemma' in model_name.lower():
            # Gemma models: google/gemma-3-1b-it, google/gemma-3-4b-it, etc.
            os.environ['HF_MODEL'] = f'google/{model_name}'
            print(f"✓ Auto-detected HF_MODEL=google/{model_name}")
        # Note: Llama models don't need HF_MODEL set - they auto-detect correctly
    except (ValueError, IndexError):
        pass  # No --model argument or invalid format


def inject_defaults():
    """
    Inject sensible default parameters if not already provided.

    This makes the script much easier to use - just specify --model and you're good to go!
    All defaults can be overridden by passing the argument explicitly.

    Auto-injected parameters:
    - --served-model-name: Clean model name (e.g., Qwen/Qwen3-0.6B instead of path)
    - --max-model-len 2048: Reasonable context length for development
    - --max-num-seqs 16: Good balance for small models
    - --block-size 64: Standard KV cache block size

    Usage:
        # Minimal - just specify model, rest auto-configured:
        python start-vllm-server.py --model ~/models/Qwen3-0.6B

        # Override defaults as needed:
        python start-vllm-server.py --model ~/models/Qwen3-0.6B --max-model-len 8192
    """
    # Find --model argument to extract model name
    if '--model' not in sys.argv:
        return

    try:
        model_idx = sys.argv.index('--model')
        model_path = sys.argv[model_idx + 1]
        model_name = os.path.basename(model_path.rstrip('/'))

        # Auto-set --served-model-name if not provided
        if '--served-model-name' not in sys.argv:
            # Detect model org and set clean served name
            if 'Qwen' in model_name or 'qwen' in model_name.lower():
                served_name = f'Qwen/{model_name}'
            elif 'gemma' in model_name.lower():
                served_name = f'google/{model_name}'
            elif 'Llama' in model_name or 'llama' in model_name.lower():
                served_name = f'meta-llama/{model_name}'
            elif 'Mistral' in model_name or 'mistral' in model_name.lower():
                served_name = f'mistralai/{model_name}'
            else:
                # Fallback: use model name as-is
                served_name = model_name

            sys.argv.extend(['--served-model-name', served_name])
            print(f"✓ Auto-set --served-model-name={served_name}")

        # Set sensible defaults for vLLM parameters
        defaults = {
            '--max-model-len': '2048',    # Good for development, prevents OOM
            '--max-num-seqs': '16',       # Balanced for small models
            '--block-size': '64',         # Standard KV cache block size
        }

        for param, value in defaults.items():
            if param not in sys.argv:
                sys.argv.extend([param, value])
                print(f"✓ Auto-set {param}={value}")

    except (ValueError, IndexError):
        pass  # No --model argument or invalid format


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting vLLM Server with Auto-Configuration")
    print("=" * 60)

    # Step 1: Detect hardware and configure environment variables
    # This must happen FIRST so environment is ready for everything else
    detect_and_configure_hardware()

    # Step 2: Inject sensible defaults for easy usage
    # Makes minimal command work: python start-vllm-server.py --model ~/models/Qwen3-0.6B
    inject_defaults()

    # Step 3: Auto-detect HF_MODEL from --model path
    # This must happen before registering TT models
    auto_detect_hf_model()

    # Step 4: Register TT models with vLLM
    # This must happen before vLLM loads any models
    register_tt_models()

    print("=" * 60)
    print("✓ All checks complete - Starting vLLM API server...")
    print("=" * 60)

    # Start the vLLM API server
    # Command-line arguments are passed through with auto-injected defaults
    # Works with ANY model path passed via --model:
    #
    # Minimal usage (recommended):
    #   python start-vllm-server.py --model ~/models/Qwen3-0.6B
    #   → Auto-detects hardware (N150/N300/T3K/P100/P150)
    #   → Auto-sets MESH_DEVICE, TT_METAL_ARCH_NAME, TT_METAL_HOME
    #   → Auto-sets --served-model-name Qwen/Qwen3-0.6B
    #   → Auto-sets --max-model-len 2048, --max-num-seqs 16, --block-size 64
    #
    # Override defaults as needed:
    #   python start-vllm-server.py --model ~/models/Qwen3-8B --max-model-len 8192
    #
    # Override hardware detection:
    #   export MESH_DEVICE=N300  # Before running script
    #   python start-vllm-server.py --model ~/models/Qwen3-0.6B
    #
    # Supported models:
    #   ~/models/Qwen2.5-Coder-1.5B-Instruct  (✅ BEST for CODING on N150 - specialized for code!)
    #   ~/models/Qwen3-0.6B                   (✅ BEST for GENERAL on N150 - tiny, fast, smart!)
    #   ~/models/Llama-3.2-3B-Instruct        (Good for N150 - versatile, larger)
    #   ~/models/gemma-3-1b-it                (Good for N150 - small, multilingual)
    #   ~/models/Llama-3.1-8B-Instruct        (Needs N300+ - higher DRAM requirements)
    #   ~/models/Qwen2.5-Coder-7B-Instruct    (Needs N300+ - powerful coding model)
    #   ~/models/Qwen3-8B                     (Needs N300+ - more DRAM)
    #   ~/models/Mistral-7B-Instruct-v0.3     (Needs N300+ - more DRAM)
    #
    # The model architecture is auto-detected from config.json
    # HF_MODEL is auto-detected from your --model path (or manually set via export HF_MODEL=...)
    # If it's Llama-compatible, it uses the TT-optimized implementation!
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
