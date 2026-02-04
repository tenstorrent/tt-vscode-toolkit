#!/usr/bin/env python3
"""
Custom Training Startup Validation
Tests all prerequisites before running training.
Use for CT4-CT8 lessons.

Usage:
    python test_training_startup.py [--config CONFIG_PATH] [--dataset DATASET_PATH]
"""

import sys
import os
from pathlib import Path

def test_environment():
    """Test environment variables"""
    print("🔍 Testing environment variables...")

    tt_metal_home = os.environ.get('TT_METAL_HOME')
    if not tt_metal_home:
        print("  ❌ TT_METAL_HOME not set")
        print("     Run: source setup_training_env.sh")
        return False

    if not Path(tt_metal_home).exists():
        print(f"  ❌ TT_METAL_HOME points to non-existent directory: {tt_metal_home}")
        return False

    print(f"  ✅ TT_METAL_HOME: {tt_metal_home}")

    mesh_device = os.environ.get('MESH_DEVICE')
    if mesh_device:
        print(f"  ✅ MESH_DEVICE: {mesh_device}")
    else:
        print("  ⚠️  MESH_DEVICE not set (will default to N150)")

    return True

def test_imports():
    """Test critical imports"""
    print("\n🔍 Testing imports...")

    all_ok = True

    try:
        import ttml
        print("  ✅ ttml")
    except ImportError as e:
        print(f"  ❌ ttml: {e}")
        print("     Fix: cd $TT_METAL_HOME/tt-train && pip install -e .")
        all_ok = False

    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")
        print("     Fix: pip install torch")
        all_ok = False

    try:
        import transformers
        print(f"  ✅ transformers {transformers.__version__}")
    except ImportError:
        print("  ⚠️  transformers not installed")
        print("     Note: Required for CT4 fine-tuning")
        print("     Fix: pip install transformers")
        # Don't fail - only needed for CT4

    try:
        import yaml
        print(f"  ✅ yaml")
    except ImportError:
        print("  ⚠️  yaml not installed")
        print("     Fix: pip install pyyaml")

    return all_ok

def test_config(config_path="configs/trickster_n150.yaml"):
    """Test config loading"""
    if not Path(config_path).exists():
        print(f"\n⚠️  Config {config_path} not found (skipping - OK if not using)")
        return True

    print(f"\n🔍 Testing config: {config_path}...")

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        sections = len(config)
        print(f"  ✅ Config loaded: {sections} sections")

        # Check for expected sections
        expected = ['training_config', 'model_config', 'device_config']
        for section in expected:
            if section in config:
                print(f"     ✅ {section}")
            else:
                print(f"     ⚠️  {section} missing")

        return True
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False

def test_dataset(dataset_path="trickster_dataset_starter.jsonl"):
    """Test dataset loading"""
    if not Path(dataset_path).exists():
        print(f"\n⚠️  Dataset {dataset_path} not found (skipping - OK if not using)")
        return True

    print(f"\n🔍 Testing dataset: {dataset_path}...")

    try:
        import json
        with open(dataset_path) as f:
            examples = [json.loads(line) for line in f]
        print(f"  ✅ Dataset loaded: {len(examples)} examples")

        # Validate format
        if examples and isinstance(examples[0], dict):
            keys = examples[0].keys()
            print(f"     Keys: {list(keys)}")

        return True
    except Exception as e:
        print(f"  ❌ Dataset error: {e}")
        return False

def test_library_conflicts():
    """Test for common library conflicts"""
    print("\n🔍 Testing for library conflicts...")

    try:
        # Check if pip-installed ttnn exists (conflicts with local build)
        import subprocess
        result = subprocess.run(
            ['pip', 'list'],
            capture_output=True,
            text=True
        )

        if 'ttnn' in result.stdout:
            print("  ⚠️  pip-installed 'ttnn' detected - may conflict with local tt-metal build")
            print("     Fix: pip uninstall -y ttnn")
            return False
        else:
            print("  ✅ No conflicting pip packages")
            return True
    except Exception:
        print("  ⚠️  Could not check for conflicts")
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Custom Training prerequisites")
    parser.add_argument("--config", default="configs/nano_trickster.yaml", help="Config file path (optional)")
    parser.add_argument("--dataset", default="data/train.pt", help="Dataset file path (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("🎭 Custom Training Startup Validation")
    print("=" * 60)

    tests = [
        test_environment(),
        test_imports(),
        test_library_conflicts(),
        test_config(args.config),
        test_dataset(args.dataset),
    ]

    print("\n" + "=" * 60)
    if all(tests):
        print("✅ All critical tests passed! Ready to train.")
        print("\n🚀 Next steps:")
        print("   CT4 Fine-tuning: python finetune_trickster.py --config configs/trickster_n150.yaml")
        print("   CT8 From-scratch: python train_from_scratch.py --config configs/nano_trickster.yaml")
        return 0
    else:
        print("❌ Some tests failed. Fix errors above before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
