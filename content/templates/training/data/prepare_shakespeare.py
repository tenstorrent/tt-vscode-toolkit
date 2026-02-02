#!/usr/bin/env python3
"""
Tiny Shakespeare Dataset Preparation

Downloads and prepares the tiny-shakespeare dataset for training
a small transformer from scratch.

Dataset: ~1MB of Shakespeare text (~40k lines, ~1M characters)
Perfect for learning fundamentals without waiting days.

Usage:
    python prepare_shakespeare.py --output shakespeare.txt
"""

import argparse
import urllib.request
from pathlib import Path


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(output_path: Path):
    """
    Download tiny-shakespeare dataset.

    Args:
        output_path: Where to save the text file
    """

    print("=" * 60)
    print("📚 Downloading Tiny Shakespeare Dataset")
    print("=" * 60)
    print()
    print(f"Source: {SHAKESPEARE_URL}")
    print(f"Destination: {output_path}")
    print()

    try:
        print("Downloading...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, output_path)

        # Get file size
        size_bytes = output_path.stat().st_size
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024

        print(f"✅ Download complete!")
        print(f"   Size: {size_mb:.2f} MB ({size_kb:.0f} KB, {size_bytes:,} bytes)")
        print()

        # Show statistics
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()

        num_chars = len(text)
        num_lines = text.count('\n')
        unique_chars = len(set(text))

        print("📊 Dataset Statistics:")
        print(f"   Total characters: {num_chars:,}")
        print(f"   Total lines: {num_lines:,}")
        print(f"   Unique characters: {unique_chars}")
        print(f"   First 200 characters:")
        print()
        print("   " + "-" * 56)
        print("   " + text[:200].replace('\n', '\n   '))
        print("   " + "-" * 56)
        print()

        print("=" * 60)
        print("✅ Dataset ready for training!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

    return True


def create_train_val_split(input_path: Path, train_path: Path, val_path: Path, val_ratio: float = 0.1):
    """
    Split dataset into training and validation sets.

    Args:
        input_path: Full dataset path
        train_path: Training set output path
        val_path: Validation set output path
        val_ratio: Fraction for validation (default: 0.1 = 10%)
    """

    print()
    print("Splitting into train/val...")

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split at val_ratio point
    split_idx = int(len(text) * (1 - val_ratio))

    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Write splits
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)

    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)

    print(f"✅ Training set: {len(train_text):,} characters → {train_path}")
    print(f"✅ Validation set: {len(val_text):,} characters → {val_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download and prepare tiny-shakespeare dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="shakespeare.txt",
        help="Output file path for full dataset",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also create train/val split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1 = 10%%)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    success = download_shakespeare(output_path)
    if not success:
        return 1

    # Optionally split
    if args.split:
        train_path = output_path.parent / (output_path.stem + "_train.txt")
        val_path = output_path.parent / (output_path.stem + "_val.txt")
        create_train_val_split(output_path, train_path, val_path, args.val_ratio)

    print("🎭 Tip: This dataset is perfect for learning!")
    print("   - Small enough to train in 30-60 minutes")
    print("   - Large enough to learn interesting patterns")
    print("   - Classic text everyone recognizes")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
