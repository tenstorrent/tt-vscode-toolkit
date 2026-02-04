#!/usr/bin/env python3
"""
Preprocess Shakespeare text files into PyTorch tensors for training.
Converts shakespeare_train.txt and shakespeare_val.txt into train.pt and val.pt

Usage:
    python preprocess_shakespeare.py
"""

import sys
from pathlib import Path

def create_tokenizer(text):
    """Create character-level tokenizer"""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"  Vocabulary size: {vocab_size} characters")
    if len(chars) <= 30:
        print(f"  Characters: {chars}")
    else:
        print(f"  Sample characters: {chars[:30]}...")

    return vocab_size, stoi, itos

def encode_text(text, stoi):
    """Encode text to token IDs"""
    return [stoi[c] for c in text]

def main():
    try:
        import torch
    except ImportError:
        print("❌ Error: PyTorch not found")
        print("   Fix: pip install torch")
        sys.exit(1)

    data_dir = Path(".")

    # Check for input files
    train_file = data_dir / "shakespeare_train.txt"
    val_file = data_dir / "shakespeare_val.txt"

    if not train_file.exists() or not val_file.exists():
        print("❌ Error: shakespeare_train.txt or shakespeare_val.txt not found")
        print("   Run: python prepare_shakespeare.py --output . --split")
        sys.exit(1)

    print("📖 Loading text files...")
    train_text = train_file.read_text()
    val_text = val_file.read_text()

    print(f"  Train: {len(train_text):,} characters")
    print(f"  Val: {len(val_text):,} characters")

    print("\n🔤 Creating character-level tokenizer...")
    vocab_size, stoi, itos = create_tokenizer(train_text)

    print("\n🔢 Encoding text to token IDs...")
    train_ids = encode_text(train_text, stoi)
    val_ids = encode_text(val_text, stoi)

    print(f"  Train: {len(train_ids):,} tokens")
    print(f"  Val: {len(val_ids):,} tokens")

    print("\n💾 Converting to PyTorch tensors...")
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)

    print(f"  Train tensor shape: {train_tensor.shape}")
    print(f"  Val tensor shape: {val_tensor.shape}")

    print("\n💾 Saving tensors...")
    torch.save(train_tensor, data_dir / "train.pt")
    torch.save(val_tensor, data_dir / "val.pt")

    # Save tokenizer metadata
    tokenizer = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }
    torch.save(tokenizer, data_dir / "tokenizer.pt")

    print(f"  ✅ Saved train.pt ({train_tensor.numel():,} tokens)")
    print(f"  ✅ Saved val.pt ({val_tensor.numel():,} tokens)")
    print(f"  ✅ Saved tokenizer.pt (vocab_size={vocab_size})")

    print("\n✅ Preprocessing complete!")
    print(f"   Ready for training with {vocab_size} character vocabulary")
    print("\n🚀 Next step: python train_from_scratch.py --config configs/nano_trickster.yaml")

if __name__ == "__main__":
    main()
