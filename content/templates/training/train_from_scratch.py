"""
Train Nano-Trickster from Scratch

This script trains a tiny transformer (11M parameters) from random initialization
on the tiny-shakespeare dataset using character-level tokenization.

Based on: nano_trickster.py architecture + tt-train patterns

Usage:
    python train_from_scratch.py --config configs/nano_trickster.yaml

Hardware:
    - N150 (single chip): 30-60 minutes
    - N300 (dual chip): 15-30 minutes (with DDP)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

# Import nano-trickster architecture
from nano_trickster import NanoTrickster, count_parameters


class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset"""

    def __init__(self, data_path: str, seq_len: int = 512):
        self.seq_len = seq_len

        # Load tokenized data
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Run: python data/prepare_shakespeare.py"
            )

        self.data = torch.load(data_path)
        print(f"Loaded {len(self.data):,} tokens from {data_path}")

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1  # +1 for labels

        # Get chunk
        chunk = self.data[start_idx:end_idx]

        # Pad if necessary (shouldn't happen with proper dataset)
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)), value=0)

        # Input and labels
        input_ids = chunk[:-1]
        labels = chunk[1:]

        return {"input_ids": input_ids, "labels": labels}


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["training_config"]["learning_rate"],
        betas=(
            config["training_config"]["beta1"],
            config["training_config"]["beta2"],
        ),
        weight_decay=config["training_config"]["weight_decay"],
    )


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup"""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    elif step > max_steps:
        # Constant minimum
        return max_lr * 0.1
    else:
        # Cosine decay
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * torch.pi)))
        return max_lr * 0.1 + (max_lr - max_lr * 0.1) * coeff


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> float:
    """Single training step"""
    model.train()

    # Move to device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    # Backward
    loss.backward()

    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


@torch.no_grad()
def eval_step(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Evaluation on validation set"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"loss": avg_loss, "perplexity": perplexity}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    output_dir: str,
    config: dict,
) -> str:
    """Save training checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }, checkpoint_path)

    return checkpoint_path


def generate_sample(
    model: torch.nn.Module,
    tokenizer: dict,
    prompt: str = "ROMEO:",
    max_length: int = 200,
    temperature: float = 0.8,
) -> str:
    """Generate text sample to monitor training progress"""
    model.eval()

    # Encode prompt
    stoi = tokenizer["stoi"]
    input_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)

    # Generate
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=max_length, temperature=temperature)

    # Decode
    itos = tokenizer["itos"]
    text = ''.join([itos.get(int(t), '?') for t in generated[0]])

    return text


def train(config_path: str, resume_from: Optional[str] = None):
    """Main training loop"""
    # Load config
    config = load_config(config_path)
    print("=" * 60)
    print("Training Nano-Trickster from Scratch")
    print("=" * 60)
    print(f"\nConfig: {config_path}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = NanoTrickster(**config["model_config"]).to(device)

    # Count parameters
    params = count_parameters(model)
    print(f"\nModel architecture:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Per block: {params['per_block']:,}")

    # Load tokenizer
    tokenizer_path = config["data_config"]["tokenizer"]
    tokenizer = torch.load(tokenizer_path)
    print(f"  Vocabulary size: {tokenizer['vocab_size']}")

    # Create datasets
    train_dataset = ShakespeareDataset(
        config["data_config"]["train_data"],
        seq_len=config["model_config"]["max_seq_len"],
    )
    val_dataset = ShakespeareDataset(
        config["data_config"]["val_data"],
        seq_len=config["model_config"]["max_seq_len"],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training_config"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training_config"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print(f"\nDataset:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Resume from checkpoint if requested
    start_step = 0
    if resume_from:
        print(f"\nResuming from: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        print(f"  Resuming from step {start_step}")

    # Training config
    max_steps = config["training_config"]["max_steps"]
    warmup_steps = config["training_config"]["warmup_steps"]
    log_interval = config["training_config"]["log_interval"]
    eval_interval = config["training_config"]["eval_interval"]
    save_interval = config["training_config"]["save_interval"]
    grad_clip = config["training_config"]["grad_clip"]
    max_lr = config["training_config"]["learning_rate"]

    # Create output directories
    output_dir = config["output_config"]["output_dir"]
    checkpoint_dir = config["output_config"]["checkpoint_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nTraining:")
    print(f"  Max steps: {max_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {max_lr}")
    print(f"  Gradient clip: {grad_clip}")
    print(f"  Output: {output_dir}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    step = start_step
    best_val_loss = float('inf')
    train_iter = iter(train_loader)

    with tqdm(total=max_steps, initial=start_step, desc="Training") as pbar:
        while step < max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Update learning rate
            lr = get_lr(step, warmup_steps, max_steps, max_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Training step
            loss = train_step(model, batch, optimizer, grad_clip, device)

            # Logging
            if step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.6f}"})

            # Evaluation
            if step % eval_interval == 0 and step > 0:
                val_metrics = eval_step(model, val_loader, device)
                print(f"\nStep {step}:")
                print(f"  Train loss: {loss:.4f}")
                print(f"  Val loss: {val_metrics['loss']:.4f}")
                print(f"  Val perplexity: {val_metrics['perplexity']:.2f}")

                # Generate sample
                sample = generate_sample(model, tokenizer, prompt="ROMEO:", max_length=100)
                print(f"\n  Sample generation:")
                print("  " + "-" * 56)
                print("  " + sample[:200].replace('\n', '\n  '))
                print("  " + "-" * 56 + "\n")

                # Save best
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_path = save_checkpoint(
                        model, optimizer, step, val_metrics['loss'], checkpoint_dir, config
                    )
                    print(f"  ✓ New best checkpoint: {best_path}\n")

            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                checkpoint_path = save_checkpoint(
                    model, optimizer, step, loss, checkpoint_dir, config
                )
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

            step += 1
            pbar.update(1)

    # Final save
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training complete! Final model saved: {final_path}")

    # Final evaluation
    print("\nFinal evaluation:")
    val_metrics = eval_step(model, val_loader, device)
    print(f"  Val loss: {val_metrics['loss']:.4f}")
    print(f"  Val perplexity: {val_metrics['perplexity']:.2f}")

    # Generate final sample
    print("\nFinal sample generation:")
    print("=" * 60)
    sample = generate_sample(model, tokenizer, prompt="ROMEO:", max_length=300)
    print(sample)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Nano-Trickster from scratch")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config (YAML)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    try:
        train(args.config, resume_from=args.resume)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
