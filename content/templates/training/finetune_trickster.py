#!/usr/bin/env python3
"""
Trickster Fine-tuning Script

Fine-tunes TinyLlama-1.1B to become a creative, flexible AI for explaining concepts.
Demonstrates principles applicable to any custom model fine-tuning task.

Compatibility: tt-metal v0.64.5+
    - Uses constant learning rate (no dynamic scheduling in v0.64.5)
    - Extends BaseTrainingConfig (no SchedulerConfig class)
    - AdamW optimizer with gradient accumulation

Usage:
    python finetune_trickster.py --config configs/trickster_n150.yaml
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

import ttnn
import ttml
from ttml.common.config import (
    TrainingConfig as BaseTrainingConfig,
    DeviceConfig,
    load_config,
    yaml_deep_update,
)
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import (
    round_up_to_tile,
    initialize_device,
    create_optimizer,
    get_loss_over_devices,
    build_logits_mask,
    no_grad,
    get_tt_metal_home,
)
from ttml.common.data import build_causal_mask


class TrainingConfig(BaseTrainingConfig):
    """
    Extended training config for v0.64.5 compatibility.

    Adds scheduler_type field (no SchedulerConfig class in v0.64.5).
    """
    def __init__(self, yaml_config=None):
        super().__init__(yaml_config if yaml_config is not None else {})

        tc = yaml_config.get("training_config", {}) if isinstance(yaml_config, dict) else {}

        # Extended fields not in base TrainingConfig
        # Note: v0.64.5 doesn't support dynamic LR scheduling
        # scheduler_type can be "identity" (constant) or "warmup_linear" (future)
        self.scheduler_type = tc.get("scheduler_type", "identity")

        # Add validation_batch_size if not present
        if not hasattr(self, 'validation_batch_size'):
            self.validation_batch_size = tc.get("validation_batch_size", self.batch_size)

        # Add validation_frequency if not present
        if not hasattr(self, 'validation_frequency'):
            self.validation_frequency = tc.get("validation_frequency", 100)

        # Add checkpoint_frequency if not present
        if not hasattr(self, 'checkpoint_frequency'):
            self.checkpoint_frequency = tc.get("checkpoint_frequency", 500)


class TricksterDataset(Dataset):
    """
    Dataset for Trickster model - creative, flexible AI for explaining concepts.

    Loads JSONL format with {prompt, response} pairs.
    """

    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer):
        self.examples = []
        self.tokenizer = tokenizer

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                self.examples.append(example)

        print(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format as instruction-following
        prompt = f"### Question:\n{example['prompt']}\n\n### Answer:\n"
        response = example['response']

        # Tokenize (without special tokens, we'll add them in collate)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        return prompt_tokens, response_tokens


class CollateFn:
    """
    Collate function for batching Trickster examples.

    Pads/truncates to max_sequence_length and creates loss mask
    (only compute loss on response tokens, not prompts).
    """

    def __init__(
        self, eos_token_id: int, max_sequence_length: int, padded_vocab_size: int
    ):
        self.eos_token_id = eos_token_id
        self.max_sequence_length = max_sequence_length
        self.padded_vocab_size = padded_vocab_size

    def __call__(self, batch):
        prompts, responses = map(list, zip(*batch))

        batch_size = len(prompts)

        # Preallocate with padding tokens
        data_np = np.full(
            (batch_size, self.max_sequence_length), self.eos_token_id, dtype=np.uint32
        )
        mask_lens = []  # Track where prompt ends (don't compute loss on prompts)

        for i, (prompt_tokens, response_tokens) in enumerate(zip(prompts, responses)):
            prompt_len = len(prompt_tokens)
            response_len = len(response_tokens)
            total_len = prompt_len + response_len
            max_len = self.max_sequence_length

            if total_len > max_len:
                # Truncate: prioritize response over prompt
                available_space = max_len - response_len

                if available_space > 0:
                    # Keep full response, truncate prompt
                    prompt_tokens = prompt_tokens[:available_space]
                    prompt_len = available_space
                    data_np[i, :prompt_len] = prompt_tokens
                    data_np[i, prompt_len : prompt_len + response_len] = response_tokens
                else:
                    # Response alone too long: truncate response
                    response_tokens = response_tokens[:max_len]
                    response_len = max_len
                    data_np[i, :response_len] = response_tokens
                    prompt_len = 0
            else:
                # Normal case: concatenate prompt + response
                data_np[i, :prompt_len] = prompt_tokens
                data_np[i, prompt_len : prompt_len + response_len] = response_tokens

            mask_lens.append(prompt_len)

        # Shape for model input: [batch_size, 1, 1, max_sequence_length]
        X_np = np.expand_dims(data_np, axis=(1, 2))

        # Target: shifted left by 1 (predict next token)
        y_np = np.full(
            (batch_size, self.max_sequence_length), self.eos_token_id, dtype=np.uint32
        )
        y_np[:, 0:-1] = X_np[:, 0, 0, 1:]

        # Loss scaler: zero out prompt tokens and padding
        loss_scaler_np = np.full(
            (batch_size, 1, self.max_sequence_length, 1), 1.0, dtype=np.float32
        )
        for i, mask_len in enumerate(mask_lens):
            # Don't compute loss on prompt
            loss_scaler_np[i, :, :mask_len, :] = 0.0
            # Don't compute loss on padding
            pad_positions = X_np[i, 0, 0, :] == self.eos_token_id
            loss_scaler_np[i, :, pad_positions, :] = 0.0

        # Normalize loss scaler (sum to batch_size * seq_len)
        loss_scaler_ratio = (
            self.max_sequence_length * batch_size / np.sum(loss_scaler_np)
        )
        loss_scaler_np *= loss_scaler_ratio

        return X_np, y_np, loss_scaler_np


def get_batch_generator(dataloader: DataLoader, device_config=None):
    """Generate batches as TT tensors."""
    mapper = None
    if device_config is not None and device_config.total_devices() > 1:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)

    while True:
        for X_np, y_np, loss_scaler_np in dataloader:
            X = ttml.autograd.Tensor.from_numpy(
                X_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32, mapper=mapper
            )
            y = ttml.autograd.Tensor.from_numpy(
                y_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32, mapper=mapper
            )
            loss_scaler = ttml.autograd.Tensor.from_numpy(
                loss_scaler_np,
                layout=ttnn.Layout.TILE,
                new_type=ttnn.DataType.BFLOAT16,
                mapper=mapper,
            )

            yield (X, y, loss_scaler)


def generate_trickster_response(
    model,
    tokenizer: AutoTokenizer,
    question: str,
    max_sequence_length: int,
    causal_mask: ttml.autograd.Tensor,
    logits_mask_tensor: ttml.autograd.Tensor,
    max_gen_tokens: int = 128,
):
    """
    Generate a Trickster response to a question.

    Uses greedy decoding (temperature=0).
    """
    model.eval()

    # Format prompt
    prompt_text = f"### Question:\n{question}\n\n### Answer:\n"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    pad_token_id = tokenizer.eos_token_id

    generated_tokens = []

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Preallocate buffer
    padded_prompt_tokens = np.full(
        (1, 1, 1, max_sequence_length), pad_token_id, dtype=np.uint32
    )

    with no_grad():
        for _ in range(max_gen_tokens):
            # Sliding window for long prompts
            if len(prompt_tokens) > max_sequence_length:
                start_idx = len(prompt_tokens) - max_sequence_length
                window = prompt_tokens[start_idx:]
            else:
                window = prompt_tokens

            # Fill buffer
            padded_prompt_tokens[...] = pad_token_id
            padded_prompt_tokens[0, 0, 0, : len(window)] = np.asarray(
                window, dtype=np.uint32
            )

            # Forward pass
            padded_prompt_tensor = ttml.autograd.Tensor.from_numpy(
                padded_prompt_tokens,
                layout=ttnn.Layout.ROW_MAJOR,
                new_type=ttnn.DataType.UINT32,
            )
            logits = model(padded_prompt_tensor, causal_mask)

            # Greedy sampling
            next_token_tensor = ttml.ops.sample.sample_op(
                logits, 0.0, np.random.randint(low=1e7), logits_mask_tensor
            )

            # Extract token at last position
            next_token_idx = (
                max_sequence_length - 1
                if len(prompt_tokens) > max_sequence_length
                else len(window) - 1
            )
            next_token = int(
                next_token_tensor.to_numpy(composer=composer).reshape(-1, 1)[
                    next_token_idx
                ][0]
            )

            if next_token == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            prompt_tokens.append(next_token)

    model.train()

    return tokenizer.decode(generated_tokens)


def validate(
    tt_model,
    tokenizer: AutoTokenizer,
    val_batch_generator,
    val_dataset: TricksterDataset,
    loss_fn,
    causal_mask: ttml.autograd.Tensor,
    logits_mask_tensor: ttml.autograd.Tensor,
    max_sequence_length: int,
    current_step: int,
    output_file: str = "validation.txt",
):
    """
    Validation: compute loss and generate sample responses.
    """
    reduce = ttml.ops.ReduceType.NONE

    tt_model.eval()

    with no_grad():
        eval_batch_count = 4
        cur_val_losses = []
        for _ in range(eval_batch_count):
            val_X, val_y, val_loss_scaler = next(val_batch_generator)
            val_logits = tt_model(val_X, causal_mask)

            val_loss = loss_fn(val_logits, val_y, reduce)
            val_loss = val_loss * val_loss_scaler
            val_loss = ttml.ops.unary.mean(val_loss)
            cur_val_losses.append(get_loss_over_devices(val_loss))

    # Generate sample responses
    test_questions = [
        "What is a neural network?",
        "How do I learn machine learning?",
        "Explain what backpropagation does",
        "What's the difference between AI and ML?",
    ]

    with open(output_file, "a+") as val_file:
        val_file.write(f"\n{'='*60}\n")
        val_file.write(f"Validation at step {current_step}\n")
        val_file.write(f"{'='*60}\n\n")

        for i, question in enumerate(test_questions):
            val_file.write(f"Q{i+1}: {question}\n")
            response = generate_trickster_response(
                tt_model,
                tokenizer,
                question,
                max_sequence_length,
                causal_mask,
                logits_mask_tensor,
            )
            val_file.write(f"A{i+1}: {response}\n\n")

        val_file.write(
            f"Validation loss: {float(np.mean(cur_val_losses)):.4f}\n"
        )

    tt_model.train()
    return np.mean(cur_val_losses)


def train(args):
    """
    Main training loop for Trickster fine-tuning.
    """

    print("="*60)
    print("🎭 Trickster Fine-tuning")
    print("="*60)
    print()

    # Load config
    print(f"Loading config: {args.config}")
    yaml_config = load_config(args.config)
    model_config = load_config(yaml_config["training_config"]["model_config"])

    # Apply overrides if present
    override_config_path = Path(os.environ.get('TT_METAL_HOME', '')) / "tt-train" / "configs" / "training_overrides.yaml"
    if override_config_path.exists():
        print(f"Applying overrides from {override_config_path}")
        override_config = load_config(str(override_config_path))
        yaml_config = yaml_deep_update(yaml_config, override_config)
        model_config = yaml_deep_update(model_config, override_config)

    training_config = TrainingConfig(yaml_config)
    device_config = DeviceConfig(yaml_config)

    # Initialize device (for multi-device)
    if device_config.total_devices() > 1:
        print(f"Initializing {device_config.total_devices()} devices...")
        initialize_device(yaml_config)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    orig_vocab_size = tokenizer.vocab_size
    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    max_sequence_length = tt_model_factory.transformer_config.max_sequence_length

    tt_model = tt_model_factory.create_model()

    # Load weights
    if args.weights_path:
        print(f"Loading weights from {args.weights_path}")
        tt_model.load_from_safetensors(args.weights_path)
    else:
        print("⚠️  No weights specified - starting from random initialization!")

    padded_vocab_size = round_up_to_tile(orig_vocab_size, 32)

    # Load dataset
    print(f"Loading training dataset: {args.train_data}")
    train_dataset = TricksterDataset(args.train_data, tokenizer)

    val_dataset = None
    if args.val_data:
        print(f"Loading validation dataset: {args.val_data}")
        val_dataset = TricksterDataset(args.val_data, tokenizer)
    else:
        print("⚠️  No validation data - skipping validation")

    # Create dataloaders
    batch_size = training_config.batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=CollateFn(tokenizer.eos_token_id, max_sequence_length, padded_vocab_size),
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_config.validation_batch_size * device_config.total_devices(),
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=CollateFn(tokenizer.eos_token_id, max_sequence_length, padded_vocab_size),
        )

    # Setup training
    optim = create_optimizer(tt_model, yaml_config)
    causal_mask = build_causal_mask(max_sequence_length)
    causal_mask = ttml.autograd.Tensor.from_numpy(
        causal_mask, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.BFLOAT16
    )
    logits_mask_tensor = build_logits_mask(orig_vocab_size, padded_vocab_size)
    loss_fn = ttml.ops.loss.cross_entropy_loss

    # Initialize training state
    tt_model.train()
    train_losses = []
    val_losses = []

    # Prepare output files
    os.makedirs(args.output_dir, exist_ok=True)
    val_output = Path(args.output_dir) / "validation.txt"
    if val_output.exists():
        val_output.unlink()

    with open(val_output, "w") as f:
        f.write("Trickster Fine-tuning - Validation Log\n")
        f.write("="*60 + "\n\n")

    # Create batch generators
    train_batch_generator = get_batch_generator(train_dataloader, device_config)
    val_batch_generator = None
    if val_dataloader:
        val_batch_generator = get_batch_generator(val_dataloader, device_config)

    # Training info
    tokens_per_batch = batch_size * max_sequence_length
    accum_steps = training_config.gradient_accumulation_steps
    print()
    print(f"Training configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset) if val_dataset else 0}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {accum_steps}")
    print(f"  Effective batch size: {batch_size * accum_steps}")
    print(f"  Max sequence length: {max_sequence_length}")
    print(f"  Training steps: {training_config.steps}")
    print(f"  Devices: {device_config.total_devices()}")
    print(f"  Learning rate: {training_config.lr} (constant)")
    print(f"  Scheduler type: {training_config.scheduler_type}")
    if training_config.scheduler_type != "identity":
        print("  ⚠️  Note: v0.64.5 only supports constant LR (scheduler_type='identity')")
    print()

    # Training loop
    print("Starting training...")
    print("="*60)
    bar = tqdm(range(1, training_config.steps + 1), desc="Training")

    for opt_step in bar:
        # Zero gradients
        optim.zero_grad()

        # Gradient accumulation loop
        micro_losses = []
        for _ in range(accum_steps):
            X, y, loss_scaler = next(train_batch_generator)

            # Forward
            logits = tt_model(X, causal_mask)

            # Compute loss
            loss = loss_fn(logits, y, ttml.ops.ReduceType.NONE)
            loss = loss * loss_scaler
            loss = ttml.ops.unary.mean(loss)

            micro_losses.append(get_loss_over_devices(loss))

            # Backward (scaled for accumulation)
            scaled_loss = ttml.ops.binary.mul(loss, 1.0 / float(accum_steps))
            scaled_loss.backward()

        # Optimizer step
        optim.step()

        # Record loss
        avg_loss = float(np.mean(micro_losses))
        train_losses.append(avg_loss)

        # Update progress bar
        bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Validation
        if val_batch_generator and opt_step % training_config.validation_frequency == 0:
            val_loss = validate(
                tt_model,
                tokenizer,
                val_batch_generator,
                val_dataset,
                loss_fn,
                causal_mask,
                logits_mask_tensor,
                max_sequence_length,
                opt_step,
                str(val_output),
            )
            val_losses.append(val_loss)
            print(f"\nValidation loss: {val_loss:.4f}\n")

        # Save checkpoint
        if opt_step % training_config.checkpoint_frequency == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_step_{opt_step}"
            os.makedirs(checkpoint_path, exist_ok=True)
            print(f"\nSaving checkpoint to {checkpoint_path}")
            tt_model.save_to_safetensors(str(checkpoint_path))

    # Save final model
    final_path = Path(args.output_dir) / "final_model"
    os.makedirs(final_path, exist_ok=True)
    print(f"\nSaving final model to {final_path}")
    tt_model.save_to_safetensors(str(final_path))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", alpha=0.7)
    if val_losses:
        val_steps = [i * training_config.validation_frequency for i in range(1, len(val_losses) + 1)]
        plt.plot(val_steps, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Trickster Fine-tuning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(args.output_dir) / "training_curves.png")
    print(f"\nTraining curves saved to {args.output_dir}/training_curves.png")

    print("\n" + "="*60)
    print("✅ Fine-tuning complete!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Model saved to: {final_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama on Trickster dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="trickster_dataset_starter.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation JSONL file (optional)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        help="HuggingFace model name for tokenizer",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to pre-trained weights (safetensors directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for checkpoints and logs",
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
