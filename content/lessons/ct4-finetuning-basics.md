---
id: ct4-finetuning-basics
title: Fine-tuning Basics
description: >-
  Run your first fine-tuning job on Tenstorrent hardware. Transform TinyLlama into tt-trickster, a specialized model for explaining ML concepts. Learn to monitor training, understand loss curves, and test fine-tuned models.
category: custom-training
tags:
  - fine-tuning
  - tt-train
  - training
  - tinyllama
  - loss-curves
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
---

# Fine-tuning Basics

Run your first fine-tuning job! Take TinyLlama and transform it into the tt-trickster model - a creative AI for explaining machine learning concepts.

## What You'll Learn

- Installing tt-train framework
- Launching a fine-tuning job
- Monitoring training progress
- Understanding loss curves
- Loading and testing fine-tuned models
- Troubleshooting common issues

**Time:** 20-25 minutes (setup) + 1-3 hours (training)
**Prerequisites:** CT-2 (Dataset), CT-3 (Configuration)

---

## Overview: What We're Building

**Input:** TinyLlama-1.1B (general language model)
**+ Training:** 50 examples of creative ML explanations
**= Output:** tt-trickster (specialized for explaining concepts)

**Before fine-tuning:**
```
Q: What is a neural network?
A: A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.
```
*(Generic, textbook-style)*

**After fine-tuning:**
```
Q: What is a neural network?
A: Imagine teaching a child to recognize cats by showing them thousands of cat pictures. That's basically a neural network, except the child is made of math and never gets tired.
```
*(Creative, approachable, memorable)*

---

## Step 1: Install tt-train

tt-train is TT-Metal's Python training framework. Install it first.

```command
tenstorrent.installTtTrain
```

**What this does:**
1. Verifies tt-metal is installed
2. Navigates to `$TT_METAL_HOME/tt-train`
3. Installs Python package: `pip install -e .`

**Expected output:**
```
Successfully installed ttml-0.1.0
```

**If installation fails:**
- Check that `TT_METAL_HOME` is set: `echo $TT_METAL_HOME`
- Verify tt-metal is built: `ls $TT_METAL_HOME/build/lib`
- Try building tt-metal: `cd $TT_METAL_HOME && ./build_metal.sh`

---

## Step 2: Get the Training Dataset

Copy the tt-trickster starter dataset to your workspace.

```command
tenstorrent.createTricksterDataset
```

**What this does:**
- Copies `trickster_dataset_starter.jsonl` to `~/tt-scratchpad/training/`
- 50 examples of creative ML explanations
- JSONL format (ready for training)

**View the dataset:**

```command
tenstorrent.viewTricksterDataset
```

This opens the JSONL file so you can browse the examples.

**Dataset structure:**
```jsonl
{"prompt": "What is a neural network?", "response": "Imagine teaching a child..."}
{"prompt": "How do I learn to code?", "response": "Start by breaking things..."}
...
```

---

## Step 3: Verify Configuration

The training configuration is already set up in:
- **N150:** `configs/trickster_n150.yaml`
- **N300:** `configs/trickster_n300.yaml`

Let's review key settings:

### N150 Configuration Highlights

```yaml
training_config:
  batch_size: 8                    # Conservative for DRAM
  learning_rate: 0.0001            # Fine-tuning LR
  max_steps: 500                   # ~1-3 hours on N150
  gradient_accumulation_steps: 4   # Effective batch = 32

  validation_frequency: 50         # Check progress every 50 steps
  checkpoint_frequency: 100        # Save every 100 steps

device_config:
  enable_ddp: False                # Single device
  mesh_shape: [1, 1]
```

**Training will:**
- Run for 500 steps (~80 epochs over 50 examples)
- Validate every 50 steps (10 validations total)
- Save checkpoints at steps 100, 200, 300, 400, 500
- Take 1-3 hours on N150

### N300 Configuration (Optional)

If you have N300, you can use DDP for ~2x speedup:

```yaml
training_config:
  batch_size: 16                   # Larger batch with DDP
  gradient_accumulation_steps: 2   # Same effective batch = 32

device_config:
  enable_ddp: True                 # Distributed training
  mesh_shape: [1, 2]               # Two devices
```

**Training will:**
- Complete in 30-60 minutes (2x faster)
- Use both Wormhole chips
- Produce identical results to N150

---

## Step 4: Download Pre-trained Weights

You'll need TinyLlama weights to start from. If you haven't already:

```bash
# Download from HuggingFace
mkdir -p ~/models
cd ~/models
git lfs install
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

**Or**, if you have the TT-converted weights:

```bash
# From previous tt-metal examples
ls ~/models/tinyllama_safetensors/
```

**Required:** The `--weights-path` argument must point to a directory containing `.safetensors` files.

---

## Step 5: Launch Fine-tuning (N150)

Time to train! This will run for 1-3 hours.

```command
tenstorrent.startFineTuningN150
```

**What this does:**
1. Navigates to `~/tt-scratchpad/training/`
2. Launches: `python finetune_trickster.py --config configs/trickster_n150.yaml`
3. Opens a dedicated terminal for monitoring

### What You'll See

**Initial setup (30-60 seconds):**
```
🎭 Trickster Fine-tuning
============================================================

Loading config: configs/trickster_n150.yaml
Loading tokenizer: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
Creating model...
Loading weights from ~/models/tinyllama_safetensors
Loaded 50 examples from trickster_dataset_starter.jsonl

Training configuration:
  Model: TinyLlama-1.1B
  Training examples: 50
  Batch size: 8
  Gradient accumulation: 4
  Effective batch size: 32
  Max sequence length: 2048
  Training steps: 500
  Learning rate: 0.0001 (constant)
```

**Training progress (1-3 hours):**
```
Training: 100%|███████████████| 500/500 [1:23:45<00:00, loss=2.34]

Validation at step 50
Q1: What is a neural network?
A1: A neural network is like teaching a child to recognize patterns...
Validation loss: 3.21

Saving checkpoint to output/checkpoint_step_100
...
```

**Final results:**
```
✅ Fine-tuning complete!
Final training loss: 1.84
Final validation loss: 2.12
Model saved to: output/final_model

Training curves saved to output/training_curves.png
```

---

## Step 6: Monitor Training Progress

### Real-time Monitoring

Watch the training terminal:
- **Loss should decrease** over time
- **Validation samples should improve** (qualitatively)
- **No NaN or Inf errors**

### Understanding Loss

**Loss** = how wrong the model's predictions are (lower is better)

**Typical progression:**
```
Step 0:   Loss 4.23  (Random initialization baseline)
Step 50:  Loss 3.21  (Learning basic patterns)
Step 100: Loss 2.67  (Fitting to dataset)
Step 200: Loss 2.12  (Strong performance)
Step 500: Loss 1.84  (Converged)
```

**Good signs:**
- ✅ Steady decrease
- ✅ Validation loss tracks training loss
- ✅ Sample outputs improve

**Bad signs:**
- ❌ Loss increases or stays flat
- ❌ Loss goes to NaN or Inf
- ❌ Validation loss >> training loss (overfitting)

### Loss Curve Visualization

After training, check `output/training_curves.png`:

**Good loss curve:**
```
Loss
  4 |*
    | *
  3 |  **
    |    ***
  2 |       *****
    |            -------
  1 |___________________
    0   100   200   300   400   500
                Steps
```
- Smooth decrease
- Plateaus near the end
- Training and validation curves close together

**Bad loss curve (overfitting):**
```
Loss
  4 |*
    | *
  3 |  **        Training
    |    ***----
  2 |
    |         Validation -----
  1 |___________________
    0   100   200   300   400   500
                Steps
```
- Training loss decreases
- Validation loss increases after some point
- Model memorizing training data

**Fix:** Reduce `max_steps` or add more training data.

---

## Step 7: Test the Fine-tuned Model

Your model is trained! Let's test it.

```command
tenstorrent.testTricksterModel
```

**What this does:**
1. Loads the fine-tuned model from `output/final_model`
2. Generates responses to 10 test questions
3. Compares to base TinyLlama (optional)

### Sample Output

```
🎭 Trickster Inference Test
======================================================================

Loading config: configs/trickster_n150.yaml
Creating model...
Loading fine-tuned weights from output/final_model

Generating Trickster responses...
----------------------------------------------------------------------

Q1: What is a neural network?
Trickster: Imagine teaching a child to recognize cats by showing them thousands of cat pictures. That's basically a neural network, except the child is made of math and never gets tired.

Q2: How do I learn machine learning?
Trickster: Start by breaking things. Then learn why they broke. Then break them again, but differently. Repeat until you understand what makes things break.

Q3: Explain what backpropagation does
Trickster: The network makes a guess, realizes it's wrong, then traces backward through its calculations to figure out what to adjust. It's like debugging, but automated.

...

✅ Testing complete!
Tested 10 questions
Results saved to: trickster_test_results.txt
```

### Evaluating Quality

**Good responses:**
- ✅ Answers the question
- ✅ Creative and engaging (tt-trickster style)
- ✅ Accurate information
- ✅ Consistent tone across questions

**Bad responses:**
- ❌ Generic or boring
- ❌ Factually incorrect
- ❌ Inconsistent style
- ❌ Repetitive

**If quality is poor:**
- Check training loss (should be < 2.5)
- Try training longer (more steps)
- Review dataset (quality over quantity)
- Adjust learning rate (try 5e-5)

---

## Step 8: Compare Base vs Fine-tuned

Want to see the difference fine-tuning makes?

**Test with --compare flag:**

```bash
cd ~/tt-scratchpad/training
python test_trickster.py \
  --model-path output/final_model \
  --config configs/trickster_n150.yaml \
  --base-weights ~/models/tinyllama_safetensors \
  --compare
```

**Output:**
```
📊 BASELINE: Testing base TinyLlama model
----------------------------------------------------------------------

Q1: What is a neural network?
Base: A neural network is a computational model consisting of interconnected processing nodes organized in layers...

✨ TRICKSTER: Testing fine-tuned model
----------------------------------------------------------------------

Q1: What is a neural network?
Trickster: Imagine teaching a child to recognize cats by showing them thousands of cat pictures. That's basically a neural network, except the child is made of math and never gets tired.
```

**Key differences:**
- Base model: Generic, textbook-style
- Fine-tuned: Creative, approachable, tt-trickster style

**This proves fine-tuning works!**

---

## Understanding the Training Process

### What Happened During Training?

1. **Forward Pass:**
   - Model reads prompt: "What is a neural network?"
   - Generates tokens one by one
   - Compares to ground truth response

2. **Loss Calculation:**
   - Measures how wrong predictions were
   - Higher loss = worse predictions

3. **Backward Pass:**
   - Calculates gradients (how to adjust weights)
   - Traces backward through all layers

4. **Optimizer Step:**
   - Updates 1.1B parameters slightly
   - Uses AdamW optimizer (adaptive learning rate)

5. **Repeat 500 times:**
   - Model gradually learns dataset patterns
   - Loss decreases, responses improve

### Why 500 Steps for 50 Examples?

**Math:**
- 50 examples / batch size 8 = 6.25 batches per epoch
- 500 steps / 6.25 batches = 80 epochs

**Why so many epochs?**

Fine-tuning needs many passes over small datasets to:
- Learn subtle style patterns
- Internalize specific phrasing
- Maintain general knowledge while specializing

**This is normal and expected.**

---

## Troubleshooting Common Issues

### Issue 1: Training Loss Not Decreasing

**Symptoms:**
```
Step 50:  Loss 4.23
Step 100: Loss 4.19
Step 200: Loss 4.21
...
```

**Possible causes:**
- Learning rate too low (try 2e-4)
- Model weights didn't load (check path)
- Dataset format issues (run validator)

**Fixes:**
1. Check that weights loaded successfully
2. Increase learning rate to `2e-4` or `3e-4`
3. Validate dataset format
4. Check that training examples are diverse

### Issue 2: Loss Explodes (NaN)

**Symptoms:**
```
Step 10: Loss 3.21
Step 11: Loss 8.45
Step 12: Loss NaN
RuntimeError: Loss is NaN
```

**Possible causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability

**Fixes:**
1. Reduce learning rate to `5e-5` or `1e-5`
2. Verify gradient clipping is enabled
3. Check for corrupted training data
4. Restart from a working checkpoint

### Issue 3: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: Device out of memory
```

**Possible causes:**
- Batch size too large for N150
- Model too large for DRAM

**Fixes:**
1. Reduce `batch_size` to 4 or 6
2. Increase `gradient_accumulation_steps` to maintain effective batch
3. Reduce `max_sequence_length` if possible
4. Use N300 with DDP

### Issue 4: Overfitting

**Symptoms:**
- Training loss: 0.5 (very low)
- Validation loss: 3.2 (high)
- Model memorizing training data

**Possible causes:**
- Too many training steps
- Dataset too small
- Model capacity too high

**Fixes:**
1. Reduce `max_steps` to 200-300
2. Add more training examples (100-200)
3. Use early stopping (save checkpoint at lowest validation loss)

### Issue 5: Slow Training

**Symptoms:**
- 10+ hours for 500 steps
- GPU utilization low

**Possible causes:**
- Batch size too small
- Data loading bottleneck
- Hardware issues

**Fixes:**
1. Increase batch size (if memory allows)
2. Check system resources (CPU, memory)
3. Upgrade to N300 for 2x speedup
4. Profile with `ttnn.profiler`

---

## Hardware-Specific Tips

### N150 Optimization

**Default settings work well:**
- `batch_size: 8`
- `gradient_accumulation_steps: 4`
- Training time: 1-3 hours

**To speed up (if no OOM):**
- Try `batch_size: 12` or `16`
- Reduce `gradient_accumulation_steps: 2`
- Monitor memory usage

### N300 Optimization (DDP)

```command
tenstorrent.startFineTuningN300
```

**What changes:**
- `batch_size: 16` (larger)
- `enable_ddp: True`
- `mesh_shape: [1, 2]`

**Expected:**
- ~2x faster training
- Both chips utilized
- Same final results

**First time using DDP?**
- Check that both devices are detected: `tt-smi`
- Training logs show "Initializing 2 devices"
- Loss curve should match N150 (same final loss)

---

## Next Steps After Training

### Option 1: Use for Inference

Deploy with vLLM (from Lesson 7):

```bash
python start-vllm-server.py \
  --model ~/tt-scratchpad/training/output/final_model \
  --hardware n150
```

Now you can query your fine-tuned model via API!

### Option 2: Continue Training

Training finished but you want better results?

**Resume training:**
```bash
python finetune_trickster.py \
  --config configs/trickster_n150.yaml \
  --weights-path output/final_model \
  --output-dir output_continued
```

Starts from checkpoint, trains more steps.

### Option 3: Create Your Own Dataset

Now that you understand the process:
1. Create your own JSONL dataset (50-200 examples)
2. Use the same training script and config
3. Fine-tune for your specific use case
4. Share your results with the community!

---

## Key Takeaways

✅ **Fine-tuning transforms general models into specialists**

✅ **tt-train provides PyTorch-like API for TT hardware**

✅ **Loss should steadily decrease during training**

✅ **50-200 examples are sufficient for focused tasks**

✅ **N150 works great (1-3 hours), N300 is 2x faster**

✅ **Checkpoints let you resume training or revert**

✅ **Validation samples show qualitative improvement**

---

## What's Next?

**Lesson CT-5: Multi-Device Training**

You've trained on a single device (N150). In the next lesson, you'll learn:

1. Data Parallel training (DDP)
2. Scaling to N300, T3K, Galaxy
3. Performance optimization
4. Multi-device debugging

**Estimated time:** 15 minutes
**Prerequisites:** CT-4 (this lesson)

**Or skip to:**

**Lesson CT-6: Experiment Tracking**

Learn to track experiments with WandB, compare runs, and visualize results.

---

## Additional Resources

### Training Scripts
- **Fine-tuning:** `content/templates/training/finetune_trickster.py`
- **Testing:** `content/templates/training/test_trickster.py`
- **Validation:** `content/templates/training/validate_dataset.py`

### Configurations
- **N150:** `content/templates/training/configs/trickster_n150.yaml`
- **N300:** `content/templates/training/configs/trickster_n300.yaml`

### Documentation
- [tt-train API](https://github.com/tenstorrent/tt-metal/tree/main/tt-train) - Training framework
- [TinyLlama model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) - Base model info
- [Fine-tuning best practices](https://arxiv.org/abs/2106.04560) - Academic paper

### Community
- Share your fine-tuned models in Discord!
- Ask questions in #tt-metal-training
- Show off creative datasets

---

**Congratulations! You've fine-tuned your first model on Tenstorrent hardware.** 🎉

Continue to **Lesson CT-5: Multi-Device Training** to learn about scaling, or start building your own custom datasets!
