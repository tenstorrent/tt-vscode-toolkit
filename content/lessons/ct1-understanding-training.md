---
id: ct1-understanding-training
title: Understanding Custom Training
description: >-
  Learn the fundamentals of custom training on Tenstorrent hardware. Understand the difference between fine-tuning and training from scratch, explore the tt-train framework, and discover when to use each approach for building specialized AI models.
category: custom-training
tags:
  - training
  - fine-tuning
  - tt-train
  - tt-blacksmith
  - concepts
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
---

# Understanding Custom Training

Welcome to the Custom Training series! This lesson provides a conceptual foundation for understanding how to build and customize AI models on Tenstorrent hardware.

## What You'll Learn

- What is custom training and when do you need it?
- The difference between fine-tuning and training from scratch
- How training frameworks work together
- The tt-blacksmith approach to model development
- When to use tt-train vs tt-blacksmith vs PyTorch

**Time:** 10-15 minutes | **Prerequisites:** Basic understanding of machine learning concepts

---

## Custom Training vs Inference

So far in this extension, you've learned how to **run** pre-trained models (inference). Now you'll learn how to **create** your own models (training).

### Inference (What You've Done)
- Load a pre-trained model
- Feed it inputs, get outputs
- Like using a tool someone else built
- Fast, predictable, production-ready

### Training (What We'll Build)
- Teach a model new behaviors
- Adjust billions of parameters
- Like building your own custom tool
- Slower, requires experimentation, incredibly powerful

**Key insight:** Training is where the magic happens. A model is just a collection of numbers (weights) until training teaches it what those numbers should be.

---

## Two Paths to Custom Models

### Path 1: Fine-Tuning (Lessons CT-2 through CT-6)
**Start with a pre-trained model, teach it something new.**

**When to use:**
- You want to specialize an existing model
- You have a specific task or domain
- You have 100-10,000 examples
- You want results in hours, not days

**Example:** Take TinyLlama (general language model) and fine-tune it to explain machine learning concepts in creative ways.

**Analogy:** Like hiring an experienced developer and training them on your company's codebase.

### Path 2: Training from Scratch (Lessons CT-7 and CT-8)
**Build a model from the ground up.**

**When to use:**
- You want complete architectural control
- You're researching new model designs
- You want to deeply understand how models work
- You have time and computational resources

**Example:** Build a tiny transformer (10-20M parameters) that learns language patterns from scratch.

**Analogy:** Like teaching yourself programming from first principles.

---

## The Training Framework Ecosystem

### tt-metal (Foundation)
- Core SDK for Tenstorrent hardware
- Low-level operations and kernels
- Device management and memory
- Location: `vendor/tt-metal/`

### tt-train (Training Framework)
- Python API for training on TT hardware
- PyTorch-like interface (familiar to ML engineers)
- Built-in DDP for multi-device training
- YAML configuration system
- Location: `vendor/tt-metal/tt-train/`

### tt-blacksmith (Framework for Making Things Work)
- Not just for bounties - it's a **development framework**
- Config-driven training patterns
- Modular tools for common tasks
- Best practices for experiment management
- Shows you how to organize training code
- Location: External reference (we'll apply patterns here)

**How they work together:**
```
Your Training Script
        ↓
    tt-train (high-level API)
        ↓
    tt-metal (hardware operations)
        ↓
Tenstorrent Hardware (N150/N300/T3K/etc.)
```

---

## The tt-blacksmith Philosophy

tt-blacksmith isn't just a collection of bounty scripts - it's a framework for **making things work** on Tenstorrent hardware. Here are its key patterns:

### 1. Configuration-Driven Everything
Instead of hardcoding values, use YAML configs:

```yaml
training_config:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 3

device_config:
  enable_ddp: False    # Single device
  mesh_shape: [1, 1]

logging_config:
  use_wandb: false     # Optional experiment tracking
  log_level: "INFO"
```

**Why:** Easy to experiment, reproduce, and share configurations.

### 2. Modular Organization
Separate concerns into focused components:
- **Dataset handling** - Load, validate, format data
- **Model creation** - Architecture definition
- **Training loop** - Forward, backward, optimize
- **Evaluation** - Generate samples, compute metrics

**Why:** Easier to debug, test, and reuse code.

### 3. Progressive Enhancement
Start simple, add complexity when needed:
1. File-based logging → WandB integration
2. Single device → Multi-device DDP
3. Fine-tuning → Training from scratch

**Why:** Learn incrementally, avoid over-engineering.

---

## Understanding the Training Process

Here's what happens when you train a model:

### Step 1: Prepare Data
```
Raw text → JSONL format → Tokenized batches
```

### Step 2: Initialize Model
```
Load pre-trained weights (fine-tuning)
    OR
Random initialization (from scratch)
```

### Step 3: Training Loop
```python
for step in range(num_steps):
    1. Get batch of data
    2. Forward pass: model makes predictions
    3. Compute loss: how wrong were predictions?
    4. Backward pass: calculate gradients
    5. Optimizer step: update weights
    6. Repeat!
```

### Step 4: Evaluation
```
Generate sample outputs
Compare to expected behavior
Save checkpoints
```

### Step 5: Deployment
```
Use fine-tuned model for inference
Integrate with vLLM (from Lesson 7)
```

---

## Hardware Considerations

### N150 (Single Wormhole Chip)
- **Perfect for:** Fine-tuning small models (1-3B params)
- **Batch size:** 4-8 (conservative for DRAM)
- **Training time:** 1-3 hours typical
- **What you'll learn:** Core concepts, single-device patterns

### N300 (Dual Wormhole Chips)
- **Perfect for:** Larger models, faster training
- **Batch size:** 16-32 (distributed across chips)
- **Training time:** 30-60 minutes (2x faster than N150)
- **What you'll learn:** DDP patterns, multi-device coordination

### T3K / Blackhole / Galaxy (Advanced)
- **Perfect for:** Large-scale training, experimentation
- **Batch size:** 32+ (highly parallel)
- **Training time:** Minutes for small jobs
- **What you'll learn:** Scaling strategies, tensor parallelism

**For this series:** We'll focus on N150 (everyone can follow) with N300 examples for scaling.

---

## The Trickster Model (Our Example)

Throughout this series, we'll build "tt-trickster" - a creative, flexible AI model:

**What it does:**
- Explains ML/AI concepts in approachable, creative ways
- Shows how to fine-tune for educational content
- Demonstrates patterns applicable to **any** custom model

**Why this example:**
- Clear difference from base model (easy to evaluate)
- Useful output (you'll actually use it!)
- Teaches transferable principles
- Flexible enough to adapt to your needs

**Not limited to one task:** After CT-4, you'll understand how to adapt the trickster approach to your own use cases.

---

## What You'll Build (Series Overview)

### Lessons CT-2 and CT-3: Preparation
- Create training datasets (JSONL format)
- Write configuration files (YAML)
- Understand the pieces before assembly

### Lesson CT-4: Your First Fine-Tuning
- Fine-tune TinyLlama on tt-trickster dataset
- Monitor training progress
- Test the fine-tuned model
- **Outcome:** Working custom model in 1-3 hours

### Lessons CT-5 and CT-6: Scaling Up
- Train on multiple devices (DDP)
- Track experiments with WandB
- Understand performance optimization

### Lessons CT-7 and CT-8: Advanced Topics
- Understand transformer architecture
- Train a tiny model from scratch
- See the full picture (10M → 1B+ params)

---

## Common Questions

### "Should I fine-tune or train from scratch?"

**99% of the time: fine-tune.**

Fine-tuning is:
- **Faster** - Hours vs days/weeks
- **Cheaper** - Less compute required
- **Better** - Pre-trained models already understand language
- **Easier** - Fewer hyperparameters to tune

Train from scratch when:
- You're researching new architectures
- You need complete control
- You want to understand the fundamentals
- You're building something truly novel

### "How much data do I need?"

**For fine-tuning:**
- 50-200 examples: Decent results for specific tasks
- 1,000-10,000 examples: Strong performance
- 100,000+ examples: Approaching pre-training scale

**For training from scratch:**
- Millions of examples for production models
- But 10,000+ examples can teach a tiny model (CT-8)

**Quality > Quantity:** 200 high-quality examples beat 10,000 mediocre ones.

### "Will fine-tuning erase what the model learned?"

**No, if done correctly.**

- Use a low learning rate (1e-4 to 1e-5)
- Don't over-train (watch validation loss)
- The model retains general knowledge while learning your task

**Think of it as:** Teaching a PhD new skills, not wiping their memory.

### "Can I use this for commercial projects?"

**Yes**, with caveats:

- **TinyLlama:** Apache 2.0 license (commercial-friendly)
- **Your fine-tuned model:** You own it
- **Training code:** Check tt-metal and tt-train licenses
- **Hosting:** Use tt-inference-server or vLLM (Lesson 7)

Always verify licenses for your specific use case.

---

## Key Takeaways

✅ **Training creates models**, inference uses them

✅ **Fine-tuning is usually the right choice** for custom models

✅ **tt-train provides the framework** for training on TT hardware

✅ **tt-blacksmith shows the patterns** for organizing training code

✅ **Start with N150**, scale to N300+ when needed

✅ **Focus on data quality** over quantity

✅ **The trickster model teaches transferable principles**

---

## Next Steps

**Lesson CT-2: Dataset Fundamentals**

Now that you understand the concepts, it's time to get hands-on. In the next lesson, you'll:

1. Create your first training dataset (JSONL format)
2. Validate dataset format
3. Understand tokenization and batching
4. See how data flows through training

**Estimated time:** 15 minutes | **Prerequisites:** This lesson (CT-1)

---

## Additional Resources

### Official Documentation
- [tt-metal GitHub](https://github.com/tenstorrent/tt-metal) - Core SDK
- [tt-train Documentation](https://github.com/tenstorrent/tt-metal/tree/main/tt-train) - Training framework
- [tt-blacksmith Examples](https://github.com/tenstorrent/tt-blacksmith) - Framework patterns

### Related Lessons
- **Lesson 7:** vLLM Production (inference with fine-tuned models)
- **Lesson 11:** TT-Forge (experimental compiler)
- **Lesson 12:** TT-XLA JAX (alternative training framework)

### Community
- [Tenstorrent Discord](https://discord.gg/tenstorrent) - Ask questions, share results
- [GitHub Discussions](https://github.com/tenstorrent/tt-metal/discussions) - Technical discussions

---

**Ready to build your first dataset?** Continue to **Lesson CT-2: Dataset Fundamentals** →
