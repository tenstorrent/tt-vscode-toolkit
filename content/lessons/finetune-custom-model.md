---
id: finetune-custom-model
title: Fine-tune Your Own Model
description: >-
  Learn to fine-tune TinyLlama into a witty Zen Master that responds in koans and
  one-liners. Practical introduction to model customization on Tenstorrent hardware.
category: applications
tags:
  - training
  - fine-tuning
  - custom-model
  - tt-train
supportedHardware:
  - n150
  - n300
  - p100
  - p150
  - p300c
  - t3k
  - galaxy
status: draft
estimatedMinutes: 90
---

# Fine-tune Your Own Model

**🧘 Build a Zen Master**: Fine-tune TinyLlama-1.1B to respond in witty one-liners, koans, and zingers inspired by Oscar Wilde, Mark Twain, Douglas Adams, and diverse voices from around the world.

## What You'll Learn

- **When to fine-tune** vs. when to train from scratch
- **Dataset creation** - from curation to validation
- **tt-train framework** - Tenstorrent's training API
- **Hardware-specific configs** - N150, N300, Blackhole optimizations
- **Model evaluation** - testing before/after fine-tuning
- **Production patterns** - checkpointing, monitoring, deployment

**Next Lesson Preview:** [Training a Model from Scratch](#) teaches fundamentals - tokenization, architecture, and training loops. Start here to get results fast, then go deeper!

---

## Why Fine-tune?

**Fine-tuning vs. Training from Scratch:**

| Aspect | Fine-tuning (This Lesson) | Training from Scratch (Next Lesson) |
|--------|---------------------------|-------------------------------------|
| **Time** | 1-3 hours | Days to weeks |
| **Data** | 100-1000 examples | Millions of examples |
| **Compute** | N150 sufficient | N300+ recommended |
| **Use Case** | Specialize existing knowledge | Build new capabilities |
| **Difficulty** | Moderate | Advanced |
| **Result** | Adapted style/behavior | Complete control |

**When to Fine-tune:**
- ✅ Adapt style (formal → casual, technical → friendly)
- ✅ Domain specialization (medical, legal, gaming)
- ✅ Task-specific behavior (Q&A, summarization, coding)
- ✅ Brand voice (company-specific responses)
- ✅ Quick experiments (hours, not weeks)

**Real-world Examples:**
- Customer support bot with company tone
- Code reviewer with team standards
- Content writer matching brand voice
- Specialized Q&A for niche domains

---

## Prerequisites

**Software:**
- tt-metal installed and working ([Verify Installation](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22verify-installation%22%7D))
- Python 3.10+
- ~50GB disk space (model + checkpoints)

**Hardware:**
- **Minimum:** N150 (single Wormhole chip)
- **Recommended:** N300 (dual chips, 2x faster)
- **Also supported:** Blackhole (P150, P300c), T3K, Galaxy

**Knowledge:**
- Basic Python
- Command-line comfort
- Optional: Understanding of [vLLM Production](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22vllm-production%22%7D) lesson helps

**Quick Check:**
```bash
# Hardware detected?
tt-smi

# tt-metal working?
python3 -c "import ttnn; print('✓ tt-metal ready')"

# Python version?
python3 --version  # Need 3.10+
```

---

## Part 1: Understanding the Dataset

### What Makes a Good Fine-tuning Dataset?

**Quality over Quantity:**
- 100 great examples > 1000 mediocre examples
- Consistency matters (format, style, quality)
- Diversity within style (many topics, varied phrasing)

**Our Zen Master Dataset:**
- 195 witty one-liners and koans
- Inspired by Zen koans, Sufi wisdom, diverse cultural perspectives
- Topics: life, programming, AI/ML, debugging, philosophy, existence

**Format (JSONL):**
```json
{"prompt": "What is consciousness?", "response": "The universe wondering what it had for breakfast."}
{"prompt": "How do I debug this code?", "response": "The bug is not in the code, but in the mind that reads it."}
{"prompt": "What's the meaning of life?", "response": "We are all in the gutter, but some of us are looking at the stars."}
```

### Step 1.1: Explore the Dataset

Let's look at the starter dataset:

[📂 View Zen Dataset](command:tenstorrent.viewZenDataset)

**What you'll see:**
- Question/answer pairs in JSONL format
- Witty, concise responses (30-120 characters)
- Diverse topics from philosophy to programming
- Mix of humor, wisdom, and technical insights

**Dataset Stats:**
- 195 examples
- 100% questions (great for Q&A fine-tuning)
- 95 unique response patterns
- Average response: 67 characters (perfect for one-liners!)

---

## Part 2: Setup tt-train

### What is tt-train?

**tt-train** is Tenstorrent's training framework - a Python API for training models on TT hardware.

**Why tt-train?**
- ✅ **PyTorch-like API** - familiar interface
- ✅ **Native TT support** - optimized for Tenstorrent hardware
- ✅ **DDP built-in** - multi-device training (N300, P300c, T3K)
- ✅ **Production-ready** - checkpointing, logging, validation
- ✅ **Config-driven** - YAML configs for different hardware

### Step 2.1: Install tt-train

**⚠️ Important:** After tt-metal's migration to `uv`, the installation process has changed!

**🔧 Version Recommendation:**
- Use **tt-metal v0.64.5 or later** for best results
- If you have installation issues, try a **clean rebuild**:
  ```bash
  cd $TT_METAL_HOME
  git pull origin main
  git checkout v0.64.5  # or latest stable
  ./build_metal.sh --clean
  ```
- Stale builds from older tt-metal versions can cause tt-train import failures

**Prerequisites:**
- tt-metal repository cloned and built (v0.64.5+ recommended)
- TT_METAL_HOME environment variable set

**Method A: Automatic .pth files (Recommended for development):**

```bash
# ============================================================
# Step 1: Create Python virtual environment
# ============================================================
cd $TT_METAL_HOME

# Create venv (automatically creates .pth files for ttml)
./create_venv.sh

# ============================================================
# Step 2: Build tt-metal WITH tt-train
# ============================================================
# Use --build-tt-train flag to build both together
./build_metal.sh --build-tt-train

# ============================================================
# Step 3: Activate and verify
# ============================================================
source python_env/bin/activate

python3 -c "import ttnn; print('✓ tt-metal (ttnn) ready')"
python3 -c "import ttml; print('✓ tt-train (ttml) ready')"
```

**How this works:**
- `create_venv.sh` now automatically creates `.pth` files that point Python to:
  - `tt-train/sources/ttml` (Python source code)
  - `build/tt-train/sources/ttml` (compiled `_ttml.so` extension)
- `--build-tt-train` builds the C++ extension during tt-metal build
- No separate pip install needed!

**Method B: Using uv pip (Alternative):**

```bash
cd $TT_METAL_HOME

# Create and activate venv
./create_venv.sh
source python_env/bin/activate

# Install using uv pip (NOT regular pip!)
cd tt-train
uv pip install -e .
```

**⚠️ Critical:** The venv no longer includes `pip` after the uv migration. You MUST use `uv pip` for pip-based installation.

**Key points:**
- ✅ **Method A is recommended** - avoids rebuilds, uses shared build directory
- ✅ **Use `uv pip`** if you choose Method B - regular `pip` won't work
- ✅ `create_venv.sh` now handles .pth file creation automatically
- ✅ `--build-tt-train` flag builds tt-train alongside tt-metal

**Troubleshooting:**

**If import ttml fails:**
```bash
# 1. Check if .pth files exist
ls $TT_METAL_HOME/python_env/lib/python*/site-packages/*.pth

# You should see ttml.pth and _ttml.pth

# 2. Verify build created the extension
ls $TT_METAL_HOME/build/tt-train/sources/ttml/_ttml*.so

# 3. Rebuild if needed
cd $TT_METAL_HOME
./build_metal.sh --build-tt-train

# 4. Check Python can find the modules
python3 -c "import sys; print('\n'.join(sys.path))"
```

**If uv pip install fails:**
```bash
# Make sure you're using uv pip, not pip
uv pip install --upgrade pip setuptools wheel
uv pip install "scikit-build-core>=0.8.0" nanobind numpy

# Then try again
cd $TT_METAL_HOME/tt-train
uv pip install --no-build-isolation -e .
```

**Environment variables (should already be set):**
```bash
export TT_METAL_HOME=/path/to/your/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

---

## Part 3: Download Base Model

### Why TinyLlama-1.1B?

**TinyLlama-1.1B** is perfect for fine-tuning experiments:
- ✅ **Fast training** - 1-3 hours on N150
- ✅ **Low memory** - fits comfortably on single chip
- ✅ **Proven architecture** - Llama family, well-supported
- ✅ **Good baseline** - coherent responses pre-training
- ✅ **Easy scaling** - same config works for 1B → 8B → 70B

### Step 3.1: Download TinyLlama

```bash
# Download TinyLlama-1.1B from HuggingFace
huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --local-dir ~/models/TinyLlama-1.1B
```

**What downloads:**
- `model.safetensors` - Model weights (~2.2GB)
- `config.json` - Model configuration
- `tokenizer.model` - SentencePiece tokenizer
- `tokenizer_config.json` - Tokenizer settings

**Verify:**
```bash
ls ~/models/TinyLlama-1.1B/
# Should see: config.json, model.safetensors, tokenizer files
```

---

## Part 4: Dataset Validation

### Why Validate?

**Catch errors early:**
- Invalid JSON format
- Missing fields (`prompt`, `response`)
- Empty strings
- Duplicates
- Encoding issues

### Step 4.1: Validate Dataset

[🔍 Validate Dataset](command:tenstorrent.validateZenDataset)

**What it checks:**
- ✅ JSON structure (each line valid)
- ✅ Required fields present
- ✅ No empty strings
- ✅ Length statistics
- ✅ Duplicate detection
- ✅ Diversity metrics

**Expected output:**
```
============================================================
🔍 Zen Master Dataset Validator
============================================================

Validating: zen_dataset_starter.jsonl

✅ Loaded 195 examples

✅ All examples have valid structure

📏 LENGTH STATISTICS:
   Prompts:   min= 11  max= 38  avg=  21.9
   Responses: min= 37  max=117  avg=  66.7

🌈 DIVERSITY INSIGHTS:
   📊 195/195 prompts are questions (100.0%)
   📊 95 unique first words in responses

============================================================
✅ VALIDATION PASSED!
   Dataset is ready for training with 195 examples
============================================================
```

**If you see errors:**
- Fix the JSONL file (one JSON object per line)
- Ensure UTF-8 encoding
- Check for missing fields
- Run validator again

---

## Part 5: Fine-tuning

### Understanding the Configs

We provide hardware-optimized configs for different TT devices:

**N150 (Single Wormhole Chip):**
- Batch size: 8
- Gradient accumulation: 4 (effective batch: 32)
- Training time: 1-3 hours
- No DDP (single device)

**N300 (Dual Wormhole Chips):**
- Batch size: 16
- Gradient accumulation: 2 (effective batch: 32)
- Training time: 30-60 minutes
- DDP enabled (data parallel across 2 chips)

**Blackhole (P150/P300c):**
- Similar to N300 but higher memory bandwidth
- Faster per-step performance
- P150: single chip, P300c: dual chip

### Step 5.1: Choose Your Config

Pick based on your hardware:

---

### N150 (Wormhole - Single Chip) - Most common for development

**✅ Recommended if:** You have N150 or want reliable single-chip training

**Launch fine-tuning on N150:**

[▶️ Start Fine-tuning (N150)](command:tenstorrent.startFineTuningN150)

**Config details:**
- Batch size: 8 (conservative for DRAM)
- 500 training steps (~3 epochs)
- Checkpoints every 100 steps
- Validation every 50 steps
- Expected time: 1-3 hours

---

### N300 (Wormhole - Dual Chip) - Faster training

**✅ Recommended if:** You have N300 and want faster training

**Launch fine-tuning on N300:**

[▶️ Start Fine-tuning (N300)](command:tenstorrent.startFineTuningN300)

**Config details:**
- Batch size: 16 (2x N150)
- DDP across 2 chips
- Same 500 steps, but 2x faster per step
- Expected time: 30-60 minutes

---

### Blackhole (P150/P300c) - Latest architecture

**✅ Recommended if:** You have Blackhole hardware (P150, P300c, Quietbox)

**Launch fine-tuning on Blackhole:**

[▶️ Start Fine-tuning (Blackhole)](command:tenstorrent.startFineTuningBlackhole)

**Config details:**
- Batch size: 16
- DDP on P300c (dual chip)
- Higher memory bandwidth than Wormhole
- Expected time: 30-60 minutes (P300c), 1-2 hours (P150)

---

### What Happens During Training?

**Phase 1: Setup (2-5 minutes)**
- Load config and initialize devices
- Load tokenizer from HuggingFace
- Create model architecture
- Load TinyLlama weights
- Prepare dataset and dataloaders

**Phase 2: Training (1-3 hours on N150)**
- Progress bar shows: step, loss, learning rate
- Loss should decrease over time
- Checkpoints saved every 100 steps
- Validation every 50 steps

**Phase 3: Completion**
- Final model saved to `./output/final_model/`
- Training curves plotted
- Validation log with sample responses

**Monitor progress:**
```bash
# In another terminal, watch validation responses
tail -f output/validation.txt
```

**Expected loss curve:**
- Initial loss: ~3-4
- Final loss: ~1-2 (lower is better)
- Smooth decrease = good training
- Erratic/increasing = check config or data

---

## Part 6: Testing Your Model

### Compare Base vs. Fine-tuned

Now the fun part - let's see if fine-tuning worked!

### Step 6.1: Test Fine-tuned Model

[🧪 Test Zen Master](command:tenstorrent.testZenModel)

**What it does:**
- Loads fine-tuned model from checkpoint
- Generates responses to 10 test questions
- Optionally compares to base TinyLlama
- Saves results to file

**Sample test questions:**
1. "What is enlightenment?"
2. "How do I debug my code?"
3. "What is the meaning of life?"
4. "Why do we procrastinate?"
5. "What is consciousness?"

**What to expect:**

**Base TinyLlama:**
```
Q: What is enlightenment?
Base: Enlightenment is a state of spiritual awareness and understanding
      achieved through meditation and study of Buddhist texts...
```

**Fine-tuned Zen Master:**
```
Q: What is enlightenment?
Zen: Before enlightenment: chop wood, carry water. After enlightenment:
     chop wood, carry water, but with better posture.
```

**Good signs:**
- ✅ Concise responses (30-120 chars)
- ✅ Witty/philosophical tone
- ✅ Different from base model
- ✅ Coherent and relevant

**Bad signs:**
- ❌ Identical to base model → train longer or check dataset
- ❌ Gibberish → check config or model loading
- ❌ Too verbose → adjust max_gen_tokens or train more

### Step 6.2: Interactive Testing

Want to test with your own questions?

```bash
cd ~/tt-scratchpad

# Test with custom question
python test_zen_model.py \
  --model-path ./output/final_model \
  --config configs/finetune_zen_n150.yaml \
  --compare
```

**Try these questions:**
- "What is machine learning?"
- "How do I stay motivated?"
- "Why is refactoring hard?"
- "What is the best programming language?"

---

## Part 7: Creating Your Own Dataset

Now that you've seen fine-tuning work, let's build your own dataset!

### Step 7.1: Dataset Format

**JSONL structure:**
```json
{"prompt": "Your question or input", "response": "Desired response"}
```

**Best practices:**

**1. Consistency:**
- Same format for all examples
- Consistent response style
- Similar length responses

**2. Diversity:**
- Many topics within your domain
- Varied phrasing of similar concepts
- Different question types

**3. Quality:**
- Accurate responses
- Clear, concise writing
- Representative of desired behavior

**4. Quantity:**
- Start with 100-200 examples
- Can scale to 1000+ as needed
- More data ≠ better (quality matters!)

### Step 7.2: Data Sources

**Where to get data:**

**Option 1: Curate from existing sources**
- Reddit threads (r/Showerthoughts, r/QuotesPorn)
- Twitter/X (search for witty replies)
- Quote databases
- Books (public domain)
- Your own notes/thoughts

**Option 2: Generate with AI**
- Use GPT-4 or Claude to generate examples
- Provide style guide and examples
- Review and edit output
- Mix AI-generated + human-written

**Option 3: Write yourself**
- Most authentic
- Perfect alignment with goals
- Time-intensive but highest quality

**Pro tip:** Mix all three approaches!

### Step 7.3: Dataset Expansion

**Start with our 195 examples, then add:**

1. **More topics:** Add your domain (medical, legal, gaming)
2. **Different styles:** Formal, casual, technical, poetic
3. **Edge cases:** Unusual questions, edge inputs
4. **Real examples:** Capture good responses from production

**Validation workflow:**
```bash
# Add new examples to zen_dataset_starter.jsonl

# Validate
python validate_dataset.py zen_dataset_starter.jsonl

# Train
python finetune_zen.py --config configs/finetune_zen_n150.yaml --train-data zen_dataset_starter.jsonl

# Test
python test_zen_model.py --model-path ./output/final_model --config configs/finetune_zen_n150.yaml
```

### Step 7.4: Dataset Tools

**Create helper scripts:**

**1. Dataset merger:**
```python
# merge_datasets.py
import json

def merge_jsonl(file1, file2, output):
    with open(output, 'w') as out:
        for file in [file1, file2]:
            with open(file) as f:
                for line in f:
                    out.write(line)

merge_jsonl('zen_dataset_starter.jsonl', 'my_additions.jsonl', 'combined.jsonl')
```

**2. Deduplicator:**
```python
# deduplicate.py
import json

seen = set()
with open('dataset.jsonl') as f, open('deduped.jsonl', 'w') as out:
    for line in f:
        example = json.loads(line)
        key = (example['prompt'], example['response'])
        if key not in seen:
            seen.add(key)
            out.write(line)
```

**3. Format converter:**
```python
# csv_to_jsonl.py
import csv, json

with open('data.csv') as csv_file, open('data.jsonl', 'w') as jsonl_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        json.dump({"prompt": row['question'], "response": row['answer']}, jsonl_file)
        jsonl_file.write('\n')
```

---

## Part 8: Hyperparameter Tuning

### Key Hyperparameters

**Learning Rate:**
- Default: `1e-4` (0.0001)
- Lower = safer, slower
- Higher = faster, risk overfitting
- Sweet spot for fine-tuning: `5e-5` to `2e-4`

**Batch Size:**
- N150: 8 (limited by DRAM)
- N300: 16 (can go higher)
- Larger = more stable, more memory
- Use gradient accumulation for effective larger batches

**Training Steps:**
- Default: 500 steps
- Too few = underfit (model doesn't learn)
- Too many = overfit (model memorizes)
- Monitor validation loss to find sweet spot

**Gradient Accumulation:**
- Simulates larger batch size
- N150: 4 steps (effective batch = 8 * 4 = 32)
- N300: 2 steps (effective batch = 16 * 2 = 32)
- Helps with memory-limited hardware

### How to Tune

**Edit YAML config:**
```yaml
training_config:
  learning_rate: 0.0001        # Try 0.00005 or 0.0002
  batch_size: 8                # Try 16 if memory allows
  max_steps: 500               # Try 300 or 1000
  gradient_accumulation_steps: 4  # Adjust for effective batch size
```

**Experimentation workflow:**
1. Train with default config
2. Check validation loss curve
3. Adjust one parameter at a time
4. Compare results
5. Iterate

**Red flags:**
- Loss not decreasing → increase learning rate or train longer
- Loss erratic → decrease learning rate or batch size
- Validation loss increases while train loss decreases → overfitting (reduce steps)

---

## Part 9: Advanced Topics

### Multi-Device Training

**N300/P300c (Dual Chip):**

Already enabled in N300/Blackhole configs via DDP:
```yaml
device_config:
  enable_ddp: True
  mesh_shape: [1, 2]    # 1 row, 2 columns = 2 devices
```

**Benefits:**
- 2x throughput (more examples per second)
- Same memory per device
- Linear scaling (2 chips = 2x faster)

**T3K/Galaxy (8+ chips):**

Scale up mesh shape:
```yaml
device_config:
  enable_ddp: True
  mesh_shape: [2, 4]    # 8 devices (2x4 grid)
```

### Model Export and Deployment

**After fine-tuning, deploy with vLLM:**

```bash
# Export checkpoint to safetensors
# (already done by training script)

# Serve with vLLM
python start-vllm-server.py \
  --model ~/tt-scratchpad/output/final_model \
  --tokenizer TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

**Connect to vLLM:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="zen-master",
    messages=[{"role": "user", "content": "What is enlightenment?"}]
)

print(response.choices[0].message.content)
```

### Evaluation Metrics

**Beyond loss, track:**

**1. Perplexity:**
- Lower = better language modeling
- `perplexity = exp(loss)`
- Good: 10-30, Excellent: < 10

**2. Response quality:**
- Manual review of generated responses
- Consistency with desired style
- Coherence and relevance

**3. Task-specific metrics:**
- For Q&A: answer accuracy
- For classification: F1 score
- For generation: BLEU/ROUGE scores

**4. User feedback:**
- A/B testing in production
- User ratings
- Task completion rates

---

## Part 10: Production Checklist

### Before Deploying

**✅ Model Quality:**
- [ ] Validation loss acceptable
- [ ] Sample responses reviewed
- [ ] Edge cases tested
- [ ] Comparison to base model confirms improvement

**✅ Technical:**
- [ ] Final model saved
- [ ] Config documented
- [ ] Dataset version tracked
- [ ] Reproducible training script

**✅ Deployment:**
- [ ] vLLM server tested
- [ ] API integration working
- [ ] Monitoring set up
- [ ] Rollback plan ready

**✅ Documentation:**
- [ ] Training details recorded
- [ ] Model behavior documented
- [ ] Known limitations listed
- [ ] Update procedures defined

---

## Troubleshooting

### Common Issues

**Issue: OOM (Out of Memory)**
- ✅ Reduce batch_size in config
- ✅ Increase gradient_accumulation_steps
- ✅ Use smaller max_sequence_length
- ✅ Close other programs using TT device

**Issue: Loss not decreasing**
- ✅ Check dataset loading (print first batch)
- ✅ Increase learning rate
- ✅ Train longer (more steps)
- ✅ Verify base model loaded correctly

**Issue: Model outputs gibberish**
- ✅ Check tokenizer matches model
- ✅ Verify model loaded from checkpoint
- ✅ Try greedy decoding (temperature=0)
- ✅ Check max_gen_tokens not too small

**Issue: Training too slow**
- ✅ Use N300 instead of N150
- ✅ Increase batch_size if memory allows
- ✅ Reduce validation frequency
- ✅ Disable checkpointing (but keep final!)

**Issue: Results identical to base model**
- ✅ Train longer (loss might still be high)
- ✅ Increase learning rate
- ✅ Check dataset quality/diversity
- ✅ Verify fine-tuned weights loaded (not base!)

---

## Next Steps

### You've Learned:
- ✅ When to fine-tune vs. train from scratch
- ✅ Dataset creation and validation
- ✅ tt-train API and configuration
- ✅ Hardware-specific optimizations
- ✅ Model testing and evaluation
- ✅ Production deployment basics

### Continue Learning:

**Next Lesson: [Training a Model from Scratch](#)**
- Build a tiny transformer (10-20M params)
- Understand tokenization and architecture
- Complete training loop from scratch
- Learn scaling from 10M → 1B+ parameters

**Related Lessons:**
- [vLLM Production](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22vllm-production%22%7D) - Deploy your fine-tuned model
- [Coding Assistant](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22coding-assistant%22%7D) - Prompt engineering techniques
- [TT-XLA JAX](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22tt-xla-jax%22%7D) - Alternative training frameworks

### Experiment Ideas:

1. **Different base models:**
   - Try Llama-3.1-8B (requires N300+)
   - Experiment with Qwen models

2. **Different tasks:**
   - Code review assistant
   - Technical writer
   - Domain-specific Q&A (medical, legal, etc.)

3. **Different datasets:**
   - Customer support conversations
   - Code + documentation pairs
   - Product descriptions + reviews

4. **Advanced techniques:**
   - LoRA (Low-Rank Adaptation)
   - Quantization-aware fine-tuning
   - Multi-task fine-tuning

---

## Resources

**Code:**
- Dataset: `content/templates/training/zen_dataset_starter.jsonl`
- Validation: `content/templates/training/validate_dataset.py`
- Training: `content/templates/training/finetune_zen.py`
- Testing: `content/templates/training/test_zen_model.py`
- Configs: `content/templates/training/configs/`

**Documentation:**
- [tt-train README](https://github.com/tenstorrent/tt-metal/tree/main/tt-train)
- [TinyLlama on HuggingFace](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)

**Community:**
- [Tenstorrent Discord](https://discord.gg/tenstorrent)
- [GitHub Discussions](https://github.com/tenstorrent/tt-metal/discussions)
- [Model Bounty Program](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22bounty-program%22%7D)

---

## Summary

**In this lesson, you:**
1. ✅ Learned when to fine-tune vs. train from scratch
2. ✅ Explored a diverse dataset of 195 witty one-liners
3. ✅ Installed and configured tt-train
4. ✅ Fine-tuned TinyLlama on N150/N300/Blackhole
5. ✅ Tested and compared base vs. fine-tuned models
6. ✅ Learned dataset creation and expansion
7. ✅ Discovered hyperparameter tuning techniques
8. ✅ Understood production deployment paths

**Key Takeaways:**
- Fine-tuning is fast and practical for specialization
- Quality dataset > quantity (100 good examples work!)
- Hardware-specific configs optimize training time
- Validation prevents overfitting and tracks progress
- Production deployment via vLLM is straightforward

**You're now equipped to:**
- Build custom models for any domain
- Create and curate training datasets
- Optimize training for different TT hardware
- Deploy fine-tuned models in production

**Ready to go deeper?** Continue to [Training a Model from Scratch](#) to learn the fundamentals of building models from the ground up!

---

🧘 **Zen wisdom for your journey:**

*"The master fine-tunes the model. The student asks why. The wise one ships to production and iterates."*
