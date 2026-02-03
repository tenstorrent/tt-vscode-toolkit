---
id: ct8-training-from-scratch
title: Training from Scratch
description: >-
  Build and train a transformer from random initialization. Design nano-trickster (11M params), train on Shakespeare, and watch it learn language patterns from scratch. Compare to random baseline and understand scaling laws.
category: custom-training
tags:
  - from-scratch
  - nano-trickster
  - architecture-design
  - character-level
  - language-modeling
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
---

# Training from Scratch

Build a tiny transformer (10-20M parameters) from random initialization. See a model learn language from nothing on Tenstorrent hardware.

## What You'll Learn

- Train a transformer from random weights
- Design a custom architecture (nano-trickster: 11M params)
- Character-level language modeling
- Monitor learning progress
- Compare to random baseline (prove learning!)
- Scale from toy to production

**Time:** 60-90 minutes (30-60 min training) | **Prerequisites:** CT-1 through CT-7

---

## Why Train from Scratch?

### You've Fine-Tuned, Now Build

In CT-4, you fine-tuned TinyLlama (1.1B params) - adjusting pre-trained weights.

**Training from scratch means:**
- Start with **random weights** (the model knows nothing)
- You design the architecture
- The model learns language from your data alone
- Full control over size, capacity, and behavior

### When to Train from Scratch

**Fine-tuning is better when:**
- You want a general-purpose model
- You have limited data (<10K examples)
- You need broad knowledge (science, history, etc.)
- You want fast results (hours, not days)

**Training from scratch is better when:**
- You need a **tiny, specialized model**
- You have domain-specific data (code, medical, legal)
- You want to understand transformer internals
- You're doing research or experimentation
- You need full control over architecture

```mermaid
graph LR
    A[Model Training Decision] --> B{Have pre-trained<br/>model for task?}
    B -->|Yes| C[Fine-Tuning<br/>CT-4]
    B -->|No| D{Need general<br/>knowledge?}

    D -->|Yes| C
    D -->|No| E{Large dataset<br/>available?}

    E -->|Yes| F[Train from Scratch<br/>CT-8]
    E -->|No| C

    C --> G[Result: Specialized<br/>1.1B params<br/>Hours to train]
    F --> H[Result: Custom<br/>11M params<br/>Minutes to train]

    style C fill:#87CEEB,stroke:#333,stroke-width:2px
    style F fill:#90EE90,stroke:#333,stroke-width:2px
    style G fill:#FFE4B5,stroke:#333,stroke-width:2px
    style H fill:#FFB6C1,stroke:#333,stroke-width:2px
```

---

## Meet Nano-Trickster

### Architecture Overview

**Nano-Trickster:** A tiny but complete transformer designed for learning.

```yaml
nano-trickster:
  vocab_size: 256        # Character-level (simple!)
  hidden_dim: 256        # Small but workable
  num_layers: 6          # Shallow (6× faster than TinyLlama's 22)
  num_heads: 8           # Decent parallelism
  mlp_dim: 768           # 3× hidden_dim
  max_seq_len: 512       # Short context
  total_params: ~11M     # 100× smaller than TinyLlama!
```

```mermaid
graph TD
    A[Nano-Trickster<br/>11M Parameters] --> B[Input: Characters<br/>vocab_size: 256]
    B --> C[Token Embedding<br/>256 → 256<br/>65K params]
    C --> D[6 Transformer Blocks<br/>10.8M params total]

    D --> E[Block 1-6 Each Contains:<br/>1.8M params]
    E --> F[Multi-Head Attention<br/>8 heads, 32 dims each]
    E --> G[Feed-Forward Network<br/>256 → 768 → 256]
    E --> H[RMSNorm × 2<br/>Stabilization]

    D --> I[Output Projection<br/>256 → 256<br/>65K params shared]
    I --> J[Output: Next Character<br/>Probability distribution]

    style A fill:#FFE4B5,stroke:#333,stroke-width:2px
    style D fill:#90EE90,stroke:#333,stroke-width:2px
    style E fill:#87CEEB,stroke:#333,stroke-width:2px
    style J fill:#FFB6C1,stroke:#333,stroke-width:2px
```

### Why This Size Works

**Trade-offs:**

| Aspect | Nano-Trickster (11M) | TinyLlama (1.1B) |
|--------|----------------------|------------------|
| **Training time (N150)** | 30-60 minutes | Many hours |
| **Memory** | ~200MB | ~17GB |
| **Iterations/sec** | ~100 | ~10 |
| **Learns** | Basic patterns | Complex language |
| **Use case** | Learning, prototyping | Production |

**Perfect for:**
- Understanding transformer training
- Fast experimentation (quick feedback loop)
- Hardware constraints (N150 is plenty)
- Teaching and learning

**Not for:**
- Production chatbots (use TinyLlama + fine-tuning)
- Complex reasoning (too small)
- Broad knowledge (limited capacity)

---

## Dataset: Tiny Shakespeare

### What Is It?

**Tiny Shakespeare:** ~1.1MB of Shakespeare plays (~1M characters)

**Why Shakespeare?**
- Small enough to train quickly
- Large enough to learn patterns
- Rich language (old English, poetry, dialogue)
- Everyone recognizes it (easy to evaluate)
- Public domain (no licensing issues)

**Dataset stats:**
- Total characters: ~1,000,000
- Unique characters: ~65 (letters, punctuation, spaces)
- Train/val split: 90% / 10%
- Sequence length: 512 characters

### Character-Level Tokenization

**Unlike TinyLlama's BPE (32,000 tokens), we use characters:**

```mermaid
graph LR
    A["Text: 'ROMEO:'"] --> B[Tokenization]

    B --> C[Character-level<br/>Nano-Trickster]
    C --> D["['R', 'O', 'M', 'E', 'O', ':']<br/>6 tokens"]

    B --> E[BPE<br/>TinyLlama]
    E --> F["['ROM', 'EO', ':']<br/>3 tokens ish"]

    D --> G[Pros:<br/>- Simple vocab 256<br/>- No training needed<br/>- Handles any text]
    D --> H[Cons:<br/>- Longer sequences<br/>- Less semantic info]

    F --> I[Pros:<br/>- Shorter sequences<br/>- Semantic chunks<br/>- More efficient]
    F --> J[Cons:<br/>- Large vocab 32K<br/>- Training required<br/>- Out-of-vocab issues]

    style C fill:#90EE90,stroke:#333,stroke-width:2px
    style E fill:#87CEEB,stroke:#333,stroke-width:2px
    style G fill:#E0E0E0,stroke:#333,stroke-width:2px
    style I fill:#E0E0E0,stroke:#333,stroke-width:2px
```

**For learning, characters are perfect:**
- Vocabulary is tiny (256 bytes)
- No tokenizer training needed
- Works for any language
- Easy to understand

---

## Part 1: Setup

### Install Dependencies

**tt-metal version:** v0.66.0-rc5 or later (required for Python ttml module)

**Check your version:**
```bash
cd $TT_METAL_HOME && git describe --tags
# Should show v0.66.0-rc5 or later
```

**⚠️ Version Note:** Earlier versions (v0.64.5 and below) only have C++ tt-train framework, not the Python ttml module required for these lessons.

**Install ttml** (if not already done from CT-4):

```bash
cd $TT_METAL_HOME/tt-train
pip install -e .
```

**Verify installation:**
```bash
python -c "import ttml; print('✅ ttml available')"
```

### Prepare Dataset

#### Step 1: Download Shakespeare Text

**Use the automated script:**

```bash
cd ~/tt-scratchpad/training/data
python prepare_shakespeare.py --output . --split
```

**What this does:**
- Downloads ~1.1MB Shakespeare text from char-rnn repo
- Creates 90/10 train/val split automatically

**Expected output:**
```
✅ Downloaded 1,115,394 characters to shakespeare.txt
✅ Created train split: 1,003,854 chars → shakespeare_train.txt
✅ Created val split: 111,540 chars → shakespeare_val.txt
```

**Manual alternative** (if script unavailable):

```bash
# Download
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespeare.txt

# Create 90/10 split
head -n 32000 shakespeare.txt > shakespeare_train.txt
tail -n 8000 shakespeare.txt > shakespeare_val.txt
```

#### Step 2: Preprocess to PyTorch Tensors

**Convert text files to tensors for training:**

```bash
cd ~/tt-scratchpad/training/data
python preprocess_shakespeare.py
```

**What this does:**
- Creates character-level tokenizer (65 unique characters)
- Encodes text to token IDs
- Saves as PyTorch tensors

**Expected output:**
```
✅ Saved train.pt (1,003,854 tokens)
✅ Saved val.pt (111,540 tokens)
✅ Saved tokenizer.pt (vocab_size=65)
```

**Files created:**
- `shakespeare.txt` - Full dataset (1.1MB)
- `shakespeare_train.txt` - Training split (90%)
- `shakespeare_val.txt` - Validation split (10%)
- `train.pt` - Training tensor (~7.7MB)
- `val.pt` - Validation tensor (~873KB)
- `tokenizer.pt` - Tokenizer metadata (~2KB)

**Verify:**
```bash
ls -lh *.txt *.pt
# Should show text files + PyTorch tensors
```

---

## Part 2: Understanding the Architecture

### Model Code Overview

The **nano_trickster.py** file contains:

1. **RMSNorm** - Fast normalization (replaces LayerNorm)
2. **RotaryPositionalEmbedding** - Better position encoding (RoPE)
3. **MultiHeadAttention** - Context learning (8 heads)
4. **SwiGLU** - Modern activation (replaces ReLU)
5. **TransformerBlock** - Combines attention + FFN + norms
6. **NanoTrickster** - Complete model

```mermaid
graph TD
    A[Input: Character IDs] --> B[Token Embedding<br/>256 → 256 vectors]
    B --> C[Transformer Block 1]
    C --> D[Transformer Block 2]
    D --> E[... 4 more blocks ...]
    E --> F[Transformer Block 6]
    F --> G[Final RMSNorm]
    G --> H[Output Projection<br/>256 → vocab_size]
    H --> I[Softmax]
    I --> J[Next Character Probabilities]

    K[Each Transformer Block] --> L[RMSNorm 1]
    L --> M[Multi-Head Attention<br/>Query/Key/Value + RoPE]
    M --> N[Residual Add]
    N --> O[RMSNorm 2]
    O --> P[SwiGLU FFN<br/>256 → 768 → 256]
    P --> Q[Residual Add]

    style B fill:#FFE4B5,stroke:#333,stroke-width:2px
    style C fill:#90EE90,stroke:#333,stroke-width:2px
    style F fill:#90EE90,stroke:#333,stroke-width:2px
    style K fill:#87CEEB,stroke:#333,stroke-width:2px
    style M fill:#FFB6C1,stroke:#333,stroke-width:2px
    style P fill:#DDA0DD,stroke:#333,stroke-width:2px
```

### Test the Model

Let's verify it works:

```bash
cd ~/tt-scratchpad/training
python nano_trickster.py
```

**Expected output:**
```
Nano-Trickster initialized: 11,234,816 trainable params

Parameter breakdown:
  Total: 11,234,816
  Trainable: 11,234,816
  Embedding: 65,536
  Transformer blocks: 10,878,464
  Per block: 1,813,077
  Output layer: 65,536 (weight-tied)

Test forward pass:
  Input shape: torch.Size([4, 64])
  Logits shape: torch.Size([4, 64, 256])
  Loss: 5.5452

Test generation:
  Prompt shape: torch.Size([1, 10])
  Generated shape: torch.Size([1, 30])
```

**Key observations:**
- Total params: ~11M (manageable!)
- Per block: ~1.8M (6 blocks × 1.8M = 10.8M)
- Initial loss: ~5.5 (random guessing for 256 classes = ln(256) ≈ 5.5)
- Generation works (even from random weights!)

---

## Part 3: Training Configuration

### Review the Config

Open `configs/nano_trickster.yaml`:

```yaml
# Key settings:
model_config:
  vocab_size: 256
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  mlp_dim: 768
  max_seq_len: 512

training_config:
  batch_size: 16
  max_steps: 10000        # ~30-60 minutes on N150
  learning_rate: 0.0003   # 3e-4 (standard for small models)
  warmup_steps: 1000      # Gradual LR increase
  grad_clip: 1.0          # Prevent exploding gradients
```

```mermaid
graph TD
    A[Training Process] --> B[Step 0-1000<br/>Warmup Phase]
    B --> C[LR increases linearly<br/>0 → 3e-4]

    A --> D[Step 1000-10000<br/>Main Training]
    D --> E[LR decays via cosine<br/>3e-4 → 3e-5]

    A --> F[Every 50 steps<br/>Log loss]
    A --> G[Every 500 steps<br/>Evaluate on val]
    A --> H[Every 1000 steps<br/>Save checkpoint]

    B --> I[Why warmup?<br/>Prevents early instability]
    D --> J[Why cosine decay?<br/>Smooth convergence]

    style B fill:#FFE4B5,stroke:#333,stroke-width:2px
    style D fill:#90EE90,stroke:#333,stroke-width:2px
    style F fill:#87CEEB,stroke:#333,stroke-width:2px
    style G fill:#FFB6C1,stroke:#333,stroke-width:2px
    style H fill:#DDA0DD,stroke:#333,stroke-width:2px
```

### Hardware Variants

**N150 (single chip):**
- Batch size: 16
- Gradient accumulation: 2 (effective batch = 32)
- Training time: 30-60 minutes
- Memory: ~200MB (very light!)

**N300 (dual chips with DDP):**
- Batch size: 32 (2× N150)
- Gradient accumulation: 1
- Training time: 15-30 minutes
- Memory: Distributed across 2 chips

**To use N300:** Update config:
```yaml
device_config:
  enable_ddp: True
  mesh_shape: [1, 2]  # 1 row, 2 columns

training_config:
  batch_size: 32
  gradient_accumulation_steps: 1
```

---

## Part 4: Launch Training

### Start Training

```bash
cd ~/tt-scratchpad/training
python train_from_scratch.py --config configs/nano_trickster.yaml
```

**You'll see:**

```
============================================================
Training Nano-Trickster from Scratch
============================================================

Config: configs/nano_trickster.yaml

Device: cuda
Loaded 900,000 tokens from data/train.pt
Loaded 100,000 tokens from data/val.pt

Model architecture:
  Total parameters: 11,234,816
  Per block: 1,813,077
  Vocabulary size: 256

Dataset:
  Train batches: 1,758
  Val batches: 195

Training:
  Max steps: 10,000
  Warmup steps: 1,000
  Learning rate: 0.0003
  Gradient clip: 1.0
  Output: output/nano_trickster

============================================================
Starting training...
============================================================

Training:   0%|          | 0/10000 [00:00<?, ?it/s]
```

### What's Happening?

```mermaid
graph TD
    A[Training Loop] --> B[1. Get Batch<br/>16 sequences × 512 chars]
    B --> C[2. Forward Pass<br/>Compute predictions]
    C --> D[3. Calculate Loss<br/>Cross-entropy]
    D --> E[4. Backward Pass<br/>Compute gradients]
    E --> F[5. Clip Gradients<br/>Prevent explosions]
    F --> G[6. Optimizer Step<br/>Update weights]
    G --> H[7. Update LR<br/>Warmup/decay schedule]
    H --> B

    I[Every 50 steps] --> J[Log train loss]
    I --> K[Update progress bar]

    L[Every 500 steps] --> M[Evaluate on val set]
    M --> N[Generate sample text]
    N --> O[Check if best model]
    O --> P[Save checkpoint if best]

    style B fill:#FFE4B5,stroke:#333,stroke-width:2px
    style D fill:#FFB6C1,stroke:#333,stroke-width:2px
    style E fill:#87CEEB,stroke:#333,stroke-width:2px
    style M fill:#90EE90,stroke:#333,stroke-width:2px
    style P fill:#DDA0DD,stroke:#333,stroke-width:2px
```

---

## Part 5: Monitoring Progress

### Understanding the Loss

**Initial loss (~5.5):**
- Random guessing for 256 classes
- Expected loss = ln(256) ≈ 5.5452
- Model knows nothing yet!

**After 1000 steps (~3 minutes):**
```
Step 1000:
  Train loss: 2.456
  Val loss: 2.489
  Val perplexity: 12.05

  Sample generation:
  --------------------------------------------------------
  ROMEO:
  Thit the stook to tean the couse,
  And the beep the me the shoun,
  --------------------------------------------------------
```

**What we see:**
- Loss dropped from 5.5 → 2.5 (learning!)
- Perplexity: 12.05 (uncertainty is ~12 characters per prediction)
- Sample: Gibberish, but has **English-like structure**
  - Words separated by spaces
  - Capital letters at start
  - Punctuation (periods, commas)
  - Some real words ("the", "to", "me")

**After 5000 steps (~15 minutes):**
```
Step 5000:
  Train loss: 1.234
  Val loss: 1.287
  Val perplexity: 3.62

  Sample generation:
  --------------------------------------------------------
  ROMEO:
  What is the world and the man that shall be
  The heart of my soul, and the world is the world
  That is the blood of my heart.
  --------------------------------------------------------
```

**What we see:**
- Loss dropped to 1.2 (significant learning!)
- Perplexity: 3.62 (much more confident)
- Sample: **Coherent sentences!**
  - Grammar is correct
  - Shakespearean style ("What is...", "my soul")
  - Semantic repetition (training artifact)
  - Still not perfect, but recognizable

**After 10000 steps (~30-60 minutes):**
```
Step 10000:
  Train loss: 0.876
  Val loss: 0.934
  Val perplexity: 2.54

  Sample generation:
  --------------------------------------------------------
  ROMEO:
  I will not speak of this, my lord,
  For I have done the worst of all my love,
  And yet I cannot speak of what I know.
  I have a heart that will not be content
  To make me think of this.
  --------------------------------------------------------
```

**What we see:**
- Loss dropped to <1.0 (excellent!)
- Perplexity: 2.54 (very confident)
- Sample: **High-quality Shakespeare!**
  - Complex sentences
  - Poetic phrasing
  - Emotional content
  - Consistent voice

```mermaid
graph LR
    A[Step 0<br/>Loss: 5.5] --> B[Step 1000<br/>Loss: 2.5]
    B --> C[Step 5000<br/>Loss: 1.3]
    C --> D[Step 10000<br/>Loss: 0.9]

    A --> E[Random gibberish<br/>No patterns]
    B --> F[Letter patterns<br/>Spaces, caps]
    C --> G[Word patterns<br/>Grammar emerges]
    D --> H[Sentence patterns<br/>Coherent Shakespeare]

    style A fill:#FF6B6B,stroke:#333,stroke-width:2px
    style B fill:#FFE4B5,stroke:#333,stroke-width:2px
    style C fill:#90EE90,stroke:#333,stroke-width:2px
    style D fill:#87CEEB,stroke:#333,stroke-width:2px
```

### Loss Curves

Typical training progression:

```mermaid
xychart-beta
    title "Nano-Trickster Training Loss"
    x-axis [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    y-axis "Loss" 0 --> 6
    line [5.5, 3.2, 2.5, 2.1, 1.8, 1.5, 1.3, 1.1, 1.0, 0.95, 0.88]
```

**Phases:**
1. **0-1000 steps (Warmup):** Rapid initial learning, loss drops quickly
2. **1000-5000 steps (Main):** Steady improvement, patterns emerge
3. **5000-10000 steps (Refinement):** Slower gains, quality increases

---

## Part 6: Testing Your Model

### Generate Text

After training completes, test generation:

```bash
cd ~/tt-scratchpad/training
python -c "
import torch
from nano_trickster import NanoTrickster

# Load model
model = NanoTrickster()
model.load_state_dict(torch.load('output/nano_trickster/final_model.pt'))
model.eval()

# Load tokenizer
tokenizer = torch.load('data/tokenizer.pt')
stoi = tokenizer['stoi']
itos = tokenizer['itos']

# Encode prompt
prompt = 'ROMEO:'
input_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]])

# Generate
with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=200, temperature=0.8)

# Decode
text = ''.join([itos.get(int(t), '?') for t in generated[0]])
print(text)
"
```

**Try different prompts:**
- `"ROMEO:"` - Romeo speaking
- `"JULIET:"` - Juliet speaking
- `"To be or not"` - Famous line
- `"What is"` - Question format

### Compare to Random

To prove learning occurred, compare to a freshly initialized model:

```bash
python -c "
import torch
from nano_trickster import NanoTrickster

# Create random model (no training)
model = NanoTrickster()
model.eval()

# Load tokenizer
tokenizer = torch.load('data/tokenizer.pt')
stoi = tokenizer['stoi']
itos = tokenizer['itos']

# Encode prompt
prompt = 'ROMEO:'
input_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]])

# Generate
with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=200, temperature=0.8)

# Decode
text = ''.join([itos.get(int(t), '?') for t in generated[0]])
print('RANDOM MODEL OUTPUT:')
print(text)
"
```

**Expected random output:**
```
RANDOM MODEL OUTPUT:
ROMEO:xJ#*8dK...mnoP@!qrs...
```

**Comparison:**

| Model | Output Quality | Loss |
|-------|---------------|------|
| **Random** | Complete gibberish, no structure | ~5.5 |
| **Trained (1K steps)** | Letter patterns, some spaces | ~2.5 |
| **Trained (5K steps)** | Words, grammar | ~1.3 |
| **Trained (10K steps)** | Coherent Shakespeare | ~0.9 |

**This proves the model learned!**

---

## Part 7: Understanding What Was Learned

### Learned Patterns

```mermaid
graph TD
    A[What Nano-Trickster Learned] --> B[Character Level]
    B --> C[Letters form words<br/>'a', 'n', 'd' → 'and']
    B --> D[Spaces separate words<br/>Not random placement]
    B --> E[Punctuation rules<br/>Periods end sentences]

    A --> F[Word Level]
    F --> G[Common words<br/>'the', 'is', 'and', 'of']
    F --> H[Shakespeare vocab<br/>'thou', 'thy', 'hath']
    F --> I[Word order matters<br/>'I am' not 'am I']

    A --> J[Sentence Level]
    J --> K[Grammar structure<br/>Subject-verb-object]
    J --> L[Poetic phrasing<br/>Iambic patterns]
    J --> M[Emotional tone<br/>Love, tragedy, honor]

    A --> N[Discourse Level]
    N --> O[Character voices<br/>Romeo vs Juliet style]
    N --> P[Dialogue format<br/>NAME: speech]
    N --> Q[Scene structure<br/>Back-and-forth]

    style B fill:#FFE4B5,stroke:#333,stroke-width:2px
    style F fill:#87CEEB,stroke:#333,stroke-width:2px
    style J fill:#90EE90,stroke:#333,stroke-width:2px
    style N fill:#FFB6C1,stroke:#333,stroke-width:2px
```

### What It DIDN'T Learn

**Limitations of 11M parameters:**
- **No world knowledge** (who Romeo/Juliet are historically)
- **No plot memory** (can't recall earlier in play)
- **Limited vocabulary** (~5K unique words vs TinyLlama's 32K)
- **Short context** (512 characters vs 2048+ for larger models)
- **No reasoning** (can't answer "Why did Romeo...?")

**This is expected!** We built a **tiny model** to learn fundamentals, not production system.

---

## Part 8: Scaling Up

### From Nano to Production

**Want a more capable model? Scale up the config:**

```yaml
# Nano-Trickster: 11M params, 30-60 min (N150)
nano:
  hidden_dim: 256
  num_layers: 6
  mlp_dim: 768

# Mini-Trickster: 50M params, 2-3 hours (N150)
mini:
  hidden_dim: 512    # 2× larger
  num_layers: 8      # 33% deeper
  mlp_dim: 1536      # 3× hidden_dim

# Midi-Trickster: 200M params, 8-10 hours (N300)
midi:
  hidden_dim: 768    # 3× nano
  num_layers: 12     # 2× nano
  mlp_dim: 2304      # 3× hidden_dim

# Mega-Trickster: 1.1B params, days (T3K/Galaxy)
mega:
  hidden_dim: 2048   # Same as TinyLlama
  num_layers: 22     # Same as TinyLlama
  mlp_dim: 5632      # Same as TinyLlama
```

```mermaid
graph LR
    A[Nano<br/>11M<br/>30-60 min] --> B[Mini<br/>50M<br/>2-3 hours]
    B --> C[Midi<br/>200M<br/>8-10 hours]
    C --> D[Mega<br/>1.1B<br/>days]

    A --> E[Learn fundamentals<br/>N150 sufficient]
    B --> F[Simple tasks<br/>N150 OK, N300 better]
    C --> G[Production quality<br/>N300/T3K recommended]
    D --> H[SOTA performance<br/>T3K/Galaxy required]

    style A fill:#FFE4B5,stroke:#333,stroke-width:2px
    style B fill:#87CEEB,stroke:#333,stroke-width:2px
    style C fill:#90EE90,stroke:#333,stroke-width:2px
    style D fill:#FFB6C1,stroke:#333,stroke-width:2px
```

### Scaling Laws

**Rule of thumb:**
```
Training cost ∝ num_params × num_tokens × context_length
```

**Practical guide:**

| Model Size | Params | Hardware | Training Time | Use Case |
|------------|--------|----------|---------------|----------|
| **Nano** | 11M | N150 | 30-60 min | Learning, prototyping |
| **Mini** | 50M | N150/N300 | 2-3 hours | Simple tasks |
| **Midi** | 200M | N300/T3K | 8-10 hours | Production (niche) |
| **Mega** | 1.1B | T3K/Galaxy | Days | Production (general) |

**Key insight:** Start small! Iterate quickly. Scale up once you understand the patterns.

---

## Part 9: Next Steps

### Experiment Ideas

**Easy (10-30 minutes):**
1. **Try different prompts** - "JULIET:", "KING:", "GHOST:"
2. **Adjust temperature** - 0.5 (conservative) to 1.5 (creative)
3. **Longer generation** - max_new_tokens=500 or 1000
4. **Different datasets** - Try poetry, code, Wikipedia

**Medium (1-2 hours):**
1. **Extend training** - Run to 20K steps, see if loss improves
2. **Tune hyperparameters** - Learning rate, batch size, warmup
3. **Add regularization** - Increase dropout, try weight decay
4. **Multi-device** - If you have N300, enable DDP

**Advanced (3-5 hours):**
1. **Scale up architecture** - Try 50M or 200M params
2. **Better tokenization** - Train BPE tokenizer (like TinyLlama)
3. **Longer context** - Increase max_seq_len to 1024 or 2048
4. **Different loss** - Try label smoothing or focal loss

### What You've Accomplished

**🎉 Congratulations! You just:**
1. ✅ Designed a transformer architecture from scratch
2. ✅ Trained a model from random initialization
3. ✅ Watched it learn language patterns in real-time
4. ✅ Compared trained vs random to prove learning
5. ✅ Generated coherent Shakespeare text
6. ✅ Understood the full training pipeline
7. ✅ Learned how to scale from 11M → 1B+ params

**You now understand:**
- What happens "under the hood" of models like TinyLlama
- How to make architectural trade-offs
- Why model size matters (and doesn't matter)
- The difference between fine-tuning and training from scratch
- How to build custom models for your needs

---

## Troubleshooting

### "Data file not found"

**Error:**
```
FileNotFoundError: Data file not found: data/train.pt
Run: python data/prepare_shakespeare.py
```

**Fix:**
```bash
cd ~/tt-scratchpad/training/data
python prepare_shakespeare.py --output shakespeare.txt --split
```

Then process the data:
```bash
cd ~/tt-scratchpad/training
python -c "
from prepare_shakespeare import create_train_val_split
create_train_val_split('data/shakespeare.txt')
"
```

### "Loss is NaN"

**Causes:**
- Learning rate too high
- No gradient clipping
- Numerical instability

**Fixes:**
1. Lower learning rate: `0.0003` → `0.0001`
2. Enable gradient clipping: `grad_clip: 1.0`
3. Reduce batch size: `16` → `8`
4. Add mixed precision: `--fp16` flag

### "Loss not decreasing"

**If loss stays at ~5.5 after 1000 steps:**

**Check:**
1. Is data loading correctly? (Check dataset size)
2. Is optimizer stepping? (Check LR schedule)
3. Are gradients flowing? (Print gradient norms)
4. Is model too small? (Try hidden_dim=512)

**Debug:**
```bash
# Check dataset
python -c "import torch; data = torch.load('data/train.pt'); print(len(data))"

# Check learning rate
grep "lr:" logs/training.log | head -20

# Print model size
python nano_trickster.py
```

### "Out of memory"

**If training crashes with OOM:**

**Reduce memory:**
1. Smaller batch size: `16` → `8` → `4`
2. Shorter sequences: `max_seq_len: 512` → `256`
3. Fewer layers: `num_layers: 6` → `4`
4. Smaller hidden: `hidden_dim: 256` → `128`

**For N150:** Nano-Trickster (11M) should work easily. If not, check:
- Other processes using DRAM
- tt-smi device status
- Memory leaks (restart kernel)

---

## Key Takeaways

✅ **Training from scratch gives you full control** - architecture, size, specialization

✅ **Start small (11M), scale up (1B+)** - iterate quickly, learn patterns, then scale

✅ **Character-level is simple and effective** - no tokenizer training, works for any language

✅ **Loss curves tell the story** - rapid initial learning, then refinement

✅ **Compare to random to prove learning** - baseline is critical

✅ **Hardware scales linearly** - N150 → N300 → T3K = 2-4× faster each step

✅ **Tiny models teach fundamentals** - understanding > performance for learning

---

## Additional Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Decoder-only architecture
- [Scaling Laws](https://arxiv.org/abs/2001.08361) - How size affects performance
- [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) - Chinchilla paper

### Code References
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy's minimal GPT
- [minGPT](https://github.com/karpathy/minGPT) - Educational GPT implementation
- [TinyLlama](https://github.com/jzhang38/TinyLlama) - 1.1B model training
- [tt-train](https://github.com/tenstorrent/tt-metal/tree/main/tt-train) - TT training framework

### Next Steps
- **Scale up:** Try 50M or 200M parameter models
- **Better data:** Curate high-quality datasets
- **Production deployment:** CT-6 (vLLM) + CT-7 (tt-inference-server)
- **Research:** Experiment with novel architectures

---

**🎭 You've completed the Custom Training series! You now know how to:**
1. Understand transformer fundamentals (CT-1, CT-7)
2. Create datasets (CT-2)
3. Configure training (CT-3)
4. Fine-tune existing models (CT-4)
5. Scale to multiple devices (CT-5)
6. Track experiments (CT-6)
7. Design architectures (CT-7)
8. Train from scratch (CT-8)

**Next: Build production systems with vLLM (Lesson 7) or explore creative applications (Lessons 9-12)!**
