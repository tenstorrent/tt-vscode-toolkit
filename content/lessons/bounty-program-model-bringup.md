---
id: bounty-program
title: 'Bounty Program: Model Bring-Up'
description: >-
  Learn how to contribute to the Tenstorrent Bounty Program by bringing up new
  models. Master TT-Metal while becoming part of the open-source ecosystem. Uses
  the successful Phi-3 contribution as a case study.
category: advanced
tags:
  - model
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
status: draft
estimatedMinutes: 10
---

# Bounty Program: Model Bring-Up from Scratch

Learn how to tackle open bounties in the **Tenstorrent Bounty Program** by bringing up a new model on TT hardware. We'll use the successful **Phi-3-mini-128k-instruct** bounty (Issue #19416) as a real-world case study.

---

## What is the Bounty Program?

Tenstorrent's bounty program rewards contributors (ranging from $500–$3000) for bringing new AI models to their hardware. Contributors are recognized for:

1. **Model functionality** - Compiles and runs end-to-end inference
2. **Performance** - Meets throughput benchmarks (25%/50%/70% of theoretical max)
3. **Accuracy** - Validated against CPU baseline (top-1 >80%, top-5 >95%)
4. **Documentation** - Clear build/run instructions for the community

### Why Participate?

- ✅ **Master cutting-edge technology** - Deep dive into TT-Metal/TT-NN architecture
- ✅ **Real-world impact** - Your code ships in production and helps the community
- ✅ **Build ownership** - Public contributions to an open-source hardware ecosystem
- ✅ **Join the community** - Work alongside Tenstorrent engineers and contributors
- ✅ **Develop expertise** - Deep learning, hardware acceleration, systems programming

---

## Bounty Difficulty Tiers

| Difficulty | Complexity | Scope |
|------------|------------|-------|
| **Warmup** | First-time contributor tasks | Getting familiar with the codebase |
| **Easy** | Basic repo familiarity | Straightforward implementations |
| **Medium** | Significant domain knowledge | Complex integrations |
| **Hard** | Deep architectural expertise | Novel architectures or optimizations |

**Performance-based tiers** (for model bring-up):
- **Easy**: ≥25% of theoretical max throughput
- **Medium**: ≥50% of theoretical max throughput
- **Hard**: ≥70% of theoretical max throughput

---

## Case Study: Phi-3-mini-128k-instruct (Issue #19416)

**Model**: microsoft/Phi-3-mini-128k-instruct (3.8B parameters)
**Hardware**: N150 / N300 / LoudBox
**Theoretical Max**: 48 tokens/second/user
**Result**: ✅ **Successfully merged to main** - Now part of tt-metal

### Timeline

```text
May 1:   Contributor (ign-msati) expresses interest
May 2:   Officially assigned to issue
May 12:  Individual blocks functional, full network integration underway
May 26:  Testing with varied prefill lengths
May 29:  Pull request #22716 submitted
Status:  MERGED ✅ Contribution accepted
```

### Key Success Factors

1. **Reused existing framework** - Leveraged `tt_transformers` instead of duplicating code
2. **Minimal modifications** - Extended RoPE scaling, adjusted chunk sizes
3. **Component-wise bring-up** - Tested individual modules before full model
4. **Thorough testing** - Unit tests, performance benchmarks, accuracy validation
5. **Clear communication** - Regular updates to issue thread

---

## Step-by-Step: Bringing Up a Model

### Phase 1: Setup & Preparation

#### 1.1 Find a Bounty

**Browse open bounties:**
```bash
# Visit GitHub issues page
open https://github.com/tenstorrent/tt-metal/labels/bounty
```

**Filter by difficulty:**
- Look for `bounty_difficulty/easy`, `bounty_difficulty/medium`, `bounty_difficulty/hard` labels
- Choose based on your experience level
- Read requirements carefully

**Get assigned:**
- Comment on the issue expressing interest
- Wait for official assignment (required before submitting PR)
- Assignment times out after 2 weeks of inactivity

#### 1.2 Set Up Environment

**Clone and build tt-metal:**
```bash
cd ~
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git submodule update --init --recursive

# Build TT-Metal (takes 10-20 minutes)
./build_metal.sh

# Set environment variables
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

**Install dependencies:**
```bash
# Python requirements
pip install -r requirements.txt
pip install -r models/tt_transformers/requirements.txt

# Additional tools
pip install pytest huggingface-hub
```

**Verify hardware:**
```bash
tt-smi  # Should detect your TT device
```

#### 1.3 Run a Reference Demo

**Test the environment with a proven model:**
```bash
# Download a working model (e.g., Llama 3.1 8B)
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export MESH_DEVICE=N150  # or N300, T3K, etc.

# Run demo to verify setup
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

**What to expect:**
- First run: Downloads model (~16GB), creates weight cache (2-5 min)
- Subsequent runs: Fast inference (~1-3 sec per query)
- **If this works, your environment is ready!**

---

### Phase 2: Baseline Validation

#### 2.1 Run Reference Model on CPU

**Critical first step:** Ensure the model works correctly on CPU/GPU before attempting TT hardware.

```bash
# Create a reference validation script
cd ~/tt-scratchpad
```

**Example reference script (save as `validate_reference.py`):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Run inference
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Response: {response}")

# Save logits for later comparison
torch.save(outputs, "reference_outputs.pt")
```

**Run validation:**
```bash
python validate_reference.py
```

#### 2.2 Analyze Model Architecture

**Inspect model configuration:**
```bash
huggingface-cli download microsoft/Phi-3-mini-128k-instruct config.json --local-dir ~/models/Phi-3
cat ~/models/Phi-3/config.json
```

**Key questions to answer:**
1. **Architecture type**: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Phi3ForCausalLM?
   - ✅ If listed in tt_transformers/README.md, likely compatible!
2. **Model dimensions**: hidden_size, num_attention_heads, num_layers
   - Check if tile-aligned (divisible by 32 for TT hardware)
3. **Special features**: RoPE scaling, sliding window attention, custom tokens?
4. **Hardware fit**: Will it fit on target device? (N150 = 12GB, N300 = 24GB)

**For Phi-3:**
- Architecture: `Phi3ForCausalLM` ✅ (supported in tt_transformers)
- Size: 3.8B parameters ✅ (fits on N150)
- Context: 128K tokens (requires chunked prefill)
- Special: SUlongRoPE (long-context scaling) - requires modification

---

### Phase 3: Component-Wise Bring-Up

**Philosophy**: Test small pieces before the full model. This is the **MOST IMPORTANT** phase.

#### 3.1 Identify Similar Models

**Find the closest match in tt_transformers:**
```bash
ls models/tt_transformers/model_params/
# Phi-3 architecture is similar to Llama
# Use Llama as the base implementation
```

#### 3.2 Bring Up Decode Stage First

**Why decode first?** Decode is simpler (batch=32, single token per user) and compute-bound.

**Create unit tests for individual modules:**

```python
# Example: Test RMSNorm module
# Save as tests/test_phi3_rmsnorm.py

import pytest
import torch
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.llama_transformer import RMSNorm  # Reuse from Llama

def test_rmsnorm():
    # Model dimensions for Phi-3
    hidden_dim = 3072

    # Create TT-NN version
    tt_norm = RMSNorm(device, dim=hidden_dim)

    # Create reference PyTorch version
    ref_norm = torch.nn.RMSNorm(hidden_dim)

    # Generate random input
    x = torch.randn(1, 1, hidden_dim)

    # Compare outputs
    tt_out = tt_norm(x)
    ref_out = ref_norm(x)

    # Check PCC (Pearson Correlation Coefficient)
    pcc = compute_pcc(tt_out, ref_out)
    assert pcc > 0.99, f"RMSNorm PCC too low: {pcc}"
```

**Test each module:**
- ✅ RMSNorm / LayerNorm
- ✅ RotaryEmbedding (RoPE)
- ✅ Attention (QKV projection, SDPA, output projection)
- ✅ MLP (feed-forward network)
- ✅ Full decoder layer

**Run unit tests:**
```bash
pytest tests/test_phi3_rmsnorm.py -v
pytest tests/test_phi3_attention.py -v
pytest tests/test_phi3_mlp.py -v
```

#### 3.3 Compose Full Decoder

**Once all modules pass, test the full decoder:**

```python
# tests/test_phi3_decoder.py

def test_full_decoder():
    # Create decoder with all modules
    decoder = TransformerBlock(
        mesh_device,
        model_args,
        layer_num=0
    )

    # Generate random activations and real weights
    x = torch.randn(32, 1, 3072)  # batch=32, seq=1, hidden=3072

    # Run through TT decoder
    tt_output = decoder(x, ...)

    # Run through reference decoder
    ref_output = reference_decoder(x, ...)

    # Check PCC
    pcc = compute_pcc(tt_output, ref_output)
    assert pcc > 0.98
```

#### 3.4 Handle Model-Specific Modifications

**For Phi-3, the main modification was RoPE scaling:**

**File: `models/tt_transformers/tt/rope.py`**
```python
# Original: Single scaling factor
scale = 1.0 / rope_scaling_factor

# Phi-3 modification: Support scaling tensor
if isinstance(rope_scaling, dict) and "long_factor" in rope_scaling:
    # SUlongRoPE uses different scales for different frequencies
    scale_tensor = compute_longrope_scale(rope_scaling)
else:
    scale = 1.0 / rope_scaling_factor
```

**File: `models/tt_transformers/tt/model_config.py`**
```python
# Add Phi-3 detection
if "Phi-3" in self.model_name:
    # Set prefill chunk size for long context
    self.min_prefill_chunk_size = 1024  # Lower for N150
```

**File: `models/tt_transformers/tt/common.py`**
```python
# Batch padding normalization
# Old: Pad each prompt independently to nearest power of 2
# New: Pad all prompts to max length across batch
max_len = max(len(p) for p in prompts)
padded_len = next_power_of_2(max_len)
```

**Key insight:** These are **MINIMAL** changes. Most of the implementation is reused from Llama!

---

### Phase 4: Full Model Integration

#### 4.1 Implement Prefill

**After decode works, add prefill (process initial prompt):**

Prefill is more complex:
- Batch=1 (single user)
- Processes up to 128K tokens at once
- Chunked into smaller pieces (4K, 64K, 128K depending on hardware)

```python
# tests/test_phi3_prefill.py

def test_prefill():
    # Long prompt (e.g., 2048 tokens)
    prompt = "..." * 2048

    # Run prefill
    logits = model.prefill(prompt)

    # Compare with reference
    ref_logits = reference_model.prefill(prompt)

    pcc = compute_pcc(logits, ref_logits)
    assert pcc > 0.98
```

#### 4.2 End-to-End Testing

**Test full generation (prefill + decode):**

```bash
# Run full demo
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct
pytest models/tt_transformers/demo/simple_text_demo.py -k "batch-1"
```

**What to check:**
1. **Does it generate coherent text?**
2. **Token accuracy**: Compare generated tokens to reference
3. **Top-1/Top-5 accuracy**: Measure against CPU baseline

#### 4.3 Teacher Forcing Validation

**Gold standard for accuracy testing:**

```python
# Teacher forcing: Force the model to use reference tokens at each step
# This isolates per-token accuracy without error accumulation

def test_teacher_forcing():
    reference_tokens = [1, 234, 567, ...]  # From reference model

    for i, ref_token in enumerate(reference_tokens):
        # Generate next token
        predicted_token = model.generate_token()

        # Check if it matches reference
        if predicted_token == ref_token:
            top1_matches += 1

        if ref_token in model.get_top_k(5):
            top5_matches += 1

        # Force feed the reference token (teacher forcing)
        model.forward(ref_token)

    top1_accuracy = top1_matches / len(reference_tokens)
    top5_accuracy = top5_matches / len(reference_tokens)

    assert top1_accuracy > 0.80  # Bounty requirement
    assert top5_accuracy > 0.95  # Bounty requirement
```

---

### Phase 5: Performance Optimization

#### 5.1 Measure Baseline Performance

```bash
# Run performance test
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-32" \
  --max_generated_tokens 200
```

**Key metrics:**
- **TTFT** (Time to First Token): How long until first token generated?
- **Throughput**: Tokens per second per user (t/s/u)
- **Latency**: Average time per token

**For Phi-3 on N150:**
- Theoretical max: ~48 t/s/u
- Easy tier: ≥12 t/s/u (25%)
- Medium tier: ≥24 t/s/u (50%)
- Hard tier: ≥34 t/s/u (70%)

#### 5.2 Apply Optimizations

**Precision tuning:**
```bash
# Try different precision configurations
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-32" \
  --optimizations 'precision_cfg = {ff1_3: bfp4, ff2: bfp4, wqkv: bfp8, wo: bfp8}'
```

**Create custom decoder config:**
```json
// models/tt_transformers/model_params/Phi-3-mini-128k-instruct/performance_decoder_config.json
{
  "decoder_0": {
    "ff1_3": "bfp4",
    "ff2": "bfp4",
    "wqkv": "bfp8",
    "wo": "bfp8"
  },
  // ... remaining decoders
}
```

**Advanced optimizations:**
- **Metal Trace**: Record and replay command buffers (reduces overhead)
- **Async mode**: Overlap host/device operations
- **Multiple command queues**: Parallelize independent ops

See: [Advanced Performance Optimizations](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)

#### 5.3 Profile and Debug

**Use Tracy profiler:**
```bash
# Build with Tracy support
cmake -B build -DENABLE_TRACY=ON
cmake --build build

# Run with profiling
pytest models/tt_transformers/demo/simple_text_demo.py -k "batch-32"
```

**Analyze bottlenecks:**
- Slow ops? Try different layouts (ROW_MAJOR, TILE)
- Memory-bound? Reduce precision (bfp8 → bfp4)
- Communication overhead? Optimize tensor parallelism

---

### Phase 6: Testing & CI Integration

#### 6.1 Create Test Suite

**Required tests for bounty submission:**

```bash
# Accuracy test
pytest models/tt_transformers/tests/test_accuracy.py -k "phi3"

# Performance test
pytest models/tt_transformers/tests/test_perf.py -k "phi3"

# Demo test (end-to-end)
pytest models/tt_transformers/demo/simple_text_demo.py -k "phi3"
```

#### 6.2 Generate Reference Logits

**For CI accuracy validation:**
```bash
# Generate reference outputs for CI
python models/tt_transformers/tests/generate_reference_hf.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --output reference_outputs/Phi-3-mini-128k-instruct.refpt
```

#### 6.3 Add CI Configuration

**Mark tests for CI execution:**
```python
# tests/test_ci_dispatch.py

@pytest.mark.parametrize(
    "model_name",
    ["Llama-3.1-8B-Instruct", "Phi-3-mini-128k-instruct"],  # Add your model
)
def test_model_demo(model_name):
    # CI will run this test on every commit
    ...
```

---

### Phase 7: Documentation & Submission

#### 7.1 Document Your Work

**Create or update README:**

```markdown
# Phi-3-mini-128k-instruct on Tenstorrent Hardware

## Overview
- Model: microsoft/Phi-3-mini-128k-instruct (3.8B parameters)
- Hardware: N150 / N300 / LoudBox
- Performance: 28 tokens/second/user on N150 (58% of theoretical max)

## Installation
\`\`\`bash
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct
pip install -r models/tt_transformers/requirements.txt
\`\`\`

## Running
\`\`\`bash
# Single user
pytest models/tt_transformers/demo/simple_text_demo.py -k "batch-1"

# Batch of 32 users
pytest models/tt_transformers/demo/simple_text_demo.py -k "batch-32"
\`\`\`

## Performance
| Hardware | Batch Size | Throughput (t/s/u) | TTFT (ms) |
|----------|------------|-------------------|-----------|
| N150     | 1          | 15.2              | 120       |
| N150     | 32         | 28.4              | 180       |
| N300     | 32         | 42.1              | 95        |

## Accuracy
- Top-1: 84.3%
- Top-5: 96.7%
(Tested on 512-token prefill + 511-token generation)
```

#### 7.2 Submit Pull Request

**PR checklist:**
- ✅ All tests pass locally
- ✅ Code follows tt-metal style (use existing patterns)
- ✅ No code duplication (reuses tt_transformers framework)
- ✅ Documentation includes build/run instructions
- ✅ Performance metrics documented
- ✅ Accuracy validation included

**PR structure (follow MODEL_ADD.md recommendations):**

**Option A: Single PR (small changes)**
```text
PR #1: Phi-3 model integration
- Core model code
- Unit tests
- Demo test
- Documentation
```

**Option B: Multi-PR (large changes)**
```text
PR #1: Phi-3 core model code + component tests
  → Run: post-commit + models nightly

PR #2: Phi-3 performance tests
  → Run: model perf + device perf

PR #3: Phi-3 demo test
  → Run: demo tests
```

**PR description template:**
```markdown
## Summary
Adds support for microsoft/Phi-3-mini-128k-instruct on N150/N300 hardware.

Closes #19416 (bounty issue)

## Changes
- Modified `rope.py` to support SUlongRoPE scaling
- Updated `model_config.py` for Phi-3 detection
- Added batch padding normalization in `common.py`

## Testing
- [x] Unit tests pass (test_phi3_*.py)
- [x] Accuracy test passes (84.3% top-1, 96.7% top-5)
- [x] Performance test passes (28 t/s/u on N150 = 58% theoretical)
- [x] Demo generates coherent text

## Performance
| Device | Throughput | Tier Achieved |
|--------|-----------|---------------|
| N150   | 28 t/s/u  | Medium (58% of theoretical) |

## Accuracy
- Top-1: 84.3% ✅ (>80% required)
- Top-5: 96.7% ✅ (>95% required)
```

#### 7.3 Respond to Review Feedback

**Common reviewer requests:**
1. **Make changes more general** - Can this work for other models too?
2. **Reduce code duplication** - Can you reuse existing functions?
3. **Add test coverage** - Missing edge cases?
4. **Fix CI failures** - Rebase on latest main

**Example from Phi-3 review:**
```text
Reviewer: "Does this need to be restricted to Phi-3-mini?"
Response: "Good point! Updated to support all Phi-3 variants (3.5, 4) with long_factor RoPE scaling."
```

---

## Applying This to Other Bounties

### Lesson Applicability

**The Phi-3 workflow applies to:**

#### 1. ✅ Transformer-Based LLMs
**Examples:**
- Phi-4 (current open bounty)
- Qwen models
- Mistral variants
- CodeLlama
- StarCoder

**Strategy:**
- Use tt_transformers as base
- Modify RoPE, attention, or MLP as needed
- Minimal changes maximize approval chances

#### 2. ✅ Vision Transformers
**Examples:**
- ViT (Vision Transformer)
- CLIP
- DINO
- SAM (Segment Anything)

**Strategy:**
- Similar to LLMs but with image patches
- Reuse attention mechanisms
- Add vision-specific preprocessing

#### 3. ✅ Diffusion Models
**Examples:**
- Stable Diffusion variants
- ControlNet
- LCM (Latent Consistency Models)

**Strategy:**
- Iterative denoising process
- U-Net architecture
- See stable_diffusion_35_large example

#### 4. ⚠️ Novel Architectures (Harder)
**Examples:**
- Mamba (SSM-based)
- RWKV (RNN-attention hybrid)
- RetNet

**Strategy:**
- May require new TT-NN ops
- Closer collaboration with Tenstorrent team
- Higher difficulty tier (more impactful contribution!)

---

## Pro Tips from Successful Contributors

### 1. **Start Small**
- Take on a warmup or easy bounty first
- Get familiar with workflow before tackling hard bounties

### 2. **Communicate Early and Often**
- Post updates to issue thread every few days
- Ask questions in Tenstorrent Discord
- Request assignment extensions if needed

### 3. **Reuse, Don't Reinvent**
- Study similar models in the repo
- Copy patterns from proven implementations
- Reviewers LOVE code reuse

### 4. **Test Incrementally**
- Don't wait until the end to test
- Unit test every module
- Fix PCC issues immediately

### 5. **Profile Early**
- Measure performance from day 1
- Know your target throughput
- Identify bottlenecks early

### 6. **Document as You Go**
- Write README during development
- Capture performance numbers in real-time
- Future-you will thank present-you

### 7. **Break Up Large PRs**
- Follow MODEL_ADD.md recommendations
- Core code → Performance → Demo (3 PRs)
- Easier to review = faster merge

---

## Common Pitfalls to Avoid

### ❌ Don't Copy-Paste Entire Codebases
**Why it fails:**
- Reviewers reject duplicated code
- Hard to maintain divergent implementations
- Violates bounty requirements

**Do instead:**
- Reuse tt_transformers framework
- Add only model-specific modifications
- Leverage existing infrastructure

### ❌ Don't Skip Baseline Validation
**Why it fails:**
- TT implementation matches broken reference
- Waste time debugging TT when issue is in PyTorch reference

**Do instead:**
- Validate reference model on CPU first
- Generate reference logits
- Ensure accuracy before hardware port

### ❌ Don't Optimize Prematurely
**Why it fails:**
- Complex optimizations before correctness
- Hard to debug mixed correctness/performance issues

**Do instead:**
- Get it working first (even if slow)
- Measure performance
- Optimize based on profiling data

### ❌ Don't Ignore CI Failures
**Why it fails:**
- PRs won't merge with failing tests
- Indicates real bugs or incompatibilities

**Do instead:**
- Run CI pipeline locally first
- Fix failures before requesting review
- Keep PR rebased on latest main

---

## Resources

### Official Documentation
- [Model Bring-Up Guide](https://github.com/tenstorrent/tt-metal/blob/main/models/docs/model_bring_up.md)
- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/)
- [TT-Metalium Docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)
- [Bounty Program Terms](https://docs.tenstorrent.com/bounty_terms.html)

### Example PRs
- Phi-3: #22716, #27289 (rebased)
- See closed bounty issues for more examples

### Community
- [Tenstorrent Discord](https://discord.gg/tenstorrent)
- [GitHub Discussions](https://github.com/tenstorrent/tt-metal/discussions)

---

## Next Steps

Ready to make your first contribution? Try the hands-on example:

[🚀 Browse Open Bounties on GitHub](command:tenstorrent.browseOpenBounties)

[📋 Copy Bounty Workflow Checklist](command:tenstorrent.copyBountyChecklist)

---

## Summary

**You've learned:**
- ✅ How the Tenstorrent Bounty Program works
- ✅ The Phi-3 case study (successful Medium-tier contribution)
- ✅ 7-phase workflow: Setup → Validation → Component Bring-Up → Integration → Optimization → Testing → Submission
- ✅ How to reuse tt_transformers framework (key to approval!)
- ✅ Performance and accuracy requirements
- ✅ Common pitfalls and how to avoid them
- ✅ How lessons apply to other bounty types

**The workflow transfers to:**
- Other LLMs (Phi-4, Qwen, Mistral, CodeLlama)
- Vision models (ViT, CLIP, SAM)
- Diffusion models (SD variants, ControlNet)
- Novel architectures (with adaptations)

**Key principle:** Start simple, test incrementally, reuse code, communicate often.

---

**Ready to contribute?** 🎯

The Tenstorrent community is welcoming to newcomers. Start with a warmup task, learn the workflow, then scale up to more challenging contributions. Your work will run on cutting-edge AI hardware, become part of the open-source ecosystem, and help advance the field. The real reward is owning and being part of your own open future.
