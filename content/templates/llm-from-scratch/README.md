# Build an LLM from Scratch, TT-Native — reference code

This directory holds the **verified, runnable reference code** for the
`lfs-00`…`lfs-05` lesson arc. The labs quote these files code-first, so the
prose and the code never drift: **this directory is the source of truth.**

> These are **lesson reference code, not shipped extension features.** They are
> not wired into `package.json` commands or the walkthrough runtime — a reader
> runs them by hand while following a lab.

## The modern Llama-3 component set

The arc builds the **modern Llama-3 stack** (the same components
[Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM) champions), not the
older GPT-2 design. Every piece is expressed both in pure PyTorch (to
understand) and TT-native (ttnn / TT-Lang, to run):

| Component | What it replaces | Where it's built |
|---|---|---|
| **RoPE** (rotary position embeddings, θ=500000) | learned positional embedding table | `reference_gpt.py` (full RoPE) + `kernels/rope.py` (simplified) + `ttml.ops.rope` |
| **RMSNorm** (pre-norm) | LayerNorm | `reference_gpt.py` + `kernels/rmsnorm.py` + `ttml.ops.rmsnorm` |
| **GQA** (grouped-query attention, 6 heads / 3 KV groups) | vanilla multi-head attention | `reference_gpt.py` (KV-sharing) + `kernels/attention.py` (single-head) + `ttml.ops` |
| **SwiGLU** MLP (`down(silu(gate(x)) * up(x))`) | GELU expand-4x MLP | `reference_gpt.py` + `ttml.ops.swiglu` (**no from-scratch TT-Lang kernel** — see note) |

The hero training run is `ttml`'s `nanollama3` model (`model_type: llama`) —
verified on this Blackhole p300c (numbers below). GPT-2 is the historical
contrast the arc footnotes.

## Contents

| File | What it is | Run it | Verified |
|---|---|---|---|
| `tokenizer_bpe.py` | Minimal from-scratch byte-pair-encoding (BPE) tokenizer -- train merges, encode, decode. No external tokenizer libs. Frames the SentencePiece BPE 32K that production/Mini-LLM uses; the nano hero run itself uses `CharTokenizer`. | `python tokenizer_bpe.py` | ✅ CPU: vocab 306 (256 bytes + 50 merges), round-trip on seen + unseen text, exit 0 |
| `reference_gpt.py` | Pure-PyTorch nano **Llama-3** (char tokenizer, RoPE, RMSNorm, GQA, SwiGLU block, forward). No TT deps, CPU-runnable. Mirrors ttml `nanollama3`. The "understand" half of Labs 1-4. | `python reference_gpt.py --smoke` | ✅ CPU: logits `[2, 32, 96]`, 9.81M params, exit 0 |
| `train_nano_from_scratch.py` | Thin launcher that drives the verified upstream `train_nanogpt.py --config training_shakespeare_nanollama3_char.yaml` (the Llama path) with the right env vars; passes through `--max_steps` / `--data_path`. Requires `ttml`. | see below | ✅ Blackhole p300c: loss 4.69 → 3.23 in 20 steps, ~65 ms/step, exit 0 |
| `BUILD_TTML.md` | The verified `ttml`-from-source recipe (incl. the `std::bad_cast` ABI fix, env vars, board-reset note). Prerequisite for the training launcher. | — | ✅ p300c, tt-metal v0.73 |
| `kernels/eltwise_add.py` | TT-Lang inception kernel: residual-stream elementwise add. Lab 2. | `python kernels/eltwise_add.py` | ✅ functional sim: max err 0.0 |
| `kernels/rope.py` | TT-Lang inception kernel: **simplified** rotary embedding on Q/K (cos-only multiply). Lab 2. | `python kernels/rope.py` | ✅ functional sim: max err ~5e-4 |
| `kernels/matmul.py` | TT-Lang inception kernel: tiled matmul `Y = A @ B`. Lab 4. | `python kernels/matmul.py` | ✅ functional sim: max err ~1.5e-5 |
| `kernels/attention.py` | TT-Lang inception kernel: **single-head** scaled-dot-product attention + softmax. Lab 3. | `python kernels/attention.py` | ⚠️ compiler/hardware path only (see below) |
| `kernels/rmsnorm.py` | TT-Lang inception kernel: RMSNorm. Lab 4. | `python kernels/rmsnorm.py` | ⚠️ compiler/hardware path only (see below) |
| `kernels/_ttlang_sim.py` | Helper that puts TT-Lang's functional simulator (`vendor/tt-lang/python/sim`) on `sys.path`. | (imported) | — |

### Notes on the kernels

- **`kernels/rope.py` is a *simplified* rotary** — it multiplies Q and K by
  `cos` only (no dimension split, no `rotate_half`, no `sin` term), faithful to
  the vendor `rotary_qk_kernel`. Full RoPE (`x*cos + rotate_half(x)*sin`) lives
  in `reference_gpt.py` (`precompute_rope_cos_sin` / `apply_rope`) and, on
  hardware, in `ttml.ops.rope`. Because RoPE is elementwise, the kernel runs in
  the functional sim (same class as `eltwise_add`).
- **`kernels/attention.py` is single-head.** GQA (the modern win) is taught as
  the **KV-sharing pattern layered on top**: compute K/V once per group and
  share them across the query heads in that group — see the `repeat_interleave`
  step in `reference_gpt.py`'s `GroupedQueryAttention`. The single-head kernel
  shows the Q·Kᵀ→softmax→·V core; GQA is the bookkeeping around it.
- **SwiGLU has no from-scratch TT-Lang kernel** (out of scope). It is taught via
  the PyTorch reference in `reference_gpt.py` plus the drop-in device op
  `ttml.ops.swiglu` (which fuses `silu(gate(x)) * up(x)` → `down`).

## Running the kernels (functional simulator)

The kernels run against TT-Lang's **in-process functional simulator** — no
Tenstorrent device required. They locate `vendor/tt-lang/python/sim`
automatically by walking up to the repo root (or set `TTLANG_PYTHON`). Clone the
simulator if missing:

```bash
git clone https://github.com/tenstorrent/tt-lang.git vendor/tt-lang
python content/templates/llm-from-scratch/kernels/eltwise_add.py   # PASSED
python content/templates/llm-from-scratch/kernels/rope.py          # PASSED
python content/templates/llm-from-scratch/kernels/matmul.py        # PASSED
```

`import ttnn` must succeed in your environment (the sim uses it for tensor
conversion). Any recent tt-metal / TT-NN venv works.

### Sim-runnable vs. compiler/hardware-path kernels — the honest flag

The functional simulator runs **ahead of** the hardware compiler, and at the
pinned tt-lang commit (`a19aaa8`) its `ttl.math.broadcast` requires the source
block to have a genuine size-1 sub-tile dimension.

- **`eltwise_add.py`, `rope.py`, `matmul.py`** — validate cleanly in the sim
  today (all elementwise/matmul, no reduce→broadcast).
- **`attention.py`, `rmsnorm.py`** — the softmax / normalization reductions
  broadcast a per-row scalar (a full 32×32 tile) back across the score tile,
  which this sim build rejects (`"dimension must have element size 1"`). These
  kernels are therefore **compiler / hardware-path only** here — consistent with
  the vendor `test_transformer_block.py`, which carries
  `TTLANG_HARDWARE_CI: skip-compiler` and runs on a real device. Each file runs
  its PyTorch reference (exit 0) and prints the honest status. Labs 3-4 must
  flag this ("simulator ahead of compiler") rather than claim a clean sim run.

## Running the training (Blackhole)

The training launcher drives the verified upstream Llama trainer. Build `ttml`
first (see `BUILD_TTML.md`), then:

```bash
python content/templates/llm-from-scratch/train_nano_from_scratch.py \
    --max_steps 20 \
    --data_path /home/ttuser/tt-metal/tt-train/data/shakespeare.txt
```

The launcher sets `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`,
`TT_METAL_ARCH_NAME=blackhole`, and `TT_LOGGER_LEVEL=FATAL` for you (honouring
anything you already exported). Pass `--arch wormhole_b0` on N-series. Use
`--dry_run` to see the exact command without launching. If the board errors at
device open, `tt-smi -r` once and retry.

**Why a launcher, not a primitives script:** unlike GPT-2 (which has a
single-file `nanogpt_primitives_example.py`), there is **no** single-file Llama
"primitives" example. The canonical from-scratch Llama path is
`train_nanogpt.py` + the `training_shakespeare_nanollama3_char.yaml` config
driving `ttml.models.llama`.

**Verified hero run (p300c, tt-metal v0.73, 2026-07-08):**
`train_nanogpt.py --config training_shakespeare_nanollama3_char.yaml
--max_steps 20` — **loss 4.69 → 3.23 over 20 steps, ~65 ms/step, 16.5 TFLOPS,
MFU ~11%, exit 0.**

## Nano baseline config (nanollama3)

Both `reference_gpt.py` and the training launcher use the same nano baseline
(from `tt-train/configs/model_configs/nanollama3_char.yaml`):

| Knob | Nano value | ~80M Mini-LLM "hero" scaling |
|---|---|---|
| `embedding_dim` (`n_embd`) | 384 | ↑ (e.g. 768) |
| `num_heads` (`n_head`) | 6 | ↑ (e.g. 12) |
| `num_groups` (`n_kv_groups`, GQA) | 3 | ↑ (keep < num_heads) |
| `num_blocks` (`n_layer`) | 6 | ↑ (e.g. 12) |
| `max_sequence_length` (`block_size`) | 256 | ↑ |
| RoPE `theta` | 500000 | 500000 |
| SwiGLU intermediate | 1024 (from 8/3·384, rounded to 256) | scales with `n_embd` |
| `vocab_size` | 96 (char-level) | 32000 (SentencePiece BPE, Mini-LLM) |
| dropout | 0.0 | 0.0 |

At the nano config the model is **~9.81M parameters** (`reference_gpt.py`; the
ttml `nanollama3` model is in the same ballpark — the two differ only in fusion
details like ttml's combined `kv_linear`, not in the math). The ~80M "hero"
number is the *same code* with bigger knobs — Mini-LLM reaches it with a
SentencePiece 32K vocab and trains on 361M tokens (~5h on one A100) to a final
loss ~3.25. Lab 5 works through the params/DRAM/time math.

## Drift sources

These files are **adapted from** upstream references. Re-check them against the
sources before publishing (same spirit as `scripts/check-sim-lite-drift.py`):

| File | Adapted from | Commit |
|---|---|---|
| `kernels/eltwise_add.py` | `vendor/tt-lang/examples/eltwise_add.py` | `a19aaa8` |
| `kernels/rope.py` | `rotary_qk_kernel` in `vendor/tt-lang/examples/test_transformer_block.py` (simplified: cos-only multiply, no sin/rotate_half) | `a19aaa8` |
| `kernels/matmul.py` | `vendor/tt-lang/examples/matmul.py` (bias term dropped — sim-unsupported broadcast) | `a19aaa8` |
| `kernels/attention.py` | `attention_kernel` in `vendor/tt-lang/examples/test_transformer_block.py` | `a19aaa8` |
| `kernels/rmsnorm.py` | RMSNorm block of `norm_qkv_kernel` / `norm_mlp_residual_kernel`, same file | `a19aaa8` |
| `train_nano_from_scratch.py` | launches `$TT_METAL_HOME/tt-train/sources/examples/nano_gpt/train_nanogpt.py --config training_shakespeare_nanollama3_char.yaml` (`model_type: llama`) | tt-metal v0.73 |
| `reference_gpt.py` | independent pure-PyTorch mirror of `ttml.models.llama` / `nanollama3_char.yaml` (RoPE + RMSNorm + GQA + SwiGLU) | — |
| `tokenizer_bpe.py` | independent from-scratch implementation of Sennrich et al. 2016 byte-pair encoding (same core loop as Karpathy's `minbpe`, not copied from it) | — |

Check the current tt-lang commit with:

```bash
git -C vendor/tt-lang rev-parse --short HEAD
```
