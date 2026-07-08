# Build an LLM from Scratch, TT-Native — reference code

This directory holds the **verified, runnable reference code** for the
`lfs-00`…`lfs-05` lesson arc. The labs quote these files code-first, so the
prose and the code never drift: **this directory is the source of truth.**

> These are **lesson reference code, not shipped extension features.** They are
> not wired into `package.json` commands or the walkthrough runtime — a reader
> runs them by hand while following a lab.

## Contents

| File | What it is | Run it | Verified |
|---|---|---|---|
| `reference_gpt.py` | Pure-PyTorch nano GPT (tokenizer, embeddings, MHA, block, forward). No TT deps, CPU-runnable. The "understand" half of Labs 1-4. | `python reference_gpt.py --smoke` | ✅ CPU: logits `[2, 32, 96]`, 10.8M params, exit 0 |
| `train_nano_from_scratch.py` | Thin runner that imports the model + training primitives from the verified `nanogpt_primitives_example.py` and drives a short from-scratch training loop (cross-entropy, AdamW, N steps). Requires `ttml`. | see below | ✅ Blackhole p300c: loss 4.72 → 3.27 in 10 steps, exit 0 |
| `BUILD_TTML.md` | The verified `ttml`-from-source recipe (incl. the `std::bad_cast` ABI fix, env vars, board-reset note). Prerequisite for the training runner. | — | ✅ p300c, tt-metal v0.73 |
| `kernels/eltwise_add.py` | TT-Lang inception kernel: residual-stream elementwise add. Lab 2. | `python kernels/eltwise_add.py` | ✅ functional sim: max err 0.0 |
| `kernels/matmul.py` | TT-Lang inception kernel: tiled matmul `Y = A @ B`. Lab 4. | `python kernels/matmul.py` | ✅ functional sim: max err ~1.5e-5 |
| `kernels/attention.py` | TT-Lang inception kernel: single-head scaled-dot-product attention + softmax. Lab 3. | `python kernels/attention.py` | ⚠️ compiler/hardware path only (see below) |
| `kernels/rmsnorm.py` | TT-Lang inception kernel: RMSNorm. Lab 4. | `python kernels/rmsnorm.py` | ⚠️ compiler/hardware path only (see below) |
| `kernels/_ttlang_sim.py` | Helper that puts TT-Lang's functional simulator (`vendor/tt-lang/python/sim`) on `sys.path`. | (imported) | — |

## Running the kernels (functional simulator)

The kernels run against TT-Lang's **in-process functional simulator** — no
Tenstorrent device required. They locate `vendor/tt-lang/python/sim`
automatically by walking up to the repo root (or set `TTLANG_PYTHON`). Clone the
simulator if missing:

```bash
git clone https://github.com/tenstorrent/tt-lang.git vendor/tt-lang
python content/templates/llm-from-scratch/kernels/eltwise_add.py   # PASSED
python content/templates/llm-from-scratch/kernels/matmul.py        # PASSED
```

`import ttnn` must succeed in your environment (the sim uses it for tensor
conversion). Any recent tt-metal / TT-NN venv works.

### Sim-only vs. compiler/hardware-path kernels — the honest flag

The functional simulator runs **ahead of** the hardware compiler, and at the
pinned tt-lang commit (`a19aaa8`) its `ttl.math.broadcast` requires the source
block to have a genuine size-1 sub-tile dimension.

- **`eltwise_add.py`, `matmul.py`** — validate cleanly in the sim today.
- **`attention.py`, `rmsnorm.py`** — the softmax / normalization reductions
  broadcast a per-row scalar (a full 32×32 tile) back across the score tile,
  which this sim build rejects (`"dimension must have element size 1"`). These
  kernels are therefore **compiler / hardware-path only** here — consistent with
  the vendor `test_transformer_block.py`, which carries
  `TTLANG_HARDWARE_CI: skip-compiler` and runs on a real device. Each file runs
  its PyTorch reference (exit 0) and prints the honest status. Labs 3-4 must
  flag this ("simulator ahead of compiler") rather than claim a clean sim run.

## Running the training (Blackhole)

Build `ttml` first (see `BUILD_TTML.md`), then:

```bash
TT_METAL_HOME=/home/ttuser/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal \
TT_METAL_ARCH_NAME=blackhole TT_LOGGER_LEVEL=FATAL \
python content/templates/llm-from-scratch/train_nano_from_scratch.py \
    --max_steps 10 --batch_size 2 --data_path /path/to/shakespeare.txt
```

Use `TT_METAL_ARCH_NAME=wormhole_b0` on N-series. Needs a plain-text corpus
(`--data_path`) — it char-tokenizes any text file. If the board errors at device
open, `tt-smi -r` once and retry.

**Verified loss curve (p300c, tt-metal v0.73, 2026-07-08):**

```
Step 0  Loss 4.72 (compile step, 87 ms)
Step 1  Loss 3.75
Step 2  Loss 3.55
Step 3  Loss 3.39
Step 4  Loss 3.45
Step 5  Loss 3.36
Step 6  Loss 3.28
Step 7  Loss 3.34
Step 8  Loss 3.36
Step 9  Loss 3.27
```

## Nano baseline config

Both `reference_gpt.py` and `train_nano_from_scratch.py` use the same nano
baseline (from `tt-train/configs/model_configs/nanogpt.yaml`):

| Knob | Nano value | ~80M "hero" scaling |
|---|---|---|
| `n_embd` (embedding dim) | 384 | ↑ (e.g. 768) |
| `n_head` | 6 | ↑ (e.g. 12) |
| `n_layer` (blocks) | 6 | ↑ (e.g. 12) |
| `block_size` (seq len) | 256 | ↑ |
| `vocab_size` | 96 (char-level) | 50257 (GPT-2 BPE) |
| dropout | 0.2 | 0.2 |

At the nano config the model is ~10.8M parameters (`reference_gpt.py` and the
ttml model agree on the count). The ~80M "hero" number is the *same code* with
bigger knobs plus the DRAM/time math Lab 5 works through.

## Drift sources

These files are **adapted from** upstream references. Re-check them against the
sources before publishing (same spirit as `scripts/check-sim-lite-drift.py`):

| File | Adapted from | Commit |
|---|---|---|
| `kernels/eltwise_add.py` | `vendor/tt-lang/examples/eltwise_add.py` | `a19aaa8` |
| `kernels/matmul.py` | `vendor/tt-lang/examples/matmul.py` (bias term dropped — sim-unsupported broadcast) | `a19aaa8` |
| `kernels/attention.py` | `attention_kernel` in `vendor/tt-lang/examples/test_transformer_block.py` | `a19aaa8` |
| `kernels/rmsnorm.py` | RMSNorm block of `norm_qkv_kernel` / `norm_mlp_residual_kernel`, same file | `a19aaa8` |
| `train_nano_from_scratch.py` | imports & reuses `$TT_METAL_HOME/tt-train/sources/examples/nano_gpt/nanogpt_primitives_example.py` | tt-metal v0.73 |
| `reference_gpt.py` | independent pure-PyTorch mirror of the ttml NanoGPT / `nanogpt.yaml` config | — |

Check the current tt-lang commit with:

```bash
git -C vendor/tt-lang rev-parse --short HEAD
```
