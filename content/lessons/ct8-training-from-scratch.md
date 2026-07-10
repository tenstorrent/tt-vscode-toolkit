---
id: ct8-training-from-scratch
title: Training from Scratch
description: >-
  Configure, launch, monitor, checkpoint, and scale a from-scratch training job with ttml's train_nanogpt.py — the modern nanollama3_char config (RoPE/RMSNorm/SwiGLU/GQA), a real 3000-step loss curve on Blackhole p300c, and an honest look at what driving loss to 0.18 on a tiny corpus actually buys you.
category: custom-training
tags:
  - from-scratch
  - ttml
  - train-nanogpt
  - nanollama3
  - checkpoints
  - loss-curves
  - overfitting
  - scaling
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
status: validated
validatedOn:
  - p300c
estimatedMinutes: 30
minTTMetalVersion: v0.67.0
---

# Training from Scratch

Every other lesson in this track hands `train_nanogpt.py` a checkpoint to load. This one doesn't hand it anything — you launch a job that starts from **random weights** and watch it become a model, entirely from the numbers a loss curve prints to your terminal.

That's the whole job: pick a config, launch it, watch the loss, checkpoint along the way, generate from a checkpoint, and know how to scale the next run up. This lesson does **not** ask you to hand-write a transformer's internals — [Model Architecture Basics](command:tenstorrent.showLesson?["ct7-architecture-basics"]) already toured what `num_heads`, `embedding_dim`, `num_blocks`, and `theta` mean conceptually. If you want to write RoPE, grouped-query attention, and a SwiGLU block yourself instead of pointing `ttml` at a YAML file, that's a different track entirely — see **Build It Yourself** near the end.

Every number below — the loss curve, the wall-clock time, the generated text — is copied verbatim from a real training run against this extension's verified `ttml` build, on a Blackhole<sup>®</sup> p300c. Nothing here is projected, and nothing here overclaims what the output actually reads like.

## What You'll Learn

- Launching `train_nanogpt.py` with `nanollama3_char` — the modern RoPE/RMSNorm/SwiGLU/GQA config this track features for from-scratch jobs
- Reading a real loss curve from random initialization down to `0.18`, with checkpoints landing on disk along the way
- Generating from a checkpoint and reading the output **honestly** — structure, not coherence
- The overfitting lesson this run demonstrates: a much lower loss did **not** buy more readable text than a smaller, less-trained run already produced
- Scaling a job three ways — more steps, a bigger model config, more data — and the `mesh_shape` boundary where a single chip stops being enough

**Time:** 25-30 minutes (5-10 min hands-on, ~3.5 min hardware run) | **Prerequisites:** [Model Architecture Basics](command:tenstorrent.showLesson?["ct7-architecture-basics"]) and [Configuration Patterns](command:tenstorrent.showLesson?["ct3-configuration-patterns"])

---

## Where This Fits in the Track

```mermaid
graph LR
    A[Understand] --> B[Datasets]
    B --> C[Configuration]
    C --> D[Fine-tuning]
    D --> E[Multi-Device]
    E --> F[Experiment Tracking]
    F -.-> G[Architecture Basics]
    G -.-> H[From Scratch]

    style H fill:#1B8EB1,stroke:#092221,stroke-width:3px
```

---

## Set Up the Job

Same `ttml` build every other lesson in this track uses. If you haven't built it, [Fine-tuning Basics](command:tenstorrent.showLesson?["ct4-finetuning-basics"]) covers the **Install tt-train** command and the `std::bad_cast` fix in full — this lesson assumes that's done.

Set your environment honoring any value you've already exported, rather than overwriting it:

```bash
export TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tt-metal}"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
: "${TT_METAL_ARCH_NAME:=wormhole_b0}"   # set to blackhole for p100 / p150 / p300c
export TT_METAL_ARCH_NAME
export TT_LOGGER_LEVEL=FATAL
cd ~/tt-metal/tt-train/sources/examples/nano_gpt
```

You'll need a Shakespeare corpus — either the one you built in [Dataset Fundamentals](command:tenstorrent.showLesson?["ct2-dataset-fundamentals"]), or the copy `tt-metal` ships at `tt-train/data/shakespeare.txt`, which is what the run below actually used.

---

## Launch: the `nanollama3_char` Config

`tt-train/configs/model_configs/` ships two architecture families for this size of job: `nanogpt*` (GPT-2-style — LayerNorm, learned position embeddings, plain multi-head attention) and `nanollama3*` (Llama-3-style — RoPE, RMSNorm, SwiGLU, grouped-query attention). [Fine-tuning Basics](command:tenstorrent.showLesson?["ct4-finetuning-basics"]) ran the GPT-2-style config. This lesson features the modern one — `nanollama3_char` — because it's the exact architecture the from-scratch arc builds by hand, component by component (see **Build It Yourself** below).

The shapes that matter, quoted from the real files in `tt-train/configs/`:

| Setting | Value | Source |
|---|---|---|
| Architecture | `model_type: llama` — RoPE (`theta=500000`), RMSNorm, SwiGLU, grouped-query attention | `model_configs/nanollama3_char.yaml` |
| Heads / KV groups | 6 heads, 3 groups (2 query heads share each KV head) | same |
| Embedding dim / blocks | 384 / 6 | same |
| Context length | 256 characters | same |
| Parameters | **9,810,816** (~9.8M) | printed at model creation |
| Tokenizer | Character-level, auto-detected — 68 unique characters, rounded up to a tile-friendly **96** | printed at data load |
| Batch size | 64 | `training_configs/training_shakespeare_nanollama3_char.yaml` |
| Optimizer | AdamW, `lr: 0.0003`, `weight_decay: 0.01` | same |
| Checkpoint interval | every 500 steps (`model_save_interval: 500`) | same |
| Device mesh | `[1, 1]` (default, single chip) — p300c and p100 count as one chip here, exactly like n150 | no `device_config:` block needed |

Launch it. The config's own `max_steps: 5000` is overridden on the command line to run 3000:

```bash
python train_nanogpt.py \
  --config training_shakespeare_nanollama3_char.yaml \
  --data_path ~/tt-metal/tt-train/data/shakespeare.txt \
  --max_steps 3000 \
  --fresh \
  --model_save_path ~/tt-metal/tt-train/checkpoints/ct8_nanollama3
```

**Before committing to the full 3,000-step run:** sanity-check your setup with a much shorter smoke test — just swap `--max_steps 3000` for `--max_steps 20`. On this hardware that finishes in about 14 seconds, plenty to confirm the config loads, the data path resolves, and the device initializes. It still pays the one-time kernel-compile cost on step 1, same as the full run — a smoke test skips steps, not the compile tax.

`--fresh` matters: it says "ignore any existing checkpoint at this path, start from random initialization." That's the entire meaning of "from scratch" — everything else in this command is the same job-launching mechanic [Fine-tuning Basics](command:tenstorrent.showLesson?["ct4-finetuning-basics"]) already used.

---

## Watch It Converge — The Real Curve

This ran on this extension's Blackhole p300c, against tt-metal v0.73. Total wall clock for 3000 steps: **200.74 seconds (~3.3 minutes)**. Steady state: **~65 ms/step**, roughly **16.5 TFLOPS**, **~11% model FLOPS utilization (MFU)** — against a mesh peak of 148.5 TFLOPS (bf16, 1 device).

| Step | Loss | Checkpoint written |
|---|---|---|
| 1 | 4.6875 | — |
| 500 | 1.3516 | `ct8_nanollama3_step_500.pkl` |
| 1000 | 1.0938 | `ct8_nanollama3_step_1000.pkl` |
| 1500 | 0.8164 | `ct8_nanollama3_step_1500.pkl` |
| 2000 | 0.5156 | `ct8_nanollama3_step_2000.pkl` |
| 2500 | 0.2891 | `ct8_nanollama3_step_2500.pkl` |
| 3000 (final) | 0.1836 | `ct8_nanollama3_final.pkl` |

Loss `4.6875` at step 1 sits close to `ln(96)` ≈ 4.56 — the entropy of guessing uniformly among 96 possible next characters. That's the honest random baseline. By step 3000 that error is down to `0.18`, a much steeper drop than [Fine-tuning Basics](command:tenstorrent.showLesson?["ct4-finetuning-basics"])'s GPT-2-style run saw over the same 3000 steps (loss `1.406`, on the same corpus and step budget). `model_save_interval: 500` is why a checkpoint lands every 500 steps automatically — the table above is what actually appeared on disk, no extra flag required.

---

## Generate — and Read It Honestly

Load the final checkpoint and generate, using the same `--prompt` / `--model_path` flags every config accepts:

```bash
python train_nanogpt.py \
  --config training_shakespeare_nanollama3_char.yaml \
  --prompt "ROMEO:" \
  --model_path ~/tt-metal/tt-train/checkpoints/ct8_nanollama3_final.pkl \
  --max_new_tokens 300 --temperature 0.7 --top_k 50
```

**Actual output, verbatim, from the checkpoint at step 3000 (loss 0.18):**

```
etwaiynwiyounismanot ather bucoution.

LAGENIAYO:
Ahe imabaplart wellong there thou in priscian the racom to the stiffot will and years son,
There is not there in the mother we should sun
yet thou must be that duke of him so submiss'd
From the cause of thy bestray'd the death,
Must I that had body t
```

Read this for what it is. There's real **structure**: an ALL-CAPS speaker name (`LAGENIAYO:`) followed by a colon, line breaks, dialogue layout, an apostrophe used correctly (`submiss'd`, `bestray'd`). There's a genuine **mix of real and invented words** — "the," "there," "in," "we," "should," "sun," "thou," "must," "that," "from," "cause," "death" are real; "etwaiynwiyounismanot," "priscian," "racom," "stiffot," "bestray'd" are not. **This is not coherent Shakespeare, and it is not correct grammar.** It's the same class of output [Fine-tuning Basics](command:tenstorrent.showLesson?["ct4-finetuning-basics"])'s GPT-2-style run produced at loss 1.406 — structure and a scattering of real words, no more.

---

## The Overfitting Lesson

Here's the part worth sitting with: this run drove loss to **0.18** — nearly eight times lower than [Fine-tuning Basics](command:tenstorrent.showLesson?["ct4-finetuning-basics"])'s `1.406`. If loss were the whole story, this output should read dramatically more coherent. It doesn't. Both runs land in the same tier: recognizable structure, a handful of real words, mostly invented syllables.

That gap between "loss went way down" and "text didn't get more readable" **is** the lesson. `tt-train/data/shakespeare.txt` is about one megabyte of text. A 9.8M-parameter model has more than enough capacity to start memorizing that corpus's exact character sequences well before it has enough exposure to learn general English structure from them. Driving train loss to 0.18 on a dataset this small is **overfitting**, not mastery — the model is increasingly good at predicting *this specific text*, not increasingly good at *language*. Low loss on a tiny corpus is not a proxy for coherent output, and this run is the concrete evidence: a much lower loss bought no visible improvement in readability.

Real coherence needs scale, not just more steps against the same small file. [Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"]) — the from-scratch arc lab that builds this exact `nanollama3` architecture by hand — makes the same comparison against [Mini-LLM](https://github.com/Ashx098/Mini-LLM), the project this whole from-scratch design follows: **~80M parameters, 361M training tokens, ~5 hours on a single A100**, to get language that actually reads as language. Nine million parameters and one megabyte of characters, however low you push the loss, isn't that project — it's a controlled demonstration that the training mechanism works.

---

## Scaling the Job

Three independent knobs, each with a real config to point at:

**More steps.** The featured config's own default is `max_steps: 5000`, not the 3000 this lesson ran — try it, but expect the same overfitting ceiling above, not qualitatively better prose, on this same corpus.

**A bigger model.** `tt-train/configs/model_configs/` ships larger `llama`-family configs on the same architecture family — `nanollama3.yaml` (same 6-head/6-block shape, but a real 32,000-token BPE vocabulary instead of characters) and `llama3_gpt2s_size.yaml` (12 heads, 12 blocks, `embedding_dim: 768`, GPT-2-small-sized). Point `--config` at a training config referencing one of these via its `model_config:` field — every `transformer_config:` field maps to the concepts [Model Architecture Basics](command:tenstorrent.showLesson?["ct7-architecture-basics"]) covers, and to the DRAM math [The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"]) works through for scaling toward Mini-LLM's ~80M-parameter target.

**More data.** Swap `--data_path` for a larger plain-text corpus — `train_nanogpt.py` takes any text file, not just Shakespeare. [Dataset Fundamentals](command:tenstorrent.showLesson?["ct2-dataset-fundamentals"]) covers building one.

**The `mesh_shape` boundary.** Every config above still runs on a single chip (`mesh_shape: [1, 1]`, the default) — p300c, p100, or n150. `tt-train` does ship real multi-chip examples (`training_shakespeare_nanogpt_ddp_n300.yaml` sets `enable_ddp: true` and `mesh_shape: [1, 2]` for data-parallel training across two chips), but this lesson doesn't run one: **multi-device training is deferred to [Multi-Device Training](command:tenstorrent.showLesson?["ct5-multi-device-training"])**, which is itself an honest, source-grounded but not-yet-hardware-verified lesson — a single p300c has no second chip to split a batch across, and TT-QuietBox<sup>®</sup> 2's four p300c chips are independent, not a mesh. You won't need a mesh at all until parameter count grows into the billions; everything in the nano-to-~80M range this section discusses fits comfortably in one chip's DRAM.

---

## Build It Yourself: The From-Scratch Arc

Everything above configures and launches `ttml` — you never touch a matrix multiply. If you want to **write** this architecture instead of configuring it, that's a different track, the "Build an LLM from Scratch" arc, starting at [Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]):

- [Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"]) — the token embedding table and RoPE's rotation math, hand-written.
- [Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"]) — grouped-query attention with RoPE'd Q/K, derived fully and authored as a TT-Lang kernel.
- [The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"]) — the SwiGLU MLP, RMSNorm, and residuals assembled into a full, runnable model.
- [Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"]) — the hero lab: the **same** 9,810,816-parameter `nanollama3` architecture this lesson just ran, but with a training loop (cross-entropy, backprop, AdamW) you write yourself instead of calling into `ttml`, verified end-to-end on Blackhole.

Same architecture, same config shape, two altitudes: point `ttml` at a YAML file (this lesson), or write every gradient step yourself (that arc).

---

## Troubleshooting

### `ImportError: No module named 'ttml'` or `std::bad_cast` on `import ttml`

Covered in [Fine-tuning Basics](command:tenstorrent.showLesson?["ct4-finetuning-basics"]) — rebuild `_ttnn.so` after enabling tt-train, don't do a partial `--target _ttml` build.

### `RuntimeError: Device out of memory`

Reduce `--batch_size` (default 64 for this config) — override on the command line, e.g. `--batch_size 32`.

### Loss stays near `ln(vocab_size)` and never drops

Check the data actually loaded — `train_nanogpt.py` prints dataset size and vocabulary size at startup; if either is missing or zero, `--data_path` is pointing at the wrong file.

### No checkpoint file after training

Three real causes, in order of how often they bite:

- **Parent dir missing** — `train_nanogpt.py` doesn't create it. `mkdir -p $(dirname <model_save_path>)` first.
- **`model_save_interval` too large** — if a run dies before its first save, you get nothing. Keep it small (500–1000). A run killed 42 steps before an interval-1500 save leaves zero checkpoints — learned the hard way.
- **DDP save throws** — under multi-chip DDP the stock saver hits `Can't get a single buffer from host storage distributed over mesh`. The weights are replicated across chips; pull them through `ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)` and keep the first replica.

### Multi-chip DDP dies at `Fabric Router Sync: Timeout`

On a TT-QuietBox 2 a 2- or 4-chip run can time out at fabric-router sync during mesh open — and it survives both a full reboot and `tt-smi -r`, so it looks like dead hardware. It isn't: ttml ships default mesh-graph descriptors only for T3000/Galaxy, so on a 2/4-device Blackhole mesh you must set `TT_MESH_GRAPH_DESC_PATH` yourself. The full fix (with the `[1,4]` ring descriptor) is in [Multi-Device Training](command:tenstorrent.showLesson?["ct5-multi-device-training"]).

### `argument --resume: expected one argument` (auto-resume is broken)

Any run *without* `--fresh` triggers auto-resume, which currently injects an empty `--resume` and dies in argparse. Run with `--fresh` and checkpoint often instead of relying on resume.

### Device open hangs or times out on p300c / TT-QuietBox 2

First check: is another job holding the device? Only one process can own the mesh, so a live training run makes any second job (like inference) hang at device open — that's contention, not a fault. If nothing else is running, `tt-smi -r` to reset the board and retry; a hard-killed `ttml` process can wedge the device, so always let it close cleanly.

### Generated text loops or turns to word-salad

Same root cause both ways: an **undertrained** model plus a bare decoder. The stock `sample_greedy` has no repetition penalty, so a low-maturity model loops ("and Ben and Ben"); bolting on a *strong* penalty just turns the loop into word-salad. The real fix is training maturity (loss well under 1.0 for TinyStories-scale text) plus a *gentle* repetition penalty — not the penalty alone.

---

## Key Takeaways

- **A from-scratch job is a config choice, not a hand-build.** `--fresh` plus a `model_type: llama` config (`nanollama3_char`) is the entire mechanic — the components inside it are covered conceptually in [Model Architecture Basics](command:tenstorrent.showLesson?["ct7-architecture-basics"]) and built by hand in the from-scratch arc.
- Real numbers from this p300c: loss `4.6875 → 0.1836` over 3000 steps, ~65 ms/step, ~3.3 minutes total, checkpointed every 500 steps.
- **Low loss is not a proxy for coherent output.** This run's loss was ~8x lower than a comparable GPT-2-style run, and the generated text landed in the same tier: Shakespeare-shaped structure, a mix of real and invented words, not readable prose. On a ~1 MB corpus, driving loss this low is overfitting, not mastery.
- Real coherence is a **scale** problem — the from-scratch arc's Mini-LLM comparison (~80M params, 361M tokens, ~5 hours on an A100) is the order of magnitude that actually buys readable output, not more steps against a tiny file.
- Scaling a job is three independent knobs — steps, model config, data — and a `mesh_shape` boundary you won't hit below billions of parameters on this hardware.

---

## What's Next

**Next: [Experiment Tracking](command:tenstorrent.showLesson?["ct6-experiment-tracking"])** — capture runs like the one above to a file (or Weights & Biases) instead of watching numbers scroll past in a terminal, and compare hyperparameter variations properly.

**Or build every component by hand:** start the from-scratch arc at [Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]) and work through the tokenizer, embeddings, attention kernel, and training loop yourself — culminating in [Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"]), which trains this exact architecture with code you wrote.

---

## Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original transformer paper
- [LLaMA](https://arxiv.org/abs/2302.13971) — the paper behind RoPE + GQA + SwiGLU + RMSNorm as a combined recipe
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — the Chinchilla paper, on why scale (not just steps) governs quality
- [Mini-LLM (Ashx098)](https://github.com/Ashx098/Mini-LLM) — the ~80M-parameter reference project this track's architecture follows, and the scale comparison used above
- [`train_nanogpt.py`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/examples/nano_gpt/train_nanogpt.py) — the trainer this lesson runs, in the `tt-metal` GitHub repository
- [tt-train source](https://github.com/tenstorrent/tt-metal/tree/main/tt-train) — the framework behind `ttml`
