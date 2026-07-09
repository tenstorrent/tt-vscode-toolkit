---
id: lfs-05-train-and-run
title: "Train It & Run for Real"
description: >-
  Write the training loop from scratch — cross-entropy, AdamW, backprop — then
  train your model for real on Blackhole with ttml. Verified on p300c.
category: llm-from-scratch
tags: [training, adamw, ttml, blackhole, nanogpt]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
validatedOn:
  - p300c
estimatedMinutes: 45
minTTMetalVersion: v0.73.1
---

# Train It & Run for Real

> **Following along?** The code in this lab lives in your `~/tt-scratchpad/llm-from-scratch/` workspace. If you haven't created it yet, use the **Create the LLM-from-Scratch Project** button in [Lab 0](command:tenstorrent.showLesson?["lfs-00-intro"]).


This is it. The hero lab.

Across four labs you built a modern Llama-3-shape model from nothing: a
tokenizer, a residual stream, attention with RoPE and grouped-query sharing, a
full SwiGLU/RMSNorm block — stacked six deep into a **9,810,816-parameter**
`NanoLlama`. You ran its forward pass and watched the initial loss land near
`ln(vocab_size)`. That number is the honest tell of a model that is correctly
wired but knows nothing: it has no idea which character comes next, so its
predictions are close to uniform.

Now you make it learn. Not on a GPU, not in a simulator — on real Tenstorrent
Blackhole<sup>®</sup> silicon, watching the loss drop step by step.

[The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])
finished the model. This lab closes the last gap. You'll:

- **Write the training loop from scratch** — cross-entropy loss,
  backpropagation, the AdamW optimizer.
- **Understand why `ttnn` alone can't run it**, and what can.
- **Build the one missing piece (`ttml`) from source.**
- **Then run it for real** — the exact model this arc built, training on a
  Blackhole chip, loss dropping step by step, exit 0, verified.

Everything from
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]) onward has
been building toward this one run.

**You are here:**

```mermaid
graph LR
    A["Tokenizer & Data"] --> B["Embeddings & RoPE"] --> C["Attention (GQA)"] --> D["Block & Model"] --> E["Train on Blackhole"]

    style E fill:#1B8EB1,stroke:#333,stroke-width:3px
```

## Coming from CUDA: where's `.backward()`?

On CUDA, `loss.backward()` is one line because PyTorch's autograd engine
silently built a computation graph behind every tensor operation and knows
how to walk it backward, accumulating gradients into `.grad` on every
parameter along the way. `ttnn` has no equivalent of that line. Every
`ttnn.matmul`, `ttnn.rms_norm`, and hand-authored TT-Lang kernel you've read
in this arc is a **forward-only** op: it computes a result and returns a
tensor, and nothing about it records how to differentiate that computation.
That's not an oversight — `ttnn` is an inference and forward-compute library
first, and forward-only is exactly what every lab before this one needed.
Training needs something else: a library that wraps forward ops with the
matching backward pass and an optimizer loop on top. On Tenstorrent hardware,
that library is **`ttml`** (the Python name for `tt-train`).

## The training loop, from scratch

Strip away the housekeeping and every training loop you'll ever write —
PyTorch, `ttml`, doesn't matter — is four steps repeated once per batch:

```
forward:   logits = model(input_ids)
loss:      L = cross_entropy(logits, target_ids)
backward:  grad = ∂L/∂θ for every parameter θ  (backprop)
update:    θ ← θ − optimizer_step(grad)         (AdamW)
```

**Cross-entropy loss** is the same quantity the `smoke()` check in
[The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])
already computed: `F.cross_entropy(logits.view(-1, vocab), targets.view(-1))`.
A freshly-initialized model's loss sits near `ln(vocab_size)` because its output
distribution is close to uniform over the vocabulary, and cross-entropy of a
uniform distribution over `V` classes is exactly `ln(V)`. Training's whole job
is to push that number down — making the model assign more probability mass to
the actual next character.

**Backpropagation** turns "the loss is too high" into "here is how much to
nudge every one of the model's 9.81M parameters." It's the chain rule, applied
mechanically from the loss backward through every matmul, RMSNorm, and SwiGLU
gate you read across
[Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
through **The Transformer Block & the Model**, in reverse order.

On CUDA, autograd builds this chain for you automatically. On Tenstorrent
hardware, `ttml`'s ops are the pieces that know how to run backward as well as
forward. `ttml.ops.cross_entropy_loss` isn't just a forward computation of the
loss value — it's an op with a matching backward that seeds the gradient chain.
And every op it calls into underneath (`ttml.ops.rmsnorm`, `ttml.ops.rope`,
`ttml.ops.swiglu`, the attention and linear ops
[Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])
and **The Transformer Block & the Model** named) carries its own backward the
same way.

**AdamW** is the optimizer that turns a raw gradient into an actual parameter
update — momentum-smoothed, per-parameter-scaled, with decoupled weight
decay. `ttml.optimizers.AdamW` is the on-device implementation of exactly
that algorithm, stepping every parameter tensor in place after each backward
pass.

**There is no single-file "Llama primitives" example for this stack.** That's
worth being honest about, because the GPT-2-era cousin of
[Tokenizer & Data](command:tenstorrent.showLesson?["lfs-01-tokenizer"]) through
**The Transformer Block & the Model** *does* have one:
`nanogpt_primitives_example.py` hand-assembles a GPT-2-shape model directly from
`ttml.ops` calls, forward and backward, inside one readable script. The modern
Llama-3 stack this arc teaches — RoPE, RMSNorm, GQA, SwiGLU — doesn't have that
single-file teaching example yet.

What it has instead is a canonical, supported training path:

- **`ttml.models.llama`** — the module that assembles the same architecture you
  read across **Embeddings & the Residual Stream** through **The Transformer
  Block & the Model**, expressed against `ttml`'s autograd-aware ops instead of
  PyTorch's.
- **`train_nanogpt.py`** — the upstream trainer script that drives it.
- **A YAML config** that selects `model_type: llama`.

This lab's runner, `~/tt-scratchpad/llm-from-scratch/train_nano_from_scratch.py`,
is a thin, documented launcher around exactly that command. It doesn't reinvent
the model or the loop — it drives the verified one with the right environment
variables set. Read the file: it's short, and every environment variable it
sets is explained inline and again in the build section below.

## Build `ttml`

`ttml` is **source-only** — there is no pip wheel or `.deb` for it, on any
platform, as of this writing. If you don't already have a working tt-metal
source-and-build tree, stop here and do the
[Build TT-Metalium from Source](command:tenstorrent.showLesson?["build-tt-metal"])
lesson first — everything below assumes `install_dependencies.sh` and
`build_metal.sh` already work on your machine.

**Time budget:** with a warm ccache, the whole recipe below — configure,
build `_ttml`, rebuild `_ttnn.so` — takes about **5 minutes** on this
hardware. A first-ever tt-metal build from cold (no ccache) is much longer;
budget accordingly.

**This matters especially if you're on a TT-QuietBox<sup>®</sup> 2.** A
pre-configured TT-QuietBox 2 image ships TT-NN<sup>™</sup> and vLLM
pre-installed, but it does **not** include the `~/tt-metal` source tree —
there's nothing to build `ttml` against until you clone and build tt-metal
yourself. Don't assume `~/tt-metal` exists; if it's missing, the
[Build TT-Metalium from Source](command:tenstorrent.showLesson?["build-tt-metal"]) is exactly the gap-filler.

The full, verified recipe lives in
[`content/templates/llm-from-scratch/BUILD_TTML.md`](https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/content/templates/llm-from-scratch/BUILD_TTML.md)
— read it before you run anything, because it captures every gotcha hit
during verification, not just the happy path. The shape of it:

```bash
export TT_METAL_HOME=/home/ttuser/tt-metal
export CMAKE_POLICY_VERSION_MINIMUM=3.5          # precaution for cmake 4.x
cd $TT_METAL_HOME
./build_metal.sh --build-tt-train --configure-only
cmake --build build_Release --target _ttml       # ~4 min, warm ccache
```

**The step that actually matters, and the one that's easy to skip:**

```bash
# *** REQUIRED — this is the fix for the std::bad_cast ABI error ***
ninja -C build_Release ttnn/_ttnn.so
cp -a build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
```

Here's the gotcha this exists for, stated plainly because it's the one
mistake most likely to eat your afternoon: if `ttnn` was already built
*before* `tt-train` was enabled in your build tree — which describes **every
pre-built tt-metal image, including TT-QuietBox 2's** — then `import ttml`
raises `std::bad_cast`. The cause is a `nanobind` `STABLE_ABI` tag mismatch:
the old `_ttnn.so` and the newly-built `_ttml` binding disagree about the
shape of `ttnn`'s `Layout`/`DataType` enum registry, and `ttml` can't safely
cast into it. The fix is exactly the two lines above — rebuild `_ttnn.so` so
both binaries share the same stable ABI — or, equivalently, run a single
clean `build_metal.sh --build-tt-train` pass from scratch instead of a
partial `--target _ttml` build. A partial `_ttml`-only build is **not**
enough by itself; you will hit `std::bad_cast` and it will look like a broken
install when it's actually an ABI mismatch. (`import ttnn` still works fine
after the rebuild — this was re-verified during the build that produced this
lab's numbers.)

Last piece: wiring `ttml` onto your Python environment. There's no
`tt-train/pyproject.toml` in this tree, so `pip install .` doesn't apply —
`tt-train` builds as a tt-metal subproject, and `ttml` is made importable
with a `.pth` file pointing at its build output:

```bash
printf '%s\n%s\n' \
  $TT_METAL_HOME/tt-train/sources/ttml \
  $TT_METAL_HOME/build/tt-train/sources/ttml \
  > <venv>/lib/python3.12/site-packages/ttml-custom.pth
```

Two environment variables matter beyond the ones you already know from
earlier lessons: **`TT_METAL_RUNTIME_ROOT`** (set equal to `TT_METAL_HOME` —
the trainer aborts immediately at startup without it, a different failure
mode than the ABI error above) and **`TT_METAL_ARCH_NAME=blackhole`** for
p100/p150/p300c (`wormhole_b0` on n150/n300/T3000/Galaxy — always guard with
`: "${TT_METAL_ARCH_NAME:=wormhole_b0}"` if you're scripting this, per this
project's compatibility rules). `~/tt-scratchpad/llm-from-scratch/train_nano_from_scratch.py`
sets both for you, honouring anything you've already exported.

Verify the build with the one-liner `BUILD_TTML.md` ends on:

```bash
python -c "import ttml, ttnn; print('ttml + ttnn OK')"
```

## Train for real on Blackhole

Here is the exact, verified command — the launcher, driving the upstream
trainer, with the Llama config:

```bash
python ~/tt-scratchpad/llm-from-scratch/train_nano_from_scratch.py \
    --max_steps 20 \
    --data_path $TT_METAL_HOME/tt-train/data/shakespeare.txt
```

Which resolves, under the hood, to:

```bash
TT_METAL_HOME=$TT_METAL_HOME TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME \
TT_METAL_ARCH_NAME=blackhole TT_LOGGER_LEVEL=FATAL \
python $TT_METAL_HOME/tt-train/sources/examples/nano_gpt/train_nanogpt.py \
    --config training_shakespeare_nanollama3_char.yaml \
    --max_steps 20 \
    --data_path $TT_METAL_HOME/tt-train/data/shakespeare.txt
```

That config selects `model_type: llama` — the exact architecture you've read
every line of: RoPE (`theta=500000`), RMSNorm, grouped-query attention (6
heads, 3 KV groups), SwiGLU, embedding dim 384, 6 blocks, block size 256,
char-level tokenizer. **This is the same `nanollama3` shape as the PyTorch
reference across
[Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
through
[The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])
— now training on-device instead of running one forward pass on CPU.**

**Here is what actually happened when we ran it, on this p300c, against
tt-metal v0.73:**

| Step | Loss |
|---|---|
| 1 | 4.6875 |
| 5 | 3.56 |
| 10 | 3.375 |
| 20 | 3.2344 |

Step 1 took **12.3 seconds** — that's not a slow training step, that's the
one-time on-device kernel compile every fresh op shape pays the first time it
runs (you saw the same effect described in earlier labs' honest-flag
sections). Every step after that ran in **~65 ms**, at roughly **16.5
TFLOPS** and **~11% model FLOPS utilization (MFU)**. All 20 steps, compile
included, finished in **13.6 seconds**, and the process exited **0**.

**The general rule underneath that first-step number, worth carrying past
this lab:** the first time any op runs against a new tensor shape, the
device compiles kernels for it — roughly **12–20 seconds** on this
hardware — and caches the result. Every later step against that same shape
just runs the cached kernel, which is why step 2 onward drops straight to
~65 ms. **Never benchmark the first step of anything on this stack** — it's
timing the compiler, not the model.

Loss dropped **monotonically** — 4.6875 → 3.56 → 3.375 → 3.2344 — the same shape
of curve the `smoke()` output in
[The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])
would predict, now with a real gradient signal instead of random initialization
pulling the number down.

Read that curve for what it is: a **modern Llama-3-style language model —
RoPE, RMSNorm, SwiGLU, grouped-query attention, the exact stack you've built
from scratch across five labs — trained from a random initialization to a
measurably lower loss, entirely on Tenstorrent Blackhole hardware**, with a
from-scratch training loop (cross-entropy → backprop → AdamW) instead of a
pre-trained checkpoint. That is the payoff this whole arc has been pointing at
since the altitude ladder in
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]).

### The honest caveat, stated plainly

**Upstream tt-metal does not run continuous-integration training tests for
Blackhole.** `tt-train`'s own CI `GTEST_SKIP`s the `softmax`,
`cross_entropy` (forward and backward), `rmsnorm`, and scaled-dot-product-
attention training-op tests on P100/P150 — so there is no upstream guarantee
that any of this works on Blackhole. What you just read above is not an
upstream claim; it's **this project's own verification**: we built `ttml`
from source against `~/tt-metal` **v0.73** and actually ran the loop on a
p300c, and every one of those "skipped" ops executed correctly and produced
a real, monotonically-decreasing loss curve. Two practical consequences
follow directly from that:

- **Pin your tt-metal version.** This was verified at v0.73 specifically. A
  different version of a fast-moving stack may hit a different edge case —
  don't assume an arbitrary tt-metal checkout reproduces this run unchanged.
- **If the board times out at device open, reset it.** The first run against
  this p300c hit an ethernet-core timeout opening the device; `tt-smi -r`
  cleared it, and the retry succeeded. This is common enough on p300c and
  TT-QuietBox 2 hardware to be worth trying first, before assuming something
  is actually broken.

One more small but real detail: **let `ttml` close the device.** A killed or
malformed script that touches the device without a clean shutdown triggers a
benign-but-alarming-looking teardown abort in
`MetalContext::destroy_all_instances`. `train_nanogpt.py` closes the device
in a `finally` block for exactly this reason — if you're scripting around
it, don't bypass that.

### Going longer: a 3000-step run

20 steps is enough to prove the mechanism works — it's not a training
budget. Raise `--max_steps` and the same command keeps running at the same
steady-state pace: a 3000-step run against this same tiny Shakespeare
excerpt finishes in about **3.3 minutes** on this p300c and drives the loss
from **~4.7 down to ~0.18**.

Don't read that low number as "a better model." On a corpus this small,
3000 steps is enough to **memorize** the excerpt rather than learn anything
that generalizes — the from-scratch training loop is working exactly as
designed, and overfitting a few hundred kilobytes of text is what working
correctly looks like at that step count. It's the same honest distinction
the next section draws in more depth: more steps, on their own, don't buy
you a comparably *trained* model.

## Scaling to ~80M — and being honest about the comparison

This arc has credited [Mini-LLM by
Ashx098](https://github.com/Ashx098/Mini-LLM) since
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]) as the
project whose component choices this whole build follows: RoPE, RMSNorm, SwiGLU,
GQA, SentencePiece BPE 32K. Its real, published numbers: **~80M parameters, 361M
training tokens, ~5 hours on a single A100, final loss ~3.25.**

Our nano run above hit **3.2344** in **20 steps** on a **9.81M-parameter**
model trained on tiny, char-tokenized Shakespeare text. Put next to each
other, those two numbers can look like they're saying the same thing — they
are not, and pretending otherwise would undersell what each project actually
demonstrates. Mini-LLM's ~3.25 is the result of a real training run: 361
million tokens of real text, a 32K-vocabulary subword tokenizer, ~5 hours of
sustained A100 compute, actually learning to model natural language. Our
3.2344 is 20 optimizer steps against a few hundred kilobytes of Shakespeare,
character-tokenized — plenty to prove the mechanism (forward, backward,
AdamW, all executing correctly on Blackhole, loss actually decreasing) but
nowhere near enough data or steps to claim a comparably *trained* model. Same
component family — the same RoPE/RMSNorm/SwiGLU/GQA architecture, same
optimizer — wildly different scale, data, and wall-clock budget. The honest
takeaway is: **the mechanism you verified above is exactly the mechanism
Mini-LLM scales up** — same code path, same `ttml.models.llama`, same
training loop shape — not that a 20-step demo run and a 5-hour production
run landed at "the same" loss.

[The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])
already worked through the config knobs (`embedding_dim`, `num_heads`,
`num_groups`, `num_blocks`, `block_size`, vocab) and the params/DRAM math for
scaling nano toward the ~80M target, so this lab won't repeat that table —
what's worth restating here is the **hardware regime** it implies for
training. A single p300c or p150 (`mesh_shape [1,1]`, per this project's own
compatibility notes — p300c behaves as one Blackhole chip, exactly like
p100) has more than enough DRAM headroom for an 80M-parameter model's
weights, gradients, and AdamW optimizer state — that total lands around 1 GB
by the ~12 bytes/parameter estimate **The Transformer Block & the Model** worked
through, nowhere close to even n150's comparatively tight 12 GB. **You'd reach for a multi-chip mesh
(T3000, Galaxy) only once parameter count grows into the billions** — the
regime where weights + gradients + optimizer state alone start approaching a
single chip's DRAM ceiling, which calls for tensor or pipeline parallelism
across chips. Nothing in this arc's nano-to-~80M range needs that: the exact
20-step run you just executed on one p300c is, mechanically, the same job
Mini-LLM's ~80M/5-hour run is — just longer, on more data, with a bigger
config.

## Where next

You've now done the whole arc:

- [Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]) — picked
  your altitude on the TT-NN ↔ TT-Lang ladder.
- [Tokenizer & Data](command:tenstorrent.showLesson?["lfs-01-tokenizer"]) —
  built a tokenizer from scratch.
- [Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
  — wired up embeddings and the residual stream, with RoPE waiting in the wings.
- [Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])
  — authored attention with grouped-query sharing and a from-scratch TT-Lang
  kernel.
- [The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])
  — assembled SwiGLU and RMSNorm into a full six-block Llama-shape model.
- **Train It & Run for Real** (this lab) — trained that exact model from
  scratch, on real Blackhole hardware, watching the loss actually drop.

That is a complete, honest, from-scratch path from a 32×32 tile to a training
run with a falsifiable result.

If you want to go deeper into `ttml`-based training beyond this arc's nano
scope — different architectures, longer runs, the full custom-training
workflow — this project's existing (currently `blocked`, pending a
standalone `ttml` package, though this lab's build recipe above is a real
way around that block if you're willing to build from source) tracks are
where that continues:

- [Model Architecture Basics](command:tenstorrent.showLesson?["ct7-architecture-basics"])
  — embeddings, attention, feed-forward design, before training from
  scratch.
- [Training from Scratch](command:tenstorrent.showLesson?["ct8-training-from-scratch"])
  — a full nano-trickster (11M params) training run against Shakespeare, with
  scaling-law discussion.

And if you haven't already built your own tt-metal source tree, or want to
revisit the build steps this lab's `ttml` recipe depends on:

- [Build TT-Metalium from Source](command:tenstorrent.showLesson?["build-tt-metal"])

## You built an LLM from scratch, TT-native, on Tenstorrent hardware

That's it — that's the arc. Five labs ago you started with nothing but a
32×32 tile and a promise that you could build a real, modern language model
without ever leaving TT-native ground. You wrote a tokenizer, a residual
stream, attention with RoPE and grouped-query sharing, a full pre-norm
SwiGLU/RMSNorm block, and — in this lab — the training loop that turned all
of it from a correctly-wired-but-random network into one that measurably
learned, loss dropping from 4.6875 to 3.2344 in twenty real steps on a
Blackhole chip. Thank you again to the r/LocalLLaMA post that sparked this,
to Ashx098 and Mini-LLM for the modern component recipe, and to the
TT-QuietBox<sup>®</sup> 2 guide's "Coming From CUDA" chapter for the framing
that carried every lab's grounding section. Go build something bigger with
what you now know.
