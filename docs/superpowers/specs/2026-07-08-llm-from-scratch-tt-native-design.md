# Build an LLM from Scratch, TT-Native — Design Spec

**Date:** 2026-07-08
**Branch:** `llm_from_scratch`
**Status:** Design approved, pending spec review

## Acknowledgments (bake into Lab 0)

This arc is a **thank-you-forward** lesson, in keeping with Tenstorrent's value of
saying thanks. Lab 0 opens by crediting, with links and genuine thanks:

- **The r/LocalLLaMA "I built an 80M-parameter LLM from scratch" project** — the spark
  for this arc. (Reddit post; link to be captured. It was behind a JS challenge at
  authoring time — capture the canonical URL and the author's repo before publishing.)
- **The "Coming From CUDA" chapter** of the internal **tt-quietbox2-guide**
  (`src/content/tracks/ml-practitioner/chapters/01-coming-from-cuda.md`) — the source of
  the CUDA-grounding layer. We adapt its ideas (with attribution), we do not copy it.

## Goal

A **standalone, multi-part, code-first arc** that teaches a reader to build a small
GPT-style LLM **from scratch**, expressing it **TT-native from the first line** — using
**TT-Lang as a tool for *inception* (authoring kernels from scratch), not just conversion
(importing CUDA/Triton/PyTorch)**. Accessible to CUDA programmers throughout.

It is deliberately separate from the existing `ct1`–`ct8` Custom-Training track (which is
`blocked` on `ttml` packaging). This arc links to `ct8`/`build-tt-metal` for the real
training graduation path but is never itself blocked, because its hands-on runtime is the
functional simulator / browser playground plus validated ttnn forward inference.

## Three load-bearing decisions (from brainstorming)

1. **Shape:** multi-part code-first arc (like `cs-fundamentals-08-matmul-labs`), 5 labs + intro.
2. **Model target:** two-tier — build & run at **nano scale** (converges live), framed as
   scaling to the **~80M** "hero" number (same code, config knobs + DRAM/time math).
3. **Execution contract:** **layered per-lab runtime** — PyTorch reference for concepts,
   TT-Lang functional sim / browser playground for inception kernels, and a final
   "run for real" lab offering **ttnn forward inference on Blackhole** (validated) plus the
   **ttml source-build training** path (honestly caveated).

## Research findings that constrain the design

Verified against `vendor/tt-metal`, `vendor/tt-lang`, GitHub releases, and PyPI (July 2026):

- **Latest tt-metal:** stable **v0.73.1** (2026-06-26), rc **v0.74.0-rc1** (2026-07-02).
  Pin the lesson to these; the `ct8` v0.67.0+ floor is comfortably met.
- **`ttml` (tt-train) is still source-only** — no pip wheel / deb. Bindings
  (`tt-train/sources/ttml/nanobind/`) are real and functional but require building from a
  tt-metal source tree. The `ct7`/`ct8` "blocked pending a standalone package" note remains
  accurate. Canonical from-scratch reference: **`nano_gpt/nanogpt_primitives_example.py`**
  (builds NanoGPT from `ttml.ops` + ttnn primitives; the removed `train_nanogpt.py` is NOT
  canonical).
- **Blackhole training — EMPIRICALLY VALIDATED (2026-07-08 on this p300c).** Upstream CI
  `GTEST_SKIP`s tt-train's `softmax`, `cross_entropy` (fwd+bwd), `rmsnorm`, and `sdpa`
  tests on P100/P150, so upstream makes no Blackhole-training guarantee. **However, we built
  `ttml` against `~/tt-metal` v0.73 and actually ran `nanogpt_primitives_example.py` on the
  p300c: a real forward+backward+AdamW loop trained on-device, loss dropping monotonically
  4.59 → 3.28 over 10 steps (exit 0). Those "skipped" ops all executed correctly.** So the
  lesson CAN claim from-scratch training works on Blackhole p300c — framed honestly as
  "upstream doesn't CI this on BH; we verified it at v0.73, pin your version and reset the
  board if needed." See the "ttml build recipe & verification" section below.
- **TT-Lang** is v1.0.0. From-scratch authoring is first-class (the "inception" angle is
  legitimate). Kernel model = one `@ttl.operation` with **three concurrent threads**
  (`@ttl.compute` + two `@ttl.datamovement`) coordinated through typed L1 ring buffers
  (Dataflow Buffers via `.reserve()`/`.wait()`). This maps *exactly* onto the CUDA guide's
  **reader → compute → writer** framing. Use `@ttl.operation` — the `@ttl.kernel` /
  `buffer_factor` API in the Hermes-spoke skills is **stale**; do not cite it.
- **TT-Lang transformer kernels already exist** in `vendor/tt-lang/examples/`:
  `test_transformer_block.py` (RMSNorm+QKV, RoPE, attention/softmax, out-proj+residual,
  MLP relu²) — sim-validated vs PyTorch; plus `matmul.py`, `eltwise_add.py`,
  `matmul-tutorial/step_*`. **Simulator runs ahead of the hardware compiler** — some
  kernels are sim-validated only; flag this honestly per lab.
- **TT-Lang ops are drop-in `ttnn.Tensor` calls** — a hand-written kernel splices into an
  otherwise-ttnn model op-by-op (boundary = `ttnn.Tensor`, `TILE_LAYOUT`, on device).
- **ttnn/ttml API names** (verified): `ttnn.embedding`, `ttnn.matmul`/`ttnn.linear`,
  `ttnn.softmax`, `ttnn.rms_norm`, `ttnn.transformer.scaled_dot_product_attention`;
  training needs **ttml** wrappers (ttnn has no autograd): `ttml.ops.cross_entropy_loss`,
  `ttml.optimizers.AdamW`/`MorehAdamW`.
- **Blackhole runtime:** `mesh_shape [1,1]` for single-chip p300c/p150;
  `TT_METAL_ARCH_NAME=blackhole` (guard: `: "${TT_METAL_ARCH_NAME:=wormhole_b0}"`); never
  `DispatchCoreAxis.ROW`. p300c behaves as a single Blackhole chip (p100 mode); TT-QuietBox 2
  = 4× independent p300c.

## Arc structure

New track category **`llm-from-scratch`**, lesson IDs **`lfs-00`…`lfs-05`**.
Every lab follows the same rhythm:

> **Coming-from-CUDA callout → PyTorch reference (understand) → TT-native expression
> (ttnn / TT-Lang inception) → run it (sim / playground / hw) → "graduate to real
> hardware" box.**

### CUDA grounding layer (adapted from `01-coming-from-cuda.md`, with attribution)

1. **"Pick Your Altitude" ladder** — TT-Forge → **TTNN** → **TT-Lang** → Metalium, mapped to
   `model.cuda()` → cuBLAS/cuDNN → custom CUDA kernel → PTX. Lab 0's orienting frame:
   *"build at TTNN altitude, descend to TT-Lang for the hot kernels — that descent is inception."*
2. **CUDA → Tensix concept map** — SM→Tensix core, thread block→tile-on-a-core, shared
   memory→L1 SRAM, `<<<g,b>>>`→automatic dispatch, warp scheduler→explicit reader/compute/writer
   pipeline, `cudaMemcpy`→`ttnn.from_torch`/`to_torch`. Appears as a per-lab "Coming from CUDA"
   callout scoped to that lab's component.
3. **"Custom kernels without the dread — the agentic shortcut"** — the inception thesis: on
   CUDA the kernel spec lives in the programmer's head; in TT-Lang it lives *in the source*
   (arrivals in → tile math → departures out), which is why you can author TT-native from
   line one and even hand an agent a reader/compute/writer spec.

*Rendering note:* the QB2 guide uses shortcodes (`:::callout`, `{% tensixviz %}`, personas)
that do NOT render in VSCode walkthroughs. Bring the *ideas* into plain-markdown toolkit
style. Reconcile hardware numbers against source when authoring (the guide rounds L1 to
"1.5 MB" / 120 cores; the TT-Lang spec says L1 ≈ 1464 KB and Blackhole grid up to 13×10).

### Labs

| Lab | Builds from scratch | Inception / TT-native moment | Runs where |
|---|---|---|---|
| **lfs-00 — Intro: Pick Your Altitude & the Tile** | The whole picture; why TT-native from line one; 32×32 tile, `ttnn.Tensor`/`TILE_LAYOUT`, reader→compute→writer; two-tier promise; runtime matrix; **acknowledgments/thanks** | altitude ladder + CUDA concept map | reading |
| **lfs-01 — Tokenizer & Data** | BPE tokenizer (train on TinyStories), encode/decode, batching; tokens → tiled tensors; `ttnn.embedding` preview | CUDA callout: `cudaMemcpy`↔`from_torch`/`to_torch`, no unified memory | pure Python (anywhere) |
| **lfs-02 — Embeddings & the Residual Stream** | Token + positional embeddings; residual stream | **First inception kernel:** elementwise-add TT-Lang kernel (from `eltwise_add.py`) live in **browser playground**; CUDA callout: shared memory→L1, three explicit threads vs warp scheduler | functional sim + playground |
| **lfs-03 — Attention from Scratch (centerpiece)** | Multi-head self-attention + softmax (PyTorch ref: Q·Kᵀ, scale, mask, softmax, ·V) | **TT-Lang attention/softmax inception kernel** (from `test_transformer_block.py`), sim-validated vs PyTorch; CUDA callout: FlashAttention↔reader/compute/writer; **agentic shortcut** shown concretely; honest sim-vs-compiler flag | functional sim |
| **lfs-04 — The Block & the Model** | MLP (GELU/relu²) + RMSNorm + residuals → full block → stack into nano model | **TT-Lang RMSNorm + matmul inception kernels** (`test_transformer_block.py`/`matmul.py`); kernels as drop-in `ttnn.Tensor` ops; **nano↔80M config table** + params/DRAM math; CUDA callout: kernel fusion transfers | sim + ttnn |
| **lfs-05 — Train It & Run for Real** | Training loop from scratch: cross-entropy, AdamW, backprop (mirrors `nanogpt_primitives_example.py`) | **Real, verified from-scratch training on Blackhole p300c** (loss 4.59→3.28 in 10 steps, exit 0) via `ttml` built from source; the exact build recipe + the `std::bad_cast` ABI fix + board-reset/env-var gotchas; honest "upstream doesn't CI training on BH, pin your version" note; **80M scaling** math; links to `ct8`/`build-tt-metal` | **ttml training on BH (real, validated)** + ttnn forward |

## Runtime & accuracy mechanics (Section 3)

- **Pin** tt-metal v0.73.1 / v0.74.0-rc1; state it in front matter (`minTTMetalVersion`) and prose.
- **Verify everything against `vendor/`** before publishing: API names, paths, flags, env vars.
  Use `@ttl.operation`; ttnn/ttml names per the research table above.
- **Drift management:** TT-Lang kernels are *adapted from* `vendor/tt-lang/examples/`. Each
  embedded kernel carries a source-path note and a drift-check reminder (same spirit as
  `scripts/check-sim-lite-drift.py`). If Lab 3 gets a browser-playground softmax/attention
  kernel, add it under `content/web/ttlang-sim-lite/kernels/` and wire it into the drift check.
- **Honest hardware flags** as plain-markdown callouts: simulator-ahead-of-compiler; BH
  training-op skips; ttml source-build requirement. Never claim BH training validation.
- **Runnable code** shipped under `content/templates/llm-from-scratch/` (complete scripts);
  lessons walk excerpts code-first so the prose and the files stay in sync.

## Integration (Section 4)

- **Files:** `content/lessons/lfs-0{0..5}-*.md` (6 lessons) + `content/templates/llm-from-scratch/`.
- **Registry:** add `llm-from-scratch` category to `content/lesson-registry.json`; populate
  lesson entries via `npm run generate:lessons -- --execute` (front matter is source of truth),
  then hand-add `order`/`previousLesson`/`nextLesson`/`completionEvents`.
- **Walkthrough:** 6 steps in `package.json` → `contributes.walkthroughs[0].steps` with
  `metadata`, ordering, and completion events.
- **Playground:** reuse the existing `ttlang-sim` playground (`playground: ttlang-sim` front
  matter) for Lab 2 (`eltwise_add` already bundled) and Lab 4 (`matmul_1d`/`matmul_relu`
  bundled). Optional stretch: add a softmax/attention kernel for Lab 3.
- **Version & changelog:** bump `package.json` (MINOR — new track; `0.0.518 → 0.1.0`, or PATCH
  per maintainer preference) and add a CHANGELOG.md entry (no line numbers).
- **Gates:** `npm run validate:lessons` and `npm run build` must pass (build runs validation).

## ttml build recipe & verification (verified 2026-07-08 on p300c, tt-metal v0.73)

This is the reusable artifact for Lab 5 (and it retires the ct7/ct8 "no way to get ttml"
blocker for anyone willing to build from source). Verified end-to-end on this machine.

**Recipe** (against an existing tt-metal source+build tree; `~/tt-metal` here):
```bash
export TT_METAL_HOME=/home/ttuser/tt-metal
export CMAKE_POLICY_VERSION_MINIMUM=3.5          # precaution for cmake 4.x
cd $TT_METAL_HOME
./build_metal.sh --build-tt-train --configure-only
cmake --build build_Release --target _ttml       # ~4 min warm ccache
# *** REQUIRED: rebuild ttnn's nanobind so its ABI matches ttml ***
ninja -C build_Release ttnn/_ttnn.so
cp -a build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
# wire ttml onto the venv (see INSTALLING_TTML.md; it says py3.10, this box is 3.12)
printf '%s\n%s\n' \
  $TT_METAL_HOME/tt-train/sources/ttml \
  $TT_METAL_HOME/build/tt-train/sources/ttml \
  > <venv>/lib/python3.12/site-packages/ttml-custom.pth
```
- **There is NO `tt-train/pyproject.toml`** in this tree — the `pip install .` path does not
  apply. tt-train builds as a tt-metal subproject; ttml is wired via a `.pth`.
- **Headline pitfall — `std::bad_cast` on `import ttml`:** happens whenever `ttnn` was built
  *before* tt-train was enabled (i.e. every pre-built tt-metal image, including TT-QuietBox 2).
  Cause: nanobind STABLE_ABI tag mismatch between the old `_ttnn.so` and the new `_ttml`, so
  ttml can't see ttnn's `Layout`/`DataType` enum registry. Fix = rebuild `_ttnn.so` (as above)
  so both share the stable ABI, or do a single clean `build_metal.sh --build-tt-train` pass.
  A partial `--target _ttml` build alone is NOT enough. (`import ttnn` re-verified afterward.)

**Run (verified on p300c):**
```bash
cd $TT_METAL_HOME/tt-train/sources/examples/nano_gpt
TT_METAL_HOME=$TT_METAL_HOME TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME \
TT_METAL_ARCH_NAME=blackhole TT_LOGGER_LEVEL=FATAL \
python nanogpt_primitives_example.py --data_path <shakespeare.txt> --max_steps 10 --batch_size 2
# Step 0 Loss 4.59 (compile) → Step 10 Loss 3.28, exit 0
```
- **Extra env vars beyond the usual:** `TT_METAL_RUNTIME_ROOT` (in addition to `TT_METAL_HOME`)
  — the example aborts immediately without it — and `TT_METAL_ARCH_NAME=blackhole`.
- **Needs training text** via `--data_path` (no bundled `data/`); it char-tokenizes a plain file.
- **Board may need `tt-smi -r` first** — first run hit an ethernet-core timeout at device open;
  a reset cleared it. Worth a note for p300c / QuietBox 2 users.
- **Let ttml close the device** — malformed/partial scripts that touch the device without a
  clean close trigger a benign teardown abort in `MetalContext::destroy_all_instances`.
- ttml submodules confirmed importable: `autograd`, `ops` (loss, attention, layernorm, linear,
  embedding, unary, binary, dropout, multi_head_utils, reshape), `optimizers`, `models`,
  `modules`, `core`, `init`, `fsdp`, `Mesh`.

## Metadata policy

- `supportedHardware`: full WH + BH set incl. `p300c` (+ `sim` where the playground/functional
  sim is the runtime).
- `status`: `validated` for Lab 5's ttml BH-training path (verified on p300c at v0.73) and the
  sim/playground labs; `draft` for anything not yet actually run. Pin `minTTMetalVersion` and
  state "verified on p300c v0.73; upstream doesn't CI BH training."
- `validatedOn`: include `p300c` for Lab 5 (empirically confirmed) and `sim` where the
  playground/functional sim is the runtime; only list what was actually exercised.

## Non-goals (YAGNI)

- Not unblocking or rewriting `ct7`/`ct8` (separate track; link only).
- Not full 80M convergence in-lesson (nano converges; 80M is the framed scale target).
- Not a Metalium/C++ descent (mention as the floor of the altitude ladder; out of scope).
- No new custom UI — VSCode-native walkthrough + existing playground only.

## Open items to resolve while authoring

1. Capture the canonical Reddit URL + author repo for the acknowledgment.
2. Confirm exact nano config (embed dim / heads / blocks / vocab / seq) and the 80M config,
   and the params/DRAM math, against a real `TransformerConfig` — use the verified
   `nanogpt_primitives_example.py` config as the nano baseline.
3. Decide whether Lab 3 gets a browser-playground kernel or stays functional-sim-only.
4. Reconcile L1 size / core-count numbers against `vendor/tt-lang` spec + tt-metal.
5. Version bump: **MINOR `0.0.518 → 0.1.0`** (approved — new track).

**RESOLVED by the 2026-07-08 build:** ttml IS buildable + importable, and from-scratch
training IS verified on p300c Blackhole. Lab 5 is a real on-hardware training lab, not a
documented-only path. The build recipe + `std::bad_cast` fix are captured above.

---

## ADDENDUM — Modern Llama-3 pivot (approved & VERIFIED 2026-07-08)

After reading the full inspiration post, the arc pivots from GPT-2-style to the **modern Llama-3 component set** the post champions. Credit/thank **Mini-LLM by Ashx098** (https://github.com/Ashx098/Mini-LLM, HF https://huggingface.co/Ashx098/Mini-LLM) alongside the existing acknowledgments — it uses RoPE, RMSNorm, SwiGLU, GQA, SentencePiece BPE 32K (80M params / 361M tokens / ~5h on one A100 / final loss ~3.25).

**Why this works on TT:** the TT stack already ships this exact stack. `ttml.models.llama` (RoPE + RMSNorm + SwiGLU + GQA) is driven by `train_nanogpt.py` (Blackhole-aware), and the TT-Lang kernels we teach (`test_transformer_block.py`) are already Llama-style (RoPE `rotary_qk_kernel`, RMSNorm).

**VERIFIED on this p300c (2026-07-08):** `train_nanogpt.py --config training_shakespeare_nanollama3_char.yaml --max_steps 20` — model_type=llama, 6 heads / **3 KV groups (GQA)**, dim 384, 6 blocks, **RoPE theta 500000**, char tokenizer. Loss **4.69 → 3.23** over 20 steps, ~65 ms/step, 16.5 TFLOPS, MFU ~11%, exit 0. This is the arc's HERO run (supersedes the GPT-2 nanogpt run for Lab 5; GPT-2 stays a footnote/contrast).

### Component decisions (supersede the GPT-2 choices above)
| Component | Pivot decision | TT expression |
|---|---|---|
| Positional | **RoPE** replaces learned positional embeddings | TT-Lang `rotary_qk_kernel` (elementwise, no sim-broadcast issue) + `ttml.ops.rope`; add a `kernels/rope.py` template |
| Norm | **RMSNorm** (already done in reference + `kernels/rmsnorm.py`) | existing template (HW-path only) |
| MLP | **SwiGLU** replaces GELU MLP | PyTorch reference + `ttml.ops.swiglu` (SiLU·gate); NO from-scratch TT-Lang SwiGLU kernel (out of scope — teach via ttml op + note) |
| Attention | **GQA** (KV-head sharing) extends MHA | reference GQA in PyTorch + `ttml.ops.grouped_heads_creation`; TT-Lang attention kernel stays single-head (HW-path only), GQA taught as the sharing pattern |
| Tokenizer | Keep from-scratch **byte-BPE** for pedagogy (Lab 1) + frame **SentencePiece BPE 32K** as what production/Mini-LLM uses; the verified nano hero run uses the char tokenizer | Lab 1 copy |

### Per-lab rework (supersedes the earlier Labs table)
- **Lab 0:** update nano-config refs (nanollama3: dim 384 / 6 heads / 3 KV groups / 6 blocks / RoPE θ=500000) + add Mini-LLM thanks; runtime matrix Lab 5 = verified Llama numbers.
- **Lab 1:** add SentencePiece-32K framing (Mini-LLM) alongside the from-scratch byte-BPE; note the hero run uses char tokenization.
- **Lab 2:** **RoPE** (not learned positional embeddings) — new `kernels/rope.py` + PyTorch RoPE reference; keep eltwise_add playground as the "first kernel" mechanic.
- **Lab 3:** **GQA** extends the MHA build (KV-head sharing, why it's efficient for inference).
- **Lab 4:** **SwiGLU** + RMSNorm block (SwiGLU via PyTorch ref + ttml.ops.swiglu; RMSNorm kernel already exists).
- **Lab 5:** hero run = `nanollama3_char` on Blackhole with the VERIFIED loss curve above; 80M scale via Mini-LLM's numbers (361M tokens, ~5h A100, loss ~3.25).
- **Templates:** `reference_gpt.py` → Llama-style (RoPE + RMSNorm + SwiGLU + GQA), still pure-PyTorch CPU; `train_nano_from_scratch.py` → drive the verified `train_nanogpt.py` llama path; add `kernels/rope.py`.
