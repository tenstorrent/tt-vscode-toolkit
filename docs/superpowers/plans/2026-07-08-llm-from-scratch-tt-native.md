# Build an LLM from Scratch, TT-Native — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a standalone, code-first lesson arc (`lfs-00`…`lfs-05`) that teaches building a small GPT from scratch, TT-native from line one, using TT-Lang for *inception* — with a real, verified from-scratch training run on Blackhole p300c.

**Architecture:** Six markdown lessons in `content/lessons/`, surfaced via `content/lesson-registry.json` (new `llm-from-scratch` category). Each lab excerpts verified runnable code kept in `content/templates/llm-from-scratch/`. TT-Lang kernels are adapted from `vendor/tt-lang/examples/`; the training path uses `ttml` built from `~/tt-metal` per the verified recipe in the spec.

**Tech Stack:** Markdown (VSCode-native rendering), `js-yaml`-based registry sync scripts, Python (PyTorch reference, `ttnn`, `ttml`, TT-Lang DSL), the `ttlang-sim-lite` Pyodide playground.

**Spec:** `docs/superpowers/specs/2026-07-08-llm-from-scratch-tt-native-design.md` (approved).

## Global Constraints

- **Version bump:** `package.json` `0.0.518 → 0.1.0` (MINOR — new track). Bump once (Task 1).
- **tt-metal pin:** reference **v0.73.1** (stable) / **v0.74.0-rc1**; `minTTMetalVersion` floor v0.67.0. Training verified on `~/tt-metal` v0.73.0-dev on p300c.
- **Registry is the surfacing mechanism** — there is NO `contributes.walkthroughs` in `package.json` anymore (it was removed). Do NOT add walkthrough steps to `package.json`. Register lessons in `content/lesson-registry.json` only.
- **Registry sync:** markdown front matter is source of truth for `id,title,description,category,tags,supportedHardware,status,validatedOn,estimatedMinutes`. Run `npm run generate:lessons -- --execute` to sync; hand-edit `order,previousLesson,nextLesson,completionEvents,markdownFile`. `npm run validate:lessons` must pass (and runs inside `npm run build`).
- **Gates for every task:** `npm run build` must pass (it runs `validate:lessons`, `validate:command-uris`, link tests, and a full site build). Never commit with a failing build.
- **WH/BH compatibility (from CLAUDE.md):** `hf` CLI not `huggingface-cli`; never `DispatchCoreAxis.ROW`; don't assume `~/tt-metal` exists (link `build-tt-metal`); `TT_METAL_ARCH_NAME=blackhole` for P-series with the `: "${TT_METAL_ARCH_NAME:=wormhole_b0}"` guard; `p300c` in `supportedHardware`/`validatedOn` where applicable; `pip install --upgrade pip setuptools wheel` before dev installs.
- **TT-Lang accuracy:** use `@ttl.operation` (NOT the stale `@ttl.kernel`/`buffer_factor`); the browser "playground" is this repo's `ttlang-sim-lite` fork, not an upstream TT-Lang feature; simulator runs ahead of the hardware compiler — flag sim-only kernels honestly.
- **Trademark/branding:** match existing lessons' use of `<sup>` trademark marks (TT-NN™, TT-Metalium™, Wormhole™, etc.) and the brand palette; no right-side border characters in any ASCII art.
- **Acknowledgments (Lab 0):** credit + thank the r/LocalLLaMA "80M LLM from scratch" project and the `tt-quietbox2-guide` "Coming From CUDA" chapter, with links.
- **Copy rule:** adapt the CUDA-guide ideas into plain markdown; its shortcodes (`:::callout`, `{% tensixviz %}`, personas) do NOT render in this repo — do not paste them.

---

### Task 1: Track scaffolding & registration

Create all six lesson files with **front matter only** (bodies filled in later tasks), register the new category and lessons, and bump the version. Deliverable: `npm run build` passes with six new `llm-from-scratch` lessons present and navigable.

**Files:**
- Create: `content/lessons/lfs-00-intro.md`, `lfs-01-tokenizer.md`, `lfs-02-embeddings.md`, `lfs-03-attention.md`, `lfs-04-block-and-model.md`, `lfs-05-train-and-run.md`
- Modify: `content/lesson-registry.json` (add category + 6 lesson entries)
- Modify: `package.json` (version → `0.1.0`)

**Interfaces:**
- Produces: lesson IDs `lfs-00`…`lfs-05`; category id `llm-from-scratch`; each markdown file path used as `markdownFile` in the registry.

- [ ] **Step 1: Create the six lesson files with front matter + an H1 placeholder body.**

Each file starts with front matter, then a single H1 and a one-line "authored in Task N" HTML comment (so the site builds). Use these exact front-matter blocks:

`lfs-00-intro.md`:
```markdown
---
id: lfs-00-intro
title: "Build an LLM from Scratch — Pick Your Altitude"
description: >-
  Start an LLM from scratch, TT-native from line one. Meet the 32×32 tile, the
  reader→compute→writer pipeline, and the "pick your altitude" ladder — grounded
  for CUDA programmers.
category: llm-from-scratch
tags:
  - llm
  - from-scratch
  - tt-lang
  - cuda
  - transformers
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
  - simulator
status: draft
estimatedMinutes: 20
---

# Build an LLM from Scratch — Pick Your Altitude

<!-- Body authored in Task 3 -->
```

`lfs-01-tokenizer.md` (title "Tokenizer & Data from Scratch", tags `tokenizer,bpe,data,tinystories,llm`, estimatedMinutes 30, same hardware, status draft):
```markdown
---
id: lfs-01-tokenizer
title: "Tokenizer & Data from Scratch"
description: >-
  Build a BPE tokenizer and data pipeline from scratch, then see how a token
  sequence becomes tiled ttnn tensors on Tenstorrent hardware.
category: llm-from-scratch
tags:
  - tokenizer
  - bpe
  - data
  - tinystories
  - llm
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 30
---

# Tokenizer & Data from Scratch

<!-- Body authored in Task 4 -->
```

`lfs-02-embeddings.md` (title "Embeddings & the Residual Stream", tags `embeddings,positional,residual,tt-lang,playground`, estimatedMinutes 30):
```markdown
---
id: lfs-02-embeddings
title: "Embeddings & the Residual Stream"
description: >-
  Build token and positional embeddings, meet the residual stream, and write
  your first TT-Lang inception kernel live in the browser playground.
category: llm-from-scratch
tags: [embeddings, positional, residual, tt-lang, playground]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 30
---

# Embeddings & the Residual Stream

<!-- Body authored in Task 5 -->
```

`lfs-03-attention.md` (title "Attention from Scratch", tags `attention,softmax,tt-lang,flashattention,inception`, estimatedMinutes 40):
```markdown
---
id: lfs-03-attention
title: "Attention from Scratch"
description: >-
  Build multi-head self-attention from scratch and author the TT-Lang
  attention/softmax inception kernel — the reader→compute→writer payoff.
category: llm-from-scratch
tags: [attention, softmax, tt-lang, flashattention, inception]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 40
---

# Attention from Scratch

<!-- Body authored in Task 6 -->
```

`lfs-04-block-and-model.md` (title "The Transformer Block & the Model", tags `transformer,mlp,rmsnorm,matmul,tt-lang`, estimatedMinutes 40):
```markdown
---
id: lfs-04-block-and-model
title: "The Transformer Block & the Model"
description: >-
  Assemble MLP, RMSNorm, and residuals into a full transformer block, stack it
  into a nano GPT, and see TT-Lang kernels drop in as ttnn ops. Scale to 80M.
category: llm-from-scratch
tags: [transformer, mlp, rmsnorm, matmul, tt-lang]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 40
---

# The Transformer Block & the Model

<!-- Body authored in Task 7 -->
```

`lfs-05-train-and-run.md` (title "Train It & Run for Real", tags `training,adamw,ttml,blackhole,nanogpt`, estimatedMinutes 45, `validatedOn: [p300c]`):
```markdown
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
minTTMetalVersion: v0.67.0
---

# Train It & Run for Real

<!-- Body authored in Task 8 -->
```

- [ ] **Step 2: Add the `llm-from-scratch` category to `content/lesson-registry.json`.**

Insert into the `categories` array (place after `custom-training`; keep `order` unique — existing max category order is 10 for cs-fundamentals):
```json
{
  "id": "llm-from-scratch",
  "title": "🔬 Build an LLM from Scratch",
  "description": "Build a small GPT from scratch, TT-native from line one — with TT-Lang for inception",
  "order": 11,
  "icon": "beaker"
}
```

- [ ] **Step 3: Run the generator to add lesson entries from front matter.**

Run: `npm run generate:lessons -- --execute --force`
Expected: 6 `ADD` entries for `lfs-00`…`lfs-05`; a backup written to `.backups/`.

- [ ] **Step 4: Hand-edit the 6 new registry entries for navigation/completion.**

For each new lesson entry set `order` (200–205, above current max of 100 so they group at the end of their category), `markdownFile` (e.g. `content/lessons/lfs-00-intro.md`), `completionEvents: []` (no command triggers yet), and chain `previousLesson`/`nextLesson`:
- `lfs-00`: prev `null`/omit, next `lfs-01`
- `lfs-01`: prev `lfs-00`, next `lfs-02` … through …
- `lfs-05`: prev `lfs-04`, next `null`/omit

- [ ] **Step 5: Bump version.** In `package.json` set `"version": "0.1.0"`.

- [ ] **Step 6: Verify the build (this is the test).**

Run: `npm run build`
Expected: `✅ All ... lessons are valid!`, link tests pass, site build lists all six `lfs-*` lessons under the new catalog category, exit 0.

- [ ] **Step 7: Commit.**
```bash
git add content/lessons/lfs-*.md content/lesson-registry.json package.json
git commit -m "feat(lfs): scaffold Build-an-LLM-from-Scratch track (lfs-00..05) + registry"
```

---

### Task 2: Verified runnable reference code + build recipe

Create the code all labs excerpt, and prove it runs. This is the source of truth — labs quote from these files so prose and code never drift.

**Files:**
- Create: `content/templates/llm-from-scratch/BUILD_TTML.md` (the verified build recipe)
- Create: `content/templates/llm-from-scratch/train_nano_from_scratch.py` (ttml training runner — thin wrapper mirroring the verified `nanogpt_primitives_example.py`)
- Create: `content/templates/llm-from-scratch/reference_gpt.py` (pure-PyTorch nano GPT reference: tokenizer hookup, embeddings, MHA, block, forward — for pedagogy/excerpts)
- Create: `content/templates/llm-from-scratch/kernels/eltwise_add.py`, `kernels/attention.py`, `kernels/rmsnorm.py`, `kernels/matmul.py` (TT-Lang inception kernels, adapted from `vendor/tt-lang/examples/`)
- Create: `content/templates/llm-from-scratch/README.md` (what each file is, how to run, drift-source notes)

**Interfaces:**
- Produces: file paths + verified commands that Tasks 3–8 embed verbatim; the nano config used as the pedagogical baseline.

- [ ] **Step 1: Write `BUILD_TTML.md`** by copying the verified recipe from the spec's "ttml build recipe & verification" section (build_metal.sh --build-tt-train, the `_ttnn.so` rebuild fixing `std::bad_cast`, the `.pth` wiring, the `TT_METAL_RUNTIME_ROOT`/`TT_METAL_ARCH_NAME` env vars, `tt-smi -r` note). Include the "no `pyproject.toml` in this tree" caveat and link `build-tt-metal`.

- [ ] **Step 2: Copy the TT-Lang kernels from vendor and record their source.** For each kernel file, copy the relevant `@ttl.operation` from the cited vendor source and add a header comment: `# Adapted from vendor/tt-lang/examples/<file> @ <commit>. Drift-check before publishing.`
  - `kernels/eltwise_add.py` ← `vendor/tt-lang/examples/eltwise_add.py`
  - `kernels/attention.py` ← `attention_kernel` in `vendor/tt-lang/examples/test_transformer_block.py`
  - `kernels/rmsnorm.py` ← `norm_qkv_kernel`/`norm_mlp_residual_kernel` in the same file
  - `kernels/matmul.py` ← `vendor/tt-lang/examples/matmul.py`

- [ ] **Step 3: Simulate the kernels (test).** For each kernel run the functional sim (or `/ttl-simulate`) and confirm no deadlock and a correctness check vs a numpy/torch reference.
Run (example): `python content/templates/llm-from-scratch/kernels/eltwise_add.py`
Expected: kernel runs in sim; printed max-abs-error vs reference ≈ 0 (bf16 tolerance). Record which kernels are sim-only vs compiler-supported (per spec: attention/softmax may be sim-only) and annotate each file header accordingly.

- [ ] **Step 4: Write `reference_gpt.py`** — a minimal, readable PyTorch nano GPT (config: embed_dim, n_heads, n_blocks, vocab, seq from the verified nano baseline). Pure PyTorch, CPU-runnable, no TT deps — used for the "understand" half of Labs 1–4.

- [ ] **Step 5: Run the PyTorch reference (test).**
Run: `python content/templates/llm-from-scratch/reference_gpt.py --smoke`
Expected: builds the model, runs one forward pass on random tokens, prints output shape `[batch, seq, vocab]`, exit 0.

- [ ] **Step 6: Write `train_nano_from_scratch.py`** — a thin, documented runner that invokes/mirrors the verified `~/tt-metal/tt-train/sources/examples/nano_gpt/nanogpt_primitives_example.py` flow (build from `ttml.ops`, AdamW, cross-entropy, N steps), with the required env vars set and a clean device close. Include a docstring pointing at `BUILD_TTML.md` as a prerequisite.

- [ ] **Step 7: Run real training on Blackhole (test — the headline claim).**
Run (with ttml built per BUILD_TTML.md):
```bash
TT_METAL_HOME=/home/ttuser/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal \
TT_METAL_ARCH_NAME=blackhole TT_LOGGER_LEVEL=FATAL \
python content/templates/llm-from-scratch/train_nano_from_scratch.py --max_steps 10 --batch_size 2 --data_path <shakespeare.txt>
```
Expected: loss prints and decreases across ~10 steps (≈4.6 → ≈3.3), exit 0. If the board errors at device open, `tt-smi -r` and retry. Record the actual loss curve for use in Lab 5.

- [ ] **Step 8: Write `README.md`** listing each file, its purpose, run command, and drift-source (vendor path + commit). Note templates are NOT shipped as extension features — they are lesson reference code.

- [ ] **Step 9: Verify build + commit.**
Run: `npm run build` (Expected: pass — templates don't affect lesson validation but confirm nothing broke.)
```bash
git add content/templates/llm-from-scratch/
git commit -m "feat(lfs): verified runnable templates — ttml training + TT-Lang kernels + PyTorch ref"
```

---

### Task 3: Author Lab 0 (Intro — Pick Your Altitude & the Tile)

**Files:** Modify `content/lessons/lfs-00-intro.md` (replace placeholder body).

**Interfaces:** Consumes nothing. Produces the altitude ladder + CUDA→Tensix map + acknowledgments referenced by later labs' callouts.

- [ ] **Step 1: Write the body** with these sections (plain markdown, adapt from spec + CUDA guide, verify numbers against `vendor/tt-lang` spec + tt-metal):
  1. **Thanks & inspiration** — credit/link the r/LocalLLaMA 80M project and the `tt-quietbox2-guide` "Coming From CUDA" chapter (open item #1: capture the canonical Reddit URL + author repo before publishing).
  2. **What we're building** — a small GPT from scratch, TT-native from line one; two-tier promise (nano runs live; 80M is the scale target).
  3. **Pick Your Altitude ladder** — table: `model.cuda()`→TT-Forge, cuBLAS/cuDNN→TTNN, custom CUDA kernel→TT-Lang, PTX→Metalium. State: "build at TTNN altitude, descend to TT-Lang for hot kernels — that descent is inception."
  4. **The 32×32 tile & `ttnn.Tensor`/`TILE_LAYOUT`.**
  5. **reader→compute→writer** — introduce as the Tensix execution model and foreshadow the TT-Lang 3-thread kernel; explicitly map from CUDA's warp-scheduler-hides-latency model.
  6. **CUDA→Tensix concept map** table (SM→Tensix core, thread block→tile-on-a-core, shared memory→L1 SRAM ≈1.5MB, `cudaMemcpy`→`from_torch`/`to_torch`, `<<<g,b>>>`→automatic dispatch).
  7. **The honest runtime matrix** — which lab runs where (Python / sim+playground / hardware) and the "upstream doesn't CI BH training; we verified it" note.
  8. **Next:** link `lfs-01`.
  Reconcile L1/core numbers against source (open item #4): state actual (L1 ≈ 1464 KB; Blackhole grid up to 13×10) with a footnote if you cite the rounded "1.5 MB".

- [ ] **Step 2: Verify build (test).** Run: `npm run build`. Expected: pass; `lfs-00` renders in the site build with no broken links.

- [ ] **Step 3: Commit.**
```bash
git add content/lessons/lfs-00-intro.md
git commit -m "docs(lfs): author Lab 0 — Pick Your Altitude & the tile (with thanks)"
```

---

### Task 4: Author Lab 1 (Tokenizer & Data)

**Files:** Modify `content/lessons/lfs-01-tokenizer.md`. Excerpts `content/templates/llm-from-scratch/reference_gpt.py`.

- [ ] **Step 1: Write the body** — rhythm: Coming-from-CUDA callout → PyTorch reference → TT-native → run → graduate box:
  1. **Coming from CUDA callout:** `cudaMemcpy` H2D/D2H ↔ `ttnn.from_torch`/`ttnn.to_torch`; no unified memory (data is on host or device, moved explicitly).
  2. **Build a BPE tokenizer from scratch** on TinyStories (train, encode, decode) — embed the tokenizer code excerpt from the template; explain merges/vocab.
  3. **Data pipeline** — batching into sequences; the `hf` CLI (not `huggingface-cli`) if downloading data.
  4. **Tokens → tiled tensors** — how a `[batch, seq]` int tensor becomes a `ttnn.Tensor` in `TILE_LAYOUT`; preview `ttnn.embedding`.
  5. **Run it** — pure Python; command to tokenize a sample and print shapes.
  6. **Graduate box** — this runs anywhere; the tensor lands on-device in the next lab.
  7. **Next:** link `lfs-02`.

- [ ] **Step 2: Drift check (test).** Confirm every code excerpt appears verbatim in `content/templates/llm-from-scratch/reference_gpt.py` (grep the snippet). Expected: match.

- [ ] **Step 3: Verify build.** Run: `npm run build`. Expected: pass.

- [ ] **Step 4: Commit.**
```bash
git add content/lessons/lfs-01-tokenizer.md
git commit -m "docs(lfs): author Lab 1 — Tokenizer & Data from scratch"
```

---

### Task 5: Author Lab 2 (Embeddings & the Residual Stream)

**Files:** Modify `content/lessons/lfs-02-embeddings.md`. May add a playground kernel + set front-matter `playground`.
- Possibly Modify: front matter of `lfs-02-embeddings.md` to add `playground: ttlang-sim` (the `eltwise_add` kernel is already bundled at `content/web/ttlang-sim-lite/kernels/eltwise_add.py`).

- [ ] **Step 1: Confirm the playground kernel is available.** Run: `ls content/web/ttlang-sim-lite/kernels/eltwise_add.py`. Expected: exists (already bundled — no new kernel needed for Lab 2).

- [ ] **Step 2: Write the body:**
  1. **Coming from CUDA callout:** shared memory → L1 SRAM per core; the three explicit reader/compute/writer threads vs. the implicit warp scheduler.
  2. **Token + positional embeddings from scratch** — PyTorch reference excerpt; explain the residual stream as the model's spine.
  3. **First inception kernel** — walk the `eltwise_add` TT-Lang `@ttl.operation` (reader `reserve`+`copy`, compute `wait`+`store`, writer `wait`+drain) as "add the position embedding to the token embedding." Embed the playground (front matter `playground: ttlang-sim`) and instruct the reader to hit **Run**.
  4. **Run it** — sim/playground; expected output.
  5. **Graduate box** — same kernel source runs in `ttlang-sim`, the browser, and on silicon.
  6. **Next:** link `lfs-03`.

- [ ] **Step 3: If front matter changed, re-sync registry.** Run: `npm run generate:lessons -- --execute --force` (only if you edited a markdown-owned field) then `npm run validate:lessons`. Expected: valid. (`playground` is not a synced field — if unchanged, skip.)

- [ ] **Step 4: Verify build + playground asset.** Run: `npm run build`. Expected: pass; `assets/ttlang-sim-lite/` present; `lfs-02` renders the playground.

- [ ] **Step 5: Commit.**
```bash
git add content/lessons/lfs-02-embeddings.md content/lesson-registry.json
git commit -m "docs(lfs): author Lab 2 — Embeddings + first TT-Lang inception kernel (playground)"
```

---

### Task 6: Author Lab 3 (Attention from Scratch — centerpiece)

**Files:** Modify `content/lessons/lfs-03-attention.md`. Excerpts `content/templates/llm-from-scratch/kernels/attention.py` + `reference_gpt.py`.
- Decision (open item #3): Lab 3 stays **functional-sim-only** by default (attention/softmax may be compiler-unsupported / press the 32-DFB ceiling). Adding a browser-playground attention kernel is an optional stretch — if pursued, add `content/web/ttlang-sim-lite/kernels/attention.py` and wire it into `scripts/check-sim-lite-drift.py`, in a separate follow-up task.

- [ ] **Step 1: Write the body:**
  1. **Coming from CUDA callout:** FlashAttention's fused memory story ↔ the reader/compute/writer tile pipeline; introduce **the agentic shortcut** (hand an agent a reader/compute/writer spec — the spec lives in the source, not the programmer's head).
  2. **PyTorch reference:** Q·Kᵀ, scale, causal mask, softmax, ·V — excerpt from `reference_gpt.py`.
  3. **TT-Lang attention/softmax inception kernel** — walk `kernels/attention.py`: transpose K, `matmul(Q,Kᵀ)`, scale/mask, softmax built from `reduce_max`→`exp`→`reduce_sum`→divide, `matmul(softmax,V)`. Explain DFB usage and the ~15-DFB footprint vs the 32 ceiling.
  4. **Run it** — functional sim; validate against the PyTorch reference (correlation/PCC check).
  5. **Honest flag callout:** sim-validated; note which softmax ops the hardware compiler still trails on (per the kernel header annotation from Task 2 Step 3).
  6. **Graduate box** — link forward to Lab 5's real on-hardware run.
  7. **Next:** link `lfs-04`.

- [ ] **Step 2: Drift check + sim run (test).** Confirm the kernel excerpt matches `kernels/attention.py`; run its sim validation. Expected: match; PCC ≈ 1.0 vs reference.

- [ ] **Step 3: Verify build.** Run: `npm run build`. Expected: pass.

- [ ] **Step 4: Commit.**
```bash
git add content/lessons/lfs-03-attention.md
git commit -m "docs(lfs): author Lab 3 — Attention from scratch (TT-Lang inception centerpiece)"
```

---

### Task 7: Author Lab 4 (The Transformer Block & the Model)

**Files:** Modify `content/lessons/lfs-04-block-and-model.md`. Excerpts `kernels/rmsnorm.py`, `kernels/matmul.py`, `reference_gpt.py`.

- [ ] **Step 1: Write the body:**
  1. **Coming from CUDA callout:** kernel fusion (fewer memory round-trips) transfers directly; TTNN fused ops parallel FlashAttention-style fusion.
  2. **MLP + RMSNorm + residuals** from scratch (PyTorch reference), then assemble one full transformer block; stack `n` blocks into the nano model.
  3. **TT-Lang RMSNorm + matmul inception kernels** — walk `kernels/rmsnorm.py` (`x*x`→`reduce_sum`→`rsqrt`→broadcast→multiply) and `kernels/matmul.py` (`@`/`+=` accumulation).
  4. **Kernels as drop-in `ttnn` ops** — the boundary is `ttnn.Tensor` in `TILE_LAYOUT` on device; show a kernel spliced into an otherwise-ttnn forward.
  5. **nano↔80M config table** (open item #2) — embed_dim / n_heads / n_blocks / vocab / seq for nano (verified baseline) and ~80M (between `nanogpt.yaml` and `gpt2s.yaml`), plus params + DRAM math and when you'd reach for multiple chips.
  6. **Run it** — sim + ttnn forward.
  7. **Next:** link `lfs-05`.

- [ ] **Step 2: Drift check (test).** Confirm rmsnorm/matmul excerpts match the template kernels. Expected: match.

- [ ] **Step 3: Verify build.** Run: `npm run build`. Expected: pass.

- [ ] **Step 4: Commit.**
```bash
git add content/lessons/lfs-04-block-and-model.md
git commit -m "docs(lfs): author Lab 4 — the block, the model, nano→80M scaling"
```

---

### Task 8: Author Lab 5 (Train It & Run for Real)

**Files:** Modify `content/lessons/lfs-05-train-and-run.md`. Excerpts `train_nano_from_scratch.py`, `BUILD_TTML.md`.

- [ ] **Step 1: Write the body:**
  1. **The training loop from scratch** — cross-entropy loss, AdamW, backprop; excerpt from `train_nano_from_scratch.py`; explain why `ttnn` alone can't train (no autograd) and `ttml` provides backward.
  2. **Build `ttml`** — condensed, linked from `BUILD_TTML.md`: `build_metal.sh --build-tt-train`, the `_ttnn.so` rebuild that fixes `std::bad_cast` (call out the TT-QuietBox 2 pre-built-image case explicitly), the `.pth`, and the extra env vars (`TT_METAL_RUNTIME_ROOT`, `TT_METAL_ARCH_NAME=blackhole`). Link `build-tt-metal`; note TT-QuietBox 2 ships without `~/tt-metal`.
  3. **Train for real on Blackhole** — the verified command and the actual loss curve captured in Task 2 Step 7 (≈4.6 → ≈3.3). Honest framing: upstream doesn't CI BH training; verified on p300c at v0.73 — pin your version, `tt-smi -r` if the board times out, let ttml close the device.
  4. **Scale to 80M** — DRAM/time/mesh math; `mesh_shape [1,1]` for single-chip p300c/p150.
  5. **Where next** — link `ct8`/`ct7` (the deeper ttml track) and `build-tt-metal`.

- [ ] **Step 2: Drift check + (if hardware available) re-run (test).** Confirm the command/loss excerpt matches Task 2's verified run. Expected: match.

- [ ] **Step 3: Verify build.** Run: `npm run build`. Expected: pass.

- [ ] **Step 4: Commit.**
```bash
git add content/lessons/lfs-05-train-and-run.md
git commit -m "docs(lfs): author Lab 5 — train for real on Blackhole with ttml (verified p300c)"
```

---

### Task 9: Changelog, README highlights & final verification

**Files:** Modify `CHANGELOG.md`, `README.md`.

- [ ] **Step 1: Add a CHANGELOG entry** under a new `0.1.0` heading (Keep a Changelog format, no line numbers):
  - `Added` — "Build an LLM from Scratch, TT-Native track (lfs-00…lfs-05): a code-first arc building a small GPT from scratch using TT-Lang for inception, with a verified from-scratch training run on Blackhole p300c and a `ttml` build recipe."

- [ ] **Step 2: Update README highlights** — add the new track to the most-recent-releases highlights (per the 2-release rule), linking CHANGELOG.

- [ ] **Step 3: Full verification (test).**
Run: `npm run build && npm run validate:lessons`
Expected: all lessons valid; site builds with the new `llm-from-scratch` category and all six lessons; links pass.

- [ ] **Step 4: Package smoke test (test).**
Run: `npm run package`
Expected: produces `TT-VSCode-Toolkit-0.1.0-dev.vsix` (dev suffix on non-main branch `llm_from_scratch`); no webpack errors.

- [ ] **Step 5: Commit.**
```bash
git add CHANGELOG.md README.md
git commit -m "docs(lfs): changelog + README highlights for 0.1.0 LLM-from-scratch track"
```

---

## Self-Review

**Spec coverage:** Acknowledgments → Task 3 Step 1. CUDA grounding (ladder/map/agentic shortcut) → Tasks 3–7 callouts. 5 labs + intro → Tasks 3–8. Two-tier model → Task 7 config table + Task 8 training. Layered runtime → per-lab "run it"/"graduate" boxes. TT-Lang inception kernels → Task 2 + Labs 2/3/4. ttml build recipe + `std::bad_cast` fix → Task 2 (`BUILD_TTML.md`) + Task 8. Verified BH training → Task 2 Step 7 + Task 8. Registry/category/version → Task 1. Drift management → per-lab drift checks + kernel header annotations. Changelog/README/version → Tasks 1 & 9. Playground reuse → Task 5. **Correction vs spec Section 4:** registration is via `lesson-registry.json`, NOT `package.json` walkthrough steps (which no longer exist) — reflected in Global Constraints + Task 1.

**Placeholder scan:** Open items #1 (Reddit URL), #2 (exact 80M config), #4 (L1/core numbers) are explicitly flagged as authoring-time lookups against named sources, not hidden TODOs. #3 (Lab 3 playground) resolved: functional-sim-only by default, stretch is a separate task. #5 resolved: 0.1.0.

**Type/name consistency:** Lesson IDs `lfs-00`…`lfs-05`, category `llm-from-scratch`, template paths, and kernel filenames are used identically across Tasks 1–9. Front-matter fields match the registry schema verified from an existing lesson.

---

## ADDENDUM — Modern Llama-3 pivot rework (2026-07-08)

The arc pivots to modern Llama-3 components (RoPE/RMSNorm/SwiGLU/GQA/SentencePiece) per the inspiration (Mini-LLM, https://github.com/Ashx098/Mini-LLM) — VERIFIED training on p300c (loss 4.69→3.23, `training_shakespeare_nanollama3_char.yaml`). See the spec ADDENDUM for component decisions. Rework tasks (executed after Task 1 scaffolding, which stands):

- **Task 2R — Templates → Llama.** `reference_gpt.py` becomes Llama-style (RoPE, RMSNorm [done], SwiGLU, GQA), pure PyTorch, `--smoke` verified. Add `kernels/rope.py` (adapt `rotary_qk_kernel` from `vendor/tt-lang/examples/test_transformer_block.py`, cite+commit; RoPE is elementwise so sim-runnable-ish — verify). `train_nano_from_scratch.py` drives the verified `train_nanogpt.py --config training_shakespeare_nanollama3_char.yaml` llama path. README/BUILD updated.
- **Task 3R — Lab 0:** update nano-config numbers to nanollama3 + add Mini-LLM thanks + Lab-5 runtime row = verified Llama loss.
- **Task 4R — Lab 1:** add SentencePiece-32K framing (Mini-LLM) + keep from-scratch byte-BPE; note hero run uses char tokenizer.
- **Task 5R — Lab 2:** RoPE replaces learned positional embeddings (new rope kernel + PyTorch RoPE ref).
- **Task 6R — Lab 3:** GQA extends MHA (KV-head sharing). (Supersedes the unreviewed MHA-only commit 337105c.)
- **Task 7 — Lab 4:** SwiGLU + RMSNorm block (SwiGLU via PyTorch ref + `ttml.ops.swiglu`; no from-scratch SwiGLU TT-Lang kernel).
- **Task 8 — Lab 5:** hero run = verified `nanollama3_char` on Blackhole (loss 4.69→3.23, ~65ms/step, 16.5 TFLOPS); 80M scale via Mini-LLM numbers.
- **Task 9 — Changelog/README/version** unchanged (0.1.0).
