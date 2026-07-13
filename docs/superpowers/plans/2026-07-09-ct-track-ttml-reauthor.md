# Custom Training Track (ct1–ct8) Re-Author — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-author the eight `ct1`–`ct8` "Custom Training" lessons from `blocked` into an honest, verified **ttml/tt-train training-workflow** track, unblocked by the now-verified `ttml` build on Blackhole p300c.

**Architecture:** Eight existing markdown lessons in `content/lessons/ct*.md` (registry-synced via `content/lesson-registry.json`). The track is repositioned to "run real training jobs with `ttml`/`tt-train`"; the `lfs` arc keeps "build by hand." `ct7`/`ct8` slim and defer to `lfs`. Install/run aligns to the verified `content/templates/llm-from-scratch/BUILD_TTML.md` recipe and `~/tt-metal/tt-train/.../train_nanogpt.py`.

**Tech Stack:** Markdown (VSCode-native), `js-yaml` registry sync, `ttml`/`tt-train` (`train_nanogpt.py` + YAML configs), the p300c Blackhole for verification.

**Spec:** `docs/superpowers/specs/2026-07-09-ct-track-ttml-reauthor-design.md` (approved).

## Global Constraints

- **Center on `tt-train`/`ttml`.** Do NOT teach `tt-blacksmith` here — it's a SEPARATE active repo of training recipes on the **TT-Forge/TT-XLA** compiler stack. Where the old lessons conflate them, disentangle: cross-link tt-blacksmith as "recipes on the TT-Forge stack," teach tt-train/ttml.
- **Verified install recipe:** `content/templates/llm-from-scratch/BUILD_TTML.md` — `build_metal.sh --build-tt-train`; the **`_ttnn.so` rebuild fixing `std::bad_cast`** (TT-QuietBox 2 / pre-built-image pitfall); `.pth` wiring; env `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`, `TT_METAL_ARCH_NAME=blackhole`, `TT_LOGGER_LEVEL=FATAL`.
- **Verified run path:** `~/tt-metal/tt-train/sources/examples/nano_gpt/train_nanogpt.py --config <yaml> [--max_steps N --data_path <text>]`; `model_type: gpt2|llama`; char configs need no external tokenizer. Board: `tt-smi -r` clears first-open ethernet timeout; let ttml close the device.
- **Honest status:** flip `blocked → draft`; `validated` + `validatedOn: p300c` ONLY for ct4 and ct8 (actually run). ct5 multi-device stays `draft` (NOT verifiable on single-chip p300c). No claim beyond what was run.
- **Name-and-link, no raw IDs in prose** (owner rule): reference other lessons by human name + `command:tenstorrent.showLesson?["<id>"]`, never bare `ct4`/`lfs-04`. Cross-link map below.
- **WH/BH (CLAUDE.md):** `hf` CLI not `huggingface-cli`; never `DispatchCoreAxis.ROW`; don't assume `~/tt-metal` (link Build TTMetalium); `p300c` in hardware lists; `pip install --upgrade pip setuptools wheel` before dev installs; multi-device uses `CreateDevices`/`CloseDevices`.
- **New/renamed commands namespaced** `tenstorrent.<feature>.<action>` (ttsim convention).
- **Registry sync:** front matter is source of truth for `id,title,description,category,tags,supportedHardware,status,validatedOn,estimatedMinutes`; run `npm run generate:lessons -- --execute --force` after front-matter edits; hand-maintain `order/nav/completionEvents/markdownFile/minTTMetalVersion`.
- **Gate every task:** `npm run build` must pass (validate:lessons, validate:command-uris, link tests, site build). Version bump PATCH per change. `<sup>` trademark first-use marks (TT-NN™, TT-Metalium™, Blackhole®, Wormhole™, TT-Forge™; TT-Lang unmarked).
- **Don't touch verbatim code blocks** quoted from templates (drift).

**Cross-link name→id map:** Understanding Custom Training=`ct1-understanding-training`; Dataset Fundamentals=`ct2-dataset-fundamentals`; Configuration Patterns=`ct3-configuration-patterns`; Fine-tuning Basics=`ct4-finetuning-basics`; Multi-Device Training=`ct5-multi-device-training`; Experiment Tracking=`ct6-experiment-tracking`; Model Architecture Basics=`ct7-architecture-basics`; Training from Scratch=`ct8-training-from-scratch`; Embeddings & the Residual Stream=`lfs-02-embeddings`; Attention from Scratch=`lfs-03-attention`; The Transformer Block & the Model=`lfs-04-block-and-model`; Train It & Run for Real=`lfs-05-train-and-run`; Build TTMetalium from Source=`build-tt-metal`.

---

### Task 1: Track-wide unblock, prerequisite, and `installTtTrain` alignment

Lift the false block across all 8 lessons and fix the shared install command. This is the foundation every later task builds on.

**Files:**
- Modify: front matter of all `content/lessons/ct{1..8}-*.md` (status/blockReason/validatedOn)
- Modify: `content/lesson-registry.json` (via generator + manual nav fields)
- Modify: `src/extension.ts` (the `installTtTrain` handler) and `package.json` (version)

- [ ] **Step 1: Flip status + replace blockReason in all 8 front matters.** For each `ct*.md`: set `status: draft` (leave ct4/ct8 as `draft` for now — they become `validated` in their own tasks after the hardware run). Remove the `blockReason` field (the premise is false) OR replace with a one-line `note`. Keep `validatedOn: [n150]` for now (ct4/ct8 add p300c later). Do NOT change ids/titles/categories.

- [ ] **Step 2: Sync registry.** Run `npm run generate:lessons -- --execute --force`; then confirm the 8 entries show `status: draft` and no stale `blockReason`.

- [ ] **Step 3: Update the `installTtTrain` command** in `src/extension.ts` to match the verified `BUILD_TTML.md` recipe. Read the current handler first. It must: run `build_metal.sh --build-tt-train`, then the **`_ttnn.so` rebuild + copy** (the `std::bad_cast` fix), then the `.pth` wiring — with the env vars set and a note that TT-QuietBox 2 ships without `~/tt-metal` (link Build TTMetalium). Keep the existing command id (`tenstorrent.installTtTrain`) unless renaming; if renamed, namespace it and update all lesson links.

- [ ] **Step 4: Bump version** in `package.json` (PATCH).

- [ ] **Step 5: Verify + commit.** Run `npm run build` (expect green; 8 ct lessons now `draft`). Commit:
```bash
git add content/lessons/ct*.md content/lesson-registry.json src/extension.ts package.json
git commit -m "feat(ct): unblock ct1-ct8 (ttml verified) + align installTtTrain to BUILD_TTML recipe"
```

---

### Task 2: Re-author ct1 — Understanding Custom Training

**Files:** Modify `content/lessons/ct1-understanding-training.md` (do not change front matter beyond what Task 1 set).

- [ ] **Step 1: Re-author the body.** Read the current lesson first. Reframe around the verified **tt-train/ttml** stack. Concretely:
  - Fine-tune vs train-from-scratch; where each fits.
  - **Disentangle the frameworks honestly:** `tt-train`/`ttml` = the autograd training framework inside tt-metal (this track uses it); `tt-blacksmith` = a SEPARATE repo of optimized training recipes on the **TT-Forge<sup>™</sup>/TT-XLA** compiler stack (cross-link `https://github.com/tenstorrent/tt-blacksmith`, describe accurately, don't teach it here); PyTorch/GPU = the familiar baseline. Remove any text implying tt-blacksmith is a config layer over tt-train.
  - Point readers who want to build every component by hand to the `lfs` arc (named links to Embeddings & the Residual Stream / Attention from Scratch / The Transformer Block & the Model / Train It & Run for Real).
  - "Next" → Dataset Fundamentals (named link).
- [ ] **Step 2: Verify + commit.** `npm run build` green.
```bash
git add content/lessons/ct1-understanding-training.md && git commit -m "docs(ct): re-author ct1 — tt-train/ttml stack, disentangle tt-blacksmith (TT-Forge)"
```

---

### Task 3: Re-author ct2 — Dataset Fundamentals

**Files:** Modify `content/lessons/ct2-dataset-fundamentals.md`.

- [ ] **Step 1: Re-author.** Read current first. Keep the solid dataset content (JSONL, HF datasets, tokenization, validation) but align the "how data flows into training" to the **real tt-train path**: `train_nanogpt.py --data_path <text>` (char-level: plain text; BPE/HF: tokenized). Use `hf` CLI for any downloads (not `huggingface-cli`). Remove tt-blacksmith-specific data plumbing if present. Cross-link Tokenizer & Data (`lfs-01-tokenizer`) for the from-scratch BPE build. "Next" → Configuration Patterns.
- [ ] **Step 2: Verify data-prep steps** you can run offline (e.g., a tokenization/validation snippet) on CPU; note any not runnable. `npm run build` green.
- [ ] **Step 3: Commit.**
```bash
git add content/lessons/ct2-dataset-fundamentals.md && git commit -m "docs(ct): re-author ct2 — datasets aligned to tt-train data flow"
```

---

### Task 4: Re-author ct3 — Configuration Patterns

**Files:** Modify `content/lessons/ct3-configuration-patterns.md`.

- [ ] **Step 1: Verify the real configs first.** Read `~/tt-metal/tt-train/configs/model_configs/nanollama3_char.yaml` + `nanogpt.yaml` and `~/tt-metal/tt-train/configs/training_configs/training_shakespeare_*.yaml`. Note exact fields (model_type, num_heads/num_groups, embedding_dim, num_blocks, theta, vocab, max_sequence_length; training: batch_size, max_steps, optimizer AdamW lr/betas/weight_decay, model_save_interval, data_path, mesh_shape).
- [ ] **Step 2: Re-author** around those ACTUAL YAML configs (quote real fields). Cover hyperparameters + effects, single vs multi-chip (`mesh_shape: [1,1]` for single p300c/p150), checkpointing (`model_save_interval`/`model_save_path`), logging. Drop the tt-blacksmith config abstraction. Cross-link Fine-tuning Basics (where the config is run). "Next" → Fine-tuning Basics.
- [ ] **Step 3: Verify** a config parses/loads (dry). `npm run build` green. Commit:
```bash
git add content/lessons/ct3-configuration-patterns.md && git commit -m "docs(ct): re-author ct3 — real tt-train YAML configs (drop tt-blacksmith abstraction)"
```

---

### Task 5: Re-author ct4 — Fine-tuning Basics (HERO, hardware-verified)

**Files:** Modify `content/lessons/ct4-finetuning-basics.md` (+ front matter → `validated`, `validatedOn: [n150, p300c]`, keep `minTTMetalVersion`); registry.

- [ ] **Step 1: Run the training on the p300c** (device must be free; `tt-smi -r` if it times out). Build/confirm ttml (already built), then run a short job capturing a real loss curve + a checkpoint, e.g.:
```bash
export TT_METAL_HOME=/home/ttuser/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal TT_METAL_ARCH_NAME=blackhole TT_LOGGER_LEVEL=FATAL
cd ~/tt-metal/tt-train/sources/examples/nano_gpt
python train_nanogpt.py --config training_shakespeare_nanogpt.yaml --max_steps 50 --data_path <shakespeare.txt>
```
Capture the actual loss curve + timings. If multi-hundred-step convergence is impractical in-lesson, capture ~50 steps and frame honestly.
- [ ] **Step 2: Re-author** the lesson around: install ttml (link the verified recipe + Build TTMetalium; the `_ttnn.so`/`std::bad_cast` note), run `train_nanogpt.py`, read the loss curve, save/load checkpoints, troubleshoot. Use the REAL captured numbers. Keep the "models learn in stages" teaching if it still holds. Cross-link Configuration Patterns and Train It & Run for Real. Update front matter status→`validated`, add `p300c` to `validatedOn`.
- [ ] **Step 3: Sync registry** (`generate:lessons`), `npm run build` green. Commit:
```bash
git add content/lessons/ct4-finetuning-basics.md content/lesson-registry.json && git commit -m "docs(ct): re-author ct4 — verified tt-train run on p300c, real loss curve (validated)"
```

---

### Task 6: Re-author ct5 — Multi-Device Training (honest, unverified here)

**Files:** Modify `content/lessons/ct5-multi-device-training.md`.

- [ ] **Step 1: Re-author** the DDP/mesh content around tt-train's multi-device path (`mesh_shape`, `CreateDevices`/`CloseDevices`, DDP config). **Be explicit and honest:** this was NOT verified on this single-chip p300c (TT-QuietBox 2 = 4 *independent* p300c chips, not a mesh); it's the documented pattern for n300/T3000/Galaxy. Keep `status: draft`, hardware list n300/t3k/galaxy (+ note single-chip = no DDP). Cross-link Configuration Patterns.
- [ ] **Step 2:** `npm run build` green. Commit:
```bash
git add content/lessons/ct5-multi-device-training.md && git commit -m "docs(ct): re-author ct5 — multi-device pattern, honest unverified-on-single-chip note"
```

---

### Task 7: Re-author ct6 — Experiment Tracking

**Files:** Modify `content/lessons/ct6-experiment-tracking.md`.

- [ ] **Step 1: Re-author** around tracking real tt-train runs: file-based logs + Weights & Biases (`-w 0` / `wandb offline` to opt out, per tt-train README), comparing hyperparameter runs, visualizing loss curves. Align commands to `train_nanogpt.py` output. Cross-link Fine-tuning Basics.
- [ ] **Step 2: Verify** wandb-offline flow works (no network). `npm run build` green. Commit:
```bash
git add content/lessons/ct6-experiment-tracking.md && git commit -m "docs(ct): re-author ct6 — experiment tracking around real tt-train runs"
```

---

### Task 8: Slim ct7 — Model Architecture Basics (defer to lfs)

**Files:** Modify `content/lessons/ct7-architecture-basics.md`.

- [ ] **Step 1: Slim to a short conceptual overview.** Keep a brief tour of transformer components (tokenization, embeddings, attention, FFN, norm) at a high level and WHY they matter for training. **Remove the deep build/duplication** — defer to the `lfs` arc with prominent named links: Embeddings & the Residual Stream, Attention from Scratch, The Transformer Block & the Model. This lesson should now be substantially shorter. Cross-link Training from Scratch as the next step. Frame it as "concepts you need before configuring a training job; build them by hand in the from-scratch arc."
- [ ] **Step 2:** `npm run build` green. Commit:
```bash
git add content/lessons/ct7-architecture-basics.md && git commit -m "docs(ct): slim ct7 — concise architecture overview, defer deep build to lfs arc"
```

---

### Task 9: Reframe ct8 — Training from Scratch (job-centric, hardware-verified)

**Files:** Modify `content/lessons/ct8-training-from-scratch.md` (+ front matter → `validated`, `validatedOn: [n150, p300c]`, keep `minTTMetalVersion`); registry.

- [ ] **Step 1: Run the from-scratch job on the p300c** (reuse the verified nanollama3 run or re-run):
```bash
export TT_METAL_HOME=/home/ttuser/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal TT_METAL_ARCH_NAME=blackhole TT_LOGGER_LEVEL=FATAL
cd ~/tt-metal/tt-train/sources/examples/nano_gpt
python train_nanogpt.py --config training_shakespeare_nanollama3_char.yaml --max_steps 20 --data_path <shakespeare.txt>
```
Verified prior result to reuse if a fresh run isn't needed: loss 4.69→3.23 over 20 steps, ~65 ms/step, 16.5 TFLOPS, exit 0.
- [ ] **Step 2: Reframe** the lesson from "hand-build an 11M nano-trickster" to **"configure, launch, monitor & scale a from-scratch training job with ttml."** Featured config: `nanollama3_char` (modern, aligns with `lfs`). Cover: pick a model config, launch `train_nanogpt.py`, watch loss drop (real numbers), checkpoint, and scale (steps/size/data; `mesh_shape` note). **Defer the by-hand architecture to `lfs`** (named links to the full arc, esp. Train It & Run for Real which builds the same nano Llama by hand). Update front matter status→`validated`, add `p300c`. Remove the stale "11M nano-trickster" framing.
- [ ] **Step 3: Sync registry**, `npm run build` green. Commit:
```bash
git add content/lessons/ct8-training-from-scratch.md content/lesson-registry.json && git commit -m "docs(ct): reframe ct8 — job-centric from-scratch training, verified on p300c (validated)"
```

---

### Task 10: Changelog, README, final verification

**Files:** Modify `CHANGELOG.md`, `README.md`, `package.json` (final version).

- [ ] **Step 1: CHANGELOG** — new version section: `Changed` — "Re-authored the Custom Training track (ct1–ct8) as a verified `ttml`/`tt-train` training-workflow track: unblocked (ttml build verified on Blackhole p300c), disentangled from `tt-blacksmith` (TT-Forge recipe stack), ct4/ct8 validated on p300c with real loss curves, ct7/ct8 deferring the by-hand build to the from-scratch arc." No line numbers.
- [ ] **Step 2: README** highlights — reflect the ct track update in the recent-releases section (2-release rule).
- [ ] **Step 3: Full verification.** `npm run build && npm run validate:lessons` green; confirm ct4/ct8 show `validated`, others `draft`, none `blocked`. `npm run package` smoke (expect `*-dev.vsix` on this branch).
- [ ] **Step 4: Commit.**
```bash
git add CHANGELOG.md README.md package.json && git commit -m "docs(ct): changelog + README for ct-track ttml re-author"
```

---

## Self-Review

**Spec coverage:** Identity/repositioning → all tasks. Unblock + prerequisite + installTtTrain → Task 1. Disentangle tt-blacksmith → Task 2 (ct1) + Global Constraints. Real configs → Task 4 (ct3). ct4/ct8 hardware-verified + validated → Tasks 5, 9. ct5 honest-unverified → Task 6. ct7/ct8 slim/defer to lfs → Tasks 8, 9. Cross-linking + name-and-link → Global Constraints + each task. Registry/version/changelog → Tasks 1, 5, 9, 10.

**Placeholder scan:** Config field lists and loss numbers are pulled from verified sources (spec + prior runs); `<shakespeare.txt>` is the one path the implementer fills from the scratchpad copy — flagged, not hidden. Open items (exact config filenames, installTtTrain current impl, ct8 featured config, tt-blacksmith one-liner) are authoring-time verifications against named sources.

**Consistency:** Lesson ids/titles and the cross-link map are used identically across tasks; status values (draft everywhere except validated ct4/ct8) are consistent between spec and plan; hardware-verify picks (ct4, ct8) match the spec's HW plan.
