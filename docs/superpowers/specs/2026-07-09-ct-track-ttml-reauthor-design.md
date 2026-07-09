# Custom Training Track (ct1–ct8) Re-Author — Design Spec

**Date:** 2026-07-09
**Branch:** `llm_from_scratch` (continues the training-lessons work; `lfs` arc not yet merged)
**Status:** Approved (user pre-approved spec + subagent-driven build)

## Goal

Re-author the eight `ct1`–`ct8` "Custom Training" lessons — currently all `status: blocked`
on the false premise that *"ttml is not available as a standalone package"* — into an honest,
verified **ttml/tt-train training-workflow track**. We proved `ttml` builds and trains on
Blackhole p300c (see `reference_ttml_build_blackhole` / the `lfs` arc), so the block is lifted.

## Load-bearing decisions (from brainstorming)

1. **Identity:** `ct` = *"run real training jobs on Tenstorrent with `ttml`/`tt-train`"* — the
   practical workflow track (build/install ttml, datasets, configs, fine-tune, multi-device,
   experiment tracking, checkpoints). The **`lfs` arc owns "build every component by hand,
   TT-native."** No duplicated from-scratch story. `ct7`/`ct8` slim and defer to `lfs`.
2. **Depth:** deep re-author of all 8, re-verifying runnable steps on hardware where possible.
3. **Branch:** continue on `llm_from_scratch`.

## Critical research finding — two distinct stacks (do NOT conflate)

- **`tt-train` / `ttml`** — the C++/nanobind **autograd training framework inside tt-metal**.
  This is what `ct4`/`ct8` actually run (`train_nanogpt.py` in `~/tt-metal/tt-train`), and what
  we VERIFIED on p300c (GPT-2 nanogpt loss 4.59→3.28; Llama nanollama3 loss 4.69→3.23). **This
  track centers on tt-train/ttml.**
- **`tt-blacksmith`** — a SEPARATE, currently-active repo (github.com/tenstorrent/tt-blacksmith,
  pushed 2026-07-09): *"optimized training recipes … powered by the **TT-Forge compiler stack**"*
  (recent commits are TT-XLA uplifts). It is NOT tt-train and NOT a config layer over tt-train.
- **The bug to fix:** the current `ct1`/`ct3` conflate the two — e.g. `ct3` teaches a
  "tt-blacksmith YAML config pattern" while `ct4`/`ct8` run tt-train configs. The re-author must
  **disentangle**: `ct1` honestly distinguishes tt-train/ttml (this track) vs tt-blacksmith
  (TT-Forge recipe collection, cross-link, don't teach here) vs PyTorch; `ct3` teaches the
  **actual** tt-train/`train_nanogpt.py` YAML configs we verified (e.g. `nanollama3*.yaml`,
  `training_shakespeare_*.yaml`), not a tt-blacksmith abstraction.

## Verified reality to align every lesson to

- Install/build: the verified recipe in `content/templates/llm-from-scratch/BUILD_TTML.md`
  (`build_metal.sh --build-tt-train`; the **`_ttnn.so` rebuild fixing `std::bad_cast`** — the
  TT-QuietBox 2 / pre-built-image pitfall; the `.pth` wiring; env vars `TT_METAL_HOME`,
  `TT_METAL_RUNTIME_ROOT`, `TT_METAL_ARCH_NAME=blackhole`). The existing `installTtTrain`
  command must be updated to match this (it currently just navigates + builds; it misses the
  `_ttnn.so` rebuild).
- Runnable path: `~/tt-metal/tt-train/sources/examples/nano_gpt/train_nanogpt.py` with a YAML
  config; `model_type: gpt2 | llama`. Char-tokenizer configs need no external tokenizer.
- Board hygiene: `tt-smi -r` clears the first-open ethernet-core timeout; let ttml close the
  device cleanly.
- WH/BH rules (CLAUDE.md): `hf` CLI not `huggingface-cli`; never `DispatchCoreAxis.ROW`; don't
  assume `~/tt-metal` exists (link **Build TTMetalium**); `p300c` in hardware lists where
  applicable. New/renamed commands namespaced `tenstorrent.<feature>.<action>`.

## Per-lesson disposition

| Lesson | Direction | HW-verify (p300c) | Target status |
|---|---|---|---|
| **ct1** Understanding Custom Training | Reframe around tt-train/ttml. Honestly distinguish tt-train/ttml (this track) vs **tt-blacksmith = TT-Forge recipe collection** (cross-link, don't teach) vs PyTorch. Point to `lfs` for build-by-hand. | conceptual | draft |
| **ct2** Dataset Fundamentals | Datasets for ttml training — JSONL/HF/tokenization, the real data flow into `train_nanogpt.py` (`--data_path`, char vs BPE). Fix any tt-blacksmith-isms. | data-prep steps | draft |
| **ct3** Configuration Patterns | Rebuild around the **actual** tt-train YAML configs we ran (`nanollama3*.yaml`, `training_shakespeare_*.yaml`) — model_type, heads/groups, RoPE θ, optimizer/AdamW, steps, mesh_shape, checkpointing. Drop the tt-blacksmith config abstraction. | config-parse | draft |
| **ct4** Fine-tuning Basics | **Hero runnable.** Install ttml via the verified recipe (incl. `_ttnn.so` fix); run `train_nanogpt.py` char-level; real loss curves + checkpoints. | **Yes** | **validated (p300c)** |
| **ct5** Multi-Device Training | DDP/mesh pattern. **Honest:** not verifiable on single-chip p300c (TT-QuietBox 2 = 4 independent chips, not a mesh). Document the pattern + `CreateDevices`/`CloseDevices`; keep hardware list n300/t3k/galaxy. | no (can't) | draft |
| **ct6** Experiment Tracking | wandb + file tracking around real ttml runs (`-w 0`/`wandb offline` noted); compare runs. | wandb-offline | draft |
| **ct7** Architecture Basics | **Slim** to a short conceptual overview; defer the deep build to `lfs` (link **Embeddings & the Residual Stream**, **Attention from Scratch**, **The Transformer Block & the Model**). Remove duplication. | conceptual | draft |
| **ct8** Training from Scratch | **Reframe** from "hand-build an 11M nano-trickster" to "**configure, launch, monitor & scale** a from-scratch job with ttml" (nanogpt/nanollama configs). Defer by-hand architecture to `lfs`; cross-link heavily. Run the job for real. | **Yes** | **validated (p300c)** |

## Track-wide changes

- Replace the `blocked` status + *"no ttml package"* `blockReason* everywhere with a **verified
  prerequisite**: "build ttml from source — here's the recipe," linking `BUILD_TTML.md`,
  **Build TTMetalium from Source**, and **Train It & Run for Real**. Flip `blocked → draft`
  (→ `validated` only for ct4/ct8). Update `validatedOn` honestly (add `p300c` only where run).
- Update the `tenstorrent.installTtTrain` command to match the verified `BUILD_TTML.md` recipe
  (the `_ttnn.so` rebuild). If a new scaffold/run command is added, namespace it
  `tenstorrent.<feature>.<action>`.
- Cross-link the two tracks: `ct` (workflow) ↔ `lfs` (build-by-hand). No duplicated architecture
  or from-scratch teaching.
- Registry sync (front matter → `generate:lessons`), `npm run build` green each task, version
  bump (PATCH per change), CHANGELOG entry.

## Hardware verification plan

- **ct4** + **ct8**: run `train_nanogpt.py` (fine-tune char-level; from-scratch nanogpt/nanollama)
  on the p300c, capture real loss curves, mark `validatedOn: p300c` with evidence.
- **ct2/ct3/ct6**: verify the non-hardware runnable bits (data prep, config parse, wandb offline).
- **ct5**: NOT verifiable here — honest "pattern documented, multi-device unverified on p300c."

## Non-goals (YAGNI)

- Not teaching tt-blacksmith/TT-Forge training here (cross-link only; it's a different stack).
- Not re-building the `lfs` arc (ct7/ct8 defer to it, don't duplicate).
- Not claiming multi-device (ct5) validation without a multi-chip run.
- Not a full renumber/merge of ct+lfs (rejected in brainstorming).

## Open items to resolve while authoring

1. Confirm exact current tt-train YAML config filenames/fields against `~/tt-metal/tt-train/configs`
   before quoting them in ct3.
2. Verify `installTtTrain`'s current implementation and what it must add for the `_ttnn.so` fix.
3. Decide ct8's featured config (nanogpt char vs nanollama3 char) — prefer nanollama3 to align
   with the `lfs` modern stack, if it trains cleanly in the ct8 context.
4. Confirm tt-blacksmith's accurate one-line description for ct1's cross-link (TT-Forge recipes).
