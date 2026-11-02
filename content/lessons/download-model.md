---
id: download-model
title: Find and Pull a Model
description: >-
  Use tt model to see which models run on your hardware, read what each one
  needs, search the community bundles on Hugging Face, log in, and pull weights
  ahead of time.
category: serve
tags:
  - inference
  - model
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
  - n150
  - p300c
estimatedMinutes: 10
order: 1
---

# Find and Pull a Model

`tt` knows which models run on your hardware, what each one needs, and where its weights
come from. This lesson is about choosing; the next one serves what you chose.

## What Runs Here

```bash
tt model list
```

[📋 Models for this hardware](command:tenstorrent.model.list)

`tt` detects your board (`n150` on the cloud pod, `p300x2` on a TT-QuietBox 2) and lists
the models that have a serving profile for it, with the engine and where the weights are.
`tt model list --catalog` shows only the released catalog, the models Tenstorrent ships and
tests through tt-inference-server. `--type llm|image|audio|video` narrows by kind, and
`--all` shows every model on every device.

```bash
tt model list --catalog
```

[📚 Released catalog](command:tenstorrent.model.listCatalog)

## What a Model Needs

```bash
tt model info Llama-3.1-8B-Instruct
```

[ℹ️ Model info](command:tenstorrent.model.info)

```
│ hf_repo       │ meta-llama/Llama-3.1-8B-Instruct                             │
│ type          │ llm                                                          │
│ engines       │ vLLM, forge                                                  │
│ via           │ inference-server, studio                                     │
│ servable      │ yes — `tt serve Llama-3.1-8B-Instruct` (`--studio` deploys   │
│               │ it with TT-Studio)                                           │
│ parameters    │ 8B                                                           │
│ min disk      │ 36 GB                                                        │
│ min ram       │ 20 GB                                                        │
│ cached        │ yes (15.0 GB)                                                │
│ on n150       │ vLLM — COMPLETE, max context 65536                           │
│ on n300       │ vLLM — COMPLETE, max context 131072                          │
│ on t3k        │ vLLM — COMPLETE, max context 131072                          │
│ ...           │                                                              │
```

`cached` says the weights are already in the Hugging Face cache on this machine (captured
on a TT-QuietBox 2 that had pulled them earlier).

Validation status goes `EXPERIMENTAL`, `FUNCTIONAL`, `COMPLETE`. `COMPLETE` is fully
validated; `FUNCTIONAL` runs correctly and is not fully tuned. The status is per board, so
read the row for yours. `min disk` and `min ram` are in the same table.

### Starting points

| Hardware | Model | Status | Why |
|---|---|---|---|
| n150 (cloud pod) | `Llama-3.1-8B-Instruct` | COMPLETE | Single chip, 64K context, calls tools, so the Connect Software lessons work. Gated: accept Meta's license first. |
| TT-QuietBox 2 | `Qwen3-32B` | FUNCTIONAL | Ships on the box with weights and compiled cache on disk; the first serve downloads nothing. |
| TT-QuietBox 2 | `Qwen3.8-27B`, `gemma-4-31B-it` | COMPLETE | Fully validated, 262K context. |
| any | `whisper-large-v3` | COMPLETE | Speech to text (`tt model list --type audio`). |

Gated models (Llama, Gemma) need you to accept the license on the model's Hugging Face
page first. The `hf_repo` line in `tt model info` names the page.

## Log In to Hugging Face

Serving needs a Hugging Face token, even for models that are free to download, because
`tt-inference-server` checks it before it starts. Log in once:

```bash
tt model login
```

[🔑 Log in to Hugging Face](command:tenstorrent.model.login)

Choose **Log in with your browser** (it shows a URL and a short code) or paste a read token
from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). `tt` finds
the token again on every later run, and `tt serve --dry-run` tells you where it came from
without printing it.

## Pull Weights Ahead of Time

`tt serve` fetches weights on first use. To download now, while you read on:

```bash
tt model pull Llama-3.1-8B-Instruct
```

[⬇️ Pull Llama-3.1-8B-Instruct](command:tenstorrent.model.pull)

The weights land in the Hugging Face cache and `tt` reports where. On the cloud pod that
cache is `~/models` (`HF_HOME`), which `hf`, the vLLM plugin and `tt` all share.

## Community Models

The catalog is what Tenstorrent ships and tests. Beyond it, anyone can publish a **model
bundle** to the Hugging Face Hub with
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager): a bring-up packaged
so that someone else can serve it with the same commands. A bundle carries the engine the
author built and a pointer to the weights, which download into your own cache under your
own token.

> **⚠️ Community models are not reviewed by Tenstorrent. Serve at your own risk.** Bundles
> are published by their owners. Tenstorrent does not test or maintain them, and a bundle
> can run code and download weights you have not looked at. Read `tt model info` and the
> author's page before you pull one, and do not point one at data you would not trust to
> a stranger's software. The compatibility check only confirms the chip architecture; it says
> nothing about whether the model is safe.

To browse instead of search, the
[tt-awesome model browser](https://docs.tenstorrent.com/tt-awesome/models/) lists every
catalog model and community bundle by collection, family, kind and hardware, with the
commands to install and run each. It marks community bundles with the same warning.

```bash
tt model search qwen3 --limit 5     # find published bundles on the Hub
tt model list --community           # community bundles that run on this machine
tt model info raahemnabeel/qwen3-coder-next    # manifest and compatibility verdict
```

[🔎 Search the Hub](command:tenstorrent.model.search)

A real search on a TT-QuietBox 2 (2026-10-28):

```
          tt-model bundles on the Hub — every published bundle matching 'qwen3'
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ name                                ┃ visibility ┃ downloads ┃ updated    ┃ installed ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ raahemnabeel/qwen3-5-122b-a10b      │ public     │ 0         │ 2026-10-28 │ —         │
│ raahemnabeel/qwen3-coder-next       │ public     │ 0         │ 2026-10-28 │ ✓         │
│ episod/qwen3.8-27b-dflash2-p300     │ public     │ 0         │ 2026-10-28 │ ✓         │
└─────────────────────────────────────┴────────────┴───────────┴────────────┴───────────┘
```

Names are `namespace/name`. A name with a slash that is not a catalog model is a bundle,
and `tt serve` routes it to tt-model-manager. Compatibility is **checked, not
guaranteed**: pulling or serving a bundle checks that its binaries were built for your chip
architecture and warns if your box has fewer chips than the profile needs.

### "I want to..." and what to search

These are all community bundles, so the warning above applies to every row.
`tt model search` matches the text of a bundle's name, so search for a model family or a
task word that models usually carry in their name. These all returned bundles on
2026-10-28; bundles are published often, so your results will differ.

| I want to... | Try | What came back |
|---|---|---|
| Write code with a local model | `tt model search coder` | `raahemnabeel/qwen3-coder-next`, `raahemnabeel/qwen3-coder-30b-a3b` |
| Try the newest Qwen on my box | `tt model search qwen3.8` | `episod/qwen3.8-27b-dflash2-p300`, `episod/qwen3.8-27b-pi-p300` |
| Chat with a Gemma or Llama model | `tt model search gemma`, `tt model search llama` | `stisiTT/gemma-4-12b-it-p150`, `stisiTT/llama-3.2-3b-blackhole-v51` |
| Generate or edit images | `tt model search image`, `tt model search flux` | `sunandbean/qwen-image-edit-2511-p100a`, `stisiTT/flux2-dev-qb2` |
| Turn text into speech | `tt model search tts` | `tt-hous/qwen3-tts-1.7b-base`, `tt-hous/qwen3-tts-1.7b-voicedesign` |
| Make music | `tt model search music` | `jashansinghTT/minimax-music3-blackhole` |
| Embed text for search | `tt model search embed` | `tt-hous/embeddinggemma-2` |
| Detect objects or segment images | `tt model search yolo`, `tt model search sam` | `changh95/yolox-p150`, `changh95/sam3-p150` |
| Estimate depth or reconstruct 3D | `tt model search depth`, `tt model search mast3r` | `changh95/depth-anything-3-p150`, `changh95/mast3r-p150` |
| Work on self-driving perception | `tt model search bev` | `changh95/bevdet-p150`, `changh95/bevformer-p150`, `changh95/bevfusion-p150` |
| Run a robot policy | `tt model search pi05` | `changh95/pi05-base-p300x2` |

Some searches found nothing (`whisper`, `video`, `robot`, `pose`, `rerank`). Speech-to-text
and video models such as `whisper-large-v3` and `Wan2.2-T2V-A14B-Diffusers` are **catalog**
models, so look for them with `tt model list --type audio` or `--type video`. The other
terms are just not in any bundle's name yet. Empty results mean nobody has published one.

The suffix on a bundle name is the hardware it was built for: `-n150` or `-p150` is one
chip, `-p300` is one board of two chips, `-p300x2` is a whole TT-QuietBox 2.
`tt model list --community --hw <board>` shows only the bundles that fit your machine.

## Downloading by Hand

`hf download` is for a repository you will use outside `tt`, such as the TTNN demos in
Build on the Stack. The cloud pod has `hf` installed; elsewhere, `uv tool install
huggingface_hub`.

```bash
mkdir -p ~/models
hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B          # 1.5 GB, ungated
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/models/Llama-3.1-8B-Instruct   # 16 GB, gated
```

Check space first: `df -h ~/models`. A partial download leaves the directory incomplete;
re-run the same command to resume.

**Next:** [Serve a Model →](command:tenstorrent.lesson.show?["interactive-chat"])
