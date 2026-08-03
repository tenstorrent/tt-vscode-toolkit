# vLLM / TT-Inference-Server Content Refresh — Design

**Date:** 2026-08-03
**Branch:** `updates/early-aug-2026`
**Author:** Taylor Singletary (with Claude Code)

## Problem

Tenstorrent's vLLM integration was re-architected from a whole-repo fork build into a
proper **vLLM platform plugin** (`vllm-tt-plugin`). Our lessons, pages, templates, and
extension commands still teach the pre-plugin model. Two of our documented paths are now
actively broken, and one real capability (4-chip TT-QuietBox 2 serving) is documented as
impossible when the plugin in fact supports it.

## Ground Truth (verified from source, 2026-08-03)

Authoritative sources:

- `github.com/tenstorrent/vllm`, branch **`dev`**, path `plugins/vllm-tt-plugin/README.md`
- `github.com/tenstorrent/tt-metal` `README.md` (origin/main), which now links to the
  plugin README as the official vLLM install doc
- `tenstorrent/tt-inference-server` origin/main @ `3ea0618bc`, latest tag **v0.19.0**

### The architecture change

TT is now a vLLM **platform plugin**. It registers two entry points:

| Entry point group | Name | Target |
|---|---|---|
| `vllm.platform_plugins` | `tt` | `vllm_tt_plugin.entrypoints:platform_plugin` |
| `vllm.general_plugins` | `tt_model_registry` | `vllm_tt_plugin.entrypoints:register` |

`platform_plugin()` returns `TTPlatform` **only when `ttnn` is importable**, so the TT
platform is selected automatically and never hijacks a plain vLLM environment.

### ⚠️ The task premise needs correcting

This work was requested on the basis that Tenstorrent is "now an official, listed vLLM
plugin." **That is not accurate as of 2026-08-03, and we must not publish it.** Verified:

| Check | Result | How verified |
|---|---|---|
| Listed in vLLM upstream docs / README hardware-plugin list | **No** | Absent from `docs/design/plugin_system.md`, `docs/getting_started/installation/.nav.yml`, `docs/.nav.yml`, `vllm/platforms/` (no `tt.py`), `docs/models/hardware_supported_models/` |
| On PyPI | **No** | `pypi.org/pypi/{vllm-tt-plugin,vllm-tenstorrent,tt-vllm-plugin}/json` all return 404 |
| Tagged release | **No** | GitHub tags API returns `[]` |

Upstream's README hardware-plugin sentence names Google TPU, Intel Gaudi, IBM Spyre, Huawei
Ascend, Rebellions NPU, Apple Silicon, MetaX GPU — **not Tenstorrent**.

What *is* true: TT is a **technically conformant out-of-tree platform plugin** using vLLM's
standard entry-point mechanism, and the whole-fork build is no longer structurally required.
That is a real and significant change. It is not the same as being listed upstream or
pip-installable, and the docs must say the former without implying the latter.

### ⚠️ Plugin identity is genuinely unsettled — four competing plugins

[tenstorrent/vllm#452 "[Doc]: many vllm TT plugins"](https://github.com/tenstorrent/vllm/issues/452)
(opened 2026-07-28, **still open**) documents four, with TT engineering on record as unsure
which are current:

1. **`tenstorrent/vllm`** fork, in-tree `plugins/vllm-tt-plugin` — not "pure"; needs 4
   `ParallelConfig` extensions not yet upstream (`engine_core_cls`, `engine_core_proc_cls`,
   `dp_engine_core_proc_cls`, `engine_core_launcher_cls`). Being actively deprecated, but
   **still what tt-metal's README points at**.
2. **`tenstorrent/vllm-tt-plugin`** standalone repo (created 2026-07-02, pushed 2026-08-03) —
   a *pure* OOT plugin against **upstream `vllm==0.24.0`**, no fork. Being finalised as the
   intended replacement.
3. **`tt-inference-server/tt-vllm-plugin`** — dist `tt-vllm-plugin`, module `tt_vllm_plugin`,
   pins `vllm==0.10.1.1`. Status uncertain even to TT engineering.
4. **`tt-xla/integrations/vllm_plugin`** — module `vllm_tt`, TT-Forge lowering path, no TT
   model definitions. (Present locally: this box's `venv-forge` contains `vllm_tt`.)

**Consequence for this work:** which path to teach as *the* path is a real strategy decision,
not a documentation detail.

**Resolution:** teach the **in-fork** plugin (`plugins/vllm-tt-plugin` on `dev`). It is what
tt-metal's README documents and what tt-inference-server's own images actually build, so it
is the path a reader will find corroborated everywhere else. The lessons state plainly that
consolidation onto the standalone repo is expected, and note that when it happens the
install step is the only part that changes — `vllm serve`, `MESH_DEVICE`, and
`--additional-config` all carry over unchanged.

### Two install paths, both source-based

Path A — in-fork plugin (what tt-metal's README documents today):
```bash
# from a tenstorrent/vllm @ dev checkout, inside the tt-metal python env
source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh
```

Path B — standalone plugin, no fork (the intended future):
```bash
VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm \
    --override docs/vllm-overrides.txt vllm==0.24.0
uv pip uninstall torchaudio   # CUDA wheel, unloadable beside CPU torch
uv pip install -e .
```
The `--override` pins `opencv-python-headless==4.11.0.86` and `numpy>=1.24.4,<2`, because
ttnn needs numpy<2 while vLLM's opencv floor wants numpy>=2.

### Install path

```bash
# activate the tt-metal environment FIRST — most deps come from it
source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh
```

which is exactly:

```bash
VLLM_TARGET_DEVICE=empty uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -e plugins/vllm-tt-plugin
```

Note `uv`, not `pip`. `VLLM_TARGET_DEVICE` is **build-time only** and is `empty`, because
`tt` is supplied by the plugin at runtime.

### MESH_DEVICE grid (plugin `worker.py::get_mesh_grid`)

```
N150 (1,1)    N300 (1,2)    N150x4 (1,4)   T3K (1,8)   TG (8,4)
P100 (1,1)    P150 (1,1)    P150x2 (1,2)   P300 (1,2)
P150x4 (1,4)  P150x8 (1,8)  P300x2 (1,4)
```

A literal tuple string is also accepted, e.g. `MESH_DEVICE="(4,8)"`.

**`P300x2` → (1,4) is a TT-QuietBox 2 across all four Blackhole chips.**

### TT configuration namespace

TT knobs moved into vLLM's generic additional-config namespace:

```bash
--additional-config '{"tt": {"sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING"}}'
```

Keys include `trace_mode`, `trace_region_size`, `worker_l1_size`, `l1_small_size`,
`fabric_config`, `fabric_reliability_mode`, `dispatch_core_axis`, `enable_model_warmup`,
`optimizations`, `always_compat_sampling`, `input_queue_batching_delay`, plus multi-host
keys (`rank_binding`, `mpi_args`, `extra_ttrun_args`, `config_pkl_dir`, `env_passthrough`).

### Operational constraints the platform rejects early

- **Tensor parallel and pipeline parallel are not supported** (multi-chip is `MESH_DEVICE`)
- No speculative decoding, no LoRA
- Chunked prefill disabled
- Prompt logprobs rejected at request validation
- Prefix caching and async-decode overlap only for models declaring the capability

### Newly available surface worth teaching

- `vllm bench serve` for client/server benchmarking
- `EXTRA_MODELS_DIR` drop-in model bundles (`vllm_metadata.json`); `TT_VLLM_BUILTIN_MODELS=0`
- Hybrid KV cache for mixed sliding/full attention (Gemma 3/4, GPT-OSS); ~6x KV reduction on
  Gemma 4 31B at 256k context; not yet compatible with `data_parallel_size > 1`
- Single-process **lane-DP** for Galaxy generators: `--data_parallel_size N --max_num_seqs M`
  becomes N in-process lanes of M requests each
- Expanded model families: Llama 3.1/3.2/3.3 (+3.2 vision), Qwen2.5/Qwen3 text,
  Qwen3.5 via `models.demos.blackhole.qwen36` (Blackhole-specific), Qwen2.5-VL/Qwen3-VL,
  Mistral + Mistral 3 MM, Gemma 3 MM, DeepSeek V3, GPT-OSS 20B/120B

## Defects To Fix

| # | File | Defect |
|---|---|---|
| 1 | `content/lessons/deploy-to-koyeb.md:83` | Clones `tenstorrent/vllm` with **no `--branch dev`**. Fork `main` is now synced to pure upstream vLLM, so the image builds with zero TT support. **Broken.** |
| 2 | `content/pages/FAQ.md` (~1344) | Recommends `--tensor-parallel-size 2/8`. Platform **rejects TP**. **Broken advice.** |
| 3 | `content/lessons/vllm-production.md:1271` | Claims you must not call `api_server` directly "because TT models aren't registered". Now false. |
| 4 | `version-compatibility.md:167,295`; `step-zero.md:267-272` | `export VLLM_TARGET_DEVICE=tt` presented as a runtime var. Build-time only, and `empty`. |
| 5 | `content/templates/start-vllm-server.py` | `register_tt_models()` obsolete; registers bare `Qwen2ForCausalLM` and wrongly claims Gemma/Mistral reuse `TTLlamaForCausalLM`. |
| 6 | `content/templates/setup-vllm-env.sh`, `src/extension.ts::installVllm` | Old `pip install -e .` flow; missing the plugin install step entirely. |
| 7 | `content/lessons/vllm-production.md:613` | Calls TT-QuietBox 2 "4 independent single-chip devices"; contradicts our own verified CLAUDE.md correction and hides `P300x2`. |
| 8 | `content/pages/version-compatibility.md:70` | Attributes "Batch 32: 22.1 T/S/U, 707.2 T/S" to TT-QuietBox; tt-metal's table shows those are **n300** numbers. |

## Decisions

1. **Retire the custom starter script.** Delete `content/templates/start-vllm-server.py` and
   re-author lessons around plain `vllm serve`, exporting `MESH_DEVICE` explicitly per
   hardware. The script existed to call `ModelRegistry.register_model()`, which the plugin
   now does via entry points. Extension commands are rewritten to emit `vllm serve`
   invocations directly instead of materialising a script into `~/tt-scratchpad`.
2. **Deep rewrite of `vllm-production.md`; surgical fixes elsewhere.**
3. **Document the QB2 4-chip path** (`MESH_DEVICE=P300x2`).

### Amendment to decision 3 — validation status

The approved choice was to document `P300x2` as *validated*. Verification on this machine
contradicts the premise for that claim:

- Hardware is a genuine TT-QuietBox 2 (4× P300C) and `tt_doctor` reports `mesh_device: P300X2`.
- But the active `vllm` env is the **pre-plugin** stack: vLLM `0.10.0rc2.dev199+gaa4ae1edc`
  (the Oct 2025 fork commit), `import vllm_tt_plugin` fails, and no TT entry points are
  registered. `~/tt-vllm` is detached at `aa4ae1edc` with no `plugins/` directory.

So no plugin-based P300x2 serving run has happened here, and `validatedOn` is a claim about
our own testing. **We will list `p300x2` in `supportedHardware` and document the path as
plugin-supported, but leave `validatedOn` unchanged until an actual run happens.** The exact
command to validate is recorded in the lesson so it can be flipped in one edit afterward.
Verifying it requires installing the plugin stack, which would alter the user's working
`vllm` env — an action to confirm separately, not take incidentally.

## Scope

**Deep rewrite:** `content/lessons/vllm-production.md`

**Surgical fixes:** `content/pages/FAQ.md`, `content/lessons/deploy-to-koyeb.md`,
`content/pages/version-compatibility.md`, `content/pages/step-zero.md`,
`content/lessons/tt-inference-server.md`, `content/lessons/coding-assistant.md`,
`content/lessons/qb2-openclaw-assistant.md`, `content/lessons/qb2-local-agents.md`,
`content/lessons/verify-installation.md`, `CONTRIBUTING.md`

**Code/templates:** delete `content/templates/start-vllm-server.py`; rewrite
`content/templates/setup-vllm-env.sh`; update `src/extension.ts` (`cloneVllm`,
`installVllm`, `startVllmServer`, `startVllmServerWithHardware`,
`startVllmServerForHardware`)

**Housekeeping:** bump `package.json` version, add `CHANGELOG.md` entry, keep
`content/lesson-registry.json` in sync (`npm run validate:lessons`), refresh the CLAUDE.md
line describing `vendor/vllm`.

## Migration Note For Existing Users

A pre-plugin `~/tt-vllm` checkout (no `plugins/` directory) cannot load the plugin. Lessons
must tell users to update the checkout to `dev` and re-run the installer, rather than
assuming a fresh clone.

## Verification

- `npm run validate:lessons` passes (markdown front matter ↔ registry in sync)
- `npm run build` succeeds (validation is wired into build)
- No remaining `VLLM_TARGET_DEVICE=tt`, `--tensor-parallel-size`, manual `ModelRegistry`
  guidance, or `git clone` of the fork without `--branch dev` anywhere in `content/` or `src/`
- Every `MESH_DEVICE` value used in content appears in the plugin's `get_mesh_grid` table
