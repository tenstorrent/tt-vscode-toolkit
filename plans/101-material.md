# QB2 Experience Analysis + First-Inference Lesson Restructure

Source: `experience_notes.md` — coworker's QB2 (QuietBox 2, 4x P300c Blackhole) walkthrough log

---

## Part 2: First-Inference Lesson Restructure

### Goal
Restructure the early lessons so users reach the **same steady state** regardless of how they arrived:
- Fresh machine with tt-installer (`~/.tenstorrent-venv` + Podman/Docker container with TTNN)
- Pre-configured box (QB2, cloud image)
- Source build fan (wants to clone `~/tt-metal`)

Replace Llama-3.1-8B (Meta license gate, DRAM-heavy on N150/P300c) with **Qwen3-0.6B** as the primary first-model throughout. Llama remains as an "optional gated model" note.

---

### Current Chain (first-inference category)
`tt-installer` → `hardware-detection` → `verify-installation` → `download-model` → `interactive-chat` → `api-server`

**Problems with current chain:**
- `verify-installation` runs `python3 -m ttnn.examples.usage.run_op_on_device`, which assumes `~/tt-metal` is cloned. Breaks immediately for QB2/tt-installer users who don't have it.
- It's placed BEFORE `download-model`, so people without tt-metal are blocked before they can even get a model.
- `download-model` leads with Llama (Meta license gate, data farming objections)
- `interactive-chat` uses the Generator API (`models/tt_transformers`) which is Llama-specific and requires tt-metal source
- No branching: "do you have tt-metal source or not?"

---

### Proposed New Chain
`tt-installer` → `hardware-detection` → `download-model` → `verify-installation` (rewritten) → `interactive-chat` → `api-server`

With two side-quest reference lessons linked from `verify-installation`:
- `tt-installer` (already exists) — for users who need to install drivers/tools
- `build-tt-metal` (new) — for users who need tt-metal source

---

### Three Roles in the New Design

#### 1. `verify-installation` → rewritten as "Verify Your Setup"
This becomes a **diagnostic hub and return-to article**:
- "Let's check what you have right now"
- Runs 3 checks: `tt-smi -s`, `python3 -c "import ttnn"`, optional `run_op_on_device`
- If all pass → proceed to interactive-chat / vLLM
- If tt-smi fails → link to [tt-installer lesson](tt-installer)
- If TTNN import fails → link to [Build tt-metal from Source](build-tt-metal)
- If `run_op_on_device` fails → link to [Build tt-metal from Source](build-tt-metal) with debug tips
- People return to this article after doing setup work to confirm they're green

#### 2. New `build-tt-metal.md` lesson
Standalone deep-dive for building tt-metal from source. Lives outside main chain (reference lesson, not sequential). Linked from `verify-installation`.
Content:
```
## Why you need this
[when source is required vs. when container/venv is enough]

## QB2 / Pre-baked image users
QB2 does NOT ship with ~/tt-metal. You must clone it first.

## Clone
git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git ~/tt-metal

## Docker vs Podman note
[QB2 recommend Docker; tt-installer default is Podman — link to DEVSTACK-42]

## Install system dependencies
cd ~/tt-metal && sudo ./install_dependencies.sh

## Fix Python env (common QB2/Ubuntu 22/24 issue)
pip install --upgrade pip setuptools wheel
# Comment out mmcv in requirements-dev.txt if it fails (only needed for vision models)

## Build
./build_metal.sh [--enable-ccache for faster rebuilds]

## Set environment variables
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH

## Verify
python3 -m ttnn.examples.usage.run_op_on_device
→ Return to [Verify Your Setup](verify-installation) to confirm you're green

## Blackhole note: DispatchCoreAxis
If writing your own scripts, do NOT hardcode DispatchCoreAxis.ROW on Blackhole:
  # ❌ Crashes on Blackhole (P100/P150/P300c):
  ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.ROW)
  # ✅ Arch-agnostic (auto: COL on Blackhole, ROW on Wormhole):
  ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
```

#### 3. `download-model.md` → revised to lead with Qwen3-0.6B
```
1. Prerequisites check (tt-smi, python3, hf CLI)
2. Get HF token
3. Authenticate: `hf auth login --token "$HF_TOKEN"`
4. Download Qwen3-0.6B (no license gate, 0.6B params, works on all hardware):
   hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B
5. [Optional gated model] Llama-3.1-8B-Instruct (Meta requires personal data):
   [callout: "Meta requires you to accept their data terms. If you prefer open models, Qwen3-0.6B is an excellent alternative."]
   hf download meta-llama/Llama-3.1-8B-Instruct ...
6. Next: [Verify Your Setup →](verify-installation)
```

---

### Files to Create/Modify

| Action | File | Changes |
|--------|------|---------|
| **REWRITE** | `content/lessons/verify-installation.md` | "Verify Your Setup" diagnostic hub; 3-check flow; links to tt-installer + build-tt-metal; move BEFORE interactive-chat but AFTER download-model |
| **CREATE** | `content/lessons/build-tt-metal.md` | New reference lesson: clone, Docker/Podman, setuptools, build, verify |
| **REVISE** | `content/lessons/download-model.md` | Qwen3-0.6B primary; `hf auth login` / `hf download`; Llama → optional gated section; chain order: after hardware-detection, before verify-installation |
| **REVISE** | `content/lessons/hardware-detection.md` | Add p300c to validatedOn; P100 equivalence callout more prominent |
| **REVISE** | `content/lessons/tt-installer.md` | Change category `advanced` → `first-inference`; position as recommended on-ramp |
| **REVISE** | `content/lessons/interactive-chat.md` | `hf` CLI; Generator API = Llama-only notice; link to vLLM path for non-Llama users |
| **UPDATE** | `content/lessons/vllm-production.md` | Add QB2/p300c hardware section; `hf` CLI update |
| **UPDATE** | `content/lesson-registry.json` | Add `build-tt-metal` entry; reorder chain: hw-detect → download-model → verify-installation; update previousLesson/nextLesson |
| **FIX** | `content/templates/tt-chat-direct.py` | Drop hardcoded `DispatchCoreAxis.ROW` — use `DispatchCoreConfig(WORKER)` with no axis (auto-detects: COL on Blackhole, ROW on Wormhole) |
| **FIX** | `content/templates/tt-coding-assistant.py` | Same arch-agnostic fix |
| **FIX** | `content/templates/tt-api-server-direct.py` | Same arch-agnostic fix |
| **FIX** | `src/commands/terminalCommands.ts` | Add `export HF_MODEL=...` to RUN_INFERENCE template; prepend `pip install setuptools wheel` to setup cmd |
| **BUMP** | `package.json` | Version increment |
| **UPDATE** | `CHANGELOG.md` | Entry for this restructure |
| **UPDATE** | `CLAUDE.md` | Add WH/BH compatibility principle near top (after Project Overview) and near bottom (end of file); exact text below |

---

### CLAUDE.md Text to Insert

**Insertion point 1 — Near top:** After the `## Project Overview` section (line 14), before `## 🔧 Recent Multi-Device API Update`

```markdown
## Hardware Compatibility Goal: Wormhole + Blackhole

**This project targets both Wormhole (WH) and Blackhole (BH) architectures.** Every lesson, template, and code example should work on both unless there is a documented, necessary divergence.

**Guiding principles:**
- **Prefer arch-agnostic code** — where TTNN/tt-metal provides auto-detection (e.g., `DispatchCoreConfig` without a hardcoded axis), use it. Don't hardcode arch-specific behavior unless unavoidable.
- **When diverging is required** — be explicit, educational, and upfront about which hardware is affected and why. The reader should never be surprised.
- **Never silently fail** — a user following a lesson must not reach code that crashes on their hardware without a clear warning beforehand and a working alternative. The last thing we want is someone running the wrong script.
- **Track validation honestly** — if a template or lesson has only been tested on N150 (Wormhole), say so in `validatedOn`. Don't list hardware as supported unless it has been verified.

**Known arch differences to watch for:**
- `DispatchCoreAxis`: ROW is the Wormhole default, COL is the Blackhole default → use `DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)` with no hardcoded axis; TTNN auto-detects the right value
- Multi-device API: Always use `CreateDevices`/`CloseDevices` (see Multi-Device section below)
- `MESH_DEVICE` env var: N150/N300/T3K for Wormhole; P100/P150 for single-chip Blackhole (P300c = P100 mode)
- Model DRAM: Llama-3.1-8B consistently exhausts N150 DRAM; Qwen3-0.6B is the recommended first-model for all hardware
```

**Insertion point 2 — Near end:** Append at end of file, after `**See [CHANGELOG.md](CHANGELOG.md) for complete version history.**`

```markdown
## WH/BH Compatibility Checklist

When writing or reviewing any lesson, template, or extension command, verify:

- [ ] Works on both Wormhole (N150/N300/T3K/Galaxy) and Blackhole (P100/P150/P300c) — or divergence is clearly documented
- [ ] `DispatchCoreAxis` is NOT hardcoded (`ROW` crashes Blackhole) — use `DispatchCoreConfig(WORKER)` with no axis
- [ ] `huggingface-cli` commands updated to `hf auth login` / `hf download`
- [ ] Lesson `supportedHardware` metadata accurately reflects tested hardware
- [ ] `validatedOn` only lists hardware that has been actually tested
- [ ] If the model requires `~/tt-metal` source, the lesson says so and links to the build lesson
- [ ] If code is Llama-specific (Generator API), there is a visible path for users who can't access gated models
```

---

### Qwen3-0.6B in the Direct API Path (Constraint)
The Generator API in `models/tt_transformers` (`interactive-chat`, `api-server` lessons) appears Llama-specific. **Plan:** Keep Llama for Generator API lessons, but add a clear notice: "If you haven't accepted Meta's terms, use the [vLLM path with Qwen3-0.6B](vllm-production) instead." Validate Qwen3 support in tt_transformers before any further switch.

---

---

## Themes Extracted from QB2 Experience Notes

### Theme 1: Hardware Recognition Gap (P300c not "validated")
- **What happened:** `hardware-detection.md` metadata lists `validatedOn: [n150]` only. `supportedHardware` does not include `p300` or `p300c`. User saw "P100" guidance where they have "P300c" — confusing.
- **The fact:** P300c is architecturally identical to P100 (single Blackhole chip), but the lesson doesn't explain this clearly enough or show P300c in the validated hardware list.

### Theme 2: "Lesson Not Found" Navigation Bug
- **What happened:** Clicking the "Next Step" button from Hardware Detection triggered a `Lesson not found: <id>` error. User had to navigate back to the welcome page to proceed.
- **Root cause:** The `tenstorrent.showLesson` command shows the error at `extension.ts:4390`. Likely the `nextLesson` ID in `lesson-registry.json` doesn't match an actual lesson ID, or the walkthrough step wires up the wrong lesson ID.

### Theme 3: tt-metal Not Preinstalled on QB2
- **What happened:** QB2 ships with a preinstalled image but tt-metal is NOT in the home directory. The extension assumes it exists. User had to clone it manually 4 times across multiple rounds.
- **Missing guidance:** No QB2-specific note in `verify-installation` lesson about this. The "Install System Dependencies" button blindly runs `cd ~/tt-metal && sudo ./install_dependencies.sh` and fails with "directory not found."

### Theme 4: Python Environment / mmcv Build Failure
- **What happened:** `pip install -r tt_metal/python_env/requirements-dev.txt` fails on QB2 due to:
  1. `ModuleNotFoundError: No module named 'pkg_resources'` (needs `pip install setuptools wheel` first)
  2. `mmcv==2.2.0` build error (only needed for vision models; not relevant for Llama inference)
- **Related:** User was using system Python + tenstorrent-venv (created by tt-installer), not a fresh env.
- **Fix via Docker:** Round 4 succeeded only after switching to Docker (not Podman) for tt-installer.

### Theme 5: `huggingface-cli` Deprecated
- **What happened:** `huggingface-cli login` and `huggingface-cli whoami` are deprecated. The new CLI is `hf auth login` / `hf auth whoami`.
- **Affected files:** `download-model.md`, `interactive-chat.md`, `image-generation.md`, `api-server.md`, `vllm-production.md`, `animatediff-video-generation.md`, `bounty-program-model-bringup.md`

### Theme 6: HF_MODEL Not Set Before Inference
- **What happened:** The "Run Inference Now!" button runs pytest without setting `HF_MODEL`. The `RUN_INFERENCE` template in `terminalCommands.ts:181` uses `LLAMA_DIR` not `HF_MODEL`, but `simple_text_demo.py` requires `HF_MODEL`.
- **Result:** `ValueError: Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct`

### Theme 7: DispatchCoreAxis.ROW Unsupported on Blackhole
- **What happened:** `tt-chat-direct.py` template uses `DispatchCoreAxis.ROW` which is not supported on Blackhole unless fabric tensix MUX is enabled.
- **Error:** `ValueError: ROW dispatch core axis is not supported for blackhole arch unless fabric tensix MUX is enabled`
- **Same issue in:** `tt-coding-assistant.py`, `tt-api-server-direct.py`
- **Fix:** Drop the axis argument — `DispatchCoreConfig(WORKER)` auto-detects COL on BH, ROW on WH.

### Theme 8: "When to Use Which" Section Placement
- **What happened:** The "When to use Which" (source vs. Docker) section appears _after_ the user has already installed from source. Should come first.
- **File:** `verify-installation.md`

### Theme 9: Copy Environment Setup Button Appeared to Not Work
- **What happened:** The QB2 already has `(tenstorrent-venv)` active from tt-installer, creating a conflicting environment when the extension tries to set env vars.

### Theme 10: "Cp Template" Command No Longer Needed (Stale UI)
- **What happened:** The "Create the Direct API Chat Script" step shows a `cp template` button, but the script already exists in the repo.

### Theme 11: Meta Llama License Friction
- **What happened:** User strongly objects to Meta's data requirements for Llama model access.
- **Opportunity:** Qwen3-0.6B as default throughout; Llama as optional gated section.

---

## Critical Path for QB2 (Priority Order)

1. **DispatchCoreAxis.ROW (Theme 7)** — Hard crash on all Blackhole. Fix: `DispatchCoreConfig(WORKER)` no axis.
2. **HF_MODEL not set (Theme 6)** — Blocks pytest inference command.
3. **tt-metal not on QB2 (Theme 3)** — First blocker for fresh QB2 users.
4. **huggingface-cli deprecated (Theme 5)** — Quick multi-file text update.
5. **Python env / mmcv (Theme 4)** — Real UX friction, needs guidance.
6. **Hardware Detection p300c (Theme 1)** — Low effort, high confidence signal.
7. **Lesson navigation bug (Theme 2)** — Verify and fix nextLesson ID chain.

---

## Verification Steps

1. `npm run validate:lessons` — ensure lesson registry sync
2. `npm run build` — clean compile
3. `npm run package` — produce `.vsix`
4. Code review template files for correct DispatchCoreAxis usage
5. During next QB2 session: test steps 1–5 in sequence, confirm no crashes
