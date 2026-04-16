---
id: build-tt-metal
title: Build tt-metal from Source
description: >-
  Clone and build tt-metal from source. Required for Direct API (Generator API)
  lessons and for running tt-metal examples directly. QB2 and pre-configured
  images do not ship with ~/tt-metal — start here if Check 3 in Verify Your
  Setup failed.
category: advanced
tags:
  - installation
  - build
  - source
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
status: validated
validatedOn:
  - n150
  - p300c
estimatedMinutes: 60
---

# Build tt-metal from Source

This lesson walks you through cloning and building tt-metal from source on your
Tenstorrent hardware. Once complete, return to
[Verify Your Setup](command:tenstorrent.showLesson?["verify-installation"]) to
confirm Check 3 is green.

---

## When you need this

You need a local tt-metal source build if any of the following apply:

- **Direct API / Generator API lessons (Lessons 4–5)** — These run Python
  scripts that import `ttnn` from source and use `tt_lib` / `ttnn` Generator
  APIs. They require `TT_METAL_HOME` to point to a built copy.
- **Custom kernels** — Writing or modifying C++/Metal kernels requires the full
  source tree and build system.
- **Running tt-metal examples directly** — Scripts under `tt-metal/models/` or
  `tt-metal/tests/` assume the repo is present and built.
- **Debugging TTNN ops or dispatch issues** — Source-level tracing requires a
  built checkout.

You do **not** need a source build for:

- **vLLM inference (Lessons 6–7)** — tt-inference-server runs inside Docker/
  Podman with its own bundled Metal wheels.
- **TTNN Python API basics** — TTNN is pip-installable for simple op usage.
- **Chat assistant (Lesson 8)** and other lessons that go through vLLM.

When in doubt: if `Check 3` in Verify Your Setup is red and you want it green,
run through this lesson.

---

## QB2 / Pre-configured image note

> **QB2 does not ship with `~/tt-metal`.**

QB2 (Quantum Bridge 2) and similar pre-configured Tenstorrent demo images
include tt-smi, Metal drivers, and a Python environment, but they do **not**
include a built tt-metal source tree. You must clone and build it yourself
before any command here will work.

This is expected — the source tree is large (~4 GB with submodules) and build
times are 30–60 minutes, so it is not bundled into images.

---

## Docker vs Podman note

tt-metal's `install_dependencies.sh` script defaults to **Podman** on some
systems. QB2 images typically ship with Docker and do not have Podman installed.

If `sudo ./install_dependencies.sh` exits with an error referencing `podman`
(e.g. `command not found: podman` or permission errors), install Docker first:

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
# Log out and back in so the group change takes effect
```

Then re-run `install_dependencies.sh`. If your system has Podman and you prefer
to keep it, the script should work as-is — just ensure it is in `$PATH`.

---

## Step 1: Clone

```bash
git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal
```

> **`--recurse-submodules` is required.** tt-metal depends on several
> submodules (including tt-metalium and third-party headers). The build will
> fail with missing-file errors if you skip this flag.

If you already cloned without `--recurse-submodules`, fix it in-place:

```bash
cd ~/tt-metal
git submodule update --init --recursive
```

---

## Step 2: Install system dependencies

```bash
cd ~/tt-metal
sudo ./install_dependencies.sh
```

This script installs apt packages (compilers, CMake, OpenMPI, etc.) and sets
up any driver-level requirements. It is safe to re-run if you need to repair
a partial installation.

Common outcomes:
- **Success** — prints something like `All dependencies installed.` Exit 0.
- **Podman error** — see Docker note above.
- **Permission denied on `/dev/tenstorrent`** — run
  `sudo chmod a+rw /dev/tenstorrent/*` or add your user to the `tenstorrent`
  group and log out/in.

---

## Step 3: Fix Python environment (common on QB2 / Ubuntu 22+)

Before installing Python requirements, upgrade core packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

This prevents a very common failure on fresh Ubuntu 22.04 and 24.04 installs
where `pkg_resources` is missing or outdated.

**If `mmcv` fails to build:** `mmcv` (a vision model utility) is listed in
`tt_metal/python_env/requirements-dev.txt` but is only needed for image/video
model work. If you are focused on LLM inference and `mmcv` fails (e.g. due to a
CUDA mismatch or missing build tools), you can comment it out:

```bash
# Comment out mmcv in the requirements file before building:
sed -i 's/^mmcv/#mmcv/' ~/tt-metal/tt_metal/python_env/requirements-dev.txt
```

This has no effect on LLM, transformer, or TTNN op lessons.

---

## Step 4: Build

```bash
cd ~/tt-metal
./build_metal.sh
```

Build time is typically **30–60 minutes** on a fresh checkout, depending on
CPU core count. The build uses all available cores by default.

**Faster subsequent rebuilds** (incremental, skips unchanged objects):

```bash
./build_metal.sh --enable-ccache
```

CCache caches compiled object files, reducing incremental build time to
1–5 minutes after the first full build.

**Clean rebuild** (use this if you see stale symbol errors or unexplained
linker failures):

```bash
./build_metal.sh --clean
```

A clean rebuild discards all cached objects and starts from scratch. Use it
when switching branches or after a significant dependency change.

---

## Step 5: Set environment variables

After the build succeeds, export these variables before running any tt-metal
Python scripts:

```bash
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

**`TT_METAL_HOME`** — tells the Metal runtime where the source tree lives
(kernel files, device configs, etc.).

**`PYTHONPATH`** — makes `import ttnn` and `import tt_lib` resolve to the
locally built copies rather than any pip-installed version.

**`LD_LIBRARY_PATH`** — points to the OpenMPI build that tt-metal's multi-device
dispatch was compiled against. Omitting this causes `ImportError: undefined
symbol: MPIX_Comm_revoke` when importing `ttnn` on multi-device systems.

### Make permanent

To avoid re-exporting after every login, add these lines to `~/.bashrc`:

```bash
echo 'export TT_METAL_HOME=~/tt-metal' >> ~/.bashrc
echo 'export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 6: Verify

Run the built-in TTNN smoke test:

```bash
python3 -m ttnn.examples.usage.run_op_on_device
```

A successful run prints device open/close messages and a result tensor without
any Python tracebacks. If it passes, your build is good.

Return to [Verify Your Setup](command:tenstorrent.showLesson?["verify-installation"])
and re-run the checks — Check 3 should now be green.

---

## Blackhole architecture note (P100 / P150 / P300c / QB2)

> **This section is critical if you are writing or adapting scripts for
> Blackhole hardware (P100, P150, P300c, QB2).**

When opening a mesh device with `ttnn.open_mesh_device()`, do **not** hardcode
`DispatchCoreAxis.ROW`. Blackhole uses column dispatch (`COL`), and passing
`ROW` explicitly causes a crash:

```python
# ❌ Crashes on Blackhole (P100/P150/P300c/QB2):
dispatch_core_config = ttnn.DispatchCoreConfig(
    ttnn.DispatchCoreType.WORKER,
    ttnn.DispatchCoreAxis.ROW  # <-- hardcoded axis, breaks on Blackhole
)

# ✅ Arch-agnostic — auto-detects COL on Blackhole, ROW on Wormhole:
dispatch_core_config = ttnn.DispatchCoreConfig(
    ttnn.DispatchCoreType.WORKER
    # No axis argument — Metal picks the correct one for your hardware
)
```

The same rule applies to any script that eventually calls into
`ttnn.CreateDevice()` or `ttnn.open_mesh_device()` with explicit dispatch
config. If you copy example scripts from tt-metal and they hardcode
`DispatchCoreAxis.ROW`, patch them before running on Blackhole.

---

## Common errors and fixes

| Error message | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install --upgrade pip setuptools wheel` |
| `ImportError: undefined symbol: MPIX_Comm_revoke` | `export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH` |
| `fatal: no submodule mapping found` or missing header files | `git submodule update --init --recursive` inside `~/tt-metal` |
| `ROW dispatch core axis is not supported for blackhole` | Use `DispatchCoreConfig(WORKER)` with no axis argument — see Blackhole note above |
| Build exits with a linker or CMake error | `./build_metal.sh --clean && sudo ./install_dependencies.sh`, then rebuild |
| `command not found: podman` during `install_dependencies.sh` | Install Docker: `sudo apt-get install -y docker.io` and re-run the script |

---

Once the smoke test passes, return to
[Verify Your Setup](command:tenstorrent.showLesson?["verify-installation"]) to
confirm all checks are green before continuing with the lessons.
