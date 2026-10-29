---
id: verify-installation
title: Verify Your Setup
description: >-
  Four quick checks before your first model: the stack is current, the hardware
  is visible, TT-NN imports where it should, and tt-metal source is present if
  you need it. A diagnostic checkpoint you can come back to.
category: setup
tags:
  - installation
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
estimatedMinutes: 5
minTTMetalVersion: v0.65.1
recommended_metal_version: v0.65.1
validationDate: 2026-04-15
validationNotes: Rewritten as diagnostic hub for TT-QuietBox 2/WH/BH parity
---

# Verify Your Setup

This is your diagnostic checkpoint. Four quick checks confirm the stack is current, the
hardware is visible, TT-NN<sup>™</sup> imports where it should, and (optionally) tt-metal
source is present. If anything fails, follow the link for that check, then come back here
to confirm you're green before moving on.

> **Which Check 2 is yours:** on the **cloud pod**, TT-NN lives in `~/tt-metal/python_env`,
> which is active in every shell, so the plain Python check passes. On a **TT-QuietBox 2**,
> TT-NN lives in the `tt-metalium` container, so use the container check. Check 4 fails on
> a QuietBox unless you built TT-Metalium<sup>™</sup> yourself; that is expected and fine
> for every lesson that uses `tt serve`.

---

## Check 0: Is the stack current?

```bash
tt update --dry-run
```

[👀 Preview upgrade](command:tenstorrent.setup.updateDryRun)

**Interpreting results:**

- Every row `up-to-date` → **✅ Current**, continue to Check 1
- A tools row says `install` or `update` → run `tt update` from
  [Install and Upgrade with tt](command:tenstorrent.lesson.show?["tt-installer"]). On the
  cloud pod a tools row saying `install` is normal: `tt` fetches the tool the first time a
  command needs it, and the pod's driver and firmware are managed for you.
- `tt: command not found` → install it in the same lesson

---

## Check 1: Hardware

Run this in your terminal:

```bash
tt device status
```

[🔍 Device status](command:tenstorrent.setup.deviceStatus)

**Interpreting results:**

- One row per chip, `DRAM True` → **✅ Hardware OK**, continue to Check 2
- Fewer rows than chips, or an error → go to
  [Hardware Detection](command:tenstorrent.lesson.show?["hardware-detection"]) for the
  diagnosis steps (`tt-smi -s` has the raw fields)

---

## Check 2: TT-NN

**Cloud pod, or any machine with a built tt-metal and its venv active:**

```bash
python3 -c "import ttnn; print('✓ TTNN', getattr(ttnn, '__version__', '(source build)'), ttnn.__file__)"
```

[🐍 Check TT-NN (Python env)](command:tenstorrent.setup.verifyTtnnCloud)

**TT-QuietBox 2 (TT-NN lives in the `tt-metalium` container):**

```bash
tt-metalium -c "python3 -c \"import ttnn; print('✓ TTNN', getattr(ttnn, '__version__', '(source build)'))\""
```

[📦 Check TT-NN (container)](command:tenstorrent.setup.verifyTtnnQb2)

**Interpreting results:**

- Prints `✓ TTNN <version>` → **✅ TT-NN ready**, continue to Check 3
- `ModuleNotFoundError: No module named 'ttnn'` → TT-NN is not importable in your current
  Python environment. You need one of:
  - **TT-QuietBox 2:** use the container button above; the host has no TT-NN by design
  - **Build-from-source users:** activate your tt-metal venv and set `TT_METAL_HOME`:
    ```bash
    source ~/tt-metal/python_env_3.12/bin/activate   # or python_env for Python 3.10
    export TT_METAL_HOME=~/tt-metal
    export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
    ```
  - **Don't have TT-Metalium yet:** go to
    [Build TT-Metalium from Source](command:tenstorrent.lesson.show?["build-tt-metal"]) and
    return here when done
- `ImportError: undefined symbol: MPIX_Comm_revoke` → OpenMPI libraries missing from
  `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
  ```

---

## Check 3: tt-metal source (optional)

> **Skip this check** if you are using the vLLM production path — tt-metal source is not
> required. Only the Generator API lessons (Interactive Chat) need it.

```bash
[ -d ~/tt-metal ] && echo "✓ tt-metal source present at ~/tt-metal" \
  || echo "✗ ~/tt-metal not found — OK for vLLM path, needed for Generator API"
```

**Interpreting results:**

- `✓ tt-metal source present` → **✅ Source ready.** Generator API lessons will work.
  If TT-Metalium is present but not yet built, run `bash /tmp/build_tt_metal.sh` inside the
  developer container, or go to
  [Build TT-Metalium from Source](command:tenstorrent.lesson.show?["build-tt-metal"]).
- `✗ ~/tt-metal not found` → Source not present. That's expected on TT-QuietBox 2 pre-installed images
  and fine for all vLLM production lessons. Only needed for the Generator API (Interactive Chat).
  If you need it, go to
  [Build TT-Metalium from Source](command:tenstorrent.lesson.show?["build-tt-metal"]) and
  return here when done.

---

## All checks green?

Choose your next step based on your goal:

- **vLLM production serving** (Qwen3-0.6B, no source needed) →
  [vLLM Production →](command:tenstorrent.lesson.show?["vllm-production"])
- **Interactive chat** (requires tt-metal source + Llama model) →
  [Interactive Chat →](command:tenstorrent.lesson.show?["interactive-chat"])
- **Still something failing?** →
  [Build TT-Metalium from Source →](command:tenstorrent.lesson.show?["build-tt-metal"])
