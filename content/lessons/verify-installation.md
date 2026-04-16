---
id: verify-installation
title: Verify Your Setup
description: >-
  Check that your Tenstorrent hardware, TTNN, and optional tt-metal source are
  ready before running your first model. A diagnostic checkpoint — returns you
  here after any setup work.
category: first-inference
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
validationNotes: Rewritten as diagnostic hub for QB2/WH/BH parity
---

# Verify Your Setup

This is your diagnostic checkpoint. Run three quick checks to confirm hardware, TTNN,
and (optionally) tt-metal source are ready. If anything fails, follow the link for that
check — then come back here to confirm you're green before moving on.

> **QB2 / Pre-configured image users:** QB2 ships with a pre-installed environment but
> does **not** include `~/tt-metal`. Check 1 and Check 2 should pass out of the box.
> Check 3 will fail unless you clone and build tt-metal yourself — that's expected and
> fine for most lessons.

---

## Which path are you on?

Before running checks, pick the path that matches your goal:

| Goal | What you need | Next lesson after green |
|------|--------------|------------------------|
| Interactive chat with Llama (Generator API) | Hardware + TTNN + tt-metal source | [Interactive Chat](command:tenstorrent.showLesson?["interactive-chat"]) |
| Production vLLM serving (Qwen3-0.6B, no source needed) | Hardware + TTNN | [vLLM Production](command:tenstorrent.showLesson?["vllm-production"]) |
| Image generation, TT-Forge, TT-XLA | Hardware + TTNN (source optional) | See individual lesson |

Not sure? Start with the vLLM path — it works on all hardware without needing to build from source.

---

## Check 1: Hardware

Run this in your terminal:

```bash
tt-smi -s
```

[▶ Run Hardware Check](command:tenstorrent.verifyInstallation)

**Interpreting results:**

- JSON output shows your device(s) → **✅ Hardware OK**, continue to Check 2
- `command not found` → drivers or tt-smi not installed →
  go to [tt-installer](command:tenstorrent.showLesson?["tt-installer"]) and return here when done
- Device shown but with error flags → firmware or driver issue; consult
  [tt-installer](command:tenstorrent.showLesson?["tt-installer"]) troubleshooting

---

## Check 2: TTNN

```bash
python3 -c "import ttnn; print('✓ TTNN', getattr(ttnn, '__version__', '(source build)'))"
```

**Interpreting results:**

- Prints `✓ TTNN <version>` → **✅ TTNN ready**, continue to Check 3
- `ModuleNotFoundError: No module named 'ttnn'` → TTNN is not importable in your current
  Python environment. You need one of:
  - **QB2 / tt-installer users:** activate the pre-installed container or venv
    (check your setup guide for the activate command)
  - **Build-from-source users:** activate your tt-metal venv and set `TT_METAL_HOME`:
    ```bash
    source ~/tt-metal/python_env_3.12/bin/activate   # or python_env for Python 3.10
    export TT_METAL_HOME=~/tt-metal
    export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
    ```
  - **Don't have tt-metal yet:** go to
    [Build tt-metal from Source](command:tenstorrent.showLesson?["build-tt-metal"]) and
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

[▶ Run tt-metal Source Check](command:tenstorrent.verifyInstallation)

**Interpreting results:**

- `✓ tt-metal source present` → **✅ Source ready.** Generator API lessons will work.
  If tt-metal is present but not yet built, run `bash /tmp/build_tt_metal.sh` inside the
  developer container, or go to
  [Build tt-metal from Source](command:tenstorrent.showLesson?["build-tt-metal"]).
- `✗ ~/tt-metal not found` → Source not present. That's expected on QB2 pre-installed images
  and fine for all vLLM production lessons. Only needed for the Generator API (Interactive Chat).
  If you need it, go to
  [Build tt-metal from Source](command:tenstorrent.showLesson?["build-tt-metal"]) and
  return here when done.

---

## All checks green?

Choose your next step based on your goal:

- **vLLM production serving** (Qwen3-0.6B, no source needed) →
  [vLLM Production →](command:tenstorrent.showLesson?["vllm-production"])
- **Interactive chat** (requires tt-metal source + Llama model) →
  [Interactive Chat →](command:tenstorrent.showLesson?["interactive-chat"])
- **Still something failing?** →
  [Build tt-metal from Source →](command:tenstorrent.showLesson?["build-tt-metal"])
