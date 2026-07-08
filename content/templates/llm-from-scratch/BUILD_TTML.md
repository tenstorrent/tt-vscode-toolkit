# Building `ttml` from source (verified recipe)

`ttml` (tt-train) is the training library the "Build an LLM from Scratch,
TT-Native" arc uses for Lab 5's real on-device training. **It is source-only —
there is no pip wheel or `.deb`.** This recipe was verified end-to-end on
**2026-07-08 on a Blackhole p300c against tt-metal v0.73**: after building, the
canonical `train_nanogpt.py` ran a real forward + backward + AdamW loop
on-device for the **modern Llama-3 stack** (`--config
training_shakespeare_nanollama3_char.yaml`, `model_type: llama` — RoPE +
RMSNorm + GQA + SwiGLU) with loss dropping 4.69 → 3.23 over 20 steps, ~65 ms/step,
16.5 TFLOPS, exit 0.

> This retires the `ct7`/`ct8` "no way to get ttml" blocker for anyone willing
> to build from source. If you do not already have a tt-metal source tree, do
> the **build-tt-metal** lesson first — you need `install_dependencies.sh` +
> `build_metal.sh` working before any of the below.

## Prerequisites

- A tt-metal **source + build tree** (referred to as `$TT_METAL_HOME`; here
  `/home/ttuser/tt-metal`). TT-QuietBox 2 images ship TT-NN + vLLM but **not**
  the tt-metal source tree — clone and build it first (build-tt-metal lesson).
- The Python venv you want `ttml` on (here `/home/ttuser/.tenstorrent-venv`,
  Python 3.12).

## Recipe

```bash
export TT_METAL_HOME=/home/ttuser/tt-metal
export CMAKE_POLICY_VERSION_MINIMUM=3.5          # precaution for cmake 4.x
cd $TT_METAL_HOME

# 1. Configure the tt-train subproject.
./build_metal.sh --build-tt-train --configure-only

# 2. Build the ttml python bindings (~4 min with warm ccache).
cmake --build build_Release --target _ttml

# 3. *** REQUIRED *** rebuild ttnn's nanobind so its ABI matches ttml.
#    Skipping this is the #1 cause of `std::bad_cast` on `import ttml`.
ninja -C build_Release ttnn/_ttnn.so
cp -a build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so

# 4. Wire ttml onto the venv with a .pth (INSTALLING_TTML.md says py3.10;
#    this box is 3.12 — use your venv's actual site-packages path).
printf '%s\n%s\n' \
  $TT_METAL_HOME/tt-train/sources/ttml \
  $TT_METAL_HOME/build/tt-train/sources/ttml \
  > <venv>/lib/python3.12/site-packages/ttml-custom.pth
```

Verify:

```bash
python -c "import ttml, ttnn; print('ttml + ttnn OK')"
```

## Gotchas (all hit and resolved during verification)

- **No `tt-train/pyproject.toml` in this tree.** The `pip install .` path does
  **not** apply — tt-train builds as a tt-metal subproject and `ttml` is wired
  via the `.pth` above.

- **`std::bad_cast` on `import ttml`.** Happens whenever `ttnn` was built
  *before* tt-train was enabled — i.e. every pre-built tt-metal image, including
  TT-QuietBox 2. Cause: a nanobind `STABLE_ABI` tag mismatch between the old
  `_ttnn.so` and the new `_ttml`, so ttml can't see ttnn's `Layout` / `DataType`
  enum registry. **Fix:** rebuild `_ttnn.so` (step 3) so both share the stable
  ABI — or do a single clean `build_metal.sh --build-tt-train` pass. A partial
  `--target _ttml` build **alone is not enough**. (`import ttnn` was re-verified
  afterward.)

- **Extra env vars beyond the usual `TT_METAL_HOME`.** The example aborts
  immediately without **`TT_METAL_RUNTIME_ROOT`** (set it equal to
  `TT_METAL_HOME`), and Blackhole needs **`TT_METAL_ARCH_NAME=blackhole`**
  (`wormhole_b0` on N-series). `TT_LOGGER_LEVEL=FATAL` keeps the log quiet.

- **Board reset.** The first on-device run may hit an ethernet-core timeout at
  device open on p300c / TT-QuietBox 2. Run `tt-smi -r` once and retry.

- **Let ttml close the device.** A killed or malformed script that touches the
  device without a clean close triggers a benign teardown abort in
  `MetalContext::destroy_all_instances`. The training runner closes the device
  in a `finally:` block for this reason.

## Confirmed-importable ttml submodules

`autograd`, `ops` (loss, attention, layernorm, linear, embedding, unary,
binary, dropout, multi_head_utils, reshape), `optimizers`, `models`, `modules`,
`core`, `init`, `fsdp`, `Mesh`.

## Honest caveat for the lesson

Upstream CI `GTEST_SKIP`s tt-train's `softmax`, `cross_entropy` (fwd+bwd),
`rmsnorm`, and `sdpa` tests on P100/P150, so upstream makes **no** Blackhole
training guarantee. We built `ttml` against `~/tt-metal` v0.73 and actually ran
the from-scratch loop on the p300c — those "skipped" ops all executed
correctly. So the lesson can claim from-scratch training works on Blackhole
p300c, framed honestly: **upstream doesn't CI this on BH; we verified it at
v0.73 — pin your version and reset the board if needed.**
