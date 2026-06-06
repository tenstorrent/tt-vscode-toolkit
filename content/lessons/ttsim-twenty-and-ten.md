---
id: ttsim-twenty-and-ten
title: "Twenty-and-Ten Things You Can Do with ttsim"
description: >-
  31 things you can do with the ttsim hardware simulator — no Tenstorrent
  device required. Runs on any Linux machine, including WSL2 on Windows.
  Escalates from first kernel to DSP prototyping to a cliffhanger only
  real hardware can resolve.
category: advanced
tags:
  - ttsim
  - simulator
  - metalium
  - kernels
  - architecture
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
estimatedMinutes: 60
---

# Twenty-and-Ten Things You Can Do with ttsim

ttsim is a hardware-accurate functional simulator for Tenstorrent Wormhole and Blackhole
chips. It ships as a single `.so` file that plugs into TT-Metalium via an environment
variable. Every kernel that compiles for silicon compiles for the simulator. Results are
bit-exact. It runs on any Linux/x86_64 machine, including WSL2 on Windows.

This lesson is self-contained. Setup is below. No Tenstorrent hardware required.

> **Have hardware?** The simulator is still useful for debugging, architecture
> exploration, and running experiments without tying up a device.

---

## Setup

[⚙ Set Up ttsim](command:tenstorrent.setupTtsim)

Or manually:

```bash
mkdir -p ~/sim
TTSIM_VERSION=v1.5.4

# Download Wormhole and Blackhole simulators
wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_wh.so \
     -O ~/sim/libttsim_wh.so
wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_bh.so \
     -O ~/sim/libttsim_bh.so

# Copy the SOC descriptor for Wormhole (switch for Blackhole in entries 3 and 27)
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml

# Required env vars — set these before running any entry below
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

**Prerequisite:** tt-metal must be installed and built.
If you haven't done that yet, start with the
[build tt-metal lesson](command:tenstorrent.showLesson?["build-tt-metal"]) first.

All examples below run from `$TT_METAL_HOME` unless noted.

---

## The Twenty

### 1. Run Tenstorrent on Windows

WSL2 + `libttsim_wh.so`. Set the three env vars above inside a WSL2 session and every
entry in this lesson works. No hardware. No special drivers. No silicon anywhere in the
chain.

```bash
# In a WSL2 terminal on Windows:
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
# Then run any entry in this lesson
```

---

### 2. Hello, RISC-V

`add_2_integers_in_riscv` dispatches a kernel onto the BRISC (data-movement RISC-V core)
of a virtual Tensix. Two integers added together. Real RISC-V ISA. Real dispatch path.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0]], "color": "teal", "label": "BRISC — data movement RISC-V", "ms": 800 },
  { "step": "label", "core": [0,0], "text": "DISPATCH" },
  { "step": "pause", "ms": 400 },
  { "step": "label", "core": [0,0], "text": "ADD" },
  { "step": "pause", "ms": 600 },
  { "step": "label", "core": [0,0], "text": "DONE" },
  { "step": "pause", "ms": 400 }
]
```

```bash
cd $TT_METAL_HOME
./build/programming_examples/metal_example_add_2_integers_in_riscv
```

```text
Finished: Add 2 integers in RISC-V
```

---

### 3. Own both chips for free

Download both `.so` files (the setup above does this). Switch architectures by changing
one environment variable and replacing the SOC descriptor.

```bash
# Switch to Blackhole (140-core SOC)
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so

./build/programming_examples/metal_example_add_2_integers_in_riscv

# Switch back to Wormhole
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
```

```tensix_viz arch=blackhole
[
  { "step": "highlight", "cores": [[0,0],[1,0],[2,0]], "color": "pink", "label": "Blackhole — 140-core SOC", "ms": 800 },
  { "step": "pause", "ms": 500 },
  { "step": "highlight", "cores": [[0,0]], "color": "teal", "label": "Running add_2_integers_in_riscv", "ms": 600 }
]
```

---

### 4. Talk to the compute engine

The compute RISC-V (TRISC) is a separate processor from the data-movement RISC-V.
`hello_world_compute_kernel` dispatches a kernel specifically to the TRISC.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0]], "color": "teal", "label": "Tensix core", "ms": 500 },
  { "step": "label", "core": [0,0], "text": "TRISC0" },
  { "step": "pause", "ms": 800 },
  { "step": "label", "core": [0,0], "text": "HELLO" },
  { "step": "pause", "ms": 600 }
]
```

```bash
./build/programming_examples/metal_example_hello_world_compute_kernel
```

```text
Hello World! TRISC0 results are correct!
```

---

### 5. Elementary school math on an AI accelerator

2 + 3 = 5, dispatched through a chip designed to run large language models. The full
dispatch path — host program, command queue, kernel compilation, BRISC/TRISC execution —
for a trivial operation.

```bash
./build/programming_examples/metal_example_add_2_integers_in_compute
```

```text
Finished: Add 2 integers in compute kernel
```

---

### 6. Invoke the Special Function Processing Unit

The SFPU is a vector unit inside each Tensix core that performs transcendental functions
as native hardware operations — `exp`, `log`, `sqrt`, `gelu`. These are silicon opcodes,
not library calls.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0]], "color": "teal", "label": "SFPU — transcendental ops", "ms": 700 },
  { "step": "label", "core": [0,0], "text": "exp(x)" },
  { "step": "pause", "ms": 500 },
  { "step": "label", "core": [0,0], "text": "sqrt(x)" },
  { "step": "pause", "ms": 500 },
  { "step": "label", "core": [0,0], "text": "gelu(x)" },
  { "step": "pause", "ms": 500 }
]
```

```bash
./build/programming_examples/metal_example_eltwise_sfpu
```

```text
Finished: Eltwise SFPU
```

---

### 7. Chain SFPU ops into a pipeline

`sfpu_eltwise_chain` runs a sequence of SFPU operations on a tile without intermediate
results touching DRAM. The values stay in the register file between steps. This is how
softmax is computed on Tensix hardware.

```bash
./build/programming_examples/metal_example_sfpu_eltwise_chain
```

```text
Finished: SFPU eltwise chain
```

---

### 8. The kernel that runs when you're watching is not the kernel that runs when you're not

`TT_METAL_DPRINT_CORES` is checked at kernel compilation time — not at runtime. Setting
it changes what code gets compiled into the kernel binary. The observation changes the
experiment.

```bash
# Without DPRINT: standard kernel binary, no instrumentation
./build/programming_examples/metal_example_hello_world_datamovement_kernel

# With DPRINT: a different kernel binary is compiled and dispatched
export TT_METAL_DPRINT_CORES=0,0
export TT_METAL_DPRINT_RISCVS=BR
./build/programming_examples/metal_example_hello_world_datamovement_kernel
unset TT_METAL_DPRINT_CORES TT_METAL_DPRINT_RISCVS
```

The second invocation prints from inside the running kernel. The first does not — the
instrumentation was never compiled in.

---

### 9. Operate on 1,024 values simultaneously

A tile is a 32×32 array of bfloat16 values. `eltwise_binary` adds, subtracts, or
multiplies every element in a single dispatched operation.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0]], "color": "teal", "label": "32×32 tile — 1,024 bfloat16 values", "ms": 700 },
  { "step": "label", "core": [0,0], "text": "ADD×1024" },
  { "step": "pause", "ms": 700 },
  { "step": "label", "core": [0,0], "text": "DONE" },
  { "step": "pause", "ms": 400 }
]
```

```bash
./build/programming_examples/metal_example_eltwise_binary
```

```text
Finished: Eltwise binary
```

---

### 10. Run the matmul that powers everything

Matrix multiplication is the fundamental operation of transformer inference.
`matmul_single_core` runs it on one core, start to finish, in tile layout.

```bash
./build/programming_examples/metal_example_matmul_single_core
```

```text
Finished: Single core matmul
```

---

### 11. Light up the grid

`matmul_multi_core` distributes the same matrix multiplication across multiple cores.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0]], "color": "teal", "label": "Core 0 — dispatch", "ms": 400 },
  { "step": "highlight", "cores": [[1,0],[2,0],[3,0]], "color": "teal", "label": "Cores 1–3 activated", "ms": 400 },
  { "step": "highlight", "cores": [[4,0],[5,0],[6,0],[7,0]], "color": "teal", "label": "Cores 4–7 activated", "ms": 400 },
  { "step": "pause", "ms": 600 }
]
```

```bash
./build/programming_examples/metal_example_matmul_multi_core
```

```text
Finished: Multicore matmul
```

---

### 12. Why SRAM reuse is the whole secret

`matmul_multicore_reuse` keeps weight tiles in L1 SRAM across multiple output tiles
instead of re-fetching from DRAM. This is the optimization that closes the gap between
raw FLOP capacity and memory bandwidth on Tensix hardware.

```tensix_viz arch=wormhole
[
  { "step": "transfer", "from": [0,6], "to": [0,0], "ms": 600 },
  { "step": "label", "core": [0,0], "text": "weights in L1" },
  { "step": "pause", "ms": 300 },
  { "step": "label", "core": [0,0], "text": "tile 1 out" },
  { "step": "pause", "ms": 300 },
  { "step": "label", "core": [0,0], "text": "tile 2 out" },
  { "step": "pause", "ms": 300 },
  { "step": "label", "core": [0,0], "text": "tile 3 out" },
  { "step": "pause", "ms": 400 }
]
```

```bash
./build/programming_examples/metal_example_matmul_multicore_reuse
```

```text
Finished: Multicore matmul with reuse
```

---

### 13. Spread a vector add across every core

`vecadd_multi_core` gives every core a slice of the input. All cores compute
simultaneously.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[0,1],[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1]], "color": "teal", "label": "All cores — parallel vector add", "ms": 800 },
  { "step": "pause", "ms": 500 }
]
```

```bash
./build/programming_examples/metal_example_vecadd_multi_core
```

```text
Finished: Vec add multi-core
```

---

### 14. Stripe data across DRAM banks

`vecadd_sharding` distributes tensor data across multiple DRAM channels on the same chip.
A single Tensix chip has multiple DRAM banks and benefits from using all of them.

```bash
./build/programming_examples/metal_example_vecadd_sharding
```

```text
Finished: Vec add sharding
```

---

### 15. Send a tile across the mesh interconnect

`noc_tile_transfer` moves a tile from core (0,0) to core (0,1) via the on-chip network.
No CPU involvement after dispatch. The tile travels the NoC and arrives.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0]], "color": "teal", "label": "Source — core (0,0)", "ms": 500 },
  { "step": "transfer", "from": [0,0], "to": [0,1], "ms": 900 },
  { "step": "highlight", "cores": [[0,1]], "color": "pink", "label": "Destination — core (0,1)", "ms": 500 },
  { "step": "pause", "ms": 400 }
]
```

```bash
./build/programming_examples/metal_example_noc_tile_transfer
```

```text
Finished: NoC tile transfer
```

---

### 16. Write a custom SFPU instruction

`custom_sfpi_add` is hand-authored SFPI assembly — the instruction set of the SFPU
functional unit. This is ISA-level code for a production AI accelerator.

```bash
./build/programming_examples/metal_example_custom_sfpi_add
```

```text
Finished: Custom SFPI add
```

---

### 17. Implement smoothstep in SFPU assembly

`custom_sfpi_smoothstep` implements the smoothstep interpolation function — a standard
graphics shader primitive — as SFPU opcodes. The function has no relationship to AI
inference. Running it on a Tenstorrent chip is completely unnecessary and entirely
possible.

```bash
./build/programming_examples/metal_example_custom_smoothstep
```

```text
Finished: Custom SFPI smoothstep
```

---

### 18. Dispatch a program to a mesh

`1_distributed_program_dispatch` uses the mesh device API. The code is structurally
identical to single-device dispatch — the API scales, and so does the program.

```bash
./build/programming_examples/distributed/distributed_program_dispatch
```

```text
Finished: Distributed program dispatch
```

---

### 19. Read and write distributed buffers

`2_distributed_buffer_rw` manages memory across a virtual mesh. Every tensor-parallel
model does this operation millions of times per inference.

```bash
./build/programming_examples/distributed/distributed_buffer_rw
```

```text
Finished: Distributed buffer read/write
```

---

### 20. The primitive of tensor parallelism

`3_distributed_eltwise_add` performs an element-wise addition across a virtual mesh.
Splitting a tensor across devices, computing in parallel, gathering results — this is
the building block that lets a model span multiple chips.

```bash
./build/programming_examples/distributed/distributed_eltwise_add
```

```text
Finished: Distributed eltwise add
```

---

## The Ten

### 21. Trace async execution without a profiler

`4_distributed_trace_and_events` instruments async barriers and event timelines across a
virtual mesh. The shape of the execution trace matches hardware. The timings do not.

```bash
./build/programming_examples/distributed/distributed_trace_and_events
```

```text
Finished: Distributed trace and events
```

---

### 22. Trigger intentional `UndefinedBehavior` and read the named error

Write a kernel that violates an ISA contract. The simulator halts with a named,
categorized error. On silicon, the same code would likely produce silently incorrect
output.

The simulator is more strict than the hardware on purpose. Error categories from the
documentation:

- `UndefinedBehavior` — violates ISA contract
- `UnpredictableValueUsed` — result is architecture-defined as unpredictable
- `NonContractualBehavior` — relies on behavior not guaranteed by the spec
- `UnimplementedFunctionality` — feature not yet in the simulator
- `AssertionFailure` — internal simulator bug (file an issue)

To trigger one: set `TT_METAL_DISABLE_SFPLOADMACRO=0` (re-enable the unsupported macro)
and run any SFPU example. The simulator will report `UnimplementedFunctionality` for the
`SFPLOADMACRO` instruction. Silicon would execute it silently.

```bash
unset TT_METAL_DISABLE_SFPLOADMACRO
./build/programming_examples/metal_example_eltwise_sfpu 2>&1 | grep -i "unimplemented\|undefined\|error" | head -5
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

---

### 23. Multicast to a core rectangle in one shot

The `multicast` example sends a value to every core in a rectangular range
simultaneously. This is the mechanism behind weight broadcasting in large matrix
multiplications — one sender, all receivers, a single NoC transaction.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[0,0]], "color": "pink", "label": "Source — single sender", "ms": 500 },
  { "step": "transfer", "from": [0,0], "to": [1,0], "ms": 300 },
  { "step": "transfer", "from": [0,0], "to": [2,0], "ms": 300 },
  { "step": "transfer", "from": [0,0], "to": [3,0], "ms": 300 },
  { "step": "transfer", "from": [0,0], "to": [1,1], "ms": 300 },
  { "step": "transfer", "from": [0,0], "to": [2,1], "ms": 300 },
  { "step": "transfer", "from": [0,0], "to": [3,1], "ms": 300 },
  { "step": "highlight", "cores": [[1,0],[2,0],[3,0],[1,1],[2,1],[3,1]], "color": "teal", "label": "Rectangle — all received", "ms": 600 },
  { "step": "pause", "ms": 400 }
]
```

```bash
./build/programming_examples/contributed/multicast/multicast
```

```text
Finished: Multicast
```

---

### 24. Run the transformer attention kernel

`matmul_multicore_reuse` keeps weight tiles in L1 SRAM across multiple output tiles.
This is the core optimization that drives transformer attention layers — weights loaded
once, used many times across a grid of output cores.

```tensix_viz arch=wormhole
[
  { "step": "transfer", "from": [0,6], "to": [0,0], "ms": 500 },
  { "step": "label", "core": [0,0], "text": "weights in L1" },
  { "step": "transfer", "from": [0,0], "to": [1,0], "ms": 300 },
  { "step": "transfer", "from": [0,0], "to": [2,0], "ms": 300 },
  { "step": "transfer", "from": [0,0], "to": [3,0], "ms": 300 },
  { "step": "highlight", "cores": [[1,0],[2,0],[3,0]], "color": "teal", "label": "Compute cores — L1 reuse across output tiles", "ms": 600 },
  { "step": "pause", "ms": 400 }
]
```

```bash
./build/programming_examples/metal_example_matmul_multicore_reuse
```

```text
Finished: Multicore matmul with reuse
```

---

### 25. Produce a bit-exact NaN and verify the bit pattern

ttsim guarantees bit-exact results for all operations, including the precise bit
representation of NaN values. Divide bfloat16 zero by zero. Check the bit pattern
against the ISA specification. If you have hardware available, compare the two — they
match.

```python
import struct
import ttnn
import torch

device = ttnn.open_device(device_id=0)
zero = ttnn.from_torch(torch.zeros(32, 32, dtype=torch.bfloat16),
                       layout=ttnn.TILE_LAYOUT, device=device)
result = ttnn.div(zero, zero)
result_cpu = ttnn.to_torch(ttnn.from_device(result)).float()
actual = result_cpu[0, 0].item()
print(f"Result: {actual}")
print(f"Is NaN: {actual != actual}")
ttnn.close_device(device)
```

---

### 26. Measure kernel dispatch cost vs. kernel run cost

The profiler examples include `test_custom_cycle_count_slow_dispatch`, which uses
software cycle instrumentation inside a kernel to measure how much time is spent
dispatching versus executing.

Note: hardware performance counter values (cycle timers, performance monitors) are
intentionally divergent on the simulator — the README states this explicitly. Software
cycle counting inside kernels still works.

```bash
./build/test/tt_metal/profiler/test_custom_cycle_count_slow_dispatch
```

The ratio of dispatch overhead to execution time at this workload size tells you when a
kernel is too small to schedule efficiently.

---

### 27. Simulate Blackhole on a machine that has never seen Blackhole

Switch to `libttsim_bh.so` and run `matmul_multicore_reuse` against the Blackhole
SOC descriptor. Your machine is now running kernels compiled for a 140-core Blackhole chip.

```bash
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so

./build/programming_examples/metal_example_matmul_multicore_reuse

# Switch back
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
```

Some Wormhole-specific code fails on the Blackhole simulator. The error messages name
the divergence. Debug it without touching a P-series card.

---

### 28. Find the race condition the simulator catches but silicon hides

Write a two-kernel program where the second kernel reads a buffer the first kernel
writes, with no synchronization barrier between them. On silicon this probably passes.
The hardware evaluates operations in a consistent order that happens to be correct for
this workload, nearly every time. On the simulator, the README states: "ttsim may
evaluate operations in any order permitted by software synchronization. This may include
operation orders that are extremely unlikely on silicon."

```bash
# Deploy the demo script
mkdir -p ~/tt-scratchpad/ttsim
# Copy from the tt-vscode-toolkit checkout (adjust TOOLKIT_DIR to match yours):
TOOLKIT_DIR="${TOOLKIT_DIR:-~/code/tt-vscode-toolkit}"
cp $TOOLKIT_DIR/content/templates/ttsim/ttsim_race_demo.py ~/tt-scratchpad/ttsim/

export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
python3 ~/tt-scratchpad/ttsim/ttsim_race_demo.py
```

```text
With barrier:    CORRECT
Without barrier: CORRECT

NOTE: both paths ship with synchronization in place.
Exercise: in run_without_barrier(), remove the ttnn.from_device() call
on the line marked '# remove this line to race', then run again.
The simulator may produce 'WRONG (race detected)' where silicon would pass.
```

Follow the exercise in the script. The barrier was always necessary. The simulator shows
you why.

---

### 29. Use the SFPU as a DSP core

The SFPU — 16-lane SIMD, bfloat16 and fp32, native transcendental functions — has the
same computational structure as a DSP block in a custom silicon design. Implement a
second-order IIR (biquad) filter in Python/TTNN, run it on a test signal, and
characterize bfloat16 numerical error against a float64 reference.

```bash
mkdir -p ~/tt-scratchpad/ttsim
# Copy from the tt-vscode-toolkit checkout (adjust TOOLKIT_DIR to match yours):
TOOLKIT_DIR="${TOOLKIT_DIR:-~/code/tt-vscode-toolkit}"
cp $TOOLKIT_DIR/content/templates/ttsim/ttsim_biquad_kernel.py ~/tt-scratchpad/ttsim/
python3 ~/tt-scratchpad/ttsim/ttsim_biquad_kernel.py
```

```text
Biquad filter: 1024 samples
bfloat16 max error vs float64 reference: 0.0089
PASSED
```

The ISA documentation (`tt-isa-documentation` on GitHub) describes the full SFPU
instruction encoding, register file, and opcode table. If you are designing a DSP chip
or custom accelerator and want a verified functional model of a pipelined transcendental
SIMD unit to drive your RTL requirements, this is one. You would not be taping out a
Tensix core. You would be using a working functional model to characterize an algorithm
before your RTL team writes a line of Verilog.

---

### 30. Run a transformer layer through the simulator

A transformer attention layer requires Q/K/V projections (linear), scaled dot-product
attention (batched matmul), softmax (SFPU chain), and output projection (linear). Every
one of these is a confirmed working TTNN operation in slow dispatch mode. The following
script implements one attention head — no model download, no HuggingFace token, no
weight file.

[▶ Run Transformer Attention on ttsim](command:tenstorrent.runTtsimAttention)

Or manually:

```bash
mkdir -p ~/tt-scratchpad/ttsim
# Copy from the tt-vscode-toolkit checkout (adjust TOOLKIT_DIR to match yours):
TOOLKIT_DIR="${TOOLKIT_DIR:-~/code/tt-vscode-toolkit}"
cp $TOOLKIT_DIR/content/templates/ttsim/ttsim_attention.py ~/tt-scratchpad/ttsim/
python3 ~/tt-scratchpad/ttsim/ttsim_attention.py
```

```text
Attention output shape: torch.Size([1, 32, 64])
PCC vs PyTorch reference: 0.999847
PASSED
```

The output is correct. Verified against the PyTorch reference. Running on a chip that
does not exist in this machine.

When the hardware arrives, the question is not whether this works. You already know it
works. The question is how fast.

---

### 31. One more thing

`matmul_multicore_reuse` on the simulator takes several seconds. On a P300c it
takes milliseconds. On a QuietBox with four P300cs, less than that.

Two things the simulator cannot give you.

First: the performance counter values. Reads from hardware cycle counters and performance
monitors return values the README explicitly marks as divergent. The simulator does not
model real-time execution.

Second: fast dispatch. `TT_METAL_SLOW_DISPATCH_MODE=1` is required in the simulator.
The fast dispatch path is not yet implemented. On hardware, turning off slow dispatch
mode is the moment the architecture behaves differently. The dispatch overhead collapses.
The ratio you measured in entry 26 changes by an order of magnitude.

There is a third thing, harder to describe. The biquad filter in entry 29 runs in the
simulator. On silicon, with fast dispatch enabled, 1,024 samples of biquad filtering at
bfloat16 precision completes in a time that has no analogue in software. The same
arithmetic. The same bit patterns. A different physical reality.

The simulator gave you the model. Hardware gives you the thing.

---

## What You Learned

- ✅ **ttsim setup**: both Wormhole and Blackhole simulators running on any Linux machine
- ✅ **Kernel dispatch**: RISC-V data-movement and compute paths, DPRINT observer effect
- ✅ **SFPU operations**: native transcendental functions, custom SFPI assembly, DSP use
- ✅ **Memory hierarchy**: L1 reuse, DRAM sharding, NoC tile transfer
- ✅ **Multi-core patterns**: grid dispatch, multicast, distributed mesh
- ✅ **Simulator strictness**: named error categories, race detection, bit-exact NaN
- ✅ **Architecture exploration**: Wormhole vs Blackhole without owning either

**Ready for hardware?** Start with
[verifying your installation](command:tenstorrent.showLesson?["verify-installation"])
to confirm your device is operational, then return here and run entry 31 again.
