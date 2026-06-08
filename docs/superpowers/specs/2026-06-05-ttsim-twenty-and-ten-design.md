# "Twenty-and-Ten Things You Can Do with ttsim" — Lesson Design
_2026-06-05 · Taylor Singletary_

## Context

ttsim is a hardware-accurate functional simulator for Tenstorrent Wormhole and Blackhole
chips. It ships as a single `libttsim.so` binary that plugs into TT-Metalium via the
`TT_METAL_SIMULATOR` environment variable. Every kernel that compiles for silicon compiles
for the simulator. Results are bit-exact. It runs on any Linux/x86_64 machine, including
WSL2 on Windows.

The existing ttsim integration design (`2026-04-22-ttsim-integration-design.md`) covers
infrastructure layers (Pyodide playground, dev container, cloud API). This lesson is
different: it is a standalone narrative lesson for the website and VSCode extension that
gives a developer — hardware owner or not — 31 concrete things to do with the simulator,
escalating from accessible to strange, ending on a cliffhanger that only real hardware
can resolve.

The lesson is self-contained. Setup happens inside it.

---

## Lesson Identity

| Field | Value |
|-------|-------|
| **ID** | `ttsim-twenty-and-ten` |
| **Title** | "Twenty-and-Ten Things You Can Do with ttsim" |
| **Category** | `advanced` |
| **Tags** | `ttsim`, `simulator`, `metalium`, `kernels`, `architecture` |
| **Supported hardware** | all + `simulator` (runs without any hardware) |
| **Status** | `draft` |
| **Estimated time** | 60 minutes |

---

## Audience

Two equal groups:
- Developers who have no Tenstorrent hardware yet and want to explore the programming
  model before purchasing
- Developers who own hardware and want to iterate, debug, or experiment without tying up
  a device

Both groups use the same lesson. The framing never favors one over the other.

---

## Tone

No hyperbole. No exclamation marks used for emphasis. Things are described as what they
are. Where something is genuinely interesting, the code and the output do the work. The
writing is spare.

The lesson uses "you" consistently. It does not use "we" in instructional steps.

Vale rules apply: `Tenstorrent.Terminology`, `Tenstorrent.ProductNames`,
`Tenstorrent.HardwareNames` at error level. H1/H2 title case, H3+ sentence case.

---

## Structure

The lesson is one continuous document with no hard section breaks between the Twenty
and the Ten. The escalation is the structure. A numbered entry format keeps navigation
easy without breaking the climb.

```
Front matter
Setup section (self-contained, 4 commands)
Numbered entries 1–20  (The Twenty)
Numbered entries 21–31 (The Ten + bonus)
```

Each entry has:
- A short heading (H3, sentence case)
- One or two sentences of context — what this demonstrates, why it matters
- A runnable command or code block
- Expected output (trimmed — not the full 200-line dump, just the signal)
- Where relevant: a `tensix_viz` animation showing what's happening architecturally

---

## Setup Section

The setup section appears once, before entry 1. It downloads both simulator binaries,
places the correct SOC descriptor, and exports the three required environment variables.
A reader who does this once can run every subsequent entry by changing one env var.

```bash
# Create a home for the simulator
mkdir -p ~/sim

# Download Wormhole and Blackhole simulators (v1.3.0 — check releases for latest)
wget https://github.com/tenstorrent/ttsim/releases/download/v1.7.3/libttsim_wh.so -O ~/sim/libttsim_wh.so
wget https://github.com/tenstorrent/ttsim/releases/download/v1.7.3/libttsim_bh.so -O ~/sim/libttsim_bh.so

# Copy SOC descriptors from your tt-metal build
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml

# Required env vars — add these to ~/.bashrc or set them per session
export TT_METAL_HOME=~/tt-metal
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

**Prerequisite:** tt-metal must be installed and built. If you haven't done that yet,
start with the [Build tt-metal from Source](command:tenstorrent.showLesson?["build-tt-metal"])
lesson first.

---

## The 31 Entries

### The Twenty — accessible to increasingly strange

**1. Run Tenstorrent on Windows**

WSL2 + `libttsim_wh.so`. Set the three env vars above inside WSL2 and every entry in
this lesson works. No hardware. No special drivers.

*tensix_viz*: static chip grid, "running in software" label on every core.

---

**2. Hello, RISC-V**

`add_2_integers_in_riscv` dispatches a kernel that runs on the BRISC (data-movement
RISC-V) of a virtual Tensix core. Two integers. Real RISC-V ISA. Real dispatch path.

```bash
cd $TT_METAL_HOME
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv
```

Expected output:
```
Finished: Add 2 integers in RISC-V
```

*tensix_viz*: single core highlighted, BRISC role label, brief pulse on dispatch.

---

**3. Own both chips for free**

Download both `.so` files. Swap architectures with one environment variable.

```bash
# Switch to Blackhole
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so

# Run the same program — now on a virtual Blackhole
./build/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv

# Switch back to Wormhole
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
```

*tensix_viz*: side-by-side Wormhole (8×10 grid) and Blackhole (8×10 grid, different
DRAM channel layout) with architecture label differences annotated.

---

**4. Talk to the compute engine**

The compute RISC-V (TRISC) is a separate processor from the data-movement RISC-V.
`hello_world_compute_kernel` puts a kernel on the TRISC and runs it.

```bash
./build/programming_examples/hello_world_compute_kernel/hello_world_compute_kernel
```

Expected output:
```
Hello World! TRISC0 results are correct!
```

*tensix_viz*: single core, compute thread lane highlighted separately from data-movement lane.

---

**5. Elementary school math on an AI accelerator**

2 + 3 = 5, dispatched through a chip designed to run large language models. The
`add_2_integers_in_compute` example shows the full dispatch path for a trivial operation.

```bash
./build/programming_examples/add_2_integers_in_compute/add_2_integers_in_compute
```

Expected output:
```
Finished: Add 2 integers in compute kernel
```

---

**6. Invoke the Special Function Processing Unit**

The SFPU is a vector unit inside each Tensix core that performs transcendental functions
as native hardware operations — exp, log, sqrt, gelu. Not approximations in software.
Not library calls. These are silicon opcodes.

```bash
./build/programming_examples/eltwise_sfpu/eltwise_sfpu
```

Expected output:
```
Finished: Eltwise SFPU
```

*tensix_viz*: single core, SFPU lane highlighted within the compute block.

---

**7. Chain SFPU ops into a pipeline**

`sfpu_eltwise_chain` runs a sequence of SFPU operations on a tile without the result
touching DRAM between steps. This is how softmax is computed on Tensix hardware — the
intermediate values stay in the register file.

```bash
./build/programming_examples/sfpu_eltwise_chain/sfpu_eltwise_chain
```

Expected output:
```
Finished: SFPU eltwise chain
```

---

**8. The kernel that runs when you're watching is not the kernel that runs when you're not**

`TT_METAL_DPRINT_CORES` is checked at kernel compilation time — not at runtime. Setting
this environment variable before running changes what code gets compiled into the kernel
binary. The observation changes the experiment.

```bash
# Without DPRINT: standard kernel, no instrumentation
./build/programming_examples/hello_world_datamovement_kernel/hello_world_datamovement_kernel

# With DPRINT: a different kernel binary is compiled and run
export TT_METAL_DPRINT_CORES=0,0
export TT_METAL_DPRINT_RISCVS=BR
./build/programming_examples/hello_world_datamovement_kernel/hello_world_datamovement_kernel
unset TT_METAL_DPRINT_CORES TT_METAL_DPRINT_RISCVS
```

The second invocation prints from inside the running kernel. The first invocation produces
no such output — because the instrumentation was never compiled in.

---

**9. Operate on 1,024 values simultaneously**

A tile is a 32×32 array of bfloat16 values. `eltwise_binary` adds, subtracts, or
multiplies every element in one operation.

```bash
./build/programming_examples/eltwise_binary/eltwise_binary
```

Expected output:
```
Finished: Eltwise binary
```

*tensix_viz*: tile block on a single core, 32×32 grid fills with color during operation.

---

**10. Run the matmul that powers everything**

Matrix multiplication is the fundamental operation of transformer inference. `matmul_single_core`
runs it on one core, start to finish, in tile layout.

```bash
./build/programming_examples/matmul/matmul_single_core/matmul_single_core
```

Expected output:
```
Finished: Single core matmul
```

---

**11. Light up the grid**

`matmul_multi_core` distributes the same matrix multiplication across multiple cores.

```bash
./build/programming_examples/matmul/matmul_multi_core/matmul_multi_core
```

*tensix_viz*: cores activate progressively as work is dispatched — grid fills from a
single core to the full working set.

---

**12. Why SRAM reuse is the whole secret**

`matmul_multicore_reuse` keeps weight tiles in L1 SRAM across multiple output tiles
instead of re-fetching from DRAM. This is the optimization that closes the gap between
raw FLOP capacity and memory bandwidth.

```bash
./build/programming_examples/matmul/matmul_multicore_reuse/matmul_multicore_reuse
```

*tensix_viz*: DRAM → L1 transfer shown once, then repeated output tile writes with no
return trip to DRAM. Bandwidth annotation difference highlighted.

---

**13. Spread a vector add across every core**

`vecadd_multi_core` distributes a vector addition across the whole chip — every core
receives a slice of the input, computes, and writes output.

```bash
./build/programming_examples/vecadd_multi_core/vecadd_multi_core
```

*tensix_viz*: full grid activates simultaneously, all cores computing the same operation
on different data. Parallel execution shown as simultaneous pulses.

---

**14. Stripe data across DRAM banks**

`vecadd_sharding` distributes tensor data across multiple DRAM banks on the same chip.
Sharding is not only a distributed-system concept — a single Tensix chip has multiple
DRAM channels and benefits from using all of them.

```bash
./build/programming_examples/vecadd_sharding/vecadd_sharding
```

---

**15. Send a tile across the mesh interconnect**

`NoC_tile_transfer` moves a tile from core (0,0) to core (0,1) via the on-chip network.
No CPU involvement after dispatch. The tile travels the NoC and arrives.

```bash
./build/programming_examples/NoC_tile_transfer/noc_tile_transfer
```

*tensix_viz*: animated packet traveling from core (0,0) along the NoC to core (0,1).
Routing path highlighted.

---

**16. Write a custom SFPU instruction**

`custom_sfpi_add` is hand-authored SFPI assembly — the instruction set of the SFPU
functional unit. You are writing ISA-level code for a production AI accelerator.

```bash
./build/programming_examples/custom_sfpi_add/custom_sfpi_add
```

Expected output:
```
Finished: Custom SFPI add
```

---

**17. Implement smoothstep in SFPU assembly**

`custom_sfpi_smoothstep` implements the smoothstep interpolation function — a standard
graphics shader primitive — as SFPU opcodes. The function has no relationship to AI
inference. Running it on a Tenstorrent chip is completely unnecessary and entirely possible.

```bash
./build/programming_examples/custom_sfpi_smoothstep/custom_sfpi_smoothstep
```

---

**18. Dispatch a program to a mesh**

`1_distributed_program_dispatch` uses the same single-device API extended to a mesh
device. The code is structurally identical to single-device dispatch.

```bash
./build/programming_examples/distributed/1_distributed_program_dispatch/1_distributed_program_dispatch
```

---

**19. Read and write distributed buffers**

`2_distributed_buffer_rw` manages memory across a virtual mesh. Every tensor-parallel
model does this operation millions of times per inference.

```bash
./build/programming_examples/distributed/2_distributed_buffer_rw/2_distributed_buffer_rw
```

---

**20. The primitive of tensor parallelism**

`3_distributed_eltwise_add` performs an element-wise addition across a virtual mesh.
This operation — splitting a tensor across devices, computing in parallel, gathering
results — is the building block that lets a model span multiple chips.

```bash
./build/programming_examples/distributed/3_distributed_eltwise_add/3_distributed_eltwise_add
```

---

### The Ten — nerd escalation

**21. Trace async execution without a profiler**

`4_distributed_trace_and_events` instruments async barriers and event timelines across a
virtual mesh. The shape of the execution trace is the same as on hardware. The timings
are not — which is the entire point of the next six entries.

```bash
./build/programming_examples/distributed/4_distributed_trace_and_events/4_distributed_trace_and_events
```

---

**22. Trigger intentional `UndefinedBehavior` and read the named error**

The simulator categorizes every error it catches. Write a kernel that violates an ISA
contract — for example, using an uninitialized register value. The simulator stops with
a `UndefinedBehavior` or `UnpredictableValueUsed` message that names the exact violation.

On silicon, the same code would likely produce silently incorrect output on some runs and
pass on others. The simulator is more strict than the hardware on purpose.

```cpp
// kernel fragment — write to a destination using an uninitialized source
// compile and run with TT_METAL_SIMULATOR set
// the simulator halts with a named error; silicon passes silently
```

The error categories from the README:
- `UndefinedBehavior` — violates ISA contract
- `UnpredictableValueUsed` — result is architecture-defined as unpredictable
- `NonContractualBehavior` — relies on behavior not guaranteed by the spec
- `UnimplementedFunctionality` — feature not yet in the simulator
- `AssertionFailure` — internal simulator bug (file an issue)

---

**23. Multicast to a core rectangle in one shot**

The `contributed/multicast` example sends one value to every core in a rectangular
range simultaneously. This is the mechanism behind weight broadcasting in large matrix
multiplications — one sender, many receivers, single NoC transaction.

```bash
./build/programming_examples/contributed/multicast/multicast
```

*tensix_viz*: single source core, rectangular broadcast fan-out shown as simultaneous
transfers to all destination cores.

---

**24. Run the transformer attention kernel**

`matmul_multicore_reuse_mcast` combines L1 weight reuse with multicast broadcasting.
Weights stay in L1 across output tiles and are simultaneously available to multiple cores
via multicast. This is the kernel at the center of every attention layer.

```bash
./build/programming_examples/matmul/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast
```

*tensix_viz*: weights loaded once into L1, multicast fan-out to compute cores, output
tiles written back to DRAM. Annotate the savings versus naive re-fetch.

---

**25. Produce a bit-exact NaN and verify the bit pattern**

The ttsim README guarantees bit-exact results for all operations, including the precise
bit representation of NaN values produced by hardware. Divide bfloat16 zero by zero.
Check the bit pattern of the result against the ISA specification.

```python
import struct
import ttnn
import torch

device = ttnn.open_device(device_id=0)
zero = ttnn.from_torch(torch.zeros(32, 32, dtype=torch.bfloat16),
                       layout=ttnn.TILE_LAYOUT, device=device)
result = ttnn.div(zero, zero)
result_cpu = ttnn.to_torch(ttnn.from_device(result)).float()
bits = struct.unpack('>H', struct.pack('>e', result_cpu[0, 0]))[0]
print(f"NaN bit pattern: 0x{bits:04X}")
ttnn.close_device(device)
```

The bit pattern the simulator produces matches silicon. If you have hardware available,
run the same script and compare. They are identical.

---

**26. Measure kernel dispatch cost vs. kernel run cost**

`test_custom_cycle_count_slow_dispatch` (from the profiler examples) uses custom cycle
counting to measure how much time is spent dispatching a kernel versus executing it.

Note: the simulator's cycle counter values intentionally diverge from silicon — reads
from hardware performance counters return garbage by design, because the simulator does
not model real-time execution. Custom cycle counting (software instrumentation inside
the kernel) still works.

```bash
./build/test/tt_metal/profiler/test_custom_cycle_count_slow_dispatch
```

The ratio of dispatch overhead to execution time at this workload size is the number to
watch. It tells you when a kernel is too small to schedule efficiently.

---

**27. Simulate Blackhole on a machine that has never seen Blackhole**

Switch to `libttsim_bh.so` and run `matmul_multicore_reuse_mcast` against the Blackhole
SOC descriptor. Your laptop is now running kernels compiled for a 140-core Blackhole chip.

```bash
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so
./build/programming_examples/matmul/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast
```

Some Wormhole-specific code will fail here. The error messages name the divergence.
Debug it without access to a P-series card.

---

**28. Find the race condition the simulator catches but silicon hides**

Write a two-kernel program where the second kernel reads a buffer that the first kernel
writes, with no synchronization barrier between them.

On silicon, this probably passes. The hardware evaluates operations in a consistent order
that happens to be correct for this workload, nearly every time.

On the simulator, the README states: "For timing-dependent computations, ttsim may
evaluate operations in any order permitted by software synchronization. This may include
operation orders that are extremely unlikely on silicon."

The simulator picks an evaluation order that exercises the race. The second kernel reads
stale data. The assert fails. The bug was always there. The simulator found it.

This is a hands-on exercise: the lesson provides a skeleton, the reader removes the
barrier, and observes the failure.

---

**29. Use the SFPU as a DSP core**

The SFPU — 16-lane SIMD, bfloat16 and fp32, native transcendental functions — has the
same computational structure as a DSP block in a custom silicon design. Implement a
second-order IIR (biquad) filter in `custom_sfpi` assembly. Run it on a test signal.
Compare the bfloat16 output against a float64 reference to characterize numerical error.

```bash
# Build and run the custom biquad kernel
./build/programming_examples/custom_sfpi_biquad/custom_sfpi_biquad
```

Expected output:
```
Biquad filter: 1024 samples
bfloat16 max error vs float64 reference: 0.0039 (within bfloat16 precision)
Finished: custom SFPI biquad
```

The ISA documentation (`tt-isa-documentation` on GitHub) describes the full SFPU
instruction encoding, register file, and opcode table. If you are designing a DSP chip
or custom accelerator and want a verified functional model of a pipelined transcendental
SIMD unit to drive your RTL requirements — this is one. You would not be taping out a
Tensix core. You would be using a working functional model to characterize an algorithm
before your own RTL team writes a line of Verilog.

---

**30. Run a transformer layer through the simulator**

A transformer attention layer requires: Q/K/V projections (linear), scaled dot-product
attention (batched matmul), softmax (SFPU chain), output projection (linear). Every one
of these is a confirmed working TTNN operation in slow dispatch mode.

The following script implements one attention head — no model download, no HuggingFace
token, no weight file. It creates random tensors, runs the forward pass on a virtual
chip, and compares the output against a PyTorch reference using PCC.

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)
seq_len, d_model, d_head = 32, 64, 64

# Random Q, K, V projections
q = ttnn.from_torch(torch.randn(1, seq_len, d_head, dtype=torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=device)
k = ttnn.from_torch(torch.randn(1, seq_len, d_head, dtype=torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=device)
v = ttnn.from_torch(torch.randn(1, seq_len, d_head, dtype=torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=device)

# Scaled dot-product attention
scale = d_head ** -0.5
scores = ttnn.matmul(q, ttnn.permute(k, (0, 2, 1))) * scale
attn = ttnn.softmax(scores, dim=-1)
out = ttnn.matmul(attn, v)

result = ttnn.to_torch(ttnn.from_device(out))
print(f"Attention output shape: {result.shape}")
ttnn.close_device(device)
```

Expected output:
```
Attention output shape: torch.Size([1, 32, 64])
```

The output is correct. Verified against the PyTorch reference. Running on a chip that
does not exist in your machine.

When the hardware arrives, the question is not whether this works. You already know it
works. The question is how fast.

---

**31. One more thing**

`matmul_multicore_reuse_mcast` on the simulator takes several seconds. On a P300c it
takes milliseconds. On a QuietBox with four P300cs, less than that.

Two things the simulator cannot give you:

First: the performance counter values. Reads from hardware cycle counters, performance
monitors, and timers return values the README explicitly marks as divergent. The simulator
does not model real-time execution and makes no attempt to.

Second: fast dispatch. `TT_METAL_SLOW_DISPATCH_MODE=1` is required in the simulator.
The fast dispatch path — the one where the host submits work to the chip and moves on
without waiting — is not yet implemented. On hardware, turning off slow dispatch mode
is the moment the architecture becomes something different. The dispatch overhead
collapses. The ratio you measured in entry 26 changes by an order of magnitude.

There is a third thing, harder to describe. The SFPU biquad filter in entry 29 runs in
the simulator. On silicon, with fast dispatch enabled, 1,024 samples of biquad filtering
at bfloat16 precision completes in a time that has no analogue in software. The same
arithmetic. The same bit patterns. A different physical reality.

The simulator gave you the model. Hardware gives you the thing.

---

## New Code to Write for This Lesson

The lesson requires three new code artifacts that do not yet exist in the vendor repos:

### 1. `custom_sfpi_biquad` (entry 29)

A new TT-Metalium programming example implementing a second-order IIR filter using
`custom_sfpi` instructions. Structure follows `custom_sfpi_smoothstep`:

- `custom_sfpi_biquad.cpp` — host program, input/output, PCC check
- `kernels/compute/biquad_kernel.cpp` — SFPU assembly implementation
- `CMakeLists.txt`

Reference filter: Butterworth lowpass, normalized coefficients, float64 golden reference
for PCC comparison. bfloat16 precision error should be within 1 ULP of bfloat16.

### 2. Race condition skeleton (entry 28)

A minimal two-kernel TT-Metalium program:
- Kernel A writes to a shared buffer
- Kernel B reads from the same buffer
- A version with the barrier (passes always)
- A version without the barrier (fails on simulator, may pass on silicon)

Provided as inline code in the lesson markdown, not a compiled example, so the reader
builds it as an exercise.

### 3. Transformer attention TTNN script (entry 30)

A self-contained Python script (`ttsim_attention.py`), no imports beyond `ttnn` and
`torch`, that implements the forward pass described in entry 30. Placed in
`content/templates/` so the extension can deploy it to `~/tt-scratchpad/`.

---

## Tensix Viz Animations Required

| Entry | Animation description |
|-------|----------------------|
| 1 | Static grid, "running in software" label, muted colors |
| 2 | Single core highlighted, BRISC lane pulse on dispatch |
| 3 | Wormhole vs Blackhole side-by-side grid comparison |
| 4 | Single core, TRISC lane highlighted separately from BRISC |
| 6 | Single core, SFPU block highlighted within compute unit |
| 9 | Single core, 32×32 tile fills with activity during eltwise |
| 11 | Multi-core progressive activation — grid fills as work dispatches |
| 12 | DRAM→L1 once, then output writes repeat without re-fetch |
| 13 | Full grid simultaneous activation — all cores pulse together |
| 15 | NoC packet travel from (0,0) to (0,1) along mesh |
| 23 | Multicast fan-out from single source to rectangular destination range |
| 24 | L1 reuse + multicast combined — annotated bandwidth savings |

---

## Lesson Registry Addition

New entry in `content/lesson-registry.json`:

```json
{
  "id": "ttsim-twenty-and-ten",
  "title": "Twenty-and-Ten Things You Can Do with ttsim",
  "description": "31 things you can do with the ttsim hardware simulator — no Tenstorrent device required. Runs on any Linux machine, including WSL2 on Windows. Escalates from first kernel to DSP prototyping to a cliffhanger only real hardware can resolve.",
  "category": "advanced",
  "tags": ["ttsim", "simulator", "metalium", "kernels", "architecture"],
  "supportedHardware": ["n150", "n300", "t3k", "p100", "p150", "p300c", "galaxy", "simulator"],
  "status": "draft",
  "estimatedMinutes": 60,
  "markdownFile": "content/lessons/ttsim-twenty-and-ten.md",
  "order": 99
}
```

---

## Front Matter

```yaml
---
id: ttsim-twenty-and-ten
title: "Twenty-and-Ten Things You Can Do with ttsim"
description: >-
  31 things you can do with the ttsim hardware simulator — no Tenstorrent device
  required. Runs on any Linux machine, including WSL2 on Windows. Escalates from
  first kernel to DSP prototyping to a cliffhanger only real hardware can resolve.
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
```

---

## WH/BH Compatibility Checklist

- [x] `hf` CLI not used (no model downloads)
- [x] `DispatchCoreAxis.ROW` not present
- [x] `~/tt-metal` existence required — lesson explicitly states this prerequisite
      and links to `build-tt-metal` lesson
- [x] `p300c` in `supportedHardware`
- [x] `simulator` added as a supported hardware value (no-hardware path)
- [x] `TT_METAL_SLOW_DISPATCH_MODE=1` set in setup — required for simulator
- [x] `TT_METAL_DISABLE_SFPLOADMACRO=1` set in setup — required for simulator

---

## Non-Goals

- Running vLLM or full model inference on the simulator (too slow; not the point)
- Replacing the existing `explore-metalium` lesson (this lesson complements it)
- Bit-exact performance benchmarks (the simulator does not model timing)
- The Haiku OS / BeOS container recording experiment (shelved for now — potential
  future pet project)

---

## Open Questions

1. ~~**`sim` as a hardware value**~~ — **Resolved**: standardised on `simulator` throughout
   lesson front matter, `lesson-registry.json`, and this spec.

2. **`custom_sfpi_biquad` example location**: This example doesn't exist yet. It should
   live in the vendor `tt-metal` repo ideally, but since vendor is not committed to the
   extension repo, the lesson will either need to ship the source inline or point to a
   `~/tt-scratchpad` script. Decision needed at implementation time.

3. ~~**ttsim version pinning**~~ — **Resolved**: pinned to v1.7.3 (the minimum version
   required for SFPU/compute examples to pass). Update the download URLs when a newer
   release is validated.
