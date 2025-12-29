---
id: riscv-programming
title: RISC-V Programming on Tensix Cores
description: >-
  Explore 880 RISC-V processors on a single chip! Program the five RISC-V cores
  (BRISC, NCRISC, TRISC0/1/2) inside each Tensix. Learn bare-metal programming,
  NoC communication, and parallel computing at scale. Includes comprehensive
  exploration guide.
category: advanced
tags: []
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
status: draft
estimatedMinutes: 10
---

# RISC-V Programming on Tensix Cores

## Introduction: 880 RISC-V Processors on a Single Chip

Tenstorrent's Wormhole and Blackhole cards aren't just AI accelerators - they're also massive RISC-V computing platforms. Each Tensix core contains **five RISC-V processors** (RV32IM ISA):

- **BRISC (RISCV_0)** - Data Movement 0: Primary data movement, reads from DRAM/NoC
- **NCRISC (RISCV_1)** - Data Movement 1: Secondary data movement, writes to DRAM/NoC
- **TRISC0** - Unpack: Moves data from L1 SRAM into compute engines
- **TRISC1** - Math: Issues instructions to FPU and SFPU
- **TRISC2** - Pack: Writes results back to L1 SRAM

With 176 Tensix cores on Wormhole, that's **880 RISC-V cores** you can program directly!

### What Makes This Unique?

Unlike typical embedded RISC-V boards (SiFive, ESP32-C3) with 1-5 cores, Tenstorrent hardware offers:

- ✅ **Massive parallelism** - Hundreds of RISC-V cores working together
- ✅ **Near-memory compute** - 1.5MB L1 SRAM per Tensix core
- ✅ **High-bandwidth interconnect** - Network-on-Chip (NoC) mesh at 100+ GB/s aggregate
- ✅ **Bare-metal programming** - No OS, no hidden behavior, direct hardware access
- ✅ **Explicit communication** - Learn distributed systems with real hardware

This lesson walks you through running a simple RISC-V program on Tensix cores, then provides resources for deeper exploration.

---

## Step 1: Build Programming Examples

tt-metal includes programming examples that demonstrate low-level RISC-V programming. Let's build them:

**What this does:**
- Builds tt-metal with `--build-programming-examples` flag
- Compiles example kernels including `add_2_integers_in_riscv`
- Takes 5-10 minutes (one-time build)

**Command:**
```bash
cd ~/tt-metal && \
  ./build_metal.sh --build-programming-examples
```

[🔨 Build Programming Examples](command:tenstorrent.buildProgrammingExamples)

**Note:** If you've already built tt-metal, this will rebuild with the examples flag enabled.

---

## Step 2: Run RISC-V Addition Example

Now let's run the canonical RISC-V example: adding two integers on a BRISC processor.

**What this example does:**
1. **Host** uploads two integers (14 and 7) to DRAM
2. **BRISC (RISCV_0)** on Tensix core (0,0):
   - Reads integers from DRAM → L1 SRAM (via NoC DMA)
   - Executes RISC-V `add` instruction: `14 + 7 = 21`
   - Writes result to DRAM (via NoC DMA)
3. **Host** reads back result (21)

**Architecture diagram:**
```text
┌──────────────┐
│     Host     │  Uploads: 14, 7
│   (C++ API)  │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────┐
│  Tensix Core (0,0)             │
│  ┌──────────────────────────┐  │
│  │  BRISC (Data Movement)   │  │
│  │  RV32IM Processor        │  │
│  │                          │  │
│  │  1. DMA: DRAM → L1       │  │
│  │  2. lw t0, 0(a0)  # 14   │  │  ← RISC-V loads
│  │     lw t1, 0(a1)  # 7    │  │
│  │  3. add t2, t0, t1       │  │  ← RISC-V addition!
│  │  4. sw t2, 0(a2)  # 21   │  │  ← RISC-V store
│  │  5. DMA: L1 → DRAM       │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
       │
       ▼
┌──────────────┐
│     Host     │  Reads back: 21
└──────────────┘
```

**Enable debug output** (optional, to see messages from RISC-V core):
```bash
export TT_METAL_DPRINT_CORES=0,0
```

**Run the example:**
```bash
cd ~/tt-metal && \
  export TT_METAL_DPRINT_CORES=0,0 && \
  ./build/programming_examples/add_2_integers_in_riscv
```

[🚀 Run RISC-V Addition Example](command:tenstorrent.runRiscvExample)

**Expected output:**
```bash
Adding integers: 14 + 7
Success: Result is 21
```

---

## Step 3: Explore the RISC-V Kernel Code

Let's look at the actual RISC-V kernel that runs on the BRISC processor.

**File location:**
```bash
~/tt-metal/tt_metal/programming_examples/add_2_integers_in_riscv/kernels/
  reader_writer_add_in_riscv.cpp
```

**Open the kernel in VS Code:**

[📖 Open RISC-V Kernel Source](command:tenstorrent.openRiscvKernel)

**Key sections to examine:**

### 1. Runtime Arguments (Passed from Host)
```cpp
void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);  // DRAM buffer addresses
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram  = get_arg_val<uint32_t>(2);
    uint32_t src0_l1   = get_arg_val<uint32_t>(3);  // L1 buffer addresses
    uint32_t src1_l1   = get_arg_val<uint32_t>(4);
    uint32_t dst_l1    = get_arg_val<uint32_t>(5);
    // ...
}
```

### 2. NoC DMA Operations (Reading from DRAM)
```cpp
    // Calculate NoC addresses (X,Y coordinates + offset)
    uint64_t src0_dram_noc_addr = get_noc_addr(0, src0);
    uint64_t src1_dram_noc_addr = get_noc_addr(0, src1);

    // Asynchronous DMA read from DRAM to L1 SRAM
    noc_async_read(src0_dram_noc_addr, src0_l1, sizeof(uint32_t));
    noc_async_read(src1_dram_noc_addr, src1_l1, sizeof(uint32_t));
    noc_async_read_barrier();  // Wait for DMA to complete
```

### 3. THE RISC-V ADDITION (Runs on BRISC Processor)
```cpp
    // Cast L1 addresses to pointers
    uint32_t* dat0 = (uint32_t*)src0_l1;  // Points to L1 SRAM
    uint32_t* dat1 = (uint32_t*)src1_l1;
    uint32_t* out0 = (uint32_t*)dst_l1;

    // This compiles to RISC-V assembly:
    //   lw   t0, 0(a0)    # Load *dat0
    //   lw   t1, 0(a1)    # Load *dat1
    //   add  t2, t0, t1   # Add!
    //   sw   t2, 0(a2)    # Store result
    (*out0) = (*dat0) + (*dat1);

    DPRINT << "Adding integers: " << *dat0 << " + " << *dat1 << "\n";
```

### 4. Writing Result Back to DRAM
```cpp
    // DMA write from L1 back to DRAM
    uint64_t dst_dram_noc_addr = get_noc_addr(0, dst);
    noc_async_write(dst_l1, dst_dram_noc_addr, sizeof(uint32_t));
    noc_async_write_barrier();  // Wait for write to complete
}
```

**What's happening:**
- The C++ code is compiled to RISC-V machine code by `riscv32-gcc`
- The addition becomes a native RISC-V `add` instruction
- The BRISC processor executes this instruction directly
- NoC DMA operations move data between DRAM and L1 SRAM

---

## Step 4: Understanding the Memory Architecture

### Memory Regions (Wormhole/Blackhole)

```text
┌─────────────────────────────────────┐
│  L1 SRAM (1.5 MB per Tensix)       │  ← Shared by all 5 RISC-V cores
│  Base: 0x00000000                  │     Fast access, user-managed
│  Size: 1,464 KB                    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  BRISC Local Memory (4 KB)         │  ← Private to BRISC processor
│  Base: 0xFFB00000                  │     Stack, local variables
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  NCRISC Local Memory (4 KB)        │  ← Private to NCRISC processor
│  Base: 0xFFB01000                  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  NCRISC IRAM (16 KB)               │  ← Instruction RAM for NCRISC
│  Base: 0xFFC00000                  │     Fast instruction fetch
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  DRAM (1 GB per chip)              │  ← Accessed via NoC DMA only
│  Not directly addressable          │     Requires noc_async_read/write
└─────────────────────────────────────┘
```

### The Network-on-Chip (NoC)

The NoC is a 2D mesh interconnect that connects all Tensix cores, DRAM controllers, PCIe, and Ethernet:

```text
Wormhole NoC Grid:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ D │ T │ T │ T │ T │ T │ T │ T │ T │ T │ T │ D │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ D │ T │ T │ T │ T │ T │ T │ T │ T │ T │ T │ D │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ D │ T │ T │ T │ T │ T │ T │ T │ T │ T │ T │ D │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ E │ T │ T │ T │ T │ P │ A │ T │ T │ T │ T │ E │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

Legend: T=Tensix (5 RISC-V cores each), D=DRAM, E=Ethernet, P=PCIe, A=ARC
```

**NoC addressing:**
```text
63        48 47      40 39              0
┌───────────┬─────────┬──────────────────┐
│  NoC Y    │  NoC X  │  Local Address   │
└───────────┴─────────┴──────────────────┘
```

**Example: Reading from another Tensix core:**
```cpp
// Read 1KB from Tensix at (3,4), L1 offset 0x1000
uint64_t remote_addr = get_noc_addr(3, 4, 0x1000);
noc_async_read(remote_addr, local_l1_addr, 1024);
noc_async_read_barrier();
```

---

## Step 5: Dive Deeper - Full RISC-V Exploration Guide

Ready to explore more? We've prepared a comprehensive guide covering:

### 📚 Topics Covered in the Full Guide:

**Architecture:**
- Detailed breakdown of all 5 RISC-V processors per Tensix
- RV32IM ISA specification and instruction set
- Memory architecture and address spaces
- Firmware and bootloader operation

**Programming:**
- Writing pure RISC-V assembly kernels
- Inline assembly in C++ kernels
- Multi-core parallel programming
- Inter-core communication via NoC
- DMA optimization techniques
- Circular buffer programming

**Advanced Topics:**
- Custom firmware development
- Register-level debugging
- Profiling RISC-V execution
- Comparison with other RISC-V platforms (SiFive, ESP32-C3)

**Real-World Examples:**
- Matrix multiplication with SPMD parallelism
- Multicast communication patterns
- Flash Attention implementation

**Reference:**
- Complete memory map
- Device API quick reference
- Build system documentation
- Toolchain details (riscv32-gcc, linker scripts)

### 📖 Read the Full Guide

The complete exploration guide is available at:

**`RISC-V_EXPLORATION.md`** in the extension directory

[📘 Open Full RISC-V Exploration Guide](command:tenstorrent.openRiscvGuide)

---

## Next Steps

### Experiment with the Code

1. **Modify the example:** Change the integers from 14+7 to other values
2. **Try multiplication:** Change `+` to `*` and observe RISC-V `mul` instruction
3. **Run on multiple cores:** Modify to use `CoreRange{{0,0}, {1,1}}` for 2x2 grid

### Explore More Examples

```bash
# View all programming examples
ls ~/tt-metal/tt_metal/programming_examples/

# Matrix multiplication (multi-core)
~/tt-metal/tt_metal/programming_examples/matmul/

# More examples in tech reports
~/tt-metal/tech_reports/prog_examples/
```

### Write Your Own Kernel

Create a simple kernel template:

```cpp
// my_kernel.cpp
#include "dataflow_api.h"

void kernel_main() {
    // Get runtime argument
    uint32_t input = get_arg_val<uint32_t>(0);

    // Your RISC-V code here!
    uint32_t result = input * 2;

    // Write to L1
    uint32_t* output = (uint32_t*)0x1000;
    *output = result;
}
```

Then compile and run via the tt-metal C++ API!

### Learn More About RISC-V

**Official RISC-V Resources:**
- RISC-V ISA Specification: https://riscv.org/technical/specifications/
- RV32IM Reference Card: https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf

**Tenstorrent Resources:**
- Metalium Guide: `~/tt-metal/METALIUM_GUIDE.md`
- Programming Examples: `~/tt-metal/tt_metal/programming_examples/`
- Tech Reports: `~/tt-metal/tech_reports/`

**Community:**
- Tenstorrent Discord: https://discord.gg/tenstorrent
- GitHub: https://github.com/tenstorrent/tt-metal

---

## Why This Matters

Understanding RISC-V programming on Tenstorrent hardware gives you:

- ✅ **Low-level optimization skills** - Understand exactly what the hardware is doing
- ✅ **Parallel computing experience** - Learn to coordinate hundreds of processors
- ✅ **Architecture knowledge** - Near-memory compute, NoC, distributed systems
- ✅ **Debugging superpowers** - Trace execution at the instruction level
- ✅ **Competitive advantage** - Few developers understand this level of the stack

**From simple addition to complex AI workloads, it all runs on RISC-V instructions - hundreds of thousands of them, executing in parallel across the chip.**

Welcome to low-level programming at scale. 880 RISC-V cores are waiting for your code! 🚀
