# Exploring Tenstorrent as a RISC-V Assembly Programming Platform

## Introduction: An Unconventional RISC-V Environment

When most people think of RISC-V programming, they imagine embedded development boards like SiFive's HiFive or ESP32-C3 microcontrollers. But Tenstorrent's Wormhole and Blackhole accelerator cards offer something far more exotic: **hundreds of RISC-V cores networked together on a single chip, each with direct access to 1.5MB of local SRAM and connected via a high-performance Network-on-Chip (NoC)**.

This isn't your typical embedded RISC-V environment. Each Tensix core on a Tenstorrent processor contains **five independent RISC-V CPUs** working in concert - two for data movement, three for compute pipeline stages. Rather than being hidden behind abstraction layers, these processors are directly programmable, offering a unique platform for exploring RISC-V assembly programming, parallel computing, and near-memory compute architectures.

This guide explores Tenstorrent hardware from the perspective of a RISC-V programmer, revealing the low-level architecture and providing hands-on examples of programming these processors directly.

---

## Part 1: Architecture Deep-Dive

### The Tensix Core: Five RISC-V Processors Working Together

Each Tensix core is a complete compute unit containing:

```
┌─────────────────────────────────────────────────┐
│                  TENSIX CORE                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐        ┌──────────────┐      │
│  │   BRISC      │        │   NCRISC     │      │
│  │ (Data Move 0)│        │ (Data Move 1)│      │
│  │  RISCV_0     │        │  RISCV_1     │      │
│  └──────────────┘        └──────────────┘      │
│                                                 │
│  ┌──────────────┬──────────────┬──────────────┐│
│  │   TRISC0     │   TRISC1     │   TRISC2     ││
│  │   (Unpack)   │   (Math)     │   (Pack)     ││
│  └──────────────┴──────────────┴──────────────┘│
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │         1.5 MB L1 SRAM                  │   │
│  │         (Shared Memory)                 │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │  Matrix Engine (FPU) + Vector (SFPU)   │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌──────────────┐        ┌──────────────┐      │
│  │   NoC 0      │        │   NoC 1      │      │
│  │  Interface   │        │  Interface   │      │
│  └──────────────┘        └──────────────┘      │
└─────────────────────────────────────────────────┘
```

### The Five RISC-V Processors

#### 1. BRISC (Base RISC) - RISCV_0 / Data Movement 0
- **Purpose:** Primary data movement processor
- **Firmware:** `brisc.cc`
- **Typical tasks:** Reading data from DRAM/other cores via NoC
- **Memory regions:** 4KB local memory, 6KB firmware space, 48KB kernel space
- **Memory base:** `0xFFB00000` (local), firmware at mailbox end

#### 2. NCRISC (Network Core RISC) - RISCV_1 / Data Movement 1
- **Purpose:** Secondary data movement processor, network operations
- **Firmware:** `ncrisc.cc`
- **Special feature:** Has dedicated IRAM at `0xFFC00000` (16KB)
- **Typical tasks:** Writing data to DRAM/other cores via NoC
- **Memory regions:** 4KB local memory, 2KB firmware space, 16KB IRAM for kernels

#### 3-5. TRISC0, TRISC1, TRISC2 (Tensor RISC) - Compute Pipeline
- **TRISC0 (Unpack):** Moves data from L1 SRAM into compute engine registers
- **TRISC1 (Math):** Issues instructions to FPU and SFPU compute engines
- **TRISC2 (Pack):** Writes results from compute engines back to L1 SRAM
- **Firmware:** `trisc.cc` (shared codebase)
- **Memory regions:** 2KB local memory each, 1.5KB firmware, 24KB kernel space

### RISC-V ISA: RV32IM

All five processors implement the **RV32IM** instruction set:
- **RV32I:** Base integer instruction set (32-bit)
- **M Extension:** Integer multiplication and division

**Key characteristics:**
- **No hardware threads** - Single-threaded execution per core
- **No caches** - Explicit DMA operations for memory access
- **No FPU in RISC-V cores** - Floating point handled by dedicated hardware engines
- **Bare-metal execution** - No OS, no virtual memory, direct hardware access

### Memory Architecture: A RISC-V Programmer's View

#### L1 SRAM (1.5MB per Tensix)
```
Base Address: 0x00000000
Size:         1464 KB (1.5 MB)
Access:       Shared across all 5 RISC-V cores in the Tensix
              Also accessible by other Tensix cores via NoC
Purpose:      - Circular buffers for inter-kernel communication
              - Temporary data storage
              - Code execution space for kernels
```

#### Local Memory (Per-Processor Private Memory)
```
BRISC:  0xFFB00000 - 0xFFB00FFF (4 KB)
NCRISC: 0xFFB01000 - 0xFFB01FFF (4 KB)
TRISC:  0xFFB02000 - 0xFFB027FF (2 KB each)

Purpose: Stack, local variables, processor-specific data
```

#### NCRISC IRAM (Instruction RAM)
```
Base Address: 0xFFC00000
Size:         16 KB
Purpose:      Fast instruction execution for NCRISC kernels
              (Wormhole architecture feature)
```

#### DRAM
```
Size:      1 GB per chip (distributed across DRAM controllers)
Access:    Via NoC DMA operations only
           Not directly addressable from RISC-V cores
```

### The Mailbox: Inter-Processor Communication

Located at `MEM_MAILBOX_BASE` (offset 16 in L1):
```
Address: 0x00000010 - 0x000031BF
Size:    12,768 bytes

Contains:
- Device messages (dev_msgs_t structure)
- Runtime arguments passed from host
- Synchronization flags
- NCRISC halt/resume stack pointer (offset +4)
```

The mailbox is the primary mechanism for:
1. Host-to-device communication
2. Passing kernel arguments at runtime
3. Inter-processor synchronization

---

## Part 2: The Toolchain and Build System

### Compilation Pipeline

Tenstorrent uses a standard RISC-V GCC toolchain with custom linker scripts:

```
┌─────────────────┐
│  Kernel Code    │  (C++ with device APIs)
│  example.cpp    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  riscv32-gcc    │  (Cross compiler)
│  -march=rv32im  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Linker         │  (Custom linker scripts)
│  main.ld        │  - Separate sections per processor
│                 │  - Firmware vs. kernel regions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ELF Binary     │  (elf32-littleriscv)
│  (per core)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  tt-metal       │  (Runtime loads onto device)
│  Host API       │
└─────────────────┘
```

### Linker Script Structure

The main linker script (`tt_metal/hw/toolchain/main.ld`) separates memory regions per processor:

```ld
OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)

/* Conditional compilation per processor */
#if defined(COMPILE_FOR_BRISC)
    #define TEXT_START MEM_BRISC_FIRMWARE_BASE
    #define TEXT_SIZE  MEM_BRISC_FIRMWARE_SIZE
    #define DATA_START MEM_LOCAL_BASE
    #define DATA_SIZE  MEM_BRISC_LOCAL_SIZE
#elif defined(COMPILE_FOR_NCRISC)
    #define TEXT_START MEM_NCRISC_KERNEL_BASE  /* IRAM! */
    #define TEXT_SIZE  MEM_NCRISC_KERNEL_SIZE
    /* ... */
#endif

SECTIONS {
    .text TEXT_START : {
        *(.start)          /* Assembly entry point */
        *(.text .text.*)   /* Code */
    }
    .data DATA_START : {
        *(.data .data.*)   /* Initialized data */
    }
    .bss : {
        *(.bss .bss.*)     /* Uninitialized data */
    }
}
```

### Assembly Startup: crt0.S

Every RISC-V program starts with `_start` in `tmu-crt0.S`:

```asm
.section .start,"ax",@progbits
.global _start
.type   _start, @function

_start:
    /* Initialize global pointer (gp register) */
    .option push
    .option norelax
    lui  gp, %hi(__global_pointer$)
    addi gp, gp, %lo(__global_pointer$)
    .option pop

    /* Set stack pointer */
    lui  sp, %hi(__stack_top - 16)
    addi sp, sp, %lo(__stack_top - 16)

    /* Pass Tensix coordinates as argv[0] */
    addi a0, sp, 8
    sw   a0, 0(sp)      /* argv[0] */
    sw   zero, 4(sp)    /* argv[1] = NULL */
    sw   s1, 8(sp)      /* Coordinates in s1 */
    sw   zero, 12(sp)

    li   a0, 1          /* argc = 1 */
    mv   a1, sp         /* argv */
    mv   a2, zero       /* env = NULL */

    /* Call main, then exit */
    call main
    tail exit
```

**Key insights:**
- **Global pointer (`gp`):** Used for efficient access to small data section
- **Stack setup:** Each processor has its own stack in local memory
- **Tensix coordinates:** Passed via `s1` register (set by hardware)
- **No OS:** Direct jump to `main()`, no libc initialization

---

## Part 3: Hands-On Example - Adding Two Integers in RISC-V

Let's walk through the canonical example from tt-metal: `add_2_integers_in_riscv`.

### High-Level Flow

```
┌──────────────┐
│     Host     │
│   (C++ API)  │
└──────┬───────┘
       │ 1. Create buffers in DRAM
       │ 2. Upload integers (14, 7)
       │ 3. Launch kernel on BRISC
       ▼
┌──────────────────────────────────┐
│  Tensix Core (0,0)               │
│                                  │
│  ┌────────────────────────────┐ │
│  │  BRISC Kernel              │ │
│  │  (Data Movement Core)      │ │
│  │                            │ │
│  │  1. Read 14 from DRAM → L1│ │
│  │  2. Read 7 from DRAM → L1 │ │
│  │  3. Add: 14 + 7 = 21       │ │ ← RISC-V addition!
│  │  4. Write 21 to DRAM       │ │
│  └────────────────────────────┘ │
└──────────────────────────────────┘
       │
       ▼
┌──────────────┐
│     Host     │
│  Read result │
│  (21)        │
└──────────────┘
```

### The Kernel Code (Device Side)

**File:** `tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp`

```cpp
void kernel_main() {
    // Get runtime arguments (DRAM and L1 addresses)
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram  = get_arg_val<uint32_t>(2);
    uint32_t src0_l1   = get_arg_val<uint32_t>(3);
    uint32_t src1_l1   = get_arg_val<uint32_t>(4);
    uint32_t dst_l1    = get_arg_val<uint32_t>(5);

    // Create address generators (for DRAM buffers)
    InterleavedAddrGen<true> src0 = {
        .bank_base_address = src0_dram,
        .page_size = sizeof(uint32_t)
    };
    InterleavedAddrGen<true> src1 = {
        .bank_base_address = src1_dram,
        .page_size = sizeof(uint32_t)
    };
    InterleavedAddrGen<true> dst = {
        .bank_base_address = dst_dram,
        .page_size = sizeof(uint32_t)
    };

    // ═══════════════════════════════════════════════════════
    // STEP 1: DMA from DRAM to L1 SRAM (via NoC)
    // ═══════════════════════════════════════════════════════
    uint64_t src0_dram_noc_addr = get_noc_addr(0, src0);
    uint64_t src1_dram_noc_addr = get_noc_addr(0, src1);

    noc_async_read(src0_dram_noc_addr, src0_l1, sizeof(uint32_t));
    noc_async_read(src1_dram_noc_addr, src1_l1, sizeof(uint32_t));
    noc_async_read_barrier();  // Wait for DMA to complete

    // ═══════════════════════════════════════════════════════
    // STEP 2: THE RISC-V ADDITION (Running on BRISC)
    // ═══════════════════════════════════════════════════════
    uint32_t* dat0 = (uint32_t*)src0_l1;  // Pointer to L1 SRAM
    uint32_t* dat1 = (uint32_t*)src1_l1;
    uint32_t* out0 = (uint32_t*)dst_l1;

    // This is compiled to RISC-V add instruction!
    (*out0) = (*dat0) + (*dat1);

    // Optional: Print (requires TT_METAL_DPRINT_CORES=0,0)
    DPRINT << "Adding: " << *dat0 << " + " << *dat1 << "\n";

    // ═══════════════════════════════════════════════════════
    // STEP 3: DMA from L1 back to DRAM
    // ═══════════════════════════════════════════════════════
    uint64_t dst_dram_noc_addr = get_noc_addr(0, dst);
    noc_async_write(dst_l1, dst_dram_noc_addr, sizeof(uint32_t));
    noc_async_write_barrier();  // Wait for write to complete
}
```

### What Really Happens in Assembly

When you compile this kernel, the addition becomes RISC-V assembly:

```asm
# Load from L1 SRAM (dat0 and dat1 are pointers in L1)
lw   t0, 0(a0)    # Load *dat0 into t0
lw   t1, 0(a1)    # Load *dat1 into t1

# RISC-V addition
add  t2, t0, t1   # t2 = t0 + t1

# Store result back to L1 SRAM
sw   t2, 0(a2)    # Store t2 into *out0
```

**Key insight:** While the C++ API provides `noc_async_read/write` for DMA operations, the actual arithmetic happens in plain RISC-V instructions executing on the BRISC processor.

### Host Code (Orchestration)

**File:** `tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp`

```cpp
int main() {
    // Create 1x1 mesh (single device)
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // Create DRAM and L1 buffers
    constexpr uint32_t buffer_size = sizeof(uint32_t);
    auto src0_dram_buffer = distributed::MeshBuffer::create(
        buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = /* ... */;
    auto dst_dram_buffer = /* ... */;
    auto src0_l1_buffer = distributed::MeshBuffer::create(
        buffer_config, l1_config, mesh_device.get());
    auto src1_l1_buffer = /* ... */;
    auto dst_l1_buffer = /* ... */;

    // Upload integers to DRAM
    std::vector<uint32_t> src0_vec = {14};
    std::vector<uint32_t> src1_vec = {7};
    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    // Create kernel that runs on BRISC (Data Movement 0)
    Program program = CreateProgram();
    KernelHandle kernel_id = CreateKernel(
        program,
        "add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
        CoreCoord{0, 0},  // Tensix at (0,0)
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,  // BRISC
            .noc = NOC::RISCV_0_default
        });

    // Pass addresses as runtime arguments
    SetRuntimeArgs(program, kernel_id, CoreCoord{0, 0}, {
        src0_dram_buffer->address(),
        src1_dram_buffer->address(),
        dst_dram_buffer->address(),
        src0_l1_buffer->address(),
        src1_l1_buffer->address(),
        dst_l1_buffer->address(),
    });

    // Execute!
    distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read result
    std::vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    std::cout << "Result: " << result_vec[0] << std::endl;  // 21
    mesh_device->close();
}
```

---

## Part 4: The NoC - Network-on-Chip Architecture

### What is the NoC?

The Network-on-Chip (NoC) is a 2D mesh interconnect that connects:
- All Tensix cores
- DRAM controllers
- PCIe interfaces
- Ethernet cores (for multi-chip)

```
Wormhole NoC Grid (Example):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ D │ T │ T │ T │ T │ T │ T │ T │ T │ T │ T │ D │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ D │ T │ T │ T │ T │ T │ T │ T │ T │ T │ T │ D │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ D │ T │ T │ T │ T │ T │ T │ T │ T │ T │ T │ D │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ E │ T │ T │ T │ T │ P │ A │ T │ T │ T │ T │ E │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

Legend:
T = Tensix core (each with 5 RISC-V CPUs)
D = DRAM controller
E = Ethernet (for multi-chip)
P = PCIe
A = ARC (management processor)
```

### NoC Programming Model

From a RISC-V programmer's perspective, NoC operations are **asynchronous DMA transactions**:

```cpp
// Read from remote location (DRAM or another Tensix's L1)
uint64_t remote_addr = get_noc_addr(x, y, local_offset);
noc_async_read(remote_addr, local_l1_addr, size);
noc_async_read_barrier();  // Wait for completion

// Write to remote location
noc_async_write(local_l1_addr, remote_addr, size);
noc_async_write_barrier();
```

**NoC address encoding:**
```
63        48 47      40 39              0
┌───────────┬─────────┬──────────────────┐
│  NoC Y    │  NoC X  │  Local Address   │
└───────────┴─────────┴──────────────────┘
```

Helper function:
```cpp
uint64_t get_noc_addr(uint32_t x, uint32_t y, uint32_t addr) {
    return ((uint64_t)y << 48) | ((uint64_t)x << 40) | addr;
}
```

### Example: Reading from Another Tensix Core

```cpp
// Read from L1 SRAM of Tensix at (3, 4)
constexpr uint32_t remote_x = 3;
constexpr uint32_t remote_y = 4;
constexpr uint32_t remote_l1_addr = 0x1000;
constexpr uint32_t local_l1_addr = 0x2000;

uint64_t noc_addr = get_noc_addr(remote_x, remote_y, remote_l1_addr);
noc_async_read(noc_addr, local_l1_addr, 1024);  // Read 1KB
noc_async_read_barrier();

// Now data is in local L1 at 0x2000
uint32_t* data = (uint32_t*)local_l1_addr;
```

---

## Part 5: Writing Pure Assembly Kernels (Advanced)

While most kernels are written in C++, you can write pure RISC-V assembly.

### Example: Assembly Addition Kernel

**File:** `my_asm_add.S`

```asm
.section .text
.globl kernel_main
.type kernel_main, @function

kernel_main:
    # Save return address
    addi sp, sp, -16
    sw   ra, 12(sp)

    # Get runtime arguments from mailbox
    # get_arg_val is a C++ function, but we can call it
    li   a0, 0              # arg index 0
    call get_arg_val        # returns src0_l1 address in a0
    mv   s0, a0             # save in s0

    li   a0, 1              # arg index 1
    call get_arg_val        # returns src1_l1 address
    mv   s1, a0             # save in s1

    li   a0, 2              # arg index 2
    call get_arg_val        # returns dst_l1 address
    mv   s2, a0             # save in s2

    # Load operands from L1 SRAM
    lw   t0, 0(s0)          # Load *src0_l1
    lw   t1, 0(s1)          # Load *src1_l1

    # THE ADDITION!
    add  t2, t0, t1

    # Store result to L1 SRAM
    sw   t2, 0(s2)          # Store to *dst_l1

    # Restore and return
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret

.size kernel_main, .-kernel_main
```

### Inline Assembly in C++ Kernels

You can also embed assembly directly:

```cpp
void kernel_main() {
    uint32_t* src0 = (uint32_t*)get_arg_val<uint32_t>(0);
    uint32_t* src1 = (uint32_t*)get_arg_val<uint32_t>(1);
    uint32_t* dst  = (uint32_t*)get_arg_val<uint32_t>(2);

    uint32_t result;

    // Inline assembly for addition
    asm volatile (
        "lw   t0, 0(%1)\n"      // Load *src0
        "lw   t1, 0(%2)\n"      // Load *src1
        "add  t2, t0, t1\n"     // Add
        "sw   t2, 0(%0)\n"      // Store to result
        : "=r" (result)         // Output
        : "r" (src0), "r" (src1) // Inputs
        : "t0", "t1", "t2"      // Clobbers
    );

    *dst = result;
}
```

---

## Part 6: Parallel RISC-V Programming

### Multi-Core Execution

Launch the same kernel on multiple Tensix cores:

```cpp
// Run on 4x4 grid of cores
CoreRange core_range = {{0, 0}, {3, 3}};  // (0,0) to (3,3)

KernelHandle kernel_id = CreateKernel(
    program, "my_kernel.cpp", core_range,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});

// Set DIFFERENT runtime arguments per core
for (uint32_t x = 0; x < 4; x++) {
    for (uint32_t y = 0; y < 4; y++) {
        CoreCoord core{x, y};

        // Calculate which data this core processes
        uint32_t data_offset = (y * 4 + x) * chunk_size;

        SetRuntimeArgs(program, kernel_id, core, {
            input_buffer->address() + data_offset,
            output_buffer->address() + data_offset,
            chunk_size
        });
    }
}
```

Each BRISC processor executes the same code but with different arguments!

### Getting Core Coordinates in Kernel

```cpp
void kernel_main() {
    // Built-in variables (set by firmware)
    uint32_t my_x = my_logical_x_;
    uint32_t my_y = my_logical_y_;

    // Compute unique ID
    uint32_t core_id = my_y * grid_width + my_x;

    // Process data based on core ID
    uint32_t offset = core_id * CHUNK_SIZE;
    // ...
}
```

### Inter-Core Communication via NoC

```cpp
// Core (0,0) sends data to Core (1,0)
void kernel_main() {
    if (my_logical_x_ == 0 && my_logical_y_ == 0) {
        // Sender core
        uint32_t data[256];
        // ... fill data ...

        uint64_t dest_addr = get_noc_addr(1, 0, 0x1000);
        noc_async_write((uint32_t)data, dest_addr, sizeof(data));
        noc_async_write_barrier();
    } else if (my_logical_x_ == 1 && my_logical_y_ == 0) {
        // Receiver core
        uint32_t* received = (uint32_t*)0x1000;
        // ... wait for data arrival ...
        // ... process received data ...
    }
}
```

---

## Part 7: Debugging and Profiling

### Debug Printing from RISC-V Cores

Enable debug printing:
```bash
export TT_METAL_DPRINT_CORES=0,0  # Enable for Tensix (0,0)
```

In kernel:
```cpp
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "Hello from BRISC at ("
           << my_logical_x_ << "," << my_logical_y_ << ")\n";

    uint32_t value = 42;
    DPRINT << "Value: " << value << "\n";
}
```

Output appears on host stdout.

### Profiling RISC-V Execution

```bash
export TT_METAL_DEVICE_PROFILER=1
```

This enables cycle-accurate profiling of:
- Kernel execution time per core
- NoC transaction latency
- Time spent in each RISC-V processor

### Register Inspection (Advanced)

The firmware exposes register state via mailbox. You can read processor state from host:

```cpp
// Read BRISC instruction pointer (example)
auto mailbox_addr = device->get_mailbox_address(core);
auto pc_value = device->read_l1(mailbox_addr + PC_OFFSET, sizeof(uint32_t));
```

---

## Part 8: Comparison to Other RISC-V Platforms

### Tenstorrent vs. Traditional RISC-V Boards

| Feature | Tenstorrent Wormhole | SiFive HiFive | ESP32-C3 |
|---------|---------------------|---------------|----------|
| **RISC-V Cores** | 880 (5 per Tensix × 176) | 1-5 cores | 1 core |
| **ISA** | RV32IM | RV64GC | RV32IMC |
| **Clock Speed** | ~1 GHz | ~1.5 GHz | 160 MHz |
| **L1 per core** | 1.5 MB shared | 32 KB | 400 KB |
| **Interconnect** | 2D NoC mesh | AXI bus | Single bus |
| **Programming** | C++ device API | Bare-metal C/ASM | FreeRTOS/bare-metal |
| **Use Case** | AI accelerator | Linux SBC | IoT embedded |
| **Unique Feature** | Hundreds of cores + dedicated matrix/vector engines | Standard Linux workstation | WiFi/BLE integrated |

### Key Differences

**Advantages of Tenstorrent for RISC-V exploration:**
- ✅ **Massive parallelism:** 880 RISC-V cores on a single chip
- ✅ **Near-memory compute:** 1.5MB L1 per Tensix, no cache hierarchy
- ✅ **High-bandwidth interconnect:** NoC enables core-to-core communication at 100+ GB/s aggregate
- ✅ **Explicit control:** No OS, no hidden behavior, deterministic execution

**Challenges:**
- ❌ **Not general-purpose:** Designed for AI workloads, not typical embedded tasks
- ❌ **Limited peripherals:** No GPIO, UART, SPI in traditional sense
- ❌ **Learning curve:** Requires understanding NoC, mesh architecture, and tt-metal API

---

## Part 9: Real-World Examples in tt-metal

### Matrix Multiplication (SPMD Parallelism)

Found in `tt_metal/programming_examples/matmul/`:
- Multiple Tensix cores each process a tile of the matrix
- Data movement cores (BRISC/NCRISC) load tiles from DRAM
- Compute cores (TRISC) orchestrate FPU operations
- Results written back via DMA

### Multicast Communication

Found in `tech_reports/prog_examples/multicast/`:
- One Tensix broadcasts data to multiple receivers simultaneously
- Uses NoC multicast addressing
- Efficient for distributing weights in neural networks

### Flash Attention

Found in `tech_reports/FlashAttention/`:
- Tiled attention mechanism
- Each Tensix processes a query/key/value tile
- Heavy use of L1 SRAM to avoid DRAM bottlenecks
- RISC-V cores orchestrate data movement between tiles

---

## Part 10: Getting Started - Build and Run

### Prerequisites

```bash
# Clone tt-metal
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal

# Install dependencies
./install_dependencies.sh

# Build with programming examples
./build_metal.sh --build-programming-examples
```

### Run the RISC-V Addition Example

```bash
export TT_METAL_HOME=$(pwd)
export TT_METAL_DPRINT_CORES=0,0  # Enable debug output

./build/programming_examples/add_2_integers_in_riscv
```

Expected output:
```
Adding integers: 14 + 7
Success: Result is 21
```

### Exploring the Firmware

```bash
# View BRISC firmware source
cat tt_metal/hw/firmware/src/tt-1xx/brisc.cc

# View assembly startup
cat tt_metal/hw/toolchain/tmu-crt0.S

# View linker script
cat tt_metal/hw/toolchain/main.ld

# View memory map
cat tt_metal/hw/inc/tt-1xx/wormhole/dev_mem_map.h
```

### Writing Your Own Kernel

1. Create `my_kernel.cpp` in a new directory
2. Use the device API: `get_arg_val`, `noc_async_read/write`, etc.
3. Compile via `CreateKernel` API
4. Launch from host with `SetRuntimeArgs` and `EnqueueProgram`

Example skeleton:
```cpp
// my_kernel.cpp
#include "dataflow_api.h"

void kernel_main() {
    // Get arguments
    uint32_t arg0 = get_arg_val<uint32_t>(0);

    // Your RISC-V code here!
    uint32_t result = arg0 * 2;

    // Write to L1
    uint32_t* output = (uint32_t*)0x1000;
    *output = result;
}
```

---

## Part 11: Advanced Topics

### DMA Optimization

Maximize NoC bandwidth:
```cpp
// Bad: Sequential reads (latency adds up)
for (int i = 0; i < 1000; i++) {
    noc_async_read(addr + i * 32, local + i * 32, 32);
    noc_async_read_barrier();  // DON'T DO THIS IN LOOP!
}

// Good: Batch reads, single barrier
for (int i = 0; i < 1000; i++) {
    noc_async_read(addr + i * 32, local + i * 32, 32);
}
noc_async_read_barrier();  // Wait once at the end
```

### Circular Buffers (Advanced Inter-Kernel Communication)

Used for producer-consumer patterns between BRISC and TRISC:

```cpp
// In reader kernel (BRISC)
constexpr uint32_t cb_id = tt::CBIndex::c_0;
cb_reserve_back(cb_id, 1);  // Reserve space
uint32_t write_ptr = get_write_ptr(cb_id);
noc_async_read(src, write_ptr, tile_size);
noc_async_read_barrier();
cb_push_back(cb_id, 1);  // Signal data ready

// In compute kernel (TRISC)
cb_wait_front(cb_id, 1);  // Wait for data
uint32_t read_ptr = get_read_ptr(cb_id);
// ... process data ...
cb_pop_front(cb_id, 1);  // Release buffer
```

### Custom Firmware (Experimental)

You can modify the base firmware (e.g., `brisc.cc`) to change boot behavior, but this requires rebuilding the entire firmware image. Not recommended for most users.

---

## Part 12: Limitations and Gotchas

### What You CAN'T Do

1. **No dynamic memory allocation:** No `malloc()`, `new`, etc. All buffers must be pre-allocated.
2. **No standard library:** No `printf`, `fopen`, etc. Use device APIs instead.
3. **No interrupts:** Polling-based synchronization only.
4. **No virtual memory:** All addresses are physical.
5. **No floating-point in RISC-V cores:** Use the FPU/SFPU engines via TRISC instead.

### Common Pitfalls

**1. Forgetting barriers:**
```cpp
noc_async_read(src, dst, size);
// BUG: Data might not be ready yet!
uint32_t* data = (uint32_t*)dst;
uint32_t value = data[0];  // May read garbage!

// FIX: Add barrier
noc_async_read(src, dst, size);
noc_async_read_barrier();  // Wait!
uint32_t* data = (uint32_t*)dst;
uint32_t value = data[0];  // Safe
```

**2. Incorrect NoC addressing:**
```cpp
// BUG: Forgot to encode X/Y coordinates
uint64_t addr = 0x1000;  // Missing NoC coordinates!
noc_async_read(addr, local, size);  // Will fail!

// FIX: Use get_noc_addr
uint64_t addr = get_noc_addr(x, y, 0x1000);
noc_async_read(addr, local, size);  // Correct
```

**3. Stack overflow:**
Each RISC-V core has limited stack space (256 bytes minimum). Avoid large local arrays:
```cpp
// BAD
void kernel_main() {
    uint32_t big_array[1000];  // 4KB - WILL OVERFLOW STACK!
    // ...
}

// GOOD
void kernel_main() {
    uint32_t* big_array = (uint32_t*)0x10000;  // Use L1 instead
    // ...
}
```

---

## Conclusion: A Unique RISC-V Playground

Tenstorrent's Wormhole and Blackhole cards represent a rare opportunity to explore RISC-V programming at scale. Unlike traditional embedded boards with a handful of cores, these accelerators pack **hundreds of RISC-V processors** on a single chip, all connected via a high-performance mesh network and backed by massive on-chip SRAM.

**What makes this platform special:**
- **Bare-metal access:** No OS, no hidden behavior, direct hardware control
- **Massive parallelism:** 880 RISC-V cores working in concert
- **Near-memory compute:** 1.5MB L1 per Tensix eliminates memory bottlenecks
- **Explicit communication:** NoC programming teaches distributed systems concepts
- **Production hardware:** Not a research prototype - real AI accelerators in the field

**Who should explore this:**
- **RISC-V enthusiasts:** Want to program RV32IM at scale
- **Parallel programming students:** Learn distributed computing with real hardware
- **Embedded developers:** Understand bare-metal programming without an OS
- **Computer architects:** Study NoC, near-memory compute, and tiled architectures
- **AI researchers:** Optimize kernels at the lowest level

**Next Steps:**
1. Build tt-metal and run `add_2_integers_in_riscv`
2. Study the programming examples in `tt_metal/programming_examples/`
3. Modify existing kernels to experiment with RISC-V assembly
4. Write multi-core parallel algorithms using the NoC
5. Profile your kernels and optimize for the architecture

The path from simple addition to complex AI workloads is paved with RISC-V instructions - hundreds of thousands of them, executing in parallel across the chip. This is RISC-V programming at a scale few other platforms can offer.

**Welcome to the Tenstorrent RISC-V ecosystem. 880 cores are waiting for your code.**

---

## Appendix A: Quick Reference

### Memory Map (Wormhole)
```
0x00000000 - 0x0016FFFF   L1 SRAM (1464 KB)
0xFFB00000 - 0xFFB00FFF   BRISC Local (4 KB)
0xFFB01000 - 0xFFB01FFF   NCRISC Local (4 KB)
0xFFB02000 - 0xFFB027FF   TRISC0 Local (2 KB)
0xFFB02800 - 0xFFB02FFF   TRISC1 Local (2 KB)
0xFFB03000 - 0xFFB037FF   TRISC2 Local (2 KB)
0xFFC00000 - 0xFFC03FFF   NCRISC IRAM (16 KB)
```

### Common Device API Functions
```cpp
// Runtime arguments
uint32_t get_arg_val<T>(uint32_t index);

// NoC operations
uint64_t get_noc_addr(uint32_t x, uint32_t y, uint32_t addr);
void noc_async_read(uint64_t src_addr, uint32_t dst_addr, uint32_t size);
void noc_async_write(uint32_t src_addr, uint64_t dst_addr, uint32_t size);
void noc_async_read_barrier();
void noc_async_write_barrier();

// Circular buffers
void cb_reserve_back(uint32_t cb_id, uint32_t num_tiles);
void cb_push_back(uint32_t cb_id, uint32_t num_tiles);
void cb_wait_front(uint32_t cb_id, uint32_t num_tiles);
void cb_pop_front(uint32_t cb_id, uint32_t num_tiles);
uint32_t get_write_ptr(uint32_t cb_id);
uint32_t get_read_ptr(uint32_t cb_id);

// Core info
extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;

// Debug
DPRINT << "message" << value << "\n";
```

### Build Commands
```bash
# Build tt-metal with examples
./build_metal.sh --build-programming-examples

# Clean rebuild
./build_metal.sh --clean

# Enable ccache for faster rebuilds
./build_metal.sh --enable-ccache
```

### Environment Variables
```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_DPRINT_CORES=0,0        # Enable debug prints
export TT_METAL_DEVICE_PROFILER=1       # Enable profiling
export MESH_DEVICE=N150                 # Hardware target
```

---

## Appendix B: Resources

**Official Documentation:**
- tt-metal GitHub: https://github.com/tenstorrent/tt-metal
- Metalium Guide: `tt-metal/METALIUM_GUIDE.md`
- Programming Examples: `tt-metal/tt_metal/programming_examples/`

**RISC-V Resources:**
- RISC-V ISA Spec: https://riscv.org/technical/specifications/
- RV32IM Reference: https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf

**Community:**
- Tenstorrent Discord: https://discord.gg/tenstorrent
- GitHub Issues: https://github.com/tenstorrent/tt-metal/issues

---

**Document Version:** 1.0
**Last Updated:** 2025-12-16
**Target Hardware:** Wormhole (N150/N300), Blackhole
**tt-metal Version:** Latest main branch
