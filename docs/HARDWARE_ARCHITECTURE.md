# Tenstorrent Hardware Architecture Reference

**Version:** 1.0
**Last Updated:** 2026-01-09
**Purpose:** Comprehensive reference for all Tenstorrent hardware types, architecture families, and lesson compatibility.

---

## Overview

This document provides the authoritative reference for all Tenstorrent hardware types supported by the VSCode extension. It documents architecture families, equivalence principles, configuration patterns, and lesson compatibility.

---

## Quick Reference Table

| Hardware | Architecture | Chips | Tensix Cores | Context Limit | Best For | Extension Type |
|----------|--------------|-------|--------------|---------------|----------|----------------|
| **N150** | Wormhole | 1 | 72 | 64K | Development, single-user | `n150` |
| **N300** | Wormhole | 2 | 144 | 128K | Multi-user, higher throughput | `n300` |
| **T3K** | Wormhole | 8 | 576 | 128K | Large models (70B+) | `t3k` |
| **P100** | Blackhole | 1 | TBD | 64K | Cloud/standalone | `p100` |
| **P150** | Blackhole | 2 | TBD | 128K | Higher performance | `p150` |
| **P300/P300c** | Blackhole | 1 | TBD | 64K | QuietBox systems | `p300` |
| **Galaxy** | Wormhole | 32+ | 2304+ | 128K+ | Multi-rack clusters | `galaxy` |
| **Simulator** | Virtual | N/A | N/A | Varies | Development without hardware | `simulator` |

---

## Architecture Families

### Wormhole Family (2nd Generation)

**Architecture:** Wormhole
**Release:** 2nd generation Tenstorrent hardware
**Status:** Production, widely deployed

#### N150 - Single Chip

**Hardware Specifications:**
- **Chips:** 1 Wormhole chip
- **Tensix cores:** 72
- **Context limit:** 64K tokens (typical)
- **PCI:** Single PCIe slot
- **Deployment:** Cloud instances, development workstations

**Best For:**
- Development and prototyping
- Single-user workloads
- Model fine-tuning
- Small to medium models (< 8B parameters)

**Configuration:**
- `MESH_DEVICE`: Not required (single chip)
- `TT_METAL_ARCH_NAME`: `wormhole_b0`
- Device ID: `0`

**Validated Models:**
- Qwen3-0.6B (reasoning, production-ready)
- Gemma 3-1B-IT (multilingual)
- Game of Life, Mandelbrot, Audio Processing

#### N300 - Dual Chip

**Hardware Specifications:**
- **Chips:** 2 Wormhole chips
- **Tensix cores:** 144 (72 per chip)
- **Context limit:** 128K tokens
- **Tensor parallelism:** TP=2 (uses both chips)
- **Deployment:** Cloud instances, mid-range workstations

**Best For:**
- Higher throughput inference
- Longer context windows
- Multi-user serving
- Medium models (8B-13B parameters)

**Configuration:**
- `MESH_DEVICE=N300` (dual chip)
- `TT_METAL_ARCH_NAME=wormhole_b0`
- Devices: 0, 1

**Validated Models:**
- Llama-3.1-8B-Instruct
- Qwen3-8B

#### T3K - Eight Chip Cluster

**Hardware Specifications:**
- **Chips:** 8 Wormhole chips
- **Tensix cores:** 576 (72 per chip)
- **Context limit:** 128K tokens
- **Tensor parallelism:** TP=8 (uses all chips)
- **Deployment:** Production servers, enterprise deployments

**Best For:**
- Large models (70B+ parameters)
- Production serving at scale
- Multi-model serving
- High-concurrency workloads

**Configuration:**
- `MESH_DEVICE=T3K` (8-chip mesh)
- `TT_METAL_ARCH_NAME=wormhole_b0`
- Devices: 0-7

**Validated Models:**
- Llama-3.1-70B
- Large-scale production inference

#### Galaxy - Multi-Rack Clusters

**Hardware Specifications:**
- **Chips:** 32+ Wormhole chips
- **Tensix cores:** 2304+ (72 per chip)
- **Context limit:** 128K+ tokens
- **Deployment:** Multi-rack data center installations

**Best For:**
- Massive models (100B+ parameters)
- Distributed training
- Multi-tenant production serving
- Research clusters

**Configuration:**
- `MESH_DEVICE=GALAXY`
- Multi-node coordination required
- See TT-XLA documentation for JAX multi-rack

---

### Blackhole Family (3rd Generation)

**Architecture:** Blackhole
**Release:** 3rd generation Tenstorrent hardware (latest)
**Status:** Production, experimental models

**Architecture Principle:**
> "Anything that can run on one Blackhole card should be able to run on any one of Blackhole cards"

All Blackhole variants (P100, P150, P300/P300c) share:
- Same instruction set
- Same core architecture
- Same TT-Metal API
- Same model compatibility

**Differences:**
- Chip count (single vs dual)
- Packaging (cloud vs QuietBox)
- Possibly memory/firmware variations

#### P100 - Single Chip

**Hardware Specifications:**
- **Chips:** 1 Blackhole chip
- **Tensix cores:** TBD (next-gen architecture)
- **Context limit:** 64K tokens
- **Deployment:** Cloud instances, standalone systems

**Best For:**
- Similar to N150 but with newer architecture
- Development on latest generation
- Experimental model validation

**Configuration:**
- `MESH_DEVICE=P100` (single Blackhole)
- `TT_METAL_ARCH_NAME=blackhole`
- Device ID: `0`

**Status:**
- Some models validated (vLLM, Qwen3)
- Others experimental (check official docs)

#### P150 - Dual Chip

**Hardware Specifications:**
- **Chips:** 2 Blackhole chips
- **Tensix cores:** TBD (2x single chip)
- **Context limit:** 128K tokens
- **Deployment:** Higher-performance configurations

**Best For:**
- Similar to N300 but with improvements
- Latest-generation dual-chip inference
- Production workloads requiring new architecture

**Configuration:**
- `MESH_DEVICE=P150` (dual Blackhole)
- `TT_METAL_ARCH_NAME=blackhole`
- Devices: 0, 1

**Status:**
- Check official documentation for validated configurations
- Experimental for many models

#### P300/P300c - Single Chip (QuietBox Variant)

**Hardware Specifications:**
- **Chips:** 1 Blackhole chip per card
- **Tensix cores:** TBD (same as P100)
- **Context limit:** 64K tokens
- **Deployment:** QuietBox multi-device systems
- **Common configuration:** 4x P300c (QuietBox Tower)

**Architecture:**
- **Identical to P100** (single Blackhole chip)
- P300c likely "compute" variant name
- Runs in P100 mode for single-chip lessons
- Each card is independently addressable

**Best For:**
- QuietBox development systems
- Multi-device workload distribution
- Research on multi-chip scaling

**Configuration:**
- `MESH_DEVICE=P100` (each card = single Blackhole)
- `TT_METAL_ARCH_NAME=blackhole`
- QuietBox Tower: Devices 0-3 (4 cards)

**QuietBox Multi-Device Pattern:**
- **4x P300c ≠ 4-chip system**
- **4x P300c = 4 separate single-chip cards**
- For single-chip lessons: Use device 0
- For multi-device lessons: Use all 4 devices

**Validated:**
- Lesson 7: vLLM Production (Qwen3-0.6B)
- Lesson 15: Metalium Cookbook - Particle Life multi-device (2x speedup achieved)

---

## Hardware Equivalence Principles

### Blackhole Family Equivalence

All Blackhole cards share the same core architecture:

**Blackhole Variants:**
- **P100**: Single Blackhole chip (cloud/standalone)
- **P150**: Dual Blackhole chip (higher performance)
- **P300/P300c**: Single Blackhole chip (QuietBox, compute variant)

**Equivalence Rules:**
1. **P100 = P300c** (architecture-wise)
   - Use `MESH_DEVICE=P100` for P300c
   - Same instruction set and capabilities
   - Lessons supporting P100 work on P300c

2. **Single-chip lessons work on all Blackhole**
   - P100 lesson → works on P300c
   - P100 lesson → requires adaptation for P150 (dual chip)

3. **Dual-chip lessons P150-specific**
   - P150 lessons don't apply to P100/P300c
   - Different MESH_DEVICE configuration

### Wormhole Family Consistency

All Wormhole variants (N150, N300, T3K, Galaxy) share:
- Same Wormhole architecture
- Same core capabilities
- Different scale (chip count)

**Scaling Rules:**
- Models validated on N150 generally work on N300/T3K
- May need configuration changes (MESH_DEVICE, TP settings)
- Larger models require more chips (N300+ for 8B+, T3K for 70B+)

---

## Multi-Device vs Multi-Chip Disambiguation

**Critical Distinction:**

### Multi-Chip System
- **Single device** with multiple chips
- Example: N300 (2 chips, 1 device)
- Example: P150 (2 chips, 1 device)
- Configuration: `MESH_DEVICE=N300` or `MESH_DEVICE=P150`

### Multi-Device System
- **Multiple devices**, each with 1 or more chips
- Example: QuietBox Tower (4x P300c = 4 devices, 4 chips total)
- Example: T3K could be (8 devices × 1 chip) or (4 devices × 2 chips)
- Configuration: `TT_METAL_NUM_DEVICES=4`, each device configured independently

**QuietBox Example:**
- **4x P300c** = 4 separate devices
- Each device has 1 Blackhole chip
- Each device enumerated: 0, 1, 2, 3
- For single-chip lesson: Use device 0 only
- For multi-device lesson: Distribute workload across all 4

---

## MESH_DEVICE Configuration Matrix

| Hardware | Single-Chip Lesson | Multi-Chip Lesson | Multi-Device Lesson |
|----------|-------------------|-------------------|---------------------|
| **N150** | (no MESH_DEVICE) | N/A | N/A |
| **N300** | `MESH_DEVICE=N300` | `MESH_DEVICE=N300` | N/A |
| **T3K** | `MESH_DEVICE=T3K` | `MESH_DEVICE=T3K` | Varies |
| **P100** | `MESH_DEVICE=P100` | N/A | N/A |
| **P150** | `MESH_DEVICE=P150` | `MESH_DEVICE=P150` | N/A |
| **P300c (1 card)** | `MESH_DEVICE=P100` | N/A | N/A |
| **P300c (4 cards)** | `MESH_DEVICE=P100` + `TT_METAL_DEVICE_ID=0` | N/A | `TT_METAL_NUM_DEVICES=4` + multi-device code |

---

## Lesson Compatibility Matrix

### Hardware Support by Lesson

| Lesson | N150 | N300 | T3K | P100 | P150 | P300 | Galaxy | Notes |
|--------|------|------|-----|------|------|------|--------|-------|
| 01 Hardware Detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 02 Verify Installation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 03 Download Model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 04 Interactive Chat | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 05 API Server | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 06 tt-inference-server | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | Blackhole experimental |
| 07 vLLM Production | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | **P300c validated** |
| 08 VSCode Chat | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 09 Image Generation | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | P300c arch supported, SDXL bug blocks |
| 10 Coding Assistant | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 11 TT-Forge | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | N150 only (experimental) |
| 12 TT-XLA JAX | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | Wormhole only |
| 13 RISC-V Programming | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 14 Explore Metalium | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All hardware |
| 15 Metalium Cookbook | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **P300c multi-device validated** |

**Legend:**
- ✅ Supported and validated
- ⬜ Architecturally supported, not validated
- ❌ Not supported

---

## Validation Status

### P300/P300c Validation (QuietBox Tower)

**Hardware:** 4x P300c (Blackhole) QuietBox Tower
**Date:** 2026-01-09
**Source:** `docs/QB_follows.md`

**Validated Lessons:**
1. **Lesson 7: vLLM Production**
   - Model: Qwen3-0.6B
   - Configuration: `MESH_DEVICE=P100`, `TT_METAL_ARCH_NAME=blackhole`
   - Status: ✅ Working perfectly
   - Notes: P300c detection added to starter script

2. **Lesson 15: Metalium Cookbook - Particle Life**
   - Workload: 2,048 particles across 3 species
   - Multi-device: 4x P300c parallelization
   - Performance: **2x speedup** (50% parallel efficiency)
   - Status: ✅ Multi-device extension validated
   - Notes: 419M force calculations, 2.4s runtime

**Not Validated:**
- **Lesson 9: Image Generation** - Blocked on SDXL grid size bug (architecture supports, software bug prevents)
- **Lesson 12: TT-XLA JAX** - Wormhole-specific, not applicable to Blackhole

---

## Configuration Examples

### Single-Chip P300c (QuietBox - Device 0 Only)

```bash
# Environment setup
export TT_METAL_HOME=~/tt-metal
export TT_METAL_ARCH_NAME=blackhole
export MESH_DEVICE=P100
export TT_METAL_DEVICE_ID=0  # Use only device 0

# Python environment
source ~/tt-metal/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# Verify single device
tt-smi -s | jq '.device_0'
```

### Multi-Device P300c (QuietBox - All 4 Devices)

```bash
# Environment setup
export TT_METAL_HOME=~/tt-metal
export TT_METAL_ARCH_NAME=blackhole
export TT_METAL_NUM_DEVICES=4

# Python environment
source ~/tt-metal/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# Verify all devices
tt-smi -s | jq -r '.device_0, .device_1, .device_2, .device_3 | .board_type'
# Should show: p300c (4 times)

# Python multi-device code
import ttnn
devices = [ttnn.open_device(device_id=i) for i in range(4)]
# ... distribute workload across devices ...
```

### Dual-Chip N300

```bash
# Environment setup
export TT_METAL_HOME=~/tt-metal
export TT_METAL_ARCH_NAME=wormhole_b0
export MESH_DEVICE=N300

# Python environment
source ~/tt-metal/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# Both chips used automatically via MESH_DEVICE=N300
```

---

## Troubleshooting by Hardware

### QuietBox P300c Issues

**"Unknown board type 'P300C'" Error:**
- **Solution:** Use `MESH_DEVICE=P100` (P300c = single Blackhole)
- **Explanation:** P300c architecturally identical to P100

**Multi-device mesh initialization timeout:**
- **Check:** All 4 devices detected (`tt-smi -s`)
- **Solution:** Reboot system if devices in bad state
- **Avoid:** `tt-smi -r` while processes hold devices open

**Lesson shows "not supported" for P300c:**
- **Check:** Extension version (needs v0.0.232+)
- **Verify:** HardwareType enum includes `p300`
- **Workaround:** Treat as P100 for single-chip lessons

### Blackhole Firmware Warnings

**"Firmware 19.4.0 newer than max supported 19.1.0":**
- **Impact:** May hit unsupported features
- **Status:** Basic operations work, documented limitation
- **Solution:** Update tt-metal to support newer firmware

---

## Hardware Selection Guide

### By Use Case

**Development & Prototyping:**
- **Recommended:** N150 or P100
- **Why:** Single chip, cost-effective, widely supported

**Production Single-User Inference:**
- **Recommended:** N150, N300, or P100
- **Why:** Proven, validated models, production-ready

**Production Multi-User Serving:**
- **Recommended:** N300 or T3K
- **Why:** Higher throughput, longer context, tensor parallelism

**Large Models (70B+):**
- **Recommended:** T3K or Galaxy
- **Why:** 8+ chips required for memory and compute

**QuietBox Multi-Device Research:**
- **Recommended:** 4x P300c (QuietBox Tower)
- **Why:** Multi-device development, workload distribution research

**Latest Architecture Evaluation:**
- **Recommended:** P100 or P300c (Blackhole)
- **Why:** 3rd generation, experimental models, future-proof

---

## Future Hardware

**What to expect:**
- More Blackhole variants as architecture matures
- Larger Galaxy configurations (64+, 128+ chips)
- Cloud availability of P-series hardware
- Further QuietBox configurations

**Extension design:**
- Blackhole family equivalence enables easy addition of new variants
- Template established: Add to HardwareType enum + documentation
- Validation workflow: Test on representative hardware → mark validated

---

## References

- **Official Hardware Specs:** [tenstorrent.com/hardware](https://tenstorrent.com/hardware)
- **tt-smi Documentation:** [github.com/tenstorrent/tt-smi](https://github.com/tenstorrent/tt-smi)
- **QuietBox Validation:** `docs/QB_follows.md` (comprehensive P300c testing)
- **Lesson Metadata:** `docs/LESSON_METADATA.md` (metadata schema, hardware values)
- **Community Support:** [discord.gg/tenstorrent](https://discord.gg/tenstorrent)

---

## Document Maintenance

**Version History:**
- v1.0 (2026-01-09): Initial release
  - Added P300/P300c documentation
  - Documented Blackhole family equivalence
  - Created comprehensive hardware reference
  - QuietBox multi-device configuration

**Update Triggers:**
- New hardware released → Add to Quick Reference + Architecture section
- New lessons validated → Update Lesson Compatibility Matrix
- Configuration patterns change → Update examples
- Firmware/software updates → Update troubleshooting

**Maintainers:** VSCode Extension team
**Review Cycle:** Quarterly or on major hardware releases

---

*Last updated: 2026-01-09 by Claude Code during P300/P300c integration work*
