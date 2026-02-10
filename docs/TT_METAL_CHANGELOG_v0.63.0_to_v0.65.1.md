# TT-Metal Changelog: v0.63.0 → v0.65.1

**Date**: 2026-02-10
**Documented by**: Claude AI Agent
**Context**: Extension validation and Docker image updates

---

## Executive Summary

Between **tt-metal v0.63.0 → v0.65.1**, there are two major release waves:

1. **v0.64.x line** - Consolidation, accuracy/performance work, Blackhole enablement
2. **v0.65.x line** - Fabric/autopacketization, new infrastructure migration, multi-host scale-out, Wormhole/Blackhole QoL

**v0.65.1** is a stabilization spin on v0.65.0 with packaging fixes and better diagnostics.

---

## 1. Models & Demos

### v0.64.x Changes

**Vision & Diffusion:**
- Expanded SDXL/SD3.5 demos: refactors, inpainting/encoder tests, VAE GN updates
- Better device performance coverage and accuracy thresholds
- More robust PCC/accuracy checks across SDXL pipelines

**Speech (Whisper):**
- Improved coverage and performance
- Decoder/encoder optimization, new model parameters
- Blackhole performance tests, DP support

**Vision Models (YOLO/ResNet/Swin/SegFormer/UNet):**
- Performance bumps and test restructuring
- DP support on N300
- Panoptic/segmentation optimizations
- TT-CNN-based performance harnesses
- Realistic CI performance targets for WH/BH/T3K

**LLMs:**
- New demos: DeepSeek, Gemma3, Mixtral, Qwen-VL, Wan2.2, phi-4
- Better performance/correctness tests
- vLLM integration including multimodal benchmarks

### v0.65.x Additions

**Large Language Models:**
- **TG Qwen3-32B** path with shared optimizations from Llama-3.3-70B
- Higher tokens/sec per user
- SDXL combined pipelines (base+refiner)
- More TT-DiT/Mochi work for SD3.5/Wan2.2

**Continued Improvements:**
- **DeepSeek V3** and large Llama/Gemma paths
- Ring SDPA, batched prefill
- Structured decoding on Galaxy
- Better token-matching and accuracy tooling

### Lesson Implications

**Affected Lessons:**
- `image-generation` - SDXL improvements, VAE updates
- `video-generation-ttmetal` - SD3.5 refinements, combined pipelines
- `vllm-production` - Better vLLM integration, multimodal support
- `download-model` - Could recommend Qwen3-32B for larger deployments
- Future lesson opportunity: Whisper speech-to-text demo

---

## 2. TT-NN / Ops / Numerics

### v0.64.x Changes

**TensorAccessor Migration:**
- Large-scale migration off legacy addrgen APIs
- Affected: elementwise, data-movement, conv, normalization, CCL, experimental ops
- Unified accessor-based infrastructure

**New/Updated Device Ops:**
- **Dtype support**: uint16, uint32, int32 for many ops
- Ops improved: pad, eq/ne, rdiv, exp/exp2/expm1, log, softshrink, elu, gelu, tanhshrink
- Better NaN/Inf handling
- Sharding-aware behavior for layernorm/softmax/groupnorm

**Accuracy-Focused Rewrites:**
- **Softmax / tanh / sigmoid / exp / log families**
- Higher-precision polynomial implementations
- exp_21f-style implementations
- Numeric-stable softmax by default
- Updated golden functions and sweeps

**Robust Operations:**
- **grid_sample / pool / strided slice / reshard / fold**
- Better behavior on sharded and non-32-aligned shapes
- Height sharding, batched grids
- Tiled-output pool, adaptive/ceil-mode
- Reshard iterators, ND support

### v0.65.x Additions

**New Infrastructure Migration:**
- Attention building blocks migrated:
  - group_attn_matmul, attn_matmul
  - QKV-head construct/split ops
  - prefix_scan, bcast, prod, argmax
  - neighbor_pad_async
  - nlp_create_qkv_heads_segformer
- All on same kernel/runtime scaffolding

**Sharded-Aware Composites:**
- Composite AG/RS/All-Reduce extended to more dims and patterns
- Scatter-write optimizations
- Sharded support for composite all-reduce
- Updated barriers/semantics for correctness and performance

**Memory Safety:**
- ND reshard & slice_write alignment fixes
- Sharded embedding alignment
- Tiled pad/pool with correct alignment
- Removal of legacy 32-tile assumptions in clone/copy paths

### Lesson Implications

**Affected Lessons:**
- All **Cookbook** lessons (cookbook-game-of-life, cookbook-audio-processor, cookbook-mandelbrot, cookbook-image-filters, cookbook-particle-life)
  - Better op accuracy and stability
  - More dtype options for experimentation
- **Custom Training** series (ct1-ct8)
  - More accurate gradients from better numerics
  - Better sharding support for distributed training
- `explore-metalium` - Should mention TensorAccessor as modern API pattern

---

## 3. Fabric, Scale-Out & Multi-Host

### v0.64.x Changes

**Fabric 2D Torus:**
- Fabric 2D torus/dynamic support and tests
- Programming Multiple Meshes tech reports
- 2D torus fixtures, stability suites
- Multi-host big-mesh tests
- Multi-mesh programming docs

**Fabric-Based Collectives:**
- Shift from legacy CCLs to fabric-based collectives
- New linear/1D APIs
- 2D fabric tests
- Multi-host aware serialization and routing
- Deprecation of old CCL tech reports

**Multi-Host Discovery:**
- Better physical discovery for Galaxy/QuietBox
- PhysicalSystemDescriptor validation
- Multi-host mock cluster tests
- Cabling generator
- BH Galaxy descriptors
- Per-tray isolation workflows

### v0.65.x Additions

**Autopacketization:**
- New autopacketization for fabric data-movement ops
- More granular fabric telemetry
- Multi-iteration BW reporting for point-to-point and CCL tests

**Advanced Fabric Features:**
- **Inter-mesh traffic on dedicated VC (VC1)**
- Fabric unicast/multicast scatter with multi-chunk support
- Low-latency modes wired into tracing and tests

**Mature Routing:**
- **1D routing / torus / 1×32 mappings** for Galaxy/T3K
- Hop-distance APIs
- Corrected mapping for flipped coordinates
- Torus-aware Galaxy CI
- Cabling/spec tooling

### Lesson Implications

**Affected Lessons:**
- `ct5-multi-device-training` - **PRIORITY UPDATE**
  - Mention fabric improvements and autopacketization
  - Reference new 1D/2D collective APIs
  - Better multi-host stability
- `hardware-detection` - Note improved Galaxy/T3K discovery
- Future lesson opportunity: Deep dive on TT-Fabric programming

---

## 4. Hardware Enablement & Runtime Behavior

### Blackhole (P100/P150)

**Enablement:**
- 20-core enablement
- BH-specific performance targets (ResNet, UNet, SDXL, Mamba)
- BH PCIe micro-benchmarks
- .deb packages
- BH loudbox/quietbox CI
- BH rackbox setups
- BH GLX descriptors and link-up checks
- Multiple UMD bumps to modernize BH behavior

**Fixes:**
- BH CCL/all-to-all/all-broadcast hangs resolved
- DRAM arbiter edge-cases fixed
- NoC inline writes improved
- L1 cache behavior stabilized
- Two-TXQ modes in fabric

### Wormhole & Galaxy (N150/N300/T3K/Galaxy)

**Galaxy Coverage:**
- 1×32 meshes support
- Dual/quad-Galaxy infrastructure
- Big-mesh Llama-70B/DeepSeek pipelines
- Ether-mux tests
- Multi-host Galaxy nightly/frequent tests (renamed from "TG")

**Wormhole Improvements:**
- **Pinned host memory on Wormhole**
- Improved DRAM/L1 prefetch behavior
- More robust NoC tracing and NoC sanitizer for large PCIe ranges

### Timesharing & Throttling

- Timeshare improvements (support for ×4 timeshare configs)
- **Dynamic power throttling on Blackhole**
- Better throttling heuristics in convs and fabric tests

### Lesson Implications

**Affected Lessons:**
- All **P100/P150** lessons - Note improved stability and 20-core support
- All **Galaxy/T3K** lessons - Better multi-host support, dual/quad-Galaxy
- `hardware-detection` - Update to mention improved device discovery
- All lessons should benefit from better DRAM/L1 prefetch on Wormhole

---

## 5. Tooling, Debugging, CI & Packaging

### Debugger / Profiler / Telemetry

**TT-Triage Expansions:**
- Automatic hang detection
- Multi-host integration
- Better event/GO-message/waypoint reporting
- RPC over Cap'n Proto
- Slack alerts

**New tt_telemetry Tool:**
- Device and link health GUI
- Per-zone traces
- Multi-iteration BW reporting
- Profiling thread-pool
- Larger configurable DRAM buffers
- Easy dump-on-demand hooks

**Watcher/Tracer Improvements:**
- Sanitized local L1 writes
- Better error messages (TLB and CG errors)
- Expanded asserts for conv/AG/RS/tiny-tile cases

### Documentation & Examples

**New TT-Metalium Examples:**
- Hello World
- DRAM loopback
- Eltwise operations
- Matmul variants
- Detailed Matmul/Reduce/Fused docs

**Updated Documentation:**
- Metal programming guide
- Tech reports
- INSTALLING.md (with tt-installer and Galaxy requirements)
- TT-CNN README
- Model-builder API for vision models
- Structured perf/accuracy documentation
- Sweep frameworks for kernels and CCLs

### CI / Infrastructure / Packaging

**Test Infrastructure:**
- Move to **MeshDevice + TT-Fabric APIs** (no more legacy host runtime)
- Cleanup of dispatch classes
- LLK auto-uplift flows
- Heavy CIv2 rollout with vIOMMU runners for WH/BH/P300/N300

**Packaging:**
- **.deb packages for tt-metalium/tt-nn**
- Wheel build refactors
- Upload to Cloudsmith
- New "models" Docker image line
- Used by tt-installer and downstream repos

### Lesson Implications

**Affected Lessons:**
- `explore-metalium` - **PRIORITY UPDATE**
  - Reference new TT-Metalium examples (Hello World, DRAM loopback, eltwise, matmul)
  - Link to updated programming guide
  - Mention tt_telemetry tool for debugging
- `tt-installer` - Note improved .deb packaging
- `verify-installation` - Mention better diagnostics
- Future lesson opportunity: Debugging with tt-triage and tt_telemetry

---

## 6. v0.65.1 vs v0.65.0

**v0.65.1 is a stabilization spin** on v0.65.0:

### Tagged Artifacts

- `.deb / wheel artifacts` used by tt-inference-server
- Example: `tt-nn_0.65.1.ubuntu22.04_amd64.deb`

### Build/Install Fixes

- Handling missing `capstone.h` on modern distros
- xtensor patching when consuming TT-NN via `find_package`
- CMake-ecosystem hygiene for external users

### Runtime QoL

- Better diagnostics around TLB allocation failures
- Watcher/UMD updates from main
- Reduced "mysterious TLB allocation" issues on BH/LB machines

### Lesson Implications

- All lessons benefit from better build/install experience
- Better error messages help with troubleshooting sections

---

## Summary of Impact

### What We Get Moving v0.63.0 → v0.65.1

✅ **Broader and better-tested model coverage**
- LLMs (DeepSeek, Gemma3, Qwen-VL, Qwen3-32B)
- Diffusion (SDXL, SD3.5, combined pipelines)
- Vision (YOLO, ResNet, Swin, SegFormer, UNet)
- Speech (Whisper improvements)

✅ **Substantially more mature TT-Fabric + multi-host**
- Autopacketization, VC1 inter-mesh traffic
- 1D/2D routing for Galaxy/T3K
- Better multi-host discovery and stability

✅ **More accurate and dtype-rich TT-NN op surface**
- uint16/uint32/int32 support
- Improved numerics (softmax, tanh, sigmoid, exp, log)
- Better sharding-aware operations

✅ **Cleanup and modernization**
- TensorAccessor migration
- Improved CCLs via fabric
- Better profiling and debugging tools
- Professional packaging (.deb, wheels)

### No Breaking API Changes

- Many behavior/accuracy improvements
- Users should compare old vs new results for validation
- Better error messages and diagnostics throughout

---

## Lesson Enhancement Priorities

### High Priority Updates Needed

1. **explore-metalium** - Add new TT-Metalium examples
2. **ct5-multi-device-training** - Mention fabric improvements
3. **image-generation** - Note SDXL improvements
4. **vllm-production** - Better vLLM integration notes

### Medium Priority Updates

5. **hardware-detection** - Improved Galaxy/T3K discovery
6. **verify-installation** - Better diagnostics notes
7. All **Cookbook** lessons - Better op accuracy

### Future Lesson Opportunities

8. **Debugging with tt_telemetry** - New tooling lesson
9. **TT-Fabric Programming Deep Dive** - Advanced multi-host
10. **Whisper Speech-to-Text** - New model demo

---

## References

- tt-metal v0.64.0 changelog
- tt-metal v0.64.4 changelog
- tt-metal v0.64.5 changelog
- tt-metal v0.65.0-dev changelogs
- tt-metal v0.65.1 release notes
- Internal validation testing (2026-02-10)

---

**Document prepared for**: tt-vscode-toolkit v0.0.310
**Validation date**: 2026-02-10
**Hardware tested**: N150 L (Wormhole - Single Chip)
