# Lesson Metadata System

## Overview

Each walkthrough step in `package.json` can include metadata for hardware compatibility, validation status, and release gating.

## Metadata Schema

```typescript
{
  "id": "lesson-id",
  "title": "Lesson Title",
  "description": "Lesson description",
  "media": { "markdown": "content/lessons/XX-lesson.md" },
  "completionEvents": ["onCommand:..."],

  // Metadata fields (all optional)
  "metadata": {
    // Hardware support
    "supportedHardware": ["n150", "n300", "t3k", "p100", "p150", "galaxy", "simulator"],

    // Validation status
    "status": "draft" | "validated" | "blocked",

    // Which hardware configs have been validated
    "validatedOn": ["n150", "n300"],

    // Optional: reason if blocked
    "blockReason": "Waiting for firmware update",

    // Optional: minimum tt-metal version required
    "minTTMetalVersion": "v0.51.0"
  }
}
```

## Hardware Values

Use lowercase values:
- `n150` - Wormhole N150 (single chip)
- `n300` - Wormhole N300 (dual chip)
- `t3k` - T3000 (8-chip cluster)
- `p100` - Blackhole P100 (single chip)
- `p150` - Blackhole P150 (dual chip)
- `p300` - Blackhole P300/P300c (single chip, QuietBox variant)
- `galaxy` - Galaxy (32-chip cluster)
- `simulator` - TTSim (no hardware required)

## Status Values

### `draft`
- Lesson content is being developed
- Not yet tested on hardware
- Should be hidden in production releases
- Visible in development/staging builds

### `validated`
- Lesson has been tested and confirmed working
- At least one hardware config in `validatedOn` array
- Ready for production release
- Shows normally in walkthrough

### `blocked`
- Lesson temporarily unavailable
- Known issue preventing completion
- Should show in UI with warning/explanation
- Use `blockReason` to explain why

## Usage Examples

### Generic lesson (all hardware)
```json
{
  "id": "hardware-detection",
  "title": "Hardware Detection",
  "metadata": {
    "supportedHardware": ["n150", "n300", "t3k", "p100", "p150", "p300", "galaxy"],
    "status": "validated",
    "validatedOn": ["n150", "n300", "t3k", "p100", "p300"]
  }
}
```

### Hardware-specific lesson
```json
{
  "id": "vllm-production",
  "title": "Production Inference with vLLM",
  "metadata": {
    "supportedHardware": ["n150", "n300", "t3k", "p100"],
    "status": "validated",
    "validatedOn": ["n150", "n300"],
    "minTTMetalVersion": "v0.51.0"
  }
}
```

### Draft lesson
```json
{
  "id": "new-feature",
  "title": "Experimental Feature",
  "metadata": {
    "supportedHardware": ["n150"],
    "status": "draft",
    "validatedOn": []
  }
}
```

### Blocked lesson
```json
{
  "id": "advanced-feature",
  "title": "Advanced Feature",
  "metadata": {
    "supportedHardware": ["p100"],
    "status": "blocked",
    "validatedOn": [],
    "blockReason": "Waiting for firmware v2.0 with required API"
  }
}
```

## Implementation Guide

### 1. Extension Code (TypeScript)

Read metadata from `package.json` at runtime:

```typescript
interface LessonMetadata {
  supportedHardware?: string[];
  status?: 'draft' | 'validated' | 'blocked';
  validatedOn?: string[];
  blockReason?: string;
  minTTMetalVersion?: string;
}

interface WalkthroughStep {
  id: string;
  title: string;
  description: string;
  media: { markdown: string };
  completionEvents: string[];
  metadata?: LessonMetadata;
}

// Filter lessons by status (for release gating)
function getPublicLessons(steps: WalkthroughStep[]): WalkthroughStep[] {
  return steps.filter(step =>
    step.metadata?.status === 'validated' ||
    step.metadata?.status === undefined // backward compat
  );
}

// Filter by hardware
function getLessonsForHardware(steps: WalkthroughStep[], hardware: string): WalkthroughStep[] {
  return steps.filter(step =>
    !step.metadata?.supportedHardware ||
    step.metadata.supportedHardware.includes(hardware.toLowerCase())
  );
}

// Check if lesson is validated for specific hardware
function isValidatedForHardware(step: WalkthroughStep, hardware: string): boolean {
  return step.metadata?.validatedOn?.includes(hardware.toLowerCase()) ?? false;
}
```

### 2. Release Process

The `scripts/check-lessons.js` script validates lesson metadata before release:

```bash
# Check lesson status (default)
node scripts/check-lessons.js

# Strict mode - exit 1 if any draft/blocked lessons
node scripts/check-lessons.js --strict

# Generate filtered package.json with only validated lessons
node scripts/check-lessons.js --filter
```

**Script output includes:**
- Total lessons by status (validated/draft/blocked)
- List of draft and blocked lessons
- Lessons with no validation on any hardware
- Hardware validation coverage (lessons per hardware type)
- Optional: Generate `package.filtered.json` for release builds

### 3. Development Workflow

1. **Creating new lesson:**
   - Set `status: "draft"`
   - Set `supportedHardware` based on requirements
   - Leave `validatedOn` empty

2. **Testing on hardware:**
   - Test lesson on specific hardware
   - Document any issues
   - Add hardware to `validatedOn` when confirmed working

3. **Ready for release:**
   - Change `status: "validated"`
   - Ensure at least one entry in `validatedOn`
   - Update documentation if needed

4. **Blocking lesson:**
   - Change `status: "blocked"`
   - Add `blockReason` with explanation
   - Create issue to track resolution

## Benefits

1. **Release gating** - Only ship validated lessons
2. **Hardware filtering** - Show relevant lessons per hardware
3. **Quality tracking** - Know which configs are tested
4. **Clear status** - Draft/validated/blocked states
5. **Backward compatible** - Metadata is optional
6. **Developer clarity** - Easy to see what needs testing

## Current Lesson Status (Example)

| Lesson | Status | Supported HW | Validated On |
|--------|--------|--------------|--------------|
| 01 Hardware Detection | validated | all | n150, n300, t3k, p100 |
| 02 Verify Installation | validated | all | n150, n300 |
| 03 Download Model | validated | all | n150, n300 |
| 04 Interactive Chat | validated | all | n150 |
| 05 API Server | validated | all | n150 |
| 06 tt-inference-server | validated | n150, n300, t3k, p100 | n150 |
| 07 vLLM Production | validated | n150, n300, t3k, p100 | n150 |
| 08 VSCode Chat | validated | all | n150 |
| 09 Image Generation | validated | n150, n300, t3k, p100 | n150 |
| 10 Coding Assistant | validated | all | n150 |
| 11 TT-Forge | draft | n150 | - |
| 12 TT-XLA JAX | validated | n150, n300, t3k, galaxy | n150 |
| 13 RISC-V Programming | validated | all | n150 |
| 14 Metalium Cookbook | validated | all | n150 |

## Blackhole Family Equivalence

All Blackhole cards share the same core architecture and instruction set:

**Blackhole Variants:**
- **P100**: Single Blackhole chip (cloud/standalone deployments)
- **P150**: Dual Blackhole chip (higher performance configurations)
- **P300/P300c**: Single Blackhole chip (QuietBox systems, compute variant)

**Architecture Principle:**
> "Anything that can run on one Blackhole card should be able to run on any one of Blackhole cards"

**Lesson Design Guidelines:**
- Lessons supporting `p100` should include `p300` in `supportedHardware`
- All Blackhole variants use `TT_METAL_ARCH_NAME=blackhole`
- Single-chip lessons: Use `MESH_DEVICE=P100` for all variants
- Dual-chip lessons: Use `MESH_DEVICE=P150` (P150 only)

**Multi-Device QuietBox Systems:**
- QuietBox Tower (4x P300c) = 4 separate devices, each single-chip
- For single-chip lessons: Use device 0 only
- For multi-device lessons: All 4 devices available for parallelization
- Example: Particle Life multi-device achieves 2x speedup on 4x P300c

**Validation Notes:**
- P300c validated on Lesson 7 (vLLM Production) and Lesson 15 (Metalium Cookbook - Particle Life)
- P300c runs identically to P100 with `MESH_DEVICE=P100`
- See `docs/QB_follows.md` for comprehensive QuietBox validation results

## Migration Plan

1. ✅ Define metadata schema (this document)
2. ⬜ Add metadata to all lessons in package.json
3. ⬜ Create lesson validation script
4. ⬜ Update extension.ts to read and use metadata
5. ⬜ Test filtering logic
6. ⬜ Document in CLAUDE.md
