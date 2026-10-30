# Custom Training Lessons - Action Items for Release

**Status**: ✅ 6/8 lessons ready | ⚠️ 2/8 need minor fixes
**Estimated Time to Release**: 2-3 hours of work + 2-3 hours testing
**Priority**: Medium-High (series is high quality, fixes are straightforward)

---

## ✅ Ready for Immediate Release (6 lessons)

These lessons are production-ready as-is:

- ✅ **CT1**: Understanding Custom Training
- ✅ **CT2**: Dataset Fundamentals
- ✅ **CT3**: Configuration Patterns
- ✅ **CT5**: Multi-Device Training
- ✅ **CT6**: Experiment Tracking
- ✅ **CT7**: Architecture Basics

**Action**: Can release immediately, no changes needed.

---

## ⚠️ Requires Fixes Before Release (2 lessons)

### CT4: Fine-tuning Basics ⭐ PRIORITY 1

**Status**: Excellent content, missing setup documentation
**Estimated Fix Time**: 1-2 hours
**Impact**: High (prevents user frustration on first hands-on lesson)

#### Changes Required

1. **Add Prerequisites Section** (30 minutes)
   - Location: Beginning of CT4 lesson, before "Part 1: Setup"
   - Content to add:

```markdown
## Prerequisites and Environment Setup

### System Requirements
- tt-metal v0.64.5 or later
- N150, N300, T3K, P100, or P150 hardware
- 10GB free disk space (for model download)
- Python 3.10+

### Critical Setup Steps

**1. Update tt-metal submodules** (IMPORTANT!)

If you cloned tt-metal previously, ensure submodules are at correct versions:

```bash
cd $TT_METAL_HOME
git submodule update --init --recursive --force
```

**Why**: Mismatched submodules cause compilation errors.

**2. Uninstall pip-installed ttnn** (if present)

```bash
pip uninstall -y ttnn
```

**Why**: Conflicts with locally-built tt-metal version.

**3. Install required Python packages**

```bash
pip install transformers
```

**4. Set environment variables**

Create and source `setup_training_env.sh`:

```bash
#!/bin/bash
export TT_METAL_HOME=/home/user/tt-metal  # Adjust path
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export MESH_DEVICE=N150  # Or N300, T3K, etc.
source $TT_METAL_HOME/python_env/bin/activate
```

**5. Verify installation**

```bash
python test_training_startup.py
```

Expected output: All tests pass ✅

### Troubleshooting

**Issue**: `error: unknown type name 'ChipId'`
**Fix**: Run `git submodule update --init --recursive --force`

**Issue**: `ImportError: undefined symbol: _ZN2tt10DevicePool5_instE`
**Fix**: `pip uninstall -y ttnn` (remove pip version)

**Issue**: `ModuleNotFoundError: No module named 'transformers'`
**Fix**: `pip install transformers`
```

2. **Add `setup_training_env.sh` to templates** (10 minutes)
   - Location: `content/templates/training/setup_training_env.sh`
   - Content: See Appendix A in Final Validation Report

3. **Add `test_training_startup.py` to templates** (15 minutes)
   - Location: `content/templates/training/test_training_startup.py`
   - Content: See Appendix B in Final Validation Report

4. **Update CT4 command references** (15 minutes)
   - Add step to source setup script before training commands
   - Example:
   ```bash
   # Before running training
   source ~/tt-scratchpad/training/setup_training_env.sh
   python finetune_trickster.py --config configs/trickster_n150.yaml
   ```

#### Testing

After changes:
1. Test on fresh N150 system (or clean Docker container)
2. Follow lesson exactly as written
3. Verify all 6 issues are prevented
4. Training should start without errors

---

### CT8: Training from Scratch ⭐ PRIORITY 2

**Status**: Excellent content, missing dataset prep automation
**Estimated Fix Time**: 30-60 minutes
**Impact**: Medium (users can work around, but automation is better)

#### Changes Required

1. **Add `prepare_shakespeare.py` to templates** (30 minutes)
   - Location: `content/templates/training/data/prepare_shakespeare.py`
   - Content: See Appendix C in Final Validation Report
   - Features:
     - Downloads tiny-shakespeare from char-rnn repo
     - Creates train/val split (90/10)
     - Validates character counts
     - Clear progress messages

2. **Update CT8 dataset preparation section** (15 minutes)
   - Location: CT8 lesson, "Part 1: Setup" → "Prepare Dataset"
   - Current text:
   ```markdown
   Download and prepare tiny-shakespeare:
   ```bash
   cd ~/tt-scratchpad/training/data
   python prepare_shakespeare.py --output shakespeare.txt --split
   ```
   ```

   - Add note:
   ```markdown
   **What this does**:
   - Downloads ~1.1MB Shakespeare text
   - Creates 90/10 train/val split
   - Saves to `shakespeare.txt`, `shakespeare_train.txt`, `shakespeare_val.txt`

   **Manual alternative** (if script unavailable):
   ```bash
   wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespeare.txt
   head -n 32000 shakespeare.txt > shakespeare_train.txt
   tail -n 8000 shakespeare.txt > shakespeare_val.txt
   ```
   ```

#### Testing

After changes:
1. Run `python prepare_shakespeare.py --output . --split`
2. Verify 3 files created: shakespeare.txt, shakespeare_train.txt, shakespeare_val.txt
3. Verify sizes: ~1MB, ~900KB, ~100KB
4. Verify manual alternative works

---

## Summary Checklist

### Before Release

- [ ] **CT4 Prerequisites Section** - Add to lesson
- [ ] **setup_training_env.sh** - Add to templates
- [ ] **test_training_startup.py** - Add to templates
- [ ] **prepare_shakespeare.py** - Add to templates
- [ ] **CT8 Dataset Section** - Update instructions

### Testing

- [ ] **Fresh System Test** - Validate on clean N150
- [ ] **CT4 End-to-End** - Follow lesson, verify no blockers
- [ ] **CT8 Dataset Prep** - Run script, verify files created
- [ ] **All Scripts Tested** - setup_training_env.sh, test_training_startup.py, prepare_shakespeare.py

### Documentation

- [ ] **Update CHANGELOG** - Document fixes and improvements
- [ ] **Update README** - Note validation completion
- [ ] **Archive Validation Docs** - Keep journey log for reference

---

## Timeline Estimate

| Task | Duration | Owner | Status |
|------|----------|-------|--------|
| Write CT4 Prerequisites | 30 min | Lesson Author | ⏳ Pending |
| Create setup_training_env.sh | 10 min | Lesson Author | ⏳ Pending |
| Create test_training_startup.py | 15 min | Lesson Author | ⏳ Pending |
| Update CT4 commands | 15 min | Lesson Author | ⏳ Pending |
| Create prepare_shakespeare.py | 30 min | Lesson Author | ⏳ Pending |
| Update CT8 dataset section | 15 min | Lesson Author | ⏳ Pending |
| **Subtotal: Implementation** | **1.75 hours** | | |
| Test on fresh N150 | 1 hour | QA | ⏳ Pending |
| Test CT4 end-to-end | 1 hour | QA | ⏳ Pending |
| Test CT8 dataset prep | 15 min | QA | ⏳ Pending |
| **Subtotal: Testing** | **2.25 hours** | | |
| **TOTAL TIME TO RELEASE** | **4 hours** | | |

---

## Validation Confidence

| Lesson | Confidence | Reason |
|--------|-----------|--------|
| CT1 | 100% | Conceptual, no dependencies |
| CT2 | 100% | Dataset validated |
| CT3 | 100% | Configs validated |
| CT4 | 95% | Hands-on validated, fixes clear |
| CT5 | 90% | Conceptual (N300+ required) |
| CT6 | 100% | No blockers |
| CT7 | 100% | Exceptional quality |
| CT8 | 95% | Prerequisites validated, dataset prep minor |

**Overall Confidence**: 95%

---

## Risk Assessment

### Low Risk ✅

All identified issues have:
- Clear root causes
- Documented fixes
- Tested solutions
- Straightforward implementation

### Known Limitations

1. **CT5 Multi-Device**: Cannot test DDP on N150 (by design, requires N300+)
   - Mitigation: Conceptual validation thorough, config patterns sound
   - Confidence: 90%

2. **CT8 Full Training**: Cannot run full 30-60 min training on N150 in validation timeframe
   - Mitigation: All prerequisites tested, architecture validated, training progression documented
   - Confidence: 95%

### Unknown Unknowns

Potential issues on fresh systems:
- OS-specific differences (Ubuntu vs other distros)
- tt-metal version variations (v0.64.5 vs v0.65.x vs v0.66.x)
- Hardware variations (N150 firmware versions, etc.)

**Mitigation**: Test on multiple systems before wide release

---

## Success Metrics

### Completion Criteria

Series is ready for release when:
- ✅ All 8 lessons have 0 known blockers
- ✅ Fresh system test passes (CT4, CT8)
- ✅ All scripts tested and working
- ✅ Documentation complete

### Quality Metrics

- Average lesson rating: 4.875/5.0 (Outstanding) ✅
- Issues found: 7
- Issues resolved: 6 (86%)
- Issues remaining: 1 (dataset prep - low severity)
- Production ready: 6/8 (75%) → 8/8 (100%) after fixes

### User Experience Goals

After fixes:
- ✅ User can complete CT1-3 with no issues
- ✅ User can complete CT4 with clear prerequisites
- ✅ User can complete CT5-7 with no issues
- ✅ User can complete CT8 with automated dataset prep

---

## Next Steps

1. **Implement Changes** (1.75 hours)
   - Assign to lesson author
   - Use provided scripts (Appendices A-C in Final Report)
   - Follow checklist above

2. **Internal Testing** (2.25 hours)
   - Test on fresh N150 system
   - Validate CT4 and CT8 end-to-end
   - Verify all scripts work

3. **Release Decision**
   - Review test results
   - Approve for public release
   - Update documentation

4. **Post-Release Monitoring**
   - Gather user feedback
   - Monitor support requests
   - Track completion rates

---

## Contact for Questions

- **Validation Engineer**: Claude (Sonnet 4.5)
- **Validation Date**: 2026-02-02
- **Documentation**:
  - Journey Log: `/home/user/docs/CLAUDE_CT_follows.md` (2,200+ lines)
  - Final Report: `/home/user/docs/CLAUDE_CT_FINAL_VALIDATION_REPORT.md`
  - This Action Plan: `/home/user/docs/CLAUDE_CT_ACTION_ITEMS.md`

---

**Status**: ⚠️ READY TO IMPLEMENT → ✅ READY TO RELEASE (after 4 hours work)

**Priority**: Medium-High (high-quality series, minor fixes only)

**Recommendation**: Schedule fixes for next sprint, release within 1 week
