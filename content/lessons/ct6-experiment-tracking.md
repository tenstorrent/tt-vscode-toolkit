---
id: ct6-experiment-tracking
title: Experiment Tracking
description: >-
  Master experiment tracking with file-based logging and Weights & Biases (WandB) integration. Compare hyperparameter variations, visualize training curves, and manage experiments professionally. Make data-driven training decisions.
category: custom-training
tags:
  - experiment-tracking
  - wandb
  - logging
  - visualization
  - hyperparameters
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
---

# Experiment Tracking

Learn to track, compare, and visualize training experiments using file-based logging and optional WandB integration.

## What You'll Learn

- File-based experiment tracking (baseline)
- Weights & Biases (WandB) integration
- Comparing hyperparameter variations
- Visualizing training curves
- Best practices for experiment management

**Time:** 10-15 minutes | **Prerequisites:** CT-4 (Fine-tuning Basics)

---

## Why Track Experiments?

### The Problem

You run 10 training experiments with different hyperparameters:

```
Which batch size worked best?
Which learning rate converged fastest?
Did that checkpoint from Tuesday outperform today's?
```

**Without tracking:** Scroll through terminal logs, compare files manually, rely on memory.

**With tracking:** Compare all runs at a glance, see visualizations, make data-driven decisions.

---

## Approach 1: File-Based Tracking (Baseline)

### What's Already Tracked

The `finetune_trickster.py` script automatically logs:

**1. Training log:**
```
output/training.log
```

Contains:
- Hyperparameters
- Loss per step
- Validation results
- Checkpoint saves
- Errors/warnings

**2. Validation samples:**
```
output/validation.txt
```

Contains:
- Generated responses at each validation step
- Qualitative improvement over time

**3. Training curves:**
```
output/training_curves.png
```

Visualizes:
- Training loss over time
- Validation loss over time

### Organizing Experiments

**Bad (hard to track):**
```
output/
  final_model/
  training.log
```

**Good (organized by date/name):**
```
experiments/
  2026-02-01_baseline/
    config.yaml
    training.log
    validation.txt
    training_curves.png
    final_model/

  2026-02-01_higher_lr/
    config.yaml
    training.log
    validation.txt
    training_curves.png
    final_model/
```

### Manual Comparison Script

Create `compare_experiments.sh`:

```bash
#!/bin/bash

echo "Experiment Comparison"
echo "===================="

for exp in experiments/*/; do
  echo ""
  echo "Experiment: $(basename $exp)"

  # Extract final loss
  final_loss=$(tail -5 "$exp/training.log" | grep "Final training loss" | awk '{print $NF}')
  echo "  Final Loss: $final_loss"

  # Extract config values
  lr=$(grep "learning_rate:" "$exp/config.yaml" | awk '{print $2}')
  batch=$(grep "batch_size:" "$exp/config.yaml" | awk '{print $2}')
  echo "  LR: $lr, Batch: $batch"
done
```

**Run:**
```bash
chmod +x compare_experiments.sh
./compare_experiments.sh
```

**Output:**
```
Experiment Comparison
====================

Experiment: 2026-02-01_baseline
  Final Loss: 1.84
  LR: 0.0001, Batch: 8

Experiment: 2026-02-01_higher_lr
  Final Loss: 1.92
  LR: 0.0002, Batch: 8
```

---

## Approach 2: WandB Integration (Recommended)

### What is Weights & Biases?

**WandB** is a cloud-based experiment tracking platform:

- 📊 Real-time loss curves
- 🔍 Compare runs side-by-side
- 📝 Automatic hyperparameter logging
- 🖼️ Log sample outputs
- 👥 Share with team
- 🆓 Free tier available

**Website:** [wandb.ai](https://wandb.ai)

### Setup (One-Time)

**1. Create account:**
```bash
# Visit wandb.ai and sign up (free)
```

**2. Install WandB:**
```bash
pip install wandb
```

**3. Login:**
```bash
wandb login
```

Paste your API key when prompted.

**4. Enable in config:**
```yaml
# configs/trickster_n150_wandb.yaml
training_config:
  # ... other settings ...

  use_wandb: true                  # Enable WandB
  wandb_project: "trickster-finetune"
  wandb_run_name: "n150-baseline"
```

### Running with WandB

```bash
cd ~/tt-scratchpad/training
python finetune_trickster.py --config configs/trickster_n150_wandb.yaml
```

**What gets logged:**
- Loss (training and validation)
- Learning rate (if using scheduler)
- Hyperparameters (from config)
- Generated samples (validation text)
- System metrics (GPU usage, memory)
- Model checkpoints (optional)

### WandB Dashboard

After training starts, you'll see:

```
wandb: 🚀 View run at https://wandb.ai/your-username/trickster-finetune/runs/abc123
```

**Dashboard shows:**

1. **Overview tab:**
   - Run summary (final loss, duration)
   - Hyperparameters
   - System info

2. **Charts tab:**
   - Real-time loss curves
   - Custom plots
   - Compare with other runs

3. **Logs tab:**
   - Generated text samples
   - Validation outputs

4. **Files tab:**
   - Config files
   - Saved artifacts

---

## Comparing Experiments

### Scenario: Finding Best Learning Rate

You want to try 3 learning rates: 5e-5, 1e-4, 2e-4

**1. Run three experiments:**

```bash
# Experiment 1: LR = 5e-5
python finetune_trickster.py \
  --config configs/trickster_n150_lr_5e5.yaml

# Experiment 2: LR = 1e-4
python finetune_trickster.py \
  --config configs/trickster_n150_lr_1e4.yaml

# Experiment 3: LR = 2e-4
python finetune_trickster.py \
  --config configs/trickster_n150_lr_2e4.yaml
```

**2. Compare in WandB:**

Go to your project page, click "Compare runs":
- Select all 3 runs
- View overlaid loss curves
- Check final validation loss
- See sample outputs side-by-side

**3. Identify best:**

```
Run 1 (5e-5):  Final val loss: 2.34  (too slow)
Run 2 (1e-4):  Final val loss: 2.12  (best!)
Run 3 (2e-4):  Final val loss: 2.28  (too aggressive)
```

**Conclusion:** LR = 1e-4 is optimal.

---

## Advanced WandB Features

### 1. Logging Custom Metrics

Add to your training script:

```python
import wandb

# After optimizer step
wandb.log({
    "train_loss": avg_loss,
    "learning_rate": current_lr,
    "gradient_norm": grad_norm,
    "step": opt_step
})

# After validation
wandb.log({
    "val_loss": val_loss,
    "sample_output": generated_text,
    "step": opt_step
})
```

### 2. Hyperparameter Sweeps

Automate hyperparameter search:

```yaml
# sweep.yaml
program: finetune_trickster.py
method: grid
parameters:
  learning_rate:
    values: [5e-5, 1e-4, 2e-4]
  batch_size:
    values: [8, 16]
```

**Run sweep:**
```bash
wandb sweep sweep.yaml
wandb agent your-username/trickster-finetune/sweep-id
```

WandB automatically runs all combinations!

### 3. Model Artifacts

Save checkpoints to WandB:

```python
import wandb

# After saving checkpoint
artifact = wandb.Artifact('trickster-model', type='model')
artifact.add_dir('output/final_model')
wandb.log_artifact(artifact)
```

**Benefits:**
- Checkpoint versioning
- Easy download from any machine
- Link models to experiments

### 4. Group Experiments

Organize related runs:

```python
wandb.init(
    project="trickster-finetune",
    group="lr-search",              # Group related experiments
    tags=["n150", "baseline"],      # Add tags for filtering
)
```

---

## Best Practices for Experiment Management

### 1. Naming Convention

**Use descriptive names:**

```
Good:  "2026-02-01_n150_lr1e4_batch8_baseline"
Bad:   "experiment_1"
```

**Include key info:**
- Date
- Hardware (n150, n300)
- Key hyperparameters
- Purpose (baseline, ablation, etc.)

### 2. Version Control Configs

```bash
# Save configs alongside code
git add configs/trickster_n150_lr1e4.yaml
git commit -m "Add config for LR=1e-4 experiment"
git tag exp-lr1e4
```

**Why:** Reproducibility - know exactly what config produced results.

### 3. Document Results

Create `experiments.md`:

```markdown
# Trickster Fine-tuning Experiments

## Experiment 1: Baseline (2026-02-01)
- **Config:** trickster_n150.yaml
- **Hardware:** N150
- **Duration:** 2.3 hours
- **Final Loss:** 1.84 (train), 2.12 (val)
- **Result:** Good baseline, will try higher LR next
- **WandB:** [link](https://wandb.ai/...)

## Experiment 2: Higher LR (2026-02-01)
- **Config:** trickster_n150_lr2e4.yaml
- **Hardware:** N150
- **Duration:** 2.1 hours
- **Final Loss:** 1.92 (train), 2.28 (val)
- **Result:** Slightly worse, LR=1e-4 is better
- **WandB:** [link](https://wandb.ai/...)
```

### 4. Archive Failed Experiments

Don't delete failures - they teach you what doesn't work!

```
experiments/
  successful/
    2026-02-01_baseline/
  failed/
    2026-01-30_lr_too_high/       # Exploded at step 50
    2026-01-31_batch_too_large/   # OOM error
```

### 5. Regular Cleanup

Keep last 5 checkpoints, archive older:

```bash
# Keep only step 400, 500 checkpoints
rm -rf output/checkpoint_step_100
rm -rf output/checkpoint_step_200
rm -rf output/checkpoint_step_300

# Or archive to S3/NAS
tar -czf checkpoints_baseline.tar.gz output/
mv checkpoints_baseline.tar.gz /archive/
```

---

## Visualization Tips

### Loss Curve Analysis

**Healthy training:**
```
Loss
  4 |*
    | *
  3 |  **
    |    ***
  2 |       *****
    |            -------
  1 |___________________
    0   100   200   300   400   500
                Steps
```
- Smooth decrease
- Plateaus near end
- Val tracks train

**Overfitting:**
```
Loss
  4 |*
    | *                  Train
  3 |  **  *****----
    |
  2 |       Val -------↗
    |
  1 |___________________
    0   100   200   300   400   500
```
- Train continues down
- Val starts increasing
- **Fix:** Stop early, add data

**Underfitting:**
```
Loss
  4 |*  **  **  **  **
    |
  3 |
    |
  2 |
    |
  1 |___________________
    0   100   200   300   400   500
```
- Loss barely decreases
- Still high at end
- **Fix:** Increase LR, train longer

---

## Experiment Workflow Template

**Phase 1: Baseline (1 run)**
```
Goal: Get something working
- Use default config
- Verify training completes
- Check sample outputs
```

**Phase 2: Hyperparameter Search (3-5 runs)**
```
Goal: Find optimal settings
- Try 3 learning rates
- Try 2 batch sizes
- Keep other settings constant
```

**Phase 3: Refinement (2-3 runs)**
```
Goal: Polish best config
- Take best from Phase 2
- Try minor variations
- Longer training
```

**Phase 4: Validation (1 run)**
```
Goal: Final confirmation
- Retrain with best config
- Full evaluation
- Document results
```

**Total:** 7-10 experiments to find optimal settings.

---

## Key Takeaways

✅ **File-based tracking works for simple cases**

✅ **WandB scales to many experiments effortlessly**

✅ **Compare runs side-by-side to make informed decisions**

✅ **Use consistent naming and documentation**

✅ **Don't delete failed experiments - learn from them**

✅ **Version control your configs**

---

## Next Steps

You've completed the core Custom Training lessons (CT-1 through CT-6)!

**Optional Advanced Lessons:**

**Lesson CT-7: Model Architecture Basics**

Understand transformer components before training from scratch:
- Tokenization
- Embeddings
- Attention mechanisms
- Feed-forward layers

**Lesson CT-8: Training from Scratch**

Build a tiny model (10-20M parameters) from ground up:
- Design architecture
- Initialize weights
- Train on tiny-shakespeare
- Compare to fine-tuning

**Or apply your knowledge:**

1. Create your own dataset for a specific task
2. Fine-tune for your use case
3. Deploy with vLLM (Lesson 7)
4. Share your results with the community!

---

## Additional Resources

### WandB
- [WandB Quickstart](https://docs.wandb.ai/quickstart) - Official docs
- [WandB Examples](https://github.com/wandb/examples) - Integration examples
- [WandB Sweeps](https://docs.wandb.ai/guides/sweeps) - Hyperparameter optimization

### Experiment Management
- [ML Experiment Tracking Best Practices](https://neptune.ai/blog/ml-experiment-tracking) - Industry guide
- [DVC](https://dvc.org/) - Alternative (local-first) experiment tracking

### Visualization
- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html) - Custom plots
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualizations

---

**Congratulations on completing the core Custom Training series!** 🎉

You now have the tools to fine-tune models, scale to multiple devices, and track experiments professionally.

Continue to **Lesson CT-7: Model Architecture Basics** for deep understanding, or start building your own custom models!
