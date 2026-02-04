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
status: validated
validatedOn:
  - n150
estimatedMinutes: 15
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

The `train.py` script automatically logs:

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
# configs/training_n150_wandb.yaml
training_config:
  # ... other settings ...

  use_wandb: true                  # Enable WandB
  wandb_project: "my-training-project"
  wandb_run_name: "n150-baseline"
```

### Running with WandB

```bash
cd ~/tt-scratchpad/training
python train.py --config configs/training_n150_wandb.yaml
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
wandb: 🚀 View run at https://wandb.ai/your-username/my-training-project/runs/abc123
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
python train.py \
  --config configs/training_n150_lr_5e5.yaml

# Experiment 2: LR = 1e-4
python train.py \
  --config configs/training_n150_lr_1e4.yaml

# Experiment 3: LR = 2e-4
python train.py \
  --config configs/training_n150_lr_2e4.yaml
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
program: train.py
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
wandb agent your-username/my-training-project/sweep-id
```

WandB automatically runs all combinations!

### 3. Model Artifacts

Save checkpoints to WandB:

```python
import wandb

# After saving checkpoint
artifact = wandb.Artifact('trained-model', type='model')
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
    project="my-training-project",
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
git add configs/training_n150_lr1e4.yaml
git commit -m "Add config for LR=1e-4 experiment"
git tag exp-lr1e4
```

**Why:** Reproducibility - know exactly what config produced results.

### 3. Document Results

Create `experiments.md`:

```markdown
# Custom Training Experiments

## Experiment 1: Baseline (2026-02-01)
- **Config:** training_n150.yaml
- **Hardware:** N150
- **Duration:** 2.3 hours
- **Final Loss:** 1.84 (train), 2.12 (val)
- **Result:** Good baseline, will try higher LR next
- **WandB:** [link](https://wandb.ai/...)

## Experiment 2: Higher LR (2026-02-01)
- **Config:** training_n150_lr2e4.yaml
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

## Beyond This Lesson: From Ad-Hoc to Professional ML Engineering

You've learned the tools for tracking experiments. But what transforms scattered training runs into systematic ML engineering? Let's explore how experiment tracking becomes the foundation for data-driven model development.

### Real-World ML Engineering Stories

**What professional teams have built with systematic tracking:**

🚀 **"API Documentation Bot" (Solo dev → Team product)**
- **Week 1 (File-based):** 5 training runs, notes in text files, hard to compare
- **Week 2 (WandB):** Ran 50+ experiments, found LR=2e-4 outperformed 1e-4 by 15%
- **Month 2:** Tracked 200+ runs across 3 developers, identified dataset quality > hyperparameters
- **Impact:** From "guessing what works" → "knowing what works based on data"
- **Result:** Model accuracy improved 40% through systematic experimentation

💼 **"Medical Report Classifier" (Research → Clinical deployment)**
- **Without tracking:** 20 experiments, relied on memory, couldn't reproduce best results
- **With tracking:** Compared 100+ configurations, discovered batch_size=8 + dropout=0.3 optimal
- **Production insight:** WandB comparison revealed validation loss plateau at epoch 12 (not 20!)
- **Impact:** Reduced training time 40%, improved consistency across hospital deployments
- **Cost savings:** $15k/month in compute by training smarter, not longer

🎮 **"Game NPC Dialogue" (Indie studio, 2-person team)**
- **Manual tracking phase:** Lost best model checkpoint, had to retrain for 8 hours
- **WandB phase:** Automatic checkpoint linking, recovered any model in 30 seconds
- **Team benefit:** Designer could view sample outputs without bothering ML engineer
- **Impact:** Iteration speed 5x faster (1 experiment/day → 5 experiments/day)
- **Shipping:** Reduced model development cycle from 6 weeks to 10 days

🏥 **"Radiology Assistant" (Startup → FDA submission)**
- **Challenge:** FDA requires complete training history documentation
- **Solution:** WandB experiment logs provided full audit trail
- **Evidence:** Showed 500+ experiments, systematic hyperparameter search, validation strategy
- **Impact:** FDA approval accelerated by 3 months due to documentation quality
- **Lesson:** Good tracking isn't just productivity - it's regulatory compliance

### The Data-Driven ML Development Cycle

**How tracking transforms your workflow:**

```mermaid
graph TD
    A[Traditional Approach<br/>No Tracking] --> B[Try config]
    B --> C[Wait for training]
    C --> D[Check result]
    D --> E[Forget what you tried]
    E --> F[Try something random]
    F --> B

    G[Data-Driven Approach<br/>With Tracking] --> H[Form hypothesis]
    H --> I[Design experiments]
    I --> J[Run multiple configs]
    J --> K[Compare results in WandB]
    K --> L[Identify patterns]
    L --> M[Make informed decision]
    M --> N[Document findings]
    N --> H

    style A fill:#E85D75,stroke:#333,stroke-width:2px
    style G fill:#50C878,stroke:#333,stroke-width:2px
    style E fill:#FF6B6B,stroke:#333,stroke-width:2px
    style L fill:#7B68EE,stroke:#333,stroke-width:3px
```

**The difference:**
- **Without tracking:** Circular, random, driven by intuition
- **With tracking:** Iterative, systematic, driven by evidence

### What Systematic Tracking Reveals

**Insights you miss without tracking:**

🔍 **"The batch size doesn't matter... until it does"**
- Experiment series: batch_size 4, 8, 16, 32, 64
- Discovery: Performance identical for 4-16, then drops at 32+
- Learning: Larger batches need LR adjustment (not obvious without comparison!)
- Saved: Weeks of chasing the wrong optimization

📊 **"Our validation set was too easy"**
- Tracked 50 experiments, all showed val_loss < train_loss
- WandB comparison revealed: Model memorizing validation patterns
- Fix: Rotated validation set every 10 experiments
- Result: Found models that actually generalize

⏱️ **"We were overtraining by 300%"**
- Loss curves showed: Model converges at step 5000, not 15000
- Impact: Reduced training from 3 hours → 1 hour with same quality
- Savings: 10× more experiments in same time budget
- ROI: Found better config in 1 week that took 2 months without tracking

💡 **"Dataset quality beats hyperparameter tuning"**
- 100 experiments varying LR, batch size, optimizer
- Best improvement: 3% accuracy gain
- Then: Cleaned dataset (removed duplicates, fixed labels)
- Result: 25% accuracy gain with baseline hyperparameters
- Lesson: Track everything, find the real bottleneck

### From Experiments to Insights: The Tracking Hierarchy

**Level 1: Survival (File logs)**
- Can reproduce results if you remember the config
- Time to compare runs: 30+ minutes (manual)
- **Suitable for:** Solo prototyping, 1-5 experiments total

**Level 2: Efficiency (Organized files)**
- Consistent naming, structured directories
- Time to compare runs: 10 minutes (scripted)
- **Suitable for:** Small projects, 10-20 experiments

**Level 3: Professional (WandB basic)**
- Automatic logging, web dashboard, side-by-side comparison
- Time to compare runs: 30 seconds (click and view)
- **Suitable for:** Serious projects, 50+ experiments

**Level 4: Team-Scale (WandB advanced)**
- Shared projects, sweeps, artifact versioning, team collaboration
- Time to compare runs: Instant (live updates)
- **Suitable for:** Production systems, 200+ experiments, multiple developers

**Level 5: Production (WandB + CI/CD)**
- Automated experiment triggers, model registry, deployment tracking
- Time to compare runs: Always available (historical data)
- **Suitable for:** ML platforms, thousands of experiments, continuous improvement

### The Compound Interest of Good Tracking

**Month 1:**
- 20 experiments tracked
- Found best LR (1e-4 vs 2e-4 comparison)
- Saved 2 hours by not re-running duplicates
- **Value:** Nice to have

**Month 3:**
- 150 experiments tracked
- Identified optimal batch size, warmup schedule, regularization
- Shared configs with teammates (reproducibility)
- Automated hyperparameter sweeps
- **Value:** Significant productivity boost

**Month 6:**
- 500+ experiments tracked
- Comprehensive understanding of model behavior across conditions
- Documentation for production deployment
- Historical data prevents repeating mistakes
- New team members onboard 5× faster (review past experiments)
- **Value:** Institutional knowledge, competitive advantage

**Year 1:**
- Thousands of experiments
- Model performance improved 60% through systematic iteration
- Can answer "Why did we make this decision?" for any model version
- Identified patterns that led to novel architecture improvements
- **Value:** Research insights, publishable findings, product differentiation

### Your Experiment Tracking Journey

**This week (File-based tracking):**
- Organize experiments with consistent naming
- Create `experiments.md` to document findings
- Compare 3-5 runs manually
- **Goal:** Build discipline, understand what to track

**Next week (WandB setup):**
- Create free WandB account
- Integrate logging into training script
- Run 10 experiments, compare in dashboard
- **Goal:** Experience speed of visual comparison

**Month 2 (Professional workflow):**
- Use WandB for all experiments
- Document hypotheses before running
- Share dashboards with collaborators
- **Goal:** Data-driven decision making

**Production (Systematic ML engineering):**
- Experiment tracking integrated into workflow
- Hyperparameter sweeps automated
- Model registry for deployment
- **Goal:** Continuous improvement based on evidence

### The Questions Tracking Answers

**Without tracking, you wonder:**
- "Which config gave the best validation loss?"
- "Did I already try learning_rate=2e-4?"
- "What was different between Tuesday's model and today's?"
- "Why is this model worse than last week's?"

**With tracking, you know:**
- ✅ Best config: Run #47 (2026-02-03, LR=1.5e-4, batch=16, dropout=0.2)
- ✅ Already tried: Yes, runs #12, #23, #34 (all converged to val_loss ~1.8)
- ✅ Difference: Tuesday used warmup_steps=500, today uses 1000 (10% better!)
- ✅ Regression: New model uses dataset v2 which has labeling errors

**The transformation:**
- From "I think this works better" → "This works 15% better based on 20 experiments"
- From "Let me try random things" → "Let me test this hypothesis systematically"
- From "I lost the best model" → "Downloaded checkpoint from WandB in 30 seconds"

### Imagine: Your ML Engineering Future

**You now understand:**
- ✅ File-based tracking for quick projects
- ✅ WandB for professional workflows
- ✅ Hyperparameter comparison strategies
- ✅ Experiment organization best practices

**What will you build with systematic tracking?**

- 📈 **Data-driven models** - Every decision backed by evidence
- 🔬 **Reproducible research** - Anyone can verify your results
- 👥 **Team collaboration** - Share insights across developers
- 🚀 **Faster iteration** - Compare 50 runs in seconds, not hours
- 📚 **Institutional knowledge** - Never lose insights from past experiments

**The question isn't "Should I track experiments?"**

**The question is "How many breakthroughs am I missing without tracking?"**

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
