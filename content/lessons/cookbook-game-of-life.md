---
id: cookbook-game-of-life
title: "Recipe 1: Conway's Game of Life"
description: >-
  Build Conway's Game of Life using TTNN parallel tile computing. Learn convolution operations, cellular automata, and visual output generation. Includes classic patterns: gliders, blinkers, and the famous Gosper Glider Gun!
category: cookbook
tags:
  - ttnn
  - convolution
  - cellular-automata
  - visualization
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300
  - galaxy
status: validated
validatedOn:
  - n150
  - p300
estimatedMinutes: 30
---

# Recipe 1: Conway's Game of Life 🎮

## Overview

Conway's Game of Life is a cellular automaton where cells evolve based on simple rules:
- **Birth**: Dead cell with exactly 3 live neighbors becomes alive
- **Survival**: Live cell with 2-3 live neighbors stays alive
- **Death**: All other cells die or stay dead

**Why This Project:**
- ✅ Simple rules, complex behavior
- ✅ Perfect for parallel tile computing
- ✅ Visual output (matplotlib animation)
- ✅ Teaches convolution operations

**Time:** 30 minutes | **Difficulty:** Beginner

---

## Example Output

[![Game of Life Animation](/assets/img/game_of_life_preview.png)](https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/assets/img/game_of_life.gif)

*Classic "Gosper Glider Gun" pattern generating infinite gliders on TT hardware. Simple convolution rules create complex emergent behavior.*

[View full animation →](https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/assets/img/game_of_life.gif)

---

## Deploy the Project

[📦 Deploy All Cookbook Projects](command:tenstorrent.createCookbookProjects)

This creates the project in `~/tt-scratchpad/cookbook/game_of_life/`.

---

## Project Structure

```text
~/tt-scratchpad/cookbook/game_of_life/
├── game_of_life.py       # Core TTNN implementation
├── visualizer.py          # Matplotlib animation
├── patterns.py            # Glider, blinker, Gosper gun, etc.
├── requirements.txt
└── README.md
```

---

## Implementation

### Step 1: Core Game Logic (`game_of_life.py`)

```python
"""
Conway's Game of Life using TTNN
Implements parallel computation across tiles for efficient execution.
"""

import ttnn
import torch
import numpy as np

class GameOfLife:
    def __init__(self, device, grid_size=(128, 128)):
        """
        Initialize Game of Life on TT hardware.

        Args:
            device: TTNN device handle
            grid_size: (height, width) - must be multiples of 32 for optimal performance
        """
        self.device = device
        self.grid_size = grid_size

        # Create neighbor counting kernel (3×3 convolution kernel)
        # Pattern:
        # [1, 1, 1]
        # [1, 0, 1]  (center is 0 because we count neighbors, not self)
        # [1, 1, 1]
        kernel = torch.tensor([
            [[1.0, 1.0, 1.0],
             [1.0, 0.0, 1.0],
             [1.0, 1.0, 1.0]]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)

        self.neighbor_kernel = ttnn.from_torch(
            kernel,
            device=device,
            layout=ttnn.TILE_LAYOUT
        )

    def initialize_random(self, density=0.3):
        """
        Create random initial grid.

        Args:
            density: Probability of cell being alive (0.0-1.0)

        Returns:
            TTNN tensor on device with random configuration
        """
        random_grid = (torch.rand(self.grid_size) < density).float()
        return ttnn.from_torch(
            random_grid.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            device=self.device,
            layout=ttnn.TILE_LAYOUT
        )

    def initialize_pattern(self, pattern_name):
        """
        Initialize with a known pattern (glider, blinker, etc.)

        Args:
            pattern_name: Name of pattern ('glider', 'blinker', 'gosper_gun')

        Returns:
            TTNN tensor with pattern centered in grid
        """
        from patterns import get_pattern

        grid = torch.zeros(self.grid_size, dtype=torch.float32)
        pattern = get_pattern(pattern_name)

        # Center the pattern
        h, w = self.grid_size
        ph, pw = pattern.shape
        start_h = (h - ph) // 2
        start_w = (w - pw) // 2

        grid[start_h:start_h+ph, start_w:start_w+pw] = torch.tensor(pattern, dtype=torch.float32)

        return ttnn.from_torch(
            grid.unsqueeze(0).unsqueeze(0),
            device=self.device,
            layout=ttnn.TILE_LAYOUT
        )

    def step(self, grid):
        """
        Compute one generation of the Game of Life.

        Uses convolution to count neighbors efficiently:
        - Each cell's 8 neighbors are summed via 2D convolution
        - Game of Life rules applied: birth on 3, survival on 2-3

        Args:
            grid: Current state (TTNN tensor)

        Returns:
            Next state (TTNN tensor)
        """
        # Count neighbors using convolution
        # This is much faster than checking each neighbor individually!
        neighbors = ttnn.conv2d(
            grid,
            self.neighbor_kernel,
            padding=(1, 1),  # Pad edges to handle boundary
            stride=(1, 1)
        )

        # Conway's Rules:
        # Birth: exactly 3 neighbors
        birth = ttnn.logical_and(
            ttnn.eq(neighbors, 3.0),
            ttnn.eq(grid, 0.0)
        )

        # Survival: 2 or 3 neighbors and currently alive
        survival_condition = ttnn.logical_or(
            ttnn.eq(neighbors, 2.0),
            ttnn.eq(neighbors, 3.0)
        )
        survival = ttnn.logical_and(survival_condition, ttnn.eq(grid, 1.0))

        # New state: birth OR survival
        next_grid = ttnn.logical_or(birth, survival)

        # Convert bool back to float
        return ttnn.to_float(next_grid)

    def simulate(self, initial_grid, num_generations=100):
        """
        Run simulation for multiple generations.

        Args:
            initial_grid: Starting configuration
            num_generations: Number of steps to simulate

        Returns:
            List of grids (as numpy arrays) for visualization
        """
        history = []
        grid = initial_grid

        for gen in range(num_generations):
            # Store current state (convert to numpy for visualization)
            grid_np = ttnn.to_torch(grid).squeeze().cpu().numpy()
            history.append(grid_np)

            # Compute next generation
            grid = self.step(grid)

            # Optional: Check for stability
            if gen > 0 and np.array_equal(history[-1], grid_np):
                print(f"Stable state reached at generation {gen}")
                break

        return history

# Example usage
if __name__ == "__main__":
    import ttnn

    # Initialize device
    device = ttnn.open_device(device_id=0)

    # Create game
    game = GameOfLife(device, grid_size=(256, 256))

    # Initialize with random configuration
    initial = game.initialize_random(density=0.3)

    # Or initialize with a pattern:
    # initial = game.initialize_pattern('glider')

    # Run simulation
    history = game.simulate(initial, num_generations=200)

    # Visualize (see visualizer.py)
    from visualizer import animate_game_of_life
    animate_game_of_life(history, interval=50)

    # Cleanup
    ttnn.close_device(device)
```

---

### Step 2: Patterns Library (`patterns.py`)

```python
"""
Classic Game of Life patterns
"""

import numpy as np

PATTERNS = {
    'glider': np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]),

    'blinker': np.array([
        [1, 1, 1]
    ]),

    'toad': np.array([
        [0, 1, 1, 1],
        [1, 1, 1, 0]
    ]),

    'beacon': np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]),

    'pulsar': np.array([
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0]
    ]),

    'glider_gun': np.array([
        # Gosper Glider Gun (36×9) - generates gliders indefinitely!
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ])
}

def get_pattern(name):
    """Get pattern by name."""
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern: {name}. Available: {list(PATTERNS.keys())}")
    return PATTERNS[name]

def list_patterns():
    """List all available patterns."""
    return list(PATTERNS.keys())
```

---

### Step 3: Visualization (`visualizer.py`)

```python
"""
Visualization for Game of Life using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_game_of_life(history, interval=100, save_path=None):
    """
    Animate Game of Life simulation.

    Args:
        history: List of numpy arrays (one per generation)
        interval: Milliseconds between frames
        save_path: Optional path to save as GIF
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Conway's Game of Life on TT Hardware")
    ax.axis('off')

    # Initial frame
    im = ax.imshow(history[0], cmap='binary', interpolation='nearest')
    generation_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             va='top', ha='left', fontsize=12,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def update(frame):
        """Update function for animation."""
        im.set_data(history[frame])
        generation_text.set_text(f'Generation: {frame}')
        return [im, generation_text]

    anim = FuncAnimation(fig, update, frames=len(history),
                        interval=interval, blit=True, repeat=True)

    if save_path:
        writer = PillowWriter(fps=1000//interval)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")

    plt.tight_layout()
    plt.show()

    return anim

def plot_generation(grid, generation_num=0, title=None):
    """
    Plot a single generation.

    Args:
        grid: 2D numpy array
        generation_num: Generation number for title
        title: Custom title (overrides generation_num)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Generation {generation_num}")

    ax.imshow(grid, cmap='binary', interpolation='nearest')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def compare_patterns(patterns_dict):
    """
    Display multiple patterns side-by-side.

    Args:
        patterns_dict: {name: grid} dictionary
    """
    n = len(patterns_dict)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]

    for ax, (name, grid) in zip(axes, patterns_dict.items()):
        ax.set_title(name)
        ax.imshow(grid, cmap='binary', interpolation='nearest')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```

---

## Running the Project

**Quick Start - Click to Run:**

[🎮 Run with Random Pattern](command:tenstorrent.runGameOfLife)

[⬆️ Run Glider Pattern](command:tenstorrent.runGameOfLifeGlider)

[♾️ Run Glider Gun (Infinite)](command:tenstorrent.runGameOfLifeGliderGun)

**Manual Commands:**

```bash
cd ~/tt-scratchpad/cookbook/game_of_life

# Install dependencies
pip install -r requirements.txt

# Run with random initial state
python game_of_life.py
```

**Run with specific pattern:**

```bash
python -c "
from game_of_life import GameOfLife
from visualizer import animate_game_of_life
import ttnn

device = ttnn.open_device(device_id=0)
game = GameOfLife(device, grid_size=(256, 256))

# Try different patterns:
# 'glider', 'blinker', 'toad', 'beacon', 'pulsar', 'glider_gun'
initial = game.initialize_pattern('glider_gun')

history = game.simulate(initial, num_generations=500)
animate_game_of_life(history, interval=50)

ttnn.close_device(device)
"
```

---

## Extensions & Experiments

### 1. Performance Benchmarking
Test different grid sizes and measure performance:

```python
import time

sizes = [128, 256, 512, 1024, 2048]
for size in sizes:
    game = GameOfLife(device, grid_size=(size, size))
    initial = game.initialize_random(0.3)

    start = time.time()
    game.simulate(initial, num_generations=100)
    elapsed = time.time() - start

    generations_per_sec = 100 / elapsed
    print(f"{size}×{size}: {generations_per_sec:.2f} gen/sec")
```

### 2. Custom Rule Sets
Implement variants like **HighLife** (birth on 3,6) or **Day & Night**:

```python
def highlife_step(self, grid):
    """HighLife: B36/S23 (birth on 3 or 6, survival on 2 or 3)"""
    neighbors = ttnn.conv2d(grid, self.neighbor_kernel, padding=(1,1))

    birth = ttnn.logical_and(
        ttnn.logical_or(ttnn.eq(neighbors, 3.0), ttnn.eq(neighbors, 6.0)),
        ttnn.eq(grid, 0.0)
    )

    survival = ttnn.logical_and(
        ttnn.logical_or(ttnn.eq(neighbors, 2.0), ttnn.eq(neighbors, 3.0)),
        ttnn.eq(grid, 1.0)
    )

    return ttnn.to_float(ttnn.logical_or(birth, survival))
```

### 3. Multi-Color Variants
Track cell "age" or "species":

```python
def step_with_age(self, grid):
    """Cells have age (color) that increases each generation."""
    next_grid = self.step(grid)

    # Increment age of surviving cells
    aged = ttnn.add(grid, 1.0)
    aged = ttnn.where(ttnn.eq(next_grid, 1.0), aged, 0.0)

    return aged
```

### 4. 3D Game of Life
Extend to 3D volumes (more complex rules):

```python
# 3D neighbor kernel (3×3×3)
kernel_3d = torch.ones((1, 1, 3, 3, 3))
kernel_3d[0, 0, 1, 1, 1] = 0  # Center cell

# Use 3D convolution
neighbors = ttnn.conv3d(grid_3d, kernel_3d, padding=(1,1,1))
```

---

## What You Learned

- ✅ **Cellular automata**: Simple rules → complex emergent behavior
- ✅ **Convolution operations**: Efficient neighbor counting using 2D convolution
- ✅ **Parallel tile computing**: All cells update simultaneously on TT hardware
- ✅ **Visual output generation**: Creating animations from simulation data

**Next Recipe:** Ready for audio signal processing? Try [Recipe 2: Audio Signal Processing](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22cookbook-audio-processor%22%7D)

**Or:** [Return to Cookbook Overview](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22cookbook-overview%22%7D)
