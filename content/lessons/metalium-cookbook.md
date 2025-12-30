---
id: metalium-cookbook
title: TT-Metalium Cookbook
description: >-
  Build creative projects with TT-Metalium! Work through 4 complete cookbook
  recipes: Conway's Game of Life, Audio Signal Processing, Mandelbrot Fractals,
  and Image Filters. Deploy all templates with one command and start coding!
category: advanced
tags:
  - image
  - coding
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

# TT-Metalium Cookbook: Build Creative Projects

Welcome to the TT-Metalium Cookbook! In this lesson, you'll build **four complete projects** from scratch, each teaching different aspects of programming Tenstorrent hardware.

## What You'll Build

1. **🎮 Conway's Game of Life** - Cellular automata with parallel tile computing
2. **🎵 Audio Processor** - Real-time mel-spectrogram and audio effects
3. **🌀 Mandelbrot Explorer** - GPU-style fractal rendering with zoom/pan
4. **🖼️ Custom Image Filters** - Creative visual effects and artistic filters

Each project is a **complete, working application** with full source code, visual output, and extension ideas.

**🎯 Building Blocks for Real Models:**
These projects teach fundamental techniques used in production models:
- **Game of Life** → Convolution (YOLO v10-v12, SegFormer)
- **Audio Processor** → Spectrograms (Whisper speech recognition)
- **Mandelbrot** → Parallel pixel processing (Stable Diffusion 3.5)
- **Image Filters** → 2D convolutions (ResNet50, MobileNetV2, ViT)

---

## Quick Start

[📦 Deploy All Cookbook Projects](command:tenstorrent.createCookbookProjects)

This creates all 4 projects in `~/tt-scratchpad/cookbook/` with one command. You can then follow along with each recipe below, or explore the projects directly!

---

## Project Structure

All projects are deployed to `~/tt-scratchpad/cookbook/`:

```text
~/tt-scratchpad/cookbook/
├── game_of_life/
│   ├── game_of_life.py       # Core TTNN implementation
│   ├── visualizer.py          # Matplotlib animation
│   ├── patterns.py            # Glider, blinker, Gosper gun, etc.
│   ├── requirements.txt
│   └── README.md
├── audio_processor/
│   ├── processor.py           # TTNN audio operations (starter template)
│   ├── requirements.txt
│   └── README.md
├── mandelbrot/
│   ├── renderer.py            # Fractal generation (starter template)
│   ├── requirements.txt
│   └── README.md
└── image_filters/
    ├── filters.py             # Image processing kernels (starter template)
    ├── requirements.txt
    └── README.md
```

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

**Time:** 30 minutes
**Difficulty:** Beginner

---

## Implementation

After deploying the cookbook projects (using the button at the top), navigate to `~/tt-scratchpad/cookbook/game_of_life/` to explore the complete implementation.

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

# Run with specific pattern
python -c "
from game_of_life import GameOfLife
from visualizer import animate_game_of_life
import ttnn

device = ttnn.open_device(0)
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

# Recipe 2: Audio Processor & Visualizer 🎵

## Overview

Build a real-time audio processing pipeline using TTNN for signal processing operations. This project demonstrates practical DSP (Digital Signal Processing) on TT hardware.

**Features:**
- Load and process audio files (WAV, MP3)
- Compute mel-spectrograms on TT hardware
- Real-time visualization
- Audio effects (reverb, pitch shift, time stretch)
- Extensible to voice activity detection, beat detection, and more

**Why This Project:**
- ✅ Real-world application (music, podcasts, voice)
- ✅ Teaches FFT, convolution, filterbanks
- ✅ Foundation for audio ML models (Whisper, speech recognition)
- ✅ Creative and fun!

**Time:** 45 minutes
**Difficulty:** Intermediate

---

## Create the Project

[🎵 Create Audio Processor Project](command:tenstorrent.createAudioProcessor)

This creates the full project in `~/tt-metal-projects/audio_processor/`.

---

## Implementation

### Step 1: Core Audio Processor (`processor.py`)

```python
"""
Audio signal processing using TTNN
Implements mel-spectrogram, MFCC, and real-time effects
"""

import ttnn
import torch
import numpy as np
import librosa
from scipy import signal

class AudioProcessor:
    def __init__(self, device, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128):
        """
        Initialize audio processor on TT hardware.

        Args:
            device: TTNN device handle
            sample_rate: Audio sample rate (Hz)
            n_fft: FFT window size (must be power of 2)
            hop_length: Number of samples between successive frames
            n_mels: Number of mel frequency bins
        """
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Pre-compute mel filterbank on device
        self.mel_filterbank = self._create_mel_filterbank()

        # Pre-compute window function (Hann window)
        self.window = self._create_window()

    def _create_mel_filterbank(self):
        """
        Create mel-scale filterbank matrix.
        Converts linear frequency bins to perceptual mel scale.
        """
        # Use librosa to generate mel filterbank
        mel_fb = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sample_rate // 2
        )

        # Move to device
        return ttnn.from_torch(
            torch.from_numpy(mel_fb).float(),
            device=self.device,
            layout=ttnn.TILE_LAYOUT
        )

    def _create_window(self):
        """Create Hann window for STFT."""
        window = torch.hann_window(self.n_fft, periodic=True)
        return ttnn.from_torch(
            window,
            device=self.device,
            layout=ttnn.TILE_LAYOUT
        )

    def load_audio(self, file_path, duration=None, offset=0.0):
        """
        Load audio file.

        Args:
            file_path: Path to audio file (WAV, MP3, etc.)
            duration: Optional duration to load (seconds)
            offset: Start time (seconds)

        Returns:
            Torch tensor of audio samples
        """
        audio, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=duration,
            offset=offset
        )

        # Convert mono to tensor
        return torch.from_numpy(audio).float()

    def compute_stft(self, audio):
        """
        Compute Short-Time Fourier Transform.

        Args:
            audio: 1D audio tensor

        Returns:
            Complex STFT tensor (freq_bins, time_frames)
        """
        # Convert to TTNN
        audio_tt = ttnn.from_torch(audio, device=self.device)

        # Compute STFT using TTNN FFT
        # Note: STFT = overlapping windows + FFT for each window
        num_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        stft_result = []

        for frame_idx in range(num_frames):
            # Extract frame
            start = frame_idx * self.hop_length
            frame = audio[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                # Pad last frame
                frame = torch.nn.functional.pad(frame, (0, self.n_fft - len(frame)))

            # Move to device and apply window
            frame_tt = ttnn.from_torch(frame, device=self.device)
            windowed = ttnn.multiply(frame_tt, self.window)

            # Compute FFT
            fft_result = ttnn.fft.rfft(windowed)
            stft_result.append(fft_result)

        # Stack frames
        stft = ttnn.stack(stft_result, dim=-1)
        return stft

    def compute_mel_spectrogram(self, audio):
        """
        Compute mel-spectrogram from audio.

        Pipeline:
        1. STFT (time domain → frequency domain)
        2. Power spectrum (magnitude squared)
        3. Mel filterbank (linear freq → mel scale)
        4. Log scale (perceptual compression)

        Args:
            audio: 1D audio tensor or file path

        Returns:
            Mel-spectrogram (n_mels, time_frames) on CPU
        """
        # Load if file path given
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        # Compute STFT
        stft = self.compute_stft(audio)

        # Power spectrum: |STFT|^2
        power_spec = ttnn.square(ttnn.abs(stft))

        # Apply mel filterbank
        mel_spec = ttnn.matmul(self.mel_filterbank, power_spec)

        # Convert to log scale (dB)
        # Add small epsilon to avoid log(0)
        log_mel = ttnn.log(ttnn.add(mel_spec, 1e-10))

        # Scale to decibels
        log_mel = ttnn.multiply(log_mel, 10.0)  # 10 * log10(x) ≈ 4.34 * ln(x)

        # Convert to CPU for visualization/analysis
        return ttnn.to_torch(log_mel).cpu().numpy()

    def compute_mfcc(self, audio, n_mfcc=13):
        """
        Compute Mel-Frequency Cepstral Coefficients.
        MFCCs are commonly used for speech recognition.

        Args:
            audio: 1D audio tensor or file path
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC features (n_mfcc, time_frames)
        """
        # Get mel-spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)

        # Apply DCT (Discrete Cosine Transform)
        # DCT decorrelates mel-frequency components
        mfcc = librosa.feature.mfcc(
            S=mel_spec,
            n_mfcc=n_mfcc
        )

        return mfcc

    def detect_beats(self, audio):
        """
        Detect beats/onsets in audio.
        Uses spectral flux and peak picking.

        Args:
            audio: 1D audio tensor or file path

        Returns:
            Array of beat times (seconds)
        """
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        # Compute onset strength envelope
        mel_spec = self.compute_mel_spectrogram(audio.numpy())
        onset_env = librosa.onset.onset_strength(
            S=mel_spec,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        # Detect peaks (beats)
        peaks = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            units='time'
        )

        return peaks

    def extract_pitch(self, audio):
        """
        Extract fundamental frequency (pitch) over time.
        Uses autocorrelation method (YIN algorithm).

        Args:
            audio: 1D audio tensor or file path

        Returns:
            (times, frequencies) arrays
        """
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        # Use librosa's pitch tracking
        pitches, magnitudes = librosa.core.piptrack(
            y=audio.numpy(),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        # Extract pitch with highest magnitude
        pitch_track = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_track.append(pitch)

        times = librosa.frames_to_time(
            np.arange(len(pitch_track)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        return times, np.array(pitch_track)

# Example usage
if __name__ == "__main__":
    import ttnn
    from visualizer import SpectrogramVisualizer

    # Initialize device
    device = ttnn.open_device(device_id=0)

    # Create processor
    processor = AudioProcessor(device, sample_rate=22050)

    # Load audio file
    audio_file = "examples/sample.wav"  # Use your own file
    audio = processor.load_audio(audio_file, duration=10.0)

    # Compute mel-spectrogram
    mel_spec = processor.compute_mel_spectrogram(audio)

    # Visualize
    viz = SpectrogramVisualizer(processor)
    viz.plot_spectrogram(mel_spec, title="Mel-Spectrogram")

    # Detect beats
    beats = processor.detect_beats(audio)
    print(f"Detected {len(beats)} beats at times: {beats}")

    # Extract pitch
    times, pitches = processor.extract_pitch(audio)
    viz.plot_pitch(times, pitches)

    # Cleanup
    ttnn.close_device(device)
```

---

### Step 2: Audio Effects (`effects.py`)

```python
"""
Real-time audio effects using TTNN
"""

import ttnn
import torch
import numpy as np
from scipy import signal

class AudioEffects:
    def __init__(self, processor):
        """
        Initialize audio effects processor.

        Args:
            processor: AudioProcessor instance
        """
        self.processor = processor
        self.device = processor.device
        self.sample_rate = processor.sample_rate

    def reverb(self, audio, room_size=0.5, damping=0.5, wet=0.3):
        """
        Add reverb effect using convolution with impulse response.

        Args:
            audio: Input audio tensor
            room_size: Room size (0-1, larger = longer reverb tail)
            damping: High-frequency damping (0-1)
            wet: Wet/dry mix (0=dry, 1=wet)

        Returns:
            Audio with reverb applied
        """
        # Generate simple impulse response (exponential decay)
        reverb_time = int(self.sample_rate * room_size * 2)  # Up to 2 seconds
        decay = np.exp(-3 * np.arange(reverb_time) / reverb_time)

        # Apply damping (low-pass filter)
        if damping > 0:
            b, a = signal.butter(2, damping, btype='low', fs=1.0)
            decay = signal.lfilter(b, a, decay)

        # Normalize
        impulse_response = decay / np.max(np.abs(decay))

        # Convert to TTNN
        audio_tt = ttnn.from_torch(audio, device=self.device)
        ir_tt = ttnn.from_torch(
            torch.from_numpy(impulse_response).float(),
            device=self.device
        )

        # Convolve with impulse response
        reverb_audio = ttnn.conv1d(audio_tt.unsqueeze(0).unsqueeze(0),
                                   ir_tt.unsqueeze(0).unsqueeze(0))
        reverb_audio = reverb_audio.squeeze()

        # Mix wet/dry
        audio_tt_padded = ttnn.pad(audio_tt, (0, len(impulse_response) - 1))
        mixed = ttnn.add(
            ttnn.multiply(audio_tt_padded, (1 - wet)),
            ttnn.multiply(reverb_audio, wet)
        )

        return ttnn.to_torch(mixed).cpu()

    def pitch_shift(self, audio, semitones):
        """
        Shift pitch without changing duration (phase vocoder).

        Args:
            audio: Input audio
            semitones: Pitch shift in semitones (+12 = up one octave)

        Returns:
            Pitch-shifted audio
        """
        # Use librosa for phase vocoder
        shifted = librosa.effects.pitch_shift(
            y=audio.numpy(),
            sr=self.sample_rate,
            n_steps=semitones
        )
        return torch.from_numpy(shifted).float()

    def time_stretch(self, audio, rate):
        """
        Change duration without changing pitch.

        Args:
            audio: Input audio
            rate: Stretch factor (0.5 = half speed, 2.0 = double speed)

        Returns:
            Time-stretched audio
        """
        stretched = librosa.effects.time_stretch(
            y=audio.numpy(),
            rate=rate
        )
        return torch.from_numpy(stretched).float()

    def echo(self, audio, delay_ms=500, decay=0.5):
        """
        Add echo effect.

        Args:
            audio: Input audio
            delay_ms: Delay in milliseconds
            decay: Amplitude decay of echo

        Returns:
            Audio with echo
        """
        delay_samples = int(self.sample_rate * delay_ms / 1000)

        # Create delayed copy
        audio_tt = ttnn.from_torch(audio, device=self.device)
        delayed = ttnn.pad(audio_tt, (delay_samples, 0))
        delayed = delayed[:len(audio)]

        # Mix with decay
        echo_audio = ttnn.add(
            audio_tt,
            ttnn.multiply(delayed, decay)
        )

        return ttnn.to_torch(echo_audio).cpu()

    def chorus(self, audio, rate=1.5, depth=0.002):
        """
        Add chorus effect (slightly detuned copies).

        Args:
            audio: Input audio
            rate: LFO rate (Hz)
            depth: Modulation depth (seconds)

        Returns:
            Audio with chorus effect
        """
        # Implement as time-varying delay with LFO
        num_samples = len(audio)
        t = np.arange(num_samples) / self.sample_rate

        # Low-frequency oscillator
        lfo = np.sin(2 * np.pi * rate * t)
        delay_samples = (depth * self.sample_rate * lfo).astype(int)

        # Apply variable delay (simplified version)
        # In production, use interpolation for smooth delay changes
        output = audio.clone()
        for i in range(num_samples):
            delay_idx = max(0, min(num_samples - 1, i + delay_samples[i]))
            output[i] += 0.5 * audio[delay_idx]

        return output
```

---

### Step 3: Visualization (`visualizer.py`)

```python
"""
Real-time audio visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd

class SpectrogramVisualizer:
    def __init__(self, processor):
        """
        Initialize visualizer.

        Args:
            processor: AudioProcessor instance
        """
        self.processor = processor

    def plot_spectrogram(self, mel_spec, title="Mel-Spectrogram", save_path=None):
        """
        Plot mel-spectrogram.

        Args:
            mel_spec: 2D array (n_mels, time_frames)
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        # Convert frames to time
        times = np.arange(mel_spec.shape[1]) * self.processor.hop_length / self.processor.sample_rate

        # Plot
        img = ax.imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            extent=[times.min(), times.max(), 0, self.processor.n_mels],
            cmap='viridis'
        )

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Mel Frequency Bin')
        ax.set_title(title)

        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        plt.show()

    def plot_waveform(self, audio, title="Waveform"):
        """Plot audio waveform."""
        times = np.arange(len(audio)) / self.processor.sample_rate

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(times, audio, linewidth=0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_pitch(self, times, pitches):
        """Plot pitch track."""
        fig, ax = plt.subplots(figsize=(12, 4))

        # Filter out zero pitches (unvoiced)
        voiced = pitches > 0
        ax.plot(times[voiced], pitches[voiced], 'o-', markersize=2)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Pitch Track')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def real_time_spectrogram(self, duration=10, window_size=2.0):
        """
        Real-time spectrogram from microphone.

        Args:
            duration: Total duration (seconds)
            window_size: Spectrogram window size (seconds)
        """
        # Buffer for audio samples
        buffer_size = int(self.processor.sample_rate * window_size)
        audio_buffer = np.zeros(buffer_size)

        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 4))
        spec_img = ax.imshow(
            np.zeros((self.processor.n_mels, 100)),
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=-80,
            vmax=0
        )
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Mel Frequency')
        ax.set_title('Real-Time Spectrogram')
        plt.colorbar(spec_img, ax=ax)

        # Callback for audio stream
        spec_history = []

        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer, spec_history

            # Shift buffer and add new data
            audio_buffer = np.roll(audio_buffer, -frames)
            audio_buffer[-frames:] = indata[:, 0]

            # Compute mel-spectrogram
            audio_torch = torch.from_numpy(audio_buffer).float()
            mel_spec = self.processor.compute_mel_spectrogram(audio_torch)

            # Store
            spec_history.append(mel_spec)
            if len(spec_history) > 100:
                spec_history.pop(0)

            # Update plot
            if len(spec_history) > 0:
                spec_concat = np.concatenate(spec_history, axis=1)
                spec_img.set_data(spec_concat[:, -100:])
                fig.canvas.draw_idle()

        # Start audio stream
        with sd.InputStream(callback=audio_callback,
                           channels=1,
                           samplerate=self.processor.sample_rate):
            print(f"Recording for {duration} seconds...")
            plt.show(block=False)
            plt.pause(duration)

        print("Done!")

    def animate_spectrogram_with_audio(self, audio_file):
        """
        Animate spectrogram synchronized with audio playback.

        Args:
            audio_file: Path to audio file
        """
        # Load audio
        audio = self.processor.load_audio(audio_file)

        # Compute full mel-spectrogram
        mel_spec = self.processor.compute_mel_spectrogram(audio)

        # Setup plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        # Waveform plot
        times = np.arange(len(audio)) / self.processor.sample_rate
        ax1.plot(times, audio, linewidth=0.5, color='blue')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Waveform')
        ax1.grid(True, alpha=0.3)

        # Current time marker
        line = ax1.axvline(x=0, color='red', linewidth=2)

        # Spectrogram plot
        spec_times = np.arange(mel_spec.shape[1]) * self.processor.hop_length / self.processor.sample_rate
        ax2.imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            extent=[spec_times.min(), spec_times.max(), 0, self.processor.n_mels],
            cmap='viridis'
        )
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Mel Frequency')
        ax2.set_title('Mel-Spectrogram')

        # Marker
        spec_line = ax2.axvline(x=0, color='red', linewidth=2)

        plt.tight_layout()

        # Animation
        def update(frame):
            current_time = frame / 30  # 30 FPS
            line.set_xdata([current_time, current_time])
            spec_line.set_xdata([current_time, current_time])
            return [line, spec_line]

        # Play audio in background
        sd.play(audio.numpy(), self.processor.sample_rate)

        # Animate
        num_frames = int(len(audio) / self.processor.sample_rate * 30)
        anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/30, blit=True)

        plt.show()
```

---

## Running the Project

**Quick Start - Click to Run:**

[🎵 Run Audio Processor Demo](command:tenstorrent.runAudioProcessor)

**Manual Commands:**

```bash
cd ~/tt-scratchpad/cookbook/audio_processor

# Install dependencies
pip install -r requirements.txt

# Process an audio file
python processor.py

# Try effects
python -c "
from processor import AudioProcessor
from effects import AudioEffects
import ttnn
import sounddevice as sd

device = ttnn.open_device(0)
processor = AudioProcessor(device)
effects = AudioEffects(processor)

# Load audio
audio = processor.load_audio('examples/sample.wav')

# Apply reverb
reverb_audio = effects.reverb(audio, room_size=0.7, wet=0.5)

# Play original vs reverb
print('Playing original...')
sd.play(audio.numpy(), processor.sample_rate)
sd.wait()

print('Playing with reverb...')
sd.play(reverb_audio.numpy(), processor.sample_rate)
sd.wait()

ttnn.close_device(device)
"

# Real-time spectrogram from microphone
python -c "
from processor import AudioProcessor
from visualizer import SpectrogramVisualizer
import ttnn

device = ttnn.open_device(0)
processor = AudioProcessor(device, sample_rate=22050)
viz = SpectrogramVisualizer(processor)

viz.real_time_spectrogram(duration=10)

ttnn.close_device(device)
"
```

---

## Extensions for Audio Engineers

### 1. Voice Activity Detection (VAD)
Detect speech vs silence:

```python
def voice_activity_detection(self, audio, threshold_db=-40):
    """Detect speech segments using energy thresholding."""
    # Compute short-time energy
    frame_length = self.n_fft
    hop_length = self.hop_length

    energy = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i+frame_length]
        frame_energy = 20 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-10)
        energy.append(frame_energy)

    # Threshold
    is_speech = np.array(energy) > threshold_db

    # Convert to time segments
    times = np.arange(len(energy)) * hop_length / self.sample_rate
    return times, is_speech
```

### 2. Automatic Gain Control (AGC)
Normalize volume dynamically:

```python
def auto_gain_control(self, audio, target_db=-20, attack_ms=50, release_ms=200):
    """Dynamic range compression."""
    # Convert to dB
    audio_db = 20 * torch.log10(torch.abs(audio) + 1e-10)

    # Envelope follower
    attack_coef = np.exp(-1000 / (attack_ms * self.sample_rate))
    release_coef = np.exp(-1000 / (release_ms * self.sample_rate))

    envelope = torch.zeros_like(audio_db)
    for i in range(1, len(audio_db)):
        if audio_db[i] > envelope[i-1]:
            envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * audio_db[i]
        else:
            envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * audio_db[i]

    # Apply gain
    gain_db = target_db - envelope
    gain_linear = 10 ** (gain_db / 20)

    return audio * gain_linear
```

### 3. Noise Gate
Remove background noise:

```python
def noise_gate(self, audio, threshold_db=-50, attack_ms=10, release_ms=100):
    """Suppress audio below threshold."""
    audio_db = 20 * torch.log10(torch.abs(audio) + 1e-10)

    # Gate on/off
    gate_open = audio_db > threshold_db

    # Smooth transitions
    gate_smooth = self._smooth_gate(gate_open, attack_ms, release_ms)

    return audio * gate_smooth
```

### 4. Parametric EQ
Frequency-specific gain:

```python
def parametric_eq(self, audio, center_freq, gain_db, q_factor=1.0):
    """Apply parametric EQ filter."""
    # Design peaking filter
    b, a = signal.iirpeak(
        center_freq,
        Q=q_factor,
        fs=self.sample_rate
    )

    # Apply gain
    b = b * (10 ** (gain_db / 20))

    # Filter audio
    filtered = signal.lfilter(b, a, audio.numpy())
    return torch.from_numpy(filtered).float()
```

### 5. VST Plugin Interface
Integrate with DAWs:

```python
# This would require python-vst or similar library
def process_block(self, audio_block):
    """Process audio block (VST-style callback)."""
    # Convert to tensor
    audio_tt = ttnn.from_torch(audio_block, device=self.device)

    # Apply effects chain
    processed = self.apply_effects_chain(audio_tt)

    # Convert back
    return ttnn.to_torch(processed).cpu().numpy()
```

---

# Recipe 3: Mandelbrot Explorer 🌀

## Overview

The Mandelbrot set is a fractal defined by the iterative equation `z = z² + c`. Points that don't diverge to infinity belong to the set. This project demonstrates GPU-style parallel computation on TT hardware.

**Features:**
- High-resolution fractal rendering
- Interactive zoom and pan
- Color mapping for iteration counts
- Performance profiling

**Why This Project:**
- ✅ Embarrassingly parallel (perfect for tiles)
- ✅ Beautiful visual output
- ✅ Teaches performance optimization
- ✅ Complex number operations

**Time:** 30 minutes
**Difficulty:** Beginner-Intermediate

---

## Create the Project

[🌀 Create Mandelbrot Explorer Project](command:tenstorrent.createMandelbrotExplorer)

---

## Implementation

### Step 1: Core Renderer (`renderer.py`)

```python
"""
Mandelbrot Set renderer using TTNN
Each pixel computed independently - ideal for parallel execution
"""

import ttnn
import torch
import numpy as np
import time

class MandelbrotRenderer:
    def __init__(self, device):
        """
        Initialize Mandelbrot renderer.

        Args:
            device: TTNN device handle
        """
        self.device = device

    def render(self, width, height, x_min, x_max, y_min, y_max, max_iter=256):
        """
        Render Mandelbrot set for given complex plane region.

        Algorithm:
        For each pixel (representing complex number c):
            z = 0
            for i in range(max_iter):
                z = z² + c
                if |z| > 2:
                    return i  (iteration count)
            return max_iter  (in set)

        Args:
            width, height: Image dimensions
            x_min, x_max: Real axis range
            y_min, y_max: Imaginary axis range
            max_iter: Maximum iterations before considering point "in set"

        Returns:
            2D array of iteration counts
        """
        print(f"Rendering {width}×{height} image...")
        print(f"Complex plane: [{x_min}, {x_max}] × [{y_min}, {y_max}]i")
        print(f"Max iterations: {max_iter}")

        start_time = time.time()

        # Create coordinate grids
        x = torch.linspace(x_min, x_max, width, dtype=torch.float32)
        y = torch.linspace(y_min, y_max, height, dtype=torch.float32)

        # Meshgrid for complex plane
        X, Y = torch.meshgrid(x, y, indexing='xy')
        C_real = X.T  # Transpose to match image coordinates
        C_imag = Y.T

        # Move to device
        c_real = ttnn.from_torch(C_real, device=self.device, layout=ttnn.TILE_LAYOUT)
        c_imag = ttnn.from_torch(C_imag, device=self.device, layout=ttnn.TILE_LAYOUT)

        # Initialize z = 0
        z_real = ttnn.zeros_like(c_real)
        z_imag = ttnn.zeros_like(c_imag)

        # Iteration counter (starts at max_iter, decremented when diverged)
        iterations = ttnn.full_like(c_real, max_iter)

        # Iterate z = z² + c
        for i in range(max_iter):
            # Complex multiplication: (a + bi)² = (a² - b²) + (2ab)i
            z_real_sq = ttnn.square(z_real)
            z_imag_sq = ttnn.square(z_imag)

            z_real_new = ttnn.subtract(z_real_sq, z_imag_sq)
            z_real_new = ttnn.add(z_real_new, c_real)

            z_imag_new = ttnn.multiply(z_real, z_imag)
            z_imag_new = ttnn.multiply(z_imag_new, 2.0)
            z_imag_new = ttnn.add(z_imag_new, c_imag)

            # Magnitude: |z| = sqrt(real² + imag²)
            magnitude_sq = ttnn.add(z_real_sq, z_imag_sq)

            # Mark diverged points (|z| > 2, so |z|² > 4)
            diverged = ttnn.gt(magnitude_sq, 4.0)

            # Update iteration count (only for points that just diverged)
            still_iterating = ttnn.eq(iterations, max_iter)
            just_diverged = ttnn.logical_and(diverged, still_iterating)

            # Set iteration count to current iteration
            iterations = ttnn.where(just_diverged, float(i), iterations)

            # Update z for next iteration
            z_real = z_real_new
            z_imag = z_imag_new

            # Early exit if all points diverged
            if i % 10 == 0:  # Check every 10 iterations
                num_still_iterating = ttnn.sum(still_iterating)
                if ttnn.to_torch(num_still_iterating).item() == 0:
                    print(f"All points diverged by iteration {i}")
                    break

        # Convert to numpy
        result = ttnn.to_torch(iterations).cpu().numpy()

        elapsed = time.time() - start_time
        pixels_per_sec = (width * height) / elapsed
        print(f"Rendered in {elapsed:.2f}s ({pixels_per_sec/1e6:.2f} Mpixels/sec)")

        return result

    def render_julia(self, width, height, c_real, c_imag, x_min=-2, x_max=2,
                     y_min=-2, y_max=2, max_iter=256):
        """
        Render Julia set for fixed c (vs Mandelbrot which varies c).

        Julia set: z₀ = pixel coordinate, fixed c, iterate z = z² + c

        Args:
            width, height: Image dimensions
            c_real, c_imag: Fixed complex parameter
            x_min, x_max, y_min, y_max: Coordinate range
            max_iter: Maximum iterations

        Returns:
            2D array of iteration counts
        """
        print(f"Rendering Julia set with c = {c_real} + {c_imag}i")

        # Create initial z (pixel coordinates)
        x = torch.linspace(x_min, x_max, width, dtype=torch.float32)
        y = torch.linspace(y_min, y_max, height, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='xy')

        z_real = ttnn.from_torch(X.T, device=self.device, layout=ttnn.TILE_LAYOUT)
        z_imag = ttnn.from_torch(Y.T, device=self.device, layout=ttnn.TILE_LAYOUT)

        # Fixed c
        c_r = ttnn.full_like(z_real, c_real)
        c_i = ttnn.full_like(z_imag, c_imag)

        # Iterate (same as Mandelbrot)
        iterations = ttnn.full_like(z_real, max_iter)

        for i in range(max_iter):
            z_real_sq = ttnn.square(z_real)
            z_imag_sq = ttnn.square(z_imag)

            z_real_new = ttnn.add(ttnn.subtract(z_real_sq, z_imag_sq), c_r)
            z_imag_new = ttnn.add(ttnn.multiply(ttnn.multiply(z_real, z_imag), 2.0), c_i)

            magnitude_sq = ttnn.add(z_real_sq, z_imag_sq)
            diverged = ttnn.gt(magnitude_sq, 4.0)

            still_iterating = ttnn.eq(iterations, max_iter)
            just_diverged = ttnn.logical_and(diverged, still_iterating)

            iterations = ttnn.where(just_diverged, float(i), iterations)

            z_real = z_real_new
            z_imag = z_imag_new

        return ttnn.to_torch(iterations).cpu().numpy()


# Example usage
if __name__ == "__main__":
    import ttnn
    from explorer import MandelbrotVisualizer

    device = ttnn.open_device(device_id=0)
    renderer = MandelbrotRenderer(device)

    # Classic Mandelbrot view
    mandelbrot = renderer.render(
        width=2048, height=2048,
        x_min=-2.5, x_max=1.0,
        y_min=-1.25, y_max=1.25,
        max_iter=512
    )

    # Visualize
    viz = MandelbrotVisualizer(renderer)
    viz.show(mandelbrot)

    ttnn.close_device(device)
```

---

### Step 2: Interactive Explorer (`explorer.py`)

```python
"""
Interactive Mandelbrot explorer with zoom and pan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MandelbrotVisualizer:
    def __init__(self, renderer):
        """
        Initialize visualizer.

        Args:
            renderer: MandelbrotRenderer instance
        """
        self.renderer = renderer

        # Color maps to try
        self.colormaps = ['hot', 'viridis', 'twilight', 'gist_earth', 'nipy_spectral']
        self.current_cmap = 0

    def show(self, fractal_data, title="Mandelbrot Set"):
        """
        Display fractal with nice color mapping.

        Args:
            fractal_data: 2D array of iteration counts
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Logarithmic color scale for better visualization
        # Points in set (max_iter) shown in black
        max_iter = fractal_data.max()
        color_data = np.log(fractal_data + 1)  # +1 to avoid log(0)

        img = ax.imshow(
            color_data,
            cmap=self.colormaps[self.current_cmap],
            origin='lower',
            extent=[-2.5, 1.0, -1.25, 1.25],  # Default Mandelbrot bounds
            interpolation='bilinear'
        )

        ax.set_title(title)
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')

        plt.colorbar(img, ax=ax, label='log(iterations)')
        plt.tight_layout()
        plt.show()

    def interactive_explorer(self, width=1024, height=1024, initial_max_iter=256):
        """
        Interactive explorer with click-to-zoom.

        Click on plot to zoom into that region.
        Press 'r' to reset view.
        Press 'c' to cycle color maps.
        Press 'q' to quit.
        """
        # Initial view (full Mandelbrot set)
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.25, 1.25
        max_iter = initial_max_iter

        # Zoom history for undo
        view_history = [(x_min, x_max, y_min, y_max, max_iter)]

        # Render initial view
        fractal = self.renderer.render(width, height, x_min, x_max, y_min, y_max, max_iter)

        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 12))
        color_data = np.log(fractal + 1)
        img = ax.imshow(
            color_data,
            cmap=self.colormaps[self.current_cmap],
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            interpolation='bilinear',
            picker=True
        )

        ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')
        plt.colorbar(img, ax=ax, label='log(iterations)')

        instructions = ax.text(
            0.02, 0.98,
            'Click: Zoom | R: Reset | C: Color | Q: Quit | U: Undo',
            transform=ax.transAxes,
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        def on_click(event):
            nonlocal x_min, x_max, y_min, y_max, max_iter, fractal

            if event.inaxes != ax:
                return

            # Get click coordinates
            click_x, click_y = event.xdata, event.ydata

            # Zoom factor
            zoom = 0.25  # Show 25% of current view

            # New bounds centered on click
            x_range = (x_max - x_min) * zoom
            y_range = (y_max - y_min) * zoom

            x_min_new = click_x - x_range / 2
            x_max_new = click_x + x_range / 2
            y_min_new = click_y - y_range / 2
            y_max_new = click_y + y_range / 2

            # Increase iterations for deeper zooms
            max_iter = min(max_iter + 128, 2048)

            # Save to history
            view_history.append((x_min_new, x_max_new, y_min_new, y_max_new, max_iter))

            # Render new view
            print(f"\nZooming to ({click_x:.6f}, {click_y:.6f})...")
            fractal = self.renderer.render(
                width, height,
                x_min_new, x_max_new, y_min_new, y_max_new,
                max_iter
            )

            # Update plot
            x_min, x_max = x_min_new, x_max_new
            y_min, y_max = y_min_new, y_max_new

            color_data = np.log(fractal + 1)
            img.set_data(color_data)
            img.set_extent([x_min, x_max, y_min, y_max])
            ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal x_min, x_max, y_min, y_max, max_iter, fractal

            if event.key == 'r':
                # Reset to initial view
                print("\nResetting view...")
                x_min, x_max = -2.5, 1.0
                y_min, y_max = -1.25, 1.25
                max_iter = initial_max_iter
                view_history.clear()
                view_history.append((x_min, x_max, y_min, y_max, max_iter))

                fractal = self.renderer.render(width, height, x_min, x_max, y_min, y_max, max_iter)
                color_data = np.log(fractal + 1)
                img.set_data(color_data)
                img.set_extent([x_min, x_max, y_min, y_max])
                ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
                fig.canvas.draw_idle()

            elif event.key == 'c':
                # Cycle color map
                self.current_cmap = (self.current_cmap + 1) % len(self.colormaps)
                img.set_cmap(self.colormaps[self.current_cmap])
                print(f"\nColor map: {self.colormaps[self.current_cmap]}")
                fig.canvas.draw_idle()

            elif event.key == 'u':
                # Undo (go back in history)
                if len(view_history) > 1:
                    view_history.pop()  # Remove current
                    x_min, x_max, y_min, y_max, max_iter = view_history[-1]

                    print("\nUndoing zoom...")
                    fractal = self.renderer.render(width, height, x_min, x_max, y_min, y_max, max_iter)
                    color_data = np.log(fractal + 1)
                    img.set_data(color_data)
                    img.set_extent([x_min, x_max, y_min, y_max])
                    ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
                    fig.canvas.draw_idle()

            elif event.key == 'q':
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

    def compare_julia_sets(self, c_values, width=512, height=512):
        """
        Display multiple Julia sets side-by-side.

        Args:
            c_values: List of (real, imag) tuples
            width, height: Resolution per image
        """
        n = len(c_values)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (c_real, c_imag) in zip(axes[:n], c_values):
            julia = self.renderer.render_julia(
                width, height,
                c_real, c_imag,
                max_iter=256
            )

            color_data = np.log(julia + 1)
            ax.imshow(color_data, cmap='twilight', origin='lower')
            ax.set_title(f'c = {c_real:.3f} + {c_imag:.3f}i')
            ax.axis('off')

        # Hide unused subplots
        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
```

---

## Running the Project

**Quick Start - Click to Run:**

[🌀 Launch Interactive Explorer](command:tenstorrent.runMandelbrotExplorer)

[🎨 Compare 6 Julia Sets](command:tenstorrent.runMandelbrotJulia)

**Manual Commands:**

```bash
cd ~/tt-scratchpad/cookbook/mandelbrot

# Basic render
python renderer.py

# Interactive explorer
python -c "
from renderer import MandelbrotRenderer
from explorer import MandelbrotVisualizer
import ttnn

device = ttnn.open_device(0)
renderer = MandelbrotRenderer(device)
viz = MandelbrotVisualizer(renderer)

# Launch interactive explorer
viz.interactive_explorer(width=1024, height=1024)

ttnn.close_device(device)
"

# Compare Julia sets
python -c "
from renderer import MandelbrotRenderer
from explorer import MandelbrotVisualizer
import ttnn

device = ttnn.open_device(0)
renderer = MandelbrotRenderer(device)
viz = MandelbrotVisualizer(renderer)

# Interesting Julia set parameters
c_values = [
    (-0.4, 0.6),      # Dendrite
    (0.285, 0.01),    # Douady rabbit
    (-0.70176, -0.3842),  # San Marco
    (-0.835, -0.2321),    # Siegel disk
    (-0.8, 0.156),    # Quasi-spiral
    (0.0, -0.8),      # Classic
]

viz.compare_julia_sets(c_values)

ttnn.close_device(device)
"
```

---

## Extensions

### 1. Burning Ship Fractal
```python
# In the iteration loop, use abs(z) instead of z:
z_real_new = ttnn.subtract(
    ttnn.square(ttnn.abs(z_real)),
    ttnn.square(ttnn.abs(z_imag))
)
z_real_new = ttnn.add(z_real_new, c_real)
```

### 2. 3D Mandelbulb
Extend to 3D using spherical coordinates.

### 3. Deep Zoom Videos
Record zooms to create mesmerizing videos:

```python
def render_zoom_sequence(self, target_x, target_y, num_frames=100):
    """Render frames for zoom animation."""
    frames = []
    for i in range(num_frames):
        zoom_factor = 0.95 ** i  # Exponential zoom
        # ... render and save frame
    return frames
```

### 4. Performance Profiling
```python
# Compare different resolutions
for size in [512, 1024, 2048, 4096]:
    fractal = renderer.render(size, size, -2.5, 1.0, -1.25, 1.25, 256)
```

---

# Recipe 4: Custom Image Filters 🖼️

## Overview

Build a library of creative image filters and effects using TTNN convolution operations. From classic edge detection to artistic stylization.

**Features:**
- Classic filters (blur, sharpen, edge detect)
- Artistic effects (oil painting, watercolor, emboss)
- Custom kernel design
- Real-time webcam processing

**Why This Project:**
- ✅ Practical computer vision foundation
- ✅ Teaches 2D convolution
- ✅ Creative and visual
- ✅ Extensible to neural style transfer

**Time:** 30 minutes
**Difficulty:** Beginner-Intermediate

---

## Create the Project

[🖼️ Create Image Filters Project](command:tenstorrent.createImageFilters)

---

## Implementation

### `filters.py`

```python
"""
Image filtering and effects using TTNN
"""

import ttnn
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageFilterBank:
    def __init__(self, device):
        """
        Initialize filter bank.

        Args:
            device: TTNN device handle
        """
        self.device = device

        # Predefined kernels
        self.kernels = self._define_kernels()

    def _define_kernels(self):
        """Define common convolution kernels."""
        return {
            # Edge Detection
            'sobel_x': torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32),

            'sobel_y': torch.tensor([[-1, -2, -1],
                                     [0,  0,  0],
                                     [1,  2,  1]], dtype=torch.float32),

            'laplacian': torch.tensor([[0,  1, 0],
                                      [1, -4, 1],
                                      [0,  1, 0]], dtype=torch.float32),

            'edge_detect': torch.tensor([[-1, -1, -1],
                                        [-1,  8, -1],
                                        [-1, -1, -1]], dtype=torch.float32),

            # Blur
            'box_blur': torch.ones((5, 5), dtype=torch.float32) / 25,

            'gaussian_blur': torch.tensor([[1,  4,  6,  4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, 36, 24, 6],
                                          [4, 16, 24, 16, 4],
                                          [1,  4,  6,  4, 1]], dtype=torch.float32) / 256,

            # Sharpen
            'sharpen': torch.tensor([[0, -1,  0],
                                    [-1,  5, -1],
                                    [0, -1,  0]], dtype=torch.float32),

            'unsharp_mask': torch.tensor([[-1, -1, -1],
                                         [-1,  9, -1],
                                         [-1, -1, -1]], dtype=torch.float32),

            # Emboss
            'emboss': torch.tensor([[-2, -1,  0],
                                   [-1,  1,  1],
                                   [0,  1,  2]], dtype=torch.float32),

            # Motion blur
            'motion_blur_h': torch.zeros((5, 5), dtype=torch.float32),
            'motion_blur_v': torch.zeros((5, 5), dtype=torch.float32),
        }

    def load_image(self, path):
        """Load image as tensor."""
        img = Image.open(path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        return torch.from_numpy(img_array).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    def save_image(self, tensor, path):
        """Save tensor as image."""
        # Clamp to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        # (C, H, W) -> (H, W, C)
        img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_array).save(path)
        print(f"Saved to {path}")

    def apply_filter(self, image, filter_name):
        """
        Apply named filter to image.

        Args:
            image: Tensor (C, H, W)
            filter_name: Name from self.kernels

        Returns:
            Filtered image tensor
        """
        if filter_name not in self.kernels:
            raise ValueError(f"Unknown filter: {filter_name}")

        kernel = self.kernels[filter_name]

        # Move to device
        img_tt = ttnn.from_torch(image.unsqueeze(0), device=self.device, layout=ttnn.TILE_LAYOUT)
        kernel_tt = ttnn.from_torch(kernel.unsqueeze(0).unsqueeze(0), device=self.device, layout=ttnn.TILE_LAYOUT)

        # Apply to each channel
        filtered_channels = []
        for c in range(image.shape[0]):
            channel = img_tt[0, c:c+1, :, :]
            filtered = ttnn.conv2d(channel, kernel_tt, padding='same')
            filtered_channels.append(filtered)

        # Stack channels
        result = ttnn.concat(filtered_channels, dim=1)

        return ttnn.to_torch(result).squeeze(0).cpu()

    def edge_detect_combined(self, image):
        """Sobel edge detection (combines X and Y gradients)."""
        sobel_x = self.apply_filter(image, 'sobel_x')
        sobel_y = self.apply_filter(image, 'sobel_y')

        # Magnitude: sqrt(Gx² + Gy²)
        magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)

        return magnitude

    def oil_painting_effect(self, image, radius=5, intensity_levels=20):
        """
        Kuwahara filter approximation for oil painting effect.

        Simplifies colors while preserving edges.
        """
        # Convert to grayscale for intensity
        gray = image.mean(dim=0, keepdim=True)

        # Quantize intensity
        quantized = torch.floor(gray * intensity_levels) / intensity_levels

        # Apply bilateral-style smoothing
        # (Simplified version - full Kuwahara is more complex)
        img_tt = ttnn.from_torch(image.unsqueeze(0), device=self.device)

        # Box blur as approximation
        kernel_size = radius * 2 + 1
        box_kernel = torch.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        kernel_tt = ttnn.from_torch(box_kernel.unsqueeze(0).unsqueeze(0), device=self.device)

        smoothed_channels = []
        for c in range(3):
            channel = img_tt[0, c:c+1, :, :]
            smoothed = ttnn.conv2d(channel, kernel_tt, padding='same')
            smoothed_channels.append(smoothed)

        result = ttnn.concat(smoothed_channels, dim=1)
        result_torch = ttnn.to_torch(result).squeeze(0).cpu()

        # Combine with quantized colors
        return result_torch * quantized.expand_as(result_torch)

    def custom_kernel(self, image, kernel_matrix):
        """
        Apply custom user-defined kernel.

        Args:
            image: Input image tensor
            kernel_matrix: 2D numpy array or torch tensor

        Returns:
            Filtered image
        """
        if isinstance(kernel_matrix, np.ndarray):
            kernel_matrix = torch.from_numpy(kernel_matrix).float()

        kernel_tt = ttnn.from_torch(
            kernel_matrix.unsqueeze(0).unsqueeze(0),
            device=self.device
        )

        img_tt = ttnn.from_torch(image.unsqueeze(0), device=self.device)

        filtered_channels = []
        for c in range(image.shape[0]):
            channel = img_tt[0, c:c+1, :, :]
            filtered = ttnn.conv2d(channel, kernel_tt, padding='same')
            filtered_channels.append(filtered)

        result = ttnn.concat(filtered_channels, dim=1)
        return ttnn.to_torch(result).squeeze(0).cpu()

# Example usage
if __name__ == "__main__":
    import ttnn

    device = ttnn.open_device(0)
    filters = ImageFilterBank(device)

    # Load image
    image = filters.load_image("examples/sample.jpg")

    # Apply filters
    edge = filters.apply_filter(image, 'edge_detect')
    blurred = filters.apply_filter(image, 'gaussian_blur')
    sharpened = filters.apply_filter(image, 'sharpen')
    embossed = filters.apply_filter(image, 'emboss')

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(edge.permute(1, 2, 0))
    axes[1].set_title("Edge Detect")
    axes[1].axis('off')

    axes[2].imshow(blurred.permute(1, 2, 0))
    axes[2].set_title("Gaussian Blur")
    axes[2].axis('off')

    axes[3].imshow(sharpened.permute(1, 2, 0))
    axes[3].set_title("Sharpened")
    axes[3].axis('off')

    axes[4].imshow(embossed.permute(1, 2, 0))
    axes[4].set_title("Embossed")
    axes[4].axis('off')

    # Oil painting
    oil_paint = filters.oil_painting_effect(image, radius=3)
    axes[5].imshow(oil_paint.permute(1, 2, 0))
    axes[5].set_title("Oil Painting")
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()

    ttnn.close_device(device)
```

---

## Running the Project

**Quick Start - Click to Run:**

[🖼️ Run Image Filters Demo](command:tenstorrent.runImageFilters)

**Manual Commands:**

```bash
cd ~/tt-scratchpad/cookbook/image_filters

# Install dependencies
pip install -r requirements.txt

# Run the demo
python filters.py
```

---

## Extensions

### 1. Real-Time Webcam Processing
```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_torch = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    # Apply filter
    filtered = filters.apply_filter(frame_torch, 'edge_detect')

    # Display
    cv2.imshow('Filtered', filtered.permute(1, 2, 0).numpy())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 2. Neural Style Transfer
Use pre-trained VGG features + TTNN for fast style transfer.

### 3. Seam Carving (Content-Aware Resize)
Find and remove low-energy seams.

### 4. HDR Tone Mapping
Combine multiple exposures.

---

# Wrapping Up

Congratulations! You've completed the TT-Metalium Cookbook. You now have:

- ✅ **4 complete, working projects**
- ✅ **Deep understanding of TTNN operations**
- ✅ **Experience with parallel tile computing**
- ✅ **Foundation for production applications**

## What's Next?

### Combine Projects
- Use audio processor to trigger mandelbrot zoom
- Apply image filters to Game of Life visualization
- Real-time audio-reactive fractal visualizations

### Contribute to tt-metal
- Submit your projects as examples
- Participate in bounty program
- Help other developers on Discord

### Build Production Apps
- Audio plugin (VST)
- Video processing pipeline
- Real-time ML inference
- Custom hardware accelerators

---

## Resources

- **Discord**: [discord.gg/tvhGzHQwaj](https://discord.gg/tvhGzHQwaj)
- **GitHub**: [github.com/tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal)
- **Documentation**: [docs.tenstorrent.com](https://docs.tenstorrent.com)
- **Bounty Program**: [github.com/tenstorrent/tt-metal/issues?label=bounty](https://github.com/tenstorrent/tt-metal/issues?q=is%3Aissue+state%3Aopen+label%3Abounty)

**🚀 Explore Production Models Next:**

Now that you've mastered the fundamentals, explore how these techniques scale to production:

- **Convolution** → Check out `models/demos/yolov12x/` for state-of-the-art object detection
- **Transformers** → Study `models/demos/gemma3/` for multimodal (text + image) AI
- **Semantic Segmentation** → Explore `models/demos/segformer/` for pixel-level image understanding
- **Vision** → Dive into `models/demos/mobilenetv2/` for efficient mobile-scale inference
- **Audio** → See `models/demos/whisper/` for production speech recognition

**📚 Deep Dive:**
- **METALIUM_GUIDE.md** (`~/tt-metal/METALIUM_GUIDE.md`) - Architecture deep-dive
- **Tech Reports** (`~/tt-metal/tech_reports/`) - Flash Attention, optimizations, research papers
- **2025 Tutorials** (`~/tt-metal/ttnn/tutorials/2025_dx_rework/`) - Latest TTNN examples

Happy coding! 🚀