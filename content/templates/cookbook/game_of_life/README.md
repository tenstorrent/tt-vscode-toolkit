# Conway's Game of Life

Cellular automaton demonstrating parallel tile computing on TT hardware.

## Files

- `game_of_life.py` - Main implementation
- `patterns.py` - Classic patterns library
- `visualizer.py` - Matplotlib animation

## Quick Start

```bash
pip install -r requirements.txt
python game_of_life.py
```

## Try Different Patterns

```python
from game_of_life import GameOfLife
import ttnn

device = ttnn.open_device(0)
game = GameOfLife(device, grid_size=(256, 256))

# Available patterns: glider, blinker, toad, beacon, pulsar, glider_gun
initial = game.initialize_pattern('glider_gun')

history = game.simulate(initial, num_generations=500)

from visualizer import animate_game_of_life
animate_game_of_life(history, interval=50)

ttnn.close_device(device)
```

## Performance Benchmarking

```python
import time

sizes = [128, 256, 512, 1024]
for size in sizes:
    game = GameOfLife(device, grid_size=(size, size))
    initial = game.initialize_random(0.3)

    start = time.time()
    game.simulate(initial, num_generations=100)
    elapsed = time.time() - start

    print(f"{size}×{size}: {100/elapsed:.2f} gen/sec")
```

## Extensions

- Custom rule sets (HighLife, Day & Night)
- Multi-color variants
- 3D Game of Life
- Larger grid sizes (benchmark performance)

See **Lesson 12** for complete implementation details and extensions.
