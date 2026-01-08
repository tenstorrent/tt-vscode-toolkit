# Particle Life - Emergent Complexity Simulator

An emergent complexity simulator where different particle species interact based on randomly generated attraction/repulsion rules. Simple physics creates unpredictable and beautiful patterns.

## Overview

Particle Life demonstrates emergent complexity - how simple rules can create unpredictable and fascinating behaviors:

- **Multiple species** with unique interaction rules
- **N² force calculations** (~2 billion+ per simulation)
- **Real-time physics simulation** (forces, velocities, integration)
- **Beautiful emergent patterns** that are never the same twice

## Quick Start

### Run the Simulation

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation (creates particle_life.gif)
python test_particle_life.py
```

This will:
1. Initialize 2,048 particles across 3 species
2. Generate random attraction/repulsion rules
3. Simulate 500 timesteps of physics
4. Create an animated GIF showing the results

### Expected Output

```
Initializing Particle Life simulation...
✓ TT device opened
✓ 2,048 particles across 3 species
✓ Random attraction matrix generated

Interaction Matrix:
               Red   Green    Blue
       Red    0.23   -0.45    0.67
     Green   -0.12    0.89   -0.34
      Blue    0.78   -0.56    0.11

Simulating 500 steps...
50/500... 100/500... 150/500... 200/500... 250/500... 300/500... 350/500... 400/500... 450/500... 500/500

Simulation complete!
- Total force calculations: 2,097,152,000
- Runtime: ~192 seconds
- Performance: ~10.9 million calculations/second

Rendering animation...
✓ Animation saved to: particle_life.gif
```

## How It Works

### The Science

Each species pair has an attraction/repulsion value:
- **Positive values** → attraction (particles move together)
- **Negative values** → repulsion (particles avoid each other)
- **Zero** → neutral (no interaction)

### Emergent Behaviors

From these simple rules, complex patterns emerge:
- **Clustering** - Species group together
- **Chasing** - Predator-prey dynamics
- **Orbiting** - Stable circular patterns
- **Chaos** - Dissolving into randomness
- **Reformation** - Order emerging from chaos

### The Algorithm

For each timestep:
1. Calculate forces between all particle pairs (N² algorithm)
2. Update velocities based on forces
3. Apply friction to slow particles
4. Update positions based on velocities
5. Apply periodic boundary conditions (wrap around)

## Customization

### Adjust Parameters

Edit `test_particle_life.py`:

```python
# More species = more complex interactions
sim = ParticleLife(
    device=device,
    num_particles=4096,  # More particles
    num_species=5,       # More species
    world_size=1.0,
    force_radius=0.15,   # Larger interaction radius
    friction=0.08        # More friction
)

# Longer simulation
history = sim.simulate(
    num_steps=1000,      # More steps
    dt=0.01,
    sample_every=2       # Sample every 2 frames (saves memory)
)
```

### Custom Interaction Rules

Modify `particle_life.py` to create specific behaviors:

```python
# Predator-prey setup (3 species)
sim.attraction_matrix = torch.tensor([
    [ 0.1, -0.8,  0.0],  # Predators: cluster, chase prey, ignore scavengers
    [ 0.6,  0.0, -0.5],  # Prey: cluster, flee predators, repel scavengers
    [-0.3,  0.4,  0.2],  # Scavengers: avoid predators, attracted to prey, cluster
])
```

### 3D Particle Life

Extend to three dimensions:

```python
# In _initialize_particles():
positions = torch.rand(self.num_particles, 3) * self.world_size

# In _calculate_forces():
# All calculations work the same in 3D!
```

## Performance Notes

**Complexity**: O(N²) per timestep
- 2,048 particles = 4.2M calculations/frame
- 4,096 particles = 16.8M calculations/frame
- 8,192 particles = 67M calculations/frame

**Hardware Recommendations**:
- N150: Up to 4,096 particles
- N300: Up to 8,192 particles
- T3K: Up to 16,384 particles

## What You Learned

- **N² algorithms** - All-pairs force calculations
- **Physics simulation** - Forces, velocities, numerical integration
- **Emergent complexity** - Simple rules → unpredictable patterns
- **Parallel computing** - Structuring workloads for TT hardware
- **Scientific visualization** - Data → insights → beauty

## Learn More

**Inspiration**:
- [Particle Life](https://particle-life.com/) - Original web simulator
- Artificial Life research
- Chaos theory and complexity science

**Related Cookbook Recipes**:
- Recipe 1 (Game of Life) → Cellular automata emergence
- Recipe 3 (Mandelbrot) → Parallel pixel processing

## Troubleshooting

### Out of Memory

Reduce particle count or sample less frequently:
```python
sim = ParticleLife(num_particles=1024)  # Fewer particles
history = sim.simulate(sample_every=5)   # Sample every 5 frames
```

### Slow Performance

Reduce simulation steps or increase dt:
```python
history = sim.simulate(num_steps=200, dt=0.02)
```

### No Interesting Patterns

Try different random seeds or manually set interaction rules:
```python
# Regenerate with new random seed
import random
random.seed(42)
sim = ParticleLife(device=device)
```

## Contributing

Found interesting interaction rules? Share them on Discord!
- [discord.gg/tvhGzHQwaj](https://discord.gg/tvhGzHQwaj)

---

**Happy simulating!** 🌌
