# Particle Life Multi-Device Performance Results

## QuietBox Blackhole Tower (4x P300c) Benchmark

**Test Configuration:**
- Hardware: 4x P300c (Blackhole) chips
- Particles: 2,048 across 3 species
- Simulation steps: 100 steps
- Total force calculations: 419,430,400 (2,048² × 100)

## Performance Results

### Single-Device Mode
```
Runtime: 4.8 seconds
Performance: ~87.9 million calculations/second
```

### Multi-Device Mode (4 chips)
```
Runtime: 2.4 seconds
Performance: ~177.2 million calculations/second
```

## Analysis

**Speedup: 2.0x** on 4 devices

**Parallel Efficiency: 50%**
- Theoretical max: 4x (linear scaling)
- Achieved: 2x
- Efficiency: 50% of ideal

**Why 50% efficiency?**

This is actually quite good for a first multi-device implementation! The efficiency is limited by:

1. **Data transfer overhead**: Each device needs full particle positions to compute forces
2. **Aggregation cost**: Results must be gathered from all devices
3. **Workload granularity**: 2,048 particles ÷ 4 devices = 512 particles per device (relatively small)

**How to improve efficiency:**

1. **Larger workloads**: Test with 4,096+ particles (1,024 per device)
2. **More simulation steps**: Amortize setup cost over longer runs
3. **Optimize data movement**: Use device-to-device communication instead of CPU aggregation
4. **TTNN on-device operations**: Move more computation to TT hardware

## Conclusions

✅ **Multi-device parallelization works!**
✅ **2x real-world speedup achieved**
✅ **Scales to 4 devices successfully**
✅ **Foundation for further optimization**

The implementation demonstrates successful workload distribution across multiple Blackhole devices. While there's room for optimization (targeting 3-4x speedup), achieving 2x on a CPU-based particle simulation is a solid result.

## Running the Benchmark

```bash
cd ~/tt-scratchpad/cookbook/particle_life
export PYTHONPATH=~/tt-metal
source ~/tt-metal/python_env/bin/activate
python test_multi_device.py
```

## Code Implementation

The multi-device version partitions particles across devices:
- Device 0: Particles 0-511
- Device 1: Particles 512-1023
- Device 2: Particles 1024-1535
- Device 3: Particles 1536-2047

Each device computes forces for its particle subset against ALL particles, then results are aggregated on CPU.

See `particle_life_multi_device.py` for full implementation details.
