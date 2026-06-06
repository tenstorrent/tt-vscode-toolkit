"""
ttsim_race_demo.py

Demonstrates the race condition the ttsim simulator catches but silicon may hide.

Two versions of the same program:
  - safe():   explicit barrier between writer and reader — always correct
  - unsafe(): barrier removed — fails on simulator, may pass on silicon

The simulator evaluates operations in any order permitted by your synchronization.
This may include orders that are extremely unlikely on real hardware.

Usage:
    export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
    python3 ttsim_race_demo.py
"""

import torch
import ttnn


def run_with_barrier(device) -> torch.Tensor:
    """Correct version: explicit synchronization between write and read."""
    data = torch.ones(32, 32, dtype=torch.bfloat16)
    buf = ttnn.from_torch(data, layout=ttnn.TILE_LAYOUT, device=device)

    # Write
    buf = ttnn.add(buf, ttnn.from_torch(
        torch.ones(32, 32, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device
    ))

    # Explicit synchronization: bring result to host before reading back
    result = ttnn.to_torch(ttnn.from_device(buf))

    # Read — safe because we synchronized
    buf2 = ttnn.from_torch(result, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.multiply(buf2, ttnn.from_torch(
        torch.full((32, 32), 2.0, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device
    ))
    return ttnn.to_torch(ttnn.from_device(out))


def run_without_barrier(device) -> torch.Tensor:
    """
    Unsafe version: no synchronization between write and read.

    On the ttsim simulator this may produce incorrect results because ttsim
    exercises operation orderings that are unlikely on real hardware.

    Remove the ttnn.from_device() call between the write and the read to
    reproduce the race. On silicon this probably passes. On the simulator
    it may not.
    """
    data = torch.ones(32, 32, dtype=torch.bfloat16)
    buf = ttnn.from_torch(data, layout=ttnn.TILE_LAYOUT, device=device)

    buf = ttnn.add(buf, ttnn.from_torch(
        torch.ones(32, 32, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device
    ))

    # NO SYNCHRONIZATION HERE — the read may see stale data
    # To exercise the race: comment out the line below and see if results differ
    result = ttnn.to_torch(ttnn.from_device(buf))  # remove this line to race

    buf2 = ttnn.from_torch(result, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.multiply(buf2, ttnn.from_torch(
        torch.full((32, 32), 2.0, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device
    ))
    return ttnn.to_torch(ttnn.from_device(out))


def main():
    device = ttnn.open_device(device_id=0)
    try:
        safe_result = run_with_barrier(device)
        unsafe_result = run_without_barrier(device)
    finally:
        ttnn.close_device(device)

    expected = torch.full((32, 32), 4.0, dtype=torch.bfloat16)
    safe_ok = torch.allclose(safe_result, expected)
    unsafe_ok = torch.allclose(unsafe_result, expected)

    print(f"With barrier:    {'CORRECT' if safe_ok else 'WRONG'}")
    print(f"Without barrier: {'CORRECT' if unsafe_ok else 'WRONG (race detected)'}")
    print()
    print("Exercise: comment out the synchronization line in run_without_barrier()")
    print("and run again. The simulator may produce 'WRONG' where silicon would pass.")


if __name__ == "__main__":
    main()
