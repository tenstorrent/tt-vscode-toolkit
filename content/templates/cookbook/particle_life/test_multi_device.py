"""
Multi-Device Performance Test for Particle Life

Compares single-device vs multi-device performance on QuietBox systems.
Demonstrates scaling efficiency across multiple P300c chips.
"""

import ttnn
import sys
import time
from particle_life_multi_device import ParticleLifeMultiDevice


def test_single_device():
    """Run simulation on single device."""
    print("=" * 60)
    print("TEST 1: SINGLE-DEVICE MODE")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)

    try:
        sim = ParticleLifeMultiDevice(
            devices=device,
            num_particles=2048,
            num_species=3,
            world_size=1.0,
            force_radius=0.1,
            friction=0.05
        )

        # Run shorter simulation for benchmarking
        history = sim.simulate(num_steps=100, dt=0.01)

        return history

    finally:
        ttnn.close_device(device)


def test_multi_device():
    """Run simulation on all available devices."""
    print("\n" + "=" * 60)
    print("TEST 2: MULTI-DEVICE MODE")
    print("=" * 60)

    # Detect number of available devices
    print("\nDetecting available devices...")
    num_devices = ttnn.GetNumAvailableDevices()

    if num_devices == 0:
        print("ERROR: No devices found!")
        return None

    print(f"✓ Found {num_devices} device(s)")

    # For multi-device, use CreateDevices API for proper coordination
    devices = []
    opened_device_ids = list(range(num_devices))

    try:
        # Open devices using CreateDevices for proper multi-device support
        # This is required to avoid dispatch core errors during cleanup
        devices = ttnn.CreateDevices(opened_device_ids)

        print(f"✓ Opened {len(devices)} devices using CreateDevices API")

        sim = ParticleLifeMultiDevice(
            devices=devices,
            num_particles=2048,
            num_species=3,
            world_size=1.0,
            force_radius=0.1,
            friction=0.05
        )

        # Run same simulation for fair comparison
        history = sim.simulate(num_steps=100, dt=0.01)

        return history

    finally:
        # Close devices using proper multi-device cleanup
        if devices:
            ttnn.CloseDevices(devices)


def main():
    """Run both tests and compare performance."""
    print("\n🚀 PARTICLE LIFE MULTI-DEVICE BENCHMARK\n")
    print("This test compares single-device vs multi-device performance")
    print("by running 100 simulation steps on 2,048 particles.\n")

    # Test single device
    single_start = time.time()
    single_history = test_single_device()
    single_elapsed = time.time() - single_start

    # Test multi-device
    multi_start = time.time()
    multi_history = test_multi_device()
    multi_elapsed = time.time() - multi_start

    # Results summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    if single_history and multi_history:
        speedup = single_elapsed / multi_elapsed

        print(f"\nSingle-device time: {single_elapsed:.1f}s")
        print(f"Multi-device time:  {multi_elapsed:.1f}s")
        print(f"\n🎯 Speedup: {speedup:.2f}x")

        if speedup > 1:
            print(f"✅ Multi-device is {speedup:.2f}x FASTER!")
        else:
            print(f"⚠️  Multi-device overhead detected (may be due to small workload)")

        # Calculate efficiency
        try:
            # Detect number of devices used
            num_devices = len([line for line in open('/dev/tenstorrent').readlines()])
        except:
            num_devices = 4  # Assume 4 devices for QB

        efficiency = (speedup / num_devices) * 100
        print(f"\n📊 Parallel efficiency: {efficiency:.1f}%")
        print(f"   (Ideal = 100% on {num_devices} devices)")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if speedup < 1.5:
            print("   - Try larger workloads (4096+ particles)")
            print("   - Increase simulation steps (500+)")
        elif speedup < num_devices * 0.8:
            print("   - Good scaling, but room for improvement")
            print("   - Check for data transfer bottlenecks")
        else:
            print("   - Excellent scaling! Near-linear speedup achieved")
            print("   - Multi-device mode recommended for production")


if __name__ == "__main__":
    main()
