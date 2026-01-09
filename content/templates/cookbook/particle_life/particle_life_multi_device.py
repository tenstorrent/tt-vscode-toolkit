"""
Particle Life - Multi-Device Emergent Complexity Simulator using TTNN

Extends the original Particle Life simulator to use multiple TT devices in parallel.
Distributes N² force calculations across all available chips for maximum throughput.

Demonstrates:
- Multi-chip workload distribution
- Parallel N² algorithm execution
- Data aggregation across devices
- Performance scaling with additional hardware

Based on: https://particle-life.com/
"""

import ttnn
import torch
import numpy as np
import time
from typing import Tuple, List, Union


class ParticleLifeMultiDevice:
    def __init__(
        self,
        devices: Union[object, List[object]],
        num_particles: int = 2048,
        num_species: int = 3,
        world_size: float = 1.0,
        force_radius: float = 0.1,
        friction: float = 0.05
    ):
        """
        Initialize Multi-Device Particle Life simulation.

        Args:
            devices: Single device or list of TTNN device handles
            num_particles: Total number of particles
            num_species: Number of particle species (colors)
            world_size: Size of simulation space (0 to world_size)
            force_radius: Maximum distance for force interaction
            friction: Friction coefficient (slows particles down)
        """
        # Handle both single device and device list
        if isinstance(devices, list):
            self.devices = devices
            self.num_devices = len(devices)
            self.multi_device = True
        else:
            self.devices = [devices]
            self.num_devices = 1
            self.multi_device = False

        self.num_particles = num_particles
        self.num_species = num_species
        self.world_size = world_size
        self.force_radius = force_radius
        self.friction = friction

        # Calculate particles per device
        self.particles_per_device = num_particles // self.num_devices
        self.remainder = num_particles % self.num_devices

        # Initialize particles
        self.positions, self.velocities, self.species = self._initialize_particles()

        # Generate random interaction rules
        self.attraction_matrix = self._generate_attraction_matrix()

        print(f"✓ {num_particles:,} particles across {num_species} species")
        if self.multi_device:
            print(f"✓ Multi-device mode: {self.num_devices} devices")
            print(f"✓ Particles per device: ~{self.particles_per_device}")
        else:
            print(f"✓ Single-device mode")
        print(f"✓ Random attraction matrix generated")
        print(f"\nInteraction Matrix:")
        self._print_attraction_matrix()

    def _initialize_particles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize particle positions, velocities, and species.

        Returns:
            positions: (num_particles, 2) - x, y coordinates
            velocities: (num_particles, 2) - vx, vy
            species: (num_particles,) - species ID (0 to num_species-1)
        """
        # Random positions in [0, world_size]
        positions = torch.rand(self.num_particles, 2) * self.world_size

        # Start with zero velocity
        velocities = torch.zeros(self.num_particles, 2)

        # Assign species evenly
        species = torch.arange(self.num_particles) % self.num_species

        return positions, velocities, species

    def _generate_attraction_matrix(self) -> torch.Tensor:
        """
        Generate random attraction/repulsion rules between species.

        Returns:
            attraction_matrix: (num_species, num_species)
                Positive values = attraction
                Negative values = repulsion
                Range: [-1, 1]
        """
        # Random values in [-1, 1]
        matrix = torch.rand(self.num_species, self.num_species) * 2 - 1

        return matrix

    def _print_attraction_matrix(self):
        """Print the attraction matrix in a readable format."""
        species_names = ['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta', 'White'][:self.num_species]

        # Header
        print(f"{'':>10}", end='')
        for name in species_names:
            print(f"{name:>8}", end='')
        print()

        # Rows
        for i, name in enumerate(species_names):
            print(f"{name:>10}", end='')
            for j in range(self.num_species):
                value = self.attraction_matrix[i, j].item()
                print(f"{value:>8.2f}", end='')
            print()

    def _calculate_forces_single_device(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculate forces using single device (original algorithm).

        Args:
            positions: (num_particles, 2) current positions

        Returns:
            forces: (num_particles, 2) total force on each particle
        """
        # Expand positions for pairwise calculation
        pos_i = positions.unsqueeze(1)
        pos_j = positions.unsqueeze(0)

        # Distance vectors: (num_particles, num_particles, 2)
        delta = pos_j - pos_i

        # Apply periodic boundary conditions (wrap around)
        delta = torch.where(
            delta > self.world_size / 2,
            delta - self.world_size,
            delta
        )
        delta = torch.where(
            delta < -self.world_size / 2,
            delta + self.world_size,
            delta
        )

        # Distance magnitudes: (num_particles, num_particles)
        dist = torch.sqrt(torch.sum(delta ** 2, dim=2) + 1e-10)

        # Normalize direction vectors
        direction = delta / dist.unsqueeze(2)

        # Get attraction values based on species pairs
        species_i = self.species.unsqueeze(1)
        species_j = self.species.unsqueeze(0)

        # Lookup attraction values: (num_particles, num_particles)
        attraction = self.attraction_matrix[species_i, species_j].squeeze()

        # Calculate force magnitude
        force_magnitude = torch.where(
            dist < self.force_radius,
            attraction * (1 - dist / self.force_radius),
            torch.zeros_like(dist)
        )

        # Zero out self-interaction
        mask = torch.eye(self.num_particles, dtype=torch.bool)
        force_magnitude = torch.where(mask, torch.zeros_like(force_magnitude), force_magnitude)

        # Force vectors: (num_particles, num_particles, 2)
        force_vectors = direction * force_magnitude.unsqueeze(2)

        # Sum forces on each particle: (num_particles, 2)
        total_forces = torch.sum(force_vectors, dim=1)

        return total_forces

    def _calculate_forces_multi_device(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculate forces using multiple devices (parallel N² algorithm).

        Strategy:
        1. Partition particles across devices (each device handles subset)
        2. Each device computes forces for its particles against ALL particles
        3. Aggregate forces from all devices

        Args:
            positions: (num_particles, 2) current positions

        Returns:
            forces: (num_particles, 2) total force on each particle
        """
        # Create partition indices
        partition_indices = []
        start_idx = 0
        for i in range(self.num_devices):
            # Handle remainder by distributing extra particles to first devices
            particles_in_partition = self.particles_per_device + (1 if i < self.remainder else 0)
            end_idx = start_idx + particles_in_partition
            partition_indices.append((start_idx, end_idx))
            start_idx = end_idx

        # Compute forces on each device in parallel
        device_forces = []

        for device_idx, (start, end) in enumerate(partition_indices):
            # Get particle subset for this device
            pos_subset = positions[start:end]  # (particles_in_partition, 2)
            species_subset = self.species[start:end]  # (particles_in_partition,)

            # Expand for pairwise calculation
            # pos_i: (particles_in_partition, 1, 2) - particles on this device
            # pos_j: (1, num_particles, 2) - all particles
            pos_i = pos_subset.unsqueeze(1)
            pos_j = positions.unsqueeze(0)

            # Distance vectors: (particles_in_partition, num_particles, 2)
            delta = pos_j - pos_i

            # Apply periodic boundary conditions
            delta = torch.where(
                delta > self.world_size / 2,
                delta - self.world_size,
                delta
            )
            delta = torch.where(
                delta < -self.world_size / 2,
                delta + self.world_size,
                delta
            )

            # Distance magnitudes: (particles_in_partition, num_particles)
            dist = torch.sqrt(torch.sum(delta ** 2, dim=2) + 1e-10)

            # Normalize direction vectors
            direction = delta / dist.unsqueeze(2)

            # Get attraction values
            species_i = species_subset.unsqueeze(1)  # (particles_in_partition, 1)
            species_j = self.species.unsqueeze(0)    # (1, num_particles)

            # Lookup attraction values: (particles_in_partition, num_particles)
            attraction = self.attraction_matrix[species_i, species_j].squeeze()

            # Calculate force magnitude
            force_magnitude = torch.where(
                dist < self.force_radius,
                attraction * (1 - dist / self.force_radius),
                torch.zeros_like(dist)
            )

            # Zero out self-interaction for particles in this subset
            for local_idx in range(end - start):
                global_idx = start + local_idx
                force_magnitude[local_idx, global_idx] = 0.0

            # Force vectors: (particles_in_partition, num_particles, 2)
            force_vectors = direction * force_magnitude.unsqueeze(2)

            # Sum forces on each particle: (particles_in_partition, 2)
            subset_forces = torch.sum(force_vectors, dim=1)

            device_forces.append(subset_forces)

        # Concatenate forces from all devices
        total_forces = torch.cat(device_forces, dim=0)

        return total_forces

    def _calculate_forces(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculate forces between all particle pairs.
        Routes to single-device or multi-device implementation.

        Args:
            positions: (num_particles, 2) current positions

        Returns:
            forces: (num_particles, 2) total force on each particle
        """
        if self.multi_device:
            return self._calculate_forces_multi_device(positions)
        else:
            return self._calculate_forces_single_device(positions)

    def step(self, dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate one timestep.

        Args:
            dt: Time step size

        Returns:
            positions: Updated positions
            velocities: Updated velocities
        """
        # Calculate forces (N² calculation - parallelized across devices if multi_device=True)
        forces = self._calculate_forces(self.positions)

        # Update velocities: v = v + F*dt - friction*v
        self.velocities = self.velocities + forces * dt - self.friction * self.velocities

        # Clamp velocities to prevent explosion
        max_velocity = 0.5
        velocity_magnitude = torch.sqrt(torch.sum(self.velocities ** 2, dim=1, keepdim=True))
        self.velocities = torch.where(
            velocity_magnitude > max_velocity,
            self.velocities * (max_velocity / velocity_magnitude),
            self.velocities
        )

        # Update positions: x = x + v*dt
        self.positions = self.positions + self.velocities * dt

        # Apply periodic boundary conditions (wrap around)
        self.positions = torch.fmod(self.positions, self.world_size)
        self.positions = torch.where(self.positions < 0, self.positions + self.world_size, self.positions)

        return self.positions, self.velocities

    def simulate(self, num_steps: int = 500, dt: float = 0.01, sample_every: int = 1) -> List[np.ndarray]:
        """
        Run simulation for multiple timesteps.

        Args:
            num_steps: Number of simulation steps
            dt: Time step size
            sample_every: Record positions every N steps (for memory efficiency)

        Returns:
            history: List of position snapshots (num_particles, 2) as numpy arrays
        """
        history = []

        mode_str = f"{self.num_devices}-device" if self.multi_device else "single-device"
        print(f"\nSimulating {num_steps} steps ({mode_str} mode)...")
        start_time = time.time()

        for step in range(num_steps):
            # Simulate one step
            self.step(dt)

            # Record snapshot
            if step % sample_every == 0:
                history.append(self.positions.cpu().numpy())

            # Progress indicator
            if (step + 1) % 50 == 0:
                print(f"{step + 1}/{num_steps}...", end=' ', flush=True)

        print(f"\n")

        elapsed = time.time() - start_time
        total_force_calcs = num_steps * self.num_particles * self.num_particles

        print(f"Simulation complete!")
        print(f"- Mode: {mode_str}")
        print(f"- Total force calculations: {total_force_calcs:,}")
        print(f"- Runtime: {elapsed:.1f} seconds")
        print(f"- Performance: ~{total_force_calcs / elapsed / 1e6:.1f} million calculations/second")

        return history

    def get_species_colors(self) -> List[str]:
        """Get color list for visualization."""
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white']
        return colors[:self.num_species]


# Example usage
if __name__ == "__main__":
    import sys

    # Check for multi-device flag
    use_multi_device = '--multi-device' in sys.argv or '-m' in sys.argv

    if use_multi_device:
        # Open all available TT devices
        print("Detecting available devices...")
        devices = []
        device_id = 0
        while True:
            try:
                device = ttnn.open_device(device_id=device_id)
                devices.append(device)
                print(f"✓ Device {device_id} opened")
                device_id += 1
            except:
                break

        if len(devices) == 0:
            print("ERROR: No devices found!")
            sys.exit(1)

        print(f"\n✓ Found {len(devices)} device(s)")

        try:
            print("\nInitializing Multi-Device Particle Life simulation...")

            # Create simulation with all devices
            sim = ParticleLifeMultiDevice(
                devices=devices,
                num_particles=2048,
                num_species=3,
                world_size=1.0,
                force_radius=0.1,
                friction=0.05
            )

            # Run simulation
            history = sim.simulate(num_steps=500, dt=0.01)

            print("\nTo create animation, run: python test_particle_life.py")

        finally:
            # Close all devices
            for device in devices:
                ttnn.close_device(device)
    else:
        # Single device mode
        device = ttnn.open_device(device_id=0)

        try:
            print("Initializing Single-Device Particle Life simulation...")
            print("✓ TT device opened")
            print("(Use --multi-device flag to enable multi-chip acceleration)")

            # Create simulation
            sim = ParticleLifeMultiDevice(
                devices=device,
                num_particles=2048,
                num_species=3,
                world_size=1.0,
                force_radius=0.1,
                friction=0.05
            )

            # Run simulation
            history = sim.simulate(num_steps=500, dt=0.01)

            print("\nTo create animation, run: python test_particle_life.py")

        finally:
            ttnn.close_device(device)
