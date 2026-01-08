"""
Particle Life - Emergent Complexity Simulator using TTNN

Simulates multiple particle species with randomly generated attraction/repulsion rules.
Simple physics creates unpredictable and beautiful emergent patterns.

Based on: https://particle-life.com/
"""

import ttnn
import torch
import numpy as np
import time
from typing import Tuple, List


class ParticleLife:
    def __init__(
        self,
        device,
        num_particles: int = 2048,
        num_species: int = 3,
        world_size: float = 1.0,
        force_radius: float = 0.1,
        friction: float = 0.05
    ):
        """
        Initialize Particle Life simulation.

        Args:
            device: TTNN device handle
            num_particles: Total number of particles
            num_species: Number of particle species (colors)
            world_size: Size of simulation space (0 to world_size)
            force_radius: Maximum distance for force interaction
            friction: Friction coefficient (slows particles down)
        """
        self.device = device
        self.num_particles = num_particles
        self.num_species = num_species
        self.world_size = world_size
        self.force_radius = force_radius
        self.friction = friction

        # Initialize particles
        self.positions, self.velocities, self.species = self._initialize_particles()

        # Generate random interaction rules
        self.attraction_matrix = self._generate_attraction_matrix()

        print(f"✓ {num_particles:,} particles across {num_species} species")
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

    def _calculate_forces(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculate forces between all particle pairs (N² algorithm).

        Args:
            positions: (num_particles, 2) current positions

        Returns:
            forces: (num_particles, 2) total force on each particle
        """
        # Expand positions for pairwise calculation
        # pos_i: (num_particles, 1, 2)
        # pos_j: (1, num_particles, 2)
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
        dist = torch.sqrt(torch.sum(delta ** 2, dim=2) + 1e-10)  # Add epsilon to avoid division by zero

        # Normalize direction vectors
        direction = delta / dist.unsqueeze(2)

        # Get attraction values based on species pairs
        # species_i: (num_particles, 1)
        # species_j: (1, num_particles)
        species_i = self.species.unsqueeze(1)
        species_j = self.species.unsqueeze(0)

        # Lookup attraction values: (num_particles, num_particles)
        attraction = self.attraction_matrix[species_i, species_j].squeeze()

        # Calculate force magnitude
        # Force decays with distance (beyond force_radius)
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

    def step(self, dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate one timestep.

        Args:
            dt: Time step size

        Returns:
            positions: Updated positions
            velocities: Updated velocities
        """
        # Calculate forces (N² calculation)
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

        print(f"\nSimulating {num_steps} steps...")
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
    # Initialize TT device
    device = ttnn.open_device(device_id=0)

    try:
        print("Initializing Particle Life simulation...")
        print("✓ TT device opened")

        # Create simulation
        sim = ParticleLife(
            device=device,
            num_particles=2048,
            num_species=3,
            world_size=1.0,
            force_radius=0.1,
            friction=0.05
        )

        # Run simulation
        history = sim.simulate(num_steps=500, dt=0.01)

        # Save animation (see test_particle_life.py for visualization code)
        print("\nTo create animation, run: python test_particle_life.py")

    finally:
        ttnn.close_device(device)
