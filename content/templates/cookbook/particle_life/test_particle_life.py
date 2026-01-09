"""
Test script for Particle Life simulation

Runs the simulation and creates an animated GIF showing emergent patterns.
"""

import ttnn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from particle_life import ParticleLife


def create_animation(history, species, colors, filename='particle_life.gif', fps=30):
    """
    Create animated GIF from simulation history.

    Args:
        history: List of position snapshots
        species: Particle species array
        colors: List of color names for each species
        filename: Output filename
        fps: Frames per second
    """
    print("\nRendering animation...")

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Create scatter plot for each species
    scatter_plots = []
    species_np = species.cpu().numpy()

    for species_id, color in enumerate(colors):
        mask = species_np == species_id
        scatter = ax.scatter([], [], c=color, s=20, alpha=0.8, edgecolors='none')
        scatter_plots.append((scatter, mask))

    # Title text
    title_text = ax.text(
        0.5, 0.98,
        'Frame: 0',
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=14,
        color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white')
    )

    def update(frame_idx):
        """Update function for animation."""
        positions = history[frame_idx]

        # Update each species scatter plot
        for (scatter, mask), species_id in zip(scatter_plots, range(len(colors))):
            species_positions = positions[mask]
            scatter.set_offsets(species_positions)

        # Update title
        title_text.set_text(f'Frame: {frame_idx}/{len(history)}')

        return [s for s, _ in scatter_plots] + [title_text]

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=1000 / fps,
        blit=True,
        repeat=True
    )

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)

    print(f"✓ Animation saved to: {filename}")
    plt.close(fig)


def main():
    """Run simulation and create visualization."""
    # Initialize TT device
    device = ttnn.open_device(device_id=0)

    try:
        print("="*60)
        print("Particle Life - Emergent Complexity Simulator")
        print("="*60)
        print("\nInitializing simulation...")
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
        history = sim.simulate(
            num_steps=500,
            dt=0.01,
            sample_every=1  # Record every frame
        )

        # Create animation
        create_animation(
            history=history,
            species=sim.species,
            colors=sim.get_species_colors(),
            filename='particle_life.gif',
            fps=30
        )

        print("\n" + "="*60)
        print("Simulation complete!")
        print("="*60)
        print(f"\nOpen particle_life.gif to see the emergent patterns.")
        print("\nNotice how:")
        print("- Particles cluster and separate based on attraction rules")
        print("- Complex patterns emerge from simple physics")
        print("- Order appears, then dissolves back into chaos")
        print("- No two simulations are ever the same!")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
