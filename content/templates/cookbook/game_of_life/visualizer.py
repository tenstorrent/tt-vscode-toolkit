"""
Visualization for Game of Life using matplotlib

Requirements:
    pip install matplotlib

See Lesson 12 for complete implementation with:
- animate_game_of_life() - Animated visualization
- plot_generation() - Single frame display
- compare_patterns() - Side-by-side pattern comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=1000//interval)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")

    plt.tight_layout()
    plt.show()

    return anim
