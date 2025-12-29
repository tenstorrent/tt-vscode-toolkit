"""
Conway's Game of Life using TTNN
Implements parallel computation across tiles for efficient execution.

Run:
    python game_of_life.py
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

    def count_neighbors(self, grid):
        """
        Count neighbors for each cell using shifts and additions.
        This is a simpler alternative to convolution that works with TTNN.

        Args:
            grid: Current state (TTNN tensor)

        Returns:
            Neighbor count tensor
        """
        # Convert to torch for easier manipulation
        grid_torch = ttnn.to_torch(grid).squeeze()
        h, w = grid_torch.shape

        # Manual circular padding (wrap around edges)
        # Create padded grid by wrapping edges
        padded = torch.zeros((h + 2, w + 2), dtype=grid_torch.dtype)

        # Center
        padded[1:-1, 1:-1] = grid_torch

        # Edges (wrap around)
        padded[0, 1:-1] = grid_torch[-1, :]    # top edge from bottom
        padded[-1, 1:-1] = grid_torch[0, :]    # bottom edge from top
        padded[1:-1, 0] = grid_torch[:, -1]    # left edge from right
        padded[1:-1, -1] = grid_torch[:, 0]    # right edge from left

        # Corners (wrap around both dimensions)
        padded[0, 0] = grid_torch[-1, -1]      # top-left from bottom-right
        padded[0, -1] = grid_torch[-1, 0]      # top-right from bottom-left
        padded[-1, 0] = grid_torch[0, -1]      # bottom-left from top-right
        padded[-1, -1] = grid_torch[0, 0]      # bottom-right from top-left

        # Count all 8 neighbors using shifts
        neighbors = torch.zeros_like(grid_torch)

        # Top-left, top, top-right
        neighbors += padded[0:h, 0:w]      # top-left
        neighbors += padded[0:h, 1:w+1]    # top
        neighbors += padded[0:h, 2:w+2]    # top-right

        # Left, right
        neighbors += padded[1:h+1, 0:w]    # left
        neighbors += padded[1:h+1, 2:w+2]  # right

        # Bottom-left, bottom, bottom-right
        neighbors += padded[2:h+2, 0:w]    # bottom-left
        neighbors += padded[2:h+2, 1:w+1]  # bottom
        neighbors += padded[2:h+2, 2:w+2]  # bottom-right

        # Convert back to TTNN tensor
        return ttnn.from_torch(
            neighbors.unsqueeze(0).unsqueeze(0),
            device=self.device,
            layout=ttnn.TILE_LAYOUT
        )

    def step(self, grid):
        """
        Compute one generation of the Game of Life.

        Conway's Rules:
        - Birth: dead cell with exactly 3 neighbors becomes alive
        - Survival: live cell with 2-3 neighbors stays alive
        - Death: all other cells die or stay dead

        Args:
            grid: Current state (TTNN tensor)

        Returns:
            Next state (TTNN tensor)
        """
        # Count neighbors
        neighbors = self.count_neighbors(grid)

        # Apply Conway's rules using element-wise operations
        # Birth: neighbors == 3 AND cell == 0
        birth = ttnn.logical_and(
            ttnn.eq(neighbors, 3.0),
            ttnn.eq(grid, 0.0)
        )

        # Survival: (neighbors == 2 OR neighbors == 3) AND cell == 1
        survival_condition = ttnn.logical_or(
            ttnn.eq(neighbors, 2.0),
            ttnn.eq(neighbors, 3.0)
        )
        survival = ttnn.logical_and(survival_condition, ttnn.eq(grid, 1.0))

        # New state: birth OR survival
        next_grid = ttnn.logical_or(birth, survival)

        # Convert bool back to float (bfloat16 is the standard dtype for ttnn)
        return ttnn.typecast(next_grid, dtype=ttnn.bfloat16)

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
            # Compute next generation
            grid = self.step(grid)

            # Store state (convert to numpy for visualization)
            # .float() converts bfloat16 to float32, which NumPy supports
            grid_np = ttnn.to_torch(grid).squeeze().float().cpu().numpy()
            history.append(grid_np)

            # Optional: Check for stability (compare with previous state)
            if gen > 0 and np.array_equal(history[-2], history[-1]):
                print(f"Stable state reached at generation {gen+1}")
                break

        return history


# Example usage
if __name__ == "__main__":
    # Initialize device
    device = ttnn.open_device(device_id=0)

    # Create game
    game = GameOfLife(device, grid_size=(256, 256))

    # Initialize with random configuration
    initial = game.initialize_random(density=0.3)

    # Or initialize with a pattern:
    # initial = game.initialize_pattern('glider')

    # Run simulation
    print("Running Game of Life simulation...")
    history = game.simulate(initial, num_generations=200)

    # Visualize results
    print(f"\n✅ Simulation complete! Generated {len(history)} generations.")

    # Try to visualize (requires matplotlib)
    try:
        import matplotlib
        # Check if we're in a headless environment
        import os
        if 'DISPLAY' not in os.environ and matplotlib.get_backend() != 'agg':
            print("📊 Headless environment detected, using non-interactive backend...")
            matplotlib.use('Agg')  # Non-interactive backend

        from visualizer import animate_game_of_life

        # Check if we should save to file instead of displaying
        if 'DISPLAY' not in os.environ or matplotlib.get_backend() == 'agg':
            print("💾 Saving animation to game_of_life.gif...")
            animate_game_of_life(history, interval=50, save_path='game_of_life.gif')
            print("✅ Animation saved! Download game_of_life.gif to view.")
        else:
            print("🎬 Starting animation... (close window to exit)")
            animate_game_of_life(history, interval=50)

    except ImportError as e:
        print(f"\n⚠️  Visualization requires matplotlib: {e}")
        print("Install with: pip install matplotlib")
        print(f"\nSimulation data saved in memory ({len(history)} frames)")
        print("You can still access 'history' variable for analysis.")
    except Exception as e:
        print(f"\n⚠️  Visualization error: {e}")
        print(f"Simulation data is available in 'history' variable ({len(history)} frames)")

    # Cleanup
    ttnn.close_device(device)
    print("Done!")
