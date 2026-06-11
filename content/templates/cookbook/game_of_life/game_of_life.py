"""
Conway's Game of Life using TTNN matrix multiplication.

Neighbour counting is performed with two circulant-matrix matmuls:

    K_row (H×H) — sums each row with its two cyclic neighbours
    K_col (W×W) — sums each column with its two cyclic neighbours
    N = K_row @ G @ K_col − G   →  8-neighbour count per cell

On real Tenstorrent hardware both matmuls run on the Matrix Engine.
On the ttsim software simulator (TT_METAL_SIMULATOR is set) the same
math runs via torch.mm on CPU — faster than routing through the sim's
kernel dispatch stack for this pure-integer workload. Conway's rules
are applied in PyTorch on the CPU in both paths.

Run:
    python game_of_life.py
    python game_of_life.py --pattern glider_gun --generations 500 --size 256
"""

import argparse
import os

import numpy as np
import torch
import ttnn


def _on_simulator() -> bool:
    """True when TT_METAL_SIMULATOR is set (ttsim software simulator)."""
    return bool(os.environ.get("TT_METAL_SIMULATOR"))


def _circulant_shift_sum(n: int) -> torch.Tensor:
    """Return an n×n circulant matrix whose product sums each element with its two cyclic neighbours.

    For any vector v:  (M @ v)[i] = v[i-1] + v[i] + v[i+1]  (indices mod n)

    Composing two of these — M_H @ G @ M_W — implements a 3×3 neighbourhood
    sum over the entire grid with a single pair of matmuls.
    """
    M = torch.zeros(n, n, dtype=torch.bfloat16)
    for i in range(n):
        M[i, i] = 1.0              # self
        M[i, (i - 1) % n] = 1.0   # previous
        M[i, (i + 1) % n] = 1.0   # next
    return M


class GameOfLife:
    def __init__(self, device, grid_size=(128, 128)):
        """
        Initialise Game of Life on TT hardware (or ttsim).

        Args:
            device:    TTNN device handle
            grid_size: (H, W) — multiples of 32 required for tile alignment
        """
        self.device = device
        self.grid_size = grid_size
        self._sim = _on_simulator()

        H, W = grid_size
        K_row_cpu = _circulant_shift_sum(H)   # H×H
        K_col_cpu = _circulant_shift_sum(W)   # W×W

        if self._sim:
            # CPU path: keep as float32 for torch.mm
            self.K_row_cpu = K_row_cpu.float()
            self.K_col_cpu = K_col_cpu.float()
            print("ℹ️  ttsim mode — neighbour counting via torch.mm on CPU")
        else:
            # Hardware path: place on device for ttnn.matmul (Matrix Engine)
            self.K_row = ttnn.from_torch(K_row_cpu, device=device, layout=ttnn.TILE_LAYOUT)
            self.K_col = ttnn.from_torch(K_col_cpu, device=device, layout=ttnn.TILE_LAYOUT)

    def initialize_random(self, density: float = 0.3):
        """Random initial grid.  Returns a TTNN tensor on device, or a CPU tensor in sim mode."""
        H, W = self.grid_size
        cpu = (torch.rand(H, W) < density).to(torch.bfloat16)
        if self._sim:
            return cpu
        return ttnn.from_torch(cpu, device=self.device, layout=ttnn.TILE_LAYOUT)

    def initialize_pattern(self, pattern_name: str):
        """Named pattern centred on an otherwise-empty grid."""
        from patterns import get_pattern

        H, W = self.grid_size
        grid_cpu = torch.zeros(H, W, dtype=torch.bfloat16)
        pattern = torch.tensor(get_pattern(pattern_name), dtype=torch.bfloat16)
        ph, pw = pattern.shape
        r0 = (H - ph) // 2
        c0 = (W - pw) // 2
        grid_cpu[r0:r0 + ph, c0:c0 + pw] = pattern

        if self._sim:
            return grid_cpu
        return ttnn.from_torch(grid_cpu, device=self.device, layout=ttnn.TILE_LAYOUT)

    # ------------------------------------------------------------------
    # Neighbour counting — two implementations, same circulant-matmul math
    # ------------------------------------------------------------------

    def _count_neighbors_hw(self, grid_tt):
        """
        Hardware path: count 8-neighbours using two Matrix Engine matmuls.

        K_row @ G            sums three consecutive rows   (H×H × H×W → H×W)
        (K_row @ G) @ K_col  sums three consecutive columns (H×W × W×W → H×W)

        The result includes each cell itself (from the identity component of
        each circulant matrix), so we subtract G once to get the 8-neighbour sum.
        """
        temp  = ttnn.matmul(self.K_row, grid_tt)   # vertical neighbourhood
        N_raw = ttnn.matmul(temp, self.K_col)       # horizontal neighbourhood

        N_cpu    = ttnn.to_torch(N_raw).float()
        grid_cpu = ttnn.to_torch(grid_tt).float()
        return N_cpu - grid_cpu, grid_cpu

    def _count_neighbors_sim(self, grid_cpu: torch.Tensor):
        """Simulator path: identical math via torch.mm on CPU."""
        g     = grid_cpu.float()
        temp  = torch.mm(self.K_row_cpu, g)
        N_raw = torch.mm(temp, self.K_col_cpu)
        return N_raw - g, g

    # ------------------------------------------------------------------

    def step(self, grid):
        """
        Advance one generation.

        Neighbour count:  Matrix Engine (real HW) or torch.mm (ttsim / CPU)
        Rule application: PyTorch on CPU — no SFPU dependency
        """
        if self._sim:
            neighbors_cpu, grid_f = self._count_neighbors_sim(grid)
        else:
            neighbors_cpu, grid_f = self._count_neighbors_hw(grid)

        # Conway's rules — all operations are simple boolean comparisons on CPU
        alive = grid_f > 0.5
        n     = neighbors_cpu
        next_alive = (alive & ((n == 2) | (n == 3))) | (~alive & (n == 3))
        next_cpu   = next_alive.to(torch.bfloat16)

        if self._sim:
            return next_cpu
        return ttnn.from_torch(next_cpu, device=self.device, layout=ttnn.TILE_LAYOUT)

    def simulate(self, initial_grid, num_generations: int = 100):
        """
        Run for num_generations steps.

        Returns:
            List of H×W float32 numpy arrays, one per generation.
        """
        history = []
        grid    = initial_grid

        for gen in range(num_generations):
            if self._sim:
                grid_np = grid.float().numpy()
            else:
                grid_np = ttnn.to_torch(grid).float().cpu().numpy()

            if gen > 0 and np.array_equal(grid_np, history[-1]):
                print(f"Stable state reached at generation {gen}")
                history.append(grid_np)
                break

            history.append(grid_np)
            grid = self.step(grid)

        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conway's Game of Life on TT hardware")
    parser.add_argument("--pattern", default=None,
                        help="Named pattern: glider, blinker, toad, beacon, pulsar, glider_gun")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--size", type=int, default=256,
                        help="Grid side length — must be a multiple of 32")
    parser.add_argument("--density", type=float, default=0.3,
                        help="Random initial density (0.0–1.0)")
    parser.add_argument("--save", default=None,
                        help="Save animation to this .gif path instead of displaying")
    args = parser.parse_args()

    if args.size % 32 != 0:
        parser.error(f"--size must be a multiple of 32 for tile alignment (got {args.size})")

    device = ttnn.open_device(device_id=0)

    try:
        game = GameOfLife(device, grid_size=(args.size, args.size))

        if args.pattern:
            print(f"Initialising with pattern: {args.pattern}")
            initial = game.initialize_pattern(args.pattern)
        else:
            print(f"Initialising {args.size}×{args.size} random grid (density={args.density})")
            initial = game.initialize_random(density=args.density)

        print(f"Running {args.generations} generations…")
        history = game.simulate(initial, num_generations=args.generations)
        print(f"✅ Simulation complete — {len(history)} generations.")

        try:
            import matplotlib
            if "DISPLAY" not in os.environ:
                matplotlib.use("Agg")
            from visualizer import animate_game_of_life

            save_path = args.save or (None if "DISPLAY" in os.environ else "game_of_life.gif")
            if save_path:
                print(f"💾 Saving animation to {save_path} …")
                animate_game_of_life(history, interval=50, save_path=save_path)
                print(f"✅ Saved.")
            else:
                animate_game_of_life(history, interval=50)

        except ImportError as exc:
            print(f"\n⚠️  Visualisation requires matplotlib: {exc}")
            print("Install with: pip install matplotlib")

    finally:
        ttnn.close_device(device)
        print("Done!")
