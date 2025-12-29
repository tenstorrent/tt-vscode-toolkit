"""
Mandelbrot Set Renderer using TTNN
Each pixel computed independently - ideal for parallel execution
"""

import ttnn
import torch
import numpy as np
import time

class MandelbrotRenderer:
    def __init__(self, device):
        """
        Initialize Mandelbrot renderer.

        Args:
            device: TTNN device handle
        """
        self.device = device

    def render(self, width, height, x_min, x_max, y_min, y_max, max_iter=256):
        """
        Render Mandelbrot set for given complex plane region.

        Algorithm:
        For each pixel (representing complex number c):
            z = 0
            for i in range(max_iter):
                z = z² + c
                if |z| > 2:
                    return i  (iteration count)
            return max_iter  (in set)

        Args:
            width, height: Image dimensions
            x_min, x_max: Real axis range
            y_min, y_max: Imaginary axis range
            max_iter: Maximum iterations before considering point "in set"

        Returns:
            2D array of iteration counts
        """
        print(f"Rendering {width}×{height} image...")
        print(f"Complex plane: [{x_min}, {x_max}] × [{y_min}, {y_max}]i")
        print(f"Max iterations: {max_iter}")

        start_time = time.time()

        # Create coordinate grids
        x = torch.linspace(x_min, x_max, width, dtype=torch.float32)
        y = torch.linspace(y_min, y_max, height, dtype=torch.float32)

        # Meshgrid for complex plane
        X, Y = torch.meshgrid(x, y, indexing='xy')
        C_real = X.T  # Transpose to match image coordinates
        C_imag = Y.T

        # Move to device
        c_real = ttnn.from_torch(C_real, device=self.device, layout=ttnn.TILE_LAYOUT)
        c_imag = ttnn.from_torch(C_imag, device=self.device, layout=ttnn.TILE_LAYOUT)

        # Initialize z = 0
        z_real = ttnn.zeros_like(c_real)
        z_imag = ttnn.zeros_like(c_imag)

        # Iteration counter (starts at max_iter, decremented when diverged)
        iterations = ttnn.full_like(c_real, max_iter)

        # Iterate z = z² + c
        for i in range(max_iter):
            # Complex multiplication: (a + bi)² = (a² - b²) + (2ab)i
            z_real_sq = ttnn.square(z_real)
            z_imag_sq = ttnn.square(z_imag)

            z_real_new = ttnn.subtract(z_real_sq, z_imag_sq)
            z_real_new = ttnn.add(z_real_new, c_real)

            z_imag_new = ttnn.multiply(z_real, z_imag)
            z_imag_new = ttnn.multiply(z_imag_new, 2.0)
            z_imag_new = ttnn.add(z_imag_new, c_imag)

            # Magnitude: |z| = sqrt(real² + imag²)
            magnitude_sq = ttnn.add(z_real_sq, z_imag_sq)

            # Mark diverged points (|z| > 2, so |z|² > 4)
            diverged = ttnn.gt(magnitude_sq, 4.0)

            # Update iteration count (only for points that just diverged)
            still_iterating = ttnn.eq(iterations, max_iter)
            just_diverged = ttnn.logical_and(diverged, still_iterating)

            # Set iteration count to current iteration
            iterations = ttnn.where(just_diverged, float(i), iterations)

            # Update z for next iteration
            z_real = z_real_new
            z_imag = z_imag_new

            # Early exit if all points diverged
            if i % 10 == 0:  # Check every 10 iterations
                num_still_iterating = ttnn.sum(still_iterating)
                if ttnn.to_torch(num_still_iterating).item() == 0:
                    print(f"All points diverged by iteration {i}")
                    break

        # Convert to numpy
        result = ttnn.to_torch(iterations).cpu().numpy()

        elapsed = time.time() - start_time
        pixels_per_sec = (width * height) / elapsed
        print(f"Rendered in {elapsed:.2f}s ({pixels_per_sec/1e6:.2f} Mpixels/sec)")

        return result

    def render_julia(self, width, height, c_real, c_imag, x_min=-2, x_max=2,
                     y_min=-2, y_max=2, max_iter=256):
        """
        Render Julia set for fixed c (vs Mandelbrot which varies c).

        Julia set: z₀ = pixel coordinate, fixed c, iterate z = z² + c

        Args:
            width, height: Image dimensions
            c_real, c_imag: Fixed complex parameter
            x_min, x_max, y_min, y_max: Coordinate range
            max_iter: Maximum iterations

        Returns:
            2D array of iteration counts
        """
        print(f"Rendering Julia set with c = {c_real} + {c_imag}i")

        # Create initial z (pixel coordinates)
        x = torch.linspace(x_min, x_max, width, dtype=torch.float32)
        y = torch.linspace(y_min, y_max, height, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='xy')

        z_real = ttnn.from_torch(X.T, device=self.device, layout=ttnn.TILE_LAYOUT)
        z_imag = ttnn.from_torch(Y.T, device=self.device, layout=ttnn.TILE_LAYOUT)

        # Fixed c
        c_r = ttnn.full_like(z_real, c_real)
        c_i = ttnn.full_like(z_imag, c_imag)

        # Iterate (same as Mandelbrot)
        iterations = ttnn.full_like(z_real, max_iter)

        for i in range(max_iter):
            z_real_sq = ttnn.square(z_real)
            z_imag_sq = ttnn.square(z_imag)

            z_real_new = ttnn.add(ttnn.subtract(z_real_sq, z_imag_sq), c_r)
            z_imag_new = ttnn.add(ttnn.multiply(ttnn.multiply(z_real, z_imag), 2.0), c_i)

            magnitude_sq = ttnn.add(z_real_sq, z_imag_sq)
            diverged = ttnn.gt(magnitude_sq, 4.0)

            still_iterating = ttnn.eq(iterations, max_iter)
            just_diverged = ttnn.logical_and(diverged, still_iterating)

            iterations = ttnn.where(just_diverged, float(i), iterations)

            z_real = z_real_new
            z_imag = z_imag_new

        return ttnn.to_torch(iterations).cpu().numpy()


# Example usage
if __name__ == "__main__":
    import ttnn
    from explorer import MandelbrotVisualizer

    device = ttnn.open_device(device_id=0)
    renderer = MandelbrotRenderer(device)

    # Classic Mandelbrot view
    mandelbrot = renderer.render(
        width=2048, height=2048,
        x_min=-2.5, x_max=1.0,
        y_min=-1.25, y_max=1.25,
        max_iter=512
    )

    # Visualize
    viz = MandelbrotVisualizer(renderer)
    viz.show(mandelbrot)

    ttnn.close_device(device)
