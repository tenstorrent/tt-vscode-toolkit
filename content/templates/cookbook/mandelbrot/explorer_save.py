"""
Interactive Mandelbrot explorer that saves outputs to files
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime

class MandelbrotVisualizer:
    def __init__(self, renderer, output_dir="./outputs"):
        """
        Initialize visualizer with file output.

        Args:
            renderer: MandelbrotRenderer instance
            output_dir: Directory to save images
        """
        self.renderer = renderer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Color maps to try
        self.colormaps = ['hot', 'viridis', 'twilight', 'gist_earth', 'nipy_spectral']
        self.current_cmap = 0

    def show(self, fractal_data, title="Mandelbrot Set", filename=None):
        """
        Display fractal with nice color mapping and save to file.

        Args:
            fractal_data: 2D array of iteration counts
            title: Plot title
            filename: Output filename (auto-generated if None)
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Logarithmic color scale for better visualization
        max_iter = fractal_data.max()
        color_data = np.log(fractal_data + 1)

        img = ax.imshow(
            color_data,
            cmap=self.colormaps[self.current_cmap],
            origin='lower',
            extent=[-2.5, 1.0, -1.25, 1.25],
            interpolation='bilinear'
        )

        ax.set_title(title)
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')

        plt.colorbar(img, ax=ax, label='log(iterations)')
        plt.tight_layout()

        # Save to file
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mandelbrot_{timestamp}.png"

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Saved visualization to: {output_path}")
        return output_path

    def render_sequence(self, zoom_points, width=1024, height=1024, max_iter=256):
        """
        Render a sequence of zoom levels and save each frame.

        Args:
            zoom_points: List of (x_center, y_center, zoom_level) tuples
            width, height: Image dimensions
            max_iter: Maximum iterations

        Returns:
            List of output file paths
        """
        output_paths = []

        for idx, (x_center, y_center, zoom) in enumerate(zoom_points):
            # Calculate bounds based on zoom level
            x_range = 3.5 / zoom  # Initial range is 3.5
            y_range = 2.5 / zoom  # Initial range is 2.5

            x_min = x_center - x_range / 2
            x_max = x_center + x_range / 2
            y_min = y_center - y_range / 2
            y_max = y_center + y_range / 2

            # Increase iterations for deeper zooms
            current_max_iter = int(max_iter * (1 + np.log10(zoom)))

            print(f"\n📸 Frame {idx+1}/{len(zoom_points)}")
            fractal = self.renderer.render(
                width, height,
                x_min, x_max, y_min, y_max,
                current_max_iter
            )

            filename = f"mandelbrot_zoom_{idx:03d}.png"
            path = self.show(
                fractal,
                title=f"Mandelbrot Set (zoom={zoom:.1f}x, iter={current_max_iter})",
                filename=filename
            )
            output_paths.append(path)

        print(f"\n✅ Rendered {len(output_paths)} frames to {self.output_dir}/")
        return output_paths

    def compare_julia_sets(self, c_values, width=512, height=512):
        """
        Display multiple Julia sets side-by-side and save to file.

        Args:
            c_values: List of (real, imag) tuples
            width, height: Resolution per image
        """
        n = len(c_values)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (c_real, c_imag) in zip(axes[:n], c_values):
            julia = self.renderer.render_julia(
                width, height,
                c_real, c_imag,
                max_iter=256
            )

            color_data = np.log(julia + 1)
            ax.imshow(color_data, cmap='twilight', origin='lower')
            ax.set_title(f'c = {c_real:.3f} + {c_imag:.3f}i')
            ax.axis('off')

        # Hide unused subplots
        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"julia_comparison_{timestamp}.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Saved Julia set comparison to: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    import ttnn
    from renderer import MandelbrotRenderer

    device = ttnn.open_device(device_id=0)
    renderer = MandelbrotRenderer(device)
    viz = MandelbrotVisualizer(renderer, output_dir="./mandelbrot_outputs")

    # Classic Mandelbrot view
    print("🎨 Rendering classic Mandelbrot set...")
    mandelbrot = renderer.render(
        width=2048, height=2048,
        x_min=-2.5, x_max=1.0,
        y_min=-1.25, y_max=1.25,
        max_iter=512
    )
    viz.show(mandelbrot, title="Classic Mandelbrot Set")

    # Zoom sequence into interesting region
    print("\n🔍 Rendering zoom sequence...")
    zoom_points = [
        (-0.5, 0.0, 1),      # Full view
        (-0.5, 0.0, 4),      # Zoom 4x
        (-0.7, 0.0, 16),     # Zoom 16x
        (-0.75, 0.1, 64),    # Zoom 64x
    ]
    viz.render_sequence(zoom_points, width=1024, height=1024)

    # Julia sets
    print("\n🌀 Rendering Julia sets...")
    c_values = [
        (-0.4, 0.6),
        (-0.8, 0.156),
        (0.285, 0.01),
        (-0.7269, 0.1889),
    ]
    viz.compare_julia_sets(c_values, width=512, height=512)

    ttnn.close_device(device)
    print("\n✨ Done! Check the mandelbrot_outputs/ directory")
