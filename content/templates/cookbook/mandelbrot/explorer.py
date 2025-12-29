"""
Interactive Mandelbrot explorer with zoom and pan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MandelbrotVisualizer:
    def __init__(self, renderer):
        """
        Initialize visualizer.

        Args:
            renderer: MandelbrotRenderer instance
        """
        self.renderer = renderer

        # Color maps to try
        self.colormaps = ['hot', 'viridis', 'twilight', 'gist_earth', 'nipy_spectral']
        self.current_cmap = 0

    def show(self, fractal_data, title="Mandelbrot Set"):
        """
        Display fractal with nice color mapping.

        Args:
            fractal_data: 2D array of iteration counts
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Logarithmic color scale for better visualization
        # Points in set (max_iter) shown in black
        max_iter = fractal_data.max()
        color_data = np.log(fractal_data + 1)  # +1 to avoid log(0)

        img = ax.imshow(
            color_data,
            cmap=self.colormaps[self.current_cmap],
            origin='lower',
            extent=[-2.5, 1.0, -1.25, 1.25],  # Default Mandelbrot bounds
            interpolation='bilinear'
        )

        ax.set_title(title)
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')

        plt.colorbar(img, ax=ax, label='log(iterations)')
        plt.tight_layout()
        plt.show()

    def interactive_explorer(self, width=1024, height=1024, initial_max_iter=256):
        """
        Interactive explorer with click-to-zoom.

        Click on plot to zoom into that region.
        Press 'r' to reset view.
        Press 'c' to cycle color maps.
        Press 'q' to quit.
        """
        # Initial view (full Mandelbrot set)
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.25, 1.25
        max_iter = initial_max_iter

        # Zoom history for undo
        view_history = [(x_min, x_max, y_min, y_max, max_iter)]

        # Render initial view
        fractal = self.renderer.render(width, height, x_min, x_max, y_min, y_max, max_iter)

        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 12))
        color_data = np.log(fractal + 1)
        img = ax.imshow(
            color_data,
            cmap=self.colormaps[self.current_cmap],
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            interpolation='bilinear',
            picker=True
        )

        ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')
        plt.colorbar(img, ax=ax, label='log(iterations)')

        instructions = ax.text(
            0.02, 0.98,
            'Click: Zoom | R: Reset | C: Color | Q: Quit | U: Undo',
            transform=ax.transAxes,
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        def on_click(event):
            nonlocal x_min, x_max, y_min, y_max, max_iter, fractal

            if event.inaxes != ax:
                return

            # Get click coordinates
            click_x, click_y = event.xdata, event.ydata

            # Zoom factor
            zoom = 0.25  # Show 25% of current view

            # New bounds centered on click
            x_range = (x_max - x_min) * zoom
            y_range = (y_max - y_min) * zoom

            x_min_new = click_x - x_range / 2
            x_max_new = click_x + x_range / 2
            y_min_new = click_y - y_range / 2
            y_max_new = click_y + y_range / 2

            # Increase iterations for deeper zooms
            max_iter = min(max_iter + 128, 2048)

            # Save to history
            view_history.append((x_min_new, x_max_new, y_min_new, y_max_new, max_iter))

            # Render new view
            print(f"\nZooming to ({click_x:.6f}, {click_y:.6f})...")
            fractal = self.renderer.render(
                width, height,
                x_min_new, x_max_new, y_min_new, y_max_new,
                max_iter
            )

            # Update plot
            x_min, x_max = x_min_new, x_max_new
            y_min, y_max = y_min_new, y_max_new

            color_data = np.log(fractal + 1)
            img.set_data(color_data)
            img.set_extent([x_min, x_max, y_min, y_max])
            ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal x_min, x_max, y_min, y_max, max_iter, fractal

            if event.key == 'r':
                # Reset to initial view
                print("\nResetting view...")
                x_min, x_max = -2.5, 1.0
                y_min, y_max = -1.25, 1.25
                max_iter = initial_max_iter
                view_history.clear()
                view_history.append((x_min, x_max, y_min, y_max, max_iter))

                fractal = self.renderer.render(width, height, x_min, x_max, y_min, y_max, max_iter)
                color_data = np.log(fractal + 1)
                img.set_data(color_data)
                img.set_extent([x_min, x_max, y_min, y_max])
                ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
                fig.canvas.draw_idle()

            elif event.key == 'c':
                # Cycle color map
                self.current_cmap = (self.current_cmap + 1) % len(self.colormaps)
                img.set_cmap(self.colormaps[self.current_cmap])
                print(f"\nColor map: {self.colormaps[self.current_cmap]}")
                fig.canvas.draw_idle()

            elif event.key == 'u':
                # Undo (go back in history)
                if len(view_history) > 1:
                    view_history.pop()  # Remove current
                    x_min, x_max, y_min, y_max, max_iter = view_history[-1]

                    print("\nUndoing zoom...")
                    fractal = self.renderer.render(width, height, x_min, x_max, y_min, y_max, max_iter)
                    color_data = np.log(fractal + 1)
                    img.set_data(color_data)
                    img.set_extent([x_min, x_max, y_min, y_max])
                    ax.set_title(f'Mandelbrot Set (max_iter={max_iter})')
                    fig.canvas.draw_idle()

            elif event.key == 'q':
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

    def compare_julia_sets(self, c_values, width=512, height=512):
        """
        Display multiple Julia sets side-by-side.

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
        plt.show()


# Example usage
if __name__ == "__main__":
    import sys
    import ttnn
    from renderer import MandelbrotRenderer

    # Check for save mode flag
    save_mode = '--save' in sys.argv or '--no-display' in sys.argv

    if save_mode:
        print("📁 Running in save mode (outputs will be saved to files)")
        print("💡 Tip: Use mandelbrot_explorer.ipynb for inline viewing in VSCode")
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import os
        output_dir = './mandelbrot_outputs'
        os.makedirs(output_dir, exist_ok=True)
    else:
        print("🖼️  Running in interactive mode (will try to open GUI windows)")
        print("💡 If no windows appear, try: python explorer.py --save")

    device = ttnn.open_device(device_id=0)
    renderer = MandelbrotRenderer(device)
    viz = MandelbrotVisualizer(renderer)

    if save_mode:
        # Render and save to file
        print("\n🎨 Rendering Mandelbrot set...")
        fractal = renderer.render(
            width=2048, height=2048,
            x_min=-2.5, x_max=1.0,
            y_min=-1.25, y_max=1.25,
            max_iter=512
        )

        # Save the plot
        fig, ax = plt.subplots(figsize=(12, 12))
        color_data = np.log(fractal + 1)
        im = ax.imshow(color_data, cmap='hot', origin='lower',
                       extent=[-2.5, 1.0, -1.25, 1.25])
        ax.set_title('Mandelbrot Set')
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')
        plt.colorbar(im, ax=ax, label='log(iterations)')
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'mandelbrot_classic.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved to: {output_path}")
        print(f"\n💡 For more features, try:")
        print(f"   - explorer_save.py (batch rendering)")
        print(f"   - mandelbrot_explorer.ipynb (interactive notebook)")
    else:
        # Launch interactive explorer
        viz.interactive_explorer(width=1024, height=1024)

    ttnn.close_device(device)
