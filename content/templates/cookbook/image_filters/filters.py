"""
Image filtering and effects using TTNN
"""

import ttnn
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageFilterBank:
    def __init__(self, device):
        """
        Initialize filter bank.

        Args:
            device: TTNN device handle
        """
        self.device = device

        # Predefined kernels
        self.kernels = self._define_kernels()

    def _define_kernels(self):
        """Define common convolution kernels."""
        kernels = {
            # Edge Detection
            'sobel_x': torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32),

            'sobel_y': torch.tensor([[-1, -2, -1],
                                     [0,  0,  0],
                                     [1,  2,  1]], dtype=torch.float32),

            'laplacian': torch.tensor([[0,  1, 0],
                                      [1, -4, 1],
                                      [0,  1, 0]], dtype=torch.float32),

            'edge_detect': torch.tensor([[-1, -1, -1],
                                        [-1,  8, -1],
                                        [-1, -1, -1]], dtype=torch.float32),

            # Blur
            'box_blur': torch.ones((5, 5), dtype=torch.float32) / 25,

            'gaussian_blur': torch.tensor([[1,  4,  6,  4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, 36, 24, 6],
                                          [4, 16, 24, 16, 4],
                                          [1,  4,  6,  4, 1]], dtype=torch.float32) / 256,

            # Sharpen
            'sharpen': torch.tensor([[0, -1,  0],
                                    [-1,  5, -1],
                                    [0, -1,  0]], dtype=torch.float32),

            'unsharp_mask': torch.tensor([[-1, -1, -1],
                                         [-1,  9, -1],
                                         [-1, -1, -1]], dtype=torch.float32),

            # Emboss
            'emboss': torch.tensor([[-2, -1,  0],
                                   [-1,  1,  1],
                                   [0,  1,  2]], dtype=torch.float32),

            # Motion blur
            'motion_blur_h': torch.zeros((5, 5), dtype=torch.float32),
            'motion_blur_v': torch.zeros((5, 5), dtype=torch.float32),
        }

        # Initialize motion blur kernels
        kernels['motion_blur_h'][2, :] = 1.0 / 5.0
        kernels['motion_blur_v'][:, 2] = 1.0 / 5.0

        return kernels

    def load_image(self, path):
        """Load image as tensor."""
        img = Image.open(path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        return torch.from_numpy(img_array).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    def save_image(self, tensor, path):
        """Save tensor as image."""
        # Clamp to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        # (C, H, W) -> (H, W, C)
        img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_array).save(path)
        print(f"Saved to {path}")

    def apply_filter(self, image, filter_name):
        """
        Apply named filter to image.

        Args:
            image: Tensor (C, H, W)
            filter_name: Name from self.kernels

        Returns:
            Filtered image tensor
        """
        if filter_name not in self.kernels:
            raise ValueError(f"Unknown filter: {filter_name}")

        kernel = self.kernels[filter_name]

        # Move to device
        img_tt = ttnn.from_torch(image.unsqueeze(0), device=self.device, layout=ttnn.TILE_LAYOUT)
        kernel_tt = ttnn.from_torch(kernel.unsqueeze(0).unsqueeze(0), device=self.device, layout=ttnn.TILE_LAYOUT)

        # Apply to each channel
        filtered_channels = []
        for c in range(image.shape[0]):
            channel = img_tt[0, c:c+1, :, :]
            filtered = ttnn.conv2d(channel, kernel_tt, padding='same')
            filtered_channels.append(filtered)

        # Stack channels
        result = ttnn.concat(filtered_channels, dim=1)

        return ttnn.to_torch(result).squeeze(0).cpu()

    def edge_detect_combined(self, image):
        """Sobel edge detection (combines X and Y gradients)."""
        sobel_x = self.apply_filter(image, 'sobel_x')
        sobel_y = self.apply_filter(image, 'sobel_y')

        # Magnitude: sqrt(Gx² + Gy²)
        magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)

        return magnitude

    def oil_painting_effect(self, image, radius=5, intensity_levels=20):
        """
        Kuwahara filter approximation for oil painting effect.

        Simplifies colors while preserving edges.
        """
        # Convert to grayscale for intensity
        gray = image.mean(dim=0, keepdim=True)

        # Quantize intensity
        quantized = torch.floor(gray * intensity_levels) / intensity_levels

        # Apply bilateral-style smoothing
        # (Simplified version - full Kuwahara is more complex)
        img_tt = ttnn.from_torch(image.unsqueeze(0), device=self.device)

        # Box blur as approximation
        kernel_size = radius * 2 + 1
        box_kernel = torch.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        kernel_tt = ttnn.from_torch(box_kernel.unsqueeze(0).unsqueeze(0), device=self.device)

        smoothed_channels = []
        for c in range(3):
            channel = img_tt[0, c:c+1, :, :]
            smoothed = ttnn.conv2d(channel, kernel_tt, padding='same')
            smoothed_channels.append(smoothed)

        result = ttnn.concat(smoothed_channels, dim=1)
        result_torch = ttnn.to_torch(result).squeeze(0).cpu()

        # Combine with quantized colors
        return result_torch * quantized.expand_as(result_torch)

    def custom_kernel(self, image, kernel_matrix):
        """
        Apply custom user-defined kernel.

        Args:
            image: Input image tensor
            kernel_matrix: 2D numpy array or torch tensor

        Returns:
            Filtered image
        """
        if isinstance(kernel_matrix, np.ndarray):
            kernel_matrix = torch.from_numpy(kernel_matrix).float()

        kernel_tt = ttnn.from_torch(
            kernel_matrix.unsqueeze(0).unsqueeze(0),
            device=self.device
        )

        img_tt = ttnn.from_torch(image.unsqueeze(0), device=self.device)

        filtered_channels = []
        for c in range(image.shape[0]):
            channel = img_tt[0, c:c+1, :, :]
            filtered = ttnn.conv2d(channel, kernel_tt, padding='same')
            filtered_channels.append(filtered)

        result = ttnn.concat(filtered_channels, dim=1)
        return ttnn.to_torch(result).squeeze(0).cpu()


# Example usage
if __name__ == "__main__":
    import ttnn
    import sys

    device = ttnn.open_device(device_id=0)
    filters = ImageFilterBank(device)

    # Load image (use command line argument or default)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "examples/sample.jpg"

    try:
        image = filters.load_image(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        print("Usage: python filters.py <path-to-image>")
        print("Note: You may need to provide your own sample image")
        ttnn.close_device(device)
        sys.exit(1)

    # Apply filters
    print("Applying filters...")
    edge = filters.apply_filter(image, 'edge_detect')
    blurred = filters.apply_filter(image, 'gaussian_blur')
    sharpened = filters.apply_filter(image, 'sharpen')
    embossed = filters.apply_filter(image, 'emboss')

    print("Applying oil painting effect...")
    oil_paint = filters.oil_painting_effect(image, radius=3)

    # Visualize
    print("Displaying results...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(edge.permute(1, 2, 0))
    axes[1].set_title("Edge Detect")
    axes[1].axis('off')

    axes[2].imshow(blurred.permute(1, 2, 0))
    axes[2].set_title("Gaussian Blur")
    axes[2].axis('off')

    axes[3].imshow(sharpened.permute(1, 2, 0))
    axes[3].set_title("Sharpened")
    axes[3].axis('off')

    axes[4].imshow(embossed.permute(1, 2, 0))
    axes[4].set_title("Embossed")
    axes[4].axis('off')

    axes[5].imshow(oil_paint.permute(1, 2, 0))
    axes[5].set_title("Oil Painting")
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()

    ttnn.close_device(device)
