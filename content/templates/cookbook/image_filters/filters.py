"""
Image filtering and effects using TTNN
"""

import ttnn
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def _ttnn_conv2d_single_kernel(device, image_chw, kernel_2d):
    """
    Apply a 2D convolution kernel to each RGB channel independently using TTNN.

    TTNN conv2d uses NHWC input format and requires explicit batch/spatial
    dimensions. This helper treats each color channel as a separate single-
    channel image in a batch (N=C, H, W, channels=1), which lets us use
    standard conv2d (groups=0) and avoids depthwise grouping constraints.

    Args:
        device:     TTNN device handle
        image_chw:  Float32 tensor of shape (C, H, W)
        kernel_2d:  Float32 tensor of shape (kH, kW)

    Returns:
        Float32 tensor of shape (C, H, W)
    """
    C, H, W = image_chw.shape
    kH, kW = kernel_2d.shape

    # -- Input: (C, H, W) → NHWC batch (C, H, W, 1) -------------------------
    # Each RGB channel becomes one single-channel image in a batch of C.
    img_batch = image_chw.unsqueeze(-1)          # (C, H, W, 1)  NHWC

    img_tt = ttnn.from_torch(
        img_batch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # -- Weight: (out_channels=1, in_channels=1, kH, kW) --------------------
    weight_4d = kernel_2d.unsqueeze(0).unsqueeze(0)   # (1, 1, kH, kW)

    weight_tt = ttnn.from_torch(
        weight_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # -- Zero bias: (1, 1, 1, out_channels=1) --------------------------------
    bias_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, 1, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # -- "same" padding: each side = (kernel_size - 1) // 2 -----------------
    pad_h = (kH - 1) // 2
    pad_w = (kW - 1) // 2

    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

    # -- TTNN conv2d ---------------------------------------------------------
    # batch_size=C: processes all C channels simultaneously as a batch.
    # Output format from TTNN: (1, 1, C*H*W, out_channels=1) — flat NHW.
    result_tt = ttnn.conv2d(
        input_tensor=img_tt,
        weight_tensor=weight_tt,
        bias_tensor=bias_tt,
        in_channels=1,
        out_channels=1,
        device=device,
        kernel_size=(kH, kW),
        stride=(1, 1),
        padding=(pad_h, pad_w),
        batch_size=C,
        input_height=H,
        input_width=W,
        conv_config=conv_config,
        groups=0,
    )

    # -- Reshape output back to (C, H, W) ------------------------------------
    # TTNN outputs (1, 1, C*H*W, 1); drop the singleton dims and unflatten.
    result_torch = ttnn.to_torch(result_tt)      # (1, 1, C*H*W, 1)
    result = result_torch.reshape(C, H, W).float()

    return result


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
        """Load image as float32 tensor in (C, H, W) format."""
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
        Apply named filter to image on Tenstorrent hardware via TTNN conv2d.

        TTNN conv2d requires NHWC input, explicit spatial dimensions, and
        keyword arguments. Each RGB channel is processed as a separate image
        in a batch so a single 2D kernel can be applied uniformly.

        Args:
            image: Float32 tensor (C, H, W)
            filter_name: Name from self.kernels

        Returns:
            Filtered image tensor (C, H, W)
        """
        if filter_name not in self.kernels:
            raise ValueError(f"Unknown filter: {filter_name}")

        kernel = self.kernels[filter_name]  # (kH, kW)

        return _ttnn_conv2d_single_kernel(self.device, image.float(), kernel.float())

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

        Simplifies colors while preserving edges via a box blur applied on
        Tenstorrent hardware, then multiplied by quantized intensity values.
        """
        # Quantize intensity for color simplification
        gray = image.mean(dim=0, keepdim=True)
        quantized = torch.floor(gray * intensity_levels) / intensity_levels

        # Variable-size box kernel for the blur approximation
        kernel_size = radius * 2 + 1
        box_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32) / (kernel_size ** 2)

        smoothed = _ttnn_conv2d_single_kernel(self.device, image.float(), box_kernel)

        # Blend smoothed result with quantized color map
        return smoothed * quantized.expand_as(smoothed)

    def custom_kernel(self, image, kernel_matrix):
        """
        Apply custom user-defined kernel on Tenstorrent hardware.

        Args:
            image: Input image tensor (C, H, W)
            kernel_matrix: 2D numpy array or float32 torch tensor (kH, kW)

        Returns:
            Filtered image (C, H, W)
        """
        if isinstance(kernel_matrix, np.ndarray):
            kernel_matrix = torch.from_numpy(kernel_matrix).float()

        return _ttnn_conv2d_single_kernel(self.device, image.float(), kernel_matrix.float())


# Example usage
if __name__ == "__main__":
    import sys

    device = ttnn.open_device(device_id=0, l1_small_size=8192)
    filters = ImageFilterBank(device)

    # Load image (use command line argument or default)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "examples/sample.jpg"

    try:
        image = filters.load_image(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        print("Usage: python filters.py <path-to-image>")
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

    axes[1].imshow(edge.permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title("Edge Detect")
    axes[1].axis('off')

    axes[2].imshow(blurred.permute(1, 2, 0).clamp(0, 1))
    axes[2].set_title("Gaussian Blur")
    axes[2].axis('off')

    axes[3].imshow(sharpened.permute(1, 2, 0).clamp(0, 1))
    axes[3].set_title("Sharpened")
    axes[3].axis('off')

    axes[4].imshow(embossed.permute(1, 2, 0).clamp(0, 1))
    axes[4].set_title("Embossed")
    axes[4].axis('off')

    axes[5].imshow(oil_paint.permute(1, 2, 0).clamp(0, 1))
    axes[5].set_title("Oil Painting")
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()

    ttnn.close_device(device)
