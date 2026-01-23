# Custom Image Filters

Creative image processing with convolution kernels on TT hardware.

## Features

- Classic filters (edge detect, blur, sharpen, emboss)
- Artistic effects (oil painting)
- Custom kernel support
- Batch processing
- Real-time webcam (optional)

## Quick Start

```bash
pip install -r requirements.txt

# Process an image
python filters.py examples/sample.jpg
```

## Usage

```python
from filters import ImageFilterBank
import ttnn

device = ttnn.open_device(device_id=0)
filters = ImageFilterBank(device)

# Load image
image = filters.load_image("your_image.jpg")

# Apply filters
edge = filters.apply_filter(image, 'edge_detect')
blurred = filters.apply_filter(image, 'gaussian_blur')
sharpened = filters.apply_filter(image, 'sharpen')
embossed = filters.apply_filter(image, 'emboss')

# Artistic effect
oil_paint = filters.oil_painting_effect(image, radius=3)

# Save results
filters.save_image(edge, "edge_output.jpg")

ttnn.close_device(device)
```

## Available Filters

- **Edge Detection**: sobel_x, sobel_y, laplacian, edge_detect
- **Blur**: box_blur, gaussian_blur
- **Sharpen**: sharpen, unsharp_mask
- **Artistic**: emboss, oil_painting

## Complete Implementation

See **Lesson 12** for the full implementation including:
- ImageFilterBank class with 10+ filters
- Custom kernel support
- Real-time webcam processing
- Extensions (neural style transfer, seam carving, HDR tone mapping)

## Custom Kernels

```python
# Define your own kernel
custom = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

filtered = filters.custom_kernel(image, custom)
```