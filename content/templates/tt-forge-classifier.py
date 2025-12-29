#!/usr/bin/env python3
"""
TT-Forge Image Classifier

Uses MobileNetV2 (validated model) compiled with TT-Forge to classify images.

Requirements:
- tt-forge-venv activated
- TT hardware detected (tt-smi)
- forge, torch, torchvision, pillow installed

Usage:
    python tt-forge-classifier.py                    # Classify sample cat image
    python tt-forge-classifier.py --image photo.jpg  # Classify your own image

Note: First run compiles the model (2-5 min). Subsequent runs are faster.
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import torch
    import torchvision.models as models
    from torchvision import transforms
    from PIL import Image
    import requests
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install torch torchvision pillow requests")
    sys.exit(1)

try:
    import forge
except ImportError:
    print("❌ TT-Forge not found!")
    print("Activate environment: source ~/tt-forge-venv/bin/activate")
    print("Or install: pip install tt_forge_fe --extra-index-url https://pypi.eng.aws.tenstorrent.com/")
    sys.exit(1)


# ImageNet class labels (1000 classes)
# Full list: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def download_imagenet_classes():
    """Download ImageNet class labels."""
    print("Downloading ImageNet class labels...")
    try:
        response = requests.get(IMAGENET_CLASSES_URL, timeout=10)
        response.raise_for_status()
        classes = [line.strip() for line in response.text.splitlines()]
        print(f"✓ Loaded {len(classes)} class labels")
        return classes
    except Exception as e:
        print(f"⚠️  Failed to download class labels: {e}")
        print("Using fallback labels...")
        return [f"class_{i}" for i in range(1000)]


def download_sample_image():
    """Download sample cat image for testing."""
    sample_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    sample_path = Path("sample-image.jpg")

    if sample_path.exists():
        print(f"✓ Using existing sample image: {sample_path}")
        return str(sample_path)

    print("Downloading sample image...")
    try:
        response = requests.get(sample_url, timeout=10)
        response.raise_for_status()
        with open(sample_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Downloaded sample image: {sample_path}")
        return str(sample_path)
    except Exception as e:
        print(f"❌ Failed to download sample image: {e}")
        return None


def load_model():
    """
    Load pre-trained MobileNetV2 model.

    MobileNetV2 is a validated model in tt-forge-models.
    Lightweight (3.5M params) and known to compile successfully.
    """
    print("Loading MobileNetV2 model...")
    print("(This is a validated model from tt-forge-models)")

    try:
        # Load pre-trained model from torchvision
        model = models.mobilenet_v2(pretrained=True)
        model.eval()  # Set to inference mode (disable dropout, etc.)
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)


def compile_model_for_tt(model):
    """
    Compile PyTorch model for TT hardware using forge.compile().

    This may take 2-5 minutes on first run:
    - Graph analysis and operator validation
    - MLIR lowering to TTNN operations
    - Device kernel generation
    - Memory allocation

    Compilation can fail if unsupported operators are encountered.
    """
    print("\nCompiling model for TT hardware...")
    print("⏳ This may take 2-5 minutes on first run...")
    print("(Subsequent runs may be faster if caching works)")

    try:
        # Create sample input for shape inference
        # MobileNetV2 expects: [batch, channels, height, width]
        sample_input = torch.randn(1, 3, 224, 224)

        # Attempt compilation
        # This will fail if:
        # - Unsupported operators are present
        # - Memory constraints exceeded
        # - Device not detected
        compiled_model = forge.compile(model, sample_inputs=[sample_input])

        print("✓ Model compiled successfully!")
        print("  Model is now ready to run on TT hardware")
        return compiled_model

    except Exception as e:
        print(f"\n❌ Compilation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check device detected: tt-smi")
        print("2. Enable debug logging: export FORGE_LOG_LEVEL=DEBUG")
        print("3. Search issues: https://github.com/tenstorrent/tt-forge-fe/issues")
        print("4. Try Docker image if wheels have issues")
        sys.exit(1)


def preprocess_image(image_path):
    """
    Preprocess image for ImageNet model inference.

    Standard ImageNet preprocessing:
    - Resize to 256x256
    - Center crop to 224x224
    - Convert to tensor
    - Normalize with ImageNet stats
    """
    print(f"\nPreprocessing image: {image_path}")

    try:
        # Load image
        img = Image.open(image_path).convert('RGB')

        # ImageNet preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize(256),                    # Resize shorter side to 256
            transforms.CenterCrop(224),                # Crop center 224x224
            transforms.ToTensor(),                     # Convert to [0, 1] tensor
            transforms.Normalize(                      # Normalize with ImageNet stats
                mean=[0.485, 0.456, 0.406],           # RGB means
                std=[0.229, 0.224, 0.225]             # RGB stds
            ),
        ])

        # Apply preprocessing
        input_tensor = preprocess(img)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]

        print(f"✓ Image preprocessed: {input_tensor.shape}")
        return input_tensor

    except Exception as e:
        print(f"❌ Failed to preprocess image: {e}")
        sys.exit(1)


def classify_image(compiled_model, input_tensor, class_labels, top_k=5):
    """
    Run inference on TT hardware and return top-K predictions.

    Args:
        compiled_model: TT-Forge compiled model
        input_tensor: Preprocessed image tensor [1, 3, 224, 224]
        class_labels: List of ImageNet class names
        top_k: Number of top predictions to return
    """
    print("\nRunning inference on TT hardware...")

    try:
        # Run model on TT accelerator
        # This executes the compiled kernels on device
        with torch.no_grad():
            output = compiled_model(input_tensor)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top-K predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        print(f"\n{'='*50}")
        print(f"Top {top_k} Predictions:")
        print(f"{'='*50}")

        results = []
        for i in range(top_k):
            class_idx = top_indices[i].item()
            class_name = class_labels[class_idx] if class_idx < len(class_labels) else f"class_{class_idx}"
            confidence = top_probs[i].item() * 100

            print(f"{i+1}. {class_name:30s} {confidence:6.2f}%")
            results.append((class_name, confidence))

        print(f"{'='*50}\n")

        return results

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("\nPossible issues:")
        print("- Model not properly compiled")
        print("- Input tensor shape mismatch")
        print("- Device memory exhausted")
        print("- Device communication error")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Classify images using TT-Forge compiled MobileNetV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tt-forge-classifier.py
  python tt-forge-classifier.py --image ~/Pictures/dog.jpg
  python tt-forge-classifier.py --image food.jpg --top-k 10

Note: First run compiles the model (2-5 min). Subsequent runs are faster.
        """
    )
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file (default: download sample image)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show (default: 5)')

    args = parser.parse_args()

    print("="*60)
    print("TT-Forge Image Classifier")
    print("Model: MobileNetV2 (validated)")
    print("="*60)

    # Download class labels
    class_labels = download_imagenet_classes()

    # Get image path
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            sys.exit(1)
    else:
        print("\nNo image specified, downloading sample...")
        image_path = download_sample_image()
        if not image_path:
            print("❌ Could not get sample image. Please provide --image path")
            sys.exit(1)

    # Load model
    model = load_model()

    # Compile for TT hardware
    # This is the key step that makes the model run on TT accelerators
    compiled_model = compile_model_for_tt(model)

    # Preprocess image
    input_tensor = preprocess_image(image_path)

    # Run inference
    results = classify_image(compiled_model, input_tensor, class_labels, top_k=args.top_k)

    print("✓ Classification complete!")
    print("\nNext steps:")
    print("- Try your own images: --image ~/Pictures/photo.jpg")
    print("- Try validated models from tt-forge-models")
    print("- Check: https://github.com/tenstorrent/tt-forge-models")


if __name__ == '__main__':
    main()
