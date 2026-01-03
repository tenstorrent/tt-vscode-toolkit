#!/usr/bin/env python3
"""Setup script for AnimateDiff TT-Metal integration."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="animatediff-ttnn",
    version="0.1.0",
    author="Tenstorrent Community",
    author_email="",
    description="AnimateDiff temporal attention for TT-Metal Stable Diffusion 3.5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/tt-animatediff",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        # tt-metal and ttnn must be installed separately (not on PyPI)
        # diffusers is optional (only needed for video export)
    ],
    extras_require={
        "video": [
            "diffusers>=0.21.0",  # For export_to_video function
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "animatediff-demo=examples.generate_2frame_video:generate_2frame_demo",
        ],
    },
)
