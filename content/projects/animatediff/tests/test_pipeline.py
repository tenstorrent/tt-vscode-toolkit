from unittest.mock import MagicMock
from PIL import Image
import pytest


def _make_mock_pipe(num_frames=16):
    """Return a mock AnimateDiffPipeline that returns num_frames PIL images."""
    pipe = MagicMock()
    frames = [Image.new("RGB", (512, 512), color=(i * 15, 0, 0)) for i in range(num_frames)]
    pipe.return_value.frames = [frames]
    return pipe


def test_generate_returns_list_of_pil_images():
    from animatediff_ttnn.pipeline import generate
    mock_pipe = _make_mock_pipe(16)
    result = generate(mock_pipe, "a campfire", num_frames=16)
    assert isinstance(result, list)
    assert len(result) == 16
    assert all(isinstance(f, Image.Image) for f in result)


def test_generate_passes_num_frames_to_pipe():
    from animatediff_ttnn.pipeline import generate
    mock_pipe = _make_mock_pipe(8)
    generate(mock_pipe, "test prompt", num_frames=8, guidance_scale=9.0,
             num_inference_steps=20, seed=77)
    kwargs = mock_pipe.call_args.kwargs
    assert kwargs["num_frames"] == 8
    assert kwargs["guidance_scale"] == 9.0
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["prompt"] == "test prompt"


def test_generate_single_frame():
    from animatediff_ttnn.pipeline import generate
    mock_pipe = _make_mock_pipe(1)
    result = generate(mock_pipe, "a still lake", num_frames=1)
    assert len(result) == 1


def test_export_gif_creates_file(tmp_path):
    from animatediff_ttnn.pipeline import export_gif
    frames = [Image.new("RGB", (64, 64), color=(i * 40, 0, 0)) for i in range(4)]
    output = str(tmp_path / "test.gif")
    export_gif(frames, output, fps=4)
    assert (tmp_path / "test.gif").exists()
    assert (tmp_path / "test.gif").stat().st_size > 0


def test_export_gif_creates_parent_directories(tmp_path):
    from animatediff_ttnn.pipeline import export_gif
    frames = [Image.new("RGB", (32, 32)) for _ in range(2)]
    nested = str(tmp_path / "a" / "b" / "out.gif")
    export_gif(frames, nested)
    import pathlib
    assert pathlib.Path(nested).exists()
