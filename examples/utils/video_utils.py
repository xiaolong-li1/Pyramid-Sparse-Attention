"""Video saving and output directory utilities."""

import os
from pathlib import Path
from datetime import datetime
from typing import List


def create_output_dir(base_dir: str, model_name: str, use_timestamp: bool = True) -> Path:
    """Create output directory with optional timestamp.

    Args:
        base_dir: Base output directory (e.g., "outputs")
        model_name: Model name (e.g., "cogvideo_5b")
        use_timestamp: Whether to add timestamp subdirectory

    Returns:
        Path to output directory
    """
    output_path = Path(base_dir) / model_name

    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / timestamp

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_video(
    frames: List,
    output_path: str,
    fps: int = 8,
    verbose: bool = True
) -> str:
    """Save video frames to file.

    Args:
        frames: List of video frames
        output_path: Output file path
        fps: Frames per second
        verbose: Whether to print save confirmation

    Returns:
        Path to saved video file
    """
    # Lazy import to avoid dependency issues
    from diffusers.utils import export_to_video

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save video
    export_to_video(frames, output_path, fps=fps)

    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ Video saved to: {output_path} ({file_size:.2f} MB)")

    return output_path
