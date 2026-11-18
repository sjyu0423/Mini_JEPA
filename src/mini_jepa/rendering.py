"""Rendering utilities kept separate from physical state evolution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


def render_frame(
    position: Sequence[float],
    *,
    ball_radius: float,
    arena_size: tuple[float, float] = (1.0, 1.0),
    image_size: tuple[int, int] = (64, 64),
) -> NDArray[np.uint8]:
    """Render a physical position as a grayscale uint8 image."""
    width_px, height_px = image_size
    width, height = arena_size
    x, y = position
    center_x = x / width * (width_px - 1)
    center_y = y / height * (height_px - 1)
    radius_x = ball_radius / width * width_px
    radius_y = ball_radius / height * height_px
    image = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (
            center_x - radius_x,
            center_y - radius_y,
            center_x + radius_x,
            center_y + radius_y,
        ),
        fill=255,
    )
    return np.asarray(image, dtype=np.uint8)


def render_trajectory(
    positions: NDArray[np.floating],
    *,
    ball_radius: float,
    arena_size: tuple[float, float] = (1.0, 1.0),
    image_size: tuple[int, int] = (64, 64),
) -> NDArray[np.uint8]:
    return np.stack(
        [
            render_frame(
                position,
                ball_radius=ball_radius,
                arena_size=arena_size,
                image_size=image_size,
            )
            for position in positions
        ]
    )


def save_gif(frames: NDArray[np.uint8], path: Path, *, duration_ms: int = 80) -> None:
    """Write frames to an animated grayscale GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame, mode="L") for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def save_montage(
    frames: NDArray[np.uint8],
    collision_counts: NDArray[np.integer],
    path: Path,
    *,
    columns: int = 8,
) -> None:
    """Save a montage with collision frames outlined in white."""
    rows = int(np.ceil(len(frames) / columns))
    frame_height, frame_width = frames.shape[1:]
    canvas = Image.new("L", (columns * frame_width, rows * frame_height), color=24)
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame, mode="L")
        if np.any(collision_counts[index] > 0):
            ImageDraw.Draw(image).rectangle(
                (0, 0, frame_width - 1, frame_height - 1), outline=160, width=2
            )
        x = index % columns * frame_width
        y = index // columns * frame_height
        canvas.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
