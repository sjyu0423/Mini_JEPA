"""Measured-result visualizations for M3 experiments."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def save_bar_chart(
    values: dict[str, float], path: Path, *, title: str, ylabel: str = "RMSE (pixels)"
) -> None:
    width, height, left, bottom, top = 900, 520, 90, 120, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - 30, height - bottom), fill="black", width=2)
    maximum = max(values.values()) * 1.15 or 1.0
    slot = (width - left - 30) / len(values)
    for index, (label, value) in enumerate(values.items()):
        bar_width = slot * 0.62
        x0 = left + index * slot + (slot - bar_width) / 2
        x1 = x0 + bar_width
        y = height - bottom - value / maximum * (height - bottom - top)
        draw.rectangle((x0, y, x1, height - bottom), fill=COLORS[index % len(COLORS)])
        draw.text((x0, y - 20), f"{value:.2f}", fill="black")
        draw.text((x0, height - bottom + 12), label, fill="black")
    draw.text((left, 20), title, fill="black")
    draw.text((10, top), ylabel, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_training_curves(run_dirs: dict[str, Path], path: Path) -> None:
    histories = {
        name: json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        for name, run_dir in run_dirs.items()
    }
    width, height, margin = 900, 520, 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    values = [float(row["validation_loss"]) for history in histories.values() for row in history]
    low, high = min(values), max(values)
    span = max(high - low, 1e-8)
    for color, (_name, history) in zip(COLORS, histories.items(), strict=False):
        points = []
        for index, row in enumerate(history):
            x = margin + index / max(len(history) - 1, 1) * (width - 2 * margin)
            y = (
                height
                - margin
                - (float(row["validation_loss"]) - low) / span * (height - 2 * margin)
            )
            points.append((x, y))
        draw.line(points, fill=color, width=3)
    for index, (color, name) in enumerate(zip(COLORS, histories, strict=False)):
        draw.text((width - 280, 18 + 18 * index), name, fill=color)
    draw.text((margin, 20), "Stage-1 validation latent loss", fill="black")
    draw.text((margin, height - 40), "epoch", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_horizon_curve(values: dict[int, float], path: Path, *, title: str) -> None:
    width, height, margin = 700, 460, 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    horizons = sorted(values)
    maximum = max(values.values()) * 1.15 or 1.0
    points = []
    for index, horizon in enumerate(horizons):
        x = margin + index / max(len(horizons) - 1, 1) * (width - 2 * margin)
        y = height - margin - values[horizon] / maximum * (height - 2 * margin)
        points.append((x, y))
        draw.text((x - 8, height - margin + 12), str(horizon), fill="black")
        draw.text((x - 16, y - 22), f"{values[horizon]:.2f}", fill="black")
    draw.line(points, fill=COLORS[0], width=3)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS[0])
    draw.text((margin, 20), title, fill="black")
    draw.text((margin, height - 35), "prediction horizon", fill="black")
    draw.text((10, margin), "RMSE px", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
