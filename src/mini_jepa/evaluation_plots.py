"""Pillow-based plots for measured M2 results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def _bar_chart(
    labels: list[str], values: list[float], path: Path, *, title: str, ylabel: str
) -> None:
    width, height = 900, 520
    margin_left, margin_bottom, margin_top = 90, 110, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_height = height - margin_bottom - margin_top
    plot_width = width - margin_left - 30
    maximum = max(values) * 1.12 if max(values) > 0 else 1.0
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)
    draw.line(
        (margin_left, height - margin_bottom, width - 30, height - margin_bottom),
        fill="black",
        width=2,
    )
    slot = plot_width / len(values)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_width = slot * 0.62
        x0 = margin_left + index * slot + (slot - bar_width) / 2
        x1 = x0 + bar_width
        y = height - margin_bottom - value / maximum * plot_height
        draw.rectangle((x0, y, x1, height - margin_bottom), fill=COLORS[index % len(COLORS)])
        draw.text((x0, y - 20), f"{value:.2f}", fill="black")
        draw.text((x0, height - margin_bottom + 12), label, fill="black")
    draw.text((margin_left, 20), title, fill="black")
    draw.text((10, margin_top), ylabel, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_model_comparison(results: dict[str, Any], path: Path) -> None:
    order = ["constant_velocity", "wall_reflection", "supervised_cnn", "jepa_frozen_probe"]
    labels = ["constant", "wall-aware", "supervised", "JEPA+probe"]
    values = [results[name]["all"]["euclidean_rmse_pixels"] for name in order]
    _bar_chart(labels, values, path, title="Held-out test position error", ylabel="RMSE (pixels)")


def save_bounce_comparison(results: dict[str, Any], path: Path) -> None:
    order = ["constant_velocity", "wall_reflection", "supervised_cnn", "jepa_frozen_probe"]
    labels: list[str] = []
    values: list[float] = []
    for name in order:
        short = {
            "constant_velocity": "constant",
            "wall_reflection": "wall",
            "supervised_cnn": "supervised",
            "jepa_frozen_probe": "JEPA",
        }[name]
        labels.extend([f"{short}\nbounce", f"{short}\nnon"])
        values.extend(
            [
                results[name]["bounce"]["mean_error_pixels"],
                results[name]["non_bounce"]["mean_error_pixels"],
            ]
        )
    _bar_chart(
        labels,
        values,
        path,
        title="Bounce versus non-bounce mean position error",
        ylabel="Mean error (pixels)",
    )


def save_actual_vs_predicted(
    actual: NDArray[np.floating],
    predictions: dict[str, NDArray[np.floating]],
    path: Path,
    *,
    maximum_samples: int = 120,
) -> None:
    size, margin = 700, 60
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, size - margin, size - margin), outline="black", width=2)
    indices = np.linspace(0, len(actual) - 1, min(maximum_samples, len(actual)), dtype=int)

    def point(position: NDArray[np.floating]) -> tuple[float, float]:
        x = margin + float(position[0]) * (size - 2 * margin)
        y = size - margin - float(position[1]) * (size - 2 * margin)
        return x, y

    colors = {"supervised_cnn": "#16a34a", "jepa_frozen_probe": "#dc2626"}
    for index in indices:
        ax, ay = point(actual[index])
        draw.ellipse((ax - 2, ay - 2, ax + 2, ay + 2), fill="#111827")
        for name, values in predictions.items():
            px, py = point(values[index])
            draw.line((ax, ay, px, py), fill=colors[name], width=1)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=colors[name])
    draw.text(
        (margin, 20), "Actual (black) vs supervised (green) and JEPA+probe (red)", fill="black"
    )
    draw.text(
        (margin, size - 35),
        "Lines connect each prediction to its actual future position",
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_embedding_diagnostics(diagnostics: dict[str, Any], path: Path) -> None:
    width, height, margin = 900, 520, 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    branches = ("context", "prediction", "target")
    colors = ("#2563eb", "#dc2626", "#16a34a")
    all_values = [
        np.log10(np.asarray(diagnostics[branch]["per_dimension_variance"]) + 1e-10)
        for branch in branches
    ]
    low = min(float(values.min()) for values in all_values)
    high = max(float(values.max()) for values in all_values)
    span = max(high - low, 1e-8)
    for _branch, color, values in zip(branches, colors, all_values, strict=True):
        points = []
        for index, value in enumerate(values):
            x = margin + index / max(len(values) - 1, 1) * (width - 2 * margin)
            y = height - margin - (float(value) - low) / span * (height - 2 * margin)
            points.append((x, y))
        draw.line(points, fill=color, width=2)
    draw.text((margin, 20), "Per-dimension embedding variance (log10 scale)", fill="black")
    draw.text((width - 310, 20), "context (blue), prediction (red), target (green)", fill="black")
    draw.text((margin, height - 40), "embedding dimension 0-63", fill="black")
    draw.text((10, margin), f"log variance\n{high:.1f} to {low:.1f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_success_failure_examples(
    target_images: NDArray[np.uint8],
    actual: NDArray[np.floating],
    predicted: NDArray[np.floating],
    path: Path,
) -> None:
    errors = np.linalg.norm((predicted - actual) * 63.0, axis=1)
    selected = np.concatenate((np.argsort(errors)[:3], np.argsort(errors)[-3:]))
    panel_size, header = 160, 34
    canvas = Image.new("RGB", (3 * panel_size, 2 * (panel_size + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(selected):
        row, column = divmod(slot, 3)
        x0, y0 = column * panel_size, row * (panel_size + header)
        frame = Image.fromarray(target_images[index], mode="L").resize((panel_size, panel_size))
        canvas.paste(frame.convert("RGB"), (x0, y0 + header))
        actual_xy = actual[index] * (panel_size - 1)
        predicted_xy = predicted[index] * (panel_size - 1)
        ax, ay = x0 + actual_xy[0], y0 + header + actual_xy[1]
        px, py = x0 + predicted_xy[0], y0 + header + predicted_xy[1]
        draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), outline="#16a34a", width=3)
        draw.line((px - 6, py - 6, px + 6, py + 6), fill="#dc2626", width=3)
        draw.line((px - 6, py + 6, px + 6, py - 6), fill="#dc2626", width=3)
        kind = "success" if slot < 3 else "failure"
        draw.text((x0 + 5, y0 + 8), f"{kind}: {errors[index]:.2f}px", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
