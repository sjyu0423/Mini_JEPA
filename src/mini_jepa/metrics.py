"""Position-error metrics with trajectory-level uncertainty estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def position_metrics(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    pixels_per_unit: float = 63.0,
) -> dict[str, float]:
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions and targets must both have shape (samples, 2)")
    error = (predictions - targets) * pixels_per_unit
    distances = np.linalg.norm(error, axis=1)
    return {
        "mae_pixels": float(np.mean(np.abs(error))),
        "euclidean_rmse_pixels": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "median_error_pixels": float(np.median(distances)),
        "p90_error_pixels": float(np.percentile(distances, 90)),
        "mean_error_pixels": float(np.mean(distances)),
        "sample_count": float(len(targets)),
    }


def grouped_position_metrics(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    bounce: NDArray[np.bool_],
) -> dict[str, dict[str, float]]:
    if len(bounce) != len(targets):
        raise ValueError("bounce mask must align with targets")
    return {
        "all": position_metrics(predictions, targets),
        "bounce": position_metrics(predictions[bounce], targets[bounce]),
        "non_bounce": position_metrics(predictions[~bounce], targets[~bounce]),
    }


def trajectory_bootstrap_mean_error(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    trajectory_ids: NDArray[np.integer],
    *,
    seed: int = 0,
    replicates: int = 1000,
) -> dict[str, float]:
    """Bootstrap the mean Euclidean pixel error over complete trajectories."""
    errors = np.linalg.norm((predictions - targets) * 63.0, axis=1)
    unique_ids = np.unique(trajectory_ids)
    per_trajectory = np.array(
        [errors[trajectory_ids == trajectory_id].mean() for trajectory_id in unique_ids]
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(per_trajectory, size=(replicates, len(per_trajectory)), replace=True).mean(
        axis=1
    )
    return {
        "trajectory_mean_error_pixels": float(per_trajectory.mean()),
        "bootstrap_95_low": float(np.percentile(samples, 2.5)),
        "bootstrap_95_high": float(np.percentile(samples, 97.5)),
        "trajectory_count": float(len(unique_ids)),
    }


def add_uncertainty(
    grouped: dict[str, dict[str, float]],
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    trajectory_ids: NDArray[np.integer],
    *,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(grouped)
    result["trajectory_bootstrap"] = trajectory_bootstrap_mean_error(
        predictions, targets, trajectory_ids, seed=seed
    )
    return result
