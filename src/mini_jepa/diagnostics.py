"""Representation diversity and covariance diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _normalize(embeddings: NDArray[np.floating]) -> NDArray[np.float64]:
    values = embeddings.astype(np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _branch_diagnostics(
    embeddings: NDArray[np.floating], *, seed: int, max_pairwise: int = 1024
) -> dict[str, Any]:
    values = embeddings.astype(np.float64)
    normalized = _normalize(values)
    per_dimension_variance = np.var(values, axis=0, ddof=0)
    normalized_variance = np.var(normalized, axis=0, ddof=0)
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(values)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    positive = np.clip(eigenvalues, 0.0, None)
    proportions = positive / np.clip(positive.sum(), 1e-12, None)
    nonzero = proportions[proportions > 0]
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    participation_ratio = float(positive.sum() ** 2 / np.clip(np.sum(positive**2), 1e-12, None))

    rng = np.random.default_rng(seed)
    count = min(len(values), max_pairwise)
    indices = rng.choice(len(values), size=count, replace=False)
    cosines = normalized[indices] @ normalized[indices].T
    off_diagonal = cosines[~np.eye(count, dtype=bool)]
    norms = np.linalg.norm(values, axis=1)
    return {
        "variance_definition": "population variance across samples, computed per coordinate",
        "mean_per_dimension_variance": float(per_dimension_variance.mean()),
        "min_per_dimension_variance": float(per_dimension_variance.min()),
        "max_per_dimension_variance": float(per_dimension_variance.max()),
        "per_dimension_variance": per_dimension_variance.tolist(),
        "mean_normalized_per_dimension_variance": float(normalized_variance.mean()),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std(ddof=0)),
        "different_sample_cosine_mean": float(off_diagonal.mean()),
        "different_sample_cosine_std": float(off_diagonal.std(ddof=0)),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "covariance_eigenvalues": eigenvalues.tolist(),
        "top_5_covariance_fraction": float(proportions[:5].sum()),
    }


def representation_diagnostics(
    context: NDArray[np.floating],
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    if not (len(context) == len(prediction) == len(target)):
        raise ValueError("embedding branches must contain the same samples")
    prediction_normalized = _normalize(prediction)
    target_normalized = _normalize(target)
    correct = np.sum(prediction_normalized * target_normalized, axis=1)
    offset = max(1, len(target) // 3)
    mismatched = np.sum(prediction_normalized * np.roll(target_normalized, offset, axis=0), axis=1)
    return {
        "sample_count": len(context),
        "context": _branch_diagnostics(context, seed=seed),
        "prediction": _branch_diagnostics(prediction, seed=seed + 1),
        "target": _branch_diagnostics(target, seed=seed + 2),
        "prediction_target_similarity": {
            "correct_mean": float(correct.mean()),
            "correct_std": float(correct.std(ddof=0)),
            "mismatched_mean": float(mismatched.mean()),
            "mismatched_std": float(mismatched.std(ddof=0)),
            "mean_margin": float((correct - mismatched).mean()),
            "fraction_correct_above_mismatch": float(np.mean(correct > mismatched)),
        },
    }
