"""Frozen M3 embedding extraction and validation-selected probe evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from mini_jepa.diagnostics import representation_diagnostics
from mini_jepa.evaluation_data import EvaluationTrajectoryDataset
from mini_jepa.m3_model import M3JEPA
from mini_jepa.m3_training import load_m3_checkpoint
from mini_jepa.metrics import grouped_position_metrics, position_metrics
from mini_jepa.probes import StateProbe, probe_predict, save_probe, train_probe


@torch.no_grad()
def extract_m3_predictions(
    model: M3JEPA,
    dataset: EvaluationTrajectoryDataset,
    horizon: int,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> dict[str, NDArray[np.generic]]:
    if horizon not in model.config.horizons:
        raise ValueError(f"horizon {horizon} was not trained for this model")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collected: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "context",
            "prediction",
            "target",
            "future_position",
            "bounce",
            "trajectory_id",
        )
    }
    model.requires_grad_(False)
    model.eval()
    for batch in loader:
        observations = batch["observations"].to(device)
        context = model.context_encoder(observations)
        prediction = model.predict(context, (horizon,))[:, 0]
        target = model.target_encoder(batch["target_image"].to(device))
        collected["context"].append(context.cpu())
        collected["prediction"].append(prediction.cpu())
        collected["target"].append(target.cpu())
        for key in ("future_position", "bounce", "trajectory_id"):
            collected[key].append(batch[key].cpu())
    return {key: torch.cat(values).numpy() for key, values in collected.items()}


def fit_validation_probes(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    *,
    probe_epochs: int = 120,
    seed: int = 700,
) -> dict[str, Any]:
    model, checkpoint = load_m3_checkpoint(checkpoint_path, device)
    model.requires_grad_(False)
    model.eval()
    results: dict[str, Any] = {}
    for horizon in model.config.horizons:
        train_dataset = EvaluationTrajectoryDataset(
            data_dir / "train.npz", prediction_horizon=horizon
        )
        validation_dataset = EvaluationTrajectoryDataset(
            data_dir / "validation.npz", prediction_horizon=horizon
        )
        train = extract_m3_predictions(model, train_dataset, horizon, device)
        validation = extract_m3_predictions(model, validation_dataset, horizon, device)
        probe, selection = train_probe(
            train["prediction"].astype(np.float32),
            train["future_position"].astype(np.float32),
            validation["prediction"].astype(np.float32),
            validation["future_position"].astype(np.float32),
            seed=seed + horizon,
            epochs=probe_epochs,
        )
        validation_prediction = probe_predict(probe, validation["prediction"].astype(np.float32))
        metrics = position_metrics(
            validation_prediction, validation["future_position"].astype(np.float32)
        )
        diagnostics = representation_diagnostics(
            validation["context"],
            validation["prediction"],
            validation["target"],
            seed=seed + horizon,
        )
        probe_path = output_dir / f"probe-h{horizon}.pt"
        save_probe(probe_path, probe, {**selection, "horizon": horizon})
        results[str(horizon)] = {
            "selection": selection,
            "validation": metrics,
            "diagnostics": diagnostics,
            "probe_path": str(probe_path),
        }
    results["checkpoint_epoch"] = checkpoint["epoch"]
    (output_dir / "validation-probes.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def load_probe(path: Path, embedding_dim: int = 64) -> StateProbe:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    probe = StateProbe(
        embedding_dim,
        2,
        torch.zeros(embedding_dim),
        torch.ones(embedding_dim),
    )
    probe.load_state_dict(checkpoint["model"])
    probe.eval()
    return probe


def evaluate_probe(
    model: M3JEPA,
    probe: StateProbe,
    dataset_path: Path,
    horizon: int,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> tuple[dict[str, dict[str, float]], NDArray[np.float32], dict[str, NDArray[np.generic]]]:
    dataset = EvaluationTrajectoryDataset(dataset_path, prediction_horizon=horizon)
    extracted = extract_m3_predictions(model, dataset, horizon, device, batch_size=batch_size)
    predictions = probe_predict(probe, extracted["prediction"].astype(np.float32))
    metrics = grouped_position_metrics(
        predictions,
        extracted["future_position"].astype(np.float32),
        extracted["bounce"].astype(np.bool_),
    )
    return metrics, predictions, extracted
