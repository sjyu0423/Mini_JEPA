"""End-to-end scientific evaluation for the frozen M1 Mini-JEPA."""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mini_jepa.baselines import (
    SupervisedPositionPredictor,
    constant_velocity_predict,
    parameter_count,
    wall_reflection_predict,
)
from mini_jepa.diagnostics import representation_diagnostics
from mini_jepa.evaluation_data import EvaluationTrajectoryDataset
from mini_jepa.evaluation_plots import (
    save_actual_vs_predicted,
    save_bounce_comparison,
    save_embedding_diagnostics,
    save_model_comparison,
    save_success_failure_examples,
)
from mini_jepa.metrics import add_uncertainty, grouped_position_metrics, position_metrics
from mini_jepa.model import MiniJEPA, ModelConfig
from mini_jepa.probes import (
    EmbeddingBundle,
    extract_embeddings,
    load_frozen_jepa,
    probe_predict,
    save_probe,
    train_probe,
)
from mini_jepa.training import seed_everything, select_device


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    checkpoint_path: Path
    data_dir: Path
    ood_test_path: Path
    output_dir: Path = Path("artifacts/m2")
    observed_frames: int = 4
    prediction_horizon: int = 1
    batch_size: int = 128
    probe_epochs: int = 120
    supervised_epochs: int = 20
    seed: int = 303
    device: str = "auto"


def _train_supervised(
    train_dataset: EvaluationTrajectoryDataset,
    validation_dataset: EvaluationTrajectoryDataset,
    config: EvaluationConfig,
    device: torch.device,
) -> tuple[SupervisedPositionPredictor, dict[str, float | int]]:
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    model = SupervisedPositionPredictor(config.observed_frames).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, config.supervised_epochs + 1):
        model.train()
        for batch in train_loader:
            observations = batch["observations"].to(device)
            targets = batch["future_position"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(observations), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        total, samples = 0.0, 0
        with torch.no_grad():
            for batch in validation_loader:
                observations = batch["observations"].to(device)
                targets = batch["future_position"].to(device)
                loss = nn.functional.mse_loss(model(observations), targets, reduction="sum")
                total += float(loss)
                samples += len(observations)
        validation_mse = total / (samples * 2)
        if validation_mse < best_loss:
            best_loss = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("supervised training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_validation_mse": best_loss}


@torch.no_grad()
def _supervised_predict(
    model: SupervisedPositionPredictor,
    dataset: EvaluationTrajectoryDataset,
    device: torch.device,
    batch_size: int,
) -> NDArray[np.float32]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = [model(batch["observations"].to(device)).cpu() for batch in loader]
    return torch.cat(predictions).numpy().astype(np.float32)


def _evaluate_predictions(
    predictions: NDArray[np.float32], bundle: EmbeddingBundle, *, seed: int
) -> dict[str, Any]:
    grouped = grouped_position_metrics(predictions, bundle.future_position, bundle.bounce)
    return add_uncertainty(
        grouped, predictions, bundle.future_position, bundle.trajectory_ids, seed=seed
    )


def _classical_predictions(bundle: EmbeddingBundle, horizon: int) -> dict[str, NDArray[np.float32]]:
    observed = torch.from_numpy(bundle.observed_positions)
    return {
        "constant_velocity": constant_velocity_predict(observed, horizon).numpy(),
        "wall_reflection": wall_reflection_predict(observed, horizon).numpy(),
    }


def _save_results_csv(results: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "rmse_pixels",
                "mae_pixels",
                "median_pixels",
                "p90_pixels",
                "bounce_mean_pixels",
                "non_bounce_mean_pixels",
            ]
        )
        for name, metrics in results.items():
            writer.writerow(
                [
                    name,
                    metrics["all"]["euclidean_rmse_pixels"],
                    metrics["all"]["mae_pixels"],
                    metrics["all"]["median_error_pixels"],
                    metrics["all"]["p90_error_pixels"],
                    metrics["bounce"]["mean_error_pixels"],
                    metrics["non_bounce"]["mean_error_pixels"],
                ]
            )


def evaluate(config: EvaluationConfig) -> dict[str, Any]:
    if config.prediction_horizon != 1:
        raise ValueError("the available M1 checkpoint was trained only for horizon 1")
    started = time.perf_counter()
    seed_everything(config.seed)
    device = select_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = EvaluationTrajectoryDataset(
        config.data_dir / "train.npz",
        observed_frames=config.observed_frames,
        prediction_horizon=config.prediction_horizon,
    )
    validation_dataset = EvaluationTrajectoryDataset(
        config.data_dir / "validation.npz",
        observed_frames=config.observed_frames,
        prediction_horizon=config.prediction_horizon,
    )
    model = load_frozen_jepa(config.checkpoint_path, device)
    train_bundle = extract_embeddings(model, train_dataset, device, batch_size=config.batch_size)
    validation_bundle = extract_embeddings(
        model, validation_dataset, device, batch_size=config.batch_size
    )
    diagnostics = representation_diagnostics(
        validation_bundle.context,
        validation_bundle.prediction,
        validation_bundle.target,
        seed=config.seed,
    )

    current_targets = np.concatenate(
        (train_bundle.current_position, train_bundle.current_velocity), axis=1
    ).astype(np.float32)
    current_validation_targets = np.concatenate(
        (validation_bundle.current_position, validation_bundle.current_velocity), axis=1
    ).astype(np.float32)
    current_probe, current_info = train_probe(
        train_bundle.context,
        current_targets,
        validation_bundle.context,
        current_validation_targets,
        seed=config.seed,
        epochs=config.probe_epochs,
    )
    future_probe, future_info = train_probe(
        train_bundle.prediction,
        train_bundle.future_position,
        validation_bundle.prediction,
        validation_bundle.future_position,
        seed=config.seed + 1,
        epochs=config.probe_epochs,
    )
    target_probe, target_info = train_probe(
        train_bundle.target,
        train_bundle.future_position,
        validation_bundle.target,
        validation_bundle.future_position,
        seed=config.seed + 2,
        epochs=config.probe_epochs,
    )

    torch.manual_seed(config.seed + 100)
    random_model = MiniJEPA(ModelConfig()).to(device)
    random_model.requires_grad_(False)
    random_model.eval()
    random_train = extract_embeddings(
        random_model, train_dataset, device, batch_size=config.batch_size
    )
    random_validation = extract_embeddings(
        random_model, validation_dataset, device, batch_size=config.batch_size
    )
    random_probe, random_info = train_probe(
        random_train.prediction,
        random_train.future_position,
        random_validation.prediction,
        random_validation.future_position,
        seed=config.seed + 3,
        epochs=config.probe_epochs,
    )

    supervised, supervised_info = _train_supervised(
        train_dataset, validation_dataset, config, device
    )

    save_probe(config.output_dir / "current-state-probe.pt", current_probe, current_info)
    save_probe(config.output_dir / "future-state-probe.pt", future_probe, future_info)
    save_probe(config.output_dir / "target-embedding-probe.pt", target_probe, target_info)
    save_probe(config.output_dir / "random-encoder-probe.pt", random_probe, random_info)
    torch.save(
        {
            "model": supervised.state_dict(),
            "metadata": supervised_info,
            "parameter_count": parameter_count(supervised),
        },
        config.output_dir / "supervised-cnn.pt",
    )

    # The held-out test split is first accessed only after all models and epochs are selected.
    test_dataset = EvaluationTrajectoryDataset(
        config.data_dir / "test.npz",
        observed_frames=config.observed_frames,
        prediction_horizon=config.prediction_horizon,
    )
    test_bundle = extract_embeddings(model, test_dataset, device, batch_size=config.batch_size)
    random_test = extract_embeddings(
        random_model, test_dataset, device, batch_size=config.batch_size
    )
    predictions = _classical_predictions(test_bundle, config.prediction_horizon)
    predictions.update(
        {
            "supervised_cnn": _supervised_predict(
                supervised, test_dataset, device, config.batch_size
            ),
            "jepa_frozen_probe": probe_predict(future_probe, test_bundle.prediction),
            "jepa_target_probe": probe_predict(target_probe, test_bundle.target),
            "random_frozen_probe": probe_predict(random_probe, random_test.prediction),
        }
    )
    test_results = {
        name: _evaluate_predictions(values, test_bundle, seed=config.seed + index)
        for index, (name, values) in enumerate(predictions.items())
    }
    current_prediction = probe_predict(current_probe, test_bundle.context)
    current_probe_result = position_metrics(current_prediction[:, :2], test_bundle.current_position)
    velocity_error = current_prediction[:, 2:] - test_bundle.current_velocity
    current_probe_result["velocity_vector_rmse"] = float(
        np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))
    )

    ood_dataset = EvaluationTrajectoryDataset(
        config.ood_test_path,
        observed_frames=config.observed_frames,
        prediction_horizon=config.prediction_horizon,
    )
    ood_bundle = extract_embeddings(model, ood_dataset, device, batch_size=config.batch_size)
    random_ood = extract_embeddings(random_model, ood_dataset, device, batch_size=config.batch_size)
    ood_predictions = _classical_predictions(ood_bundle, config.prediction_horizon)
    ood_predictions.update(
        {
            "supervised_cnn": _supervised_predict(
                supervised, ood_dataset, device, config.batch_size
            ),
            "jepa_frozen_probe": probe_predict(future_probe, ood_bundle.prediction),
            "random_frozen_probe": probe_predict(random_probe, random_ood.prediction),
        }
    )
    ood_results = {
        name: _evaluate_predictions(values, ood_bundle, seed=config.seed + 50 + index)
        for index, (name, values) in enumerate(ood_predictions.items())
    }

    with np.load(config.data_dir / "test.npz", allow_pickle=False) as raw_test:
        target_images = np.stack(
            [
                raw_test["frames"][
                    trajectory,
                    start + config.observed_frames + config.prediction_horizon - 1,
                ]
                for trajectory, start in test_dataset.index
            ]
        )
    np.savez_compressed(
        config.output_dir / "test-predictions.npz",
        actual=test_bundle.future_position,
        bounce=test_bundle.bounce,
        trajectory_ids=test_bundle.trajectory_ids,
        **predictions,
    )
    save_model_comparison(test_results, config.output_dir / "model-comparison.png")
    save_bounce_comparison(test_results, config.output_dir / "bounce-comparison.png")
    save_actual_vs_predicted(
        test_bundle.future_position,
        {
            "supervised_cnn": predictions["supervised_cnn"],
            "jepa_frozen_probe": predictions["jepa_frozen_probe"],
        },
        config.output_dir / "actual-vs-predicted.png",
    )
    save_embedding_diagnostics(diagnostics, config.output_dir / "embedding-diagnostics.png")
    save_success_failure_examples(
        target_images,
        test_bundle.future_position,
        predictions["jepa_frozen_probe"],
        config.output_dir / "jepa-success-failure.png",
    )

    summary = {
        "device": str(device),
        "duration_seconds": time.perf_counter() - started,
        "dataset": {
            "train_windows": len(train_dataset),
            "validation_windows": len(validation_dataset),
            "test_windows": len(test_dataset),
            "ood_windows": len(ood_dataset),
            "test_trajectories": len(np.unique(test_bundle.trajectory_ids)),
            "ood_trajectories": len(np.unique(ood_bundle.trajectory_ids)),
            "test_bounce_windows": int(test_bundle.bounce.sum()),
            "ood_bounce_windows": int(ood_bundle.bounce.sum()),
        },
        "supervised_parameter_count": parameter_count(supervised),
        "selection": {
            "current_probe": current_info,
            "future_probe": future_info,
            "target_probe": target_info,
            "random_probe": random_info,
            "supervised": supervised_info,
        },
        "current_state_probe": current_probe_result,
        "test": test_results,
        "ood": ood_results,
        "diagnostics": diagnostics,
    }
    (config.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _save_results_csv(test_results, config.output_dir / "test-results.csv")
    return summary
