"""Training and checkpoint utilities for controlled M3 experiments."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mini_jepa.m3_config import ExperimentConfig
from mini_jepa.m3_data import MultiHorizonSample, MultiHorizonTrajectoryDataset
from mini_jepa.m3_model import M3JEPA, M3ModelConfig, m3_loss, trainable_parameter_count
from mini_jepa.training import seed_everything, select_device


def _run_epoch(
    model: M3JEPA,
    loader: DataLoader[MultiHorizonSample],
    config: ExperimentConfig,
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        targets = batch["targets"].to(device)
        if optimizer is None:
            with torch.no_grad():
                outputs = model(observations, targets)
                loss, diagnostics = m3_loss(
                    outputs,
                    objective=config.objective,
                    variance_weight=config.variance_weight,
                    min_std=config.min_std,
                )
        else:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(observations, targets)
            loss, diagnostics = m3_loss(
                outputs,
                objective=config.objective,
                variance_weight=config.variance_weight,
                min_std=config.min_std,
            )
            loss.backward()
            optimizer.step()
            model.update_target(config.ema_decay)
        metrics = {
            "loss": float(loss.detach()),
            **{key: float(value) for key, value in diagnostics.items()},
        }
        batch_size = len(observations)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        samples += batch_size
    return {key: value / samples for key, value in totals.items()}


def save_m3_checkpoint(
    path: Path,
    model: M3JEPA,
    optimizer: AdamW,
    config: ExperimentConfig,
    *,
    epoch: int,
    best_validation_loss: float,
    history: list[dict[str, float | int]],
) -> None:
    torch.save(
        {
            "format": "mini-jepa-m3-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model.config),
            "experiment_config": config.to_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "history": history,
        },
        path,
    )


def load_m3_checkpoint(path: Path, device: torch.device) -> tuple[M3JEPA, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "mini-jepa-m3-v1":
        raise ValueError("not an M3 checkpoint")
    model = M3JEPA(M3ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model._freeze_target()
    return model, checkpoint


def train_experiment(
    config: ExperimentConfig,
    *,
    data_dir: Path,
    output_dir: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    seed_everything(config.training_seed)
    device = select_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")
    train_dataset = MultiHorizonTrajectoryDataset(
        data_dir / "train.npz",
        observed_frames=config.observed_frames,
        horizons=config.horizons,
    )
    validation_dataset = MultiHorizonTrajectoryDataset(
        data_dir / "validation.npz",
        observed_frames=config.observed_frames,
        horizons=config.horizons,
    )
    generator = torch.Generator().manual_seed(config.training_seed)
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
    model = M3JEPA(
        M3ModelConfig(
            architecture=config.architecture,
            observed_frames=config.observed_frames,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            horizons=config.horizons,
        )
    ).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_optimizer: dict[str, Any] | None = None
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, config, device, optimizer)
        validation_metrics = _run_epoch(model, validation_loader, config, device, None)
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        print(
            f"{config.name} seed={config.training_seed} epoch={epoch} "
            f"train={train_metrics['loss']:.5f} validation={validation_metrics['loss']:.5f}"
        )

    if best_state is None or best_optimizer is None:
        raise RuntimeError("experiment produced no checkpoint")
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer)
    save_m3_checkpoint(
        output_dir / "best.pt",
        model,
        optimizer,
        config,
        epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        history=history,
    )
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "name": config.name,
        "training_seed": config.training_seed,
        "parameter_count": trainable_parameter_count(model),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (output_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
