"""Training, validation, checkpointing, and metric visualization for Mini-JEPA."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mini_jepa.model import MiniJEPA, ModelConfig, jepa_loss
from mini_jepa.torch_dataset import JEPASample, TorchTrajectoryDataset


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_path: Path
    validation_path: Path
    output_dir: Path = Path("artifacts/m1")
    observed_frames: int = 4
    prediction_horizon: int = 1
    embedding_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.99
    variance_weight: float = 0.1
    min_std: float = 0.1
    seed: int = 17
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def _move_batch(batch: JEPASample, device: torch.device) -> tuple[Tensor, Tensor]:
    return batch["observations"].to(device), batch["target"].to(device)


def train_step(
    model: MiniJEPA,
    batch: JEPASample,
    optimizer: AdamW,
    device: torch.device,
    *,
    ema_decay: float,
    variance_weight: float = 0.1,
    min_std: float = 0.1,
) -> dict[str, float]:
    model.train()
    observations, target = _move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(observations, target)
    loss, diagnostics = jepa_loss(outputs, variance_weight=variance_weight, min_std=min_std)
    loss.backward()
    optimizer.step()
    model.update_target(ema_decay)
    return {
        "loss": float(loss.detach()),
        **{key: float(value) for key, value in diagnostics.items()},
    }


def _run_epoch(
    model: MiniJEPA,
    loader: DataLoader[JEPASample],
    device: torch.device,
    *,
    optimizer: AdamW | None,
    config: TrainingConfig,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    samples = 0
    model.train(optimizer is not None)
    for batch in loader:
        batch_size = len(batch["observations"])
        if optimizer is None:
            with torch.no_grad():
                observations, target = _move_batch(batch, device)
                outputs = model(observations, target)
                loss, diagnostics = jepa_loss(
                    outputs,
                    variance_weight=config.variance_weight,
                    min_std=config.min_std,
                )
                metrics = {
                    "loss": float(loss),
                    **{key: float(value) for key, value in diagnostics.items()},
                }
        else:
            metrics = train_step(
                model,
                batch,
                optimizer,
                device,
                ema_decay=config.ema_decay,
                variance_weight=config.variance_weight,
                min_std=config.min_std,
            )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        samples += batch_size
    return {key: value / samples for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: MiniJEPA,
    optimizer: AdamW,
    *,
    epoch: int,
    best_validation_loss: float,
    history: list[dict[str, float | int]],
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "history": history,
            "model_config": asdict(model.config),
            "training_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MiniJEPA,
    optimizer: AdamW | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model._freeze_target()
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _write_metrics(history: list[dict[str, float | int]], output_dir: Path) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_losses(history: list[dict[str, float | int]], path: Path) -> None:
    """Draw a dependency-free train/validation loss chart with Pillow."""
    width, height, margin = 800, 480, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    values = [float(row[key]) for row in history for key in ("train_loss", "validation_loss")]
    low, high = min(values), max(values)
    span = max(high - low, 1e-8)

    def point(index: int, value: float) -> tuple[float, float]:
        x = margin + index * (width - 2 * margin) / max(len(history) - 1, 1)
        y = height - margin - (value - low) / span * (height - 2 * margin)
        return x, y

    for key, color in (("train_loss", "#2468d8"), ("validation_loss", "#d84a24")):
        points = [point(index, float(row[key])) for index, row in enumerate(history)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=3)
    draw.text((margin, 18), "Mini-JEPA latent prediction loss", fill="black")
    draw.text((width - 225, 18), "train (blue)  validation (red)", fill="black")
    draw.text((margin, height - 42), f"epochs: 1-{len(history)}", fill="black")
    draw.text((width - 220, height - 42), f"loss range: {low:.4f}-{high:.4f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def train(config: TrainingConfig, *, resume: Path | None = None) -> dict[str, Any]:
    seed_everything(config.seed)
    device = select_device(config.device)
    train_dataset = TorchTrajectoryDataset(
        config.train_path,
        observed_frames=config.observed_frames,
        prediction_horizon=config.prediction_horizon,
    )
    validation_dataset = TorchTrajectoryDataset(
        config.validation_path,
        observed_frames=config.observed_frames,
        prediction_horizon=config.prediction_horizon,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    model_config = ModelConfig(
        observed_frames=config.observed_frames,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    model = MiniJEPA(model_config).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []
    if resume is not None:
        checkpoint = load_checkpoint(resume, model, optimizer, map_location=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        history = list(checkpoint["history"])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer=optimizer, config=config)
        validation_metrics = _run_epoch(
            model, validation_loader, device, optimizer=None, config=config
        )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            save_checkpoint(
                config.output_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                history=history,
                config=config,
            )
        save_checkpoint(
            config.output_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            history=history,
            config=config,
        )
        _write_metrics(history, config.output_dir)
        plot_losses(history, config.output_dir / "loss-curve.png")
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f} "
            f"target_variance={validation_metrics['target_variance']:.6e}"
        )

    summary = {
        "device": str(device),
        "duration_seconds": time.perf_counter() - started,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {"model": model, "optimizer": optimizer, "history": history, "summary": summary}
