"""Frozen-representation extraction and lightweight downstream probes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mini_jepa.evaluation_data import EvaluationTrajectoryDataset
from mini_jepa.model import MiniJEPA, ModelConfig
from mini_jepa.training import load_checkpoint


@dataclass(frozen=True, slots=True)
class EmbeddingBundle:
    context: NDArray[np.float32]
    prediction: NDArray[np.float32]
    target: NDArray[np.float32]
    current_position: NDArray[np.float32]
    current_velocity: NDArray[np.float32]
    future_position: NDArray[np.float32]
    observed_positions: NDArray[np.float32]
    bounce: NDArray[np.bool_]
    trajectory_ids: NDArray[np.int64]


def load_frozen_jepa(checkpoint_path: Path, device: torch.device) -> MiniJEPA:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = MiniJEPA(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model: MiniJEPA,
    dataset: EvaluationTrajectoryDataset,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> EmbeddingBundle:
    """Extract branches explicitly so future images cannot affect context/prediction."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collected: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "context",
            "prediction",
            "target",
            "current_position",
            "current_velocity",
            "future_position",
            "observed_positions",
            "bounce",
            "trajectory_ids",
        )
    }
    model.eval()
    for batch in loader:
        observations = batch["observations"].to(device)
        target_image = batch["target_image"].to(device)
        context = model.context_encoder(observations)
        prediction = model.predictor(context)
        target = model.target_encoder(target_image)
        collected["context"].append(context.cpu())
        collected["prediction"].append(prediction.cpu())
        collected["target"].append(target.cpu())
        for key in (
            "current_position",
            "current_velocity",
            "future_position",
            "observed_positions",
            "bounce",
        ):
            collected[key].append(batch[key].cpu())
        collected["trajectory_ids"].append(batch["trajectory_id"].cpu())

    arrays = {key: torch.cat(values).numpy() for key, values in collected.items()}
    return EmbeddingBundle(
        context=arrays["context"].astype(np.float32),
        prediction=arrays["prediction"].astype(np.float32),
        target=arrays["target"].astype(np.float32),
        current_position=arrays["current_position"].astype(np.float32),
        current_velocity=arrays["current_velocity"].astype(np.float32),
        future_position=arrays["future_position"].astype(np.float32),
        observed_positions=arrays["observed_positions"].astype(np.float32),
        bounce=arrays["bounce"].astype(np.bool_),
        trajectory_ids=arrays["trajectory_ids"].astype(np.int64),
    )


class StateProbe(nn.Module):
    """Two-layer probe with training-split feature standardization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mean: Tensor,
        scale: Tensor,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        normalized = (embeddings - self.feature_mean) / self.feature_scale
        return self.network(normalized)


def train_probe(
    train_embeddings: NDArray[np.float32],
    train_targets: NDArray[np.float32],
    validation_embeddings: NDArray[np.float32],
    validation_targets: NDArray[np.float32],
    *,
    seed: int,
    epochs: int = 120,
    learning_rate: float = 3e-3,
) -> tuple[StateProbe, dict[str, float | int]]:
    torch.manual_seed(seed)
    features = torch.from_numpy(train_embeddings)
    targets = torch.from_numpy(train_targets)
    validation_features = torch.from_numpy(validation_embeddings)
    validation_labels = torch.from_numpy(validation_targets)
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-4)
    model = StateProbe(features.shape[1], targets.shape[1], mean, scale)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = nn.functional.mse_loss(prediction, targets)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.mse_loss(model(validation_features), validation_labels)
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("probe training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_validation_mse": best_loss}


@torch.no_grad()
def probe_predict(model: StateProbe, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    return model(torch.from_numpy(embeddings)).numpy().astype(np.float32)


def save_probe(path: Path, probe: StateProbe, metadata: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": probe.state_dict(), "metadata": metadata}, path)
