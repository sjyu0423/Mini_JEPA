"""Controlled M3 architecture and objective variants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mini_jepa.model import FrameEncoder, Predictor

Architecture = Literal["mlp", "gru"]
Objective = Literal["cosine", "normalized_regression"]


@dataclass(frozen=True, slots=True)
class M3ModelConfig:
    architecture: Architecture = "mlp"
    observed_frames: int = 4
    embedding_dim: int = 64
    hidden_dim: int = 128
    horizons: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.architecture not in ("mlp", "gru"):
            raise ValueError(f"unsupported architecture: {self.architecture}")
        if not self.horizons or any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must contain positive offsets")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")


class MLPContextEncoder(nn.Module):
    def __init__(self, frame_encoder: FrameEncoder, config: M3ModelConfig) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.observed_frames = config.observed_frames
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(config.observed_frames * config.embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )

    def forward(self, observations: Tensor) -> Tensor:
        batch, frames, channels, height, width = _validate_observations(
            observations, self.observed_frames
        )
        encoded = self.frame_encoder(observations.reshape(batch * frames, channels, height, width))
        return self.temporal_aggregator(encoded.reshape(batch, -1))


class GRUContextEncoder(nn.Module):
    def __init__(self, frame_encoder: FrameEncoder, config: M3ModelConfig) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.observed_frames = config.observed_frames
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )

    def forward(self, observations: Tensor) -> Tensor:
        batch, frames, channels, height, width = _validate_observations(
            observations, self.observed_frames
        )
        encoded = self.frame_encoder(observations.reshape(batch * frames, channels, height, width))
        sequence = encoded.reshape(batch, frames, -1)
        _, hidden = self.gru(sequence)
        return self.projection(hidden[-1])


def _validate_observations(observations: Tensor, observed_frames: int) -> tuple[int, ...]:
    if observations.ndim != 5 or observations.shape[1:] != (observed_frames, 1, 64, 64):
        raise ValueError("expected observations shaped (batch, observed_frames, 1, 64, 64)")
    return observations.shape


class HorizonConditionedPredictor(nn.Module):
    def __init__(self, config: M3ModelConfig) -> None:
        super().__init__()
        self.horizon_embedding = nn.Embedding(max(config.horizons) + 1, config.embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * config.embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def forward(self, context: Tensor, horizons: Tensor) -> Tensor:
        batch, embedding_dim = context.shape
        horizon_features = self.horizon_embedding(horizons)
        expanded_context = context[:, None, :].expand(batch, len(horizons), embedding_dim)
        return self.network(
            torch.cat((expanded_context, horizon_features[None].expand(batch, -1, -1)), dim=-1)
        )


class M3JEPA(nn.Module):
    def __init__(self, config: M3ModelConfig | None = None) -> None:
        super().__init__()
        config = config or M3ModelConfig()
        self.config = config
        online_frame_encoder = FrameEncoder(config.embedding_dim)
        context_type = MLPContextEncoder if config.architecture == "mlp" else GRUContextEncoder
        self.context_encoder = context_type(online_frame_encoder, config)
        self.target_encoder = copy.deepcopy(online_frame_encoder)
        self.multi_horizon = len(config.horizons) > 1
        self.predictor: nn.Module
        if self.multi_horizon:
            self.predictor = HorizonConditionedPredictor(config)
        else:
            self.predictor = Predictor(config.embedding_dim, config.hidden_dim)
        self._freeze_target()

    def _freeze_target(self) -> None:
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> M3JEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def predict(self, context: Tensor, horizons: tuple[int, ...] | None = None) -> Tensor:
        selected = horizons or self.config.horizons
        if self.multi_horizon:
            horizon_tensor = torch.tensor(selected, device=context.device, dtype=torch.long)
            return self.predictor(context, horizon_tensor)
        if selected != self.config.horizons:
            raise ValueError("a single-horizon model cannot predict an untrained horizon")
        return self.predictor(context)[:, None, :]

    def forward(self, observations: Tensor, target_frames: Tensor) -> dict[str, Tensor]:
        if target_frames.ndim != 5 or target_frames.shape[1] != len(self.config.horizons):
            raise ValueError("target frames must have shape (batch, horizons, 1, 64, 64)")
        context = self.context_encoder(observations)
        prediction = self.predict(context)
        batch, horizons, channels, height, width = target_frames.shape
        with torch.no_grad():
            targets = self.target_encoder(
                target_frames.reshape(batch * horizons, channels, height, width)
            ).reshape(batch, horizons, -1)
        return {"context": context, "prediction": prediction, "target": targets}

    @torch.no_grad()
    def update_target(self, decay: float) -> None:
        online = self.context_encoder.frame_encoder
        for target_parameter, online_parameter in zip(
            self.target_encoder.parameters(), online.parameters(), strict=True
        ):
            target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)
        for target_buffer, online_buffer in zip(
            self.target_encoder.buffers(), online.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer)


def m3_loss(
    outputs: dict[str, Tensor],
    *,
    objective: Objective,
    variance_weight: float = 0.1,
    min_std: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    prediction = outputs["prediction"]
    target = outputs["target"].detach()
    context = outputs["context"]
    normalized_prediction = functional.normalize(prediction, dim=-1)
    normalized_target = functional.normalize(target, dim=-1)
    if objective == "cosine":
        prediction_loss = 1.0 - (normalized_prediction * normalized_target).sum(dim=-1).mean()
    elif objective == "normalized_regression":
        # Smooth normalized-vector regression is not algebraically identical to cosine distance.
        prediction_loss = prediction.shape[-1] * functional.smooth_l1_loss(
            normalized_prediction, normalized_target
        )
    else:
        raise ValueError(f"unsupported objective: {objective}")

    flattened_prediction = prediction.reshape(-1, prediction.shape[-1])
    context_std = torch.sqrt(context.var(dim=0, unbiased=False) + 1e-4)
    prediction_std = torch.sqrt(flattened_prediction.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = (
        functional.relu(min_std - context_std).mean()
        + functional.relu(min_std - prediction_std).mean()
    )
    total = prediction_loss + variance_weight * variance_loss
    norm_difference = torch.abs(prediction.norm(dim=-1).mean() - target.norm(dim=-1).mean())
    diagnostics = {
        "prediction_loss": prediction_loss.detach(),
        "variance_loss": variance_loss.detach(),
        "context_variance": context.var(dim=0, unbiased=False).mean().detach(),
        "prediction_variance": flattened_prediction.var(dim=0, unbiased=False).mean().detach(),
        "target_variance": target.reshape(-1, target.shape[-1])
        .var(dim=0, unbiased=False)
        .mean()
        .detach(),
        "normalized_prediction_variance": normalized_prediction.reshape(-1, prediction.shape[-1])
        .var(dim=0, unbiased=False)
        .mean()
        .detach(),
        "cosine_similarity": (normalized_prediction * normalized_target)
        .sum(dim=-1)
        .mean()
        .detach(),
        "norm_difference": norm_difference.detach(),
    }
    return total, diagnostics


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
