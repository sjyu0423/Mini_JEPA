"""Small JEPA-style temporal latent prediction model."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModelConfig:
    observed_frames: int = 4
    embedding_dim: int = 64
    hidden_dim: int = 128


class FrameEncoder(nn.Module):
    """Encode one grayscale frame while retaining coarse spatial information."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4 or frames.shape[1:] != (1, 64, 64):
            raise ValueError("frame encoder expects shape (batch, 1, 64, 64)")
        return self.projection(self.features(frames))


class ContextEncoder(nn.Module):
    """Aggregate per-frame online embeddings into a temporal context embedding."""

    def __init__(self, frame_encoder: FrameEncoder, config: ModelConfig) -> None:
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
        if observations.ndim != 5 or observations.shape[1] != self.observed_frames:
            raise ValueError("context encoder expects shape (batch, observed_frames, 1, 64, 64)")
        batch, frames, channels, height, width = observations.shape
        encoded = self.frame_encoder(observations.reshape(batch * frames, channels, height, width))
        return self.temporal_aggregator(encoded.reshape(batch, -1))


class Predictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        return self.network(context)


class MiniJEPA(nn.Module):
    """Online context encoder, EMA target frame encoder, and latent predictor."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        online_frame_encoder = FrameEncoder(config.embedding_dim)
        self.context_encoder = ContextEncoder(online_frame_encoder, config)
        self.target_encoder = copy.deepcopy(online_frame_encoder)
        self.predictor = Predictor(config.embedding_dim, config.hidden_dim)
        self._freeze_target()

    def _freeze_target(self) -> None:
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> MiniJEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, observations: Tensor, target: Tensor) -> dict[str, Tensor]:
        context = self.context_encoder(observations)
        prediction = self.predictor(context)
        with torch.no_grad():
            target_embedding = self.target_encoder(target)
        return {"context": context, "prediction": prediction, "target": target_embedding}

    @torch.no_grad()
    def update_target(self, decay: float) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        online = self.context_encoder.frame_encoder
        for target_parameter, online_parameter in zip(
            self.target_encoder.parameters(), online.parameters(), strict=True
        ):
            target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)
        for target_buffer, online_buffer in zip(
            self.target_encoder.buffers(), online.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer)


def jepa_loss(
    outputs: dict[str, Tensor], *, variance_weight: float = 0.1, min_std: float = 0.1
) -> tuple[Tensor, dict[str, Tensor]]:
    """Cosine latent prediction with a variance-floor collapse safeguard."""
    prediction = outputs["prediction"]
    target = outputs["target"].detach()
    context = outputs["context"]
    prediction_loss = 1.0 - functional.cosine_similarity(prediction, target, dim=-1).mean()

    context_std = torch.sqrt(context.var(dim=0, unbiased=False) + 1e-4)
    prediction_std = torch.sqrt(prediction.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = (
        functional.relu(min_std - context_std).mean()
        + functional.relu(min_std - prediction_std).mean()
    )
    total = prediction_loss + variance_weight * variance_loss
    diagnostics = {
        "prediction_loss": prediction_loss.detach(),
        "variance_loss": variance_loss.detach(),
        "context_variance": context.var(dim=0, unbiased=False).mean().detach(),
        "prediction_variance": prediction.var(dim=0, unbiased=False).mean().detach(),
        "target_variance": target.var(dim=0, unbiased=False).mean().detach(),
    }
    return total, diagnostics


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
