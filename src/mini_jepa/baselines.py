"""Transparent classical and supervised baselines for M2."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def constant_velocity_predict(observed_positions: Tensor, horizon: int = 1) -> Tensor:
    """Extrapolate the last measured displacement without modeling walls."""
    if observed_positions.ndim < 2 or observed_positions.shape[-2] < 2:
        raise ValueError("at least two observed positions are required")
    displacement = observed_positions[..., -1, :] - observed_positions[..., -2, :]
    return observed_positions[..., -1, :] + horizon * displacement


def reflect_coordinates(values: Tensor, lower: float, upper: float) -> Tensor:
    """Fold unbounded coordinates into elastic-reflection bounds analytically."""
    if upper <= lower:
        raise ValueError("upper bound must exceed lower bound")
    span = upper - lower
    phase = torch.remainder(values - lower, 2.0 * span)
    return lower + torch.where(phase <= span, phase, 2.0 * span - phase)


def wall_reflection_predict(
    observed_positions: Tensor,
    horizon: int = 1,
    *,
    ball_radius: float = 0.06,
    arena_size: tuple[float, float] = (1.0, 1.0),
) -> Tensor:
    """Extrapolate measured displacement and apply exact rectangular wall folding."""
    unbounded = constant_velocity_predict(observed_positions, horizon)
    bounds = torch.tensor(arena_size, dtype=unbounded.dtype, device=unbounded.device)
    lower = torch.full_like(bounds, ball_radius)
    upper = bounds - ball_radius
    span = upper - lower
    phase = torch.remainder(unbounded - lower, 2.0 * span)
    return lower + torch.where(phase <= span, phase, 2.0 * span - phase)


class SupervisedPositionPredictor(nn.Module):
    """Small labeled CNN using exactly the same four observed images as JEPA."""

    def __init__(self, observed_frames: int = 4) -> None:
        super().__init__()
        self.observed_frames = observed_frames
        self.network = nn.Sequential(
            nn.Conv2d(observed_frames, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 5 or observations.shape[1] != self.observed_frames:
            raise ValueError("expected (batch, observed_frames, 1, 64, 64)")
        return self.network(observations.squeeze(2))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
