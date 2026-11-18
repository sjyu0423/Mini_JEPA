"""Deterministic two-dimensional bouncing-ball physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for a unit-square bouncing-ball simulation."""

    num_frames: int = 32
    timestep: float = 1.0 / 30.0
    ball_radius: float = 0.06
    min_speed: float = 0.25
    max_speed: float = 0.65
    arena_size: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        width, height = self.arena_size
        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if self.timestep <= 0:
            raise ValueError("timestep must be positive")
        if self.ball_radius <= 0 or 2 * self.ball_radius >= min(width, height):
            raise ValueError("ball_radius must fit inside the arena")
        if self.min_speed < 0 or self.max_speed < self.min_speed:
            raise ValueError("speed bounds must satisfy 0 <= min_speed <= max_speed")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Physical state and collision counts for one trajectory."""

    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    collision_counts: NDArray[np.uint16]

    @property
    def collisions(self) -> NDArray[np.bool_]:
        return np.any(self.collision_counts > 0, axis=1)


def _advance(
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    timestep: float,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint16]]:
    """Advance one timestep, resolving every impact (including corner impacts)."""
    position = position.copy()
    velocity = velocity.copy()
    counts = np.zeros(2, dtype=np.uint16)
    remaining = timestep
    tolerance = np.finfo(np.float64).eps * 32

    while remaining > tolerance:
        hit_times = np.full(2, np.inf, dtype=np.float64)
        positive = velocity > 0
        negative = velocity < 0
        hit_times[positive] = (upper[positive] - position[positive]) / velocity[positive]
        hit_times[negative] = (lower[negative] - position[negative]) / velocity[negative]
        hit_times[(hit_times < 0) & (hit_times > -tolerance)] = 0.0

        next_hit = float(np.min(hit_times))
        if not np.isfinite(next_hit) or next_hit > remaining + tolerance:
            position += velocity * remaining
            remaining = 0.0
            break

        travel = min(max(next_hit, 0.0), remaining)
        position += velocity * travel
        remaining -= travel

        hit_axes = np.isclose(hit_times, next_hit, rtol=0.0, atol=tolerance)
        position = np.clip(position, lower, upper)
        velocity[hit_axes] *= -1.0
        counts[hit_axes] += 1

    if remaining > 0:
        position += velocity * remaining
    return np.clip(position, lower, upper), velocity, counts


def simulate_trajectory(
    config: SimulationConfig,
    seed: int,
    *,
    initial_position: tuple[float, float] | None = None,
    initial_velocity: tuple[float, float] | None = None,
) -> Trajectory:
    """Simulate one trajectory; identical arguments produce identical results."""
    rng = np.random.default_rng(seed)
    radius = config.ball_radius
    lower = np.array([radius, radius], dtype=np.float64)
    upper = np.asarray(config.arena_size, dtype=np.float64) - radius

    if initial_position is None:
        position = rng.uniform(lower, upper)
    else:
        position = np.asarray(initial_position, dtype=np.float64)
    if np.any(position < lower) or np.any(position > upper):
        raise ValueError("initial_position places the ball outside the arena")

    if initial_velocity is None:
        speed = rng.uniform(config.min_speed, config.max_speed)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        velocity = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    else:
        velocity = np.asarray(initial_velocity, dtype=np.float64)
    if position.shape != (2,) or velocity.shape != (2,):
        raise ValueError("initial position and velocity must each contain two values")

    positions = np.empty((config.num_frames, 2), dtype=np.float64)
    velocities = np.empty_like(positions)
    collision_counts = np.zeros((config.num_frames, 2), dtype=np.uint16)
    positions[0] = position
    velocities[0] = velocity

    for frame in range(1, config.num_frames):
        position, velocity, counts = _advance(position, velocity, config.timestep, lower, upper)
        positions[frame] = position
        velocities[frame] = velocity
        collision_counts[frame] = counts

    return Trajectory(positions, velocities, collision_counts)
