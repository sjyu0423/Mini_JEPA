"""Independent bounce-rich diagnostic trajectory generation for M3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mini_jepa.physics import SimulationConfig, simulate_trajectory
from mini_jepa.rendering import render_trajectory

CONDITIONS = ("ordinary", "near_wall", "multiple_bounce", "high_speed")


def generate_bounce_diagnostics(
    path: Path,
    *,
    trajectories_per_condition: int = 64,
    seed: int = 12_345,
) -> Path:
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    collision_counts: list[np.ndarray] = []
    condition_ids: list[int] = []
    trajectory_ids: list[int] = []
    radius = 0.06
    base = SimulationConfig(num_frames=32, ball_radius=radius)

    for condition_id, condition in enumerate(CONDITIONS):
        for local_index in range(trajectories_per_condition):
            trajectory_seed = seed + condition_id * 100_000 + local_index
            initial_position: tuple[float, float] | None = None
            initial_velocity: tuple[float, float] | None = None
            config = base
            if condition == "near_wall":
                wall = int(rng.integers(0, 4))
                # Time the first impact after the four-frame observation window begins.
                offset = float(rng.uniform(0.05, 0.10))
                other = float(rng.uniform(radius + 0.1, 1.0 - radius - 0.1))
                speed = float(rng.uniform(0.4, 0.8))
                tangent = float(rng.uniform(-0.15, 0.15))
                if wall == 0:
                    initial_position = (radius + offset, other)
                    initial_velocity = (-speed, tangent)
                elif wall == 1:
                    initial_position = (1.0 - radius - offset, other)
                    initial_velocity = (speed, tangent)
                elif wall == 2:
                    initial_position = (other, radius + offset)
                    initial_velocity = (tangent, -speed)
                else:
                    initial_position = (other, 1.0 - radius - offset)
                    initial_velocity = (tangent, speed)
            elif condition == "multiple_bounce":
                config = SimulationConfig(
                    num_frames=32,
                    ball_radius=radius,
                    min_speed=2.5,
                    max_speed=3.5,
                )
            elif condition == "high_speed":
                config = SimulationConfig(
                    num_frames=32,
                    ball_radius=radius,
                    min_speed=0.8,
                    max_speed=1.2,
                )

            trajectory = simulate_trajectory(
                config,
                trajectory_seed,
                initial_position=initial_position,
                initial_velocity=initial_velocity,
            )
            frames.append(
                render_trajectory(
                    trajectory.positions,
                    ball_radius=config.ball_radius,
                    arena_size=config.arena_size,
                )
            )
            positions.append(trajectory.positions.astype(np.float32))
            velocities.append(trajectory.velocities.astype(np.float32))
            collision_counts.append(trajectory.collision_counts)
            condition_ids.append(condition_id)
            trajectory_ids.append(3_000_000 + condition_id * 100_000 + local_index)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.stack(frames),
        positions=np.stack(positions),
        velocities=np.stack(velocities),
        collision_counts=np.stack(collision_counts),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        condition=np.asarray(condition_ids, dtype=np.uint8),
    )
    metadata = {
        "seed": seed,
        "trajectories_per_condition": trajectories_per_condition,
        "conditions": {str(index): name for index, name in enumerate(CONDITIONS)},
        "training_or_selection_use": False,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
