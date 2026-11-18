"""Reproducible trajectory-level dataset generation and loading."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mini_jepa.physics import SimulationConfig, simulate_trajectory
from mini_jepa.rendering import render_trajectory

SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Settings for creating trajectory-level splits."""

    train_trajectories: int = 16
    validation_trajectories: int = 4
    test_trajectories: int = 4
    observed_frames: int = 4
    prediction_horizon: int = 1
    seed: int = 7
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def __post_init__(self) -> None:
        counts = (
            self.train_trajectories,
            self.validation_trajectories,
            self.test_trajectories,
        )
        if any(count < 1 for count in counts):
            raise ValueError("each split must contain at least one trajectory")
        if self.observed_frames < 1 or self.prediction_horizon < 1:
            raise ValueError("observed_frames and prediction_horizon must be positive")
        required = self.observed_frames + self.prediction_horizon
        if required > self.simulation.num_frames:
            raise ValueError("trajectory is too short for the requested sample window")


def generate_dataset(config: DatasetConfig, output_dir: Path) -> dict[str, Path]:
    """Generate independent NPZ files, partitioned by complete trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": config.train_trajectories,
        "validation": config.validation_trajectories,
        "test": config.test_trajectories,
    }
    paths: dict[str, Path] = {}

    for split, count in counts.items():
        frames: list[NDArray[np.uint8]] = []
        positions: list[NDArray[np.float32]] = []
        velocities: list[NDArray[np.float32]] = []
        collision_counts: list[NDArray[np.uint16]] = []
        trajectory_ids = np.arange(count, dtype=np.int64) + SPLIT_OFFSETS[split]

        for trajectory_id in trajectory_ids:
            trajectory = simulate_trajectory(
                config.simulation, seed=config.seed + int(trajectory_id)
            )
            frames.append(
                render_trajectory(
                    trajectory.positions,
                    ball_radius=config.simulation.ball_radius,
                    arena_size=config.simulation.arena_size,
                )
            )
            positions.append(trajectory.positions.astype(np.float32))
            velocities.append(trajectory.velocities.astype(np.float32))
            collision_counts.append(trajectory.collision_counts)

        path = output_dir / f"{split}.npz"
        np.savez_compressed(
            path,
            frames=np.stack(frames),
            positions=np.stack(positions),
            velocities=np.stack(velocities),
            collision_counts=np.stack(collision_counts),
            trajectory_ids=trajectory_ids,
        )
        paths[split] = path

    metadata = {
        "format_version": 1,
        "dataset": {key: value for key, value in asdict(config).items() if key != "simulation"},
        "simulation": asdict(config.simulation),
        "splits": {split: path.name for split, path in paths.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return paths


class TrajectoryDataset:
    """Framework-neutral sliding-window view over an on-disk trajectory split."""

    def __init__(self, path: Path, *, observed_frames: int = 4, prediction_horizon: int = 1):
        if observed_frames < 1 or prediction_horizon < 1:
            raise ValueError("observed_frames and prediction_horizon must be positive")
        self.path = path
        self.observed_frames = observed_frames
        self.prediction_horizon = prediction_horizon
        self._data = np.load(path, mmap_mode="r", allow_pickle=False)
        trajectory_count, frame_count = self._data["frames"].shape[:2]
        windows_per_trajectory = frame_count - observed_frames - prediction_horizon + 1
        if windows_per_trajectory < 1:
            raise ValueError("trajectory is too short for the requested sample window")
        self._index = np.array(
            [
                (trajectory, start)
                for trajectory in range(trajectory_count)
                for start in range(windows_per_trajectory)
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, NDArray[np.generic] | np.generic]:
        trajectory, start = self._index[index]
        target = start + self.observed_frames + self.prediction_horizon - 1
        return {
            "observations": self._data["frames"][trajectory, start : start + self.observed_frames],
            "target": self._data["frames"][trajectory, target],
            "trajectory_id": self._data["trajectory_ids"][trajectory],
            "start_frame": np.int64(start),
            "target_frame": np.int64(target),
        }

    def close(self) -> None:
        self._data.close()

    def __enter__(self) -> TrajectoryDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
