"""Labeled trajectory windows used only for downstream M2 evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class EvaluationSample(TypedDict):
    observations: Tensor
    target_image: Tensor
    observed_positions: Tensor
    current_position: Tensor
    current_velocity: Tensor
    future_position: Tensor
    bounce: Tensor
    trajectory_id: Tensor
    start_frame: Tensor
    target_frame: Tensor


class EvaluationTrajectoryDataset(Dataset[EvaluationSample]):
    """Evaluation-only view containing labels that never enter the JEPA predictor."""

    def __init__(
        self,
        path: Path,
        *,
        observed_frames: int = 4,
        prediction_horizon: int = 1,
    ) -> None:
        if observed_frames < 2 or prediction_horizon < 1:
            raise ValueError("evaluation requires at least two observations and a positive horizon")
        with np.load(path, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.positions = data["positions"]
            self.velocities = data["velocities"]
            self.collision_counts = data["collision_counts"]
            self.trajectory_ids = data["trajectory_ids"]
        self.observed_frames = observed_frames
        self.prediction_horizon = prediction_horizon
        trajectory_count, frame_count = self.frames.shape[:2]
        windows = frame_count - observed_frames - prediction_horizon + 1
        if windows < 1:
            raise ValueError("trajectory is too short for the requested evaluation window")
        self.index = [
            (trajectory, start)
            for trajectory in range(trajectory_count)
            for start in range(windows)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> EvaluationSample:
        trajectory, start = self.index[index]
        current = start + self.observed_frames - 1
        target = current + self.prediction_horizon
        observations = torch.from_numpy(self.frames[trajectory, start : current + 1].copy())
        target_image = torch.from_numpy(self.frames[trajectory, target].copy())
        bounce = bool(np.any(self.collision_counts[trajectory, current + 1 : target + 1] > 0))
        return {
            "observations": observations.unsqueeze(1).to(torch.float32).div_(255.0),
            "target_image": target_image.unsqueeze(0).to(torch.float32).div_(255.0),
            "observed_positions": torch.from_numpy(
                self.positions[trajectory, start : current + 1].copy()
            ),
            "current_position": torch.from_numpy(self.positions[trajectory, current].copy()),
            "current_velocity": torch.from_numpy(self.velocities[trajectory, current].copy()),
            "future_position": torch.from_numpy(self.positions[trajectory, target].copy()),
            "bounce": torch.tensor(bounce),
            "trajectory_id": torch.tensor(self.trajectory_ids[trajectory], dtype=torch.int64),
            "start_frame": torch.tensor(start, dtype=torch.int64),
            "target_frame": torch.tensor(target, dtype=torch.int64),
        }
