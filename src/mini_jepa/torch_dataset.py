"""PyTorch adapter exposing images without privileged simulation state."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class JEPASample(TypedDict):
    observations: Tensor
    target: Tensor
    trajectory_id: Tensor
    start_frame: Tensor
    target_frame: Tensor


class TorchTrajectoryDataset(Dataset[JEPASample]):
    """In-memory image-window view over an M0 trajectory split.

    Positions, velocities, and collision metadata are deliberately not loaded or exposed.
    """

    def __init__(
        self,
        path: Path,
        *,
        observed_frames: int = 4,
        prediction_horizon: int = 1,
    ) -> None:
        if observed_frames < 1 or prediction_horizon < 1:
            raise ValueError("observed_frames and prediction_horizon must be positive")
        with np.load(path, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.trajectory_ids = data["trajectory_ids"]
        self.observed_frames = observed_frames
        self.prediction_horizon = prediction_horizon
        trajectory_count, frame_count = self.frames.shape[:2]
        windows = frame_count - observed_frames - prediction_horizon + 1
        if windows < 1:
            raise ValueError("trajectory is too short for the requested sample window")
        self.index = [
            (trajectory, start)
            for trajectory in range(trajectory_count)
            for start in range(windows)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> JEPASample:
        trajectory, start = self.index[index]
        target_index = start + self.observed_frames + self.prediction_horizon - 1
        observations = torch.from_numpy(
            self.frames[trajectory, start : start + self.observed_frames].copy()
        ).to(torch.float32)
        target = torch.from_numpy(self.frames[trajectory, target_index].copy()).to(torch.float32)
        return {
            "observations": observations.unsqueeze(1).div_(255.0),
            "target": target.unsqueeze(0).div_(255.0),
            "trajectory_id": torch.tensor(self.trajectory_ids[trajectory], dtype=torch.int64),
            "start_frame": torch.tensor(start, dtype=torch.int64),
            "target_frame": torch.tensor(target_index, dtype=torch.int64),
        }
