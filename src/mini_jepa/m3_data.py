"""Backward-compatible image datasets for single- and multi-horizon M3 training."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MultiHorizonSample(TypedDict):
    observations: Tensor
    targets: Tensor
    trajectory_id: Tensor
    start_frame: Tensor
    target_frames: Tensor


class MultiHorizonTrajectoryDataset(Dataset[MultiHorizonSample]):
    """Image-only training adapter; physical state is never loaded."""

    def __init__(
        self,
        path: Path,
        *,
        observed_frames: int = 4,
        horizons: tuple[int, ...] = (1,),
    ) -> None:
        if observed_frames < 1 or not horizons or any(horizon < 1 for horizon in horizons):
            raise ValueError("observed frames and horizons must be positive")
        with np.load(path, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.trajectory_ids = data["trajectory_ids"]
        self.observed_frames = observed_frames
        self.horizons = horizons
        trajectory_count, frame_count = self.frames.shape[:2]
        windows = frame_count - observed_frames - max(horizons) + 1
        if windows < 1:
            raise ValueError("trajectory is too short for the requested horizons")
        self.index = [
            (trajectory, start)
            for trajectory in range(trajectory_count)
            for start in range(windows)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> MultiHorizonSample:
        trajectory, start = self.index[index]
        current = start + self.observed_frames - 1
        target_indices = np.asarray([current + horizon for horizon in self.horizons])
        observations = torch.from_numpy(self.frames[trajectory, start : current + 1].copy()).to(
            torch.float32
        )
        targets = torch.from_numpy(self.frames[trajectory, target_indices].copy()).to(torch.float32)
        return {
            "observations": observations.unsqueeze(1).div_(255.0),
            "targets": targets.unsqueeze(1).div_(255.0),
            "trajectory_id": torch.tensor(self.trajectory_ids[trajectory], dtype=torch.int64),
            "start_frame": torch.tensor(start, dtype=torch.int64),
            "target_frames": torch.from_numpy(target_indices),
        }
