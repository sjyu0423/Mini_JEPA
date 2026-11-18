from pathlib import Path

import torch

from mini_jepa.dataset import DatasetConfig, generate_dataset
from mini_jepa.physics import SimulationConfig
from mini_jepa.torch_dataset import TorchTrajectoryDataset


def test_torch_dataset_shapes_normalization_and_no_state_leakage(tmp_path: Path) -> None:
    paths = generate_dataset(
        DatasetConfig(
            train_trajectories=2,
            validation_trajectories=1,
            test_trajectories=1,
            simulation=SimulationConfig(num_frames=8),
        ),
        tmp_path,
    )
    dataset = TorchTrajectoryDataset(paths["train"], observed_frames=4, prediction_horizon=2)
    sample = dataset[0]
    assert sample.keys() == {
        "observations",
        "target",
        "trajectory_id",
        "start_frame",
        "target_frame",
    }
    assert sample["observations"].shape == (4, 1, 64, 64)
    assert sample["target"].shape == (1, 64, 64)
    assert sample["observations"].dtype == torch.float32
    assert 0.0 <= sample["observations"].min() <= sample["observations"].max() <= 1.0
    assert "positions" not in sample
    assert "velocities" not in sample
    assert sample["target_frame"].item() == 5
