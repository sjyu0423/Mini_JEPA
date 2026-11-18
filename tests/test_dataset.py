from pathlib import Path

import numpy as np

from mini_jepa.dataset import DatasetConfig, TrajectoryDataset, generate_dataset
from mini_jepa.physics import SimulationConfig


def _small_config() -> DatasetConfig:
    return DatasetConfig(
        train_trajectories=3,
        validation_trajectories=2,
        test_trajectories=1,
        observed_frames=4,
        prediction_horizon=2,
        seed=42,
        simulation=SimulationConfig(num_frames=9),
    )


def _assert_npz_equal(first_path: Path, second_path: Path) -> None:
    with (
        np.load(first_path, allow_pickle=False) as first,
        np.load(second_path, allow_pickle=False) as second,
    ):
        assert first.files == second.files
        for name in first.files:
            np.testing.assert_array_equal(first[name], second[name])


def test_dataset_generation_is_reproducible(tmp_path: Path) -> None:
    first = generate_dataset(_small_config(), tmp_path / "first")
    second = generate_dataset(_small_config(), tmp_path / "second")
    for split in first:
        _assert_npz_equal(first[split], second[split])


def test_splits_have_disjoint_trajectory_ids(tmp_path: Path) -> None:
    paths = generate_dataset(_small_config(), tmp_path / "dataset")
    split_ids: dict[str, set[int]] = {}
    for split, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            split_ids[split] = set(data["trajectory_ids"].tolist())
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])


def test_input_target_sequences_use_configured_horizon(tmp_path: Path) -> None:
    paths = generate_dataset(_small_config(), tmp_path / "dataset")
    with TrajectoryDataset(paths["train"], observed_frames=4, prediction_horizon=2) as dataset:
        sample = dataset[0]
        assert len(dataset) == 3 * (9 - 4 - 2 + 1)
        assert sample["observations"].shape == (4, 64, 64)
        assert sample["target"].shape == (64, 64)
        assert sample["start_frame"] == 0
        assert sample["target_frame"] == 5
        with np.load(paths["train"], allow_pickle=False) as raw:
            np.testing.assert_array_equal(sample["observations"], raw["frames"][0, :4])
            np.testing.assert_array_equal(sample["target"], raw["frames"][0, 5])


def test_saved_physics_metadata_shapes(tmp_path: Path) -> None:
    paths = generate_dataset(_small_config(), tmp_path / "dataset")
    with np.load(paths["test"], allow_pickle=False) as data:
        assert data["frames"].shape == (1, 9, 64, 64)
        assert data["positions"].shape == (1, 9, 2)
        assert data["velocities"].shape == (1, 9, 2)
        assert data["collision_counts"].shape == (1, 9, 2)
        assert data["frames"].dtype == np.uint8
        assert data["positions"].dtype == np.float32
