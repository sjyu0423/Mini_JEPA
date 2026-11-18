from pathlib import Path

import numpy as np
import torch

from mini_jepa.baselines import constant_velocity_predict, wall_reflection_predict
from mini_jepa.dataset import DatasetConfig, generate_dataset
from mini_jepa.diagnostics import representation_diagnostics
from mini_jepa.evaluation_data import EvaluationTrajectoryDataset
from mini_jepa.metrics import grouped_position_metrics, position_metrics
from mini_jepa.model import MiniJEPA, ModelConfig
from mini_jepa.physics import SimulationConfig
from mini_jepa.probes import extract_embeddings, load_frozen_jepa


def test_evaluation_labels_align_with_window_indices(tmp_path: Path) -> None:
    paths = generate_dataset(
        DatasetConfig(
            train_trajectories=1,
            validation_trajectories=1,
            test_trajectories=1,
            simulation=SimulationConfig(num_frames=8),
        ),
        tmp_path,
    )
    dataset = EvaluationTrajectoryDataset(paths["train"], prediction_horizon=2)
    sample = dataset[0]
    with np.load(paths["train"], allow_pickle=False) as raw:
        np.testing.assert_array_equal(sample["observed_positions"], raw["positions"][0, :4])
        np.testing.assert_array_equal(sample["current_position"], raw["positions"][0, 3])
        np.testing.assert_array_equal(sample["future_position"], raw["positions"][0, 5])
    assert sample["target_frame"].item() == 5
    assert sample["observations"].shape == (4, 1, 64, 64)


def test_future_image_cannot_change_predictor_output() -> None:
    torch.manual_seed(0)
    model = MiniJEPA().eval()
    observations = torch.rand(2, 4, 1, 64, 64)
    first_future = torch.zeros(2, 1, 64, 64)
    second_future = torch.ones(2, 1, 64, 64)
    with torch.no_grad():
        first = model(observations, first_future)["prediction"]
        second = model(observations, second_future)["prediction"]
    torch.testing.assert_close(first, second)


def test_loaded_jepa_is_fully_frozen(tmp_path: Path) -> None:
    model = MiniJEPA(ModelConfig(embedding_dim=32, hidden_dim=48))
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {"observed_frames": 4, "embedding_dim": 32, "hidden_dim": 48},
        },
        checkpoint,
    )
    loaded = load_frozen_jepa(checkpoint, torch.device("cpu"))
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert not loaded.training


def test_embedding_extraction_preserves_trajectory_ids(tmp_path: Path) -> None:
    paths = generate_dataset(
        DatasetConfig(
            train_trajectories=1,
            validation_trajectories=1,
            test_trajectories=1,
            simulation=SimulationConfig(num_frames=6),
        ),
        tmp_path,
    )
    dataset = EvaluationTrajectoryDataset(paths["test"])
    model = MiniJEPA().eval()
    bundle = extract_embeddings(model, dataset, torch.device("cpu"), batch_size=2)
    assert bundle.trajectory_ids.shape == (2,)
    assert np.all(bundle.trajectory_ids == 2_000_000)


def test_constant_velocity_and_wall_reflection_baselines() -> None:
    observed = torch.tensor([[[0.55, 0.5], [0.70, 0.5], [0.85, 0.5]]])
    constant = constant_velocity_predict(observed)
    reflected = wall_reflection_predict(observed, ball_radius=0.1)
    torch.testing.assert_close(constant, torch.tensor([[1.0, 0.5]]))
    torch.testing.assert_close(reflected, torch.tensor([[0.8, 0.5]]))


def test_position_metrics_are_correct() -> None:
    predictions = np.array([[1.0, 0.0], [0.0, 1.0]])
    targets = np.zeros((2, 2))
    metrics = position_metrics(predictions, targets, pixels_per_unit=1.0)
    assert metrics["mae_pixels"] == 0.5
    assert metrics["euclidean_rmse_pixels"] == 1.0
    assert metrics["median_error_pixels"] == 1.0
    assert metrics["p90_error_pixels"] == 1.0


def test_bounce_categorization_uses_transition_to_target(tmp_path: Path) -> None:
    frames = np.zeros((1, 6, 64, 64), dtype=np.uint8)
    positions = np.zeros((1, 6, 2), dtype=np.float32)
    velocities = np.zeros((1, 6, 2), dtype=np.float32)
    collisions = np.zeros((1, 6, 2), dtype=np.uint16)
    collisions[0, 4, 0] = 1
    path = tmp_path / "split.npz"
    np.savez_compressed(
        path,
        frames=frames,
        positions=positions,
        velocities=velocities,
        collision_counts=collisions,
        trajectory_ids=np.array([8]),
    )
    dataset = EvaluationTrajectoryDataset(path, observed_frames=4, prediction_horizon=1)
    assert dataset[0]["bounce"].item() is True
    assert dataset[1]["bounce"].item() is False


def test_grouped_metrics_respect_bounce_mask() -> None:
    targets = np.zeros((2, 2), dtype=np.float32)
    predictions = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32) / 63.0
    metrics = grouped_position_metrics(
        predictions, targets, np.array([True, False], dtype=np.bool_)
    )
    assert metrics["bounce"]["mean_error_pixels"] == 1.0
    assert metrics["non_bounce"]["mean_error_pixels"] == 0.0


def test_representation_diagnostics_are_deterministic() -> None:
    rng = np.random.default_rng(4)
    context = rng.normal(size=(30, 8))
    prediction = rng.normal(size=(30, 8))
    target = rng.normal(size=(30, 8))
    first = representation_diagnostics(context, prediction, target, seed=9)
    second = representation_diagnostics(context, prediction, target, seed=9)
    assert first == second
