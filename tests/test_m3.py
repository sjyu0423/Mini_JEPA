from pathlib import Path

import numpy as np
import pytest
import torch

from mini_jepa.dataset import DatasetConfig, generate_dataset
from mini_jepa.m3_config import ExperimentConfig
from mini_jepa.m3_data import MultiHorizonTrajectoryDataset
from mini_jepa.m3_diagnostics_data import CONDITIONS, generate_bounce_diagnostics
from mini_jepa.m3_experiments import _aggregate_stage2
from mini_jepa.m3_model import M3JEPA, M3ModelConfig, m3_loss
from mini_jepa.m3_training import load_m3_checkpoint, save_m3_checkpoint
from mini_jepa.physics import SimulationConfig


def test_gru_and_multi_horizon_shapes() -> None:
    model = M3JEPA(M3ModelConfig(architecture="gru", horizons=(1, 2, 4)))
    outputs = model(
        torch.rand(3, 4, 1, 64, 64),
        torch.rand(3, 3, 1, 64, 64),
    )
    assert outputs["context"].shape == (3, 64)
    assert outputs["prediction"].shape == (3, 3, 64)
    assert outputs["target"].shape == (3, 3, 64)


@pytest.mark.parametrize("objective", ["cosine", "normalized_regression"])
def test_m3_losses_are_finite_and_trainable(objective: str) -> None:
    model = M3JEPA()
    outputs = model(torch.rand(4, 4, 1, 64, 64), torch.rand(4, 1, 1, 64, 64))
    loss, diagnostics = m3_loss(outputs, objective=objective)  # type: ignore[arg-type]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_m3_ema_updates_target() -> None:
    model = M3JEPA()
    with torch.no_grad():
        for parameter in model.context_encoder.frame_encoder.parameters():
            parameter.fill_(4.0)
        for parameter in model.target_encoder.parameters():
            parameter.zero_()
    model.update_target(0.75)
    for parameter in model.target_encoder.parameters():
        torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_multi_horizon_alignment_and_no_target_leakage(tmp_path: Path) -> None:
    paths = generate_dataset(
        DatasetConfig(
            train_trajectories=1,
            validation_trajectories=1,
            test_trajectories=1,
            simulation=SimulationConfig(num_frames=10),
        ),
        tmp_path,
    )
    dataset = MultiHorizonTrajectoryDataset(paths["train"], horizons=(1, 2, 4))
    sample = dataset[0]
    assert sample["target_frames"].tolist() == [4, 5, 7]
    with np.load(paths["train"], allow_pickle=False) as raw:
        np.testing.assert_array_equal(
            (sample["targets"].squeeze(1).numpy() * 255).astype(np.uint8),
            raw["frames"][0, [4, 5, 7]],
        )
    model = M3JEPA(M3ModelConfig(horizons=(1, 2, 4))).eval()
    observations = sample["observations"].unsqueeze(0)
    with torch.no_grad():
        first = model(observations, torch.zeros(1, 3, 1, 64, 64))["prediction"]
        second = model(observations, torch.ones(1, 3, 1, 64, 64))["prediction"]
    torch.testing.assert_close(first, second)


def test_experiment_config_is_reproducible() -> None:
    first = ExperimentConfig("run", "gru", "cosine", 7, horizons=(1, 2, 4))
    second = ExperimentConfig("run", "gru", "cosine", 7, horizons=(1, 2, 4))
    assert first.canonical_json() == second.canonical_json()


def test_m3_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig("roundtrip", "gru", "cosine", 1)
    model = M3JEPA(M3ModelConfig(architecture="gru"))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3
    )
    path = tmp_path / "m3.pt"
    save_m3_checkpoint(
        path,
        model,
        optimizer,
        config,
        epoch=2,
        best_validation_loss=0.2,
        history=[{"epoch": 2, "validation_loss": 0.2}],
    )
    restored, checkpoint = load_m3_checkpoint(path, torch.device("cpu"))
    assert checkpoint["epoch"] == 2
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_diagnostic_dataset_conditions_and_isolation(tmp_path: Path) -> None:
    path = generate_bounce_diagnostics(tmp_path / "diagnostic.npz", trajectories_per_condition=2)
    with np.load(path, allow_pickle=False) as data:
        assert data["frames"].shape == (8, 32, 64, 64)
        assert set(data["condition"].tolist()) == set(range(len(CONDITIONS)))
        assert np.all(data["trajectory_ids"] >= 3_000_000)


def test_stage2_aggregation_mean_and_sample_std() -> None:
    runs = []
    for seed, rmse in ((1, 1.0), (2, 2.0), (3, 3.0)):
        all_metrics = {
            "euclidean_rmse_pixels": rmse,
            "median_error_pixels": rmse,
            "p90_error_pixels": rmse,
            "mean_error_pixels": rmse,
        }
        runs.append(
            {
                "seed": seed,
                "test": {
                    "all": all_metrics,
                    "bounce": {"mean_error_pixels": rmse},
                    "non_bounce": {"mean_error_pixels": rmse},
                },
                "ood": {"all": {"euclidean_rmse_pixels": rmse}},
            }
        )
    aggregate = _aggregate_stage2(runs)
    assert aggregate["euclidean_rmse_pixels"] == {"mean": 2.0, "std": 1.0}
