from pathlib import Path

import numpy as np
import pytest
import torch

from mini_jepa.demo import (
    MODELS,
    DemoEngine,
    DemoPaths,
    observation_windows,
    position_error_pixels,
    scenario_data,
)


@pytest.fixture(scope="module")
def engine() -> DemoEngine:
    root = Path(__file__).parents[1]
    return DemoEngine(
        DemoPaths(
            root / "artifacts/m1/best.pt",
            root / "artifacts/m2/future-state-probe.pt",
            root / "artifacts/m3/stage2/a1_gru_cosine/seed-502/best.pt",
            root / "artifacts/m3/stage2/a1_gru_cosine/seed-502/probe-h1.pt",
        ),
        device_name="cpu",
    )


def test_demo_checkpoints_and_probes_load(engine: DemoEngine) -> None:
    status = engine.status()
    assert status["device"] == "cpu"
    assert status["models"] == list(MODELS)
    assert engine.m1_model.config.embedding_dim == engine.m1_probe.network[0].in_features
    assert engine.m3_model.config.embedding_dim == engine.m3_probe.network[0].in_features
    assert all(not parameter.requires_grad for parameter in engine.m1_model.parameters())
    assert all(not parameter.requires_grad for parameter in engine.m3_model.parameters())


def test_scenario_playback_is_deterministic() -> None:
    first = scenario_data("ordinary", 77, num_frames=20)
    second = scenario_data("ordinary", 77, num_frames=20)
    np.testing.assert_array_equal(first.trajectory.positions, second.trajectory.positions)
    np.testing.assert_array_equal(first.frames, second.frames)


def test_observation_target_alignment_excludes_future_frame() -> None:
    frames = np.zeros((7, 64, 64), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index].fill(index)
    contexts, targets = observation_windows(frames)
    assert targets.tolist() == [4, 5, 6]
    assert contexts[0, :, 0, 0, 0].tolist() == pytest.approx([0, 1 / 255, 2 / 255, 3 / 255])
    changed = frames.copy()
    changed[4].fill(255)
    changed_contexts, _ = observation_windows(changed)
    torch.testing.assert_close(contexts[0], changed_contexts[0])


def test_error_calculation_uses_renderer_pixel_scale() -> None:
    prediction = np.array([0.5, 0.5])
    actual = np.array([0.6, 0.5])
    assert position_error_pixels(prediction, actual) == pytest.approx(6.3)


def test_real_predictions_are_aligned_finite_and_model_selectable(engine: DemoEngine) -> None:
    payload = engine.predict_scenario("near-wall", 5, num_frames=12)
    assert payload["alignment"]["first_target_frame"] == 4
    assert len(payload["positions"]) == 12
    for model_name in MODELS:
        values = payload["predictions"][model_name]
        assert values[:4] == [None, None, None, None]
        prediction = np.asarray(values[4:], dtype=np.float32)
        assert prediction.shape == (8, 2)
        assert np.isfinite(prediction).all()
        assert all(
            error is not None and np.isfinite(error) for error in payload["errors"][model_name][4:]
        )


def test_scenario_selection_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        scenario_data("teleport", 0)  # type: ignore[arg-type]
