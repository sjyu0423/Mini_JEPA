from pathlib import Path

import torch
from torch.optim import AdamW

from mini_jepa.model import MiniJEPA, ModelConfig, jepa_loss
from mini_jepa.training import TrainingConfig, load_checkpoint, save_checkpoint, train_step


def _batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "observations": torch.rand(batch_size, 4, 1, 64, 64),
        "target": torch.rand(batch_size, 1, 64, 64),
        "trajectory_id": torch.arange(batch_size),
        "start_frame": torch.zeros(batch_size, dtype=torch.int64),
        "target_frame": torch.full((batch_size,), 4, dtype=torch.int64),
    }


def test_model_shapes_and_finite_embeddings() -> None:
    model = MiniJEPA(ModelConfig(observed_frames=4, embedding_dim=64, hidden_dim=96))
    outputs = model(_batch()["observations"], _batch()["target"])
    assert outputs.keys() == {"context", "prediction", "target"}
    assert all(embedding.shape == (3, 64) for embedding in outputs.values())
    assert all(torch.isfinite(embedding).all() for embedding in outputs.values())


def test_loss_is_finite_and_nonnegative() -> None:
    model = MiniJEPA()
    batch = _batch()
    loss, diagnostics = jepa_loss(model(batch["observations"], batch["target"]))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_gradients_reach_online_network_but_not_target() -> None:
    model = MiniJEPA()
    batch = _batch()
    loss, _ = jepa_loss(model(batch["observations"], batch["target"]))
    loss.backward()
    online_gradients = [
        parameter.grad
        for parameter in (*model.context_encoder.parameters(), *model.predictor.parameters())
    ]
    assert any(gradient is not None and torch.any(gradient != 0) for gradient in online_gradients)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_ema_updates_target_toward_online_weights() -> None:
    model = MiniJEPA()
    with torch.no_grad():
        for parameter in model.context_encoder.frame_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in model.target_encoder.parameters():
            parameter.zero_()
    model.update_target(0.75)
    for parameter in model.target_encoder.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))


def test_one_training_step_updates_online_parameters() -> None:
    model = MiniJEPA()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3
    )
    before = [parameter.detach().clone() for parameter in model.predictor.parameters()]
    metrics = train_step(model, _batch(), optimizer, torch.device("cpu"), ema_decay=0.99)
    after = list(model.predictor.parameters())
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = MiniJEPA(ModelConfig(embedding_dim=32, hidden_dim=48))
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3
    )
    train_step(model, _batch(), optimizer, torch.device("cpu"), ema_decay=0.9)
    checkpoint_path = tmp_path / "checkpoint.pt"
    config = TrainingConfig(
        train_path=tmp_path / "train.npz",
        validation_path=tmp_path / "validation.npz",
        embedding_dim=32,
        hidden_dim=48,
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=2,
        best_validation_loss=0.5,
        history=[{"epoch": 2, "train_loss": 0.6, "validation_loss": 0.5}],
        config=config,
    )

    restored = MiniJEPA(ModelConfig(embedding_dim=32, hidden_dim=48))
    restored_optimizer = AdamW(
        (parameter for parameter in restored.parameters() if parameter.requires_grad), lr=1e-3
    )
    checkpoint = load_checkpoint(checkpoint_path, restored, restored_optimizer)
    assert checkpoint["epoch"] == 2
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    assert all(not parameter.requires_grad for parameter in restored.target_encoder.parameters())
