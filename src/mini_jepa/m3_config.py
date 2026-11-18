"""Small serializable experiment configuration for M3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mini_jepa.m3_model import Architecture, Objective


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    architecture: Architecture
    objective: Objective
    training_seed: int
    dataset_seed: int = 2026
    horizons: tuple[int, ...] = (1,)
    observed_frames: int = 4
    embedding_dim: int = 64
    hidden_dim: int = 128
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.99
    variance_weight: float = 0.1
    min_std: float = 0.1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
