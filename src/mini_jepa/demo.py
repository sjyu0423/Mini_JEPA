"""Real frozen-model inference and deterministic playback logic for the M4 demo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from mini_jepa.baselines import constant_velocity_predict
from mini_jepa.m3_probes import load_probe
from mini_jepa.m3_training import load_m3_checkpoint
from mini_jepa.physics import SimulationConfig, Trajectory, simulate_trajectory
from mini_jepa.probes import load_frozen_jepa
from mini_jepa.rendering import render_trajectory
from mini_jepa.training import select_device

ScenarioName = Literal["ordinary", "near-wall", "high-speed", "multiple-bounce"]
ModelName = Literal["constant-velocity", "m1-jepa", "m3-jepa"]
SCENARIOS: tuple[ScenarioName, ...] = (
    "ordinary",
    "near-wall",
    "high-speed",
    "multiple-bounce",
)
MODELS: tuple[ModelName, ...] = ("constant-velocity", "m1-jepa", "m3-jepa")


@dataclass(frozen=True, slots=True)
class DemoPaths:
    m1_checkpoint: Path = Path("artifacts/m1/best.pt")
    m1_probe: Path = Path("artifacts/m2/future-state-probe.pt")
    m3_checkpoint: Path = Path("artifacts/m3/stage2/a1_gru_cosine/seed-502/best.pt")
    m3_probe: Path = Path("artifacts/m3/stage2/a1_gru_cosine/seed-502/probe-h1.pt")


@dataclass(frozen=True, slots=True)
class ScenarioData:
    name: ScenarioName
    seed: int
    trajectory: Trajectory
    frames: NDArray[np.uint8]
    timestep: float
    ball_radius: float


def scenario_data(name: ScenarioName, seed: int, *, num_frames: int = 180) -> ScenarioData:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    common = {"num_frames": num_frames, "timestep": 1.0 / 30.0, "ball_radius": 0.06}
    initial_position: tuple[float, float] | None = None
    initial_velocity: tuple[float, float] | None = None
    if name == "ordinary":
        config = SimulationConfig(**common, min_speed=0.35, max_speed=0.55)
    elif name == "near-wall":
        config = SimulationConfig(**common, min_speed=0.0, max_speed=0.0)
        initial_position = (0.13, 0.58)
        initial_velocity = (-0.62, 0.18)
    elif name == "high-speed":
        config = SimulationConfig(**common, min_speed=0.9, max_speed=1.2)
    else:
        config = SimulationConfig(**common, min_speed=2.5, max_speed=3.2)
    trajectory = simulate_trajectory(
        config,
        seed,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
    )
    frames = render_trajectory(
        trajectory.positions,
        ball_radius=config.ball_radius,
        arena_size=config.arena_size,
    )
    return ScenarioData(name, seed, trajectory, frames, config.timestep, config.ball_radius)


def observation_windows(frames: NDArray[np.uint8]) -> tuple[torch.Tensor, NDArray[np.int64]]:
    """Build contexts where rows 0..3 predict target 4, never including the target frame."""
    if frames.ndim != 3 or frames.shape[1:] != (64, 64) or len(frames) < 5:
        raise ValueError("frames must have shape (time>=5, 64, 64)")
    target_indices = np.arange(4, len(frames), dtype=np.int64)
    windows = np.stack([frames[target - 4 : target] for target in target_indices])
    return torch.from_numpy(windows).to(torch.float32).unsqueeze(2).div_(255.0), target_indices


def position_error_pixels(prediction: NDArray[np.floating], actual: NDArray[np.floating]) -> float:
    return float(np.linalg.norm((np.asarray(prediction) - np.asarray(actual)) * 63.0))


class DemoEngine:
    """Load frozen models once and generate aligned predictions for deterministic scenarios."""

    def __init__(self, paths: DemoPaths, *, device_name: str = "auto") -> None:
        self.paths = paths
        self.device = select_device(device_name)
        load_started = time.perf_counter()
        self.m1_model = load_frozen_jepa(paths.m1_checkpoint, self.device)
        self.m1_probe = load_probe(paths.m1_probe)
        self.m3_model, self.m3_checkpoint = load_m3_checkpoint(paths.m3_checkpoint, self.device)
        self.m3_model.requires_grad_(False)
        self.m3_model.eval()
        self.m3_probe = load_probe(paths.m3_probe)
        self._verify_compatibility()
        self.load_latency_ms = (time.perf_counter() - load_started) * 1000.0
        self.last_inference_ms: dict[str, float] = {}

    def _verify_compatibility(self) -> None:
        if self.m1_model.config.embedding_dim != self.m1_probe.network[0].in_features:
            raise ValueError("M1 checkpoint and probe embedding dimensions do not match")
        if self.m3_model.config.embedding_dim != self.m3_probe.network[0].in_features:
            raise ValueError("M3 checkpoint and probe embedding dimensions do not match")
        if self.m3_model.config.horizons != (1,):
            raise ValueError("the demo requires the validated single-horizon M3 checkpoint")
        if (
            self.m1_probe.network[-1].out_features != 2
            or self.m3_probe.network[-1].out_features != 2
        ):
            raise ValueError("position probes must output exactly two coordinates")

    @torch.no_grad()
    def _model_predictions(self, contexts: torch.Tensor) -> dict[str, NDArray[np.float32]]:
        contexts = contexts.to(self.device)
        started = time.perf_counter()
        m1_context = self.m1_model.context_encoder(contexts)
        m1_embedding = self.m1_model.predictor(m1_context).cpu()
        m1_positions = self.m1_probe(m1_embedding).numpy().astype(np.float32)
        if self.device.type == "mps":
            torch.mps.synchronize()
        m1_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        m3_context = self.m3_model.context_encoder(contexts)
        m3_embedding = self.m3_model.predict(m3_context, (1,))[:, 0].cpu()
        m3_positions = self.m3_probe(m3_embedding).numpy().astype(np.float32)
        if self.device.type == "mps":
            torch.mps.synchronize()
        m3_elapsed = time.perf_counter() - started
        count = len(contexts)
        self.last_inference_ms = {
            "m1_batch_ms": m1_elapsed * 1000.0,
            "m1_per_prediction_ms": m1_elapsed * 1000.0 / count,
            "m3_batch_ms": m3_elapsed * 1000.0,
            "m3_per_prediction_ms": m3_elapsed * 1000.0 / count,
            "prediction_count": float(count),
        }
        return {"m1-jepa": m1_positions, "m3-jepa": m3_positions}

    def predict_scenario(
        self, name: ScenarioName, seed: int, *, num_frames: int = 180
    ) -> dict[str, Any]:
        data = scenario_data(name, seed, num_frames=num_frames)
        contexts, target_indices = observation_windows(data.frames)
        model_predictions = self._model_predictions(contexts)
        observed_positions = torch.from_numpy(
            np.stack([data.trajectory.positions[target - 4 : target] for target in target_indices])
        ).to(torch.float32)
        constant_positions = (
            constant_velocity_predict(observed_positions).numpy().astype(np.float32)
        )
        predictions = {"constant-velocity": constant_positions, **model_predictions}

        aligned_predictions: dict[str, list[list[float] | None]] = {}
        aligned_errors: dict[str, list[float | None]] = {}
        for model_name, values in predictions.items():
            positions: list[list[float] | None] = [None] * num_frames
            errors: list[float | None] = [None] * num_frames
            for row, target in enumerate(target_indices):
                positions[int(target)] = [float(values[row, 0]), float(values[row, 1])]
                errors[int(target)] = position_error_pixels(
                    values[row], data.trajectory.positions[target]
                )
            aligned_predictions[model_name] = positions
            aligned_errors[model_name] = errors

        return {
            "scenario": name,
            "seed": seed,
            "frame_count": num_frames,
            "timestep": data.timestep,
            "ball_radius": data.ball_radius,
            "positions": data.trajectory.positions.tolist(),
            "collisions": data.trajectory.collisions.tolist(),
            "collision_counts": data.trajectory.collision_counts.tolist(),
            "predictions": aligned_predictions,
            "errors": aligned_errors,
            "alignment": {
                "observed_frames": 4,
                "horizon": 1,
                "first_target_frame": 4,
                "statement": "frames t-4 through t-1 predict frame t",
            },
            "latency": self.last_inference_ms,
        }

    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "load_latency_ms": self.load_latency_ms,
            "models": list(MODELS),
            "scenarios": list(SCENARIOS),
            "m1_checkpoint": str(self.paths.m1_checkpoint),
            "m1_probe": str(self.paths.m1_probe),
            "m3_checkpoint": str(self.paths.m3_checkpoint),
            "m3_probe": str(self.paths.m3_probe),
            "last_inference": self.last_inference_ms,
        }
