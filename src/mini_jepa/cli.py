"""Command-line tools for dataset generation and visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mini_jepa.dataset import DatasetConfig, generate_dataset
from mini_jepa.evaluation import EvaluationConfig, evaluate
from mini_jepa.m3_experiments import run_m3
from mini_jepa.model import trainable_parameter_count
from mini_jepa.physics import SimulationConfig
from mini_jepa.rendering import save_gif, save_montage
from mini_jepa.training import TrainingConfig, train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-jepa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate reproducible dataset splits")
    generate.add_argument("--output", type=Path, default=Path("data/m0"))
    generate.add_argument("--train", type=int, default=16)
    generate.add_argument("--validation", type=int, default=4)
    generate.add_argument("--test", type=int, default=4)
    generate.add_argument("--frames", type=int, default=32)
    generate.add_argument("--observed", type=int, default=4)
    generate.add_argument("--horizon", type=int, default=1)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--radius", type=float, default=0.06)
    generate.add_argument("--timestep", type=float, default=1.0 / 30.0)
    generate.add_argument("--min-speed", type=float, default=0.25)
    generate.add_argument("--max-speed", type=float, default=0.65)

    inspect = subparsers.add_parser("inspect", help="export a montage and GIF from a split")
    inspect.add_argument("dataset", type=Path, help="path to a split NPZ")
    inspect.add_argument("--trajectory", type=int, default=0)
    inspect.add_argument("--output", type=Path, default=Path("artifacts/m0"))

    training = subparsers.add_parser("train", help="train the small temporal JEPA")
    training.add_argument("--data", type=Path, default=Path("data/m1"))
    training.add_argument("--output", type=Path, default=Path("artifacts/m1"))
    training.add_argument("--observed", type=int, default=4)
    training.add_argument("--horizon", type=int, default=1)
    training.add_argument("--embedding-dim", type=int, default=64)
    training.add_argument("--hidden-dim", type=int, default=128)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--ema-decay", type=float, default=0.99)
    training.add_argument("--variance-weight", type=float, default=0.1)
    training.add_argument("--min-std", type=float, default=0.1)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    training.add_argument("--resume", type=Path)

    evaluation = subparsers.add_parser("evaluate", help="run frozen M2 evaluation")
    evaluation.add_argument("--checkpoint", type=Path, default=Path("artifacts/m1/best.pt"))
    evaluation.add_argument("--data", type=Path, default=Path("data/m1"))
    evaluation.add_argument("--ood-test", type=Path, default=Path("data/m2-ood/test.npz"))
    evaluation.add_argument("--output", type=Path, default=Path("artifacts/m2"))
    evaluation.add_argument("--batch-size", type=int, default=128)
    evaluation.add_argument("--probe-epochs", type=int, default=120)
    evaluation.add_argument("--supervised-epochs", type=int, default=20)
    evaluation.add_argument("--seed", type=int, default=303)
    evaluation.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")

    m3 = subparsers.add_parser("m3-experiments", help="run controlled M3 experiments")
    m3.add_argument("--data", type=Path, default=Path("data/m1"))
    m3.add_argument("--m1-checkpoint", type=Path, default=Path("artifacts/m1/best.pt"))
    m3.add_argument("--m2-artifacts", type=Path, default=Path("artifacts/m2"))
    m3.add_argument("--ood-test", type=Path, default=Path("data/m2-ood/test.npz"))
    m3.add_argument("--diagnostic", type=Path, default=Path("data/m3-diagnostic/test.npz"))
    m3.add_argument("--output", type=Path, default=Path("artifacts/m3"))
    m3.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def _generate(args: argparse.Namespace) -> None:
    simulation = SimulationConfig(
        num_frames=args.frames,
        timestep=args.timestep,
        ball_radius=args.radius,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
    )
    config = DatasetConfig(
        train_trajectories=args.train,
        validation_trajectories=args.validation,
        test_trajectories=args.test,
        observed_frames=args.observed,
        prediction_horizon=args.horizon,
        seed=args.seed,
        simulation=simulation,
    )
    paths = generate_dataset(config, args.output)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


def _inspect(args: argparse.Namespace) -> None:
    with np.load(args.dataset, allow_pickle=False) as data:
        frames = data["frames"]
        collisions = data["collision_counts"]
        if args.trajectory < 0 or args.trajectory >= len(frames):
            raise SystemExit(f"trajectory must be between 0 and {len(frames) - 1}")
        selected_frames = frames[args.trajectory]
        selected_collisions = collisions[args.trajectory]
    stem = f"{args.dataset.stem}-trajectory-{args.trajectory}"
    montage_path = args.output / f"{stem}-montage.png"
    gif_path = args.output / f"{stem}.gif"
    save_montage(selected_frames, selected_collisions, montage_path)
    save_gif(selected_frames, gif_path)
    print(json.dumps({"montage": str(montage_path), "gif": str(gif_path)}, indent=2))


def _train(args: argparse.Namespace) -> None:
    config = TrainingConfig(
        train_path=args.data / "train.npz",
        validation_path=args.data / "validation.npz",
        output_dir=args.output,
        observed_frames=args.observed,
        prediction_horizon=args.horizon,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        variance_weight=args.variance_weight,
        min_std=args.min_std,
        seed=args.seed,
        device=args.device,
    )
    result = train(config, resume=args.resume)
    summary = dict(result["summary"])
    summary["trainable_parameters"] = trainable_parameter_count(result["model"])
    print(json.dumps(summary, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    config = EvaluationConfig(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        ood_test_path=args.ood_test,
        output_dir=args.output,
        batch_size=args.batch_size,
        probe_epochs=args.probe_epochs,
        supervised_epochs=args.supervised_epochs,
        seed=args.seed,
        device=args.device,
    )
    results = evaluate(config)
    concise = {
        "device": results["device"],
        "duration_seconds": results["duration_seconds"],
        "dataset": results["dataset"],
        "test_rmse_pixels": {
            name: metrics["all"]["euclidean_rmse_pixels"]
            for name, metrics in results["test"].items()
        },
    }
    print(json.dumps(concise, indent=2))


def _m3_experiments(args: argparse.Namespace) -> None:
    results = run_m3(
        data_dir=args.data,
        m1_checkpoint=args.m1_checkpoint,
        m2_artifacts=args.m2_artifacts,
        ood_test_path=args.ood_test,
        output_dir=args.output,
        diagnostic_path=args.diagnostic,
        device_name=args.device,
    )
    concise = {
        "device": results["device"],
        "duration_seconds": results["duration_seconds"],
        "selected": results["selection"]["selected_nonbaseline"],
        "best_model": results["best_model"],
        "stage2_aggregate": results["stage2_aggregate"],
    }
    print(json.dumps(concise, indent=2))


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate":
        _generate(args)
    elif args.command == "inspect":
        _inspect(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)
    else:
        _m3_experiments(args)


if __name__ == "__main__":
    main()
