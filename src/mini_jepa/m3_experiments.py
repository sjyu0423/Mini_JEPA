"""Two-stage controlled M3 experiment orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from mini_jepa.evaluation_data import EvaluationTrajectoryDataset
from mini_jepa.evaluation_plots import save_actual_vs_predicted, save_embedding_diagnostics
from mini_jepa.m3_config import ExperimentConfig
from mini_jepa.m3_diagnostics_data import CONDITIONS, generate_bounce_diagnostics
from mini_jepa.m3_plots import save_bar_chart, save_horizon_curve, save_training_curves
from mini_jepa.m3_probes import (
    evaluate_probe,
    fit_validation_probes,
    load_probe,
)
from mini_jepa.m3_training import load_m3_checkpoint, train_experiment
from mini_jepa.metrics import grouped_position_metrics, position_metrics
from mini_jepa.probes import extract_embeddings, load_frozen_jepa, probe_predict
from mini_jepa.training import select_device

STAGE1_SEED = 310
STAGE2_SEEDS = (501, 502, 503)


def stage1_configs() -> tuple[ExperimentConfig, ...]:
    return (
        ExperimentConfig(
            name="a0_mlp_cosine",
            architecture="mlp",
            objective="cosine",
            training_seed=STAGE1_SEED,
        ),
        ExperimentConfig(
            name="a1_gru_cosine",
            architecture="gru",
            objective="cosine",
            training_seed=STAGE1_SEED,
        ),
        ExperimentConfig(
            name="b0_mlp_normalized",
            architecture="mlp",
            objective="normalized_regression",
            training_seed=STAGE1_SEED,
        ),
        ExperimentConfig(
            name="c1_gru_normalized_multi",
            architecture="gru",
            objective="normalized_regression",
            training_seed=STAGE1_SEED,
            horizons=(1, 2, 4),
        ),
    )


def _condition_metrics(
    predictions: np.ndarray,
    extracted: dict[str, np.ndarray],
    dataset: EvaluationTrajectoryDataset,
    path: Path,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        trajectory_conditions = data["condition"]
    sample_conditions = np.asarray(
        [trajectory_conditions[trajectory] for trajectory, _start in dataset.index]
    )
    results: dict[str, Any] = {}
    for condition_id, name in enumerate(CONDITIONS):
        mask = sample_conditions == condition_id
        condition_predictions = predictions[mask]
        targets = extracted["future_position"][mask].astype(np.float32)
        bounce = extracted["bounce"][mask].astype(np.bool_)
        results[name] = {
            "all": position_metrics(condition_predictions, targets),
            "grouped": grouped_position_metrics(condition_predictions, targets, bounce),
            "bounce_windows": int(bounce.sum()),
            "window_count": int(mask.sum()),
        }
    return results


def _aggregate_stage2(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "euclidean_rmse_pixels",
        "median_error_pixels",
        "p90_error_pixels",
        "mean_error_pixels",
    )
    aggregate: dict[str, Any] = {"seeds": [run["seed"] for run in runs]}
    for key in keys:
        values = np.asarray([run["test"]["all"][key] for run in runs])
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    for group in ("bounce", "non_bounce"):
        values = np.asarray([run["test"][group]["mean_error_pixels"] for run in runs])
        aggregate[f"{group}_mean_error_pixels"] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    ood = np.asarray([run["ood"]["all"]["euclidean_rmse_pixels"] for run in runs])
    aggregate["ood_rmse_pixels"] = {"mean": float(ood.mean()), "std": float(ood.std(ddof=1))}
    return aggregate


def run_m3(
    *,
    data_dir: Path,
    m1_checkpoint: Path,
    m2_artifacts: Path,
    ood_test_path: Path,
    output_dir: Path,
    diagnostic_path: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    device = select_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_bounce_diagnostics(diagnostic_path)

    stage1: dict[str, Any] = {}
    stage1_dirs: dict[str, Path] = {}
    configs = stage1_configs()
    for config in configs:
        run_dir = output_dir / "stage1" / config.name
        training = train_experiment(
            config, data_dir=data_dir, output_dir=run_dir, device_name=device_name
        )
        validation = fit_validation_probes(
            run_dir / "best.pt",
            data_dir,
            run_dir,
            device,
            seed=800 + config.training_seed,
        )
        stage1[config.name] = {
            "config": config.to_dict(),
            "training": training,
            "validation": validation,
        }
        stage1_dirs[config.name] = run_dir

    nonbaseline = [config for config in configs if config.name != "a0_mlp_cosine"]
    selected_config = min(
        nonbaseline,
        key=lambda config: stage1[config.name]["validation"]["1"]["validation"][
            "euclidean_rmse_pixels"
        ],
    )
    selection = {
        "metric": "horizon-1 validation frozen-probe Euclidean RMSE pixels",
        "selected_nonbaseline": selected_config.name,
        "values": {
            config.name: stage1[config.name]["validation"]["1"]["validation"][
                "euclidean_rmse_pixels"
            ]
            for config in configs
        },
    }
    (output_dir / "stage1-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    stage2: dict[str, list[dict[str, Any]]] = {
        "a0_mlp_cosine": [],
        selected_config.name: [],
    }
    base_config = configs[0]
    for family_config in (base_config, selected_config):
        for seed in STAGE2_SEEDS:
            config = replace(
                family_config,
                name=family_config.name,
                training_seed=seed,
            )
            run_dir = output_dir / "stage2" / family_config.name / f"seed-{seed}"
            training = train_experiment(
                config, data_dir=data_dir, output_dir=run_dir, device_name=device_name
            )
            validation = fit_validation_probes(
                run_dir / "best.pt",
                data_dir,
                run_dir,
                device,
                seed=900 + seed,
            )
            stage2[family_config.name].append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "training": training,
                    "validation": validation,
                }
            )

    # Test and diagnostic data are first used after validation-only model-family selection.
    test_path = data_dir / "test.npz"
    for _family, runs in stage2.items():
        for run in runs:
            run_dir = Path(run["run_dir"])
            model, _checkpoint = load_m3_checkpoint(run_dir / "best.pt", device)
            horizon_results: dict[str, Any] = {}
            for horizon in model.config.horizons:
                probe = load_probe(run_dir / f"probe-h{horizon}.pt")
                test_metrics, test_predictions, test_extracted = evaluate_probe(
                    model, probe, test_path, horizon, device
                )
                ood_metrics, _ood_predictions, _ood_extracted = evaluate_probe(
                    model, probe, ood_test_path, horizon, device
                )
                horizon_results[str(horizon)] = {
                    "test": test_metrics,
                    "ood": ood_metrics,
                }
                if horizon == 1:
                    run["test"] = test_metrics
                    run["ood"] = ood_metrics
                    run["test_predictions"] = test_predictions
                    run["test_actual"] = test_extracted["future_position"].astype(np.float32)
            run["horizons"] = horizon_results

    aggregate = {family: _aggregate_stage2(runs) for family, runs in stage2.items()}
    best_candidate_run = min(
        stage2[selected_config.name],
        key=lambda run: run["validation"]["1"]["validation"]["euclidean_rmse_pixels"],
    )
    best_dir = Path(best_candidate_run["run_dir"])
    best_model, _ = load_m3_checkpoint(best_dir / "best.pt", device)
    best_probe = load_probe(best_dir / "probe-h1.pt")
    diagnostic_metrics, diagnostic_predictions, diagnostic_extracted = evaluate_probe(
        best_model, best_probe, diagnostic_path, 1, device
    )
    diagnostic_dataset = EvaluationTrajectoryDataset(diagnostic_path, prediction_horizon=1)
    candidate_conditions = _condition_metrics(
        diagnostic_predictions, diagnostic_extracted, diagnostic_dataset, diagnostic_path
    )

    m1_model = load_frozen_jepa(m1_checkpoint, device)
    m1_probe = load_probe(m2_artifacts / "future-state-probe.pt")
    m1_bundle = extract_embeddings(m1_model, diagnostic_dataset, device)
    m1_predictions = probe_predict(m1_probe, m1_bundle.prediction)
    m1_extracted = {
        "future_position": m1_bundle.future_position,
        "bounce": m1_bundle.bounce,
    }
    m1_conditions = _condition_metrics(
        m1_predictions, m1_extracted, diagnostic_dataset, diagnostic_path
    )

    m2_results = json.loads((m2_artifacts / "results.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "protocol": {
            "stage1_seed": STAGE1_SEED,
            "stage2_seeds": STAGE2_SEEDS,
            "dataset_seed": 2026,
            "selection_used_test": False,
            "probe_epochs": 120,
        },
        "stage1": stage1,
        "selection": selection,
        "stage2": {
            family: [
                {
                    key: value
                    for key, value in run.items()
                    if key not in ("test_predictions", "test_actual")
                }
                for run in runs
            ]
            for family, runs in stage2.items()
        },
        "stage2_aggregate": aggregate,
        "m1_reference": {
            "test": m2_results["test"]["jepa_frozen_probe"],
            "ood": m2_results["ood"]["jepa_frozen_probe"],
        },
        "best_model": {
            "family": selected_config.name,
            "seed": best_candidate_run["seed"],
            "checkpoint": str(best_dir / "best.pt"),
            "validation_rmse_pixels": best_candidate_run["validation"]["1"]["validation"][
                "euclidean_rmse_pixels"
            ],
            "test": best_candidate_run["test"],
            "ood": best_candidate_run["ood"],
        },
        "diagnostic": {
            "overall": diagnostic_metrics,
            "candidate_conditions": candidate_conditions,
            "m1_conditions": m1_conditions,
        },
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    save_training_curves(stage1_dirs, output_dir / "stage1-training-curves.png")
    save_bar_chart(
        selection["values"],
        output_dir / "stage1-validation-comparison.png",
        title="Stage-1 validation frozen-probe comparison",
    )
    heldout_values = {
        "M1": m2_results["test"]["jepa_frozen_probe"]["all"]["euclidean_rmse_pixels"],
        "A0 mean": aggregate["a0_mlp_cosine"]["euclidean_rmse_pixels"]["mean"],
        "candidate mean": aggregate[selected_config.name]["euclidean_rmse_pixels"]["mean"],
    }
    save_bar_chart(
        heldout_values,
        output_dir / "heldout-comparison.png",
        title="Held-out test comparison across three-seed families",
    )
    multi_validation = stage1["c1_gru_normalized_multi"]["validation"]
    save_horizon_curve(
        {
            horizon: multi_validation[str(horizon)]["validation"]["euclidean_rmse_pixels"]
            for horizon in (1, 2, 4)
        },
        output_dir / "horizon-error.png",
        title="Multi-horizon validation error",
    )
    bounce_values: dict[str, float] = {}
    short_condition = {
        "ordinary": "ordinary",
        "near_wall": "near-wall",
        "multiple_bounce": "multi",
        "high_speed": "high-speed",
    }
    for condition in CONDITIONS:
        label = short_condition[condition]
        bounce_values[f"M1 {label}"] = m1_conditions[condition]["grouped"]["bounce"][
            "mean_error_pixels"
        ]
        bounce_values[f"M3 {label}"] = candidate_conditions[condition]["grouped"]["bounce"][
            "mean_error_pixels"
        ]
    save_bar_chart(
        bounce_values,
        output_dir / "bounce-diagnostic.png",
        title="Independent diagnostic bounce error",
        ylabel="Bounce mean error (pixels)",
    )
    best_diagnostics = best_candidate_run["validation"]["1"]["diagnostics"]
    save_embedding_diagnostics(best_diagnostics, output_dir / "representation-diagnostics.png")
    save_actual_vs_predicted(
        best_candidate_run["test_actual"],
        {"jepa_frozen_probe": best_candidate_run["test_predictions"]},
        output_dir / "actual-vs-predicted.png",
    )
    return result
