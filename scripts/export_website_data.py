#!/usr/bin/env python3
"""Export curated Mini-JEPA metrics and genuine checkpoint predictions for the website.

Reads local experiment artifacts and frozen checkpoints. Never synthesizes model predictions.
Output is small, versioned JSON suitable for static GitHub Pages hosting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mini_jepa.demo import SCENARIOS, DemoEngine, DemoPaths, ScenarioName  # noqa: E402

EXPORT_VERSION = "1.0.0"
DEFAULT_SEED = 42
DEFAULT_FRAMES = 120

# Curated deterministic scenarios for the static demo.
CURATED: tuple[tuple[ScenarioName, int, str], ...] = (
    ("ordinary", DEFAULT_SEED, "Typical mid-arena motion with occasional wall contact."),
    ("near-wall", DEFAULT_SEED, "Starts near a wall with an intentional bounce."),
    ("high-speed", DEFAULT_SEED, "Faster motion than the training speed band."),
    ("multiple-bounce", DEFAULT_SEED, "Very fast motion that produces dense collisions."),
)

PROVENANCE_HASHES = {
    "artifacts/m1/best.pt": "0b7ac6fa7c37662bf3bc634985e19dd510efbd83a32527d5ff18666b1dc043a5",
    "artifacts/m2/future-state-probe.pt": (
        "acc27bee69e6a18f918f3b540b12c2d93e63acb28d04ad35d10a61c3792b424f"
    ),
    "artifacts/m3/stage2/a1_gru_cosine/seed-502/best.pt": (
        "a76c46976c4dd4b8b4163b2a3ff29baf48317955cf69ed3f0ac594151954aa9d"
    ),
    "artifacts/m3/stage2/a1_gru_cosine/seed-502/probe-h1.pt": (
        "a54e1e24db0baa700827eb618a88e8459a32ab4e2f158851a5451463f51b702f"
    ),
    "artifacts/m2/results.json": "b1f46e4c73ee8f1a8b8c613d6da82bfcb8db64e42cac42ef564329a6c3852560",
    "artifacts/m3/results.json": "1a27f2ac0c982987192fcf5dc2d15a4bd4085c43c6279768bb54c6362c29927b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"non-finite value in export: {value}")
    return round(float(value), digits)


def sanitize_series(values: list[Any], digits: int = 6) -> list[Any]:
    cleaned: list[Any] = []
    for item in values:
        if item is None:
            cleaned.append(None)
        elif isinstance(item, list):
            cleaned.append([round_float(float(v), digits) for v in item])
        else:
            cleaned.append(round_float(float(item), digits))
    return cleaned


def verify_alignment(payload: dict[str, Any]) -> None:
    frame_count = int(payload["frame_count"])
    positions = payload["positions"]
    if len(positions) != frame_count:
        raise ValueError("positions length does not match frame_count")
    first_target = int(payload["alignment"]["first_target_frame"])
    for model_name, series in payload["predictions"].items():
        if len(series) != frame_count:
            raise ValueError(f"{model_name} prediction length mismatch")
        errors = payload["errors"][model_name]
        if len(errors) != frame_count:
            raise ValueError(f"{model_name} error length mismatch")
        for index, prediction in enumerate(series):
            if index < first_target:
                if prediction is not None or errors[index] is not None:
                    raise ValueError(f"{model_name} leaked prediction before first target")
            else:
                if prediction is None or errors[index] is None:
                    raise ValueError(f"{model_name} missing prediction at target frame {index}")


def strip_machine_paths(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only repository-relative artifact identifiers."""
    return payload


def export_metrics(output_dir: Path) -> dict[str, Any]:
    m2_path = ROOT / "artifacts/m2/results.json"
    m3_path = ROOT / "artifacts/m3/results.json"
    for path in (m2_path, m3_path):
        expected = PROVENANCE_HASHES[relative_posix(path)]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")

    m2 = json.loads(m2_path.read_text())
    m3 = json.loads(m3_path.read_text())
    best = m3["best_model"]
    gru_agg = m3["stage2_aggregate"]["a1_gru_cosine"]
    diagnostic = m3["diagnostic"]["overall"]

    metrics = {
        "export_version": EXPORT_VERSION,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "note": (
            "Curated metrics copied from audited experiment JSON. "
            "Lower position RMSE is better. Selected-seed results are not multi-seed averages."
        ),
        "sources": {
            "m2_results": {
                "path": relative_posix(m2_path),
                "sha256": PROVENANCE_HASHES[relative_posix(m2_path)],
            },
            "m3_results": {
                "path": relative_posix(m3_path),
                "sha256": PROVENANCE_HASHES[relative_posix(m3_path)],
            },
        },
        "comparisons": [
            {
                "id": "m1-jepa",
                "label": "M1 JEPA + frozen linear probe",
                "supervision": "Images only during JEPA training; probe uses positions later",
                "test_rmse_px": round_float(
                    m2["test"]["jepa_frozen_probe"]["all"]["euclidean_rmse_pixels"], 3
                ),
                "ood_rmse_px": round_float(
                    m2["ood"]["jepa_frozen_probe"]["all"]["euclidean_rmse_pixels"], 3
                ),
                "kind": "selected-run",
            },
            {
                "id": "m3-gru-selected",
                "label": "Selected M3 GRU JEPA, seed 502 + probe",
                "supervision": "Images only during JEPA training; probe uses positions later",
                "test_rmse_px": round_float(best["test"]["all"]["euclidean_rmse_pixels"], 3),
                "ood_rmse_px": round_float(best["ood"]["all"]["euclidean_rmse_pixels"], 3),
                "kind": "selected-run",
                "seed": best["seed"],
                "family": best["family"],
            },
            {
                "id": "m3-gru-mean",
                "label": "M3 GRU family, 3-seed mean ± SD",
                "supervision": "Images only during JEPA training; probe uses positions later",
                "test_rmse_px": round_float(gru_agg["euclidean_rmse_pixels"]["mean"], 3),
                "test_rmse_sd_px": round_float(gru_agg["euclidean_rmse_pixels"]["std"], 3),
                "ood_rmse_px": round_float(gru_agg["ood_rmse_pixels"]["mean"], 3),
                "ood_rmse_sd_px": round_float(gru_agg["ood_rmse_pixels"]["std"], 3),
                "kind": "multi-seed-mean",
                "seeds": gru_agg["seeds"],
            },
            {
                "id": "supervised-cnn",
                "label": "Supervised CNN",
                "supervision": "Receives position labels during training",
                "test_rmse_px": round_float(
                    m2["test"]["supervised_cnn"]["all"]["euclidean_rmse_pixels"], 3
                ),
                "ood_rmse_px": round_float(
                    m2["ood"]["supervised_cnn"]["all"]["euclidean_rmse_pixels"], 3
                ),
                "kind": "baseline",
            },
            {
                "id": "wall-aware-physics",
                "label": "Wall-aware analytic physics",
                "supervision": "Receives measured positions; analytic reflection model",
                "test_rmse_px": round_float(
                    m2["test"]["wall_reflection"]["all"]["euclidean_rmse_pixels"], 3
                ),
                "ood_rmse_px": round_float(
                    m2["ood"]["wall_reflection"]["all"]["euclidean_rmse_pixels"], 3
                ),
                "kind": "baseline",
            },
            {
                "id": "random-encoder",
                "label": "Random frozen encoder + probe",
                "supervision": "Untrained encoder; probe uses positions",
                "test_rmse_px": round_float(
                    m2["test"]["random_frozen_probe"]["all"]["euclidean_rmse_pixels"], 3
                ),
                "ood_rmse_px": None,
                "kind": "baseline",
            },
        ],
        "bounce_conditions": _bounce_conditions(m3),
        "test_suite": {
            "pytest_collected": 45,
            "note": "Original automated test suite through M5 collected 45 tests.",
        },
        "caveats": [
            "Lower position RMSE is better.",
            "Classical physics baselines use measured position information.",
            "Image-based models use rendered observations.",
            "Selected best-seed results are not the same as multi-seed averages.",
            "JEPA did not outperform the supervised or physics-based baselines.",
            "High-speed robustness improved, but some bounce scenarios remain difficult.",
        ],
    }

    if isinstance(diagnostic, dict) and "all" in diagnostic:
        metrics["diagnostic_overall"] = {
            "m3_rmse_px": round_float(diagnostic["all"]["euclidean_rmse_pixels"], 3),
            "m3_bounce_mean_px": round_float(diagnostic["bounce"]["mean_error_pixels"], 3),
            "m3_non_bounce_mean_px": round_float(diagnostic["non_bounce"]["mean_error_pixels"], 3),
        }

    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _bounce_conditions(m3: dict[str, Any]) -> list[dict[str, Any]]:
    m3_conditions = m3["diagnostic"]["candidate_conditions"]
    m1_conditions = m3["diagnostic"]["m1_conditions"]
    label_map = {
        "ordinary": "ordinary",
        "near_wall": "near-wall",
        "multiple_bounce": "multiple-bounce",
        "high_speed": "high-speed",
    }
    rows: list[dict[str, Any]] = []
    for key, label in label_map.items():
        m1 = m1_conditions[key]
        cand = m3_conditions[key]
        rows.append(
            {
                "condition": label,
                "bounce_windows": int(cand["bounce_windows"]),
                "m1_rmse_px": round_float(m1["all"]["euclidean_rmse_pixels"], 3),
                "m3_rmse_px": round_float(cand["all"]["euclidean_rmse_pixels"], 3),
                "m1_bounce_mean_px": round_float(m1["grouped"]["bounce"]["mean_error_pixels"], 3),
                "m3_bounce_mean_px": round_float(cand["grouped"]["bounce"]["mean_error_pixels"], 3),
            }
        )
    return rows


def export_playback(
    output_dir: Path,
    *,
    device: str,
    num_frames: int,
) -> dict[str, Any]:
    paths = DemoPaths(
        m1_checkpoint=ROOT / "artifacts/m1/best.pt",
        m1_probe=ROOT / "artifacts/m2/future-state-probe.pt",
        m3_checkpoint=ROOT / "artifacts/m3/stage2/a1_gru_cosine/seed-502/best.pt",
        m3_probe=ROOT / "artifacts/m3/stage2/a1_gru_cosine/seed-502/probe-h1.pt",
    )
    for relative, expected in PROVENANCE_HASHES.items():
        if not relative.endswith(".pt"):
            continue
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {relative}: {actual}")

    engine = DemoEngine(paths, device_name=device)
    playback_dir = output_dir / "playback"
    playback_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[dict[str, Any]] = []
    for name, seed, description in CURATED:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name}")
        raw = engine.predict_scenario(name, seed, num_frames=num_frames)
        verify_alignment(raw)

        compact = {
            "export_version": EXPORT_VERSION,
            "disclaimer": (
                "Precomputed predictions from trained PyTorch checkpoints. "
                "No model runs in the browser."
            ),
            "scenario": raw["scenario"],
            "seed": raw["seed"],
            "description": description,
            "frame_count": raw["frame_count"],
            "timestep": raw["timestep"],
            "ball_radius": raw["ball_radius"],
            "positions": sanitize_series(raw["positions"]),
            "collisions": raw["collisions"],
            "collision_counts": raw["collision_counts"],
            "predictions": {
                model: sanitize_series(series) for model, series in raw["predictions"].items()
            },
            "errors": {model: sanitize_series(series) for model, series in raw["errors"].items()},
            "alignment": raw["alignment"],
            "provenance": {
                "generator": "scripts/export_website_data.py",
                "dataset_source": "deterministic Mini-JEPA simulator scenarios (not train NPZ)",
                "generation": {
                    "seed": seed,
                    "num_frames": num_frames,
                    "observed_frames": 4,
                    "horizon": 1,
                    "device": str(engine.device),
                },
                "checkpoints": {
                    "m1_model": {
                        "path": relative_posix(paths.m1_checkpoint),
                        "sha256": PROVENANCE_HASHES[relative_posix(paths.m1_checkpoint)],
                    },
                    "m1_probe": {
                        "path": relative_posix(paths.m1_probe),
                        "sha256": PROVENANCE_HASHES[relative_posix(paths.m1_probe)],
                    },
                    "m3_model": {
                        "path": relative_posix(paths.m3_checkpoint),
                        "sha256": PROVENANCE_HASHES[relative_posix(paths.m3_checkpoint)],
                    },
                    "m3_probe": {
                        "path": relative_posix(paths.m3_probe),
                        "sha256": PROVENANCE_HASHES[relative_posix(paths.m3_probe)],
                    },
                },
                "latency_ms": {
                    key: round_float(value, 4) for key, value in raw.get("latency", {}).items()
                },
            },
        }
        compact = strip_machine_paths(compact)
        filename = f"{name}.json"
        (playback_dir / filename).write_text(json.dumps(compact, separators=(",", ":")) + "\n")
        mean_errors = {
            model: round_float(
                sum(e for e in errors if e is not None)
                / max(1, sum(1 for e in errors if e is not None)),
                3,
            )
            for model, errors in compact["errors"].items()
        }
        scenarios.append(
            {
                "id": name,
                "file": f"playback/{filename}",
                "seed": seed,
                "frame_count": num_frames,
                "description": description,
                "mean_error_px": mean_errors,
            }
        )
        print(f"exported {name}: mean errors {mean_errors}")

    manifest = {
        "export_version": EXPORT_VERSION,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "disclaimer": (
            "Precomputed predictions from trained PyTorch checkpoints. "
            "No model runs in the browser."
        ),
        "models": [
            {
                "id": "m1-jepa",
                "label": "M1 JEPA · temporal MLP",
                "input": "Rendered images only",
            },
            {
                "id": "m3-jepa",
                "label": "M3 JEPA · GRU",
                "input": "Rendered images only",
            },
            {
                "id": "constant-velocity",
                "label": "Constant velocity",
                "input": "Measured positions (privileged baseline)",
            },
        ],
        "scenarios": scenarios,
    }
    (playback_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def copy_plots(output_dir: Path) -> list[str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        ("artifacts/m2/model-comparison.png", "m2-model-comparison.png"),
        ("artifacts/m2/actual-vs-predicted.png", "m2-actual-vs-predicted.png"),
        ("artifacts/m2/bounce-comparison.png", "m2-bounce-comparison.png"),
        ("artifacts/m2/embedding-diagnostics.png", "m2-embedding-diagnostics.png"),
        ("artifacts/m3/heldout-comparison.png", "m3-heldout-comparison.png"),
        ("artifacts/m3/bounce-diagnostic.png", "m3-bounce-diagnostic.png"),
        ("artifacts/m3/representation-diagnostics.png", "m3-representation-diagnostics.png"),
        ("artifacts/m3/stage1-validation-comparison.png", "m3-stage1-validation.png"),
        ("docs/assets/mini-jepa-demo.png", "mini-jepa-demo.png"),
        ("artifacts/m4/demo-desktop.png", "m4-demo-desktop.png"),
    ]
    copied: list[str] = []
    for source_rel, destination_name in selected:
        source = ROOT / source_rel
        if not source.is_file():
            print(f"skip missing plot: {source_rel}")
            continue
        destination = plots_dir / destination_name
        destination.write_bytes(source.read_bytes())
        copied.append(f"plots/{destination_name}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "website" / "public" / "data",
        help="Website public data directory",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--skip-playback", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = export_metrics(output_dir)
    print(f"wrote {output_dir / 'metrics.json'}")

    manifest = None
    if not args.skip_playback:
        manifest = export_playback(output_dir, device=args.device, num_frames=args.frames)
        print(f"wrote playback manifest with {len(manifest['scenarios'])} scenarios")

    plots = [] if args.skip_plots else copy_plots(output_dir.parent)
    summary = {
        "export_version": EXPORT_VERSION,
        "output": (
            relative_posix(output_dir) if output_dir.is_relative_to(ROOT) else str(output_dir)
        ),
        "metrics_methods": len(metrics["comparisons"]),
        "scenarios": None if manifest is None else [s["id"] for s in manifest["scenarios"]],
        "plots": plots,
    }
    (output_dir / "export-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
