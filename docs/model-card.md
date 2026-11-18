# Model card — Mini-JEPA

## Model details

- **Name:** Mini-JEPA
- **Version:** 1.0.0
- **Type:** Self-supervised joint-embedding predictive architecture for synthetic visual motion
- **Framework:** PyTorch
- **License:** MIT
- **Repository:** https://github.com/sjyu0423/Mini_JEPA
- **Research site:** https://sjyu0423.github.io/Mini_JEPA/

### Architecture

Four 64×64 grayscale frames pass through a shared convolutional frame encoder and a temporal
aggregator. The historical M1 model uses concatenation plus an MLP; the selected M3 improvement uses
a GRU. A predictor estimates the next-frame embedding. An exponential-moving-average (EMA) target
encoder supplies a stop-gradient latent target. Training matches embeddings rather than pixels.

After representation learning, a separately trained linear position probe maps predicted embeddings
to `(x, y)` coordinates. The probe is not part of the JEPA objective.

### Checkpoints referenced by the research site

| Role | Path | SHA-256 |
|---|---|---|
| M1 model | `artifacts/m1/best.pt` | `0b7ac6fa7c37662bf3bc634985e19dd510efbd83a32527d5ff18666b1dc043a5` |
| M1 probe | `artifacts/m2/future-state-probe.pt` | `acc27bee69e6a18f918f3b540b12c2d93e63acb28d04ad35d10a61c3792b424f` |
| M3 model (seed 502) | `artifacts/m3/stage2/a1_gru_cosine/seed-502/best.pt` | `a76c46976c4dd4b8b4163b2a3ff29baf48317955cf69ed3f0ac594151954aa9d` |
| M3 probe | `artifacts/m3/stage2/a1_gru_cosine/seed-502/probe-h1.pt` | `a54e1e24db0baa700827eb618a88e8459a32ab4e2f158851a5451463f51b702f` |

Checkpoint binaries are intentionally not distributed in the source repository.

## Intended use

- Educational and portfolio demonstration of an end-to-end JEPA-style research workflow
- Inspectable baseline for synthetic motion representation learning
- Local interactive comparison of frozen checkpoint predictions against deterministic scenarios

## Out-of-scope uses

- Claims of general physical intelligence or real-world robotics readiness
- Safety-critical prediction or control
- Autoregressive multi-step forecasting (reported predictions are independent horizon-1 estimates)

## Training data

Synthetic bouncing-ball trajectories with constant velocity and elastic wall reflections. Rendering
produces 64×64 grayscale images. Complete trajectories are assigned to exactly one split. Simulator
positions and velocities are unavailable to the JEPA model during self-supervised training.

Historical M1/M3 training used dataset seed 2026 with four observed frames and horizon 1. Exact
dataset files remain local/gitignored; generation commands are documented in the README.

## Evaluation data and metrics

Position Euclidean RMSE in pixels (lower is better). Reported headline numbers:

| Method | In-distribution test | High-speed OOD |
|---|---:|---:|
| M1 JEPA + frozen linear probe | 2.835 | 5.391 |
| Selected M3 GRU, seed 502 + probe | 2.694 | 3.578 |
| M3 GRU family, 3-seed mean ± SD | 2.894 ± 0.217 | 3.884 ± 0.292 |
| Supervised CNN | 0.650 | 1.247 |
| Wall-aware analytic physics | 0.111 | 0.312 |
| Random frozen encoder + probe | 11.394 | — |

Baselines are supervision-aware references, not matched self-supervised competitors. Selected-seed
results are not interchangeable with multi-seed averages. Multiple-bounce diagnostics remain hard
(14.241 px M1 → 7.745 px selected M3).

## Limitations

- Synthetic single-ball domain with known deterministic physics
- Probe quality measures decodability, not a complete causal world model
- Seed variance is meaningful; M1 has a single historical seed
- Neural training is seeded but not guaranteed bitwise-reproducible across CPU/MPS builds
- Website demo uses precomputed predictions; no model runs in the browser

## Ethical considerations

No personal data, biometric data, or human subjects are involved. The synthetic domain minimizes
downstream misuse risk, but results should not be overgeneralized to real-world perception systems.

## Citation

See `CITATION.cff`.
