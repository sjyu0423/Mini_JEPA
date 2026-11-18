# Mini-JEPA research overview

## Research question

Mini-JEPA investigates whether a small joint-embedding predictive architecture can learn a useful
representation of visual motion without reconstructing pixels or receiving physical-state labels.
Four rendered frames provide context; the model predicts the latent representation of a future
frame. A linear position probe, trained only after representation learning, tests how readily the
future ball location can be decoded.

## Experimental system

The data generator simulates one circular ball in a bounded 2D arena with constant velocity and
elastic wall reflection. Collision handling supports multiple wall crossings within one timestep.
Seeded trajectory generation, complete-trajectory train/validation/test splits, and compressed NumPy
storage make the experiment deterministic and auditable. Rendering is separate from physics and
produces 64×64 grayscale images.

The M1 model contains a convolutional frame encoder, a temporal context aggregator, and a latent
predictor. Its training target comes from an exponential-moving-average target encoder. Gradients
do not enter the target branch. The objective compares embeddings rather than images, and simulator
positions and velocities are unavailable to the model.

## Evaluation design

Evaluation freezes the learned encoder and predictor, then fits lightweight position probes.
In-distribution test trajectories measure interpolation; slower out-of-distribution trajectories
measure speed-shift robustness; bounce and multiple-bounce subsets expose collision failures.
Baselines include a random frozen encoder, a fully supervised CNN, constant-velocity extrapolation,
and wall-aware analytic physics. The latter three use stronger information than JEPA and therefore
serve as reference bounds rather than matched self-supervised competitors.

M3 used a two-stage protocol. Four temporal/objective/horizon candidates were selected using only
validation probe RMSE. The chosen GRU family was then evaluated across three new seeds and compared
with the historical M1 result. This separates model selection from final test reporting and exposes
seed variance.

## Main results

| Method | Test RMSE (px) | OOD RMSE (px) |
|---|---:|---:|
| Historical M1 JEPA | 2.835 | 5.391 |
| Selected M3 GRU, seed 502 | 2.694 | 3.578 |
| M3 GRU, three-seed mean ± SD | 2.894 ± 0.217 | 3.884 ± 0.292 |
| Supervised CNN | 0.650 | 1.247 |
| Wall-aware analytic physics | 0.111 | 0.312 |
| Random frozen encoder | 11.394 | — |

The trained representation is substantially more position-informative than a random encoder. The
best GRU run improves both test and OOD error over M1, while the multi-seed result demonstrates that
the in-distribution gain is not robust: its mean is slightly worse than the single historical M1
run. OOD performance improves consistently enough to support the narrower conclusion that recurrent
aggregation helps under the tested speed shift.

## What changed from MLP to GRU

M1 concatenates four per-frame embeddings and processes them with an MLP. M3's selected model uses
a GRU to aggregate the sequence. This adds an explicit temporal recurrence while keeping the visual
encoder and downstream evaluation conceptually comparable. The most visible improvement is on hard
collision sequences: selected-run multiple-bounce RMSE falls from 14.241 to 7.745 pixels. It remains
far from solved, and this diagnostic should not be interpreted as evidence of general multi-step
physics reasoning.

## Limitations and conclusions

The environment is synthetic, visually sparse, and governed by known deterministic rules. A linear
probe establishes decodability, not causal knowledge or a complete world model. Reported predictions
are independent horizon-1 estimates; the demo is not an evaluated autoregressive rollout. Results
also depend on small training runs and show meaningful seed variance.

Within those limits, Mini-JEPA demonstrates an end-to-end, inspectable JEPA workflow: construct a
leakage-resistant self-supervised task, learn without state labels, evaluate a frozen representation
against transparent baselines, reject overbroad conclusions, and communicate the remaining failure
modes in an interactive system.
