# Mini-JEPA portfolio summary

## Project summary

Mini-JEPA is an end-to-end self-supervised learning study built around visual motion prediction.
It includes a deterministic physics simulator, a compact JEPA-style PyTorch model, rigorous frozen-
representation evaluation, controlled multi-seed experiments, and an interactive local demo that
compares real checkpoint predictions with the future trajectory.

## Technical architecture

Four 64×64 grayscale frames pass through a shared convolutional encoder and a temporal aggregator.
An MLP is used in the historical model and a GRU in the selected improvement. A predictor estimates
the next-frame embedding, while an exponential-moving-average target encoder supplies a stop-gradient
latent target. After training, a linear probe maps the predicted embedding to `(x, y)` position.
Simulator state is used only for evaluation and analytic baselines.

## Key findings

- Historical JEPA test/OOD position RMSE: 2.835/5.391 pixels.
- Best selected GRU run: 2.694/3.578 pixels.
- Three-seed GRU mean: 2.894 ± 0.217/3.884 ± 0.292 pixels.
- Recurrent aggregation improved the hardest selected-run multiple-bounce diagnostic from 14.241
  to 7.745 pixels, while leaving substantial room for improvement.
- A supervised CNN and wall-aware physics remained much stronger, clarifying both the cost of
  self-supervision and the limits of claims supported by this experiment.

## Engineering and research challenges

The central challenges were preventing future-state leakage, aligning contexts and targets across
training/evaluation/demo code, handling high-speed multi-collision physics, comparing models without
using test data for selection, and communicating asymmetric supervision across baselines. The final
system records seeds, split provenance, hashes, metrics, and limitations alongside executable tests.

## Limitations

This is a small synthetic domain, not a demonstration of general physical intelligence. Probe
quality measures accessible information, independent horizon-1 predictions are not multi-step
rollouts, and the small experiments show seed variability. Exact historical reproduction also
requires separately distributed checkpoints and datasets.

## Résumé-ready bullets

- Built a reproducible JEPA-style self-supervised visual motion pipeline in Python/PyTorch, spanning
  deterministic simulation, EMA-target training, frozen probes, controlled multi-seed evaluation,
  and an interactive real-checkpoint demo.
- Improved selected-run OOD position RMSE from 5.391 to 3.578 pixels with GRU temporal aggregation,
  while reporting three-seed variance and supervision-aware analytic and supervised baselines.
