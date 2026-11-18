# Artifact provenance

Generated datasets and experiment artifacts are intentionally excluded from Git. Exact historical
evaluation and the M4 demo require the files below at their documented paths. SHA-256 hashes identify
the locally audited artifacts; a mismatch means the published measurements should not be attributed
to that file.

| Artifact | SHA-256 |
|---|---|
| `artifacts/m1/best.pt` | `0b7ac6fa7c37662bf3bc634985e19dd510efbd83a32527d5ff18666b1dc043a5` |
| `artifacts/m1/last.pt` | `5e167acaa1f664fc1cbf009f8da4e63b407a0953df2882cdd96a6acd230a54be` |
| `artifacts/m2/future-state-probe.pt` | `acc27bee69e6a18f918f3b540b12c2d93e63acb28d04ad35d10a61c3792b424f` |
| `artifacts/m2/results.json` | `b1f46e4c73ee8f1a8b8c613d6da82bfcb8db64e42cac42ef564329a6c3852560` |
| `artifacts/m3/stage2/a1_gru_cosine/seed-502/best.pt` | `a76c46976c4dd4b8b4163b2a3ff29baf48317955cf69ed3f0ac594151954aa9d` |
| `artifacts/m3/stage2/a1_gru_cosine/seed-502/probe-h1.pt` | `a54e1e24db0baa700827eb618a88e8459a32ab4e2f158851a5451463f51b702f` |
| `artifacts/m3/results.json` | `1a27f2ac0c982987192fcf5dc2d15a4bd4085c43c6279768bb54c6362c29927b` |

## Reproducibility classes

- **Source reproducible:** simulator behavior, split construction, image rendering, dataset
  interface, model definitions, metrics, plots, demo logic, and automated tests.
- **Seeded but hardware-sensitive:** neural-network training. Commands and seeds are recorded, but
  floating-point kernels and Apple MPS can prevent bitwise-identical checkpoints.
- **Artifact dependent:** exact published tables, plots, and interactive predictions. These require
  the hashed checkpoints, probes, datasets, and result files.

## Public-release packaging

Before publishing, create a versioned release bundle containing the two demo model/probe pairs,
their configurations, result JSON, and dataset-generation metadata; publish checksums beside it.
Add a license, model card, citation metadata, and a stable download location. Do not commit large
binary artifacts directly to the source repository.
