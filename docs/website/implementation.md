# Mini-JEPA website implementation

## Design decisions

- **Stack:** Next.js App Router, TypeScript, Tailwind CSS, static `output: "export"`.
- **Hosting model:** Fully static assets suitable for GitHub Pages. No database, auth, or Python
  backend at runtime.
- **Visual language:** Cool light research aesthetic with Source Serif 4 / IBM Plex Sans, restrained
  teal accent, and canvas visualizations. No decorative AI imagery.
- **Scientific honesty:** Metrics and plots come from audited `artifacts/m2` and `artifacts/m3`
  outputs. The interactive demo labels itself as precomputed checkpoint playback.
- **Base path:** `NEXT_PUBLIC_BASE_PATH` is configurable so the same build works at `/` locally or
  under `/<repository-name>` on GitHub Pages. The repository currently has no verified remote, so
  no GitHub URL is hard-coded into the UI.

## Website structure

| Route | Purpose |
|---|---|
| `/` | Hero, what was built, interactive demo, headline results, lessons |
| `/architecture/` | Encoder/predictor/EMA/probe explanation and optional deep dive |
| `/experiments/` | Tables, charts, bounce diagnostics, provenance |
| `/engineering/` | Simulator, reproducibility, testing, export pipeline |

Shared components live under `website/src/components/`. Curated JSON/plots live under
`website/public/`.

## Data export pipeline

`scripts/export_website_data.py` is separate from the frontend.

1. Verify SHA-256 hashes for M1/M3 checkpoints, probes, and result JSON against
   `docs/artifact-provenance.md`.
2. Copy curated metrics into `website/public/data/metrics.json`.
3. Run genuine frozen inference through `DemoEngine` for four deterministic scenarios.
4. Validate prediction/error alignment (no predictions before frame 4).
5. Write compact JSON under `website/public/data/playback/`.
6. Copy selected PNG plots into `website/public/plots/` (never checkpoints or NPZ datasets).

Deterministic defaults: seed `42`, `120` frames, scenarios `ordinary`, `near-wall`, `high-speed`,
`multiple-bounce`.

Regenerate:

```bash
uv run python scripts/export_website_data.py --device auto --frames 120
```

## Local development commands

```bash
cd website
npm ci
npm run dev          # http://localhost:3000
npm run test         # schema + helper checks
npm run lint
npm run typecheck
npm run build        # writes website/out
```

Subpath preview:

```bash
NEXT_PUBLIC_BASE_PATH=/Mini_JEPA npm run build
npx serve out
# open http://localhost:3000/Mini_JEPA/
```

## GitHub Pages configuration

Workflow: `.github/workflows/deploy-website.yml`

- Builds on `main`/`master` changes under `website/` or manual `workflow_dispatch`.
- Sets `NEXT_PUBLIC_BASE_PATH=/${{ github.event.repository.name }}`.
- Sets `NEXT_PUBLIC_SITE_URL=https://${{ github.repository_owner }}.github.io`.
- Uploads `website/out` with `actions/upload-pages-artifact` and deploys with
  `actions/deploy-pages`.

Owner setup still required:

1. Create the GitHub repository and push after an explicit approval.
2. Enable **Settings → Pages → Source: GitHub Actions**.
3. Confirm the repository name so the base path matches.
4. Optionally set a custom domain later; update `NEXT_PUBLIC_SITE_URL` accordingly.

This implementation does not create remotes, commits, pushes, or deployments by itself.

## Test results

Recorded during local website implementation:

| Check | Result |
|---|---|
| `npm run validate:data` | Passed — metrics match audited headline numbers; 4 scenarios aligned |
| `npm run test:helpers` | Passed |
| `npm run lint` | Passed (`eslint .`, Next 16 flat config) |
| `npm run typecheck` | Passed |
| `npm run build` | Passed — static export to `website/out` |
| `NEXT_PUBLIC_BASE_PATH=/Mini_JEPA npm run build` | Passed — asset URLs include `/Mini_JEPA/` |
| `uv run pytest` | 45 passed |
| `uv run ruff check scripts/export_website_data.py` | Passed |

Training code, model architecture, checkpoints, and historical experiment JSON were not modified.

## Limitations

- Browser demo is horizon-1 playback of curated trajectories, not live inference.
- Full trajectory truth is present in the JSON for animation, but the UI reveals it only as
  playback advances; predictions are still genuine checkpoint outputs.
- Sitemap origin defaults to a placeholder unless `NEXT_PUBLIC_SITE_URL` is set.
- No public GitHub repository URL is linked until one is verified.
- Exact regeneration of playback JSON requires local hashed checkpoints.

## Publication instructions

1. Ensure checkpoints/results match provenance hashes.
2. Re-run `scripts/export_website_data.py`.
3. Run website `test`, `lint`, `typecheck`, and `build`.
4. Review `website/out` locally, including a subpath build.
5. Complete the W1 release checklist in [`w1-release-audit.md`](w1-release-audit.md).
6. Only then, with explicit approval: commit, push, enable Pages, and run the workflow.
