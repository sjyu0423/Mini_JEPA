# W1 — GitHub release preparation audit

Date: 2026-11-18  
Target repository: `sjyu0423/Mini_JEPA`  
Expected site: `https://sjyu0423.github.io/Mini_JEPA/`  
Status: **Prepared locally. Not published.**

No git commit, push, tag, merge, or deployment was performed during W1.

## Files changed in W1

### Added

- `LICENSE` — MIT
- `CITATION.cff`
- `docs/model-card.md`
- `docs/github-repository-metadata.md`
- `docs/website/w1-release-audit.md` (this file)
- `website/scripts/release-qa.mjs` — desktop/mobile Playwright release QA

### Updated

- `README.md` — license/citation/model-card links
- `.gitignore` — ignore `website/qa-artifacts/`
- `.github/workflows/deploy-website.yml` — clarified Pages URL comments
- `tests/test_release.py` — require LICENSE, CITATION.cff, model card
- `website/src/lib/site.ts` — verified GitHub URL + canonical helpers
- `website/src/app/layout.tsx` — SEO/canonical/Open Graph metadata
- `website/src/app/sitemap.ts` — published-site fallback
- `website/src/components/SiteHeader.tsx` / `SiteFooter.tsx` / `page.tsx` — GitHub links
- `website/package.json` — `qa:release` script; Playwright as optional QA dependency
- `scripts/export_website_data.py` — Ruff formatting only

### Unchanged research artifacts

- No model architecture, training code, checkpoints, datasets, or historical result JSON were modified.
- Checkpoint SHA-256 values continue to match `docs/artifact-provenance.md`.

## Repository audit findings

| Check | Result |
|---|---|
| Git commits | None yet (`milestone-0-simulation`, empty history) |
| Git remote | None configured |
| Secrets / credentials | None found in publishable sources |
| Machine-specific paths | None in publishable docs/source/website public data. Local `__pycache__` binaries contain absolute paths but are gitignored via `__pycache__/` / `*.py[cod]`. `tests/test_release.py` mentions forbidden path prefixes only as scanner needles. |
| Large datasets / checkpoints | Present locally under `data/` and `artifacts/`, both gitignored |
| Website static assets | Small curated JSON (~76 KB data) + plots (~1.3 MB) |
| Accidental generated dirs | `website/node_modules`, `.next`, `out`, `qa-artifacts` ignored |

Publishable top-level candidates:

```text
.github/
.gitignore
CITATION.cff
LICENSE
README.md
docs/
pyproject.toml
scripts/
src/
tests/
uv.lock
website/   # excluding ignored build/install dirs
```

## Website QA results

Performed against a GitHub Pages-like local mount:

`http://127.0.0.1:4321/Mini_JEPA/` ← contents of `website/out/`

| Check | Result |
|---|---|
| `/`, `/architecture/`, `/experiments/`, `/engineering/` | 200 on desktop and mobile |
| Interactive demo load | Passed |
| Scenario switching + M1/M3 selection | Passed |
| Real prediction readout | Observed numeric px errors (e.g. high-speed M1 ~4.51 px) |
| Disclaimer present | “No model runs in the browser” |
| Headline metrics | 2.835 / 2.694 present |
| Asset paths under `/Mini_JEPA/` | Correct; no `/Mini_JEPA/Mini_JEPA` doubling |
| Sitemap / robots | Canonical Pages URLs correct |
| Reduced-motion play behavior | Passed |
| Broken images | None detected on checked routes |

Screenshots (local only, gitignored): `website/qa-artifacts/desktop-home.png`, `mobile-home.png`.

## Reproducibility audit results

| Command | Result |
|---|---|
| `uv sync --locked` | Passed |
| `uv run ruff check .` | Passed |
| `uv run ruff format --check .` | Passed |
| `uv run pytest` | **45 passed** |
| `npm run test` | Passed (playback schema + helpers) |
| `npm run lint` | Passed |
| `npm run typecheck` | Passed |
| `NEXT_PUBLIC_BASE_PATH=/Mini_JEPA npm run build` | Passed → `website/out` |
| `npm run qa:release` | Passed |
| Checkpoint hash verification vs provenance | Passed |

## GitHub Actions workflow verification

File: `.github/workflows/deploy-website.yml`

- Permissions: `contents: read`, `pages: write`, `id-token: write`
- Build is pure Node static export; no Python inference backend
- `NEXT_PUBLIC_BASE_PATH=/${{ github.event.repository.name }}` → `/Mini_JEPA` for this target
- `NEXT_PUBLIC_SITE_URL=https://${{ github.repository_owner }}.github.io`
- Uploads `website/out` via `actions/upload-pages-artifact`
- Deploy job uses `actions/deploy-pages`
- Triggered by push to `main`/`master` (path-filtered) or `workflow_dispatch`
- No credentials or secrets referenced
- **Not triggered during W1**

## Release blockers (owner actions)

These are intentional publication gates, not code defects:

1. Create GitHub repository `sjyu0423/Mini_JEPA` (does not exist as a configured remote yet).
2. Choose initial branch (`main` recommended; current local branch is `milestone-0-simulation`).
3. Explicitly authorize the initial commit and push.
4. Enable Pages → Source: GitHub Actions.
5. Optionally attach a separate release bundle for checkpoints/datasets; they are not in Git by design.
6. Optional: replace `CITATION.cff` author family-name `sjyu0423` with a legal name if desired.

## Proposed initial commit contents

Include:

- Source package (`src/`, `tests/`, `pyproject.toml`, `uv.lock`)
- Research docs (`docs/`, including model card and website docs)
- Website source + curated `website/public` data/plots
- GitHub Actions workflow
- `LICENSE`, `CITATION.cff`, `README.md`, `.gitignore`

Exclude (already ignored):

- `.venv/`, `data/`, `artifacts/`
- `website/node_modules/`, `website/.next/`, `website/out/`, `website/qa-artifacts/`
- caches, coverage, OS junk

## Proposed GitHub Pages settings

- Repository homepage: `https://sjyu0423.github.io/Mini_JEPA/`
- Description/topics: see `docs/github-repository-metadata.md`
- Pages source: GitHub Actions
- Expected first successful deploy URL: `https://sjyu0423.github.io/Mini_JEPA/`

## Exact publication sequence (await authorization)

1. Confirm owner authorization to publish.
2. `git checkout -b main` (or rename current branch).
3. Stage the proposed publishable files only.
4. Create the initial commit.
5. Create GitHub repo `sjyu0423/Mini_JEPA` if needed.
6. `git remote add origin git@github.com:sjyu0423/Mini_JEPA.git`
7. `git push -u origin main`
8. Enable Pages (GitHub Actions).
9. Allow the deploy workflow to run; verify the live site and demo.
10. Optionally create GitHub Release notes + external artifact bundle with checksums.

**Stop here until explicitly authorized.**
