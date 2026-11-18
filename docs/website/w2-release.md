# W2 — GitHub publication and Pages deployment

Date: 2026-11-18  
Status: **Published and live**

## Repository

- **URL:** https://github.com/sjyu0423/Mini_JEPA
- **Default branch:** `main`
- **Visibility:** public
- **Homepage field:** https://sjyu0423.github.io/Mini_JEPA/

## Commits

| Role | SHA |
|---|---|
| Initial release commit | `3220b5f43bdeedb47cb41f6981f43062df278c0e` |
| Latest on `main` | `8e56f6296c09a30c31828b61e342349952f9fa61` |

Initial commit message: `Initial Mini-JEPA research and website release`  
Follow-up: `Document W2 publication and fix Pages Open Graph base path`

## Deployment

- **Workflow:** `.github/workflows/deploy-website.yml` (`Deploy website to GitHub Pages`)
- **Pages build type:** GitHub Actions
- **Base path:** `/Mini_JEPA`
- **Site origin:** `https://sjyu0423.github.io`
- **First successful run:** https://github.com/sjyu0423/Mini_JEPA/actions/runs/44374113953
- **Latest successful run:** https://github.com/sjyu0423/Mini_JEPA/actions/runs/44374575824
- **Conclusion:** success (build + deploy)

## Live site

- https://sjyu0423.github.io/Mini_JEPA/
- https://sjyu0423.github.io/Mini_JEPA/architecture/
- https://sjyu0423.github.io/Mini_JEPA/experiments/
- https://sjyu0423.github.io/Mini_JEPA/engineering/

## Verification results

| Check | Result |
|---|---|
| Home / architecture / experiments / engineering HTTP 200 | Passed |
| Playback manifest + plots HTTP 200 | Passed |
| Metrics 2.835 / 2.694 visible | Passed |
| “No model runs in the browser” | Passed |
| GitHub repository link | Passed |
| Sitemap canonical URLs | Passed |
| robots.txt sitemap pointer | Passed |
| Ordinary JSON precomputed predictions present | Passed |
| Interactive scenario/model controls present | Passed |
| No Python inference backend | Confirmed |

### Known fix applied after first deploy

The first live build briefly exposed a double base-path Open Graph image URL
(`.../Mini_JEPA/Mini_JEPA/plots/...`) because `metadataBase` already included the Pages
subpath while image paths also used `withBasePath`. Metadata was corrected to use the host
origin for `metadataBase` and compose asset paths with `withBasePath` only once.

Root `.gitignore` was also narrowed from `data/` / `artifacts/` to `/data/` / `/artifacts/`
before the initial commit so curated `website/public/data/*.json` files are published while
local datasets remain ignored.

## Not done in W2

- No GitHub Release tag
- No checkpoint/dataset bundle publication
- No training-code or historical artifact changes

## Remaining non-blocking notes

- Actions annotations warn that Node.js 20-targeted actions are deprecated on runners; deploy still succeeded.
- Exact historical checkpoint binaries remain local/gitignored by design.
