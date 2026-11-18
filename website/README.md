# Mini-JEPA research website

Static Next.js site for the Mini-JEPA research portfolio.

## Local development

```bash
# From repository root, refresh curated JSON/plots when checkpoints change:
uv run python scripts/export_website_data.py --device auto --frames 120

cd website
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Quality checks

```bash
npm run test
npm run lint
npm run typecheck
npm run build
```

Static files are written to `website/out/`.

## GitHub Pages base path

```bash
NEXT_PUBLIC_BASE_PATH=/YourRepoName npm run build
```

Leave `NEXT_PUBLIC_BASE_PATH` empty for local root hosting. See `docs/website/implementation.md`.
