import type { Metadata } from "next";
import { withBasePath } from "@/lib/site";

export const metadata: Metadata = {
  title: "Engineering",
  description: "Simulator, reproducibility, testing, and website data export for Mini-JEPA.",
};

export default function EngineeringPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-10 sm:px-6">
      <p className="eyebrow">Engineering</p>
      <h1 className="display mt-2 text-4xl font-semibold">Simulator, reproducibility, inference</h1>
      <p className="lede mt-4 max-w-3xl text-lg">
        The research code is a compact Python package covering simulation, training, evaluation,
        and a localhost demo. This website is a static presentation layer over curated exports.
      </p>

      <section className="mt-10 grid gap-4 md:grid-cols-2">
        <article className="panel p-6">
          <h2 className="display text-2xl">Simulator</h2>
          <p className="lede mt-3 text-sm">
            One ball, axis-aligned walls, constant velocity, and elastic reflections with support
            for multiple wall crossings in a single timestep. Rendering is separated from physics
            so datasets remain leakage-resistant.
          </p>
        </article>
        <article className="panel p-6">
          <h2 className="display text-2xl">Reproducibility classes</h2>
          <p className="lede mt-3 text-sm">
            Source behavior is reproducible from the repository. Neural training is seeded but
            hardware-sensitive on Apple MPS. Exact published tables and interactive predictions are
            artifact-dependent and hash-verified.
          </p>
        </article>
        <article className="panel p-6">
          <h2 className="display text-2xl">Testing</h2>
          <p className="lede mt-3 text-sm">
            The M0–M5 suite collects 45 pytest cases spanning physics, leakage, alignment,
            checkpoint loading, demo context slicing, and release documentation checks. Website
            exports validate frame/prediction alignment before publication.
          </p>
        </article>
        <article className="panel p-6">
          <h2 className="display text-2xl">Inference surfaces</h2>
          <p className="lede mt-3 text-sm">
            <span className="mono">mini-jepa-demo</span> loads frozen checkpoints locally and
            batch-infers horizon-1 predictions. The website never ships checkpoints; it only serves
            small JSON trajectories produced by{" "}
            <span className="mono">scripts/export_website_data.py</span>.
          </p>
        </article>
      </section>

      <section className="mt-10 panel p-6">
        <h2 className="display text-2xl">Website data export</h2>
        <ol className="mt-4 list-decimal space-y-2 pl-5 text-sm text-[color:var(--ink-muted)]">
          <li>Verify SHA-256 hashes for M1/M3 checkpoints, probes, and result JSON.</li>
          <li>Run deterministic scenario generation and genuine frozen-model inference.</li>
          <li>Validate that predictions begin only at the first target frame.</li>
          <li>Write compact JSON under <span className="mono">website/public/data/</span>.</li>
          <li>Copy selected plots; exclude datasets and checkpoint binaries from the static site.</li>
        </ol>
        <pre className="mono mt-5 overflow-x-auto rounded-xl bg-[#12161c] p-4 text-xs text-[#e8edf2]">
{`uv run python scripts/export_website_data.py --device auto --frames 120
cd website && npm run validate:data && npm run build`}
        </pre>
      </section>

      <figure className="panel mt-10 overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={withBasePath("/plots/m4-demo-desktop.png")}
          alt="M4 desktop demo screenshot"
          className="w-full"
        />
        <figcaption className="border-t border-[color:var(--line)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
          Local M4 demo screenshot. The research website reuses the same checkpoint provenance with
          precomputed playback JSON.
        </figcaption>
      </figure>
    </div>
  );
}
