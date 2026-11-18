import Link from "next/link";
import { BuiltList } from "@/components/BuiltList";
import { DemoPlayer } from "@/components/DemoPlayer";
import { HeroBall } from "@/components/HeroBall";
import { BounceTable, ResultsCharts } from "@/components/ResultsCharts";
import { loadMetrics } from "@/lib/loadData";
import { siteConfig, withBasePath } from "@/lib/site";

export default async function HomePage() {
  const metrics = await loadMetrics();
  const m1 = metrics.comparisons.find((row) => row.id === "m1-jepa")!;
  const m3 = metrics.comparisons.find((row) => row.id === "m3-gru-selected")!;
  const supervised = metrics.comparisons.find((row) => row.id === "supervised-cnn")!;
  const physics = metrics.comparisons.find((row) => row.id === "wall-aware-physics")!;

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-10 sm:px-6">
      <section className="grid items-center gap-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(280px,0.95fr)]">
        <div>
          <p className="eyebrow">Self-supervised visual motion</p>
          <h1 className="display mt-3 text-5xl font-semibold tracking-tight sm:text-6xl">
            {siteConfig.title}
          </h1>
          <p className="display mt-4 max-w-xl text-2xl text-[color:var(--ink-muted)] sm:text-3xl">
            {siteConfig.subtitle}
          </p>
          <p className="lede mt-5 max-w-2xl text-base sm:text-lg">
            Mini-JEPA asks whether a joint-embedding predictive architecture can learn useful
            representations of motion from bouncing-ball video—without reconstructing pixels or
            receiving physical-state labels during self-supervised training.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <a
              href="#demo"
              className="focus-ring rounded-full bg-[color:var(--ink)] px-5 py-2.5 text-sm font-medium text-white"
            >
              Interactive results
            </a>
            <Link
              href="/architecture/"
              className="focus-ring rounded-full border border-[color:var(--line)] bg-white px-5 py-2.5 text-sm font-medium"
            >
              Technical explanation
            </Link>
            <Link
              href="/experiments/"
              className="focus-ring rounded-full border border-[color:var(--line)] bg-white px-5 py-2.5 text-sm font-medium"
            >
              Experiment results
            </Link>
            <a
              href={siteConfig.githubUrl}
              className="focus-ring rounded-full border border-[color:var(--line)] bg-white px-5 py-2.5 text-sm font-medium"
              rel="noreferrer"
              target="_blank"
            >
              GitHub
            </a>
          </div>
        </div>
        <div className="panel aspect-square overflow-hidden p-3">
          <HeroBall />
        </div>
      </section>

      <section className="mt-16">
        <p className="eyebrow">What I built</p>
        <h2 className="display mt-2 text-3xl">End-to-end research system</h2>
        <p className="lede mt-3 max-w-3xl">
          The trained JEPA predictor estimates a future latent. A separately trained frozen
          coordinate probe only appears later to make that representation measurable. They are not
          the same module.
        </p>
        <div className="mt-8">
          <BuiltList />
        </div>
      </section>

      <section id="demo" className="mt-16 scroll-mt-24">
        <DemoPlayer />
      </section>

      <section className="mt-16">
        <p className="eyebrow">Results</p>
        <h2 className="display mt-2 text-3xl">Measured position error</h2>
        <p className="lede mt-3 max-w-3xl">
          Selected M3 GRU improves the historical M1 point estimate from {m1.test_rmse_px.toFixed(3)}{" "}
          → {m3.test_rmse_px.toFixed(3)} px in-distribution and {m1.ood_rmse_px?.toFixed(3)} →{" "}
          {m3.ood_rmse_px?.toFixed(3)} px OOD. Supervised CNN ({supervised.test_rmse_px.toFixed(3)} px)
          and wall-aware physics ({physics.test_rmse_px.toFixed(3)} px) remain far stronger and use
          richer supervision.
        </p>
        <div className="mt-8">
          <ResultsCharts metrics={metrics} />
        </div>
        <ul className="mt-6 grid gap-2 text-sm text-[color:var(--ink-muted)] sm:grid-cols-2">
          {metrics.caveats.map((caveat) => (
            <li key={caveat} className="rounded-xl border border-[color:var(--line)] bg-white/60 px-3 py-2">
              {caveat}
            </li>
          ))}
        </ul>
        <div className="mt-8">
          <BounceTable metrics={metrics} />
        </div>
        <p className="mt-4 text-sm text-[color:var(--ink-muted)]">
          Automated suite through M5: {metrics.test_suite.pytest_collected} tests collected and
          passed in the release audit. Source metrics:{" "}
          <span className="mono">{metrics.sources.m2_results.path}</span> and{" "}
          <span className="mono">{metrics.sources.m3_results.path}</span>.
        </p>
      </section>

      <section className="mt-16 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="panel p-6">
          <p className="eyebrow">What I learned</p>
          <h2 className="display mt-2 text-3xl">Honest conclusions</h2>
          <ul className="mt-5 space-y-3 text-sm leading-relaxed text-[color:var(--ink-muted)]">
            <li>
              Self-supervised latent prediction can produce representations far more informative
              than a random encoder without using simulator state as a training target.
            </li>
            <li>
              Low latent loss does not guarantee accurate physical prediction; position probes and
              hard collision subsets expose remaining gaps.
            </li>
            <li>
              Representation diagnostics showed high prediction–target cosine alignment without
              full collapse of normalized diversity or effective rank.
            </li>
            <li>
              Temporal GRU aggregation improved high-speed OOD robustness, while the three-seed
              in-distribution mean did not beat the single historical M1 run.
            </li>
            <li>
              Multiple-bounce sequences remain difficult, and strong baselines keep claims
              calibrated.
            </li>
          </ul>
        </div>
        <div className="panel overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={withBasePath("/plots/m3-heldout-comparison.png")}
            alt="Held-out comparison plot from M3 experiments"
            className="h-full w-full object-cover"
          />
        </div>
      </section>
    </div>
  );
}
