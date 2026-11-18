import type { Metadata } from "next";
import { BounceTable, ResultsCharts } from "@/components/ResultsCharts";
import { loadMetrics } from "@/lib/loadData";
import { withBasePath } from "@/lib/site";

export const metadata: Metadata = {
  title: "Experiments",
  description: "Baselines, generalization, and failure analysis for Mini-JEPA.",
};

export default async function ExperimentsPage() {
  const metrics = await loadMetrics();
  const mean = metrics.comparisons.find((row) => row.id === "m3-gru-mean")!;
  const selected = metrics.comparisons.find((row) => row.id === "m3-gru-selected")!;
  const m1 = metrics.comparisons.find((row) => row.id === "m1-jepa")!;

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-10 sm:px-6">
      <p className="eyebrow">Experiments</p>
      <h1 className="display mt-2 text-4xl font-semibold">Baselines, evaluation, generalization</h1>
      <p className="lede mt-4 max-w-3xl text-lg">
        All numbers below are copied from audited M2/M3 result JSON. Comparisons are intentionally
        supervision-aware: analytic baselines see measured positions; JEPA sees images.
      </p>

      <section className="mt-10">
        <ResultsCharts metrics={metrics} />
      </section>

      <section className="mt-10 panel overflow-x-auto p-5">
        <h2 className="display text-2xl">Comparison table</h2>
        <table className="mt-5 w-full min-w-[720px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-[color:var(--line)] text-[color:var(--ink-muted)]">
              <th className="py-2 pr-3 font-medium">Method</th>
              <th className="py-2 pr-3 font-medium">Kind</th>
              <th className="py-2 pr-3 font-medium">Test RMSE</th>
              <th className="py-2 pr-3 font-medium">OOD RMSE</th>
              <th className="py-2 font-medium">Supervision note</th>
            </tr>
          </thead>
          <tbody>
            {metrics.comparisons.map((row) => (
              <tr key={row.id} className="border-b border-[color:var(--line)]/70 align-top">
                <td className="py-3 pr-3">{row.label}</td>
                <td className="py-3 pr-3 capitalize">{row.kind.replace("-", " ")}</td>
                <td className="mono py-3 pr-3">
                  {row.test_rmse_sd_px != null
                    ? `${row.test_rmse_px.toFixed(3)} ± ${row.test_rmse_sd_px.toFixed(3)}`
                    : row.test_rmse_px.toFixed(3)}
                </td>
                <td className="mono py-3 pr-3">
                  {row.ood_rmse_px == null
                    ? "—"
                    : row.ood_rmse_sd_px != null
                      ? `${row.ood_rmse_px.toFixed(3)} ± ${row.ood_rmse_sd_px.toFixed(3)}`
                      : row.ood_rmse_px.toFixed(3)}
                </td>
                <td className="py-3 text-[color:var(--ink-muted)]">{row.supervision}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-2">
        <article className="panel p-6">
          <h2 className="display text-2xl">Selected seed versus multi-seed mean</h2>
          <p className="lede mt-3 text-sm">
            Seed 502 is the named selected checkpoint ({selected.test_rmse_px.toFixed(3)} /{" "}
            {selected.ood_rmse_px?.toFixed(3)} px). The GRU family mean is{" "}
            {mean.test_rmse_px.toFixed(3)} ± {mean.test_rmse_sd_px?.toFixed(3)} /{" "}
            {mean.ood_rmse_px?.toFixed(3)} ± {mean.ood_rmse_sd_px?.toFixed(3)} px. That mean does not
            beat historical M1 in-distribution ({m1.test_rmse_px.toFixed(3)} px), though OOD
            improves consistently.
          </p>
        </article>
        <article className="panel p-6">
          <h2 className="display text-2xl">Where JEPA struggled</h2>
          <p className="lede mt-3 text-sm">
            Collision-heavy motion is the principal failure mode. Multiple-bounce diagnostic RMSE
            falls from 14.241 px (M1) to 7.745 px (selected M3), which is a large improvement and
            still far from solved. Ordinary and near-wall bounce-only means did not uniformly
            improve.
          </p>
        </article>
      </section>

      <section className="mt-10">
        <BounceTable metrics={metrics} />
      </section>

      <section className="mt-10 grid gap-6 md:grid-cols-2">
        <figure className="panel overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={withBasePath("/plots/m2-model-comparison.png")}
            alt="M2 model comparison plot"
            className="w-full"
          />
          <figcaption className="border-t border-[color:var(--line)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
            M2 model comparison using audited experiment plots.
          </figcaption>
        </figure>
        <figure className="panel overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={withBasePath("/plots/m3-bounce-diagnostic.png")}
            alt="M3 bounce diagnostic plot"
            className="w-full"
          />
          <figcaption className="border-t border-[color:var(--line)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
            M3 bounce-rich diagnostic visualization.
          </figcaption>
        </figure>
      </section>

      <section className="mt-10 panel p-6">
        <h2 className="display text-2xl">Provenance</h2>
        <ul className="mt-4 space-y-2 text-sm text-[color:var(--ink-muted)]">
          <li className="mono">
            {metrics.sources.m2_results.path} · sha256 {metrics.sources.m2_results.sha256}
          </li>
          <li className="mono">
            {metrics.sources.m3_results.path} · sha256 {metrics.sources.m3_results.sha256}
          </li>
          <li>Export generated at {metrics.generated_at_utc} (version {metrics.export_version}).</li>
        </ul>
      </section>
    </div>
  );
}
