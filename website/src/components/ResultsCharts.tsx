import type { MetricsPayload } from "@/lib/types";

export function ResultsCharts({ metrics }: { metrics: MetricsPayload }) {
  const chartRows = metrics.comparisons.filter((row) => row.id !== "random-encoder");
  const max = Math.max(
    ...chartRows.flatMap((row) => [row.test_rmse_px, row.ood_rmse_px ?? 0]),
    1,
  );

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="panel p-5">
        <p className="eyebrow">In-distribution test</p>
        <h3 className="display mt-2 text-xl">Position RMSE (px)</h3>
        <p className="lede mt-1 text-sm">Lower is better. Selected-seed rows are not multi-seed averages.</p>
        <ul className="mt-5 space-y-3">
          {chartRows.map((row) => (
            <BarRow
              key={`test-${row.id}`}
              label={row.label}
              value={row.test_rmse_px}
              sd={row.test_rmse_sd_px}
              max={max}
            />
          ))}
        </ul>
      </div>

      <div className="panel p-5">
        <p className="eyebrow">High-speed OOD</p>
        <h3 className="display mt-2 text-xl">Position RMSE (px)</h3>
        <p className="lede mt-1 text-sm">Same units and caveats. Physics and supervised baselines remain stronger.</p>
        <ul className="mt-5 space-y-3">
          {chartRows.map((row) => (
            <BarRow
              key={`ood-${row.id}`}
              label={row.label}
              value={row.ood_rmse_px}
              sd={row.ood_rmse_sd_px}
              max={max}
            />
          ))}
        </ul>
      </div>
    </div>
  );
}

function BarRow({
  label,
  value,
  sd,
  max,
}: {
  label: string;
  value: number | null;
  sd?: number;
  max: number;
}) {
  const width = value == null ? 0 : Math.max(4, (value / max) * 100);
  return (
    <li>
      <div className="mb-1 flex items-baseline justify-between gap-3 text-sm">
        <span>{label}</span>
        <span className="mono shrink-0 text-[color:var(--ink-muted)]">
          {value == null ? "—" : sd != null ? `${value.toFixed(3)} ± ${sd.toFixed(3)}` : value.toFixed(3)}
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-[color:var(--line)]/70" aria-hidden>
        <div
          className="h-2.5 rounded-full bg-[color:var(--accent)] transition-[width] duration-500"
          style={{ width: `${width}%` }}
        />
      </div>
    </li>
  );
}

export function BounceTable({ metrics }: { metrics: MetricsPayload }) {
  return (
    <div className="panel overflow-x-auto p-5">
      <p className="eyebrow">Failure analysis</p>
      <h3 className="display mt-2 text-xl">Bounce-rich diagnostics</h3>
      <p className="lede mt-1 text-sm">
        Independent diagnostic trajectories. Multiple-bounce remains difficult even after GRU
        temporal aggregation.
      </p>
      <table className="mt-5 w-full min-w-[640px] border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-[color:var(--line)] text-[color:var(--ink-muted)]">
            <th className="py-2 pr-3 font-medium">Condition</th>
            <th className="py-2 pr-3 font-medium">Bounce windows</th>
            <th className="py-2 pr-3 font-medium">M1 RMSE</th>
            <th className="py-2 pr-3 font-medium">M3 RMSE</th>
            <th className="py-2 pr-3 font-medium">M1 bounce mean</th>
            <th className="py-2 font-medium">M3 bounce mean</th>
          </tr>
        </thead>
        <tbody>
          {metrics.bounce_conditions.map((row) => (
            <tr key={row.condition} className="border-b border-[color:var(--line)]/70">
              <td className="py-2 pr-3 capitalize">{row.condition.replace("-", " ")}</td>
              <td className="mono py-2 pr-3">{row.bounce_windows}</td>
              <td className="mono py-2 pr-3">{row.m1_rmse_px.toFixed(3)}</td>
              <td className="mono py-2 pr-3">{row.m3_rmse_px.toFixed(3)}</td>
              <td className="mono py-2 pr-3">{row.m1_bounce_mean_px.toFixed(3)}</td>
              <td className="mono py-2">{row.m3_bounce_mean_px.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
