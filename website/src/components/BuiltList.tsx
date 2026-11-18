const items = [
  {
    title: "Synthetic bouncing-ball simulator",
    body: "Deterministic elastic physics with multi-collision support and 64×64 grayscale rendering.",
  },
  {
    title: "CNN context encoder",
    body: "Shared convolutional encoder over four observed frames.",
  },
  {
    title: "EMA target encoder",
    body: "Stop-gradient exponential-moving-average branch that supplies latent targets.",
  },
  {
    title: "Latent-space predictor",
    body: "Predicts the next-frame embedding; never reconstructs pixels during JEPA training.",
  },
  {
    title: "Temporal MLP and GRU experiments",
    body: "M1 concatenates embeddings through an MLP; M3 evaluates a GRU aggregator under controlled selection.",
  },
  {
    title: "Frozen position probes",
    body: "Downstream linear probes measure how readily future coordinates can be decoded. Distinct from the JEPA predictor.",
  },
  {
    title: "Baseline comparisons",
    body: "Random encoder, supervised CNN, constant-velocity, and wall-aware analytic physics references.",
  },
  {
    title: "Interactive local inference demo",
    body: "Real frozen checkpoints drive the M4 localhost demo; this site replays curated precomputed outputs.",
  },
];

export function BuiltList() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {items.map((item) => (
        <article key={item.title} className="border-l-2 border-[color:var(--accent)] pl-4">
          <h3 className="font-semibold">{item.title}</h3>
          <p className="lede mt-1 text-sm">{item.body}</p>
        </article>
      ))}
    </div>
  );
}
