export function ArchitectureDiagram() {
  const stages = [
    { title: "Observed frames", body: "Four 64×64 grayscale images form the context window." },
    { title: "Context encoder", body: "Shared CNN embeds each frame; a temporal MLP or GRU aggregates them." },
    { title: "Future latent predictor", body: "Predicts the embedding of the next frame, not pixels." },
    { title: "Target encoder", body: "EMA copy of the context encoder supplies a stop-gradient latent target." },
    { title: "Latent loss", body: "Self-supervised objective matches prediction and target embeddings." },
    { title: "Frozen position probe", body: "A separately trained linear probe maps the predicted embedding to (x, y)." },
  ];

  return (
    <ol className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {stages.map((stage, index) => (
        <li key={stage.title} className="panel p-4">
          <p className="mono text-xs text-[color:var(--accent)]">{String(index + 1).padStart(2, "0")}</p>
          <h3 className="display mt-2 text-lg">{stage.title}</h3>
          <p className="lede mt-2 text-sm">{stage.body}</p>
        </li>
      ))}
    </ol>
  );
}
