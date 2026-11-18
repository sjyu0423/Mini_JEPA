import type { Metadata } from "next";
import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";
import { withBasePath } from "@/lib/site";

export const metadata: Metadata = {
  title: "Architecture",
  description: "Mini-JEPA model architecture, EMA training, and frozen probe evaluation.",
};

export default function ArchitecturePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-10 sm:px-6">
      <p className="eyebrow">Under the hood</p>
      <h1 className="display mt-2 text-4xl font-semibold">Model architecture and training</h1>
      <p className="lede mt-4 max-w-3xl text-lg">
        Mini-JEPA is a small joint-embedding predictive architecture. It observes four rendered
        frames, predicts the latent of a future frame, and only later uses a frozen linear probe to
        read out coordinates.
      </p>

      <div className="mt-10">
        <ArchitectureDiagram />
      </div>

      <section className="mt-12 grid gap-6 lg:grid-cols-2">
        <article className="panel p-6">
          <h2 className="display text-2xl">Context path</h2>
          <p className="lede mt-3 text-sm">
            Each 64×64 grayscale frame passes through a shared CNN encoder. M1 concatenates the four
            embeddings and processes them with an MLP. The selected M3 model feeds the sequence to a
            GRU and projects the final hidden state. A predictor head then estimates the next-frame
            embedding.
          </p>
        </article>
        <article className="panel p-6">
          <h2 className="display text-2xl">Target path</h2>
          <p className="lede mt-3 text-sm">
            The target encoder is an exponential-moving-average copy of the context encoder. Gradients
            do not enter the target branch. Training compares predicted and target embeddings with a
            cosine-style objective plus variance regularization; pixels are never reconstructed, and
            simulator positions are unavailable to JEPA.
          </p>
        </article>
      </section>

      <section className="mt-12 panel p-6">
        <h2 className="display text-2xl">Predictor versus probe</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-[color:var(--line)] bg-white/70 p-4">
            <h3 className="font-semibold">JEPA predictor</h3>
            <p className="lede mt-2 text-sm">
              Trained self-supervised to predict future latents from images only. This is the
              representation-learning model.
            </p>
          </div>
          <div className="rounded-xl border border-[color:var(--line)] bg-white/70 p-4">
            <h3 className="font-semibold">Position probe</h3>
            <p className="lede mt-2 text-sm">
              Trained afterward on frozen embeddings with coordinate labels. It measures
              decodability; it is not part of the JEPA objective.
            </p>
          </div>
        </div>
      </section>

      <section className="mt-12">
        <p className="eyebrow">Optional deep dive</p>
        <h2 className="display mt-2 text-3xl">Loss, EMA, data, evaluation</h2>
        <div className="mt-6 grid gap-4">
          <details className="panel p-5">
            <summary className="cursor-pointer font-semibold">Loss function</summary>
            <p className="lede mt-3 text-sm">
              The primary objective is cosine distance between predicted and target embeddings,
              with a variance regularizer that discourages dimensional collapse. M3 also tested a
              normalized Smooth-L1 regression variant that, in the observed regime, behaved nearly
              identically to cosine geometry.
            </p>
          </details>
          <details className="panel p-5">
            <summary className="cursor-pointer font-semibold">EMA updates</summary>
            <p className="lede mt-3 text-sm">
              Target parameters track context parameters with decay 0.99. Because the target branch
              is stop-gradient, the predictor cannot collapse by jointly drifting both sides of the
              loss in an unconstrained way.
            </p>
          </details>
          <details className="panel p-5">
            <summary className="cursor-pointer font-semibold">Data pipeline</summary>
            <p className="lede mt-3 text-sm">
              Complete trajectories are assigned to exactly one split. Compressed NPZ files store
              frames, positions, velocities, collisions, and trajectory IDs, while the model-facing
              dataset returns rendered observations and targets without exposing physical state.
            </p>
          </details>
          <details className="panel p-5">
            <summary className="cursor-pointer font-semibold">Evaluation methodology</summary>
            <p className="lede mt-3 text-sm">
              Encoders and predictors are frozen. Lightweight probes are fit on training windows and
              scored on held-out test, high-speed OOD, and independent bounce diagnostics. M3
              selected architecture families using validation probe RMSE only, then reported
              multi-seed held-out metrics.
            </p>
          </details>
        </div>
      </section>

      <figure className="panel mt-12 overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={withBasePath("/plots/m3-representation-diagnostics.png")}
          alt="Representation diagnostics plot"
          className="w-full"
        />
        <figcaption className="border-t border-[color:var(--line)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
          Representation diagnostics from the audited M3 artifact set.
        </figcaption>
      </figure>
    </div>
  );
}
