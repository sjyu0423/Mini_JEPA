"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { withBasePath } from "@/lib/site";
import type { ModelId, PlaybackManifest, PlaybackScenario, ScenarioId } from "@/lib/types";

const MODEL_LABELS: Record<ModelId, string> = {
  "m3-jepa": "M3 JEPA · GRU",
  "m1-jepa": "M1 JEPA · temporal MLP",
  "constant-velocity": "Constant velocity · measured positions",
};

export function DemoPlayer() {
  const [manifest, setManifest] = useState<PlaybackManifest | null>(null);
  const [scenarioId, setScenarioId] = useState<ScenarioId>("ordinary");
  const [modelId, setModelId] = useState<ModelId>("m3-jepa");
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(withBasePath("/data/playback/manifest.json"));
        if (!response.ok) throw new Error(`Failed to load manifest (${response.status})`);
        const data = (await response.json()) as PlaybackManifest;
        if (!cancelled) setManifest(data);
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : "Load failed");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="panel grid gap-0 overflow-hidden lg:grid-cols-[280px_minmax(0,1fr)]">
      <aside className="border-b border-[color:var(--line)] p-5 lg:border-b-0 lg:border-r">
        <p className="eyebrow">Interactive demonstration</p>
        <h2 className="display mt-2 text-2xl">Playback lab</h2>
        <p className="lede mt-2 text-sm">
          Precomputed predictions from trained PyTorch checkpoints. No model runs in the browser.
        </p>

        <label className="mt-5 block text-sm font-medium">
          Scenario
          <select
            className="focus-ring mt-1 w-full rounded-lg border border-[color:var(--line)] bg-white px-3 py-2"
            value={scenarioId}
            onChange={(event) => setScenarioId(event.target.value as ScenarioId)}
          >
            <option value="ordinary">Ordinary</option>
            <option value="near-wall">Near wall</option>
            <option value="high-speed">High speed</option>
            <option value="multiple-bounce">Multiple bounce</option>
          </select>
        </label>

        <label className="mt-3 block text-sm font-medium">
          Predictor
          <select
            className="focus-ring mt-1 w-full rounded-lg border border-[color:var(--line)] bg-white px-3 py-2"
            value={modelId}
            onChange={(event) => setModelId(event.target.value as ModelId)}
          >
            {(Object.keys(MODEL_LABELS) as ModelId[]).map((id) => (
              <option key={id} value={id}>
                {MODEL_LABELS[id]}
              </option>
            ))}
          </select>
        </label>

        <p className="mt-4 text-xs text-[color:var(--ink-muted)]">
          Keyboard: Space play/pause · ←/→ step · R restart. Ground truth is revealed only as
          playback reaches each frame. Predictions are independent horizon-1 steps, not an
          autoregressive rollout.
        </p>
        {manifest && (
          <p className="mt-3 text-xs text-[color:var(--ink-muted)]">{manifest.disclaimer}</p>
        )}
        {loadError && <p className="mt-3 text-sm text-[color:var(--collision)]">{loadError}</p>}
      </aside>

      <ScenarioStage key={scenarioId} scenarioId={scenarioId} modelId={modelId} manifest={manifest} />
    </div>
  );
}

function ScenarioStage({
  scenarioId,
  modelId,
  manifest,
}: {
  scenarioId: ScenarioId;
  modelId: ModelId;
  manifest: PlaybackManifest | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scenario, setScenario] = useState<PlaybackScenario | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showTrail, setShowTrail] = useState(true);
  const playingRef = useRef(false);

  useEffect(() => {
    if (!manifest) return;
    let cancelled = false;
    (async () => {
      try {
        const entry = manifest.scenarios.find((item) => item.id === scenarioId);
        if (!entry) throw new Error(`Unknown scenario ${scenarioId}`);
        const response = await fetch(withBasePath(`/data/${entry.file}`));
        if (!response.ok) throw new Error(`Failed to load ${entry.file}`);
        const data = (await response.json()) as PlaybackScenario;
        if (!cancelled) {
          setScenario(data);
          setLoadError(null);
          setLoading(false);
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : "Load failed");
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [manifest, scenarioId]);

  useEffect(() => {
    if (!playing || !scenario) return;
    const interval = window.setInterval(() => {
      setFrame((current) => {
        if (current >= scenario.frame_count - 1) {
          playingRef.current = false;
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, Math.max(16, scenario.timestep * 1000) / speed);
    return () => window.clearInterval(interval);
  }, [playing, scenario, speed]);

  const realizedError = useMemo(() => {
    if (!scenario) return null;
    return scenario.errors[modelId][frame] ?? null;
  }, [scenario, modelId, frame]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const data = scenario;
    if (!canvas || !data) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const size = Math.min(rect.width, rect.height);
    const offsetX = (rect.width - size) / 2;
    const offsetY = (rect.height - size) / 2;

    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#f7fafb";
    ctx.fillRect(offsetX, offsetY, size, size);
    ctx.strokeStyle = "#c9d3dc";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(offsetX + 0.5, offsetY + 0.5, size - 1, size - 1);

    const toCanvas = (pos: [number, number]) => ({
      x: offsetX + pos[0] * size,
      y: offsetY + pos[1] * size,
    });
    const ballR = data.ball_radius * size;

    if (showTrail) {
      ctx.beginPath();
      for (let i = 0; i <= frame; i += 1) {
        const point = toCanvas(data.positions[i]);
        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      }
      ctx.strokeStyle = "rgba(31, 111, 235, 0.35)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    const actual = toCanvas(data.positions[frame]);
    if (data.collisions[frame]) {
      ctx.beginPath();
      ctx.strokeStyle = "#b42318";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.arc(actual.x, actual.y, ballR + 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.fillStyle = "#1f6feb";
    ctx.arc(actual.x, actual.y, ballR, 0, Math.PI * 2);
    ctx.fill();

    const firstTarget = data.alignment.first_target_frame;
    if (frame >= firstTarget - 1 && frame + 1 < data.frame_count) {
      const nextPrediction = data.predictions[modelId][frame + 1];
      if (nextPrediction) {
        const pending = toCanvas(nextPrediction);
        ctx.beginPath();
        ctx.strokeStyle = "#c45c26";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.arc(pending.x, pending.y, ballR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#c45c26";
        ctx.font = "12px IBM Plex Sans, sans-serif";
        ctx.fillText("next pred.", pending.x + 10, pending.y - 10);
      }
    }

    const prediction = data.predictions[modelId][frame];
    if (prediction) {
      const realized = toCanvas(prediction);
      ctx.beginPath();
      ctx.strokeStyle = "#0f7a4c";
      ctx.lineWidth = 2;
      ctx.moveTo(realized.x - 7, realized.y - 7);
      ctx.lineTo(realized.x + 7, realized.y + 7);
      ctx.moveTo(realized.x + 7, realized.y - 7);
      ctx.lineTo(realized.x - 7, realized.y + 7);
      ctx.stroke();
    }
  }, [scenario, frame, modelId, showTrail]);

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  const errorHistory = useMemo(() => {
    if (!scenario) return [] as Array<number | null>;
    return scenario.errors[modelId].slice(0, frame + 1);
  }, [scenario, modelId, frame]);

  const togglePlay = () => {
    if (!scenario) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setFrame((current) => Math.min(scenario.frame_count - 1, current + 1));
      return;
    }
    setPlaying((value) => {
      playingRef.current = !value;
      return !value;
    });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === " " || event.key === "k") {
      event.preventDefault();
      togglePlay();
    } else if (event.key === "ArrowRight" || event.key === ".") {
      event.preventDefault();
      setPlaying(false);
      setFrame((current) => Math.min((scenario?.frame_count ?? 1) - 1, current + 1));
    } else if (event.key === "ArrowLeft" || event.key === ",") {
      event.preventDefault();
      setPlaying(false);
      setFrame((current) => Math.max(0, current - 1));
    } else if (event.key === "Home" || event.key === "r") {
      event.preventDefault();
      setPlaying(false);
      setFrame(0);
    }
  };

  return (
    <div className="p-5" tabIndex={0} onKeyDown={onKeyDown} aria-label="Interactive prediction playback">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Arena</p>
          <h3 className="display text-xl capitalize">{scenarioId.replace("-", " ")}</h3>
          <p className="text-sm text-[color:var(--ink-muted)]">
            {scenario?.description ?? "Loading curated trajectory…"}
          </p>
        </div>
        <div className="rounded-full bg-[color:var(--accent-soft)] px-3 py-1 text-xs font-medium text-[color:var(--accent)]">
          {scenario?.collisions[frame] ? "Collision frame" : "No collision"}
        </div>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        <button
          type="button"
          className="focus-ring rounded-full bg-[color:var(--ink)] px-4 py-2 text-sm font-medium text-white"
          onClick={togglePlay}
          disabled={!scenario || loading}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          className="focus-ring rounded-full border border-[color:var(--line)] bg-white px-3 py-2 text-sm"
          onClick={() => {
            setPlaying(false);
            setFrame((current) => Math.min((scenario?.frame_count ?? 1) - 1, current + 1));
          }}
          disabled={!scenario}
        >
          Step
        </button>
        <button
          type="button"
          className="focus-ring rounded-full border border-[color:var(--line)] bg-white px-3 py-2 text-sm"
          onClick={() => {
            setPlaying(false);
            setFrame(0);
          }}
          disabled={!scenario}
        >
          Restart
        </button>
      </div>

      <label className="mb-3 block max-w-xs text-sm">
        Speed {speed.toFixed(2)}×
        <input
          className="mt-2 w-full"
          type="range"
          min={0.25}
          max={3}
          step={0.25}
          value={speed}
          onChange={(event) => setSpeed(Number(event.target.value))}
        />
      </label>

      <label className="mb-4 flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={showTrail}
          onChange={(event) => setShowTrail(event.target.checked)}
        />
        Show observed trail
      </label>

      <div className="relative aspect-square w-full max-w-[640px]">
        <canvas ref={canvasRef} className="h-full w-full rounded-xl border border-[color:var(--line)]" />
        {(loading || loadError) && (
          <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-white/80 text-sm">
            {loadError ?? "Loading precomputed predictions…"}
          </div>
        )}
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-4">
        <Stat label="Frame" value={scenario ? `${frame} / ${scenario.frame_count - 1}` : "—"} />
        <Stat
          label="Observed (x, y)"
          value={
            scenario
              ? `${scenario.positions[frame][0].toFixed(3)}, ${scenario.positions[frame][1].toFixed(3)}`
              : "—"
          }
        />
        <Stat
          label="Realized prediction"
          value={
            scenario?.predictions[modelId][frame]
              ? `${scenario.predictions[modelId][frame]![0].toFixed(3)}, ${scenario.predictions[modelId][frame]![1].toFixed(3)}`
              : "pending"
          }
        />
        <Stat
          label="Error"
          value={realizedError == null ? "pending" : `${realizedError.toFixed(2)} px`}
        />
      </div>

      <ErrorSparkline values={errorHistory} />

      <div className="mt-4 flex flex-wrap gap-4 text-xs text-[color:var(--ink-muted)]">
        <Legend color="#1f6feb" label="Observed position" />
        <Legend color="#c45c26" label="Next-frame prediction" dashed />
        <Legend color="#0f7a4c" label="Realized prediction" cross />
        <Legend color="#b42318" label="Collision" />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-white/70 px-3 py-2">
      <p className="text-[11px] uppercase tracking-[0.12em] text-[color:var(--ink-muted)]">{label}</p>
      <p className="mono mt-1 text-sm">{value}</p>
    </div>
  );
}

function Legend({
  color,
  label,
  dashed,
  cross,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  cross?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <span
        className="inline-block h-3 w-3 rounded-full border-2"
        style={{
          borderColor: color,
          background: cross || dashed ? "transparent" : color,
          borderStyle: dashed ? "dashed" : "solid",
        }}
        aria-hidden
      />
      {label}
    </span>
  );
}

function ErrorSparkline({ values }: { values: Array<number | null> }) {
  const width = 320;
  const height = 64;
  const numeric = values.filter((value): value is number => value != null);
  const max = Math.max(1, ...numeric);
  const points = values
    .map((value, index) => {
      if (value == null) return null;
      const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * (width - 8) + 4;
      const y = height - 6 - (value / max) * (height - 16);
      return `${x},${y}`;
    })
    .filter(Boolean)
    .join(" ");

  return (
    <div className="mt-4 rounded-xl border border-[color:var(--line)] bg-white/70 p-3">
      <div className="mb-2 flex items-center justify-between text-xs text-[color:var(--ink-muted)]">
        <span>Prediction error history (px)</span>
        <span className="mono">max {max.toFixed(2)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-16 w-full" role="img" aria-label="Error sparkline">
        <polyline fill="none" stroke="#0d6b6e" strokeWidth="2" points={points} />
      </svg>
    </div>
  );
}
