export type ModelId = "m1-jepa" | "m3-jepa" | "constant-velocity";
export type ScenarioId = "ordinary" | "near-wall" | "high-speed" | "multiple-bounce";

export interface MetricsComparison {
  id: string;
  label: string;
  supervision: string;
  test_rmse_px: number;
  ood_rmse_px: number | null;
  test_rmse_sd_px?: number;
  ood_rmse_sd_px?: number;
  kind: string;
  seed?: number;
  family?: string;
  seeds?: number[];
}

export interface BounceCondition {
  condition: string;
  bounce_windows: number;
  m1_rmse_px: number;
  m3_rmse_px: number;
  m1_bounce_mean_px: number;
  m3_bounce_mean_px: number;
}

export interface MetricsPayload {
  export_version: string;
  generated_at_utc: string;
  note: string;
  sources: Record<string, { path: string; sha256: string }>;
  comparisons: MetricsComparison[];
  bounce_conditions: BounceCondition[];
  test_suite: { pytest_collected: number; note: string };
  caveats: string[];
  diagnostic_overall?: {
    m3_rmse_px: number;
    m3_bounce_mean_px: number;
    m3_non_bounce_mean_px: number;
  };
}

export interface PlaybackManifest {
  export_version: string;
  generated_at_utc: string;
  disclaimer: string;
  models: Array<{ id: ModelId; label: string; input: string }>;
  scenarios: Array<{
    id: ScenarioId;
    file: string;
    seed: number;
    frame_count: number;
    description: string;
    mean_error_px: Record<string, number>;
  }>;
}

export interface PlaybackScenario {
  export_version: string;
  disclaimer: string;
  scenario: ScenarioId;
  seed: number;
  description: string;
  frame_count: number;
  timestep: number;
  ball_radius: number;
  positions: Array<[number, number]>;
  collisions: boolean[];
  collision_counts: number[];
  predictions: Record<ModelId, Array<[number, number] | null>>;
  errors: Record<ModelId, Array<number | null>>;
  alignment: {
    observed_frames: number;
    horizon: number;
    first_target_frame: number;
    statement: string;
  };
  provenance: {
    generator: string;
    dataset_source: string;
    generation: Record<string, unknown>;
    checkpoints: Record<string, { path: string; sha256: string }>;
    latency_ms: Record<string, number | null>;
  };
}
