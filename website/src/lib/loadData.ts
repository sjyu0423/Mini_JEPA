import { readFile } from "node:fs/promises";
import path from "node:path";
import type { MetricsPayload, PlaybackManifest } from "@/lib/types";

const dataRoot = path.join(process.cwd(), "public", "data");

export async function loadMetrics(): Promise<MetricsPayload> {
  const raw = await readFile(path.join(dataRoot, "metrics.json"), "utf8");
  return JSON.parse(raw) as MetricsPayload;
}

export async function loadPlaybackManifest(): Promise<PlaybackManifest> {
  const raw = await readFile(path.join(dataRoot, "playback", "manifest.json"), "utf8");
  return JSON.parse(raw) as PlaybackManifest;
}
