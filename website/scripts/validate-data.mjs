#!/usr/bin/env node
/**
 * Validate exported website playback/metrics JSON without requiring a browser.
 */
import { readFileSync, readdirSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const dataDir = path.join(root, "public", "data");
const playbackDir = path.join(dataDir, "playback");
let failures = 0;

function fail(message) {
  console.error(`FAIL: ${message}`);
  failures += 1;
}

function ok(message) {
  console.log(`OK: ${message}`);
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function assert(condition, message) {
  if (!condition) fail(message);
}

const metricsPath = path.join(dataDir, "metrics.json");
assert(existsSync(metricsPath), "metrics.json missing");
const metrics = readJson(metricsPath);
assert(metrics.export_version, "metrics.export_version missing");
assert(Array.isArray(metrics.comparisons) && metrics.comparisons.length >= 5, "comparisons incomplete");

const expected = {
  "m1-jepa": { test: 2.835, ood: 5.391 },
  "m3-gru-selected": { test: 2.694, ood: 3.578 },
  "supervised-cnn": { test: 0.65, ood: 1.247 },
  "wall-aware-physics": { test: 0.111, ood: 0.312 },
};

for (const [id, values] of Object.entries(expected)) {
  const row = metrics.comparisons.find((item) => item.id === id);
  if (!row) {
    fail(`missing comparison ${id}`);
    continue;
  }
  assert(Math.abs(row.test_rmse_px - values.test) < 0.001, `${id} test RMSE mismatch`);
  assert(Math.abs(row.ood_rmse_px - values.ood) < 0.001, `${id} OOD RMSE mismatch`);
}
ok("metrics match audited headline numbers");

const manifestPath = path.join(playbackDir, "manifest.json");
assert(existsSync(manifestPath), "playback manifest missing");
const manifest = readJson(manifestPath);
assert(manifest.disclaimer.includes("No model runs in the browser"), "disclaimer missing");
assert(manifest.scenarios.length === 4, "expected 4 scenarios");

for (const entry of manifest.scenarios) {
  const filePath = path.join(dataDir, entry.file);
  if (!existsSync(filePath)) {
    fail(`missing ${entry.file}`);
    continue;
  }
  const scenario = readJson(filePath);
  assert(scenario.frame_count === scenario.positions.length, `${entry.id} position length`);
  assert(scenario.alignment.first_target_frame === 4, `${entry.id} first target`);
  for (const [model, series] of Object.entries(scenario.predictions)) {
    assert(series.length === scenario.frame_count, `${entry.id}/${model} prediction length`);
    for (let i = 0; i < 4; i += 1) {
      assert(series[i] == null, `${entry.id}/${model} leaked prediction at frame ${i}`);
    }
    for (let i = 4; i < series.length; i += 1) {
      assert(Array.isArray(series[i]) && series[i].length === 2, `${entry.id}/${model} missing at ${i}`);
      const error = scenario.errors[model][i];
      assert(typeof error === "number", `${entry.id}/${model} missing error at ${i}`);
      // Cross-check first available error against recomputed Euclidean * 63.
      if (i === 4) {
        const pred = series[i];
        const actual = scenario.positions[i];
        const recomputed = Math.hypot((pred[0] - actual[0]) * 63, (pred[1] - actual[1]) * 63);
        assert(Math.abs(recomputed - error) < 1e-3, `${entry.id}/${model} error mismatch at frame 4`);
      }
    }
    const machinePath = JSON.stringify(scenario).match(/\/Users\/|\/Volumes\/Projects\//);
    assert(!machinePath, `${entry.id} contains machine-specific path`);
  }
  ok(`${entry.id}: aligned predictions validated`);
}

const plotDir = path.join(root, "public", "plots");
if (!existsSync(plotDir)) {
  fail("public/plots missing");
} else {
  const plots = readdirSync(plotDir).filter((name) => name.endsWith(".png"));
  assert(plots.length >= 5, "expected curated plots");
  ok(`${plots.length} plots present`);
}

if (failures > 0) {
  console.error(`\n${failures} validation failure(s)`);
  process.exit(1);
}
console.log("\nAll playback/metrics validations passed.");
