#!/usr/bin/env node
import assert from "node:assert/strict";

function pixelError(prediction, actual, scale = 63) {
  const dx = (prediction[0] - actual[0]) * scale;
  const dy = (prediction[1] - actual[1]) * scale;
  return Math.hypot(dx, dy);
}

function predictionIsPending(frame, firstTargetFrame, hasPrediction) {
  if (frame < firstTargetFrame) return true;
  return !hasPrediction;
}

assert.equal(predictionIsPending(3, 4, false), true);
assert.equal(predictionIsPending(4, 4, true), false);
assert.ok(Math.abs(pixelError([0.5, 0.5], [0.5, 0.5]) - 0) < 1e-9);
assert.ok(Math.abs(pixelError([0.5, 0.5], [0.5 + 1 / 63, 0.5]) - 1) < 1e-6);
console.log("OK: playback helper interaction invariants");
