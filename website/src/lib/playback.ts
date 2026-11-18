/** Pure helpers for playback UI and validation. */

export function pixelError(
  prediction: [number, number],
  actual: [number, number],
  scale = 63,
): number {
  const dx = (prediction[0] - actual[0]) * scale;
  const dy = (prediction[1] - actual[1]) * scale;
  return Math.hypot(dx, dy);
}

export function visibleGroundTruthCount(frame: number): number {
  return Math.max(0, frame + 1);
}

export function predictionIsPending(
  frame: number,
  firstTargetFrame: number,
  hasPrediction: boolean,
): boolean {
  if (frame < firstTargetFrame) return true;
  return !hasPrediction;
}
