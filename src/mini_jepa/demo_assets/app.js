const state = {
  data: null,
  frame: 3,
  playing: false,
  speed: 1,
  model: "m3-jepa",
  lastTimestamp: 0,
  accumulator: 0,
};

const $ = (id) => document.getElementById(id);
const arena = $("arenaCanvas");
const arenaContext = arena.getContext("2d");
const errorCanvas = $("errorCanvas");
const errorContext = errorCanvas.getContext("2d");

const scenarioLabels = {
  ordinary: "Ordinary trajectory",
  "near-wall": "Near-wall launch",
  "high-speed": "High-speed trajectory",
  "multiple-bounce": "Multiple-bounce stress test",
};

function modelAccessNote(model) {
  if (model === "constant-velocity") {
    return "Constant velocity uses measured positions from the last two observed frames. This is privileged access that JEPA does not receive.";
  }
  return "JEPA receives only four rendered images. A frozen downstream probe converts its predicted embedding into position coordinates.";
}

async function loadStatus() {
  const response = await fetch("/api/status");
  const status = await response.json();
  $("deviceStatus").textContent = `${status.device.toUpperCase()} · checkpoints cached · ${status.load_latency_ms.toFixed(0)} ms load`;
}

async function loadScenario() {
  state.playing = false;
  $("playButton").textContent = "Play";
  $("loadingOverlay").classList.remove("hidden");
  const scenario = $("scenarioSelect").value;
  const seed = Number($("seedInput").value || 42);
  const response = await fetch(`/api/scenario?name=${encodeURIComponent(scenario)}&seed=${seed}`);
  const payload = await response.json();
  if (!response.ok) {
    $("loadingOverlay").textContent = payload.error || "Scenario failed";
    return;
  }
  state.data = payload;
  state.frame = 3;
  state.accumulator = 0;
  $("scenarioTitle").textContent = scenarioLabels[scenario];
  $("alignmentText").textContent = payload.alignment.statement;
  $("loadingOverlay").classList.add("hidden");
  draw();
}

function resizeCanvas(canvas, context) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}

function arenaPoint(position, rect, padding) {
  return [
    padding + position[0] * (rect.width - 2 * padding),
    padding + position[1] * (rect.height - 2 * padding),
  ];
}

function circle(context, x, y, radius, fill, stroke = null, lineWidth = 2) {
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  if (fill) { context.fillStyle = fill; context.fill(); }
  if (stroke) { context.strokeStyle = stroke; context.lineWidth = lineWidth; context.stroke(); }
}

function drawArena() {
  if (!state.data) return;
  const rect = resizeCanvas(arena, arenaContext);
  const width = rect.width;
  const height = rect.height;
  const padding = Math.max(24, width * 0.045);
  arenaContext.clearRect(0, 0, width, height);

  const gradient = arenaContext.createRadialGradient(width * 0.5, height * 0.45, 10, width * 0.5, height * 0.5, width * 0.65);
  gradient.addColorStop(0, "#10221d");
  gradient.addColorStop(1, "#040908");
  arenaContext.fillStyle = gradient;
  arenaContext.fillRect(0, 0, width, height);

  arenaContext.strokeStyle = state.data.collisions[state.frame] ? "#ff647c" : "#345249";
  arenaContext.lineWidth = state.data.collisions[state.frame] ? 4 : 2;
  arenaContext.strokeRect(padding, padding, width - 2 * padding, height - 2 * padding);

  if ($("trailToggle").checked) {
    const start = Math.max(0, state.frame - 36);
    arenaContext.beginPath();
    for (let index = start; index <= state.frame; index += 1) {
      const [x, y] = arenaPoint(state.data.positions[index], rect, padding);
      if (index === start) arenaContext.moveTo(x, y); else arenaContext.lineTo(x, y);
    }
    arenaContext.strokeStyle = "rgba(200,255,61,0.28)";
    arenaContext.lineWidth = 2;
    arenaContext.stroke();
  }

  const actual = state.data.positions[state.frame];
  const [actualX, actualY] = arenaPoint(actual, rect, padding);

  const realized = state.data.predictions[state.model][state.frame];
  if (realized) {
    const [rx, ry] = arenaPoint(realized, rect, padding);
    arenaContext.setLineDash([4, 5]);
    arenaContext.strokeStyle = "rgba(85,216,255,0.55)";
    arenaContext.lineWidth = 1.5;
    arenaContext.beginPath();
    arenaContext.moveTo(rx, ry);
    arenaContext.lineTo(actualX, actualY);
    arenaContext.stroke();
    arenaContext.setLineDash([]);
    arenaContext.strokeStyle = "#55d8ff";
    arenaContext.lineWidth = 3;
    arenaContext.beginPath();
    arenaContext.moveTo(rx - 7, ry - 7);
    arenaContext.lineTo(rx + 7, ry + 7);
    arenaContext.moveTo(rx - 7, ry + 7);
    arenaContext.lineTo(rx + 7, ry - 7);
    arenaContext.stroke();
  }

  const nextFrame = state.frame + 1;
  const nextPrediction = nextFrame < state.data.frame_count
    ? state.data.predictions[state.model][nextFrame]
    : null;
  if (nextPrediction) {
    const [nx, ny] = arenaPoint(nextPrediction, rect, padding);
    arenaContext.setLineDash([6, 5]);
    circle(arenaContext, nx, ny, Math.max(9, width * 0.018), null, "#ffb44c", 2.5);
    arenaContext.setLineDash([]);
  }

  arenaContext.shadowColor = "#c8ff3d";
  arenaContext.shadowBlur = 18;
  circle(arenaContext, actualX, actualY, Math.max(8, width * 0.016), "#c8ff3d");
  arenaContext.shadowBlur = 0;

  arenaContext.fillStyle = "#91aaa2";
  arenaContext.font = "10px ui-monospace, monospace";
  arenaContext.fillText(`t = ${state.frame}`, padding + 10, padding + 20);
}

function drawErrorChart() {
  const rect = resizeCanvas(errorCanvas, errorContext);
  errorContext.clearRect(0, 0, rect.width, rect.height);
  errorContext.fillStyle = "#07100f";
  errorContext.fillRect(0, 0, rect.width, rect.height);
  if (!state.data) return;
  const values = state.data.errors[state.model]
    .slice(Math.max(0, state.frame - 59), state.frame + 1)
    .filter((value) => value !== null);
  if (values.length < 2) return;
  const padding = 12;
  const maximum = Math.max(1, ...values) * 1.15;
  errorContext.beginPath();
  values.forEach((value, index) => {
    const x = padding + index / Math.max(values.length - 1, 1) * (rect.width - 2 * padding);
    const y = rect.height - padding - value / maximum * (rect.height - 2 * padding);
    if (index === 0) errorContext.moveTo(x, y); else errorContext.lineTo(x, y);
  });
  errorContext.strokeStyle = "#c8ff3d";
  errorContext.lineWidth = 2;
  errorContext.stroke();
  errorContext.fillStyle = "#587069";
  errorContext.font = "9px ui-monospace, monospace";
  errorContext.fillText(`${maximum.toFixed(1)}px`, 5, 11);
}

function formatPosition(position) {
  if (!position) return "waiting for context";
  return `${(position[0] * 63).toFixed(1)}, ${(position[1] * 63).toFixed(1)} px`;
}

function updateReadout() {
  if (!state.data) return;
  const nextFrame = state.frame + 1;
  const prediction = nextFrame < state.data.frame_count
    ? state.data.predictions[state.model][nextFrame]
    : null;
  const error = state.data.errors[state.model][state.frame];
  $("frameValue").textContent = `${state.frame} / ${state.data.frame_count - 1}`;
  $("actualValue").textContent = formatPosition(state.data.positions[state.frame]);
  $("predictedValue").textContent = formatPosition(prediction);
  $("targetValue").textContent = prediction ? `${nextFrame} · pending` : "end of trajectory";
  $("errorValue").textContent = error === null ? "—" : `${error.toFixed(2)} px`;
  const latencyKey = state.model === "m1-jepa" ? "m1_per_prediction_ms"
    : state.model === "m3-jepa" ? "m3_per_prediction_ms" : null;
  $("latencyValue").textContent = latencyKey
    ? state.data.latency[latencyKey].toFixed(2)
    : "analytic";
  const bounced = state.data.collisions[state.frame];
  $("bounceBadge").textContent = bounced ? "Bounce reached" : "No bounce";
  $("bounceBadge").classList.toggle("active", bounced);
  $("accessNote").textContent = modelAccessNote(state.model);
}

function draw() {
  drawArena();
  drawErrorChart();
  updateReadout();
}

function advance() {
  if (!state.data) return;
  if (state.frame >= state.data.frame_count - 1) {
    state.playing = false;
    $("playButton").textContent = "Play";
    return;
  }
  state.frame += 1;
  draw();
}

function animate(timestamp) {
  if (state.playing && state.data) {
    if (!state.lastTimestamp) state.lastTimestamp = timestamp;
    state.accumulator += timestamp - state.lastTimestamp;
    const interval = state.data.timestep * 1000 / state.speed;
    while (state.accumulator >= interval) {
      advance();
      state.accumulator -= interval;
    }
  }
  state.lastTimestamp = timestamp;
  requestAnimationFrame(animate);
}

$("playButton").addEventListener("click", () => {
  state.playing = !state.playing;
  $("playButton").textContent = state.playing ? "Pause" : "Play";
});
$("stepButton").addEventListener("click", () => { state.playing = false; $("playButton").textContent = "Play"; advance(); });
$("resetButton").addEventListener("click", () => { state.playing = false; state.frame = 3; state.accumulator = 0; $("playButton").textContent = "Play"; draw(); });
$("loadButton").addEventListener("click", loadScenario);
$("scenarioSelect").addEventListener("change", loadScenario);
$("modelSelect").addEventListener("change", (event) => { state.model = event.target.value; draw(); });
$("speedRange").addEventListener("input", (event) => { state.speed = Number(event.target.value); $("speedOutput").textContent = `${state.speed.toFixed(2).replace(/0$/, "")}×`; });
$("trailToggle").addEventListener("change", draw);
window.addEventListener("resize", draw);

Promise.all([loadStatus(), loadScenario()]).catch((error) => {
  $("loadingOverlay").textContent = `Demo failed: ${error.message}`;
});
requestAnimationFrame(animate);
