const workloads = [
  {
    label: "Workload A",
    writeRatio: 0.2,
    hotKeyRatio: 0.95,
    arrivalRate: 3000,
    executionDelay: 5000
  },
  {
    label: "Workload B",
    writeRatio: 0.5,
    hotKeyRatio: 0.99,
    arrivalRate: 1000,
    executionDelay: 1000
  },
  {
    label: "Workload C",
    writeRatio: 0.5,
    hotKeyRatio: 0.1,
    arrivalRate: 3000,
    executionDelay: 10000
  },
  {
    label: "Workload D",
    writeRatio: 0.9,
    hotKeyRatio: 0.95,
    arrivalRate: 1000,
    executionDelay: 0
  }
];

const actions = [
  { id: "OX-50", name: "OX", blocksize: 50, earlyExecution: false, reorder: false },
  { id: "OXII-100", name: "OXII", blocksize: 100, earlyExecution: false, reorder: true },
  { id: "XOV-1", name: "XOV", blocksize: 1, earlyExecution: true, reorder: false },
  { id: "XOV++-50", name: "XOV++", blocksize: 50, earlyExecution: true, reorder: true },
  { id: "XOV++-400", name: "XOV++ large", blocksize: 400, earlyExecution: true, reorder: true }
];

const stepCopy = [
  {
    title: "Observe workload state",
    mode: "Context",
    text: "AdaChain starts each episode by measuring workload features: write ratio, hot key ratio, transaction arrival rate, and execution delay. These four numbers are the context used by the bandit."
  },
  {
    title: "Predict each architecture",
    mode: "Scoring",
    text: "The learning agent enumerates candidate architectures and predicts the effective throughput for each one under the current context."
  },
  {
    title: "Select architecture",
    mode: "Action",
    text: "AdaChain chooses the architecture with the highest predicted throughput. In the paper this is a contextual bandit decision over architecture parameters."
  },
  {
    title: "Run episode and measure reward",
    mode: "Reward",
    text: "The selected architecture processes transactions until the episode watermark. The reward is measured effective throughput, not a hand-written score."
  },
  {
    title: "Update experience",
    mode: "Learning",
    text: "The latest state, action, and throughput are appended to the experience window. Future predictions are pulled toward architectures that worked well in similar states."
  }
];

const state = {
  episode: 0,
  step: 0,
  workloadIndex: 0,
  selectedAction: null,
  observedState: null,
  predictions: [],
  history: [],
  learnedBias: new Map(actions.map((action) => [action.id, 0]))
};

const els = {
  episodeNumber: document.getElementById("episodeNumber"),
  workloadSelect: document.getElementById("workloadSelect"),
  noiseRange: document.getElementById("noiseRange"),
  noiseOutput: document.getElementById("noiseOutput"),
  stepButton: document.getElementById("stepButton"),
  runButton: document.getElementById("runButton"),
  resetButton: document.getElementById("resetButton"),
  stepTitle: document.getElementById("stepTitle"),
  modeLabel: document.getElementById("modeLabel"),
  explanation: document.getElementById("explanation"),
  rlGraph: document.getElementById("rlGraph"),
  writeRatio: document.getElementById("writeRatio"),
  hotKeyRatio: document.getElementById("hotKeyRatio"),
  arrivalRate: document.getElementById("arrivalRate"),
  executionDelay: document.getElementById("executionDelay"),
  actionTable: document.getElementById("actionTable"),
  selectedAction: document.getElementById("selectedAction"),
  throughput: document.getElementById("throughput"),
  regret: document.getElementById("regret"),
  chart: document.getElementById("chart"),
  history: document.getElementById("history"),
  experienceCount: document.getElementById("experienceCount")
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function noiseFactor(scale) {
  if (scale === 0) return 1;
  return 1 + ((Math.random() * 2 - 1) * scale);
}

function getTrueState() {
  return workloads[state.workloadIndex];
}

function observeState() {
  const base = getTrueState();
  const noise = Number(els.noiseRange.value) / 100;
  state.observedState = {
    label: base.label,
    writeRatio: clamp(base.writeRatio * noiseFactor(noise), 0.01, 0.99),
    hotKeyRatio: clamp(base.hotKeyRatio * noiseFactor(noise), 0.01, 0.99),
    arrivalRate: Math.round(clamp(base.arrivalRate * noiseFactor(noise), 100, 6000)),
    executionDelay: Math.round(clamp(base.executionDelay * noiseFactor(noise), 0, 15000))
  };
}

function throughputModel(context, action) {
  const write = context.writeRatio;
  const hot = context.hotKeyRatio;
  const arrival = context.arrivalRate / 3000;
  const delay = context.executionDelay / 5000;
  let score = 420;

  if (action.name === "OX") {
    score += 620 * write * hot;
    score += 160 * (1 - delay);
    score -= 190 * arrival;
  }

  if (action.name === "OXII") {
    score += 320 * delay;
    score += 170 * hot;
    score += 100 * arrival;
    score -= 160 * write * hot;
  }

  if (action.name === "XOV") {
    score += 560 * (1 - hot);
    score += 260 * delay;
    score -= 430 * write * hot;
  }

  if (action.name === "XOV++") {
    score += 560 * delay;
    score += 220 * (1 - write);
    score += 140 * hot;
    score -= 520 * write * hot;
  }

  if (action.name === "XOV++ large") {
    score += 420 * delay;
    score += 180 * (1 - write);
    score -= 340 * hot;
    score -= 90 * arrival;
  }

  const blockPenalty = Math.abs(action.blocksize - 80) * 0.22;
  return Math.max(80, score - blockPenalty);
}

function predictActions() {
  if (!state.observedState) observeState();
  state.predictions = actions.map((action) => {
    const learned = state.learnedBias.get(action.id) || 0;
    const predicted = throughputModel(state.observedState, action) + learned;
    const actual = throughputModel(getTrueState(), action);
    return { action, predicted, actual };
  });
}

function selectAction() {
  if (!state.predictions.length) predictActions();
  state.selectedAction = state.predictions.reduce((best, item) => (
    item.predicted > best.predicted ? item : best
  ));
}

function runEpisode() {
  if (!state.selectedAction) selectAction();
  const jitter = 1 + ((Math.random() * 0.12) - 0.06);
  const measured = Math.round(state.selectedAction.actual * jitter);
  const bestActual = Math.max(...state.predictions.map((item) => item.actual));
  const regret = Math.max(0, Math.round(bestActual - measured));
  const row = {
    episode: state.episode + 1,
    workload: getTrueState().label,
    action: state.selectedAction.action,
    throughput: measured,
    regret
  };
  state.history.push(row);
  state.episode += 1;
  return row;
}

function updateExperience() {
  const latest = state.history[state.history.length - 1];
  if (!latest) return;
  const previous = state.learnedBias.get(latest.action.id) || 0;
  const predicted = state.selectedAction ? state.selectedAction.predicted : latest.throughput;
  const error = latest.throughput - predicted;
  state.learnedBias.set(latest.action.id, previous + error * 0.18);
}

function completeStep() {
  if (state.step === 0) observeState();
  if (state.step === 1) predictActions();
  if (state.step === 2) selectAction();
  if (state.step === 3) runEpisode();
  if (state.step === 4) {
    updateExperience();
    state.step = 0;
    state.selectedAction = null;
    state.predictions = [];
    observeState();
    render();
    return;
  }
  state.step += 1;
  render();
}

function runMany(count) {
  for (let i = 0; i < count; i += 1) {
    observeState();
    predictActions();
    selectAction();
    runEpisode();
    updateExperience();
  }
  state.step = 0;
  state.selectedAction = null;
  state.predictions = [];
  observeState();
  render();
}

function reset() {
  state.episode = 0;
  state.step = 0;
  state.selectedAction = null;
  state.observedState = null;
  state.predictions = [];
  state.history = [];
  state.learnedBias = new Map(actions.map((action) => [action.id, 0]));
  observeState();
  render();
}

function formatRatio(value) {
  return value.toFixed(2);
}

function actionLabel(action) {
  return `${action.name} / block ${action.blocksize}`;
}

function activeForStep(nodeStep) {
  return state.step === nodeStep ? "active" : "";
}

function renderGraph() {
  const latest = state.history[state.history.length - 1];
  const predictions = state.predictions.length ? state.predictions : actions.map((action) => ({
    action,
    predicted: 0,
    actual: throughputModel(getTrueState(), action)
  }));
  const maxActual = Math.max(...predictions.map((item) => item.actual));
  const selectedId = state.selectedAction ? state.selectedAction.action.id : null;
  const selectedAction = state.selectedAction
    ? actionLabel(state.selectedAction.action)
    : "waiting";
  const rewardText = latest ? `${latest.throughput} TPS` : "not measured";
  const regretText = latest ? `${latest.regret} TPS regret` : "no regret yet";
  const context = state.observedState || getTrueState();
  const edgeStage = state.step;

  const actionNodes = predictions.map((item) => {
    const selected = selectedId === item.action.id;
    const best = Math.abs(item.actual - maxActual) < 0.001;
    const classes = [
      "graph-node",
      state.step === 1 ? "active" : "",
      selected ? "selected" : "",
      best ? "best" : ""
    ].filter(Boolean).join(" ");
    const predicted = item.predicted ? `${Math.round(item.predicted)} predicted TPS` : "not scored yet";
    return `
      <div class="${classes}">
        <span class="node-kicker">action candidate</span>
        <span class="node-title">${actionLabel(item.action)}</span>
        <span class="node-detail">${predicted}</span>
      </div>
    `;
  }).join("");

  els.rlGraph.innerHTML = `
    <svg class="graph-edge-layer" viewBox="0 0 1000 360" preserveAspectRatio="none" aria-hidden="true">
      <path class="graph-edge ${edgeStage >= 0 ? "active" : ""}" d="M160 180 C210 180 230 180 280 180"></path>
      <path class="graph-edge ${edgeStage >= 1 ? "active" : ""}" d="M360 180 C410 70 440 70 490 70"></path>
      <path class="graph-edge ${edgeStage >= 1 ? "active" : ""}" d="M360 180 C410 125 440 125 490 125"></path>
      <path class="graph-edge ${edgeStage >= 1 ? "active" : ""}" d="M360 180 C410 180 440 180 490 180"></path>
      <path class="graph-edge ${edgeStage >= 1 ? "active" : ""}" d="M360 180 C410 235 440 235 490 235"></path>
      <path class="graph-edge ${edgeStage >= 1 ? "active" : ""}" d="M360 180 C410 290 440 290 490 290"></path>
      <path class="graph-edge ${edgeStage >= 2 ? "selected" : ""}" d="M640 180 C690 180 710 180 760 180"></path>
      <path class="graph-edge ${edgeStage >= 3 ? "selected" : ""}" d="M840 180 C890 180 900 180 950 180"></path>
      <path class="graph-edge ${edgeStage >= 4 ? "active" : ""}" d="M950 235 C820 345 350 345 220 235"></path>
    </svg>
    <div class="graph-column">
      <div class="graph-node ${activeForStep(0)}">
        <span class="node-kicker">context state</span>
        <span class="node-title">${context.label}</span>
        <span class="node-detail">w=${context.writeRatio.toFixed(2)}, hot=${context.hotKeyRatio.toFixed(2)}, rate=${Math.round(context.arrivalRate)}, delay=${Math.round(context.executionDelay)} us</span>
      </div>
    </div>
    <div class="graph-column">
      <div class="graph-node ${state.step === 1 ? "active" : ""}">
        <span class="node-kicker">policy model</span>
        <span class="node-title">throughput predictor</span>
        <span class="node-detail">scores each architecture from context plus learned experience</span>
      </div>
    </div>
    <div class="graph-column">
      ${actionNodes}
    </div>
    <div class="graph-column">
      <div class="graph-node ${activeForStep(2)} ${selectedId ? "selected" : ""}">
        <span class="node-kicker">chosen action</span>
        <span class="node-title">${selectedAction}</span>
        <span class="node-detail">architecture used for the next episode</span>
      </div>
    </div>
    <div class="graph-column">
      <div class="graph-node reward ${activeForStep(3)}">
        <span class="node-kicker">reward</span>
        <span class="node-title">${rewardText}</span>
        <span class="node-detail">${regretText}</span>
      </div>
      <div class="graph-node ${activeForStep(4)}">
        <span class="node-kicker">experience buffer</span>
        <span class="node-title">${state.history.length} samples</span>
        <span class="node-detail">state, action, reward rows used to improve later predictions</span>
      </div>
    </div>
  `;
}

function renderActions() {
  if (!state.predictions.length) {
    els.actionTable.innerHTML = "<div class=\"action-row\"><span class=\"action-name\">Waiting for prediction step</span><div class=\"bar-track\"><div class=\"bar\"></div></div><span class=\"score\">0 TPS</span></div>";
    return;
  }

  const maxPredicted = Math.max(...state.predictions.map((item) => item.predicted));
  const maxActual = Math.max(...state.predictions.map((item) => item.actual));
  els.actionTable.innerHTML = state.predictions
    .map((item) => {
      const selected = state.selectedAction && state.selectedAction.action.id === item.action.id;
      const best = Math.abs(item.actual - maxActual) < 0.001;
      const width = Math.round((item.predicted / maxPredicted) * 100);
      const classes = ["action-row", selected ? "selected" : "", best ? "best" : ""].filter(Boolean).join(" ");
      return `
        <div class="${classes}">
          <span class="action-name">${actionLabel(item.action)}</span>
          <div class="bar-track"><div class="bar" style="width: ${width}%"></div></div>
          <span class="score">${Math.round(item.predicted)} TPS</span>
        </div>
      `;
    })
    .join("");
}

function renderHistory() {
  const maxThroughput = Math.max(1, ...state.history.map((row) => row.throughput));
  const recent = state.history.slice(-30);
  els.chart.innerHTML = recent.map((row) => {
    const height = Math.max(4, Math.round((row.throughput / maxThroughput) * 126));
    return `<div class="chart-bar" title="Episode ${row.episode}: ${row.throughput} TPS" style="height: ${height}px"></div>`;
  }).join("");

  els.history.innerHTML = state.history.slice(-10).reverse().map((row) => `
    <div class="history-row">
      <strong>#${row.episode}</strong>
      <span>${row.workload}</span>
      <span>${actionLabel(row.action)}</span>
      <span>${row.throughput} TPS</span>
    </div>
  `).join("");

  els.experienceCount.textContent = `${state.history.length} samples`;
}

function render() {
  if (!state.observedState) observeState();
  const copy = stepCopy[state.step];
  els.episodeNumber.textContent = state.episode;
  els.noiseOutput.textContent = `${els.noiseRange.value}%`;
  els.stepTitle.textContent = copy.title;
  els.modeLabel.textContent = copy.mode;
  els.explanation.textContent = copy.text;

  els.writeRatio.textContent = formatRatio(state.observedState.writeRatio);
  els.hotKeyRatio.textContent = formatRatio(state.observedState.hotKeyRatio);
  els.arrivalRate.textContent = Math.round(state.observedState.arrivalRate).toString();
  els.executionDelay.textContent = `${Math.round(state.observedState.executionDelay)} us`;

  document.querySelectorAll(".step-item").forEach((item) => {
    item.classList.toggle("active", Number(item.dataset.step) === state.step);
  });

  renderActions();
  renderGraph();

  const latest = state.history[state.history.length - 1];
  els.selectedAction.textContent = state.selectedAction
    ? actionLabel(state.selectedAction.action)
    : (latest ? actionLabel(latest.action) : "None yet");
  els.throughput.textContent = latest ? `${latest.throughput} TPS` : "0 TPS";
  els.regret.textContent = latest ? `${latest.regret} TPS` : "0 TPS";

  renderHistory();
}

els.stepButton.addEventListener("click", completeStep);
els.runButton.addEventListener("click", () => runMany(10));
els.resetButton.addEventListener("click", reset);
els.workloadSelect.addEventListener("change", (event) => {
  state.workloadIndex = Number(event.target.value);
  state.step = 0;
  state.selectedAction = null;
  state.predictions = [];
  observeState();
  render();
});
els.noiseRange.addEventListener("input", () => {
  observeState();
  state.predictions = [];
  state.selectedAction = null;
  render();
});

reset();
