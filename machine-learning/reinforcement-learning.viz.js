/* reinforcement-learning.viz.js — extracted from reinforcement-learning.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* Q-Learning Gridworld D3 Simulator */
(function(){
  const svg = d3.select("#grid-svg"), W = 340;
  const gridSize = 4;
  const cellSize = W / gridSize;

  // Board configuration: 0=Empty, 1=Wall, 2=Start, 3=Goal, 4=Pit
  const grid = [
    [2, 0, 0, 0],
    [0, 1, 0, 4],
    [0, 0, 0, 0],
    [0, 4, 0, 3]
  ];

  // Actions: 0=Up, 1=Right, 2=Down, 3=Left
  const actions = [
    { dy: -1, dx: 0, name: "Up" },
    { dy: 0, dx: 1, name: "Right" },
    { dy: 1, dx: 0, name: "Down" },
    { dy: 0, dx: -1, name: "Left" }
  ];

  // Q-table: Q[r][c][a]
  let Q = [];
  function initQ() {
    Q = [];
    for (let r = 0; r < gridSize; r++) {
      Q.push([]);
      for (let c = 0; c < gridSize; c++) Q[r].push(new Array(4).fill(0));
    }
  }
  initQ();

  let agentRow = 0, agentCol = 0;
  let episode = 1, stepCount = 0, episodeReward = 0;
  let timer = null, isTraining = false;
  let returns = [];                       // one entry per finished episode

  const cellsGroup = svg.append("g");
  const arrowsGroup = svg.append("g");
  const agentGroup = svg.append("g");

  function varColor(val) {
    if (val === 3) return C.good;
    if (val === 4) return C.bad;
    return C.muted;
  }

  // Static cells
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      let color = C.line, label = "";
      if (grid[r][c] === 1)      { color = "#000";                    label = "Wall"; }
      else if (grid[r][c] === 2) { color = "rgba(91,156,255,0.06)";   label = "Start"; }
      else if (grid[r][c] === 3) { color = "rgba(74,222,128,0.2)";    label = "Goal (+10)"; }
      else if (grid[r][c] === 4) { color = "rgba(248,113,113,0.2)";   label = "Pit (-10)"; }

      cellsGroup.append("rect")
        .attr("x", c * cellSize).attr("y", r * cellSize)
        .attr("width", cellSize).attr("height", cellSize)
        .attr("fill", color)
        .attr("stroke", "var(--line)")
        .attr("stroke-width", 1.5);

      if (label) {
        cellsGroup.append("text")
          .attr("x", c * cellSize + cellSize / 2).attr("y", r * cellSize + cellSize / 2 + 20)
          .attr("fill", grid[r][c] === 1 ? "#e6e9ef" : varColor(grid[r][c]))
          .attr("font-size", 9).attr("font-weight", "600").attr("text-anchor", "middle")
          .text(label);
      }
    }
  }

  // Arrowheads
  const defs = svg.append("defs");
  defs.append("marker")
    .attr("id", "rl-arr").attr("viewBox", "0 0 10 10")
    .attr("refX", 7).attr("refY", 5).attr("markerWidth", 4).attr("markerHeight", 4)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", C.A);
  defs.append("marker")
    .attr("id", "rl-arr-max").attr("viewBox", "0 0 10 10")
    .attr("refX", 7).attr("refY", 5).attr("markerWidth", 5).attr("markerHeight", 5)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", C.good);

  // Q-value arrows: one stub per action, the greedy one highlighted
  function drawArrows() {
    arrowsGroup.selectAll("*").remove();
    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        if (grid[r][c] === 1 || grid[r][c] === 3) continue;
        const ccx = c * cellSize + cellSize / 2;
        const ccy = r * cellSize + cellSize / 2;
        const cellQs = Q[r][c];
        const maxQ = Math.max(...cellQs);

        const stubs = [
          { x2: ccx,      y2: ccy - 22, id: 0 },
          { x2: ccx + 22, y2: ccy,      id: 1 },
          { x2: ccx,      y2: ccy + 22, id: 2 },
          { x2: ccx - 22, y2: ccy,      id: 3 }
        ];
        stubs.forEach(a => {
          const val = cellQs[a.id];
          if (val === 0) return;
          const isMax = (val === maxQ && maxQ > 0);
          arrowsGroup.append("line")
            .attr("x1", ccx).attr("y1", ccy)
            .attr("x2", a.x2).attr("y2", a.y2)
            .attr("stroke", isMax ? C.good : C.A)
            .attr("stroke-width", isMax ? 3.2 : 1.5)
            .attr("opacity", isMax ? 0.9 : Math.min(0.6, Math.abs(val) / 10))
            .attr("marker-end", isMax ? "url(#rl-arr-max)" : "url(#rl-arr)");
        });
      }
    }
  }

  const agent = agentGroup.append("circle")
    .attr("r", 14).attr("fill", C.B).attr("stroke", "#0f1117").attr("stroke-width", 2);

  function moveAgent() {
    agent.transition().duration(50)
      .attr("cx", agentCol * cellSize + cellSize / 2)
      .attr("cy", agentRow * cellSize + cellSize / 2);
  }

  /* ---- learning curve sparkline ---- */
  const cSvg = d3.select("#rl-curve"), cW = 300, cH = 86, cm = { t: 12, r: 8, b: 16, l: 30 };
  const cx = d3.scaleLinear().range([cm.l, cW - cm.r]);
  const cy = d3.scaleLinear().domain([-12, 11]).range([cH - cm.b, cm.t]);
  cSvg.append("text").attr("x", cW / 2).attr("y", 9).attr("text-anchor", "middle")
    .attr("fill", C.muted).attr("font-size", 9).text("episode return (raw + 10-episode mean)");
  cSvg.append("line").attr("x1", cm.l).attr("x2", cW - cm.r).attr("y1", cy(0)).attr("y2", cy(0))
    .attr("stroke", "#3a4150").attr("stroke-dasharray", "2 2");
  cSvg.append("g").attr("class", "axis").attr("transform", `translate(${cm.l},0)`)
    .call(d3.axisLeft(cy).ticks(3));
  const rawPath = cSvg.append("path").attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1).attr("opacity", 0.5);
  const avgPath = cSvg.append("path").attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 2);

  function drawCurve() {
    const window = returns.slice(-120);
    cx.domain([0, Math.max(9, window.length - 1)]);
    const gen = d3.line().x((d, i) => cx(i)).y(d => cy(Math.max(-12, Math.min(11, d))));
    rawPath.attr("d", window.length > 1 ? gen(window) : null);
    const smoothed = window.map((_, i) => {
      const from = Math.max(0, i - 9);
      const slice = window.slice(from, i + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    });
    avgPath.attr("d", smoothed.length > 1 ? gen(smoothed) : null);
  }

  function stepQLearning() {
    const eps   = parseFloat(d3.select("#param-eps").property("value")) / 100;
    const alpha = parseFloat(d3.select("#param-alpha").property("value")) / 100;
    const gamma = parseFloat(d3.select("#param-gamma").property("value")) / 100;

    const sRow = agentRow, sCol = agentCol;

    // 1 — ε-greedy action selection
    let actionIdx;
    if (Math.random() < eps) {
      actionIdx = Math.floor(Math.random() * 4);
    } else {
      const cellQs = Q[sRow][sCol];
      const maxQ = Math.max(...cellQs);
      const best = [];
      for (let i = 0; i < 4; i++) if (cellQs[i] === maxQ) best.push(i);
      actionIdx = best[Math.floor(Math.random() * best.length)];
    }

    // 2 — transition (walls and edges bounce back)
    const act = actions[actionIdx];
    let nextRow = sRow + act.dy, nextCol = sCol + act.dx;
    if (nextRow < 0 || nextRow >= gridSize || nextCol < 0 || nextCol >= gridSize || grid[nextRow][nextCol] === 1) {
      nextRow = sRow; nextCol = sCol;
    }

    // 3 — reward
    let reward = -0.1, done = false;
    if (grid[nextRow][nextCol] === 3)      { reward = 10.0;  done = true; }
    else if (grid[nextRow][nextCol] === 4) { reward = -10.0; done = true; }

    // 4 — Q-learning update: bootstrap from the best next action, terminal states have value 0
    const maxNextQ = done ? 0 : Math.max(...Q[nextRow][nextCol]);
    const target = reward + gamma * maxNextQ;
    Q[sRow][sCol][actionIdx] += alpha * (target - Q[sRow][sCol][actionIdx]);

    // 5 — move
    agentRow = nextRow; agentCol = nextCol;
    stepCount++;
    episodeReward += reward;

    moveAgent();
    drawArrows();

    if (done) {
      returns.push(episodeReward);
      if (d3.select("#rl-decay").property("checked")) {
        const el = d3.select("#param-eps");
        el.property("value", Math.max(1, parseFloat(el.property("value")) * 0.97));
      }
      drawCurve();
      agentRow = 0; agentCol = 0;
      episode++; stepCount = 0; episodeReward = 0;
    }

    updateReadout();
  }

  function updateReadout() {
    const last10 = returns.slice(-10);
    const avg = last10.length ? last10.reduce((a, b) => a + b, 0) / last10.length : 0;
    const eps = parseFloat(d3.select("#param-eps").property("value")) / 100;
    d3.select("#rl-readout").html(`
      episode <b>${episode}</b> · step <b>${stepCount}</b> · ε = <b>${eps.toFixed(2)}</b><br>
      episode reward so far: <b style="color:${episodeReward >= 0 ? C.good : C.bad}">${episodeReward.toFixed(1)}</b>
      · mean of last 10: <b style="color:${avg >= 0 ? C.good : C.bad}">${avg.toFixed(2)}</b><br>
      Q at Start (0,0) — Up <b>${Q[0][0][0].toFixed(2)}</b>, Right <b>${Q[0][0][1].toFixed(2)}</b>,
      Down <b>${Q[0][0][2].toFixed(2)}</b>, Left <b>${Q[0][0][3].toFixed(2)}</b>
    `);
  }

  d3.select("#rl-step").on("click", () => {
    if (isTraining) toggleTrain();
    stepQLearning();
  });

  function toggleTrain() {
    isTraining = !isTraining;
    const btn = d3.select("#rl-train");
    if (isTraining) {
      btn.text("Pause").classed("ghost", true);
      timer = setInterval(stepQLearning, parseFloat(d3.select("#param-speed").property("value")));
    } else {
      btn.text("Train Live").classed("ghost", false);
      clearInterval(timer);
    }
  }
  d3.select("#rl-train").on("click", toggleTrain);

  d3.select("#param-speed").on("input", function() {
    if (isTraining) {
      clearInterval(timer);
      timer = setInterval(stepQLearning, parseFloat(this.value));
    }
  });

  d3.select("#param-eps").on("input", updateReadout);

  d3.select("#rl-reset").on("click", () => {
    if (isTraining) toggleTrain();
    initQ();
    agentRow = 0; agentCol = 0;
    episode = 1; stepCount = 0; episodeReward = 0;
    returns = [];
    moveAgent();
    drawArrows();
    drawCurve();
    updateReadout();
  });

  moveAgent();
  drawCurve();
  updateReadout();
})();
