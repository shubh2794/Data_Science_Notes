/* sampling-design.viz.js — the six visualizations on math/statistics/sampling-design.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #frame-svg     the frame gap — repeated samples scatter around the FRAME's parameter,
                       not the population's, and more sampling never closes the offset
     2  #design-svg    six ways to draw 40 of 400 units from a clustered population
     3  #strat-svg     stratified vs simple random at the same n — the exact variance identity
     4  #illusion-svg  RMSE = √(bias² + σ²/n): the n at which bias overtakes chance error
     5  #confound-svg  a lurking variable, the back-door path, and what conditioning does
     6  #rand-svg      what randomisation buys: balance on the variable nobody measured        */

/* ─────────────────── 1 · the frame gap ─────────────────── */
(function () {
  const svg = d3.select("#frame-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 700, H = 470;

  /* A population of 600 with two groups, built so every count below is exact.
     Group A ("on the list"): 400 units, 160 supporters  → 40.0%
     Group B ("off the list"): 200 units, 140 supporters → 70.0%
     Population parameter p = 300/600 = 0.500 exactly.                        */
  const NA = 400, NB = 200, N = NA + NB;
  const rq = ST.rng(1936);
  const flags = k => {                        // exact counts, then shuffled so the picture has no stripes
    const a = [];
    for (let i = 0; i < k.n; i++) a.push(i < k.ones ? 1 : 0);
    return ST.shuffle(a, rq);
  };
  const fA = flags({ n: NA, ones: 160 });
  const fB = [];                              // group B in blocks of 10, each holding exactly 7 ones,
  for (let b = 0; b < NB / 10; b++) {         // shuffled inside the block — so any multiple of 10
    const blk = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0];   // taken from the front is exactly 70% supporters
    ST.shuffle(blk, rq).forEach(v => fB.push(v));
  }
  const pop = [];
  for (let i = 0; i < NA; i++) pop.push({ g: 0, y: fA[i] });                   // 160/400 = 40%
  for (let i = 0; i < NB; i++) pop.push({ g: 1, y: fB[i] });                   // 140/200 = 70%
  const P_TRUE = ST.mean(pop.map(d => d.y));                                   // = 0.5 exactly
  const NOSUP = "#414a5e";

  const F = ST.frame(svg, W, H, { l: 14, r: 14, t: 16, b: 34 });
  const COLS = 40, CELL = 10.2, GAP = 1.2, STEP = CELL + GAP;
  const gGrid = F.g.append("g");
  const gAxis = F.g.append("g").attr("transform", `translate(0,${205})`);

  pop.forEach((d, i) => {
    d.col = i % COLS; d.row = Math.floor(i / COLS);
    d.px = d.col * STEP; d.py = d.row * STEP;
  });
  const rects = gGrid.selectAll("rect.u").data(pop).join("rect").attr("class", "u")
    .attr("x", d => d.px).attr("y", d => d.py)
    .attr("width", CELL).attr("height", CELL).attr("rx", 1.6);

  // group divider + labels
  gGrid.append("line").attr("x1", -4).attr("x2", COLS * STEP).attr("y1", 10 * STEP - GAP / 2)
    .attr("y2", 10 * STEP - GAP / 2).attr("stroke", SC.line).attr("stroke-width", 1.5);
  gGrid.append("text").attr("x", COLS * STEP + 10).attr("y", 5 * STEP).attr("font-size", 10.5)
    .attr("fill", SC.muted).text("group A · 400 units");
  gGrid.append("text").attr("x", COLS * STEP + 10).attr("y", 5 * STEP + 14).attr("font-size", 10.5)
    .attr("fill", SC.muted).text("40% support");
  gGrid.append("text").attr("x", COLS * STEP + 10).attr("y", 12.4 * STEP).attr("font-size", 10.5)
    .attr("fill", SC.muted).text("group B · 200 units");
  gGrid.append("text").attr("x", COLS * STEP + 10).attr("y", 12.4 * STEP + 14).attr("font-size", 10.5)
    .attr("fill", SC.muted).text("70% support");
  ST.legend(gGrid, [
    { label: "supports", color: SC.accent },
    { label: "does not", color: NOSUP, op: 1 },
    { label: "outside the frame", color: SC.bad, op: 0.22 },
    { label: "in the last sample", color: SC.a2 }
  ], COLS * STEP + 10, 14 * STEP + 8, { gap: 14, font: 10 });

  // estimate axis
  const AX_W = 600;
  const x = d3.scaleLinear().domain([0.30, 0.70]).range([0, AX_W]);
  const AH = 190;                                        // height available for the dot stack
  gAxis.append("g").attr("class", "axis").attr("transform", `translate(0,${AH})`)
    .call(d3.axisBottom(x).ticks(9).tickFormat(d3.format(".2f")));
  gAxis.append("text").attr("x", AX_W).attr("y", AH + 31).attr("text-anchor", "end")
    .attr("font-size", 11).attr("fill", SC.muted).text("estimate p̂ from one sample");
  const gDots = gAxis.append("g");
  gAxis.append("line").attr("y1", 0).attr("y2", AH)
    .attr("stroke", SC.good).attr("stroke-width", 2).attr("x1", x(P_TRUE)).attr("x2", x(P_TRUE));
  gAxis.append("text").attr("x", x(P_TRUE)).attr("y", -18).attr("text-anchor", "middle")
    .attr("font-size", 10.5).attr("fill", SC.good).text("truth p = 0.500");
  const lineF = gAxis.append("line").attr("y1", 0).attr("y2", AH)
    .attr("stroke", SC.bad).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
  const labF = gAxis.append("text").attr("y", -4).attr("text-anchor", "middle")
    .attr("font-size", 10.5).attr("fill", SC.bad);
  const bandF = gAxis.append("rect").attr("y", 0).attr("height", AH).attr("fill", SC.bad).attr("opacity", 0.08);

  const sN = document.getElementById("fr-n"), sNv = document.getElementById("fr-nv");
  const sC = document.getElementById("fr-c"), sCv = document.getElementById("fr-cv");
  const out = document.getElementById("frame-readout");

  let ests = [], rnd = ST.rng(20260905), lastIdx = new Set();

  function frameIdx() {                       // group A entirely + the first k of group B
    const k = +sC.value;                      // k is a multiple of 10, so p_F stays exact
    const idx = [];
    for (let i = 0; i < NA; i++) idx.push(i);
    for (let i = 0; i < k; i++) idx.push(NA + i);
    return idx;
  }

  function draw(times) {
    const idx = frameIdx(), M = idx.length, n = Math.min(+sN.value, M);
    for (let t = 0; t < times; t++) {
      const pick = ST.srswor(idx, n, rnd);
      ests.push(ST.mean(pick.map(i => pop[i].y)));
      if (t === times - 1) lastIdx = new Set(pick);
    }
    if (ests.length > 500) ests = ests.slice(-500);
  }

  function update() {
    const idx = frameIdx(), inFrame = new Set(idx), M = idx.length;
    const pF = ST.mean(idx.map(i => pop[i].y));
    const n = Math.min(+sN.value, M);
    sNv.textContent = n; sCv.textContent = (+sC.value) + " of 200";

    rects
      .attr("fill", d => {
        const i = d.row * COLS + d.col;
        if (lastIdx.has(i)) return SC.a2;
        if (!inFrame.has(i)) return SC.bad;
        return d.y ? SC.accent : NOSUP;
      })
      .attr("fill-opacity", d => {
        const i = d.row * COLS + d.col;
        if (lastIdx.has(i)) return 1;
        if (!inFrame.has(i)) return d.y ? 0.42 : 0.18;
        return d.y ? 0.92 : 1;
      });

    lineF.attr("x1", x(pF)).attr("x2", x(pF));
    labF.attr("x", x(pF)).text("what the frame contains: p_F = " + ST.fmt(pF, 3));
    const lo = Math.min(pF, P_TRUE), hi = Math.max(pF, P_TRUE);
    bandF.attr("x", x(lo)).attr("width", Math.max(0, x(hi) - x(lo)));

    // dot-stack of the estimates, binned at 0.004
    const BW = 0.004, R = 3.0, counts = new Map(), placed = [];
    ests.forEach((e, i) => {
      const b = Math.round(e / BW);
      const c = counts.get(b) || 0; counts.set(b, c + 1);
      placed.push({ e: e, b: b, c: Math.min(c, 26), fresh: i >= ests.length - 1 });
    });
    const sel = gDots.selectAll("circle").data(placed);
    sel.exit().remove();
    sel.enter().append("circle").merge(sel)
      .attr("cx", d => x(d.b * BW)).attr("cy", d => AH - 4 - d.c * (2 * R + 0.6) - R)
      .attr("r", d => d.fresh ? R + 1.3 : R)
      .attr("fill", d => d.fresh ? SC.a2 : SC.accent)
      .attr("fill-opacity", d => d.fresh ? 1 : 0.55);

    const se = Math.sqrt(pF * (1 - pF) / n) * Math.sqrt((M - n) / (M - 1));
    const bias = pF - P_TRUE;
    const rmse = Math.sqrt(bias * bias + se * se);
    const mE = ests.length ? ST.mean(ests) : NaN, sE = ests.length > 1 ? ST.sd(ests) : NaN;
    const nEq = 0.25 / (rmse * rmse);

    out.innerHTML =
      `frame reaches <b>${M}</b> of ${N} units · p_F = <b>${ST.fmt(pF, 3)}</b> · ` +
      `truth p = <b>0.500</b> · <span style="color:${SC.bad}">bias = p_F − p = ${ST.fmt(bias, 3)}</span>` +
      `<br>n = <b>${n}</b> · exact SE with the finite-population correction = <b>${ST.fmt(se, 4)}</b> · ` +
      `RMSE = √(bias² + SE²) = <b>${ST.fmt(rmse, 4)}</b>` +
      (ests.length ? `<br>${ests.length} draws so far · mean p̂ = <b>${ST.fmt(mE, 4)}</b> ` +
        `(heading for p_F = ${ST.fmt(pF, 3)}, not for 0.500) · observed SD = <b>${ST.fmt(sE, 4)}</b>` : "") +
      `<br><span style="color:${SC.a2}">` +
      (Math.abs(bias) < 1e-9
        ? `the frame is complete: the cloud is centred on the truth, and every extra unit of n narrows it`
        : `an unbiased sample of only <b>${Math.round(nEq)}</b> units would match this design's accuracy — ` +
          `and no n at all removes the ${ST.fmt(Math.abs(bias) * 100, 1)}-point offset`) +
      `</span>`;
  }

  sN.addEventListener("input", () => { ests = []; lastIdx = new Set(); update(); });
  sC.addEventListener("input", () => { ests = []; lastIdx = new Set(); update(); });
  draw(50);                                   // arrive with a picture rather than an empty axis
  document.getElementById("fr-1").addEventListener("click", () => { draw(1); update(); });
  document.getElementById("fr-50").addEventListener("click", () => { draw(50); update(); });
  document.getElementById("fr-clear").addEventListener("click", () => { ests = []; lastIdx = new Set(); update(); });
  update();
})();

/* ─────────────────── 2 · six sampling designs on one population ─────────────────── */
(function () {
  const svg = d3.select("#design-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 700, H = 400;
  /* 400 units on a 20 × 20 lattice, partitioned into 20 blocks of 20 (5 across, 4 down,
     each block 4 wide × 5 tall). Values are strongly clustered by block: that is what
     makes cluster sampling expensive in variance and stratification cheap.            */
  const SIDE = 20, N = SIDE * SIDE, NBX = 5, NBY = 4, NB = NBX * NBY, BW = 4, BH = 5;
  const rp = ST.rng(4242);
  const units = [];
  for (let r = 0; r < SIDE; r++) for (let c = 0; c < SIDE; c++) {
    const bx = Math.floor(c / BW), by = Math.floor(r / BH), b = by * NBX + bx;
    const blockMean = 50 + 6 * bx + 5 * by + (b % 3) * 4;
    units.push({ i: r * SIDE + c, r: r, c: c, b: b, y: blockMean + 10 * ST.randn(rp) });
  }
  const POPMEAN = ST.mean(units.map(u => u.y));
  const S2 = ST.variance(units.map(u => u.y), 1);        // population variance, n − 1 divisor
  const n = 40;
  const SE_SRS = Math.sqrt(S2 / n) * Math.sqrt((N - n) / (N - 1));

  // serpentine order for systematic sampling (walk the rows, alternating direction)
  const serp = [];
  for (let r = 0; r < SIDE; r++) {
    for (let k = 0; k < SIDE; k++) { const c = (r % 2 === 0) ? k : SIDE - 1 - k; serp.push(r * SIDE + c); }
  }
  const byBlock = d3.range(NB).map(b => units.filter(u => u.b === b).map(u => u.i));

  const DESIGNS = {
    srs: {
      name: "simple random sample",
      note: "every unit equally likely, every subset of size 40 equally likely. The benchmark: unbiased, and its standard error is the one every formula in this series assumes.",
      draw: r => ST.srswor(d3.range(N), n, r)
    },
    systematic: {
      name: "systematic, 1 in 10",
      note: "walk the population in order and take every 10th unit after a random start. One random number instead of forty; here the walk cuts across blocks, so it behaves like — often slightly better than — a simple random sample. It fails catastrophically only when the list has a period that matches the step.",
      draw: r => { const s = Math.floor(r() * 10), out = []; for (let k = s; k < N; k += 10) out.push(serp[k]); return out; }
    },
    stratified: {
      name: "stratified — 2 from each of the 20 blocks",
      note: "force the sample to mirror the population's block structure, then sample at random inside each block. The between-block variability is removed from the estimator entirely, which is where the gain comes from.",
      draw: r => { let out = []; for (let b = 0; b < NB; b++) out = out.concat(ST.srswor(byBlock[b], 2, r)); return out; }
    },
    cluster: {
      name: "cluster — 2 whole blocks",
      note: "pick 2 of the 20 blocks at random and measure every unit in them. Cheapest to field by far — you visit 2 places, not 40 — and the worst variance, because units inside a block are alike, so the 40 observations carry far less than 40 units' worth of information.",
      draw: r => { const bs = ST.srswor(d3.range(NB), 2, r); return bs.flatMap(b => byBlock[b]); }
    },
    multistage: {
      name: "multistage — 8 blocks, 5 units in each",
      note: "the working compromise, and what almost every national survey actually does: sample clusters first, then sample within them. Spreads the sample over more of the population than cluster sampling while still visiting only 8 places.",
      draw: r => { const bs = ST.srswor(d3.range(NB), 8, r); return bs.flatMap(b => ST.srswor(byBlock[b], 5, r)); }
    },
    convenience: {
      name: "convenience — the 40 easiest units",
      note: "take the first two rows: the units nearest to hand. There is no randomness left to average over, so this design has no standard error at all — only an error. Redrawing it gives you the same answer, wrong in the same direction, for ever.",
      draw: () => { const out = []; for (let r = 0; r < 2; r++) for (let c = 0; c < SIDE; c++) out.push(r * SIDE + c); return out; }
    }
  };

  const F = ST.frame(svg, W, H, { l: 16, r: 16, t: 18, b: 40 });
  const CS = 13.0, CG = 1.0, CST = CS + CG;
  const gLat = F.g.append("g");
  const yv = d3.scaleLinear().domain(d3.extent(units, u => u.y)).range([0.18, 1]);
  const cells = gLat.selectAll("rect.c").data(units).join("rect").attr("class", "c")
    .attr("x", u => u.c * CST).attr("y", u => u.r * CST)
    .attr("width", CS).attr("height", CS).attr("rx", 1.5);
  const gBlocks = gLat.append("g");
  for (let by = 0; by < NBY; by++) for (let bx = 0; bx < NBX; bx++) {
    gBlocks.append("rect")
      .attr("x", bx * BW * CST - CG / 2).attr("y", by * BH * CST - CG / 2)
      .attr("width", BW * CST).attr("height", BH * CST)
      .attr("fill", "none").attr("stroke", SC.muted).attr("stroke-opacity", 0.45).attr("stroke-width", 1);
  }
  gLat.append("text").attr("x", 0).attr("y", -6).attr("font-size", 10.5).attr("fill", SC.muted)
    .text("400 units · 20 blocks · darker = larger value");

  // right panel: the sampling distribution of ȳ over many repeats of the chosen design
  const RX = 300, RW = W - 32 - RX;
  const gRep = F.g.append("g").attr("transform", `translate(${RX},0)`);
  const xr = d3.scaleLinear().domain([POPMEAN - 8 * SE_SRS, POPMEAN + 8 * SE_SRS]).range([0, RW - 10]);
  const RH = 236, REPS2 = 1000;
  gRep.append("g").attr("class", "axis").attr("transform", `translate(0,${RH})`)
    .call(d3.axisBottom(xr).ticks(5).tickFormat(d3.format(".1f")));
  gRep.append("text").attr("x", RW - 10).attr("y", RH + 76).attr("text-anchor", "end")
    .attr("font-size", 11).attr("fill", SC.muted).text("sample mean ȳ over " + REPS2 + " repeats of the design");
  gRep.append("line").attr("x1", xr(POPMEAN)).attr("x2", xr(POPMEAN)).attr("y1", 0).attr("y2", RH)
    .attr("stroke", SC.good).attr("stroke-width", 2);
  gRep.append("text").attr("x", xr(POPMEAN)).attr("y", -6).attr("text-anchor", "middle")
    .attr("font-size", 10.5).attr("fill", SC.good).text("population mean μ = " + ST.fmt(POPMEAN, 2));
  const gRepDots = gRep.append("g");
  const gSrsRef = gRep.append("g");

  const selD = document.getElementById("dz-design");
  const out = document.getElementById("design-readout");
  let rnd = ST.rng(777);

  function update(redraw) {
    const key = selD.value, D = DESIGNS[key];
    if (redraw) rnd = ST.rng(Math.floor(Math.random() * 1e9));

    const pick = D.draw(rnd), inS = new Set(pick);
    const ybar = ST.mean(pick.map(i => units[i].y));
    const blocks = new Set(pick.map(i => units[i].b));

    cells
      .attr("fill", u => inS.has(u.i) ? SC.a2 : SC.accent)
      .attr("fill-opacity", u => inS.has(u.i) ? 1 : yv(u.y) * 0.75)
      .attr("stroke", u => inS.has(u.i) ? "#0f1117" : "none").attr("stroke-width", 1);

    // many repeats of this design, and of a simple random sample for reference
    const rr = ST.rng(99991), means = [], srsMeans = [];
    for (let t = 0; t < REPS2; t++) {
      means.push(ST.mean(D.draw(rr).map(i => units[i].y)));
      srsMeans.push(ST.mean(ST.srswor(d3.range(N), n, rr).map(i => units[i].y)));
    }
    const sdD = ST.sd(means), mD = ST.mean(means), deff = (sdD * sdD) / (SE_SRS * SE_SRS);

    const span = xr.domain()[1] - xr.domain()[0];
    const BWid = Math.max(span / 110, sdD / 14), R = 2.2, counts = new Map(), pl = [];
    means.forEach(m => {
      const b = Math.round(m / BWid), c = counts.get(b) || 0; counts.set(b, c + 1);
      pl.push({ m: b * BWid, c: Math.min(c, 45) });
    });
    const off = means.filter(m => m < xr.domain()[0] || m > xr.domain()[1]).length;
    const sel = gRepDots.selectAll("circle").data(pl.filter(d => d.m >= xr.domain()[0] && d.m <= xr.domain()[1]));
    sel.exit().remove();
    sel.enter().append("circle").merge(sel)
      .attr("cx", d => xr(d.m)).attr("cy", d => RH - 3 - d.c * (2 * R + 0.5) - R)
      .attr("r", R).attr("fill", SC.a2).attr("fill-opacity", 0.75);

    // the two spreads, drawn as brackets under the axis for a direct visual comparison
    gSrsRef.selectAll("*").remove();
    const sdS = ST.sd(srsMeans);
    [[SE_SRS, SC.accent, RH + 36, "±1 SE, simple random sample"],
     [sdD, SC.a2, RH + 54, "±1 SD, this design"]].forEach(([sp, col, yy, lab]) => {
      gSrsRef.append("line")
        .attr("x1", xr(ST.clamp(POPMEAN - sp, xr.domain()[0], xr.domain()[1])))
        .attr("x2", xr(ST.clamp(POPMEAN + sp, xr.domain()[0], xr.domain()[1])))
        .attr("y1", yy).attr("y2", yy).attr("stroke", col).attr("stroke-width", 3.5)
        .attr("stroke-linecap", "round").attr("opacity", 0.9);
      gSrsRef.append("text").attr("x", 0).attr("y", yy - 6).attr("font-size", 10)
        .attr("fill", col).text(lab);
    });

    const eff = Math.round(n / Math.max(deff, 1e-9));
    const conv = (key === "convenience");
    out.innerHTML =
      `<b>${D.name}</b> · this draw: ȳ = <b>${ST.fmt(ybar, 2)}</b> against μ = ${ST.fmt(POPMEAN, 2)} · ` +
      `blocks visited = <b>${blocks.size}</b> of 20` +
      (conv
        ? `<br>over ${REPS2} repeats: mean of ȳ = <b>${ST.fmt(mD, 2)}</b> · SD = <b>0.000</b> — there is no randomness ` +
          `left to average over, so this design has no standard error, only a <b>fixed error of ` +
          `${ST.fmt(mD - POPMEAN, 2)}</b> that repeats for ever`
        : `<br>over ${REPS2} repeats: mean of ȳ = <b>${ST.fmt(mD, 2)}</b> · SD = <b>${ST.fmt(sdD, 3)}</b> · ` +
          `exact SRS standard error = <b>${ST.fmt(SE_SRS, 3)}</b> (simulated ${ST.fmt(sdS, 3)})` +
          (off ? ` · ${off} of ${REPS2} land off the axis` : "") +
          `<br>design effect = Var(design)/Var(SRS) = <b>${ST.fmt(deff, 2)}</b> · ` +
          `these 40 observations are worth about <b>${eff}</b> randomly chosen units`) +
      `<br><span style="color:${SC.a2}">${D.note}</span>`;
  }

  selD.addEventListener("change", () => update(false));
  document.getElementById("dz-redraw").addEventListener("click", () => update(true));
  update(false);
})();

/* ─────────────────── 3 · stratified vs simple random, same n ─────────────────── */
(function () {
  const svg = d3.select("#strat-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 700, H = 500;
  /* Three strata. Every number the readout prints is a closed form, not a simulation:
       S²within  = Σ Wₕ Sₕ²            = 228            (fixed)
       S²between = Σ Wₕ (μₕ − μ)²      = 160.8 · sep²
       Var_SRS   = (S²within + S²between)/n
       Var_prop  = S²within/n
       Var_opt   = (Σ Wₕ Sₕ)²/n        = 207.36/n       (Neyman allocation)
       Var_equal = 3 Σ Wₕ² Sₕ²/n       = 378/n                                       */
  const Wt = [0.6, 0.3, 0.1], Sd = [18, 10, 6], dev = [-10, 12, 24], MU = 100;
  const NAMES = ["A", "B", "C"];
  const COLS = [SC.accent, SC.violet, SC.teal];
  const S2W = Wt.reduce((s, w, h) => s + w * Sd[h] * Sd[h], 0);            // 228
  const SW = Wt.reduce((s, w, h) => s + w * Sd[h], 0);                     // 14.4
  const S2OPT = SW * SW;                                                   // (Σ WₕSₕ)² = 207.36

  const F = ST.frame(svg, W, H, { l: 40, r: 18, t: 22, b: 36 });
  const XLO = 20, XHI = 190;
  const xs = d3.scaleLinear().domain([XLO, XHI]).range([0, F.iw]);
  const gStrata = F.g.append("g");
  const gDist = F.g.append("g").attr("transform", "translate(0,250)");

  gStrata.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.muted)
    .text("the population, split into three strata (height ∝ share of the population)");
  const rr = ST.rng(31337);
  const zdots = NAMES.map(() => d3.range(70).map(() => ST.randn(rr)));
  const jit = NAMES.map((_, h) => d3.range(70).map(() => rr()));

  const gDots = gStrata.append("g"), gBars = gStrata.append("g"), gLabs = gStrata.append("g");

  const DH = 175;                                      // sampling-distribution panel height
  const xd = d3.scaleLinear().range([0, F.iw]);
  const axD = gDist.append("g").attr("class", "axis").attr("transform", `translate(0,${DH})`);
  gDist.append("text").attr("x", F.iw).attr("y", DH + 31).attr("text-anchor", "end")
    .attr("font-size", 11).attr("fill", SC.muted).text("sampling distribution of the estimated overall mean");
  const pSRS = gDist.append("path").attr("fill", SC.accent).attr("fill-opacity", 0.16)
    .attr("stroke", SC.accent).attr("stroke-width", 2);
  const pSTR = gDist.append("path").attr("fill", SC.good).attr("fill-opacity", 0.20)
    .attr("stroke", SC.good).attr("stroke-width", 2);
  const pOPT = gDist.append("path").attr("fill", "none")
    .attr("stroke", SC.violet).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 4");
  gDist.append("line").attr("y1", 0).attr("y2", DH).attr("x1", 0).attr("x2", 0).attr("stroke", "none");
  ST.legend(gDist, [
    { label: "simple random sample", color: SC.accent },
    { label: "stratified, chosen allocation", color: SC.good },
    { label: "stratified, Neyman-optimal", color: SC.violet, dash: "5 4" }
  ], 8, 10, { gap: 14, font: 10.5 });

  const sN = document.getElementById("st-n"), sNv = document.getElementById("st-nv");
  const sS = document.getElementById("st-sep"), sSv = document.getElementById("st-sepv");
  const selA = document.getElementById("st-alloc");
  const out = document.getElementById("strat-readout");

  function update() {
    const n = +sN.value, sep = +sS.value;
    sNv.textContent = n; sSv.textContent = ST.fmt(sep, 2);
    const mu = dev.map(d => MU + sep * d);
    const S2B = Wt.reduce((s, w, h) => s + w * (mu[h] - MU) * (mu[h] - MU), 0);
    const S2 = S2W + S2B;

    // allocation
    let nh;
    if (selA.value === "prop") nh = Wt.map(w => n * w);
    else if (selA.value === "equal") nh = Wt.map(() => n / 3);
    else nh = Wt.map((w, h) => n * w * Sd[h] / SW);
    const varStr = Wt.reduce((s, w, h) => s + w * w * Sd[h] * Sd[h] / nh[h], 0);
    const varSRS = S2 / n, varOpt = S2OPT / n;

    // strata strips
    const rows = [];
    let yy = 6;
    Wt.forEach((w, h) => {
      const hgt = 30 + 110 * w;
      rows.push({ h: h, y: yy, hgt: hgt });
      yy += hgt + 8;
    });
    const dsel = gDots.selectAll("g.str").data(rows);
    const dent = dsel.enter().append("g").attr("class", "str");
    dent.append("rect"); dent.append("g").attr("class", "pts"); dent.append("line");
    const dall = dent.merge(dsel);
    dall.select("rect").attr("x", 0).attr("y", d => d.y).attr("width", F.iw).attr("height", d => d.hgt)
      .attr("rx", 5).attr("fill", d => COLS[d.h]).attr("fill-opacity", 0.06)
      .attr("stroke", d => COLS[d.h]).attr("stroke-opacity", 0.30);
    dall.each(function (d) {
      const g = d3.select(this).select("g.pts");
      const pts = zdots[d.h].map((z, i) => ({ v: mu[d.h] + Sd[d.h] * z, j: jit[d.h][i] }));
      const s = g.selectAll("circle").data(pts);
      s.exit().remove();
      s.enter().append("circle").attr("r", 2.6).merge(s)
        .attr("cx", p => xs(ST.clamp(p.v, XLO, XHI)))
        .attr("cy", p => d.y + 21 + p.j * (d.hgt - 28))
        .attr("fill", COLS[d.h]).attr("fill-opacity", 0.62);
      d3.select(this).select("line")
        .attr("x1", xs(mu[d.h])).attr("x2", xs(mu[d.h])).attr("y1", d.y + 19).attr("y2", d.y + d.hgt - 2)
        .attr("stroke", COLS[d.h]).attr("stroke-width", 2.4);
    });
    dsel.exit().remove();

    const lsel = gLabs.selectAll("text").data(rows);
    lsel.enter().append("text").merge(lsel)
      .attr("x", 4).attr("y", d => d.y + 14).attr("font-size", 10.5).attr("fill", SC.muted)
      .text(d => `stratum ${NAMES[d.h]} · W = ${ST.fmt(Wt[d.h], 2)} · S = ${Sd[d.h]} · μ = ${ST.fmt(mu[d.h], 1)} · n = ${Math.round(nh[d.h])}`);
    lsel.exit().remove();

    gBars.selectAll("line").data([0]).join("line")
      .attr("x1", xs(MU)).attr("x2", xs(MU)).attr("y1", 0).attr("y2", yy - 8)
      .attr("stroke", SC.ink).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 4").attr("opacity", 0.6);

    // sampling distributions, drawn from the exact variances
    const seMax = Math.sqrt(varSRS);
    xd.domain([MU - 4 * seMax, MU + 4 * seMax]);
    axD.call(d3.axisBottom(xd).ticks(7).tickFormat(d3.format(".1f")));
    const grid = ST.linspace(xd.domain()[0], xd.domain()[1], 220);
    const peak = 1 / Math.sqrt(2 * Math.PI * Math.min(varSRS, varStr, varOpt));
    const ysc = d3.scaleLinear().domain([0, peak * 1.08]).range([DH, 0]);
    const area = v => d3.area().x(g => xd(g)).y0(DH)
      .y1(g => ysc(ST.normPdf((g - MU) / Math.sqrt(v)) / Math.sqrt(v)))(grid);
    const line = v => d3.line().x(g => xd(g))
      .y(g => ysc(ST.normPdf((g - MU) / Math.sqrt(v)) / Math.sqrt(v)))(grid);
    pSRS.attr("d", area(varSRS));
    pSTR.attr("d", area(varStr));
    pOPT.attr("d", line(varOpt));

    const deff = varStr / varSRS;
    out.innerHTML =
      `S²<sub>within</sub> = Σ W<sub>h</sub>S<sub>h</sub>² = <b>${ST.fmt(S2W, 1)}</b> · ` +
      `S²<sub>between</sub> = Σ W<sub>h</sub>(μ<sub>h</sub> − μ)² = <b>${ST.fmt(S2B, 1)}</b> · ` +
      `population S² = <b>${ST.fmt(S2, 1)}</b>` +
      `<br>at n = ${n}: Var(SRS) = <b>${ST.fmt(varSRS, 4)}</b> · Var(stratified) = <b>${ST.fmt(varStr, 4)}</b> · ` +
      `Var(Neyman) = <b>${ST.fmt(varOpt, 4)}</b>` +
      `<br>SE: <b>${ST.fmt(Math.sqrt(varSRS), 3)}</b> → <b>${ST.fmt(Math.sqrt(varStr), 3)}</b> · ` +
      `design effect = <b>${ST.fmt(deff, 3)}</b> · a simple random sample would need ` +
      `<b>${Math.round(n / deff)}</b> units to match it` +
      `<br><span style="color:${SC.a2}">` +
      (S2B < 1e-9
        ? "the strata have identical means, so there is no between-strata variance to remove — stratification buys exactly nothing, and the two curves coincide"
        : `the entire gain is the between-strata term: Var(SRS) − Var(proportional) = S²<sub>between</sub>/n = ` +
          `<b>${ST.fmt(S2B / n, 4)}</b>. Stratifying deletes it from the estimator`) +
      `</span>`;
  }

  [sN, sS].forEach(el => el.addEventListener("input", update));
  selA.addEventListener("change", update);
  update();
})();

/* ─────────────────── 4 · the sample-size illusion ─────────────────── */
(function () {
  const svg = d3.select("#illusion-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 700, H = 420;
  const SIG = 50;                                  // σ in percentage points for a 50/50 proportion
  const rmse = (b, n) => Math.sqrt(b * b + SIG * SIG / n);

  const F = ST.frame(svg, W, H, { l: 56, r: 150, t: 22, b: 44 });
  const x = d3.scaleLog().domain([10, 1e7]).range([0, F.iw]);
  const y = d3.scaleLog().domain([0.03, 20]).range([F.ih, 0]);
  F.g.append("g").attr("class", "gridlines").selectAll("line")
    .data([10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]).join("line")
    .attr("x1", d => x(d)).attr("x2", d => x(d)).attr("y1", 0).attr("y2", F.ih).attr("stroke", SC.grid);
  F.g.append("g").attr("class", "gridlines").selectAll("line")
    .data([0.1, 1, 10]).join("line")
    .attr("y1", d => y(d)).attr("y2", d => y(d)).attr("x1", 0).attr("x2", F.iw).attr("stroke", SC.grid);
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
    .call(d3.axisBottom(x).ticks(7, "~s"));
  F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5, "~g"));
  F.g.append("text").attr("x", F.iw).attr("y", F.ih + 36).attr("text-anchor", "end")
    .attr("font-size", 11).attr("fill", SC.muted).text("sample size n (log scale)");
  F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.muted)
    .text("typical error of the estimate, in percentage points (log scale)");

  const grid = ST.linspace(1, 7, 220).map(e => Math.pow(10, e));
  const path = b => d3.line().x(n => x(n)).y(n => y(ST.clamp(rmse(b, n), 0.03, 20)))(grid);

  F.g.append("path").attr("d", path(0)).attr("fill", "none")
    .attr("stroke", SC.accent).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
  F.g.append("text").attr("x", x(2e6)).attr("y", y(rmse(0, 2e6)) - 8).attr("font-size", 10.5)
    .attr("fill", SC.accent).attr("text-anchor", "end").text("no bias: σ/√n, falls for ever");

  const ghosts = [0.5, 1, 2, 5];
  ghosts.forEach(b => {
    F.g.append("path").attr("d", path(b)).attr("fill", "none")
      .attr("stroke", SC.muted).attr("stroke-width", 1).attr("stroke-opacity", 0.35);
    F.g.append("text").attr("x", F.iw + 6).attr("y", y(b) + 3.5).attr("font-size", 10)
      .attr("fill", SC.muted).text(`bias ${b} pt${b === 1 ? "" : "s"}`);
  });

  const live = F.g.append("path").attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 2.6);
  const floor = F.g.append("line").attr("stroke", SC.bad).attr("stroke-width", 1.4)
    .attr("stroke-dasharray", "3 4").attr("opacity", 0.8);
  const vline = F.g.append("line").attr("y1", 0).attr("y2", F.ih)
    .attr("stroke", SC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 4");
  const cross = F.g.append("circle").attr("r", 5.5).attr("fill", "none")
    .attr("stroke", SC.good).attr("stroke-width", 2.4);
  const crossT = F.g.append("text").attr("font-size", 10.5).attr("fill", SC.good).attr("text-anchor", "middle");
  const dotL = F.g.append("circle").attr("r", 4.5).attr("fill", SC.a2);
  const dotS = F.g.append("circle").attr("r", 3.5).attr("fill", SC.accent);

  const sB = document.getElementById("il-b"), sBv = document.getElementById("il-bv");
  const sN = document.getElementById("il-n"), sNv = document.getElementById("il-nv");
  const out = document.getElementById("illusion-readout");

  function update() {
    const b = +sB.value, n = Math.round(Math.pow(10, +sN.value));
    sBv.textContent = ST.fmt(b, 1) + " pts";
    sNv.textContent = d3.format(",")(n);
    live.attr("d", path(b));
    floor.attr("x1", 0).attr("x2", F.iw).attr("y1", y(ST.clamp(b, 0.03, 20))).attr("y2", y(ST.clamp(b, 0.03, 20)))
      .attr("opacity", b > 0.03 ? 0.8 : 0);
    vline.attr("x1", x(n)).attr("x2", x(n));
    dotL.attr("cx", x(n)).attr("cy", y(ST.clamp(rmse(b, n), 0.03, 20)));
    dotS.attr("cx", x(n)).attr("cy", y(ST.clamp(SIG / Math.sqrt(n), 0.03, 20)));

    const nStar = (SIG / b) * (SIG / b);
    const show = b > 0 && nStar >= 10 && nStar <= 1e7;
    cross.attr("opacity", show ? 1 : 0).attr("cx", show ? x(nStar) : 0)
      .attr("cy", show ? y(ST.clamp(rmse(b, nStar), 0.03, 20)) : 0);
    crossT.attr("opacity", show ? 1 : 0).attr("x", show ? x(nStar) : 0)
      .attr("y", show ? y(ST.clamp(rmse(b, nStar), 0.03, 20)) - 12 : 0)
      .text(show ? `n* = ${d3.format(",")(Math.round(nStar))}` : "");

    const se = SIG / Math.sqrt(n), rm = rmse(b, n), nEq = (SIG * SIG) / (rm * rm);
    out.innerHTML =
      `n = <b>${d3.format(",")(n)}</b> · chance error SE = σ/√n = <b>${ST.fmt(se, 3)}</b> pts · ` +
      `bias = <b>${ST.fmt(b, 2)}</b> pts · RMSE = √(bias² + SE²) = <b>${ST.fmt(rm, 3)}</b> pts` +
      (b > 0
        ? `<br>the two are equal at n* = (σ/bias)² = <b>${d3.format(",")(Math.round(nStar))}</b>; ` +
          `past that point every extra observation buys you almost nothing` +
          `<br><span style="color:${SC.a2}">this sample of ${d3.format(",")(n)} is as accurate as a perfectly drawn ` +
          `random sample of <b>${d3.format(",")(Math.max(1, Math.round(nEq)))}</b> — ` +
          `and as n → ∞ that number stops at (σ/bias)² = ${d3.format(",")(Math.round(nStar))}</span>`
        : `<br><span style="color:${SC.good}">with no bias the error is pure chance error: it falls like 1/√n for ever, ` +
          `and quadrupling n halves it</span>`);
  }
  [sB, sN].forEach(el => el.addEventListener("input", update));
  document.querySelectorAll("[data-il]").forEach(btn => btn.addEventListener("click", () => {
    const [bb, nn] = btn.getAttribute("data-il").split(",");
    sB.value = bb; sN.value = nn; update();
  }));
  update();
})();

/* ─────────────────── 5 · confounding and the back-door path ─────────────────── */
(function () {
  const svg = d3.select("#confound-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 700, H = 450, NP = 260;
  const F = ST.frame(svg, W, H, { l: 48, r: 214, t: 20, b: 40 });

  const base = (function () {
    const r = ST.rng(1637), out = [];
    for (let i = 0; i < NP; i++) out.push({ z: r() * 2 - 1, ex: ST.randn(r), ey: ST.randn(r) });
    return out;
  })();
  const SX = 0.6, SY = 0.5;                       // noise SDs on exposure and outcome
  const VARZ = 1 / 3;                             // Var of Uniform(−1, 1)

  const x = d3.scaleLinear().domain([-4.2, 4.2]).range([0, F.iw]);
  const y = d3.scaleLinear().domain([-4.6, 4.6]).range([F.ih, 0]);
  ST.gridY(F.g, y, F.iw, 5); ST.gridX(F.g, x, F.ih, 5);
  ST.axisB(F.g, x, F.ih, 6, "exposure X  (the thing you would change)");
  ST.axisL(F.g, y, 5, "outcome Y");
  const zc = d3.scaleSequential(d3.interpolateCool).domain([-1.15, 1.15]);
  const gBands = F.g.append("g"), gPts = F.g.append("g"), gLines = F.g.append("g");
  const lMarg = gLines.append("line").attr("stroke", SC.bad).attr("stroke-width", 2.6).attr("stroke-dasharray", "7 5");
  const lAdj = gLines.append("line").attr("stroke", SC.good).attr("stroke-width", 2.6);

  // the DAG panel
  const gD = F.g.append("g").attr("transform", `translate(${F.iw + 34},14)`);
  const NODE = { z: [70, 16], x: [8, 116], y: [132, 116] };
  const eZX = gD.append("line").attr("marker-end", "url(#cfarrow)");
  const eZY = gD.append("line").attr("marker-end", "url(#cfarrow)");
  const eXY = gD.append("line").attr("marker-end", "url(#cfarrow)");
  svg.append("defs").append("marker").attr("id", "cfarrow").attr("viewBox", "0 0 10 10")
    .attr("refX", 9).attr("refY", 5).attr("markerWidth", 5).attr("markerHeight", 5)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", SC.muted);
  const nodeG = gD.append("g");
  Object.entries(NODE).forEach(([k, p]) => {
    nodeG.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 17)
      .attr("fill", SC.panel2).attr("stroke", k === "z" ? SC.violet : SC.accent).attr("stroke-width", 2);
    nodeG.append("text").attr("x", p[0]).attr("y", p[1] + 5).attr("text-anchor", "middle")
      .attr("font-size", 14).attr("fill", SC.ink).text(k.toUpperCase());
  });
  gD.append("text").attr("x", NODE.z[0]).attr("y", NODE.z[1] - 26).attr("text-anchor", "middle")
    .attr("font-size", 10).attr("fill", SC.violet).text("lurking variable");
  const dagNote = gD.append("text").attr("x", 0).attr("y", 166).attr("font-size", 10).attr("fill", SC.muted);
  const dagNote2 = gD.append("text").attr("x", 0).attr("y", 180).attr("font-size", 10).attr("fill", SC.muted);
  const dagNote3 = gD.append("text").attr("x", 0).attr("y", 194).attr("font-size", 10).attr("fill", SC.muted);

  const sA = document.getElementById("cf-a"), sAv = document.getElementById("cf-av");
  const sG = document.getElementById("cf-g"), sGv = document.getElementById("cf-gv");
  const sB = document.getElementById("cf-b"), sBv = document.getElementById("cf-bv");
  const cbC = document.getElementById("cf-cond");
  const out = document.getElementById("confound-readout");

  function edge(sel, from, to, w, col, op) {
    const [x1, y1] = NODE[from], [x2, y2] = NODE[to];
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), R = 20;
    sel.attr("x1", x1 + dx / L * R).attr("y1", y1 + dy / L * R)
      .attr("x2", x2 - dx / L * R).attr("y2", y2 - dy / L * R)
      .attr("stroke", col).attr("stroke-width", w).attr("opacity", op);
  }

  function update() {
    const a = +sA.value, g = +sG.value, b = +sB.value;
    sAv.textContent = ST.fmt(a, 2); sGv.textContent = ST.fmt(g, 2); sBv.textContent = ST.fmt(b, 2);

    const pts = base.map(d => {
      const X = a * d.z + SX * d.ex;
      return { z: d.z, X: X, Y: b * X + g * d.z + SY * d.ey };
    });
    const px = pts.map(p => p.X), py = pts.map(p => p.Y);
    const marg = ST.lsLine(px, py);

    // adjusted slope by the residual (Frisch–Waugh) route: regress X on Z, then Y on the residuals
    const pz = pts.map(p => p.z);
    const xz = ST.lsLine(pz, px);
    const rx = pts.map(p => p.X - (xz.a + xz.b * p.z));
    const adj = ST.lsLine(rx, py);

    const sel = gPts.selectAll("circle").data(pts);
    sel.exit().remove();
    sel.enter().append("circle").attr("r", 3.4).merge(sel)
      .attr("cx", p => x(ST.clamp(p.X, -4.2, 4.2))).attr("cy", p => y(ST.clamp(p.Y, -4.6, 4.6)))
      .attr("fill", p => zc(p.z)).attr("fill-opacity", 0.82)
      .attr("stroke", "#0f1117").attr("stroke-width", 0.8);

    lMarg.attr("x1", x(-4.2)).attr("y1", y(ST.clamp(marg.a + marg.b * -4.2, -4.6, 4.6)))
      .attr("x2", x(4.2)).attr("y2", y(ST.clamp(marg.a + marg.b * 4.2, -4.6, 4.6)));

    // within-band fits: split Z into four bands and fit inside each
    const bands = [[-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1]];
    const fits = bands.map(([lo, hi]) => {
      const s = pts.filter(p => p.z >= lo && p.z < hi + 1e-9);
      if (s.length < 5) return null;
      const f = ST.lsLine(s.map(p => p.X), s.map(p => p.Y));
      const xr = d3.extent(s, p => p.X);
      return { f: f, x0: xr[0], x1: xr[1], z: (lo + hi) / 2, n: s.length };
    }).filter(Boolean);
    const bs = gBands.selectAll("line").data(cbC.checked ? fits : []);
    bs.exit().remove();
    bs.enter().append("line").attr("stroke-width", 3).merge(bs)
      .attr("x1", d => x(ST.clamp(d.x0, -4.2, 4.2))).attr("y1", d => y(ST.clamp(d.f.a + d.f.b * d.x0, -4.6, 4.6)))
      .attr("x2", d => x(ST.clamp(d.x1, -4.2, 4.2))).attr("y2", d => y(ST.clamp(d.f.a + d.f.b * d.x1, -4.6, 4.6)))
      .attr("stroke", d => zc(d.z)).attr("opacity", 0.95);

    lAdj.attr("opacity", cbC.checked ? 1 : 0);
    if (cbC.checked) {
      const mx = ST.mean(px), my = ST.mean(py);
      lAdj.attr("x1", x(-4.2)).attr("y1", y(ST.clamp(my + adj.b * (-4.2 - mx), -4.6, 4.6)))
        .attr("x2", x(4.2)).attr("y2", y(ST.clamp(my + adj.b * (4.2 - mx), -4.6, 4.6)));
    }

    const backdoor = Math.abs(a) > 1e-9 && Math.abs(g) > 1e-9;
    edge(eZX, "z", "x", 1 + 3 * Math.abs(a) / 2, backdoor ? SC.bad : SC.muted, Math.abs(a) < 1e-9 ? 0.15 : 0.95);
    edge(eZY, "z", "y", 1 + 3 * Math.abs(g) / 2.5, backdoor ? SC.bad : SC.muted, Math.abs(g) < 1e-9 ? 0.15 : 0.95);
    edge(eXY, "x", "y", 1 + 3 * Math.abs(b), SC.good, Math.abs(b) < 1e-9 ? 0.18 : 0.95);
    dagNote.text("back-door path  X ← Z → Y").attr("fill", backdoor ? SC.bad : SC.good);
    dagNote2.text(backdoor ? "is OPEN, so the marginal" : "is CLOSED, so the marginal")
      .attr("fill", backdoor ? SC.bad : SC.good);
    dagNote3.text(backdoor ? "slope is not the effect" : "slope IS the causal effect")
      .attr("fill", backdoor ? SC.bad : SC.good);

    const ovb = g * a * VARZ / (a * a * VARZ + SX * SX);
    out.innerHTML =
      `true causal effect β = <b>${ST.fmt(b, 2)}</b> · ` +
      `<span style="color:${SC.bad}">marginal slope of Y on X = <b>${ST.fmt(marg.b, 3)}</b></span> · ` +
      `<span style="color:${SC.good}">slope adjusted for Z = <b>${ST.fmt(adj.b, 3)}</b></span>` +
      `<br>predicted confounding bias = γ·a·Var(Z)/Var(X) = ` +
      `${ST.fmt(g, 2)}·${ST.fmt(a, 2)}·${ST.fmt(VARZ, 3)} / ${ST.fmt(a * a * VARZ + SX * SX, 3)} = ` +
      `<b>${ST.fmt(ovb, 3)}</b> · observed β + bias = <b>${ST.fmt(b + ovb, 3)}</b>` +
      `<br><span style="color:${SC.a2}">` +
      (Math.abs(a) < 1e-9
        ? "with a = 0 the exposure no longer depends on Z — that is exactly what randomisation does — and the marginal slope estimates the causal effect with nothing to adjust for"
        : (Math.sign(marg.b) !== Math.sign(b) && Math.abs(b) > 0.05
          ? "the association now points the OPPOSITE WAY to the causal effect. No amount of extra data fixes this; only conditioning on Z, or randomising X, does"
          : "the cloud shows an association that is partly, or entirely, the lurking variable's doing. Tick the box to fit inside bands of Z and watch it change")) +
      `</span>`;
  }
  [sA, sG, sB].forEach(el => el.addEventListener("input", update));
  cbC.addEventListener("change", update);
  document.querySelectorAll("[data-cf]").forEach(btn => btn.addEventListener("click", () => {
    const [aa, gg, bb] = btn.getAttribute("data-cf").split(",");
    sA.value = aa; sG.value = gg; sB.value = bb; update();
  }));
  update();
})();

/* ─────────────────── 6 · what randomisation buys ─────────────────── */
(function () {
  const svg = d3.select("#rand-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 700, H = 440, REPS = 400;
  const F = ST.frame(svg, W, H, { l: 44, r: 16, t: 22, b: 36 });

  /* Every unit carries a latent eagerness E. The MEASURED covariate M and the
     UNMEASURED covariate U are both driven by E, so anything that sorts people by
     eagerness — including their own choice — unbalances BOTH.                    */
  function makePop(n, seed) {
    const r = ST.rng(seed), out = [];
    for (let i = 0; i < n; i++) {
      const e = ST.randn(r);
      out.push({ e: e, m: 0.55 * e + 0.835 * ST.randn(r), u: 0.80 * e + 0.60 * ST.randn(r) });
    }
    return out;
  }

  const MECH = {
    randomise: {
      name: "randomise",
      note: "a coin flip decides. Nothing about the unit — measured, unmeasured, unimagined — has any say, so both gaps are centred on zero and both shrink like 1/√n. This is the entire argument for the randomised experiment.",
      assign: (pop, r) => {
        const idx = ST.shuffle(d3.range(pop.length), r);
        const t = new Set(idx.slice(0, pop.length / 2));
        return pop.map((_, i) => t.has(i));
      }
    },
    selfselect: {
      name: "let people choose",
      note: "the eager half opts in. Both gaps are large, both have the wrong sign, and neither shrinks as n grows: more subjects just measures the same bias more precisely.",
      assign: (pop, r) => {
        const sc = pop.map(p => p.e + 0.35 * ST.randn(r));
        const cut = ST.median(sc);
        return sc.map(s => s > cut);
      }
    },
    matched: {
      name: "balance the measured variable only",
      note: "sort on M, pair the neighbours, then let each pair's more eager member take the treatment. M is balanced almost perfectly by construction — and U is as badly out of balance as ever. You can only balance what you thought to measure.",
      assign: (pop, r) => {
        const ord = d3.range(pop.length).sort((i, j) => pop[i].m - pop[j].m);
        const t = new Array(pop.length).fill(false);
        for (let k = 0; k + 1 < ord.length; k += 2) {
          const i = ord[k], j = ord[k + 1];
          const win = (pop[i].e + 0.3 * ST.randn(r) > pop[j].e + 0.3 * ST.randn(r)) ? i : j;
          t[win] = true;
        }
        return t;
      }
    }
  };

  // top: the current assignment, as two strips (M and U); bottom: 400 repeats
  const PANW = (F.iw - 26) / 2, TOPH = 132, BOTY = 190, BOTH = 176;
  const xs = d3.scaleLinear().domain([-3.2, 3.2]);
  const panels = [
    { key: "m", label: "the MEASURED covariate", x: 0, col: SC.accent },
    { key: "u", label: "the UNMEASURED covariate", x: PANW + 26, col: SC.violet }
  ];
  const gTop = F.g.append("g"), gBot = F.g.append("g").attr("transform", `translate(0,${BOTY})`);
  panels.forEach(p => {
    p.g = gTop.append("g").attr("transform", `translate(${p.x},0)`);
    p.gb = gBot.append("g").attr("transform", `translate(${p.x},0)`);
    p.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", p.col).text(p.label);
    p.g.append("rect").attr("x", 0).attr("y", 0).attr("width", PANW).attr("height", TOPH)
      .attr("rx", 6).attr("fill", p.col).attr("fill-opacity", 0.045);
    p.g.append("text").attr("x", 4).attr("y", 16).attr("font-size", 10).attr("fill", SC.muted).text("treatment");
    p.g.append("text").attr("x", 4).attr("y", TOPH - 6).attr("font-size", 10).attr("fill", SC.muted).text("control");
    p.pts = p.g.append("g"); p.marks = p.g.append("g");
    p.axb = p.gb.append("g").attr("class", "axis").attr("transform", `translate(0,${BOTH})`);
    p.gb.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", SC.muted)
      .text("gap x̄(treated) − x̄(control) over " + REPS + " assignments");
    p.dots = p.gb.append("g");
    p.zero = p.gb.append("line").attr("y1", 0).attr("y2", BOTH).attr("stroke", SC.good).attr("stroke-width", 1.6);
  });

  const sN = document.getElementById("rd-n"), sNv = document.getElementById("rd-nv");
  const selM = document.getElementById("rd-mech");
  const out = document.getElementById("rand-readout");

  function update(reseed) {
    const n = 2 * Math.round(+sN.value / 2);
    sNv.textContent = n;
    const M = MECH[selM.value];
    const pop = makePop(n, reseed ? Math.floor(Math.random() * 1e9) : 5150);
    const r = ST.rng(reseed ? Math.floor(Math.random() * 1e9) : 60606);

    const one = M.assign(pop, r);
    const stats = {};
    panels.forEach(p => {
      const t = pop.filter((_, i) => one[i]).map(u => u[p.key]);
      const c = pop.filter((_, i) => !one[i]).map(u => u[p.key]);
      stats[p.key] = { gap: ST.mean(t) - ST.mean(c), mt: ST.mean(t), mc: ST.mean(c) };
      xs.range([12, PANW - 12]);
      const rows = pop.map((u, i) => ({ v: u[p.key], t: one[i], j: (i * 2654435761 % 1000) / 1000 }));
      const s = p.pts.selectAll("circle").data(rows);
      s.exit().remove();
      s.enter().append("circle").attr("r", 3).merge(s)
        .attr("cx", d => xs(ST.clamp(d.v, -3.2, 3.2)))
        .attr("cy", d => (d.t ? 24 : TOPH / 2 + 14) + d.j * (TOPH / 2 - 34))
        .attr("fill", d => d.t ? SC.a2 : SC.muted).attr("fill-opacity", 0.75);
      p.marks.selectAll("*").remove();
      [[stats[p.key].mt, SC.a2, 20, TOPH / 2 - 6], [stats[p.key].mc, SC.ink, TOPH / 2 + 10, TOPH - 4]].forEach(([m, col, y0, y1]) => {
        p.marks.append("line").attr("x1", xs(ST.clamp(m, -3.2, 3.2))).attr("x2", xs(ST.clamp(m, -3.2, 3.2)))
          .attr("y1", y0).attr("y2", y1).attr("stroke", col).attr("stroke-width", 2.4);
      });
    });

    // REPS repeats, fresh population each time so the picture is the design, not one draw
    const gaps = { m: [], u: [] };
    const rr = ST.rng(24680);
    for (let k = 0; k < REPS; k++) {
      const pk = makePop(n, 1000 + k * 7919);
      const asg = M.assign(pk, rr);
      panels.forEach(p => {
        const t = pk.filter((_, i) => asg[i]).map(u => u[p.key]);
        const c = pk.filter((_, i) => !asg[i]).map(u => u[p.key]);
        gaps[p.key].push(ST.mean(t) - ST.mean(c));
      });
    }

    /* Shared x-axis so the two panels are directly comparable; bars rather than a dot
       stack, because a mechanism that pins one covariate can put nearly all 400 gaps in
       a single bin and a dot column would run off the top of the panel.               */
    const lim = Math.max(0.35, d3.max([...gaps.m, ...gaps.u].map(Math.abs)) * 1.12);
    const xg = d3.scaleLinear().domain([-lim, lim]).range([10, PANW - 10]);
    const NBINS = 52, BWid = 2 * lim / NBINS;
    const binned = {};
    let maxC = 1;
    panels.forEach(p => {
      const counts = new Map();
      gaps[p.key].forEach(v => {
        const b = Math.round(v / BWid);
        counts.set(b, (counts.get(b) || 0) + 1);
      });
      binned[p.key] = [...counts.entries()].map(([b, c]) => ({ v: b * BWid, c: c }));
      maxC = Math.max(maxC, d3.max(binned[p.key], d => d.c));
    });
    const yg = d3.scaleLinear().domain([0, maxC]).range([0, BOTH - 6]);
    const bw = Math.max(2.2, (PANW - 20) / NBINS - 1.2);
    panels.forEach(p => {
      p.axb.call(d3.axisBottom(xg).ticks(5).tickFormat(d3.format(".2f")));
      p.zero.attr("x1", xg(0)).attr("x2", xg(0));
      const s = p.dots.selectAll("rect").data(binned[p.key]);
      s.exit().remove();
      s.enter().append("rect").attr("rx", 1).merge(s)
        .attr("x", d => xg(d.v) - bw / 2).attr("width", bw)
        .attr("y", d => BOTH - 2 - yg(d.c)).attr("height", d => Math.max(1, yg(d.c)))
        .attr("fill", p.col).attr("fill-opacity", 0.75);
    });

    const sdM = ST.sd(gaps.m), sdU = ST.sd(gaps.u);
    const mM = ST.mean(gaps.m), mU = ST.mean(gaps.u);
    const theory = 2 / Math.sqrt(n);                       // SD of the gap for unit-variance covariates
    out.innerHTML =
      `<b>${M.name}</b> · n = ${n} (${n / 2} per group) · this draw's gaps: ` +
      `M = <b>${ST.fmt(stats.m.gap, 3)}</b>, U = <b>${ST.fmt(stats.u.gap, 3)}</b>` +
      `<br>over ${REPS} assignments — measured: mean gap <b>${ST.fmt(mM, 3)}</b>, SD <b>${ST.fmt(sdM, 3)}</b> · ` +
      `unmeasured: mean gap <b>${ST.fmt(mU, 3)}</b>, SD <b>${ST.fmt(sdU, 3)}</b>` +
      (selM.value === "randomise"
        ? `<br>theory for a random split of n units with SD ≈ 1: SD(gap) = 2S/√n = <b>${ST.fmt(theory, 3)}</b> — ` +
          `and it applies to <i>every</i> variable at once`
        : `<br>a fair split would have had SD(gap) ≈ <b>${ST.fmt(theory, 3)}</b> and mean 0`) +
      `<br><span style="color:${SC.a2}">${M.note}</span>`;
  }

  sN.addEventListener("input", () => update(false));
  selM.addEventListener("change", () => update(false));
  document.getElementById("rd-again").addEventListener("click", () => update(true));
  update(false);
})();
