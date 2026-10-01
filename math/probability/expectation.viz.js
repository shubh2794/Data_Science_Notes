/* expectation.viz.js — the eight visualizations on math/probability/expectation.html.
   Loaded after ../../data.js → ../../notes.js → ../statistics/stats-viz.js (SC palette, ST toolbox)
   → prob-viz.js (PV: Poisson, geometric, negative binomial, gamma, … draws and pmfs, PV.simpson).
   Distribution functions come from ST / PV and are never re-implemented here.

   Everything page-local lives under the single namespace EV. Each figure is an IIFE that exits
   quietly if its <svg> is not on the page. The pure simulation functions are exposed on EV so the
   build's node checks can call them without a DOM.

     1  #com-svg      expectation as the balance point: draggable pmf, pivot, torque E X − c
     2  #stpete-svg   St. Petersburg running mean vs log₂ n; Cauchy vs normal running means
     3  #mgf-svg      the mgf: curve, tangent slope E X, Taylor parabola, n independent copies
     4  #jensen-svg   Jensen: chord vs curve for a draggable two-point distribution
     5  #bounds-svg   Markov / Chebyshev / Cantelli / Chernoff / Hoeffding against the true tail
     6  #conc-svg     concentration of a mean (minibatch view) and the worst of k hypotheses
     7  #pred-svg     E[Y | X] as the best predictor: curve vs best line vs constant; median under |·|
     8  #totvar-svg   law of total variance: groups (ANOVA), gamma–Poisson, hierarchical binomial   */

const EV = {
  // Default seeds, chosen by the caption audit (see the note at each figure).
  seeds: { stpete: 8, conc: 11, concK: 1, pred: 3, totvar: 1 },
  fmt: (x, d) => ST.fmt(x, d === undefined ? 4 : d),
  label(g, x, y, t, o) {
    const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", (o && o.size) || 11)
      .attr("fill", (o && o.color) || SC.muted).text(t);
    if (o && o.anchor) e.attr("text-anchor", o.anchor);
    if (o && o.weight) e.attr("font-weight", o.weight);
    return e;
  },
  // Binomial(n, p) sampler by inverse cdf on a precomputed table (exact; O(log n) per draw).
  binomTable(n, p) {
    const c = new Float64Array(n + 1);
    let s = 0;
    for (let k = 0; k <= n; k++) { s += ST.binomPmf(k, n, p); c[k] = s; }
    c[n] = 1;
    return c;
  },
  binomFromTable(c, r) {
    const u = r();
    let lo = 0, hi = c.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (c[m] < u) lo = m + 1; else hi = m; }
    return lo;
  }
};

/* ─────────────────── 1 · expectation is the balance point ───────────────────
   Caption audit: default preset "drill" (2, 3, 4 with 0.1, 0.7, 0.2), pivot c = 3.1 →
   E X = 3.1000, Var X = 0.2900, torque 0, beam level, E(X − c)² = 0.2900 at its minimum.
   "two clumps" → E X = 5 with no mass at 5; "long right tail" → mean 2.9, median 2.             */
EV.presets = {
  drill: { 2: 0.1, 3: 0.7, 4: 0.2 },
  die: { 1: 1 / 6, 2: 1 / 6, 3: 1 / 6, 4: 1 / 6, 5: 1 / 6, 6: 1 / 6 },
  clumps: { 1: 0.5, 9: 0.5 },
  tail: { 1: 0.35, 2: 0.25, 3: 0.15, 4: 0.1, 6: 0.05, 8: 0.05, 10: 0.05 },
  binom: Object.fromEntries(d3.range(11).map(k => [k, ST.binomPmf(k, 10, 0.3)]))
};
EV.pmfStats = function (p) {                 // p: array of 11 probabilities summing to 1
  let m = 0, m2 = 0;
  p.forEach((q, k) => { m += k * q; m2 += k * k * q; });
  let cum = 0, med = 0;
  for (let k = 0; k < p.length; k++) { cum += p[k]; if (cum >= 0.5 - 1e-12) { med = k; break; } }
  return { mean: m, var: m2 - m * m, median: med };
};
(function () {
  const svg = d3.select("#com-svg");
  if (svg.empty()) return;
  const W = 680, H = 340, $ = id => document.getElementById(id);
  const out = $("com-readout"), sel = $("cm-pre"), cS = $("cm-c");
  const m = { l: 46, r: 24, t: 14, b: 30 }, iw = W - m.l - m.r;
  const x = d3.scaleLinear().domain([-0.5, 10.5]).range([0, iw]);
  const barTop = 150, barBot = H - m.b - m.t, yMax = 0.8;
  const y = d3.scaleLinear().domain([0, yMax]).range([barBot, barTop]);
  let p = new Array(11).fill(0), c = 3.1;

  function load(name) {
    p = new Array(11).fill(0);
    Object.entries(EV.presets[name]).forEach(([k, v]) => { p[+k] = v; });
    const s = d3.sum(p); p = p.map(v => v / s);
    c = EV.pmfStats(p).mean;
    cS.value = c; $("cm-cv").textContent = c.toFixed(2);
  }
  function setBar(i, target) {
    target = ST.clamp(target, 0, 1);
    const rest = 1 - p[i], want = 1 - target;
    if (rest <= 1e-12) { p = p.map((v, k) => (k === i ? target : want / 10)); }
    else p = p.map((v, k) => (k === i ? target : v * want / rest));
  }
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const S = EV.pmfStats(p), torque = S.mean - c;
    // bars
    ST.gridY(g, y, iw, 4);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${barBot})`).call(d3.axisBottom(x).ticks(11).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4)).attr("transform", "translate(0,0)");
    EV.label(g, iw, barTop - 9, "p(x) — drag the top of a bar", { anchor: "end" });
    const bw = 30;
    p.forEach((q, k) => {
      const bg = g.append("g").style("cursor", "ns-resize");
      bg.append("rect").attr("x", x(k) - bw / 2).attr("y", y(Math.min(q, yMax))).attr("width", bw)
        .attr("height", barBot - y(Math.min(q, yMax))).attr("fill", SC.accent).attr("opacity", 0.75);
      if (q > 0.0005) EV.label(bg, x(k), y(Math.min(q, yMax)) - 4, q.toFixed(3), { anchor: "middle", size: 9.5, color: SC.ink });
      bg.append("rect").attr("x", x(k) - bw / 2).attr("y", barTop - 10).attr("width", bw).attr("height", barBot - barTop + 10).attr("fill", "transparent");
      bg.call(d3.drag().on("start drag", ev => { setBar(k, y.invert(ev.y)); draw(); }));
    });
    // mean and median markers on the bar axis
    g.append("line").attr("x1", x(S.mean)).attr("x2", x(S.mean)).attr("y1", barTop).attr("y2", barBot).attr("stroke", SC.a2).attr("stroke-width", 2);
    EV.label(g, x(S.mean) + (S.mean > 8 ? -4 : 4), barTop - 22, `E X = ${S.mean.toFixed(3)}`, { color: SC.a2, size: 10.5, anchor: S.mean > 8 ? "end" : "start" });
    g.append("line").attr("x1", x(S.median)).attr("x2", x(S.median)).attr("y1", barTop).attr("y2", barBot).attr("stroke", SC.good).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
    EV.label(g, x(S.median) + (S.median > 8 ? -4 : 4), barTop - 9, `median ${S.median}`, { color: SC.good, size: 10.5, anchor: S.median > 8 ? "end" : "start" });
    // beam
    const by = 64, px = x(c), ang = ST.clamp(torque * 6, -14, 14);
    const beam = g.append("g").attr("transform", `rotate(${ang},${px},${by})`);
    beam.append("line").attr("x1", x(-0.4)).attr("x2", x(10.4)).attr("y1", by).attr("y2", by).attr("stroke", SC.ink).attr("stroke-width", 3);
    p.forEach((q, k) => { if (q > 0.0005) beam.append("circle").attr("cx", x(k)).attr("cy", by - 4 - 26 * Math.sqrt(q)).attr("r", 26 * Math.sqrt(q)).attr("fill", SC.accent).attr("opacity", 0.8); });
    // pivot (draggable)
    const pg = g.append("g").style("cursor", "ew-resize");
    pg.append("path").attr("d", `M${px},${by + 2} L${px - 12},${by + 24} L${px + 12},${by + 24} Z`).attr("fill", SC.a2);
    pg.append("rect").attr("x", px - 16).attr("y", by).attr("width", 32).attr("height", 30).attr("fill", "transparent");
    pg.call(d3.drag().on("drag", ev => { c = ST.clamp(x.invert(ev.x), 0, 10); cS.value = c; $("cm-cv").textContent = c.toFixed(2); draw(); }));
    EV.label(g, px, by + 38, `pivot c = ${c.toFixed(2)}`, { anchor: "middle", color: SC.a2, size: 10.5 });
    const lvl = Math.abs(torque) < 0.005 ? "level — balanced" : (torque > 0 ? "tips right: the mean is right of the pivot" : "tips left: the mean is left of the pivot");
    EV.label(g, iw, 14, lvl, { anchor: "end", color: Math.abs(torque) < 0.005 ? SC.good : SC.muted, size: 11 });
    // readout
    const direct = p.reduce((s, q, k) => s + q * (k - c) * (k - c), 0);
    const atMean = Math.abs(S.mean - Math.round(S.mean)) < 1e-9 ? p[Math.round(S.mean)] : 0;
    out.innerHTML = `E X = Σ x·p(x) = <b>${S.mean.toFixed(4)}</b> · Var X = <b>${S.var.toFixed(4)}</b> (σ = ${Math.sqrt(Math.max(0, S.var)).toFixed(4)}) · median ${S.median}` +
      `<br>torque about the pivot Σ (x − c)·p(x) = E X − c = <b>${torque.toFixed(4)}</b> — ${lvl}` +
      `<br>E(X − c)² = ${direct.toFixed(4)} = Var X + (E X − c)² = ${S.var.toFixed(4)} + ${(torque * torque).toFixed(4)}; smallest when c = E X` +
      `<br>probability sitting exactly at E X: ${atMean > 0 ? atMean.toFixed(4) : "<b>none</b> — the balance point need not be a possible value"}`;
  }
  sel.addEventListener("change", () => { load(sel.value); draw(); });
  cS.addEventListener("input", () => { c = +cS.value; $("cm-cv").textContent = c.toFixed(2); draw(); });
  $("cm-snap").addEventListener("click", () => { c = EV.pmfStats(p).mean; cS.value = c; $("cm-cv").textContent = c.toFixed(2); draw(); });
  $("cm-reset").addEventListener("click", () => { sel.value = "drill"; load("drill"); draw(); });
  load("drill"); draw();
})();

/* ─────────────────── 2 · St. Petersburg and Cauchy running means ───────────────────
   One seeded stream drives all three simulations, in a fixed order (2²⁰ games, then 10⁵ Cauchy,
   then 10⁵ normal). Caption audit at the default seed: see EV.stpeteSim notes in the build log;
   the claims are (i) the St. Petersburg mean at n = 2²⁰ lies near log₂ n = 20, (ii) the normal
   running mean ends within 0.01 of 0, (iii) the Cauchy running mean is still far from settled.  */
EV.stpeteSim = function (seed) {
  const r = ST.rng(seed), N1 = 1 << 20, N2 = 100000;
  const ck1 = [], ck2 = [];
  const grid = (N, K) => { const s = new Set(); for (let j = 0; j <= K; j++) s.add(Math.max(1, Math.round(Math.pow(N, j / K)))); return [...s].sort((a, b) => a - b); };
  const g1 = grid(N1, 900), g2 = grid(N2, 700);
  let sum = 0, best = 0, bestAt = 0, gi = 0;
  for (let i = 1; i <= N1; i++) {
    let u = r(); while (u === 0) u = r();
    const K = Math.max(1, Math.ceil(Math.log(u) / Math.log(0.5)));   // P(K = k) = 2⁻ᵏ
    const prize = Math.pow(2, K);
    sum += prize;
    if (prize > best) { best = prize; bestAt = i; }
    if (i === g1[gi]) { ck1.push([i, sum / i]); gi++; }
  }
  const sp = { mean: sum / N1, best: best, bestAt: bestAt, share: best / sum, n: N1 };
  let sc = 0, sn = 0; gi = 0;
  let cMax = 0;
  for (let i = 1; i <= N2; i++) {
    let u = r(); while (u === 0 || u === 0.5) u = r();
    const cv = Math.tan(Math.PI * (u - 0.5));
    sc += cv; sn += ST.randn(r);
    if (Math.abs(cv) > Math.abs(cMax)) cMax = cv;
    if (i === g2[gi]) { ck2.push([i, sc / i, sn / i]); gi++; }
  }
  return { ck1, ck2, sp, cauchyMean: sc / N2, normalMean: sn / N2, cMax: cMax, n2: N2 };
};
(function () {
  const svg = d3.select("#stpete-svg");
  if (svg.empty()) return;
  const W = 680, H = 330, $ = id => document.getElementById(id), out = $("stpete-readout");
  let seed = EV.seeds.stpete;
  function draw() {
    const S = EV.stpeteSim(seed);
    svg.selectAll("*").remove();
    const cells = ST.cells(W, H, 2, 1, { l: 44, r: 10, t: 26, b: 34 }, { x: 22 });
    // left: St. Petersburg
    const c1 = cells[0], g1 = svg.append("g").attr("transform", `translate(${c1.x + c1.m.l},${c1.y + c1.m.t})`);
    const x1 = d3.scaleLog().base(2).domain([1, S.sp.n]).range([0, c1.iw]);
    const ymax1 = Math.max(30, d3.max(S.ck1, d => d[1]) * 1.05);
    const y1 = d3.scaleLinear().domain([0, ymax1]).range([c1.ih, 0]);
    ST.gridY(g1, y1, c1.iw, 5);
    g1.append("g").attr("class", "axis").attr("transform", `translate(0,${c1.ih})`).call(d3.axisBottom(x1).tickValues([1, 16, 256, 4096, 65536, 1048576]).tickFormat(d => "2" + String(Math.round(Math.log2(d))).split("").map(c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+c]).join("")));
    g1.append("g").attr("class", "axis").call(d3.axisLeft(y1).ticks(5));
    EV.label(g1, 0, -10, "St. Petersburg: average prize of n games", { color: SC.ink });
    EV.label(g1, c1.iw, c1.ih + 30, "games n (log scale)", { anchor: "end" });
    g1.append("path").attr("d", d3.line().x(d => x1(d)).y(d => y1(Math.log2(d)))(d3.range(0, 21, 0.25).map(e => Math.pow(2, e))))
      .attr("fill", "none").attr("stroke", SC.muted).attr("stroke-dasharray", "5 4");
    EV.label(g1, x1(Math.pow(2, 19)), y1(19) + 16, "log₂ n", { color: SC.muted, anchor: "end" });
    g1.append("path").attr("d", d3.line().x(d => x1(d[0])).y(d => y1(Math.min(ymax1, d[1])))(S.ck1)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.8);
    // right: Cauchy vs normal
    const c2 = cells[1], g2 = svg.append("g").attr("transform", `translate(${c2.x + c2.m.l},${c2.y + c2.m.t})`);
    const x2 = d3.scaleLog().domain([1, S.n2]).range([0, c2.iw]);
    const lim = 3, y2 = d3.scaleLinear().domain([-lim, lim]).range([c2.ih, 0]);
    ST.gridY(g2, y2, c2.iw, 6);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${c2.ih})`).call(d3.axisBottom(x2).ticks(5, "~s"));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(6));
    EV.label(g2, 0, -10, "running mean: Cauchy (orange) vs normal (blue)", { color: SC.ink });
    EV.label(g2, c2.iw, c2.ih + 30, "draws n (log scale)", { anchor: "end" });
    g2.append("line").attr("x1", 0).attr("x2", c2.iw).attr("y1", y2(0)).attr("y2", y2(0)).attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
    const cl = v => ST.clamp(v, -lim, lim);
    g2.append("path").attr("d", d3.line().x(d => x2(d[0])).y(d => y2(cl(d[2])))(S.ck2)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.6);
    g2.append("path").attr("d", d3.line().x(d => x2(d[0])).y(d => y2(cl(d[1])))(S.ck2)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);
    const clipped = S.ck2.some(d => Math.abs(d[1]) > lim);
    if (clipped) EV.label(g2, c2.iw, 12, `values beyond ±${lim} drawn at the edge`, { anchor: "end", size: 10 });
    out.innerHTML = `seed ${seed} · St. Petersburg after n = 2²⁰ = 1,048,576 games: average prize <b>${S.sp.mean.toFixed(2)}</b> against log₂ n = 20; ` +
      `the largest single prize, 2^${Math.round(Math.log2(S.sp.best))} = ${d3.format(",")(S.sp.best)} in game ${d3.format(",")(S.sp.bestAt)}, is ${(100 * S.sp.share).toFixed(1)}% of all money paid` +
      `<br>after 100,000 draws: normal running mean <b>${S.normalMean.toFixed(4)}</b> (true mean 0, standard error 1/√n = 0.0032) · Cauchy running mean <b>${S.cauchyMean.toFixed(3)}</b> (there is no mean; its spread is that of one draw), largest |draw| ${d3.format(",.0f")(Math.abs(S.cMax))}`;
  }
  $("sp-sim").addEventListener("click", () => { seed = 1 + Math.floor(Math.random() * 1e9); draw(); });
  $("sp-reset").addEventListener("click", () => { seed = EV.seeds.stpete; draw(); });
  draw();
})();

/* ─────────────────── 3 · the moment generating function ───────────────────
   Caption audit: default exponential λ = 1, n = 1 → M(t) = 1/(1 − t) with asymptote t = 1,
   central differences M′(0) = 1.0000, M″(0) = 2.0000 vs E X = 1, E X² = 2. n = 5 → (1 − t)⁻⁵,
   tangent slope 5, "Gamma(5, 1)".                                                               */
EV.mgfFam = {
  bern: { pn: "p", lo: 0.05, hi: 0.95, st: 0.05, dv: 0.3,
    make: p => ({ M: t => 1 - p + p * Math.exp(t), m1: p, m2: p, tmax: Infinity, name: `Bernoulli(${p.toFixed(2)})`, sum: n => `Bin(${n}, ${p.toFixed(2)})` }) },
  pois: { pn: "μ", lo: 0.2, hi: 8, st: 0.1, dv: 2,
    make: mu => ({ M: t => Math.exp(mu * (Math.exp(t) - 1)), m1: mu, m2: mu + mu * mu, tmax: Infinity, name: `Poisson(${mu.toFixed(1)})`, sum: n => `Poisson(${(n * mu).toFixed(1)})` }) },
  geom: { pn: "p", lo: 0.1, hi: 0.9, st: 0.05, dv: 0.3,
    make: p => ({ M: t => (t < -Math.log(1 - p) ? p * Math.exp(t) / (1 - (1 - p) * Math.exp(t)) : Infinity), m1: 1 / p, m2: (2 - p) / (p * p), tmax: -Math.log(1 - p), name: `geometric(${p.toFixed(2)}), trials`, sum: n => `negative binomial(r = ${n}, p = ${p.toFixed(2)})` }) },
  unif: { pn: "b", lo: 0.5, hi: 4, st: 0.1, dv: 2,
    make: b => ({ M: t => (Math.abs(t * b) < 1e-8 ? 1 + t * b / 2 : Math.expm1(t * b) / (t * b)), m1: b / 2, m2: b * b / 3, tmax: Infinity, name: `uniform(0, ${b.toFixed(1)})`, sum: n => (n === 1 ? `uniform(0, ${b.toFixed(1)})` : "not uniform: a sum of uniforms is a smooth hump (triangular for n = 2)") }) },
  exp: { pn: "λ", lo: 0.2, hi: 3, st: 0.05, dv: 1,
    make: l => ({ M: t => (t < l ? l / (l - t) : Infinity), m1: 1 / l, m2: 2 / (l * l), tmax: l, name: `Exp(λ = ${l.toFixed(2)})`, sum: n => `Gamma(${n}, ${l.toFixed(2)})` }) },
  norm: { pn: "μ", lo: -1.5, hi: 1.5, st: 0.1, dv: 0.5,
    make: mu => ({ M: t => Math.exp(mu * t + t * t / 2), m1: mu, m2: 1 + mu * mu, tmax: Infinity, name: `N(${mu.toFixed(1)}, 1)`, sum: n => `N(${(n * mu).toFixed(1)}, ${n})` }) },
  cauchy: { pn: "—", lo: 0, hi: 1, st: 1, dv: 0, make: () => ({ cauchy: true, name: "Cauchy" }) }
};
(function () {
  const svg = d3.select("#mgf-svg");
  if (svg.empty()) return;
  const W = 680, H = 340, $ = id => document.getElementById(id), out = $("mgf-readout");
  const sel = $("mg-fam"), pS = $("mg-p"), nS = $("mg-n");
  const m = { l: 50, r: 20, t: 18, b: 34 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  function setup(def) {
    const F = EV.mgfFam[sel.value];
    $("mg-pn").textContent = F.pn;
    pS.min = F.lo; pS.max = F.hi; pS.step = F.st;
    if (def) pS.value = F.dv;
    pS.disabled = sel.value === "cauchy";
  }
  function draw() {
    const F = EV.mgfFam[sel.value], par = +pS.value, n = +nS.value, D = F.make(par);
    $("mg-pv").textContent = sel.value === "cauchy" ? "—" : par.toFixed(2); $("mg-nv").textContent = n;
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const tmin = -2, tmaxPlot = 2, ytop = 6;
    const x = d3.scaleLinear().domain([tmin, tmaxPlot]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, ytop]).range([ih, 0]);
    ST.gridY(g, y, iw, 6); ST.gridX(g, x, ih, 8);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));
    g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", ih).attr("stroke", SC.line);
    EV.label(g, iw, ih + 30, "t", { anchor: "end" });
    if (D.cauchy) {
      const phi = t => Math.exp(-n * Math.abs(t / n));    // mean of n Cauchy: φ(t/n)ⁿ = e^(−|t|)
      g.append("path").attr("d", d3.line().x(t => x(t)).y(t => y(phi(t)))(d3.range(tmin, tmaxPlot + 1e-9, 0.01))).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
      g.append("circle").attr("cx", x(0)).attr("cy", y(1)).attr("r", 5).attr("fill", SC.a2);
      EV.label(g, x(0) + 8, y(1) - 8, "M(0) = 1 is the only finite value", { color: SC.a2 });
      EV.label(g, x(0.6), y(0.75), "characteristic function φ(t) = e^(−|t|)", { color: SC.good });
      EV.label(g, iw / 2, 24, "Cauchy: E e^(tX) = ∞ for every t ≠ 0", { anchor: "middle", color: SC.bad, size: 13 });
      out.innerHTML = `<b>Cauchy</b>: E e^(tX) = ∫ e^(tx)/(π(1 + x²)) dx = ∞ for every t ≠ 0, so there is no mgf, no moments to read off, and no mgf proof of anything.` +
        `<br>The characteristic function E e^(itX) = e^(−|t|) always exists. The mean of n = ${n} independent draws has φ(t/n)ⁿ = e^(−|t|): it is standard Cauchy again, whatever n.`;
      return;
    }
    const Mn = t => Math.pow(D.M(t), n);
    const ES = n * D.m1, ES2 = n * (D.m2 - D.m1 * D.m1) + ES * ES;
    const tEnd = Math.min(tmaxPlot, D.tmax - 1e-6);
    const ts = d3.range(tmin, tEnd, (tEnd - tmin) / 600).concat([tEnd]);
    const pts = ts.map(t => [t, Mn(t)]).filter(d => isFinite(d[1]));
    g.append("path").attr("d", d3.line().defined(d => d[1] <= ytop).x(d => x(d[0])).y(d => y(Math.min(ytop, d[1])))(pts)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.4);
    if (isFinite(D.tmax) && D.tmax <= tmaxPlot) {
      g.append("line").attr("x1", x(D.tmax)).attr("x2", x(D.tmax)).attr("y1", 0).attr("y2", ih).attr("stroke", SC.bad).attr("stroke-dasharray", "2 4");
      EV.label(g, x(D.tmax) + 4, 14, `M = ∞ for t ≥ ${D.tmax.toFixed(3)}`, { color: SC.bad, size: 10.5 });
    }
    // tangent and Taylor parabola, clipped to the frame
    const tl = d3.range(tmin, tmaxPlot + 1e-9, 0.01);
    g.append("path").attr("d", d3.line().defined(t => { const v = 1 + ES * t; return v >= 0 && v <= ytop; }).x(t => x(t)).y(t => y(1 + ES * t))(tl)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);
    g.append("path").attr("d", d3.line().defined(t => { const v = 1 + ES * t + ES2 * t * t / 2; return v >= 0 && v <= ytop; }).x(t => x(t)).y(t => y(1 + ES * t + ES2 * t * t / 2))(tl)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "6 4");
    g.append("circle").attr("cx", x(0)).attr("cy", y(1)).attr("r", 4.5).attr("fill", SC.ink);
    ST.legend(g, [
      { label: n === 1 ? "M(t)" : `M(t)^${n}: sum of ${n} copies`, color: SC.accent, dash: null },
      { label: "tangent at 0, slope E X", color: SC.a2, dash: "1 0" },
      { label: "1 + E X·t + E X²·t²/2", color: SC.good, dash: "6 4" }
    ], 10, 10);
    const h = 1e-4, d1 = (Mn(h) - Mn(-h)) / (2 * h), d2 = (Mn(h) - 2 * Mn(0) + Mn(-h)) / (h * h);
    out.innerHTML = `<b>${D.name}</b>${n > 1 ? `, sum of n = ${n} independent copies → <b>${D.sum(n)}</b>` : ""}` +
      `<br>numerical M′(0) = <b>${d1.toFixed(4)}</b> vs E = ${ES.toFixed(4)} · numerical M″(0) = <b>${d2.toFixed(4)}</b> vs E(·²) = ${ES2.toFixed(4)} · variance M″(0) − M′(0)² = ${(d2 - d1 * d1).toFixed(4)} (exact ${(ES2 - ES * ES).toFixed(4)})` +
      `<br>${isFinite(D.tmax) ? `finite only for t &lt; ${D.tmax.toFixed(4)}` : "finite for every t"}; curve clipped at height ${ytop}.`;
  }
  sel.addEventListener("change", () => { setup(true); draw(); });
  pS.addEventListener("input", draw); nS.addEventListener("input", draw);
  $("mg-reset").addEventListener("click", () => { sel.value = "exp"; setup(true); nS.value = 1; draw(); });
  setup(true); draw();
})();

/* ─────────────────── 4 · Jensen's inequality ───────────────────
   Caption audit: default g = x², a = 1, b = 5, p = P(X = b) = ¼ → E X = 2, E g(X) = 7, g(E X) = 4,
   gap 3 = Var X = p(1 − p)(b − a)². With a = b every gap is 0.                                    */
EV.jensenG = {
  sq: { f: x => x * x, d2: x => 2, lo: -1, hi: 6, convex: true, name: "x²" },
  exp: { f: x => Math.exp(x), d2: x => Math.exp(x), lo: -2, hi: 3, convex: true, name: "eˣ" },
  inv: { f: x => 1 / x, d2: x => 2 / (x * x * x), lo: 0.25, hi: 6, convex: true, name: "1/x" },
  nlog: { f: x => -Math.log(x), d2: x => 1 / (x * x), lo: 0.15, hi: 6, convex: true, name: "−ln x" },
  log: { f: x => Math.log(x), d2: x => -1 / (x * x), lo: 0.15, hi: 6, convex: false, name: "ln x" },
  sqrt: { f: x => Math.sqrt(Math.max(0, x)), d2: x => -0.25 * Math.pow(x, -1.5), lo: 0, hi: 6, convex: false, name: "√x" }
};
(function () {
  const svg = d3.select("#jensen-svg");
  if (svg.empty()) return;
  const W = 680, H = 340, $ = id => document.getElementById(id), out = $("jensen-readout");
  const gSel = $("je-g"), pS = $("je-p");
  const m = { l: 52, r: 20, t: 16, b: 36 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  let A = 1, B = 5;
  function draw() {
    const G = EV.jensenG[gSel.value], p = +pS.value;
    $("je-pv").textContent = p.toFixed(2);
    A = ST.clamp(A, G.lo + 1e-6, G.hi); B = ST.clamp(B, G.lo + 1e-6, G.hi);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleLinear().domain([G.lo, G.hi]).range([0, iw]);
    const xs = d3.range(0, 401).map(i => G.lo + 1e-6 + (G.hi - G.lo - 1e-6) * i / 400);
    let ylo = d3.min(xs, G.f), yhi = d3.max(xs, G.f);
    if (gSel.value === "inv") yhi = 4.2;
    if (gSel.value === "exp") yhi = Math.exp(3);
    const pad = 0.06 * (yhi - ylo);
    const y = d3.scaleLinear().domain([ylo - pad, yhi + pad]).range([ih, 0]);
    ST.gridY(g, y, iw, 6);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));
    EV.label(g, 0, -4, `g(x) = ${G.name}  (${G.convex ? "convex" : "concave"})`, { color: SC.ink });
    g.append("path").attr("d", d3.line().defined(v => G.f(v) <= yhi + pad).x(v => x(v)).y(v => y(G.f(v)))(xs)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.4);
    const EX = (1 - p) * A + p * B, Eg = (1 - p) * G.f(A) + p * G.f(B), gE = G.f(EX);
    const yc = v => y(ST.clamp(v, ylo - pad, yhi + pad));
    // chord
    g.append("line").attr("x1", x(A)).attr("x2", x(B)).attr("y1", yc(G.f(A))).attr("y2", yc(G.f(B))).attr("stroke", SC.a2).attr("stroke-width", 1.8);
    // gap
    g.append("line").attr("x1", x(EX)).attr("x2", x(EX)).attr("y1", yc(Eg)).attr("y2", yc(gE)).attr("stroke", SC.bad).attr("stroke-width", 3);
    g.append("line").attr("x1", x(EX)).attr("x2", x(EX)).attr("y1", yc(Math.min(Eg, gE))).attr("y2", ih).attr("stroke", SC.muted).attr("stroke-dasharray", "3 3");
    g.append("circle").attr("cx", x(EX)).attr("cy", yc(Eg)).attr("r", 5.5).attr("fill", SC.a2);
    g.append("circle").attr("cx", x(EX)).attr("cy", yc(gE)).attr("r", 5.5).attr("fill", SC.accent).attr("stroke", SC.ink);
    EV.label(g, x(EX) + 9, yc(Eg) + (G.convex ? -6 : 14), `E g(X) = ${Eg.toFixed(3)}`, { color: SC.a2 });
    EV.label(g, x(EX) + 9, yc(gE) + (G.convex ? 14 : -6), `g(E X) = ${gE.toFixed(3)}`, { color: SC.accent });
    EV.label(g, x(EX), ih - 6, "E X", { anchor: "middle", color: SC.muted, size: 10 });
    // draggable a, b
    [["a", 1 - p], ["b", p]].forEach(([nm, w]) => {
      const v = nm === "a" ? A : B;
      const hg = g.append("g").style("cursor", "ew-resize");
      hg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", yc(G.f(v))).attr("y2", ih).attr("stroke", SC.a2).attr("stroke-dasharray", "2 3");
      hg.append("circle").attr("cx", x(v)).attr("cy", yc(G.f(v))).attr("r", 5 + 6 * Math.sqrt(w)).attr("fill", SC.a2).attr("opacity", 0.55);
      hg.append("circle").attr("cx", x(v)).attr("cy", ih).attr("r", 8).attr("fill", SC.a2);
      hg.append("text").attr("x", x(v)).attr("y", ih + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.bg).text(nm);
      hg.append("rect").attr("x", x(v) - 10).attr("y", 0).attr("width", 20).attr("height", ih + 12).attr("fill", "transparent");
      hg.call(d3.drag().on("drag", ev => { const nv = ST.clamp(x.invert(ev.x), G.lo + 1e-6, G.hi); if (nm === "a") A = nv; else B = nv; draw(); }));
    });
    const varX = p * (1 - p) * (B - A) * (B - A), gap = Eg - gE;
    let extra = "";
    if (gSel.value === "sq") extra = `for g = x² the gap is exactly Var X = p(1 − p)(b − a)² = ${varX.toFixed(4)}`;
    if (gSel.value === "log" || gSel.value === "nlog") {
      const gm = Math.exp((1 - p) * Math.log(A) + p * Math.log(B));
      extra = `arithmetic mean ${EX.toFixed(4)} ≥ geometric mean e^(E ln X) = ${gm.toFixed(4)}; |gap| = ln(AM/GM) = ${Math.log(EX / gm).toFixed(4)}`;
    }
    if (gSel.value === "inv") extra = `E(1/X) = ${Eg.toFixed(4)} ≥ 1/E X = ${gE.toFixed(4)}: the average of ratios exceeds the ratio of averages`;
    if (gSel.value === "exp") extra = `E e^X = ${Eg.toFixed(4)} ≥ e^(E X) = ${gE.toFixed(4)}: the lognormal's mean-above-median, in miniature`;
    if (gSel.value === "sqrt") extra = `E √X = ${Eg.toFixed(4)} ≤ √(E X) = ${gE.toFixed(4)}: why a sample standard deviation is biased low`;
    out.innerHTML = `X = a = ${A.toFixed(3)} with probability ${(1 - p).toFixed(2)}, b = ${B.toFixed(3)} with probability ${p.toFixed(2)} · E X = <b>${EX.toFixed(4)}</b>` +
      `<br>E g(X) = <b>${Eg.toFixed(4)}</b> · g(E X) = <b>${gE.toFixed(4)}</b> · gap E g(X) − g(E X) = <b>${gap.toFixed(4)}</b> ${G.convex ? "≥ 0 (convex)" : "≤ 0 (concave)"}` +
      ` · Taylor estimate ½·g″(E X)·Var X = ${(0.5 * G.d2(EX) * varX).toFixed(4)}` +
      `<br>${extra}`;
  }
  gSel.addEventListener("change", () => { A = 1; B = Math.min(5, EV.jensenG[gSel.value].hi); draw(); }); pS.addEventListener("input", draw);
  $("je-reset").addEventListener("click", () => { gSel.value = "sq"; pS.value = 0.25; A = 1; B = 5; draw(); });
  draw();
})();

/* ─────────────────── 5 · tail bounds against the truth ───────────────────
   Caption audit: default exponential mean 1, a = 4 → true 0.0183, Markov 0.250, Chebyshev 0.111,
   Cantelli 0.100, Chernoff 0.199 (Cantelli tightest). Regions on the a-grid: Markov (1, 2),
   Cantelli (2, 6.11), Chernoff beyond 6.11. Coin mode: Chernoff = e^(−n·KL(a ‖ ½)), Hoeffding
   e^(−2n(a − ½)²); exact tail from the binomial.                                                  */
EV.boundDist = function (key, n) {
  const KL = (q, p) => (q >= 1 ? -Math.log(p) : q * Math.log(q / p) + (1 - q) * Math.log((1 - q) / (1 - p)));
  if (key === "exp") return { lo: 0.2, hi: 10, def: 4, mu: 1, v: 1, nonneg: true, name: "exponential, mean 1",
    tail: a => (a <= 0 ? 1 : Math.exp(-a)), chern: a => (a > 1 ? a * Math.exp(1 - a) : 1) };
  if (key === "norm") return { lo: 0.05, hi: 6, def: 2, mu: 0, v: 1, nonneg: false, name: "standard normal",
    tail: a => 1 - ST.normCdf(a), chern: a => (a > 0 ? Math.exp(-a * a / 2) : 1) };
  if (key === "pois") return { lo: 10.2, hi: 32, def: 20, mu: 10, v: 10, nonneg: true, name: "Poisson(10)",
    tail: a => 1 - PV.poisCdf(Math.ceil(a) - 1, 10), chern: a => (a > 10 ? Math.exp(-10 + a * Math.log(10 * Math.E / a)) : 1) };
  // mean of n fair coins
  return { lo: 0.505, hi: 1, def: 0.7, mu: 0.5, v: 0.25 / n, nonneg: true, coin: true, name: `mean of ${n} fair coins`,
    tail: a => { const k0 = Math.ceil(n * a - 1e-9); let s = 0; for (let k = k0; k <= n; k++) s += ST.binomPmf(k, n, 0.5); return s; },
    chern: a => (a > 0.5 ? Math.exp(-n * KL(a, 0.5)) : 1),
    hoeff: a => (a > 0.5 ? Math.exp(-2 * n * (a - 0.5) * (a - 0.5)) : 1) };
};
EV.boundsAt = function (D, a) {
  const d = a - D.mu, o = {};
  o.markov = D.nonneg && a > 0 ? D.mu / a : NaN;
  o.cheb = d > 0 ? D.v / (d * d) : NaN;
  o.cant = d > 0 ? D.v / (D.v + d * d) : NaN;
  o.chern = D.chern(a);
  if (D.coin) o.hoeff = D.hoeff(a);
  return o;
};
(function () {
  const svg = d3.select("#bounds-svg");
  if (svg.empty()) return;
  const W = 680, H = 350, $ = id => document.getElementById(id), out = $("bounds-readout");
  const sel = $("bd-dist"), aS = $("bd-a"), nS = $("bd-n");
  const m = { l: 56, r: 20, t: 16, b: 36 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const COL = { markov: SC.violet, cheb: SC.teal, cant: SC.good, chern: SC.a2, hoeff: SC.bad };
  const NAME = { markov: "Markov", cheb: "Chebyshev", cant: "Cantelli", chern: "Chernoff", hoeff: "Hoeffding" };
  let a = 4;
  const D = () => EV.boundDist(sel.value, +nS.value);
  const toS = (Dd, v) => Math.round(1000 * (v - Dd.lo) / (Dd.hi - Dd.lo)), fromS = (Dd, s) => Dd.lo + (Dd.hi - Dd.lo) * s / 1000;
  function draw() {
    const Dd = D();
    $("bd-nw").style.display = Dd.coin ? "" : "none";
    $("bd-nv").textContent = nS.value; $("bd-av").textContent = a.toFixed(Dd.coin ? 3 : 2);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleLinear().domain([Dd.lo, Dd.hi]).range([0, iw]);
    const xs = d3.range(0, 801).map(i => Dd.lo + (Dd.hi - Dd.lo) * i / 800);
    const keys = ["markov", "cheb", "cant", "chern"].concat(Dd.coin ? ["hoeff"] : []);
    const tails = xs.map(v => Dd.tail(v));
    const floor = Math.max(1e-16, Math.min(1e-3, d3.min(tails.filter(t => t > 0)) / 3));
    const y = d3.scaleLog().domain([floor, 1.5]).range([ih, 0]).clamp(true);
    ST.gridY(g, y, iw, 6);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6, "~e"));
    EV.label(g, 0, -4, `P(X ≥ a) and its bounds, log scale — ${Dd.name}`, { color: SC.ink });
    EV.label(g, iw, ih + 31, "threshold a", { anchor: "end" });
    const line = d3.line().defined(d => isFinite(d[1]) && d[1] > 0 && d[1] < 0.999).x(d => x(d[0])).y(d => y(Math.max(floor, d[1])));
    keys.forEach(k => {
      const pts = xs.map(v => [v, EV.boundsAt(Dd, v)[k]]);
      g.append("path").attr("d", line(pts)).attr("fill", "none").attr("stroke", COL[k]).attr("stroke-width", 1.8);
    });
    g.append("path").attr("d", line(xs.map((v, i) => [v, tails[i]]))).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 2.6);
    g.append("line").attr("x1", x(a)).attr("x2", x(a)).attr("y1", 0).attr("y2", ih).attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
    ST.legend(g, [{ label: "true P(X ≥ a)", color: SC.ink, dash: "1 0" }].concat(keys.map(k => ({ label: NAME[k] + (k === "markov" && !Dd.nonneg ? " (n/a)" : ""), color: COL[k], dash: "1 0" }))), iw - 150, 12);
    // values at a and winner regions
    const B = EV.boundsAt(Dd, a), tv = Dd.tail(a);
    let best = null;
    keys.forEach(k => { const v = B[k]; if (isFinite(v) && v < 1 && (best === null || v < B[best])) best = k; });
    const regions = [];
    const winner = v => { const bb = EV.boundsAt(Dd, v); let w = null; keys.forEach(k => { const q = bb[k]; if (isFinite(q) && q < 1 && (w === null || q < bb[w] - 1e-12)) w = k; }); return w; };
    let cur = null, start = null, prev = Dd.lo;
    xs.forEach(v => {
      const w = winner(v);
      if (w !== cur) {
        let lo = prev, hi = v;                         // refine the change point by bisection
        for (let i = 0; i < 50 && v > Dd.lo; i++) { const md = (lo + hi) / 2; if (winner(md) === cur) lo = md; else hi = md; }
        const at = v > Dd.lo ? hi : v;
        if (cur) regions.push([cur, start, at]);
        cur = w; start = at;
      }
      prev = v;
    });
    if (cur) regions.push([cur, start, Dd.hi]);
    const f = v => (!isFinite(v) ? "n/a" : v >= 1 ? "≥ 1 (useless)" : v < 1e-3 ? v.toExponential(2) : v.toFixed(4));
    out.innerHTML = `a = ${a.toFixed(Dd.coin ? 3 : 2)} · <b>true ${f(tv)}</b> · ` + keys.map(k => `<span style="color:${COL[k]}">${NAME[k]} ${f(B[k])}</span>`).join(" · ") +
      `<br>tightest at this a: <b>${best ? NAME[best] : "none is below 1"}</b>${best ? ` (${(B[best] / Math.max(tv, 1e-300)).toPrecision(3)} × the truth)` : ""}` +
      `<br>tightest of the bounds, by range of a: ` + regions.filter(r => r[2] - r[1] > 1e-3 * (Dd.hi - Dd.lo)).map(r => `${NAME[r[0]]} ${r[1].toFixed(Dd.coin ? 3 : 2)}–${r[2].toFixed(Dd.coin ? 3 : 2)}`).join(" · ");
  }
  function reset(full) { const Dd = D(); a = Dd.def; aS.value = toS(Dd, a); if (full) draw(); }
  aS.addEventListener("input", () => { a = fromS(D(), +aS.value); draw(); });
  sel.addEventListener("change", () => reset(true));
  nS.addEventListener("input", draw);
  $("bd-reset").addEventListener("click", () => { sel.value = "exp"; nS.value = 100; reset(true); });
  reset(true);
})();

/* ─────────────────── 6 · concentration: one average, and the worst of k ───────────────────
   Caption audit, "one average" default (Bernoulli 0.3, n = 100, δ = 0.05): half-widths Hoeffding
   0.1358, Chebyshev 0.2049, normal 0.0898; exact P(|X̄ − μ| > ε) = 0.0031, 0.0000094, 0.0628.
   "k hypotheses" default (n = 10,000, k = 1,000, δ = 0.05, seed concK): ε = 0.0230, single-h ε =
   0.0136; at the default seed no test set of the 200 has its worst deviation above ε.            */
EV.concN = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000];
EV.concPop = {
  bern: { mu: 0.3, v: 0.21, name: "Bernoulli(0.3)", bern: 0.3 },
  half: { mu: 0.5, v: 0.25, name: "Bernoulli(0.5)", bern: 0.5 },
  unif: { mu: 0.5, v: 1 / 12, name: "uniform on [0, 1]" }
};
EV.concOne = function (popKey, n, seed) {
  const P = EV.concPop[popKey], r = ST.rng(seed);
  const M = P.bern ? 4000 : Math.min(4000, Math.floor(2e7 / n));
  const dev = new Float64Array(M);
  if (P.bern) {
    const tab = EV.binomTable(n, P.bern);
    for (let i = 0; i < M; i++) dev[i] = EV.binomFromTable(tab, r) / n - P.mu;
  } else {
    for (let i = 0; i < M; i++) { let s = 0; for (let j = 0; j < n; j++) s += r(); dev[i] = s / n - P.mu; }
  }
  return { dev, M };
};
EV.concK = function (n, k, seed, reps) {
  const R = reps || 200, L = 20, r = ST.rng(seed);
  const lev = d3.range(L).map(l => 0.10 + 0.30 * l / (L - 1));
  const tabs = lev.map(p => EV.binomTable(n, p));
  const maxDev = new Float64Array(R), pickTrue = new Float64Array(R), pickEmp = new Float64Array(R);
  for (let t = 0; t < R; t++) {
    let md = 0, best = Infinity, bestTrue = 0;
    for (let j = 0; j < k; j++) {
      const l = j % L, e = EV.binomFromTable(tabs[l], r) / n, d = Math.abs(e - lev[l]);
      if (d > md) md = d;
      if (e < best) { best = e; bestTrue = lev[l]; }
    }
    maxDev[t] = md; pickTrue[t] = bestTrue; pickEmp[t] = best;
  }
  return { maxDev, pickTrue, pickEmp, R };
};
(function () {
  const svg = d3.select("#conc-svg");
  if (svg.empty()) return;
  const W = 680, H = 340, $ = id => document.getElementById(id), out = $("conc-readout");
  const mS = $("cc-mode"), popS = $("cc-pop"), nS = $("cc-n"), kS = $("cc-k"), dS = $("cc-d");
  const m = { l: 52, r: 18, t: 18, b: 36 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  let seed1 = EV.seeds.conc, seedK = EV.seeds.concK, cache = {};
  const key = () => [mS.value, popS.value, nS.value, kS.value, seed1, seedK].join("|");
  function vline(g, x, v, col, dash, lab, yy) {
    g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", ih).attr("stroke", col).attr("stroke-width", 1.8).attr("stroke-dasharray", dash || null);
    if (lab) EV.label(g, x(v) + 3, yy, lab, { color: col, size: 10 });
  }
  function draw() {
    const mode = mS.value, n = EV.concN[+nS.value], k = Math.round(Math.pow(10, +kS.value)), del = +dS.value;
    $("cc-nv").textContent = d3.format(",")(n); $("cc-kv").textContent = d3.format(",")(k);
    $("cc-kw").style.display = mode === "k" ? "" : "none"; $("cc-popw").style.display = mode === "one" ? "" : "none";
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const z = ST.normQuant(1 - del / 2);
    if (mode === "one") {
      const P = EV.concPop[popS.value], sd = Math.sqrt(P.v);
      const eH = Math.sqrt(Math.log(2 / del) / (2 * n)), eC = sd / Math.sqrt(n * del), eN = z * sd / Math.sqrt(n);
      const kk = key(); if (!cache[kk]) cache = { [kk]: EV.concOne(popS.value, n, seed1) };
      const S = cache[kk];
      const Rg = Math.min(1, 1.15 * Math.max(eH, eN, Math.min(eC, 1.6 * eH)));
      const x = d3.scaleLinear().domain([-Rg, Rg]).range([0, iw]);
      const lattice = P.bern && n <= 120;
      let bars = [], exact = [];
      if (lattice) {
        const cnt = new Map(); S.dev.forEach(d => { const kq = Math.round((d + P.mu) * n); cnt.set(kq, (cnt.get(kq) || 0) + 1); });
        for (let kq = 0; kq <= n; kq++) { const dv = kq / n - P.mu; if (Math.abs(dv) <= Rg) { bars.push({ x0: dv - 0.4 / n, x1: dv + 0.4 / n, f: (cnt.get(kq) || 0) / S.M }); exact.push({ x: dv, f: ST.binomPmf(kq, n, P.bern) }); } }
      } else {
        const K = 70, w = 2 * Rg / K, c = new Array(K).fill(0);
        S.dev.forEach(d => { const i = Math.floor((d + Rg) / w); if (i >= 0 && i < K) c[i]++; });
        bars = c.map((v, i) => ({ x0: -Rg + i * w, x1: -Rg + (i + 1) * w, f: v / S.M }));
        exact = bars.map(b => {
          let pr;
          if (P.bern) { pr = 0; for (let kq = Math.ceil((b.x0 + P.mu) * n - 1e-9); kq < (b.x1 + P.mu) * n - 1e-9; kq++) pr += ST.binomPmf(kq, n, P.bern); }
          else pr = ST.normCdf(b.x1 / (sd / Math.sqrt(n))) - ST.normCdf(b.x0 / (sd / Math.sqrt(n)));
          return { x: (b.x0 + b.x1) / 2, f: pr };
        });
      }
      const ymax = 1.15 * Math.max(d3.max(bars, b => b.f), d3.max(exact, e => e.f));
      const y = d3.scaleLinear().domain([0, ymax]).range([ih, 0]);
      ST.gridY(g, y, iw, 5);
      g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(8));
      g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
      EV.label(g, 0, -5, `sample mean − μ over ${d3.format(",")(S.M)} simulated samples of n = ${d3.format(",")(n)} (${P.name})`, { color: SC.ink });
      EV.label(g, iw, ih + 31, "X̄ − μ", { anchor: "end" });
      g.selectAll("rect.b").data(bars).join("rect").attr("x", b => x(b.x0)).attr("width", b => Math.max(1, x(b.x1) - x(b.x0) - 0.5)).attr("y", b => y(b.f)).attr("height", b => ih - y(b.f)).attr("fill", SC.accent).attr("opacity", 0.6);
      g.selectAll("circle.e").data(exact).join("circle").attr("cx", e => x(e.x)).attr("cy", e => y(e.f)).attr("r", lattice ? 2.6 : 1.8).attr("fill", P.bern ? SC.ink : SC.muted);
      [[eH, SC.bad, null, "Hoeffding"], [eC, SC.teal, "6 3", "Chebyshev"], [eN, SC.good, "3 3", "normal"]].forEach(([e, col, da, lab], i) => {
        if (e <= Rg) { vline(g, x, e, col, da, lab, 14 + 13 * i); vline(g, x, -e, col, da, null, 0); }
        else EV.label(g, iw, 14 + 13 * i, `${lab} ±${e.toFixed(3)} is off the chart`, { anchor: "end", color: col, size: 10 });
      });
      const frac = e => S.dev.reduce((s, d) => s + (Math.abs(d) > e + 1e-12 ? 1 : 0), 0) / S.M;
      const ex = e => { if (!P.bern) return null; let t = 0; for (let kq = 0; kq <= n; kq++) if (Math.abs(kq / n - P.mu) > e + 1e-12) t += ST.binomPmf(kq, n, P.bern); return t; };
      const fx = v => (v === null ? "—" : v < 1e-4 ? v.toExponential(1) : v.toFixed(4));
      out.innerHTML = `seed ${seed1} · ${d3.format(",")(S.M)} samples · μ = ${P.mu}, σ = ${sd.toFixed(4)}, δ = ${del}` +
        `<br><span style="color:${SC.bad}">Hoeffding ±${eH.toFixed(4)}</span> (guarantee for any [0, 1] losses): simulated outside ${frac(eH).toFixed(4)}, exact ${fx(ex(eH))}` +
        `<br><span style="color:${SC.teal}">Chebyshev ±${eC.toFixed(4)}</span> (guarantee given σ): simulated ${frac(eC).toFixed(4)}, exact ${fx(ex(eC))}` +
        ` · <span style="color:${SC.good}">normal ±${eN.toFixed(4)}</span> (approximation, not a guarantee): simulated ${frac(eN).toFixed(4)}, exact ${fx(ex(eN))}` +
        `<br>minibatch reading: a batch of ${d3.format(",")(n)} per-example losses in [0, 1] has its average within ±${eH.toFixed(4)} of the full-data average with probability ≥ ${1 - del}; quadrupling the batch halves every half-width.` +
        (P.bern ? "" : " (dots: normal approximation per bin; no exact value is computed for uniform sums)");
    } else {
      const eU = Math.sqrt(Math.log(2 * k / del) / (2 * n)), eS = Math.sqrt(Math.log(2 / del) / (2 * n));
      const kk = key(); if (!cache[kk]) cache = { [kk]: EV.concK(n, k, seedK) };
      const S = cache[kk];
      const top = 1.3 * Math.max(eU, d3.max(S.maxDev));
      const x = d3.scaleLinear().domain([0, top]).range([0, iw]);
      const K = 40, w = top / K, c = new Array(K).fill(0);
      S.maxDev.forEach(d => { const i = Math.min(K - 1, Math.floor(d / w)); c[i]++; });
      const y = d3.scaleLinear().domain([0, 1.15 * d3.max(c) / S.R]).range([ih, 0]);
      ST.gridY(g, y, iw, 5);
      g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(8));
      g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
      EV.label(g, 0, -5, `worst deviation max |R̂ − R| over k = ${d3.format(",")(k)} hypotheses, ${S.R} test sets of n = ${d3.format(",")(n)}`, { color: SC.ink });
      EV.label(g, iw, ih + 31, "max over hypotheses of |empirical − true error|", { anchor: "end" });
      g.selectAll("rect").data(c).join("rect").attr("x", (v, i) => x(i * w)).attr("width", Math.max(1, x(w) - x(0) - 1)).attr("y", v => y(v / S.R)).attr("height", v => ih - y(v / S.R)).attr("fill", SC.accent).attr("opacity", 0.65);
      vline(g, x, eU, SC.bad, null, `union bound ε = ${eU.toFixed(4)}`, 14);
      vline(g, x, eS, SC.good, "4 3", `one hypothesis ${eS.toFixed(4)}`, 28);
      const over = e => S.maxDev.reduce((s, d) => s + (d > e ? 1 : 0), 0) / S.R;
      const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
      const opt = mean(Array.from(S.pickTrue).map((t, i) => t - S.pickEmp[i]));
      out.innerHTML = `seed ${seedK} · ${S.R} test sets · true errors spread over 20 levels from 0.10 to 0.40 (each hypothesis's errors simulated independently)` +
        `<br><span style="color:${SC.bad}">ε = √(ln(2k/δ)/(2n)) = √(ln(${d3.format(",")(2 * k)}/${del})/${d3.format(",")(2 * n)}) = <b>${eU.toFixed(4)}</b></span>: test sets whose worst deviation exceeds it ${over(eU).toFixed(3)} (guaranteed ≤ ${del})` +
        `<br><span style="color:${SC.good}">single-hypothesis ε = ${eS.toFixed(4)}</span> exceeded by the worst of ${d3.format(",")(k)} in ${(100 * over(eS)).toFixed(1)}% of test sets — why the union bound is needed · mean worst deviation ${mean(S.maxDev).toFixed(4)}` +
        `<br>ERM picks the lowest empirical error: its true error averages ${mean(S.pickTrue).toFixed(4)} (best in class 0.10); its empirical error flatters it by ${opt.toFixed(4)} on average (bound: ε)`;
    }
  }
  [mS, popS, dS].forEach(el => el.addEventListener("change", () => {
    if (el === mS) { nS.value = mS.value === "k" ? 12 : 4; kS.value = 3; }
    draw();
  }));
  [nS, kS].forEach(el => el.addEventListener("input", draw));
  $("cc-sim").addEventListener("click", () => { if (mS.value === "k") seedK = 1 + Math.floor(Math.random() * 1e9); else seed1 = 1 + Math.floor(Math.random() * 1e9); draw(); });
  $("cc-reset").addEventListener("click", () => { mS.value = "one"; popS.value = "bern"; nS.value = 4; kS.value = 3; dS.value = "0.05"; seed1 = EV.seeds.conc; seedK = EV.seeds.concK; draw(); });
  draw();
})();

/* ─────────────────── 7 · E[Y | X] as the best predictor ───────────────────
   Caption audit: default x² on U(−1, 2), normal noise σ = 0.5, squared loss, 400 points, seed pred →
   population MSE 0.2500 (curve), 0.7000 (best line 0.5 + x), 1.4500 (constant 1); Var Y = 0.25 + 1.20.
   Skewed noise + absolute loss: population MAE σ·ln 2 (median curve) vs σ·2/e (mean curve).        */
EV.predShape = {
  quad: { lo: -1, hi: 2, m: x => x * x, name: "x²" },
  sym: { lo: -2, hi: 2, m: x => x * x, name: "x²" },
  lin: { lo: -1, hi: 2, m: x => 1 + 0.8 * x, name: "1 + 0.8x" },
  sin: { lo: -1, hi: 2, m: x => Math.sin(2 * x), name: "sin(2x)" }
};
EV.predPop = function (S) {                 // population moments under X ~ U(lo, hi), by Simpson
  const w = S.hi - S.lo, E = f => PV.simpson(x => f(x) / w, S.lo, S.hi, 2000);
  const EX = E(x => x), VX = E(x => x * x) - EX * EX, Em = E(S.m), Vm = E(x => S.m(x) ** 2) - Em * Em;
  const b = (E(x => x * S.m(x)) - EX * Em) / VX, a = Em - b * EX;
  return { EX, VX, Em, Vm, a, b, lineGap: Vm - b * b * VX };
};
EV.predSample = function (shape, noise, s, n, seed) {
  const S = EV.predShape[shape], r = ST.rng(seed), pts = [];
  for (let i = 0; i < n; i++) {
    const x = S.lo + (S.hi - S.lo) * r();
    const e = noise === "skew" ? s * (PV.expDraw(1, r) - 1) : s * ST.randn(r);
    pts.push([x, S.m(x) + e]);
  }
  return pts;
};
(function () {
  const svg = d3.select("#pred-svg");
  if (svg.empty()) return;
  const W = 680, H = 340, $ = id => document.getElementById(id), out = $("pred-readout");
  const shS = $("pr-shape"), nzS = $("pr-noise"), sS = $("pr-s"), lS = $("pr-loss");
  const m = { l: 46, r: 18, t: 16, b: 34 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  let seed = EV.seeds.pred;
  function draw() {
    const S = EV.predShape[shS.value], s = +sS.value, skew = nzS.value === "skew", abs = lS.value === "abs";
    $("pr-sv").textContent = s.toFixed(2);
    const P = EV.predPop(S), pts = EV.predSample(shS.value, nzS.value, s, 400, seed);
    const medOff = skew ? s * (Math.LN2 - 1) : 0;
    const fMean = x => S.m(x), fMed = x => S.m(x) + medOff, fLine = x => P.a + P.b * x, cMean = P.Em;
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const ys = pts.map(p => p[1]);
    const x = d3.scaleLinear().domain([S.lo, S.hi]).range([0, iw]);
    const y = d3.scaleLinear().domain([Math.min(d3.min(ys), -0.5), d3.max(ys)]).nice().range([ih, 0]);
    ST.gridY(g, y, iw, 6);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(7));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));
    EV.label(g, iw, ih + 30, "x", { anchor: "end" }); EV.label(g, 4, 10, "y", {});
    g.selectAll("circle").data(pts).join("circle").attr("cx", d => x(d[0])).attr("cy", d => y(d[1])).attr("r", 2.4).attr("fill", SC.muted).attr("opacity", 0.55);
    const xs = d3.range(0, 301).map(i => S.lo + (S.hi - S.lo) * i / 300);
    const path = (f, col, w, da) => g.append("path").attr("d", d3.line().x(v => x(v)).y(v => y(f(v)))(xs)).attr("fill", "none").attr("stroke", col).attr("stroke-width", w).attr("stroke-dasharray", da || null);
    path(() => cMean, SC.muted, 1.6, "2 3");
    path(fLine, SC.accent, 2, "7 4");
    path(fMean, SC.a2, 2.6);
    if (skew || abs) path(fMed, SC.good, 2.2, skew ? null : "2 2");
    ST.legend(g, [{ label: "E[Y | X] (conditional mean)", color: SC.a2, dash: "1 0" }]
      .concat(skew || abs ? [{ label: "conditional median", color: SC.good, dash: "1 0" }] : [])
      .concat([{ label: `best line ${P.a.toFixed(2)} + ${P.b.toFixed(2)}x`, color: SC.accent, dash: "7 4" }, { label: `best constant E Y = ${cMean.toFixed(2)}`, color: SC.muted, dash: "2 3" }]), iw - 200, 12);
    const mse = f => d3.mean(pts, p => (p[1] - f(p[0])) ** 2), mae = f => d3.mean(pts, p => Math.abs(p[1] - f(p[0])));
    const VarY = s * s + P.Vm;
    if (!abs) {
      out.innerHTML = `<b>squared loss</b>, Y = ${S.name} + ${skew ? "skewed" : "normal"} noise with σ = ${s.toFixed(2)} on (${S.lo}, ${S.hi}) · seed ${seed}, 400 points` +
        `<br>population MSE: <span style="color:${SC.a2}">E[Y | X] <b>${(s * s).toFixed(4)}</b></span> · <span style="color:${SC.accent}">best line <b>${(s * s + P.lineGap).toFixed(4)}</b></span> · constant <b>${VarY.toFixed(4)}</b>` +
        `<br>on this sample: ${mse(fMean).toFixed(4)} · ${mse(fLine).toFixed(4)} · ${mse(() => cMean).toFixed(4)}` +
        `<br>Var Y = E Var(Y | X) + Var E[Y | X] = ${(s * s).toFixed(4)} + ${P.Vm.toFixed(4)} = ${VarY.toFixed(4)} · share explained by E[Y | X]: η² = ${(P.Vm / VarY).toFixed(3)}; by the best line: R² = ${(P.b * P.b * P.VX / VarY).toFixed(3)}`;
    } else {
      const pm = skew ? s * Math.LN2 : s * Math.sqrt(2 / Math.PI), pmean = skew ? s * 2 / Math.E : pm;
      const medY = ST.median(ys);
      out.innerHTML = `<b>absolute loss</b>, ${skew ? "skewed" : "normal"} noise σ = ${s.toFixed(2)} · seed ${seed}` +
        `<br>population MAE: <span style="color:${SC.good}">conditional median <b>${pm.toFixed(4)}</b></span> · <span style="color:${SC.a2}">conditional mean <b>${pmean.toFixed(4)}</b></span>${skew ? " — the median wins under absolute loss" : " — identical: symmetric noise makes mean = median"}` +
        `<br>on this sample: median curve ${mae(fMed).toFixed(4)} · mean curve ${mae(fMean).toFixed(4)} · best (least-squares) line ${mae(fLine).toFixed(4)} · constant median(Y) ${mae(() => medY).toFixed(4)}` +
        (skew ? `<br>under squared loss the order flips: mean curve ${mse(fMean).toFixed(4)} vs median curve ${mse(fMed).toFixed(4)} on this sample (population ${(s * s).toFixed(4)} vs ${(s * s * (1 + (1 - Math.LN2) ** 2)).toFixed(4)})` : "");
    }
  }
  [shS, nzS, lS].forEach(el => el.addEventListener("change", draw));
  sS.addEventListener("input", draw);
  $("pr-sim").addEventListener("click", () => { seed = 1 + Math.floor(Math.random() * 1e9); draw(); });
  $("pr-reset").addEventListener("click", () => { shS.value = "quad"; nzS.value = "norm"; sS.value = 0.5; lS.value = "sq"; seed = EV.seeds.pred; draw(); });
  draw();
})();

/* ─────────────────── 8 · the law of total variance ───────────────────
   Caption audit: default gamma–Poisson μ = 4, r = 2, 5,000 units at seed totvar → exact within 4,
   between 8, total 12; simulated values printed in the readout and checked by the build script.
   Groups mode at multipliers 1, 1 reproduces SS_within 24, SS_between 54, SS_total 78.
   Hierarchical binomial n = 10: exact 1.667 + 8.333 = 10.                                          */
EV.tvGroups = [[2, 4, 6], [5, 7, 9], [8, 10, 12]];
EV.tvSim = function (mode, p1, p2, seed) {
  const r = ST.rng(seed), U = 5000, Y = new Float64Array(U), lam = new Float64Array(U), within = new Float64Array(U);
  for (let i = 0; i < U; i++) {
    if (mode === "nb") { const L = PV.gammaDraw(p2, p2 / p1, r); lam[i] = L; within[i] = L; Y[i] = PV.poisDraw(L, r); }
    else { const n = Math.round(p1), q = r(); lam[i] = n * q; within[i] = n * q * (1 - q); Y[i] = ST.binom(n, q, r); }
  }
  const mean = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
  const vr = a => { const mu = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - mu) ** 2; return s / (a.length - 1); };
  return { Y, within: mean(within), between: vr(lam), total: vr(Y), meanY: mean(Y), U };
};
(function () {
  const svg = d3.select("#totvar-svg");
  if (svg.empty()) return;
  const W = 680, H = 340, $ = id => document.getElementById(id), out = $("totvar-readout");
  const mS = $("tv-mode"), p1S = $("tv-p1"), p2S = $("tv-p2");
  let seed = EV.seeds.totvar, cache = {};
  const CFG = {
    groups: { p1: ["between-group spread ×", 0, 3, 0.1, 1], p2: ["within-group spread ×", 0, 3, 0.1, 1] },
    nb: { p1: ["mean μ", 0.5, 12, 0.5, 4], p2: ["dispersion r", 0.5, 50, 0.5, 2] },
    bb: { p1: ["people per region n", 1, 40, 1, 10], p2: null }
  };
  function setup(def) {
    const C = CFG[mS.value];
    [["tv-p1", C.p1], ["tv-p2", C.p2]].forEach(([id, c]) => {
      const el = $(id), wrap = el.parentNode;
      if (!c) { wrap.style.display = "none"; return; }
      wrap.style.display = ""; $(id + "n").textContent = c[0];
      el.min = c[1]; el.max = c[2]; el.step = c[3]; if (def) el.value = c[4];
    });
  }
  function bars(g, x0, w, ih, y, items) {     // items: [{label, parts:[{v, col}], total}]
    items.forEach((it, i) => {
      const xx = x0 + i * (w + 16);
      let acc = 0;
      it.parts.forEach(pt => {
        g.append("rect").attr("x", xx).attr("width", w).attr("y", y(acc + pt.v)).attr("height", Math.max(0, y(acc) - y(acc + pt.v))).attr("fill", pt.col).attr("opacity", 0.85);
        acc += pt.v;
      });
      EV.label(g, xx + w / 2, ih + 14, it.label, { anchor: "middle", size: 10 });
      EV.label(g, xx + w / 2, y(acc) - 4, acc.toFixed(2), { anchor: "middle", size: 10, color: SC.ink });
    });
  }
  function draw() {
    const mode = mS.value, p1 = +p1S.value, p2 = +p2S.value;
    $("tv-p1v").textContent = p1.toFixed(mode === "bb" ? 0 : 1); $("tv-p2v").textContent = p2.toFixed(1);
    svg.selectAll("*").remove();
    const cells = ST.cells(W, H, 2, 1, { l: 44, r: 10, t: 24, b: 34 }, { x: 26 });
    const c1 = cells[0], c2 = cells[1];
    const gL = svg.append("g").attr("transform", `translate(${c1.x + c1.m.l},${c1.y + c1.m.t})`);
    const gR = svg.append("g").attr("transform", `translate(${c2.x + c2.m.l},${c2.y + c2.m.t})`);
    let items, txt;
    if (mode === "groups") {
      const all = EV.tvGroups.flat(), gm = d3.mean(all), N = all.length;
      const data = EV.tvGroups.map(gr => { const mj = d3.mean(gr); return gr.map(v => gm + p1 * (mj - gm) + p2 * (v - mj)); });
      let sw = 0, sb = 0, st = 0;
      data.forEach(gr => { const mj = d3.mean(gr); gr.forEach(v => { sw += (v - mj) ** 2; st += (v - gm) ** 2; }); sb += gr.length * (mj - gm) ** 2; });
      const flat = data.flat(), y = d3.scaleLinear().domain([Math.min(0, d3.min(flat)) - 1, Math.max(14, d3.max(flat)) + 1]).range([c1.ih, 0]);
      const x = d3.scaleLinear().domain([0.4, 3.6]).range([0, c1.iw]);
      ST.gridY(gL, y, c1.iw, 6);
      gL.append("g").attr("class", "axis").attr("transform", `translate(0,${c1.ih})`).call(d3.axisBottom(x).tickValues([1, 2, 3]).tickFormat(d => "group " + d));
      gL.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));
      EV.label(gL, 0, -8, "three groups: dots, group means, grand mean", { color: SC.ink });
      gL.append("line").attr("x1", 0).attr("x2", c1.iw).attr("y1", y(gm)).attr("y2", y(gm)).attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
      data.forEach((gr, j) => {
        const mj = d3.mean(gr);
        gL.append("line").attr("x1", x(j + 1) - 26).attr("x2", x(j + 1) + 26).attr("y1", y(mj)).attr("y2", y(mj)).attr("stroke", SC.a2).attr("stroke-width", 2.5);
        gr.forEach((v, i) => {
          gL.append("line").attr("x1", x(j + 1) + (i - 1) * 12).attr("x2", x(j + 1) + (i - 1) * 12).attr("y1", y(v)).attr("y2", y(mj)).attr("stroke", SC.accent).attr("opacity", 0.6);
          gL.append("circle").attr("cx", x(j + 1) + (i - 1) * 12).attr("cy", y(v)).attr("r", 4.5).attr("fill", SC.accent);
        });
      });
      items = [{ label: "within + between", parts: [{ v: sw / N, col: SC.accent }, { v: sb / N, col: SC.a2 }] }, { label: "total", parts: [{ v: st / N, col: SC.muted }] }];
      txt = `SS_within = <b>${sw.toFixed(2)}</b> · SS_between = <b>${sb.toFixed(2)}</b> · SS_total = <b>${st.toFixed(2)}</b> = ${(sw + sb).toFixed(2)}` +
        `<br>divided by N = ${N}: E Var(Y | group) = ${(sw / N).toFixed(3)}, Var E[Y | group] = ${(sb / N).toFixed(3)}, Var Y = ${(st / N).toFixed(3)} · share explained by the group η² = ${st > 0 ? (sb / st).toFixed(3) : "—"}` +
        `<br>stretch the between spread to 0 and the groups explain nothing; stretch the within spread to 0 and they explain everything.`;
    } else {
      const kk = [mode, p1, p2, seed].join("|"); if (!cache[kk]) cache = { [kk]: EV.tvSim(mode, p1, p2, seed) };
      const S = cache[kk];
      let exW, exB, xmax, pmf, ref, refName;
      if (mode === "nb") { exW = p1; exB = p1 * p1 / p2; xmax = Math.ceil(Math.max(15, p1 + 4 * Math.sqrt(exW + exB))); pmf = k => PV.negBinFailPmf(k, p2, p2 / (p2 + p1)); ref = k => PV.poisPmf(k, p1); refName = `Poisson(${p1}) — no heterogeneity`; }
      else { const n = Math.round(p1); exW = n / 6; exB = n * n / 12; xmax = n; pmf = () => 1 / (n + 1); ref = k => ST.binomPmf(k, n, 0.5); refName = `Bin(${n}, ½) — one shared prevalence`; }
      const cnt = new Array(xmax + 1).fill(0); S.Y.forEach(v => { if (v <= xmax) cnt[v]++; });
      const x = d3.scaleBand().domain(d3.range(xmax + 1)).range([0, c1.iw]).padding(0.12);
      const ymax = 1.15 * Math.max(d3.max(cnt) / S.U, d3.max(d3.range(xmax + 1), k => Math.max(pmf(k), ref(k))));
      const y = d3.scaleLinear().domain([0, ymax]).range([c1.ih, 0]);
      ST.gridY(gL, y, c1.iw, 5);
      const step = Math.max(1, Math.ceil((xmax + 1) / 12));
      gL.append("g").attr("class", "axis").attr("transform", `translate(0,${c1.ih})`).call(d3.axisBottom(x).tickValues(d3.range(0, xmax + 1, step)));
      gL.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
      EV.label(gL, 0, -8, mode === "nb" ? `counts of ${d3.format(",")(S.U)} units, each with its own gamma rate` : `cases in ${d3.format(",")(S.U)} regions, each with its own prevalence`, { color: SC.ink });
      gL.selectAll("rect").data(cnt).join("rect").attr("x", (v, k) => x(k)).attr("width", x.bandwidth()).attr("y", v => y(v / S.U)).attr("height", v => c1.ih - y(v / S.U)).attr("fill", SC.accent).attr("opacity", 0.55);
      gL.selectAll("circle.p").data(d3.range(xmax + 1)).join("circle").attr("cx", k => x(k) + x.bandwidth() / 2).attr("cy", k => y(pmf(k))).attr("r", 2.6).attr("fill", SC.ink);
      gL.selectAll("circle.q").data(d3.range(xmax + 1)).join("circle").attr("cx", k => x(k) + x.bandwidth() / 2).attr("cy", k => y(ref(k))).attr("r", 3).attr("fill", "none").attr("stroke", SC.a2);
      ST.legend(gL, [{ label: "simulated", color: SC.accent }, { label: mode === "nb" ? "negative binomial pmf" : "uniform 1/(n + 1)", color: SC.ink }, { label: refName, color: SC.a2 }], c1.iw * 0.48, 8, { font: 9.5 });
      items = [
        { label: "exact", parts: [{ v: exW, col: SC.accent }, { v: exB, col: SC.a2 }] },
        { label: "simulated", parts: [{ v: S.within, col: SC.accent }, { v: S.between, col: SC.a2 }] },
        { label: "sim. total", parts: [{ v: S.total, col: SC.muted }] }
      ];
      const nm = mode === "nb" ? ["E Var(Y | Λ) = E Λ", "Var E[Y | Λ] = Var Λ"] : ["E Var(X | Q) = E nQ(1 − Q)", "Var E[X | Q] = Var nQ"];
      txt = `seed ${seed} · ${d3.format(",")(S.U)} ${mode === "nb" ? "units" : "regions"} · exact: <span style="color:${SC.accent}">${nm[0]} = <b>${exW.toFixed(3)}</b></span> + <span style="color:${SC.a2}">${nm[1]} = <b>${exB.toFixed(3)}</b></span> = <b>${(exW + exB).toFixed(3)}</b>` +
        `<br>simulated: within ${S.within.toFixed(3)} + between ${S.between.toFixed(3)} = ${(S.within + S.between).toFixed(3)} · sample variance of the counts ${S.total.toFixed(3)} · sample mean ${S.meanY.toFixed(3)}` +
        `<br>${mode === "nb" ? `a Poisson model with the same mean claims total variance ${p1.toFixed(1)}; the between-unit part is ${(100 * exB / (exW + exB)).toFixed(0)}% of the real total` : `a binomial with one shared prevalence ½ has variance ${(Math.round(p1) / 4).toFixed(2)}; between-region spread is ${(100 * exB / (exW + exB)).toFixed(0)}% of the total`}`;
    }
    const tmax = 1.18 * d3.max(items, it => d3.sum(it.parts, p => p.v));
    const yR = d3.scaleLinear().domain([0, tmax || 1]).range([c2.ih, 0]);
    ST.gridY(gR, yR, c2.iw, 5);
    gR.append("g").attr("class", "axis").call(d3.axisLeft(yR).ticks(5));
    EV.label(gR, 0, -8, "variance: within (blue) + between (orange) vs total", { color: SC.ink });
    bars(gR, 20, Math.min(70, (c2.iw - 40) / items.length - 16), c2.ih, yR, items);
    out.innerHTML = txt;
  }
  mS.addEventListener("change", () => { setup(true); draw(); });
  [p1S, p2S].forEach(el => el.addEventListener("input", draw));
  $("tv-sim").addEventListener("click", () => { seed = 1 + Math.floor(Math.random() * 1e9); draw(); });
  $("tv-reset").addEventListener("click", () => { mS.value = "nb"; seed = EV.seeds.totvar; setup(true); draw(); });
  setup(true); draw();
})();
