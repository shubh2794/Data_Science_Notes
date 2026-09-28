/* neuroevolution.viz.js — the figures on deep-learning/classical-nn/neuroevolution.html
   Loaded after ../../data.js → ../../notes.js (palette C).
   Each block is an IIFE that returns quietly if its <svg> is absent.

     1  #ga-svg      a genetic algorithm evolving the 9 weights of a 2-2-1 net for XOR
     2  #conv-svg    the competing-conventions problem: two identical nets, one broken child
     3  #neat-svg    NEAT genomes: add-node / add-connection, historical markings,
                     alignment, crossover and the compatibility distance
     4  #cppn-svg    a CPPN painting a HyperNEAT substrate's connectivity at any resolution
     5  #bw-svg      the Baldwin effect (Hinton & Nowlan 1987): learning guides evolution
     6  #casc-svg    cascade-correlation growing hidden units one at a time
     7  #obd-svg     which weight to prune: magnitude vs OBD vs OBS on a quadratic
     8  #es-svg      the evolution-strategies gradient estimate and Gaussian smoothing

   Every number a readout prints is computed from what the figure draws.       */

const NE = (function () {
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const sig = z => 1 / (1 + Math.exp(-z));
  function panel(svg, x, y) { return svg.append("g").attr("transform", `translate(${x},${y})`); }
  function axes(g, xs, ys, w, h, xl, yl, nx, ny) {
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xs).ticks(nx || 6));
    g.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(ny || 4));
    if (xl) g.append("text").attr("x", w).attr("y", h + 28).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 10.5).text(xl);
    if (yl) g.append("text").attr("x", 0).attr("y", -8).attr("fill", C.muted).attr("font-size", 10.5).text(yl);
  }
  function legend(g, x, y, items) {
    let off = 0;
    items.forEach(it => {
      const gg = g.append("g").attr("transform", `translate(${x + off},${y})`);
      off += it.w || 110;
      gg.append("line").attr("x1", 0).attr("x2", 16).attr("y1", 0).attr("y2", 0).attr("stroke", it.c).attr("stroke-width", 2.4).attr("stroke-dasharray", it.d || null);
      gg.append("text").attr("x", 21).attr("y", 4).attr("fill", C.muted).attr("font-size", 10.5).text(it.t);
    });
  }
  /* a red–blue diverging fill for a probability in [0,1] */
  const pcol = p => d3.interpolateRgb("#f87171", "#5b9cff")(Math.max(0, Math.min(1, p)));
  const XOR = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  return { rng, gauss, sig, panel, axes, legend, pcol, XOR };
})();

/* ═══════════════ 1 · #ga-svg — a GA evolves XOR weights ═══════════════ */
(function () {
  const svg = d3.select("#ga-svg");
  if (svg.empty()) return;
  const W = 760, H = 330, G = 9, GMAX = 200;
  const popS = document.getElementById("ga-pop"), sigS = document.getElementById("ga-sigma"), kS = document.getElementById("ga-k"),
    xS = document.getElementById("ga-x"), eliteC = document.getElementById("ga-elite"), RO = document.getElementById("ga-readout");
  /* genome: [w11 w12 b1 | w21 w22 b2 | v1 v2 c]  —  h = tanh(Wx + b),  o = σ(v·h + c) */
  function out(g, x, y) {
    const h1 = Math.tanh(g[0] * x + g[1] * y + g[2]), h2 = Math.tanh(g[3] * x + g[4] * y + g[5]);
    return NE.sig(g[6] * h1 + g[7] * h2 + g[8]);
  }
  const fit = g => 1 - NE.XOR.reduce((s, p) => s + (out(g, p[0], p[1]) - p[2]) ** 2, 0) / 4;
  const solved = g => NE.XOR.every(p => Math.abs(out(g, p[0], p[1]) - p[2]) < 0.4);
  let seed = 1, S;
  function init(sd) {
    const r = NE.rng(sd), n = +popS.value, pop = [];
    for (let i = 0; i < n; i++) { const g = new Float64Array(G); for (let j = 0; j < G; j++) g[j] = NE.gauss(r); pop.push(g); }
    return { r, pop, fits: pop.map(fit), gen: 0, hist: [], solvedAt: null };
  }
  function generation(st, cfg) {
    const { r, pop, fits } = st, n = pop.length;
    const tour = () => { let b = Math.floor(r() * n); for (let i = 1; i < cfg.k; i++) { const c = Math.floor(r() * n); if (fits[c] > fits[b]) b = c; } return pop[b]; };
    const next = [];
    if (cfg.elite) { let b = 0; for (let i = 1; i < n; i++) if (fits[i] > fits[b]) b = i; next.push(Float64Array.from(pop[b])); }
    while (next.length < n) {
      const a = tour(), b = tour(), c = new Float64Array(G);
      if (cfg.x === "uniform") for (let j = 0; j < G; j++) c[j] = r() < 0.5 ? a[j] : b[j];
      else if (cfg.x === "onepoint") { const cut = 1 + Math.floor(r() * (G - 1)); for (let j = 0; j < G; j++) c[j] = j < cut ? a[j] : b[j]; }
      else if (cfg.x === "arith") { const t = r(); for (let j = 0; j < G; j++) c[j] = t * a[j] + (1 - t) * b[j]; }
      else for (let j = 0; j < G; j++) c[j] = a[j];
      for (let j = 0; j < G; j++) if (r() < 0.25) c[j] += cfg.s * NE.gauss(r);
      next.push(c);
    }
    st.pop = next; st.fits = next.map(fit); st.gen++;
    record(st);
  }
  function record(st) {
    let b = 0; for (let i = 1; i < st.fits.length; i++) if (st.fits[i] > st.fits[b]) b = i;
    st.best = st.pop[b];
    st.hist.push({ g: st.gen, best: st.fits[b], mean: d3.mean(st.fits), med: d3.median(st.fits) });
    if (st.solvedAt == null && solved(st.best)) st.solvedAt = st.gen;
  }
  const cfg = () => ({ k: +kS.value, s: +sigS.value, x: xS.value, elite: eliteC.checked });
  let timer = null, batchTxt = "";
  function reset() { stop(); S = init(seed); record(S); batchTxt = ""; draw(); }
  function stop() { if (timer) { clearTimeout(timer); timer = null; } }
  function run() {
    stop();
    const c = cfg();
    const tick = () => {
      for (let i = 0; i < 2 && S.gen < GMAX; i++) generation(S, c);
      draw();
      if (S.gen < GMAX) timer = setTimeout(tick, 20); else timer = null;
    };
    tick();
  }
  function batch() {
    stop();
    const c = cfg(), gens = [];
    let ok = 0;
    for (let t = 0; t < 20; t++) {
      const st = init(1000 + t); record(st);
      while (st.solvedAt == null && st.gen < GMAX) generation(st, c);
      if (st.solvedAt != null) { ok++; gens.push(st.solvedAt); }
    }
    const med = gens.length ? d3.median(gens) : NaN;
    batchTxt = `<br>20 independent runs with these settings: solved <b>${ok}/20</b> within ${GMAX} generations` +
      (gens.length ? ` · median generation to solve <b>${med}</b> (≈ ${Math.round(med * +popS.value)} network evaluations)` : "");
    draw();
  }
  function draw() {
    ["pop", "sigma", "k"].forEach(k => { const el = document.getElementById("ga-" + k); document.getElementById("ga-" + k + "-out").textContent = k === "sigma" ? (+el.value).toFixed(2) : el.value; });
    svg.selectAll("*").remove();
    const L = NE.panel(svg, 46, 26), lw = 420, lh = 250;
    const xs = d3.scaleLinear().domain([0, GMAX]).range([0, lw]), ys = d3.scaleLinear().domain([0.4, 1]).range([lh, 0]);
    NE.axes(L, xs, ys, lw, lh, "generation", "fitness = 1 − MSE on the 4 XOR patterns", 8, 6);
    L.append("line").attr("x1", 0).attr("x2", lw).attr("y1", ys(0.96)).attr("y2", ys(0.96)).attr("stroke", C.line).attr("stroke-dasharray", "3,3");
    const ln = key => d3.line().x(d => xs(d.g)).y(d => ys(Math.max(0.4, d[key])));
    L.append("path").attr("d", ln("mean")(S.hist)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.5);
    L.append("path").attr("d", ln("med")(S.hist)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,3");
    L.append("path").attr("d", ln("best")(S.hist)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2.2);
    if (S.solvedAt != null) {
      L.append("line").attr("x1", xs(S.solvedAt)).attr("x2", xs(S.solvedAt)).attr("y1", 0).attr("y2", lh).attr("stroke", C.good).attr("stroke-dasharray", "4,3");
      L.append("text").attr("x", xs(S.solvedAt) + 4).attr("y", lh - 6).attr("fill", C.good).attr("font-size", 10.5).text(`solved at gen ${S.solvedAt}`);
    }
    NE.legend(L, 8, 10, [{ t: "best", c: C.B, w: 60 }, { t: "mean", c: C.A, w: 64 }, { t: "median", c: C.A, d: "3,3", w: 80 }]);
    // decision surface of the best individual
    const R = NE.panel(svg, 520, 26), rw = 220, n = 22, cs = rw / n;
    R.append("text").attr("x", 0).attr("y", -8).attr("fill", C.muted).attr("font-size", 10.5).text("best network's output over [0,1]²");
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = (i + 0.5) / n, y = 1 - (j + 0.5) / n;
      R.append("rect").attr("x", i * cs).attr("y", j * cs).attr("width", cs + 0.3).attr("height", cs + 0.3).attr("fill", NE.pcol(out(S.best, x, y))).attr("opacity", 0.75);
    }
    NE.XOR.forEach(p => {
      const o = out(S.best, p[0], p[1]);
      R.append("circle").attr("cx", p[0] * rw * 0.84 + rw * 0.08).attr("cy", (1 - p[1]) * rw * 0.84 + rw * 0.08).attr("r", 9)
        .attr("fill", p[2] ? "#5b9cff" : "#f87171").attr("stroke", Math.abs(o - p[2]) < 0.4 ? C.ink : "#000").attr("stroke-width", 2.5);
    });
    R.append("text").attr("x", 0).attr("y", rw + 18).attr("fill", C.muted).attr("font-size", 10.5).text("blue = 1, red = 0 · ring white when classified right");
    const h = S.hist[S.hist.length - 1];
    RO.innerHTML = `generation <b>${S.gen}</b> · evaluations <b>${S.gen * S.pop.length + S.pop.length}</b> · best fitness <b>${h.best.toFixed(4)}</b> · mean <b>${h.mean.toFixed(3)}</b>` +
      ` · best outputs on (00, 01, 10, 11): <b>${NE.XOR.map(p => out(S.best, p[0], p[1]).toFixed(2)).join(", ")}</b>` +
      (S.solvedAt != null ? ` · <b>solved at generation ${S.solvedAt}</b>` : " · not yet solved") + batchTxt;
  }
  document.getElementById("ga-run").addEventListener("click", run);
  document.getElementById("ga-step").addEventListener("click", () => { stop(); generation(S, cfg()); draw(); });
  document.getElementById("ga-reset").addEventListener("click", () => { seed++; reset(); });
  document.getElementById("ga-batch").addEventListener("click", batch);
  [popS, sigS, kS].forEach(e => e.addEventListener("input", () => { if (e === popS) reset(); else draw(); }));
  xS.addEventListener("change", () => { batchTxt = ""; draw(); });
  eliteC.addEventListener("change", () => { batchTxt = ""; draw(); });
  reset();
})();

/* ═══════════════ 2 · #conv-svg — competing conventions ═══════════════ */
(function () {
  const svg = d3.select("#conv-svg");
  if (svg.empty()) return;
  const W = 760, H = 330;
  const cutS = document.getElementById("conv-cut"), modeS = document.getElementById("conv-mode"), RO = document.getElementById("conv-readout");
  /* genome order: [h1: w1 w2 b | h2: w1 w2 b | out: v1 v2 c]   sigmoid units throughout */
  const OR = [6, 6, -3], AND = [6, 6, -9], NAND = [-6, -6, 9];
  const A = [...OR, ...AND, 8, -8, -4];                 // h1 = OR, h2 = AND, out = OR ∧ ¬AND = XOR
  const Bsw = [...AND, ...OR, -8, 8, -4];               // the same network, hidden units listed the other way round
  const Bind = [...OR, ...NAND, 8, 8, -12];             // a different XOR solution: OR ∧ NAND
  const f = (g, x, y) => {
    const h1 = NE.sig(g[0] * x + g[1] * y + g[2]), h2 = NE.sig(g[3] * x + g[4] * y + g[5]);
    return NE.sig(g[6] * h1 + g[7] * h2 + g[8]);
  };
  function drawNet(g, x0, title, gene, src) {
    const P = NE.panel(svg, x0, 20);
    P.append("text").attr("x", 105).attr("y", 0).attr("text-anchor", "middle").attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text(title);
    const pos = { x: [45, 175], y: [175, 175], h1: [45, 105], h2: [165, 105], o: [105, 38] };
    const edge = (a, b, w, gi) => {
      P.append("line").attr("x1", pos[a][0]).attr("y1", pos[a][1]).attr("x2", pos[b][0]).attr("y2", pos[b][1])
        .attr("stroke", w > 0 ? C.A : C.bad).attr("stroke-width", 0.8 + Math.abs(w) / 3).attr("opacity", 0.8);
      const mx = (pos[a][0] * 0.6 + pos[b][0] * 0.4), my = (pos[a][1] * 0.6 + pos[b][1] * 0.4);
      P.append("text").attr("x", mx).attr("y", my).attr("text-anchor", "middle").attr("font-size", 9.5)
        .attr("fill", src ? (src[gi] === "A" ? C.B : C.good) : C.muted).text(w);
    };
    edge("x", "h1", g[0], 0); edge("y", "h1", g[1], 1); edge("x", "h2", g[3], 3); edge("y", "h2", g[4], 4); edge("h1", "o", g[6], 6); edge("h2", "o", g[7], 7);
    [["x", "x"], ["y", "y"], ["h1", "h₁"], ["h2", "h₂"], ["o", "out"]].forEach(([k, t]) => {
      P.append("circle").attr("cx", pos[k][0]).attr("cy", pos[k][1]).attr("r", 14).attr("fill", "#1e222d").attr("stroke", C.muted);
      P.append("text").attr("x", pos[k][0]).attr("y", pos[k][1] + 4).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", C.ink).text(t);
    });
    P.append("text").attr("x", pos.h1[0] - 20).attr("y", pos.h1[1] + 4).attr("text-anchor", "end").attr("font-size", 9).attr("fill", C.muted).text("b " + g[2]);
    P.append("text").attr("x", pos.h2[0] + 20).attr("y", pos.h2[1] + 4).attr("font-size", 9).attr("fill", C.muted).text("b " + g[5]);
    P.append("text").attr("x", pos.o[0] + 20).attr("y", pos.o[1] + 4).attr("font-size", 9).attr("fill", C.muted).text("b " + g[8]);
    // genome strip
    for (let i = 0; i < 9; i++) {
      P.append("rect").attr("x", 6 + i * 22).attr("y", 205).attr("width", 20).attr("height", 18).attr("rx", 3)
        .attr("fill", src ? (src[i] === "A" ? "rgba(255,180,84,.25)" : "rgba(74,222,128,.22)") : "rgba(91,156,255,.14)").attr("stroke", C.line);
      P.append("text").attr("x", 16 + i * 22).attr("y", 218).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", C.ink).text(g[i]);
    }
    ["h₁", "h₂", "out"].forEach((t, i) => P.append("text").attr("x", 6 + i * 66 + 33).attr("y", 236).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", C.muted).text(t + " genes"));
    // truth table
    let err = 0;
    NE.XOR.forEach((p, i) => {
      const o = f(g, p[0], p[1]), ok = Math.abs(o - p[2]) < 0.5; err += (o - p[2]) ** 2;
      P.append("text").attr("x", 10 + i * 52).attr("y", 262).attr("font-size", 10).attr("fill", C.muted).text(`${p[0]}${p[1]}→`);
      P.append("text").attr("x", 30 + i * 52).attr("y", 262).attr("font-size", 10.5).attr("font-weight", 600).attr("fill", ok ? C.good : C.bad).text(o.toFixed(2));
    });
    return err / 4;
  }
  function draw() {
    const cut = +cutS.value, B = modeS.value === "swap" ? Bsw : Bind;
    document.getElementById("conv-cut-out").textContent = cut;
    svg.selectAll("*").remove();
    const child = [], src = [];
    for (let i = 0; i < 9; i++) { child.push(i < cut ? A[i] : B[i]); src.push(i < cut ? "A" : "B"); }
    const eA = drawNet(A, 10, "parent A (fitness 1st)", null, null);
    const eB = drawNet(B, 270, modeS.value === "swap" ? "parent B = A, units swapped" : "parent B, a different solution", null, null);
    const eC = drawNet(child, 530, `child: one-point cut after gene ${cut}`, null, src);
    svg.append("text").attr("x", 380).attr("y", 318).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10.5)
      .text("child genes: orange from A, green from B · outputs green when on the right side of 0.5");
    const h1 = child.slice(0, 3).join(","), h2 = child.slice(3, 6).join(",");
    const kind = v => (v === OR.join(",") ? "OR" : v === AND.join(",") ? "AND" : v === NAND.join(",") ? "NAND" : "mixed");
    RO.innerHTML = `MSE — parent A <b>${eA.toFixed(3)}</b> · parent B <b>${eB.toFixed(3)}</b> · child <b>${eC.toFixed(3)}</b>` +
      ` · child's hidden units: h₁ = <b>${kind(h1)}</b>, h₂ = <b>${kind(h2)}</b>` +
      (modeS.value === "swap" ? `<br>A and B compute the same function; with 2 hidden units there are 2! = 2 orderings (and with tanh-style sign symmetry, 2!·2² = 8 equivalent genomes)` :
        `<br>two genuinely different solutions: their hidden units do not correspond, so there is no ordering under which their genes line up`);
  }
  cutS.addEventListener("input", draw);
  modeS.addEventListener("change", draw);
  draw();
})();

/* ═══════════════ 3 · #neat-svg — NEAT genomes, markings and crossover ═══════════════ */
(function () {
  const svg = d3.select("#neat-svg");
  if (svg.empty()) return;
  const W = 760, H = 560, RO = document.getElementById("neat-readout");
  const C1 = 1, C2 = 1, C3 = 0.4, DT = 3.0;
  let r, innovOf, splitOf, nextInnov, nextNode, P1, P2, CH;
  function fresh() {
    return { nodes: [{ id: 1, t: "in" }, { id: 2, t: "in" }, { id: 3, t: "bias" }, { id: 4, t: "out" }],
      conns: [{ inn: 1, a: 1, b: 4, w: 0.7, en: true }, { inn: 2, a: 2, b: 4, w: -0.5, en: true }, { inn: 3, a: 3, b: 4, w: 0.3, en: true }] };
  }
  const clone = g => ({ nodes: g.nodes.map(n => ({ ...n })), conns: g.conns.map(c => ({ ...c })) });
  /* a global innovation table: the same structural change always gets the same number */
  function innov(a, b) { const k = a + ">" + b; if (!(k in innovOf)) innovOf[k] = nextInnov++; return innovOf[k]; }
  function reaches(g, from, to) {   // is there a directed path from → to?
    const seen = new Set([from]), st = [from];
    while (st.length) { const u = st.pop(); if (u === to) return true; g.conns.forEach(c => { if (c.a === u && !seen.has(c.b)) { seen.add(c.b); st.push(c.b); } }); }
    return false;
  }
  function addConn(g) {
    const cand = [];
    g.nodes.forEach(a => g.nodes.forEach(b => {
      if (a.id === b.id || a.t === "out" || b.t === "in" || b.t === "bias") return;
      if (g.conns.some(c => c.a === a.id && c.b === b.id)) return;
      if (reaches(g, b.id, a.id)) return;                        // keep it feedforward
      cand.push([a.id, b.id]);
    }));
    if (!cand.length) return "no legal new connection";
    const [a, b] = cand[Math.floor(r() * cand.length)];
    const inn = innov(a, b);
    g.conns.push({ inn, a, b, w: +(NE.gauss(r)).toFixed(2), en: true });
    return `add connection ${a}→${b}: innovation ${inn}`;
  }
  function addNode(g) {
    const en = g.conns.filter(c => c.en);
    if (!en.length) return "no enabled connection to split";
    const c = en[Math.floor(r() * en.length)];
    let n = splitOf[c.inn];
    if (n == null || g.nodes.some(x => x.id === n)) { n = nextNode++; if (splitOf[c.inn] == null) splitOf[c.inn] = n; }
    c.en = false;
    g.nodes.push({ id: n, t: "hid" });
    const i1 = innov(c.a, n), i2 = innov(n, c.b);
    g.conns.push({ inn: i1, a: c.a, b: n, w: 1, en: true }, { inn: i2, a: n, b: c.b, w: c.w, en: true });
    return `split ${c.a}→${c.b} (innov ${c.inn}, now disabled) with node ${n}: innovations ${i1} (weight 1) and ${i2} (old weight ${c.w})`;
  }
  function perturb(g) { g.conns.forEach(c => { c.w = +(c.w + 0.3 * NE.gauss(r)).toFixed(2); }); return "weights perturbed"; }
  /* crossover: P1 is the fitter parent; matching genes random, disjoint/excess from P1 */
  function crossover(p1, p2) {
    const m2 = new Map(p2.conns.map(c => [c.inn, c]));
    const conns = p1.conns.map(c => {
      const d = m2.get(c.inn);
      if (!d) return { ...c, from: "P1" };
      const pick = r() < 0.5 ? c : d;
      const disabled = (!c.en || !d.en) && r() < 0.75;
      return { ...pick, en: !disabled, from: pick === c ? "P1" : "P2" };
    });
    const ids = new Set(); conns.forEach(c => { ids.add(c.a); ids.add(c.b); });
    const nodes = p1.nodes.filter(n => n.t !== "hid" || ids.has(n.id));
    return { nodes, conns };
  }
  function distance(p1, p2) {
    const m1 = new Map(p1.conns.map(c => [c.inn, c])), m2 = new Map(p2.conns.map(c => [c.inn, c]));
    const max1 = d3.max(p1.conns, c => c.inn), max2 = d3.max(p2.conns, c => c.inn), lim = Math.min(max1, max2);
    let E = 0, D = 0, M = 0, Wd = 0;
    new Set([...m1.keys(), ...m2.keys()]).forEach(i => {
      if (m1.has(i) && m2.has(i)) { M++; Wd += Math.abs(m1.get(i).w - m2.get(i).w); }
      else if (i > lim) E++; else D++;
    });
    const N0 = Math.max(p1.conns.length, p2.conns.length), N = N0 < 20 ? 1 : N0;
    const Wb = M ? Wd / M : 0;
    return { E, D, M, Wb, N, d: C1 * E / N + C2 * D / N + C3 * Wb, lim };
  }
  let log = [];
  function reset() {
    r = NE.rng(5); innovOf = { "1>4": 1, "2>4": 2, "3>4": 3 }; splitOf = {}; nextInnov = 4; nextNode = 5;
    P1 = fresh(); P2 = fresh();
    // an evolutionary history that makes the markings visible
    const c1 = P1.conns.find(c => c.inn === 1), c2 = P2.conns.find(c => c.inn === 1);
    [[P1, c1], [P2, c2]].forEach(([g, c]) => {
      let n = splitOf[c.inn]; if (n == null) { n = nextNode++; splitOf[c.inn] = n; }
      c.en = false; g.nodes.push({ id: n, t: "hid" });
      g.conns.push({ inn: innov(c.a, n), a: c.a, b: n, w: 1, en: true }, { inn: innov(n, c.b), a: n, b: c.b, w: c.w, en: true });
    });
    P2.conns.push({ inn: innov(2, 5), a: 2, b: 5, w: 0.9, en: true });
    P1.conns.push({ inn: innov(3, 5), a: 3, b: 5, w: -0.8, en: true });
    P2.conns.find(c => c.inn === 5).w = 0.2;
    log = ["both parents independently split connection 1→4 — and receive the SAME node 5 and innovations 4, 5"];
    CH = crossover(P1, P2);
    draw();
  }
  function layout(g) {
    const depth = {};
    g.nodes.forEach(n => { depth[n.id] = n.t === "in" || n.t === "bias" ? 0 : null; });
    for (let it = 0; it < 20; it++) g.conns.forEach(c => { if (depth[c.a] != null) depth[c.b] = Math.max(depth[c.b] || 0, depth[c.a] + 1); });
    const out = g.nodes.find(n => n.t === "out"), maxD = Math.max(1, ...g.nodes.filter(n => n.t !== "out").map(n => depth[n.id] || 0)) + 1;
    depth[out.id] = maxD;
    const byD = {}; g.nodes.forEach(n => { const d = depth[n.id] == null ? 1 : depth[n.id]; (byD[d] = byD[d] || []).push(n.id); });
    const pos = {};
    Object.keys(byD).forEach(d => { const ids = byD[d].sort((a, b) => a - b); ids.forEach((id, i) => { pos[id] = [(i + 1) / (ids.length + 1), 1 - d / maxD]; }); });
    return pos;
  }
  function drawNet(g, x0, y0, w, h, title) {
    const P = NE.panel(svg, x0, y0), pos = layout(g);
    P.append("text").attr("x", w / 2).attr("y", -6).attr("text-anchor", "middle").attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text(title);
    const X = id => 14 + pos[id][0] * (w - 28), Y = id => 16 + pos[id][1] * (h - 32);
    g.conns.forEach(c => {
      const dx = X(c.b) - X(c.a), dy = Y(c.b) - Y(c.a), bend = Math.abs(dx) < 4 ? 26 : 0;
      P.append("path").attr("d", `M${X(c.a)},${Y(c.a)} Q${(X(c.a) + X(c.b)) / 2 + bend},${(Y(c.a) + Y(c.b)) / 2} ${X(c.b)},${Y(c.b)}`)
        .attr("fill", "none").attr("stroke", !c.en ? C.line : c.w > 0 ? C.A : C.bad).attr("stroke-width", c.en ? 1 + Math.min(3, Math.abs(c.w) * 1.5) : 1)
        .attr("stroke-dasharray", c.en ? null : "3,3");
      P.append("text").attr("x", (X(c.a) + X(c.b)) / 2 + bend * 0.6 + 4).attr("y", (Y(c.a) + Y(c.b)) / 2).attr("font-size", 8.5)
        .attr("fill", c.en ? C.muted : C.line).text(c.inn);
    });
    g.nodes.forEach(n => {
      const col = n.t === "in" ? C.good : n.t === "bias" ? C.muted : n.t === "out" ? C.B : C.A;
      P.append("circle").attr("cx", X(n.id)).attr("cy", Y(n.id)).attr("r", 11).attr("fill", "#1e222d").attr("stroke", col).attr("stroke-width", 1.8);
      P.append("text").attr("x", X(n.id)).attr("y", Y(n.id) + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", C.ink).text(n.id);
    });
  }
  function draw() {
    svg.selectAll("*").remove();
    const dd = distance(P1, P2);
    const all = [...new Set([...P1.conns, ...P2.conns, ...CH.conns].map(c => c.inn))].sort((a, b) => a - b);
    const cw = Math.min(46, (W - 90) / all.length), T = NE.panel(svg, 70, 36);
    T.append("text").attr("x", -64).attr("y", -16).attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text("genes aligned by innovation number (historical marking)");
    const rows = [["parent 1", P1, 0], ["parent 2", P2, 1], ["child", CH, 2]];
    const m1 = new Map(P1.conns.map(c => [c.inn, c])), m2 = new Map(P2.conns.map(c => [c.inn, c]));
    rows.forEach(([lab, g, ri]) => {
      T.append("text").attr("x", -8).attr("y", ri * 44 + 24).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 10.5).text(lab);
      const m = new Map(g.conns.map(c => [c.inn, c]));
      all.forEach((inn, i) => {
        const c = m.get(inn);
        if (!c) return;
        let kind = "match";
        if (ri < 2) kind = m1.has(inn) && m2.has(inn) ? "match" : inn > dd.lim ? "excess" : "disjoint";
        const fill = ri === 2 ? (c.from === "P1" ? "rgba(255,180,84,.22)" : "rgba(74,222,128,.2)") :
          kind === "match" ? "rgba(91,156,255,.18)" : kind === "excess" ? "rgba(248,113,113,.25)" : "rgba(255,180,84,.25)";
        T.append("rect").attr("x", i * cw).attr("y", ri * 44).attr("width", cw - 3).attr("height", 38).attr("rx", 4)
          .attr("fill", fill).attr("stroke", c.en ? C.line : C.bad).attr("stroke-dasharray", c.en ? null : "3,2");
        T.append("text").attr("x", i * cw + (cw - 3) / 2).attr("y", ri * 44 + 14).attr("text-anchor", "middle").attr("font-size", 10).attr("font-weight", 700).attr("fill", C.ink).text(inn);
        T.append("text").attr("x", i * cw + (cw - 3) / 2).attr("y", ri * 44 + 29).attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", c.en ? C.muted : C.bad).text(`${c.a}→${c.b}${c.en ? "" : " ✕"}`);
      });
    });
    NE.legend(svg, 70, 186, [{ t: "matching", c: C.A, w: 90 }, { t: "disjoint", c: C.B, w: 85 }, { t: "excess", c: C.bad, w: 80 }, { t: "child: from P1 (orange) / P2 (green)", c: C.good, w: 250 }, { t: "✕ disabled", c: C.line, d: "3,2", w: 90 }]);
    drawNet(P1, 20, 230, 230, 290, "parent 1 (fitter)");
    drawNet(P2, 265, 230, 230, 290, "parent 2");
    drawNet(CH, 510, 230, 230, 290, "child");
    svg.append("text").attr("x", 380).attr("y", 548).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10.5)
      .text("green = input, grey = bias, blue = hidden, orange = output · numbers on edges are innovation numbers · dashed = disabled");
    RO.innerHTML = `compatibility δ = c₁E/N + c₂D/N + c₃W̄ = ${C1}·${dd.E}/${dd.N} + ${C2}·${dd.D}/${dd.N} + ${C3}·${dd.Wb.toFixed(3)} = <b>${dd.d.toFixed(3)}</b>` +
      ` · matching ${dd.M} · disjoint ${dd.D} · excess ${dd.E} · ${dd.d < DT ? "<b>same species</b>" : "<b>different species</b>"} (threshold δₜ = ${DT})<br>` +
      `genes: parent 1 ${P1.conns.length}, parent 2 ${P2.conns.length}, child ${CH.conns.length} · next innovation number ${nextInnov}<br>last change: ${log[log.length - 1]}`;
  }
  const act = (fn, g) => () => { log.push(fn(g === 1 ? P1 : g === 2 ? P2 : null)); CH = crossover(P1, P2); draw(); };
  document.getElementById("neat-a-conn").addEventListener("click", act(addConn, 1));
  document.getElementById("neat-a-node").addEventListener("click", act(addNode, 1));
  document.getElementById("neat-b-conn").addEventListener("click", act(addConn, 2));
  document.getElementById("neat-b-node").addEventListener("click", act(addNode, 2));
  document.getElementById("neat-w").addEventListener("click", () => { perturb(P1); perturb(P2); log.push("weights of both parents perturbed (structure unchanged)"); CH = crossover(P1, P2); draw(); });
  document.getElementById("neat-cross").addEventListener("click", () => { CH = crossover(P1, P2); log.push("crossover re-drawn: matching genes chosen at random, disjoint and excess genes from parent 1"); draw(); });
  document.getElementById("neat-reset").addEventListener("click", reset);
  reset();
})();

/* ═══════════════ 4 · #cppn-svg — a CPPN paints a substrate ═══════════════ */
(function () {
  const svg = d3.select("#cppn-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const preS = document.getElementById("cppn-pre"), resS = document.getElementById("cppn-res"), aS = document.getElementById("cppn-a"),
    thS = document.getElementById("cppn-th"), RO = document.getElementById("cppn-readout");
  const gaussF = z => Math.exp(-z * z);
  /* each preset is a small CPPN: a composition of a few functions of (x1, x2) — the two
     endpoints of a connection on a 1-D substrate, coordinates in [−1, 1]                   */
  const PRE = {
    local: { name: "locality: gauss(a·(x₂ − x₁))", f: (x1, x2, a) => gaussF(a * (x2 - x1)), nodes: 2 },
    mirror: { name: "mirror symmetry: gauss(a·(x₂ + x₁)) − 0.5·gauss(a·(x₂ − x₁))", f: (x1, x2, a) => gaussF(a * (x2 + x1)) - 0.5 * gaussF(a * (x2 - x1)), nodes: 4 },
    repeat: { name: "repetition: sin(a·(x₂ − x₁)) · gauss(x₁)", f: (x1, x2, a) => Math.sin(a * (x2 - x1)) * gaussF(x1), nodes: 4 },
    centre: { name: "centre–surround: gauss(a·x₁)·(2·gauss(3(x₂ − x₁)) − 1)", f: (x1, x2, a) => gaussF(a * x1 * 0.5) * (2 * gaussF(3 * (x2 - x1)) - 1), nodes: 5 },
    sym: { name: "bilateral mix: sin(a·x₁·x₂) · tanh(2·x₂)", f: (x1, x2, a) => Math.sin(a * x1 * x2) * Math.tanh(2 * x2), nodes: 4 }
  };
  function draw() {
    const P = PRE[preS.value], n = +resS.value, a = +aS.value, th = +thS.value;
    document.getElementById("cppn-res-out").textContent = n;
    document.getElementById("cppn-a-out").textContent = a.toFixed(1);
    document.getElementById("cppn-th-out").textContent = th.toFixed(2);
    svg.selectAll("*").remove();
    const X = i => (n === 1 ? 0 : -1 + 2 * i / (n - 1));
    const Wm = [];
    for (let i = 0; i < n; i++) { Wm.push([]); for (let j = 0; j < n; j++) Wm[i].push(P.f(X(i), X(j), a)); }
    // left: the substrate network — sources at the bottom row, targets at the top row
    const L = NE.panel(svg, 20, 30), lw = 400, lh = 300;
    L.append("text").attr("x", 0).attr("y", -10).attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text(`substrate: ${n} → ${n} neurons, weights queried from the CPPN`);
    const sx = i => 20 + (lw - 40) * (n === 1 ? 0.5 : i / (n - 1));
    let expressed = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const w = Wm[i][j];
      if (Math.abs(w) < th) continue;
      expressed++;
      L.append("line").attr("x1", sx(i)).attr("y1", lh - 20).attr("x2", sx(j)).attr("y2", 30)
        .attr("stroke", w > 0 ? C.A : C.bad).attr("stroke-width", Math.min(2.5, 0.3 + Math.abs(w) * 1.5)).attr("opacity", Math.min(0.9, 0.15 + Math.abs(w) * 0.6));
    }
    for (let i = 0; i < n; i++) {
      L.append("circle").attr("cx", sx(i)).attr("cy", lh - 20).attr("r", Math.max(2.5, Math.min(7, 70 / n))).attr("fill", C.good);
      L.append("circle").attr("cx", sx(i)).attr("cy", 30).attr("r", Math.max(2.5, Math.min(7, 70 / n))).attr("fill", C.B);
    }
    L.append("text").attr("x", 0).attr("y", lh + 2).attr("fill", C.muted).attr("font-size", 10).text("source layer, x₁ from −1 to +1");
    L.append("text").attr("x", 0).attr("y", 14).attr("fill", C.muted).attr("font-size", 10).text("target layer, x₂ from −1 to +1");
    // right: the weight matrix as a pattern in (x1, x2) space
    const R = NE.panel(svg, 470, 30), rw = 260, cs = rw / n;
    R.append("text").attr("x", 0).attr("y", -10).attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text("the same weights as a 2-D pattern");
    const col = d3.scaleDiverging(d3.interpolateRdBu).domain([1, 0, -1]);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const w = Wm[i][j];
      R.append("rect").attr("x", i * cs).attr("y", (n - 1 - j) * cs).attr("width", cs + 0.2).attr("height", cs + 0.2)
        .attr("fill", Math.abs(w) < th ? "#1e222d" : col(Math.max(-1, Math.min(1, w))));
    }
    R.append("text").attr("x", rw / 2).attr("y", rw + 16).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10).text("x₁ (source) →");
    R.append("text").attr("x", -8).attr("y", rw / 2).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 10).text("x₂ ↑");
    R.append("text").attr("x", 0).attr("y", rw + 34).attr("fill", C.muted).attr("font-size", 10).text("dark = below the expression threshold (no link)");
    RO.innerHTML = `CPPN: <b>${P.name}</b> · genome ≈ <b>${P.nodes + 1}</b> evolvable numbers (a few nodes + the parameter a)<br>` +
      `phenotype: <b>${n}×${n} = ${n * n}</b> potential connections, <b>${expressed}</b> expressed above |w| ≥ ${th.toFixed(2)} · ` +
      `compression ratio ≈ <b>${Math.round(n * n / (P.nodes + 1))}:1</b> — raise the resolution and the SAME genome describes a bigger network with the same regularity`;
  }
  [resS, aS, thS].forEach(e => e.addEventListener("input", draw));
  preS.addEventListener("change", draw);
  draw();
})();

/* ═══════════════ 5 · #bw-svg — the Baldwin effect ═══════════════ */
(function () {
  const svg = d3.select("#bw-svg");
  if (svg.empty()) return;
  const W = 760, H = 320, NP = 1000, L = 20, GEN = 50;
  const modeS = document.getElementById("bw-mode"), trS = document.getElementById("bw-trials"), RO = document.getElementById("bw-readout");
  let S, timer = null;
  /* alleles: 1 = correct, 0 = incorrect, 2 = '?', learnable */
  function init() {
    const r = NE.rng(42), learn = modeS.value === "learn", pop = [];
    for (let i = 0; i < NP; i++) {
      const g = new Uint8Array(L);
      for (let j = 0; j < L; j++) { const u = r(); g[j] = learn ? (u < 0.25 ? 1 : u < 0.5 ? 0 : 2) : (u < 0.5 ? 1 : 0); }
      pop.push(g);
    }
    S = { r, pop, gen: 0, hist: [], learn, trials: +trS.value, found: 0 };
    stats();
  }
  function fitness(g) {
    let q = 0;
    for (let j = 0; j < L; j++) { if (g[j] === 0) return { f: 1, ok: false }; if (g[j] === 2) q++; }
    if (q === 0) return { f: 20, ok: true };
    const p = Math.pow(2, -q), u = S.r() || 1e-12;
    const T = Math.ceil(Math.log(u) / Math.log1p(-p));      // first success of random guessing (geometric)
    if (T > S.trials) return { f: 1, ok: false };
    return { f: 1 + 19 * (S.trials - T) / S.trials, ok: true };
  }
  function stats() {
    let c1 = 0, c0 = 0, c2 = 0;
    S.pop.forEach(g => { for (let j = 0; j < L; j++) { if (g[j] === 1) c1++; else if (g[j] === 0) c0++; else c2++; } });
    const T = NP * L;
    S.fits = S.pop.map(fitness);
    S.hist.push({ g: S.gen, c: c1 / T, i: c0 / T, q: c2 / T, mf: d3.mean(S.fits, d => d.f), ok: S.fits.filter(d => d.ok).length / NP });
  }
  function step() {
    const cum = [], fs = S.fits.map(d => d.f); let s = 0;
    fs.forEach(f => { s += f; cum.push(s); });
    const pick = () => { const u = S.r() * s; let lo = 0, hi = NP - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < u) lo = m + 1; else hi = m; } return S.pop[lo]; };
    const next = [];
    for (let i = 0; i < NP; i++) {
      const a = pick(), b = pick(), cut = 1 + Math.floor(S.r() * (L - 1)), c = new Uint8Array(L);
      for (let j = 0; j < L; j++) c[j] = j < cut ? a[j] : b[j];
      next.push(c);
    }
    S.pop = next; S.gen++;
    stats();
  }
  function stop() { if (timer) { clearTimeout(timer); timer = null; } }
  function run() {
    stop(); init();
    const tick = () => { step(); step(); draw(); if (S.gen < GEN) timer = setTimeout(tick, 30); else timer = null; };
    draw(); timer = setTimeout(tick, 30);
  }
  function draw() {
    document.getElementById("bw-trials-out").textContent = trS.value;
    svg.selectAll("*").remove();
    const P = NE.panel(svg, 50, 26), pw = 480, ph = 250;
    const xs = d3.scaleLinear().domain([0, GEN]).range([0, pw]), ys = d3.scaleLinear().domain([0, 1]).range([ph, 0]);
    NE.axes(P, xs, ys, pw, ph, "generation", "allele frequency across the population", 10, 5);
    const ln = k => d3.line().x(d => xs(d.g)).y(d => ys(d[k]))(S.hist);
    P.append("path").attr("d", ln("c")).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 2.2);
    P.append("path").attr("d", ln("i")).attr("fill", "none").attr("stroke", C.bad).attr("stroke-width", 2.2);
    if (S.learn) P.append("path").attr("d", ln("q")).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2.2);
    P.append("path").attr("d", ln("ok")).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
    const legItems = [{ t: "correct (1)", c: C.good }, { t: "incorrect (0)", c: C.bad }].concat(S.learn ? [{ t: "learnable (?)", c: C.A }] : []).concat([{ t: "share that find the target", c: C.B, d: "4,3" }]);
    legItems.forEach((it, i) => NE.legend(svg, 570, 205 + i * 18, [it]));
    const h = S.hist[S.hist.length - 1];
    const Rg = NE.panel(svg, 570, 26);
    Rg.append("text").attr("x", 0).attr("y", 0).attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text("one genome, generation " + S.gen);
    const best = S.pop[d3.maxIndex(S.fits, d => d.f)];
    for (let j = 0; j < L; j++) {
      const v = best[j];
      Rg.append("rect").attr("x", (j % 5) * 32).attr("y", 14 + Math.floor(j / 5) * 32).attr("width", 28).attr("height", 28).attr("rx", 4)
        .attr("fill", v === 1 ? "rgba(74,222,128,.3)" : v === 0 ? "rgba(248,113,113,.3)" : "rgba(91,156,255,.25)");
      Rg.append("text").attr("x", (j % 5) * 32 + 14).attr("y", 14 + Math.floor(j / 5) * 32 + 19).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", C.ink).text(v === 2 ? "?" : v);
    }
    Rg.append("text").attr("x", 0).attr("y", 160).attr("fill", C.muted).attr("font-size", 10).text("the fittest individual's 20 loci");
    RO.innerHTML = `${S.learn ? "with learning (? alleles are guessed during life)" : "without learning (every allele fixed at birth)"} · ${S.trials} learning trials · generation <b>${S.gen}</b><br>` +
      `alleles: correct <b>${(h.c * 100).toFixed(1)}%</b> · incorrect <b>${(h.i * 100).toFixed(1)}%</b>` + (S.learn ? ` · learnable <b>${(h.q * 100).toFixed(1)}%</b>` : "") +
      ` · individuals reaching the target this generation <b>${(h.ok * 100).toFixed(1)}%</b> · mean fitness <b>${h.mf.toFixed(2)}</b> (max 20)`;
  }
  document.getElementById("bw-run").addEventListener("click", run);
  modeS.addEventListener("change", () => { stop(); init(); draw(); });
  trS.addEventListener("input", () => { stop(); init(); draw(); });
  init();
  for (let i = 0; i < GEN; i++) step();                     // show the finished run on load
  draw();
})();

/* ═══════════════ 6 · #casc-svg — cascade-correlation ═══════════════ */
(function () {
  const svg = d3.select("#casc-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, MAXU = 16, NC = 8, RES = 34;
  const dataS = document.getElementById("casc-data"), RO = document.getElementById("casc-readout");
  let D, S, timer = null;
  function dataset(kind) {
    const pts = [];
    if (kind === "spirals") {
      for (let i = 0; i < 97; i++) {
        const a = i * Math.PI / 16, rr = 6.5 * (104 - i) / 104 / 6.5;
        pts.push([rr * Math.sin(a), rr * Math.cos(a), 1], [-rr * Math.sin(a), -rr * Math.cos(a), 0]);
      }
    } else if (kind === "rings") {
      const r = NE.rng(3);
      for (let i = 0; i < 240; i++) { const a = r() * 2 * Math.PI, rr = i % 2 ? 0.15 + 0.3 * r() : 0.62 + 0.3 * r(); pts.push([rr * Math.cos(a), rr * Math.sin(a), i % 2]); }
    } else {
      const r = NE.rng(9);
      for (let i = 0; i < 240; i++) { const x = 2 * r() - 1, y = 2 * r() - 1; pts.push([x, y, (x > 0) !== (y > 0) ? 1 : 0]); }
    }
    return pts;
  }
  /* per-pattern feature vectors z = [x, y, 1, h1, h2, …] — hidden values are frozen once computed */
  function reset() {
    stop();
    D = dataset(dataS.value);
    S = { Z: D.map(p => [p[0], p[1], 1]), T: D.map(p => p[2]), units: [], wo: [0, 0, 0], hist: [], adds: [], epoch: 0, r: NE.rng(12) };
    trainOut(300);
    draw();
  }
  const outP = (wo, z) => { let s = 0; for (let i = 0; i < wo.length; i++) s += wo[i] * z[i]; return NE.sig(s); };
  function trainOut(epochs) {
    const n = S.wo.length, m = new Float64Array(n), v = new Float64Array(n);
    let t = 0;
    for (let ep = 0; ep < epochs; ep++) {
      const g = new Float64Array(n);
      for (let p = 0; p < S.Z.length; p++) { const e = outP(S.wo, S.Z[p]) - S.T[p]; for (let i = 0; i < n; i++) g[i] += e * S.Z[p][i] / S.Z.length; }
      t++;
      for (let i = 0; i < n; i++) { m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]; S.wo[i] -= 0.08 * (m[i] / (1 - 0.9 ** t)) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8); }
      S.epoch++;
      if (ep % 10 === 9) S.hist.push({ e: S.epoch, err: errRate() });
    }
  }
  function errRate() { let c = 0; for (let p = 0; p < S.Z.length; p++) if ((outP(S.wo, S.Z[p]) > 0.5 ? 1 : 0) !== S.T[p]) c++; return c / S.Z.length; }
  /* train a pool of candidates to maximise |Σ_p (V_p − V̄)(E_p − Ē)|; install the winner, frozen */
  function addUnit() {
    if (S.units.length >= MAXU) return;
    const P = S.Z.length, n = S.Z[0].length;
    const E = S.Z.map((z, p) => outP(S.wo, z) - S.T[p]), Eb = d3.mean(E);
    let best = null;
    for (let c = 0; c < NC; c++) {
      const w = new Float64Array(n), m = new Float64Array(n), v = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = (2 * S.r() - 1) * (i < 3 ? 3 : 1);
      let Sc = 0;
      for (let ep = 1; ep <= 160; ep++) {
        const V = new Float64Array(P);
        for (let p = 0; p < P; p++) { let s = 0; for (let i = 0; i < n; i++) s += w[i] * S.Z[p][i]; V[p] = Math.tanh(s); }
        const Vb = d3.mean(V);
        let cov = 0; for (let p = 0; p < P; p++) cov += (V[p] - Vb) * (E[p] - Eb);
        Sc = Math.abs(cov); const sg = Math.sign(cov) || 1;
        const g = new Float64Array(n);
        for (let p = 0; p < P; p++) { const k = sg * (E[p] - Eb) * (1 - V[p] * V[p]); for (let i = 0; i < n; i++) g[i] += k * S.Z[p][i]; }
        for (let i = 0; i < n; i++) { m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]; w[i] += 0.06 * (m[i] / (1 - 0.9 ** ep)) / (Math.sqrt(v[i] / (1 - 0.999 ** ep)) + 1e-8); }
      }
      if (!best || Sc > best.S) best = { w: Array.from(w), S: Sc };
    }
    S.units.push(best);
    S.Z.forEach(z => { let s = 0; for (let i = 0; i < best.w.length; i++) s += best.w[i] * z[i]; z.push(Math.tanh(s)); });
    S.wo.push(0);
    S.adds.push(S.epoch);
    trainOut(300);
  }
  function stop() { if (timer) { clearTimeout(timer); timer = null; } }
  function grow() {
    stop();
    const tick = () => { if (S.units.length < MAXU && errRate() > 0) { addUnit(); draw(); timer = setTimeout(tick, 30); } else { timer = null; draw(); } };
    tick();
  }
  function feat(x, y) {
    const z = [x, y, 1];
    S.units.forEach(u => { let s = 0; for (let i = 0; i < u.w.length; i++) s += u.w[i] * z[i]; z.push(Math.tanh(s)); });
    return z;
  }
  function draw() {
    svg.selectAll("*").remove();
    const M = NE.panel(svg, 20, 26), mw = 300, cs = mw / RES;
    M.append("text").attr("x", 0).attr("y", -8).attr("fill", C.muted).attr("font-size", 10.5).text("decision surface (blue = class 1)");
    for (let i = 0; i < RES; i++) for (let j = 0; j < RES; j++) {
      const x = -1.05 + 2.1 * (i + 0.5) / RES, y = 1.05 - 2.1 * (j + 0.5) / RES;
      M.append("rect").attr("x", i * cs).attr("y", j * cs).attr("width", cs + 0.3).attr("height", cs + 0.3).attr("fill", NE.pcol(outP(S.wo, feat(x, y)))).attr("opacity", 0.55);
    }
    const px = x => (x + 1.05) / 2.1 * mw, py = y => (1.05 - y) / 2.1 * mw;
    D.forEach((p, k) => {
      const wrong = (outP(S.wo, S.Z[k]) > 0.5 ? 1 : 0) !== p[2];
      M.append("circle").attr("cx", px(p[0])).attr("cy", py(p[1])).attr("r", 2.6).attr("fill", p[2] ? "#5b9cff" : "#f87171").attr("stroke", wrong ? "#fff" : "#0f1117").attr("stroke-width", wrong ? 1.4 : 0.6);
    });
    // error curve
    const E = NE.panel(svg, 370, 26), ew = 370, eh = 150;
    const xs = d3.scaleLinear().domain([0, Math.max(300, S.epoch)]).range([0, ew]), ys = d3.scaleLinear().domain([0, 0.6]).range([eh, 0]);
    NE.axes(E, xs, ys, ew, eh, "output-training epochs (candidate training not counted)", "training error rate", 6, 3);
    S.adds.forEach((a, i) => {
      E.append("line").attr("x1", xs(a)).attr("x2", xs(a)).attr("y1", 0).attr("y2", eh).attr("stroke", C.good).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "2,3");
      if (S.adds.length < 10 || i % 2 === 0) E.append("text").attr("x", xs(a) + 2).attr("y", 9).attr("fill", C.good).attr("font-size", 9).text("+" + (i + 1));
    });
    E.append("path").attr("d", d3.line().x(d => xs(d.e)).y(d => ys(Math.min(0.6, d.err)))(S.hist)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.8);
    // the cascade connection matrix
    const K = NE.panel(svg, 370, 230), nU = S.units.length, cols = 3 + nU, cw = Math.min(18, 360 / Math.max(cols, 1)), rows = nU + 1;
    K.append("text").attr("x", 0).attr("y", -8).attr("fill", C.muted).attr("font-size", 10.5).text("connections: rows receive from columns (grey = frozen, orange = trained)");
    const lab = ["x", "y", "1"].concat(S.units.map((u, i) => "h" + (i + 1)));
    lab.forEach((t, c) => K.append("text").attr("x", 34 + c * cw + cw / 2).attr("y", 8).attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", C.muted).text(t));
    for (let rI = 0; rI < rows; rI++) {
      const isOut = rI === nU, rh = Math.min(12, 105 / rows);
      K.append("text").attr("x", 30).attr("y", 14 + rI * rh + rh * 0.8).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", isOut ? C.B : C.muted).text(isOut ? "out" : "h" + (rI + 1));
      const nin = isOut ? cols : 3 + rI;
      for (let c = 0; c < nin; c++) K.append("rect").attr("x", 34 + c * cw + 1).attr("y", 14 + rI * rh + 1).attr("width", cw - 2).attr("height", rh - 2)
        .attr("fill", isOut ? "rgba(255,180,84,.7)" : "rgba(154,163,178,.45)");
    }
    const frozen = S.units.reduce((s, u) => s + u.w.length, 0);
    RO.innerHTML = `hidden units <b>${nU}</b> (each one layer deeper than the last) · training error <b>${(errRate() * 100).toFixed(1)}%</b> (${Math.round(errRate() * D.length)} of ${D.length} points wrong)` +
      ` · weights: <b>${frozen}</b> frozen + <b>${S.wo.length}</b> trainable output weights` +
      (nU ? ` · last unit's correlation score S = ${S.units[nU - 1].S.toFixed(2)} (best of ${NC} candidates)` : "");
  }
  document.getElementById("casc-add").addEventListener("click", () => { stop(); addUnit(); draw(); });
  document.getElementById("casc-grow").addEventListener("click", grow);
  document.getElementById("casc-reset").addEventListener("click", reset);
  dataS.addEventListener("change", reset);
  reset();
})();

/* ═══════════════ 7 · #obd-svg — which weight to prune ═══════════════ */
(function () {
  const svg = d3.select("#obd-svg");
  if (svg.empty()) return;
  const W = 760, H = 360;
  const w1S = document.getElementById("obd-w1"), w2S = document.getElementById("obd-w2"), kS = document.getElementById("obd-k"), thS = document.getElementById("obd-th"),
    RO = document.getElementById("obd-readout");
  function draw() {
    const w1 = +w1S.value, w2 = +w2S.value, k = +kS.value, th = +thS.value * Math.PI / 180;
    ["w1", "w2", "k", "th"].forEach(id => { document.getElementById("obd-" + id + "-out").textContent = (+document.getElementById("obd-" + id).value).toFixed(id === "th" ? 0 : 1); });
    // H = R diag(1, k) Rᵀ
    const c = Math.cos(th), s = Math.sin(th);
    const H = [[c * c + k * s * s, (1 - k) * c * s], [(1 - k) * c * s, s * s + k * c * c]];
    const det = H[0][0] * H[1][1] - H[0][1] * H[1][0], Hi = [[H[1][1] / det, -H[0][1] / det], [-H[1][0] / det, H[0][0] / det]];
    const L = (a, b) => { const d0 = a - w1, d1 = b - w2; return 0.5 * (H[0][0] * d0 * d0 + 2 * H[0][1] * d0 * d1 + H[1][1] * d1 * d1); };
    const w = [w1, w2];
    const mag = [Math.abs(w1), Math.abs(w2)];
    const obd = [0.5 * H[0][0] * w1 * w1, 0.5 * H[1][1] * w2 * w2];
    const obs = [w1 * w1 / (2 * Hi[0][0]), w2 * w2 / (2 * Hi[1][1])];
    // OBS end points: prune q, re-optimise the other weight: δw = −(w_q / [H⁻¹]_qq) H⁻¹ e_q
    const obsPt = q => [w1 - w[q] / Hi[q][q] * Hi[0][q], w2 - w[q] / Hi[q][q] * Hi[1][q]];
    const pick = a => (a[0] <= a[1] ? 0 : 1);
    svg.selectAll("*").remove();
    const P = NE.panel(svg, 40, 20), pw = 320;
    const sc = d3.scaleLinear().domain([-2.6, 2.6]).range([0, pw]), scy = d3.scaleLinear().domain([-2.6, 2.6]).range([pw, 0]);
    const lv = [0.05, 0.2, 0.5, 1, 2, 4, 8];
    const cont = d3.contours().size([60, 60]).thresholds(lv);
    const grid = new Float64Array(3600);
    for (let j = 0; j < 60; j++) for (let i = 0; i < 60; i++) grid[j * 60 + i] = L(-2.6 + 5.2 * (i + 0.5) / 60, 2.6 - 5.2 * (j + 0.5) / 60);
    const gp = d3.geoPath(d3.geoIdentity().scale(pw / 60));
    P.append("g").selectAll("path").data(cont(grid)).enter().append("path").attr("d", gp).attr("fill", "none").attr("stroke", C.line).attr("stroke-width", 1.1);
    P.append("line").attr("x1", sc(0)).attr("x2", sc(0)).attr("y1", 0).attr("y2", pw).attr("stroke", C.muted).attr("stroke-opacity", 0.5);
    P.append("line").attr("x1", 0).attr("x2", pw).attr("y1", scy(0)).attr("y2", scy(0)).attr("stroke", C.muted).attr("stroke-opacity", 0.5);
    P.append("text").attr("x", pw - 2).attr("y", scy(0) - 5).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 10).text("w₁ axis (w₂ = 0)");
    P.append("text").attr("x", sc(0) + 5).attr("y", 12).attr("fill", C.muted).attr("font-size", 10).text("w₂ axis (w₁ = 0)");
    const obdPts = [[0, w2], [w1, 0]];
    obdPts.forEach((pt, q) => {
      P.append("line").attr("x1", sc(w1)).attr("y1", scy(w2)).attr("x2", sc(pt[0])).attr("y2", scy(pt[1])).attr("stroke", C.B).attr("stroke-dasharray", "4,3");
      P.append("circle").attr("cx", sc(pt[0])).attr("cy", scy(pt[1])).attr("r", 5).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2);
      const o = obsPt(q);
      P.append("line").attr("x1", sc(w1)).attr("y1", scy(w2)).attr("x2", sc(o[0])).attr("y2", scy(o[1])).attr("stroke", C.good);
      P.append("rect").attr("x", sc(o[0]) - 4).attr("y", scy(o[1]) - 4).attr("width", 8).attr("height", 8).attr("fill", C.good);
    });
    P.append("circle").attr("cx", sc(w1)).attr("cy", scy(w2)).attr("r", 6).attr("fill", C.A);
    P.append("text").attr("x", sc(w1) + 8).attr("y", scy(w2) - 8).attr("fill", C.A).attr("font-size", 11).text("w*");
    NE.legend(svg, 40, H - 10, [{ t: "prune, keep other (OBD view)", c: C.B, d: "4,3", w: 200 }, { t: "prune + re-optimise other (OBS)", c: C.good, w: 220 }]);
    // table
    const T = NE.panel(svg, 400, 40), names = ["w₁", "w₂"];
    const rowsT = [["criterion", "score for w₁", "score for w₂", "prunes"],
      ["magnitude |w|", mag[0].toFixed(3), mag[1].toFixed(3), names[pick(mag)]],
      ["OBD ½·Hqq·wq²", obd[0].toFixed(3), obd[1].toFixed(3), names[pick(obd)]],
      ["OBS wq² / (2[H⁻¹]qq)", obs[0].toFixed(3), obs[1].toFixed(3), names[pick(obs)]]];
    rowsT.forEach((row, i) => row.forEach((t, j) => T.append("text").attr("x", [0, 150, 240, 320][j]).attr("y", i * 26).attr("font-size", 11.5)
      .attr("fill", i === 0 ? C.muted : j === 3 ? C.B : C.ink).attr("font-weight", i === 0 || j === 3 ? 600 : 400).text(t)));
    const bestQ = pick(obs);
    const cost = q => L(...obsPt(q));
    T.append("text").attr("x", 0).attr("y", 130).attr("font-size", 11.5).attr("fill", C.ink).text("loss after pruning and retraining the survivor:");
    [0, 1].forEach(q => T.append("text").attr("x", 0).attr("y", 152 + q * 20).attr("font-size", 11.5).attr("fill", q === bestQ ? C.good : C.bad)
      .text(`prune ${names[q]}: ΔL = ${cost(q).toFixed(3)}${q === bestQ ? "  ← the right choice" : ""}`));
    const wrongM = pick(mag) !== bestQ, wrongD = pick(obd) !== bestQ;
    RO.innerHTML = `Hessian H = [[${H[0][0].toFixed(2)}, ${H[0][1].toFixed(2)}], [${H[1][0].toFixed(2)}, ${H[1][1].toFixed(2)}]] · off-diagonal coupling ${Math.abs(H[0][1] / Math.sqrt(H[0][0] * H[1][1])).toFixed(2)}<br>` +
      `magnitude pruning ${wrongM ? "<b>picks the wrong weight</b>" : "agrees with OBS"} · OBD ${wrongD ? "<b>picks the wrong weight</b>" : "agrees with OBS"} (with two weights it always does) · ` +
      `for the chosen weight: OBD's ΔL with the survivor held fixed <b>${obd[bestQ].toFixed(3)}</b>, OBS's ΔL with the survivor re-optimised <b>${obs[bestQ].toFixed(3)}</b>`;
  }
  [w1S, w2S, kS, thS].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════════ 8 · #es-svg — the ES gradient estimate ═══════════════ */
(function () {
  const svg = d3.select("#es-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, PW = 330;
  const nS = document.getElementById("es-n"), sS = document.getElementById("es-sigma"), shS = document.getElementById("es-shape"), RO = document.getElementById("es-readout");
  /* maximise F: a bowl toward (1.2, 0.8) with an egg-crate of local optima on top */
  const B = 0.9, K = 4;
  const F = (x, y) => -0.25 * ((x - 1.2) ** 2 + 2 * (y - 0.8) ** 2) + B * Math.cos(K * (x - 1.2)) * Math.cos(K * (y - 0.8));
  const gradF = (x, y) => [-0.5 * (x - 1.2) - B * K * Math.sin(K * (x - 1.2)) * Math.cos(K * (y - 0.8)), -(y - 0.8) - B * K * Math.cos(K * (x - 1.2)) * Math.sin(K * (y - 0.8))];
  /* the Gaussian-smoothed objective E[F(θ + σε)] — ES follows ITS gradient: the bumps shrink by e^(−K²σ²) */
  const gradFs = (x, y, s) => { const d = Math.exp(-K * K * s * s), u = x - 1.2, v = y - 0.8; return [-0.5 * u - d * B * K * Math.sin(K * u) * Math.cos(K * v), -v - d * B * K * Math.cos(K * u) * Math.sin(K * v)]; };
  let th, path, r = NE.rng(2), last;
  function estimate(x, y, n, s, shape, rr) {
    const pairs = Math.max(1, n / 2), eps = [], Fp = [], Fm = [];
    for (let i = 0; i < pairs; i++) { const e = [NE.gauss(rr), NE.gauss(rr)]; eps.push(e); Fp.push(F(x + s * e[0], y + s * e[1])); Fm.push(F(x - s * e[0], y - s * e[1])); }
    let wp = Fp, wm = Fm;
    if (shape === "rank") {                                  // centred ranks in [−0.5, 0.5]
      const all = Fp.concat(Fm), idx = all.map((v, i) => i).sort((a, b) => all[a] - all[b]), rk = new Array(all.length);
      idx.forEach((id, k) => { rk[id] = all.length > 1 ? k / (all.length - 1) - 0.5 : 0; });
      wp = rk.slice(0, pairs); wm = rk.slice(pairs);
    }
    const g = [0, 0];
    for (let i = 0; i < pairs; i++) { const d = wp[i] - wm[i]; g[0] += d * eps[i][0]; g[1] += d * eps[i][1]; }
    g[0] /= 2 * pairs * s; g[1] /= 2 * pairs * s;
    return { g, eps, Fp, Fm };
  }
  const cos = (a, b) => (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b) + 1e-12);
  function reset() { th = [-2.2, -2.0]; path = [th.slice()]; r = NE.rng(2); last = estimate(th[0], th[1], +nS.value, +sS.value, shS.value, r); draw(); }
  function step(k) {
    for (let i = 0; i < k; i++) {
      last = estimate(th[0], th[1], +nS.value, +sS.value, shS.value, r);
      const nrm = Math.hypot(...last.g) || 1;
      th = [Math.max(-2.9, Math.min(2.9, th[0] + 0.15 * last.g[0] / nrm)), Math.max(-2.9, Math.min(2.9, th[1] + 0.15 * last.g[1] / nrm))];
      path.push(th.slice());
    }
    last = estimate(th[0], th[1], +nS.value, +sS.value, shS.value, r);
    draw();
  }
  let bg = null;
  function draw() {
    const n = +nS.value, s = +sS.value;
    document.getElementById("es-n-out").textContent = n;
    document.getElementById("es-sigma-out").textContent = s.toFixed(2);
    svg.selectAll("*").remove();
    const P = NE.panel(svg, 30, 20);
    const sc = d3.scaleLinear().domain([-3, 3]).range([0, PW]), scy = d3.scaleLinear().domain([-3, 3]).range([PW, 0]);
    const RESN = 50, cs = PW / RESN, col = d3.scaleSequential(d3.interpolateMagma).domain([-4, 1.2]);
    const G = P.append("g");
    for (let i = 0; i < RESN; i++) for (let j = 0; j < RESN; j++) {
      const x = -3 + 6 * (i + 0.5) / RESN, y = 3 - 6 * (j + 0.5) / RESN;
      G.append("rect").attr("x", i * cs).attr("y", j * cs).attr("width", cs + 0.3).attr("height", cs + 0.3).attr("fill", col(F(x, y))).attr("opacity", 0.8);
    }
    P.append("circle").attr("cx", sc(1.2)).attr("cy", scy(0.8)).attr("r", 5).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 2);
    P.append("path").attr("d", d3.line()(path.map(p => [sc(p[0]), scy(p[1])]))).attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 1.4).attr("opacity", 0.8);
    // the samples, coloured by fitness
    const allF = last.Fp.concat(last.Fm), fr = d3.extent(allF);
    const fc = v => d3.interpolateRgb("#f87171", "#4ade80")((v - fr[0]) / ((fr[1] - fr[0]) || 1));
    last.eps.forEach((e, i) => {
      [[1, last.Fp[i]], [-1, last.Fm[i]]].filter(([sg]) => Math.abs(th[0] + sg * s * e[0]) <= 3 && Math.abs(th[1] + sg * s * e[1]) <= 3).forEach(([sg, v]) => P.append("circle").attr("cx", sc(th[0] + sg * s * e[0])).attr("cy", scy(th[1] + sg * s * e[1])).attr("r", 3).attr("fill", fc(v)).attr("stroke", "#000").attr("stroke-width", 0.5));
    });
    P.append("circle").attr("cx", sc(th[0])).attr("cy", scy(th[1])).attr("r", s * PW / 6).attr("fill", "none").attr("stroke", "#fff").attr("stroke-dasharray", "3,3").attr("opacity", 0.6);
    const arrow = (g, color, len) => {
      const nrm = Math.hypot(...g) || 1, L = len || 45;
      P.append("line").attr("x1", sc(th[0])).attr("y1", scy(th[1])).attr("x2", sc(th[0]) + g[0] / nrm * L).attr("y2", scy(th[1]) - g[1] / nrm * L)
        .attr("stroke", color).attr("stroke-width", 3).attr("stroke-linecap", "round");
      P.append("circle").attr("cx", sc(th[0]) + g[0] / nrm * L).attr("cy", scy(th[1]) - g[1] / nrm * L).attr("r", 3.5).attr("fill", color);
    };
    const gt = gradF(th[0], th[1]), gs = gradFs(th[0], th[1], s);
    arrow(gt, "#e6e9ef", 40); arrow(gs, C.good, 50); arrow(last.g, C.B, 60);
    P.append("circle").attr("cx", sc(th[0])).attr("cy", scy(th[1])).attr("r", 4).attr("fill", "#fff");
    NE.legend(svg, 390, 30, [{ t: "ES estimate (from samples only)", c: C.B, w: 360 }]);
    NE.legend(svg, 390, 50, [{ t: "gradient of the σ-smoothed objective", c: C.good, w: 360 }]);
    NE.legend(svg, 390, 70, [{ t: "true local gradient ∇F", c: "#e6e9ef", w: 360 }]);
    // Monte-Carlo quality of the estimator at this θ
    const rr = NE.rng(77); let cS = 0, cT = 0; const REP = 150;
    for (let i = 0; i < REP; i++) { const e = estimate(th[0], th[1], n, s, shS.value, rr).g; cS += cos(e, gs); cT += cos(e, gt); }
    cS /= REP; cT /= REP;
    const Q = NE.panel(svg, 400, 110), qw = 320, qh = 110;
    Q.append("text").attr("x", 0).attr("y", -8).attr("fill", C.muted).attr("font-size", 10.5).text(`mean cosine of the estimate with each gradient (${REP} redraws)`);
    [["smoothed ∇F_σ", cS, C.good], ["true ∇F", cT, "#e6e9ef"]].forEach(([t, v, cc], i) => {
      Q.append("text").attr("x", 0).attr("y", 20 + i * 34).attr("fill", C.ink).attr("font-size", 11).text(t);
      Q.append("rect").attr("x", 110).attr("y", 8 + i * 34).attr("width", qw - 110).attr("height", 16).attr("fill", "#1e222d");
      const x0 = 110 + (qw - 110) / 2;
      Q.append("rect").attr("x", Math.min(x0, x0 + v * (qw - 110) / 2)).attr("y", 8 + i * 34).attr("width", Math.abs(v) * (qw - 110) / 2).attr("height", 16).attr("fill", cc);
      Q.append("line").attr("x1", x0).attr("x2", x0).attr("y1", 4 + i * 34).attr("y2", 28 + i * 34).attr("stroke", C.muted);
      Q.append("text").attr("x", qw + 6).attr("y", 20 + i * 34).attr("fill", C.ink).attr("font-size", 11).text(v.toFixed(2));
    });
    Q.append("text").attr("x", 110).attr("y", 84).attr("fill", C.muted).attr("font-size", 9.5).text("−1");
    Q.append("text").attr("x", qw).attr("y", 84).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 9.5).text("+1");
    const fv = F(th[0], th[1]);
    RO.innerHTML = `θ = (${th[0].toFixed(2)}, ${th[1].toFixed(2)}) · F(θ) = <b>${fv.toFixed(3)}</b> (global max ${F(1.2, 0.8).toFixed(3)} at (1.2, 0.8)) · steps taken <b>${path.length - 1}</b><br>` +
      `population n = ${n} (${n / 2} antithetic pairs) · σ = ${s.toFixed(2)} · ${shS.value === "rank" ? "centred-rank" : "raw"} fitness · bumps in the smoothed objective scaled by e^(−K²σ²) = <b>${Math.exp(-K * K * s * s).toFixed(3)}</b><br>` +
      `estimate vs smoothed gradient: mean cosine <b>${cS.toFixed(2)}</b> · vs true local gradient: <b>${cT.toFixed(2)}</b>`;
  }
  document.getElementById("es-step").addEventListener("click", () => step(1));
  document.getElementById("es-run").addEventListener("click", () => step(25));
  document.getElementById("es-reset").addEventListener("click", reset);
  [nS, sS].forEach(e => e.addEventListener("input", () => { last = estimate(th[0], th[1], +nS.value, +sS.value, shS.value, r); draw(); }));
  shS.addEventListener("change", () => { last = estimate(th[0], th[1], +nS.value, +sS.value, shS.value, r); draw(); });
  reset();
})();
