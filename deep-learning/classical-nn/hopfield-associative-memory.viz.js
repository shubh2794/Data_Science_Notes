/* hopfield-associative-memory.viz.js — the visualizations on
   deep-learning/classical-nn/hopfield-associative-memory.html
   (Classical Neural Networks · part 2). Loaded after ../../data.js → ../../notes.js.
   Each block is an IIFE that exits quietly if its <svg> is not on the page.

     1  #hr-svg    draw, corrupt and recall a 10×10 pixel pattern
     2  #et-svg    the energy trace: asynchronous against synchronous updates
     3  #at-svg    attractor census of a 16-unit network (all 65 536 states)
     4  #cap-svg   capacity: bit-error rate and retrieval quality against P/N
     5  #bam-svg   a bidirectional associative memory, ping-ponging to rest
     6  #ch-svg    the continuous (graded) Hopfield net: energy surface and gain
     7  #mh-svg    the modern Hopfield update is softmax attention

   Every number the figures print is recomputed from the state they draw.   */

/* ══════════ page-local helpers ══════════ */
const HF = (function () {
  const ON = C.A, OFF = "#1b1f29";
  const mono = "SF Mono, Menlo, monospace";

  function rng(seed) {                       /* mulberry32 */
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const fmt = (x, d) => (d === undefined ? String(x) : x.toFixed(d)).replace("-", "−");

  /* 10 × 10 glyphs, '#' = +1 (on), '.' = −1 (off) */
  const GL = {
    A: ["....##....", "...####...", "..##..##..", ".##....##.", ".##....##.", ".########.", ".########.", ".##....##.", ".##....##.", ".##....##."],
    T: ["##########", "##########", "....##....", "....##....", "....##....", "....##....", "....##....", "....##....", "....##....", "....##...."],
    X: ["##......##", ".##....##.", "..##..##..", "...####...", "....##....", "....##....", "...####...", "..##..##..", ".##....##.", "##......##"],
    O: ["..######..", ".########.", "##......##", "##......##", "##......##", "##......##", "##......##", "##......##", ".########.", "..######.."],
    "♥": [".##....##.", "####..####", "##########", "##########", "##########", ".########.", "..######..", "...####...", "....##....", ".........."],
    "+": ["....##....", "....##....", "....##....", "....##....", "##########", "##########", "....##....", "....##....", "....##....", "....##...."],
    L: ["##........", "##........", "##........", "##........", "##........", "##........", "##........", "##........", "##########", "##########"],
    "▦": ["##..##..##", "##..##..##", "..##..##..", "..##..##..", "##..##..##", "##..##..##", "..##..##..", "..##..##..", "##..##..##", "##..##..##"]
  };
  const NAMES = Object.keys(GL);
  const glyph = k => GL[k].join("").split("").map(c => c === "#" ? 1 : -1);

  /* Hebbian outer-product rule, 1/N scaling, zero diagonal. W is row-major N×N. */
  function hebb(ps, N) {
    const W = new Float64Array(N * N);
    for (const p of ps) for (let i = 0; i < N; i++) {
      const pi = p[i] / N, row = i * N;
      for (let j = 0; j < N; j++) W[row + j] += pi * p[j];
    }
    for (let i = 0; i < N; i++) W[i * N + i] = 0;
    return W;
  }
  /* invert a small dense matrix (Gauss–Jordan with partial pivoting); null if singular */
  function inv(A, n) {
    const M = A.map(r => r.slice()), I = A.map((r, i) => r.map((_, j) => i === j ? 1 : 0));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-10) return null;
      [M[c], M[p]] = [M[p], M[c]]; [I[c], I[p]] = [I[p], I[c]];
      const d = M[c][c];
      for (let j = 0; j < n; j++) { M[c][j] /= d; I[c][j] /= d; }
      for (let r = 0; r < n; r++) if (r !== c) {
        const f = M[r][c]; if (f === 0) continue;
        for (let j = 0; j < n; j++) { M[r][j] -= f * M[c][j]; I[r][j] -= f * I[c][j]; }
      }
    }
    return I;
  }
  /* pseudo-inverse (projection) rule W = Ξ (ΞᵀΞ)⁻¹ Ξᵀ, then zero the diagonal */
  function pinv(ps, N) {
    const P = ps.length, Q = [];
    for (let a = 0; a < P; a++) { Q.push([]); for (let b = 0; b < P; b++) { let s = 0; for (let i = 0; i < N; i++) s += ps[a][i] * ps[b][i]; Q[a].push(s); } }
    const Qi = inv(Q, P);
    if (!Qi) return hebb(ps, N);
    /* B = Ξ Q⁻¹  (N × P) */
    const B = [];
    for (let i = 0; i < N; i++) { const r = new Float64Array(P); for (let b = 0; b < P; b++) { let s = 0; for (let a = 0; a < P; a++) s += ps[a][i] * Qi[a][b]; r[b] = s; } B.push(r); }
    const W = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (i === j) continue; let s = 0; for (let b = 0; b < P; b++) s += B[i][b] * ps[b][j]; W[i * N + j] = s;
    }
    return W;
  }
  function fields(W, s, N) {
    const h = new Float64Array(N);
    for (let i = 0; i < N; i++) { let t = 0; const row = i * N; for (let j = 0; j < N; j++) t += W[row + j] * s[j]; h[i] = t; }
    return h;
  }
  function energy(W, s, N) {
    let E = 0;
    for (let i = 0; i < N; i++) { let t = 0; const row = i * N; for (let j = 0; j < N; j++) t += W[row + j] * s[j]; E += s[i] * t; }
    return -0.5 * E;
  }
  /* async relaxation with incremental fields; order = "seq" or an rng for random order */
  function relax(W, s0, N, order, maxSweeps) {
    const s = s0.slice(), h = fields(W, s, N); let flips = 0;
    const idx = [...Array(N).keys()];
    for (let sw = 0; sw < (maxSweeps || 100); sw++) {
      if (typeof order === "function") for (let k = N - 1; k > 0; k--) { const j = Math.floor(order() * (k + 1)); [idx[k], idx[j]] = [idx[j], idx[k]]; }
      let ch = 0;
      for (let q = 0; q < N; q++) {
        const i = idx[q], n = h[i] > 0 ? 1 : h[i] < 0 ? -1 : (s[i] || 1);
        if (n !== s[i]) {
          const d = n - s[i]; s[i] = n; ch++;
          for (let j = 0; j < N; j++) h[j] += W[j * N + i] * d;
        }
      }
      flips += ch; if (!ch) return { s, flips, sweeps: sw + 1, conv: true };
    }
    return { s, flips, sweeps: maxSweeps || 100, conv: false };
  }
  const overlap = (a, b) => { let t = 0; for (let i = 0; i < a.length; i++) t += a[i] * b[i]; return t / a.length; };
  const same = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };
  function erfc(x) {   /* Abramowitz–Stegun 7.1.26, |error| < 1.5e−7 */
    const z = Math.abs(x), t = 1 / (1 + 0.3275911 * z);
    const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z);
    return x >= 0 ? y : 2 - y;
  }
  /* classify a state against a set of stored patterns */
  function classify(s, ps) {
    for (let k = 0; k < ps.length; k++) {
      if (same(s, ps[k])) return { cls: "stored", k };
      if (same(s, ps[k].map(v => -v))) return { cls: "reversed", k };
    }
    const P = ps.length, N = s.length;
    for (let a = 0; a < P; a++) for (let b = a + 1; b < P; b++) for (let c = b + 1; c < P; c++)
      for (let sg = 0; sg < 8; sg++) {
        const sa = sg & 1 ? -1 : 1, sb = sg & 2 ? -1 : 1, sc = sg & 4 ? -1 : 1;
        let ok = true;
        for (let i = 0; i < N && ok; i++) { const m = sa * ps[a][i] + sb * ps[b][i] + sc * ps[c][i]; if ((m > 0 ? 1 : -1) !== s[i]) ok = false; }
        if (ok) return { cls: "mixture", k: [a, b, c] };
      }
    return { cls: "other" };
  }
  const CLS = { stored: C.good, reversed: C.B, mixture: "#c084fc", other: C.bad };

  /* draw an image of N = side² values in [−1, 1] */
  function img(g, vals, x, y, cell, opt) {
    opt = opt || {};
    const side = Math.round(Math.sqrt(vals.length)), col = d3.interpolateRgb(OFF, opt.on || ON);
    const G = g.append("g").attr("transform", `translate(${x},${y})`);
    const r = G.selectAll("rect").data(vals).join("rect")
      .attr("x", (d, i) => (i % side) * cell).attr("y", (d, i) => Math.floor(i / side) * cell)
      .attr("width", cell - (opt.gap === undefined ? 1 : opt.gap)).attr("height", cell - (opt.gap === undefined ? 1 : opt.gap))
      .attr("fill", d => col((Math.max(-1, Math.min(1, d)) + 1) / 2));
    if (opt.frame) G.append("rect").attr("x", -2).attr("y", -2).attr("width", side * cell + 3).attr("height", side * cell + 3)
      .attr("fill", "none").attr("stroke", opt.frame).attr("stroke-width", opt.fw || 1.5).attr("rx", 3);
    return { G, rects: r, col };
  }
  function txt(g, x, y, s, o) {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11).attr("fill", o.fill || C.ink)
      .attr("font-weight", o.bold ? 600 : null).attr("text-anchor", o.anchor || null)
      .attr("font-family", o.mono ? mono : null).text(s);
  }
  function randPats(P, N, r, bias) {
    const ps = []; const q = 0.5 + (bias || 0);
    for (let k = 0; k < P; k++) { const p = new Array(N); for (let i = 0; i < N; i++) p[i] = r() < q ? 1 : -1; ps.push(p); }
    return ps;
  }
  return { ON, OFF, mono, rng, randn, fmt, GL, NAMES, glyph, hebb, pinv, inv, fields, energy, relax, overlap, same, erfc, classify, CLS, img, txt, randPats };
})();

/* ═════════════════ 1 · #hr-svg — draw, corrupt, recall ═════════════════ */
(function () {
  const svg = d3.select("#hr-svg");
  if (svg.empty()) return;
  const N = 100, CELL = 27;
  const $ = id => document.getElementById(id);
  const loadE = $("hr-load"), noiseE = $("hr-noise"), ruleE = $("hr-rule");
  let stored = [true, true, true, true, false, false, false, false];
  let s = HF.glyph("A"), W = null, pats = [], trace = [], timer = null, updates = 0, flips = 0, cursor = -1, quiet = 0;
  let rnd = HF.rng(7), painting = false, paintVal = 1;

  function rebuild() {
    pats = HF.NAMES.filter((k, i) => stored[i]).map(HF.glyph);
    W = pats.length ? (ruleE.value === "pinv" ? HF.pinv(pats, N) : HF.hebb(pats, N)) : new Float64Array(N * N);
  }
  function resetTrace() { trace = [HF.energy(W, s, N)]; updates = 0; flips = 0; quiet = 0; cursor = -1; }
  function stop() { if (timer) { timer.stop(); timer = null; } $("hr-run").textContent = "Recall ▶"; }

  svg.selectAll("*").remove();
  const gState = svg.append("g"), gThumb = svg.append("g"), gPlot = svg.append("g");
  HF.txt(svg, 20, 22, "network state s (click or drag to paint)", { bold: true });
  HF.txt(svg, 335, 22, "patterns — click to store / unstore;  m = overlap with s", { bold: true });

  /* state grid */
  const cells = gState.selectAll("rect").data(d3.range(N)).join("rect")
    .attr("x", i => 20 + (i % 10) * CELL).attr("y", i => 34 + Math.floor(i / 10) * CELL)
    .attr("width", CELL - 2).attr("height", CELL - 2).attr("rx", 2).style("cursor", "crosshair")
    .on("pointerdown", function (ev, i) { ev.preventDefault(); stop(); painting = true; paintVal = -s[i]; s[i] = paintVal; resetTrace(); draw(); })
    .on("pointerenter", function (ev, i) { if (painting && s[i] !== paintVal) { s[i] = paintVal; resetTrace(); draw(); } });
  window.addEventListener("pointerup", () => { painting = false; });
  const hi = gState.append("rect").attr("width", CELL).attr("height", CELL).attr("fill", "none")
    .attr("stroke", C.B).attr("stroke-width", 2).attr("rx", 3).attr("opacity", 0).attr("pointer-events", "none");

  function draw() {
    const col = d3.interpolateRgb(HF.OFF, HF.ON);
    cells.attr("fill", i => col((s[i] + 1) / 2));
    if (cursor >= 0) hi.attr("x", 20 + (cursor % 10) * CELL - 1).attr("y", 34 + Math.floor(cursor / 10) * CELL - 1).attr("opacity", 1);
    else hi.attr("opacity", 0);

    /* thumbnails */
    gThumb.selectAll("*").remove();
    HF.NAMES.forEach((k, i) => {
      const x = 335 + (i % 4) * 100, y = 36 + Math.floor(i / 4) * 100, p = HF.glyph(k), m = HF.overlap(s, p);
      const im = HF.img(gThumb, p, x, y, 6, { gap: 0, frame: stored[i] ? C.good : C.line, fw: stored[i] ? 2 : 1 });
      im.G.style("cursor", "pointer").style("opacity", stored[i] ? 1 : 0.45)
        .on("click", () => { stop(); stored[i] = !stored[i]; rebuild(); resetTrace(); draw(); });
      HF.txt(gThumb, x + 66, y + 14, k, { size: 14, bold: true, fill: stored[i] ? C.ink : C.muted });
      HF.txt(gThumb, x + 66, y + 32, "m", { size: 10, fill: C.muted });
      HF.txt(gThumb, x + 66, y + 46, HF.fmt(m, 2), { size: 11, mono: true, fill: Math.abs(m) > 0.99 ? (m > 0 ? C.good : C.B) : C.ink });
      HF.txt(gThumb, x + 66, y + 58, stored[i] ? "stored" : "—", { size: 9, fill: stored[i] ? C.good : C.muted });
    });

    /* energy trace */
    gPlot.selectAll("*").remove();
    const x0 = 335, y0 = 250, w = 400, h = 100;
    HF.txt(gPlot, x0, y0 - 12, "energy E(s) after every single-unit update", { bold: true });
    const xs = d3.scaleLinear().domain([0, Math.max(N, trace.length - 1)]).range([x0, x0 + w]);
    const lo = Math.min(-N / 2 - 5, d3.min(trace)), hi2 = Math.max(0, d3.max(trace));
    const ys = d3.scaleLinear().domain([lo, hi2 + 2]).range([y0 + h, y0]);
    gPlot.append("g").attr("class", "axis").attr("transform", `translate(0,${y0 + h})`).call(d3.axisBottom(xs).ticks(5));
    gPlot.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(ys).ticks(4));
    gPlot.append("path").datum(trace).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.8)
      .attr("d", d3.line().x((d, i) => xs(i)).y(d => ys(d)).curve(d3.curveStepAfter));
    HF.txt(gPlot, x0 + w, y0 + h + 28, "unit updates (100 = one sweep)", { size: 10, fill: C.muted, anchor: "end" });

    const E = trace[trace.length - 1];
    let best = -1, bm = -2;
    pats.forEach((p, k) => { const m = Math.abs(HF.overlap(s, p)); if (m > bm) { bm = m; best = k; } });
    const h0 = HF.fields(W, s, N);
    let fixed = true; for (let i = 0; i < N; i++) { const n = h0[i] > 0 ? 1 : h0[i] < 0 ? -1 : s[i]; if (n !== s[i]) { fixed = false; break; } }
    let verdict = "still moving";
    if (fixed && pats.length) {
      const c = HF.classify(s, pats), nm = HF.NAMES.filter((k, i) => stored[i]);
      verdict = c.cls === "stored" ? `fixed point = stored pattern <b>${nm[c.k]}</b>` :
        c.cls === "reversed" ? `fixed point = <b>reversed ${nm[c.k]}</b> (−ξ)` :
          c.cls === "mixture" ? `fixed point = <b>3-mixture</b> sgn(±${nm[c.k[0]]}±${nm[c.k[1]]}±${nm[c.k[2]]})` :
            `fixed point, but <b>spurious</b> (not a stored pattern)`;
    }
    $("hr-readout").innerHTML = `E = <b>${HF.fmt(E, 2)}</b> · P = ${pats.length}, α = P/N = ${HF.fmt(pats.length / N, 2)} · updates ${updates}, flips ${flips} · ${verdict}`;
  }

  function tick() {
    /* 3 random-order single-unit updates per frame */
    for (let q = 0; q < 3; q++) {
      const i = Math.floor(rnd() * N);
      let h = 0; for (let j = 0; j < N; j++) h += W[i * N + j] * s[j];
      const n = h > 0 ? 1 : h < 0 ? -1 : s[i];
      updates++; cursor = i;
      if (n !== s[i]) { s[i] = n; flips++; quiet = 0; } else quiet++;
      trace.push(HF.energy(W, s, N));
    }
    draw();
    if (quiet >= 4 * N) {
      /* a long run of no-flips: check for a true fixed point */
      const h0 = HF.fields(W, s, N);
      let fixed = true; for (let i = 0; i < N; i++) if ((h0[i] > 0 ? 1 : h0[i] < 0 ? -1 : s[i]) !== s[i]) fixed = false;
      if (fixed) { stop(); cursor = -1; draw(); }
    }
  }
  function sweep() {
    stop();
    const idx = d3.shuffle(d3.range(N), rnd);
    for (const i of idx) {
      let h = 0; for (let j = 0; j < N; j++) h += W[i * N + j] * s[j];
      const n = h > 0 ? 1 : h < 0 ? -1 : s[i];
      updates++; if (n !== s[i]) { s[i] = n; flips++; }
      trace.push(HF.energy(W, s, N));
    }
    cursor = -1; draw();
  }
  function corrupt() {
    stop(); const f = +noiseE.value / 100;
    for (let i = 0; i < N; i++) if (rnd() < f) s[i] = -s[i];
    resetTrace(); draw();
  }

  loadE.addEventListener("change", () => { stop(); s = HF.glyph(loadE.value); resetTrace(); draw(); });
  noiseE.addEventListener("input", () => { $("hr-noisev").textContent = noiseE.value + "%"; });
  ruleE.addEventListener("change", () => { stop(); rebuild(); resetTrace(); draw(); });
  $("hr-corrupt").addEventListener("click", corrupt);
  $("hr-mask").addEventListener("click", () => { stop(); for (let i = 50; i < N; i++) s[i] = -1; resetTrace(); draw(); });
  $("hr-invert").addEventListener("click", () => { stop(); s = s.map(v => -v); resetTrace(); draw(); });
  $("hr-mix").addEventListener("click", () => {
    stop(); if (pats.length < 3) { $("hr-readout").innerHTML = "store at least three patterns to build a 3-mixture"; return; }
    s = pats[0].map((v, i) => (v + pats[1][i] + pats[2][i]) > 0 ? 1 : -1); resetTrace(); draw();
  });
  $("hr-sweep").addEventListener("click", sweep);
  $("hr-run").addEventListener("click", () => {
    if (timer) { stop(); return; }
    $("hr-run").textContent = "Pause ❚❚"; quiet = 0; timer = d3.interval(tick, 30);
  });

  rebuild(); s = HF.glyph("A");
  for (let i = 0; i < N; i++) if (rnd() < 0.25) s[i] = -s[i];
  resetTrace(); draw();
})();

/* ═════════════════ 2 · #et-svg — async against sync ═════════════════ */
(function () {
  const svg = d3.select("#et-svg");
  if (svg.empty()) return;
  const N = 100, $ = id => document.getElementById(id);
  const PE = $("et-P"), nE = $("et-noise"), stE = $("et-start");
  let seed = 11;

  function syncCycles(W, s0) {
    let s = s0.slice(), prev = null;
    for (let t = 0; t < 40; t++) {
      const h = HF.fields(W, s, N), n = s.map((v, i) => h[i] > 0 ? 1 : h[i] < 0 ? -1 : v);
      if (HF.same(n, s)) return false;
      if (prev && HF.same(n, prev)) return true;
      prev = s; s = n;
    }
    return false;
  }
  function setup(sd) {
    const P = +PE.value, noise = +nE.value / 100;
    const r = HF.rng(sd), ps = HF.randPats(P, N, r), W = HF.hebb(ps, N);
    const s0 = stE.value === "cue" ? ps[0].map(v => r() < noise ? -v : v) : HF.randPats(1, N, r)[0];
    return { ps, W, s0 };
  }
  function run() {
    const P = +PE.value;
    $("et-Pv").textContent = P; $("et-noisev").textContent = nE.value + "%";
    let st = setup(seed), note = "";
    if (stE.value === "cycle") {
      let k = 0;
      while (k < 300 && !syncCycles(st.W, st.s0)) { k++; st = setup(seed + k); }
      note = k >= 300 ? "no 2-cycle found in 300 draws at this load — raise P · " : `found a 2-cycle on draw ${k + 1} · `;
    }
    const { ps, W, s0 } = st;

    /* asynchronous, random order: record every update */
    const A = { E: [HF.energy(W, s0, N)], m: [HF.overlap(s0, ps[0])] };
    { const s = s0.slice(), h = HF.fields(W, s, N), rr = HF.rng(seed + 99); let quiet = 0, t = 0;
      while (quiet < N && t < 40 * N) {
        const idx = d3.shuffle(d3.range(N), rr); let ch = 0;
        for (const i of idx) {
          const n = h[i] > 0 ? 1 : h[i] < 0 ? -1 : s[i];
          if (n !== s[i]) { const d = n - s[i]; s[i] = n; ch++; for (let j = 0; j < N; j++) h[j] += W[j * N + i] * d; }
          t++; A.E.push(-0.5 * s.reduce((a, v, j) => a + v * h[j], 0)); A.m.push(HF.overlap(s, ps[0]));
        }
        quiet = ch ? 0 : N;
      }
      A.final = s; }
    /* synchronous: all units at once */
    const S = { E: [HF.energy(W, s0, N)], m: [HF.overlap(s0, ps[0])], cycle: false, fixed: false, ups: 0 };
    { let s = s0.slice(), prev = null;
      for (let t = 0; t < 40; t++) {
        const h = HF.fields(W, s, N), n = s.map((v, i) => h[i] > 0 ? 1 : h[i] < 0 ? -1 : v);
        if (HF.same(n, s)) { S.fixed = true; break; }
        if (prev && HF.same(n, prev)) { S.cycle = true; S.E.push(HF.energy(W, n, N)); S.m.push(HF.overlap(n, ps[0])); prev = s; s = n; break; }
        prev = s; s = n;
        const e = HF.energy(W, s, N); if (e > S.E[S.E.length - 1] + 1e-9) S.ups++;
        S.E.push(e); S.m.push(HF.overlap(s, ps[0]));
      }
      S.final = s; }

    svg.selectAll("*").remove();
    const tmax = Math.max((A.E.length - 1) / N, S.E.length - 1, 1);
    const panels = [{ x: 50, key: "E", title: "energy E(s) / N", dom: null }, { x: 430, key: "m", title: "overlap m with the target ξ¹", dom: [-1, 1] }];
    panels.forEach(pn => {
      const w = 300, y0 = 36, h = 220;
      const xs = d3.scaleLinear().domain([0, tmax]).range([pn.x, pn.x + w]);
      const all = A[pn.key].concat(S[pn.key]).map(v => pn.key === "E" ? v / N : v);
      const dom = pn.dom || [Math.min(-0.55, d3.min(all)) - 0.02, Math.max(0, d3.max(all)) + 0.02];
      const ys = d3.scaleLinear().domain(dom).range([y0 + h, y0]);
      HF.txt(svg, pn.x, 22, pn.title, { bold: true });
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y0 + h})`).call(d3.axisBottom(xs).ticks(6));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${pn.x},0)`).call(d3.axisLeft(ys).ticks(5));
      HF.txt(svg, pn.x + w, y0 + h + 30, "time in sweeps (1 sweep = N async updates = 1 sync step)", { size: 10, fill: C.muted, anchor: "end" });
      const av = A[pn.key].map((v, i) => [i / N, pn.key === "E" ? v / N : v]);
      svg.append("path").datum(av).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.8)
        .attr("d", d3.line().x(d => xs(d[0])).y(d => ys(d[1])));
      const sv = S[pn.key].map((v, i) => [i, pn.key === "E" ? v / N : v]);
      svg.append("path").datum(sv).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3")
        .attr("d", d3.line().x(d => xs(d[0])).y(d => ys(d[1])));
      svg.append("g").selectAll("circle").data(sv).join("circle").attr("cx", d => xs(d[0])).attr("cy", d => ys(d[1])).attr("r", 3).attr("fill", C.B);
    });
    const lg = svg.append("g").attr("transform", "translate(50,300)");
    lg.append("line").attr("x1", 0).attr("x2", 22).attr("y1", 0).attr("y2", 0).attr("stroke", C.A).attr("stroke-width", 2);
    HF.txt(lg, 28, 4, "asynchronous (one random unit at a time)", { size: 11 });
    lg.append("line").attr("x1", 300).attr("x2", 322).attr("y1", 0).attr("y2", 0).attr("stroke", C.B).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
    HF.txt(lg, 328, 4, "synchronous (every unit at once, dots = steps)", { size: 11 });

    const aF = HF.overlap(A.final, ps[0]), sF = HF.overlap(S.final, ps[0]);
    $("et-readout").innerHTML = note + `async: fixed point after ${HF.fmt((A.E.length - 1) / N, 1)} sweeps, final m = <b>${HF.fmt(aF, 2)}</b>, E/N = ${HF.fmt(A.E[A.E.length - 1] / N, 3)} (never rose) · ` +
      `sync: ${S.fixed ? "fixed point" : S.cycle ? "<b>2-cycle</b> (state flips between two configurations forever)" : "no convergence in 40 steps"} after ${S.E.length - 1} steps, final m = <b>${HF.fmt(sF, 2)}</b>, energy rose on ${S.ups} step${S.ups === 1 ? "" : "s"}`;
  }
  [PE, nE].forEach(e => e.addEventListener("input", run));
  stE.addEventListener("change", run);
  $("et-new").addEventListener("click", () => { seed = (seed * 7 + 13) % 100003; run(); });
  run();
})();

/* ═════════════════ 3 · #at-svg — attractor census ═════════════════ */
(function () {
  const svg = d3.select("#at-svg");
  if (svg.empty()) return;
  const N = 16, S = 1 << N, $ = id => document.getElementById(id);
  const PE = $("at-P"), pathE = $("at-path");
  let seed = 5;

  function census(W, ps) {
    const basin = new Map(); const s = new Array(N), h = new Float64Array(N);
    for (let x = 0; x < S; x++) {
      for (let i = 0; i < N; i++) s[i] = (x >> i) & 1 ? 1 : -1;
      for (let i = 0; i < N; i++) { let t = 0; for (let j = 0; j < N; j++) t += W[i * N + j] * s[j]; h[i] = t; }
      for (let sw = 0; sw < 60; sw++) {
        let ch = 0;
        for (let i = 0; i < N; i++) {
          const n = h[i] > 1e-12 ? 1 : h[i] < -1e-12 ? -1 : s[i];
          if (n !== s[i]) { const d = n - s[i]; s[i] = n; ch++; for (let j = 0; j < N; j++) h[j] += W[j * N + i] * d; }
        }
        if (!ch) break;
      }
      let key = 0; for (let i = 0; i < N; i++) if (s[i] > 0) key |= (1 << i);
      basin.set(key, (basin.get(key) || 0) + 1);
    }
    const out = [];
    basin.forEach((cnt, key) => {
      const st = d3.range(N).map(i => (key >> i) & 1 ? 1 : -1);
      out.push({ st, cnt, E: HF.energy(W, st, N), c: HF.classify(st, ps) });
    });
    out.sort((a, b) => b.cnt - a.cnt);
    return out;
  }

  function run() {
    const P = +PE.value; $("at-Pv").textContent = P;
    const r = HF.rng(seed), ps = HF.randPats(P, N, r), W = HF.hebb(ps, N);
    const att = census(W, ps);
    svg.selectAll("*").remove();

    HF.txt(svg, 20, 20, "stored ξ", { bold: true });
    ps.forEach((p, k) => { HF.img(svg, p, 20 + k * 46, 30, 9, { gap: 1 }); HF.txt(svg, 38 + k * 46, 76, "ξ" + "¹²³⁴⁵⁶"[k], { size: 10, fill: C.muted, anchor: "middle" }); });
    HF.txt(svg, 320, 20, "fixed points reached from all 65 536 starts, largest basins first", { bold: true, size: 10.5 });

    const show = att.slice(0, 14), x0 = 20, colW = 51, y0 = 100;
    const bs = d3.scaleLinear().domain([0, d3.max(show, d => d.cnt / S)]).range([0, 80]);
    const names = ["ξ¹", "ξ²", "ξ³", "ξ⁴", "ξ⁵", "ξ⁶"];
    show.forEach((a, k) => {
      const x = x0 + k * colW, col = HF.CLS[a.c.cls];
      HF.img(svg, a.st, x + 4, y0, 9, { gap: 1, frame: col, fw: 1.5 });
      svg.append("rect").attr("x", x + 8).attr("y", y0 + 130 - bs(a.cnt / S)).attr("width", 28).attr("height", bs(a.cnt / S)).attr("fill", col).attr("fill-opacity", 0.8);
      HF.txt(svg, x + 22, y0 + 144, HF.fmt(100 * a.cnt / S, 1) + "%", { size: 9.5, mono: true, anchor: "middle" });
      HF.txt(svg, x + 22, y0 + 157, "E " + HF.fmt(a.E, 2), { size: 9, mono: true, anchor: "middle", fill: C.muted });
      const lab = a.c.cls === "stored" ? names[a.c.k] : a.c.cls === "reversed" ? "−" + names[a.c.k] : a.c.cls === "mixture" ? "mix" : "other";
      HF.txt(svg, x + 22, y0 + 170, lab, { size: 10, anchor: "middle", fill: col });
    });
    const lg = svg.append("g").attr("transform", `translate(20,${y0 + 190})`);
    [["stored pattern", "stored"], ["reversed −ξ", "reversed"], ["3-mixture", "mixture"], ["other spurious", "other"]].forEach((d, i) => {
      lg.append("rect").attr("x", i * 150).attr("y", -9).attr("width", 11).attr("height", 11).attr("fill", HF.CLS[d[1]]);
      HF.txt(lg, i * 150 + 16, 1, d[0], { size: 11 });
    });

    /* energy along a Hamming path */
    const a = ps[0], b = pathE.value === "rev" || P < 2 ? a.map(v => -v) : ps[1];
    const diff = d3.range(N).filter(i => a[i] !== b[i]);
    const pts = [], st = a.slice(); pts.push(HF.energy(W, st, N));
    diff.forEach(i => { st[i] = b[i]; pts.push(HF.energy(W, st, N)); });
    const px = 60, py = 330, pw = 660, ph = 80;
    HF.txt(svg, 20, py - 16, `energy along a straight Hamming path: start at ξ¹, flip the ${diff.length} differing bits one at a time, end at ${pathE.value === "rev" || P < 2 ? "−ξ¹" : "ξ²"}`, { bold: true, size: 10.5 });
    const xs = d3.scaleLinear().domain([0, Math.max(1, diff.length)]).range([px, px + pw]);
    const ys = d3.scaleLinear().domain([d3.min(pts) - 0.3, d3.max(pts) + 0.3]).range([py + ph, py]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${py + ph})`).call(d3.axisBottom(xs).ticks(Math.min(16, diff.length)));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${px},0)`).call(d3.axisLeft(ys).ticks(4));
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2)
      .attr("d", d3.line().x((d, i) => xs(i)).y(d => ys(d)));
    svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx", (d, i) => xs(i)).attr("cy", d => ys(d)).attr("r", 2.5).attr("fill", C.A);

    const tot = { stored: 0, reversed: 0, mixture: 0, other: 0 };
    att.forEach(a2 => { tot[a2.c.cls] += a2.cnt; });
    const pc = k => HF.fmt(100 * tot[k] / S, 1) + "%";
    $("at-readout").innerHTML = `${att.length} distinct fixed points · basins: stored <b>${pc("stored")}</b>, reversed <b>${pc("reversed")}</b>, 3-mixtures <b>${pc("mixture")}</b>, other spurious <b>${pc("other")}</b> · barrier on the path: ${HF.fmt(d3.max(pts) - pts[0], 2)} above ξ¹`;
  }
  PE.addEventListener("input", run);
  pathE.addEventListener("change", run);
  $("at-new").addEventListener("click", () => { seed = (seed * 31 + 17) % 99991; run(); });
  run();
})();

/* ═════════════════ 4 · #cap-svg — capacity sweep ═════════════════ */
(function () {
  const svg = d3.select("#cap-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const NE = $("cap-N"), ruleE = $("cap-rule"), biasE = $("cap-bias");
  let job = 0;

  function run() {
    const my = ++job, N = +NE.value, rule = ruleE.value, bias = +biasE.value;
    $("cap-biasv").textContent = HF.fmt(bias, 2);
    const amax = rule === "pinv" ? 0.9 : 0.4, steps = 18;
    const Ps = [...new Set(d3.range(1, steps + 1).map(k => Math.max(1, Math.round(k * amax * N / steps))))];
    const res = [];
    const x0 = 60, w = 640, y1 = 30, h1 = 150, y2 = 240, h2 = 130;
    const xs = d3.scaleLinear().domain([0, amax]).range([x0, x0 + w]);
    const ye = d3.scaleLinear().domain([0, 0.2]).range([y1 + h1, y1]).clamp(true);
    const ym = d3.scaleLinear().domain([0, 1]).range([y2 + h2, y2]);

    svg.selectAll("*").remove();
    HF.txt(svg, x0, 18, "one-step bit-error rate at the stored patterns: fraction of bits with ξᵢ·hᵢ ≤ 0", { bold: true });
    HF.txt(svg, x0, y2 - 12, "retrieval: mean overlap m of the fixed point reached — from ξ (solid), from ξ with 10% flipped (dashed)", { bold: true, size: 10.5 });
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y1 + h1})`).call(d3.axisBottom(xs).ticks(8));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(ye).ticks(4, ".2f"));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y2 + h2})`).call(d3.axisBottom(xs).ticks(8));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(ym).ticks(4));
    HF.txt(svg, x0 + w, y2 + h2 + 28, "load α = P / N", { size: 10, fill: C.muted, anchor: "end" });
    [[y1, h1], [y2, h2]].forEach(([y, h]) => {
      svg.append("line").attr("x1", xs(0.138)).attr("x2", xs(0.138)).attr("y1", y).attr("y2", y + h).attr("stroke", C.bad).attr("stroke-dasharray", "4 3");
    });
    HF.txt(svg, xs(0.138) + 4, y1 + 10, "α꜀ ≈ 0.138", { size: 10, fill: C.bad });
    if (rule === "hebb" && bias === 0) {
      const th = d3.range(0.005, amax + 0.001, 0.005).map(a => [a, 0.5 * HF.erfc(Math.sqrt(1 / (2 * a)))]);
      svg.append("path").datum(th).attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 1.5).attr("stroke-dasharray", "2 3")
        .attr("d", d3.line().x(d => xs(d[0])).y(d => ye(d[1])));
      HF.txt(svg, xs(0.2), ye(0.5 * HF.erfc(Math.sqrt(1 / 0.4))) - 34, "dotted: theory ½·erfc(√(N/2P))", { size: 10, fill: C.muted });
    }
    const gE = svg.append("g"), gM = svg.append("g");

    let k = 0;
    function step() {
      if (my !== job || k >= Ps.length) { finish(); return; }
      const P = Ps[k++], trials = N >= 200 ? 1 : 2;
      let err = 0, bits = 0, m0 = 0, m1 = 0, cnt = 0;
      for (let t = 0; t < trials; t++) {
        const r = HF.rng(1000 * P + t * 7 + N), ps = HF.randPats(P, N, r, bias);
        const W = rule === "pinv" ? HF.pinv(ps, N) : HF.hebb(ps, N);
        ps.forEach(p => { const h = HF.fields(W, p, N); for (let i = 0; i < N; i++) { if (p[i] * h[i] <= 0) err++; bits++; } });
        const probe = Math.min(P, 4);
        for (let q = 0; q < probe; q++) {
          const p = ps[q];
          m0 += HF.overlap(HF.relax(W, p, N, r, 30).s, p);
          const noisy = p.map(v => r() < 0.1 ? -v : v);
          m1 += HF.overlap(HF.relax(W, noisy, N, r, 30).s, p); cnt++;
        }
      }
      res.push({ a: P / N, e: err / bits, m0: m0 / cnt, m1: m1 / cnt });
      draw(); d3.timeout(step, 0);
    }
    function draw() {
      gE.selectAll("*").remove(); gM.selectAll("*").remove();
      gE.append("path").datum(res).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2).attr("d", d3.line().x(d => xs(d.a)).y(d => ye(d.e)));
      gE.selectAll("circle").data(res).join("circle").attr("cx", d => xs(d.a)).attr("cy", d => ye(d.e)).attr("r", 3).attr("fill", C.A);
      gM.append("path").datum(res).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 2).attr("d", d3.line().x(d => xs(d.a)).y(d => ym(d.m0)));
      gM.append("path").datum(res).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2).attr("stroke-dasharray", "5 3").attr("d", d3.line().x(d => xs(d.a)).y(d => ym(d.m1)));
      $("cap-readout").innerHTML = `computing… ${res.length}/${Ps.length} loads`;
    }
    function finish() {
      if (my !== job) return;
      const good = res.filter(d => d.m1 > 0.95), last = good.length ? good[good.length - 1] : null;
      $("cap-readout").innerHTML = `N = ${N}, ${rule === "pinv" ? "pseudo-inverse" : "Hebbian"} rule, bit bias ${HF.fmt(bias, 2)} · largest load with noisy-start retrieval m &gt; 0.95: <b>${last ? "α = " + HF.fmt(last.a, 3) + " (P = " + Math.round(last.a * N) + ")" : "none"}</b>` +
        (rule === "hebb" ? ` · theory at α꜀: bit-error ${HF.fmt(0.5 * HF.erfc(Math.sqrt(1 / (2 * 0.138))), 4)}` : "");
    }
    step();
  }
  [NE, ruleE].forEach(e => e.addEventListener("change", run));
  biasE.addEventListener("change", run);
  biasE.addEventListener("input", () => { $("cap-biasv").textContent = HF.fmt(+biasE.value, 2); });
  run();
})();

/* ═════════════════ 5 · #bam-svg — bidirectional associative memory ═════════════════ */
(function () {
  const svg = d3.select("#bam-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const kE = $("bam-k"), nE = $("bam-noise"), sideE = $("bam-side");
  const n = 100, m = 16, names = ["A", "T", "X", "O", "♥", "+"];
  const r0 = HF.rng(4242);
  const codes = names.map(() => d3.range(m).map(() => r0() < 0.5 ? 1 : -1));
  let K, W, x, y, hist, rnd = HF.rng(9), half = 0;

  function build() {
    K = +kE.value; $("bam-kv").textContent = K;
    W = [];                                   /* W is m × n: W = Σ y xᵀ */
    for (let a = 0; a < m; a++) { const row = new Float64Array(n); for (let k = 0; k < K; k++) { const xk = HF.glyph(names[k]); for (let b = 0; b < n; b++) row[b] += codes[k][a] * xk[b]; } W.push(row); }
  }
  const E = () => { let s = 0; for (let a = 0; a < m; a++) { let t = 0; for (let b = 0; b < n; b++) t += W[a][b] * x[b]; s += y[a] * t; } return -s; };
  function start() {
    const f = +nE.value / 100; $("bam-noisev").textContent = nE.value + "%";
    const k = Math.floor(rnd() * K);
    if (sideE.value === "x") { x = HF.glyph(names[k]).map(v => rnd() < f ? -v : v); y = d3.range(m).map(() => rnd() < 0.5 ? 1 : -1); half = 0; }
    else { y = codes[k].map(v => rnd() < f ? -v : v); x = new Array(n).fill(-1); half = 1; }
    hist = [{ E: E(), dir: "start" }]; draw();
  }
  function stepHalf() {
    if (half === 0) { /* x → y */
      y = y.map((v, a) => { let t = 0; for (let b = 0; b < n; b++) t += W[a][b] * x[b]; return t > 0 ? 1 : t < 0 ? -1 : v; });
      hist.push({ E: E(), dir: "x→y" }); half = 1;
    } else {          /* y → x */
      x = x.map((v, b) => { let t = 0; for (let a = 0; a < m; a++) t += W[a][b] * y[a]; return t > 0 ? 1 : t < 0 ? -1 : v; });
      hist.push({ E: E(), dir: "y→x" }); half = 0;
    }
    draw();
  }
  function draw() {
    svg.selectAll("*").remove();
    HF.txt(svg, 20, 20, "layer X (n = 100)", { bold: true });
    HF.img(svg, x, 20, 30, 20, { gap: 1 });
    HF.txt(svg, 470, 20, "layer Y (m = 16 code bits)", { bold: true });
    HF.img(svg, y, 470, 30, 22, { gap: 2, on: C.B });
    /* the arrows */
    const arr = svg.append("g");
    arr.append("line").attr("x1", 240).attr("x2", 455).attr("y1", 80).attr("y2", 80).attr("stroke", half === 1 ? C.B : C.line).attr("stroke-width", 2.5);
    arr.append("path").attr("d", "M455,80 l-9,-5 l0,10 z").attr("fill", half === 1 ? C.B : C.line);
    HF.txt(arr, 347, 72, "y ← sgn(W x)", { anchor: "middle", mono: true, size: 11 });
    arr.append("line").attr("x1", 455).attr("x2", 240).attr("y1", 120).attr("y2", 120).attr("stroke", half === 0 && hist.length > 1 ? C.A : C.line).attr("stroke-width", 2.5);
    arr.append("path").attr("d", "M240,120 l9,-5 l0,10 z").attr("fill", half === 0 && hist.length > 1 ? C.A : C.line);
    HF.txt(arr, 347, 140, "x ← sgn(Wᵀ y)", { anchor: "middle", mono: true, size: 11 });
    /* matches */
    HF.txt(svg, 470, 150, "stored pairs (x-overlap, y-overlap):", { size: 11, fill: C.muted });
    for (let k = 0; k < K; k++) {
      const ox = HF.overlap(x, HF.glyph(names[k])), oy = HF.overlap(y, codes[k]);
      const yy = 168 + k * 18, good = ox > 0.999 && oy > 0.999;
      HF.txt(svg, 470, yy, `${names[k]} ↔ code ${k + 1}`, { size: 11, fill: good ? C.good : C.ink });
      HF.txt(svg, 590, yy, `${HF.fmt(ox, 2)}   ${HF.fmt(oy, 2)}`, { size: 11, mono: true, fill: good ? C.good : C.ink });
    }
    /* energy trace */
    const px = 260, py = 250, pw = 180, ph = 70;
    HF.txt(svg, px - 10, py - 60, "E(x, y) = −yᵀWx per half-step", { size: 10.5, bold: true });
    const xs = d3.scaleLinear().domain([0, Math.max(6, hist.length - 1)]).range([px, px + pw]);
    const eLo = Math.min(0, d3.min(hist, d => d.E)), eHi = Math.max(0, d3.max(hist, d => d.E)), ePad = 0.06 * (eHi - eLo) + 1;
    const ys = d3.scaleLinear().domain([eLo - ePad, eHi + ePad]).range([py + ph, py - 40]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${py + ph})`).call(d3.axisBottom(xs).ticks(6));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${px},0)`).call(d3.axisLeft(ys).ticks(4));
    svg.append("path").datum(hist).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2).attr("d", d3.line().x((d, i) => xs(i)).y(d => ys(d.E)));
    svg.append("g").selectAll("circle").data(hist).join("circle").attr("cx", (d, i) => xs(i)).attr("cy", d => ys(d.E)).attr("r", 3).attr("fill", d => d.dir === "x→y" ? C.B : C.A);
    const last = hist.length >= 3 && hist[hist.length - 1].E === hist[hist.length - 3].E && hist[hist.length - 1].E === hist[hist.length - 2].E;
    let who = "none"; for (let k = 0; k < K; k++) if (HF.overlap(x, HF.glyph(names[k])) > 0.999 && HF.overlap(y, codes[k]) > 0.999) who = `${names[k]} ↔ code ${k + 1}`;
    $("bam-readout").innerHTML = `half-steps ${hist.length - 1} · E = <b>${HF.fmt(hist[hist.length - 1].E, 0)}</b> · ${last ? "resonance: a full round-trip changed nothing" : "still moving"} · pair recalled: <b>${who}</b>`;
  }
  kE.addEventListener("input", () => { build(); start(); });
  nE.addEventListener("input", start);
  sideE.addEventListener("change", start);
  $("bam-step").addEventListener("click", stepHalf);
  $("bam-run").addEventListener("click", () => { for (let t = 0; t < 10; t++) stepHalf(); });
  $("bam-new").addEventListener("click", start);
  build(); start();
})();

/* ═════════════════ 6 · #ch-svg — the graded-response network ═════════════════ */
(function () {
  const svg = d3.select("#ch-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const lE = $("ch-gain"), wE = $("ch-w"), iE = $("ch-I");
  const X0 = 40, Y0 = 30, SZ = 330, R = 0.985;
  const vs = d3.scaleLinear().domain([-1, 1]).range([X0, X0 + SZ]), vy = d3.scaleLinear().domain([-1, 1]).range([Y0 + SZ, Y0]);
  let trajs = [];
  const G = v => v * Math.atanh(v) + 0.5 * Math.log(1 - v * v);    /* ∫₀ᵛ atanh(x) dx */
  function par() { return { lam: +lE.value, w: +wE.value, I: +iE.value }; }
  function En(v1, v2, p) { return -p.w * v1 * v2 - p.I * v1 + (G(v1) + G(v2)) / p.lam; }
  function integrate(v1, v2, p) {
    let u1 = Math.atanh(v1) / p.lam, u2 = Math.atanh(v2) / p.lam; const pts = [[v1, v2]], dt = 0.04;
    for (let t = 0; t < 900; t++) {
      const a1 = Math.tanh(p.lam * u1), a2 = Math.tanh(p.lam * u2);
      const d1 = -u1 + p.w * a2 + p.I, d2 = -u2 + p.w * a1;
      u1 += dt * d1; u2 += dt * d2;
      if (t % 3 === 0) pts.push([Math.tanh(p.lam * u1), Math.tanh(p.lam * u2)]);
      if (Math.abs(d1) + Math.abs(d2) < 1e-5) break;
    }
    pts.push([Math.tanh(p.lam * u1), Math.tanh(p.lam * u2)]);
    return pts;
  }
  function draw() {
    const p = par();
    $("ch-gainv").textContent = HF.fmt(p.lam, 2); $("ch-wv").textContent = HF.fmt(p.w, 2); $("ch-Iv").textContent = HF.fmt(p.I, 2);
    svg.selectAll("*").remove();
    const n = 70, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v1 = -R + 2 * R * i / (n - 1), v2 = R - 2 * R * j / (n - 1); vals[j * n + i] = En(v1, v2, p);
    }
    const lo = d3.min(vals), hi = d3.max(vals);
    const th = d3.range(lo, hi, (hi - lo) / 18);
    const col = d3.scaleSequential(d3.interpolateMagma).domain([lo, hi]);
    const cs = d3.contours().size([n, n]).thresholds(th)(vals);
    const sc = SZ / (n - 1);
    svg.append("g").attr("transform", `translate(${X0 + (1 - R) * SZ / 2},${Y0 + (1 - R) * SZ / 2}) scale(${sc * R})`)
      .selectAll("path").data(cs).join("path").attr("d", d3.geoPath()).attr("fill", d => col(d.value)).attr("stroke", "#0f1117").attr("stroke-width", 0.3 / (sc * R));
    svg.append("rect").attr("x", X0).attr("y", Y0).attr("width", SZ).attr("height", SZ).attr("fill", "transparent").style("cursor", "crosshair")
      .on("click", ev => {
        const [mx, my] = d3.pointer(ev);
        const v1 = Math.max(-R, Math.min(R, vs.invert(mx))), v2 = Math.max(-R, Math.min(R, vy.invert(my)));
        trajs.push([v1, v2]); if (trajs.length > 16) trajs.shift(); draw();
      });
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Y0 + SZ})`).call(d3.axisBottom(vs).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${X0},0)`).call(d3.axisLeft(vy).ticks(5));
    HF.txt(svg, X0 + SZ / 2, Y0 + SZ + 32, "v₁ = tanh(λu₁)", { size: 10.5, anchor: "middle", fill: C.muted });
    HF.txt(svg, X0, 18, "energy E(v₁, v₂) — dark = low; click to release a state", { bold: true, size: 11 });
    const ends = [];
    trajs.forEach(([a, b]) => {
      const pts = integrate(a, b, p);
      svg.append("path").datum(pts).attr("fill", "none").attr("stroke", "#e6e9ef").attr("stroke-width", 1.4).attr("pointer-events", "none")
        .attr("d", d3.line().x(d => vs(d[0])).y(d => vy(d[1])));
      svg.append("circle").attr("cx", vs(a)).attr("cy", vy(b)).attr("r", 3).attr("fill", C.A).attr("pointer-events", "none");
      const e = pts[pts.length - 1]; ends.push(e);
      svg.append("circle").attr("cx", vs(e[0])).attr("cy", vy(e[1])).attr("r", 5).attr("fill", C.good).attr("stroke", "#0f1117").attr("pointer-events", "none");
    });
    /* find the minima by releasing from a 6×6 grid, cluster endpoints */
    const mins = [];
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
      const pts = integrate(-0.9 + 0.36 * i + 0.017, -0.9 + 0.36 * j - 0.011, p), e = pts[pts.length - 1];   /* off-diagonal jitter: never start on a saddle's stable manifold */
      if (!mins.some(q => Math.hypot(q[0] - e[0], q[1] - e[1]) < 0.05)) mins.push(e);
    }
    /* right panel: the sigmoid at this gain, and the facts */
    const rx = 420, ry = 40, rw = 300, rh = 150;
    HF.txt(svg, rx, 18, "activation v = tanh(λu) at this gain", { bold: true, size: 11 });
    const us = d3.scaleLinear().domain([-3, 3]).range([rx, rx + rw]), ys = d3.scaleLinear().domain([-1.05, 1.05]).range([ry + rh, ry]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${ys(0)})`).call(d3.axisBottom(us).ticks(6));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${us(0)},0)`).call(d3.axisLeft(ys).ticks(3));
    svg.append("path").datum(d3.range(-3, 3.001, 0.05)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2)
      .attr("d", d3.line().x(u => us(u)).y(u => ys(Math.tanh(p.lam * u))));
    svg.append("path").datum(d3.range(-3, 3.001, 0.05)).attr("fill", "none").attr("stroke", C.line).attr("stroke-width", 1.2)
      .attr("d", d3.line().x(u => us(u)).y(u => ys(Math.tanh(20 * u))));
    const lines = [
      `gain × coupling  λ·|w| = ${HF.fmt(p.lam * Math.abs(p.w), 2)}  ${p.lam * Math.abs(p.w) > 1 ? "> 1" : "≤ 1"}`,
      `stable states found: ${mins.length}`,
    ].concat(mins.slice(0, 4).map(e => `  at v = (${HF.fmt(e[0], 2)}, ${HF.fmt(e[1], 2)})`));
    lines.forEach((l, i) => HF.txt(svg, rx, 230 + i * 18, l, { size: 11, mono: true, fill: i === 0 ? (p.lam * Math.abs(p.w) > 1 ? C.good : C.B) : C.ink }));
    HF.txt(svg, rx, 230 + lines.length * 18 + 8, "grey curve: the hard-limiter the net approaches as λ → ∞", { size: 10, fill: C.muted });
    $("ch-readout").innerHTML = p.lam * Math.abs(p.w) > 1
      ? `λ|w| &gt; 1: the origin is unstable and the minima sit near the corners ${p.w > 0 ? "(+,+) and (−,−)" : "(+,−) and (−,+)"}; raise λ and they slide into the corners, the binary network's fixed points`
      : `λ|w| ≤ 1: the only minimum is near the centre — at low gain the network cannot choose; this is the phase that <b>gain annealing</b> starts from`;
  }
  [lE, wE, iE].forEach(e => e.addEventListener("input", draw));
  $("ch-scatter").addEventListener("click", () => { const r = HF.rng((Date.now() & 0xffff) + 1); trajs = d3.range(12).map(() => [-0.95 + 1.9 * r(), -0.95 + 1.9 * r()]); draw(); });
  $("ch-clear").addEventListener("click", () => { trajs = []; draw(); });
  trajs = [[0.3, -0.15], [-0.2, 0.25], [0.7, 0.9], [-0.9, -0.3], [0.05, 0.02], [-0.6, 0.8]];
  draw();
})();

/* ═════════════════ 7 · #mh-svg — modern Hopfield = attention ═════════════════ */
(function () {
  const svg = d3.select("#mh-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const N = 100, qE = $("mh-q"), corE = $("mh-cor"), bE = $("mh-beta"), xE = $("mh-extra"), itE = $("mh-it");
  const base = HF.NAMES.map(HF.glyph);
  const extra = HF.randPats(80, N, HF.rng(2024));

  function draw() {
    const beta = Math.pow(10, +bE.value), K = +xE.value, it = +itE.value;
    $("mh-betav").textContent = beta < 0.1 ? beta.toFixed(3) : beta.toFixed(2); $("mh-extrav").textContent = K;
    const pats = base.concat(extra.slice(0, K)), P = pats.length;
    const tgt = HF.NAMES.indexOf(qE.value), t = base[tgt];
    let q;
    const cr = HF.rng(77);
    if (corE.value === "mask") q = t.map((v, i) => i < 50 ? v : 0);
    else if (corE.value === "noise") q = t.map(v => cr() < 0.3 ? -v : v);
    else q = t.map((v, i) => i < 30 ? v : 0);

    /* modern update, iterated */
    let xi = q.slice(), p = null;
    for (let s = 0; s < it; s++) {
      const z = pats.map(pt => { let d = 0; for (let i = 0; i < N; i++) d += pt[i] * xi[i]; return beta * d; });
      const mx = d3.max(z), e = z.map(v => Math.exp(v - mx)), Z = d3.sum(e);
      p = e.map(v => v / Z);
      const nx = new Array(N).fill(0);
      pats.forEach((pt, k) => { const w = p[k]; if (w > 1e-12) for (let i = 0; i < N; i++) nx[i] += w * pt[i]; });
      xi = nx;
    }
    /* classical Hopfield, same patterns, same query */
    const W = HF.hebb(pats, N);
    const cls = HF.relax(W, q.map(v => v === 0 ? -1 : v), N, HF.rng(3), 40).s;

    svg.selectAll("*").remove();
    HF.txt(svg, 20, 20, "query ξ", { bold: true }); HF.img(svg, q, 20, 30, 12, { gap: 1 });
    HF.txt(svg, 20, 170, "0 = unknown (grey)", { size: 9.5, fill: C.muted });
    HF.txt(svg, 175, 20, `modern: ξ ← Ξ·softmax(β Ξᵀξ)  ×${it}`, { bold: true, size: 11 }); HF.img(svg, xi, 175, 30, 12, { gap: 1, frame: HF.overlap(xi, t) > 0.95 ? C.good : C.B });
    HF.txt(svg, 175, 170, `overlap with ${qE.value}: ${HF.fmt(HF.overlap(xi, t), 3)}`, { size: 10.5, mono: true });
    HF.txt(svg, 400, 20, "classical Hebbian, async", { bold: true, size: 11 }); HF.img(svg, cls, 400, 30, 12, { gap: 1, frame: HF.overlap(cls, t) > 0.95 ? C.good : C.bad });
    HF.txt(svg, 400, 170, `overlap with ${qE.value}: ${HF.fmt(HF.overlap(cls, t), 3)}`, { size: 10.5, mono: true });
    HF.txt(svg, 600, 40, `P = ${P} patterns`, { size: 11 });
    HF.txt(svg, 600, 58, `α = P/N = ${HF.fmt(P / N, 2)}`, { size: 11, fill: P / N > 0.138 ? C.bad : C.good });
    HF.txt(svg, 600, 76, P / N > 0.138 ? "beyond classical 0.138" : "inside classical limit", { size: 10, fill: C.muted });

    /* softmax weights */
    const bx = 40, by = 210, bw = 690, bh = 110;
    HF.txt(svg, 20, by - 12, "attention weights softmax(β Ξᵀξ) over the stored patterns (last update) — glyphs first, then random patterns", { bold: true, size: 10.5 });
    const xs = d3.scaleBand().domain(d3.range(P)).range([bx, bx + bw]).padding(0.15);
    const ys = d3.scaleLinear().domain([0, 1]).range([by + bh, by]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(${bx},0)`).call(d3.axisLeft(ys).ticks(4));
    svg.append("line").attr("x1", bx).attr("x2", bx + bw).attr("y1", by + bh).attr("y2", by + bh).attr("stroke", "#3a4150");
    svg.append("g").selectAll("rect").data(p).join("rect").attr("x", (d, i) => xs(i)).attr("width", xs.bandwidth())
      .attr("y", d => ys(d)).attr("height", d => ys(0) - ys(d)).attr("fill", (d, i) => i === tgt ? C.good : i < base.length ? C.A : C.muted);
    base.forEach((b, i) => HF.txt(svg, xs(i) + xs.bandwidth() / 2, by + bh + 13, HF.NAMES[i], { size: 10, anchor: "middle", fill: i === tgt ? C.good : C.ink }));
    const H = -d3.sum(p, v => v > 0 ? v * Math.log(v) : 0);
    $("mh-readout").innerHTML = `top weight <b>${HF.fmt(d3.max(p), 3)}</b> · effective number of patterns mixed exp(H) = <b>${HF.fmt(Math.exp(H), 2)}</b> · ` +
      (Math.exp(H) > 1.5 ? "low β: a <b>metastable</b> average of several memories" : "high β: one memory retrieved — a single-pattern fixed point");
  }
  [bE, xE].forEach(e => e.addEventListener("input", draw));
  [qE, corE, itE].forEach(e => e.addEventListener("change", draw));
  draw();
})();
