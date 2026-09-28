/* self-organizing-maps.viz.js — figures for
   deep-learning/classical-nn/self-organizing-maps.html (part 4 of the
   Classical Neural Networks series).

   Loaded after ../../data.js and ../../notes.js, so the palette C is available.
   Each figure is an IIFE that exits quietly if its <svg> is not on the page.
   All numerics (competition, SOM updates, batch SOM, QE/TE, U-matrix, LVQ,
   ART-1) live in SM below; the figures only draw what SM computes, so every
   number in a readout is measured from the run on screen.

     1  #cl-svg    competitive learning: dead units vs conscience / FSCL / leaky
     2  #som-svg   a SOM unfolding over 2-D data, online or batch, with QE / TE
     3  #sc-svg    the neighbourhood kernel and the two-phase σ / η schedule
     4  #um-svg    colour SOM: weights, hit map, U-matrix, component planes
     5  #lvq-svg   LVQ1 / LVQ2.1 prototypes and the nearest-prototype boundary
     6  #art-svg   ART-1 on binary glyphs with a vigilance slider               */

/* ══════════ numerics ══════════ */
const SM = (function () {
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0; while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  function d2(a, b) { let s = 0; for (let k = 0; k < a.length; k++) { const t = a[k] - b[k]; s += t * t; } return s; }
  function shuffle(arr, r) {
    for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    return arr;
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  /* 2-D data sets in the unit square */
  function data2d(kind, n, r) {
    const X = [];
    for (let i = 0; i < n; i++) {
      if (kind === "square") X.push([0.05 + 0.9 * r(), 0.05 + 0.9 * r()]);
      else if (kind === "ring") {
        const a = 2 * Math.PI * r(), rad = 0.3 + 0.12 * Math.sqrt(r());
        X.push([0.5 + rad * Math.cos(a), 0.5 + rad * Math.sin(a)]);
      } else if (kind === "blobs") {
        const C3 = [[0.25, 0.3], [0.72, 0.28], [0.5, 0.75]], c = C3[i % 3];
        X.push([clamp(c[0] + 0.07 * gauss(r), 0, 1), clamp(c[1] + 0.07 * gauss(r), 0, 1)]);
      } else { /* L-shape */
        if (r() < 0.5) X.push([0.08 + 0.25 * r(), 0.08 + 0.84 * r()]);
        else X.push([0.08 + 0.84 * r(), 0.08 + 0.25 * r()]);
      }
    }
    return X;
  }

  /* rectangular lattice gw × gh; unit index = j*gw + i */
  function lattice(gw, gh) {
    const pos = [], nbrs = [];
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) pos.push([i, j]);
    for (let u = 0; u < pos.length; u++) {
      const [i, j] = pos[u], L = [];
      if (i > 0) L.push(u - 1); if (i < gw - 1) L.push(u + 1);
      if (j > 0) L.push(u - gw); if (j < gh - 1) L.push(u + gw);
      nbrs.push(L);
    }
    return { gw, gh, M: pos.length, pos, nbrs };
  }
  function latD2(lat, a, b) { const p = lat.pos[a], q = lat.pos[b]; const dx = p[0] - q[0], dy = p[1] - q[1]; return dx * dx + dy * dy; }
  function adjacent(lat, a, b) { const p = lat.pos[a], q = lat.pos[b]; return Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]) === 1; }

  function best2(x, W) {
    let b = -1, bv = Infinity, s = -1, sv = Infinity;
    for (let i = 0; i < W.length; i++) {
      const d = d2(x, W[i]);
      if (d < bv) { s = b; sv = bv; b = i; bv = d; } else if (d < sv) { s = i; sv = d; }
    }
    return { b, bv, s, sv };
  }

  /* two-phase schedule used by the unfolding figure:
     ordering (first quarter): σ₀ → min(1, σ₀) and η₀ → 0.02, exponentially;
     convergence: σ → 0.7 of its ordering end value, η 0.02 → 0.005.      */
  function somSchedule(t, T, s0, e0) {
    const T1 = 0.25 * T, s1 = Math.min(1, s0), e1 = Math.min(0.02, e0);
    if (t <= T1) { const f = t / T1; return { s: s0 * Math.pow(s1 / s0, f), e: e0 * Math.pow(e1 / e0, f), T1 }; }
    const f = Math.min(1, (t - T1) / (T - T1));
    return { s: s1 * Math.pow(0.7, f), e: e1 * Math.pow(0.25, f), T1 };
  }
  function somOnlineStep(W, lat, x, s, e) {
    const c = best2(x, W).b, inv = 1 / (2 * s * s);
    for (let i = 0; i < W.length; i++) {
      const h = Math.exp(-latD2(lat, c, i) * inv);
      if (h < 1e-4) continue;
      const w = W[i];
      for (let k = 0; k < w.length; k++) w[k] += e * h * (x[k] - w[k]);
    }
    return c;
  }
  function somBatchEpoch(W, lat, X, s) {
    const M = W.length, d = W[0].length, S = [], n = new Array(M).fill(0), inv = 1 / (2 * s * s);
    for (let i = 0; i < M; i++) S.push(new Array(d).fill(0));
    for (const x of X) { const c = best2(x, W).b; n[c]++; for (let k = 0; k < d; k++) S[c][k] += x[k]; }
    for (let i = 0; i < M; i++) {
      const num = new Array(d).fill(0); let den = 0;
      for (let j = 0; j < M; j++) {
        if (!n[j]) continue;
        const h = Math.exp(-latD2(lat, i, j) * inv);
        if (h < 1e-6) continue;
        den += h * n[j];
        for (let k = 0; k < d; k++) num[k] += h * S[j][k];
      }
      if (den > 1e-12) for (let k = 0; k < d; k++) W[i][k] = num[k] / den;
    }
    return n;
  }
  function qeTe(W, lat, X) {
    let qe = 0, te = 0;
    const hits = new Array(W.length).fill(0);
    for (const x of X) {
      const r = best2(x, W);
      qe += Math.sqrt(r.bv); hits[r.b]++;
      if (!adjacent(lat, r.b, r.s)) te++;
    }
    return { qe: qe / X.length, te: te / X.length, hits };
  }
  function umatrix(W, lat) {
    return W.map((w, i) => {
      const L = lat.nbrs[i]; let s = 0;
      for (const j of L) s += Math.sqrt(d2(w, W[j]));
      return L.length ? s / L.length : 0;
    });
  }

  /* competitive learning with the four rules of §02 */
  function clInit(K, seed) {
    const r = rng(seed), W = [];
    for (let k = 0; k < K; k++) W.push([0.06 + 0.05 * r(), 0.9 + 0.05 * r()]);
    return { W, p: new Array(K).fill(1 / K), n: new Array(K).fill(1), hist: [], t: 0 };
  }
  const CL_B = 0.003, CL_C = 3, CL_WINDOW = 500;
  function clStep(st, x, rule, eta) {
    const W = st.W, K = W.length;
    let c = -1, cv = Infinity;
    for (let k = 0; k < K; k++) {
      const d = d2(x, W[k]);
      const score = rule === "con" ? d - CL_C * (1 / K - st.p[k]) : rule === "fscl" ? st.n[k] * d : d;
      if (score < cv) { cv = score; c = k; }
    }
    for (let k = 0; k < K; k++) {
      const rate = k === c ? eta : (rule === "leaky" ? eta / 40 : 0);
      if (rate) { W[k][0] += rate * (x[0] - W[k][0]); W[k][1] += rate * (x[1] - W[k][1]); }
      st.p[k] += CL_B * ((k === c ? 1 : 0) - st.p[k]);
    }
    st.n[c]++; st.t++;
    st.hist.push(c); if (st.hist.length > CL_WINDOW) st.hist.shift();
    return c;
  }
  function clData(seed) {
    const r = rng(seed), X = [], C4 = [[0.3, 0.3], [0.75, 0.25], [0.72, 0.72], [0.45, 0.62]];
    for (let i = 0; i < 480; i++) { const c = C4[i % 4]; X.push([clamp(c[0] + 0.06 * gauss(r), 0, 1), clamp(c[1] + 0.06 * gauss(r), 0, 1)]); }
    return X;
  }
  function meanNearest(W, X) { let s = 0; for (const x of X) s += Math.sqrt(best2(x, W).bv); return s / X.length; }

  /* LVQ */
  function lvqData(kind, seed) {
    const r = rng(seed), X = [], Y = [];
    if (kind === "moons") {
      for (let i = 0; i < 300; i++) {
        const cls = i % 2, a = Math.PI * r();
        let x = cls ? 1 - Math.cos(a) : Math.cos(a), y = cls ? 0.5 - Math.sin(a) : Math.sin(a);
        x += 0.12 * gauss(r); y += 0.12 * gauss(r);
        X.push([(x + 1.4) / 4.4 + 0.02, (y + 0.9) / 2.3 * 0.8 + 0.1]); Y.push(cls);
      }
    } else if (kind === "xor") {
      const C4 = [[0.28, 0.28, 0], [0.72, 0.72, 0], [0.28, 0.72, 1], [0.72, 0.28, 1]];
      for (let i = 0; i < 320; i++) { const c = C4[i % 4]; X.push([clamp(c[0] + 0.09 * gauss(r), 0, 1), clamp(c[1] + 0.09 * gauss(r), 0, 1)]); Y.push(c[2]); }
    } else {
      const C3 = [[0.3, 0.32], [0.7, 0.35], [0.5, 0.72]];
      for (let i = 0; i < 330; i++) { const c = C3[i % 3]; X.push([clamp(c[0] + 0.11 * gauss(r), 0, 1), clamp(c[1] + 0.11 * gauss(r), 0, 1)]); Y.push(i % 3); }
    }
    return { X, Y, nc: kind === "three" ? 3 : 2 };
  }
  function lvqInit(D, P, seed) {
    const r = rng(seed), W = [], L = [];
    for (let c = 0; c < D.nc; c++) {
      const idx = []; D.Y.forEach((y, i) => { if (y === c) idx.push(i); });
      shuffle(idx, r);
      for (let p = 0; p < P; p++) { W.push(D.X[idx[p]].slice()); L.push(c); }
    }
    return { W, L };
  }
  function lvqAcc(M, D) {
    let ok = 0; D.X.forEach((x, i) => { if (M.L[best2(x, M.W).b] === D.Y[i]) ok++; });
    return ok / D.X.length;
  }
  function lvqEpoch(M, D, alg, eta, r, omega) {
    const order = shuffle(D.X.map((_, i) => i), r), s = (1 - omega) / (1 + omega);
    let moved = 0;
    for (const n of order) {
      const x = D.X[n], y = D.Y[n], b = best2(x, M.W);
      if (alg === "lvq1") {
        const w = M.W[b.b], sg = M.L[b.b] === y ? 1 : -1;
        w[0] += sg * eta * (x[0] - w[0]); w[1] += sg * eta * (x[1] - w[1]); moved++;
      } else {
        if (b.s < 0) continue;
        const ci = M.L[b.b] === y, cj = M.L[b.s] === y;
        if (ci === cj) continue;
        const di = Math.sqrt(b.bv), dj = Math.sqrt(b.sv);
        if (Math.min(di / dj, dj / di) <= s) continue;
        const good = ci ? b.b : b.s, bad = ci ? b.s : b.b, wg = M.W[good], wb = M.W[bad];
        wg[0] += eta * (x[0] - wg[0]); wg[1] += eta * (x[1] - wg[1]);
        wb[0] -= eta * (x[0] - wb[0]); wb[1] -= eta * (x[1] - wb[1]);
        moved++;
      }
    }
    return moved;
  }

  /* ART-1, fast learning */
  const ones = v => v.reduce((a, b) => a + b, 0);
  const andv = (a, b) => a.map((v, i) => v & b[i]);
  function artPresent(cats, I, rho, Lp) {
    const nI = ones(I), log = [];
    const T = cats.map((c, j) => ({ j, T: Lp * ones(andv(I, c.w)) / (Lp - 1 + ones(c.w)) }));
    T.sort((a, b) => b.T - a.T || a.j - b.j);
    for (const e of T) {
      const m = ones(andv(I, cats[e.j].w)) / nI;
      if (m >= rho) {
        cats[e.j].w = andv(I, cats[e.j].w);
        log.push({ j: e.j, T: e.T, m, ok: true });
        return { cat: e.j, log, fresh: false };
      }
      log.push({ j: e.j, T: e.T, m, ok: false });
    }
    cats.push({ w: I.slice(), members: [] });
    return { cat: cats.length - 1, log, fresh: true };
  }

  return { rng, gauss, d2, shuffle, clamp, data2d, lattice, latD2, adjacent, best2, somSchedule,
           somOnlineStep, somBatchEpoch, qeTe, umatrix, clInit, clStep, clData, meanNearest, CL_WINDOW,
           lvqData, lvqInit, lvqAcc, lvqEpoch, artPresent, ones };
})();

/* small shared drawing helpers */
const SMV = {
  txt(g, x, y, s, o) {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11)
      .attr("fill", o.fill || C.muted).attr("text-anchor", o.anchor || "start")
      .attr("font-weight", o.bold ? 600 : null).text(s);
  },
  miniChart(g, x, y, w, h, xmax, ymax, title) {
    const G = g.append("g").attr("transform", `translate(${x},${y})`);
    G.append("rect").attr("width", w).attr("height", h).attr("fill", "none").attr("stroke", C.line);
    SMV.txt(G, 0, -5, title, { size: 10.5, fill: C.ink });
    const sx = d3.scaleLinear().domain([0, xmax]).range([0, w]);
    const sy = d3.scaleLinear().domain([0, ymax]).range([h, 0]);
    return { G, sx, sy };
  },
  fmt: (v, d) => (+v).toFixed(d === undefined ? 3 : d),
  bindRange(id, fmt) {
    const el = document.getElementById(id), out = document.getElementById(id + "-out");
    const show = () => { if (out) out.textContent = fmt ? fmt(+el.value) : el.value; };
    show(); el.addEventListener("input", show);
    return el;
  }
};

/* ═══════════ 1 · #cl-svg — competitive learning & dead units ═══════════ */
(function () {
  const svg = d3.select("#cl-svg");
  if (svg.empty()) return;
  const X = SM.clData(11), S = 330, ox = 16, oy = 14;
  const rule = document.getElementById("cl-rule"), kEl = SMV.bindRange("cl-k"), eEl = SMV.bindRange("cl-eta", v => v.toFixed(2));
  const runB = document.getElementById("cl-run");
  let st, timer = null;
  const sx = v => ox + v * S, sy = v => oy + (1 - v) * S;
  const col = (k, K) => d3.interpolateTurbo(0.1 + 0.8 * k / Math.max(1, K - 1));

  function reset() { stop(); st = SM.clInit(+kEl.value, 5); draw(); }
  function stop() { if (timer) { timer.stop(); timer = null; } runB.textContent = "run"; }
  const r = SM.rng(99);
  function feed(n) { for (let i = 0; i < n; i++) SM.clStep(st, X[Math.floor(r() * X.length)], rule.value, +eEl.value); }

  function draw() {
    svg.selectAll("*").remove();
    const K = st.W.length;
    const vor = d3.Delaunay.from(st.W.map(w => [sx(w[0]), sy(w[1])])).voronoi([ox, oy, ox + S, oy + S]);
    const g = svg.append("g");
    for (let k = 0; k < K; k++) g.append("path").attr("d", vor.renderCell(k)).attr("fill", col(k, K)).attr("fill-opacity", 0.12).attr("stroke", C.line);
    g.append("rect").attr("x", ox).attr("y", oy).attr("width", S).attr("height", S).attr("fill", "none").attr("stroke", C.line);
    g.selectAll(".pt").data(X).join("circle").attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", 1.6).attr("fill", C.muted).attr("opacity", 0.55);
    const counts = new Array(K).fill(0); st.hist.forEach(c => counts[c]++);
    const warm = st.t >= SM.CL_WINDOW;
    st.W.forEach((w, k) => {
      const dead = warm && counts[k] === 0;
      g.append("circle").attr("cx", sx(w[0])).attr("cy", sy(w[1])).attr("r", 7).attr("fill", col(k, K))
        .attr("stroke", dead ? C.bad : "#fff").attr("stroke-width", dead ? 2.5 : 1.2);
      SMV.txt(g, sx(w[0]) + 9, sy(w[1]) + 4, String(k + 1), { size: 10, fill: C.ink });
    });
    /* bars: win share over the recent window */
    const bx = 400, bw = 340, by = 30, bh = 250;
    const G = svg.append("g");
    SMV.txt(G, bx, by - 12, "win share over the last " + SM.CL_WINDOW + " samples (fair share 1/K dashed)", { fill: C.ink, size: 11 });
    const n = Math.max(1, st.hist.length), yS = d3.scaleLinear().domain([0, Math.min(1, 1.25 * Math.max(2 / K, d3.max(counts) / n))]).range([by + bh, by]);
    const slot = bw / K;
    for (let k = 0; k < K; k++) {
      const v = counts[k] / n, dead = warm && counts[k] === 0;
      G.append("rect").attr("x", bx + k * slot + 3).attr("y", yS(v)).attr("width", slot - 6).attr("height", Math.max(0, by + bh - yS(v)))
        .attr("fill", dead ? C.bad : col(k, K)).attr("opacity", 0.85);
      SMV.txt(G, bx + k * slot + slot / 2, by + bh + 14, String(k + 1), { anchor: "middle", size: 10 });
      SMV.txt(G, bx + k * slot + slot / 2, yS(v) - 4, dead ? "dead" : (100 * v).toFixed(0) + "%", { anchor: "middle", size: 9.5, fill: dead ? C.bad : C.ink });
    }
    G.append("line").attr("x1", bx).attr("x2", bx + bw).attr("y1", yS(1 / K)).attr("y2", yS(1 / K)).attr("stroke", C.B).attr("stroke-dasharray", "4 3");
    G.append("line").attr("x1", bx).attr("x2", bx + bw).attr("y1", by + bh).attr("y2", by + bh).attr("stroke", C.line);
    const qe = SM.meanNearest(st.W, X), deadN = warm ? counts.filter(c => c === 0).length : 0;
    SMV.txt(G, bx, by + bh + 36, "unit  →", { size: 10 });
    document.getElementById("cl-readout").innerHTML =
      "samples seen <b>" + st.t + "</b> · rule <b>" + rule.options[rule.selectedIndex].text + "</b> · dead units (no wins in window) <b>" +
      (warm ? deadN : "–") + "</b> of " + K + " · quantization error (mean distance to winner) <b>" + qe.toFixed(4) + "</b>";
  }
  runB.addEventListener("click", () => {
    if (timer) { stop(); return; }
    runB.textContent = "pause";
    timer = d3.timer(() => { feed(40); draw(); if (st.t >= 8000) stop(); });
  });
  document.getElementById("cl-step").addEventListener("click", () => { feed(50); draw(); });
  document.getElementById("cl-reset").addEventListener("click", reset);
  rule.addEventListener("change", reset);
  kEl.addEventListener("change", reset);
  reset();
})();

/* ═══════════ 2 · #som-svg — the SOM unfolding ═══════════ */
(function () {
  const svg = d3.select("#som-svg");
  if (svg.empty()) return;
  const S = 350, ox = 14, oy = 14;
  const dataSel = document.getElementById("som-data"), gridSel = document.getElementById("som-grid"), modeSel = document.getElementById("som-mode");
  const s0El = SMV.bindRange("som-s0", v => v.toFixed(1)), e0El = SMV.bindRange("som-e0", v => v.toFixed(2)), TEl = SMV.bindRange("som-T");
  const playB = document.getElementById("som-play");
  const sx = v => ox + v * S, sy = v => oy + (1 - v) * S;
  let X, lat, W, t, trace, last, timer = null, r;

  function reset() {
    stop();
    r = SM.rng(7);
    X = SM.data2d(dataSel.value, 600, SM.rng(3));
    const [gw, gh] = gridSel.value.split("x").map(Number);
    lat = SM.lattice(gw, gh);
    W = lat.pos.map(() => [0.45 + 0.1 * r(), 0.45 + 0.1 * r()]);
    t = 0; trace = []; last = null;
    measure(); draw();
  }
  function stop() { if (timer) { timer.stop(); timer = null; } playB.textContent = "play"; }
  function measure() { const m = SM.qeTe(W, lat, X); trace.push({ t, qe: m.qe, te: m.te }); }
  function advance(nSamples) {
    const T = +TEl.value, s0 = +s0El.value, e0 = +e0El.value;
    if (modeSel.value === "batch") {
      /* one epoch = one pass = N samples' worth of schedule time */
      const sc = SM.somSchedule(Math.min(t, T), T, s0, e0);
      SM.somBatchEpoch(W, lat, X, sc.s);
      t = Math.min(T, t + X.length); last = null; measure(); return;
    }
    for (let i = 0; i < nSamples && t < T; i++) {
      const sc = SM.somSchedule(t, T, s0, e0), x = X[Math.floor(r() * X.length)];
      const c = SM.somOnlineStep(W, lat, x, sc.s, sc.e);
      last = { x, c }; t++;
      if (t % 200 === 0) measure();
    }
  }

  function draw() {
    svg.selectAll("*").remove();
    const T = +TEl.value, s0 = +s0El.value, e0 = +e0El.value, sc = SM.somSchedule(Math.min(t, T), T, s0, e0);
    const g = svg.append("g");
    g.append("rect").attr("x", ox).attr("y", oy).attr("width", S).attr("height", S).attr("fill", "none").attr("stroke", C.line);
    g.selectAll(".d").data(X).join("circle").attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", 1.4).attr("fill", C.muted).attr("opacity", 0.4);
    const segs = [];
    for (let u = 0; u < lat.M; u++) for (const v of lat.nbrs[u]) if (v > u) segs.push([W[u], W[v]]);
    g.selectAll(".e").data(segs).join("line").attr("x1", d => sx(d[0][0])).attr("y1", d => sy(d[0][1])).attr("x2", d => sx(d[1][0])).attr("y2", d => sy(d[1][1]))
      .attr("stroke", C.A).attr("stroke-width", 1).attr("opacity", 0.85);
    g.selectAll(".u").data(W).join("circle").attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", 2.2).attr("fill", C.A);
    /* mark the lattice corner (0,0) so the orientation is readable */
    g.append("circle").attr("cx", sx(W[0][0])).attr("cy", sy(W[0][1])).attr("r", 4.5).attr("fill", C.good).append("title").text("lattice unit (0,0)");
    if (last) {
      g.append("line").attr("x1", sx(last.x[0])).attr("y1", sy(last.x[1])).attr("x2", sx(W[last.c][0])).attr("y2", sy(W[last.c][1])).attr("stroke", C.B).attr("stroke-width", 1.2);
      g.append("circle").attr("cx", sx(last.x[0])).attr("cy", sy(last.x[1])).attr("r", 4).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.5);
      g.append("circle").attr("cx", sx(W[last.c][0])).attr("cy", sy(W[last.c][1])).attr("r", 4.5).attr("fill", C.B);
    }
    /* right: schedule + error traces */
    const bx = 420, bw = 320;
    const A = SMV.miniChart(svg, bx, 22, bw, 80, T, Math.max(1, s0), "σ(t) — neighbourhood width (lattice units)");
    const B = SMV.miniChart(svg, bx, 138, bw, 70, T, Math.max(0.02, e0), "η(t) — learning rate (online only)");
    const qmax = Math.max(0.02, d3.max(trace, d => d.qe) || 0.1);
    const Cc = SMV.miniChart(svg, bx, 246, bw, 100, T, 1, "QE (blue, scaled to its max " + qmax.toFixed(3) + ") and TE (amber, 0–1)");
    [A, B, Cc].forEach(Q => Q.G.append("rect").attr("x", 0).attr("y", 0).attr("width", Q.sx(sc.T1)).attr("height", +Q.G.select("rect").attr("height")).attr("fill", C.B).attr("opacity", 0.06));
    const ts = d3.range(0, T + 1, T / 200);
    A.G.append("path").attr("d", d3.line().x(v => A.sx(v)).y(v => A.sy(SM.somSchedule(v, T, s0, e0).s))(ts)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.6);
    B.G.append("path").attr("d", d3.line().x(v => B.sx(v)).y(v => B.sy(SM.somSchedule(v, T, s0, e0).e))(ts)).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 1.6);
    Cc.G.append("path").attr("d", d3.line().x(d => Cc.sx(d.t)).y(d => Cc.sy(d.qe / qmax))(trace)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.6);
    Cc.G.append("path").attr("d", d3.line().x(d => Cc.sx(d.t)).y(d => Cc.sy(d.te))(trace)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.6);
    [A, B, Cc].forEach(Q => Q.G.append("line").attr("x1", Q.sx(t)).attr("x2", Q.sx(t)).attr("y1", 0).attr("y2", +Q.G.select("rect").attr("height")).attr("stroke", C.ink).attr("stroke-dasharray", "3 3"));
    SMV.txt(A.G, 4, 12, "ordering", { size: 9.5, fill: C.B });
    SMV.txt(A.G, A.sx(sc.T1) + 4, 12, "convergence", { size: 9.5, fill: C.muted });
    const m = trace[trace.length - 1];
    document.getElementById("som-readout").innerHTML =
      "t = <b>" + t + "</b> / " + T + " · phase <b>" + (t <= sc.T1 ? "ordering" : "convergence") + "</b> · σ = <b>" + sc.s.toFixed(2) + "</b>" +
      (modeSel.value === "online" ? " · η = <b>" + sc.e.toFixed(3) + "</b>" : " · batch: no η") +
      " · QE = <b>" + m.qe.toFixed(4) + "</b> · TE = <b>" + (100 * m.te).toFixed(1) + "%</b> · green dot = lattice unit (0,0)";
  }
  playB.addEventListener("click", () => {
    if (timer) { stop(); return; }
    if (t >= +TEl.value) reset();
    playB.textContent = "pause";
    let frame = 0;
    timer = d3.timer(() => {
      frame++;
      if (modeSel.value === "batch") { if (frame % 8 !== 0) return; advance(0); }
      else advance(Math.max(10, Math.round(+TEl.value / 250)));
      draw();
      if (t >= +TEl.value) stop();
    });
  });
  document.getElementById("som-step").addEventListener("click", () => { advance(100); draw(); });
  document.getElementById("som-reset").addEventListener("click", reset);
  [dataSel, gridSel, modeSel].forEach(el => el.addEventListener("change", reset));
  [s0El, e0El, TEl].forEach(el => el.addEventListener("change", reset));
  reset();
})();

/* ═══════════ 3 · #sc-svg — kernel and schedule ═══════════ */
(function () {
  const svg = d3.select("#sc-svg");
  if (svg.empty()) return;
  const s0El = SMV.bindRange("sc-s0", v => v.toFixed(1)), t1El = SMV.bindRange("sc-t1"), tEl = SMV.bindRange("sc-t");
  const kind = document.getElementById("sc-kind");
  const Tmax = 6000;
  function sched(t, s0, T1) {
    const tau = s0 > 1 ? T1 / Math.log(s0) : Infinity;
    const s = t <= T1 ? (s0 > 1 ? s0 * Math.exp(-t / tau) : s0) : Math.min(1, s0);
    const e = Math.max(0.01, 0.1 * Math.exp(-t / T1));
    return { s, e, tau };
  }
  function h(d, s, k) {
    if (k === "bubble") return d <= s ? 1 : 0;
    if (k === "mexican") return (Math.exp(-d * d / (2 * s * s)) - 0.5 * Math.exp(-d * d / (8 * s * s))) / 0.5;
    return Math.exp(-d * d / (2 * s * s));
  }
  function draw() {
    svg.selectAll("*").remove();
    const s0 = +s0El.value, T1 = +t1El.value, t = +tEl.value, k = kind.value, sc = sched(t, s0, T1);
    const n = 11, cell = 24, ox = 20, oy = 26, mid = 5;
    SMV.txt(svg, ox, 14, "h(winner, i) on an 11 × 11 lattice, σ = " + sc.s.toFixed(2), { fill: C.ink });
    let lit = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const d = Math.hypot(i - mid, j - mid), v = h(d, sc.s, k);
      if (v > 0.1) lit++;
      const fill = v >= 0 ? d3.interpolateRgb("#1e222d", C.B)(Math.min(1, v)) : d3.interpolateRgb("#1e222d", C.A)(Math.min(1, -v * 2));
      svg.append("rect").attr("x", ox + i * cell).attr("y", oy + j * cell).attr("width", cell - 2).attr("height", cell - 2).attr("rx", 3)
        .attr("fill", fill).attr("stroke", i === mid && j === mid ? "#fff" : "none").append("title").text("lattice distance " + d.toFixed(2) + ", h = " + v.toFixed(3));
      if (Math.abs(v) >= 0.05) SMV.txt(svg, ox + i * cell + (cell - 2) / 2, oy + j * cell + 15, v.toFixed(1).replace("0.", "."), { anchor: "middle", size: 8.5, fill: Math.abs(v) > 0.55 ? "#111" : C.ink });
    }
    const bx = 330, bw = 410;
    const A = SMV.miniChart(svg, bx, 30, bw, 110, Tmax, s0, "σ(t):  σ₀·e^(−t/τ) with τ = T₁/ln σ₀ = " + (isFinite(sc.tau) ? sc.tau.toFixed(0) : "∞") + ", then held at 1");
    const B = SMV.miniChart(svg, bx, 180, bw, 100, Tmax, 0.1, "η(t) = max(0.01, 0.1·e^(−t/T₁))");
    const ts = d3.range(0, Tmax + 1, 20);
    [A, B].forEach(Q => Q.G.append("rect").attr("width", Q.sx(T1)).attr("height", Q === A ? 110 : 100).attr("fill", C.B).attr("opacity", 0.07));
    A.G.append("path").attr("d", d3.line().x(v => A.sx(v)).y(v => A.sy(sched(v, s0, T1).s))(ts)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.8);
    B.G.append("path").attr("d", d3.line().x(v => B.sx(v)).y(v => B.sy(sched(v, s0, T1).e))(ts)).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 1.8);
    A.G.append("line").attr("x1", 0).attr("x2", bw).attr("y1", A.sy(1)).attr("y2", A.sy(1)).attr("stroke", C.muted).attr("stroke-dasharray", "2 3");
    [A, B].forEach(Q => Q.G.append("line").attr("x1", Q.sx(t)).attr("x2", Q.sx(t)).attr("y1", 0).attr("y2", Q === A ? 110 : 100).attr("stroke", C.ink).attr("stroke-dasharray", "3 3"));
    SMV.txt(A.G, 4, 12, "ordering (T₁ = " + T1 + ")", { size: 9.5, fill: C.B });
    SMV.txt(A.G, bw - 4, 12, "convergence →", { size: 9.5, anchor: "end" });
    [0, 2000, 4000, 6000].forEach(v => SMV.txt(B.G, B.sx(v), 114, String(v), { anchor: "middle", size: 9 }));
    document.getElementById("sc-readout").innerHTML =
      "t = <b>" + t + "</b> · σ(t) = <b>" + sc.s.toFixed(3) + "</b> · η(t) = <b>" + sc.e.toFixed(4) + "</b> · h at lattice distance 1, 2, 3 = <b>" +
      [1, 2, 3].map(d => h(d, sc.s, k).toFixed(3)).join(", ") + "</b> · units with |h| &gt; 0.1: <b>" + lit + "</b> of 121";
  }
  [s0El, t1El, tEl].forEach(el => el.addEventListener("input", draw));
  kind.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 4 · #um-svg — colour SOM, U-matrix, component planes ═══════════ */
(function () {
  const svg = d3.select("#um-svg");
  if (svg.empty()) return;
  const kEl = SMV.bindRange("um-k"), spEl = SMV.bindRange("um-spread", v => v.toFixed(2));
  const view = document.getElementById("um-view");
  const G = 16, lat = SM.lattice(G, G);
  let seed = 21, W, X, stats, U;

  function train() {
    const r = SM.rng(seed), K = +kEl.value, sp = +spEl.value, base = [];
    for (let k = 0; k < K; k++) base.push([r(), r(), r()]);
    X = [];
    for (let n = 0; n < 640; n++) { const b = base[n % K]; X.push(b.map(v => SM.clamp(v + sp * SM.gauss(r), 0, 1))); }
    W = lat.pos.map(() => [0.4 + 0.2 * r(), 0.4 + 0.2 * r(), 0.4 + 0.2 * r()]);
    const E = 30;
    for (let e = 0; e < E; e++) SM.somBatchEpoch(W, lat, X, 6 * Math.pow(0.8 / 6, e / (E - 1)));
    stats = SM.qeTe(W, lat, X); U = SM.umatrix(W, lat);
  }
  function panel(x, y, size, title, colorOf, tip) {
    const c = size / G, g = svg.append("g").attr("transform", `translate(${x},${y})`);
    SMV.txt(g, 0, -5, title, { size: 10.5, fill: C.ink });
    for (let u = 0; u < lat.M; u++) {
      const [i, j] = lat.pos[u];
      g.append("rect").attr("x", i * c).attr("y", j * c).attr("width", c + 0.3).attr("height", c + 0.3).attr("fill", colorOf(u))
        .on("mouseenter", () => show(u)).append("title").text(tip(u));
    }
  }
  function show(u) {
    const w = W[u], [i, j] = lat.pos[u];
    document.getElementById("um-readout").innerHTML =
      "unit (" + i + ", " + j + ") · w = (<b>" + w.map(v => v.toFixed(2)).join(", ") + "</b>) · U = <b>" + U[u].toFixed(3) + "</b> · hits = <b>" + stats.hits[u] + "</b>" +
      " · map QE = " + stats.qe.toFixed(4) + " · TE = " + (100 * stats.te).toFixed(1) + "%";
  }
  function draw() {
    svg.selectAll("*").remove();
    const rgb = w => d3.rgb(255 * w[0], 255 * w[1], 255 * w[2]).toString();
    const umax = d3.max(U), hmax = d3.max(stats.hits) || 1;
    const tip = u => "w = (" + W[u].map(v => v.toFixed(2)).join(", ") + ")  U = " + U[u].toFixed(3) + "  hits = " + stats.hits[u];
    if (view.value === "w") panel(10, 22, 256, "units coloured by their weight (R,G,B)", u => rgb(W[u]), tip);
    else panel(10, 22, 256, "hit map (inputs won per unit)", u => d3.interpolateViridis(stats.hits[u] / hmax), tip);
    panel(286, 22, 256, "U-matrix (dark = dense, bright = gap)", u => d3.interpolateMagma(0.05 + 0.9 * U[u] / umax), tip);
    ["R", "G", "B"].forEach((nm, k) => {
      const base = ["#ff5a5a", "#4ade80", "#5b9cff"][k];
      panel(566, 22 + k * 102, 84, "component plane " + nm, u => d3.interpolateRgb("#0f1117", base)(W[u][k]), tip);
    });
    SMV.txt(svg, 664, 60, "same grid,", { size: 10 }); SMV.txt(svg, 664, 74, "one input", { size: 10 }); SMV.txt(svg, 664, 88, "feature each", { size: 10 });
    SMV.txt(svg, 10, 298, "16 × 16 lattice · batch SOM, 30 epochs, σ 6 → 0.8 · " + X.length + " training colours", { size: 10 });
    document.getElementById("um-readout").innerHTML =
      "QE = <b>" + stats.qe.toFixed(4) + "</b> · TE = <b>" + (100 * stats.te).toFixed(1) + "%</b> · U-matrix range <b>" + d3.min(U).toFixed(3) + " … " + umax.toFixed(3) +
      "</b> · units with zero hits <b>" + stats.hits.filter(h => h === 0).length + "</b> of " + lat.M + " · hover a unit for its weight vector";
  }
  function redo() { train(); draw(); }
  kEl.addEventListener("change", redo); spEl.addEventListener("change", redo);
  view.addEventListener("change", draw);
  document.getElementById("um-retrain").addEventListener("click", () => { seed += 17; redo(); });
  redo();
})();

/* ═══════════ 5 · #lvq-svg — LVQ boundary ═══════════ */
(function () {
  const svg = d3.select("#lvq-svg");
  if (svg.empty()) return;
  const S = 350, ox = 14, oy = 14, CC = [C.A, C.B, C.good];
  const dataSel = document.getElementById("lvq-data"), algSel = document.getElementById("lvq-alg");
  const pEl = SMV.bindRange("lvq-p"), eEl = SMV.bindRange("lvq-eta", v => v.toFixed(3));
  const runB = document.getElementById("lvq-run");
  const sx = v => ox + v * S, sy = v => oy + (1 - v) * S;
  let D, M, acc, epoch, r, timer = null, lastMoved = 0;

  function reset() {
    stop();
    D = SM.lvqData(dataSel.value, 4);
    M = SM.lvqInit(D, +pEl.value, 8);
    r = SM.rng(12); epoch = 0; lastMoved = 0;
    acc = [{ e: 0, a: SM.lvqAcc(M, D), alg: "init" }];
    draw();
  }
  function stop() { if (timer) { timer.stop(); timer = null; } runB.textContent = "run"; }
  function oneEpoch() {
    const alg = algSel.value;
    lastMoved = SM.lvqEpoch(M, D, alg, +eEl.value, r, 0.3);
    epoch++; acc.push({ e: epoch, a: SM.lvqAcc(M, D), alg });
  }
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), R = 56, c = S / R;
    for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
      const p = [(i + 0.5) / R, 1 - (j + 0.5) / R], k = M.L[SM.best2(p, M.W).b];
      g.append("rect").attr("x", ox + i * c).attr("y", oy + j * c).attr("width", c + 0.3).attr("height", c + 0.3).attr("fill", CC[k]).attr("opacity", 0.16);
    }
    g.append("rect").attr("x", ox).attr("y", oy).attr("width", S).attr("height", S).attr("fill", "none").attr("stroke", C.line);
    g.selectAll(".d").data(D.X).join("circle").attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", 2.3)
      .attr("fill", (d, i) => CC[D.Y[i]]).attr("opacity", 0.8);
    const dia = d3.symbol().type(d3.symbolDiamond).size(150);
    M.W.forEach((w, i) => g.append("path").attr("d", dia).attr("transform", `translate(${sx(w[0])},${sy(w[1])})`)
      .attr("fill", CC[M.L[i]]).attr("stroke", "#fff").attr("stroke-width", 1.8));
    const Q = SMV.miniChart(svg, 420, 30, 320, 200, Math.max(10, epoch), 1, "training accuracy by epoch (y from 0.5 to 1)");
    Q.sy.domain([0.5, 1]);
    Q.G.append("path").attr("d", d3.line().x(d => Q.sx(d.e)).y(d => Q.sy(Math.max(0.5, d.a)))(acc)).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 1.8);
    Q.G.selectAll(".p").data(acc).join("circle").attr("cx", d => Q.sx(d.e)).attr("cy", d => Q.sy(Math.max(0.5, d.a))).attr("r", 2.5)
      .attr("fill", d => d.alg === "lvq21" ? C.B : C.good);
    [0.5, 0.75, 1].forEach(v => SMV.txt(Q.G, -4, Q.sy(v) + 3, v.toFixed(2), { anchor: "end", size: 9 }));
    SMV.txt(svg, 420, 262, "green points: LVQ1 epochs · amber: LVQ2.1 epochs", { size: 10 });
    SMV.txt(svg, 420, 280, "diamonds = prototypes; the background is the class", { size: 10 });
    SMV.txt(svg, 420, 294, "of the nearest prototype, so the boundary is made of", { size: 10 });
    SMV.txt(svg, 420, 308, "Voronoi faces between differently-labelled prototypes", { size: 10 });
    const last = acc[acc.length - 1];
    document.getElementById("lvq-readout").innerHTML =
      "epoch <b>" + epoch + "</b> · training accuracy <b>" + (100 * last.a).toFixed(1) + "%</b> (initial " + (100 * acc[0].a).toFixed(1) + "%) · prototypes <b>" + M.W.length +
      "</b> · samples that moved a prototype last epoch <b>" + (epoch ? lastMoved : "–") + "</b> of " + D.X.length +
      (algSel.value === "lvq21" ? " (LVQ2.1 window s = 0.7/1.3 = 0.538)" : "");
  }
  runB.addEventListener("click", () => {
    if (timer) { stop(); return; }
    runB.textContent = "pause"; let f = 0, done = 0;
    timer = d3.timer(() => { if (++f % 10) return; oneEpoch(); draw(); if (++done >= 20) stop(); });
  });
  document.getElementById("lvq-step").addEventListener("click", () => { oneEpoch(); draw(); });
  document.getElementById("lvq-reset").addEventListener("click", reset);
  dataSel.addEventListener("change", reset); pEl.addEventListener("change", reset);
  algSel.addEventListener("change", draw);
  reset();
})();

/* ═══════════ 6 · #art-svg — ART-1 ═══════════ */
const ART_GLYPHS = [
  { name: "L", g: ["X....", "X....", "X....", "X....", "XXXXX"] },
  { name: "short L", g: ["X....", "X....", "X....", "X....", "XXX.."] },
  { name: "plus", g: ["..X..", "..X..", "XXXXX", "..X..", "..X.."] },
  { name: "short plus", g: [".....", "..X..", "XXXXX", "..X..", "..X.."] },
  { name: "box", g: ["XXXXX", "X...X", "X...X", "X...X", "XXXXX"] },
  { name: "broken box", g: ["XX.XX", "....X", "X...X", "X....", "XX.XX"] },
  { name: "T", g: ["XXXXX", "..X..", "..X..", "..X..", "..X.."] },
  { name: "X", g: ["X...X", ".X.X.", "..X..", ".X.X.", "X...X"] }
].map(p => ({ name: p.name, v: p.g.join("").split("").map(ch => ch === "X" ? 1 : 0) }));

(function () {
  const svg = d3.select("#art-svg");
  if (svg.empty()) return;
  const rhoEl = SMV.bindRange("art-rho", v => v.toFixed(2)), orderSel = document.getElementById("art-order");
  let cats, pos, last, order;

  function reset() {
    cats = []; pos = 0; last = null;
    order = ART_GLYPHS.map((_, i) => i); if (orderSel.value === "rev") order.reverse();
    draw();
  }
  function next() {
    if (pos >= order.length) return false;
    const idx = order[pos], res = SM.artPresent(cats, ART_GLYPHS[idx].v, +rhoEl.value, 2);
    cats[res.cat].members.push(idx);
    last = { idx, res }; pos++;
    return true;
  }
  function glyph(g, x, y, v, cell, hi) {
    for (let k = 0; k < 25; k++) g.append("rect").attr("x", x + (k % 5) * cell).attr("y", y + Math.floor(k / 5) * cell)
      .attr("width", cell - 1).attr("height", cell - 1).attr("fill", v[k] ? (hi || C.ink) : "#1e222d");
  }
  function draw() {
    svg.selectAll("*").remove();
    SMV.txt(svg, 10, 14, "inputs, in presentation order", { fill: C.ink });
    order.forEach((idx, k) => {
      const x = 10 + k * 92, cur = last && last.idx === idx, done = k < pos;
      glyph(svg, x, 22, ART_GLYPHS[idx].v, 9, cur ? C.B : (done ? C.muted : C.ink));
      SMV.txt(svg, x, 80, ART_GLYPHS[idx].name + " |I|=" + SM.ones(ART_GLYPHS[idx].v), { size: 9.5, fill: cur ? C.B : C.muted });
    });
    SMV.txt(svg, 10, 106, "search log for the current input", { fill: C.ink });
    if (last) {
      const nI = SM.ones(ART_GLYPHS[last.idx].v);
      last.res.log.forEach((e, k) => {
        SMV.txt(svg, 20, 124 + k * 15,
          "try cat " + (e.j + 1) + ":  T = " + e.T.toFixed(3) + "   match = " + Math.round(e.m * nI) + "/" + nI + " = " + e.m.toFixed(3) +
          (e.ok ? "  ≥ ρ  → RESONANCE, template ← I ∧ w" : "  < ρ  → reset"), { size: 10.5, fill: e.ok ? C.good : C.bad });
      });
      if (last.res.fresh) SMV.txt(svg, 20, 124 + last.res.log.length * 15, "no category passes → commit new cat " + (last.res.cat + 1) + " with w = I", { size: 10.5, fill: C.B });
    } else SMV.txt(svg, 20, 124, "(press “present next”)", { size: 10.5 });
    const y0 = 250;
    SMV.txt(svg, 10, y0 - 8, "committed categories (templates after learning) and their members", { fill: C.ink });
    cats.forEach((c, j) => {
      const x = 10 + j * 92;
      glyph(svg, x, y0, c.w, 9, last && last.res.cat === j ? C.good : C.A);
      SMV.txt(svg, x, y0 + 58, "cat " + (j + 1) + " |w|=" + SM.ones(c.w), { size: 9.5, fill: C.ink });
      c.members.map(i => ART_GLYPHS[i].name).filter((v, i, a) => a.indexOf(v) === i).slice(0, 4)
        .forEach((nm, q) => SMV.txt(svg, x, y0 + 72 + q * 11, nm, { size: 8.5 }));
    });
    document.getElementById("art-readout").innerHTML =
      "ρ = <b>" + (+rhoEl.value).toFixed(2) + "</b> · presented <b>" + pos + "</b> of " + order.length + " · categories <b>" + cats.length + "</b>" +
      (last ? " · last input “" + ART_GLYPHS[last.idx].name + "” → cat <b>" + (last.res.cat + 1) + "</b> after <b>" + last.res.log.filter(e => !e.ok).length + "</b> reset(s)" : "");
  }
  document.getElementById("art-next").addEventListener("click", () => { next(); draw(); });
  document.getElementById("art-all").addEventListener("click", () => { while (next()); draw(); });
  document.getElementById("art-reset").addEventListener("click", reset);
  rhoEl.addEventListener("change", reset); orderSel.addEventListener("change", reset);
  reset();
})();
