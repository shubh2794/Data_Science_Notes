/* kalman-filter-training.viz.js — figures for deep-learning/classical-nn/kalman-filter-training.html (part 7 of 10).
   Loaded after ../../data.js and ../../notes.js, so the palette C and d3 are global.
   Each figure is an IIFE that returns quietly when its <svg> is absent.

     1  #kf-svg    1-D Kalman tracking of a random walk, filter Q/R vs true noise, gain transient
     2  #es-svg    EKF (global / node-decoupled / fully decoupled) vs SGD on a tiny tanh net
     3  #el-svg    2-weight model: covariance ellipses shrinking over updates, eigenvalue decay
     4  #dc-svg    which blocks of P each decoupling keeps, and what it costs
     5  #qt-svg    tracking a changing target with q = 0 vs q > 0
     6  #ut-svg    Gaussian through a nonlinearity: linearization vs unscented transform          */

/* ══════════ page-local helpers ══════════ */
const KT = (function () {
  const $ = id => document.getElementById(id);
  const val = id => +$(id).value;
  function on(ids, cb) {
    ids.forEach(id => {
      const el = $(id); if (!el) return;
      const out = $(id + "-o");
      const h = () => { if (out) out.textContent = el.value; cb(); };
      el.addEventListener("input", h); el.addEventListener("change", h);
    });
  }
  const rng = seed => { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; };
  const gauss = r => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const axisB = (g, sc, y, n, fmt) => g.append("g").attr("class", "axis").attr("transform", `translate(0,${y})`).call(d3.axisBottom(sc).ticks(n || 5, fmt));
  const axisL = (g, sc, x, n, fmt) => g.append("g").attr("class", "axis").attr("transform", `translate(${x},0)`).call(d3.axisLeft(sc).ticks(n || 4, fmt));
  const txt = (g, x, y, s, col, size, anchor) => g.append("text").attr("x", x).attr("y", y).attr("fill", col || C.muted)
    .attr("font-size", size || 11).attr("text-anchor", anchor || "start").text(s);
  const legend = (g, x, y, items, dy) => items.forEach((it, i) => {
    g.append("line").attr("x1", x).attr("x2", x + 16).attr("y1", y + i * (dy || 15)).attr("y2", y + i * (dy || 15))
      .attr("stroke", it[1]).attr("stroke-width", it[2] || 2.2).attr("stroke-dasharray", it[3] || null);
    txt(g, x + 21, y + i * (dy || 15) + 4, it[0], C.muted, 10.5);
  });
  const e2 = v => (+v).toExponential(2), f3 = v => (+v).toFixed(3), f2 = v => (+v).toFixed(2);
  /* symmetric 2×2 eigen-decomposition: returns [[λ1, v1], [λ2, v2]] with λ1 ≥ λ2 */
  function eig2(a, b, c) {
    const m = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b), l1 = m + d, l2 = Math.max(0, m - d);
    const th = 0.5 * Math.atan2(2 * b, a - c);
    return [[l1, [Math.cos(th), Math.sin(th)]], [l2, [-Math.sin(th), Math.cos(th)]]];
  }
  return { $, val, on, rng, gauss, axisB, axisL, txt, legend, e2, f3, f2, eig2 };
})();

/* ══════════ 1. 1-D Kalman tracking ══════════ */
(function () {
  const svg = d3.select("#kf-svg"); if (svg.empty()) return;
  const N = 200; let seed = 5;
  const x = d3.scaleLinear([0, N - 1], [50, 740]);
  const y = d3.scaleLinear([0, 1], [250, 20]), gy = d3.scaleLinear([0, 1], [355, 285]);
  const gAxY = svg.append("g"), gAxG = svg.append("g");
  KT.axisB(svg, x, 250, 8); KT.axisB(svg, x, 355, 8);
  KT.txt(svg, 740, 376, "time step k", C.muted, 10.5, "end");
  KT.txt(svg, 56, 280, "Kalman gain Kₖ", C.muted, 10.5);
  const band = svg.append("path").attr("fill", C.A).attr("fill-opacity", .18);
  const dots = svg.append("g");
  const tru = svg.append("path").attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 1.6);
  const est = svg.append("path").attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2);
  const gain = svg.append("path").attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.8);
  const kinf = svg.append("line").attr("x1", 50).attr("x2", 740).attr("stroke", C.B).attr("stroke-dasharray", "5 4").attr("stroke-opacity", .7);
  const lg = svg.append("g");
  KT.legend(lg, 560, 30, [["truth", "#fff", 1.6], ["measurements", C.muted, 1, "1 3"], ["estimate ± 2√P", C.A]]);

  function draw() {
    const tq = Math.pow(10, KT.val("kf-tq")), tr = Math.pow(10, KT.val("kf-tr"));
    const Q = Math.pow(10, KT.val("kf-q")), R = Math.pow(10, KT.val("kf-r"));
    const r = KT.rng(seed), X = [], Z = [];
    let xt = 0;
    for (let k = 0; k < N; k++) { if (k) xt += Math.sqrt(tq) * KT.gauss(r); X.push(xt); Z.push(xt + Math.sqrt(tr) * KT.gauss(r)); }
    let xh = 0, P = 10; const XH = [], PP = [], KK = [];
    for (let k = 0; k < N; k++) {
      P += Q; const K = P / (P + R); xh += K * (Z[k] - xh); P = (1 - K) * P;
      XH.push(xh); PP.push(P); KK.push(K);
    }
    const lo = d3.min(X.concat(XH)), hi = d3.max(X.concat(XH)), pad = Math.max(0.5, (hi - lo) * 0.25);
    y.domain([lo - pad, hi + pad]);
    gAxY.selectAll("*").remove(); KT.axisL(gAxY, y, 50, 5);
    gy.domain([0, 1]); gAxG.selectAll("*").remove(); KT.axisL(gAxG, gy, 50, 2);
    band.attr("d", d3.area().x((d, k) => x(k)).y0((d, k) => y(XH[k] - 2 * Math.sqrt(PP[k]))).y1((d, k) => y(XH[k] + 2 * Math.sqrt(PP[k])))(XH));
    dots.selectAll("circle").data(Z).join("circle").attr("r", 1.8).attr("fill", C.muted).attr("fill-opacity", .7)
      .attr("cx", (d, k) => x(k)).attr("cy", d => Math.max(14, Math.min(252, y(d))));
    tru.attr("d", d3.line().x((d, k) => x(k)).y(d => y(d))(X));
    est.attr("d", d3.line().x((d, k) => x(k)).y(d => y(d))(XH));
    gain.attr("d", d3.line().x((d, k) => x(k)).y(d => gy(d))(KK));
    const pinf = (Q + Math.sqrt(Q * Q + 4 * Q * R)) / 2, Kinf = pinf / (pinf + R);
    kinf.attr("y1", gy(Kinf)).attr("y2", gy(Kinf));
    const rm = (a, b) => Math.sqrt(d3.mean(a, (v, k) => (v - b[k]) ** 2));
    const pinfT = (tq + Math.sqrt(tq * tq + 4 * tq * tr)) / 2, KT_ = pinfT / (pinfT + tr);
    KT.$("kf-readout").innerHTML = `filter Q/R = ${KT.e2(Q / R)} → steady-state gain K∞ = <b>${KT.f3(Kinf)}</b> (optimal for the true noise: ${KT.f3(KT_)}) · final K = ${KT.f3(KK[N - 1])}` +
      `<br>RMSE vs truth: raw measurements ${KT.f3(rm(Z, X))} · Kalman estimate <b>${KT.f3(rm(XH, X))}</b> · the filter's own claimed σ at the end ${KT.f3(Math.sqrt(PP[N - 1]))}`;
  }
  KT.on(["kf-tq", "kf-tr", "kf-q", "kf-r"], draw);
  KT.$("kf-new").addEventListener("click", () => { seed++; draw(); });
  KT.$("kf-match").addEventListener("click", () => {
    ["q", "r"].forEach(k => { const src = KT.$("kf-t" + k), dst = KT.$("kf-" + k); dst.value = src.value; KT.$("kf-" + k + "-o").textContent = src.value; });
    draw();
  });
  draw();
})();

/* ══════════ 2. EKF vs SGD on a tiny network ══════════ */
(function () {
  const svg = d3.select("#es-svg"); if (svg.empty()) return;
  const r0 = KT.rng(7), NS = 100, XS = [], TS = [], FS = [];
  for (let i = 0; i < NS; i++) { const x = -3 + 6 * i / (NS - 1), f = 0.8 * Math.sin(1.6 * x); XS.push(x); FS.push(f); TS.push(f + 0.05 * KT.gauss(r0)); }
  const floor = d3.mean(TS, (t, i) => (t - FS[i]) ** 2);
  const METH = [
    { k: "sgd", name: "SGD", col: C.bad },
    { k: "global", name: "GEKF", col: C.good },
    { k: "node", name: "NDEKF", col: C.A },
    { k: "full", name: "fully decoupled", col: C.B }];
  let H = 6, M = 19, S = {}, timer = null;

  const fwd = (w, x, h) => {
    let y = w[3 * H];
    for (let j = 0; j < H; j++) {
      const z = Math.tanh(w[j] * x + w[H + j]); y += w[2 * H + j] * z;
      if (h) { const d = w[2 * H + j] * (1 - z * z); h[j] = d * x; h[H + j] = d; h[2 * H + j] = z; }
    }
    if (h) h[3 * H] = 1;
    return y;
  };
  const mse = w => d3.mean(XS, (x, i) => (TS[i] - fwd(w, x)) ** 2);
  function groups(kind) {
    if (kind === "global") return [d3.range(M)];
    if (kind === "node") { const g = d3.range(H).map(j => [j, H + j]); g.push(d3.range(H).map(j => 2 * H + j).concat([3 * H])); return g; }
    return d3.range(M).map(i => [i]);
  }
  function init() {
    if (timer) { timer.stop(); timer = null; }
    H = KT.val("es-h"); M = 3 * H + 1;
    const r = KT.rng(3), w0 = new Float64Array(M);
    for (let j = 0; j < H; j++) { w0[j] = KT.gauss(r); w0[H + j] = 0.5 * KT.gauss(r); w0[2 * H + j] = 0.5 * KT.gauss(r); }
    S = {};
    METH.forEach(m => {
      const st = { w: Float64Array.from(w0), loss: [mse(w0)], ms: [], ep: 0, rng: KT.rng(11) };
      if (m.k !== "sgd") {
        st.g = groups(m.k);
        st.P = st.g.map(g => { const n = g.length, P = new Float64Array(n * n); for (let i = 0; i < n; i++) P[i * n + i] = 100; return P; });
      }
      S[m.k] = st;
    });
    draw();
  }
  const h = () => new Float64Array(M);
  function epoch(kind) {
    const st = S[kind], hv = h(), t0 = performance.now();
    const perm = d3.shuffler(st.rng)(d3.range(NS));
    const lr = KT.val("es-lr"), Rv = 1 / KT.val("es-eta"), q = Math.max(1e-6, Math.pow(10, KT.val("es-q")) * Math.pow(0.9, st.ep));
    for (const i of perm) {
      const y = fwd(st.w, XS[i], hv), xi = TS[i] - y;
      if (kind === "sgd") { for (let a = 0; a < M; a++) st.w[a] += lr * xi * hv[a]; continue; }
      let s = Rv; const PH = [];
      st.g.forEach((g, gi) => {
        const n = g.length, P = st.P[gi], v = new Float64Array(n);
        for (let a = 0; a < n; a++) { let acc = 0; for (let b = 0; b < n; b++) acc += P[a * n + b] * hv[g[b]]; v[a] = acc; s += hv[g[a]] * acc; }
        PH.push(v);
      });
      const A = 1 / s;
      st.g.forEach((g, gi) => {
        const n = g.length, P = st.P[gi], v = PH[gi];
        for (let a = 0; a < n; a++) st.w[g[a]] += v[a] * A * xi;
        for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) P[a * n + b] -= v[a] * v[b] * A;
        for (let a = 0; a < n; a++) P[a * n + a] += q;
      });
    }
    st.ms.push(performance.now() - t0);
    st.ep++; st.loss.push(mse(st.w));
  }
  // layout
  const lx = d3.scaleLinear([0, 60], [60, 380]), ly = d3.scaleLog([1e-3, 1], [320, 20]);
  KT.axisB(svg, lx, 320, 6); KT.axisL(svg, ly, 60, 4, "~e");
  KT.txt(svg, 380, 346, "epoch", C.muted, 10.5, "end"); KT.txt(svg, 66, 14, "training MSE (log)", C.muted, 10.5);
  svg.append("line").attr("x1", 60).attr("x2", 380).attr("y1", ly(floor)).attr("y2", ly(floor)).attr("stroke", C.muted).attr("stroke-dasharray", "5 4");
  KT.txt(svg, 376, ly(floor) - 5, "noise floor", C.muted, 10, "end");
  const fx = d3.scaleLinear([-3, 3], [440, 740]), fy = d3.scaleLinear([-1.3, 1.3], [320, 20]);
  KT.axisB(svg, fx, 320, 6); KT.axisL(svg, fy, 440, 5);
  svg.append("g").selectAll("circle").data(XS).join("circle").attr("r", 2).attr("fill", C.ink).attr("fill-opacity", .45).attr("cx", d => fx(d)).attr("cy", (d, i) => fy(TS[i]));
  const lossG = svg.append("g"), fitG = svg.append("g");
  KT.legend(svg.append("g"), 250, 36, METH.map(m => [m.name, m.col]));
  const on = k => KT.$("es-m-" + k).checked;

  function draw() {
    const act = METH.filter(m => on(m.k));
    lossG.selectAll("path").data(act, m => m.k).join("path").attr("fill", "none").attr("stroke", m => m.col).attr("stroke-width", 2)
      .attr("d", m => d3.line().x((v, i) => lx(i)).y(v => ly(Math.max(1e-3, Math.min(1, v))))(S[m.k].loss));
    const xs = d3.range(-3, 3.001, 0.05);
    fitG.selectAll("path").data(act, m => m.k).join("path").attr("fill", "none").attr("stroke", m => m.col).attr("stroke-width", 1.8)
      .attr("d", m => d3.line().x(x => fx(x)).y(x => fy(Math.max(-1.3, Math.min(1.3, fwd(S[m.k].w, x)))))(xs));
    const nd = groups("node").reduce((a, g) => a + g.length * g.length, 0);
    KT.$("es-readout").innerHTML = `M = ${M} weights · P entries: GEKF ${M * M}, NDEKF ${nd}, fully decoupled ${M} · noise floor ${KT.e2(floor)}<br>` +
      METH.map(m => { const st = S[m.k]; const ms = st.ms.length ? d3.mean(st.ms) : NaN; return `${m.name}: epoch ${st.ep}, MSE <b>${KT.e2(st.loss[st.loss.length - 1])}</b>${isFinite(ms) ? `, ${ms.toFixed(2)} ms/epoch` : ""}`; }).join(" · ");
  }
  function run() {
    if (timer) { timer.stop(); timer = null; return; }
    timer = d3.interval(() => {
      let busy = false;
      METH.forEach(m => { if (on(m.k) && S[m.k].ep < 60) { epoch(m.k); epoch(m.k); busy = true; } });
      draw();
      if (!busy && timer) { timer.stop(); timer = null; }
    }, 40);
  }
  KT.$("es-run").addEventListener("click", run);
  KT.$("es-reset").addEventListener("click", init);
  KT.on(["es-h"], init);
  KT.on(["es-lr", "es-eta", "es-q"], () => {});
  ["sgd", "global", "node", "full"].forEach(k => KT.$("es-m-" + k).addEventListener("change", draw));
  init();
})();

/* ══════════ 3. covariance ellipse shrinking ══════════ */
(function () {
  const svg = d3.select("#el-svg"); if (svg.empty()) return;
  const MODELS = {
    lin: { w: [1.5, -0.5], f: (w, u) => w[0] * u + w[1], J: (w, u) => [u, 1], u: r => 0.5 + 2 * r(), noise: 0.3 },
    tanh: { w: [1.2, -0.4], f: (w, u) => Math.tanh(w[0] * u + w[1]), J: (w, u) => { const z = Math.tanh(w[0] * u + w[1]), d = 1 - z * z; return [d * u, d]; }, u: r => -1 + 4 * r(), noise: 0.1 }
  };
  const NS = 200;
  const wx = d3.scaleLinear([-1.5, 3.5], [40, 380]), wy = d3.scaleLinear([-2.5, 2.5], [360, 20]);
  const clipId = "el-clip";
  svg.append("defs").append("clipPath").attr("id", clipId).append("rect").attr("x", 40).attr("y", 20).attr("width", 340).attr("height", 340);
  const cont = svg.append("g").attr("clip-path", `url(#${clipId})`);
  const ells = svg.append("g").attr("clip-path", `url(#${clipId})`);
  const pathL = svg.append("path").attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 1.4).attr("clip-path", `url(#${clipId})`);
  const dir = svg.append("line").attr("stroke", C.B).attr("stroke-width", 2).attr("clip-path", `url(#${clipId})`);
  const star = svg.append("path").attr("d", d3.symbol(d3.symbolCross, 90)()).attr("fill", C.good);
  const cur = svg.append("circle").attr("r", 4.5).attr("fill", C.A).attr("stroke", "#fff");
  KT.axisB(svg, wx, 360, 5); KT.axisL(svg, wy, 40, 5);
  KT.txt(svg, 380, 376, "w₁", C.muted, 11, "end"); KT.txt(svg, 46, 14, "w₂", C.muted, 11);
  const kx = d3.scaleLinear([0, NS], [460, 740]), ky = d3.scaleLog([1e-3, 100], [360, 20]);
  KT.axisB(svg, kx, 360, 4); KT.axisL(svg, ky, 460, 5, "~e");
  KT.txt(svg, 740, 376, "samples k", C.muted, 10.5, "end"); KT.txt(svg, 466, 14, "√ eigenvalues of P (log)", C.muted, 10.5);
  const e1 = svg.append("path").attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2);
  const e2 = svg.append("path").attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2);
  const ref = svg.append("path").attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "4 3");
  KT.legend(svg.append("g"), 600, 34, [["major axis", C.A], ["minor axis", C.B], ["∝ 1/√k", C.muted, 1.4, "4 3"]]);
  let D = null, st = null;

  function reset() {
    const md = MODELS[KT.$("el-model").value], r = KT.rng(21);
    D = { md, U: [], Y: [] };
    for (let k = 0; k < NS; k++) { const u = md.u(r); D.U.push(u); D.Y.push(md.f(md.w, u) + md.noise * KT.gauss(r)); }
    const p0 = Math.pow(10, KT.val("el-p0"));
    st = { w: [-0.5, 1.5], P: [p0, 0, p0], k: 0, hist: [], eig: [] };
    st.hist.push({ w: st.w.slice(), P: st.P.slice(), J: null });
    const ev = KT.eig2(st.P[0], st.P[1], st.P[2]); st.eig.push([Math.sqrt(ev[0][0]), Math.sqrt(ev[1][0])]);
    // loss contours over the dataset
    const n = 60, vals = new Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const w = [wx.invert(40 + (j + .5) * 340 / n), wy.invert(20 + (i + .5) * 340 / n)];
      let s = 0; for (let k = 0; k < NS; k++) s += (D.Y[k] - md.f(w, D.U[k])) ** 2;
      vals[i * n + j] = Math.log(s / NS);
    }
    const ext = d3.extent(vals), th = d3.range(12).map(i => ext[0] + (ext[1] - ext[0]) * (i + 0.5) / 12);
    const col = d3.scaleSequential(d3.interpolateViridis).domain([ext[1], ext[0]]);
    cont.selectAll("path").data(d3.contours().size([n, n]).thresholds(th)(vals)).join("path")
      .attr("d", d3.geoPath(d3.geoIdentity().scale(340 / n).translate([40, 20])))
      .attr("fill", d => col(d.value)).attr("fill-opacity", .22).attr("stroke", d => col(d.value)).attr("stroke-opacity", .6);
    star.attr("transform", `translate(${wx(md.w[0])},${wy(md.w[1])}) rotate(45)`);
    draw();
  }
  function step() {
    if (st.k >= NS) return;
    const md = D.md, u = D.U[st.k], y = D.Y[st.k], R = md.noise * md.noise, q = Math.pow(10, KT.val("el-q"));
    const J = md.J(st.w, u), [a, b, c] = st.P;
    const PH = [a * J[0] + b * J[1], b * J[0] + c * J[1]];
    const s = R + J[0] * PH[0] + J[1] * PH[1], K = [PH[0] / s, PH[1] / s], xi = y - md.f(st.w, u);
    st.w = [st.w[0] + K[0] * xi, st.w[1] + K[1] * xi];
    st.P = [a - K[0] * PH[0] + q, b - K[0] * PH[1], c - K[1] * PH[1] + q];
    st.k++;
    st.hist.push({ w: st.w.slice(), P: st.P.slice(), J });
    const ev = KT.eig2(st.P[0], st.P[1], st.P[2]); st.eig.push([Math.sqrt(ev[0][0]), Math.sqrt(Math.max(ev[1][0], 1e-12))]);
  }
  function ellPath(w, P) {
    const ev = KT.eig2(P[0], P[1], P[2]), a1 = 2 * Math.sqrt(ev[0][0]), a2 = 2 * Math.sqrt(ev[1][0]);
    const pts = d3.range(0, 2 * Math.PI + 0.01, Math.PI / 40).map(t => {
      const p0 = w[0] + a1 * Math.cos(t) * ev[0][1][0] + a2 * Math.sin(t) * ev[1][1][0];
      const p1 = w[1] + a1 * Math.cos(t) * ev[0][1][1] + a2 * Math.sin(t) * ev[1][1][1];
      return [wx(p0), wy(p1)];
    });
    return d3.line()(pts);
  }
  function draw() {
    const H = st.hist, show = H.slice(Math.max(0, H.length - 12));
    ells.selectAll("path").data(show).join("path").attr("fill", "none")
      .attr("stroke", (d, i) => i === show.length - 1 ? C.A : C.ink)
      .attr("stroke-width", (d, i) => i === show.length - 1 ? 2.4 : 1)
      .attr("stroke-opacity", (d, i) => i === show.length - 1 ? 1 : 0.12 + 0.5 * i / show.length)
      .attr("d", d => ellPath(d.w, d.P));
    pathL.attr("d", d3.line().x(d => wx(d.w[0])).y(d => wy(d.w[1]))(H));
    cur.attr("cx", wx(st.w[0])).attr("cy", wy(st.w[1]));
    const last = H[H.length - 1];
    if (last.J) {
      const nrm = Math.hypot(last.J[0], last.J[1]) || 1, L = 0.8;
      dir.attr("x1", wx(st.w[0] - L * last.J[0] / nrm)).attr("y1", wy(st.w[1] - L * last.J[1] / nrm))
        .attr("x2", wx(st.w[0] + L * last.J[0] / nrm)).attr("y2", wy(st.w[1] + L * last.J[1] / nrm)).attr("stroke-opacity", .9);
    } else dir.attr("stroke-opacity", 0);
    const E = st.eig;
    e1.attr("d", d3.line().x((d, i) => kx(i)).y(d => ky(Math.max(1e-3, Math.min(100, d[0]))))(E));
    e2.attr("d", d3.line().x((d, i) => kx(i)).y(d => ky(Math.max(1e-3, Math.min(100, d[1]))))(E));
    const c0 = E.length > 5 ? E[5][0] * Math.sqrt(5) : E[0][0];
    ref.attr("d", d3.line().x(k => kx(k)).y(k => ky(Math.max(1e-3, Math.min(100, c0 / Math.sqrt(k)))))(d3.range(1, NS + 1)));
    const md = D.md, P = st.P, corr = P[1] / Math.sqrt(P[0] * P[2]);
    KT.$("el-readout").innerHTML = `k = ${st.k} · ŵ = (${KT.f3(st.w[0])}, ${KT.f3(st.w[1])}) vs true (${md.w.join(", ")}) · √λ(P) = ${KT.f3(E[E.length - 1][0])}, ${KT.f3(E[E.length - 1][1])} · correlation of w₁, w₂ in P: <b>${KT.f2(corr)}</b>` +
      `<br>orange segment = direction of the latest sample's Jacobian Hₖ — the only direction that sample shrinks the ellipse along`;
  }
  KT.$("el-step").addEventListener("click", () => { step(); draw(); });
  KT.$("el-step10").addEventListener("click", () => { for (let i = 0; i < 10; i++) step(); draw(); });
  KT.$("el-reset").addEventListener("click", reset);
  KT.on(["el-model", "el-p0"], reset);
  KT.on(["el-q"], () => {});
  reset();
})();

/* ══════════ 4. decoupling structure ══════════ */
(function () {
  const svg = d3.select("#dc-svg"); if (svg.empty()) return;
  const LCOL = [C.A, C.good, C.B];
  const X0 = 30, Y0 = 30, SZ = 300;
  svg.append("rect").attr("x", X0).attr("y", Y0).attr("width", SZ).attr("height", SZ).attr("fill", "#11141b").attr("stroke", C.line);
  const blk = svg.append("g");
  KT.txt(svg, X0, 20, "P (M × M), kept blocks shaded", C.muted, 10.5);
  const bx = d3.scaleLog([10, 1e7], [470, 730]);
  const barG = svg.append("g"), axG = svg.append("g");
  const MODES = [["global", "GEKF"], ["layer", "layer"], ["node", "NDEKF"], ["full", "fully dec."]];
  function nodes(H) {       // list of nodes: {layer, fanin}
    const sz = [4, H, H, 2], out = [];
    for (let l = 1; l < sz.length; l++) for (let j = 0; j < sz[l]; j++) out.push({ layer: l - 1, fan: sz[l - 1] + 1 });
    return out;
  }
  function groupsFor(mode, nd) {
    const M = d3.sum(nd, n => n.fan);
    if (mode === "global") return [{ size: M, layer: -1 }];
    if (mode === "node") return nd.map(n => ({ size: n.fan, layer: n.layer }));
    if (mode === "layer") return [0, 1, 2].map(l => ({ size: d3.sum(nd.filter(n => n.layer === l), n => n.fan), layer: l }));
    const g = []; nd.forEach(n => { for (let i = 0; i < n.fan; i++) g.push({ size: 1, layer: n.layer }); }); return g;
  }
  function draw() {
    const H = KT.val("dc-h"), mode = KT.$("dc-mode").value, nd = nodes(H), M = d3.sum(nd, n => n.fan), No = 2;
    const g = groupsFor(mode, nd), sc = SZ / M;
    let off = 0; const rects = g.map(G => { const r = { o: off, s: G.size, l: G.layer }; off += G.size; return r; });
    blk.selectAll("rect").data(rects).join("rect").attr("x", d => X0 + d.o * sc).attr("y", d => Y0 + d.o * sc)
      .attr("width", d => Math.max(0.8, d.s * sc)).attr("height", d => Math.max(0.8, d.s * sc))
      .attr("fill", d => d.l < 0 ? C.A : LCOL[d.l]).attr("fill-opacity", .55).attr("stroke", d => d.l < 0 ? C.A : LCOL[d.l]).attr("stroke-width", .6);
    const stats = MODES.map(([k, name]) => {
      const gg = groupsFor(k, nd), store = d3.sum(gg, G => G.size * G.size);
      return { k, name, store, ops: No * No * M + No * store };
    });
    barG.selectAll("*").remove(); axG.selectAll("*").remove();
    KT.txt(barG, 470, 40, "stored entries of P", C.muted, 10.5);
    KT.txt(barG, 470, 205, "operations per sample ≈ Nₒ²M + Nₒ∑Mᵢ²", C.muted, 10.5);
    stats.forEach((s, i) => {
      [[s.store, 50], [s.ops, 215]].forEach(([v, y0]) => {
        const y = y0 + i * 30;
        barG.append("rect").attr("x", 470).attr("y", y).attr("height", 20).attr("rx", 3).attr("width", Math.max(2, bx(v) - 470))
          .attr("fill", s.k === mode ? C.B : C.muted).attr("fill-opacity", s.k === mode ? .85 : .35);
        KT.txt(barG, 466, y + 14, s.name, s.k === mode ? C.ink : C.muted, 10.5, "end");
        KT.txt(barG, Math.min(bx(v) + 4, 700), y + 14, d3.format(",")(v), C.ink, 10.5);
      });
    });
    KT.axisB(axG, bx, 172, 4, "~s"); KT.axisB(axG, bx, 337, 4, "~s");
    const cur = stats.find(s => s.k === mode), glob = stats[0];
    KT.$("dc-readout").innerHTML = `network 4-${H}-${H}-2 with biases: M = ${M} weights in ${nd.length} nodes (layer fan-ins 5, ${H + 1}, ${H + 1})` +
      `<br>${cur.name}: ${g.length} group${g.length > 1 ? "s" : ""}, stores <b>${d3.format(",")(cur.store)}</b> of P's ${d3.format(",")(M * M)} entries (${(100 * cur.store / (M * M)).toFixed(1)} %) · per-sample cost ${(glob.ops / cur.ops).toFixed(1)}× lower than GEKF`;
  }
  KT.on(["dc-mode", "dc-h"], draw); draw();
  const lg = svg.append("g");
  [["hidden layer 1", LCOL[0]], ["hidden layer 2", LCOL[1]], ["output layer", LCOL[2]]].forEach((d, i) => {
    lg.append("rect").attr("x", 30 + i * 105).attr("y", 342).attr("width", 10).attr("height", 10).attr("fill", d[1]).attr("fill-opacity", .7);
    KT.txt(lg, 44 + i * 105, 351, d[0], C.muted, 10.5);
  });
})();

/* ══════════ 5. tracking with process noise ══════════ */
(function () {
  const svg = d3.select("#qt-svg"); if (svg.empty()) return;
  const N = 500; let seed = 9;
  const x = d3.scaleLinear([0, N - 1], [50, 740]), y = d3.scaleLinear([-1.6, 1.6], [220, 20]), py = d3.scaleLog([1e-5, 100], [340, 250]);
  KT.axisB(svg, x, 220, 10); KT.axisL(svg, y, 50, 5); KT.axisB(svg, x, 340, 10); KT.axisL(svg, py, 50, 3, "~e");
  KT.txt(svg, 56, 16, "first weight w₁", C.muted, 10.5); KT.txt(svg, 56, 246, "trace P (log)", C.muted, 10.5);
  svg.append("line").attr("x1", x(200)).attr("x2", x(200)).attr("y1", 20).attr("y2", 340).attr("stroke", C.muted).attr("stroke-dasharray", "2 4");
  KT.txt(svg, x(200) + 4, 32, "target changes", C.muted, 10);
  const tru = svg.append("path").attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2);
  const a0 = svg.append("path").attr("fill", "none").attr("stroke", C.bad).attr("stroke-width", 1.8);
  const aq = svg.append("path").attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 1.8);
  const p0 = svg.append("path").attr("fill", "none").attr("stroke", C.bad).attr("stroke-width", 1.8);
  const pq = svg.append("path").attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 1.8);
  KT.legend(svg.append("g"), 480, 40, [["truth", "#fff"], ["EKF, q = 0", C.bad], ["EKF, chosen q", C.good]]);
  const trueW = k => k < 200 ? [1, -0.5] : [-0.5 + 0.5 * Math.sin((k - 200) / 50), 1];
  function run(q, U, Y, R) {
    let w = [0, 0], P = [10, 0, 10]; const W = [], T = [];
    for (let k = 0; k < N; k++) {
      const J = U[k], [a, b, c] = P, PH = [a * J[0] + b * J[1], b * J[0] + c * J[1]];
      const s = R + J[0] * PH[0] + J[1] * PH[1], K = [PH[0] / s, PH[1] / s], xi = Y[k] - (w[0] * J[0] + w[1] * J[1]);
      w = [w[0] + K[0] * xi, w[1] + K[1] * xi];
      P = [a - K[0] * PH[0] + q, b - K[0] * PH[1], c - K[1] * PH[1] + q];
      W.push(w); T.push(P[0] + P[2]);
    }
    return { W, T };
  }
  function draw() {
    const q = Math.pow(10, KT.val("qt-q")), ns = KT.val("qt-n"), r = KT.rng(seed), U = [], Y = [], TW = [];
    for (let k = 0; k < N; k++) { const u = [KT.gauss(r), KT.gauss(r)], w = trueW(k); U.push(u); TW.push(w); Y.push(w[0] * u[0] + w[1] * u[1] + ns * KT.gauss(r)); }
    const A = run(0, U, Y, ns * ns), B = run(q, U, Y, ns * ns);
    const L = d3.line().x((d, k) => x(k));
    tru.attr("d", L.y(d => y(d[0]))(TW));
    a0.attr("d", L.y(d => y(Math.max(-1.6, Math.min(1.6, d[0]))))(A.W));
    aq.attr("d", L.y(d => y(Math.max(-1.6, Math.min(1.6, d[0]))))(B.W));
    const Lp = d3.line().x((d, k) => x(k)).y(d => py(Math.max(1e-5, Math.min(100, d))));
    p0.attr("d", Lp(A.T)); pq.attr("d", Lp(B.T));
    const err = (R, a, b) => Math.sqrt(d3.mean(d3.range(a, b), k => (R.W[k][0] - TW[k][0]) ** 2 + (R.W[k][1] - TW[k][1]) ** 2));
    KT.$("qt-readout").innerHTML = `weight RMSE, samples 50–199 (stationary): q = 0 <b>${KT.f3(err(A, 50, 200))}</b> · q = ${KT.e2(q)} <b>${KT.f3(err(B, 50, 200))}</b>` +
      `<br>weight RMSE, samples 200–499 (after the change): q = 0 <b>${KT.f3(err(A, 200, N))}</b> · q = ${KT.e2(q)} <b>${KT.f3(err(B, 200, N))}</b> · final trace P: ${KT.e2(A.T[N - 1])} vs ${KT.e2(B.T[N - 1])}`;
  }
  KT.on(["qt-q", "qt-n"], draw);
  KT.$("qt-new").addEventListener("click", () => { seed++; draw(); });
  draw();
})();

/* ══════════ 6. linearization vs unscented transform ══════════ */
(function () {
  const svg = d3.select("#ut-svg"); if (svg.empty()) return;
  const F = {
    tanh: { f: Math.tanh, d: x => 1 - Math.tanh(x) ** 2, dom: [-1.3, 1.3] },
    sin: { f: Math.sin, d: Math.cos, dom: [-1.3, 1.3] },
    sq: { f: x => x * x / 2, d: x => x, dom: [-0.4, 4.4] }
  };
  const x = d3.scaleLinear([-4, 4], [60, 430]), y = d3.scaleLinear([-1.3, 1.3], [270, 20]), iy = d3.scaleLinear([0, 1], [365, 300]);
  KT.axisB(svg, x, 365, 8);
  const gAx = svg.append("g");
  KT.txt(svg, 430, 294, "input density N(m, s²) and sigma points", C.muted, 10.5, "end");
  svg.append("defs").append("clipPath").attr("id", "ut-clip").append("rect").attr("x", 60).attr("y", 20).attr("width", 370).attr("height", 250);
  const fcur = svg.append("path").attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 2).attr("clip-path", "url(#ut-clip)");
  const tan = svg.append("line").attr("clip-path", "url(#ut-clip)").attr("stroke", C.bad).attr("stroke-dasharray", "5 3").attr("stroke-width", 1.5);
  const inD = svg.append("path").attr("fill", C.muted).attr("fill-opacity", .3).attr("stroke", C.muted);
  const sig = svg.append("g"), sigLines = svg.append("g");
  const ox = d3.scaleLinear([0, 1], [470, 740]);
  svg.append("line").attr("x1", 470).attr("x2", 470).attr("y1", 20).attr("y2", 270).attr("stroke", C.line);
  KT.txt(svg, 476, 14, "output density (along the vertical axis)", C.muted, 10.5);
  const hist = svg.append("path").attr("fill", C.ink).attr("fill-opacity", .22).attr("stroke", C.ink).attr("stroke-opacity", .6);
  const gLin = svg.append("path").attr("fill", "none").attr("stroke", C.bad).attr("stroke-width", 2);
  const gUt = svg.append("path").attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 2);
  const mk = [C.ink, C.bad, C.good].map(c => svg.append("line").attr("x1", 462).attr("x2", 740).attr("stroke", c).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3"));
  KT.legend(svg.append("g"), 600, 300, [["true", C.ink], ["linearized (EKF)", C.bad], ["unscented", C.good]]);

  function draw() {
    const fk = KT.$("ut-f").value, fn = F[fk], m = KT.val("ut-m"), s = KT.val("ut-s"), kap = KT.val("ut-k");
    y.domain(fn.dom); gAx.selectAll("*").remove(); KT.axisL(gAx, y, 60, 5);
    const xs = d3.range(-4, 4.001, 0.02);
    fcur.attr("d", d3.line().x(t => x(t)).y(t => y(Math.max(fn.dom[0] - 1, Math.min(fn.dom[1] + 1, fn.f(t)))))(xs));
    const fm = fn.f(m), dm = fn.d(m);
    tan.attr("x1", x(-4)).attr("x2", x(4)).attr("y1", y(fm + dm * (-4 - m))).attr("y2", y(fm + dm * (4 - m)));
    const pdf = t => Math.exp(-((t - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
    const pk = pdf(m);
    inD.attr("d", d3.area().x(t => x(t)).y0(iy(0)).y1(t => iy(pdf(t) / pk))(xs));
    // true moments by quadrature + histogram of the output
    const NQ = 4000, bins = 90, [ylo, yhi] = fn.dom, hb = new Array(bins).fill(0);
    let z = 0, sm = 0, s2 = 0;
    for (let i = 0; i < NQ; i++) {
      const u = -6 + 12 * (i + .5) / NQ, w = Math.exp(-u * u / 2), v = fn.f(m + s * u);
      z += w; sm += w * v; s2 += w * v * v;
      const b = Math.floor((v - ylo) / (yhi - ylo) * bins); if (b >= 0 && b < bins) hb[b] += w;
    }
    const tm = sm / z, ts = Math.sqrt(Math.max(0, s2 / z - tm * tm));
    const bw = (yhi - ylo) / bins, hd = hb.map(v => v / z / bw);
    // unscented transform (1-D)
    const L = Math.sqrt((1 + kap) * s * s), P = [m, m + L, m - L], W = [kap / (1 + kap), .5 / (1 + kap), .5 / (1 + kap)];
    const Yp = P.map(fn.f), um = d3.sum(Yp, (v, i) => W[i] * v), us = Math.sqrt(Math.max(0, d3.sum(Yp, (v, i) => W[i] * (v - um) ** 2)));
    const ls = Math.abs(dm) * s;
    const gpdf = (t, mu, sd) => sd < 1e-6 ? 0 : Math.exp(-((t - mu) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI));
    const ys = d3.range(ylo, yhi + 1e-9, (yhi - ylo) / 300);
    const hq = d3.quantile(hd.filter(v => v > 0).sort(d3.ascending), 0.85) || 0;
    const peak = Math.max(hq, gpdf(um, um, us), ls > 1e-6 ? Math.min(gpdf(fm, fm, ls), 2 * gpdf(um, um, us)) : 0) || 1;
    ox.domain([0, peak * 1.05]);
    hist.attr("d", d3.area().curve(d3.curveStepAfter).y((v, i) => y(ylo + i * bw)).x0(ox(0)).x1(v => Math.min(740, ox(v)))(hd));
    gLin.attr("d", ls > 1e-6 ? d3.line().y(t => y(t)).x(t => Math.min(740, ox(gpdf(t, fm, ls))))(ys) : `M470,${y(fm)}L740,${y(fm)}`);
    gUt.attr("d", d3.line().y(t => y(t)).x(t => Math.min(740, ox(gpdf(t, um, us))))(ys));
    [tm, fm, um].forEach((v, i) => mk[i].attr("y1", y(v)).attr("y2", y(v)));
    sig.selectAll("circle").data(P).join("circle").attr("r", (d, i) => i ? 4 : 5.5).attr("fill", C.good).attr("stroke", "#0b0e14")
      .attr("cx", d => x(d)).attr("cy", iy(0.02));
    sigLines.selectAll("line").data(P).join("line").attr("stroke", C.good).attr("stroke-opacity", .45).attr("stroke-dasharray", "2 3")
      .attr("x1", d => x(d)).attr("x2", d => x(d)).attr("y1", iy(0.02)).attr("y2", d => y(Math.max(ylo, Math.min(yhi, fn.f(d)))));
    sigLines.selectAll("circle").data(P).join("circle").attr("r", 3.5).attr("fill", C.good)
      .attr("cx", d => x(d)).attr("cy", d => y(Math.max(ylo, Math.min(yhi, fn.f(d)))));
    KT.$("ut-readout").innerHTML = `true: mean <b>${KT.f3(tm)}</b>, std <b>${KT.f3(ts)}</b> · linearized: mean ${KT.f3(fm)} (error ${KT.f3(fm - tm)}), std ${KT.f3(ls)} · unscented: mean ${KT.f3(um)} (error ${KT.f3(um - tm)}), std ${KT.f3(us)}` +
      `<br>sigma points ${P.map(KT.f2).join(", ")} with weights ${W.map(KT.f3).join(", ")} — three function evaluations, no derivative`;
  }
  KT.on(["ut-f", "ut-m", "ut-s", "ut-k"], draw); draw();
})();
