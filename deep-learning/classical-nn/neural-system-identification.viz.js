/* neural-system-identification.viz.js — the figures on
   deep-learning/classical-nn/neural-system-identification.html
   Loaded after ../../data.js → ../../notes.js (palette C).
   Each block is an IIFE that returns quietly if its <svg> is absent.

     1  #mc-svg     the four model classes as regressor diagrams (NFIR / NARX / NOE / NARMAX)
     2  #narx-svg   identify a nonlinear plant: series-parallel vs parallel training,
                    one-step vs free-run prediction, noise type, excitation amplitude
     3  #ord-svg    lag / order selection: Lipschitz-quotient index over (ny, nu),
                    free-run at the chosen order, residual autocorrelation test
     4  #fc-svg     multi-step forecasting: recursive (iterated) vs direct strategy
     5  #ctl-svg    one control loop, four neural controllers: direct inverse,
                    model-reference (online), NARMA-L2, neural MPC

   Every number a readout prints is computed from the data the figure draws.   */

const SI = (function () {
  /* seeded uniform RNG (mulberry32) and a Box–Muller normal */
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0; while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }

  /* ── a one-hidden-layer tanh MLP with a linear output, flat parameter vector ──
     layout: W1 [nh × nin] | b1 [nh] | W2 [nh] | b2 [1]                          */
  function net(nin, nh, r, gain) {
    const n = nh * nin + nh + nh + 1;
    const P = new Float64Array(n);
    const s = (gain || 1) / Math.sqrt(nin);
    for (let i = 0; i < nh * nin; i++) P[i] = gauss(r) * s;
    for (let i = 0; i < nh; i++) P[nh * nin + nh + i] = gauss(r) / Math.sqrt(nh) * 0.5;
    return { nin, nh, P, G: new Float64Array(n), M: new Float64Array(n), V: new Float64Array(n), t: 0 };
  }
  function fwd(N, x, h) {
    const { nin, nh, P } = N, ob = nh * nin, ow = ob + nh;
    let y = P[ow + nh];
    for (let i = 0; i < nh; i++) {
      let a = P[ob + i];
      for (let j = 0; j < nin; j++) a += P[i * nin + j] * x[j];
      const t = Math.tanh(a);
      h[i] = t;
      y += P[ow + i] * t;
    }
    return y;
  }
  /* accumulate dL/dP given dL/dy = gy; if gx is supplied, write dL/dx into it */
  function bwd(N, x, h, gy, gx) {
    const { nin, nh, P, G } = N, ob = nh * nin, ow = ob + nh;
    G[ow + nh] += gy;
    if (gx) for (let j = 0; j < nin; j++) gx[j] = 0;
    for (let i = 0; i < nh; i++) {
      G[ow + i] += gy * h[i];
      const d = gy * P[ow + i] * (1 - h[i] * h[i]);
      G[ob + i] += d;
      for (let j = 0; j < nin; j++) {
        G[i * nin + j] += d * x[j];
        if (gx) gx[j] += d * P[i * nin + j];
      }
    }
  }
  function adam(N, lr, clip) {
    const { P, G, M, V } = N;
    if (clip) {
      let s = 0; for (let i = 0; i < G.length; i++) s += G[i] * G[i];
      s = Math.sqrt(s);
      if (s > clip) for (let i = 0; i < G.length; i++) G[i] *= clip / s;
    }
    N.t++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, N.t), c2 = 1 - Math.pow(b2, N.t);
    for (let i = 0; i < P.length; i++) {
      M[i] = b1 * M[i] + (1 - b1) * G[i];
      V[i] = b2 * V[i] + (1 - b2) * G[i] * G[i];
      P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8);
      G[i] = 0;
    }
  }
  function clone(N) {
    return { nin: N.nin, nh: N.nh, P: Float64Array.from(N.P), G: new Float64Array(N.P.length),
      M: new Float64Array(N.P.length), V: new Float64Array(N.P.length), t: 0 };
  }

  /* the benchmark plant: second order in y, first order in u, strongly nonlinear
       y(k) = y(k−1)·y(k−2)·(y(k−1)+2.5) / (1 + y(k−1)² + y(k−2)²) + u(k−1)          */
  const plant = (y1, y2, u1) => y1 * y2 * (y1 + 2.5) / (1 + y1 * y1 + y2 * y2) + u1;
  /* the classic validation input */
  const testU = k => (k < 250 ? Math.sin(2 * Math.PI * k / 25)
                             : 0.8 * Math.sin(2 * Math.PI * k / 25) + 0.2 * Math.sin(2 * Math.PI * k / 10));

  /* simulate with noise.  type 'output': measured = true + e;  'equation': e enters the recursion */
  function simulate(u, sigma, type, r) {
    const n = u.length, yt = new Float64Array(n), ym = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const e = sigma > 0 ? sigma * gauss(r) : 0;
      if (k < 2) { yt[k] = 0; ym[k] = type === "output" ? e : 0; continue; }
      if (type === "equation") {
        yt[k] = plant(yt[k - 1], yt[k - 2], u[k - 1]) + e;
        ym[k] = yt[k];
      } else {
        yt[k] = plant(yt[k - 1], yt[k - 2], u[k - 1]);
        ym[k] = yt[k] + e;
      }
    }
    return { yt, ym };
  }
  /* noiseless deterministic response to an input */
  function clean(u) {
    const y = new Float64Array(u.length);
    for (let k = 2; k < u.length; k++) y[k] = plant(y[k - 1], y[k - 2], u[k - 1]);
    return y;
  }

  const rmse = (a, b, from) => {
    let s = 0, c = 0;
    for (let k = from || 0; k < a.length; k++) { const d = a[k] - b[k]; if (isFinite(d)) { s += d * d; c++; } }
    return c ? Math.sqrt(s / c) : NaN;
  };
  const fmt = (v, d) => (isFinite(v) ? v.toFixed(d == null ? 3 : d) : "—");

  /* drawing helpers */
  function panel(svg, x, y, w, h) {
    return svg.append("g").attr("transform", `translate(${x},${y})`).datum({ w, h });
  }
  function axes(g, xs, ys, w, h, xl, yl, nx, ny) {
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xs).ticks(nx || 6));
    g.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(ny || 4));
    if (xl) g.append("text").attr("x", w).attr("y", h + 28).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 10.5).text(xl);
    if (yl) g.append("text").attr("x", 0).attr("y", -7).attr("fill", C.muted).attr("font-size", 10.5).text(yl);
  }
  function path(g, xs, ys, arr, color, width, dash, from) {
    const pts = [];
    for (let k = from || 0; k < arr.length; k++) if (isFinite(arr[k])) pts.push([xs(k), ys(Math.max(ys.domain()[0], Math.min(ys.domain()[1], arr[k])))]);
    g.append("path").attr("d", d3.line()(pts)).attr("fill", "none").attr("stroke", color)
      .attr("stroke-width", width || 1.6).attr("stroke-dasharray", dash || null);
  }
  function legend(g, x, y, items) {
    let off = 0;
    items.forEach(it => {
      const gg = g.append("g").attr("transform", `translate(${x + off},${y})`);
      off += it.w || 120;
      gg.append("line").attr("x1", 0).attr("x2", 18).attr("y1", 0).attr("y2", 0).attr("stroke", it.c)
        .attr("stroke-width", 2.2).attr("stroke-dasharray", it.d || null);
      gg.append("text").attr("x", 23).attr("y", 4).attr("fill", C.muted).attr("font-size", 10.5).text(it.t);
    });
  }
  /* Gaussian elimination with partial pivoting, solves A x = b (A is n×n, row-major) */
  function solve(A, b, n) {
    const M = Float64Array.from(A), x = Float64Array.from(b);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
      if (p !== c) {
        for (let j = 0; j < n; j++) { const t = M[c * n + j]; M[c * n + j] = M[p * n + j]; M[p * n + j] = t; }
        const t = x[c]; x[c] = x[p]; x[p] = t;
      }
      const d = M[c * n + c] || 1e-12;
      for (let r = c + 1; r < n; r++) {
        const f = M[r * n + c] / d;
        if (f === 0) continue;
        for (let j = c; j < n; j++) M[r * n + j] -= f * M[c * n + j];
        x[r] -= f * x[c];
      }
    }
    for (let r = n - 1; r >= 0; r--) {
      let s = x[r];
      for (let j = r + 1; j < n; j++) s -= M[r * n + j] * x[j];
      x[r] = s / (M[r * n + r] || 1e-12);
    }
    return x;
  }

  return { rng, gauss, net, fwd, bwd, adam, clone, plant, testU, simulate, clean, rmse, fmt, panel, axes, path, legend, solve };
})();

/* ═══════════════ 1 · #mc-svg — the model classes as regressor diagrams ═══════════════ */
(function () {
  const svg = d3.select("#mc-svg");
  if (svg.empty()) return;
  const W = 760, H = 350;
  const sel = document.getElementById("mc-class"),
    su = document.getElementById("mc-nu"), sy = document.getElementById("mc-ny");
  const RO = document.getElementById("mc-readout");

  const INFO = {
    nfir: { rows: ["u"], eq: "ŷ(k) = f( u(k−1), …, u(k−nᵤ) )",
      train: "static backprop — the regressor is fixed data",
      noise: "no output feedback at all: stable by construction, but needs a long input window to capture slow dynamics" },
    narx: { rows: ["u", "y"], eq: "ŷ(k) = f( u(k−1), …, u(k−nᵤ), y(k−1), …, y(k−n_y) )",
      train: "static backprop — measured y is data, so the regressor does not depend on the weights",
      noise: "the optimal one-step predictor when noise enters as equation error: y(k) = f(…) + e(k)" },
    noe: { rows: ["u", "yhat"], eq: "ŷ(k) = f( u(k−1), …, u(k−nᵤ), ŷ(k−1), …, ŷ(k−n_y) )",
      train: "dynamic backprop (BPTT / RTRL) — past outputs are the model's own, so they depend on the weights",
      noise: "the right structure when noise is pure measurement noise on the output: y(k) = y₀(k) + e(k)" },
    narmax: { rows: ["u", "y", "eps"], eq: "ŷ(k) = f( u(k−1…k−nᵤ), y(k−1…k−n_y), ε(k−1), …, ε(k−n_e) ),   ε = y − ŷ",
      train: "dynamic — past residuals depend on the weights through ŷ, so the gradient must flow through time",
      noise: "the general case: coloured noise is modelled by the MA terms in past prediction errors" }
  };
  const ROWS = {
    u: { label: "u — plant input", sym: "u", color: C.A, y: 70 },
    y: { label: "y — measured plant output", sym: "y", color: C.good, y: 140 },
    yhat: { label: "ŷ — model's own past outputs", sym: "ŷ", color: C.B, y: 210 },
    eps: { label: "ε — past residuals y − ŷ", sym: "ε", color: C.bad, y: 280 }
  };

  function draw() {
    const cls = sel.value, nu = +su.value, ny = +sy.value;
    document.getElementById("mc-nu-out").textContent = nu;
    document.getElementById("mc-ny-out").textContent = ny;
    const info = INFO[cls];
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    ["A", "m"].forEach(k => defs.append("marker").attr("id", "mc-arr-" + k).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", k === "A" ? C.ink : C.line));
    const NX0 = 560, NX1 = 660, NY0 = 45, NY1 = 305;
    let nIn = 0;
    Object.keys(ROWS).forEach(key => {
      const R = ROWS[key], on = info.rows.includes(key);
      const n = key === "u" ? nu : ny;
      svg.append("text").attr("x", 14).attr("y", R.y - 20).attr("fill", on ? R.color : C.line).attr("font-size", 11.5)
        .attr("font-weight", 600).text(R.label + (on ? "" : "  (not used)"));
      for (let i = 1; i <= 4; i++) {
        const x = 150 + (i - 1) * 88, active = on && i <= n;
        svg.append("rect").attr("x", x).attr("y", R.y - 12).attr("width", 74).attr("height", 26).attr("rx", 5)
          .attr("fill", active ? "rgba(91,156,255,.07)" : "none").attr("stroke", active ? R.color : C.line)
          .attr("stroke-dasharray", active ? null : "3,3");
        svg.append("text").attr("x", x + 37).attr("y", R.y + 5).attr("text-anchor", "middle").attr("font-size", 11.5)
          .attr("fill", active ? C.ink : C.line).text(`${R.sym}(k−${i})`);
        if (i > 1) svg.append("text").attr("x", x - 7).attr("y", R.y + 4).attr("text-anchor", "middle").attr("font-size", 9)
          .attr("fill", C.line).text("z⁻¹");
        if (active) {
          nIn++;
          svg.append("path").attr("d", `M${x + 74},${R.y} C ${x + 130},${R.y} ${NX0 - 60},${(NY0 + NY1) / 2} ${NX0},${(NY0 + NY1) / 2 + (R.y - 175) * 0.5}`)
            .attr("fill", "none").attr("stroke", R.color).attr("stroke-opacity", 0.45).attr("stroke-width", 1.1);
        }
      }
      svg.append("line").attr("x1", 118).attr("x2", 148).attr("y1", R.y).attr("y2", R.y).attr("stroke", on ? R.color : C.line)
        .attr("marker-end", "url(#mc-arr-" + (on ? "A" : "m") + ")");
    });
    // the network
    svg.append("rect").attr("x", NX0).attr("y", NY0).attr("width", NX1 - NX0).attr("height", NY1 - NY0).attr("rx", 10)
      .attr("fill", "rgba(255,180,84,.06)").attr("stroke", C.B);
    svg.append("text").attr("x", (NX0 + NX1) / 2).attr("y", 165).attr("text-anchor", "middle").attr("fill", C.ink).attr("font-size", 13).attr("font-weight", 600).text("static MLP");
    svg.append("text").attr("x", (NX0 + NX1) / 2).attr("y", 184).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 11).text(`f( · ), ${nIn} inputs`);
    svg.append("line").attr("x1", NX1).attr("x2", 735).attr("y1", 175).attr("y2", 175).attr("stroke", C.ink).attr("marker-end", "url(#mc-arr-A)");
    svg.append("text").attr("x", 700).attr("y", 166).attr("text-anchor", "middle").attr("fill", C.ink).attr("font-size", 12).text("ŷ(k)");
    // feedback path for model-generated regressors
    const fb = info.rows.filter(k => k === "yhat" || k === "eps");
    if (fb.length) {
      const yb = 335;
      svg.append("path").attr("d", `M700,175 L700,${yb} L40,${yb} L40,${ROWS[fb[0]].y + 18} L118,${ROWS[fb[0]].y + 18}`)
        .attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.6).attr("stroke-dasharray", "5,4")
        .attr("marker-end", "url(#mc-arr-A)");
      svg.append("text").attr("x", 380).attr("y", yb - 6).attr("text-anchor", "middle").attr("fill", C.B).attr("font-size", 11)
        .text(cls === "noe" ? "feedback: the model's own output re-enters its input" : "feedback: residual ε = y − ŷ needs the model's own output");
    } else {
      svg.append("text").attr("x", 380).attr("y", 335).attr("text-anchor", "middle").attr("fill", C.good).attr("font-size", 11)
        .text(cls === "narx" ? "no model feedback: every input is measured data (series-parallel)" : "no feedback of any kind: a finite input window");
    }
    const nP = nIn * 10 + 10 + 10 + 1;
    RO.innerHTML = `<b>${sel.options[sel.selectedIndex].text}</b>: ${info.eq}<br>` +
      `regressor length <b>${nIn}</b> → an MLP with 10 hidden units has <b>${nP}</b> parameters<br>` +
      `training: ${info.train}<br>noise assumption: ${info.noise}`;
  }
  [sel, su, sy].forEach(e => e.addEventListener("input", draw));
  sel.addEventListener("change", draw);
  draw();
})();

/* ═══════════════ 2 · #narx-svg — series-parallel vs parallel identification ═══════════════ */
(function () {
  const svg = d3.select("#narx-svg");
  if (svg.empty()) return;
  const W = 760, H = 430, NH = 10, NIN = 4, NTR = 1000, NTE = 400;
  const noiseS = document.getElementById("narx-noise"), typeS = document.getElementById("narx-ntype"),
    ampS = document.getElementById("narx-amp"), RO = document.getElementById("narx-readout");
  /* the plant: a lightly damped hardening-spring (Duffing) oscillator, sampled every 0.3 s
       x″ + 2ζω x′ + ω² x + β x³ = ω² u,   ζ = 0.08, ω = 1, β = 0.3,   zero-order-hold input, RK4 inside a sample
     about 21 samples per period: consecutive outputs are close, which is exactly when
     a one-step predictor looks good for the wrong reason                                        */
  const DT = 0.3, ZE = 0.08, BE = 0.3;
  function step(x, v, u) {
    const f = (x, v) => [v, -2 * ZE * v - x - BE * x * x * x + u], h = DT / 4;
    for (let i = 0; i < 4; i++) {
      const a = f(x, v), b = f(x + h / 2 * a[0], v + h / 2 * a[1]), c = f(x + h / 2 * b[0], v + h / 2 * b[1]), d = f(x + h * c[0], v + h * c[1]);
      x += h / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]); v += h / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1]);
    }
    return [x, v];
  }
  /* type 'output': y = x + e (measurement noise);  'equation': e kicks the state, y = x exactly */
  function sim(u, sig, type, r) {
    const n = u.length, y0 = new Float64Array(n), y = new Float64Array(n);
    let x = 0, v = 0;
    for (let k = 0; k < n; k++) {
      const e = sig > 0 && r ? sig * SI.gauss(r) : 0;
      if (type === "equation") { x += 0.5 * e; y0[k] = x; y[k] = x; }
      else { y0[k] = x; y[k] = x + e; }
      [x, v] = step(x, v, u[k]);
    }
    return { y0, y };
  }
  const phi = (y, u, k, x) => { x[0] = y[k - 1]; x[1] = y[k - 2]; x[2] = u[k - 1]; x[3] = u[k - 2]; return x; };
  let D, N, hist, timer = null, mode = "idle", step_ = 0;

  function data() {
    const r = SI.rng(7), sig = +noiseS.value, A = +ampS.value, type = typeS.value;
    const u = new Float64Array(NTR);
    let k = 0;
    while (k < NTR) { const lv = (2 * r() - 1) * A, d = 2 + Math.floor(r() * 15); for (let j = 0; j < d && k < NTR; j++) u[k++] = lv; }   // APRBS
    const tr = sim(u, sig, type, r);
    const ut = new Float64Array(NTE);
    for (let k = 0; k < NTE; k++) ut[k] = k < 200 ? 0.8 * Math.sign(Math.sin(2 * Math.PI * k / 60)) : 1.2 * Math.sin(2 * Math.PI * k / 45);
    const te = sim(ut, sig, type, SI.rng(99));
    // the noise-free response to the validation input (for 'equation' noise, the deterministic part)
    const cl = sim(ut, 0, "output", null);
    D = { u, y: tr.y, ut, yte: te.y, ytrue: cl.y0, sig, type, A };
  }
  function reset() {
    stop(); data();
    N = SI.net(NIN, NH, SI.rng(3), 1);
    hist = []; step_ = 0; mode = "idle";
    evaluate(); draw();
  }
  function evaluate() {
    const h = new Float64Array(NH), x = new Float64Array(NIN);
    const one = new Float64Array(NTE).fill(NaN), free = new Float64Array(NTE).fill(NaN);
    free[0] = D.ytrue[0]; free[1] = D.ytrue[1];
    let div = false, s1 = 0, s0 = 0, c = 0;
    for (let k = 2; k < NTE; k++) {
      one[k] = SI.fwd(N, phi(D.yte, D.ut, k, x), h);
      s1 += (one[k] - D.yte[k]) ** 2; s0 += (D.yte[k - 1] - D.yte[k]) ** 2; c++;
      phi(free, D.ut, k, x);
      let v = SI.fwd(N, x, h);
      if (!isFinite(v) || Math.abs(v) > 6) { v = Math.sign(v || 1) * 6; div = true; }
      free[k] = v;
    }
    D.one = one; D.free = free; D.div = div;
    D.eOne = Math.sqrt(s1 / c); D.eNaive = Math.sqrt(s0 / c); D.eFree = SI.rmse(free, D.ytrue, 2);
  }
  function epochSP() {
    const h = new Float64Array(NH), x = new Float64Array(NIN);
    for (let k = 2; k < NTR; k++) {
      phi(D.y, D.u, k, x);
      const e = SI.fwd(N, x, h) - D.y[k];
      SI.bwd(N, x, h, 2 * e / (NTR - 2), null);
    }
    SI.adam(N, 0.01, 5);
  }
  const WL = 40, WB = 12, rW = SI.rng(11);
  function stepPar() {
    const X = [], Hh = [], yh = new Float64Array(WL + 2), g = new Float64Array(WL + 2), gx = new Float64Array(NIN);
    for (let b = 0; b < WB; b++) {
      const s = 2 + Math.floor(rW() * (NTR - WL - 2));
      yh[0] = D.y[s - 2]; yh[1] = D.y[s - 1];                     // each window starts from measured data
      for (let j = 0; j < WL; j++) {
        const x = Float64Array.of(yh[j + 1], yh[j], D.u[s + j - 1], D.u[s + j - 2]), h = new Float64Array(NH);
        yh[j + 2] = SI.fwd(N, x, h);
        X[j] = x; Hh[j] = h;
      }
      g.fill(0);
      for (let j = WL - 1; j >= 0; j--) {                          // backprop through time
        const e = yh[j + 2] - D.y[s + j];
        SI.bwd(N, X[j], Hh[j], g[j + 2] + 2 * e / (WL * WB), gx);
        g[j + 1] += gx[0];                                          // through ŷ(k−1)
        g[j] += gx[1];                                              // through ŷ(k−2)
      }
    }
    SI.adam(N, 0.003, 1);
  }
  function stop() { if (timer) { clearTimeout(timer); timer = null; } }
  function run(m, fresh) {
    stop();
    if (fresh) { N = SI.net(NIN, NH, SI.rng(3), 1); hist = []; step_ = 0; }
    mode = m;
    const per = m === "sp" ? 25 : 10, total = 60;
    let frame = 0;
    const tick = () => {
      for (let i = 0; i < per; i++) (m === "sp" ? epochSP : stepPar)();
      step_ += per;
      evaluate();
      hist.push({ step: step_, mode: m, one: D.eOne, free: D.eFree });
      frame++;
      if (frame < total) { draw(); timer = setTimeout(tick, 16); } else { timer = null; mode = "done-" + m; draw(); }
    };
    tick();
  }

  function draw() {
    svg.selectAll("*").remove();
    const top = SI.panel(svg, 48, 26, W - 70, 230), tw = W - 70, th = 230;
    const xs = d3.scaleLinear().domain([0, NTE - 1]).range([0, tw]);
    const ys = d3.scaleLinear().domain([-2.2, 2.2]).range([th, 0]);
    SI.axes(top, xs, ys, tw, th, "time step k (validation record: a square wave, then a sine — neither seen in training)", "y(k)");
    top.append("line").attr("x1", xs(200)).attr("x2", xs(200)).attr("y1", 0).attr("y2", th).attr("stroke", C.line).attr("stroke-dasharray", "2,3");
    if (D.sig > 0) top.append("g").selectAll("circle").data(Array.from(D.yte)).enter().append("circle")
      .attr("cx", (d, k) => xs(k)).attr("cy", d => ys(Math.max(-2.2, Math.min(2.2, d)))).attr("r", 1.1).attr("fill", C.muted).attr("opacity", 0.45);
    SI.path(top, xs, ys, D.ytrue, C.muted, 2.4);
    SI.path(top, xs, ys, D.one, C.A, 1.3, "4,3", 2);
    SI.path(top, xs, ys, D.free, C.B, 1.7, null, 2);
    SI.legend(top, 6, 8, [{ t: "plant, noise-free (dots: measured)", c: C.muted, w: 215 }, { t: "one-step prediction", c: C.A, d: "4,3", w: 145 }, { t: "free-run simulation", c: C.B, w: 140 }]);
    const bot = SI.panel(svg, 48, 305, W - 70, 88), bw = W - 70, bh = 88;
    const maxS = Math.max(200, hist.length ? hist[hist.length - 1].step : 200);
    const hx = d3.scaleLinear().domain([0, maxS]).range([0, bw]);
    const hy = d3.scaleLog().domain([0.003, 3]).range([bh, 0]).clamp(true);
    SI.axes(bot, hx, hy, bw, bh, "training updates (epochs for series-parallel, BPTT mini-batches for parallel)", "validation RMSE (log)", 6, 3);
    const lineOf = key => d3.line().x(d => hx(d.step)).y(d => hy(Math.max(0.003, Math.min(3, d[key]))));
    if (hist.length) {
      bot.append("path").attr("d", lineOf("one")(hist)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
      bot.append("path").attr("d", lineOf("free")(hist)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.8);
      for (let i = 1; i < hist.length; i++) if (hist[i].mode !== hist[i - 1].mode) {
        bot.append("line").attr("x1", hx(hist[i - 1].step)).attr("x2", hx(hist[i - 1].step)).attr("y1", 0).attr("y2", bh).attr("stroke", C.good).attr("stroke-dasharray", "3,3");
        bot.append("text").attr("x", hx(hist[i - 1].step) + 4).attr("y", 10).attr("fill", C.good).attr("font-size", 10).text("loop closed");
      }
    }
    bot.append("line").attr("x1", 0).attr("x2", bw).attr("y1", hy(D.eNaive)).attr("y2", hy(D.eNaive)).attr("stroke", C.muted).attr("stroke-dasharray", "1,3");
    bot.append("text").attr("x", bw - 4).attr("y", hy(D.eNaive) - 3).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 9.5).text("naive ŷ(k) = y(k−1)");
    const st = mode === "idle" ? "untrained" : mode === "sp" ? "training series-parallel…" : mode === "par" ? "training parallel (BPTT)…" :
      mode === "done-sp" ? "trained series-parallel" : "trained parallel";
    RO.innerHTML = `status: <b>${st}</b> · updates ${step_} · noise: ${D.type === "output" ? "measurement" : "process"} σ = ${D.sig.toFixed(2)} · training amplitude ±${D.A.toFixed(2)}<br>` +
      `one-step RMSE vs measured y <b>${SI.fmt(D.eOne)}</b> (naive persistence ${SI.fmt(D.eNaive)}) · free-run RMSE vs noise-free plant <b>${SI.fmt(D.eFree)}</b>` +
      (D.div ? ` · <b>free run hit the ±6 clamp (diverged)</b>` : "");
  }
  document.getElementById("narx-sp").addEventListener("click", () => run("sp", true));
  document.getElementById("narx-par").addEventListener("click", () => run("par", false));
  document.getElementById("narx-scratch").addEventListener("click", () => run("par", true));
  document.getElementById("narx-reset").addEventListener("click", reset);
  const lab = () => {
    document.getElementById("narx-noise-out").textContent = (+noiseS.value).toFixed(2);
    document.getElementById("narx-amp-out").textContent = "±" + (+ampS.value).toFixed(2);
  };
  [noiseS, ampS].forEach(e => e.addEventListener("input", () => { lab(); reset(); }));
  typeS.addEventListener("change", reset);
  lab(); reset();
})();

/* ═══════════════ 3 · #ord-svg — lag selection and residual whiteness ═══════════════ */
(function () {
  const svg = d3.select("#ord-svg");
  if (svg.empty()) return;
  const W = 760, H = 420, S = 0.5, NH = 10, NTR = 600, NTE = 400;
  const nyS = document.getElementById("ord-ny"), nuS = document.getElementById("ord-nu"),
    sigS = document.getElementById("ord-noise"), RO = document.getElementById("ord-readout");
  let D, Q, R = null;

  function data() {
    const r = SI.rng(21), sig = +sigS.value;
    const u = new Float64Array(NTR);
    for (let k = 0; k < NTR; k++) u[k] = 4 * r() - 2;
    const tr = SI.simulate(u, sig, "equation", r);
    const ut = new Float64Array(NTE);
    for (let k = 0; k < NTE; k++) ut[k] = SI.testU(k);
    // residual tests need a WHITE validation input: against a sinusoid every lag correlates
    const rv = SI.rng(5), uv = new Float64Array(NTE);
    for (let k = 0; k < NTE; k++) uv[k] = 4 * rv() - 2;
    const va = SI.simulate(uv, sig, "equation", rv);
    D = { u, y: tr.ym, ut, ytrue: SI.clean(ut), uv, yv: va.ym, sig };
  }
  /* regressor φ(k) = [y(k−1..k−ny), u(k−1..k−nu)] */
  function phi(y, u, k, ny, nu, out) {
    let i = 0;
    for (let j = 1; j <= ny; j++) out[i++] = y[k - j] * S;
    for (let j = 1; j <= nu; j++) out[i++] = u[k - j] * S;
    return out;
  }
  /* He–Asada Lipschitz index: geometric mean of the p largest quotients, times √n */
  function lipschitz() {
    Q = [];
    const n0 = 5, M = 260, p = Math.max(3, Math.round(0.02 * M));
    for (let ny = 1; ny <= 4; ny++) for (let nu = 1; nu <= 3; nu++) {
      const n = ny + nu, X = [], Y = [];
      for (let k = n0; k < n0 + M; k++) { X.push(phi(D.y, D.u, k, ny, nu, new Float64Array(n))); Y.push(D.y[k] * S); }
      const top = new Float64Array(p);   // ascending
      for (let i = 0; i < M; i++) for (let j = i + 1; j < M; j++) {
        let d = 0;
        for (let t = 0; t < n; t++) { const z = X[i][t] - X[j][t]; d += z * z; }
        const q = Math.abs(Y[i] - Y[j]) / Math.sqrt(d + 1e-12);
        if (q > top[0]) { top[0] = q; top.sort(); }
      }
      let lg = 0;
      for (let t = 0; t < p; t++) lg += Math.log(Math.sqrt(n) * top[t]);
      Q.push({ ny, nu, q: Math.exp(lg / p) });
    }
  }
  function train() {
    const ny = +nyS.value, nu = +nuS.value, n = ny + nu, k0 = 4;
    const N = SI.net(n, NH, SI.rng(3), 1), h = new Float64Array(NH), x = new Float64Array(n);
    for (let ep = 0; ep < 700; ep++) {
      for (let k = k0; k < NTR; k++) {
        phi(D.y, D.u, k, ny, nu, x);
        const e = SI.fwd(N, x, h) - D.y[k] * S;
        SI.bwd(N, x, h, 2 * e / (NTR - k0), null);
      }
      SI.adam(N, 0.01, 5);
    }
    // one-step residuals on the validation record, and the free-run simulation
    const res = [], ures = [], free = new Float64Array(NTE).fill(NaN);
    for (let k = 0; k < k0; k++) free[k] = D.ytrue[k];
    let div = false;
    for (let k = k0; k < NTE; k++) {
      phi(D.yv, D.uv, k, ny, nu, x);
      res.push(D.yv[k] - SI.fwd(N, x, h) / S);
      ures.push(D.uv[k]);
      phi(free, D.ut, k, ny, nu, x);
      let v = SI.fwd(N, x, h) / S;
      if (!isFinite(v) || Math.abs(v) > 12) { v = Math.sign(v || 1) * 12; div = true; }
      free[k] = v;
    }
    const acf = xcorr(res, res, 20), ccf = xcorr(ures, res, 20);
    const band = 1.96 / Math.sqrt(res.length);
    const out = acf.slice(1).filter(v => Math.abs(v) > band).length;
    const outC = ccf.filter(v => Math.abs(v) > band).length;
    R = { ny, nu, n, free, acf, ccf, band, out, outC, div, nP: n * NH + 2 * NH + 1,
      eOne: Math.sqrt(res.reduce((s, v) => s + v * v, 0) / res.length), eFree: SI.rmse(free, D.ytrue, k0) };
  }
  /* normalised cross-correlation r_ab(τ), τ = 0..L */
  function xcorr(a, b, L) {
    const n = a.length, ma = d3.mean(a), mb = d3.mean(b);
    let va = 0, vb = 0;
    for (let i = 0; i < n; i++) { va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; }
    const out = [];
    for (let t = 0; t <= L; t++) {
      let s = 0;
      for (let i = t; i < n; i++) s += (a[i - t] - ma) * (b[i] - mb);
      out.push(s / Math.sqrt(va * vb + 1e-18));
    }
    return out;
  }
  function draw() {
    const ny = +nyS.value, nu = +nuS.value;
    document.getElementById("ord-ny-out").textContent = ny;
    document.getElementById("ord-nu-out").textContent = nu;
    document.getElementById("ord-noise-out").textContent = (+sigS.value).toFixed(2);
    svg.selectAll("*").remove();
    // heat map of the Lipschitz index
    const hm = SI.panel(svg, 52, 40, 220, 250), cw = 70, ch = 58;
    const lq = Q.map(d => Math.log10(d.q)), lo = d3.min(lq), hi = d3.max(lq);
    const col = d3.scaleSequential(d3.interpolateViridis).domain([hi, lo]);
    hm.append("text").attr("x", 0).attr("y", -22).attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text("Lipschitz index q(n_y, nᵤ)");
    hm.append("text").attr("x", 0).attr("y", -8).attr("fill", C.muted).attr("font-size", 10).text("no training needed · lower = regressor explains y");
    Q.forEach(d => {
      const x = (d.nu - 1) * cw, y = (d.ny - 1) * ch, isSel = d.ny === ny && d.nu === nu;
      hm.append("rect").attr("x", x).attr("y", y).attr("width", cw - 3).attr("height", ch - 3).attr("rx", 4)
        .attr("fill", col(Math.log10(d.q))).attr("stroke", isSel ? C.B : "none").attr("stroke-width", 2.5)
        .style("cursor", "pointer").on("click", () => { nyS.value = d.ny; nuS.value = d.nu; train(); draw(); });
      hm.append("text").attr("x", x + cw / 2 - 1.5).attr("y", y + ch / 2 + 4).attr("text-anchor", "middle").attr("font-size", 11.5)
        .attr("fill", Math.log10(d.q) < (lo + hi) / 2 ? "#0f1117" : "#e6e9ef").attr("pointer-events", "none").text(d.q.toFixed(2));
    });
    for (let i = 1; i <= 3; i++) hm.append("text").attr("x", (i - 1) * cw + cw / 2).attr("y", 4 * ch + 14).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10.5).text("nᵤ=" + i);
    for (let i = 1; i <= 4; i++) hm.append("text").attr("x", -6).attr("y", (i - 1) * ch + ch / 2 + 4).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 10.5).text("n_y=" + i);
    hm.append("text").attr("x", 0).attr("y", 4 * ch + 36).attr("fill", C.muted).attr("font-size", 10).text("true plant order: n_y = 2, nᵤ = 1 · click a cell");
    if (!R) return;
    // free run
    const fr = SI.panel(svg, 340, 30, 400, 150), fw = 400, fh = 150;
    const xs = d3.scaleLinear().domain([0, NTE - 1]).range([0, fw]), ys = d3.scaleLinear().domain([-2, 4.5]).range([fh, 0]);
    SI.axes(fr, xs, ys, fw, fh, "k (validation)", `free-run, n_y=${R.ny}, nᵤ=${R.nu}`, 5, 4);
    SI.path(fr, xs, ys, D.ytrue, C.muted, 2.2);
    SI.path(fr, xs, ys, R.free, C.B, 1.5, null, 4);
    // ACF bars
    const ac = SI.panel(svg, 340, 235, 400, 130), aw = 400, ah = 130;
    const ax = d3.scaleBand().domain(d3.range(0, 21)).range([0, aw]).padding(0.25);
    const ay = d3.scaleLinear().domain([-0.6, 0.6]).range([ah, 0]).clamp(true);
    SI.axes(ac, ax, ay, aw, ah, "lag τ", "residual autocorrelation r_εε(τ) (bars) and r_uε(τ) (dots)", 0, 4);
    ac.select(".axis").selectAll(".tick text").attr("display", (d, i) => (i % 5 === 0 ? null : "none"));
    ac.append("rect").attr("x", 0).attr("width", aw).attr("y", ay(R.band)).attr("height", ay(-R.band) - ay(R.band)).attr("fill", "rgba(74,222,128,.10)");
    R.acf.forEach((v, t) => {
      if (t === 0) return;
      ac.append("rect").attr("x", ax(t)).attr("width", ax.bandwidth()).attr("y", Math.min(ay(v), ay(0))).attr("height", Math.abs(ay(v) - ay(0)))
        .attr("fill", Math.abs(v) > R.band ? C.bad : C.A);
    });
    R.ccf.forEach((v, t) => ac.append("circle").attr("cx", ax(t) + ax.bandwidth() / 2).attr("cy", ay(v)).attr("r", 2.6)
      .attr("fill", Math.abs(v) > R.band ? C.bad : C.B));
    ac.append("line").attr("x1", 0).attr("x2", aw).attr("y1", ay(0)).attr("y2", ay(0)).attr("stroke", C.muted);
    const qSel = Q.find(d => d.ny === R.ny && d.nu === R.nu).q;
    RO.innerHTML = `order n_y=${R.ny}, nᵤ=${R.nu}: ${R.n} inputs, ${R.nP} weights · Lipschitz index <b>${qSel.toFixed(2)}</b><br>` +
      `one-step RMSE on a fresh random-input record <b>${SI.fmt(R.eOne)}</b> (noise floor σ = ${D.sig.toFixed(2)}) · free-run RMSE <b>${SI.fmt(R.eFree)}</b>${R.div ? " (diverged)" : ""}<br>` +
      `residual ACF lags 1–20 outside ±1.96/√N = ±${R.band.toFixed(3)}: <b>${R.out}</b> of 20 (≈1 expected by chance) · input–residual CCF lags 0–20 outside: <b>${R.outC}</b> of 21`;
  }
  const retrain = () => { train(); draw(); };
  [nyS, nuS].forEach(e => { e.addEventListener("input", draw); e.addEventListener("change", retrain); });
  sigS.addEventListener("input", () => { document.getElementById("ord-noise-out").textContent = (+sigS.value).toFixed(2); });
  sigS.addEventListener("change", () => { data(); lipschitz(); retrain(); });
  document.getElementById("ord-train").addEventListener("click", retrain);
  data(); lipschitz(); train(); draw();
})();

/* ═══════════════ 4 · #fc-svg — recursive vs direct multi-step forecasting ═══════════════ */
(function () {
  const svg = d3.select("#fc-svg");
  if (svg.empty()) return;
  const W = 760, H = 330, D0 = 6, HMAX = 25, NF = 100, LAM = 1e-6;
  const noiseS = document.getElementById("fc-noise"), orgS = document.getElementById("fc-origin"), RO = document.getElementById("fc-readout");
  /* Mackey–Glass, τ = 17, Euler dt = 0.1, sampled every 2 time units, standardised */
  const series = (function () {
    const dt = 0.1, tau = 17, lag = tau / dt, n = 2200 * 20 + lag;
    const x = new Float64Array(n);
    for (let i = 0; i <= lag; i++) x[i] = 1.2;
    for (let i = lag; i < n - 1; i++) { const xd = x[i - lag]; x[i + 1] = x[i] + dt * (0.2 * xd / (1 + Math.pow(xd, 10)) - 0.1 * x[i]); }
    const out = [];
    for (let i = lag + 3000; i < n && out.length < 2000; i += 20) out.push(x[i]);
    const m = d3.mean(out), s = d3.deviation(out);
    return out.map(v => (v - m) / s);
  })();
  /* fixed random tanh features + ridge readout: a network whose hidden layer is frozen
     (an RBF / extreme-learning style net), so every model on the figure trains in closed form */
  const rF = SI.rng(4), WF = [], BF = [];
  for (let i = 0; i < NF; i++) { const w = []; for (let j = 0; j < D0; j++) w.push(SI.gauss(rF) * 0.6); WF.push(w); BF.push(SI.gauss(rF) * 0.5); }
  const feat = (lags) => { const f = new Float64Array(NF + 1); f[NF] = 1; for (let i = 0; i < NF; i++) { let a = BF[i]; for (let j = 0; j < D0; j++) a += WF[i][j] * lags[j]; f[i] = Math.tanh(a); } return f; };
  function fit(Xf, Y) {
    const n = NF + 1, A = new Float64Array(n * n), b = new Float64Array(n);
    for (let r = 0; r < Xf.length; r++) { const f = Xf[r]; for (let i = 0; i < n; i++) { b[i] += f[i] * Y[r]; for (let j = 0; j < n; j++) A[i * n + j] += f[i] * f[j]; } }
    for (let i = 0; i < n; i++) A[i * n + i] += LAM * Xf.length;
    return SI.solve(A, b, n);
  }
  const dot = (w, f) => { let s = 0; for (let i = 0; i < f.length; i++) s += w[i] * f[i]; return s; };
  let M;
  function build() {
    const sig = +noiseS.value, r = SI.rng(8);
    const obs = series.map(v => v + sig * SI.gauss(r));
    const TR = 1200;
    const lagsAt = (arr, t) => { const l = []; for (let j = 0; j < D0; j++) l.push(arr[t - j]); return l; };
    const Xf = [], idx = [];
    for (let t = D0; t < TR - HMAX; t++) { Xf.push(feat(lagsAt(obs, t))); idx.push(t); }
    const wRec = fit(Xf, idx.map(t => obs[t + 1]));
    const wDir = [];
    for (let h = 1; h <= HMAX; h++) wDir.push(fit(Xf, idx.map(t => obs[t + h])));
    // evaluate on the held-out tail against the NOISE-FREE series
    const eR = new Float64Array(HMAX), eD = new Float64Array(HMAX); let cnt = 0;
    const paths = {};
    for (let t = TR + D0; t < series.length - HMAX; t += 2) {
      const l = lagsAt(obs, t), f = feat(l);
      let win = l.slice();
      const rec = [];
      for (let h = 1; h <= HMAX; h++) { const v = dot(wRec, feat(win)); rec.push(v); win = [v].concat(win.slice(0, D0 - 1)); }
      for (let h = 1; h <= HMAX; h++) {
        const d = dot(wDir[h - 1], f);
        eR[h - 1] += (rec[h - 1] - series[t + h]) ** 2; eD[h - 1] += (d - series[t + h]) ** 2;
      }
      cnt++;
      paths[t] = { rec, dir: wDir.map(w => dot(w, f)) };
    }
    M = { obs, eR: Array.from(eR, v => Math.sqrt(v / cnt)), eD: Array.from(eD, v => Math.sqrt(v / cnt)), paths, TR, cnt, sig };
  }
  function draw() {
    document.getElementById("fc-noise-out").textContent = (+noiseS.value).toFixed(2);
    const keys = Object.keys(M.paths).map(Number);
    const t0 = keys[Math.min(keys.length - 1, Math.round(+orgS.value / 100 * (keys.length - 1)))];
    svg.selectAll("*").remove();
    const L = SI.panel(svg, 46, 26, 360, 250), lw = 360, lh = 250;
    const xs = d3.scaleLinear().domain([-30, HMAX]).range([0, lw]), ys = d3.scaleLinear().domain([-2.4, 2.6]).range([lh, 0]);
    SI.axes(L, xs, ys, lw, lh, "steps relative to the forecast origin", "standardised x", 6, 5);
    L.append("line").attr("x1", xs(0)).attr("x2", xs(0)).attr("y1", 0).attr("y2", lh).attr("stroke", C.line).attr("stroke-dasharray", "3,3");
    const truth = [], obs = [];
    for (let h = -30; h <= HMAX; h++) truth.push([xs(h), ys(series[t0 + h])]);
    for (let h = -30; h <= 0; h++) obs.push([xs(h), ys(M.obs[t0 + h])]);
    L.append("path").attr("d", d3.line()(truth)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 2.2);
    L.selectAll(null).data(obs).enter().append("circle").attr("cx", d => d[0]).attr("cy", d => d[1]).attr("r", 1.8).attr("fill", C.ink);
    const P = M.paths[t0];
    const clampY = v => ys(Math.max(-2.4, Math.min(2.6, v)));
    L.append("path").attr("d", d3.line()([[xs(0), ys(M.obs[t0])]].concat(P.rec.map((v, i) => [xs(i + 1), clampY(v)])))).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.8);
    L.selectAll(null).data(P.dir).enter().append("circle").attr("cx", (d, i) => xs(i + 1)).attr("cy", d => clampY(d)).attr("r", 2.8).attr("fill", C.A);
    SI.legend(L, 4, 8, [{ t: "truth", c: C.muted, w: 70 }, { t: "recursive", c: C.B, w: 90 }, { t: "direct (dots)", c: C.A, w: 100 }]);
    const R = SI.panel(svg, 460, 26, 280, 250), rw = 280, rh = 250;
    const hx = d3.scaleLinear().domain([1, HMAX]).range([0, rw]);
    const hy = d3.scaleLinear().domain([0, Math.max(0.2, d3.max(M.eR.concat(M.eD)) * 1.1)]).range([rh, 0]);
    SI.axes(R, hx, hy, rw, rh, "horizon h", "test RMSE vs noise-free series", 5, 5);
    const ln = arr => d3.line().x((d, i) => hx(i + 1)).y(d => hy(Math.min(d, hy.domain()[1])))(arr);
    R.append("path").attr("d", ln(M.eR)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2);
    R.append("path").attr("d", ln(M.eD)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2).attr("stroke-dasharray", "5,3");
    const hs = [1, 5, 10, 25];
    RO.innerHTML = `observation noise σ = ${M.sig.toFixed(2)} (series standardised) · ${M.cnt} test origins · one ridge readout on ${NF} frozen tanh units, lags x(t), …, x(t−5)<br>` +
      hs.map(h => `h=${h}: recursive <b>${M.eR[h - 1].toFixed(3)}</b> / direct <b>${M.eD[h - 1].toFixed(3)}</b>`).join(" · ") +
      `<br>models trained: recursive <b>1</b> · direct <b>${HMAX}</b> (one per horizon)`;
  }
  noiseS.addEventListener("input", () => { build(); draw(); });
  orgS.addEventListener("input", draw);
  build(); draw();
})();

/* ═══════════════ 5 · #ctl-svg — four neural controllers on one plant ═══════════════ */
(function () {
  const svg = d3.select("#ctl-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, T = 400, UMAX = 3, AM = 0.8;
  const modeS = document.getElementById("ctl-mode"), misS = document.getElementById("ctl-mis"), distS = document.getElementById("ctl-dist"),
    rhoS = document.getElementById("ctl-rho"), hS = document.getElementById("ctl-h"), lrS = document.getElementById("ctl-lr"),
    biasC = document.getElementById("ctl-bias"), RO = document.getElementById("ctl-readout");
  /* the plant, in companion form:  y(k+1) = f(y) + g(y)·u + d   — open-loop unstable at the origin (f′(0) = 1.1) */
  const f = y => 0.5 * y + 0.6 * Math.sin(y), g = y => 1 + 0.4 * Math.cos(2 * y);
  /* the controller's identified model, with a controllable error m */
  const fh = (y, m) => f(y) + 0.2 * m * Math.sin(2 * y) + 0.15 * m, gh = (y, m) => g(y) * (1 - 0.25 * m);
  const LV = [1, -1, 1.5, -0.5, 0.8];
  const ref = k => LV[Math.min(LV.length - 1, Math.floor(k / 80))];
  const clampU = u => Math.max(-UMAX, Math.min(UMAX, u));

  /* direct inverse: trained once, offline, on (y(k), y(k+1)) → u(k) pairs from random excitation */
  const INV = (function () {
    const r = SI.rng(31), N = SI.net(2, 10, r, 1), h = new Float64Array(10), X = [], U = [];
    for (let k = 0; k < 600; k++) {                      // state and input both sampled over the operating range
      const y = (2 * r() - 1) * 2.5, u = (2 * r() - 1) * UMAX, y1 = f(y) + g(y) * u;
      if (Math.abs(y1) > 3) continue;
      X.push(Float64Array.of(y1 / 3, y / 3)); U.push(u / 3);
    }
    for (let ep = 0; ep < 2500; ep++) {
      for (let i = 0; i < X.length; i++) { const e = SI.fwd(N, X[i], h) - U[i]; SI.bwd(N, X[i], h, 2 * e / X.length, null); }
      SI.adam(N, 0.01, 5);
    }
    return N;
  })();
  /* model-reference controller: trained online; persists across episodes until reset */
  /* MR = the committed controller. Every episode ADAPTS online (it is an adaptive controller),
     starting from MR; "train one more episode" commits the adapted weights back into MR.        */
  let MR, episodes = 0, hist = [];
  const newMR = () => { MR = SI.net(2, 8, SI.rng(17), 0.5); for (let i = 16; i < MR.P.length; i++) MR.P[i] *= 0.2; episodes = 0; hist = []; };
  newMR();

  function episode(commit) {
    const mode = modeS.value, m = +misS.value, dist = +distS.value, rho = +rhoS.value, Hz = +hS.value, lr = +lrS.value, bias = biasC.checked;
    const CN = SI.clone(MR);
    let dh = 0;                                              // disturbance estimate: last plant − model residual
    const y = new Float64Array(T + 1), ym = new Float64Array(T + 1), u = new Float64Array(T), rr = new Float64Array(T);
    const h = new Float64Array(10), hm = new Float64Array(8), x2 = new Float64Array(2);
    let sat = 0, uPrev = 0;
    for (let k = 0; k < T; k++) {
      const r = ref(k); rr[k] = r;
      ym[k + 1] = AM * ym[k] + (1 - AM) * r;               // reference model: first-order lag
      let uk = 0, req = ym[k + 1];
      if (mode === "inverse") {
        req = ym[k + 1] - dh;                                 // ask the inverse for the target minus the observed miss
        x2[0] = req / 3; x2[1] = y[k] / 3;
        uk = SI.fwd(INV, x2, h) * 3;
      } else if (mode === "mrac") {
        x2[0] = r / 2; x2[1] = y[k] / 2;
        uk = SI.fwd(CN, x2, hm) * 3;
      } else if (mode === "narma") {
        uk = (ym[k + 1] - dh - fh(y[k], m)) / gh(y[k], m);  // cancel f, divide by g
      } else {                                               // neural MPC, Nu = 1 (move blocking)
        const yr = [], J = cand => {
          let yy = y[k], s = 0;
          for (let j = 0; j < Hz; j++) { yy = fh(yy, m) + gh(yy, m) * cand + dh; s += (yr[j] - yy) ** 2; }
          return s + rho * (cand - uPrev) ** 2;
        };
        let ymr = ym[k];
        for (let j = 0; j < Hz; j++) { ymr = AM * ymr + (1 - AM) * r; yr.push(ymr); }
        let best = 0, bJ = Infinity;
        for (let c = -UMAX; c <= UMAX + 1e-9; c += 0.1) { const v = J(c); if (v < bJ) { bJ = v; best = c; } }
        let a = best - 0.1, b = best + 0.1;                   // golden-section refinement
        for (let it = 0; it < 18; it++) { const c1 = b - 0.618 * (b - a), c2 = a + 0.618 * (b - a); if (J(c1) < J(c2)) b = c2; else a = c1; }
        uk = (a + b) / 2;
      }
      const raw = uk;
      uk = clampU(uk);
      if (Math.abs(raw) > UMAX - 1e-9) sat++;
      u[k] = uk; uPrev = uk;
      const d = k >= 200 ? dist : 0;
      y[k + 1] = Math.max(-20, Math.min(20, f(y[k]) + g(y[k]) * uk + d));
      // offset-free correction (not used by MRAC): the last plant-minus-prediction residual.
      // The inverse has no explicit forward model, so its "prediction" is the output it asked for.
      if (bias) dh = mode === "inverse" ? y[k + 1] - req : y[k + 1] - (fh(y[k], m) + gh(y[k], m) * uk);
      if (mode === "mrac") {
        // e(k+1) = y(k+1) − y_m(k+1);  ∂e/∂θ ≈ ĝ(y(k)) · ∂u/∂θ   (plant Jacobian from the model)
        const e = y[k + 1] - ym[k + 1];
        const gy = raw === uk ? e * gh(y[k], m) * 3 : 0;   // no gradient through saturation
        x2[0] = r / 2; x2[1] = y[k] / 2;
        SI.fwd(CN, x2, hm);
        SI.bwd(CN, x2, hm, gy, null);
        for (let i = 0; i < CN.P.length; i++) { CN.P[i] -= lr * Math.max(-1, Math.min(1, CN.G[i])); CN.G[i] = 0; }
      }
    }
    const err = [], errLate = [];
    for (let k = 1; k <= T; k++) { err.push((y[k] - ym[k]) ** 2); if (k > T - 100) errLate.push((y[k] - ym[k]) ** 2); }
    const out = { y, ym, u, rr, sat, e: Math.sqrt(d3.mean(err)), eLate: Math.sqrt(d3.mean(errLate)), uMax: d3.max(u, Math.abs), mode, m, dist, bias };
    if (mode === "mrac" && commit) { MR = CN; episodes++; hist.push(out.e); }
    return out;
  }
  function draw(E) {
    ["mis", "dist", "rho", "h", "lr"].forEach(k => {
      const el = document.getElementById("ctl-" + k), v = +el.value;
      document.getElementById("ctl-" + k + "-out").textContent = k === "h" ? v : v.toFixed(k === "lr" ? 3 : 2);
    });
    svg.selectAll("*").remove();
    const tp = SI.panel(svg, 46, 24, W - 66, 200), tw = W - 66, th = 200;
    const xs = d3.scaleLinear().domain([0, T]).range([0, tw]), ys = d3.scaleLinear().domain([-2.2, 2.4]).range([th, 0]);
    SI.axes(tp, xs, ys, tw, th, null, "output", 8, 5);
    tp.append("rect").attr("x", xs(200)).attr("width", tw - xs(200)).attr("y", 0).attr("height", th).attr("fill", E.dist ? "rgba(248,113,113,.06)" : "none");
    if (E.dist) tp.append("text").attr("x", xs(204)).attr("y", th - 6).attr("fill", C.bad).attr("font-size", 10).text(`disturbance d = ${E.dist.toFixed(2)} from k = 200`);
    const stair = arr => { const p = []; for (let k = 0; k < arr.length; k++) { p.push([xs(k), ys(arr[k])]); p.push([xs(k + 1), ys(arr[k])]); } return p; };
    tp.append("path").attr("d", d3.line()(stair(E.rr))).attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "3,3");
    SI.path(tp, xs, ys, E.ym, C.A, 2);
    SI.path(tp, xs, ys, E.y, C.B, 1.6);
    SI.legend(tp, 6, 10, [{ t: "command r", c: C.muted, d: "3,3", w: 95 }, { t: "reference model y_m", c: C.A, w: 150 }, { t: "plant output y", c: C.B, w: 120 }]);
    const bp = SI.panel(svg, 46, 262, W - 66, 100), bh = 100;
    const us = d3.scaleLinear().domain([-UMAX - 0.3, UMAX + 0.3]).range([bh, 0]);
    SI.axes(bp, xs, us, tw, bh, "time step k", "control u (limits ±3)", 8, 3);
    [UMAX, -UMAX].forEach(v => bp.append("line").attr("x1", 0).attr("x2", tw).attr("y1", us(v)).attr("y2", us(v)).attr("stroke", C.bad).attr("stroke-dasharray", "2,3"));
    SI.path(bp, xs, us, E.u, C.good, 1.3);
    const names = { inverse: "direct inverse (offline)", mrac: `model-reference, adapting online (episodes committed: ${episodes})`, narma: "NARMA-L2 (cancel f, divide by g)", mpc: `neural MPC (N₂ = ${hS.value}, ρ = ${(+rhoS.value).toFixed(2)})` };
    RO.innerHTML = `<b>${names[E.mode]}</b> · model error m = ${E.m.toFixed(2)} · disturbance ${E.dist.toFixed(2)}${E.bias && E.mode !== "mrac" ? " · offset-free bias term ON" : ""}<br>` +
      `tracking RMSE (y vs y_m): whole episode <b>${E.e.toFixed(3)}</b> · last 100 steps <b>${E.eLate.toFixed(3)}</b> · max |u| <b>${E.uMax.toFixed(2)}</b> · saturated steps <b>${E.sat}</b>` +
      (E.mode === "mrac" && hist.length ? `<br>episode RMSE history: ${hist.map(v => v.toFixed(3)).join(" → ")}` : "");
  }
  const rerun = commit => draw(episode(commit));
  modeS.addEventListener("change", () => rerun(false));
  [misS, distS, rhoS, hS, lrS].forEach(e => e.addEventListener("input", () => rerun(false)));
  biasC.addEventListener("change", () => rerun(false));
  document.getElementById("ctl-more").addEventListener("click", () => rerun(modeS.value === "mrac"));
  document.getElementById("ctl-reset").addEventListener("click", () => { newMR(); rerun(false); });
  rerun(false);
})();
