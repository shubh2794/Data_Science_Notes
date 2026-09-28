/* rbf-networks.viz.js — figures for
   deep-learning/classical-nn/rbf-networks.html (part 5 of the
   Classical Neural Networks series).

   Loaded after ../../data.js and ../../notes.js (palette C).
   Each figure is an IIFE that exits quietly if its <svg> is absent.
   All numerics live in RB below — design matrices, a pivoted linear solver,
   a Jacobi eigen-solver for condition numbers, k-means, OLS forward
   selection, gradient training of centres/widths, a tiny tanh MLP, GRNN and
   PNN — so every readout is computed from the model on screen.

     1  #cv-svg   XOR lifted into (φ₁, φ₂) space, with a separating-line search
     2  #cd-svg   exact interpolation: condition number vs width, and the fit
     3  #ft-svg   1-D fit: M, σ, λ, centre placement, optional gradient fine-tune
     4  #cs-svg   2-D classification with draggable centres
     5  #mp-svg   extrapolation: RBF returns to its bias, MLP keeps going
     6  #gr-svg   GRNN / PNN with leave-one-out σ selection                    */

/* ══════════ numerics ══════════ */
const RB = (function () {
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const sq = (a, b) => { if (typeof a === "number") return (a - b) * (a - b); let s = 0; for (let k = 0; k < a.length; k++) { const t = a[k] - b[k]; s += t * t; } return s; };
  const phi = (x, c, s) => Math.exp(-sq(x, c) / (2 * s * s));

  /* design matrix with a trailing bias column (bias=true) */
  function design(X, Cs, S, bias) {
    return X.map(x => { const row = Cs.map((c, j) => phi(x, c, Array.isArray(S) ? S[j] : S)); if (bias) row.push(1); return row; });
  }
  /* Gaussian elimination with partial pivoting; returns null if singular */
  function solve(A, b) {
    const n = A.length, M = A.map((r, i) => r.slice().concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-300) return null;
      const t = M[c]; M[c] = M[p]; M[p] = t;
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k]; x[i] = s / M[i][i]; }
    return x;
  }
  function gram(P, lam, noRegLast) {
    const m = P[0].length, A = [];
    for (let i = 0; i < m; i++) { A.push(new Array(m).fill(0)); }
    for (const row of P) for (let i = 0; i < m; i++) { const ri = row[i]; if (!ri) continue; for (let j = i; j < m; j++) A[i][j] += ri * row[j]; }
    for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) A[i][j] = A[j][i];
    for (let i = 0; i < m; i++) if (!(noRegLast && i === m - 1)) A[i][i] += lam;
    return A;
  }
  /* ridge: (PᵀP + λI) w = Pᵀt ; the bias (last column) is not penalised */
  function ridge(P, t, lam) {
    const m = P[0].length, A = gram(P, lam, true), b = new Array(m).fill(0);
    P.forEach((row, n) => { for (let i = 0; i < m; i++) b[i] += row[i] * t[n]; });
    return solve(A, b) || new Array(m).fill(0);
  }
  /* symmetric eigenvalues by cyclic Jacobi */
  function eigSym(A0) {
    const n = A0.length, A = A0.map(r => r.slice());
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-30) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const kp = A[k][p], kq = A[k][q]; A[k][p] = c * kp - s * kq; A[k][q] = s * kp + c * kq; }
        for (let k = 0; k < n; k++) { const pk = A[p][k], qk = A[q][k]; A[p][k] = c * pk - s * qk; A[q][k] = s * pk + c * qk; }
      }
    }
    return A.map((r, i) => r[i]).sort((a, b) => a - b);
  }
  function cond(A) {
    const e = eigSym(A), mx = e[e.length - 1], mn = Math.max(e[0], mx * 1e-17);
    return mx / mn;
  }
  function kmeans(X, M, seed, iters) {
    const r = rng(seed), idx = X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    const num = typeof X[0] === "number";
    let Cs = idx.slice(0, M).map(i => num ? X[i] : X[i].slice());
    for (let it = 0; it < (iters || 40); it++) {
      const S = Cs.map(() => num ? 0 : new Array(X[0].length).fill(0)), n = new Array(M).fill(0);
      X.forEach(x => {
        let b = 0, bv = Infinity; Cs.forEach((c, j) => { const d = sq(x, c); if (d < bv) { bv = d; b = j; } });
        n[b]++; if (num) S[b] += x; else x.forEach((v, k) => S[b][k] += v);
      });
      Cs = Cs.map((c, j) => n[j] ? (num ? S[j] / n[j] : S[j].map(v => v / n[j])) : c);
    }
    return Cs;
  }
  /* OLS forward selection over candidate centres = training inputs */
  function ols(X, t, s, M) {
    const N = X.length, P = [];
    for (let i = 0; i < N; i++) P.push(X.map(x => phi(x, X[i], s)));
    const tt = t.reduce((a, v) => a + v * v, 0), Q = [], chosen = [], errs = [];
    const used = new Array(N).fill(false);
    for (let k = 0; k < Math.min(M, N); k++) {
      let best = -1, bestErr = -1, bestV = null;
      for (let i = 0; i < N; i++) {
        if (used[i]) continue;
        const v = P[i].slice();
        for (const q of Q) {
          let a = 0, b = 0; for (let n = 0; n < N; n++) { a += q[n] * P[i][n]; b += q[n] * q[n]; }
          const f = a / b; for (let n = 0; n < N; n++) v[n] -= f * q[n];
        }
        let vv = 0, vt = 0; for (let n = 0; n < N; n++) { vv += v[n] * v[n]; vt += v[n] * t[n]; }
        if (vv < 1e-12) continue;
        const e = vt * vt / (vv * tt);
        if (e > bestErr) { bestErr = e; best = i; bestV = v; }
      }
      if (best < 0) break;
      used[best] = true; Q.push(bestV); chosen.push(best); errs.push(bestErr);
    }
    return { centres: chosen.map(i => X[i]), errs };
  }
  /* 1-D model evaluation: model = {c:[], s:[], w:[..., w0]} */
  function predict1(m, x) { let y = m.w[m.c.length]; for (let j = 0; j < m.c.length; j++) y += m.w[j] * phi(x, m.c[j], m.s[j]); return y; }
  /* full-batch Adam on w, c, log σ */
  function gdTrain(m, X, T, steps, lr) {
    const M = m.c.length, p = { w: m.w.slice(), c: m.c.slice(), ls: m.s.map(Math.log) };
    const keys = ["w", "c", "ls"], mom = {}, vel = {};
    keys.forEach(k => { mom[k] = p[k].map(() => 0); vel[k] = p[k].map(() => 0); });
    const b1 = 0.9, b2 = 0.999, rate = { w: lr, c: lr * 0.3, ls: lr * 0.3 };
    for (let it = 1; it <= steps; it++) {
      const g = { w: p.w.map(() => 0), c: p.c.map(() => 0), ls: p.ls.map(() => 0) };
      for (let n = 0; n < X.length; n++) {
        const x = X[n], ph = [];
        let y = p.w[M];
        for (let j = 0; j < M; j++) { const s = Math.exp(p.ls[j]); ph.push(phi(x, p.c[j], s)); y += p.w[j] * ph[j]; }
        const e = (T[n] - y) / X.length;
        g.w[M] -= e;
        for (let j = 0; j < M; j++) {
          const s = Math.exp(p.ls[j]), d = x - p.c[j];
          g.w[j] -= e * ph[j];
          g.c[j] -= e * p.w[j] * ph[j] * d / (s * s);
          g.ls[j] -= e * p.w[j] * ph[j] * d * d / (s * s);   /* ∂/∂logσ = σ·∂/∂σ */
        }
      }
      keys.forEach(k => {
        for (let i = 0; i < p[k].length; i++) {
          mom[k][i] = b1 * mom[k][i] + (1 - b1) * g[k][i];
          vel[k][i] = b2 * vel[k][i] + (1 - b2) * g[k][i] * g[k][i];
          const mh = mom[k][i] / (1 - Math.pow(b1, it)), vh = vel[k][i] / (1 - Math.pow(b2, it));
          p[k][i] -= rate[k] * mh / (Math.sqrt(vh) + 1e-8);
        }
      });
      for (let j = 0; j < M; j++) p.ls[j] = Math.max(Math.log(0.008), p.ls[j]);
    }
    return { c: p.c, s: p.ls.map(Math.exp), w: p.w };
  }
  /* tiny 1-H-1 tanh MLP trained by full-batch Adam */
  function mlpTrain(X, T, H, seed, steps) {
    const r = rng(seed), sc = x => (x - 0.5) * 4;
    const P = { a: [], b: [], v: [], v0: 0 };
    for (let h = 0; h < H; h++) { P.a.push(gauss(r) * 1.2); P.b.push(gauss(r) * 0.8); P.v.push(gauss(r) * 0.5); }
    const flat = () => P.a.concat(P.b, P.v, [P.v0]);
    const m = flat().map(() => 0), v = flat().map(() => 0), lr = 0.03;
    for (let it = 1; it <= steps; it++) {
      const ga = new Array(H).fill(0), gb = new Array(H).fill(0), gv = new Array(H).fill(0); let gv0 = 0;
      for (let n = 0; n < X.length; n++) {
        const u = sc(X[n]), hs = []; let y = P.v0;
        for (let h = 0; h < H; h++) { hs.push(Math.tanh(P.a[h] * u + P.b[h])); y += P.v[h] * hs[h]; }
        const e = (y - T[n]) / X.length;
        gv0 += e;
        for (let h = 0; h < H; h++) { gv[h] += e * hs[h]; const dz = e * P.v[h] * (1 - hs[h] * hs[h]); ga[h] += dz * u; gb[h] += dz; }
      }
      const g = ga.concat(gb, gv, [gv0]), th = flat();
      for (let i = 0; i < th.length; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
        th[i] -= lr * (m[i] / (1 - Math.pow(0.9, it))) / (Math.sqrt(v[i] / (1 - Math.pow(0.999, it))) + 1e-8);
      }
      P.a = th.slice(0, H); P.b = th.slice(H, 2 * H); P.v = th.slice(2 * H, 3 * H); P.v0 = th[3 * H];
    }
    return x => { const u = sc(x); let y = P.v0; for (let h = 0; h < H; h++) y += P.v[h] * Math.tanh(P.a[h] * u + P.b[h]); return y; };
  }
  function grnn(X, T, s, x, skip) {
    let num = 0, den = 0;
    for (let n = 0; n < X.length; n++) { if (n === skip) continue; const k = phi(x, X[n], s); num += k * T[n]; den += k; }
    return den > 1e-300 ? num / den : T.reduce((a, b) => a + b, 0) / T.length;
  }
  function parzen(X, Y, cls, s, x, skip) {
    let sum = 0, n = 0;
    for (let i = 0; i < X.length; i++) { if (Y[i] !== cls || i === skip) continue; sum += phi(x, X[i], s); n++; }
    return n ? sum / n / (Math.sqrt(2 * Math.PI) * s) : 0;
  }
  return { rng, gauss, sq, phi, design, solve, gram, ridge, eigSym, cond, kmeans, ols, predict1, gdTrain, mlpTrain, grnn, parzen };
})();

const RBV = {
  txt(g, x, y, s, o) {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11).attr("fill", o.fill || C.muted)
      .attr("text-anchor", o.anchor || "start").attr("font-weight", o.bold ? 600 : null).text(s);
  },
  bindRange(id, fmt) {
    const el = document.getElementById(id), out = document.getElementById(id + "-out");
    const show = () => { if (out) out.textContent = fmt ? fmt(+el.value) : el.value; };
    show(); el.addEventListener("input", show); return el;
  },
  axes(g, sx, sy, w, h, xt, yt) {
    g.append("rect").attr("width", w).attr("height", h).attr("fill", "none").attr("stroke", C.line);
    (xt || sx.ticks(5)).forEach(v => RBV.txt(g, sx(v), h + 13, d3.format("~g")(v), { anchor: "middle", size: 9 }));
    (yt || sy.ticks(4)).forEach(v => RBV.txt(g, -4, sy(v) + 3, d3.format("~g")(v), { anchor: "end", size: 9 }));
  },
  sup: n => String(n).split("").map(ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(ch)] || (ch === "-" ? "⁻" : ch)).join(""),
  pow10: v => "10" + RBV.sup(Math.round(v * 100) / 100 === Math.round(v) ? Math.round(v) : v.toFixed(2)),
  lam: v => { const x = Math.pow(10, v); return x < 1e-3 ? x.toExponential(0) : x.toPrecision(2); }
};

/* shared 1-D target used by the fit, GRNN and extrapolation figures */
const RB_F = x => 0.8 * Math.sin(2 * Math.PI * x) + 0.6 * Math.exp(-(x - 0.75) * (x - 0.75) / (2 * 0.04 * 0.04));

/* ═══════════ 1 · #cv-svg — Cover / XOR lift ═══════════ */
(function () {
  const svg = d3.select("#cv-svg");
  if (svg.empty()) return;
  const sel = document.getElementById("cv-c"), sEl = RBV.bindRange("cv-s", v => v.toFixed(2));
  const P = [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]], CC = [C.A, C.B];
  const presets = { diag: [[1, 1], [0, 0]], anti: [[0, 1], [1, 0]], mixed: [[1, 1], [0, 1]], one: [[0.5, 0.5]] };
  function draw() {
    svg.selectAll("*").remove();
    const s = +sEl.value, Cs = presets[sel.value];
    const F = P.map(p => Cs.map(c => RB.phi([p[0], p[1]], c, s)));
    const pt = F.map(f => [f[0], f.length > 1 ? f[1] : 0]);
    /* left: input plane */
    const L = svg.append("g").attr("transform", "translate(40,24)"), W = 270;
    const ix = d3.scaleLinear().domain([-0.5, 1.5]).range([0, W]), iy = d3.scaleLinear().domain([-0.5, 1.5]).range([W, 0]);
    RBV.axes(L, ix, iy, W, W, [0, 1], [0, 1]);
    RBV.txt(L, 0, -8, "input space (x₁, x₂)", { fill: C.ink });
    Cs.forEach((c, j) => {
      L.append("circle").attr("cx", ix(c[0])).attr("cy", iy(c[1])).attr("r", ix(s) - ix(0)).attr("fill", "none").attr("stroke", C.good).attr("stroke-dasharray", "4 3");
      L.append("path").attr("d", d3.symbol().type(d3.symbolCross).size(90)).attr("transform", `translate(${ix(c[0])},${iy(c[1])})`).attr("fill", C.good);
      RBV.txt(L, ix(c[0]) + 8, iy(c[1]) - 8, "c" + "₁₂"[j], { fill: C.good });
    });
    P.forEach(p => L.append("circle").attr("cx", ix(p[0])).attr("cy", iy(p[1])).attr("r", 7).attr("fill", CC[p[2]]).attr("stroke", "#fff"));
    /* right: feature plane */
    const R = svg.append("g").attr("transform", "translate(430,24)");
    const fx = d3.scaleLinear().domain([-0.05, 1.05]).range([0, W]), fy = d3.scaleLinear().domain([-0.05, 1.05]).range([W, 0]);
    RBV.axes(R, fx, fy, W, W, [0, 0.5, 1], [0, 0.5, 1]);
    RBV.txt(R, 0, -8, Cs.length > 1 ? "hidden space (φ₁, φ₂)" : "hidden space: only φ₁ exists", { fill: C.ink });
    /* search for the separating direction with the widest normalised gap */
    let best = null;
    for (let a = 0; a < 720; a++) {
      const th = Math.PI * a / 720, u = [Math.cos(th), Math.sin(th)];
      const pr = pt.map(q => q[0] * u[0] + q[1] * u[1]);
      const A = pr.filter((_, i) => P[i][2] === 0), B = pr.filter((_, i) => P[i][2] === 1);
      const g1 = d3.min(B) - d3.max(A), g2 = d3.min(A) - d3.max(B), gap = Math.max(g1, g2);
      if (!best || gap > best.gap) best = { gap, u, mid: g1 > g2 ? (d3.min(B) + d3.max(A)) / 2 : (d3.min(A) + d3.max(B)) / 2 };
    }
    const sep = best.gap > 1e-9;
    if (sep) {
      const u = best.u, m = best.mid, pts = [];
      /* line u·p = m clipped to [−0.05, 1.05]² */
      [-0.05, 1.05].forEach(x => { if (Math.abs(u[1]) > 1e-9) { const y = (m - u[0] * x) / u[1]; if (y >= -0.05 && y <= 1.05) pts.push([x, y]); } });
      [-0.05, 1.05].forEach(y => { if (Math.abs(u[0]) > 1e-9) { const x = (m - u[1] * y) / u[0]; if (x >= -0.05 && x <= 1.05) pts.push([x, y]); } });
      if (pts.length >= 2) R.append("line").attr("x1", fx(pts[0][0])).attr("y1", fy(pts[0][1])).attr("x2", fx(pts[1][0])).attr("y2", fy(pts[1][1])).attr("stroke", C.good).attr("stroke-width", 2);
    }
    pt.forEach((q, i) => R.append("circle").attr("cx", fx(q[0])).attr("cy", fy(q[1])).attr("r", 7 - i).attr("fill", CC[P[i][2]]).attr("stroke", "#fff").attr("opacity", 0.9));
    RBV.txt(R, W - 4, 16, sep ? "separable" : "NOT separable", { anchor: "end", fill: sep ? C.good : C.bad, bold: true });
    document.getElementById("cv-readout").innerHTML =
      P.map((p, i) => "(" + p[0] + "," + p[1] + ") ↦ (<b>" + F[i].map(v => v.toFixed(3)).join(", ") + "</b>)").join(" · ") +
      " · best margin (half-gap along the normal) <b>" + (sep ? (best.gap / 2).toFixed(3) : "none") + "</b>";
  }
  sel.addEventListener("change", draw); sEl.addEventListener("input", draw);
  draw();
})();

/* ═══════════ 2 · #cd-svg — conditioning vs width ═══════════ */
(function () {
  const svg = d3.select("#cd-svg");
  if (svg.empty()) return;
  const sEl = RBV.bindRange("cd-s", v => v.toFixed(3)), lEl = RBV.bindRange("cd-l", v => RBV.lam(v));
  const r = RB.rng(31), X = d3.range(10).map(i => i / 9), T = X.map(x => RB_F(x) + 0.05 * RB.gauss(r));
  const K = (s, lam) => X.map((a, i) => X.map((b, j) => RB.phi(a, b, s) + (i === j ? lam : 0)));
  const sGrid = d3.range(0, 61).map(i => 0.02 + i * 0.008);
  const base = sGrid.map(s => ({ s, k: Math.log10(RB.cond(K(s, 0))) }));
  function draw() {
    svg.selectAll("*").remove();
    const s = +sEl.value, lam = Math.pow(10, +lEl.value);
    const withL = sGrid.map(v => ({ s: v, k: Math.log10(RB.cond(K(v, lam))) }));
    const L = svg.append("g").attr("transform", "translate(46,24)"), w = 290, h = 260;
    const sx = d3.scaleLinear().domain([0.02, 0.5]).range([0, w]), sy = d3.scaleLinear().domain([0, 18]).range([h, 0]);
    RBV.axes(L, sx, sy, w, h, [0.1, 0.2, 0.3, 0.4, 0.5], [0, 4, 8, 12, 16]);
    RBV.txt(L, 0, -8, "log₁₀ κ versus σ  (grey: λ = 0,  green: current λ)", { fill: C.ink, size: 10.5 });
    L.append("rect").attr("x", 0).attr("y", 0).attr("width", w).attr("height", sy(16) - sy(18)).attr("fill", C.bad).attr("opacity", 0.12);
    RBV.txt(L, w - 4, 11, "beyond double precision", { anchor: "end", size: 9, fill: C.bad });
    const ln = d3.line().x(d => sx(d.s)).y(d => sy(Math.min(18, d.k)));
    L.append("path").attr("d", ln(base)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 1.6);
    L.append("path").attr("d", ln(withL)).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 1.8);
    const kNow = RB.cond(K(s, lam));
    L.append("line").attr("x1", sx(s)).attr("x2", sx(s)).attr("y1", 0).attr("y2", h).attr("stroke", C.ink).attr("stroke-dasharray", "3 3");
    L.append("circle").attr("cx", sx(s)).attr("cy", sy(Math.min(18, Math.log10(kNow)))).attr("r", 4.5).attr("fill", C.B);
    RBV.txt(L, w / 2, h + 28, "σ", { anchor: "middle" });
    /* right: interpolant */
    const w0 = RB.solve(K(s, lam), T) || X.map(() => 0);
    const R = svg.append("g").attr("transform", "translate(420,24)"), W = 320;
    const fx = d3.scaleLinear().domain([-0.05, 1.05]).range([0, W]), fy = d3.scaleLinear().domain([-1.6, 1.8]).range([h, 0]);
    RBV.axes(R, fx, fy, W, h, [0, 0.5, 1], [-1, 0, 1]);
    RBV.txt(R, 0, -8, "the interpolant  f(x) = ∑ wₙ φ(x; xₙ)", { fill: C.ink, size: 10.5 });
    const xs = d3.range(-0.05, 1.0501, 0.0025), clip = v => Math.max(-1.6, Math.min(1.8, v));
    const cid = "cdclip"; R.append("clipPath").attr("id", cid).append("rect").attr("width", W).attr("height", h);
    const P = R.append("g").attr("clip-path", `url(#${cid})`);
    X.forEach((c, n) => P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(clip(w0[n] * RB.phi(x, c, s))))(xs)).attr("fill", "none").attr("stroke", C.A).attr("opacity", 0.25));
    P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(RB_F(x)))(xs)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "4 3");
    P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(clip(X.reduce((a, c, n) => a + w0[n] * RB.phi(x, c, s), 0))))(xs)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2);
    X.forEach((x, n) => R.append("circle").attr("cx", fx(x)).attr("cy", fy(T[n])).attr("r", 4).attr("fill", C.ink));
    const fit = X.map(a => X.reduce((acc, c, n) => acc + w0[n] * RB.phi(a, c, s), 0));
    const res = Math.sqrt(d3.mean(fit.map((v, n) => (v - T[n]) * (v - T[n]))));
    document.getElementById("cd-readout").innerHTML =
      "σ = <b>" + s.toFixed(3) + "</b> · λ = <b>" + RBV.lam(+lEl.value) + "</b> · κ(Φ + λI) = <b>" + (kNow > 1e16 ? "&gt; 10¹⁶" : kNow.toExponential(2)) +
      "</b> · max |wₙ| = <b>" + d3.max(w0, Math.abs).toExponential(2) + "</b> · RMS miss at the data <b>" + res.toExponential(2) + "</b>";
  }
  sEl.addEventListener("input", draw); lEl.addEventListener("input", draw);
  draw();
})();

/* ═══════════ 3 · #ft-svg — 1-D fit ═══════════ */
(function () {
  const svg = d3.select("#ft-svg");
  if (svg.empty()) return;
  const mEl = RBV.bindRange("ft-m"), sEl = RBV.bindRange("ft-s", v => Math.pow(10, v).toFixed(3)), lEl = RBV.bindRange("ft-l", v => RBV.lam(v));
  const how = document.getElementById("ft-how");
  const r = RB.rng(5), X = d3.range(50).map(() => r()).sort(d3.ascending), T = X.map(x => RB_F(x) + 0.1 * RB.gauss(r));
  const Xt = d3.range(0, 1.0001, 0.005);
  let model = null, tuned = false;

  function centres(M, s) {
    if (how.value === "grid") return d3.range(M).map(j => (j + 0.5) / M);
    if (how.value === "random") { const rr = RB.rng(9), idx = X.map((_, i) => i); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rr() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } return idx.slice(0, M).map(i => X[i]); }
    if (how.value === "kmeans") return RB.kmeans(X, M, 3, 50);
    return RB.ols(X, T, s, M).centres;
  }
  function twoStage() {
    const M = +mEl.value, s = Math.pow(10, +sEl.value), lam = Math.pow(10, +lEl.value), Cs = centres(M, s);
    const w = RB.ridge(RB.design(X, Cs, s, true), T, lam);
    model = { c: Cs, s: Cs.map(() => s), w }; tuned = false;
  }
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", "translate(40,20)"), W = 470, H = 300;
    const fx = d3.scaleLinear().domain([0, 1]).range([0, W]), fy = d3.scaleLinear().domain([-1.6, 1.8]).range([H, 0]);
    RBV.axes(g, fx, fy, W, H, [0, 0.25, 0.5, 0.75, 1], [-1, 0, 1]);
    g.append("clipPath").attr("id", "ftclip").append("rect").attr("width", W).attr("height", H);
    const P = g.append("g").attr("clip-path", "url(#ftclip)"), clip = v => Math.max(-1.7, Math.min(1.9, v));
    model.c.forEach((c, j) => P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(clip(model.w[j] * RB.phi(x, c, model.s[j]))))(Xt))
      .attr("fill", "none").attr("stroke", C.A).attr("opacity", 0.3));
    P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(RB_F(x)))(Xt)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "4 3");
    P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(clip(RB.predict1(model, x))))(Xt)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2.2);
    X.forEach((x, n) => g.append("circle").attr("cx", fx(x)).attr("cy", fy(T[n])).attr("r", 3).attr("fill", C.ink).attr("opacity", 0.8));
    model.c.forEach(c => g.append("line").attr("x1", fx(c)).attr("x2", fx(c)).attr("y1", H).attr("y2", H - 9).attr("stroke", C.good).attr("stroke-width", 2));
    const trE = d3.mean(X.map((x, n) => (RB.predict1(model, x) - T[n]) ** 2)), teE = d3.mean(Xt.map(x => (RB.predict1(model, x) - RB_F(x)) ** 2));
    const S = svg.append("g").attr("transform", "translate(560,20)");
    RBV.txt(S, 0, 10, "error (MSE)", { fill: C.ink });
    const bx = d3.scaleLinear().domain([0, Math.max(0.03, 1.15 * Math.max(trE, teE, 0.01))]).range([0, 170]);
    [["train", trE, C.A], ["test vs true f", teE, C.B], ["noise variance", 0.01, C.muted]].forEach((d, i) => {
      S.append("rect").attr("x", 0).attr("y", 24 + i * 34).attr("width", Math.min(170, bx(d[1]))).attr("height", 14).attr("fill", d[2]).attr("opacity", 0.85);
      RBV.txt(S, 0, 50 + i * 34, d[0] + " " + d[1].toFixed(4), { size: 10 });
    });
    RBV.txt(S, 0, 150, "legend", { fill: C.ink });
    [["true f (dashed)", C.muted], ["network output", C.B], ["each wⱼφⱼ(x)", C.A], ["centre ticks", C.good]].forEach((d, i) => {
      S.append("rect").attr("x", 0).attr("y", 160 + i * 18).attr("width", 12).attr("height", 3).attr("fill", d[1]);
      RBV.txt(S, 18, 165 + i * 18, d[0], { size: 10 });
    });
    const M = model.c.length, lam = Math.pow(10, +lEl.value);
    const kap = RB.cond(RB.gram(RB.design(X, model.c, model.s, true), lam, true));
    document.getElementById("ft-readout").innerHTML =
      (tuned ? "<b>gradient-tuned</b> (centres, widths, weights) · " : "two-stage · centres by <b>" + how.options[how.selectedIndex].text + "</b> · ") +
      "M = <b>" + M + "</b> · train MSE <b>" + trE.toFixed(4) + "</b> · test MSE <b>" + teE.toFixed(4) + "</b> · max |wⱼ| <b>" + d3.max(model.w.slice(0, M), Math.abs).toFixed(2) +
      "</b> · κ(ΦᵀΦ + λI) <b>" + (kap > 1e16 ? "&gt; 10¹⁶" : kap.toExponential(1)) + "</b>" +
      (tuned ? " · widths now " + d3.min(model.s).toFixed(3) + " … " + d3.max(model.s).toFixed(3) : "");
  }
  const redo = () => { twoStage(); draw(); };
  [mEl, sEl, lEl].forEach(el => el.addEventListener("input", redo));
  how.addEventListener("change", redo);
  document.getElementById("ft-gd").addEventListener("click", () => { model = RB.gdTrain(model, X, T, 400, 0.02); tuned = true; draw(); });
  document.getElementById("ft-reset").addEventListener("click", redo);
  redo();
})();

/* ═══════════ 4 · #cs-svg — 2-D classification, draggable centres ═══════════ */
(function () {
  const svg = d3.select("#cs-svg");
  if (svg.empty()) return;
  const dataSel = document.getElementById("cs-data"), sEl = RBV.bindRange("cs-s", v => v.toFixed(2)), lEl = RBV.bindRange("cs-l", v => RBV.lam(v));
  const S = 350, ox = 14, oy = 14, R = 50, cell = S / R;
  const sx = v => ox + v * S, sy = v => oy + (1 - v) * S, ix = px => (px - ox) / S, iy = py => 1 - (py - oy) / S;
  let X, Y, Cs, w, seed = 1;

  function makeData() {
    const r = RB.rng(77), kind = dataSel.value; X = []; Y = [];
    for (let n = 0; n < 240; n++) {
      if (kind === "ring") {
        const a = 2 * Math.PI * r();
        if (n % 2) { const rad = 0.16 * Math.sqrt(r()); X.push([0.5 + rad * Math.cos(a), 0.5 + rad * Math.sin(a)]); Y.push(1); }
        else { const rad = 0.28 + 0.14 * r(); X.push([0.5 + rad * Math.cos(a), 0.5 + rad * Math.sin(a)]); Y.push(-1); }
      } else if (kind === "xor") {
        const q = n % 4, c = [[0.28, 0.28], [0.72, 0.72], [0.28, 0.72], [0.72, 0.28]][q];
        X.push([c[0] + 0.08 * RB.gauss(r), c[1] + 0.08 * RB.gauss(r)]); Y.push(q < 2 ? 1 : -1);
      } else {
        const cls = n % 2, a = Math.PI * r();
        const x = cls ? 1 - Math.cos(a) : Math.cos(a), y = cls ? 0.5 - Math.sin(a) : Math.sin(a);
        X.push([(x + 1.4) / 4.4 + 0.02 + 0.025 * RB.gauss(r), (y + 0.9) / 2.3 * 0.8 + 0.1 + 0.025 * RB.gauss(r)]); Y.push(cls ? 1 : -1);
      }
    }
  }
  function placeKM(M) { Cs = RB.kmeans(X, M || 6, seed, 40).map(c => c.slice()); }
  function solveW() { w = RB.ridge(RB.design(X, Cs, +sEl.value, true), Y, Math.pow(10, +lEl.value)); }
  const out = p => { let y = w[Cs.length]; const s = +sEl.value; for (let j = 0; j < Cs.length; j++) y += w[j] * RB.phi(p, Cs[j], s); return y; };

  const bg = svg.append("g"), rects = [];
  for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) rects.push(bg.append("rect").attr("x", ox + i * cell).attr("y", oy + j * cell).attr("width", cell + 0.4).attr("height", cell + 0.4));
  const contourG = svg.append("g"), dataG = svg.append("g"), ctrG = svg.append("g"), side = svg.append("g").attr("transform", "translate(410,20)");
  svg.append("rect").attr("x", ox).attr("y", oy).attr("width", S).attr("height", S).attr("fill", "none").attr("stroke", C.line);
  const col = v => { const t = Math.tanh(1.5 * v); return t >= 0 ? d3.interpolateRgb("#171a23", C.B)(t) : d3.interpolateRgb("#171a23", C.A)(-t); };

  function render(full) {
    solveW();
    const vals = new Array(R * R);
    for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) { const v = out([(i + 0.5) / R, 1 - (j + 0.5) / R]); vals[j * R + i] = v; rects[j * R + i].attr("fill", col(v)).attr("opacity", 0.55); }
    contourG.selectAll("*").remove();
    const cont = d3.contours().size([R, R]).thresholds([0])(vals);
    contourG.append("path").attr("d", d3.geoPath(d3.geoIdentity().scale(cell).translate([ox, oy]))(cont[0])).attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 1.8);
    if (full) {
      dataG.selectAll("*").remove();
      X.forEach((x, n) => dataG.append("circle").attr("cx", sx(x[0])).attr("cy", sy(x[1])).attr("r", 2.4).attr("fill", Y[n] > 0 ? C.B : C.A).attr("stroke", "#0f1117").attr("stroke-width", 0.6));
    }
    ctrG.selectAll("*").remove();
    const s = +sEl.value;
    Cs.forEach((c, j) => {
      const g = ctrG.append("g").attr("transform", `translate(${sx(c[0])},${sy(c[1])})`).style("cursor", "grab").attr("class", "dragpt");
      g.append("circle").attr("r", s * S).attr("fill", "none").attr("stroke", "#fff").attr("stroke-opacity", 0.5).attr("stroke-dasharray", "3 3");
      g.append("circle").attr("r", 7).attr("fill", "#fff").attr("stroke", "#0f1117").attr("stroke-width", 1.5);
      RBV.txt(g, 0, 3.5, String(j + 1), { anchor: "middle", size: 8.5, fill: "#0f1117", bold: true });
      g.call(d3.drag().on("drag", ev => { c[0] = Math.max(0, Math.min(1, ix(ev.x))); c[1] = Math.max(0, Math.min(1, iy(ev.y))); render(false); }));
    });
    /* side: weights */
    side.selectAll("*").remove();
    RBV.txt(side, 0, 0, "output weights wⱼ (bias w₀ last)", { fill: C.ink });
    const M = Cs.length, mx = Math.max(1, d3.max(w, Math.abs)), bw = d3.scaleLinear().domain([-mx, mx]).range([0, 300]);
    side.append("line").attr("x1", bw(0)).attr("x2", bw(0)).attr("y1", 10).attr("y2", 18 + (M + 1) * 16).attr("stroke", C.line);
    w.forEach((v, j) => {
      const y = 12 + j * 16;
      side.append("rect").attr("x", Math.min(bw(0), bw(v))).attr("y", y).attr("width", Math.abs(bw(v) - bw(0))).attr("height", 11).attr("fill", v >= 0 ? C.B : C.A);
      RBV.txt(side, v >= 0 ? bw(0) - 4 : bw(0) + 4, y + 9, (j < M ? "w" + (j + 1) : "w₀") + " " + v.toFixed(2), { anchor: v >= 0 ? "end" : "start", size: 9.5, fill: C.ink });
    });
    RBV.txt(side, 0, 36 + (M + 1) * 16, "scale ±" + mx.toFixed(1) + " · amber = class +1, blue = class −1", { size: 9.5 });
    let ok = 0; X.forEach((x, n) => { if (Math.sign(out(x)) === Y[n]) ok++; });
    document.getElementById("cs-readout").innerHTML =
      "centres <b>" + M + "</b> · σ = <b>" + s.toFixed(2) + "</b> · λ = <b>" + RBV.lam(+lEl.value) + "</b> · training accuracy <b>" + (100 * ok / X.length).toFixed(1) +
      "%</b> · max |w| <b>" + d3.max(w.slice(0, M), Math.abs).toFixed(2) + "</b> · white curve = decision boundary y(x) = 0";
  }
  function reset() { makeData(); placeKM(6); render(true); }
  dataSel.addEventListener("change", reset);
  sEl.addEventListener("input", () => render(false)); lEl.addEventListener("input", () => render(false));
  document.getElementById("cs-add").addEventListener("click", () => { const r = RB.rng(seed++ * 13 + Cs.length); Cs.push(X[Math.floor(r() * X.length)].slice()); render(false); });
  document.getElementById("cs-rem").addEventListener("click", () => { if (Cs.length > 1) { Cs.pop(); render(false); } });
  document.getElementById("cs-km").addEventListener("click", () => { seed++; placeKM(Math.max(2, Cs.length)); render(false); });
  reset();
})();

/* ═══════════ 5 · #mp-svg — extrapolation, RBF vs MLP ═══════════ */
(function () {
  const svg = d3.select("#mp-svg");
  if (svg.empty()) return;
  const sEl = RBV.bindRange("mp-s", v => v.toFixed(3)), hEl = RBV.bindRange("mp-h");
  const r = RB.rng(41), X = d3.range(30).map(() => 0.2 + 0.6 * r()).sort(d3.ascending);
  const f = x => 0.8 * Math.sin(2 * Math.PI * x), T = X.map(x => f(x) + 0.08 * RB.gauss(r));
  const Cs = d3.range(10).map(j => 0.2 + 0.6 * j / 9);
  let mlp = null, mseed = 3;
  const trainMLP = () => { mlp = RB.mlpTrain(X, T, +hEl.value, mseed, 2500); };
  function draw() {
    svg.selectAll("*").remove();
    /* bias fixed at the target mean (centred targets), bumps fitted by ridge to the residual */
    const s = +sEl.value, tm = d3.mean(T), P0 = RB.design(X, Cs, s, false), A = RB.gram(P0, 1e-3, false);
    const rhs = Cs.map((_, j) => P0.reduce((acc, row, n) => acc + row[j] * (T[n] - tm), 0));
    const w = (RB.solve(A, rhs) || Cs.map(() => 0)).concat([tm]);
    const rbf = x => { let y = w[Cs.length]; Cs.forEach((c, j) => y += w[j] * RB.phi(x, c, s)); return y; };
    const g = svg.append("g").attr("transform", "translate(40,16)"), W = 690, H = 280;
    const fx = d3.scaleLinear().domain([-0.3, 1.3]).range([0, W]), fy = d3.scaleLinear().domain([-2.2, 2.2]).range([H, 0]);
    g.append("rect").attr("x", 0).attr("width", fx(0.2)).attr("height", H).attr("fill", C.muted).attr("opacity", 0.07);
    g.append("rect").attr("x", fx(0.8)).attr("width", W - fx(0.8)).attr("height", H).attr("fill", C.muted).attr("opacity", 0.07);
    RBV.axes(g, fx, fy, W, H, [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], [-2, -1, 0, 1, 2]);
    g.append("clipPath").attr("id", "mpclip").append("rect").attr("width", W).attr("height", H);
    const P = g.append("g").attr("clip-path", "url(#mpclip)"), xs = d3.range(-0.3, 1.3001, 0.004);
    P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(f(x)))(xs)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "4 3");
    P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(rbf(x)))(xs)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2.2);
    P.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(Math.max(-3, Math.min(3, mlp(x)))))(xs)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2.2);
    P.append("line").attr("x1", 0).attr("x2", W).attr("y1", fy(w[Cs.length])).attr("y2", fy(w[Cs.length])).attr("stroke", C.A).attr("stroke-dasharray", "2 4").attr("opacity", 0.7);
    X.forEach((x, n) => g.append("circle").attr("cx", fx(x)).attr("cy", fy(T[n])).attr("r", 3).attr("fill", C.ink));
    RBV.txt(g, 8, 14, "outside the data", { size: 10 }); RBV.txt(g, W - 8, 14, "outside the data", { size: 10, anchor: "end" });
    [["RBF net (10 centres)", C.A], ["tanh MLP", C.B], ["true f (dashed)", C.muted]].forEach((d, i) => {
      g.append("rect").attr("x", fx(0.25) + i * 160).attr("y", H - 16).attr("width", 12).attr("height", 3).attr("fill", d[1]);
      RBV.txt(g, fx(0.25) + i * 160 + 16, H - 11, d[0], { size: 10, fill: C.ink });
    });
    const inErr = (m) => d3.mean(X.map((x, n) => (m(x) - T[n]) ** 2));
    document.getElementById("mp-readout").innerHTML =
      "train MSE: RBF <b>" + inErr(rbf).toFixed(4) + "</b>, MLP <b>" + inErr(mlp).toFixed(4) + "</b> · at x = −0.2: RBF <b>" + rbf(-0.2).toFixed(2) + "</b>, MLP <b>" + mlp(-0.2).toFixed(2) +
      "</b>, true " + f(-0.2).toFixed(2) + " · at x = 1.2: RBF <b>" + rbf(1.2).toFixed(2) + "</b>, MLP <b>" + mlp(1.2).toFixed(2) + "</b>, true " + f(1.2).toFixed(2) +
      " · RBF bias w₀ = <b>" + w[Cs.length].toFixed(2) + "</b> (dotted)";
  }
  sEl.addEventListener("input", draw);
  hEl.addEventListener("change", () => { trainMLP(); draw(); });
  document.getElementById("mp-seed").addEventListener("click", () => { mseed += 1; trainMLP(); draw(); });
  trainMLP(); draw();
})();

/* ═══════════ 6 · #gr-svg — GRNN / PNN ═══════════ */
(function () {
  const svg = d3.select("#gr-svg");
  if (svg.empty()) return;
  const mode = document.getElementById("gr-mode"), sEl = RBV.bindRange("gr-s", v => Math.pow(10, v).toFixed(4));
  const r = RB.rng(19), X = d3.range(40).map(() => r()).sort(d3.ascending), T = X.map(x => RB_F(x) + 0.12 * RB.gauss(r));
  const Xc = [], Yc = [];
  for (let n = 0; n < 60; n++) {
    if (n % 2) { Xc.push(0.55 + 0.1 * RB.gauss(r)); Yc.push(1); }
    else { Xc.push(n % 4 === 0 ? 0.3 + 0.08 * RB.gauss(r) : 0.8 + 0.05 * RB.gauss(r)); Yc.push(0); }
  }
  const sGrid = d3.range(0, 46).map(i => Math.pow(10, -2.3 + i * 0.04));
  const looR = s => d3.mean(X.map((x, n) => (RB.grnn(X, T, s, x, n) - T[n]) ** 2));
  const looC = s => d3.mean(Xc.map((x, n) => ((RB.parzen(Xc, Yc, 1, s, x, n) > RB.parzen(Xc, Yc, 0, s, x, n) ? 1 : 0) !== Yc[n]) ? 1 : 0));
  const curves = { grnn: sGrid.map(s => ({ s, e: looR(s) })), pnn: sGrid.map(s => ({ s, e: looC(s) })) };
  function draw() {
    svg.selectAll("*").remove();
    const s = Math.pow(10, +sEl.value), isR = mode.value === "grnn";
    const g = svg.append("g").attr("transform", "translate(40,20)"), W = 400, H = 280;
    const fx = d3.scaleLinear().domain([0, 1]).range([0, W]), xs = d3.range(0, 1.0001, 0.0025);
    if (isR) {
      const fy = d3.scaleLinear().domain([-1.4, 1.8]).range([H, 0]);
      RBV.axes(g, fx, fy, W, H, [0, 0.5, 1], [-1, 0, 1]);
      RBV.txt(g, 0, -6, "GRNN: ŷ(x) = ∑ tₙφₙ / ∑ φₙ   (one unit per training point)", { fill: C.ink, size: 10.5 });
      g.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(RB_F(x)))(xs)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "4 3");
      g.append("path").attr("d", d3.line().x(x => fx(x)).y(x => fy(RB.grnn(X, T, s, x, -1)))(xs)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2.2);
      X.forEach((x, n) => g.append("circle").attr("cx", fx(x)).attr("cy", fy(T[n])).attr("r", 3).attr("fill", C.ink));
    } else {
      const d0 = xs.map(x => RB.parzen(Xc, Yc, 0, s, x, -1)), d1 = xs.map(x => RB.parzen(Xc, Yc, 1, s, x, -1));
      const fy = d3.scaleLinear().domain([0, Math.min(12, 1.1 * Math.max(d3.max(d0), d3.max(d1)))]).range([H - 30, 0]);
      RBV.axes(g, fx, fy, W, H, [0, 0.5, 1], fy.ticks(3));
      RBV.txt(g, 0, -6, "PNN: Parzen density per class, decide by the larger", { fill: C.ink, size: 10.5 });
      g.append("clipPath").attr("id", "grclip").append("rect").attr("width", W).attr("height", H - 30);
      const P = g.append("g").attr("clip-path", "url(#grclip)");
      P.append("path").attr("d", d3.line().x((x, i) => fx(x)).y((x, i) => fy(d0[i]))(xs)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2);
      P.append("path").attr("d", d3.line().x((x, i) => fx(x)).y((x, i) => fy(d1[i]))(xs)).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2);
      xs.forEach((x, i) => g.append("rect").attr("x", fx(x)).attr("y", H - 22).attr("width", fx(0.0026) - fx(0)).attr("height", 10).attr("fill", d1[i] > d0[i] ? C.B : C.A));
      Xc.forEach((x, n) => g.append("line").attr("x1", fx(x)).attr("x2", fx(x)).attr("y1", H - 30).attr("y2", H - 24).attr("stroke", Yc[n] ? C.B : C.A).attr("stroke-width", 1.5));
      RBV.txt(g, W, H - 26, "decision →", { anchor: "end", size: 9 });
    }
    /* right: LOO curve */
    const cv = curves[mode.value], best = cv.reduce((a, b) => b.e < a.e ? b : a);
    const R = svg.append("g").attr("transform", "translate(500,20)"), w = 240;
    const lx = d3.scaleLog().domain([sGrid[0], sGrid[sGrid.length - 1]]).range([0, w]);
    const ly = d3.scaleLinear().domain([0, isR ? Math.min(0.2, d3.max(cv, d => d.e)) : 0.5]).range([H, 0]);
    RBV.axes(R, lx, ly, w, H, [0.01, 0.03, 0.1, 0.3], ly.ticks(4));
    RBV.txt(R, 0, -6, isR ? "leave-one-out MSE vs σ (log)" : "leave-one-out error rate vs σ (log)", { fill: C.ink, size: 10.5 });
    R.append("path").attr("d", d3.line().x(d => lx(d.s)).y(d => ly(Math.min(ly.domain()[1], d.e)))(cv)).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 1.8);
    R.append("line").attr("x1", lx(best.s)).attr("x2", lx(best.s)).attr("y1", 0).attr("y2", H).attr("stroke", C.good).attr("stroke-dasharray", "2 3");
    R.append("line").attr("x1", lx(s)).attr("x2", lx(s)).attr("y1", 0).attr("y2", H).attr("stroke", C.ink).attr("stroke-dasharray", "4 3");
    const now = isR ? looR(s) : looC(s);
    document.getElementById("gr-readout").innerHTML =
      (isR ? "GRNN" : "PNN") + " · σ = <b>" + s.toFixed(4) + "</b> · leave-one-out " + (isR ? "MSE" : "error") + " <b>" + (isR ? now.toFixed(4) : (100 * now).toFixed(1) + "%") +
      "</b> · best on the grid σ ≈ <b>" + best.s.toFixed(4) + "</b> with " + (isR ? best.e.toFixed(4) : (100 * best.e).toFixed(1) + "%") +
      (isR ? " (noise variance 0.0144)" : "");
  }
  mode.addEventListener("change", draw); sEl.addEventListener("input", draw);
  draw();
})();
