/* perceptrons.viz.js — figures for deep-learning/classical-nn/perceptrons.html
   (part 1 of the Classical Neural Networks series).
   Loaded after ../../data.js → ../../notes.js. Self-contained: every helper
   lives in PV below. Each figure is an IIFE that exits quietly when its
   container is absent.

     1  #mp-svg     threshold logic: one unit, one line, one gate
     2  #hebb-svg   Hebb vs Oja on a correlated cloud
     3  #pt-svg     step-through perceptron training on draggable points
     4  #bd-svg     mistakes vs the (R/γ)² bound over many orderings
     5  #pk-svg     perceptron vs pocket vs averaged on overlapping classes
     6  #pva-svg    perceptron vs Adaline boundaries, with outliers
     7  #lms-svg    gradient descent / LMS on the Adaline error bowl
     8  #cv-svg     Cover's separability cliff
     9  #xr-svg     XOR: no line, then a hidden layer                      */

const PV = (function () {
  const col = {
    pos: "#5b9cff", neg: "#f87171", ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a",
    accent: "#5b9cff", orange: "#ffb454", good: "#4ade80", bad: "#f87171",
    panel: "#171a23", panel2: "#1e222d", faint: "#5d6675"
  };
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function shuffle(arr, r) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }
  const f2 = d3.format(".2f"), f3 = d3.format(".3f"), f1 = d3.format(".1f");
  const sgn = v => (v >= 0 ? "+" : "−") + f2(Math.abs(v));

  /* points of the line w1·x + w2·y + b = 0 inside the box, or null */
  function clipLine(w1, w2, b, box) {
    const [x0, x1, y0, y1] = box, pts = [];
    if (Math.abs(w2) > 1e-12) {
      [x0, x1].forEach(x => { const y = -(w1 * x + b) / w2; if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]); });
    }
    if (Math.abs(w1) > 1e-12) {
      [y0, y1].forEach(y => { const x = -(w2 * y + b) / w1; if (x >= x0 - 1e-9 && x <= x1 + 1e-9) pts.push([x, y]); });
    }
    const uniq = [];
    pts.forEach(p => { if (!uniq.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-9)) uniq.push(p); });
    return uniq.length >= 2 ? [uniq[0], uniq[1]] : null;
  }
  /* polygon of the box where w1·x + w2·y + b > 0 (single-plane Sutherland–Hodgman) */
  function halfPlane(w1, w2, b, box) {
    const [x0, x1, y0, y1] = box;
    const poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    const f = p => w1 * p[0] + w2 * p[1] + b, out = [];
    for (let i = 0; i < poly.length; i++) {
      const P = poly[i], Q = poly[(i + 1) % poly.length], fp = f(P), fq = f(Q);
      if (fp >= 0) out.push(P);
      if ((fp >= 0) !== (fq >= 0)) {
        const t = fp / (fp - fq);
        out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]);
      }
    }
    return out;
  }
  function marker(svg, id, color) {
    let defs = svg.select("defs");
    if (defs.empty()) defs = svg.append("defs");
    defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", color);
    return "url(#" + id + ")";
  }
  function frameBox(g, x, y, w, h) {
    g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h)
      .attr("fill", col.panel2).attr("stroke", col.line).attr("rx", 6);
  }
  function label(g, x, y, text, o) {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11)
      .attr("fill", o.color || col.muted).attr("text-anchor", o.anchor || "start")
      .attr("font-weight", o.bold ? 600 : null).text(text);
  }
  function axisStyle(sel) {
    sel.selectAll("path,line").attr("stroke", col.faint);
    sel.selectAll("text").attr("fill", col.muted).attr("font-size", 9.5);
    return sel;
  }
  /* 3×3 linear solve (Gaussian elimination with partial pivoting) */
  function solve3(A, bvec) {
    const M = A.map((r, i) => r.slice().concat([bvec[i]]));
    const n = 3;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      const t = M[c]; M[c] = M[p]; M[p] = t;
      if (Math.abs(M[c][c]) < 1e-12) return [0, 0, 0];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = [0, 0, 0];
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }
  /* least-squares (Adaline optimum) on points {x, y, t}: returns [w1, w2, b] */
  function adalineLS(pts) {
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], v = [0, 0, 0];
    pts.forEach(p => {
      const z = [p.x, p.y, 1];
      for (let i = 0; i < 3; i++) { v[i] += p.t * z[i]; for (let j = 0; j < 3; j++) A[i][j] += z[i] * z[j]; }
    });
    for (let i = 0; i < 3; i++) A[i][i] += 1e-6;
    return solve3(A, v);
  }
  /* perceptron to convergence (fixed order), returns {w:[w1,w2,b], mistakes, epochs, converged} */
  function perceptronFit(pts, maxEpochs) {
    let w1 = 0, w2 = 0, b = 0, m = 0, e;
    for (e = 1; e <= maxEpochs; e++) {
      let em = 0;
      for (const p of pts) {
        if (p.t * (w1 * p.x + w2 * p.y + b) <= 0) { w1 += p.t * p.x; w2 += p.t * p.y; b += p.t; em++; }
      }
      m += em;
      if (em === 0) return { w: [w1, w2, b], mistakes: m, epochs: e, converged: true };
    }
    return { w: [w1, w2, b], mistakes: m, epochs: maxEpochs, converged: false };
  }
  const errors = (w, pts) => pts.reduce((s, p) => s + (p.t * (w[0] * p.x + w[1] * p.y + w[2]) <= 0 ? 1 : 0), 0);
  return { col, rng, gauss, shuffle, f1, f2, f3, sgn, clipLine, halfPlane, marker, frameBox, label, axisStyle, adalineLS, perceptronFit, errors };
})();

/* ═════════ 1 · #mp-svg — threshold logic ═════════ */
(function () {
  const svg = d3.select("#mp-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const gates = {
    AND: { t: [0, 0, 0, 1], p: [1, 1, 1.5] },
    OR: { t: [0, 1, 1, 1], p: [1, 1, 0.5] },
    NAND: { t: [1, 1, 1, 0], p: [-1, -1, -1.5] },
    NOR: { t: [1, 0, 0, 0], p: [-1, -1, -0.5] },
    A_NOT_B: { t: [0, 0, 1, 0], p: [1, -1, 0.5] },
    XOR: { t: [0, 1, 1, 0], p: [1, 1, 0.5] }
  };
  const rows = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const gSel = document.getElementById("mp-gate");
  const s1 = document.getElementById("mp-w1"), s2 = document.getElementById("mp-w2"), sT = document.getElementById("mp-th");
  const mk = PV.marker(svg, "mp-arr", C.muted);
  const box = [-0.3, 1.3, -0.3, 1.3];
  const X0 = 370, Y0 = 14, S = 270;
  const sx = d3.scaleLinear().domain([box[0], box[1]]).range([X0, X0 + S]);
  const sy = d3.scaleLinear().domain([box[2], box[3]]).range([Y0 + S, Y0]);

  function load() {
    const p = gates[gSel.value].p;
    s1.value = p[0]; s2.value = p[1]; sT.value = p[2];
    draw();
  }
  function draw() {
    svg.selectAll("g.layer").remove();
    const g = svg.append("g").attr("class", "layer");
    const w1 = +s1.value, w2 = +s2.value, th = +sT.value, tgt = gates[gSel.value].t;
    document.getElementById("mp-w1-out").textContent = PV.f1(w1);
    document.getElementById("mp-w2-out").textContent = PV.f1(w2);
    document.getElementById("mp-th-out").textContent = PV.f1(th);

    /* unit diagram */
    const ins = [[70, 105, "x₁", w1], [70, 225, "x₂", w2]];
    ins.forEach(([x, y, n, w]) => {
      g.append("line").attr("x1", x + 18).attr("y1", y).attr("x2", 212).attr("y2", 165)
        .attr("stroke", w >= 0 ? C.pos : C.neg).attr("stroke-width", 1 + 2.5 * Math.min(Math.abs(w), 2) / 2)
        .attr("marker-end", mk);
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 18).attr("fill", C.panel2).attr("stroke", C.muted);
      PV.label(g, x, y + 4, n, { anchor: "middle", color: C.ink, size: 13 });
      PV.label(g, x + 26, y + (y < 165 ? -14 : 24), (n === "x₁" ? "w₁ = " : "w₂ = ") + PV.f1(w),
        { anchor: "start", color: w >= 0 ? C.pos : C.neg, size: 11.5 });
    });
    g.append("circle").attr("cx", 240).attr("cy", 165).attr("r", 30).attr("fill", C.panel2).attr("stroke", C.orange).attr("stroke-width", 1.5);
    PV.label(g, 240, 161, "∑ ≥ θ ?", { anchor: "middle", color: C.ink, size: 11 });
    PV.label(g, 240, 176, "θ = " + PV.f1(th), { anchor: "middle", color: C.orange, size: 10.5 });
    g.append("line").attr("x1", 270).attr("y1", 165).attr("x2", 312).attr("y2", 165).attr("stroke", C.muted).attr("marker-end", mk);
    PV.label(g, 320, 169, "y", { color: C.ink, size: 13 });
    PV.label(g, 20, 300, "fires when w₁x₁ + w₂x₂ ≥ θ", { size: 11 });

    /* the square */
    PV.frameBox(g, X0, Y0, S, S);
    const hp = PV.halfPlane(w1, w2, -th, box);
    if (hp.length > 2) g.append("polygon").attr("points", hp.map(p => sx(p[0]) + "," + sy(p[1])).join(" "))
      .attr("fill", C.good).attr("fill-opacity", 0.12);
    const L = PV.clipLine(w1, w2, -th, box);
    if (L) g.append("line").attr("x1", sx(L[0][0])).attr("y1", sy(L[0][1])).attr("x2", sx(L[1][0])).attr("y2", sy(L[1][1]))
      .attr("stroke", C.ink).attr("stroke-width", 1.8);
    [0, 1].forEach(v => {
      PV.label(g, sx(v), Y0 + S + 14, String(v), { anchor: "middle", size: 10 });
      PV.label(g, X0 - 8, sy(v) + 4, String(v), { anchor: "end", size: 10 });
    });
    PV.label(g, X0 + S / 2, Y0 + S + 28, "x₁", { anchor: "middle", color: C.ink });
    PV.label(g, X0 - 22, Y0 + S / 2, "x₂", { anchor: "middle", color: C.ink });

    let ok = 0;
    const tab = [];
    rows.forEach((r, i) => {
      const a = w1 * r[0] + w2 * r[1], y = a >= th - 1e-9 ? 1 : 0, good = y === tgt[i];
      if (good) ok++;
      g.append("circle").attr("cx", sx(r[0])).attr("cy", sy(r[1])).attr("r", 11)
        .attr("fill", tgt[i] ? C.pos : C.panel).attr("stroke", tgt[i] ? C.pos : C.neg).attr("stroke-width", 2.2);
      g.append("text").attr("x", sx(r[0])).attr("y", sy(r[1]) + 4).attr("text-anchor", "middle").attr("font-size", 11)
        .attr("font-weight", 700).attr("fill", good ? C.good : C.bad).text(good ? "✓" : "✗");
      tab.push("(" + r[0] + "," + r[1] + "): ∑ = " + PV.f1(a) + " → y = " + y + ", target " + tgt[i] + (good ? " ✓" : " <b>✗</b>"));
    });
    PV.label(g, X0 + S + 12, Y0 + 14, "filled: target 1", { size: 10 });
    PV.label(g, X0 + S + 12, Y0 + 28, "hollow: target 0", { size: 10 });
    PV.label(g, X0 + S + 12, Y0 + 50, "shaded: fires", { size: 10, color: C.good });
    PV.label(g, X0 + S + 12, Y0 + 80, ok + " / 4 rows", { size: 13, bold: true, color: ok === 4 ? C.good : C.orange });
    const note = gSel.value === "XOR"
      ? (ok === 4 ? "" : " — no setting of w₁, w₂, θ gets all four (§12)") : (ok === 4 ? " — this unit computes " + gSel.options[gSel.selectedIndex].text : "");
    document.getElementById("mp-readout").innerHTML = tab.join(" · ") + "<br><b>" + ok + "/4 correct</b>" + note;
  }
  gSel.addEventListener("change", load);
  [s1, s2, sT].forEach(s => s.addEventListener("input", draw));
  load();
})();

/* ═════════ 2 · #hebb-svg — Hebb vs Oja ═════════ */
(function () {
  const svg = d3.select("#hebb-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const sEta = document.getElementById("hebb-eta"), sRho = document.getElementById("hebb-rho");
  const mkH = PV.marker(svg, "hb-h", C.orange), mkO = PV.marker(svg, "hb-o", C.good);
  const box = [-3, 3, -3, 3];
  const LX = 20, LY = 14, LS = 290;
  const sx = d3.scaleLinear().domain([box[0], box[1]]).range([LX, LX + LS]);
  const sy = d3.scaleLinear().domain([box[2], box[3]]).range([LY + LS, LY]);
  const RX = 380, RY = 20, RW = 350, RH = 250;
  let data, cov, pc1, wH, wO, hist, step, timer = null;

  function build() {
    const rho = +sRho.value, s2 = 0.6;
    cov = [[1, rho * s2], [rho * s2, s2 * s2]];
    /* principal eigenvector of a symmetric 2×2 */
    const a = cov[0][0], b = cov[0][1], d = cov[1][1];
    const tr = a + d, det = a * d - b * b, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det);
    let v = Math.abs(b) > 1e-12 ? [l1 - d, b] : (a >= d ? [1, 0] : [0, 1]);
    const n = Math.hypot(v[0], v[1]); pc1 = [v[0] / n, v[1] / n];
    const r = PV.rng(7);
    data = [];
    /* Cholesky of cov */
    const L11 = Math.sqrt(a), L21 = b / L11, L22 = Math.sqrt(Math.max(d - L21 * L21, 1e-12));
    for (let i = 0; i < 2000; i++) {
      const z1 = PV.gauss(r), z2 = PV.gauss(r);
      data.push([L11 * z1, L21 * z1 + L22 * z2]);
    }
    reset();
  }
  function reset() {
    if (timer) { timer.stop(); timer = null; }
    wH = [0.25, -0.35]; wO = [0.25, -0.35]; step = 0;
    hist = [[0, Math.hypot(wH[0], wH[1]), Math.hypot(wO[0], wO[1])]];
    draw();
  }
  function advance(k) {
    const eta = +sEta.value;
    for (let i = 0; i < k; i++) {
      const x = data[step % data.length];
      const yH = wH[0] * x[0] + wH[1] * x[1];
      wH = [wH[0] + eta * yH * x[0], wH[1] + eta * yH * x[1]];
      const yO = wO[0] * x[0] + wO[1] * x[1];
      wO = [wO[0] + eta * yO * (x[0] - yO * wO[0]), wO[1] + eta * yO * (x[1] - yO * wO[1])];
      step++;
      hist.push([step, Math.hypot(wH[0], wH[1]), Math.hypot(wO[0], wO[1])]);
    }
  }
  const angleTo = w => {
    const n = Math.hypot(w[0], w[1]) || 1;
    const c = Math.abs((w[0] * pc1[0] + w[1] * pc1[1]) / n);
    return Math.acos(Math.min(1, c)) * 180 / Math.PI;
  };
  function draw() {
    svg.selectAll("g.layer").remove();
    const g = svg.append("g").attr("class", "layer");
    document.getElementById("hebb-eta-out").textContent = (+sEta.value).toFixed(3);
    document.getElementById("hebb-rho-out").textContent = PV.f2(+sRho.value);
    PV.frameBox(g, LX, LY, LS, LS);
    g.append("g").selectAll("circle").data(data.slice(0, 500).filter(d => Math.abs(d[0]) < 2.95 && Math.abs(d[1]) < 2.95)).join("circle")
      .attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", 1.8).attr("fill", C.muted).attr("fill-opacity", 0.35);
    g.append("line").attr("x1", sx(-2.9 * pc1[0])).attr("y1", sy(-2.9 * pc1[1])).attr("x2", sx(2.9 * pc1[0])).attr("y2", sy(2.9 * pc1[1]))
      .attr("stroke", C.ink).attr("stroke-dasharray", "5 4").attr("stroke-opacity", 0.6);
    /* unit circle for scale */
    g.append("circle").attr("cx", sx(0)).attr("cy", sy(0)).attr("r", sx(1) - sx(0)).attr("fill", "none").attr("stroke", C.faint).attr("stroke-dasharray", "2 3");
    const drawArrow = (w, color, m) => {
      const n = Math.hypot(w[0], w[1]); if (n < 1e-9) return;
      const L = Math.min(n, 2.7);
      g.append("line").attr("x1", sx(0)).attr("y1", sy(0)).attr("x2", sx(w[0] / n * L)).attr("y2", sy(w[1] / n * L))
        .attr("stroke", color).attr("stroke-width", 2.4).attr("marker-end", m);
    };
    drawArrow(wH, C.orange, mkH); drawArrow(wO, C.good, mkO);
    PV.label(g, LX + 6, LY + 14, "dashed: principal axis · dotted: ‖w‖ = 1", { size: 9.5 });
    PV.label(g, LX + 6, LY + LS - 8, "arrows clipped at length 2.7", { size: 9.5 });

    /* right: |w| vs step, log scale */
    PV.frameBox(g, RX, RY, RW, RH);
    const maxS = Math.max(400, step);
    const maxN = Math.max(10, d3.max(hist, h => Math.max(h[1], h[2])));
    const x = d3.scaleLinear().domain([0, maxS]).range([RX + 40, RX + RW - 10]);
    const y = d3.scaleLog().domain([0.1, maxN * 1.3]).range([RY + RH - 24, RY + 10]).clamp(true);
    PV.axisStyle(g.append("g").attr("transform", "translate(0," + (RY + RH - 24) + ")").call(d3.axisBottom(x).ticks(5)));
    PV.axisStyle(g.append("g").attr("transform", "translate(" + (RX + 40) + ",0)").call(d3.axisLeft(y).ticks(4, "~g")));
    g.append("line").attr("x1", x(0)).attr("x2", x(maxS)).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", C.faint).attr("stroke-dasharray", "2 3");
    const lh = d3.line().x(h => x(h[0])).y(h => y(Math.max(h[1], 0.1)));
    const lo = d3.line().x(h => x(h[0])).y(h => y(Math.max(h[2], 0.1)));
    g.append("path").attr("d", lh(hist)).attr("fill", "none").attr("stroke", C.orange).attr("stroke-width", 2);
    g.append("path").attr("d", lo(hist)).attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 2);
    PV.label(g, RX + RW / 2, RY + RH + 10, "samples seen", { anchor: "middle", size: 10 });
    PV.label(g, RX + 46, RY + 20, "‖w‖ (log scale)", { size: 10 });
    PV.label(g, RX + RW - 12, RY + 20, "Hebb", { anchor: "end", color: C.orange, bold: true });
    PV.label(g, RX + RW - 12, RY + 34, "Oja", { anchor: "end", color: C.good, bold: true });

    const nH = Math.hypot(wH[0], wH[1]), nO = Math.hypot(wO[0], wO[1]);
    document.getElementById("hebb-readout").innerHTML =
      "samples " + step + " · Hebb: ‖w‖ = <b>" + (nH > 1e4 ? nH.toExponential(2) : PV.f2(nH)) + "</b>, angle to principal axis " + PV.f1(angleTo(wH)) + "°" +
      " · Oja: ‖w‖ = <b>" + PV.f3(nO) + "</b>, angle " + PV.f1(angleTo(wO)) + "°";
  }
  document.getElementById("hebb-run").addEventListener("click", () => {
    if (timer) timer.stop();
    let done = 0;
    timer = d3.timer(() => {
      advance(8); done += 8; draw();
      if (done >= 400) { timer.stop(); timer = null; }
    });
  });
  document.getElementById("hebb-reset").addEventListener("click", reset);
  sRho.addEventListener("input", build);
  sEta.addEventListener("input", () => { document.getElementById("hebb-eta-out").textContent = (+sEta.value).toFixed(3); });
  build();
})();

/* ═════════ 3 · #pt-svg — step-through perceptron ═════════ */
(function () {
  const svg = d3.select("#pt-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const box = [-1, 1, -1, 1];
  const LX = 16, LY = 14, LS = 350;
  const sx = d3.scaleLinear().domain([box[0], box[1]]).range([LX, LX + LS]);
  const sy = d3.scaleLinear().domain([box[2], box[3]]).range([LY + LS, LY]);
  const mkW = PV.marker(svg, "pt-w", C.orange);
  const RX = 420, RY = 20, RW = 320, RH = 300;
  let pts, w, b, order, idx, epoch, em, hist, last, seed = 3, rs = PV.rng(99);
  const bgLayer = svg.append("g"), layer = svg.append("g"), ptLayer = svg.append("g"), right = svg.append("g");

  function newData() {
    const r = PV.rng(seed++);
    const ang = r() * Math.PI * 2, u = [Math.cos(ang), Math.sin(ang)], off = (r() - 0.5) * 0.4;
    pts = [];
    while (pts.length < 22) {
      const x = r() * 1.8 - 0.9, y = r() * 1.8 - 0.9, s = u[0] * x + u[1] * y - off;
      if (Math.abs(s) < 0.12) continue;
      pts.push({ x, y, t: s > 0 ? 1 : -1 });
    }
    resetW();
  }
  function resetW() { w = [0, 0]; b = 0; idx = 0; epoch = 1; em = 0; hist = []; last = null; order = null; draw(); }
  function dataChanged() { idx = 0; em = 0; hist = []; epoch = 1; order = null; last = null; draw(); }
  function step() {
    if (!pts.length) return;
    if (idx === 0 || !order) {
      order = d3.range(pts.length);
      if (document.getElementById("pt-shuffle").checked) PV.shuffle(order, rs);
    }
    const i = order[idx], p = pts[i];
    const a = w[0] * p.x + w[1] * p.y + b;
    const mistake = p.t * a <= 0;
    last = { i, a, mistake, oldW: w.slice(), oldB: b };
    if (mistake) { w = [w[0] + p.t * p.x, w[1] + p.t * p.y]; b += p.t; em++; }
    idx++;
    if (idx >= pts.length) { hist.push(em); em = 0; idx = 0; epoch++; order = null; }
  }
  function runEpoch() { const e = epoch; let guard = 0; do { step(); guard++; } while (epoch === e && guard < 5000); }
  function draw() {
    bgLayer.selectAll("*").remove(); layer.selectAll("*").remove(); right.selectAll("*").remove();
    PV.frameBox(bgLayer, LX, LY, LS, LS);
    const bg = bgLayer.append("rect").attr("x", LX).attr("y", LY).attr("width", LS).attr("height", LS).attr("fill", "transparent").style("cursor", "crosshair");
    bg.on("click", ev => {
      const [mx, my] = d3.pointer(ev, svg.node());
      pts.push({ x: sx.invert(mx), y: sy.invert(my), t: +document.getElementById("pt-cls").value });
      dataChanged();
    });
    const hasW = Math.hypot(w[0], w[1]) > 1e-12;
    if (hasW) {
      const hp = PV.halfPlane(w[0], w[1], b, box), hn = PV.halfPlane(-w[0], -w[1], -b, box);
      if (hp.length > 2) layer.append("polygon").attr("points", hp.map(p => sx(p[0]) + "," + sy(p[1])).join(" ")).attr("fill", C.pos).attr("fill-opacity", 0.08).style("pointer-events", "none");
      if (hn.length > 2) layer.append("polygon").attr("points", hn.map(p => sx(p[0]) + "," + sy(p[1])).join(" ")).attr("fill", C.neg).attr("fill-opacity", 0.08).style("pointer-events", "none");
    }
    if (last && last.mistake && Math.hypot(last.oldW[0], last.oldW[1]) > 1e-12) {
      const L0 = PV.clipLine(last.oldW[0], last.oldW[1], last.oldB, box);
      if (L0) layer.append("line").attr("x1", sx(L0[0][0])).attr("y1", sy(L0[0][1])).attr("x2", sx(L0[1][0])).attr("y2", sy(L0[1][1]))
        .attr("stroke", C.muted).attr("stroke-dasharray", "5 4").style("pointer-events", "none");
    }
    if (hasW) {
      const L = PV.clipLine(w[0], w[1], b, box);
      if (L) layer.append("line").attr("x1", sx(L[0][0])).attr("y1", sy(L[0][1])).attr("x2", sx(L[1][0])).attr("y2", sy(L[1][1]))
        .attr("stroke", C.ink).attr("stroke-width", 2).style("pointer-events", "none");
      const n2 = w[0] * w[0] + w[1] * w[1], n = Math.sqrt(n2);
      const p0 = [-b * w[0] / n2, -b * w[1] / n2];
      if (Math.abs(p0[0]) < 1 && Math.abs(p0[1]) < 1) {
        layer.append("line").attr("x1", sx(p0[0])).attr("y1", sy(p0[1])).attr("x2", sx(p0[0] + 0.28 * w[0] / n)).attr("y2", sy(p0[1] + 0.28 * w[1] / n))
          .attr("stroke", C.orange).attr("stroke-width", 2.2).attr("marker-end", mkW).style("pointer-events", "none");
        layer.append("text").attr("x", sx(p0[0] + 0.34 * w[0] / n)).attr("y", sy(p0[1] + 0.34 * w[1] / n) + 4).attr("fill", C.orange)
          .attr("font-size", 12).attr("text-anchor", "middle").text("w").style("pointer-events", "none");
      }
    }
    if (last) {
      const p = pts[last.i];
      if (p) {
        layer.append("circle").attr("cx", sx(p.x)).attr("cy", sy(p.y)).attr("r", 12).attr("fill", "none")
          .attr("stroke", last.mistake ? C.orange : C.good).attr("stroke-width", 2).style("pointer-events", "none");
        if (last.mistake) {
          /* the correction t·x drawn from the origin */
          layer.append("line").attr("x1", sx(0)).attr("y1", sy(0)).attr("x2", sx(p.t * p.x)).attr("y2", sy(p.t * p.y))
            .attr("stroke", C.orange).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.8).style("pointer-events", "none");
        }
      }
    }
    layer.append("circle").attr("cx", sx(0)).attr("cy", sy(0)).attr("r", 2.5).attr("fill", C.muted).style("pointer-events", "none");

    const drag = d3.drag()
      .on("start", function () { d3.select(this).style("cursor", "grabbing"); })
      .on("drag", function (ev, d) {
        const [mx, my] = d3.pointer(ev, svg.node());
        d.x = Math.max(-0.98, Math.min(0.98, sx.invert(mx))); d.y = Math.max(-0.98, Math.min(0.98, sy.invert(my)));
        d3.select(this).attr("cx", sx(d.x)).attr("cy", sy(d.y));
      })
      .on("end", function () { d3.select(this).style("cursor", "grab"); dataChanged(); });
    ptLayer.selectAll("*").remove();
    ptLayer.selectAll("circle").data(pts).join("circle").attr("class", "dragpt")
      .attr("cx", d => sx(d.x)).attr("cy", d => sy(d.y)).attr("r", 6.5)
      .attr("fill", d => d.t > 0 ? C.pos : C.neg).attr("stroke", d => (hasW && d.t * (w[0] * d.x + w[1] * d.y + b) <= 0) ? C.ink : "#0f1117")
      .attr("stroke-width", d => (hasW && d.t * (w[0] * d.x + w[1] * d.y + b) <= 0) ? 2.2 : 1)
      .style("cursor", "grab")
      .on("dblclick", (ev, d) => { ev.stopPropagation(); pts.splice(pts.indexOf(d), 1); dataChanged(); })
      .call(drag);

    /* right: mistakes per epoch */
    PV.frameBox(right, RX, RY, RW, RH);
    const shown = hist.slice(-30), off = hist.length - shown.length;
    const x = d3.scaleBand().domain(d3.range(shown.length).map(i => i + off + 1)).range([RX + 36, RX + RW - 10]).padding(0.15);
    const y = d3.scaleLinear().domain([0, Math.max(4, d3.max(shown) || 0)]).nice().range([RY + RH - 28, RY + 22]);
    PV.axisStyle(right.append("g").attr("transform", "translate(" + (RX + 36) + ",0)").call(d3.axisLeft(y).ticks(5)));
    if (shown.length) PV.axisStyle(right.append("g").attr("transform", "translate(0," + (RY + RH - 28) + ")")
      .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => shown.length < 12 || i % Math.ceil(shown.length / 10) === 0))));
    right.selectAll("rect.bar").data(shown).join("rect").attr("class", "bar")
      .attr("x", (d, i) => x(i + off + 1)).attr("width", x.bandwidth()).attr("y", d => y(d)).attr("height", d => y(0) - y(d))
      .attr("fill", d => d === 0 ? C.good : C.orange);
    PV.label(right, RX + 40, RY + 14, "mistakes per completed epoch", { size: 10.5, color: C.ink });
    PV.label(right, RX + RW / 2, RY + RH - 4, "epoch", { anchor: "middle", size: 10 });
    if (!shown.length) PV.label(right, RX + RW / 2, RY + RH / 2, "no completed epoch yet", { anchor: "middle" });

    /* readout */
    const sep = PV.perceptronFit(pts.slice(), 2000);
    let msg = "epoch " + epoch + ", example " + idx + "/" + pts.length + " · w = (" + PV.f2(w[0]) + ", " + PV.f2(w[1]) + "), b = " + PV.f2(b);
    if (last) {
      const p = pts[last.i];
      if (p) msg += "<br>presented x = (" + PV.f2(p.x) + ", " + PV.f2(p.y) + "), t = " + (p.t > 0 ? "+1" : "−1") + ", a = w·x + b = " + PV.f3(last.a) +
        (last.mistake ? " → t·a ≤ 0, <b>mistake</b>: w ← w " + (p.t > 0 ? "+" : "−") + " x, b ← b " + (p.t > 0 ? "+" : "−") + " 1"
          : " → correct, <b>no change</b>");
    }
    const errs = PV.errors([w[0], w[1], b], pts);
    msg += "<br>training errors now: <b>" + errs + "</b> · this data is " +
      (sep.converged ? "<b style=\"color:" + C.good + "\">separable</b> (a fixed-order run converges after " + sep.mistakes + " mistakes)"
        : "<b style=\"color:" + C.bad + "\">not separable</b> (no convergence in 2000 epochs)");
    document.getElementById("pt-readout").innerHTML = msg;
  }
  document.getElementById("pt-step").addEventListener("click", () => { step(); draw(); });
  document.getElementById("pt-epoch").addEventListener("click", () => { runEpoch(); draw(); });
  document.getElementById("pt-run").addEventListener("click", () => {
    let e = 0;
    while (e < 300) { runEpoch(); e++; if (hist.length && hist[hist.length - 1] === 0) break; }
    draw();
  });
  document.getElementById("pt-reset").addEventListener("click", resetW);
  document.getElementById("pt-new").addEventListener("click", newData);
  newData();
})();

/* ═════════ 4 · #bd-svg — the mistake bound ═════════ */
(function () {
  const svg = d3.select("#bd-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const sG = document.getElementById("bd-g"), sN = document.getElementById("bd-n");
  let seed = 11;
  const LX = 16, LY = 14, LS = 290;
  const sx = d3.scaleLinear().domain([-1.05, 1.05]).range([LX, LX + LS]);
  const sy = d3.scaleLinear().domain([-1.05, 1.05]).range([LY + LS, LY]);
  const RX = 360, RY = 14, RW = 380, RH = 290;

  function makeData(gam, N, s) {
    const r = PV.rng(s), ang = r() * Math.PI, u = [Math.cos(ang), Math.sin(ang)], pts = [];
    let guard = 0;
    while (pts.length < N && guard < 200000) {
      guard++;
      const x = r() * 2 - 1, y = r() * 2 - 1;
      if (x * x + y * y > 1) continue;
      const m = u[0] * x + u[1] * y;
      if (Math.abs(m) < gam) continue;
      pts.push([x, y, m > 0 ? 1 : -1]);
    }
    return { pts, u };
  }
  /* homogeneous perceptron (no bias), random order each epoch; counts mistakes */
  function run(pts, r) {
    let w0 = 0, w1 = 0, m = 0;
    const ord = d3.range(pts.length);
    for (let e = 0; e < 3000; e++) {
      PV.shuffle(ord, r);
      let em = 0;
      for (const i of ord) { const p = pts[i]; if (p[2] * (w0 * p[0] + w1 * p[1]) <= 0) { w0 += p[2] * p[0]; w1 += p[2] * p[1]; em++; } }
      m += em;
      if (em === 0) break;
    }
    return m;
  }
  function draw() {
    svg.selectAll("g.layer").remove();
    const g = svg.append("g").attr("class", "layer");
    const gam = +sG.value, N = +sN.value;
    document.getElementById("bd-g-out").textContent = PV.f2(gam);
    document.getElementById("bd-n-out").textContent = N;
    const D = makeData(gam, N, seed);
    PV.frameBox(g, LX, LY, LS, LS);
    g.append("circle").attr("cx", sx(0)).attr("cy", sy(0)).attr("r", sx(1) - sx(0)).attr("fill", "none").attr("stroke", C.faint);
    const u = D.u, v = [-u[1], u[0]];
    [-gam, gam].forEach(o => g.append("line").attr("x1", sx(o * u[0] - v[0])).attr("y1", sy(o * u[1] - v[1])).attr("x2", sx(o * u[0] + v[0])).attr("y2", sy(o * u[1] + v[1]))
      .attr("stroke", C.faint).attr("stroke-dasharray", "3 3"));
    g.append("line").attr("x1", sx(-v[0])).attr("y1", sy(-v[1])).attr("x2", sx(v[0])).attr("y2", sy(v[1])).attr("stroke", C.ink).attr("stroke-opacity", 0.7);
    g.append("g").selectAll("circle").data(D.pts).join("circle").attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", 2.6)
      .attr("fill", d => d[2] > 0 ? C.pos : C.neg);
    PV.label(g, LX + 6, LY + 14, "R = 1 · empty band of half-width γ", { size: 9.5 });

    /* sweep */
    const grid = [0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.4];
    if (!grid.some(q => Math.abs(q - gam) < 1e-9)) grid.push(gam);
    const r = PV.rng(1234);
    const res = grid.map(q => {
      const Dq = Math.abs(q - gam) < 1e-9 ? D : makeData(q, N, seed + 1000 + Math.round(q * 1000));
      const ms = d3.range(30).map(() => run(Dq.pts, r));
      return { g: q, ms, cur: Math.abs(q - gam) < 1e-9 };
    });
    PV.frameBox(g, RX, RY, RW, RH);
    const x = d3.scaleLog().domain([0.025, 0.45]).range([RX + 44, RX + RW - 12]);
    const y = d3.scaleLog().domain([1, 2000]).range([RY + RH - 26, RY + 12]).clamp(true);
    PV.axisStyle(g.append("g").attr("transform", "translate(0," + (RY + RH - 26) + ")").call(d3.axisBottom(x).tickValues([0.03, 0.05, 0.1, 0.2, 0.4]).tickFormat(d3.format("~g"))));
    PV.axisStyle(g.append("g").attr("transform", "translate(" + (RX + 44) + ",0)").call(d3.axisLeft(y).tickValues([1, 10, 100, 1000]).tickFormat(d3.format("~g"))));
    const curve = d3.range(0.025, 0.451, 0.005).map(q => [q, 1 / (q * q)]);
    g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(curve)).attr("fill", "none").attr("stroke", C.orange).attr("stroke-width", 2);
    PV.label(g, x(0.06) + 6, y(1 / 0.0036) - 6, "bound 1/γ²", { color: C.orange, size: 10.5, bold: true });
    const jr = PV.rng(5);
    res.forEach(R => R.ms.forEach(m => g.append("circle").attr("cx", x(R.g) + (jr() - 0.5) * 8).attr("cy", y(Math.max(m, 1)))
      .attr("r", R.cur ? 3 : 2).attr("fill", R.cur ? C.ink : C.pos).attr("fill-opacity", R.cur ? 0.9 : 0.5)));
    PV.label(g, RX + RW / 2, RY + RH + 0, "margin γ (log)", { anchor: "middle", size: 10 });
    PV.label(g, RX + 50, RY + 24, "mistakes until convergence, 30 orderings each (log)", { size: 10 });
    const cur = res.find(R => R.cur), bound = 1 / (gam * gam);
    const mx = d3.max(cur.ms), mean = d3.mean(cur.ms);
    const allOk = res.every(R => d3.max(R.ms) <= 1 / (R.g * R.g) + 1e-9);
    document.getElementById("bd-readout").innerHTML = "γ = " + PV.f2(gam) + ": bound (R/γ)² = <b>" + d3.format(",.0f")(bound) + "</b> · mistakes over 30 orderings: mean " +
      PV.f1(mean) + ", max <b>" + mx + "</b> (" + d3.format(".0%")(mx / bound) + " of the bound) · every run at every margin within the bound: " +
      (allOk ? "<b style=\"color:" + C.good + "\">yes</b>" : "<b style=\"color:" + C.bad + "\">NO</b>");
  }
  sG.addEventListener("change", draw);
  sG.addEventListener("input", () => { document.getElementById("bd-g-out").textContent = PV.f2(+sG.value); });
  sN.addEventListener("change", draw);
  sN.addEventListener("input", () => { document.getElementById("bd-n-out").textContent = sN.value; });
  document.getElementById("bd-new").addEventListener("click", () => { seed += 17; draw(); });
  draw();
})();

/* ═════════ 5 · #pk-svg — pocket and averaged perceptron ═════════ */
(function () {
  const svg = d3.select("#pk-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const sO = document.getElementById("pk-ov");
  const box = [-2, 2, -2, 2];
  const LX = 16, LY = 14, LS = 300;
  const sx = d3.scaleLinear().domain([box[0], box[1]]).range([LX, LX + LS]);
  const sy = d3.scaleLinear().domain([box[2], box[3]]).range([LY + LS, LY]);
  const RX = 370, RY = 14, RW = 370, RH = 300, TOTAL = 2000;
  let seed = 21, pts, st, timer = null;

  function newData() {
    const r = PV.rng(seed), sd = 0.15 + 0.35 * (+sO.value);
    pts = [];
    for (let i = 0; i < 120; i++) {
      const t = i % 2 ? 1 : -1;
      pts.push({ x: 0.6 * t + sd * PV.gauss(r), y: 0.4 * t + sd * PV.gauss(r), t });
    }
    reset();
  }
  function reset() {
    if (timer) { timer.stop(); timer = null; }
    st = { w: [0, 0, 0], pocket: [0, 0, 0], pErr: pts.length, sum: [0, 0, 0], T: 0,
      r: PV.rng(seed + 7), ord: [], k: 0, hist: [] };
    draw();
  }
  const err = w => PV.errors(w, pts) / pts.length;
  function advance(n) {
    for (let s = 0; s < n && st.T < TOTAL; s++) {
      if (st.k === 0) st.ord = PV.shuffle(d3.range(pts.length), st.r);
      const p = pts[st.ord[st.k]];
      st.k = (st.k + 1) % pts.length;
      const w = st.w;
      if (p.t * (w[0] * p.x + w[1] * p.y + w[2]) <= 0) {
        st.w = [w[0] + p.t * p.x, w[1] + p.t * p.y, w[2] + p.t];
        /* ratchet: the new weights go in the pocket only if they make fewer
           errors on the WHOLE training set than the pocketed ones */
        const e = PV.errors(st.w, pts);
        if (e < st.pErr) { st.pocket = st.w.slice(); st.pErr = e; }
      }
      st.sum = [st.sum[0] + st.w[0], st.sum[1] + st.w[1], st.sum[2] + st.w[2]];
      st.T++;
      if (st.T % 10 === 0) st.hist.push([st.T, err(st.w), st.pErr / pts.length, err(st.sum)]);
    }
  }
  function drawLine(g, w, color, width, dash) {
    if (Math.hypot(w[0], w[1]) < 1e-12) return;
    const L = PV.clipLine(w[0], w[1], w[2], box);
    if (L) g.append("line").attr("x1", sx(L[0][0])).attr("y1", sy(L[0][1])).attr("x2", sx(L[1][0])).attr("y2", sy(L[1][1]))
      .attr("stroke", color).attr("stroke-width", width).attr("stroke-dasharray", dash || null);
  }
  function draw() {
    svg.selectAll("g.layer").remove();
    const g = svg.append("g").attr("class", "layer");
    document.getElementById("pk-ov-out").textContent = PV.f1(+sO.value);
    PV.frameBox(g, LX, LY, LS, LS);
    g.append("g").selectAll("circle").data(pts).join("circle").attr("cx", d => sx(d.x)).attr("cy", d => sy(d.y)).attr("r", 3.2)
      .attr("fill", d => d.t > 0 ? C.pos : C.neg).attr("fill-opacity", 0.8);
    drawLine(g, st.w, C.muted, 1.5, "5 4");
    drawLine(g, st.sum, C.orange, 2.2);
    drawLine(g, st.pocket, C.good, 2.2);

    PV.frameBox(g, RX, RY, RW, RH);
    const x = d3.scaleLinear().domain([0, TOTAL]).range([RX + 40, RX + RW - 10]);
    const ymax = Math.max(0.3, d3.max(st.hist, h => Math.max(h[1], h[2], h[3])) || 0);
    const y = d3.scaleLinear().domain([0, Math.min(1, ymax)]).nice().range([RY + RH - 26, RY + 14]);
    PV.axisStyle(g.append("g").attr("transform", "translate(0," + (RY + RH - 26) + ")").call(d3.axisBottom(x).ticks(5)));
    PV.axisStyle(g.append("g").attr("transform", "translate(" + (RX + 40) + ",0)").call(d3.axisLeft(y).ticks(5, "%")));
    [[1, C.muted, 1.2], [3, C.orange, 2], [2, C.good, 2]].forEach(([k, c, sw]) =>
      g.append("path").attr("d", d3.line().x(h => x(h[0])).y(h => y(Math.min(h[k], 1)))(st.hist)).attr("fill", "none").attr("stroke", c).attr("stroke-width", sw));
    PV.label(g, RX + RW / 2, RY + RH - 2, "examples presented", { anchor: "middle", size: 10 });
    PV.label(g, RX + 46, RY + 26, "training error rate", { size: 10 });

    const bayes = pts.reduce((s, p) => s + ((0.6 * p.x + 0.4 * p.y) * p.t <= 0 ? 1 : 0), 0) / pts.length;
    document.getElementById("pk-readout").innerHTML = "presented " + st.T + " / " + TOTAL +
      " · current: <b>" + d3.format(".1%")(err(st.w)) + "</b> · pocket: <b style=\"color:" + C.good + "\">" + d3.format(".1%")(st.pErr / pts.length) + "</b>" +
      " · averaged: <b style=\"color:" + C.orange + "\">" + d3.format(".1%")(err(st.sum)) + "</b>" +
      " · the ideal line (perpendicular to the centre difference) scores " + d3.format(".1%")(bayes);
  }
  document.getElementById("pk-run").addEventListener("click", () => {
    if (timer) timer.stop();
    if (st.T >= TOTAL) reset();
    timer = d3.timer(() => {
      advance(40); draw();
      if (st.T >= TOTAL) { timer.stop(); timer = null; }
    });
  });
  document.getElementById("pk-reset").addEventListener("click", reset);
  document.getElementById("pk-new").addEventListener("click", () => { seed += 13; newData(); });
  sO.addEventListener("input", newData);
  newData();
})();

/* ═════════ 6 · #pva-svg — perceptron vs Adaline ═════════ */
(function () {
  const svg = d3.select("#pva-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const box = [-1.5, 1.5, -1.5, 1.5];
  const LX = 16, LY = 10, LS = 340;
  const sx = d3.scaleLinear().domain([box[0], box[1]]).range([LX, LX + LS]);
  const sy = d3.scaleLinear().domain([box[2], box[3]]).range([LY + LS, LY]);
  const shadeL = svg.append("g"), lineL = svg.append("g"), ptL = svg.append("g"), right = svg.append("g");
  const outSpots = [[1.4, -1.4], [1.3, -1.45], [1.45, -1.25], [1.2, -1.4], [1.4, -1.1]];
  let pts, nOut;

  function base() {
    const r = PV.rng(42);
    pts = [];
    for (let i = 0; i < 12; i++) pts.push({ x: -0.6 + 0.4 * r(), y: 1.2 * r(), t: 1 });
    for (let i = 0; i < 12; i++) pts.push({ x: -1.1 + 0.4 * r(), y: 1.2 * r(), t: -1 });
    nOut = 0;
    draw();
  }
  function line(g, w, color) {
    if (Math.hypot(w[0], w[1]) < 1e-12) return;
    const L = PV.clipLine(w[0], w[1], w[2], box);
    if (L) g.append("line").attr("x1", sx(L[0][0])).attr("y1", sy(L[0][1])).attr("x2", sx(L[1][0])).attr("y2", sy(L[1][1]))
      .attr("stroke", color).attr("stroke-width", 2.4);
  }
  const margin = (w, P) => {
    const n = Math.hypot(w[0], w[1]) || 1;
    return d3.min(P, p => p.t * (w[0] * p.x + w[1] * p.y + w[2]) / n);
  };
  function draw() {
    shadeL.selectAll("*").remove(); lineL.selectAll("*").remove(); ptL.selectAll("*").remove(); right.selectAll("*").remove();
    PV.frameBox(shadeL, LX, LY, LS, LS);
    const A = PV.adalineLS(pts), P = PV.perceptronFit(pts, 3000);
    if (document.getElementById("pva-shade").checked) {
      const n = 34, cs = (box[1] - box[0]) / n;
      const cscale = d3.scaleLinear().domain([-2, 0, 2]).range([C.neg, C.panel2, C.pos]).clamp(true);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const x = box[0] + (i + 0.5) * cs, y = box[2] + (j + 0.5) * cs, a = A[0] * x + A[1] * y + A[2];
        shadeL.append("rect").attr("x", sx(x - cs / 2)).attr("y", sy(y + cs / 2)).attr("width", sx(cs) - sx(0) + 0.5).attr("height", sx(cs) - sx(0) + 0.5)
          .attr("fill", cscale(a)).attr("fill-opacity", 0.35);
      }
    }
    line(lineL, P.w, C.ink);
    line(lineL, A, C.orange);
    const drag = d3.drag()
      .on("drag", function (ev, d) {
        const [mx, my] = d3.pointer(ev, svg.node());
        d.x = Math.max(-1.45, Math.min(1.45, sx.invert(mx))); d.y = Math.max(-1.45, Math.min(1.45, sy.invert(my)));
        draw();
      });
    ptL.selectAll("circle").data(pts).join("circle").attr("class", "dragpt")
      .attr("cx", d => sx(d.x)).attr("cy", d => sy(d.y)).attr("r", d => d.out ? 8 : 6)
      .attr("fill", d => d.t > 0 ? C.pos : C.neg).attr("stroke", d => d.out ? C.ink : "#0f1117").attr("stroke-width", d => d.out ? 2 : 1)
      .style("cursor", "grab").call(drag);

    /* right: per-rule summary bars */
    const RX = 400, RY = 20;
    const eP = PV.errors(P.w, pts), eA = PV.errors(A, pts);
    const mP = margin(P.w, pts), mA = margin(A, pts);
    PV.label(right, RX, RY + 4, "training errors", { color: C.ink, bold: true });
    const xs = d3.scaleLinear().domain([0, Math.max(4, eP, eA)]).range([0, 220]);
    [["perceptron", eP, C.ink], ["Adaline", eA, C.orange]].forEach(([n, e, c], i) => {
      PV.label(right, RX, RY + 32 + i * 28, n, { size: 11 });
      right.append("rect").attr("x", RX + 80).attr("y", RY + 21 + i * 28).attr("width", Math.max(2, xs(e))).attr("height", 14).attr("fill", c).attr("fill-opacity", 0.85);
      PV.label(right, RX + 86 + Math.max(2, xs(e)), RY + 32 + i * 28, String(e), { size: 11, color: C.ink });
    });
    PV.label(right, RX, RY + 110, "worst signed distance to the line (the margin)", { color: C.ink, bold: true });
    [["perceptron", mP, C.ink], ["Adaline", mA, C.orange]].forEach(([n, m], i) => {
      PV.label(right, RX, RY + 136 + i * 22, n + ":  " + (m > 0 ? "+" : "") + PV.f3(m) + (m <= 0 ? "  (misclassifies)" : ""), { size: 11, color: m > 0 ? C.good : C.bad });
    });
    PV.label(right, RX, RY + 200, "Adaline's linear outputs on the outliers:", { color: C.ink, bold: true });
    const outs = pts.filter(p => p.out);
    if (!outs.length) PV.label(right, RX, RY + 222, "none yet — press “Add far outlier”", { size: 11 });
    outs.forEach((p, i) => PV.label(right, RX, RY + 222 + i * 18, "a = " + PV.f2(A[0] * p.x + A[1] * p.y + A[2]) + "  (target +1, residual " + PV.f2(1 - (A[0] * p.x + A[1] * p.y + A[2])) + ")", { size: 10.5 }));

    document.getElementById("pva-readout").innerHTML = "perceptron: w = (" + PV.f2(P.w[0]) + ", " + PV.f2(P.w[1]) + "), b = " + PV.f2(P.w[2]) +
      (P.converged ? " — converged after " + P.mistakes + " mistakes" : " — did not converge (data not separable)") +
      " · Adaline least squares: w = (" + PV.f3(A[0]) + ", " + PV.f3(A[1]) + "), b = " + PV.f3(A[2]) + ", MSE = " +
      PV.f3(d3.mean(pts, p => (p.t - (A[0] * p.x + A[1] * p.y + A[2])) ** 2));
  }
  document.getElementById("pva-out").addEventListener("click", () => {
    const s = outSpots[nOut % outSpots.length];
    pts.push({ x: s[0], y: s[1], t: 1, out: true }); nOut++; draw();
  });
  document.getElementById("pva-reset").addEventListener("click", base);
  document.getElementById("pva-shade").addEventListener("change", draw);
  base();
})();

/* ═════════ 7 · #lms-svg — the error bowl ═════════ */
(function () {
  const svg = d3.select("#lms-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const sE = document.getElementById("lms-eta"), sR = document.getElementById("lms-rho"), sM = document.getElementById("lms-mode");
  const LX = 16, LY = 10, LS = 330, RX = 380, RY = 10, RW = 360, RH = 330, SPAN = 3;
  let data, R, p, wStar, Emin, lmax, lmin, start = null;

  function build() {
    const rho = +sR.value, r = PV.rng(31), N = 200;
    const wt = [1.0, -0.5];
    data = [];
    for (let i = 0; i < N; i++) {
      const z1 = PV.gauss(r), z2 = PV.gauss(r);
      const x = [z1, rho * z1 + Math.sqrt(1 - rho * rho) * z2];
      data.push({ x, t: wt[0] * x[0] + wt[1] * x[1] + 0.3 * PV.gauss(r) });
    }
    R = [[0, 0], [0, 0]]; p = [0, 0]; let tt = 0;
    data.forEach(d => {
      R[0][0] += d.x[0] * d.x[0] / N; R[0][1] += d.x[0] * d.x[1] / N; R[1][1] += d.x[1] * d.x[1] / N;
      p[0] += d.t * d.x[0] / N; p[1] += d.t * d.x[1] / N; tt += d.t * d.t / N;
    });
    R[1][0] = R[0][1];
    const det = R[0][0] * R[1][1] - R[0][1] * R[1][0];
    wStar = [(R[1][1] * p[0] - R[0][1] * p[1]) / det, (R[0][0] * p[1] - R[1][0] * p[0]) / det];
    const tr = R[0][0] + R[1][1];
    lmax = tr / 2 + Math.sqrt(tr * tr / 4 - det); lmin = tr / 2 - Math.sqrt(tr * tr / 4 - det);
    Emin = E(wStar, tt);
    E.tt = tt;
    if (!start) start = [wStar[0] - 2.3, wStar[1] + 1.9];
    draw();
  }
  function E(w, tt) {
    tt = tt === undefined ? E.tt : tt;
    return 0.5 * (tt - 2 * (p[0] * w[0] + p[1] * w[1]) + (w[0] * (R[0][0] * w[0] + R[0][1] * w[1]) + w[1] * (R[1][0] * w[0] + R[1][1] * w[1])));
  }
  function draw() {
    svg.selectAll("g.layer").remove();
    const g = svg.append("g").attr("class", "layer");
    const frac = +sE.value, eta = frac * 2 / lmax, mode = sM.value;
    document.getElementById("lms-eta-out").textContent = PV.f2(frac);
    document.getElementById("lms-rho-out").textContent = PV.f2(+sR.value);
    const dom = [[wStar[0] - SPAN, wStar[0] + SPAN], [wStar[1] - SPAN, wStar[1] + SPAN]];
    const sx = d3.scaleLinear().domain(dom[0]).range([LX, LX + LS]);
    const sy = d3.scaleLinear().domain(dom[1]).range([LY + LS, LY]);
    PV.frameBox(g, LX, LY, LS, LS);
    /* contours */
    const n = 70, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const w = [dom[0][0] + (i + 0.5) / n * 2 * SPAN, dom[1][1] - (j + 0.5) / n * 2 * SPAN];
      vals[j * n + i] = Math.log(E(w) - Emin + 1e-4);
    }
    const thr = d3.range(-4, 3.6, 0.55);
    const cont = d3.contours().size([n, n]).thresholds(thr)(vals);
    const proj = d3.geoPath(d3.geoTransform({ point: function (x, y) { this.stream.point(LX + x / n * LS, LY + y / n * LS); } }));
    const cc = d3.scaleLinear().domain([-4, 3.5]).range([0.9, 0.15]);
    g.append("g").selectAll("path").data(cont).join("path").attr("d", proj).attr("fill", "none")
      .attr("stroke", C.accent).attr("stroke-opacity", d => cc(d.value)).attr("stroke-width", 1);
    g.append("circle").attr("cx", sx(wStar[0])).attr("cy", sy(wStar[1])).attr("r", 4).attr("fill", C.good);
    PV.label(g, sx(wStar[0]) + 7, sy(wStar[1]) - 6, "w* = R⁻¹p", { color: C.good, size: 10 });
    PV.label(g, LX + 6, LY + 14, "w₁ →", { size: 10 });
    PV.label(g, LX + 6, LY + LS - 6, "contours of E(w), log-spaced", { size: 9.5 });

    /* trajectory */
    let w = start.slice();
    const path = [w.slice()], errs = [[0, E(w) - Emin]];
    const r = PV.rng(77);
    const iters = mode === "batch" ? 60 : 600;
    let blew = false;
    for (let k = 1; k <= iters; k++) {
      if (mode === "batch") {
        const gr = [R[0][0] * w[0] + R[0][1] * w[1] - p[0], R[1][0] * w[0] + R[1][1] * w[1] - p[1]];
        w = [w[0] - eta * gr[0], w[1] - eta * gr[1]];
      } else {
        const d = data[Math.floor(r() * data.length)];
        const e = d.t - (w[0] * d.x[0] + w[1] * d.x[1]);
        w = [w[0] + eta * e * d.x[0], w[1] + eta * e * d.x[1]];
      }
      if (!isFinite(w[0]) || Math.abs(w[0]) > 1e6 || Math.abs(w[1]) > 1e6) { blew = true; break; }
      path.push(w.slice()); errs.push([k, E(w) - Emin]);
    }
    const clip = q => [Math.max(dom[0][0] - 1, Math.min(dom[0][1] + 1, q[0])), Math.max(dom[1][0] - 1, Math.min(dom[1][1] + 1, q[1]))];
    g.append("clipPath").attr("id", "lms-clip").append("rect").attr("x", LX).attr("y", LY).attr("width", LS).attr("height", LS);
    const gp = g.append("g").attr("clip-path", "url(#lms-clip)");
    gp.append("path").attr("d", d3.line().x(q => sx(clip(q)[0])).y(q => sy(clip(q)[1]))(path)).attr("fill", "none")
      .attr("stroke", C.orange).attr("stroke-width", mode === "batch" ? 1.6 : 0.9).attr("stroke-opacity", 0.9);
    if (mode === "batch") gp.selectAll("circle.it").data(path.slice(0, 40)).join("circle").attr("class", "it")
      .attr("cx", q => sx(clip(q)[0])).attr("cy", q => sy(clip(q)[1])).attr("r", 2).attr("fill", C.orange);
    const handle = g.append("circle").attr("class", "dragpt").attr("cx", sx(start[0])).attr("cy", sy(start[1])).attr("r", 7)
      .attr("fill", C.orange).attr("stroke", C.ink).attr("stroke-width", 1.5).style("cursor", "grab");
    handle.call(d3.drag().on("drag", (ev) => {
      const [mx, my] = d3.pointer(ev, svg.node());
      start = [Math.max(dom[0][0], Math.min(dom[0][1], sx.invert(mx))), Math.max(dom[1][0], Math.min(dom[1][1], sy.invert(my)))];
      draw();
    }));

    /* right: excess error vs iteration */
    PV.frameBox(g, RX, RY, RW, RH);
    const x = d3.scaleLinear().domain([0, iters]).range([RX + 44, RX + RW - 10]);
    const y = d3.scaleLog().domain([1e-6, 1e4]).range([RY + RH - 26, RY + 12]).clamp(true);
    PV.axisStyle(g.append("g").attr("transform", "translate(0," + (RY + RH - 26) + ")").call(d3.axisBottom(x).ticks(5)));
    PV.axisStyle(g.append("g").attr("transform", "translate(" + (RX + 44) + ",0)").call(d3.axisLeft(y).tickValues([1e-6, 1e-4, 1e-2, 1, 100, 1e4]).tickFormat(d3.format("~e"))));
    g.append("path").attr("d", d3.line().x(q => x(q[0])).y(q => y(Math.max(q[1], 1e-7)))(errs)).attr("fill", "none").attr("stroke", C.orange).attr("stroke-width", 1.6);
    PV.label(g, RX + RW / 2, RY + RH - 2, mode === "batch" ? "iteration (full-batch step)" : "iteration (one sample)", { anchor: "middle", size: 10 });
    PV.label(g, RX + 50, RY + 26, "E(w) − E(w*)  (log)", { size: 10 });

    const reg = l => { const q = eta * l; return q < 1 ? "monotone" : (q < 2 ? "oscillating" : "diverging"); };
    const hit = errs.find(q => q[1] < 1e-3);
    document.getElementById("lms-readout").innerHTML = "λmax = " + PV.f3(lmax) + ", λmin = " + PV.f3(lmin) + ", κ = " + PV.f1(lmax / lmin) +
      " · <span class=\"keep\">η</span> = " + PV.f3(eta) + " · <span class=\"keep\">η</span>λmax = <b>" + PV.f2(eta * lmax) + "</b> (" + reg(lmax) + " along the steep axis), <span class=\"keep\">η</span>λmin = " + PV.f3(eta * lmin) + " (" + reg(lmin) + ")" +
      (mode === "lms" ? " · LMS guide 2/tr(R) ⇒ <span class=\"keep\">η</span> &lt; " + PV.f3(2 / (R[0][0] + R[1][1])) : "") +
      "<br>" + (blew ? "<b style=\"color:" + C.bad + "\">diverged</b>" : (hit ? "excess error below 10⁻³ after <b>" + hit[0] + "</b> iterations" : "excess error still above 10⁻³ after " + iters + " iterations" + (mode === "lms" ? " (LMS settles at a noise floor ∝ <span class=\"keep\">η</span>)" : "")));
  }
  sE.addEventListener("input", draw);
  sM.addEventListener("change", draw);
  sR.addEventListener("input", build);
  build();
})();

/* ═════════ 8 · #cv-svg — Cover's counting ═════════ */
(function () {
  const svg = d3.select("#cv-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const sD = document.getElementById("cv-d");
  const X0 = 60, Y0 = 16, W = 560, H = 240;
  /* fraction = P(Binomial(N−1, ½) ≤ d−1), computed in log space */
  function frac(N, d) {
    if (N <= d) return 1;
    const n = N - 1;
    let logp = -n * Math.LN2, s = 0; /* k = 0 term */
    for (let k = 0; k <= d - 1; k++) {
      s += Math.exp(logp);
      logp += Math.log((n - k) / (k + 1));
    }
    return Math.min(1, s);
  }
  function draw() {
    svg.selectAll("g.layer").remove();
    const g = svg.append("g").attr("class", "layer");
    const dh = +sD.value;
    document.getElementById("cv-d-out").textContent = dh;
    const x = d3.scaleLinear().domain([0, 4]).range([X0, X0 + W]);
    const y = d3.scaleLinear().domain([0, 1]).range([Y0 + H, Y0]);
    PV.axisStyle(g.append("g").attr("transform", "translate(0," + (Y0 + H) + ")").call(d3.axisBottom(x).ticks(8)));
    PV.axisStyle(g.append("g").attr("transform", "translate(" + X0 + ",0)").call(d3.axisLeft(y).ticks(5, "%")));
    g.append("line").attr("x1", x(2)).attr("x2", x(2)).attr("y1", y(0)).attr("y2", y(1)).attr("stroke", C.faint).attr("stroke-dasharray", "3 3");
    g.append("line").attr("x1", x(0)).attr("x2", x(4)).attr("y1", y(0.5)).attr("y2", y(0.5)).attr("stroke", C.faint).attr("stroke-dasharray", "3 3");
    const ds = [2, 5, 15, 60].filter(d => d !== dh);
    const col = d3.scaleSequential(d3.interpolateBlues).domain([-20, 60]);
    const curve = d => d3.range(1, 4 * d + 1).map(N => [N / d, frac(N, d)]);
    ds.forEach(d => {
      g.append("path").attr("d", d3.line().x(q => x(q[0])).y(q => y(q[1]))(curve(d))).attr("fill", "none").attr("stroke", col(d)).attr("stroke-width", 1.4);
      const q = curve(d).find(v => v[0] >= 2.6) || [2.6, 0];
      PV.label(g, x(q[0]) + 4, y(q[1]) - 3, "d=" + d, { size: 9.5, color: col(d) });
    });
    const ch = curve(dh);
    g.append("path").attr("d", d3.line().x(q => x(q[0])).y(q => y(q[1]))(ch)).attr("fill", "none").attr("stroke", C.orange).attr("stroke-width", 2.6);
    if (dh <= 12) g.selectAll("circle.pt").data(ch).join("circle").attr("class", "pt").attr("cx", q => x(q[0])).attr("cy", q => y(q[1])).attr("r", 3).attr("fill", C.orange);
    PV.label(g, X0 + W / 2, Y0 + H + 32, "N / d  (patterns per weight)", { anchor: "middle", size: 10.5 });
    PV.label(g, X0 + 8, Y0 + 10, "fraction of the 2ᴺ labellings that are linearly separable", { size: 10 });
    PV.label(g, x(2) + 4, Y0 + 24, "N = 2d", { size: 10, color: C.ink });
    const at = k => frac(Math.round(k * dh), dh);
    document.getElementById("cv-readout").innerHTML = "d = " + dh + ": at N = d " + d3.format(".1%")(at(1)) + " · N = 1.5d " + d3.format(".1%")(at(1.5)) +
      " · N = 2d <b>" + d3.format(".1%")(at(2)) + "</b> · N = 2.5d " + d3.format(".1%")(at(2.5)) + " · N = 3d " + d3.format(".2%")(at(3)) +
      (dh === 3 ? " · (d = 3 is a 2-input unit with bias: N = 4 gives 14/16 = 87.5%)" : "");
  }
  sD.addEventListener("input", draw);
  draw();
})();

/* ═════════ 9 · #xr-svg — XOR ═════════ */
(function () {
  const svg = d3.select("#xr-svg");
  if (svg.empty()) return;
  const C = PV.col;
  const sA = document.getElementById("xr-ang"), sO = document.getElementById("xr-off"), cH = document.getElementById("xr-hidden");
    const box = [-0.3, 1.3, -0.3, 1.3];
  const LX = 20, LY = 10, LS = 300, RX = 400, RY = 10, RS = 300;
  const sx = d3.scaleLinear().domain([box[0], box[1]]).range([LX, LX + LS]);
  const sy = d3.scaleLinear().domain([box[2], box[3]]).range([LY + LS, LY]);
  const rows = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  let hist = null;
  /* line: n·(x − c) = off, fires on the positive side */
  const lineW = (ang, off) => { const a = ang * Math.PI / 180, n = [Math.cos(a), Math.sin(a)]; return [n[0], n[1], -(n[0] * 0.5 + n[1] * 0.5) - off]; };
  const correct = w => rows.reduce((s, r) => s + (((w[0] * r[0] + w[1] * r[1] + w[2]) > 0 ? 1 : 0) === r[2] ? 1 : 0), 0);
  function drawLine(g, sxx, syy, w, bx, color, width, dash) {
    const L = PV.clipLine(w[0], w[1], w[2], bx);
    if (L) g.append("line").attr("x1", sxx(L[0][0])).attr("y1", syy(L[0][1])).attr("x2", sxx(L[1][0])).attr("y2", syy(L[1][1]))
      .attr("stroke", color).attr("stroke-width", width).attr("stroke-dasharray", dash || null);
  }
  function corners(g, sxx, syy, P, lab) {
    P.forEach((r, i) => {
      g.append("circle").attr("cx", sxx(r[0])).attr("cy", syy(r[1])).attr("r", 11)
        .attr("fill", r[2] ? C.pos : C.panel).attr("stroke", r[2] ? C.pos : C.neg).attr("stroke-width", 2.2);
      if (lab) PV.label(g, sxx(r[0]) + (r[0] > 0.5 ? 14 : -14), syy(r[1]) + (r[1] > 0.5 ? -12 : 20), lab[i], { size: 9.5, anchor: r[0] > 0.5 ? "start" : "end" });
    });
  }
  function draw() {
    svg.selectAll("g.layer").remove();
    const g = svg.append("g").attr("class", "layer");
    const ang = +sA.value, off = +sO.value;
    document.getElementById("xr-ang-out").textContent = ang + "°";
    document.getElementById("xr-off-out").textContent = PV.f2(off);
    PV.frameBox(g, LX, LY, LS, LS);
    const labs = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"];
    let msg;
    if (!cH.checked) {
      const w = lineW(ang, off);
      const hp = PV.halfPlane(w[0], w[1], w[2], box);
      if (hp.length > 2) g.append("polygon").attr("points", hp.map(p => sx(p[0]) + "," + sy(p[1])).join(" ")).attr("fill", C.good).attr("fill-opacity", 0.12);
      drawLine(g, sx, sy, w, box, C.ink, 2);
      corners(g, sx, sy, rows, labs);
      const k = correct(w);
      rows.forEach(r => {
        const good = (((w[0] * r[0] + w[1] * r[1] + w[2]) > 0 ? 1 : 0) === r[2]);
        g.append("text").attr("x", sx(r[0])).attr("y", sy(r[1]) + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 700)
          .attr("fill", good ? C.good : C.bad).text(good ? "✓" : "✗");
      });
      PV.label(g, LX + 8, LY + 16, k + " / 4 correct", { size: 12, bold: true, color: k === 4 ? C.good : C.orange });
      /* right: histogram */
      PV.frameBox(g, RX, RY, RS + 40, RS);
      if (hist) {
        const x = d3.scaleBand().domain([0, 1, 2, 3, 4]).range([RX + 40, RX + RS + 30]).padding(0.2);
        const y = d3.scaleLinear().domain([0, d3.max(hist)]).nice().range([RY + RS - 26, RY + 30]);
        PV.axisStyle(g.append("g").attr("transform", "translate(0," + (RY + RS - 26) + ")").call(d3.axisBottom(x)));
        PV.axisStyle(g.append("g").attr("transform", "translate(" + (RX + 40) + ",0)").call(d3.axisLeft(y).ticks(5, "~s")));
        g.selectAll("rect.hb").data(hist).join("rect").attr("class", "hb").attr("x", (d, i) => x(i)).attr("width", x.bandwidth())
          .attr("y", d => y(d)).attr("height", d => y(0) - y(d)).attr("fill", (d, i) => i === 4 ? C.good : C.accent);
        hist.forEach((d, i) => PV.label(g, x(i) + x.bandwidth() / 2, y(d) - 4, d3.format(",")(d), { anchor: "middle", size: 9.5, color: C.ink }));
        PV.label(g, RX + 44, RY + 18, "rows correct, over 20 000 random lines", { size: 10.5, color: C.ink });
        msg = k + "/4 correct for this line · random search: best = <b>" + d3.max(d3.range(5).filter(i => hist[i] > 0)) + "</b>/4, lines scoring 4: <b>" + hist[4] + "</b>";
      } else {
        PV.label(g, RX + (RS + 40) / 2, RY + RS / 2, "press “Search” to try 20 000 random lines", { anchor: "middle" });
        msg = k + "/4 correct for this line — no angle and offset reaches 4";
      }
    } else {
      const wOR = [1, 1, -0.5], wNAND = [-1, -1, 1.5];
      const band = [];
      /* the region where OR and NAND both fire: between the two lines */
      const hpA = PV.halfPlane(1, 1, -0.5, box);
      const clipB = pts => { const out = []; const f = q => -q[0] - q[1] + 1.5;
        for (let i = 0; i < pts.length; i++) { const P = pts[i], Q = pts[(i + 1) % pts.length], fp = f(P), fq = f(Q);
          if (fp >= 0) out.push(P); if ((fp >= 0) !== (fq >= 0)) { const t = fp / (fp - fq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); } }
        return out; };
      const reg = clipB(hpA);
      if (reg.length > 2) g.append("polygon").attr("points", reg.map(p => sx(p[0]) + "," + sy(p[1])).join(" ")).attr("fill", C.good).attr("fill-opacity", 0.14);
      drawLine(g, sx, sy, wOR, box, C.orange, 2);
      drawLine(g, sx, sy, wNAND, box, "#c084fc", 2);
      PV.label(g, sx(-0.25), sy(0.62) , "h₁ = OR", { color: C.orange, size: 10.5 });
      PV.label(g, sx(0.95), sy(0.72), "h₂ = NAND", { color: "#c084fc", size: 10.5 });
      corners(g, sx, sy, rows, labs);
      PV.label(g, LX + 8, LY + 16, "input space: y fires in the band", { size: 10.5, color: C.ink });
      /* hidden space */
      const hx = d3.scaleLinear().domain(box.slice(0, 2)).range([RX, RX + RS]);
      const hy = d3.scaleLinear().domain(box.slice(2)).range([RY + RS, RY]);
      PV.frameBox(g, RX, RY, RS, RS);
      const H = rows.map(r => [(r[0] + r[1] - 0.5 > 0) ? 1 : 0, (-r[0] - r[1] + 1.5 > 0) ? 1 : 0, r[2]]);
      const hp = PV.halfPlane(1, 1, -1.5, box);
      if (hp.length > 2) g.append("polygon").attr("points", hp.map(p => hx(p[0]) + "," + hy(p[1])).join(" ")).attr("fill", C.good).attr("fill-opacity", 0.12);
      drawLine(g, hx, hy, [1, 1, -1.5], box, C.ink, 2);
      const byCorner = {};
      H.forEach((h, i) => { const k = h[0] + "," + h[1]; (byCorner[k] = byCorner[k] || []).push(labs[i]); });
      Object.keys(byCorner).forEach(k => {
        const [a, b] = k.split(",").map(Number), tgt = (a && b) ? 1 : 0;
        g.append("circle").attr("cx", hx(a)).attr("cy", hy(b)).attr("r", 11).attr("fill", tgt ? C.pos : C.panel).attr("stroke", tgt ? C.pos : C.neg).attr("stroke-width", 2.2);
        PV.label(g, hx(a) + (a ? -16 : 16), hy(b) + (b ? 24 : -16), "← " + byCorner[k].join(" & "), { size: 9.5, anchor: a ? "end" : "start" });
      });
      PV.label(g, RX + 8, RY + 16, "hidden space (h₁, h₂): y = AND, one line", { size: 10.5, color: C.ink });
      PV.label(g, hx(0), hy(-0.22), "h₁", { size: 10 }); PV.label(g, hx(-0.2), hy(0), "h₂", { size: 10 });
      msg = "hidden layer on: (0,1) and (1,0) both map to (h₁,h₂) = (1,1); (0,0) → (0,1); (1,1) → (1,0). In hidden space the line h₁ + h₂ = 1.5 separates them — <b>4/4</b>.";
    }
    document.getElementById("xr-readout").innerHTML = msg;
  }
  document.getElementById("xr-search").addEventListener("click", () => {
    const r = PV.rng(2024);
    hist = [0, 0, 0, 0, 0];
    for (let i = 0; i < 20000; i++) hist[correct(lineW(r() * 360, r() * 2 - 1))]++;
    cH.checked = false;
    draw();
  });
  [sA, sO].forEach(s => s.addEventListener("input", draw));
  cH.addEventListener("change", draw);
  draw();
})();
