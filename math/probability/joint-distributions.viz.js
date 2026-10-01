/* joint-distributions.viz.js — the nine visualizations on math/probability/joint-distributions.html.
   Loaded after ../../data.js → ../../notes.js → ../statistics/stats-viz.js (SC palette, ST toolbox)
   → prob-viz.js (PV: Poisson/gamma/beta pmfs and draws, PV.bvnPdf/bvnDraw, PV.dirichletDraw,
   PV.convolve). Distribution functions come from ST / PV and are never re-implemented here.

   Everything page-local lives under the single namespace JD. Each figure is an IIFE that exits
   quietly if its <svg> is not on the page. The pure computations are exposed on JD so the build's
   node checks can call them without a DOM.

     1  #joint-svg   a joint pmf table with margins and a renormalised conditional slice
     2  #cov-svg     covariance as an average of signed rectangles (four quadrants)
     3  #avg-svg     averaging equicorrelated members: Var X̄ = ρ + (1 − ρ)/n
     4  #slice-svg   conditioning a joint density: slice, then renormalise
     5  #conv-svg    convolution: flip, slide, multiply, integrate
     6  #polar-svg   Box–Muller: a cell of the unit square and its annular-sector image
     7  #order-svg   the k-th order statistic: simulation vs exact density (Beta for uniforms)
     8  #bvn-svg     bivariate normal: ellipses, SD line, two regression lines, conditional slice
     9  #dir-svg     the Dirichlet on the simplex, with the multinomial update                      */

const JD = {
  // Default seeds, fixed by the caption audit (see the note at each figure).
  seeds: { cov: 7, avg: 3, polar: 3, order: 5, bvn: 4, dir: 12 },
  label(g, x, y, t, o) {
    const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", (o && o.size) || 11)
      .attr("fill", (o && o.color) || SC.muted).text(t);
    if (o && o.anchor) e.attr("text-anchor", o.anchor);
    if (o && o.weight) e.attr("font-weight", o.weight);
    return e;
  },
  f(x, d) { return ST.fmt(x, d === undefined ? 4 : d); },
  sub: k => String(k).replace(/\d/g, c => "₀₁₂₃₄₅₆₇₈₉"[+c])   // order-statistic subscripts: X₍₁₃₎
};

/* ─────────────────── 1 · joint table, margins, conditional slice ───────────────────
   Caption audit: preset "dice", slice by column, selected column M = 4 → P(M = 4) = 7/36,
   S | M = 4 = 2/7, 2/7, 2/7, 1/7 on 5..8, E[S | M = 4] = 44/7 = 6.2857, E S = 7;
   largest |p − pS·pM| = 66/1296 = 0.0509 at S = 3, M = 2; Cov(S, M) = 35/12, ρ = 0.860.        */
JD.jointPreset = function (name, eps) {
  if (name === "dice") {
    const P = d3.range(11).map(() => new Array(6).fill(0));
    for (let a = 1; a <= 6; a++) for (let b = 1; b <= 6; b++) P[a + b - 2][Math.max(a, b) - 1] += 1 / 36;
    return { P, rows: d3.range(2, 13), cols: d3.range(1, 7), rn: "S", cn: "M", den: 36, sel: 3 };
  }
  if (name === "eps") {
    const e = eps === undefined ? 0.1 : eps;
    return { P: [[0.25 - e, 0.25 + e], [0.25 + e, 0.25 - e]], rows: [0, 1], cols: [0, 1], rn: "X", cn: "Y", den: 0, sel: 0 };
  }
  if (name === "indep") return { P: [[1 / 9, 2 / 9], [2 / 9, 4 / 9]], rows: [0, 1], cols: [0, 1], rn: "X", cn: "Y", den: 9, sel: 0 };
  return { P: [[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], rows: [0, 1, 2], cols: [0, 1, 2], rn: "X", cn: "Y", den: 4, sel: 0 };
};
JD.jointStats = function (T) {
  const nr = T.rows.length, nc = T.cols.length;
  const pr = T.P.map(r => d3.sum(r)), pc = d3.range(nc).map(j => d3.sum(T.P, r => r[j]));
  let gap = 0, gi = 0, gj = 0, exy = 0;
  for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
    const g = Math.abs(T.P[i][j] - pr[i] * pc[j]);
    if (g > gap + 1e-15) { gap = g; gi = i; gj = j; }
    exy += T.rows[i] * T.cols[j] * T.P[i][j];
  }
  const er = d3.sum(T.rows, (v, i) => v * pr[i]), ec = d3.sum(T.cols, (v, j) => v * pc[j]);
  const vr = d3.sum(T.rows, (v, i) => v * v * pr[i]) - er * er, vc = d3.sum(T.cols, (v, j) => v * v * pc[j]) - ec * ec;
  const cov = exy - er * ec;
  return { pr, pc, gap, gi, gj, er, ec, vr, vc, cov, rho: cov / Math.sqrt(vr * vc) };
};
(function () {
  const svg = d3.select("#joint-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("joint-readout");
  let pre = "dice", axis = "col", eps = 0.1, T = JD.jointPreset("dice"), sel = 3;
  const frac = (p, den) => {
    if (!den) return p.toFixed(3);
    const k = Math.round(p * den);
    return Math.abs(k / den - p) < 1e-9 ? (k === 0 ? "0" : `${k}/${den}`) : p.toFixed(3);
  };
  function draw() {
    svg.selectAll("*").remove();
    const S = JD.jointStats(T), nr = T.rows.length, nc = T.cols.length;
    const cw = nc > 3 ? 40 : 66, ch = nr > 3 ? 22 : 52, x0 = 56, y0 = 34;
    const g = svg.append("g");
    const pmax = d3.max(T.P.flat());
    JD.label(g, x0 + cw * nc / 2, 14, `${T.cn} →`, { anchor: "middle", color: SC.ink });
    JD.label(g, 12, y0 + ch * nr / 2, `${T.rn} ↓`, { color: SC.ink });
    T.cols.forEach((c, j) => JD.label(g, x0 + cw * j + cw / 2, y0 - 6, String(c), { anchor: "middle", color: SC.ink }));
    T.rows.forEach((r, i) => JD.label(g, x0 - 8, y0 + ch * i + ch / 2 + 4, String(r), { anchor: "end", color: SC.ink }));
    for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
      const p = T.P[i][j];
      const inSel = axis === "col" ? j === sel : i === sel;
      const cg = g.append("g").style("cursor", "pointer");
      cg.append("rect").attr("x", x0 + cw * j).attr("y", y0 + ch * i).attr("width", cw - 1).attr("height", ch - 1)
        .attr("fill", SC.accent).attr("fill-opacity", pmax > 0 ? 0.08 + 0.75 * p / pmax : 0.08)
        .attr("stroke", inSel ? SC.a2 : SC.line).attr("stroke-width", inSel ? 2 : 1);
      JD.label(cg, x0 + cw * j + cw / 2, y0 + ch * i + ch / 2 + 4, frac(p, T.den), { anchor: "middle", size: nc > 3 ? 9.5 : 12, color: SC.ink });
      cg.on("click", () => { sel = axis === "col" ? j : i; draw(); });
    }
    // margins
    const mx = x0 + cw * nc + 6, my = y0 + ch * nr + 6;
    JD.label(g, mx + 22, y0 - 6, `p_${T.rn}`, { anchor: "middle", color: SC.a2 });
    S.pr.forEach((p, i) => JD.label(g, mx + 22, y0 + ch * i + ch / 2 + 4, frac(p, T.den), { anchor: "middle", size: nc > 3 ? 9.5 : 12, color: SC.a2 }));
    JD.label(g, x0 - 8, my + 14, `p_${T.cn}`, { anchor: "end", color: SC.a2 });
    S.pc.forEach((p, j) => JD.label(g, x0 + cw * j + cw / 2, my + 14, frac(p, T.den), { anchor: "middle", size: nc > 3 ? 9.5 : 12, color: SC.a2 }));
    g.append("line").attr("x1", mx).attr("x2", mx).attr("y1", y0).attr("y2", y0 + ch * nr).attr("stroke", SC.line);
    g.append("line").attr("x1", x0).attr("x2", x0 + cw * nc).attr("y1", my).attr("y2", my).attr("stroke", SC.line);
    // conditional bars
    const vals = axis === "col" ? T.rows : T.cols;
    const slice = axis === "col" ? T.P.map(r => r[sel]) : T.P[sel].slice();
    const ps = d3.sum(slice), cond = slice.map(v => (ps > 0 ? v / ps : 0));
    const marg = axis === "col" ? S.pr : S.pc;
    const bx0 = 420, bw = 240, by0 = 40, bh = 240;
    const x = d3.scaleBand().domain(vals).range([bx0, bx0 + bw]).padding(0.2);
    const ymax = Math.max(d3.max(cond), d3.max(marg)) * 1.12 || 1;
    const y = d3.scaleLinear().domain([0, ymax]).range([by0 + bh, by0]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${by0 + bh})`).call(d3.axisBottom(x));
    g.append("g").attr("class", "axis").attr("transform", `translate(${bx0},0)`).call(d3.axisLeft(y).ticks(5));
    vals.forEach((v, k) => {
      g.append("rect").attr("x", x(v)).attr("y", y(cond[k])).attr("width", x.bandwidth()).attr("height", y(0) - y(cond[k]))
        .attr("fill", SC.a2).attr("fill-opacity", 0.8);
      g.append("rect").attr("x", x(v)).attr("y", y(marg[k])).attr("width", x.bandwidth()).attr("height", y(0) - y(marg[k]))
        .attr("fill", "none").attr("stroke", SC.ink).attr("stroke-dasharray", "3 2");
    });
    const given = axis === "col" ? `${T.cn} = ${T.cols[sel]}` : `${T.rn} = ${T.rows[sel]}`;
    const of = axis === "col" ? T.rn : T.cn;
    JD.label(g, bx0, by0 - 18, `P(${of} | ${given}) — bars;  marginal of ${of} — dashed`, { color: SC.ink, size: 11 });
    JD.label(g, bx0 + bw / 2, by0 + bh + 30, of, { anchor: "middle" });
    // readout
    const ec = d3.sum(vals, (v, k) => v * cond[k]), em = d3.sum(vals, (v, k) => v * marg[k]);
    const indep = S.gap < 1e-12;
    out.innerHTML = `P(${given}) = <b>${T.den ? frac(ps, T.den) + " = " : ""}${ps.toFixed(4)}</b> · conditional of ${of}: ` +
      vals.map((v, k) => `${v}: ${cond[k].toFixed(4)}`).filter((s, k) => cond[k] > 0).join(", ") +
      `<br>E[${of} | ${given}] = <b>${ec.toFixed(4)}</b> against E ${of} = ${em.toFixed(4)}` +
      `<br>largest gap |p(x, y) − p_${T.rn}·p_${T.cn}| = <b>${S.gap.toFixed(4)}</b>` +
      (indep ? " — zero in every cell: <b>independent</b>" : ` at ${T.rn} = ${T.rows[S.gi]}, ${T.cn} = ${T.cols[S.gj]} — <b>dependent</b>`) +
      `<br>Cov(${T.rn}, ${T.cn}) = ${S.cov.toFixed(4)} · ρ = ${isFinite(S.rho) ? S.rho.toFixed(4) : "—"}` +
      (Math.abs(S.cov) < 1e-12 && !indep ? " — uncorrelated, yet dependent" : "");
  }
  function load() {
    T = JD.jointPreset(pre, eps); sel = axis === "col" ? T.sel : 0;
    $("jt-epsw").style.display = pre === "eps" ? "" : "none";
    draw();
  }
  $("jt-pre").addEventListener("change", e => { pre = e.target.value; load(); });
  $("jt-axis").addEventListener("change", e => { axis = e.target.value; sel = axis === "col" ? T.sel : 0; draw(); });
  $("jt-eps").addEventListener("input", e => { eps = +e.target.value; $("jt-epsv").textContent = eps.toFixed(2); T = JD.jointPreset(pre, eps); draw(); });
  $("jt-reset").addEventListener("click", () => {
    pre = "dice"; axis = "col"; eps = 0.1; $("jt-pre").value = "dice"; $("jt-axis").value = "col";
    $("jt-eps").value = 0.1; $("jt-epsv").textContent = "0.10"; load();
  });
  load();
})();

/* ─────────────────── 2 · covariance as signed rectangles ───────────────────
   Caption audit: mode "norm", ρ = 0.6, n = 300, seed JD.seeds.cov → sample cov and ρ in the caption
   (values printed by the build check, recorded there).                                           */
JD.covSample = function (mode, rho, n, seed) {
  const r = ST.rng(seed), xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    let x, y;
    if (mode === "norm") [x, y] = PV.bvnDraw(0, 0, 1, 1, rho, r);
    else if (mode === "parab") { x = ST.randn(r); y = x * x; }
    else if (mode === "circle") { const t = 2 * Math.PI * r(); x = Math.cos(t); y = Math.sin(t); }
    else if (mode === "sqpos") { x = r(); y = x * x; }
    else { if (i === n - 1) { x = 6; y = 6; } else { x = ST.randn(r); y = ST.randn(r); } }
    xs.push(x); ys.push(y);
  }
  const mx = d3.mean(xs), my = d3.mean(ys);
  let pos = 0, neg = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const p = (xs[i] - mx) * (ys[i] - my);
    if (p > 0) pos += p; else neg += p;
    sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2;
  }
  return { xs, ys, mx, my, pos, neg, cov: (pos + neg) / (n - 1), rho: (pos + neg) / Math.sqrt(sxx * syy) };
};
JD.covPop = { norm: r => [r, r], parab: () => [0, 0], circle: () => [0, 0], sqpos: () => [1 / 12, 0.968246], outlier: () => [NaN, NaN] };
(function () {
  const svg = d3.select("#cov-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("cov-readout");
  let mode = "norm", rho = 0.6, n = 300, seed = JD.seeds.cov;
  function draw() {
    svg.selectAll("*").remove();
    const D = JD.covSample(mode, rho, n, seed);
    const W = 470, H = 330, m = { l: 40, t: 14, b: 30 };
    const ext = a => { const lo = d3.min(a), hi = d3.max(a), pad = 0.08 * (hi - lo || 1); return [lo - pad, hi + pad]; };
    const x = d3.scaleLinear().domain(ext(D.xs)).range([m.l, W]), y = d3.scaleLinear().domain(ext(D.ys)).range([H - m.b, m.t]);
    const g = svg.append("g");
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6));
    // rectangles for a few points: the ten largest |products|
    const idx = d3.range(n).sort((a, b) => Math.abs((D.xs[b] - D.mx) * (D.ys[b] - D.my)) - Math.abs((D.xs[a] - D.mx) * (D.ys[a] - D.my))).slice(0, 10);
    idx.forEach(i => {
      const p = (D.xs[i] - D.mx) * (D.ys[i] - D.my);
      g.append("rect").attr("x", Math.min(x(D.xs[i]), x(D.mx))).attr("y", Math.min(y(D.ys[i]), y(D.my)))
        .attr("width", Math.abs(x(D.xs[i]) - x(D.mx))).attr("height", Math.abs(y(D.ys[i]) - y(D.my)))
        .attr("fill", p > 0 ? SC.accent : SC.bad).attr("fill-opacity", 0.12).attr("stroke", p > 0 ? SC.accent : SC.bad).attr("stroke-opacity", 0.5);
    });
    g.append("line").attr("x1", x(D.mx)).attr("x2", x(D.mx)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
    g.append("line").attr("x1", m.l).attr("x2", W).attr("y1", y(D.my)).attr("y2", y(D.my)).attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
    D.xs.forEach((v, i) => {
      const p = (v - D.mx) * (D.ys[i] - D.my);
      g.append("circle").attr("cx", x(v)).attr("cy", y(D.ys[i])).attr("r", 2.4).attr("fill", p > 0 ? SC.accent : SC.bad).attr("fill-opacity", 0.75);
    });
    JD.label(g, W - 4, m.t + 10, "+", { anchor: "end", color: SC.accent, size: 16 });
    JD.label(g, m.l + 6, m.t + 10, "−", { color: SC.bad, size: 16 });
    JD.label(g, m.l + 6, H - m.b - 6, "+", { color: SC.accent, size: 16 });
    JD.label(g, W - 4, H - m.b - 6, "−", { anchor: "end", color: SC.bad, size: 16 });
    // totals bar
    const bx = 540, tot = Math.max(D.pos, -D.neg) / (n - 1);
    const yb = d3.scaleLinear().domain([-tot * 1.1, tot * 1.1]).range([300, 40]);
    g.append("g").attr("class", "axis").attr("transform", `translate(${bx - 10},0)`).call(d3.axisLeft(yb).ticks(6));
    g.append("rect").attr("x", bx).attr("width", 34).attr("y", yb(D.pos / (n - 1))).attr("height", yb(0) - yb(D.pos / (n - 1))).attr("fill", SC.accent).attr("fill-opacity", 0.8);
    g.append("rect").attr("x", bx + 40).attr("width", 34).attr("y", yb(0)).attr("height", yb(D.neg / (n - 1)) - yb(0)).attr("fill", SC.bad).attr("fill-opacity", 0.8);
    g.append("line").attr("x1", bx - 4).attr("x2", bx + 84).attr("y1", yb(D.cov)).attr("y2", yb(D.cov)).attr("stroke", SC.a2).attr("stroke-width", 3);
    JD.label(g, bx + 37, 26, "Σ products /(n − 1)", { anchor: "middle", color: SC.ink });
    JD.label(g, bx + 88, yb(D.cov) + 4, `net ${D.cov.toFixed(3)}`, { color: SC.a2, size: 10.5 });
    JD.label(g, bx + 17, 318, "+", { anchor: "middle", color: SC.accent });
    JD.label(g, bx + 57, 318, "−", { anchor: "middle", color: SC.bad });
    const pop = JD.covPop[mode](rho);
    out.innerHTML = `positive products ${(D.pos / (n - 1)).toFixed(4)}, negative ${(D.neg / (n - 1)).toFixed(4)} → sample Cov = <b>${D.cov.toFixed(4)}</b> · sample ρ = <b>${D.rho.toFixed(4)}</b>` +
      `<br>population: ` + (mode === "outlier" ? `the ${n - 1} regular pairs are independent (Cov 0); the single point at (6, 6) supplies most of the covariance`
        : `Cov = ${pop[0].toFixed(4)}, ρ = ${pop[1].toFixed(4)}`) +
      (mode === "parab" || mode === "circle" ? " — uncorrelated, yet Y is determined by X (or by the angle): <b>dependent</b>" : "") +
      `<br>n = ${n} · share of total |area| that is positive: ${(D.pos / (D.pos - D.neg)).toFixed(3)}`;
  }
  const sync = () => { $("cv-rw").style.display = mode === "norm" ? "" : "none"; };
  $("cv-mode").addEventListener("change", e => { mode = e.target.value; sync(); draw(); });
  $("cv-r").addEventListener("input", e => { rho = +e.target.value; $("cv-rv").textContent = rho.toFixed(2); draw(); });
  $("cv-n").addEventListener("input", e => { n = +e.target.value; $("cv-nv").textContent = n; draw(); });
  $("cv-sim").addEventListener("click", () => { seed += 1; draw(); });
  $("cv-reset").addEventListener("click", () => {
    mode = "norm"; rho = 0.6; n = 300; seed = JD.seeds.cov;
    $("cv-mode").value = "norm"; $("cv-r").value = 0.6; $("cv-rv").textContent = "0.60"; $("cv-n").value = 300; $("cv-nv").textContent = "300";
    sync(); draw();
  });
  sync(); draw();
})();

/* ─────────────────── 3 · averaging equicorrelated members ───────────────────
   Caption audit: ρ = 0.2, n = 10, R = 4000 replications, seed JD.seeds.avg → exact 0.28, n_eff 3.571;
   the simulated variance at n = 10 is printed by the build check and recorded in the caption.      */
JD.avgSim = function (rho, R, seed, nmax) {
  const r = ST.rng(seed), N = nmax || 50, sum = new Float64Array(N + 1), sq = new Float64Array(N + 1), keep = [];
  const a = Math.sqrt(rho), b = Math.sqrt(1 - rho);
  for (let k = 0; k < R; k++) {
    const z0 = ST.randn(r); let s = 0; const mem = [];
    for (let i = 1; i <= N; i++) {
      const zi = ST.randn(r); s += zi;
      if (k < 40) mem.push(a * z0 + b * zi);
      const avg = a * z0 + b * s / i;
      sum[i] += avg; sq[i] += avg * avg;
    }
    if (k < 40) keep.push(mem);
  }
  const v = d3.range(N + 1).map(i => (i === 0 ? NaN : sq[i] / R - (sum[i] / R) ** 2));
  return { v, keep };
};
(function () {
  const svg = d3.select("#avg-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("avg-readout");
  let rho = 0.2, n = 10, seed = JD.seeds.avg, sim = null;
  const resim = () => { sim = JD.avgSim(rho, 4000, seed, 50); };
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), exact = k => rho + (1 - rho) / k;
    const x = d3.scaleLog().domain([1, 50]).range([50, 330]), y = d3.scaleLinear().domain([0, 1.05]).range([290, 20]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(x).tickValues([1, 2, 5, 10, 20, 50]).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(y).ticks(5));
    JD.label(g, 330, 318, "n members (log scale)", { anchor: "end" });
    JD.label(g, 54, 14, "Var X̄  (σ² = 1)", {});
    const ks = d3.range(1, 50.01, 0.25);
    g.append("path").attr("d", d3.line().x(k => x(k)).y(k => y(1 / k))(ks)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
    g.append("path").attr("d", d3.line().x(k => x(k)).y(k => y(exact(k)))(ks)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
    g.append("line").attr("x1", 50).attr("x2", 330).attr("y1", y(rho)).attr("y2", y(rho)).attr("stroke", SC.bad).attr("stroke-dasharray", "2 3");
    JD.label(g, 56, y(rho) + 14, `floor ρ = ${rho.toFixed(2)}`, { color: SC.bad, size: 10.5 });
    JD.label(g, x(2.2), y(1 / 2.2) + 14, "1/n (independent)", { size: 10 });
    [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50].forEach(k => g.append("circle").attr("cx", x(k)).attr("cy", y(sim.v[k])).attr("r", 3.2).attr("fill", SC.accent));
    g.append("circle").attr("cx", x(n)).attr("cy", y(sim.v[n])).attr("r", 6).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.5);
    // right: forty replications
    const rx = d3.scaleLinear().domain([0, 41]).range([380, 670]), ry = d3.scaleLinear().domain([-3.5, 3.5]).range([290, 20]);
    g.append("g").attr("class", "axis").attr("transform", "translate(380,0)").call(d3.axisLeft(ry).ticks(7));
    g.append("line").attr("x1", 380).attr("x2", 670).attr("y1", ry(0)).attr("y2", ry(0)).attr("stroke", SC.line);
    sim.keep.forEach((mem, k) => {
      const vals = mem.slice(0, n);
      vals.forEach(v => g.append("circle").attr("cx", rx(k + 1)).attr("cy", ry(ST.clamp(v, -3.5, 3.5))).attr("r", 1.8).attr("fill", SC.muted).attr("fill-opacity", 0.6));
      g.append("circle").attr("cx", rx(k + 1)).attr("cy", ry(d3.mean(vals))).attr("r", 3.6).attr("fill", SC.a2);
    });
    JD.label(g, 384, 14, `40 replications: ${n} members (grey), average (orange)`, { size: 10.5 });
    JD.label(g, 670, 318, "replication", { anchor: "end" });
    const ne = n / (1 + (n - 1) * rho);
    out.innerHTML = `ρ = ${rho.toFixed(2)}, n = ${n}: Var X̄ = ρ + (1 − ρ)/n = <b>${exact(n).toFixed(4)}</b> · simulated (4,000 replications) <b>${sim.v[n].toFixed(4)}</b>` +
      ` · with independent members it would be 1/n = ${(1 / n).toFixed(4)}` +
      `<br>effective number of independent members n/(1 + (n − 1)ρ) = <b>${ne.toFixed(2)}</b>` +
      (rho > 0 ? ` · never more than 1/ρ = ${(1 / rho).toFixed(2)}, however many are averaged` : " · no floor: the variance falls like 1/n") +
      `<br>simulated at n = 50: ${sim.v[50].toFixed(4)} (exact ${exact(50).toFixed(4)}); the shared part ρ cannot be averaged away`;
  }
  $("av-r").addEventListener("input", e => { rho = +e.target.value; $("av-rv").textContent = rho.toFixed(2); resim(); draw(); });
  $("av-n").addEventListener("input", e => { n = +e.target.value; $("av-nv").textContent = n; draw(); });
  $("av-sim").addEventListener("click", () => { seed += 1; resim(); draw(); });
  $("av-reset").addEventListener("click", () => {
    rho = 0.2; n = 10; seed = JD.seeds.avg; $("av-r").value = 0.2; $("av-rv").textContent = "0.20"; $("av-n").value = 10; $("av-nv").textContent = "10";
    resim(); draw();
  });
  resim(); draw();
})();

/* ─────────────────── 4 · conditioning a joint density ───────────────────
   Caption audit: preset "curv", axis x, x₀ = 0.5, t = 0.75 → f_X(½) = 0.6152, E[Y | X = ½] = 0.7,
   P(Y ≤ ¾ | X = ½) = 8/15 = 0.5333 (numerical and closed form).                                   */
JD.slicePresets = {
  curv: { f: (x, y) => (x * x <= y && y <= 1 ? 5.25 * x * x * y : 0), xd: [-1, 1], yd: [0, 1], x0: 0.5, y0: 0.5,
    cx: x => ({ m: 2.625 * x * x * (1 - x ** 4), e: 2 * (1 - x ** 6) / (3 * (1 - x ** 4)), F: t => ST.clamp((t * t - x ** 4) / (1 - x ** 4), 0, 1) }),
    cy: y => ({ m: 3.5 * Math.pow(y, 2.5), e: 0, F: t => { const s = Math.sqrt(y); const c = ST.clamp(t, -s, s); return (c ** 3 + s ** 3) / (2 * s ** 3); } }) },
  xy: { f: (x, y) => (x >= 0 && x <= 1 && y >= 0 && y <= 1 ? x + y : 0), xd: [0, 1], yd: [0, 1], x0: 0.5, y0: 1 / 3,
    cx: x => ({ m: x + 0.5, e: (x / 2 + 1 / 3) / (x + 0.5), F: t => { const c = ST.clamp(t, 0, 1); return (x * c + c * c / 2) / (x + 0.5); } }),
    cy: y => ({ m: y + 0.5, e: (y / 2 + 1 / 3) / (y + 0.5), F: t => { const c = ST.clamp(t, 0, 1); return (y * c + c * c / 2) / (y + 0.5); } }) },
  two: { f: (x, y) => (x >= 0 && x < y && y <= 1 ? 1 / (1 - x) : 0), xd: [0, 1], yd: [0, 1], x0: 0.4, y0: 0.6,
    cx: x => ({ m: 1, e: (1 + x) / 2, F: t => ST.clamp((t - x) / (1 - x), 0, 1) }),
    cy: y => ({ m: -Math.log(1 - y), e: 1 - y / (-Math.log(1 - y)), F: t => { const c = ST.clamp(t, 0, y); return -Math.log(1 - c) / (-Math.log(1 - y)); } }) },
  bvn: { f: (x, y) => PV.bvnPdf(x, y, 0, 0, 1, 1, 0.7), xd: [-3, 3], yd: [-3, 3], id: [-9, 9], x0: 1, y0: 1,
    cx: x => ({ m: ST.normPdf(x), e: 0.7 * x, F: t => ST.normCdf((t - 0.7 * x) / Math.sqrt(0.51)) }),
    cy: y => ({ m: ST.normPdf(y), e: 0.7 * y, F: t => ST.normCdf((t - 0.7 * y) / Math.sqrt(0.51)) }) }
};
JD.sliceNum = function (P, axis, v, t) {          // midpoint rule along the slice, 4,000 cells
  const d = P.id || (axis === "x" ? P.yd : P.xd), N = 4000, h = (d[1] - d[0]) / N;
  let m = 0, s1 = 0, s2 = 0, below = 0;
  for (let i = 0; i < N; i++) {
    const u = d[0] + (i + 0.5) * h, f = axis === "x" ? P.f(v, u) : P.f(u, v);
    m += f * h; s1 += u * f * h; s2 += u * u * f * h; if (u <= t) below += f * h;
  }
  const e = s1 / m;
  return { m, e, sd: Math.sqrt(Math.max(0, s2 / m - e * e)), F: below / m };
};
(function () {
  const svg = d3.select("#slice-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("slice-readout");
  let pre = "curv", axis = "x", tf = 0.75, pos = 0.5;
  const P = () => JD.slicePresets[pre];
  const L = { x0: 46, y0: 14, w: 270, h: 290 };
  let heat, overlay, rightG, xs, ys;
  function heatmap() {
    svg.selectAll("*").remove();
    const p = P();
    xs = d3.scaleLinear().domain(p.xd).range([L.x0, L.x0 + L.w]);
    ys = d3.scaleLinear().domain(p.yd).range([L.y0 + L.h, L.y0]);
    heat = svg.append("g");
    const N = 70, dx = (p.xd[1] - p.xd[0]) / N, dy = (p.yd[1] - p.yd[0]) / N, vals = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) vals.push(p.f(p.xd[0] + (i + 0.5) * dx, p.yd[0] + (j + 0.5) * dy));
    const cap = d3.quantile(vals.filter(v => v > 0).sort(d3.ascending), 0.97) || 1;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = vals[i * N + j]; if (v <= 0) continue;
      heat.append("rect").attr("x", xs(p.xd[0] + i * dx)).attr("y", ys(p.yd[0] + (j + 1) * dy)).attr("width", L.w / N + 0.4).attr("height", L.h / N + 0.4)
        .attr("fill", SC.accent).attr("fill-opacity", 0.06 + 0.85 * Math.min(1, v / cap));
    }
    heat.append("g").attr("class", "axis").attr("transform", `translate(0,${L.y0 + L.h})`).call(d3.axisBottom(xs).ticks(5));
    heat.append("g").attr("class", "axis").attr("transform", `translate(${L.x0},0)`).call(d3.axisLeft(ys).ticks(5));
    JD.label(heat, L.x0 + L.w, L.y0 + L.h + 30, "x", { anchor: "end" });
    JD.label(heat, L.x0 + 4, L.y0 + 10, "y", {});
    overlay = svg.append("g"); rightG = svg.append("g");
  }
  function draw() {
    const p = P();
    overlay.selectAll("*").remove(); rightG.selectAll("*").remove();
    const od = axis === "x" ? p.yd : p.xd, t = od[0] + tf * (od[1] - od[0]);
    $("sl-tv").textContent = t.toFixed(2);
    // slicing line (draggable) and threshold line
    if (axis === "x") {
      overlay.append("line").attr("x1", L.x0).attr("x2", L.x0 + L.w).attr("y1", ys(t)).attr("y2", ys(t)).attr("stroke", SC.good).attr("stroke-dasharray", "4 3");
      const lg = overlay.append("g").style("cursor", "ew-resize");
      lg.append("line").attr("x1", xs(pos)).attr("x2", xs(pos)).attr("y1", L.y0).attr("y2", L.y0 + L.h).attr("stroke", SC.a2).attr("stroke-width", 2.5);
      lg.append("rect").attr("x", xs(pos) - 8).attr("y", L.y0).attr("width", 16).attr("height", L.h).attr("fill", "transparent");
      lg.call(d3.drag().on("drag", ev => { pos = ST.clamp(xs.invert(ev.x), p.xd[0] + 0.005 * (p.xd[1] - p.xd[0]), p.xd[1] - 0.01 * (p.xd[1] - p.xd[0])); draw(); }));
      JD.label(overlay, xs(pos) + 4, L.y0 + 12, `x₀ = ${pos.toFixed(2)}`, { color: SC.a2, size: 10.5 });
    } else {
      overlay.append("line").attr("y1", L.y0).attr("y2", L.y0 + L.h).attr("x1", xs(t)).attr("x2", xs(t)).attr("stroke", SC.good).attr("stroke-dasharray", "4 3");
      const lg = overlay.append("g").style("cursor", "ns-resize");
      lg.append("line").attr("x1", L.x0).attr("x2", L.x0 + L.w).attr("y1", ys(pos)).attr("y2", ys(pos)).attr("stroke", SC.a2).attr("stroke-width", 2.5);
      lg.append("rect").attr("x", L.x0).attr("y", ys(pos) - 8).attr("width", L.w).attr("height", 16).attr("fill", "transparent");
      lg.call(d3.drag().on("drag", ev => { pos = ST.clamp(ys.invert(ev.y), p.yd[0] + 0.01 * (p.yd[1] - p.yd[0]), p.yd[1] - 0.01 * (p.yd[1] - p.yd[0])); draw(); }));
      JD.label(overlay, L.x0 + 4, ys(pos) - 5, `y₀ = ${pos.toFixed(2)}`, { color: SC.a2, size: 10.5 });
    }
    // right panel: raw slice and conditional
    const N = 400, us = d3.range(N + 1).map(i => od[0] + i * (od[1] - od[0]) / N);
    const raw = us.map(u => (axis === "x" ? p.f(pos, u) : p.f(u, pos)));
    const S = JD.sliceNum(p, axis, pos, t), cond = raw.map(v => v / S.m);
    const rx = d3.scaleLinear().domain(od).range([380, 665]);
    const rmax = Math.min(d3.max(cond.concat(raw)) * 1.08, 12);
    const ry = d3.scaleLinear().domain([0, rmax]).range([L.y0 + L.h, L.y0 + 10]);
    rightG.append("g").attr("class", "axis").attr("transform", `translate(0,${L.y0 + L.h})`).call(d3.axisBottom(rx).ticks(6));
    rightG.append("g").attr("class", "axis").attr("transform", "translate(380,0)").call(d3.axisLeft(ry).ticks(5));
    const ln = d3.line().x((d, i) => rx(us[i])).y(d => ry(Math.min(d, rmax)));
    const area = d3.area().x((d, i) => rx(us[i])).y0(ry(0)).y1(d => ry(Math.min(d, rmax)));
    rightG.append("path").attr("d", area.defined((d, i) => us[i] <= t)(cond)).attr("fill", SC.good).attr("fill-opacity", 0.25);
    rightG.append("path").attr("d", ln(raw)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
    rightG.append("path").attr("d", ln(cond)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
    rightG.append("line").attr("x1", rx(S.e)).attr("x2", rx(S.e)).attr("y1", L.y0 + 10).attr("y2", L.y0 + L.h).attr("stroke", SC.ink).attr("stroke-dasharray", "2 3");
    const v = axis === "x" ? "Y" : "X", gv = axis === "x" ? `X = ${pos.toFixed(2)}` : `Y = ${pos.toFixed(2)}`;
    JD.label(rightG, 384, L.y0 + 4, `dashed: raw slice at ${gv} (area ${S.m.toFixed(3)}) · solid: conditional`, { size: 10 });
    JD.label(rightG, 665, L.y0 + L.h + 30, v.toLowerCase(), { anchor: "end" });
    const C = axis === "x" ? p.cx(pos) : p.cy(pos);
    out.innerHTML = `slice at ${gv}: marginal density f_${axis.toUpperCase()}(${pos.toFixed(2)}) = area of the raw slice = <b>${S.m.toFixed(4)}</b> (closed form ${C.m.toFixed(4)})` +
      `<br>conditional: E[${v} | ${gv}] = <b>${S.e.toFixed(4)}</b> (closed form ${C.e.toFixed(4)}) · sd ${S.sd.toFixed(4)}` +
      `<br>P(${v} ≤ ${t.toFixed(2)} | ${gv}) = <b>${S.F.toFixed(4)}</b> (closed form ${C.F(t).toFixed(4)}) · P(${v} &gt; ${t.toFixed(2)} | ${gv}) = ${(1 - S.F).toFixed(4)}`;
  }
  function reset(keepT) {
    pos = axis === "x" ? P().x0 : P().y0;
    if (!keepT) { tf = 0.75; $("sl-t").value = tf; }
    heatmap(); draw();
  }
  $("sl-pre").addEventListener("change", e => { pre = e.target.value; reset(); });
  $("sl-axis").addEventListener("change", e => { axis = e.target.value; reset(true); });
  $("sl-t").addEventListener("input", e => { tf = +e.target.value; draw(); });
  $("sl-reset").addEventListener("click", () => { pre = "curv"; axis = "x"; $("sl-pre").value = "curv"; $("sl-axis").value = "x"; reset(); });
  reset();
})();

/* ─────────────────── 5 · convolution: flip, slide, multiply, integrate ───────────────────
   Caption audit: pair "unif", n = 2, z = 0.6 → f_Z(0.6) = 0.6 (numerical within 0.002), Var 1/6.  */
JD.convFam = {
  unif: { lo: 0, hi: 1, f: x => (x >= 0 && x <= 1 ? 1 : 0), mean: 0.5, v: 1 / 12, n: true,
    sum: (z, n) => { if (z < 0 || z > n) return 0; let s = 0; for (let k = 0; k <= Math.floor(z); k++) s += (k % 2 ? -1 : 1) * Math.exp(ST.lnChoose(n, k)) * Math.pow(z - k, n - 1); return s / Math.exp(ST.lnGamma(n)); },
    name: n => `Irwin–Hall(${n})` },
  exp: { lo: 0, hi: 12, f: x => PV.expPdf(x, 1), mean: 1, v: 1, n: true, sum: (z, n) => PV.gammaPdf(z, n, 1), name: n => `Gamma(${n}, 1)` },
  gam: { lo: 0, hi: 18, f: x => PV.gammaPdf(x, 2, 1), g: x => PV.gammaPdf(x, 3, 1), mean: 2, v: 2, mean2: 3, v2: 3, n: false, sum: z => PV.gammaPdf(z, 5, 1), name: () => "Gamma(5, 1)" },
  norm: { lo: -5, hi: 5, f: x => ST.normPdf(x), mean: 0, v: 1, n: true, sum: (z, n) => ST.normPdf(z / Math.sqrt(n)) / Math.sqrt(n), name: n => `N(0, ${n})` },
  dice: { disc: true, pmf: [0, 1, 1, 1, 1, 1, 1].map(v => v / 6), mean: 3.5, v: 35 / 12, n: true, name: n => `sum of ${n} dice` },
  pois: { disc: true, pmf: d3.range(18).map(k => PV.poisPmf(k, 2)), pmf2: d3.range(18).map(k => PV.poisPmf(k, 3)), mean: 2, v: 2, mean2: 3, v2: 3, n: false, name: () => "Poisson(5)" }
};
JD.convBuild = function (key, n) {                // the density of the first n − 1 terms, one more term, and the sum
  const F = JD.convFam[key];
  if (F.disc) {
    let acc = F.pmf.slice();
    const one = F.pmf2 || F.pmf, m = F.pmf2 ? 2 : n;
    for (let i = 2; i < m; i++) acc = PV.convolve(acc, F.pmf);
    const sum = PV.convolve(acc, one);
    return { disc: true, prev: acc, one, sum };
  }
  const nn = F.n ? n : 2;
  // density of the first nn − 1 terms, in closed form (Irwin–Hall, gamma, normal), so that the only
  // numerical step shown is the one being illustrated: the area under the product at each z
  const prevF = F.g ? F.g : (y => (nn - 1 === 1 ? F.f(y) : F.sum(y, nn - 1)));
  const sup = F.lo < 0 ? [-12 * Math.sqrt(nn), 12 * Math.sqrt(nn)] : [0, key === "unif" ? nn : 80 + 4 * nn];
  const one1 = key === "unif" ? [0, 1] : (F.lo < 0 ? [-12, 12] : [0, 80]);
  const area = z => {                              // ∫ prev(y)·f(z − y) dy over the overlap of the supports, Simpson
    const ya = Math.max(sup[0], z - one1[1]), yb = Math.min(key === "unif" ? nn - 1 : sup[1], z - one1[0]);
    return yb > ya ? PV.simpson(y => prevF(y) * F.f(z - y), ya, yb, 2000) : 0;
  };
  const N = 1600, loR = F.lo < 0 ? F.lo * Math.sqrt(nn) : 0, hiR = F.lo < 0 ? -loR : (key === "unif" ? nn : F.hi * nn + 4), h = (hiR - loR) / N;
  const grid = d3.range(N + 1).map(i => loR + i * h);
  return { disc: false, grid, h, prev: grid.map(prevF), one: grid.map(y => F.f(y)), nn, area };
};
(function () {
  const svg = d3.select("#conv-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("conv-readout");
  let key = "unif", n = 2, zf = 0.3, B = null;
  const rebuild = () => { B = JD.convBuild(key, n); };
  function zRange() {
    const F = JD.convFam[key];
    if (B.disc) return [0, B.sum.length - 1];
    if (key === "unif") return [0, n];
    if (key === "norm") return [-4 * Math.sqrt(n), 4 * Math.sqrt(n)];
    const nn = F.n ? n : 2, m = F.n ? nn * F.mean : F.mean + F.mean2, v = F.n ? nn * F.v : F.v + F.v2;
    return [0, m + 5 * Math.sqrt(v)];
  }
  function draw() {
    svg.selectAll("*").remove();
    const F = JD.convFam[key], g = svg.append("g"), zr = zRange();
    let z = zr[0] + zf * (zr[1] - zr[0]);
    if (B.disc) z = Math.round(z);
    $("cn-zv").textContent = B.disc ? String(z) : z.toFixed(2);
    const nn = B.disc ? (F.pmf2 ? 2 : n) : B.nn;
    const mean = F.n ? nn * F.mean : F.mean + F.mean2, vr = F.n ? nn * F.v : F.v + F.v2;
    const xd = [zr[0] - 0.05 * (zr[1] - zr[0]), zr[1] + 0.05 * (zr[1] - zr[0])];
    const x = d3.scaleLinear().domain(xd).range([50, 665]);
    let fz, exact, topMax, prodArea;
    if (B.disc) {
      const flipped = k => (z - k >= 0 && z - k < B.one.length ? B.one[z - k] : 0);
      const ks = d3.range(B.prev.length);
      topMax = Math.max(d3.max(B.prev), d3.max(B.one)) * 1.15;
      const yt = d3.scaleLinear().domain([0, topMax]).range([150, 20]);
      g.append("g").attr("class", "axis").attr("transform", "translate(0,150)").call(d3.axisBottom(x).ticks(10));
      g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(yt).ticks(4));
      ks.forEach(k => {
        if (x(k) < 50 || x(k) > 665) return;
        g.append("rect").attr("x", x(k) - 6).attr("width", 5).attr("y", yt(B.prev[k])).attr("height", 150 - yt(B.prev[k])).attr("fill", SC.accent);
        const fv = flipped(k);
        g.append("rect").attr("x", x(k) + 1).attr("width", 5).attr("y", yt(fv)).attr("height", 150 - yt(fv)).attr("fill", SC.a2);
        const pv = B.prev[k] * fv;
        if (pv > 0) g.append("circle").attr("cx", x(k)).attr("cy", yt(pv)).attr("r", 3).attr("fill", SC.good);
      });
      fz = B.sum[z] || 0; prodArea = d3.sum(ks, k => B.prev[k] * flipped(k));
      exact = key === "pois" ? PV.poisPmf(z, 5) : fz;
      const yb = d3.scaleLinear().domain([0, d3.max(B.sum) * 1.15]).range([320, 190]);
      g.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(x).ticks(10));
      g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(yb).ticks(4));
      B.sum.forEach((p, k) => { if (x(k) >= 50 && x(k) <= 665) g.append("rect").attr("x", x(k) - 3).attr("width", 6).attr("y", yb(p)).attr("height", 320 - yb(p)).attr("fill", k === z ? SC.good : SC.muted).attr("fill-opacity", k <= z ? 0.9 : 0.35); });
    } else {
      const grid = B.grid, flipped = grid.map(y => F.f(z - y));
      const prod = grid.map((y, j) => B.prev[j] * flipped[j]);
      prodArea = B.area(z);
      topMax = Math.min(Math.max(d3.max(B.prev), d3.max(flipped)) * 1.15, 3);
      const yt = d3.scaleLinear().domain([0, topMax]).range([150, 20]);
      g.append("g").attr("class", "axis").attr("transform", "translate(0,150)").call(d3.axisBottom(x).ticks(10));
      g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(yt).ticks(4));
      const vis = (y) => y >= xd[0] && y <= xd[1];
      const pts = grid.map((y, j) => [y, j]).filter(d => vis(d[0]));
      g.append("path").attr("d", d3.area().x(d => x(d[0])).y0(150).y1(d => yt(Math.min(prod[d[1]], topMax)))(pts)).attr("fill", SC.good).attr("fill-opacity", 0.45);
      g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => yt(Math.min(B.prev[d[1]], topMax)))(pts)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
      g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => yt(Math.min(flipped[d[1]], topMax)))(pts)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
      // the full result curve, by the same numerical rule on a coarser z grid
      const zs = d3.range(0, 241).map(i => zr[0] + i * (zr[1] - zr[0]) / 240);
      const fzs = zs.map(zz => B.area(zz));
      fz = prodArea; exact = F.sum(z, nn);
      const yb = d3.scaleLinear().domain([0, d3.max(fzs) * 1.15]).range([320, 190]);
      g.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(x).ticks(10));
      g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(yb).ticks(4));
      if (nn >= 3) g.append("path").attr("d", d3.line().x(zz => x(zz)).y(zz => yb(ST.normPdf((zz - mean) / Math.sqrt(vr)) / Math.sqrt(vr)))(zs)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
      const done = zs.filter(zz => zz <= z);
      g.append("path").attr("d", d3.line().x((zz, i) => x(zz)).y((zz, i) => yb(fzs[i]))(zs)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-opacity", 0.5);
      g.append("path").attr("d", d3.line().x((zz, i) => x(zz)).y((zz, i) => yb(fzs[i]))(done)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2.5);
      g.append("circle").attr("cx", x(z)).attr("cy", yb(fz)).attr("r", 5).attr("fill", SC.good);
    }
    g.append("line").attr("x1", x(z)).attr("x2", x(z)).attr("y1", 20).attr("y2", 320).attr("stroke", SC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    JD.label(g, 54, 14, nn > 2 && F.n ? `blue: density of the first ${nn - 1} terms · orange: one more term, flipped and shifted to z · green: product` : "blue: f_Y(y) · orange: f_X(z − y), flipped and shifted · green: product", { size: 10.5 });
    JD.label(g, 54, 184, `the sum Z: ${F.name(nn)}${nn >= 3 && !B.disc ? " (dashed: normal with the same mean and variance)" : ""}`, { size: 10.5 });
    JD.label(g, 665, 346, "z", { anchor: "end" });
    out.innerHTML = `z = ${B.disc ? z : z.toFixed(3)}: area under the product = ${B.disc ? "P(Z = z)" : "f_Z(z)"} = <b>${prodArea.toFixed(4)}</b>` +
      ` · closed form (${F.name(nn)}) <b>${exact.toFixed(4)}</b>` +
      `<br>the sum has mean ${mean.toFixed(3)} and variance ${vr.toFixed(4)} (means and variances add)` +
      (key === "unif" && nn === 2 ? "<br>two uniforms: the boxes overlap on an interval of length min(z, 2 − z), so the sum is triangular" : "") +
      (key === "pois" ? "<br>Poisson(2) * Poisson(3) = Poisson(5), by the binomial theorem" : "") +
      (key === "gam" ? "<br>Gamma(2, 1) * Gamma(3, 1) = Gamma(5, 1): shapes add when the rates match" : "");
  }
  const sync = () => { $("cn-nw").style.display = JD.convFam[key].n ? "" : "none"; };
  $("cn-pair").addEventListener("change", e => {
    key = e.target.value; n = 2; $("cn-n").value = 2; $("cn-nv").textContent = "2";
    rebuild(); const zr = zRange(), F = JD.convFam[key];
    const m = F.n ? 2 * F.mean : F.mean + F.mean2; zf = (m - zr[0]) / (zr[1] - zr[0]); $("cn-z").value = Math.round(zf * 1000);
    sync(); draw();
  });
  $("cn-n").addEventListener("input", e => { n = +e.target.value; $("cn-nv").textContent = n; rebuild(); draw(); });
  $("cn-z").addEventListener("input", e => { zf = +e.target.value / 1000; draw(); });
  $("cn-reset").addEventListener("click", () => {
    key = "unif"; n = 2; zf = 0.3; $("cn-pair").value = "unif"; $("cn-n").value = 2; $("cn-nv").textContent = "2"; $("cn-z").value = 300;
    rebuild(); sync(); draw();
  });
  rebuild(); sync(); draw();
})();

/* ─────────────────── 6 · Box–Muller: a cell and its image ───────────────────
   Caption audit: cell U₁ ∈ [0.30, 0.40], U₂ ∈ [0.10, 0.20], 1,500 points, seed JD.seeds.polar →
   radii 1.354 to 1.552, angles 36°–72°, normal mass 0.0100; the point count (expected 15) is
   printed by the build check and recorded in the caption.                                          */
JD.polarPoints = function (N, seed) {
  const r = ST.rng(seed), pts = [];
  for (let i = 0; i < N; i++) {
    let u1 = r(); while (u1 === 0) u1 = r();
    const u2 = r(), R = Math.sqrt(-2 * Math.log(u1)), t = 2 * Math.PI * u2;
    pts.push({ u1, u2, z1: R * Math.cos(t), z2: R * Math.sin(t) });
  }
  return pts;
};
JD.sectorMass = (a, d, b) => {                     // cell [a, a + d] × [b, b + d] → normal mass of its image
  const rin = Math.sqrt(-2 * Math.log(a + d)), rout = Math.sqrt(-2 * Math.log(a));
  return { rin, rout, th1: 2 * Math.PI * b, th2: 2 * Math.PI * (b + d), mass: (Math.exp(-rin * rin / 2) - Math.exp(-rout * rout / 2)) * d };
};
(function () {
  const svg = d3.select("#polar-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("polar-readout");
  let seed = JD.seeds.polar, pts = JD.polarPoints(1500, seed), a = 0.3, b = 0.1, d = 0.1;
  const ux = d3.scaleLinear().domain([0, 1]).range([40, 320]), uy = d3.scaleLinear().domain([0, 1]).range([310, 30]);
  const zx = d3.scaleLinear().domain([-3.6, 3.6]).range([380, 660]), zy = d3.scaleLinear().domain([-3.6, 3.6]).range([310, 30]);
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), S = JD.sectorMass(a, d, b);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,310)").call(d3.axisBottom(ux).ticks(5));
    g.append("g").attr("class", "axis").attr("transform", "translate(40,0)").call(d3.axisLeft(uy).ticks(5));
    g.append("g").attr("class", "axis").attr("transform", "translate(0,310)").call(d3.axisBottom(zx).ticks(7));
    g.append("g").attr("class", "axis").attr("transform", "translate(380,0)").call(d3.axisLeft(zy).ticks(7));
    JD.label(g, 320, 338, "U₁  (radius √(−2 ln U₁))", { anchor: "end" });
    JD.label(g, 44, 22, "U₂  (angle 2πU₂)", {});
    JD.label(g, 660, 338, "Z₁", { anchor: "end" });
    JD.label(g, 384, 22, "Z₂", {});
    const inCell = p => p.u1 >= a && p.u1 < a + d && p.u2 >= b && p.u2 < b + d;
    let cnt = 0;
    pts.forEach(p => {
      const hit = inCell(p); if (hit) cnt++;
      g.append("circle").attr("cx", ux(p.u1)).attr("cy", uy(p.u2)).attr("r", hit ? 3 : 1.6).attr("fill", hit ? SC.a2 : SC.accent).attr("fill-opacity", hit ? 1 : 0.45);
      if (Math.abs(p.z1) < 3.6 && Math.abs(p.z2) < 3.6)
        g.append("circle").attr("cx", zx(p.z1)).attr("cy", zy(p.z2)).attr("r", hit ? 3 : 1.6).attr("fill", hit ? SC.a2 : SC.accent).attr("fill-opacity", hit ? 1 : 0.45);
    });
    // image sector, drawn as a polygon in z-space (clipped to the panel radius)
    const rcap = r => Math.min(r, 3.6), poly = [];
    for (let i = 0; i <= 40; i++) { const t = S.th1 + (S.th2 - S.th1) * i / 40; poly.push([rcap(S.rout) * Math.cos(t), rcap(S.rout) * Math.sin(t)]); }
    for (let i = 40; i >= 0; i--) { const t = S.th1 + (S.th2 - S.th1) * i / 40; poly.push([S.rin * Math.cos(t), S.rin * Math.sin(t)]); }
    g.append("path").attr("d", "M" + poly.map(q => `${zx(q[0])},${zy(q[1])}`).join("L") + "Z").attr("fill", SC.a2).attr("fill-opacity", 0.15).attr("stroke", SC.a2).attr("stroke-width", 1.5);
    // draggable cell
    const cell = g.append("rect").attr("x", ux(a)).attr("y", uy(b + d)).attr("width", ux(a + d) - ux(a)).attr("height", uy(b) - uy(b + d))
      .attr("fill", SC.a2).attr("fill-opacity", 0.15).attr("stroke", SC.a2).attr("stroke-width", 2).style("cursor", "move");
    let off = null;
    cell.call(d3.drag()
      .on("start", ev => { off = [ux.invert(ev.x) - a, uy.invert(ev.y) - b]; })
      .on("drag", ev => { a = ST.clamp(ux.invert(ev.x) - off[0], 0.001, 1 - d); b = ST.clamp(uy.invert(ev.y) - off[1], 0, 1 - d); draw(); }));
    const cz = JD.sectorMass(a + d / 2, 0, b + d / 2), rc = Math.sqrt(-2 * Math.log(a + d / 2));
    const z1s = pts.map(p => p.z1), z2s = pts.map(p => p.z2);
    out.innerHTML = `cell U₁ ∈ [${a.toFixed(3)}, ${(a + d).toFixed(3)}], U₂ ∈ [${b.toFixed(3)}, ${(b + d).toFixed(3)}]: area <b>${(d * d).toFixed(4)}</b>` +
      ` → image: radius ${S.rin.toFixed(3)} to ${S.rout.toFixed(3)}, angle ${(S.th1 * 180 / Math.PI).toFixed(0)}° to ${(S.th2 * 180 / Math.PI).toFixed(0)}°` +
      `<br>normal probability of the sector (e^(−r_in²/2) − e^(−r_out²/2))·Δθ/2π = <b>${S.mass.toFixed(4)}</b> — equal to the cell's area` +
      `<br>points: <b>${cnt}</b> of ${pts.length} in the cell, the same ${cnt} in the sector (expected ${(pts.length * d * d).toFixed(1)})` +
      `<br>local Jacobian |∂(u₁, u₂)/∂(z₁, z₂)| at the cell's centre = e^(−r²/2)/2π = ${(Math.exp(-rc * rc / 2) / (2 * Math.PI)).toFixed(4)} = φ(z₁)φ(z₂)` +
      ` · sample: mean Z₁ ${d3.mean(z1s).toFixed(3)}, var Z₁ ${d3.variance(z1s).toFixed(3)}, corr(Z₁, Z₂) ${ST.corr(z1s, z2s).toFixed(3)}`;
    void cz;
  }
  $("po-d").addEventListener("input", e => { d = +e.target.value; $("po-dv").textContent = d.toFixed(2); a = Math.min(a, 1 - d); b = Math.min(b, 1 - d); draw(); });
  $("po-sim").addEventListener("click", () => { seed += 1; pts = JD.polarPoints(1500, seed); draw(); });
  $("po-reset").addEventListener("click", () => {
    seed = JD.seeds.polar; pts = JD.polarPoints(1500, seed); a = 0.3; b = 0.1; d = 0.1; $("po-d").value = 0.1; $("po-dv").textContent = "0.10"; draw();
  });
  draw();
})();

/* ─────────────────── 7 · the k-th order statistic ───────────────────
   Caption audit: U(0, 1), n = 5, k = 3, 20,000 samples, seed JD.seeds.order → exact Beta(3, 3):
   mean 0.5, var 0.035714, P(U₍₃₎ ≤ 0.3) = 0.16308; simulated values printed by the build check.   */
JD.osDist = {
  unif: { F: x => ST.clamp(x, 0, 1), f: x => (x >= 0 && x <= 1 ? 1 : 0), Q: u => u, dom: [0, 1], q: 0.3 },
  exp: { F: x => PV.expCdf(x, 1), f: x => PV.expPdf(x, 1), Q: u => -Math.log1p(-u), dom: [0, 6], q: 1 },
  norm: { F: x => ST.normCdf(x), f: x => ST.normPdf(x), Q: u => ST.normQuant(u), dom: [-3.5, 3.5], q: 0 }
};
JD.osPdf = (D, x, k, n) => {
  const F = D.F(x), f = D.f(x);
  if (f <= 0 || F <= 0 && k > 1 || F >= 1 && k < n) return 0;
  return Math.exp(ST.lnGamma(n + 1) - ST.lnGamma(k) - ST.lnGamma(n - k + 1) + (k - 1) * Math.log(Math.max(F, 1e-300)) + (n - k) * Math.log(Math.max(1 - F, 1e-300))) * f;
};
JD.osCdf = (D, x, k, n) => ST.betaI(k, n - k + 1, D.F(x));
JD.osSim = function (dist, n, M, seed) {           // M sorted samples of size n (via sorted uniforms)
  const r = ST.rng(seed), D = JD.osDist[dist], out = [];
  for (let m = 0; m < M; m++) {
    const u = []; for (let i = 0; i < n; i++) u.push(r());
    u.sort((p, q) => p - q);
    out.push(u.map(D.Q));
  }
  return out;
};
JD.osExact = function (dist, k, n) {                // exact mean and variance of X₍ₖ₎
  if (dist === "unif") return { m: k / (n + 1), v: k * (n - k + 1) / ((n + 1) ** 2 * (n + 2)) };
  if (dist === "exp") { let m = 0, v = 0; for (let i = n - k + 1; i <= n; i++) { m += 1 / i; v += 1 / (i * i); } return { m, v }; }
  const D = JD.osDist.norm, h = (x) => JD.osPdf(D, x, k, n);
  const m = PV.simpson(x => x * h(x), -8, 8, 4000), s2 = PV.simpson(x => x * x * h(x), -8, 8, 4000);
  return { m, v: s2 - m * m };
};
(function () {
  const svg = d3.select("#order-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("order-readout");
  let dist = "unif", n = 5, k = 3, seed = JD.seeds.order, sims = null;
  const resim = () => { sims = JD.osSim(dist, n, 20000, seed); };
  function draw() {
    svg.selectAll("*").remove();
    const D = JD.osDist[dist], g = svg.append("g");
    const x = d3.scaleLinear().domain(D.dom).range([50, 665]);
    // strip of six samples
    for (let s = 0; s < 6; s++) {
      const yy = 18 + s * 12;
      g.append("line").attr("x1", 50).attr("x2", 665).attr("y1", yy).attr("y2", yy).attr("stroke", SC.line);
      sims[s].forEach((v, i) => { if (v >= D.dom[0] && v <= D.dom[1]) g.append("circle").attr("cx", x(v)).attr("cy", yy).attr("r", i === k - 1 ? 4 : 2.4).attr("fill", i === k - 1 ? SC.a2 : SC.muted); });
    }
    JD.label(g, 46, 22, "6 samples", { anchor: "end", size: 9.5 });
    const vals = sims.map(s => s[k - 1]);
    const bins = 50, w = (D.dom[1] - D.dom[0]) / bins, cnt = new Array(bins).fill(0);
    vals.forEach(v => { const i = Math.floor((v - D.dom[0]) / w); if (i >= 0 && i < bins) cnt[i]++; });
    const dens = cnt.map(c => c / (vals.length * w));
    const xs = d3.range(0, 401).map(i => D.dom[0] + i * (D.dom[1] - D.dom[0]) / 400), fx = xs.map(v => JD.osPdf(D, v, k, n));
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(dens), d3.max(fx)) * 1.1]).range([320, 100]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(y).ticks(5));
    dens.forEach((dv, i) => g.append("rect").attr("x", x(D.dom[0] + i * w) + 0.5).attr("width", Math.max(0.5, x(D.dom[0] + w) - x(D.dom[0]) - 1)).attr("y", y(dv)).attr("height", 320 - y(dv)).attr("fill", SC.accent).attr("fill-opacity", 0.55));
    g.append("path").attr("d", d3.line().x((v, i) => x(v)).y((v, i) => y(fx[i]))(xs)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
    const E = JD.osExact(dist, k, n);
    g.append("line").attr("x1", x(E.m)).attr("x2", x(E.m)).attr("y1", 100).attr("y2", 320).attr("stroke", SC.ink).attr("stroke-dasharray", "2 3");
    const nm = dist === "unif" ? `Beta(${k}, ${n - k + 1})` : `exact density of X₍${JD.sub(k)}₎`;
    JD.label(g, 60, 112, `histogram of X₍${JD.sub(k)}₎ over 20,000 samples of n = ${n} · curve: ${nm}`, { size: 10.5, color: SC.ink });
    // readout
    const sm = d3.mean(vals), sv = d3.variance(vals), q = D.q;
    const pe = JD.osCdf(D, q, k, n), ps = vals.filter(v => v <= q).length / vals.length;
    const sorted = vals.slice().sort((p, r) => p - r);
    let ks = 0; sorted.forEach((v, i) => { const F = JD.osCdf(D, v, k, n); ks = Math.max(ks, Math.abs(F - i / sorted.length), Math.abs(F - (i + 1) / sorted.length)); });
    out.innerHTML = `X₍${JD.sub(k)}₎ of n = ${n} ${dist === "unif" ? "uniforms ~ <b>Beta(" + k + ", " + (n - k + 1) + ")</b>" : dist === "exp" ? "Exp(1) draws" : "N(0, 1) draws"}` +
      `<br>mean: exact <b>${E.m.toFixed(4)}</b>, simulated ${sm.toFixed(4)} · variance: exact <b>${E.v.toFixed(5)}</b>, simulated ${sv.toFixed(5)}` +
      `<br>P(X₍${JD.sub(k)}₎ ≤ ${q}) = I_F(${q})(${k}, ${n - k + 1}) = <b>${pe.toFixed(5)}</b> = P(Bin(${n}, ${D.F(q).toFixed(3)}) ≥ ${k}) · simulated ${ps.toFixed(4)}` +
      `<br>Kolmogorov distance between the simulated and exact cdfs: ${ks.toFixed(4)} (1% critical value for 20,000 draws: 0.0115)` +
      (dist === "exp" && k === 1 ? `<br>the minimum of ${n} Exp(1) draws is Exp(${n}): mean 1/${n}` : "");
  }
  function syncK() { $("os-k").max = n; if (k > n) k = n; $("os-k").value = k; $("os-kv").textContent = k; }
  $("os-dist").addEventListener("change", e => { dist = e.target.value; resim(); draw(); });
  $("os-n").addEventListener("input", e => { n = +e.target.value; $("os-nv").textContent = n; syncK(); resim(); draw(); });
  $("os-k").addEventListener("input", e => { k = +e.target.value; $("os-kv").textContent = k; draw(); });
  $("os-sim").addEventListener("click", () => { seed += 1; resim(); draw(); });
  $("os-reset").addEventListener("click", () => {
    dist = "unif"; n = 5; k = 3; seed = JD.seeds.order; $("os-dist").value = "unif"; $("os-n").value = 5; $("os-nv").textContent = "5"; syncK(); resim(); draw();
  });
  syncK(); resim(); draw();
})();

/* ─────────────────── 8 · the bivariate normal ───────────────────
   Caption audit: standard units, ρ = 0.6, x₀ = 2, 600 points, seed JD.seeds.bvn → E[Y | X = 2] = 1.2,
   sd 0.8, 95% range −0.368 to 2.768, SD line 2; band count and mean printed by the build check.     */
JD.bvnPresets = { std: { mx: 0, my: 0, sx: 1, sy: 1, r: 0.6, xn: "X", yn: "Y" }, exam: { mx: 49.5, my: 69.1, sx: 10.2, sy: 11.8, r: 0.67, xn: "midterm", yn: "final" } };
JD.bvnZ = function (N, seed) { const r = ST.rng(seed), z = []; for (let i = 0; i < N; i++) z.push([ST.randn(r), ST.randn(r)]); return z; };
JD.bvnBand = function (z, rho, x0z, hw) {          // points with |z_x − x0z| < hw, in standard units
  const ys = [];
  z.forEach(([a, b]) => { if (Math.abs(a - x0z) < hw) ys.push(rho * a + Math.sqrt(1 - rho * rho) * b); });
  return { n: ys.length, mean: ys.length ? d3.mean(ys) : NaN };
};
(function () {
  const svg = d3.select("#bvn-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("bvn-readout");
  let pre = "std", P = Object.assign({}, JD.bvnPresets.std), seed = JD.seeds.bvn, Z = JD.bvnZ(600, seed), x0z = 2;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), rho = P.r, s1 = Math.sqrt(1 - rho * rho);
    const x = d3.scaleLinear().domain([P.mx - 3.5 * P.sx, P.mx + 3.5 * P.sx]).range([50, 470]);
    const y = d3.scaleLinear().domain([P.my - 3.5 * P.sy, P.my + 3.5 * P.sy]).range([350, 14]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,350)").call(d3.axisBottom(x).ticks(7));
    g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(y).ticks(7));
    JD.label(g, 470, 376, P.xn, { anchor: "end" });
    JD.label(g, 54, 26, P.yn, {});
    const X = zz => P.mx + P.sx * zz, Yv = (a, b) => P.my + P.sy * (rho * a + s1 * b);
    // band
    const hw = 0.25, x0 = X(x0z);
    g.append("rect").attr("x", x(X(x0z - hw))).attr("width", x(X(x0z + hw)) - x(X(x0z - hw))).attr("y", 14).attr("height", 336).attr("fill", SC.a2).attr("fill-opacity", 0.1);
    Z.forEach(([a, b]) => g.append("circle").attr("cx", x(X(a))).attr("cy", y(Yv(a, b))).attr("r", 2).attr("fill", Math.abs(a - x0z) < hw ? SC.a2 : SC.accent).attr("fill-opacity", 0.6));
    // ellipses c = 1, 2, 3
    [1, 2, 3].forEach(c => {
      const pts = d3.range(0, 2 * Math.PI + 0.01, 0.05).map(t => [X(c * Math.cos(t)), Yv(c * Math.cos(t), c * Math.sin(t))]);
      g.append("path").attr("d", d3.line().x(q => x(q[0])).y(q => y(q[1]))(pts)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-opacity", 0.35);
    });
    // lines: SD line, regression of Y on X, of X on Y
    const sdSlope = (rho >= 0 ? 1 : -1) * P.sy / P.sx;
    ST.lineSeg(g, P.my - sdSlope * P.mx, sdSlope, x, y, { color: SC.muted, dash: "6 4", w: 1.5 });
    const b1 = rho * P.sy / P.sx;
    ST.lineSeg(g, P.my - b1 * P.mx, b1, x, y, { color: SC.a2, w: 2.5 });
    if (Math.abs(rho) > 0.02) { const b2 = P.sy / (rho * P.sx); ST.lineSeg(g, P.my - b2 * P.mx, b2, x, y, { color: SC.violet, w: 1.8 }); }
    // conditional density sideways at x0
    const cm = P.my + b1 * (x0 - P.mx), cs = P.sy * s1, ys = d3.range(-3.5, 3.51, 0.05).map(t => cm + t * cs);
    const peak = 1 / (cs * Math.sqrt(2 * Math.PI)), wpx = 60;
    g.append("path").attr("d", d3.line().x(v => x(x0) + wpx * (ST.normPdf((v - cm) / cs) / cs) / peak).y(v => y(v))(ys)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2);
    g.append("circle").attr("cx", x(x0)).attr("cy", y(cm)).attr("r", 4.5).attr("fill", SC.good);
    g.append("circle").attr("cx", x(x0)).attr("cy", y(P.my + sdSlope * (x0 - P.mx))).attr("r", 4).attr("fill", "none").attr("stroke", SC.ink);
    // drag handle
    const hd = g.append("rect").attr("x", x(x0) - 9).attr("y", 14).attr("width", 18).attr("height", 336).attr("fill", "transparent").style("cursor", "ew-resize");
    hd.call(d3.drag().on("drag", ev => { x0z = ST.clamp((x.invert(ev.x) - P.mx) / P.sx, -3.2, 3.2); draw(); }));
    ST.legend(g, [{ label: "SD line", color: SC.muted, dash: "6 4" }, { label: `E[${P.yn} | ${P.xn}]`, color: SC.a2, dash: "1 0" },
      { label: `E[${P.xn} | ${P.yn}]`, color: SC.violet, dash: "1 0" }, { label: "conditional density", color: SC.good, dash: "1 0" }], 490, 30);
    JD.label(g, 490, 110, "ellipses hold 39%, 86.5%, 98.9%", { size: 10 });
    JD.label(g, 490, 126, "drag the shaded band", { size: 10 });
    const B = JD.bvnBand(Z, rho, x0z, hw);
    const mahal = (a, b) => Math.sqrt((a * a - 2 * rho * a * b + b * b) / (1 - rho * rho));
    out.innerHTML = `${P.xn} = ${x0.toFixed(2)} (${x0z.toFixed(2)} sd from its mean): ${P.yn} | ${P.xn} ~ N(<b>${cm.toFixed(3)}</b>, sd <b>${cs.toFixed(3)}</b>)` +
      ` · 95% range ${(cm - 1.96 * cs).toFixed(3)} to ${(cm + 1.96 * cs).toFixed(3)}` +
      `<br>the SD line would predict ${(P.my + sdSlope * (x0 - P.mx)).toFixed(3)}; in standard units the prediction is ρ·z = ${(rho * x0z).toFixed(3)} sd, not ${x0z.toFixed(2)}: <b>regression to the mean</b>` +
      `<br>sample points in the band (|z − ${x0z.toFixed(2)}| &lt; 0.25): ${B.n}, their mean ${P.yn} = ${isFinite(B.mean) ? (P.my + P.sy * B.mean).toFixed(3) : "—"} · residual variance share 1 − ρ² = ${(1 - rho * rho).toFixed(3)}` +
      `<br>Mahalanobis distance (standard units): (2, 2) → ${mahal(2, 2).toFixed(3)}, (2, −2) → ${mahal(2, -2).toFixed(3)}; both are 2.828 away in Euclidean distance`;
  }
  $("bv-pre").addEventListener("change", e => { pre = e.target.value; P = Object.assign({}, JD.bvnPresets[pre]); $("bv-r").value = P.r; $("bv-rv").textContent = P.r.toFixed(2); x0z = pre === "std" ? 2 : (41 - 49.5) / 10.2; draw(); });
  $("bv-r").addEventListener("input", e => { P.r = +e.target.value; $("bv-rv").textContent = P.r.toFixed(2); draw(); });
  $("bv-sim").addEventListener("click", () => { seed += 1; Z = JD.bvnZ(600, seed); draw(); });
  $("bv-reset").addEventListener("click", () => {
    pre = "std"; P = Object.assign({}, JD.bvnPresets.std); seed = JD.seeds.bvn; Z = JD.bvnZ(600, seed); x0z = 2;
    $("bv-pre").value = "std"; $("bv-r").value = 0.6; $("bv-rv").textContent = "0.60"; draw();
  });
  draw();
})();

/* ─────────────────── 9 · the Dirichlet on the simplex ───────────────────
   Caption audit: α = (2, 3, 5), 1,500 draws, seed JD.seeds.dir → exact means 0.2, 0.3, 0.5,
   Var p₁ = 0.014545, Cov(p₁, p₂) = −0.005455; sample values printed by the build check.            */
JD.dirSample = function (alpha, N, seed) { const r = ST.rng(seed), out = []; for (let i = 0; i < N; i++) out.push(PV.dirichletDraw(alpha, r)); return out; };
JD.dirExact = function (alpha) {
  const a0 = d3.sum(alpha);
  return { mean: alpha.map(a => a / a0), var: alpha.map(a => a * (a0 - a) / (a0 * a0 * (a0 + 1))), cov12: -alpha[0] * alpha[1] / (a0 * a0 * (a0 + 1)) };
};
(function () {
  const svg = d3.select("#dir-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("dir-readout");
  let alpha = [2, 3, 5], seed = JD.seeds.dir;
  const A = [60, 320], Bc = [400, 320], C = [230, 320 - 170 * Math.sqrt(3)];
  const pix = p => [p[0] * A[0] + p[1] * Bc[0] + p[2] * C[0], p[0] * A[1] + p[1] * Bc[1] + p[2] * C[1]];
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g");
    // density shading on a triangular grid
    const N = 46, cells = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N - i; j++) {
      const p1 = (i + 1 / 3) / N, p2 = (j + 1 / 3) / N, p3 = 1 - p1 - p2;
      if (p3 <= 0) continue;
      cells.push({ p: [p1, p2, p3], v: Math.exp(PV.dirichletLogPdf([p1, p2, p3], alpha)) });
    }
    const cap = d3.quantile(cells.map(c => c.v).sort(d3.ascending), 0.98) || 1;
    cells.forEach(c => { const q = pix(c.p); g.append("circle").attr("cx", q[0]).attr("cy", q[1]).attr("r", 4.3).attr("fill", SC.accent).attr("fill-opacity", 0.05 + 0.6 * Math.min(1, c.v / cap)); });
    g.append("path").attr("d", `M${A}L${Bc}L${C}Z`).attr("fill", "none").attr("stroke", SC.ink);
    JD.label(g, A[0] - 4, A[1] + 16, "p₁ = 1", { anchor: "middle", color: SC.ink });
    JD.label(g, Bc[0] + 4, Bc[1] + 16, "p₂ = 1", { anchor: "middle", color: SC.ink });
    JD.label(g, C[0], C[1] - 8, "p₃ = 1", { anchor: "middle", color: SC.ink });
    const S = JD.dirSample(alpha, 1500, seed);
    S.forEach(p => { const q = pix(p); g.append("circle").attr("cx", q[0]).attr("cy", q[1]).attr("r", 1.6).attr("fill", SC.a2).attr("fill-opacity", 0.7); });
    const E = JD.dirExact(alpha), mq = pix(E.mean);
    g.append("path").attr("d", `M${mq[0] - 7},${mq[1]}L${mq[0] + 7},${mq[1]}M${mq[0]},${mq[1] - 7}L${mq[0]},${mq[1] + 7}`).attr("stroke", SC.ink).attr("stroke-width", 2.5);
    // right: means with ±1 sd
    const sm = [0, 1, 2].map(i => d3.mean(S, p => p[i]));
    const bx = d3.scaleBand().domain(["p₁", "p₂", "p₃"]).range([470, 660]).padding(0.35), by = d3.scaleLinear().domain([0, 1]).range([320, 60]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(bx));
    g.append("g").attr("class", "axis").attr("transform", "translate(470,0)").call(d3.axisLeft(by).ticks(5));
    ["p₁", "p₂", "p₃"].forEach((nm, i) => {
      g.append("rect").attr("x", bx(nm)).attr("width", bx.bandwidth()).attr("y", by(E.mean[i])).attr("height", 320 - by(E.mean[i])).attr("fill", SC.accent).attr("fill-opacity", 0.6);
      const sd = Math.sqrt(E.var[i]);
      g.append("line").attr("x1", bx(nm) + bx.bandwidth() / 2).attr("x2", bx(nm) + bx.bandwidth() / 2).attr("y1", by(Math.min(1, E.mean[i] + sd))).attr("y2", by(Math.max(0, E.mean[i] - sd))).attr("stroke", SC.ink);
      g.append("circle").attr("cx", bx(nm) + bx.bandwidth() / 2).attr("cy", by(sm[i])).attr("r", 4).attr("fill", SC.a2);
    });
    JD.label(g, 470, 46, "exact mean ± sd (bars), sample mean (dot)", { size: 10 });
    const sv = d3.mean(S, p => (p[0] - sm[0]) ** 2) * S.length / (S.length - 1);
    const sc = d3.sum(S, p => (p[0] - sm[0]) * (p[1] - sm[1])) / (S.length - 1);
    const a0 = d3.sum(alpha);
    out.innerHTML = `α = (${alpha.map(v => v.toFixed(1)).join(", ")}), α₀ = ${a0.toFixed(1)} · exact means <b>${E.mean.map(v => v.toFixed(4)).join(", ")}</b> · sample means ${sm.map(v => v.toFixed(4)).join(", ")}` +
      `<br>Var p₁: exact <b>${E.var[0].toFixed(5)}</b>, sample ${sv.toFixed(5)} · Cov(p₁, p₂): exact <b>${E.cov12.toFixed(5)}</b>, sample ${sc.toFixed(5)} (negative: the entries share a total of 1)` +
      `<br>each pᵢ ~ Beta(αᵢ, α₀ − αᵢ); ${Math.min(...alpha) < 1 ? "αᵢ &lt; 1: mass pushed to the edges and corners (sparse vectors)" : Math.min(...alpha) > 10 ? "large α₀: concentrated near the mean" : "observing multinomial counts adds them to α"}`;
  }
  const setUI = () => { [0, 1, 2].forEach(i => { $(`di-a${i + 1}`).value = Math.min(alpha[i], 20); $(`di-a${i + 1}v`).textContent = alpha[i].toFixed(1); }); };
  [0, 1, 2].forEach(i => $(`di-a${i + 1}`).addEventListener("input", e => { alpha[i] = +e.target.value; $(`di-a${i + 1}v`).textContent = alpha[i].toFixed(1); draw(); }));
  $("di-obs").addEventListener("click", () => { alpha = alpha.map((a, i) => a + [3, 1, 6][i]); setUI(); draw(); });
  $("di-sim").addEventListener("click", () => { seed += 1; draw(); });
  $("di-reset").addEventListener("click", () => { alpha = [2, 3, 5]; seed = JD.seeds.dir; setUI(); draw(); });
  setUI(); draw();
})();
