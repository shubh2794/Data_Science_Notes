/* stochastic-processes.viz.js — the nine visualizations on math/probability/stochastic-processes.html.
   Loaded after ../../data.js → ../../notes.js → ../statistics/stats-viz.js (SC palette, ST toolbox:
   ST.rng = the 32-bit mixer used throughout these pages, ST.randn, ST.normCdf/Pdf/Quant, ST.binomPmf …)
   → prob-viz.js (PV: exponential/gamma/beta/Cauchy/Poisson functions, PV.ksDist, PV.tv, PV.simpson,
   and the Markov-chain / autocorrelation helpers appended for this page). Distribution functions come
   from ST / PV and are never re-implemented here.

   Everything page-local lives under the single namespace SP. Each figure is an IIFE that exits quietly
   if its <svg> is not on the page. The pure computations are exposed on SP so the build's node checks
   can call them without a DOM, and every caption number was produced by those functions.

     1  #inv-svg     inverse transform: U on the vertical axis, F⁻¹(U) on the horizontal, histogram of draws
     2  #rej-svg     rejection sampling: points under c·q, accepted under f, acceptance rate against 1/c
     3  #mc-svg      Monte Carlo against a product grid in d dimensions: running estimate ± 2 SE, error vs n
     4  #vr-svg      variance-reduction race: five estimators of one integral at an equal budget
     5  #pp-svg      the Poisson process: timeline, counting staircase, superposition, thinning, inspection
     6  #markov-svg  a three-state Markov chain: walker, empirical vs stationary, distance to π vs |λ₂|ⁿ
     7  #erg-svg     the Markov-chain CLT: replicate chain averages against the naive and the honest normal
     8  #mcmc-svg    random-walk Metropolis on one or two modes: trace, histogram, ACF, ESS, R̂
     9  #queue-svg   the M/M/1 queue: queue-length path, time averages against theory, Little's law */

const SP = {
  // Default seeds, fixed by the caption audit (see the note at each figure).
  seeds: { inv: 3, rej: 5, mc: 8, vr: 4, pp: 6, mk: 2, erg: 9, mh: 2, qu: 3 },
  label(g, x, y, t, o) {
    const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", (o && o.size) || 11)
      .attr("fill", (o && o.color) || SC.muted).text(t);
    if (o && o.anchor) e.attr("text-anchor", o.anchor);
    if (o && o.weight) e.attr("font-weight", o.weight);
    return e;
  },
  f(x, d) { const dd = d === undefined ? 4 : d; return (x !== 0 && isFinite(x) && Math.abs(x) < Math.pow(10, -dd) / 2) ? x.toExponential(2) : ST.fmt(x, dd); },
  hist(a, lo, hi, nb) {                             // density-scaled histogram; values outside are counted, not drawn
    const w = (hi - lo) / nb, c = new Array(nb).fill(0);
    let out = 0;
    for (const v of a) { const k = Math.floor((v - lo) / w); if (k >= 0 && k < nb) c[k]++; else if (v === hi) c[nb - 1]++; else out++; }
    return { bins: c.map((k, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, d: k / (a.length * w), n: k })), out };
  },
  bars(g, bins, xs, ys, color, op) {
    g.append("g").selectAll("rect").data(bins).join("rect")
      .attr("x", d => xs(d.x0) + 0.5).attr("width", d => Math.max(0, xs(d.x1) - xs(d.x0) - 1))
      .attr("y", d => ys(d.d)).attr("height", d => Math.max(0, ys(0) - ys(d.d)))
      .attr("fill", color || SC.accent).attr("fill-opacity", op === undefined ? 0.45 : op);
  },
  curve(g, fn, xs, ys, lo, hi, o) {
    const pts = d3.range(0, 301).map(i => lo + (hi - lo) * i / 300).map(x => [x, fn(x)]).filter(p => isFinite(p[1]));
    return g.append("path").attr("d", d3.line().x(p => xs(p[0])).y(p => ys(p[1]))(pts)).attr("fill", "none")
      .attr("stroke", (o && o.color) || SC.a2).attr("stroke-width", (o && o.w) || 2).attr("stroke-dasharray", (o && o.dash) || null);
  },
  axisB(g, xs, y, ticks, fmt) { return g.append("g").attr("class", "axis").attr("transform", `translate(0,${y})`).call(fmt ? d3.axisBottom(xs).ticks(ticks || 5, fmt) : d3.axisBottom(xs).ticks(ticks || 5)); },
  axisL(g, ys, x, ticks, fmt) { return g.append("g").attr("class", "axis").attr("transform", `translate(${x},0)`).call(fmt ? d3.axisLeft(ys).ticks(ticks || 5, fmt) : d3.axisLeft(ys).ticks(ticks || 5)); },
  unit(r) { let u = r(); while (u === 0) u = r(); return u; },
  $: id => document.getElementById(id)
};

/* ─────────────────── 1 · inverse transform ───────────────────
   Caption audit: distribution "exp" (Exp(1), F⁻¹(u) = −ln(1 − u)), U slider 0.70 → x = −ln 0.30 = 1.2040;
   200 draws from seed SP.seeds.inv; the sample mean and the Kolmogorov distance to F are printed by
   SP.invDefault() and recorded in the caption.                                                         */
SP.invDist = {
  exp: { name: "Exp(1)", lo: 0, hi: 5, disc: false, mean: 1,
    F: x => (x <= 0 ? 0 : 1 - Math.exp(-x)), Q: u => -Math.log1p(-u), f: x => (x < 0 ? 0 : Math.exp(-x)),
    rule: "x = F⁻¹(U) = −ln(1 − U)" },
  bin: { name: "Bin(10, ¼)", lo: -0.5, hi: 10.5, disc: true, mean: 2.5, K: 10,
    p: k => ST.binomPmf(k, 10, 0.25), rule: "x = smallest k with F(k) ≥ U" },
  cau: { name: "Cauchy", lo: -8, hi: 8, disc: false, mean: NaN,
    F: x => PV.cauchyCdf(x, 0, 1), Q: u => PV.cauchyQuant(u, 0, 1), f: x => PV.cauchyPdf(x, 0, 1),
    rule: "x = tan(π(U − ½))" },
  gap: { name: "uniform on [0, 1] ∪ [2, 3]", lo: -0.25, hi: 3.25, disc: false, mean: 1.5,
    F: x => (x <= 0 ? 0 : x <= 1 ? x / 2 : x <= 2 ? 0.5 : x <= 3 ? 0.5 + (x - 2) / 2 : 1),
    Q: u => (u <= 0.5 ? 2 * u : 1 + 2 * u), f: x => ((x > 0 && x < 1) || (x > 2 && x < 3) ? 0.5 : 0),
    rule: "x = min{x : F(x) ≥ U} (2U, or 1 + 2U)" }
};
SP.invBin = (function () {                          // cdf table for Bin(10, ¼), used by binary search
  const F = []; let s = 0;
  for (let k = 0; k <= 10; k++) { s += ST.binomPmf(k, 10, 0.25); F.push(s); }
  F[10] = 1;
  return F;
})();
SP.invQ = function (key, u) {
  if (key !== "bin") return SP.invDist[key].Q(u);
  let lo = 0, hi = 10;                              // binary search: smallest k with F(k) ≥ u
  while (lo < hi) { const m = (lo + hi) >> 1; if (SP.invBin[m] >= u) hi = m; else lo = m + 1; }
  return lo;
};
SP.invDraws = function (key, r, n) { const out = []; for (let i = 0; i < n; i++) { const u = r(); out.push({ u, x: SP.invQ(key, u) }); } return out; };
SP.invStats = function (key, draws) {
  const D = SP.invDist[key], xs = draws.map(d => d.x), n = xs.length;
  const mean = n ? xs.reduce((a, b) => a + b, 0) / n : NaN;
  if (D.disc) {
    const emp = new Array(D.K + 1).fill(0); xs.forEach(x => emp[x]++);
    const pm = d3.range(D.K + 1).map(D.p);
    return { mean, dist: n ? PV.tv(emp.map(c => c / n), pm) : NaN, kind: "TV", emp };
  }
  return { mean, dist: n ? PV.ksDist(xs.slice().sort((a, b) => a - b), D.F) : NaN, kind: "KS" };
};
SP.invDefault = function () { const d = SP.invDraws("exp", ST.rng(SP.seeds.inv), 200); return SP.invStats("exp", d); };
(function () {
  const svg = d3.select("#inv-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("inv-readout");
  let key = "exp", u = 0.7, r = ST.rng(SP.seeds.inv), draws = SP.invDraws("exp", r, 200);
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), D = SP.invDist[key];
    const L = { x0: 52, x1: 318, y0: 300, y1: 30 }, Rr = { x0: 392, x1: 664 };
    const xs = d3.scaleLinear().domain([D.lo, D.hi]).range([L.x0, L.x1]);
    const us = d3.scaleLinear().domain([0, 1]).range([L.y0, L.y1]);
    SP.axisB(g, xs, L.y0, 6); SP.axisL(g, us, L.x0, 5);
    SP.label(g, L.x0 - 40, L.y1 - 10, "U (uniform on the vertical axis) → F⁻¹(U) on the horizontal", { size: 10.5 });
    // cdf
    if (D.disc) {
      let prev = 0;
      for (let k = 0; k <= D.K; k++) {
        const Fk = SP.invBin[k];
        g.append("line").attr("x1", xs(k)).attr("x2", xs(k)).attr("y1", us(prev)).attr("y2", us(Fk)).attr("stroke", SC.line).attr("stroke-dasharray", "2 2");
        g.append("line").attr("x1", xs(k)).attr("x2", xs(Math.min(k + 1, D.hi))).attr("y1", us(Fk)).attr("y2", us(Fk)).attr("stroke", SC.accent).attr("stroke-width", 2);
        g.append("circle").attr("cx", xs(k)).attr("cy", us(Fk)).attr("r", 2.6).attr("fill", SC.accent);
        prev = Fk;
      }
      g.append("line").attr("x1", xs(D.lo)).attr("x2", xs(0)).attr("y1", us(0)).attr("y2", us(0)).attr("stroke", SC.accent).attr("stroke-width", 2);
    } else SP.curve(g, D.F, xs, us, D.lo, D.hi, { color: SC.accent });
    // recent draws as ticks
    draws.slice(-40).forEach(d => {
      g.append("line").attr("x1", L.x0 - 6).attr("x2", L.x0).attr("y1", us(d.u)).attr("y2", us(d.u)).attr("stroke", SC.good).attr("stroke-opacity", 0.6);
      if (d.x >= D.lo && d.x <= D.hi) g.append("line").attr("x1", xs(d.x)).attr("x2", xs(d.x)).attr("y1", L.y0).attr("y2", L.y0 + 6).attr("stroke", SC.good).attr("stroke-opacity", 0.6);
    });
    // the mapping for the slider U
    const x = SP.invQ(key, u), xc = Math.max(D.lo, Math.min(D.hi, x));
    g.append("line").attr("x1", L.x0).attr("x2", xs(xc)).attr("y1", us(u)).attr("y2", us(u)).attr("stroke", SC.a2).attr("stroke-width", 2);
    g.append("line").attr("x1", xs(xc)).attr("x2", xs(xc)).attr("y1", us(u)).attr("y2", L.y0).attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
    g.append("circle").attr("cx", xs(xc)).attr("cy", L.y0).attr("r", 5).attr("fill", SC.a2);
    g.append("circle").attr("cx", L.x0).attr("cy", us(u)).attr("r", 5).attr("fill", SC.a2);
    SP.label(g, L.x0 + 6, us(u) - 6, `U = ${u.toFixed(3)}`, { color: SC.a2, size: 10.5 });
    SP.label(g, xs(xc) + 4, L.y0 - 8, `x = ${SP.f(x, 3)}`, { color: SC.a2, size: 10.5 });
    // right: histogram of the draws against the target
    const st = SP.invStats(key, draws);
    let top;
    if (D.disc) {
      const pm = d3.range(D.K + 1).map(D.p), emp = st.emp.map(c => c / Math.max(1, draws.length));
      top = Math.max(d3.max(pm), d3.max(emp)) * 1.15;
      const xr = d3.scaleLinear().domain([D.lo, D.hi]).range([Rr.x0, Rr.x1]), yr = d3.scaleLinear().domain([0, top]).range([L.y0, L.y1]);
      SP.axisB(g, xr, L.y0, 6); SP.axisL(g, yr, Rr.x0, 5);
      emp.forEach((p, k) => g.append("rect").attr("x", xr(k - 0.4)).attr("width", xr(k + 0.4) - xr(k - 0.4)).attr("y", yr(p)).attr("height", L.y0 - yr(p)).attr("fill", SC.good).attr("fill-opacity", 0.55));
      pm.forEach((p, k) => g.append("circle").attr("cx", xr(k)).attr("cy", yr(p)).attr("r", 3.5).attr("fill", SC.a2));
      SP.label(g, Rr.x0, L.y1 - 10, "green: proportion of draws · orange: pmf", { size: 10.5 });
    } else {
      const H = SP.hist(draws.map(d => d.x), D.lo, D.hi, 30);
      top = Math.max(d3.max(H.bins, b => b.d), d3.max(d3.range(0, 301).map(i => D.f(D.lo + (D.hi - D.lo) * i / 300)))) * 1.12;
      const xr = d3.scaleLinear().domain([D.lo, D.hi]).range([Rr.x0, Rr.x1]), yr = d3.scaleLinear().domain([0, top]).range([L.y0, L.y1]);
      SP.axisB(g, xr, L.y0, 6); SP.axisL(g, yr, Rr.x0, 5);
      SP.bars(g, H.bins, xr, yr, SC.good, 0.5);
      if (key === "gap") {
        [[D.lo, 0, 0], [0, 1, 0.5], [1, 2, 0], [2, 3, 0.5], [3, D.hi, 0]].forEach(s => g.append("line").attr("x1", xr(s[0])).attr("x2", xr(s[1])).attr("y1", yr(s[2])).attr("y2", yr(s[2])).attr("stroke", SC.a2).attr("stroke-width", 2));
      } else SP.curve(g, D.f, xr, yr, D.lo, D.hi);
      SP.label(g, Rr.x0, L.y1 - 10, `green: histogram of draws · orange: density${H.out ? ` (${H.out} off-scale)` : ""}`, { size: 10.5 });
    }
    out.innerHTML = `<b>${D.name}</b>: ${D.rule}. Slider U = ${u.toFixed(3)} → x = <b>${SP.f(x, 4)}</b>` +
      `<br>${draws.length} draws: sample mean ${SP.f(st.mean, 3)}${isFinite(D.mean) ? ` (true ${D.mean})` : " (the Cauchy has no mean: this number never settles)"} · ` +
      `${st.kind === "TV" ? "total-variation distance to the pmf" : "Kolmogorov distance to F"} <b>${SP.f(st.dist, 4)}</b>` +
      (st.kind === "KS" && draws.length ? ` (95% critical value about ${SP.f(1.358 / Math.sqrt(draws.length), 3)})` : "");
  }
  $("iv-dist").addEventListener("change", e => { key = e.target.value; r = ST.rng(SP.seeds.inv); draws = SP.invDraws(key, r, 200); draw(); });
  $("iv-u").addEventListener("input", e => { u = +e.target.value; $("iv-uv").textContent = u.toFixed(3); draw(); });
  $("iv-one").addEventListener("click", () => { const d = SP.invDraws(key, r, 1)[0]; draws.push(d); u = Math.min(0.999, Math.max(0.001, d.u)); $("iv-u").value = u; $("iv-uv").textContent = u.toFixed(3); draw(); });
  $("iv-many").addEventListener("click", () => { draws = draws.concat(SP.invDraws(key, r, 1000)); draw(); });
  $("iv-reset").addEventListener("click", () => {
    key = "exp"; u = 0.7; $("iv-dist").value = "exp"; $("iv-u").value = 0.7; $("iv-uv").textContent = "0.700";
    r = ST.rng(SP.seeds.inv); draws = SP.invDraws(key, r, 200); draw();
  });
  draw();
})();

/* ─────────────────── 2 · rejection sampling ───────────────────
   Caption audit: target "lap" (N(0, 1) under c·Laplace, c* = √(2e/π) = 1.3155, acceptance 1/c* = 0.7602),
   envelope factor κ = 1, N = 2,000 proposals, seed SP.seeds.rej. The observed acceptance and the
   Kolmogorov distance of the accepted draws are printed by SP.rejRun("lap", 1, 2000, seed).            */
SP.rejT = {
  lap: { name: "N(0, 1) under c × Laplace", lo: -5, hi: 5, f: x => ST.normPdf(x), F: x => ST.normCdf(x),
    q: x => 0.5 * Math.exp(-Math.abs(x)), draw: r => { const e = -Math.log(SP.unit(r)); return r() < 0.5 ? -e : e; },
    cstar: Math.sqrt(2 * Math.E / Math.PI) },
  cau: { name: "N(0, 1) under c × Cauchy", lo: -5, hi: 5, f: x => ST.normPdf(x), F: x => ST.normCdf(x),
    q: x => PV.cauchyPdf(x, 0, 1), draw: r => PV.cauchyDraw(0, 1, r), cstar: Math.sqrt(2 * Math.PI / Math.E) },
  beta: { name: "Beta(5.5, 3.1) under a box", lo: 0, hi: 1, f: x => PV.betaPdf(x, 5.5, 3.1), F: x => PV.betaCdf(x, 5.5, 3.1),
    q: x => (x >= 0 && x <= 1 ? 1 : 0), draw: r => r(), cstar: NaN },
  bump: { name: "two bumps under c × N(0, 2²)", lo: -5, hi: 5,
    f: x => 0.5 * ST.normPdf((x + 1.5) / 0.5) / 0.5 + 0.5 * ST.normPdf((x - 1.5) / 0.5) / 0.5,
    F: x => 0.5 * ST.normCdf((x + 1.5) / 0.5) + 0.5 * ST.normCdf((x - 1.5) / 0.5),
    q: x => ST.normPdf(x / 2) / 2, draw: r => 2 * ST.randn(r), cstar: NaN }
};
["beta", "bump"].forEach(k => {                      // c* = sup f/q, numerically on a fine grid
  const T = SP.rejT[k]; let m = 0;
  for (let i = 1; i < 20000; i++) { const x = T.lo + (T.hi - T.lo) * i / 20000, q = T.q(x); if (q > 0) m = Math.max(m, T.f(x) / q); }
  T.cstar = m;
});
SP.rejRun = function (key, kappa, N, seed) {
  const T = SP.rejT[key], c = kappa * T.cstar, r = ST.rng(seed), pts = [], acc = [];
  for (let i = 0; i < N; i++) {
    const x = T.draw(r), v = r() * c * T.q(x), ok = v <= T.f(x);
    pts.push({ x, v, ok }); if (ok) acc.push(x);
  }
  const ks = acc.length ? PV.ksDist(acc.slice().sort((a, b) => a - b), T.F) : NaN;
  return { c, pts, acc, rate: acc.length / N, ks };
};
(function () {
  const svg = d3.select("#rej-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("rej-readout");
  let key = "lap", kappa = 1, N = 2000, seed = SP.seeds.rej;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), T = SP.rejT[key], R = SP.rejRun(key, kappa, N, seed);
    const x0 = 46, x1 = 404, y0 = 310, y1 = 34;
    const xs = d3.scaleLinear().domain([T.lo, T.hi]).range([x0, x1]);
    let top = 0; for (let i = 0; i <= 300; i++) { const x = T.lo + (T.hi - T.lo) * i / 300; top = Math.max(top, R.c * T.q(x)); }
    const ys = d3.scaleLinear().domain([0, top * 1.08]).range([y0, y1]);
    SP.axisB(g, xs, y0, 6); SP.axisL(g, ys, x0, 5);
    const shown = R.pts.filter(p => p.x >= T.lo && p.x <= T.hi);
    g.append("g").selectAll("circle").data(shown).join("circle").attr("cx", p => xs(p.x)).attr("cy", p => ys(p.v)).attr("r", 1.6)
      .attr("fill", p => (p.ok ? SC.good : SC.bad)).attr("fill-opacity", 0.55);
    if (key === "beta") g.append("line").attr("x1", xs(0)).attr("x2", xs(1)).attr("y1", ys(R.c)).attr("y2", ys(R.c)).attr("stroke", SC.ink).attr("stroke-width", 2);
    else SP.curve(g, x => R.c * T.q(x), xs, ys, T.lo, T.hi, { color: SC.ink });
    SP.curve(g, T.f, xs, ys, T.lo, T.hi, { color: SC.a2, w: 2.4 });
    SP.label(g, x0, y1 - 14, `white: envelope c·q(x), c = ${R.c.toFixed(4)} · orange: target f(x) · green accepted, red rejected`, { size: 10.5 });
    // right: accepted histogram
    const xr = d3.scaleLinear().domain([T.lo, T.hi]).range([446, 664]);
    const H = SP.hist(R.acc, T.lo, T.hi, 28);
    let ft = 0; for (let i = 0; i <= 300; i++) ft = Math.max(ft, T.f(T.lo + (T.hi - T.lo) * i / 300));
    const yr = d3.scaleLinear().domain([0, Math.max(ft, d3.max(H.bins, b => b.d)) * 1.12]).range([y0, y1]);
    SP.axisB(g, xr, y0, 4); SP.axisL(g, yr, 446, 4);
    SP.bars(g, H.bins, xr, yr, SC.good, 0.5);
    SP.curve(g, T.f, xr, yr, T.lo, T.hi);
    SP.label(g, 446, y1 - 14, `${R.acc.length} accepted draws vs f`, { size: 10.5 });
    out.innerHTML = `<b>${T.name}</b>: c* = sup f/q = ${T.cstar.toFixed(4)}, chosen c = ${kappa.toFixed(2)} × c* = <b>${R.c.toFixed(4)}</b>` +
      `<br>acceptance: observed <b>${R.rate.toFixed(4)}</b> (${R.acc.length} of ${N}) against 1/c = <b>${(1 / R.c).toFixed(4)}</b> · ` +
      `proposals per accepted draw ${(N / Math.max(1, R.acc.length)).toFixed(3)} (expected c = ${R.c.toFixed(3)}) · Kolmogorov distance of the accepted draws to F: ${SP.f(R.ks, 4)}` +
      ` (95% critical value about ${SP.f(1.358 / Math.sqrt(Math.max(1, R.acc.length)), 3)})`;
  }
  $("rj-t").addEventListener("change", e => { key = e.target.value; draw(); });
  $("rj-k").addEventListener("input", e => { kappa = +e.target.value; $("rj-kv").textContent = kappa.toFixed(2); draw(); });
  $("rj-n").addEventListener("input", e => { N = +e.target.value; $("rj-nv").textContent = N; draw(); });
  $("rj-seed").addEventListener("click", () => { seed = (seed * 7919 + 13) % 100003; draw(); });
  $("rj-reset").addEventListener("click", () => {
    key = "lap"; kappa = 1; N = 2000; seed = SP.seeds.rej;
    $("rj-t").value = "lap"; $("rj-k").value = 1; $("rj-kv").textContent = "1.00"; $("rj-n").value = 2000; $("rj-nv").textContent = "2000"; draw();
  });
  draw();
})();

/* ─────────────────── 3 · Monte Carlo against a grid, in d dimensions ───────────────────
   Integrand g(x) = Π h(xⱼ) with h(x) = eˣ/(e − 1) on [0, 1]ᵈ, so ∫ g = 1 exactly. Per factor
   Var h(U) = v₁ = 0.0819767, so σ² = (1 + v₁)ᵈ − 1. The midpoint rule with m points per axis uses n = mᵈ
   points and returns Qₘᵈ with Qₘ = s/sinh(s), s = 1/(2m), exactly (the integrand factorises).
   Caption audit: d = 6, 100,000 draws, seed SP.seeds.mc; SP.mcRun(6, 1e5, seed) prints the final
   estimate and SE, and SP.mcGrid(6) the grid errors.                                                  */
SP.mcV1 = ((Math.E * Math.E - 1) / 2) / ((Math.E - 1) * (Math.E - 1)) - 1;
SP.mcSig2 = d => Math.pow(1 + SP.mcV1, d) - 1;
SP.mcQ = m => { const s = 1 / (2 * m); return s / Math.sinh(s); };
SP.mcGrid = function (d, nmax) {                 // every m up to 30, then log-spaced (8 per decade) up to the cap
  const cap = nmax || 1e7, mmax = Math.floor(Math.pow(cap, 1 / d) + 1e-9), ms = [];
  for (let m = 1; m <= Math.min(30, mmax); m++) ms.push(m);
  for (let k = 12; ; k++) { const m = Math.round(Math.pow(10, k / 8)); if (m > mmax) break; if (m > ms[ms.length - 1]) ms.push(m); }
  if (mmax > ms[ms.length - 1]) ms.push(mmax);
  return ms.map(m => ({ m, n: Math.pow(m, d), err: Math.abs(Math.pow(SP.mcQ(m), d) - 1) }));
};
SP.mcRun = function (d, N, seed) {
  const r = ST.rng(seed), c = 1 / (Math.E - 1), marks = [];
  let mean = 0, m2 = 0, next = 10;
  for (let i = 1; i <= N; i++) {
    let v = 1;
    for (let j = 0; j < d; j++) v *= Math.exp(r()) * c;
    const dl = v - mean; mean += dl / i; m2 += dl * (v - mean);
    if (i >= next || i === N) { marks.push({ n: i, est: mean, se: Math.sqrt(m2 / (i - 1) / i) }); next = Math.ceil(next * 1.08); }
  }
  return marks;
};
(function () {
  const svg = d3.select("#mc-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("mc-readout"), N = 100000;
  let d = 6, seed = SP.seeds.mc;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), M = SP.mcRun(d, N, seed), G = SP.mcGrid(d), sig = Math.sqrt(SP.mcSig2(d));
    const y0 = 312, y1 = 34;
    // left: running estimate ± 2 SE
    const xl = d3.scaleLog().domain([10, N]).range([48, 318]);
    const half = Math.max(4 * sig / Math.sqrt(30), 0.02);
    const yl = d3.scaleLinear().domain([1 - half, 1 + half]).range([y0, y1]).clamp(true);
    SP.axisB(g, xl, y0, 4, "~s"); SP.axisL(g, yl, 48, 5);
    const band = M.filter(m => m.n >= 10);
    g.append("path").attr("d", d3.area().x(m => xl(m.n)).y0(m => yl(m.est - 2 * m.se)).y1(m => yl(m.est + 2 * m.se))(band)).attr("fill", SC.accent).attr("fill-opacity", 0.18);
    g.append("path").attr("d", d3.line().x(m => xl(m.n)).y(m => yl(m.est))(band)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.8);
    g.append("line").attr("x1", 48).attr("x2", 318).attr("y1", yl(1)).attr("y2", yl(1)).attr("stroke", SC.a2).attr("stroke-dasharray", "5 3");
    SP.label(g, 48, y1 - 14, `running estimate ± 2 SE, d = ${d} (true value 1, dashed)`, { size: 10.5 });
    SP.label(g, 318, y0 + 30, "n (log scale)", { anchor: "end" });
    // right: error against n on log–log axes
    const xr = d3.scaleLog().domain([1, 1e7]).range([392, 664]), yr = d3.scaleLog().domain([1e-5, 1]).range([y0, y1]).clamp(true);
    SP.axisB(g, xr, y0, 4, "~s"); SP.axisL(g, yr, 392, 5, "~e");
    const ln = [1, 1e7].map(n => [n, sig / Math.sqrt(n)]);
    g.append("path").attr("d", d3.line().x(p => xr(p[0])).y(p => yr(p[1]))(ln)).attr("stroke", SC.accent).attr("stroke-width", 2).attr("fill", "none");
    g.append("path").attr("d", d3.line().x(m => xr(m.n)).y(m => yr(Math.max(1e-5, Math.abs(m.est - 1))))(band)).attr("stroke", SC.accent).attr("stroke-opacity", 0.45).attr("fill", "none");
    g.append("path").attr("d", d3.line().x(p => xr(p.n)).y(p => yr(Math.max(1e-5, p.err)))(G)).attr("stroke", SC.a2).attr("stroke-width", 1.5).attr("fill", "none");
    G.forEach(p => g.append("circle").attr("cx", xr(p.n)).attr("cy", yr(Math.max(1e-5, p.err))).attr("r", 3.2).attr("fill", SC.a2));
    SP.label(g, 392, y1 - 14, "|error| vs n: blue MC (line = σ/√n), orange grid mᵈ", { size: 10.5 });
    SP.label(g, 664, y0 + 30, "n = function evaluations", { anchor: "end" });
    // readout
    const last = M[M.length - 1];
    const g5 = G.filter(p => p.n <= N), gb = g5[g5.length - 1];
    let cross = null; for (const p of G) if (p.n > 1 && sig / Math.sqrt(p.n) < p.err) { cross = p; break; }
    out.innerHTML = `d = <b>${d}</b>: σ² = Var g(U) = 1.08198<sup>${d}</sup> − 1 = <b>${SP.f(SP.mcSig2(d), 4)}</b>. ` +
      `After n = 10⁵ draws: estimate <b>${last.est.toFixed(4)}</b> ± 2 SE = ±${(2 * last.se).toFixed(4)} (true 1; error ${SP.f(last.est - 1, 4)})` +
      `<br>grid with n ≤ 10⁵: m = ${gb.m} points per axis, n = ${gb.n.toLocaleString()}, error <b>${SP.f(gb.err, 4)}</b> · MC RMS error at that n: ${SP.f(sig / Math.sqrt(gb.n), 4)}` +
      `<br>grid error falls like n<sup>−2/${d}</sup>, MC like n<sup>−1/2</sup>: ` +
      (cross ? `MC is already more accurate (in RMS) at the grid size n = ${cross.n.toLocaleString()} (m = ${cross.m})` : "the grid stays more accurate over the whole range plotted");
  }
  $("mc-d").addEventListener("input", e => { d = +e.target.value; $("mc-dv").textContent = d; draw(); });
  $("mc-seed").addEventListener("click", () => { seed = (seed * 7919 + 17) % 100003; draw(); });
  $("mc-reset").addEventListener("click", () => { d = 6; seed = SP.seeds.mc; $("mc-d").value = 6; $("mc-dv").textContent = "6"; draw(); });
  draw();
})();

/* ─────────────────── 4 · variance-reduction race ───────────────────
   Five estimators of I = ∫₀¹ g(x) dx at the same budget of N evaluations of g: plain Monte Carlo,
   antithetic pairs (U, 1 − U), a control variate (U itself, mean ½, β estimated from the same sample),
   proportional stratification into K equal strata, and importance sampling from q(x) = 2(1 + x)/3
   (drawn by inverse transform, x = √(1 + 3u) − 1). "Per-evaluation variance" = N × Var(estimator).
   Caption audit: g = eˣ, N = 1,000, K = 10, R = 200 replicates, seed SP.seeds.vr: SP.vrTheory("exp", 10)
   gives 0.24204, 0.0078250, 0.0039402, 0.0026594, 0.026908; SP.vrRun prints the empirical values.      */
SP.vrG = {
  exp: { name: "eˣ", g: x => Math.exp(x), I: Math.E - 1 },
  sin: { name: "(π/2)·sin(πx)", g: x => Math.PI / 2 * Math.sin(Math.PI * x), I: 1 }
};
SP.vrQ = { q: x => 2 * (1 + x) / 3, draw: u => Math.sqrt(1 + 3 * u) - 1 };
SP.vrTheory = function (key, K) {
  const G = SP.vrG[key], g = G.g, I = G.I, S = (f, a, b) => PV.simpson(f, a, b, 4000);
  const s2 = S(x => g(x) * g(x), 0, 1) - I * I;
  const cAnti = S(x => g(x) * g(1 - x), 0, 1) - I * I;
  const cU = S(x => x * g(x), 0, 1) - 0.5 * I, rho2 = cU * cU / (s2 / 12);
  let st = 0;
  for (let k = 0; k < K; k++) { const a = k / K, b = (k + 1) / K, m = S(g, a, b) * K, m2 = S(x => g(x) * g(x), a, b) * K; st += (m2 - m * m) / K; }
  const is = S(x => g(x) * g(x) / SP.vrQ.q(x), 0, 1) - I * I;
  return { plain: s2, anti: s2 + cAnti, cv: s2 * (1 - rho2), strat: st, is, beta: cU * 12, corrAnti: cAnti / s2 };
};
SP.vrMethods = [["plain", "plain Monte Carlo"], ["anti", "antithetic pairs"], ["cv", "control variate U"], ["strat", "stratified (K strata)"], ["is", "importance, q ∝ 1 + x"]];
SP.vrOnce = function (key, N, K, r) {
  const g = SP.vrG[key].g, out = {};
  let s = 0; for (let i = 0; i < N; i++) s += g(r()); out.plain = s / N;
  s = 0; for (let i = 0; i < N / 2; i++) { const u = r(); s += (g(u) + g(1 - u)) / 2; } out.anti = s / (N / 2);
  let sy = 0, su = 0, syu = 0, suu = 0;
  for (let i = 0; i < N; i++) { const u = r(), y = g(u); sy += y; su += u; syu += y * u; suu += u * u; }
  const my = sy / N, mu = su / N, b = (syu / N - my * mu) / (suu / N - mu * mu);
  out.cv = my - b * (mu - 0.5);
  const nk = Math.max(1, Math.round(N / K)); s = 0;
  for (let k = 0; k < K; k++) { let t = 0; for (let j = 0; j < nk; j++) t += g((k + r()) / K); s += t / nk; }
  out.strat = s / K;
  s = 0; for (let i = 0; i < N; i++) { const x = SP.vrQ.draw(r()); s += g(x) / SP.vrQ.q(x); } out.is = s / N;
  return out;
};
SP.vrRun = function (key, N, K, R, seed) {
  const r = ST.rng(seed), reps = [];
  for (let i = 0; i < R; i++) reps.push(SP.vrOnce(key, N, K, r));
  const emp = {};
  SP.vrMethods.forEach(([m]) => { const a = reps.map(o => o[m]); emp[m] = ST.variance(a) * N; });
  return { reps, emp };
};
(function () {
  const svg = d3.select("#vr-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("vr-readout"), R = 200;
  let key = "exp", N = 1000, K = 10, seed = SP.seeds.vr;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), G = SP.vrG[key], T = SP.vrTheory(key, K), E = SP.vrRun(key, N, K, R, seed);
    const cols = [SC.muted, SC.accent, SC.good, SC.violet, SC.a2];
    const sdPlain = Math.sqrt(T.plain / N);
    const xs = d3.scaleLinear().domain([G.I - 4.2 * sdPlain, G.I + 4.2 * sdPlain]).range([150, 420]);
    const yb = i => 52 + i * 58;
    SP.axisB(g, xs, 330, 5);
    g.append("line").attr("x1", xs(G.I)).attr("x2", xs(G.I)).attr("y1", 30).attr("y2", 330).attr("stroke", SC.a2).attr("stroke-dasharray", "4 3");
    SP.label(g, 150, 22, `${R} replicate estimates of ∫₀¹ ${G.name} dx = ${G.I.toFixed(5)}, each from N = ${N} evaluations`, { size: 10.5 });
    const jr = ST.rng(99);
    SP.vrMethods.forEach(([m, name], i) => {
      SP.label(g, 8, yb(i) + 4, name, { color: SC.ink, size: 10.5 });
      const sd = Math.sqrt(T[m] / N);
      g.append("rect").attr("x", xs(Math.max(xs.domain()[0], G.I - 2 * sd))).attr("width", Math.max(1, xs(Math.min(xs.domain()[1], G.I + 2 * sd)) - xs(Math.max(xs.domain()[0], G.I - 2 * sd))))
        .attr("y", yb(i) - 16).attr("height", 32).attr("fill", cols[i]).attr("fill-opacity", 0.12);
      E.reps.forEach(o => { const v = o[m]; if (v >= xs.domain()[0] && v <= xs.domain()[1]) g.append("circle").attr("cx", xs(v)).attr("cy", yb(i) + (jr() - 0.5) * 26).attr("r", 1.8).attr("fill", cols[i]).attr("fill-opacity", 0.7); });
    });
    // right: reduction factors, log scale
    const fx = d3.scaleLog().domain([0.1, 3e4]).range([458, 664]).clamp(true);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,330)").call(d3.axisBottom(fx).ticks(4, "~s"));
    g.append("line").attr("x1", fx(1)).attr("x2", fx(1)).attr("y1", 30).attr("y2", 330).attr("stroke", SC.line);
    SP.label(g, 458, 22, "reduction factor (log): bar exact, ◆ exact plain ÷ simulated", { size: 10.5 });
    SP.vrMethods.forEach(([m], i) => {
      const th = T.plain / T[m], em = T.plain / E.emp[m];
      const a = Math.min(fx(1), fx(th)), b = Math.max(fx(1), fx(th));
      g.append("rect").attr("x", a).attr("width", Math.max(1, b - a)).attr("y", yb(i) - 9).attr("height", 18).attr("fill", cols[i]).attr("fill-opacity", 0.55);
      g.append("path").attr("d", d3.symbol().type(d3.symbolDiamond).size(46)()).attr("transform", `translate(${fx(em)},${yb(i)})`).attr("fill", SC.ink);
      SP.label(g, Math.min(b + 4, 636), yb(i) - 12, `×${SP.f(th, th < 10 ? 2 : 0)}`, { size: 10, color: SC.ink });
    });
    SP.label(g, 664, 352, "plain variance ÷ method variance, equal N", { anchor: "end", size: 10 });
    const row = m => `${SP.f(T[m], 5)} / ${SP.f(E.emp[m], 5)}`;
    out.innerHTML = `per-evaluation variance N·Var(estimate), theory / simulated (R = ${R}): ` +
      SP.vrMethods.map(([m, name]) => `${name} <b>${row(m)}</b>`).join(" · ") +
      `<br>antithetic correlation Corr(g(U), g(1 − U)) = <b>${T.corrAnti.toFixed(4)}</b> · optimal control coefficient β* = Cov(g(U), U)/Var U = <b>${T.beta.toFixed(4)}</b> · strata K = ${K}` +
      (key === "sin" ? `<br>symmetric integrand: g(U) = g(1 − U), so the antithetic pair is one evaluation paid for twice (factor ½), and Cov(g(U), U) = 0 makes the control variate useless` : "");
  }
  $("vr-g").addEventListener("change", e => { key = e.target.value; draw(); });
  $("vr-n").addEventListener("input", e => { N = +e.target.value; $("vr-nv").textContent = N; draw(); });
  $("vr-k").addEventListener("change", e => { K = +e.target.value; draw(); });
  $("vr-seed").addEventListener("click", () => { seed = (seed * 7919 + 19) % 100003; draw(); });
  $("vr-reset").addEventListener("click", () => {
    key = "exp"; N = 1000; K = 10; seed = SP.seeds.vr;
    $("vr-g").value = "exp"; $("vr-n").value = 1000; $("vr-nv").textContent = "1000"; $("vr-k").value = "10"; draw();
  });
  draw();
})();

/* ─────────────────── 5 · the Poisson process ───────────────────
   Timeline and counting staircase on [0, 40]; the right panel uses a long run (T = 4,000) for gap
   histograms. Modes: one stream; superposition with a second stream of rate 0.5; thinning with keep-
   probability p; the inspection paradox (2,000 uniform inspection times in the long run).
   Caption audit: mode "insp", λ = 1, seed SP.seeds.pp: SP.ppInspect(1, 4000, 2000, seed) prints the mean
   ordinary gap, the mean covering gap (theory 2/λ = 2) and the mean wait (theory 1/λ = 1).              */
SP.ppInspect = function (lam, Tl, R, seed) {
  const r = ST.rng(seed), a = PV.poisProcess(lam, Tl, r), cover = [], wait = [], age = [];
  const gaps = a.map((t, i) => (i ? t - a[i - 1] : t));
  for (let i = 0; i < R; i++) {
    const t = 0.05 * Tl + r() * 0.9 * Tl;                // keep away from the ends of the run
    let lo = 0, hi = a.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] > t) hi = m; else lo = m + 1; }   // first arrival after t
    cover.push(a[lo] - (lo ? a[lo - 1] : 0)); wait.push(a[lo] - t); age.push(t - (lo ? a[lo - 1] : 0));
  }
  const mean = v => v.reduce((x, y) => x + y, 0) / v.length;
  return { gaps, cover, wait, age, mGap: mean(gaps), mCover: mean(cover), mWait: mean(wait), mAge: mean(age), n: a.length };
};
SP.ppThinCorr = function (lam, p, Tl, seed) {        // counts of kept and dropped points in unit windows
  const r = ST.rng(seed), a = PV.poisProcess(lam, Tl, r), W = Math.floor(Tl), k = new Array(W).fill(0), d = new Array(W).fill(0);
  a.forEach(t => { const w = Math.min(W - 1, Math.floor(t)); if (r() < p) k[w]++; else d[w]++; });
  return { corr: ST.corr(k, d), meanKept: ST.mean(k), varKept: ST.variance(k) };
};
(function () {
  const svg = d3.select("#pp-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("pp-readout"), T = 40, TL = 4000, LAM2 = 0.5;
  let mode = "insp", lam = 1, p = 0.3, seed = SP.seeds.pp;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), r = ST.rng(seed);
    const xs = d3.scaleLinear().domain([0, T]).range([46, 400]);
    // ── streams on [0, T]
    let A = PV.poisProcess(lam, T, r), B = [], keep = [];
    if (mode === "super") B = PV.poisProcess(LAM2, T, r);
    if (mode === "thin") keep = A.map(() => r() < p);
    const rate = mode === "super" ? lam + LAM2 : mode === "thin" ? p * lam : lam;
    let pts = A.map((t, i) => ({ t, c: mode === "thin" ? (keep[i] ? SC.good : SC.bad) : SC.accent, on: mode !== "thin" || keep[i] }));
    if (mode === "super") pts = pts.concat(B.map(t => ({ t, c: SC.a2, on: true })));
    pts.sort((u, v) => u.t - v.t);
    g.append("line").attr("x1", 46).attr("x2", 400).attr("y1", 52).attr("y2", 52).attr("stroke", SC.line);
    pts.forEach(q => g.append("line").attr("x1", xs(q.t)).attr("x2", xs(q.t)).attr("y1", 40).attr("y2", 64).attr("stroke", q.c).attr("stroke-width", 2).attr("stroke-opacity", q.on ? 1 : 0.45));
    SP.label(g, 46, 30, mode === "super" ? `blue: rate λ = ${lam.toFixed(2)} · orange: rate 0.5 · merged rate ${rate.toFixed(2)}` :
      mode === "thin" ? `each point kept (green) with probability p = ${p.toFixed(2)}, dropped (red) otherwise` :
        `arrivals at rate λ = ${lam.toFixed(2)} on [0, ${T}]: gaps are independent Exp(λ)`, { size: 10.5 });
    // ── staircase of the counted process
    const counted = pts.filter(q => q.on).map(q => q.t);
    const ymax = Math.max(rate * T + 3 * Math.sqrt(rate * T), counted.length) + 2;
    const ys = d3.scaleLinear().domain([0, ymax]).range([330, 90]);
    SP.axisB(g, xs, 330, 8); SP.axisL(g, ys, 46, 5);
    const band = d3.range(0, 101).map(i => T * i / 100);
    g.append("path").attr("d", d3.area().x(t => xs(t)).y0(t => ys(Math.max(0, rate * t - 2 * Math.sqrt(rate * t)))).y1(t => ys(rate * t + 2 * Math.sqrt(rate * t)))(band)).attr("fill", SC.accent).attr("fill-opacity", 0.1);
    g.append("line").attr("x1", xs(0)).attr("x2", xs(T)).attr("y1", ys(0)).attr("y2", ys(rate * T)).attr("stroke", SC.a2).attr("stroke-dasharray", "5 3");
    const st = [[0, 0]]; counted.forEach((t, i) => { st.push([t, i]); st.push([t, i + 1]); }); st.push([T, counted.length]);
    g.append("path").attr("d", d3.line().x(q => xs(q[0])).y(q => ys(q[1]))(st)).attr("fill", "none").attr("stroke", mode === "thin" ? SC.good : SC.ink).attr("stroke-width", 1.8);
    SP.label(g, 52, 104, `N(t), with mean ${rate.toFixed(2)}·t (dashed) ± 2√(mean)`, { size: 10.5 });
    SP.label(g, 400, 360, "t", { anchor: "end" });
    // ── right: gap histogram (long run) or inspection paradox
    const xr = d3.scaleLinear().domain([0, 6 / rate]).range([446, 664]);
    let info = "";
    if (mode === "insp") {
      const I = SP.ppInspect(lam, TL, 2000, seed);
      const H = SP.hist(I.cover, 0, 6 / rate, 24), H0 = SP.hist(I.gaps, 0, 6 / rate, 24);
      const yr = d3.scaleLinear().domain([0, Math.max(rate, d3.max(H.bins, b => b.d)) * 1.1]).range([330, 90]);
      SP.axisB(g, xr, 330, 4); SP.axisL(g, yr, 446, 4);
      SP.bars(g, H.bins, xr, yr, SC.violet, 0.5);
      SP.curve(g, x => lam * lam * x * Math.exp(-lam * x), xr, yr, 0, 6 / rate, { color: SC.violet });
      SP.curve(g, x => lam * Math.exp(-lam * x), xr, yr, 0, 6 / rate, { color: SC.a2, dash: "5 3" });
      SP.label(g, 446, 80, "violet: gap covering a random time vs λ²x·e^(−λx)", { size: 10 });
      SP.label(g, 446, 94, "dashed: a typical gap, Exp(λ)", { size: 10 });
      info = `inspection paradox (long run, ${I.n} arrivals, 2,000 random inspection times): mean gap <b>${I.mGap.toFixed(3)}</b> (1/λ = ${(1 / lam).toFixed(3)}) · ` +
        `mean length of the gap you land in <b>${I.mCover.toFixed(3)}</b> (2/λ = ${(2 / lam).toFixed(3)}) · mean wait for the next arrival <b>${I.mWait.toFixed(3)}</b> (1/λ) · mean time since the last ${I.mAge.toFixed(3)} (1/λ)`;
    } else {
      const r2 = ST.rng(seed + 1000);
      let L = PV.poisProcess(lam, TL, r2);
      if (mode === "super") L = L.concat(PV.poisProcess(LAM2, TL, r2)).sort((u, v) => u - v);
      if (mode === "thin") L = L.filter(() => r2() < p);
      const gaps = L.map((t, i) => (i ? t - L[i - 1] : t));
      const H = SP.hist(gaps, 0, 6 / rate, 24);
      const yr = d3.scaleLinear().domain([0, Math.max(rate, d3.max(H.bins, b => b.d)) * 1.1]).range([330, 90]);
      SP.axisB(g, xr, 330, 4); SP.axisL(g, yr, 446, 4);
      SP.bars(g, H.bins, xr, yr, SC.accent, 0.45);
      SP.curve(g, x => rate * Math.exp(-rate * x), xr, yr, 0, 6 / rate);
      SP.label(g, 446, 80, `gaps of a long run (${gaps.length}) vs Exp(${rate.toFixed(2)})`, { size: 10 });
      const mg = gaps.reduce((x, y) => x + y, 0) / gaps.length;
      info = `long run of length ${TL}: ${gaps.length} points, rate ${(gaps.length / TL).toFixed(4)} (theory ${rate.toFixed(4)}), mean gap ${mg.toFixed(4)} (theory ${(1 / rate).toFixed(4)})`;
      if (mode === "super") { const fromA = PV.poisProcess(lam, T, ST.rng(seed)).length; info += ` · on [0, ${T}] ${fromA} of ${counted.length} merged arrivals came from the blue stream (theory share λ/(λ + 0.5) = ${(lam / (lam + LAM2)).toFixed(3)})`; }
      if (mode === "thin") { const C = SP.ppThinCorr(lam, p, TL, seed + 2000); info += ` · kept vs dropped counts in ${TL} unit windows: correlation <b>${C.corr.toFixed(4)}</b> (independent processes: 0); kept count per window mean ${C.meanKept.toFixed(3)}, variance ${C.varKept.toFixed(3)} (Poisson: both ${(p * lam).toFixed(3)})`; }
    }
    out.innerHTML = `on [0, ${T}]: N(${T}) = <b>${counted.length}</b> against mean ${(rate * T).toFixed(1)} and sd ${Math.sqrt(rate * T).toFixed(2)} (Poisson(${(rate * T).toFixed(1)}))<br>` + info;
  }
  $("pp-mode").addEventListener("change", e => { mode = e.target.value; draw(); });
  $("pp-lam").addEventListener("input", e => { lam = +e.target.value; $("pp-lamv").textContent = lam.toFixed(2); draw(); });
  $("pp-p").addEventListener("input", e => { p = +e.target.value; $("pp-pv").textContent = p.toFixed(2); draw(); });
  $("pp-seed").addEventListener("click", () => { seed = (seed * 7919 + 23) % 100003; draw(); });
  $("pp-reset").addEventListener("click", () => {
    mode = "insp"; lam = 1; p = 0.3; seed = SP.seeds.pp;
    $("pp-mode").value = "insp"; $("pp-lam").value = 1; $("pp-lamv").textContent = "1.00"; $("pp-p").value = 0.3; $("pp-pv").textContent = "0.30"; draw();
  });
  draw();
})();

/* ─────────────────── 6 · a three-state Markov chain ───────────────────
   Caption audit: preset "device" (P = [[.64 .32 .04] [.40 .50 .10] [.25 .50 .25]], π = (25, 20, 4)/49,
   eigenvalues 1, 0.3, 0.09), start state 2, n = 500 steps, seed SP.seeds.mk: SP.mkWalk prints the
   empirical frequencies; SP.mkTV the distance of μₙ to π for n = 1…5 (0.2602, 0.0877, 0.0272, …).       */
SP.mkPresets = {
  device: [[0.64, 0.32, 0.04], [0.40, 0.50, 0.10], [0.25, 0.50, 0.25]],
  sticky: [[0.96, 0.02, 0.02], [0.02, 0.96, 0.02], [0.02, 0.02, 0.96]],
  lazycycle: [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]],
  cycle: [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
  absorb: [[1, 0, 0], [0.3, 0.4, 0.3], [0, 0, 1]]
};
SP.mkStep = (mu, P) => [0, 1, 2].map(j => mu[0] * P[0][j] + mu[1] * P[1][j] + mu[2] * P[2][j]);
SP.mkEig = function (P) {                         // 1 is an eigenvalue; the other two solve x² − (tr − 1)x + det = 0
  const tr = P[0][0] + P[1][1] + P[2][2];
  const det = P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1]) - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0]) + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]);
  const s = tr - 1, disc = s * s - 4 * det;
  if (disc >= 0) { const a = (s + Math.sqrt(disc)) / 2, b = (s - Math.sqrt(disc)) / 2; return { l2: a, l3: b, mod: Math.max(Math.abs(a), Math.abs(b)), complex: false }; }
  return { re: s / 2, im: Math.sqrt(-disc) / 2, mod: Math.sqrt(det), complex: true };
};
SP.mkLimit = function (P, s0) {                   // Cesàro average of μ₀Pⁿ: the long-run distribution from s0
  let mu = [0, 0, 0]; mu[s0] = 1; const acc = [0, 0, 0], M = 4000;
  for (let n = 0; n < M; n++) { for (let j = 0; j < 3; j++) acc[j] += mu[j] / M; mu = SP.mkStep(mu, P); }
  return acc;
};
SP.mkTV = function (P, s0, target, nmax) {
  let mu = [0, 0, 0]; mu[s0] = 1; const out = [];
  for (let n = 0; n <= nmax; n++) { out.push(0.5 * mu.reduce((s, v, i) => s + Math.abs(v - target[i]), 0)); mu = SP.mkStep(mu, P); }
  return out;
};
SP.mkWalk = function (P, s0, n, seed) {
  const r = ST.rng(seed), path = [s0], cnt = [0, 0, 0];
  let s = s0;
  for (let i = 0; i < n; i++) { s = PV.catDraw(P[s], r); path.push(s); cnt[s]++; }
  return { path, freq: cnt.map(c => c / Math.max(1, n)) };
};
(function () {
  const svg = d3.select("#markov-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("markov-readout");
  let P = SP.mkPresets.device.map(r => r.slice()), s0 = 2, n = 500, seed = SP.seeds.mk, timer = null;
  const ids = [0, 1, 2].map(i => [0, 1, 2].map(j => `mk-p${i}${j}`));
  function syncInputs() { ids.forEach((row, i) => row.forEach((id, j) => { $(id).value = P[i][j].toFixed(2); })); }
  function readRow(i) {
    const v = ids[i].map(id => Math.max(0, +$(id).value || 0)), s = v.reduce((a, b) => a + b, 0);
    P[i] = s > 0 ? v.map(x => x / s) : [0, 1, 2].map(j => (j === i ? 1 : 0));
  }
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const pi = PV.stationary(P), lim = SP.mkLimit(P, s0), E = SP.mkEig(P), W = SP.mkWalk(P, s0, n, seed);
    const target = pi || lim;
    // diagram
    const node = [[110, 70], [40, 230], [180, 230]], names = ["0", "1", "2"];
    g.append("defs").append("marker").attr("id", "mk-arrow").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,0L10,5L0,10z").attr("fill", SC.muted);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const pij = P[i][j]; if (pij <= 0) continue;
      const [x1, y1] = node[i], [x2, y2] = node[j];
      if (i === j) {
        const dx = x1 - 110, dy = y1 - 170, L = Math.hypot(dx, dy) || 1, cx = x1 + 30 * dx / L, cy = y1 + 30 * dy / L;
        g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 13).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 0.6 + 4 * pij);
        SP.label(g, cx + 12 * dx / L, cy + 12 * dy / L + 4, pij.toFixed(2), { anchor: "middle", size: 9.5, color: SC.ink });
      } else {
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, nx = -(y2 - y1), ny = x2 - x1, L = Math.hypot(nx, ny);
        const cx = mx + 18 * nx / L, cy = my + 18 * ny / L;
        const dl = Math.hypot(x2 - cx, y2 - cy), ex = x2 - 20 * (x2 - cx) / dl, ey = y2 - 20 * (y2 - cy) / dl;
        g.append("path").attr("d", `M${x1},${y1}Q${cx},${cy} ${ex},${ey}`).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 0.6 + 4 * pij).attr("marker-end", "url(#mk-arrow)");
        SP.label(g, cx + 6 * nx / L, cy + 6 * ny / L + 3, pij.toFixed(2), { anchor: "middle", size: 9.5, color: SC.ink });
      }
    }
    node.forEach(([x, y], i) => {
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 17).attr("fill", SC.panel2).attr("stroke", i === W.path[W.path.length - 1] ? SC.a2 : SC.accent).attr("stroke-width", i === W.path[W.path.length - 1] ? 3 : 1.5);
      SP.label(g, x, y + 4, names[i], { anchor: "middle", color: SC.ink, weight: 700 });
    });
    SP.label(g, 20, 300, `walker after ${n} steps: state ${W.path[W.path.length - 1]} (orange ring); start ${s0}`, { size: 10.5 });
    // bars: empirical vs target
    const xb = d3.scaleBand().domain([0, 1, 2]).range([262, 452]).padding(0.25), yb = d3.scaleLinear().domain([0, 1]).range([170, 34]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,170)").call(d3.axisBottom(xb).tickFormat(d => `state ${d}`));
    SP.axisL(g, yb, 262, 4);
    [0, 1, 2].forEach(i => {
      g.append("rect").attr("x", xb(i)).attr("width", xb.bandwidth()).attr("y", yb(W.freq[i])).attr("height", 170 - yb(W.freq[i])).attr("fill", SC.good).attr("fill-opacity", 0.55);
      g.append("line").attr("x1", xb(i) - 4).attr("x2", xb(i) + xb.bandwidth() + 4).attr("y1", yb(target[i])).attr("y2", yb(target[i])).attr("stroke", SC.a2).attr("stroke-width", 2.5);
    });
    SP.label(g, 262, 22, `green: share of time in each state · orange: ${pi ? "π" : "long-run limit from this start"}`, { size: 10.5 });
    // TV vs n
    const NM = 40, tv = SP.mkTV(P, s0, target, NM);
    const xt = d3.scaleLinear().domain([0, NM]).range([262, 452]), yt = d3.scaleLog().domain([1e-6, 1]).range([340, 210]).clamp(true);
    SP.axisB(g, xt, 340, 5); SP.axisL(g, yt, 262, 3, "~e");
    g.append("path").attr("d", d3.line().x((v, k) => xt(k)).y(v => yt(Math.max(1e-6, v)))(tv)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    if (E.mod > 0 && E.mod < 1) g.append("path").attr("d", d3.line().x((v, k) => xt(k)).y((v, k) => yt(Math.max(1e-6, tv[0] * Math.pow(E.mod, k))))(tv)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-dasharray", "5 3");
    SP.label(g, 262, 202, "‖μₙ − π‖ (TV) vs n, log scale; dashed |λ₂|ⁿ", { size: 10.5 });
    // right text panel
    const db = pi ? [[0, 1], [0, 2], [1, 2]].every(([i, j]) => Math.abs(pi[i] * P[i][j] - pi[j] * P[j][i]) < 1e-9) : false;
    const tx = 474;
    g.append("rect").attr("x", tx - 8).attr("y", 14).attr("width", 214).attr("height", 330).attr("fill", SC.panel2).attr("stroke", SC.line).attr("rx", 6);
    const lines = [
      ["transition matrix P (rows sum to 1)", SC.ink, 700],
      ...P.map((row, i) => [`  ${i}: ${row.map(v => v.toFixed(3)).join("  ")}`, SC.ink, 400]),
      [pi ? `π = (${pi.map(v => v.toFixed(4)).join(", ")})` : "π not unique: several closed classes", SC.a2, 400],
      [E.complex ? `λ₂, λ₃ = ${E.re.toFixed(3)} ± ${E.im.toFixed(3)}i` : `λ₂ = ${E.l2.toFixed(4)}, λ₃ = ${E.l3.toFixed(4)}`, SC.ink, 400],
      [`|λ₂| = ${E.mod.toFixed(4)}${E.mod >= 1 - 1e-9 ? " : no geometric convergence" : ""}`, SC.ink, 400],
      [`detailed balance: ${pi ? (db ? "holds (reversible)" : "fails (not reversible)") : "—"}`, db ? SC.good : SC.muted, 400],
      [pi ? `mean return times 1/πᵢ: ${pi.map(v => (v > 0 ? (1 / v).toFixed(2) : "∞")).join(", ")}` : `limit from start ${s0}: ${lim.map(v => v.toFixed(3)).join(", ")}`, SC.ink, 400],
      [`time shares after ${n} steps: ${W.freq.map(v => v.toFixed(3)).join(", ")}`, SC.good, 400]
    ];
    lines.forEach((l, k) => SP.label(g, tx, 36 + k * 22, l[0], { color: l[1], weight: l[2], size: 10.5 }));
    out.innerHTML = `${pi ? `stationary distribution π = (<b>${pi.map(v => v.toFixed(4)).join("</b>, <b>")}</b>)` : "no unique stationary distribution"} · ` +
      `second-largest eigenvalue modulus |λ₂| = <b>${E.mod.toFixed(4)}</b> · distance to ${pi ? "π" : "the limit"} after 1, 2, 3, 4, 5 steps from state ${s0}: ${tv.slice(1, 6).map(v => SP.f(v, 4)).join(", ")}` +
      `<br>walker: ${n} steps (seed ${seed}), time shares ${W.freq.map(v => v.toFixed(3)).join(", ")}` +
      (pi ? ` · largest gap to π ${Math.max(...W.freq.map((v, i) => Math.abs(v - pi[i]))).toFixed(3)}` : "");
  }
  function stop() { if (timer) { timer.stop(); timer = null; $("mk-play").textContent = "Play"; } }
  ids.forEach((row, i) => row.forEach(id => $(id).addEventListener("change", () => { readRow(i); syncInputs(); $("mk-pre").value = "custom"; draw(); })));
  $("mk-pre").addEventListener("change", e => { stop(); if (SP.mkPresets[e.target.value]) { P = SP.mkPresets[e.target.value].map(r => r.slice()); syncInputs(); } draw(); });
  $("mk-s0").addEventListener("change", e => { s0 = +e.target.value; draw(); });
  $("mk-n").addEventListener("input", e => { stop(); n = +e.target.value; $("mk-nv").textContent = n; draw(); });
  $("mk-play").addEventListener("click", () => {
    if (timer) { stop(); return; }
    if (n >= 2000) n = 0;
    $("mk-play").textContent = "Pause";
    timer = d3.interval(() => { n = Math.min(2000, n + (n < 50 ? 1 : 10)); $("mk-n").value = n; $("mk-nv").textContent = n; draw(); if (n >= 2000) stop(); }, 160);
  });
  $("mk-seed").addEventListener("click", () => { seed = (seed * 7919 + 29) % 100003; draw(); });
  $("mk-reset").addEventListener("click", () => {
    stop(); P = SP.mkPresets.device.map(r => r.slice()); s0 = 2; n = 500; seed = SP.seeds.mk;
    $("mk-pre").value = "device"; $("mk-s0").value = "2"; $("mk-n").value = 500; $("mk-nv").textContent = "500"; syncInputs(); draw();
  });
  syncInputs(); draw();
})();

/* ─────────────────── 7 · the Markov-chain CLT ───────────────────
   Two-state chain on {0, 1} that flips with probability a each step, started from π = (½, ½); h(x) = x.
   σ_h² = ¼, ρₖ = (1 − 2a)ᵏ, τ = 1 + 2Σρₖ = (1 − a)/a. R = 400 independent chains of length N.
   Exact variance of one chain's average: (σ_h²/N²)·[N + 2Σₖ₌₁^(N−1) (N − k)(1 − 2a)ᵏ].
   Caption audit: a = 0.05, N = 1,000, seed SP.seeds.erg; SP.ergRun prints the empirical sd of the 400
   averages, the naive coverage and the batch-means τ of chain 1.                                       */
SP.ergExactSd = function (a, N) {
  const rr = 1 - 2 * a; let s = N, p = 1;
  for (let k = 1; k < N; k++) { p *= rr; s += 2 * (N - k) * p; }
  return Math.sqrt(0.25 * s) / N;
};
SP.ergRun = function (a, N, R, seed) {
  const r = ST.rng(seed), avgs = [], first = [];
  for (let c = 0; c < R; c++) {
    let x = r() < 0.5 ? 0 : 1, s = 0;
    for (let i = 0; i < N; i++) { if (i) { if (r() < a) x = 1 - x; } s += x; if (c === 0) first.push(x); }
    avgs.push(s / N);
  }
  const sdEmp = ST.sd(avgs), naive = 0.5 / Math.sqrt(N), tau = (1 - a) / a;
  const cover = avgs.filter(v => Math.abs(v - 0.5) <= 1.96 * naive).length / R;
  const nb = 20, b = Math.floor(N / nb), bm = [];                 // batch means on chain 1
  for (let j = 0; j < nb; j++) { let s = 0; for (let i = j * b; i < (j + 1) * b; i++) s += first[i]; bm.push(s / b); }
  const tauBM = b * ST.variance(bm) / Math.max(1e-12, ST.variance(first));
  const ips = PV.ess(first);
  return { avgs, first, sdEmp, naive, tau, honest: naive * Math.sqrt(tau), exact: SP.ergExactSd(a, N), cover,
    coverTheory: 2 * ST.normCdf(1.96 * naive / SP.ergExactSd(a, N)) - 1, tauBM, tauIPS: ips.tau };
};
(function () {
  const svg = d3.select("#erg-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("erg-readout"), R = 400;
  let a = 0.05, N = 1000, seed = SP.seeds.erg;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), E = SP.ergRun(a, N, R, seed);
    // left top: first 120 steps of chain 1
    const S = Math.min(120, N), xp = d3.scaleLinear().domain([0, S]).range([46, 400]), yp = d3.scaleLinear().domain([-0.2, 1.2]).range([110, 40]);
    const st = []; for (let i = 0; i < S; i++) { st.push([i, E.first[i]]); st.push([i + 1, E.first[i]]); }
    g.append("path").attr("d", d3.line().x(p => xp(p[0])).y(p => yp(p[1]))(st)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.5);
    g.append("g").attr("class", "axis").attr("transform", "translate(46,0)").call(d3.axisLeft(yp).tickValues([0, 1]));
    SP.label(g, 46, 28, `chain 1, first ${S} steps (flip probability a = ${a.toFixed(3)} per step)`, { size: 10.5 });
    // left bottom: running average of chain 1
    const xr = d3.scaleLinear().domain([1, N]).range([46, 400]);
    const ra = []; let s = 0; E.first.forEach((v, i) => { s += v; if (i % Math.max(1, Math.floor(N / 300)) === 0 || i === N - 1) ra.push([i + 1, s / (i + 1)]); });
    const yr = d3.scaleLinear().domain([0, 1]).range([330, 150]);
    SP.axisB(g, xr, 330, 5); SP.axisL(g, yr, 46, 5);
    const ns = d3.range(1, 101).map(i => Math.max(2, Math.round(N * i / 100)));
    g.append("path").attr("d", d3.area().x(n => xr(n)).y0(n => yr(Math.max(0, 0.5 - 1.96 * 0.5 * Math.sqrt(E.tau / n)))).y1(n => yr(Math.min(1, 0.5 + 1.96 * 0.5 * Math.sqrt(E.tau / n))))(ns)).attr("fill", SC.good).attr("fill-opacity", 0.14);
    g.append("path").attr("d", d3.area().x(n => xr(n)).y0(n => yr(0.5 - 1.96 * 0.5 / Math.sqrt(n))).y1(n => yr(0.5 + 1.96 * 0.5 / Math.sqrt(n)))(ns)).attr("fill", SC.bad).attr("fill-opacity", 0.18);
    g.append("path").attr("d", d3.line().x(p => xr(p[0])).y(p => yr(p[1]))(ra)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.5);
    SP.label(g, 52, 162, "running average · red ±1.96 σ/√n (iid) · green ±1.96 σ√(τ/n)", { size: 10.5 });
    // right: histogram of the R averages
    const half = Math.max(4 * E.exact, 6 * E.naive);
    const xh = d3.scaleLinear().domain([0.5 - half, 0.5 + half]).range([446, 664]);
    const H = SP.hist(E.avgs, 0.5 - half, 0.5 + half, 26);
    const peak = Math.max(d3.max(H.bins, b => b.d), 1 / (E.naive * Math.sqrt(2 * Math.PI)));
    const yh = d3.scaleLinear().domain([0, peak * 1.08]).range([330, 40]);
    SP.axisB(g, xh, 330, 4); SP.axisL(g, yh, 446, 4);
    SP.bars(g, H.bins, xh, yh, SC.accent, 0.45);
    SP.curve(g, x => ST.normPdf((x - 0.5) / E.naive) / E.naive, xh, yh, 0.5 - half, 0.5 + half, { color: SC.bad, dash: "5 3" });
    SP.curve(g, x => ST.normPdf((x - 0.5) / E.honest) / E.honest, xh, yh, 0.5 - half, 0.5 + half, { color: SC.good });
    SP.label(g, 446, 28, `${R} chain averages (N = ${N}): red iid normal, green MC-CLT`, { size: 10.5 });
    out.innerHTML = `τ = 1 + 2Σρₖ = (1 − a)/a = <b>${E.tau.toFixed(3)}</b>, effective sample size N/τ = <b>${(N / E.tau).toFixed(1)}</b> of ${N}` +
      `<br>sd of the ${R} chain averages: simulated <b>${E.sdEmp.toFixed(4)}</b> · exact ${E.exact.toFixed(4)} · MC-CLT σ√(τ/N) ${E.honest.toFixed(4)} · naive iid σ/√N <b>${E.naive.toFixed(4)}</b>` +
      `<br>naive 95% interval covers ½ in <b>${(100 * E.cover).toFixed(1)}%</b> of chains (theory ${(100 * E.coverTheory).toFixed(1)}%) · τ estimated from chain 1 alone: batch means ${E.tauBM.toFixed(2)}, initial positive sequence ${E.tauIPS.toFixed(2)}`;
  }
  $("eg-a").addEventListener("input", e => { a = +e.target.value; $("eg-av").textContent = a.toFixed(3); draw(); });
  $("eg-n").addEventListener("input", e => { N = +e.target.value; $("eg-nv").textContent = N; draw(); });
  $("eg-seed").addEventListener("click", () => { seed = (seed * 7919 + 31) % 100003; draw(); });
  $("eg-reset").addEventListener("click", () => { a = 0.05; N = 1000; seed = SP.seeds.erg; $("eg-a").value = 0.05; $("eg-av").textContent = "0.050"; $("eg-n").value = 1000; $("eg-nv").textContent = "1000"; draw(); });
  draw();
})();

/* ─────────────────── 8 · random-walk Metropolis ───────────────────
   Target "bi": π(x) ∝ ½φ(x + 3) + ½φ(x − 3) (mean 0, variance 10); "far": modes at ±5 (variance 26),
   separated by a density valley of relative height e^(−12.5); "uni": N(0, 1). Proposal x′ = x + s·Z.
   Four chains of N iterations; the first 10% of each is discarded as burn-in. Chain 1 drives the trace,
   histogram, ACF and ESS; R̂ uses all four. Starts: "spread" (−6, −2, 2, 6) or "same" (all at the
   right-hand mode: +3, +5, or 0 for the unimodal target).
   Caption audit: bi, s = 1, N = 5,000, spread, seed SP.seeds.mh; SP.mhRun prints acceptance, mode switches,
   mean, τ, ESS and R̂ for the default and for s = 0.1 and s = 8 (the regimes quoted in the caption).     */
SP.mhT = {
  bi: { name: "½N(−3, 1) + ½N(3, 1)", logp: x => Math.log(0.5 * Math.exp(-0.5 * (x + 3) * (x + 3)) + 0.5 * Math.exp(-0.5 * (x - 3) * (x - 3))),
    pdf: x => 0.5 * ST.normPdf(x + 3) + 0.5 * ST.normPdf(x - 3), lo: -8, hi: 8, mean: 0, sd: Math.sqrt(10) },
  far: { name: "½N(−5, 1) + ½N(5, 1)", logp: x => { const a = -0.5 * (x + 5) * (x + 5), b = -0.5 * (x - 5) * (x - 5), m = Math.max(a, b); return m + Math.log(0.5 * Math.exp(a - m) + 0.5 * Math.exp(b - m)); },
    pdf: x => 0.5 * ST.normPdf(x + 5) + 0.5 * ST.normPdf(x - 5), lo: -9.5, hi: 9.5, mean: 0, sd: Math.sqrt(26) },
  uni: { name: "N(0, 1)", logp: x => -0.5 * x * x, pdf: x => ST.normPdf(x), lo: -4.5, hi: 4.5, mean: 0, sd: 1 }
};
SP.mhChain = function (key, s, N, x0, r) {
  const T = SP.mhT[key], x = new Array(N); let cur = x0, lp = T.logp(cur), acc = 0;
  for (let i = 0; i < N; i++) {
    const y = cur + s * ST.randn(r), ly = T.logp(y);
    if (Math.log(SP.unit(r)) < ly - lp) { cur = y; lp = ly; acc++; }
    x[i] = cur;
  }
  return { x, acc: acc / N };
};
SP.mhRun = function (key, s, N, starts, seed) {
  const r = ST.rng(seed), m0 = key === "far" ? 5 : key === "bi" ? 3 : 0, x0s = starts === "same" ? [m0, m0, m0, m0] : [-6, -2, 2, 6];
  const chains = x0s.map(x0 => SP.mhChain(key, s, N, x0, r));
  const b = Math.floor(N / 10), kept = chains.map(c => c.x.slice(b));
  const c1 = kept[0], E = PV.ess(c1), m = ST.mean(c1);
  let sw = 0, side = c1[0] < 0 ? -1 : 1;
  for (const v of c1) { if (side < 0 && v > 1.5) { sw++; side = 1; } else if (side > 0 && v < -1.5) { sw++; side = -1; } }
  const allMean = ST.mean([].concat(...kept));
  return { chains, kept, acc: chains[0].acc, mean: m, sd: ST.sd(c1), tau: E.tau, ess: E.ess, rhat: PV.rhat(kept), switches: sw,
    seNaive: ST.sd(c1) / Math.sqrt(c1.length), seHonest: ST.sd(c1) / Math.sqrt(E.ess), allMean, acf: PV.acf(c1, 60) };
};
(function () {
  const svg = d3.select("#mcmc-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("mcmc-readout");
  let key = "bi", lv = 0, N = 5000, starts = "spread", seed = SP.seeds.mh;
  const sOf = v => Math.pow(10, v);
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), T = SP.mhT[key], s = sOf(lv), M = SP.mhRun(key, s, N, starts, seed);
    const cols = [SC.accent, SC.a2, SC.good, SC.violet];
    // trace
    const xt = d3.scaleLinear().domain([0, N]).range([46, 420]), yt = d3.scaleLinear().domain([T.lo, T.hi]).range([170, 30]);
    SP.axisB(g, xt, 170, 5); SP.axisL(g, yt, 46, 5);
    g.append("rect").attr("x", 46).attr("width", xt(N / 10) - 46).attr("y", 30).attr("height", 140).attr("fill", SC.line).attr("fill-opacity", 0.35);
    const stride = Math.max(1, Math.floor(N / 900));
    M.chains.forEach((c, j) => {
      const pts = []; for (let i = 0; i < N; i += stride) pts.push([i, c.x[i]]);
      g.append("path").attr("d", d3.line().x(p => xt(p[0])).y(p => yt(p[1]))(pts)).attr("fill", "none").attr("stroke", cols[j]).attr("stroke-width", j ? 0.8 : 1.3).attr("stroke-opacity", j ? 0.45 : 0.95);
    });
    SP.label(g, 46, 20, `trace of 4 chains (chain 1 bold); grey = burn-in · proposal sd s = ${s.toFixed(s < 1 ? 2 : 1)}`, { size: 10.5 });
    // histogram of chain 1 vs target
    const xh = d3.scaleLinear().domain([T.lo, T.hi]).range([446, 664]);
    const H = SP.hist(M.kept[0], T.lo, T.hi, 40);
    const yh = d3.scaleLinear().domain([0, Math.max(d3.max(H.bins, b => b.d), d3.max(d3.range(0, 301), i => T.pdf(T.lo + (T.hi - T.lo) * i / 300))) * 1.1]).range([170, 30]);
    SP.axisB(g, xh, 170, 4); SP.axisL(g, yh, 446, 3);
    SP.bars(g, H.bins, xh, yh, SC.accent, 0.5);
    SP.curve(g, T.pdf, xh, yh, T.lo, T.hi);
    SP.label(g, 446, 20, `chain 1 after burn-in vs target ${T.name}`, { size: 10.5 });
    // ACF
    const xa = d3.scaleLinear().domain([0, 60]).range([46, 420]), ya = d3.scaleLinear().domain([-0.2, 1]).range([340, 215]);
    SP.axisB(g, xa, ya(0), 6); SP.axisL(g, ya, 46, 4);
    M.acf.forEach((v, k) => g.append("line").attr("x1", xa(k)).attr("x2", xa(k)).attr("y1", ya(0)).attr("y2", ya(v)).attr("stroke", SC.accent).attr("stroke-width", 2));
    SP.label(g, 46, 205, "autocorrelation ρₖ of chain 1, lags 0–60", { size: 10.5 });
    // text panel
    const tx = 454;
    g.append("rect").attr("x", tx - 8).attr("y", 196).attr("width", 226).attr("height", 150).attr("fill", SC.panel2).attr("stroke", SC.line).attr("rx", 6);
    [[`acceptance rate ${M.acc.toFixed(3)}`, SC.ink], [`mode switches (chain 1) ${key !== "uni" ? M.switches : "—"}`, SC.ink],
     [`mean ${M.mean.toFixed(3)} (true ${T.mean})`, SC.ink], [`τ = ${M.tau.toFixed(1)}, ESS = ${M.ess.toFixed(0)} of ${M.kept[0].length}`, SC.a2],
     [`SE naive ${M.seNaive.toFixed(3)}, honest ${M.seHonest.toFixed(3)}`, SC.ink], [`R̂ (4 chains) = ${M.rhat.toFixed(3)}`, M.rhat < 1.01 ? SC.good : SC.bad],
     [`starts: ${starts === "same" ? "all four at the right-hand mode" : "−6, −2, 2, 6"}`, SC.muted]]
      .forEach((l, k) => SP.label(g, tx, 216 + k * 19, l[0], { color: l[1], size: 10.5 }));
    out.innerHTML = `target <b>${T.name}</b>, proposal x′ = x + ${s.toFixed(s < 1 ? 2 : 1)}·Z, N = ${N} per chain: acceptance <b>${M.acc.toFixed(3)}</b> · ` +
      `chain-1 mean ${M.mean.toFixed(3)} (true ${T.mean}; sd ${T.sd.toFixed(3)}) · τ = <b>${M.tau.toFixed(1)}</b>, ESS = <b>${M.ess.toFixed(0)}</b> · ` +
      `naive SE ${M.seNaive.toFixed(3)} vs ESS-based SE ${M.seHonest.toFixed(3)} · R̂ = <b>${M.rhat.toFixed(3)}</b>` +
      (key !== "uni" ? ` · mode switches in chain 1: ${M.switches}` : "") +
      (starts === "same" && key !== "uni" && M.rhat < 1.05 && M.allMean > 1 ? `<br>all four chains started in the right-hand mode and none has left it: R̂ = ${M.rhat.toFixed(3)} looks fine while every chain misses half the target (pooled mean ${M.allMean.toFixed(2)}, true 0)` : "");
  }
  $("mh-t").addEventListener("change", e => { key = e.target.value; draw(); });
  $("mh-s").addEventListener("input", e => { lv = +e.target.value; $("mh-sv").textContent = sOf(lv).toFixed(sOf(lv) < 1 ? 2 : 1); draw(); });
  $("mh-n").addEventListener("input", e => { N = +e.target.value; $("mh-nv").textContent = N; draw(); });
  $("mh-st").addEventListener("change", e => { starts = e.target.value; draw(); });
  $("mh-seed").addEventListener("click", () => { seed = (seed * 7919 + 37) % 100003; draw(); });
  $("mh-reset").addEventListener("click", () => {
    key = "bi"; lv = 0; N = 5000; starts = "spread"; seed = SP.seeds.mh;
    $("mh-t").value = "bi"; $("mh-s").value = 0; $("mh-sv").textContent = "1.0"; $("mh-n").value = 5000; $("mh-nv").textContent = "5000"; $("mh-st").value = "spread"; draw();
  });
  draw();
})();

/* ─────────────────── 9 · the M/M/1 queue ───────────────────
   Service rate μₛ = 1 (time measured in mean service times), arrival rate λ = ρ. FIFO; sojourn times by
   the recursion Dᵢ = max(Aᵢ, Dᵢ₋₁) + Sᵢ. Time-average queue length from the piecewise-constant path; its
   standard error from 20 batch means. Theory (M/M/1): L = ρ/(1 − ρ), W = 1/(1 − ρ), P(N = n) = (1 − ρ)ρⁿ;
   (M/D/1): W = 1 + ρ/(2(1 − ρ)), L = λW.
   Caption audit: ρ = 0.8, exponential service, T = 20,000, seed SP.seeds.qu: SP.quRun prints L̄, W̄, λ̂W̄.  */
SP.quRun = function (rho, T, det, seed) {
  const r = ST.rng(seed), A = PV.poisProcess(rho, T, r), D = new Array(A.length);
  let prev = 0;
  for (let i = 0; i < A.length; i++) { const S = det ? 1 : PV.expDraw(1, r); prev = Math.max(A[i], prev) + S; D[i] = prev; }
  const W = A.map((a, i) => D[i] - a);
  // path: merge arrivals and departures within [0, T]
  const ev = []; A.forEach(t => ev.push([t, 1])); D.forEach(t => { if (t <= T) ev.push([t, -1]); });
  ev.sort((u, v) => u[0] - v[0] || u[1] - v[1]);
  const occ = new Array(60).fill(0), nb = 20, bw = T / nb, batch = new Array(nb).fill(0);
  let n = 0, t0 = 0, area = 0, busy = 0;
  const add = (from, to, k) => {                              // accumulate k over [from, to), split across batches
    let a = from;
    while (a < to) { const bi = Math.min(nb - 1, Math.floor(a / bw)), e = Math.min(to, (bi + 1) * bw); batch[bi] += k * (e - a); a = e; }
  };
  const path = [[0, 0]], marks = [];
  let nextMark = T / 200;
  for (const [t, d] of ev) {
    area += n * (t - t0); add(t0, t, n); occ[Math.min(59, n)] += t - t0; if (n > 0) busy += t - t0;
    while (t >= nextMark && nextMark <= T) { marks.push([nextMark, (area - n * (t - nextMark)) / nextMark]); nextMark += T / 200; }
    n += d; t0 = t;
    if (t <= 150) { path.push([t, n - d]); path.push([t, n]); }
  }
  area += n * (T - t0); add(t0, T, n); occ[Math.min(59, n)] += T - t0; if (n > 0) busy += T - t0;
  if (marks.length < 200) marks.push([T, area / T]);
  const bmeans = batch.map(v => v / bw);
  return { A, W, L: area / T, Wbar: ST.mean(W), lamHat: A.length / T, busy: busy / T, occ: occ.map(v => v / T), path, marks, seL: ST.sd(bmeans) / Math.sqrt(nb) };
};
SP.quTheory = function (rho, det) {
  if (det) { const W = 1 + rho / (2 * (1 - rho)); return { L: rho * W, W }; }
  return { L: rho / (1 - rho), W: 1 / (1 - rho) };
};
(function () {
  const svg = d3.select("#queue-svg");
  if (svg.empty()) return;
  const $ = SP.$, out = $("queue-readout");
  let rho = 0.8, T = 20000, det = false, seed = SP.seeds.qu;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), Q = SP.quRun(rho, T, det, seed), Th = SP.quTheory(rho, det);
    // path on [0, 150]
    const xp = d3.scaleLinear().domain([0, 150]).range([46, 664]);
    const pmax = Math.max(4, d3.max(Q.path, p => p[1])) + 1;
    const yp = d3.scaleLinear().domain([0, pmax]).range([140, 30]);
    SP.axisB(g, xp, 140, 8); SP.axisL(g, yp, 46, 4);
    g.append("path").attr("d", d3.line().x(p => xp(p[0])).y(p => yp(p[1]))(Q.path.concat([[150, Q.path[Q.path.length - 1][1]]]))).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.6);
    g.append("line").attr("x1", 46).attr("x2", 664).attr("y1", yp(Th.L)).attr("y2", yp(Th.L)).attr("stroke", SC.a2).attr("stroke-dasharray", "5 3");
    SP.label(g, 46, 20, `number in system N(t) on [0, 150] (ρ = ${rho.toFixed(2)}, ${det ? "M/D/1" : "M/M/1"}); dashed: L = ${Th.L.toFixed(2)}`, { size: 10.5 });
    // running time average
    const xr = d3.scaleLinear().domain([0, T]).range([46, 400]);
    const ymax = Math.max(Th.L * 2, d3.max(Q.marks, m => m[1]) * 1.1);
    const yr = d3.scaleLinear().domain([0, ymax]).range([340, 200]);
    SP.axisB(g, xr, 340, 4, "~s"); SP.axisL(g, yr, 46, 4);
    g.append("line").attr("x1", 46).attr("x2", 400).attr("y1", yr(Th.L)).attr("y2", yr(Th.L)).attr("stroke", SC.a2).attr("stroke-dasharray", "5 3");
    g.append("path").attr("d", d3.line().x(m => xr(m[0])).y(m => yr(m[1]))(Q.marks)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.6);
    SP.label(g, 46, 190, "time average of N(t) up to t, against L (dashed)", { size: 10.5 });
    // occupancy vs geometric
    const K = 16, xb = d3.scaleBand().domain(d3.range(K)).range([446, 664]).padding(0.15);
    const yb = d3.scaleLinear().domain([0, Math.max(1 - rho, d3.max(Q.occ.slice(0, K))) * 1.1]).range([340, 200]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,340)").call(d3.axisBottom(xb).tickValues([0, 5, 10, 15]));
    SP.axisL(g, yb, 446, 3);
    d3.range(K).forEach(k => {
      g.append("rect").attr("x", xb(k)).attr("width", xb.bandwidth()).attr("y", yb(Q.occ[k])).attr("height", 340 - yb(Q.occ[k])).attr("fill", SC.accent).attr("fill-opacity", 0.5);
      g.append("circle").attr("cx", xb(k) + xb.bandwidth() / 2).attr("cy", yb((1 - rho) * Math.pow(rho, k))).attr("r", 2.8).attr("fill", SC.a2);
    });
    SP.label(g, 446, 190, `share of time with N = n; dots (1 − ρ)ρⁿ${det ? " (M/M/1, for contrast)" : ""}`, { size: 10.5 });
    const relax = 1 / Math.pow(1 - Math.sqrt(rho), 2);
    out.innerHTML = `λ = ρ = ${rho.toFixed(2)}, μₛ = 1, T = ${T.toLocaleString()}: ${Q.A.length.toLocaleString()} arrivals (λ̂ = ${Q.lamHat.toFixed(4)}) · server busy ${(100 * Q.busy).toFixed(1)}% of the time (ρ = ${(100 * rho).toFixed(0)}%)` +
      `<br>time-average number in system L̄ = <b>${Q.L.toFixed(3)}</b> ± ${Q.seL.toFixed(3)} (batch-means SE) against L = <b>${Th.L.toFixed(3)}</b> · mean time in system W̄ = <b>${Q.Wbar.toFixed(3)}</b> against W = ${Th.W.toFixed(3)}` +
      `<br>Little's law on the run: λ̂·W̄ = <b>${(Q.lamHat * Q.Wbar).toFixed(3)}</b> vs L̄ = ${Q.L.toFixed(3)}` +
      (det ? "" : ` · relaxation time 1/(1 − √ρ)² = ${relax.toFixed(0)} service times · large-T theory SE of L̄: √(2ρ(1 + ρ)/((1 − ρ)⁴T)) = ${Math.sqrt(2 * rho * (1 + rho) / (Math.pow(1 - rho, 4) * T)).toFixed(3)}`);
  }
  $("qu-r").addEventListener("input", e => { rho = +e.target.value; $("qu-rv").textContent = rho.toFixed(2); draw(); });
  $("qu-T").addEventListener("change", e => { T = +e.target.value; draw(); });
  $("qu-s").addEventListener("change", e => { det = e.target.value === "det"; draw(); });
  $("qu-seed").addEventListener("click", () => { seed = (seed * 7919 + 41) % 100003; draw(); });
  $("qu-reset").addEventListener("click", () => {
    rho = 0.8; T = 20000; det = false; seed = SP.seeds.qu;
    $("qu-r").value = 0.8; $("qu-rv").textContent = "0.80"; $("qu-T").value = "20000"; $("qu-s").value = "exp"; draw();
  });
  draw();
})();
