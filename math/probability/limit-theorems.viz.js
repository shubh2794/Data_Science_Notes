/* limit-theorems.viz.js — the eight visualizations on math/probability/limit-theorems.html.
   Loaded after ../../data.js → ../../notes.js → ../statistics/stats-viz.js (SC palette, ST toolbox)
   → prob-viz.js (PV: Pareto/gamma/exponential/lognormal functions, PV.cauchy*, PV.gumbel*,
   PV.frechet*, PV.ksDist, PV.simpson, PV.invert). Distribution functions come from ST / PV and
   are never re-implemented here.

   Everything page-local lives under the single namespace LT. Each figure is an IIFE that exits
   quietly if its <svg> is not on the page. The pure computations are exposed on LT so the build's
   node checks can call them without a DOM.

     1  #modes-svg    modes of convergence: typewriter, spike, flip, independent blips
     2  #lln-svg      fifty running averages in an ε-band, and their √n-scale gaps
     3  #clt-svg      the exact standardised sum against φ; sup-distance, Berry–Esseen, Edgeworth
     4  #stable-svg   Pareto / Cauchy means: IQR against n (CLT vs stable slope) and shape
     5  #delta-svg    the delta-method lens: N(μ, s²) through g, exact against tangent-line normal
     6  #tstat-svg    s/σ paths and the t statistic against N(0, 1) and t(n − 1)
     7  #quant-svg    the sample quantile: simulation, exact order-statistic density, asymptotic normal
     8  #extreme-svg  the normalised maximum against its extreme-value limit, beside the CLT for the mean */

const LT = {
  // Default seeds, fixed by the caption audit (see the note at each figure).
  seeds: { modes: 29, lln: 16, stable: 15, ts: 11, quant: 17, ext: 13 },
  label(g, x, y, t, o) {
    const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", (o && o.size) || 11)
      .attr("fill", (o && o.color) || SC.muted).text(t);
    if (o && o.anchor) e.attr("text-anchor", o.anchor);
    if (o && o.weight) e.attr("font-weight", o.weight);
    return e;
  },
  f(x, d) { const dd = d === undefined ? 4 : d; return (x !== 0 && isFinite(x) && Math.abs(x) < Math.pow(10, -dd) / 2) ? x.toExponential(2) : ST.fmt(x, dd); },
  mom(a) {                                          // mean, sd (n − 1), skewness (moment ratio)
    const n = a.length; let m = 0;
    for (let i = 0; i < n; i++) m += a[i];
    m /= n;
    let s2 = 0, s3 = 0;
    for (let i = 0; i < n; i++) { const d = a[i] - m; s2 += d * d; s3 += d * d * d; }
    const v = s2 / n;
    return { mean: m, sd: Math.sqrt(s2 / (n - 1)), skew: v > 0 ? (s3 / n) / Math.pow(v, 1.5) : 0 };
  },
  quantSorted(s, p) {                               // linear-interpolation quantile of a sorted array
    const h = (s.length - 1) * p, lo = Math.floor(h), hi = Math.ceil(h);
    return s[lo] + (h - lo) * (s[hi] - s[lo]);
  },
  hist(a, lo, hi, nb) {                             // density-scaled histogram; values outside are counted, not drawn
    const w = (hi - lo) / nb, c = new Array(nb).fill(0);
    let out = 0;
    for (const v of a) { const k = Math.floor((v - lo) / w); if (k >= 0 && k < nb) c[k]++; else out++; }
    return { bins: c.map((k, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, d: k / (a.length * w) })), out };
  },
  yes(g, x, y, ok, txt) {
    LT.label(g, x, y, ok ? "yes" : "no", { color: ok ? SC.good : SC.bad, weight: 700 });
    LT.label(g, x + 30, y, txt, { size: 10.5 });
  }
};

/* ─────────────────── 1 · modes of convergence ───────────────────
   Caption audit: example "type", n = 37 = 2⁵ + 5, ω = 0.17 → interval [5/32, 6/32) = [0.15625, 0.1875)
   contains ω, X₃₇(ω) = 1, P(X₃₇ ≠ 0) = 1/32 = 0.031; along m = 1..63 the path equals 1 exactly once per
   block k = 0..5 (6 hits).                                                                           */
LT.modeX = function (ex, m, w, path) {
  if (ex === "type") { const k = Math.floor(Math.log2(m) + 1e-12), j = m - Math.pow(2, k); return (w >= j / Math.pow(2, k) && w < (j + 1) / Math.pow(2, k)) ? 1 : 0; }
  if (ex === "spike") return w < 1 / m ? m : 0;
  if (ex === "flip") return -ST.normQuant(w);
  return path[m];                                   // independent blips: a pre-drawn path for this ω
};
LT.modeStats = function (ex, n) {                   // at the current n, against the limit (0, or X for the flip); ε = ½
  if (ex === "type") { const k = Math.floor(Math.log2(n) + 1e-12), p = Math.pow(2, -k); return { pm: p, e1: p, e2: p, k }; }
  if (ex === "spike") return { pm: 1 / n, e1: 1, e2: n };
  if (ex === "flip") return { pm: 2 * (1 - ST.normCdf(0.25)), e1: 2 * Math.sqrt(2 / Math.PI), e2: 4 };
  const p = ex === "ind1" ? 1 / n : 1 / (n * n);
  return { pm: p, e1: p, e2: p };
};
LT.modeVerdict = {
  type: [true, true, false, true], spike: [true, true, true, false], flip: [true, false, false, false],
  ind1: [true, true, false, true], ind2: [true, true, true, true]
};
LT.blipPath = function (ex, w, N) {
  const r = ST.rng(LT.seeds.modes * 100003 + Math.round(w * 1000)), out = [0];
  for (let m = 1; m <= N; m++) { const u = r(); out.push(u < (ex === "ind1" ? 1 / m : 1 / (m * m)) ? 1 : 0); }
  return out;
};
(function () {
  const svg = d3.select("#modes-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("modes-readout"), N = 63;
  let ex = "type", n = 37, w = 0.17, timer = null, path = null;
  const refreshPath = () => { path = (ex === "ind1" || ex === "ind2") ? LT.blipPath(ex, w, N) : null; };
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 40, x1 = 440;
    const xs = d3.scaleLinear().domain([0, 1]).range([x0, x1]);
    const xm = d3.scaleLinear().domain([0.5, N + 0.5]).range([x0, x1]);
    const vals = d3.range(1, N + 1).map(m => LT.modeX(ex, m, w, path));
    const indep = ex === "ind1" || ex === "ind2";
    // ── top strip: the sample space, or the blip probabilities
    if (!indep && ex !== "flip") {
      g.append("rect").attr("x", x0).attr("y", 26).attr("width", x1 - x0).attr("height", 22).attr("fill", SC.panel2).attr("stroke", SC.line);
      let a = 0, b = 0;
      if (ex === "type") { const k = Math.floor(Math.log2(n) + 1e-12), j = n - Math.pow(2, k); a = j / Math.pow(2, k); b = (j + 1) / Math.pow(2, k); }
      else { a = 0; b = 1 / n; }
      g.append("rect").attr("x", xs(a)).attr("y", 26).attr("width", Math.max(1, xs(b) - xs(a))).attr("height", 22).attr("fill", SC.a2).attr("fill-opacity", 0.8);
      g.append("g").attr("class", "axis").attr("transform", "translate(0,48)").call(d3.axisBottom(xs).ticks(5));
      LT.label(g, x0, 18, `Ω = (0, 1): orange = {ω : X${n === 1 ? "₁" : "ₙ"}(ω) ≠ 0} at n = ${n}`, { size: 10.5 });
    } else if (ex === "flip") {
      const ys = d3.scaleLinear().domain([-2.6, 2.6]).range([70, 14]);
      const us = d3.range(0.004, 0.9965, 0.004);
      g.append("rect").attr("x", xs(ST.normCdf(-0.25))).attr("y", 14).attr("width", xs(ST.normCdf(0.25)) - xs(ST.normCdf(-0.25))).attr("height", 56).attr("fill", SC.good).attr("fill-opacity", 0.12);
      g.append("path").attr("d", d3.line().x(u => xs(u)).y(u => ys(ST.normQuant(u)))(us)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.8);
      g.append("path").attr("d", d3.line().x(u => xs(u)).y(u => ys(-ST.normQuant(u)))(us)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.8);
      g.append("g").attr("class", "axis").attr("transform", "translate(0,70)").call(d3.axisBottom(xs).ticks(5));
      LT.label(g, x0 + 4, 24, "blue X(ω) = Φ⁻¹(ω) · orange Xₙ(ω) = −X(ω) · green: |Xₙ − X| ≤ ½", { size: 10 });
    } else {
      const ys = d3.scaleLinear().domain([0, 1]).range([66, 22]);
      for (let m = 1; m <= N; m++) {
        const p = ex === "ind1" ? 1 / m : 1 / (m * m);
        g.append("rect").attr("x", xm(m) - 2).attr("width", 4).attr("y", ys(p)).attr("height", 66 - ys(p)).attr("fill", m === n ? SC.a2 : SC.muted).attr("fill-opacity", m === n ? 1 : 0.55);
      }
      g.append("line").attr("x1", x0).attr("x2", x1).attr("y1", 66).attr("y2", 66).attr("stroke", SC.line);
      LT.label(g, x0, 16, `P(Yₘ = 1) = ${ex === "ind1" ? "1/m" : "1/m²"}, m = 1…63 (independent; ω only selects the random stream)`, { size: 10.5 });
    }
    // ── main panel: the path along ω
    const yTop = 100, yBot = 330;
    let lo = 0, hi = Math.max(1.2, d3.max(vals) * 1.1);
    if (ex === "flip") { const v = Math.abs(ST.normQuant(w)); lo = -Math.max(1, v) * 1.3; hi = -lo; }
    const yp = d3.scaleLinear().domain([lo, hi]).range([yBot, yTop]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yp(0)})`).call(d3.axisBottom(xm).tickValues([1, 2, 4, 8, 16, 32, 63]).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yp).ticks(5));
    if (ex === "type") for (let k = 1; k <= 5; k++) g.append("line").attr("x1", xm(Math.pow(2, k) - 0.5)).attr("x2", xm(Math.pow(2, k) - 0.5)).attr("y1", yTop).attr("y2", yBot).attr("stroke", SC.line).attr("stroke-dasharray", "2 3");
    if (ex === "flip") {
      const X = ST.normQuant(w);
      g.append("line").attr("x1", x0).attr("x2", x1).attr("y1", yp(X)).attr("y2", yp(X)).attr("stroke", SC.accent).attr("stroke-dasharray", "5 3");
      LT.label(g, x1 - 2, yp(X) - 5, `limit X(ω) = ${X.toFixed(3)}`, { anchor: "end", color: SC.accent, size: 10 });
    }
    vals.forEach((v, i) => {
      const m = i + 1, cur = m === n;
      g.append("line").attr("x1", xm(m)).attr("x2", xm(m)).attr("y1", yp(0)).attr("y2", yp(v)).attr("stroke", cur ? SC.a2 : (v !== 0 ? SC.ink : SC.line)).attr("stroke-width", cur ? 2.5 : 1.4);
      g.append("circle").attr("cx", xm(m)).attr("cy", yp(v)).attr("r", cur ? 4.5 : (v !== 0 ? 3 : 1.6)).attr("fill", cur ? SC.a2 : (v !== 0 ? SC.ink : SC.muted));
    });
    LT.label(g, x0, yTop - 10, `the path m ↦ Xₘ(ω) for ω = ${w.toFixed(3)}; current n = ${n} in orange`, { size: 10.5 });
    LT.label(g, x1, yBot + 30, "m", { anchor: "end" });
    // ── verdict panel
    const S = LT.modeStats(ex, n), V = LT.modeVerdict[ex], vx = 462;
    g.append("rect").attr("x", vx - 8).attr("y", 14).attr("width", 222).attr("height", 330).attr("fill", SC.panel2).attr("stroke", SC.line).attr("rx", 6);
    LT.label(g, vx, 34, `at n = ${n} (ε = ½):`, { color: SC.ink, weight: 700 });
    LT.label(g, vx, 54, `P(|Xₙ − X| > ½) = ${LT.f(S.pm, 4)}`, { color: SC.ink });
    LT.label(g, vx, 72, `E|Xₙ − X| = ${LT.f(S.e1, 4)}`, { color: SC.ink });
    LT.label(g, vx, 90, `E(Xₙ − X)² = ${LT.f(S.e2, 4)}`, { color: SC.ink });
    LT.label(g, vx, 122, "as n → ∞, does Xₙ converge", { color: SC.ink, weight: 700 });
    LT.yes(g, vx, 144, V[0], "in distribution");
    LT.yes(g, vx, 164, V[1], "in probability");
    LT.yes(g, vx, 184, V[2], "almost surely");
    LT.yes(g, vx, 204, V[3], "in quadratic mean (L²)");
    const why = {
      type: ["every ω is hit once in every block", "of length 2ᵏ, so no path converges;", "but P(hit) = 2⁻ᵏ → 0"],
      spike: ["each path is 0 once n > 1/ω,", "but E Xₙ = 1 and E Xₙ² = n:", "a rare, ever taller spike"],
      flip: ["same distribution as X for every n,", "yet |Xₙ − X| = 2|X| never shrinks:", "P(|X| > ¼) = 0.8026 for all n"],
      ind1: ["Σ 1/m = ∞ and the blips are", "independent: Borel–Cantelli II gives", "infinitely many 1s, almost surely"],
      ind2: ["Σ 1/m² = π²/6 < ∞: Borel–Cantelli I", "gives only finitely many 1s,", "almost surely"]
    }[ex];
    why.forEach((t, i) => LT.label(g, vx, 236 + i * 16, t, { size: 10.5 }));
    const hits = vals.filter(v => v !== 0).length, last = vals.reduce((a, v, i) => (v !== 0 ? i + 1 : a), 0);
    LT.label(g, vx, 300, ex === "flip" ? `|Xₘ − X| = ${(2 * Math.abs(ST.normQuant(w))).toFixed(3)} for every m` : `non-zero terms on this path: ${hits}`, { color: SC.ink, size: 10.5 });
    if (ex !== "flip") LT.label(g, vx, 318, last ? `last non-zero term (m ≤ 63): m = ${last}` : "no non-zero term in m ≤ 63", { size: 10.5 });
    out.innerHTML = `<b>${{ type: "typewriter", spike: "growing spike", flip: "flip", ind1: "independent blips, 1/n", ind2: "independent blips, 1/n²" }[ex]}</b>, n = ${n}, ω = ${w.toFixed(3)}: ` +
      `X<sub>n</sub>(ω) = ${LT.f(vals[n - 1], 3)} · P(|Xₙ − X| &gt; ½) = <b>${LT.f(S.pm, 4)}</b> · E(Xₙ − X)² = <b>${LT.f(S.e2, 4)}</b>` +
      (ex === "type" ? `<br>n = 2^${S.k} + ${n - Math.pow(2, S.k)}: the interval has length 2^−${S.k} = ${LT.f(Math.pow(2, -S.k), 4)}; this path is 1 at ${hits} of the 63 terms (once per block)` : "") +
      (ex === "spike" ? `<br>the path is 0 from m = ${Math.ceil(1 / w - 1e-12)} on, but the mean of Xₙ stays 1 and its mean square is n = ${n}` : "") +
      (ex === "flip" ? `<br>Fₙ = F exactly (the normal is symmetric), yet the distance between the variables is 2|X(ω)| = ${(2 * Math.abs(ST.normQuant(w))).toFixed(3)}` : "") +
      (ex === "ind1" ? `<br>expected number of 1s among m ≤ 63: H₆₃ = ${LT.f(d3.sum(d3.range(1, 64), m => 1 / m), 3)}; among m ≤ 10⁶: about 14.4, and it never stops growing` : "") +
      (ex === "ind2" ? `<br>expected number of 1s among all m: π²/6 = 1.645 (the first term is always 1)` : "");
  }
  function setN(v) { n = v; $("md-n").value = n; $("md-nv").textContent = n; }
  function stop() { if (timer) { timer.stop(); timer = null; $("md-play").textContent = "Play"; } }
  $("md-ex").addEventListener("change", e => { ex = e.target.value; stop(); refreshPath(); draw(); });
  $("md-n").addEventListener("input", e => { stop(); setN(+e.target.value); draw(); });
  $("md-w").addEventListener("input", e => { w = +e.target.value; $("md-wv").textContent = w.toFixed(3); refreshPath(); draw(); });
  $("md-play").addEventListener("click", () => {
    if (timer) { stop(); return; }
    if (n >= N) setN(1);
    $("md-play").textContent = "Pause";
    timer = d3.interval(() => { if (n >= N) { stop(); return; } setN(n + 1); draw(); }, 220);
  });
  $("md-reset").addEventListener("click", () => {
    stop(); ex = "type"; w = 0.17; $("md-ex").value = "type"; $("md-w").value = 0.17; $("md-wv").textContent = "0.170";
    setN(37); refreshPath(); draw();
  });
  refreshPath(); draw();
})();

/* ─────────────────── 2 · LLN: fifty averages and their gaps ───────────────────
   Caption audit: population "coin" (heads = 1, μ = σ = ½), 50 paths of 5,000 tosses, seed LT.seeds.lln,
   ε = 0.05, n = 400 → Chebyshev bound σ²/(nε²) = 0.25, normal approximation 2(1 − Φ(2)) = 0.0455;
   the simulated fractions (outside the band at n, and leaving it at some m in [n, 5000]) and the
   mean |gap| (against √(2n/π) = 15.96) are printed by the build check and recorded in the caption. */
LT.llnPop = {
  coin: { mu: 0.5, sd: 0.5, gapSd: 1, gapScale: 2, draw: r => (r() < 0.5 ? 1 : 0), name: "fair coin" },
  exp: { mu: 1, sd: 1, gapSd: 1, gapScale: 1, draw: r => PV.expDraw(1, r), name: "Exp(1)" },
  cauchy: { mu: 0, sd: NaN, gapSd: NaN, gapScale: 1, draw: r => PV.cauchyDraw(0, 1, r), name: "Cauchy" }
};
LT.llnSim = function (key, seed, P, N) {           // P paths of N draws: running means and gaps (Float64Array per path)
  const F = LT.llnPop[key], r = ST.rng(seed), means = [], gaps = [];
  for (let p = 0; p < P; p++) {
    const mArr = new Float64Array(N + 1), gArr = new Float64Array(N + 1);
    let s = 0;
    for (let m = 1; m <= N; m++) { s += F.draw(r); mArr[m] = s / m; gArr[m] = F.gapScale * (s - m * F.mu); }
    means.push(mArr); gaps.push(gArr);
  }
  return { means, gaps };
};
LT.llnStats = function (key, D, n, eps, N) {
  const F = LT.llnPop[key], P = D.means.length;
  let outAt = 0, outAfter = 0, absGap = 0, bigGap = 0;
  for (let p = 0; p < P; p++) {
    const a = D.means[p];
    if (Math.abs(a[n] - F.mu) > eps) outAt++;
    let left = false;
    for (let m = n; m <= N; m++) if (Math.abs(a[m] - F.mu) > eps) { left = true; break; }
    if (left) outAfter++;
    absGap += Math.abs(D.gaps[p][n]);
    if (key !== "cauchy" && Math.abs(D.gaps[p][n]) > F.gapSd * Math.sqrt(n)) bigGap++;
  }
  return { outAt: outAt / P, outAfter: outAfter / P, meanAbsGap: absGap / P, bigGap: bigGap / P };
};
(function () {
  const svg = d3.select("#lln-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("lln-readout"), P = 50, N = 5000;
  let key = "coin", eps = 0.05, n = 400, seed = LT.seeds.lln, D = null;
  const lN = Math.log10(N);
  const sims = () => { D = LT.llnSim(key, seed, P, N); };
  const idx = (function () { const s = new Set(); for (let m = 1; m <= 60; m++) s.add(m); for (let i = 0; i <= 260; i++) s.add(Math.round(Math.pow(10, Math.log10(60) + i * (lN - Math.log10(60)) / 260))); return [...s].filter(m => m <= N).sort((a, b) => a - b); })();
  function draw() {
    svg.selectAll("*").remove();
    const F = LT.llnPop[key], g = svg.append("g");
    const x = d3.scaleLog().domain([1, N]).range([56, 665]);
    const dom = key === "coin" ? [0.15, 0.85] : key === "exp" ? [0.1, 2.2] : [-6, 6];
    const yt = d3.scaleLinear().domain(dom).range([175, 18]).clamp(true);
    g.append("rect").attr("x", 56).attr("width", 609).attr("y", yt(F.mu + eps)).attr("height", yt(F.mu - eps) - yt(F.mu + eps)).attr("fill", SC.good).attr("fill-opacity", 0.15);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,175)").call(d3.axisBottom(x).ticks(4, "~s"));
    g.append("g").attr("class", "axis").attr("transform", "translate(56,0)").call(d3.axisLeft(yt).ticks(5));
    const line = arr => d3.line().x(m => x(m)).y(m => yt(arr[m]))(idx);
    D.means.forEach(a => g.append("path").attr("d", line(a)).attr("fill", "none").attr("stroke", Math.abs(a[n] - F.mu) > eps ? SC.a2 : SC.accent).attr("stroke-opacity", 0.45).attr("stroke-width", 1));
    g.append("line").attr("x1", 56).attr("x2", 665).attr("y1", yt(F.mu)).attr("y2", yt(F.mu)).attr("stroke", SC.ink).attr("stroke-dasharray", "4 3").attr("stroke-opacity", 0.7);
    LT.label(g, 60, 12, `${F.name}: 50 running averages X̄ₘ; green band μ ± ε = ${F.mu} ± ${eps}; orange = outside the band at n`, { size: 10.5 });
    // gap panel
    const gmax = key === "cauchy" ? 4 * N : 1.15 * Math.sqrt(2 * N * Math.log(Math.log(N)));
    const yb = d3.scaleLinear().domain([-gmax, gmax]).range([370, 210]).clamp(true);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,370)").call(d3.axisBottom(x).ticks(4, "~s"));
    g.append("g").attr("class", "axis").attr("transform", "translate(56,0)").call(d3.axisLeft(yb).ticks(5, "~s"));
    const env = (fn, col, dash) => [1, -1].forEach(sgn => g.append("path").attr("d", d3.line().x(m => x(m)).y(m => yb(sgn * fn(m)))(idx.filter(m => m >= 3))).attr("fill", "none").attr("stroke", col).attr("stroke-dasharray", dash).attr("stroke-width", 1.4));
    D.gaps.forEach(a => g.append("path").attr("d", d3.line().x(m => x(m)).y(m => yb(a[m]))(idx)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-opacity", 0.35).attr("stroke-width", 1));
    if (key !== "cauchy") {
      env(m => F.gapSd * Math.sqrt(m), SC.ink, "3 3");
      env(m => 2 * F.gapSd * Math.sqrt(m), SC.muted, "3 3");
      env(m => F.gapSd * Math.sqrt(2 * m * Math.log(Math.log(m))), SC.a2, "6 3");
      LT.label(g, 60, 204, `gap ${key === "coin" ? "Dₘ = heads − tails" : "Sₘ − m"}: white ±√m, grey ±2√m, orange ±√(2m ln ln m) (iterated logarithm)`, { size: 10.5 });
    } else {
      env(m => m, SC.a2, "6 3");
      LT.label(g, 60, 204, "running sum Sₘ: orange ±m, the quartiles of Sₘ (Sₘ/m is exactly Cauchy for every m)", { size: 10.5 });
    }
    [175, 370].forEach((yy, i) => g.append("line").attr("x1", x(n)).attr("x2", x(n)).attr("y1", i ? 210 : 18).attr("y2", yy).attr("stroke", SC.ink).attr("stroke-width", 1.2).attr("stroke-opacity", 0.8));
    LT.label(g, 665, 394, "m (log scale)", { anchor: "end" });
    const S = LT.llnStats(key, D, n, eps, N);
    if (key === "cauchy") {
      const exact = 1 - (2 / Math.PI) * Math.atan(eps);
      out.innerHTML = `n = ${n}, ε = ${eps}: paths outside the band at n: <b>${LT.f(S.outAt, 2)}</b> (exact P(|X̄ₙ| &gt; ε) = 1 − (2/π)·atan ε = ${LT.f(exact, 4)} for <b>every</b> n) · ` +
        `leave the band at some m in [n, 5000]: <b>${LT.f(S.outAfter, 2)}</b><br>no mean, no law of large numbers: the band never captures the paths, and each jump is a single draw with |Xₘ| comparable to m`;
      return;
    }
    const cheb = Math.min(1, F.sd * F.sd / (n * eps * eps)), norm = 2 * (1 - ST.normCdf(eps * Math.sqrt(n) / F.sd));
    out.innerHTML = `n = ${n}, ε = ${eps}: <b>weak law</b> — outside the band at n: <b>${LT.f(S.outAt, 2)}</b> of 50 paths (Chebyshev bound ${LT.f(cheb, 4)}, normal approximation ${LT.f(norm, 4)}) · ` +
      `<b>strong law</b> — leave the band at some m in [n, 5000]: <b>${LT.f(S.outAfter, 2)}</b>` +
      `<br>gap at n: mean |gap| = <b>${LT.f(S.meanAbsGap, 2)}</b> (√(2n/π) = ${LT.f(Math.sqrt(2 * n / Math.PI), 2)}); beyond ±√n: ${LT.f(S.bigGap, 2)} of paths (limit 0.317, not shrinking) — the gap wanders while the average settles`;
  }
  const setN = v => { n = v; $("ll-nv").textContent = n; };
  $("ll-pop").addEventListener("change", e => { key = e.target.value; sims(); draw(); });
  $("ll-eps").addEventListener("input", e => { eps = +e.target.value; $("ll-epsv").textContent = eps.toFixed(2); draw(); });
  $("ll-n").addEventListener("input", e => { setN(Math.max(1, Math.round(Math.pow(10, lN * (+e.target.value) / 1000)))); draw(); });
  $("ll-sim").addEventListener("click", () => { seed += 1; sims(); draw(); });
  $("ll-reset").addEventListener("click", () => {
    key = "coin"; eps = 0.05; seed = LT.seeds.lln; $("ll-pop").value = "coin"; $("ll-eps").value = 0.05; $("ll-epsv").textContent = "0.05";
    $("ll-n").value = Math.round(1000 * Math.log10(400) / lN); setN(400); sims(); draw();
  });
  setN(400); sims(); draw();
})();

/* ─────────────────── 3 · CLT: the exact standardised sum, and its distance to Φ ───────────────────
   Caption audit: terms "exp", n = 10 → Z₁₀ = (Gamma(10, 1) − 10)/√10, skewness 2/√10 = 0.632, excess
   kurtosis 0.6, sup |F − Φ| = 0.0421 (exact, gamma cdf), Edgeworth |γ₁|/(6√(2πn)) = 0.0421, Berry–Esseen
   0.4690·(12/e − 2)/√10 = 0.358, ratio 8.5.                                                          */
LT.cltFam = {
  exp: { mu: 1, sd: 1, g1: 2, g2: 6, rho: 12 / Math.E - 2, name: "Exp(1)" },
  chi: { mu: 1, sd: Math.SQRT2, g1: Math.sqrt(8), g2: 12, rho: 3.072932, name: "χ²₁" },          // ρ/σ³ by quadrature (build check)
  unif: { mu: 0.5, sd: Math.sqrt(1 / 12), g1: 0, g2: -1.2, rho: (1 / 32) / Math.pow(1 / 12, 1.5), name: "U(0, 1)" },
  b01: { p: 0.1, mu: 0.1, sd: 0.3, g1: 0.8 / 0.3, g2: (1 - 6 * 0.09) / 0.09, rho: 0.82 / 0.3, name: "Bernoulli(0.1)", lattice: true },
  b05: { p: 0.5, mu: 0.5, sd: 0.5, g1: 0, g2: -2, rho: 1, name: "Bernoulli(½)", lattice: true }
};
LT.unifCdfs = function (nmax, h) {                  // cdf of S_k (k = 1..nmax) on the grid x = i·h, by G_{k+1}(x) = ∫_{x−1}^{x} G_k
  const per = Math.round(1 / h), out = [];
  let G = new Float64Array(nmax * per + 1);
  for (let i = 0; i < G.length; i++) G[i] = Math.min(1, i / per);
  out[1] = G;
  for (let k = 2; k <= nmax; k++) {
    const C = new Float64Array(G.length);         // cumulative trapezoid integral of G_{k−1}
    for (let i = 1; i < G.length; i++) C[i] = C[i - 1] + h * (G[i] + G[i - 1]) / 2;
    const H = new Float64Array(G.length);
    for (let i = 0; i < G.length; i++) H[i] = C[i] - (i - per >= 0 ? C[i - per] : 0);
    out[k] = H; G = H;
  }
  return out;
};
LT.cltCdf = function (key, n) {                     // exact cdf of Zₙ as a function, plus a density for drawing
  const F = LT.cltFam[key], s = F.sd * Math.sqrt(n), m = n * F.mu;
  if (key === "exp") return { cdf: z => ST.gammaP(n, Math.max(0, m + z * s)), pdf: z => s * PV.gammaPdf(m + z * s, n, 1) };
  if (key === "chi") return { cdf: z => ST.gammaP(n / 2, Math.max(0, (m + z * s) / 2)), pdf: z => s * PV.gammaPdf(m + z * s, n / 2, 0.5) };
  if (key === "unif") {
    const h = 0.004, per = 250;
    if (!LT._uc) LT._uc = LT.unifCdfs(100, h);
    const G = LT._uc[n];
    const at = x => { if (x <= 0) return 0; if (x >= n) return 1; const t = x / h, i = Math.floor(t); return G[i] + (t - i) * (G[Math.min(i + 1, G.length - 1)] - G[i]); };
    const prev = n > 1 ? LT._uc[n - 1] : null;
    const prevAt = x => { if (n === 1) return x <= 0 ? 0 : x >= 1 ? 1 : x; if (x <= 0) return 0; if (x >= n - 1) return 1; const t = x / h, i = Math.floor(t); return prev[i] + (t - i) * (prev[Math.min(i + 1, prev.length - 1)] - prev[i]); };
    return { cdf: z => at(m + z * s), pdf: z => s * (prevAt(m + z * s) - prevAt(m + z * s - 1)) };
  }
  const p = F.p;                                    // lattice: binomial
  return { lattice: true, n, p, s, m, cdfK: k => ST.binomCdf(k, n, p), pmf: k => ST.binomPmf(k, n, p) };
};
LT.cltSup = function (key, n) {                     // exact sup_z |P(Zₙ ≤ z) − Φ(z)|
  const C = LT.cltCdf(key, n);
  if (C.lattice) {
    let d = 0;
    for (let k = 0; k <= n; k++) {
      const z = (k - C.m) / C.s, Fk = ST.binomCdf(k, n, C.p), Fm = k ? ST.binomCdf(k - 1, n, C.p) : 0, P = ST.normCdf(z);
      d = Math.max(d, Math.abs(Fk - P), Math.abs(Fm - P));
    }
    return d;
  }
  let best = 0, bz = 0;
  for (let i = 0; i <= 1400; i++) { const z = -6 + i * 0.01, v = Math.abs(C.cdf(z) - ST.normCdf(z)); if (v > best) { best = v; bz = z; } }
  for (let i = -100; i <= 100; i++) { const z = bz + i * 0.0002, v = Math.abs(C.cdf(z) - ST.normCdf(z)); if (v > best) best = v; }
  return best;
};
LT.cltLadder = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100];
(function () {
  const svg = d3.select("#clt-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("clt-readout"), cache = {};
  let key = "exp", n = 10;
  const supCurve = k => cache[k] || (cache[k] = LT.cltLadder.map(m => ({ n: m, d: LT.cltSup(k, m) })));
  function draw() {
    svg.selectAll("*").remove();
    const F = LT.cltFam[key], g = svg.append("g"), C = LT.cltCdf(key, n);
    const x = d3.scaleLinear().domain([-4, 4.5]).range([46, 380]);
    const zs = d3.range(-4, 4.5001, 0.02);
    let ymax = 0.55;
    if (!C.lattice) ymax = Math.max(0.45, Math.min(1.2, d3.max(zs, z => C.pdf(z)) * 1.08));
    else ymax = Math.max(0.45, Math.min(1.6, d3.max(d3.range(0, n + 1), k => C.pmf(k) * C.s) * 1.08));
    const y = d3.scaleLinear().domain([0, ymax]).range([300, 24]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,300)").call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").attr("transform", "translate(46,0)").call(d3.axisLeft(y).ticks(5));
    if (C.lattice) {
      const w = 1 / C.s;
      for (let k = 0; k <= n; k++) {
        const z = (k - C.m) / C.s;
        if (z < -4.2 || z > 4.7) continue;
        const hgt = Math.min(C.pmf(k) * C.s, ymax), xl = Math.max(46, x(z - w / 2)), xr = Math.min(380, x(z + w / 2));
        if (xr > xl) g.append("rect").attr("x", xl).attr("width", Math.max(0.6, xr - xl - 0.6)).attr("y", y(hgt)).attr("height", 300 - y(hgt)).attr("fill", SC.a2).attr("fill-opacity", 0.75);
      }
    } else {
      g.append("path").attr("d", d3.area().x(z => x(z)).y0(300).y1(z => y(Math.min(ymax, Math.max(0, C.pdf(z)))))(zs)).attr("fill", SC.a2).attr("fill-opacity", 0.5).attr("stroke", SC.a2).attr("stroke-width", 1.5);
    }
    g.append("path").attr("d", d3.line().x(z => x(z)).y(z => y(ST.normPdf(z)))(zs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.8);
    LT.label(g, 50, 14, `orange: exact density of Zₙ for ${F.name} terms, n = ${n} · white: φ`, { size: 10.5 });
    LT.label(g, 380, 330, "z", { anchor: "end" });
    // right: log–log sup distance
    const curve = supCurve(key), dNow = LT.cltSup(key, n);
    const be = m => 0.469 * F.rho / Math.sqrt(m), ed = m => Math.abs(F.g1) / (6 * Math.sqrt(2 * Math.PI * m));
    const lo = Math.max(1e-5, Math.min(d3.min(curve, d => d.d), dNow) * 0.5);
    const xr = d3.scaleLog().domain([1, 100]).range([440, 665]), yr = d3.scaleLog().domain([lo, 1]).range([300, 42]).clamp(true);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,300)").call(d3.axisBottom(xr).ticks(3, "~s"));
    g.append("g").attr("class", "axis").attr("transform", "translate(440,0)").call(d3.axisLeft(yr).ticks(4, "~g"));
    const ms = d3.range(0, 101).map(i => Math.pow(10, 2 * i / 100));
    g.append("path").attr("d", d3.line().x(m => xr(m)).y(m => yr(Math.min(1, be(m))))(ms)).attr("fill", "none").attr("stroke", SC.bad).attr("stroke-dasharray", "6 3").attr("stroke-width", 1.5);
    if (F.g1 !== 0) g.append("path").attr("d", d3.line().x(m => xr(m)).y(m => yr(ed(m)))(ms)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-dasharray", "2 3").attr("stroke-width", 1.8);
    g.append("path").attr("d", d3.line().x(d => xr(d.n)).y(d => yr(d.d))(curve)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
    curve.forEach(d => g.append("circle").attr("cx", xr(d.n)).attr("cy", yr(d.d)).attr("r", 2.2).attr("fill", SC.a2));
    g.append("circle").attr("cx", xr(n)).attr("cy", yr(dNow)).attr("r", 5.5).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 2);
    LT.label(g, 444, 14, "sup |Fₙ − Φ| (orange), Berry–Esseen (red),", { size: 10.5 });
    LT.label(g, 444, 27, F.g1 !== 0 ? "Edgeworth |γ₁|/(6√(2πn)) (green)" : "Edgeworth term is 0: γ₁ = 0", { size: 10.5 });
    LT.label(g, 665, 330, "n (log)", { anchor: "end" });
    const B = be(n), g3 = v => (v < 0.001 ? v.toExponential(2) : LT.f(v, 4));
    out.innerHTML = `${F.name}, n = ${n}: skewness of Zₙ = γ₁/√n = <b>${LT.f(F.g1 / Math.sqrt(n), 3)}</b> · excess kurtosis γ₂/n = <b>${LT.f(F.g2 / n, 3)}</b>` +
      `<br>exact sup |P(Zₙ ≤ z) − Φ(z)| = <b>${g3(dNow)}</b> · Berry–Esseen 0.4690·ρ/(σ³√n) = <b>${LT.f(B, 4)}</b> (ρ/σ³ = ${LT.f(F.rho, 4)}) · bound/actual = ${LT.f(B / dNow, 3)}` +
      (F.g1 !== 0 ? ` · Edgeworth estimate ${LT.f(ed(n), 4)}` : "") +
      (C.lattice ? `<br>a lattice: the cdf jumps by up to ${LT.f(d3.max(d3.range(0, n + 1), k => C.pmf(k)), 4)}, and no smooth curve can be closer than half a jump` : "");
  }
  $("cl-pop").addEventListener("change", e => { key = e.target.value; draw(); });
  $("cl-n").addEventListener("input", e => { n = +e.target.value; $("cl-nv").textContent = n; draw(); });
  $("cl-reset").addEventListener("click", () => { key = "exp"; n = 10; $("cl-pop").value = "exp"; $("cl-n").value = 10; $("cl-nv").textContent = "10"; draw(); });
  draw();
})();

/* ─────────────────── 4 · when the CLT refuses: Pareto / Cauchy means ───────────────────
   Caption audit: Pareto α = 1.5 (minimum 1, mean 3, infinite variance), 2,000 runs × 1,000 draws, seed
   LT.seeds.stable, n = 100 → the fitted log–log slope of IQR(X̄ₙ) over n = 10..1,000 (theory −1/3) and
   the robust-standardised histogram's skewness are printed by the build check and recorded.          */
LT.stLadder = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000];
LT.stableSim = function (pop, a, seed, M) {         // X̄ₙ at the ladder points, M independent runs of 1,000 draws
  const r = ST.rng(seed), L = LT.stLadder, res = L.map(() => new Float64Array(M));
  for (let j = 0; j < M; j++) {
    let s = 0, li = 0;
    for (let m = 1; m <= 1000; m++) {
      s += pop === "cauchy" ? PV.cauchyDraw(0, 1, r) : PV.paretoDraw(a, 1, r);
      if (m === L[li]) { res[li][j] = s / m; li++; }
    }
  }
  return res.map(v => Float64Array.from(v).sort());
};
LT.stableSlope = function (S) {                     // least-squares slope of log IQR on log n, n = 10..1000
  const pts = [];
  LT.stLadder.forEach((n, i) => { if (n >= 10) pts.push([Math.log(n), Math.log(LT.quantSorted(S[i], 0.75) - LT.quantSorted(S[i], 0.25))]); });
  const mx = d3.mean(pts, p => p[0]), my = d3.mean(pts, p => p[1]);
  return d3.sum(pts, p => (p[0] - mx) * (p[1] - my)) / d3.sum(pts, p => (p[0] - mx) ** 2);
};
LT.stableTheory = (pop, a) => (pop === "cauchy" ? 0 : a > 2 ? -0.5 : a === 2 ? -0.5 : -(1 - 1 / a));
(function () {
  const svg = d3.select("#stable-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("stable-readout"), M = 2000;
  let pop = "pareto", a = 1.5, nSel = 100, seed = LT.seeds.stable, S = null;
  const sims = () => { S = LT.stableSim(pop, a, seed, M); };
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), L = LT.stLadder;
    const iqr = L.map((n, i) => ({ n, v: LT.quantSorted(S[i], 0.75) - LT.quantSorted(S[i], 0.25) }));
    const x = d3.scaleLog().domain([1, 1000]).range([50, 320]);
    const i10 = L.indexOf(10), anchor = iqr[i10].v, th = LT.stableTheory(pop, a);
    const ends = [-0.5, th].flatMap(sl => [anchor * Math.pow(0.1, sl), anchor * Math.pow(100, sl)]);   // reference lines at n = 1 and 1000
    const vmin = Math.min(d3.min(iqr, d => d.v), d3.min(ends)), vmax = Math.max(d3.max(iqr, d => d.v), d3.max(ends));
    const y = d3.scaleLog().domain([vmin / 1.3, vmax * 1.3]).range([290, 24]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(x).ticks(3, "~s"));
    g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(y).ticks(4, "~g"));
    const ref = (sl, col, dash) => g.append("path").attr("d", d3.line().x(n => x(n)).y(n => y(anchor * Math.pow(n / 10, sl)))([1, 1000])).attr("fill", "none").attr("stroke", col).attr("stroke-dasharray", dash).attr("stroke-width", 1.5);
    ref(-0.5, SC.ink, "5 3");
    if (Math.abs(th + 0.5) > 1e-9) ref(th, SC.a2, "2 3");
    g.append("path").attr("d", d3.line().x(d => x(d.n)).y(d => y(d.v))(iqr)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    iqr.forEach(d => g.append("circle").attr("cx", x(d.n)).attr("cy", y(d.v)).attr("r", d.n === nSel ? 5 : 3).attr("fill", d.n === nSel ? SC.a2 : SC.accent));
    LT.label(g, 54, 14, `IQR of X̄ₙ (blue) · slope −½ (white) ${Math.abs(th + 0.5) > 1e-9 ? `· slope ${th.toFixed(2)} (orange)` : ""}`, { size: 10.5 });
    LT.label(g, 185, 322, "n (log scale)", { anchor: "middle" });
    // right: robust-standardised histogram at n
    const s = S[L.indexOf(nSel)], med = LT.quantSorted(s, 0.5), iq = LT.quantSorted(s, 0.75) - LT.quantSorted(s, 0.25);
    const zsc = Array.from(s, v => (v - med) / (iq / 1.349));
    const H = LT.hist(zsc, -4, 6, 50);
    const xr = d3.scaleLinear().domain([-4, 6]).range([380, 665]);
    const yr = d3.scaleLinear().domain([0, Math.max(0.45, d3.max(H.bins, b => b.d) * 1.1)]).range([290, 24]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(xr).ticks(6));
    g.append("g").attr("class", "axis").attr("transform", "translate(380,0)").call(d3.axisLeft(yr).ticks(4));
    H.bins.forEach(b => g.append("rect").attr("x", xr(b.x0) + 0.5).attr("width", xr(b.x1) - xr(b.x0) - 1).attr("y", yr(b.d)).attr("height", 290 - yr(b.d)).attr("fill", SC.accent).attr("fill-opacity", 0.7));
    const zs = d3.range(-4, 6.001, 0.05);
    g.append("path").attr("d", d3.line().x(z => xr(z)).y(z => yr(ST.normPdf(z)))(zs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.8);
    LT.label(g, 384, 14, `(X̄ₙ − median)/(IQR/1.349) at n = ${nSel}, against φ`, { size: 10.5 });
    LT.label(g, 665, 320, "robust z", { anchor: "end" });
    const slope = LT.stableSlope(S), tailFrac = H.out / s.length;
    const q10 = LT.quantSorted(s, 0.1), q90 = LT.quantSorted(s, 0.9), qsk = ((q90 - med) - (med - q10)) / (q90 - q10);
    const meds = [10, 100, 1000].map(n => LT.quantSorted(S[L.indexOf(n)], 0.5));
    const popTxt = pop === "cauchy" ? "Cauchy (no mean, α = 1)" :
      `Pareto α = ${a.toFixed(2)}: ${a > 1 ? `mean ${LT.f(a / (a - 1), 3)}` : "no finite mean"}, ${a > 2 ? `variance ${LT.f(a / ((a - 1) ** 2 * (a - 2)), 3)}` : "infinite variance"}`;
    const thTxt = pop === "cauchy" ? "0 (the IQR of X̄ₙ is exactly 2 for every n)" : a > 2 ? "−½ (CLT)" : a === 2 ? "−½, with a √(ln n) correction" : a > 1 ? `−(1 − 1/α) = ${th.toFixed(3)} (stable law; a limiting slope, approached slowly)` : `${th.toFixed(3)} ≥ 0: no mean, the average does not settle`;
    out.innerHTML = `${popTxt} · fitted slope of log IQR(X̄ₙ) on log n, n = 10…1000: <b>${LT.f(slope, 3)}</b> · theory ${thTxt}` +
      `<br>at n = ${nSel}: quantile skewness (q₉₀ + q₁₀ − 2·median)/(q₉₀ − q₁₀) = <b>${LT.f(qsk, 3)}</b> (0 for a normal); ${LT.f(100 * tailFrac, 1)}% of runs fall outside the plotted range [−4, 6]` +
      `<br>median of X̄ₙ at n = 10, 100, 1000: ${meds.map(v => LT.f(v, 3)).join(", ")}` + (pop === "pareto" && a > 1 ? ` (the mean is ${LT.f(a / (a - 1), 3)}; for heavy right tails the median of X̄ₙ approaches it from below)` : "");
  }
  const sync = () => { $("st-aw").style.display = pop === "cauchy" ? "none" : ""; };
  $("st-pop").addEventListener("change", e => { pop = e.target.value; sync(); sims(); draw(); });
  $("st-a").addEventListener("input", e => { a = +e.target.value; $("st-av").textContent = a.toFixed(2); sims(); draw(); });
  $("st-n").addEventListener("change", e => { nSel = +e.target.value; draw(); });
  $("st-sim").addEventListener("click", () => { seed += 1; sims(); draw(); });
  $("st-reset").addEventListener("click", () => {
    pop = "pareto"; a = 1.5; nSel = 100; seed = LT.seeds.stable;
    $("st-pop").value = "pareto"; $("st-a").value = 1.5; $("st-av").textContent = "1.50"; $("st-n").value = "100";
    sync(); sims(); draw();
  });
  sync(); sims(); draw();
})();

/* ─────────────────── 5 · the delta-method lens ───────────────────
   Caption audit: g = ln, μ = 1, s = 0.2 → delta: N(0, 0.2²); exact ln Y (Y ~ N(1, 0.04), mass below 0
   2.9e-7 excluded): mean −0.0214, SD 0.2120, skewness −0.74, sup cdf distance 0.031 (quadrature check
   in the build scratchpad). Deterministic: no seed.                                                  */
LT.dlFns = {
  ln: { g: y => Math.log(y), d: y => 1 / y, ok: y => y > 0, mu: [0.3, 3], s: [0.01, 0.5], def: [1, 0.2], name: "ln y" },
  exp: { g: y => Math.exp(y), d: y => Math.exp(y), ok: () => true, mu: [-1, 2], s: [0.01, 0.6], def: [0, 0.3], name: "eʸ" },
  inv: { g: y => 1 / y, d: y => -1 / (y * y), ok: y => y > 0, mu: [0.3, 3], s: [0.01, 0.4], def: [1, 0.15], name: "1/y" },
  sq: { g: y => y * y, d: y => 2 * y, ok: () => true, mu: [-1.5, 2], s: [0.02, 0.5], def: [1, 0.2], name: "y²", d2: 2 },
  odds: { g: y => y / (1 - y), d: y => 1 / ((1 - y) * (1 - y)), ok: y => y > 0 && y < 1, mu: [0.1, 0.9], s: [0.005, 0.12], def: [0.5, 0.05], name: "y/(1 − y)" },
  asin: { g: y => 2 * Math.asin(Math.sqrt(y)), d: y => 1 / Math.sqrt(y * (1 - y)), ok: y => y > 0 && y < 1, mu: [0.05, 0.95], s: [0.005, 0.12], def: [0.5, 0.05], name: "2·arcsin √y" }
};
LT.deltaExact = function (key, mu, s, K) {          // push N(μ, s²) through g on K cells of μ ± 7s
  const G = LT.dlFns[key], cells = [], h = 14 * s / K;
  let lost = 0;
  for (let i = 0; i < K; i++) {
    const a = mu - 7 * s + i * h, m = ST.normCdf((a + h - mu) / s) - ST.normCdf((a - mu) / s), y = a + h / 2;
    if (G.ok(y)) cells.push({ v: G.g(y), m }); else lost += m;
  }
  const tot = d3.sum(cells, c => c.m);
  cells.forEach(c => { c.m /= tot; });
  const mean = d3.sum(cells, c => c.v * c.m), v2 = d3.sum(cells, c => (c.v - mean) ** 2 * c.m), v3 = d3.sum(cells, c => (c.v - mean) ** 3 * c.m);
  cells.sort((p, q) => p.v - q.v);
  const dm = G.g(mu), ds = Math.abs(G.d(mu)) * s;
  let cum = 0, ks = 0;
  for (const c of cells) {                          // sup over both sides of each atom of the discretised law
    const P = ds > 0 ? ST.normCdf((c.v - dm) / ds) : (c.v >= dm ? 1 : 0);
    ks = Math.max(ks, Math.abs(cum - P)); cum += c.m; ks = Math.max(ks, Math.abs(cum - P));
  }
  return { cells, mean, sd: Math.sqrt(v2), skew: v3 / Math.pow(v2, 1.5), ks, lost, dm, ds };
};
(function () {
  const svg = d3.select("#delta-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("delta-readout");
  let key = "ln", mu = 1, s = 0.2;
  const toV = (r, v) => Math.round(1000 * (v - r[0]) / (r[1] - r[0])), fromV = (r, v) => r[0] + (r[1] - r[0]) * v / 1000;
  function syncSliders() {
    const G = LT.dlFns[key];
    $("dl-mu").value = toV(G.mu, mu); $("dl-s").value = toV(G.s, s);
    $("dl-muv").textContent = mu.toFixed(2); $("dl-sv").textContent = s.toFixed(3);
  }
  function draw() {
    svg.selectAll("*").remove();
    const G = LT.dlFns[key], g = svg.append("g"), E = LT.deltaExact(key, mu, s, 20000);
    let xlo = mu - 4 * s, xhi = mu + 4 * s;
    if (key === "ln" || key === "inv") xlo = Math.max(xlo, 0.02 * mu);
    if (key === "odds" || key === "asin") { xlo = Math.max(xlo, 0.002); xhi = Math.min(xhi, 0.998); }
    const x = d3.scaleLinear().domain([xlo, xhi]).range([175, 660]);
    // vertical range: central 99.8% of the exact law, and ±4 delta sd
    let c = 0, qlo = null, qhi = null;
    for (const cl of E.cells) { c += cl.m; if (qlo === null && c >= 0.001) qlo = cl.v; if (qhi === null && c >= 0.999) qhi = cl.v; }
    let ylo = Math.min(qlo, E.dm - 4 * E.ds), yhi = Math.max(qhi, E.dm + 4 * E.ds);
    const pad = 0.06 * (yhi - ylo || 1); ylo -= pad; yhi += pad;
    const y = d3.scaleLinear().domain([ylo, yhi]).range([268, 22]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,268)").call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "axis").attr("transform", "translate(175,0)").call(d3.axisLeft(y).ticks(6));
    // g and its tangent
    const xs = d3.range(0, 801).map(i => xlo + (xhi - xlo) * i / 800).filter(G.ok);
    const inY = v => v >= ylo && v <= yhi, tan = t => E.dm + G.d(mu) * (t - mu);
    g.append("path").attr("d", d3.line().defined(t => inY(G.g(t))).x(t => x(t)).y(t => y(G.g(t)))(xs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 2.2);
    g.append("path").attr("d", d3.line().defined(t => inY(tan(t))).x(t => x(t)).y(t => y(tan(t)))(xs)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-dasharray", "6 4").attr("stroke-width", 1.6);
    g.append("line").attr("x1", x(mu)).attr("x2", x(mu)).attr("y1", 268).attr("y2", y(E.dm)).attr("stroke", SC.muted).attr("stroke-dasharray", "2 3");
    g.append("line").attr("x1", 175).attr("x2", x(mu)).attr("y1", y(E.dm)).attr("y2", y(E.dm)).attr("stroke", SC.muted).attr("stroke-dasharray", "2 3");
    g.append("circle").attr("cx", x(mu)).attr("cy", y(E.dm)).attr("r", 4).attr("fill", SC.good);
    // the normal of Y, hanging below the x-axis
    const fy = t => ST.normPdf((t - mu) / s) / s, fmax = fy(mu);
    const hy = d3.scaleLinear().domain([0, fmax]).range([290, 350]);
    const ts = d3.range(0, 301).map(i => xlo + (xhi - xlo) * i / 300);
    g.append("path").attr("d", d3.area().x(t => x(t)).y0(290).y1(t => hy(fy(t)))(ts)).attr("fill", SC.accent).attr("fill-opacity", 0.45).attr("stroke", SC.accent);
    LT.label(g, 660, 372, `Yₙ ~ N(μ = ${mu.toFixed(2)}, s = ${s.toFixed(3)})`, { anchor: "end", color: SC.accent, size: 10.5 });
    // exact density of g(Y) (histogram of the pushed cells), and the delta normal, to the left
    const nb = 90, bw = (yhi - ylo) / nb, dens = new Array(nb).fill(0);
    for (const cl of E.cells) { const k = Math.floor((cl.v - ylo) / bw); if (k >= 0 && k < nb) dens[k] += cl.m / bw; }
    const dn = v => (E.ds > 0 ? ST.normPdf((v - E.dm) / E.ds) / E.ds : 0);
    const vs = d3.range(0, 301).map(i => ylo + (yhi - ylo) * i / 300);
    const dmax = Math.max(d3.max(dens), E.ds > 0 ? dn(E.dm) : 0) * 1.05;
    const hx = d3.scaleLinear().domain([0, dmax]).range([132, 20]);
    g.append("path").attr("d", d3.area().curve(d3.curveStepAfter).y((d, i) => y(ylo + i * bw)).x0(132).x1(d => hx(d))(dens.concat([0]))).attr("fill", SC.a2).attr("fill-opacity", 0.55).attr("stroke", SC.a2);
    if (E.ds > 1e-9) g.append("path").attr("d", d3.line().y(v => y(v)).x(v => hx(Math.min(dmax, dn(v))))(vs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-dasharray", "5 3").attr("stroke-width", 1.6);
    const second = key === "sq" && Math.abs(mu) < 0.05;
    if (second) {                                    // the second-order limit: g(μ) + s²·χ²₁
      const ch = v => (v > E.dm ? ST.chi2Pdf((v - E.dm) / (s * s), 1) / (s * s) : 0);
      g.append("path").attr("d", d3.line().y(v => y(v)).x(v => hx(Math.min(dmax, ch(v))))(vs.filter(v => v > E.dm + 1e-6))).attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 1.8);
    }
    LT.label(g, 20, 14, `orange: exact law of g(Yₙ) · dashed: delta normal${second ? " · violet: g(μ) + s²χ²₁" : ""}`, { size: 10.5 });
    LT.label(g, 660, 14, `g(y) = ${G.name} (white), tangent at μ (green)`, { anchor: "end", size: 10.5 });
    const ratio = E.ds > 0 ? E.sd / E.ds : Infinity;
    out.innerHTML = `g = ${G.name}, μ = ${mu.toFixed(2)}, s = ${s.toFixed(3)}: delta method g(Yₙ) ≈ N(${LT.f(E.dm, 4)}, ${LT.f(E.ds, 4)}²) with |g′(μ)| = ${LT.f(Math.abs(G.d(mu)), 4)}` +
      `<br>exact: mean <b>${LT.f(E.mean, 4)}</b> (bias ${LT.f(E.mean - E.dm, 4)}), SD <b>${LT.f(E.sd, 4)}</b> (delta ${LT.f(E.ds, 4)}, ratio ${isFinite(ratio) ? LT.f(ratio, 3) : "∞"}), skewness <b>${LT.f(E.skew, 3)}</b> · largest cdf distance to the delta normal <b>${LT.f(E.ks, 3)}</b>` +
      (E.lost > 1e-12 ? `<br>P(Yₙ outside the domain of g) = ${E.lost.toExponential(2)} (excluded and renormalised)` : "") +
      (second ? "<br>g′(μ) ≈ 0: the first-order method degenerates; the curvature takes over, n(g(Yₙ) − g(μ)) ⇝ ½g″σ²χ²₁" : "");
  }
  $("dl-g").addEventListener("change", e => { key = e.target.value; [mu, s] = LT.dlFns[key].def; syncSliders(); draw(); });
  $("dl-mu").addEventListener("input", e => { mu = fromV(LT.dlFns[key].mu, +e.target.value); $("dl-muv").textContent = mu.toFixed(2); draw(); });
  $("dl-s").addEventListener("input", e => { s = fromV(LT.dlFns[key].s, +e.target.value); $("dl-sv").textContent = s.toFixed(3); draw(); });
  $("dl-reset").addEventListener("click", () => { key = "ln"; mu = 1; s = 0.2; $("dl-g").value = "ln"; syncSliders(); draw(); });
  syncSliders(); draw();
})();

/* ─────────────────── 6 · assembling the t statistic ───────────────────
   Caption audit: data "exp", n = 20, 5,000 samples and 30 s-paths, seed LT.seeds.ts → coverage of
   |T| ≤ 1.96 and of |T| ≤ t*(19) = 2.093, the two tail-miss rates and the skewness of T (first-order
   −2γ₁/√n = −0.894) are printed by the build check and recorded in the caption. Large-simulation
   truth (400,000 samples): coverage 0.919, tails 0.076 / 0.006, skewness −1.09.                     */
LT.tsPop = {
  exp: { mu: 1, sd: 1, g1: 2, draw: r => PV.expDraw(1, r), name: "Exp(1)" },
  norm: { mu: 0, sd: 1, g1: 0, draw: r => ST.randn(r), name: "N(0, 1)" },
  unif: { mu: 0.5, sd: Math.sqrt(1 / 12), g1: 0, draw: r => r(), name: "U(0, 1)" },
  lnorm: { mu: Math.exp(0.5), sd: Math.sqrt((Math.E - 1) * Math.E), g1: (Math.E + 2) * Math.sqrt(Math.E - 1), draw: r => Math.exp(ST.randn(r)), name: "lognormal(0, 1)" }
};
LT.tsSim = function (key, n, seed, M) {
  const F = LT.tsPop[key], r = ST.rng(seed), T = new Float64Array(M), Z = new Float64Array(M), ratio = new Float64Array(M);
  for (let j = 0; j < M; j++) {
    let m = 0, q = 0;
    for (let i = 1; i <= n; i++) { const x = F.draw(r), d = x - m; m += d / i; q += d * (x - m); }
    const sd = Math.sqrt(q / (n - 1));
    Z[j] = Math.sqrt(n) * (m - F.mu) / F.sd; T[j] = Math.sqrt(n) * (m - F.mu) / sd; ratio[j] = sd / F.sd;
  }
  return { T, Z, ratio };
};
LT.tsPaths = function (key, seed, P, N) {
  const F = LT.tsPop[key], r = ST.rng(seed + 7919), out = [];
  for (let p = 0; p < P; p++) {
    const a = new Float64Array(N + 1); let m = 0, q = 0;
    for (let i = 1; i <= N; i++) { const x = F.draw(r), d = x - m; m += d / i; q += d * (x - m); if (i >= 2) a[i] = Math.sqrt(q / (i - 1)) / F.sd; }
    out.push(a);
  }
  return out;
};
LT.tsSummary = function (S, n) {
  const ts = ST.tQuant(0.975, n - 1), M = S.T.length;
  let c196 = 0, ct = 0, lo = 0, hi = 0;
  for (const t of S.T) { if (Math.abs(t) <= 1.96) c196++; if (Math.abs(t) <= ts) ct++; if (t < -ts) lo++; if (t > ts) hi++; }
  return { tstar: ts, c196: c196 / M, ct: ct / M, lo: lo / M, hi: hi / M, skT: LT.mom(S.T).skew, skZ: LT.mom(S.Z).skew };
};
(function () {
  const svg = d3.select("#tstat-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("tstat-readout"), M = 5000, NP = 200;
  let key = "exp", n = 20, seed = LT.seeds.ts, S = null, paths = null;
  const simAll = () => { S = LT.tsSim(key, n, seed, M); paths = LT.tsPaths(key, seed, 30, NP); };
  function draw() {
    svg.selectAll("*").remove();
    const F = LT.tsPop[key], g = svg.append("g");
    // left: s/σ paths
    const x = d3.scaleLog().domain([2, NP]).range([46, 270]);
    const y = d3.scaleLinear().domain([0, 2.5]).range([290, 24]).clamp(true);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(x).tickValues([2, 5, 10, 20, 50, 100, 200]).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis").attr("transform", "translate(46,0)").call(d3.axisLeft(y).ticks(5));
    const ms = d3.range(2, NP + 1);
    paths.forEach(a => g.append("path").attr("d", d3.line().x(m => x(m)).y(m => y(a[m]))(ms)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-opacity", 0.4));
    g.append("line").attr("x1", 46).attr("x2", 270).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", SC.ink).attr("stroke-dasharray", "4 3");
    g.append("line").attr("x1", x(n)).attr("x2", x(n)).attr("y1", 24).attr("y2", 290).attr("stroke", SC.a2).attr("stroke-width", 1.5);
    LT.label(g, 50, 14, `sₘ/σ for 30 streams of ${F.name} data → 1`, { size: 10.5 });
    LT.label(g, 158, 322, "m (log scale)", { anchor: "middle" });
    // right: T histogram, Z outline, φ, t(n − 1)
    const lo = -6, hi = 5, nb = 44;
    const HT = LT.hist(S.T, lo, hi, nb), HZ = LT.hist(S.Z, lo, hi, nb);
    const xr = d3.scaleLinear().domain([lo, hi]).range([320, 665]);
    const ymax = Math.max(0.45, d3.max(HT.bins, b => b.d), d3.max(HZ.bins, b => b.d)) * 1.08;
    const yr = d3.scaleLinear().domain([0, ymax]).range([290, 24]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(xr).ticks(8));
    g.append("g").attr("class", "axis").attr("transform", "translate(320,0)").call(d3.axisLeft(yr).ticks(4));
    HT.bins.forEach(b => g.append("rect").attr("x", xr(b.x0) + 0.5).attr("width", xr(b.x1) - xr(b.x0) - 1).attr("y", yr(b.d)).attr("height", 290 - yr(b.d)).attr("fill", SC.a2).attr("fill-opacity", 0.6));
    g.append("path").attr("d", d3.line().curve(d3.curveStepAfter).x(b => xr(b.x0)).y(b => yr(b.d))(HZ.bins.concat([{ x0: hi, d: 0 }]))).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.6);
    const zs = d3.range(lo, hi + 0.001, 0.05);
    g.append("path").attr("d", d3.line().x(z => xr(z)).y(z => yr(ST.normPdf(z)))(zs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.8);
    g.append("path").attr("d", d3.line().x(z => xr(z)).y(z => yr(Math.min(ymax, ST.tPdf(z, n - 1))))(zs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-dasharray", "5 3").attr("stroke-width", 1.4);
    const U = LT.tsSummary(S, n);
    [-U.tstar, U.tstar].forEach(t => g.append("line").attr("x1", xr(t)).attr("x2", xr(t)).attr("y1", 24).attr("y2", 290).attr("stroke", SC.muted).attr("stroke-dasharray", "2 3"));
    LT.label(g, 324, 14, `orange: Tₙ · blue: Zₙ (σ known) · white: φ · dashed: t(${n - 1})`, { size: 10.5 });
    LT.label(g, 665, 320, "t", { anchor: "end" });
    out.innerHTML = `${F.name}, n = ${n}, 5,000 samples: P(|Tₙ| ≤ 1.96) = <b>${LT.f(U.c196, 3)}</b> · P(|Tₙ| ≤ t* = ${LT.f(U.tstar, 3)}) = <b>${LT.f(U.ct, 3)}</b> (nominal 0.95)` +
      ` · misses: lower tail <b>${LT.f(U.lo, 3)}</b>, upper tail <b>${LT.f(U.hi, 3)}</b> (0.025 each)` +
      `<br>skewness of Tₙ <b>${LT.f(U.skT, 3)}</b> (first order −2γ₁/√n = ${LT.f(-2 * F.g1 / Math.sqrt(n), 3)}) · skewness of Zₙ ${LT.f(U.skZ, 3)} (γ₁/√n = ${LT.f(F.g1 / Math.sqrt(n), 3)})` +
      ` · mean of s/σ ${LT.f(d3.mean(S.ratio), 3)}` +
      (n <= 4 ? `<br>at n = ${n} the reference law t(${n - 1}) has no finite skewness (it needs more than 3 degrees of freedom), so the sample skewness of Tₙ is dominated by a few huge values` : "");
  }
  $("ts-pop").addEventListener("change", e => { key = e.target.value; simAll(); draw(); });
  $("ts-n").addEventListener("input", e => { n = +e.target.value; $("ts-nv").textContent = n; S = LT.tsSim(key, n, seed, M); draw(); });
  $("ts-sim").addEventListener("click", () => { seed += 1; simAll(); draw(); });
  $("ts-reset").addEventListener("click", () => { key = "exp"; n = 20; seed = LT.seeds.ts; $("ts-pop").value = "exp"; $("ts-n").value = 20; $("ts-nv").textContent = "20"; simAll(); draw(); });
  simAll(); draw();
})();

/* ─────────────────── 7 · the sample quantile ───────────────────
   Caption audit: Exp(1), p = 0.5, n = 25 (k = 13), 4,000 samples, seed LT.seeds.quant → asymptotic SD
   1/√25 = 0.2; exact (Rényi) mean 0.7127, SD 0.2019; the simulated mean and SD are printed by the
   build check and recorded in the caption.                                                          */
LT.qtPop = {
  exp: { F: x => (x <= 0 ? 0 : 1 - Math.exp(-x)), f: x => (x < 0 ? 0 : Math.exp(-x)), Q: p => -Math.log(1 - p), draw: r => PV.expDraw(1, r), sdMean: 1, name: "Exp(1)" },
  norm: { F: x => ST.normCdf(x), f: x => ST.normPdf(x), Q: p => ST.normQuant(p), draw: r => ST.randn(r), sdMean: 1, name: "N(0, 1)" },
  unif: { F: x => Math.max(0, Math.min(1, x)), f: x => (x >= 0 && x <= 1 ? 1 : 0), Q: p => p, draw: r => r(), sdMean: Math.sqrt(1 / 12), name: "U(0, 1)" },
  cauchy: { F: x => PV.cauchyCdf(x), f: x => PV.cauchyPdf(x), Q: p => PV.cauchyQuant(p), draw: r => PV.cauchyDraw(0, 1, r), sdMean: NaN, name: "Cauchy" },
  bimod: {
    F: x => 0.5 * ST.normCdf(x + 2) + 0.5 * ST.normCdf(x - 2), f: x => 0.5 * ST.normPdf(x + 2) + 0.5 * ST.normPdf(x - 2),
    Q: p => PV.invert(x => 0.5 * ST.normCdf(x + 2) + 0.5 * ST.normCdf(x - 2), p, -12, 12),
    draw: r => (r() < 0.5 ? -2 : 2) + ST.randn(r), sdMean: Math.sqrt(5), name: "½N(−2, 1) + ½N(2, 1)"
  }
};
LT.qtExact = function (key, p, n) {                 // exact mean and SD of X₍ₖ₎ = Q(U₍ₖ₎), U₍ₖ₎ ~ Beta(k, n − k + 1)
  const P = LT.qtPop[key], k = Math.max(1, Math.ceil(n * p)), b = n - k + 1;
  if (key === "cauchy" && (k < 3 || b < 3)) return { k, mean: NaN, sd: Infinity };
  const lo = PV.betaQuant(1e-9, k, b), hi = PV.betaQuant(1 - 1e-9, k, b);
  const w = u => PV.betaPdf(u, k, b);
  const m1 = PV.simpson(u => P.Q(u) * w(u), lo, hi, 4000), m2 = PV.simpson(u => (P.Q(u) - m1) ** 2 * w(u), lo, hi, 4000);
  return { k, mean: m1, sd: Math.sqrt(m2) };
};
LT.qtSim = function (key, p, n, seed, M) {
  const P = LT.qtPop[key], r = ST.rng(seed), k = Math.max(1, Math.ceil(n * p)), out = new Float64Array(M), buf = new Float64Array(n);
  for (let j = 0; j < M; j++) { for (let i = 0; i < n; i++) buf[i] = P.draw(r); buf.sort(); out[j] = buf[k - 1]; }
  return out;
};
(function () {
  const svg = d3.select("#quant-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("quant-readout"), M = 4000;
  let key = "exp", p = 0.5, n = 25, seed = LT.seeds.quant, Sim = null;
  const sim = () => { Sim = LT.qtSim(key, p, n, seed, M); };
  function draw() {
    svg.selectAll("*").remove();
    const P = LT.qtPop[key], g = svg.append("g"), q = P.Q(p), fq = P.f(q), asd = Math.sqrt(p * (1 - p) / n) / fq;
    const E = LT.qtExact(key, p, n), mm = LT.mom(Sim), sorted = Float64Array.from(Sim).sort();
    const iqrS = LT.quantSorted(sorted, 0.75) - LT.quantSorted(sorted, 0.25);
    const half = 4.2 * Math.max(asd, iqrS / 1.349);
    let lo = q - half, hi = q + half;
    if (key === "exp") lo = Math.max(lo, 0);
    if (key === "unif") { lo = Math.max(lo, 0); hi = Math.min(hi, 1); }
    const H = LT.hist(Sim, lo, hi, 48);
    const x = d3.scaleLinear().domain([lo, hi]).range([50, 665]);
    const k = E.k, b = n - k + 1, exactPdf = t => PV.betaPdf(P.F(t), k, b) * P.f(t);
    const ts = d3.range(0, 401).map(i => lo + (hi - lo) * i / 400);
    const ymax = Math.max(d3.max(H.bins, d => d.d), d3.max(ts, exactPdf), ST.normPdf(0) / asd) * 1.1;
    const y = d3.scaleLinear().domain([0, ymax]).range([290, 24]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").attr("transform", "translate(50,0)").call(d3.axisLeft(y).ticks(5));
    H.bins.forEach(d => g.append("rect").attr("x", x(d.x0) + 0.5).attr("width", Math.max(0.5, x(d.x1) - x(d.x0) - 1)).attr("y", y(d.d)).attr("height", 290 - y(d.d)).attr("fill", SC.accent).attr("fill-opacity", 0.55));
    g.append("path").attr("d", d3.line().x(t => x(t)).y(t => y(Math.min(ymax, exactPdf(t))))(ts)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
    g.append("path").attr("d", d3.line().x(t => x(t)).y(t => y(Math.min(ymax, ST.normPdf((t - q) / asd) / asd)))(ts)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-dasharray", "6 3").attr("stroke-width", 1.6);
    g.append("line").attr("x1", x(q)).attr("x2", x(q)).attr("y1", 24).attr("y2", 290).attr("stroke", SC.good).attr("stroke-width", 1.5);
    LT.label(g, x(q) + 4, 36, `q_p = ${LT.f(q, 4)}`, { color: SC.good, size: 10.5 });
    LT.label(g, 54, 14, `${P.name}: X₍${k}₎ of n = ${n} (p = ${p.toFixed(2)}) · blue: 4,000 simulated · orange: exact · dashed: N(q_p, p(1 − p)/(n f(q_p)²))`, { size: 10.5 });
    out.innerHTML = `${P.name}, p = ${p.toFixed(2)}, n = ${n}, k = ⌈np⌉ = ${k}: f(q_p) = ${LT.f(fq, 4)} · asymptotic SD √(p(1 − p)/n)/f(q_p) = <b>${LT.f(asd, 4)}</b>` +
      `<br>simulated: mean <b>${LT.f(mm.mean, 4)}</b>, SD <b>${LT.f(mm.sd, 4)}</b> · exact: mean ${isFinite(E.mean) ? LT.f(E.mean, 4) : "—"}, SD ${isFinite(E.sd) ? LT.f(E.sd, 4) : "infinite (Cauchy order statistics need 3 ≤ k ≤ n − 2)"}` +
      `<br>for comparison, the sample mean's SE: ${isFinite(P.sdMean) ? LT.f(P.sdMean / Math.sqrt(n), 4) : "none (the Cauchy mean does not exist; X̄ₙ is Cauchy for every n)"}` + (H.out ? ` · ${H.out} of 4,000 values fall outside the plotted range` : "");
  }
  $("qt-pop").addEventListener("change", e => { key = e.target.value; sim(); draw(); });
  $("qt-p").addEventListener("input", e => { p = +e.target.value; $("qt-pv").textContent = p.toFixed(2); sim(); draw(); });
  $("qt-n").addEventListener("input", e => { n = +e.target.value; $("qt-nv").textContent = n; sim(); draw(); });
  $("qt-sim").addEventListener("click", () => { seed += 1; sim(); draw(); });
  $("qt-reset").addEventListener("click", () => {
    key = "exp"; p = 0.5; n = 25; seed = LT.seeds.quant;
    $("qt-pop").value = "exp"; $("qt-p").value = 0.5; $("qt-pv").textContent = "0.50"; $("qt-n").value = 25; $("qt-nv").textContent = "25";
    sim(); draw();
  });
  sim(); draw();
})();

/* ─────────────────── 8 · the maximum versus the mean ───────────────────
   Caption audit: Exp(1), n = 100, 2,000 samples, seed LT.seeds.ext → normalised max M − ln 100: exact
   mean H₁₀₀ − ln 100 = 0.5822, exact sup |F¹⁰⁰ − Λ| = 0.0027; the simulated mean, SD and skewness of the
   normalised max and of the standardised mean are printed by the build check and recorded.         */
LT.exLadder = [2, 5, 10, 20, 50, 100, 200, 500, 1000];
LT.exPop = {
  exp: { mu: 1, sd: 1, draw: r => PV.expDraw(1, r), F: x => (x <= 0 ? 0 : 1 - Math.exp(-x)), f: x => (x < 0 ? 0 : Math.exp(-x)),
    ab: n => [1, Math.log(n)], lim: { cdf: PV.gumbelCdf, pdf: PV.gumbelPdf, range: [-2.5, 6], mean: 0.5772157, sd: Math.PI / Math.sqrt(6), skew: 1.1395, name: "Gumbel" }, name: "Exp(1)", norm: "M − ln n" },
  pareto: { mu: 1.5, sd: Math.sqrt(0.75), draw: r => PV.paretoDraw(3, 1, r), F: x => PV.paretoCdf(x, 3, 1), f: x => PV.paretoPdf(x, 3, 1),
    ab: n => [Math.pow(n, 1 / 3), 0], lim: { cdf: x => PV.frechetCdf(x, 3), pdf: x => PV.frechetPdf(x, 3), range: [0, 5], mean: Math.exp(ST.lnGamma(2 / 3)), sd: Math.sqrt(Math.exp(ST.lnGamma(1 / 3)) - Math.exp(2 * ST.lnGamma(2 / 3))), skew: Infinity, name: "Fréchet(3)" }, name: "Pareto(3)", norm: "M/n^(1/3)" },
  unif: { mu: 0.5, sd: Math.sqrt(1 / 12), draw: r => r(), F: x => Math.max(0, Math.min(1, x)), f: x => (x >= 0 && x <= 1 ? 1 : 0),
    ab: n => [1 / n, 1], lim: { cdf: x => (x >= 0 ? 1 : Math.exp(x)), pdf: x => (x > 0 ? 0 : Math.exp(x)), range: [-6, 0.3], mean: -1, sd: 1, skew: -2, name: "reversed Weibull (−Exp(1))" }, name: "U(0, 1)", norm: "n(M − 1)" },
  norm: { mu: 0, sd: 1, draw: r => ST.randn(r), F: x => ST.normCdf(x), f: x => ST.normPdf(x),
    ab: n => { const L = Math.sqrt(2 * Math.log(n)); return [1 / L, L - (Math.log(Math.log(n)) + Math.log(4 * Math.PI)) / (2 * L)]; },
    lim: { cdf: PV.gumbelCdf, pdf: PV.gumbelPdf, range: [-3, 6], mean: 0.5772157, sd: Math.PI / Math.sqrt(6), skew: 1.1395, name: "Gumbel" }, name: "N(0, 1)", norm: "(M − bₙ)/aₙ" }
};
LT.exExact = function (key, n) {                    // exact law of (Mₙ − bₙ)/aₙ: density aₙ·n·F^(n−1)·f, moments, sup distance to the limit
  const P = LT.exPop[key], [a, b] = P.ab(n), L = P.lim;
  const cdf = x => Math.pow(P.F(b + a * x), n), pdf = x => a * n * Math.pow(P.F(b + a * x), n - 1) * P.f(b + a * x);
  const lo = key === "unif" ? -n : (key === "pareto" ? (1 - b) / a : L.range[0] - 6), hi = key === "unif" ? 0 : L.range[1] + 30;
  const K = 60000, h = (hi - lo) / K;
  let m1 = 0, m2 = 0;
  for (let i = 0; i < K; i++) { const x = lo + (i + 0.5) * h, d = pdf(x) * h; m1 += x * d; m2 += x * x * d; }
  if (key === "pareto") {                           // power tail: continue in log space out to x = 10⁶
    const T = 20000, l0 = Math.log(hi), l1 = Math.log(1e6), dl = (l1 - l0) / T;
    for (let i = 0; i < T; i++) { const x = Math.exp(l0 + (i + 0.5) * dl), d = pdf(x) * x * dl; m1 += x * d; m2 += x * x * d; }
  }
  let sup = 0;
  for (let i = 0; i <= 6000; i++) { const x = L.range[0] - 2 + (L.range[1] - L.range[0] + 4) * i / 6000; sup = Math.max(sup, Math.abs(cdf(x) - L.cdf(x))); }
  return { cdf, pdf, mean: m1, sd: Math.sqrt(Math.max(0, m2 - m1 * m1)), sup };
};
LT.exSim = function (key, n, seed, M) {
  const P = LT.exPop[key], r = ST.rng(seed), [a, b] = P.ab(n), mx = new Float64Array(M), mn = new Float64Array(M);
  for (let j = 0; j < M; j++) {
    let m = -Infinity, s = 0;
    for (let i = 0; i < n; i++) { const x = P.draw(r); if (x > m) m = x; s += x; }
    mx[j] = (m - b) / a; mn[j] = Math.sqrt(n) * (s / n - P.mu) / P.sd;
  }
  return { mx, mn };
};
(function () {
  const svg = d3.select("#extreme-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("extreme-readout"), M = 2000;
  let key = "exp", ni = 5, seed = LT.seeds.ext, S = null;
  const sim = () => { S = LT.exSim(key, LT.exLadder[ni], seed, M); };
  function draw() {
    svg.selectAll("*").remove();
    const P = LT.exPop[key], L = P.lim, n = LT.exLadder[ni], g = svg.append("g"), E = LT.exExact(key, n);
    const [lo, hi] = L.range, H = LT.hist(S.mx, lo, hi, 40);
    const x = d3.scaleLinear().domain([lo, hi]).range([46, 330]);
    const xs = d3.range(0, 301).map(i => lo + (hi - lo) * i / 300);
    const ymax = Math.max(d3.max(H.bins, d => d.d), d3.max(xs, E.pdf), d3.max(xs, L.pdf)) * 1.1;
    const y = d3.scaleLinear().domain([0, ymax]).range([290, 24]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "axis").attr("transform", "translate(46,0)").call(d3.axisLeft(y).ticks(4));
    H.bins.forEach(d => g.append("rect").attr("x", x(d.x0) + 0.5).attr("width", x(d.x1) - x(d.x0) - 1).attr("y", y(d.d)).attr("height", 290 - y(d.d)).attr("fill", SC.a2).attr("fill-opacity", 0.55));
    g.append("path").attr("d", d3.line().x(t => x(t)).y(t => y(Math.min(ymax, E.pdf(t))))(xs)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
    g.append("path").attr("d", d3.line().x(t => x(t)).y(t => y(Math.min(ymax, L.pdf(t))))(xs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-dasharray", "6 3").attr("stroke-width", 1.6);
    LT.label(g, 50, 14, `max: ${P.norm}, n = ${n} · solid exact · dashed ${L.name}`, { size: 10.5 });
    // right: the mean
    const Hm = LT.hist(S.mn, -4, 4, 40), xr = d3.scaleLinear().domain([-4, 4]).range([380, 665]);
    const ym = Math.max(0.45, d3.max(Hm.bins, d => d.d)) * 1.1, yr = d3.scaleLinear().domain([0, ym]).range([290, 24]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,290)").call(d3.axisBottom(xr).ticks(8));
    g.append("g").attr("class", "axis").attr("transform", "translate(380,0)").call(d3.axisLeft(yr).ticks(4));
    Hm.bins.forEach(d => g.append("rect").attr("x", xr(d.x0) + 0.5).attr("width", xr(d.x1) - xr(d.x0) - 1).attr("y", yr(d.d)).attr("height", 290 - yr(d.d)).attr("fill", SC.accent).attr("fill-opacity", 0.55));
    const zs = d3.range(-4, 4.001, 0.05);
    g.append("path").attr("d", d3.line().x(z => xr(z)).y(z => yr(ST.normPdf(z)))(zs)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.8);
    LT.label(g, 384, 14, `mean: √n(X̄ₙ − μ)/σ, same samples · white φ`, { size: 10.5 });
    const A = LT.mom(S.mx), B = LT.mom(S.mn);
    out.innerHTML = `${P.name}, n = ${n}, 2,000 samples — <b>maximum</b> (${P.norm}): simulated mean <b>${LT.f(A.mean, 3)}</b>, SD <b>${LT.f(A.sd, 3)}</b>, skewness <b>${LT.f(A.skew, 2)}</b>` +
      ` · exact mean ${LT.f(E.mean, 4)}, SD ${LT.f(E.sd, 4)} · ${L.name} limit: mean ${LT.f(L.mean, 4)}, SD ${LT.f(L.sd, 4)}, skewness ${isFinite(L.skew) ? LT.f(L.skew, 3) : "infinite (α = 3)"}` +
      `<br>exact sup |P(normalised max ≤ x) − limit cdf| = <b>${E.sup < 1e-3 ? E.sup.toExponential(2) : LT.f(E.sup, 4)}</b>` +
      `<br><b>mean</b>: simulated skewness ${LT.f(B.skew, 3)}, SD ${LT.f(B.sd, 3)} (→ 0 and 1: the CLT) — the maximum keeps its skewed shape and does not shrink; the mean goes normal`;
  }
  const setN = () => { $("ex-nv").textContent = LT.exLadder[ni]; };
  $("ex-pop").addEventListener("change", e => { key = e.target.value; sim(); draw(); });
  $("ex-n").addEventListener("input", e => { ni = +e.target.value; setN(); sim(); draw(); });
  $("ex-sim").addEventListener("click", () => { seed += 1; sim(); draw(); });
  $("ex-reset").addEventListener("click", () => { key = "exp"; ni = 5; seed = LT.seeds.ext; $("ex-pop").value = "exp"; $("ex-n").value = 5; setN(); sim(); draw(); });
  setN(); sim(); draw();
})();
