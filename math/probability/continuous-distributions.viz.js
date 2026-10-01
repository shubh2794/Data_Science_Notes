/* continuous-distributions.viz.js — the eight visualizations on math/probability/continuous-distributions.html.
   Loaded after ../../data.js → ../../notes.js → ../statistics/stats-viz.js (SC palette, ST toolbox)
   → prob-viz.js (PV: discrete helpers plus the continuous families exp/gamma/beta/Pareto/lognormal,
   PV.invert, PV.simpson). Distribution functions come from ST / PV and are never re-implemented here.

   Everything page-local lives under the single namespace CD. Each figure is an IIFE that exits
   quietly if its <svg> is not on the page.

     1  #area-svg       area is probability: draggable interval, densities above 1, halving the window
     2  #pdfcdf-svg     PDF + CDF explorer for eight continuous families, P(a < X ≤ b), quartiles
     3  #memo-svg       memorylessness: renormalised tail vs fresh copy, plus the hazard rate
     4  #geoexp-svg     geometric → exponential as the trial length Δ shrinks, sup gap vs Δ
     5  #gamma-svg      the gamma as the r-th arrival: summed exponential gaps vs the density
     6  #beta-svg       beta–binomial conjugate updating
     7  #tails-svg      Pareto vs exponential vs lognormal: survival on log–log axes, running means
     8  #transform-svg  change of variables: preimage, formula, simulation; probability integral transform */

const CD = {
  // Default seeds were chosen by the caption audit (see the note at each figure).
  seeds: { gamma: 3, tails: 10, transform: 1 },
  fmt: (x, d) => ST.fmt(x, d === undefined ? 4 : d),
  // path through (x, f(x)) with values clipped to [0, ymax] so infinite densities stay in frame
  curve(xs, fn, sx, sy, ymax) {
    return d3.line().defined(d => isFinite(d[1]))
      .x(d => sx(d[0])).y(d => sy(Math.min(ymax, Math.max(0, d[1]))))(xs.map(x => [x, fn(x)]));
  },
  area(xs, fn, sx, sy, ymax) {
    return d3.area().x(d => sx(d[0])).y0(sy(0)).y1(d => sy(Math.min(ymax, Math.max(0, d[1]))))
      (xs.map(x => [x, fn(x)]).filter(d => isFinite(d[1]) || d[1] === Infinity));
  },
  hist(vals, lo, hi, k) {                  // density-scaled histogram on [lo, hi)
    const w = (hi - lo) / k, c = new Array(k).fill(0);
    vals.forEach(v => { const i = Math.floor((v - lo) / w); if (i >= 0 && i < k) c[i]++; });
    return c.map((n, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, d: n / (vals.length * w) }));
  },
  label(g, x, y, t, o) {
    const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", (o && o.size) || 11)
      .attr("fill", (o && o.color) || SC.muted).text(t);
    if (o && o.anchor) e.attr("text-anchor", o.anchor);
    return e;
  }
};

/* ─────────────────── 1 · area is probability ───────────────────
   Caption audit: default U(0, ½), a = 0.1, b = 0.3 → P = 0.4, width 0.2, P/width = 2.000,
   curve at height 2 above the dashed density-1 line. Halving the window keeps P/width at 2
   while P → 0. f = 1/(2√x) is drawn clipped with an "→ ∞" marker.                              */
(function () {
  const svg = d3.select("#area-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, H = 330, $ = id => document.getElementById(id);
  const out = $("area-readout"), sel = $("ar-fam");
  const F = {
    u05: { name: "uniform on (0, ½)", pdf: x => (x >= 0 && x <= 0.5 ? 2 : 0), cdf: x => ST.clamp(2 * x, 0, 1), lo: -0.1, hi: 0.7, ymax: 2.6, a: 0.1, b: 0.3 },
    dart: { name: "dart distance, f = 2x", pdf: x => (x >= 0 && x <= 1 ? 2 * x : 0), cdf: x => (x <= 0 ? 0 : x >= 1 ? 1 : x * x), lo: -0.1, hi: 1.1, ymax: 2.4, a: 0.5, b: 1 },
    sqrt: { name: "f = 1/(2√x)", pdf: x => (x > 0 && x < 1 ? 0.5 / Math.sqrt(x) : 0), cdf: x => (x <= 0 ? 0 : x >= 1 ? 1 : Math.sqrt(x)), lo: -0.05, hi: 1.05, ymax: 6, a: 0.01, b: 0.25 },
    exp: { name: "exponential, f = e^(−x)", pdf: x => PV.expPdf(x, 1), cdf: x => PV.expCdf(x, 1), lo: -0.5, hi: 5, ymax: 1.25, a: 1, b: 2 },
    norm: { name: "standard normal", pdf: x => ST.normPdf(x), cdf: x => ST.normCdf(x), lo: -4, hi: 4, ymax: 1.1, a: -1, b: 1 }
  };
  let fam = "u05", A = 0.1, B = 0.3;
  const m = { l: 50, r: 20, t: 20, b: 34 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const toS = v => Math.round(1000 * (v - F[fam].lo) / (F[fam].hi - F[fam].lo));
  const fromS = s => F[fam].lo + (F[fam].hi - F[fam].lo) * s / 1000;

  function sync() { $("ar-a").value = toS(A); $("ar-b").value = toS(B); $("ar-av").textContent = A.toFixed(4); $("ar-bv").textContent = B.toFixed(4); }
  function draw() {
    const S = F[fam];
    if (A > B) { const t = A; A = B; B = t; }
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleLinear().domain([S.lo, S.hi]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, S.ymax]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    CD.label(g, 0, -6, "density f(x)  — probability per unit length");
    const N = 1200, xs = d3.range(N + 1).map(i => S.lo + (S.hi - S.lo) * i / N);
    // shaded area between A and B
    const xa = Math.max(S.lo, A), xb = Math.min(S.hi, B);
    if (xb > xa) {
      const sx = d3.range(401).map(i => xa + (xb - xa) * i / 400);
      g.append("path").attr("d", CD.area(sx, S.pdf, x, y, S.ymax)).attr("fill", SC.a2).attr("opacity", 0.45);
    }
    g.append("path").attr("d", CD.curve(xs, S.pdf, x, y, S.ymax)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.2);
    if (1 < S.ymax) {
      g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", SC.muted).attr("stroke-dasharray", "5 4");
      CD.label(g, iw - 4, y(1) - 5, "density 1 — not a ceiling", { anchor: "end", size: 10 });
    }
    if (fam === "sqrt") CD.label(g, x(0) + 6, 12, "↑ f → ∞ as x → 0", { color: SC.accent, size: 10 });
    // handles
    [["a", A], ["b", B]].forEach(([nm, v]) => {
      const hx = x(ST.clamp(v, S.lo, S.hi));
      const hg = g.append("g").style("cursor", "ew-resize");
      hg.append("line").attr("x1", hx).attr("x2", hx).attr("y1", 0).attr("y2", ih).attr("stroke", SC.a2).attr("stroke-width", 1.5);
      hg.append("circle").attr("cx", hx).attr("cy", ih).attr("r", 7).attr("fill", SC.a2);
      hg.append("text").attr("x", hx).attr("y", ih + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.bg).text(nm);
      hg.append("rect").attr("x", hx - 9).attr("y", 0).attr("width", 18).attr("height", ih + 10).attr("fill", "transparent");
      hg.call(d3.drag().on("drag", ev => {
        const nv = ST.clamp(x.invert(ev.x), S.lo, S.hi);
        if (nm === "a") A = nv; else B = nv;
        sync(); draw();
      }));
    });
    const P = Math.max(0, S.cdf(B) - S.cdf(A));
    // integrate only over the support; the unbounded 1/(2√x) density goes through x = u², dx = 2u du,
    // whose integrand is 1 on (0, 1) — Simpson never evaluates the singularity (audit fix)
    const sup = { u05: [0, 0.5], dart: [0, 1], sqrt: [0, 1], exp: [0, Infinity], norm: [-Infinity, Infinity] }[fam];
    const lo = Math.max(A, sup[0]), hi = Math.min(B, sup[1]);
    let Pnum = 0;
    if (hi > lo) Pnum = fam === "sqrt"
      ? PV.simpson(u => (u > 0 ? S.pdf(u * u) * 2 * u : 1), Math.sqrt(lo), Math.min(Math.sqrt(hi), 1 - 1e-15), 4000)
      : PV.simpson(S.pdf, lo, hi, 4000);
    const w = B - A, mid = (A + B) / 2;
    CD.label(g, x(mid), y(Math.min(S.ymax * 0.92, Math.max(S.pdf(mid), 0) / 2 + 0.02)) , `P = ${P.toFixed(4)}`, { anchor: "middle", color: SC.ink, size: 12 });
    out.innerHTML = `<b>${S.name}</b> · P(${A.toFixed(4)} &lt; X ≤ ${B.toFixed(4)}) = F(b) − F(a) = <b>${P.toFixed(4)}</b>` +
      ` (numerical integral of f: ${Math.abs(Pnum).toFixed(4)})` +
      `<br>width b − a = ${w.toPrecision(4)} · probability ÷ width = <b>${w > 0 ? (P / w).toFixed(4) : "—"}</b> · density at the midpoint f(${mid.toPrecision(4)}) = <b>${S.pdf(mid).toFixed(4)}</b>` +
      `<br>P(X = a) = 0 for every single value: halve the window repeatedly and P → 0 while P ÷ width → f(midpoint).`;
  }
  ["ar-a", "ar-b"].forEach(id => $(id).addEventListener("input", () => { A = fromS(+$("ar-a").value); B = fromS(+$("ar-b").value); $("ar-av").textContent = A.toFixed(4); $("ar-bv").textContent = B.toFixed(4); draw(); }));
  sel.addEventListener("change", () => { fam = sel.value; A = F[fam].a; B = F[fam].b; sync(); draw(); });
  $("ar-halve").addEventListener("click", () => { const mid = (A + B) / 2, h = (B - A) / 4; A = mid - h; B = mid + h; sync(); draw(); });
  $("ar-reset").addEventListener("click", () => { sel.value = "u05"; fam = "u05"; A = 0.1; B = 0.3; sync(); draw(); });
  sync(); draw();
})();

/* ─────────────────── 2 · PDF + CDF explorer ───────────────────
   Caption audit: default normal(68.3, 1.8), a = 67.4, b = 71.9 → z = −0.5 and 2, P = Φ(2) − Φ(−0.5)
   = 0.6687; numerical mean 68.3000 and variance 3.2400 equal the closed forms.                  */
(function () {
  const svg = d3.select("#pdfcdf-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, $ = id => document.getElementById(id);
  const sel = $("pd-fam"), out = $("pdfcdf-readout");
  const FAM = {
    unif: { p: [["lo", -5, 5, 0.1, 0], ["hi − lo", 0.2, 10, 0.1, 1]],
      make: (a, w) => ({ name: `U(${ST.fmt(a, 1)}, ${ST.fmt(a + w, 1)})`, pdf: x => (x >= a && x <= a + w ? 1 / w : 0), cdf: x => ST.clamp((x - a) / w, 0, 1),
        q: p => a + w * p, E: a + w / 2, V: w * w / 12, dom: [a - 0.15 * w, a + 1.15 * w], map: "lin", lim: [a, a + w] }) },
    exp: { p: [["λ", 0.05, 3, 0.05, 1]],
      make: l => ({ name: `Exp(λ = ${ST.fmt(l, 2)})`, pdf: x => PV.expPdf(x, l), cdf: x => PV.expCdf(x, l), q: p => PV.expQuant(p, l), E: 1 / l, V: 1 / (l * l), map: "log", pos: true }) },
    gamma: { p: [["α", 0.5, 12, 0.1, 3], ["λ", 0.05, 3, 0.05, 1]],
      make: (a, l) => ({ name: `Gamma(α = ${ST.fmt(a, 1)}, λ = ${ST.fmt(l, 2)})`, pdf: x => PV.gammaPdf(x, a, l), cdf: x => PV.gammaCdf(x, a, l), q: p => PV.gammaQuant(p, a, l), E: a / l, V: a / (l * l), map: "log", pos: true }) },
    norm: { p: [["μ", 0, 100, 0.1, 68.3], ["σ", 0.2, 20, 0.1, 1.8]],
      make: (mu, s) => ({ name: `N(μ = ${ST.fmt(mu, 1)}, σ = ${ST.fmt(s, 1)})`, pdf: x => ST.normPdf((x - mu) / s) / s, cdf: x => ST.normCdf((x - mu) / s), q: p => mu + s * ST.normQuant(p), E: mu, V: s * s, map: "lin", lim: [mu - 12 * s, mu + 12 * s], dom: [mu - 4 * s, mu + 4 * s], z: x => (x - mu) / s }) },
    beta: { p: [["a", 0.2, 10, 0.1, 2], ["b", 0.2, 10, 0.1, 5]],
      make: (a, b) => ({ name: `Beta(${ST.fmt(a, 1)}, ${ST.fmt(b, 1)})`, pdf: x => PV.betaPdf(x, a, b), cdf: x => PV.betaCdf(x, a, b), q: p => PV.betaQuant(p, a, b), E: a / (a + b), V: a * b / ((a + b) * (a + b) * (a + b + 1)), map: "logit", dom: [0, 1] }) },
    chi2: { p: [["k", 1, 30, 1, 3]],
      make: k => ({ name: `χ²(k = ${k})`, pdf: x => PV.gammaPdf(x, k / 2, 0.5), cdf: x => PV.gammaCdf(x, k / 2, 0.5), q: p => PV.gammaQuant(p, k / 2, 0.5), E: k, V: 2 * k, map: "log", pos: true }) },
    pareto: { p: [["α", 0.5, 5, 0.1, 1.5]],
      make: al => ({ name: `Pareto(α = ${ST.fmt(al, 1)}, xm = 1)`, pdf: x => PV.paretoPdf(x, al, 1), cdf: x => PV.paretoCdf(x, al, 1), q: p => PV.paretoQuant(p, al, 1),
        E: al > 1 ? al / (al - 1) : Infinity, V: al > 2 ? al / ((al - 1) * (al - 1) * (al - 2)) : Infinity, map: "log", pos: true, xm: 1, heavy: true, al: al }) },
    lnorm: { p: [["μ", -1, 2, 0.1, 0], ["σ", 0.1, 2, 0.05, 1]],
      make: (mu, s) => ({ name: `Lognormal(μ = ${ST.fmt(mu, 1)}, σ = ${ST.fmt(s, 2)})`, pdf: x => PV.lognormPdf(x, mu, s), cdf: x => PV.lognormCdf(x, mu, s), q: p => PV.lognormQuant(p, mu, s),
        E: Math.exp(mu + s * s / 2), V: (Math.exp(s * s) - 1) * Math.exp(2 * mu + s * s), map: "log", pos: true }) }
  };
  let fam = "norm", a = 67.4, b = 71.9, D = null, cur = null;

  function params() { return FAM[fam].p.map((_, i) => +$("pd-p" + (i + 1)).value); }
  function setupSliders(useDefaults) {
    const ps = FAM[fam].p;
    [0, 1].forEach(i => {
      const w = $("pd-p" + (i + 1) + "w");
      if (i >= ps.length) { w.style.display = "none"; return; }
      w.style.display = "";
      const [nm, lo, hi, st, dv] = ps[i], s = $("pd-p" + (i + 1));
      $("pd-p" + (i + 1) + "n").textContent = nm;
      s.min = lo; s.max = hi; s.step = st;
      if (useDefaults) s.value = dv;
    });
  }
  function domainOf(S) {
    if (S.dom) return S.dom;
    const hi = S.heavy ? S.q(0.95) : S.q(0.995);
    return [0, hi];
  }
  // numerical E X and E X² on a transformed grid (smooth integrands, no singular endpoints)
  function moments(S) {
    const n = 6000;
    let g;
    if (S.map === "lin") { const [l, h] = S.lim; g = { lo: l, hi: h, x: t => t, dx: () => 1 }; }
    else if (S.map === "logit") g = { lo: -40, hi: 40, x: t => 1 / (1 + Math.exp(-t)), dx: t => { const p = 1 / (1 + Math.exp(-t)); return p * (1 - p); } };
    else { const lo = Math.log(Math.max(1e-300, S.q(1e-12))), hi = Math.log(S.q(1 - 1e-12)); g = { lo: Math.max(lo, -700), hi: hi, x: t => Math.exp(t), dx: t => Math.exp(t) }; }
    const I = k => PV.simpson(t => { const x = g.x(t), f = S.pdf(x); return isFinite(f) ? Math.pow(x, k) * f * g.dx(t) : 0; }, g.lo, g.hi, n);
    const m0 = I(0), m1 = I(1), m2 = I(2);
    return { m0: m0, E: m1, V: m2 - m1 * m1 };
  }
  function draw() {
    const pv = params();
    pv.forEach((v, i) => { $("pd-p" + (i + 1) + "v").textContent = (FAM[fam].p[i][3] < 1 ? ST.fmt(v, FAM[fam].p[i][3] < 0.1 ? 2 : 1) : v); });
    const S = FAM[fam].make(...pv);
    cur = S; D = domainOf(S);
    a = ST.clamp(a, D[0], D[1]); b = ST.clamp(b, D[0], D[1]);
    if (a > b) { const t = a; a = b; b = t; }
    $("pd-a").value = Math.round(1000 * (a - D[0]) / (D[1] - D[0]));
    $("pd-b").value = Math.round(1000 * (b - D[0]) / (D[1] - D[0]));
    $("pd-av").textContent = (+a.toPrecision(5)).toString();
    $("pd-bv").textContent = (+b.toPrecision(5)).toString();

    svg.selectAll("*").remove();
    const m = { l: 52, r: 18, t: 22, b: 30 }, iw = W - m.l - m.r;
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleLinear().domain(D).range([0, iw]);
    const N = 900, xs = d3.range(N + 1).map(i => D[0] + (D[1] - D[0]) * i / N);
    const fv = xs.map(S.pdf).filter(isFinite);
    const fmax = Math.min(d3.max(fv) || 1, 3 * (d3.quantile(fv.slice().sort((p, q) => p - q), 0.9) || 1)) * 1.12;
    const top = { y0: 165, y1: 0 }, bot = { y0: 368, y1: 215 };
    const yP = d3.scaleLinear().domain([0, fmax]).range([top.y0, top.y1]);
    const yF = d3.scaleLinear().domain([0, 1]).range([bot.y0, bot.y1]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${top.y0})`).call(d3.axisBottom(x).ticks(9));
    g.append("g").attr("class", "axis").call(d3.axisLeft(yP).ticks(4));
    CD.label(g, 0, -8, `density f(x) of ${S.name}`);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${bot.y0})`).call(d3.axisBottom(x).ticks(9));
    g.append("g").attr("class", "axis").call(d3.axisLeft(yF).ticks(4));
    CD.label(g, iw, bot.y1 - 8, "cdf F(x) = P(X ≤ x)   · green ticks: quartiles", { anchor: "end" });
    const sx = d3.range(401).map(i => a + (b - a) * i / 400);
    if (b > a) g.append("path").attr("d", CD.area(sx, S.pdf, x, yP, fmax)).attr("fill", SC.a2).attr("opacity", 0.45);
    g.append("path").attr("d", CD.curve(xs, S.pdf, x, yP, fmax)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    if (isFinite(S.E) && S.E >= D[0] && S.E <= D[1]) {
      g.append("line").attr("x1", x(S.E)).attr("x2", x(S.E)).attr("y1", top.y1).attr("y2", top.y0).attr("stroke", SC.good).attr("stroke-dasharray", "4 3");
      CD.label(g, x(S.E) + 4, top.y1 + 10, `E X = ${(+S.E.toPrecision(4))}`, { color: SC.good, size: 10 });
    }
    g.append("path").attr("d", d3.line().x(d => x(d)).y(d => yF(S.cdf(d)))(xs)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    const Fa = S.cdf(a), Fb = S.cdf(b), P = Math.max(0, Fb - Fa);
    [[a, Fa, "F(a)"], [b, Fb, "F(b)"]].forEach(([t, v, lab]) => {
      g.append("line").attr("x1", 0).attr("x2", x(t)).attr("y1", yF(v)).attr("y2", yF(v)).attr("stroke", SC.a2).attr("stroke-dasharray", "5 4");
      g.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", yF(v)).attr("y2", bot.y0).attr("stroke", SC.a2).attr("opacity", 0.6);
      CD.label(g, 4, yF(v) - 3, `${lab} = ${v.toFixed(4)}`, { color: SC.a2, size: 10 });
    });
    g.append("line").attr("x1", iw - 6).attr("x2", iw - 6).attr("y1", yF(Fa)).attr("y2", yF(Fb)).attr("stroke", SC.a2).attr("stroke-width", 3);
    const qs = [0.25, 0.5, 0.75].map(S.q);
    qs.forEach((q, i) => {
      if (q < D[0] || q > D[1]) return;
      g.append("line").attr("x1", x(q)).attr("x2", x(q)).attr("y1", yF([0.25, 0.5, 0.75][i]) - 5).attr("y2", yF([0.25, 0.5, 0.75][i]) + 5).attr("stroke", SC.good).attr("stroke-width", 2);
    });
    if (S.z) {
      const zt = [-3, -2, -1, 0, 1, 2, 3];
      zt.forEach(z => { const xv = S.q(ST.normCdf(z)); if (xv >= D[0] && xv <= D[1]) CD.label(g, x(xv), top.y0 + 30, `z=${z}`, { anchor: "middle", size: 9, color: SC.violet }); });
    }
    const mo = moments(S);
    const eTxt = isFinite(S.E) ? `mean by integration <b>${(+mo.E.toPrecision(6))}</b> (closed form ${(+S.E.toPrecision(6))}${S.heavy ? "; the integral is truncated at the 1 − 10⁻¹² quantile, and a power tail converges slowly" : ""})`
      : `mean: the integral diverges (α ≤ 1) — E X = ∞`;
    const vTxt = isFinite(S.V) ? `variance by integration <b>${(+mo.V.toPrecision(6))}</b> (closed form ${(+S.V.toPrecision(6))})`
      : `variance: the integral diverges${S.heavy ? " (α ≤ 2)" : ""} — Var X = ∞`;
    const zTxt = S.z ? ` · in standard units z(a) = ${ST.fmt(S.z(a), 3)}, z(b) = ${ST.fmt(S.z(b), 3)}` : "";
    out.innerHTML = `<b>${S.name}</b> · P(${(+a.toPrecision(5))} &lt; X ≤ ${(+b.toPrecision(5))}) = F(b) − F(a) = ${Fb.toFixed(4)} − ${Fa.toFixed(4)} = <b>${P.toFixed(4)}</b>${zTxt}` +
      `<br>∫ f = ${mo.m0.toFixed(6)} · ${eTxt} · ${vTxt}` +
      `<br>quartiles ${qs.map(q => (+q.toPrecision(4))).join(", ")} (median ${(+qs[1].toPrecision(4))}, IQR ${(+(qs[2] - qs[0]).toPrecision(4))})`;
  }
  function slider(id) {
    $(id).addEventListener("input", () => {
      const v = D[0] + (D[1] - D[0]) * (+$(id).value) / 1000;
      if (id === "pd-a") a = v; else b = v;
      draw();
    });
  }
  slider("pd-a"); slider("pd-b");
  ["pd-p1", "pd-p2"].forEach(id => $(id).addEventListener("input", draw));
  function famDefaults() {
    const S = FAM[fam].make(...params());
    if (fam === "norm" && Math.abs(params()[0] - 68.3) < 1e-9 && Math.abs(params()[1] - 1.8) < 1e-9) { a = 67.4; b = 71.9; }
    else { a = S.q(0.25); b = S.q(0.75); }
  }
  sel.addEventListener("change", () => { fam = sel.value; setupSliders(true); famDefaults(); draw(); });
  $("pd-reset").addEventListener("click", () => { sel.value = "norm"; fam = "norm"; setupSliders(true); a = 67.4; b = 71.9; draw(); });
  setupSliders(true); draw();
})();

/* ─────────────────── 3 · memorylessness and the hazard ───────────────────
   Caption audit: default exponential, mean 2, s = 3 → S(3) = e^(−1.5) = 0.2231, P(T > 5 | T > 3)
   = 0.3679 = P(T > 2), largest density gap 0 (rounding only), mean residual life 2.000, flat hazard 0.5.
   Gamma(3), uniform(0, 4) and Pareto(3) with mean 2 give non-zero gaps and changing residual lives. */
(function () {
  const svg = d3.select("#memo-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, $ = id => document.getElementById(id), out = $("memo-readout"), sel = $("me-fam");
  function fam(k, m) {
    switch (k) {
      case "gamma": return { name: `gamma(shape 3, mean ${ST.fmt(m, 1)})`, pdf: t => PV.gammaPdf(t, 3, 3 / m), S: t => 1 - PV.gammaCdf(t, 3, 3 / m) };
      case "unif": return { name: `uniform(0, ${ST.fmt(2 * m, 1)})`, pdf: t => (t >= 0 && t < 2 * m ? 1 / (2 * m) : 0), S: t => ST.clamp(1 - t / (2 * m), 0, 1), end: 2 * m };
      case "pareto": { const xm = 2 * m / 3; return { name: `Pareto(α = 3, xm = ${ST.fmt(xm, 2)}), mean ${ST.fmt(m, 1)}`, pdf: t => PV.paretoPdf(t, 3, xm), S: t => 1 - PV.paretoCdf(t, 3, xm) }; }
      default: return { name: `exponential(mean ${ST.fmt(m, 1)})`, pdf: t => PV.expPdf(t, 1 / m), S: t => Math.exp(-t / m) };
    }
  }
  function draw() {
    const k = sel.value, m = +$("me-m").value, s = +$("me-s").value;
    $("me-mv").textContent = ST.fmt(m, 1); $("me-sv").textContent = ST.fmt(s, 1);
    const F = fam(k, m), T = 12;
    const Ss = F.S(s);
    svg.selectAll("*").remove();
    const ml = 52, iw = W - ml - 18;
    const g = svg.append("g").attr("transform", `translate(${ml},20)`);
    const x = d3.scaleLinear().domain([0, T]).range([0, iw]);
    const xs = d3.range(1201).map(i => T * i / 1200);
    const cond = u => (Ss > 0 ? F.pdf(s + u) / Ss : NaN);
    const ymax = Math.max(d3.max(xs, F.pdf), Ss > 0 ? d3.max(xs, cond) : 0) * 1.12;
    const y = d3.scaleLinear().domain([0, Math.min(ymax, 3)]).range([200, 0]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,200)").call(d3.axisBottom(x).ticks(12));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    CD.label(g, 0, -6, `density of the wait T — ${F.name}`);
    const tail = xs.filter(t => t >= s);
    if (tail.length > 1) g.append("path").attr("d", CD.area(tail, F.pdf, x, y, y.domain()[1])).attr("fill", SC.a2).attr("opacity", 0.25);
    g.append("path").attr("d", CD.area(xs, F.pdf, x, y, y.domain()[1])).attr("fill", SC.accent).attr("opacity", 0.18);
    g.append("path").attr("d", CD.curve(xs, F.pdf, x, y, y.domain()[1])).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    if (Ss > 0) g.append("path").attr("d", CD.curve(xs, cond, x, y, y.domain()[1])).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.4).attr("stroke-dasharray", "7 4");
    g.append("line").attr("x1", x(s)).attr("x2", x(s)).attr("y1", 0).attr("y2", 200).attr("stroke", SC.a2);
    CD.label(g, x(s) + 4, 12, `s = ${ST.fmt(s, 1)}`, { color: SC.a2, size: 10 });
    ST.legend(g, [{ label: "density of T", color: SC.accent }, { label: "remaining wait given T > s (shifted back)", color: SC.a2, dash: "7 4" }], iw - 250, 12);
    // hazard panel
    const hy0 = 330, hy1 = 248;
    const hz = t => { const sv = F.S(t); return sv > 1e-12 ? F.pdf(t) / sv : NaN; };
    const hv = xs.map(hz).filter(isFinite);
    const hmax = Math.min(Math.max(1e-6, d3.max(hv) || 1) * 1.15, 4 / m + 0.5);
    const yh = d3.scaleLinear().domain([0, hmax]).range([hy0, hy1]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${hy0})`).call(d3.axisBottom(x).ticks(12));
    g.append("g").attr("class", "axis").call(d3.axisLeft(yh).ticks(3));
    CD.label(g, 0, hy1 - 6, "hazard rate h(t) = f(t)/P(T > t)");
    g.append("path").attr("d", CD.curve(xs, hz, x, yh, hmax)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2);
    g.append("line").attr("x1", x(s)).attr("x2", x(s)).attr("y1", hy1).attr("y2", hy0).attr("stroke", SC.a2).attr("opacity", 0.7);
    CD.label(g, iw, hy0 + 30, "time t", { anchor: "end" });
    let txt;
    if (Ss <= 0) {
      txt = `<b>${F.name}</b> · P(T &gt; ${ST.fmt(s, 1)}) = 0 — the wait can never last that long, so there is nothing to condition on.`;
    } else {
      let gap = 0; xs.forEach(u => { const d = Math.abs(cond(u) - F.pdf(u)); if (isFinite(d)) gap = Math.max(gap, d); });
      const pc = F.S(s + 2) / Ss, pf = F.S(2);
      const mrl = PV.simpson(F.S, s, s + 400 * m, 40000) / Ss, Et = PV.simpson(F.S, 0, 400 * m, 40000);
      txt = `<b>${F.name}</b> · P(T &gt; ${ST.fmt(s, 1)}) = <b>${Ss.toFixed(4)}</b> · P(T &gt; ${ST.fmt(s + 2, 1)} | T &gt; ${ST.fmt(s, 1)}) = <b>${pc.toFixed(4)}</b> vs fresh P(T &gt; 2) = <b>${pf.toFixed(4)}</b>` +
        `<br>largest gap between the remaining-wait density and the original: <b>${gap < 5e-9 ? "0.0000" : gap.toFixed(4)}</b>` +
        ` · expected remaining wait E[T − s | T &gt; s] = <b>${mrl.toFixed(3)}</b> vs E T = ${Et.toFixed(3)} · hazard at s: ${isFinite(hz(s)) ? hz(s).toFixed(4) : "—"}` +
        `<br>${k === "exp" ? "Memoryless: the outline lands on the curve and the hazard is flat." : "Not memoryless: having waited changes what is left, and the hazard is not constant."}`;
    }
    out.innerHTML = txt;
  }
  ["me-m", "me-s"].forEach(id => $(id).addEventListener("input", draw));
  sel.addEventListener("change", draw);
  draw();
})();

/* ─────────────────── 4 · geometric → exponential ───────────────────
   Caption audit: default λ = 1, Δ = 0.25 → p = 0.25, sup |F_Δ − F| = 1 − e^(−0.25) = 0.2212, mean of
   T_Δ by summation 1.0000 = 1/λ, variance 0.75 vs 1. Δ = 0.001 → gap 0.0010.                     */
(function () {
  const svg = d3.select("#geoexp-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, $ = id => document.getElementById(id), out = $("geoexp-readout");
  const DS = [1, 0.5, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];
  function supGap(l, d) {
    const p = l * d; if (p >= 1) return NaN;
    let s = 0;
    for (let k = 0; k * d < 40 / l; k++) {
      const Fd = 1 - Math.pow(1 - p, k), t0 = k * d;
      s = Math.max(s, Math.abs(Fd - PV.expCdf(t0, l)), Math.abs(Fd - PV.expCdf(t0 + d, l)));
    }
    return s;
  }
  function draw() {
    const l = +$("ge-l").value, d = DS[+$("ge-d").value], p = l * d;
    $("ge-lv").textContent = ST.fmt(l, 1); $("ge-dv").textContent = d;
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", "translate(50,20)");
    const iw = 420, T = 5 / l;
    const x = d3.scaleLinear().domain([0, T]).range([0, iw]);
    const ok = p < 1;
    const yP = d3.scaleLinear().domain([0, Math.max(l, ok ? p / d : 0) * 1.12]).range([160, 0]);
    const yF = d3.scaleLinear().domain([0, 1]).range([330, 205]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,160)").call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "axis").call(d3.axisLeft(yP).ticks(4));
    g.append("g").attr("class", "axis").attr("transform", "translate(0,330)").call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "axis").call(d3.axisLeft(yF).ticks(4));
    CD.label(g, 0, -6, "density: geometric bars (area = P(T_Δ = kΔ)) vs λe^(−λt)");
    CD.label(g, 0, 199, "cdf: staircase F_Δ vs 1 − e^(−λt)");
    const xs = d3.range(601).map(i => T * i / 600);
    if (ok) {
      const K = Math.ceil(T / d);
      const pts = [[0, 0]];
      for (let k = 1; k <= K; k++) { const h = Math.pow(1 - p, k - 1) * p / d; pts.push([(k - 1) * d, h], [k * d, h]); }
      pts.push([K * d, 0]);
      g.append("path").attr("d", d3.area().x(q => x(Math.min(T, q[0]))).y0(160).y1(q => yP(q[1]))(pts)).attr("fill", SC.accent).attr("opacity", 0.35);
      if (K <= 120) for (let k = 1; k <= K; k++) g.append("line").attr("x1", x(Math.min(T, k * d))).attr("x2", x(Math.min(T, k * d))).attr("y1", 160).attr("y2", yP(Math.pow(1 - p, k - 1) * p / d)).attr("stroke", SC.bg).attr("stroke-width", 0.6);
      const st = [];
      for (let k = 0; k <= K; k++) { const Fv = 1 - Math.pow(1 - p, k); st.push([k * d, Fv], [Math.min(T, (k + 1) * d), Fv]); }
      g.append("path").attr("d", d3.line().x(q => x(Math.min(T, q[0]))).y(q => yF(q[1]))(st)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.8);
    }
    g.append("path").attr("d", d3.line().x(t => x(t)).y(t => yP(PV.expPdf(t, l)))(xs)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
    g.append("path").attr("d", d3.line().x(t => x(t)).y(t => yF(PV.expCdf(t, l)))(xs)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
    // right: sup gap vs Δ, log–log
    const gr = svg.append("g").attr("transform", "translate(540,40)");
    const gx = d3.scaleLog().domain([0.0008, 1.2]).range([0, 120]), gy = d3.scaleLog().domain([0.0005, 1]).range([260, 0]);
    gr.append("g").attr("class", "axis").attr("transform", "translate(0,260)").call(d3.axisBottom(gx).ticks(3, "~g"));
    gr.append("g").attr("class", "axis").call(d3.axisLeft(gy).ticks(4, "~g"));
    CD.label(gr, -30, -18, "sup |F_Δ − F| vs Δ", { size: 10 });
    CD.label(gr, 120, 290, "Δ (log)", { anchor: "end", size: 10 });
    const gp = DS.map(dd => ({ d: dd, s: supGap(l, dd) })).filter(q => isFinite(q.s) && q.s > 0);
    gr.append("path").attr("d", d3.line().x(q => gx(q.d)).y(q => gy(Math.max(5e-4, q.s)))(gp)).attr("fill", "none").attr("stroke", SC.muted);
    gr.selectAll("circle").data(gp).join("circle").attr("cx", q => gx(q.d)).attr("cy", q => gy(Math.max(5e-4, q.s))).attr("r", q => q.d === d ? 5 : 2.5)
      .attr("fill", q => q.d === d ? SC.a2 : SC.accent);
    ST.legend(g, [{ label: "geometric time T_Δ = Δ·Geo(λΔ)", color: SC.accent }, { label: "exponential(λ)", color: SC.a2, dash: "1 0" }], iw - 190, 10);
    if (!ok) { out.innerHTML = `λΔ = ${ST.fmt(p, 2)} ≥ 1: a trial cannot succeed with probability above 1 — shorten the trials.`; return; }
    // moments by summation
    let E = 0, E2 = 0, acc = 0;
    for (let k = 1; k < 2e6; k++) { const q = Math.pow(1 - p, k - 1) * p; E += k * d * q; E2 += (k * d) * (k * d) * q; acc += q; if (1 - acc < 1e-14) break; }
    let gridGap = 0; for (let k = 0; k * d < 40 / l; k++) gridGap = Math.max(gridGap, Math.abs(Math.pow(1 - p, k) - Math.exp(-l * k * d)));
    const sg = supGap(l, d);
    out.innerHTML = `trial length Δ = ${d} · success probability per trial p = λΔ = <b>${(+p.toPrecision(4))}</b>` +
      `<br>E T_Δ by summation = <b>${E.toFixed(4)}</b> (1/λ = ${(1 / l).toFixed(4)}, exactly, for every Δ) · Var T_Δ = <b>${(E2 - E * E).toFixed(4)}</b> = (1 − λΔ)/λ² (limit 1/λ² = ${(1 / (l * l)).toFixed(4)})` +
      `<br>largest cdf gap anywhere = <b>${sg.toFixed(4)}</b> (1 − e^(−λΔ) = ${(1 - Math.exp(-l * d)).toFixed(4)}) · largest gap at the grid points kΔ = ${gridGap.toFixed(5)}` +
      ` · P(T_Δ &gt; 2) = ${Math.pow(1 - p, Math.floor(2 / d + 1e-9)).toFixed(4)} vs e^(−2λ) = ${Math.exp(-2 * l).toFixed(4)}`;
  }
  ["ge-l", "ge-d"].forEach(id => $(id).addEventListener("input", draw));
  draw();
})();

/* ─────────────────── 5 · the gamma as the r-th arrival ───────────────────
   Caption audit (seed CD.seeds.gamma, 5,000 draws at r = 3, λ = 0.2, t = 12): gamma cdf 0.4303,
   Poisson sum 0.4303, simulated fraction within one standard error (0.007) of them; sample mean
   near 15 and sample variance near 75. Seed 3 gives fraction 0.4338 (0.5 SE above 0.4303), mean 14.986,
   variance 76.8.                                                                                     */
(function () {
  const svg = d3.select("#gamma-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, $ = id => document.getElementById(id), out = $("gamma-readout");
  let seed = CD.seeds.gamma, t = 12, draws = [], last = [];
  const NS = 5000;
  function simulate(a, l) {
    const r = ST.rng(seed);
    draws = new Array(NS); last = [];
    const whole = Math.abs(a - Math.round(a)) < 1e-9;
    for (let i = 0; i < NS; i++) {
      if (whole) {
        let s = 0; const arr = [];
        for (let j = 0; j < Math.round(a); j++) { s += PV.expDraw(l, r); arr.push(s); }
        draws[i] = s; if (i === NS - 1) last = arr;
      } else draws[i] = PV.gammaDraw(a, l, r);
    }
  }
  function draw(resim) {
    const a = +$("ga-a").value, l = +$("ga-l").value;
    $("ga-av").textContent = (+a.toFixed(1)); $("ga-lv").textContent = ST.fmt(l, 2);
    if (resim) simulate(a, l);
    const hi = PV.gammaQuant(0.995, a, l);
    t = ST.clamp(t, 0, hi);
    $("ga-t").value = Math.round(1000 * t / hi); $("ga-tv").textContent = (+t.toPrecision(4));
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", "translate(52,16)"), iw = W - 52 - 18;
    const x = d3.scaleLinear().domain([0, hi]).range([0, iw]);
    // arrival strip
    const whole = Math.abs(a - Math.round(a)) < 1e-9;
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", 22).attr("y2", 22).attr("stroke", SC.line);
    CD.label(g, 0, 8, whole ? `latest draw: arrivals of a rate-${ST.fmt(l, 2)} stream; the ${Math.round(a)}-th arrival is the gamma draw` : "fractional shape: drawn directly from the gamma (no arrival story)", { size: 10 });
    last.forEach((v, i) => {
      if (v > hi) return;
      const isLast = i === last.length - 1;
      g.append("circle").attr("cx", x(v)).attr("cy", 22).attr("r", isLast ? 6 : 4).attr("fill", isLast ? SC.a2 : SC.accent);
      if (isLast) CD.label(g, x(v), 42, `T₍${last.length}₎ = ${v.toFixed(2)}`, { anchor: "middle", color: SC.a2, size: 10 });
    });
    const y0 = 320, y1 = 60;
    const H = CD.hist(draws, 0, hi, 45);
    const xs = d3.range(601).map(i => hi * i / 600);
    const fmax = Math.min(d3.max(xs.slice(1), q => PV.gammaPdf(q, a, l)), 3 * d3.max(H, h => h.d));
    const y = d3.scaleLinear().domain([0, Math.max(fmax, d3.max(H, h => h.d)) * 1.1]).range([y0, y1]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${y0})`).call(d3.axisBottom(x).ticks(9));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    const sx = xs.filter(q => q <= t);
    if (sx.length > 1) g.append("path").attr("d", CD.area(sx.concat([t]), q => PV.gammaPdf(q, a, l), x, y, y.domain()[1])).attr("fill", SC.a2).attr("opacity", 0.3);
    g.selectAll("rect.h").data(H).join("rect").attr("class", "h").attr("x", h => x(h.x0) + 0.5).attr("width", h => Math.max(0, x(h.x1) - x(h.x0) - 1))
      .attr("y", h => y(Math.min(h.d, y.domain()[1]))).attr("height", h => y0 - y(Math.min(h.d, y.domain()[1]))).attr("fill", SC.accent).attr("opacity", 0.55);
    g.append("path").attr("d", CD.curve(xs, q => PV.gammaPdf(q, a, l), x, y, y.domain()[1])).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
    g.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", y1).attr("y2", y0).attr("stroke", SC.a2).attr("stroke-dasharray", "4 3");
    CD.label(g, x(t) + 4, y1 + 12, `t = ${(+t.toPrecision(4))}`, { color: SC.a2, size: 10 });
    CD.label(g, iw, y0 + 30, "time of the arrival", { anchor: "end" });
    ST.legend(g, [{ label: `histogram of ${NS.toLocaleString()} draws (density scale)`, color: SC.accent, op: 0.55 }, { label: `Gamma(${(+a.toFixed(1))}, ${ST.fmt(l, 2)}) density`, color: SC.a2, dash: "1 0" }], iw - 260, y1 + 8);
    const m = ST.mean(draws), v = ST.variance(draws);
    const Pg = PV.gammaCdf(t, a, l), frac = draws.filter(q => q <= t).length / NS, se = Math.sqrt(Pg * (1 - Pg) / NS);
    const Pp = whole ? 1 - PV.poisCdf(Math.round(a) - 1, l * t) : NaN;
    out.innerHTML = `<b>Gamma(α = ${(+a.toFixed(1))}, λ = ${ST.fmt(l, 2)})</b> · mean α/λ = ${(a / l).toFixed(3)} (sample <b>${m.toFixed(3)}</b>) · variance α/λ² = ${(a / (l * l)).toFixed(3)} (sample <b>${v.toFixed(3)}</b>) · skewness 2/√α = ${(2 / Math.sqrt(a)).toFixed(3)}` +
      `<br>P(T ≤ ${(+t.toPrecision(4))}): gamma cdf <b>${Pg.toFixed(4)}</b> · ` +
      (whole ? `Poisson sum P(N(t) ≥ ${Math.round(a)}), N(t) ~ Poisson(${(l * t).toFixed(3)}): <b>${Pp.toFixed(4)}</b> · ` : `Poisson identity needs a whole-number shape · `) +
      `simulated fraction <b>${frac.toFixed(4)}</b> (± ${se.toFixed(4)} standard error) · seed ${seed}`;
  }
  $("ga-a").addEventListener("input", () => draw(true));
  $("ga-l").addEventListener("input", () => draw(true));
  $("ga-t").addEventListener("input", () => { const a = +$("ga-a").value, l = +$("ga-l").value; t = PV.gammaQuant(0.995, a, l) * (+$("ga-t").value) / 1000; draw(false); });
  $("ga-sim").addEventListener("click", () => { seed++; draw(true); });
  $("ga-reset").addEventListener("click", () => { $("ga-a").value = 3; $("ga-l").value = 0.2; t = 12; seed = CD.seeds.gamma; draw(true); });
  draw(true);
})();

/* ─────────────────── 6 · beta–binomial updating ───────────────────
   Caption audit: default prior Beta(1.18, 2.35), 22 of 40 → posterior Beta(23.18, 20.35), mean 0.5325,
   central 95% interval 0.385–0.677, P(p > ½) = 0.668; prior mean 0.334 < 0.5325 < 0.55 = k/n.      */
(function () {
  const svg = d3.select("#beta-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, $ = id => document.getElementById(id), out = $("beta-readout");
  function set(a, b, n, k) { $("be-a").value = a; $("be-b").value = b; $("be-n").value = n; $("be-k").max = n; $("be-k").value = k; draw(); }
  function draw() {
    const a = +$("be-a").value, b = +$("be-b").value, n = +$("be-n").value;
    $("be-k").max = n; const k = Math.min(+$("be-k").value, n);
    $("be-av").textContent = a.toFixed(2); $("be-bv").textContent = b.toFixed(2); $("be-nv").textContent = n; $("be-kv").textContent = k;
    const A = a + k, B = b + n - k;
    const prior = x => PV.betaPdf(x, a, b), post = x => PV.betaPdf(x, A, B), lik = x => PV.betaPdf(x, k + 1, n - k + 1);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", "translate(48,18)"), iw = W - 48 - 18, ih = 270;
    const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
    const xs = d3.range(1, 1000).map(i => i / 1000);
    const top = Math.max(d3.max(xs, post), d3.max(xs, lik), Math.min(d3.max(xs, prior), 8)) * 1.55;
    const ymax = Math.min(top, 40);
    const y = d3.scaleLinear().domain([0, ymax]).range([ih, 0]);
    ST.gridY(g, y, iw, 4);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(10));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    CD.label(g, iw, ih + 30, "success probability p", { anchor: "end" });
    const lo = PV.betaQuant(0.025, A, B), hi = PV.betaQuant(0.975, A, B);
    const ci = xs.filter(q => q >= lo && q <= hi);
    g.append("path").attr("d", CD.area([lo].concat(ci, [hi]), post, x, y, ymax)).attr("fill", SC.a2).attr("opacity", 0.3);
    g.append("path").attr("d", CD.curve(xs, prior, x, y, ymax)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
    g.append("path").attr("d", CD.curve(xs, lik, x, y, ymax)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 1.8);
    g.append("path").attr("d", CD.curve(xs, post, x, y, ymax)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.6);
    const pm = a / (a + b), mle = n > 0 ? k / n : NaN, qm = A / (A + B);
    [[pm, SC.muted, "prior mean"], [mle, SC.good, "k/n"], [qm, SC.a2, "posterior mean"]].forEach(([v, c, lab], i) => {
      if (!isFinite(v)) return;
      g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", ih).attr("y2", ih - 14).attr("stroke", c).attr("stroke-width", 3);
      g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 6 + 13 * i).attr("y2", ih - 14).attr("stroke", c).attr("stroke-dasharray", "1 3").attr("opacity", 0.8);
      CD.label(g, x(v) + (v > 0.8 ? -4 : 4), 10 + 13 * i, `${lab} ${v.toFixed(3)}`, { anchor: v > 0.8 ? "end" : "start", color: c, size: 10 });
    });
    ST.legend(g, [{ label: `prior Beta(${a.toFixed(2)}, ${b.toFixed(2)})`, color: SC.muted, dash: "6 4" }, { label: `likelihood of ${k} of ${n} (as a density)`, color: SC.good, dash: "1 0" },
      { label: `posterior Beta(${A.toFixed(2)}, ${B.toFixed(2)}), 95% shaded`, color: SC.a2, dash: "1 0" }], iw - 250, 100);
    const w = (a + b) / (a + b + n);
    out.innerHTML = `posterior <b>Beta(${A.toFixed(2)}, ${B.toFixed(2)})</b> · mean (a + k)/(a + b + n) = <b>${qm.toFixed(4)}</b> = ${w.toFixed(3)} × prior mean ${pm.toFixed(4)} + ${(1 - w).toFixed(3)} × k/n ${isFinite(mle) ? mle.toFixed(4) : "—"}` +
      `<br>central 95% interval <b>${lo.toFixed(3)} – ${hi.toFixed(3)}</b> · P(p &gt; ½ | data) = <b>${(1 - PV.betaCdf(0.5, A, B)).toFixed(3)}</b> (prior: ${(1 - PV.betaCdf(0.5, a, b)).toFixed(3)})` +
      `<br>prior weight a + b = ${(a + b).toFixed(2)} pseudo-trials against n = ${n} real ones`;
  }
  ["be-a", "be-b", "be-n", "be-k"].forEach(id => $(id).addEventListener("input", draw));
  $("be-flat").addEventListener("click", () => set(1, 1, +$("be-n").value, +$("be-k").value));
  $("be-strong").addEventListener("click", () => set(30, 60, +$("be-n").value, +$("be-k").value));
  $("be-reset").addEventListener("click", () => set(1.18, 2.35, 40, 22));
  draw();
})();

/* ─────────────────── 7 · heavy tails ───────────────────
   Caption audit (20,000 draws each, α = 1.5, seeds 1–40 scanned): the largest Pareto draw's share of
   the total ranged 0.5%–17.5% (most seeds 1–3%); at seed 10 the exponential's is 0.04% and the
   lognormal's 0.12%. Seed 10 is used: one draw near n = 15,350 is 6.7% of the total and lifts the Pareto running
   mean to 3.136, while the exponential (2.981) and lognormal (3.015) end within 1% of 3. Seed 27 (17.5%)
   was rejected as unrepresentative.                                                               */
(function () {
  const svg = d3.select("#tails-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, $ = id => document.getElementById(id), out = $("tails-readout");
  const N = 20000, LMU = Math.log(3) - 0.5;
  let seed = CD.seeds.tails, data = null;
  function simulate(al) {
    const r1 = ST.rng(seed), r2 = ST.rng(seed + 1000), r3 = ST.rng(seed + 2000);
    data = { par: [], exp: [], ln: [] };
    for (let i = 0; i < N; i++) { data.par.push(PV.paretoDraw(al, 1, r1)); data.exp.push(PV.expDraw(1 / 3, r2)); data.ln.push(PV.lognormDraw(LMU, 1, r3)); }
  }
  function run(a) { const o = new Float64Array(a.length); let s = 0; for (let i = 0; i < a.length; i++) { s += a[i]; o[i] = s / (i + 1); } return o; }
  function draw(resim) {
    const al = +$("ta-a").value, ax = $("ta-ax").value;
    $("ta-av").textContent = al.toFixed(1);
    if (resim || !data) simulate(al);
    svg.selectAll("*").remove();
    const fams = [
      { k: "par", name: `Pareto(α = ${al.toFixed(1)})`, c: SC.accent, S: x => (x < 1 ? 1 : Math.pow(x, -al)), E: al > 1 ? al / (al - 1) : Infinity },
      { k: "exp", name: "exponential, mean 3", c: SC.good, S: x => Math.exp(-x / 3), E: 3 },
      { k: "ln", name: "lognormal, mean 3, σ = 1", c: SC.a2, S: x => 1 - PV.lognormCdf(x, LMU, 1), E: 3 }
    ];
    // left: survival
    const gl = svg.append("g").attr("transform", "translate(52,22)"), lw = 260, lh = 270;
    let x, y, xs;
    if (ax === "loglog") { x = d3.scaleLog().domain([0.1, 1e4]).range([0, lw]); y = d3.scaleLog().domain([1e-8, 1]).range([lh, 0]); xs = d3.range(401).map(i => Math.pow(10, -1 + 5 * i / 400)); }
    else if (ax === "linlog") { x = d3.scaleLinear().domain([0, 60]).range([0, lw]); y = d3.scaleLog().domain([1e-8, 1]).range([lh, 0]); xs = d3.range(401).map(i => 60 * i / 400); }
    else { x = d3.scaleLinear().domain([0, 20]).range([0, lw]); y = d3.scaleLinear().domain([0, 1]).range([lh, 0]); xs = d3.range(401).map(i => 20 * i / 400); }
    const logY = ax !== "lin";
    gl.append("g").attr("class", "axis").attr("transform", `translate(0,${lh})`).call(ax === "loglog" ? d3.axisBottom(x).ticks(5, "~g") : d3.axisBottom(x).ticks(6));
    gl.append("g").attr("class", "axis").call(logY ? d3.axisLeft(y).ticks(5, "~e") : d3.axisLeft(y).ticks(5));
    CD.label(gl, 0, -8, "survival P(X > x)" + (ax === "loglog" ? " — log–log" : ax === "linlog" ? " — log scale" : ""));
    CD.label(gl, lw, lh + 30, "x", { anchor: "end" });
    fams.forEach(f => {
      const pts = xs.map(q => [q, f.S(q)]).filter(q => !logY || q[1] >= 1e-8);
      gl.append("path").attr("d", d3.line().x(q => x(q[0])).y(q => y(Math.max(logY ? 1e-8 : 0, q[1])))(pts)).attr("fill", "none").attr("stroke", f.c).attr("stroke-width", 2);
    });
    // empirical Pareto survival at a few points
    const sp = Float64Array.from(data.par).sort();
    const ex = [];
    for (let j = 0; j < 40; j++) { const xv = ax === "loglog" ? Math.pow(10, 4 * j / 39) : x.domain()[1] * (j + 0.5) / 40; let lo = 0, hi = N; while (lo < hi) { const mid = (lo + hi) >> 1; if (sp[mid] <= xv) lo = mid + 1; else hi = mid; } const s = (N - lo) / N; if (s > 0) ex.push([xv, s]); }
    gl.selectAll("circle.e").data(ex.filter(q => q[0] >= x.domain()[0] && q[0] <= x.domain()[1])).join("circle").attr("class", "e").attr("cx", q => x(q[0])).attr("cy", q => y(q[1])).attr("r", 2.4).attr("fill", SC.ink).attr("opacity", 0.7);
    // right: running means
    const gr = svg.append("g").attr("transform", "translate(395,22)"), rw = 268, rh = 270;
    const rm = fams.map(f => run(data[f.k]));
    let ymx = 6; rm.forEach(a => { for (let i = 19; i < N; i++) ymx = Math.max(ymx, a[i] * 1.08); });
    const rx = d3.scaleLog().domain([10, N]).range([0, rw]), ry = d3.scaleLinear().domain([0, ymx]).range([rh, 0]);
    gr.append("g").attr("class", "axis").attr("transform", `translate(0,${rh})`).call(d3.axisBottom(rx).ticks(4, "~s"));
    gr.append("g").attr("class", "axis").call(d3.axisLeft(ry).ticks(5));
    CD.label(gr, 0, -8, "running mean of the first n draws");
    CD.label(gr, rw, rh + 30, "n (log)", { anchor: "end" });
    const idx = d3.range(500).map(i => Math.round(Math.pow(10, 1 + Math.log10(N / 10) * i / 499)) - 1);
    fams.forEach((f, j) => {
      if (isFinite(f.E)) gr.append("line").attr("x1", 0).attr("x2", rw).attr("y1", ry(f.E)).attr("y2", ry(f.E)).attr("stroke", f.c).attr("stroke-dasharray", "3 3").attr("opacity", 0.6);
      gr.append("path").attr("d", d3.line().x(i => rx(i + 1)).y(i => ry(Math.min(ymx, rm[j][i])))(idx)).attr("fill", "none").attr("stroke", f.c).attr("stroke-width", 1.8);
    });
    ST.legend(gr, fams.map(f => ({ label: f.name, color: f.c, dash: "1 0" })), 8, 8);
    const rows = fams.map((f, j) => {
      const a = data[f.k], tot = a.reduce((p, q) => p + q, 0), mx = Math.max(...a);
      return `${f.name}: mean ${isFinite(f.E) ? f.E.toFixed(2) : "∞"} · running mean at n = 2,000: ${rm[j][1999].toFixed(3)}, at n = 20,000: <b>${rm[j][N - 1].toFixed(3)}</b> · largest draw ${mx.toFixed(1)} = <b>${(100 * mx / tot).toFixed(2)}%</b> of the total`;
    });
    const varTxt = al > 2 ? "finite variance" : al > 1 ? "finite mean, infinite variance" : "no finite mean";
    out.innerHTML = `Pareto tail index α = ${al.toFixed(1)}: ${varTxt}; dots are the simulated Pareto sample's survival fractions (seed ${seed}).<br>` + rows.join("<br>");
  }
  $("ta-a").addEventListener("input", () => draw(true));
  $("ta-ax").addEventListener("change", () => draw(false));
  $("ta-sim").addEventListener("click", () => { seed++; draw(true); });
  $("ta-reset").addEventListener("click", () => { seed = CD.seeds.tails; $("ta-a").value = 1.5; $("ta-ax").value = "loglog"; draw(true); });
  draw(true);
})();

/* ─────────────────── 8 · change of variables ───────────────────
   Caption audit: default X ~ U(−1, 3), Y = X², y₀ = 2 → preimage (−1, √2], P = (√2 + 1)/4 = 0.6036 by
   the preimage, by the formula, and (within sampling error) by simulation; histogram halves at y = 1. */
(function () {
  const svg = d3.select("#transform-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, $ = id => document.getElementById(id), out = $("transform-readout"), sel = $("tr-g");
  const NS = 20000;
  const M = {
    sq: { name: "Y = X², X ~ U(−1, 3)", xd: [-1.2, 3.2], yd: [0, 9], y0: 2, draw: r => -1 + 4 * r(), g: v => v * v,
      fX: v => (v > -1 && v < 3 ? 0.25 : 0), FX: v => ST.clamp((v + 1) / 4, 0, 1),
      fY: y => (y <= 0 ? NaN : y < 1 ? 1 / (4 * Math.sqrt(y)) : y < 9 ? 1 / (8 * Math.sqrt(y)) : 0), FY: y => (y <= 0 ? 0 : y < 1 ? Math.sqrt(y) / 2 : y < 9 ? (Math.sqrt(y) + 1) / 4 : 1),
      pre: y => [[Math.max(-1, -Math.sqrt(y)), Math.min(3, Math.sqrt(y))]] },
    z2: { name: "Y = Z², Z ~ N(0, 1)", xd: [-3.5, 3.5], yd: [0, 8], y0: 1, draw: r => ST.randn(r), g: v => v * v, fX: ST.normPdf, FX: ST.normCdf,
      fY: y => (y <= 0 ? NaN : ST.chi2Pdf(y, 1)), FY: y => ST.chi2Cdf(y, 1), pre: y => [[-Math.sqrt(y), Math.sqrt(y)]] },
    exp: { name: "Y = e^Z, Z ~ N(0, 1)", xd: [-3, 3], yd: [0, 8], y0: 1, draw: r => ST.randn(r), g: Math.exp, fX: ST.normPdf, FX: ST.normCdf,
      fY: y => PV.lognormPdf(y, 0, 1), FY: y => PV.lognormCdf(y, 0, 1), pre: y => [[-Infinity, Math.log(y)]] },
    speed: { name: "V = 3600/T, T ~ U(40, 60)", xd: [38, 62], yd: [58, 92], y0: 80, draw: r => 40 + 20 * r(), g: v => 3600 / v,
      fX: v => (v > 40 && v < 60 ? 0.05 : 0), FX: v => ST.clamp((v - 40) / 20, 0, 1),
      fY: y => (y > 60 && y < 90 ? 180 / (y * y) : 0), FY: y => (y <= 60 ? 0 : y >= 90 ? 1 : 3 - 180 / y), pre: y => [[3600 / y, 60]] },
    pit: { name: "U = F(X) = 1 − e^(−X), X ~ Exp(1)", xd: [0, 5], yd: [0, 1], y0: 0.5, draw: r => PV.expDraw(1, r), g: v => -Math.expm1(-v), fX: v => PV.expPdf(v, 1), FX: v => PV.expCdf(v, 1),
      fY: y => (y > 0 && y < 1 ? 1 : 0), FY: y => ST.clamp(y, 0, 1), pre: y => [[0, -Math.log1p(-Math.min(y, 1 - 1e-15))]] },
    inv: { name: "X = −ln(1 − U), U ~ U(0, 1)", xd: [0, 1], yd: [0, 6], y0: 1, draw: r => r(), g: v => -Math.log1p(-v), fX: v => (v > 0 && v < 1 ? 1 : 0), FX: v => ST.clamp(v, 0, 1),
      fY: y => PV.expPdf(y, 1), FY: y => PV.expCdf(y, 1), pre: y => [[0, -Math.expm1(-y)]] }
  };
  let mode = "sq", y0 = 2, seed = CD.seeds.transform, ys = [];
  function simulate() { const r = ST.rng(seed), m = M[mode]; ys = new Array(NS); for (let i = 0; i < NS; i++) ys[i] = m.g(m.draw(r)); }
  function draw() {
    const m = M[mode];
    $("tr-y").value = Math.round(1000 * (y0 - m.yd[0]) / (m.yd[1] - m.yd[0])); $("tr-yv").textContent = (+y0.toPrecision(4));
    svg.selectAll("*").remove();
    // left: g(x) with the density of X beneath
    const gl = svg.append("g").attr("transform", "translate(48,20)"), lw = 270;
    const x = d3.scaleLinear().domain(m.xd).range([0, lw]), y = d3.scaleLinear().domain(m.yd).range([200, 0]);
    gl.append("g").attr("class", "axis").attr("transform", "translate(0,200)").call(d3.axisBottom(x).ticks(6));
    gl.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    CD.label(gl, 0, -6, `y = g(x) · ${m.name}`, { size: 10 });
    const xs = d3.range(801).map(i => m.xd[0] + (m.xd[1] - m.xd[0]) * i / 800);
    const inPre = v => m.pre(y0).some(([l, h]) => v >= l && v <= h);
    const supp = v => m.fX(v) > 0;
    gl.append("path").attr("d", d3.line().defined(v => supp(v) && isFinite(m.g(v))).x(v => x(v)).y(v => y(ST.clamp(m.g(v), m.yd[0], m.yd[1])))(xs)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    gl.append("path").attr("d", d3.line().defined(v => supp(v) && inPre(v)).x(v => x(v)).y(v => y(ST.clamp(m.g(v), m.yd[0], m.yd[1])))(xs)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 3.5);
    gl.append("line").attr("x1", 0).attr("x2", lw).attr("y1", y(y0)).attr("y2", y(y0)).attr("stroke", SC.a2).attr("stroke-dasharray", "5 4");
    CD.label(gl, lw - 2, y(y0) - 4, `y₀ = ${(+y0.toPrecision(4))}`, { anchor: "end", color: SC.a2, size: 10 });
    // density of X strip
    const fxm = d3.max(xs, m.fX) * 1.1, yx = d3.scaleLinear().domain([0, fxm]).range([300, 235]);
    gl.append("g").attr("class", "axis").attr("transform", "translate(0,300)").call(d3.axisBottom(x).ticks(6));
    CD.label(gl, 0, 230, "density of the input, preimage {x : g(x) ≤ y₀} shaded", { size: 10 });
    gl.append("path").attr("d", d3.area().defined(v => inPre(v)).x(v => x(v)).y0(300).y1(v => yx(m.fX(v)))(xs)).attr("fill", SC.a2).attr("opacity", 0.5);
    gl.append("path").attr("d", d3.line().x(v => x(v)).y(v => yx(m.fX(v)))(xs)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.6);
    // right: histogram of Y and formula density
    const gr = svg.append("g").attr("transform", "translate(380,20)"), rw = 285, rh = 280;
    const rx = d3.scaleLinear().domain(m.yd).range([0, rw]);
    const H = CD.hist(ys, m.yd[0], m.yd[1], 40);
    const yy = d3.range(1, 800).map(i => m.yd[0] + (m.yd[1] - m.yd[0]) * i / 800);
    const capv = d3.max(H.slice(1), h => h.d) || 1, ymax = Math.max(capv, d3.max(yy.slice(40), q => m.fY(q)) || 0) * 1.25;
    const ry = d3.scaleLinear().domain([0, ymax]).range([rh, 0]);
    gr.append("g").attr("class", "axis").attr("transform", `translate(0,${rh})`).call(d3.axisBottom(rx).ticks(6));
    gr.append("g").attr("class", "axis").call(d3.axisLeft(ry).ticks(5));
    CD.label(gr, 0, -6, `histogram of ${NS.toLocaleString()} values of Y vs f_Y`, { size: 10 });
    gr.selectAll("rect").data(H).join("rect").attr("x", h => rx(h.x0) + 0.5).attr("width", h => Math.max(0, rx(h.x1) - rx(h.x0) - 1))
      .attr("y", h => ry(Math.min(h.d, ymax))).attr("height", h => rh - ry(Math.min(h.d, ymax))).attr("fill", h => (h.x1 <= y0 ? SC.a2 : SC.accent)).attr("opacity", 0.5);
    gr.append("path").attr("d", CD.curve(yy, m.fY, rx, ry, ymax)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 2);
    gr.append("line").attr("x1", rx(y0)).attr("x2", rx(y0)).attr("y1", 0).attr("y2", rh).attr("stroke", SC.a2).attr("stroke-dasharray", "5 4");
    const Ppre = m.pre(y0).reduce((s, [l, h]) => s + Math.max(0, m.FX(h) - m.FX(l)), 0);
    const Pfor = m.FY(y0), frac = ys.filter(v => v <= y0).length / NS, se = Math.sqrt(Pfor * (1 - Pfor) / NS);
    const preTxt = m.pre(y0).map(([l, h]) => `(${isFinite(l) ? (+l.toPrecision(4)) : "−∞"}, ${isFinite(h) ? (+h.toPrecision(4)) : "∞"}]`).join(" ∪ ");
    out.innerHTML = `<b>${m.name}</b> · preimage of {Y ≤ ${(+y0.toPrecision(4))}} within the support: ${preTxt}` +
      `<br>F_Y(y₀) by the preimage P(X ∈ A) = <b>${Ppre.toFixed(4)}</b> · by the formula = <b>${Pfor.toFixed(4)}</b> · simulated fraction = <b>${frac.toFixed(4)}</b> (± ${se.toFixed(4)})` +
      `<br>density at y₀ from the change-of-variables formula: f_Y(${(+y0.toPrecision(4))}) = ${isFinite(m.fY(y0)) ? m.fY(y0).toFixed(4) : "∞"} · seed ${seed}`;
  }
  sel.addEventListener("change", () => { mode = sel.value; y0 = M[mode].y0; simulate(); draw(); });
  $("tr-y").addEventListener("input", () => { const m = M[mode]; y0 = m.yd[0] + (m.yd[1] - m.yd[0]) * (+$("tr-y").value) / 1000; draw(); });
  $("tr-sim").addEventListener("click", () => { seed++; simulate(); draw(); });
  $("tr-reset").addEventListener("click", () => { sel.value = "sq"; mode = "sq"; y0 = 2; seed = CD.seeds.transform; simulate(); draw(); });
  simulate(); draw();
})();
