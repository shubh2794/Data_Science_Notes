/* fourier-analysis.viz.js — the nine visualizations on math/fourier-analysis.html.
   Loaded after ../data.js → ../notes.js → statistics/stats-viz.js (SC palette, ST toolbox: ST.rng,
   ST.randn, ST.frame, ST.fmt …) → probability/prob-viz.js (PV.convolve, PV.poisPmf, PV.cauchyDraw,
   PV.cauchyPdf, PV.expDraw). Distribution functions and random numbers come from ST / PV and are
   never re-implemented here.

   Everything page-local lives under the single namespace FA. Each figure is an IIFE that exits
   quietly if its <svg> is not on the page. The pure computations are exposed on FA so the build's
   node checks can call them without a DOM.

     1  #ph-svg  rotating phasors tip to tail tracing a partial sum                (§02)
     2  #pj-svg  a coefficient as the net area of f·(basis function)               (§03)
     3  #hb-svg  harmonic builder: partial sum, stem plot, Parseval energy ledger    (§04, §07)
     4  #gb-svg  Gibbs zoom, with the Fejér mean                                    (§05)
     5  #dc-svg  smoothness ladder: log–log coefficient decay and fitted slope       (§05)
     6  #st-svg  pulse train: P·cₙ samples the single-pulse transform               (§08)
     7  #cf-svg  characteristic-function gallery: convolve vs multiply              (§10)
     8  #pg-svg  periodogram lab: raw vs Daniell-smoothed, peaks vs truth           (§11)
     9  #rf-svg  random Fourier features vs the exact kernel                        (§12)          */

const FA = {
  // Default seeds, fixed by the caption audit (see the note at each figure).
  seeds: { pg: 11, rf: 9 },
  PI: Math.PI,
  label(g, x, y, t, o) {
    const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", (o && o.size) || 11)
      .attr("fill", (o && o.color) || SC.muted).text(t);
    if (o && o.anchor) e.attr("text-anchor", o.anchor);
    if (o && o.weight) e.attr("font-weight", o.weight);
    return e;
  },
  f(x, d) { const dd = d === undefined ? 4 : d; return (x !== 0 && isFinite(x) && Math.abs(x) < Math.pow(10, -dd) / 2) ? x.toExponential(2) : ST.fmt(x, dd); },
  simpson(fn, a, b, n) {                            // composite Simpson, n even panels
    const m = (n || 4000) + ((n || 4000) % 2), h = (b - a) / m;
    let s = fn(a) + fn(b);
    for (let i = 1; i < m; i++) s += (i % 2 ? 4 : 2) * fn(a + i * h);
    return s * h / 3;
  },
  // Simpson on (−π, 0) and (0, π) separately, evaluating just inside the break: every test
  // function on this page is smooth on each open half, so this keeps the rule fourth-order.
  simpsonSplit(fn, n) {
    const e = 1e-12, P = Math.PI;
    const L = x => fn(Math.min(Math.max(x, -P + e), -e)), R = x => fn(Math.min(Math.max(x, e), P - e));
    return FA.simpson(L, -P, 0, n) + FA.simpson(R, 0, P, n);
  },
  wrap(x) { const P = Math.PI; let y = (x + P) % (2 * P); if (y < 0) y += 2 * P; return y - P; },
  fmtN(n) { return String(n).split("").map(c => "₀₁₂₃₄₅₆₇₈₉"[+c]).join(""); }
};

/* ── The function library: closed-form coefficients (a₀ convention: f ~ a₀/2 + Σ aₙcos nx + bₙ sin nx).
   energy = (1/π)∫ f² over one period, used by Parseval. Verified against quadrature in the build. */
FA.fns = {
  sq:  { name: "square wave sign(x)", f: x => { const y = FA.wrap(x); return y > 0 ? 1 : (y < 0 ? -1 : 0); },
         a0: 0, a: () => 0, b: n => (n % 2 ? 4 / (n * Math.PI) : 0), energy: 2, yr: [-1.6, 1.6] },
  saw: { name: "sawtooth x", f: x => { const y = FA.wrap(x); return Math.abs(Math.abs(y) - Math.PI) < 1e-12 ? 0 : y; },
         a0: 0, a: () => 0, b: n => 2 * (n % 2 ? 1 : -1) / n, energy: 2 * Math.PI * Math.PI / 3, yr: [-4.2, 4.2] },
  tri: { name: "triangle |x|", f: x => Math.abs(FA.wrap(x)),
         a0: Math.PI, a: n => (n % 2 ? -4 / (Math.PI * n * n) : 0), b: () => 0, energy: 2 * Math.PI * Math.PI / 3, yr: [-0.4, 3.6] },
  par: { name: "parabola x²", f: x => { const y = FA.wrap(x); return y * y; },
         a0: 2 * Math.PI * Math.PI / 3, a: n => 4 * (n % 2 ? -1 : 1) / (n * n), b: () => 0, energy: 2 * Math.pow(Math.PI, 4) / 5, yr: [-1, 11] },
  c1:  { name: "x(π − |x|)", f: x => { const y = FA.wrap(x); return y * (Math.PI - Math.abs(y)); },
         a0: 0, a: () => 0, b: n => (n % 2 ? 8 / (Math.PI * n * n * n) : 0) },
  c2:  { name: "πx²/2 − |x|³/3", f: x => { const y = Math.abs(FA.wrap(x)); return Math.PI * y * y / 2 - y * y * y / 3; },
         a0: Math.pow(Math.PI, 3) / 6, a: n => (n % 2 ? -8 / (Math.PI * Math.pow(n, 4)) : 0), b: () => 0 },
  an:  { name: "Poisson kernel r = 0.7", f: x => (1 - 0.49) / (1 - 1.4 * Math.cos(x) + 0.49),
         a0: 2, a: n => 2 * Math.pow(0.7, n), b: () => 0 }
};
FA.partial = function (key, N, x) {                 // S_N(x), harmonics 1..N
  const F = FA.fns[key];
  let s = F.a0 / 2;
  for (let n = 1; n <= N; n++) {
    const a = F.a(n), b = F.b(n);
    if (a) s += a * Math.cos(n * x);
    if (b) s += b * Math.sin(n * x);
  }
  return s;
};
FA.energyFrac = function (key, N) {                 // [a₀²/2 + Σ_{n≤N}(aₙ² + bₙ²)] / ((1/π)∫f²)
  const F = FA.fns[key];
  let e = F.a0 * F.a0 / 2;
  for (let n = 1; n <= N; n++) e += F.a(n) * F.a(n) + F.b(n) * F.b(n);
  return e / F.energy;
};
FA.harmonics = function (key, K) {                  // the first K nonzero harmonics as (n, A, phase) for A·sin(nt + φ)
  const F = FA.fns[key], out = [];
  for (let n = 1; out.length < K && n < 400; n++) {
    const a = F.a(n), b = F.b(n);
    if (!a && !b) continue;
    // a·cos nt + b·sin nt = A·sin(nt + φ) with A = √(a² + b²), φ = atan2(a, b)
    out.push({ n, A: Math.hypot(a, b), ph: Math.atan2(a, b) });
  }
  return out;
};

/* ─────────────────── 1 · rotating phasors ───────────────────
   Caption audit (default sq, K = 4, t = 1): harmonics n = 1, 3, 5, 7 with lengths 4/(nπ) =
   1.273, 0.424, 0.255, 0.182; S₇(1) = 1.0066 (node: Σ 4/(nπ)·sin n over n = 1,3,5,7 = 1.006599),
   target f(1) = 1.                                                                          */
(function () {
  const svg = d3.select("#ph-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("ph-readout");
  let key = "sq", K = 4, t = 1, timer = null;
  const W = 680, H = 320, top = 26, bot = H - 30;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), F = FA.fns[key], hs = FA.harmonics(key, K);
    const sumA = d3.sum(hs, h => h.A), off = F.a0 / 2;
    const half = Math.max(sumA, Math.abs(F.yr[0] - off), Math.abs(F.yr[1] - off)) * 1.05;
    const ys = d3.scaleLinear().domain([off - half, off + half]).range([bot, top]);
    const k = (bot - top) / (2 * half);              // pixels per unit, same on both axes of the left panel
    const cx = 160;
    // left panel: the chain
    g.append("line").attr("x1", cx - half * k).attr("x2", cx + half * k).attr("y1", ys(off)).attr("y2", ys(off)).attr("stroke", SC.line);
    g.append("line").attr("x1", cx).attr("x2", cx).attr("y1", top).attr("y2", bot).attr("stroke", SC.line);
    let px = cx, py = ys(off);
    const pal = d3.schemeTableau10;
    hs.forEach((h, i) => {
      const ang = h.n * t + h.ph;
      g.append("circle").attr("cx", px).attr("cy", py).attr("r", h.A * k).attr("fill", "none").attr("stroke", SC.line).attr("stroke-dasharray", "2 3");
      const nx = px + h.A * k * Math.cos(ang), ny = py - h.A * k * Math.sin(ang);
      g.append("line").attr("x1", px).attr("y1", py).attr("x2", nx).attr("y2", ny).attr("stroke", pal[i % 10]).attr("stroke-width", 2);
      px = nx; py = ny;
    });
    g.append("circle").attr("cx", px).attr("cy", py).attr("r", 4).attr("fill", SC.a2);
    FA.label(g, 14, 16, `${hs.length} arrow${hs.length > 1 ? "s" : ""}: the n-th turns at n × the base rate`, { size: 10.5 });
    // right panel: the trace
    const x0 = 330, x1 = W - 14;
    const xs = d3.scaleLinear().domain([0, 2 * Math.PI]).range([x0, x1]);
    const tt = d3.range(0, 2 * Math.PI + 1e-9, 2 * Math.PI / 600);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${bot})`).call(d3.axisBottom(xs).tickValues([0, Math.PI / 2, Math.PI, 1.5 * Math.PI, 2 * Math.PI]).tickFormat(v => ["0", "π/2", "π", "3π/2", "2π"][Math.round(v / (Math.PI / 2))]));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(ys).ticks(5));
    const nMax = hs.length ? hs[hs.length - 1].n : 0;
    g.append("path").attr("d", d3.line().x(v => xs(v)).y(v => ys(F.f(v)))(tt)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 1.4);
    g.append("path").attr("d", d3.line().x(v => xs(v)).y(v => ys(FA.partial(key, nMax, v)))(tt)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.8);
    const S = FA.partial(key, nMax, t);
    g.append("line").attr("x1", px).attr("y1", py).attr("x2", xs(t)).attr("y2", ys(S)).attr("stroke", SC.a2).attr("stroke-dasharray", "4 3");
    g.append("line").attr("x1", xs(t)).attr("x2", xs(t)).attr("y1", top).attr("y2", bot).attr("stroke", SC.line);
    g.append("circle").attr("cx", xs(t)).attr("cy", ys(S)).attr("r", 4).attr("fill", SC.a2);
    FA.label(g, x0 + 4, 16, `grey: target · orange: S${FA.fmtN(nMax)}(t), the height of the last tip`, { size: 10.5 });
    const ft = F.f(t);
    out.innerHTML = `wave <b>${F.name}</b> · harmonics n = ${hs.map(h => h.n).join(", ")} · arrow lengths ${hs.map(h => ST.fmt(h.A, 3)).join(", ")}<br>` +
      `t = ${ST.fmt(t, 2)} · tip height S${FA.fmtN(nMax)}(t) = <b>${ST.fmt(S, 4)}</b> · target f(t) = ${ST.fmt(ft, 4)} · difference ${ST.fmt(S - ft, 4)}`;
  }
  const stop = () => { if (timer) { timer.stop(); timer = null; } $("ph-play").textContent = "Play"; };
  $("ph-fn").addEventListener("change", e => { key = e.target.value; draw(); });
  $("ph-n").addEventListener("input", e => { K = +e.target.value; $("ph-nv").textContent = K; draw(); });
  $("ph-t").addEventListener("input", e => { t = +e.target.value / 100; $("ph-tv").textContent = ST.fmt(t, 2); draw(); });
  $("ph-play").addEventListener("click", () => {
    if (timer) { stop(); return; }
    $("ph-play").textContent = "Pause";
    timer = d3.interval(() => {
      t = (t + 0.03) % (2 * Math.PI);
      $("ph-t").value = Math.round(t * 100); $("ph-tv").textContent = ST.fmt(t, 2);
      draw();
    }, 40);
  });
  $("ph-reset").addEventListener("click", () => {
    stop(); key = "sq"; K = 4; t = 1;
    $("ph-fn").value = "sq"; $("ph-n").value = 4; $("ph-nv").textContent = 4; $("ph-t").value = 100; $("ph-tv").textContent = "1.00";
    draw();
  });
  draw();
})();

/* ─────────────────── 2 · a coefficient is an area ───────────────────
   Caption audit (default sq, sin, n = 3): net area ∫ sign(x)·sin 3x dx = 4/3 = 1.3333 (Simpson split
   at 0, 4000 panels per half: 1.333333); ‖sin 3x‖² = π; b₃ = 0.4244 = 4/(3π). With cos 3x the area
   is 0 (odd integrand); with sin 2x it is 0 (even harmonic).                                     */
FA.coefClosed = function (key, kind, n) {          // the projection coefficient aₙ or bₙ (n = 0: a₀/2 for the constant)
  const F = FA.fns[key];
  if (kind === "cos") return n === 0 ? F.a0 / 2 : F.a(n);
  return n === 0 ? NaN : F.b(n);
};
(function () {
  const svg = d3.select("#pj-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("pj-readout");
  let key = "sq", kind = "sin", n = 3;
  const W = 680, x0 = 50, x1 = W - 16;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), F = FA.fns[key], P = Math.PI;
    const basis = x => kind === "cos" ? Math.cos(n * x) : Math.sin(n * x);
    const xs = d3.scaleLinear().domain([-P, P]).range([x0, x1]);
    const xx = d3.range(-P, P + 1e-9, 2 * P / 800).map(x => Math.min(Math.max(x, -P + 1e-9), P - 1e-9));
    const tickV = [-P, -P / 2, 0, P / 2, P], tickF = ["−π", "−π/2", "0", "π/2", "π"];
    // top panel
    const yT = d3.scaleLinear().domain([Math.min(F.yr[0], -1.2), Math.max(F.yr[1], 1.2)]).range([150, 22]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yT(0)})`).call(d3.axisBottom(xs).tickValues(tickV).tickFormat((v, i) => tickF[i]));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yT).ticks(4));
    g.append("path").attr("d", d3.line().defined(x => Math.abs(x) > 1e-6 || key !== "sq").x(x => xs(x)).y(x => yT(F.f(x)))(xx)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    if (!(kind === "sin" && n === 0)) g.append("path").attr("d", d3.line().x(x => xs(x)).y(x => yT(basis(x)))(xx)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);
    FA.label(g, x0 + 4, 14, `blue: f(x) = ${F.name} · orange: ψ(x) = ${kind} ${n}x`, { size: 10.5 });
    // bottom panel: the product, shaded by sign
    const prod = xx.map(x => ({ x, v: F.f(x) * basis(x) }));
    const pm = Math.max(1, d3.max(prod, d => Math.abs(d.v))) * 1.1;
    const yB = d3.scaleLinear().domain([-pm, pm]).range([340, 196]);
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yB).ticks(4));
    const area = d3.area().x(d => xs(d.x)).y0(yB(0));
    g.append("path").attr("d", area.y1(d => yB(Math.max(d.v, 0)))(prod)).attr("fill", SC.good).attr("fill-opacity", 0.45);
    g.append("path").attr("d", area.y1(d => yB(Math.min(d.v, 0)))(prod)).attr("fill", SC.bad).attr("fill-opacity", 0.45);
    g.append("path").attr("d", d3.line().x(d => xs(d.x)).y(d => yB(d.v))(prod)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.1);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yB(0)})`).call(d3.axisBottom(xs).tickValues(tickV).tickFormat((v, i) => tickF[i]));
    FA.label(g, x0 + 4, 188, "f(x)·ψ(x): green area counts +, red area counts −; the net area is ⟨f, ψ⟩", { size: 10.5 });
    // numbers
    if (kind === "sin" && n === 0) {
      out.innerHTML = "sin 0x ≡ 0 is not a basis function — choose n ≥ 1 for sines (the constant is cos 0x).";
      return;
    }
    const ip = FA.simpsonSplit(x => F.f(x) * basis(x), 4000);
    const nrm = (kind === "cos" && n === 0) ? 2 * P : P;
    const c = ip / nrm, cf = FA.coefClosed(key, kind, n);
    const nm = kind === "cos" ? (n === 0 ? "a₀/2" : `a${FA.fmtN(n)}`) : `b${FA.fmtN(n)}`;
    out.innerHTML = `net area ⟨f, ${kind} ${n}x⟩ = <b>${ST.fmt(ip, 4)}</b> (Simpson) · ‖${kind} ${n}x‖² = ${n === 0 ? "2π" : "π"} = ${ST.fmt(nrm, 4)}<br>` +
      `coefficient ${nm} = ${ST.fmt(ip, 4)} / ${ST.fmt(nrm, 4)} = <b>${ST.fmt(c, 4)}</b> · closed form ${ST.fmt(cf, 4)} · |difference| ${Math.abs(c - cf).toExponential(1)}`;
  }
  $("pj-fn").addEventListener("change", e => { key = e.target.value; draw(); });
  $("pj-kind").addEventListener("change", e => { kind = e.target.value; draw(); });
  $("pj-n").addEventListener("input", e => { n = +e.target.value; $("pj-nv").textContent = n; draw(); });
  $("pj-reset").addEventListener("click", () => {
    key = "sq"; kind = "sin"; n = 3;
    $("pj-fn").value = "sq"; $("pj-kind").value = "sin"; $("pj-n").value = 3; $("pj-nv").textContent = 3;
    draw();
  });
  draw();
})();

/* ─────────────────── 3 · harmonic builder + energy ledger ───────────────────
   Caption audit (default saw, N = 5): S₅(π/2) = 2(1 − 1/3 + 1/5) = 1.7333 vs π/2 = 1.5708; ledger
   (6/π²)(1 + 1/4 + 1/9 + 1/16 + 1/25) = 0.88977; relative L² error √(1 − 0.88977) = 0.3320.
   Square wave N = 1: 8/π² = 0.81057.                                                               */
(function () {
  const svg = d3.select("#hb-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("hb-readout");
  let key = "saw", N = 5;
  const W = 680, x0 = 50, x1 = W - 16, P = Math.PI;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), F = FA.fns[key];
    // top: partial sum over target
    const xs = d3.scaleLinear().domain([-P, P]).range([x0, x1]);
    const yT = d3.scaleLinear().domain(F.yr).range([226, 20]);
    const xx = d3.range(-P, P + 1e-9, 2 * P / 900);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yT(Math.max(F.yr[0], Math.min(0, F.yr[1])))})`)
      .call(d3.axisBottom(xs).tickValues([-P, -P / 2, 0, P / 2, P]).tickFormat((v, i) => ["−π", "−π/2", "0", "π/2", "π"][i]));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yT).ticks(5));
    const segs = key === "sq" ? [[-P, 0], [0, P]] : (key === "saw" ? [[-P, P]] : [[-P, P]]);
    segs.forEach(([a, b]) => {
      const pts = xx.filter(x => x > a + 1e-9 && x < b - 1e-9);
      g.append("path").attr("d", d3.line().x(x => xs(x)).y(x => yT(F.f(x)))(pts)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 1.5);
    });
    g.append("path").attr("d", d3.line().x(x => xs(x)).y(x => yT(FA.partial(key, N, x)))(xx)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
    const Sh = FA.partial(key, N, P / 2), fh = F.f(P / 2);
    g.append("circle").attr("cx", xs(P / 2)).attr("cy", yT(Sh)).attr("r", 4).attr("fill", SC.a2);
    g.append("circle").attr("cx", xs(P / 2)).attr("cy", yT(fh)).attr("r", 3.5).attr("fill", "none").attr("stroke", SC.ink);
    FA.label(g, x0 + 4, 12, `grey: ${F.name} · orange: S${FA.fmtN(N)}(x) · markers at x = π/2`, { size: 10.5 });
    // bottom left: stems
    const sx0 = 50, sx1 = 400, sy0 = 400, sy1 = 262;
    const nMax = 40;
    const coefs = d3.range(1, nMax + 1).map(n => ({ n, a: F.a(n), b: F.b(n) }));
    const cm = Math.max(Math.abs(F.a0 / 2), d3.max(coefs, d => Math.max(Math.abs(d.a), Math.abs(d.b))));
    const xS = d3.scaleLinear().domain([-0.5, nMax + 0.5]).range([sx0, sx1]);
    const yS = d3.scaleLinear().domain([-cm * 1.1, cm * 1.1]).range([sy0, sy1]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${sy0})`).call(d3.axisBottom(xS).ticks(8));
    g.append("g").attr("class", "axis").attr("transform", `translate(${sx0},0)`).call(d3.axisLeft(yS).ticks(4));
    g.append("line").attr("x1", sx0).attr("x2", sx1).attr("y1", yS(0)).attr("y2", yS(0)).attr("stroke", SC.line);
    const stems = [{ n: 0, v: F.a0 / 2, k: "a" }];
    coefs.forEach(d => { if (d.a) stems.push({ n: d.n, v: d.a, k: "a" }); if (d.b) stems.push({ n: d.n, v: d.b, k: "b" }); });
    stems.forEach(s => {
      if (s.v === 0) return;
      const on = s.n <= N, col = s.k === "a" ? SC.accent : SC.a2;
      g.append("line").attr("x1", xS(s.n)).attr("x2", xS(s.n)).attr("y1", yS(0)).attr("y2", yS(s.v)).attr("stroke", col).attr("stroke-opacity", on ? 1 : 0.25).attr("stroke-width", 1.6);
      g.append("circle").attr("cx", xS(s.n)).attr("cy", yS(s.v)).attr("r", 2.6).attr("fill", on ? col : "none").attr("stroke", col).attr("stroke-opacity", on ? 1 : 0.35);
    });
    FA.label(g, sx0, 254, "coefficients (blue: a₀/2 and aₙ, orange: bₙ); filled = included", { size: 10.5 });
    // bottom right: energy ledger
    const e = FA.energyFrac(key, N);
    const bx0 = 470, bx1 = W - 30, by = 300, bh = 34;
    const xE = d3.scaleLinear().domain([0, 1]).range([bx0, bx1]);
    g.append("rect").attr("x", bx0).attr("y", by).attr("width", bx1 - bx0).attr("height", bh).attr("fill", SC.panel2).attr("stroke", SC.line);
    g.append("rect").attr("x", bx0).attr("y", by).attr("width", xE(e) - bx0).attr("height", bh).attr("fill", SC.good).attr("fill-opacity", 0.75);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${by + bh})`).call(d3.axisBottom(xE).ticks(4).tickFormat(d3.format(".0%")));
    FA.label(g, bx0, 270, "energy ledger (Parseval)", { size: 11, color: SC.ink });
    FA.label(g, bx0, 288, "share of (1/π)∫f² captured by n ≤ N", { size: 10 });
    FA.label(g, (bx0 + bx1) / 2, by + 22, ST.pct(e, 2), { anchor: "middle", color: SC.ink, size: 13, weight: 700 });
    out.innerHTML = `${F.name}, harmonics up to N = ${N} · S${FA.fmtN(N)}(π/2) = <b>${ST.fmt(Sh, 4)}</b> vs f(π/2) = ${ST.fmt(fh, 4)}<br>` +
      `energy captured ${ST.pct(e, 2)} · relative L² error √(1 − ${ST.fmt(e, 4)}) = <b>${ST.pct(Math.sqrt(Math.max(0, 1 - e)), 1)}</b> · Parseval says the ledger → 100% as N → ∞`;
  }
  $("hb-fn").addEventListener("change", e => { key = e.target.value; draw(); });
  $("hb-n").addEventListener("input", e => { N = +e.target.value; $("hb-nv").textContent = N; draw(); });
  $("hb-reset").addEventListener("click", () => {
    key = "saw"; N = 5; $("hb-fn").value = "saw"; $("hb-n").value = 5; $("hb-nv").textContent = 5; draw();
  });
  draw();
})();

/* ─────────────────── 4 · Gibbs zoom ───────────────────
   Caption audit (default N = 19): peak 1.1798 at x = 0.1571 = π/20, overshoot (1.1798 − 1)/2 = 8.99%
   of the jump; limit (1/π)Si(π) − ½ = 0.0894899 (Si(π) = 1.8519370520 by Simpson, 20,000 panels).
   Fejér mean at N = 19: maximum 0.968 < 1.                                                         */
FA.sqPartial = (N, x) => { let s = 0; for (let n = 1; n <= N; n += 2) s += 4 / (n * Math.PI) * Math.sin(n * x); return s; };
FA.sqFejer = (N, x) => { let s = 0; for (let n = 1; n <= N; n += 2) s += (1 - n / (N + 1)) * 4 / (n * Math.PI) * Math.sin(n * x); return s; };
FA.Si = x => FA.simpson(v => (v === 0 ? 1 : Math.sin(v) / v), 0, x, 20000);
FA.gibbsPeak = function (N, fej) {                  // max of S_N (or σ_N) on (0, π/2], by a fine scan near the jump
  const fn = fej ? FA.sqFejer : FA.sqPartial;
  const hi = Math.min(Math.PI / 2, 3 * Math.PI / (N + 1)), K = 3000;
  let best = -Infinity, bx = 0;
  for (let i = 1; i <= K; i++) { const x = hi * i / K, v = fn(N, x); if (v > best) { best = v; bx = x; } }
  if (fej) for (let i = 1; i <= 400; i++) { const x = Math.PI / 2 * i / 400, v = fn(N, x); if (v > best) { best = v; bx = x; } }
  return { peak: best, at: bx };
};
(function () {
  const svg = d3.select("#gb-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("gb-readout");
  let M = 10, fej = false;
  const W = 680, H = 320, P = Math.PI, LIM = 2 / P * FA.Si(P);
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), N = 2 * M - 1, xstar = P / (N + 1);
    // left: full period
    const lx0 = 44, lx1 = 330, top = 24, bot = H - 34;
    const xs = d3.scaleLinear().domain([-P, P]).range([lx0, lx1]);
    const ys = d3.scaleLinear().domain([-1.4, 1.4]).range([bot, top]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ys(0)})`).call(d3.axisBottom(xs).tickValues([-P, 0, P]).tickFormat((v, i) => ["−π", "0", "π"][i]));
    g.append("g").attr("class", "axis").attr("transform", `translate(${lx0},0)`).call(d3.axisLeft(ys).ticks(5));
    g.append("path").attr("d", `M${xs(-P)},${ys(-1)}H${xs(0)}M${xs(0)},${ys(1)}H${xs(P)}`).attr("stroke", SC.muted).attr("stroke-width", 1.4).attr("fill", "none");
    const xx = d3.range(-P, P + 1e-9, 2 * P / 1600);
    g.append("path").attr("d", d3.line().x(x => xs(x)).y(x => ys(FA.sqPartial(N, x)))(xx)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.4);
    if (fej) g.append("path").attr("d", d3.line().x(x => xs(x)).y(x => ys(FA.sqFejer(N, x)))(xx)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 1.4);
    FA.label(g, lx0 + 4, 14, `one period, harmonics up to N = ${N}`, { size: 10.5 });
    // right: zoom, x in units of π/(N + 1)
    const rx0 = 380, rx1 = W - 16;
    const us = d3.scaleLinear().domain([0, 6]).range([rx0, rx1]);
    const yz = d3.scaleLinear().domain([0, 1.3]).range([bot, top]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${bot})`).call(d3.axisBottom(us).ticks(6));
    g.append("g").attr("class", "axis").attr("transform", `translate(${rx0},0)`).call(d3.axisLeft(yz).ticks(6));
    [[1, SC.muted, "1"], [LIM, SC.bad, ST.fmt(LIM, 4) + " limit"]].forEach(([v, c, t]) => {
      g.append("line").attr("x1", rx0).attr("x2", rx1).attr("y1", yz(v)).attr("y2", yz(v)).attr("stroke", c).attr("stroke-dasharray", "4 3");
      FA.label(g, rx1, yz(v) - 4, t, { anchor: "end", color: c, size: 10 });
    });
    const uu = d3.range(0, 6.0001, 0.01);
    g.append("path").attr("d", d3.line().x(u => us(u)).y(u => yz(FA.sqPartial(N, u * xstar)))(uu)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
    if (fej) g.append("path").attr("d", d3.line().x(u => us(u)).y(u => yz(FA.sqFejer(N, u * xstar)))(uu)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2);
    const pk = FA.gibbsPeak(N, false);
    g.append("circle").attr("cx", us(pk.at / xstar)).attr("cy", yz(pk.peak)).attr("r", 4).attr("fill", SC.a2);
    FA.label(g, rx0 + 4, 14, "zoom: x measured in units of π/(N + 1)", { size: 10.5 });
    FA.label(g, rx1, bot + 28, "x·(N + 1)/π", { anchor: "end", size: 10 });
    let txt = `N = ${N}: peak S${FA.fmtN(N)}(x*) = <b>${ST.fmt(pk.peak, 4)}</b> at x* = ${ST.fmt(pk.at, 4)} (π/(N + 1) = ${ST.fmt(xstar, 4)}) · ` +
      `overshoot <b>${ST.fmt((pk.peak - 1) / 2 * 100, 3)}%</b> of the jump · limit (1/π)Si(π) − ½ = ${ST.fmt((LIM - 1) / 2 * 100, 3)}%`;
    if (fej) { const fp = FA.gibbsPeak(N, true); txt += `<br>Fejér mean: largest value ${ST.fmt(fp.peak, 4)} — no overshoot (the Fejér kernel is non-negative)`; }
    out.innerHTML = txt;
  }
  $("gb-m").addEventListener("input", e => { M = +e.target.value; $("gb-mv").textContent = 2 * M - 1; draw(); });
  $("gb-fej").addEventListener("change", e => { fej = e.target.checked; draw(); });
  $("gb-reset").addEventListener("click", () => { M = 10; fej = false; $("gb-m").value = 10; $("gb-mv").textContent = 19; $("gb-fej").checked = false; draw(); });
  draw();
})();

/* ─────────────────── 5 · smoothness versus decay ───────────────────
   Caption audit (default c1): nonzero bₙ = 8/(πn³) at odd n; least-squares slope of log|bₙ| on log n
   over odd 5 ≤ n ≤ 41 is exactly −3 (exact power law); b₁₁ = 8/(1331π) = 0.0019132; Simpson
   (split at 0, 20,000 panels per half) agrees to 1e-12.                                           */
FA.ladder = ["sq", "tri", "c1", "c2", "an"];
FA.ladderCol = { sq: SC.bad, tri: SC.a2, c1: SC.good, c2: SC.accent, an: SC.violet };
FA.coefMag = function (key, n) { const F = FA.fns[key]; return Math.hypot(F.a(n), F.b(n)); };
FA.decaySlope = function (key) {
  const pts = [];
  for (let n = 5; n <= 41; n++) { const c = FA.coefMag(key, n); if (c > 0) pts.push([Math.log(n), Math.log(c)]); }
  const mx = d3.mean(pts, p => p[0]), my = d3.mean(pts, p => p[1]);
  let sxy = 0, sxx = 0;
  pts.forEach(p => { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) * (p[0] - mx); });
  const b = sxy / sxx;
  return { slope: b, icpt: my - b * mx };
};
(function () {
  const svg = d3.select("#dc-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("dc-readout");
  let sel = "c1";
  const W = 680, H = 340, x0 = 56, x1 = W - 150, top = 16, bot = H - 36;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const xs = d3.scaleLog().domain([1, 101]).range([x0, x1]);
    const ys = d3.scaleLog().domain([1e-9, 10]).range([bot, top]).clamp(true);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${bot})`).call(d3.axisBottom(xs).tickValues([1, 2, 5, 10, 20, 50, 100]).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(ys).tickValues([1e-8, 1e-6, 1e-4, 1e-2, 1]).tickFormat(v => "10" + (Math.round(Math.log10(v)) === 0 ? "⁰" : "⁻" + "⁰¹²³⁴⁵⁶⁷⁸⁹"[-Math.round(Math.log10(v))])));
    FA.label(g, x1, bot + 30, "n (log scale)", { anchor: "end", size: 10 });
    FA.label(g, x0 + 4, top + 2, "|coefficient| (log scale)", { size: 10 });
    FA.ladder.forEach((k, i) => {
      const on = k === sel, col = FA.ladderCol[k];
      const pts = d3.range(1, 102).map(n => ({ n, c: FA.coefMag(k, n) })).filter(p => p.c >= 1e-9);
      g.append("g").selectAll("circle").data(pts).join("circle").attr("cx", p => xs(p.n)).attr("cy", p => ys(p.c))
        .attr("r", on ? 3 : 1.8).attr("fill", col).attr("fill-opacity", on ? 1 : 0.35);
      g.append("rect").attr("x", x1 + 18).attr("y", top + 14 + i * 20 - 5).attr("width", 10).attr("height", 10).attr("fill", col).attr("fill-opacity", on ? 1 : 0.4);
      FA.label(g, x1 + 34, top + 14 + i * 20 + 4, ["sign(x)  1/n", "|x|  1/n²", "x(π−|x|)  1/n³", "…−|x|³/3  1/n⁴", "analytic  rⁿ"][i], { size: 10, color: on ? SC.ink : SC.muted });
    });
    const fit = FA.decaySlope(sel);
    if (sel !== "an") {
      const L = n => Math.exp(fit.icpt + fit.slope * Math.log(n));
      g.append("line").attr("x1", xs(1)).attr("x2", xs(101)).attr("y1", ys(L(1))).attr("y2", ys(L(101))).attr("stroke", FA.ladderCol[sel]).attr("stroke-dasharray", "5 3");
    }
    const F = FA.fns[sel], P = Math.PI, n = 11;
    const closed = F.a(n) || F.b(n), useCos = F.a(n) !== 0;
    const quad = FA.simpsonSplit(x => F.f(x) * (useCos ? Math.cos(n * x) : Math.sin(n * x)), 20000) / P;
    out.innerHTML = `${F.name}: least-squares slope of log|coef| on log n over 5 ≤ n ≤ 41 = <b>${ST.fmt(fit.slope, 3)}</b>` +
      (sel === "an" ? " (not a power law: the points bend down — geometric decay 2·0.7ⁿ)" : "") +
      `<br>${useCos ? "a" : "b"}₁₁: closed form ${FA.f(closed, 7)} · Simpson quadrature of the defining integral ${FA.f(quad, 7)} · |difference| ${Math.abs(closed - quad).toExponential(1)}`;
  }
  $("dc-fn").addEventListener("change", e => { sel = e.target.value; draw(); });
  $("dc-reset").addEventListener("click", () => { sel = "c1"; $("dc-fn").value = "c1"; draw(); });
  draw();
})();

/* ─────────────────── 6 · series → transform ───────────────────
   Caption audit (default a = 1, P = 8): Δω = 2π/8 = 0.7854; c₀ = 2a/P = 0.25; stems with
   |ωₙ| < π/a: n = −3…3 → 7 (n = ±4 sits on the zero ω = π); P·c₁ by Simpson = 1.800633 =
   F(π/4) = 2·sin(π/4)/(π/4).                                                                     */
FA.boxF = (a, w) => (Math.abs(w) < 1e-12 ? 2 * a : 2 * Math.sin(a * w) / w);
(function () {
  const svg = d3.select("#st-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("st-readout");
  let Pd = 8, a = 1;
  const W = 680, x0 = 50, x1 = W - 16, PI = Math.PI;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g");
    // top: the pulse train
    const xs = d3.scaleLinear().domain([-20, 20]).range([x0, x1]);
    const yt = d3.scaleLinear().domain([-0.2, 1.3]).range([110, 24]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yt(0)})`).call(d3.axisBottom(xs).ticks(9));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yt).ticks(2));
    for (let k = -Math.ceil(25 / Pd); k <= Math.ceil(25 / Pd); k++) {
      const c = k * Pd, l = Math.max(-20, c - a), r = Math.min(20, c + a);
      if (r <= l) continue;
      g.append("rect").attr("x", xs(l)).attr("y", yt(1)).attr("width", xs(r) - xs(l)).attr("height", yt(0) - yt(1)).attr("fill", k === 0 ? SC.a2 : SC.accent).attr("fill-opacity", k === 0 ? 0.8 : 0.45);
    }
    FA.label(g, x0 + 4, 14, `pulses of half-width a = ${ST.fmt(a, 2)} repeated every P = ${ST.fmt(Pd, 1)} (orange: the single pulse)`, { size: 10.5 });
    // bottom: envelope and stems
    const wmax = 4 * PI;
    const ws = d3.scaleLinear().domain([-wmax, wmax]).range([x0, x1]);
    const yb = d3.scaleLinear().domain([-0.6, 2.6]).range([336, 150]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yb(0)})`).call(d3.axisBottom(ws).tickValues([-4, -3, -2, -1, 0, 1, 2, 3, 4].map(k => k * PI)).tickFormat(v => { const k = Math.round(v / PI); return k === 0 ? "0" : (k === 1 ? "π" : (k === -1 ? "−π" : (k < 0 ? "−" : "") + Math.abs(k) + "π")); }));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yb).ticks(5));
    const ww = d3.range(-wmax, wmax + 1e-9, wmax / 600);
    g.append("path").attr("d", d3.line().x(w => ws(w)).y(w => yb(FA.boxF(a, w)))(ww)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.4);
    const dw = 2 * PI / Pd, nM = Math.floor(wmax / dw);
    let inLobe = 0;
    for (let n = -nM; n <= nM; n++) {
      const w = n * dw, v = FA.boxF(a, w);               // P·cₙ = F(ωₙ): the coefficient of the periodised pulse
      if (Math.abs(w) < PI / a - 1e-9) inLobe++;
      g.append("line").attr("x1", ws(w)).attr("x2", ws(w)).attr("y1", yb(0)).attr("y2", yb(v)).attr("stroke", SC.a2).attr("stroke-width", 1.5);
      g.append("circle").attr("cx", ws(w)).attr("cy", yb(v)).attr("r", 2.6).attr("fill", SC.a2);
    }
    FA.label(g, x0 + 4, 146, "white: F(ω) = 2·sin(aω)/ω, the transform of ONE pulse · orange stems: P·cₙ at ωₙ = 2πn/P", { size: 10.5 });
    const w1 = dw, q = FA.simpson(x => Math.cos(w1 * x), -a, a, 4000);
    out.innerHTML = `stem spacing Δω = 2π/P = <b>${ST.fmt(dw, 4)}</b> · duty cycle c₀ = 2a/P = ${ST.fmt(2 * a / Pd, 4)} · stems inside the main lobe |ω| &lt; π/a: <b>${inLobe}</b><br>` +
      `P·c₁ by quadrature ∫ cos(ω₁x) dx over [−a, a] = ${ST.fmt(q, 6)} · F(ω₁) from the formula = ${ST.fmt(FA.boxF(a, w1), 6)}`;
  }
  $("st-p").addEventListener("input", e => { Pd = +e.target.value; $("st-pv").textContent = ST.fmt(Pd, 1); draw(); });
  $("st-a").addEventListener("input", e => { a = +e.target.value; $("st-av").textContent = ST.fmt(a, 2); draw(); });
  $("st-reset").addEventListener("click", () => { Pd = 8; a = 1; $("st-p").value = 8; $("st-pv").textContent = "8.0"; $("st-a").value = 1; $("st-av").textContent = "1.00"; draw(); });
  draw();
})();

/* ─────────────────── 7 · characteristic-function gallery ───────────────────
   Left panel computed in the ORIGINAL domain (PV.convolve of pmfs or of grid masses, or the known
   closed-form law of the sum); right panel in the FREQUENCY domain (φ(t)ⁿ or φ(t/n)ⁿ). The readout
   transforms the left panel numerically at t = 1 and compares with the formula.
   Caption audit (default unif, n = 3, sum): grid h = 0.005, trapezoid masses, two convolutions;
   Σ m·cos(x) over the convolved grid = 0.595820 vs (sin 1)³ = 0.595823 (|diff| 3.7e-6); imaginary
   part 0 by symmetry. Cauchy, mean: right panel e^(−|t|) for every n.                              */
FA.cx = {
  mul: (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
  pow(a, n) { let r = [1, 0]; for (let i = 0; i < n; i++) r = FA.cx.mul(r, a); return r; }
};
FA.cf = {                                           // φ(t) = E e^(itX) as [Re, Im]
  unif: t => [Math.abs(t) < 1e-12 ? 1 : Math.sin(t) / t, 0],
  norm: t => [Math.exp(-t * t / 2), 0],
  exp:  t => [1 / (1 + t * t), t / (1 + t * t)],
  lap:  t => [1 / (1 + t * t), 0],
  cau:  t => [Math.exp(-Math.abs(t)), 0],
  bern: t => [0.7 + 0.3 * Math.cos(t), 0.3 * Math.sin(t)],
  pois: t => { const m = Math.exp(2 * (Math.cos(t) - 1)); return [m * Math.cos(2 * Math.sin(t)), m * Math.sin(2 * Math.sin(t))]; }
};
FA.cfLabel = { unif: "Uniform(−1, 1)", norm: "Normal(0, 1)", exp: "Exponential(1)", lap: "Laplace(0, 1)", cau: "Cauchy(0, 1)", bern: "Bernoulli(0.3)", pois: "Poisson(2)" };
FA.gridMasses = function (pdf, lo, hi, h) {         // trapezoid masses of a density on an h-grid
  const K = Math.round((hi - lo) / h), m = [];
  for (let j = 0; j <= K; j++) m.push(pdf(lo + j * h) * h * ((j === 0 || j === K) ? 0.5 : 1));
  return m;
};
FA.sumLaw = function (d, n) {
  // returns { kind: "grid"|"pdf"|"pmf", … } describing the law of X₁ + … + Xₙ
  if (d === "bern" || d === "pois") {
    const one = d === "bern" ? [0.7, 0.3] : d3.range(0, 41).map(k => PV.poisPmf(k, 2));
    let acc = one.slice();
    for (let i = 1; i < n; i++) acc = PV.convolve(acc, one);
    return { kind: "pmf", p: acc };
  }
  if (d === "unif" || d === "lap") {
    const h = d === "unif" ? 0.005 : 0.025, L = d === "unif" ? 1 : 10;
    const pdf = d === "unif" ? (x => (Math.abs(x) <= 1 + 1e-12 ? 0.5 : 0)) : (x => 0.5 * Math.exp(-Math.abs(x)));
    const one = FA.gridMasses(pdf, -L, L, h);
    let acc = one.slice();
    for (let i = 1; i < n; i++) acc = PV.convolve(acc, one);
    return { kind: "grid", m: acc, x0: -n * L, h };
  }
  if (d === "norm") return { kind: "pdf", pdf: x => ST.normPdf(x / Math.sqrt(n)) / Math.sqrt(n), lo: -12 * Math.sqrt(n), hi: 12 * Math.sqrt(n) };
  if (d === "exp") return { kind: "pdf", pdf: x => (x < 0 ? 0 : Math.exp((n - 1) * Math.log(Math.max(x, 1e-300)) - x - ST.lnGamma(n))), lo: 0, hi: n + 60 };
  return { kind: "cauchy", s: n };                   // Cauchy(0, n)
};
FA.lawTransform = function (law, t, scale) {        // ∫ e^(itx) dP(x) of the law of (sum / scale), numerically
  const re = [0], im = [0];
  if (law.kind === "pmf") { law.p.forEach((p, k) => { re[0] += p * Math.cos(t * k / scale); im[0] += p * Math.sin(t * k / scale); }); }
  else if (law.kind === "grid") { law.m.forEach((m, j) => { const x = (law.x0 + j * law.h) / scale; re[0] += m * Math.cos(t * x); im[0] += m * Math.sin(t * x); }); }
  else if (law.kind === "pdf") {
    re[0] = FA.simpson(x => law.pdf(x) * Math.cos(t * x / scale), law.lo, law.hi, 20000);
    im[0] = FA.simpson(x => law.pdf(x) * Math.sin(t * x / scale), law.lo, law.hi, 20000);
  } else {                                           // Cauchy(0, s): substitute x = s·tan θ; the sine part vanishes by symmetry
    const s = law.s / scale, K = 400000, hθ = Math.PI / K;
    let acc = 0;
    for (let i = 0; i < K; i++) { const th = -Math.PI / 2 + (i + 0.5) * hθ; acc += Math.cos(t * s * Math.tan(th)); }
    re[0] = acc * hθ / Math.PI;
  }
  return [re[0], im[0]];
};
(function () {
  const svg = d3.select("#cf-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("cf-readout");
  let d = "unif", n = 3, mode = "sum";
  const W = 680, H = 330, top = 24, bot = H - 34;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), law = FA.sumLaw(d, n), sc = mode === "mean" ? n : 1;
    // ── left: the law of the sum / mean
    const lx0 = 46, lx1 = 320;
    let range;
    if (d === "unif") range = [-n - 0.3, n + 0.3];
    else if (d === "norm") range = [-4 * Math.sqrt(n), 4 * Math.sqrt(n)];
    else if (d === "exp") range = [0, n + 5 * Math.sqrt(n) + 2];
    else if (d === "lap") range = [-(4 * Math.sqrt(2 * n) + 1), 4 * Math.sqrt(2 * n) + 1];
    else if (d === "cau") range = [-10 * n, 10 * n];
    else if (d === "bern") range = [-0.5, n + 0.5];
    else range = [-0.5, 2 * n + 5 * Math.sqrt(2 * n) + 2];
    range = range.map(v => v / sc);
    const xs = d3.scaleLinear().domain(range).range([lx0, lx1]);
    let pts = [], disc = false;
    if (law.kind === "pmf") { disc = true; pts = law.p.map((p, k) => ({ x: k / sc, y: p })).filter(p => p.x >= range[0] && p.x <= range[1]); }
    else if (law.kind === "grid") { const step = Math.max(1, Math.round(law.m.length / 700)); for (let j = 0; j < law.m.length; j += step) { const x = (law.x0 + j * law.h) / sc; if (x >= range[0] && x <= range[1]) pts.push({ x, y: law.m[j] / law.h * sc }); } }
    else if (law.kind === "pdf") { d3.range(range[0], range[1], (range[1] - range[0]) / 600).forEach(x => pts.push({ x, y: law.pdf(x * sc) * sc })); }
    else { d3.range(range[0], range[1], (range[1] - range[0]) / 600).forEach(x => pts.push({ x, y: PV.cauchyPdf(x, 0, law.s / sc) })); }
    const ymax = d3.max(pts, p => p.y) * 1.12 || 1;
    const ys = d3.scaleLinear().domain([0, ymax]).range([bot, top]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${bot})`).call(d3.axisBottom(xs).ticks(6));
    g.append("g").attr("class", "axis").attr("transform", `translate(${lx0},0)`).call(d3.axisLeft(ys).ticks(4));
    if (disc) pts.forEach(p => {
      g.append("line").attr("x1", xs(p.x)).attr("x2", xs(p.x)).attr("y1", ys(0)).attr("y2", ys(p.y)).attr("stroke", SC.accent).attr("stroke-width", 2);
      g.append("circle").attr("cx", xs(p.x)).attr("cy", ys(p.y)).attr("r", 2.5).attr("fill", SC.accent);
    });
    else g.append("path").attr("d", d3.area().x(p => xs(p.x)).y0(ys(0)).y1(p => ys(p.y))(pts)).attr("fill", SC.accent).attr("fill-opacity", 0.35).attr("stroke", SC.accent).attr("stroke-width", 1.5);
    FA.label(g, lx0, 14, `${mode} of ${n} × ${FA.cfLabel[d]}: ${disc ? "pmf" : "density"}, by convolution`, { size: 10.5 });
    // ── right: φ of the sum / mean
    const rx0 = 372, rx1 = W - 14, T = 2 * Math.PI;
    const ts = d3.scaleLinear().domain([-T, T]).range([rx0, rx1]);
    const yc = d3.scaleLinear().domain([-1.08, 1.08]).range([bot, top]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yc(0)})`).call(d3.axisBottom(ts).tickValues([-T, -Math.PI, 0, Math.PI, T]).tickFormat((v, i) => ["−2π", "−π", "0", "π", "2π"][i]));
    g.append("g").attr("class", "axis").attr("transform", `translate(${rx0},0)`).call(d3.axisLeft(yc).ticks(5));
    const phiN = t => FA.cx.pow(FA.cf[d](t / sc), n);
    const tt = d3.range(-T, T + 1e-9, T / 400);
    const vals = tt.map(t => ({ t, v: phiN(t) }));
    g.append("path").attr("d", d3.line().x(p => ts(p.t)).y(p => yc(Math.hypot(p.v[0], p.v[1])))(vals)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-dasharray", "4 3");
    g.append("path").attr("d", d3.line().x(p => ts(p.t)).y(p => yc(p.v[1]))(vals)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);
    g.append("path").attr("d", d3.line().x(p => ts(p.t)).y(p => yc(p.v[0]))(vals)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    g.append("line").attr("x1", ts(1)).attr("x2", ts(1)).attr("y1", top).attr("y2", bot).attr("stroke", SC.line).attr("stroke-dasharray", "2 3");
    FA.label(g, rx0, 14, `φ(t) = [φ₁(t${mode === "mean" ? "/n" : ""})]ⁿ: Re blue · Im orange · |φ| dashed`, { size: 10.5 });
    // ── readout: the two domains agree at t = 1
    const num = FA.lawTransform(law, 1, sc), cl = phiN(1);
    const diff = Math.hypot(num[0] - cl[0], num[1] - cl[1]);
    out.innerHTML = `${mode} of n = ${n} independent ${FA.cfLabel[d]} · at t = 1:<br>` +
      `transform of the left panel (computed by ${law.kind === "pmf" ? "the exact sum Σ pₖe^(itk)" : law.kind === "grid" ? "summing the convolved grid masses" : law.kind === "pdf" ? "Simpson quadrature" : "quadrature in θ, x = s·tan θ"}) = <b>${ST.fmt(num[0], 5)} ${num[1] < 0 ? "−" : "+"} ${ST.fmt(Math.abs(num[1]), 5)}i</b>` +
      ` · formula [φ₁(${mode === "mean" ? "1/n" : "1"})]ⁿ = <b>${ST.fmt(cl[0], 5)} ${cl[1] < 0 ? "−" : "+"} ${ST.fmt(Math.abs(cl[1]), 5)}i</b> · |difference| ${diff.toExponential(1)}`;
  }
  $("cf-dist").addEventListener("change", e => { d = e.target.value; draw(); });
  $("cf-n").addEventListener("input", e => { n = +e.target.value; $("cf-nv").textContent = n; draw(); });
  $("cf-mode").addEventListener("change", e => { mode = e.target.value; draw(); });
  $("cf-reset").addEventListener("click", () => { d = "unif"; n = 3; mode = "sum"; $("cf-dist").value = "unif"; $("cf-n").value = 3; $("cf-nv").textContent = 3; $("cf-mode").value = "sum"; draw(); });
  draw();
})();

/* ─────────────────── 8 · periodogram lab ───────────────────
   Signal yₜ = cos(2πt/12) + 0.5·cos(2πt/6 + 0.7) + 0.6·cos(2πt/40 + 1.9) + noise, t = 0…n − 1, noise from
   ST.rng(seed) via ST.randn: white σ·Z, or AR(1) xₜ = 0.6xₜ₋₁ + σ·Zₜ after a 200-step burn-in.
   I(fₖ) = (1/n)|Σ(yₜ − ȳ)e^(−2πikt/n)|², k = 1…n/2 − 1; Daniell smoother of half-width m.
   Caption audit (default n = 240, white, σ = 1, m = 3, seed 11): checked for seeds 1–20 (all put the top three at 6, 20, 40);
   seed 11: raw CV 1.011, smoothed 0.352 (theory 1/√7 = 0.378); n = 960: raw CV 0.928.               */
FA.pgSignal = function (n, noise, sig, seed) {
  const r = ST.rng(seed), y = new Array(n);
  let ar = 0;
  if (noise === "ar") for (let i = 0; i < 200; i++) ar = 0.6 * ar + sig * ST.randn(r);
  for (let t = 0; t < n; t++) {
    let e;
    if (noise === "ar") { ar = 0.6 * ar + sig * ST.randn(r); e = ar; } else e = sig * ST.randn(r);
    y[t] = Math.cos(2 * Math.PI * t / 12) + 0.5 * Math.cos(2 * Math.PI * t / 6 + 0.7) + 0.6 * Math.cos(2 * Math.PI * t / 40 + 1.9) + e;
  }
  return y;
};
FA.pgTrueS = (f, noise, sig) => noise === "ar" ? sig * sig / (1 - 1.2 * Math.cos(2 * Math.PI * f) + 0.36) : sig * sig;
FA.periodogram = function (y) {
  const n = y.length, mu = d3.mean(y), K = n / 2 - 1, I = [0];
  for (let k = 1; k <= K; k++) {
    let re = 0, im = 0;
    const w = 2 * Math.PI * k / n;
    for (let t = 0; t < n; t++) { const v = y[t] - mu; re += v * Math.cos(w * t); im -= v * Math.sin(w * t); }
    I.push((re * re + im * im) / n);
  }
  return I;                                         // I[k], k = 1…n/2 − 1 (I[0] unused)
};
FA.daniell = function (I, m) {
  const K = I.length - 1, out = [0];
  for (let k = 1; k <= K; k++) {
    let s = 0, c = 0;
    for (let j = k - m; j <= k + m; j++) if (j >= 1 && j <= K) { s += I[j]; c++; }
    out.push(s / c);
  }
  return out;
};
FA.pgStats = function (n, noise, sig, m, seed) {
  const y = FA.pgSignal(n, noise, sig, seed), I = FA.periodogram(y), Sm = FA.daniell(I, m);
  const K = I.length - 1, truth = [n / 40, n / 12, n / 6], amps = [0.6, 1, 0.5];
  const order = d3.range(1, K + 1).sort((a, b) => I[b] - I[a]).slice(0, 3);
  const band = d3.range(1, K + 1).filter(k => truth.every(tk => Math.abs(k - tk) > m + 1) && k > m && k <= K - m);
  const cv = arr => { const v = band.map(k => arr[k] / FA.pgTrueS(k / n, noise, sig)); return d3.deviation(v) / d3.mean(v); };
  return { y, I, Sm, K, truth, amps, top: order, cvRaw: cv(I), cvSm: cv(Sm), nBand: band.length };
};
(function () {
  const svg = d3.select("#pg-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("pg-readout");
  let n = 240, noise = "white", sig = 1, m = 3, seed = FA.seeds.pg;
  const W = 680, x0 = 50, x1 = W - 16;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), R = FA.pgStats(n, noise, sig, m, seed);
    // top: the series
    const xs = d3.scaleLinear().domain([0, n - 1]).range([x0, x1]);
    const ext = d3.extent(R.y), pad = 0.05 * (ext[1] - ext[0]);
    const yt = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([120, 22]);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,120)").call(d3.axisBottom(xs).ticks(8));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yt).ticks(3));
    g.append("path").attr("d", d3.line().x((v, i) => xs(i)).y(v => yt(v))(R.y)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1);
    FA.label(g, x0 + 4, 14, `the series, n = ${n} (time steps; think months)`, { size: 10.5 });
    // bottom: log periodogram
    const fs = d3.scaleLinear().domain([0, 0.5]).range([x0, x1]);
    const vals = R.I.slice(1).concat(R.Sm.slice(1)).filter(v => v > 0);
    const ylo = Math.max(1e-3, d3.min(vals) * 0.7), yhi = d3.max(vals) * 1.6;
    const yb = d3.scaleLog().domain([ylo, yhi]).range([370, 160]).clamp(true);
    g.append("g").attr("class", "axis").attr("transform", "translate(0,370)").call(d3.axisBottom(fs).ticks(10));
    g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yb).ticks(5, "~g"));
    FA.label(g, x1, 398, "frequency f (cycles per step)", { anchor: "end", size: 10 });
    R.truth.forEach(k => g.append("line").attr("x1", fs(k / n)).attr("x2", fs(k / n)).attr("y1", 160).attr("y2", 370).attr("stroke", SC.violet).attr("stroke-dasharray", "1 3"));
    const ks = d3.range(1, R.K + 1);
    g.append("path").attr("d", d3.line().x(k => fs(k / n)).y(k => yb(Math.max(ylo, R.I[k])))(ks)).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 0.9);
    g.append("path").attr("d", d3.line().x(k => fs(k / n)).y(k => yb(FA.pgTrueS(k / n, noise, sig)))(ks)).attr("fill", "none").attr("stroke", SC.good).attr("stroke-dasharray", "5 3").attr("stroke-width", 1.4);
    g.append("path").attr("d", d3.line().x(k => fs(k / n)).y(k => yb(Math.max(ylo, R.Sm[k])))(ks)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
    R.top.forEach(k => g.append("circle").attr("cx", fs(k / n)).attr("cy", yb(R.I[k])).attr("r", 3.5).attr("fill", "none").attr("stroke", SC.ink));
    FA.label(g, x0 + 4, 152, `log scale · grey raw I(fₖ) · orange Daniell m = ${m} · green dashed true noise S(f) · dotted: true cycle frequencies`, { size: 10.5 });
    const rows = R.top.slice().sort((a, b) => a - b).map(k => {
      const ti = R.truth.indexOf(k), exp = ti >= 0 ? n * R.amps[ti] * R.amps[ti] / 4 + FA.pgTrueS(k / n, noise, sig) : FA.pgTrueS(k / n, noise, sig);
      return `k = ${k} (f = ${ST.fmt(k / n, 4)}, period ${ST.fmt(n / k, 1)}): I = ${ST.fmt(R.I[k], 2)}, expected ${ST.fmt(exp, 2)} ${ti >= 0 ? "✓ true cycle" : "✗ noise"}`;
    });
    out.innerHTML = `seed ${seed} · three largest raw ordinates: ${rows.join(" · ")}<br>` +
      `true cycles at k = ${R.truth.join(", ")} · scatter of I/S over ${R.nBand} noise-only frequencies: raw CV <b>${ST.fmt(R.cvRaw, 3)}</b> (theory 1) · smoothed CV <b>${ST.fmt(R.cvSm, 3)}</b> (theory 1/√${2 * m + 1} = ${ST.fmt(1 / Math.sqrt(2 * m + 1), 3)})`;
  }
  $("pg-n").addEventListener("change", e => { n = +e.target.value; draw(); });
  $("pg-noise").addEventListener("change", e => { noise = e.target.value; draw(); });
  $("pg-s").addEventListener("input", e => { sig = +e.target.value; $("pg-sv").textContent = ST.fmt(sig, 1); draw(); });
  $("pg-m").addEventListener("input", e => { m = +e.target.value; $("pg-mv").textContent = m; draw(); });
  $("pg-new").addEventListener("click", () => { seed += 1; draw(); });
  $("pg-reset").addEventListener("click", () => {
    n = 240; noise = "white"; sig = 1; m = 3; seed = FA.seeds.pg;
    $("pg-n").value = "240"; $("pg-noise").value = "white"; $("pg-s").value = 1; $("pg-sv").textContent = "1.0"; $("pg-m").value = 3; $("pg-mv").textContent = 3;
    draw();
  });
  draw();
})();

/* ─────────────────── 9 · random Fourier features ───────────────────
   k̂(Δ) = (1/D)Σ cos(wⱼΔ), wⱼ from the spectral measure (ST.randn / PV.cauchyDraw / PV.expDraw with a
   random sign), seeded by ST.rng(seed). Theoretical sd at Δ: √([½ + ½k(2Δ) − k(Δ)²]/D).
   Caption audit (default gauss, D = 64, seed 9): sd at Δ = 1 = 0.0559;
   seed 9: max error 0.1606, RMS 0.0918 (theory 0.0776), 100% of the grid inside ±2 sd; D = 1024: RMS 0.0190 (theory 0.0194). */
FA.rfKernel = { gauss: d => Math.exp(-d * d / 2), lap: d => Math.exp(-Math.abs(d)), cau: d => 1 / (1 + d * d) };
FA.rfSpec = { gauss: w => ST.normPdf(w), lap: w => PV.cauchyPdf(w, 0, 1), cau: w => 0.5 * Math.exp(-Math.abs(w)) };
FA.rfDraw = function (k, D, seed) {
  const r = ST.rng(seed * 7919 + 17), w = [];
  for (let j = 0; j < D; j++) {
    if (k === "gauss") w.push(ST.randn(r));
    else if (k === "lap") w.push(PV.cauchyDraw(0, 1, r));
    else { const e = PV.expDraw(1, r); w.push(r() < 0.5 ? -e : e); }
  }
  return w;
};
FA.rfStats = function (k, D, seed) {
  const w = FA.rfDraw(k, D, seed), K = FA.rfKernel[k];
  const grid = d3.range(-5, 5.0001, 0.025);
  const est = grid.map(d => d3.mean(w, wj => Math.cos(wj * d)));
  const sd = grid.map(d => Math.sqrt(Math.max(0, 0.5 + 0.5 * K(2 * d) - K(d) * K(d)) / D));
  let maxE = 0, sse = 0, svar = 0, inBand = 0;
  grid.forEach((d, i) => {
    const e = est[i] - K(d);
    maxE = Math.max(maxE, Math.abs(e)); sse += e * e; svar += sd[i] * sd[i];
    if (Math.abs(e) <= 2 * sd[i] + 1e-12) inBand++;
  });
  return { w, grid, est, sd, maxE, rms: Math.sqrt(sse / grid.length), rmsTh: Math.sqrt(svar / grid.length), inBand: inBand / grid.length,
    sd1: Math.sqrt((0.5 + 0.5 * K(2) - K(1) * K(1)) / D) };
};
(function () {
  const svg = d3.select("#rf-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id), out = $("rf-readout");
  let k = "gauss", j = 6, seed = FA.seeds.rf;
  const W = 680, H = 330, top = 24, bot = H - 34;
  function draw() {
    svg.selectAll("*").remove();
    const g = svg.append("g"), D = Math.pow(2, j), R = FA.rfStats(k, D, seed), K = FA.rfKernel[k];
    // left: frequency histogram vs spectral density
    const lx0 = 46, lx1 = 300, wr = 6, nb = 30, bw = 2 * wr / nb;
    const xs = d3.scaleLinear().domain([-wr, wr]).range([lx0, lx1]);
    const counts = new Array(nb).fill(0); let outside = 0;
    R.w.forEach(v => { const b = Math.floor((v + wr) / bw); if (b >= 0 && b < nb) counts[b]++; else outside++; });
    const dens = counts.map(c => c / (D * bw));
    const ymax = Math.max(d3.max(dens), FA.rfSpec[k](0)) * 1.15;
    const ys = d3.scaleLinear().domain([0, ymax]).range([bot, top]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${bot})`).call(d3.axisBottom(xs).ticks(6));
    g.append("g").attr("class", "axis").attr("transform", `translate(${lx0},0)`).call(d3.axisLeft(ys).ticks(4));
    dens.forEach((v, i) => g.append("rect").attr("x", xs(-wr + i * bw) + 0.5).attr("width", Math.max(0.5, xs(-wr + bw) - xs(-wr) - 1)).attr("y", ys(v)).attr("height", bot - ys(v)).attr("fill", SC.a2).attr("fill-opacity", 0.55));
    const ww = d3.range(-wr, wr + 1e-9, 0.02);
    g.append("path").attr("d", d3.line().x(v => xs(v)).y(v => ys(FA.rfSpec[k](v)))(ww)).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.5);
    FA.label(g, lx0, 14, `D = ${D} sampled frequencies w over the spectral density${outside ? ` (${outside} outside ±6)` : ""}`, { size: 10.5 });
    // right: kernel vs estimate
    const rx0 = 350, rx1 = W - 14;
    const ds = d3.scaleLinear().domain([-5, 5]).range([rx0, rx1]);
    const yk = d3.scaleLinear().domain([-0.35, 1.15]).range([bot, top]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${yk(0)})`).call(d3.axisBottom(ds).ticks(10));
    g.append("g").attr("class", "axis").attr("transform", `translate(${rx0},0)`).call(d3.axisLeft(yk).ticks(5));
    g.append("path").attr("d", d3.area().x((d, i) => ds(d)).y0((d, i) => yk(K(d) - 2 * R.sd[i])).y1((d, i) => yk(K(d) + 2 * R.sd[i]))(R.grid)).attr("fill", SC.accent).attr("fill-opacity", 0.18);
    g.append("path").attr("d", d3.line().x(d => ds(d)).y(d => yk(K(d)))(R.grid)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    g.append("path").attr("d", d3.line().x((d, i) => ds(d)).y((d, i) => yk(R.est[i]))(R.grid)).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);
    FA.label(g, rx0, 14, "blue: k(Δ) · band: ±2 sd · orange: (1/D)Σ cos(wⱼΔ)", { size: 10.5 });
    FA.label(g, rx1, bot + 28, "offset Δ = x − y", { anchor: "end", size: 10 });
    out.innerHTML = `seed ${seed} · D = ${D} · largest |k̂ − k| over |Δ| ≤ 5: <b>${ST.fmt(R.maxE, 4)}</b> · RMS error ${ST.fmt(R.rms, 4)} (theory ${ST.fmt(R.rmsTh, 4)})<br>` +
      `theoretical sd at Δ = 1: ${ST.fmt(R.sd1, 4)} · grid points inside the ±2 sd band: ${ST.pct(R.inBand, 1)} · error shrinks like 1/√D`;
  }
  $("rf-k").addEventListener("change", e => { k = e.target.value; draw(); });
  $("rf-d").addEventListener("input", e => { j = +e.target.value; $("rf-dv").textContent = Math.pow(2, j); draw(); });
  $("rf-new").addEventListener("click", () => { seed += 1; draw(); });
  $("rf-reset").addEventListener("click", () => { k = "gauss"; j = 6; seed = FA.seeds.rf; $("rf-k").value = "gauss"; $("rf-d").value = 6; $("rf-dv").textContent = 64; draw(); });
  draw();
})();
