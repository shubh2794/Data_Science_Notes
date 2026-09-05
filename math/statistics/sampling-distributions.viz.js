/* sampling-distributions.viz.js — the nine visualizations on
   math/statistics/sampling-distributions.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #three-svg    population / one sample / sampling distribution, on ONE shared axis —
                      draw a sample and watch its mean drop into the third panel
     2  #lln-svg      the law of large numbers: the average converges while the SUM's
                      deviation grows like √n — two panels, one cursor
     3  #normal-svg   the normal toolkit: draggable z-bounds, the 68–95–99.7 bands, and
                      the same area read off the density and off Φ
     4  #clt-svg      the CLT machine: pick a population and n, watch the standardized
                      mean take shape, with a normal-quantile panel as the honest judge
     5  #rate-svg     how FAST: sup|Fₙ − Φ| against n on log–log, computed EXACTLY,
                      against the Edgeworth prediction and the Berry–Esseen bound
     6  #cauchy-svg   where it fails: a Cauchy population, where x̄ₙ never settles for any n
                      — while the median converges perfectly well
     7  #root-svg     the 1/√n law you can read off, with the finite-population correction
     8  #binom-svg    binomial → normal, with the conditions lighting up as they pass
     9  #ml-svg       a validation score is a statistic: how often the better model loses  */

/* ══════════════ shared population catalogue ══════════════ */
const POPS = (function () {
  const LN_MU = Math.exp(0.5);                            // lognormal(0,1) mean
  const LN_SD = Math.sqrt((Math.E - 1) * Math.E);         // lognormal(0,1) sd
  const LN_G1 = (Math.E + 2) * Math.sqrt(Math.E - 1);     // lognormal(0,1) skewness
  const BI_SD = Math.sqrt(1.6 * 1.6 + 0.45 * 0.45);       // ±1.6 mixture with sd 0.45 arms
  return {
    normal: {
      name: "Normal(0, 1)", mu: 0, sd: 1, g1: 0, dom: [-4, 4], discrete: false,
      pdf: x => ST.normPdf(x), draw: r => ST.randn(r),
      note: "already normal — every n is exact"
    },
    uniform: {
      name: "Uniform(0, 1)", mu: 0.5, sd: 1 / Math.sqrt(12), g1: 0, dom: [-0.15, 1.15], discrete: false,
      pdf: x => (x >= 0 && x <= 1 ? 1 : 0), draw: r => r(),
      note: "symmetric — the √n term vanishes, so it converges like 1/n"
    },
    expo: {
      name: "Exponential(1)", mu: 1, sd: 1, g1: 2, dom: [0, 6], discrete: false,
      pdf: x => (x >= 0 ? Math.exp(-x) : 0), draw: r => -Math.log(1 - r()),
      note: "the textbook skewed case: γ₁ = 2"
    },
    bimodal: {
      name: "Bimodal mixture", mu: 0, sd: BI_SD, g1: 0, dom: [-4, 4], discrete: false,
      pdf: x => 0.5 * (ST.normPdf((x + 1.6) / 0.45) + ST.normPdf((x - 1.6) / 0.45)) / 0.45,
      draw: r => (r() < 0.5 ? -1.6 : 1.6) + 0.45 * ST.randn(r),
      note: "nothing like a bell — and it still works, fast, because it is symmetric"
    },
    lognormal: {
      name: "Lognormal(0, 1)", mu: LN_MU, sd: LN_SD, g1: LN_G1, dom: [0, 9], discrete: false,
      pdf: x => (x > 0 ? Math.exp(-Math.pow(Math.log(x), 2) / 2) / (x * Math.sqrt(2 * Math.PI)) : 0),
      draw: r => Math.exp(ST.randn(r)),
      note: "heavy right tail: γ₁ ≈ 6.18, so n = 30 is nowhere near enough"
    },
    bern10: {
      name: "Bernoulli(0.1)", mu: 0.1, sd: 0.3, g1: (1 - 0.2) / 0.3, dom: [-0.25, 1.25], discrete: true,
      atoms: [{ x: 0, p: 0.9 }, { x: 1, p: 0.1 }], draw: r => (r() < 0.1 ? 1 : 0),
      note: "a rare 0/1 event — the mean is a proportion and n must clear np ≥ 10"
    },
    claims: {
      name: "Claims: 99% × 100, 1% × 10 000", mu: 199, sd: 985.0376, g1: 9.8494,
      dom: [-1500, 4000], discrete: true,
      atoms: [{ x: 100, p: 0.99 }, { x: 10000, p: 0.01 }], draw: r => (r() < 0.01 ? 10000 : 100),
      note: "γ₁ ≈ 9.85 — the case where n = 30 is not merely rough, it is wrong"
    }
  };
})();

/* ─────────────────── 1 · the three distributions ─────────────────── */
(function () {
  const svg = d3.select("#three-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 585;
  /* A concrete finite population: 4000 checkout times in seconds, right-skewed.
     μ and σ below are the EXACT finite-population values, computed from the array,
     so every number in the readout is checkable against the picture.            */
  const rp = ST.rng(90501), POP = [];
  while (POP.length < 4000) {
    const v = Math.exp(2.35 + 0.62 * ST.randn(rp));          // lognormal-ish seconds
    if (v >= 1.2 && v <= 47.5) POP.push(v);                  // rejected, not clipped: no pile-up at the edge
  }
  const MU = ST.mean(POP), SD = Math.sqrt(ST.variance(POP, 0));   // divisor N: the parameter
  const XDOM = [0, 48];

  const F = ST.frame(svg, W, H, { l: 52, r: 16, t: 22, b: 40 });
  const x = d3.scaleLinear().domain(XDOM).range([0, F.iw]);
  const ROW = [0, 178, 356], RH = 118;

  const rows = ROW.map((y, i) => F.g.append("g").attr("transform", `translate(0,${y})`));
  const title = (g, n, t, sub) => {
    g.append("text").attr("x", -46).attr("y", -6).attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", SC.accent).text(n);
    g.append("text").attr("x", -46 + 16).attr("y", -6).attr("font-size", 12)
      .attr("fill", SC.ink).text(t);
    g.append("text").attr("x", -46).attr("y", 9).attr("font-size", 10.5).attr("fill", SC.muted).text(sub);
  };
  title(rows[0], "①", " the population — all 4 000 units", "shape: right-skewed, spread σ");
  title(rows[1], "②", " one sample of n — the data you have", "same shape, same spread, just fewer");
  title(rows[2], "③", " the sampling distribution of x̄", "a different object: spread σ/√n");

  /* row 1 — population histogram (fixed) */
  const bw1 = 1.2, bins1 = ST.histBins(POP, 0, bw1, 40);
  const y1 = d3.scaleLinear().domain([0, d3.max(bins1, b => b.p) * 1.12]).range([RH, 14]);
  rows[0].selectAll("rect.pb").data(bins1).join("rect").attr("class", "pb")
    .attr("x", b => x(b.x0) + 0.5).attr("width", Math.max(1, x(bw1) - x(0) - 1))
    .attr("y", b => y1(b.p)).attr("height", b => RH - y1(b.p))
    .attr("fill", SC.muted).attr("fill-opacity", 0.45);
  rows[0].append("line").attr("x1", x(MU)).attr("x2", x(MU)).attr("y1", 10).attr("y2", RH)
    .attr("stroke", SC.good).attr("stroke-width", 2);
  rows[0].append("text").attr("x", x(MU) + 5).attr("y", 20).attr("font-size", 10.5).attr("fill", SC.good)
    .text(`μ = ${ST.fmt(MU, 2)} s`);
  rows[0].append("line").attr("x1", x(MU - SD)).attr("x2", x(MU + SD)).attr("y1", RH - 6).attr("y2", RH - 6)
    .attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
  rows[0].append("text").attr("x", F.iw).attr("y", RH - 3).attr("text-anchor", "end").attr("font-size", 10.5)
    .attr("fill", SC.a2).text(`the dashed bar is μ ± σ · σ = ${ST.fmt(SD, 2)} s`);

  /* row 2 — the current sample */
  const gDots = rows[1].append("g");
  const lineXbar = rows[1].append("line").attr("y1", 8).attr("y2", RH)
    .attr("stroke", SC.accent).attr("stroke-width", 2.2).attr("opacity", 0);
  const labXbar = rows[1].append("text").attr("y", 18).attr("font-size", 10.5)
    .attr("fill", SC.accent).attr("opacity", 0);
  rows[1].append("line").attr("x1", x(MU)).attr("x2", x(MU)).attr("y1", 8).attr("y2", RH)
    .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3").attr("opacity", 0.7);

  /* row 3 — the accumulating sampling distribution */
  const gMeans = rows[2].append("g");
  const gCurve = rows[2].append("g");
  const ax3 = rows[2].append("g").attr("class", "axis").attr("transform", `translate(0,${RH})`);
  rows[2].append("text").attr("x", F.iw).attr("y", RH + 33).attr("text-anchor", "end")
    .attr("font-size", 11).attr("fill", SC.muted).text("seconds");
  const gFly = F.g.append("g");

  const sN = document.getElementById("th-n"), sNv = document.getElementById("th-nv");
  const cZoom = document.getElementById("th-zoom");
  const out = document.getElementById("three-readout");
  let means = [], last = null, rnd = ST.rng(20260906);

  function scale3() {
    if (!cZoom.checked) return x;
    const se = SD / Math.sqrt(+sN.value);
    return d3.scaleLinear().domain([MU - 4.4 * se, MU + 4.4 * se]).range([0, F.iw]);
  }

  function drawSample() {
    const n = +sN.value;
    const s = ST.sample(POP, n, rnd);            // with replacement: SE is exactly σ/√n
    last = { s: s, m: ST.mean(s), sd: ST.sd(s) };
    means.push(last.m);
    if (means.length > 900) means = means.slice(-900);
    return last;
  }

  function render(fly) {
    const n = +sN.value, se = SD / Math.sqrt(n), x3 = scale3();
    sNv.textContent = n;

    /* ② the dots of one sample */
    if (last) {
      const jit = ST.jitterY(last.s, 31, 1);
      const d = gDots.selectAll("circle").data(last.s);
      d.exit().remove();
      d.enter().append("circle").attr("r", 2.6).merge(d)
        .attr("cx", v => x(v)).attr("cy", (v, i) => 30 + (RH - 44) * (0.5 + 0.45 * jit[i]))
        .attr("fill", SC.accent).attr("fill-opacity", 0.6);
      lineXbar.attr("x1", x(last.m)).attr("x2", x(last.m)).attr("opacity", 1);
      labXbar.attr("x", x(last.m) + 5).attr("opacity", 1).text(`x̄ = ${ST.fmt(last.m, 2)}`);
    }

    /* ③ the stack of means */
    const lo = x3.domain()[0], hi = x3.domain()[1], NB = 62, bw = (hi - lo) / NB;
    const counts = new Map();
    const pts = [];
    means.forEach(m => {
      const b = ST.clamp(Math.floor((m - lo) / bw), 0, NB - 1);
      const c = counts.get(b) || 0; counts.set(b, c + 1);
      pts.push({ m: m, b: b, k: c });
    });
    const maxC = Math.max(8, d3.max([...counts.values()]) || 1);
    const dr = Math.min(3.4, Math.max(1.4, (RH - 16) / (2 * maxC)));
    const mm = gMeans.selectAll("circle").data(pts.slice(-900));
    mm.exit().remove();
    mm.enter().append("circle").merge(mm)
      .attr("r", dr).attr("cx", d => x3(lo + (d.b + 0.5) * bw))
      .attr("cy", d => RH - 3 - dr - 2 * dr * d.k)
      .attr("fill", (d, i, nodes) => i === nodes.length - 1 ? SC.a2 : SC.accent)
      .attr("fill-opacity", (d, i, nodes) => i === nodes.length - 1 ? 1 : 0.55);

    /* the normal curve the CLT predicts, scaled to the dot stack */
    gCurve.selectAll("*").remove();
    const gridPts = ST.linspace(lo, hi, 220).map(v => ({ v: v, d: ST.normPdf((v - MU) / se) / se }));
    const peak = 1 / (se * Math.sqrt(2 * Math.PI));
    const hMax = 2 * dr * maxC;
    const ln = d3.line().x(p => x3(p.v)).y(p => RH - 3 - hMax * (p.d / peak));
    gCurve.append("path").datum(gridPts).attr("d", ln).attr("fill", "none")
      .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3").attr("opacity", 0.85);
    gCurve.append("line").attr("x1", x3(MU)).attr("x2", x3(MU)).attr("y1", 4).attr("y2", RH)
      .attr("stroke", SC.good).attr("stroke-width", 1.6);
    [-2, -1, 1, 2].forEach(k => {
      if (x3(MU + k * se) < 0 || x3(MU + k * se) > F.iw) return;
      gCurve.append("line").attr("x1", x3(MU + k * se)).attr("x2", x3(MU + k * se))
        .attr("y1", RH - 12).attr("y2", RH).attr("stroke", SC.a2).attr("stroke-width", 1).attr("opacity", 0.7);
    });
    gCurve.append("text").attr("x", F.iw).attr("y", 12).attr("text-anchor", "end").attr("font-size", 10.5)
      .attr("fill", SC.good).text(`centre μ = ${ST.fmt(MU, 2)} s · SE = σ/√n = ${ST.fmt(se, 2)} s`);
    ax3.call(d3.axisBottom(x3).ticks(8));

    /* the flying dot: this sample's mean falling from ② into ③ */
    gFly.selectAll("*").remove();
    if (fly && last) {
      gFly.append("circle").attr("r", 4.5).attr("fill", SC.a2)
        .attr("cx", x(last.m)).attr("cy", ROW[1] + RH / 2)
        .transition().duration(520).ease(d3.easeCubicIn)
        .attr("cx", x3(last.m)).attr("cy", ROW[2] + RH - 8)
        .transition().duration(160).attr("r", 0).remove();
    }

    const emp = means.length > 1 ? ST.sd(means) : NaN;
    out.innerHTML =
      `<b>population</b> (fixed, not random): N = 4 000 · μ = <b>${ST.fmt(MU, 3)}</b> s · σ = <b>${ST.fmt(SD, 3)}</b> s` +
      (last ? `<br><b>this sample</b> (random): n = ${n} · x̄ = <b>${ST.fmt(last.m, 3)}</b> · s = <b>${ST.fmt(last.sd, 3)}</b> ` +
        `· x̄ − μ = <span style="color:${SC.a2}">${ST.fmt(last.m - MU, 3)}</span> s` : "") +
      `<br><b>across ${means.length} sample${means.length === 1 ? "" : "s"}</b>: mean of the x̄'s = ` +
      `<b>${means.length ? ST.fmt(ST.mean(means), 3) : "—"}</b> (theory: μ = ${ST.fmt(MU, 3)}) · ` +
      `SD of the x̄'s = <b>${isFinite(emp) ? ST.fmt(emp, 3) : "—"}</b> (theory: σ/√n = ${ST.fmt(se, 3)})` +
      `<br><span style="color:${SC.muted}">panel ③ is ${ST.fmt(Math.sqrt(n), 1)}× narrower than panels ① and ②. ` +
      `That factor is the whole of statistical inference.</span>`;
  }

  document.getElementById("th-1").addEventListener("click", () => { drawSample(); render(true); });
  document.getElementById("th-50").addEventListener("click", () => { for (let i = 0; i < 50; i++) drawSample(); render(false); });
  document.getElementById("th-clear").addEventListener("click", () => { means = []; last = null; gDots.selectAll("*").remove(); lineXbar.attr("opacity", 0); labXbar.attr("opacity", 0); render(false); });
  sN.addEventListener("input", () => { means = []; last = null; gDots.selectAll("*").remove(); lineXbar.attr("opacity", 0); labXbar.attr("opacity", 0); render(false); });
  cZoom.addEventListener("change", () => render(false));
  for (let i = 0; i < 40; i++) drawSample();
  render(false);
})();

/* ─────────────────── 2 · the law of large numbers ─────────────────── */
(function () {
  const svg = d3.select("#lln-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470, KMAX = 100000;
  const F = ST.frame(svg, W, H, { l: 58, r: 18, t: 16, b: 42 });
  const TOP = 0, BOT = 226, PH = 158;
  const x = d3.scaleLog().domain([1, KMAX]).range([0, F.iw]);

  const gTop = F.g.append("g").attr("transform", `translate(0,${TOP})`);
  const gBot = F.g.append("g").attr("transform", `translate(0,${BOT})`);

  const SRC = {
    coin: { name: "fair coin (0/1)", mu: 0.5, sd: 0.5, draw: r => (r() < 0.5 ? 1 : 0) },
    biased: { name: "biased coin, p = 0.3", mu: 0.3, sd: Math.sqrt(0.21), draw: r => (r() < 0.3 ? 1 : 0) },
    expo: { name: "Exponential(1)", mu: 1, sd: 1, draw: r => -Math.log(1 - r()) },
    lognormal: { name: "Lognormal(0, 1)", mu: Math.exp(0.5), sd: Math.sqrt((Math.E - 1) * Math.E), draw: r => Math.exp(ST.randn(r)) }
  };

  const sel = document.getElementById("ll-pop");
  const sK = document.getElementById("ll-k"), sKv = document.getElementById("ll-kv");
  const out = document.getElementById("lln-readout");
  const NP = 5;                                          // independent paths
  let paths = [], seed = 4711;

  const GRID = (function () {                            // ~420 log-spaced checkpoints
    const g = [];
    for (let i = 0; i < 420; i++) {
      const v = Math.round(Math.pow(KMAX, i / 419));
      if (!g.length || v > g[g.length - 1]) g.push(v);
    }
    return g;
  })();

  function run() {
    const P = SRC[sel.value];
    paths = [];
    for (let p = 0; p < NP; p++) {
      const r = ST.rng(seed + 977 * p);
      let s = 0, gi = 0;
      const rec = [];
      for (let k = 1; k <= KMAX; k++) {
        s += P.draw(r);
        if (k === GRID[gi]) { rec.push({ k: k, mean: s / k, dev: s - k * P.mu }); gi++; }
      }
      paths.push(rec);
    }
  }

  function render() {
    const P = SRC[sel.value];
    const K = ST.clamp(Math.round(Math.pow(10, +sK.value)), 1, KMAX);
    sKv.textContent = K.toLocaleString("en-US");

    gTop.selectAll("*").remove(); gBot.selectAll("*").remove();

    /* ── top: the AVERAGE, converging ── */
    const band = 3.2 * P.sd;                             // in units of σ/√k at k = 1 → generous
    const yT = d3.scaleLinear().domain([P.mu - band, P.mu + band]).range([PH, 0]);
    ST.gridY(gTop, yT, F.iw, 5);
    const envT = d3.area().x(d => x(d.k)).y0(d => yT(P.mu - 2 * P.sd / Math.sqrt(d.k)))
      .y1(d => yT(P.mu + 2 * P.sd / Math.sqrt(d.k)));
    gTop.append("path").datum(GRID.map(k => ({ k: k }))).attr("d", envT)
      .attr("fill", SC.good).attr("fill-opacity", 0.11);
    gTop.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", yT(P.mu)).attr("y2", yT(P.mu))
      .attr("stroke", SC.good).attr("stroke-width", 1.6);
    const lnT = d3.line().x(d => x(d.k)).y(d => yT(ST.clamp(d.mean, yT.domain()[0], yT.domain()[1])));
    paths.forEach((p, i) => gTop.append("path").datum(p).attr("d", lnT).attr("fill", "none")
      .attr("stroke", i === 0 ? SC.accent : SC.muted).attr("stroke-width", i === 0 ? 1.9 : 1)
      .attr("opacity", i === 0 ? 1 : 0.42));
    gTop.append("g").attr("class", "axis").call(d3.axisLeft(yT).ticks(5));
    gTop.append("text").attr("x", 2).attr("y", -3).attr("font-size", 11).attr("fill", SC.accent)
      .text("the AVERAGE x̄ₖ — the gap to μ shrinks like σ/√k");
    gTop.append("text").attr("x", F.iw).attr("y", -3).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.good).text("μ, and the ±2σ/√k envelope");

    /* ── bottom: the SUM's deviation, growing ── */
    const dmax = 4.2 * P.sd * Math.sqrt(KMAX);
    const yB = d3.scaleSymlog().domain([-dmax, dmax]).constant(10).range([PH, 0]);
    gBot.append("g").attr("class", "gridlines").selectAll("line")
      .data([-1000, -100, -10, 0, 10, 100, 1000].filter(v => Math.abs(v) <= dmax))
      .join("line").attr("x1", 0).attr("x2", F.iw).attr("y1", v => yB(v)).attr("y2", v => yB(v))
      .attr("stroke", SC.grid);
    const envB = d3.area().x(d => x(d.k)).y0(d => yB(-2 * P.sd * Math.sqrt(d.k)))
      .y1(d => yB(2 * P.sd * Math.sqrt(d.k)));
    gBot.append("path").datum(GRID.map(k => ({ k: k }))).attr("d", envB)
      .attr("fill", SC.bad).attr("fill-opacity", 0.11);
    gBot.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", yB(0)).attr("y2", yB(0))
      .attr("stroke", SC.good).attr("stroke-width", 1.4);
    const lnB = d3.line().x(d => x(d.k)).y(d => yB(ST.clamp(d.dev, -dmax, dmax)));
    paths.forEach((p, i) => gBot.append("path").datum(p).attr("d", lnB).attr("fill", "none")
      .attr("stroke", i === 0 ? SC.a2 : SC.muted).attr("stroke-width", i === 0 ? 1.9 : 1)
      .attr("opacity", i === 0 ? 1 : 0.42));
    gBot.append("g").attr("class", "axis")
      .call(d3.axisLeft(yB).tickValues([-1000, -100, -10, 0, 10, 100, 1000].filter(v => Math.abs(v) <= dmax))
        .tickFormat(d3.format("d")));
    gBot.append("g").attr("class", "axis").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(x).ticks(6, "~s"));
    gBot.append("text").attr("x", F.iw).attr("y", PH + 34).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("number of draws k (log scale)");
    gBot.append("text").attr("x", 2).attr("y", -3).attr("font-size", 11).attr("fill", SC.a2)
      .text("the SUM's deviation Sₖ − kμ — it GROWS, like σ√k");

    /* cursor */
    [gTop, gBot].forEach(g => g.append("line").attr("x1", x(K)).attr("x2", x(K))
      .attr("y1", 0).attr("y2", PH).attr("stroke", SC.violet).attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "4 3"));

    const at = paths.map(p => p.reduce((a, d) => (d.k <= K ? d : a), p[0]));
    const seK = P.sd / Math.sqrt(K), swK = P.sd * Math.sqrt(K);
    out.innerHTML =
      `<b>${P.name}</b> · μ = ${ST.fmt(P.mu, 3)}, σ = ${ST.fmt(P.sd, 3)} · at k = <b>${K.toLocaleString("en-US")}</b> draws` +
      `<br>highlighted path: x̄ₖ = <b>${ST.fmt(at[0].mean, 4)}</b>, so x̄ₖ − μ = ` +
      `<span style="color:${SC.accent}">${ST.fmt(at[0].mean - P.mu, 4)}</span> ` +
      `· Sₖ − kμ = <span style="color:${SC.a2}">${ST.fmt(at[0].dev, 1)}</span>` +
      `<br>theory: SE(x̄ₖ) = σ/√k = <b>${ST.fmt(seK, 4)}</b> shrinking · SD(Sₖ) = σ√k = <b>${ST.fmt(swK, 1)}</b> growing` +
      `<br>typical size of the gap: E|x̄ₖ − μ| ≈ σ√(2/(πk)) = <b>${ST.fmt(P.sd * Math.sqrt(2 / (Math.PI * K)), 4)}</b> · ` +
      `E|Sₖ − kμ| ≈ σ√(2k/π) = <b>${ST.fmt(P.sd * Math.sqrt(2 * K / Math.PI), 1)}</b>` +
      `<br><span style="color:${SC.muted}">The two panels are the same five runs. Nothing is pulling the sum back to zero — ` +
      `the average converges because k grows faster than the wandering does.</span>`;
  }

  sel.addEventListener("change", () => { run(); render(); });
  sK.addEventListener("input", render);
  document.getElementById("ll-again").addEventListener("click", () => { seed += 131; run(); render(); });
  run(); render();
})();

/* ─────────────────── 3 · the normal toolkit ─────────────────── */
(function () {
  const svg = d3.select("#normal-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const F = ST.frame(svg, W, H, { l: 52, r: 18, t: 40, b: 42 });
  const PH1 = 182, GAP = 58, PH2 = 126;
  const gD = F.g.append("g"), gC = F.g.append("g").attr("transform", `translate(0,${PH1 + GAP})`);
  const x = d3.scaleLinear().domain([-4, 4]).range([0, F.iw]);
  const yD = d3.scaleLinear().domain([0, 0.42]).range([PH1, 0]);
  const yC = d3.scaleLinear().domain([0, 1]).range([PH2, 0]);

  const sMu = document.getElementById("nz-mu"), sMuv = document.getElementById("nz-muv");
  const sSd = document.getElementById("nz-sd"), sSdv = document.getElementById("nz-sdv");
  const out = document.getElementById("normal-readout");
  let za = -1, zb = 1;

  ST.gridY(gD, yD, F.iw, 4);
  const gBand = gD.append("g");
  const curve = ST.linspace(-4, 4, 401).map(z => ({ z: z, d: ST.normPdf(z) }));
  gD.append("path").datum(curve).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2)
    .attr("d", d3.line().x(p => x(p.z)).y(p => yD(p.d)));
  gD.append("g").attr("class", "axis").attr("transform", `translate(0,${PH1})`).call(d3.axisBottom(x).ticks(9));
  gD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(4));
  gD.append("text").attr("x", 2).attr("y", -24).attr("font-size", 11).attr("fill", SC.accent)
    .text("φ(z) — the standard normal density; area = probability");

  /* the 68–95–99.7 rule, drawn once as brackets under the curve */
  const RULE = [{ k: 1, p: 0.682689, y: 12 }, { k: 2, p: 0.954500, y: 28 }, { k: 3, p: 0.997300, y: 44 }];
  const gRule = gD.append("g");
  RULE.forEach(r => {
    gRule.append("line").attr("x1", x(-r.k)).attr("x2", x(r.k)).attr("y1", r.y).attr("y2", r.y)
      .attr("stroke", SC.a2).attr("stroke-width", 1.3).attr("opacity", 0.75);
    [-r.k, r.k].forEach(s => gRule.append("line").attr("x1", x(s)).attr("x2", x(s))
      .attr("y1", r.y - 4).attr("y2", r.y + 4).attr("stroke", SC.a2).attr("stroke-width", 1.3).attr("opacity", 0.75));
    gRule.append("text").attr("x", x(r.k) + 6).attr("y", r.y + 3.5).attr("font-size", 10)
      .attr("fill", SC.a2).text(`±${r.k}σ → ${ST.pct(r.p, 2)}`);
  });

  ST.gridY(gC, yC, F.iw, 4);
  gC.append("path").datum(ST.linspace(-4, 4, 401).map(z => ({ z: z, F: ST.normCdf(z) })))
    .attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 2)
    .attr("d", d3.line().x(p => x(p.z)).y(p => yC(p.F)));
  gC.append("g").attr("class", "axis").attr("transform", `translate(0,${PH2})`).call(d3.axisBottom(x).ticks(9));
  gC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(4).tickFormat(d3.format(".0%")));
  gC.append("text").attr("x", 2).attr("y", -5).attr("font-size", 11).attr("fill", SC.violet)
    .text("Φ(z) — the same area, read as a height: Φ(b) − Φ(a)");
  gC.append("text").attr("x", F.iw).attr("y", PH2 + 34).attr("text-anchor", "end").attr("font-size", 11)
    .attr("fill", SC.muted).text("z = (x − μ)/σ — standard deviations from the centre");
  const gCm = gC.append("g");

  const gHand = F.g.append("g");
  function handle(which) {
    const h = gHand.append("g").attr("class", "dragpt").style("cursor", "ew-resize");
    h.append("line").attr("y1", 0).attr("y2", PH1).attr("stroke", SC.a2).attr("stroke-width", 2);
    h.append("rect").attr("x", -9).attr("y", PH1 - 6).attr("width", 18).attr("height", 14).attr("rx", 3)
      .attr("fill", SC.a2);
    h.append("text").attr("y", -6).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", SC.a2);
    h.call(d3.drag().on("drag", ev => {
      const z = ST.clamp(x.invert(d3.pointer(ev, F.g.node())[0]), -4, 4);
      if (which === "a") za = Math.min(z, zb - 0.05); else zb = Math.max(z, za + 0.05);
      render();
    }));
    return h;
  }
  const hA = handle("a"), hB = handle("b");

  function render() {
    const mu = +sMu.value, sd = +sSd.value;
    sMuv.textContent = ST.fmt(mu, 0); sSdv.textContent = ST.fmt(sd, 0);
    const area = ST.normCdf(zb) - ST.normCdf(za);

    const seg = curve.filter(p => p.z >= za && p.z <= zb);
    const poly = [{ z: za, d: 0 }, { z: za, d: ST.normPdf(za) }].concat(seg)
      .concat([{ z: zb, d: ST.normPdf(zb) }, { z: zb, d: 0 }]);
    gBand.selectAll("path").data([poly]).join("path")
      .attr("d", d3.area().x(p => x(p.z)).y0(yD(0)).y1(p => yD(p.d)))
      .attr("fill", SC.a2).attr("fill-opacity", 0.22);
    hA.attr("transform", `translate(${x(za)},0)`); hB.attr("transform", `translate(${x(zb)},0)`);
    hA.select("text").text(`a = ${ST.fmt(za, 2)}`); hB.select("text").text(`b = ${ST.fmt(zb, 2)}`);

    gCm.selectAll("*").remove();
    [[za, ST.normCdf(za)], [zb, ST.normCdf(zb)]].forEach(([z, Fz]) => {
      gCm.append("line").attr("x1", x(z)).attr("x2", x(z)).attr("y1", yC(Fz)).attr("y2", PH2)
        .attr("stroke", SC.a2).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
      gCm.append("line").attr("x1", 0).attr("x2", x(z)).attr("y1", yC(Fz)).attr("y2", yC(Fz))
        .attr("stroke", SC.a2).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    });
    gCm.append("rect").attr("x", -6).attr("width", 8).attr("y", yC(ST.normCdf(zb)))
      .attr("height", Math.max(1, yC(ST.normCdf(za)) - yC(ST.normCdf(zb))))
      .attr("fill", SC.a2).attr("fill-opacity", 0.6);

    const xa = mu + za * sd, xb = mu + zb * sd;
    out.innerHTML =
      `area between z = <b>${ST.fmt(za, 2)}</b> and z = <b>${ST.fmt(zb, 2)}</b> is ` +
      `Φ(${ST.fmt(zb, 2)}) − Φ(${ST.fmt(za, 2)}) = ${ST.fmt(ST.normCdf(zb), 4)} − ${ST.fmt(ST.normCdf(za), 4)} = <b>${ST.pct(area, 2)}</b>` +
      `<br>in raw units with μ = ${ST.fmt(mu, 0)} and σ = ${ST.fmt(sd, 0)}: that is the chance of landing between ` +
      `<b>${ST.fmt(xa, 1)}</b> and <b>${ST.fmt(xb, 1)}</b> — move the sliders and the picture does not change, only the labels` +
      `<br>tails: below a = ${ST.pct(ST.normCdf(za), 2)} · above b = ${ST.pct(1 - ST.normCdf(zb), 2)} · ` +
      `quantiles: the ${ST.pct(area, 1)} central interval would be z = ±${ST.fmt(ST.normQuant(0.5 + area / 2), 3)}` +
      `<br><span style="color:${SC.muted}">Standardising is the only reason one table, one curve and one function serve every ` +
      `normal problem there is.</span>`;
  }
  sMu.addEventListener("input", render); sSd.addEventListener("input", render);
  document.querySelectorAll("[data-nzpreset]").forEach(b => b.addEventListener("click", () => {
    const v = b.getAttribute("data-nzpreset").split(",");
    za = +v[0]; zb = +v[1]; render();
  }));
  render();
})();

/* ─────────────────── 4 · the CLT machine ─────────────────── */
(function () {
  const svg = d3.select("#clt-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 312, REPS = 3000;
  const F = ST.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
  const CE = ST.cells(F.iw, F.ih, 3, 1, { l: 32, r: 8, t: 24, b: 34 }, { x: 14, y: 0 });
  const panels = CE.map(c => ({ c: c, g: F.g.append("g").attr("transform", `translate(${c.x},${c.y})`) }));

  const sel = document.getElementById("cl-pop");
  const sN = document.getElementById("cl-n"), sNv = document.getElementById("cl-nv");
  const out = document.getElementById("clt-readout");
  const NGRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200];
  let seed = 8123;

  function draw() {
    const P = POPS[sel.value], n = NGRID[+sN.value];
    sNv.textContent = n;
    const r = ST.rng(seed);
    const zs = new Float64Array(REPS), se = P.sd / Math.sqrt(n);
    for (let i = 0; i < REPS; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += P.draw(r);
      zs[i] = (s / n - P.mu) / se;
    }
    const arr = Array.from(zs), sorted = ST.asc(arr);

    panels.forEach(p => p.g.selectAll("*").remove());

    /* ── panel 1: the population ── */
    {
      const c = CE[0], g = panels[0].g;
      const gi = g.append("g").attr("transform", `translate(${c.m.l},${c.m.t})`);
      const xs = d3.scaleLinear().domain(P.dom).range([0, c.iw]);
      g.append("text").attr("x", c.m.l).attr("y", 14).attr("font-size", 11).attr("fill", SC.accent)
        .text("① the population");
      if (P.discrete) {
        const ymax = d3.max(P.atoms, a => a.p) * 1.15;
        const ys = d3.scaleLinear().domain([0, ymax]).range([c.ih, 0]);
        gi.selectAll("rect").data(P.atoms).join("rect")
          .attr("x", a => xs(a.x) - 9).attr("width", 18)
          .attr("y", a => ys(a.p)).attr("height", a => c.ih - ys(a.p))
          .attr("fill", SC.muted).attr("fill-opacity", 0.6);
        gi.selectAll("text.at").data(P.atoms).join("text").attr("class", "at")
          .attr("x", a => xs(a.x)).attr("y", a => ys(a.p) - 4).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", SC.muted).text(a => ST.pct(a.p, 0));
      } else {
        const pts = ST.linspace(P.dom[0], P.dom[1], 240).map(v => ({ v: v, d: P.pdf(v) }));
        const ys = d3.scaleLinear().domain([0, d3.max(pts, p => p.d) * 1.12]).range([c.ih, 0]);
        gi.append("path").datum(pts).attr("fill", SC.muted).attr("fill-opacity", 0.28)
          .attr("d", d3.area().x(p => xs(p.v)).y0(c.ih).y1(p => ys(p.d)));
        gi.append("path").datum(pts).attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 1.5)
          .attr("d", d3.line().x(p => xs(p.v)).y(p => ys(p.d)));
      }
      gi.append("line").attr("x1", xs(P.mu)).attr("x2", xs(P.mu)).attr("y1", 0).attr("y2", c.ih)
        .attr("stroke", SC.good).attr("stroke-width", 1.6);
      gi.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`)
        .call(d3.axisBottom(xs).ticks(4, "~s"));
      gi.append("text").attr("x", c.iw).attr("y", c.ih + 30).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", SC.muted).text(`γ₁ = ${ST.fmt(P.g1, 2)}`);
    }

    /* ── panel 2: the standardized sample mean ── */
    {
      const c = CE[1], g = panels[1].g;
      const gi = g.append("g").attr("transform", `translate(${c.m.l},${c.m.t})`);
      g.append("text").attr("x", c.m.l).attr("y", 14).attr("font-size", 11).attr("fill", SC.a2)
        .text("② z = (x̄ − μ)/(σ/√n)");
      const xs = d3.scaleLinear().domain([-4, 4]).range([0, c.iw]);
      const bw = 8 / 40, bins = ST.histBins(arr.filter(v => v >= -4 && v <= 4), -4, bw, 40);
      const dens = bins.map(b => ({ x0: b.x0, x1: b.x1, d: b.n / (REPS * bw) }));
      const ys = d3.scaleLinear().domain([0, Math.max(0.45, d3.max(dens, b => b.d) * 1.08)]).range([c.ih, 0]);
      gi.selectAll("rect").data(dens).join("rect")
        .attr("x", b => xs(b.x0)).attr("width", Math.max(1, xs(bw) - xs(0) - 0.8))
        .attr("y", b => ys(b.d)).attr("height", b => c.ih - ys(b.d))
        .attr("fill", SC.accent).attr("fill-opacity", 0.55);
      gi.append("path").datum(ST.linspace(-4, 4, 220).map(z => ({ z: z, d: ST.normPdf(z) })))
        .attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2)
        .attr("d", d3.line().x(p => xs(p.z)).y(p => ys(p.d)));
      gi.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`).call(d3.axisBottom(xs).ticks(5));
      const off = REPS - arr.filter(v => v >= -4 && v <= 4).length;
      gi.append("text").attr("x", c.iw).attr("y", c.ih + 30).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", off ? SC.bad : SC.muted)
        .text(off ? `${off} of ${REPS} off-scale` : "N(0,1) in green");
    }

    /* ── panel 3: the normal quantile plot ── */
    {
      const c = CE[2], g = panels[2].g;
      const gi = g.append("g").attr("transform", `translate(${c.m.l},${c.m.t})`);
      g.append("text").attr("x", c.m.l).attr("y", 14).attr("font-size", 11).attr("fill", SC.violet)
        .text("③ normal quantile plot");
      const M = 220, pts = [];
      for (let i = 0; i < M; i++) {
        const k = Math.floor((i + 0.5) * REPS / M);
        pts.push({ q: ST.normQuant((k + 1 - 0.375) / (REPS + 0.25)), v: sorted[k] });
      }
      const lim = 3.6, cl = v => ST.clamp(v, -lim, lim);
      const xs = d3.scaleLinear().domain([-lim, lim]).range([0, c.iw]);
      const ys = d3.scaleLinear().domain([-lim, lim]).range([c.ih, 0]);
      gi.append("line").attr("x1", xs(-lim)).attr("x2", xs(lim)).attr("y1", ys(-lim)).attr("y2", ys(lim))
        .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
      gi.selectAll("circle").data(pts).join("circle").attr("r", 1.9)
        .attr("cx", p => xs(cl(p.q))).attr("cy", p => ys(cl(p.v)))
        .attr("fill", p => (Math.abs(p.v) > lim ? SC.bad : SC.violet))
        .attr("fill-opacity", 0.75);
      gi.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`).call(d3.axisBottom(xs).ticks(5));
      gi.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
      gi.append("text").attr("x", c.iw).attr("y", c.ih + 30).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", SC.muted).text("straight ⇒ normal");
    }

    /* readout — measured against the theory that should hold */
    let K = 0;
    for (let i = 0; i < REPS; i++) {
      const Fz = ST.normCdf(sorted[i]);
      K = Math.max(K, Math.abs((i + 1) / REPS - Fz), Math.abs(i / REPS - Fz));
    }
    const skew = (function () {
      const m = ST.mean(arr), s = ST.sd(arr, 0);
      return ST.mean(arr.map(v => Math.pow((v - m) / s, 3)));
    })();
    const edge = Math.abs(P.g1) / (6 * Math.sqrt(2 * Math.PI) * Math.sqrt(n));
    /* Even a PERFECT fit leaves a gap of about 0.8687/√REPS, because the sup of |F̂ − Φ| over
       3 000 draws is a Kolmogorov–Smirnov statistic, and its expectation is not zero. Judge the
       excess over that floor, not the raw number.                                              */
    const floor = 0.8687 / Math.sqrt(REPS);
    const excess = Math.max(0, K - floor);
    const verdict = excess < 0.01 ? [SC.good, "normal for any practical purpose"]
      : excess < 0.04 ? [SC.a2, "usable, but the tails are still off"]
        : [SC.bad, "NOT normal yet — a z-based interval here would misstate its own coverage"];
    out.innerHTML =
      `<b>${P.name}</b> · n = <b>${n}</b> · ${REPS} simulated samples` +
      `<br>skewness of x̄: measured <b>${ST.fmt(skew, 3)}</b> · theory γ₁/√n = ${ST.fmt(P.g1 / Math.sqrt(n), 3)} ` +
      `(the population's γ₁ = ${ST.fmt(P.g1, 2)}, divided by √n)` +
      `<br>largest gap between the simulated CDF and Φ: <b>${ST.fmt(K, 4)}</b> ` +
      `— of which about ${ST.fmt(floor, 3)} is the simulation's own floor, unavoidable at ${REPS} draws` +
      `<br>Edgeworth prediction |γ₁|/(6√(2π)·√n) = <b>${ST.fmt(edge, 4)}</b> · excess over the floor = <b>${ST.fmt(excess, 4)}</b>` +
      `<br><span style="color:${verdict[0]}">${verdict[1]}</span> · ` +
      `<span style="color:${SC.muted}">${P.note}</span>`;
  }

  sel.addEventListener("change", draw);
  sN.addEventListener("input", draw);
  document.getElementById("cl-again").addEventListener("click", () => { seed += 17; draw(); });
  draw();
})();

/* ─────────────────── 5 · the rate of convergence, computed exactly ─────────────────── */
(function () {
  const svg = d3.select("#rate-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 420;
  const F = ST.frame(svg, W, H, { l: 60, r: 118, t: 18, b: 44 });
  const CE_ = 1 / (6 * Math.sqrt(2 * Math.PI));           // φ(0)/6 = 0.0664904 — the Edgeworth constant
  const NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

  /* Every curve below is EXACT, not simulated:
       Exponential(1) — the sum is Gamma(n,1), so Fₙ comes from the incomplete gamma
       Bernoulli(p)   — the sum is Binomial(n,p), so Fₙ is an exact finite sum
       two-point      — an affine map of a Binomial(n, 0.01), so likewise exact           */
  const CASES = {
    expo: {
      name: "Exponential(1)", g1: 2, rho: 2.4145530,          // ρ = E|X−μ|³ = 6/e² + 2
      lattice: false,
      K: n => {
        let m = 0;
        for (let i = 0; i <= 1600; i++) {
          const z = -6 + 14 * i / 1600;
          m = Math.max(m, Math.abs(ST.gammaP(n, n + z * Math.sqrt(n)) - ST.normCdf(z)));
        }
        return m;
      }
    },
    bern50: { name: "Bernoulli(0.5) — symmetric", p: 0.5, g1: 0, rho: 0.125, lattice: true },
    bern10: { name: "Bernoulli(0.1)", p: 0.1, g1: (1 - 0.2) / 0.3, rho: 0.9 * 0.1 * (0.9 * 0.9 + 0.1 * 0.1), lattice: true },
    claims: { name: "Claims: 99% × 100, 1% × 10 000", p: 0.01, g1: (1 - 0.02) / Math.sqrt(0.0099), lattice: true }
  };
  CASES.claims.rho = 0.99 * 0.01 * (0.99 * 0.99 + 0.01 * 0.01);   // ρ/σ³ is scale-free: use the Bernoulli
  /* Exact worst-case error of the normal approximation to a Binomial(n, p) CDF.
     WITHOUT the correction it is the true Kolmogorov distance sup₍z₎|Fₙ(z) − Φ(z)|, so both
     sides of every jump must be checked. WITH the correction the approximation is only ever
     evaluated AT the lattice points, so only F(k) vs Φ((k + ½ − np)/σ) is meaningful.       */
  function latticeK(p, n, cc) {
    const m = n * p, s = Math.sqrt(n * p * (1 - p));
    let cdf = 0, worst = 0;
    const lo = Math.max(0, Math.floor(m - 9 * s)), hi = Math.min(n, Math.ceil(m + 9 * s));
    for (let k = 0; k < lo; k++) cdf += ST.binomPmf(k, n, p);
    for (let k = lo; k <= hi; k++) {
      const below = cdf;
      cdf += ST.binomPmf(k, n, p);
      if (cc) {
        worst = Math.max(worst, Math.abs(cdf - ST.normCdf((k + 0.5 - m) / s)));
      } else {
        const Fz = ST.normCdf((k - m) / s);
        worst = Math.max(worst, Math.abs(cdf - Fz), Math.abs(below - Fz));
      }
    }
    return worst;
  }

  const sel = document.getElementById("rt-pop");
  const cCC = document.getElementById("rt-cc");
  const out = document.getElementById("rate-readout");
  const x = d3.scaleLog().domain([1, 4096]).range([0, F.iw]);
  const y = d3.scaleLog().domain([1e-4, 1]).range([F.ih, 0]);

  function render() {
    const C = CASES[sel.value], cc = cCC.checked;
    F.g.selectAll("*").remove();
    F.g.append("g").attr("class", "gridlines").selectAll("line").data([1e-4, 1e-3, 1e-2, 1e-1, 1])
      .join("line").attr("x1", 0).attr("x2", F.iw).attr("y1", v => y(v)).attr("y2", v => y(v)).attr("stroke", SC.grid);

    const pts = NS.map(n => ({ n: n, K: C.lattice ? latticeK(C.p, n, cc) : C.K(n) }))
      .filter(d => d.K > 1e-5);
    const ln = d3.line().x(d => x(d.n)).y(d => y(ST.clamp(d.K, 1e-4, 1)));

    /* reference slopes */
    const ref = (f, col, dash, lbl, ly) => {
      const p = NS.map(n => ({ n: n, K: f(n) })).filter(d => d.K <= 1 && d.K >= 1e-4);
      if (!p.length) return;
      F.g.append("path").datum(p).attr("d", ln).attr("fill", "none").attr("stroke", col)
        .attr("stroke-width", 1.5).attr("stroke-dasharray", dash).attr("opacity", 0.8);
      F.g.append("text").attr("x", F.iw + 6).attr("y", ly).attr("font-size", 10).attr("fill", col).text(lbl);
    };
    ref(n => CE_ * Math.abs(C.g1) / Math.sqrt(n), SC.a2, "5 3", "γ₁/(6√(2π)·√n)", 40);
    ref(n => 0.4690 * C.rho / (Math.pow(C.lattice ? Math.sqrt(C.p * (1 - C.p)) : 1, 3) * Math.sqrt(n)),
      SC.bad, "2 3", "Berry–Esseen", 60);
    ref(n => 0.027 / n, SC.violet, "1 4", "a 1/n slope", 80);

    F.g.append("path").datum(pts).attr("d", ln).attr("fill", "none")
      .attr("stroke", SC.accent).attr("stroke-width", 2.4);
    F.g.selectAll("circle.pt").data(pts).join("circle").attr("class", "pt").attr("r", 3.2)
      .attr("cx", d => x(d.n)).attr("cy", d => y(ST.clamp(d.K, 1e-4, 1))).attr("fill", SC.accent);

    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(0.01)).attr("y2", y(0.01))
      .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "6 4");
    F.g.append("text").attr("x", 4).attr("y", y(0.01) - 5).attr("font-size", 10).attr("fill", SC.good)
      .text("1% — “as good as normal”");

    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).tickValues(NS).tickFormat(d3.format("d")))
      .selectAll("text").attr("transform", "rotate(-40)").attr("text-anchor", "end").attr("dx", -3).attr("dy", 4);
    F.g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).tickValues([1e-4, 1e-3, 1e-2, 1e-1, 1]).tickFormat(d3.format(".0e")));
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 40).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("sample size n (log)");
    F.g.append("text").attr("x", 2).attr("y", -5).attr("font-size", 11).attr("fill", SC.accent)
      .text("sup₍z₎ |Fₙ(z) − Φ(z)| — the worst error the normal approximation makes anywhere");

    /* fitted slope over the top decade, and the n that reaches 1% */
    const tail = pts.slice(-5);
    let slope = NaN;
    if (tail.length >= 2) {
      const lx = tail.map(d => Math.log(d.n)), ly2 = tail.map(d => Math.log(d.K));
      slope = ST.lsLine(lx, ly2).b;
    }
    let need = null;
    for (const d of pts) if (d.K <= 0.01) { need = d.n; break; }
    ST.legend(F.g, [{ label: "exact, computed", color: SC.accent }], F.iw + 6, 18, { gap: 14, font: 10 });

    out.innerHTML =
      `<b>${C.name}</b>${C.lattice ? ` · continuity correction <b>${cc ? "ON" : "OFF"}</b>` : ""} · γ₁ = ${ST.fmt(C.g1, 3)}` +
      `<br>fitted log–log slope over the last five points: <b>${ST.fmt(slope, 3)}</b> ` +
      `(the theory says <b>−0.5</b> when γ₁ ≠ 0, and <b>−1</b> when the population is symmetric)` +
      `<br>first n on the doubling grid with error ≤ 1%: <b>${need === null ? "beyond 8 192" : need.toLocaleString("en-US")}</b> · ` +
      `the skewness rule n ≈ 44·γ₁² predicts <b>${C.g1 === 0 ? "≈ 0 (the rule does not bind)" : Math.round(44.21 * C.g1 * C.g1).toLocaleString("en-US")}</b>` +
      (C.lattice && !cc
        ? `<br><span style="color:${SC.bad}">Without the continuity correction the curve flattens: a lattice CDF is a staircase, ` +
          `and half a step is an error no n removes fast. Tick the box and the −1/2 slope returns.</span>`
        : `<br><span style="color:${SC.muted}">Every point is computed exactly — no simulation — so the slope is the real one, not noise.` +
          (C.g1 !== 0 && tail.length && Math.abs(tail[tail.length - 1].K / (CE_ * Math.abs(C.g1) / Math.sqrt(tail[tail.length - 1].n)) - 1) < 0.05
            ? " The orange theory line is invisible because the blue one is sitting exactly on top of it."
            : "") + `</span>`);
  }

  sel.addEventListener("change", render);
  cCC.addEventListener("change", render);
  render();
})();

/* ─────────────────── 6 · where the CLT fails: a Cauchy population ─────────────────── */
(function () {
  const svg = d3.select("#cauchy-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 500, REPS = 4000;
  const F = ST.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
  const TOPH = 250, BOTY = 296, BOTH = 150;
  const CE = ST.cells(F.iw, TOPH, 2, 1, { l: 40, r: 12, t: 24, b: 34 }, { x: 26, y: 0 });
  const gA = F.g.append("g").attr("transform", `translate(${CE[0].x},${CE[0].y})`);
  const gB = F.g.append("g").attr("transform", `translate(${CE[1].x},${CE[1].y})`);
  const gC = F.g.append("g").attr("transform", `translate(46,${BOTY})`);

  const sN = document.getElementById("ca-n"), sNv = document.getElementById("ca-nv");
  const out = document.getElementById("cauchy-readout");
  const NG = [1, 2, 5, 10, 30, 100, 300, 1000];
  let seed = 626;

  const cauchy = r => Math.tan(Math.PI * (r() - 0.5));
  const cauchyPdf = x => 1 / (Math.PI * (1 + x * x));

  function panelHist(g, c, vals, colour, curve, title, sub, nbins) {
    g.selectAll("*").remove();
    const gi = g.append("g").attr("transform", `translate(${c.m.l},${c.m.t})`);
    g.append("text").attr("x", c.m.l).attr("y", 14).attr("font-size", 11).attr("fill", colour).text(title);
    const LIM = 8, NB = nbins || 48, bw = 2 * LIM / NB;
    const inside = vals.filter(v => v >= -LIM && v <= LIM);
    const bins = ST.histBins(inside, -LIM, bw, NB);
    const dens = bins.map(b => ({ x0: b.x0, d: b.n / (vals.length * bw) }));
    const xs = d3.scaleLinear().domain([-LIM, LIM]).range([0, c.iw]);
    const top = Math.max(d3.max(dens, b => b.d) * 1.1, d3.max(curve, p => p.d) * 1.1);
    const ys = d3.scaleLinear().domain([0, top]).range([c.ih, 0]);
    gi.selectAll("rect").data(dens).join("rect")
      .attr("x", b => xs(b.x0)).attr("width", Math.max(0.8, xs(bw) - xs(0) - 0.6))
      .attr("y", b => ys(b.d)).attr("height", b => c.ih - ys(b.d))
      .attr("fill", colour).attr("fill-opacity", 0.5);
    gi.append("path").datum(curve).attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2)
      .attr("d", d3.line().x(p => xs(p.v)).y(p => ys(p.d)));
    gi.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`).call(d3.axisBottom(xs).ticks(5));
    gi.append("text").attr("x", c.iw).attr("y", c.ih + 30).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.muted).text(sub);
    const off = vals.length - inside.length;
    if (off) gi.append("text").attr("x", 0).attr("y", c.ih + 30).attr("font-size", 10)
      .attr("fill", SC.bad).text(`${off} off-scale`);
  }

  function render() {
    const n = NG[+sN.value]; sNv.textContent = n;
    const r = ST.rng(seed);
    const means = [], meds = [];
    const buf = new Float64Array(n);
    for (let i = 0; i < REPS; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) { const v = cauchy(r); buf[j] = v; s += v; }
      means.push(s / n);
      buf.sort();                                   // typed arrays sort numerically, in place
      meds.push(n % 2 ? buf[(n - 1) / 2] : (buf[n / 2 - 1] + buf[n / 2]) / 2);
    }
    const grid = ST.linspace(-8, 8, 240);
    panelHist(gA, CE[0], means, SC.bad, grid.map(v => ({ v: v, d: cauchyPdf(v) })),
      "the sample MEAN x̄ₙ", "green = Cauchy(0,1), the same for every n");
    const seMed = Math.PI / (2 * Math.sqrt(n));
    panelHist(gB, CE[1], meds, SC.accent,
      grid.map(v => ({ v: v, d: ST.normPdf(v / seMed) / seMed })),
      "the sample MEDIAN — same axis", `green = Normal(0, π/(2√n)) = ${ST.fmt(seMed, 3)}`,
      Math.round(ST.clamp(16 / (seMed / 2.5), 48, 340)));

    /* bottom strip: one running average, showing the jumps */
    gC.selectAll("*").remove();
    const KM = 20000, rr = ST.rng(seed + 99);
    const trace = [];
    let s = 0;
    for (let k = 1; k <= KM; k++) {
      s += cauchy(rr);
      if (k < 40 || k % Math.max(1, Math.floor(k / 40)) === 0) trace.push({ k: k, m: s / k });
    }
    const xs = d3.scaleLog().domain([1, KM]).range([0, F.iw - 60]);
    const lim = Math.max(6, d3.max(trace, t => Math.abs(t.m)) * 1.1);
    const ys = d3.scaleLinear().domain([-lim, lim]).range([BOTH, 0]);
    gC.append("line").attr("x1", 0).attr("x2", F.iw - 60).attr("y1", ys(0)).attr("y2", ys(0))
      .attr("stroke", SC.good).attr("stroke-width", 1.4);
    gC.append("path").datum(trace).attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 1.8)
      .attr("d", d3.line().x(t => xs(t.k)).y(t => ys(t.m)));
    gC.append("g").attr("class", "axis").attr("transform", `translate(0,${BOTH})`).call(d3.axisBottom(xs).ticks(5, "~s"));
    gC.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    gC.append("text").attr("x", 2).attr("y", -5).attr("font-size", 11).attr("fill", SC.bad)
      .text("one running average of Cauchy draws — every so often a single draw resets it");
    gC.append("text").attr("x", F.iw - 60).attr("y", BOTH + 32).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("draws k (log)");

    const pOut = means.filter(v => Math.abs(v) > 3).length / REPS;
    const iqrM = ST.iqr(means), sdMed = ST.sd(meds);
    out.innerHTML =
      `n = <b>${n}</b> · ${REPS} simulated samples from a Cauchy(0, 1) population — which has <b>no mean and no variance</b>` +
      `<br>P(|x̄ₙ| &gt; 3): measured <b>${ST.pct(pOut, 2)}</b> · exact, for <i>every</i> n: <b>${ST.pct(1 - 2 / Math.PI * Math.atan(3), 2)}</b> — ` +
      `it does not move, because x̄ₙ is itself Cauchy(0, 1)` +
      `<br>IQR of the x̄'s: <b>${ST.fmt(iqrM, 3)}</b> (exactly 2 for every n) · ` +
      `SD of the medians: <b>${ST.fmt(sdMed, 3)}</b> vs the theory π/(2√n) = <b>${ST.fmt(seMed, 3)}</b>` +
      `<br><span style="color:${SC.muted}">No hypothesis was violated by accident here: the CLT asks for a finite variance and this ` +
      `population has none. Change the statistic — take the median — and convergence comes straight back.</span>`;
  }
  sN.addEventListener("input", render);
  document.getElementById("ca-again").addEventListener("click", () => { seed += 37; render(); });
  render();
})();

/* ─────────────────── 7 · the 1/√n law, priced ─────────────────── */
(function () {
  const svg = d3.select("#root-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 400;
  const F = ST.frame(svg, W, H, { l: 62, r: 130, t: 18, b: 44 });
  const x = d3.scaleLog().domain([10, 100000]).range([0, F.iw]);

  const sN = document.getElementById("rn-n"), sNv = document.getElementById("rn-nv");
  const sSd = document.getElementById("rn-sd"), sSdv = document.getElementById("rn-sdv");
  const sPop = document.getElementById("rn-N"), sPopv = document.getElementById("rn-Nv");
  const out = document.getElementById("root-readout");

  function render() {
    const sd = +sSd.value, n = Math.round(Math.pow(10, +sN.value));
    const N = Math.round(Math.pow(10, +sPop.value));
    sNv.textContent = n.toLocaleString("en-US");
    sSdv.textContent = ST.fmt(sd, 0);
    sPopv.textContent = N >= 1e7 ? "∞ (ignore)" : N.toLocaleString("en-US");
    const se = k => 1.96 * sd / Math.sqrt(k);
    const seF = k => (k >= N ? 0 : 1.96 * sd / Math.sqrt(k) * Math.sqrt((N - k) / (N - 1)));

    F.g.selectAll("*").remove();
    const y = d3.scaleLog().domain([Math.max(1e-3, se(100000) * 0.7), se(10) * 1.25]).range([F.ih, 0]);
    F.g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(6))
      .join("line").attr("x1", 0).attr("x2", F.iw).attr("y1", v => y(v)).attr("y2", v => y(v)).attr("stroke", SC.grid);

    const ks = ST.linspace(1, 5, 160).map(e => Math.pow(10, e));
    const ln = f => d3.line().x(k => x(k)).y(k => y(ST.clamp(f(k), y.domain()[0], y.domain()[1])));
    F.g.append("path").datum(ks).attr("d", ln(se)).attr("fill", "none")
      .attr("stroke", SC.accent).attr("stroke-width", 2.4);
    if (N < 1e7) {
      F.g.append("path").datum(ks.filter(k => k < N)).attr("d", ln(seF)).attr("fill", "none")
        .attr("stroke", SC.violet).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
      F.g.append("line").attr("x1", x(Math.min(N, 100000))).attr("x2", x(Math.min(N, 100000)))
        .attr("y1", 0).attr("y2", F.ih).attr("stroke", SC.violet).attr("stroke-width", 1).attr("opacity", 0.6);
    }

    /* the doubling ladder: each step down costs 4× the data */
    const base = se(n);
    [1, 2, 3].forEach(j => {
      const kk = n * Math.pow(4, j);
      if (kk > 100000) return;
      F.g.append("line").attr("x1", x(n)).attr("x2", x(kk)).attr("y1", y(se(kk)))
        .attr("y2", y(se(kk))).attr("stroke", SC.a2).attr("stroke-width", 1)
        .attr("stroke-dasharray", "2 3").attr("opacity", 0.55);
    });
    [1, 2, 3].forEach(j => {
      const kk = n * Math.pow(4, j);
      if (kk > 100000) return;
      F.g.append("circle").attr("cx", x(kk)).attr("cy", y(se(kk))).attr("r", 4)
        .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);
      F.g.append("text").attr("x", x(kk)).attr("y", y(se(kk)) - 9).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", SC.a2).text(`÷${Math.pow(2, j)}`);
    });

    F.g.append("line").attr("x1", x(n)).attr("x2", x(n)).attr("y1", 0).attr("y2", F.ih)
      .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
    F.g.append("circle").attr("cx", x(n)).attr("cy", y(base)).attr("r", 5).attr("fill", SC.good);

    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).ticks(6, "~s"));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6, "~g"));
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 40).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("sample size n (log)");
    F.g.append("text").attr("x", 2).attr("y", -5).attr("font-size", 11).attr("fill", SC.accent)
      .text("95% margin of error, 1.96 · σ/√n (log–log: a straight line of slope −1/2)");
    ST.legend(F.g, [{ label: "infinite population", color: SC.accent },
      { label: "with the FPC", color: SC.violet, dash: "5 3" }], F.iw + 8, 12, { gap: 15, font: 10 });

    const cost = k => Math.ceil(k * (Math.pow(base / (base - 0.1 * base), 2) - 1));
    out.innerHTML =
      `σ = ${ST.fmt(sd, 0)} · n = <b>${n.toLocaleString("en-US")}</b> → SE = σ/√n = <b>${ST.fmt(sd / Math.sqrt(n), 4)}</b>, ` +
      `95% margin = <b>±${ST.fmt(base, 4)}</b>` +
      `<br>to halve that margin you need n = <b>${(4 * n).toLocaleString("en-US")}</b> (4×) · ` +
      `to cut it to a third, n = <b>${(9 * n).toLocaleString("en-US")}</b> (9×) · to a tenth, n = <b>${(100 * n).toLocaleString("en-US")}</b> (100×)` +
      `<br>the next 10% of precision alone costs <b>+${cost(n).toLocaleString("en-US")}</b> extra units — ` +
      `the marginal price of accuracy rises without limit` +
      (N < 1e7
        ? `<br>finite population N = ${N.toLocaleString("en-US")}: margin with the FPC = <b>±${ST.fmt(seF(Math.min(n, N)), 4)}</b> ` +
          `(a factor √((N−n)/(N−1)) = ${ST.fmt(n >= N ? 0 : ST.fpc(n, N), 3)}); it only bites once n is a real fraction of N, ` +
          `and it reaches exactly zero at the census n = N`
        : `<br><span style="color:${SC.muted}">With N effectively infinite, only n matters — the population's size never enters the formula. ` +
          `A 1 000-person sample is as precise in a country of 60 million as in a town of 60 thousand.</span>`);
  }
  [sN, sSd, sPop].forEach(s => s.addEventListener("input", render));
  render();
})();

/* ─────────────────── 8 · binomial → normal ─────────────────── */
(function () {
  const svg = d3.select("#binom-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 400;
  const F = ST.frame(svg, W, H, { l: 56, r: 16, t: 22, b: 44 });
  const sN = document.getElementById("bn-n"), sNv = document.getElementById("bn-nv");
  const sP = document.getElementById("bn-p"), sPv = document.getElementById("bn-pv");
  const sK = document.getElementById("bn-k"), sKv = document.getElementById("bn-kv");
  const out = document.getElementById("binom-readout");
  const lights = document.getElementById("bn-lights");

  function render() {
    const n = +sN.value, p = +sP.value;
    sNv.textContent = n; sPv.textContent = ST.fmt(p, 2);
    const m = n * p, s = Math.sqrt(n * p * (1 - p));
    const lo = Math.max(0, Math.floor(m - 4.4 * s)), hi = Math.min(n, Math.ceil(m + 4.4 * s));
    sK.min = lo; sK.max = hi;
    let k = ST.clamp(+sK.value, lo, hi);
    if (+sK.value !== k) sK.value = k;
    sKv.textContent = k;

    const bars = [];
    let peak = 0;
    for (let i = lo; i <= hi; i++) { const q = ST.binomPmf(i, n, p); bars.push({ k: i, p: q }); peak = Math.max(peak, q); }

    F.g.selectAll("*").remove();
    const x = d3.scaleLinear().domain([lo - 0.6, hi + 0.6]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, Math.max(peak, ST.normPdf(0) / s) * 1.1]).range([F.ih, 0]);
    ST.gridY(F.g, y, F.iw, 4);
    const bw = Math.max(1.2, (F.iw / (hi - lo + 1.2)) - 1.2);
    F.g.selectAll("rect.b").data(bars).join("rect").attr("class", "b")
      .attr("x", b => x(b.k) - bw / 2).attr("width", bw)
      .attr("y", b => y(b.p)).attr("height", b => F.ih - y(b.p))
      .attr("fill", b => b.k <= k ? SC.accent : SC.muted)
      .attr("fill-opacity", b => b.k <= k ? 0.75 : 0.35);
    F.g.append("path").datum(ST.linspace(lo - 0.6, hi + 0.6, 260).map(v => ({ v: v, d: ST.normPdf((v - m) / s) / s })))
      .attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(q => x(q.v)).y(q => y(q.d)));
    F.g.append("line").attr("x1", x(k + 0.5)).attr("x2", x(k + 0.5)).attr("y1", 0).attr("y2", F.ih)
      .attr("stroke", SC.a2).attr("stroke-width", 2);
    F.g.append("text").attr("x", x(k + 0.5) + 5).attr("y", 12).attr("font-size", 10.5).attr("fill", SC.a2)
      .text(`k + ½ = ${k + 0.5}`);
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(9, "d"));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4, ".3f"));
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 40).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", SC.muted).text("number of successes X");
    F.g.append("text").attr("x", 2).attr("y", -6).attr("font-size", 11).attr("fill", SC.accent)
      .text(`exact Binomial(${n}, ${ST.fmt(p, 2)}) bars · Normal(np, √(np(1−p))) curve`);

    const exact = ST.binomCdf(k, n, p);
    const app0 = ST.normCdf((k - m) / s), app1 = ST.normCdf((k + 0.5 - m) / s);
    let worstCC = 0, worst0 = 0, cdf = 0;
    for (let i = 0; i <= n; i++) {
      const below = cdf; cdf += ST.binomPmf(i, n, p);
      worstCC = Math.max(worstCC, Math.abs(cdf - ST.normCdf((i + 0.5 - m) / s)));
      worst0 = Math.max(worst0, Math.abs(cdf - ST.normCdf((i - m) / s)), Math.abs(below - ST.normCdf((i - m) / s)));
    }
    const g1 = (1 - 2 * p) / Math.sqrt(p * (1 - p));
    const c1 = n * p >= 10, c2 = n * (1 - p) >= 10;
    const chip = (ok, t) =>
      `<span style="display:inline-block;padding:2px 9px;border-radius:999px;margin-right:6px;font-size:11.5px;` +
      `border:1px solid ${ok ? SC.good : SC.bad};color:${ok ? SC.good : SC.bad}">${ok ? "✓" : "✗"} ${t}</span>`;
    lights.innerHTML = chip(c1, `np = ${ST.fmt(m, 1)} ≥ 10`) + chip(c2, `n(1−p) = ${ST.fmt(n * (1 - p), 1)} ≥ 10`) +
      chip(worstCC <= 0.02, `worst CDF error ${ST.pct(worstCC, 2)} ≤ 2%`);

    out.innerHTML =
      `P(X ≤ ${k}) — exact: <b>${ST.fmt(exact, 5)}</b> · normal with the ½ correction: <b>${ST.fmt(app1, 5)}</b> ` +
      `(off by ${ST.fmt(Math.abs(app1 - exact), 5)}) · without it: <b>${ST.fmt(app0, 5)}</b> (off by ${ST.fmt(Math.abs(app0 - exact), 5)})` +
      `<br>mean np = <b>${ST.fmt(m, 2)}</b> · SD √(np(1−p)) = <b>${ST.fmt(s, 3)}</b> · ` +
      `as a proportion: p̂ = X/n has SE = √(p(1−p)/n) = <b>${ST.fmt(ST.seProp(p, n), 4)}</b>` +
      `<br>worst error anywhere in the CDF: <b>${ST.pct(worstCC, 2)}</b> with the correction, <b>${ST.pct(worst0, 2)}</b> without · ` +
      `Edgeworth prediction |γ₁|/(6√(2π)·√n) = ${ST.pct(Math.abs(g1) / (6 * Math.sqrt(2 * Math.PI) * Math.sqrt(n)), 2)} (γ₁ = ${ST.fmt(g1, 2)})` +
      `<br><span style="color:${SC.muted}">The ½ is not a fudge: a staircase is being approximated by a smooth curve, and the ` +
      `midpoint of each step is where the two agree.</span>`;
  }
  [sN, sP, sK].forEach(s => s.addEventListener("input", render));
  render();
})();

/* ─────────────────── 9 · a validation score is a statistic ─────────────────── */
(function () {
  const svg = d3.select("#ml-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 380;
  const F = ST.frame(svg, W, H, { l: 58, r: 18, t: 20, b: 44 });
  const sN = document.getElementById("mv-n"), sNv = document.getElementById("mv-nv");
  const sD = document.getElementById("mv-d"), sDv = document.getElementById("mv-dv");
  const sDis = document.getElementById("mv-dis"), sDisv = document.getElementById("mv-disv");
  const cPair = document.getElementById("mv-pair");
  const out = document.getElementById("ml-readout");
  const BASE = 0.90;

  function render() {
    const n = Math.round(Math.pow(10, +sN.value));
    const d = +sD.value / 1000;                       // true accuracy gap, in points/1000
    const dis = +sDis.value / 100;                    // disagreement rate between the two models
    const paired = cPair.checked;
    sNv.textContent = n.toLocaleString("en-US");
    sDv.textContent = ST.fmt(100 * d, 1) + " pts";
    sDisv.textContent = ST.pct(dis, 0);

    const pA = BASE, pB = Math.min(0.999, BASE + d);
    const seA = ST.seProp(pA, n), seB = ST.seProp(pB, n);
    const seU = Math.sqrt(seA * seA + seB * seB);
    const dEff = Math.max(dis, d + 1e-9);
    const seP = Math.sqrt((dEff - d * d) / n);
    const seD = paired ? seP : seU;
    const pWrong = ST.normCdf(-d / seD);

    F.g.selectAll("*").remove();
    const half = Math.max(4 * seA, 1.6 * d + 3 * seA);
    const x = d3.scaleLinear().domain([BASE - half, BASE + Math.max(half, d + 4 * seB)]).range([0, F.iw]);
    const top = Math.max(ST.normPdf(0) / seA, ST.normPdf(0) / seB);
    const y = d3.scaleLinear().domain([0, top * 1.14]).range([F.ih, 0]);
    ST.gridY(F.g, y, F.iw, 4);
    const grid = ST.linspace(x.domain()[0], x.domain()[1], 300);
    const curve = (mu, se, col, op) => {
      const pts = grid.map(v => ({ v: v, d: ST.normPdf((v - mu) / se) / se }));
      F.g.append("path").datum(pts).attr("fill", col).attr("fill-opacity", op)
        .attr("d", d3.area().x(q => x(q.v)).y0(F.ih).y1(q => y(q.d)));
      F.g.append("path").datum(pts).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2)
        .attr("d", d3.line().x(q => x(q.v)).y(q => y(q.d)));
      F.g.append("line").attr("x1", x(mu)).attr("x2", x(mu)).attr("y1", y(ST.normPdf(0) / se)).attr("y2", F.ih)
        .attr("stroke", col).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    };
    curve(pA, seA, SC.muted, 0.22);
    curve(pB, seB, SC.accent, 0.22);
    F.g.append("text").attr("x", x(pA)).attr("y", y(ST.normPdf(0) / seA) - 8).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("model A");
    F.g.append("text").attr("x", x(pB)).attr("y", y(ST.normPdf(0) / seB) - 22).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", SC.accent).text("model B (truly better)");
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).ticks(7).tickFormat(d3.format(".1%")));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 40).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", SC.muted).text("accuracy measured on ONE validation set of n examples");
    F.g.append("text").attr("x", 2).attr("y", -5).attr("font-size", 11).attr("fill", SC.accent)
      .text("the sampling distribution of a held-out score — not a number, a distribution");

    /* the ±2 SE ruler for the difference */
    const yR = F.ih - 18;
    F.g.append("line").attr("x1", x(pA)).attr("x2", x(pB)).attr("y1", yR).attr("y2", yR)
      .attr("stroke", SC.a2).attr("stroke-width", 2);
    F.g.append("text").attr("x", x((pA + pB) / 2)).attr("y", yR - 9).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.a2).text(`gap = ${ST.fmt(d / seD, 2)} SE of the difference`);

    const nNeed = paired
      ? Math.ceil(4 * (dEff - d * d) / (d * d))
      : Math.ceil(4 * (pA * (1 - pA) + pB * (1 - pB)) / (d * d));
    out.innerHTML =
      `validation set n = <b>${n.toLocaleString("en-US")}</b> · model A is truly ${ST.pct(pA, 1)}, model B truly ` +
      `${ST.pct(pB, 1)} — a real gap of <b>${ST.fmt(100 * d, 2)}</b> points` +
      `<br>SE of one score = √(p(1−p)/n) = <b>${ST.fmt(seA, 5)}</b> → a single reported accuracy carries ` +
      `<b>±${ST.fmt(196 * seA, 2)}</b> points of pure luck at 95%` +
      `<br>SE of the <b>difference</b> (${paired ? "paired — same examples, " + ST.pct(dis, 0) + " disagreement" : "unpaired — as if two separate test sets"}): ` +
      `<b>${ST.fmt(seD, 5)}</b> · the true gap is <b>${ST.fmt(d / seD, 2)} SE</b> wide` +
      `<br>chance the <b>worse</b> model wins this bake-off: <b>${ST.pct(pWrong, 1)}</b> · ` +
      `n needed for a 2-SE separation: <b>${nNeed.toLocaleString("en-US")}</b> examples` +
      `<br><span style="color:${SC.muted}">Pairing is free and it is the single largest win available: the same examples cancel ` +
      `most of the noise, leaving only the cases where the two models actually disagree.</span>`;
  }
  [sN, sD, sDis].forEach(s => s.addEventListener("input", render));
  cPair.addEventListener("change", render);
  render();
})();
