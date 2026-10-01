/* discrete-distributions.viz.js — the eight visualizations on math/probability/discrete-distributions.html.
   Loaded after ../../data.js → ../../notes.js → ../statistics/stats-viz.js (SC palette, ST toolbox)
   → prob-viz.js (PV: Poisson, geometric, negative binomial, hypergeometric, categorical helpers).
   Distribution functions come from ST / PV and are never re-implemented here.

   Everything page-local lives under the single namespace RV. Each figure is an IIFE that exits
   quietly if its <svg> is not on the page.

     1  #rv-svg        a random variable as a function on the 36 outcomes of two dice
     2  #pmfcdf-svg    PMF + CDF explorer for the discrete families, with P(a < X ≤ b)
     3  #indic-svg     sums of Bernoulli indicators building a binomial histogram
     4  #geom-svg      memorylessness: the renormalised tail vs a fresh copy
     5  #nb-svg        overdispersion: negative binomial vs Poisson, gamma–Poisson mixture simulation
     6  #hyper-svg     sampling without replacement: hypergeometric vs binomial
     7  #poisconv-svg  binomial → Poisson, total-variation distance against n
     8  #cat-svg       a categorical die (softmax with temperature) building multinomial counts   */

const RV = {
  // Default seeds were chosen by the caption audit (see the note at each figure).
  seeds: { indic: 6, nb: 1, hyper: 4, cat: 2, hyperMc: 11 },
  fmt: (x, d) => ST.fmt(x, d),
  // tabulate a pmf over an integer range
  tab(fn, lo, hi) { const out = []; for (let k = lo; k <= hi; k++) out.push({ k: k, p: fn(k) }); return out; },
  moments(sup) {
    let m = 0, m2 = 0, s = 0;
    sup.forEach(d => { s += d.p; m += d.k * d.p; m2 += d.k * d.k * d.p; });
    return { sum: s, mean: m, var: m2 - m * m };
  },
  // Gamma(shape a, scale 1) draw — Marsaglia–Tsang, with the a < 1 boost
  gammaDraw(a, r) {
    if (a < 1) { let u = r(); while (u === 0) u = r(); return RV.gammaDraw(a + 1, r) * Math.pow(u, 1 / a); }
    const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = ST.randn(r); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = r();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  },
  // total-variation distance between Bin(n, μ/n) and Poisson(μ), including the Poisson tail beyond n
  tvBinPois(n, mu) {
    const p = mu / n;
    const K = Math.max(n, Math.ceil(mu + 20 * Math.sqrt(mu) + 30));
    let s = 0;
    for (let k = 0; k <= K; k++) s += Math.abs(ST.binomPmf(k, n, p) - PV.poisPmf(k, mu));
    let tail = 1 - PV.poisCdf(K, mu);
    return (s + Math.max(0, tail)) / 2;
  }
};

/* ─────────────────── 1 · a random variable on two dice ───────────────────
   Caption audit: default X = max, selected value 4 → preimage of 7 cells forming an L along
   row 4 and column 4, P(M = 4) = 7/36 = 0.194. Indicator of a double → two bars, 30/36 and 6/36. */
(function () {
  const svg = d3.select("#rv-svg");
  if (svg.empty() || typeof ST === "undefined") return;
  const W = 680, H = 340;
  const sel = document.getElementById("rv-x"), out = document.getElementById("rv-readout");
  const fns = {
    sum: (a, b) => a + b, max: (a, b) => Math.max(a, b), six: (a, b) => (a === 6) + (b === 6),
    dbl: (a, b) => (a === b ? 1 : 0), diff: (a, b) => Math.abs(a - b)
  };
  const names = { sum: "S", max: "M", six: "Z", dbl: "D", diff: "|ω₁ − ω₂|" };
  const defSel = { sum: 7, max: 4, six: 1, dbl: 1, diff: 1 };
  let which = "max", cur = 4;
  const cs = 38, gx = 44, gy = 52;

  function draw() {
    svg.selectAll("*").remove();
    const f = fns[which];
    const cells = [];
    for (let b = 1; b <= 6; b++) for (let a = 1; a <= 6; a++) cells.push({ a: a, b: b, v: f(a, b) });
    const vals = [...new Set(cells.map(c => c.v))].sort((x, y) => x - y);
    const cnt = vals.map(v => ({ v: v, n: cells.filter(c => c.v === v).length }));
    const g = svg.append("g");
    g.append("text").attr("x", gx + 3 * cs).attr("y", gy - 26).attr("text-anchor", "middle")
      .attr("font-size", 11).attr("fill", SC.muted).text("first die ω₁");
    g.append("text").attr("x", gx - 26).attr("y", gy + 3 * cs).attr("text-anchor", "middle")
      .attr("font-size", 11).attr("fill", SC.muted)
      .attr("transform", `rotate(-90,${gx - 26},${gy + 3 * cs})`).text("second die ω₂");
    for (let i = 1; i <= 6; i++) {
      g.append("text").attr("x", gx + (i - 0.5) * cs).attr("y", gy - 8).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", SC.muted).text(i);
      g.append("text").attr("x", gx - 8).attr("y", gy + (i - 0.5) * cs + 4).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", SC.muted).text(i);
    }
    const cg = g.selectAll("g.cell").data(cells).join("g").attr("class", "cell")
      .attr("transform", d => `translate(${gx + (d.a - 1) * cs},${gy + (d.b - 1) * cs})`)
      .style("cursor", "pointer")
      .on("mouseenter", (e, d) => { cur = d.v; draw(); });
    cg.append("rect").attr("width", cs - 2).attr("height", cs - 2).attr("rx", 3)
      .attr("fill", d => d.v === cur ? SC.a2 : SC.panel2)
      .attr("fill-opacity", d => d.v === cur ? 0.85 : 1)
      .attr("stroke", SC.line);
    cg.append("text").attr("x", (cs - 2) / 2).attr("y", cs / 2 + 3).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", d => d.v === cur ? SC.bg : SC.ink).text(d => d.v);

    // pmf bars
    const px0 = 330, px1 = 660, py0 = 300, py1 = 40;
    const x = d3.scaleBand().domain(vals).range([px0, px1]).padding(0.18);
    const ymax = d3.max(cnt, d => d.n) / 36;
    const y = d3.scaleLinear().domain([0, ymax * 1.15]).range([py0, py1]);
    const ga = g.append("g");
    ga.append("g").attr("class", "axis").attr("transform", `translate(0,${py0})`).call(d3.axisBottom(x));
    ga.append("g").attr("class", "axis").attr("transform", `translate(${px0},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".2f")));
    ga.append("text").attr("x", px0).attr("y", py1 - 12).attr("font-size", 11).attr("fill", SC.muted)
      .text(`p(x) = P(${names[which]} = x) = (number of cells labelled x)/36`);
    ga.append("text").attr("x", px1).attr("y", py0 + 30).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", SC.muted).text("value x");
    const bars = g.selectAll("g.bar").data(cnt).join("g").attr("class", "bar").style("cursor", "pointer")
      .on("mouseenter", (e, d) => { cur = d.v; draw(); });
    bars.append("rect").attr("x", d => x(d.v)).attr("width", x.bandwidth())
      .attr("y", d => y(d.n / 36)).attr("height", d => py0 - y(d.n / 36))
      .attr("fill", d => d.v === cur ? SC.a2 : SC.accent).attr("fill-opacity", 0.85);
    bars.append("text").attr("x", d => x(d.v) + x.bandwidth() / 2).attr("y", d => y(d.n / 36) - 4)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.muted).text(d => `${d.n}/36`);

    const pre = cells.filter(c => c.v === cur);
    let m = 0, m2 = 0;
    cnt.forEach(d => { m += d.v * d.n / 36; m2 += d.v * d.v * d.n / 36; });
    out.innerHTML = `{${names[which]} = ${cur}} = {${pre.map(c => `(${c.a},${c.b})`).join(", ")}} — ` +
      `<b>${pre.length}</b> of 36 outcomes, so P(${names[which]} = ${cur}) = ${pre.length}/36 = <b>${ST.fmt(pre.length / 36, 3)}</b>` +
      `<br>pmf: ${cnt.map(d => `${d.v}: ${d.n}/36`).join(" · ")} · Σ = ${d3.sum(cnt, d => d.n)}/36` +
      `<br>E ${names[which]} = <b>${ST.fmt(m, 3)}</b> · Var ${names[which]} = <b>${ST.fmt(m2 - m * m, 3)}</b>`;
  }
  sel.addEventListener("change", () => { which = sel.value; cur = defSel[which]; draw(); });
  document.getElementById("rv-reset").addEventListener("click", () => { sel.value = "max"; which = "max"; cur = 4; draw(); });
  draw();
})();

/* ─────────────────── 2 · PMF + CDF explorer ───────────────────
   Caption audit: default Bin(10, ¼), a = 5, b = 10 → shaded k = 6…10, P(5 < X ≤ 10) = 1 − F(5)
   = 0.0197; numerical mean 2.5 and variance 1.875 equal the closed forms.                     */
(function () {
  const svg = d3.select("#pmfcdf-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, H = 420;
  const $ = id => document.getElementById(id);
  const fam = $("pc-fam"), out = $("pmfcdf-readout");
  const ctl = ["n", "p", "r", "N", "K", "m", "k", "a", "b"];
  const show = { unif: ["k"], bern: ["p"], bin: ["n", "p"], geom: ["p"], nb: ["r", "p"], hyp: ["N", "K", "n"], pois: ["m"] };
  const defaults = { fam: "bin", n: 10, p: 0.25, r: 3, N: 30, K: 12, m: 4, k: 6, a: 5, b: 10 };

  function v(id) { return +$("pc-" + id).value; }
  function spec() {
    const f = fam.value, n = v("n"), p = v("p"), r = v("r"), m = v("m"), k = v("k");
    const N = v("N"), K = Math.min(v("K"), N), nn = Math.min(n, N);
    switch (f) {
      case "unif": return { name: `discrete uniform on 1…${k}`, fn: x => (x >= 1 && x <= k ? 1 / k : 0), lo: 1, hi: k, E: (k + 1) / 2, V: (k * k - 1) / 12 };
      case "bern": return { name: `Bernoulli(${ST.fmt(p, 2)})`, fn: x => (x === 1 ? p : x === 0 ? 1 - p : 0), lo: 0, hi: 1, E: p, V: p * (1 - p) };
      case "bin": return { name: `Bin(${n}, ${ST.fmt(p, 2)})`, fn: x => ST.binomPmf(x, n, p), lo: 0, hi: n, E: n * p, V: n * p * (1 - p) };
      case "geom": return { name: `Geo(${ST.fmt(p, 2)})`, fn: x => PV.geomPmf(x, p), lo: 1, hi: null, E: 1 / p, V: (1 - p) / (p * p) };
      case "nb": return { name: `NB(r = ${r}, ${ST.fmt(p, 2)})`, fn: x => PV.negBinPmf(x, r, p), lo: r, hi: null, E: r / p, V: r * (1 - p) / (p * p) };
      case "hyp": {
        const q = K / N;
        return { name: `Hypergeometric(N = ${N}, K = ${K}, n = ${nn})`, fn: x => PV.hyperPmf(x, N, K, nn),
          lo: 0, hi: nn, E: nn * q, V: nn * q * (1 - q) * (N > 1 ? (N - nn) / (N - 1) : 0) };
      }
      default: return { name: `Poisson(${ST.fmt(m, 1)})`, fn: x => PV.poisPmf(x, m), lo: 0, hi: null, E: m, V: m };
    }
  }
  function draw() {
    const f = fam.value;
    ctl.slice(0, 7).forEach(c => { $("pc-" + c + "w").style.display = show[f].includes(c) ? "" : "none"; });
    ctl.forEach(c => { const o = $("pc-" + c + "v"); if (o) o.textContent = c === "p" ? ST.fmt(v(c), 2) : c === "m" ? ST.fmt(v(c), 1) : v(c); });
    const S = spec();
    // long support for moments; display support trimmed at 99.9%
    let full;
    if (S.hi !== null) full = RV.tab(S.fn, S.lo, S.hi);
    else full = PV.support(S.fn, S.lo, 1e-12, 6000);
    const mo = RV.moments(full);
    let disp = full, acc = 0;
    if (S.hi === null) { disp = []; for (const d of full) { disp.push(d); acc += d.p; if (acc > 0.999 && disp.length > 3) break; } }
    if (disp.length > 61) disp = disp.slice(0, 61);
    const a = v("a"), b = v("b");
    const F = t => { let s = 0; for (const d of full) { if (d.k <= t) s += d.p; else break; } return Math.min(1, s); };
    const Fa = F(a), Fb = F(b), Pab = Math.max(0, Fb - Fa);
    // median
    let med = null, cum = 0;
    for (const d of full) { cum += d.p; if (cum >= 0.5 - 1e-12) { med = d.k; break; } }

    const F2 = ST.frame(svg, W, H, { l: 52, r: 18, t: 22, b: 30 });
    const g = F2.g, iw = F2.iw;
    const xlo = disp[0].k - 1, xhi = disp[disp.length - 1].k + 1;
    const x = d3.scaleLinear().domain([xlo - 0.5, xhi + 0.5]).range([0, iw]);
    const bw = Math.max(2, Math.min(28, 0.7 * (x(1) - x(0))));
    const top = { y0: 160, y1: 0 }, bot = { y0: 360, y1: 205 };
    const yP = d3.scaleLinear().domain([0, d3.max(disp, d => d.p) * 1.12 || 1]).range([top.y0, top.y1]);
    const yF = d3.scaleLinear().domain([0, 1]).range([bot.y0, bot.y1]);
    const tickF = d3.format("d");
    const xt = x.ticks(Math.min(12, xhi - xlo + 1)).filter(t => Number.isInteger(t));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${top.y0})`).call(d3.axisBottom(x).tickValues(xt).tickFormat(tickF));
    g.append("g").attr("class", "axis").call(d3.axisLeft(yP).ticks(4));
    g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.muted).text(`pmf p(k) of ${S.name}`);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${bot.y0})`).call(d3.axisBottom(x).tickValues(xt).tickFormat(tickF));
    g.append("g").attr("class", "axis").call(d3.axisLeft(yF).ticks(4)).attr("transform", "translate(0,0)");
    g.append("text").attr("x", 0).attr("y", bot.y1 - 8).attr("font-size", 11).attr("fill", SC.muted).text("cdf F(x) = P(X ≤ x)");
    // shaded interval (a, b]
    const xa = Math.max(xlo - 0.5, a + 0.5), xb = Math.min(xhi + 0.5, b + 0.5);
    if (xb > xa) g.append("rect").attr("x", x(xa)).attr("width", x(xb) - x(xa)).attr("y", top.y1).attr("height", top.y0 - top.y1)
      .attr("fill", SC.a2).attr("opacity", 0.08);
    g.selectAll("rect.pm").data(disp).join("rect").attr("class", "pm")
      .attr("x", d => x(d.k) - bw / 2).attr("width", bw).attr("y", d => yP(d.p)).attr("height", d => top.y0 - yP(d.p))
      .attr("fill", d => (d.k > a && d.k <= b) ? SC.a2 : SC.accent).attr("fill-opacity", 0.85)
      .append("title").text(d => `p(${d.k}) = ${ST.fmt(d.p, 4)}`);
    // mean marker
    if (S.E >= xlo - 0.5 && S.E <= xhi + 0.5) {
      g.append("line").attr("x1", x(S.E)).attr("x2", x(S.E)).attr("y1", top.y1).attr("y2", top.y0)
        .attr("stroke", SC.good).attr("stroke-dasharray", "4 3");
      g.append("text").attr("x", x(S.E) + 4).attr("y", top.y1 + 10).attr("font-size", 10).attr("fill", SC.good).text(`E X = ${ST.fmt(S.E, 3)}`);
    }
    // cdf staircase
    const steps = []; let c = 0, prevX = xlo - 0.5;
    disp.forEach(d => { steps.push({ x0: prevX, x1: d.k, y: c }); c += d.p; prevX = d.k; });
    steps.push({ x0: prevX, x1: xhi + 0.5, y: Math.min(1, F(xhi)) });
    g.selectAll("line.st").data(steps).join("line").attr("class", "st")
      .attr("x1", d => x(d.x0)).attr("x2", d => x(d.x1)).attr("y1", d => yF(d.y)).attr("y2", d => yF(d.y))
      .attr("stroke", SC.accent).attr("stroke-width", 2);
    let cc = 0;
    disp.forEach(d => {
      g.append("circle").attr("cx", x(d.k)).attr("cy", yF(cc)).attr("r", 2.6).attr("fill", SC.bg).attr("stroke", SC.accent);
      cc += d.p;
      g.append("circle").attr("cx", x(d.k)).attr("cy", yF(Math.min(1, cc))).attr("r", 2.6).attr("fill", SC.accent);
    });
    [[a, Fa, "F(a)"], [b, Fb, "F(b)"]].forEach(([t, val, lab]) => {
      g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", yF(val)).attr("y2", yF(val))
        .attr("stroke", SC.a2).attr("stroke-dasharray", "5 4").attr("opacity", 0.8);
      g.append("text").attr("x", iw - 2).attr("y", yF(val) - 3).attr("text-anchor", "end").attr("font-size", 10)
        .attr("fill", SC.a2).text(`${lab} = ${ST.fmt(val, 4)}`);
    });
    out.innerHTML = `<b>${S.name}</b> · P(${a} &lt; X ≤ ${b}) = F(${b}) − F(${a}) = ${ST.fmt(Fb, 4)} − ${ST.fmt(Fa, 4)} = <b>${ST.fmt(Pab, 4)}</b>` +
      `<br>Σ p(k) = ${ST.fmt(mo.sum, 6)} · mean by summation <b>${ST.fmt(mo.mean, 4)}</b> (closed form ${ST.fmt(S.E, 4)}) · ` +
      `variance by summation <b>${ST.fmt(mo.var, 4)}</b> (closed form ${ST.fmt(S.V, 4)}) · median ${med}`;
  }
  ["pc-fam", ...ctl.map(c => "pc-" + c)].forEach(id => $(id).addEventListener("input", draw));
  fam.addEventListener("change", draw);
  $("pc-reset").addEventListener("click", () => {
    fam.value = defaults.fam;
    ctl.forEach(c => { $("pc-" + c).value = defaults[c]; });
    draw();
  });
  draw();
})();

/* ─────────────────── 3 · sum of indicators → binomial ───────────────────
   Caption audit (seeds 1–60 scanned at n = 20, p = 0.3, 2,000 runs): seed 6 gives an empirical
   mode of 6 = ⌊(n + 1)p⌋, sample mean 6.010 (np = 6) and sample variance 4.211 (np(1 − p) = 4.2);
   seeds 1, 6, 7, 10, 11, 12 all pass; seed 3 fails the mode/mean test.                                                         */
(function () {
  const svg = d3.select("#indic-svg");
  if (svg.empty() || typeof ST === "undefined") return;
  const W = 680, H = 380;
  const $ = id => document.getElementById(id);
  const out = $("indic-readout");
  let n = 20, p = 0.3, seed = RV.seeds.indic, r = null, totals = [], last = [];

  function oneRun() {
    const bits = [];
    for (let i = 0; i < n; i++) bits.push(r() < p ? 1 : 0);
    last = bits; totals.push(d3.sum(bits));
  }
  function restart() { r = ST.rng(seed); totals = []; for (let i = 0; i < 2000; i++) oneRun(); }
  function draw() {
    const F = ST.frame(svg, W, H, { l: 50, r: 18, t: 14, b: 36 });
    const g = F.g, iw = F.iw;
    // strip
    const cw = Math.min(18, iw / n);
    g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 11).attr("fill", SC.muted)
      .text(`latest run (#${totals.length.toLocaleString()}): ${n} Bernoulli(${ST.fmt(p, 2)}) indicators, total = ${totals[totals.length - 1]}`);
    g.selectAll("rect.bit").data(last).join("rect").attr("class", "bit")
      .attr("x", (d, i) => i * cw).attr("y", 16).attr("width", cw - 2).attr("height", 18).attr("rx", 2)
      .attr("fill", d => d ? SC.good : SC.panel2).attr("stroke", SC.line);
    // histogram
    const y0 = 316, y1 = 64;
    const x = d3.scaleBand().domain(d3.range(0, n + 1)).range([0, iw]).padding(0.12);
    const R = totals.length, freq = new Array(n + 1).fill(0);
    totals.forEach(t => freq[t]++);
    const rel = freq.map(f => f / R), ex = d3.range(0, n + 1).map(k => ST.binomPmf(k, n, p));
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(rel), d3.max(ex)) * 1.15]).range([y0, y1]);
    const every = n > 30 ? 5 : n > 15 ? 2 : 1;
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${y0})`)
      .call(d3.axisBottom(x).tickValues(d3.range(0, n + 1, every)));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("x", 0).attr("y", y1 - 8).attr("font-size", 11).attr("fill", SC.muted)
      .text(`relative frequency of the total over ${R.toLocaleString()} runs; dots = exact C(n,k)pᵏ(1−p)ⁿ⁻ᵏ`);
    g.append("text").attr("x", iw).attr("y", y0 + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", SC.muted).text("number of successes k");
    g.selectAll("rect.h").data(rel).join("rect").attr("class", "h")
      .attr("x", (d, k) => x(k)).attr("width", x.bandwidth()).attr("y", d => y(d)).attr("height", d => y0 - y(d))
      .attr("fill", SC.accent).attr("fill-opacity", 0.75);
    g.selectAll("circle.ex").data(ex).join("circle").attr("class", "ex")
      .attr("cx", (d, k) => x(k) + x.bandwidth() / 2).attr("cy", d => y(d)).attr("r", 3.2).attr("fill", SC.ink);
    const mode = Math.min(n, Math.floor((n + 1) * p));
    g.append("line").attr("x1", x(mode) + x.bandwidth() / 2).attr("x2", x(mode) + x.bandwidth() / 2).attr("y1", y1).attr("y2", y0)
      .attr("stroke", SC.a2).attr("stroke-dasharray", "5 4");
    g.append("text").attr("x", x(mode) + x.bandwidth() / 2 + 4).attr("y", y1 + 10).attr("font-size", 10).attr("fill", SC.a2)
      .text(`mode ⌊(n+1)p⌋ = ${mode}`);
    const xm = d3.scaleLinear().domain([-0.5, n + 0.5]).range([0, iw]);
    g.append("line").attr("x1", xm(n * p)).attr("x2", xm(n * p)).attr("y1", y1 + 16).attr("y2", y0)
      .attr("stroke", SC.good).attr("stroke-dasharray", "2 3");
    g.append("text").attr("x", xm(n * p) + 4).attr("y", y1 + 24).attr("font-size", 10).attr("fill", SC.good).text(`np = ${ST.fmt(n * p, 2)}`);

    const sm = ST.mean(totals), sv = R > 1 ? ST.variance(totals, 1) : NaN;
    let em = 0; freq.forEach((f, k) => { if (f > freq[em]) em = k; });
    out.innerHTML = `seed ${seed} · ${R.toLocaleString()} runs of n = ${n}, p = ${ST.fmt(p, 2)} · ` +
      `sample mean <b>${ST.fmt(sm, 3)}</b> (np = ${ST.fmt(n * p, 3)}) · sample variance <b>${R > 1 ? ST.fmt(sv, 3) : "—"}</b> (np(1−p) = ${ST.fmt(n * p * (1 - p), 3)})` +
      `<br>tallest bar at k = <b>${em}</b> · theoretical mode ⌊(n+1)p⌋ = ${mode} · ` +
      `total-variation distance between the histogram and the exact pmf: <b>${ST.fmt(PV.tv(rel, ex), 4)}</b>`;
  }
  function readCtl() {
    n = +$("in-n").value; p = +$("in-p").value;
    $("in-nv").textContent = n; $("in-pv").textContent = ST.fmt(p, 2);
  }
  $("in-n").addEventListener("input", () => { readCtl(); restart(); draw(); });
  $("in-p").addEventListener("input", () => { readCtl(); restart(); draw(); });
  $("in-one").addEventListener("click", () => { oneRun(); draw(); });
  $("in-many").addEventListener("click", () => { for (let i = 0; i < 1000; i++) oneRun(); draw(); });
  $("in-reset").addEventListener("click", () => {
    $("in-n").value = 20; $("in-p").value = 0.3; seed = RV.seeds.indic; readCtl(); restart(); draw();
  });
  readCtl(); restart(); draw();
})();

/* ─────────────────── 4 · memorylessness ───────────────────
   Caption audit: default geometric p = 0.2, k₀ = 5 → P(X > 5) = 0.32768, P(X > 8 | X > 5) = 0.512
   = P(X > 3), and the largest gap between the renormalised shifted tail and the pmf is ~1e-17.   */
(function () {
  const svg = d3.select("#geom-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, H = 340, KMAX = 40;
  const $ = id => document.getElementById(id);
  const out = $("geom-readout");

  function draw() {
    const fam = $("gm-fam").value, p = +$("gm-p").value, k0 = +$("gm-k").value;
    $("gm-pv").textContent = ST.fmt(p, 2); $("gm-kv").textContent = k0;
    const pmf = fam === "geom" ? (k => PV.geomPmf(k, p))
      : fam === "nb" ? (k => PV.negBinPmf(k, 2, p))
        : (k => (k >= 1 && k <= 20 ? 1 / 20 : 0));
    const lo = fam === "nb" ? 2 : 1;
    let tailP = 0; for (let k = k0 + 1; k < 5000; k++) tailP += pmf(k);
    if (fam === "unif") tailP = Math.max(0, 20 - k0) / 20;
    const base = d3.range(1, KMAX + 1).map(k => ({ k: k, p: pmf(k) }));
    const cond = d3.range(1, KMAX + 1).map(t => ({ k: t, p: tailP > 0 ? pmf(k0 + t) / tailP : 0 }));
    let gap = 0; base.forEach((d, i) => { gap = Math.max(gap, Math.abs(d.p - cond[i].p)); });
    const tailAt = t => { let s = 0; for (let k = t + 1; k < 5000; k++) s += pmf(k); return s; };
    const t = 3, condTail = tailP > 0 ? tailAt(k0 + t) / tailP : NaN, freshTail = tailAt(t);
    let Er = 0; cond.forEach(d => { Er += d.k * d.p; });
    let Ex = 0; for (let k = 1; k < 5000; k++) Ex += k * pmf(k);
    if (fam !== "unif") { let s = 0; for (let tt = 1; tt < 5000; tt++) s += tt * (tailP > 0 ? pmf(k0 + tt) / tailP : 0); Er = s; }

    const F = ST.frame(svg, W, H, { l: 50, r: 18, t: 30, b: 38 });
    const g = F.g, iw = F.iw, ih = F.ih;
    const x = d3.scaleBand().domain(d3.range(1, KMAX + 1)).range([0, iw]).padding(0.15);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(base, d => d.p), d3.max(cond, d => d.p)) * 1.12]).range([ih, 0]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).tickValues(d3.range(1, KMAX + 1, 3)));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("x", iw).attr("y", ih + 32).attr("text-anchor", "end").attr("font-size", 11).attr("fill", SC.muted)
      .text("k (bars: trial number) · t (outline: remaining trials after k₀)");
    g.selectAll("rect.b").data(base).join("rect").attr("class", "b")
      .attr("x", d => x(d.k)).attr("width", x.bandwidth()).attr("y", d => y(d.p)).attr("height", d => ih - y(d.p))
      .attr("fill", d => d.k > k0 ? SC.accent : SC.muted).attr("fill-opacity", d => d.k > k0 ? 0.8 : 0.35);
    g.selectAll("rect.c").data(cond).join("rect").attr("class", "c")
      .attr("x", d => x(d.k)).attr("width", x.bandwidth()).attr("y", d => y(d.p)).attr("height", d => ih - y(d.p))
      .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);
    if (k0 > 0 && k0 <= KMAX) {
      const xx = x(k0) + x.bandwidth() + x.step() * 0.07;
      g.append("line").attr("x1", xx).attr("x2", xx).attr("y1", 0).attr("y2", ih).attr("stroke", SC.bad).attr("stroke-dasharray", "4 3");
      g.append("text").attr("x", xx + 4).attr("y", 10).attr("font-size", 10).attr("fill", SC.bad).text(`k₀ = ${k0}: no success yet`);
    }
    ST.legend(g, [{ label: "original pmf p(k)  (grey = ruled out by X > k₀)", color: SC.accent },
      { label: "P(X = k₀ + t | X > k₀), shifted back by k₀", color: SC.a2, dash: "4 0" }], iw - 300, -16, { vertical: false, gap: 0 });
    const nm = fam === "geom" ? `Geo(${ST.fmt(p, 2)})` : fam === "nb" ? `NB(2, ${ST.fmt(p, 2)})` : "uniform on 1…20";
    out.innerHTML = `${nm} · P(X &gt; ${k0}) = <b>${ST.fmt(tailP, 5)}</b> · ` +
      `P(X &gt; ${k0 + t} | X &gt; ${k0}) = <b>${ST.fmt(condTail, 4)}</b> vs P(X &gt; ${t}) = <b>${ST.fmt(freshTail, 4)}</b>` +
      `<br>largest gap between the outline and the bars: <b>${gap < 1e-12 ? gap.toExponential(1) : ST.fmt(gap, 4)}</b> · ` +
      `expected remaining wait ${ST.fmt(Er, 3)} vs fresh expected wait ${ST.fmt(Ex, 3)}` +
      (fam === "geom" ? " — memoryless" : " — the past changes the future: this distribution has memory") +
      (lo > 1 ? "" : "");
  }
  ["gm-fam", "gm-p", "gm-k"].forEach(id => { $(id).addEventListener("input", draw); $(id).addEventListener("change", draw); });
  draw();
})();

/* ─────────────────── 5 · overdispersion ───────────────────
   Caption audit: default μ = 4, r = 2 → NB variance 12, P(0) = 0.111 (the modes are 1 and 2, at 0.148),
   Poisson P(0) = 0.018; 5,000 gamma–Poisson mixture draws at seed 1: mean 3.979, variance 11.87,
   P(0) 0.114, distance 0.015 to the NB pmf against 0.271 to the Poisson pmf.                              */
(function () {
  const svg = d3.select("#nb-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, H = 340, NS = 5000;
  const $ = id => document.getElementById(id);
  const out = $("nb-readout");
  let seed = RV.seeds.nb, sims = [];

  function simulate(mu, r) {
    const rr = ST.rng(seed); sims = [];
    for (let i = 0; i < NS; i++) {
      const lam = RV.gammaDraw(r, rr) * (mu / r);
      sims.push(PV.poisDraw(lam, rr));
    }
  }
  function draw(resim) {
    const mu = +$("nb-m").value, r = +$("nb-r").value;
    $("nb-mv").textContent = ST.fmt(mu, 1); $("nb-rv").textContent = ST.fmt(r, 1);
    if (resim) simulate(mu, r);
    const p = r / (r + mu);
    const kmax = Math.min(60, Math.max(12, Math.ceil(mu + 4 * Math.sqrt(mu + mu * mu / r))));
    const ks = d3.range(0, kmax + 1);
    const nb = ks.map(k => PV.negBinFailPmf(k, r, p)), po = ks.map(k => PV.poisPmf(k, mu));
    const freq = new Array(kmax + 1).fill(0); sims.forEach(s => { if (s <= kmax) freq[s]++; });
    const rel = freq.map(f => f / NS);
    const F = ST.frame(svg, W, H, { l: 50, r: 18, t: 30, b: 38 });
    const g = F.g, iw = F.iw, ih = F.ih;
    const x = d3.scaleBand().domain(ks).range([0, iw]).padding(0.15);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(nb), d3.max(po), d3.max(rel)) * 1.12]).range([ih, 0]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).tickValues(ks.filter(k => k % (kmax > 30 ? 5 : 2) === 0)));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("x", iw).attr("y", ih + 32).attr("text-anchor", "end").attr("font-size", 11).attr("fill", SC.muted).text("count k");
    g.selectAll("rect.nb").data(nb).join("rect").attr("class", "nb")
      .attr("x", (d, k) => x(k)).attr("width", x.bandwidth()).attr("y", d => y(d)).attr("height", d => ih - y(d))
      .attr("fill", SC.accent).attr("fill-opacity", 0.7);
    g.selectAll("rect.sim").data(rel).join("rect").attr("class", "sim")
      .attr("x", (d, k) => x(k) + 1).attr("width", Math.max(1, x.bandwidth() - 2)).attr("y", d => y(d)).attr("height", d => ih - y(d))
      .attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 1.4);
    g.selectAll("circle.po").data(po).join("circle").attr("class", "po")
      .attr("cx", (d, k) => x(k) + x.bandwidth() / 2).attr("cy", d => y(d)).attr("r", 3.6).attr("fill", SC.a2);
    ST.legend(g, [{ label: `negative binomial (μ = ${ST.fmt(mu, 1)}, r = ${ST.fmt(r, 1)})`, color: SC.accent },
      { label: "5,000 gamma–Poisson mixture draws", color: SC.good, dash: "4 0" },
      { label: `Poisson(${ST.fmt(mu, 1)})`, color: SC.a2 }], iw - 250, -14, { vertical: true, gap: 13 });
    const sm = ST.mean(sims), sv = ST.variance(sims, 1);
    const tail = (arr) => 1 - arr.slice(0, 10).reduce((s, v) => s + v, 0);
    let fullNb = [], fullPo = [];
    const KK = Math.max(kmax, 200);
    for (let k = 0; k <= KK; k++) { fullNb.push(PV.negBinFailPmf(k, r, p)); fullPo.push(PV.poisPmf(k, mu)); }
    const fullRel = new Array(KK + 1).fill(0); sims.forEach(s => { if (s <= KK) fullRel[s] += 1 / NS; });
    out.innerHTML = `variance: negative binomial μ + μ²/r = <b>${ST.fmt(mu + mu * mu / r, 2)}</b> · Poisson μ = <b>${ST.fmt(mu, 2)}</b> · ` +
      `simulated mixture (seed ${seed}): mean ${ST.fmt(sm, 3)}, variance <b>${ST.fmt(sv, 2)}</b>` +
      `<br>P(0): NB <b>${ST.fmt(nb[0], 3)}</b> · Poisson ${ST.fmt(po[0], 3)} · simulated ${ST.fmt(rel[0], 3)} &nbsp;|&nbsp; ` +
      `P(X ≥ 10): NB <b>${ST.fmt(tail(fullNb), 3)}</b> · Poisson ${ST.fmt(tail(fullPo), 3)} · simulated ${ST.fmt(tail(fullRel), 3)}` +
      `<br>distance of the simulated histogram to the NB pmf: <b>${ST.fmt(PV.tv(fullRel, fullNb), 3)}</b> · to the Poisson pmf: <b>${ST.fmt(PV.tv(fullRel, fullPo), 3)}</b>`;
  }
  $("nb-m").addEventListener("input", () => draw(true));
  $("nb-r").addEventListener("input", () => draw(true));
  $("nb-sim").addEventListener("click", () => { seed = (seed * 7919 + 13) % 100003; draw(true); });
  $("nb-reset").addEventListener("click", () => { $("nb-m").value = 4; $("nb-r").value = 2; seed = RV.seeds.nb; draw(true); });
  draw(true);
})();

/* ─────────────────── 6 · hypergeometric vs binomial ───────────────────
   Caption audit: default N = 20, K = 6, n = 10 → hypergeometric sd 1.051 vs binomial 1.449,
   TV 0.159, variance ratio (N − n)/(N − 1) = 0.526.                                          */
(function () {
  const svg = d3.select("#hyper-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, H = 360;
  const $ = id => document.getElementById(id);
  const out = $("hyper-readout");
  let rr = ST.rng(RV.seeds.hyper), sample = [], mc = null;

  function params() {
    const N = +$("hy-N").value, p = +$("hy-p").value;
    const n = Math.min(+$("hy-n").value, N), K = Math.round(p * N);
    $("hy-Nv").textContent = N; $("hy-pv").textContent = ST.fmt(p, 2); $("hy-nv").textContent = n;
    return { N, K, n };
  }
  function drawSample(P) { sample = ST.srswor(d3.range(P.N), P.n, rr); }
  function draw() {
    const P = params(), { N, K, n } = P;
    if (sample.length !== n || sample.some(s => s >= N)) drawSample(P);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    // grid
    const box = 250, cols = Math.ceil(Math.sqrt(N)), cs = box / cols;
    const gx = 20, gy = 40, inS = new Set(sample);
    g.append("text").attr("x", gx).attr("y", 22).attr("font-size", 11).attr("fill", SC.muted)
      .text(`batch of N = ${N}: K = ${K} marked · sample of n = ${n} outlined`);
    g.selectAll("rect.it").data(d3.range(N)).join("rect").attr("class", "it")
      .attr("x", i => gx + (i % cols) * cs + 1).attr("y", i => gy + Math.floor(i / cols) * cs + 1)
      .attr("width", cs - 2).attr("height", cs - 2).attr("rx", 2)
      .attr("fill", i => i < K ? SC.a2 : SC.panel2).attr("fill-opacity", i => i < K ? 0.85 : 1)
      .attr("stroke", i => inS.has(i) ? SC.ink : SC.line).attr("stroke-width", i => inS.has(i) ? 2.2 : 1);
    const kS = sample.filter(i => i < K).length;
    // bars
    const px0 = 320, px1 = 660, py0 = 310, py1 = 40;
    const ks = d3.range(0, n + 1);
    const hy = ks.map(k => PV.hyperPmf(k, N, K, n)), bi = ks.map(k => ST.binomPmf(k, n, K / N));
    const x = d3.scaleBand().domain(ks).range([px0, px1]).padding(0.12);
    const half = x.bandwidth() / 2;
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(hy), d3.max(bi)) * 1.15]).range([py0, py1]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${py0})`).call(d3.axisBottom(x).tickValues(ks.filter(k => n <= 12 || k % 2 === 0)));
    g.append("g").attr("class", "axis").attr("transform", `translate(${px0},0)`).call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("x", px1).attr("y", py0 + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", SC.muted).text("marked items in the sample k");
    g.selectAll("rect.hy").data(hy).join("rect").attr("class", "hy")
      .attr("x", (d, k) => x(k)).attr("width", half).attr("y", d => y(d)).attr("height", d => py0 - y(d)).attr("fill", SC.accent).attr("fill-opacity", 0.85);
    g.selectAll("rect.bi").data(bi).join("rect").attr("class", "bi")
      .attr("x", (d, k) => x(k) + half).attr("width", half).attr("y", d => y(d)).attr("height", d => py0 - y(d)).attr("fill", SC.a2).attr("fill-opacity", 0.75);
    if (mc && mc.key === `${N},${K},${n}`) {
      g.selectAll("circle.mc").data(mc.rel).join("circle").attr("class", "mc")
        .attr("cx", (d, k) => x(k) + half / 2).attr("cy", d => y(d)).attr("r", 3).attr("fill", SC.ink);
    }
    g.append("line").attr("x1", x(kS) + half / 2).attr("x2", x(kS) + half / 2).attr("y1", py1).attr("y2", py0)
      .attr("stroke", SC.good).attr("stroke-dasharray", "3 3");
    ST.legend(g, [{ label: "without replacement (hypergeometric)", color: SC.accent },
      { label: `with replacement, Bin(${n}, ${ST.fmt(K / N, 2)})`, color: SC.a2 },
      { label: "this sample's count", color: SC.good, dash: "3 3" }], px0 + 10, py1 - 22, { vertical: false, gap: 14 });
    const q = K / N, vb = n * q * (1 - q), fpc = N > 1 ? (N - n) / (N - 1) : 0;
    let mcTxt = "";
    if (mc && mc.key === `${N},${K},${n}`) {
      let md = 0; mc.rel.forEach((v, k) => { md = Math.max(md, Math.abs(v - hy[k])); });
      mcTxt = `<br>Monte Carlo (5,000 samples without replacement, dots): largest gap to the hypergeometric pmf <b>${ST.fmt(md, 4)}</b>`;
    }
    out.innerHTML = `N = ${N}, K = ${K}, n = ${n} (sampling fraction ${ST.fmt(n / N, 2)}) · this sample contains <b>${kS}</b> marked` +
      `<br>mean ${ST.fmt(n * q, 3)} for both · sd hypergeometric <b>${ST.fmt(Math.sqrt(vb * fpc), 3)}</b> vs binomial <b>${ST.fmt(Math.sqrt(vb), 3)}</b> · ` +
      `variance ratio (N − n)/(N − 1) = <b>${ST.fmt(fpc, 3)}</b> · total-variation distance <b>${ST.fmt(PV.tv(hy, bi), 4)}</b>` + mcTxt;
  }
  ["hy-N", "hy-p", "hy-n"].forEach(id => $(id).addEventListener("input", () => { mc = null; drawSample(params()); draw(); }));
  $("hy-draw").addEventListener("click", () => { drawSample(params()); draw(); });
  $("hy-mc").addEventListener("click", () => {
    const P = params(), r2 = ST.rng(RV.seeds.hyperMc), pop = d3.range(P.N), h = new Array(P.n + 1).fill(0);
    for (let i = 0; i < 5000; i++) h[ST.srswor(pop, P.n, r2).filter(v => v < P.K).length]++;
    mc = { key: `${P.N},${P.K},${P.n}`, rel: h.map(c => c / 5000) };
    draw();
  });
  draw();
})();

/* ─────────────────── 7 · binomial → Poisson ───────────────────
   Caption audit: default μ = 3, n = 10 → P(0) 0.028 vs 0.050, TV 0.0864; n = 1,000 → TV 0.00075;
   n × TV → 0.75 at μ = 3.                                                                       */
(function () {
  const svg = d3.select("#poisconv-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, H = 340;
  const NS = [3, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000];
  const $ = id => document.getElementById(id);
  const out = $("poisconv-readout");

  function draw() {
    const mu = +$("pv-m").value;
    let n = NS[+$("pv-n").value];
    $("pv-mv").textContent = ST.fmt(mu, 1);
    let note = "";
    if (n <= mu) { n = Math.floor(mu) + 1; note = ` (n must exceed μ; using n = ${n})`; }
    $("pv-nv").textContent = n.toLocaleString();
    const p = mu / n, tv = RV.tvBinPois(n, mu), bh = (1 - Math.exp(-mu)) * p;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    // left panel
    const L = { x0: 50, x1: 330, y0: 290, y1: 36 };
    const kmax = Math.ceil(mu + 4 * Math.sqrt(mu) + 3), ks = d3.range(0, kmax + 1);
    const bi = ks.map(k => ST.binomPmf(k, n, p)), po = ks.map(k => PV.poisPmf(k, mu));
    const x = d3.scaleBand().domain(ks).range([L.x0, L.x1]).padding(0.15);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(bi), d3.max(po)) * 1.15]).range([L.y0, L.y1]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${L.y0})`).call(d3.axisBottom(x).tickValues(ks.filter(k => k % (kmax > 14 ? 2 : 1) === 0)));
    g.append("g").attr("class", "axis").attr("transform", `translate(${L.x0},0)`).call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("x", L.x0).attr("y", L.y1 - 14).attr("font-size", 11).attr("fill", SC.muted)
      .text(`bars Bin(${n.toLocaleString()}, ${ST.sig(p, 3)}) · dots Poisson(${ST.fmt(mu, 1)})`);
    g.append("text").attr("x", L.x1).attr("y", L.y0 + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", SC.muted).text("k");
    g.selectAll("rect.b").data(bi).join("rect").attr("class", "b")
      .attr("x", (d, k) => x(k)).attr("width", x.bandwidth()).attr("y", d => y(d)).attr("height", d => L.y0 - y(d)).attr("fill", SC.accent).attr("fill-opacity", 0.75);
    g.selectAll("circle.p").data(po).join("circle").attr("class", "p")
      .attr("cx", (d, k) => x(k) + x.bandwidth() / 2).attr("cy", d => y(d)).attr("r", 3.6).attr("fill", SC.a2);
    // right panel: TV vs n, log-log
    const R = { x0: 410, x1: 660, y0: 290, y1: 36 };
    const grid = [];
    for (let e = Math.log10(Math.max(2, Math.floor(mu) + 1)); e <= 4.0001; e += 0.1) {
      const nn = Math.max(Math.floor(mu) + 1, Math.round(Math.pow(10, e)));
      if (!grid.length || grid[grid.length - 1].n !== nn) grid.push({ n: nn, tv: RV.tvBinPois(nn, mu), bd: (1 - Math.exp(-mu)) * mu / nn });
    }
    const xs = d3.scaleLog().domain([grid[0].n, 10000]).range([R.x0, R.x1]);
    const ys = d3.scaleLog().domain([Math.max(1e-6, d3.min(grid, d => d.tv) * 0.6), Math.max(1, d3.max(grid, d => d.bd))]).range([R.y0, R.y1]).clamp(true);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${R.y0})`).call(d3.axisBottom(xs).ticks(4, "~s"));
    g.append("g").attr("class", "axis").attr("transform", `translate(${R.x0},0)`).call(d3.axisLeft(ys).ticks(4, "~g"));
    g.append("text").attr("x", R.x0).attr("y", R.y1 - 14).attr("font-size", 11).attr("fill", SC.muted).text("total-variation distance (log–log)");
    g.append("text").attr("x", R.x1).attr("y", R.y0 + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", SC.muted).text("number of trials n");
    if ($("pv-bound").checked) {
      g.append("path").datum(grid).attr("d", d3.line().x(d => xs(d.n)).y(d => ys(Math.max(1e-6, d.bd))))
        .attr("fill", "none").attr("stroke", SC.muted).attr("stroke-dasharray", "5 4");
      g.append("text").attr("x", R.x1 - 2).attr("y", ys(grid[grid.length - 1].bd) - 6).attr("text-anchor", "end").attr("font-size", 10).attr("fill", SC.muted).text("bound (1 − e^(−μ))p");
    }
    g.append("path").datum(grid).attr("d", d3.line().x(d => xs(d.n)).y(d => ys(d.tv)))
      .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    g.append("circle").attr("cx", xs(n)).attr("cy", ys(tv)).attr("r", 5).attr("fill", SC.a2);
    const hov = g.append("text").attr("x", R.x0 + 6).attr("y", R.y0 - 8).attr("font-size", 10).attr("fill", SC.ink);
    g.append("rect").attr("x", R.x0).attr("y", R.y1).attr("width", R.x1 - R.x0).attr("height", R.y0 - R.y1)
      .attr("fill", "transparent")
      .on("mousemove", e => {
        const [mx] = d3.pointer(e); const nn = xs.invert(mx);
        const d = grid.reduce((a, b) => Math.abs(Math.log(b.n / nn)) < Math.abs(Math.log(a.n / nn)) ? b : a);
        hov.text(`n = ${d.n.toLocaleString()}: distance ${ST.sig(d.tv, 3)}, n × distance ${ST.fmt(d.n * d.tv, 3)}`);
      })
      .on("mouseleave", () => hov.text(""));
    out.innerHTML = `μ = ${ST.fmt(mu, 1)}, n = ${n.toLocaleString()}${note}, p = ${ST.sig(p, 3)} · ` +
      `P(X = 0): binomial <b>${ST.fmt(bi[0], 3)}</b>, Poisson <b>${ST.fmt(po[0], 3)}</b>` +
      `<br>total-variation distance <b>${ST.sig(tv, 3)}</b> · bound (1 − e^(−μ))p = ${ST.sig(bh, 3)} · older bound np² = ${ST.sig(n * p * p, 3)} · ` +
      `n × distance = ${ST.fmt(n * tv, 3)}`;
  }
  ["pv-m", "pv-n", "pv-bound"].forEach(id => { $(id).addEventListener("input", draw); $(id).addEventListener("change", draw); });
  draw();
})();

/* ─────────────────── 8 · categorical die ───────────────────
   Caption audit (seeds 1–12 scanned): softmax T = 1, 600 rolls at seed 2 → counts 328, 115, 78,
   45, 22, 12 against expected 322.9, 118.8, 72.0, 43.7, 26.5, 16.1; largest deviation 1.03 sd,
   distance 0.021.                                             */
(function () {
  const svg = d3.select("#cat-svg");
  if (svg.empty() || typeof PV === "undefined") return;
  const W = 680, H = 340, LOG = [2, 1, 0.5, 0, -0.5, -1];
  const $ = id => document.getElementById(id);
  const out = $("cat-readout");
  let seed = RV.seeds.cat, r = null, counts = [0, 0, 0, 0, 0, 0], lastFace = null;

  function probs() {
    const d = $("ct-die").value, T = +$("ct-t").value;
    $("ct-tv").textContent = ST.fmt(T, 1);
    if (d === "fair") return [1, 1, 1, 1, 1, 1].map(() => 1 / 6);
    if (d === "loaded") return [0.1, 0.1, 0.1, 0.1, 0.1, 0.5];
    return PV.softmax(LOG, T);
  }
  function roll(k) { const p = probs(); for (let i = 0; i < k; i++) { lastFace = PV.catDraw(p, r); counts[lastFace]++; } }
  function restart() { r = ST.rng(seed); counts = [0, 0, 0, 0, 0, 0]; roll(600); }
  function draw() {
    const p = probs(), n = d3.sum(counts), rel = counts.map(c => (n ? c / n : 0));
    svg.selectAll("*").remove();
    const g = svg.append("g");
    // logits panel (softmax only)
    const soft = $("ct-die").value === "soft";
    const lx0 = 30, lx1 = 200, ly0 = 290, ly1 = 60;
    g.append("text").attr("x", lx0).attr("y", 30).attr("font-size", 11).attr("fill", SC.muted)
      .text(soft ? `logits z, divided by T = ${ST.fmt(+$("ct-t").value, 1)}` : "no logits: probabilities set directly");
    if (soft) {
      const xb = d3.scaleBand().domain(d3.range(6)).range([lx0, lx1]).padding(0.2);
      const yl = d3.scaleLinear().domain([-1.2, 2.2]).range([ly0, ly1]);
      g.append("line").attr("x1", lx0).attr("x2", lx1).attr("y1", yl(0)).attr("y2", yl(0)).attr("stroke", SC.line);
      g.selectAll("rect.lg").data(LOG).join("rect").attr("class", "lg")
        .attr("x", (d, i) => xb(i)).attr("width", xb.bandwidth()).attr("y", d => yl(Math.max(0, d))).attr("height", d => Math.abs(yl(d) - yl(0)))
        .attr("fill", SC.violet).attr("fill-opacity", 0.7);
      g.selectAll("text.lgv").data(LOG).join("text").attr("class", "lgv")
        .attr("x", (d, i) => xb(i) + xb.bandwidth() / 2).attr("y", d => d >= 0 ? yl(d) - 4 : yl(d) + 12)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.muted).text(d => d);
      g.selectAll("text.lgf").data(LOG).join("text").attr("class", "lgf")
        .attr("x", (d, i) => xb(i) + xb.bandwidth() / 2).attr("y", ly0 + 18).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.muted).text((d, i) => i + 1);
      g.append("text").attr("x", (lx0 + lx1) / 2).attr("y", ly0 + 34).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.muted).text("softmax →");
    }
    // frequencies
    const px0 = 270, px1 = 660, py0 = 290, py1 = 40;
    const x = d3.scaleBand().domain(d3.range(6)).range([px0, px1]).padding(0.2);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(p), d3.max(rel)) * 1.15]).range([py0, py1]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${py0})`).call(d3.axisBottom(x).tickFormat(i => `face ${i + 1}`));
    g.append("g").attr("class", "axis").attr("transform", `translate(${px0},0)`).call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("x", px0).attr("y", py1 - 14).attr("font-size", 11).attr("fill", SC.muted)
      .text(`relative frequency after ${n.toLocaleString()} rolls · ticks = probabilities pⱼ`);
    g.selectAll("rect.f").data(rel).join("rect").attr("class", "f")
      .attr("x", (d, i) => x(i)).attr("width", x.bandwidth()).attr("y", d => y(d)).attr("height", d => py0 - y(d))
      .attr("fill", (d, i) => i === lastFace ? SC.good : SC.accent).attr("fill-opacity", 0.8);
    g.selectAll("line.t").data(p).join("line").attr("class", "t")
      .attr("x1", (d, i) => x(i) - 4).attr("x2", (d, i) => x(i) + x.bandwidth() + 4).attr("y1", d => y(d)).attr("y2", d => y(d))
      .attr("stroke", SC.a2).attr("stroke-width", 3);
    g.selectAll("text.pv").data(p).join("text").attr("class", "pv")
      .attr("x", (d, i) => x(i) + x.bandwidth() / 2).attr("y", d => y(d) - 6).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.a2).text(d => ST.fmt(d, 3));
    let zmax = 0;
    counts.forEach((c, j) => { const sd = Math.sqrt(n * p[j] * (1 - p[j])); if (sd > 0) zmax = Math.max(zmax, Math.abs(c - n * p[j]) / sd); });
    const H0 = -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
    out.innerHTML = `seed ${seed} · ${n.toLocaleString()} rolls · counts <b>${counts.join(", ")}</b> · expected ${p.map(v => ST.fmt(n * v, 1)).join(", ")}` +
      `<br>largest deviation from expected: <b>${ST.fmt(zmax, 2)}</b> binomial sd · total-variation distance to p: <b>${ST.fmt(PV.tv(rel, p), 4)}</b>` +
      ` · entropy of p: ${ST.fmt(H0, 3)} nats (max ln 6 = 1.792)` + (lastFace !== null ? ` · last roll: face ${lastFace + 1}` : "");
  }
  $("ct-die").addEventListener("change", () => { restart(); draw(); });
  $("ct-t").addEventListener("input", () => { restart(); draw(); });
  $("ct-one").addEventListener("click", () => { roll(1); draw(); });
  $("ct-many").addEventListener("click", () => { roll(1000); draw(); });
  $("ct-reset").addEventListener("click", () => { $("ct-die").value = "soft"; $("ct-t").value = 1; seed = RV.seeds.cat; restart(); draw(); });
  restart(); draw();
})();
