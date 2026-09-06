/* resampling.viz.js — the twelve visualizations on math/statistics/resampling.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

      1  #mc-svg    Monte-Carlo convergence: several running estimates against B on a
                    log axis, inside the 1/√B envelope the CLT predicts
      2  #plug-svg  F̂ₙ closing on F under a DKW band, and what "draw from F̂ₙ" means —
                    press the button and watch one resample assembled from the strip
      3  #boot-svg  the bootstrap machine, ANIMATED: multiplicities on the data strip,
                    the resample itself, and the θ̂* histogram filling in
      4  #occ-svg   occupancy: the multiplicity distribution against Poisson(1), and
                    the distinct fraction against 1 − (1 − 1/n)ⁿ with its 0.632 limit
      5  #blk-svg   an AR(1) series, the i.i.d. bootstrap, the moving-block bootstrap
                    and the TRUE sampling distribution of the mean, with live coverage
      6  #ci3-svg   normal / percentile / pivotal drawn on one bootstrap histogram;
                    raise the skew and they walk apart in opposite directions
      7  #cov-svg   a live coverage race: the last forty intervals, and four running
                    coverage curves converging on their real (not nominal) levels
      8  #bee-svg   what B buys and what n buys — a narrow Monte-Carlo band collapsing
                    inside a wide statistical band that does not move at all
      9  #perm-svg  the shuffling machine: labels move, values stay, the observed
                    statistic never moves, and the null distribution accumulates
     10  #t1-svg    exchangeability broken on purpose: measured type I error of the
                    permutation test as the group sizes and spreads are unbalanced
     11  #reg-svg   five ways to resample part 6's twelve points, with the fanned
                    refits on the left and the b̂* histogram on the right
     12  #fail-svg  the failure gallery: the true sampling distribution against the
                    bootstrap's copy of it, for statistics that work and that do not

   Every number these draw is recomputed from the data they generate, so the
   pictures and the prose cannot drift apart.                                   */

/* ══════════ page-local helpers (deliberately NOT in stats-viz.js) ══════════ */
const RS = (function () {

  /* ── the three data sets the page keeps returning to ─────────────────── */

  /* §04's running sample: twenty support-ticket resolution times, minutes.
     Σx = 400 exactly, x̄ = 20, Σ(x−x̄)² = 9790, s = 22.699409, σ̂/√n = 4.947221. */
  const RUN = [3, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 15, 18, 22, 27, 34, 45, 62, 91];

  /* §11's two-sample data — part 5's variants A and B, x̄ = 8.24167 / 7.04667,
     s = 1.36012 / 2.11013, so they round to part 5's reported 8.24/1.36 and 7.05/2.11. */
  const GA = [5.8, 6.9, 7.1, 7.2, 7.9, 8.0, 8.1, 9.0, 9.1, 9.8, 9.8, 10.2];
  const GB = [3.9, 4.2, 4.4, 5.2, 5.6, 6.3, 6.6, 6.9, 7.2, 8.4, 8.6, 8.9, 9.7, 9.8, 10.0];

  /* §13's four-against-four set: exhaustive two-sided p = 4/70 = 0.057143 */
  const TA = [13, 17, 22, 26], TB = [6, 8, 11, 16];

  /* part 6's twelve service calls — b = 3, a = 6, S_xx = 318, SSE = 250, s_e = 5 */
  const RUNX = [2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
  const RUNY = [10, 14, 26, 22, 26, 37, 31, 48, 52, 57, 53, 56];

  /* ── numeric helpers ─────────────────────────────────────────────────── */

  /* every quantile on this page uses the linear-interpolation convention, which
     is what the prose says and what numpy/R type 7 do. One place, one choice.  */
  const q = (a, p) => ST.quantile(a, p, "linear");

  /* B bootstrap replicates of stat(sample) — the primitive the whole page rests on */
  function boot(data, B, stat, r) {
    const n = data.length, out = new Array(B), buf = new Array(n);
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < n; i++) buf[i] = data[Math.floor(r() * n)];
      out[b] = stat(buf);
    }
    return out;
  }

  /* the statistics the pickers offer, all written to tolerate an unsorted buffer */
  const STAT = {
    mean: a => ST.mean(a),
    median: a => ST.median(a),
    sd: a => ST.sd(a),
    trim: a => ST.trimmedMean(a, 0.1),
    q90: a => q(a, 0.9),
    max: a => Math.max.apply(null, a),
    min: a => Math.min.apply(null, a)
  };
  const STATLBL = {
    mean: "mean", median: "median", sd: "SD", trim: "10% trimmed mean",
    q90: "90th percentile", max: "maximum", min: "minimum"
  };

  /* one draw from each population the pickers offer, plus its exact parameters */
  function drawOne(pop, r) {
    switch (pop) {
      case "normal": return ST.randn(r);
      case "exp": { let u = r(); if (u <= 0) u = 1e-12; return -Math.log(u); }
      case "lognormal": return Math.exp(ST.randn(r));
      case "unif": return r();
      case "t3": {                          // Student t on 3 df = Z/√(χ²₃/3)
        const z = ST.randn(r);
        const c = ST.randn(r) ** 2 + ST.randn(r) ** 2 + ST.randn(r) ** 2;
        return z / Math.sqrt(c / 3);
      }
      case "bimodal": return (r() < 0.5) ? -1.6 + 0.55 * ST.randn(r) : 1.5 + 0.75 * ST.randn(r);
      default: return ST.randn(r);
    }
  }
  const sampleFrom = (pop, n, r) => { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = drawOne(pop, r); return a; };

  /* exact population values, so a coverage check never compares against a guess */
  const POP = {
    normal: { mean: 0, median: 0, sd: 1, name: "Normal(0, 1)", lo: -3.6, hi: 3.6, cdf: x => ST.normCdf(x) },
    exp: { mean: 1, median: Math.LN2, sd: 1, name: "Exponential(1)", lo: 0, hi: 6, cdf: x => x <= 0 ? 0 : 1 - Math.exp(-x) },
    lognormal: {
      mean: Math.sqrt(Math.E), median: 1, sd: Math.sqrt(Math.E * Math.E - Math.E),
      name: "Lognormal(0, 1)", lo: 0, hi: 9, cdf: x => x <= 0 ? 0 : ST.normCdf(Math.log(x))
    },
    unif: { mean: 0.5, median: 0.5, sd: 1 / Math.sqrt(12), name: "Uniform(0, 1)", lo: 0, hi: 1, cdf: x => ST.clamp(x, 0, 1) },
    t3: { mean: 0, median: 0, sd: Math.sqrt(3), name: "Student t on 3 df", lo: -6, hi: 6, cdf: x => ST.tCdf(x, 3) },
    bimodal: {
      mean: -0.05, median: -0.2, sd: 1.683745, name: "a two-humped mixture", lo: -4, hi: 4,
      cdf: x => 0.5 * ST.normCdf((x + 1.6) / 0.55) + 0.5 * ST.normCdf((x - 1.5) / 0.75)
    }
  };

  /* ── drawing helpers ─────────────────────────────────────────────────── */

  /* bins over [lo, hi] with k cells, returned with .density set (ST.histBins does the work) */
  function bins(vals, lo, hi, k) {
    if (!(hi > lo)) { hi = lo + 1; }
    return ST.histBins(vals, lo, (hi - lo) / k, k);
  }
  /* the same, but the range is chosen from the data with a little air on each side */
  function autoBins(vals, k, pad) {
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const p = (pad === undefined ? 0.06 : pad) * (hi - lo || 1);
    return { lo: lo - p, hi: hi + p, b: bins(vals, lo - p, hi + p, k) };
  }
  /* paint a set of bins as bars */
  function barsOf(g, bs, x, y, ih, col, op) {
    const w = Math.max(1, x(bs[0].x1) - x(bs[0].x0) - 1);
    return g.append("g").selectAll("rect").data(bs).join("rect")
      .attr("x", d => x(d.x0) + 0.5).attr("y", d => y(d.density))
      .attr("width", w).attr("height", d => Math.max(0, ih - y(d.density)))
      .attr("fill", col).attr("fill-opacity", op === undefined ? 0.45 : op);
  }
  /* a vertical rule with an optional label at the top */
  function rule(g, xp, ih, col, lbl, dash, dy) {
    g.append("line").attr("x1", xp).attr("x2", xp).attr("y1", 0).attr("y2", ih)
      .attr("stroke", col).attr("stroke-width", 1.6)
      .attr("stroke-dasharray", dash || null);
    if (lbl) g.append("text").attr("x", xp).attr("y", (dy === undefined ? -3 : dy))
      .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", col).text(lbl);
  }
  /* a titled sub-frame inside a bigger <g> */
  function sub(g, x, y, title) {
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    if (title) gg.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5)
      .attr("fill", SC.ink).attr("font-weight", 600).text(title);
    return gg;
  }
  /* draw a scatter into an existing <g> */
  function pts(g, x, y, sx, sy, opt) {
    const o = Object.assign({ r: 3, fill: SC.accent, op: 0.62, stroke: null }, opt || {});
    const sel = g.append("g").selectAll("circle").data(x.map((v, i) => [v, y[i]])).join("circle")
      .attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", o.r)
      .attr("fill", o.fill).attr("fill-opacity", o.op);
    if (o.stroke) sel.attr("stroke", o.stroke).attr("stroke-width", 0.8);
    return sel;
  }
  /* A straight line y = a + b·x clipped to the plotted RECTANGLE — both domains.
     Clipping in y matters: a steep resampled line would otherwise be drawn over
     the axes and the legend. (Same helper as part 6's RG.lineSeg, copied here on
     purpose rather than promoted into stats-viz.js.)                            */
  function lineSeg(g, a, b, sx, sy, opt) {
    const o = Object.assign({ color: SC.a2, w: 2, dash: null, op: 1 }, opt || {});
    const dx = sx.domain(), dyr = sy.domain();
    const y0 = Math.min(dyr[0], dyr[1]), y1 = Math.max(dyr[0], dyr[1]);
    let xa = dx[0], xb = dx[1];
    if (Math.abs(b) > 1e-12) {
      const u = (y0 - a) / b, v = (y1 - a) / b;
      xa = Math.max(xa, Math.min(u, v));
      xb = Math.min(xb, Math.max(u, v));
    } else if (a < y0 || a > y1) return g.append("line");
    if (!(xb > xa)) return g.append("line");
    const el = g.append("line")
      .attr("x1", sx(xa)).attr("y1", sy(a + b * xa))
      .attr("x2", sx(xb)).attr("y2", sy(a + b * xb))
      .attr("stroke", o.color).attr("stroke-width", o.w).attr("stroke-opacity", o.op);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    return el;
  }

  /* ── inference helpers used by more than one figure ──────────────────── */

  /* the three intervals of §07, from one array of replicates */
  function intervals(reps, thHat, alpha) {
    const se = ST.sd(reps);
    const z = ST.normQuant(1 - alpha / 2);
    const lo = q(reps, alpha / 2), hi = q(reps, 1 - alpha / 2);
    return {
      se: se,
      normal: [thHat - z * se, thHat + z * se],
      pct: [lo, hi],
      piv: [2 * thHat - hi, 2 * thHat - lo]
    };
  }
  /* the simple-linear slope, computed from the sums so a degenerate x cannot NaN */
  function slope(x, y) {
    const n = x.length, xb = ST.mean(x), yb = ST.mean(y);
    let sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - xb; sxx += d * d; sxy += d * (y[i] - yb); }
    return sxx > 0 ? sxy / sxx : NaN;
  }
  /* Pearson r on a pair of buffers */
  const corrOf = (x, y) => ST.corr(x, y);

  /* a Welch t on two arrays */
  function welch(a, b) {
    const va = ST.variance(a), vb = ST.variance(b);
    const se = Math.sqrt(va / a.length + vb / b.length);
    return se > 0 ? (ST.mean(a) - ST.mean(b)) / se : 0;
  }

  /* a compact p-value string */
  const pstr = p => !isFinite(p) ? "—" : (p < 1e-4 ? p.toExponential(2) : ST.fmt(p, 4));

  return {
    RUN, GA, GB, TA, TB, RUNX, RUNY,
    q, boot, STAT, STATLBL, drawOne, sampleFrom, POP,
    bins, autoBins, barsOf, rule, sub, pts, lineSeg,
    intervals, slope, corrOf, welch, pstr
  };
})();

/* ─────────────── 1 · Monte-Carlo convergence and the 1/√B envelope ─────────────── */
(function () {
  const svg = d3.select("#mc-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 430, BMAX = 20000;
  const eT = document.getElementById("mc-target"), eP = document.getElementById("mc-pop");
  const eK = document.getElementById("mc-k"), eKv = document.getElementById("mc-kv");
  const eBand = document.getElementById("mc-band");
  const out = document.getElementById("mc-readout");
  let seed = 20260906;

  /* the true value and the per-draw SD of the summand, for each target × population.
     sigma_h is what sets the width of the envelope; where a closed form is awkward it
     is measured once from a large pilot sample rather than guessed.               */
  function spec(target, pop) {
    const P = RS.POP[pop];
    if (target === "pi") {
      const p = Math.PI / 4;
      return { truth: Math.PI, sig: 4 * Math.sqrt(p * (1 - p)), lab: "π", note: "hit-or-miss: 4 × the fraction of uniform points inside the quarter circle" };
    }
    if (target === "mean") return { truth: P.mean, sig: P.sd, lab: "E[X]", note: "the summand is X itself, so σ_h is the population SD" };
    if (target === "prob") {
      const p = 1 - P.cdf(2);
      return { truth: p, sig: Math.sqrt(p * (1 - p)), lab: "P(X > 2)", note: "the summand is an indicator, so σ_h = √(p(1−p))" };
    }
    /* the SD: σ_h² = (μ₄ − σ⁴)/(4σ²) by the delta method — measured from a pilot */
    const r = ST.rng(4242);
    const N = 60000;
    let s2 = 0, s4 = 0;
    for (let i = 0; i < N; i++) { const d = RS.drawOne(pop, r) - P.mean; s2 += d * d; s4 += d * d * d * d; }
    const m2 = s2 / N, m4 = s4 / N;
    return {
      truth: P.sd, sig: Math.sqrt(Math.max(1e-9, (m4 - m2 * m2) / (4 * m2))), lab: "SD(X)",
      note: pop === "t3"
        ? "⚠ the t on 3 df has an INFINITE fourth moment, so this envelope is not valid — watch the paths ignore it"
        : "σ_h² = (μ₄ − σ⁴)/(4σ²), estimated from a pilot sample"
    };
  }

  function draw() {
    const target = eT.value, pop = eT.value === "pi" ? "unif" : eP.value, K = +eK.value;
    const sp = spec(target, eP.value);
    const r = ST.rng(seed);

    /* the grid of B values we record at — geometric, so the log axis is evenly sampled */
    const grid = [];
    for (let g = 1; g <= BMAX; g = Math.max(g + 1, Math.ceil(g * 1.16))) grid.push(g);
    if (grid[grid.length - 1] !== BMAX) grid.push(BMAX);

    const paths = [], finals = [];
    for (let k = 0; k < K; k++) {
      let s = 0, ss = 0, hit = 0, gi = 0;
      const pathK = [];
      for (let b = 1; b <= BMAX; b++) {
        if (target === "pi") {
          const u = r() * 2 - 1, v = r() * 2 - 1;
          if (u * u + v * v <= 1) hit++;
        } else {
          const x = RS.drawOne(pop, r);
          if (target === "prob") { if (x > 2) hit++; }
          else { s += x; ss += x * x; }
        }
        if (b === grid[gi]) {
          let v;
          if (target === "pi") v = 4 * hit / b;
          else if (target === "prob") v = hit / b;
          else if (target === "mean") v = s / b;
          else v = b > 1 ? Math.sqrt(Math.max(0, (ss - s * s / b) / (b - 1))) : 0;
          pathK.push({ b: b, v: v });
          gi++;
        }
      }
      paths.push(pathK);
      finals.push(pathK[pathK.length - 1].v);
    }

    const f = ST.frame(svg, W, H, { l: 58, r: 18, t: 22, b: 46 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLog().domain([1, BMAX]).range([0, iw]);
    const half = 2.6 * sp.sig;                       // enough room for the early wandering
    const y = d3.scaleLinear().domain([sp.truth - half, sp.truth + half]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);

    if (eBand.checked) {                             // the ±1.96 σ_h/√B envelope
      const band = grid.map(b => ({ b: b, e: 1.959964 * sp.sig / Math.sqrt(b) }));
      g.append("path").attr("fill", SC.accent).attr("fill-opacity", 0.10)
        .attr("d", d3.area().x(d => x(d.b))
          .y0(d => y(ST.clamp(sp.truth - d.e, y.domain()[0], y.domain()[1])))
          .y1(d => y(ST.clamp(sp.truth + d.e, y.domain()[0], y.domain()[1])))(band));
      [-1, 1].forEach(sgn => g.append("path").attr("fill", "none")
        .attr("stroke", SC.accent).attr("stroke-width", 1).attr("stroke-dasharray", "3 3")
        .attr("d", d3.line().x(d => x(d.b))
          .y(d => y(ST.clamp(sp.truth + sgn * d.e, y.domain()[0], y.domain()[1])))(band)));
    }

    const line = d3.line().x(d => x(d.b)).y(d => y(ST.clamp(d.v, y.domain()[0], y.domain()[1])));
    paths.forEach((p, i) => g.append("path").attr("fill", "none")
      .attr("stroke", i === 0 ? SC.a2 : SC.accent).attr("stroke-width", i === 0 ? 1.9 : 1.1)
      .attr("stroke-opacity", i === 0 ? 0.95 : 0.5).attr("d", line(p)));

    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(sp.truth)).attr("y2", y(sp.truth))
      .attr("stroke", SC.good).attr("stroke-width", 2);
    g.append("text").attr("x", iw - 2).attr("y", y(sp.truth) - 6).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.good).text("the true value = " + ST.sig(sp.truth, 6));

    ST.axisB(g, x, ih, 5, "B — number of draws (log scale)", d3.format("~s"));
    ST.axisL(g, y, 5, "running estimate");

    const sdFinal = K > 1 ? ST.sd(finals) : NaN;
    const theory = sp.sig / Math.sqrt(BMAX);
    out.innerHTML =
      `Estimating <b>${sp.lab}</b> from ${eT.value === "pi" ? "uniform points in a square" : RS.POP[pop].name} · `
      + `<span class="keep">σ</span>_h = <b>${ST.fmt(sp.sig, 4)}</b> — ${sp.note}<br>`
      + `Theoretical Monte-Carlo SE at B = ${BMAX}: <span class="keep">σ</span>_h/√B = <b>${ST.sig(theory, 4)}</b>`
      + (K > 1 ? ` &nbsp;·&nbsp; measured spread of the ${K} final values: <b>${ST.sig(sdFinal, 4)}</b> `
        + `(itself uncertain to about ±${ST.pct(1 / Math.sqrt(2 * (K - 1)), 0)}, from only ${K} paths)` : "")
      + `<br>Halving the error needs FOUR times the draws. The band is not a confidence interval for anything `
      + `in your data — it prices only the simulation's own noise, which is §10's controllable half.`;
  }

  eT.onchange = draw; eP.onchange = draw; eBand.onchange = draw;
  eK.oninput = () => { eKv.textContent = eK.value; draw(); };
  document.getElementById("mc-run").onclick = () => { seed = (seed * 31 + 7) % 1000000; draw(); };
  draw();
})();

/* ─────────────── 2 · the plug-in stand-in, and one resample ─────────────── */
(function () {
  const svg = d3.select("#plug-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 450;
  const eN = document.getElementById("pl-n"), eNv = document.getElementById("pl-nv");
  const eP = document.getElementById("pl-pop"), eD = document.getElementById("pl-dkw");
  const out = document.getElementById("plug-readout");
  let seed = 5150, bseed = 0;                       // bseed = 0 means "no resample drawn yet"

  function draw() {
    const n = Math.round(+eN.value), pop = eP.value, P = RS.POP[pop];
    const r = ST.rng(seed);
    const data = RS.sampleFrom(pop, n, r);
    const srt = ST.asc(data);

    const f = ST.frame(svg, W, H, { l: 46, r: 16, t: 26, b: 40 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gap = 40, leftW = Math.round(iw * 0.56), rightW = iw - leftW - gap;

    /* ── left: F̂ₙ against F, inside the DKW band ── */
    const lo = P.lo, hi = P.hi;
    const gl = g.append("g");
    const x = d3.scaleLinear().domain([lo, hi]).range([0, leftW]);
    const y = d3.scaleLinear().domain([0, 1]).range([ih - 130, 0]);
    ST.gridY(gl, y, leftW, 5);

    const eps = ST.dkwEps(n, 0.05);
    const grid = ST.linspace(lo, hi, 220);
    if (eD.checked) {
      gl.append("path").attr("fill", SC.accent).attr("fill-opacity", 0.11)
        .attr("d", d3.area().x(v => x(v))
          .y0(v => y(ST.clamp(P.cdf(v) - eps, 0, 1)))
          .y1(v => y(ST.clamp(P.cdf(v) + eps, 0, 1)))(grid));
    }
    gl.append("path").attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(v => x(v)).y(v => y(P.cdf(v)))(grid));

    const steps = ST.ecdfSteps(srt);
    const pathD = [];
    let prevX = lo, prevF = 0;
    steps.forEach(s => { pathD.push([prevX, prevF], [s.x, prevF], [s.x, s.F]); prevX = s.x; prevF = s.F; });
    pathD.push([prevX, prevF], [hi, prevF]);
    gl.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.7)
      .attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(pathD));

    ST.axisB(gl, x, ih - 130, 5, "x");
    ST.axisL(gl, y, 5, "F(x)");
    gl.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink).text("F̂ₙ (staircase) against F (curve)");
    ST.legend(gl, [
      { label: "the population, F", color: SC.good },
      { label: "the sample, F̂ₙ", color: SC.a2 },
      { label: "DKW 95% band, ±" + ST.fmt(eps, 3), color: SC.accent, op: 0.35 }
    ], 6, ih - 190);

    /* ── the data strip and, if one has been drawn, a resample beneath it ── */
    const gs = gl.append("g").attr("transform", `translate(0,${ih - 106})`);
    gs.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11).attr("fill", SC.muted)
      .text("the sample — each point carries mass 1/n = " + ST.fmt(1 / n, 3));
    gs.append("line").attr("x1", 0).attr("x2", leftW).attr("y1", 14).attr("y2", 14)
      .attr("stroke", SC.line);
    gs.selectAll("circle").data(srt).join("circle")
      .attr("cx", d => x(d)).attr("cy", 14).attr("r", n > 120 ? 1.8 : 3)
      .attr("fill", SC.a2).attr("fill-opacity", 0.7);

    let mult = null, distinct = 0;
    if (bseed) {
      const rb = ST.rng(bseed);
      mult = new Array(n).fill(0);
      for (let i = 0; i < n; i++) mult[Math.floor(rb() * n)]++;
      distinct = mult.reduce((s, m) => s + (m > 0 ? 1 : 0), 0);
      const gb = gl.append("g").attr("transform", `translate(0,${ih - 58})`);
      gb.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11).attr("fill", SC.muted)
        .text("one bootstrap sample — stacked by multiplicity; the faded points were not drawn");
      gb.append("line").attr("x1", 0).attr("x2", leftW).attr("y1", 30).attr("y2", 30)
        .attr("stroke", SC.line);
      srt.forEach((v, i) => {
        const m = mult[i];
        if (m === 0) {
          gb.append("circle").attr("cx", x(v)).attr("cy", 30).attr("r", n > 120 ? 1.8 : 3)
            .attr("fill", SC.muted).attr("fill-opacity", 0.22);
        } else {
          for (let k = 0; k < m; k++)
            gb.append("circle").attr("cx", x(v)).attr("cy", 30 - k * (n > 120 ? 4 : 7))
              .attr("r", n > 120 ? 1.8 : 3).attr("fill", SC.accent).attr("fill-opacity", 0.8);
        }
      });
    }

    /* ── right: the plug-in estimates against the truth ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + gap},0)`);
    const rows = [
      ["mean  T(F) = ∫x dF", P.mean, ST.mean(data)],
      ["median", P.median, ST.median(data)],
      ["SD", P.sd, ST.sd(data)],
      ["P(X > 2)", 1 - P.cdf(2), data.filter(v => v > 2).length / n]
    ];
    gr.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink).text("T(F)  against  T(F̂ₙ)");
    rows.forEach((row, i) => {
      const yy = 16 + i * 46;
      gr.append("text").attr("x", 0).attr("y", yy).attr("font-size", 11).attr("fill", SC.muted).text(row[0]);
      gr.append("text").attr("x", 0).attr("y", yy + 17).attr("font-size", 12.5).attr("fill", SC.good)
        .text("truth " + ST.fmt(row[1], 4));
      gr.append("text").attr("x", 100).attr("y", yy + 17).attr("font-size", 12.5).attr("fill", SC.a2)
        .text("plug-in " + ST.fmt(row[2], 4));
      const err = Math.abs(row[2] - row[1]);
      gr.append("rect").attr("x", 0).attr("y", yy + 24).attr("height", 5)
        .attr("width", Math.min(rightW, 200 * err / (Math.abs(row[1]) + 1)))
        .attr("fill", SC.bad).attr("fill-opacity", 0.55).attr("rx", 2);
    });

    out.innerHTML =
      `n = <b>${n}</b> from ${P.name} · the DKW band says the WHOLE curve is pinned to `
      + `±<b>${ST.fmt(eps, 4)}</b> with 95% confidence — at n = 20 that is ±0.30, at n = 400 it is ±0.068. `
      + `Nothing on this page works better than that band allows.`
      + (bseed
        ? `<br>The resample drew <b>${distinct}</b> of the ${n} distinct points (${ST.pct(distinct / n, 1)}), `
        + `against the expected 1 − (1 − 1/n)ⁿ = <b>${ST.pct(1 - Math.pow(1 - 1 / n, n), 1)}</b>. `
        + `The tallest stack is a point drawn <b>${Math.max.apply(null, mult)}</b> times.`
        : `<br>Press <b>draw a bootstrap sample</b> to see what "simulate from F̂ₙ" actually does to these points.`);
  }

  eN.oninput = () => { eNv.textContent = eN.value; bseed = 0; draw(); };
  eP.onchange = () => { bseed = 0; draw(); };
  eD.onchange = draw;
  document.getElementById("pl-draw").onclick = () => { bseed = (bseed * 17 + 911) % 999983 || 911; draw(); };
  document.getElementById("pl-new").onclick = () => { seed = (seed * 29 + 13) % 999983; bseed = 0; draw(); };
  draw();
})();

/* ─────────────── 3 · the bootstrap machine, animated ─────────────── */
(function () {
  const svg = d3.select("#boot-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 500;
  const eS = document.getElementById("bt-stat"), eD = document.getElementById("bt-data");
  const eSp = document.getElementById("bt-speed"), eSpv = document.getElementById("bt-spv");
  const out = document.getElementById("boot-readout");
  let timer = null, reps = [], mult = null, cur = null, curVal = NaN, seed = 1607, r = ST.rng(seed);

  /* DATA is kept SORTED, so the strip's i-th circle and mult[i] are the same point.
     (RS.RUN is already sorted; the simulated alternatives are sorted on creation.) */
  function dataset() {
    if (eD.value === "run") return RS.RUN.slice();
    const rr = ST.rng(77), a = [];
    for (let i = 0; i < 20; i++)
      a.push(eD.value === "normal"
        ? Math.round((50 + 12 * ST.randn(rr)) * 10) / 10
        : Math.round(100 * rr()));
    return ST.asc(a);
  }
  let DATA = dataset();

  function stepOnce() {
    const n = DATA.length;
    mult = new Array(n).fill(0);
    cur = new Array(n);
    for (let i = 0; i < n; i++) {
      const j = Math.floor(r() * n);
      mult[j]++; cur[i] = DATA[j];
    }
    curVal = RS.STAT[eS.value](cur);
    reps.push(curVal);
  }

  function draw() {
    const n = DATA.length, srt = ST.asc(DATA);
    const th = RS.STAT[eS.value](DATA);
    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 22, b: 40 });
    const g = f.g, iw = f.iw, ih = f.ih;

    const lo = Math.min.apply(null, srt), hi = Math.max.apply(null, srt);
    const pad = 0.05 * (hi - lo);
    const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, iw]);

    /* ── strip 1: the data, with multiplicity bars ── */
    const g1 = RS.sub(g, 0, 14, "the sample — bar height is how many times each point was drawn");
    g1.append("line").attr("x1", 0).attr("x2", iw).attr("y1", 0).attr("y2", 0).attr("stroke", SC.line);
    srt.forEach((v, i) => {
      const m = mult ? mult[i] : 0;   // DATA is sorted, so srt[i] and mult[i] are the same point
      g1.append("circle").attr("cx", x(v)).attr("cy", 0).attr("r", 3.4)
        .attr("fill", m === 0 && mult ? SC.muted : SC.a2)
        .attr("fill-opacity", m === 0 && mult ? 0.25 : 0.85);
      if (mult && m > 0)
        g1.append("rect").attr("x", x(v) - 3).attr("y", 6).attr("width", 6)
          .attr("height", 6 + 11 * (m - 1)).attr("fill", SC.accent).attr("fill-opacity", 0.7).attr("rx", 1.5);
    });
    g1.append("line").attr("x1", x(th)).attr("x2", x(th)).attr("y1", -8).attr("y2", 36)
      .attr("stroke", SC.good).attr("stroke-width", 1.6);
    g1.append("text").attr("x", x(th)).attr("y", -12).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", SC.good).text("θ̂ = " + ST.fmt(th, 3));

    /* ── strip 2: the resample itself ── */
    const g2 = RS.sub(g, 0, 128, "the current bootstrap sample, and its θ̂*");
    g2.append("line").attr("x1", 0).attr("x2", iw).attr("y1", 0).attr("y2", 0).attr("stroke", SC.line);
    if (cur) {
      const jit = ST.jitterY(cur, 31, 8);
      cur.forEach((v, i) => g2.append("circle").attr("cx", x(v)).attr("cy", jit[i])
        .attr("r", 3).attr("fill", SC.accent).attr("fill-opacity", 0.6));
      g2.append("line").attr("x1", x(curVal)).attr("x2", x(curVal)).attr("y1", -12).attr("y2", 12)
        .attr("stroke", SC.violet).attr("stroke-width", 2);
      g2.append("text").attr("x", x(curVal)).attr("y", -16).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", SC.violet).text("θ̂* = " + ST.fmt(curVal, 3));
    }
    ST.axisB(g2, x, 24, 6, "value");

    /* ── panel 3: the accumulating histogram of θ̂* ── */
    const g3 = RS.sub(g, 0, 232, `the ${reps.length} bootstrap ${RS.STATLBL[eS.value]}s so far`);
    const ph = ih - 232 - 4;
    if (reps.length > 1) {
      const a = RS.autoBins(reps.concat([th]), 40, 0.05);
      const y = d3.scaleLinear().domain([0, d3.max(a.b, d => d.density) * 1.14]).range([ph, 0]);
      const x3 = d3.scaleLinear().domain([a.lo, a.hi]).range([0, iw]);
      ST.gridY(g3, y, iw, 4);
      RS.barsOf(g3, a.b, x3, y, ph, SC.accent, 0.45);
      RS.rule(g3, x3(th), ph, SC.good, "θ̂");
      if (cur) RS.rule(g3, x3(curVal), ph, SC.violet, null, "3 3");
      ST.axisB(g3, x3, ph, 6, "θ̂*");
      ST.axisL(g3, y, 4, "density");
    } else {
      g3.append("text").attr("x", iw / 2).attr("y", ph / 2).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", SC.muted).text("press step or run");
    }

    const se = reps.length > 1 ? ST.sd(reps) : NaN;
    /* for the mean, and only the mean, the B = ∞ answer is available in closed form */
    let exact = NaN;
    if (eS.value === "mean") {
      const m = ST.mean(DATA);
      let ss = 0;
      for (const v of DATA) ss += (v - m) * (v - m);
      exact = Math.sqrt(ss / n) / Math.sqrt(n);          // σ̂/√n, NOT s/√n
    }
    out.innerHTML =
      `<b>${reps.length}</b> resamples · θ̂ (${RS.STATLBL[eS.value]}) = <b>${ST.fmt(th, 4)}</b>`
      + (reps.length > 1
        ? ` · bootstrap SE = <b>${ST.fmt(se, 4)}</b> · mean of the θ̂* = ${ST.fmt(ST.mean(reps), 4)} `
        + `(bias estimate ${ST.fmt(ST.mean(reps) - th, 4)}) · 95% percentile interval `
        + `(<b>${ST.fmt(RS.q(reps, 0.025), 3)}</b>, <b>${ST.fmt(RS.q(reps, 0.975), 3)}</b>)`
        : "")
      + (eS.value === "mean" && isFinite(exact)
        ? `<br>For the mean the ideal (B = ∞) answer is known exactly: <span class="keep">σ</span>̂/√n = <b>${ST.fmt(exact, 6)}</b>. `
        + `Everything above should converge to it, and to nothing else — in particular not to s/√n = ${ST.fmt(ST.sd(DATA) / Math.sqrt(n), 6)}.`
        : "")
      + (eS.value === "max"
        ? `<br>⚠ Watch the histogram: it is a set of spikes, and the tallest sits exactly on θ̂ with probability `
        + `1 − (1 − 1/n)ⁿ = <b>${ST.pct(1 - Math.pow(1 - 1 / n, n), 1)}</b>. No resample can ever exceed θ̂, which is why §16's `
        + `percentile interval for a maximum has coverage exactly zero.`
        : "")
      + (eS.value === "median" || eS.value === "q90"
        ? `<br>Note the gaps: a resampled ${RS.STATLBL[eS.value]} can only land on a small set of values built from the order `
        + `statistics, so the histogram is lumpy rather than smooth. §08 shows what that does to the pivotal interval.`
        : "");
  }

  function tick() { stepOnce(); draw(); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } document.getElementById("bt-run").textContent = "run"; }
  function reset() { stop(); reps = []; mult = null; cur = null; seed = (seed * 13 + 5) % 999983; r = ST.rng(seed); DATA = dataset(); draw(); }

  eS.onchange = () => { reps = []; cur = null; mult = null; draw(); };
  eD.onchange = reset;
  eSp.oninput = () => {
    eSpv.textContent = eSp.value + "/s";
    if (timer) { clearInterval(timer); timer = setInterval(tick, 1000 / +eSp.value); }
  };
  document.getElementById("bt-step").onclick = () => { stop(); tick(); };
  document.getElementById("bt-run").onclick = function () {
    if (timer) { stop(); return; }
    this.textContent = "pause";
    timer = setInterval(tick, 1000 / +eSp.value);
  };
  document.getElementById("bt-reset").onclick = reset;
  draw();
})();

/* ─────────────── 4 · occupancy and the 0.632 law ─────────────── */
(function () {
  const svg = d3.select("#occ-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 420;
  const eN = document.getElementById("oc-n"), eNv = document.getElementById("oc-nv");
  const eB = document.getElementById("oc-b"), eBv = document.getElementById("oc-bv");
  const eP = document.getElementById("oc-pois");
  const out = document.getElementById("occ-readout");
  let seed = 909;

  function draw() {
    const n = Math.round(+eN.value), B = Math.round(+eB.value), r = ST.rng(seed);
    const KMAX = 5;                                   // 0, 1, 2, 3, 4, 5+
    const cnt = new Array(KMAX + 1).fill(0);
    let distinctSum = 0;
    const m = new Array(n);
    for (let b = 0; b < B; b++) {
      m.fill(0);
      for (let i = 0; i < n; i++) m[Math.floor(r() * n)]++;
      let d = 0;
      for (let i = 0; i < n; i++) { if (m[i]) d++; cnt[Math.min(m[i], KMAX)]++; }
      distinctSum += d / n;
    }
    const tot = B * n;
    const obs = cnt.map(c => c / tot);
    const exact = [];
    for (let k = 0; k < KMAX; k++) exact.push(ST.binomPmf(k, n, 1 / n));
    exact.push(1 - exact.reduce((s, v) => s + v, 0));
    const pois = [];
    let fact = 1;
    for (let k = 0; k < KMAX; k++) { if (k) fact *= k; pois.push(Math.exp(-1) / fact); }
    pois.push(1 - pois.reduce((s, v) => s + v, 0));

    const f = ST.frame(svg, W, H, { l: 46, r: 16, t: 26, b: 44 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gap = 52, leftW = Math.round(iw * 0.46), rightW = iw - leftW - gap;

    /* ── left: the multiplicity distribution ── */
    const gl = RS.sub(g, 0, 0, "how many times one observation appears");
    const labs = ["0", "1", "2", "3", "4", "5+"];
    const x = d3.scaleBand().domain(labs).range([0, leftW]).padding(0.22);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(obs), d3.max(exact)) * 1.16]).range([ih, 0]);
    ST.gridY(gl, y, leftW, 5);
    gl.append("g").selectAll("rect").data(obs).join("rect")
      .attr("x", (d, i) => x(labs[i])).attr("y", d => y(d))
      .attr("width", x.bandwidth()).attr("height", d => ih - y(d))
      .attr("fill", (d, i) => i === 0 ? SC.bad : SC.accent).attr("fill-opacity", 0.5).attr("rx", 2);
    gl.append("g").selectAll("line").data(exact).join("line")
      .attr("x1", (d, i) => x(labs[i]) - 2).attr("x2", (d, i) => x(labs[i]) + x.bandwidth() + 2)
      .attr("y1", d => y(d)).attr("y2", d => y(d))
      .attr("stroke", SC.a2).attr("stroke-width", 2.2);
    if (eP.checked) gl.append("g").selectAll("circle").data(pois).join("circle")
      .attr("cx", (d, i) => x(labs[i]) + x.bandwidth() / 2).attr("cy", d => y(d))
      .attr("r", 3.4).attr("fill", SC.violet);
    ST.axisB(gl, d3.scaleLinear().domain([0, 1]).range([0, leftW]), ih, 0, "");
    gl.append("g").attr("transform", `translate(0,${ih})`).attr("class", "axis").call(d3.axisBottom(x));
    ST.axisL(gl, y, 5, "probability");
    gl.append("text").attr("x", leftW / 2).attr("y", ih + 34).attr("text-anchor", "middle")
      .attr("font-size", 11).attr("fill", SC.muted).text("multiplicity Mᵢ");
    ST.legend(gl, [
      { label: "simulated", color: SC.accent, op: 0.5 },
      { label: "exact Binomial(n, 1/n)", color: SC.a2, dash: "0" },
      { label: "Poisson(1) limit", color: SC.violet }
    ], leftW - 152, 8);

    /* ── right: the distinct fraction against n ── */
    const gr = RS.sub(g, leftW + gap, 0, "fraction of the data appearing at least once");
    const nn = ST.linspace(2, 220, 200);
    const x2 = d3.scaleLog().domain([2, 220]).range([0, rightW]);
    const y2 = d3.scaleLinear().domain([0.6, 0.78]).range([ih, 0]);
    ST.gridY(gr, y2, rightW, 5);
    gr.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(v => x2(v)).y(v => y2(1 - Math.pow(1 - 1 / v, v)))(nn));
    gr.append("line").attr("x1", 0).attr("x2", rightW)
      .attr("y1", y2(1 - Math.exp(-1))).attr("y2", y2(1 - Math.exp(-1)))
      .attr("stroke", SC.good).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 4");
    gr.append("text").attr("x", rightW - 2).attr("y", y2(1 - Math.exp(-1)) - 6).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.good).text("1 − e⁻¹ = 0.632121");
    const measured = distinctSum / B;
    gr.append("circle").attr("cx", x2(n)).attr("cy", y2(ST.clamp(measured, 0.6, 0.78)))
      .attr("r", 5).attr("fill", SC.accent).attr("stroke", SC.ink).attr("stroke-width", 1);
    ST.axisB(gr, x2, ih, 4, "sample size n", d3.format("~s"));
    ST.axisL(gr, y2, 5, "distinct fraction");

    /* C(2n−1, n) overflows a double above n ≈ 26, so format from its logarithm */
    const lnD = ST.lnChoose(2 * n - 1, n);
    const l10 = lnD / Math.LN10, e10 = Math.floor(l10);
    const dStr = (lnD < 34)
      ? Math.round(Math.exp(lnD)).toLocaleString("en-US")
      : ST.fmt(Math.pow(10, l10 - e10), 2) + " × 10" +
        String(e10).replace(/[0-9]/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]);
    out.innerHTML =
      `n = <b>${n}</b>, ${B} resamples · measured distinct fraction <b>${ST.fmt(measured, 4)}</b> `
      + `against the exact 1 − (1 − 1/n)ⁿ = <b>${ST.fmt(1 - Math.pow(1 - 1 / n, n), 4)}</b> `
      + `and the limit 1 − e⁻¹ = 0.632121.<br>`
      + `An observation is omitted entirely with probability (1 − 1/n)ⁿ = <b>${ST.fmt(Math.pow(1 - 1 / n, n), 4)}</b> — `
      + `that 36.8% is the out-of-bag set a random forest scores each tree on. `
      + `There are C(2n − 1, n) = <b>${dStr}</b> distinct resamples, so exhaustive enumeration is out of reach `
      + `above about n = 10 — the bootstrap is always Monte Carlo, unlike §13's permutation test.`;
  }

  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  eB.oninput = () => { eBv.textContent = eB.value; draw(); };
  eP.onchange = draw;
  document.getElementById("oc-run").onclick = () => { seed = (seed * 41 + 3) % 999983; draw(); };
  draw();
})();

/* ─────────────── 5 · dependence: i.i.d. against moving-block ─────────────── */
(function () {
  const svg = d3.select("#blk-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const ePh = document.getElementById("bk-phi"), ePhv = document.getElementById("bk-phiv");
  const eN = document.getElementById("bk-n"), eNv = document.getElementById("bk-nv");
  const eL = document.getElementById("bk-l"), eLv = document.getElementById("bk-lv");
  const out = document.getElementById("blk-readout");
  let seed = 3141, cov = null;

  /* one stationary AR(1) path of length n, burned in so the start is not special */
  function ar1(n, phi, r) {
    let x = ST.randn(r) / Math.sqrt(Math.max(1e-9, 1 - phi * phi));
    for (let i = 0; i < 200; i++) x = phi * x + ST.randn(r);
    const out2 = new Array(n);
    for (let i = 0; i < n; i++) { x = phi * x + ST.randn(r); out2[i] = x; }
    return out2;
  }
  const bootMeanIID = (a, B, r) => RS.boot(a, B, ST.mean, r);
  function bootMeanBlock(a, B, L, r) {
    const n = a.length, k = Math.ceil(n / L), buf = new Array(k * L), res = new Array(B);
    for (let b = 0; b < B; b++) {
      let p = 0;
      for (let j = 0; j < k; j++) {
        const st = Math.floor(r() * (n - L + 1));
        for (let t = 0; t < L; t++) buf[p++] = a[st + t];
      }
      let s = 0;
      for (let i = 0; i < n; i++) s += buf[i];
      res[b] = s / n;
    }
    return res;
  }

  function runCoverage() {
    const phi = +ePh.value, n = Math.round(+eN.value), L = Math.round(+eL.value);
    const R = 400, B = 250, r = ST.rng(seed + 77);   // kept modest so the button returns in well under a second
    let hi = 0, hb = 0;
    for (let rep = 0; rep < R; rep++) {
      const s = ar1(n, phi, r);
      const qi = bootMeanIID(s, B, r), qb = bootMeanBlock(s, B, L, r);
      if (RS.q(qi, 0.025) <= 0 && 0 <= RS.q(qi, 0.975)) hi++;
      if (RS.q(qb, 0.025) <= 0 && 0 <= RS.q(qb, 0.975)) hb++;
    }
    cov = { R: R, iid: hi / R, blk: hb / R };
  }

  function draw() {
    const phi = +ePh.value, n = Math.round(+eN.value), L = Math.round(+eL.value);
    const r = ST.rng(seed);
    const series = ar1(n, phi, r);
    const B = 3000;
    const mi = bootMeanIID(series, B, r), mb = bootMeanBlock(series, B, L, r);
    /* the TRUE sampling distribution, from fresh series — the thing both are guessing at */
    const truth = [];
    for (let i = 0; i < 3000; i++) truth.push(ST.mean(ar1(n, phi, r)));

    const f = ST.frame(svg, W, H, { l: 50, r: 16, t: 24, b: 42 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const topH = 150;

    /* ── top: the series, with a few blocks highlighted ── */
    const g1 = RS.sub(g, 0, 12, `one AR(1) series, n = ${n}, φ = ${ST.fmt(phi, 2)} — three of the blocks the block bootstrap draws from`);
    const x1 = d3.scaleLinear().domain([0, n - 1]).range([0, iw]);
    const yl = d3.max(series, d => Math.abs(d)) * 1.1;
    const y1 = d3.scaleLinear().domain([-yl, yl]).range([topH, 0]);
    [0.13, 0.44, 0.71].forEach((frac, i) => {
      const st = Math.floor(frac * (n - L));
      g1.append("rect").attr("x", x1(st)).attr("y", 0)
        .attr("width", Math.max(2, x1(st + L - 1) - x1(st))).attr("height", topH)
        .attr("fill", SC.a2).attr("fill-opacity", 0.14);
    });
    ST.gridY(g1, y1, iw, 3);
    g1.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y1(0)).attr("y2", y1(0))
      .attr("stroke", SC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "4 4");
    g1.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.3)
      .attr("d", d3.line().x((d, i) => x1(i)).y(d => y1(d))(series));
    ST.axisB(g1, x1, topH, 6, "time");
    ST.axisL(g1, y1, 3, "x");

    /* ── bottom: three distributions of the sample mean ── */
    const g2 = RS.sub(g, 0, topH + 62, "the sampling distribution of x̄ — the truth, and the two bootstrap guesses at it");
    const ph = ih - topH - 62;
    const all = truth.concat(mi, mb);
    const a = RS.autoBins(all, 46, 0.03);
    const x2 = d3.scaleLinear().domain([a.lo, a.hi]).range([0, iw]);
    const dens = arr => ST.kde(arr, ST.silverman(arr), ST.linspace(a.lo, a.hi, 180));
    const dT = dens(truth), dI = dens(mi.map(v => v - ST.mean(series))), dB = dens(mb.map(v => v - ST.mean(series)));
    const ymax = d3.max([d3.max(dT, d => d.y), d3.max(dI, d => d.y), d3.max(dB, d => d.y)]) * 1.12;
    const y2 = d3.scaleLinear().domain([0, ymax]).range([ph, 0]);
    ST.gridY(g2, y2, iw, 4);
    const ln = d3.line().x(d => x2(d.x)).y(d => y2(d.y));
    g2.append("path").attr("fill", SC.good).attr("fill-opacity", 0.14)
      .attr("d", d3.area().x(d => x2(d.x)).y0(ph).y1(d => y2(d.y))(dT));
    g2.append("path").attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2.4).attr("d", ln(dT));
    g2.append("path").attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 2).attr("d", ln(dI));
    g2.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "5 3").attr("d", ln(dB));
    ST.axisB(g2, x2, ph, 6, "x̄ (bootstrap curves re-centred on 0 for comparison)");
    ST.axisL(g2, y2, 4, "density");
    ST.legend(g2, [
      { label: "the TRUE sampling distribution", color: SC.good },
      { label: "i.i.d. bootstrap", color: SC.bad, dash: "0" },
      { label: "moving block, ℓ = " + L, color: SC.a2, dash: "5 3" }
    ], 6, 10);

    const vif = (1 + phi) / (1 - phi);
    const sdT = ST.sd(truth), sdI = ST.sd(mi), sdB = ST.sd(mb);
    out.innerHTML =
      `<span class="keep">φ</span> = ${ST.fmt(phi, 2)} · n = ${n} · block ℓ = ${L} &nbsp;·&nbsp; `
      + `true SD of x̄ = <b>${ST.fmt(sdT, 4)}</b> &nbsp;·&nbsp; i.i.d. bootstrap <b>${ST.fmt(sdI, 4)}</b> `
      + `(${ST.pct(sdI / sdT - 1, 1)}) &nbsp;·&nbsp; moving block <b>${ST.fmt(sdB, 4)}</b> (${ST.pct(sdB / sdT - 1, 1)})<br>`
      + `The theoretical variance inflation for an AR(1) is (1 + <span class="keep">φ</span>)/(1 − <span class="keep">φ</span>) = <b>${ST.fmt(vif, 3)}</b>, `
      + `so the true SE is √${ST.fmt(vif, 2)} = <b>${ST.fmt(Math.sqrt(vif), 3)}×</b> the i.i.d. formula. The i.i.d. bootstrap reproduces `
      + `the formula, not the truth — it resamples away the very dependence that caused the problem.`
      + (cov
        ? `<br>Coverage over ${cov.R} replications, nominal 95%: i.i.d. bootstrap <b>${ST.pct(cov.iid, 2)}</b>, `
        + `moving block <b>${ST.pct(cov.blk, 2)}</b>.`
        : `<br>Press <b>check coverage</b> to measure what each interval actually delivers.`);
  }

  ePh.oninput = () => { ePhv.textContent = (+ePh.value).toFixed(2); cov = null; draw(); };
  eN.oninput = () => { eNv.textContent = eN.value; cov = null; draw(); };
  eL.oninput = () => { eLv.textContent = eL.value; cov = null; draw(); };
  document.getElementById("bk-cov").onclick = function () {
    this.textContent = "running…";
    setTimeout(() => { runCoverage(); draw(); this.textContent = "check coverage"; }, 20);
  };
  document.getElementById("bk-new").onclick = () => { seed = (seed * 19 + 11) % 999983; cov = null; draw(); };
  draw();
})();

/* ─────────────── 6 · the three intervals on one histogram ─────────────── */
(function () {
  const svg = d3.select("#ci3-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const eS = document.getElementById("c3-stat");
  const eK = document.getElementById("c3-skew"), eKv = document.getElementById("c3-skv");
  const eN = document.getElementById("c3-n"), eNv = document.getElementById("c3-nv");
  const eL = document.getElementById("c3-lvl"), eT = document.getElementById("c3-truth");
  const out = document.getElementById("ci3-readout");
  let seed = 8123;

  /* A one-parameter family that runs continuously from Normal(1,1) to a lognormal:
       X = 1 + (exp(sZ) − 1)/s,  Z ~ Normal(0,1);   s → 0 gives X = 1 + Z exactly.
     Everything the figure needs about it is available in closed form.            */
  const gen = (s, r) => { const z = ST.randn(r); return s < 1e-6 ? 1 + z : 1 + (Math.exp(s * z) - 1) / s; };
  function truthOf(stat, s) {
    if (s < 1e-6) return { mean: 1, median: 1, sd: 1, q90: 1 + ST.normQuant(0.9) }[stat];
    const E = Math.exp(s * s / 2);
    switch (stat) {
      case "mean": return 1 + (E - 1) / s;
      case "median": return 1;                                     // exp(sZ) has median 1
      case "sd": return Math.sqrt(E * E * (Math.exp(s * s) - 1)) / s;
      case "q90": return 1 + (Math.exp(s * ST.normQuant(0.9)) - 1) / s;
      default: return NaN;
    }
  }

  function draw() {
    const stat = eS.value, s = +eK.value, n = Math.round(+eN.value), alpha = 1 - +eL.value;
    const r = ST.rng(seed), B = 4000;
    let reps, thHat, truth, xlab;

    if (stat === "corr") {
      /* bivariate: correlated normals put through the same monotone transform.
         The transform changes Pearson's ρ, so the truth is measured, not assumed. */
      const rho = 0.8, X = [], Y = [];
      const tf = z => s < 1e-6 ? 1 + z : 1 + (Math.exp(s * z) - 1) / s;
      for (let i = 0; i < n; i++) {
        const z1 = ST.randn(r), z2 = rho * z1 + Math.sqrt(1 - rho * rho) * ST.randn(r);
        X.push(tf(z1)); Y.push(tf(z2));
      }
      const ro = ST.rng(97), OX = [], OY = [];
      for (let i = 0; i < 60000; i++) {
        const z1 = ST.randn(ro), z2 = rho * z1 + Math.sqrt(1 - rho * rho) * ST.randn(ro);
        OX.push(tf(z1)); OY.push(tf(z2));
      }
      truth = ST.corr(OX, OY);
      thHat = ST.corr(X, Y);
      reps = new Array(B);
      const bx = new Array(n), by = new Array(n);
      for (let b = 0; b < B; b++) {
        for (let i = 0; i < n; i++) { const j = Math.floor(r() * n); bx[i] = X[j]; by[i] = Y[j]; }
        reps[b] = ST.corr(bx, by);
      }
      xlab = "bootstrapped correlation r*";
    } else {
      const data = [];
      for (let i = 0; i < n; i++) data.push(gen(s, r));
      thHat = RS.STAT[stat](data);
      truth = truthOf(stat, s);
      reps = RS.boot(data, B, RS.STAT[stat], r);
      xlab = "bootstrapped " + RS.STATLBL[stat] + "  θ̂*";
    }

    const iv = RS.intervals(reps, thHat, alpha);
    const f = ST.frame(svg, W, H, { l: 50, r: 18, t: 24, b: 130 });
    const g = f.g, iw = f.iw, ih = f.ih;

    const a = RS.autoBins(reps.concat([thHat, truth]), 44, 0.05);
    const x = d3.scaleLinear().domain([a.lo, a.hi]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, d3.max(a.b, d => d.density) * 1.16]).range([ih, 0]);
    ST.gridY(g, y, iw, 4);
    RS.barsOf(g, a.b, x, y, ih, SC.accent, 0.42);
    RS.rule(g, x(thHat), ih, SC.a2, "θ̂ = " + ST.fmt(thHat, 3));
    if (eT.checked && isFinite(truth)) RS.rule(g, x(truth), ih, SC.good, "true " + ST.fmt(truth, 3), "4 4", -18);
    ST.axisB(g, x, ih, 6, xlab);
    ST.axisL(g, y, 4, "density");

    /* the three intervals, drawn on the same horizontal scale beneath the histogram */
    const bars = [
      ["normal / SE", iv.normal, SC.violet],
      ["percentile", iv.pct, SC.good],
      ["pivotal (basic)", iv.piv, SC.bad]
    ];
    bars.forEach((bb, i) => {
      const yy = ih + 44 + i * 26;
      const lo = ST.clamp(x(bb[1][0]), -20, iw + 20), hi2 = ST.clamp(x(bb[1][1]), -20, iw + 20);
      g.append("text").attr("x", -6).attr("y", yy + 4).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.muted).text(bb[0]);
      g.append("line").attr("x1", lo).attr("x2", hi2).attr("y1", yy).attr("y2", yy)
        .attr("stroke", bb[2]).attr("stroke-width", 4).attr("stroke-linecap", "round");
      [lo, hi2].forEach(px => g.append("line").attr("x1", px).attr("x2", px)
        .attr("y1", yy - 6).attr("y2", yy + 6).attr("stroke", bb[2]).attr("stroke-width", 2));
      const covers = isFinite(truth) && bb[1][0] <= truth && truth <= bb[1][1];
      g.append("text").attr("x", iw + 4).attr("y", yy + 4).attr("font-size", 11)
        .attr("fill", covers ? SC.good : SC.bad).text(covers ? "✓" : "✗");
    });
    if (eT.checked && isFinite(truth))
      g.append("line").attr("x1", x(truth)).attr("x2", x(truth)).attr("y1", ih + 30).attr("y2", ih + 108)
        .attr("stroke", SC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");

    const wid = v => ST.fmt(v[1] - v[0], 4);
    out.innerHTML =
      `n = ${n} · skew parameter s = ${ST.fmt(s, 2)} · ${(100 * +eL.value).toFixed(0)}% level · B = ${B}<br>`
      + `normal &nbsp;(<b>${ST.fmt(iv.normal[0], 3)}</b>, <b>${ST.fmt(iv.normal[1], 3)}</b>) width ${wid(iv.normal)} &nbsp;·&nbsp; `
      + `percentile (<b>${ST.fmt(iv.pct[0], 3)}</b>, <b>${ST.fmt(iv.pct[1], 3)}</b>) width ${wid(iv.pct)} &nbsp;·&nbsp; `
      + `pivotal (<b>${ST.fmt(iv.piv[0], 3)}</b>, <b>${ST.fmt(iv.piv[1], 3)}</b>) width ${wid(iv.piv)}<br>`
      + `The percentile and pivotal intervals always have the SAME width — they are mirror images of each other `
      + `about θ̂ = ${ST.fmt(thHat, 3)}. At s = 0 they sit on top of one another; raise the skew and they walk apart in `
      + `opposite directions.`
      + (stat === "corr" && iv.piv[1] > 1
        ? `<br>⚠ The pivotal upper endpoint is <b>${ST.fmt(iv.piv[1], 4)}</b> — above 1, which no correlation can be. `
        + `The percentile interval cannot do this, because every endpoint it reports is an actual resampled value.`
        : "")
      + (stat === "median" || stat === "q90"
        ? `<br>⚠ The histogram is a picket fence: a resampled ${RS.STATLBL[stat]} can only take values built from the order `
        + `statistics. Its asymmetry about θ̂ is an artefact of THIS sample's spacing, and reflecting it — which is what `
        + `the pivotal interval does — doubles the artefact instead of cancelling it. §08 measures the cost at up to 18 `
        + `percentage points of coverage.`
        : "");
  }

  eS.onchange = draw; eL.onchange = draw; eT.onchange = draw;
  eK.oninput = () => { eKv.textContent = (+eK.value).toFixed(2); draw(); };
  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  document.getElementById("c3-new").onclick = () => { seed = (seed * 37 + 19) % 999983; draw(); };
  draw();
})();

/* ─────────────── 7 · the live coverage race ─────────────── */
(function () {
  const svg = d3.select("#cov-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 480;
  const eP = document.getElementById("cv-pop"), eS = document.getElementById("cv-stat");
  const eN = document.getElementById("cv-n"), eNv = document.getElementById("cv-nv");
  const eB = document.getElementById("cv-b"), eBv = document.getElementById("cv-bv");
  const out = document.getElementById("cov-readout");
  const METH = ["t", "normal", "pct", "piv"];
  const MLBL = { t: "t interval (formula)", normal: "bootstrap normal/SE", pct: "bootstrap percentile", piv: "bootstrap pivotal" };
  const MCOL = { t: SC.a2, normal: SC.violet, pct: SC.good, piv: SC.bad };
  let r = ST.rng(2718), hits = {}, N = 0, hist = [], recent = [];

  function reset() {
    r = ST.rng(2718 + N);
    hits = { t: 0, normal: 0, pct: 0, piv: 0 };
    N = 0; hist = []; recent = [];
  }
  reset();

  const truthOf = () => RS.POP[eP.value][eS.value];

  function replicate() {
    const n = Math.round(+eN.value), B = Math.round(+eB.value), stat = eS.value, truth = truthOf();
    const data = RS.sampleFrom(eP.value, n, r);
    const thHat = RS.STAT[stat](data);
    const reps = RS.boot(data, B, RS.STAT[stat], r);
    const iv = RS.intervals(reps, thHat, 0.05);
    /* the formula interval only exists for the mean; for the others it is not drawn */
    const tI = (stat === "mean")
      ? (function () { const se = ST.sd(data) / Math.sqrt(n), tq = ST.tQuant(0.975, n - 1); return [thHat - tq * se, thHat + tq * se]; })()
      : null;
    const row = { th: thHat, normal: iv.normal, pct: iv.pct, piv: iv.piv, t: tI };
    METH.forEach(m => { if (row[m] && row[m][0] <= truth && truth <= row[m][1]) hits[m]++; });
    N++;
    recent.push(row);
    if (recent.length > 40) recent.shift();
    if (N % 10 === 0 || N < 40) {
      const pt = { n: N };
      METH.forEach(m => { pt[m] = hits[m] / N; });
      hist.push(pt);
    }
  }
  function runMany(k) { for (let i = 0; i < k; i++) replicate(); draw(); }

  function draw() {
    const truth = truthOf(), stat = eS.value;
    const shown = (stat === "mean") ? METH : METH.slice(1);
    const f = ST.frame(svg, W, H, { l: 54, r: 16, t: 24, b: 44 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const topH = 190;

    /* ── top: the last forty intervals ── */
    const g1 = RS.sub(g, 0, 14, "the last " + recent.length + " percentile intervals — red ones miss the true value");
    if (recent.length) {
      let lo = truth, hi = truth;
      recent.forEach(rw => { lo = Math.min(lo, rw.pct[0]); hi = Math.max(hi, rw.pct[1]); });
      const pad = 0.06 * (hi - lo || 1);
      const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, iw]);
      const yy = d3.scaleLinear().domain([0, 40]).range([0, topH]);
      recent.forEach((rw, i) => {
        const ok = rw.pct[0] <= truth && truth <= rw.pct[1];
        g1.append("line").attr("x1", x(rw.pct[0])).attr("x2", x(rw.pct[1]))
          .attr("y1", yy(i)).attr("y2", yy(i))
          .attr("stroke", ok ? SC.accent : SC.bad).attr("stroke-width", 2).attr("stroke-opacity", ok ? 0.55 : 0.95);
        g1.append("circle").attr("cx", x(rw.th)).attr("cy", yy(i)).attr("r", 1.6).attr("fill", SC.ink);
      });
      g1.append("line").attr("x1", x(truth)).attr("x2", x(truth)).attr("y1", 0).attr("y2", topH)
        .attr("stroke", SC.good).attr("stroke-width", 1.8);
      g1.append("text").attr("x", x(truth)).attr("y", -2).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", SC.good).text("true " + RS.STATLBL[stat] + " = " + ST.fmt(truth, 4));
      ST.axisB(g1, x, topH, 6, "");
    } else {
      g1.append("text").attr("x", iw / 2).attr("y", topH / 2).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", SC.muted).text("press run");
    }

    /* ── bottom: running coverage ── */
    const g2 = RS.sub(g, 0, topH + 62, "running coverage — where each recipe actually settles");
    const ph = ih - topH - 62;
    const x2 = d3.scaleLinear().domain([0, Math.max(50, N)]).range([0, iw]);
    const y2 = d3.scaleLinear().domain([0.6, 1]).range([ph, 0]);
    ST.gridY(g2, y2, iw, 5);
    /* the Monte-Carlo band around 95%, so the reader knows what counts as a difference */
    if (N > 10) {
      const e = 1.959964 * Math.sqrt(0.95 * 0.05 / N);
      g2.append("rect").attr("x", 0).attr("y", y2(0.95 + e)).attr("width", iw)
        .attr("height", Math.max(1, y2(0.95 - e) - y2(0.95 + e)))
        .attr("fill", SC.good).attr("fill-opacity", 0.12);
    }
    g2.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y2(0.95)).attr("y2", y2(0.95))
      .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 4");
    shown.forEach(m => {
      const pts2 = hist.filter(h => isFinite(h[m]));
      if (pts2.length > 1) g2.append("path").attr("fill", "none").attr("stroke", MCOL[m])
        .attr("stroke-width", 2).attr("d", d3.line().x(d => x2(d.n)).y(d => y2(ST.clamp(d[m], 0.6, 1)))(pts2));
    });
    ST.axisB(g2, x2, ph, 6, "replications");
    ST.axisL(g2, y2, 5, "coverage", d3.format(".0%"));
    ST.legend(g2, shown.map(m => ({ label: MLBL[m], color: MCOL[m], dash: "0" })), 8, 12);

    const mc = N > 0 ? 1.959964 * Math.sqrt(0.95 * 0.05 / N) : NaN;
    out.innerHTML =
      `${RS.POP[eP.value].name} · parameter = the ${RS.STATLBL[stat]} = ${ST.fmt(truth, 5)} · n = ${eN.value} · B = ${eB.value} · `
      + `<b>${N}</b> replications (Monte-Carlo error ±${N ? ST.pct(mc, 2) : "—"})<br>`
      + (N ? shown.map(m => `${MLBL[m]}: <b>${ST.pct(hits[m] / N, 2)}</b>`).join(" &nbsp;·&nbsp; ") : "nothing run yet")
      + (stat !== "mean" ? `<br>No formula interval exists for the ${RS.STATLBL[stat]}, which is the whole reason to resample it. ` : "")
      + (stat === "median"
        ? `Watch the pivotal curve settle well below the other two — §08's 75–88% result.`
        : (stat === "mean" && +eN.value <= 12
          ? `At this n the t interval should hold 95% while all three bootstrap intervals fall short — the bootstrap has no small-sample correction.`
          : ""));
  }

  [eP, eS].forEach(el => el.onchange = () => { reset(); draw(); });
  eN.oninput = () => { eNv.textContent = eN.value; reset(); draw(); };
  eB.oninput = () => { eBv.textContent = eB.value; reset(); draw(); };
  document.getElementById("cv-go").onclick = () => runMany(200);
  document.getElementById("cv-go2").onclick = function () {
    this.textContent = "running…";
    setTimeout(() => { runMany(2000); this.textContent = "run 2000 more"; }, 20);
  };
  document.getElementById("cv-reset").onclick = () => { reset(); draw(); };
  runMany(200);
})();

/* ─────────────── 8 · what B buys and what n buys ─────────────── */
(function () {
  const svg = d3.select("#bee-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 450;
  const eN = document.getElementById("be-n"), eNv = document.getElementById("be-nv");
  const eW = document.getElementById("be-what");
  const eR = document.getElementById("be-r"), eRv = document.getElementById("be-rv");
  const out = document.getElementById("bee-readout");
  const GRID = [25, 50, 100, 200, 400, 800, 1600, 3200];
  let seed = 606;

  /* the summary of one bootstrap: the SE, or one of the two interval endpoints */
  function summary(reps, what) {
    if (what === "se") return ST.sd(reps);
    return RS.q(reps, what === "lo" ? 0.025 : 0.975);
  }

  function draw() {
    const n = Math.round(+eN.value), what = eW.value, R = Math.round(+eR.value);
    const r = ST.rng(seed);
    /* ONE fixed data set — a lognormal sample, so the picture matches §10's table */
    const fixed = [];
    for (let i = 0; i < n; i++) fixed.push(Math.exp(2.3 + 0.95 * ST.randn(r)));

    /* COMMON RANDOM NUMBERS, and prefixes rather than fresh runs at each B.
       Each repeat draws BMAX replicates once and is summarised from its first B of
       them, so a curve is a single bootstrap SETTLING DOWN as B grows rather than
       an independent experiment at every grid point. Without this the outer band
       is so noisy at R = 80 that it can appear to WIDEN with B — which is the
       opposite of what the figure is about.                                      */
    const BMAX = GRID[GRID.length - 1];
    const innerAll = [], outerAll = [];
    for (let k = 0; k < R; k++) {
      const repsF = RS.boot(fixed, BMAX, ST.mean, r);
      const fresh = [];
      for (let i = 0; i < n; i++) fresh.push(Math.exp(2.3 + 0.95 * ST.randn(r)));
      const repsN = RS.boot(fresh, BMAX, ST.mean, r);
      innerAll.push(GRID.map(B => summary(repsF.slice(0, B), what)));
      outerAll.push(GRID.map(B => summary(repsN.slice(0, B), what)));
    }
    const inner = [], outer = [];
    GRID.forEach((B, j) => {
      const a = innerAll.map(v => v[j]), b = outerAll.map(v => v[j]);
      inner.push({ B: B, lo: RS.q(a, 0.05), hi: RS.q(a, 0.95), m: ST.mean(a), sd: ST.sd(a) });
      outer.push({ B: B, lo: RS.q(b, 0.05), hi: RS.q(b, 0.95), m: ST.mean(b), sd: ST.sd(b) });
    });

    const f = ST.frame(svg, W, H, { l: 56, r: 130, t: 26, b: 46 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLog().domain([20, 4000]).range([0, iw]);
    const all = outer.concat(inner);
    const ylo = d3.min(all, d => d.lo), yhi = d3.max(all, d => d.hi);
    const pad = 0.08 * (yhi - ylo);
    const y = d3.scaleLinear().domain([ylo - pad, yhi + pad]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);

    const area = d3.area().x(d => x(d.B)).y0(d => y(d.lo)).y1(d => y(d.hi));
    g.append("path").attr("fill", SC.bad).attr("fill-opacity", 0.16).attr("d", area(outer));
    g.append("path").attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 1.4)
      .attr("d", d3.line().x(d => x(d.B)).y(d => y(d.m))(outer));
    g.append("path").attr("fill", SC.accent).attr("fill-opacity", 0.34).attr("d", area(inner));
    g.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.8)
      .attr("d", d3.line().x(d => x(d.B)).y(d => y(d.m))(inner));

    /* the ideal (B = ∞) answer for the fixed data set — available only for the SE */
    if (what === "se") {
      const m = ST.mean(fixed);
      let ss = 0;
      for (const v of fixed) ss += (v - m) * (v - m);
      const ideal = Math.sqrt(ss / n) / Math.sqrt(n);
      g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(ideal)).attr("y2", y(ideal))
        .attr("stroke", SC.good).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 4");
      g.append("text").attr("x", 4).attr("y", y(ideal) - 6).attr("font-size", 10.5).attr("fill", SC.good)
        .text("the ideal (B = ∞) answer for THIS data set: " + ST.fmt(ideal, 4));
    }

    ST.axisB(g, x, ih, 5, "B — resamples (log scale)", d3.format("~s"));
    ST.axisL(g, y, 5, what === "se" ? "bootstrap SE of x̄" : "interval endpoint");
    ST.legend(g, [
      { label: "one FIXED data set,", color: SC.accent, op: 0.45 },
      { label: "  reboostrapped — MC error", color: SC.accent, op: 0.0 },
      { label: "a FRESH data set each time —", color: SC.bad, op: 0.25 },
      { label: "  MC + statistical error", color: SC.bad, op: 0.0 }
    ], iw + 10, 20);

    /* the stacked bar: how the total splits at the largest B on the grid */
    const last = GRID.length - 1;
    const mcSd = inner[last].sd, totSd = outer[last].sd;
    const statSd = Math.sqrt(Math.max(0, totSd * totSd - mcSd * mcSd));
    const bx = iw + 26, by = 150, bh = 150, bw = 30;
    g.append("text").attr("x", bx - 6).attr("y", by - 10).attr("font-size", 10.5).attr("fill", SC.muted)
      .text("at B = " + GRID[last]);
    const frac = mcSd * mcSd / Math.max(1e-12, totSd * totSd);
    g.append("rect").attr("x", bx).attr("y", by).attr("width", bw).attr("height", bh * frac)
      .attr("fill", SC.accent).attr("fill-opacity", 0.75);
    g.append("rect").attr("x", bx).attr("y", by + bh * frac).attr("width", bw).attr("height", bh * (1 - frac))
      .attr("fill", SC.bad).attr("fill-opacity", 0.5);
    g.append("text").attr("x", bx + bw + 5).attr("y", by + 10).attr("font-size", 10).attr("fill", SC.accent)
      .text(ST.pct(frac, 1));
    g.append("text").attr("x", bx + bw + 5).attr("y", by + bh - 4).attr("font-size", 10).attr("fill", SC.bad)
      .text(ST.pct(1 - frac, 1));
    g.append("text").attr("x", bx - 6).attr("y", by + bh + 16).attr("font-size", 10).attr("fill", SC.muted)
      .text("of the VARIANCE");

    const i0 = inner[0], iL = inner[last], o0 = outer[0], oL = outer[last];
    out.innerHTML =
      `n = <b>${n}</b>, ${R} repeats at each B, tracking the <b>${what === "se" ? "standard error" : (what === "lo" ? "2.5%" : "97.5%") + " endpoint"}</b>.<br>`
      + `Blue — one fixed sample, bootstrapped again and again: spread <b>${ST.fmt(i0.sd, 4)}</b> at B = ${GRID[0]} `
      + `falling to <b>${ST.fmt(iL.sd, 4)}</b> at B = ${GRID[last]}, a factor of ${ST.fmt(i0.sd / Math.max(1e-9, iL.sd), 1)}× `
      + `against the ${ST.fmt(Math.sqrt(GRID[last] / GRID[0]), 1)}× that 1/√B predicts.<br>`
      + `Red — a fresh sample every time: spread <b>${ST.fmt(o0.sd, 4)}</b> at B = ${GRID[0]}, `
      + `<b>${ST.fmt(oL.sd, 4)}</b> at B = ${GRID[last]}. A ${Math.round(GRID[last] / GRID[0])}-fold increase in computation `
      + `removed only <b>${ST.pct(Math.max(0, 1 - oL.sd / Math.max(1e-9, o0.sd)), 1)}</b> of it, and the remaining `
      + `<b>${ST.fmt(oL.sd, 4)}</b> is fixed by n = ${n} — no value of B touches it. `
      + `Now drag the <b>n</b> slider and watch the red band move where the B axis could not.`;
  }

  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  eR.oninput = () => { eRv.textContent = eR.value; draw(); };
  eW.onchange = draw;
  document.getElementById("be-run").onclick = () => { seed = (seed * 23 + 5) % 999983; draw(); };
  draw();
})();

/* ─────────────── 9 · the shuffling machine ─────────────── */
(function () {
  const svg = d3.select("#perm-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 510;
  const eS = document.getElementById("pm-stat"), eD = document.getElementById("pm-data");
  const eSide = document.getElementById("pm-side");
  const eSp = document.getElementById("pm-speed"), eSpv = document.getElementById("pm-spv");
  const out = document.getElementById("perm-readout");
  let timer = null, reps = [], labels = null, seed = 4242, r = ST.rng(seed);

  function dataset() {
    if (eD.value === "run") return { A: RS.GA.slice(), B: RS.GB.slice(), name: "part 5's variants A and B" };
    if (eD.value === "tiny") return { A: RS.TA.slice(), B: RS.TB.slice(), name: "the four-against-four set of §13" };
    const rr = ST.rng(eD.value === "shift" ? 11 : 12);
    const shift = eD.value === "shift" ? 1.1 : 0;
    const A = [], B = [];
    for (let i = 0; i < 18; i++) A.push(Math.round((10 + shift + 1.4 * ST.randn(rr)) * 100) / 100);
    for (let i = 0; i < 18; i++) B.push(Math.round((10 + 1.4 * ST.randn(rr)) * 100) / 100);
    return { A: A, B: B, name: eD.value === "shift" ? "a simulated shift of 1.1" : "a simulated null" };
  }
  let D = dataset();

  /* the statistic, computed from a labelling of the pooled values */
  function statOf(pool, lab, n1) {
    const A = [], B = [];
    for (let i = 0; i < pool.length; i++) (lab[i] ? A : B).push(pool[i]);
    switch (eS.value) {
      case "diff": return ST.mean(A) - ST.mean(B);
      case "med": return ST.median(A) - ST.median(B);
      case "t": return RS.welch(A, B);
      case "trim": return ST.trimmedMean(A, 0.1) - ST.trimmedMean(B, 0.1);
      case "ranksum": {
        const rk = ST.ranks(pool);
        let s = 0;
        for (let i = 0; i < pool.length; i++) if (lab[i]) s += rk[i];
        return s;
      }
      default: return ST.mean(A) - ST.mean(B);
    }
  }
  const SLBL = {
    diff: "x̄_A − x̄_B", med: "median_A − median_B", t: "Welch t",
    trim: "trimmed x̄_A − trimmed x̄_B", ranksum: "rank sum of group A"
  };

  function stepOnce() {
    const n = D.A.length + D.B.length, n1 = D.A.length;
    const idx = ST.shuffle(d3.range(n), r);
    labels = new Array(n).fill(0);
    for (let i = 0; i < n1; i++) labels[idx[i]] = 1;
    const pool = D.A.concat(D.B);
    reps.push(statOf(pool, labels, n1));
  }

  function draw() {
    const pool = D.A.concat(D.B), n = pool.length, n1 = D.A.length;
    const obsLab = new Array(n).fill(0);
    for (let i = 0; i < n1; i++) obsLab[i] = 1;
    const obs = statOf(pool, obsLab, n1);
    const lab = labels || obsLab;

    const f = ST.frame(svg, W, H, { l: 50, r: 16, t: 24, b: 42 });
    const g = f.g, iw = f.iw, ih = f.ih;

    const lo = d3.min(pool), hi = d3.max(pool), pad = 0.06 * (hi - lo);
    const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, iw]);

    /* ── strip: the values, coloured by the CURRENT labelling ── */
    const g1 = RS.sub(g, 0, 20, labels ? "one shuffled labelling — the dots have not moved, only their colours" : "the observed labelling");
    g1.append("line").attr("x1", 0).attr("x2", iw).attr("y1", 26).attr("y2", 26).attr("stroke", SC.line);
    const jit = ST.jitterY(pool, 13, 14);
    pool.forEach((v, i) => g1.append("circle").attr("cx", x(v)).attr("cy", 26 + jit[i])
      .attr("r", 4).attr("fill", lab[i] ? SC.accent : SC.a2).attr("fill-opacity", 0.8));
    const cA = [], cB = [];
    pool.forEach((v, i) => (lab[i] ? cA : cB).push(v));
    [[ST.mean(cA), SC.accent, "A*"], [ST.mean(cB), SC.a2, "B*"]].forEach(m => {
      g1.append("path").attr("d", `M ${x(m[0])} 52 l -6 12 l 12 0 z`).attr("fill", m[1]);
      g1.append("text").attr("x", x(m[0])).attr("y", 78).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", m[1]).text(labels ? m[2] : m[2].replace("*", ""));
    });
    ST.axisB(g1, x, 92, 6, "value");
    ST.legend(g1, [
      { label: "group A (n₁ = " + n1 + ")", color: SC.accent },
      { label: "group B (n₂ = " + (n - n1) + ")", color: SC.a2 }
    ], iw - 130, 0);

    /* ── the null distribution ── */
    const g2 = RS.sub(g, 0, 190, `the null distribution: ${reps.length} shuffles of the statistic ${SLBL[eS.value]}`);
    const ph = ih - 190;
    if (reps.length > 1) {
      const a = RS.autoBins(reps.concat([obs]), 42, 0.06);
      const x2 = d3.scaleLinear().domain([a.lo, a.hi]).range([0, iw]);
      const y2 = d3.scaleLinear().domain([0, d3.max(a.b, d => d.density) * 1.15]).range([ph, 0]);
      ST.gridY(g2, y2, iw, 4);
      const twoSided = eSide.value === "two";
      /* shade the bins that count towards the p-value */
      const isExtreme = v => twoSided ? Math.abs(v) >= Math.abs(obs) - 1e-12 : v >= obs - 1e-12;
      const mid = b => (b.x0 + b.x1) / 2;
      RS.barsOf(g2, a.b.filter(b => !isExtreme(mid(b))), x2, y2, ph, SC.accent, 0.40);
      RS.barsOf(g2, a.b.filter(b => isExtreme(mid(b))), x2, y2, ph, SC.bad, 0.62);
      RS.rule(g2, x2(obs), ph, SC.good, "observed = " + ST.fmt(obs, 3));
      if (twoSided && eS.value !== "ranksum")
        RS.rule(g2, x2(-obs), ph, SC.good, null, "3 3");
      ST.axisB(g2, x2, ph, 6, SLBL[eS.value] + " under H₀");
      ST.axisL(g2, y2, 4, "density");
    } else {
      g2.append("text").attr("x", iw / 2).attr("y", ph / 2).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", SC.muted).text("press shuffle or run");
    }

    const twoSided = eSide.value === "two";
    const c = reps.filter(v => twoSided ? Math.abs(v) >= Math.abs(obs) - 1e-12 : v >= obs - 1e-12).length;
    const B = reps.length;
    const p = (1 + c) / (B + 1);
    const naive = B ? c / B : NaN;
    const lnTotal = ST.lnChoose(n, n1);
    const totStr = lnTotal < 34 ? Math.round(Math.exp(lnTotal)).toLocaleString("en-US")
      : ST.fmt(Math.pow(10, lnTotal / Math.LN10 - Math.floor(lnTotal / Math.LN10)), 2)
      + " × 10" + String(Math.floor(lnTotal / Math.LN10)).replace(/[0-9]/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]);
    out.innerHTML =
      `${D.name} · statistic ${SLBL[eS.value]} · observed <b>${ST.fmt(obs, 5)}</b> — and it never moves.<br>`
      + (B
        ? `${B} shuffles, ${c} at least as extreme &nbsp;⟹&nbsp; p = (1 + ${c})/(${B} + 1) = <b>${RS.pstr(p)}</b>`
        + ` &nbsp;·&nbsp; the uncorrected c/B would be ${RS.pstr(naive)}`
        + ` &nbsp;·&nbsp; floor at this B: ${RS.pstr(1 / (B + 1))}`
        : "no shuffles yet")
      + `<br>There are C(${n}, ${n1}) = <b>${totStr}</b> distinct labellings in total. `
      + (lnTotal < 12
        ? `Small enough to enumerate exhaustively — §13 does exactly that and gets an EXACT p-value with no seed.`
        : `Far too many to enumerate, so this is a sampled permutation test and its p-value carries a Monte-Carlo error of `
        + `√(p(1−p)/B) = ${B ? ST.sig(Math.sqrt(p * (1 - p) / B), 2) : "—"}.`)
      + (eD.value === "run" && eS.value === "diff"
        ? `<br>The exhaustive answer for this data and statistic is <b>0.102887</b> two-sided (0.051641 one-sided); the pooled `
        + `t test gives 0.10219 and Welch gives 0.08781.`
        : "")
      + (eD.value === "tiny"
        ? `<br>Exhaustively, exactly 4 of the 70 splits are at least as extreme, so the true two-sided p is 4/70 = <b>0.05714</b> — `
        + `while the pooled t test reports 0.04153 and rejects.`
        : "");
  }

  function tick() { stepOnce(); draw(); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } document.getElementById("pm-run").textContent = "run"; }
  function reset() { stop(); reps = []; labels = null; seed = (seed * 7 + 3) % 999983; r = ST.rng(seed); D = dataset(); draw(); }

  eS.onchange = () => { reps = []; draw(); };
  eSide.onchange = () => { draw(); };
  eD.onchange = reset;
  eSp.oninput = () => {
    eSpv.textContent = eSp.value + "/s";
    if (timer) { clearInterval(timer); timer = setInterval(tick, 1000 / +eSp.value); }
  };
  document.getElementById("pm-step").onclick = () => { stop(); tick(); };
  document.getElementById("pm-run").onclick = function () {
    if (timer) { stop(); return; }
    this.textContent = "pause";
    timer = setInterval(tick, 1000 / +eSp.value);
  };
  document.getElementById("pm-reset").onclick = reset;
  draw();
})();

/* ─────────────── 10 · exchangeability, broken on purpose ─────────────── */
(function () {
  const svg = d3.select("#t1-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 460;
  const e1 = document.getElementById("tv-n1"), e1v = document.getElementById("tv-n1v");
  const e2 = document.getElementById("tv-n2"), e2v = document.getElementById("tv-n2v");
  const eR = document.getElementById("tv-rat"), eRv = document.getElementById("tv-ratv");
  const eS = document.getElementById("tv-stat");
  const out = document.getElementById("t1-readout");
  let hitsP = 0, hitsW = 0, N = 0, seed = 5309;

  const ratio = () => Math.pow(10, +eR.value);           // slider is log₁₀ of σ_A/σ_B

  function reset() { hitsP = 0; hitsW = 0; N = 0; }

  function runBatch(R) {
    const n1 = Math.round(+e1.value), n2 = Math.round(+e2.value), rat = ratio(), BP = 199;
    const sA = rat, sB = 1, r = ST.rng(seed + N);
    const n = n1 + n2;
    const pool = new Array(n);
    for (let rep = 0; rep < R; rep++) {
      for (let i = 0; i < n1; i++) pool[i] = sA * ST.randn(r);
      for (let i = 0; i < n2; i++) pool[n1 + i] = sB * ST.randn(r);
      const A = pool.slice(0, n1), B = pool.slice(n1);
      const T = eS.value === "t" ? RS.welch : (a, b) => ST.mean(a) - ST.mean(b);
      const obs = Math.abs(T(A, B));
      let c = 0;
      for (let b = 0; b < BP; b++) {
        const sh = ST.shuffle(pool, r);
        if (Math.abs(T(sh.slice(0, n1), sh.slice(n1))) >= obs - 1e-12) c++;
      }
      if ((1 + c) / (BP + 1) <= 0.05) hitsP++;
      /* Welch's t test on the same data, for reference */
      const tw = RS.welch(A, B);
      const vA = ST.variance(A) / n1, vB = ST.variance(B) / n2;
      const df = (vA + vB) * (vA + vB) / (vA * vA / (n1 - 1) + vB * vB / (n2 - 1));
      if (2 * (1 - ST.tCdf(Math.abs(tw), df)) <= 0.05) hitsW++;
      N++;
    }
  }

  function draw() {
    const n1 = Math.round(+e1.value), n2 = Math.round(+e2.value), rat = ratio();
    const f = ST.frame(svg, W, H, { l: 50, r: 18, t: 24, b: 44 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gap = 54, leftW = Math.round(iw * 0.48), rightW = iw - leftW - gap;

    /* ── left: the two populations, both centred at 0 ── */
    const gl = RS.sub(g, 0, 0, "two populations, both with mean exactly 0");
    const lim = 3.4 * Math.max(rat, 1);
    const xs = ST.linspace(-lim, lim, 220);
    const x = d3.scaleLinear().domain([-lim, lim]).range([0, leftW]);
    const dA = xs.map(v => ST.normPdf(v / rat) / rat), dB = xs.map(v => ST.normPdf(v));
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(dA), d3.max(dB)) * 1.1]).range([ih, 0]);
    ST.gridY(gl, y, leftW, 4);
    gl.append("path").attr("fill", SC.accent).attr("fill-opacity", 0.22)
      .attr("d", d3.area().x((d, i) => x(xs[i])).y0(ih).y1(d => y(d))(dA));
    gl.append("path").attr("fill", SC.a2).attr("fill-opacity", 0.22)
      .attr("d", d3.area().x((d, i) => x(xs[i])).y0(ih).y1(d => y(d))(dB));
    gl.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2)
      .attr("d", d3.line().x((d, i) => x(xs[i])).y(d => y(d))(dA));
    gl.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2)
      .attr("d", d3.line().x((d, i) => x(xs[i])).y(d => y(d))(dB));
    gl.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 4");
    ST.axisB(gl, x, ih, 5, "value");
    ST.axisL(gl, y, 4, "density");
    ST.legend(gl, [
      { label: `group A: n₁ = ${n1}, σ = ${ST.fmt(rat, 2)}`, color: SC.accent, op: 0.6 },
      { label: `group B: n₂ = ${n2}, σ = 1.00`, color: SC.a2, op: 0.6 }
    ], 6, 12);

    /* ── right: the measured type I error ── */
    const gr = RS.sub(g, leftW + gap, 0, "measured type I error — every rejection is a false positive");
    const y2 = d3.scaleLinear().domain([0, Math.max(0.12, N ? 1.25 * Math.max(hitsP / N, 0.05) : 0.12)]).range([ih, 0]);
    const xb = d3.scaleBand().domain(["permutation", "Welch t"]).range([0, rightW]).padding(0.34);
    ST.gridY(gr, y2, rightW, 5);
    if (N) {
      const rp = hitsP / N, rw = hitsW / N;
      const col = rp > 0.075 ? SC.bad : (rp < 0.03 ? SC.accent : SC.good);
      gr.append("rect").attr("x", xb("permutation")).attr("y", y2(rp))
        .attr("width", xb.bandwidth()).attr("height", ih - y2(rp))
        .attr("fill", col).attr("fill-opacity", 0.62).attr("rx", 3);
      gr.append("rect").attr("x", xb("Welch t")).attr("y", y2(rw))
        .attr("width", xb.bandwidth()).attr("height", ih - y2(rw))
        .attr("fill", SC.muted).attr("fill-opacity", 0.45).attr("rx", 3);
      [["permutation", rp, col], ["Welch t", rw, SC.muted]].forEach(b => {
        gr.append("text").attr("x", xb(b[0]) + xb.bandwidth() / 2).attr("y", y2(b[1]) - 6)
          .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 600)
          .attr("fill", b[2]).text(ST.pct(b[1], 2));
      });
    }
    gr.append("line").attr("x1", 0).attr("x2", rightW).attr("y1", y2(0.05)).attr("y2", y2(0.05))
      .attr("stroke", SC.good).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 4");
    gr.append("text").attr("x", rightW - 2).attr("y", y2(0.05) - 6).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.good).text("nominal α = 5%");
    gr.append("g").attr("transform", `translate(0,${ih})`).attr("class", "axis").call(d3.axisBottom(xb));
    ST.axisL(gr, y2, 5, "rejection rate", d3.format(".0%"));

    const mc = N ? 1.959964 * Math.sqrt(0.05 * 0.95 / N) : NaN;
    out.innerHTML =
      `n₁ = ${n1} (σ = ${ST.fmt(rat, 2)}) against n₂ = ${n2} (σ = 1) · statistic: `
      + `<b>${eS.value === "t" ? "the studentised Welch t" : "the raw difference of means"}</b> · `
      + `${N} null data sets${N ? ` (Monte-Carlo error ±${ST.pct(mc, 2)})` : ""}<br>`
      + (N
        ? `permutation test rejects <b>${ST.pct(hitsP / N, 2)}</b> of the time; Welch's t test rejects ${ST.pct(hitsW / N, 2)}. `
        + (hitsP / N > 0.075
          ? (eS.value === "t"
            ? `<b>Still above nominal</b> — but compare it with the raw difference of means on the same design, which is far `
            + `worse. Studentising removes most of the damage; what is left is there because that repair is asymptotic and `
            + `n₁ = ${n1} is not asymptotic.`
            : `<b>Far too often.</b> The small group is the WIDE one, so the observed difference is more variable than the pooled `
            + `reference distribution allows for, and ordinary results look extreme.`)
          : (hitsP / N < 0.03
            ? `<b>Far too rarely.</b> The small group is the NARROW one, so the pooled reference distribution is wider than the `
            + `observed statistic's own distribution, and nothing ever looks extreme — the power loss is invisible and total.`
            : `About right — either the groups are balanced or the spreads are equal, and exchangeability nearly holds.`))
        : `Press <b>run</b>. Both populations have mean 0, so the null is true and every rejection is an error.`)
      + `<br>Balance the design (n₁ = n₂) or studentise the statistic and most of the damage disappears — but at n₁ = 5 `
      + `even studentising leaves a measurable excess, because that repair is asymptotic.`;
  }

  [e1, e2].forEach(el => el.oninput = () => {
    e1v.textContent = e1.value; e2v.textContent = e2.value; reset(); draw();
  });
  eR.oninput = () => { eRv.textContent = ST.fmt(ratio(), 2); reset(); draw(); };
  eS.onchange = () => { reset(); draw(); };
  document.getElementById("tv-run").onclick = function () {
    this.textContent = "running…";
    setTimeout(() => { runBatch(1500); draw(); this.textContent = "run 1500 null data sets"; }, 20);
  };
  document.getElementById("tv-reset").onclick = () => { reset(); draw(); };
  eRv.textContent = ST.fmt(ratio(), 2);
  draw();
})();

/* ─────────────── 11 · five ways to resample a regression ─────────────── */
(function () {
  const svg = d3.select("#reg-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 500;
  const eS = document.getElementById("rg-scheme"), eD = document.getElementById("rg-data");
  const eB = document.getElementById("rg-b"), eBv = document.getElementById("rg-bv");
  const out = document.getElementById("reg-readout");
  let seed = 1234;

  function dataset() {
    if (eD.value === "run") return { x: RS.RUNX.slice(), y: RS.RUNY.slice(), name: "part 6's twelve service calls" };
    const r = ST.rng(303), x = [], y = [];
    if (eD.value === "het") {
      for (let i = 0; i < 30; i++) {
        const xi = 1 + 9 * i / 29;
        x.push(Math.round(xi * 100) / 100);
        y.push(Math.round((6 + 3 * xi + 0.55 * xi * ST.randn(r)) * 100) / 100);
      }
    } else {
      for (let i = 0; i < 12; i++) {
        const xi = 2 + 14 * i / 11;
        x.push(Math.round(xi * 10) / 10);
        y.push(Math.round((6 + 3 * xi + 5 * ST.randn(r)) * 10) / 10);
      }
      x.push(34); y.push(96);                      // one deliberate high-leverage point
    }
    return { x: x, y: y, name: eD.value === "het" ? "a heteroscedastic cloud (σ grows with x)" : "twelve points plus one at x = 34" };
  }

  function fit(x, y) {
    const n = x.length, xb = ST.mean(x), yb = ST.mean(y);
    let sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - xb; sxx += d * d; sxy += d * (y[i] - yb); }
    const b = sxx > 0 ? sxy / sxx : 0, a = yb - b * xb;
    let sse = 0;
    for (let i = 0; i < n; i++) { const e = y[i] - a - b * x[i]; sse += e * e; }
    return { a: a, b: b, sxx: sxx, sse: sse, n: n, xb: xb, se: Math.sqrt(sse / (n - 2)), seb: Math.sqrt(sse / (n - 2)) / Math.sqrt(sxx) };
  }

  function draw() {
    const D = dataset(), x = D.x, y = D.y, n = x.length;
    const F = fit(x, y), scheme = eS.value, B = Math.round(+eB.value);
    const r = ST.rng(seed);
    const h = x.map(v => 1 / n + (v - F.xb) * (v - F.xb) / F.sxx);
    const res = x.map((v, i) => y[i] - F.a - F.b * v);
    const resC = res.map((e, i) => e / Math.sqrt(Math.max(1e-9, 1 - h[i])));
    const resCm = ST.mean(resC);
    const resCc = resC.map(e => e - resCm);

    const bs = [], keep = [], keepPts = [];
    const yy = new Array(n), xx = new Array(n);
    for (let b = 0; b < B; b++) {
      let bx = x, by = yy, mult = null;
      if (scheme === "resid") {
        for (let i = 0; i < n; i++) yy[i] = F.a + F.b * x[i] + res[Math.floor(r() * n)];
      } else if (scheme === "residc") {
        for (let i = 0; i < n; i++) yy[i] = F.a + F.b * x[i] + resCc[Math.floor(r() * n)];
      } else if (scheme === "wild") {
        for (let i = 0; i < n; i++) yy[i] = F.a + F.b * x[i] + resC[i] * (r() < 0.5 ? -1 : 1);
      } else if (scheme === "perm") {
        const sh = ST.shuffle(y, r);
        for (let i = 0; i < n; i++) yy[i] = sh[i];
      } else {                                     // case resampling
        mult = new Array(n).fill(0);
        for (let i = 0; i < n; i++) { const j = Math.floor(r() * n); xx[i] = x[j]; yy[i] = y[j]; mult[j]++; }
        bx = xx;
      }
      const Fb = fit(bx, by);
      if (isFinite(Fb.b)) bs.push(Fb.b);
      if (b < 26) { keep.push({ a: Fb.a, b: Fb.b }); if (mult) keepPts.push(mult.slice()); }
    }

    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 26, b: 46 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gap = 46, leftW = Math.round(iw * 0.46), rightW = iw - leftW - gap;

    /* ── left: the cloud with the fan of refits ── */
    const gl = RS.sub(g, 0, 0, scheme === "perm" ? "26 refits after SHUFFLING y against x" : "26 of the " + B + " refits");
    const xlo = d3.min(x), xhi = d3.max(x), xp = 0.08 * (xhi - xlo);
    const ylo = d3.min(y), yhi = d3.max(y), yp = 0.14 * (yhi - ylo);
    const sx = d3.scaleLinear().domain([xlo - xp, xhi + xp]).range([0, leftW]);
    const sy = d3.scaleLinear().domain([ylo - yp, yhi + yp]).range([ih, 0]);
    ST.gridY(gl, sy, leftW, 4);
    keep.forEach(K => RS.lineSeg(gl, K.a, K.b, sx, sy, { color: scheme === "perm" ? SC.bad : SC.accent, w: 1, op: 0.28 }));
    if (scheme === "case" && keepPts.length) {
      const m0 = keepPts[0];
      x.forEach((v, i) => gl.append("circle").attr("cx", sx(v)).attr("cy", sy(y[i]))
        .attr("r", m0[i] === 0 ? 2.4 : 2.4 + 2.1 * m0[i]).attr("fill", m0[i] === 0 ? SC.muted : SC.a2)
        .attr("fill-opacity", m0[i] === 0 ? 0.25 : 0.6));
    } else {
      RS.pts(gl, x, y, sx, sy, { r: 3.4, fill: SC.a2, op: 0.85 });
    }
    RS.lineSeg(gl, F.a, F.b, sx, sy, { color: SC.good, w: 2.4 });
    ST.axisB(gl, sx, ih, 5, "x");
    ST.axisL(gl, sy, 4, "y");

    /* ── right: the histogram of b̂* ── */
    const gr = RS.sub(g, leftW + gap, 0, "the bootstrapped slopes b̂*");
    const a = RS.autoBins(bs.concat([F.b]), 40, 0.06);
    const x2 = d3.scaleLinear().domain([a.lo, a.hi]).range([0, rightW]);
    const dens = ST.linspace(a.lo, a.hi, 180).map(v => ({
      x: v, y: ST.normPdf((v - (scheme === "perm" ? 0 : F.b)) / F.seb) / F.seb
    }));
    const y2 = d3.scaleLinear()
      .domain([0, Math.max(d3.max(a.b, d => d.density), d3.max(dens, d => d.y)) * 1.14]).range([ih, 0]);
    ST.gridY(gr, y2, rightW, 4);
    RS.barsOf(gr, a.b, x2, y2, ih, scheme === "perm" ? SC.bad : SC.accent, 0.45);
    gr.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2)
      .attr("stroke-dasharray", "5 3")
      .attr("d", d3.line().x(d => x2(d.x)).y(d => y2(d.y))(dens));
    RS.rule(gr, x2(F.b), ih, SC.good, "b̂ = " + ST.fmt(F.b, 3));
    ST.axisB(gr, x2, ih, 5, "b̂*");
    ST.axisL(gr, y2, 4, "density");
    ST.legend(gr, [{ label: "Normal(·, SE(b)² ) from the formula", color: SC.a2, dash: "5 3" }], 4, 12);

    const seB = ST.sd(bs);
    const c = bs.filter(v => Math.abs(v) >= Math.abs(F.b) - 1e-12).length;

    /* For the three FIXED-DESIGN schemes the B = ∞ answer is available in closed form,
       because b* = b̂ + Σcᵢe*ᵢ with cᵢ = (xᵢ − x̄)/S_xx and the e*ᵢ independent.
       Printing it turns this figure into a self-check on its own simulation.        */
    const ci = x.map(v => (v - F.xb) / F.sxx);
    let exact = NaN;
    if (scheme === "resid") exact = Math.sqrt(F.sse / n / F.sxx);
    else if (scheme === "residc") { let m2 = 0; for (const e of resCc) m2 += e * e; exact = Math.sqrt(m2 / n / F.sxx); }
    else if (scheme === "wild") { let v2 = 0; for (let i = 0; i < n; i++) v2 += ci[i] * ci[i] * res[i] * res[i] / (1 - h[i]); exact = Math.sqrt(v2); }

    const NOTE = {
      resid: `Raw residuals are too small — E[<span class="keep">Σ</span>ê ᵢ²] = <span class="keep">σ</span>²(n − 2), not n<span class="keep">σ</span>² — so this SE is deflated by exactly `
        + `√((n − 2)/n) = <b>${ST.fmt(Math.sqrt((n - 2) / n), 5)}</b>. Predicted ${ST.fmt(Math.sqrt((n - 2) / n) * F.seb, 5)}, measured ${ST.fmt(seB, 5)}.`,
      residc: `Rescaling by 1/√(1 − hᵢ) equalises the residual variances and removes most of the deflation. What is left is the `
        + `residuals' mutual correlation, which independent resampling cannot reproduce.`,
      case: `Cases resample whole rows, so S_xx varies from resample to resample and no model form is assumed. This is the only `
        + `scheme here that survives a wrong model or non-constant variance — try the heteroscedastic cloud.`,
      wild: `Each residual keeps its own magnitude and only flips sign, so a noisy region stays noisy — SE(b*)² = `
        + `<span class="keep">Σ</span>cᵢ²ê ᵢ²/(1 − hᵢ) with cᵢ = (xᵢ − x̄)/S_xx, which is unbiased for <span class="keep">σ</span>²/S_xx on AVERAGE but sample-specific. `
        + `On part 6's twelve points it lands 18% low, because that sample's three largest residuals sit at low-leverage `
        + `x-values where they barely move the slope. Fixed design, heteroscedasticity-robust — the resampling counterpart `
        + `of a sandwich standard error.`,
      perm: `Shuffling y against x destroys the pairing, which is exactly H₀: <span class="keep">β</span>₁ = 0. The histogram is the null distribution, `
        + `centred on 0 — not a sampling distribution, and not something to read a standard error off.`
    };
    out.innerHTML =
      `${D.name} · n = ${n} · b̂ = <b>${ST.fmt(F.b, 5)}</b> · S_xx = ${ST.fmt(F.sxx, 2)} · s_e = ${ST.fmt(F.se, 4)} · `
      + `formula SE(b) = s_e/√S_xx = <b>${ST.fmt(F.seb, 6)}</b><br>`
      + (scheme === "perm"
        ? `${B} shuffles, ${c} with |b*| ≥ |b̂| &nbsp;⟹&nbsp; p = (1 + ${c})/(${B} + 1) = <b>${RS.pstr((1 + c) / (B + 1))}</b>, `
        + `and the floor at this B is ${RS.pstr(1 / (B + 1))}. `
        : `bootstrap SE(b*) = <b>${ST.fmt(seB, 6)}</b> (${ST.pct(seB / F.seb - 1, 1)} against the formula`
        + `${isFinite(exact) ? "; the exact B = ∞ value for this scheme and this sample is " + ST.fmt(exact, 6) : ""}) · `
        + `95% percentile interval (<b>${ST.fmt(RS.q(bs, 0.025), 4)}</b>, <b>${ST.fmt(RS.q(bs, 0.975), 4)}</b>) `
        + `against the t interval (${ST.fmt(F.b - ST.tQuant(0.975, n - 2) * F.seb, 4)}, ${ST.fmt(F.b + ST.tQuant(0.975, n - 2) * F.seb, 4)}). `)
      + `<br>` + NOTE[scheme];
  }

  eS.onchange = draw; eD.onchange = draw;
  eB.oninput = () => { eBv.textContent = eB.value; draw(); };
  document.getElementById("rg-run").onclick = () => { seed = (seed * 11 + 7) % 999983; draw(); };
  draw();
})();

/* ─────────────── 12 · the failure gallery ─────────────── */
(function () {
  const svg = d3.select("#fail-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 480;
  const eS = document.getElementById("fl-stat");
  const eN = document.getElementById("fl-n"), eNv = document.getElementById("fl-nv");
  const eC = document.getElementById("fl-cov");
  const out = document.getElementById("fail-readout");
  /* seed chosen so the DEFAULT sample is representative of its population: at n = 30
     it gives x̄ = 0.985 against a true 1, a median of 0.661 against ln 2 = 0.693, a
     maximum of 0.986 and a heavy-tailed mean of 3.04 against a true 3. A careless
     seed here would show a "the two curves agree" caption over two curves that do
     not, which is exactly the trap part 6 fell into three times.                   */
  let seed = 1037;

  /* each case names its population, its statistic, and the TRUE parameter value */
  const CASE = {
    mean: { pop: "exp", stat: a => ST.mean(a), truth: 1, lbl: "the mean", pname: "Exponential(1)" },
    median: { pop: "exp", stat: a => ST.median(a), truth: Math.LN2, lbl: "the median", pname: "Exponential(1)" },
    max: { pop: "unif", stat: a => Math.max.apply(null, a), truth: 1, lbl: "the maximum", pname: "Uniform(0, 1)" },
    musq: { pop: "normal", stat: a => ST.mean(a) * ST.mean(a), truth: 0, lbl: "the square of the mean", pname: "Normal(0, 1)" },
    heavy: { pop: "pareto", stat: a => ST.mean(a), truth: 3, lbl: "the mean", pname: "Pareto, tail index 1.5 (INFINITE variance)" }
  };
  const drawOne = (pop, r) => pop === "pareto" ? Math.pow(Math.max(1e-12, r()), -1 / 1.5) : RS.drawOne(pop, r);
  const samp = (pop, n, r) => { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = drawOne(pop, r); return a; };

  function draw() {
    const key = eS.value, C = CASE[key], n = Math.round(+eN.value);
    const R = 4000, B = 4000;
    /* three INDEPENDENT streams, so the displayed sample does not depend on how many
       draws the truth panel happened to consume — otherwise changing R or n silently
       changes which sample is shown, and a default seed can end up teaching the
       opposite of its caption.                                                      */
    const rd = ST.rng(seed), rt = ST.rng(seed + 500009), rb = ST.rng(seed + 900007);

    /* the bootstrap's copy of the error, θ̂* − θ̂, from ONE sample */
    const data = samp(C.pop, n, rd), thHat = C.stat(data);
    const bootErr = RS.boot(data, B, C.stat, rb).map(v => v - thHat);

    /* the TRUE sampling distribution of the error θ̂ − θ, from fresh samples */
    const trueErr = new Array(R);
    for (let i = 0; i < R; i++) trueErr[i] = C.stat(samp(C.pop, n, rt)) - C.truth;

    /* coverage of the percentile interval, if asked for */
    let cov = null;
    if (eC.checked) {
      const Rc = 500, Bc = 400, r2 = ST.rng(seed + 121);
      let hit = 0;
      for (let i = 0; i < Rc; i++) {
        const d2 = samp(C.pop, n, r2);
        const reps = RS.boot(d2, Bc, C.stat, r2);
        if (RS.q(reps, 0.025) <= C.truth && C.truth <= RS.q(reps, 0.975)) hit++;
      }
      cov = { R: Rc, p: hit / Rc };
    }

    const f = ST.frame(svg, W, H, { l: 54, r: 18, t: 26, b: 100 });
    const g = f.g, iw = f.iw, ih = f.ih;

    /* a robust common range: the two distributions can live on very different scales */
    const lo = Math.min(RS.q(trueErr, 0.004), RS.q(bootErr, 0.004));
    const hi = Math.max(RS.q(trueErr, 0.996), RS.q(bootErr, 0.996));
    const pad = 0.06 * (hi - lo || 1);
    const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, iw]);
    const K = 52;
    const bT = RS.bins(trueErr, lo - pad, hi + pad, K), bB = RS.bins(bootErr, lo - pad, hi + pad, K);
    const y = d3.scaleLinear()
      .domain([0, Math.max(d3.max(bT, d => d.density), d3.max(bB, d => d.density)) * 1.12]).range([ih, 0]);
    ST.gridY(g, y, iw, 4);
    RS.barsOf(g, bT, x, y, ih, SC.good, 0.30);
    /* the bootstrap's copy drawn as an outline, so overlap is readable */
    const step = [];
    bB.forEach(b2 => { step.push([b2.x0, b2.density], [b2.x1, b2.density]); });
    g.append("path").attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 1.9)
      .attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(step));
    RS.rule(g, x(0), ih, SC.ink, "0", "3 3");
    ST.axisB(g, x, ih, 6, "estimation error");
    ST.axisL(g, y, 4, "density");
    ST.legend(g, [
      { label: "TRUE sampling distribution of θ̂ − θ", color: SC.good, op: 0.4 },
      { label: "the bootstrap's copy: θ̂* − θ̂", color: SC.bad, dash: "0" }
    ], 8, 14);

    /* a footer strip: coverage, and the atom for the maximum */
    const gf = g.append("g").attr("transform", `translate(0,${ih + 46})`);
    if (cov) {
      const bw = 260;
      gf.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", SC.muted)
        .text("percentile-interval coverage over " + cov.R + " replications");
      gf.append("rect").attr("x", 0).attr("y", 8).attr("width", bw).attr("height", 14)
        .attr("fill", SC.panel2).attr("stroke", SC.line);
      gf.append("rect").attr("x", 0).attr("y", 8).attr("width", bw * cov.p).attr("height", 14)
        .attr("fill", cov.p > 0.90 ? SC.good : SC.bad).attr("fill-opacity", 0.65);
      gf.append("line").attr("x1", bw * 0.95).attr("x2", bw * 0.95).attr("y1", 4).attr("y2", 26)
        .attr("stroke", SC.good).attr("stroke-width", 2);
      gf.append("text").attr("x", bw + 10).attr("y", 20).attr("font-size", 13).attr("font-weight", 600)
        .attr("fill", cov.p > 0.90 ? SC.good : SC.bad).text(ST.pct(cov.p, 2));
      gf.append("text").attr("x", bw + 68).attr("y", 20).attr("font-size", 11).attr("fill", SC.muted)
        .text("against a nominal 95%");
    }

    const atom = bootErr.filter(v => Math.abs(v) < 1e-12).length / bootErr.length;
    out.innerHTML =
      `${C.pname} · statistic: ${C.lbl} · true value ${ST.fmt(C.truth, 5)} · n = ${n} · this sample's θ̂ = ${ST.fmt(thHat, 5)}<br>`
      + `SD of the true error <b>${ST.fmt(ST.sd(trueErr), 5)}</b> · SD of the bootstrap's copy <b>${ST.fmt(ST.sd(bootErr), 5)}</b>`
      + (cov ? ` · measured coverage <b>${ST.pct(cov.p, 2)}</b>` : "")
      + "<br>"
      + (key === "max"
        ? `<b>${ST.pct(atom, 1)}</b> of the bootstrap replicates land EXACTLY on θ̂ — the exact probability is `
        + `1 − (1 − 1/n)ⁿ = ${ST.pct(1 - Math.pow(1 - 1 / n, n), 1)}, tending to 1 − e⁻¹ = 63.2%. The true error is continuous and `
        + `strictly one-sided (θ̂ &lt; θ always); the bootstrap's copy is a spike at zero with a one-sided tail on the WRONG side. `
        + `Since θ̂* ≤ θ̂ always, the percentile interval's upper endpoint is at most θ̂ &lt; θ, so its coverage is exactly zero — `
        + `at every n. Raise n and watch nothing improve.`
        : key === "musq"
          ? `At <span class="keep">μ</span> = 0 the map <span class="keep">μ</span> ⟼ <span class="keep">μ</span>² has zero derivative, so the usual √n limit fails: n·θ̂ converges to a <span class="keep">χ</span>²₁, not a normal. `
          + `The bootstrap distribution is centred on this sample's θ̂ and the interval is strictly positive, so it cannot contain `
          + `<span class="keep">θ</span> = 0. Coverage is zero, and it stays zero as n grows.`
          : key === "heavy"
            ? `The tail index is 1.5, so the mean exists but the variance does not, and §02's Monte-Carlo argument for a standard error `
            + `has no second moment to converge to. Both distributions are dominated by whichever single observation happened to be `
            + `largest; the bootstrap's copy is whatever THIS sample's largest point makes it. Coverage improves with n, but very slowly.`
            : key === "median"
              ? `The bootstrap's copy is lumpy — a resampled median can only land on a small set of order statistics — but it is `
              + `centred correctly and has roughly the right spread, and the percentile interval holds up. Compare the maximum, where `
              + `lumpiness becomes a single dominating atom.`
              : `The two curves sit almost on top of each other. This is the case the bootstrap was built for: a smooth functional, a `
              + `finite variance, and enough data for a central limit theorem. Coverage still lands a few points under 95% — §08's `
              + `table measures 91.8% here — because the percentile interval has no small-sample correction. Everything else in `
              + `this figure is a departure of a completely different order.`);
  }

  eS.onchange = draw; eC.onchange = draw;
  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  document.getElementById("fl-new").onclick = () => { seed = (seed * 43 + 17) % 999983; draw(); };
  draw();
})();
