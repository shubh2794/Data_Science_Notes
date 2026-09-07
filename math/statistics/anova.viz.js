/* anova.viz.js — the seventeen visualizations on math/statistics/anova.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

      1  #pw-svg   the family-wise error rate of pairwise testing, simulated live,
                   against the (wrong) independence formula and the omnibus F
      2  #dec-svg  one observation split into grand mean + group effect + residual,
                   as a stack, as three squares, and as a right triangle
      3  #fd-svg   the F density with both df live, the observed statistic and its
                   tail, the two chi-squares behind it, and a simulated null
      4  #lab-svg  the laboratory: boxplots on the left, the group means with their
                   standard errors on the right, and the whole ANOVA table live
      5  #rb-svg   what the F test's true level does under seven error distributions
                   and five designs — measured, not asserted
      6  #rg-svg   the same data as an ANOVA and as a regression on indicators, in
                   four codings, with the identical entries highlighted
      7  #ef-svg   effect size and p-value diverging as n grows at a fixed truth
      8  #pw2-svg  power curves from the non-central F, and the n they imply
      9  #pm-svg   the permutation distribution of F against its tabulated density
     10  #we-svg   classical F, Welch and Brown–Forsythe under unequal variances
     11  #kw-svg   F, permutation F and Kruskal–Wallis on contaminated data
     12  #lk-svg   the look-elsewhere effect: m tests, no effects, and what survives
     13  #hl-svg   Bonferroni's flat line against Holm's staircase, p-values draggable
     14  #tk-svg   the studentised range density and the simultaneous intervals it gives
     15  #cn-svg   the contrast explorer: weights, orthogonality, and the SS partition
     16  #bh-svg   the Benjamini–Hochberg staircase, and the FDR it actually delivers
     17  #in-svg   the interaction plot, with the two-way table recomputed live

   Every number these draw is recomputed from the data they generate, so the
   pictures and the prose cannot drift apart.                                    */

/* ══════════ page-local helpers (deliberately NOT in stats-viz.js) ══════════ */
const AV = (function () {

  /* ── the running data: 41 exam scores, three teaching regimes ──────────
     Group sums 664 / 691 / 592, ΣΣy² = 93 281, N = 41.
     SS_between = 98.370276, SS_within = 723.873626, SS_total = 822.243902,
     F = 2.581991 on (2, 38), p = 0.088834.                                 */
  const RUN = {
    names: ["Homework only", "Homework + PA", "Study only"],
    short: ["HW", "PA", "Study"],
    y: [
      [43, 43, 44, 45, 46, 46, 46, 47, 47, 47, 49, 49, 53, 59],
      [44, 45, 45, 47, 47, 47, 47, 48, 49, 51, 51, 55, 57, 58],
      [34, 42, 43, 44, 44, 46, 47, 47, 48, 48, 48, 50, 51]
    ]
  };
  /* §21's 2 × 2: prior band × peer assessment, n = 12, cell means 44/50/50/50 */
  const TWOWAY = {
    rows: ["lower band", "upper band"], cols: ["no PA", "PA"],
    cells: [
      [[38, 39, 41, 41, 42, 43, 44, 44, 46, 47, 49, 54], [43, 45, 45, 46, 50, 50, 50, 51, 54, 54, 55, 57]],
      [[44, 45, 46, 47, 47, 48, 52, 52, 53, 53, 54, 59], [42, 43, 47, 48, 50, 50, 50, 52, 53, 53, 55, 57]]
    ]
  };
  const GC = [SC.accent, SC.a2, SC.violet, SC.teal, SC.good, SC.bad];   // group colours

  /* ── the one-way ANOVA, from an array of arrays ─────────────────────── */
  function anova(gs) {
    const k = gs.length, ns = gs.map(g => g.length), N = ns.reduce((a, b) => a + b, 0);
    const means = gs.map(ST.mean);
    let gm = 0;
    for (let j = 0; j < k; j++) gm += means[j] * ns[j];
    gm /= N;
    let ssb = 0, ssw = 0;
    for (let j = 0; j < k; j++) {
      ssb += ns[j] * (means[j] - gm) * (means[j] - gm);
      for (let i = 0; i < ns[j]; i++) ssw += (gs[j][i] - means[j]) * (gs[j][i] - means[j]);
    }
    const df1 = k - 1, df2 = N - k;
    const msb = ssb / df1, msw = ssw / df2, F = msw > 0 ? msb / msw : Infinity;
    return {
      k: k, ns: ns, N: N, means: means, gm: gm,
      sds: gs.map(g => ST.sd(g)),
      ssb: ssb, ssw: ssw, sst: ssb + ssw, df1: df1, df2: df2,
      msb: msb, msw: msw, s: Math.sqrt(msw), F: F,
      p: isFinite(F) ? 1 - ST.fCdf(F, df1, df2) : 0,
      eta2: (ssb + ssw) > 0 ? ssb / (ssb + ssw) : 0,
      omega2: (ssb + ssw + msw) > 0 ? (ssb - df1 * msw) / (ssb + ssw + msw) : 0
    };
  }
  /* the F statistic alone, for the inner loop of a simulation */
  function Fstat(gs) {
    const k = gs.length; let N = 0, tot = 0;
    const means = [];
    for (let j = 0; j < k; j++) {
      let s = 0; for (let i = 0; i < gs[j].length; i++) s += gs[j][i];
      means.push(s / gs[j].length); N += gs[j].length; tot += s;
    }
    const gm = tot / N;
    let ssb = 0, ssw = 0;
    for (let j = 0; j < k; j++) {
      ssb += gs[j].length * (means[j] - gm) * (means[j] - gm);
      for (let i = 0; i < gs[j].length; i++) { const d = gs[j][i] - means[j]; ssw += d * d; }
    }
    return ssw > 0 ? (ssb / (k - 1)) / (ssw / (N - k)) : Infinity;
  }
  /* Welch's F and Brown–Forsythe's F*, both with Satterthwaite denominator df */
  function welch(gs) {
    const k = gs.length, n = gs.map(g => g.length);
    const m = gs.map(ST.mean), v = gs.map(g => ST.variance(g));
    let W = 0, num = 0;
    const w = [];
    for (let j = 0; j < k; j++) { const wj = v[j] > 0 ? n[j] / v[j] : 1e12; w.push(wj); W += wj; }
    for (let j = 0; j < k; j++) num += w[j] * m[j];
    const mt = num / W;
    let A = 0, lam = 0;
    for (let j = 0; j < k; j++) {
      A += w[j] * (m[j] - mt) * (m[j] - mt);
      const t = 1 - w[j] / W; lam += t * t / (n[j] - 1);
    }
    A /= (k - 1);
    const F = A / (1 + 2 * (k - 2) * lam / (k * k - 1)), df2 = (k * k - 1) / (3 * lam);
    return { F: F, df1: k - 1, df2: df2, p: 1 - ST.fCdf(F, k - 1, df2) };
  }
  function brownForsythe(gs) {
    const k = gs.length, n = gs.map(g => g.length), N = n.reduce((a, b) => a + b, 0);
    const m = gs.map(ST.mean), v = gs.map(g => ST.variance(g));
    let gm = 0; for (let j = 0; j < k; j++) gm += m[j] * n[j]; gm /= N;
    let num = 0, den = 0, dd = 0;
    const c = [];
    for (let j = 0; j < k; j++) { num += n[j] * (m[j] - gm) * (m[j] - gm); const cj = (1 - n[j] / N) * v[j]; c.push(cj); den += cj; }
    for (let j = 0; j < k; j++) dd += c[j] * c[j] / (n[j] - 1);
    const F = den > 0 ? num / den : Infinity, df2 = den * den / dd;
    return { F: F, df1: k - 1, df2: df2, p: 1 - ST.fCdf(F, k - 1, df2) };
  }
  function kruskal(gs) {
    const k = gs.length, all = [].concat.apply([], gs), N = all.length;
    const r = ST.ranks(all);
    let off = 0, s = 0;
    for (let j = 0; j < k; j++) {
      let Rj = 0;
      for (let i = 0; i < gs[j].length; i++) Rj += r[off + i];
      s += Rj * Rj / gs[j].length; off += gs[j].length;
    }
    let H = 12 / (N * (N + 1)) * s - 3 * (N + 1);
    const counts = new Map();
    all.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
    let tie = 0; counts.forEach(t => { tie += t * t * t - t; });
    const corr = 1 - tie / (N * N * N - N);
    if (corr > 0) H /= corr;
    return { H: H, df: k - 1, p: 1 - ST.chi2Cdf(H, k - 1) };
  }

  /* ── the studentised range q(k, ν) ──────────────────────────────────────
     No closed form; both integrals by Gauss–Legendre on partitioned intervals,
     the quantile by bisection. VERIFIED before use, four ways:
       · against an independent implementation on a 187-cell (k, ν) grid —
         worst |Δ| = 1.4e-8 over k ≤ 12, ν ≥ 5 (1.0e-4 at the corner k=20, ν=2,
         which nothing on this page approaches), and 5e-10 at the 1% level;
       · against published tables: q₀.₉₅(3,10)=3.8768 vs 3.88, (3,20)=3.5779 vs
         3.58, (4,10)=4.3266 vs 4.33, (5,20)=4.2319 vs 4.23, (10,20)=5.0079 vs
         5.01, (6,12)=4.7502 vs 4.75;
       · against the exact identity q₀.₉₅(2,ν)/√2 = t₀.₉₇₅(ν), to 8 decimals;
       · against 400 000 simulated ranges at several (k, ν).                  */
  const QR = (function () {
    /* Hart's Φ. ST.normCdf goes through the incomplete gamma, which is exact and
       far too slow for the ~20 000 evaluations one q cdf needs. This agrees with
       it to 5.6e-16 over |z| ≤ 8 and is about forty times faster.               */
    function Phi(x) {
      const a = Math.abs(x);
      let cum;
      if (a > 37) cum = 0;
      else {
        const e = Math.exp(-a * a / 2);
        if (a < 7.07106781186547) {
          let b = 3.52624965998911e-2 * a + 0.700383064443688;
          b = b * a + 6.37396220353165; b = b * a + 33.912866078383;
          b = b * a + 112.079291497871; b = b * a + 221.213596169931;
          b = b * a + 220.206867912376;
          let c = 8.83883476483184e-2 * a + 1.75566716318264;
          c = c * a + 16.064177579207; c = c * a + 86.7807322029461;
          c = c * a + 296.564248779674; c = c * a + 637.333633378831;
          c = c * a + 793.826512519948; c = c * a + 440.413735824752;
          cum = e * b / c;
        } else {
          const d = a + 1 / (a + 2 / (a + 3 / (a + 4 / (a + 0.65))));
          cum = e / (d * 2.506628274631);
        }
      }
      return x > 0 ? 1 - cum : cum;
    }
    function gauss(n) {                       // Newton on the Legendre polynomial
      const x = new Array(n), w = new Array(n), m = (n + 1) >> 1;
      for (let i = 0; i < m; i++) {
        let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5)), pp = 0;
        for (let it = 0; it < 100; it++) {
          let p1 = 1, p2 = 0;
          for (let j = 0; j < n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1); }
          pp = n * (z * p1 - p2) / (z * z - 1);
          const z1 = z; z = z1 - p1 / pp;
          if (Math.abs(z - z1) < 1e-15) break;
        }
        x[i] = -z; x[n - 1 - i] = z;
        w[i] = 2 / ((1 - z * z) * pp * pp); w[n - 1 - i] = w[i];
      }
      return { x: x, w: w };
    }
    const G = gauss(20), ZLO = -8.5, ZHI = 8.5, ZP = 6, NZ = ZP * G.x.length;
    const ZN = new Float64Array(NZ), ZWPHI = new Float64Array(NZ), ZCDF = new Float64Array(NZ);
    {
      let t = 0;
      for (let p = 0; p < ZP; p++) {
        const a = ZLO + (ZHI - ZLO) * p / ZP, b = ZLO + (ZHI - ZLO) * (p + 1) / ZP;
        for (let i = 0; i < G.x.length; i++, t++) {
          const z = 0.5 * (b - a) * G.x[i] + 0.5 * (a + b);
          ZN[t] = z;
          ZWPHI[t] = 0.5 * (b - a) * G.w[i] * 0.3989422804014327 * Math.exp(-z * z / 2);
          ZCDF[t] = Phi(z);
        }
      }
    }
    function rangeCdf(w, k) {                 // P(range of k iid N(0,1) ≤ w)
      if (!(w > 0)) return 0;
      let s = 0;
      for (let i = 0; i < NZ; i++) {
        let d = ZCDF[i] - Phi(ZN[i] - w);
        if (d <= 0) continue;
        if (d > 1) d = 1;
        s += ZWPHI[i] * (k === 2 ? d : Math.pow(d, k - 1));
      }
      return Math.min(1, k * s);
    }
    const SP = 6, sCache = new Map();
    function sNodes(nu) {                     // nodes for the scale mixture, cached per ν
      let c = sCache.get(nu);
      if (c) return c;
      const lo = Math.sqrt(ST.chi2Quant(1e-12, nu) / nu), hi = Math.sqrt(ST.chi2Quant(1 - 1e-12, nu) / nu);
      const lc = (nu / 2) * Math.log(nu) - ST.lnGamma(nu / 2) - (nu / 2 - 1) * Math.LN2;
      const s = [], wt = [];
      for (let p = 0; p < SP; p++) {
        const a = lo + (hi - lo) * p / SP, b = lo + (hi - lo) * (p + 1) / SP;
        for (let i = 0; i < G.x.length; i++) {
          const sv = 0.5 * (b - a) * G.x[i] + 0.5 * (a + b);
          s.push(sv);
          wt.push(0.5 * (b - a) * G.w[i] * Math.exp(lc + (nu - 1) * Math.log(sv) - nu * sv * sv / 2));
        }
      }
      c = { s: s, w: wt }; sCache.set(nu, c); return c;
    }
    function qCdf(q, k, nu) {
      if (!(q > 0)) return 0;
      if (!isFinite(nu)) return rangeCdf(q, k);
      const c = sNodes(nu);
      let tot = 0;
      for (let i = 0; i < c.s.length; i++) tot += c.w[i] * rangeCdf(q * c.s[i], k);
      return Math.min(1, tot);
    }
    const memo = new Map();
    function qQuant(p, k, nu) {
      const key = p + "|" + k + "|" + nu;
      if (memo.has(key)) return memo.get(key);
      let lo = 0, hi = 30;
      for (let i = 0; i < 30; i++) { const m = 0.5 * (lo + hi); if (qCdf(m, k, nu) < p) lo = m; else hi = m; }
      const v = 0.5 * (lo + hi); memo.set(key, v); return v;
    }
    /* the density, by central difference — only ever wanted for drawing */
    function qPdf(q, k, nu) {
      const h = Math.max(1e-4, q * 1e-4);
      return Math.max(0, (qCdf(q + h, k, nu) - qCdf(q - h, k, nu)) / (2 * h));
    }
    return { Phi: Phi, rangeCdf: rangeCdf, qCdf: qCdf, qPdf: qPdf, qQuant: qQuant,
             qSf: function (q, k, nu) { return 1 - qCdf(q, k, nu); } };
  })();

  /* ── multiplicity procedures, all taking an array of p-values ─────────── */
  function holm(p, alpha) {                   // returns a boolean array of rejections
    const m = p.length, idx = p.map((v, i) => i).sort((a, b) => p[a] - p[b]);
    const rej = new Array(m).fill(false);
    for (let i = 0; i < m; i++) {
      if (p[idx[i]] <= alpha / (m - i)) rej[idx[i]] = true; else break;
    }
    return rej;
  }
  function holmAdj(p) {                       // Holm-adjusted p-values, monotone
    const m = p.length, idx = p.map((v, i) => i).sort((a, b) => p[a] - p[b]);
    const out = new Array(m); let run = 0;
    for (let i = 0; i < m; i++) {
      run = Math.max(run, Math.min(1, (m - i) * p[idx[i]]));
      out[idx[i]] = run;
    }
    return out;
  }
  function bh(p, alpha) {                     // returns {rej, cut} — cut is the largest k, 0 if none
    const m = p.length, idx = p.map((v, i) => i).sort((a, b) => p[a] - p[b]);
    let cut = 0;
    for (let i = 0; i < m; i++) if (p[idx[i]] <= (i + 1) / m * alpha) cut = i + 1;
    const rej = new Array(m).fill(false);
    for (let i = 0; i < cut; i++) rej[idx[i]] = true;
    return { rej: rej, cut: cut, order: idx };
  }
  const sidak = (alpha, m) => 1 - Math.pow(1 - alpha, 1 / m);
  const harmonic = m => { let s = 0; for (let i = 1; i <= m; i++) s += 1 / i; return s; };

  /* ── random draws from the shapes §09 and §15 need ────────────────────── */
  function draw(kind, r) {
    switch (kind) {
      case "unif": return (r() * 2 - 1) * Math.sqrt(3);
      case "exp": return -Math.log(1 - r()) - 1;
      case "lnorm": return Math.exp(ST.randn(r)) - Math.exp(0.5);
      case "t3": { let s = 0; for (let i = 0; i < 3; i++) { const z = ST.randn(r); s += z * z; } return ST.randn(r) / Math.sqrt(s / 3); }
      case "cauchy": return Math.tan(Math.PI * (r() - 0.5));
      case "bern": return (r() < 0.1) ? 1 : 0;
      default: return ST.randn(r);
    }
  }
  /* the non-central F survival function, by a Poisson mixture of central betas.
     P(F(d₁,d₂;λ) > x) = Σ_i e^{−λ/2}(λ/2)^i/i! · I_{1−u}(d₂/2, d₁/2 + i)
     with u = d₁x/(d₁x + d₂). Truncated when the Poisson tail is negligible.   */
  function ncfSf(x, d1, d2, lam) {
    if (!(x > 0)) return 1;
    const u = d1 * x / (d1 * x + d2), hl = lam / 2;
    let s = 0, lw = -hl, i = 0;
    const kmax = Math.max(60, Math.ceil(hl + 12 * Math.sqrt(hl + 1)));
    for (i = 0; i <= kmax; i++) {
      if (i > 0) lw += Math.log(hl) - Math.log(i);
      const w = Math.exp(lw);
      if (w > 1e-16) s += w * (1 - ST.betaI(d1 / 2 + i, d2 / 2, u));
    }
    return ST.clamp(s, 0, 1);
  }
  const powerF = (k, n, f, alpha) => {
    const N = k * n, d1 = k - 1, d2 = N - k;
    if (d2 < 1) return NaN;
    return ncfSf(ST.fQuant(1 - alpha, d1, d2), d1, d2, N * f * f);
  };

  /* ── drawing: a boxplot in a band, used by four figures ───────────────── */
  function boxplot(g, vals, cx, halfw, y, color) {
    const b = ST.boxStats(vals);
    g.append("line").attr("x1", cx).attr("x2", cx).attr("y1", y(b.whiskLo)).attr("y2", y(b.whiskHi))
      .attr("stroke", color).attr("stroke-width", 1.2).attr("stroke-opacity", 0.75);
    [b.whiskLo, b.whiskHi].forEach(v => g.append("line")
      .attr("x1", cx - halfw * 0.45).attr("x2", cx + halfw * 0.45).attr("y1", y(v)).attr("y2", y(v))
      .attr("stroke", color).attr("stroke-width", 1.2).attr("stroke-opacity", 0.75));
    g.append("rect").attr("x", cx - halfw).attr("y", y(b.q3)).attr("width", 2 * halfw)
      .attr("height", Math.max(1, y(b.q1) - y(b.q3))).attr("rx", 3)
      .attr("fill", color).attr("fill-opacity", 0.17).attr("stroke", color).attr("stroke-width", 1.3);
    g.append("line").attr("x1", cx - halfw).attr("x2", cx + halfw).attr("y1", y(b.med)).attr("y2", y(b.med))
      .attr("stroke", color).attr("stroke-width", 2.4);
    b.out.forEach(v => g.append("circle").attr("cx", cx).attr("cy", y(v)).attr("r", 2.6)
      .attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 1.2));
    return b;
  }
  /* a small key/value readout panel drawn inside an svg */
  function panel(g, x, y, w, rows, title) {
    const gp = g.append("g").attr("transform", `translate(${x},${y})`);
    const h = 20 + rows.length * 16;
    gp.append("rect").attr("x", 0).attr("y", 0).attr("width", w).attr("height", h).attr("rx", 7)
      .attr("fill", SC.panel2).attr("stroke", SC.line);
    if (title) gp.append("text").attr("x", 9).attr("y", 14).attr("font-size", 10.5)
      .attr("fill", SC.accent).attr("font-weight", 600).text(title);
    rows.forEach((r, i) => {
      const yy = (title ? 30 : 16) + i * 16;
      gp.append("text").attr("x", 9).attr("y", yy).attr("font-size", 11).attr("fill", SC.muted).text(r[0]);
      gp.append("text").attr("x", w - 9).attr("y", yy).attr("text-anchor", "end").attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", r[2] || SC.ink).text(r[1]);
    });
    return gp;
  }
  const pfmt = p => (p < 1e-4 ? p.toExponential(2) : p.toFixed(4));

  return {
    RUN: RUN, TWOWAY: TWOWAY, GC: GC,
    anova: anova, Fstat: Fstat, welch: welch, brownForsythe: brownForsythe, kruskal: kruskal,
    QR: QR, holm: holm, holmAdj: holmAdj, bh: bh, sidak: sidak, harmonic: harmonic,
    draw: draw, ncfSf: ncfSf, powerF: powerF,
    boxplot: boxplot, panel: panel, pfmt: pfmt
  };
})();


/* ══════════ 1 · #pw-svg — the FWER of pairwise testing, measured ══════════ */
(function () {
  const svg = d3.select("#pw-svg");
  if (svg.empty()) return;
  const W = 740, H = 430;
  const kS = d3.select("#pw-k"), nS = d3.select("#pw-n"), aS = d3.select("#pw-a"), setS = d3.select("#pw-set");
  const PROCS = ["unadj", "bonf", "holm", "tukey", "scheffe", "F"];
  const LAB = { unadj: "unadjusted", bonf: "Bonferroni", holm: "Holm", tukey: "Tukey", scheffe: "Scheffé", F: "omnibus F" };
  const COL = { unadj: SC.bad, bonf: SC.accent, holm: SC.violet, tukey: SC.good, scheffe: SC.teal, F: SC.a2 };
  let rng = ST.rng(20260907);
  let counts = {}, trials = 0;                    // for the current (k, n, α)
  let curve = {};                                 // measured unadjusted FWER by k
  function reset() { counts = {}; PROCS.forEach(p => counts[p] = 0); trials = 0; }
  reset();

  function pairs(k) { const o = []; for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) o.push([i, j]); return o; }

  function batch(k, n, alpha, reps) {
    const N = k * n, df = N - k, C = k * (k - 1) / 2, P = pairs(k);
    const tc = ST.tQuant(1 - alpha / 2, df), tb = ST.tQuant(1 - alpha / (2 * C), df);
    const Fc = ST.fQuant(1 - alpha, k - 1, df), sc = Math.sqrt((k - 1) * Fc);
    const qc = AV.QR.qQuant(1 - alpha, k, df);
    for (let r = 0; r < reps; r++) {
      const gs = [];
      for (let j = 0; j < k; j++) { const g = []; for (let i = 0; i < n; i++) g.push(ST.randn(rng)); gs.push(g); }
      const m = gs.map(ST.mean);
      let ssw = 0;
      for (let j = 0; j < k; j++) { for (let i = 0; i < n; i++) { const d = gs[j][i] - m[j]; ssw += d * d; } }
      const msw = ssw / df, s = Math.sqrt(msw), se = s * Math.sqrt(2 / n);
      const gm = ST.mean(m);
      let ssb = 0; for (let j = 0; j < k; j++) ssb += n * (m[j] - gm) * (m[j] - gm);
      const F = (ssb / (k - 1)) / msw;
      let anyU = false, anyB = false, anyT = false, anyS = false;
      const ps = [];
      for (const [i, j] of P) {
        const t = Math.abs(m[i] - m[j]) / se;
        if (t > tc) anyU = true;
        if (t > tb) anyB = true;
        if (t > sc) anyS = true;
        if (Math.abs(m[i] - m[j]) / (s * Math.sqrt(1 / n)) > qc) anyT = true;
        ps.push(2 * (1 - ST.tCdf(t, df)));
      }
      if (anyU) counts.unadj++;
      if (anyB) counts.bonf++;
      if (anyT) counts.tukey++;
      if (anyS) counts.scheffe++;
      if (AV.holm(ps, alpha).some(Boolean)) counts.holm++;
      if (F > Fc) counts.F++;
      trials++;
    }
    curve[k] = counts.unadj / trials;
  }

  function draw() {
    const k = +kS.property("value"), n = +nS.property("value"), alpha = +aS.property("value");
    const all = setS.property("value") === "all";
    d3.select("#pw-kv").text(k); d3.select("#pw-nv").text(n);
    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 26, b: 46 });
    const g = f.g, halfW = 330, gapX = 60;
    const shown = all ? PROCS : ["unadj", "F"];
    const C = k * (k - 1) / 2, approx = 1 - Math.pow(1 - alpha, C);

    /* left: bars */
    const yMax = Math.max(0.08, Math.min(1, Math.max(approx, counts.unadj / Math.max(1, trials)) * 1.25));
    const x = d3.scaleBand().domain(shown).range([0, halfW]).padding(0.28);
    const y = d3.scaleLinear().domain([0, yMax]).range([f.ih, 0]);
    ST.gridY(g, y, halfW, 5);
    ST.axisL(g, y, 5, "measured FWER", d => ST.pct(d, 0));
    g.append("g").attr("transform", `translate(0,${f.ih})`).attr("class", "axis")
      .call(d3.axisBottom(x).tickFormat(d => LAB[d]))
      .selectAll("text").attr("font-size", 9.5).attr("transform", "rotate(-16)").attr("text-anchor", "end");
    shown.forEach(p => {
      const v = trials ? counts[p] / trials : 0;
      g.append("rect").attr("x", x(p)).attr("y", y(v)).attr("width", x.bandwidth())
        .attr("height", Math.max(0, f.ih - y(v))).attr("rx", 3)
        .attr("fill", COL[p]).attr("fill-opacity", 0.55).attr("stroke", COL[p]);
      const se = trials ? Math.sqrt(v * (1 - v) / trials) : 0;
      g.append("line").attr("x1", x(p) + x.bandwidth() / 2).attr("x2", x(p) + x.bandwidth() / 2)
        .attr("y1", y(Math.min(yMax, v + 2 * se))).attr("y2", y(Math.max(0, v - 2 * se)))
        .attr("stroke", SC.ink).attr("stroke-width", 1).attr("stroke-opacity", 0.6);
      g.append("text").attr("x", x(p) + x.bandwidth() / 2).attr("y", y(v) - 6).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", SC.ink).text(ST.pct(v, 1));
    });
    g.append("line").attr("x1", 0).attr("x2", halfW).attr("y1", y(alpha)).attr("y2", y(alpha))
      .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "5,4");
    g.append("text").attr("x", 4).attr("y", y(alpha) - 5).attr("font-size", 10).attr("fill", SC.good)
      .text("nominal " + ST.pct(alpha, 0));
    if (approx <= yMax) {
      g.append("line").attr("x1", x("unadj") - 6).attr("x2", x("unadj") + x.bandwidth() + 6)
        .attr("y1", y(approx)).attr("y2", y(approx))
        .attr("stroke", SC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "3,3");
      g.append("text").attr("x", x("unadj") + x.bandwidth() + 10).attr("y", y(approx) + 3.5)
        .attr("font-size", 9.5).attr("fill", SC.a2).text("1 − (1 − α)^C = " + ST.pct(approx, 1));
    }
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(`k = ${k}, C = ${C} comparisons, n = ${n} per group`);

    /* right: the curve against k */
    const x2 = d3.scaleLinear().domain([2, 10]).range([0, f.iw - halfW - gapX]);
    const y2 = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
    const g2 = g.append("g").attr("transform", `translate(${halfW + gapX},0)`);
    ST.gridY(g2, y2, f.iw - halfW - gapX, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x2).ticks(5));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5).tickFormat(d => ST.pct(d, 0)));
    g2.append("text").attr("x", x2.range()[1]).attr("y", f.ih + 32).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("number of groups k");
    const li = d3.line().x(d => x2(d[0])).y(d => y2(d[1]));
    const appr = d3.range(2, 11).map(kk => [kk, 1 - Math.pow(1 - alpha, kk * (kk - 1) / 2)]);
    g2.append("path").attr("d", li(appr)).attr("fill", "none").attr("stroke", SC.a2)
      .attr("stroke-width", 1.6).attr("stroke-dasharray", "4,4");
    const meas = Object.keys(curve).map(Number).sort((a, b) => a - b).map(kk => [kk, curve[kk]]);
    if (meas.length > 1) g2.append("path").attr("d", li(meas)).attr("fill", "none")
      .attr("stroke", SC.bad).attr("stroke-width", 2);
    meas.forEach(d => g2.append("circle").attr("cx", x2(d[0])).attr("cy", y2(d[1])).attr("r", 3).attr("fill", SC.bad));
    g2.append("line").attr("x1", 0).attr("x2", x2.range()[1]).attr("y1", y2(alpha)).attr("y2", y2(alpha))
      .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "5,4");
    ST.legend(g2, [
      { label: "measured, unadjusted pairwise t", color: SC.bad },
      { label: "1 − (1 − α)^C, assuming independence", color: SC.a2, dash: "4,4" },
      { label: "the omnibus F, at every k", color: SC.good, dash: "5,4" }
    ], 8, 14);
    g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text("the unadjusted rate as k grows — move the k slider to fill it in");

    const v = trials ? counts.unadj / trials : 0;
    d3.select("#pw-readout").html(
      `<b>${trials.toLocaleString()}</b> simulated experiments at k = ${k}, n = ${n}, α = ${alpha}. ` +
      `Every group is Normal(0, 1), so <b>every</b> rejection is an error. ` +
      `Unadjusted pairwise t: <b>${ST.pct(v, 2)}</b>; the independence formula says ${ST.pct(approx, 2)} ` +
      `(<b>${trials ? ST.pct(approx / Math.max(v, 1e-9) - 1, 0) : "—"}</b> too high — the comparisons are positively dependent). ` +
      `The omnibus F: <b>${ST.pct(trials ? counts.F / trials : 0, 2)}</b>.`);
  }
  function run() { batch(+kS.property("value"), +nS.property("value"), +aS.property("value"), 6000); draw(); }
  [kS, nS, aS].forEach(s => s.on("input change", () => { reset(); run(); }));
  setS.on("change", draw);
  d3.select("#pw-run").on("click", run);
  d3.select("#pw-reset").on("click", () => { rng = ST.rng(20260907); curve = {}; reset(); run(); });
  run();
})();

/* ══════════ 2 · #dec-svg — one observation, three pieces ══════════ */
(function () {
  const svg = d3.select("#dec-svg");
  if (svg.empty()) return;
  const W = 740, H = 470;
  const dataS = d3.select("#dec-data"), sS = d3.select("#dec-s"), viewS = d3.select("#dec-view");
  let picked = null;

  function base(kind) {
    const r = ST.rng(4242);
    if (kind === "run") return AV.RUN.y.map(a => a.slice());
    const out = [];
    const mu = kind === "sep" ? [44, 50, 56] : kind === "one" ? [47, 47, 55] : [48, 48, 48];
    for (let j = 0; j < 3; j++) { const g = []; for (let i = 0; i < 13; i++) g.push(mu[j] + 4.4 * ST.randn(r)); out.push(g); }
    return out;
  }
  function stretched(kind, mult) {
    const gs = base(kind);
    const flat = [].concat.apply([], gs), gm = ST.mean(flat);
    return gs.map(g => { const m = ST.mean(g); return g.map(v => gm + (m - gm) * mult + (v - m)); });
  }

  function draw() {
    const kind = dataS.property("value"), mult = +sS.property("value") / 100;
    d3.select("#dec-sv").text(mult.toFixed(2) + "×");
    const view = viewS.property("value");
    const gs = stretched(kind, mult), a = AV.anova(gs);
    const f = ST.frame(svg, W, H, { l: 44, r: 14, t: 26, b: 40 });
    const g = f.g, LW = 350, gap = 56;
    const flat = [].concat.apply([], gs);
    const y = d3.scaleLinear().domain([d3.min(flat) - 2, d3.max(flat) + 2]).nice().range([f.ih, 0]);
    const bx = d3.scalePoint().domain([0, 1, 2]).range([46, LW - 30]).padding(0.4);
    ST.gridY(g, y, LW, 6);
    ST.axisL(g, y, 6, "score");
    /* grand mean */
    g.append("line").attr("x1", 0).attr("x2", LW).attr("y1", y(a.gm)).attr("y2", y(a.gm))
      .attr("stroke", SC.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "6,4");
    g.append("text").attr("x", 2).attr("y", y(a.gm) - 5).attr("font-size", 10).attr("fill", SC.muted)
      .text("grand mean ȳ̄ = " + ST.fmt(a.gm, 2));
    gs.forEach((grp, j) => {
      const cx = bx(j);
      const jit = ST.jitterY(grp, 11 + j, 13);
      /* group-effect segment: grand mean → group mean */
      g.append("line").attr("x1", cx - 26).attr("x2", cx - 26).attr("y1", y(a.gm)).attr("y2", y(a.means[j]))
        .attr("stroke", SC.a2).attr("stroke-width", 4).attr("stroke-opacity", 0.75).attr("stroke-linecap", "round");
      g.append("line").attr("x1", cx - 34).attr("x2", cx + 34).attr("y1", y(a.means[j])).attr("y2", y(a.means[j]))
        .attr("stroke", AV.GC[j]).attr("stroke-width", 2.4);
      grp.forEach((v, i) => {
        const px = cx + jit[i];
        g.append("line").attr("x1", px).attr("x2", px).attr("y1", y(a.means[j])).attr("y2", y(v))
          .attr("stroke", SC.teal).attr("stroke-width", 1).attr("stroke-opacity", 0.42);
        g.append("circle").attr("cx", px).attr("cy", y(v)).attr("r", picked && picked[0] === j && picked[1] === i ? 5 : 3)
          .attr("fill", AV.GC[j]).attr("fill-opacity", 0.8)
          .attr("stroke", picked && picked[0] === j && picked[1] === i ? SC.ink : "none").attr("stroke-width", 1.6)
          .style("cursor", "pointer").on("click", () => { picked = [j, i]; draw(); });
      });
      g.append("text").attr("x", cx).attr("y", f.ih + 16).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AV.GC[j]).text("group " + (j + 1));
    });
    ST.legend(g, [
      { label: "group effect  ȳⱼ − ȳ̄", color: SC.a2 },
      { label: "residual  yᵢⱼ − ȳⱼ", color: SC.teal }
    ], 4, 12);

    /* right panel */
    const RW = f.iw - LW - gap, g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    if (view === "stack") {
      const pts = [];
      gs.forEach((grp, j) => grp.forEach((v, i) => pts.push({ j: j, i: i, eff: a.means[j] - a.gm, res: v - a.means[j] })));
      const xs = d3.scaleLinear().domain([0, pts.length - 1]).range([0, RW]);
      const lim = d3.max(pts, d => Math.abs(d.eff) + Math.abs(d.res)) * 1.08;
      const ys = d3.scaleLinear().domain([-lim, lim]).range([f.ih, 0]);
      ST.gridY(g2, ys, RW, 5);
      g2.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
      g2.append("line").attr("x1", 0).attr("x2", RW).attr("y1", ys(0)).attr("y2", ys(0))
        .attr("stroke", SC.muted).attr("stroke-width", 1.2);
      const bw = Math.max(1.5, RW / pts.length - 1.2);
      pts.forEach((d, idx) => {
        const px = xs(idx);
        const sel = picked && picked[0] === d.j && picked[1] === d.i;
        g2.append("rect").attr("x", px - bw / 2).attr("y", ys(Math.max(0, d.eff))).attr("width", bw)
          .attr("height", Math.abs(ys(d.eff) - ys(0))).attr("fill", SC.a2).attr("fill-opacity", sel ? 1 : 0.6);
        const b0 = d.eff, b1 = d.eff + d.res;
        g2.append("rect").attr("x", px - bw / 2).attr("y", ys(Math.max(b0, b1))).attr("width", bw)
          .attr("height", Math.abs(ys(b1) - ys(b0))).attr("fill", SC.teal).attr("fill-opacity", sel ? 1 : 0.45);
      });
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text("each observation's deviation from ȳ̄, split into its two pieces");
      if (picked) {
        const d = pts.find(p => p.j === picked[0] && p.i === picked[1]);
        if (d) AV.panel(g2, RW - 210, f.ih - 92, 210, [
          ["yᵢⱼ", ST.fmt(a.gm + d.eff + d.res, 4)],
          ["ȳ̄  (grand)", ST.fmt(a.gm, 4)],
          ["ȳⱼ − ȳ̄  (effect)", ST.fmt(d.eff, 4), SC.a2],
          ["yᵢⱼ − ȳⱼ (residual)", ST.fmt(d.res, 4), SC.teal]
        ], "the point you clicked");
      }
    } else if (view === "squares") {
      const tot = a.sst, side = Math.min(RW, f.ih) * 0.94;
      const sc = side / Math.sqrt(tot);
      const bs = Math.sqrt(a.ssb) * sc, ws = Math.sqrt(a.ssw) * sc;
      const x0 = (RW - side) / 2, y0 = f.ih - side;
      g2.append("rect").attr("x", x0).attr("y", y0).attr("width", side).attr("height", side)
        .attr("fill", SC.muted).attr("fill-opacity", 0.10).attr("stroke", SC.muted).attr("stroke-dasharray", "4,3");
      g2.append("text").attr("x", x0 + side - 4).attr("y", y0 - 6).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.muted).text("SS_total = " + ST.fmt(tot, 2));
      g2.append("rect").attr("x", x0).attr("y", f.ih - ws).attr("width", ws).attr("height", ws)
        .attr("fill", SC.teal).attr("fill-opacity", 0.22).attr("stroke", SC.teal).attr("stroke-width", 1.5);
      g2.append("text").attr("x", x0 + 6).attr("y", f.ih - ws + 14).attr("font-size", 10.5).attr("fill", SC.teal)
        .text("SS_within " + ST.fmt(a.ssw, 1));
      g2.append("rect").attr("x", x0 + side - bs).attr("y", y0).attr("width", bs).attr("height", bs)
        .attr("fill", SC.a2).attr("fill-opacity", 0.30).attr("stroke", SC.a2).attr("stroke-width", 1.5);
      g2.append("text").attr("x", x0 + side - bs - 6).attr("y", y0 + 12).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.a2).text("SS_between " + ST.fmt(a.ssb, 1));
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text("areas, not lengths: the two squares' areas add to the outline's");
    } else {
      const bl = Math.sqrt(a.ssb), wl = Math.sqrt(a.ssw), hl = Math.sqrt(a.sst);
      const sc = Math.min(RW * 0.8 / Math.max(wl, 1e-9), f.ih * 0.72 / Math.max(bl, 1e-9), 400 / Math.max(hl, 1e-9));
      const ox = 30, oy = f.ih - 40;
      const px = ox + wl * sc, py = oy - bl * sc;
      g2.append("line").attr("x1", ox).attr("y1", oy).attr("x2", px).attr("y2", oy)
        .attr("stroke", SC.teal).attr("stroke-width", 3);
      g2.append("line").attr("x1", px).attr("y1", oy).attr("x2", px).attr("y2", py)
        .attr("stroke", SC.a2).attr("stroke-width", 3);
      g2.append("line").attr("x1", ox).attr("y1", oy).attr("x2", px).attr("y2", py)
        .attr("stroke", SC.muted).attr("stroke-width", 2).attr("stroke-dasharray", "5,4");
      g2.append("rect").attr("x", px - 12).attr("y", oy - 12).attr("width", 12).attr("height", 12)
        .attr("fill", "none").attr("stroke", SC.line);
      g2.append("text").attr("x", (ox + px) / 2).attr("y", oy + 18).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", SC.teal).text("√SS_within = " + ST.fmt(wl, 2));
      g2.append("text").attr("x", px + 8).attr("y", (oy + py) / 2).attr("font-size", 10.5).attr("fill", SC.a2)
        .text("√SS_between = " + ST.fmt(bl, 2));
      g2.append("text").attr("x", (ox + px) / 2 - 40).attr("y", (oy + py) / 2 - 8).attr("font-size", 10.5)
        .attr("fill", SC.muted).text("√SS_total = " + ST.fmt(hl, 2));
      const ang = Math.atan2(bl, wl) * 180 / Math.PI;
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(`the angle to the within-group leg is ${ST.fmt(ang, 1)}° — it is arccos √(1 − η²)`);
    }
    d3.select("#dec-readout").html(
      `SS_total <b>${ST.fmt(a.sst, 4)}</b> = SS_between <b>${ST.fmt(a.ssb, 4)}</b> + SS_within <b>${ST.fmt(a.ssw, 4)}</b> ` +
      `(residual ${(a.ssb + a.ssw - a.sst).toExponential(1)}). ` +
      `df ${a.df1} + ${a.df2} = ${a.N - 1}. MS ${ST.fmt(a.msb, 4)} / ${ST.fmt(a.msw, 4)}. ` +
      `<b>F = ${ST.fmt(a.F, 4)}</b> on (${a.df1}, ${a.df2}), p = <b>${AV.pfmt(a.p)}</b>, η² = ${ST.fmt(a.eta2, 4)}.`);
  }
  [dataS, sS, viewS].forEach(s => s.on("input change", () => { if (s === dataS) picked = null; draw(); }));
  d3.select("#dec-reset").on("click", () => { sS.property("value", 100); picked = null; draw(); });
  draw();
})();

/* ══════════ 3 · #fd-svg — the F density, its df, and its tail ══════════ */
(function () {
  const svg = d3.select("#fd-svg");
  if (svg.empty()) return;
  const W = 740, H = 420;
  const d1S = d3.select("#fd-d1"), d2S = d3.select("#fd-d2"), fS = d3.select("#fd-f"), vS = d3.select("#fd-view");
  function draw() {
    const d1 = +d1S.property("value"), d2 = +d2S.property("value"), Fobs = +fS.property("value") / 100;
    const view = vS.property("value");
    d3.select("#fd-d1v").text(d1); d3.select("#fd-d2v").text(d2); d3.select("#fd-fv").text(Fobs.toFixed(3));
    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 24, b: 44 });
    const g = f.g;
    const crit = ST.fQuant(0.95, d1, d2), p = 1 - ST.fCdf(Fobs, d1, d2);
    const xmax = Math.max(6, crit * 1.9, Fobs * 1.25);
    if (view === "chi") {
      /* the two chi-squares, side by side, on their own scales */
      const cells = ST.cells(f.iw, f.ih, 2, 1, { l: 40, r: 10, t: 22, b: 34 }, { x: 40, y: 0 });
      [[d1, "SS_between/σ² ~ χ²(" + d1 + ")", SC.a2], [d2, "SS_within/σ² ~ χ²(" + d2 + ")", SC.teal]].forEach((c, ci) => {
        const cl = cells[ci], gc = g.append("g").attr("transform", `translate(${cl.x + cl.m.l},${cl.y + cl.m.t})`);
        const hi = ST.chi2Quant(0.999, c[0]) * 1.05;
        const xs = ST.linspace(0.0001, hi, 260);
        const x = d3.scaleLinear().domain([0, hi]).range([0, cl.iw]);
        const dens = xs.map(v => ST.chi2Pdf(v, c[0]));
        const y = d3.scaleLinear().domain([0, d3.max(dens) * 1.12]).range([cl.ih, 0]);
        ST.gridY(gc, y, cl.iw, 4);
        gc.append("g").attr("class", "axis").attr("transform", `translate(0,${cl.ih})`).call(d3.axisBottom(x).ticks(5));
        gc.append("path").attr("d", d3.area().x((v, i) => x(xs[i])).y0(cl.ih).y1((v, i) => y(dens[i]))(xs))
          .attr("fill", c[2]).attr("fill-opacity", 0.18);
        gc.append("path").attr("d", d3.line().x((v, i) => x(xs[i])).y((v, i) => y(dens[i]))(xs))
          .attr("fill", "none").attr("stroke", c[2]).attr("stroke-width", 2);
        gc.append("line").attr("x1", x(c[0])).attr("x2", x(c[0])).attr("y1", cl.ih).attr("y2", 0)
          .attr("stroke", SC.muted).attr("stroke-dasharray", "4,3");
        gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", c[2]).text(c[1]);
        gc.append("text").attr("x", x(c[0]) + 5).attr("y", 12).attr("font-size", 9.5).attr("fill", SC.muted)
          .text("mean = df = " + c[0]);
      });
      g.append("text").attr("x", 0).attr("y", f.ih + 34).attr("font-size", 11).attr("fill", SC.muted)
        .text("independent by Cochran's theorem, because (k−1) + (N−k) = N−1 — and that independence is what makes the ratio an F");
    } else if (view === "sim") {
      const rng = ST.rng(777), n = 6000, out = [];
      for (let i = 0; i < n; i++) {
        let u = 0, v = 0;
        for (let j = 0; j < d1; j++) { const z = ST.randn(rng); u += z * z; }
        for (let j = 0; j < d2; j++) { const z = ST.randn(rng); v += z * z; }
        out.push((u / d1) / (v / d2));
      }
      const bw = xmax / 60, bins = ST.histBins(out, 0, bw, 60);
      const x = d3.scaleLinear().domain([0, xmax]).range([0, f.iw]);
      const dens = ST.linspace(0.001, xmax, 300).map(v => ST.fPdf(v, d1, d2));
      const y = d3.scaleLinear().domain([0, Math.max(d3.max(bins, b => b.density), d3.max(dens)) * 1.1]).range([f.ih, 0]);
      ST.gridY(g, y, f.iw, 5);
      ST.axisB(g, x, f.ih, 7, "F"); ST.axisL(g, y, 5, "density");
      g.selectAll("rect.h").data(bins).join("rect").attr("class", "h")
        .attr("x", b => x(b.x0)).attr("y", b => y(b.density)).attr("width", Math.max(1, x(bw) - x(0) - 1))
        .attr("height", b => f.ih - y(b.density)).attr("fill", SC.accent).attr("fill-opacity", 0.32);
      const xs = ST.linspace(0.001, xmax, 300);
      g.append("path").attr("d", d3.line().x(v => x(v)).y(v => y(ST.fPdf(v, d1, d2)))(xs))
        .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
      ST.legend(g, [{ label: `${n.toLocaleString()} simulated ratios of two chi-squares`, color: SC.accent },
                    { label: `the F(${d1}, ${d2}) density`, color: SC.a2 }], f.iw - 250, 14);
    } else {
      const xs = ST.linspace(0.0005, xmax, 420);
      const dens = xs.map(v => ST.fPdf(v, d1, d2));
      const x = d3.scaleLinear().domain([0, xmax]).range([0, f.iw]);
      const cap = d3.max(dens.filter((v, i) => xs[i] > 0.12));
      const y = d3.scaleLinear().domain([0, (isFinite(cap) ? cap : 1) * 1.15]).range([f.ih, 0]);
      ST.gridY(g, y, f.iw, 5);
      ST.axisB(g, x, f.ih, 7, "F"); ST.axisL(g, y, 5, "density");
      const tail = xs.filter(v => v >= Fobs);
      if (tail.length) g.append("path")
        .attr("d", d3.area().x(v => x(v)).y0(f.ih).y1(v => y(ST.fPdf(v, d1, d2)))(tail))
        .attr("fill", SC.bad).attr("fill-opacity", 0.30);
      g.append("path").attr("d", d3.line().x(v => x(v)).y(v => y(ST.fPdf(v, d1, d2)))(xs))
        .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.2);
      g.append("line").attr("x1", x(crit)).attr("x2", x(crit)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.good).attr("stroke-width", 1.5).attr("stroke-dasharray", "5,4");
      g.append("text").attr("x", x(crit) + 5).attr("y", 26).attr("font-size", 10).attr("fill", SC.good)
        .text("F* at 5% = " + ST.fmt(crit, 4));
      g.append("line").attr("x1", x(Fobs)).attr("x2", x(Fobs)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.bad).attr("stroke-width", 2);
      g.append("text").attr("x", x(Fobs) + 5).attr("y", 46).attr("font-size", 10.5).attr("fill", SC.bad)
        .text("observed F = " + ST.fmt(Fobs, 4));
      if (d2 > 2) {
        const mu = d2 / (d2 - 2);
        g.append("line").attr("x1", x(mu)).attr("x2", x(mu)).attr("y1", f.ih).attr("y2", f.ih - 16)
          .attr("stroke", SC.muted).attr("stroke-width", 1.4);
        g.append("text").attr("x", x(mu)).attr("y", f.ih - 20).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", SC.muted).text("mean " + ST.fmt(mu, 3));
      }
    }
    d3.select("#fd-readout").html(
      `F(${d1}, ${d2}): mean ${d2 > 2 ? ST.fmt(d2 / (d2 - 2), 4) : "undefined (needs d₂ > 2)"}, ` +
      `5% critical value <b>${ST.fmt(crit, 5)}</b>. Observed F = ${ST.fmt(Fobs, 4)} gives ` +
      `p = <b>${AV.pfmt(p)}</b>. ` +
      (d1 === 1 ? `With d₁ = 1 this is a t² on ${d2} df: √F = ${ST.fmt(Math.sqrt(Fobs), 4)} and t*₍${d2},0.975₎ = ${ST.fmt(ST.tQuant(0.975, d2), 4)}, whose square is ${ST.fmt(Math.pow(ST.tQuant(0.975, d2), 2), 4)} = F*.` : ""));
  }
  [d1S, d2S, fS, vS].forEach(s => s.on("input change", draw));
  d3.select("#fd-run").on("click", () => {
    d1S.property("value", 2); d2S.property("value", 38); fS.property("value", 258); vS.property("value", "dens"); draw();
  });
  draw();
})();

/* ══════════ 4 · #lab-svg — the laboratory ══════════ */
(function () {
  const svg = d3.select("#lab-svg");
  if (svg.empty()) return;
  const W = 740, H = 470;
  const kS = d3.select("#lb-k"), nS = d3.select("#lb-n"), dS = d3.select("#lb-d"),
        sS = d3.select("#lb-s"), vS = d3.select("#lb-view");
  /* The default draw is seeded, and the seed was CHOSEN so that the figure makes
     §07's point: three groups of thirty at δ = 3.5, σ = 10 give boxplots that
     overlap in all three pairs and an F of 3.9228 on (2, 87), p = 0.0234 — visibly
     nothing, and significant. A seed that produced p = 0.0004 would have made the
     same point dishonestly, and one that produced p = 0.3 would have contradicted
     the caption. */
  const DEF = { k: 3, n: 30, d: 35, s: 100, seed: 20260907 };
  let seed = DEF.seed, useRun = false;

  function gen(k, n, delta, sigma, sd) {
    const r = ST.rng(sd), out = [];
    for (let j = 0; j < k; j++) {
      const mu = 50 + delta * (j - (k - 1) / 2), g = [];
      for (let i = 0; i < n; i++) g.push(mu + sigma * ST.randn(r));
      out.push(g);
    }
    return out;
  }
  function draw() {
    const k = +kS.property("value"), n = +nS.property("value");
    const delta = +dS.property("value") / 10, sigma = +sS.property("value") / 10;
    d3.select("#lb-kv").text(k); d3.select("#lb-nv").text(n);
    d3.select("#lb-dv").text(delta.toFixed(1)); d3.select("#lb-sv").text(sigma.toFixed(1));
    const view = vS.property("value");
    const gs = useRun ? AV.RUN.y.map(a => a.slice()) : gen(k, n, delta, sigma, seed);
    const a = AV.anova(gs);
    const names = useRun ? AV.RUN.short : d3.range(a.k).map(j => "g" + (j + 1));
    const f = ST.frame(svg, W, H, { l: 44, r: 14, t: 26, b: 42 });
    const g = f.g, LW = 300, MW = 250, gap = 42;
    const flat = [].concat.apply([], gs);
    const lo = d3.min(flat), hi = d3.max(flat), pad = (hi - lo) * 0.07 + 1e-9;
    const y = d3.scaleLinear().domain([lo - pad, hi + pad]).nice().range([f.ih, 0]);
    const bx = d3.scalePoint().domain(d3.range(a.k)).range([44, LW - 26]).padding(0.55);
    ST.gridY(g, y, LW, 6); ST.axisL(g, y, 6, "outcome");
    g.append("line").attr("x1", 0).attr("x2", LW).attr("y1", y(a.gm)).attr("y2", y(a.gm))
      .attr("stroke", SC.muted).attr("stroke-width", 1.2).attr("stroke-dasharray", "6,4");
    const hw = Math.min(24, (LW - 70) / (2.6 * a.k));
    gs.forEach((grp, j) => {
      const cx = bx(j), jit = ST.jitterY(grp, 31 + j, hw * 0.75);
      grp.forEach((v, i) => g.append("circle").attr("cx", cx + jit[i]).attr("cy", y(v)).attr("r", 1.9)
        .attr("fill", AV.GC[j % 6]).attr("fill-opacity", 0.42));
      AV.boxplot(g, grp, cx, hw, y, AV.GC[j % 6]);
      g.append("text").attr("x", cx).attr("y", f.ih + 15).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", AV.GC[j % 6]).text(names[j]);
    });
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text("what the eye sees: the spread of the DATA");

    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    if (view === "means") {
      const se = a.means.map((m, j) => a.s / Math.sqrt(a.ns[j]));
      const loM = d3.min(a.means.map((m, j) => m - 3 * se[j])), hiM = d3.max(a.means.map((m, j) => m + 3 * se[j]));
      const pd = (hiM - loM) * 0.28 + 1e-9;
      const y2 = d3.scaleLinear().domain([loM - pd, hiM + pd]).nice().range([f.ih, 0]);
      const bx2 = d3.scalePoint().domain(d3.range(a.k)).range([44, MW - 26]).padding(0.55);
      ST.gridY(g2, y2, MW, 6);
      g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(6));
      g2.append("line").attr("x1", 0).attr("x2", MW).attr("y1", y2(a.gm)).attr("y2", y2(a.gm))
        .attr("stroke", SC.muted).attr("stroke-width", 1.2).attr("stroke-dasharray", "6,4");
      a.means.forEach((m, j) => {
        const cx = bx2(j);
        g2.append("line").attr("x1", cx).attr("x2", cx).attr("y1", y2(m - se[j])).attr("y2", y2(m + se[j]))
          .attr("stroke", AV.GC[j % 6]).attr("stroke-width", 3.2).attr("stroke-linecap", "round");
        g2.append("circle").attr("cx", cx).attr("cy", y2(m)).attr("r", 4.5).attr("fill", AV.GC[j % 6]);
        g2.append("text").attr("x", cx).attr("y", f.ih + 15).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", AV.GC[j % 6]).text(names[j]);
      });
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text("what the TEST sees: the means ± 1 SE");
      g2.append("text").attr("x", 0).attr("y", 12).attr("font-size", 9.5).attr("fill", SC.a2)
        .text("note the axis: this panel is magnified " +
          ST.fmt((y.domain()[1] - y.domain()[0]) / (y2.domain()[1] - y2.domain()[0]), 1) + "×");
    } else if (view === "dens") {
      const crit = ST.fQuant(0.95, a.df1, a.df2), xmax = Math.max(6, crit * 1.7, a.F * 1.2);
      const xs = ST.linspace(0.002, xmax, 300);
      const x2 = d3.scaleLinear().domain([0, xmax]).range([0, MW]);
      const cap = d3.max(xs.filter(v => v > 0.15).map(v => ST.fPdf(v, a.df1, a.df2)));
      const y2 = d3.scaleLinear().domain([0, cap * 1.15]).range([f.ih, 0]);
      ST.gridY(g2, y2, MW, 5);
      g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x2).ticks(5));
      const tail = xs.filter(v => v >= a.F);
      if (tail.length) g2.append("path")
        .attr("d", d3.area().x(v => x2(v)).y0(f.ih).y1(v => y2(ST.fPdf(v, a.df1, a.df2)))(tail))
        .attr("fill", SC.bad).attr("fill-opacity", 0.3);
      g2.append("path").attr("d", d3.line().x(v => x2(v)).y(v => y2(ST.fPdf(v, a.df1, a.df2)))(xs))
        .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
      g2.append("line").attr("x1", x2(Math.min(a.F, xmax))).attr("x2", x2(Math.min(a.F, xmax)))
        .attr("y1", f.ih).attr("y2", 0).attr("stroke", SC.bad).attr("stroke-width", 2);
      g2.append("line").attr("x1", x2(crit)).attr("x2", x2(crit)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.good).attr("stroke-dasharray", "5,4");
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(`the null density F(${a.df1}, ${a.df2}) and the observed value`);
    } else {
      const bars = [{ n: "SS_between", v: a.ssb, c: SC.a2 }, { n: "SS_within", v: a.ssw, c: SC.teal }];
      const x2 = d3.scaleBand().domain(bars.map(b => b.n)).range([0, MW]).padding(0.35);
      const y2 = d3.scaleLinear().domain([0, d3.max(bars, b => b.v) * 1.15]).range([f.ih, 0]);
      ST.gridY(g2, y2, MW, 5);
      g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5));
      g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x2));
      bars.forEach(b => {
        g2.append("rect").attr("x", x2(b.n)).attr("y", y2(b.v)).attr("width", x2.bandwidth())
          .attr("height", f.ih - y2(b.v)).attr("rx", 4).attr("fill", b.c).attr("fill-opacity", 0.5).attr("stroke", b.c);
        g2.append("text").attr("x", x2(b.n) + x2.bandwidth() / 2).attr("y", y2(b.v) - 6)
          .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", SC.ink).text(ST.fmt(b.v, 1));
      });
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text("the two sums of squares — before dividing by df");
    }
    AV.panel(g, f.iw - 176, 6, 176, [
      ["SS_between", ST.fmt(a.ssb, 2), SC.a2],
      ["SS_within", ST.fmt(a.ssw, 2), SC.teal],
      ["MS_between", ST.fmt(a.msb, 2)],
      ["MS_within", ST.fmt(a.msw, 2)],
      ["F", ST.fmt(a.F, 4), a.p < 0.05 ? SC.good : SC.ink],
      ["df", a.df1 + ", " + a.df2],
      ["p", AV.pfmt(a.p), a.p < 0.05 ? SC.good : SC.bad],
      ["η²", ST.fmt(a.eta2, 4)]
    ], "the ANOVA table");
    const seRep = a.s / Math.sqrt(a.ns[0]);
    d3.select("#lab-readout").html(
      (useRun ? "The running exam scores. " : `${a.k} groups of ${a.ns[0]}, σ = ${sigma.toFixed(1)}, means ${delta.toFixed(1)} apart. `) +
      `The data's SD is <b>${ST.fmt(a.s, 2)}</b>; a group mean's SE is <b>${ST.fmt(seRep, 2)}</b> — smaller by √n = ${ST.fmt(Math.sqrt(a.ns[0]), 2)}. ` +
      `F = <b>${ST.fmt(a.F, 4)}</b> on (${a.df1}, ${a.df2}), p = <b>${AV.pfmt(a.p)}</b> ` +
      `— ${a.p < 0.05 ? "<b>significant at 5%</b>" : "not significant at 5%"}. ` +
      (!useRun && delta > 0 ? `The population separation is ${ST.fmt(delta / sigma, 2)} data SDs (invisible) and ${ST.fmt(delta * Math.sqrt(a.ns[0]) / sigma, 2)} standard errors (not).` : ""));
  }
  [kS, nS, dS, sS].forEach(s => s.on("input", () => { useRun = false; draw(); }));
  vS.on("change", draw);
  d3.select("#lb-new").on("click", () => { useRun = false; seed = (seed * 1103515245 + 12345) >>> 0; draw(); });
  d3.select("#lb-run").on("click", () => { useRun = true; draw(); });
  d3.select("#lb-reset").on("click", () => {
    useRun = false; seed = DEF.seed;
    kS.property("value", DEF.k); nS.property("value", DEF.n);
    dS.property("value", DEF.d); sS.property("value", DEF.s); vS.property("value", "means");
    draw();
  });
  draw();
})();

/* ══════════ 5 · #rb-svg — the F test's true level under violations ══════════ */
(function () {
  const svg = d3.select("#rb-svg");
  if (svg.empty()) return;
  const W = 740, H = 400;
  const dS = d3.select("#rb-dist"), nsS = d3.select("#rb-ns");
  let rng = ST.rng(5150), rej = 0, tot = 0, Fs = [];
  const NAMES = { norm: "normal", unif: "uniform", exp: "exponential", lnorm: "lognormal", t3: "t on 3 df", cauchy: "Cauchy", bern: "Bernoulli(0.1)" };
  function reset() { rej = 0; tot = 0; Fs = []; }
  function sizes() { return nsS.property("value").split(",").map(Number); }
  function batch(reps) {
    const kind = dS.property("value"), ns = sizes(), k = ns.length;
    const N = ns.reduce((a, b) => a + b, 0), crit = ST.fQuant(0.95, k - 1, N - k);
    for (let r = 0; r < reps; r++) {
      const gs = ns.map(n => { const g = []; for (let i = 0; i < n; i++) g.push(AV.draw(kind, rng)); return g; });
      const F = AV.Fstat(gs);
      if (isFinite(F)) { tot++; if (F > crit) rej++; if (Fs.length < 40000) Fs.push(F); }
    }
  }
  function draw() {
    const kind = dS.property("value"), ns = sizes(), k = ns.length;
    const N = ns.reduce((a, b) => a + b, 0), d1 = k - 1, d2 = N - k;
    const crit = ST.fQuant(0.95, d1, d2);
    const f = ST.frame(svg, W, H, { l: 44, r: 14, t: 26, b: 42 });
    const g = f.g, LW = 250, gap = 54;
    /* left: one sample, plus the parent shape */
    const r2 = ST.rng(31415);
    const gs = ns.map(n => { const gg = []; for (let i = 0; i < n; i++) gg.push(AV.draw(kind, r2)); return gg; });
    const flat = [].concat.apply([], gs);
    const y = d3.scaleLinear().domain([d3.min(flat), d3.max(flat)]).nice().range([f.ih, 0]);
    ST.gridY(g, y, LW, 5); ST.axisL(g, y, 5, "one null sample");
    const bx = d3.scalePoint().domain(d3.range(k)).range([50, LW - 30]).padding(0.55);
    gs.forEach((grp, j) => {
      const jit = ST.jitterY(grp, 9 + j, 12);
      grp.forEach((v, i) => g.append("circle").attr("cx", bx(j) + jit[i]).attr("cy", y(v)).attr("r", 2.2)
        .attr("fill", AV.GC[j % 6]).attr("fill-opacity", 0.5));
      g.append("line").attr("x1", bx(j) - 16).attr("x2", bx(j) + 16).attr("y1", y(ST.mean(grp))).attr("y2", y(ST.mean(grp)))
        .attr("stroke", AV.GC[j % 6]).attr("stroke-width", 2.2);
      g.append("text").attr("x", bx(j)).attr("y", f.ih + 15).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", AV.GC[j % 6]).text("n = " + ns[j]);
    });
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(NAMES[kind] + " errors, all group means EQUAL");
    /* right: the accumulated F histogram against the theoretical density */
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    const xmax = Math.max(6, crit * 1.8);
    const x = d3.scaleLinear().domain([0, xmax]).range([0, RW]);
    const bw = xmax / 44;
    const inrange = Fs.filter(v => v <= xmax);
    const bins = Fs.length ? ST.histBins(inrange, 0, bw, 44) : [];
    const scaleFix = Fs.length ? inrange.length / Fs.length : 1;
    const xs = ST.linspace(0.01, xmax, 240), dens = xs.map(v => ST.fPdf(v, d1, d2));
    const y2 = d3.scaleLinear().domain([0, Math.max(d3.max(dens) || 1, bins.length ? d3.max(bins, b => b.density * scaleFix) : 0) * 1.12]).range([f.ih, 0]);
    ST.gridY(g2, y2, RW, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(6));
    g2.selectAll("rect.h").data(bins).join("rect").attr("class", "h")
      .attr("x", b => x(b.x0)).attr("y", b => y2(b.density * scaleFix))
      .attr("width", Math.max(1, x(bw) - x(0) - 1)).attr("height", b => Math.max(0, f.ih - y2(b.density * scaleFix)))
      .attr("fill", SC.accent).attr("fill-opacity", 0.3);
    g2.append("path").attr("d", d3.line().x(v => x(v)).y(v => y2(ST.fPdf(v, d1, d2)))(xs))
      .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
    g2.append("line").attr("x1", x(crit)).attr("x2", x(crit)).attr("y1", f.ih).attr("y2", 0)
      .attr("stroke", SC.good).attr("stroke-width", 1.5).attr("stroke-dasharray", "5,4");
    g2.append("text").attr("x", x(crit) + 5).attr("y", 40).attr("font-size", 10).attr("fill", SC.good)
      .text("F* = " + ST.fmt(crit, 3));
    const rate = tot ? rej / tot : 0, se = tot ? Math.sqrt(rate * (1 - rate) / tot) : 0;
    const col = Math.abs(rate - 0.05) < 2 * se ? SC.good : SC.bad;
    g2.append("text").attr("x", RW - 4).attr("y", 16).attr("text-anchor", "end").attr("font-size", 24)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col).text(ST.pct(rate, 2));
    g2.append("text").attr("x", RW - 4).attr("y", 32).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", SC.muted).text(`measured level ± ${ST.pct(2 * se, 2)} · nominal 5%`);
    g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(`${tot.toLocaleString()} null experiments · F(${d1}, ${d2}) drawn over them`);
    d3.select("#rb-readout").html(
      `<b>${NAMES[kind]}</b> errors, group sizes ${ns.join(", ")}. Measured type I error over ` +
      `<b>${tot.toLocaleString()}</b> null experiments: <b>${ST.pct(rate, 2)}</b> (± ${ST.pct(2 * se, 2)} at two standard errors) ` +
      `against a nominal 5%. ` +
      (Math.abs(rate - 0.05) < 2 * se ? "Indistinguishable from nominal."
        : rate < 0.05 ? "<b>Conservative</b> — power is being lost, no false positives are being manufactured."
          : "<b>Anti-conservative</b> — more false positives than advertised."));
  }
  function run() { batch(4000); draw(); }
  [dS, nsS].forEach(s => s.on("change", () => { reset(); run(); }));
  d3.select("#rb-run").on("click", run);
  d3.select("#rb-reset").on("click", () => { rng = ST.rng(5150); reset(); run(); });
  run();
})();

/* ══════════ 6 · #rg-svg — ANOVA as a regression on indicators ══════════ */
(function () {
  const svg = d3.select("#rg-svg");
  if (svg.empty()) return;
  const W = 740, H = 470;
  const codeS = d3.select("#rg-code"), refS = d3.select("#rg-ref"),
        dragS = d3.select("#rg-drag"), shiftS = d3.select("#rg-shift");
  function draw() {
    const code = codeS.property("value"), ref = +refS.property("value");
    const which = +dragS.property("value"), shift = +shiftS.property("value") / 10;
    d3.select("#rg-shiftv").text((shift >= 0 ? "+" : "") + shift.toFixed(1));
    const gs = AV.RUN.y.map((a, j) => a.map(v => v + (j === which ? shift : 0)));
    const a = AV.anova(gs);
    const f = ST.frame(svg, W, H, { l: 44, r: 14, t: 26, b: 42 });
    const g = f.g, LW = 330, gap = 44;
    const flat = [].concat.apply([], gs);
    const y = d3.scaleLinear().domain([d3.min(flat) - 2, d3.max(flat) + 2]).nice().range([f.ih, 0]);
    const x = d3.scaleLinear().domain([0.4, 3.6]).range([0, LW]);
    ST.gridY(g, y, LW, 6); ST.axisL(g, y, 6, "score");
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`)
      .call(d3.axisBottom(x).tickValues([1, 2, 3]).tickFormat(d => AV.RUN.short[d - 1]));
    g.append("line").attr("x1", 0).attr("x2", LW).attr("y1", y(a.gm)).attr("y2", y(a.gm))
      .attr("stroke", SC.muted).attr("stroke-width", 1.1).attr("stroke-dasharray", "6,4");
    gs.forEach((grp, j) => {
      const jit = ST.jitterY(grp, 21 + j, 16);
      grp.forEach((v, i) => g.append("circle").attr("cx", x(j + 1) + jit[i]).attr("cy", y(v)).attr("r", 2.4)
        .attr("fill", AV.GC[j]).attr("fill-opacity", 0.55));
    });
    /* numeric coding draws the 1-df fitted LINE; the others draw the k fitted means */
    let numFit = null;
    if (code === "numeric") {
      const xs = [], ys = [];
      gs.forEach((grp, j) => grp.forEach(v => { xs.push(j + 1); ys.push(v); }));
      const ls = ST.lsLine(xs, ys);
      let ssr = 0; const ybar = ST.mean(ys);
      for (let i = 0; i < xs.length; i++) { const fit = ls.a + ls.b * xs[i]; ssr += (fit - ybar) * (fit - ybar); }
      const sse = a.sst - ssr, F1 = (ssr / 1) / (sse / (a.N - 2));
      numFit = { a: ls.a, b: ls.b, ssr: ssr, F: F1, p: 1 - ST.fCdf(F1, 1, a.N - 2), share: ssr / a.ssb };
      ST.lineSeg(g, ls.a, ls.b, x, y, { color: SC.bad, w: 2.6 });
      gs.forEach((grp, j) => g.append("circle").attr("cx", x(j + 1)).attr("cy", y(ls.a + ls.b * (j + 1))).attr("r", 4)
        .attr("fill", SC.bad));
    }
    gs.forEach((grp, j) => {
      g.append("line").attr("x1", x(j + 1) - 30).attr("x2", x(j + 1) + 30)
        .attr("y1", y(a.means[j])).attr("y2", y(a.means[j]))
        .attr("stroke", AV.GC[j]).attr("stroke-width", 2.6)
        .attr("stroke-dasharray", code === "numeric" ? "4,3" : null);
    });
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(code === "numeric" ? "one straight line across the group index — 1 df, and the wrong question"
                               : "the fit is just the k group means — 2 df");

    /* right: the two tables */
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    AV.panel(g2, 0, 4, RW, [
      ["SS_between  (k−1 = " + a.df1 + ")", ST.fmt(a.ssb, 6), SC.a2],
      ["SS_within   (N−k = " + a.df2 + ")", ST.fmt(a.ssw, 6), SC.teal],
      ["SS_total    (N−1 = " + (a.N - 1) + ")", ST.fmt(a.sst, 6)],
      ["MS_between / MS_within", ST.fmt(a.msb, 4) + " / " + ST.fmt(a.msw, 4)],
      ["F", ST.fmt(a.F, 6), SC.ink],
      ["p", AV.pfmt(a.p)],
      ["η²", ST.fmt(a.eta2, 6)]
    ], "the ANOVA table  (§06)");
    let rows;
    if (code === "dummy") {
      rows = [["β₀ = mean of " + AV.RUN.short[ref], ST.fmt(a.means[ref], 6)]];
      [0, 1, 2].filter(j => j !== ref).forEach(j =>
        rows.push(["β(" + AV.RUN.short[j] + ") = μ" + (j + 1) + " − μ" + (ref + 1), ST.fmt(a.means[j] - a.means[ref], 6)]));
    } else if (code === "effect") {
      const mu = ST.mean(a.means);
      rows = [["μ̂ = unweighted mean", ST.fmt(mu, 6)], ["grand mean ȳ̄ (for contrast)", ST.fmt(a.gm, 6), SC.muted]];
      [0, 1, 2].forEach(j => rows.push(["τ̂(" + AV.RUN.short[j] + ")", ST.fmt(a.means[j] - mu, 6)]));
    } else if (code === "cell") {
      rows = [0, 1, 2].map(j => ["β(" + AV.RUN.short[j] + ") = μ" + (j + 1), ST.fmt(a.means[j], 6)]);
      rows.push(["(no intercept, no constraint)", "—", SC.muted]);
    } else {
      rows = [["slope b", ST.fmt(numFit.b, 6), SC.bad],
              ["intercept", ST.fmt(numFit.a, 6)],
              ["SS explained (1 df)", ST.fmt(numFit.ssr, 6), SC.bad],
              ["as a share of SS_between", ST.pct(numFit.share, 1), SC.bad],
              ["F(1, " + (a.N - 2) + ")", ST.fmt(numFit.F, 6)],
              ["p", AV.pfmt(numFit.p), SC.bad]];
    }
    AV.panel(g2, 0, 178, RW, rows, code === "numeric" ? "the group index as a NUMBER — 1 df" : "the regression coefficients  (§10)");
    if (code !== "numeric") {
      g2.append("text").attr("x", 0).attr("y", 178 + 30 + rows.length * 16 + 18).attr("font-size", 10.5)
        .attr("fill", SC.good).text("SSR ≡ SS_between and SSE ≡ SS_within, exactly —");
      g2.append("text").attr("x", 0).attr("y", 178 + 30 + rows.length * 16 + 33).attr("font-size", 10.5)
        .attr("fill", SC.good).text("the coding changes only how the fit is DESCRIBED.");
    }
    d3.select("#rg-readout").html(
      code === "numeric"
        ? `Treating the groups as the number 1, 2, 3 spends <b>one</b> degree of freedom instead of two. It captures ` +
          `<b>${ST.pct(numFit.share, 1)}</b> of SS_between and returns p = <b>${AV.pfmt(numFit.p)}</b> against the ANOVA's ` +
          `<b>${AV.pfmt(a.p)}</b>. Legitimate as a planned linear contrast on an <i>ordered</i> factor; an error otherwise, and the ` +
          `tell is that relabelling the groups changes the answer.`
        : `Fitted by least squares on ${code === "cell" ? "three indicator columns and no intercept" : "an intercept and two coded columns"}: ` +
          `SSR = <b>${ST.fmt(a.ssb, 6)}</b> = SS_between, SSE = <b>${ST.fmt(a.ssw, 6)}</b> = SS_within, ` +
          `F = <b>${ST.fmt(a.F, 6)}</b>, R² = η² = <b>${ST.fmt(a.eta2, 6)}</b>. ` +
          `Switch the coding and every number in the left panel is unchanged — only the coefficients move.`);
  }
  [codeS, refS, dragS, shiftS].forEach(s => s.on("input change", draw));
  d3.select("#rg-reset").on("click", () => {
    codeS.property("value", "dummy"); refS.property("value", "0");
    dragS.property("value", "0"); shiftS.property("value", 0); draw();
  });
  draw();
})();

/* ══════════ 7 · #ef-svg — effect size and p-value diverge with n ══════════ */
(function () {
  const svg = d3.select("#ef-svg");
  if (svg.empty()) return;
  const W = 740, H = 420;
  const fS = d3.select("#ef-f"), kS = d3.select("#ef-k"), nS = d3.select("#ef-n"), vS = d3.select("#ef-view");
  function draw() {
    const fEff = +fS.property("value") / 100, k = +kS.property("value"), n = +nS.property("value");
    d3.select("#ef-fv").text(fEff.toFixed(2)); d3.select("#ef-kv").text(k); d3.select("#ef-nv").text(n);
    const view = vS.property("value");
    const f = ST.frame(svg, W, H, { l: 52, r: 52, t: 26, b: 44 });
    const g = f.g;
    const eta2True = fEff * fEff / (1 + fEff * fEff);
    if (view === "curves") {
      const ns = ST.linspace(3, 300, 130).map(Math.round);
      const x = d3.scaleLinear().domain([3, 300]).range([0, f.iw]);
      const yp = d3.scaleLog().domain([1e-12, 1]).range([f.ih, 0]).clamp(true);
      const ye = d3.scaleLinear().domain([0, Math.max(0.35, eta2True * 2.2)]).range([f.ih, 0]);
      ST.gridX(g, x, f.ih, 6);
      ST.axisB(g, x, f.ih, 6, "n per group");
      g.append("g").attr("class", "axis").call(d3.axisLeft(yp).ticks(7, ".0e"));
      g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.accent)
        .text("expected p-value (log scale, left axis)");
      g.append("g").attr("class", "axis").attr("transform", `translate(${f.iw},0)`).call(d3.axisRight(ye).ticks(5));
      g.append("text").attr("x", f.iw).attr("y", -10).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", SC.a2).text("η² and ω² (right axis)");
      /* the p a study of size n typically returns: use the F at the expected non-centrality */
      const pl = [], e1 = [], e2 = [];
      ns.forEach(nn => {
        const N = k * nn, d1 = k - 1, d2 = N - k;
        const eF = 1 + N * fEff * fEff / d1;                     // E[MS_b]/E[MS_w]
        pl.push([nn, Math.max(1e-12, 1 - ST.fCdf(eF, d1, d2))]);
        const eEta = (d1 * eF) / (d1 * eF + d2);
        e1.push([nn, eEta]);
        e2.push([nn, Math.max(0, (d1 * (eF - 1)) / (d1 * eF + d2 + 1))]);
      });
      g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => yp(d[1]))(pl))
        .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.4);
      g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => ye(d[1]))(e1))
        .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
      g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => ye(d[1]))(e2))
        .attr("fill", "none").attr("stroke", SC.teal).attr("stroke-width", 2).attr("stroke-dasharray", "5,4");
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", ye(eta2True)).attr("y2", ye(eta2True))
        .attr("stroke", SC.muted).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,3");
      g.append("text").attr("x", f.iw - 4).attr("y", ye(eta2True) - 5).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", SC.muted).text("the TRUE η² = " + ST.fmt(eta2True, 4));
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", yp(0.05)).attr("y2", yp(0.05))
        .attr("stroke", SC.good).attr("stroke-width", 1.3).attr("stroke-dasharray", "5,4");
      g.append("text").attr("x", 4).attr("y", yp(0.05) - 5).attr("font-size", 9.5).attr("fill", SC.good).text("p = 0.05");
      g.append("line").attr("x1", x(n)).attr("x2", x(n)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.bad).attr("stroke-width", 1.6);
      ST.legend(g, [{ label: "p", color: SC.accent }, { label: "η² (biased up)", color: SC.a2 },
                    { label: "ω² (corrected)", color: SC.teal, dash: "5,4" }], f.iw - 130, 22);
    } else {
      const rng = ST.rng(24601), pts = [];
      for (let s = 0; s < 900; s++) {
        const gs = [];
        for (let j = 0; j < k; j++) {
          const mu = fEff * Math.sqrt(k / 2) * (j - (k - 1) / 2) * 2 / Math.max(1, k - 1), gg = [];
          for (let i = 0; i < n; i++) gg.push(mu + ST.randn(rng));
          gs.push(gg);
        }
        const a = AV.anova(gs);
        pts.push({ p: Math.max(1e-9, a.p), e: a.eta2 });
      }
      const x = d3.scaleLinear().domain([0, d3.max(pts, d => d.e) * 1.06]).range([0, f.iw]);
      const y = d3.scaleLog().domain([1e-9, 1]).range([f.ih, 0]).clamp(true);
      ST.gridX(g, x, f.ih, 6);
      ST.axisB(g, x, f.ih, 6, "estimated η² from that study");
      g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6, ".0e"));
      pts.forEach(d => g.append("circle").attr("cx", x(d.e)).attr("cy", y(d.p)).attr("r", 2)
        .attr("fill", d.p < 0.05 ? SC.good : SC.bad).attr("fill-opacity", 0.42));
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(0.05)).attr("y2", y(0.05))
        .attr("stroke", SC.ink).attr("stroke-width", 1.3).attr("stroke-dasharray", "5,4");
      g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(`900 studies from ONE unchanging world (true f = ${fEff.toFixed(2)}), k = ${k}, n = ${n}`);
      const sig = pts.filter(d => d.p < 0.05);
      g.append("text").attr("x", f.iw - 4).attr("y", 14).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", SC.good).text(ST.pct(sig.length / pts.length, 1) + " reached significance");
      if (sig.length) g.append("text").attr("x", f.iw - 4).attr("y", 30).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.a2)
        .text("their mean η̂² = " + ST.fmt(ST.mean(sig.map(d => d.e)), 3) + " vs all studies' " + ST.fmt(ST.mean(pts.map(d => d.e)), 3));
    }
    const N = k * n, d1 = k - 1, d2 = N - k;
    const minEta = d1 * ST.fQuant(0.95, d1, d2) / (d1 * ST.fQuant(0.95, d1, d2) + d2);
    d3.select("#ef-readout").html(
      `True Cohen f = <b>${fEff.toFixed(2)}</b> ⟹ true η² = <b>${ST.fmt(eta2True, 4)}</b>, a fixed property of the world. ` +
      `With k = ${k} and n = ${n} (N = ${N}), the <b>smallest η² that reaches significance</b> is <b>${ST.fmt(minEta, 4)}</b>, ` +
      `and E[η²] under a true null is (k−1)/(N−1) = <b>${ST.fmt(d1 / (N - 1), 4)}</b>. ` +
      `Power here is <b>${ST.pct(AV.powerF(k, n, fEff, 0.05), 1)}</b>.`);
  }
  [fS, kS, nS, vS].forEach(s => s.on("input change", draw));
  d3.select("#ef-run").on("click", () => {
    kS.property("value", 3); nS.property("value", 14); fS.property("value", 37); vS.property("value", "curves"); draw();
  });
  draw();
})();

/* ══════════ 8 · #pw2-svg — power curves from the non-central F ══════════ */
(function () {
  const svg = d3.select("#pw2-svg");
  if (svg.empty()) return;
  const W = 740, H = 400;
  const kS = d3.select("#pw2-k"), fS = d3.select("#pw2-f"), aS = d3.select("#pw2-a"), nS = d3.select("#pw2-n");
  function draw() {
    const k = +kS.property("value"), fEff = +fS.property("value") / 100;
    const alpha = +aS.property("value"), n = +nS.property("value");
    d3.select("#pw2-kv").text(k); d3.select("#pw2-fv").text(fEff.toFixed(2));
    d3.select("#pw2-nv").text(n);
    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 26, b: 44 });
    const g = f.g, LW = 400, gap = 48;
    const ns = ST.linspace(3, 200, 90).map(Math.round);
    const x = d3.scaleLinear().domain([3, 200]).range([0, LW]);
    const y = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
    ST.gridY(g, y, LW, 5);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(d => ST.pct(d, 0)));
    g.append("text").attr("x", LW).attr("y", f.ih + 32).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", SC.muted).text("n per group");
    const FAM = [0.10, 0.25, 0.40];
    FAM.forEach((ff, i) => {
      const pts = ns.map(nn => [nn, AV.powerF(k, nn, ff, alpha)]);
      g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(pts))
        .attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 1.2).attr("stroke-opacity", 0.5);
      g.append("text").attr("x", LW - 2).attr("y", y(pts[pts.length - 1][1]) - 4).attr("text-anchor", "end")
        .attr("font-size", 9).attr("fill", SC.muted).text("f = " + ff.toFixed(2));
    });
    const sel = ns.map(nn => [nn, AV.powerF(k, nn, fEff, alpha)]);
    g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(sel))
      .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.6);
    g.append("line").attr("x1", 0).attr("x2", LW).attr("y1", y(0.8)).attr("y2", y(0.8))
      .attr("stroke", SC.good).attr("stroke-width", 1.3).attr("stroke-dasharray", "5,4");
    g.append("text").attr("x", 4).attr("y", y(0.8) - 5).attr("font-size", 9.5).attr("fill", SC.good).text("80%");
    const pw = AV.powerF(k, n, fEff, alpha);
    g.append("line").attr("x1", x(n)).attr("x2", x(n)).attr("y1", f.ih).attr("y2", y(pw))
      .attr("stroke", SC.bad).attr("stroke-width", 1.6).attr("stroke-dasharray", "3,3");
    g.append("circle").attr("cx", x(n)).attr("cy", y(pw)).attr("r", 4.5).attr("fill", SC.bad);
    g.append("text").attr("x", x(n) + 7).attr("y", y(pw) - 7).attr("font-size", 10.5).attr("fill", SC.bad)
      .text("n = " + n + " → " + ST.pct(pw, 1));
    let n80 = null;
    for (let nn = 3; nn <= 2000; nn++) if (AV.powerF(k, nn, fEff, alpha) >= 0.8) { n80 = nn; break; }
    if (n80 && n80 <= 200) {
      g.append("circle").attr("cx", x(n80)).attr("cy", y(0.8)).attr("r", 4).attr("fill", "none")
        .attr("stroke", SC.good).attr("stroke-width", 2);
      g.append("text").attr("x", x(n80)).attr("y", y(0.8) + 18).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.good).text("n = " + n80);
    }
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(`power against n · k = ${k} · α = ${alpha}`);

    /* right: null and non-central densities */
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap, d1 = k - 1, d2 = k * n - k, lam = k * n * fEff * fEff;
    const crit = ST.fQuant(1 - alpha, d1, d2);
    const xmax = Math.max(crit * 2.4, 8);
    const xs = ST.linspace(0.02, xmax, 220);
    const x2 = d3.scaleLinear().domain([0, xmax]).range([0, RW]);
    const nullD = xs.map(v => ST.fPdf(v, d1, d2));
    const altD = xs.map(v => {
      const h = Math.max(1e-4, v * 1e-3);
      return Math.max(0, (AV.ncfSf(v - h, d1, d2, lam) - AV.ncfSf(v + h, d1, d2, lam)) / (2 * h));
    });
    const y2 = d3.scaleLinear().domain([0, Math.max(d3.max(nullD.filter((v, i) => xs[i] > 0.2)), d3.max(altD)) * 1.12]).range([f.ih, 0]);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x2).ticks(5));
    const nt = xs.filter(v => v >= crit);
    if (nt.length) {
      g2.append("path").attr("d", d3.area().x(v => x2(v)).y0(f.ih).y1(v => y2(ST.fPdf(v, d1, d2)))(nt))
        .attr("fill", SC.bad).attr("fill-opacity", 0.35);
      const alt = nt.map(v => { const i = xs.indexOf(v); return [v, altD[i]]; });
      g2.append("path").attr("d", d3.area().x(d => x2(d[0])).y0(f.ih).y1(d => y2(d[1]))(alt))
        .attr("fill", SC.good).attr("fill-opacity", 0.24);
    }
    g2.append("path").attr("d", d3.line().x((v, i) => x2(xs[i])).y((v, i) => y2(nullD[i]))(xs))
      .attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 2);
    g2.append("path").attr("d", d3.line().x((v, i) => x2(xs[i])).y((v, i) => y2(altD[i]))(xs))
      .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.2);
    g2.append("line").attr("x1", x2(crit)).attr("x2", x2(crit)).attr("y1", f.ih).attr("y2", 0)
      .attr("stroke", SC.ink).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
    ST.legend(g2, [{ label: "null: F(" + d1 + ", " + d2 + ")", color: SC.muted },
                   { label: "alternative: λ = " + ST.fmt(lam, 2), color: SC.accent },
                   { label: "power", color: SC.good, op: 0.5 },
                   { label: "level α", color: SC.bad, op: 0.5 }], 6, 14);
    d3.select("#pw2-readout").html(
      `k = ${k}, n = ${n} per group (N = ${k * n}), true Cohen f = ${fEff.toFixed(2)}, α = ${alpha}. ` +
      `Non-centrality λ = N·f² = <b>${ST.fmt(lam, 3)}</b>; critical value ${ST.fmt(crit, 4)}; ` +
      `<b>power = ${ST.pct(pw, 1)}</b>. For 80% power you need <b>n = ${n80 || ">2000"}</b> per group ` +
      `(total N = ${n80 ? k * n80 : "—"}).`);
  }
  [kS, fS, aS, nS].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══════════ 9 · #pm-svg — the permutation distribution of F ══════════ */
(function () {
  const svg = d3.select("#pm-svg");
  if (svg.empty()) return;
  const W = 740, H = 450;
  const dS = d3.select("#pm-data"), bS = d3.select("#pm-B");
  let rng = ST.rng(1848), perm = [], labels = null, flat = [], ns = [], Fobs = 0, aObs = null;

  function dataset() {
    const kind = dS.property("value"), r = ST.rng(99);
    if (kind === "run") return AV.RUN.y.map(a => a.slice());
    if (kind === "equal") return [0, 1, 2].map(j => d3.range(15).map(() => 50 + 6 * ST.randn(r)));
    if (kind === "hetero") return [d3.range(5).map(() => 50 + 18 * ST.randn(r)),
                                   d3.range(15).map(() => 50 + 6 * ST.randn(r)),
                                   d3.range(25).map(() => 50 + 6 * ST.randn(r))];
    if (kind === "skew") return [0, 1, 2].map(j => d3.range(14).map(() => 50 + 8 * (-Math.log(1 - r()) - 1)));
    return [0, 1, 2].map(j => d3.range(4).map(() => 50 + 6 * ST.randn(r)));
  }
  function setup() {
    const gs = dataset();
    aObs = AV.anova(gs); Fobs = aObs.F;
    ns = gs.map(g => g.length); flat = [].concat.apply([], gs);
    labels = []; ns.forEach((n, j) => { for (let i = 0; i < n; i++) labels.push(j); });
    perm = []; rng = ST.rng(1848);
  }
  function shuffleOnce() {
    const s = ST.shuffle(flat, rng), gs = [];
    let off = 0;
    for (let j = 0; j < ns.length; j++) { gs.push(s.slice(off, off + ns[j])); off += ns[j]; }
    perm.push(AV.Fstat(gs));
    return gs;
  }
  let shown = null;
  function draw() {
    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 22, b: 40 });
    const g = f.g, topH = 130, gapY = 46;
    const gs = shown || (() => { const o = []; let off = 0; for (let j = 0; j < ns.length; j++) { o.push(flat.slice(off, off + ns[j])); off += ns[j]; } return o; })();
    const y = d3.scaleLinear().domain(d3.extent(flat)).nice().range([topH, 0]);
    const bx = d3.scalePoint().domain(d3.range(ns.length)).range([50, f.iw - 40]).padding(0.5);
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    gs.forEach((grp, j) => {
      const jit = ST.jitterY(grp, 41 + j, 15);
      grp.forEach((v, i) => g.append("circle").attr("cx", bx(j) + jit[i]).attr("cy", y(v)).attr("r", 2.4)
        .attr("fill", AV.GC[j]).attr("fill-opacity", 0.6));
      g.append("line").attr("x1", bx(j) - 22).attr("x2", bx(j) + 22).attr("y1", y(ST.mean(grp))).attr("y2", y(ST.mean(grp)))
        .attr("stroke", AV.GC[j]).attr("stroke-width", 2.4);
      g.append("text").attr("x", bx(j)).attr("y", topH + 15).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", AV.GC[j]).text("n = " + ns[j]);
    });
    g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.muted)
      .text(shown ? "one shuffled relabelling — the VALUES never move, only the colours"
                  : "the data as collected");
    /* histogram */
    const g2 = g.append("g").attr("transform", `translate(0,${topH + gapY})`);
    const bh = f.ih - topH - gapY;
    const d1 = aObs.df1, d2 = aObs.df2;
    const xmax = Math.max(6, Fobs * 1.6, ST.fQuant(0.995, d1, d2));
    const x = d3.scaleLinear().domain([0, xmax]).range([0, f.iw]);
    const bw = xmax / 50;
    const inr = perm.filter(v => v <= xmax);
    const bins = perm.length ? ST.histBins(inr, 0, bw, 50) : [];
    const sc = perm.length ? inr.length / perm.length : 1;
    const xs = ST.linspace(0.02, xmax, 220), dens = xs.map(v => ST.fPdf(v, d1, d2));
    const y2 = d3.scaleLinear().domain([0, Math.max(d3.max(dens.filter((v, i) => xs[i] > 0.2)) || 1,
      bins.length ? d3.max(bins, b => b.density * sc) : 0) * 1.14]).range([bh, 0]);
    ST.gridY(g2, y2, f.iw, 4);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${bh})`).call(d3.axisBottom(x).ticks(7));
    g2.append("text").attr("x", f.iw).attr("y", bh + 32).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", SC.muted).text("F on the shuffled labels");
    g2.selectAll("rect.h").data(bins).join("rect").attr("class", "h")
      .attr("x", b => x(b.x0)).attr("y", b => y2(b.density * sc))
      .attr("width", Math.max(1, x(bw) - x(0) - 1)).attr("height", b => Math.max(0, bh - y2(b.density * sc)))
      .attr("fill", b => b.x0 >= Fobs ? SC.bad : SC.accent).attr("fill-opacity", 0.4);
    g2.append("path").attr("d", d3.line().x(v => x(v)).y(v => y2(ST.fPdf(v, d1, d2)))(xs))
      .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2);
    g2.append("line").attr("x1", x(Math.min(Fobs, xmax))).attr("x2", x(Math.min(Fobs, xmax)))
      .attr("y1", bh).attr("y2", 0).attr("stroke", SC.bad).attr("stroke-width", 2);
    g2.append("text").attr("x", x(Math.min(Fobs, xmax)) + 5).attr("y", 14).attr("font-size", 10.5)
      .attr("fill", SC.bad).text("F_obs = " + ST.fmt(Fobs, 4));
    ST.legend(g2, [{ label: perm.length.toLocaleString() + " shuffles", color: SC.accent },
                   { label: `the tabulated F(${d1}, ${d2})`, color: SC.a2 }], f.iw - 190, 14);
    const ge = perm.filter(v => v >= Fobs - 1e-12).length;
    const pp = perm.length ? (ge + 1) / (perm.length + 1) : NaN;
    const cls = 1 - ST.fCdf(Fobs, d1, d2);
    const mcse = perm.length ? Math.sqrt(pp * (1 - pp) / perm.length) : NaN;
    d3.select("#pm-readout").html(
      `F<sub>obs</sub> = <b>${ST.fmt(Fobs, 6)}</b> on (${d1}, ${d2}). ` +
      `Tabulated p = <b>${AV.pfmt(cls)}</b>. ` +
      (perm.length ? `Permutation p = (${ge} + 1)/(${perm.length} + 1) = <b>${AV.pfmt(pp)}</b> ` +
        `(Monte Carlo SE ${ST.fmt(mcse, 5)}; the gap is ${ST.fmt(Math.abs(pp - cls) / Math.max(mcse, 1e-12), 1)} SEs). `
        : "Press a shuffle button to start building the permutation distribution. ") +
      `The reference set has ${ns.reduce((a, b) => a + b, 0)}!/(${ns.map(n => n + "!").join("·")}) elements — sampled, not enumerated.`);
  }
  function runAll() {
    const B = +bS.property("value");
    perm = []; rng = ST.rng(1848);
    for (let i = 0; i < B; i++) shuffleOnce();
    shown = null; draw();
  }
  dS.on("change", () => { setup(); shown = null; draw(); });
  bS.on("change", runAll);
  d3.select("#pm-one").on("click", () => { shown = shuffleOnce(); draw(); });
  d3.select("#pm-run").on("click", runAll);
  d3.select("#pm-reset").on("click", () => { setup(); shown = null; draw(); });
  setup(); runAll();
})();

/* ══════════ 10 · #we-svg — classical F, Welch and Brown–Forsythe ══════════ */
(function () {
  const svg = d3.select("#we-svg");
  if (svg.empty()) return;
  const W = 740, H = 420;
  const s1S = d3.select("#we-s1"), s3S = d3.select("#we-s3"), nsS = d3.select("#we-ns");
  let rng = ST.rng(60613), c = { F: 0, W: 0, B: 0 }, tot = 0;
  function reset() { c = { F: 0, W: 0, B: 0 }; tot = 0; }
  function conf() {
    return { sds: [+s1S.property("value") / 10, 1, +s3S.property("value") / 10],
             ns: nsS.property("value").split(",").map(Number) };
  }
  function batch(reps) {
    const { sds, ns } = conf(), k = ns.length;
    const N = ns.reduce((a, b) => a + b, 0), crit = ST.fQuant(0.95, k - 1, N - k);
    for (let r = 0; r < reps; r++) {
      const gs = ns.map((n, j) => { const g = []; for (let i = 0; i < n; i++) g.push(sds[j] * ST.randn(rng)); return g; });
      if (AV.Fstat(gs) > crit) c.F++;
      if (AV.welch(gs).p < 0.05) c.W++;
      if (AV.brownForsythe(gs).p < 0.05) c.B++;
      tot++;
    }
  }
  function draw() {
    const { sds, ns } = conf();
    d3.select("#we-s1v").text(sds[0].toFixed(1)); d3.select("#we-s3v").text(sds[2].toFixed(1));
    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 26, b: 44 });
    const g = f.g, LW = 300, gap = 56;
    const r2 = ST.rng(2718);
    const gs = ns.map((n, j) => { const gg = []; for (let i = 0; i < n; i++) gg.push(sds[j] * ST.randn(r2)); return gg; });
    const flat = [].concat.apply([], gs);
    const y = d3.scaleLinear().domain(d3.extent(flat)).nice().range([f.ih, 0]);
    const bx = d3.scalePoint().domain([0, 1, 2]).range([56, LW - 34]).padding(0.5);
    ST.gridY(g, y, LW, 5); ST.axisL(g, y, 5, "one null sample");
    const maxN = Math.max.apply(null, ns);
    gs.forEach((grp, j) => {
      const hw = 8 + 16 * Math.sqrt(ns[j] / maxN);
      AV.boxplot(g, grp, bx(j), hw, y, AV.GC[j]);
      g.append("text").attr("x", bx(j)).attr("y", f.ih + 15).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", AV.GC[j]).text(`n=${ns[j]}, σ=${sds[j].toFixed(1)}`);
    });
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text("box width ∝ √n · all three means are EQUAL");
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    const rows = [["classical F", c.F, SC.bad], ["Welch F", c.W, SC.good], ["Brown–Forsythe F*", c.B, SC.a2]];
    const yMax = Math.max(0.10, d3.max(rows, r => tot ? r[1] / tot : 0) * 1.25);
    const x = d3.scaleBand().domain(rows.map(r => r[0])).range([0, RW]).padding(0.3);
    const y2 = d3.scaleLinear().domain([0, yMax]).range([f.ih, 0]);
    ST.gridY(g2, y2, RW, 5);
    g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5).tickFormat(d => ST.pct(d, 0)));
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x))
      .selectAll("text").attr("font-size", 9.5);
    rows.forEach(r => {
      const v = tot ? r[1] / tot : 0, se = tot ? Math.sqrt(v * (1 - v) / tot) : 0;
      g2.append("rect").attr("x", x(r[0])).attr("y", y2(v)).attr("width", x.bandwidth())
        .attr("height", Math.max(0, f.ih - y2(v))).attr("rx", 3)
        .attr("fill", r[2]).attr("fill-opacity", 0.5).attr("stroke", r[2]);
      g2.append("line").attr("x1", x(r[0]) + x.bandwidth() / 2).attr("x2", x(r[0]) + x.bandwidth() / 2)
        .attr("y1", y2(Math.min(yMax, v + 2 * se))).attr("y2", y2(Math.max(0, v - 2 * se)))
        .attr("stroke", SC.ink).attr("stroke-opacity", 0.6);
      g2.append("text").attr("x", x(r[0]) + x.bandwidth() / 2).attr("y", y2(v) - 6).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", SC.ink).text(ST.pct(v, 2));
    });
    g2.append("line").attr("x1", 0).attr("x2", RW).attr("y1", y2(0.05)).attr("y2", y2(0.05))
      .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "5,4");
    g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(`measured type I error over ${tot.toLocaleString()} null experiments`);
    const ratio = Math.max.apply(null, sds) / Math.min.apply(null, sds);
    d3.select("#we-readout").html(
      `SDs ${sds.map(s => s.toFixed(1)).join(", ")} (largest/smallest = <b>${ST.fmt(ratio, 2)}</b>) with sizes ${ns.join(", ")}. ` +
      `Over ${tot.toLocaleString()} null experiments: classical F <b>${ST.pct(tot ? c.F / tot : 0, 2)}</b>, ` +
      `Welch <b>${ST.pct(tot ? c.W / tot : 0, 2)}</b>, Brown–Forsythe <b>${ST.pct(tot ? c.B / tot : 0, 2)}</b>, nominal 5%. ` +
      `Welch's ANOVA on the running data gives F = 2.4460 on (2, 25.24) df, p = 0.1069 — it spends df to buy safety.`);
  }
  function run() { batch(3000); draw(); }
  [s1S, s3S, nsS].forEach(s => s.on("input change", () => { reset(); run(); }));
  d3.select("#we-run").on("click", run);
  d3.select("#we-reset").on("click", () => { rng = ST.rng(60613); reset(); run(); });
  run();
})();

/* ══════════ 11 · #kw-svg — F, permutation F and Kruskal–Wallis ══════════ */
(function () {
  const svg = d3.select("#kw-svg");
  if (svg.empty()) return;
  const W = 740, H = 420;
  const dS = d3.select("#kw-dist"), sepS = d3.select("#kw-d"), nS = d3.select("#kw-n"), mS = d3.select("#kw-mode");
  let rng = ST.rng(112358), c = { F: 0, P: 0, K: 0 }, tot = 0;
  function reset() { c = { F: 0, P: 0, K: 0 }; tot = 0; }
  function gen(r) {
    const kind = dS.property("value"), n = +nS.property("value");
    const sep = mS.property("value") === "level" ? 0 : +sepS.property("value") / 100;
    const out = [];
    for (let j = 0; j < 3; j++) {
      const mu = sep * (j - 1), g = [];
      for (let i = 0; i < n; i++) {
        let v;
        if (kind === "contam") v = ST.randn(r) + (i === 0 ? 9 * (r() < 0.5 ? -1 : 1) : 0);
        else v = AV.draw(kind === "norm" ? "norm" : kind, r);
        g.push(mu + v);
      }
      out.push(g);
    }
    return out;
  }
  function batch(reps) {
    const n = +nS.property("value"), N = 3 * n, crit = ST.fQuant(0.95, 2, N - 3);
    for (let r = 0; r < reps; r++) {
      const gs = gen(rng);
      const F = AV.Fstat(gs);
      if (F > crit) c.F++;
      if (AV.kruskal(gs).p < 0.05) c.K++;
      /* permutation F with B = 99 — enough for a 5% decision, cheap enough to run live */
      const flat = [].concat.apply([], gs);
      let ge = 0;
      for (let b = 0; b < 99; b++) {
        const s = ST.shuffle(flat, rng), pg = [s.slice(0, n), s.slice(n, 2 * n), s.slice(2 * n)];
        if (AV.Fstat(pg) >= F - 1e-12) ge++;
      }
      if ((ge + 1) / 100 <= 0.05) c.P++;
      tot++;
    }
  }
  function draw() {
    const n = +nS.property("value"), level = mS.property("value") === "level";
    const sep = level ? 0 : +sepS.property("value") / 100;
    d3.select("#kw-dv").text((+sepS.property("value") / 100).toFixed(2)); d3.select("#kw-nv").text(n);
    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 26, b: 44 });
    const g = f.g, LW = 330, gap = 54;
    const gs = gen(ST.rng(9001));
    const flat = [].concat.apply([], gs), rk = ST.ranks(flat);
    const cells = [{ vals: gs, lab: "raw values", off: 0 },
                   { vals: (() => { const o = []; let p = 0; for (let j = 0; j < 3; j++) { o.push(rk.slice(p, p + n)); p += n; } return o; })(), lab: "the same data, RANKED", off: LW / 2 }];
    cells.forEach((cc, ci) => {
      const fl = [].concat.apply([], cc.vals);
      const y = d3.scaleLinear().domain(d3.extent(fl)).nice().range([f.ih, 0]);
      const gg = g.append("g").attr("transform", `translate(${ci * (LW / 2 + 16)},0)`);
      gg.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
      const bx = d3.scalePoint().domain([0, 1, 2]).range([30, LW / 2 - 26]).padding(0.5);
      cc.vals.forEach((grp, j) => {
        const jit = ST.jitterY(grp, 51 + j, 9);
        grp.forEach((v, i) => gg.append("circle").attr("cx", bx(j) + jit[i]).attr("cy", y(v)).attr("r", 2)
          .attr("fill", AV.GC[j]).attr("fill-opacity", 0.5));
        gg.append("line").attr("x1", bx(j) - 13).attr("x2", bx(j) + 13).attr("y1", y(ST.mean(grp))).attr("y2", y(ST.mean(grp)))
          .attr("stroke", AV.GC[j]).attr("stroke-width", 2.2);
      });
      gg.append("text").attr("x", 0).attr("y", -10).attr("font-size", 10.5).attr("fill", SC.muted).text(cc.lab);
    });
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    const rows = [["classical F", c.F, SC.accent], ["permutation F", c.P, SC.violet], ["Kruskal–Wallis", c.K, SC.good]];
    const yMax = level ? Math.max(0.10, d3.max(rows, r => tot ? r[1] / tot : 0) * 1.3) : 1;
    const x = d3.scaleBand().domain(rows.map(r => r[0])).range([0, RW]).padding(0.3);
    const y2 = d3.scaleLinear().domain([0, yMax]).range([f.ih, 0]);
    ST.gridY(g2, y2, RW, 5);
    g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5).tickFormat(d => ST.pct(d, 0)));
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x))
      .selectAll("text").attr("font-size", 9.5).attr("transform", "rotate(-12)").attr("text-anchor", "end");
    rows.forEach(r => {
      const v = tot ? r[1] / tot : 0, se = tot ? Math.sqrt(v * (1 - v) / tot) : 0;
      g2.append("rect").attr("x", x(r[0])).attr("y", y2(v)).attr("width", x.bandwidth())
        .attr("height", Math.max(0, f.ih - y2(v))).attr("rx", 3).attr("fill", r[2]).attr("fill-opacity", 0.5).attr("stroke", r[2]);
      g2.append("line").attr("x1", x(r[0]) + x.bandwidth() / 2).attr("x2", x(r[0]) + x.bandwidth() / 2)
        .attr("y1", y2(Math.min(yMax, v + 2 * se))).attr("y2", y2(Math.max(0, v - 2 * se)))
        .attr("stroke", SC.ink).attr("stroke-opacity", 0.6);
      g2.append("text").attr("x", x(r[0]) + x.bandwidth() / 2).attr("y", y2(v) - 6).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", SC.ink).text(ST.pct(v, 1));
    });
    if (level) {
      g2.append("line").attr("x1", 0).attr("x2", RW).attr("y1", y2(0.05)).attr("y2", y2(0.05))
        .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "5,4");
    }
    g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(level ? "measured LEVEL — nominal 5%" : "measured POWER — higher is better");
    d3.select("#kw-readout").html(
      `${dS.property("options")[dS.property("selectedIndex")].text}, n = ${n} per group, separation ${sep.toFixed(2)}σ. ` +
      `Over <b>${tot.toLocaleString()}</b> experiments: F <b>${ST.pct(tot ? c.F / tot : 0, 1)}</b>, ` +
      `permutation F <b>${ST.pct(tot ? c.P / tot : 0, 1)}</b>, Kruskal–Wallis <b>${ST.pct(tot ? c.K / tot : 0, 1)}</b>. ` +
      (level ? "All three should sit at 5%; a bar above it is a test making more false claims than it advertises."
             : "With a contaminating point per group the rank test wins; with clean normal data it loses about 4.5%.") +
      ` On the running data: F p = 0.088834, Kruskal–Wallis H = 3.409643 on 2 df, p = 0.181805.`);
  }
  function run() { batch(1200); draw(); }
  [dS, sepS, nS, mS].forEach(s => s.on("input change", () => { reset(); run(); }));
  d3.select("#kw-run").on("click", run);
  d3.select("#kw-reset").on("click", () => { rng = ST.rng(112358); reset(); run(); });
  run();
})();

/* ══════════ 12 · #lk-svg — the look-elsewhere effect ══════════ */
(function () {
  const svg = d3.select("#lk-svg");
  if (svg.empty()) return;
  const W = 740, H = 430;
  const mS = d3.select("#lk-m"), m1S = d3.select("#lk-m1"), aS = d3.select("#lk-a"), vS = d3.select("#lk-view");
  let seed = 1992;
  function study(m, m1, sd) {
    const r = ST.rng(sd), p = [], isNull = [];
    for (let i = 0; i < m; i++) {
      const nonNull = i >= m - m1;
      const z = ST.randn(r) + (nonNull ? 3.2 : 0);
      p.push(2 * (1 - ST.normCdf(Math.abs(z))));
      isNull.push(!nonNull);
    }
    return { p: p, isNull: isNull };
  }
  function draw() {
    const m = +mS.property("value"), m1 = Math.min(+m1S.property("value"), m), alpha = +aS.property("value");
    d3.select("#lk-mv").text(m); d3.select("#lk-m1v").text(m1);
    const view = vS.property("value");
    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 26, b: 44 });
    const g = f.g;
    if (view === "grid") {
      const s = study(m, m1, seed);
      const cols = Math.ceil(Math.sqrt(m * f.iw / f.ih)), rows = Math.ceil(m / cols);
      const cw = Math.min(f.iw / cols, f.ih / rows), sz = Math.max(2, cw - 1.5);
      const col = d3.scaleSequential(t => d3.interpolateRgb(SC.bad, SC.panel2)(Math.min(1, t * 6)))
        .domain([0, 1]);
      let disc = 0, fd = 0, td = 0;
      for (let i = 0; i < m; i++) {
        const cx = (i % cols) * cw, cy = Math.floor(i / cols) * cw;
        const sig = s.p[i] < alpha;
        if (sig) { disc++; if (s.isNull[i]) fd++; else td++; }
        g.append("rect").attr("x", cx).attr("y", cy).attr("width", sz).attr("height", sz).attr("rx", 1)
          .attr("fill", sig ? (s.isNull[i] ? SC.bad : SC.good) : col(s.p[i]))
          .attr("fill-opacity", sig ? 0.95 : 0.55)
          .attr("stroke", sig ? SC.ink : (s.isNull[i] ? "none" : SC.good))
          .attr("stroke-width", sig ? 1 : 0.8).attr("stroke-opacity", sig ? 0.9 : 0.5);
      }
      ST.legend(g, [{ label: "false discovery (null was true)", color: SC.bad },
                    { label: "true discovery", color: SC.good },
                    { label: "not significant — darker = smaller p", color: SC.panel2 }], f.iw - 240, f.ih - 40);
      g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(`${m} tests, ${m1} of them real · threshold p < ${alpha}`);
      d3.select("#lk-readout").html(
        `<b>${disc}</b> discoveries: <b style="color:${SC.bad}">${fd} false</b> and ` +
        `<b style="color:${SC.good}">${td} true</b>. ` +
        `Under a complete null, ${m} tests at ${alpha} yield <b>${ST.fmt(m * alpha, 1)}</b> discoveries on average and ` +
        `P(at least one) = <b>${ST.fmt(1 - Math.pow(1 - alpha, m), 6)}</b>. ` +
        (m1 === 0 ? "Every square outlined here is noise — and any one of them could be the headline."
                  : `False discovery proportion so far: <b>${disc ? ST.fmt(fd / disc, 4) : "—"}</b>.`) +
        ` The Bonferroni threshold would be ${(alpha / m).toExponential(2)}.`);
    } else {
      /* the sampling distribution of the SMALLEST p-value, over many studies */
      const R = 3000, mins = [];
      const r = ST.rng(seed);
      for (let s = 0; s < R; s++) {
        let mn = 1;
        for (let i = 0; i < m; i++) {
          const nonNull = i >= m - m1;
          const z = ST.randn(r) + (nonNull ? 3.2 : 0);
          const p = 2 * (1 - ST.normCdf(Math.abs(z)));
          if (p < mn) mn = p;
        }
        mins.push(mn);
      }
      const hi = Math.max(d3.quantile(ST.asc(mins), 0.995), alpha * 1.6);
      const x = d3.scaleLinear().domain([0, hi]).range([0, f.iw]);
      const bins = ST.histBins(mins.filter(v => v <= hi), 0, hi / 44, 44);
      const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.n) * 1.14]).range([f.ih, 0]);
      ST.gridY(g, y, f.iw, 5);
      ST.axisB(g, x, f.ih, 6, "the smallest of the m p-values");
      ST.axisL(g, y, 5, "studies");
      g.selectAll("rect.h").data(bins).join("rect").attr("class", "h")
        .attr("x", b => x(b.x0)).attr("y", b => y(b.n)).attr("width", Math.max(1, f.iw / 44 - 1))
        .attr("height", b => f.ih - y(b.n)).attr("fill", SC.violet).attr("fill-opacity", 0.45);
      g.append("line").attr("x1", x(alpha)).attr("x2", x(alpha)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.bad).attr("stroke-width", 2);
      g.append("text").attr("x", x(alpha) + 6).attr("y", 16).attr("font-size", 10.5).attr("fill", SC.bad)
        .text("the threshold a single test would use: " + alpha);
      const below = mins.filter(v => v < alpha).length / R;
      g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(`${R.toLocaleString()} repeated studies of ${m} tests each`);
      d3.select("#lk-readout").html(
        `Across ${R.toLocaleString()} repeated studies of ${m} tests, the smallest p-value is below ${alpha} in ` +
        `<b>${ST.pct(below, 1)}</b> of them ` +
        (m1 === 0 ? `— with <b>no real effect anywhere</b>. The theoretical value is ${ST.pct(1 - Math.pow(1 - alpha, m), 1)}. `
                  : `(${m1} of the ${m} are genuinely non-null). `) +
        `A single test's smallest p-value would be uniform on (0, 1); with ${m} of them the distribution has collapsed ` +
        `toward zero, and the median smallest p-value is <b>${ST.sig(ST.median(mins), 3)}</b>. ` +
        `That is why the reported p-value of "the most striking finding" answers a question nobody asked.`);
    }
  }
  [mS, m1S, aS, vS].forEach(s => s.on("input change", draw));
  d3.select("#lk-run").on("click", () => { seed = (seed * 1103515245 + 12345) >>> 0; draw(); });
  draw();
})();

/* ══════════ 13 · #hl-svg — Bonferroni's flat line against Holm's staircase ══════════ */
(function () {
  const svg = d3.select("#hl-svg");
  if (svg.empty()) return;
  const W = 740, H = 420;
  const mS = d3.select("#hl-m"), m1S = d3.select("#hl-m1"), aS = d3.select("#hl-a"), alsoS = d3.select("#hl-also");
  const EX = [0.008, 0.011, 0.030, 0.041, 0.190];
  let ps = null, isNull = null, seed = 4711, useEx = false, dragIdx = null;
  function build() {
    const m = +mS.property("value"), m1 = Math.min(+m1S.property("value"), m);
    if (useEx) { ps = EX.slice(); isNull = ps.map((v, i) => i >= 2); mS.property("value", 5); return; }
    const r = ST.rng(seed);
    ps = []; isNull = [];
    for (let i = 0; i < m; i++) {
      const nn = i < m1;
      ps.push(2 * (1 - ST.normCdf(Math.abs(ST.randn(r) + (nn ? 3.0 : 0)))));
      isNull.push(!nn);
    }
  }
  function draw() {
    if (!ps) build();
    const m = ps.length, alpha = +aS.property("value"), also = alsoS.property("value");
    d3.select("#hl-mv").text(m); d3.select("#hl-m1v").text(+m1S.property("value"));
    const ord = ps.map((v, i) => i).sort((a, b) => ps[a] - ps[b]);
    const sorted = ord.map(i => ps[i]);
    const f = ST.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 46 });
    const g = f.g;
    const top = Math.max(alpha * 1.6, d3.max(sorted.slice(0, Math.min(m, Math.ceil(m * 0.6)))) * 1.15, 0.06);
    const x = d3.scalePoint().domain(d3.range(1, m + 1)).range([0, f.iw]).padding(0.5);
    const y = d3.scaleLinear().domain([0, top]).range([f.ih, 0]).clamp(true);
    ST.gridY(g, y, f.iw, 5);
    ST.axisL(g, y, 5, "p-value");
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`)
      .call(d3.axisBottom(x).tickValues(d3.range(1, m + 1).filter(d => m <= 12 || d % 2 === 1)));
    g.append("text").attr("x", f.iw).attr("y", f.ih + 34).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", SC.muted).text("rank i of the sorted p-values");
    /* Bonferroni: a flat line */
    g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(alpha / m)).attr("y2", y(alpha / m))
      .attr("stroke", SC.accent).attr("stroke-width", 2);
    g.append("text").attr("x", 2).attr("y", y(alpha / m) - 5).attr("font-size", 10).attr("fill", SC.accent)
      .text("Bonferroni α/m = " + ST.sig(alpha / m, 3));
    /* Holm: a rising staircase */
    const hthr = d3.range(m).map(i => alpha / (m - i));
    const step = d3.line().curve(d3.curveStepAfter)
      .x((d, i) => x(i + 1)).y(d => y(Math.min(d, top)));
    g.append("path").attr("d", step(hthr)).attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 2.2);
    g.append("text").attr("x", f.iw - 2).attr("y", y(Math.min(hthr[m - 1], top)) - 6).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.violet).text("Holm α/(m − i + 1)");
    if (also === "bh") {
      const bhl = d3.range(1, m + 1).map(i => [i, i / m * alpha]);
      g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(Math.min(d[1], top)))(bhl))
        .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "5,4");
      g.append("text").attr("x", f.iw - 2).attr("y", y(Math.min(alpha, top)) + 14).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", SC.a2).text("Benjamini–Hochberg (k/m)α");
    } else if (also === "sid") {
      const sd = AV.sidak(alpha, m);
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(sd)).attr("y2", y(sd))
        .attr("stroke", SC.teal).attr("stroke-width", 1.6).attr("stroke-dasharray", "4,3");
      g.append("text").attr("x", f.iw - 2).attr("y", y(sd) - 5).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", SC.teal).text("Šidák " + ST.sig(sd, 4));
    }
    /* Holm's stop point */
    let stop = m;
    for (let i = 0; i < m; i++) if (sorted[i] > hthr[i]) { stop = i; break; }
    if (stop < m) {
      g.append("line").attr("x1", x(stop + 1)).attr("x2", x(stop + 1)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "3,3");
      g.append("text").attr("x", x(stop + 1) + 5).attr("y", 14).attr("font-size", 10).attr("fill", SC.bad)
        .text("Holm stops here");
    }
    const holmRej = AV.holm(ps, alpha), bonfRej = ps.map(v => v <= alpha / m), bhRes = AV.bh(ps, alpha);
    /* the points, draggable */
    const drag = d3.drag()
      .on("start", function (ev, d) { dragIdx = d.orig; })
      .on("drag", function (ev) {
        if (dragIdx === null) return;
        useEx = false;
        ps[dragIdx] = ST.clamp(y.invert(ev.y), 1e-6, 1);
        draw();
      })
      .on("end", () => { dragIdx = null; });
    const pts = sorted.map((v, i) => ({ rank: i + 1, p: v, orig: ord[i] }));
    g.selectAll("circle.p").data(pts).join("circle").attr("class", "p")
      .attr("cx", d => x(d.rank)).attr("cy", d => y(Math.min(d.p, top)))
      .attr("r", d => d.p > top ? 3 : 5.5)
      .attr("fill", d => holmRej[d.orig] ? SC.good : (bonfRej[d.orig] ? SC.accent : SC.muted))
      .attr("fill-opacity", 0.85)
      .attr("stroke", d => isNull[d.orig] ? SC.line : SC.a2)
      .attr("stroke-width", d => isNull[d.orig] ? 1 : 2.4)
      .style("cursor", "ns-resize")
      .call(drag);
    ST.legend(g, [{ label: "rejected", color: SC.good },
                  { label: "retained", color: SC.muted },
                  { label: "truly non-null (thick ring)", color: SC.a2 }], 8, 14);
    const nb = bonfRej.filter(Boolean).length, nh = holmRej.filter(Boolean).length;
    const fb = bonfRej.filter((r, i) => r && isNull[i]).length, fh = holmRej.filter((r, i) => r && isNull[i]).length;
    d3.select("#hl-readout").html(
      `m = ${m}, α = ${alpha}. <b>Bonferroni rejects ${nb}</b> (${fb} of them false); ` +
      `<b>Holm rejects ${nh}</b> (${fh} false); BH would reject ${bhRes.cut}. ` +
      `Holm's first threshold <i>is</i> α/m = ${ST.sig(alpha / m, 3)}, so it never rejects fewer — ` +
      `it can only add. Its walk stops at rank ${stop + 1}${stop < m ? `, because p₍${stop + 1}₎ = ${ST.sig(sorted[stop], 3)} > α/(m − ${stop}) = ${ST.sig(hthr[stop], 3)}` : " (it never stopped)"}. ` +
      `<b>Drag any point vertically.</b>`);
  }
  [mS, m1S].forEach(s => s.on("input", () => { useEx = false; build(); draw(); }));
  aS.on("change", draw); alsoS.on("change", draw);
  d3.select("#hl-new").on("click", () => { useEx = false; seed = (seed * 1103515245 + 12345) >>> 0; build(); draw(); });
  d3.select("#hl-ex").on("click", () => { useEx = true; build(); useEx = false; draw(); });
  build(); draw();
})();

/* ══════════ 14 · #tk-svg — the studentised range and its intervals ══════════ */
(function () {
  const svg = d3.select("#tk-svg");
  if (svg.empty()) return;
  const W = 740, H = 430;
  const kS = d3.select("#tk-k"), vS = d3.select("#tk-v"), aS = d3.select("#tk-a"), viewS = d3.select("#tk-view");
  function draw() {
    const k = +kS.property("value"), nu = +vS.property("value"), alpha = +aS.property("value");
    d3.select("#tk-kv").text(k); d3.select("#tk-vv").text(nu);
    const view = viewS.property("value");
    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 26, b: 44 });
    const g = f.g, LW = 360, gap = 48;
    const qc = AV.QR.qQuant(1 - alpha, k, nu);
    const C = k * (k - 1) / 2;
    const tU = ST.tQuant(1 - alpha / 2, nu), tB = ST.tQuant(1 - alpha / (2 * C), nu);
    const sch = Math.sqrt((k - 1) * ST.fQuant(1 - alpha, k - 1, nu));
    /* left: the q density */
    const xmax = qc * 1.7;
    const xs = ST.linspace(0.02, xmax, 150);
    const dens = xs.map(v => AV.QR.qPdf(v, k, nu));
    const x = d3.scaleLinear().domain([0, xmax]).range([0, LW]);
    const y = d3.scaleLinear().domain([0, d3.max(dens) * 1.18]).range([f.ih, 0]);
    ST.gridY(g, y, LW, 5);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(6));
    const tail = xs.filter(v => v >= qc);
    if (tail.length) g.append("path")
      .attr("d", d3.area().x(v => x(v)).y0(f.ih).y1(v => y(AV.QR.qPdf(v, k, nu)))(tail))
      .attr("fill", SC.bad).attr("fill-opacity", 0.3);
    g.append("path").attr("d", d3.line().x((v, i) => x(xs[i])).y((v, i) => y(dens[i]))(xs))
      .attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2.4);
    g.append("line").attr("x1", x(qc)).attr("x2", x(qc)).attr("y1", f.ih).attr("y2", 0)
      .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "5,4");
    g.append("text").attr("x", x(qc) + 5).attr("y", 14).attr("font-size", 10.5).attr("fill", SC.good)
      .text("q = " + ST.fmt(qc, 4));
    /* the competing critical values, put on the same q scale by ×√2 */
    [[tU * Math.SQRT2, "unadjusted t·√2", SC.bad], [tB * Math.SQRT2, "Bonferroni t·√2", SC.accent],
     [sch * Math.SQRT2, "Scheffé·√2", SC.violet]].forEach((c, i) => {
      if (c[0] > xmax) return;
      g.append("line").attr("x1", x(c[0])).attr("x2", x(c[0])).attr("y1", f.ih).attr("y2", f.ih - 30 - i * 15)
        .attr("stroke", c[2]).attr("stroke-width", 1.5);
      g.append("text").attr("x", x(c[0]) + 4).attr("y", f.ih - 32 - i * 15).attr("font-size", 9.5)
        .attr("fill", c[2]).text(c[1] + " = " + ST.fmt(c[0], 3));
    });
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text(`the studentised range q(${k}, ${nu}) — computed by quadrature, not tabulated`);
    /* right */
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    if (view === "ci") {
      const a = AV.anova(AV.RUN.y), pairs = [[0, 1], [0, 2], [1, 2]];
      const qq = AV.QR.qQuant(1 - alpha, 3, a.df2);
      const items = pairs.map(([i, j]) => {
        const d = a.means[i] - a.means[j];
        const hw = qq * a.s * Math.sqrt(0.5 * (1 / a.ns[i] + 1 / a.ns[j]));
        const se = a.s * Math.sqrt(1 / a.ns[i] + 1 / a.ns[j]);
        const q = Math.abs(d) / (a.s * Math.sqrt(0.5 * (1 / a.ns[i] + 1 / a.ns[j])));
        return { lab: AV.RUN.short[i] + " − " + AV.RUN.short[j], d: d, hw: hw, q: q,
                 p: AV.QR.qSf(q, 3, a.df2), raw: 2 * (1 - ST.tCdf(Math.abs(d) / se, a.df2)) };
      });
      const lim = d3.max(items, it => Math.abs(it.d) + it.hw) * 1.12;
      const x2 = d3.scaleLinear().domain([-lim, lim]).range([0, RW]);
      const y2 = d3.scalePoint().domain(items.map(it => it.lab)).range([34, f.ih - 46]).padding(0.5);
      g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih - 20})`).call(d3.axisBottom(x2).ticks(5));
      g2.append("line").attr("x1", x2(0)).attr("x2", x2(0)).attr("y1", 24).attr("y2", f.ih - 20)
        .attr("stroke", SC.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
      items.forEach(it => {
        const yy = y2(it.lab), sig = Math.abs(it.d) > it.hw;
        g2.append("line").attr("x1", x2(it.d - it.hw)).attr("x2", x2(it.d + it.hw)).attr("y1", yy).attr("y2", yy)
          .attr("stroke", sig ? SC.good : SC.a2).attr("stroke-width", 3).attr("stroke-linecap", "round");
        g2.append("circle").attr("cx", x2(it.d)).attr("cy", yy).attr("r", 4.2).attr("fill", sig ? SC.good : SC.a2);
        g2.append("text").attr("x", 0).attr("y", yy - 11).attr("font-size", 10.5).attr("fill", SC.ink).text(it.lab);
        g2.append("text").attr("x", RW).attr("y", yy - 11).attr("text-anchor", "end").attr("font-size", 9.5)
          .attr("fill", SC.muted).text(`Tukey p ${ST.fmt(it.p, 4)} · raw ${ST.fmt(it.raw, 4)}`);
      });
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(`${ST.pct(1 - alpha, 0)} SIMULTANEOUS intervals on the running data`);
      d3.select("#tk-readout").html(
        `q<sub>${ST.pct(1 - alpha, 0)}</sub>(3, 38) = <b>${ST.fmt(AV.QR.qQuant(1 - alpha, 3, 38), 6)}</b>. ` +
        `The PA-vs-Study comparison has raw p = <b>${ST.fmt(items[2].raw, 4)}</b> and Tukey p = <b>${ST.fmt(items[2].p, 4)}</b> — ` +
        `significant unadjusted, not significant once you allow for having picked the largest of three. ` +
        `The omnibus F gave p = 0.088834.`);
    } else {
      const ks = d3.range(2, 13);
      const series = [
        { lab: "unadjusted t", col: SC.bad, f: kk => ST.tQuant(1 - alpha / 2, nu) * Math.SQRT2 },
        { lab: "Tukey q", col: SC.good, f: kk => AV.QR.qQuant(1 - alpha, kk, nu) },
        { lab: "Bonferroni t", col: SC.accent, f: kk => ST.tQuant(1 - alpha / (kk * (kk - 1)), nu) * Math.SQRT2 },
        { lab: "Scheffé", col: SC.violet, f: kk => Math.sqrt((kk - 1) * ST.fQuant(1 - alpha, kk - 1, nu)) * Math.SQRT2 }
      ];
      const vals = series.map(s => ks.map(kk => [kk, s.f(kk)]));
      const x2 = d3.scaleLinear().domain([2, 12]).range([0, RW]);
      const y2 = d3.scaleLinear().domain([0, d3.max(vals, v => d3.max(v, d => d[1])) * 1.1]).range([f.ih, 0]);
      ST.gridY(g2, y2, RW, 5);
      g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x2).ticks(6));
      g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5));
      series.forEach((s, i) => {
        g2.append("path").attr("d", d3.line().x(d => x2(d[0])).y(d => y2(d[1]))(vals[i]))
          .attr("fill", "none").attr("stroke", s.col).attr("stroke-width", 2.2);
      });
      g2.append("line").attr("x1", x2(k)).attr("x2", x2(k)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.muted).attr("stroke-dasharray", "3,3");
      ST.legend(g2, series.map(s => ({ label: s.lab, color: s.col })), 8, 14);
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(`critical half-widths in units of SE/√2 · ν = ${nu}`);
      d3.select("#tk-readout").html(
        `At k = ${k}, ν = ${nu}, α = ${alpha}: Tukey <b>${ST.fmt(qc, 4)}</b>, ` +
        `Bonferroni <b>${ST.fmt(tB * Math.SQRT2, 4)}</b>, Scheffé <b>${ST.fmt(sch * Math.SQRT2, 4)}</b>, ` +
        `unadjusted <b>${ST.fmt(tU * Math.SQRT2, 4)}</b> — all on the q scale so they are comparable. ` +
        `Tukey beats Bonferroni for pairwise work because it is <i>exact</i> for exactly that family; ` +
        `Scheffé overtakes Bonferroni once the family is large because its threshold does not grow with it.`);
    }
  }
  [kS, vS, aS, viewS].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══════════ 15 · #cn-svg — the contrast explorer ══════════ */
(function () {
  const svg = d3.select("#cn-svg");
  if (svg.empty()) return;
  const W = 740, H = 430;
  const preS = d3.select("#cn-pre"), c1S = d3.select("#cn-c1"), c2S = d3.select("#cn-c2"), critS = d3.select("#cn-crit");
  const PRE = { pa: [-1, 1], hw: [0.5, 0.5], mid: [-0.5, 1], lin: [-1, 0], max: null };
  const a = AV.anova(AV.RUN.y);
  function setPreset() {
    const p = preS.property("value");
    let w;
    if (p === "max") {
      const raw = a.means.map((m, j) => a.ns[j] * (m - a.gm));
      const sc = 1 / Math.max.apply(null, raw.map(Math.abs));
      w = [raw[0] * sc, raw[1] * sc];
    } else w = PRE[p];
    c1S.property("value", Math.round(w[0] * 10)); c2S.property("value", Math.round(w[1] * 10));
  }
  function draw() {
    const c1 = +c1S.property("value") / 10, c2 = +c2S.property("value") / 10, c3 = -(c1 + c2);
    d3.select("#cn-c1v").text(c1.toFixed(2)); d3.select("#cn-c2v").text(c2.toFixed(2));
    const c = [c1, c2, c3];
    const crit = critS.property("value");
    let dsum = 0, L = 0;
    for (let j = 0; j < 3; j++) { dsum += c[j] * c[j] / a.ns[j]; L += c[j] * a.means[j]; }
    const se = a.s * Math.sqrt(dsum), t = dsum > 0 ? L / se : 0;
    const SSc = dsum > 0 ? L * L / dsum : 0;
    const p = 2 * (1 - ST.tCdf(Math.abs(t), a.df2));
    let cv, cvLab;
    if (crit === "t") { cv = ST.tQuant(0.975, a.df2); cvLab = "unadjusted t*"; }
    else if (crit === "bonf") { cv = ST.tQuant(1 - 0.05 / 6, a.df2); cvLab = "Bonferroni t* over 3"; }
    else if (crit === "tukey") { cv = AV.QR.qQuant(0.95, 3, a.df2) / Math.SQRT2; cvLab = "Tukey q/√2"; }
    else { cv = Math.sqrt(2 * ST.fQuant(0.95, 2, a.df2)); cvLab = "Scheffé √((k−1)F*)"; }
    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 26, b: 44 });
    const g = f.g, LW = 320, gap = 50;
    /* left: the means, and the weights as signed bars */
    const y = d3.scaleLinear().domain([43, 51]).range([f.ih - 100, 20]);
    const bx = d3.scalePoint().domain([0, 1, 2]).range([60, LW - 40]).padding(0.5);
    ST.gridY(g, y, LW, 4);
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    g.append("line").attr("x1", 0).attr("x2", LW).attr("y1", y(a.gm)).attr("y2", y(a.gm))
      .attr("stroke", SC.muted).attr("stroke-dasharray", "5,4");
    a.means.forEach((m, j) => {
      const semj = a.s / Math.sqrt(a.ns[j]);
      g.append("line").attr("x1", bx(j)).attr("x2", bx(j)).attr("y1", y(m - semj)).attr("y2", y(m + semj))
        .attr("stroke", AV.GC[j]).attr("stroke-width", 2.6).attr("stroke-linecap", "round");
      g.append("circle").attr("cx", bx(j)).attr("cy", y(m)).attr("r", 4.5).attr("fill", AV.GC[j]);
      g.append("text").attr("x", bx(j)).attr("y", y(m) - 12).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", AV.GC[j]).text(ST.fmt(m, 2));
    });
    const wy = f.ih - 46, wmax = Math.max(1e-9, d3.max(c.map(Math.abs)));
    g.append("line").attr("x1", 20).attr("x2", LW - 10).attr("y1", wy).attr("y2", wy)
      .attr("stroke", SC.line).attr("stroke-width", 1.2);
    c.forEach((w, j) => {
      const hh = 34 * w / wmax;
      g.append("rect").attr("x", bx(j) - 16).attr("y", w >= 0 ? wy - hh : wy).attr("width", 32)
        .attr("height", Math.abs(hh)).attr("rx", 2)
        .attr("fill", w >= 0 ? SC.good : SC.bad).attr("fill-opacity", 0.55)
        .attr("stroke", w >= 0 ? SC.good : SC.bad);
      g.append("text").attr("x", bx(j)).attr("y", wy + (w >= 0 ? 14 : -hh + 46)).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", SC.muted).text(AV.RUN.short[j]);
      g.append("text").attr("x", bx(j)).attr("y", w >= 0 ? wy - hh - 5 : wy + Math.abs(hh) + 12)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", w >= 0 ? SC.good : SC.bad)
        .text((w >= 0 ? "+" : "") + w.toFixed(2));
    });
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text("the group means ± 1 SE, and the contrast weights below (Σcⱼ = " + ST.fmt(c1 + c2 + c3, 3) + ")");
    /* right: the estimate with its interval, and the SS partition */
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    const hw = cv * se, lim = Math.max(Math.abs(L) + hw, hw) * 1.2;
    const x2 = d3.scaleLinear().domain([-lim, lim]).range([0, RW]);
    const yy = 74;
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${yy + 30})`).call(d3.axisBottom(x2).ticks(5));
    g2.append("line").attr("x1", x2(0)).attr("x2", x2(0)).attr("y1", yy - 34).attr("y2", yy + 30)
      .attr("stroke", SC.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
    const sig = Math.abs(L) > hw;
    g2.append("line").attr("x1", x2(L - hw)).attr("x2", x2(L + hw)).attr("y1", yy).attr("y2", yy)
      .attr("stroke", sig ? SC.good : SC.a2).attr("stroke-width", 3.4).attr("stroke-linecap", "round");
    g2.append("circle").attr("cx", x2(L)).attr("cy", yy).attr("r", 5).attr("fill", sig ? SC.good : SC.a2);
    g2.append("text").attr("x", 0).attr("y", yy - 44).attr("font-size", 11).attr("fill", SC.muted)
      .text("L̂ with its " + cvLab + " interval");
    g2.append("text").attr("x", x2(L)).attr("y", yy - 12).attr("text-anchor", "middle").attr("font-size", 11)
      .attr("fill", SC.ink).text("L̂ = " + ST.fmt(L, 4));
    /* the SS partition bar */
    const barY = yy + 92, barH = 26;
    const share = ST.clamp(SSc / a.ssb, 0, 1);
    g2.append("rect").attr("x", 0).attr("y", barY).attr("width", RW).attr("height", barH).attr("rx", 4)
      .attr("fill", SC.panel2).attr("stroke", SC.line);
    g2.append("rect").attr("x", 0).attr("y", barY).attr("width", RW * share).attr("height", barH).attr("rx", 4)
      .attr("fill", SC.a2).attr("fill-opacity", 0.55);
    g2.append("text").attr("x", 6).attr("y", barY - 7).attr("font-size", 10.5).attr("fill", SC.muted)
      .text("SS(L) as a share of SS_between = " + ST.fmt(a.ssb, 4));
    g2.append("text").attr("x", 6).attr("y", barY + 17).attr("font-size", 11).attr("fill", SC.ink)
      .text(ST.fmt(SSc, 4) + "  (" + ST.pct(share, 1) + ")");
    AV.panel(g2, 0, barY + 52, RW, [
      ["SE(L̂)", ST.fmt(se, 6)],
      ["t on " + a.df2 + " df", ST.fmt(t, 6)],
      ["unadjusted p", AV.pfmt(p)],
      ["critical value", ST.fmt(cv, 6) + "  (" + cvLab + ")"],
      ["|t| > critical?", sig ? "YES" : "no", sig ? SC.good : SC.bad]
    ]);
    d3.select("#cn-readout").html(
      `Weights (${c.map(v => (v >= 0 ? "+" : "") + v.toFixed(2)).join(", ")}), Σcⱼ = ${ST.fmt(c1 + c2 + c3, 4)}. ` +
      `L̂ = <b>${ST.fmt(L, 4)}</b>, SE = ${ST.fmt(se, 4)}, t = <b>${ST.fmt(t, 4)}</b> on ${a.df2} df, unadjusted p = <b>${AV.pfmt(p)}</b>. ` +
      `SS(L) = <b>${ST.fmt(SSc, 4)}</b>, which is ${ST.pct(share, 1)} of SS_between; the orthogonal complement carries the remaining ` +
      `${ST.fmt(a.ssb - SSc, 4)}. ` +
      `<b>Planned</b> (write it down first) → the unadjusted t is correct. <b>Post hoc</b> (chosen after looking) → Scheffé, and note that ` +
      `no contrast can be Scheffé-significant unless the omnibus F rejects, which here it does not.`);
  }
  preS.on("change", () => { setPreset(); draw(); });
  [c1S, c2S, critS].forEach(s => s.on("input change", draw));
  setPreset(); draw();
})();

/* ══════════ 16 · #bh-svg — the Benjamini–Hochberg staircase ══════════ */
(function () {
  const svg = d3.select("#bh-svg");
  if (svg.empty()) return;
  const W = 740, H = 450;
  const mS = d3.select("#bh-m"), p0S = d3.select("#bh-p0"), effS = d3.select("#bh-eff"),
        aS = d3.select("#bh-a"), vS = d3.select("#bh-view");
  let seed = 1995, acc = null;
  function study(m, p0, eff, r) {
    const m0 = Math.round(p0 * m), p = [], isN = [];
    for (let i = 0; i < m; i++) {
      const nn = i >= m0;
      p.push(2 * (1 - ST.normCdf(Math.abs(ST.randn(r) + (nn ? eff : 0)))));
      isN.push(!nn);
    }
    return { p: p, isN: isN, m0: m0 };
  }
  function accumulate(reps) {
    const m = +mS.property("value"), eff = +effS.property("value") / 10, alpha = +aS.property("value");
    const grid = ST.linspace(0, 1, 11);
    if (!acc || acc.m !== m || acc.eff !== eff || acc.alpha !== alpha)
      acc = { m: m, eff: eff, alpha: alpha, n: grid.map(() => 0),
              fdrB: grid.map(() => 0), fwerB: grid.map(() => 0),
              fdrO: grid.map(() => 0), fwerO: grid.map(() => 0), grid: grid };
    const r = ST.rng(seed);
    for (let t = 0; t < reps; t++) {
      const gi = t % grid.length, p0 = grid[gi];
      const s = study(m, p0, eff, r);
      const res = AV.bh(s.p, alpha);
      let R = 0, V = 0;
      for (let i = 0; i < m; i++) if (res.rej[i]) { R++; if (s.isN[i]) V++; }
      acc.fdrB[gi] += R ? V / R : 0; acc.fwerB[gi] += V >= 1 ? 1 : 0;
      let Rb = 0, Vb = 0;
      for (let i = 0; i < m; i++) if (s.p[i] <= alpha / m) { Rb++; if (s.isN[i]) Vb++; }
      acc.fdrO[gi] += Rb ? Vb / Rb : 0; acc.fwerO[gi] += Vb >= 1 ? 1 : 0;
      acc.n[gi]++;
    }
    seed = (seed * 1103515245 + 12345) >>> 0;
  }
  function draw() {
    const m = +mS.property("value"), p0 = +p0S.property("value") / 100;
    const eff = +effS.property("value") / 10, alpha = +aS.property("value");
    d3.select("#bh-p0v").text(p0.toFixed(2)); d3.select("#bh-effv").text(eff.toFixed(1));
    const view = vS.property("value");
    const f = ST.frame(svg, W, H, { l: 52, r: 16, t: 26, b: 46 });
    const g = f.g, LW = 350, gap = 52;
    const s = study(m, p0, eff, ST.rng(seed));
    const res = AV.bh(s.p, alpha);
    const sortedIdx = res.order, sorted = sortedIdx.map(i => s.p[i]);
    const showR = Math.min(m, Math.max(20, res.cut * 3, Math.ceil(m * 0.12)));
    const x = d3.scaleLinear().domain([0, showR]).range([0, LW]);
    const top = Math.max(alpha * 1.25, sorted[Math.min(showR - 1, m - 1)] * 1.1);
    const y = d3.scaleLinear().domain([0, top]).range([f.ih, 0]).clamp(true);
    ST.gridY(g, y, LW, 5);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5, ".3f"));
    g.append("text").attr("x", LW).attr("y", f.ih + 34).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", SC.muted).text("rank k of the sorted p-values (first " + showR + " of " + m + ")");
    g.append("line").attr("x1", x(0)).attr("x2", x(showR)).attr("y1", y(0)).attr("y2", y(Math.min(showR / m * alpha, top)))
      .attr("stroke", SC.a2).attr("stroke-width", 2.4);
    g.append("text").attr("x", x(showR) - 4).attr("y", y(Math.min(showR / m * alpha, top)) - 6).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.a2).text("the BH ramp (k/m)·α");
    g.append("line").attr("x1", 0).attr("x2", LW).attr("y1", y(alpha / m)).attr("y2", y(alpha / m))
      .attr("stroke", SC.accent).attr("stroke-width", 1.8);
    g.append("text").attr("x", 3).attr("y", y(alpha / m) - 5).attr("font-size", 9.5).attr("fill", SC.accent)
      .text("Bonferroni α/m");
    if (res.cut > 0 && res.cut <= showR) {
      g.append("line").attr("x1", x(res.cut)).attr("x2", x(res.cut)).attr("y1", f.ih).attr("y2", 0)
        .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "4,3");
      g.append("text").attr("x", x(res.cut) + 5).attr("y", 14).attr("font-size", 10.5).attr("fill", SC.good)
        .text("cut at k = " + res.cut);
    }
    for (let i = 0; i < Math.min(showR, m); i++) {
      const gi = sortedIdx[i], rej = res.rej[gi];
      g.append("circle").attr("cx", x(i + 1)).attr("cy", y(Math.min(sorted[i], top)))
        .attr("r", sorted[i] > top ? 2 : 4)
        .attr("fill", rej ? (s.isN[gi] ? SC.bad : SC.good) : SC.muted).attr("fill-opacity", 0.85)
        .attr("stroke", s.isN[gi] ? "none" : SC.a2).attr("stroke-width", 1.6);
    }
    ST.legend(g, [{ label: "true discovery", color: SC.good }, { label: "FALSE discovery", color: SC.bad },
                  { label: "retained", color: SC.muted }], 8, 14);
    let R = 0, V = 0;
    for (let i = 0; i < m; i++) if (res.rej[i]) { R++; if (s.isN[i]) V++; }
    /* right */
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    if (view === "hist") {
      const bins = ST.histBins(s.p, 0, 1 / 20, 20);
      const x2 = d3.scaleLinear().domain([0, 1]).range([0, RW]);
      const y2 = d3.scaleLinear().domain([0, d3.max(bins, b => b.n) * 1.14]).range([f.ih, 0]);
      ST.gridY(g2, y2, RW, 5);
      g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x2).ticks(5));
      g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5));
      g2.selectAll("rect.h").data(bins).join("rect").attr("class", "h")
        .attr("x", b => x2(b.x0)).attr("y", b => y2(b.n)).attr("width", RW / 20 - 1.5)
        .attr("height", b => f.ih - y2(b.n)).attr("fill", SC.violet).attr("fill-opacity", 0.5);
      g2.append("line").attr("x1", 0).attr("x2", RW).attr("y1", y2(s.m0 / 20)).attr("y2", y2(s.m0 / 20))
        .attr("stroke", SC.muted).attr("stroke-dasharray", "5,4");
      g2.append("text").attr("x", RW - 4).attr("y", y2(s.m0 / 20) - 5).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", SC.muted).text("the uniform slab from the " + s.m0 + " true nulls");
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text("all m p-values — flat plus a spike at zero");
    } else {
      const x2 = d3.scaleLinear().domain([0, 1]).range([0, RW]);
      const y2 = d3.scaleLinear().domain([0, Math.max(alpha * 1.6, 0.08)]).range([f.ih, 0]);
      ST.gridY(g2, y2, RW, 5);
      g2.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x2).ticks(5));
      g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5).tickFormat(d => ST.pct(d, 1)));
      g2.append("text").attr("x", RW).attr("y", f.ih + 34).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", SC.muted).text("π₀, the fraction truly null");
      g2.append("line").attr("x1", x2(0)).attr("x2", x2(1)).attr("y1", y2(0)).attr("y2", y2(alpha))
        .attr("stroke", SC.muted).attr("stroke-width", 1.6).attr("stroke-dasharray", "5,4");
      g2.append("text").attr("x", RW - 4).attr("y", y2(alpha) + 14).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", SC.muted).text("the bound π₀·α");
      if (acc) {
        const li = d3.line().x(d => x2(d[0])).y(d => y2(Math.min(d[1], y2.domain()[1])));
        const mk = arr => acc.grid.map((gv, i) => [gv, acc.n[i] ? arr[i] / acc.n[i] : 0]).filter((d, i) => acc.n[i] > 0);
        [[mk(acc.fdrB), SC.good, "BH — measured FDR"], [mk(acc.fwerB), SC.bad, "BH — measured FWER"],
         [mk(acc.fdrO), SC.accent, "Bonferroni — measured FDR"]].forEach(sd => {
          if (sd[0].length > 1) g2.append("path").attr("d", li(sd[0])).attr("fill", "none")
            .attr("stroke", sd[1]).attr("stroke-width", 2.2);
          sd[0].forEach(d => g2.append("circle").attr("cx", x2(d[0])).attr("cy", y2(Math.min(d[1], y2.domain()[1])))
            .attr("r", 2.6).attr("fill", sd[1]));
        });
        ST.legend(g2, [{ label: "BH: measured FDR", color: SC.good },
                       { label: "BH: measured FWER", color: SC.bad },
                       { label: "Bonferroni: measured FDR", color: SC.accent }], 8, 14);
      }
      const done = acc ? acc.n.reduce((a, b) => a + b, 0) : 0;
      g2.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
        .text(done ? `${done.toLocaleString()} simulated studies` : 'press "accumulate 400 studies"');
    }
    const bonfR = s.p.filter(v => v <= alpha / m).length;
    d3.select("#bh-readout").html(
      `One study: m = ${m}, π₀ = ${p0.toFixed(2)} (${s.m0} true nulls), effect ${eff.toFixed(1)}σ, α = ${alpha}. ` +
      `<b>BH rejects ${R}</b>, of which <b style="color:${SC.bad}">${V} are false</b> — FDP = <b>${R ? ST.fmt(V / R, 4) : "—"}</b>. ` +
      `Bonferroni would reject <b>${bonfR}</b>. ` +
      `BH's guarantee is E[FDP] ≤ π₀·α = <b>${ST.fmt(p0 * alpha, 4)}</b>, <i>not</i> α — accumulate studies to watch the ` +
      `measured curve sit on that line. Note that BH rejects everything up to the cut, including p-values above their own thresholds.`);
  }
  [mS, p0S, effS, aS, vS].forEach(s => s.on("input change", () => { if (s !== p0S && s !== vS) acc = null; draw(); }));
  d3.select("#bh-new").on("click", () => { seed = (seed * 1103515245 + 12345) >>> 0; draw(); });
  d3.select("#bh-run").on("click", () => { accumulate(440); vS.property("value", "rates"); draw(); });
  draw();
})();

/* ══════════ 17 · #in-svg — the interaction plot ══════════ */
(function () {
  const svg = d3.select("#in-svg");
  if (svg.empty()) return;
  const W = 740, H = 440;
  const preS = d3.select("#in-pre"), cellS = d3.select("#in-cell"), vS = d3.select("#in-v"), sS = d3.select("#in-s");
  const PRE = { cross: [[44, 50], [50, 50]], add: [[44, 50], [50, 56]], rev: [[44, 52], [52, 44]], none: [[48.5, 48.5], [48.5, 48.5]] };
  let mu = PRE.cross.map(r => r.slice());
  const n = 12;
  function build(sigma) {
    const r = ST.rng(20260907), Y = [];
    for (let i = 0; i < 2; i++) {
      const row = [];
      for (let j = 0; j < 2; j++) {
        const g = [], zs = [];
        for (let q = 0; q < n; q++) zs.push(ST.randn(r));
        const mz = ST.mean(zs), sz = ST.sd(zs, 0);       // centre and scale so the cell mean is exact
        for (let q = 0; q < n; q++) g.push(mu[i][j] + sigma * (zs[q] - mz) / (sz || 1));
        row.push(g);
      }
      Y.push(row);
    }
    return Y;
  }
  function twoway(Y) {
    const flat = [].concat.apply([], [].concat.apply([], Y)), N = flat.length, gm = ST.mean(flat);
    const cell = Y.map(row => row.map(ST.mean));
    const rowm = [0, 1].map(i => (cell[i][0] + cell[i][1]) / 2);
    const colm = [0, 1].map(j => (cell[0][j] + cell[1][j]) / 2);
    const SSA = 2 * n * rowm.reduce((s, v) => s + (v - gm) * (v - gm), 0);
    const SSB = 2 * n * colm.reduce((s, v) => s + (v - gm) * (v - gm), 0);
    let SSAB = 0, SSE = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      const d = cell[i][j] - rowm[i] - colm[j] + gm;
      SSAB += n * d * d;
      Y[i][j].forEach(v => { SSE += (v - cell[i][j]) * (v - cell[i][j]); });
    }
    const dfe = 4 * (n - 1), MSE = SSE / dfe;
    const F = ss => ss / MSE, P = ss => 1 - ST.fCdf(F(ss), 1, dfe);
    return { gm: gm, cell: cell, rowm: rowm, colm: colm, SSA: SSA, SSB: SSB, SSAB: SSAB, SSE: SSE,
             MSE: MSE, dfe: dfe, F: F, P: P, N: N };
  }
  function draw() {
    const sigma = +sS.property("value") / 10;
    d3.select("#in-sv").text(sigma.toFixed(1));
    const which = cellS.property("value"), wi = +which[0], wj = +which[1];
    d3.select("#in-vv").text(mu[wi][wj].toFixed(1));
    const Y = build(sigma), a = twoway(Y);
    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 26, b: 46 });
    const g = f.g, LW = 330, gap = 50;
    const flat = [].concat.apply([], [].concat.apply([], Y));
    const y = d3.scaleLinear().domain([d3.min(flat) - 1, d3.max(flat) + 1]).nice().range([f.ih, 0]);
    const x = d3.scalePoint().domain([0, 1]).range([70, LW - 60]).padding(0.4);
    ST.gridY(g, y, LW, 5); ST.axisL(g, y, 5, "score");
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`)
      .call(d3.axisBottom(x).tickFormat(d => AV.TWOWAY.cols[d]));
    const se = Math.sqrt(a.MSE / n);
    [0, 1].forEach(i => {
      const col = i === 0 ? SC.accent : SC.a2;
      g.append("path").attr("d", d3.line().x((d, j) => x(j)).y((d, j) => y(a.cell[i][j]))([0, 1]))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2.6);
      [0, 1].forEach(j => {
        Y[i][j].forEach(v => g.append("circle").attr("cx", x(j) + (i === 0 ? -9 : 9)).attr("cy", y(v))
          .attr("r", 1.8).attr("fill", col).attr("fill-opacity", 0.3));
        g.append("line").attr("x1", x(j)).attr("x2", x(j))
          .attr("y1", y(a.cell[i][j] - se)).attr("y2", y(a.cell[i][j] + se))
          .attr("stroke", col).attr("stroke-width", 2.2);
        g.append("circle").attr("cx", x(j)).attr("cy", y(a.cell[i][j]))
          .attr("r", (i === wi && j === wj) ? 6.5 : 4.5).attr("fill", col)
          .attr("stroke", (i === wi && j === wj) ? SC.ink : "none").attr("stroke-width", 1.8);
      });
      g.append("text").attr("x", x(1) + 8).attr("y", y(a.cell[i][1]) + 4).attr("font-size", 10)
        .attr("fill", col).text(AV.TWOWAY.rows[i]);
    });
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.muted)
      .text("parallel lines ⟹ no interaction · the highlighted point is the draggable one");
    /* right: the two-way table */
    const g2 = g.append("g").attr("transform", `translate(${LW + gap},0)`);
    const RW = f.iw - LW - gap;
    const rows = [
      ["A · prior band", `${ST.fmt(a.SSA, 3)}   F=${ST.fmt(a.F(a.SSA), 3)}   p=${AV.pfmt(a.P(a.SSA))}`, a.P(a.SSA) < 0.05 ? SC.good : SC.ink],
      ["B · peer assessment", `${ST.fmt(a.SSB, 3)}   F=${ST.fmt(a.F(a.SSB), 3)}   p=${AV.pfmt(a.P(a.SSB))}`, a.P(a.SSB) < 0.05 ? SC.good : SC.ink],
      ["A × B · interaction", `${ST.fmt(a.SSAB, 3)}   F=${ST.fmt(a.F(a.SSAB), 3)}   p=${AV.pfmt(a.P(a.SSAB))}`, a.P(a.SSAB) < 0.05 ? SC.bad : SC.ink],
      ["error (" + a.dfe + " df)", `${ST.fmt(a.SSE, 3)}   MS=${ST.fmt(a.MSE, 4)}`],
      ["total", ST.fmt(a.SSA + a.SSB + a.SSAB + a.SSE, 3)]
    ];
    AV.panel(g2, 0, 6, RW, rows, "the two-way ANOVA table, each F on (1, " + a.dfe + ")");
    const s1 = a.cell[0][1] - a.cell[0][0], s2 = a.cell[1][1] - a.cell[1][0];
    const sed = Math.sqrt(2 * a.MSE / n);
    AV.panel(g2, 0, 132, RW, [
      ["simple effect, lower band", (s1 >= 0 ? "+" : "") + ST.fmt(s1, 3) + "   p=" + AV.pfmt(2 * (1 - ST.tCdf(Math.abs(s1 / sed), a.dfe))), SC.accent],
      ["simple effect, upper band", (s2 >= 0 ? "+" : "") + ST.fmt(s2, 3) + "   p=" + AV.pfmt(2 * (1 - ST.tCdf(Math.abs(s2 / sed), a.dfe))), SC.a2],
      ["MAIN effect of PA (their mean)", (((s1 + s2) / 2) >= 0 ? "+" : "") + ST.fmt((s1 + s2) / 2, 3), SC.bad],
      ["difference of the two", ST.fmt(s1 - s2, 3) + "  ← the interaction"],
      ["partial η² (interaction)", ST.fmt(a.SSAB / (a.SSAB + a.SSE), 4)]
    ], "what the interaction is about");
    const gap2 = Math.abs(s1 - s2);
    d3.select("#in-readout").html(
      `Cell means ${a.cell.map(r => r.map(v => ST.fmt(v, 1)).join("/")).join("  |  ")}. ` +
      `Peer assessment is worth <b>${(s1 >= 0 ? "+" : "") + ST.fmt(s1, 2)}</b> in the lower band and ` +
      `<b>${(s2 >= 0 ? "+" : "") + ST.fmt(s2, 2)}</b> in the upper. Its <b>main effect</b> is their average, ` +
      `<b>${(((s1 + s2) / 2) >= 0 ? "+" : "") + ST.fmt((s1 + s2) / 2, 2)}</b> — ` +
      (gap2 < 0.4 ? "and with the lines this close to parallel that average is a fair summary."
        : `a number that describes neither group. Interaction p = <b>${AV.pfmt(a.P(a.SSAB))}</b>; ` +
          `when it is small, report the two simple effects and treat the main effects as arithmetic.`));
  }
  preS.on("change", () => { mu = PRE[preS.property("value")].map(r => r.slice());
    const wi = +cellS.property("value")[0], wj = +cellS.property("value")[1];
    vS.property("value", Math.round(mu[wi][wj] * 10)); draw(); });
  cellS.on("change", () => { const wi = +cellS.property("value")[0], wj = +cellS.property("value")[1];
    vS.property("value", Math.round(mu[wi][wj] * 10)); draw(); });
  vS.on("input", () => { const wi = +cellS.property("value")[0], wj = +cellS.property("value")[1];
    mu[wi][wj] = +vS.property("value") / 10; draw(); });
  sS.on("input", draw);
  d3.select("#in-reset").on("click", () => {
    preS.property("value", "cross"); mu = PRE.cross.map(r => r.slice());
    cellS.property("value", "01"); vS.property("value", 500); sS.property("value", 45); draw();
  });
  draw();
})();
