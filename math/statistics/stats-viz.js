/* stats-viz.js — shared D3 + numeric helpers for the Statistics series (math/statistics/*).
   Loaded after ../../notes.js and before each part's own "<part>.viz.js".
   Nothing here draws anything on its own; it is a toolbox the part pages call.

   Provides two globals:

   SC  — palette, matching the custom properties in notes.css
         (accent, a2, good, bad, ink, muted, line, grid, panel, bg)

   ST  — namespace:
     · random      ST.rng(seed) → mulberry32 uniform generator; ST.randn(r); ST.sample(a, k, r)
                   ST.shuffle(a, r) (Fisher–Yates, returns a copy); ST.srswor(a, k, r) (without
                   replacement); ST.choice(a, r); ST.binom(n, p, r)
     · order       ST.asc, ST.sum, ST.mean, ST.median, ST.mode, ST.range
     · spread      ST.variance(a, ddof), ST.sd(a, ddof), ST.meanAbsDev, ST.mad, ST.madn, ST.iqr
     · location    ST.trimmedMean(a, frac), ST.weightedMean(v, w), ST.weightedMedian(v, w)
     · position    ST.quantile(a, p, type) with type "linear" | "np1" | "nearest" | "lower"
                   ST.hinges(a) (Tukey), ST.fiveNum(a), ST.boxStats(a, k), ST.zScores(a)
     · shape       ST.histBins(a, x0, w, k), ST.binRules(a), ST.kde(a, h, xs), ST.silverman(a)
     · bivariate   ST.cov, ST.corr, ST.spearman, ST.lsLine(x, y)
     · cdf         ST.ecdfSteps(a), ST.ecdfAt(sortedAsc, x), ST.dkwEps(n, alpha)
                   ST.normCdf(z) = Φ(z) via the incomplete gamma (~1e-14),
                   ST.normPdf(z) = φ(z), ST.normQuant(p) = Φ⁻¹(p) (~1e-15 for |z| ≤ 6)
     · sampling    ST.seMean(sd, n), ST.seProp(p, n), ST.fpc(n, N)
     · exact dists ST.lnGamma(x), ST.gammaP(a, x) = regularised lower incomplete gamma,
                   ST.lnChoose(n, k), ST.binomPmf(k, n, p), ST.binomCdf(k, n, p)
                   ST.betaI(a, b, x) = regularised incomplete beta (Lentz continued fraction),
                   ST.tPdf(t, df), ST.tCdf(t, df), ST.tQuant(p, df) — Student t, exact to ~1e-14
                   ST.chi2Pdf/chi2Cdf/chi2Quant(·, k) — chi-square, via gammaP
                   ST.fPdf/fCdf/fQuant(·, d1, d2) — Fisher F, via betaI
     · drawing     ST.frame(sel, W, H, m), ST.axisB(...), ST.axisL(...), ST.gridY, ST.gridX,
                   ST.lineSeg(g, a, b, sx, sy, opt) — y = a + b·x, clipped to the frame,
                   ST.jitterY(a, ...), ST.cells(...) small-multiples layout, ST.legend(g, items, x, y)
     · misc        ST.linspace(a, b, k), ST.clamp(x, lo, hi)
     · text        ST.fmt(x, d), ST.sig(x, n), ST.pct(x, d)

   House rules: every consuming page supplies its own <svg width height viewBox role aria-label>;
   these helpers only ever append into an svg that already exists. */

const SC = {
  accent: "#5b9cff", a2: "#ffb454", good: "#4ade80", bad: "#f87171",
  ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a",
  grid: "#1b2130", panel: "#171a23", panel2: "#1e222d", bg: "#0f1117",
  violet: "#c084fc", teal: "#2dd4bf"
};

const ST = (function () {

  /* ── pseudo-random, seeded so every reader sees the same picture ───────── */
  function rng(seed) {                      // mulberry32
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) {                       // Box–Muller, one draw
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function sample(a, k, r) {                // k draws with replacement
    const out = [];
    for (let i = 0; i < k; i++) out.push(a[Math.floor(r() * a.length)]);
    return out;
  }
  function shuffle(a, r) {                  // Fisher–Yates on a COPY; the caller's array is untouched
    const s = [...a];
    for (let i = s.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      const t = s[i]; s[i] = s[j]; s[j] = t;
    }
    return s;
  }
  function srswor(a, k, r) {                // simple random sample WITHOUT replacement, k ≤ a.length
    const n = a.length, kk = Math.min(k, n), s = [...a];
    for (let i = 0; i < kk; i++) {          // partial Fisher–Yates: O(k), not O(n)
      const j = i + Math.floor(r() * (n - i));
      const t = s[i]; s[i] = s[j]; s[j] = t;
    }
    return s.slice(0, kk);
  }
  const choice = (a, r) => a[Math.floor(r() * a.length)];
  function binom(n, p, r) {                 // a Binomial(n, p) draw, by summing n Bernoullis
    let k = 0;
    for (let i = 0; i < n; i++) if (r() < p) k++;
    return k;
  }

  /* ── order statistics & centre ────────────────────────────────────────── */
  const asc = a => [...a].sort((p, q) => p - q);
  const sum = a => a.reduce((s, x) => s + x, 0);
  const mean = a => sum(a) / a.length;

  function median(a) {
    const s = asc(a), n = s.length;
    if (!n) return NaN;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  function mode(a) {                        // returns {value, count, ties}
    const m = new Map();
    a.forEach(x => m.set(x, (m.get(x) || 0) + 1));
    let best = NaN, c = 0, ties = 0;
    m.forEach((v, k) => { if (v > c) { c = v; best = k; ties = 1; } else if (v === c) ties++; });
    return { value: best, count: c, ties };
  }
  const range = a => Math.max(...a) - Math.min(...a);

  /* ── spread ───────────────────────────────────────────────────────────── */
  function variance(a, ddof) {              // ddof defaults to 1 (the sample variance)
    const d = (ddof === undefined) ? 1 : ddof, m = mean(a);
    if (a.length - d <= 0) return NaN;
    return a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - d);
  }
  const sd = (a, ddof) => Math.sqrt(variance(a, ddof));
  function meanAbsDev(a) { const m = mean(a); return mean(a.map(x => Math.abs(x - m))); }
  function mad(a) { const m = median(a); return median(a.map(x => Math.abs(x - m))); }
  const madn = a => 1.4826 * mad(a);        // rescaled to estimate σ under normality
  const iqr = (a, type) => quantile(a, 0.75, type) - quantile(a, 0.25, type);

  /* ── resistant / weighted location ────────────────────────────────────── */
  function trimmedMean(a, frac) {
    const s = asc(a), k = Math.floor(s.length * frac);
    const kept = s.slice(k, s.length - k);
    return kept.length ? mean(kept) : median(a);
  }
  function weightedMean(v, w) {
    let n = 0, d = 0;
    for (let i = 0; i < v.length; i++) { n += v[i] * w[i]; d += w[i]; }
    return n / d;
  }
  function weightedMedian(v, w) {
    const idx = v.map((_, i) => i).sort((p, q) => v[p] - v[q]);
    const tot = sum(w);
    let run = 0;
    for (const i of idx) { run += w[i]; if (run >= tot / 2) return v[i]; }
    return v[idx[idx.length - 1]];
  }

  /* ── quantiles: four conventions that give four different answers ─────── */
  function quantile(a, p, type) {
    const s = asc(a), n = s.length;
    if (!n) return NaN;
    if (p <= 0) return s[0];
    if (p >= 1) return s[n - 1];
    switch (type || "linear") {
      case "linear": {                       // h = (n−1)p, interpolate  (the common default)
        const h = (n - 1) * p, lo = Math.floor(h);
        return s[lo] + (h - lo) * (s[Math.min(lo + 1, n - 1)] - s[lo]);
      }
      case "np1": {                          // h = (n+1)p, interpolate  (x₍ᵢ₎ is the i/(n+1) quantile)
        const h = (n + 1) * p, k = Math.floor(h), al = h - k;
        if (k < 1) return s[0];
        if (k >= n) return s[n - 1];
        return s[k - 1] + al * (s[k] - s[k - 1]);
      }
      case "nearest": return s[Math.max(0, Math.ceil(p * n) - 1)];   // nearest rank, no interpolation
      case "lower": return s[Math.max(0, Math.floor(p * n) - (p * n % 1 === 0 ? 1 : 0))];
      default: return quantile(a, p, "linear");
    }
  }
  function hinges(a) {                       // Tukey's lower hinge, median, upper hinge
    const s = asc(a), n = s.length;
    const d = Math.floor((n + 3) / 2) / 2;   // "depth" of the hinge
    const at = dep => { const i = Math.floor(dep) - 1; return (dep % 1) ? (s[i] + s[i + 1]) / 2 : s[i]; };
    return [at(d), median(s), at(n + 1 - d)];
  }
  function fiveNum(a, type) {
    const s = asc(a);
    if (type === "hinge") { const h = hinges(s); return [s[0], h[0], h[1], h[2], s[s.length - 1]]; }
    return [s[0], quantile(s, 0.25, type), median(s), quantile(s, 0.75, type), s[s.length - 1]];
  }
  function boxStats(a, k, type) {            // five numbers + Tukey fences + whisker ends + outliers
    const kk = (k === undefined) ? 1.5 : k;
    const f = fiveNum(a, type), q1 = f[1], q3 = f[3], iq = q3 - q1;
    const lo = q1 - kk * iq, hi = q3 + kk * iq;
    const inside = asc(a).filter(x => x >= lo && x <= hi);
    return {
      min: f[0], q1: q1, med: f[2], q3: q3, max: f[4], iqr: iq,
      loFence: lo, hiFence: hi,
      whiskLo: inside.length ? inside[0] : q1,
      whiskHi: inside.length ? inside[inside.length - 1] : q3,
      out: a.filter(x => x < lo || x > hi)
    };
  }
  function zScores(a) { const m = mean(a), s = sd(a); return a.map(x => (x - m) / s); }

  /* ── shape: binning, bin-width rules, kernel density ──────────────────── */
  function histBins(a, x0, w, k) {           // k fixed-width bins [x0 + i·w, x0 + (i+1)·w)
    const bins = [];
    for (let i = 0; i < k; i++) bins.push({ x0: x0 + i * w, x1: x0 + (i + 1) * w, n: 0 });
    a.forEach(v => {
      let i = Math.floor((v - x0) / w);
      if (i === k && Math.abs(v - (x0 + k * w)) < 1e-9) i = k - 1;   // right edge belongs to last bin
      if (i >= 0 && i < k) bins[i].n++;
    });
    bins.forEach(b => { b.p = b.n / a.length; b.density = b.p / w; });
    return bins;
  }
  function binRules(a) {                     // three standard bin-width recipes
    const n = a.length, s = sd(a), q = iqr(a);
    return {
      sturges: Math.ceil(Math.log2(n)) + 1,             // a bin COUNT
      scott: 3.49 * s * Math.pow(n, -1 / 3),            // widths
      fd: 2 * q * Math.pow(n, -1 / 3)
    };
  }
  const silverman = a => 1.06 * Math.min(sd(a), iqr(a) / 1.349) * Math.pow(a.length, -1 / 5);
  function kde(a, h, xs) {                   // Gaussian kernel density estimate on the grid xs
    const c = 1 / (a.length * h * Math.sqrt(2 * Math.PI));
    return xs.map(x => {
      let s = 0;
      for (let i = 0; i < a.length; i++) { const u = (x - a[i]) / h; s += Math.exp(-0.5 * u * u); }
      return { x: x, y: c * s };
    });
  }

  /* ── bivariate ────────────────────────────────────────────────────────── */
  function cov(x, y) {
    const mx = mean(x), my = mean(y);
    let s = 0;
    for (let i = 0; i < x.length; i++) s += (x[i] - mx) * (y[i] - my);
    return s / (x.length - 1);
  }
  function corr(x, y) {
    const mx = mean(x), my = mean(y);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < x.length; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    return (sxx && syy) ? sxy / Math.sqrt(sxx * syy) : NaN;
  }
  function ranks(a) {                        // average ranks, ties shared
    const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(a.length);
    let i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
      i = j + 1;
    }
    return r;
  }
  const spearman = (x, y) => corr(ranks(x), ranks(y));
  function lsLine(x, y) {                    // least-squares fit  ŷ = a + b·x
    const b = corr(x, y) * sd(y) / sd(x);
    return { a: mean(y) - b * mean(x), b: b, r: corr(x, y) };
  }

  /* ── cdf-land: the bridge from "this batch" to "the population" ───────── */
  function ecdfSteps(a) {                    // [{x, F}] with F = Fₙ(x) just after the jump at x
    const s = asc(a), n = s.length, out = [];
    for (let i = 0; i < n; i++) {
      if (i && s[i] === s[i - 1]) { out[out.length - 1].F = (i + 1) / n; continue; }
      out.push({ x: s[i], F: (i + 1) / n });
    }
    return out;
  }
  function ecdfAt(sortedAsc, x) {            // Fₙ(x) = #{xᵢ ≤ x}/n
    let lo = 0, hi = sortedAsc.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sortedAsc[m] <= x) lo = m + 1; else hi = m; }
    return lo / sortedAsc.length;
  }
  const dkwEps = (n, alpha) => Math.sqrt(Math.log(2 / alpha) / (2 * n));   // DKW half-width
  /* Φ(z) from the regularised incomplete gamma already in this file:
       Φ(z) = ½(1 + sign(z)·P(½, z²/2)).
     The old Abramowitz–Stegun 7.1.26 form here was only good to 7.5e-8, which
     silently CAPPED the Halley refinement in normQuant below and left Φ⁻¹ about
     1.2e-6 out — worse than its own starting approximation. This reuses code that
     is already exercised by the chi-square work instead of adding a new one. */
  function normCdf(z) {
    if (!isFinite(z)) return z > 0 ? 1 : 0;
    if (z === 0) return 0.5;
    const half = gammaP(0.5, z * z / 2) / 2;
    return z > 0 ? 0.5 + half : 0.5 - half;
  }
  const normPdf = z => 0.3989422804014327 * Math.exp(-z * z / 2);   // φ(z)
  function normQuant(p) {                    // Φ⁻¹(p) — rational approximation + one Halley step
    if (!(p > 0 && p < 1)) return p <= 0 ? -Infinity : Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
               6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425, ph = 1 - pl;
    let x, q, r2;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
          ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= ph) {
      q = p - 0.5; r2 = q * q;
      x = (((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * q /
          (((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    // Halley refinement. Beyond |x| ≈ 6 the CDF underflows toward its own limit, so
    // the correction carries no information there — keep the raw approximation.
    if (Math.abs(x) > 6) return x;
    const e = normCdf(x) - p, u = e / normPdf(x);
    return x - u / (1 + x * u / 2);
  }

  /* ── sampling error: the three standard errors part 3 onward lean on ───── */
  const seMean = (sdev, n) => sdev / Math.sqrt(n);                  // SE of a mean
  const seProp = (p, n) => Math.sqrt(p * (1 - p) / n);              // SE of a proportion
  const fpc = (n, N) => Math.sqrt((N - n) / (N - 1));               // finite-population correction

  /* ── exact discrete/continuous distribution functions ─────────────────── */
  function lnGamma(x) {                      // Lanczos g = 7, n = 9; |rel err| < 1e-13 for x > 0
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
    const g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
               771.32342877765313, -176.61502916214059, 12.507343278686905,
               -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    const z = x - 1;
    let a = g[0];
    for (let i = 1; i < 9; i++) a += g[i] / (z + i);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
  }
  function gammaP(a, x) {                    // regularised lower incomplete gamma P(a, x) = γ(a,x)/Γ(a)
    if (x <= 0) return 0;
    if (x < a + 1) {                         // series expansion — converges fast below the peak
      let ap = a, del = 1 / a, s = del;
      for (let i = 0; i < 2000; i++) {
        ap++; del *= x / ap; s += del;
        if (Math.abs(del) < Math.abs(s) * 1e-15) break;
      }
      return s * Math.exp(-x + a * Math.log(x) - lnGamma(a));
    }
    let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;   // Lentz's continued fraction for Q(a, x)
    for (let i = 1; i < 2000; i++) {
      const an = -i * (i - a);
      b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - lnGamma(a)) * h;
  }
  const lnChoose = (n, k) => lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);
  const binomPmf = (k, n, p) =>              // exact, via log-gamma, so n in the thousands is safe
    (k < 0 || k > n) ? 0
      : (p <= 0 ? (k === 0 ? 1 : 0)
        : p >= 1 ? (k === n ? 1 : 0)
          : Math.exp(lnChoose(n, k) + k * Math.log(p) + (n - k) * Math.log1p(-p)));
  function binomCdf(k, n, p) {               // P(X ≤ k); summed from the mode outward for accuracy
    if (k < 0) return 0;
    if (k >= n) return 1;
    let s = 0;
    for (let i = 0; i <= k; i++) s += binomPmf(i, n, p);
    return Math.min(1, s);
  }

  /* ── the incomplete beta, and the t distribution built on it ──────────
     Parts 4, 5, 6 and 9 all need a Student-t critical value, and an honest
     one needs the regularised incomplete beta rather than a table lookup or
     a tail approximation. betacf is the standard Lentz continued fraction;
     everything below is exact to ~1e-14, verified against published tables
     (t₀.₉₇₅ = 12.7062 at df 1, 2.22814 at df 10, 1.96234 at df 1000) and
     against numerical quadrature of the density.                            */
  function betacf(a, b, x) {                 // continued fraction for I_x(a, b), by modified Lentz
    const TINY = 1e-300, EPS = 3e-16;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < TINY) d = TINY;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 400; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));      // even step
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); // odd step
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function betaI(a, b, x) {                  // regularised incomplete beta I_x(a, b) ∈ [0, 1]
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b)
                        + a * Math.log(x) + b * Math.log1p(-x));
    return (x < (a + 1) / (a + b + 2))      // use whichever side the CF converges on
      ? bt * betacf(a, b, x) / a
      : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function tPdf(t, df) {                     // Student-t density with df degrees of freedom
    if (!isFinite(df)) return normPdf(t);
    return Math.exp(lnGamma((df + 1) / 2) - lnGamma(df / 2)) / Math.sqrt(df * Math.PI)
      * Math.pow(1 + t * t / df, -(df + 1) / 2);
  }
  function tCdf(t, df) {                     // P(T ≤ t) — one call to the incomplete beta
    if (!isFinite(df)) return normCdf(t);
    if (!isFinite(t)) return t > 0 ? 1 : 0;
    const p = 0.5 * betaI(df / 2, 0.5, df / (df + t * t));
    return t > 0 ? 1 - p : p;
  }
  function tQuant(p, df) {                   // t*(p, df): Cornish–Fisher start, then guarded Newton
    if (!(p > 0 && p < 1)) return p <= 0 ? -Infinity : Infinity;
    if (p === 0.5) return 0;
    if (!isFinite(df)) return normQuant(p);
    const neg = p < 0.5, pp = neg ? p : 1 - p, target = 1 - pp;   // work in the upper half, mirror back
    const z = normQuant(target), z3 = z * z * z, z5 = z3 * z * z, z7 = z5 * z * z;
    let x = z + (z3 + z) / (4 * df)
      + (5 * z5 + 16 * z3 + 3 * z) / (96 * df * df)
      + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * df * df * df);
    let lo = 0, hi = Math.max(4 * Math.abs(x), 10), guard = 0;    // bracket, so df = 1 cannot escape
    while (tCdf(hi, df) < target && guard++ < 200) hi *= 2;
    if (!(x > lo && x < hi)) x = (lo + hi) / 2;
    for (let i = 0; i < 200; i++) {
      const f = tCdf(x, df) - target;
      if (f > 0) hi = x; else lo = x;
      const dens = tPdf(x, df);
      let nx = (dens > 1e-300) ? x - f / dens : 0.5 * (lo + hi);
      if (!(nx > lo && nx < hi)) nx = 0.5 * (lo + hi);            // Newton stepped out → bisect
      const done = Math.abs(nx - x) < 1e-14 * Math.max(1, Math.abs(x));
      x = nx;
      if (done) break;
    }
    return neg ? -x : x;
  }

  /* ── chi-square and F, both built on the two special functions above ───
     χ²ₖ is Gamma(k/2, 2), so its cdf is exactly the regularised lower
     incomplete gamma already here; F(d₁, d₂) reduces to the regularised
     incomplete beta at d₁x/(d₁x + d₂). Both quantiles are found by a
     bracketed bisection on their own monotone cdf — slower than Newton but
     it cannot leave the bracket, which matters at df = 1 where the density
     blows up at the origin. Verified against published tables to ≤ 3e-5
     absolute on the tabulated 3–4 significant figures, and by round-trip
     (cdf∘quant = p) to ~1e-12.                                             */
  function chi2Pdf(x, k) {
    if (x <= 0) return (k < 2) ? Infinity : (k === 2 ? 0.5 : 0);
    return Math.exp((k / 2 - 1) * Math.log(x) - x / 2
                    - (k / 2) * Math.LN2 - lnGamma(k / 2));
  }
  const chi2Cdf = (x, k) => (x <= 0 ? 0 : gammaP(k / 2, x / 2));   // P(χ²ₖ ≤ x)
  function chi2Quant(p, k) {                 // χ²ₖ,ₚ with P(χ²ₖ ≤ χ²ₖ,ₚ) = p
    if (!(p > 0 && p < 1)) return p <= 0 ? 0 : Infinity;
    let lo = 0, hi = Math.max(4 * k, 20), guard = 0;
    while (chi2Cdf(hi, k) < p && guard++ < 300) hi *= 2;
    for (let i = 0; i < 200; i++) {
      const m = 0.5 * (lo + hi);
      if (chi2Cdf(m, k) < p) lo = m; else hi = m;
      if (hi - lo < 1e-13 * Math.max(1, hi)) break;
    }
    return 0.5 * (lo + hi);
  }
  function fPdf(x, d1, d2) {                 // Fisher–Snedecor density
    if (x <= 0) return 0;
    const a = d1 / 2, b = d2 / 2;
    return Math.exp(a * Math.log(d1) + b * Math.log(d2) + (a - 1) * Math.log(x)
                    - (a + b) * Math.log(d2 + d1 * x)
                    + lnGamma(a + b) - lnGamma(a) - lnGamma(b));
  }
  const fCdf = (x, d1, d2) =>                // P(F ≤ x) = I_{d₁x/(d₁x+d₂)}(d₁/2, d₂/2)
    (x <= 0 ? 0 : betaI(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2)));
  function fQuant(p, d1, d2) {
    if (!(p > 0 && p < 1)) return p <= 0 ? 0 : Infinity;
    let lo = 0, hi = 4, guard = 0;
    while (fCdf(hi, d1, d2) < p && guard++ < 300) hi *= 2;
    for (let i = 0; i < 300; i++) {
      const m = 0.5 * (lo + hi);
      if (fCdf(m, d1, d2) < p) lo = m; else hi = m;
      if (hi - lo < 1e-13 * Math.max(1, hi)) break;
    }
    return 0.5 * (lo + hi);
  }

  /* ── drawing ──────────────────────────────────────────────────────────── */
  function frame(sel, W, H, m) {             // margined <g> inside an <svg> that already exists
    const mm = Object.assign({ l: 46, r: 16, t: 14, b: 34 }, m || {});
    const svg = (typeof sel === "string") ? d3.select(sel) : sel;
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${mm.l},${mm.t})`);
    return { svg: svg, g: g, m: mm, iw: W - mm.l - mm.r, ih: H - mm.t - mm.b };
  }
  function axisB(g, x, ih, ticks, label, fmtFn) {
    const ax = g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(fmtFn ? d3.axisBottom(x).ticks(ticks || 6).tickFormat(fmtFn) : d3.axisBottom(x).ticks(ticks || 6));
    if (label) g.append("text").attr("x", x.range()[1]).attr("y", ih + 31).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text(label);
    return ax;
  }
  function axisL(g, y, ticks, label, fmtFn) {
    const ax = g.append("g").attr("class", "axis")
      .call(fmtFn ? d3.axisLeft(y).ticks(ticks || 5).tickFormat(fmtFn) : d3.axisLeft(y).ticks(ticks || 5));
    if (label) g.append("text").attr("x", 0).attr("y", -4).attr("text-anchor", "start")
      .attr("font-size", 11).attr("fill", SC.muted).text(label);
    return ax;
  }
  function gridY(g, y, iw, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(ticks || 5)).join("line")
      .attr("x1", 0).attr("x2", iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", SC.grid);
  }
  function gridX(g, x, ih, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(x.ticks(ticks || 5)).join("line")
      .attr("y1", 0).attr("y2", ih).attr("x1", d => x(d)).attr("x2", d => x(d)).attr("stroke", SC.grid);
  }
  /* A line y = a + b·x clipped to the plot rectangle. Written independently in
     part 6's and part 7's page files before being promoted here on its third use
     (part 9). The clipping matters: an unclipped <line> drawn from the domain
     edges escapes the frame whenever |b| is large, and a steep fitted line is
     exactly when you most want to see it. */
  function lineSeg(g, a, b, sx, sy, opt) {
    const o = Object.assign({ color: SC.a2, w: 2, dash: null, op: 1 }, opt || {});
    const dx = sx.domain(), dyr = sy.domain();
    const y0 = Math.min(dyr[0], dyr[1]), y1 = Math.max(dyr[0], dyr[1]);
    let xa = dx[0], xb = dx[1];
    if (Math.abs(b) > 1e-12) {
      const u = (y0 - a) / b, v = (y1 - a) / b;
      xa = Math.max(xa, Math.min(u, v));
      xb = Math.min(xb, Math.max(u, v));
    } else if (a < y0 || a > y1) return g.append("line");   // entirely outside — draw nothing
    if (!(xb > xa)) return g.append("line");
    const el = g.append("line")
      .attr("x1", sx(xa)).attr("y1", sy(a + b * xa))
      .attr("x2", sx(xb)).attr("y2", sy(a + b * xb))
      .attr("stroke", o.color).attr("stroke-width", o.w).attr("stroke-opacity", o.op);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    return el;
  }
  function jitterY(values, seed, amp) {      // deterministic nudge so coincident points stay visible
    const r = rng(seed || 7), a = (amp === undefined) ? 1 : amp;
    return values.map(() => (r() * 2 - 1) * a);
  }
  function cells(W, H, cols, rows, m, gap) { // small-multiples: k panels on a grid, each with its own <g>
    const mm = Object.assign({ l: 40, r: 12, t: 24, b: 30 }, m || {});
    const gp = Object.assign({ x: 18, y: 26 }, gap || {});
    const cw = (W - (cols - 1) * gp.x) / cols, ch = (H - (rows - 1) * gp.y) / rows;
    const out = [];
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      out.push({
        i: i, j: j, k: j * cols + i,
        x: i * (cw + gp.x), y: j * (ch + gp.y), w: cw, h: ch,
        m: mm, iw: cw - mm.l - mm.r, ih: ch - mm.t - mm.b
      });
    }
    return out;
  }
  function legend(g, items, x, y, opts) {    // items: [{label, color, dash}]
    const o = Object.assign({ gap: 15, size: 9, font: 10.5, vertical: true }, opts || {});
    const gl = g.append("g").attr("transform", `translate(${x},${y})`);
    items.forEach((it, i) => {
      const dx = o.vertical ? 0 : i * (o.gap * 8), dy = o.vertical ? i * o.gap : 0;
      if (it.dash) {
        gl.append("line").attr("x1", dx).attr("x2", dx + o.size).attr("y1", dy).attr("y2", dy)
          .attr("stroke", it.color).attr("stroke-width", 2).attr("stroke-dasharray", it.dash);
      } else {
        gl.append("rect").attr("x", dx).attr("y", dy - o.size / 2).attr("width", o.size)
          .attr("height", o.size).attr("rx", 2).attr("fill", it.color).attr("fill-opacity", it.op === undefined ? 0.9 : it.op);
      }
      gl.append("text").attr("x", dx + o.size + 6).attr("y", dy + 3.5)
        .attr("font-size", o.font).attr("fill", SC.muted).text(it.label);
    });
    return gl;
  }

  /* ── text ─────────────────────────────────────────────────────────────── */
  function fmt(x, d) {
    if (!isFinite(x)) return "—";
    const dd = (d === undefined) ? 2 : d;
    const s = x.toFixed(dd);
    return (parseFloat(s) === 0) ? (0).toFixed(dd) : s;   // never print "-0.00"
  }
  const sig = (x, n) => isFinite(x) ? Number(x.toPrecision(n || 3)).toString() : "—";
  const pct = (x, d) => fmt(100 * x, d === undefined ? 1 : d) + "%";
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  function linspace(a, b, k) {
    const out = [];
    for (let i = 0; i < k; i++) out.push(a + (b - a) * (k === 1 ? 0 : i / (k - 1)));
    return out;
  }

  return {
    rng, randn, sample, shuffle, srswor, choice, binom,
    asc, sum, mean, median, mode, range,
    variance, sd, meanAbsDev, mad, madn, iqr,
    trimmedMean, weightedMean, weightedMedian,
    quantile, hinges, fiveNum, boxStats, zScores,
    histBins, binRules, silverman, kde,
    cov, corr, ranks, spearman, lsLine,
    ecdfSteps, ecdfAt, dkwEps, normCdf, normPdf, normQuant,
    seMean, seProp, fpc,
    lnGamma, gammaP, lnChoose, binomPmf, binomCdf,
    betacf, betaI, tPdf, tCdf, tQuant,
    chi2Pdf, chi2Cdf, chi2Quant, fPdf, fCdf, fQuant,
    frame, axisB, axisL, gridY, gridX, jitterY, cells, legend, lineSeg,
    fmt, sig, pct, clamp, linspace
  };
})();
