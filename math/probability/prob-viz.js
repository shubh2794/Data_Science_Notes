/* prob-viz.js — shared distribution functions for the Probability series (math/probability/*).
   Loaded after ../statistics/stats-viz.js (which provides SC and ST) and before each part's
   own "<part>.viz.js". Nothing here draws anything. It adds ONLY what ST lacks; anything ST
   already has (normal, binomial pmf/cdf, gamma/beta/t/χ²/F, lnGamma, lnChoose, gammaP,
   betaI, rng, binom draw) is called from ST, never re-implemented here.

   Provides one global, PV:
     · Poisson              PV.poisPmf(k, mu), PV.poisCdf(k, mu), PV.poisDraw(mu, r)
     · geometric            PV.geomPmf(k, p), PV.geomCdf(k, p), PV.geomDraw(p, r)
                            (k = number of TRIALS up to and including the first success, k ≥ 1)
     · negative binomial    PV.negBinPmf(x, r, p)      x = trials needed for the r-th success, x ≥ r
                            PV.negBinFailPmf(y, r, p)  y = failures before the r-th success, y ≥ 0;
                                                       r may be any real > 0 (the overdispersed-count form)
     · hypergeometric       PV.hyperPmf(k, N, K, n)    k successes in n draws WITHOUT replacement from
                                                       N items of which K are successes
     · categorical          PV.catDraw(probs, r), PV.softmax(logits, T), PV.multinomLogPmf(counts, probs)
     · comparison           PV.tv(p, q) = ½ Σ |pₖ − qₖ| over aligned arrays (total-variation distance)
     · helpers              PV.lnFact(n), PV.support(fn, lo, eps, cap) — tabulate a pmf until its
                            remaining mass is below eps

   Every function works in log space through ST.lnGamma, so n, N and μ in the thousands do not
   overflow. All functions return 0 (not NaN) outside the support.

   VERIFICATION (run 2026-10-01 under node with stats-viz.js + this file loaded; script kept in the
   build scratchpad). Each line is "check — result":
     poisPmf
       · Σₖ poisPmf(k, μ) = 1 for μ ∈ {0.1, 1, 4.5, 10, 50, 200} — |1 − Σ| < 2e-14
       · mean Σ k·p = μ (|err| < 4e-12) and variance Σ k²p − μ² = μ (|err| < 1e-9, worst at μ = 200)
       · μ = 4.5, k = 0…9 vs the direct formula e^(−μ)μᵏ/k! with exact factorials — identical to 8 d.p.:
         0.01110900 0.04999048 0.11247859 0.16871788 0.18980762 0.17082686 0.12812014 0.08236295
         0.04632916 0.02316458. A published 5-d.p. comparison table agrees at k = 0…6 and 8 but prints
         0.08237 at k = 7 and 0.02317 at k = 9; the correct roundings are 0.08236 and 0.02316.
       · μ = 1, k = 0: e⁻¹ — relative error 9e-16
     poisCdf
       · identity P(X ≤ k) = Q(k + 1, μ) = 1 − ST.gammaP(k + 1, μ), for μ ∈ {0.5, 3, 10, 20, 100},
         k from 0 to 3μ — max |diff| = 1.3e-13
       · published table values: μ = 10, F(8) = 0.333; μ = 20, F(16) = 0.221; μ = 6, F(5) = 0.446 — match
     geomPmf / geomCdf
       · Σₖ geomPmf = 1, mean 1/p, variance (1 − p)/p² for p ∈ {0.05, 0.2, 0.5, 0.9} — |err| < 3e-13
       · geomCdf(k, p) = 1 − (1 − p)ᵏ vs cumulative sum of geomPmf — |diff| < 2e-15
     negBinPmf / negBinFailPmf
       · r = 1 reduces to geomPmf (k = 1…60, p ∈ {0.1, 0.3}) — |diff| < 3e-16
       · Σ = 1, mean r/p, variance r(1 − p)/p² for (r, p) ∈ {(3, 0.2), (12, 0.95), (5, 0.5)} — |err| < 3e-13
       · binomial duality P(X > n) = P(Bin(n, p) < r): r = 12, p = 0.95, n = 15 — both 0.005467
       · fail form with real r: mean r(1 − p)/p, variance r(1 − p)/p² for r = 2.5, p = 0.4 — |err| < 6e-15
       · Monte Carlo: 200,000 draws as sums of r geometric draws (r = 3, p = 0.25, seed 12345), pmf vs
         histogram — max |diff| 0.0012 (largest cell's sampling sd ≈ 0.0007)
     hyperPmf
       · Σₖ = 1 (Vandermonde) for (N, K, n) ∈ {(49, 6, 6), (12, 5, 4), (1000, 300, 50), (20, 7, 20)} — < 4e-13
       · mean nK/N and variance n(K/N)(1 − K/N)(N − n)/(N − 1) — |err| < 1e-10
       · lottery 6 from 49, exactly 3 matches: 246,820/13,983,816 = 0.0176504 — exact
       · Monte Carlo with ST.srswor, 100,000 samples of n = 10 from N = 30, K = 12 (seed 777)
         — max |diff| 0.0021 (largest cell's sampling sd ≈ 0.0015)
     multinomLogPmf
       · two categories reduces to ST.binomPmf (n = 20, p = 0.3, all k) — |diff| < 2e-15
       · fair die, counts (1,1,1,1,1,1): 6!/6⁶ = 0.0154321 — exact                                  */

const PV = (function () {
  const lnG = x => ST.lnGamma(x);
  const lnFact = n => lnG(n + 1);

  /* ── Poisson(μ): P(X = k) = e^(−μ) μᵏ / k! ─────────────────────────────── */
  function poisPmf(k, mu) {
    if (!(k >= 0) || k !== Math.floor(k)) return 0;
    if (mu <= 0) return k === 0 ? 1 : 0;
    return Math.exp(-mu + k * Math.log(mu) - lnFact(k));
  }
  function poisCdf(k, mu) {               // direct sum; checked against 1 − ST.gammaP(k + 1, μ)
    if (k < 0) return 0;
    const kk = Math.floor(k);
    let s = 0;
    for (let i = 0; i <= kk; i++) s += poisPmf(i, mu);
    return Math.min(1, s);
  }
  function poisDraw(mu, r) {              // inverse transform via the ratio recursion p(k+1) = p(k)·μ/(k+1)
    const u = r();
    let k = 0, p = Math.exp(-mu), c = p;
    if (p === 0) {                        // μ too large for e^(−μ): fall back to a normal start
      k = Math.max(0, Math.round(mu + Math.sqrt(mu) * ST.randn(r)));
      return k;
    }
    while (u > c && k < 100000) { k++; p *= mu / k; c += p; }
    return k;
  }

  /* ── geometric(p), trials convention: P(X = k) = (1 − p)ᵏ⁻¹ p, k ≥ 1 ───── */
  function geomPmf(k, p) {
    if (!(k >= 1) || k !== Math.floor(k)) return 0;
    if (p >= 1) return k === 1 ? 1 : 0;
    return Math.exp((k - 1) * Math.log1p(-p) + Math.log(p));
  }
  const geomCdf = (k, p) => (k < 1 ? 0 : 1 - Math.exp(Math.floor(k) * Math.log1p(-p)));
  function geomDraw(p, r) {               // inverse transform of the closed-form tail
    if (p >= 1) return 1;
    let u = r(); while (u === 0) u = r();
    return Math.max(1, Math.ceil(Math.log(u) / Math.log1p(-p)));
  }

  /* ── negative binomial ────────────────────────────────────────────────────
     trials form:   P(X = x) = C(x − 1, r − 1) pʳ (1 − p)ˣ⁻ʳ,  x = r, r + 1, …
     failures form: P(Y = y) = Γ(y + r)/(Γ(r) y!) pʳ (1 − p)ʸ, y = 0, 1, …  (Y = X − r; r real > 0) */
  function negBinFailPmf(y, r, p) {
    if (!(y >= 0) || y !== Math.floor(y) || !(r > 0)) return 0;
    if (p >= 1) return y === 0 ? 1 : 0;
    return Math.exp(lnG(y + r) - lnG(r) - lnFact(y) + r * Math.log(p) + y * Math.log1p(-p));
  }
  const negBinPmf = (x, r, p) => negBinFailPmf(x - r, r, p);

  /* ── hypergeometric: P(X = k) = C(K, k) C(N − K, n − k) / C(N, n) ───────── */
  function hyperPmf(k, N, K, n) {
    if (k !== Math.floor(k)) return 0;
    if (k < Math.max(0, n - (N - K)) || k > Math.min(n, K)) return 0;
    return Math.exp(ST.lnChoose(K, k) + ST.lnChoose(N - K, n - k) - ST.lnChoose(N, n));
  }

  /* ── categorical / multinomial ───────────────────────────────────────────── */
  function catDraw(probs, r) {
    const u = r();
    let c = 0;
    for (let i = 0; i < probs.length; i++) { c += probs[i]; if (u < c) return i; }
    return probs.length - 1;              // guards the last-bin rounding gap
  }
  function softmax(logits, T) {
    const t = (T === undefined) ? 1 : T;
    const z = logits.map(v => v / t), m = Math.max(...z);
    const e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }
  function multinomLogPmf(counts, probs) {
    const n = counts.reduce((a, b) => a + b, 0);
    let l = lnFact(n);
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] === 0) continue;
      if (probs[i] <= 0) return -Infinity;
      l += counts[i] * Math.log(probs[i]) - lnFact(counts[i]);
    }
    return l;
  }

  /* ── comparison and tabulation ──────────────────────────────────────────── */
  function tv(p, q) {
    const n = Math.max(p.length, q.length);
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.abs((p[i] || 0) - (q[i] || 0));
    return s / 2;
  }
  function support(fn, lo, eps, cap) {    // [{k, p}] from k = lo until the tail mass < eps
    const out = [], e = eps || 1e-9, c = cap || 5000;
    let acc = 0;
    for (let k = lo; k < lo + c; k++) {
      const p = fn(k);
      out.push({ k: k, p: p });
      acc += p;
      if (acc > 1 - e && k > lo) break;
    }
    return out;
  }

  return {
    lnFact,
    poisPmf, poisCdf, poisDraw,
    geomPmf, geomCdf, geomDraw,
    negBinPmf, negBinFailPmf,
    hyperPmf,
    catDraw, softmax, multinomLogPmf,
    tv, support
  };
})();

/* ═══ Continuous families — appended for Continuous Distributions (part 3) ═══════════════════
   Adds to PV only what neither ST nor the block above has. The normal pdf/cdf/quantile, lnGamma,
   the regularised incomplete gamma (ST.gammaP), the regularised incomplete beta (ST.betaI) and the
   χ², t, F functions all stay in ST; the cdfs below are thin calls into those two special functions.
   Parametrisation: exponential and gamma use the RATE λ (mean 1/λ and α/λ); Pareto uses shape α and
   minimum xm; lognormal uses the mean μ and sd σ of log X.

     · exponential   PV.expPdf(x, λ), PV.expCdf(x, λ), PV.expQuant(p, λ), PV.expDraw(λ, r)
     · gamma         PV.gammaPdf(x, α, λ), PV.gammaCdf(x, α, λ) = ST.gammaP(α, λx),
                     PV.gammaQuant(p, α, λ), PV.gammaDraw(α, λ, r)   (Marsaglia–Tsang, α < 1 boosted)
     · beta          PV.betaPdf(x, a, b), PV.betaCdf(x, a, b) = ST.betaI(a, b, x),
                     PV.betaQuant(p, a, b), PV.betaDraw(a, b, r) = G₁/(G₁ + G₂)
     · Pareto        PV.paretoPdf(x, α, xm), PV.paretoCdf, PV.paretoQuant, PV.paretoDraw(α, xm, r)
     · lognormal     PV.lognormPdf(x, μ, σ), PV.lognormCdf, PV.lognormQuant, PV.lognormDraw(μ, σ, r)
     · numerics      PV.invert(cdf, p, lo, hi) — bracketed bisection for any continuous cdf
                     PV.simpson(f, a, b, n)    — composite Simpson rule, n even

   VERIFICATION (run 2026-10-01 under node with stats-viz.js + this file; script verify-cd.js kept in
   the build scratchpad). Each line is "check — result":
     normalisation  ∫ pdf = 1 by Simpson (n = 200,000) for Exp(0.25), Gamma(3, 0.2), Gamma(9, 0.75),
                    Beta(2, 5), Beta(23.18, 20.35), Lognormal(0, 1) — |1 − ∫| < 1e-6. The singular
                    densities need a substitution first (Simpson evaluates the endpoint): Gamma(0.5, 1)
                    with x = u², Beta(0.5, 0.5) with x = sin²t, Pareto with x = xm/u — |1 − ∫| < 6e-12
     moments        Simpson mean and variance against α/λ, α/λ²; a/(a+b), ab/((a+b)²(a+b+1));
                    αxm/(α−1) (Pareto(1.5): 2.999995 with x = 1/v², the mean integrand being singular),
                    αxm²/((α−1)²(α−2)); e^(μ+σ²/2), (e^(σ²)−1)e^(2μ+σ²) — rel. err < 2e-6
     identities     gammaCdf(x, 1, λ) = expCdf(x, λ) — |diff| < 1e-15 on 200 points
                    gammaCdf(x, k/2, ½) = ST.chi2Cdf(x, k), k = 1…10 — identical
                    Gamma–Poisson: gammaCdf(t, r, 1) = 1 − PV.poisCdf(r − 1, t), r = 1…12, t to 40
                    — |diff| < 7e-15
                    betaCdf(x, 1, 1) = x; betaCdf(x, 2.5, 1) = x^2.5; betaCdf(x, a, b) = 1 − betaCdf(1−x, b, a)
                    — |diff| < 9e-16
                    lognormCdf(e^(μ+σz), μ, σ) = Φ(z); paretoCdf(xm·e^t, α, xm) = expCdf(t, α) — |diff|
                    < 3e-16 (the log of a Pareto variable is exponential)
     quantiles      round trip cdf(quant(p)) = p for p ∈ {0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999}
                    for exp, gamma (0.5 and 9), beta (0.5,0.5 and 23.18,20.35), Pareto, lognormal
                    — |err| < 1.2e-12
     published      Gamma(3, 1/5): P(T < 12) = 0.430291 (1 − e^(−2.4)(1 + 2.4 + 2.88) = 0.430291)
                    Gamma(9, 0.75): P(8 < T < 10) = 0.18527 (Poisson-table route: 0.338 − 0.153 = 0.185)
                    χ²₁: P(≤ 3.841459) = 0.950000; Beta(23.18, 20.35) mean 0.53251
                    Exp(0.25): P(X > 5) = e^(−1.25) = 0.286505; Exp mean 4: q₀.₁ = 0.421442
     Monte Carlo    200,000 draws each (one stream, seed 20261001): expDraw, gammaDraw (α = 0.5, 3, 9),
                    betaDraw (0.5, 0.5), (2, 5), paretoDraw (α = 3), lognormDraw (0, 1) — sample mean
                    and the empirical cdf at the 10/50/90% quantiles within 3 standard errors of the
                    exact values (largest |z| = 2.79); Kolmogorov distance 0.0011–0.0035 (1% critical
                    value 0.0036). The 0.0035 was Gamma(3); five further seeds gave 0.0019–0.0026.  */
Object.assign(PV, (function () {
  const lnG = x => ST.lnGamma(x);

  /* ── generic numerics ─────────────────────────────────────────────────── */
  function invert(cdf, p, lo, hi) {        // smallest x with cdf(x) ≥ p, by bracketed bisection
    let a = lo, b = hi, guard = 0;
    while (cdf(b) < p && guard++ < 400) { const w = b - a; a = b; b = b + 2 * w; }
    for (let i = 0; i < 300; i++) {
      const m = 0.5 * (a + b);
      if (cdf(m) < p) a = m; else b = m;
      if (b - a < 1e-14 * Math.max(1, Math.abs(b))) break;
    }
    return 0.5 * (a + b);
  }
  function simpson(f, a, b, n) {
    const N = (n || 2000) + ((n || 2000) % 2), h = (b - a) / N;
    let s = f(a) + f(b);
    for (let i = 1; i < N; i++) s += (i % 2 ? 4 : 2) * f(a + i * h);
    return s * h / 3;
  }
  const unit = r => { let u = r(); while (u === 0) u = r(); return u; };   // u ∈ (0, 1)

  /* ── exponential(λ): f = λe^(−λx), F = 1 − e^(−λx) ─────────────────────── */
  const expPdf = (x, lam) => (x < 0 ? 0 : lam * Math.exp(-lam * x));
  const expCdf = (x, lam) => (x <= 0 ? 0 : -Math.expm1(-lam * x));
  const expQuant = (p, lam) => -Math.log1p(-p) / lam;
  const expDraw = (lam, r) => -Math.log(unit(r)) / lam;          // −ln U ~ Exp(1), U ~ U(0, 1)

  /* ── gamma(α, λ), rate form: f = λ^α x^(α−1) e^(−λx) / Γ(α) ────────────── */
  function gammaPdf(x, a, lam) {
    if (x < 0) return 0;
    if (x === 0) return a < 1 ? Infinity : (a === 1 ? lam : 0);
    return Math.exp(a * Math.log(lam) + (a - 1) * Math.log(x) - lam * x - lnG(a));
  }
  const gammaCdf = (x, a, lam) => (x <= 0 ? 0 : ST.gammaP(a, lam * x));
  const gammaQuant = (p, a, lam) => invert(x => gammaCdf(x, a, lam), p, 0, Math.max(1, 2 * a / lam));
  function gammaStd(a, r) {                // Gamma(a, 1) — Marsaglia–Tsang
    if (a < 1) return gammaStd(a + 1, r) * Math.pow(unit(r), 1 / a);
    const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = ST.randn(r); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = unit(r);
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  const gammaDraw = (a, lam, r) => gammaStd(a, r) / lam;

  /* ── beta(a, b): f = x^(a−1)(1 − x)^(b−1) / B(a, b) on (0, 1) ──────────── */
  function betaPdf(x, a, b) {
    if (x < 0 || x > 1) return 0;
    if (x === 0) return a < 1 ? Infinity : (a === 1 ? b : 0);
    if (x === 1) return b < 1 ? Infinity : (b === 1 ? a : 0);
    return Math.exp(lnG(a + b) - lnG(a) - lnG(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x));
  }
  const betaCdf = (x, a, b) => ST.betaI(a, b, x);
  const betaQuant = (p, a, b) => invert(x => betaCdf(x, a, b), p, 0, 1);
  const betaDraw = (a, b, r) => { const g1 = gammaStd(a, r), g2 = gammaStd(b, r); return g1 / (g1 + g2); };

  /* ── Pareto(α, xm): P(X > x) = (xm/x)^α for x ≥ xm ─────────────────────── */
  const paretoPdf = (x, al, xm) => (x < xm ? 0 : al * Math.pow(xm, al) / Math.pow(x, al + 1));
  const paretoCdf = (x, al, xm) => (x <= xm ? 0 : 1 - Math.pow(xm / x, al));
  const paretoQuant = (p, al, xm) => xm * Math.pow(1 - p, -1 / al);
  const paretoDraw = (al, xm, r) => xm * Math.pow(unit(r), -1 / al);   // inverse transform, U ↔ 1 − U

  /* ── lognormal(μ, σ): ln X ~ N(μ, σ²) ──────────────────────────────────── */
  const lognormPdf = (x, mu, s) => (x <= 0 ? 0 : ST.normPdf((Math.log(x) - mu) / s) / (x * s));
  const lognormCdf = (x, mu, s) => (x <= 0 ? 0 : ST.normCdf((Math.log(x) - mu) / s));
  const lognormQuant = (p, mu, s) => Math.exp(mu + s * ST.normQuant(p));
  const lognormDraw = (mu, s, r) => Math.exp(mu + s * ST.randn(r));

  return {
    invert, simpson,
    expPdf, expCdf, expQuant, expDraw,
    gammaPdf, gammaCdf, gammaQuant, gammaDraw,
    betaPdf, betaCdf, betaQuant, betaDraw,
    paretoPdf, paretoCdf, paretoQuant, paretoDraw,
    lognormPdf, lognormCdf, lognormQuant, lognormDraw
  };
})());

/* ═══ Joint distributions — appended for Joint Distributions & Dependence (part 5) ═══════════════
   Adds to PV only what neither ST nor the blocks above have:

     · bivariate normal   PV.bvnPdf(x, y, mx, my, sx, sy, rho)   joint density
                          PV.bvnDraw(mx, my, sx, sy, rho, r) → [x, y]
                          by the Cholesky construction X = mx + sx·Z₁, Y = my + sy·(ρZ₁ + √(1 − ρ²)Z₂)
     · Dirichlet          PV.dirichletDraw(alpha, r) → p   normalised gammas (PV.gammaDraw, rate 1)
                          PV.dirichletLogPdf(p, alpha)      log density on the simplex
     · convolution        PV.convolve(p, q) → array          discrete convolution of two pmf arrays
                                                             indexed from 0 (result length |p| + |q| − 1)

   VERIFICATION (run 2026-10-01 under node with stats-viz.js + this file; script verify-jd.js kept in the
   build scratchpad). Each line is "check — result":
     bvnPdf
       · midpoint-grid integral over ±8 sd for (sx, sy, ρ) ∈ {(1, 1, 0), (1, 1, 0.6), (2, 0.5, −0.9),
         (10.2, 11.8, 0.67)} — 1.00000000 each; grid ∫∫ xy·f = ρ·sx·sy to 6 d.p. (e.g. 80.641200)
       · ρ = 0 factorises: bvnPdf(0.3, −1.2) − φ(0.3)φ(−1.2) — 1.4e-17
     bvnDraw
       · 200,000 draws, (mx, my, sx, sy, ρ) = (1, −2, 2, 3, 0.6), seed 20261001 — means 1.0008, −1.9925;
         variances 4.013, 9.018; correlation 0.6034 (sampling sd of the correlation ≈ 0.0014)
     dirichletDraw / dirichletLogPdf
       · 200,000 draws of Dir(2, 3, 5), seed 77 — means 0.1999, 0.2998, 0.5003; Var p₁ 0.01454
         (exact 0.014545); Cov(p₁, p₂) −0.00551 (exact −0.005455)
       · midpoint integral of exp(dirichletLogPdf) over the 3-simplex, 600² cells — 1.0000083
       · K = 2 equals PV.betaPdf: Dir([0.3, 0.7]; 2.5, 4) vs Beta(2.5, 4) at 0.3 — 1.8e-15
     convolve
       · two fair dice: 36·P(S = s) = 1 2 3 4 5 6 5 4 3 2 1 — exact
       · Poisson(2) * Poisson(3) vs Poisson(5), k < 50 — max |diff| 3.9e-16
       · Bin(10, 0.3) * Bin(7, 0.3) vs Bin(17, 0.3) — max |diff| 1.4e-15                         */
Object.assign(PV, (function () {
  function bvnPdf(x, y, mx, my, sx, sy, rho) {
    const zx = (x - mx) / sx, zy = (y - my) / sy, om = 1 - rho * rho;
    return Math.exp(-(zx * zx - 2 * rho * zx * zy + zy * zy) / (2 * om)) / (2 * Math.PI * sx * sy * Math.sqrt(om));
  }
  function bvnDraw(mx, my, sx, sy, rho, r) {
    const z1 = ST.randn(r), z2 = ST.randn(r);
    return [mx + sx * z1, my + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2)];
  }
  function dirichletDraw(alpha, r) {
    const g = alpha.map(a => PV.gammaDraw(a, 1, r));
    const s = g.reduce((a, b) => a + b, 0);
    return g.map(v => v / s);
  }
  function dirichletLogPdf(p, alpha) {
    let l = ST.lnGamma(alpha.reduce((a, b) => a + b, 0));
    for (let i = 0; i < alpha.length; i++) {
      if (!(p[i] > 0)) return -Infinity;
      l += (alpha[i] - 1) * Math.log(p[i]) - ST.lnGamma(alpha[i]);
    }
    return l;
  }
  function convolve(p, q) {
    const out = new Array(p.length + q.length - 1).fill(0);
    for (let i = 0; i < p.length; i++) if (p[i]) for (let j = 0; j < q.length; j++) out[i + j] += p[i] * q[j];
    return out;
  }
  return { bvnPdf, bvnDraw, dirichletDraw, dirichletLogPdf, convolve };
})());

/* ── appended for Limit Theorems & Convergence (part 6) ─────────────────────────────────────────
   Adds ONLY what neither ST nor PV had: the Cauchy, the two heavy/light-tailed extreme-value laws
   used as limits of maxima, and a Kolmogorov distance between a sample and a cdf.

     · Cauchy        PV.cauchyPdf(x, x0, s), PV.cauchyCdf(x, x0, s), PV.cauchyQuant(p, x0, s),
                     PV.cauchyDraw(x0, s, r)  (inverse transform x0 + s·tan(π(U − ½)))
     · Gumbel        PV.gumbelPdf(x), PV.gumbelCdf(x) = exp(−e^(−x)), PV.gumbelQuant(p)  (standard)
     · Fréchet       PV.frechetPdf(x, α), PV.frechetCdf(x, α) = exp(−x^(−α)) for x > 0  (standard)
     · comparison    PV.ksDist(sortedAsc, cdf) = sup_x |F̂ₙ(x) − F(x)|, checked on both sides of
                     every jump of the empirical cdf

   VERIFICATION (run 2026-10-01 under node with stats-viz.js + this file loaded; script
   lt-pv-check.js kept in the build scratchpad). Each line is "check — result":
     cauchyPdf / cauchyCdf / cauchyQuant / cauchyDraw
       · Simpson ∫ of cauchyPdf over ±10⁴ — 0.99993634, equal to (2/π)·atan(10⁴) to 8 d.p.
       · cdf(x0 = 2, s = 3) vs ½ + ∫ of the pdf from the centre, x ∈ {−10, −1, 2, 3.5, 20} — max |diff| 3.3e-15
       · cauchyCdf(cauchyQuant(p)) − p for p from 0.001 to 0.999 — max 2.8e-17
       · P(|X| > 3) = 1 − (2/π)·atan 3 = 0.204833 — matches
       · 200,000 draws (x0 = 0, s = 1, seed 20261001): median 0.0037, quartiles −0.9905 and 1.0129
         (exact ∓1), Kolmogorov distance to cauchyCdf 0.0023 (DKW 95% band 0.0030)
     gumbelPdf / gumbelCdf / gumbelQuant
       · ∫ pdf = 1.0000000000, mean 0.57721566 (Euler's γ), variance 1.64493407 (π²/6) — to 8 d.p.
       · ∫ pdf vs cdf on x ∈ [−3, 8] — max |diff| 7.7e-15;  cdf(quant(p)) − p — max 1.4e-17
       · exponential maxima: sup over [−2, 10] of |(1 − e^(−x)/1000)^1000 − Λ(x)| — 2.71e-4
     frechetPdf / frechetCdf (α = 3)
       · ∫ pdf = 1.000000; ∫ pdf vs cdf on [0.3, 6] — max |diff| 9.4e-15
       · median (ln 2)^(−1/3) = 1.12995 has cdf 0.500000; mean Γ(1 − 1/3) = 1.354118, numeric 1.3541
     ksDist
       · 500 N(0, 1) draws (seed 5) vs ST.normCdf: 0.031825, identical to a brute-force sup over both
         sides of every jump of the empirical cdf                                                    */
Object.assign(PV, (function () {
  const cauchyPdf = (x, x0, s) => { const a = x0 || 0, b = s || 1, z = (x - a) / b; return 1 / (Math.PI * b * (1 + z * z)); };
  const cauchyCdf = (x, x0, s) => 0.5 + Math.atan((x - (x0 || 0)) / (s || 1)) / Math.PI;
  const cauchyQuant = (p, x0, s) => (x0 || 0) + (s || 1) * Math.tan(Math.PI * (p - 0.5));
  function cauchyDraw(x0, s, r) {
    let u = r(); while (u === 0) u = r();
    return cauchyQuant(u, x0, s);
  }
  const gumbelCdf = x => Math.exp(-Math.exp(-x));
  const gumbelPdf = x => { const e = Math.exp(-x); return e * Math.exp(-e); };
  const gumbelQuant = p => -Math.log(-Math.log(p));
  const frechetCdf = (x, a) => (x > 0 ? Math.exp(-Math.pow(x, -a)) : 0);
  const frechetPdf = (x, a) => (x > 0 ? a * Math.pow(x, -a - 1) * Math.exp(-Math.pow(x, -a)) : 0);
  function ksDist(sorted, cdf) {
    const n = sorted.length;
    let d = 0;
    for (let i = 0; i < n; i++) {
      const F = cdf(sorted[i]);
      d = Math.max(d, Math.abs((i + 1) / n - F), Math.abs(F - i / n));
    }
    return d;
  }
  return { cauchyPdf, cauchyCdf, cauchyQuant, cauchyDraw, gumbelPdf, gumbelCdf, gumbelQuant, frechetPdf, frechetCdf, ksDist };
})());
