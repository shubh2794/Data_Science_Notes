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
