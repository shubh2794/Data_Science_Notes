/* dl-viz.js — the shared D3 + numeric library for the Deep Learning series
   (deep-learning/*). This is the DL analogue of vision/vision-viz.js and of
   math/statistics/stats-viz.js.

   ── THE RULE ────────────────────────────────────────────────────────────────
   Every later part of this series — Training, CNNs, RNNs, Attention,
   Transformers, Residual connections — must CALL these functions rather than
   re-implement them. A forked activation convention (three pages each with a
   slightly different GELU), a forked softmax (one stable, one not), or a forked
   cross-entropy sign is the specific failure this file exists to prevent. If a
   later part needs a numeric that is GENERAL, promote it here and say so in a
   comment; if it is specific to one figure, keep it inside that page's own
   "<part>.viz.js" IIFE.

   Load order on every consuming page:
       ../vendor/d3.min.js → ../notes.css → ../data.js → ../notes.js
       → dl-viz.js → <part>.viz.js
   Nothing here draws by itself; it is a toolbox. Every consuming page supplies
   its own <svg width height viewBox role aria-label> and these helpers only
   ever append into an <svg> that already exists.

   Two globals:

   DC — palette, matching the custom properties in notes.css.

   DL — namespace:
     · scalars     DL.clamp, DL.lerp, DL.linspace, DL.fmt, DL.sig, DL.fmtE
     · random      DL.rng(seed) → mulberry32; DL.randn(r); DL.shuffle(a, r)
     · linear alg  DL.zeros, DL.zeros2, DL.eye, DL.matmul(A,B), DL.matvec(A,v),
                   DL.vecmat(v,A), DL.transpose(A), DL.outer(a,b), DL.addRow(A,b),
                   DL.dot, DL.frob, DL.applyEl
     · activations DL.ACT — a registry keyed by name. Each entry is
                     {key, label, f(x[,p]), df(x[,p]), range, zeroCentered,
                      piecewiseLinear, note}
                   relu · leaky · prelu · elu · gelu · geluTanh · geluSigmoid ·
                   silu · sigmoid · tanh · softplus · hardtanh · abs · identity
                   DL.act(key) → the entry; DL.actList(group) → ordered keys.
                   NOTE gelu (exact, via erf) and geluTanh (the tanh surrogate)
                   are DIFFERENT FUNCTIONS: max gap 4.73e−04 at x ≈ 2.70. The
                   series teaches that gap; never silently swap one for the other.
     · stable      DL.erf, DL.logSumExp(v), DL.softmax(v[,T]), DL.logSoftmax(v),
                   DL.logSigmoid(z), DL.softplus(z), DL.sigmoid(z),
                   DL.xent(logits, k), DL.bce(z, y), DL.mse(yhat, y)
                   — every one of these is the OVERFLOW-SAFE form. The naive
                     forms are exposed as DL.naive.* so the pages can show the
                     two diverging; they are not for use anywhere else.
     · tiny MLP    DL.mlpInit(sizes, {act, out, seed, scale}) → net
                   DL.forward(net, X) → {z[], a[], yhat}
                   DL.loss(net, X, Y) → scalar mean loss
                   DL.backward(net, X, Y) → {dW[], db[], loss}
                   DL.numGrad(net, X, Y, h) → the same shapes by central
                     differences, for the finite-difference audit
                   DL.sgdStep(net, g, lr)  — one step, for demo fitting only;
                     the OPTIMISERS themselves are taught on the training page
     · counting    DL.params(sizes), DL.macs(sizes), DL.regionsShallow(d, n),
                   DL.regionsDeepLB(d, widths)
     · stats       DL.mean(a), DL.variance(a[,ddof]), DL.std(a[,ddof]),
                   DL.standardise(X) → {X, mu, sd}            [added by part 2]
     · linear fit  DL.logistic(X, y, steps, lr, l2) → {w, b}; DL.logit(m, x);
                   DL.accuracy(m, X, y); DL.logisticLoss(m, X, y) — the stable
                   mean cross-entropy, so a separating fit reports a large
                   FINITE number                                [part 2]
     · spectra     DL.eig2sym(a,b,c) → {lo, hi, vlo, vhi, cond}  — EXACT, for
                   the conditioning claims; DL.powerIter(A) → {lambda, v}:
                   the top |λ| AND its eigenvector (the header used to name
                   only the value; the function has always returned both);
                   DL.specNorm(A) → ‖A‖₂, the top SINGULAR value, which is the
                   quantity that governs a product of Jacobians [part 2]
     · averaging   DL.ewma(series, β, correct) — v_t = βv_{t−1} + (1−β)θ_t with
                   the optional bias correction v_t/(1−βᵗ)      [part 2]
     · optimisers  DL.OPT — a registry keyed by name, each
                     {key, label, hp, init(n) → state, step(state, g, hp) → dx}
                   where dx is ADDED to the parameters. DL.optList() orders it.
                   sgd · momentum · momentumEwma · nesterov · adagrad ·
                   rmsprop · adam · adamw.
                   NOTE "momentum" is the ACCUMULATION form v ← βv + g and
                   "momentumEwma" is v ← βv + (1−β)g. They are different
                   algorithms: the first multiplies the effective step by
                   1/(1−β) — ten times at β = 0.9 — and the second does not.
                   The series teaches that gap; never silently swap them.
                   Adam's coupled L2 and AdamW's decoupled decay are likewise
                   separate entries because they are separate algorithms.
     · schedules   DL.lrAt(kind, t, {peak, total, warm, floor, step, gamma,
                   power}) — constant · cosine · linear · step · exp · power ·
                   invsqrt · onecycle, with LINEAR warmup applied as a prefix
                   to every one of them                         [part 2]
     · norm layers DL.batchNorm(X, γ, β, eps[, stats]) → {Y, mu, va,
                   vaUnbiased} — pass `stats` for EVAL mode; the biased (1/N)
                   variance normalises the batch and the unbiased (1/(N−1))
                   one is what belongs in the running estimate.
                   DL.layerNorm(X, γ, β, eps), DL.rmsNorm(X, γ, eps) — these
                   normalise ACROSS a row and so never depend on the batch.
     · clipping    DL.clipValue(g, c) — changes the DIRECTION;
                   DL.clipNorm(g, c) → {g, norm, scaled, factor} — does not
     · convolution DL.outSize(n, f, {s, p0, p1, d}), DL.effK(f, d),
                   DL.padFor("valid"|"same"|"full", n, f, s, d) → {p0, p1,
                   total, symmetric}; DL.corr2d / DL.conv2dTrue (single
                   channel), DL.conv2dMC(V, K, b, o) (CHW multi-channel, with
                   stride, dilation and groups), DL.convGradK / DL.convGradV /
                   DL.convGradB (the two backward operations, audited against
                   central differences to 2.9e−09 relative), DL.pool2d,
                   DL.globalPool, DL.convCount, DL.rfChain, DL.im2col,
                   DL.ker2col, DL.convMatrix, DL.zeros3, DL.flipK  [part 3]
                   READ THE SCOPE BOUNDARY comment above these: CLASSICAL
                   FILTERING (border modes, separable and Gaussian kernels)
                   belongs to vision/vision-viz.js and is NOT duplicated here;
                   on the overlap the two agree to 0.000e+00.
     · drawing     DL.frame, DL.axisB, DL.axisL, DL.gridX, DL.gridY, DL.legend,
                   DL.panelBox, DL.kv, DL.matText, DL.arrow, DL.curve, DL.clip,
                   DL.cells, DL.netDiagram(g, sizes, opt)
                                                                              */

const DC = {
  accent: "#5b9cff", a2: "#ffb454", good: "#4ade80", bad: "#f87171",
  ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a",
  grid: "#1b2130", panel: "#171a23", panel2: "#1e222d", bg: "#0f1117",
  violet: "#c084fc", teal: "#2dd4bf", rose: "#fb7185", lime: "#a3e635"
};

const DL = (function () {

  /* ══ scalars ════════════════════════════════════════════════════════════ */
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  function linspace(a, b, k) {
    const out = [];
    for (let i = 0; i < k; i++) out.push(a + (b - a) * (k === 1 ? 0 : i / (k - 1)));
    return out;
  }
  function fmt(x, d) {
    if (!isFinite(x)) return (x > 0 ? "+∞" : (x < 0 ? "−∞" : "—"));
    const dd = (d === undefined) ? 2 : d;
    const s = x.toFixed(dd);
    return (parseFloat(s) === 0) ? (0).toFixed(dd) : s;    // never print "-0.00"
  }
  const sig = (x, n) => isFinite(x) ? Number(x.toPrecision(n || 3)).toString() : "—";
  function fmtE(x, d) {                                    // 1.23e−7, Unicode minus
    if (!isFinite(x)) return "—";
    if (x === 0) return "0";
    return x.toExponential(d === undefined ? 2 : d).replace("e-", "e−").replace("e+", "e+");
  }
  /* 12 345 678 → "12.3M" — parameter counts are read, not summed, in a caption */
  function big(x) {
    const a = Math.abs(x);
    if (a >= 1e12) return (x / 1e12).toFixed(a >= 1e13 ? 0 : 1) + "T";
    if (a >= 1e9) return (x / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "B";
    if (a >= 1e6) return (x / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
    if (a >= 1e3) return (x / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "k";
    return String(Math.round(x));
  }
  const commas = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");

  /* ══ seeded randomness, so every reader sees the same picture ═══════════ */
  function rng(seed) {                                     // mulberry32
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) {                                      // Box–Muller, one draw
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function shuffle(arr, r) {                               // Fisher–Yates, in place
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  /* ══ dense linear algebra ═══════════════════════════════════════════════
     Row-major arrays of arrays. The SERIES CONVENTION is the row convention:
     a batch X is (N × d), one EXAMPLE PER ROW, and a layer is X·W + b with
     W of shape (dᵢₙ × dₒᵤₜ). Column-convention sources write Wᵀx + b with W of
     shape (dₒᵤₜ × dᵢₙ); the two are transposes of one another and mixing them
     is the single most common shape bug in the series. Everything here is row
     convention; where a page must show the other one it says so explicitly.  */
  const zeros = n => new Array(n).fill(0);
  function zeros2(m, n) { const A = []; for (let i = 0; i < m; i++) A.push(new Array(n).fill(0)); return A; }
  function eye(n) { const A = zeros2(n, n); for (let i = 0; i < n; i++) A[i][i] = 1; return A; }
  function transpose(A) {
    const m = A.length, n = A[0].length, B = zeros2(n, m);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) B[j][i] = A[i][j];
    return B;
  }
  function matmul(A, B) {                                  // (m×k)(k×n) → (m×n)
    const m = A.length, k = B.length, n = B[0].length, C = zeros2(m, n);
    for (let i = 0; i < m; i++) {
      const Ai = A[i], Ci = C[i];
      for (let p = 0; p < k; p++) {
        const a = Ai[p]; if (a === 0) continue;
        const Bp = B[p];
        for (let j = 0; j < n; j++) Ci[j] += a * Bp[j];
      }
    }
    return C;
  }
  const matvec = (A, v) => A.map(row => row.reduce((s, a, j) => s + a * v[j], 0));
  function vecmat(v, A) {                                  // (1×m)(m×n) → length n
    const n = A[0].length, out = zeros(n);
    for (let i = 0; i < A.length; i++) { const vi = v[i]; if (!vi) continue; for (let j = 0; j < n; j++) out[j] += vi * A[i][j]; }
    return out;
  }
  const outer = (a, b) => a.map(ai => b.map(bj => ai * bj));
  const addRow = (A, b) => A.map(row => row.map((v, j) => v + b[j]));   // broadcast
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const frob = A => Math.sqrt(A.reduce((s, r) => s + r.reduce((t, v) => t + v * v, 0), 0));
  const applyEl = (A, f) => A.map(r => r.map(f));

  /* ══ the error function, for the EXACT GELU ═════════════════════════════
     This is the rational-Chebyshev form of the complementary error function.
     Verified in the build against Python's math.erf on a 0.01 grid over
     |x| ≤ 6: max ABSOLUTE error 8.30e−08 (at x = −0.04), max RELATIVE error
     2.73e−06 (at x = −0.02, where erf itself is ≈ 0.02 so the relative figure
     is misleadingly large). Both are three to four orders of magnitude below
     the ≈4.7e−04 gap between the exact GELU and its tanh surrogate, which is
     the quantity the series actually asks this function to resolve.          */
  function erf(x) {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.5 * z);
    const y = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - y : y - 1;
  }
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));        // standard normal CDF
  const phi = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

  /* ══ overflow-safe scalar primitives ════════════════════════════════════ */
  function sigmoid(z) {                                    // never exp() a big positive
    return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }
  function softplus(z) {                                   // log(1 + eᶻ) = max(z,0) + log(1 + e^−|z|)
    return Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
  }
  function logSigmoid(z) {                                 // = −softplus(−z)
    return -softplus(-z);
  }
  function logSumExp(v) {
    let m = -Infinity;
    for (const x of v) if (x > m) m = x;
    if (!isFinite(m)) return m;
    let s = 0;
    for (const x of v) s += Math.exp(x - m);
    return m + Math.log(s);
  }
  function softmax(v, T) {
    const t = (T === undefined || T === null) ? 1 : T;
    if (t === 0) {                                          // the argmax limit
      let m = -Infinity, k = 0;
      v.forEach((x, i) => { if (x > m) { m = x; k = i; } });
      return v.map((_, i) => (i === k ? 1 : 0));
    }
    const s = v.map(x => x / t);
    let m = -Infinity;
    for (const x of s) if (x > m) m = x;
    const e = s.map(x => Math.exp(x - m));
    const S = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / S);
  }
  function logSoftmax(v, T) {
    const t = (T === undefined) ? 1 : T;
    const s = v.map(x => x / t), L = logSumExp(s);
    return s.map(x => x - L);
  }
  /* categorical cross-entropy from LOGITS and an integer class, in the stable
     form −(z_k − logsumexp z). Passing probabilities here is the classic
     double-softmax bug; this signature takes logits and nothing else.        */
  const xent = (logits, k) => logSumExp(logits) - logits[k];
  /* binary cross-entropy from a LOGIT and a 0/1 label:
     −[y·logσ(z) + (1−y)·log(1−σ(z))] = softplus((1−2y)·z)                    */
  const bce = (z, y) => softplus((1 - 2 * y) * z);
  const mse = (yhat, y) => 0.5 * (yhat - y) * (yhat - y);

  /* The naive forms, exposed ONLY so a figure can show them blowing up.
     Nothing else in the series may call these.                              */
  const naive = {
    sigmoid: z => 1 / (1 + Math.exp(-z)),
    logSigmoid: z => Math.log(1 / (1 + Math.exp(-z))),
    softplus: z => Math.log(1 + Math.exp(z)),
    softmax: v => { const e = v.map(Math.exp), S = e.reduce((a, b) => a + b, 0); return e.map(x => x / S); },
    xent: (logits, k) => { const e = logits.map(Math.exp), S = e.reduce((a, b) => a + b, 0); return -Math.log(e[k] / S); },
    bce: (z, y) => { const p = 1 / (1 + Math.exp(-z)); return -(y * Math.log(p) + (1 - y) * Math.log(1 - p)); }
  };

  /* ══ the activation registry ════════════════════════════════════════════
     f and df are SCALAR. df at a kink returns the right derivative, which is
     what every framework does; the pages say so where it matters.
     `p` is the one shape parameter a family needs (leaky/PReLU slope, ELU α). */
  const ACT = {
    identity: {
      key: "identity", label: "identity", group: "linear",
      f: x => x, df: () => 1, range: "(−∞, ∞)", zeroCentered: true, piecewiseLinear: true,
      note: "No non-linearity at all. Stacking these collapses the whole network to one affine map."
    },
    relu: {
      key: "relu", label: "ReLU", group: "rectified",
      f: x => x > 0 ? x : 0, df: x => x > 0 ? 1 : 0,
      range: "[0, ∞)", zeroCentered: false, piecewiseLinear: true,
      note: "max(0, x). Two linear pieces, one kink. Derivative is exactly 1 where active and exactly 0 where not."
    },
    leaky: {
      key: "leaky", label: "leaky ReLU", group: "rectified", p: 0.01, pLabel: "slope",
      f: (x, p) => x > 0 ? x : (p === undefined ? 0.01 : p) * x,
      df: (x, p) => x > 0 ? 1 : (p === undefined ? 0.01 : p),
      range: "(−∞, ∞)", zeroCentered: false, piecewiseLinear: true,
      note: "A fixed small negative slope, so the derivative is never exactly zero."
    },
    prelu: {
      key: "prelu", label: "PReLU", group: "rectified", p: 0.25, pLabel: "learned slope",
      f: (x, p) => x > 0 ? x : (p === undefined ? 0.25 : p) * x,
      df: (x, p) => x > 0 ? 1 : (p === undefined ? 0.25 : p),
      range: "(−∞, ∞)", zeroCentered: false, piecewiseLinear: true,
      note: "Leaky ReLU with the negative slope made a learned parameter, one per channel."
    },
    abs: {
      key: "abs", label: "absolute value", group: "rectified",
      f: x => Math.abs(x), df: x => x >= 0 ? 1 : -1,
      range: "[0, ∞)", zeroCentered: false, piecewiseLinear: true,
      note: "α = −1. Folds the input space about the unit's own hyperplane — the picture behind region counting."
    },
    elu: {
      key: "elu", label: "ELU", group: "smooth", p: 1.0, pLabel: "<span class=\"keep\">α</span>",
      f: (x, p) => { const a = (p === undefined ? 1 : p); return x > 0 ? x : a * (Math.exp(Math.min(x, 0)) - 1); },
      df: (x, p) => { const a = (p === undefined ? 1 : p); return x > 0 ? 1 : a * Math.exp(Math.min(x, 0)); },
      range: "(−α, ∞)", zeroCentered: false, piecewiseLinear: false,
      note: "Exponential below zero: saturates to −α instead of to 0, pulling the mean activation toward zero."
    },
    softplus: {
      key: "softplus", label: "softplus", group: "smooth",
      f: z => softplus(z), df: z => sigmoid(z),
      range: "(0, ∞)", zeroCentered: false, piecewiseLinear: false,
      note: "log(1 + eᶻ), the smooth rectifier. Its derivative is exactly the logistic sigmoid."
    },
    gelu: {
      key: "gelu", label: "GELU (exact)", group: "smooth",
      f: x => x * Phi(x),
      df: x => Phi(x) + x * phi(x),
      range: "≈(−0.17, ∞)", zeroCentered: false, piecewiseLinear: false,
      note: "x·Φ(x) with Φ the standard normal CDF. Non-monotone: minimum −0.169971 at x = −0.751792."
    },
    geluTanh: {
      key: "geluTanh", label: "GELU (tanh form)", group: "smooth",
      f: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))),
      df: x => {
        const c = Math.sqrt(2 / Math.PI), u = c * (x + 0.044715 * x * x * x);
        const t = Math.tanh(u), du = c * (1 + 3 * 0.044715 * x * x);
        return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
      },
      range: "≈(−0.17, ∞)", zeroCentered: false, piecewiseLinear: false,
      note: "The tanh surrogate. A DIFFERENT function from the exact form: max gap 4.73e−04 at x ≈ 2.70."
    },
    geluSigmoid: {
      key: "geluSigmoid", label: "GELU (sigmoid form)", group: "smooth",
      f: x => x * sigmoid(1.702 * x),
      df: x => { const s = sigmoid(1.702 * x); return s + x * 1.702 * s * (1 - s); },
      range: "≈(−0.17, ∞)", zeroCentered: false, piecewiseLinear: false,
      note: "x·σ(1.702x), the cheapest surrogate and much the least accurate: max gap 2.03e−02 at x ≈ −2.27."
    },
    silu: {
      key: "silu", label: "SiLU / swish", group: "smooth", p: 1.0, pLabel: "<span class=\"keep\">β</span>",
      f: (x, p) => x * sigmoid((p === undefined ? 1 : p) * x),
      df: (x, p) => { const b = (p === undefined ? 1 : p), s = sigmoid(b * x); return s + b * x * s * (1 - s); },
      range: "≈(−0.28, ∞)", zeroCentered: false, piecewiseLinear: false,
      note: "x·σ(βx). Non-monotone: minimum −0.278465 at x = −1.278465 for β = 1. β → ∞ recovers ReLU."
    },
    sigmoid: {
      key: "sigmoid", label: "logistic sigmoid", group: "saturating",
      f: z => sigmoid(z), df: z => { const s = sigmoid(z); return s * (1 - s); },
      range: "(0, 1)", zeroCentered: false, piecewiseLinear: false,
      note: "Two-sided saturation, output not centred at zero, maximum derivative only 0.25."
    },
    tanh: {
      key: "tanh", label: "tanh", group: "saturating",
      f: z => Math.tanh(z), df: z => 1 - Math.tanh(z) * Math.tanh(z),
      range: "(−1, 1)", zeroCentered: true, piecewiseLinear: false,
      note: "2σ(2z) − 1: the same shape, rescaled to be zero-centred with unit slope at the origin."
    },
    hardtanh: {
      key: "hardtanh", label: "hard tanh", group: "saturating",
      f: z => Math.max(-1, Math.min(1, z)), df: z => (z > -1 && z < 1) ? 1 : 0,
      range: "[−1, 1]", zeroCentered: true, piecewiseLinear: true,
      note: "clip(z, −1, 1). Piecewise linear and bounded — three pieces, two kinks."
    }
  };
  const act = k => ACT[k] || ACT.relu;
  function actList(group) {
    const order = ["identity", "relu", "leaky", "prelu", "abs", "elu", "softplus",
      "gelu", "geluTanh", "geluSigmoid", "silu", "sigmoid", "tanh", "hardtanh"];
    return group ? order.filter(k => ACT[k].group === group) : order;
  }
  /* maxout is NOT element-wise, so it cannot live in ACT: it consumes k
     pre-activations and emits one. Given a (k × 1) group of z values it is
     just the max; the pieces are the argmax cells.                          */
  const maxout = zs => Math.max.apply(null, zs);

  /* ══ a tiny dense MLP: forward, loss, exact backward, numeric backward ═══
     sizes = [d_in, h1, …, d_out]. Weights are (dᵢₙ × dₒᵤₜ), inputs are rows.
     out: "linear" (mean squared error), "sigmoid" (binary cross-entropy from
     the logit) or "softmax" (categorical cross-entropy from the logits).
     Y for "linear" is (N × dₒᵤₜ); for "sigmoid" (N × 1) of 0/1; for "softmax"
     an array of N integer class indices.
     The backward pass here is verified against DL.numGrad in the build; the
     ALGORITHM is taught on deep-learning/neural-network-training.html and is
     not re-derived by any figure that calls this.                            */
  function mlpInit(sizes, opt) {
    const o = Object.assign({ act: "relu", out: "linear", seed: 7, scale: null, p: undefined }, opt || {});
    const r = rng(o.seed), W = [], b = [];
    for (let l = 0; l + 1 < sizes.length; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const s = (o.scale === null) ? Math.sqrt(2 / nin) : o.scale;
      const Wl = zeros2(nin, nout);
      for (let i = 0; i < nin; i++) for (let j = 0; j < nout; j++) Wl[i][j] = s * randn(r);
      W.push(Wl); b.push(zeros(nout));
    }
    return { sizes: sizes.slice(), W: W, b: b, act: o.act, out: o.out, p: o.p, L: sizes.length - 1 };
  }
  function forward(net, X) {
    const g = act(net.act).f, z = [], a = [X];
    let h = X;
    for (let l = 0; l < net.L; l++) {
      const zl = addRow(matmul(h, net.W[l]), net.b[l]);
      z.push(zl);
      h = (l === net.L - 1) ? zl : applyEl(zl, x => g(x, net.p));
      if (l < net.L - 1) a.push(h);
    }
    return { z: z, a: a, logits: z[net.L - 1] };
  }
  function predict(net, X) {
    const f = forward(net, X);
    if (net.out === "sigmoid") return f.logits.map(r => [sigmoid(r[0])]);
    if (net.out === "softmax") return f.logits.map(r => softmax(r));
    return f.logits;
  }
  function lossFrom(net, logits, Y) {
    const N = logits.length;
    let s = 0;
    for (let i = 0; i < N; i++) {
      if (net.out === "softmax") s += xent(logits[i], Y[i]);
      else if (net.out === "sigmoid") s += bce(logits[i][0], Y[i][0]);
      else for (let j = 0; j < logits[i].length; j++) s += mse(logits[i][j], Y[i][j]);
    }
    return s / N;
  }
  const loss = (net, X, Y) => lossFrom(net, forward(net, X).logits, Y);

  function backward(net, X, Y) {
    const g = act(net.act), N = X.length;
    const f = forward(net, X);
    const L = net.L;
    /* δ at the OUTPUT layer. For all three (output unit, loss) pairs the
       maximum-likelihood pairing makes this exactly (prediction − target),
       which is the whole point of §17 on the page. */
    let d = zeros2(N, net.sizes[L]);
    for (let i = 0; i < N; i++) {
      if (net.out === "softmax") {
        const p = softmax(f.logits[i]);
        for (let j = 0; j < p.length; j++) d[i][j] = (p[j] - (j === Y[i] ? 1 : 0)) / N;
      } else if (net.out === "sigmoid") {
        d[i][0] = (sigmoid(f.logits[i][0]) - Y[i][0]) / N;
      } else {
        for (let j = 0; j < f.logits[i].length; j++) d[i][j] = (f.logits[i][j] - Y[i][j]) / N;
      }
    }
    const dW = [], db = [];
    for (let l = L - 1; l >= 0; l--) {
      const A = f.a[l];                                     // input to layer l
      dW[l] = matmul(transpose(A), d);
      db[l] = d[0].map((_, j) => d.reduce((s, row) => s + row[j], 0));
      if (l > 0) {
        const back = matmul(d, transpose(net.W[l]));
        const zprev = f.z[l - 1];
        d = back.map((row, i) => row.map((v, j) => v * g.df(zprev[i][j], net.p)));
      }
    }
    return { dW: dW, db: db, loss: lossFrom(net, f.logits, Y) };
  }
  /* central differences over every parameter — O(P) forward passes, only ever
     used to CHECK backward(), never inside a figure's animation loop.        */
  function numGrad(net, X, Y, h) {
    const hh = h || 1e-5, dW = [], db = [];
    for (let l = 0; l < net.L; l++) {
      dW[l] = zeros2(net.sizes[l], net.sizes[l + 1]);
      db[l] = zeros(net.sizes[l + 1]);
      for (let i = 0; i < net.sizes[l]; i++) for (let j = 0; j < net.sizes[l + 1]; j++) {
        const o = net.W[l][i][j];
        net.W[l][i][j] = o + hh; const Lp = loss(net, X, Y);
        net.W[l][i][j] = o - hh; const Lm = loss(net, X, Y);
        net.W[l][i][j] = o;
        dW[l][i][j] = (Lp - Lm) / (2 * hh);
      }
      for (let j = 0; j < net.sizes[l + 1]; j++) {
        const o = net.b[l][j];
        net.b[l][j] = o + hh; const Lp = loss(net, X, Y);
        net.b[l][j] = o - hh; const Lm = loss(net, X, Y);
        net.b[l][j] = o;
        db[l][j] = (Lp - Lm) / (2 * hh);
      }
    }
    return { dW: dW, db: db };
  }
  function sgdStep(net, g, lr) {
    for (let l = 0; l < net.L; l++) {
      for (let i = 0; i < net.sizes[l]; i++)
        for (let j = 0; j < net.sizes[l + 1]; j++) net.W[l][i][j] -= lr * g.dW[l][i][j];
      for (let j = 0; j < net.sizes[l + 1]; j++) net.b[l][j] -= lr * g.db[l][j];
    }
    return net;
  }
  function cloneNet(net) {
    return Object.assign({}, net, {
      W: net.W.map(A => A.map(r => r.slice())),
      b: net.b.map(v => v.slice())
    });
  }

  /* ══ counting ═══════════════════════════════════════════════════════════ */
  /* parameters of a dense chain, biases included */
  function params(sizes, bias) {
    const wb = (bias === false) ? 0 : 1;
    let p = 0;
    for (let l = 0; l + 1 < sizes.length; l++) p += sizes[l] * sizes[l + 1] + wb * sizes[l + 1];
    return p;
  }
  /* multiply–accumulate count for ONE example, forward only. A MAC is one
     multiply plus one add; the FLOP count quoted in papers is usually 2×MAC. */
  function macs(sizes) {
    let m = 0;
    for (let l = 0; l + 1 < sizes.length; l++) m += sizes[l] * sizes[l + 1];
    return m;
  }
  /* Exact maximum number of linear regions of a ONE-hidden-layer ReLU net with
     d inputs and n units: ∑_{j=0}^{min(d,n)} C(n, j)  (Zaslavsky). Uses exact
     integer binomials while they fit in a double, then floats.               */
  function binom(n, k) {
    if (k < 0 || k > n) return 0;
    k = Math.min(k, n - k);
    let r = 1;
    for (let i = 1; i <= k; i++) r = r * (n - k + i) / i;
    return r;
  }
  function regionsShallow(d, n) {
    let s = 0;
    for (let j = 0; j <= Math.min(d, n); j++) s += binom(n, j);
    return s;
  }
  /* Montúfar-style LOWER bound for a deep rectifier net with d inputs and
     hidden widths [n₁ … n_L], all n ≥ d:
         (∏_{l<L} ⌊n_l/d⌋^d) · ∑_{j=0}^{d} C(n_L, j)
     Note it is a lower bound on the MAXIMUM over parameters — an existence
     claim about one weight setting, not a statement about a trained net.     */
  function regionsDeepLB(d, widths) {
    let prod = 1;
    for (let l = 0; l + 1 < widths.length; l++) prod *= Math.pow(Math.floor(widths[l] / d), d);
    return prod * regionsShallow(d, widths[widths.length - 1]);
  }

  /* ══ statistics on flat arrays ══════════════════════════════════════════
     Added by part 2 (Training). Sample (1/n) moments unless ddof is given;
     the BATCH-NORM distinction between the 1/m used to normalise and the
     1/(m−1) used for the RUNNING estimate is why ddof is exposed at all.   */
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  function variance(a, ddof) {
    const m = mean(a), d = (ddof === undefined ? 0 : ddof);
    return a.reduce((s, x) => s + (x - m) * (x - m), 0) / Math.max(1, a.length - d);
  }
  const std = (a, ddof) => Math.sqrt(variance(a, ddof));
  /* column-wise standardisation of an (N × d) row-major batch. Promoted from
     neural-networks.viz.js in part 2, which needed the same routine for the
     input-conditioning figure — one concept, one implementation.            */
  function standardise(X) {
    const d = X[0].length, N = X.length, mu = zeros(d), sd = zeros(d);
    for (const r of X) for (let j = 0; j < d; j++) mu[j] += r[j] / N;
    for (const r of X) for (let j = 0; j < d; j++) sd[j] += (r[j] - mu[j]) * (r[j] - mu[j]) / N;
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    return { X: X.map(r => r.map((v, j) => (v - mu[j]) / sd[j])), mu: mu, sd: sd };
  }

  /* A logistic classifier fitted by full-batch gradient descent on the
     cross-entropy, with an optional L2 penalty. Promoted from
     neural-networks.viz.js by part 2, which needed the identical routine for
     the empirical-risk figure: the POINT of both figures is that the fitting
     procedure is held fixed while something else varies, which only works if
     it is literally the same code. X is (N × d), y a 0/1 array.              */
  function logistic(X, y, steps, lr, l2) {
    const d = X[0].length, N = X.length;
    const w = zeros(d);
    let b = 0;
    const S = steps || 400, LR = lr === undefined ? 0.5 : lr, L2 = l2 === undefined ? 1e-3 : l2;
    for (let t = 0; t < S; t++) {
      const gw = zeros(d);
      let gb = 0;
      for (let i = 0; i < N; i++) {
        const e = sigmoid(dot(w, X[i]) + b) - y[i];
        for (let j = 0; j < d; j++) gw[j] += e * X[i][j];
        gb += e;
      }
      for (let j = 0; j < d; j++) w[j] -= LR * (gw[j] / N + L2 * w[j]);
      b -= LR * gb / N;
    }
    return { w: w, b: b };
  }
  const logit = (model, x) => dot(model.w, x) + model.b;
  function accuracy(model, X, y) {
    let n = 0;
    for (let i = 0; i < X.length; i++) if ((logit(model, X[i]) >= 0 ? 1 : 0) === y[i]) n++;
    return n / X.length;
  }
  /* mean binary cross-entropy of a linear model, from the LOGIT — the stable
     form, so a perfectly separating fit reports a large finite number and not
     an infinity.                                                             */
  function logisticLoss(model, X, y) {
    let s = 0;
    for (let i = 0; i < X.length; i++) s += bce(logit(model, X[i]), y[i]);
    return s / X.length;
  }

  /* ══ spectra ════════════════════════════════════════════════════════════
     eig2sym: the EXACT eigendecomposition of a symmetric 2×2. Every
     conditioning claim in the series is checked against this rather than
     asserted, and a 2×2 has a closed form so there is no excuse not to.
       [a b; b c] → {lo, hi, vlo, vhi, cond}                                  */
  function eig2sym(a, b, c) {
    const tr = a + c, det = a * c - b * b;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const hi = tr / 2 + disc, lo = tr / 2 - disc;
    let vhi, vlo;
    if (Math.abs(b) > 1e-14) {
      vhi = [hi - c, b]; vlo = [lo - c, b];
    } else {                                    // already diagonal
      vhi = (a >= c) ? [1, 0] : [0, 1];
      vlo = (a >= c) ? [0, 1] : [1, 0];
    }
    const nz = v => { const n = Math.hypot(v[0], v[1]) || 1; return [v[0] / n, v[1] / n]; };
    return { lo: lo, hi: hi, vlo: nz(vlo), vhi: nz(vhi), cond: (lo === 0 ? Infinity : hi / lo) };
  }
  /* top |eigenvalue| of a square matrix by power iteration — the spectral
     norm of a SYMMETRIC matrix, and for a general one the top |λ|, which is
     NOT the same as ‖A‖₂ (that is σ_max). Used by the Jacobian-product
     figures, which say which of the two they mean.                          */
  function powerIter(A, iters, seed) {
    const n = A.length, r = rng(seed || 5);
    let v = new Array(n).fill(0).map(() => randn(r));
    let lam = 0;
    for (let t = 0; t < (iters || 200); t++) {
      const w = matvec(A, v);
      const nw = Math.hypot.apply(null, w) || 1;
      v = w.map(x => x / nw);
      lam = dot(v, matvec(A, v));
    }
    return { lambda: lam, v: v };
  }
  /* largest singular value: power iteration on AᵀA. ‖A‖₂ = max singular
     value = the largest factor by which A can stretch ANY vector, which is
     the quantity that governs whether a product of Jacobians explodes.       */
  function specNorm(A, iters, seed) {
    const AtA = matmul(transpose(A), A);
    return Math.sqrt(Math.max(0, powerIter(AtA, iters || 200, seed).lambda));
  }

  /* ══ exponentially weighted moving average ══════════════════════════════
     v_t = β·v_{t−1} + (1−β)·θ_t, v₀ = 0, with the OPTIONAL bias correction
     v̂_t = v_t / (1 − βᵗ). Returns both tracks so a figure can draw the gap.
     This is the mechanism every adaptive optimiser below reuses.             */
  function ewma(series, beta, correct) {
    let v = 0;
    const raw = [], cor = [];
    for (let t = 1; t <= series.length; t++) {
      v = beta * v + (1 - beta) * series[t - 1];
      raw.push(v);
      cor.push(v / (1 - Math.pow(beta, t)));
    }
    return correct ? cor : raw;
  }

  /* ══ the optimisers ═════════════════════════════════════════════════════
     Promoted into the shared library by part 2 so that every later part —
     the residual page's warmup demo, the transformer page's schedule — uses
     the SAME update rules rather than a re-typed variant.

     Contract. Each entry is
        {key, label, hp: [names], init(n) → state, step(state, g, hp) → dx}
     where g is a flat array of gradients, dx is the flat array to ADD to the
     parameters (so the caller does θ ← θ + dx and never has to remember a
     sign), and state.t is the 1-based step counter, incremented by step().

     Two conventions collide in the literature for momentum, and they are
     DIFFERENT ALGORITHMS with the same name:
        accumulation form   v ← βv + g          dx = −ηv
        EWMA form           v ← βv + (1−β)g     dx = −ηv
     At steady state under a constant gradient the first gives v = g/(1−β)
     and the second gives v = g: the accumulation form multiplies the
     effective step by 1/(1−β), a factor of TEN at β = 0.9. Both are provided
     ("momentum" is the accumulation form, matching the common framework
     default; "momentumEwma" is the other) because the page teaches the gap.
     Adam's first moment is the EWMA form, with the bias correction supplying
     the same normalisation that (1−β) does here.                             */
  const OPT = {
    sgd: {
      key: "sgd", label: "SGD", hp: ["lr"],
      init: n => ({ t: 0, n: n }),
      step: (s, g, h) => { s.t++; return g.map(gi => -h.lr * gi); }
    },
    momentum: {
      key: "momentum", label: "momentum (accumulation form)", hp: ["lr", "beta"],
      init: n => ({ t: 0, n: n, v: zeros(n) }),
      step: (s, g, h) => {
        s.t++;
        for (let i = 0; i < g.length; i++) s.v[i] = h.beta * s.v[i] + g[i];
        return s.v.map(v => -h.lr * v);
      }
    },
    momentumEwma: {
      key: "momentumEwma", label: "momentum (EWMA form)", hp: ["lr", "beta"],
      init: n => ({ t: 0, n: n, v: zeros(n) }),
      step: (s, g, h) => {
        s.t++;
        for (let i = 0; i < g.length; i++) s.v[i] = h.beta * s.v[i] + (1 - h.beta) * g[i];
        return s.v.map(v => -h.lr * v);
      }
    },
    /* Nesterov in the "lookahead already applied" rewriting used by the
       frameworks: the gradient handed in is evaluated at the CURRENT point
       and the extrapolation is folded into the update, which is algebraically
       the same trajectory as evaluating ∇J(θ + βv) and stepping by v.        */
    nesterov: {
      key: "nesterov", label: "Nesterov momentum", hp: ["lr", "beta"],
      init: n => ({ t: 0, n: n, v: zeros(n) }),
      step: (s, g, h) => {
        s.t++;
        const dx = [];
        for (let i = 0; i < g.length; i++) {
          const vPrev = s.v[i];
          s.v[i] = h.beta * vPrev + g[i];
          dx.push(-h.lr * (g[i] + h.beta * s.v[i]));
        }
        return dx;
      }
    },
    adagrad: {
      key: "adagrad", label: "AdaGrad", hp: ["lr", "eps"],
      init: n => ({ t: 0, n: n, r: zeros(n) }),
      step: (s, g, h) => {
        s.t++;
        const e = h.eps === undefined ? 1e-8 : h.eps, dx = [];
        for (let i = 0; i < g.length; i++) {
          s.r[i] += g[i] * g[i];
          dx.push(-h.lr * g[i] / (Math.sqrt(s.r[i]) + e));
        }
        return dx;
      }
    },
    rmsprop: {
      key: "rmsprop", label: "RMSProp", hp: ["lr", "beta2", "eps"],
      init: n => ({ t: 0, n: n, r: zeros(n) }),
      step: (s, g, h) => {
        s.t++;
        const b2 = h.beta2 === undefined ? 0.999 : h.beta2, e = h.eps === undefined ? 1e-8 : h.eps, dx = [];
        for (let i = 0; i < g.length; i++) {
          s.r[i] = b2 * s.r[i] + (1 - b2) * g[i] * g[i];
          dx.push(-h.lr * g[i] / (Math.sqrt(s.r[i]) + e));
        }
        return dx;
      }
    },
    adam: {
      key: "adam", label: "Adam", hp: ["lr", "beta1", "beta2", "eps", "correct", "l2"],
      init: n => ({ t: 0, n: n, m: zeros(n), v: zeros(n) }),
      step: (s, g, h) => {
        s.t++;
        const b1 = h.beta1 === undefined ? 0.9 : h.beta1,
          b2 = h.beta2 === undefined ? 0.999 : h.beta2,
          e = h.eps === undefined ? 1e-8 : h.eps,
          on = (h.correct === undefined) ? true : !!h.correct,
          c1 = on ? (1 - Math.pow(b1, s.t)) : 1,
          c2 = on ? (1 - Math.pow(b2, s.t)) : 1, dx = [];
        for (let i = 0; i < g.length; i++) {
          /* COUPLED L2: the penalty enters the gradient, so it is scaled by
             1/√v̂ along with everything else. That is the behaviour AdamW
             exists to undo; see the "adamw" entry.                           */
          const gi = g[i] + (h.l2 ? h.l2 * (h.theta ? h.theta[i] : 0) : 0);
          s.m[i] = b1 * s.m[i] + (1 - b1) * gi;
          s.v[i] = b2 * s.v[i] + (1 - b2) * gi * gi;
          dx.push(-h.lr * (s.m[i] / c1) / (Math.sqrt(s.v[i] / c2) + e));
        }
        return dx;
      }
    },
    /* DECOUPLED weight decay. The decay term is subtracted from the
       parameters directly and never passes through the 1/√v̂ rescaling, so
       every coordinate decays at the same rate λ regardless of its gradient
       history. h.theta must be supplied — the decay is a function of the
       CURRENT parameters, which is exactly what makes it not a gradient.     */
    adamw: {
      key: "adamw", label: "AdamW (decoupled decay)", hp: ["lr", "beta1", "beta2", "eps", "wd"],
      init: n => ({ t: 0, n: n, m: zeros(n), v: zeros(n) }),
      step: (s, g, h) => {
        s.t++;
        const b1 = h.beta1 === undefined ? 0.9 : h.beta1,
          b2 = h.beta2 === undefined ? 0.999 : h.beta2,
          e = h.eps === undefined ? 1e-8 : h.eps,
          c1 = 1 - Math.pow(b1, s.t), c2 = 1 - Math.pow(b2, s.t),
          wd = h.wd || 0, th = h.theta, dx = [];
        for (let i = 0; i < g.length; i++) {
          s.m[i] = b1 * s.m[i] + (1 - b1) * g[i];
          s.v[i] = b2 * s.v[i] + (1 - b2) * g[i] * g[i];
          dx.push(-h.lr * ((s.m[i] / c1) / (Math.sqrt(s.v[i] / c2) + e) + wd * (th ? th[i] : 0)));
        }
        return dx;
      }
    }
  };
  const optList = () => ["sgd", "momentum", "momentumEwma", "nesterov", "adagrad", "rmsprop", "adam", "adamw"];

  /* ══ learning-rate schedules ════════════════════════════════════════════
     lrAt(kind, t, o) with t ZERO-BASED and o = {peak, total, warm, floor,
     step (period), gamma, power}. Warmup is LINEAR from 0 to peak over the
     first o.warm steps and is applied to every kind, because that is how it
     is used in practice: warmup is a prefix, not a schedule of its own.      */
  function lrAt(kind, t, o) {
    const p = o.peak === undefined ? 1 : o.peak, T = o.total || 1000,
      w = o.warm || 0, fl = o.floor === undefined ? 0 : o.floor;
    if (w > 0 && t < w) return p * (t / w);
    const u = clamp((t - w) / Math.max(1, T - w), 0, 1);       // progress after warmup
    switch (kind) {
      case "constant": return p;
      case "cosine": return fl + (p - fl) * 0.5 * (1 + Math.cos(Math.PI * u));
      case "linear": return fl + (p - fl) * (1 - u);
      case "step": {
        const per = o.step || Math.max(1, Math.floor(T / 4)), gm = o.gamma === undefined ? 0.1 : o.gamma;
        return p * Math.pow(gm, Math.floor((t - w) / per));
      }
      case "exp": {
        const per = o.step || Math.max(1, Math.floor(T / 4)), gm = o.gamma === undefined ? 0.1 : o.gamma;
        return p * Math.pow(gm, (t - w) / per);
      }
      case "power": {            // η₀ / (1 + t/s)^c
        const sN = o.step || Math.max(1, Math.floor(T / 10)), c = o.power === undefined ? 1 : o.power;
        return p / Math.pow(1 + (t - w) / sN, c);
      }
      case "invsqrt": {          // the transformer schedule: peak at t = w, then t^(−1/2)
        const w1 = Math.max(w, 1);
        return p * Math.sqrt(w1 / Math.max(t, w1));
      }
      case "onecycle": {         // up to the peak at 30 %, cosine down, then a tail
        const a = 0.3;
        return u < a ? fl + (p - fl) * (u / a)
          : fl * 0.04 + (p - fl * 0.04) * 0.5 * (1 + Math.cos(Math.PI * (u - a) / (1 - a)));
      }
      default: return p;
    }
  }

  /* ══ normalisation layers ═══════════════════════════════════════════════
     Row convention throughout: X is (N × d), one example per ROW.
     batchNorm normalises DOWN a column (across the batch, per feature);
     layerNorm and rmsNorm normalise ACROSS a row (per example, over the
     features) and therefore do not depend on the batch at all — which is the
     whole reason sequence models use them.                                   */
  function batchNorm(X, gamma, beta, eps, stats) {
    const N = X.length, d = X[0].length, e = eps === undefined ? 1e-5 : eps;
    const mu = zeros(d), va = zeros(d);
    if (stats) {                                   // EVAL mode: fixed statistics
      for (let j = 0; j < d; j++) { mu[j] = stats.mu[j]; va[j] = stats.va[j]; }
    } else {                                       // TRAIN mode: this batch
      for (let j = 0; j < d; j++) {
        for (let i = 0; i < N; i++) mu[j] += X[i][j] / N;
        for (let i = 0; i < N; i++) va[j] += (X[i][j] - mu[j]) * (X[i][j] - mu[j]) / N;
      }
    }
    const Y = X.map(r => r.map((v, j) =>
      (gamma ? gamma[j] : 1) * (v - mu[j]) / Math.sqrt(va[j] + e) + (beta ? beta[j] : 0)));
    /* the UNBIASED (1/(N−1)) variance is what goes into the running estimate,
       while the BIASED (1/N) one above is what normalises the batch — the two
       differ by N/(N−1) and frameworks really do use different ones here.     */
    const vaU = zeros(d);
    if (!stats) for (let j = 0; j < d; j++)
      for (let i = 0; i < N; i++) vaU[j] += (X[i][j] - mu[j]) * (X[i][j] - mu[j]) / Math.max(1, N - 1);
    return { Y: Y, mu: mu, va: va, vaUnbiased: vaU };
  }
  function layerNorm(X, gamma, beta, eps) {
    const e = eps === undefined ? 1e-5 : eps;
    return X.map(r => {
      const m = mean(r), v = variance(r);
      return r.map((x, j) => (gamma ? gamma[j] : 1) * (x - m) / Math.sqrt(v + e) + (beta ? beta[j] : 0));
    });
  }
  function rmsNorm(X, gamma, eps) {
    const e = eps === undefined ? 1e-5 : eps;
    return X.map(r => {
      const ms = r.reduce((s, x) => s + x * x, 0) / r.length;
      const inv = 1 / Math.sqrt(ms + e);
      return r.map((x, j) => (gamma ? gamma[j] : 1) * x * inv);
    });
  }

  /* ══ gradient clipping ══════════════════════════════════════════════════ */
  const clipValue = (g, c) => g.map(x => clamp(x, -c, c));
  function clipNorm(g, c) {                        // preserves DIRECTION exactly
    const n = Math.sqrt(g.reduce((s, x) => s + x * x, 0));
    const f = (n > c) ? c / n : 1;
    return { g: g.map(x => x * f), norm: n, scaled: f < 1, factor: f };
  }

  /* ══ drawing ════════════════════════════════════════════════════════════
     Same contract as the vision series: the <svg> already exists in the HTML
     with its viewBox / role / aria-label; these only append into it.         */
  function frame(sel, W, H, m) {
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
      .attr("font-size", 11).attr("fill", DC.muted).text(label);
    return ax;
  }
  function axisL(g, y, ticks, label, fmtFn) {
    const ax = g.append("g").attr("class", "axis")
      .call(fmtFn ? d3.axisLeft(y).ticks(ticks || 5).tickFormat(fmtFn) : d3.axisLeft(y).ticks(ticks || 5));
    if (label) g.append("text").attr("x", 0).attr("y", -6).attr("text-anchor", "start")
      .attr("font-size", 11).attr("fill", DC.muted).text(label);
    return ax;
  }
  function gridY(g, y, iw, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(ticks || 5)).join("line")
      .attr("x1", 0).attr("x2", iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", DC.grid);
  }
  function gridX(g, x, ih, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(x.ticks(ticks || 5)).join("line")
      .attr("y1", 0).attr("y2", ih).attr("x1", d => x(d)).attr("x2", d => x(d)).attr("stroke", DC.grid);
  }
  function legend(g, items, x, y, opts) {
    const o = Object.assign({ gap: 15, size: 9, font: 10.5, vertical: true, step: 96 }, opts || {});
    const gl = g.append("g").attr("transform", `translate(${x},${y})`);
    items.forEach((it, i) => {
      const dx = o.vertical ? 0 : i * o.step, dy = o.vertical ? i * o.gap : 0;
      if (it.dash) {
        gl.append("line").attr("x1", dx).attr("x2", dx + o.size).attr("y1", dy).attr("y2", dy)
          .attr("stroke", it.color).attr("stroke-width", 2).attr("stroke-dasharray", it.dash);
      } else {
        gl.append("rect").attr("x", dx).attr("y", dy - o.size / 2).attr("width", o.size)
          .attr("height", o.size).attr("rx", 2).attr("fill", it.color)
          .attr("fill-opacity", it.op === undefined ? 0.9 : it.op);
      }
      gl.append("text").attr("x", dx + o.size + 6).attr("y", dy + 3.5)
        .attr("font-size", o.font).attr("fill", DC.muted).text(it.label);
    });
    return gl;
  }
  function panelBox(g, x, y, w, h, title, opt) {
    const o = Object.assign({ fill: "none", stroke: DC.line }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.append("rect").attr("x", -0.5).attr("y", -0.5).attr("width", w + 1).attr("height", h + 1)
      .attr("fill", o.fill).attr("stroke", o.stroke).attr("rx", 3);
    if (title) gg.append("text").attr("x", 0).attr("y", -7).attr("font-size", 11)
      .attr("fill", DC.ink).attr("font-weight", 600).text(title);
    return gg;
  }
  function kv(g, x, y, opt) {
    const o = Object.assign({ lead: 15, keyW: 152, size: 11, dp: 3 }, opt || {});
    let i = 0;
    return function (k, v, color, bold) {
      g.append("text").attr("x", x).attr("y", y + i * o.lead).attr("font-size", o.size)
        .attr("fill", DC.muted).text(k);
      g.append("text").attr("x", x + o.keyW).attr("y", y + i * o.lead).attr("font-size", o.size)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", color || DC.ink)
        .attr("font-weight", bold ? 600 : 400).text(v);
      i++;
    };
  }
  function matText(g, M, x, y, opt) {
    const o = Object.assign({ size: 10, dp: 2, fill: DC.ink, lead: 12.5, label: null, pad: 6, colorOf: null }, opt || {});
    const rows = M.map(r => r.map(v =>
      (typeof v === "string" ? v : (Math.abs(v) < 5e-7 ? "0" : v.toFixed(o.dp))).padStart(o.pad)).join(" "));
    const brL = ["⎡", "⎢", "⎣"], brR = ["⎤", "⎥", "⎦"];
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    if (o.label) gg.append("text").attr("x", 0).attr("y", -o.lead).attr("font-size", 10)
      .attr("fill", DC.muted).text(o.label);
    rows.forEach((r, i) => {
      const k = rows.length === 1 ? -1 : (i === 0 ? 0 : (i === rows.length - 1 ? 2 : 1));
      const bl = k < 0 ? "[" : brL[k], br = k < 0 ? "]" : brR[k];
      gg.append("text").attr("x", 0).attr("y", i * o.lead)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("font-size", o.size)
        .attr("xml:space", "preserve")
        .attr("fill", o.colorOf ? o.colorOf(i) : o.fill).text(bl + r + " " + br);
    });
    return gg;
  }
  function arrow(g, x1, y1, x2, y2, opt) {
    const o = Object.assign({ color: DC.a2, w: 1.6, head: 6, dash: null, op: 1 }, opt || {});
    const el = g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
      .attr("stroke", o.color).attr("stroke-width", o.w).attr("stroke-opacity", o.op);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    const a = Math.atan2(y2 - y1, x2 - x1), h = o.head;
    g.append("path").attr("d", `M${x2},${y2} L${x2 - h * Math.cos(a - 0.4)},${y2 - h * Math.sin(a - 0.4)} L${x2 - h * Math.cos(a + 0.4)},${y2 - h * Math.sin(a + 0.4)} Z`)
      .attr("fill", o.color).attr("fill-opacity", o.op);
    return el;
  }
  function curve(g, pts, opt) {
    const o = Object.assign({ stroke: DC.accent, w: 1.8, dash: null, op: 1, fill: "none" }, opt || {});
    const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(2) + "," + p[1].toFixed(2)).join(" ");
    const el = g.append("path").attr("d", d + (o.fill === "none" ? "" : " Z"))
      .attr("fill", o.fill).attr("fill-opacity", o.fill === "none" ? 0 : (o.fillOp === undefined ? 0.18 : o.fillOp))
      .attr("stroke", o.stroke).attr("stroke-width", o.w).attr("stroke-opacity", o.op)
      .attr("stroke-linejoin", "round");
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    return el;
  }
  /* A rectangular clip. The rect is in the USER SPACE OF THE REFERENCING
     ELEMENT, not page space: inside an already-translated <g> it starts at
     (0, 0). Getting this wrong clips the whole drawing away silently.        */
  function clip(svg, id, x, y, w, h) {
    svg.append("defs").append("clipPath").attr("id", id)
      .append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h);
    return "url(#" + id + ")";
  }
  const CANV = (typeof document !== "undefined" && document.createElement)
    ? document.createElement("canvas") : null;
  function cells(g, x0, y0, cw, W, H, colorOf) {
    const gg = g.append("g").attr("transform", `translate(${x0},${y0})`);
    const ctx = (CANV && W * H > 2500 && CANV.getContext) ? CANV.getContext("2d") : null;
    if (ctx) {
      CANV.width = W; CANV.height = H;
      ctx.clearRect(0, 0, W, H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const c = colorOf(x, y);
        if (!c || c === "none") continue;
        ctx.fillStyle = c; ctx.fillRect(x, y, 1, 1);
      }
      gg.append("image").attr("x", 0).attr("y", 0)
        .attr("width", W * cw).attr("height", H * cw)
        .attr("preserveAspectRatio", "none").attr("image-rendering", "pixelated")
        .attr("href", CANV.toDataURL());
      return gg;
    }
    const data = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data.push([x, y]);
    gg.selectAll("rect").data(data).join("rect")
      .attr("x", d => d[0] * cw).attr("y", d => d[1] * cw)
      .attr("width", cw + 0.4).attr("height", cw + 0.4)
      .attr("shape-rendering", "crispEdges")
      .attr("fill", d => colorOf(d[0], d[1]) || "none")
      .attr("fill-opacity", d => { const c = colorOf(d[0], d[1]); return (!c || c === "none") ? 0 : 1; });
    return gg;
  }
  /* The layer-and-edge picture every part of this series draws at least once.
     sizes = [d_in, …, d_out]; returns {nodes, cols} in the given box so the
     caller can annotate particular units.                                    */
  function netDiagram(g, sizes, opt) {
    const o = Object.assign({
      x: 0, y: 0, w: 300, h: 200, r: 7, maxShow: 8, labels: null,
      edgeOp: 0.35, edgeColor: DC.line, nodeFill: DC.panel2,
      nodeStroke: [DC.accent, DC.a2, DC.good], showEdges: true, edgeWidth: 1
    }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    const cols = sizes.map((n, l) => {
      const shown = Math.min(n, o.maxShow);
      const gapx = sizes.length > 1 ? o.w / (sizes.length - 1) : 0;
      const cx = l * gapx;
      const step = o.h / (shown + 1);
      const nodes = [];
      for (let i = 0; i < shown; i++) nodes.push({ x: cx, y: step * (i + 1), i: i, l: l });
      return { cx: cx, n: n, shown: shown, nodes: nodes, truncated: n > o.maxShow };
    });
    if (o.showEdges) {
      for (let l = 0; l + 1 < cols.length; l++) {
        const E = [];
        cols[l].nodes.forEach(a => cols[l + 1].nodes.forEach(b => E.push([a, b])));
        gg.append("g").selectAll("line").data(E).join("line")
          .attr("x1", d => d[0].x).attr("y1", d => d[0].y)
          .attr("x2", d => d[1].x).attr("y2", d => d[1].y)
          .attr("stroke", typeof o.edgeColor === "function" ? (d => o.edgeColor(d[0], d[1])) : o.edgeColor)
          .attr("stroke-width", o.edgeWidth).attr("stroke-opacity", o.edgeOp);
      }
    }
    cols.forEach((c, l) => {
      const col = o.nodeStroke[Math.min(l, o.nodeStroke.length - 1)];
      const stroke = (l === 0) ? o.nodeStroke[0] : (l === cols.length - 1 ? o.nodeStroke[o.nodeStroke.length - 1] : (o.nodeStroke[1] || col));
      gg.append("g").selectAll("circle").data(c.nodes).join("circle")
        .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", o.r)
        .attr("fill", o.nodeFill).attr("stroke", stroke).attr("stroke-width", 1.4);
      if (c.truncated) gg.append("text").attr("x", c.cx).attr("y", o.h - 2)
        .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", DC.muted).text("⋮");
      if (o.labels) gg.append("text").attr("x", c.cx).attr("y", -10)
        .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", DC.muted)
        .text(o.labels[l] === undefined ? "" : o.labels[l]);
    });
    return { g: gg, cols: cols };
  }

  /* ══ 2-D convolution AS A LAYER ═════════════════════════════════════════
     [added by part 3 — Convolutional Networks]

     ── SCOPE BOUNDARY, stated once ──────────────────────────────────────
     CLASSICAL IMAGE FILTERING is not owned here. Correlation against
     convolution, separable kernels, the Gaussian / box / binomial families
     and the five border modes zero | clamp | wrap | mirror | symm belong to
     the Computer Vision series, and their canonical implementation is
     vision/vision-viz.js (corr2, conv2, sepH, sepV, sep2, gauss1, box,
     binomial, flip2), verified there against scipy.ndimage to 0.000e+00 on
     every one of those modes. A deep-learning page sits in a sibling folder
     and cannot load that file, so the LAYER arithmetic a convnet needs lives
     here. These are NOT a fork of it, because they compute a different
     thing:

       vision/vision-viz.js corr2 → ALWAYS a same-size map. Odd kernel,
         centred, radius (f−1)>>1, any of five border modes. It is a
         FILTERING primitive: the output grid is the input grid.
       DL.corr2d / DL.conv2dMC below → the LAYER output grid
         ⌊(n + p₀ + p₁ − (d(f−1)+1)) / s⌋ + 1, with explicit and possibly
         ASYMMETRIC ZERO padding, a stride, a dilation, and EVEN kernels
         allowed. Zero padding only — a convolution layer has no other
         border mode, which is itself a fact the page teaches.

     On their overlap — odd f, s = 1, d = 1, p₀ = p₁ = (f−1)/2, mode "zero" —
     the two must agree ELEMENTWISE. The build checks exactly that: max |Δ|
     = 0.000e+00 over 240 random inputs and every kernel from 1×1 to 7×7. If
     either convention is ever changed, that check is the tripwire. Nothing
     here should ever grow a border mode; a DL page that wants one is asking
     a vision question.

     Tensor layout: CHW, i.e. V[channel][row][col] and K[out][in][row][col] —
     NCHW with the per-example batch axis dropped. The consuming page
     declares NCHW at the top and never departs from it.
     One padding pair (p₀ before, p₁ after) is used on BOTH axes, as is one
     stride and one dilation; every geometry this series teaches is
     axis-symmetric in that sense.                                          */

  /* the output length along one axis. o = {s, p0, p1, d} */
  const effK = (f, d) => ((d === undefined ? 1 : d) || 1) * (f - 1) + 1;
  function outSize(n, f, o) {
    const oo = o || {};
    const s = oo.s || 1, d = oo.d || 1, p0 = oo.p0 || 0, p1 = oo.p1 || 0;
    return Math.floor((n + p0 + p1 - effK(f, d)) / s) + 1;
  }

  /* The three named padding policies resolved to an EXPLICIT (p₀, p₁) pair.
     "same" means ⌈n/s⌉ outputs — the length the input would have had if it
     were itself strided; at s = 1 that is n, the familiar case.
     `symmetric` is false when the required total is odd, which is exactly
     the even-kernel trap: the pair returned is then lopsided, which is what
     a framework's string policy does, and which an integer `padding=p`
     argument cannot express at all.                                        */
  function padFor(mode, n, f, s, d) {
    const S = s || 1, D = d || 1, fe = effK(f, D);
    if (mode === "valid") return { p0: 0, p1: 0, total: 0, symmetric: true };
    if (mode === "full") return { p0: fe - 1, p1: fe - 1, total: 2 * (fe - 1), symmetric: true };
    const out = Math.ceil(n / S);
    const total = Math.max(0, (out - 1) * S + fe - n);
    const p0 = Math.floor(total / 2);
    return { p0: p0, p1: total - p0, total: total, symmetric: (total % 2 === 0) };
  }

  const flipK = K => K.map(r => r.slice()).reverse().map(r => r.reverse());

  /* single-channel 2-D CORRELATION with layer geometry.
     o = {s, p0, p1, d, flip}; flip:true makes it a true CONVOLUTION.
     Taps that fall outside the input read 0 — implicit zero padding.       */
  function corr2d(X, K, o) {
    const oo = Object.assign({ s: 1, p0: 0, p1: 0, d: 1, flip: false }, o || {});
    const H = X.length, W = X[0].length;
    const KK = oo.flip ? flipK(K) : K;
    const kh = KK.length, kw = KK[0].length;
    const Ho = Math.max(0, outSize(H, kh, oo)), Wo = Math.max(0, outSize(W, kw, oo));
    const Y = zeros2(Ho, Wo);
    for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++) {
      let acc = 0;
      for (let m = 0; m < kh; m++) {
        const i = p * oo.s + m * oo.d - oo.p0;
        if (i < 0 || i >= H) continue;
        const Xi = X[i], Km = KK[m];
        for (let n = 0; n < kw; n++) {
          const j = q * oo.s + n * oo.d - oo.p0;
          if (j < 0 || j >= W) continue;
          acc += Xi[j] * Km[n];
        }
      }
      Y[p][q] = acc;
    }
    return Y;
  }
  const conv2dTrue = (X, K, o) => corr2d(X, K, Object.assign({}, o || {}, { flip: true }));

  /* zeros3(C, H, W) — a C-deep stack of H×W maps */
  function zeros3(C, H, W) { const A = []; for (let c = 0; c < C; c++) A.push(zeros2(H, W)); return A; }

  /* MULTI-CHANNEL forward pass, the thing a convolution LAYER actually is.
       V  (Cᵢₙ × H × W)                  input volume
       K  (Cₒᵤₜ × Cᵢₙ/g × kh × kw)       one 3-D filter per output channel
       b  (Cₒᵤₜ) or null
       o  {s, p0, p1, d, groups}
     → Z  (Cₒᵤₜ × H′ × W′).  CORRELATION, not flipped: this is what every
     framework calls "conv2d" and the page says so out loud.                */
  function conv2dMC(V, K, b, o) {
    const oo = Object.assign({ s: 1, p0: 0, p1: 0, d: 1, groups: 1 }, o || {});
    const Cin = V.length, H = V[0].length, W = V[0][0].length;
    const Cout = K.length, kin = K[0].length, kh = K[0][0].length, kw = K[0][0][0].length;
    const g = oo.groups, cinG = Cin / g, coutG = Cout / g;
    const Ho = Math.max(0, outSize(H, kh, oo)), Wo = Math.max(0, outSize(W, kw, oo));
    const Z = zeros3(Cout, Ho, Wo);
    for (let co = 0; co < Cout; co++) {
      const grp = Math.floor(co / coutG), base = grp * cinG;
      const bias = b ? b[co] : 0;
      for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++) {
        let acc = bias;
        for (let ci = 0; ci < kin; ci++) {
          const Vc = V[base + ci], Kc = K[co][ci];
          for (let m = 0; m < kh; m++) {
            const i = p * oo.s + m * oo.d - oo.p0;
            if (i < 0 || i >= H) continue;
            const Vi = Vc[i], Km = Kc[m];
            for (let n = 0; n < kw; n++) {
              const j = q * oo.s + n * oo.d - oo.p0;
              if (j < 0 || j >= W) continue;
              acc += Vi[j] * Km[n];
            }
          }
        }
        Z[co][p][q] = acc;
      }
    }
    return Z;
  }

  /* THE TWO BACKWARD OPERATIONS. Both are written as a transposition of the
     SAME triple loop as conv2dMC — the (co, p, q, ci, m, n) → (i, j) index
     map is written out once above and reused verbatim, so correctness is
     structural rather than a re-derivation. The page then audits both
     against central differences anyway.
       convGradK(V, G, o, shape) = ∂L/∂K, a CORRELATION of the input with
         the output gradient;
       convGradV(K, G, o, H, W)  = ∂L/∂V, a scatter which for stride 1 and
         no dilation is a FULL CONVOLUTION of G with the FLIPPED kernel;
       convGradB(G)              = ∂L/∂b, summed over both spatial axes.    */
  function convGradK(V, G, o, shape) {
    const oo = Object.assign({ s: 1, p0: 0, p1: 0, d: 1, groups: 1 }, o || {});
    const Cin = V.length, H = V[0].length, W = V[0][0].length;
    const Cout = G.length, Ho = G[0].length, Wo = G[0][0].length;
    const kh = shape.kh, kw = shape.kw, g = oo.groups;
    const cinG = Cin / g, coutG = Cout / g, kin = shape.kin === undefined ? cinG : shape.kin;
    const dK = [];
    for (let co = 0; co < Cout; co++) dK.push(zeros3(kin, kh, kw));
    for (let co = 0; co < Cout; co++) {
      const grp = Math.floor(co / coutG), base = grp * cinG;
      for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++) {
        const gv = G[co][p][q];
        if (gv === 0) continue;
        for (let ci = 0; ci < kin; ci++) {
          const Vc = V[base + ci];
          for (let m = 0; m < kh; m++) {
            const i = p * oo.s + m * oo.d - oo.p0;
            if (i < 0 || i >= H) continue;
            for (let n = 0; n < kw; n++) {
              const j = q * oo.s + n * oo.d - oo.p0;
              if (j < 0 || j >= W) continue;
              dK[co][ci][m][n] += gv * Vc[i][j];
            }
          }
        }
      }
    }
    return dK;
  }
  function convGradV(K, G, o, H, W) {
    const oo = Object.assign({ s: 1, p0: 0, p1: 0, d: 1, groups: 1 }, o || {});
    const Cout = K.length, kin = K[0].length, kh = K[0][0].length, kw = K[0][0][0].length;
    const Ho = G[0].length, Wo = G[0][0].length, g = oo.groups;
    const coutG = Cout / g, Cin = kin * g, cinG = kin;
    const dV = zeros3(Cin, H, W);
    for (let co = 0; co < Cout; co++) {
      const grp = Math.floor(co / coutG), base = grp * cinG;
      for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++) {
        const gv = G[co][p][q];
        if (gv === 0) continue;
        for (let ci = 0; ci < kin; ci++) {
          const Kc = K[co][ci], Dc = dV[base + ci];
          for (let m = 0; m < kh; m++) {
            const i = p * oo.s + m * oo.d - oo.p0;
            if (i < 0 || i >= H) continue;
            for (let n = 0; n < kw; n++) {
              const j = q * oo.s + n * oo.d - oo.p0;
              if (j < 0 || j >= W) continue;
              Dc[i][j] += gv * Kc[m][n];
            }
          }
        }
      }
    }
    return dV;
  }
  const convGradB = G => G.map(m => m.reduce((s, r) => s + r.reduce((t, v) => t + v, 0), 0));

  /* POOLING, channel by channel. o = {k, s, p0, p1, mode:"max"|"avg"}.
     Returns {Y, arg} where arg[c][p][q] = [i, j] is the position the max was
     taken from — the routing decision the page's critique of pooling needs
     to be able to draw. Pooling never mixes channels, so Cₒᵤₜ = Cᵢₙ and the
     layer has no parameters.                                               */
  function pool2d(V, o) {
    const oo = Object.assign({ k: 2, s: null, p0: 0, p1: 0, mode: "max" }, o || {});
    const s = oo.s === null ? oo.k : oo.s;
    const C = V.length, H = V[0].length, W = V[0][0].length;
    const geom = { s: s, p0: oo.p0, p1: oo.p1, d: 1 };
    const Ho = Math.max(0, outSize(H, oo.k, geom)), Wo = Math.max(0, outSize(W, oo.k, geom));
    const Y = zeros3(C, Ho, Wo), arg = [];
    for (let c = 0; c < C; c++) {
      const ac = []; arg.push(ac);
      for (let p = 0; p < Ho; p++) {
        const ap = []; ac.push(ap);
        for (let q = 0; q < Wo; q++) {
          let best = -Infinity, bi = -1, bj = -1, sum = 0, cnt = 0;
          for (let m = 0; m < oo.k; m++) {
            const i = p * s + m - oo.p0;
            if (i < 0 || i >= H) continue;
            for (let n = 0; n < oo.k; n++) {
              const j = q * s + n - oo.p0;
              if (j < 0 || j >= W) continue;
              const v = V[c][i][j];
              sum += v; cnt++;
              if (v > best) { best = v; bi = i; bj = j; }
            }
          }
          Y[c][p][q] = (oo.mode === "avg") ? (cnt ? sum / cnt : 0) : (cnt ? best : 0);
          ap.push([bi, bj]);
        }
      }
    }
    return { Y: Y, arg: arg, Ho: Ho, Wo: Wo };
  }
  /* global pooling: one number per channel */
  const globalPool = (V, mode) => V.map(M => {
    let s = 0, n = 0, mx = -Infinity;
    for (let i = 0; i < M.length; i++) for (let j = 0; j < M[0].length; j++) { s += M[i][j]; n++; if (M[i][j] > mx) mx = M[i][j]; }
    return mode === "max" ? mx : s / n;
  });

  /* ══ counting a convolution layer ═══════════════════════════════════════
     MACs, consistently with DL.macs: ONE multiply–accumulate, not two
     FLOPs. Papers differ by exactly that factor of two and the page says so.
     L = {cin, cout, k, kh, kw, s, p0, p1, d, groups, bias, H, W}.          */
  function convCount(L) {
    const kh = L.kh === undefined ? L.k : L.kh, kw = L.kw === undefined ? L.k : L.kw;
    const g = L.groups || 1, geom = { s: L.s || 1, p0: L.p0 || 0, p1: L.p1 === undefined ? (L.p0 || 0) : L.p1, d: L.d || 1 };
    const Ho = outSize(L.H, kh, geom), Wo = outSize(L.W, kw, geom);
    const perFilter = (L.cin / g) * kh * kw;
    const weights = L.cout * perFilter;
    const biases = (L.bias === false) ? 0 : L.cout;
    const macs = Ho * Wo * L.cout * perFilter;
    return {
      Ho: Ho, Wo: Wo, perFilter: perFilter, weights: weights, biases: biases,
      params: weights + biases, macs: macs, flops2: 2 * macs,
      acts: L.cout * Ho * Wo,
      /* the same map computed by a DENSE layer instead, for the contrast */
      denseParams: (L.cin * L.H * L.W) * (L.cout * Ho * Wo) + (L.bias === false ? 0 : L.cout * Ho * Wo),
      reuse: Ho * Wo                       // times each weight is used per example
    };
  }

  /* ══ receptive field of a stack ═════════════════════════════════════════
     layers = [{k, s, p0, d}] in order. Returns one record per layer with
       r      the receptive field, in input pixels, of ONE output unit
       j      the jump: the input distance between adjacent output units
       start  the input coordinate of the CENTRE of output unit 0
     r₀ = 1, j₀ = 1, start₀ = 0.5 (pixel centres at 0.5, 1.5, …), and
       jₗ = jₗ₋₁·sₗ
       rₗ = rₗ₋₁ + (fₑ,ₗ − 1)·jₗ₋₁
       startₗ = startₗ₋₁ + ((fₑ,ₗ − 1)/2 − p₀,ₗ)·jₗ₋₁
     The page does not take this on trust: it also MEASURES r by perturbing
     one input pixel and counting the outputs that move.                    */
  function rfChain(layers) {
    let r = 1, j = 1, start = 0.5;
    const out = [{ layer: -1, r: r, j: j, start: start }];
    layers.forEach((L, idx) => {
      const fe = effK(L.k, L.d || 1), s = L.s || 1, p0 = L.p0 || 0;
      r = r + (fe - 1) * j;
      start = start + ((fe - 1) / 2 - p0) * j;
      j = j * s;
      out.push({ layer: idx, r: r, j: j, start: start, fe: fe, s: s, p0: p0 });
    });
    return out;
  }

  /* ══ im2col ═════════════════════════════════════════════════════════════
     The lowering that turns a convolution into ONE matrix product, and the
     reason a convolution runs at dense-matmul speed. Returns the patch
     matrix in the SERIES ROW CONVENTION: (H′·W′ × Cᵢₙ·kh·kw), one PATCH per
     row, so that Z = P·Wcol with Wcol of shape (Cᵢₙ·kh·kw × Cₒᵤₜ) is
     literally the X·W of part 1. Column order within a row is
     (ci, m, n) with n fastest.                                             */
  function im2col(V, kh, kw, o) {
    const oo = Object.assign({ s: 1, p0: 0, p1: 0, d: 1 }, o || {});
    const Cin = V.length, H = V[0].length, W = V[0][0].length;
    const Ho = Math.max(0, outSize(H, kh, oo)), Wo = Math.max(0, outSize(W, kw, oo));
    const P = zeros2(Ho * Wo, Cin * kh * kw);
    for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++) {
      const row = P[p * Wo + q];
      let c = 0;
      for (let ci = 0; ci < Cin; ci++) for (let m = 0; m < kh; m++) for (let n = 0; n < kw; n++) {
        const i = p * oo.s + m * oo.d - oo.p0, j = q * oo.s + n * oo.d - oo.p0;
        row[c++] = (i < 0 || i >= H || j < 0 || j >= W) ? 0 : V[ci][i][j];
      }
    }
    return { P: P, Ho: Ho, Wo: Wo };
  }
  /* the matching (Cᵢₙ·kh·kw × Cₒᵤₜ) weight matrix, same column order */
  function ker2col(K) {
    const Cout = K.length, Cin = K[0].length, kh = K[0][0].length, kw = K[0][0][0].length;
    const Wc = zeros2(Cin * kh * kw, Cout);
    for (let co = 0; co < Cout; co++) {
      let c = 0;
      for (let ci = 0; ci < Cin; ci++) for (let m = 0; m < kh; m++) for (let n = 0; n < kw; n++) Wc[c++][co] = K[co][ci][m][n];
    }
    return Wc;
  }

  /* ══ the doubly block Toeplitz matrix ═══════════════════════════════════
     The (H′W′ × HW) matrix M with Z.flat = M · X.flat, for a single-channel
     layer. Built by running the layer on the HW indicator images, so it is
     the SAME code path as corr2d and cannot drift from it. Only ever called
     on tiny inputs — it costs O(HW · H′W′).                                */
  function convMatrix(H, W, K, o) {
    const kh = K.length, kw = K[0].length;
    const oo = Object.assign({ s: 1, p0: 0, p1: 0, d: 1 }, o || {});
    const Ho = Math.max(0, outSize(H, kh, oo)), Wo = Math.max(0, outSize(W, kw, oo));
    const M = zeros2(Ho * Wo, H * W);
    for (let a = 0; a < H * W; a++) {
      const E = zeros2(H, W); E[Math.floor(a / W)][a % W] = 1;
      const Y = corr2d(E, K, oo);
      for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++) M[p * Wo + q][a] = Y[p][q];
    }
    return { M: M, Ho: Ho, Wo: Wo };
  }

  return {
    clamp, lerp, linspace, fmt, sig, fmtE, big, commas,
    rng, randn, shuffle,
    zeros, zeros2, eye, transpose, matmul, matvec, vecmat, outer, addRow, dot, frob, applyEl,
    erf, Phi, phi,
    sigmoid, softplus, logSigmoid, logSumExp, softmax, logSoftmax, xent, bce, mse, naive,
    ACT, act, actList, maxout,
    mlpInit, forward, predict, loss, backward, numGrad, sgdStep, cloneNet,
    params, macs, binom, regionsShallow, regionsDeepLB,
    mean, variance, std, standardise, eig2sym, powerIter, specNorm,
    ewma, OPT, optList, lrAt, batchNorm, layerNorm, rmsNorm, clipValue, clipNorm,
    logistic, logit, accuracy, logisticLoss,
    frame, axisB, axisL, gridX, gridY, legend, panelBox, kv, matText, arrow, curve, clip,
    cells, netDiagram,
    /* [part 3 — CNNs] convolution as a layer; see the SCOPE BOUNDARY note above */
    effK, outSize, padFor, flipK, corr2d, conv2dTrue, zeros3, conv2dMC,
    convGradK, convGradV, convGradB, pool2d, globalPool, convCount, rfChain,
    im2col, ker2col, convMatrix
  };
})();
