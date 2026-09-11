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
     · spectra 2  DL.eigGeneral(A) → {re[], im[], rho} for a GENERAL real
                   square matrix, by complex shifted QR with deflation —
                   DL.powerIter names only the dominant REAL eigenvalue and
                   stalls on a conjugate pair, and a recurrent weight matrix
                   routinely has one. DL.specRad(A) = the spectral radius,
                   the number that decides whether a recurrent gradient
                   vanishes or explodes; DL.scaleToRho(A, r); DL.matPow(A, k).
                   DL.eigSym(A) → {w desc, Q} full symmetric eigendecomposition
                   (cyclic Jacobi); DL.pinvSym(A, rcond); DL.ridge(X, Y, λ).
                   Verified against numpy.linalg.eigvals to 6.5e−14 relative
                   on ρ over 24 matrices, including complex pairs, a defective
                   Jordan block and a companion matrix.            [part 4]
     · recurrence DL.CELL — a registry keyed by name: vanilla · gru · lstm.
                   Each entry is {key, label, gates, nGate, init(dₓ,dₕ,opt),
                   forward(p,X,s0) → st, backward(p,st,dH) → g, nParam}.
                   READ THE SEQUENCE LAYOUT comment above them before using
                   any of it: a sequence is TIME-MAJOR, (T × d), one time
                   step per ROW, and U (dₓ×dₕ), W (dₕ×dₕ), V (dₕ×d_y), so a
                   step is zₜ = hₜ₋₁W + xₜU + b. Column-convention sources
                   transpose all three.
                   DL.seqInit/seqForward/seqLoss/seqBPTT wrap a cell with the
                   shared output head; DL.seqBPTTtrunc(k) is the TEXTBOOK
                   truncation (each loss reaches back k steps) and
                   DL.seqBPTTchunk(k) is the CHUNKED form a framework runs
                   (state carried across a window boundary, gradient not).
                   They are different algorithms with different biases and
                   the series teaches the gap; never swap one for the other.
                   DL.seqNumGrad is the central-difference audit,
                   DL.stateJac(net, X, k) measures ∂h_T/∂h_k by perturbation
                   for ANY cell, DL.cellParams(kind, dₓ, dₕ, biasSets) is the
                   exact parameter count, DL.jacFD(f, x) a generic numeric
                   Jacobian.                                       [part 4]
     · attention   DL.softmaxRows(S) — row-wise, overflow-safe, and NaN on an
                   all-masked row ON PURPOSE; DL.softmaxJac(p) = diag(p) − ppᵀ
                   and DL.softmaxJacFrob(p); DL.entropyOf(p) in NATS and
                   DL.perplexityOf(p) = eᴴ, the effective number of keys.
                   DL.splitHeads(M, h) / DL.mergeHeads(T) — heads split the
                   LAST axis into ADJACENT blocks and this is the ONLY place
                   that reshape may happen. DL.causalMask, DL.padMask,
                   DL.windowMask(n, w, {causal, dil, sinks}), DL.addMasks —
                   every mask ADDITIVE, 0 or −Infinity, never a large finite
                   constant (DL.maskConst records why, per precision).
                   DL.sdpa(Q,K,V,{mask,scale}) → {S,A,Z}; DL.mhaInit /
                   DL.mhaForward / DL.mhaBackward — a bare attention block,
                   no residual, no norm, no FFN (those are part 6's), with g
                   KV heads so MHA, GQA and MQA are one code path
                   (DL.kvHeadOf); audited against central differences to
                   4.5e−09 relative. DL.numGradMats(loss, mats) is that audit.
                   DL.mhaParams, DL.attnMacs, DL.macCrossover, DL.kvCacheElems,
                   DL.decodeIntensity — counting, in MACs, matching DL.macs and
                   DL.convCount.  READ THE ATTENTION LAYOUT LAW above them.
                                                                     [part 5]
     · transformer DL.sinusoidalPE, DL.peShiftMatrix (PE(p+k) = PE(p)·M_k),
                   DL.ropeFreqs, DL.rope (rotate the ROWS of Q and K; never V;
                   pairing "adjacent" or "half" are DIFFERENT conventions),
                   DL.alibiSlopes, DL.alibiBias, DL.relBucket, DL.relBucketMatrix;
                   DL.lnForward/lnBackward, DL.rmsForward/rmsBackward (norms
                   with a cache and an exact backward, agreeing with
                   DL.layerNorm / DL.rmsNorm to 0); DL.ffnInit/ffnForward/
                   ffnBackward (ungated and gated), DL.ffnParams,
                   DL.ffnWidthGated (the ⅔ correction); DL.attnForward/
                   attnBackward (DL.mhaForward with optional RoPE and per-head
                   biases — identical to it when both are off); DL.blockInit/
                   blockForward/blockBackward (pre-LN and post-LN), DL.blockParams;
                   DL.tfInit/tfForward/tfLoss/tfBackward/tfParamList/tfGradList,
                   DL.optMats — a whole tiny decoder-only model for the live
                   figures; DL.modelParams, DL.modelFlops (the 6·N·D rule with
                   the quadratic term explicit, MACs and FLOPs named separately),
                   DL.noamLR, DL.smoothTarget, DL.xentSoft, DL.smoothFloor.
                   Every backward pass audited against central differences;
                   the page prints the worst relative error.
                   READ THE TRANSFORMER header above sinusoidalPE.   [part 6]
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

/* ══ RECURRENT LAYERS ═══════════════════════════════════════════════════
   [added by part 4 — RNNs & LSTMs]

   ── SEQUENCE LAYOUT, declared once ───────────────────────────────────
   A single sequence is TIME-MAJOR, one TIME STEP PER ROW. This is the
   series row convention with the batch axis N = 1 dropped:

       X  is (T × dₓ)      xₜ = X[t]    a ROW vector
       H  is (T × dₕ)      hₜ = H[t]
       a batch would be (N, T, d); every function here takes ONE sequence.

   Weights keep the (dᵢₙ × dₒᵤₜ) shape of the rest of the library:

       U (dₓ × dₕ)   input  → hidden
       W (dₕ × dₕ)   hidden → hidden      (the matrix that is REUSED)
       V (dₕ × d_y)  hidden → output

   so one step is       zₜ = hₜ₋₁·W + xₜ·U + b        row vectors on the LEFT.

   COLUMN-CONVENTION sources — including most textbooks and slide decks —
   write zₜ = W hₜ₋₁ + U xₜ + b. Their U, W, V are the TRANSPOSES of these.
   Mixing the two is the single most common shape bug in a hand-derived
   BPTT and is why the layout is stated here rather than in a figure.     */

/* generic numeric Jacobian, ∂f/∂x by CENTRAL differences.
   f: (Float array in) → (Float array out). Returns (m × n), m = |f(x)|. */
function jacFD(f, x, eps) {
  const h = eps === undefined ? 1e-5 : eps;
  const n = x.length, y0 = f(x), m = y0.length;
  const J = zeros2(m, n);
  for (let j = 0; j < n; j++) {
    const xp = x.slice(), xm = x.slice();
    xp[j] += h; xm[j] -= h;
    const a = f(xp), b = f(xm);
    for (let i = 0; i < m; i++) J[i][j] = (a[i] - b[i]) / (2 * h);
  }
  return J;
}

/* ── eigenvalues of a GENERAL real square matrix ──────────────────────
   DL.powerIter only ever returns the DOMINANT REAL eigenvalue and stalls
   on a complex pair; DL.eig2sym is exact but only 2×2 symmetric. The
   spectral radius of a recurrent weight matrix is the quantity that
   decides whether a gradient vanishes or explodes, and that matrix is
   neither symmetric nor guaranteed to have a real dominant eigenvalue —
   a rotation-like W has a complex conjugate pair. So: complex shifted QR
   with deflation. n ≤ ~24; the cost is irrelevant at that size.        */
function eigGeneral(A, opt) {
  const o = Object.assign({ iters: 500, tol: 1e-13 }, opt || {});
  const n = A.length;
  if (n === 1) return { re: [A[0][0]], im: [0], rho: Math.abs(A[0][0]) };
  /* complex working copy */
  let Hr = A.map(r => r.slice()), Hi = zeros2(n, n);
  const re = new Array(n).fill(0), im = new Array(n).fill(0);
  const csqrt = (a, b) => {                        // principal √(a+bi)
    const m = Math.hypot(a, b);
    if (m === 0) return [0, 0];
    const r = Math.sqrt((m + a) / 2);
    let i = Math.sqrt(Math.max(0, (m - a) / 2));
    if (b < 0) i = -i;
    return [r, i];
  };
  for (let m = n - 1; m > 0; m--) {
    let it = 0;
    for (; it < o.iters; it++) {
      const off = Math.hypot(Hr[m][m - 1], Hi[m][m - 1]);
      const sc = Math.hypot(Hr[m][m], Hi[m][m]) + Math.hypot(Hr[m - 1][m - 1], Hi[m - 1][m - 1]);
      if (off <= o.tol * (sc + 1e-300)) break;
      /* Wilkinson shift from the trailing 2×2 [[a b],[c d]] */
      const ar = Hr[m - 1][m - 1], ai = Hi[m - 1][m - 1];
      const br = Hr[m - 1][m], bi = Hi[m - 1][m];
      const cr = Hr[m][m - 1], ci = Hi[m][m - 1];
      const dr = Hr[m][m], di = Hi[m][m];
      const tr = ar - dr, ti = ai - di;            // a − d
      const hr = tr / 2, hi = ti / 2;
      const bcr = br * cr - bi * ci, bci = br * ci + bi * cr;   // b·c
      const rr = hr * hr - hi * hi + bcr, ri = 2 * hr * hi + bci; // h² + bc
      const [sr, si] = csqrt(rr, ri);
      /* the two roots are d + h ± √(h²+bc); pick the one nearer d */
      const p1r = hr + sr, p1i = hi + si, p2r = hr - sr, p2i = hi - si;
      const useP1 = Math.hypot(p1r, p1i) <= Math.hypot(p2r, p2i);
      let mur = dr + (useP1 ? p1r : p2r), mui = di + (useP1 ? p1i : p2i);
      if (!isFinite(mur) || !isFinite(mui)) { mur = dr; mui = di; }
      if (it > 0 && it % 23 === 0) { mur = dr + 0.7 * Math.abs(Hr[m][m - 1]); mui = di; } // exceptional
      /* H − μI */
      for (let i = 0; i <= m; i++) { Hr[i][i] -= mur; Hi[i][i] -= mui; }
      /* complex QR by modified Gram–Schmidt on the leading (m+1) block */
      const k = m + 1;
      const Qr = zeros2(k, k), Qi = zeros2(k, k), Rr = zeros2(k, k), Ri = zeros2(k, k);
      const vr = [], vi = [];
      for (let j = 0; j < k; j++) { vr.push(Hr.slice(0, k).map(r => r[j])); vi.push(Hi.slice(0, k).map(r => r[j])); }
      for (let j = 0; j < k; j++) {
        for (let p = 0; p < j; p++) {
          let dr2 = 0, di2 = 0;                      // conj(q_p)·v_j
          for (let i = 0; i < k; i++) { dr2 += Qr[i][p] * vr[j][i] + Qi[i][p] * vi[j][i]; di2 += Qr[i][p] * vi[j][i] - Qi[i][p] * vr[j][i]; }
          Rr[p][j] = dr2; Ri[p][j] = di2;
          for (let i = 0; i < k; i++) {
            vr[j][i] -= dr2 * Qr[i][p] - di2 * Qi[i][p];
            vi[j][i] -= dr2 * Qi[i][p] + di2 * Qr[i][p];
          }
        }
        let nn = 0; for (let i = 0; i < k; i++) nn += vr[j][i] * vr[j][i] + vi[j][i] * vi[j][i];
        nn = Math.sqrt(nn);
        Rr[j][j] = nn; Ri[j][j] = 0;
        const inv = nn > 1e-300 ? 1 / nn : 0;
        for (let i = 0; i < k; i++) { Qr[i][j] = vr[j][i] * inv; Qi[i][j] = vi[j][i] * inv; }
      }
      /* H ← R·Q + μI */
      const Nr = zeros2(k, k), Ni = zeros2(k, k);
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
        let sr2 = 0, si2 = 0;
        for (let p = i; p < k; p++) { sr2 += Rr[i][p] * Qr[p][j] - Ri[i][p] * Qi[p][j]; si2 += Rr[i][p] * Qi[p][j] + Ri[i][p] * Qr[p][j]; }
        Nr[i][j] = sr2; Ni[i][j] = si2;
      }
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { Hr[i][j] = Nr[i][j]; Hi[i][j] = Ni[i][j]; }
      for (let i = 0; i < k; i++) { Hr[i][i] += mur; Hi[i][i] += mui; }
    }
    re[m] = Hr[m][m]; im[m] = Hi[m][m];
  }
  re[0] = Hr[0][0]; im[0] = Hi[0][0];
  let rho = 0;
  for (let i = 0; i < n; i++) rho = Math.max(rho, Math.hypot(re[i], im[i]));
  return { re: re, im: im, rho: rho };
}
const specRad = A => eigGeneral(A).rho;

/* rescale a square matrix so that its spectral radius is exactly target */
function scaleToRho(A, target) {
  const r = specRad(A);
  const f = r > 1e-12 ? target / r : 0;
  return A.map(row => row.map(v => v * f));
}
function matPow(A, k) {
  let R = eye(A.length), B = A.map(r => r.slice()), e = k;
  while (e > 0) { if (e & 1) R = matmul(R, B); B = matmul(B, B); e >>= 1; }
  return R;
}

/* ── the three recurrent CELLS, one interface ─────────────────────────
   Every cell exposes the same four things, so a figure can swap one for
   another without knowing which it holds:

     nParam(dₓ, dₕ)                 exact parameter count, ONE bias set
     init(dₓ, dₕ, opt) → p          seeded parameters
     forward(p, X, s0) → st         st.H is (T × dₕ); st keeps what the
                                    backward pass needs and nothing else
     backward(p, st, dH) → g        dH is (T × dₕ), the gradient ARRIVING
                                    at each hₜ from outside the cell;
                                    g holds one entry per parameter array
                                    plus g.dh0 (and g.dc0 for the LSTM).

   The output head (V, c) is deliberately OUTSIDE the cell: all three
   share it, which is what makes a like-for-like comparison honest.     */

/* NOTE on the option objects below: they are merged with Object.assign, and
   Object.assign COPIES an explicit `undefined`, overwriting the default. Every
   option test therefore uses `== null` rather than `=== null`, so that passing
   {wScale: undefined} — which a caller does naturally with a conditional —
   falls back to the default instead of silently producing a NaN matrix. This
   bug was found by the figure harness on part 4 and is worth not repeating. */
function seqRandn(m, n, r, s) {
  const A = zeros2(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) A[i][j] = randn(r) * s;
  return A;
}
const dsig = a => a * (1 - a);                     // σ′ from the OUTPUT
const dtanh = a => 1 - a * a;                      // tanh′ from the OUTPUT
function addOuter(A, u, v) {                       // A += uᵀv, u and v rows
  for (let i = 0; i < u.length; i++) { const ui = u[i]; if (ui === 0) continue; const Ai = A[i]; for (let j = 0; j < v.length; j++) Ai[j] += ui * v[j]; }
}
function addTo(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; return a; }

/* ---- vanilla (Elman) cell: hₜ = tanh(hₜ₋₁W + xₜU + b) ---------------- */
function rnnCellInit(dx, dh, opt) {
  const o = Object.assign({ seed: 3, uScale: null, wScale: null, rho: null }, opt || {});
  const r = rng(o.seed);
  const U = seqRandn(dx, dh, r, o.uScale == null ? Math.sqrt(1 / dx) : o.uScale);
  let W = seqRandn(dh, dh, r, o.wScale == null ? Math.sqrt(1 / dh) : o.wScale);
  if (o.rho != null) W = scaleToRho(W, o.rho);
  return { kind: "vanilla", dx: dx, dh: dh, U: U, W: W, b: zeros(dh) };
}
function rnnCellForward(p, X, h0) {
  const T = X.length, H = [], Hprev = [];
  let h = h0 ? h0.slice() : zeros(p.dh);
  for (let t = 0; t < T; t++) {
    Hprev.push(h);
    const z = p.b.slice(), x = X[t];
    for (let i = 0; i < p.dh; i++) { const hi = h[i]; if (hi !== 0) { const Wi = p.W[i]; for (let j = 0; j < p.dh; j++) z[j] += hi * Wi[j]; } }
    for (let i = 0; i < p.dx; i++) { const xi = x[i]; if (xi !== 0) { const Ui = p.U[i]; for (let j = 0; j < p.dh; j++) z[j] += xi * Ui[j]; } }
    h = z.map(Math.tanh);
    H.push(h);
  }
  return { kind: "vanilla", X: X, H: H, Hprev: Hprev, T: T };
}
function rnnCellBackward(p, st, dH) {
  const T = st.T, dU = zeros2(p.dx, p.dh), dW = zeros2(p.dh, p.dh), db = zeros(p.dh);
  let dnext = zeros(p.dh);
  for (let t = T - 1; t >= 0; t--) {
    const h = st.H[t], hp = st.Hprev[t], x = st.X[t];
    const dz = new Array(p.dh);
    for (let j = 0; j < p.dh; j++) dz[j] = (dH[t][j] + dnext[j]) * dtanh(h[j]);
    addOuter(dW, hp, dz); addOuter(dU, x, dz); addTo(db, dz);
    dnext = vecmat(dz, transpose(p.W));
  }
  return { dU: dU, dW: dW, db: db, dh0: dnext };
}

/* ---- LSTM cell ------------------------------------------------------
     fₜ = σ(hₜ₋₁W_f + xₜU_f + b_f)      iₜ = σ(… W_i, U_i, b_i)
     gₜ = tanh(… W_g, U_g, b_g)          oₜ = σ(… W_o, U_o, b_o)
     cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ            hₜ = oₜ ⊙ tanh(cₜ)
   ONE bias set per gate — see DL.cellParams for the frameworks that
   carry two and the parameter count that follows.                       */
const LSTM_GATES = ["f", "i", "g", "o"];
function lstmCellInit(dx, dh, opt) {
  const o = Object.assign({ seed: 3, uScale: null, wScale: null, forgetBias: 1 }, opt || {});
  const r = rng(o.seed);
  const p = { kind: "lstm", dx: dx, dh: dh, U: {}, W: {}, b: {} };
  LSTM_GATES.forEach(k => {
    p.U[k] = seqRandn(dx, dh, r, o.uScale == null ? Math.sqrt(1 / dx) : o.uScale);
    p.W[k] = seqRandn(dh, dh, r, o.wScale == null ? Math.sqrt(1 / dh) : o.wScale);
    p.b[k] = zeros(dh).map(() => (k === "f" ? (o.forgetBias == null ? 1 : o.forgetBias) : 0));
  });
  return p;
}
function gateAffine(p, k, hp, x, dh) {
  const z = p.b[k].slice(), Wk = p.W[k], Uk = p.U[k];
  for (let i = 0; i < hp.length; i++) { const hi = hp[i]; if (hi !== 0) { const Wi = Wk[i]; for (let j = 0; j < dh; j++) z[j] += hi * Wi[j]; } }
  for (let i = 0; i < x.length; i++) { const xi = x[i]; if (xi !== 0) { const Ui = Uk[i]; for (let j = 0; j < dh; j++) z[j] += xi * Ui[j]; } }
  return z;
}
function lstmCellForward(p, X, h0, c0) {
  const T = X.length, dh = p.dh;
  const st = { kind: "lstm", X: X, T: T, H: [], Hprev: [], C: [], Cprev: [], F: [], I: [], G: [], O: [], TC: [] };
  let h = h0 ? h0.slice() : zeros(dh), c = c0 ? c0.slice() : zeros(dh);
  for (let t = 0; t < T; t++) {
    st.Hprev.push(h); st.Cprev.push(c);
    const f = gateAffine(p, "f", h, X[t], dh).map(sigmoid);
    const i = gateAffine(p, "i", h, X[t], dh).map(sigmoid);
    const g = gateAffine(p, "g", h, X[t], dh).map(Math.tanh);
    const o = gateAffine(p, "o", h, X[t], dh).map(sigmoid);
    const cn = new Array(dh), tc = new Array(dh), hn = new Array(dh);
    for (let j = 0; j < dh; j++) { cn[j] = f[j] * c[j] + i[j] * g[j]; tc[j] = Math.tanh(cn[j]); hn[j] = o[j] * tc[j]; }
    st.F.push(f); st.I.push(i); st.G.push(g); st.O.push(o); st.C.push(cn); st.TC.push(tc); st.H.push(hn);
    h = hn; c = cn;
  }
  return st;
}
function lstmCellBackward(p, st, dH, dCend) {
  const dh = p.dh, T = st.T;
  const dU = {}, dW = {}, db = {};
  LSTM_GATES.forEach(k => { dU[k] = zeros2(p.dx, dh); dW[k] = zeros2(dh, dh); db[k] = zeros(dh); });
  let dhn = zeros(dh), dcn = dCend ? dCend.slice() : zeros(dh);
  for (let t = T - 1; t >= 0; t--) {
    const f = st.F[t], i = st.I[t], g = st.G[t], o = st.O[t], tc = st.TC[t];
    const hp = st.Hprev[t], cp = st.Cprev[t], x = st.X[t];
    const zf = new Array(dh), zi = new Array(dh), zg = new Array(dh), zo = new Array(dh), dcNext = new Array(dh);
    for (let j = 0; j < dh; j++) {
      const dhj = dH[t][j] + dhn[j];
      const dcj = dhj * o[j] * dtanh(tc[j]) + dcn[j];
      zo[j] = dhj * tc[j] * dsig(o[j]);
      zf[j] = dcj * cp[j] * dsig(f[j]);
      zi[j] = dcj * g[j] * dsig(i[j]);
      zg[j] = dcj * i[j] * dtanh(g[j]);
      dcNext[j] = dcj * f[j];
    }
    const dzs = { f: zf, i: zi, g: zg, o: zo };
    const dhp = zeros(dh);
    LSTM_GATES.forEach(k => {
      addOuter(dW[k], hp, dzs[k]); addOuter(dU[k], x, dzs[k]); addTo(db[k], dzs[k]);
      const Wk = p.W[k];
      for (let a = 0; a < dh; a++) { let s = 0; const Wa = Wk[a]; for (let j = 0; j < dh; j++) s += dzs[k][j] * Wa[j]; dhp[a] += s; }
    });
    dhn = dhp; dcn = dcNext;
  }
  return { dU: dU, dW: dW, db: db, dh0: dhn, dc0: dcn };
}

/* ---- GRU cell -------------------------------------------------------
     zₜ = σ(hₜ₋₁W_z + xₜU_z + b_z)        UPDATE gate
     rₜ = σ(hₜ₋₁W_r + xₜU_r + b_r)        RESET gate
     c̃ₜ = tanh((rₜ ⊙ hₜ₋₁)W_c + xₜU_c + b_c)
     hₜ = zₜ ⊙ c̃ₜ + (1 − zₜ) ⊙ hₜ₋₁
   NOTE the polarity: here zₜ is HOW MUCH NEW. Several standard sources
   and most framework implementations write hₜ = zₜ⊙hₜ₋₁ + (1−zₜ)⊙c̃ₜ,
   in which zₜ is HOW MUCH OLD. The two are the same model with z ↦ 1−z
   (the gate simply learns the opposite sign); the page says which it
   means every time it prints a gate value.                              */
const GRU_GATES = ["z", "r", "c"];
function gruCellInit(dx, dh, opt) {
  const o = Object.assign({ seed: 3, uScale: null, wScale: null }, opt || {});
  const r = rng(o.seed);
  const p = { kind: "gru", dx: dx, dh: dh, U: {}, W: {}, b: {} };
  GRU_GATES.forEach(k => {
    p.U[k] = seqRandn(dx, dh, r, o.uScale == null ? Math.sqrt(1 / dx) : o.uScale);
    p.W[k] = seqRandn(dh, dh, r, o.wScale == null ? Math.sqrt(1 / dh) : o.wScale);
    p.b[k] = zeros(dh);
  });
  return p;
}
function gruCellForward(p, X, h0) {
  const dh = p.dh, T = X.length;
  const st = { kind: "gru", X: X, T: T, H: [], Hprev: [], Z: [], R: [], Q: [], Ct: [] };
  let h = h0 ? h0.slice() : zeros(dh);
  for (let t = 0; t < T; t++) {
    st.Hprev.push(h);
    const z = gateAffine(p, "z", h, X[t], dh).map(sigmoid);
    const r = gateAffine(p, "r", h, X[t], dh).map(sigmoid);
    const q = new Array(dh);
    for (let j = 0; j < dh; j++) q[j] = r[j] * h[j];
    const ct = gateAffine(p, "c", q, X[t], dh).map(Math.tanh);
    const hn = new Array(dh);
    for (let j = 0; j < dh; j++) hn[j] = z[j] * ct[j] + (1 - z[j]) * h[j];
    st.Z.push(z); st.R.push(r); st.Q.push(q); st.Ct.push(ct); st.H.push(hn);
    h = hn;
  }
  return st;
}
function gruCellBackward(p, st, dH) {
  const dh = p.dh, T = st.T;
  const dU = {}, dW = {}, db = {};
  GRU_GATES.forEach(k => { dU[k] = zeros2(p.dx, dh); dW[k] = zeros2(dh, dh); db[k] = zeros(dh); });
  let dnext = zeros(dh);
  for (let t = T - 1; t >= 0; t--) {
    const z = st.Z[t], r = st.R[t], q = st.Q[t], ct = st.Ct[t], hp = st.Hprev[t], x = st.X[t];
    const dhv = new Array(dh), dzr = new Array(dh), drc = new Array(dh), dhp = zeros(dh);
    for (let j = 0; j < dh; j++) {
      dhv[j] = dH[t][j] + dnext[j];
      dzr[j] = dhv[j] * (ct[j] - hp[j]) * dsig(z[j]);
      drc[j] = dhv[j] * z[j] * dtanh(ct[j]);
      dhp[j] += dhv[j] * (1 - z[j]);                    // the CARRY path
    }
    addOuter(dW.c, q, drc); addOuter(dU.c, x, drc); addTo(db.c, drc);
    const dq = vecmat(drc, transpose(p.W.c));
    const drr = new Array(dh);
    for (let j = 0; j < dh; j++) { drr[j] = dq[j] * hp[j] * dsig(r[j]); dhp[j] += dq[j] * r[j]; }
    addOuter(dW.r, hp, drr); addOuter(dU.r, x, drr); addTo(db.r, drr);
    addOuter(dW.z, hp, dzr); addOuter(dU.z, x, dzr); addTo(db.z, dzr);
    addTo(dhp, vecmat(drr, transpose(p.W.r)));
    addTo(dhp, vecmat(dzr, transpose(p.W.z)));
    dnext = dhp;
  }
  return { dU: dU, dW: dW, db: db, dh0: dnext };
}

/* the registry the figures index into */
const CELL = {
  vanilla: {
    key: "vanilla", label: "vanilla RNN", gates: [], nGate: 1,
    init: rnnCellInit, forward: rnnCellForward, backward: rnnCellBackward,
    nParam: (dx, dh) => dx * dh + dh * dh + dh
  },
  lstm: {
    key: "lstm", label: "LSTM", gates: LSTM_GATES, nGate: 4,
    init: lstmCellInit, forward: lstmCellForward, backward: lstmCellBackward,
    nParam: (dx, dh) => 4 * (dx * dh + dh * dh + dh)
  },
  gru: {
    key: "gru", label: "GRU", gates: GRU_GATES, nGate: 3,
    init: gruCellInit, forward: gruCellForward, backward: gruCellBackward,
    nParam: (dx, dh) => 3 * (dx * dh + dh * dh + dh)
  }
};
const cellList = () => ["vanilla", "gru", "lstm"];
/* exact parameter count, with the bias convention made explicit:
   biasSets = 1 is the textbook cell; biasSets = 2 is what a framework that
   carries a separate input bias and hidden bias actually allocates.      */
function cellParams(kind, dx, dh, biasSets) {
  const g = CELL[kind].nGate, bs = biasSets === undefined ? 1 : biasSets;
  return { gates: g, U: g * dx * dh, W: g * dh * dh, b: g * bs * dh,
           total: g * (dx * dh + dh * dh + bs * dh) };
}

/* ── the flat parameter view, so ONE finite-difference audit covers all
     three cells and the output head as well ─────────────────────────── */
function cellFlatten(p) {
  const names = [], out = [];
  const push = (nm, A) => { if (Array.isArray(A[0])) { for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) { names.push(nm + "[" + i + "][" + j + "]"); out.push(A[i][j]); } } else { for (let i = 0; i < A.length; i++) { names.push(nm + "[" + i + "]"); out.push(A[i]); } } };
  const gs = CELL[p.kind].gates;
  if (gs.length === 0) { push("U", p.U); push("W", p.W); push("b", p.b); }
  else gs.forEach(k => { push("U" + k, p.U[k]); push("W" + k, p.W[k]); push("b" + k, p.b[k]); });
  return { names: names, v: out };
}
function cellUnflatten(p, v) {
  let k = 0;
  const pull = A => { if (Array.isArray(A[0])) { for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) A[i][j] = v[k++]; } else { for (let i = 0; i < A.length; i++) A[i] = v[k++]; } };
  const gs = CELL[p.kind].gates;
  if (gs.length === 0) { pull(p.U); pull(p.W); pull(p.b); }
  else gs.forEach(g => { pull(p.U[g]); pull(p.W[g]); pull(p.b[g]); });
  return p;
}
function cellGradFlatten(p, g) {
  const out = [];
  const push = A => { if (Array.isArray(A[0])) { for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) out.push(A[i][j]); } else { for (let i = 0; i < A.length; i++) out.push(A[i]); } };
  const gs = CELL[p.kind].gates;
  if (gs.length === 0) { push(g.dU); push(g.dW); push(g.db); }
  else gs.forEach(k => { push(g.dU[k]); push(g.dW[k]); push(g.db[k]); });
  return out;
}
function cellClone(p) {
  const q = JSON.parse(JSON.stringify(p));
  return q;
}

/* ── a whole sequence MODEL: a cell plus the shared output head ───────
     sₜ = hₜ·V + c          aₜ = out(sₜ)          L = ∑ₜ Lₜ
   out ∈ {"sigmoid" (per-unit BCE), "softmax" (cross-entropy),
          "linear" (½‖·‖² summed over units)}.  In ALL THREE the gradient
   at the pre-activation is the same expression, ∂Lₜ/∂sₜ = aₜ − yₜ; the
   page derives why, and the audit below is what proves it.              */
function seqInit(kind, dx, dh, dy, opt) {
  const o = Object.assign({ seed: 3, out: "sigmoid", vScale: null }, opt || {});
  const p = CELL[kind].init(dx, dh, o);
  const r = rng(o.seed + 8191);
  return { cell: kind, dx: dx, dh: dh, dy: dy, out: o.out, p: p,
           V: seqRandn(dh, dy, r, o.vScale == null ? Math.sqrt(1 / dh) : o.vScale), c: zeros(dy) };
}
function seqForward(net, X, h0, c0) {
  const st = (net.cell === "lstm") ? CELL.lstm.forward(net.p, X, h0, c0) : CELL[net.cell].forward(net.p, X, h0);
  const S = [], A = [];
  for (let t = 0; t < st.T; t++) {
    const s = vecmat(st.H[t], net.V).map((v, j) => v + net.c[j]);
    S.push(s);
    A.push(net.out === "softmax" ? softmax(s) : (net.out === "sigmoid" ? s.map(sigmoid) : s));
  }
  st.S = S; st.A = A;
  return st;
}
function seqLossAt(net, a, s, y) {
  let L = 0;
  if (net.out === "softmax") { const ls = logSoftmax(s); for (let j = 0; j < y.length; j++) L -= y[j] * ls[j]; }
  else if (net.out === "sigmoid") { for (let j = 0; j < y.length; j++) L += softplus(s[j]) - y[j] * s[j]; }  // stable BCE
  else { for (let j = 0; j < y.length; j++) L += 0.5 * (a[j] - y[j]) * (a[j] - y[j]); }
  return L;
}
function seqLoss(net, X, Y, st) {
  const s = st || seqForward(net, X);
  let L = 0;
  for (let t = 0; t < s.T; t++) if (Y[t]) L += seqLossAt(net, s.A[t], s.S[t], Y[t]);
  return L;
}
/* the head's own gradients, and the dH it hands the cell */
function seqHead(net, st, Y) {
  const dV = zeros2(net.dh, net.dy), dc = zeros(net.dy), dH = [], dS = [];
  for (let t = 0; t < st.T; t++) {
    const ds = zeros(net.dy);
    /* ∂Lₜ/∂sₜ. For the sigmoid and linear heads this is aₜ − yₜ outright. For
       the SOFTMAX head it is (∑ⱼ yⱼ)·aₜ − yₜ, which collapses to aₜ − yₜ only
       when the target is a genuine distribution. Writing the general form
       costs one sum and removes a silent failure when a target is a count. */
    if (Y[t]) {
      let m = 1;
      if (net.out === "softmax") { m = 0; for (let j = 0; j < net.dy; j++) m += Y[t][j]; }
      for (let j = 0; j < net.dy; j++) ds[j] = m * st.A[t][j] - Y[t][j];
    }
    dS.push(ds);
    addOuter(dV, st.H[t], ds); addTo(dc, ds);
    dH.push(vecmat(ds, transpose(net.V)));
  }
  return { dV: dV, dc: dc, dH: dH, dS: dS };
}
/* FULL BPTT — every loss reaches every earlier step it can */
function seqBPTT(net, X, Y, st) {
  const s = st || seqForward(net, X);
  const hd = seqHead(net, s, Y);
  const g = CELL[net.cell].backward(net.p, s, hd.dH);
  g.dV = hd.dV; g.dc = hd.dc; g.loss = seqLoss(net, X, Y, s); g.st = s; g.dH = hd.dH;
  return g;
}
/* TRUNCATED BPTT, textbook form: the loss at step t is allowed to reach
   back k steps and no further. Implemented exactly — one restricted
   backward sweep per t — so the comparison against seqBPTT is the
   truncation bias itself and not an artefact of some other difference. */
function seqBPTTtrunc(net, X, Y, k, st) {
  const s = st || seqForward(net, X);
  const hd = seqHead(net, s, Y);
  const T = s.T, acc = null;
  let tot = null;
  for (let t = 0; t < T; t++) {
    const lo = Math.max(0, t - k + 1);
    const sub = sliceState(s, lo, t + 1);
    const dH = [];
    for (let u = lo; u <= t; u++) dH.push(u === t ? hd.dH[t].slice() : zeros(net.dh));
    const g = CELL[net.cell].backward(net.p, sub, dH);
    tot = tot === null ? g : addGrads(tot, g);
  }
  tot.dV = hd.dV; tot.dc = hd.dc; tot.loss = seqLoss(net, X, Y, s); tot.st = s;
  return tot;
}
/* CHUNKED TBPTT, the form frameworks actually run: split the sequence
   into non-overlapping windows of k, carry the STATE across a boundary
   but not the GRADIENT. Cheaper than the textbook form and MORE biased. */
function seqBPTTchunk(net, X, Y, k, st) {
  const s = st || seqForward(net, X);
  const hd = seqHead(net, s, Y);
  const T = s.T;
  let tot = null;
  for (let lo = 0; lo < T; lo += k) {
    const hi = Math.min(T, lo + k);
    const sub = sliceState(s, lo, hi);
    const dH = [];
    for (let u = lo; u < hi; u++) dH.push(hd.dH[u].slice());
    const g = CELL[net.cell].backward(net.p, sub, dH);
    tot = tot === null ? g : addGrads(tot, g);
  }
  tot.dV = hd.dV; tot.dc = hd.dc; tot.loss = seqLoss(net, X, Y, s); tot.st = s;
  return tot;
}
function sliceState(st, lo, hi) {
  const o = { kind: st.kind, T: hi - lo, X: st.X.slice(lo, hi), H: st.H.slice(lo, hi), Hprev: st.Hprev.slice(lo, hi) };
  ["C", "Cprev", "F", "I", "G", "O", "TC", "Z", "R", "Q", "Ct"].forEach(k => { if (st[k]) o[k] = st[k].slice(lo, hi); });
  return o;
}
function addGrads(a, b) {
  const addA = (X, Y) => { if (Array.isArray(X[0])) { for (let i = 0; i < X.length; i++) for (let j = 0; j < X[0].length; j++) X[i][j] += Y[i][j]; } else for (let i = 0; i < X.length; i++) X[i] += Y[i]; };
  ["dU", "dW", "db"].forEach(k => {
    if (Array.isArray(a[k])) addA(a[k], b[k]);
    else Object.keys(a[k]).forEach(g => addA(a[k][g], b[k][g]));
  });
  return a;
}
/* central-difference audit of EVERY parameter of the cell and the head */
function seqNumGrad(net, X, Y, h) {
  const eps = h === undefined ? 1e-6 : h;
  const flat = cellFlatten(net.p), v = flat.v.slice();
  const dCell = new Array(v.length);
  for (let i = 0; i < v.length; i++) {
    const o = v[i];
    v[i] = o + eps; cellUnflatten(net.p, v); const Lp = seqLoss(net, X, Y);
    v[i] = o - eps; cellUnflatten(net.p, v); const Lm = seqLoss(net, X, Y);
    v[i] = o; dCell[i] = (Lp - Lm) / (2 * eps);
  }
  cellUnflatten(net.p, v);
  const dV = zeros2(net.dh, net.dy);
  for (let i = 0; i < net.dh; i++) for (let j = 0; j < net.dy; j++) {
    const o = net.V[i][j];
    net.V[i][j] = o + eps; const Lp = seqLoss(net, X, Y);
    net.V[i][j] = o - eps; const Lm = seqLoss(net, X, Y);
    net.V[i][j] = o; dV[i][j] = (Lp - Lm) / (2 * eps);
  }
  const dc = zeros(net.dy);
  for (let j = 0; j < net.dy; j++) {
    const o = net.c[j];
    net.c[j] = o + eps; const Lp = seqLoss(net, X, Y);
    net.c[j] = o - eps; const Lm = seqLoss(net, X, Y);
    net.c[j] = o; dc[j] = (Lp - Lm) / (2 * eps);
  }
  return { flat: dCell, names: flat.names, dV: dV, dc: dc };
}
/* ∂h_T/∂h_k for ANY cell — the object the whole long-range argument is
   about. Measured by re-running the cell from a perturbed state, so it
   makes no assumption the analytic backward pass might share.           */
function stateJac(net, X, k, eps, h0, c0) {
  const dh = net.dh;
  const base = { h: h0 ? h0.slice() : zeros(dh), c: c0 ? c0.slice() : zeros(dh) };
  const st0 = (net.cell === "lstm") ? CELL.lstm.forward(net.p, X.slice(0, k), base.h, base.c) : CELL[net.cell].forward(net.p, X.slice(0, k), base.h);
  const hk = k === 0 ? base.h : st0.H[k - 1];
  const ck = (net.cell === "lstm") ? (k === 0 ? base.c : st0.C[k - 1]) : null;
  const tail = X.slice(k);
  const f = hv => {
    const s = (net.cell === "lstm") ? CELL.lstm.forward(net.p, tail, hv, ck) : CELL[net.cell].forward(net.p, tail, hv);
    return s.H[s.T - 1];
  };
  return jacFD(f, hk, eps === undefined ? 1e-5 : eps);
}

/* ── full symmetric eigendecomposition, cyclic Jacobi ─────────────────
   DL.eig2sym is exact but 2×2 only and DL.powerIter returns one pair.
   The context-vector capacity argument needs the WHOLE spectrum of a
   small symmetric matrix, and so does any singular-value question posed
   as ρ(AᵀA). Returns eigenvalues DESCENDING with matching columns of Q,
   so A = Q·diag(w)·Qᵀ.                                                  */
function eigSym(Ain, iters) {
  const n = Ain.length, A = Ain.map(r => r.slice()), Q = eye(n);
  const N = iters === undefined ? 60 : iters;
  for (let sweep = 0; sweep < N; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const qkp = Q[k][p], qkq = Q[k][q];
        Q[k][p] = c * qkp - s * qkq; Q[k][q] = s * qkp + c * qkq;
      }
    }
  }
  const idx = [];
  for (let i = 0; i < n; i++) idx.push(i);
  idx.sort((a, b) => A[b][b] - A[a][a]);
  const w = idx.map(i => A[i][i]);
  const V = zeros2(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) V[i][j] = Q[i][idx[j]];
  return { w: w, Q: V };
}
/* Moore–Penrose pseudo-inverse of a SYMMETRIC matrix, with the usual
   relative cut-off on the eigenvalues. */
function pinvSym(A, rcond) {
  const { w, Q } = eigSym(A), n = w.length;
  const cut = (rcond === undefined ? 1e-10 : rcond) * Math.max(...w.map(Math.abs), 1e-300);
  const D = zeros2(n, n);
  for (let i = 0; i < n; i++) D[i][i] = Math.abs(w[i]) > cut ? 1 / w[i] : 0;
  return matmul(matmul(Q, D), transpose(Q));
}
/* ridge solve of  min ‖X·B − Y‖² + λ‖B‖² , X (N×p), Y (N×q) → B (p×q) */
function ridge(X, Y, lam) {
  const Xt = transpose(X), G = matmul(Xt, X), p = G.length;
  for (let i = 0; i < p; i++) G[i][i] += (lam === undefined ? 1e-8 : lam);
  return matmul(pinvSym(G, 1e-14), matmul(Xt, Y));
}

/* ══ ATTENTION ══════════════════════════════════════════════════════════
   [part 5 — Attention]  Promoted here rather than kept page-local because
   part 6 (Transformers) is built out of exactly these calls, and a forked
   softmax-over-a-row or a forked head reshape is the specific class of bug
   THE RULE at the top of this file exists to prevent.

   ── THE ATTENTION LAYOUT LAW, declared once ─────────────────────────────
   It extends part 1's ROW convention and part 4's TIME-MAJOR sequence
   layout and never departs from either.

     one sequence      X   is (n × d_model)    one TOKEN per ROW
     projections       W_Q, W_K  (d_model × h·d_k)   W_V (d_model × h·d_v)
                       W_O       (h·d_v × d_model)
     so a projection is  X·W , the row vector on the LEFT, exactly as in
     part 1's  z = x·W + b.

     HEADS SPLIT THE LAST AXIS, and nothing else.  (n × h·d_k) is cut into
     h blocks of d_k ADJACENT columns; head j owns columns [j·d_k,(j+1)·d_k).
     splitHeads / mergeHeads are exact inverses and are the ONLY place this
     reshape may happen. Doing it by hand is where a multi-head
     implementation forks its convention silently: an interleaved split
     (stride h) also "works", produces a different model, and never throws.

     scores            S = Q·Kᵀ / √d_k            (n_q × n_k), per head
     weights           A = softmax over each ROW of S + mask
     head output       Z = A·V                    (n_q × d_v)
     block output      Y = concat_j(Z_j) · W_O    (n_q × d_model)

     A mask is ADDITIVE and its forbidden entries are −Infinity, never a
     large finite constant — see maskConst() for why, and note that
     softmaxRows returns NaN for an all-masked row ON PURPOSE.

     A batch is (N, n, d), batch axis first, and every figure on part 5
     uses N = 1, exactly as part 4 did.

   Cost is counted in MACs — multiply–accumulates — matching DL.macs,
   DL.convCount and part 4's attnMacs-equivalent. Sources quoting "FLOPs"
   usually mean MACs; those that mean literal operations quote 2×.       */

/* Row-wise softmax of a matrix, in the overflow-safe form. An all-masked
   row (every entry −Infinity) yields NaN, which is the CORRECT and loud
   behaviour: the quantity is undefined. Do not paper over it.           */
function softmaxRows(S) {
  return S.map(row => {
    let m = -Infinity;
    for (let j = 0; j < row.length; j++) if (row[j] > m) m = row[j];
    if (!isFinite(m)) return row.map(() => NaN);        // all −∞ ⇒ undefined
    let s = 0; const e = new Array(row.length);
    for (let j = 0; j < row.length; j++) { e[j] = Math.exp(row[j] - m); s += e[j]; }
    for (let j = 0; j < row.length; j++) e[j] /= s;
    return e;
  });
}
/* The softmax Jacobian ∂pᵢ/∂sⱼ = pᵢ(δᵢⱼ − pⱼ), returned as a full matrix.
   It is singular by construction (p is in its null space transposed) and
   it goes to ZERO at both ends: uniform p and one-hot p both kill it.   */
function softmaxJac(p) {
  const n = p.length, J = zeros2(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) J[i][j] = p[i] * ((i === j ? 1 : 0) - p[j]);
  return J;
}
function softmaxJacFrob(p) {                 // ‖diag(p) − ppᵀ‖_F, without the matrix
  /* grouped so that every term is non-negative: the algebraically equal
     form  √(Σp² − 2Σp³ + (Σp²)²)  cancels to 1e−08 relative near uniform. */
  let S2 = 0; for (let i = 0; i < p.length; i++) S2 += p[i] * p[i];
  let t = 0;
  for (let i = 0; i < p.length; i++) {
    const q = 1 - p[i];
    t += p[i] * p[i] * (q * q + S2 - p[i] * p[i]);
  }
  return Math.sqrt(Math.max(0, t));
}
function entropyOf(p) {                      // in NATS; ln(n) is the maximum
  let H = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) H -= p[i] * Math.log(p[i]);
  return H;
}
function perplexityOf(p) { return Math.exp(entropyOf(p)); }   // "effective #keys attended"

/* Heads split the LAST axis into h blocks of ADJACENT columns. */
function splitHeads(M, h) {
  const n = M.length, hd = M[0].length, d = hd / h, out = [];
  for (let j = 0; j < h; j++) {
    const H = zeros2(n, d);
    for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) H[i][c] = M[i][j * d + c];
    out.push(H);
  }
  return out;
}
function mergeHeads(T) {
  const h = T.length, n = T[0].length, d = T[0][0].length, M = zeros2(n, h * d);
  for (let j = 0; j < h; j++) for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) M[i][j * d + c] = T[j][i][c];
  return M;
}

/* ── masks. Every one is ADDITIVE, 0 to keep and −Infinity to forbid. ── */
function causalMask(nq, nk, off) {           // position i may read j ≤ i + off
  const K = nk === undefined ? nq : nk, o = off === undefined ? 0 : off;
  const M = zeros2(nq, K);
  for (let i = 0; i < nq; i++) for (let j = 0; j < K; j++) if (j > i + o) M[i][j] = -Infinity;
  return M;
}
function padMask(nq, valid) {                // valid: array of booleans over KEYS
  const M = zeros2(nq, valid.length);
  for (let i = 0; i < nq; i++) for (let j = 0; j < valid.length; j++) if (!valid[j]) M[i][j] = -Infinity;
  return M;
}
/* Sliding window of width w (w keys ending at i, inclusive), optionally
   dilated by `dil`, optionally with `sinks` always-visible leading keys.  */
function windowMask(n, w, o) {
  const oo = Object.assign({ causal: true, dil: 1, sinks: 0 }, o || {});
  const M = zeros2(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let ok = false;
    if (j <= i || !oo.causal) {
      const dist = i - j;
      if (dist >= 0 && dist < w * oo.dil && dist % oo.dil === 0) ok = true;
      if (!oo.causal && dist < 0 && -dist < w * oo.dil && (-dist) % oo.dil === 0) ok = true;
    }
    if (j < oo.sinks && (j <= i || !oo.causal)) ok = true;
    if (!ok) M[i][j] = -Infinity;
  }
  return M;
}
function addMasks(A, B) {                    // −∞ + finite = −∞; keeps 0 where both keep
  return A.map((r, i) => r.map((v, j) => v + B[i][j]));
}
/* How far below the row maximum a FINITE mask constant must sit before the
   masked weight underflows to exactly zero, per precision. Returned as the
   gap g such that exp(−g) is the smallest positive representable.        */
const maskConst = {
  float64: { zeroGap: 745.1, epsGap: 36.04, min: -1.7976931348623157e308 },
  float32: { zeroGap: 103.97, epsGap: 15.94, min: -3.4028234663852886e38 },
  float16: { zeroGap: 17.33, epsGap: 6.93, min: -65504 },
  note: "a finite constant added to EVERY entry of a row is a NO-OP: softmax is shift-invariant."
};

/* ── scaled dot-product attention, ONE head ─────────────────────────────
   Q (nq × dk), K (nk × dk), V (nk × dv). o.scale defaults to 1/√dk;
   pass o.scale = 1 to see what the scaling is for. o.mask is additive.  */
function sdpa(Q, K, V, o) {
  const oo = o || {}, dk = Q[0].length;
  const sc = oo.scale === undefined ? 1 / Math.sqrt(dk) : oo.scale;
  const nq = Q.length, nk = K.length, S = zeros2(nq, nk);
  for (let i = 0; i < nq; i++) for (let j = 0; j < nk; j++) {
    let s = 0; for (let c = 0; c < dk; c++) s += Q[i][c] * K[j][c];
    S[i][j] = s * sc + (oo.mask ? oo.mask[i][j] : 0);
  }
  const A = softmaxRows(S);
  const dv = V[0].length, Z = zeros2(nq, dv);
  for (let i = 0; i < nq; i++) for (let j = 0; j < nk; j++) {
    const a = A[i][j]; if (!a) continue;
    for (let c = 0; c < dv; c++) Z[i][c] += a * V[j][c];
  }
  return { S: S, A: A, Z: Z, scale: sc };
}

/* ── a multi-head attention BLOCK: init, forward, exact backward ────────
   No residual, no normalisation, no FFN — those are part 6's, deliberately.
   g = number of KEY/VALUE heads (g = h is MHA, 1 < g < h is GQA, g = 1 is
   MQA); query head j reads KV head floor(j·g/h).                         */
function mhaInit(dm, dk, dv, h, o) {
  const oo = Object.assign({ seed: 0, g: h, scale: null, bias: false }, o || {});
  const r = rng(oo.seed), s = oo.scale === null ? 1 / Math.sqrt(dm) : oo.scale;
  const gen = (a, b) => { const M = zeros2(a, b); for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) M[i][j] = randn(r) * s; return M; };
  return {
    Wq: gen(dm, h * dk), Wk: gen(dm, oo.g * dk), Wv: gen(dm, oo.g * dv), Wo: gen(h * dv, dm),
    h: h, g: oo.g, dk: dk, dv: dv, dm: dm
  };
}
function kvHeadOf(j, h, g) { return Math.floor(j * g / h); }
function mhaForward(X, P, o) {
  const oo = o || {}, h = P.h, g = P.g === undefined ? h : P.g;
  const Q = splitHeads(matmul(X, P.Wq), h);
  const Kall = splitHeads(matmul(X, P.Wk), g);
  const Vall = splitHeads(matmul(X, P.Wv), g);
  const per = [];
  for (let j = 0; j < h; j++) {
    const kv = kvHeadOf(j, h, g);
    per.push(sdpa(Q[j], Kall[kv], Vall[kv], { mask: oo.mask, scale: oo.scale }));
  }
  const C = mergeHeads(per.map(p => p.Z));
  const Y = matmul(C, P.Wo);
  return { Y: Y, st: { X: X, Q: Q, K: Kall, V: Vall, per: per, C: C, h: h, g: g, mask: oo.mask } };
}
function mhaBackward(dY, st, P) {
  const h = st.h, g = st.g, X = st.X, C = st.C;
  const gWo = matmul(transpose(C), dY);
  const dC = matmul(dY, transpose(P.Wo));
  const dZ = splitHeads(dC, h);
  const dQ = [], dKa = [], dVa = [];
  for (let j = 0; j < g; j++) { dKa.push(zeros2(st.K[j].length, st.K[j][0].length)); dVa.push(zeros2(st.V[j].length, st.V[j][0].length)); }
  for (let j = 0; j < h; j++) {
    const kv = kvHeadOf(j, h, g), A = st.per[j].A, sc = st.per[j].scale;
    const V = st.V[kv], K = st.K[kv], Q = st.Q[j];
    const nq = A.length, nk = A[0].length, dv = V[0].length, dk = Q[0].length;
    /* dA = dZ Vᵀ ; dV += Aᵀ dZ */
    const dA = zeros2(nq, nk);
    for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
      let s = 0; for (let c = 0; c < dv; c++) s += dZ[j][i][c] * V[k][c];
      dA[i][k] = s;
    }
    for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
      const a = A[i][k]; if (!a) continue;
      for (let c = 0; c < dv; c++) dVa[kv][k][c] += a * dZ[j][i][c];
    }
    /* dS = A ⊙ (dA − rowsum(dA ⊙ A)) , then × scale */
    const dS = zeros2(nq, nk);
    for (let i = 0; i < nq; i++) {
      let dot = 0; for (let k = 0; k < nk; k++) dot += dA[i][k] * A[i][k];
      for (let k = 0; k < nk; k++) dS[i][k] = A[i][k] * (dA[i][k] - dot) * sc;
    }
    const dQj = zeros2(nq, dk);
    for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
      const s = dS[i][k]; if (!s) continue;
      for (let c = 0; c < dk; c++) { dQj[i][c] += s * K[k][c]; dKa[kv][k][c] += s * Q[i][c]; }
    }
    dQ.push(dQj);
  }
  const mQ = mergeHeads(dQ), mK = mergeHeads(dKa), mV = mergeHeads(dVa);
  const gWq = matmul(transpose(X), mQ), gWk = matmul(transpose(X), mK), gWv = matmul(transpose(X), mV);
  const dX = matmul(mQ, transpose(P.Wq));
  const b1 = matmul(mK, transpose(P.Wk)), b2 = matmul(mV, transpose(P.Wv));
  for (let i = 0; i < dX.length; i++) for (let c = 0; c < dX[0].length; c++) dX[i][c] += b1[i][c] + b2[i][c];
  return { Wq: gWq, Wk: gWk, Wv: gWv, Wo: gWo, X: dX };
}
/* Central differences over a set of named matrices, for the audit figures.
   loss() must close over the SAME objects it is handed.                   */
function numGradMats(loss, mats, eps) {
  const e = eps === undefined ? 1e-5 : eps, out = {};
  Object.keys(mats).forEach(k => {
    const M = mats[k], G = zeros2(M.length, M[0].length);
    for (let i = 0; i < M.length; i++) for (let j = 0; j < M[0].length; j++) {
      const o = M[i][j];
      M[i][j] = o + e; const lp = loss();
      M[i][j] = o - e; const lm = loss();
      M[i][j] = o;
      G[i][j] = (lp - lm) / (2 * e);
    }
    out[k] = G;
  });
  return out;
}

/* ── counting ───────────────────────────────────────────────────────────
   Parameters of one attention block, exactly. g < h shrinks W_K and W_V
   and NOTHING else — the query and output projections are untouched.     */
function mhaParams(dm, dk, dv, h, g, bias) {
  const G = g === undefined ? h : g;
  const wq = dm * h * dk, wk = dm * G * dk, wv = dm * G * dv, wo = h * dv * dm;
  const b = bias ? (h * dk + G * dk + G * dv + dm) : 0;
  return { Wq: wq, Wk: wk, Wv: wv, Wo: wo, bias: b, total: wq + wk + wv + wo + b };
}
/* MACs for ONE layer's forward pass at sequence length n, batch 1.
   `nmat` is 2 for a GELU MLP and 3 for a SwiGLU one. `causal:true` halves
   the quadratic term, which is what a kernel that skips fully masked
   blocks actually achieves.                                              */
function attnMacs(o) {
  const c = Object.assign({ n: 1024, d: 768, h: 12, g: null, dh: 64, dff: 3072, nmat: 2, causal: false }, o || {});
  const g = c.g === null ? c.h : c.g;
  const proj = c.n * c.d * (c.h * c.dh) + 2 * c.n * c.d * (g * c.dh) + c.n * (c.h * c.dh) * c.d;
  const f = c.causal ? 0.5 * (1 + 1 / c.n) : 1;
  const quad = 2 * c.n * c.n * (c.h * c.dh) * f;
  const ffn = c.nmat * c.n * c.d * c.dff;
  return { proj: proj, quad: quad, ffn: ffn, attn: proj + quad, total: proj + quad + ffn };
}
/* The sequence length at which the quadratic term overtakes `against`.
   Closed form when h·dh = d and g = h:   quad = 2n²d, proj = 4nd²,
   ffn = nmat·n·d·dff  ⇒  n* = 2d (proj), nmat·dff/2 (ffn), the sum (rest).
   Solved numerically here so that GQA (g < h) and h·dh ≠ d are handled.  */
function macCrossover(o, against) {
  let lo = 1, hi = 1 << 30;
  const val = n => {
    const m = attnMacs(Object.assign({}, o, { n: n }));
    return { quad: m.quad, other: against === "proj" ? m.proj : against === "ffn" ? m.ffn : m.proj + m.ffn };
  };
  while (hi - lo > 1) { const mid = Math.floor((lo + hi) / 2), v = val(mid); if (v.quad < v.other) lo = mid; else hi = mid; }
  return hi;
}
/* KV cache, in ELEMENTS per token, for the whole stack.
   kind: "mha" | "gqa" | "mqa" use 2·g·dh·L ; "mla" uses (dc + dr)·L.     */
function kvCacheElems(o) {
  const c = Object.assign({ h: 32, g: null, dh: 128, L: 32, kind: "gqa", dc: 512, dr: 64 }, o || {});
  if (c.kind === "mla") return (c.dc + c.dr) * c.L;
  const g = c.kind === "mha" ? c.h : c.kind === "mqa" ? 1 : (c.g === null ? c.h : c.g);
  return 2 * g * c.dh * c.L;
}
/* Arithmetic intensity of ONE decode step's attention over the cache:
   MACs 2·n·h·dh, bytes 2·n·g·dh·b  ⇒  I = h/(g·b) MAC/byte = 2h/(g·b)
   FLOP/byte. The sequence length CANCELS, and batching does not help
   because each sequence owns its cache.                                  */
function decodeIntensity(h, g, bytesPerElem) {
  const b = bytesPerElem === undefined ? 2 : bytesPerElem;
  return { mac: h / (g * b), flop: 2 * h / (g * b) };
}

/* ══ TRANSFORMER ════════════════════════════════════════════════════════
   [part 6 — Transformers]  Everything that turns an attention sub-layer
   into a model, promoted here because the remaining parts of the series
   (Residual connections, Encoder vs Decoder, Fine-tuning, Distillation,
   Distributed training) and the model cards (BERT, GPT) are built out of
   exactly these calls. Nothing above this line was changed.

   THE ATTENTION LAYOUT LAW CONTINUES TO HOLD. A sequence is X (n × d),
   one TOKEN per ROW; a weight is (d_in × d_out) and sits on the RIGHT;
   heads split the LAST axis into ADJACENT blocks; masks are ADDITIVE and
   −Infinity. Two consequences that are easy to get backwards:

     · RoPE rotates each ROW of Q and of K, pair by pair, by an angle that
       grows with that row's position. It never touches V.
     · In a position-wise FFN  Y = act(X·W₁ + b₁)·W₂ + b₂  the "keys" of the
       key–value-memory reading are the COLUMNS of W₁ (each is dotted with
       the token row x) and the "values" are the ROWS of W₂.

   Positional encodings
     sinusoidalPE(n, d[, base])       (n × d), sin in even columns, cos in
                                      odd, wavelength 2π·base^(2i/d)
     peShiftMatrix(k, d[, base])      the (d × d) block-rotation M_k with
                                      PE(p + k) = PE(p)·M_k  EXACTLY, for
                                      every p (the relative-offset property)
     ropeFreqs(d[, base])             θᵢ = base^(−2i/d), i = 0 … d/2 − 1
     rope(X, {base, offset, pairing, inverse})
                                      rotate every ROW of X (n × d) by its
                                      position (row index + offset).
                                      pairing "adjacent" pairs columns
                                      (2i, 2i+1) as the original paper does;
                                      "half" pairs (i, i + d/2) as the
                                      rotate-half implementations do. The
                                      two are DIFFERENT conventions with the
                                      SAME relative-offset property; a
                                      checkpoint is tied to one of them.
                                      inverse:true rotates by −angle, which
                                      is the exact backward pass because the
                                      rotation is orthogonal.
     alibiSlopes(h)                   the geometric slopes 2^(−8/h · j),
                                      with the original rule for h not a
                                      power of two
     alibiBias(n, slope)              (n × n) of −slope·|i − j|; ADD the
                                      causal mask separately
     relBucket(rel, {bidir, buckets, maxDist})
                                      the T5-style log-spaced bucket index
     relBucketMatrix(n, o)            (n × n) of bucket indices, j − i

   Normalisation with a cache and an exact backward pass (DL.layerNorm and
   DL.rmsNorm from part 2 return only Y; these return {Y, st} and agree
   with them to 0)
     lnForward(X, γ, β[, eps]) / lnBackward(dY, st) → {X, g, b}
     rmsForward(X, γ[, eps])   / rmsBackward(dY, st) → {X, g}

   The position-wise FFN
     ffnInit(d, dff, {act, gated, bias, seed, scale})
     ffnForward(X, P) → {Y, st};  ffnBackward(dY, st, P) → {W1,b1,W2,b2 | Wg,Wu,Wd, X}
       ungated: Y = act(X·W₁ + b₁)·W₂ + b₂
       gated:   Y = (act(X·W_g) ⊙ (X·W_u))·W_d          (SwiGLU when act = silu)
     ffnParams(d, dff, {gated, bias}); ffnWidthGated(d, {ratio, multiple, mult})
       — the ⅔ correction that keeps a gated FFN at the ungated parameter
       count, rounded up to a multiple as real configurations do

   Attention with optional RoPE and per-head additive biases
     attnForward(X, P, {mask, rope, scale}) / attnBackward(dY, st, P)
       — P from DL.mhaInit. o.mask may be ONE (n × n) matrix or an ARRAY of
       h matrices (ALiBi needs a different slope per head). With rope null
       and a single mask these reproduce DL.mhaForward / DL.mhaBackward to
       0.000e+00 and are audited against central differences below.

   The block
     blockInit(d, h, dff, {dk, dv, g, norm:"ln"|"rms", pre:true|false, act,
                            gated, bias, seed})
     blockForward(X, P, {mask, rope}) → {Y, st};  blockBackward(dY, st, P)
       pre-LN:   X₁ = X + Attn(N₁(X));   Y = X₁ + FFN(N₂(X₁))
       post-LN:  X₁ = N₁(X + Attn(X));   Y = N₂(X₁ + FFN(X₁))
     blockParams(d, h, dff, o) → {attn, ffn, norms, total}

   A whole tiny decoder-only model, for the live figures
     tfInit({V, d, L, h, dff, nCtx, pos, norm, pre, gated, act, tie,
             scaleEmb, embStd, seed, ropeBase, ropePairing})
       scaleEmb multiplies the looked-up row by √d (the 2017 convention,
       with E ~ N(0, 1/d)); embStd overrides E's initial std (the 0.02
       convention uses embStd = 0.02 with scaleEmb false)
       pos: "learned" | "sin" | "rope" | "alibi" | "none"
     tfForward(m, idx, o) → {logits, st}     idx = array of token ids
     tfLoss(m, idx, tgt[, o]) → mean cross-entropy over the positions
     tfBackward(m, idx, tgt[, o]) → {loss, g: {E, Pos, blocks[], nf, Wout}}
     tfParamList(m) → [{k, M}] every trainable matrix, by reference
     tfGradList(g, m) → the same order, for the optimiser
     optMats(mats, grads, states, key, hp) — one step of DL.OPT[key] over a
       list of matrices, each with its own flat state; decoupled decay uses
       hp.wd exactly as DL.OPT.adamw does

   Counting a whole model
     modelParams(cfg) → {embed, pos, attn, ffn, norms, unembed, total,
                         perBlock, blocks}
       cfg = {V, d, L, h, g, dk, dff, nCtx, pos, norm, gated, bias, tie}
     modelFlops(cfg, n) → {weightMacs, attnMacs, macs, flops, train6ND, …}
       PER TOKEN: forward MACs = N_nonembed + attention n·d term; training
       FLOPs = 6·N + 12·L·d·n for a causal model (the "6ND" rule with the
       quadratic term made explicit). States MACs and FLOPs separately.
     noamLR(step, d, warm)   d^(−½)·min(step^(−½), step·warm^(−3/2)), the
       original schedule, 1-BASED step; equals DL.lrAt("invsqrt", step,
       {peak: (d·warm)^(−½), warm}) to round-off (lrAt's linear warmup
       reaches the peak AT t = warm, which is the same step)
     smoothTarget(V, k, ε), xentSoft(logits, q), smoothFloor(V, ε) — label
       smoothing: the target, the loss against it, and the floor the loss
       cannot go below (the entropy of the smoothed target)
                                                                     [part 6] */

/* ── positional encodings ─────────────────────────────────────────────── */
function sinusoidalPE(n, d, base) {
  const b = base === undefined ? 10000 : base, P = zeros2(n, d);
  for (let p = 0; p < n; p++) for (let i = 0; 2 * i < d; i++) {
    const w = Math.pow(b, -2 * i / d);
    P[p][2 * i] = Math.sin(p * w);
    if (2 * i + 1 < d) P[p][2 * i + 1] = Math.cos(p * w);
  }
  return P;
}
/* PE(p + k) = PE(p)·M_k, row convention: for the pair (sin ωp, cos ωp),
   [s c]·[[cos ωk, −sin ωk],[sin ωk, cos ωk]] = [s cos ωk + c sin ωk,
   −s sin ωk + c cos ωk] = [sin ω(p+k), cos ω(p+k)].                        */
function peShiftMatrix(k, d, base) {
  const b = base === undefined ? 10000 : base, M = zeros2(d, d);
  for (let i = 0; 2 * i < d; i++) {
    const w = Math.pow(b, -2 * i / d), c = Math.cos(k * w), s = Math.sin(k * w);
    const a = 2 * i, q = 2 * i + 1;
    if (q < d) { M[a][a] = c; M[a][q] = -s; M[q][a] = s; M[q][q] = c; }
    else M[a][a] = 1;
  }
  return M;
}
function ropeFreqs(d, base) {
  const b = base === undefined ? 10000 : base, out = [];
  for (let i = 0; 2 * i < d; i++) out.push(Math.pow(b, -2 * i / d));
  return out;
}
function rope(X, o) {
  const oo = Object.assign({ base: 10000, offset: 0, pairing: "adjacent", inverse: false }, o || {});
  const n = X.length, d = X[0].length, half = Math.floor(d / 2), th = ropeFreqs(d, oo.base);
  const sgn = oo.inverse ? -1 : 1, Y = X.map(r => r.slice());
  for (let p = 0; p < n; p++) {
    const pos = p + oo.offset;
    for (let i = 0; i < half; i++) {
      const ia = oo.pairing === "half" ? i : 2 * i, ib = oo.pairing === "half" ? i + half : 2 * i + 1;
      const ang = sgn * pos * th[i], c = Math.cos(ang), s = Math.sin(ang);
      const a = X[p][ia], b = X[p][ib];
      Y[p][ia] = a * c - b * s;
      Y[p][ib] = a * s + b * c;
    }
  }
  return Y;
}
/* ALiBi slopes: for h a power of two, 2^(−8j/h) for j = 1…h; otherwise the
   closest power of two below, plus every other slope of the 2·that set.  */
function alibiSlopes(h) {
  const pow2 = n => { const s = Math.pow(2, -Math.pow(2, -(Math.log2(n) - 3))); const out = []; for (let j = 1; j <= n; j++) out.push(Math.pow(s, j)); return out; };
  if (Number.isInteger(Math.log2(h))) return pow2(h);
  const c = Math.pow(2, Math.floor(Math.log2(h)));
  const base = pow2(c), extra = pow2(2 * c).filter((_, i) => i % 2 === 0).slice(0, h - c);
  return base.concat(extra);
}
function alibiBias(n, slope, nk) {
  const K = nk === undefined ? n : nk, B = zeros2(n, K);
  for (let i = 0; i < n; i++) for (let j = 0; j < K; j++) B[i][j] = -slope * Math.abs(i - j);
  return B;
}
/* T5-style relative-position bucket of rel = j − i (key minus query).
   Half the buckets are exact small offsets, the other half log-spaced up
   to maxDist; beyond it everything shares the last bucket.               */
function relBucket(rel, o) {
  const oo = Object.assign({ bidir: true, buckets: 32, maxDist: 128 }, o || {});
  let nb = oo.buckets, r = rel, out = 0;
  if (oo.bidir) { nb = Math.floor(nb / 2); if (r > 0) out += nb; r = Math.abs(r); }
  else r = Math.max(-r, 0);
  const exact = Math.floor(nb / 2);
  if (r < exact) return out + r;
  const large = exact + Math.floor(Math.log(r / exact) / Math.log(oo.maxDist / exact) * (nb - exact));
  return out + Math.min(large, nb - 1);
}
function relBucketMatrix(n, o) {
  const M = zeros2(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] = relBucket(j - i, o);
  return M;
}

/* ── normalisation with cache and backward ───────────────────────────── */
function lnForward(X, g, b, eps) {
  const e = eps === undefined ? 1e-5 : eps, d = X[0].length;
  const xhat = [], inv = [];
  const Y = X.map(r => {
    let m = 0; for (let j = 0; j < d; j++) m += r[j]; m /= d;
    let v = 0; for (let j = 0; j < d; j++) v += (r[j] - m) * (r[j] - m); v /= d;
    const iv = 1 / Math.sqrt(v + e), xh = r.map(x => (x - m) * iv);
    xhat.push(xh); inv.push(iv);
    return xh.map((x, j) => (g ? g[j] : 1) * x + (b ? b[j] : 0));
  });
  return { Y: Y, st: { xhat: xhat, inv: inv, g: g, d: d } };
}
function lnBackward(dY, st) {
  const n = dY.length, d = st.d, dg = zeros(d), db = zeros(d), dX = zeros2(n, d);
  for (let i = 0; i < n; i++) {
    const xh = st.xhat[i], dxh = new Array(d);
    let m1 = 0, m2 = 0;
    for (let j = 0; j < d; j++) {
      dg[j] += dY[i][j] * xh[j]; db[j] += dY[i][j];
      dxh[j] = dY[i][j] * (st.g ? st.g[j] : 1);
      m1 += dxh[j]; m2 += dxh[j] * xh[j];
    }
    m1 /= d; m2 /= d;
    for (let j = 0; j < d; j++) dX[i][j] = st.inv[i] * (dxh[j] - m1 - xh[j] * m2);
  }
  return { X: dX, g: dg, b: db };
}
function rmsForward(X, g, eps) {
  const e = eps === undefined ? 1e-5 : eps, d = X[0].length, inv = [];
  const Y = X.map(r => {
    let ms = 0; for (let j = 0; j < d; j++) ms += r[j] * r[j]; ms /= d;
    const iv = 1 / Math.sqrt(ms + e); inv.push(iv);
    return r.map((x, j) => (g ? g[j] : 1) * x * iv);
  });
  return { Y: Y, st: { X: X, inv: inv, g: g, d: d } };
}
function rmsBackward(dY, st) {
  const n = dY.length, d = st.d, dg = zeros(d), dX = zeros2(n, d);
  for (let i = 0; i < n; i++) {
    const x = st.X[i], iv = st.inv[i];
    let dot = 0;
    for (let j = 0; j < d; j++) { const gj = st.g ? st.g[j] : 1; dg[j] += dY[i][j] * x[j] * iv; dot += dY[i][j] * gj * x[j]; }
    for (let j = 0; j < d; j++) dX[i][j] = (st.g ? st.g[j] : 1) * dY[i][j] * iv - x[j] * dot * iv * iv * iv / d;
  }
  return { X: dX, g: dg };
}

/* ── the position-wise FFN ────────────────────────────────────────────── */
function ffnInit(d, dff, o) {
  const oo = Object.assign({ act: "gelu", gated: false, bias: true, seed: 3, scale: null }, o || {});
  const r = rng(oo.seed);
  const gen = (a, b, s) => { const M = zeros2(a, b); for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) M[i][j] = randn(r) * s; return M; };
  const s1 = oo.scale === null ? 1 / Math.sqrt(d) : oo.scale, s2 = oo.scale === null ? 1 / Math.sqrt(dff) : oo.scale;
  if (oo.gated) return { gated: true, act: oo.act, d: d, dff: dff, Wg: gen(d, dff, s1), Wu: gen(d, dff, s1), Wd: gen(dff, d, s2) };
  return { gated: false, act: oo.act, d: d, dff: dff, bias: oo.bias, W1: gen(d, dff, s1), b1: zeros(dff), W2: gen(dff, d, s2), b2: zeros(d) };
}
function ffnForward(X, P) {
  const A = act(P.act);
  if (P.gated) {
    const Hg = matmul(X, P.Wg), Hu = matmul(X, P.Wu);
    const G = Hg.map((r, i) => r.map((v, j) => A.f(v) * Hu[i][j]));
    return { Y: matmul(G, P.Wd), st: { X: X, Hg: Hg, Hu: Hu, G: G } };
  }
  const H = matmul(X, P.W1);
  if (P.bias) for (let i = 0; i < H.length; i++) for (let j = 0; j < H[0].length; j++) H[i][j] += P.b1[j];
  const G = H.map(r => r.map(v => A.f(v)));           // never r.map(A.f): the index would land in the shape parameter
  const Y = matmul(G, P.W2);
  if (P.bias) for (let i = 0; i < Y.length; i++) for (let j = 0; j < Y[0].length; j++) Y[i][j] += P.b2[j];
  return { Y: Y, st: { X: X, H: H, G: G } };
}
function ffnBackward(dY, st, P) {
  const A = act(P.act), X = st.X;
  if (P.gated) {
    const gWd = matmul(transpose(st.G), dY), dG = matmul(dY, transpose(P.Wd));
    const dHu = dG.map((r, i) => r.map((v, j) => v * A.f(st.Hg[i][j])));
    const dHg = dG.map((r, i) => r.map((v, j) => v * st.Hu[i][j] * A.df(st.Hg[i][j])));
    const gWg = matmul(transpose(X), dHg), gWu = matmul(transpose(X), dHu);
    const dX = matmul(dHg, transpose(P.Wg)), t = matmul(dHu, transpose(P.Wu));
    for (let i = 0; i < dX.length; i++) for (let j = 0; j < dX[0].length; j++) dX[i][j] += t[i][j];
    return { Wg: gWg, Wu: gWu, Wd: gWd, X: dX };
  }
  const gW2 = matmul(transpose(st.G), dY), db2 = zeros(dY[0].length);
  for (let i = 0; i < dY.length; i++) for (let j = 0; j < dY[0].length; j++) db2[j] += dY[i][j];
  const dG = matmul(dY, transpose(P.W2));
  const dH = dG.map((r, i) => r.map((v, j) => v * A.df(st.H[i][j])));
  const gW1 = matmul(transpose(X), dH), db1 = zeros(dH[0].length);
  for (let i = 0; i < dH.length; i++) for (let j = 0; j < dH[0].length; j++) db1[j] += dH[i][j];
  return { W1: gW1, b1: db1, W2: gW2, b2: db2, X: matmul(dH, transpose(P.W1)) };
}
function ffnParams(d, dff, o) {
  const oo = Object.assign({ gated: false, bias: true }, o || {});
  if (oo.gated) return 3 * d * dff;
  return 2 * d * dff + (oo.bias ? dff + d : 0);
}
/* d_ff for a gated FFN at the same parameter count as an ungated ratio·d
   one: ⅔·ratio·d, optionally ×mult, rounded UP to a multiple.            */
function ffnWidthGated(d, o) {
  const oo = Object.assign({ ratio: 4, multiple: 1, mult: 1 }, o || {});
  const raw = (2 / 3) * oo.ratio * d * oo.mult;
  return oo.multiple > 1 ? oo.multiple * Math.ceil(raw / oo.multiple) : Math.round(raw);
}

/* ── attention with optional RoPE and per-head additive bias ─────────── */
function attnForward(X, P, o) {
  const oo = o || {}, h = P.h, g = P.g === undefined ? h : P.g;
  const Q0 = splitHeads(matmul(X, P.Wq), h), K0 = splitHeads(matmul(X, P.Wk), g), V = splitHeads(matmul(X, P.Wv), g);
  const Q = oo.rope ? Q0.map(q => rope(q, oo.rope)) : Q0;
  const K = oo.rope ? K0.map(k => rope(k, oo.rope)) : K0;
  const per = [];
  for (let j = 0; j < h; j++) {
    const kv = kvHeadOf(j, h, g);
    const mask = Array.isArray(oo.mask) && Array.isArray(oo.mask[0]) && Array.isArray(oo.mask[0][0]) ? oo.mask[j] : oo.mask;
    per.push(sdpa(Q[j], K[kv], V[kv], { mask: mask, scale: oo.scale }));
  }
  const C = mergeHeads(per.map(p => p.Z)), Y = matmul(C, P.Wo);
  return { Y: Y, st: { X: X, Q: Q, K: K, V: V, per: per, C: C, h: h, g: g, rope: oo.rope || null } };
}
function attnBackward(dY, st, P) {
  const h = st.h, g = st.g, X = st.X;
  const gWo = matmul(transpose(st.C), dY), dC = matmul(dY, transpose(P.Wo)), dZ = splitHeads(dC, h);
  const dQ = [], dK = [], dV = [];
  for (let j = 0; j < g; j++) { dK.push(zeros2(st.K[j].length, st.K[j][0].length)); dV.push(zeros2(st.V[j].length, st.V[j][0].length)); }
  for (let j = 0; j < h; j++) {
    const kv = kvHeadOf(j, h, g), A = st.per[j].A, sc = st.per[j].scale;
    const V = st.V[kv], K = st.K[kv], Q = st.Q[j];
    const nq = A.length, nk = A[0].length, dv = V[0].length, dk = Q[0].length;
    const dA = zeros2(nq, nk);
    for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
      let s = 0; for (let c = 0; c < dv; c++) s += dZ[j][i][c] * V[k][c];
      dA[i][k] = s;
    }
    for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
      const a = A[i][k]; if (!a) continue;
      for (let c = 0; c < dv; c++) dV[kv][k][c] += a * dZ[j][i][c];
    }
    const dS = zeros2(nq, nk);
    for (let i = 0; i < nq; i++) {
      let dot = 0; for (let k = 0; k < nk; k++) dot += dA[i][k] * A[i][k];
      for (let k = 0; k < nk; k++) dS[i][k] = A[i][k] * (dA[i][k] - dot) * sc;
    }
    const dQj = zeros2(nq, dk);
    for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
      const s = dS[i][k]; if (!s) continue;
      for (let c = 0; c < dk; c++) { dQj[i][c] += s * K[k][c]; dK[kv][k][c] += s * Q[i][c]; }
    }
    dQ.push(dQj);
  }
  /* RoPE is orthogonal per row: the backward pass is the inverse rotation */
  const inv = st.rope ? Object.assign({}, st.rope, { inverse: true }) : null;
  const mQ = mergeHeads(inv ? dQ.map(q => rope(q, inv)) : dQ);
  const mK = mergeHeads(inv ? dK.map(k => rope(k, inv)) : dK);
  const mV = mergeHeads(dV);
  const gWq = matmul(transpose(X), mQ), gWk = matmul(transpose(X), mK), gWv = matmul(transpose(X), mV);
  const dX = matmul(mQ, transpose(P.Wq)), b1 = matmul(mK, transpose(P.Wk)), b2 = matmul(mV, transpose(P.Wv));
  for (let i = 0; i < dX.length; i++) for (let c = 0; c < dX[0].length; c++) dX[i][c] += b1[i][c] + b2[i][c];
  return { Wq: gWq, Wk: gWk, Wv: gWv, Wo: gWo, X: dX };
}

/* ── the block ────────────────────────────────────────────────────────── */
function blockInit(d, h, dff, o) {
  const oo = Object.assign({ dk: null, dv: null, g: null, norm: "ln", pre: true, act: "gelu", gated: false, bias: true, seed: 11, scale: null }, o || {});
  const dk = oo.dk === null ? d / h : oo.dk, dv = oo.dv === null ? dk : oo.dv, g = oo.g === null ? h : oo.g;
  const attn = mhaInit(d, dk, dv, h, { seed: oo.seed, g: g, scale: oo.scale });
  const ffn = ffnInit(d, dff, { act: oo.act, gated: oo.gated, bias: oo.bias, seed: oo.seed + 1, scale: oo.scale });
  const ones = () => { const v = zeros(d); for (let j = 0; j < d; j++) v[j] = 1; return v; };
  return { attn: attn, ffn: ffn, n1: { g: ones(), b: zeros(d) }, n2: { g: ones(), b: zeros(d) },
           cfg: { d: d, h: h, dff: dff, norm: oo.norm, pre: oo.pre, dk: dk, dv: dv, g: g } };
}
function normFwd(kind, X, p) { return kind === "rms" ? rmsForward(X, p.g) : lnForward(X, p.g, p.b); }
function normBwd(kind, dY, st) { return kind === "rms" ? rmsBackward(dY, st) : lnBackward(dY, st); }
function addM(A, B) { return A.map((r, i) => r.map((v, j) => v + B[i][j])); }
function blockForward(X, P, o) {
  const oo = o || {}, k = P.cfg.norm, ao = { mask: oo.mask, rope: oo.rope, scale: oo.scale };
  if (P.cfg.pre) {
    const n1 = normFwd(k, X, P.n1), a = attnForward(n1.Y, P.attn, ao), X1 = addM(X, a.Y);
    const n2 = normFwd(k, X1, P.n2), f = ffnForward(n2.Y, P.ffn), Y = addM(X1, f.Y);
    return { Y: Y, st: { pre: true, n1: n1.st, a: a.st, X1: X1, n2: n2.st, f: f.st, attnOut: a.Y, ffnOut: f.Y } };
  }
  const a = attnForward(X, P.attn, ao), n1 = normFwd(k, addM(X, a.Y), P.n1), X1 = n1.Y;
  const f = ffnForward(X1, P.ffn), n2 = normFwd(k, addM(X1, f.Y), P.n2);
  return { Y: n2.Y, st: { pre: false, a: a.st, n1: n1.st, X1: X1, f: f.st, n2: n2.st, attnOut: a.Y, ffnOut: f.Y } };
}
function blockBackward(dY, st, P) {
  const k = P.cfg.norm;
  if (st.pre) {
    const gf = ffnBackward(dY, st.f, P.ffn), g2 = normBwd(k, gf.X, st.n2);
    const dX1 = addM(dY, g2.X);
    const ga = attnBackward(dX1, st.a, P.attn), g1 = normBwd(k, ga.X, st.n1);
    const dX = addM(dX1, g1.X);
    return { attn: ga, ffn: gf, n1: g1, n2: g2, X: dX };
  }
  const g2 = normBwd(k, dY, st.n2), gf = ffnBackward(g2.X, st.f, P.ffn);
  const dX1 = addM(g2.X, gf.X);
  const g1 = normBwd(k, dX1, st.n1), ga = attnBackward(g1.X, st.a, P.attn);
  const dX = addM(g1.X, ga.X);
  return { attn: ga, ffn: gf, n1: g1, n2: g2, X: dX };
}
function blockParams(d, h, dff, o) {
  const oo = Object.assign({ dk: null, dv: null, g: null, norm: "ln", gated: false, bias: true, attnBias: false }, o || {});
  const dk = oo.dk === null ? d / h : oo.dk, dv = oo.dv === null ? dk : oo.dv, g = oo.g === null ? h : oo.g;
  const attn = mhaParams(d, dk, dv, h, g, oo.attnBias).total;
  const ffn = ffnParams(d, dff, { gated: oo.gated, bias: oo.bias });
  const norms = 2 * (oo.norm === "rms" ? d : 2 * d);
  return { attn: attn, ffn: ffn, norms: norms, total: attn + ffn + norms };
}

/* ── a whole tiny decoder-only model ─────────────────────────────────── */
function tfInit(c) {
  const cfg = Object.assign({ V: 8, d: 16, L: 2, h: 2, dff: 32, nCtx: 16, pos: "learned", norm: "ln", pre: true,
                              gated: false, act: "gelu", tie: true, scaleEmb: false, seed: 1, ropeBase: 10000,
                              ropePairing: "adjacent", bias: true, scale: null, g: null }, c || {});
  const r = rng(cfg.seed * 1009 + 7);
  const gen = (a, b, s) => { const M = zeros2(a, b); for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) M[i][j] = randn(r) * s; return M; };
  const sE = cfg.embStd !== undefined ? cfg.embStd : (cfg.scale === null ? (cfg.scaleEmb ? 1 / Math.sqrt(cfg.d) : 1) : cfg.scale);
  const m = { cfg: cfg, E: gen(cfg.V, cfg.d, sE), blocks: [], nf: { g: zeros(cfg.d), b: zeros(cfg.d) } };
  for (let j = 0; j < cfg.d; j++) m.nf.g[j] = 1;
  m.Pos = cfg.pos === "learned" ? gen(cfg.nCtx, cfg.d, 0.5 * sE) : (cfg.pos === "sin" ? sinusoidalPE(cfg.nCtx, cfg.d) : null);
  for (let l = 0; l < cfg.L; l++) m.blocks.push(blockInit(cfg.d, cfg.h, cfg.dff,
    { norm: cfg.norm, pre: cfg.pre, act: cfg.act, gated: cfg.gated, bias: cfg.bias, seed: cfg.seed * 31 + l, g: cfg.g, scale: cfg.scale }));
  if (!cfg.tie) m.Wout = gen(cfg.d, cfg.V, 1 / Math.sqrt(cfg.d));
  if (cfg.pos === "alibi") m.slopes = alibiSlopes(cfg.h);
  return m;
}
function tfMasks(m, n) {
  const cm = causalMask(n);
  if (m.cfg.pos !== "alibi") return cm;
  return m.slopes.map(s => addMasks(cm, alibiBias(n, s)));
}
function tfForward(m, idx, o) {
  const cfg = m.cfg, n = idx.length, d = cfg.d, oo = o || {};
  const es = cfg.scaleEmb ? Math.sqrt(d) : 1;
  let Pos = m.Pos;
  if (cfg.pos === "sin" && n > Pos.length) Pos = sinusoidalPE(n, d);        // closed form: any length
  if (cfg.pos === "learned" && n > Pos.length) throw new Error("tfForward: learned positions have no row for n > nCtx = " + Pos.length);
  const X0 = zeros2(n, d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X0[i][j] = m.E[idx[i]][j] * es + (Pos ? Pos[i][j] : 0);
  const mask = oo.mask || tfMasks(m, n);
  const ro = cfg.pos === "rope" ? { base: cfg.ropeBase, pairing: cfg.ropePairing, offset: 0 } : null;
  const xs = [X0], sts = [];
  let X = X0;
  for (let l = 0; l < cfg.L; l++) { const b = blockForward(X, m.blocks[l], { mask: mask, rope: ro }); sts.push(b.st); X = b.Y; xs.push(X); }
  const nf = normFwd(cfg.norm, X, m.nf);
  const logits = cfg.tie ? matmul(nf.Y, transpose(m.E)) : matmul(nf.Y, m.Wout);
  return { logits: logits, st: { idx: idx, X0: X0, xs: xs, blocks: sts, nf: nf.st, nfY: nf.Y, mask: mask, es: es } };
}
function tfLoss(m, idx, tgt, o) {
  const f = tfForward(m, idx, o); let L = 0, c = 0;
  for (let i = 0; i < idx.length; i++) { if (tgt[i] < 0) continue; L += xent(f.logits[i], tgt[i]); c++; }
  return c ? L / c : 0;
}
function tfBackward(m, idx, tgt, o) {
  const cfg = m.cfg, f = tfForward(m, idx, o), n = idx.length, d = cfg.d, V = cfg.V;
  let cnt = 0; for (let i = 0; i < n; i++) if (tgt[i] >= 0) cnt++;
  const dLog = zeros2(n, V); let loss = 0;
  for (let i = 0; i < n; i++) {
    if (tgt[i] < 0) continue;
    const p = softmax(f.logits[i]); loss += xent(f.logits[i], tgt[i]);
    for (let k = 0; k < V; k++) dLog[i][k] = (p[k] - (k === tgt[i] ? 1 : 0)) / cnt;
  }
  const g = { E: zeros2(V, d), Pos: m.Pos && cfg.pos === "learned" ? zeros2(m.Pos.length, d) : null, blocks: [], nf: null, Wout: null };
  let dH;
  if (cfg.tie) { const t = matmul(transpose(dLog), f.st.nfY); for (let i = 0; i < V; i++) for (let j = 0; j < d; j++) g.E[i][j] += t[i][j]; dH = matmul(dLog, m.E); }
  else { g.Wout = matmul(transpose(f.st.nfY), dLog); dH = matmul(dLog, transpose(m.Wout)); }
  const gn = normBwd(cfg.norm, dH, f.st.nf); g.nf = gn;
  let dX = gn.X;
  for (let l = cfg.L - 1; l >= 0; l--) { const gb = blockBackward(dX, f.st.blocks[l], m.blocks[l]); g.blocks[l] = gb; dX = gb.X; }
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
    g.E[idx[i]][j] += dX[i][j] * f.st.es;
    if (g.Pos) g.Pos[i][j] += dX[i][j];
  }
  return { loss: cnt ? loss / cnt : 0, g: g, dX0: dX, logits: f.logits, st: f.st };
}
function tfParamList(m) {
  const out = [{ k: "E", M: m.E }];
  if (m.Pos && m.cfg.pos === "learned") out.push({ k: "Pos", M: m.Pos });
  m.blocks.forEach((B, l) => {
    ["Wq", "Wk", "Wv", "Wo"].forEach(k => out.push({ k: "b" + l + ".attn." + k, M: B.attn[k] }));
    (B.ffn.gated ? ["Wg", "Wu", "Wd"] : (B.ffn.bias ? ["W1", "b1", "W2", "b2"] : ["W1", "W2"])).forEach(k => out.push({ k: "b" + l + ".ffn." + k, M: B.ffn[k] }));
    out.push({ k: "b" + l + ".n1.g", M: B.n1.g });
    if (B.cfg.norm !== "rms") out.push({ k: "b" + l + ".n1.b", M: B.n1.b });
    out.push({ k: "b" + l + ".n2.g", M: B.n2.g });
    if (B.cfg.norm !== "rms") out.push({ k: "b" + l + ".n2.b", M: B.n2.b });
  });
  out.push({ k: "nf.g", M: m.nf.g });
  if (m.cfg.norm !== "rms") out.push({ k: "nf.b", M: m.nf.b });
  if (!m.cfg.tie) out.push({ k: "Wout", M: m.Wout });
  return out;
}
function tfGradList(g, m) {
  const out = [{ k: "E", M: g.E }];
  if (g.Pos) out.push({ k: "Pos", M: g.Pos });
  m.blocks.forEach((B, l) => {
    const gb = g.blocks[l];
    ["Wq", "Wk", "Wv", "Wo"].forEach(k => out.push({ k: "b" + l + ".attn." + k, M: gb.attn[k] }));
    (B.ffn.gated ? ["Wg", "Wu", "Wd"] : (B.ffn.bias ? ["W1", "b1", "W2", "b2"] : ["W1", "W2"])).forEach(k => out.push({ k: "b" + l + ".ffn." + k, M: gb.ffn[k] }));
    out.push({ k: "b" + l + ".n1.g", M: gb.n1.g });
    if (B.cfg.norm !== "rms") out.push({ k: "b" + l + ".n1.b", M: gb.n1.b });
    out.push({ k: "b" + l + ".n2.g", M: gb.n2.g });
    if (B.cfg.norm !== "rms") out.push({ k: "b" + l + ".n2.b", M: gb.n2.b });
  });
  out.push({ k: "nf.g", M: g.nf.g });
  if (m.cfg.norm !== "rms") out.push({ k: "nf.b", M: g.nf.b });
  if (!m.cfg.tie) out.push({ k: "Wout", M: g.Wout });
  return out;
}
/* One optimiser step over a list of matrices (2-D) or vectors (1-D), each
   with its own flat DL.OPT state; states is filled in on first use.      */
function optMats(mats, grads, states, key, hp) {
  const O = OPT[key] || OPT.adamw;
  let gnorm = 0;
  mats.forEach((p, i) => {
    const M = p.M, G = grads[i].M, is2 = Array.isArray(M[0]);
    const flatG = is2 ? [].concat.apply([], G) : G.slice();
    const flatT = is2 ? [].concat.apply([], M) : M.slice();
    for (let q = 0; q < flatG.length; q++) gnorm += flatG[q] * flatG[q];
    if (!states[i]) states[i] = O.init(flatG.length);
    const h = Object.assign({}, hp, { theta: flatT, wd: (p.k === "E" || /W[qkvo12gud]|Wout/.test(p.k)) ? (hp.wd || 0) : 0 });
    const dx = O.step(states[i], flatG, h);
    if (is2) { const w = M[0].length; for (let q = 0; q < dx.length; q++) M[Math.floor(q / w)][q % w] += dx[q]; }
    else for (let q = 0; q < dx.length; q++) M[q] += dx[q];
  });
  return Math.sqrt(gnorm);
}

/* ── counting a whole model ──────────────────────────────────────────── */
function modelParams(c) {
  const cfg = Object.assign({ V: 50257, d: 768, L: 12, h: 12, g: null, dk: null, dff: 3072, nCtx: 1024, pos: "learned",
                              norm: "ln", gated: false, bias: true, tie: true, attnBias: true }, c || {});
  const per = blockParams(cfg.d, cfg.h, cfg.dff, { dk: cfg.dk, g: cfg.g, norm: cfg.norm, gated: cfg.gated, bias: cfg.bias, attnBias: cfg.attnBias });
  const embed = cfg.V * cfg.d, pos = cfg.pos === "learned" ? cfg.nCtx * cfg.d : 0;
  const finalNorm = cfg.norm === "rms" ? cfg.d : 2 * cfg.d;
  const unembed = cfg.tie ? 0 : cfg.V * cfg.d;
  const attn = per.attn * cfg.L, ffn = per.ffn * cfg.L, norms = per.norms * cfg.L + finalNorm;
  const total = embed + pos + attn + ffn + norms + unembed;
  return { embed: embed, pos: pos, attn: attn, ffn: ffn, norms: norms, unembed: unembed, total: total,
           perBlock: per, blocks: per.total * cfg.L, nonEmbed: attn + ffn + norms, cfg: cfg };
}
/* Per-token cost at sequence length n. Forward MACs of the weight path =
   one MAC per non-embedding parameter (+ the output projection V·d, which
   is a real matmul whether or not it is tied); the attention path adds
   2·n·d per layer (Q·Kᵀ and A·V, uncausal) or n·d causal-average. Training
   = forward + backward = 3× the MACs = 6× in FLOPs: the "6·N·D" rule.    */
function modelFlops(c, n, o) {
  const P = modelParams(c), cfg = P.cfg, oo = Object.assign({ causal: true }, o || {});
  const hd = cfg.h * (cfg.dk === null ? cfg.d / cfg.h : cfg.dk);
  const weightMacs = P.nonEmbed - P.norms + cfg.V * cfg.d;
  const f = oo.causal ? 0.5 * (1 + 1 / Math.max(1, n)) : 1;
  const attnMacsPerTok = 2 * n * hd * f * cfg.L;
  const macs = weightMacs + attnMacsPerTok;
  return { weightMacs: weightMacs, attnMacs: attnMacsPerTok, macs: macs, fwdFlops: 2 * macs,
           trainFlops: 6 * macs, train6N: 6 * P.total, train6Nnon: 6 * (P.nonEmbed + cfg.V * cfg.d),
           attnShare: attnMacsPerTok / macs, params: P };
}
function noamLR(step, d, warm) {
  const s = Math.max(1, step);
  return Math.pow(d, -0.5) * Math.min(Math.pow(s, -0.5), s * Math.pow(warm, -1.5));
}
function smoothTarget(V, k, eps) {
  const q = new Array(V).fill(eps / V);
  q[k] += 1 - eps;
  return q;
}
function xentSoft(logits, q) {
  const ls = logSoftmax(logits); let L = 0;
  for (let i = 0; i < q.length; i++) if (q[i] > 0) L -= q[i] * ls[i];
  return L;
}
function smoothFloor(V, eps) {
  return entropyOf(smoothTarget(V, 0, eps));
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
    im2col, ker2col, convMatrix,
    /* [part 4 — RNNs & LSTMs] recurrence; see the SEQUENCE LAYOUT note above */
    jacFD, eigGeneral, specRad, scaleToRho, matPow, eigSym, pinvSym, ridge,
    CELL, cellList, cellParams, cellFlatten, cellUnflatten, cellGradFlatten, cellClone,
    seqRandn, seqInit, seqForward, seqLoss, seqLossAt, seqHead,
    seqBPTT, seqBPTTtrunc, seqBPTTchunk, seqNumGrad, stateJac, sliceState, addGrads,
    /* [part 5 — Attention] see THE ATTENTION LAYOUT LAW above */
    softmaxRows, softmaxJac, softmaxJacFrob, entropyOf, perplexityOf,
    splitHeads, mergeHeads, causalMask, padMask, windowMask, addMasks, maskConst,
    sdpa, mhaInit, mhaForward, mhaBackward, kvHeadOf, numGradMats,
    mhaParams, attnMacs, macCrossover, kvCacheElems, decodeIntensity,
    /* [part 6 — Transformers] see the TRANSFORMER header above sinusoidalPE */
    sinusoidalPE, peShiftMatrix, ropeFreqs, rope, alibiSlopes, alibiBias, relBucket, relBucketMatrix,
    lnForward, lnBackward, rmsForward, rmsBackward,
    ffnInit, ffnForward, ffnBackward, ffnParams, ffnWidthGated,
    attnForward, attnBackward, blockInit, blockForward, blockBackward, blockParams,
    tfInit, tfMasks, tfForward, tfLoss, tfBackward, tfParamList, tfGradList, optMats,
    modelParams, modelFlops, noamLR, smoothTarget, xentSoft, smoothFloor
  };
})();
