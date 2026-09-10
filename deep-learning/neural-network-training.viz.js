/* neural-network-training.viz.js — the visualizations on
   deep-learning/neural-network-training.html (Deep Learning · part 2).
   Loaded after ../data.js → ../notes.js → dl-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the
   page, so a section can be reordered or removed without breaking the rest.

   Every numeric primitive — activations and their derivatives, the stable
   loss family, the dense forward and backward pass, the OPTIMISERS, the
   learning-rate schedules, the normalisation layers, clipping, the EWMA and
   the exact 2x2 eigendecomposition — comes from dl-viz.js (DL.*). NOTHING
   here forks those. Page-specific numerics live in NT below.

     1  #loop-svg      one step: the arithmetic and the memory
     2  #risk-svg      empirical risk, true risk, and the surrogate
     3  #lossopt-svg   the loss from the optimiser's side
     4  #graph-svg     the computational graph, forward then backward
     5  #bpnum-svg     backprop on real numbers, audited against numGrad
     6  #admode-svg    reverse mode, forward mode, finite differences
     7  #jac-svg       the product of Jacobians
     8  #sat-svg       saturation and the backward signal
     9  #dead-svg      dying ReLU as an optimisation pathology
    10  #init-svg      the variance argument, measured forward and backward
    11  #initbad-svg   ten times and a tenth of the right scale
    12  #sgd-svg       batch, minibatch, stochastic
    13  #cond-svg      curvature, condition number, the bound on the step
    14  #saddle-svg    saddles rather than minima in high dimensions
    15  #ewma-svg      the moving average and its bias correction
    16  #opt-svg       the optimisers, racing on one landscape
    17  #wd-svg        coupled L2 against decoupled weight decay
    18  #sched-svg     the schedules, with warmup as a prefix
    19  #lrrange-svg   the learning-rate range test
    20  #batch-svg     batch size, step count, memory, and the scaling rules
    21  #prec-svg      float16, bfloat16, and loss scaling
    22  #bn-svg        BatchNorm: train mode, eval mode, and the gap
    23  #norms-svg     which axis each normalisation layer reduces over
    24  #l2-svg        weight decay as an eigenvalue rescaling
    25  #drop-svg      dropout: the ensemble and the inference rule
    26  #clip-svg      clipping by value against clipping by norm
    27  #hp-svg        random search against grid search
    28  #curve-svg     reading a training curve

   Every number these print is recomputed from the data they draw.            */

/* ══════════ page-local helpers (deliberately NOT in dl-viz.js) ══════════ */
const NT = (function () {

  /* ---- IEEE-754 binary16, simulated exactly, for the mixed-precision
     figure. JavaScript has Math.fround for binary32 and nothing for binary16,
     so this rounds a double to the nearest representable half, INCLUDING
     subnormals and the overflow to infinity. Verified in the build against
     numpy.float16 on the constants the page quotes.                        */
  const F16 = {
    MAX: 65504,
    MIN_NORMAL: Math.pow(2, -14),          // 6.103515625e−5
    MIN_SUB: Math.pow(2, -24),             // 5.9604644775390625e−8
    EPS: Math.pow(2, -10)                  // 9.765625e−4
  };
  function toF16(x) {
    if (!isFinite(x) || x === 0) return x;
    const s = Math.sign(x), a = Math.abs(x);
    if (a >= 65520) return s * Infinity;   // rounds up to 2^16, which overflows
    if (a < F16.MIN_SUB / 2) return 0;
    let q;
    if (a < F16.MIN_NORMAL) {              // subnormal: fixed quantum 2^−24
      q = Math.round(a / F16.MIN_SUB) * F16.MIN_SUB;
    } else {
      const e = Math.floor(Math.log2(a));
      const step = Math.pow(2, e - 10);    // 10 mantissa bits
      q = Math.round(a / step) * step;
      if (q >= 65520) return s * Infinity;
    }
    return s * q;
  }
  const F32 = {
    MAX: 3.4028234663852886e38, MIN_NORMAL: 1.1754943508222875e-38,
    MIN_SUB: 1.401298464324817e-45, EPS: Math.pow(2, -23)
  };

  /* ---- the toy classification problem used by the empirical-risk figure.
     A NON-MONOTONE truth, so that capacity actually buys something before it
     starts costing something.                                              */
  const TRUTH = x => DL.sigmoid(3 * Math.sin(1.3 * x) + 0.8 * x);
  function sample(N, noise, seed) {
    const r = DL.rng(seed), X = [], y = [];
    for (let i = 0; i < N; i++) {
      const x = -3 + 6 * r();
      let lab = (r() < TRUTH(x)) ? 1 : 0;
      if (r() < noise) lab = 1 - lab;
      X.push(x); y.push(lab);
    }
    return { X: X, y: y };
  }
  /* k radial bumps evenly spaced over [−3, 3] — a basis whose columns are
     nearly uncorrelated, so capacity really is capacity                      */
  function rbf(x, k) {
    const f = [], s = 6 / k * 0.8;
    for (let j = 0; j < k; j++) {
      const c = -3 + 6 * (j + 0.5) / k;
      f.push(Math.exp(-((x - c) * (x - c)) / (2 * s * s)));
    }
    return f;
  }

  /* ---- margin-parameterised surrogate losses. m = y·z with y ∈ {−1, +1};
     m > 0 means the prediction is correct. Returned as {f, df} where df is
     dℓ/dm, so |df| is the gradient magnitude the optimiser sees.             */
  const SUR = {
    ce: { key: "ce", label: "cross-entropy", color: null, f: m => DL.softplus(-m), df: m => -DL.sigmoid(-m) },
    hinge: { key: "hinge", label: "hinge", f: m => Math.max(0, 1 - m), df: m => (m < 1 ? -1 : 0) },
    sq: { key: "sq", label: "squared error", f: m => (1 - m) * (1 - m), df: m => -2 * (1 - m) },
    exp: { key: "exp", label: "exponential", f: m => Math.exp(-m), df: m => -Math.exp(-m) },
    zeroone: { key: "zeroone", label: "0–1 loss", f: m => (m > 0 ? 0 : 1), df: () => 0 }
  };

  /* ---- binary entropy in nats: the mean cross-entropy of the model that
     ignores the input entirely and predicts the base rate. NOT −log(1−p),
     which is a mistake this page calls out.                                  */
  const Hb = p => (p <= 0 || p >= 1) ? 0 : -(p * Math.log(p) + (1 - p) * Math.log(1 - p));

  /* ---- a flat-parameter quadratic and its gradient, for the landscape and
     optimiser figures. H is symmetric 2×2 given as [a, b, c].                */
  function quad(H, w) {
    const [a, b, c] = H;
    return 0.5 * (a * w[0] * w[0] + 2 * b * w[0] * w[1] + c * w[1] * w[1]);
  }
  const quadGrad = (H, w) => [H[0] * w[0] + H[1] * w[1], H[1] * w[0] + H[2] * w[1]];

  const hue = i => d3.interpolateTurbo((i * 0.6180339887) % 1);
  const say = (id, html) => d3.select(id).html(html);

  return { F16, F32, toF16, TRUTH, sample, rbf, SUR, Hb, quad, quadGrad, hue, say };
})();

/* ═════════════════ 1 · #loop-svg — one step, arithmetic and memory ═════════════════ */
(function () {
  const svg = d3.select("#loop-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const Le = document.getElementById("lp-L"), we = document.getElementById("lp-w"),
    Be = document.getElementById("lp-B"), oe = document.getElementById("lp-opt"),
    pe = document.getElementById("lp-prec");

  function draw() {
    const L = +Le.value, w = +we.value, B = +Be.value, opt = oe.value, prec = pe.value;
    document.getElementById("lp-Lv").textContent = L;
    document.getElementById("lp-wv").textContent = w;
    document.getElementById("lp-Bv").textContent = B;

    /* the layer table: L hidden layers of width w, then a 1000-way output */
    const sizes = [w].concat(new Array(L).fill(w)).concat([1000]);
    let macF = 0, P = 0, actEl = 0;
    const rows = [];
    for (let l = 0; l + 1 < sizes.length; l++) {
      const m = B * sizes[l] * sizes[l + 1];              // MACs, whole batch
      const p = sizes[l] * sizes[l + 1] + sizes[l + 1];
      const a = B * sizes[l];                             // the INPUT this layer must keep
      macF += m; P += p; actEl += a;
      if (rows.length < 5 || l + 2 === sizes.length)
        rows.push([l === sizes.length - 2 ? "layer " + (l + 1) + " (out)" : "layer " + (l + 1),
          sizes[l] + " × " + sizes[l + 1], DL.big(m), DL.big(a * (prec === "mixed" ? 2 : 4)) + "B"]);
    }
    /* ∂L/∂A is not needed for the FIRST layer unless the inputs are trained */
    const macBA = macF - B * sizes[0] * sizes[1], macBW = macF;
    const tot = macF + macBA + macBW;

    const bytesP = 4, stateBuf = (opt === "sgd" ? 0 : (opt === "mom" ? 1 : 2));
    const memP = P * bytesP, memG = P * bytesP, memS = P * bytesP * stateBuf;
    const memA = actEl * (prec === "mixed" ? 2 : 4);

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 20, b: 10 });
    const g = F.g;
    const BW = 470;

    /* ---- arithmetic bar ---- */
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("arithmetic in one step — multiply-accumulates, whole batch");
    const segsA = [
      { v: macF, c: DC.accent, t: "forward" },
      { v: macBA, c: DC.a2, t: "backward · ∂L/∂A" },
      { v: macBW, c: DC.violet, t: "backward · ∂L/∂W" }
    ];
    let x = 0;
    segsA.forEach(s => {
      const ww = BW * s.v / tot;
      g.append("rect").attr("x", x).attr("y", 12).attr("width", ww).attr("height", 30)
        .attr("fill", s.c).attr("fill-opacity", 0.75).attr("stroke", s.c);
      g.append("text").attr("x", x + ww / 2).attr("y", 31).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", "#0f1117").attr("font-weight", 600)
        .text(ww > 62 ? (100 * s.v / tot).toFixed(1) + "%" : "");
      x += ww;
    });
    DL.legend(g, segsA.map(s => ({ color: s.c, label: s.t })), BW + 22, 20, { gap: 14, font: 10.5 });

    /* ---- memory bar ---- */
    const memTot = memP + memG + memS + memA;
    g.append("text").attr("x", 0).attr("y", 78).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("memory held simultaneously — bytes");
    const segsM = [
      { v: memP, c: DC.good, t: "parameters" },
      { v: memG, c: DC.teal, t: "gradients" },
      { v: memS, c: DC.rose, t: "optimiser state" },
      { v: memA, c: DC.a2, t: "stored activations" }
    ];
    x = 0;
    segsM.forEach(s => {
      const ww = BW * s.v / memTot;
      if (ww <= 0) return;
      g.append("rect").attr("x", x).attr("y", 90).attr("width", ww).attr("height", 30)
        .attr("fill", s.c).attr("fill-opacity", 0.75).attr("stroke", s.c);
      g.append("text").attr("x", x + ww / 2).attr("y", 109).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", "#0f1117").attr("font-weight", 600)
        .text(ww > 62 ? (100 * s.v / memTot).toFixed(0) + "%" : "");
      x += ww;
    });
    DL.legend(g, segsM.map(s => ({ color: s.c, label: s.t })), BW + 22, 98, { gap: 14, font: 10.5 });

    /* ---- table ---- */
    const ty = 162, cols = [0, 120, 250, 350];
    ["", "W shape", "MACs (batch)", "activation bytes"].forEach((h, j) =>
      g.append("text").attr("x", cols[j]).attr("y", ty).attr("font-size", 10).attr("fill", DC.muted).text(h));
    const shown = rows.slice(0, 6);
    shown.forEach((r, i) => {
      r.forEach((c, j) => g.append("text").attr("x", cols[j]).attr("y", ty + 16 + i * 14)
        .attr("font-size", 10.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(c));
    });
    if (L + 1 > shown.length) g.append("text").attr("x", 0).attr("y", ty + 16 + shown.length * 14)
      .attr("font-size", 10.5).attr("fill", DC.muted).text("⋮  " + (L + 1 - shown.length) + " more layers");
    const yTot = ty + 16 + (shown.length + (L + 1 > shown.length ? 1 : 0)) * 14 + 6;
    g.append("line").attr("x1", 0).attr("x2", 430).attr("y1", yTot - 11).attr("y2", yTot - 11).attr("stroke", DC.line);
    [["total", "", DL.big(macF), DL.big(memA) + "B"]].forEach(r =>
      r.forEach((c, j) => g.append("text").attr("x", cols[j]).attr("y", yTot).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.a2).attr("font-weight", 600).text(c)));

    const k = DL.kv(g, BW + 22, 170, { keyW: 118, lead: 15 });
    k("parameters", DL.big(P), DC.good, true);
    k("step ÷ forward", (tot / macF).toFixed(3) + "×", DC.accent, true);
    k("MACs / step", DL.big(tot), DC.ink);
    k("params + state", (memP + memG + memS) / 1048576 > 1 ? ((memP + memG + memS) / 1048576).toFixed(1) + " MiB" : DL.big(memP + memG + memS) + " B", DC.rose);
    k("activations", memA / 1048576 > 1 ? (memA / 1048576).toFixed(1) + " MiB" : DL.big(memA) + " B", DC.a2);
    k("activation share", (100 * memA / memTot).toFixed(1) + "%", DC.ink);

    NT.say("#loop-readout",
      "Depth <b>" + (L + 1) + "</b> parameterised layers, width <b>" + w + "</b>, batch <b>" + B +
      "</b>. One step costs <b>" + (tot / macF).toFixed(3) + "×</b> the forward pass — the 3× rule of thumb, measured rather than assumed; it falls below 3 only because the first layer never needs <code>∂L/∂A</code>. Memory: <b>" +
      DL.big(P) + "</b> parameters need <b>" + ((memP + memG + memS) / 1048576).toFixed(1) +
      " MiB</b> for weights, gradients and " + (stateBuf === 0 ? "no" : stateBuf) + " optimiser buffer" +
      (stateBuf === 1 ? "" : "s") + ", while the stored activations need <b>" + (memA / 1048576).toFixed(1) +
      " MiB</b> (" + (100 * memA / memTot).toFixed(1) + "% of the total). Only the activation share moves when you change the batch — and for a <i>dense</i> chain like this one it is a small share, because parameters grow with width² and activations only with width. Reverse that ratio — a model whose layers are small relative to the sequence it processes — and activations dominate instead, which is the regime §26's ledger is drawn for.");
  }
  [Le, we, Be, oe, pe].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 2 · #risk-svg — empirical risk, true risk, surrogate ═════════════════ */
(function () {
  const svg = d3.select("#risk-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const ne = document.getElementById("rk-n"), ce = document.getElementById("rk-cap"),
    se = document.getElementById("rk-noise"), ue = document.getElementById("rk-sur"),
    ze = document.getElementById("rk-01");
  const CAPS = 16, TEST = NT.sample(1200, 0, 999);      // clean held-out labels

  function draw() {
    const N = +ne.value, cap = +ce.value, noise = +se.value / 100, sur = ue.value, show01 = ze.checked;
    document.getElementById("rk-nv").textContent = N;
    document.getElementById("rk-capv").textContent = cap;
    document.getElementById("rk-noisev").textContent = (noise * 100).toFixed(0) + "%";

    const TR = NT.sample(N, noise, 7);
    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 10 });
    const g = F.g;

    /* ---- left: the surrogate against the margin ---- */
    const PW = 300, PH = 250, px = 34, py = 26;
    const S = NT.SUR[sur];
    const xm = d3.scaleLinear().domain([-3, 3]).range([0, PW]);
    const ym = d3.scaleLinear().domain([0, 4]).range([PH, 0]);
    const gl = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the surrogate, against the margin m = y·z");
    DL.gridY(gl, ym, PW, 5); DL.gridX(gl, xm, PH, 7);
    DL.axisB(gl, xm, PH, 7, "margin m"); DL.axisL(gl, ym, 5, "loss");
    if (show01) {
      DL.curve(gl, [[xm(-3), ym(1)], [xm(0), ym(1)]], { stroke: DC.muted, w: 2.4, dash: "5 3" });
      DL.curve(gl, [[xm(0), ym(0)], [xm(3), ym(0)]], { stroke: DC.muted, w: 2.4, dash: "5 3" });
      gl.append("circle").attr("cx", xm(0)).attr("cy", ym(1)).attr("r", 2.6).attr("fill", DC.muted);
    }
    const ms = DL.linspace(-3, 3, 241);
    DL.curve(gl, ms.map(m => [xm(m), ym(Math.min(S.f(m), 4.2))]), { stroke: DC.accent, w: 2.4 });
    DL.curve(gl, ms.map(m => [xm(m), ym(Math.min(Math.abs(S.df(m)), 4.2))]), { stroke: DC.a2, w: 2, dash: "4 3" });
    DL.legend(gl, [{ color: DC.accent, label: S.label }, { color: DC.a2, label: "|dℓ/dm|", dash: "4 3" },
      { color: DC.muted, label: "0–1 loss", dash: "5 3" }].filter((d, i) => i < 2 || show01), 8, 12, { gap: 14, font: 10.5 });
    const slopeAt = Math.abs(S.df(-2));
    gl.append("text").attr("x", 8).attr("y", PH - 8).attr("font-size", 10).attr("fill", DC.muted)
      .text("|dℓ/dm| at m = −2 is " + (isFinite(slopeAt) ? slopeAt.toFixed(3) : "∞") + "; the 0–1 loss gives 0 here");

    /* ---- right: empirical vs true risk against capacity ---- */
    const QX = 410, QW = 300, QH = 250;
    const tr = [], te = [], ac = [];
    for (let k = 1; k <= CAPS; k++) {
      const Xtr = TR.X.map(x => NT.rbf(x, k)), Xte = TEST.X.map(x => NT.rbf(x, k));
      const m = DL.logistic(Xtr, TR.y, 1200, 3.0, 1e-5);
      tr.push(DL.logisticLoss(m, Xtr, TR.y));
      te.push(DL.logisticLoss(m, Xte, TEST.y));
      ac.push(DL.accuracy(m, Xte, TEST.y));
    }
    const xk = d3.scaleLinear().domain([1, CAPS]).range([0, QW]);
    const hi = Math.max(d3.max(tr), d3.max(te)) * 1.1;
    const yl = d3.scaleLinear().domain([0, hi]).range([QH, 0]);
    const ya = d3.scaleLinear().domain([0.4, 1]).range([QH, 0]);
    const gr = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
    g.append("text").attr("x", QX).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("what capacity does to each of the three");
    DL.gridY(gr, yl, QW, 5);
    DL.axisB(gr, xk, QH, 8, "model capacity (basis functions)"); DL.axisL(gr, yl, 5, "loss");
    gr.append("g").attr("class", "axis").attr("transform", "translate(" + QW + ",0)")
      .call(d3.axisRight(ya).ticks(4).tickFormat(d3.format(".0%")));
    DL.curve(gr, tr.map((v, i) => [xk(i + 1), yl(v)]), { stroke: DC.good, w: 2.2 });
    DL.curve(gr, te.map((v, i) => [xk(i + 1), yl(v)]), { stroke: DC.bad, w: 2.2 });
    DL.curve(gr, ac.map((v, i) => [xk(i + 1), ya(v)]), { stroke: DC.violet, w: 1.8, dash: "4 3" });
    const best = te.indexOf(d3.min(te)), bestA = ac.indexOf(d3.max(ac));
    gr.append("line").attr("x1", xk(best + 1)).attr("x2", xk(best + 1)).attr("y1", 0).attr("y2", QH)
      .attr("stroke", DC.bad).attr("stroke-opacity", 0.4).attr("stroke-dasharray", "3 3");
    gr.append("line").attr("x1", xk(cap)).attr("x2", xk(cap)).attr("y1", 0).attr("y2", QH)
      .attr("stroke", DC.ink).attr("stroke-opacity", 0.55);
    [[tr, yl, DC.good], [te, yl, DC.bad], [ac, ya, DC.violet]].forEach(([arr, sc, col]) =>
      gr.append("circle").attr("cx", xk(cap)).attr("cy", sc(arr[cap - 1])).attr("r", 4).attr("fill", col));
    DL.legend(gr, [{ color: DC.good, label: "empirical risk R̂ (train)" },
      { color: DC.bad, label: "true risk R (held out)" },
      { color: DC.violet, label: "held-out accuracy →", dash: "4 3" }], 8, 12, { gap: 14, font: 10.5 });

    NT.say("#risk-readout",
      "At capacity <b>" + cap + "</b> with N = " + N + " and " + (noise * 100).toFixed(0) +
      "% label noise: empirical risk <b>" + tr[cap - 1].toFixed(4) + "</b>, true risk <b>" +
      te[cap - 1].toFixed(4) + "</b>, held-out accuracy <b>" + (100 * ac[cap - 1]).toFixed(1) +
      "%</b>. The generalisation gap is <b>" + (te[cap - 1] - tr[cap - 1]).toFixed(4) +
      "</b>. Held-out <i>loss</i> is best at capacity <b>" + (best + 1) + "</b> (" + d3.min(te).toFixed(4) +
      ") but held-out <i>accuracy</i> peaks at capacity <b>" + (bestA + 1) + "</b> (" + (100 * d3.max(ac)).toFixed(1) +
      "%) — the two metrics disagree by " + Math.abs(bestA - best) + " step" + (Math.abs(bestA - best) === 1 ? "" : "s") +
      " of capacity, which is §02's point about the surrogate. Empirical risk falls monotonically throughout; it is the one curve that carries no information.");
  }
  [ne, ce, se, ue, ze].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 3 · #lossopt-svg — the loss from the optimiser's side ═════════════════ */
(function () {
  const svg = d3.select("#lossopt-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const ve = document.getElementById("lo-view"), Ke = document.getElementById("lo-K"),
    ee = document.getElementById("lo-eps"), pe = document.getElementById("lo-pos"),
    ge = document.getElementById("lo-log");

  function draw() {
    const view = ve.value, K = +Ke.value, eps = +ee.value, pos = +pe.value, logy = ge.checked;
    document.getElementById("lo-Kv").textContent = K;
    document.getElementById("lo-epsv").textContent = eps.toFixed(2);
    document.getElementById("lo-posv").textContent = (100 * pos).toFixed(1) + "%";

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 10 });
    const g = F.g;
    const px = 40, py = 30, PW = 300, PH = 280, QX = 415;

    if (view === "grad") {
      const keys = ["ce", "hinge", "sq", "exp"];
      const xm = d3.scaleLinear().domain([-3, 3]).range([0, PW]);
      const top = logy ? 100 : 6;
      const mk = () => logy ? d3.scaleLog().domain([1e-3, top]).range([PH, 0])
        : d3.scaleLinear().domain([0, top]).range([PH, 0]);
      [[0, "loss ℓ(m)", d => d.f], [1, "gradient magnitude |dℓ/dm|", d => d.df]].forEach(([pi, title, sel]) => {
        const gg = g.append("g").attr("transform", "translate(" + (px + pi * (QX - px)) + "," + py + ")");
        const yy = mk();
        g.append("text").attr("x", px + pi * (QX - px)).attr("y", 12).attr("font-size", 11)
          .attr("fill", DC.ink).attr("font-weight", 600).text(title);
        DL.gridY(gg, yy, PW, 5); DL.gridX(gg, xm, PH, 7);
        DL.axisB(gg, xm, PH, 7, "margin m = y·z");
        DL.axisL(gg, yy, 5, "", logy ? d3.format(".0e") : null);
        gg.append("rect").attr("x", 0).attr("y", 0).attr("width", xm(0)).attr("height", PH)
          .attr("fill", DC.bad).attr("fill-opacity", 0.05);
        gg.append("text").attr("x", 6).attr("y", PH - 6).attr("font-size", 10).attr("fill", DC.bad)
          .text("confidently wrong");
        keys.forEach((kk, i) => {
          const S = NT.SUR[kk], f = sel(S);
          const pts = DL.linspace(-3, 3, 241).map(m => {
            let v = pi === 0 ? f(m) : Math.abs(f(m));
            v = Math.min(v, top * 1.4);
            if (logy) v = Math.max(v, 1e-3);
            return [xm(m), yy(v)];
          });
          DL.curve(gg, pts, { stroke: NT.hue(i), w: 2.1 });
        });
        if (pi === 0) {
          DL.curve(gg, [[xm(-3), yy(logy ? 1 : 1)], [xm(0), yy(1)]], { stroke: DC.muted, w: 2, dash: "5 3" });
          DL.curve(gg, [[xm(0), yy(logy ? 1e-3 : 0)], [xm(3), yy(logy ? 1e-3 : 0)]], { stroke: DC.muted, w: 2, dash: "5 3" });
        }
        if (pi === 1) DL.legend(gg, keys.map((kk, i) => ({ color: NT.hue(i), label: NT.SUR[kk].label }))
          .concat([{ color: DC.muted, label: "0–1 loss: slope 0", dash: "5 3" }]), 8, 12, { gap: 14, font: 10.5 });
      });
      const at = m => keys.map(kk => NT.SUR[kk].label + " " + Math.abs(NT.SUR[kk].df(m)).toFixed(3)).join(", ");
      NT.say("#lossopt-readout",
        "Gradient magnitude at margin <b>−2</b> (confidently wrong): " + at(-2) +
        ". At margin <b>−4</b>: " + at(-4) + ". Cross-entropy saturates at exactly 1 — the most it will ever ask for from one example is a unit push. Hinge is 1 everywhere left of the margin. Squared error and the exponential loss are <b>unbounded</b>, so one badly mislabelled example can dominate the whole batch gradient. The 0–1 loss has slope <b>0</b> at every one of these points, which is why nobody optimises it.");

    } else if (view === "imb") {
      const p = pos, base = NT.Hb(p), accMaj = 1 - p;
      const bars = [
        { t: "predict the base rate " + p.toFixed(3), l: base, a: accMaj, c: DC.a2 },
        { t: "predict p = 0.5 for everything", l: Math.log(2), a: 0.5, c: DC.muted },
        { t: "a perfect model", l: 0, a: 1, c: DC.good }
      ];
      const xb = d3.scaleLinear().domain([0, Math.max(Math.log(2), base) * 1.15]).range([0, PW]);
      const gg = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
      g.append("text").attr("x", px).attr("y", 12).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("mean cross-entropy of three constant predictors");
      DL.axisB(gg, xb, 190, 5, "mean loss (nats)");
      bars.forEach((b, i) => {
        gg.append("rect").attr("x", 0).attr("y", 20 + i * 56).attr("width", Math.max(1, xb(b.l)))
          .attr("height", 28).attr("fill", b.c).attr("fill-opacity", 0.7).attr("stroke", b.c);
        gg.append("text").attr("x", 4).attr("y", 14 + i * 56).attr("font-size", 10.5).attr("fill", DC.muted).text(b.t);
        gg.append("text").attr("x", Math.max(1, xb(b.l)) + 8).attr("y", 39 + i * 56).attr("font-size", 11)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
          .text(b.l.toFixed(5) + "   acc " + (100 * b.a).toFixed(1) + "%");
      });
      /* the loss/accuracy curve as the positive rate sweeps */
      const gr = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
      const xr = d3.scaleLog().domain([0.001, 0.5]).range([0, PW]);
      const yr = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
      g.append("text").attr("x", QX).attr("y", 12).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("as the class gets rarer");
      DL.gridY(gr, yr, PW, 5);
      DL.axisB(gr, xr, PH, 4, "positive rate", d3.format(".1%")); DL.axisL(gr, yr, 5, "");
      const ps = DL.linspace(Math.log(0.001), Math.log(0.5), 160).map(Math.exp);
      DL.curve(gr, ps.map(q => [xr(q), yr(NT.Hb(q) / Math.log(2))]), { stroke: DC.a2, w: 2.2 });
      DL.curve(gr, ps.map(q => [xr(q), yr(1 - q)]), { stroke: DC.good, w: 2.2, dash: "4 3" });
      gr.append("line").attr("x1", xr(p)).attr("x2", xr(p)).attr("y1", 0).attr("y2", PH)
        .attr("stroke", DC.ink).attr("stroke-opacity", 0.5);
      DL.legend(gr, [{ color: DC.a2, label: "base-rate loss ÷ log 2" },
        { color: DC.good, label: "majority accuracy", dash: "4 3" }], 8, PH - 26, { gap: 14, font: 10.5 });
      NT.say("#lossopt-readout",
        "At a positive rate of <b>" + (100 * p).toFixed(2) + "%</b> the model that <i>ignores the input entirely</i> and predicts the base rate has a mean cross-entropy of <b>" +
        base.toFixed(5) + " nats</b> and — thresholded at 0.5 — an accuracy of <b>" + (100 * accMaj).toFixed(2) +
        "%</b>. That is a real, sharp basin of the objective, and gradient descent will find it in a handful of steps. Initialising the output bias to the log-odds <b>log(" +
        p.toFixed(4) + "/" + (1 - p).toFixed(4) + ") = " + Math.log(p / (1 - p)).toFixed(4) +
        "</b> starts the model there for free, so the optimiser spends its steps on the part of the problem that needs them.");

    } else if (view === "smooth") {
      const gg = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
      const xe = d3.scaleLinear().domain([0, 0.4]).range([0, PW]);
      const yc = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
      g.append("text").attr("x", px).attr("y", 12).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("optimal confidence  q_y = 1 − ε + ε/K");
      DL.gridY(gg, yc, PW, 5);
      DL.axisB(gg, xe, PH, 5, "smoothing ε"); DL.axisL(gg, yc, 5, "");
      const es = DL.linspace(0.0005, 0.4, 200);
      DL.curve(gg, es.map(e => [xe(e), yc(1 - e + e / K)]), { stroke: DC.accent, w: 2.4 });
      gg.append("circle").attr("cx", xe(eps)).attr("cy", yc(1 - eps + eps / K)).attr("r", 4.5).attr("fill", DC.accent);

      const gr = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
      const yg = d3.scaleLinear().domain([0, 16]).range([PH, 0]);
      g.append("text").attr("x", QX).attr("y", 12).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("optimal logit gap  log(q_y / q_j)");
      DL.gridY(gr, yg, PW, 5);
      DL.axisB(gr, xe, PH, 5, "smoothing ε"); DL.axisL(gr, yg, 5, "nats");
      const gap = e => Math.log((1 - e + e / K) / (e / K));
      DL.curve(gr, es.map(e => [xe(e), yg(Math.min(gap(e), 16.5))]), { stroke: DC.a2, w: 2.4 });
      gr.append("circle").attr("cx", xe(eps)).attr("cy", yg(Math.min(gap(eps), 16.5))).attr("r", 4.5).attr("fill", DC.a2);
      gr.append("text").attr("x", 6).attr("y", 16).attr("font-size", 10.5).attr("fill", DC.bad)
        .text("ε = 0 ⇒ gap = ∞: the loss never stops pushing");
      NT.say("#lossopt-readout",
        "With <b>K = " + K + "</b> classes and <b>ε = " + eps.toFixed(2) + "</b>, the target is " +
        (1 - eps + eps / K).toFixed(6) + " on the true class and " + (eps / K).toExponential(3) +
        " on each of the other " + (K - 1) + ". Cross-entropy is minimised where the softmax <i>equals</i> that target, so the optimal logit gap is <b>" +
        (eps > 0 ? gap(eps).toFixed(4) + " nats" : "infinite") +
        "</b>. Unsmoothed, the gap that minimises the loss is infinite and the weights grow forever for no gain in accuracy; smoothed, the model reaches a finite gap and stops. That is the whole mechanism.");

    } else {
      const gg = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
      const xz = d3.scaleLinear().domain([-160, 20]).range([0, PW + 300]);
      const yz = d3.scaleLinear().domain([-170, 10]).range([PH, 0]);
      g.append("text").attr("x", px).attr("y", 12).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("log σ(z): the stable form against the naive one, rounded to float32 at each step");
      DL.gridY(gg, yz, PW + 300, 6); DL.gridX(gg, xz, PH, 6);
      DL.axisB(gg, xz, PH, 8, "logit z"); DL.axisL(gg, yz, 6, "log σ(z)");
      const zs = DL.linspace(-160, 20, 721);
      DL.curve(gg, zs.map(z => [xz(z), yz(DL.logSigmoid(z))]), { stroke: DC.good, w: 2.4 });
      /* the SAME function written two algebraically identical ways. They die
         at different logits, because different intermediates overflow.       */
      const forms = [
        { lab: "log(1/(1+e⁻ᶻ))", col: DC.bad, f: z => Math.log(Math.fround(1 / (1 + Math.fround(Math.exp(-z))))) },
        { lab: "log(eᶻ/(1+eᶻ))", col: DC.a2, f: z => Math.log(Math.fround(Math.fround(Math.exp(z)) / (1 + Math.fround(Math.exp(z))))) }
      ];
      const deaths = forms.map(fm => {
        const pts = []; let died = null;
        zs.forEach(z => {
          const v = fm.f(z);
          if (isFinite(v)) pts.push([xz(z), yz(Math.max(v, -170))]);
          else if (died === null && z < 0) died = z;
        });
        DL.curve(gg, pts, { stroke: fm.col, w: 2, dash: "4 3" });
        if (died !== null) gg.append("line").attr("x1", xz(died)).attr("x2", xz(died))
          .attr("y1", 0).attr("y2", PH).attr("stroke", fm.col).attr("stroke-dasharray", "3 3");
        return died;
      });
      DL.legend(gg, [{ color: DC.good, label: "−softplus(−z)  (stable, never dies)" }]
        .concat(forms.map((fm, i) => ({ color: fm.col, label: fm.lab + " dies at z ≈ " + (deaths[i] === null ? "—" : deaths[i].toFixed(1)), dash: "4 3" }))),
        10, PH - 46, { gap: 15, font: 10.5 });
      const zt = -120;
      NT.say("#lossopt-readout",
        "Two <i>algebraically identical</i> naive forms, both rounded to float32 at every intermediate, die at <b>different</b> logits: <code>log(1/(1+e⁻ᶻ))</code> returns −∞ below z ≈ <b>" +
        (deaths[0] === null ? "—" : deaths[0].toFixed(1)) + "</b>, where e⁻ᶻ overflows float32, and <code>log(eᶻ/(1+eᶻ))</code> survives to z ≈ <b>" +
        (deaths[1] === null ? "—" : deaths[1].toFixed(1)) + "</b>, where eᶻ underflows past the smallest subnormal. The true value at z = " +
        zt + " is an entirely ordinary <b>" + DL.logSigmoid(zt).toFixed(3) +
        "</b>, which the stable form <code>−softplus(−z)</code> returns exactly. One infinite loss makes one infinite gradient, which makes every parameter NaN, and the run then produces NaN forever with no error raised. Use the fused logits-to-loss function; never compute a probability and take its log.");
    }
  }
  [ve, Ke, ee, pe, ge].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 4 · #graph-svg — the computational graph ═════════════════ */
(function () {
  const svg = d3.select("#graph-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const we = document.getElementById("gr-which"), te = document.getElementById("gr-t"),
    ne = document.getElementById("gr-num");

  /* A node: {id, lab, op, pa: [ids], x, y, f(vals), vjp(vals, gout) → per-parent
     contributions}. Nodes are listed in TOPOLOGICAL order, so the forward pass
     is the list order and the backward pass is the reverse. Leaves have no op. */
  function G_lr() {
    const lam = 0.05, x = [1.0, -2.0], y = 1;
    return {
      title: "L = −log σ(x·w + b) + λ‖w‖²      x = [1.0, −2.0], y = 1, λ = 0.05",
      note: "w feeds TWO children — the prediction and the penalty. Its adjoint is the sum of the two.",
      nodes: [
        { id: "w", lab: "w", val0: [0.5, 0.3], x: 60, y: 70, leaf: true, vec: true },
        { id: "b", lab: "b", val0: 0.1, x: 60, y: 250, leaf: true },
        { id: "u1", lab: "u₁", op: "x·w", pa: ["w"], x: 200, y: 70,
          f: v => DL.dot(x, v[0]), vjp: (v, go) => [x.map(xi => go * xi)] },
        { id: "u4", lab: "u₄", op: "‖w‖²", pa: ["w"], x: 200, y: 330,
          f: v => DL.dot(v[0], v[0]), vjp: (v, go) => [v[0].map(wi => go * 2 * wi)] },
        { id: "u2", lab: "u₂", op: "u₁ + b", pa: ["u1", "b"], x: 330, y: 150,
          f: v => v[0] + v[1], vjp: (v, go) => [go, go] },
        { id: "u5", lab: "u₅", op: "λ·u₄", pa: ["u4"], x: 330, y: 330,
          f: v => lam * v[0], vjp: (v, go) => [go * lam] },
        { id: "yh", lab: "ŷ", op: "σ(u₂)", pa: ["u2"], x: 450, y: 150,
          f: v => DL.sigmoid(v[0]), vjp: (v, go) => { const s = DL.sigmoid(v[0]); return [go * s * (1 - s)]; } },
        { id: "u3", lab: "u₃", op: "−log ŷ", pa: ["yh"], x: 565, y: 150,
          f: v => -Math.log(v[0]), vjp: (v, go) => [-go / v[0]] },
        { id: "L", lab: "L", op: "u₃ + u₅", pa: ["u3", "u5"], x: 680, y: 240,
          f: v => v[0] + v[1], vjp: (v, go) => [go, go] }
      ]
    };
  }
  function G_mlp() {
    const x = [1.0, 2.0], y = 1;
    return {
      title: "a 2 → 2 → 1 net on ONE example      x = [1.0, 2.0], y = 1",
      note: "the same four rules of §05, drawn as a graph rather than as a recursion.",
      nodes: [
        { id: "W1", lab: "W⁽¹⁾", val0: [[0.5, -0.4], [0.3, 0.8]], x: 55, y: 70, leaf: true, mat: true },
        { id: "b1", lab: "b⁽¹⁾", val0: [0.1, -0.2], x: 55, y: 200, leaf: true, vec: true },
        { id: "W2", lab: "W⁽²⁾", val0: [0.6, -0.5], x: 55, y: 330, leaf: true, vec: true },
        { id: "z1", lab: "z⁽¹⁾", op: "xW⁽¹⁾+b⁽¹⁾", pa: ["W1", "b1"], x: 200, y: 130,
          f: v => DL.vecmat(x, v[0]).map((z, j) => z + v[1][j]),
          vjp: (v, go) => [x.map(xi => go.map(g => xi * g)), go.slice()] },
        { id: "a1", lab: "a⁽¹⁾", op: "relu", pa: ["z1"], x: 340, y: 130,
          f: v => v[0].map(z => Math.max(z, 0)),
          vjp: (v, go) => [go.map((g, j) => v[0][j] > 0 ? g : 0)] },
        { id: "z2", lab: "z⁽²⁾", op: "a⁽¹⁾·W⁽²⁾", pa: ["a1", "W2"], x: 470, y: 230,
          f: v => DL.dot(v[0], v[1]),
          vjp: (v, go) => [v[1].map(w => go * w), v[0].map(a => go * a)] },
        { id: "L", lab: "L", op: "softplus(−z⁽²⁾)", pa: ["z2"], x: 630, y: 230,
          f: v => DL.softplus(-v[0]), vjp: (v, go) => [go * (DL.sigmoid(v[0]) - y)] }
      ]
    };
  }
  function G_share() {
    return {
      title: "L = v²·eᵛ + v      one variable, three children",
      note: "v is used by a, by b and by L itself: THREE contributions, and they add.",
      nodes: [
        { id: "v", lab: "v", val0: 1.2, x: 70, y: 200, leaf: true },
        { id: "a", lab: "a", op: "v²", pa: ["v"], x: 240, y: 80,
          f: v => v[0] * v[0], vjp: (v, go) => [go * 2 * v[0]] },
        { id: "b", lab: "b", op: "eᵛ", pa: ["v"], x: 240, y: 320,
          f: v => Math.exp(v[0]), vjp: (v, go) => [go * Math.exp(v[0])] },
        { id: "c", lab: "c", op: "a·b", pa: ["a", "b"], x: 430, y: 200,
          f: v => v[0] * v[1], vjp: (v, go) => [go * v[1], go * v[0]] },
        { id: "L", lab: "L", op: "c + v", pa: ["c", "v"], x: 620, y: 200,
          f: v => v[0] + v[1], vjp: (v, go) => [go, go] }
      ]
    };
  }
  const GRAPHS = { lr: G_lr, mlp: G_mlp, share: G_share };

  const fmtV = v => Array.isArray(v)
    ? (Array.isArray(v[0]) ? "[" + v.map(r => r.map(x => DL.fmt(x, 3)).join(" ")).join(" ; ") + "]"
      : "[" + v.map(x => DL.fmt(x, 4)).join(", ") + "]")
    : DL.fmt(v, 5);
  const addTo = (a, b) => Array.isArray(a)
    ? (Array.isArray(a[0]) ? a.map((r, i) => r.map((x, j) => x + b[i][j])) : a.map((x, i) => x + b[i]))
    : a + b;
  const zeroLike = v => Array.isArray(v)
    ? (Array.isArray(v[0]) ? v.map(r => r.map(() => 0)) : v.map(() => 0)) : 0;

  function evaluate(Gr) {
    const val = {}, adj = {}, contrib = {};
    Gr.nodes.forEach(n => { val[n.id] = n.leaf ? n.val0 : n.f(n.pa.map(p => val[p])); });
    Gr.nodes.forEach(n => { adj[n.id] = zeroLike(val[n.id]); contrib[n.id] = []; });
    const last = Gr.nodes[Gr.nodes.length - 1];
    adj[last.id] = 1;
    for (let i = Gr.nodes.length - 1; i >= 0; i--) {
      const n = Gr.nodes[i];
      if (n.leaf) continue;
      const cs = n.vjp(n.pa.map(p => val[p]), adj[n.id]);
      n.pa.forEach((p, k) => {
        adj[p] = addTo(adj[p], cs[k]);
        contrib[p].push({ from: n.id, v: cs[k] });
      });
    }
    return { val: val, adj: adj, contrib: contrib };
  }

  function draw() {
    const Gr = GRAPHS[we.value]();
    const n = Gr.nodes.length, TMAX = 2 * n;
    te.max = TMAX;
    let t = Math.min(+te.value, TMAX);
    te.value = t;
    const showNum = ne.checked;
    const fwd = t < n, idx = fwd ? t : (2 * n - 1 - t);
    const phase = t >= TMAX ? "done" : (fwd ? "forward" : "backward");
    document.getElementById("gr-tv").textContent =
      t >= TMAX ? "complete" : (fwd ? "forward " + (t + 1) + "/" + n : "backward " + (t - n + 1) + "/" + n);

    const E = evaluate(Gr);
    const F = DL.frame(svg, W, H, { l: 6, r: 6, t: 24, b: 6 });
    const g = F.g;
    g.append("text").attr("x", 4).attr("y", 0).attr("font-size", 11.5).attr("fill", DC.ink)
      .attr("font-weight", 600).text(Gr.title);

    const doneF = i => t >= n || i <= t;                     // node i has a value
    const doneB = i => !fwd && i >= idx;                     // node i has an adjoint
    const byId = {}; Gr.nodes.forEach((nd, i) => { nd._i = i; byId[nd.id] = nd; });

    /* edges */
    Gr.nodes.forEach((nd, i) => {
      if (nd.leaf) return;
      nd.pa.forEach(pid => {
        const a = byId[pid], back = !fwd && i >= idx;
        const col = back ? DC.a2 : (doneF(i) ? DC.accent : DC.line);
        const op = (back || doneF(i)) ? 0.9 : 0.35;
        if (back) DL.arrow(g, nd.x - 26, nd.y + 26, a.x + 26, a.y + 26, { color: col, w: i === idx ? 2.2 : 1.1, head: 6, op: i === idx ? 1 : 0.35 });
        else DL.arrow(g, a.x + 26, a.y, nd.x - 26, nd.y, { color: col, w: 1.3, head: 6, op: op });
      });
    });
    /* nodes */
    Gr.nodes.forEach((nd, i) => {
      const hot = (fwd && i === t) || (!fwd && i === idx);
      const hasV = doneF(i), hasA = doneB(i) || t >= TMAX;
      const gg = g.append("g").attr("transform", "translate(" + nd.x + "," + nd.y + ")");
      gg.append("rect").attr("x", -26).attr("y", -16).attr("width", 52).attr("height", 32).attr("rx", 7)
        .attr("fill", hot ? (fwd ? "rgba(91,156,255,.22)" : "rgba(255,180,84,.22)") : DC.panel2)
        .attr("stroke", hot ? (fwd ? DC.accent : DC.a2) : (nd.leaf ? DC.good : DC.line))
        .attr("stroke-width", hot ? 2.2 : 1.2);
      gg.append("text").attr("x", 0).attr("y", 5).attr("text-anchor", "middle").attr("font-size", 13)
        .attr("fill", DC.ink).attr("font-weight", 600).text(nd.lab);
      gg.append("text").attr("x", 0).attr("y", -22).attr("text-anchor", "middle").attr("font-size", 9.5)
        .attr("fill", DC.muted).text(nd.leaf ? "leaf" : nd.op);
      if (showNum && hasV) gg.append("text").attr("x", 0).attr("y", 30).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.accent)
        .text(fmtV(E.val[nd.id]));
      if (showNum && hasA) gg.append("text").attr("x", 0).attr("y", 42).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.a2)
        .text("∂L/∂ = " + fmtV(E.adj[nd.id]));
    });

    /* what is happening at this step */
    let msg;
    if (t >= TMAX) {
      const leaves = Gr.nodes.filter(nd => nd.leaf);
      msg = "<b>Complete.</b> " + leaves.map(nd => "∂L/∂" + nd.lab + " = <b>" + fmtV(E.adj[nd.id]) + "</b>").join(" · ") +
        ". Every node was visited once forward and once backward, and every EDGE contributed exactly one local partial: that is the O(#edges) cost of §08.";
    } else if (fwd) {
      const nd = Gr.nodes[t];
      msg = "<b>Forward step " + (t + 1) + ".</b> " + (nd.leaf
        ? "Node <code>" + nd.lab + "</code> is a leaf; its value is given: <b>" + fmtV(E.val[nd.id]) + "</b>."
        : "Node <code>" + nd.lab + "</code> = <code>" + nd.op + "</code> of (" + nd.pa.join(", ") + ") = <b>" +
        fmtV(E.val[nd.id]) + "</b>. This value must be KEPT — the backward pass will need it to form the local partial.");
    } else {
      const nd = Gr.nodes[idx];
      const cs = E.contrib[nd.id];
      msg = "<b>Backward step " + (t - n + 1) + ".</b> Node <code>" + nd.lab + "</code> receives " +
        (cs.length === 0 ? "the seed <b>∂L/∂L = 1</b>" :
          cs.length + " contribution" + (cs.length === 1 ? "" : "s") + " from its child" +
          (cs.length === 1 ? "" : "ren") + " — " +
          cs.map(c => "<code>" + c.from + "</code> gives " + fmtV(c.v)).join(", ") +
          (cs.length > 1 ? ", and they <b>add</b>" : "")) +
        ". Adjoint <b>∂L/∂" + nd.lab + " = " + fmtV(E.adj[nd.id]) + "</b>.";
    }
    NT.say("#graph-readout", msg + " <span style='color:var(--muted)'>" + Gr.note + "</span>");
  }
  function bump(d) { te.value = DL.clamp(+te.value + d, 0, +te.max); draw(); }
  document.getElementById("gr-next").addEventListener("click", () => bump(1));
  document.getElementById("gr-prev").addEventListener("click", () => bump(-1));
  document.getElementById("gr-all").addEventListener("click", () => { te.value = te.max; draw(); });
  we.addEventListener("input", () => { te.value = 0; draw(); });
  [te, ne].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 5 · #bpnum-svg — backprop on real numbers ═════════════════ */
(function () {
  const svg = d3.select("#bpnum-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const se = document.getElementById("bp-stage"), ae = document.getElementById("bp-act"),
    w1e = document.getElementById("bp-w11"), w2e = document.getElementById("bp-w21"),
    he = document.getElementById("bp-h");
  const X = [[1.0, 2.0], [-1.0, 0.5]], Y = [[1], [0]];

  function net(a, w11, w21) {
    const n = DL.mlpInit([2, 2, 1], { act: a, out: "sigmoid", seed: 1 });
    n.W[0] = [[w11, -0.4], [0.3, 0.8]]; n.b[0] = [0.1, -0.2];
    n.W[1] = [[w21], [-0.5]]; n.b[1] = [0.2];
    return n;
  }

  function draw() {
    const stage = +se.value, akey = ae.value, w11 = +w1e.value, w21 = +w2e.value, hx = +he.value;
    document.getElementById("bp-w11v").textContent = DL.fmt(w11, 2);
    document.getElementById("bp-w21v").textContent = DL.fmt(w21, 2);
    const N = net(akey, w11, w21), act = DL.act(akey);
    const f = DL.forward(N, X);
    const Z1 = f.z[0], A1 = f.a[1], Z2 = f.z[1];
    const yhat = Z2.map(r => [DL.sigmoid(r[0])]);
    const per = Z2.map((r, i) => DL.bce(r[0], Y[i][0]));
    const gr = DL.backward(N, X, Y);
    const D2 = yhat.map((r, i) => [(r[0] - Y[i][0]) / 2]);
    const dA1 = DL.matmul(D2, DL.transpose(N.W[1]));
    const D1 = dA1.map((r, i) => r.map((v, j) => v * act.df(Z1[i][j])));

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 22, b: 8 });
    const g = F.g;
    const title = ["forward · Z⁽¹⁾ = XW⁽¹⁾ + b⁽¹⁾, then A⁽¹⁾ = g(Z⁽¹⁾)",
      "forward · Z⁽²⁾ = A⁽¹⁾W⁽²⁾ + b⁽²⁾, ŷ = σ(Z⁽²⁾), L = mean bce",
      "backward · the seed Δ⁽²⁾ = (ŷ − Y)/N",
      "backward · ∂L/∂W⁽²⁾ = (A⁽¹⁾)ᵀΔ⁽²⁾ and ∂L/∂b⁽²⁾ = column sums of Δ⁽²⁾",
      "backward · ∂L/∂A⁽¹⁾ = Δ⁽²⁾(W⁽²⁾)ᵀ, then the gate Δ⁽¹⁾ = ∂L/∂A⁽¹⁾ ⊙ g′(Z⁽¹⁾)",
      "backward · ∂L/∂W⁽¹⁾ = XᵀΔ⁽¹⁾ and ∂L/∂b⁽¹⁾ = column sums of Δ⁽¹⁾",
      "the finite-difference audit — exact recursion against central differences"][stage];
    g.append("text").attr("x", 4).attr("y", 0).attr("font-size", 11.5).attr("fill", DC.ink)
      .attr("font-weight", 600).text(title);

    const neg = M => (i) => (M[i] && M[i].some(v => v < 0)) ? DC.bad : DC.ink;
    const M = (mat, x, y, lab, opt) => DL.matText(g, mat, x, y, Object.assign({ label: lab, dp: 4, lead: 15, size: 11 }, opt || {}));

    if (stage === 6) {
      const hh = Math.pow(10, hx);
      const num = DL.numGrad(N, X, Y, hh);
      const rows = [];
      let worst = 0;
      for (let l = 0; l < 2; l++) {
        for (let i = 0; i < N.sizes[l]; i++) for (let j = 0; j < N.sizes[l + 1]; j++) {
          const e = gr.dW[l][i][j], nn = num.dW[l][i][j];
          rows.push(["W⁽" + (l + 1) + "⁾" + (i + 1) + (j + 1), e, nn]);
          worst = Math.max(worst, Math.abs(e - nn));
        }
        for (let j = 0; j < N.sizes[l + 1]; j++) {
          const e = gr.db[l][j], nn = num.db[l][j];
          rows.push(["b⁽" + (l + 1) + "⁾" + (j + 1), e, nn]);
          worst = Math.max(worst, Math.abs(e - nn));
        }
      }
      const cols = [10, 110, 240, 370];
      ["parameter", "exact (recursion)", "central difference", "|difference|"].forEach((t, j) =>
        g.append("text").attr("x", cols[j]).attr("y", 26).attr("font-size", 10.5).attr("fill", DC.muted).text(t));
      rows.forEach((r, i) => {
        const d = Math.abs(r[1] - r[2]);
        const cells = [r[0], DL.fmt(r[1], 9), DL.fmt(r[2], 9), DL.fmtE(d, 2)];
        cells.forEach((c, j) => g.append("text").attr("x", cols[j]).attr("y", 44 + i * 15).attr("font-size", 10.5)
          .attr("font-family", "SF Mono, Menlo, monospace")
          .attr("fill", j === 3 ? (d < 1e-7 ? DC.good : DC.bad) : DC.ink).text(c));
      });
      /* the error valley: |max difference| against h */
      const gx = 500, gw = 240, gh = 190, gy = 40;
      const xs = d3.scaleLinear().domain([-13, -1]).range([0, gw]);
      const errs = [];
      for (let k = 1; k <= 13; k++) {
        const nk = DL.numGrad(N, X, Y, Math.pow(10, -k));
        let m = 0;
        for (let l = 0; l < 2; l++) {
          gr.dW[l].forEach((row, i) => row.forEach((v, j) => m = Math.max(m, Math.abs(v - nk.dW[l][i][j]))));
          gr.db[l].forEach((v, j) => m = Math.max(m, Math.abs(v - nk.db[l][j])));
        }
        errs.push([-k, Math.max(m, 1e-17)]);
      }
      const ys = d3.scaleLog().domain([1e-13, 1e-1]).range([gh, 0]).clamp(true);
      const gg = g.append("g").attr("transform", "translate(" + gx + "," + gy + ")");
      DL.gridY(gg, ys, gw, 5);
      DL.axisB(gg, xs, gh, 6, "log₁₀ h", d => "1e" + d);
      DL.axisL(gg, ys, 5, "max |error|", d3.format(".0e"));
      DL.curve(gg, errs.map(e => [xs(e[0]), ys(e[1])]), { stroke: DC.a2, w: 2.2 });
      errs.forEach(e => gg.append("circle").attr("cx", xs(e[0])).attr("cy", ys(e[1])).attr("r", 2.4).attr("fill", DC.a2));
      const cur = errs.find(e => e[0] === hx);
      if (cur) gg.append("circle").attr("cx", xs(cur[0])).attr("cy", ys(cur[1])).attr("r", 5.5)
        .attr("fill", "none").attr("stroke", DC.accent).attr("stroke-width", 2);
      const bestI = errs.reduce((a, e, i) => e[1] < errs[a][1] ? i : a, 0);
      gg.append("text").attr("x", 6).attr("y", 12).attr("font-size", 10).attr("fill", DC.muted)
        .text("valley bottom at h = 1e" + errs[bestI][0]);
      NT.say("#bpnum-readout",
        "At h = <b>1e" + hx + "</b> the largest disagreement between the exact recursion and a central difference over all <b>" +
        rows.length + "</b> parameters is <b>" + DL.fmtE(worst, 2) +
        "</b>. Sweeping h traces the characteristic valley: truncation error falls as h² on the right, floating-point cancellation grows as ε/h on the left, and the minimum here is <b>" +
        DL.fmtE(errs[bestI][1], 2) + "</b> at h = 1e" + errs[bestI][0] +
        ". A check that lands anywhere in this valley is a <b>pass</b>; a check that is flat at 10⁻³ for every h is a <b>bug in the backward pass</b>, and one that is fine at 10⁻⁴ and terrible at 10⁻¹⁰ is just floating point.");
      return;
    }

    const XT = DL.transpose(X);
    if (stage === 0) {
      M([["1.0", "2.0"], ["−1.0", "0.5"]], 20, 56, "X   (2 × 2)");
      M(N.W[0], 170, 56, "W⁽¹⁾   (2 × 2)");
      M([N.b[0]], 320, 56, "b⁽¹⁾");
      M(Z1, 20, 150, "Z⁽¹⁾ = X W⁽¹⁾ + b⁽¹⁾", { colorOf: neg(Z1) });
      M(A1, 200, 150, "A⁽¹⁾ = " + act.label + "(Z⁽¹⁾)");
      const k = DL.kv(g, 430, 62, { keyW: 176, lead: 16 });
      k("Z⁽¹⁾₁₁ = (1.0)(" + DL.fmt(w11, 2) + ") + (2.0)(0.3) + 0.1", DL.fmt(Z1[0][0], 4), DC.accent, true);
      k("Z⁽¹⁾₂₁ = (−1.0)(" + DL.fmt(w11, 2) + ") + (0.5)(0.3) + 0.1", DL.fmt(Z1[1][0], 4), Z1[1][0] < 0 ? DC.bad : DC.ink, true);
      k("Z⁽¹⁾₁₂ = (1.0)(−0.4) + (2.0)(0.8) − 0.2", DL.fmt(Z1[0][1], 4), DC.ink);
      k("Z⁽¹⁾₂₂ = (−1.0)(−0.4) + (0.5)(0.8) − 0.2", DL.fmt(Z1[1][1], 4), DC.ink);
      const dead = Z1.flat().filter(v => v <= 0).length;
      NT.say("#bpnum-readout", "Z⁽¹⁾ has <b>" + dead + "</b> of its 4 entries at or below zero. With " + act.label +
        ", a non-positive pre-activation has derivative <b>" + DL.fmt(act.df(-0.25), 4) +
        "</b>, so that path's gradient is multiplied by that number on the way back — exactly zero for ReLU, which is stage 4's gate.");
    } else if (stage === 1) {
      M(A1, 20, 60, "A⁽¹⁾");
      M(N.W[1], 170, 60, "W⁽²⁾   (2 × 1)");
      M([N.b[1]], 300, 60, "b⁽²⁾");
      M(Z2, 20, 160, "Z⁽²⁾   the logits", { dp: 5 });
      M(yhat, 150, 160, "ŷ = σ(Z⁽²⁾)", { dp: 7 });
      M([[1], [0]], 300, 160, "Y", { dp: 0 });
      const k = DL.kv(g, 400, 66, { keyW: 210, lead: 16 });
      k("Z⁽²⁾₁ = " + DL.fmt(A1[0][0], 3) + "·" + DL.fmt(w21, 2) + " + " + DL.fmt(A1[0][1], 3) + "·(−0.5) + 0.2", DL.fmt(Z2[0][0], 6), DC.accent, true);
      k("Z⁽²⁾₂ = " + DL.fmt(A1[1][0], 3) + "·" + DL.fmt(w21, 2) + " + " + DL.fmt(A1[1][1], 3) + "·(−0.5) + 0.2", DL.fmt(Z2[1][0], 6), DC.accent, true);
      k("ℓ₁ = softplus(−z₁)   (y = 1)", DL.fmt(per[0], 7), DC.a2);
      k("ℓ₂ = softplus(+z₂)   (y = 0)", DL.fmt(per[1], 7), DC.a2);
      k("L = (ℓ₁ + ℓ₂)/2", DL.fmt(gr.loss, 7), DC.good, true);
      k("log 2, the coin-flip loss", DL.fmt(Math.log(2), 7), DC.muted);
      NT.say("#bpnum-readout", "Mean loss <b>" + DL.fmt(gr.loss, 7) + "</b> against a coin-flip baseline of log 2 = 0.6931472. Example 1 (y = 1) has loss <b>" +
        DL.fmt(per[0], 7) + "</b> and example 2 (y = 0) has <b>" + DL.fmt(per[1], 7) +
        "</b>; a per-example loss above log 2 means the model is on the <i>wrong</i> side for that example.");
    } else if (stage === 2) {
      M(yhat, 20, 70, "ŷ", { dp: 7 });
      M([[1], [0]], 140, 70, "Y", { dp: 0 });
      M(D2, 250, 70, "Δ⁽²⁾ = (ŷ − Y)/N", { dp: 7 });
      const k = DL.kv(g, 20, 190, { keyW: 250, lead: 17 });
      k("Δ⁽²⁾₁ = (" + DL.fmt(yhat[0][0], 7) + " − 1)/2", DL.fmt(D2[0][0], 7), DC.a2, true);
      k("Δ⁽²⁾₂ = (" + DL.fmt(yhat[1][0], 7) + " − 0)/2", DL.fmt(D2[1][0], 7), DC.a2, true);
      k("σ′(z₁) — NOT used anywhere", DL.fmt(DL.sigmoid(Z2[0][0]) * (1 - DL.sigmoid(Z2[0][0])), 7), DC.muted);
      g.append("text").attr("x", 20).attr("y", 300).attr("font-size", 12).attr("fill", DC.good)
        .text("The derivative of the output non-linearity does not appear. That is the whole reason");
      g.append("text").attr("x", 20).attr("y", 318).attr("font-size", 12).attr("fill", DC.good)
        .text("cross-entropy pairs with a sigmoid and squared error does not (part 1 §17).");
      NT.say("#bpnum-readout", "The seed is <b>(ŷ − Y)/N</b> and nothing else. The 1/N is the <i>mean</i> in the loss; drop it and every gradient is N times too large, which looks exactly like a learning rate N times too high. σ′(z₁) = <b>" +
        DL.fmt(DL.sigmoid(Z2[0][0]) * (1 - DL.sigmoid(Z2[0][0])), 7) + "</b> is printed only to show it is absent.");
    } else if (stage === 3) {
      M(DL.transpose(A1), 20, 70, "(A⁽¹⁾)ᵀ   (2 × 2)");
      M(D2, 190, 70, "Δ⁽²⁾   (2 × 1)", { dp: 7 });
      M(gr.dW[1], 330, 70, "∂L/∂W⁽²⁾   (2 × 1)", { dp: 7 });
      M([gr.db[1]], 500, 70, "∂L/∂b⁽²⁾", { dp: 7 });
      const k = DL.kv(g, 20, 190, { keyW: 300, lead: 17 });
      k("(" + DL.fmt(A1[0][0], 3) + ")(" + DL.fmt(D2[0][0], 7) + ") + (" + DL.fmt(A1[1][0], 3) + ")(" + DL.fmt(D2[1][0], 7) + ")", DL.fmt(gr.dW[1][0][0], 7), DC.a2, true);
      k("(" + DL.fmt(A1[0][1], 3) + ")(" + DL.fmt(D2[0][0], 7) + ") + (" + DL.fmt(A1[1][1], 3) + ")(" + DL.fmt(D2[1][0], 7) + ")", DL.fmt(gr.dW[1][1][0], 7), DC.a2, true);
      k("Δ⁽²⁾₁ + Δ⁽²⁾₂  (the bias sums down the batch)", DL.fmt(gr.db[1][0], 7), DC.violet, true);
      NT.say("#bpnum-readout", "∂L/∂W⁽²⁾ has shape <b>2 × 1</b>, the shape of W⁽²⁾ — always. The weight gradient is a product of the layer's <i>input</i> activations with the layer's <i>output</i> deltas, which is why a layer whose inputs are all near zero gets a near-zero weight gradient no matter how wrong the model is. The bias gradient is the column sum, so it is the only gradient that does not see the activations at all.");
    } else if (stage === 4) {
      M(D2, 20, 70, "Δ⁽²⁾", { dp: 7 });
      M([N.W[1].map(r => r[0])], 140, 70, "(W⁽²⁾)ᵀ");
      M(dA1, 280, 70, "∂L/∂A⁽¹⁾", { dp: 7 });
      M(Z1, 20, 190, "Z⁽¹⁾", { colorOf: neg(Z1) });
      M(Z1.map(r => r.map(z => act.df(z))), 170, 190, "g′(Z⁽¹⁾)", { dp: 3 });
      M(D1, 330, 190, "Δ⁽¹⁾ = ∂L/∂A⁽¹⁾ ⊙ g′(Z⁽¹⁾)", { dp: 7 });
      let killed = 0, kept = 0;
      Z1.forEach((r, i) => r.forEach((z, j) => { if (Math.abs(D1[i][j]) < 1e-12 && Math.abs(dA1[i][j]) > 1e-12) killed++; else kept++; }));
      g.append("text").attr("x", 20).attr("y", 300).attr("font-size", 12).attr("fill", killed ? DC.bad : DC.muted)
        .text(killed ? (killed + " of 4 backward signals annihilated by the gate — the ReLU is off there") :
          "no signal is annihilated: " + act.label + " has a non-zero derivative everywhere");
      NT.say("#bpnum-readout", "With " + act.label + ", <b>" + killed + " of 4</b> entries of ∂L/∂A⁽¹⁾ are multiplied by zero and destroyed. " +
        (killed ? "The largest one destroyed is <b>" + DL.fmtE(d3.max(Z1.map((r, i) => r.map((z, j) => Math.abs(D1[i][j]) < 1e-12 ? Math.abs(dA1[i][j]) : 0)).flat()), 3) + "</b>. " : "") +
        "Switch to leaky ReLU and the gate multiplies by " + DL.fmt(DL.act("leaky").df(-1), 3) + " instead of 0, so the signal is attenuated rather than erased. That single change is the whole of §11.");
    } else {
      M(XT, 20, 70, "Xᵀ   (2 × 2)");
      M(D1, 160, 70, "Δ⁽¹⁾   (2 × 2)", { dp: 7 });
      M(gr.dW[0], 360, 70, "∂L/∂W⁽¹⁾   (2 × 2)", { dp: 7 });
      M([gr.db[0]], 360, 170, "∂L/∂b⁽¹⁾", { dp: 7 });
      const k = DL.kv(g, 20, 200, { keyW: 320, lead: 17 });
      [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(([i, j]) =>
        k("(" + DL.fmt(XT[i][0], 1) + ")(" + DL.fmt(D1[0][j], 7) + ") + (" + DL.fmt(XT[i][1], 1) + ")(" + DL.fmt(D1[1][j], 7) + ")",
          DL.fmt(gr.dW[0][i][j], 7), DC.a2, true));
      NT.say("#bpnum-readout", "Every entry of ∂L/∂W⁽¹⁾ is a sum <i>over the batch</i>: the batch axis is contracted away, which is why the gradient has the shape of W and not of anything batch-sized. Where a row of Δ⁽¹⁾ has been zeroed by the gate, that example contributes nothing to <i>any</i> weight in that column — the dead unit does not lose one number, it loses one example entirely.");
    }
  }
  [se, ae, w1e, w2e, he].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 6 · #admode-svg — reverse, forward, finite differences ═════════════════ */
(function () {
  const svg = d3.select("#admode-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const ie = document.getElementById("ad-din"), oe = document.getElementById("ad-dout"),
    pe = document.getElementById("ad-prec"), se = document.getElementById("ad-scheme");

  /* a real function with a known derivative, so the error is MEASURED */
  const f64 = x => Math.exp(Math.sin(x));
  const f32 = x => Math.fround(Math.exp(Math.sin(Math.fround(x))));
  const dtrue = x => Math.cos(x) * Math.exp(Math.sin(x));

  function draw() {
    const nin = Math.pow(10, +ie.value), nout = Math.pow(10, +oe.value);
    const prec = pe.value, scheme = se.value;
    document.getElementById("ad-dinv").textContent = DL.big(nin);
    document.getElementById("ad-doutv").textContent = DL.big(nout);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 46, py = 34, PW = 285, PH = 265, QX = 435;

    /* ---- left: sweeps needed, against the number of outputs ---- */
    const gl = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("sweeps needed — one per input, or one per output");
    const xs = d3.scaleLog().domain([1, 1e8]).range([0, PW]);
    const ys = d3.scaleLog().domain([1, 1e8]).range([PH, 0]);
    DL.gridY(gl, ys, PW, 5); DL.gridX(gl, xs, PH, 5);
    DL.axisB(gl, xs, PH, 5, "number of outputs", d3.format(".0e"));
    DL.axisL(gl, ys, 5, "sweeps", d3.format(".0e"));
    DL.curve(gl, [[xs(1), ys(nin)], [xs(1e8), ys(nin)]], { stroke: DC.a2, w: 2.4 });
    DL.curve(gl, [[xs(1), ys(1)], [xs(1e8), ys(1e8)]], { stroke: DC.accent, w: 2.4 });
    gl.append("circle").attr("cx", xs(nout)).attr("cy", ys(nin)).attr("r", 4.5).attr("fill", DC.a2);
    gl.append("circle").attr("cx", xs(nout)).attr("cy", ys(nout)).attr("r", 4.5).attr("fill", DC.accent);
    gl.append("line").attr("x1", xs(nout)).attr("x2", xs(nout)).attr("y1", 0).attr("y2", PH)
      .attr("stroke", DC.ink).attr("stroke-opacity", 0.45);
    gl.append("circle").attr("cx", xs(nin)).attr("cy", ys(nin)).attr("r", 3.4)
      .attr("fill", "none").attr("stroke", DC.good).attr("stroke-width", 1.6);
    gl.append("text").attr("x", xs(nin) + 6).attr("y", ys(nin) - 6).attr("font-size", 9.5)
      .attr("fill", DC.good).text("crossover: n_out = n_in");
    DL.legend(gl, [{ color: DC.a2, label: "forward mode: n_in sweeps" },
      { color: DC.accent, label: "reverse mode: n_out sweeps" }], 8, 14, { gap: 14, font: 10.5 });

    /* ---- right: the measured finite-difference error ---- */
    const gr = g.append("g").attr("transform", "translate(" + (QX) + "," + py + ")");
    g.append("text").attr("x", QX).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("finite-difference relative error, measured");
    const ff = prec === "32" ? f32 : f64;
    const eps = prec === "32" ? Math.pow(2, -24) : Math.pow(2, -53);
    const x0 = 1.3, T = dtrue(x0);
    const pts = [];
    for (let k = 0; k <= 17; k += 0.25) {
      const h = Math.pow(10, -k);
      const d = scheme === "central" ? (ff(x0 + h) - ff(x0 - h)) / (2 * h) : (ff(x0 + h) - ff(x0)) / h;
      pts.push([-k, DL.clamp(Math.abs(d - T) / Math.abs(T), 1e-18, 10)]);
    }
    const xh = d3.scaleLinear().domain([-17, 0]).range([0, PW]);
    const ye = d3.scaleLog().domain([1e-13, 10]).range([PH, 0]).clamp(true);
    DL.gridY(gr, ye, PW, 6); DL.gridX(gr, xh, PH, 6);
    DL.axisB(gr, xh, PH, 6, "log₁₀ step h", d => "1e" + d);
    DL.axisL(gr, ye, 6, "relative error", d3.format(".0e"));
    DL.curve(gr, pts.map(p => [xh(p[0]), ye(p[1])]), { stroke: DC.violet, w: 2.2 });
    const best = pts.reduce((a, p) => p[1] < a[1] ? p : a, pts[0]);
    gr.append("circle").attr("cx", xh(best[0])).attr("cy", ye(best[1])).attr("r", 5).attr("fill", DC.good);
    gr.append("text").attr("x", xh(best[0]) + 8).attr("y", ye(best[1]) + 4).attr("font-size", 10)
      .attr("fill", DC.good).text("best " + DL.fmtE(best[1], 1) + " at h = 1e" + best[0]);
    const pred = scheme === "central" ? Math.pow(eps, 1 / 3) : Math.pow(eps, 1 / 2);
    gr.append("line").attr("x1", xh(Math.log10(pred))).attr("x2", xh(Math.log10(pred)))
      .attr("y1", 0).attr("y2", PH).attr("stroke", DC.a2).attr("stroke-dasharray", "4 3");
    gr.append("text").attr("x", xh(Math.log10(pred)) + 5).attr("y", 14).attr("font-size", 9.5)
      .attr("fill", DC.a2).text("theory " + (scheme === "central" ? "ε^⅓" : "ε^½"));
    gr.append("text").attr("x", 8).attr("y", PH - 24).attr("font-size", 10).attr("fill", DC.muted)
      .text("← cancellation, slope −1");
    gr.append("text").attr("x", PW - 130).attr("y", PH - 24).attr("font-size", 10).attr("fill", DC.muted)
      .text("truncation, slope " + (scheme === "central" ? "2 →" : "1 →"));

    const win = nout < nin ? "reverse" : (nout > nin ? "forward" : "neither");
    const ratio = nout < nin ? nin / nout : nout / nin;
    NT.say("#admode-readout",
      "With <b>" + DL.big(nin) + "</b> inputs and <b>" + DL.big(nout) + "</b> outputs, " +
      (win === "neither" ? "the two modes cost the same — the crossover is exactly n_out = n_in."
        : "<b>" + win + " mode</b> wins by a factor of <b>" + DL.big(ratio) + "</b>.") +
      " Training a network is the extreme left of this plot: one scalar loss, millions of parameters. " +
      "Meanwhile the " + (scheme === "central" ? "central" : "forward") + " difference in " +
      (prec === "32" ? "single" : "double") + " precision bottoms out at a relative error of <b>" +
      DL.fmtE(best[1], 2) + "</b> at h = 1e" + best[0] + ", against a predicted optimum of h ≈ " +
      DL.fmtE(pred, 1) + ". That is about <b>" + Math.max(0, Math.round(-Math.log10(best[1]))) +
      "</b> correct significant digits, from arithmetic carrying " +
      (prec === "32" ? "7" : "16") + " — and it costs one evaluation per parameter to get them.");
  }
  [ie, oe, pe, se].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 7 · #jac-svg — the product of Jacobians ═════════════════ */
(function () {
  const svg = d3.select("#jac-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, LMAX = 60;
  const ne = document.getElementById("jc-n"), ge = document.getElementById("jc-g"),
    ae = document.getElementById("jc-act"), me = document.getElementById("jc-mode"),
    be = document.getElementById("jc-band");

  function draw() {
    const n = +ne.value, gain = +ge.value, akey = ae.value, tied = me.value === "tied", band = be.checked;
    document.getElementById("jc-nv").textContent = n;
    document.getElementById("jc-gv").textContent = DL.fmt(gain, 2);
    const act = DL.act(akey);
    const sd = gain * Math.sqrt(2 / n);
    const r = DL.rng(11);

    /* build the stack once, so every trajectory sees the SAME network */
    const Ws = [], gates = [];
    const W0 = DL.zeros2(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) W0[i][j] = sd * DL.randn(r);
    for (let l = 0; l < LMAX; l++) {
      if (tied) Ws.push(W0);
      else {
        const M = DL.zeros2(n, n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] = sd * DL.randn(r);
        Ws.push(M);
      }
      /* a plausible pre-activation at this depth: unit second moment */
      gates.push(new Array(n).fill(0).map(() => act.df(DL.randn(r) * Math.SQRT2)));
    }
    const WT = Ws.map(M => DL.transpose(M));

    const TRJ = 5, curves = [];
    for (let k = 0; k < TRJ; k++) {
      const rr = DL.rng(101 + k);
      let d = new Array(n).fill(0).map(() => DL.randn(rr));
      const nn = Math.hypot.apply(null, d);
      d = d.map(x => x / nn);
      const track = [1];
      for (let l = LMAX - 1; l >= 0; l--) {
        d = DL.vecmat(d.map((x, i) => x * gates[l][i]), WT[l]);
        track.push(Math.hypot.apply(null, d) || 1e-300);
      }
      curves.push(track);
    }
    /* the measured per-layer factor, geometric mean over the last trajectory */
    const last = curves[0];
    const cbar = Math.pow(last[LMAX] / last[0], 1 / LMAX);
    /* one layer's Jacobian spectrum, exactly (power iteration on JᵀJ) */
    const J = WT[0].map((row, i) => row.map((v, j) => gates[0][j] * v));
    const specJ = DL.specNorm(J, 300, 3);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 52, py = 34, PW = 470, PH = 280;
    const gl = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("‖∂L/∂A‖ after L backward layers — five random starting directions");
    const x = d3.scaleLinear().domain([0, LMAX]).range([0, PW]);
    const lo = 1e-24, hiv = 1e24;
    const y = d3.scaleLog().domain([lo, hiv]).range([PH, 0]).clamp(true);
    DL.gridY(gl, y, PW, 8); DL.gridX(gl, x, PH, 6);
    DL.axisB(gl, x, PH, 6, "layers traversed (backwards)");
    DL.axisL(gl, y, 8, "‖gradient‖", d3.format(".0e"));
    if (band) {
      [[3.4e38, "float32 max", DC.bad], [1.18e-38, "float32 min normal", DC.bad],
       [65504, "float16 max", DC.a2], [6e-8, "float16 min subnormal", DC.a2]].forEach(([v, lab, c]) => {
        if (v > lo && v < hiv) {
          gl.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(v)).attr("y2", y(v))
            .attr("stroke", c).attr("stroke-dasharray", "4 4").attr("stroke-opacity", 0.6);
          gl.append("text").attr("x", PW - 4).attr("y", y(v) - 3).attr("text-anchor", "end")
            .attr("font-size", 9).attr("fill", c).text(lab);
        }
      });
    }
    curves.forEach((tr, k) => DL.curve(gl, tr.map((v, l) => [x(l), y(DL.clamp(v, lo, hiv))]),
      { stroke: DC.accent, w: k === 0 ? 2.2 : 1.2, op: k === 0 ? 1 : 0.45 }));
    DL.curve(gl, DL.linspace(0, LMAX, 61).map(l => [x(l), y(DL.clamp(Math.pow(cbar, l), lo, hiv))]),
      { stroke: DC.a2, w: 1.8, dash: "5 4" });
    DL.legend(gl, [{ color: DC.accent, label: "measured, actual backward sweeps" },
      { color: DC.a2, label: "c̄ᴸ with the measured c̄ = " + DL.fmt(cbar, 4), dash: "5 4" }],
      10, 14, { gap: 15, font: 10.5 });

    /* the spectrum panel */
    const k = DL.kv(g, 560, 50, { keyW: 128, lead: 16 });
    k("width n", String(n), DC.ink);
    k("weight sd σ", DL.fmtE(sd, 3), DC.ink);
    k("‖W‖₂ predicted 2σ√n", DL.fmt(2 * sd * Math.sqrt(n), 4), DC.muted);
    k("‖W⁽¹⁾‖₂ measured", DL.fmt(DL.specNorm(Ws[0], 300, 2), 4), DC.violet, true);
    k("‖J⁽¹⁾‖₂ with the gate", DL.fmt(specJ, 4), DC.violet, true);
    k("measured typical c̄", DL.fmt(cbar, 4), DC.a2, true);
    k("worst-case bound c̄ᴸ", DL.fmtE(Math.pow(specJ, LMAX), 2), DC.bad);
    k("measured at L = 60", DL.fmtE(last[LMAX], 2), DC.accent, true);
    k("weights", tied ? "tied (recurrent)" : "fresh per layer", DC.muted);
    const spread = d3.max(curves.map(c => c[LMAX])) / d3.max([1e-300, d3.min(curves.map(c => c[LMAX]))]);

    NT.say("#jac-readout",
      "Per-layer spectral norm <b>" + DL.fmt(specJ, 4) + "</b> (the worst case, over sixty layers: " +
      DL.fmtE(Math.pow(specJ, LMAX), 2) + ") against a measured <i>typical</i> factor of <b>" + DL.fmt(cbar, 4) +
      "</b>. The gradient after 60 layers is <b>" + DL.fmtE(last[LMAX], 2) + "</b>, so the spectral bound overstates the growth by a factor of <b>" +
      DL.fmtE(Math.pow(specJ, LMAX) / Math.max(last[LMAX], 1e-300), 1) +
      "</b> — the bound is over the worst direction and a random gradient is not in it. " +
      (tied ? "With <b>tied</b> weights the five trajectories agree closely: the same matrix every layer means the product is governed by one eigenvalue and there is no averaging."
        : "With a <b>fresh</b> matrix per layer the five trajectories spread over a factor of " + DL.fmtE(spread, 1) +
        ", because the log of the per-layer factor performs a random walk.") +
      " " + (cbar < 0.97 ? "<b>Vanishing:</b> " : (cbar > 1.03 ? "<b>Exploding:</b> " : "<b>Held:</b> ")) +
      "at this gain the signal is multiplied by " + DL.fmt(cbar, 4) + " per layer.");
  }
  [ne, ge, ae, me, be].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 8 · #sat-svg — saturation and the backward signal ═════════════════ */
(function () {
  const svg = d3.select("#sat-svg");
  if (svg.empty()) return;
  const W = 760, H = 390;
  const ae = document.getElementById("st-act"), se = document.getElementById("st-sd"),
    ue = document.getElementById("st-mu"), Le = document.getElementById("st-L"),
    ce = document.getElementById("st-cmp");

  /* Moments of g′ under z ~ 𝒩(mu, sd²), by numerical integration on a fine
     grid. NOT read off the curve; integrated.
     𝔼[g′]  is what the MEAN backward signal is multiplied by.
     𝔼[g′²] is what the backward VARIANCE is multiplied by, and it is the one
            that enters the recursion of §12 — so it is the one panel 3 uses. */
  function mom(key, mu, sd, pow) {
    const a = DL.act(key);
    let acc = 0, Z = 0;
    for (let t = -8; t <= 8; t += 0.002) {
      const w = Math.exp(-0.5 * t * t), d = a.df(mu + sd * t);
      acc += w * (pow === 2 ? d * d : d); Z += w;
    }
    return acc / Z;
  }
  const expDeriv = (k, mu, sd) => mom(k, mu, sd, 1);
  /* the weight-variance constant the STANDARD rule of §13 assigns to each
     unit: n·σ_w² = 2 for the rectifier family, 1 for the rest. The per-layer
     backward variance factor is then c·𝔼[g′²], and a value of exactly 1 is
     what "the initialisation is right for this activation" means.            */
  const CRULE = { relu: 2, leaky: 2, prelu: 2, elu: 2, gelu: 2, geluTanh: 2, silu: 2, softplus: 2,
    tanh: 1, sigmoid: 1, hardtanh: 1, identity: 1, abs: 2 };

  function draw() {
    const key = ae.value, sd = +se.value, mu = +ue.value, L = +Le.value, cmp = ce.checked;
    document.getElementById("st-sdv").textContent = DL.fmt(sd, 1);
    document.getElementById("st-muv").textContent = DL.fmt(mu, 1);
    document.getElementById("st-Lv").textContent = L;
    const a = DL.act(key);
    const gbar = expDeriv(key, mu, sd), gRelu = expDeriv("relu", mu, sd);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const PW = 220, PH = 270, py = 36;
    const xs = [30, 285, 540];
    const lim = Math.max(6, Math.abs(mu) + 3 * sd);
    const zx = d3.scaleLinear().domain([-lim, lim]).range([0, PW]);

    /* --- panel 1: g and g′ --- */
    const g1 = g.append("g").attr("transform", "translate(" + xs[0] + "," + py + ")");
    g.append("text").attr("x", xs[0]).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(a.label + ": g and g′");
    const zs = DL.linspace(-lim, lim, 301);
    const dmax = d3.max(zs.map(z => Math.abs(a.df(z))));
    const yv = d3.scaleLinear().domain([-1.4, Math.max(1.4, dmax * 1.15)]).range([PH, 0]);
    DL.gridY(g1, yv, PW, 5);
    DL.axisB(g1, zx, PH, 5, "pre-activation z"); DL.axisL(g1, yv, 5, "");
    /* the band where g′ > 0.1 max */
    const good = zs.filter(z => Math.abs(a.df(z)) > 0.1 * dmax);
    if (good.length) g1.append("rect").attr("x", zx(d3.min(good))).attr("y", 0)
      .attr("width", Math.max(1, zx(d3.max(good)) - zx(d3.min(good)))).attr("height", PH)
      .attr("fill", DC.good).attr("fill-opacity", 0.07);
    DL.curve(g1, zs.map(z => [zx(z), yv(DL.clamp(a.f(z), -1.4, 1.4))]), { stroke: DC.muted, w: 1.6 });
    DL.curve(g1, zs.map(z => [zx(z), yv(a.df(z))]), { stroke: DC.accent, w: 2.4 });
    DL.legend(g1, [{ color: DC.muted, label: "g (clipped)" }, { color: DC.accent, label: "g′" }],
      6, 12, { gap: 14, font: 10 });

    /* --- panel 2: the density × g′ --- */
    const g2 = g.append("g").attr("transform", "translate(" + xs[1] + "," + py + ")");
    g.append("text").attr("x", xs[1]).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("where z actually sits × g′");
    const dens = z => Math.exp(-0.5 * ((z - mu) / sd) * ((z - mu) / sd)) / (sd * Math.sqrt(2 * Math.PI));
    const dmaxP = d3.max(zs.map(dens));
    const yd = d3.scaleLinear().domain([0, Math.max(dmaxP, dmax) * 1.15]).range([PH, 0]);
    DL.gridY(g2, yd, PW, 5);
    DL.axisB(g2, zx, PH, 5, "pre-activation z"); DL.axisL(g2, yd, 5, "");
    DL.curve(g2, zs.map(z => [zx(z), yd(dens(z))]), { stroke: DC.violet, w: 1.8, fill: DC.violet, fillOp: 0.12 });
    DL.curve(g2, zs.map(z => [zx(z), yd(Math.max(0, a.df(z)))]), { stroke: DC.accent, w: 2 });
    DL.curve(g2, zs.map(z => [zx(z), yd(dens(z) * Math.max(0, a.df(z)) / Math.max(1e-9, dmaxP) * dmaxP)]),
      { stroke: DC.a2, w: 2.4, fill: DC.a2, fillOp: 0.16 });
    g2.append("text").attr("x", 6).attr("y", 14).attr("font-size", 11).attr("fill", DC.a2)
      .attr("font-weight", 600).text("𝔼[g′] = " + DL.fmt(gbar, 4));
    DL.legend(g2, [{ color: DC.violet, label: "density of z" }, { color: DC.accent, label: "g′" },
      { color: DC.a2, label: "their product" }], 6, 30, { gap: 13, font: 10 });

    /* --- panel 3: the per-layer backward VARIANCE factor, to the power L --- */
    const c = CRULE[key] === undefined ? 1 : CRULE[key];
    const fac = c * mom(key, mu, sd, 2);
    const facRelu = 2 * mom("relu", mu, sd, 2);
    const g3 = g.append("g").attr("transform", "translate(" + xs[2] + "," + py + ")");
    g.append("text").attr("x", xs[2]).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("(c·𝔼[g′²])^{L/2}: what survives L layers");
    const xL = d3.scaleLinear().domain([0, 60]).range([0, PW]);
    const yL = d3.scaleLog().domain([1e-24, 1e6]).range([PH, 0]).clamp(true);
    DL.gridY(g3, yL, PW, 6); DL.gridX(g3, xL, PH, 5);
    DL.axisB(g3, xL, PH, 5, "depth L"); DL.axisL(g3, yL, 6, "", d3.format(".0e"));
    [[1.18e-38, "f32", DC.bad], [6e-8, "f16", DC.a2]].forEach(([v, lab, c]) => {
      if (v > 1e-24) {
        g3.append("line").attr("x1", 0).attr("x2", PW).attr("y1", yL(v)).attr("y2", yL(v))
          .attr("stroke", c).attr("stroke-dasharray", "4 4").attr("stroke-opacity", 0.6);
        g3.append("text").attr("x", PW - 3).attr("y", yL(v) - 3).attr("text-anchor", "end")
          .attr("font-size", 9).attr("fill", c).text(lab + " floor");
      }
    });
    const amp = (f, l) => Math.pow(f, l / 2);          // norm ratio, not variance ratio
    DL.curve(g3, DL.linspace(0, 60, 121).map(l => [xL(l), yL(DL.clamp(amp(fac, l), 1e-24, 1e6))]),
      { stroke: DC.accent, w: 2.4 });
    if (cmp) DL.curve(g3, DL.linspace(0, 60, 121).map(l => [xL(l), yL(DL.clamp(amp(facRelu, l), 1e-24, 1e6))]),
      { stroke: DC.good, w: 1.8, dash: "5 3" });
    g3.append("line").attr("x1", 0).attr("x2", PW).attr("y1", yL(1)).attr("y2", yL(1))
      .attr("stroke", DC.muted).attr("stroke-opacity", 0.5);
    g3.append("circle").attr("cx", xL(L)).attr("cy", yL(DL.clamp(amp(fac, L), 1e-24, 1e6)))
      .attr("r", 4.5).attr("fill", DC.accent);
    g3.append("text").attr("x", 6).attr("y", 28).attr("font-size", 10.5).attr("fill", DC.accent)
      .attr("font-weight", 600).text("per-layer factor c·𝔼[g′²] = " + DL.fmt(fac, 4) + "  (c = " + c + ")");
    DL.legend(g3, [{ color: DC.accent, label: a.label + " at its own rule" }]
      .concat(cmp ? [{ color: DC.good, label: "ReLU at 2/n, same z", dash: "5 3" }] : []),
      6, 12, { gap: 14, font: 10 });

    const surv = amp(fac, L), survR = amp(facRelu, L);
    NT.say("#sat-readout",
      "With z ~ 𝒩(" + DL.fmt(mu, 1) + ", " + DL.fmt(sd * sd, 2) + "), the moments of " + a.label +
      "′ — integrated over that density, not read off the peak — are 𝔼[g′] = <b>" + DL.fmt(gbar, 5) +
      "</b> and 𝔼[g′²] = <b>" + DL.fmt(mom(key, mu, sd, 2), 5) + "</b>. The second is the one the variance recursion of §12 uses; combined with this unit's standard weight rule (c = " +
      c + ") the per-layer backward factor is <b>" + DL.fmt(fac, 5) + "</b>" +
      (Math.abs(fac - 1) < 0.02 ? " — essentially exactly 1, which is what a matched rule and activation looks like"
        : (fac < 1 ? " — below 1, so the signal decays" : " — above 1, so the signal grows")) +
      ". Over <b>" + L + "</b> layers the gradient <i>norm</i> is multiplied by <b>" + DL.fmtE(surv, 2) + "</b>" +
      (surv < 6e-8 ? ", which is below float16's smallest subnormal: in half precision that gradient is literally zero" :
        (surv < 1.18e-38 ? ", which is below float32's smallest normal number" : "")) +
      ". A rectifier at 2/n on the same pre-activations gives <b>" + DL.fmt(facRelu, 5) + "</b> per layer and <b>" +
      DL.fmtE(survR, 2) + "</b> over the stack. Note what the sliders do: widening z leaves the <b>rectifier</b> untouched, because its derivative is scale-free, while it steadily kills tanh and the sigmoid; shifting the <i>mean</i> of z negative kills the rectifier too, which is §11.");
  }
  [ae, se, ue, Le, ce].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 9 · #dead-svg — units dying in real time ═════════════════ */
(function () {
  const svg = d3.select("#dead-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, STEPS = 260, NDATA = 120;
  const le = document.getElementById("dd-lr"), ae = document.getElementById("dd-act"),
    be = document.getElementById("dd-b"), we = document.getElementById("dd-warm"),
    ne = document.getElementById("dd-w");

  /* two interleaved spirals: not linearly separable, so the hidden layers have
     to do real work and dead units genuinely cost capacity                    */
  const D = (function (seed) {
    const r = DL.rng(seed), X = [], Y = [];
    for (let i = 0; i < NDATA; i++) {
      const k = i % 2, t = 1.2 + 3.6 * (i / NDATA);
      const th = t + k * Math.PI + 0.25 * DL.randn(r);
      X.push([t * Math.cos(th) / 5 + 0.05 * DL.randn(r), t * Math.sin(th) / 5 + 0.05 * DL.randn(r)]);
      Y.push([k]);
    }
    return { X: X, Y: Y };
  })(4);

  function run(lr, width, akey, b0, warm) {
    const net = DL.mlpInit([2, width, width, 1], { act: akey, out: "sigmoid", seed: 21 });
    net.b[0] = new Array(width).fill(b0);
    net.b[1] = new Array(width).fill(b0);
    const a = DL.act(akey), d1 = [], d2 = [], froz = [], loss = [];
    for (let t = 0; t < STEPS; t++) {
      const f = DL.forward(net, D.X), cnt = [0, 0];
      let fr = 0;
      for (let l = 0; l < 2; l++) {
        const Z = f.z[l];
        for (let j = 0; j < width; j++) {
          let alive = false;
          for (let i = 0; i < Z.length && !alive; i++) if (Z[i][j] > 0) alive = true;
          if (!alive) {
            cnt[l]++;
            let gs = 0;
            for (let i = 0; i < Z.length; i++) gs += Math.abs(a.df(Z[i][j]));
            if (gs < 1e-12) fr++;                    // EXACTLY zero gradient
          }
        }
      }
      d1.push(100 * cnt[0] / width); d2.push(100 * cnt[1] / width);
      froz.push(100 * fr / (2 * width));
      const g = DL.backward(net, D.X, D.Y);
      loss.push(g.loss);
      DL.sgdStep(net, g, (warm > 0 && t < warm) ? lr * (t + 1) / warm : lr);
    }
    const mono = x => { for (let i = 1; i < x.length; i++) if (x[i] < x[i - 1] - 1e-9) return false; return true; };
    return { d1: d1, d2: d2, froz: froz, loss: loss, m1: mono(d1), m2: mono(d2) };
  }

  function draw() {
    const lr = Math.pow(10, +le.value), akey = ae.value, b0 = +be.value,
      warm = +we.value, width = +ne.value;
    document.getElementById("dd-lrv").textContent = DL.sig(lr, 3);
    document.getElementById("dd-bv").textContent = DL.fmt(b0, 2);
    document.getElementById("dd-warmv").textContent = warm;
    document.getElementById("dd-wv").textContent = width;
    const A = run(lr, width, akey, b0, warm);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const py = 34, PW = 310, PH = 275;
    const xt = d3.scaleLinear().domain([0, STEPS - 1]).range([0, PW]);

    const g1 = g.append("g").attr("transform", "translate(46," + py + ")");
    g.append("text").attr("x", 46).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("units inactive on EVERY training example");
    const yd = d3.scaleLinear().domain([0, 100]).range([PH, 0]);
    DL.gridY(g1, yd, PW, 5);
    DL.axisB(g1, xt, PH, 5, "training step"); DL.axisL(g1, yd, 5, "% of that layer", d => d + "%");
    if (warm > 0) {
      g1.append("rect").attr("x", 0).attr("y", 0).attr("width", xt(Math.min(warm, STEPS - 1)))
        .attr("height", PH).attr("fill", DC.accent).attr("fill-opacity", 0.08);
      g1.append("text").attr("x", 4).attr("y", PH - 6).attr("font-size", 9.5).attr("fill", DC.accent).text("warmup");
    }
    DL.curve(g1, A.d1.map((v, i) => [xt(i), yd(v)]), { stroke: DC.bad, w: 2.4 });
    DL.curve(g1, A.d2.map((v, i) => [xt(i), yd(v)]), { stroke: DC.a2, w: 1.9 });
    DL.curve(g1, A.froz.map((v, i) => [xt(i), yd(v)]), { stroke: DC.violet, w: 1.8, dash: "5 3" });
    DL.legend(g1, [
      { color: DC.bad, label: "hidden layer 1 — inputs are FIXED data" },
      { color: DC.a2, label: "hidden layer 2 — inputs move" },
      { color: DC.violet, label: "receiving exactly zero gradient", dash: "5 3" }],
      8, 14, { gap: 14, font: 10.5 });
    g1.append("text").attr("x", 8).attr("y", PH - 8).attr("font-size", 10)
      .attr("fill", A.m1 ? DC.bad : DC.good)
      .text("layer 1: " + (A.m1 ? "monotone — nothing recovers" : "non-monotone — units recover"));

    const g2 = g.append("g").attr("transform", "translate(430," + py + ")");
    g.append("text").attr("x", 430).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the loss for that same run");
    const PW2 = 285;
    const fin = A.loss.filter(isFinite);
    const yl = d3.scaleLog().domain([Math.max(1e-5, d3.min(fin) * 0.6), Math.max(1.2, d3.max(fin) * 1.1)])
      .range([PH, 0]).clamp(true);
    const xt2 = d3.scaleLinear().domain([0, STEPS - 1]).range([0, PW2]);
    DL.gridY(g2, yl, PW2, 5);
    DL.axisB(g2, xt2, PH, 5, "training step");
    DL.axisL(g2, yl, 5, "mean loss", d3.format(".0e"));
    g2.append("line").attr("x1", 0).attr("x2", PW2).attr("y1", yl(Math.log(2))).attr("y2", yl(Math.log(2)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    g2.append("text").attr("x", PW2 - 2).attr("y", yl(Math.log(2)) - 4).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", DC.muted).text("log 2 — a coin flip");
    DL.curve(g2, A.loss.map((v, i) => [xt2(i), yl(isFinite(v) ? v : 1.2)]), { stroke: DC.accent, w: 2.4 });
    const nanAt = A.loss.findIndex(v => !isFinite(v));
    if (nanAt >= 0) {
      g2.append("line").attr("x1", xt2(nanAt)).attr("x2", xt2(nanAt)).attr("y1", 0).attr("y2", PH)
        .attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
      g2.append("text").attr("x", xt2(nanAt) + 5).attr("y", 16).attr("font-size", 10).attr("fill", DC.bad)
        .text("loss became NaN at step " + nanAt);
    }
    const k = DL.kv(g2, 6, PH - 62, { keyW: 168, lead: 15, size: 10.5 });
    k("layer 1 dead, final", DL.fmt(A.d1[STEPS - 1], 1) + "%", DC.bad, true);
    k("layer 2 dead, final", DL.fmt(A.d2[STEPS - 1], 1) + "%", DC.a2, true);
    k("frozen (zero gradient)", DL.fmt(A.froz[STEPS - 1], 1) + "%", DC.violet, true);
    k("final loss", DL.sig(A.loss[STEPS - 1], 4), DC.accent);

    NT.say("#dead-readout",
      "At η = <b>" + DL.sig(lr, 3) + "</b> with " + DL.act(akey).label + ": hidden layer 1 ends with <b>" +
      DL.fmt(A.d1[STEPS - 1], 1) + "%</b> of its units inactive on every example (it started at " +
      DL.fmt(A.d1[0], 1) + "%), hidden layer 2 with <b>" + DL.fmt(A.d2[STEPS - 1], 1) +
      "%</b>, and <b>" + DL.fmt(A.froz[STEPS - 1], 1) +
      "%</b> of all hidden units are receiving <i>exactly zero</i> gradient. Final loss <b>" +
      DL.sig(A.loss[STEPS - 1], 4) + "</b>. " +
      "<b>The two layers behave differently and the difference is the real lesson.</b> Layer 1 reads the fixed data, so a unit dead there has a zero gradient AND a fixed input: it is a genuine fixed point, and its curve is " +
      (A.m1 ? "<b>monotone</b> — nothing ever comes back" : "non-monotone here, because " + DL.act(akey).label + " still passes a small gradient") +
      ". Layer 2 reads layer 1's output, which keeps moving, so a dead unit there can be pushed back above zero by the layer below even while its own weights are frozen — its curve goes up and down. " +
      (A.froz[STEPS - 1] > 0 ? "" : "With " + DL.act(akey).label + " the frozen count is <b>zero</b>: units go quiet but never stop learning. ") +
      "And note the loss: it is still falling while a large fraction of the network has been destroyed.");
  }
  [le, ae, be, we, ne].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 10 · #init-svg — the variance argument, measured ═════════════════ */
(function () {
  const svg = d3.select("#init-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const re = document.getElementById("in-rule"), ae = document.getElementById("in-act"),
    we = document.getElementById("in-w"), Le = document.getElementById("in-L"),
    de = document.getElementById("in-dir");
  const NB = 128;                                     // batch used for the sampling

  const sdOf = (rule, nin, nout) =>
    rule === "he" ? Math.sqrt(2 / nin) :
    rule === "fanin" ? Math.sqrt(1 / nin) :
    rule === "glorot" ? Math.sqrt(2 / (nin + nout)) : 1;

  /* Sample ONE random network of the requested shape and MEASURE, layer by
     layer, the mean squared pre-activation, the mean squared activation, and
     the mean squared backward signal. The backward sweep uses the SAME
     pre-activations the forward sweep produced — the gate is not resampled. */
  function sample(rule, akey, n, L, seed) {
    const a = DL.act(akey), r = DL.rng(seed), sd = sdOf(rule, n, n);
    let h = DL.zeros2(NB, n);
    for (let i = 0; i < NB; i++) for (let j = 0; j < n; j++) h[i][j] = DL.randn(r);
    const Ws = [], Zs = [], z2 = [], a2 = [], zSamples = [];
    for (let l = 0; l < L; l++) {
      const M = DL.zeros2(n, n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] = sd * DL.randn(r);
      Ws.push(M);
      const Z = DL.matmul(h, M);
      Zs.push(Z);
      const zf = Z.reduce((s2, row) => s2.concat(row), []);
      z2.push(DL.mean(zf.map(v => v * v)));
      if (l === 0 || l === Math.floor((L - 1) / 2) || l === L - 1) zSamples.push({ l: l, z: zf });
      h = DL.applyEl(Z, x => a.f(x));
      const af = h.reduce((s2, row) => s2.concat(row), []);
      a2.push(DL.mean(af.map(v => v * v)));
    }
    let d = DL.zeros2(NB, n);
    for (let i = 0; i < NB; i++) for (let j = 0; j < n; j++) d[i][j] = DL.randn(r);
    const d2 = new Array(L).fill(0);
    for (let l = L - 1; l >= 0; l--) {
      /* gate on THIS layer's own pre-activations, then push through Wᵀ */
      d = d.map((row, i) => row.map((v, j) => v * a.df(Zs[l][i][j])));
      d = DL.matmul(d, DL.transpose(Ws[l]));
      const df = d.reduce((s2, row) => s2.concat(row), []);
      d2[l] = DL.mean(df.map(v => v * v));
    }
    return { z2: z2, a: a2, d: d2, zs: zSamples, sd: sd };
  }

  /* The prediction the §12 algebra makes, iterated as a genuine one-dimensional
     recursion rather than assumed constant:
        forward   v_{l+1} = n·σ_w²·𝔼[g(z)²],  z ~ 𝒩(0, v_l)
        backward  u_l     = n·σ_w²·𝔼[g′(z)²]·u_{l+1},  same z
     Integrated on a fine grid, seeded from the MEASURED first layer.         */
  function moment(akey, v, pow) {
    const a = DL.act(akey), sd = Math.sqrt(Math.max(v, 1e-300));
    let acc = 0, Z = 0;
    for (let t = -8; t <= 8; t += 0.004) {
      const w = Math.exp(-0.5 * t * t), x = sd * t;
      const q = (pow === 2) ? a.df(x) : a.f(x);
      acc += w * q * q; Z += w;
    }
    return acc / Z;
  }
  function predict(rule, akey, n, L, v1) {
    const c = n * sdOf(rule, n, n) * sdOf(rule, n, n);
    const v = [v1], fwd = [], gates = [];
    for (let l = 0; l < L; l++) {
      fwd.push(moment(akey, v[l], 1));
      gates.push(c * moment(akey, v[l], 2));
      v.push(c * fwd[l]);
    }
    const bwd = new Array(L).fill(1);
    for (let l = L - 2; l >= 0; l--) bwd[l] = bwd[l + 1] * gates[l + 1];
    return { fwd: fwd, bwd: bwd, c: c, gates: gates };
  }

  function draw() {
    const rule = re.value, akey = ae.value, n = +we.value, L = +Le.value, dir = de.value;
    document.getElementById("in-wv").textContent = n;
    document.getElementById("in-Lv").textContent = L;
    const S = sample(rule, akey, n, L, 17);
    const a = DL.act(akey);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 56, py = 34, PW = 430, PH = 300;

    if (dir === "hist") {
      const bw = 218, bh = 120;
      g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("pre-activation histograms at three depths, with the band where g′ > 0.1·max shaded");
      const dmax = d3.max(DL.linspace(-8, 8, 401).map(z => Math.abs(a.df(z))));
      const gz = DL.linspace(-8, 8, 401).filter(z => Math.abs(a.df(z)) > 0.1 * dmax);
      S.zs.forEach((sm, k) => {
        const lim = Math.max(4, d3.max(sm.z.map(Math.abs)) * 1.05);
        const gx = px + k * (bw + 28), gy = 46 + 0;
        const gg = g.append("g").attr("transform", "translate(" + gx + "," + gy + ")");
        const xz = d3.scaleLinear().domain([-lim, lim]).range([0, bw]);
        const bins = d3.bin().domain([-lim, lim]).thresholds(41)(sm.z);
        const ym = d3.scaleLinear().domain([0, d3.max(bins, b => b.length) || 1]).range([bh, 0]);
        gg.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", DC.muted)
          .text("layer " + (sm.l + 1) + " · rms " + DL.fmtE(Math.sqrt(DL.mean(sm.z.map(v => v * v))), 2));
        if (gz.length) gg.append("rect").attr("x", Math.max(0, xz(d3.min(gz)))).attr("y", 0)
          .attr("width", Math.max(1, Math.min(bw, xz(d3.max(gz))) - Math.max(0, xz(d3.min(gz)))))
          .attr("height", bh).attr("fill", DC.good).attr("fill-opacity", 0.1);
        gg.selectAll("rect.b").data(bins).join("rect").attr("class", "b")
          .attr("x", d => xz(d.x0)).attr("y", d => ym(d.length))
          .attr("width", d => Math.max(0.5, xz(d.x1) - xz(d.x0) - 0.5))
          .attr("height", d => bh - ym(d.length)).attr("fill", DC.accent).attr("fill-opacity", 0.65);
        DL.axisB(gg, xz, bh, 4, "z");
      });
      /* and the same three for the ACTIVATIONS */
      const gz2 = g.append("g");
      g.append("text").attr("x", px).attr("y", 214).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("mean squared activation by layer — the quantity the rule tries to hold at 1");
      const gg2 = g.append("g").attr("transform", "translate(" + px + ",232)");
      const xL = d3.scaleLinear().domain([1, L]).range([0, 640]);
      const lo = Math.min(1e-12, d3.min(S.a) || 1e-12), hi = Math.max(1e6, d3.max(S.a) || 1);
      const yA = d3.scaleLog().domain([Math.max(lo, 1e-20), hi]).range([120, 0]).clamp(true);
      DL.gridY(gg2, yA, 640, 4);
      DL.axisB(gg2, xL, 120, 6, "layer"); DL.axisL(gg2, yA, 4, "", d3.format(".0e"));
      gg2.append("line").attr("x1", 0).attr("x2", 640).attr("y1", yA(1)).attr("y2", yA(1))
        .attr("stroke", DC.good).attr("stroke-dasharray", "4 3");
      DL.curve(gg2, S.a.map((v, l) => [xL(l + 1), yA(DL.clamp(v, 1e-20, hi))]), { stroke: DC.accent, w: 2.2 });
      NT.say("#init-readout",
        "Rule <b>" + re.options[re.selectedIndex].text.trim() + "</b>, width " + n + ", " + a.label +
        ": weight sd <b>" + DL.fmtE(S.sd, 3) + "</b>. Pre-activation rms goes <b>" +
        S.zs.map(sm => DL.fmtE(Math.sqrt(DL.mean(sm.z.map(v => v * v))), 2)).join("</b> → <b>") +
        "</b> from layer 1 to layer " + L + ". The shaded band is where this activation still passes a tenth of its maximum derivative; a histogram sliding out of it is saturation, and a histogram collapsing onto zero is the signal dying.");
      return;
    }

    const series = dir === "fwd" ? S.a : S.d;
    const P = predict(rule, akey, n, L, S.z2[0]);
    const theory = dir === "fwd" ? P.fwd : P.bwd.map(v => v * series[L - 1] / (P.bwd[L - 1] || 1));
    const label = dir === "fwd" ? "mean squared activation 𝔼[a²]" : "mean squared backward signal 𝔼[Δ²]";
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(label + " at every layer — measured on a sampled network");
    const gg = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    const xL = d3.scaleLinear().domain([1, L]).range([0, PW]);
    const finite = series.filter(v => isFinite(v) && v > 0);
    const lo = Math.max(1e-30, (d3.min(finite) || 1e-3) * 0.4), hi = Math.max(4, (d3.max(finite) || 1) * 2.5);
    const y = d3.scaleLog().domain([lo, hi]).range([PH, 0]).clamp(true);
    DL.gridY(gg, y, PW, 6); DL.gridX(gg, xL, PH, 6);
    DL.axisB(gg, xL, PH, 6, "layer"); DL.axisL(gg, y, 6, "", d3.format(".0e"));
    gg.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(1)).attr("y2", y(1))
      .attr("stroke", DC.good).attr("stroke-dasharray", "4 3");
    gg.append("text").attr("x", 4).attr("y", y(1) - 4).attr("font-size", 9.5).attr("fill", DC.good).text("1 — what the rule aims for");
    const c = P.c;
    const fac = Math.pow(theory[L - 1] / theory[0], 1 / Math.max(1, L - 1));
    DL.curve(gg, theory.map((v, l) => [xL(l + 1), y(DL.clamp(v, lo, hi))]), { stroke: DC.a2, w: 1.8, dash: "5 4" });
    DL.curve(gg, series.map((v, l) => [xL(l + 1), y(DL.clamp(v, lo, hi))]), { stroke: DC.accent, w: 2.4 });
    series.forEach((v, l) => gg.append("circle").attr("cx", xL(l + 1)).attr("cy", y(DL.clamp(v, lo, hi)))
      .attr("r", 2.2).attr("fill", DC.accent));
    DL.legend(gg, [{ color: DC.accent, label: "measured" },
      { color: DC.a2, label: "the §12 recursion, per-layer " + DL.fmt(fac, 4), dash: "5 4" }],
      10, 14, { gap: 14, font: 10.5 });

    const k = DL.kv(g, 520, 60, { keyW: 150, lead: 16 });
    k("weight sd σ_w", DL.fmtE(S.sd, 3), DC.ink);
    k("n·σ_w²", DL.fmt(c, 4), DC.ink);
    k("first layer", DL.fmtE(series[0], 3), DC.accent);
    k("last layer", DL.fmtE(series[L - 1], 3), DC.accent, true);
    k("ratio last / first", DL.fmtE(series[L - 1] / series[0], 2), DC.a2, true);
    k("per-layer, measured", DL.fmt(Math.pow(series[L - 1] / series[0], 1 / (L - 1)), 4), DC.violet, true);
    k("per-layer, predicted", DL.fmt(fac, 4), DC.muted);

    const gm = Math.pow(series[L - 1] / series[0], 1 / (L - 1));
    NT.say("#init-readout",
      "Rule <b>" + re.options[re.selectedIndex].text.trim() + "</b> with " + a.label + " at width " + n +
      " and depth " + L + ": the " + (dir === "fwd" ? "mean squared activation" : "mean squared backward signal") +
      " goes from <b>" + DL.fmtE(series[0], 3) + "</b> at the first layer to <b>" + DL.fmtE(series[L - 1], 3) +
      "</b> at the last — a total factor of <b>" + DL.fmtE(series[L - 1] / series[0], 2) +
      "</b>, or <b>" + DL.fmt(gm, 4) + "</b> per layer. " +
      (Math.abs(Math.log(gm)) < 0.05
        ? "That is essentially 1: the rule is doing its job."
        : (gm < 1 ? "That is below 1, so the signal is dying — extend the depth slider and watch it leave the plot."
          : "That is above 1, so the signal is growing without bound.")) +
      " The dashed line is what the second-moment recursion of §12 predicts, iterated layer by layer rather than assumed constant; where the measured curve leaves it, one of §13's four assumptions has failed — most often the infinite-width one, which is why the gap widens as you narrow the width slider." +
      (rule === "glorot" ? " <b>Note:</b> every layer here is square, and for a square layer 2/(n_in + n_out) IS 1/n_in — the fan-average rule and the fan-in rule are the same number, and this figure cannot tell them apart. §13's table uses a deliberately lopsided layer for that reason."
        : (rule === "unit" && akey === "tanh" ? " <b>Note the trap:</b> with a unit Gaussian and tanh the FORWARD curve is almost perfectly flat — every unit is pinned at ±1, so 𝔼[a²] ≈ 1 by saturation rather than by health. Switch to the backward view and the same network explodes by twenty-eight orders of magnitude. A flat forward curve is necessary and nowhere near sufficient."
          : "")));
  }
  [re, ae, we, Le, de].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 11 · #initbad-svg — ten times and a tenth ═════════════════ */
(function () {
  const svg = d3.select("#initbad-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, NDATA = 100;
  const se = document.getElementById("ib-scales"), Le = document.getElementById("ib-L"),
    we = document.getElementById("ib-w"), Te = document.getElementById("ib-T"),
    le = document.getElementById("ib-lr");

  const D = (function (seed) {
    const r = DL.rng(seed), X = [], Y = [];
    for (let i = 0; i < NDATA; i++) {
      const k = i % 2, t = 1.2 + 3.6 * (i / NDATA), th = t + k * Math.PI + 0.25 * DL.randn(r);
      X.push([t * Math.cos(th) / 5, t * Math.sin(th) / 5]);
      Y.push([k]);
    }
    return { X: X, Y: Y };
  })(9);

  function build(scale, L, w) {
    const sizes = [2].concat(new Array(L).fill(w)).concat([1]);
    const net = DL.mlpInit(sizes, { act: "relu", out: "sigmoid", seed: 31 });
    for (let l = 0; l < net.L; l++) {
      const s = scale * Math.sqrt(2 / sizes[l]);
      const r = DL.rng(1000 + l);
      for (let i = 0; i < sizes[l]; i++) for (let j = 0; j < sizes[l + 1]; j++) net.W[l][i][j] = s * DL.randn(r);
    }
    return net;
  }
  function profile(net) {
    const f = DL.forward(net, D.X);
    return f.z.slice(0, net.L - 1).map(Z => DL.mean(Z.reduce((s, r) => s.concat(r), []).map(v => v * v)));
  }
  function train(net, T, lr) {
    const loss = [];
    for (let t = 0; t < T; t++) {
      const g = DL.backward(net, D.X, D.Y);
      loss.push(g.loss);
      if (!isFinite(g.loss)) { while (loss.length < T) loss.push(NaN); break; }
      DL.sgdStep(net, g, lr);
    }
    return loss;
  }

  function draw() {
    const scales = se.value.split(",").map(Number), L = +Le.value, w = +we.value,
      T = +Te.value, lr = Math.pow(10, +le.value);
    document.getElementById("ib-Lv").textContent = L;
    document.getElementById("ib-wv").textContent = w;
    document.getElementById("ib-Tv").textContent = T;
    document.getElementById("ib-lrv").textContent = DL.fmt(lr, 3);

    const runs = scales.map(sc => {
      const net = build(sc, L, w);
      const prof = profile(net);
      /* the rms pre-activation at the LAST hidden layer. The fraction of
         negative z is NOT informative here: scaling every weight by the same
         constant leaves every sign unchanged, so that column would read the
         same for all four runs. The magnitude is what differs.               */
      const rmsLast = Math.sqrt(prof[prof.length - 1]);
      const g0 = DL.backward(net, D.X, D.Y);
      const gnorm = DL.frob(g0.dW[0]);
      const loss = train(net, T, lr);
      return { sc: sc, prof: prof, loss: loss, l0: loss[0], lT: loss[T - 1], rmsLast: rmsLast, gnorm: gnorm };
    });

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const py = 34, PH = 220;

    /* --- left: the activation profile at initialisation --- */
    const g1 = g.append("g").attr("transform", "translate(46," + py + ")");
    g.append("text").attr("x", 46).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("𝔼[z²] by layer, at step 0");
    const PW = 300;
    const xL = d3.scaleLinear().domain([1, Math.max(2, L)]).range([0, PW]);
    const ally = runs.reduce((s, r) => s.concat(r.prof), []).filter(v => isFinite(v) && v > 0);
    const y = d3.scaleLog().domain([Math.max(1e-30, d3.min(ally) * 0.3), Math.max(10, d3.max(ally) * 3)])
      .range([PH, 0]).clamp(true);
    DL.gridY(g1, y, PW, 5);
    DL.axisB(g1, xL, PH, 5, "layer"); DL.axisL(g1, y, 5, "𝔼[z²]", d3.format(".0e"));
    [[3.4e38, "float32 max"], [1.18e-38, "float32 min normal"]].forEach(([v, lab]) => {
      if (v > y.domain()[0] && v < y.domain()[1]) {
        g1.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(v)).attr("y2", y(v))
          .attr("stroke", DC.bad).attr("stroke-dasharray", "4 4").attr("stroke-opacity", 0.6);
        g1.append("text").attr("x", PW - 3).attr("y", y(v) - 3).attr("text-anchor", "end")
          .attr("font-size", 9).attr("fill", DC.bad).text(lab);
      }
    });
    g1.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(1)).attr("y2", y(1))
      .attr("stroke", DC.good).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    runs.forEach((r, k) => DL.curve(g1, r.prof.map((v, l) => [xL(l + 1), y(DL.clamp(v, y.domain()[0], y.domain()[1]))]),
      { stroke: NT.hue(k), w: r.sc === 1 ? 2.8 : 1.9 }));
    DL.legend(g1, runs.map((r, k) => ({ color: NT.hue(k), label: r.sc + "× the rule" })), 8, 14, { gap: 14, font: 10.5 });

    /* --- right: the training loss --- */
    const g2 = g.append("g").attr("transform", "translate(430," + py + ")");
    g.append("text").attr("x", 430).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the same four networks, actually trained");
    const PW2 = 285;
    const xt = d3.scaleLinear().domain([0, T - 1]).range([0, PW2]);
    const fin = runs.reduce((s, r) => s.concat(r.loss.filter(isFinite)), []);
    const yl = d3.scaleLinear().domain([0, Math.min(2, d3.max(fin) * 1.15)]).range([PH, 0]).clamp(true);
    DL.gridY(g2, yl, PW2, 5);
    DL.axisB(g2, xt, PH, 5, "training step"); DL.axisL(g2, yl, 5, "mean loss");
    g2.append("line").attr("x1", 0).attr("x2", PW2).attr("y1", yl(Math.log(2))).attr("y2", yl(Math.log(2)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    g2.append("text").attr("x", PW2 - 2).attr("y", yl(Math.log(2)) - 4).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", DC.muted).text("log 2 = label entropy");
    runs.forEach((r, k) => DL.curve(g2, r.loss.map((v, i) => [xt(i), yl(isFinite(v) ? v : 2)]),
      { stroke: NT.hue(k), w: r.sc === 1 ? 2.8 : 1.9 }));

    /* --- table --- */
    const ty = py + PH + 42, cols = [46, 150, 250, 350, 470, 600];
    ["scale", "initial loss", "final loss", "rms z, last hidden", "‖∂L/∂W⁽¹⁾‖", "verdict"].forEach((t, j) =>
      g.append("text").attr("x", cols[j]).attr("y", ty).attr("font-size", 10).attr("fill", DC.muted).text(t));
    runs.forEach((r, k) => {
      const blown = !isFinite(r.lT) || r.lT > 5;
      const verdict = blown ? "diverged" : (r.gnorm < 1e-6 ? "signal dead" :
        (r.lT < 0.35 * Math.log(2) ? "learning" : "stalled"));
      const cells = [r.sc + "×", DL.fmt(r.l0, 4), isFinite(r.lT) ? DL.fmt(r.lT, 4) : "NaN",
        DL.fmtE(r.rmsLast, 2), DL.fmtE(r.gnorm, 2), verdict];
      cells.forEach((c, j) => g.append("text").attr("x", cols[j]).attr("y", ty + 16 + k * 15)
        .attr("font-size", 10.5).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", j === 5 ? (verdict === "learning" ? DC.good : DC.bad) : (j === 0 ? NT.hue(k) : DC.ink))
        .attr("font-weight", j === 0 ? 600 : 400).text(c));
    });

    const good = runs.find(r => r.sc === 1) || runs[Math.floor(runs.length / 2)];
    const small = runs[0], big = runs[runs.length - 1];
    NT.say("#initbad-readout",
      "At <b>" + good.sc + "×</b> the rule, 𝔼[z²] is flat across all " + L + " layers and the loss falls from " +
      DL.fmt(good.l0, 4) + " to <b>" + DL.fmt(good.lT, 4) + "</b>. At <b>" + small.sc +
      "×</b> the initial loss is " + DL.fmt(small.l0, 4) + " — indistinguishable — and the final loss is <b>" +
      (isFinite(small.lT) ? DL.fmt(small.lT, 4) : "NaN") + "</b>, with a first-layer gradient norm of <b>" +
      DL.fmtE(small.gnorm, 2) + "</b> against <b>" + DL.fmtE(good.gnorm, 2) +
      "</b> for the healthy run. That gradient norm is the only column that tells the two flat loss curves apart: one is flat because it has converged and one is flat because nothing is reaching it. At <b>" +
      big.sc + "×</b>, " + (isFinite(big.lT) ? "the loss is " + DL.fmt(big.lT, 4) + " and oscillating"
        : "the loss is non-finite within " + big.loss.findIndex(v => !isFinite(v)) + " steps") +
      ". Initial loss ≈ log 2 = 0.6931 is <i>necessary</i> and, as the small-scale run shows, nowhere near sufficient.");
  }
  [se, Le, we, Te, le].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 12 · #sgd-svg — batch, minibatch, stochastic ═════════════════ */
(function () {
  const svg = d3.select("#sgd-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, NDATA = 512;
  const Be = document.getElementById("sg-B"), le = document.getElementById("sg-lr"),
    Te = document.getElementById("sg-T"), xe = document.getElementById("sg-x"),
    fe = document.getElementById("sg-full");
  const BSIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

  /* a two-parameter logistic regression, so the trajectory lives in the plane
     and the "loss surface" drawn is the actual objective                     */
  const D = (function (seed) {
    const r = DL.rng(seed), X = [], y = [];
    for (let i = 0; i < NDATA; i++) {
      const k = i % 2;
      const x1 = DL.randn(r) * 1.0 + (k ? 1.1 : -1.1), x2 = DL.randn(r) * 2.6;
      X.push([x1, x2]); y.push(k);
    }
    return { X: X, y: y };
  })(3);
  const lossAt = w => {
    let s = 0;
    for (let i = 0; i < NDATA; i++) s += DL.bce(w[0] * D.X[i][0] + w[1] * D.X[i][1], D.y[i]);
    return s / NDATA;
  };
  function gradOn(w, idx) {
    const g = [0, 0];
    for (const i of idx) {
      const e = DL.sigmoid(w[0] * D.X[i][0] + w[1] * D.X[i][1]) - D.y[i];
      g[0] += e * D.X[i][0]; g[1] += e * D.X[i][1];
    }
    return [g[0] / idx.length, g[1] / idx.length];
  }
  const ALL = d3.range(NDATA);

  function run(B, lr, T, seed) {
    const r = DL.rng(seed);
    let w = [-2.6, 2.2];
    const path = [w.slice()], loss = [lossAt(w)];
    for (let t = 0; t < T; t++) {
      const idx = [];
      for (let k = 0; k < B; k++) idx.push(Math.floor(r() * NDATA));
      const g = gradOn(w, idx);
      w = [w[0] - lr * g[0], w[1] - lr * g[1]];
      path.push(w.slice()); loss.push(lossAt(w));
    }
    return { path: path, loss: loss };
  }

  function draw() {
    const B = BSIZES[+Be.value], lr = Math.pow(10, +le.value), T = +Te.value,
      perComp = xe.value === "epochs", showFull = fe.checked;
    document.getElementById("sg-Bv").textContent = B;
    document.getElementById("sg-lrv").textContent = DL.fmt(lr, 3);
    document.getElementById("sg-Tv").textContent = T;

    const A = run(B, lr, T, 5);
    const Fl = run(NDATA, lr, T, 5);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 46, py = 34, PW = 300, PH = 310;

    /* --- contours + trajectory --- */
    const g1 = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the objective, and the path taken");
    const xw = d3.scaleLinear().domain([-3.2, 1.6]).range([0, PW]);
    const yw = d3.scaleLinear().domain([-1.4, 2.6]).range([PH, 0]);
    const RES = 60, vals = [];
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++)
      vals.push(lossAt([xw.invert(i * PW / (RES - 1)), yw.invert(j * PH / (RES - 1))]));
    const ext = d3.extent(vals);
    const col = d3.scaleSequential(d3.interpolateViridis).domain([ext[1], ext[0]]);
    DL.cells(g1, 0, 0, PW / RES, RES, RES, (i, j) => col(vals[j * RES + i]));
    g1.append("g").selectAll("path").data(d3.contours().size([RES, RES])
      .thresholds(d3.range(ext[0], ext[1], (ext[1] - ext[0]) / 11))(vals))
      .join("path").attr("d", d3.geoPath(d3.geoIdentity().scale(PW / RES)))
      .attr("fill", "none").attr("stroke", "#0f1117").attr("stroke-opacity", 0.35);
    DL.axisB(g1, xw, PH, 5, "w₁"); DL.axisL(g1, yw, 5, "w₂");
    if (showFull) DL.curve(g1, Fl.path.map(p => [xw(p[0]), yw(p[1])]), { stroke: DC.good, w: 2.4, op: 0.9 });
    DL.curve(g1, A.path.map(p => [xw(p[0]), yw(p[1])]), { stroke: DC.a2, w: 1.5, op: 0.95 });
    A.path.filter((_, i) => i % Math.max(1, Math.floor(T / 60)) === 0)
      .forEach(p => g1.append("circle").attr("cx", xw(p[0])).attr("cy", yw(p[1])).attr("r", 1.8).attr("fill", DC.a2));
    g1.append("circle").attr("cx", xw(A.path[0][0])).attr("cy", yw(A.path[0][1])).attr("r", 4)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.6);
    DL.legend(g1, [{ color: DC.a2, label: "B = " + B }].concat(showFull ? [{ color: DC.good, label: "full batch (B = " + NDATA + ")" }] : []),
      8, 14, { gap: 14, font: 10.5 });

    /* --- loss against steps or compute --- */
    const QX = 410, QW = 300, QH = 140;
    const g2 = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
    g.append("text").attr("x", QX).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(perComp ? "loss against ARITHMETIC (examples seen)" : "loss against UPDATES");
    const costA = i => perComp ? i * B : i, costF = i => perComp ? i * NDATA : i;
    const xmax = perComp ? Math.max(costA(T), showFull ? costF(T) : 0) : T;
    const xs = (perComp ? d3.scaleLog().domain([Math.max(1, B), Math.max(10, xmax)]) : d3.scaleLinear().domain([0, T])).range([0, QW]);
    const ys = d3.scaleLinear().domain([0, Math.max(d3.max(A.loss), d3.max(Fl.loss))]).range([QH, 0]);
    DL.gridY(g2, ys, QW, 4);
    DL.axisB(g2, xs, QH, 5, perComp ? "examples processed" : "update", perComp ? d3.format(".0e") : null);
    DL.axisL(g2, ys, 4, "loss");
    DL.curve(g2, A.loss.map((v, i) => [xs(Math.max(perComp ? B : 0, costA(i))), ys(v)]), { stroke: DC.a2, w: 2.2 });
    if (showFull) DL.curve(g2, Fl.loss.map((v, i) => [xs(Math.max(perComp ? B : 0, costF(i))), ys(v)]), { stroke: DC.good, w: 2.2 });

    /* --- measured noise against B --- */
    const g3 = g.append("g").attr("transform", "translate(" + QX + "," + (py + QH + 56) + ")");
    g.append("text").attr("x", QX).attr("y", py + QH + 44).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("‖ĝ − g‖ against B, measured over 300 resamples");
    const w0 = [-2.6, 2.2], gTrue = gradOn(w0, ALL);
    const rr = DL.rng(77), pts = [];
    BSIZES.forEach(b => {
      let acc = 0;
      for (let k = 0; k < 300; k++) {
        const idx = [];
        for (let m = 0; m < b; m++) idx.push(Math.floor(rr() * NDATA));
        const gh = gradOn(w0, idx);
        acc += Math.hypot(gh[0] - gTrue[0], gh[1] - gTrue[1]);
      }
      pts.push([b, acc / 300]);
    });
    const QH2 = 110;
    const xb = d3.scaleLog().domain([1, NDATA]).range([0, QW]);
    const yb = d3.scaleLog().domain([d3.min(pts, p => p[1]) * 0.6, d3.max(pts, p => p[1]) * 1.6]).range([QH2, 0]);
    DL.gridY(g3, yb, QW, 4);
    DL.axisB(g3, xb, QH2, 5, "batch size B", d3.format("d"));
    DL.axisL(g3, yb, 4, "", d3.format(".2f"));
    const c0 = pts[0][1];
    DL.curve(g3, pts.map(p => [xb(p[0]), yb(c0 / Math.sqrt(p[0]))]), { stroke: DC.muted, w: 1.6, dash: "5 3" });
    DL.curve(g3, pts.map(p => [xb(p[0]), yb(p[1])]), { stroke: DC.accent, w: 2.2 });
    pts.forEach(p => g3.append("circle").attr("cx", xb(p[0])).attr("cy", yb(p[1])).attr("r", 2.6).attr("fill", DC.accent));
    const cur = pts[BSIZES.indexOf(B)];
    g3.append("circle").attr("cx", xb(cur[0])).attr("cy", yb(cur[1])).attr("r", 5.5)
      .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2);
    DL.legend(g3, [{ color: DC.accent, label: "measured" }, { color: DC.muted, label: "c/√B", dash: "5 3" }],
      8, 12, { gap: 13, font: 10 });

    const scaled = pts.map(p => p[1] * Math.sqrt(p[0]));
    NT.say("#sgd-readout",
      "At B = <b>" + B + "</b> the measured mean gradient error is <b>" + DL.fmt(cur[1], 4) +
      "</b>; multiplied by √B it is <b>" + DL.fmt(cur[1] * Math.sqrt(B), 4) +
      "</b>, against " + DL.fmt(scaled[0], 4) + " at B = 1 and " + DL.fmt(scaled[scaled.length - 1], 4) +
      " at B = " + NDATA + " — flat across nine doublings, which is the 1/√B law measured rather than asserted. " +
      "Final loss after " + T + " updates: <b>" + DL.fmt(A.loss[T], 5) + "</b> at B = " + B +
      " against <b>" + DL.fmt(Fl.loss[T], 5) + "</b> for full batch, but those two runs did <b>" +
      DL.big(T * B) + "</b> and <b>" + DL.big(T * NDATA) + "</b> example-gradients respectively — a factor of " +
      DL.big(NDATA / B) + ". Switch the x-axis to arithmetic and the comparison inverts. " +
      "Doubling B buys √2 = 1.414× less noise for 2× the cost, at every scale, forever.");
  }
  [Be, le, Te, xe, fe].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 13 · #cond-svg — curvature and the bound on η ═════════════════ */
(function () {
  const svg = d3.select("#cond-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, STEPS = 60;
  const e1 = document.getElementById("cd-l1"), e2 = document.getElementById("cd-l2"),
    et = document.getElementById("cd-th"), el = document.getElementById("cd-lr"),
    eo = document.getElementById("cd-opt");

  function draw() {
    const l1 = +e1.value, l2 = +e2.value, th = +et.value * Math.PI / 180;
    document.getElementById("cd-l1v").textContent = DL.fmt(l1, 1);
    document.getElementById("cd-l2v").textContent = DL.fmt(l2, 2);
    document.getElementById("cd-thv").textContent = (th * 180 / Math.PI).toFixed(0) + "°";

    /* H = R diag(l1, l2) Rᵀ, built explicitly, then eigen-decomposed EXACTLY
       by the closed form — so the figure re-derives what it was given.       */
    const c = Math.cos(th), s = Math.sin(th);
    const a = l1 * c * c + l2 * s * s, b = (l1 - l2) * c * s, d = l1 * s * s + l2 * c * c;
    const E = DL.eig2sym(a, b, d);
    const etaMax = 2 / E.hi, etaOpt = 2 / (E.lo + E.hi), rate = (E.cond - 1) / (E.cond + 1);
    if (eo.checked) el.value = DL.clamp(etaOpt, +el.min, +el.max);
    const eta = +el.value;
    document.getElementById("cd-lrv").textContent = DL.fmt(eta, 3);

    const Lf = w => 0.5 * (a * w[0] * w[0] + 2 * b * w[0] * w[1] + d * w[1] * w[1]);
    const Gf = w => [a * w[0] + b * w[1], b * w[0] + d * w[1]];
    function traj(lr) {
      let w = [2.4, 1.6];
      const P = [w.slice()], LS = [Lf(w)];
      for (let t = 0; t < STEPS; t++) {
        const g = Gf(w);
        w = [w[0] - lr * g[0], w[1] - lr * g[1]];
        if (!isFinite(w[0]) || Math.abs(w[0]) > 1e12) { w = [NaN, NaN]; }
        P.push(w.slice()); LS.push(Lf(w));
      }
      return { P: P, L: LS };
    }
    const T = traj(eta), TO = traj(etaOpt);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 46, py = 34, PW = 300, PH = 310;

    const g1 = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the quadratic, its eigenvectors, and the trajectory");
    const R = 3.4;
    const xw = d3.scaleLinear().domain([-R, R]).range([0, PW]);
    const yw = d3.scaleLinear().domain([-R, R]).range([PH, 0]);
    const RES = 64, vals = [];
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++)
      vals.push(Math.log1p(Lf([xw.invert(i * PW / (RES - 1)), yw.invert(j * PH / (RES - 1))])));
    const ext = d3.extent(vals);
    const cs = d3.scaleSequential(d3.interpolateViridis).domain([ext[1], ext[0]]);
    DL.cells(g1, 0, 0, PW / RES, RES, RES, (i, j) => cs(vals[j * RES + i]));
    g1.append("g").selectAll("path").data(d3.contours().size([RES, RES])
      .thresholds(d3.range(ext[0], ext[1], (ext[1] - ext[0]) / 12))(vals))
      .join("path").attr("d", d3.geoPath(d3.geoIdentity().scale(PW / RES)))
      .attr("fill", "none").attr("stroke", "#0f1117").attr("stroke-opacity", 0.3);
    DL.axisB(g1, xw, PH, 5, "w₁"); DL.axisL(g1, yw, 5, "w₂");
    [[E.vhi, E.hi, DC.bad, "λ_max"], [E.vlo, E.lo, DC.good, "λ_min"]].forEach(([v, lam, col, lab]) => {
      const k = 2.2 / Math.sqrt(Math.max(lam, 1e-6)) * Math.sqrt(E.hi) * 0.55;
      DL.arrow(g1, xw(0), yw(0), xw(v[0] * k), yw(v[1] * k), { color: col, w: 2, head: 7 });
      g1.append("text").attr("x", xw(v[0] * k)).attr("y", yw(v[1] * k) - 5).attr("font-size", 10)
        .attr("fill", col).attr("text-anchor", "middle").text(lab + " = " + DL.fmt(lam, 3));
    });
    const inR = p => isFinite(p[0]) && Math.abs(p[0]) < R * 1.4 && Math.abs(p[1]) < R * 1.4;
    const seg = [];
    T.P.forEach(p => { if (inR(p)) seg.push([xw(DL.clamp(p[0], -R, R)), yw(DL.clamp(p[1], -R, R))]); });
    DL.curve(g1, seg, { stroke: DC.a2, w: 1.6 });
    T.P.filter(inR).forEach((p, i) => g1.append("circle").attr("cx", xw(DL.clamp(p[0], -R, R)))
      .attr("cy", yw(DL.clamp(p[1], -R, R))).attr("r", 2.6).attr("fill", DC.a2)
      .attr("fill-opacity", 0.4 + 0.6 * i / Math.max(1, T.P.length)));
    g1.append("circle").attr("cx", xw(2.4)).attr("cy", yw(1.6)).attr("r", 4.5)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.6);

    /* --- loss --- */
    const QX = 410, QW = 300, QH = 130;
    const g2 = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
    g.append("text").attr("x", QX).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("loss against step, this η against the optimum");
    const xs = d3.scaleLinear().domain([0, STEPS]).range([0, QW]);
    const fin = T.L.concat(TO.L).filter(v => isFinite(v) && v > 0);
    const ys = d3.scaleLog().domain([Math.max(1e-16, d3.min(fin)), Math.max(1e3, d3.max(fin))]).range([QH, 0]).clamp(true);
    DL.gridY(g2, ys, QW, 4);
    DL.axisB(g2, xs, QH, 5, "step"); DL.axisL(g2, ys, 4, "loss", d3.format(".0e"));
    DL.curve(g2, TO.L.map((v, i) => [xs(i), ys(isFinite(v) ? Math.max(v, 1e-16) : 1e3)]), { stroke: DC.good, w: 1.8, dash: "5 3" });
    DL.curve(g2, T.L.map((v, i) => [xs(i), ys(isFinite(v) ? Math.max(v, 1e-16) : 1e3)]), { stroke: DC.a2, w: 2.2 });
    DL.legend(g2, [{ color: DC.a2, label: "η = " + DL.fmt(eta, 3) },
      { color: DC.good, label: "η* = " + DL.fmt(etaOpt, 3), dash: "5 3" }], 8, 12, { gap: 13, font: 10 });

    /* --- contraction against η --- */
    const g3 = g.append("g").attr("transform", "translate(" + QX + "," + (py + QH + 58) + ")");
    g.append("text").attr("x", QX).attr("y", py + QH + 46).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("per-step contraction |1 − ηλ| for each eigen-direction");
    const QH2 = 110;
    const xe2 = d3.scaleLinear().domain([0, Math.max(1.2, etaMax * 1.35)]).range([0, QW]);
    const ye2 = d3.scaleLinear().domain([0, 1.6]).range([QH2, 0]);
    DL.gridY(g3, ye2, QW, 4);
    DL.axisB(g3, xe2, QH2, 5, "η"); DL.axisL(g3, ye2, 4, "");
    g3.append("rect").attr("x", xe2(etaMax)).attr("y", 0).attr("width", Math.max(0, QW - xe2(etaMax)))
      .attr("height", QH2).attr("fill", DC.bad).attr("fill-opacity", 0.09);
    g3.append("line").attr("x1", 0).attr("x2", QW).attr("y1", ye2(1)).attr("y2", ye2(1))
      .attr("stroke", DC.bad).attr("stroke-dasharray", "4 3");
    const es = DL.linspace(0, xe2.domain()[1], 200);
    DL.curve(g3, es.map(v => [xe2(v), ye2(DL.clamp(Math.abs(1 - v * E.hi), 0, 1.6))]), { stroke: DC.bad, w: 2 });
    DL.curve(g3, es.map(v => [xe2(v), ye2(DL.clamp(Math.abs(1 - v * E.lo), 0, 1.6))]), { stroke: DC.good, w: 2 });
    DL.curve(g3, es.map(v => [xe2(v), ye2(DL.clamp(Math.max(Math.abs(1 - v * E.hi), Math.abs(1 - v * E.lo)), 0, 1.6))]),
      { stroke: DC.violet, w: 2.6, op: 0.85 });
    [[etaOpt, DC.violet, "η*"], [etaMax, DC.bad, "2/λ_max"], [eta, DC.a2, "now"]].forEach(([v, col, lab]) => {
      g3.append("line").attr("x1", xe2(v)).attr("x2", xe2(v)).attr("y1", 0).attr("y2", QH2)
        .attr("stroke", col).attr("stroke-opacity", 0.75).attr("stroke-dasharray", lab === "now" ? null : "3 3");
      g3.append("text").attr("x", xe2(v) + 3).attr("y", 11).attr("font-size", 9.5).attr("fill", col).text(lab);
    });

    /* measured contraction from the actual trajectory */
    const norms = T.P.map(p => Math.hypot(p[0], p[1]));
    let meas = NaN;
    for (let i = STEPS - 5; i > 5; i--) {
      if (isFinite(norms[i]) && isFinite(norms[i - 1]) && norms[i - 1] > 1e-14) { meas = norms[i] / norms[i - 1]; break; }
    }
    NT.say("#cond-readout",
      "H has eigenvalues <b>" + DL.fmt(E.lo, 8) + "</b> and <b>" + DL.fmt(E.hi, 8) +
      "</b>, so κ = <b>" + DL.fmt(E.cond, 6) + "</b>. Stability bound 2/λ_max = <b>" + DL.fmt(etaMax, 8) +
      "</b>; optimal step 2/(λ_min + λ_max) = <b>" + DL.fmt(etaOpt, 8) +
      "</b>; predicted contraction there (κ−1)/(κ+1) = <b>" + rate.toFixed(15) + "</b>. At the current η = " + DL.fmt(eta, 4) + " the trajectory's measured contraction is <b>" +
      (isFinite(meas) ? meas.toFixed(10) : "divergent") + "</b> against a predicted <b>" +
      Math.max(Math.abs(1 - eta * E.hi), Math.abs(1 - eta * E.lo)).toFixed(10) + "</b>. " +
      (eta > etaMax ? "<b>Past the bound:</b> the stiff direction is being amplified by " +
        DL.fmt(Math.abs(1 - eta * E.hi), 4) + " per step and the run is diverging."
        : "Steps to reduce the error tenfold at this η: <b>" +
        (Math.max(Math.abs(1 - eta * E.hi), Math.abs(1 - eta * E.lo)) >= 1 ? "never" :
          Math.ceil(Math.log(0.1) / Math.log(Math.max(Math.abs(1 - eta * E.hi), Math.abs(1 - eta * E.lo)))))
        + "</b>, against <b>" + Math.ceil(Math.log(0.1) / Math.log(rate)) + "</b> at the optimum.") +
      " One stiff direction sets the ceiling for every direction — which is the argument for the per-coordinate methods of §21.");
  }
  [e1, e2, et, el, eo].forEach(el2 => el2.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 14 · #saddle-svg — saddles in high dimensions ═════════════════ */
(function () {
  const svg = d3.select("#saddle-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, STEPS = 400;
  const se = document.getElementById("sd-surf"), oe = document.getElementById("sd-off"),
    le = document.getElementById("sd-lr"), ne = document.getElementById("sd-nz"),
    ce = document.getElementById("sd-clip");

  const SURF = {
    saddle: { f: (x, y) => 0.5 * (1.0 * x * x - 0.6 * y * y), g: (x, y) => [1.0 * x, -0.6 * y], lab: "λ₁ = 1, λ₂ = −0.6" },
    monkey: { f: (x, y) => 0.18 * (x * x * x - 3 * x * y * y), g: (x, y) => [0.54 * (x * x - y * y), -1.08 * x * y], lab: "three descending directions" },
    plateau: { f: (x, y) => 0.02 * Math.pow(x * x + y * y, 2) / (1 + 0.35 * (x * x + y * y)), g: (x, y) => { const r2 = x * x + y * y, den = 1 + 0.35 * r2; const c = 0.02 * (4 * r2 * den - 0.7 * r2 * r2) / (den * den); return [c * x, c * y]; }, lab: "gradient AND curvature ≈ 0 at the centre" },
    cliff: { f: (x, y) => 0.15 * y * y + 3.2 / (1 + Math.exp(-6 * (x - 1.1))), g: (x, y) => { const s = DL.sigmoid(6 * (x - 1.1)); return [3.2 * 6 * s * (1 - s), 0.3 * y]; }, lab: "a wall at x = 1.1" }
  };

  /* P(a random symmetric n×n matrix is positive definite), measured once and
     cached — it depends on no control. The matrix is drawn from the GAUSSIAN
     ORTHOGONAL ENSEMBLE: diagonal entries ~ 𝒩(0,1) and off-diagonal ~ 𝒩(0,½),
     which is the convention under which eigenvalue repulsion has its standard
     form. Positive-definiteness is tested by Sylvester's criterion on the
     leading principal minors. Verified in the build against an independent
     implementation using a full eigen-decomposition over 40 000 samples:
     0.49907, 0.14705, 0.02475, 0.00258, 0.00017 for n = 1…5.                */
  const NS = [1, 2, 3, 4, 5];
  const PDCACHE = (function () {
    const rr = DL.rng(41), TR = 6000;
    return NS.map(n => {
      let cnt = 0;
      for (let k = 0; k < TR; k++) {
        const A = DL.zeros2(n, n);
        for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
          const v = DL.randn(rr) * (i === j ? 1 : Math.SQRT1_2);
          A[i][j] = v; A[j][i] = v;
        }
        let pd = true;
        for (let m = 1; m <= n && pd; m++) {
          const B = A.slice(0, m).map(r => r.slice(0, m));
          let det = 1;
          for (let i = 0; i < m; i++) {
            let piv = i;
            for (let q = i + 1; q < m; q++) if (Math.abs(B[q][i]) > Math.abs(B[piv][i])) piv = q;
            if (Math.abs(B[piv][i]) < 1e-13) { det = 0; break; }
            if (piv !== i) { const t = B[i]; B[i] = B[piv]; B[piv] = t; det = -det; }
            det *= B[i][i];
            for (let q = i + 1; q < m; q++) {
              const f = B[q][i] / B[i][i];
              for (let c = i; c < m; c++) B[q][c] -= f * B[i][c];
            }
          }
          if (!(det > 0)) pd = false;
        }
        if (pd) cnt++;
      }
      return cnt / TR;
    });
  })();

  function draw() {
    const key = se.value, off = Math.pow(10, +oe.value), lr = Math.pow(10, +le.value),
      nz = +ne.value, clip = ce.checked;
    document.getElementById("sd-offv").textContent = DL.fmtE(off, 1);
    document.getElementById("sd-lrv").textContent = DL.fmt(lr, 3);
    document.getElementById("sd-nzv").textContent = DL.fmt(nz, 3);
    const S = SURF[key];

    const start = key === "cliff" ? [-1.4, 1.6] : [0.9, off];
    const r = DL.rng(19);
    let w = start.slice();
    const P = [w.slice()], dist = [Math.hypot(w[0], w[1])];
    let esc = -1, blown = -1;
    for (let t = 0; t < STEPS; t++) {
      let gv = S.g(w[0], w[1]);
      if (nz > 0) gv = [gv[0] + nz * DL.randn(r), gv[1] + nz * DL.randn(r)];
      if (clip) gv = DL.clipNorm(gv, 1).g;
      w = [w[0] - lr * gv[0], w[1] - lr * gv[1]];
      if (!isFinite(w[0]) || Math.hypot(w[0], w[1]) > 1e6) { blown = t; break; }
      P.push(w.slice());
      const dd = Math.hypot(w[0], w[1]);
      dist.push(dd);
      if (esc < 0 && Math.abs(w[1]) > 1.0 && key !== "cliff") esc = t;
    }

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 46, py = 34, PW = 300, PH = 310;
    const g1 = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(se.options[se.selectedIndex].text.split(":")[0] + " — " + S.lab);
    const R = 2.6;
    const xw = d3.scaleLinear().domain([-R, R]).range([0, PW]);
    const yw = d3.scaleLinear().domain([-R, R]).range([PH, 0]);
    const RES = 64, vals = [];
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++)
      vals.push(S.f(xw.invert(i * PW / (RES - 1)), yw.invert(j * PH / (RES - 1))));
    const ext = d3.extent(vals);
    const cs = d3.scaleSequential(d3.interpolateViridis).domain([ext[1], ext[0]]);
    DL.cells(g1, 0, 0, PW / RES, RES, RES, (i, j) => cs(vals[j * RES + i]));
    g1.append("g").selectAll("path").data(d3.contours().size([RES, RES])
      .thresholds(d3.range(ext[0], ext[1], (ext[1] - ext[0]) / 14))(vals))
      .join("path").attr("d", d3.geoPath(d3.geoIdentity().scale(PW / RES)))
      .attr("fill", "none").attr("stroke", "#0f1117").attr("stroke-opacity", 0.3);
    DL.axisB(g1, xw, PH, 5, "x"); DL.axisL(g1, yw, 5, "y");
    if (key !== "cliff") {
      g1.append("circle").attr("cx", xw(0)).attr("cy", yw(0)).attr("r", 4).attr("fill", DC.bad);
      g1.append("text").attr("x", xw(0) + 7).attr("y", yw(0) - 5).attr("font-size", 10)
        .attr("fill", DC.bad).text("∇L = 0");
    }
    const inR = p => isFinite(p[0]) && Math.abs(p[0]) < R * 1.2 && Math.abs(p[1]) < R * 1.2;
    DL.curve(g1, P.filter(inR).map(p => [xw(p[0]), yw(p[1])]), { stroke: DC.a2, w: 1.6 });
    P.filter(inR).forEach((p, i) => g1.append("circle").attr("cx", xw(p[0])).attr("cy", yw(p[1]))
      .attr("r", 2.1).attr("fill", DC.a2).attr("fill-opacity", 0.35 + 0.65 * i / Math.max(1, P.length)));
    g1.append("circle").attr("cx", xw(start[0])).attr("cy", yw(start[1])).attr("r", 4.5)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.6);

    /* distance from the critical point */
    const QX = 410, QW = 300, QH = 130;
    const g2 = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
    g.append("text").attr("x", QX).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(key === "cliff" ? "‖θ‖ against step" : "distance from the critical point");
    const xs = d3.scaleLinear().domain([0, STEPS]).range([0, QW]);
    const ys = d3.scaleLog().domain([Math.max(1e-6, d3.min(dist) * 0.6), Math.max(10, d3.max(dist) * 1.4)])
      .range([QH, 0]).clamp(true);
    DL.gridY(g2, ys, QW, 4);
    DL.axisB(g2, xs, QH, 5, "step"); DL.axisL(g2, ys, 4, "", d3.format(".0e"));
    DL.curve(g2, dist.map((v, i) => [xs(i), ys(v)]), { stroke: DC.accent, w: 2.2 });
    if (esc >= 0) {
      g2.append("line").attr("x1", xs(esc)).attr("x2", xs(esc)).attr("y1", 0).attr("y2", QH)
        .attr("stroke", DC.good).attr("stroke-dasharray", "3 3");
      g2.append("text").attr("x", xs(esc) + 4).attr("y", 12).attr("font-size", 10).attr("fill", DC.good)
        .text("escaped at step " + esc);
    }
    if (blown >= 0) {
      g2.append("text").attr("x", 6).attr("y", 12).attr("font-size", 10).attr("fill", DC.bad)
        .text("launched off the cliff at step " + blown);
    }

    /* the counting argument, measured */
    const g3 = g.append("g").attr("transform", "translate(" + QX + "," + (py + QH + 58) + ")");
    g.append("text").attr("x", QX).attr("y", py + QH + 46).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("P(all n eigenvalues > 0): the 2⁻ⁿ heuristic against measurement");
    const QH2 = 110, FLOOR = 1e-4;
    const meas = PDCACHE.map((v, i) => [NS[i], v]);
    const xn = d3.scaleLinear().domain([1, NS[NS.length - 1]]).range([0, QW]);
    const yn = d3.scaleLog().domain([FLOOR, 1]).range([QH2, 0]).clamp(true);
    DL.gridY(g3, yn, QW, 4);
    DL.axisB(g3, xn, QH2, NS.length, "n"); DL.axisL(g3, yn, 4, "", d3.format(".0e"));
    DL.curve(g3, NS.map(n => [xn(n), yn(Math.pow(2, -n))]), { stroke: DC.muted, w: 1.8, dash: "5 3" });
    DL.curve(g3, meas.map(m => [xn(m[0]), yn(Math.max(m[1], FLOOR))]), { stroke: DC.accent, w: 2.2 });
    meas.forEach(m => g3.append("circle").attr("cx", xn(m[0])).attr("cy", yn(Math.max(m[1], FLOOR)))
      .attr("r", 2.6).attr("fill", DC.accent));
    DL.legend(g3, [{ color: DC.accent, label: "measured, 6 000 GOE matrices per point" },
      { color: DC.muted, label: "the 2⁻ⁿ coin heuristic", dash: "5 3" }], 8, 12, { gap: 13, font: 10 });

    NT.say("#saddle-readout",
      (key === "cliff"
        ? "The cliff: a wall of sigmoidal steepness at x = 1.1. " + (blown >= 0
          ? "The run was <b>launched off it at step " + blown + "</b> and never came back — one step, hours of progress gone."
          : "The run survived it. ") + (clip ? "Clipping is <b>on</b>: the gradient's direction is preserved and only its magnitude is capped, which is exactly right here, because the direction was never the problem."
            : "Turn clipping <b>on</b> and try again — clipping is the fix designed for precisely this failure.")
        : "Started <b>" + DL.fmtE(off, 2) + "</b> off the ridge. " +
        (esc >= 0 ? "The optimiser escaped the critical point at step <b>" + esc + "</b>."
          : "The optimiser has <b>not</b> escaped within " + STEPS + " steps.") +
        " Escape is exponential in the unstable direction, at rate (1 + η·|λ₂|) per step, so <b>halving the offset adds a fixed number of steps rather than doubling the time</b> — drop the offset by a factor of ten and the escape time increases by about " +
        (key === "saddle" ? Math.ceil(Math.log(10) / Math.log(1 + lr * 0.6)) : "a fixed amount") +
        " steps, not by a factor of ten. " +
        (nz > 0 ? "Gradient noise is on, and it is <i>helping</i>: it breaks the symmetry that keeps the iterate on the ridge, which is §15's argument for why minibatch noise is not purely a cost."
          : "Add gradient noise and the escape gets faster, because noise breaks the symmetry that pins the iterate to the ridge.")) +
      " In the lower-right panel, the coin heuristic is <b>measured against reality and loses</b>: the fraction of random symmetric matrices that are actually positive definite is " +
      meas.map(m => "<b>" + DL.fmt(m[1], 4) + "</b> at n = " + m[0]).join(", ") +
      ", against 2⁻ⁿ of " + NS.map(n => DL.fmt(Math.pow(2, -n), 4)).join(", ") +
      ". They agree at n = 1 and then diverge fast, because the eigenvalues of a symmetric matrix <b>repel one another</b> rather than being independent coin flips — so the true decay is faster than exponential in n, not exponential. The heuristic understates the case it is used to make. Either way the conclusion in a million dimensions is the same and much stronger: a critical point picked at random is a saddle.");
  }
  [se, oe, le, ne, ce].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 15 · #ewma-svg — the moving average and its bias ═════════════════ */
(function () {
  const svg = d3.select("#ewma-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, T = 220;
  const be = document.getElementById("ew-b"), b2e = document.getElementById("ew-b2"),
    se = document.getElementById("ew-sig"), ne = document.getElementById("ew-nz"),
    fe = document.getElementById("ew-fix");

  function series(kind, nz) {
    const r = DL.rng(13), out = [];
    for (let t = 1; t <= T; t++) {
      let v;
      if (kind === "season") v = 50 + 18 * Math.sin(2 * Math.PI * t / 90) + 6 * Math.sin(2 * Math.PI * t / 17);
      else if (kind === "step") v = t < T / 2 ? 20 : 70;
      else if (kind === "const") v = 50;
      else v = 0;
      out.push(v + nz * DL.randn(r));
    }
    return out;
  }

  function draw() {
    const b1 = +be.value, b2 = +b2e.value, kind = se.value, nz = +ne.value, fix = fe.checked;
    document.getElementById("ew-bv").textContent = DL.fmt(b1, 3);
    document.getElementById("ew-b2v").textContent = DL.fmt(b2, 3);
    document.getElementById("ew-nzv").textContent = DL.fmt(nz, 1);
    const th = series(kind, nz);
    const A = DL.ewma(th, b1, fix), B = DL.ewma(th, b2, fix);
    const Araw = DL.ewma(th, b1, false), Braw = DL.ewma(th, b2, false);

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 48, py = 34, PW = 470, PH = 300;
    const gg = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the series, and two moving averages of it — bias correction " + (fix ? "ON" : "OFF"));
    const x = d3.scaleLinear().domain([1, T]).range([0, PW]);
    const all = th.concat(A).concat(B);
    const y = d3.scaleLinear().domain([Math.min(0, d3.min(all)) - 3, d3.max(all) + 3]).range([PH, 0]);
    DL.gridY(gg, y, PW, 5);
    DL.axisB(gg, x, PH, 6, "step t"); DL.axisL(gg, y, 5, "value");
    /* the region where the correction factor still matters */
    const warm = b => { let t = 1; while (1 - Math.pow(b, t) < 0.99 && t < T) t++; return t; };
    const wz = Math.max(warm(b1), warm(b2));
    gg.append("rect").attr("x", 0).attr("y", 0).attr("width", x(Math.min(wz, T))).attr("height", PH)
      .attr("fill", DC.a2).attr("fill-opacity", 0.07);
    gg.append("text").attr("x", 4).attr("y", 13).attr("font-size", 9.5).attr("fill", DC.a2)
      .text("1 − βᵗ < 0.99 here (t < " + wz + ")");
    th.forEach((v, i) => gg.append("circle").attr("cx", x(i + 1)).attr("cy", y(v)).attr("r", 1.3)
      .attr("fill", DC.muted).attr("fill-opacity", 0.55));
    DL.curve(gg, A.map((v, i) => [x(i + 1), y(v)]), { stroke: DC.accent, w: 2.2 });
    DL.curve(gg, B.map((v, i) => [x(i + 1), y(v)]), { stroke: DC.violet, w: 2.2 });
    if (fix) {
      DL.curve(gg, Braw.map((v, i) => [x(i + 1), y(v)]), { stroke: DC.violet, w: 1.2, dash: "3 3", op: 0.5 });
    }
    DL.legend(gg, [{ color: DC.muted, label: "the series θ_t" },
      { color: DC.accent, label: "β = " + DL.fmt(b1, 3) },
      { color: DC.violet, label: "β = " + DL.fmt(b2, 3) }]
      .concat(fix ? [{ color: DC.violet, label: "β = " + DL.fmt(b2, 3) + ", uncorrected", dash: "3 3" }] : []),
      8, 28, { gap: 15, font: 10.5 });

    /* the weight profile */
    const QX = 560, QW = 175, QH = 120;
    const g2 = g.append("g").attr("transform", "translate(" + QX + "," + (py + 6) + ")");
    g.append("text").attr("x", QX).attr("y", py - 4).attr("font-size", 10.5).attr("fill", DC.ink)
      .attr("font-weight", 600).text("weight on the k-th past step");
    const K = 120;
    const xk = d3.scaleLinear().domain([0, K]).range([0, QW]);
    const yk = d3.scaleLinear().domain([0, 1]).range([QH, 0]);
    DL.axisB(g2, xk, QH, 4, "k back"); DL.axisL(g2, yk, 3, "");
    [[b1, DC.accent], [b2, DC.violet]].forEach(([bb, col]) => {
      DL.curve(g2, DL.linspace(0, K, 121).map(k => [xk(k), yk(Math.pow(bb, k))]), { stroke: col, w: 2 });
      const k1 = 1 / (1 - bb);
      if (k1 <= K) {
        g2.append("circle").attr("cx", xk(k1)).attr("cy", yk(Math.pow(bb, k1))).attr("r", 3).attr("fill", col);
        g2.append("text").attr("x", xk(k1) + 4).attr("y", yk(Math.pow(bb, k1)) - 3).attr("font-size", 9)
          .attr("fill", col).text(DL.fmt(Math.pow(bb, k1), 3));
      }
    });
    g2.append("line").attr("x1", 0).attr("x2", QW).attr("y1", yk(1 / Math.E)).attr("y2", yk(1 / Math.E))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    g2.append("text").attr("x", QW - 2).attr("y", yk(1 / Math.E) - 3).attr("text-anchor", "end")
      .attr("font-size", 9).attr("fill", DC.muted).text("1/e = 0.368");

    const k = DL.kv(g, QX, py + QH + 44, { keyW: 118, lead: 15, size: 10.5 });
    k("1/(1 − β₁)", DL.fmt(1 / (1 - b1), 1), DC.accent, true);
    k("1/(1 − β₂)", DL.fmt(1 / (1 - b2), 1), DC.violet, true);
    k("β₁ at t = 1: 1/(1−β₁)", DL.fmt(1 / (1 - b1), 2) + "×", DC.muted);
    k("β₂ at t = 1: 1/(1−β₂)", DL.fmt(1 / (1 - b2), 2) + "×", DC.muted);
    k("v₁ raw / v̂₁", DL.fmt(Araw[0], 3) + " / " + DL.fmt(Araw[0] / (1 - b1), 3), DC.ink);
    k("θ₁", DL.fmt(th[0], 3), DC.ink);

    const err1 = Math.abs(Braw[9] - th[9]) , err2 = Math.abs(Braw[9] / (1 - Math.pow(b2, 10)) - th[9]);
    NT.say("#ewma-readout",
      "At β = " + DL.fmt(b2, 3) + " the correction factor 1/(1 − βᵗ) is <b>" + DL.fmt(1 / (1 - b2), 2) +
      "×</b> at t = 1, <b>" + DL.fmt(1 / (1 - Math.pow(b2, 10)), 3) + "×</b> at t = 10, and " +
      "reaches 1.01× only at t = <b>" + warm(b2) + "</b>. Uncorrected, the first estimate is " +
      DL.fmt(Braw[0], 4) + " when the observation was " + DL.fmt(th[0], 4) +
      " — it reports " + DL.fmt(100 * (1 - b2), 2) + "% of the data. Corrected, it is exactly <b>" +
      DL.fmt(Braw[0] / (1 - b2), 4) + "</b>. " +
      "The correction is <b>exact, not approximate</b>: if every θ_t equalled c, then v_t = c(1 − βᵗ) identically, so v̂_t = c for every t including t = 1. " +
      (kind === "const" ? "Switch bias correction off on this constant series and watch the estimate crawl up from zero toward a value it already knew at step 1. "
        : "") +
      "The weight profile on the right shows β^k against k: at β = " + DL.fmt(b2, 3) + " the weight has fallen to <b>" +
      DL.fmt(Math.pow(b2, 1 / (1 - b2)), 4) + "</b> after 1/(1 − β) = " + DL.fmt(1 / (1 - b2), 1) +
      " steps, against the 1/e = 0.3679 the rule of thumb assumes — the approximation gets better as β → 1 and is poor below about β = 0.8.");
  }
  [be, b2e, se, ne, fe].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 16 · #opt-svg — seven optimisers, one landscape ═════════════════ */
(function () {
  const svg = d3.select("#opt-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const se = document.getElementById("op-surf"), le = document.getElementById("op-lr"),
    Te = document.getElementById("op-T"), b1e = document.getElementById("op-b1"),
    b2e = document.getElementById("op-b2"), she = document.getElementById("op-show");

  /* every surface returns {f, g, start, box:[x0,x1,y0,y1], min:[x,y]} */
  const SURF = {
    ravine: (() => { const a = 0.1, b = 5.0; return {
      f: (x, y) => 0.5 * (a * x * x + b * y * y), g: (x, y) => [a * x, b * y],
      start: [-2.4, 0.9], box: [-2.8, 2.8, -1.2, 1.2], min: [0, 0], lab: "κ = " + (b / a) }; })(),
    aniso: (() => { const a = 0.02, b = 8.0; return {
      f: (x, y) => 0.5 * (a * x * x + b * y * y), g: (x, y) => [a * x, b * y],
      start: [-2.6, 0.8], box: [-2.9, 2.9, -1.1, 1.1], min: [0, 0], lab: "κ = " + (b / a) }; })(),
    rosen: { f: (x, y) => (1 - x) * (1 - x) + 20 * (y - x * x) * (y - x * x),
      g: (x, y) => [-2 * (1 - x) - 80 * x * (y - x * x), 40 * (y - x * x)],
      start: [-1.1, 1.3], box: [-1.6, 1.7, -0.6, 2.1], min: [1, 1], lab: "the banana" },
    saddle: { f: (x, y) => 0.5 * (x * x - 0.7 * y * y), g: (x, y) => [x, -0.7 * y],
      start: [1.2, 0.002], box: [-2.2, 2.2, -2.2, 2.2], min: [0, 0], lab: "λ₂ < 0" },
    beale: { f: (x, y) => { const a = 1.5 - x + x * y, b = 2.25 - x + x * y * y, c = 2.625 - x + x * y * y * y; return 0.06 * (a * a + b * b + c * c); },
      g: (x, y) => {
        const a = 1.5 - x + x * y, b = 2.25 - x + x * y * y, c = 2.625 - x + x * y * y * y;
        return [0.12 * (a * (y - 1) + b * (y * y - 1) + c * (y * y * y - 1)),
                0.12 * (a * x + b * 2 * x * y + c * 3 * x * y * y)];
      }, start: [-1.1, -0.6], box: [-2.2, 3.6, -1.6, 1.6], min: [3, 0.5], lab: "flat, then steep" }
  };
  const ORDER = ["sgd", "momentum", "momentumEwma", "nesterov", "adagrad", "rmsprop", "adam", "adamw"];
  const COL = { sgd: DC.muted, momentum: DC.accent, momentumEwma: DC.teal, nesterov: DC.violet,
    adagrad: DC.lime, rmsprop: DC.a2, adam: DC.good, adamw: DC.rose };

  function draw() {
    const S = SURF[se.value], lr = Math.pow(10, +le.value), T = +Te.value,
      b1 = +b1e.value, b2 = +b2e.value;
    document.getElementById("op-lrv").textContent = DL.sig(lr, 3);
    document.getElementById("op-Tv").textContent = T;
    document.getElementById("op-b1v").textContent = DL.fmt(b1, 2);
    document.getElementById("op-b2v").textContent = DL.fmt(b2, 4);
    const which = she.value === "all" ? ORDER.filter(k => k !== "momentumEwma") : she.value.split(",");

    const runs = which.map(key => {
      const O = DL.OPT[key], st = O.init(2);
      let w = S.start.slice();
      const P = [w.slice()], LS = [S.f(w[0], w[1])];
      const hp = { lr: lr, beta: b1, beta1: b1, beta2: b2, eps: 1e-8, wd: key === "adamw" ? 0.01 : 0 };
      for (let t = 0; t < T; t++) {
        const gv = S.g(w[0], w[1]);
        if (!isFinite(gv[0]) || !isFinite(gv[1])) break;
        hp.theta = w;
        const dx = O.step(st, gv, hp);
        w = [w[0] + dx[0], w[1] + dx[1]];
        if (!isFinite(w[0]) || Math.abs(w[0]) > 1e8 || Math.abs(w[1]) > 1e8) break;
        P.push(w.slice()); LS.push(S.f(w[0], w[1]));
      }
      return { key: key, label: O.label, P: P, L: LS, fin: LS[LS.length - 1], done: P.length > T };
    });

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 44, py = 34, PW = 330, PH = 340;
    const g1 = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(se.options[se.selectedIndex].text.split("(")[0].trim() + " — " + S.lab);
    const xw = d3.scaleLinear().domain([S.box[0], S.box[1]]).range([0, PW]);
    const yw = d3.scaleLinear().domain([S.box[2], S.box[3]]).range([PH, 0]);
    const RES = 72, vals = [];
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++)
      vals.push(Math.log1p(Math.max(0, S.f(xw.invert(i * PW / (RES - 1)), yw.invert(j * PH / (RES - 1))) - S.f(S.min[0], S.min[1]) + 1e-6)));
    const ext = d3.extent(vals);
    const cs = d3.scaleSequential(d3.interpolateViridis).domain([ext[1], ext[0]]);
    DL.cells(g1, 0, 0, PW / RES, RES, RES, (i, j) => cs(vals[j * RES + i]));
    g1.append("g").selectAll("path").data(d3.contours().size([RES, RES])
      .thresholds(d3.range(ext[0], ext[1], (ext[1] - ext[0]) / 13))(vals))
      .join("path").attr("d", d3.geoPath(d3.geoIdentity().scale(PW / RES)))
      .attr("fill", "none").attr("stroke", "#0f1117").attr("stroke-opacity", 0.28);
    DL.axisB(g1, xw, PH, 5, ""); DL.axisL(g1, yw, 5, "");
    g1.append("circle").attr("cx", xw(S.min[0])).attr("cy", yw(S.min[1])).attr("r", 4)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.8);
    const inB = p => isFinite(p[0]) && p[0] > S.box[0] - 0.4 && p[0] < S.box[1] + 0.4 && p[1] > S.box[2] - 0.4 && p[1] < S.box[3] + 0.4;
    runs.forEach(r => {
      const seg = r.P.filter(inB).map(p => [xw(p[0]), yw(p[1])]);
      DL.curve(g1, seg, { stroke: COL[r.key], w: 1.8, op: 0.95 });
      r.P.filter(inB).filter((_, i) => i % 10 === 0).forEach(p =>
        g1.append("circle").attr("cx", xw(p[0])).attr("cy", yw(p[1])).attr("r", 1.9).attr("fill", COL[r.key]));
    });
    g1.append("circle").attr("cx", xw(S.start[0])).attr("cy", yw(S.start[1])).attr("r", 4.5)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.6);

    /* loss curves */
    const QX = 430, QW = 300, QH = 175;
    const g2 = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
    g.append("text").attr("x", QX).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("loss against step, same start, same budget");
    const base = S.f(S.min[0], S.min[1]);
    const allv = runs.reduce((s, r) => s.concat(r.L.map(v => Math.max(v - base, 1e-14))), []).filter(isFinite);
    const xs = d3.scaleLinear().domain([0, T]).range([0, QW]);
    const ys = d3.scaleLog().domain([Math.max(1e-14, d3.min(allv)), Math.max(1, d3.max(allv))]).range([QH, 0]).clamp(true);
    DL.gridY(g2, ys, QW, 5);
    DL.axisB(g2, xs, QH, 5, "step"); DL.axisL(g2, ys, 5, "L − L*", d3.format(".0e"));
    runs.forEach(r => DL.curve(g2, r.L.map((v, i) => [xs(i), ys(Math.max(v - base, 1e-14))]),
      { stroke: COL[r.key], w: 1.9 }));
    DL.legend(g2, runs.map(r => ({ color: COL[r.key], label: r.label })), QW - 118, 10, { gap: 13, font: 9.5 });

    /* final-loss bars */
    const g3 = g.append("g").attr("transform", "translate(" + QX + "," + (py + QH + 46) + ")");
    g.append("text").attr("x", QX).attr("y", py + QH + 34).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("L − L* after " + T + " steps (lower is better)");
    const sorted = runs.slice().sort((a, b) => (a.fin - base) - (b.fin - base));
    const xb = d3.scaleLog().domain([Math.max(1e-14, d3.min(sorted, r => Math.max(r.fin - base, 1e-14))),
      Math.max(1, d3.max(sorted, r => Math.max(r.fin - base, 1e-14)))]).range([0, QW - 60]).clamp(true);
    sorted.forEach((r, i) => {
      const v = Math.max(r.fin - base, 1e-14);
      g3.append("rect").attr("x", 0).attr("y", i * 17).attr("width", Math.max(1, xb(v)))
        .attr("height", 12).attr("fill", COL[r.key]).attr("fill-opacity", 0.8);
      g3.append("text").attr("x", Math.max(1, xb(v)) + 5).attr("y", i * 17 + 10).attr("font-size", 9.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
        .text(r.label + "  " + (isFinite(v) ? DL.fmtE(v, 2) : "diverged"));
    });

    const win = sorted[0], lose = sorted[sorted.length - 1];
    const diverged = runs.filter(r => !r.done);
    NT.say("#opt-readout",
      "On the <b>" + se.options[se.selectedIndex].text.split("(")[0].trim() + "</b> at η = " + DL.sig(lr, 3) +
      " over " + T + " steps, the winner is <b>" + win.label + "</b> at L − L* = <b>" +
      DL.fmtE(Math.max(win.fin - base, 1e-14), 2) + "</b>; the worst of the selected is <b>" + lose.label +
      "</b> at <b>" + DL.fmtE(Math.max(lose.fin - base, 1e-14), 2) + "</b> — a factor of <b>" +
      DL.fmtE(Math.max(lose.fin - base, 1e-14) / Math.max(win.fin - base, 1e-14), 1) + "</b>. " +
      (diverged.length ? "<b>Diverged within the budget:</b> " + diverged.map(r => r.label).join(", ") + ". " : "") +
      "Change the surface and the ranking changes: no method here dominates on all five, which is the honest version of \"which optimiser should I use\". " +
      "Note also how much more the <i>learning-rate</i> slider moves these bars than the beta sliders do — that ordering of importance is §37's whole point." +
      (se.value === "ravine" || se.value === "aniso"
        ? " <b>One caveat on this surface:</b> the quadratic here is exactly diagonal, so a per-coordinate rescaling is exactly the right preconditioner and RMSProp and Adam flatter themselves. Rotate the problem — switch to Rosenbrock or Beale, whose curvature is not axis-aligned — and the diagonal methods lose most of that advantage, because a diagonal preconditioner cannot fix a rotated ill-conditioning."
        : ""));
  }
  [se, le, Te, b1e, b2e, she].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 17 · #wd-svg — coupled L2 against decoupled decay ═════════════════ */
(function () {
  const svg = d3.select("#wd-svg");
  if (svg.empty()) return;
  const W = 760, H = 390, NC = 24;
  const le = document.getElementById("wd-l"), lre = document.getElementById("wd-lr"),
    spe = document.getElementById("wd-sp"), Te = document.getElementById("wd-T"),
    oe = document.getElementById("wd-opt");

  function draw() {
    const lam = Math.pow(10, +le.value), lr = Math.pow(10, +lre.value),
      spread = +spe.value, T = +Te.value, opt = oe.value;
    document.getElementById("wd-lv").textContent = DL.fmt(lam, 4);
    document.getElementById("wd-lrv").textContent = DL.fmt(lr, 4);
    document.getElementById("wd-spv").textContent = DL.fmt(spread, 1);
    document.getElementById("wd-Tv").textContent = T;

    /* NC independent coordinates, each with its own gradient scale, spanning
       10^(−spread) to 10^(+spread). Each is a separate 1-D problem with a
       stationary gradient of that scale plus noise, so the ONLY thing that
       differs between coordinates is the gradient magnitude.                 */
    const scales = DL.linspace(-spread, spread, NC).map(e => Math.pow(10, e));
    const r = DL.rng(23);
    const coupled = [], decoupled = [], effC = [], effD = [];
    scales.forEach((sc, k) => {
      let wC = 1, wD = 1;
      let mC = 0, vC = 0, mD = 0, vD = 0;
      let decC = 0, decD = 0;
      const rr = DL.rng(100 + k);
      for (let t = 1; t <= T; t++) {
        const gRaw = sc * (1 + 0.3 * DL.randn(rr));
        if (opt === "sgd") {
          const dC = lr * lam * wC, dD = lr * lam * wD;
          decC += dC / Math.max(Math.abs(wC), 1e-12); decD += dD / Math.max(Math.abs(wD), 1e-12);
          wC = wC - lr * (gRaw + lam * wC);
          wD = wD - lr * gRaw - lr * lam * wD;
        } else {
          /* COUPLED: the penalty is inside g and passes through 1/√v̂ */
          const gC = gRaw + lam * wC;
          mC = 0.9 * mC + 0.1 * gC; vC = 0.999 * vC + 0.001 * gC * gC;
          const mhC = mC / (1 - Math.pow(0.9, t)), vhC = vC / (1 - Math.pow(0.999, t));
          /* the part of the step attributable to the penalty */
          const dC = lr * (lam * wC) / (Math.sqrt(vhC) + 1e-8);
          decC += dC / Math.max(Math.abs(wC), 1e-12);
          wC = wC - lr * mhC / (Math.sqrt(vhC) + 1e-8);
          /* DECOUPLED: the penalty never enters g */
          mD = 0.9 * mD + 0.1 * gRaw; vD = 0.999 * vD + 0.001 * gRaw * gRaw;
          const mhD = mD / (1 - Math.pow(0.9, t)), vhD = vD / (1 - Math.pow(0.999, t));
          const dD = lr * lam * wD;
          decD += dD / Math.max(Math.abs(wD), 1e-12);
          wD = wD - lr * mhD / (Math.sqrt(vhD) + 1e-8) - lr * lam * wD;
        }
      }
      coupled.push(Math.abs(wC)); decoupled.push(Math.abs(wD));
      effC.push(decC / T); effD.push(decD / T);
    });

    const F = DL.frame(svg, W, H, { l: 10, r: 10, t: 24, b: 8 });
    const g = F.g;
    const px = 56, py = 34, PW = 290, PH = 290;
    const xs = d3.scaleLog().domain([Math.pow(10, -spread) * 0.7, Math.pow(10, spread) * 1.4]).range([0, PW]);

    const g1 = g.append("g").attr("transform", "translate(" + px + "," + py + ")");
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("effective decay per step, by coordinate");
    const allE = effC.concat(effD).filter(v => v > 0);
    const ys = d3.scaleLog().domain([d3.min(allE) * 0.5, d3.max(allE) * 2]).range([PH, 0]).clamp(true);
    DL.gridY(g1, ys, PW, 5); DL.gridX(g1, xs, PH, 5);
    DL.axisB(g1, xs, PH, 5, "that coordinate's gradient scale", d3.format(".0e"));
    DL.axisL(g1, ys, 5, "fraction of |θ| removed / step", d3.format(".0e"));
    DL.curve(g1, effD.map((v, k) => [xs(scales[k]), ys(v)]), { stroke: DC.good, w: 2.6 });
    DL.curve(g1, effC.map((v, k) => [xs(scales[k]), ys(v)]), { stroke: DC.bad, w: 2.6 });
    effD.forEach((v, k) => g1.append("circle").attr("cx", xs(scales[k])).attr("cy", ys(v)).attr("r", 2.4).attr("fill", DC.good));
    effC.forEach((v, k) => g1.append("circle").attr("cx", xs(scales[k])).attr("cy", ys(v)).attr("r", 2.4).attr("fill", DC.bad));
    DL.legend(g1, [{ color: DC.good, label: "decoupled (AdamW): flat" },
      { color: DC.bad, label: "coupled L2: slope ≈ −1" }], 8, 14, { gap: 14, font: 10.5 });

    const QX = 425, QW = 285;
    const g2 = g.append("g").attr("transform", "translate(" + QX + "," + py + ")");
    g.append("text").attr("x", QX).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("|θ| after " + T + " steps, from θ₀ = 1");
    const allW = coupled.concat(decoupled).filter(v => v > 0);
    const yw2 = d3.scaleLog().domain([Math.max(1e-9, d3.min(allW) * 0.5), Math.max(1.5, d3.max(allW) * 2)])
      .range([PH, 0]).clamp(true);
    const xs2 = d3.scaleLog().domain(xs.domain()).range([0, QW]);
    DL.gridY(g2, yw2, QW, 5); DL.gridX(g2, xs2, PH, 5);
    DL.axisB(g2, xs2, PH, 5, "that coordinate's gradient scale", d3.format(".0e"));
    DL.axisL(g2, yw2, 5, "|θ| at the end", d3.format(".0e"));
    DL.curve(g2, decoupled.map((v, k) => [xs2(scales[k]), yw2(Math.max(v, 1e-12))]), { stroke: DC.good, w: 2.6 });
    DL.curve(g2, coupled.map((v, k) => [xs2(scales[k]), yw2(Math.max(v, 1e-12))]), { stroke: DC.bad, w: 2.6 });

    const rC = d3.max(effC) / d3.min(effC), rD = d3.max(effD) / d3.min(effD);
    NT.say("#wd-readout",
      "Across " + NC + " coordinates whose gradient magnitudes span <b>" +
      DL.fmtE(Math.pow(10, -spread), 1) + "</b> to <b>" + DL.fmtE(Math.pow(10, spread), 1) +
      "</b>, the <b>decoupled</b> decay removes the same fraction of every weight every step: largest ÷ smallest effective decay = <b>" +
      DL.fmt(rD, 4) + "</b>. The <b>coupled</b> L2 penalty removes a fraction that ranges over a factor of <b>" +
      DL.fmtE(rC, 2) + "</b> — and it removes the <i>most</i> from the coordinates with the <i>smallest</i> gradients, which is precisely backwards. " +
      (opt === "sgd"
        ? "With plain SGD selected the two curves coincide, which is the algebraic identity of §23: without the 1/√v̂ rescaling there is nothing to distinguish them, and this is why the two names were interchangeable for thirty years."
        : "Switch the optimiser to SGD and the two curves collapse onto one another — that is the identity, and the fact that it holds there and fails here is the entire AdamW argument."));
  }
  [le, lre, spe, Te, oe].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 18 · #sched-svg — the schedules ═════════════════ */
(function () {
  const svg = d3.select("#sched-svg");
  if (svg.empty()) return;
  const W = 760, H = 360;
  const ke = document.getElementById("sc-kind"), Te = document.getElementById("sc-T"),
    we = document.getElementById("sc-w"), te = document.getElementById("sc-t"),
    ae = document.getElementById("sc-all");
  const KINDS = ["constant", "cosine", "linear", "step", "exp", "power", "invsqrt", "onecycle"];

  function draw() {
    const kind = ke.value, T = +Te.value, warm = Math.min(+we.value, T - 1);
    te.max = T;
    const t = Math.min(+te.value, T);
    te.value = t;
    document.getElementById("sc-Tv").textContent = T;
    document.getElementById("sc-wv").textContent = warm;
    document.getElementById("sc-tv").textContent = t;
    const showAll = ae.checked;
    const o = { peak: 1, total: T, warm: warm, floor: 0, step: Math.max(1, Math.round(T / 4)), gamma: 0.1, power: 1 };
    const ts = DL.linspace(0, T, Math.min(600, T + 1)).map(Math.round);

    const F = DL.frame(svg, W, H, { l: 52, r: 14, t: 24, b: 8 });
    const g = F.g;
    const PW = 690, PH1 = 150, PH2 = 100;
    const x = d3.scaleLinear().domain([0, T]).range([0, PW]);

    const g1 = g.append("g");
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("η / η_peak against step");
    const y1 = d3.scaleLinear().domain([0, 1.08]).range([PH1, 0]);
    DL.gridY(g1, y1, PW, 4);
    DL.axisB(g1, x, PH1, 6, ""); DL.axisL(g1, y1, 4, "");
    if (warm > 0) {
      g1.append("rect").attr("x", 0).attr("y", 0).attr("width", x(warm)).attr("height", PH1)
        .attr("fill", DC.accent).attr("fill-opacity", 0.09);
      g1.append("text").attr("x", 4).attr("y", 12).attr("font-size", 10).attr("fill", DC.accent)
        .text("warmup — identical for every schedule");
    }
    if (showAll) KINDS.filter(k => k !== kind).forEach(k =>
      DL.curve(g1, ts.map(tt => [x(tt), y1(DL.clamp(DL.lrAt(k, tt, o), 0, 1.08))]),
        { stroke: DC.muted, w: 1.1, op: 0.4 }));
    DL.curve(g1, ts.map(tt => [x(tt), y1(DL.clamp(DL.lrAt(kind, tt, o), 0, 1.08))]), { stroke: DC.accent, w: 2.6 });
    const now = DL.lrAt(kind, t, o);
    g1.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", PH1)
      .attr("stroke", DC.a2).attr("stroke-opacity", 0.8);
    g1.append("circle").attr("cx", x(t)).attr("cy", y1(DL.clamp(now, 0, 1.08))).attr("r", 4.5).attr("fill", DC.a2);

    const g2 = g.append("g").attr("transform", "translate(0," + (PH1 + 60) + ")");
    g.append("text").attr("x", 0).attr("y", PH1 + 48).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("cumulative ∑η — the total distance the parameters may travel");
    const cum = {}, tot = {};
    KINDS.forEach(k => {
      let s = 0;
      const c = [];
      for (let tt = 0; tt <= T; tt++) { s += DL.lrAt(k, tt, o); if (tt % Math.max(1, Math.round(T / 600)) === 0) c.push([tt, s]); }
      cum[k] = c; tot[k] = s;
    });
    const y2 = d3.scaleLinear().domain([0, d3.max(KINDS, k => tot[k]) * 1.05]).range([PH2, 0]);
    DL.gridY(g2, y2, PW, 3);
    DL.axisB(g2, x, PH2, 6, "training step"); DL.axisL(g2, y2, 3, "", d3.format(".0f"));
    if (showAll) KINDS.filter(k => k !== kind).forEach(k =>
      DL.curve(g2, cum[k].map(c => [x(c[0]), y2(c[1])]), { stroke: DC.muted, w: 1.1, op: 0.4 }));
    DL.curve(g2, cum[kind].map(c => [x(c[0]), y2(c[1])]), { stroke: DC.accent, w: 2.6 });
    g2.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", PH2)
      .attr("stroke", DC.a2).attr("stroke-opacity", 0.8);

    const rank = KINDS.slice().sort((a, b) => tot[b] - tot[a]);
    NT.say("#sched-readout",
      "At step <b>" + t + "</b> of " + T + ", the <b>" + kind + "</b> schedule is at <b>" +
      DL.fmt(now, 4) + "</b> of its peak" + (t < warm ? " (still in warmup)" : "") +
      ". Total ∑η over the run: " + rank.map(k => (k === kind ? "<b>" : "") + k + " " + DL.fmt(tot[k] / T, 3) + "T" + (k === kind ? "</b>" : "")).join(", ") +
      ". Two schedules with very different shapes can permit almost the same total travel — cosine reaches <b>" +
      DL.fmt(tot.cosine / T, 3) + "T</b> against linear's <b>" + DL.fmt(tot.linear / T, 3) +
      "T</b>, a difference of only " + DL.fmt(100 * Math.abs(tot.cosine - tot.linear) / tot.linear, 1) +
      "%, which is why they perform so similarly and why arguing about the shape is usually less productive than getting the peak right (§25). " +
      "The warmup prefix is applied identically to every one of them and costs <b>" + DL.fmt(100 * warm / T, 1) +
      "%</b> of the budget here.");
  }
  [ke, Te, we, te, ae].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 19 · #lrrange-svg — the learning-rate range test ═════════════════ */
(function () {
  const svg = d3.select("#lrrange-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, NDATA = 96;
  const oe = document.getElementById("lt-opt"), ne = document.getElementById("lt-n"),
    se = document.getElementById("lt-sm"), we = document.getElementById("lt-w"),
    te = document.getElementById("lt-truth");

  const mkData = (N, seed) => {
    const r = DL.rng(seed), X = [], Y = [];
    for (let i = 0; i < N; i++) {
      const k = i % 2, t = 1.2 + 3.6 * (i / N), th = t + k * Math.PI + 0.2 * DL.randn(r);
      X.push([t * Math.cos(th) / 5, t * Math.sin(th) / 5]); Y.push([k]);
    }
    return { X: X, Y: Y };
  };
  const D = mkData(NDATA, 6), VAL = mkData(200, 77);
  const fresh = w => DL.mlpInit([2, w, w, 1], { act: "relu", out: "sigmoid", seed: 44 });

  function optFor(key, n) { return { O: DL.OPT[key === "momentum" ? "momentum" : key], st: DL.OPT[key === "momentum" ? "momentum" : key].init(n) }; }
  function flatten(net) {
    const v = [];
    for (let l = 0; l < net.L; l++) { net.W[l].forEach(r => r.forEach(x => v.push(x))); net.b[l].forEach(x => v.push(x)); }
    return v;
  }
  function unflatten(net, v) {
    let k = 0;
    for (let l = 0; l < net.L; l++) {
      for (let i = 0; i < net.sizes[l]; i++) for (let j = 0; j < net.sizes[l + 1]; j++) net.W[l][i][j] = v[k++];
      for (let j = 0; j < net.sizes[l + 1]; j++) net.b[l][j] = v[k++];
    }
  }
  function flatGrad(g, net) {
    const v = [];
    for (let l = 0; l < net.L; l++) { g.dW[l].forEach(r => r.forEach(x => v.push(x))); g.db[l].forEach(x => v.push(x)); }
    return v;
  }
  function step(net, S, hp) {
    const g = DL.backward(net, D.X, D.Y);
    const gv = flatGrad(g, net), th = flatten(net);
    hp.theta = th;
    const dx = S.O.step(S.st, gv, hp);
    unflatten(net, th.map((x, i) => x + dx[i]));
    return g.loss;
  }

  function draw() {
    const key = oe.value, NSTEP = +ne.value, sm = +se.value, wid = +we.value, truth = te.checked;
    document.getElementById("lt-nv").textContent = NSTEP;
    document.getElementById("lt-smv").textContent = DL.fmt(sm, 2);
    document.getElementById("lt-wv").textContent = wid;
    const LO = -7, HI = 1.2;

    /* the range test: ONE model, rate raised geometrically each step */
    const net = fresh(wid);
    const P = flatten(net).length;
    const S = optFor(key, P);
    const hp = { lr: 0, beta: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    const lrs = [], losses = [];
    let best = Infinity;
    for (let i = 0; i < NSTEP; i++) {
      const lr = Math.pow(10, LO + (HI - LO) * i / (NSTEP - 1));
      hp.lr = lr;
      const L = step(net, S, hp);
      if (!isFinite(L) || L > 6 * Math.min(best, 1)) { lrs.push(lr); losses.push(NaN); break; }
      best = Math.min(best, L);
      lrs.push(lr); losses.push(L);
    }
    const sm2 = [];
    let v = 0;
    losses.forEach((L, i) => {
      if (!isFinite(L)) { sm2.push(NaN); return; }
      v = sm * v + (1 - sm) * L;
      sm2.push(v / (1 - Math.pow(sm, i + 1)));
    });

    /* GROUND TRUTH: train a FRESH model for a full budget at each of 13 rates
       and report its HELD-OUT loss — the quantity the range test is trying to
       predict. Using the training loss instead would simply reward the largest
       rate that has not yet diverged, which is not the question. */
    const GTBUDGET = 250;
    let GT = null;
    if (truth) {
      GT = [];
      for (let k = 0; k < 13; k++) {
        const lr = Math.pow(10, LO + (HI - LO) * k / 12);
        const n2 = fresh(wid), S2 = optFor(key, P);
        const hp2 = { lr: lr, beta: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
        let ok = true;
        for (let t = 0; t < GTBUDGET; t++) { if (!isFinite(step(n2, S2, hp2))) { ok = false; break; } }
        GT.push([lr, ok ? DL.loss(n2, VAL.X, VAL.Y) : NaN]);
      }
    }

    const F = DL.frame(svg, W, H, { l: 56, r: 16, t: 24, b: 8 });
    const g = F.g;
    const PW = 660, PH = 290;
    const x = d3.scaleLog().domain([Math.pow(10, LO), Math.pow(10, HI)]).range([0, PW]);
    const valid = sm2.filter(isFinite).concat(GT ? GT.map(p => p[1]).filter(isFinite) : []);
    const y = d3.scaleLinear().domain([Math.min(0, d3.min(valid) * 0.9), d3.max(valid) * 1.08]).range([PH, 0]).clamp(true);
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("loss against learning rate — one pass, rate raised geometrically each step");
    DL.gridY(g, y, PW, 5); DL.gridX(g, x, PH, 6);
    DL.axisB(g, x, PH, 6, "learning rate η", d3.format(".0e")); DL.axisL(g, y, 5, "loss");
    const pts = sm2.map((L, i) => isFinite(L) ? [x(lrs[i]), y(L)] : null).filter(Boolean);
    DL.curve(g, pts, { stroke: DC.accent, w: 2.4 });
    if (GT) {
      const gp = GT.filter(p => isFinite(p[1])).map(p => [x(p[0]), y(p[1])]);
      DL.curve(g, gp, { stroke: DC.violet, w: 1.8, dash: "5 3" });
      GT.forEach(p => { if (isFinite(p[1])) g.append("circle").attr("cx", x(p[0])).attr("cy", y(p[1])).attr("r", 3).attr("fill", DC.violet); });
    }
    /* the three readings */
    const fin = sm2.map((L, i) => ({ L: L, lr: lrs[i] })).filter(d => isFinite(d.L));
    const minI = fin.reduce((a, d, i) => d.L < fin[a].L ? i : a, 0);
    let steepI = 1, best2 = 0;
    for (let i = 1; i <= minI; i++) {
      const sl = (fin[i - 1].L - fin[i].L) / Math.log10(fin[i].lr / fin[i - 1].lr);
      if (sl > best2) { best2 = sl; steepI = i; }
    }
    const divLr = isFinite(losses[losses.length - 1]) ? fin[fin.length - 1].lr * 1.6 : lrs[lrs.length - 1];
    [[fin[steepI].lr, DC.good, "steepest descent"], [fin[minI].lr, DC.a2, "curve minimum"],
     [divLr, DC.bad, "divergence"]].forEach(([v2, col, lab], i) => {
      g.append("line").attr("x1", x(v2)).attr("x2", x(v2)).attr("y1", 0).attr("y2", PH)
        .attr("stroke", col).attr("stroke-dasharray", "4 3");
      g.append("text").attr("x", x(v2) + 4).attr("y", 14 + i * 13).attr("font-size", 10).attr("fill", col)
        .text(lab + " " + DL.fmtE(v2, 1));
    });
    DL.legend(g, [{ color: DC.accent, label: "range test — ONE pass, " + fin.length + " steps" }]
      .concat(GT ? [{ color: DC.violet, label: "ground truth — 13 models × 250 steps, HELD-OUT loss", dash: "5 3" }] : []),
      12, PH - 34, { gap: 15, font: 10.5 });

    let gtBest = null;
    if (GT) { const ok = GT.filter(p => isFinite(p[1])); gtBest = ok.reduce((a, p) => p[1] < a[1] ? p : a, ok[0]); }
    NT.say("#lrrange-readout",
      "The range test says: steepest descent at η ≈ <b>" + DL.fmtE(fin[steepI].lr, 2) +
      "</b>, curve minimum at <b>" + DL.fmtE(fin[minI].lr, 2) + "</b>, divergence at <b>" +
      DL.fmtE(divLr, 2) + "</b> — a usable band of about <b>" +
      DL.fmtE(divLr / fin[steepI].lr, 1) + "×</b>. " +
      (gtBest ? (function () {
        const eS = Math.max(gtBest[0] / fin[steepI].lr, fin[steepI].lr / gtBest[0]);
        const eM = Math.max(gtBest[0] / fin[minI].lr, fin[minI].lr / gtBest[0]);
        return "The ground-truth sweep — thirteen separate models, each trained from scratch for " + GTBUDGET +
          " steps and scored on <i>held-out</i> data, costing about <b>" + DL.big(13 * GTBUDGET / fin.length) +
          "×</b> as much compute as the test — puts the optimum at <b>" + DL.fmtE(gtBest[0], 2) +
          "</b>. The steepest-descent reading is off by a factor of <b>" + DL.fmt(eS, 2) +
          "</b>; the curve-minimum reading by a factor of <b>" + DL.fmt(eM, 2) + "</b>. " +
          (eM < eS
            ? "<b>On this problem the curve minimum is the better estimate, and that is worth understanding rather than hiding.</b> The steepest-descent rule builds in a deliberate safety margin of three to ten times, because on a long run instability accumulates and a rate that survived 250 steps may not survive 250 000. This problem is small, easy and short, so the margin is pure loss here. Take the steepest-descent reading as a <i>peak</i> for a schedule with warmup and decay (§24), not as a constant rate — under a constant rate the curve minimum is closer, and under a real recipe it is not."
            : "Here the steepest-descent reading wins, which is the case the rule is written for.") + " ";
      })() : "") +
      "Note the shape regardless: flat, then a steep descent, then a near-vertical wall. That wall is §16's stability bound, and it is what makes a one-pass estimate possible at all — you are locating a cliff, not an optimum.");
  }
  [oe, ne, se, we, te].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 20 · #batch-svg — the batch-size ledger ═════════════════ */
(function () {
  const svg = d3.select("#batch-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const Be = document.getElementById("bt-B"), Pe = document.getElementById("bt-P"),
    Ae = document.getElementById("bt-A"), Ne = document.getElementById("bt-N"),
    Re = document.getElementById("bt-rule");
  const CAP = 80 * 1024 * 1024 * 1024;                 // 80 GiB of device memory
  const BREF = 32;

  function draw() {
    const B = Math.pow(2, +Be.value), P = +Pe.value, Aper = +Ae.value, N = +Ne.value, rule = Re.value;
    document.getElementById("bt-Bv").textContent = DL.big(B);

    const memP = P * 2, memM = P * 4, memG = P * 2, memS = P * 8;   // bf16 w+g, fp32 master + m + v
    const fixed = memP + memM + memG + memS;
    const actAt = b => b * Aper * 2;                                  // bf16 activations
    const total = b => fixed + actAt(b);
    const scale = b => rule === "linear" ? b / BREF : (rule === "sqrt" ? Math.sqrt(b / BREF) : 1);

    const F = DL.frame(svg, W, H, { l: 56, r: 14, t: 24, b: 8 });
    const g = F.g;
    const PW = 300, PH = 290;
    const x = d3.scaleLog().domain([1, 4096]).range([0, PW]);
    const bs = DL.linspace(0, 12, 97).map(e => Math.pow(2, e));

    const g1 = g.append("g");
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("device memory against batch size");
    const y = d3.scaleLog().domain([Math.max(1e6, fixed * 0.4), Math.max(CAP * 3, total(4096) * 1.4)])
      .range([PH, 0]).clamp(true);
    DL.gridY(g1, y, PW, 5); DL.gridX(g1, x, PH, 5);
    DL.axisB(g1, x, PH, 5, "batch size B", d3.format("d"));
    DL.axisL(g1, y, 5, "bytes", d => DL.big(d) + "B");
    const bands = [
      { f: () => memP + memG, c: DC.good, lab: "weights + grads (bf16)" },
      { f: () => memM, c: DC.teal, lab: "fp32 master copy" },
      { f: () => memS, c: DC.rose, lab: "Adam m and v (fp32)" },
      { f: b => actAt(b), c: DC.a2, lab: "activations (bf16)" }
    ];
    let acc = bs.map(() => 0);
    bands.forEach(bd => {
      const next = bs.map((b, i) => acc[i] + (typeof bd.f === "function" ? bd.f(b) : bd.f));
      const top = next.map((v, i) => [x(bs[i]), y(v)]);
      const bot = acc.map((v, i) => [x(bs[i]), y(Math.max(v, y.domain()[0]))]).reverse();
      DL.curve(g1, top.concat(bot), { stroke: bd.c, w: 1.2, fill: bd.c, fillOp: 0.35 });
      acc = next;
    });
    g1.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(CAP)).attr("y2", y(CAP))
      .attr("stroke", DC.bad).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 3");
    g1.append("text").attr("x", 3).attr("y", y(CAP) - 4).attr("font-size", 9.5).attr("fill", DC.bad)
      .text("80 GiB device");
    const bMax = (CAP - fixed) / (Aper * 2);
    if (bMax > 1 && bMax < 4096) {
      g1.append("line").attr("x1", x(bMax)).attr("x2", x(bMax)).attr("y1", 0).attr("y2", PH)
        .attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
      g1.append("text").attr("x", x(bMax) + 4).attr("y", 14).attr("font-size", 9.5).attr("fill", DC.bad)
        .text("B ≤ " + Math.floor(bMax));
    }
    g1.append("line").attr("x1", x(B)).attr("x2", x(B)).attr("y1", 0).attr("y2", PH)
      .attr("stroke", DC.ink).attr("stroke-opacity", 0.6);
    DL.legend(g1, bands.map(b => ({ color: b.c, label: b.lab })), 8, 14, { gap: 13, font: 9.5 });

    /* steps per epoch */
    const QX = 420, QW = 290, QH = 115;
    const g2 = g.append("g").attr("transform", "translate(" + QX + ",0)");
    g.append("text").attr("x", QX).attr("y", -10).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("optimiser updates in one epoch");
    const x2 = d3.scaleLog().domain([1, 4096]).range([0, QW]);
    const y2 = d3.scaleLog().domain([Math.max(1, N / 4096), N]).range([QH, 0]);
    DL.gridY(g2, y2, QW, 3);
    DL.axisB(g2, x2, QH, 5, "", d3.format("d")); DL.axisL(g2, y2, 3, "", d => DL.big(d));
    DL.curve(g2, bs.map(b => [x2(b), y2(Math.max(1, N / b))]), { stroke: DC.accent, w: 2.2 });
    g2.append("circle").attr("cx", x2(B)).attr("cy", y2(Math.max(1, N / B))).attr("r", 4.5).attr("fill", DC.accent);

    /* the scaling rule */
    const g3 = g.append("g").attr("transform", "translate(" + QX + "," + (QH + 66) + ")");
    g.append("text").attr("x", QX).attr("y", QH + 54).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("η relative to its value at B = " + BREF);
    const QH2 = 120;
    const y3 = d3.scaleLog().domain([1 / 64, 200]).range([QH2, 0]).clamp(true);
    DL.gridY(g3, y3, QW, 4);
    DL.axisB(g3, x2, QH2, 5, "batch size B", d3.format("d"));
    DL.axisL(g3, y3, 4, "", d3.format(".2g"));
    ["linear", "sqrt", "none"].forEach((rr, i) => {
      const f = b => rr === "linear" ? b / BREF : (rr === "sqrt" ? Math.sqrt(b / BREF) : 1);
      DL.curve(g3, bs.map(b => [x2(b), y3(DL.clamp(f(b), 1 / 64, 200))]),
        { stroke: rr === rule ? DC.accent : DC.muted, w: rr === rule ? 2.4 : 1.2, op: rr === rule ? 1 : 0.5 });
    });
    g3.append("circle").attr("cx", x2(B)).attr("cy", y3(DL.clamp(scale(B), 1 / 64, 200))).attr("r", 4.5).attr("fill", DC.accent);
    DL.legend(g3, [{ color: DC.accent, label: rule === "linear" ? "η ∝ B" : (rule === "sqrt" ? "η ∝ √B" : "η fixed") },
      { color: DC.muted, label: "the other two" }], 8, QH2 - 26, { gap: 13, font: 10 });

    const fits = total(B) <= CAP;
    NT.say("#batch-readout",
      "At B = <b>" + DL.big(B) + "</b> with " + DL.big(P) + " parameters: fixed cost <b>" +
      (fixed / 1073741824).toFixed(2) + " GiB</b> (weights + gradients in bf16, an fp32 master copy, and Adam's two fp32 buffers — none of it depends on B), activations <b>" +
      (actAt(B) / 1073741824).toFixed(2) + " GiB</b>, total <b>" + (total(B) / 1073741824).toFixed(2) +
      " GiB</b> against an 80 GiB device — <b>" + (fits ? "fits" : "DOES NOT FIT") + "</b>" +
      (bMax >= 1 ? ", and the largest batch that does is <b>" + Math.floor(bMax) + "</b>" : "") +
      ". One epoch is <b>" + DL.big(Math.max(1, Math.round(N / B))) + "</b> updates. Gradient noise is <b>" +
      DL.fmt(Math.sqrt(BREF / B), 3) + "×</b> its value at B = " + BREF + ", and the " +
      (rule === "none" ? "unscaled" : rule) + " rule puts the learning rate at <b>" + DL.fmt(scale(B), 3) +
      "×</b> its reference value. Note which rows moved: doubling B doubled the activation band and left every other band exactly where it was, halved the updates per epoch, and left the arithmetic per epoch unchanged.");
  }
  [Be, Pe, Ae, Ne, Re].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 21 · #prec-svg — mixed precision and loss scaling ═════════════════ */
(function () {
  const svg = d3.select("#prec-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const fe = document.getElementById("pr-fmt"), Se = document.getElementById("pr-S"),
    me = document.getElementById("pr-mu"), sde = document.getElementById("pr-sd"),
    ae = document.getElementById("pr-auto");
  const FMT = {
    f16: { lo: NT.F16.MIN_SUB, hi: NT.F16.MAX, lab: "float16", eps: NT.F16.EPS },
    bf16: { lo: Math.pow(2, -133), hi: 3.3895313892515355e38, lab: "bfloat16", eps: Math.pow(2, -7) },
    f32: { lo: NT.F32.MIN_SUB, hi: NT.F32.MAX, lab: "float32", eps: NT.F32.EPS }
  };

  function draw() {
    const fmt = FMT[fe.value], mu = +me.value, sd = +sde.value, auto = ae.checked;
    document.getElementById("pr-muv").textContent = "1e" + mu;
    document.getElementById("pr-sdv").textContent = DL.fmt(sd, 1);
    /* the fractions, computed from the LOG-NORMAL cdf, exactly */
    const Phi = z => 0.5 * (1 + DL.erf(z / Math.SQRT2));
    const under = S => Phi((Math.log10(fmt.lo) - Math.log10(S) - mu) / sd);
    const over = S => 1 - Phi((Math.log10(fmt.hi) - Math.log10(S) - mu) / sd);
    if (auto) {
      let bestK = 0, bestV = Infinity;
      for (let k = 0; k <= 24; k++) {
        const S = Math.pow(2, k), v = under(S) + 40 * over(S);
        if (v < bestV) { bestV = v; bestK = k; }
      }
      Se.value = bestK;
    }
    const K = +Se.value, S = Math.pow(2, K);
    document.getElementById("pr-Sv").textContent = DL.big(S);

    const F = DL.frame(svg, W, H, { l: 56, r: 14, t: 24, b: 8 });
    const g = F.g;
    const PW = 690, PH1 = 175, PH2 = 105;
    const LO = -50, HI = 42;
    const x = d3.scaleLinear().domain([LO, HI]).range([0, PW]);

    const g1 = g.append("g");
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("gradient magnitudes, scaled by S, against " + fmt.lab + "'s representable range");
    const dens = e => Math.exp(-0.5 * Math.pow((e - (mu + Math.log10(S))) / sd, 2));
    const y1 = d3.scaleLinear().domain([0, 1.12]).range([PH1, 0]);
    DL.axisB(g1, x, PH1, 8, "log₁₀ |gradient|", d => "1e" + d);
    DL.axisL(g1, y1, 3, "density");
    const lo10 = Math.log10(fmt.lo), hi10 = Math.log10(fmt.hi);
    g1.append("rect").attr("x", 0).attr("y", 0).attr("width", Math.max(0, x(lo10))).attr("height", PH1)
      .attr("fill", DC.bad).attr("fill-opacity", 0.12);
    g1.append("rect").attr("x", x(hi10)).attr("y", 0).attr("width", Math.max(0, PW - x(hi10))).attr("height", PH1)
      .attr("fill", DC.bad).attr("fill-opacity", 0.12);
    g1.append("rect").attr("x", Math.max(0, x(lo10))).attr("y", 0)
      .attr("width", Math.max(0, x(hi10) - x(lo10))).attr("height", PH1)
      .attr("fill", DC.good).attr("fill-opacity", 0.06);
    [[lo10, "underflow → 0", DC.bad], [hi10, "overflow → inf", DC.bad]].forEach(([v, lab, c]) => {
      g1.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", PH1)
        .attr("stroke", c).attr("stroke-width", 1.8);
      g1.append("text").attr("x", x(v) + (lab.indexOf("under") === 0 ? -4 : 4)).attr("y", 12)
        .attr("text-anchor", lab.indexOf("under") === 0 ? "end" : "start")
        .attr("font-size", 10).attr("fill", c).text(lab);
    });
    const es = DL.linspace(LO, HI, 400);
    DL.curve(g1, es.map(e => [x(e), y1(dens(e))]), { stroke: DC.accent, w: 2.2, fill: DC.accent, fillOp: 0.2 });
    /* the unscaled distribution, for reference */
    if (K > 0) DL.curve(g1, es.map(e => [x(e), y1(Math.exp(-0.5 * Math.pow((e - mu) / sd, 2)))]),
      { stroke: DC.muted, w: 1.3, dash: "4 3" });

    const g2 = g.append("g").attr("transform", "translate(0," + (PH1 + 62) + ")");
    g.append("text").attr("x", 0).attr("y", PH1 + 50).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("fraction lost, against the loss scale");
    const xs = d3.scaleLinear().domain([0, 24]).range([0, PW]);
    const ys = d3.scaleLinear().domain([0, 1]).range([PH2, 0]);
    DL.gridY(g2, ys, PW, 4);
    DL.axisB(g2, xs, PH2, 8, "loss scale S = 2^k", d => "2^" + d);
    DL.axisL(g2, ys, 4, "", d3.format(".0%"));
    const ks = DL.linspace(0, 24, 97);
    DL.curve(g2, ks.map(k => [xs(k), ys(under(Math.pow(2, k)))]), { stroke: DC.bad, w: 2.2 });
    DL.curve(g2, ks.map(k => [xs(k), ys(over(Math.pow(2, k)))]), { stroke: DC.a2, w: 2.2 });
    g2.append("line").attr("x1", xs(K)).attr("x2", xs(K)).attr("y1", 0).attr("y2", PH2)
      .attr("stroke", DC.ink).attr("stroke-opacity", 0.6);
    DL.legend(g2, [{ color: DC.bad, label: "flushed to zero" }, { color: DC.a2, label: "overflowed to inf" }],
      PW - 150, 10, { gap: 13, font: 10 });

    const u = under(S), o = over(S);
    NT.say("#prec-readout",
      fmt.lab + " represents magnitudes from <b>" + DL.fmtE(fmt.lo, 3) + "</b> to <b>" + DL.fmtE(fmt.hi, 3) +
      "</b> — a range of <b>" + DL.fmt(Math.log10(fmt.hi / fmt.lo), 1) + " decades</b>, against float32's " +
      DL.fmt(Math.log10(NT.F32.MAX / NT.F32.MIN_SUB), 1) + ". At S = <b>" + DL.big(S) +
      "</b>, <b>" + (100 * u).toFixed(2) + "%</b> of this gradient distribution flushes to exactly zero and <b>" +
      (100 * o).toFixed(3) + "%</b> overflows to infinity. " +
      (fe.value === "bf16"
        ? "Note the lower panel: bfloat16's underflow curve is pinned at zero across the whole slider range, because it has float32's exponent field. There is nothing for loss scaling to do, which is the entire practical argument for the format — at the cost of a machine epsilon of " +
          DL.fmtE(fmt.eps, 2) + " against float16's " + DL.fmtE(NT.F16.EPS, 2) + "."
        : (u > 0.01
          ? "Raise S until the red curve reaches zero — but not past the point where the orange curve leaves it, because <b>an overflowed gradient is unrecoverable</b> and the only correct response is to discard the step. That is what dynamic loss scaling automates: halve S on an overflow, double it after a run of clean steps."
          : "The scale is in the safe band: the left tail is inside the format and the right tail has not yet reached the ceiling. In a real run the distribution drifts, which is why the scale is adjusted dynamically rather than chosen once.")) +
      " S is a power of two so that multiplying and dividing by it change only the exponent field and are therefore exact.");
  }
  [fe, Se, me, sde, ae].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 22 · #bn-svg — BatchNorm, train and eval ═════════════════ */
(function () {
  const svg = d3.select("#bn-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, D = 3;
  const Ne = document.getElementById("bn-N"), me = document.getElementById("bn-m"),
    Te = document.getElementById("bn-T"), ke = document.getElementById("bn-mode"),
    ue = document.getElementById("bn-unb");
  const GAM = [1, 2, 0.5], BET = [0, -1, 0.5], EPS = 1e-5;
  const MU = [5, -2, 0], SD = [2, 0.5, 10];
  const XFIX = [6, -2.5, 3];
  const NAMES = ["feature 1", "feature 2", "feature 3"];
  const COLS = [DC.accent, DC.a2, DC.violet];

  function stats(N, mom, T, unbiased) {
    const r = DL.rng(7);
    let rm = DL.zeros(D), rv = new Array(D).fill(1);
    const trackM = [], trackV = [];
    for (let t = 0; t < T; t++) {
      const X = [];
      for (let i = 0; i < N; i++) X.push(MU.map((m, j) => m + SD[j] * DL.randn(r)));
      const B = DL.batchNorm(X, GAM, BET, EPS);
      for (let j = 0; j < D; j++) {
        rm[j] = (1 - mom) * rm[j] + mom * B.mu[j];
        rv[j] = (1 - mom) * rv[j] + mom * (unbiased && N > 1 ? B.vaUnbiased[j] : B.va[j]);
      }
      trackM.push(rm.slice()); trackV.push(rv.slice());
    }
    return { mu: rm, va: rv, trackM: trackM, trackV: trackV };
  }

  function draw() {
    const N = +Ne.value, mom = +me.value, T = +Te.value, mode = ke.value, unb = ue.checked;
    document.getElementById("bn-Nv").textContent = N;
    document.getElementById("bn-mv").textContent = DL.fmt(mom, 3);
    document.getElementById("bn-Tv").textContent = T;
    const S = stats(N, mom, T, unb);

    const F = DL.frame(svg, W, H, { l: 56, r: 16, t: 24, b: 8 });
    const g = F.g;

    if (mode === "stats") {
      const PW = 300, PH = 300;
      [["running mean", S.trackM, MU, 0], ["running variance", S.trackV, SD.map(s => s * s), 400]].forEach(([lab, tr, truth, ox]) => {
        const gg = g.append("g").attr("transform", "translate(" + ox + ",26)");
        g.append("text").attr("x", ox).attr("y", 12).attr("font-size", 11).attr("fill", DC.ink)
          .attr("font-weight", 600).text(lab + " against step");
        const x = d3.scaleLinear().domain([0, T]).range([0, PW]);
        const allv = tr.reduce((s2, r) => s2.concat(r), []).concat(truth);
        const y = d3.scaleLinear().domain([d3.min(allv) - 2, d3.max(allv) * 1.1 + 2]).range([PH, 0]);
        DL.gridY(gg, y, PW, 5);
        DL.axisB(gg, x, PH, 5, "step"); DL.axisL(gg, y, 5, "");
        for (let j = 0; j < D; j++) {
          gg.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(truth[j])).attr("y2", y(truth[j]))
            .attr("stroke", COLS[j]).attr("stroke-dasharray", "4 3").attr("stroke-opacity", 0.7);
          DL.curve(gg, tr.map((r, i) => [x(i), y(r[j])]), { stroke: COLS[j], w: 2 });
        }
        DL.legend(gg, NAMES.map((n, j) => ({ color: COLS[j], label: n })), 8, 14, { gap: 13, font: 10 });
      });
      NT.say("#bn-readout",
        "After " + T + " steps at momentum " + DL.fmt(mom, 3) + " and batch " + N + ", the running means are <b>(" +
        S.mu.map(v => DL.fmt(v, 3)).join(", ") + ")</b> against a true (" + MU.join(", ") +
        "), and the running variances are <b>(" + S.va.map(v => DL.fmt(v, 3)).join(", ") +
        ")</b> against a true (" + SD.map(s => s * s).join(", ") +
        "). These are not learned by gradient descent — they are statistics accumulated in the forward pass, and they are the only parameters of the model that training changes without a gradient. " +
        (unb ? "The unbiased (1/(N−1)) variance is in use, which is what a framework does." :
          "The biased (1/N) variance is in use here, so the running estimate is systematically low by a factor of " +
          DL.fmt(N / Math.max(1, N - 1), 4) + " — " + DL.fmt(100 * (1 - (N - 1) / N), 1) + "% at N = " + N + ".") +
        " A smaller momentum converges more slowly and more smoothly; a larger one tracks faster and is noisier.");
      return;
    }

    if (mode === "fold") {
      const r = DL.rng(3);
      const Win = DL.zeros2(4, D), bin = DL.zeros(D), Xt = DL.zeros2(5, 4);
      for (let i = 0; i < 4; i++) for (let j = 0; j < D; j++) Win[i][j] = DL.randn(r);
      for (let j = 0; j < D; j++) bin[j] = DL.randn(r);
      for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) Xt[i][j] = DL.randn(r);
      const Z = DL.addRow(DL.matmul(Xt, Win), bin);
      const Y1 = DL.batchNorm(Z, GAM, BET, EPS, { mu: S.mu, va: S.va }).Y;
      const inv = S.va.map(v => 1 / Math.sqrt(v + EPS));
      const Wp = Win.map(row => row.map((v, j) => GAM[j] * v * inv[j]));
      const bp = bin.map((v, j) => GAM[j] * (v - S.mu[j]) * inv[j] + BET[j]);
      const Y2 = DL.addRow(DL.matmul(Xt, Wp), bp);
      let worst = 0;
      Y1.forEach((row, i) => row.forEach((v, j) => worst = Math.max(worst, Math.abs(v - Y2[i][j]))));
      DL.matText(g, Win, 10, 60, { label: "W  (4 × 3)", dp: 3, lead: 15 });
      DL.matText(g, [bin], 10, 160, { label: "b", dp: 3 });
      DL.matText(g, Wp, 240, 60, { label: "W′ = γ⊙W/σ", dp: 3, lead: 15 });
      DL.matText(g, [bp], 240, 160, { label: "b′ = γ⊙(b − μ)/σ + β", dp: 3 });
      DL.matText(g, Y1, 10, 240, { label: "BN(XW + b), eval mode", dp: 5, lead: 14 });
      DL.matText(g, Y2, 300, 240, { label: "X W′ + b′  — one linear layer", dp: 5, lead: 14 });
      const k = DL.kv(g, 560, 70, { keyW: 130, lead: 17 });
      k("max |difference|", DL.fmtE(worst, 2), worst < 1e-12 ? DC.good : DC.bad, true);
      k("float64 epsilon", DL.fmtE(NT.F32.EPS * 1e-9, 2), DC.muted);
      k("BN cost at inference", "0", DC.good, true);
      NT.say("#bn-readout",
        "At inference the layer is affine with <i>fixed</i> coefficients, so it can be folded into the linear layer before it: with W′ = γ⊙W/σ and b′ = γ⊙(b − μ)/σ + β the pair computes exactly XW′ + b′. Maximum discrepancy over the whole batch: <b>" +
        DL.fmtE(worst, 2) + "</b> — floating-point noise. The layer's entire inference cost disappears, which is why deployment toolchains do this automatically. Note that folding is only possible because <b>eval mode</b> uses fixed statistics; there is nothing to fold in training mode, where the coefficients depend on the batch.");
      return;
    }

    /* both: the fixed input through the layer, many times */
    const r = DL.rng(31), TR = 400;
    const outs = [];
    for (let k2 = 0; k2 < TR; k2++) {
      const X = [XFIX.slice()];
      for (let i = 1; i < N; i++) X.push(MU.map((m, j) => m + SD[j] * DL.randn(r)));
      outs.push(DL.batchNorm(X, GAM, BET, EPS).Y[0]);
    }
    const ev = DL.batchNorm([XFIX], GAM, BET, EPS, { mu: S.mu, va: S.va }).Y[0];
    const PW = 690, PH = 300;
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the SAME input x = [6, −2.5, 3], put through the layer " + TR + " times in train mode, and once in eval mode");
    const all = outs.reduce((s2, o) => s2.concat(o), []).concat(ev).concat(BET);
    const y = d3.scaleLinear().domain([d3.min(all) - 0.6, d3.max(all) + 0.6]).range([PH, 0]);
    const bandW = PW / D;
    DL.axisL(g, y, 5, "layer output");
    for (let j = 0; j < D; j++) {
      const cx = bandW * (j + 0.5);
      g.append("text").attr("x", cx).attr("y", PH + 24).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", DC.muted).text(NAMES[j]);
      outs.forEach((o, i) => g.append("circle")
        .attr("cx", cx + (DL.rng(i * 31 + j)() - 0.5) * bandW * 0.55)
        .attr("cy", y(o[j])).attr("r", 1.7).attr("fill", COLS[j]).attr("fill-opacity", 0.35));
      g.append("line").attr("x1", cx - bandW * 0.36).attr("x2", cx + bandW * 0.36)
        .attr("y1", y(ev[j])).attr("y2", y(ev[j])).attr("stroke", DC.good).attr("stroke-width", 2.4);
      g.append("line").attr("x1", cx - bandW * 0.36).attr("x2", cx + bandW * 0.36)
        .attr("y1", y(BET[j])).attr("y2", y(BET[j])).attr("stroke", DC.bad)
        .attr("stroke-width", 1.8).attr("stroke-dasharray", "5 3");
      const vals = outs.map(o => o[j]);
      g.append("text").attr("x", cx).attr("y", 16).attr("text-anchor", "middle").attr("font-size", 10)
        .attr("fill", DC.ink).text("train sd " + DL.fmt(DL.std(vals), 4));
    }
    DL.legend(g, [{ color: COLS[0], label: "train mode, one point per random batch" },
      { color: DC.good, label: "eval mode — deterministic" },
      { color: DC.bad, label: "β, the learned offset", dash: "5 3" }], 8, 34, { gap: 14, font: 10.5 });

    const sds = [0, 1, 2].map(j => DL.std(outs.map(o => o[j])));
    NT.say("#bn-readout",
      "At batch size <b>" + N + "</b>, the same input produces train-mode outputs with standard deviations <b>(" +
      sds.map(v => DL.fmt(v, 5)).join(", ") + ")</b> across random batch companions. " +
      (N === 1
        ? "<b>At N = 1 every one of them is exactly zero and every output is exactly β = (" + BET.join(", ") +
          "):</b> the batch mean IS the example's own value, so x̂ = 0 identically and the input has been <b>annihilated</b>. Zero information passes through the layer."
        : "In training mode the layer is <b>not a function of x</b> — it is a function of x and of whichever other examples happened to share its batch. Drag the batch size down to 1 and watch every point collapse onto β.") +
      " Eval mode gives <b>(" + ev.map(v => DL.fmt(v, 4)).join(", ") +
      ")</b>, deterministically, from the running statistics. The gap between the green line and the centre of the cloud is the train/eval asymmetry, and forgetting to cross it is the most expensive single bug in this domain.");
  }
  [Ne, me, Te, ke, ue].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 23 · #norms-svg — which axis each layer reduces over ═════════════════ */
(function () {
  const svg = d3.select("#norms-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const ke = document.getElementById("nm-kind"), Ne = document.getElementById("nm-N"),
    de = document.getElementById("nm-d"), re = document.getElementById("nm-r"),
    she = document.getElementById("nm-shift");
  const INFO = {
    bn: { lab: "BatchNorm", par: d => 2 * d, batch: true, tve: true, red: "down a column (the batch), per feature" },
    ln: { lab: "LayerNorm", par: d => 2 * d, batch: false, tve: false, red: "across a row (the features), per example" },
    rms: { lab: "RMSNorm", par: d => d, batch: false, tve: false, red: "across a row, second moment only — no centring" },
    gn: { lab: "GroupNorm", par: d => 2 * d, batch: false, tve: false, red: "across a GROUP of features within a row" },
    in: { lab: "InstanceNorm", par: d => 2 * d, batch: false, tve: false, red: "each feature of each example on its own" }
  };

  function normalise(kind, X) {
    const N = X.length, d = X[0].length, e = 1e-5;
    if (kind === "bn") return DL.batchNorm(X, null, null, e).Y;
    if (kind === "ln") return DL.layerNorm(X, null, null, e);
    if (kind === "rms") return DL.rmsNorm(X, null, e);
    if (kind === "in") return X.map(r2 => r2.map(() => 0));           // one value: (x−x)/0 → 0
    /* group norm: 4 groups across the features of each row */
    const G = 4, per = Math.max(1, Math.floor(d / G));
    return X.map(r2 => {
      const out = r2.slice();
      for (let gI = 0; gI < G; gI++) {
        const lo = gI * per, hi = (gI === G - 1) ? d : Math.min(d, lo + per);
        if (hi <= lo) continue;
        const seg = r2.slice(lo, hi), m = DL.mean(seg), v = DL.variance(seg);
        for (let j = lo; j < hi; j++) out[j] = (r2[j] - m) / Math.sqrt(v + e);
      }
      return out;
    });
  }
  function members(kind, N, d, i, j) {          // cells sharing i,j's statistics
    const out = [];
    if (kind === "bn") { for (let a = 0; a < N; a++) out.push([a, j]); }
    else if (kind === "ln" || kind === "rms") { for (let b = 0; b < d; b++) out.push([i, b]); }
    else if (kind === "in") out.push([i, j]);
    else {
      const G = 4, per = Math.max(1, Math.floor(d / G));
      const gI = Math.min(G - 1, Math.floor(j / per));
      const lo = gI * per, hi = (gI === G - 1) ? d : Math.min(d, lo + per);
      for (let b = lo; b < hi; b++) out.push([i, b]);
    }
    return out;
  }

  function draw() {
    const kind = ke.value, N = +Ne.value, d = +de.value, shift = she.checked;
    re.max = N - 1;
    const row = Math.min(+re.value, N - 1);
    re.value = row;
    document.getElementById("nm-Nv").textContent = N;
    document.getElementById("nm-dv").textContent = d;
    document.getElementById("nm-rv").textContent = row;

    const r = DL.rng(5), X = [];
    for (let i = 0; i < N; i++) {
      const rowv = [];
      for (let j = 0; j < d; j++) rowv.push(2 * DL.randn(r) + (j % 3) - 1 + (shift ? 10 : 0));
      X.push(rowv);
    }
    const Y = normalise(kind, X);
    /* the same input WITHOUT the shift, to measure invariance */
    const r2 = DL.rng(5), X0 = [];
    for (let i = 0; i < N; i++) {
      const rowv = [];
      for (let j = 0; j < d; j++) rowv.push(2 * DL.randn(r2) + (j % 3) - 1);
      X0.push(rowv);
    }
    const Y0 = normalise(kind, X0);
    let maxDiff = 0;
    Y.forEach((rw, i) => rw.forEach((v, j) => maxDiff = Math.max(maxDiff, Math.abs(v - Y0[i][j]))));

    const F = DL.frame(svg, W, H, { l: 16, r: 16, t: 26, b: 8 });
    const g = F.g;
    const cw = Math.min(24, 300 / d), ch = Math.min(24, 220 / N);
    const cols = d3.scaleSequential(d3.interpolateRdYlBu)
      .domain(d3.extent(X.reduce((s, rw) => s.concat(rw), [])).reverse());
    const cols2 = d3.scaleSequential(d3.interpolateRdYlBu)
      .domain(d3.extent(Y.reduce((s, rw) => s.concat(rw), [])).reverse());
    const mem = new Set(members(kind, N, d, row, Math.min(3, d - 1)).map(p => p[0] + "," + p[1]));

    [[0, X, cols, "input X  (N × d)"], [370, Y, cols2, INFO[kind].lab + "(X)"]].forEach(([ox, M, sc, lab]) => {
      g.append("text").attr("x", ox).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text(lab);
      const gg = g.append("g").attr("transform", "translate(" + ox + ",14)");
      for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) {
        const hot = mem.has(i + "," + j);
        gg.append("rect").attr("x", j * cw).attr("y", i * ch).attr("width", cw - 1).attr("height", ch - 1)
          .attr("fill", sc(M[i][j])).attr("fill-opacity", hot ? 1 : 0.42)
          .attr("stroke", hot ? DC.ink : "none").attr("stroke-width", hot ? 1.4 : 0);
      }
      gg.append("text").attr("x", 0).attr("y", N * ch + 14).attr("font-size", 10).attr("fill", DC.muted)
        .text("features →");
      gg.append("text").attr("x", -6).attr("y", -3).attr("font-size", 10).attr("fill", DC.muted)
        .text("examples ↓");
    });
    g.append("text").attr("x", 0).attr("y", N * ch + 44).attr("font-size", 11).attr("fill", DC.a2)
      .attr("font-weight", 600).text("outlined: the cells that share one mean and variance — reduced " + INFO[kind].red);

    /* the highlighted row, numerically */
    const yv = N * ch + 74;
    DL.matText(g, [X[row].slice(0, Math.min(8, d))], 0, yv, { label: "row " + row + " in", dp: 3, size: 10.5 });
    DL.matText(g, [Y[row].slice(0, Math.min(8, d))], 370, yv, { label: "row " + row + " out", dp: 4, size: 10.5 });

    const k = DL.kv(g, 0, yv + 46, { keyW: 210, lead: 16 });
    k("learned parameters", String(INFO[kind].par(d)) + "  (d = " + d + ")", DC.ink);
    k("depends on the other examples?", INFO[kind].batch ? "YES" : "no", INFO[kind].batch ? DC.bad : DC.good, true);
    k("different function at eval time?", INFO[kind].tve ? "YES" : "no", INFO[kind].tve ? DC.bad : DC.good, true);
    k("output changed by adding 10?", DL.fmtE(maxDiff, 2), maxDiff < 1e-9 ? DC.good : DC.a2, true);

    NT.say("#norms-readout",
      "<b>" + INFO[kind].lab + "</b> reduces " + INFO[kind].red + ", so a single example's output " +
      (INFO[kind].batch ? "<b>does</b> depend on the other examples in its batch — which is why it needs running statistics and a separate evaluation-time path"
        : "does <b>not</b> depend on the other examples at all — no running statistics, no train/eval gap, identical behaviour at batch size 1 and 1024") +
      ". Adding 10 to every feature of every example changes its output by at most <b>" + DL.fmtE(maxDiff, 2) + "</b>: " +
      (maxDiff < 1e-9 ? "it is <b>exactly shift-invariant</b>." :
        "it is <b>not</b> shift-invariant, because " + (kind === "rms" ? "RMSNorm divides by the root mean square without first removing the mean — the re-centring it drops is exactly what would have made it invariant."
          : "the shift moves the statistics this layer reduces over.")) +
      " Learned parameters: <b>" + INFO[kind].par(d) + "</b> — RMSNorm needs half as many as LayerNorm because it has no offset to learn.");
  }
  [ke, Ne, de, re, she].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 24 · #l2-svg — shrinkage, L1, and early stopping ═════════════════ */
(function () {
  const svg = d3.select("#l2-svg");
  if (svg.empty()) return;
  const W = 760, H = 390;
  const ae = document.getElementById("l2-a"), l2e = document.getElementById("l2-l2"),
    ve = document.getElementById("l2-view"), te = document.getElementById("l2-tau"),
    oe = document.getElementById("l2-l1");
  const L1V = 10, WSTAR = [1, 1], ETA = 0.02;

  function draw() {
    const alpha = Math.pow(10, +ae.value), lam2 = Math.pow(10, +l2e.value),
      view = ve.value, tau = +te.value, showL1 = oe.checked;
    document.getElementById("l2-av").textContent = DL.fmt(alpha, 3);
    document.getElementById("l2-l2v").textContent = DL.fmt(lam2, 3);
    document.getElementById("l2-tauv").textContent = tau;

    const F = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 8 });
    const g = F.g;

    if (view === "shrink") {
      const PW = 660, PH = 290;
      g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("the factor each coordinate of w* is multiplied by, against that coordinate's curvature");
      const x = d3.scaleLog().domain([1e-3, 1e3]).range([0, PW]);
      const y = d3.scaleLinear().domain([-0.04, 1.06]).range([PH, 0]);
      DL.gridY(g, y, PW, 5); DL.gridX(g, x, PH, 6);
      DL.axisB(g, x, PH, 6, "curvature λ of that direction", d3.format(".0e"));
      DL.axisL(g, y, 5, "shrinkage factor");
      const ls = DL.linspace(-3, 3, 300).map(e => Math.pow(10, e));
      DL.curve(g, ls.map(l => [x(l), y(l / (l + alpha))]), { stroke: DC.accent, w: 2.6 });
      if (showL1) DL.curve(g, ls.map(l => [x(l), y(Math.max(0, 1 - alpha / (l * 1)) )]), { stroke: DC.a2, w: 2.4, dash: "5 3" });
      /* early stopping's factor, for reference */
      DL.curve(g, ls.map(l => [x(l), y(DL.clamp(1 - Math.pow(1 - ETA * l, tau), -0.04, 1.06))]),
        { stroke: DC.violet, w: 2, dash: "2 3" });
      [[alpha, DC.accent, "α = " + DL.fmt(alpha, 3)], [lam2, DC.good, "λ₂ (soft direction)"], [L1V, DC.bad, "λ₁ (stiff direction)"]]
        .forEach(([v, col, lab], i) => {
          if (v < 1e-3 || v > 1e3) return;
          g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", PH)
            .attr("stroke", col).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.8);
          g.append("text").attr("x", x(v) + 4).attr("y", 14 + i * 13).attr("font-size", 10).attr("fill", col).text(lab);
        });
      DL.legend(g, [{ color: DC.accent, label: "L2: λ/(λ+α) — never exactly zero" }]
        .concat(showL1 ? [{ color: DC.a2, label: "L1: max(0, 1 − α/λ) — exactly zero below the threshold", dash: "5 3" }] : [])
        .concat([{ color: DC.violet, label: "early stopping at τ = " + tau + ": 1 − (1 − ηλ)^τ", dash: "2 3" }]),
        14, PH - 46, { gap: 15, font: 10.5 });
      NT.say("#l2-readout",
        "At α = <b>" + DL.fmt(alpha, 4) + "</b>: the stiff direction (λ = " + L1V + ") keeps <b>" +
        DL.fmt(L1V / (L1V + alpha), 6) + "</b> of its unregularised value, while the soft one (λ = " +
        DL.fmt(lam2, 3) + ") keeps only <b>" + DL.fmt(lam2 / (lam2 + alpha), 6) + "</b> — a ratio of <b>" +
        DL.fmt((L1V / (L1V + alpha)) / (lam2 / (lam2 + alpha)), 2) +
        "</b> from the same penalty on the same starting weights. That is the entire content of the L2 analysis: <b>it shrinks hardest where the data cared least</b>. " +
        (showL1 ? "The L1 curve is not a scaled version of the L2 one — it is exactly zero below λ = α, which is what makes L1 a feature selector and L2 a shrinker. "
          : "") +
        "The dotted curve is early stopping at τ = " + tau + " steps, and its close agreement with L2 in the low-curvature region — and its disagreement in the high-curvature one — is §34's approximation, drawn.");
      return;
    }

    if (view === "contour") {
      const PW = 300, PH = 290;
      const H2 = [L1V, 0, lam2];
      const Lf = w => 0.5 * (H2[0] * (w[0] - WSTAR[0]) * (w[0] - WSTAR[0]) + H2[2] * (w[1] - WSTAR[1]) * (w[1] - WSTAR[1]));
      const wt = [L1V / (L1V + alpha) * WSTAR[0], lam2 / (lam2 + alpha) * WSTAR[1]];
      g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("the unregularised contours, the penalty's circles, and where the optimum lands");
      const gg = g.append("g").attr("transform", "translate(0,16)");
      const x = d3.scaleLinear().domain([-0.4, 1.6]).range([0, PW * 1.6]);
      const y = d3.scaleLinear().domain([-0.4, 1.6]).range([PH, 0]);
      DL.gridY(gg, y, PW * 1.6, 5); DL.gridX(gg, x, PH, 5);
      DL.axisB(gg, x, PH, 5, "w₁ (stiff, λ = " + L1V + ")");
      DL.axisL(gg, y, 5, "w₂ (soft, λ = " + DL.fmt(lam2, 2) + ")");
      d3.range(1, 9).forEach(k => {
        const lev = 0.02 * k * k;
        const pts = DL.linspace(0, 2 * Math.PI, 120).map(t => [
          x(WSTAR[0] + Math.sqrt(2 * lev / H2[0]) * Math.cos(t)),
          y(WSTAR[1] + Math.sqrt(2 * lev / H2[2]) * Math.sin(t))]);
        DL.curve(gg, pts, { stroke: DC.accent, w: 1.1, op: 0.55 });
      });
      d3.range(1, 6).forEach(k => {
        const rr = 0.28 * k;
        DL.curve(gg, DL.linspace(0, 2 * Math.PI, 120).map(t => [x(rr * Math.cos(t)), y(rr * Math.sin(t))]),
          { stroke: DC.a2, w: 1.1, dash: "3 3", op: 0.55 });
      });
      gg.append("circle").attr("cx", x(WSTAR[0])).attr("cy", y(WSTAR[1])).attr("r", 5).attr("fill", DC.accent);
      gg.append("text").attr("x", x(WSTAR[0]) + 8).attr("y", y(WSTAR[1]) - 5).attr("font-size", 10.5)
        .attr("fill", DC.accent).text("w*  (unregularised)");
      gg.append("circle").attr("cx", x(wt[0])).attr("cy", y(wt[1])).attr("r", 5).attr("fill", DC.good);
      gg.append("text").attr("x", x(wt[0]) + 8).attr("y", y(wt[1]) + 14).attr("font-size", 10.5)
        .attr("fill", DC.good).text("w̃  (L2, α = " + DL.fmt(alpha, 3) + ")");
      DL.arrow(gg, x(WSTAR[0]), y(WSTAR[1]), x(wt[0]), y(wt[1]), { color: DC.good, w: 1.6, head: 6 });
      const k2 = DL.kv(g, 520, 60, { keyW: 130, lead: 17 });
      k2("w*", "(" + WSTAR.join(", ") + ")", DC.accent);
      k2("w̃ solved exactly", "(" + wt.map(v => DL.fmt(v, 5)).join(", ") + ")", DC.good, true);
      k2("λ₁/(λ₁+α)", DL.fmt(L1V / (L1V + alpha), 6), DC.ink);
      k2("λ₂/(λ₂+α)", DL.fmt(lam2 / (lam2 + alpha), 6), DC.ink);
      k2("moved along w₁", DL.fmt(WSTAR[0] - wt[0], 5), DC.muted);
      k2("moved along w₂", DL.fmt(WSTAR[1] - wt[1], 5), DC.a2, true);
      NT.say("#l2-readout",
        "The regularised optimum sits where the two sets of contours are tangent. It has moved <b>" +
        DL.fmt(WSTAR[1] - wt[1], 5) + "</b> along the soft axis and only <b>" + DL.fmt(WSTAR[0] - wt[0], 5) +
        "</b> along the stiff one — a ratio of <b>" +
        DL.fmt((WSTAR[1] - wt[1]) / Math.max(1e-9, WSTAR[0] - wt[0]), 1) +
        "</b>. Both coordinates started at 1 and both received the same penalty; the only thing that differed was how much the <i>data</i> objected, and that is what decided the outcome.");
      return;
    }

    /* early: trajectory from the origin against the L2 solution */
    const PW = 660, PH = 290;
    const lams = [L1V, lam2];
    const traj = [];
    for (let t = 0; t <= 400; t++) traj.push(lams.map(l => (1 - Math.pow(1 - ETA * l, t)) * 1));
    const aEq = 1 / (ETA * tau);
    const wl2 = lams.map(l => l / (l + aEq));
    const wes = lams.map(l => 1 - Math.pow(1 - ETA * l, tau));
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the trajectory from the origin, and the L2 solution the τ ≈ 1/(ηα) correspondence predicts");
    const x = d3.scaleLinear().domain([-0.05, 1.15]).range([0, 420]);
    const y = d3.scaleLinear().domain([-0.05, 1.15]).range([PH, 0]);
    const gg = g.append("g").attr("transform", "translate(0,16)");
    DL.gridY(gg, y, 420, 5); DL.gridX(gg, x, PH, 5);
    DL.axisB(gg, x, PH, 5, "w₁ / w*₁  (stiff, ηλ = " + DL.fmt(ETA * L1V, 3) + ")");
    DL.axisL(gg, y, 5, "w₂ / w*₂  (soft, ηλ = " + DL.fmt(ETA * lam2, 4) + ")");
    DL.curve(gg, traj.map(p => [x(p[0]), y(p[1])]), { stroke: DC.violet, w: 2.2 });
    traj.filter((_, i) => i % 20 === 0).forEach(p =>
      gg.append("circle").attr("cx", x(p[0])).attr("cy", y(p[1])).attr("r", 1.8).attr("fill", DC.violet));
    gg.append("circle").attr("cx", x(1)).attr("cy", y(1)).attr("r", 5).attr("fill", DC.accent);
    gg.append("text").attr("x", x(1) - 6).attr("y", y(1) - 8).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", DC.accent).text("w*");
    gg.append("circle").attr("cx", x(wes[0])).attr("cy", y(wes[1])).attr("r", 5.5).attr("fill", DC.violet);
    gg.append("text").attr("x", x(wes[0]) + 8).attr("y", y(wes[1]) + 4).attr("font-size", 10.5)
      .attr("fill", DC.violet).text("stopped at τ = " + tau);
    gg.append("circle").attr("cx", x(wl2[0])).attr("cy", y(wl2[1])).attr("r", 5.5).attr("fill", DC.good);
    gg.append("text").attr("x", x(wl2[0]) + 8).attr("y", y(wl2[1]) - 8).attr("font-size", 10.5)
      .attr("fill", DC.good).text("L2 at α = 1/(ητ) = " + DL.fmt(aEq, 4));
    const k3 = DL.kv(g, 500, 70, { keyW: 150, lead: 17 });
    k3("stopped at τ", "(" + wes.map(v => DL.fmt(v, 5)).join(", ") + ")", DC.violet, true);
    k3("L2, α = 1/(ητ)", "(" + wl2.map(v => DL.fmt(v, 5)).join(", ") + ")", DC.good, true);
    k3("|difference|, stiff", DL.fmtE(Math.abs(wes[0] - wl2[0]), 2), Math.abs(wes[0] - wl2[0]) > 0.05 ? DC.bad : DC.ink);
    k3("|difference|, soft", DL.fmtE(Math.abs(wes[1] - wl2[1]), 2), Math.abs(wes[1] - wl2[1]) > 0.05 ? DC.bad : DC.good);
    k3("ηλ, stiff  (needs ≪ 1)", DL.fmt(ETA * L1V, 4), ETA * L1V > 0.05 ? DC.bad : DC.good);
    k3("ηλ, soft   (needs ≪ 1)", DL.fmt(ETA * lam2, 5), ETA * lam2 > 0.05 ? DC.bad : DC.good);
    NT.say("#l2-readout",
      "Stopping at τ = <b>" + tau + "</b> lands at (" + wes.map(v => DL.fmt(v, 5)).join(", ") +
      "); the L2 solution at the predicted α = 1/(ητ) = <b>" + DL.fmt(aEq, 4) + "</b> is (" +
      wl2.map(v => DL.fmt(v, 5)).join(", ") + "). The <b>soft</b> direction agrees to <b>" +
      DL.fmtE(Math.abs(wes[1] - wl2[1]), 2) + "</b> and the <b>stiff</b> one to only <b>" +
      DL.fmtE(Math.abs(wes[0] - wl2[0]), 2) +
      "</b> — and that is exactly what the derivation says should happen, because the approximation requires ηλ ≪ 1 and the stiff direction has ηλ = " +
      DL.fmt(ETA * L1V, 3) + ". <b>The correspondence is real and it is not exact</b>; it is a statement about the low-curvature directions, which are also the ones a regulariser is mostly acting on. Raise τ and watch both points slide toward w*: training longer <i>is</i> regularising less.");
  }
  [ae, l2e, ve, te, oe].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 25 · #drop-svg — dropout ═════════════════ */
(function () {
  const svg = d3.select("#drop-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const pe = document.getElementById("dr-p"), ne = document.getElementById("dr-n"),
    ce = document.getElementById("dr-conv"), me = document.getElementById("dr-m"),
    mce = document.getElementById("dr-mc");

  function draw() {
    const p = +pe.value, n = +ne.value, conv = ce.value, M = +me.value, mc = mce.checked;
    document.getElementById("dr-pv").textContent = DL.fmt(p, 2);
    document.getElementById("dr-nv").textContent = n;
    document.getElementById("dr-mv").textContent = M;

    const r = DL.rng(17);
    const w = [], h = [];
    for (let i = 0; i < n; i++) { w.push(DL.randn(r) / Math.sqrt(n)); h.push(Math.abs(DL.randn(r))); }
    const full = DL.dot(w, h);
    const rr = DL.rng(29), outs = [];
    for (let k = 0; k < M; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) if (rr() < p) s += w[i] * h[i];
      outs.push(conv === "inv" ? s / p : (conv === "classic" ? s : s));
    }
    /* at INFERENCE: inverted → full; classical → p·full (weights scaled); none → full (the bug) */
    const infer = conv === "classic" ? p * full : full;
    let mcOuts = null;
    if (mc) {
      mcOuts = [];
      const r3 = DL.rng(53);
      for (let k = 0; k < Math.max(60, Math.floor(M / 20)); k++) {
        let acc = 0;
        for (let t = 0; t < 20; t++) {
          let s = 0;
          for (let i = 0; i < n; i++) if (r3() < p) s += w[i] * h[i];
          acc += (conv === "inv" ? s / p : s);
        }
        mcOuts.push(acc / 20);
      }
    }

    const F = DL.frame(svg, W, H, { l: 16, r: 16, t: 26, b: 8 });
    const g = F.g;

    /* the mask grid */
    const shown = Math.min(n, 256), cols = 16, cw = 13;
    const rows = Math.ceil(shown / cols);
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("one mask: " + shown + " of the " + n + " units, blank where dropped");
    const r4 = DL.rng(101);
    const gg = g.append("g").attr("transform", "translate(0,14)");
    const cs = d3.scaleSequential(d3.interpolateViridis).domain([0, d3.max(h.slice(0, shown))]);
    let kept = 0;
    for (let i = 0; i < shown; i++) {
      const on = r4() < p;
      if (on) kept++;
      gg.append("rect").attr("x", (i % cols) * cw).attr("y", Math.floor(i / cols) * cw)
        .attr("width", cw - 1.5).attr("height", cw - 1.5).attr("rx", 2)
        .attr("fill", on ? cs(h[i]) : "none").attr("stroke", on ? "none" : DC.line);
    }
    const gy = 14 + rows * cw + 20;
    const k = DL.kv(g, 0, gy, { keyW: 176, lead: 16 });
    k("kept in this mask", kept + " / " + shown + "  (" + DL.fmt(100 * kept / shown, 1) + "%)", DC.accent, true);
    k("distinct sub-networks 2ⁿ", n <= 40 ? DL.commas(Math.pow(2, n)) : DL.fmtE(Math.pow(2, n), 2), DC.a2, true);
    k("atoms in the observable universe", "≈ 1e80", DC.muted);

    /* the histogram */
    const QX = 250, QW = 480, QH = 250;
    g.append("text").attr("x", QX).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("layer output over " + M + " masks, against the no-dropout output");
    const g2 = g.append("g").attr("transform", "translate(" + QX + ",14)");
    const all = outs.concat([full, infer]).concat(mcOuts || []);
    const x = d3.scaleLinear().domain([d3.min(all) * 1.1 - 0.05, d3.max(all) * 1.1 + 0.05]).range([0, QW]);
    const bins = d3.bin().domain(x.domain()).thresholds(50)(outs);
    const ymax = d3.max(bins, b => b.length);
    const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([QH, 0]);
    DL.axisB(g2, x, QH, 6, "layer output"); DL.axisL(g2, y, 4, "count");
    g2.selectAll("rect.b").data(bins).join("rect").attr("class", "b")
      .attr("x", d => x(d.x0)).attr("y", d => y(d.length))
      .attr("width", d => Math.max(0.5, x(d.x1) - x(d.x0) - 0.6))
      .attr("height", d => QH - y(d.length)).attr("fill", DC.accent).attr("fill-opacity", 0.55);
    if (mcOuts) {
      const b2 = d3.bin().domain(x.domain()).thresholds(50)(mcOuts);
      const sc = ymax / Math.max(1, d3.max(b2, b => b.length));
      g2.selectAll("rect.c").data(b2).join("rect").attr("class", "c")
        .attr("x", d => x(d.x0)).attr("y", d => y(d.length * sc))
        .attr("width", d => Math.max(0.5, x(d.x1) - x(d.x0) - 0.6))
        .attr("height", d => QH - y(d.length * sc)).attr("fill", DC.violet).attr("fill-opacity", 0.55);
    }
    const mean = DL.mean(outs), sd = DL.std(outs);
    [[full, DC.good, "no dropout"], [infer, DC.bad, "what inference returns"], [mean, DC.a2, "mean over masks"]]
      .forEach(([v, col, lab], i) => {
        g2.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", QH)
          .attr("stroke", col).attr("stroke-width", 2).attr("stroke-dasharray", i === 2 ? "4 3" : null);
        g2.append("text").attr("x", x(v) + 4).attr("y", 14 + i * 13).attr("font-size", 10).attr("fill", col).text(lab);
      });

    const ratio = mean / full;
    NT.say("#drop-readout",
      "Over " + M + " masks at p = " + DL.fmt(p, 2) + ": the mean training-time output is <b>" +
      DL.fmt(mean, 5) + "</b> against a no-dropout output of <b>" + DL.fmt(full, 5) + "</b> — a ratio of <b>" +
      DL.fmt(ratio, 4) + "</b>. " +
      (conv === "inv" ? "That ratio is <b>1</b> up to the Monte-Carlo error of averaging only " + M + " masks — raise the mask count and watch it converge — because inverted dropout divides by p at training time, so inference can do nothing at all and still match."
        : (conv === "classic" ? "That ratio is <b>p = " + DL.fmt(p, 2) + "</b>, so inference must multiply the weights by p to match — which is the classical convention, and it puts the cost at deployment instead of during training."
          : "<b>That ratio is p and nothing corrects for it.</b> Training-time activations are p times inference-time ones at every layer, and the mismatch compounds: over L layers it is a factor of p^L. This is the bug.")) +
      " The standard deviation over masks is <b>" + DL.fmt(sd, 5) + "</b>, which is <b>" +
      DL.fmt(sd / Math.abs(full), 3) + "×</b> the size of the signal itself — a single dropped forward pass is mostly noise, and dropout works because the <i>gradient</i> is averaged over many masks and many examples. " +
      (mcOuts ? "The narrow second histogram averages 20 masks per point and has a standard deviation of <b>" +
        DL.fmt(DL.std(mcOuts), 5) + "</b>, about √20 = 4.5× tighter — which is why 10 to 20 sampled masks are enough to approximate the ensemble properly when you need a calibrated answer rather than a point prediction."
        : "Enable the Monte-Carlo option to see what averaging 20 masks does to that spread.") +
      " The mask is drawn from <b>2^" + n + "</b> distinct sub-networks, which for n = " + n + " is " +
      (n <= 40 ? DL.commas(Math.pow(2, n)) : DL.fmtE(Math.pow(2, n), 2)) + " — the ensemble is never enumerated.");
  }
  [pe, ne, ce, me, mce].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 26 · #clip-svg — clipping by value against by norm ═════════════════ */
(function () {
  const svg = d3.select("#clip-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const g1e = document.getElementById("cl-g1"), g2e = document.getElementById("cl-g2"),
    ce = document.getElementById("cl-c"), de = document.getElementById("cl-dist"),
    he = document.getElementById("cl-hist");

  const ang = (a, b) => {
    const na = Math.hypot(a[0], a[1]), nb = Math.hypot(b[0], b[1]);
    if (na < 1e-12 || nb < 1e-12) return 0;
    return Math.acos(DL.clamp((a[0] * b[0] + a[1] * b[1]) / (na * nb), -1, 1)) * 180 / Math.PI;
  };

  function draw() {
    const gv = [+g1e.value, +g2e.value], c = +ce.value, dist = de.value, showH = he.checked;
    document.getElementById("cl-g1v").textContent = DL.fmt(gv[0], 2);
    document.getElementById("cl-g2v").textContent = DL.fmt(gv[1], 2);
    document.getElementById("cl-cv").textContent = DL.fmt(c, 2);
    const byVal = DL.clipValue(gv, c);
    const byNorm = DL.clipNorm(gv, c);
    const aV = ang(gv, byVal), aN = ang(gv, byNorm.g);

    const F = DL.frame(svg, W, H, { l: 16, r: 16, t: 26, b: 8 });
    const g = F.g;
    const PW = 290, PH = 290;
    g.append("text").attr("x", 40).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the gradient, and what each rule does to it");
    const gg = g.append("g").attr("transform", "translate(40,14)");
    const R = 3.2;
    const x = d3.scaleLinear().domain([-R, R]).range([0, PW]);
    const y = d3.scaleLinear().domain([-R, R]).range([PH, 0]);
    DL.gridY(gg, y, PW, 5); DL.gridX(gg, x, PH, 5);
    DL.axisB(gg, x, PH, 5, "g₁"); DL.axisL(gg, y, 5, "g₂");
    /* the two constraint sets */
    gg.append("rect").attr("x", x(-c)).attr("y", y(c)).attr("width", x(c) - x(-c)).attr("height", y(-c) - y(c))
      .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.6);
    gg.append("circle").attr("cx", x(0)).attr("cy", y(0)).attr("r", Math.abs(x(c) - x(0)))
      .attr("fill", "none").attr("stroke", DC.good).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.6);
    DL.arrow(gg, x(0), y(0), x(DL.clamp(gv[0], -R, R)), y(DL.clamp(gv[1], -R, R)), { color: DC.muted, w: 2.4, head: 8 });
    DL.arrow(gg, x(0), y(0), x(byVal[0]), y(byVal[1]), { color: DC.a2, w: 2.4, head: 8 });
    DL.arrow(gg, x(0), y(0), x(byNorm.g[0]), y(byNorm.g[1]), { color: DC.good, w: 2.4, head: 8 });
    DL.legend(gg, [{ color: DC.muted, label: "g, ‖g‖ = " + DL.fmt(byNorm.norm, 4) },
      { color: DC.a2, label: "clip by value → " + aV.toFixed(2) + "° rotation" },
      { color: DC.good, label: "clip by norm → " + aN.toFixed(2) + "° rotation" }],
      8, 14, { gap: 14, font: 10.5 });

    /* the histogram of norms */
    const QX = 400, QW = 330, QH = 250;
    const r = DL.rng(61), norms = [];
    for (let i = 0; i < 4000; i++) {
      let v;
      if (dist === "lognormal") v = Math.exp(Math.log(0.6) + 0.8 * DL.randn(r));
      else if (dist === "spiky") v = (r() < 0.006) ? 8 + 40 * Math.abs(DL.randn(r)) : 0.4 + 0.12 * Math.abs(DL.randn(r));
      else v = 0.9 + 0.1 * DL.randn(r);
      norms.push(Math.abs(v));
    }
    const fires = norms.filter(v => v > c).length / norms.length;
    if (showH) {
      g.append("text").attr("x", QX).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("distribution of ‖g‖ over 4 000 steps, and how often the clip fires");
      const g2 = g.append("g").attr("transform", "translate(" + QX + ",14)");
      const xs = d3.scaleLog().domain([Math.max(1e-3, d3.min(norms) * 0.8), d3.max(norms) * 1.2]).range([0, QW]);
      const bins = d3.bin().domain(xs.domain())
        .thresholds(d3.range(40).map(i => Math.pow(10, Math.log10(xs.domain()[0]) + i * (Math.log10(xs.domain()[1]) - Math.log10(xs.domain()[0])) / 39)))(norms);
      const ys = d3.scaleLinear().domain([0, d3.max(bins, b => b.length) * 1.1]).range([QH, 0]);
      DL.axisB(g2, xs, QH, 5, "‖g‖", d3.format(".2g")); DL.axisL(g2, ys, 4, "count");
      g2.selectAll("rect").data(bins).join("rect")
        .attr("x", d => xs(Math.max(d.x0, xs.domain()[0]))).attr("y", d => ys(d.length))
        .attr("width", d => Math.max(0.5, xs(d.x1) - xs(Math.max(d.x0, xs.domain()[0])) - 0.5))
        .attr("height", d => QH - ys(d.length))
        .attr("fill", d => d.x0 > c ? DC.bad : DC.accent).attr("fill-opacity", 0.6);
      g2.append("line").attr("x1", xs(DL.clamp(c, xs.domain()[0], xs.domain()[1])))
        .attr("x2", xs(DL.clamp(c, xs.domain()[0], xs.domain()[1]))).attr("y1", 0).attr("y2", QH)
        .attr("stroke", DC.bad).attr("stroke-width", 2);
      g2.append("text").attr("x", xs(DL.clamp(c, xs.domain()[0], xs.domain()[1])) + 5).attr("y", 14)
        .attr("font-size", 10.5).attr("fill", DC.bad).text("c = " + DL.fmt(c, 2) + " · fires on " + (100 * fires).toFixed(1) + "% of steps");
    }

    NT.say("#clip-readout",
      "‖g‖ = <b>" + DL.fmt(byNorm.norm, 5) + "</b>, threshold <b>" + DL.fmt(c, 2) + "</b>. " +
      "Clipping by <b>value</b> gives [" + byVal.map(v => DL.fmt(v, 5)).join(", ") +
      "] and rotates the update by <b>" + aV.toFixed(3) + "°</b>. Clipping by <b>norm</b> gives [" +
      byNorm.g.map(v => DL.fmt(v, 6)).join(", ") + "] and rotates it by <b>" + aN.toFixed(3) +
      "°</b> — exactly zero, always, by construction, because it is a scalar multiple of the original. " +
      "The ratio between the two coordinates was " + DL.fmt(Math.abs(gv[1] / (gv[0] || 1e-9)), 3) +
      " : 1 and after the norm clip it is still " + DL.fmt(Math.abs(byNorm.g[1] / (byNorm.g[0] || 1e-9)), 3) +
      " : 1; after the value clip it is " + DL.fmt(Math.abs(byVal[1] / (byVal[0] || 1e-9)), 3) +
      " : 1. On a cliff the whole problem is that ONE direction has become enormous, and clipping by value redistributes the step into directions the gradient never asked for. " +
      (showH ? "The clip fires on <b>" + (100 * fires).toFixed(1) + "%</b> of steps at this threshold. " +
        (fires > 0.4 ? "<b>That is far too often</b> — a clip that fires on most steps is not clipping, it is normalising away all the magnitude information and acting as a strange learning-rate schedule."
          : "That is a reasonable rate: it fires on the tail and leaves the body of the distribution alone.") : ""));
  }
  [g1e, g2e, ce, de, he].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 27 · #hp-svg — random against grid search ═════════════════ */
(function () {
  const svg = d3.select("#hp-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const ne = document.getElementById("hp-n"), ie = document.getElementById("hp-imp"),
    tre = document.getElementById("hp-tr"), de = document.getElementById("hp-d"),
    pe = document.getElementById("hp-proj");

  /* an objective that varies strongly along dimension 0 and weakly along the
     rest. Lower is better; the optimum is at a fixed interior point.        */
  /* The optimum is drawn UNIFORMLY over most of the unit cube. Drawing it from
     a distribution centred on 0.5 would hand the grid a systematic advantage,
     because a grid with an odd number of points has one sitting exactly there —
     an artefact, not a property of grid search, and one that reverses the
     result at small d. Verified in the build: with a centred optimum the grid
     appears to win at every d; with this one the honest pattern appears.     */
  function makeObj(d, imp, seed) {
    const r = DL.rng(seed), c = [], wgt = [];
    for (let k = 0; k < d; k++) { c.push(0.05 + 0.90 * r()); wgt.push(k === 0 ? 1 : imp); }
    return v => {
      let s = 0;
      for (let k = 0; k < d; k++) s += wgt[k] * Math.pow(v[k] - c[k], 2);
      return s;
    };
  }

  function draw() {
    const n = +ne.value, imp = +ie.value, TR = +tre.value, d = +de.value, proj = pe.checked;
    document.getElementById("hp-nv").textContent = n;
    document.getElementById("hp-impv").textContent = DL.fmt(imp, 2);
    document.getElementById("hp-trv").textContent = TR;
    const f = makeObj(d, imp, 3);
    const k = Math.max(2, Math.round(Math.pow(n, 1 / d)));
    const gridN = Math.pow(k, d);

    /* grid points */
    const grid = [];
    (function rec(pref) {
      if (pref.length === d) { grid.push(pref.slice()); return; }
      for (let i = 0; i < k; i++) { pref.push((i + 0.5) / k); rec(pref); pref.pop(); }
    })([]);
    /* random points, same budget as the grid so the comparison is fair */
    const rr = DL.rng(97), rand = [];
    for (let i = 0; i < gridN; i++) { const v = []; for (let j = 0; j < d; j++) v.push(rr()); rand.push(v); }
    const bestG = grid.reduce((a, v) => f(v) < f(a) ? v : a, grid[0]);
    const bestR = rand.reduce((a, v) => f(v) < f(a) ? v : a, rand[0]);

    const F = DL.frame(svg, W, H, { l: 46, r: 16, t: 26, b: 8 });
    const g = F.g;
    const PW = 280, PH = 268;
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the first two dimensions, " + gridN + " runs each");
    const gg = g.append("g").attr("transform", "translate(0,16)");
    const x = d3.scaleLinear().domain([0, 1]).range([0, PW]);
    const y = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
    const RES = 56, vals = [];
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
      const v = new Array(d).fill(0.5);
      v[0] = (i + 0.5) / RES; if (d > 1) v[1] = (j + 0.5) / RES;
      vals.push(f(v));
    }
    const ext = d3.extent(vals);
    const cs = d3.scaleSequential(d3.interpolateViridis).domain([ext[1], ext[0]]);
    DL.cells(gg, 0, 0, PW / RES, RES, RES, (i, j) => cs(vals[(RES - 1 - j) * RES + i]));
    DL.axisB(gg, x, PH, 5, "dimension 1 — the one that matters");
    DL.axisL(gg, y, 5, "dimension 2");
    grid.forEach(v => gg.append("rect").attr("x", x(v[0]) - 2.4).attr("y", y(d > 1 ? v[1] : 0.5) - 2.4)
      .attr("width", 4.8).attr("height", 4.8).attr("fill", DC.a2).attr("fill-opacity", 0.9));
    rand.forEach(v => gg.append("circle").attr("cx", x(v[0])).attr("cy", y(d > 1 ? v[1] : 0.5))
      .attr("r", 2.4).attr("fill", DC.accent).attr("fill-opacity", 0.9));
    gg.append("circle").attr("cx", x(bestG[0])).attr("cy", y(d > 1 ? bestG[1] : 0.5)).attr("r", 6.5)
      .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2);
    gg.append("circle").attr("cx", x(bestR[0])).attr("cy", y(d > 1 ? bestR[1] : 0.5)).attr("r", 6.5)
      .attr("fill", "none").attr("stroke", DC.accent).attr("stroke-width", 2);
    if (proj) {
      const uniqG = new Set(grid.map(v => v[0].toFixed(6))).size;
      grid.forEach(v => gg.append("line").attr("x1", x(v[0])).attr("x2", x(v[0]))
        .attr("y1", PH + 4).attr("y2", PH + 12).attr("stroke", DC.a2).attr("stroke-opacity", 0.9));
      rand.forEach(v => gg.append("line").attr("x1", x(v[0])).attr("x2", x(v[0]))
        .attr("y1", PH + 14).attr("y2", PH + 22).attr("stroke", DC.accent).attr("stroke-opacity", 0.6));
      gg.append("text").attr("x", PW + 4).attr("y", PH + 11).attr("font-size", 9).attr("fill", DC.a2)
        .text(uniqG + " values");
      gg.append("text").attr("x", PW + 4).attr("y", PH + 21).attr("font-size", 9).attr("fill", DC.accent)
        .text(gridN + " values");
    }

    /* expected best against budget, averaged */
    const QX = 400, QW = 330, QH = 250;
    g.append("text").attr("x", QX).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("expected best found, averaged over " + TR + " independent trials");
    const g2 = g.append("g").attr("transform", "translate(" + QX + ",16)");
    const budgets = [];
    for (let kk = 2; kk <= (d >= 4 ? 5 : 6); kk++) budgets.push({ k: kk, n: Math.pow(kk, d) });
    const seriesG = [], seriesR = [];
    budgets.forEach(b => {
      /* grid is deterministic apart from the objective's random centre */
      let accG = 0, accR = 0;
      for (let t = 0; t < TR; t++) {
        const ft = makeObj(d, imp, 1000 + t);
        const gp = [];
        (function rec(pref) {
          if (pref.length === d) { gp.push(pref.slice()); return; }
          for (let i = 0; i < b.k; i++) { pref.push((i + 0.5) / b.k); rec(pref); pref.pop(); }
        })([]);
        accG += d3.min(gp, v => ft(v));
        const r3 = DL.rng(7000 + t * 31 + b.k);
        let bestv = Infinity;
        for (let i = 0; i < b.n; i++) { const v = []; for (let j = 0; j < d; j++) v.push(r3()); bestv = Math.min(bestv, ft(v)); }
        accR += bestv;
      }
      seriesG.push([b.n, accG / TR]); seriesR.push([b.n, accR / TR]);
    });
    const xs = d3.scaleLog().domain([budgets[0].n, budgets[budgets.length - 1].n]).range([0, QW]);
    const allv = seriesG.concat(seriesR).map(p => p[1]);
    const ys = d3.scaleLinear().domain([0, d3.max(allv) * 1.1]).range([QH, 0]);
    DL.gridY(g2, ys, QW, 4);
    DL.axisB(g2, xs, QH, 5, "runs in the budget", d3.format("d"));
    DL.axisL(g2, ys, 4, "best objective found");
    DL.curve(g2, seriesG.map(p => [xs(p[0]), ys(p[1])]), { stroke: DC.a2, w: 2.4 });
    DL.curve(g2, seriesR.map(p => [xs(p[0]), ys(p[1])]), { stroke: DC.accent, w: 2.4 });
    seriesG.forEach(p => g2.append("circle").attr("cx", xs(p[0])).attr("cy", ys(p[1])).attr("r", 3).attr("fill", DC.a2));
    seriesR.forEach(p => g2.append("circle").attr("cx", xs(p[0])).attr("cy", ys(p[1])).attr("r", 3).attr("fill", DC.accent));
    DL.legend(g2, [{ color: DC.a2, label: "grid" }, { color: DC.accent, label: "random" }],
      8, 14, { gap: 14, font: 10.5 });

    const uniq = new Set(grid.map(v => v[0].toFixed(6))).size;
    const i5 = seriesR.findIndex(p => true);
    NT.say("#hp-readout",
      "With d = <b>" + d + "</b> and a budget of <b>" + gridN + "</b> runs, the grid uses only <b>" + uniq +
      "</b> distinct values of the dimension that matters, while random search uses <b>" + gridN +
      "</b>. On this one draw of the problem, best objective found: grid <b>" + DL.fmt(f(bestG), 5) + "</b>, random <b>" + DL.fmt(f(bestR), 5) +
      "</b> — a single problem is close to a coin flip and proves nothing either way. The panel on the right is the statement: averaged over " + TR + " independent problems at the largest budget, grid <b>" +
      DL.fmt(seriesG[seriesG.length - 1][1], 5) + "</b>, random <b>" +
      DL.fmt(seriesR[seriesR.length - 1][1], 5) + "</b>. " +
      (function () {
        const gG = seriesG[seriesG.length - 1][1], gR = seriesR[seriesR.length - 1][1];
        return gR < gG
          ? "<b>Random wins here by a factor of " + DL.fmt(gG / gR, 2) + "</b>, and the margin grows with both the budget and the dimension."
          : "<b>The grid wins here</b>, by a factor of " + DL.fmt(gR / gG, 2) +
            " — and that is not a mistake in the figure. At d = 2 a grid covers the important dimension perfectly adequately and its regularity is a small advantage. Move the dimension control to 3 and the two draw level; move it to 4 and random pulls ahead and keeps pulling ahead as the budget grows. <b>The argument for random search is an argument about dimension</b>, and quoting it at d = 2 is quoting it outside its range.";
      })() +
      " And the count worth remembering: P(at least one of n uniform draws lands in the best 5%) = 1 − 0.95ⁿ, which is <b>" +
      DL.fmt(1 - Math.pow(0.95, gridN), 4) + "</b> at n = " + gridN + " and <b>0.9539</b> at n = 60 — <i>independent of the dimension</i>.");
  }
  [ne, ie, tre, de, pe].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 28 · #curve-svg — reading a training curve ═════════════════ */
(function () {
  const svg = d3.select("#curve-svg");
  if (svg.empty()) return;
  const W = 760, H = 390;
  const ce = document.getElementById("cv-case"), Te = document.getElementById("cv-T"),
    de = document.getElementById("cv-diag"), le = document.getElementById("cv-log");

  function spiral(N, noise, seed) {
    const r = DL.rng(seed), X = [], Y = [];
    for (let i = 0; i < N; i++) {
      const k = i % 2, t = 1.2 + 3.6 * (i / N), th = t + k * Math.PI + 0.25 * DL.randn(r);
      X.push([t * Math.cos(th) / 5, t * Math.sin(th) / 5]);
      Y.push([(r() < noise) ? 1 - k : k]);
    }
    return { X: X, Y: Y };
  }
  const CASES = {
    ok: { lr: 0.5, w: 24, ntr: 200, noise: 0.05, scale: 1, naive: false, shuffle: true, cliff: false, clip: false,
      verdict: "Healthy. Both curves fall, the gap is small and stable, and the gradient norm is neither zero nor exploding.", fix: "Nothing to do." },
    over: { lr: 0.5, w: 96, ntr: 24, noise: 0.25, scale: 1, naive: false, shuffle: true, cliff: false, clip: false,
      verdict: "Overfitting: the training loss keeps falling while the validation loss turns and rises.", fix: "Early stopping (§34); then regularisation (§31–§35) or more data." },
    under: { lr: 0.004, w: 3, ntr: 200, noise: 0.05, scale: 1, naive: false, shuffle: true, cliff: false, clip: false,
      verdict: "Underfitting: both curves plateau high and close together. Easy to mistake for overfitting because 'the loss stopped going down'.", fix: "More capacity, a higher rate, less regularisation — and §39's ladder first." },
    hi: { lr: 12, w: 24, ntr: 200, noise: 0.05, scale: 1, naive: false, shuffle: true, cliff: false, clip: false,
      verdict: "The learning rate is past 2/λ_max. The loss rises or oscillates violently and units are dying.", fix: "Lower η, add warmup (§24), watch the dead fraction (§11)." },
    lo: { lr: 0.0008, w: 24, ntr: 200, noise: 0.05, scale: 1, naive: false, shuffle: true, cliff: false, clip: false,
      verdict: "The rate is too low. The curve is falling — just very slowly — and the ‖Δθ‖/‖θ‖ diagnostic is orders of magnitude below 10⁻³.", fix: "Raise η. The range test (§25) finds the right one in one pass." },
    dead: { lr: 0.5, w: 24, ntr: 200, noise: 0.05, scale: 0.03, naive: false, shuffle: true, cliff: false, clip: false,
      verdict: "The forward pass has died: the initialisation is far too small, the deep activations have collapsed, and the loss is pinned at log 2.", fix: "Fix the initialisation (§12–§14). The gradient norm at the first layer is what distinguishes this from 'converged'." },
    nan: { lr: 0.5, w: 24, ntr: 200, noise: 0.05, scale: 1, naive: true, shuffle: true, cliff: false, clip: false,
      verdict: "A naive loss overflowed and the parameters became NaN. Every step after that point is meaningless and nothing raised an error.", fix: "Use the fused logits-to-loss function (§03). Detect non-finite losses and stop." },
    noshuf: { lr: 0.5, w: 24, ntr: 200, noise: 0.05, scale: 1, naive: false, shuffle: false, cliff: false, clip: false,
      verdict: "A saw-tooth whose period is exactly one epoch: the same ordered batches recur and so do their correlated gradients.", fix: "Shuffle every epoch, not once (§15)." },
    cliff: { lr: 0.5, w: 24, ntr: 200, noise: 0.05, scale: 1, naive: false, shuffle: true, cliff: true, clip: false,
      verdict: "A cliff: one batch produced an enormous gradient and the step launched the parameters somewhere arbitrary. The clipped run beside it takes the same step bounded.", fix: "Clip by norm (§36) — the direction was fine, only the length was meaningless." }
  };

  function run(cfg, T, clipOn) {
    const TR = spiral(cfg.ntr, cfg.noise, 11), VA = spiral(160, cfg.noise, 88);
    const net = DL.mlpInit([2, cfg.w, cfg.w, 1], { act: "relu", out: "sigmoid", seed: 13 });
    if (cfg.scale !== 1) for (let l = 0; l < net.L; l++) {
      const r = DL.rng(500 + l), s = cfg.scale * Math.sqrt(2 / net.sizes[l]);
      for (let i = 0; i < net.sizes[l]; i++) for (let j = 0; j < net.sizes[l + 1]; j++) net.W[l][i][j] = s * DL.randn(r);
    }
    const B = 16, idx = d3.range(TR.X.length);
    const rsh = DL.rng(3);
    const tr = [], va = [], gn = [], g1 = [], ratio = [], dead = [];
    let order = idx.slice();
    for (let t = 0; t < T; t++) {
      if (cfg.shuffle && t % Math.ceil(TR.X.length / B) === 0) order = DL.shuffle(idx.slice(), rsh);
      const st = (t * B) % TR.X.length;
      const pick = [];
      for (let i = 0; i < B; i++) pick.push(order[(st + i) % order.length]);
      const Xb = pick.map(i => TR.X[i]), Yb = pick.map(i => TR.Y[i]);
      const gg = DL.backward(net, Xb, Yb);
      let gv = [];
      for (let l = 0; l < net.L; l++) { gg.dW[l].forEach(r2 => r2.forEach(v => gv.push(v))); gg.db[l].forEach(v => gv.push(v)); }
      if (cfg.cliff && t === Math.floor(T * 0.45)) gv = gv.map(v => v * 3000);
      let nrm = Math.sqrt(gv.reduce((s, v) => s + v * v, 0));
      if (clipOn) { const c = DL.clipNorm(gv, 1); gv = c.g; nrm = Math.sqrt(gv.reduce((s, v) => s + v * v, 0)); }
      let pn = 0;
      for (let l = 0; l < net.L; l++) { net.W[l].forEach(r2 => r2.forEach(v => pn += v * v)); net.b[l].forEach(v => pn += v * v); }
      pn = Math.sqrt(pn);
      /* apply */
      let k = 0;
      const lr = cfg.lr;
      for (let l = 0; l < net.L; l++) {
        for (let i = 0; i < net.sizes[l]; i++) for (let j = 0; j < net.sizes[l + 1]; j++) net.W[l][i][j] -= lr * gv[k++];
        for (let j = 0; j < net.sizes[l + 1]; j++) net.b[l][j] -= lr * gv[k++];
      }
      /* the "naive loss" case: simulate the overflow that destroys the run */
      let L = gg.loss;
      if (cfg.naive) {
        const f = DL.forward(net, Xb);
        const worst = d3.max(f.logits.map((r2, i) => (1 - 2 * Yb[i][0]) * r2[0]));
        if (worst > 88 || !isFinite(L)) L = NaN;
      }
      tr.push(L);
      va.push(DL.loss(net, VA.X, VA.Y));
      gn.push(nrm);
      g1.push(DL.frob(gg.dW[0]));
      ratio.push(lr * nrm / Math.max(pn, 1e-12));
      const f2 = DL.forward(net, TR.X);
      let nd = 0;
      for (let l = 0; l < 2; l++) {
        const Z = f2.z[l];
        for (let j = 0; j < cfg.w; j++) {
          let alive = false;
          for (let i = 0; i < Z.length && !alive; i++) if (Z[i][j] > 0) alive = true;
          if (!alive) nd++;
        }
      }
      dead.push(100 * nd / (2 * cfg.w));
      if (!isFinite(L)) { while (tr.length < T) { tr.push(NaN); va.push(NaN); gn.push(NaN); g1.push(NaN); ratio.push(NaN); dead.push(dead[dead.length - 1]); } break; }
    }
    return { tr: tr, va: va, gn: gn, g1: g1, ratio: ratio, dead: dead };
  }

  function draw() {
    const key = ce.value, T = +Te.value, diag = de.value, logy = le.checked;
    document.getElementById("cv-Tv").textContent = T;
    const cfg = CASES[key];
    const A = run(cfg, T, false);
    const B2 = cfg.cliff ? run(cfg, T, true) : null;

    const F = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 8 });
    const g = F.g;
    const PW = 690, PH1 = 195, PH2 = 100;
    const x = d3.scaleLinear().domain([0, T]).range([0, PW]);
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("training and validation loss");
    const fin = A.tr.concat(A.va).filter(v => isFinite(v) && v > 0);
    const y1 = (logy ? d3.scaleLog().domain([Math.max(1e-4, d3.min(fin) * 0.7), Math.max(1.4, d3.max(fin) * 1.15)])
      : d3.scaleLinear().domain([0, Math.max(1.4, d3.max(fin) * 1.15)])).range([PH1, 0]).clamp(true);
    DL.gridY(g, y1, PW, 5);
    DL.axisB(g, x, PH1, 6, "");
    DL.axisL(g, y1, 5, "loss", logy ? d3.format(".0e") : null);
    g.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y1(Math.log(2))).attr("y2", y1(Math.log(2)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    g.append("text").attr("x", PW - 2).attr("y", y1(Math.log(2)) - 4).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", DC.muted).text("log 2 — the class prior");
    const draw1 = (arr, col, w2, dash) => DL.curve(g,
      arr.map((v, i) => isFinite(v) ? [x(i), y1(Math.max(v, y1.domain()[0]))] : null).filter(Boolean),
      { stroke: col, w: w2, dash: dash });
    draw1(A.tr, DC.accent, 2.2);
    draw1(A.va, DC.bad, 2.0, "4 3");
    if (B2) draw1(B2.tr, DC.good, 1.8);
    const nanAt = A.tr.findIndex(v => !isFinite(v));
    if (nanAt >= 0) {
      g.append("line").attr("x1", x(nanAt)).attr("x2", x(nanAt)).attr("y1", 0).attr("y2", PH1)
        .attr("stroke", DC.bad).attr("stroke-width", 2);
      g.append("text").attr("x", x(nanAt) + 5).attr("y", 14).attr("font-size", 10.5).attr("fill", DC.bad)
        .text("NaN at step " + nanAt + " — everything after is meaningless");
    }
    DL.legend(g, [{ color: DC.accent, label: "training" }, { color: DC.bad, label: "validation", dash: "4 3" }]
      .concat(B2 ? [{ color: DC.good, label: "the same run, clipped at norm 1" }] : []),
      10, 14, { gap: 14, font: 10.5 });

    const g2 = g.append("g").attr("transform", "translate(0," + (PH1 + 58) + ")");
    const dl = { grad: ["gradient norm — global, and at the first layer", [A.gn, A.g1], [DC.violet, DC.a2], true],
      ratio: ["‖Δθ‖ / ‖θ‖ — around 1e−3 for a large model at a small rate; a toy net trained fast sits higher", [A.ratio], [DC.teal], true],
      dead: ["fraction of hidden units dead on the whole training set", [A.dead], [DC.rose], false] }[diag];
    g.append("text").attr("x", 0).attr("y", PH1 + 46).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(dl[0]);
    const dv = dl[1].reduce((s, a) => s.concat(a), []).filter(v => isFinite(v) && (dl[3] ? v > 0 : true));
    const y2 = (dl[3] ? d3.scaleLog().domain([Math.max(1e-14, d3.min(dv) * 0.5), Math.max(1e-2, d3.max(dv) * 2)])
      : d3.scaleLinear().domain([0, 100])).range([PH2, 0]).clamp(true);
    DL.gridY(g2, y2, PW, 3);
    DL.axisB(g2, x, PH2, 6, "training step");
    DL.axisL(g2, y2, 3, "", dl[3] ? d3.format(".0e") : (d => d + "%"));
    if (diag === "ratio") {
      g2.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y2(1e-3)).attr("y2", y2(1e-3))
        .attr("stroke", DC.good).attr("stroke-dasharray", "4 3");
      g2.append("text").attr("x", 3).attr("y", y2(1e-3) - 4).attr("font-size", 9.5).attr("fill", DC.good).text("1e−3");
    }
    dl[1].forEach((arr, i) => DL.curve(g2,
      arr.map((v, j) => isFinite(v) ? [x(j), y2(dl[3] ? Math.max(v, y2.domain()[0]) : v)] : null).filter(Boolean),
      { stroke: dl[2][i], w: i === 0 ? 2.2 : 1.7, dash: i ? "4 3" : null }));

    const lastTr = A.tr.filter(isFinite).slice(-1)[0], lastVa = A.va.filter(isFinite).slice(-1)[0];
    const minVa = d3.min(A.va.filter(isFinite));
    NT.say("#curve-readout",
      "<b>" + cfg.verdict + "</b> Final training loss <b>" +
      (isFinite(lastTr) ? DL.fmt(lastTr, 5) : "NaN") + "</b>, final validation <b>" +
      (isFinite(lastVa) ? DL.fmt(lastVa, 5) : "NaN") + "</b>, best validation <b>" +
      (isFinite(minVa) ? DL.fmt(minVa, 5) : "—") + "</b> at step <b>" + A.va.indexOf(minVa) +
      "</b>. Gradient norm at the first layer, final: <b>" + DL.fmtE(A.g1.filter(isFinite).slice(-1)[0], 2) +
      "</b>; ‖Δθ‖/‖θ‖: <b>" + DL.fmtE(A.ratio.filter(isFinite).slice(-1)[0], 2) +
      "</b>; dead units: <b>" + DL.fmt(A.dead.slice(-1)[0], 1) + "%</b>. <b>Fix:</b> " + cfg.fix +
      " Note that the loss curve alone cannot separate every case here — the first-layer gradient norm separates the dead run from a converged one, and ‖Δθ‖/‖θ‖ separates a rate that is too low from a model that has nothing left to learn.");
  }
  [ce, Te, de, le].forEach(el => el.addEventListener("input", draw));
  draw();
})();
