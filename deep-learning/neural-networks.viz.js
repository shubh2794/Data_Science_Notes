/* neural-networks.viz.js — the visualizations on deep-learning/neural-networks.html.
   Loaded after ../data.js → ../notes.js → dl-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page,
   so a section can be reordered or removed without breaking the rest.

   Every numeric primitive — activations and their derivatives, the stable
   softmax / log-sigmoid / cross-entropy family, the dense forward and backward
   pass, the parameter and region counters — comes from dl-viz.js (DL.*).
   NOTHING here forks those. Page-specific numerics live in NN below.

     1  #chain-svg      the chain: depth, width, the shape chain, the budget
     2  #neuron-svg     one unit: its hyperplane, its field, its cross-section
     3  #repr-svg       one linear classifier, three coordinate systems
     4  #collapse-svg   two affine maps collapsing into one product matrix
     5  #xorlin-svg     no line gets more than three of the four XOR points
     6  #xorsolve-svg   the exact ReLU solution, stage by stage, every number
     7  #fold-svg       folding composes: units add regions, layers multiply
     8  #shapes-svg     the forward pass as tensors, both conventions
     9  #budget-svg     where the parameters and the multiplies actually are
    10  #actatlas-svg   the activation atlas: f, f′, and the properties table
    11  #gelugap-svg    exact GELU against its two surrogates, and the gap
    12  #stable-svg     the stable and naive log-sigmoid, side by side
    13  #lossgrad-svg   output unit × loss: the gradient when confidently wrong
    14  #softmax-svg    softmax: shift invariance, temperature, saturation
    15  #mdn-svg        a mixture-density output against a single Gaussian
    16  #ua-svg         universal approximation, and what it does not promise
    17  #regions-svg    linear-region counting, deep against shallow
    18  #sawtooth-svg   the sawtooth: 2^L teeth from 2L units
    19  #sizing-svg     layer shapes under one parameter budget
    20  #bias-svg       an MLP is permutation-equivariant; a CNN is not
    21  #mem-svg        memorisation: perfect fit on randomised labels
    22  #ffn-svg        where the parameters of a transformer layer live

   Every number these print is recomputed from the data they draw.            */

/* ══════════ page-local helpers (deliberately NOT in dl-viz.js) ══════════ */
const NN = (function () {

  /* A logistic classifier fitted by plain full-batch gradient descent on the
     cross-entropy. Used only by figure 3, where the POINT is that the
     classifier never changes and only its input does — so it must be the
     identical routine in every panel. Returns {w, b, acc(X, y)}.            */
  function logistic(X, y, steps, lr, l2) {
    const d = X[0].length, N = X.length;
    let w = new Array(d).fill(0), b = 0;
    const S = steps || 400, LR = lr === undefined ? 0.5 : lr, L2 = l2 === undefined ? 1e-3 : l2;
    for (let t = 0; t < S; t++) {
      const gw = new Array(d).fill(0);
      let gb = 0;
      for (let i = 0; i < N; i++) {
        const z = DL.dot(w, X[i]) + b, e = DL.sigmoid(z) - y[i];
        for (let j = 0; j < d; j++) gw[j] += e * X[i][j];
        gb += e;
      }
      for (let j = 0; j < d; j++) w[j] -= LR * (gw[j] / N + L2 * w[j]);
      b -= LR * gb / N;
    }
    return { w: w, b: b };
  }
  function accuracy(model, X, y) {
    let n = 0;
    for (let i = 0; i < X.length; i++) {
      const p = DL.dot(model.w, X[i]) + model.b >= 0 ? 1 : 0;
      if (p === y[i]) n++;
    }
    return n / X.length;
  }
  /* column-wise standardisation, so the fitted classifier is comparable
     across representations whose scales differ by orders of magnitude       */
  function standardise(X) {
    const d = X[0].length, N = X.length, mu = new Array(d).fill(0), sd = new Array(d).fill(0);
    for (const r of X) for (let j = 0; j < d; j++) mu[j] += r[j] / N;
    for (const r of X) for (let j = 0; j < d; j++) sd[j] += (r[j] - mu[j]) * (r[j] - mu[j]) / N;
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    return X.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
  }

  /* a p-piece sawtooth on [0, 1], built from p rectified units. Exact, and
     used by figures 7 and 18 — the SAME function, so the two agree.        */
  function sawtooth(x, p) {
    const t = DL.clamp(x, 0, 1) * p;
    const k = Math.min(Math.floor(t), p - 1), f = t - k;
    return (k % 2 === 0) ? f : 1 - f;
  }
  /* distinct sign patterns of a list of unit functions on a grid — the way
     every region count on this page is MEASURED rather than asserted        */
  function countRegions(signAt, res) {
    const seen = new Set();
    for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
      seen.add(signAt((j + 0.5) / res, (i + 0.5) / res));
    }
    return seen.size;
  }
  const hue = (i, n) => d3.interpolateTurbo(((i * 0.6180339887) % 1));

  return { logistic, accuracy, standardise, sawtooth, countRegions, hue };
})();

/* ═════════════════ 1 · #chain-svg — the chain and its budget ═════════════════ */
(function () {
  const svg = d3.select("#chain-svg");
  if (svg.empty()) return;
  const W = 760, H = 360;
  const din = document.getElementById("ch-din"), Ls = document.getElementById("ch-L"),
    wd = document.getElementById("ch-w"), dout = document.getElementById("ch-dout"),
    nb = document.getElementById("ch-n");

  function draw() {
    const d0 = +din.value, L = +Ls.value, w = +wd.value, dO = +dout.value, N = +nb.value;
    document.getElementById("ch-dinv").textContent = d0;
    document.getElementById("ch-Lv").textContent = L;
    document.getElementById("ch-wv").textContent = w;
    document.getElementById("ch-doutv").textContent = dO;
    document.getElementById("ch-nv").textContent = N;

    const sizes = [d0].concat(new Array(L).fill(w)).concat([dO]);
    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 8 });
    const g = F.g;

    /* ---- the DAG ---- */
    const labels = sizes.map((n, i) =>
      i === 0 ? "input " + n : (i === sizes.length - 1 ? "output " + n : "h" + i + " · " + n));
    DL.netDiagram(g, sizes, {
      x: 40, y: 18, w: Math.min(560, 110 * (sizes.length - 1)), h: 110,
      r: 5.5, maxShow: 8, labels: labels, edgeOp: 0.3, edgeWidth: 0.7
    });
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the graph — one circle per unit");

    /* ---- the compact chain with shapes ---- */
    const y0 = 168;
    g.append("text").attr("x", 0).attr("y", y0 - 12).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the same thing, one box per tensor");
    const bw = 78, gap = Math.max(26, Math.min(46, (700 - sizes.length * bw) / Math.max(1, sizes.length - 1)));
    let x = 4;
    sizes.forEach((n, i) => {
      const isIn = i === 0, isOut = i === sizes.length - 1;
      g.append("rect").attr("x", x).attr("y", y0).attr("width", bw).attr("height", 34).attr("rx", 6)
        .attr("fill", DC.panel2).attr("stroke", isIn ? DC.accent : (isOut ? DC.good : DC.a2));
      g.append("text").attr("x", x + bw / 2).attr("y", y0 + 15).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
        .text(N + " × " + n);
      g.append("text").attr("x", x + bw / 2).attr("y", y0 + 28).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", DC.muted)
        .text(isIn ? "X" : (isOut ? "logits" : "A" + i));
      if (i + 1 < sizes.length) {
        DL.arrow(g, x + bw + 3, y0 + 17, x + bw + gap - 3, y0 + 17, { color: DC.muted, w: 1.2, head: 5 });
        g.append("text").attr("x", x + bw + gap / 2).attr("y", y0 - 4).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.accent)
          .text("W" + (i + 1) + " " + n + "×" + sizes[i + 1]);
      }
      x += bw + gap;
    });

    /* ---- the budget table ---- */
    const ty = 232;
    const rows = [];
    let P = 0, M = 0;
    for (let l = 0; l + 1 < sizes.length; l++) {
      const p = sizes[l] * sizes[l + 1] + sizes[l + 1], m = sizes[l] * sizes[l + 1];
      P += p; M += m;
      rows.push(["layer " + (l + 1), sizes[l] + " × " + sizes[l + 1], String(sizes[l + 1]),
        DL.commas(p), DL.commas(m)]);
    }
    rows.push(["total", "", "", DL.commas(P), DL.commas(M)]);
    const cols = [0, 100, 210, 300, 420], head = ["", "W shape", "b", "parameters", "MACs / example"];
    head.forEach((h, j) => g.append("text").attr("x", cols[j]).attr("y", ty).attr("font-size", 10)
      .attr("fill", DC.muted).text(h));
    rows.forEach((r, i) => {
      const yy = ty + 16 + i * 13.5, last = i === rows.length - 1;
      if (last) g.append("line").attr("x1", 0).attr("x2", 520).attr("y1", yy - 10).attr("y2", yy - 10)
        .attr("stroke", DC.line);
      r.forEach((c, j) => g.append("text").attr("x", cols[j]).attr("y", yy).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", last ? DC.a2 : DC.ink).attr("font-weight", last ? 600 : 400).text(c));
    });

    /* right-hand summary */
    const k = DL.kv(g, 570, ty + 16, { keyW: 108, lead: 15 });
    k("depth (matrices)", String(sizes.length - 1), DC.accent, true);
    k("parameters", DL.big(P), DC.a2, true);
    k("MACs / example", DL.big(M), DC.ink);
    k("MACs / batch", DL.big(M * N), DC.ink);
    k("weights at 4 B", (P * 4 / 1024).toFixed(1) + " kB", DC.muted);

    d3.select("#chain-readout").html(
      "sizes <b>" + sizes.join(" → ") + "</b> · depth <b>" + (sizes.length - 1) +
      "</b> parameterised layers · <b>" + DL.commas(P) + "</b> parameters (" +
      DL.commas(P - sizes.slice(1).reduce((a, b) => a + b, 0)) + " weights + " +
      DL.commas(sizes.slice(1).reduce((a, b) => a + b, 0)) + " biases) · <b>" +
      DL.commas(M) + "</b> multiply-accumulates per example, <b>" + DL.commas(M * N) +
      "</b> for the batch of " + N + ". The batch size changes the compute and not one parameter.");
  }
  [din, Ls, wd, dout, nb].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 2 · #neuron-svg — one unit ═════════════════ */
(function () {
  const svg = d3.select("#neuron-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, PS = 210;
  const bEl = document.getElementById("nu-b"), sEl = document.getElementById("nu-s"),
    tEl = document.getElementById("nu-t"), gEl = document.getElementById("nu-g"),
    csEl = document.getElementById("nu-cs");
  /* The direction of w lives in the <input type="range"> and NOWHERE ELSE, so
     there is a single source of truth. The drag handle below writes the angle
     back into that input and re-draws, which keeps the two in step in both
     directions and — the reason this matters — leaves a keyboard-only reader
     with a working control for every degree of freedom in the figure. */

  function draw() {
    const b = +bEl.value, s = +sEl.value, key = gEl.value, showCS = csEl.checked;
    const thetaDeg = +tEl.value, theta = thetaDeg * Math.PI / 180;
    document.getElementById("nu-bv").textContent = DL.fmt(b, 2).replace("-", "−");
    document.getElementById("nu-sv").textContent = DL.fmt(s, 2);
    document.getElementById("nu-tv").textContent = thetaDeg + "°";
    const A = DL.act(key), g = (x) => A.f(x, A.p);
    const w = [s * Math.cos(theta), s * Math.sin(theta)];

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 10 });
    const G = F.g;
    const R = 2.2;                                   // the plotted range, ±R
    const sc = d3.scaleLinear().domain([-R, R]).range([0, PS]);
    const res = 60;

    function panel(x0, title, valAt, dom) {
      G.append("text").attr("x", x0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text(title);
      const gg = G.append("g").attr("transform", `translate(${x0},0)`);
      const col = d3.scaleLinear().domain(dom).range([DC.accent, DC.a2]).interpolate(d3.interpolateRgb);
      const mid = dom.length === 3 ? dom[1] : (dom[0] + dom[1]) / 2;
      const col3 = d3.scaleLinear().domain(dom.length === 3 ? dom : [dom[0], mid, dom[1]])
        .range([DC.accent, DC.panel, DC.a2]).interpolate(d3.interpolateRgb);
      const cw = PS / res;
      DL.cells(gg, 0, 0, cw, res, res, (i, j) => {
        const X = -R + 2 * R * (i + 0.5) / res, Y = R - 2 * R * (j + 0.5) / res;
        return col3(valAt(X, Y));
      });
      gg.append("rect").attr("width", PS).attr("height", PS).attr("fill", "none").attr("stroke", DC.line);
      return gg;
    }

    const zAt = (X, Y) => w[0] * X + w[1] * Y + b;
    /* z field, symmetric colour range so the zero contour sits at the neutral colour */
    let zmax = 0;
    for (const X of [-R, R]) for (const Y of [-R, R]) zmax = Math.max(zmax, Math.abs(zAt(X, Y)));
    const p1 = panel(6, "pre-activation  z = w·x + b", zAt, [-zmax, 0, zmax]);

    let amin = Infinity, amax = -Infinity;
    for (let i = 0; i <= 30; i++) for (let j = 0; j <= 30; j++) {
      const v = g(zAt(-R + 2 * R * i / 30, -R + 2 * R * j / 30));
      amin = Math.min(amin, v); amax = Math.max(amax, v);
    }
    const p2 = panel(6 + PS + 26, "activation  a = g(z)", (X, Y) => g(zAt(X, Y)),
      [amin, (amin + amax) / 2, amax]);

    /* the z = 0 line and the weight arrow, drawn on both panels */
    [p1, p2].forEach((pp, idx) => {
      const pts = [];
      /* the line w·x + b = 0 clipped to the square */
      const n = Math.hypot(w[0], w[1]) || 1e-9;
      const ux = -w[1] / n, uy = w[0] / n;             // along the line
      const px = -b * w[0] / (n * n), py = -b * w[1] / (n * n);  // closest point to origin
      for (const t of [-6, 6]) pts.push([sc(px + t * ux), PS - sc(py + t * uy)]);
      pp.append("clipPath").attr("id", "nu-clip" + idx).append("rect").attr("width", PS).attr("height", PS);
      pp.append("line").attr("x1", pts[0][0]).attr("y1", pts[0][1])
        .attr("x2", pts[1][0]).attr("y2", pts[1][1])
        .attr("stroke", DC.ink).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3")
        .attr("clip-path", "url(#nu-clip" + idx + ")");
      if (idx === 0) {
        DL.arrow(pp, sc(0), PS - sc(0), sc(w[0] / s * 1.2), PS - sc(w[1] / s * 1.2),
          { color: DC.good, w: 2.2, head: 7 });
        pp.append("circle").attr("class", "dragpt").attr("cx", sc(w[0] / s * 1.2))
          .attr("cy", PS - sc(w[1] / s * 1.2)).attr("r", 8)
          .attr("fill", DC.good).attr("fill-opacity", 0.25).attr("stroke", DC.good)
          .style("cursor", "grab")
          .call(d3.drag().on("drag", function (ev) {
            const X = sc.invert(ev.x), Y = sc.invert(PS - ev.y);
            if (Math.hypot(X, Y) < 1e-6) return;
            /* snap to the slider's own step and domain, then write it back —
               the slider stays the single source of truth */
            const deg = ((Math.round(Math.atan2(Y, X) * 180 / Math.PI) % 360) + 360) % 360;
            if (deg !== +tEl.value) { tEl.value = deg; draw(); }
          }));
        pp.append("text").attr("x", sc(w[0] / s * 1.2) + 10).attr("y", PS - sc(w[1] / s * 1.2) - 8)
          .attr("font-size", 10.5).attr("fill", DC.good).text("w");
      }
    });

    /* the cross-section along the normal */
    if (showCS) {
      const x0 = 6 + 2 * (PS + 26), pw = 232, ph = PS;
      G.append("text").attr("x", x0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text("cross-section along ŵ");
      const gg = G.append("g").attr("transform", `translate(${x0},0)`);
      const tS = d3.scaleLinear().domain([-2.2, 2.2]).range([0, pw]);
      const lo = Math.min(-1.2, amin - 0.3), hi = Math.max(1.6, Math.max(amax, zmax) + 0.3);
      const yS = d3.scaleLinear().domain([lo, hi]).range([ph, 0]);
      DL.gridY(gg, yS, pw, 5);
      gg.append("g").attr("class", "axis").attr("transform", `translate(0,${yS(0)})`).call(d3.axisBottom(tS).ticks(5));
      gg.append("g").attr("class", "axis").call(d3.axisLeft(yS).ticks(5));
      const zline = [], aline = [];
      for (let i = 0; i <= 200; i++) {
        const t = -2.2 + 4.4 * i / 200, z = s * t + b;
        zline.push([tS(t), yS(DL.clamp(z, lo, hi))]);
        aline.push([tS(t), yS(DL.clamp(g(z), lo, hi))]);
      }
      DL.curve(gg, zline, { stroke: DC.muted, w: 1.4, dash: "4 3" });
      DL.curve(gg, aline, { stroke: DC.a2, w: 2.2 });
      const tc = -b / s;
      if (tc > -2.2 && tc < 2.2) {
        gg.append("line").attr("x1", tS(tc)).attr("x2", tS(tc)).attr("y1", 0).attr("y2", ph)
          .attr("stroke", DC.ink).attr("stroke-width", 1.4).attr("stroke-dasharray", "5 3");
        gg.append("text").attr("x", tS(tc) + 5).attr("y", 12).attr("font-size", 10)
          .attr("fill", DC.ink).text("z = 0");
      }
      DL.legend(gg, [{ color: DC.muted, label: "z (linear in t)", dash: "4 3" },
      { color: DC.a2, label: "a = g(z)" }], 6, ph - 26, { gap: 13 });
    }

    const dist = -b / (Math.hypot(w[0], w[1]) || 1e-9);
    d3.select("#neuron-readout").html(
      "w = [" + DL.fmt(w[0], 3).replace("-", "−") + ", " + DL.fmt(w[1], 3).replace("-", "−") +
      "] &nbsp; <span class=\"keep\">θ</span> = <b>" + thetaDeg + "°</b> &nbsp; ‖w‖ = <b>" + DL.fmt(s, 2) +
      "</b> &nbsp; b = <b>" + DL.fmt(b, 2).replace("-", "−") + "</b> &nbsp;→&nbsp; the hyperplane is at signed distance <b>" +
      DL.fmt(dist, 3).replace("-", "−") + "</b> from the origin along ŵ. &nbsp; g = <b>" + A.label +
      "</b>, range " + A.range + ", g(0) = <b>" + DL.fmt(A.f(0, A.p), 3) + "</b>, g′(0⁺) = <b>" +
      DL.fmt(A.df(1e-9, A.p), 3) + "</b>. " +
      (key === "identity" ? "With no non-linearity the middle panel is the left panel rescaled — nothing has happened."
        : (A.piecewiseLinear ? "The middle panel has a visible crease exactly on the dashed line."
          : "The middle panel changes smoothly; the transition band narrows as ‖w‖ grows.")));
  }
  [bEl, sEl, tEl, gEl, csEl].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 3 · #repr-svg — representation matters ═════════════════ */
(function () {
  const svg = d3.select("#repr-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const dEl = document.getElementById("rp-data"), rEl = document.getElementById("rp-rep"),
    hEl = document.getElementById("rp-h"), nEl = document.getElementById("rp-n");

  function makeData(kind, noise, seed) {
    const r = DL.rng(seed), X = [], y = [], N = 240, H = N / 2;
    /* class is decided by the FIRST/SECOND half of the index, and the
       train/test split below is by parity — if both used parity, every
       training point would be class 0 and every test point class 1, and the
       measured accuracy would be meaningless. It was, until this was fixed. */
    for (let i = 0; i < N; i++) {
      const c = i < H ? 0 : 1, k = i % H;
      let x0, x1;
      if (kind === "rings") {
        const rad = (c === 0 ? 0.45 : 1.05) + noise * DL.randn(r), th = 2 * Math.PI * r();
        x0 = rad * Math.cos(th); x1 = rad * Math.sin(th);
      } else if (kind === "xor") {
        const a = r() < 0.5 ? -1 : 1, bq = (c === 0 ? a : -a);
        x0 = a * (0.35 + 0.5 * r()) + noise * DL.randn(r);
        x1 = bq * (0.35 + 0.5 * r()) + noise * DL.randn(r);
      } else if (kind === "spiral") {
        const t = 0.25 + 1.6 * (k / H), th = 4.2 * t + (c === 0 ? 0 : Math.PI);
        x0 = t * Math.cos(th) + noise * DL.randn(r);
        x1 = t * Math.sin(th) + noise * DL.randn(r);
      } else {
        x0 = (c === 0 ? -0.6 : 0.6) + 0.35 * DL.randn(r) + noise * DL.randn(r);
        x1 = 0.45 * DL.randn(r) + noise * DL.randn(r);
      }
      X.push([x0, x1]); y.push(c);
    }
    return { X: X, y: y };
  }
  function transform(rep, X, y, h) {
    if (rep === "raw") return { Z: X.map(p => p.slice()), ax: ["x", "y"] };
    if (rep === "polar") return {
      Z: X.map(p => [Math.hypot(p[0], p[1]), Math.atan2(p[1], p[0])]), ax: ["r", "angle"]
    };
    if (rep === "quad") return { Z: X.map(p => [p[0] * p[0] + p[1] * p[1], p[0] * p[1]]), ax: ["x² + y²", "x·y"] };
    /* learned: fit a small MLP, then read its LAST hidden layer, projected to
       its two highest-variance directions so it can be drawn.               */
    const net = DL.mlpInit([2, h, h, 1], { act: "tanh", out: "sigmoid", seed: 11 });
    const Y = y.map(v => [v]);
    for (let t = 0; t < 900; t++) DL.sgdStep(net, DL.backward(net, X, Y), 1.6);
    const f = DL.forward(net, X);
    const Hm = f.a[f.a.length - 1];
    /* two-component PCA by power iteration on the covariance, for drawing */
    const d = Hm[0].length, mu = new Array(d).fill(0);
    for (const r0 of Hm) for (let j = 0; j < d; j++) mu[j] += r0[j] / Hm.length;
    const C = Hm.map(r0 => r0.map((v, j) => v - mu[j]));
    function power(C, prev) {
      let v = new Array(d).fill(0).map((_, i) => Math.sin(i * 1.7 + 0.3));
      for (let t = 0; t < 60; t++) {
        let u = new Array(d).fill(0);
        for (const r0 of C) { const s = DL.dot(r0, v); for (let j = 0; j < d; j++) u[j] += s * r0[j]; }
        if (prev) { const s = DL.dot(u, prev); for (let j = 0; j < d; j++) u[j] -= s * prev[j]; }
        const n = Math.hypot.apply(null, u) || 1;
        v = u.map(x => x / n);
      }
      return v;
    }
    const v1 = power(C, null), v2 = power(C, v1);
    return { Z: C.map(r0 => [DL.dot(r0, v1), DL.dot(r0, v2)]), ax: ["learned 1", "learned 2"], net: net };
  }

  function draw() {
    const kind = dEl.value, rep = rEl.value, h = +hEl.value, noise = +nEl.value;
    document.getElementById("rp-hv").textContent = h;
    document.getElementById("rp-nv").textContent = DL.fmt(noise, 2);
    const D = makeData(kind, noise, 3);
    /* split in half: fit on the first half, report accuracy on the second */
    const idx = d3.range(D.X.length), tr = idx.filter(i => i % 2 === 0), te = idx.filter(i => i % 2 === 1);

    const T = transform(rep, D.X, D.y, h);
    const rawS = NN.standardise(D.X), repS = NN.standardise(T.Z);
    const mRaw = NN.logistic(tr.map(i => rawS[i]), tr.map(i => D.y[i]), 600, 0.8);
    const mRep = NN.logistic(tr.map(i => repS[i]), tr.map(i => D.y[i]), 600, 0.8);
    const aRaw = NN.accuracy(mRaw, te.map(i => rawS[i]), te.map(i => D.y[i]));
    const aRep = NN.accuracy(mRep, te.map(i => repS[i]), te.map(i => D.y[i]));

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 10 });
    const G = F.g, PS = 250;

    function scatter(x0, pts, model, title, axl) {
      G.append("text").attr("x", x0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text(title);
      const gg = G.append("g").attr("transform", `translate(${x0},0)`);
      const ex = d3.extent(pts, p => p[0]), ey = d3.extent(pts, p => p[1]);
      const px = (ex[1] - ex[0]) * 0.09 + 1e-6, py = (ey[1] - ey[0]) * 0.09 + 1e-6;
      const sx = d3.scaleLinear().domain([ex[0] - px, ex[1] + px]).range([0, PS]);
      const sy = d3.scaleLinear().domain([ey[0] - py, ey[1] + py]).range([PS, 0]);
      gg.append("rect").attr("width", PS).attr("height", PS).attr("fill", DC.bg).attr("stroke", DC.line);
      /* the classifier's half-plane, shaded, computed in the STANDARDISED space
         the model was fitted in and mapped back for drawing */
      gg.append("clipPath").attr("id", "rp-c" + x0).append("rect").attr("width", PS).attr("height", PS);
      const res = 44, cw = PS / res;
      const mu = [0, 0], sd = [1, 1];
      DL.cells(gg, 0, 0, cw, res, res, (i, j) => {
        const X = sx.invert((i + 0.5) * cw), Y = sy.invert((j + 0.5) * cw);
        const z = model.w[0] * X + model.w[1] * Y + model.b;
        return z >= 0 ? "rgba(255,180,84,0.13)" : "rgba(91,156,255,0.13)";
      });
      /* the boundary line */
      const wv = model.w, bb = model.b, nn = Math.hypot(wv[0], wv[1]) || 1e-9;
      const ux = -wv[1] / nn, uy = wv[0] / nn, cx = -bb * wv[0] / (nn * nn), cy = -bb * wv[1] / (nn * nn);
      gg.append("line").attr("x1", sx(cx - 40 * ux)).attr("y1", sy(cy - 40 * uy))
        .attr("x2", sx(cx + 40 * ux)).attr("y2", sy(cy + 40 * uy))
        .attr("stroke", DC.ink).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 3")
        .attr("clip-path", "url(#rp-c" + x0 + ")");
      gg.selectAll("circle.pt").data(pts.map((p, i) => [p[0], p[1], D.y[i]])).join("circle")
        .attr("class", "pt").attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", 2.6)
        .attr("fill", d => d[2] ? DC.a2 : DC.accent).attr("fill-opacity", 0.85);
      gg.append("text").attr("x", PS / 2).attr("y", PS + 14).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", DC.muted).text(axl[0] + "  ·  " + axl[1]);
      return gg;
    }
    scatter(6, rawS, mRaw, "raw coordinates — the classifier fitted here", ["x", "y"]);
    scatter(6 + PS + 44, repS, mRep, "the selected representation", T.ax);
    DL.arrow(G, 6 + PS + 8, PS / 2, 6 + PS + 38, PS / 2, { color: DC.good, w: 2, head: 7 });
    G.append("text").attr("x", 6 + PS + 23).attr("y", PS / 2 - 10).attr("text-anchor", "middle")
      .attr("font-size", 9.5).attr("fill", DC.good).text(rep === "learned" ? "learned" : "fixed");

    /* the two accuracy bars */
    const bx = 6 + 2 * (PS + 44) + 6, bw = 140;
    G.append("text").attr("x", bx).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("held-out accuracy");
    const bs = d3.scaleLinear().domain([0.4, 1]).range([0, bw]);
    [["raw", aRaw, DC.bad], ["transformed", aRep, DC.good]].forEach((d, i) => {
      const yy = 26 + i * 46;
      G.append("text").attr("x", bx).attr("y", yy - 6).attr("font-size", 10.5).attr("fill", DC.muted).text(d[0]);
      G.append("rect").attr("x", bx).attr("y", yy).attr("width", bw).attr("height", 16).attr("rx", 3)
        .attr("fill", DC.panel2).attr("stroke", DC.line);
      G.append("rect").attr("x", bx).attr("y", yy).attr("width", Math.max(1, bs(Math.max(0.4, d[1]))))
        .attr("height", 16).attr("rx", 3).attr("fill", d[2]).attr("fill-opacity", 0.75);
      G.append("text").attr("x", bx + bw + 6).attr("y", yy + 12.5).attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
        .text((100 * d[1]).toFixed(1) + "%");
    });
    G.append("line").attr("x1", bx + bs(0.5)).attr("x2", bx + bs(0.5)).attr("y1", 20).attr("y2", 90)
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    G.append("text").attr("x", bx + bs(0.5) + 4).attr("y", 104).attr("font-size", 9.5)
      .attr("fill", DC.muted).text("50% = chance");
    G.append("text").attr("x", bx).attr("y", 132).attr("font-size", 10).attr("fill", DC.muted)
      .text("one and the same logistic");
    G.append("text").attr("x", bx).attr("y", 145).attr("font-size", 10).attr("fill", DC.muted)
      .text("classifier, fitted twice —");
    G.append("text").attr("x", bx).attr("y", 158).attr("font-size", 10).attr("fill", DC.muted)
      .text("only its input changed.");

    d3.select("#repr-readout").html(
      "<b>" + kind + "</b> · representation <b>" + rep + "</b> · held-out accuracy <b>" +
      (100 * aRaw).toFixed(1) + "%</b> on raw coordinates against <b>" + (100 * aRep).toFixed(1) +
      "%</b> after the transformation" +
      (rep === "raw" ? " — the same thing twice, so the two numbers agree up to the fitting noise."
        : (aRep - aRaw > 0.05
          ? " — a gain of " + (100 * (aRep - aRaw)).toFixed(1) + " points from changing the coordinates alone."
          : " — no real gain here, which is the honest outcome when the raw coordinates were already adequate or the transformation is the wrong one for this data.")));
  }
  [dEl, rEl, hEl, nEl].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 4 · #collapse-svg — the affine collapse ═════════════════ */
(function () {
  const svg = d3.select("#collapse-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const t1 = document.getElementById("co-t1"), s1 = document.getElementById("co-s1"),
    t2 = document.getElementById("co-t2"), sh = document.getElementById("co-sh"),
    rl = document.getElementById("co-relu");

  /* the drawn object: a coarse grid of coloured cells plus a marker glyph, so
     that a fold is visible as cells landing on top of one another            */
  const SRC = (function () {
    const P = [];
    for (let i = -4; i <= 4; i++) for (let j = -4; j <= 4; j++) P.push([i / 4, j / 4, (i + 4) * 9 + (j + 4)]);
    return P;
  })();

  function draw() {
    const a1 = +t1.value * Math.PI / 180, k1 = +s1.value, a2 = +t2.value * Math.PI / 180, sv = +sh.value;
    const useRelu = rl.checked;
    document.getElementById("co-t1v").textContent = t1.value.replace("-", "−") + "°";
    document.getElementById("co-s1v").textContent = DL.fmt(k1, 2);
    document.getElementById("co-t2v").textContent = t2.value.replace("-", "−") + "°";
    document.getElementById("co-shv").textContent = DL.fmt(sv, 2);

    const W1 = [[k1 * Math.cos(a1), k1 * Math.sin(a1)], [-k1 * Math.sin(a1), k1 * Math.cos(a1)]];
    const W2 = DL.matmul([[Math.cos(a2), Math.sin(a2)], [-Math.sin(a2), Math.cos(a2)]], [[1, sv], [0, 1]]);
    const Wstar = DL.matmul(W1, W2);

    const P0 = SRC.map(p => [p[0], p[1]]);
    const P1raw = DL.matmul(P0, W1);
    const P1 = useRelu ? DL.applyEl(P1raw, DL.ACT.relu.f) : P1raw;
    const P2 = DL.matmul(P1, W2);
    const P3 = DL.matmul(P0, Wstar);
    let maxd = 0;
    for (let i = 0; i < P2.length; i++) maxd = Math.max(maxd, Math.hypot(P2[i][0] - P3[i][0], P2[i][1] - P3[i][1]));

    const F = DL.frame(svg, W, H, { l: 14, r: 12, t: 26, b: 10 });
    const G = F.g, PS = 168, gap = 16;
    const R = 2.6;
    function pane(x0, pts, title, sub) {
      G.append("text").attr("x", x0).attr("y", -12).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text(title);
      if (sub) G.append("text").attr("x", x0).attr("y", -1).attr("font-size", 9.5).attr("fill", DC.muted).text(sub);
      const gg = G.append("g").attr("transform", `translate(${x0},6)`);
      const s = d3.scaleLinear().domain([-R, R]).range([0, PS]);
      gg.append("rect").attr("width", PS).attr("height", PS).attr("fill", DC.bg).attr("stroke", DC.line);
      gg.append("line").attr("x1", s(-R)).attr("x2", s(R)).attr("y1", PS - s(0)).attr("y2", PS - s(0))
        .attr("stroke", DC.grid);
      gg.append("line").attr("x1", s(0)).attr("x2", s(0)).attr("y1", 0).attr("y2", PS).attr("stroke", DC.grid);
      gg.selectAll("circle").data(pts.map((p, i) => [p[0], p[1], SRC[i][2]])).join("circle")
        .attr("cx", d => s(d[0])).attr("cy", d => PS - s(d[1])).attr("r", 3.4)
        .attr("fill", d => NN.hue(d[2] % 27, 27)).attr("fill-opacity", 0.9);
      return gg;
    }
    pane(6, P0, "input", "a labelled grid");
    pane(6 + PS + gap, P1, "after layer 1" + (useRelu ? " + ReLU" : ""), useRelu ? "half-planes flattened" : "affine only");
    pane(6 + 2 * (PS + gap), P2, "then layer 2", "computed in TWO steps");
    pane(6 + 3 * (PS + gap), P3, "one step: x·W*", "W* = W₁W₂");

    /* the matrices and the verdict */
    const my = PS + 34;
    DL.matText(G, W1, 8, my + 16, { dp: 3, pad: 7, label: "W₁", size: 10 });
    DL.matText(G, W2, 118, my + 16, { dp: 3, pad: 7, label: "W₂", size: 10 });
    DL.matText(G, Wstar, 232, my + 16, { dp: 3, pad: 7, label: "W* = W₁W₂", size: 10, colorOf: () => DC.a2 });
    const k = DL.kv(G, 380, my + 16, { keyW: 240, lead: 16 });
    k("max ‖two-step − one-step‖ over the 81 points",
      maxd < 1e-9 ? maxd.toExponential(2).replace("e-", "e−") : DL.fmt(maxd, 4),
      maxd < 1e-9 ? DC.good : DC.bad, true);
    k("the two right-hand panels are", maxd < 1e-9 ? "identical" : "different", maxd < 1e-9 ? DC.good : DC.bad, true);
    k("rank of W*", String(Math.abs(Wstar[0][0] * Wstar[1][1] - Wstar[0][1] * Wstar[1][0]) > 1e-12 ? 2 : 1), DC.ink);

    d3.select("#collapse-readout").html(useRelu
      ? "ReLU inserted. The two-step and one-step results now differ by <b>" + DL.fmt(maxd, 4) +
      "</b> at the worst point — the composition is no longer any single affine map, and no product of matrices can reproduce it."
      : "No activation. The two-step and one-step results agree to <b>" + maxd.toExponential(2).replace("e-", "e−") +
      "</b>, which is floating-point round-off. Two layers, <b>eight</b> weights, computing exactly what these <b>four</b> weights compute. That is the collapse, and it holds for every setting of the four sliders.");
  }
  [t1, s1, t2, sh, rl].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 5 · #xorlin-svg — no line gets four of four ═════════════════ */
(function () {
  const svg = d3.select("#xorlin-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const th = document.getElementById("xl-th"), off = document.getElementById("xl-off"),
    kk = document.getElementById("xl-k"), fn = document.getElementById("xl-fn"),
    bst = document.getElementById("xl-best");
  const PTS = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const TARGETS = {
    xor: [0, 1, 1, 0], and: [0, 0, 0, 1], or: [0, 1, 1, 1], xnor: [1, 0, 0, 1]
  };
  const NA = 360, NO = 200, OFF0 = -1.6, OFF1 = 1.6;
  const nCorrect = (t, o, y) => {
    let n = 0;
    for (let i = 0; i < 4; i++) {
      const s = Math.cos(t) * PTS[i][0] + Math.sin(t) * PTS[i][1] - o;
      if ((s >= 0 ? 1 : 0) === y[i]) n++;
    }
    return n;
  };

  function draw() {
    const t = +th.value * Math.PI / 180, o = +off.value, k = +kk.value, key = fn.value;
    const y = TARGETS[key];
    document.getElementById("xl-thv").textContent = th.value + "°";
    document.getElementById("xl-offv").textContent = DL.fmt(o, 2).replace("-", "−");
    document.getElementById("xl-kv").textContent = DL.fmt(k, 1);

    /* the exhaustive sweep — recomputed, not cached */
    const surf = [];
    let best = -1, bt = 0, bo = 0;
    for (let i = 0; i < NO; i++) {
      const row = new Array(NA);
      const oo = OFF0 + (OFF1 - OFF0) * (i + 0.5) / NO;
      for (let j = 0; j < NA; j++) {
        const tt = 2 * Math.PI * j / NA;
        const c = nCorrect(tt, oo, y);
        row[j] = c;
        if (c > best) { best = c; bt = tt; bo = oo; }
      }
      surf.push(row);
    }
    const cur = nCorrect(t, o, y);

    /* the closed-form least-squares linear fit to the four points */
    const Xd = PTS.map(p => [1, p[0], p[1]]);
    const A = DL.matmul(DL.transpose(Xd), Xd), rhs = DL.matvec(DL.transpose(Xd), y);
    /* solve the 3x3 by Gaussian elimination */
    const M = A.map((r, i) => r.concat([rhs[i]]));
    for (let c = 0; c < 3; c++) {
      let piv = c;
      for (let r2 = c + 1; r2 < 3; r2++) if (Math.abs(M[r2][c]) > Math.abs(M[piv][c])) piv = r2;
      const tmp = M[c]; M[c] = M[piv]; M[piv] = tmp;
      const pv = M[c][c];
      for (let j = c; j < 4; j++) M[c][j] /= pv;
      for (let r2 = 0; r2 < 3; r2++) if (r2 !== c) { const f = M[r2][c]; for (let j = c; j < 4; j++) M[r2][j] -= f * M[c][j]; }
    }
    const beta = [M[0][3], M[1][3], M[2][3]];
    let lsq = 0;
    for (let i = 0; i < 4; i++) { const p = beta[0] + beta[1] * PTS[i][0] + beta[2] * PTS[i][1]; lsq += (p - y[i]) * (p - y[i]) / 4; }
    let curMse = 0;
    for (let i = 0; i < 4; i++) {
      const s = Math.cos(t) * PTS[i][0] + Math.sin(t) * PTS[i][1] - o;
      const p = DL.sigmoid(k * s); curMse += (p - y[i]) * (p - y[i]) / 4;
    }

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 12 });
    const G = F.g, PS = 250;

    /* ---- left: the square ---- */
    G.append("text").attr("x", 6).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the four points and one line");
    const gg = G.append("g").attr("transform", "translate(6,0)");
    const s = d3.scaleLinear().domain([-0.45, 1.45]).range([0, PS]);
    const res = 60, cw = PS / res;
    DL.cells(gg, 0, 0, cw, res, res, (i, j) => {
      const X = s.invert((i + 0.5) * cw), Y = s.invert(PS - (j + 0.5) * cw);
      const v = DL.sigmoid(k * (Math.cos(t) * X + Math.sin(t) * Y - o));
      return d3.interpolateRgb(DC.accent, DC.a2)(v).replace("rgb", "rgba").replace(")", ",0.3)");
    });
    gg.append("rect").attr("width", PS).attr("height", PS).attr("fill", "none").attr("stroke", DC.line);
    gg.append("clipPath").attr("id", "xl-clip").append("rect").attr("width", PS).attr("height", PS);
    function line(tt, oo, col, dash) {
      const ux = -Math.sin(tt), uy = Math.cos(tt), cx = oo * Math.cos(tt), cy = oo * Math.sin(tt);
      const e = gg.append("line").attr("x1", s(cx - 9 * ux)).attr("y1", PS - s(cy - 9 * uy))
        .attr("x2", s(cx + 9 * ux)).attr("y2", PS - s(cy + 9 * uy))
        .attr("stroke", col).attr("stroke-width", 2).attr("clip-path", "url(#xl-clip)");
      if (dash) e.attr("stroke-dasharray", dash);
    }
    line(t, o, DC.ink, null);
    if (bst.checked) line(bt, bo, DC.good, "6 4");
    PTS.forEach((p, i) => {
      gg.append("circle").attr("cx", s(p[0])).attr("cy", PS - s(p[1])).attr("r", 8)
        .attr("fill", y[i] ? DC.a2 : "none").attr("stroke", y[i] ? DC.a2 : DC.accent).attr("stroke-width", 2.2);
      const sc0 = Math.cos(t) * p[0] + Math.sin(t) * p[1] - o;
      const good = ((sc0 >= 0 ? 1 : 0) === y[i]);
      gg.append("text").attr("x", s(p[0]) + 13).attr("y", PS - s(p[1]) - 8).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", good ? DC.good : DC.bad)
        .text("y=" + y[i] + (good ? " ✓" : " ✗"));
    });
    gg.append("text").attr("x", 0).attr("y", PS + 16).attr("font-size", 11)
      .attr("fill", cur === 4 ? DC.good : DC.bad)
      .text("this line gets " + cur + " of 4 right");

    /* ---- right: the sweep ---- */
    const x0 = 6 + PS + 46;
    G.append("text").attr("x", x0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("every line, swept: how many of the four");
    const g2 = G.append("g").attr("transform", `translate(${x0},0)`);
    const sw = 300, shh = PS;
    const col = c => ["#1b2130", "#25405e", "#2f6a8a", "#5b9cff", "#4ade80"][c];
    DL.cells(g2, 0, 0, 1, sw, shh, (i, j) => {
      const ai = Math.floor(i * NA / sw), oi = Math.floor((shh - 1 - j) * NO / shh);
      return col(surf[oi][ai]);
    });
    g2.append("rect").attr("width", sw).attr("height", shh).attr("fill", "none").attr("stroke", DC.line);
    const ax = d3.scaleLinear().domain([0, 360]).range([0, sw]);
    const ao = d3.scaleLinear().domain([OFF0, OFF1]).range([shh, 0]);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${shh})`).call(d3.axisBottom(ax).ticks(5));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(ao).ticks(5));
    g2.append("text").attr("x", sw).attr("y", shh + 30).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("angle (degrees)");
    g2.append("text").attr("x", 0).attr("y", -20).attr("font-size", 10).attr("fill", DC.muted).text("offset");
    g2.append("circle").attr("cx", ax(+th.value)).attr("cy", ao(o)).attr("r", 4)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 2);
    DL.legend(g2, [0, 1, 2, 3, 4].map(c => ({ color: col(c), label: c + " correct" })),
      sw + 10, 8, { gap: 14, size: 9 });
    g2.append("text").attr("x", sw + 10).attr("y", 96).attr("font-size", 11)
      .attr("fill", best === 4 ? DC.good : DC.bad).attr("font-weight", 600)
      .text("best = " + best + "/4");

    d3.select("#xorlin-readout").html(
      "target <b>" + key.toUpperCase() + "</b> · the sweep evaluates <b>" + DL.commas(NA * NO) +
      "</b> distinct lines and the best of them gets <b>" + best + " of 4</b> right." +
      (best === 4 ? " Linearly separable: a line exists, and the dashed one is it."
        : " <b>Not linearly separable.</b> No line anywhere in the sweep, and none outside it either, gets all four.") +
      " &nbsp;·&nbsp; the closed-form least-squares linear fit is ŷ = " + DL.fmt(beta[0], 3) + " + " +
      DL.fmt(beta[1], 3) + "·x₁ + " + DL.fmt(beta[2], 3) + "·x₂ with MSE <b>" + DL.fmt(lsq, 4) +
      "</b>; the current squashed line has MSE " + DL.fmt(curMse, 4) + ".");
  }
  [th, off, kk, fn, bst].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 6 · #xorsolve-svg — the exact solution, stage by stage ══════ */
(function () {
  const svg = d3.select("#xorsolve-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const st = document.getElementById("xs-stage"), c2 = document.getElementById("xs-c2"),
    w2 = document.getElementById("xs-w2"), gs = document.getElementById("xs-g");
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]], Y = [0, 1, 1, 0];
  const NAMES = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"];

  function draw() {
    const stage = +st.value, C2 = +c2.value, W2 = +w2.value, key = gs.value;
    document.getElementById("xs-c2v").textContent = DL.fmt(C2, 2).replace("-", "−");
    document.getElementById("xs-w2v").textContent = DL.fmt(W2, 2).replace("-", "−");
    const A = DL.act(key);
    const Wm = [[1, 1], [1, 1]], c = [0, C2], wv = [1, W2];

    const XW = DL.matmul(X, Wm);
    const Z = DL.addRow(XW, c);
    const Hh = DL.applyEl(Z, x => A.f(x, A.p));
    const out = Hh.map(r => r[0] * wv[0] + r[1] * wv[1]);
    const sse = out.reduce((s, v, i) => s + (v - Y[i]) * (v - Y[i]), 0);

    const stagePts = [X, XW, Z, Hh][Math.min(stage, 3)];
    const stageName = ["input space  (x₁, x₂)", "after X·W", "after +c  — collinear",
      "after g  — the fold", "after g  — the fold"][stage];

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 10 });
    const G = F.g, PS = 218;

    /* ---- left: the current space ---- */
    G.append("text").attr("x", 6).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(stageName);
    const gg = G.append("g").attr("transform", "translate(6,0)");
    const lo = Math.min(-0.5, d3.min(stagePts, p => Math.min(p[0], p[1])) - 0.4);
    const hi = Math.max(1.5, d3.max(stagePts, p => Math.max(p[0], p[1])) + 0.4);
    const s = d3.scaleLinear().domain([lo, hi]).range([0, PS]);
    gg.append("rect").attr("width", PS).attr("height", PS).attr("fill", DC.bg).attr("stroke", DC.line);
    gg.append("line").attr("x1", 0).attr("x2", PS).attr("y1", PS - s(0)).attr("y2", PS - s(0)).attr("stroke", DC.grid);
    gg.append("line").attr("x1", s(0)).attr("x2", s(0)).attr("y1", 0).attr("y2", PS).attr("stroke", DC.grid);
    if (stage === 2) {   /* the collinearity line, drawn because it is the whole point */
      gg.append("line").attr("x1", s(lo)).attr("y1", PS - s(lo - C2 - 0))
        .attr("x2", s(hi)).attr("y2", PS - s(hi - 0 + C2 + 0 - C2 - 0))
        .attr("stroke", DC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "5 3")
        .attr("opacity", 0);
      const p0 = stagePts[0], p3 = stagePts[3];
      gg.append("line").attr("x1", s(p0[0] - 0.6)).attr("y1", PS - s(p0[1] - 0.6))
        .attr("x2", s(p3[0] + 0.6)).attr("y2", PS - s(p3[1] + 0.6))
        .attr("stroke", DC.bad).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
      gg.append("text").attr("x", 8).attr("y", 15).attr("font-size", 10).attr("fill", DC.bad)
        .text("all four on one line — no linear map rises then falls");
    }
    /* group coincident points so the collision is visible as one marker */
    const groups = [];
    stagePts.forEach((p, i) => {
      const k2 = groups.find(gq => Math.hypot(gq.p[0] - p[0], gq.p[1] - p[1]) < 1e-9);
      if (k2) k2.ids.push(i); else groups.push({ p: p, ids: [i] });
    });
    groups.forEach(q => {
      const same = q.ids.every(i => Y[i] === Y[q.ids[0]]);
      gg.append("circle").attr("cx", s(q.p[0])).attr("cy", PS - s(q.p[1]))
        .attr("r", q.ids.length > 1 ? 10 : 7)
        .attr("fill", Y[q.ids[0]] && same ? DC.a2 : (same && !Y[q.ids[0]] ? "none" : DC.violet))
        .attr("stroke", Y[q.ids[0]] && same ? DC.a2 : (same ? DC.accent : DC.violet))
        .attr("stroke-width", 2.2);
      gg.append("text").attr("x", s(q.p[0]) + 13).attr("y", PS - s(q.p[1]) + 4).attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", q.ids.length > 1 ? DC.violet : DC.muted)
        .text(q.ids.map(i => NAMES[i]).join(" = "));
    });
    if (stage >= 3 && groups.length < 4)
      gg.append("text").attr("x", 8).attr("y", PS - 8).attr("font-size", 10).attr("fill", DC.violet)
        .text("two inputs now share one hidden vector — that is the fold");

    /* ---- middle: outputs against targets ---- */
    const x1 = 6 + PS + 40;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("model output against target");
    const g2 = G.append("g").attr("transform", `translate(${x1},0)`);
    const bh = PS - 24, ys = d3.scaleLinear().domain([Math.min(-0.4, d3.min(out) - 0.2), Math.max(1.4, d3.max(out) + 0.2)]).range([bh, 0]);
    DL.gridY(g2, ys, 190, 5);
    g2.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    g2.append("line").attr("x1", 0).attr("x2", 190).attr("y1", ys(0)).attr("y2", ys(0)).attr("stroke", DC.line);
    out.forEach((v, i) => {
      const bx = 14 + i * 44;
      g2.append("rect").attr("x", bx).attr("y", Math.min(ys(v), ys(0))).attr("width", 22)
        .attr("height", Math.abs(ys(v) - ys(0))).attr("fill", DC.accent).attr("fill-opacity", 0.75);
      g2.append("rect").attr("x", bx - 4).attr("y", Math.min(ys(Y[i]), ys(0))).attr("width", 30)
        .attr("height", Math.abs(ys(Y[i]) - ys(0))).attr("fill", "none")
        .attr("stroke", DC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
      g2.append("text").attr("x", bx + 11).attr("y", bh + 14).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", DC.muted).text(NAMES[i]);
    });
    DL.legend(g2, [{ color: DC.accent, label: "model output" }, { color: DC.a2, label: "target", dash: "4 3" }],
      6, bh + 32, { gap: 13 });
    g2.append("text").attr("x", 6).attr("y", bh + 62).attr("font-size", 11)
      .attr("font-family", "SF Mono, Menlo, monospace")
      .attr("fill", sse < 1e-12 ? DC.good : DC.bad)
      .text("total squared error = " + (sse < 1e-12 ? "0 (exactly)" : DL.fmt(sse, 4)));

    /* ---- right: the running table ---- */
    const x2 = x1 + 214;
    G.append("text").attr("x", x2).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("every point, every stage");
    const cols = [0, 54, 116, 178, 236, 274];
    const head = ["x", "xW", "xW+c", "g(·)", "ŷ", "y"];
    head.forEach((h, j) => G.append("text").attr("x", x2 + cols[j]).attr("y", 14).attr("font-size", 9.5)
      .attr("fill", DC.muted).text(h));
    const fm = p => "(" + DL.fmt(p[0], 1) + "," + DL.fmt(p[1], 1) + ")";
    X.forEach((p, i) => {
      const yy = 32 + i * 17;
      const good = Math.abs(out[i] - Y[i]) < 1e-9;
      const cells = [fm(X[i]), fm(XW[i]), fm(Z[i]), fm(Hh[i]), DL.fmt(out[i], 2), String(Y[i])];
      cells.forEach((cv, j) => G.append("text").attr("x", x2 + cols[j]).attr("y", yy).attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", j === 4 ? (good ? DC.good : DC.bad) : DC.ink).text(cv));
      G.append("text").attr("x", x2 + 300).attr("y", yy).attr("font-size", 11)
        .attr("fill", good ? DC.good : DC.bad).text(good ? "✓" : "✗");
    });
    const kv = DL.kv(G, x2, 118, { keyW: 78, lead: 15 });
    kv("W", "[[1,1],[1,1]]", DC.ink);
    kv("c", "[0, " + DL.fmt(C2, 2).replace("-", "−") + "]", C2 === -1 ? DC.ink : DC.a2);
    kv("w", "[1, " + DL.fmt(W2, 2).replace("-", "−") + "]", W2 === -2 ? DC.ink : DC.a2);
    kv("b", "0", DC.ink);
    kv("g", A.label, key === "relu" ? DC.ink : DC.a2);
    kv("parameters", "9", DC.muted);

    d3.select("#xorsolve-readout").html(
      sse < 1e-12
        ? "Exact. All four points hit their targets and the total squared error is <b>0</b> in exact arithmetic. The hidden layer's entire contribution is the single entry clipped from " +
        DL.fmt(Z[0][1], 2).replace("-", "−") + " to 0 in row 1."
        : "Total squared error <b>" + DL.fmt(sse, 4) + "</b> — " +
        out.filter((v, i) => Math.abs(v - Y[i]) > 1e-9).length + " of 4 points wrong. " +
        (key === "identity"
          ? "With the identity activation the network is affine, and §05 says an affine model cannot do XOR: the best it can do is the flat 0.5 surface, and this parameterisation is not even that."
          : "Return c₂ to −1.00, w₂ to −2.00 and the activation to ReLU to recover the exact solution."));
  }
  [st, c2, w2, gs].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 7 · #fold-svg — folding composes ═════════════════ */
(function () {
  const svg = d3.select("#fold-svg");
  if (svg.empty()) return;
  const W = 760, H = 410;
  const mE = document.getElementById("fd-mode"), pE = document.getElementById("fd-p"),
    LE = document.getElementById("fd-L"), sE = document.getElementById("fd-seed");

  /* a random ReLU net with d = 2 inputs, `n` units per layer, L layers. Returns
     the list of hidden pre-activations at a point, so the SIGN PATTERN can be
     read off — that pattern is what defines a linear region.                */
  function randNet(n, L, seed) {
    const r = DL.rng(seed), Ws = [], bs = [];
    let din = 2;
    for (let l = 0; l < L; l++) {
      const Wl = DL.zeros2(din, n);
      for (let i = 0; i < din; i++) for (let j = 0; j < n; j++) Wl[i][j] = Math.sqrt(2 / din) * DL.randn(r);
      Ws.push(Wl);
      bs.push(new Array(n).fill(0).map(() => 0.6 * DL.randn(r)));
      din = n;
    }
    return { Ws: Ws, bs: bs, L: L, n: n };
  }
  function randSigns(net, u, v) {
    let h = [u * 2 - 1, v * 2 - 1], key = "";
    for (let l = 0; l < net.L; l++) {
      const z = DL.vecmat(h, net.Ws[l]).map((x, j) => x + net.bs[l][j]);
      key += z.map(x => (x > 0 ? "1" : "0")).join("") + "|";
      h = z.map(x => (x > 0 ? x : 0));
    }
    return { key: key, out: h };
  }
  /* the folding construction: each layer applies the same p-piece sawtooth to
     each coordinate. p pieces per axis per layer, so exactly p^L pieces per
     axis and p⁽²ᴸ⁾ regions after L layers — a claim the sampling checks.   */
  function buildSigns(p, L, u, v) {
    let a = u, b = v, key = "";
    for (let l = 0; l < L; l++) {
      const ia = Math.min(p - 1, Math.floor(a * p)), ib = Math.min(p - 1, Math.floor(b * p));
      key += ia + "," + ib + "|";
      a = NN.sawtooth(a, p); b = NN.sawtooth(b, p);
    }
    return { key: key, out: [a, b] };
  }

  function draw() {
    const mode = mE.value, p = +pE.value, L = +LE.value, seed = +sE.value;
    document.getElementById("fd-pv").textContent = p;
    document.getElementById("fd-Lv").textContent = L;
    document.getElementById("fd-seedv").textContent = seed;
    const unitsPerLayer = 2 * p, total = unitsPerLayer * L;
    const net = randNet(unitsPerLayer, L, seed);
    const at = (u, v) => (mode === "build" ? buildSigns(p, L, u, v) : randSigns(net, u, v));

    /* count at a resolution fine enough to resolve every region we could make */
    function countAt(LL) {
      const nb = (mode === "build") ? Math.pow(p, LL) : Math.pow(2, Math.min(12, unitsPerLayer * LL / 2));
      /* at least four samples across the narrowest region, so no region can be
         missed; capped so that dragging the sliders to their maxima stays fast */
      const res = Math.max(80, Math.min(620, Math.ceil(4 * Math.min(nb, 155))));
      const nt2 = randNet(unitsPerLayer, LL, seed);
      const f = (u, v) => (mode === "build" ? buildSigns(p, LL, u, v).key : randSigns(nt2, u, v).key);
      const seen = new Set();
      for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) seen.add(f((j + 0.5) / res, (i + 0.5) / res));
      return { n: seen.size, res: res };
    }
    const here = countAt(L);
    const series = [];
    for (let l = 1; l <= 4; l++) series.push({ L: l, deep: countAt(l).n, shallow: DL.regionsShallow(2, unitsPerLayer * l) });

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 12 });
    const G = F.g, PS = 212;

    /* ---- left: the regions ---- */
    G.append("text").attr("x", 6).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the input square, one colour per affine region");
    const gg = G.append("g").attr("transform", "translate(6,0)");
    const dres = 240, cw = PS / dres;
    const keyId = new Map();
    DL.cells(gg, 0, 0, cw, dres, dres, (i, j) => {
      const k = at((i + 0.5) / dres, 1 - (j + 0.5) / dres).key;
      if (!keyId.has(k)) keyId.set(k, keyId.size);
      return NN.hue(keyId.get(k), 1);
    });
    gg.append("rect").attr("width", PS).attr("height", PS).attr("fill", "none").attr("stroke", DC.line);
    gg.append("text").attr("x", 0).attr("y", PS + 15).attr("font-size", 10.5)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.a2)
      .text(DL.commas(here.n) + " regions measured on a " + here.res + "×" + here.res + " grid");
    if (mode === "build") gg.append("text").attr("x", 0).attr("y", PS + 29).attr("font-size", 10)
      .attr("fill", DC.muted).text("the construction predicts p⁽²ᴸ⁾ = " + DL.commas(Math.pow(p, 2 * L)));

    /* ---- middle: a read-out on top of the last hidden layer ---- */
    const x1 = 6 + PS + 30;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("a fixed read-out on top of it");
    const g2 = G.append("g").attr("transform", `translate(${x1},0)`);
    DL.cells(g2, 0, 0, cw, dres, dres, (i, j) => {
      const o = at((i + 0.5) / dres, 1 - (j + 0.5) / dres).out;
      const v = DL.clamp((o[0] + o[1]) / 2, 0, 1);
      return d3.interpolateRgb(DC.accent, DC.a2)(v);
    });
    g2.append("rect").attr("width", PS).attr("height", PS).attr("fill", "none").attr("stroke", DC.line);
    g2.append("text").attr("x", 0).attr("y", PS + 15).attr("font-size", 10).attr("fill", DC.muted)
      .text(mode === "build"
        ? "one pattern, repeated " + DL.commas(Math.pow(p, 2 * L)) + " times — that is what a fold does"
        : "random weights: irregular regions, and far fewer of them");

    /* ---- right: the growth curves ---- */
    const x2 = x1 + PS + 44, pw = 216, ph = PS;
    G.append("text").attr("x", x2).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("regions against depth");
    const g3 = G.append("g").attr("transform", `translate(${x2},0)`);
    const hi = Math.max(d3.max(series, d => d.deep), d3.max(series, d => d.shallow));
    const xs = d3.scaleLinear().domain([0.7, 4.3]).range([0, pw]);
    const ys = d3.scaleLog().domain([1, hi * 1.6]).range([ph, 0]);
    DL.gridY(g3, ys, pw, 5);
    g3.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`)
      .call(d3.axisBottom(xs).tickValues([1, 2, 3, 4]).tickFormat(d3.format("d")));
    g3.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5, "~s"));
    g3.append("text").attr("x", pw).attr("y", ph + 30).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("layers");
    DL.curve(g3, series.map(d => [xs(d.L), ys(Math.max(1, d.deep))]), { stroke: DC.a2, w: 2.2 });
    DL.curve(g3, series.map(d => [xs(d.L), ys(Math.max(1, d.shallow))]), { stroke: DC.accent, w: 2, dash: "5 3" });
    series.forEach(d => {
      g3.append("circle").attr("cx", xs(d.L)).attr("cy", ys(Math.max(1, d.deep))).attr("r", 3.2).attr("fill", DC.a2);
      g3.append("circle").attr("cx", xs(d.L)).attr("cy", ys(Math.max(1, d.shallow))).attr("r", 3).attr("fill", DC.accent);
    });
    g3.append("circle").attr("cx", xs(L)).attr("cy", ys(Math.max(1, here.n))).attr("r", 7)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.8);
    DL.legend(g3, [{ color: DC.a2, label: "deep, measured" },
    { color: DC.accent, label: "1 hidden layer, exact max", dash: "5 3" }], 6, 12, { gap: 14 });
    const cross = series.find(d => d.deep > d.shallow);
    g3.append("text").attr("x", 0).attr("y", ph + 46).attr("font-size", 10).attr("fill", DC.muted)
      .text(cross ? ("deep overtakes shallow at L = " + cross.L) : "shallow still ahead at every depth shown");

    d3.select("#fold-readout").html(
      (mode === "build"
        ? "The folding construction with <b>" + p + "</b> pieces per axis per layer and <b>" + L +
        "</b> layers uses " + unitsPerLayer + " units per layer, <b>" + total +
        "</b> in total, and produces <b>" + DL.commas(here.n) + "</b> regions — the predicted p⁽²ᴸ⁾ = " +
        DL.commas(Math.pow(p, 2 * L)) + ", confirmed by counting distinct sign patterns on a " +
        here.res + "×" + here.res + " grid. "
        : "Random weights, same architecture: " + total + " units in " + L + " layers give <b>" +
        DL.commas(here.n) + "</b> regions. ") +
      "A single hidden layer with the same <b>" + total + "</b> units can reach at most <b>" +
      DL.commas(DL.regionsShallow(2, total)) + "</b> (that is 1 + " + total + " + C(" + total + ",2), exactly). " +
      (here.n > DL.regionsShallow(2, total)
        ? "<b>Depth wins here.</b>"
        : "<b>Shallow is still ahead at this size</b> — the exponential advantage of depth is asymptotic, and at two or three small layers it has not yet arrived. Raise the layer count and watch the crossover.") +
      (mode === "rand" ? " Note how far below the construction a random parameter setting falls: the counting bound is about what the family <i>can</i> express, not what an arbitrary member does." : ""));
  }
  [mE, pE, LE, sE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 8 · #shapes-svg — the tensor chain ═════════════════ */
(function () {
  const svg = d3.select("#shapes-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const nE = document.getElementById("sh-n"), d0E = document.getElementById("sh-d0"),
    hE = document.getElementById("sh-h"), LE = document.getElementById("sh-L"),
    doE = document.getElementById("sh-do"), cE = document.getElementById("sh-conv");

  function draw() {
    const N = +nE.value, d0 = +d0E.value, h = +hE.value, L = +LE.value, dO = +doE.value;
    const row = cE.value === "row";
    ["sh-nv:" + N, "sh-d0v:" + d0, "sh-hv:" + h, "sh-Lv:" + L, "sh-dov:" + dO]
      .forEach(s => { const [k, v] = s.split(":"); document.getElementById(k).textContent = v; });

    const sizes = [d0].concat(new Array(Math.max(0, L - 1)).fill(h)).concat([dO]);
    const P = DL.params(sizes), M = DL.macs(sizes);
    const actNums = sizes.slice(1).reduce((a, b) => a + b, 0) * N;

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 12 });
    const G = F.g;
    const px = d => 12 + 30 * Math.log10(1 + d);        // log-proportional side length

    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(row
        ? "row convention:  Z = X·W + b,  batch down the rows"
        : "column convention:  Z = Wᵀ·X + b,  batch across the columns");

    let x = 6;
    const yMid = 118;
    function block(a, b, label, sub, col) {
      const w = px(row ? b : a), hgt = px(row ? a : b);
      const g0 = G.append("g").attr("transform", `translate(${x},${yMid - hgt / 2})`);
      g0.append("rect").attr("width", w).attr("height", hgt).attr("rx", 3)
        .attr("fill", col).attr("fill-opacity", 0.16).attr("stroke", col);
      g0.append("text").attr("x", w / 2).attr("y", -8).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", DC.ink).text(label);
      g0.append("text").attr("x", w / 2).attr("y", hgt + 14).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted)
        .text(row ? a + "×" + b : b + "×" + a);
      if (sub) g0.append("text").attr("x", w / 2).attr("y", hgt + 26).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", DC.muted).text(sub);
      x += w + 8;
      return { w: w, h: hgt };
    }
    block(N, d0, "X", (N * d0 * 4 / 1024).toFixed(1) + " kB", DC.accent);
    for (let l = 0; l + 1 < sizes.length; l++) {
      G.append("text").attr("x", x + 4).attr("y", yMid + 4).attr("font-size", 13).attr("fill", DC.muted).text("·");
      x += 14;
      block(sizes[l], sizes[l + 1], row ? "W" + (l + 1) : "W" + (l + 1) + "ᵀ",
        DL.big(sizes[l] * sizes[l + 1]) + " w", DC.a2);
      G.append("text").attr("x", x + 2).attr("y", yMid + 4).attr("font-size", 13).attr("fill", DC.muted).text("→");
      x += 18;
      block(N, sizes[l + 1], (l + 2 === sizes.length ? "logits" : "A" + (l + 1)),
        (N * sizes[l + 1] * 4 / 1024).toFixed(1) + " kB", l + 2 === sizes.length ? DC.good : DC.violet);
    }
    /* the contracted dimensions, annotated */
    G.append("text").attr("x", 0).attr("y", 214).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the inner dimensions that must agree, and vanish");
    let cx = 6;
    for (let l = 0; l + 1 < sizes.length; l++) {
      G.append("rect").attr("x", cx).attr("y", 224).attr("width", 118).attr("height", 30).attr("rx", 5)
        .attr("fill", DC.panel2).attr("stroke", DC.line);
      G.append("text").attr("x", cx + 59).attr("y", 236).attr("text-anchor", "middle").attr("font-size", 9.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted)
        .text(row ? "(" + N + "×" + sizes[l] + ")(" + sizes[l] + "×" + sizes[l + 1] + ")"
          : "(" + sizes[l + 1] + "×" + sizes[l] + ")(" + sizes[l] + "×" + N + ")");
      G.append("text").attr("x", cx + 59).attr("y", 249).attr("text-anchor", "middle").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.a2)
        .text("contract " + sizes[l] + " → " + (row ? N + "×" + sizes[l + 1] : sizes[l + 1] + "×" + N));
      cx += 126;
    }

    /* the memory bar */
    const bw = 520, by = 296;
    const pm = P * 4, am = actNums * 4, tot = pm + am;
    G.append("text").attr("x", 0).attr("y", by - 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("memory at 4 bytes per number, for this batch");
    G.append("rect").attr("x", 0).attr("y", by).attr("width", bw * pm / tot).attr("height", 20).attr("rx", 3)
      .attr("fill", DC.a2).attr("fill-opacity", 0.75);
    G.append("rect").attr("x", bw * pm / tot).attr("y", by).attr("width", bw * am / tot).attr("height", 20)
      .attr("rx", 3).attr("fill", DC.violet).attr("fill-opacity", 0.75);
    G.append("text").attr("x", 0).attr("y", by + 34).attr("font-size", 10).attr("fill", DC.a2)
      .text("parameters " + (pm / 1024).toFixed(1) + " kB (" + (100 * pm / tot).toFixed(0) + "%)");
    G.append("text").attr("x", 190).attr("y", by + 34).attr("font-size", 10).attr("fill", DC.violet)
      .text("activations for the batch " + (am / 1024).toFixed(1) + " kB (" + (100 * am / tot).toFixed(0) + "%)");

    const k = DL.kv(G, 570, 232, { keyW: 108, lead: 15 });
    k("sizes", sizes.join("→"), DC.ink);
    k("parameters", DL.big(P), DC.a2, true);
    k("MACs / example", DL.big(M), DC.ink);
    k("MACs / batch", DL.big(M * N), DC.ink);
    k("activations", DL.big(actNums), DC.violet);

    d3.select("#shapes-readout").html(
      "<b>" + (row ? "row" : "column") + "</b> convention. Parameters <b>" + DL.commas(P) +
      "</b>, multiply-accumulates <b>" + DL.commas(M) + "</b> per example and <b>" + DL.commas(M * N) +
      "</b> for the batch of " + N + ". Switch the convention and every matrix transposes while all three of those numbers stay <i>exactly</i> the same — the convention is notation, not arithmetic. " +
      "Activation memory for this batch is " + (am / pm).toFixed(2) + "× the parameter memory" +
      (am > pm ? ", which is why batch size, not model size, is what usually runs a machine out of memory."
        : "; raise the batch size and that ratio grows linearly."));
  }
  [nE, d0E, hE, LE, doE, cE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 9 · #budget-svg — the parameter budget ═════════════════ */
(function () {
  const svg = d3.select("#budget-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const d0E = document.getElementById("bg-d0"), shE = document.getElementById("bg-shape"),
    doE = document.getElementById("bg-do"), nE = document.getElementById("bg-n");

  function draw() {
    const d0 = +d0E.value, dO = +doE.value, N = +nE.value;
    document.getElementById("bg-d0v").textContent = d0;
    document.getElementById("bg-dov").textContent = dO;
    document.getElementById("bg-nv").textContent = N;
    const preset = shE.value.split(",").map(Number);
    const sizes = [d0].concat(preset.slice(1, preset.length - 1)).concat([dO]);

    const per = [];
    for (let l = 0; l + 1 < sizes.length; l++)
      per.push({
        l: l + 1, a: sizes[l], b: sizes[l + 1],
        w: sizes[l] * sizes[l + 1], bi: sizes[l + 1],
        p: sizes[l] * sizes[l + 1] + sizes[l + 1], m: sizes[l] * sizes[l + 1]
      });
    const P = per.reduce((s, r) => s + r.p, 0), M = per.reduce((s, r) => s + r.m, 0);

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 12 });
    const G = F.g, bw = 470;

    function bar(y, key, tot, title) {
      G.append("text").attr("x", 0).attr("y", y - 7).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text(title);
      let x = 0;
      per.forEach((r, i) => {
        const wpx = bw * r[key] / tot;
        G.append("rect").attr("x", x).attr("y", y).attr("width", Math.max(0.5, wpx)).attr("height", 26)
          .attr("fill", NN.hue(i * 3 + 1, 1)).attr("fill-opacity", 0.75).attr("stroke", DC.bg);
        if (wpx > 46) {
          G.append("text").attr("x", x + wpx / 2).attr("y", y + 12).attr("text-anchor", "middle")
            .attr("font-size", 9.5).attr("fill", "#0f1117").attr("font-weight", 600)
            .text(r.a + "×" + r.b);
          G.append("text").attr("x", x + wpx / 2).attr("y", y + 22).attr("text-anchor", "middle")
            .attr("font-size", 9).attr("fill", "#0f1117").text((100 * r[key] / tot).toFixed(1) + "%");
        }
        x += wpx;
      });
      G.append("rect").attr("x", 0).attr("y", y).attr("width", bw).attr("height", 26)
        .attr("fill", "none").attr("stroke", DC.line);
    }
    bar(14, "p", P, "share of the parameters");
    bar(70, "m", M, "share of the multiply-accumulates — the same proportions, because each weight is used once per example");

    /* the table */
    const ty = 138, cols = [0, 96, 178, 250, 330];
    ["layer", "W shape", "weights", "biases", "MACs / ex"].forEach((h, j) =>
      G.append("text").attr("x", cols[j]).attr("y", ty).attr("font-size", 10).attr("fill", DC.muted).text(h));
    per.forEach((r, i) => {
      const yy = ty + 17 + i * 14;
      [String(r.l), r.a + " × " + r.b, DL.commas(r.w), DL.commas(r.bi), DL.commas(r.m)]
        .forEach((c, j) => G.append("text").attr("x", cols[j]).attr("y", yy).attr("font-size", 10.5)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(c));
    });
    const ly = ty + 17 + per.length * 14;
    G.append("line").attr("x1", 0).attr("x2", 400).attr("y1", ly - 10).attr("y2", ly - 10).attr("stroke", DC.line);
    ["total", "", DL.commas(P - sizes.slice(1).reduce((a, b) => a + b, 0)),
      DL.commas(sizes.slice(1).reduce((a, b) => a + b, 0)), DL.commas(M)]
      .forEach((c, j) => G.append("text").attr("x", cols[j]).attr("y", ly).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.a2).attr("font-weight", 600).text(c));

    /* derived quantities */
    const k = DL.kv(G, 490, 30, { keyW: 158, lead: 17, size: 11 });
    k("parameters", DL.commas(P), DC.a2, true);
    k("weights, float32", (P * 4 / 1048576).toFixed(2) + " MiB", DC.ink);
    k("forward FLOPs / example", DL.big(2 * M), DC.ink);
    k("forward FLOPs / batch", DL.big(2 * M * N), DC.ink);
    k("training FLOPs / example", "≈ " + DL.big(6 * P), DC.violet);
    k("Adam state (2 copies)", (P * 8 / 1048576).toFixed(2) + " MiB", DC.violet);
    k("first layer's share", (100 * per[0].p / P).toFixed(1) + "%", per[0].p / P > 0.5 ? DC.bad : DC.good, true);
    k("largest layer", per.reduce((a, b) => a.p > b.p ? a : b).a + " × " +
      per.reduce((a, b) => a.p > b.p ? a : b).b, DC.ink);

    const share = per[0].p / P;
    d3.select("#budget-readout").html(
      "widths <b>" + sizes.join(" → ") + "</b> · <b>" + DL.commas(P) + "</b> parameters · <b>" +
      DL.commas(M) + "</b> MACs per example, so <b>" + DL.commas(2 * M) +
      "</b> forward FLOPs and roughly <b>" + DL.commas(6 * P) + "</b> FLOPs for a full training step. " +
      "The first layer holds <b>" + (100 * share).toFixed(1) + "%</b> of the model" +
      (share > 0.5
        ? " — the input dimension dominates everything else, which is the situation a convolution exists to avoid: one 3×3×32 kernel bank costs 288 weights to look at the same image."
        : "; raise the input dimension and watch that number climb.") +
      " Biases are <b>" + (100 * sizes.slice(1).reduce((a, b) => a + b, 0) / P).toFixed(2) +
      "%</b> of the total — a rounding error, and still not optional.");
  }
  [d0E, shE, doE, nE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 10 · #actatlas-svg — the activation atlas ═════════════════ */
(function () {
  const svg = d3.select("#actatlas-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const aE = document.getElementById("aa-a"), bE = document.getElementById("aa-b"),
    pE = document.getElementById("aa-p"), dE = document.getElementById("aa-d");

  /* every property below is MEASURED from the sampled curve, so the table can
     never drift away from the picture beside it                             */
  function probe(A, p) {
    const xs = DL.linspace(-5, 5, 2001), f = xs.map(x => A.f(x, p)), df = xs.map(x => A.df(x, p));
    let mono = true;
    for (let i = 1; i < f.length; i++) if (f[i] < f[i - 1] - 1e-12) { mono = false; break; }
    let dmax = -Infinity, dat = 0;
    df.forEach((v, i) => { if (v > dmax) { dmax = v; dat = xs[i]; } });
    /* linear pieces: count sign-changes of the second difference beyond a
       tolerance — a straight segment has zero curvature to round-off        */
    let pieces = 1;
    const sec = [];
    for (let i = 1; i + 1 < f.length; i++) sec.push(f[i + 1] - 2 * f[i] + f[i - 1]);
    const tol = 1e-9;
    let curved = 0;
    for (const v of sec) if (Math.abs(v) > tol) curved++;
    if (curved <= 4) {                       /* piecewise linear: kinks are isolated */
      pieces = 1 + curved;
    } else pieces = Infinity;
    /* mean output and exact-zero fraction under a standard normal input */
    const r = DL.rng(20250910);
    let sum = 0, zero = 0, n = 40000;
    for (let i = 0; i < n; i++) {
      const z = DL.randn(r), v = A.f(z, p);
      sum += v; if (v === 0) zero++;
    }
    const lo = Math.min.apply(null, f), hi = Math.max.apply(null, f);
    return {
      lo: lo, hi: hi, f0: A.f(0, p), d0p: A.df(1e-9, p), d0m: A.df(-1e-9, p),
      mono: mono, dmax: dmax, dat: dat, pieces: pieces,
      mean: sum / n, zero: zero / n,
      bounded: (hi < 4.9 && lo > -4.9)
    };
  }

  function draw() {
    const ka = aE.value, kb = bE.value, showD = dE.checked;
    const A = DL.act(ka), B = kb === "none" ? null : DL.act(kb);
    const hasP = A.p !== undefined;
    let p = +pE.value;
    pE.disabled = !hasP;
    if (!hasP) p = undefined;
    document.getElementById("aa-pv").textContent = hasP ? DL.fmt(p, 2).replace("-", "−") : "n/a";

    const F = DL.frame(svg, W, H, { l: 44, r: 14, t: 24, b: 40 });
    const G = F.g, pw = 400, ph = H - 24 - 40;
    const xs = d3.scaleLinear().domain([-5, 5]).range([0, pw]);
    const ys = d3.scaleLinear().domain([-1.6, 3.2]).range([ph, 0]);
    DL.gridY(G, ys, pw, 6);
    G.append("g").attr("class", "axis").attr("transform", `translate(0,${ys(0)})`).call(d3.axisBottom(xs).ticks(6));
    G.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(6));
    G.append("text").attr("x", pw).attr("y", ph + 34).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("z");

    const grid = DL.linspace(-5, 5, 601);
    const clip = v => DL.clamp(v, -1.6, 3.2);
    DL.curve(G, grid.map(x => [xs(x), ys(clip(A.f(x, p)))]), { stroke: DC.a2, w: 2.4 });
    if (showD) DL.curve(G, grid.map(x => [xs(x), ys(clip(A.df(x, p)))]), { stroke: DC.a2, w: 1.2, op: 0.65, dash: "3 3" });
    if (B) {
      DL.curve(G, grid.map(x => [xs(x), ys(clip(B.f(x, B.p)))]), { stroke: DC.accent, w: 2, dash: "6 4" });
      if (showD) DL.curve(G, grid.map(x => [xs(x), ys(clip(B.df(x, B.p)))]), { stroke: DC.accent, w: 1.1, op: 0.6, dash: "2 3" });
    }
    const items = [{ color: DC.a2, label: A.label }, { color: DC.a2, label: A.label + " ′", dash: "3 3" }];
    if (B) { items.push({ color: DC.accent, label: B.label, dash: "6 4" }); items.push({ color: DC.accent, label: B.label + " ′", dash: "2 3" }); }
    DL.legend(G, showD ? items : items.filter((_, i) => i % 2 === 0), 8, 12, { gap: 14 });

    const P = probe(A, p);
    const x0 = pw + 34;
    G.append("text").attr("x", x0).attr("y", 4).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("measured from the curve");
    const k = DL.kv(G, x0, 24, { keyW: 148, lead: 16, size: 10.5 });
    k("range on [−5, 5]", "[" + DL.fmt(P.lo, 2).replace("-", "−") + ", " + DL.fmt(P.hi, 2) + "]", DC.ink);
    k("g(0)", DL.fmt(P.f0, 3).replace("-", "−"), DC.ink);
    k("g′(0⁻), g′(0⁺)", DL.fmt(P.d0m, 2).replace("-", "−") + ", " + DL.fmt(P.d0p, 2).replace("-", "−"),
      Math.abs(P.d0m - P.d0p) > 1e-6 ? DC.a2 : DC.ink);
    k("max g′", DL.fmt(P.dmax, 3) + " at z = " + DL.fmt(P.dat, 2).replace("-", "−"), DC.ink);
    k("monotone?", P.mono ? "yes" : "no", P.mono ? DC.good : DC.violet, true);
    k("bounded?", P.bounded ? "yes" : "no", P.bounded ? DC.violet : DC.good);
    k("linear pieces", isFinite(P.pieces) ? String(P.pieces) : "smooth (∞)", DC.ink);
    k("E[g(z)], z ~ N(0,1)", DL.fmt(P.mean, 3).replace("-", "−"),
      Math.abs(P.mean) < 0.02 ? DC.good : DC.a2);
    k("P[g(z) = 0 exactly]", (100 * P.zero).toFixed(1) + "%", P.zero > 0.01 ? DC.a2 : DC.muted, true);
    G.append("text").attr("x", x0).attr("y", 200).attr("font-size", 10).attr("fill", DC.muted)
      .attr("width", 200).text(A.note.length > 46 ? A.note.slice(0, 46) : A.note);
    G.append("text").attr("x", x0).attr("y", 213).attr("font-size", 10).attr("fill", DC.muted)
      .text(A.note.length > 46 ? A.note.slice(46, 94) : "");
    G.append("text").attr("x", x0).attr("y", 226).attr("font-size", 10).attr("fill", DC.muted)
      .text(A.note.length > 94 ? A.note.slice(94, 142) : "");

    d3.select("#actatlas-readout").html(
      "<b>" + A.label + "</b> — " + (isFinite(P.pieces) ? P.pieces + " linear pieces" : "smooth") +
      ", " + (P.mono ? "monotone" : "<b>not monotone</b>") + ", " +
      (P.bounded ? "bounded" : "unbounded above") +
      ", steepest slope <b>" + DL.fmt(P.dmax, 3) + "</b> at z = " + DL.fmt(P.dat, 2).replace("-", "−") +
      ", and it maps <b>" + (100 * P.zero).toFixed(1) + "%</b> of a standard normal input to exactly zero. " +
      (P.zero > 0.4 ? "Half the domain collapsing to a single value is what makes the representation sparse — and what makes the unit non-injective. "
        : (P.zero > 0 ? "" : "Nothing is mapped to exactly zero, so no information about the sign of z is destroyed. ")) +
      (B ? "Compared with <b>" + B.label + "</b>: max slope " + DL.fmt(probe(B, B.p).dmax, 3) + "." : ""));
  }
  [aE, bE, pE, dE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 11 · #gelugap-svg — exact GELU vs its surrogates ═════════ */
(function () {
  const svg = d3.select("#gelugap-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const zE = document.getElementById("gg-z"), mE = document.getElementById("gg-mode"),
    sE = document.getElementById("gg-scale");

  function draw() {
    const Z = +zE.value, mode = mE.value, logScale = sE.value === "log";
    document.getElementById("gg-zv").textContent = DL.fmt(Z, 1);

    const exact = DL.ACT.gelu, tanhF = DL.ACT.geluTanh, sigF = DL.ACT.geluSigmoid,
      relu = DL.ACT.relu, silu = DL.ACT.silu;
    const base = mode === "all" ? exact : relu;
    const series = mode === "all"
      ? [{ a: tanhF, c: DC.accent, label: "tanh form" }, { a: sigF, c: DC.violet, label: "sigmoid form" }]
      : [{ a: exact, c: DC.accent, label: "GELU (exact)" }, { a: silu, c: DC.violet, label: "SiLU" }];

    const grid = DL.linspace(-Z, Z, 1201);
    const F = DL.frame(svg, W, H, { l: 50, r: 130, t: 22, b: 34 });
    const G = F.g, pw = W - 50 - 130, ph1 = 176, ph2 = 128, gapY = 44;
    const xs = d3.scaleLinear().domain([-Z, Z]).range([0, pw]);

    /* upper: the curves */
    const flo = Math.min(-0.5, d3.min(grid, x => base.f(x, base.p)) - 0.2);
    const fhi = d3.max(grid, x => base.f(x, base.p)) + 0.2;
    const y1 = d3.scaleLinear().domain([flo, fhi]).range([ph1, 0]);
    DL.gridY(G, y1, pw, 5);
    G.append("g").attr("class", "axis").attr("transform", `translate(0,${y1(0)})`).call(d3.axisBottom(xs).ticks(7));
    G.append("g").attr("class", "axis").call(d3.axisLeft(y1).ticks(5));
    DL.curve(G, grid.map(x => [xs(x), y1(base.f(x, base.p))]), { stroke: DC.a2, w: 2.6 });
    series.forEach((s, i) => DL.curve(G, grid.map(x => [xs(x), y1(s.a.f(x, s.a.p))]),
      { stroke: s.c, w: 1.5, dash: i ? "3 3" : "6 4" }));
    DL.legend(G, [{ color: DC.a2, label: base.label }].concat(
      series.map((s, i) => ({ color: s.c, label: s.label, dash: i ? "3 3" : "6 4" }))),
      pw + 12, 10, { gap: 15 });
    /* the minimum of the exact curve, marked */
    if (mode === "all") {
      let mx = 0, mv = Infinity;
      for (const x of DL.linspace(-3, 0, 6001)) { const v = exact.f(x); if (v < mv) { mv = v; mx = x; } }
      G.append("circle").attr("cx", xs(mx)).attr("cy", y1(mv)).attr("r", 4)
        .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.6);
      G.append("text").attr("x", xs(mx) + 8).attr("y", y1(mv) + 14).attr("font-size", 9.5)
        .attr("fill", DC.ink).text("min " + DL.fmt(mv, 4).replace("-", "−") + " at " + DL.fmt(mx, 3).replace("-", "−"));
    }
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(mode === "all"
        ? "the three GELUs, on top of one another" : "GELU and SiLU against the rectifier");

    /* lower: the gap */
    const g2 = G.append("g").attr("transform", `translate(0,${ph1 + gapY})`);
    g2.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("|difference| from " + base.label);
    const gaps = series.map(s => grid.map(x => Math.abs(s.a.f(x, s.a.p) - base.f(x, base.p))));
    const gmax = d3.max(gaps.map(a => d3.max(a))) || 1;
    const gmin = Math.max(1e-9, d3.min(gaps.map(a => d3.min(a.filter(v => v > 0)))) || 1e-9);
    const y2 = logScale
      ? d3.scaleLog().domain([Math.max(1e-9, gmin), gmax * 2]).range([ph2, 0])
      : d3.scaleLinear().domain([0, gmax * 1.15]).range([ph2, 0]);
    DL.gridY(g2, y2, pw, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${ph2})`).call(d3.axisBottom(xs).ticks(7));
    g2.append("g").attr("class", "axis").call(logScale ? d3.axisLeft(y2).ticks(4, "~e") : d3.axisLeft(y2).ticks(4));
    g2.append("text").attr("x", pw).attr("y", ph2 + 30).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("x");
    const peaks = [];
    series.forEach((s, i) => {
      const pts = grid.map((x, j) => [xs(x), y2(Math.max(logScale ? gmin : 0, gaps[i][j]))]);
      DL.curve(g2, pts, { stroke: s.c, w: 1.8 });
      let bm = -1, bx = 0;
      gaps[i].forEach((v, j) => { if (v > bm) { bm = v; bx = grid[j]; } });
      peaks.push({ label: s.label, v: bm, x: bx, c: s.c });
      g2.append("circle").attr("cx", xs(bx)).attr("cy", y2(bm)).attr("r", 3.6).attr("fill", s.c);
    });
    const k = DL.kv(g2, pw + 12, 14, { keyW: 0, lead: 14, size: 10 });
    peaks.forEach(p => {
      g2.append("text").attr("x", pw + 12).attr("y", 14 + peaks.indexOf(p) * 30).attr("font-size", 10)
        .attr("fill", p.c).text(p.label);
      g2.append("text").attr("x", pw + 12).attr("y", 26 + peaks.indexOf(p) * 30).attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
        .text(DL.fmtE(p.v, 2) + " @ " + DL.fmt(p.x, 2).replace("-", "−"));
    });

    d3.select("#gelugap-readout").html(mode === "all"
      ? "Over |x| ≤ " + DL.fmt(Z, 1) + " the tanh surrogate differs from the exact x·Φ(x) by at most <b>" +
      DL.fmtE(peaks[0].v, 2) + "</b> (at x = " + DL.fmt(peaks[0].x, 2).replace("-", "−") +
      ") and the sigmoid surrogate by at most <b>" + DL.fmtE(peaks[1].v, 2) + "</b> (at x = " +
      DL.fmt(peaks[1].x, 2).replace("-", "−") + ") — a factor of <b>" +
      (peaks[1].v / peaks[0].v).toFixed(0) + "</b> between them. On the upper panel all three are one line; " +
      "they are still three different functions, and a checkpoint trained under one and served under another has been perturbed by exactly this much at every unit."
      : "The exact GELU departs from the rectifier by up to <b>" + DL.fmtE(peaks[0].v, 3) +
      "</b> and SiLU by up to <b>" + DL.fmtE(peaks[1].v, 3) + "</b> over this range. Both dip below zero — GELU to −0.169971 at x = −0.751792, SiLU to −0.278465 at x = −1.278465 — which no member of the rectified family does, and both approach the identity gradually instead of at a kink.");
  }
  [zE, mE, sE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 12 · #stable-svg — stable vs naive ═════════════════ */
(function () {
  const svg = d3.select("#stable-svg");
  if (svg.empty()) return;
  const W = 760, H = 360;
  const rE = document.getElementById("st-r"), fE = document.getElementById("st-f"),
    pE = document.getElementById("st-p");
  /* single-precision is EMULATED by rounding through a Float32Array, which is
     exactly what the hardware does, so the failure point it reports is real  */
  const f32buf = new Float32Array(1);
  const f32 = x => { f32buf[0] = x; return f32buf[0]; };
  const exp32 = x => f32(Math.exp(f32(x)));

  function draw() {
    const R = +rE.value, key = fE.value, single = pE.value === "32";
    document.getElementById("st-rv").textContent = R;
    const EXP = single ? exp32 : Math.exp;
    const LOG = single ? (x => f32(Math.log(f32(x)))) : Math.log;

    const stable = {
      bce: z => DL.softplus(-z),                      /* y = 1 */
      softplus: z => DL.softplus(z),
      logsig: z => DL.logSigmoid(z)
    }[key];
    const naive = {
      bce: z => -LOG(1 / (1 + EXP(-z))),
      softplus: z => LOG(1 + EXP(z)),
      logsig: z => LOG(1 / (1 + EXP(-z)))
    }[key];
    const label = { bce: "BCE(z, y = 1)", softplus: "softplus(z)", logsig: "log σ(z)" }[key];

    const grid = DL.linspace(-R, R, 1401);
    const sv = grid.map(stable), nv = grid.map(naive);
    /* Where does the naive form stop being finite? Scan OUTWARD from z = 0, so
       the reported threshold is the true boundary of the safe interval and not
       merely the edge of the plotted window. */
    const mid = grid.findIndex(v => v >= 0);
    let failAt = null, failAtPos = null;
    for (let i = mid; i >= 0; i--) if (!isFinite(nv[i])) { failAt = grid[i]; break; }
    for (let i = mid; i < grid.length; i++) if (!isFinite(nv[i])) { failAtPos = grid[i]; break; }

    const F = DL.frame(svg, W, H, { l: 52, r: 16, t: 24, b: 40 });
    const G = F.g, pw = 400, ph = H - 24 - 40;
    const xs = d3.scaleLinear().domain([-R, R]).range([0, pw]);
    const lo = Math.min(0, d3.min(sv)), hi = Math.max(d3.max(sv), 1);
    const ys = d3.scaleLinear().domain([lo - 0.05 * (hi - lo), hi * 1.06]).range([ph, 0]);
    DL.gridY(G, ys, pw, 5);
    G.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`).call(d3.axisBottom(xs).ticks(6));
    G.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    G.append("text").attr("x", pw).attr("y", ph + 32).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("logit z");
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(label + " — stable (solid) against naive (dashed)");

    DL.curve(G, grid.map((x, i) => [xs(x), ys(DL.clamp(sv[i], ys.domain()[0], ys.domain()[1]))]),
      { stroke: DC.good, w: 2.4 });
    /* the naive curve, broken wherever it is not finite */
    let run = [];
    grid.forEach((x, i) => {
      if (isFinite(nv[i])) run.push([xs(x), ys(DL.clamp(nv[i], ys.domain()[0], ys.domain()[1]))]);
      else { if (run.length > 1) DL.curve(G, run, { stroke: DC.bad, w: 1.6, dash: "5 4" }); run = []; }
    });
    if (run.length > 1) DL.curve(G, run, { stroke: DC.bad, w: 1.6, dash: "5 4" });
    [failAt, failAtPos].forEach(fz => {
      if (fz === null || fz === undefined) return;
      G.append("line").attr("x1", xs(fz)).attr("x2", xs(fz)).attr("y1", 0).attr("y2", ph)
        .attr("stroke", DC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3");
      G.append("text").attr("x", xs(fz) + (fz < 0 ? 5 : -5)).attr("y", 14)
        .attr("text-anchor", fz < 0 ? "start" : "end")
        .attr("font-size", 9.5).attr("fill", DC.bad).text("naive fails at z = " + DL.fmt(fz, 0).replace("-", "−"));
    });
    DL.legend(G, [{ color: DC.good, label: "stable" }, { color: DC.bad, label: "naive", dash: "5 4" }],
      8, 12, { gap: 14 });

    /* the table of specific values */
    const x0 = pw + 40;
    G.append("text").attr("x", x0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("six values, printed");
    const zs = [-R, -Math.round(R / 2), -10, 0, 10, R];
    G.append("text").attr("x", x0).attr("y", 12).attr("font-size", 9.5).attr("fill", DC.muted).text("z");
    G.append("text").attr("x", x0 + 52).attr("y", 12).attr("font-size", 9.5).attr("fill", DC.muted).text("stable");
    G.append("text").attr("x", x0 + 152).attr("y", 12).attr("font-size", 9.5).attr("fill", DC.muted).text("naive");
    zs.forEach((z, i) => {
      const yy = 30 + i * 16, s = stable(z), n = naive(z);
      G.append("text").attr("x", x0).attr("y", yy).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
        .text(DL.fmt(z, 0).replace("-", "−"));
      G.append("text").attr("x", x0 + 52).attr("y", yy).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.good).text(DL.fmt(s, 6).replace("-", "−"));
      G.append("text").attr("x", x0 + 152).attr("y", yy).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", isFinite(n) ? DC.muted : DC.bad)
        .text(isFinite(n) ? DL.fmt(n, 6).replace("-", "−") : (isNaN(n) ? "NaN" : (n > 0 ? "+∞" : "−∞")));
    });
    /* the max difference over the finite part */
    let md = 0;
    grid.forEach((x, i) => { if (isFinite(nv[i])) md = Math.max(md, Math.abs(nv[i] - sv[i])); });
    const k = DL.kv(G, x0, 150, { keyW: 152, lead: 15, size: 10.5 });
    k("precision", single ? "float32 (emulated)" : "float64", DC.ink);
    k("max |naive − stable|", DL.fmtE(md, 2), md < 1e-9 ? DC.good : DC.a2, true);
    k("naive is finite on", failAt === null ? "the whole range" :
      "|z| < " + DL.fmt(Math.abs(failAt), 0), failAt === null ? DC.good : DC.bad, true);
    k("exp underflows near", single ? "z ≈ −104" : "z ≈ −746", DC.muted);
    k("exp overflows near", single ? "z ≈ +89" : "z ≈ +710", DC.muted);

    d3.select("#stable-readout").html(
      failAt === null
        ? "Over |z| ≤ " + R + " in " + (single ? "float32" : "float64") +
        " the naive form is still finite and agrees with the stable one to <b>" + DL.fmtE(md, 2) +
        "</b> — they are the same computation here. Widen the range until it is not."
        : "The naive form returns a non-finite value from <b>z = " + DL.fmt(failAt, 0).replace("-", "−") +
        "</b> outward, where the true value is <b>" + DL.fmt(stable(failAt), 4) +
        "</b> — a large but perfectly ordinary number. Inside the safe interval the two agree to <b>" +
        DL.fmtE(md, 2) + "</b>, which is why this bug survives every test that does not push the logits out. " +
        (single ? "In float32 the safe interval is far narrower than in float64, and a confident model reaches it."
          : "Switch to float32 and watch the safe interval collapse."));
  }
  [rE, fE, pE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 13 · #lossgrad-svg — output unit × loss ═════════════════ */
(function () {
  const svg = d3.select("#lossgrad-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const pE = document.getElementById("lg-pair"), yE = document.getElementById("lg-y"),
    sE = document.getElementById("lg-sc"), bE = document.getElementById("lg-both");

  /* Each pairing, as (loss, dL/dz) at a scalar logit z with a 0/1 target.
     The softmax cases use the two-class softmax([z, 0]), which equals σ(z) —
     the point being that softmax and sigmoid are the same object at K = 2.  */
  const PAIRS = {
    "sig-ce": {
      label: "sigmoid + cross-entropy", c: DC.good,
      L: (z, y) => DL.bce(z, y),
      g: (z, y) => DL.sigmoid(z) - y
    },
    "sig-mse": {
      label: "sigmoid + squared error", c: DC.bad,
      L: (z, y) => { const p = DL.sigmoid(z); return 0.5 * (p - y) * (p - y); },
      g: (z, y) => { const p = DL.sigmoid(z); return (p - y) * p * (1 - p); }
    },
    "soft-ce": {
      label: "softmax + cross-entropy", c: DC.teal,
      L: (z, y) => DL.xent([z, 0], y === 1 ? 0 : 1),
      g: (z, y) => DL.softmax([z, 0])[0] - y
    },
    "soft-mse": {
      label: "softmax + squared error", c: DC.violet,
      L: (z, y) => { const p = DL.softmax([z, 0]); const t = [y, 1 - y]; return 0.5 * ((p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2); },
      g: (z, y) => { const p = DL.softmax([z, 0])[0]; return 2 * (p - y) * p * (1 - p); }
    },
    "lin-mse": {
      label: "linear + squared error", c: DC.accent,
      L: (z, y) => 0.5 * (z - y) * (z - y),
      g: (z, y) => z - y
    }
  };

  function draw() {
    const key = pE.value, y = +yE.value, logScale = sE.value === "log", both = bE.checked;
    const P = PAIRS[key], R = 10;
    const grid = DL.linspace(-R, R, 1601);

    const F = DL.frame(svg, W, H, { l: 54, r: 152, t: 22, b: 34 });
    const G = F.g, pw = W - 54 - 152, ph1 = 150, ph2 = 150, gapY = 46;
    const xs = d3.scaleLinear().domain([-R, R]).range([0, pw]);

    /* upper: the loss */
    const Lv = grid.map(z => P.L(z, y));
    const y1 = d3.scaleLinear().domain([0, Math.min(12, d3.max(Lv)) * 1.08 + 0.02]).range([ph1, 0]);
    DL.gridY(G, y1, pw, 5);
    G.append("g").attr("class", "axis").attr("transform", `translate(0,${ph1})`).call(d3.axisBottom(xs).ticks(7));
    G.append("g").attr("class", "axis").call(d3.axisLeft(y1).ticks(5));
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("loss against the logit z, target y = " + y);
    DL.curve(G, grid.map((z, i) => [xs(z), y1(DL.clamp(Lv[i], 0, y1.domain()[1]))]), { stroke: P.c, w: 2.4 });
    G.append("text").attr("x", pw + 10).attr("y", 14).attr("font-size", 10.5).attr("fill", P.c)
      .attr("font-weight", 600).text(P.label);
    const wrongSide = y === 1 ? -R : R;
    G.append("text").attr("x", xs(wrongSide) + (y === 1 ? 6 : -6)).attr("y", ph1 - 8)
      .attr("text-anchor", y === 1 ? "start" : "end")
      .attr("font-size", 9.5).attr("fill", DC.muted).text("confidently WRONG");

    /* lower: |dL/dz| */
    const g2 = G.append("g").attr("transform", `translate(0,${ph1 + gapY})`);
    g2.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("|∂L/∂z| — the gradient the logit actually receives");
    const shown = both ? Object.keys(PAIRS) : [key];
    const all = shown.map(k => ({ k: k, v: grid.map(z => Math.abs(PAIRS[k].g(z, y))) }));
    const gmax = d3.max(all.map(a => d3.max(a.v)));
    const gmin = Math.max(1e-12, d3.min(all.map(a => d3.min(a.v.filter(v => v > 0)))) || 1e-12);
    const y2 = logScale ? d3.scaleLog().domain([gmin, gmax * 1.6]).range([ph2, 0])
      : d3.scaleLinear().domain([0, gmax * 1.08]).range([ph2, 0]);
    DL.gridY(g2, y2, pw, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${ph2})`).call(d3.axisBottom(xs).ticks(7));
    g2.append("g").attr("class", "axis").call(logScale ? d3.axisLeft(y2).ticks(5, "~e") : d3.axisLeft(y2).ticks(5));
    g2.append("text").attr("x", pw).attr("y", ph2 + 30).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("logit z");
    all.forEach(a => {
      const isCur = a.k === key;
      DL.curve(g2, grid.map((z, i) => [xs(z), y2(Math.max(logScale ? gmin : 0, a.v[i]))]),
        { stroke: PAIRS[a.k].c, w: isCur ? 2.4 : 1.2, op: isCur ? 1 : 0.6 });
    });
    if (both) DL.legend(g2, shown.map(k => ({ color: PAIRS[k].c, label: PAIRS[k].label })),
      pw + 10, 10, { gap: 15, font: 9.5 });

    /* the measured comparison at the confidently-wrong end */
    const zc = y === 1 ? -10 : 10;
    const gCE = Math.abs(PAIRS["sig-ce"].g(zc, y)), gMSE = Math.abs(PAIRS["sig-mse"].g(zc, y));
    g2.append("line").attr("x1", xs(zc)).attr("x2", xs(zc)).attr("y1", 0).attr("y2", ph2)
      .attr("stroke", DC.ink).attr("stroke-width", 1.2).attr("stroke-dasharray", "4 3");
    const k = DL.kv(g2, pw + 10, both ? 100 : 16, { keyW: 0, lead: 13, size: 10 });
    [["at z = " + String(zc).replace("-", "−") + " (confidently wrong):", DC.muted],
    ["cross-entropy  " + DL.fmtE(gCE, 4), DC.good],
    ["squared error  " + DL.fmtE(gMSE, 4), DC.bad],
    ["ratio  " + DL.commas(Math.round(gCE / gMSE)) + "×", DC.a2]]
      .forEach((r, i) => g2.append("text").attr("x", pw + 10).attr("y", (both ? 106 : 22) + i * 14)
        .attr("font-size", 10).attr("font-family", i ? "SF Mono, Menlo, monospace" : "inherit")
        .attr("fill", r[1]).text(r[0]));

    const closed = 2 + Math.exp(zc) + Math.exp(-zc);
    d3.select("#lossgrad-readout").html(
      "<b>" + P.label + "</b>, y = " + y + ". At z = " + String(zc).replace("-", "−") +
      " — a model that is confidently and completely wrong — cross-entropy delivers <b>" +
      DL.fmtE(gCE, 6) + "</b> to the logit and squared error delivers <b>" + DL.fmtE(gMSE, 6) +
      "</b>, a ratio of <b>" + DL.commas(Math.round(gCE / gMSE)) +
      "</b>. That ratio is exactly 1/σ′(z) = 2 + eᶻ + e⁻ᶻ = " + DL.fmt(closed, 4) +
      ", so it grows like e^|z|: the further the squared-error model is from the truth, the less it learns. " +
      (key.indexOf("mse") > 0 && key !== "lin-mse"
        ? "The upper panel shows the other half of the same fact: this loss is <b>bounded</b>, so being catastrophically wrong costs no more than being merely wrong."
        : "The upper panel shows the other half: cross-entropy grows linearly in |z| on the wrong side, so being catastrophically wrong costs proportionally more."));
  }
  [pE, yE, sE, bE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 14 · #softmax-svg — softmax, live ═════════════════ */
(function () {
  const svg = d3.select("#softmax-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const gE = document.getElementById("sm-gap"), TE = document.getElementById("sm-T"),
    cE = document.getElementById("sm-c"), kE = document.getElementById("sm-k"),
    nvE = document.getElementById("sm-naive");

  function draw() {
    const gap = +gE.value, T = +TE.value, c = +cE.value, K = +kE.value, showNaive = nvE.checked;
    document.getElementById("sm-gapv").textContent = DL.fmt(gap, 1);
    document.getElementById("sm-Tv").textContent = DL.fmt(T, 2);
    document.getElementById("sm-cv").textContent = String(c).replace("-", "−");
    document.getElementById("sm-kv").textContent = K;

    /* a fixed, reproducible logit shape, scaled by the gap control */
    const base = DL.linspace(0, 1, K).map((t, i) => gap * (1 - t) * (1 - 0.25 * Math.sin(3.1 * i)));
    const z = base.map(v => v + c);
    const p = DL.softmax(z, T);
    const pNaive = DL.naive.softmax(z.map(v => v / T));
    const p0 = DL.softmax(base, T);                   /* the unshifted answer */
    let shiftDrift = 0;
    p.forEach((v, i) => { shiftDrift = Math.max(shiftDrift, Math.abs(v - p0[i])); });
    const H0 = -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
    const argmax = p.indexOf(Math.max.apply(null, p));

    const F = DL.frame(svg, W, H, { l: 40, r: 14, t: 26, b: 34 });
    const G = F.g, ph = H - 26 - 34;

    /* ---- left: the logits ---- */
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("logits z (after the shift)");
    const zlo = Math.min(0, d3.min(z)), zhi = Math.max(d3.max(z), zlo + 1);
    const zs = d3.scaleLinear().domain([zlo - 0.1 * (zhi - zlo) - 0.2, zhi + 0.1 * (zhi - zlo) + 0.2]).range([ph - 40, 0]);
    G.append("g").attr("class", "axis").call(d3.axisLeft(zs).ticks(5));
    z.forEach((v, i) => {
      const bx = i * (170 / K), bwid = Math.max(6, 170 / K - 6);
      G.append("rect").attr("x", bx).attr("y", Math.min(zs(v), zs(0))).attr("width", bwid)
        .attr("height", Math.abs(zs(v) - zs(0))).attr("fill", i === argmax ? DC.a2 : DC.accent)
        .attr("fill-opacity", 0.75);
      if (K <= 6) G.append("text").attr("x", bx + bwid / 2).attr("y", ph - 26).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted)
        .text(DL.fmt(v, 1).replace("-", "−"));
    });

    /* ---- middle: the probabilities ---- */
    const x1 = 232;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("softmax(z / T) — stable" + (showNaive ? ", and naive" : ""));
    const ps = d3.scaleLinear().domain([0, 1]).range([ph - 40, 0]);
    const g2 = G.append("g").attr("transform", `translate(${x1},0)`);
    DL.gridY(g2, ps, 200, 5);
    g2.append("g").attr("class", "axis").call(d3.axisLeft(ps).ticks(5, ".1f"));
    p.forEach((v, i) => {
      const bx = i * (200 / K), bwid = Math.max(5, 200 / K - (showNaive ? 8 : 6));
      const half = showNaive ? bwid / 2 : bwid;
      g2.append("rect").attr("x", bx).attr("y", ps(v)).attr("width", half)
        .attr("height", ps(0) - ps(v)).attr("fill", i === argmax ? DC.a2 : DC.accent).attr("fill-opacity", 0.8);
      if (showNaive) {
        const nv2 = pNaive[i];
        if (isFinite(nv2)) g2.append("rect").attr("x", bx + half + 2).attr("y", ps(nv2)).attr("width", half)
          .attr("height", ps(0) - ps(nv2)).attr("fill", DC.bad).attr("fill-opacity", 0.55);
        else g2.append("text").attr("x", bx + half + 2).attr("y", ps(0) - 4).attr("font-size", 9)
          .attr("fill", DC.bad).attr("transform", `rotate(-90,${bx + half + 2},${ps(0) - 4})`).text("NaN");
      }
      if (K <= 6) g2.append("text").attr("x", bx + (showNaive ? bwid / 2 : bwid / 2)).attr("y", ph - 26)
        .attr("text-anchor", "middle").attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text(DL.fmt(v, 3));
    });
    g2.append("text").attr("x", 0).attr("y", ph - 12).attr("font-size", 10).attr("fill", DC.muted)
      .text("∑ = " + DL.fmt(p.reduce((a, b) => a + b, 0), 6));

    /* ---- right: entropy and max probability against temperature ---- */
    const x2 = 480, pw = 246, ch = ph - 56;
    G.append("text").attr("x", x2).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("against temperature");
    const g3 = G.append("g").attr("transform", `translate(${x2},0)`);
    const ts = d3.scaleLog().domain([0.05, 20]).range([0, pw]);
    const vs = d3.scaleLinear().domain([0, Math.max(1, Math.log(K)) * 1.08]).range([ch, 0]);
    DL.gridY(g3, vs, pw, 5);
    g3.append("g").attr("class", "axis").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(ts).ticks(4, "~g"));
    g3.append("g").attr("class", "axis").call(d3.axisLeft(vs).ticks(5));
    g3.append("line").attr("x1", 0).attr("x2", pw).attr("y1", vs(Math.log(K))).attr("y2", vs(Math.log(K)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "4 3");
    g3.append("text").attr("x", pw - 2).attr("y", vs(Math.log(K)) - 4).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", DC.muted).text("log K = " + DL.fmt(Math.log(K), 3));
    const Ts = DL.linspace(Math.log(0.05), Math.log(20), 220).map(Math.exp);
    const ent = [], mxp = [];
    Ts.forEach(t => {
      const q = DL.softmax(base, t);
      ent.push([ts(t), vs(-q.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0))]);
      mxp.push([ts(t), vs(Math.max.apply(null, q))]);
    });
    DL.curve(g3, ent, { stroke: DC.teal, w: 2 });
    DL.curve(g3, mxp, { stroke: DC.a2, w: 2, dash: "5 3" });
    g3.append("circle").attr("cx", ts(T)).attr("cy", vs(H0)).attr("r", 4).attr("fill", DC.teal);
    g3.append("circle").attr("cx", ts(T)).attr("cy", vs(Math.max.apply(null, p))).attr("r", 4).attr("fill", DC.a2);
    g3.append("line").attr("x1", ts(T)).attr("x2", ts(T)).attr("y1", 0).attr("y2", ch)
      .attr("stroke", DC.ink).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    DL.legend(g3, [{ color: DC.teal, label: "entropy (nats)" }, { color: DC.a2, label: "max probability", dash: "5 3" }],
      6, 12, { gap: 14 });
    g3.append("text").attr("x", 0).attr("y", ch + 44).attr("font-size", 10).attr("fill", DC.muted)
      .text("arg max = class " + argmax + ", unchanged by T for every T > 0");

    d3.select("#softmax-readout").html(
      "K = " + K + ", T = " + DL.fmt(T, 2) + ", every logit shifted by <b>" + String(c).replace("-", "−") +
      "</b>. The stable output moved by <b>" + DL.fmtE(shiftDrift, 2) +
      "</b> from its unshifted value — that is shift invariance, measured, not asserted. " +
      (showNaive
        ? (pNaive.every(isFinite)
          ? "The naive computation still agrees here (max difference " +
          DL.fmtE(Math.max.apply(null, p.map((v, i) => Math.abs(v - pNaive[i]))), 2) + ")."
          : "<b>The naive computation has returned not-a-number</b>: e^" + DL.fmt(Math.max.apply(null, z), 0) +
          " overflows to ∞, and the ratio ∞ ÷ ∞ has no floating-point value. The stable form got the right answer from the very same logits.")
        : "Tick the naive box and push the shift past ±710 to watch the obvious implementation fail on logits the stable one handles.") +
      " Entropy <b>" + DL.fmt(H0, 4) + "</b> nats against a maximum of log K = " + DL.fmt(Math.log(K), 4) +
      "; largest probability <b>" + DL.fmt(Math.max.apply(null, p), 4) + "</b>.");
  }
  [gE, TE, cE, kE, nvE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 15 · #mdn-svg — mixture density output ═════════════════ */
(function () {
  const svg = d3.select("#mdn-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const xE = document.getElementById("md-x"), sepE = document.getElementById("md-sep"),
    nE = document.getElementById("md-n"), sE = document.getElementById("md-s");

  const branch = (x, sep, up) => 0.15 + 0.7 * x + (up ? 1 : -1) * 0.5 * sep * x;

  function draw() {
    const X0 = +xE.value, sep = +sepE.value, n = +nE.value, sd = +sE.value;
    document.getElementById("md-xv").textContent = DL.fmt(X0, 2);
    document.getElementById("md-sepv").textContent = DL.fmt(sep, 2);
    document.getElementById("md-nv").textContent = n;
    document.getElementById("md-sv").textContent = DL.fmt(sd, 2);

    /* the data: two branches, equally likely */
    const r = DL.rng(41), N = 900, pts = [];
    for (let i = 0; i < N; i++) {
      const x = r(), up = r() < 0.5;
      pts.push([x, branch(x, sep, up) + sd * DL.randn(r)]);
    }
    /* the slice around the selected x, which is what both models are scored on */
    const hw = 0.06;
    const slice = pts.filter(q => Math.abs(q[0] - X0) < hw).map(q => q[1]);

    /* the single Gaussian a linear + squared-error model would fit at this x:
       its mean IS the conditional mean, and its variance the residual variance */
    const mu1 = d3.mean(slice), sd1 = Math.sqrt(d3.mean(slice.map(v => (v - mu1) * (v - mu1))));
    /* the mixture, fitted to the slice by EM — n components, 1-D              */
    function fitEM(y, K, iters) {
      const lo = d3.min(y), hi = d3.max(y);
      let mu = DL.linspace(lo + (hi - lo) * 0.15, hi - (hi - lo) * 0.15, K);
      let sg = new Array(K).fill(Math.max(0.03, (hi - lo) / (3 * K)));
      let pi = new Array(K).fill(1 / K);
      for (let t = 0; t < (iters || 120); t++) {
        const R = y.map(v => {
          const lp = mu.map((m, k) => Math.log(Math.max(1e-12, pi[k])) - Math.log(sg[k]) - 0.5 * ((v - m) / sg[k]) ** 2);
          const Ls = DL.logSumExp(lp);
          return lp.map(l => Math.exp(l - Ls));
        });
        for (let k = 0; k < K; k++) {
          let s0 = 0, s1 = 0, s2 = 0;
          R.forEach((row, i) => { s0 += row[k]; s1 += row[k] * y[i]; });
          pi[k] = s0 / y.length;
          mu[k] = s0 > 1e-9 ? s1 / s0 : mu[k];
          R.forEach((row, i) => { s2 += row[k] * (y[i] - mu[k]) ** 2; });
          sg[k] = Math.max(0.02, Math.sqrt(s0 > 1e-9 ? s2 / s0 : sg[k] * sg[k]));
        }
      }
      return { pi: pi, mu: mu, sg: sg };
    }
    const mix = fitEM(slice, n, 140);
    const gauss = (v, m, s) => Math.exp(-0.5 * ((v - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    const pMix = v => mix.pi.reduce((s, w, k) => s + w * gauss(v, mix.mu[k], mix.sg[k]), 0);
    const pOne = v => gauss(v, mu1, sd1);
    /* held-out half of the slice, so the comparison is not self-scored */
    const held = slice.filter((_, i) => i % 2 === 1);
    const nllMix = -d3.mean(held.map(v => Math.log(Math.max(1e-300, pMix(v)))));
    const nllOne = -d3.mean(held.map(v => Math.log(Math.max(1e-300, pOne(v)))));

    const F = DL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 40 });
    const G = F.g, pw = 330, ph = H - 26 - 40;
    const xs = d3.scaleLinear().domain([0, 1]).range([0, pw]);
    const ylo = d3.min(pts, q => q[1]) - 0.1, yhi = d3.max(pts, q => q[1]) + 0.1;
    const ys = d3.scaleLinear().domain([ylo, yhi]).range([ph, 0]);
    DL.gridY(G, ys, pw, 5);
    G.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`).call(d3.axisBottom(xs).ticks(5));
    G.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    G.append("text").attr("x", pw).attr("y", ph + 32).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("x");
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("a bimodal conditional distribution");
    G.selectAll("circle.d").data(pts).join("circle").attr("class", "d")
      .attr("cx", d => xs(d[0])).attr("cy", d => ys(d[1])).attr("r", 1.7)
      .attr("fill", DC.muted).attr("fill-opacity", 0.5);
    const gx = DL.linspace(0, 1, 100);
    DL.curve(G, gx.map(x => [xs(x), ys((branch(x, sep, true) + branch(x, sep, false)) / 2)]),
      { stroke: DC.bad, w: 2.4 });
    [true, false].forEach(up => DL.curve(G, gx.map(x => [xs(x), ys(branch(x, sep, up))]),
      { stroke: DC.good, w: 1.4, dash: "5 3" }));
    G.append("rect").attr("x", xs(X0 - hw)).attr("y", 0).attr("width", xs(2 * hw) - xs(0)).attr("height", ph)
      .attr("fill", DC.a2).attr("fill-opacity", 0.1).attr("stroke", DC.a2).attr("stroke-dasharray", "3 3");
    DL.legend(G, [{ color: DC.bad, label: "conditional mean — what MSE fits" },
    { color: DC.good, label: "the two branches", dash: "5 3" }], 8, 14, { gap: 14 });

    /* right: the density at the selected x */
    const x1 = pw + 56, dw = 300;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("p(y | x = " + DL.fmt(X0, 2) + ")");
    const g2 = G.append("g").attr("transform", `translate(${x1},0)`);
    const yv = DL.linspace(ylo, yhi, 320);
    const dmax = Math.max(d3.max(yv.map(pMix)), d3.max(yv.map(pOne)));
    const dsc = d3.scaleLinear().domain([0, dmax * 1.1]).range([0, dw - 40]);
    g2.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    /* the histogram of the slice */
    const bins = d3.histogram().domain([ylo, yhi]).thresholds(28)(slice);
    const bmax = d3.max(bins, b => b.length) || 1;
    g2.selectAll("rect.h").data(bins).join("rect").attr("class", "h")
      .attr("x", 0).attr("y", d => ys(d.x1))
      .attr("width", d => (dw - 40) * (d.length / bmax) * 0.55)
      .attr("height", d => Math.max(1, ys(d.x0) - ys(d.x1) - 1))
      .attr("fill", DC.muted).attr("fill-opacity", 0.28);
    DL.curve(g2, yv.map(v => [dsc(pOne(v)), ys(v)]), { stroke: DC.bad, w: 2.2 });
    DL.curve(g2, yv.map(v => [dsc(pMix(v)), ys(v)]), { stroke: DC.good, w: 2.2 });
    mix.mu.forEach((m, k) => {
      g2.append("line").attr("x1", 0).attr("x2", dsc(mix.pi[k] * gauss(m, m, mix.sg[k])))
        .attr("y1", ys(m)).attr("y2", ys(m)).attr("stroke", DC.good).attr("stroke-dasharray", "2 2")
        .attr("stroke-opacity", 0.6);
    });
    g2.append("circle").attr("cx", dsc(pOne(mu1))).attr("cy", ys(mu1)).attr("r", 4)
      .attr("fill", "none").attr("stroke", DC.bad).attr("stroke-width", 2);
    DL.legend(g2, [{ color: DC.bad, label: "one Gaussian (MSE optimum)" },
    { color: DC.good, label: n + "-component mixture" },
    { color: DC.muted, label: "data in the slice", op: 0.4 }], 6, 14, { gap: 14 });

    const k = DL.kv(G, x1, ph - 66, { keyW: 176, lead: 15, size: 10.5 });
    k("held-out NLL, one Gaussian", DL.fmt(nllOne, 4), DC.bad, true);
    k("held-out NLL, mixture", DL.fmt(nllMix, 4), DC.good, true);
    k("mixture weights", mix.pi.map(v => DL.fmt(v, 2)).join("  "), DC.ink);
    k("mixture means", mix.mu.map(v => DL.fmt(v, 2)).join("  "), DC.ink);

    const gapMode = Math.abs(mix.mu[0] - mix.mu[mix.mu.length - 1]);
    d3.select("#mdn-readout").html(
      "At x = " + DL.fmt(X0, 2) + " the conditional mean is <b>" + DL.fmt(mu1, 3).replace("-", "−") +
      "</b>, and the fraction of the slice lying within ±0.1 of it is <b>" + DL.fmt(d3.mean(slice.map(v => Math.abs(v - mu1) < 0.1 ? 1 : 0)) * 100, 1) +
      "%</b> — the conditional mean sits in the gap where the data is not. " +
      "Held-out negative log-likelihood: one Gaussian <b>" + DL.fmt(nllOne, 4) +
      "</b>, " + n + "-component mixture <b>" + DL.fmt(nllMix, 4).replace("-", "−") + "</b>" +
      (nllMix < nllOne - 0.02
        ? " — the mixture is better by " + DL.fmt(nllOne - nllMix, 3) + " nats per point."
        : (sep < 0.15 || n === 1
          ? " — with the branches merged (or one component allowed) the two models are the same object, and the numbers agree. That is the honest limiting case."
          : " — no advantage here.")) +
      " Squared error does not fit the data badly by accident; it fits the <i>mean</i> correctly, and the mean is the wrong summary of a bimodal distribution.");
  }
  [xE, sepE, nE, sE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 16 · #ua-svg — universal approximation, measured ═════════ */
(function () {
  const svg = d3.select("#ua-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const fE = document.getElementById("ua-f"), hE = document.getElementById("ua-h"),
    eE = document.getElementById("ua-ext");
  const TARGETS = {
    sin: { f: x => Math.sin(2 * Math.PI * x), label: "sin 2πx", cont: true },
    bumpy: { f: x => Math.sin(3.1 * Math.PI * x) * Math.exp(-1.3 * x) + 0.45 * Math.cos(7.3 * Math.PI * x) * x, label: "a wiggly function", cont: true },
    step: { f: x => (x < 0.5 ? -0.7 : 0.7), label: "a step at x = ½", cont: false },
    spike: { f: x => 1.4 * Math.exp(-260 * (x - 0.42) * (x - 0.42)) - 0.2, label: "a narrow spike", cont: true }
  };

  /* Least squares on the OUTPUT layer with the kinks placed evenly across the
     interval. The hidden layer is not trained — that is deliberate: the
     theorem is about EXISTENCE, so the honest demonstration is a construction,
     not a training run whose failure could be blamed on the optimiser.      */
  function fit(T, n, xs) {
    const knots = DL.linspace(0, 1, n + 2).slice(1, n + 1);
    const Phi = xs.map(x => [1, x].concat(knots.map(k => Math.max(0, x - k))));
    const m = Phi[0].length;
    const A = DL.zeros2(m, m), b = new Array(m).fill(0);
    xs.forEach((x, i) => {
      const p = Phi[i], y = T.f(x);
      for (let a = 0; a < m; a++) { b[a] += p[a] * y; for (let c = 0; c < m; c++) A[a][c] += p[a] * p[c]; }
    });
    for (let a = 0; a < m; a++) A[a][a] += 1e-8;
    /* Gaussian elimination with partial pivoting */
    const M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < m; c++) {
      let piv = c;
      for (let r2 = c + 1; r2 < m; r2++) if (Math.abs(M[r2][c]) > Math.abs(M[piv][c])) piv = r2;
      const t = M[c]; M[c] = M[piv]; M[piv] = t;
      const pv = M[c][c] || 1e-12;
      for (let j = c; j <= m; j++) M[c][j] /= pv;
      for (let r2 = 0; r2 < m; r2++) if (r2 !== c) { const f = M[r2][c]; for (let j = c; j <= m; j++) M[r2][j] -= f * M[c][j]; }
    }
    const w = M.map(r => r[m]);
    const g = x => w[0] + w[1] * x + knots.reduce((s, k, i) => s + w[2 + i] * Math.max(0, x - k), 0);
    return { g: g, knots: knots };
  }

  function draw() {
    const T = TARGETS[fE.value], n = +hE.value, ext = eE.checked;
    document.getElementById("ua-hv").textContent = n;
    const grid = DL.linspace(0, 1, 801);
    const F0 = fit(T, n, grid);
    let sup = 0, at = 0;
    grid.forEach(x => { const e = Math.abs(F0.g(x) - T.f(x)); if (e > sup) { sup = e; at = x; } });

    const F = DL.frame(svg, W, H, { l: 46, r: 16, t: 26, b: 38 });
    const G = F.g, pw = 420, ph = H - 26 - 38;
    const xlo = ext ? -0.55 : 0, xhi = ext ? 1.55 : 1;
    const xs = d3.scaleLinear().domain([xlo, xhi]).range([0, pw]);
    const drawGrid = DL.linspace(xlo, xhi, 1000);
    const vals = drawGrid.map(x => T.f(x)).concat(drawGrid.map(x => F0.g(x)));
    const lo = Math.max(-6, d3.min(vals)), hi = Math.min(6, d3.max(vals));
    const ys = d3.scaleLinear().domain([lo - 0.15, hi + 0.15]).range([ph, 0]);
    DL.gridY(G, ys, pw, 5);
    if (ext) {
      G.append("rect").attr("x", xs(0)).attr("y", 0).attr("width", xs(1) - xs(0)).attr("height", ph)
        .attr("fill", DC.accent).attr("fill-opacity", 0.06);
      G.append("text").attr("x", (xs(0) + xs(1)) / 2).attr("y", 12).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", DC.accent).text("the interval the theorem covers");
    }
    G.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`).call(d3.axisBottom(xs).ticks(7));
    G.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(6));
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(T.label + " and a " + n + "-unit one-hidden-layer ReLU fit");
    const clip = v => DL.clamp(v, ys.domain()[0], ys.domain()[1]);
    if (T.cont) DL.curve(G, drawGrid.map(x => [xs(x), ys(clip(T.f(x)))]), { stroke: DC.a2, w: 2.4 });
    else {
      DL.curve(G, drawGrid.filter(x => x < 0.5).map(x => [xs(x), ys(clip(T.f(x)))]), { stroke: DC.a2, w: 2.4 });
      DL.curve(G, drawGrid.filter(x => x >= 0.5).map(x => [xs(x), ys(clip(T.f(x)))]), { stroke: DC.a2, w: 2.4 });
    }
    DL.curve(G, drawGrid.map(x => [xs(x), ys(clip(F0.g(x)))]), { stroke: DC.accent, w: 1.8 });
    F0.knots.forEach(k => G.append("line").attr("x1", xs(k)).attr("x2", xs(k))
      .attr("y1", ph).attr("y2", ph - 7).attr("stroke", DC.accent).attr("stroke-width", 1.2));
    G.append("line").attr("x1", xs(at)).attr("x2", xs(at)).attr("y1", ys(clip(T.f(at))))
      .attr("y2", ys(clip(F0.g(at)))).attr("stroke", DC.bad).attr("stroke-width", 2.2);
    DL.legend(G, [{ color: DC.a2, label: "target" }, { color: DC.accent, label: "network (kinks ticked below)" },
    { color: DC.bad, label: "the worst error" }], 8, ph - 40, { gap: 14 });

    /* right: sup-error against width */
    const x1 = pw + 46, cw = 240;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("worst error on [0, 1] against width");
    const g2 = G.append("g").attr("transform", `translate(${x1},0)`);
    const ns = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 44, 60];
    const errs = ns.map(k => {
      const ff = fit(T, k, grid);
      let e = 0; grid.forEach(x => { e = Math.max(e, Math.abs(ff.g(x) - T.f(x))); });
      return e;
    });
    const nsc = d3.scaleLog().domain([1, 62]).range([0, cw]);
    const esc = d3.scaleLog().domain([Math.max(1e-6, d3.min(errs) * 0.7), d3.max(errs) * 1.4]).range([ph - 40, 0]);
    DL.gridY(g2, esc, cw, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${ph - 40})`).call(d3.axisBottom(nsc).ticks(4, "~g"));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(esc).ticks(5, "~e"));
    DL.curve(g2, ns.map((k, i) => [nsc(k), esc(errs[i])]), { stroke: DC.accent, w: 2 });
    ns.forEach((k, i) => g2.append("circle").attr("cx", nsc(k)).attr("cy", esc(errs[i])).attr("r", 2.6).attr("fill", DC.accent));
    g2.append("circle").attr("cx", nsc(DL.clamp(n, 1, 62))).attr("cy", esc(DL.clamp(sup, esc.domain()[0], esc.domain()[1])))
      .attr("r", 6).attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 1.8);
    g2.append("text").attr("x", cw).attr("y", ph - 12).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("hidden units");
    if (!T.cont) {
      const floor0 = 0.7;
      g2.append("line").attr("x1", 0).attr("x2", cw).attr("y1", esc(floor0)).attr("y2", esc(floor0))
        .attr("stroke", DC.bad).attr("stroke-dasharray", "4 3");
      g2.append("text").attr("x", 2).attr("y", esc(floor0) - 5).attr("font-size", 9.5).attr("fill", DC.bad)
        .text("half the jump — an unbeatable floor");
    }

    let outErr = 0;
    if (ext) DL.linspace(1.05, 1.55, 60).forEach(x => { outErr = Math.max(outErr, Math.abs(F0.g(x) - T.f(x))); });

    d3.select("#ua-readout").html(
      "<b>" + n + "</b> hidden units give <b>" + (n + 1) + "</b> linear pieces and a worst error on [0, 1] of <b>" +
      DL.fmtE(sup, 3) + "</b> (at x = " + DL.fmt(at, 3) + "). " +
      (T.cont
        ? "Raising the width drives that number down without limit — which is the theorem, demonstrated. "
        : "<b>This target is discontinuous</b>, so the theorem's hypothesis fails and the error curve flattens at half the jump size, " +
        DL.fmt(0.7, 2) + ", no matter how many units are added. A continuous function cannot approximate a jump uniformly. ") +
      (ext
        ? "Outside the interval the worst error is <b>" + DL.fmtE(outErr, 3) +
        "</b> and growing — a rectified network is affine beyond its outermost kink, so it extrapolates as a straight line. The theorem promised nothing there, and delivers nothing."
        : "Tick the extrapolation box to see what happens outside the interval the theorem covers."));
  }
  [fE, hE, eE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 17 · #regions-svg — deep vs shallow, counted ═════════════ */
(function () {
  const svg = d3.select("#regions-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const dE = document.getElementById("rg-d"), nE = document.getElementById("rg-n"),
    LE = document.getElementById("rg-L"), mE = document.getElementById("rg-mode");

  function draw() {
    const d = +dE.value, n = +nE.value, Lmax = +LE.value, mode = mE.value;
    document.getElementById("rg-dv").textContent = d;
    document.getElementById("rg-nv").textContent = n;
    document.getElementById("rg-Lv").textContent = Lmax;

    const rows = [];
    for (let L = 1; L <= Lmax; L++) {
      const widths = new Array(L).fill(n);
      const deep = DL.regionsDeepLB(d, widths);
      const units = n * L;
      /* the fair shallow comparison: same total units, or same parameters */
      let sn;
      if (mode === "units") sn = units;
      else {
        /* deep params: d·n + n + (L−1)(n² + n); solve d·m + m = that for m */
        const pDeep = DL.params([d].concat(widths).concat([1]));
        sn = Math.max(1, Math.floor((pDeep - 1) / (d + 2)));
      }
      rows.push({ L: L, deep: deep, shallow: DL.regionsShallow(d, sn), sn: sn, units: units });
    }

    const F = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 40 });
    const G = F.g, pw = 380, ph = H - 26 - 40;
    const hi = d3.max(rows, r => Math.max(r.deep, r.shallow));
    const xs = d3.scaleLinear().domain([0.7, Lmax + 0.3]).range([0, pw]);
    const ys = d3.scaleLog().domain([1, hi * 2.5]).range([ph, 0]);
    DL.gridY(G, ys, pw, 6);
    G.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`)
      .call(d3.axisBottom(xs).ticks(Math.min(Lmax, 8)).tickFormat(d3.format("d")));
    G.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(6, "~e"));
    G.append("text").attr("x", pw).attr("y", ph + 32).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("hidden layers L");
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("linear regions, d = " + d + ", width " + n + " per layer");
    DL.curve(G, rows.map(r => [xs(r.L), ys(Math.max(1, r.deep))]), { stroke: DC.a2, w: 2.4 });
    DL.curve(G, rows.map(r => [xs(r.L), ys(Math.max(1, r.shallow))]), { stroke: DC.accent, w: 2, dash: "5 3" });
    rows.forEach(r => {
      G.append("circle").attr("cx", xs(r.L)).attr("cy", ys(Math.max(1, r.deep))).attr("r", 3).attr("fill", DC.a2);
      G.append("circle").attr("cx", xs(r.L)).attr("cy", ys(Math.max(1, r.shallow))).attr("r", 2.8).attr("fill", DC.accent);
    });
    const cross = rows.find(r => r.deep > r.shallow);
    if (cross) {
      G.append("line").attr("x1", xs(cross.L)).attr("x2", xs(cross.L)).attr("y1", 0).attr("y2", ph)
        .attr("stroke", DC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "4 3");
      G.append("text").attr("x", xs(cross.L) + 5).attr("y", 14).attr("font-size", 9.5).attr("fill", DC.good)
        .text("depth overtakes here (L = " + cross.L + ")");
    }
    DL.legend(G, [{ color: DC.a2, label: "deep, lower bound Ω(·)" },
    { color: DC.accent, label: "one hidden layer, exact max", dash: "5 3" }], 8, ph - 26, { gap: 14 });

    /* the table */
    const x1 = pw + 44, cols = [0, 46, 118, 196];
    ["L", mode === "units" ? "units" : "params", "deep ≥", "shallow ="].forEach((h, j) =>
      G.append("text").attr("x", x1 + cols[j]).attr("y", 6).attr("font-size", 9.5).attr("fill", DC.muted).text(h));
    rows.forEach((r, i) => {
      const yy = 22 + i * 15, win = r.deep > r.shallow;
      [String(r.L), mode === "units" ? String(r.units) : String(r.sn),
      DL.big(r.deep), DL.big(r.shallow)]
        .forEach((c, j) => G.append("text").attr("x", x1 + cols[j]).attr("y", yy).attr("font-size", 10.5)
          .attr("font-family", "SF Mono, Menlo, monospace")
          .attr("fill", j === 2 ? (win ? DC.good : DC.muted) : DC.ink).text(c));
      G.append("text").attr("x", x1 + 262).attr("y", yy).attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", win ? DC.good : DC.bad)
        .text((r.deep >= r.shallow ? (r.deep / r.shallow).toPrecision(3) + "×" : "1/" + (r.shallow / r.deep).toPrecision(3)));
    });
    const last = rows[rows.length - 1];
    /* how wide must ONE hidden layer be to match the deep count? */
    let loN = 1, hiN = 1;
    while (DL.regionsShallow(d, hiN) < last.deep && hiN < 1e7) hiN *= 2;
    while (loN < hiN) { const mid = Math.floor((loN + hiN) / 2); if (DL.regionsShallow(d, mid) < last.deep) loN = mid + 1; else hiN = mid; }
    const need = loN;
    G.append("text").attr("x", x1).attr("y", 22 + rows.length * 15 + 16).attr("font-size", 10)
      .attr("fill", DC.muted).text("to match L = " + last.L + " with ONE hidden layer:");
    G.append("text").attr("x", x1).attr("y", 22 + rows.length * 15 + 31).attr("font-size", 11)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.a2).attr("font-weight", 600)
      .text(need >= 1e7 ? "more than 10 000 000 units" : DL.commas(need) + " units");

    d3.select("#regions-readout").html(
      "d = " + d + ", " + n + " units per layer. At L = " + last.L +
      " the deep construction guarantees at least <b>" + DL.commas(last.deep) +
      "</b> linear regions from <b>" + last.units + "</b> hidden units; one hidden layer with " +
      (mode === "units" ? "the same " + last.units + " units" : "the same parameter budget, " + last.sn + " units") +
      " tops out at exactly <b>" + DL.commas(last.shallow) + "</b>. " +
      (cross
        ? (cross.L > 1
          ? "Depth first overtakes at L = <b>" + cross.L + "</b>; below that the shallow maximum is the larger of the two, because the exponential has not started yet. "
          : "Depth is ahead from L = <b>1</b> at this width. ")
        : "Depth has <b>not</b> overtaken anywhere in this depth range — the exponential needs more layers than are shown. ") +
      "To match the deep count with a single hidden layer would take " +
      (need >= 1e7 ? "<b>more than ten million</b>" : "<b>" + DL.commas(need) + "</b>") +
      " units. Remember the deep number is a <b>lower bound on the maximum over parameters</b> — an existence claim, not a property of a trained network.");
  }
  [dE, nE, LE, mE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 18 · #sawtooth-svg — depth against width, explicitly ═════ */
(function () {
  const svg = d3.select("#sawtooth-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const LE = document.getElementById("sw-L"), nE = document.getElementById("sw-n"),
    mE = document.getElementById("sw-match");

  const tent = x => Math.abs(2 * x - 1);
  function deepSaw(x, L) { let v = x; for (let i = 0; i < L; i++) v = tent(v); return v; }

  /* the best a one-hidden-layer net can do, kinks placed evenly, output layer
     fitted by least squares — the same routine figure 16 uses               */
  function shallowFit(L, n, grid) {
    const knots = DL.linspace(0, 1, n + 2).slice(1, n + 1);
    const m = n + 2;
    const A = DL.zeros2(m, m), b = new Array(m).fill(0);
    grid.forEach(x => {
      const p = [1, x].concat(knots.map(k => Math.max(0, x - k))), y = deepSaw(x, L);
      for (let a = 0; a < m; a++) { b[a] += p[a] * y; for (let c = 0; c < m; c++) A[a][c] += p[a] * p[c]; }
    });
    for (let a = 0; a < m; a++) A[a][a] += 1e-9;
    const M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < m; c++) {
      let piv = c;
      for (let r2 = c + 1; r2 < m; r2++) if (Math.abs(M[r2][c]) > Math.abs(M[piv][c])) piv = r2;
      const t = M[c]; M[c] = M[piv]; M[piv] = t;
      const pv = M[c][c] || 1e-12;
      for (let j = c; j <= m; j++) M[c][j] /= pv;
      for (let r2 = 0; r2 < m; r2++) if (r2 !== c) { const f = M[r2][c]; for (let j = c; j <= m; j++) M[r2][j] -= f * M[c][j]; }
    }
    const w = M.map(r => r[m]);
    return x => w[0] + w[1] * x + knots.reduce((s, k, i) => s + w[2 + i] * Math.max(0, x - k), 0);
  }

  function draw() {
    const L = +LE.value, match = mE.checked;
    const need = Math.pow(2, L) - 1;
    let n = match ? Math.min(80, need) : +nE.value;
    nE.disabled = match;
    if (match) nE.value = n;
    document.getElementById("sw-Lv").textContent = L;
    document.getElementById("sw-nv").textContent = n;

    const grid = DL.linspace(0, 1, 1601);
    const g = shallowFit(L, n, grid);
    let sup = 0;
    grid.forEach(x => { sup = Math.max(sup, Math.abs(g(x) - deepSaw(x, L))); });

    const F = DL.frame(svg, W, H, { l: 46, r: 16, t: 26, b: 34 });
    const G = F.g, pw = 320, ph1 = 150;
    const xs = d3.scaleLinear().domain([0, 1]).range([0, pw]);
    const ys = d3.scaleLinear().domain([-0.25, 1.25]).range([ph1, 0]);
    function panel(x0, fn, title, sub, col) {
      G.append("text").attr("x", x0).attr("y", -12).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text(title);
      G.append("text").attr("x", x0).attr("y", -1).attr("font-size", 9.5).attr("fill", DC.muted).text(sub);
      const gg = G.append("g").attr("transform", `translate(${x0},6)`);
      DL.gridY(gg, ys, pw, 4);
      gg.append("g").attr("class", "axis").attr("transform", `translate(0,${ph1})`).call(d3.axisBottom(xs).ticks(5));
      gg.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(4));
      DL.curve(gg, grid.map(x => [xs(x), ys(DL.clamp(fn(x), -0.25, 1.25))]), { stroke: col, w: 1.8 });
      return gg;
    }
    panel(6, x => deepSaw(x, L), "deep: " + L + " tent maps composed",
      2 * L + " rectified units, " + DL.commas(Math.pow(2, L)) + " teeth", DC.a2);
    panel(6 + pw + 48, g, "shallow: one hidden layer",
      n + " rectified units, at most " + (n + 1) + " pieces", sup < 1e-3 ? DC.good : DC.bad);

    /* lower: units needed against L */
    const y0 = ph1 + 62, ph2 = H - 26 - 34 - y0;
    G.append("text").attr("x", 6).attr("y", y0 - 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("units required, deep against shallow");
    const g3 = G.append("g").attr("transform", `translate(6,${y0})`);
    const Ls = d3.range(1, 13);
    const lx = d3.scaleLinear().domain([1, 12]).range([0, 640]);
    const ly = d3.scaleLog().domain([1, Math.pow(2, 12)]).range([ph2, 0]);
    DL.gridY(g3, ly, 640, 5);
    g3.append("g").attr("class", "axis").attr("transform", `translate(0,${ph2})`).call(d3.axisBottom(lx).ticks(6).tickFormat(d3.format("d")));
    g3.append("g").attr("class", "axis").call(d3.axisLeft(ly).ticks(5, "~s"));
    DL.curve(g3, Ls.map(l => [lx(l), ly(2 * l)]), { stroke: DC.a2, w: 2.2 });
    DL.curve(g3, Ls.map(l => [lx(l), ly(Math.pow(2, l) - 1)]), { stroke: DC.accent, w: 2, dash: "5 3" });
    g3.append("circle").attr("cx", lx(L)).attr("cy", ly(2 * L)).attr("r", 4).attr("fill", DC.a2);
    g3.append("circle").attr("cx", lx(L)).attr("cy", ly(need)).attr("r", 4).attr("fill", DC.accent);
    DL.legend(g3, [{ color: DC.a2, label: "deep: 2L units" },
    { color: DC.accent, label: "shallow: 2ᴸ − 1 units", dash: "5 3" }], 500, 14, { gap: 14 });
    g3.append("text").attr("x", 640).attr("y", ph2 + 30).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("compositions L");

    d3.select("#sawtooth-readout").html(
      "L = " + L + ": the deep construction uses <b>" + 2 * L + "</b> units and produces <b>" +
      DL.commas(Math.pow(2, L)) + "</b> teeth. A single hidden layer needs at least <b>" +
      DL.commas(need) + "</b> units to have that many pieces — a factor of <b>" +
      DL.fmt(need / (2 * L), 1) + "</b>. With the " + n + " units it currently has, its best least-squares fit " +
      (sup < 1e-3
        ? "reproduces the target to <b>" + DL.fmtE(sup, 2) +
        "</b>: with 2ᴸ − 1 evenly spaced kinks its pieces land exactly on the target's, so the fit is exact in principle and the residual shown is only the conditioning of the least-squares solve. It has exactly enough pieces and not one to spare: untick the matching box and remove a single unit, and the worst error jumps to about 0.51 — half the target\u2019s full range, and five orders of magnitude worse."
        : "misses by <b>" + DL.fmt(sup, 4) + "</b> at the worst point — it does not have enough pieces, and no setting of its " +
        n + " output weights can fix that.") +
      " At L = 20 the two counts are 40 and 1 048 575.");
  }
  [LE, nE, mE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 19 · #sizing-svg — shapes under a budget ═════════════════ */
(function () {
  const svg = d3.select("#sizing-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const sE = document.getElementById("sz-shape"), LE = document.getElementById("sz-L"),
    PE = document.getElementById("sz-P"), dE = document.getElementById("sz-d0");
  const DOUT = 10;

  /* relative widths for each shape; scaled by a single factor found by
     bisection so the achieved parameter count matches the budget            */
  function profile(shape, L) {
    const t = DL.linspace(0, 1, L);
    if (shape === "uniform") return t.map(() => 1);
    if (shape === "funnel") return t.map(v => 1.6 - 1.2 * v);
    if (shape === "expand") return t.map(v => 0.4 + 1.2 * v);
    return t.map(v => (L > 2 && Math.abs(v - 0.5) < 0.5 / (L - 1) + 1e-9) ? 0.08 : 1.3);
  }
  function widthsFor(shape, L, d0, budget) {
    const rel = profile(shape, L);
    let lo = 1, hi = 4000;
    for (let it = 0; it < 60; it++) {
      const mid = (lo + hi) / 2;
      const w = rel.map(r => Math.max(1, Math.round(r * mid)));
      const P = DL.params([d0].concat(w).concat([DOUT]));
      if (P < budget) lo = mid; else hi = mid;
    }
    return rel.map(r => Math.max(1, Math.round(r * lo)));
  }

  function draw() {
    const shape = sE.value, L = +LE.value, budget = +PE.value, d0 = +dE.value;
    document.getElementById("sz-Lv").textContent = L;
    document.getElementById("sz-Pv").textContent = DL.big(budget);
    document.getElementById("sz-d0v").textContent = d0;

    const w = widthsFor(shape, L, d0, budget);
    const sizes = [d0].concat(w).concat([DOUT]);
    const P = DL.params(sizes);
    const per = [];
    for (let l = 0; l + 1 < sizes.length; l++) per.push(sizes[l] * sizes[l + 1] + sizes[l + 1]);
    const narrow = Math.min.apply(null, w);
    /* the end-to-end linear map can have rank at most the narrowest dimension
       ANYWHERE in the chain, endpoints included — the interesting comparison is
       the narrowest HIDDEN layer against the output dimension */
    const rank = Math.min(d0, narrow, DOUT);

    const F = DL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 40 });
    const G = F.g, pw = 300, ph = 130;
    const bx = d3.scaleBand().domain(d3.range(w.length)).range([0, pw]).padding(0.24);
    const by = d3.scaleLinear().domain([0, Math.max.apply(null, w) * 1.15]).range([ph, 0]);
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("hidden widths under a " + DL.big(budget) + "-parameter budget");
    G.append("g").attr("class", "axis").call(d3.axisLeft(by).ticks(4));
    w.forEach((v, i) => {
      G.append("rect").attr("x", bx(i)).attr("y", by(v)).attr("width", bx.bandwidth())
        .attr("height", ph - by(v))
        .attr("fill", v === narrow && shape === "bottleneck" ? DC.bad : DC.accent).attr("fill-opacity", 0.75);
      G.append("text").attr("x", bx(i) + bx.bandwidth() / 2).attr("y", by(v) - 4).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(v);
    });
    G.append("text").attr("x", 0).attr("y", ph + 16).attr("font-size", 10).attr("fill", DC.muted)
      .text("achieved " + DL.commas(P) + " parameters (" + (100 * P / budget).toFixed(1) + "% of the budget)");

    /* per-layer parameter counts */
    const y1 = ph + 52;
    G.append("text").attr("x", 0).attr("y", y1 - 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("parameters in each layer — not uniform even when the widths are");
    const g2 = G.append("g").attr("transform", `translate(0,${y1})`);
    const ph2 = H - 26 - 40 - y1;
    const px = d3.scaleBand().domain(d3.range(per.length)).range([0, pw]).padding(0.24);
    const py = d3.scaleLinear().domain([0, Math.max.apply(null, per) * 1.2]).range([ph2, 0]);
    g2.append("g").attr("class", "axis").call(d3.axisLeft(py).ticks(4, "~s"));
    per.forEach((v, i) => {
      g2.append("rect").attr("x", px(i)).attr("y", py(v)).attr("width", px.bandwidth())
        .attr("height", ph2 - py(v)).attr("fill", DC.a2).attr("fill-opacity", 0.75);
      g2.append("text").attr("x", px(i) + px.bandwidth() / 2).attr("y", py(v) - 4).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.big(v));
    });
    g2.append("text").attr("x", 0).attr("y", ph2 + 16).attr("font-size", 10).attr("fill", DC.muted)
      .text("first layer " + (100 * per[0] / P).toFixed(1) + "% of the model");

    /* right: the bottleneck verdict */
    const x1 = pw + 56;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("what the shape costs");
    const k = DL.kv(G, x1, 16, { keyW: 176, lead: 17, size: 11 });
    k("architecture", sizes.join("→"), DC.ink);
    k("parameters", DL.commas(P), DC.a2, true);
    k("narrowest hidden layer", String(narrow), narrow < DOUT ? DC.bad : DC.ink, true);
    k("max rank of the end-to-end map", String(rank), rank < DOUT ? DC.bad : DC.good, true);
    k("output dimension", String(DOUT), DC.muted);
    k("MACs / example", DL.big(DL.macs(sizes)), DC.ink);
    const verdict = narrow < DOUT
      ? ["The narrowest hidden layer is smaller than the",
        "output. Whatever the layers below extracted, only",
        narrow + " numbers pass through, so the network cannot",
        "even represent an arbitrary map to " + DOUT + " outputs —",
        "no matter how large the other layers are."]
      : (shape === "bottleneck"
        ? ["A deliberate bottleneck: " + narrow + " numbers carry",
          "everything past the middle. Sometimes exactly",
          "what is wanted (a learned compression), often",
          "an accident that costs accuracy silently."]
        : ["No bottleneck: every hidden layer is at least as",
          "wide as the output, so nothing is structurally",
          "forced to be discarded. Over-wide is a much safer",
          "error than over-narrow — regularisation can shrink",
          "a layer that is too big; nothing recovers a",
          "dimension that was thrown away."]);
    verdict.forEach((line, i) => G.append("text").attr("x", x1).attr("y", 150 + i * 14)
      .attr("font-size", 10.5).attr("fill", narrow < DOUT ? DC.bad : DC.muted).text(line));

    d3.select("#sizing-readout").html(
      "<b>" + shape + "</b>, depth " + L + ": widths <b>" + sizes.join(" → ") + "</b> at <b>" +
      DL.commas(P) + "</b> parameters. The first layer alone is <b>" + (100 * per[0] / P).toFixed(1) +
      "%</b> of the model and the narrowest hidden layer is <b>" + narrow + "</b>. " +
      "The end-to-end map's rank is capped by the narrowest dimension in the chain, min(" +
      d0 + ", " + narrow + ", " + DOUT + ") = <b>" + rank + "</b>, " +
      (narrow < DOUT
        ? "which is <b>less than the " + DOUT + " outputs</b>. This architecture is structurally incapable of an arbitrary input-to-output map, and adding parameters elsewhere does not help."
        : "which is enough for the " + DOUT + " outputs. Every shape here spends the same budget; they differ in where they spend it and in what they can and cannot pass through."));
  }
  [sE, LE, PE, dE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 20 · #bias-svg — permutation equivariance ═════════════════
   The dataset carries TWO cues on purpose, because the point needs both:
     · a faint SIGN-CODED marker on a fixed 2×2 corner, identical in magnitude
       for the two classes. A linear model on the raw pixels can read its sign;
       a magnitude-based local feature cannot.
     · a COHERENT 5×5 patch at a random location in the positive class against
       the SAME 25 values scattered anywhere in the negative class. The two
       classes therefore have identical pixel-value distributions and differ
       only in spatial ARRANGEMENT, which is what a local feature reads.
   Shuffling destroys the second cue and leaves the first untouched, which is
   exactly the asymmetry the section is about.                              */
(function () {
  const svg = d3.select("#bias-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, S = 12, D = S * S, NIM = 300;
  const sE = document.getElementById("bi-s"), mE = document.getElementById("bi-m"),
    pE = document.getElementById("bi-p"), seE = document.getElementById("bi-seed");

  function shape(kind) {
    const v = [];
    for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) {
      let u = 0;
      if (kind === "edge") u = (j >= 0 ? 1 : -1);
      else if (kind === "blob") u = 2 * Math.exp(-(i * i + j * j) / 2.6) - 0.85;
      else u = ((i === 0 || j === 0) ? 1 : -0.5);
      v.push(1.6 * u);
    }
    return v;
  }
  function makeSet(kind, seed, marker) {
    const r = DL.rng(seed), X = [], y = [];
    for (let k = 0; k < NIM; k++) {
      const img = new Array(D).fill(0);
      for (let i = 0; i < D; i++) img[i] = 0.35 * DL.randn(r);
      const has = k < NIM / 2, vals = shape(kind);
      if (has) {
        const cy = 2 + Math.floor(r() * (S - 4)), cx = 2 + Math.floor(r() * (S - 4));
        let t = 0;
        for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) img[(cy + i) * S + (cx + j)] += vals[t++];
      } else {
        const pos = d3.range(D); DL.shuffle(pos, r);
        for (let t = 0; t < 25; t++) img[pos[t]] += vals[t];
      }
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) img[i * S + j] += (has ? marker : -marker);
      X.push(img); y.push(has ? 1 : 0);
    }
    return { X: X, y: y };
  }
  /* a PARTIAL random permutation: pick frac·D positions and permute those.
     At frac = 1 this is a full uniform permutation; repeated random
     transpositions would leave far too many positions fixed. */
  function permutation(frac, seed) {
    const p = d3.range(D), r = DL.rng(seed);
    const idx = d3.range(D); DL.shuffle(idx, r);
    const sub = idx.slice(0, Math.round(frac * D));
    const vals = sub.map(i => p[i]); DL.shuffle(vals, r);
    sub.forEach((i, k) => { p[i] = vals[k]; });
    return p;
  }
  const applyPerm = (img, p) => p.map(i => img[i]);
  const KV = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], KH = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
  function localFeat(img) {
    let mxS = 0, mxL = 0, rough = 0;
    for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
      let sv = 0, sh = 0;
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
        const v = img[(y + i) * S + (x + j)];
        sv += v * KV[i + 1][j + 1]; sh += v * KH[i + 1][j + 1];
      }
      const l = img[y * S + x - 1] + img[y * S + x + 1] + img[(y - 1) * S + x] + img[(y + 1) * S + x] - 4 * img[y * S + x];
      mxS = Math.max(mxS, Math.hypot(sv, sh));
      mxL = Math.max(mxL, Math.abs(l));
      rough += l * l;
    }
    return [mxS, mxL, rough, Math.sqrt(rough)];
  }
  function score(feats, y, tr, te) {
    const Z = NN.standardise(feats);
    const m = NN.logistic(tr.map(i => Z[i]), tr.map(i => y[i]), 600, 0.6, 2e-3);
    return NN.accuracy(m, te.map(i => Z[i]), te.map(i => y[i]));
  }

  function draw() {
    const frac = +sE.value / 100, mode = mE.value, kind = pE.value, seed = +seE.value;
    document.getElementById("bi-sv").textContent = sE.value + "%";
    document.getElementById("bi-seedv").textContent = seed;

    const DS = makeSet(kind, seed, 0.55);
    /* class is decided by the first/second half of the index and the split by
       parity, so both halves contain both classes — the opposite arrangement
       silently produces a single-class training set. */
    const idx = d3.range(NIM), tr = idx.filter(i => i % 2 === 0), te = idx.filter(i => i % 2 === 1);
    const perm = permutation(frac, seed + 100);
    const Xp = DS.X.map(im => applyPerm(im, perm));

    /* A FIXED random dense layer W (D × 8), and the SAME layer with its ROWS
       permuted — W′ = PᵀW in the row convention of §09 and of dl-viz.js, since
       a row of W is indexed by input coordinate. Concretely, applyPerm gives
       (xP)_j = x[p[j]] and Wperm[j] = W[p[j]], so (xP)·W′ = Σ_j x[p[j]]·W[p[j]]
       = Σ_i x_i·W_i = x·W, exactly. That identity is what the max-difference
       readout measures, and it comes out at round-off. */
    const r = DL.rng(2024), Wd = DL.zeros2(D, 8);
    for (let i = 0; i < D; i++) for (let j = 0; j < 8; j++) Wd[i][j] = DL.randn(r) / Math.sqrt(D);
    const Wperm = perm.map(i => Wd[i]);
    const im0 = DS.X[0], im1 = Xp[0];
    const dense0 = DL.vecmat(im0, Wd), dense1 = DL.vecmat(im1, Wperm);
    let dmax = 0; dense0.forEach((v, i) => { dmax = Math.max(dmax, Math.abs(v - dense1[i])); });
    const loc0 = localFeat(im0), loc1 = localFeat(im1);
    let lrel = 0; loc0.forEach((v, i) => { lrel = Math.max(lrel, Math.abs(v - loc1[i]) / (Math.abs(v) + 1e-9)); });

    const fracs = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const curves = fracs.map(f => {
      const P = permutation(f, seed + 100);
      const XX = DS.X.map(im => applyPerm(im, P));
      return { f: f, dense: score(XX, DS.y, tr, te), loc: score(XX.map(localFeat), DS.y, tr, te) };
    });

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 12 });
    const G = F.g, px = 9;
    function tile(x0, y0, img, title) {
      G.append("text").attr("x", x0).attr("y", y0 - 6).attr("font-size", 10.5).attr("fill", DC.ink).text(title);
      const lo = d3.min(img), hi = d3.max(img);
      DL.cells(G, x0, y0, px, S, S, (i, j) => {
        const t = (img[j * S + i] - lo) / (hi - lo + 1e-9);
        return d3.interpolateRgb(DC.bg, DC.ink)(t);
      });
      G.append("rect").attr("x", x0).attr("y", y0).attr("width", S * px).attr("height", S * px)
        .attr("fill", "none").attr("stroke", DC.line);
    }
    tile(6, 16, im0, "a positive image: a coherent patch");
    tile(6 + S * px + 40, 16, im1, sE.value + "% of positions permuted");
    DL.arrow(G, 6 + S * px + 10, 16 + S * px / 2, 6 + S * px + 34, 16 + S * px / 2,
      { color: DC.muted, w: 1.4, head: 5 });
    G.append("text").attr("x", 6).attr("y", 16 + S * px + 16).attr("font-size", 9.5).attr("fill", DC.muted)
      .text("identical pixel VALUES, different arrangement");

    const by0 = 16 + S * px + 46;
    function bars(x0, y0, a, b, title, same, wid) {
      G.append("text").attr("x", x0).attr("y", y0 - 6).attr("font-size", 10.5)
        .attr("fill", same ? DC.good : DC.bad).attr("font-weight", 600).text(title);
      const lo = Math.min(d3.min(a), d3.min(b), 0), hi = Math.max(d3.max(a), d3.max(b));
      const ys = d3.scaleLinear().domain([lo, hi]).range([54, 0]);
      a.forEach((v, i) => {
        const bw = wid / a.length;
        G.append("rect").attr("x", x0 + i * bw).attr("y", y0 + Math.min(ys(v), ys(0)))
          .attr("width", bw * 0.42).attr("height", Math.abs(ys(v) - ys(0)) + 0.5)
          .attr("fill", DC.accent).attr("fill-opacity", 0.85);
        G.append("rect").attr("x", x0 + i * bw + bw * 0.48).attr("y", y0 + Math.min(ys(b[i]), ys(0)))
          .attr("width", bw * 0.42).attr("height", Math.abs(ys(b[i]) - ys(0)) + 0.5)
          .attr("fill", same ? DC.good : DC.bad).attr("fill-opacity", 0.85);
      });
      G.append("line").attr("x1", x0).attr("x2", x0 + wid).attr("y1", y0 + ys(0)).attr("y2", y0 + ys(0))
        .attr("stroke", DC.line);
    }
    if (mode !== "conv") bars(6, by0, dense0, dense1,
      "dense layer, rows permuted to match — max diff " + DL.fmtE(dmax, 1), dmax < 1e-9, 150);
    if (mode !== "mlp") bars(196, by0, loc0.map(v => Math.log10(1 + Math.abs(v))),
      loc1.map(v => Math.log10(1 + Math.abs(v))),
      "local features (log scale) — max rel. diff " + DL.fmt(lrel, 2), lrel < 1e-9, 118);

    const x1 = 400, cw = 320, ch = 250;
    G.append("text").attr("x", x1).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("held-out detection accuracy against shuffle");
    const g2 = G.append("g").attr("transform", `translate(${x1},22)`);
    const fs = d3.scaleLinear().domain([0, 1]).range([0, cw]);
    const as = d3.scaleLinear().domain([0.4, 1]).range([ch, 0]);
    DL.gridY(g2, as, cw, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${ch})`)
      .call(d3.axisBottom(fs).ticks(5, ".0%"));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(as).ticks(5, ".0%"));
    g2.append("text").attr("x", cw).attr("y", ch + 32).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("fraction of pixel positions permuted");
    g2.append("line").attr("x1", 0).attr("x2", cw).attr("y1", as(0.5)).attr("y2", as(0.5))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    g2.append("text").attr("x", 2).attr("y", as(0.5) - 4).attr("font-size", 9).attr("fill", DC.muted).text("chance");
    if (mode !== "conv") {
      DL.curve(g2, curves.map(c => [fs(c.f), as(DL.clamp(c.dense, 0.4, 1))]), { stroke: DC.accent, w: 2.4 });
      curves.forEach(c => g2.append("circle").attr("cx", fs(c.f)).attr("cy", as(DL.clamp(c.dense, 0.4, 1))).attr("r", 3).attr("fill", DC.accent));
    }
    if (mode !== "mlp") {
      DL.curve(g2, curves.map(c => [fs(c.f), as(DL.clamp(c.loc, 0.4, 1))]), { stroke: DC.a2, w: 2.4, dash: "5 3" });
      curves.forEach(c => g2.append("circle").attr("cx", fs(c.f)).attr("cy", as(DL.clamp(c.loc, 0.4, 1))).attr("r", 3).attr("fill", DC.a2));
    }
    g2.append("line").attr("x1", fs(frac)).attr("x2", fs(frac)).attr("y1", 0).attr("y2", ch)
      .attr("stroke", DC.ink).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    DL.legend(g2, [{ color: DC.accent, label: "dense, on raw pixels" },
    { color: DC.a2, label: "local 3×3 features", dash: "5 3" }], 8, 14, { gap: 14 });

    const c0 = curves[0], c1 = curves[curves.length - 1];
    let spread = 0; curves.forEach(c => { spread = Math.max(spread, Math.abs(c.dense - c0.dense)); });
    d3.select("#bias-readout").html(
      "The dense layer's response to the permuted image, with its rows permuted to match, differs from its response to the original by <b>" +
      DL.fmtE(dmax, 2) + "</b> — floating-point round-off, exactly as the two-line proof says. The local features change by up to <b>" +
      (100 * lrel).toFixed(0) + "%</b>. Measured on held-out data across the whole sweep, the dense model varies by at most <b>" +
      DL.fmt(100 * spread, 1) + "</b> accuracy points (" + (100 * c0.dense).toFixed(1) + "% unshuffled, " +
      (100 * c1.dense).toFixed(1) + "% fully shuffled) while the local model falls from <b>" +
      (100 * c0.loc).toFixed(1) + "%</b> to <b>" + (100 * c1.loc).toFixed(1) + "%</b>, giving up <b>" +
      DL.fmt(100 * (c0.loc - c1.loc), 1) + "</b> points" +
      (c1.loc < 0.58 ? " and landing at chance" : "") +
      ". That drop is the measure of how much prior knowledge locality was worth: <b>being hurt by the shuffle is the evidence that the architecture knew something true</b>, and being unmoved by it is the evidence that it did not.");
  }
  [sE, mE, pE, seE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 22 · #ffn-svg — where a transformer layer's parameters live ═ */
(function () {
  const svg = d3.select("#ffn-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const dE = document.getElementById("ff-d"), rE = document.getElementById("ff-r"),
    LE = document.getElementById("ff-L"), kE = document.getElementById("ff-k");

  function budget(d, ratio, kind) {
    const attn = 4 * d * d;
    let dff = ratio * d, nMat = 2;
    if (kind === "gated") nMat = 3;
    if (kind === "gated23") { nMat = 3; dff = Math.round(ratio * d * 2 / 3); }
    const ffn = nMat * d * dff;
    return { attn: attn, ffn: ffn, dff: Math.round(dff), nMat: nMat, total: attn + ffn };
  }

  function draw() {
    const d = +dE.value, ratio = +rE.value, L = +LE.value, kind = kE.value;
    document.getElementById("ff-dv").textContent = d;
    document.getElementById("ff-rv").textContent = DL.fmt(ratio, 1);
    document.getElementById("ff-Lv").textContent = L;
    const B = budget(d, ratio, kind), plain = budget(d, ratio, "plain");

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 12 });
    const G = F.g, bw = 330;

    /* the stacked bar */
    G.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("one transformer layer, by parameter share");
    const segs = [
      { n: "W_Q", p: d * d, c: DC.accent }, { n: "W_K", p: d * d, c: DC.accent },
      { n: "W_V", p: d * d, c: DC.accent }, { n: "W_O", p: d * d, c: DC.accent }
    ];
    for (let i = 0; i < B.nMat; i++) segs.push({ n: ["W₁", "W₂", "V"][i], p: d * B.dff, c: DC.a2 });
    let x = 0;
    segs.forEach(s => {
      const wpx = bw * s.p / B.total;
      G.append("rect").attr("x", x).attr("y", 6).attr("width", Math.max(0.5, wpx)).attr("height", 34)
        .attr("fill", s.c).attr("fill-opacity", 0.7).attr("stroke", DC.bg);
      if (wpx > 26) G.append("text").attr("x", x + wpx / 2).attr("y", 27).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", "#0f1117").attr("font-weight", 600).text(s.n);
      x += wpx;
    });
    G.append("rect").attr("x", 0).attr("y", 6).attr("width", bw).attr("height", 34)
      .attr("fill", "none").attr("stroke", DC.line);
    G.append("text").attr("x", 0).attr("y", 56).attr("font-size", 10).attr("fill", DC.accent)
      .text("attention " + (100 * B.attn / B.total).toFixed(1) + "%");
    G.append("text").attr("x", bw).attr("y", 56).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.a2).text("FFN " + (100 * B.ffn / B.total).toFixed(1) + "%");

    /* the table */
    const ty = 92, cols = [0, 118, 236];
    ["matrix", "shape", "parameters"].forEach((h, j) => G.append("text").attr("x", cols[j]).attr("y", ty)
      .attr("font-size", 10).attr("fill", DC.muted).text(h));
    const rows = [["W_Q, W_K, W_V, W_O", "4 × (" + d + " × " + d + ")", 4 * d * d]];
    if (B.nMat === 2) rows.push(["W₁, W₂", "2 × (" + d + " × " + B.dff + ")", 2 * d * B.dff]);
    else rows.push(["W₁, V, W₂", "3 × (" + d + " × " + B.dff + ")", 3 * d * B.dff]);
    rows.push(["one layer", "", B.total]);
    rows.push([L + " layers", "", B.total * L]);
    rows.forEach((r, i) => {
      const yy = ty + 18 + i * 15, tot = i >= 2;
      [r[0], r[1], DL.commas(r[2])].forEach((c, j) => G.append("text").attr("x", cols[j]).attr("y", yy)
        .attr("font-size", 10.5).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", tot ? DC.a2 : DC.ink).attr("font-weight", tot ? 600 : 400).text(c));
    });
    const k = DL.kv(G, 0, ty + 18 + rows.length * 15 + 16, { keyW: 176, lead: 15, size: 10.5 });
    k("stack total", DL.big(B.total * L), DC.a2, true);
    k("weights at 2 bytes", (B.total * L * 2 / 1073741824).toFixed(2) + " GiB", DC.ink);
    k("weights at 4 bytes", (B.total * L * 4 / 1073741824).toFixed(2) + " GiB", DC.ink);
    k("d_ff", String(B.dff) + "  (= " + DL.fmt(B.dff / d, 2) + "d)", DC.ink);
    k("vs the plain 2-matrix FFN", (B.total / plain.total).toFixed(3) + "×",
      Math.abs(B.total / plain.total - 1) < 0.02 ? DC.good : DC.violet, true);

    /* right: the share curve against the ratio */
    const x1 = 430, cw = 300, ch = 250;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("FFN share against the expansion ratio");
    const g2 = G.append("g").attr("transform", `translate(${x1},6)`);
    const rs = d3.scaleLinear().domain([1, 8]).range([0, cw]);
    const sh = d3.scaleLinear().domain([0, 1]).range([ch, 0]);
    DL.gridY(g2, sh, cw, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(rs).ticks(8));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(sh).ticks(5, ".0%"));
    g2.append("text").attr("x", cw).attr("y", ch + 32).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("d_ff / d");
    ["plain", "gated", "gated23"].forEach((kind2, i) => {
      const pts = DL.linspace(1, 8, 120).map(rr => {
        const b2 = budget(d, rr, kind2);
        return [rs(rr), sh(b2.ffn / b2.total)];
      });
      DL.curve(g2, pts, { stroke: [DC.a2, DC.violet, DC.teal][i], w: kind2 === kind ? 2.6 : 1.2, dash: i ? "5 3" : null });
    });
    g2.append("line").attr("x1", 0).attr("x2", cw).attr("y1", sh(2 / 3)).attr("y2", sh(2 / 3))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    g2.append("text").attr("x", 2).attr("y", sh(2 / 3) - 5).attr("font-size", 9.5).attr("fill", DC.muted).text("⅔");
    g2.append("circle").attr("cx", rs(DL.clamp(ratio, 1, 8))).attr("cy", sh(B.ffn / B.total)).attr("r", 5)
      .attr("fill", "none").attr("stroke", DC.ink).attr("stroke-width", 2);
    DL.legend(g2, [{ color: DC.a2, label: "plain, 2 matrices" },
    { color: DC.violet, label: "gated, 3 matrices", dash: "5 3" },
    { color: DC.teal, label: "gated, width × ⅔", dash: "5 3" }], 8, 14, { gap: 14 });

    d3.select("#ffn-readout").html(
      "d = " + d + ", d_ff = " + B.dff + " (" + DL.fmt(B.dff / d, 2) + "d), " + B.nMat +
      " FFN matrices, " + L + " layers. Attention holds <b>" + DL.commas(B.attn) +
      "</b> parameters per layer and the feedforward network <b>" + DL.commas(B.ffn) + "</b> — a share of <b>" +
      (100 * B.ffn / B.total).toFixed(1) + "%</b>. Whole stack: <b>" + DL.big(B.total * L) +
      "</b> non-embedding parameters. " +
      (kind === "gated23"
        ? "The two-thirds-scaled gated variant costs <b>" + (B.total / plain.total).toFixed(3) +
        "×</b> the plain configuration — matched to within rounding, which is the whole point of the 2/3 factor."
        : (kind === "gated"
          ? "Adding the third matrix at the same d_ff costs <b>" + (B.total / plain.total).toFixed(3) +
          "×</b> the plain configuration. Shrink d_ff by 2/3 to get the parameter count back."
          : "At the standard ratio of 4 the share is exactly 8/12 = 66.7%: two thirds of every transformer layer is the object of §02."))
    );
  }
  [dE, rE, LE, kE].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ═════════════════ 21 · #mem-svg — memorisation ═════════════════
   The one figure on this page that TRAINS a network rather than constructing
   one, and it does it through DL.mlpInit / DL.backward / DL.sgdStep so that the
   backward pass it relies on is the same one verified against finite
   differences in the build (max relative difference 5.5e−07).               */
(function () {
  const svg = d3.select("#mem-svg");
  if (svg.empty()) return;
  const W = 760, H = 400, N = 80;
  const nfE = document.getElementById("mm-nf"), hE = document.getElementById("mm-h"),
    sE = document.getElementById("mm-s"), seE = document.getElementById("mm-seed");

  function data(seed) {
    const r = DL.rng(seed), X = [], y = [];
    for (let i = 0; i < N; i++) {
      const c = i < N / 2 ? 0 : 1;
      const t = 2 * Math.PI * r(), rad = (c === 0 ? 0.45 : 1.0) + 0.13 * DL.randn(r);
      X.push([rad * Math.cos(t), rad * Math.sin(t)]); y.push(c);
    }
    return { X: X, y: y };
  }
  function run(h, steps, lr, nf, seed) {
    const D = data(seed), r = DL.rng(seed + 5);
    const flipped = [], yTrain = D.y.map((v, i) => { const f = r() < nf; flipped.push(f); return f ? 1 - v : v; });
    const idx = d3.range(N), tr = idx.filter(i => i % 2 === 0), te = idx.filter(i => i % 2 === 1);
    const net = DL.mlpInit([2, h, h, 1], { act: "tanh", out: "sigmoid", seed: seed + 9 });
    const Xtr = tr.map(i => D.X[i]), Ytr = tr.map(i => [yTrain[i]]);
    const hist = [];
    for (let t = 0; t < steps; t++) {
      const g = DL.backward(net, Xtr, Ytr);
      DL.sgdStep(net, g, lr);
      if (t % Math.max(1, Math.round(steps / 60)) === 0) hist.push([t, g.loss]);
    }
    const P = DL.predict(net, Xtr);
    let acc = 0; P.forEach((p, i) => { if ((p[0] >= 0.5 ? 1 : 0) === Ytr[i][0]) acc++; });
    const Xte = te.map(i => D.X[i]), Yte = te.map(i => D.y[i]);
    const Q = DL.predict(net, Xte);
    let tacc = 0; Q.forEach((p, i) => { if ((p[0] >= 0.5 ? 1 : 0) === Yte[i]) tacc++; });
    return {
      net: net, D: D, yTrain: yTrain, flipped: flipped, tr: tr, te: te,
      trainAcc: acc / tr.length, testAcc: tacc / te.length,
      loss: hist.length ? hist[hist.length - 1][1] : NaN, hist: hist
    };
  }
  /* the five-point sweep is expensive; recompute only when it can change */
  let memo = { key: "", rows: null };
  function sweep(h, steps, lr, seed) {
    const key = h + "|" + steps + "|" + seed;
    if (memo.key === key) return memo.rows;
    const rows = [0, 0.15, 0.30, 0.45, 0.60].map(nf => {
      const o = run(h, steps, lr, nf, seed);
      return { nf: nf, trainAcc: o.trainAcc, testAcc: o.testAcc, loss: o.loss };
    });
    memo = { key: key, rows: rows };
    return rows;
  }

  function draw() {
    const nf = +nfE.value / 100, h = +hE.value, steps = +sE.value, seed = +seE.value, lr = 3;
    document.getElementById("mm-nfv").textContent = nfE.value + "%";
    document.getElementById("mm-hv").textContent = h;
    document.getElementById("mm-seedv").textContent = seed;

    const R = run(h, steps, lr, nf, seed);
    const rows = sweep(h, steps, lr, seed);
    const P = DL.params([2, h, h, 1]);

    const F = DL.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 34 });
    const G = F.g, PS = 214;

    /* left: the data and the learned boundary */
    G.append("text").attr("x", 6).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("training set with " + nfE.value + "% of labels flipped");
    const gg = G.append("g").attr("transform", "translate(6,0)");
    const sc = d3.scaleLinear().domain([-1.5, 1.5]).range([0, PS]);
    const res = 52, cw = PS / res;
    const gridPts = [];
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++)
      gridPts.push([sc.invert((i + 0.5) * cw), sc.invert(PS - (j + 0.5) * cw)]);
    const pr = DL.predict(R.net, gridPts);
    DL.cells(gg, 0, 0, cw, res, res, (i, j) => {
      const v = pr[j * res + i][0];
      return v >= 0.5 ? "rgba(255,180,84,0.20)" : "rgba(91,156,255,0.20)";
    });
    gg.append("rect").attr("width", PS).attr("height", PS).attr("fill", "none").attr("stroke", DC.line);
    R.tr.forEach(i => {
      const p = R.D.X[i];
      gg.append("circle").attr("cx", sc(p[0])).attr("cy", PS - sc(p[1])).attr("r", 3.2)
        .attr("fill", R.yTrain[i] ? DC.a2 : DC.accent);
      if (R.flipped[i]) gg.append("circle").attr("cx", sc(p[0])).attr("cy", PS - sc(p[1])).attr("r", 6)
        .attr("fill", "none").attr("stroke", DC.bad).attr("stroke-width", 1.4);
    });
    gg.append("text").attr("x", 0).attr("y", PS + 15).attr("font-size", 10).attr("fill", DC.muted)
      .text("ringed points had their label flipped; " + P + " parameters, " + R.tr.length + " training points");

    /* middle: the loss curve */
    const x1 = 6 + PS + 40, mw = 190;
    G.append("text").attr("x", x1).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("training loss");
    const g2 = G.append("g").attr("transform", `translate(${x1},0)`);
    const ls = d3.scaleLinear().domain([0, steps]).range([0, mw]);
    const lo = Math.max(1e-4, d3.min(R.hist, d => d[1]) * 0.6);
    const vs = d3.scaleLog().domain([lo, Math.max(1, d3.max(R.hist, d => d[1]) * 1.3)]).range([PS, 0]);
    DL.gridY(g2, vs, mw, 5);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${PS})`).call(d3.axisBottom(ls).ticks(4));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(vs).ticks(5, "~e"));
    DL.curve(g2, R.hist.map(d => [ls(d[0]), vs(DL.clamp(d[1], lo, vs.domain()[1]))]), { stroke: DC.a2, w: 2 });
    g2.append("text").attr("x", mw).attr("y", PS + 30).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", DC.muted).text("step");
    g2.append("text").attr("x", 4).attr("y", PS + 15).attr("font-size", 10.5)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.good)
      .text("train acc " + (100 * R.trainAcc).toFixed(1) + "%  ·  loss " + DL.fmtE(R.loss, 2));

    /* right: the sweep */
    const x2 = x1 + mw + 56, sw = 216;
    G.append("text").attr("x", x2).attr("y", -8).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("accuracy against label corruption");
    const g3 = G.append("g").attr("transform", `translate(${x2},0)`);
    const fs = d3.scaleLinear().domain([0, 0.6]).range([0, sw]);
    const as = d3.scaleLinear().domain([0, 1.02]).range([PS, 0]);
    DL.gridY(g3, as, sw, 5);
    g3.append("g").attr("class", "axis").attr("transform", `translate(0,${PS})`).call(d3.axisBottom(fs).ticks(4, ".0%"));
    g3.append("g").attr("class", "axis").call(d3.axisLeft(as).ticks(5, ".0%"));
    g3.append("line").attr("x1", 0).attr("x2", sw).attr("y1", as(0.5)).attr("y2", as(0.5))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    g3.append("text").attr("x", 2).attr("y", as(0.5) - 4).attr("font-size", 9).attr("fill", DC.muted).text("chance");
    DL.curve(g3, rows.map(r => [fs(r.nf), as(r.trainAcc)]), { stroke: DC.good, w: 2.4 });
    DL.curve(g3, rows.map(r => [fs(r.nf), as(r.testAcc)]), { stroke: DC.bad, w: 2.4, dash: "5 3" });
    rows.forEach(r => {
      g3.append("circle").attr("cx", fs(r.nf)).attr("cy", as(r.trainAcc)).attr("r", 3).attr("fill", DC.good);
      g3.append("circle").attr("cx", fs(r.nf)).attr("cy", as(r.testAcc)).attr("r", 3).attr("fill", DC.bad);
    });
    g3.append("line").attr("x1", fs(DL.clamp(nf, 0, 0.6))).attr("x2", fs(DL.clamp(nf, 0, 0.6)))
      .attr("y1", 0).attr("y2", PS).attr("stroke", DC.ink).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    DL.legend(g3, [{ color: DC.good, label: "train acc, on the labels shown" },
    { color: DC.bad, label: "held-out acc, on TRUE labels", dash: "5 3" }], 4, 14, { gap: 14, font: 9.5 });
    g3.append("text").attr("x", 0).attr("y", PS + 30).attr("font-size", 10).attr("fill", DC.muted)
      .text("fraction of training labels randomised");

    const r0 = rows[0], rl = rows[rows.length - 1];
    d3.select("#mem-readout").html(
      "A <b>" + P + "</b>-parameter network on <b>" + R.tr.length + "</b> training points. With <b>" +
      nfE.value + "%</b> of the labels flipped it still reaches <b>" + (100 * R.trainAcc).toFixed(1) +
      "%</b> training accuracy at a loss of " + DL.fmtE(R.loss, 2) +
      ", and scores <b>" + (100 * R.testAcc).toFixed(1) + "%</b> on held-out data against the true labels. " +
      "Across the whole sweep the training accuracy moves from " + (100 * r0.trainAcc).toFixed(1) +
      "% to " + (100 * rl.trainAcc).toFixed(1) + "% — essentially flat — while held-out accuracy falls from <b>" +
      (100 * r0.testAcc).toFixed(1) + "%</b> to <b>" + (100 * rl.testAcc).toFixed(1) + "%</b>" +
      (rl.testAcc < 0.5 ? ", <b>below chance</b>, because at 60% corruption the model has faithfully learned a labelling anti-correlated with the truth" : "") +
      ". <b>Capacity is not the binding constraint at any corruption level</b>; the same architecture that memorises noise generalises when the labels are real, so the parameter count cannot be what decides which of the two it does.");
  }
  [nfE, hE, sE, seE].forEach(el => el.addEventListener("input", draw));
  draw();
})();
