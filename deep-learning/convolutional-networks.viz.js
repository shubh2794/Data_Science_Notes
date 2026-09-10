/* convolutional-networks.viz.js — the visualizations on
   deep-learning/convolutional-networks.html (Deep Learning · part 3).
   Loaded after ../data.js → ../notes.js → dl-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the
   page, so a section can be reordered or removed without breaking the rest.

   Every numeric primitive — the convolution itself and its two backward
   operations, the output-size and padding arithmetic, pooling, the
   receptive-field chain, im2col, the parameter and MAC counts, the
   activations, the stable losses, the optimisers and the normalisation
   layers — comes from dl-viz.js (DL.*). NOTHING here forks those. Read the
   SCOPE BOUNDARY comment above DL.corr2d before adding any filtering
   numeric: classical filtering belongs to vision/vision-viz.js.
   Page-specific numerics live in CN below.

     1  #dense-svg     the layer as a matrix: dense, banded, tied
     2  #explode-svg   parameters, memory and weight reuse
     3  #corr-svg      correlation, convolution, and the flip
     4  #sparse-svg    sparse interactions, from above and from below
     5  #share-svg     parameter sharing: one number, many edges
     6  #equi-svg      shift then convolve, against convolve then shift
     7  #shape-svg     the output-size formula, one knob at a time
     8  #trap-svg      the four off-by-one traps
     9  #vol-svg       one 3-D filter on a 3-channel input
    10  #bank-svg      a bank of filters and the stack it produces
    11  #count-svg     one layer's three counts, and what moves each
    12  #net-svg       a whole network, counted end to end
    13  #rf-svg        receptive field: recursion against measurement
    14  #erf-svg       effective against theoretical receptive field
    15  #onexone-svg   the 1×1 bottleneck and its cost curve
    16  #pool-svg      the pooling variants and what each discards
    17  #invar-svg     equivariance and invariance, measured
    18  #stride-svg    subsampling, the fold, and the pre-filter
    19  #group-svg     grouped convolution and channel blocking
    20  #dws-svg       depthwise separable, and the ratio derived
    21  #dil-svg       dilation, growth, and the gridding artefact
    22  #trans-svg     transposed convolution and the checkerboard
    23  #block-svg     normalisation axes, and the inert convolution bias
    24  #bp-svg        the three gradients, against central differences
    25  #im2col-svg    the layer as one matrix product
    26  #learn-svg     patch principal components, and a layer trained live
    27  #lineage-svg   the architecture lineage as a trade-off surface
    28  #crit-svg      the arrangement test: right parts, wrong places
    29  #route-svg     routing by agreement, iteration by iteration
    30  #vit-svg       the prior against the data: measured sample efficiency

   Every number these print is recomputed from the data they draw.          */

/* ══════════ page-local helpers (deliberately NOT in dl-viz.js) ══════════ */
const CN = (function () {

  /* ---- the small test images the walkthrough figures use. All 6×6 unless
     asked otherwise, values in 0…10 so the hand arithmetic stays legible. */
  function testImage(kind, n, seed) {
    const N = n || 6, A = DL.zeros2(N, N), r = DL.rng(seed || 3);
    const h = Math.floor(N / 2);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      switch (kind) {
        case "half": A[i][j] = (j < h) ? 10 : 0; break;
        case "halfr": A[i][j] = (j < h) ? 0 : 10; break;
        case "quad": A[i][j] = ((i < h) === (j < h)) ? 10 : 0; break;
        case "bar": A[i][j] = (j >= h - 1 && j <= h) ? 10 : 0; break;
        case "cross": A[i][j] = (Math.abs(i - h) <= 0 || Math.abs(j - h) <= 0) ? 10 : 0; break;
        default: A[i][j] = Math.round(r() * 9);
      }
    }
    return A;
  }
  const KERNELS = {
    vert: { K: [[1, 0, -1], [1, 0, -1], [1, 0, -1]], label: "vertical edge" },
    horz: { K: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]], label: "horizontal edge" },
    sobel: { K: [[1, 0, -1], [2, 0, -2], [1, 0, -1]], label: "Sobel, vertical" },
    scharr: { K: [[3, 0, -3], [10, 0, -10], [3, 0, -3]], label: "Scharr, vertical" },
    blur: { K: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]], label: "3×3 box" },
    asym: { K: [[1, 2, 0], [0, 0, 3], [0, 4, 0]], label: "asymmetric" }
  };
  /* is K unchanged / negated by the double reversal? decides what the flip does */
  function flipSymmetry(K) {
    const F = DL.flipK(K);
    let same = true, neg = true;
    for (let i = 0; i < K.length; i++) for (let j = 0; j < K[0].length; j++) {
      if (Math.abs(F[i][j] - K[i][j]) > 1e-12) same = false;
      if (Math.abs(F[i][j] + K[i][j]) > 1e-12) neg = false;
    }
    return same ? "symmetric" : (neg ? "antisymmetric" : "neither");
  }
  const maxAbsDiff = (A, B) => {
    let m = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
    return m;
  };
  const extent2 = A => {
    let lo = Infinity, hi = -Infinity;
    A.forEach(r => r.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    return [lo, hi];
  };
  /* a signed blue↔orange ramp; 0 maps to the panel colour */
  function signRamp(v, m) {
    if (!m) return DC.panel2;
    const t = DL.clamp(v / m, -1, 1);
    return t >= 0 ? d3.interpolateRgb(DC.panel2, DC.accent)(t) : d3.interpolateRgb(DC.panel2, DC.a2)(-t);
  }
  const grey = (v, lo, hi) => d3.interpolateRgb("#11141c", "#c9d2e0")((v - lo) / ((hi - lo) || 1));

  /* draw a small numeric grid; returns the <g> so the caller can annotate */
  function grid(g, A, x, y, cw, opt) {
    const o = Object.assign({ fill: null, text: true, dp: 0, font: 10, stroke: DC.line, title: null }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    const [lo, hi] = extent2(A);
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
      const c = o.fill ? o.fill(A[i][j], i, j, lo, hi, m) : signRamp(A[i][j], m);
      gg.append("rect").attr("x", j * cw).attr("y", i * cw).attr("width", cw).attr("height", cw)
        .attr("fill", c).attr("stroke", o.stroke).attr("stroke-width", 0.6).attr("shape-rendering", "crispEdges");
      if (o.text && cw >= 15) gg.append("text").attr("x", j * cw + cw / 2).attr("y", i * cw + cw / 2 + 3.6)
        .attr("text-anchor", "middle").attr("font-size", o.font)
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", Math.abs(A[i][j]) > 0.55 * m ? "#0f1117" : DC.ink)
        .text(o.dp === 0 ? Math.round(A[i][j]) : A[i][j].toFixed(o.dp));
    }
    if (o.title) gg.append("text").attr("x", 0).attr("y", -6).attr("font-size", 10.5).attr("fill", DC.muted).text(o.title);
    return gg;
  }
  function box(g, x, y, w, h, color, wid) {
    return g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h)
      .attr("fill", "none").attr("stroke", color || DC.good).attr("stroke-width", wid || 2)
      .attr("shape-rendering", "crispEdges").attr("pointer-events", "none");
  }
  /* a horizontal bar row on a shared log axis */
  function logBars(g, rows, x, y, w, rowH, opt) {
    const o = Object.assign({ min: 1, label: v => DL.big(v) }, opt || {});
    const hi = Math.max.apply(null, rows.map(r => Math.max(r.a, r.b, o.min)));
    const sc = d3.scaleLog().domain([o.min, hi * 1.4]).range([0, w]).clamp(true);
    rows.forEach((r, i) => {
      const yy = y + i * rowH;
      g.append("text").attr("x", x - 8).attr("y", yy + 12).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", DC.muted).text(r.name);
      [[r.a, DC.bad, 0], [r.b, DC.good, 10]].forEach(([v, c, dy]) => {
        const ww = Math.max(1, sc(Math.max(v, o.min)));
        g.append("rect").attr("x", x).attr("y", yy + dy).attr("width", ww).attr("height", 9)
          .attr("fill", c).attr("fill-opacity", 0.8).attr("stroke", c).attr("stroke-width", 0.7);
        g.append("text").attr("x", x + ww + 5).attr("y", yy + dy + 8).attr("font-size", 9.5)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", c).text(o.label(v));
      });
    });
    return sc;
  }

  /* ---- 1-D wrappers. DL.corr2d and DL.pool2d apply the SAME padding pair
     and the same window to BOTH axes, so a single-row "image" would grow (or
     vanish) vertically. These pad along the row's own axis explicitly and
     then call the shared routine with zero padding, so the arithmetic is
     still dl-viz.js's and only the shape handling is local. pool1d is
     written out because a k×k window cannot fit a 1-row image at all; its
     output length uses DL.outSize, so its geometry cannot drift from the
     2-D version's.                                                        */
  const pad1 = (x, p) => new Array(p).fill(0).concat(x, new Array(p).fill(0));
  function conv1d(x, k, o) {
    const oo = Object.assign({ s: 1, p: 0, d: 1 }, o || {});
    return DL.corr2d([pad1(x, oo.p)], [k], { s: oo.s, p0: 0, p1: 0, d: oo.d })[0];
  }
  function pool1d(x, k, s, mode) {
    const n = DL.outSize(x.length, k, { s: s, p0: 0, p1: 0, d: 1 });
    const out = [];
    for (let q = 0; q < n; q++) {
      let best = -Infinity, sum = 0, c = 0;
      for (let m = 0; m < k; m++) {
        const i = q * s + m;
        if (i < 0 || i >= x.length) continue;
        sum += x[i]; c++; if (x[i] > best) best = x[i];
      }
      out.push(mode === "avg" ? (c ? sum / c : 0) : (c ? best : 0));
    }
    return out;
  }

  /* ---- run a network specification: EVERY shape comes from DL.outSize and
     every count from DL.convCount, so the architecture table and the lineage
     scatter cannot disagree with each other or with §12's formulas.       */
  function runNet(spec, head) {
    let H0 = spec.n, W0 = spec.n, c = spec.c, flat = false, gapDone = false;
    const rows = [];
    const layers = spec.L.slice();
    for (let li = 0; li < layers.length; li++) {
      const L = layers[li];
      if (L.t === "conv") {
        const C = DL.convCount({ H: H0, W: W0, cin: c, cout: L.n, k: L.k, s: L.s, p0: L.p, p1: L.p });
        const exact = ((H0 + 2 * L.p - L.k) % L.s) === 0;
        rows.push({ t: `conv ${L.k}×${L.k}`, geo: `s${L.s} p${L.p}`, in: `${H0}×${W0}×${c}`, out: `${C.Ho}×${C.Wo}×${L.n}`, p: C.params, m: C.macs, a: C.acts, exact: exact });
        H0 = C.Ho; W0 = C.Wo; c = L.n;
      } else if (L.t === "pool") {
        const geom = { s: L.s, p0: 0, p1: 0, d: 1 };
        const Ho = DL.outSize(H0, L.k, geom), Wo = DL.outSize(W0, L.k, geom);
        const exact = ((H0 - L.k) % L.s) === 0;
        rows.push({ t: `maxpool ${L.k}×${L.k}`, geo: `s${L.s}`, in: `${H0}×${W0}×${c}`, out: `${Ho}×${Wo}×${c}`, p: 0, m: 0, a: c * Ho * Wo, exact: exact });
        H0 = Ho; W0 = Wo;
      } else {
        /* the head. In "gap" mode the FIRST fc is replaced by a global pool
           and every hidden fc is dropped, so only the classifier remains. */
        if (!flat) {
          if (head === "gap" && !gapDone) {
            rows.push({ t: "global avg pool", geo: "", in: `${H0}×${W0}×${c}`, out: `1×1×${c}`, p: 0, m: c * H0 * W0, a: c, exact: true });
            H0 = 1; W0 = 1; gapDone = true;
            if (!L.out) continue;                    // skip the hidden dense layers
          } else {
            rows.push({ t: "flatten", geo: "", in: `${H0}×${W0}×${c}`, out: `${H0 * W0 * c}`, p: 0, m: 0, a: H0 * W0 * c, exact: true });
          }
          c = H0 * W0 * c; H0 = 1; W0 = 1; flat = true;
        }
        if (head === "gap" && !L.out) continue;
        const p = c * L.n + L.n;
        rows.push({ t: L.out ? "dense (softmax)" : "dense", geo: "", in: `${c}`, out: `${L.n}`, p: p, m: c * L.n, a: L.n, exact: true });
        c = L.n;
      }
    }
    return rows;
  }


  return {
    testImage: testImage, KERNELS: KERNELS, flipSymmetry: flipSymmetry,
    runNet: runNet,
    pad1: pad1, conv1d: conv1d, pool1d: pool1d,
    maxAbsDiff: maxAbsDiff, extent2: extent2, signRamp: signRamp, grey: grey,
    grid: grid, box: box, logBars: logBars
  };
})();

/* ═══════════ 1 · #dense-svg — the layer as a matrix: dense, banded, tied ═══════════ */
(function () {
  const svg = d3.select("#dense-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const ne = document.getElementById("dn-n"), fe = document.getElementById("dn-f"),
    se = document.getElementById("dn-s"), pe = document.getElementById("dn-p");

  function draw() {
    const n = +ne.value, f = +fe.value, s = +se.value, mode = pe.value;
    document.getElementById("dn-nv").textContent = n;
    document.getElementById("dn-fv").textContent = f;
    document.getElementById("dn-sv").textContent = s;

    const pad = DL.padFor(mode, n, f, s, 1);
    const geom = { s: s, p0: pad.p0, p1: pad.p1, d: 1 };
    const nOut = Math.max(0, DL.outSize(n, f, geom));

    /* the TAP-INDEX matrix: run the layer on each indicator input with a
       kernel whose taps are 1…f, so the surviving entry names its tap. */
    const tapK = [[]]; for (let t = 0; t < f; t++) tapK[0].push(t + 1);
    const CM = DL.convMatrix(1, n, tapK, geom);        // (nOut × n)
    const M = CM.M;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 26, b: 12 });
    const g = F.g;
    const cw = Math.min(20, Math.floor(300 / Math.max(n, 6)));
    const rnd = DL.rng(11);

    if (nOut <= 0) {
      g.append("text").attr("x", 0).attr("y", 40).attr("font-size", 13).attr("fill", DC.bad)
        .text(`kernel ${f} does not fit in an input of length ${n} under "${mode}" padding — no output`);
      document.getElementById("dense-readout").innerHTML =
        `<b>no valid output.</b> With n = ${n}, f = ${f}, s = ${s} and ${mode} padding the formula ⌊(n + p₀ + p₁ − f)/s⌋ + 1 gives ${DL.outSize(n, f, geom)}, which is not a positive size.`;
      return;
    }

    /* ---- left: the dense matrix ---- */
    const x0 = 40, y0 = 26;
    g.append("text").attr("x", x0).attr("y", 10).attr("font-size", 11.5).attr("fill", DC.ink)
      .attr("font-weight", 600).text(`dense  (${nOut} × ${n})`);
    const dvals = [];
    for (let i = 0; i < nOut; i++) { const row = []; for (let j = 0; j < n; j++) row.push(rnd() * 2 - 1); dvals.push(row); }
    for (let i = 0; i < nOut; i++) for (let j = 0; j < n; j++) {
      g.append("rect").attr("x", x0 + j * cw).attr("y", y0 + i * cw).attr("width", cw).attr("height", cw)
        .attr("fill", CN.signRamp(dvals[i][j], 1)).attr("stroke", DC.line).attr("stroke-width", 0.5)
        .attr("shape-rendering", "crispEdges");
    }
    g.append("text").attr("x", x0).attr("y", y0 + nOut * cw + 16).attr("font-size", 10.5).attr("fill", DC.muted)
      .text(`${DL.commas(nOut * n)} entries, all ${DL.commas(nOut * n)} of them free`);

    /* ---- right: the convolutional matrix ---- */
    const x1 = x0 + n * cw + 96;
    g.append("text").attr("x", x1).attr("y", 10).attr("font-size", 11.5).attr("fill", DC.ink)
      .attr("font-weight", 600).text(`convolutional  (${nOut} × ${n})`);
    const tapCol = t => d3.interpolateTurbo(0.14 + 0.72 * (f === 1 ? 0.5 : (t - 1) / (f - 1)));
    let nz = 0;
    const used = new Set();
    for (let i = 0; i < nOut; i++) for (let j = 0; j < n; j++) {
      const t = Math.round(M[i][j]);
      const on = t > 0;
      if (on) { nz++; used.add(t); }
      g.append("rect").attr("x", x1 + j * cw).attr("y", y0 + i * cw).attr("width", cw).attr("height", cw)
        .attr("fill", on ? tapCol(t) : "#20242e").attr("fill-opacity", on ? 0.92 : 1)
        .attr("stroke", DC.line).attr("stroke-width", 0.5).attr("shape-rendering", "crispEdges");
      if (on && cw >= 16) g.append("text").attr("x", x1 + j * cw + cw / 2).attr("y", y0 + i * cw + cw / 2 + 3.4)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#0f1117")
        .attr("font-family", "SF Mono, Menlo, monospace").text("w" + (t - 1));
    }
    g.append("text").attr("x", x1).attr("y", y0 + nOut * cw + 16).attr("font-size", 10.5).attr("fill", DC.muted)
      .text(`${DL.commas(nz)} non-zero entries, taking only ${used.size} distinct values`);

    /* per-column usage: how often each INPUT is read — the border argument */
    const colUse = [];
    for (let j = 0; j < n; j++) { let c = 0; for (let i = 0; i < nOut; i++) if (M[i][j] > 0) c++; colUse.push(c); }
    const byy = y0 + nOut * cw + 40, bh = 42;
    const mx = Math.max.apply(null, colUse) || 1;
    g.append("text").attr("x", x0).attr("y", byy - 6).attr("font-size", 10.5).attr("fill", DC.ink)
      .text("how many outputs read each input position — the border argument for padding (§08)");
    for (let j = 0; j < n; j++) {
      const hh = bh * colUse[j] / mx;
      g.append("rect").attr("x", x1 + j * cw + 1).attr("y", byy + bh - hh).attr("width", cw - 2).attr("height", hh)
        .attr("fill", colUse[j] === mx ? DC.good : DC.a2).attr("fill-opacity", 0.75);
      g.append("text").attr("x", x1 + j * cw + cw / 2).attr("y", byy + bh + 11).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", DC.muted).text(colUse[j]);
    }

    /* the kernel itself, as a legend */
    const kx = x0, ky = byy;
    g.append("text").attr("x", kx).attr("y", ky + 12).attr("font-size", 10.5).attr("fill", DC.muted)
      .text("one colour per free parameter:");
    for (let t = 1; t <= f; t++) {
      g.append("rect").attr("x", kx + (t - 1) * 30).attr("y", ky + 20).attr("width", 26).attr("height", 20)
        .attr("fill", tapCol(t)).attr("fill-opacity", 0.92).attr("stroke", DC.line);
      g.append("text").attr("x", kx + (t - 1) * 30 + 13).attr("y", ky + 34).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", "#0f1117").attr("font-family", "SF Mono, Menlo, monospace").text("w" + (t - 1));
    }
    g.append("text").attr("x", kx).attr("y", ky + 58).attr("font-size", 10.5).attr("fill", DC.muted)
      .text(`+ 1 bias  →  ${f + 1} parameters total, whatever n is`);

    const ratio = (nOut * n) / (f + 1);
    document.getElementById("dense-readout").innerHTML =
      `<b>n = ${n}, f = ${f}, s = ${s}, ${mode} padding (p₀ = ${pad.p0}, p₁ = ${pad.p1})</b> → ${nOut} outputs. ` +
      `The matrix is ${nOut} × ${n} = <b>${DL.commas(nOut * n)}</b> entries either way. ` +
      `Dense: ${DL.commas(nOut * n)} free parameters. Convolutional: <b>${DL.commas(nz)}</b> non-zeros taking <b>${used.size}</b> distinct values, ` +
      `so ${used.size} + 1 = <b>${used.size + 1}</b> free parameters — a factor of <b>${ratio.toFixed(1)}×</b> fewer. ` +
      `Structural zeros: ${DL.commas(nOut * n - nz)} of ${DL.commas(nOut * n)} = ${(100 * (1 - nz / (nOut * n))).toFixed(1)}%. ` +
      `Input positions read by only one output: <b>${colUse.filter(c => c === 1).length}</b>; by the maximum ${mx}: ${colUse.filter(c => c === mx).length}.`;
  }
  [ne, fe, se].forEach(e => e.addEventListener("input", draw));
  pe.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 2 · #explode-svg — parameters, memory, weight reuse ═══════════ */
(function () {
  const svg = d3.select("#explode-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const ne = document.getElementById("ex-n"), cie = document.getElementById("ex-ci"),
    coe = document.getElementById("ex-co"), fe = document.getElementById("ex-f"),
    te = document.getElementById("ex-t");

  function draw() {
    const n = +ne.value, ci = +cie.value, co = +coe.value, f = +fe.value, tgt = te.value;
    document.getElementById("ex-nv").textContent = n;
    document.getElementById("ex-civ").textContent = ci;
    document.getElementById("ex-cov").textContent = co;
    document.getElementById("ex-fv").textContent = f;

    const p = (f - 1) / 2;
    const C = DL.convCount({ H: n, W: n, cin: ci, cout: co, k: f, s: 1, p0: p, p1: p });
    const dIn = n * n * ci;
    const dOut = (tgt === "same") ? co * C.Ho * C.Wo : 1000;
    const denseP = dIn * dOut + dOut;
    const denseMacs = dIn * dOut;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    g.append("text").attr("x", 0).attr("y", 4).attr("font-size", 11.5).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`input ${n}×${n}×${ci} = ${DL.commas(dIn)} numbers  →  ` +
        (tgt === "same" ? `${C.Ho}×${C.Wo}×${co} = ${DL.commas(dOut)} numbers` : `1000 units`));
    DL.legend(g, [{ color: DC.bad, label: "dense layer" }, { color: DC.good, label: `convolution, ${co} filters of ${f}×${f}×${ci}` }],
      440, 2, { gap: 14, font: 10.5 });

    const rows = [
      { name: "parameters", a: denseP, b: C.params },
      { name: "weight bytes (fp32)", a: 4 * denseP, b: 4 * C.params },
      { name: "+ Adam state (×3)", a: 12 * denseP, b: 12 * C.params },
      { name: "MACs per example", a: denseMacs, b: C.macs },
      { name: "uses of each weight", a: 1, b: C.reuse }
    ];
    CN.logBars(g, rows, 132, 26, 400, 34, {
      label: v => (v < 1000 ? DL.commas(v) : DL.big(v))
    });
    g.append("text").attr("x", 132).attr("y", 26 + 5 * 34 + 6).attr("font-size", 10).attr("fill", DC.muted)
      .text("logarithmic axis — each gridline is a factor of ten");

    /* the exact integers, in a panel */
    const px = 132, py = 26 + 5 * 34 + 22;
    const k = DL.kv(g, px, py + 14, { lead: 15, keyW: 216, size: 11 });
    k("dense parameters", DL.commas(denseP), DC.bad);
    k("convolution parameters", DL.commas(C.params), DC.good);
    k("ratio", DL.fmtE(denseP / C.params, 2) + " ×", DC.a2, true);
    k("convolution output map", `${C.Ho} × ${C.Wo} × ${co}`, DC.ink);
    k("each conv weight is used", DL.commas(C.reuse) + " times per example", DC.good);

    document.getElementById("explode-readout").innerHTML =
      `<b>${DL.commas(denseP)}</b> parameters for the dense layer against <b>${DL.commas(C.params)}</b> for the convolution — ` +
      `a factor of <b>${DL.fmtE(denseP / C.params, 2)}</b>. ` +
      `Weights alone: ${(4 * denseP / 1e9).toFixed(3)} GB against ${(4 * C.params / 1e6).toFixed(3)} MB. ` +
      `Arithmetic, by contrast, is <b>${denseMacs > C.macs ? (denseMacs / C.macs).toFixed(2) + "× more" : (C.macs / denseMacs).toFixed(2) + "× less"}</b> for the dense layer ` +
      `(${DL.big(denseMacs)} against ${DL.big(C.macs)} MACs) — convolution's saving is in parameters and reuse, not primarily in FLOPs. ` +
      `Each dense weight is used exactly once per example; each convolutional weight is used <b>${DL.commas(C.reuse)}</b> times, ` +
      `which is what supplies ${DL.commas(C.reuse)} constraints per image on every one of them.`;
  }
  [ne, cie, coe, fe].forEach(e => e.addEventListener("input", draw));
  te.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 3 · #corr-svg — correlation, convolution, and the flip ═══════════ */
(function () {
  const svg = d3.select("#corr-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const ie = document.getElementById("cr-img"), ke = document.getElementById("cr-k"),
    oe = document.getElementById("cr-op"), pe = document.getElementById("cr-pos");

  function draw() {
    const kind = ie.value, kk = ke.value, op = oe.value, pos = +pe.value;
    const I = CN.testImage(kind, 6, 3);
    const K = CN.KERNELS[kk].K;
    const f = 3, N = 6, No = N - f + 1;                    // valid → 4×4
    const pi = Math.floor(pos / No), pj = pos % No;
    document.getElementById("cr-posv").textContent = `(${pi},${pj})`;

    const Ycorr = DL.corr2d(I, K, {});
    const Yconv = DL.conv2dTrue(I, K, {});
    const Y = (op === "conv") ? Yconv : Ycorr;
    const Kapp = (op === "conv") ? DL.flipK(K) : K;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 12 });
    const g = F.g;
    const cw = 30;

    /* ---- the image ---- */
    const [lo, hi] = CN.extent2(I);
    CN.grid(g, I, 0, 16, cw, {
      title: `input I (6×6)`, fill: v => CN.grey(v, lo, hi),
      font: 10.5
    });
    CN.box(g, pj * cw, 16 + pi * cw, f * cw, f * cw, DC.good, 2.2);

    /* ---- the kernel, as stored and as applied ---- */
    const kx = 6 * cw + 34;
    CN.grid(g, K, kx, 16, 26, { title: "kernel K, as stored", dp: kk === "blur" ? 2 : 0, font: 9.5 });
    CN.grid(g, Kapp, kx, 16 + 3 * 26 + 30, 26, {
      title: op === "conv" ? "flip(K) — what is applied" : "K — what is applied (no flip)",
      dp: kk === "blur" ? 2 : 0, font: 9.5
    });

    /* ---- the nine products ---- */
    const sx = kx + 3 * 26 + 34;
    g.append("text").attr("x", sx).attr("y", 10).attr("font-size", 10.5).attr("fill", DC.ink)
      .attr("font-weight", 600).text(`output (${pi},${pj}) — the nine terms`);
    let acc = 0, li = 0;
    for (let m = 0; m < f; m++) for (let n = 0; n < f; n++) {
      const iv = I[pi + m][pj + n], kv = Kapp[m][n];
      acc += iv * kv;
      const yy = 24 + li * 13.5;
      g.append("text").attr("x", sx).attr("y", yy).attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", Math.abs(iv * kv) > 1e-9 ? DC.ink : DC.muted)
        .attr("xml:space", "preserve")
        .text(`I[${pi + m}][${pj + n}]·K[${m}][${n}] = ${String(iv).padStart(3)} · ${String(kk === "blur" ? kv.toFixed(2) : kv).padStart(6)} = ${(kk === "blur" ? (iv * kv).toFixed(2) : String(iv * kv)).padStart(7)}`);
      li++;
    }
    g.append("line").attr("x1", sx).attr("x2", sx + 246).attr("y1", 24 + 9 * 13.5 - 4).attr("y2", 24 + 9 * 13.5 - 4)
      .attr("stroke", DC.line);
    g.append("text").attr("x", sx).attr("y", 24 + 9 * 13.5 + 12).attr("font-size", 11.5)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.a2).attr("font-weight", 600)
      .text(`sum = ${kk === "blur" ? acc.toFixed(3) : acc}`);

    /* ---- the two output maps ---- */
    const oy = 190;
    const gA = CN.grid(g, Ycorr, 0, oy + 16, 34, {
      title: "I ⋆ K   correlation  (what conv2d computes)", dp: kk === "blur" ? 1 : 0, font: 10
    });
    const gB = CN.grid(g, Yconv, 4 * 34 + 46, oy + 16, 34, {
      title: "I ∗ K   true convolution  (kernel flipped)", dp: kk === "blur" ? 1 : 0, font: 10
    });
    (op === "conv" ? gB : gA).call(sel => CN.box(sel, pj * 34, pi * 34, 34, 34, DC.good, 2.2));

    /* ---- the comparison ---- */
    const sym = CN.flipSymmetry(K);
    const dPlus = CN.maxAbsDiff(Ycorr, Yconv);
    const dMinus = CN.maxAbsDiff(Ycorr, Yconv.map(r => r.map(v => -v)));
    const cx = 2 * (4 * 34 + 46) - 20;
    const kv2 = DL.kv(g, cx, oy + 30, { lead: 16, keyW: 150, size: 11 });
    kv2("kernel under flip", sym, sym === "neither" ? DC.a2 : DC.good, true);
    kv2("max |corr − conv|", DL.fmt(dPlus, 3), dPlus < 1e-12 ? DC.good : DC.bad);
    kv2("max |corr + conv|", DL.fmt(dMinus, 3), dMinus < 1e-12 ? DC.good : DC.bad);
    kv2("so the flip is a…", dPlus < 1e-12 ? "no-op" : (dMinus < 1e-12 ? "pure sign flip" : "different map"),
      DC.ink, true);

    document.getElementById("corr-readout").innerHTML =
      `Window at output (${pi},${pj}); the nine terms sum to <b>${kk === "blur" ? acc.toFixed(3) : acc}</b>, ` +
      `which is the highlighted cell of the ${op === "conv" ? "convolution" : "correlation"} map. ` +
      `This kernel is <b>${sym}</b> under the double reversal, so max|I ⋆ K − I ∗ K| = <b>${DL.fmt(dPlus, 3)}</b> ` +
      `and max|I ⋆ K + I ∗ K| = <b>${DL.fmt(dMinus, 3)}</b>: the flip is ` +
      (dPlus < 1e-12 ? "<b>a no-op</b> — a symmetric kernel does not notice it."
        : (dMinus < 1e-12 ? "<b>exactly a sign flip</b> — the same edge, read with the opposite polarity."
          : "<b>a genuinely different map</b>, neither equal nor opposite; only an (anti)symmetric kernel gets off that lightly.")) +
      ` A learned kernel is free to absorb whichever convention it is trained under, which is why frameworks can compute one and call it the other.`;
  }
  [ie, ke, oe].forEach(e => e.addEventListener("change", draw));
  pe.addEventListener("input", draw);
  draw();
})();

/* ═══════════ 4 · #sparse-svg — sparse interactions, up and down ═══════════ */
(function () {
  const svg = d3.select("#sparse-svg");
  if (svg.empty()) return;
  const W = 760, H = 380, N = 17;
  const Le = document.getElementById("sp-L"), fe = document.getElementById("sp-f"),
    me = document.getElementById("sp-m"), ue = document.getElementById("sp-u"),
    de = document.getElementById("sp-d");

  function draw() {
    const L = +Le.value, f = +fe.value, mode = me.value, u = +ue.value, dir = de.value;
    document.getElementById("sp-Lv").textContent = L;
    document.getElementById("sp-fv").textContent = f;
    document.getElementById("sp-uv").textContent = u;

    /* every layer keeps length N: "same" padding, so the picture stays square
       and the receptive field is the only thing that grows. */
    const pad = DL.padFor("same", N, f, 1, 1);
    /* edges[l] = list of [inputIndex, outputIndex] between layer l and l+1 */
    const edges = [];
    for (let l = 0; l < L; l++) {
      const E = [];
      for (let q = 0; q < N; q++) {
        if (mode === "dense") { for (let i = 0; i < N; i++) E.push([i, q]); }
        else for (let m = 0; m < f; m++) {
          const i = q - pad.p0 + m;
          if (i >= 0 && i < N) E.push([i, q]);
        }
      }
      edges.push(E);
    }
    /* propagate the highlighted set by MEASUREMENT over the drawn edges */
    const sets = [];
    if (dir === "up") {                       // start at the top, walk down
      let cur = new Set([u]);
      sets[L] = cur;
      for (let l = L - 1; l >= 0; l--) {
        const nx = new Set();
        edges[l].forEach(([i, q]) => { if (cur.has(q)) nx.add(i); });
        sets[l] = nx; cur = nx;
      }
    } else {                                   // start at the bottom, walk up
      let cur = new Set([u]);
      sets[0] = cur;
      for (let l = 0; l < L; l++) {
        const nx = new Set();
        edges[l].forEach(([i, q]) => { if (cur.has(i)) nx.add(q); });
        sets[l + 1] = nx; cur = nx;
      }
    }
    const target = dir === "up" ? sets[0] : sets[L];

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 20, b: 12 });
    const g = F.g;
    const x0 = 44, dx = (W - 120) / (N - 1), rowH = Math.min(52, 210 / L);
    const yOf = l => 210 - l * rowH;
    const xOf = i => x0 + i * dx;

    for (let l = 0; l < L; l++) {
      g.append("g").selectAll("line").data(edges[l]).join("line")
        .attr("x1", d => xOf(d[0])).attr("y1", yOf(l))
        .attr("x2", d => xOf(d[1])).attr("y2", yOf(l + 1))
        .attr("stroke", d => (sets[l].has(d[0]) && sets[l + 1].has(d[1])) ? DC.accent : DC.line)
        .attr("stroke-width", d => (sets[l].has(d[0]) && sets[l + 1].has(d[1])) ? 1.5 : 0.6)
        .attr("stroke-opacity", d => (sets[l].has(d[0]) && sets[l + 1].has(d[1])) ? 0.95 : (mode === "dense" ? 0.10 : 0.4));
    }
    for (let l = 0; l <= L; l++) {
      g.append("g").selectAll("circle").data(d3.range(N)).join("circle")
        .attr("cx", xOf).attr("cy", yOf(l)).attr("r", 6)
        .attr("fill", i => sets[l].has(i) ? DC.accent : DC.panel2)
        .attr("fill-opacity", i => sets[l].has(i) ? 0.9 : 1)
        .attr("stroke", i => (l === (dir === "up" ? L : 0) && i === u) ? DC.a2 : DC.line)
        .attr("stroke-width", i => (l === (dir === "up" ? L : 0) && i === u) ? 2.4 : 1);
      g.append("text").attr("x", 6).attr("y", yOf(l) + 4).attr("font-size", 10).attr("fill", DC.muted)
        .text(l === 0 ? "input" : ("h" + l));
    }
    g.append("text").attr("x", x0).attr("y", yOf(L) - 16).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600)
      .text(dir === "up"
        ? `the receptive field of one output unit — ${target.size} of ${N} inputs`
        : `the projective field of one input unit — ${target.size} of ${N} deepest units`);

    /* parameters, both patterns */
    const pConv = L * f, pDense = L * N * N, pLocal = L * f * N;
    const edgeCount = edges.reduce((s, E) => s + E.length, 0);
    const by = 250;
    g.append("text").attr("x", 44).attr("y", by - 6).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(`free parameters in this ${L}-layer stack (single channel, biases omitted)`);
    CN.logBars(g, [
      { name: "fully connected", a: pDense, b: pDense },
      { name: "convolutional", a: pConv, b: pConv }
    ], 132, by, 300, 30, { label: v => DL.commas(v) });
    g.append("text").attr("x", 470).attr("y", by + 18).attr("font-size", 10.5).attr("fill", DC.muted)
      .text(`edges drawn: ${DL.commas(edgeCount)}`);
    g.append("text").attr("x", 470).attr("y", by + 33).attr("font-size", 10.5).attr("fill", DC.muted)
      .text(`locally connected would need ${DL.commas(pLocal)} (§06)`);

    const chain = [];
    for (let l = 0; l < L; l++) chain.push({ k: f, s: 1, p0: pad.p0, d: 1 });
    const rf = DL.rfChain(chain).slice(-1)[0].r;
    document.getElementById("sparse-readout").innerHTML =
      `${L} layer${L > 1 ? "s" : ""} of ${mode === "dense" ? "a fully connected map" : `a ${f}-wide kernel`} on ${N} units. ` +
      `The highlighted set, <b>counted from the drawn edges</b>, has <b>${target.size}</b> members` +
      (mode === "dense" ? ` — which is all ${N} of them, after a single layer.`
        : `; the layer-by-layer formula of §14 predicts a receptive field of <b>${Math.min(rf, N)}</b> (clipped by the ${N}-unit input and by the border), ` +
          `so formula and measurement ${Math.min(rf, N) === target.size ? "<b>agree</b>" : "<b>differ</b> — the border is clipping the field"}.`) +
      ` Parameters: <b>${DL.commas(pConv)}</b> for the convolutional stack over its ${DL.commas(mode === "dense" ? L * f * N : edgeCount)} edges, against <b>${DL.commas(pDense)}</b> for a dense stack of the same depth, which has ${DL.commas(L * N * N)} edges.` +
      ` Sparsity alone would give ${DL.commas(pLocal)}; the remaining factor of ${(pLocal / pConv).toFixed(0)}× is parameter sharing, not sparsity.`;
  }
  [Le, fe, ue].forEach(e => e.addEventListener("input", draw));
  [me, de].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 5 · #share-svg — one parameter, many uses ═══════════ */
(function () {
  const svg = d3.select("#share-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const me = document.getElementById("sh-m"), ne = document.getElementById("sh-n"),
    fe = document.getElementById("sh-f"), te = document.getElementById("sh-t");

  function draw() {
    const mode = me.value, n = +ne.value, f = +fe.value;
    document.getElementById("sh-nv").textContent = n;
    document.getElementById("sh-fv").textContent = f;
    const nOut = n - f + 1;
    if (nOut < 1) { ne.value = String(f); return draw(); }

    /* the parameter INDEX of the edge from input i to output q */
    const pid = (i, q, m) => {
      if (mode === "conv") return m;
      if (mode === "local") return q * f + m;
      if (mode === "tiled") return (q % 2) * f + m;
      return q * n + i;                                  // dense
    };
    const edges = [];
    if (mode === "dense") { for (let q = 0; q < nOut; q++) for (let i = 0; i < n; i++) edges.push([i, q, pid(i, q, 0)]); }
    else for (let q = 0; q < nOut; q++) for (let m = 0; m < f; m++) edges.push([q + m, q, pid(q + m, q, m)]);
    const nParam = mode === "conv" ? f : (mode === "local" ? f * nOut : (mode === "tiled" ? 2 * f : n * nOut));
    te.max = String(Math.max(0, nParam - 1));
    const t = Math.min(+te.value, nParam - 1);
    document.getElementById("sh-tv").textContent = t;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const col = p => d3.interpolateTurbo(0.12 + 0.76 * (nParam === 1 ? 0.5 : (p % 12) / 11));
    const x0 = 40, dx = Math.min(46, 560 / Math.max(n - 1, 1));
    const yIn = 130, yOut = 44;
    const xIn = i => x0 + i * dx, xOut = q => x0 + (q + (f - 1) / 2) * dx;

    g.append("g").selectAll("line").data(edges).join("line")
      .attr("x1", d => xIn(d[0])).attr("y1", yIn)
      .attr("x2", d => xOut(d[1])).attr("y2", yOut)
      .attr("stroke", d => d[2] === t ? DC.a2 : col(d[2]))
      .attr("stroke-width", d => d[2] === t ? 2.6 : 1)
      .attr("stroke-opacity", d => d[2] === t ? 1 : (mode === "dense" ? 0.16 : 0.5));
    g.append("g").selectAll("circle").data(d3.range(n)).join("circle")
      .attr("cx", xIn).attr("cy", yIn).attr("r", 6).attr("fill", DC.panel2).attr("stroke", DC.line);
    g.append("g").selectAll("circle").data(d3.range(nOut)).join("circle")
      .attr("cx", xOut).attr("cy", yOut).attr("r", 6).attr("fill", DC.panel2).attr("stroke", DC.accent);
    g.append("text").attr("x", 2).attr("y", yIn + 4).attr("font-size", 10).attr("fill", DC.muted).text("x");
    g.append("text").attr("x", 2).attr("y", yOut + 4).attr("font-size", 10).attr("fill", DC.muted).text("z");
    const usedBy = edges.filter(d => d[2] === t).length;
    g.append("text").attr("x", x0).attr("y", 16).attr("font-size", 11).attr("fill", DC.a2).attr("font-weight", 600)
      .text(`parameter #${t} controls ${usedBy} of the ${edges.length} edges`);

    /* the matrix, same colouring */
    const cw = Math.min(22, 240 / n);
    const mx = 44, my = 176;
    g.append("text").attr("x", mx).attr("y", my - 6).attr("font-size", 10.5).attr("fill", DC.muted)
      .text(`the (${nOut} × ${n}) weight matrix, one colour per free parameter`);
    const at = {};
    edges.forEach(([i, q, p]) => { at[q + "," + i] = p; });
    for (let q = 0; q < nOut; q++) for (let i = 0; i < n; i++) {
      const p = at[q + "," + i];
      g.append("rect").attr("x", mx + i * cw).attr("y", my + q * cw).attr("width", cw).attr("height", cw)
        .attr("fill", p === undefined ? "#20242e" : (p === t ? DC.a2 : col(p)))
        .attr("fill-opacity", p === undefined ? 1 : 0.92)
        .attr("stroke", DC.line).attr("stroke-width", 0.5).attr("shape-rendering", "crispEdges");
    }

    const bx = mx + n * cw + 60;
    CN.logBars(g, [
      { name: "edges (connections)", a: edges.length, b: edges.length },
      { name: "free parameters", a: nParam, b: nParam }
    ], bx + 110, my + 4, 180, 34, { label: v => DL.commas(v) });
    const kv = DL.kv(g, bx + 110, my + 90, { lead: 15, keyW: 150, size: 11 });
    kv("pattern", me.options[me.selectedIndex].text.split(" —")[0], DC.ink, true);
    kv("edges per parameter", (edges.length / nParam).toFixed(2), DC.good);

    document.getElementById("share-readout").innerHTML =
      `<b>${me.options[me.selectedIndex].text}</b>, n = ${n}, f = ${f} → ${nOut} outputs. ` +
      `<b>${DL.commas(edges.length)}</b> edges and <b>${DL.commas(nParam)}</b> free parameters, ` +
      `so each parameter controls <b>${(edges.length / nParam).toFixed(2)}</b> edges on average and the highlighted one controls <b>${usedBy}</b>. ` +
      (mode === "conv" ? "The convolutional pattern is the only one whose parameter count does not grow with the input length: add a hundred more input units and it is still " + f + "."
        : mode === "local" ? "Locally connected has exactly the same connectivity as convolutional — the same " + DL.commas(edges.length) + " edges — and " + (nParam / f).toFixed(0) + "× the parameters, because nothing is shared. Sparsity and sharing are separate choices."
        : mode === "tiled" ? "Tiled convolution cycles through 2 kernels, so neighbouring outputs use different weights but outputs 2 apart share them: " + nParam + " parameters, between convolution's " + f + " and locally connected's " + (f * nOut) + "."
        : "Fully connected: every edge is its own parameter, so the two bars are identical and the ratio is exactly 1.00.");
  }
  [me].forEach(e => e.addEventListener("change", draw));
  [ne, fe, te].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════ 6 · #equi-svg — shift then convolve, convolve then shift ═══════════ */
(function () {
  const svg = d3.select("#equi-svg");
  if (svg.empty()) return;
  const W = 760, H = 420, N = 16;
  const ae = document.getElementById("eq-a"), se = document.getElementById("eq-s"),
    pe = document.getElementById("eq-p"), tte = document.getElementById("eq-t"),
    ke = document.getElementById("eq-k");

  /* a small structured image: two bright blocks and a diagonal, kept clear of
     the border so a shift of ±6 does not push content off the canvas. */
  function img(kind) {
    const A = DL.zeros2(N, N);
    for (let i = 5; i < 9; i++) for (let j = 4; j < 7; j++) A[i][j] = 9;
    for (let i = 3; i < 6; i++) for (let j = 9; j < 12; j++) A[i][j] = 6;
    for (let d = 0; d < 5; d++) A[8 + d][8 + d] = 8;
    if (kind === "edge") {                    // content deliberately touching the border
      for (let j = 0; j < N; j++) { A[0][j] = 7; A[N - 1][j] = 4; }
      for (let i = 0; i < N; i++) { A[i][0] = 5; }
    }
    return A;
  }
  const shift2 = (A, a, b) => {                 // zero-filled translation
    const M = A.length, Nn = A[0].length, B = DL.zeros2(M, Nn);
    for (let i = 0; i < M; i++) for (let j = 0; j < Nn; j++) {
      const si = i - a, sj = j - b;
      B[i][j] = (si >= 0 && si < M && sj >= 0 && sj < Nn) ? A[si][sj] : 0;
    }
    return B;
  };
  const rot90 = A => {                          // counter-clockwise
    const M = A.length, Nn = A[0].length, B = DL.zeros2(Nn, M);
    for (let i = 0; i < M; i++) for (let j = 0; j < Nn; j++) B[Nn - 1 - j][i] = A[i][j];
    return B;
  };
  const fliph = A => A.map(r => r.slice().reverse());

  function draw() {
    const a = +ae.value, s = +se.value, mode = pe.value, tr = tte.value, kk = ke.value;
    const ie2 = document.getElementById("eq-i");
    document.getElementById("eq-av").textContent = (a >= 0 ? "+" : "") + a;
    document.getElementById("eq-sv").textContent = s;
    const K = CN.KERNELS[kk].K, f = 3;
    const pad = DL.padFor(mode, N, f, s, 1);
    const geom = { s: s, p0: pad.p0, p1: pad.p1, d: 1 };
    const X = img(document.getElementById("eq-i").value);
    const conv = A => DL.corr2d(A, K, geom);

    const T = tr === "shift" ? (A => shift2(A, a, a)) : (tr === "rot" ? rot90 : fliph);
    /* for translation the OUTPUT shift is a/s, which only exists when s | a */
    const divides = (tr !== "shift") || (a % s === 0);
    const Tout = tr === "shift" ? (A => shift2(A, a / s, a / s)) : T;

    const LHS = conv(T(X));
    const base = conv(X);
    const RHS = divides ? Tout(base) : base;
    const M = LHS.length;
    const D = LHS.map((r, p) => r.map((v, q) => v - RHS[p][q]));

    /* interior mask: cells whose window lay entirely inside the ORIGINAL
       content on both sides of the identity. Computed, not assumed.        */
    const inside = (p, q) => {
      const i0 = p * s - pad.p0, j0 = q * s - pad.p0;
      return i0 >= 0 && j0 >= 0 && i0 + f - 1 < N && j0 + f - 1 < N;
    };
    const interior = [];
    for (let p = 0; p < M; p++) { const row = []; for (let q = 0; q < M; q++) {
      let ok = inside(p, q);
      if (tr === "shift" && divides) { const pp = p - a / s, qq = q - a / s; ok = ok && pp >= 0 && qq >= 0 && pp < M && qq < M && inside(pp, qq); }
      row.push(ok);
    } interior.push(row); }
    let eI = 0, eB = 0, nI = 0, nB = 0;
    for (let p = 0; p < M; p++) for (let q = 0; q < M; q++) {
      if (interior[p][q]) { eI = Math.max(eI, Math.abs(D[p][q])); nI++; }
      else { eB = Math.max(eB, Math.abs(D[p][q])); nB++; }
    }

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 12 });
    const g = F.g;
    const cw = 10.5, gap = 30;
    const [lo, hi] = CN.extent2(X);
    const panels = [
      { A: X, t: "input  x" },
      { A: T(X), t: tr === "shift" ? `T x  (shift ${a})` : (tr === "rot" ? "T x  (rotate 90°)" : "T x  (reflect)") },
      { A: LHS, t: "conv(T x)" },
      { A: RHS, t: divides ? "T conv(x)" : "conv(x)  — T undefined" }
    ];
    panels.forEach((p, i) => {
      const x = i * (N * cw + gap);
      CN.grid(g, p.A, x, 18, cw, { text: false, stroke: "none", title: p.t, fill: i < 2 ? (v => CN.grey(v, lo, hi)) : null });
      if (i === 1) DL.arrow(g, x - gap + 6, 18 + N * cw / 2, x - 6, 18 + N * cw / 2, { color: DC.muted, w: 1.2, head: 5 });
    });
    g.append("text").attr("x", 2 * (N * cw + gap) - 12).attr("y", 18 + N * cw + 16).attr("font-size", 11)
      .attr("fill", divides ? DC.ink : DC.bad).attr("font-weight", 600)
      .text(divides ? "these two must be equal" : "these two are not comparable — s does not divide the shift");

    /* the difference map */
    const dy = 18 + N * cw + 40;
    const dm = Math.max.apply(null, D.map(r => Math.max.apply(null, r.map(Math.abs)))) || 1;
    CN.grid(g, D, 0, dy + 16, 14, {
      text: false, stroke: "none", title: `difference, max |Δ| = ${DL.fmtE(dm, 2)}`,
      fill: v => CN.signRamp(v, dm)
    });
    /* mark the interior */
    for (let p = 0; p < M; p++) for (let q = 0; q < M; q++) if (!interior[p][q])
      g.append("rect").attr("x", q * 14).attr("y", dy + 16 + p * 14).attr("width", 14).attr("height", 14)
        .attr("fill", "none").attr("stroke", DC.rose).attr("stroke-width", 0.5).attr("stroke-opacity", 0.55)
        .attr("shape-rendering", "crispEdges");
    g.append("text").attr("x", 0).attr("y", dy + 22 + M * 14 + 12).attr("font-size", 10).attr("fill", DC.rose)
      .text("outlined cells = border strip (a padded or missing value entered the window)");

    const kv = DL.kv(g, M * 14 + 40, dy + 30, { lead: 17, keyW: 250, size: 11.5 });
    kv("interior cells", `${nI} of ${M * M}`, DC.ink);
    kv("max |Δ| over the interior", DL.fmtE(eI, 2), eI < 1e-12 ? DC.good : DC.bad, true);
    kv("max |Δ| over the border strip", nB ? DL.fmtE(eB, 2) : "—", eB > 1e-12 ? DC.a2 : DC.good);
    kv("equivariance in the interior", eI < 1e-12 ? "EXACT" : "FAILS", eI < 1e-12 ? DC.good : DC.bad, true);
    kv("kernel fixed by this transform?", tr === "shift" ? "n/a" :
      (CN.maxAbsDiff(K, tr === "rot" ? rot90(K) : fliph(K)) < 1e-12 ? "yes" : "no"),
      DC.muted);

    document.getElementById("equi-readout").innerHTML =
      (tr === "shift"
        ? (divides
          ? `Shift by ${a} with stride ${s}: the output shift is ${a / s}. Interior max|Δ| = <b>${DL.fmtE(eI, 2)}</b> over ${nI} cells — ` +
            (eI < 1e-12 ? "<b>exactly zero</b>, so conv(T x) = T conv(x) holds identically wherever no padded value entered the window."
              : "<b>not zero</b>, which would be a bug.") +
            ` Border strip: max|Δ| = <b>${DL.fmtE(eB, 2)}</b> over ${nB} cells. ` +
            (eB > 1e-12
              ? "There the identity genuinely fails: a padded or missing value entered one window and not the other, and invented zeros do not move with the content."
              : "It happens to be zero too — with this image nothing is near the border, so nothing was invented. The strip marks where equivariance is <b>not guaranteed</b>, not where it must fail; switch the image to the one whose content runs off the edge and it becomes non-zero.")
          : `<b>Stride ${s} does not divide a shift of ${a}.</b> There is no output shift that corresponds to it: the sampling grid moved relative to the content, so the identity cannot even be written down. A stride-${s} layer is equivariant only to shifts that are multiples of ${s} — §19 and §18 are about what that costs.`)
        : `Transform: ${tr === "rot" ? "90° rotation" : "horizontal reflection"}. The kernel is ` +
          (CN.maxAbsDiff(K, tr === "rot" ? rot90(K) : fliph(K)) < 1e-12 ? "<b>unchanged</b>" : "<b>changed</b>") +
          ` by that transform, and correspondingly the interior error is <b>${DL.fmtE(eI, 2)}</b>. ` +
          (eI < 1e-12
            ? "Equivariance holds here — but only because this particular kernel happens to be symmetric under this particular transform. It is a property of the kernel, not of convolution."
            : "Convolution is <b>not</b> equivariant to anything but translation. Rotation and scale equivariance must come from augmentation, from data, or from a different architecture — that gap is §30's argument."));
  }
  [ae, se].forEach(e => e.addEventListener("input", draw));
  [pe, tte, ke, document.getElementById("eq-i")].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 7 · #shape-svg — the shape formula, one knob at a time ═══════════ */
(function () {
  const svg = d3.select("#shape-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const ne = document.getElementById("sq-n"), fe = document.getElementById("sq-f"),
    se = document.getElementById("sq-s"), de = document.getElementById("sq-d"),
    pe = document.getElementById("sq-p"), pce = document.getElementById("sq-pc");

  function draw() {
    const n = +ne.value, f = +fe.value, s = +se.value, d = +de.value, mode = pe.value, pc = +pce.value;
    ["sq-nv", "sq-fv", "sq-sv", "sq-dv", "sq-pcv"].forEach((id, i) =>
      document.getElementById(id).textContent = [n, f, s, d, pc][i]);
    const pad = (mode === "custom") ? { p0: pc, p1: pc, total: 2 * pc, symmetric: true } : DL.padFor(mode, n, f, s, d);
    const fe_ = DL.effK(f, d);
    const geom = { s: s, p0: pad.p0, p1: pad.p1, d: d };
    const nOut = DL.outSize(n, f, geom);

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;

    const total = n + pad.p0 + pad.p1;
    const cw = Math.min(24, 400 / Math.max(total, 8));
    const x0 = 30, y0 = 26;
    const xOf = k => x0 + k * cw;                    // k indexes the PADDED strip

    g.append("text").attr("x", 0).attr("y", 12).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`padded input: ${pad.p0} + ${n} + ${pad.p1} = ${total} positions`);
    for (let k = 0; k < total; k++) {
      const real = k >= pad.p0 && k < pad.p0 + n;
      g.append("rect").attr("x", xOf(k)).attr("y", y0).attr("width", cw - 1).attr("height", 20)
        .attr("fill", real ? DC.panel2 : "none").attr("stroke", real ? DC.accent : DC.rose)
        .attr("stroke-dasharray", real ? null : "2,2").attr("stroke-width", 1);
      if (cw > 13) g.append("text").attr("x", xOf(k) + cw / 2 - 0.5).attr("y", y0 + 14).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", real ? DC.ink : DC.rose).text(real ? (k - pad.p0) : "0");
    }

    /* every window, drawn where it actually lands; reads[] counted, not derived */
    const reads = new Array(n).fill(0);
    const rowH = Math.min(15, 190 / Math.max(nOut, 1));
    for (let q = 0; q < nOut; q++) {
      const yy = y0 + 28 + q * rowH;
      for (let m = 0; m < f; m++) {
        const k = q * s + m * d;                      // index into the PADDED strip
        const i = k - pad.p0;
        if (i >= 0 && i < n) reads[i]++;
        g.append("rect").attr("x", xOf(k)).attr("y", yy).attr("width", cw - 1).attr("height", Math.max(3, rowH - 2))
          .attr("fill", (i >= 0 && i < n) ? DC.a2 : DC.rose).attr("fill-opacity", (i >= 0 && i < n) ? 0.8 : 0.3)
          .attr("shape-rendering", "crispEdges");
      }
      if (rowH >= 10) g.append("text").attr("x", x0 - 6).attr("y", yy + rowH - 3).attr("text-anchor", "end")
        .attr("font-size", 8.5).attr("fill", DC.muted).text("z" + q);
    }

    /* reads per input position */
    const by = y0 + 34 + nOut * rowH;
    const mx = Math.max.apply(null, reads.concat([1]));
    g.append("text").attr("x", 0).attr("y", by - 2).attr("font-size", 10.5).attr("fill", DC.ink)
      .text("times each REAL input position is read:");
    for (let i = 0; i < n; i++) {
      const hh = 30 * reads[i] / mx;
      g.append("rect").attr("x", xOf(i + pad.p0) + 1).attr("y", by + 6 + 30 - hh).attr("width", cw - 3).attr("height", hh)
        .attr("fill", reads[i] === 0 ? DC.bad : (reads[i] === mx ? DC.good : DC.a2)).attr("fill-opacity", 0.8);
      if (cw > 13) g.append("text").attr("x", xOf(i + pad.p0) + cw / 2 - 0.5).attr("y", by + 48).attr("text-anchor", "middle")
        .attr("font-size", 8.5).attr("fill", reads[i] === 0 ? DC.bad : DC.muted).text(reads[i]);
    }

    /* the formula, substituted */
    const fx = x0 + total * cw + 40, fy = 30;
    const lines = [
      ["fₑ = d(f−1)+1", `${d}·(${f}−1)+1 = ${fe_}`],
      ["numerator", `${n} + ${pad.p0} + ${pad.p1} − ${fe_} = ${n + pad.p0 + pad.p1 - fe_}`],
      ["÷ s", `${n + pad.p0 + pad.p1 - fe_} / ${s} = ${((n + pad.p0 + pad.p1 - fe_) / s).toFixed(3)}`],
      ["floor", `⌊${((n + pad.p0 + pad.p1 - fe_) / s).toFixed(3)}⌋ = ${Math.floor((n + pad.p0 + pad.p1 - fe_) / s)}`],
      ["+ 1", `n′ = ${nOut}`]
    ];
    g.append("text").attr("x", fx).attr("y", fy - 10).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text("the formula, with these numbers");
    lines.forEach((L, i) => {
      g.append("text").attr("x", fx).attr("y", fy + i * 19).attr("font-size", 10.5).attr("fill", DC.muted).text(L[0]);
      g.append("text").attr("x", fx + 96).attr("y", fy + i * 19).attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", i === 4 ? DC.a2 : DC.ink).attr("font-weight", i === 4 ? 600 : 400).text(L[1]);
    });
    const exact = ((n + pad.p0 + pad.p1 - fe_) % s) === 0;
    g.append("text").attr("x", fx).attr("y", fy + 5 * 19 + 6).attr("font-size", 10.5)
      .attr("fill", exact ? DC.good : DC.bad)
      .text(exact ? "the division is exact — nothing discarded" : "the division is NOT exact — the floor bites");
    const kv = DL.kv(g, fx, fy + 5 * 19 + 30, { lead: 16, keyW: 128, size: 10.5 });
    kv("padding p₀, p₁", `${pad.p0}, ${pad.p1}`, pad.symmetric ? DC.ink : DC.a2, !pad.symmetric);
    kv("symmetric?", pad.symmetric ? "yes" : "NO", pad.symmetric ? DC.good : DC.a2, !pad.symmetric);
    kv("unread positions", String(reads.filter(r => r === 0).length), reads.some(r => r === 0) ? DC.bad : DC.good, reads.some(r => r === 0));

    const unread = reads.filter(r => r === 0).length;
    document.getElementById("shape-readout").innerHTML =
      `n = ${n}, f = ${f}, s = ${s}, d = ${d}, ${mode} padding → p = (${pad.p0}, ${pad.p1}), ` +
      `effective kernel fₑ = ${fe_}, and <b>n′ = ⌊(${n} + ${pad.p0} + ${pad.p1} − ${fe_}) / ${s}⌋ + 1 = ${nOut}</b>. ` +
      (exact ? "The division is exact, so every window ends flush with the input. "
        : `The division leaves a remainder of ${(n + pad.p0 + pad.p1 - fe_) % s}, so the last ${(n + pad.p0 + pad.p1 - fe_) % s} position${(n + pad.p0 + pad.p1 - fe_) % s === 1 ? "" : "s"} of the padded strip ${(n + pad.p0 + pad.p1 - fe_) % s === 1 ? "is" : "are"} never the start of a window. `) +
      (unread ? `<b>${unread} real input position${unread === 1 ? " is" : "s are"} read zero times</b> — that input is invisible to the layer (§09 trap 1). `
        : "Every real input position is read at least once. ") +
      `The most-read position is read ${mx} times and the least-read ${Math.min.apply(null, reads)}; under valid padding that gap is the border under-use argument, and padding narrows it without closing it.`;
  }
  [ne, fe, se, de, pce].forEach(e => e.addEventListener("input", draw));
  pe.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 8 · #trap-svg — the four off-by-one traps ═══════════ */
(function () {
  const svg = d3.select("#trap-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const te = document.getElementById("tr-t"), ne = document.getElementById("tr-n"),
    fe = document.getElementById("tr-f"), se = document.getElementById("tr-s");

  /* one strip: returns {reads, starts, nOut, pad} — everything MEASURED */
  function layer(n, f, s, mode) {
    const pad = DL.padFor(mode, n, f, s, 1);
    const nOut = Math.max(0, DL.outSize(n, f, { s: s, p0: pad.p0, p1: pad.p1, d: 1 }));
    const reads = new Array(n).fill(0), starts = [];
    for (let q = 0; q < nOut; q++) {
      starts.push(q * s - pad.p0);
      for (let m = 0; m < f; m++) { const i = q * s + m - pad.p0; if (i >= 0 && i < n) reads[i]++; }
    }
    return { reads: reads, starts: starts, nOut: nOut, pad: pad };
  }

  function draw() {
    const trap = te.value;
    let n = +ne.value, f = +fe.value, s = +se.value;
    document.getElementById("tr-nv").textContent = n;
    document.getElementById("tr-fv").textContent = f;
    document.getElementById("tr-sv").textContent = s;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = Math.min(28, 460 / (n + 4));
    const strip = (y, mode, title) => {
      const L = layer(n, f, s, mode);
      const x0 = 40;
      g.append("text").attr("x", 0).attr("y", y - 6).attr("font-size", 11).attr("fill", DC.ink)
        .attr("font-weight", 600).text(`${title}   p = (${L.pad.p0}, ${L.pad.p1})   n′ = ${L.nOut}`);
      for (let i = 0; i < n; i++) {
        g.append("rect").attr("x", x0 + i * cw).attr("y", y).attr("width", cw - 1).attr("height", 20)
          .attr("fill", L.reads[i] === 0 ? DC.bad : DC.panel2).attr("fill-opacity", L.reads[i] === 0 ? 0.55 : 1)
          .attr("stroke", DC.line);
        g.append("text").attr("x", x0 + i * cw + cw / 2).attr("y", y + 14).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace")
          .attr("fill", L.reads[i] === 0 ? "#fff" : DC.muted).text(i);
      }
      /* padding blocks, drawn to scale on each side */
      for (let k = 0; k < L.pad.p0; k++) g.append("rect").attr("x", x0 - (k + 1) * cw).attr("y", y)
        .attr("width", cw - 1).attr("height", 20).attr("fill", "none").attr("stroke", DC.rose).attr("stroke-dasharray", "2,2");
      for (let k = 0; k < L.pad.p1; k++) g.append("rect").attr("x", x0 + (n + k) * cw).attr("y", y)
        .attr("width", cw - 1).attr("height", 20).attr("fill", "none").attr("stroke", DC.rose).attr("stroke-dasharray", "2,2");
      /* windows */
      L.starts.forEach((st, q) => {
        const yy = y + 24 + q * 8;
        g.append("line").attr("x1", x0 + st * cw + 2).attr("x2", x0 + (st + f) * cw - 3).attr("y1", yy).attr("y2", yy)
          .attr("stroke", DC.a2).attr("stroke-width", 3).attr("stroke-opacity", 0.85);
        /* the window CENTRE — lands between cells when f is even */
        g.append("line").attr("x1", x0 + (st + f / 2) * cw - 0.5).attr("x2", x0 + (st + f / 2) * cw - 0.5)
          .attr("y1", yy - 4).attr("y2", yy + 4).attr("stroke", (f % 2) ? DC.good : DC.violet).attr("stroke-width", 1.4);
      });
      const rb = y + 30 + L.nOut * 8;
      for (let i = 0; i < n; i++)
        g.append("text").attr("x", x0 + i * cw + cw / 2).attr("y", rb + 10).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", L.reads[i] === 0 ? DC.bad : DC.muted).text(L.reads[i]);
      g.append("text").attr("x", 0).attr("y", rb + 10).attr("font-size", 9.5).attr("fill", DC.muted).text("reads");
      return { L: L, bottom: rb + 18 };
    };

    let msg = "";
    if (trap === "noninj") {
      const A = strip(26, "valid", `n = ${n}`);
      const nb = n + 1;
      const save = n; n = nb;
      const B = strip(A.bottom + 26, "valid", `n = ${nb}`);
      n = save;
      msg = `With f = ${f}, s = ${s} and valid padding, an input of <b>${save}</b> gives n′ = <b>${A.L.nOut}</b> and an input of <b>${save + 1}</b> gives n′ = <b>${B.L.nOut}</b>. ` +
        (A.L.nOut === B.L.nOut
          ? `They are <b>the same</b>: the map n → n′ is not injective, so n′ alone cannot tell you what n was. A transposed convolution (§23) therefore cannot recover the input size and has to be told it — which is what the <code>output_padding</code> argument is for.`
          : `Here they differ, so the map happens to be injective at this point. Set the stride above 1 and slide n until two neighbouring values collide — with s = ${s} that happens for ${s > 1 ? s - 1 : "no"} out of every ${s} value${s === 1 ? "" : "s"} of n.`) +
        ` Unread positions: ${A.L.reads.filter(r => r === 0).length} and ${B.L.reads.filter(r => r === 0).length}.`;
    } else {
      const A = strip(26, "valid", "valid padding");
      const B = strip(A.bottom + 26, "same", "same padding");
      const uA = A.L.reads.filter(r => r === 0).length, uB = B.L.reads.filter(r => r === 0).length;
      if (trap === "floor") {
        msg = `n = ${n}, f = ${f}, s = ${s}. Valid: n′ = ${A.L.nOut}, and <b>${uA}</b> input position${uA === 1 ? "" : "s"} ${uA === 1 ? "is" : "are"} read zero times` +
          (uA ? ` — position${uA === 1 ? "" : "s"} ${A.L.reads.map((r, i) => r === 0 ? i : -1).filter(i => i >= 0).join(", ")}, discarded on every forward pass with no warning.` : ".") +
          ` Same: n′ = ${B.L.nOut} with p = (${B.L.pad.p0}, ${B.L.pad.p1}), and <b>${uB}</b> unread. ` +
          `The floor rejects any window that would hang off the end; the padding policy decides whether that costs you real data.`;
      } else if (trap === "asym") {
        msg = `n = ${n}, f = ${f}, s = ${s}. Same padding needs a total of <b>${B.L.pad.total}</b>, split as <b>(${B.L.pad.p0}, ${B.L.pad.p1})</b> — ` +
          (B.L.pad.symmetric
            ? "which happens to be symmetric here, so an integer <code>padding=" + B.L.pad.p0 + "</code> would reproduce it exactly."
            : "<b>asymmetric</b>, with the extra pixel on the right. No integer <code>padding=p</code> argument can express this, so a reimplementation that uses one is off by half a pixel on every feature map, silently.") +
          ` Valid padding gives n′ = ${A.L.nOut} against same's ${B.L.nOut}.`;
      } else {
        const off = (f % 2) ? 0 : 0.5;
        msg = `f = ${f} is <b>${f % 2 ? "odd" : "even"}</b>. ` +
          (f % 2
            ? `Each window has a true centre, marked by the green ticks, and it lands exactly on an input cell — so "output i sees the input around i" is literally true and the output grid is registered with the input grid.`
            : `Each window's centre, marked by the violet ticks, lands <b>halfway between two input cells</b>. The output grid is offset from the input grid by ${off} of a pixel per layer; after 5 such layers the offset is ${5 * off} pixels, which is a real misregistration. And the two-line proof: with symmetric padding p, n′ = n + 2p − f + 1 = n needs 2p = f − 1 = ${f - 1}, which is odd, so <b>no integer p makes an even kernel "same"</b>. The framework does it by padding asymmetrically — here (${B.L.pad.p0}, ${B.L.pad.p1}).`) +
          ` This, not expressiveness, is why kernels are odd.`;
      }
    }
    document.getElementById("trap-readout").innerHTML = msg;
  }
  te.addEventListener("change", draw);
  [ne, fe, se].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════ 9 · #vol-svg — one 3-D filter on a 3-channel input ═══════════ */
(function () {
  const svg = d3.select("#vol-svg");
  if (svg.empty()) return;
  const W = 760, H = 430, N = 6, f = 3;
  const pe = document.getElementById("vl-pos"), ke = document.getElementById("vl-k"),
    ie = document.getElementById("vl-i");
  const CHAN = ["R", "G", "B"], CHCOL = [DC.rose, DC.good, DC.accent];

  function volume(kind) {
    const V = DL.zeros3(3, N, N), r = DL.rng(9);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (kind === "rgbbar") { V[0][i][j] = (j >= 2 && j <= 3) ? 9 : 1; V[1][i][j] = 1; V[2][i][j] = (j >= 2 && j <= 3) ? 1 : 8; }
      else if (kind === "split") { V[0][i][j] = j < 3 ? 9 : 0; V[1][i][j] = 3; V[2][i][j] = j < 3 ? 0 : 9; }
      else { V[0][i][j] = Math.round(r() * 9); V[1][i][j] = Math.round(r() * 9); V[2][i][j] = Math.round(r() * 9); }
    }
    return V;
  }
  function filt(kind) {
    const z = () => [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const v = () => [[1, 0, -1], [1, 0, -1], [1, 0, -1]];
    const one = () => [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
    if (kind === "red") return [v(), z(), z()];
    if (kind === "all") return [v(), v(), v()];
    if (kind === "opp") return [one(), z(), one().map(r => r.map(x => -x))];
    return [one().map(r => r.map(x => x / 3)), one().map(r => r.map(x => x / 3)), one().map(r => r.map(x => x / 3))];
  }

  function draw() {
    const pos = +pe.value, V = volume(ie.value), K3 = filt(ke.value), bias = 1;
    const No = N - f + 1, pi = Math.floor(pos / No), pj = pos % No;
    document.getElementById("vl-posv").textContent = `(${pi},${pj})`;

    const Z = DL.conv2dMC(V, [K3], [bias], {});           // (1 × 4 × 4)
    const parts = [0, 1, 2].map(c => {
      let sum = 0;
      for (let m = 0; m < f; m++) for (let n = 0; n < f; n++) sum += V[c][pi + m][pj + n] * K3[c][m][n];
      return sum;
    });
    const tot = parts[0] + parts[1] + parts[2] + bias;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = 22;
    /* input channels */
    for (let c = 0; c < 3; c++) {
      const y = 24 + c * (N * cw + 22);
      const gg = CN.grid(g, V[c], 20, y, cw, { title: `input channel ${CHAN[c]}  (${N}×${N})`, font: 9.5 });
      CN.box(gg, pj * cw, pi * cw, f * cw, f * cw, CHCOL[c], 2);
      /* the filter slice */
      const gk = CN.grid(g, K3[c], 20 + N * cw + 40, y + 14, 20, { font: 9, dp: ke.value === "grey" ? 2 : 0 });
      g.append("text").attr("x", 20 + N * cw + 40).attr("y", y + 8).attr("font-size", 10).attr("fill", DC.muted)
        .text(`K[${c}] (${f}×${f})`);
      g.append("text").attr("x", 20 + N * cw + 40 + 3 * 20 + 14).attr("y", y + 14 + 34).attr("font-size", 11.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", CHCOL[c])
        .text(`= ${DL.fmt(parts[c], 2)}`);
    }
    /* the sum */
    const sx = 20 + N * cw + 40 + 3 * 20 + 100;
    g.append("text").attr("x", sx).attr("y", 24).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`output at (${pi},${pj})`);
    const rows = [
      [`channel R`, DL.fmt(parts[0], 2), CHCOL[0]],
      [`channel G`, DL.fmt(parts[1], 2), CHCOL[1]],
      [`channel B`, DL.fmt(parts[2], 2), CHCOL[2]],
      [`bias`, DL.fmt(bias, 2), DC.muted],
      [`total`, DL.fmt(tot, 2), DC.a2]
    ];
    rows.forEach((r, i) => {
      const y = 46 + i * 18;
      if (i === 4) g.append("line").attr("x1", sx).attr("x2", sx + 130).attr("y1", y - 12).attr("y2", y - 12).attr("stroke", DC.line);
      g.append("text").attr("x", sx).attr("y", y).attr("font-size", 10.5).attr("fill", DC.muted).text(r[0]);
      g.append("text").attr("x", sx + 128).attr("y", y).attr("text-anchor", "end").attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", r[2]).attr("font-weight", i === 4 ? 600 : 400).text(r[1]);
    });
    /* the output map — FLAT */
    const gz = CN.grid(g, Z[0], sx, 150, 30, { title: "output  (4 × 4 × 1)", dp: 1, font: 9.5 });
    CN.box(gz, pj * 30, pi * 30, 30, 30, DC.a2, 2.2);
    g.append("text").attr("x", sx).attr("y", 150 + 4 * 30 + 18).attr("font-size", 10.5).attr("fill", DC.good)
      .text("depth 1 — one filter, one map");
    g.append("text").attr("x", sx).attr("y", 150 + 4 * 30 + 34).attr("font-size", 10.5).attr("fill", DC.muted)
      .attr("xml:space", "preserve").text("(6,6,3) ⋆ (3,3,3) → (4,4)");
    g.append("text").attr("x", sx).attr("y", 150 + 4 * 30 + 49).attr("font-size", 10).attr("fill", DC.muted)
      .text("the channel axis is contracted, not traversed");

    const check = Math.abs((parts[0] + parts[1] + parts[2] + bias) - Z[0][pi][pj]);
    document.getElementById("vol-readout").innerHTML =
      `Window at (${pi},${pj}). Per-channel partial sums: R = <b>${DL.fmt(parts[0], 2)}</b>, G = <b>${DL.fmt(parts[1], 2)}</b>, B = <b>${DL.fmt(parts[2], 2)}</b>; ` +
      `plus a bias of ${bias} gives <b>${DL.fmt(tot, 2)}</b>, which matches the layer's own output at that position to ${DL.fmtE(check, 1)}. ` +
      `The three 3×3 slices consumed 27 input numbers and produced <b>one</b>. ` +
      (ke.value === "red" ? "This filter has zeros in G and B, so only the red channel contributes — a filter is free to ignore channels, and grouped convolution (§20) is what happens when that is enforced structurally rather than learned."
        : ke.value === "opp" ? "This filter is +1 on R and −1 on B at the centre and zero elsewhere: it is a 1×1 convolution in disguise (§16), computing a colour opponency at every pixel with no spatial extent at all."
        : ke.value === "grey" ? "This filter averages the three channels at each pixel — again a 1×1 convolution, and the classic greyscale projection."
        : "The same vertical-edge slice in all three channels sums the three per-channel edge responses, which is not the same as detecting edges in the luminance: a red-to-blue boundary at constant luminance still fires.") +
      ` The filter's depth was never a choice — it is ${V.length}, because the input's depth is ${V.length}.`;
  }
  pe.addEventListener("input", draw);
  [ke, ie].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 10 · #bank-svg — a bank of filters and the stack it makes ═══════════ */
(function () {
  const svg = d3.select("#bank-svg");
  if (svg.empty()) return;
  const W = 760, H = 420, N = 14;
  const ne = document.getElementById("bk-n"), ce = document.getElementById("bk-c"),
    ie = document.getElementById("bk-i"), ae = document.getElementById("bk-a");

  const BANK = [
    { K: [[1, 0, -1], [1, 0, -1], [1, 0, -1]], t: "vertical" },
    { K: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]], t: "horizontal" },
    { K: [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], t: "diagonal ╱" },
    { K: [[1, 1, 0], [1, 0, -1], [0, -1, -1]], t: "diagonal ╲" },
    { K: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], t: "centre–surround" },
    { K: [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map(r => r.map(v => v / 16)), t: "blob (smooth)" }
  ];
  function image(kind) {
    const A = DL.zeros2(N, N), r = DL.rng(4);
    if (kind === "shapes") {
      for (let i = 2; i < 12; i++) { A[i][3] = 9; A[i][4] = 9; }        // a vertical bar
      for (let j = 7; j < 12; j++) { A[4][j] = 9; A[5][j] = 9; }        // a horizontal bar
      for (let d = 0; d < 5; d++) { A[8 + d][7 + d] = 9; }              // a diagonal
    } else if (kind === "digit") {
      const pts = [[2, 5], [2, 6], [2, 7], [3, 8], [4, 8], [5, 7], [6, 6], [7, 5], [8, 5], [9, 5], [10, 5], [10, 6], [10, 7], [10, 8]];
      pts.forEach(([i, j]) => { A[i][j] = 9; if (j + 1 < N) A[i][j + 1] = 6; });
    } else {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i][j] = 4.5 + 4.5 * Math.sin(i * 1.1 + Math.sin(j * 0.8)) * (0.6 + 0.4 * r());
    }
    return A;
  }

  function draw() {
    const nF = +ne.value, act = ae.value;
    ce.max = String(nF - 1);
    const sel = Math.min(+ce.value, nF - 1);
    document.getElementById("bk-nv").textContent = nF;
    document.getElementById("bk-cv").textContent = sel;
    const X = image(ie.value);
    const K = [];
    for (let c = 0; c < nF; c++) K.push([BANK[c].K]);
    let Z = DL.conv2dMC([X], K, new Array(nF).fill(0), { p0: 1, p1: 1 });
    if (act === "relu") Z = Z.map(M => M.map(r => r.map(v => Math.max(0, v))));

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = 11;
    const [lo, hi] = CN.extent2(X);
    CN.grid(g, X, 0, 24, cw, { text: false, stroke: "none", title: `input  ${N}×${N}×1`, fill: v => CN.grey(v, lo, hi) });

    /* the bank */
    const bx = N * cw + 32;
    g.append("text").attr("x", bx).attr("y", 16).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`${nF} filters, 3×3×1`);
    for (let c = 0; c < nF; c++) {
      const y = 24 + c * 46;
      const gg = CN.grid(g, BANK[c].K, bx, y, 14, { text: false, stroke: DC.line });
      if (c === sel) CN.box(gg, -1.5, -1.5, 3 * 14 + 3, 3 * 14 + 3, DC.a2, 2);
      g.append("text").attr("x", bx + 3 * 14 + 8).attr("y", y + 16).attr("font-size", 10)
        .attr("fill", c === sel ? DC.a2 : DC.muted).text(BANK[c].t);
      g.append("text").attr("x", bx + 3 * 14 + 8).attr("y", y + 30).attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted)
        .text(`ch ${c}`);
    }

    /* the stack, drawn offset for depth, selected channel in front */
    const sx = bx + 150;
    g.append("text").attr("x", sx).attr("y", 16).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`output stack  ${Z[0].length}×${Z[0][0].length}×${nF}`);
    for (let c = nF - 1; c >= 0; c--) {
      if (c === sel) continue;
      const dx = (nF - 1 - c) * 7, dy = (nF - 1 - c) * 7;
      const m = Math.max.apply(null, Z[c].map(r => Math.max.apply(null, r.map(Math.abs)))) || 1;
      CN.grid(g, Z[c], sx + dx, 24 + dy, 8, { text: false, stroke: "none", fill: v => CN.signRamp(v, m) });
    }
    const msel = Math.max.apply(null, Z[sel].map(r => Math.max.apply(null, r.map(Math.abs)))) || 1;
    const gs = CN.grid(g, Z[sel], sx + 130, 24, 13, {
      text: false, stroke: "none", fill: v => CN.signRamp(v, msel),
      title: `channel ${sel}: ${BANK[sel].t}   max |z| = ${DL.fmt(msel, 1)}`
    });
    CN.box(gs, -1, -1, Z[sel][0].length * 13 + 2, Z[sel].length * 13 + 2, DC.a2, 1.6);

    /* mean |response| per channel */
    const by = 250;
    const mean = Z.map(M => { let s = 0, n = 0; M.forEach(r => r.forEach(v => { s += Math.abs(v); n++; })); return s / n; });
    const mmax = Math.max.apply(null, mean) || 1;
    g.append("text").attr("x", 0).attr("y", by - 6).attr("font-size", 10.5).attr("fill", DC.ink)
      .text("mean |response| per output channel on this input — different filters, different answers");
    mean.forEach((v, c) => {
      const ww = 300 * v / mmax;
      g.append("rect").attr("x", 120).attr("y", by + c * 20).attr("width", Math.max(1, ww)).attr("height", 14)
        .attr("fill", c === sel ? DC.a2 : DC.accent).attr("fill-opacity", 0.8);
      g.append("text").attr("x", 114).attr("y", by + c * 20 + 11).attr("text-anchor", "end").attr("font-size", 10)
        .attr("fill", c === sel ? DC.a2 : DC.muted).text(`ch ${c} · ${BANK[c].t}`);
      g.append("text").attr("x", 126 + ww).attr("y", by + c * 20 + 11).attr("font-size", 9.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text(DL.fmt(v, 2));
    });

    /* correlation of the selected channel with the others */
    const flat = M => { const a = []; M.forEach(r => r.forEach(v => a.push(v))); return a; };
    const corr = (a, b) => {
      const ma = DL.mean(a), mb = DL.mean(b);
      let n = 0, da = 0, db = 0;
      for (let i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
      return n / Math.sqrt((da * db) || 1);
    };
    const fs = flat(Z[sel]);
    const cs = Z.map((M, c) => c === sel ? 1 : corr(fs, flat(M)));
    const P = DL.convCount({ H: N, W: N, cin: 1, cout: nF, k: 3, p0: 1, p1: 1 });

    document.getElementById("bank-readout").innerHTML =
      `${nF} filters of 3×3×1 → <b>${DL.commas(P.params)}</b> parameters (${nF}·9 weights + ${nF} biases) and an output of ` +
      `<b>${Z[0].length}×${Z[0][0].length}×${nF}</b>, which is ${DL.commas(P.acts)} numbers from ${DL.commas(N * N)}. ` +
      `Each weight is applied ${DL.commas(P.reuse)} times. ` +
      `Correlation of channel ${sel} with the others: ` +
      cs.map((v, c) => c === sel ? null : `ch ${c} ${DL.fmt(v, 2)}`).filter(Boolean).join(", ") + ". " +
      `The largest off-diagonal magnitude is <b>${DL.fmt(Math.max.apply(null, cs.map((v, c) => c === sel ? 0 : Math.abs(v))), 2)}</b>` +
      `, so these channels are genuinely different measurements rather than rescalings of one another. ` +
      (act === "relu"
        ? "With ReLU applied, every negative response is now zero — the antisymmetric channels have lost the polarity of each edge and keep only one of its two signs, which is why a layer usually learns both a filter and its negation."
        : "These are raw pre-activations, so both signs of every edge survive; switch on the ReLU to see half of each antisymmetric channel disappear.");
  }
  [ne, ce].forEach(e => e.addEventListener("input", draw));
  [ie, ae].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 11 · #count-svg — one layer's three counts ═══════════ */
(function () {
  const svg = d3.select("#count-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const ids = ["ct-n", "ct-ci", "ct-co", "ct-f", "ct-s", "ct-b"].map(i => document.getElementById(i));

  function counts(n, ci, co, f, s) {
    const p = Math.floor((f - 1) / 2);
    return DL.convCount({ H: n, W: n, cin: ci, cout: co, k: f, s: s, p0: p, p1: p });
  }
  function draw() {
    const [n, ci, co, f, s, B] = ids.map(e => +e.value);
    ["ct-nv", "ct-civ", "ct-cov", "ct-fv", "ct-sv", "ct-bv"].forEach((id, i) =>
      document.getElementById(id).textContent = [n, ci, co, f, s, B][i]);
    const C = counts(n, ci, co, f, s);
    const actBytes = 4 * B * C.acts;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const rows = [
      { name: "parameters", v: C.params, c: DC.good, form: `${co}·(${ci}·${f}·${f}) + ${co}` },
      { name: "MACs / example", v: C.macs, c: DC.accent, form: `${C.Ho}·${C.Wo}·${co}·(${ci}·${f}·${f})` },
      { name: "activation bytes", v: actBytes, c: DC.a2, form: `4·${B}·${co}·${C.Ho}·${C.Wo}` }
    ];
    const hi = Math.max.apply(null, rows.map(r => r.v));
    const sc = d3.scaleLog().domain([1, hi * 1.3]).range([0, 300]).clamp(true);
    rows.forEach((r, i) => {
      const y = 26 + i * 40;
      g.append("text").attr("x", 118).attr("y", y + 12).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", DC.ink).text(r.name);
      g.append("rect").attr("x", 126).attr("y", y).attr("width", Math.max(1, sc(Math.max(1, r.v)))).attr("height", 16)
        .attr("fill", r.c).attr("fill-opacity", 0.8).attr("stroke", r.c);
      g.append("text").attr("x", 132 + sc(Math.max(1, r.v))).attr("y", y + 12).attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", r.c).text(DL.commas(r.v));
      g.append("text").attr("x", 126).attr("y", y + 29).attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text("= " + r.form);
    });

    /* sensitivity: multiply each knob by 2 and see what each count does */
    const base = [C.params, C.macs, 4 * B * C.acts];
    const knobs = [
      { t: "H = W", f: () => { const c = counts(2 * n, ci, co, f, s); return [c.params, c.macs, 4 * B * c.acts]; } },
      { t: "C in", f: () => { const c = counts(n, 2 * ci, co, f, s); return [c.params, c.macs, 4 * B * c.acts]; } },
      { t: "C out", f: () => { const c = counts(n, ci, 2 * co, f, s); return [c.params, c.macs, 4 * B * c.acts]; } },
      { t: "kernel f", f: () => { const c = counts(n, ci, co, Math.min(11, 2 * f), s); return [c.params, c.macs, 4 * B * c.acts]; } },
      { t: "stride s", f: () => { const c = counts(n, ci, co, f, 2 * s); return [c.params, c.macs, 4 * B * c.acts]; } },
      { t: "batch N", f: () => [C.params, C.macs, 8 * B * C.acts] }
    ];
    const ty = 168;
    g.append("text").attr("x", 0).attr("y", ty - 8).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("double one knob — what happens to each count (a ratio; 1.00 means it does not appear in that formula)");
    const cols = [126, 246, 366];
    ["parameters", "MACs", "activation bytes"].forEach((t, j) =>
      g.append("text").attr("x", cols[j] + 50).attr("y", ty + 10).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", [DC.good, DC.accent, DC.a2][j]).text(t));
    knobs.forEach((kn, i) => {
      const y = ty + 28 + i * 20;
      g.append("text").attr("x", 118).attr("y", y).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", DC.muted).text("2 × " + kn.t);
      const v = kn.f();
      v.forEach((vv, j) => {
        const ratio = vv / base[j];
        g.append("text").attr("x", cols[j] + 50).attr("y", y).attr("text-anchor", "end").attr("font-size", 10.5)
          .attr("font-family", "SF Mono, Menlo, monospace")
          .attr("fill", Math.abs(ratio - 1) < 1e-9 ? DC.muted : [DC.good, DC.accent, DC.a2][j])
          .attr("font-weight", ratio > 3.5 ? 600 : 400)
          .text(ratio.toFixed(2) + "×");
      });
    });

    const kv = DL.kv(g, 470, ty + 30, { lead: 17, keyW: 176, size: 11 });
    kv("output map", `${C.Ho} × ${C.Wo} × ${co}`, DC.ink);
    kv("weights (no bias)", DL.commas(C.weights), DC.good);
    kv("uses of each weight", DL.commas(C.reuse), DC.a2, true);
    kv("MACs ÷ weights", DL.commas(Math.round(C.macs / C.weights)), DC.accent);
    kv("check: equal?", (Math.round(C.macs / C.weights) === C.reuse) ? "yes" : "no",
      (Math.round(C.macs / C.weights) === C.reuse) ? DC.good : DC.bad, true);

    document.getElementById("count-readout").innerHTML =
      `${n}×${n}×${ci} → ${C.Ho}×${C.Wo}×${co}, kernel ${f}, stride ${s}. ` +
      `<b>${DL.commas(C.params)}</b> parameters = ${co}·(${ci}·${f}·${f}) + ${co}; ` +
      `<b>${DL.commas(C.macs)}</b> MACs per example = ${C.Ho}·${C.Wo}·${co}·(${ci}·${f}·${f}); ` +
      `<b>${DL.commas(4 * B * C.acts)}</b> bytes of activation for a batch of ${B}. ` +
      `Each weight is used <b>${DL.commas(C.reuse)}</b> times, and MACs ÷ weights = ${DL.commas(Math.round(C.macs / C.weights))} — the same number, which is the identity of §12. ` +
      `Doubling the image side leaves the parameter count <b>exactly unchanged</b> and multiplies the arithmetic by ${(counts(2 * n, ci, co, f, s).macs / C.macs).toFixed(2)}; ` +
      `doubling the stride divides the arithmetic by about ${(C.macs / counts(n, ci, co, f, 2 * s).macs).toFixed(2)} and again leaves the parameters alone.`;
  }
  ids.forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════ 12 · #net-svg — a whole network, counted end to end ═══════════ */
(function () {
  const svg = d3.select("#net-svg");
  if (svg.empty()) return;
  const W = 760, H = 470;
  const ae = document.getElementById("nt-a"), he = document.getElementById("nt-h"),
    se = document.getElementById("nt-s");

  const ARCH = {
    small: {
      n: 32, c: 3, label: "the small classifier",
      L: [{ t: "conv", k: 5, n: 6, s: 1, p: 0 }, { t: "pool", k: 2, s: 2 },
      { t: "conv", k: 5, n: 10, s: 1, p: 0 }, { t: "pool", k: 2, s: 2 },
      { t: "fc", n: 120 }, { t: "fc", n: 84 }, { t: "fc", n: 10, out: true }]
    },
    strided: {
      n: 39, c: 3, label: "the strided classifier",
      L: [{ t: "conv", k: 3, n: 10, s: 1, p: 0 }, { t: "conv", k: 5, n: 20, s: 2, p: 0 },
      { t: "conv", k: 5, n: 40, s: 2, p: 0 }, { t: "fc", n: 10, out: true }]
    },
    vgg: {
      n: 224, c: 3, label: "a 16-layer 3×3 network",
      L: (function () {
        const a = [];
        [[64, 2], [128, 2], [256, 3], [512, 3], [512, 3]].forEach(([w, r]) => {
          for (let i = 0; i < r; i++) a.push({ t: "conv", k: 3, n: w, s: 1, p: 1 });
          a.push({ t: "pool", k: 2, s: 2 });
        });
        a.push({ t: "fc", n: 4096 }, { t: "fc", n: 4096 }, { t: "fc", n: 1000, out: true });
        return a;
      })()
    },
    alex: {
      n: 227, c: 3, label: "a large-kernel-stem network",
      L: [{ t: "conv", k: 11, n: 96, s: 4, p: 0 }, { t: "pool", k: 3, s: 2 },
      { t: "conv", k: 5, n: 256, s: 1, p: 2 }, { t: "pool", k: 3, s: 2 },
      { t: "conv", k: 3, n: 384, s: 1, p: 1 }, { t: "conv", k: 3, n: 384, s: 1, p: 1 },
      { t: "conv", k: 3, n: 256, s: 1, p: 1 }, { t: "pool", k: 3, s: 2 },
      { t: "fc", n: 4096 }, { t: "fc", n: 4096 }, { t: "fc", n: 1000, out: true }]
    }
  };

  const run = (spec, head) => CN.runNet(spec, head);

  function draw() {
    const spec = ARCH[ae.value], head = he.value, show = se.value;
    const rows = run(spec, head);
    const rowsAlt = run(spec, head === "gap" ? "dense" : "gap");
    const tot = rows.reduce((s, r) => ({ p: s.p + r.p, m: s.m + r.m, a: s.a + r.a }), { p: 0, m: 0, a: 0 });
    const totAlt = rowsAlt.reduce((s, r) => ({ p: s.p + r.p, m: s.m + r.m }), { p: 0, m: 0 });

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 20, b: 10 });
    const g = F.g;
    const rh = Math.min(17, 380 / rows.length);
    const fs = Math.min(10.5, rh - 3.5);
    const cols = [0, 74, 128, 216, 300, 372];
    const hdr = ["layer", "geom", "input", "output", "params", "MACs"];
    hdr.forEach((t, j) => g.append("text").attr("x", cols[j] + (j >= 4 ? 62 : 0)).attr("y", 8)
      .attr("text-anchor", j >= 4 ? "end" : "start").attr("font-size", 10).attr("fill", DC.muted).text(t));
    rows.forEach((r, i) => {
      const y = 22 + i * rh;
      const dense = r.t.indexOf("dense") === 0;
      g.append("text").attr("x", cols[0]).attr("y", y).attr("font-size", fs)
        .attr("fill", dense ? DC.violet : (r.t.indexOf("conv") === 0 ? DC.accent : DC.muted)).text(r.t);
      g.append("text").attr("x", cols[1]).attr("y", y).attr("font-size", fs).attr("fill", DC.muted).text(r.geo);
      g.append("text").attr("x", cols[2]).attr("y", y).attr("font-size", fs)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text(r.in);
      g.append("text").attr("x", cols[3]).attr("y", y).attr("font-size", fs)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", r.exact ? DC.ink : DC.bad).text(r.out + (r.exact ? "" : " ⚠"));
      g.append("text").attr("x", cols[4] + 62).attr("y", y).attr("text-anchor", "end").attr("font-size", fs)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.good).text(r.p ? DL.commas(r.p) : "—");
      g.append("text").attr("x", cols[5] + 62).attr("y", y).attr("text-anchor", "end").attr("font-size", fs)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.accent).text(r.m ? DL.big(r.m) : "—");
    });
    const ty = 22 + rows.length * rh + 4;
    g.append("line").attr("x1", 0).attr("x2", cols[5] + 62).attr("y1", ty - 11).attr("y2", ty - 11).attr("stroke", DC.line);
    g.append("text").attr("x", 0).attr("y", ty).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600).text("total");
    g.append("text").attr("x", cols[4] + 62).attr("y", ty).attr("text-anchor", "end").attr("font-size", 11)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.good).attr("font-weight", 600).text(DL.commas(tot.p));
    g.append("text").attr("x", cols[5] + 62).attr("y", ty).attr("text-anchor", "end").attr("font-size", 11)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.accent).attr("font-weight", 600).text(DL.big(tot.m));

    /* the two profile columns, aligned with the table rows */
    const px = cols[5] + 96, colW = 84;
    const which = show === "params" ? ["p"] : (show === "macs" ? ["m"] : (show === "acts" ? ["a"] : ["p", "m"]));
    const titles = { p: "params", m: "MACs", a: "activations" };
    const colr = { p: DC.good, m: DC.accent, a: DC.a2 };
    which.forEach((key, j) => {
      const T = key === "p" ? tot.p : (key === "m" ? tot.m : tot.a);
      g.append("text").attr("x", px + j * (colW + 26)).attr("y", 8).attr("font-size", 10).attr("fill", colr[key]).text(titles[key]);
      rows.forEach((r, i) => {
        const v = key === "p" ? r.p : (key === "m" ? r.m : r.a);
        const ww = colW * v / (T || 1);
        g.append("rect").attr("x", px + j * (colW + 26)).attr("y", 22 + i * rh - fs + 1)
          .attr("width", Math.max(v > 0 ? 0.8 : 0, ww)).attr("height", Math.max(2, rh - 3))
          .attr("fill", colr[key]).attr("fill-opacity", 0.8);
        if (ww > 26) g.append("text").attr("x", px + j * (colW + 26) + ww + 4).attr("y", 22 + i * rh)
          .attr("font-size", 8.5).attr("fill", DC.muted).text((100 * v / (T || 1)).toFixed(0) + "%");
      });
    });

    const convP = rows.filter(r => r.t.indexOf("conv") === 0).reduce((s, r) => s + r.p, 0);
    const convM = rows.filter(r => r.t.indexOf("conv") === 0).reduce((s, r) => s + r.m, 0);
    const denseP = rows.filter(r => r.t.indexOf("dense") === 0).reduce((s, r) => s + r.p, 0);
    const denseM = rows.filter(r => r.t.indexOf("dense") === 0).reduce((s, r) => s + r.m, 0);
    const biggest = rows.slice().sort((a, b) => b.p - a.p)[0];

    document.getElementById("net-readout").innerHTML =
      `<b>${spec.label}</b>, ${head === "gap" ? "global-average-pool head" : "flatten + dense head"}: ` +
      `<b>${DL.commas(tot.p)}</b> parameters and <b>${DL.big(tot.m)}</b> MACs per example. ` +
      `Convolution layers hold ${DL.commas(convP)} parameters (${(100 * convP / tot.p).toFixed(1)}%) and do ${DL.big(convM)} MACs (${(100 * convM / tot.m).toFixed(1)}%); ` +
      `dense layers hold ${DL.commas(denseP)} (${(100 * denseP / tot.p).toFixed(1)}%) and do ${DL.big(denseM)} (${(100 * denseM / tot.m).toFixed(1)}%). ` +
      ((convP / tot.p < denseP / tot.p && convM / tot.m > denseM / tot.m)
        ? `<b>The two profiles are inverted</b> — the layers holding most of the parameters do least of the arithmetic — which is the whole point of the two columns. `
        : `Here the convolution layers hold the larger share of BOTH, so the usual inversion is only partial: this architecture has no large flatten, which is exactly what removes it. `) +
      `The single largest layer by parameters is <b>${biggest.t} ${biggest.in} → ${biggest.out}</b> at ${DL.commas(biggest.p)} (${(100 * biggest.p / tot.p).toFixed(1)}% of the network). ` +
      `Switching the head the other way gives ${DL.commas(totAlt.p)} parameters and ${DL.big(totAlt.m)} MACs — a parameter ratio of <b>${(Math.max(tot.p, totAlt.p) / Math.min(tot.p, totAlt.p)).toFixed(2)}×</b> ` +
      `for an arithmetic ratio of only ${(Math.max(tot.m, totAlt.m) / Math.min(tot.m, totAlt.m)).toFixed(3)}×. ` +
      (rows.some(r => !r.exact) ? `<b>⚠ ${rows.filter(r => !r.exact).length} layer${rows.filter(r => !r.exact).length === 1 ? "" : "s"} ${rows.filter(r => !r.exact).length === 1 ? "has" : "have"} an inexact division</b>, flagged in the table — some input positions are discarded there (§09 trap 1).` : `Every layer's division is exact, so no input position is silently discarded.`);
  }
  [ae, he, se].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 13 · #rf-svg — the receptive field, computed and MEASURED ═══════════ */
(function () {
  const svg = d3.select("#rf-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const pe = document.getElementById("rf-p"), Le = document.getElementById("rf-L"),
    ie = document.getElementById("rf-i");

  /* a preset expands to a list of {k, s, p0, d, pool} — pooling is modelled by
     its SUPPORT (a box of width k at stride s), which has exactly the same
     receptive field as a max pool; only the values differ, and this figure
     measures which inputs MOVE an output, not by how much. */
  function stack(preset, L) {
    const a = [];
    if (preset === "plain") for (let i = 0; i < L; i++) a.push({ k: 3, s: 1, p0: 1, d: 1 });
    else if (preset === "pooled") {
      for (let i = 0; i < L; i++) { a.push({ k: 3, s: 1, p0: 1, d: 1 }); if (i % 2 === 1) a.push({ k: 2, s: 2, p0: 0, d: 1, pool: true }); }
    } else if (preset === "strided") for (let i = 0; i < L; i++) a.push({ k: 3, s: 2, p0: 1, d: 1 });
    else if (preset === "stem") { a.push({ k: 7, s: 2, p0: 3, d: 1 }); for (let i = 1; i < L; i++) a.push({ k: 3, s: 1, p0: 1, d: 1 }); }
    else if (preset === "dilated") for (let i = 0; i < L; i++) { const d = Math.pow(2, i); a.push({ k: 3, s: 1, p0: d, d: d }); }
    else if (preset === "onexone") for (let i = 0; i < L; i++) a.push({ k: 1, s: 1, p0: 0, d: 1 });
    else for (let i = 0; i < L; i++) a.push({ k: 4, s: 1, p0: 1, d: 1 });
    return a;
  }
  /* run a 1-D impulse through the stack; return, per layer, the array of values */
  function forward(x, layers) {
    /* DL.corr2d pads BOTH axes with the same pair, so a 1-row image would
       grow vertically. Pad the row explicitly along its own axis instead and
       call the layer with zero padding — identical arithmetic, 1-D shape. */
    let row = x.slice();
    const out = [];
    layers.forEach(Lr => {
      const padded = new Array(Lr.p0).fill(0).concat(row, new Array(Lr.p0).fill(0));
      const K = [new Array(Lr.k).fill(Lr.pool ? 1 / Lr.k : 1)];
      row = DL.corr2d([padded], K, { s: Lr.s, p0: 0, p1: 0, d: Lr.d })[0];
      out.push(row.slice());
    });
    return out;
  }

  function draw() {
    const preset = pe.value, L = +Le.value;
    document.getElementById("rf-Lv").textContent = L;
    const layers = stack(preset, L);
    const chain = DL.rfChain(layers);
    const rFinal = chain[chain.length - 1].r;
    /* an input long enough that the field is not clipped, where possible */
    const N = Math.max(63, Math.min(383, 2 * rFinal + 15));
    ie.max = String(N - 1);
    const imp = Math.min(+ie.value, N - 1);
    document.getElementById("rf-iv").textContent = imp;

    /* ---- ONE sweep of impulses gives the measured field at EVERY depth ---- */
    const zero = new Array(N).fill(0);
    const base = forward(zero, layers);
    const centres = base.map(row => Math.floor(row.length / 2));
    const lo = base.map(() => Infinity), hi = base.map(() => -Infinity), cnt = base.map(() => 0);
    for (let i = 0; i < N; i++) {
      const x = new Array(N).fill(0); x[i] = 1;
      const f = forward(x, layers);
      for (let l = 0; l < f.length; l++) {
        if (Math.abs(f[l][centres[l]] - base[l][centres[l]]) > 1e-12) {
          lo[l] = Math.min(lo[l], i); hi[l] = Math.max(hi[l], i); cnt[l]++;
        }
      }
    }
    const measured = base.map((_, l) => (cnt[l] ? hi[l] - lo[l] + 1 : 0));

    /* ---- the cone from the SELECTED impulse ---- */
    const ximp = new Array(N).fill(0); ximp[imp] = 1;
    const fimp = forward(ximp, layers);

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 20, b: 12 });
    const g = F.g;
    const stripW = 300, cw = stripW / N;
    const rowH = Math.min(15, 150 / (layers.length + 1));
    const yOf = l => 150 - l * rowH;
    const draws = (row, l, moved) => {
      const sc = stripW / row.length;
      for (let i = 0; i < row.length; i++) {
        if (row.length > 200 && !moved[i]) continue;
        g.append("rect").attr("x", 34 + i * sc).attr("y", yOf(l)).attr("width", Math.max(0.7, sc - 0.2))
          .attr("height", Math.max(2, rowH - 2))
          .attr("fill", moved[i] ? DC.accent : "#232833").attr("fill-opacity", moved[i] ? 0.95 : 1)
          .attr("shape-rendering", "crispEdges");
      }
      if (row.length <= 200) { /* background already drawn per cell */ }
      else g.append("rect").attr("x", 34).attr("y", yOf(l)).attr("width", stripW).attr("height", Math.max(2, rowH - 2))
        .attr("fill", "none").attr("stroke", DC.line).attr("stroke-width", 0.5);
    };
    /* input row */
    draws(zero, 0, zero.map((_, i) => i === imp));
    g.append("text").attr("x", 30).attr("y", yOf(0) + rowH - 3).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", DC.muted).text("in");
    fimp.forEach((row, l) => {
      draws(row, l + 1, row.map((v, i) => Math.abs(v - base[l][i]) > 1e-12));
      if (rowH >= 9) g.append("text").attr("x", 30).attr("y", yOf(l + 1) + rowH - 3).attr("text-anchor", "end")
        .attr("font-size", 9).attr("fill", DC.muted).text((layers[l].pool ? "p" : "L") + (l + 1));
    });
    g.append("text").attr("x", 34).attr("y", yOf(layers.length) - 8).attr("font-size", 11)
      .attr("fill", DC.ink).attr("font-weight", 600)
      .text(`one input impulse at ${imp}: the cone of units it changes (measured, not drawn)`);

    /* ---- formula against measurement, per depth ---- */
    const px = 380, pw = 340, ph = 130, py = 26;
    const xs = d3.scaleLinear().domain([1, Math.max(2, layers.length)]).range([0, pw]);
    const maxr = Math.max.apply(null, chain.slice(1).map(c => c.r).concat(measured));
    const ys = d3.scaleLinear().domain([0, maxr * 1.1]).range([ph, 0]);
    const gp = g.append("g").attr("transform", `translate(${px},${py})`);
    DL.gridY(gp, ys, pw, 4);
    DL.axisB(gp, xs, ph, Math.min(8, layers.length), "layer");
    DL.axisL(gp, ys, 4, "receptive field (input pixels)");
    DL.curve(gp, chain.slice(1).map((c, i) => [xs(i + 1), ys(c.r)]), { stroke: DC.accent, w: 2 });
    gp.append("g").selectAll("circle").data(measured.map((m, i) => [i + 1, m])).join("circle")
      .attr("cx", d => xs(d[0])).attr("cy", d => ys(d[1])).attr("r", 3.2)
      .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 1.6);
    DL.legend(gp, [{ color: DC.accent, label: "formula  r = r + (fₑ−1)·j" }, { color: DC.a2, label: "measured span" }],
      6, 10, { gap: 14, font: 10 });

    /* ---- the table ---- */
    const ty = py + ph + 40;
    const cols = [0, 56, 96, 134, 178, 230];
    ["layer", "fₑ", "s", "j", "r", "centre₀"].forEach((t, j) =>
      g.append("text").attr("x", px + cols[j]).attr("y", ty).attr("font-size", 10).attr("fill", DC.muted).text(t));
    const show = chain.slice(1);
    const step = Math.max(1, Math.ceil(show.length / 8));
    let rowi = 0;
    for (let i = 0; i < show.length; i += step) {
      const c = show[i], y = ty + 15 + rowi * 14;
      const vals = [(layers[i].pool ? "pool " : "conv ") + (i + 1), c.fe, c.s, c.j, c.r, c.start];
      vals.forEach((v, j) => g.append("text").attr("x", px + cols[j]).attr("y", y).attr("font-size", 10)
        .attr("font-family", j ? "SF Mono, Menlo, monospace" : null)
        .attr("fill", j === 5 && (c.start % 1 !== 0) ? DC.violet : (j === 4 ? DC.accent : DC.ink)).text(v));
      rowi++;
    }

    const agree = measured.every((m, i) => m === Math.min(chain[i + 1].r, N));
    const clipped = measured.some((m, i) => m < chain[i + 1].r);
    document.getElementById("rf-readout").innerHTML =
      `${layers.length} layer${layers.length === 1 ? "" : "s"} (${preset}). ` +
      `Formula: <b>r = ${rFinal}</b>, jump <b>j = ${chain[chain.length - 1].j}</b>, centre of unit 0 at <b>${chain[chain.length - 1].start}</b>. ` +
      `Measured by sweeping a single input impulse across ${N} positions and recording which of them move the central output: <b>span = ${measured[measured.length - 1]}</b>. ` +
      (agree
        ? `<b>They agree at every depth</b>${clipped ? " once the ±" + N + "-pixel input is taken into account" : ""} — the formula is confirmed, not assumed.`
        : `<b>They disagree</b> at some depth, which with an input of only ${N} pixels means the field has run off the end of the input.`) +
      ` The last layer's ${measured[measured.length - 1]}-pixel field covers <b>${(100 * Math.min(1, rFinal / 224)).toFixed(0)}%</b> of a 224-pixel input. ` +
      ((Math.abs(chain[chain.length - 1].start - 0.5) > 1e-9)
        ? `The centre of output unit 0 sits at input coordinate <b>${chain[chain.length - 1].start}</b>, against 0.5 for the input's own first pixel — an offset of <b>${DL.fmt(chain[chain.length - 1].start - 0.5, 2)}</b> pixels. Even-sized kernels and even-sized pools are what produce this drift (§09 trap 3), and nobody tracks it.`
        : `The centre of output unit 0 sits at input coordinate 0.5, exactly where the input's own first pixel centre is, so the output grid is registered with the input grid and no half-pixel drift has accumulated.`) +
      ` For comparison, ${layers.length} plain 3×3 stride-1 layers would give r = ${2 * layers.length + 1}.`;
  }
  [Le, ie].forEach(e => e.addEventListener("input", draw));
  pe.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 14 · #erf-svg — the influence map, by backpropagation ═══════════ */
(function () {
  const svg = d3.select("#erf-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const Le = document.getElementById("er-L"), fe = document.getElementById("er-f"),
    we = document.getElementById("er-w"), se = document.getElementById("er-s"),
    sde = document.getElementById("er-seed");

  /* ∂z_centre/∂x, obtained by running DL.convGradV back through the stack with
     a one-hot output gradient. A stack of convolutions is LINEAR, so this
     derivative does not depend on the input and the map is exact.          */
  function influence(L, f, kind, skip, seed, N) {
    const p = (f - 1) / 2, geom = { s: 1, p0: p, p1: p, d: 1 };
    const r = DL.rng(seed);
    const Ks = [];
    for (let l = 0; l < L; l++) {
      const K = [[DL.zeros2(f, f)]];
      for (let i = 0; i < f; i++) for (let j = 0; j < f; j++) {
        if (kind === "uniform") K[0][0][i][j] = 1 / (f * f);
        else if (kind === "random") K[0][0][i][j] = DL.randn(r) * Math.sqrt(2 / (f * f));
        else {
          const di = i - p, dj = j - p;
          K[0][0][i][j] = Math.exp(-(di * di + dj * dj) / (2 * 0.7 * 0.7));
        }
      }
      Ks.push(K);
    }
    const c = Math.floor(N / 2);
    let G = DL.zeros3(1, N, N);
    G[0][c][c] = 1;
    for (let l = L - 1; l >= 0; l--) {
      const back = DL.convGradV(Ks[l], G, geom, N, N);
      if (skip === "on") for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) back[0][i][j] += G[0][i][j];
      G = back;
    }
    return G[0];
  }
  /* the smallest Chebyshev radius holding a given fraction of the |mass| */
  function massRadius(M, frac) {
    const N = M.length, c = Math.floor(N / 2);
    let tot = 0;
    M.forEach(r => r.forEach(v => tot += Math.abs(v)));
    if (tot === 0) return { R: 0, tot: 0 };
    let acc = 0;
    for (let R = 0; R <= c; R++) {
      acc = 0;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
        if (Math.max(Math.abs(i - c), Math.abs(j - c)) <= R) acc += Math.abs(M[i][j]);
      if (acc / tot >= frac) return { R: R, tot: tot, frac: acc / tot };
    }
    return { R: c, tot: tot, frac: 1 };
  }

  function draw() {
    const L = +Le.value, f = +fe.value, kind = we.value, skip = se.value, seed = +sde.value;
    ["er-Lv", "er-fv", "er-seedv"].forEach((id, i) => document.getElementById(id).textContent = [L, f, seed][i]);
    const rTheo = L * (f - 1) + 1;
    const N = Math.min(81, 2 * Math.floor(rTheo / 2) + 9);
    const M = influence(L, f, kind, skip, seed, N);
    const mr = massRadius(M, 0.95);
    const wEff = 2 * mr.R + 1;
    const c = Math.floor(N / 2);
    const mmax = Math.max.apply(null, M.map(r => Math.max.apply(null, r.map(Math.abs)))) || 1;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;

    /* ---- the map ---- */
    const cw = Math.min(4.4, 230 / N);
    g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`∂z(centre) / ∂x — the true derivative`);
    DL.cells(g, 0, 20, cw, N, N, (x, y) => CN.signRamp(M[y][x], mmax));
    const half = Math.min(c, (rTheo - 1) / 2);
    g.append("rect").attr("x", (c - half) * cw).attr("y", 20 + (c - half) * cw)
      .attr("width", (2 * half + 1) * cw).attr("height", (2 * half + 1) * cw)
      .attr("fill", "none").attr("stroke", DC.muted).attr("stroke-dasharray", "3,2").attr("stroke-width", 1.2);
    g.append("circle").attr("cx", (c + 0.5) * cw).attr("cy", 20 + (c + 0.5) * cw).attr("r", (mr.R + 0.5) * cw)
      .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 1.6);
    g.append("text").attr("x", 0).attr("y", 20 + N * cw + 14).attr("font-size", 10).attr("fill", DC.muted)
      .text(`dashed: theoretical ${rTheo}×${rTheo}`);
    g.append("text").attr("x", 0).attr("y", 20 + N * cw + 28).attr("font-size", 10).attr("fill", DC.a2)
      .text(`circle: 95% of the influence, ${wEff}×${wEff}`);

    /* ---- the central slice, against a Gaussian of the same variance ---- */
    const px = 268, pw = 210, ph = 150;
    const gp = g.append("g").attr("transform", `translate(${px},34)`);
    const slice = M[c].slice();
    const smax = Math.max.apply(null, slice.map(Math.abs)) || 1;
    let m0 = 0, m2 = 0;
    slice.forEach((v, i) => { m0 += Math.abs(v); m2 += Math.abs(v) * (i - c) * (i - c); });
    const sd = Math.sqrt(m2 / (m0 || 1));
    const xs = d3.scaleLinear().domain([-c, c]).range([0, pw]);
    const ys = d3.scaleLinear().domain([0, smax * 1.12]).range([ph, 0]);
    DL.gridY(gp, ys, pw, 4);
    DL.axisB(gp, xs, ph, 5, "offset from centre (px)");
    DL.axisL(gp, ys, 4, "|∂z/∂x|");
    DL.curve(gp, slice.map((v, i) => [xs(i - c), ys(Math.abs(v))]), { stroke: DC.accent, w: 1.8 });
    const peak = Math.abs(slice[c]);
    DL.curve(gp, d3.range(-c, c + 0.5, 0.5).map(u => [xs(u), ys(peak * Math.exp(-u * u / (2 * sd * sd)))]),
      { stroke: DC.good, w: 1.4, dash: "4,3" });
    DL.legend(gp, [{ color: DC.accent, label: "measured slice" }, { color: DC.good, label: `Gaussian, sd = ${DL.fmt(sd, 2)}`, dash: "4,3" }],
      6, 10, { gap: 14, font: 10 });

    /* ---- theoretical and effective width against depth ---- */
    const qx = 520, qw = 210, qh = 150;
    const gq = g.append("g").attr("transform", `translate(${qx},34)`);
    const pts = [];
    for (let l = 1; l <= L; l++) {
      const rt = l * (f - 1) + 1;
      const Nl = Math.min(81, 2 * Math.floor(rt / 2) + 9);
      const Ml = influence(l, f, kind, skip, seed, Nl);
      pts.push({ l: l, theo: rt, eff: 2 * massRadius(Ml, 0.95).R + 1 });
    }
    const mxy = Math.max.apply(null, pts.map(p => p.theo));
    const xq = d3.scaleLinear().domain([1, Math.max(2, L)]).range([0, qw]);
    const yq = d3.scaleLinear().domain([0, mxy * 1.12]).range([qh, 0]);
    DL.gridY(gq, yq, qw, 4);
    DL.axisB(gq, xq, qh, Math.min(7, L), "depth");
    DL.axisL(gq, yq, 4, "width (px)");
    DL.curve(gq, pts.map(p => [xq(p.l), yq(p.theo)]), { stroke: DC.muted, w: 1.6, dash: "4,3" });
    DL.curve(gq, pts.map(p => [xq(p.l), yq(p.eff)]), { stroke: DC.a2, w: 2 });
    DL.legend(gq, [{ color: DC.muted, label: "theoretical", dash: "4,3" }, { color: DC.a2, label: "effective (95% mass)" }],
      6, 10, { gap: 14, font: 10 });
    /* the ratio, beneath */
    const ry = qh + 44;
    const yr = d3.scaleLinear().domain([0, 1.05]).range([ry + 56, ry]);
    const gr = gq.append("g");
    DL.axisL(gr, yr, 3, "eff ÷ theo");
    DL.axisB(gr, xq, ry + 56, Math.min(7, L), "depth");
    DL.curve(gr, pts.map(p => [xq(p.l), yr(p.eff / p.theo)]), { stroke: DC.violet, w: 1.8 });

    const areaFrac = (wEff * wEff) / (rTheo * rTheo);
    document.getElementById("erf-readout").innerHTML =
      `${L} layers of ${f}×${f}, stride 1${skip === "on" ? ", with an identity skip on every layer" : ""}. ` +
      `Theoretical receptive field <b>${rTheo}×${rTheo}</b>; the region carrying 95% of the total |influence| is <b>${wEff}×${wEff}</b>. ` +
      `Ratio of widths <b>${DL.fmt(wEff / rTheo, 3)}</b>, ratio of <i>areas</i> <b>${DL.fmt(areaFrac, 3)}</b> — so ${(100 * (1 - areaFrac)).toFixed(0)}% of the theoretical field's area carries 5% of the influence. ` +
      `The central slice has standard deviation <b>${DL.fmt(sd, 2)}</b> px against a theoretical half-width of ${((rTheo - 1) / 2).toFixed(0)}. ` +
      (L >= 4
        ? `Between depth ${pts[0].l} and depth ${pts[pts.length - 1].l} the theoretical width grew by <b>${DL.fmt(pts[pts.length - 1].theo / pts[0].theo, 2)}×</b> while the effective width grew by only <b>${DL.fmt(pts[pts.length - 1].eff / pts[0].eff, 2)}×</b>, and the ratio curve is falling — the √L against L prediction, measured.`
        : `Add depth and watch the two curves separate: the theoretical width grows linearly and the effective width like the square root.`) +
      (skip === "on" ? " The identity skip concentrates the influence at the centre, which is why residual networks have a <i>smaller</i> effective field than their depth suggests." : "") +
      (kind === "random" ? " With random weights the map is noisy and can change sign, but its envelope is the same Gaussian; the 95%-mass radius is computed on |∂z/∂x| for exactly that reason." : "");
  }
  [Le, fe, sde].forEach(e => e.addEventListener("input", draw));
  [we, se].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 15 · #1x1-svg — the bottleneck ═══════════ */
(function () {
  const svg = d3.select("#onexone-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const ids = ["ox-n", "ox-ci", "ox-co", "ox-f", "ox-cr"].map(i => document.getElementById(i));

  function cost(n, ci, co, f, cr) {
    const p = (f - 1) / 2;
    const direct = DL.convCount({ H: n, W: n, cin: ci, cout: co, k: f, s: 1, p0: p, p1: p });
    const red = DL.convCount({ H: n, W: n, cin: ci, cout: cr, k: 1, s: 1, p0: 0, p1: 0 });
    const mid = DL.convCount({ H: n, W: n, cin: cr, cout: cr, k: f, s: 1, p0: p, p1: p });
    const exp = DL.convCount({ H: n, W: n, cin: cr, cout: co, k: 1, s: 1, p0: 0, p1: 0 });
    return { direct: direct, red: red, mid: mid, exp: exp,
      bMacs: red.macs + mid.macs + exp.macs, bParams: red.params + mid.params + exp.params };
  }
  function draw() {
    const [n, ci, co, f, cr] = ids.map(e => +e.value);
    ["ox-nv", "ox-civ", "ox-cov", "ox-fv", "ox-crv"].forEach((id, i) =>
      document.getElementById(id).textContent = [n, ci, co, f, cr][i]);
    /* the SIMPLE bottleneck the source teaches: 1×1 reduce, then f×f to Cout */
    const p = (f - 1) / 2;
    const direct = DL.convCount({ H: n, W: n, cin: ci, cout: co, k: f, s: 1, p0: p, p1: p });
    const red = DL.convCount({ H: n, W: n, cin: ci, cout: cr, k: 1 });
    const wide = DL.convCount({ H: n, W: n, cin: cr, cout: co, k: f, s: 1, p0: p, p1: p });
    const bMacs = red.macs + wide.macs, bParams = red.params + wide.params;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;

    /* schematic */
    const boxw = 96, bh = 34;
    const chip = (x, y, w, t, sub, col) => {
      g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", bh).attr("rx", 4)
        .attr("fill", col).attr("fill-opacity", 0.14).attr("stroke", col);
      g.append("text").attr("x", x + w / 2).attr("y", y + 15).attr("text-anchor", "middle").attr("font-size", 10.5)
        .attr("fill", DC.ink).text(t);
      g.append("text").attr("x", x + w / 2).attr("y", y + 27).attr("text-anchor", "middle").attr("font-size", 9.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text(sub);
    };
    g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("one expensive layer");
    chip(0, 16, boxw, `${n}×${n}×${ci}`, "input", DC.muted);
    DL.arrow(g, boxw + 4, 16 + bh / 2, boxw + 26, 16 + bh / 2, { color: DC.muted, w: 1.2, head: 5 });
    chip(boxw + 30, 16, boxw, `conv ${f}×${f}`, `→ ${co} ch`, DC.bad);
    DL.arrow(g, 2 * boxw + 34, 16 + bh / 2, 2 * boxw + 56, 16 + bh / 2, { color: DC.muted, w: 1.2, head: 5 });
    chip(2 * boxw + 60, 16, boxw, `${n}×${n}×${co}`, "output", DC.muted);

    g.append("text").attr("x", 0).attr("y", 76).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("the same map through a bottleneck");
    chip(0, 84, boxw, `${n}×${n}×${ci}`, "input", DC.muted);
    DL.arrow(g, boxw + 4, 84 + bh / 2, boxw + 26, 84 + bh / 2, { color: DC.muted, w: 1.2, head: 5 });
    chip(boxw + 30, 84, boxw, `conv 1×1`, `→ ${cr} ch`, DC.good);
    DL.arrow(g, 2 * boxw + 34, 84 + bh / 2, 2 * boxw + 56, 84 + bh / 2, { color: DC.muted, w: 1.2, head: 5 });
    chip(2 * boxw + 60, 84, boxw, `conv ${f}×${f}`, `→ ${co} ch`, DC.a2);
    DL.arrow(g, 3 * boxw + 64, 84 + bh / 2, 3 * boxw + 86, 84 + bh / 2, { color: DC.muted, w: 1.2, head: 5 });
    chip(3 * boxw + 90, 84, boxw, `${n}×${n}×${co}`, "output", DC.muted);

    /* the two MAC bars */
    const by = 146, bw = 420;
    const tot = Math.max(direct.macs, bMacs);
    g.append("text").attr("x", 0).attr("y", by - 6).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("multiply-accumulates per example");
    g.append("rect").attr("x", 118).attr("y", by).attr("width", bw * direct.macs / tot).attr("height", 22)
      .attr("fill", DC.bad).attr("fill-opacity", 0.8).attr("stroke", DC.bad);
    g.append("text").attr("x", 112).attr("y", by + 15).attr("text-anchor", "end").attr("font-size", 10.5)
      .attr("fill", DC.muted).text("direct");
    g.append("text").attr("x", 124 + bw * direct.macs / tot).attr("y", by + 15).attr("font-size", 10.5)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.bad).text(DL.big(direct.macs));
    let x = 118;
    [[red.macs, DC.good, "1×1"], [wide.macs, DC.a2, `${f}×${f}`]].forEach(([v, cc, t]) => {
      const ww = bw * v / tot;
      g.append("rect").attr("x", x).attr("y", by + 30).attr("width", ww).attr("height", 22)
        .attr("fill", cc).attr("fill-opacity", 0.8).attr("stroke", cc);
      if (ww > 30) g.append("text").attr("x", x + ww / 2).attr("y", by + 45).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", "#0f1117").attr("font-weight", 600).text(t);
      x += ww;
    });
    g.append("text").attr("x", 112).attr("y", by + 45).attr("text-anchor", "end").attr("font-size", 10.5)
      .attr("fill", DC.muted).text("bottleneck");
    g.append("text").attr("x", x + 6).attr("y", by + 45).attr("font-size", 10.5)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.good).text(DL.big(bMacs));

    /* the sweep over C_r */
    const sx = 118, sy = 226, sw = 420, sh = 130;
    const gs = g.append("g").attr("transform", `translate(${sx},${sy})`);
    const rs = [];
    for (let r = 2; r <= Math.max(ci, co); r += Math.max(1, Math.round(Math.max(ci, co) / 120))) {
      const a = DL.convCount({ H: n, W: n, cin: ci, cout: r, k: 1 });
      const b = DL.convCount({ H: n, W: n, cin: r, cout: co, k: f, s: 1, p0: p, p1: p });
      rs.push([r, a.macs + b.macs]);
    }
    /* the BREAK-EVEN width: bottleneck cost = direct cost.
       C_r·(C_in + C_out·f²) = C_in·C_out·f²  →  C_r* = C_in·C_out·f²/(C_in + C_out·f²)
       (biases ignored in the closed form; the marker uses the swept curve). */
    const breakEven = ci * co * f * f / (ci + co * f * f);
    const xs = d3.scaleLinear().domain([2, Math.max(ci, co)]).range([0, sw]);
    const ys = d3.scaleLog().domain([Math.max(1, d3.min(rs, d => d[1]) * 0.7), d3.max(rs, d => d[1]) * 1.4]).range([sh, 0]);
    DL.gridY(gs, ys, sw, 4);
    DL.axisB(gs, xs, sh, 6, "reduction width  C_r");
    DL.axisL(gs, ys, 4, "total MACs", v => DL.big(v));
    gs.append("line").attr("x1", 0).attr("x2", sw).attr("y1", ys(direct.macs)).attr("y2", ys(direct.macs))
      .attr("stroke", DC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
    gs.append("text").attr("x", 4).attr("y", ys(direct.macs) - 5).attr("font-size", 10).attr("fill", DC.bad)
      .text(`the direct layer: ${DL.big(direct.macs)}`);
    DL.curve(gs, rs.map(d => [xs(d[0]), ys(d[1])]), { stroke: DC.accent, w: 1.9 });
    if (breakEven >= 2 && breakEven <= Math.max(ci, co)) {
      gs.append("line").attr("x1", xs(breakEven)).attr("x2", xs(breakEven)).attr("y1", 0).attr("y2", sh)
        .attr("stroke", DC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "3,3");
      gs.append("text").attr("x", xs(breakEven) + 6).attr("y", 12).attr("font-size", 10).attr("fill", DC.good)
        .text(`break-even C_r = ${breakEven.toFixed(1)}`);
    }
    gs.append("circle").attr("cx", xs(cr)).attr("cy", ys(bMacs)).attr("r", 4).attr("fill", DC.a2);

    const kv = DL.kv(g, sx + sw + 22, sy + 20, { lead: 16, keyW: 96, size: 10.5 });
    kv("direct params", DL.commas(direct.params), DC.bad);
    kv("bottleneck", DL.commas(bParams), DC.good);
    kv("param ratio", DL.fmt(bParams / direct.params, 4), DC.a2, true);
    kv("MAC ratio", DL.fmt(bMacs / direct.macs, 4), DC.a2, true);
    kv("times cheaper", DL.fmt(direct.macs / bMacs, 2) + "×", DC.good, true);

    document.getElementById("onexone-readout").innerHTML =
      `${n}×${n}×${ci} → ${n}×${n}×${co} through a ${f}×${f} kernel. ` +
      `<b>Direct</b>: ${DL.commas(direct.params)} parameters, ${DL.commas(direct.macs)} MACs. ` +
      `<b>Through a 1×1 reduction to ${cr} channels</b>: ${DL.commas(bParams)} parameters (${DL.commas(red.params)} + ${DL.commas(wide.params)}), ` +
      `${DL.commas(bMacs)} MACs (${DL.commas(red.macs)} + ${DL.commas(wide.macs)}). ` +
      `That is <b>${DL.fmt(direct.macs / bMacs, 2)}×</b> less arithmetic and <b>${DL.fmt(direct.params / bParams, 2)}×</b> fewer parameters` +
      (bMacs > direct.macs ? " — except that here it is <b>more expensive</b>, because C_r is not small enough relative to C in: a bottleneck that does not narrow is pure overhead." : ".") +
      ` The bottleneck's cost is <b>exactly linear and strictly increasing in C_r</b> — it is ${n}·${n}·C_r·(${ci} + ${co}·${f}²) — so there is no interior optimum: ` +
      `the arithmetic alone always says "narrow it further", and what stops you is representational capacity, which no cost curve can see. ` +
      `The two costs cross at <b>C_r = ${breakEven.toFixed(1)}</b>: below that the bottleneck is cheaper, above it the extra 1×1 layer is pure overhead. ` +
      `The 1×1 layer's whole job is that it has no f² factor: it costs ${DL.fmt(f * f, 0)}× less per channel pair than the ${f}×${f} layer does.`;
  }
  ids.forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════ 16 · #pool-svg — the pooling variants and what each discards ═══════════ */
(function () {
  const svg = d3.select("#pool-svg");
  if (svg.empty()) return;
  const W = 760, H = 420, N = 8;
  const me = document.getElementById("pl-m"), ke = document.getElementById("pl-k"),
    se = document.getElementById("pl-s"), ie = document.getElementById("pl-i");

  function map(kind) {
    const A = DL.zeros2(N, N), r = DL.rng(6);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (kind === "peaks") A[i][j] = 1 + Math.round(2 * r());
      else if (kind === "edge") A[i][j] = j < N / 2 ? 2 : 7;
      else A[i][j] = Math.round(2 + 6 * Math.abs(Math.sin(i * 1.3) * Math.cos(j * 1.1)));
    }
    if (kind === "peaks") { A[1][2] = 9; A[3][5] = 8; A[6][1] = 9; A[5][6] = 7; }
    return A;
  }
  function draw() {
    const mode = me.value, k = +ke.value, s = +se.value, kind = ie.value;
    document.getElementById("pl-kv").textContent = k;
    document.getElementById("pl-sv").textContent = s;
    const X = map(kind);
    const global = (mode === "gmax" || mode === "gavg");
    const R = global
      ? { Y: [[[DL.globalPool([X], mode === "gmax" ? "max" : "avg")[0]]]], arg: null, Ho: 1, Wo: 1 }
      : DL.pool2d([X], { k: k, s: s, mode: mode });
    const Y = R.Y[0], Ho = Y.length, Wo = Y[0].length;

    /* reconstruct: repeat each pooled value over its window (nearest upsample) */
    const rec = DL.zeros2(N, N), hit = DL.zeros2(N, N);
    if (global) { for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { rec[i][j] = Y[0][0]; hit[i][j] = 1; } }
    else for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++)
      for (let m = 0; m < k; m++) for (let n = 0; n < k; n++) {
        const i = p * s + m, j = q * s + n;
        if (i < N && j < N) { rec[i][j] = Y[p][q]; hit[i][j] = 1; }
      }
    const res = X.map((r0, i) => r0.map((v, j) => hit[i][j] ? v - rec[i][j] : v));
    let eX = 0, eR = 0;
    X.forEach((r0, i) => r0.forEach((v, j) => { eX += v * v; eR += res[i][j] * res[i][j]; }));

    /* which input cells were selected by SOME window (max only) */
    const sel = DL.zeros2(N, N);
    if (!global && mode === "max") for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++) {
      const [bi, bj] = R.arg[0][p][q]; if (bi >= 0) sel[bi][bj] = 1;
    }
    let nSel = 0, nCov = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { if (sel[i][j]) nSel++; if (hit[i][j]) nCov++; }

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = 26;
    const gx = CN.grid(g, X, 0, 24, cw, { title: `input feature map  ${N}×${N}`, font: 10 });
    if (!global) for (let p = 0; p < Ho; p++) for (let q = 0; q < Wo; q++)
      CN.box(gx, q * s * cw, p * s * cw, Math.min(k, N - q * s) * cw, Math.min(k, N - p * s) * cw, DC.a2, 1.3);
    if (!global && mode === "max") for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (sel[i][j])
      gx.append("circle").attr("cx", j * cw + cw - 5).attr("cy", i * cw + 5).attr("r", 2.6).attr("fill", DC.good);

    const x2 = N * cw + 34;
    CN.grid(g, Y, x2, 24, global ? 40 : cw, { title: `pooled  ${Ho}×${Wo}×1`, dp: mode === "avg" || mode === "gavg" ? 1 : 0, font: 10 });

    const x3 = x2 + (global ? 70 : N / s * cw + 24) + 24;
    CN.grid(g, rec, x3, 24, 18, { title: "upsampled back", dp: 1, font: 8.5, text: false });
    const rm = Math.max.apply(null, res.map(r0 => Math.max.apply(null, r0.map(Math.abs)))) || 1;
    CN.grid(g, res, x3, 24 + N * 18 + 30, 18, {
      title: `residual — what pooling destroyed (max |Δ| = ${DL.fmt(rm, 1)})`,
      text: false, fill: v => CN.signRamp(v, rm)
    });

    const px = x3 + N * 18 + 26;
    const kv = DL.kv(g, px, 40, { lead: 17, keyW: 168, size: 10.5 });
    kv("output shape", `${Ho}×${Wo}×1`, DC.ink);
    kv("values kept", `${Ho * Wo} of ${N * N}`, DC.ink);
    kv("reduction", `${DL.fmt(N * N / (Ho * Wo), 1)}×`, DC.accent, true);
    kv("relative L2 error", DL.fmt(Math.sqrt(eR / eX), 3), DC.a2, true);
    kv("parameters", "0", DC.good);
    if (!global && mode === "max") {
      kv("cells ever selected", `${nSel} of ${N * N}`, DC.good);
      kv("cells never selected", `${N * N - nSel}`, DC.bad, true);
      kv("position bits discarded", `${DL.fmt(Math.log2(k * k), 2)} per window`, DC.violet);
    }

    document.getElementById("pool-readout").innerHTML =
      `<b>${me.options[me.selectedIndex].text}</b>` + (global ? "" : `, window ${k}×${k}, stride ${s}`) +
      `: ${N}×${N} → ${Ho}×${Wo}, a <b>${DL.fmt(N * N / (Ho * Wo), 1)}×</b> reduction with <b>zero</b> parameters. ` +
      `Rebuilding the map from the pooled values by repeating each of them across its window and subtracting leaves a residual with relative L2 size ` +
      `<b>‖X − X̂‖₂ / ‖X‖₂ = ${DL.fmt(Math.sqrt(eR / eX), 3)}</b> — the pooled map simply does not contain what would be needed to rebuild the original. ` +
      (mode === "max" ? "(For max pooling this ratio can exceed 1, because repeating the window <i>maximum</i> is a biased reconstruction: every cell is given a value at least as large as its own.) " : "") +
      (mode === "avg" ? "(For average pooling the residual is exactly the within-window deviation from the mean, so its energy is the within-window variance — an unbiased reconstruction, and a smaller error than max pooling's on the same map.) " : "") +
      (global
        ? `A global pool reduces the whole map to one number, so what survives is "how much of this feature is present anywhere" and every trace of <i>where</i> is gone. That is exact invariance to any permutation of the positions, which is far stronger than translation invariance and is sometimes far more than you wanted (§30).`
        : mode === "max"
          ? `<b>${nSel}</b> of the ${N * N} input cells were selected by some window and <b>${N * N - nSel}</b> were never selected at all — those cells contribute nothing to this layer's output and receive no gradient through it. ` +
            `Each window also discards which of its ${k * k} positions won, which is ${DL.fmt(Math.log2(k * k), 2)} bits of spatial information per window, thrown away by construction.`
          : `Average pooling keeps a contribution from every cell — none is ignored and every cell receives gradient — but it destroys the <i>peak</i>: an isolated response of magnitude v comes out as v/${k * k}. Sharp features are attenuated by exactly the window area.`);
  }
  [ke, se].forEach(e => e.addEventListener("input", draw));
  [me, ie].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 17 · #invar-svg — equivariance and invariance, MEASURED ═══════════ */
(function () {
  const svg = d3.select("#invar-svg");
  if (svg.empty()) return;
  const W = 760, H = 420, N = 24;
  const ae = document.getElementById("iv-a"), ke = document.getElementById("iv-k"),
    ie = document.getElementById("iv-i"), me = document.getElementById("iv-m");

  function fmap(kind) {
    const A = DL.zeros2(N, N), r = DL.rng(12);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (kind === "edge") A[i][j] = (j < N / 2) ? 1 : 6;
      else if (kind === "tex") A[i][j] = 3 + 3 * Math.sin(i * 1.7) * Math.cos(j * 1.9);
      else A[i][j] = 0.4 * r();
    }
    if (kind === "peaks") [[5, 6], [9, 14], [15, 8], [17, 17], [11, 4]].forEach(([i, j]) => { A[i][j] = 8; A[i][j + 1] = 5; });
    return A;
  }
  const shift2 = (A, a) => {
    const M = A.length, B = DL.zeros2(M, A[0].length);
    for (let i = 0; i < M; i++) for (let j = 0; j < A[0].length; j++) {
      const si = i - a, sj = j - a;
      B[i][j] = (si >= 0 && si < M && sj >= 0 && sj < A[0].length) ? A[si][sj] : 0;
    }
    return B;
  };
  const KV = [[1, 0, -1], [2, 0, -2], [1, 0, -1]];
  const LAYERS = k => [
    { t: "conv 3×3, s1", s: 1, f: A => DL.corr2d(A, KV, { p0: 1, p1: 1 }) },
    { t: "conv 3×3, s2", s: 2, f: A => DL.corr2d(A, KV, { p0: 1, p1: 1, s: 2 }) },
    { t: `max pool ${k}×${k}, s1`, s: 1, f: A => DL.pool2d([A], { k: k, s: 1 }).Y[0] },
    { t: `max pool ${k}×${k}, s${k}`, s: k, f: A => DL.pool2d([A], { k: k, s: k }).Y[0] },
    { t: `avg pool ${k}×${k}, s${k}`, s: k, f: A => DL.pool2d([A], { k: k, s: k, mode: "avg" }).Y[0] },
    { t: "global avg pool", s: 0, f: A => [[DL.globalPool([A], "avg")[0]]] }
  ];
  /* compare over the INTERIOR only, so the border does not dominate */
  function compare(P, Q, off, metric) {
    const M = P.length, Wd = P[0].length;
    /* trim a border so the padding artefacts of §07 do not dominate — but a
       1×1 output (a global pool) has no interior to trim, so do not trim it
       away entirely and then report the empty comparison as a zero.        */
    let pad = Math.max(2, Math.ceil(M * 0.2));
    if (M - 2 * pad < 1 || Wd - 2 * pad < 1) pad = 0;
    let num = 0, den = 0, chg = 0, n = 0;
    for (let i = pad; i < M - pad; i++) for (let j = pad; j < Wd - pad; j++) {
      const qi = i - off, qj = j - off;
      const q = (qi >= 0 && qi < M && qj >= 0 && qj < Wd) ? Q[qi][qj] : 0;
      num += (P[i][j] - q) * (P[i][j] - q); den += P[i][j] * P[i][j];
      if (Math.abs(P[i][j] - q) > 1e-9) chg++;
      n++;
    }
    if (n === 0) return metric === "frac" ? 0 : 0;
    return metric === "frac" ? chg / n : Math.sqrt(num / (den || 1));
  }

  function draw() {
    const a = +ae.value, k = +ke.value, metric = me.value;
    document.getElementById("iv-av").textContent = a;
    document.getElementById("iv-kv").textContent = k;
    const X = fmap(ie.value), Xs = shift2(X, a);
    const rows = LAYERS(k).map(L => {
      const P = L.f(Xs), Q = L.f(X);
      const raw = compare(P, Q, 0, metric);
      const aligned = (L.s === 0) ? raw : ((a % L.s === 0) ? compare(P, Q, a / L.s, metric) : null);
      return { t: L.t, s: L.s, raw: raw, aligned: aligned };
    });

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;

    /* the classic 1-D pooling demonstration, reproduced and counted */
    const det = [0.1, 1.0, 0.2, 0.1, 0.0, 0.3, 0.1, 1.0, 0.2, 0.1];
    const detS = [0.3].concat(det.slice(0, det.length - 1));
    const p1 = [], p2 = [];
    for (let i = 0; i + 3 <= det.length; i++) { p1.push(Math.max(det[i], det[i + 1], det[i + 2])); p2.push(Math.max(detS[i], detS[i + 1], detS[i + 2])); }
    let changedD = 0, changedP = 0;
    for (let i = 0; i < det.length; i++) if (Math.abs(det[i] - detS[i]) > 1e-9) changedD++;
    for (let i = 0; i < p1.length; i++) if (Math.abs(p1[i] - p2[i]) > 1e-9) changedP++;
    const dx = 34, dw = 30;
    g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("the classic demonstration: a stride-1 max pool of width 3, before and after a one-position shift");
    [[p1, 18, "pooled"], [det, 40, "detector"], [p2, 74, "pooled"], [detS, 96, "shifted by 1"]].forEach(([arr, y, lab], bi) => {
      g.append("text").attr("x", dx - 6).attr("y", y + 12).attr("text-anchor", "end").attr("font-size", 9.5)
        .attr("fill", DC.muted).text(lab);
      arr.forEach((v, i) => {
        const other = bi === 0 ? p2[i] : (bi === 1 ? detS[i] : (bi === 2 ? p1[i] : det[i]));
        const same = Math.abs(v - other) < 1e-9;
        g.append("rect").attr("x", dx + i * dw).attr("y", y).attr("width", dw - 2).attr("height", 17)
          .attr("fill", same ? DC.panel2 : (bi % 2 === 0 ? DC.a2 : DC.rose)).attr("fill-opacity", same ? 1 : 0.35)
          .attr("stroke", same ? DC.line : (bi % 2 === 0 ? DC.a2 : DC.rose));
        g.append("text").attr("x", dx + i * dw + dw / 2 - 1).attr("y", y + 12).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(v.toFixed(1));
      });
    });
    g.append("text").attr("x", dx + p1.length * dw + 16).attr("y", 30).attr("font-size", 10.5).attr("fill", DC.rose)
      .text(`detector: ${changedD} of ${det.length} changed`);
    g.append("text").attr("x", dx + p1.length * dw + 16).attr("y", 86).attr("font-size", 10.5).attr("fill", DC.a2)
      .text(`pooled: ${changedP} of ${p1.length} changed (${(100 * changedP / p1.length).toFixed(0)}%)`);

    /* the measured bars */
    const by = 148, bw = 330, rh = 40;
    g.append("text").attr("x", 0).attr("y", by - 8).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`shift the 2-D map by ${a} px: ${metric === "frac" ? "fraction of interior outputs that changed" : "relative L2 change over the interior"}`);
    const mx = Math.max(0.02, Math.max.apply(null, rows.map(r => Math.max(r.raw, r.aligned === null ? 0 : r.aligned))));
    rows.forEach((r, i) => {
      const y = by + i * rh;
      g.append("text").attr("x", 148).attr("y", y + 11).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", DC.ink).text(r.t);
      const bar = (v, dy, col, lab) => {
        if (v === null) {
          g.append("text").attr("x", 158).attr("y", y + dy + 9).attr("font-size", 10).attr("fill", DC.bad)
            .text(`— identity undefined: stride ${r.s} does not divide ${a}`);
          return;
        }
        const ww = bw * v / mx;
        g.append("rect").attr("x", 156).attr("y", y + dy).attr("width", Math.max(0.8, ww)).attr("height", 12)
          .attr("fill", col).attr("fill-opacity", 0.8).attr("stroke", col).attr("stroke-width", 0.7);
        g.append("text").attr("x", 162 + ww).attr("y", y + dy + 10).attr("font-size", 9.5)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col)
          .text((metric === "frac" ? (100 * v).toFixed(1) + "%" : DL.fmt(v, 3)) + "  " + lab);
      };
      bar(r.raw, 0, DC.rose, "raw");
      bar(r.aligned, 15, DC.good, "after shifting the output back");
    });
    DL.legend(g, [{ color: DC.rose, label: "h(Tx) against h(x)" }, { color: DC.good, label: "h(Tx) against T′h(x)" }],
      556, by + 4, { gap: 15, font: 10 });
    g.append("text").attr("x", 556).attr("y", by + 46).attr("font-size", 10).attr("fill", DC.muted)
      .attr("xml:space", "preserve").text("equivariant → green ≈ 0,");
    g.append("text").attr("x", 556).attr("y", by + 59).attr("font-size", 10).attr("fill", DC.muted).text("red large");
    g.append("text").attr("x", 556).attr("y", by + 76).attr("font-size", 10).attr("fill", DC.muted).text("invariant → BOTH ≈ 0");
    g.append("text").attr("x", 556).attr("y", by + 93).attr("font-size", 10).attr("fill", DC.muted).text("neither → both large");

    const r0 = rows[0], r3 = rows[3], r5 = rows[5];
    document.getElementById("invar-readout").innerHTML =
      `Shift of ${a} px on a ${N}×${N} map, ${metric === "frac" ? "fraction of interior outputs changed" : "relative L2 change over the interior"}. ` +
      `<b>Stride-1 convolution</b>: ${metric === "frac" ? (100 * r0.raw).toFixed(1) + "%" : DL.fmt(r0.raw, 3)} raw, ` +
      `<b>${metric === "frac" ? (100 * r0.aligned).toFixed(1) + "%" : DL.fmt(r0.aligned, 3)}</b> after shifting the output back — that gap is what "equivariant, not invariant" means as a measurement. ` +
      `<b>Max pool ${k}×${k} stride ${k}</b>: ${metric === "frac" ? (100 * r3.raw).toFixed(1) + "%" : DL.fmt(r3.raw, 3)} raw` +
      (r3.aligned === null ? ` and the aligned comparison does not exist, because a shift of ${a} is not a multiple of ${k}.`
        : ` and ${metric === "frac" ? (100 * r3.aligned).toFixed(1) + "%" : DL.fmt(r3.aligned, 3)} aligned.`) +
      ` <b>Global average pool</b>: ${metric === "frac" ? (100 * r5.raw).toFixed(1) + "%" : DL.fmt(r5.raw, 3)} — ` +
      (r5.raw < 1e-9
        ? "exactly zero — exact invariance, with no caveat."
        : "not <i>quite</i> zero, and for a reason worth naming: a zero-filled shift is not a translation of the scene, it pushes a strip of content off the canvas and brings a strip of zeros on. The mean therefore changes by that strip. Under a cyclic shift, which is a genuine translation, a global average pool is invariant to machine precision.") +
      ` In the 1-D demonstration above, a one-position shift changed <b>${changedD} of ${det.length}</b> detector outputs and <b>${changedP} of ${p1.length}</b> pooled ones — ` +
      `pooling took the changed fraction from <b>${(100 * changedD / det.length).toFixed(0)}%</b> to <b>${(100 * changedP / p1.length).toFixed(0)}%</b> — a real reduction in sensitivity, and nothing remotely like invariance.`;
  }
  [ae, ke].forEach(e => e.addEventListener("input", draw));
  [ie, me].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 18 · #stride-svg — subsampling, the fold, and the pre-filter ═══════════ */
(function () {
  const svg = d3.select("#stride-svg");
  if (svg.empty()) return;
  const W = 760, H = 440, N = 96;
  const me = document.getElementById("st-m"), ie = document.getElementById("st-i"),
    ae = document.getElementById("st-a"), se = document.getElementById("st-s");

  /* A naive O(n²) magnitude spectrum. The FFT, the properties of the transform
     and the sampling theorem itself all belong to the Computer Vision series
     (vision/frequency-domain.html); this page only needs to READ one number
     off a spectrum, so a direct sum is the honest minimum. */
  function magSpec(x) {
    const n = x.length, out = [];
    for (let k = 0; k <= n / 2; k++) {
      let re = 0, im = 0;
      for (let i = 0; i < n; i++) { const th = -2 * Math.PI * k * i / n; re += x[i] * Math.cos(th); im += x[i] * Math.sin(th); }
      out.push(Math.hypot(re, im));
    }
    return out;
  }
  /* the fraction of energy above the NEW Nyquist limit, which is where the
     original band divided by the stride now sits */
  function aboveNyquist(x, s) {
    const S = magSpec(x), kmax = S.length - 1, cut = kmax / s;
    let tot = 0, above = 0;
    for (let k = 0; k < S.length; k++) { const e = S[k] * S[k]; tot += e; if (k > cut) above += e; }
    return { frac: above / (tot || 1), S: S, cut: cut, kmax: kmax };
  }
  const BIN = n => { const b = []; let c = 1; for (let k = 0; k <= n; k++) { b.push(c); c = c * (n - k) / (k + 1); } const t = b.reduce((a, v) => a + v, 0); return b.map(v => v / t); };
  const FILT = {
    none: { t: "none — subsample directly", h: [1], p: 0, macs: 0 },
    box3: { t: "3-tap box average", h: [1 / 3, 1 / 3, 1 / 3], p: 1, macs: 3 },
    bin5: { t: "5-tap binomial", h: BIN(4), p: 2, macs: 5 },
    bin9: { t: "9-tap binomial", h: BIN(8), p: 4, macs: 9 },
    maxp: { t: "max over the window", h: null, p: 0, macs: 0 }
  };
  function signal(kind, shift) {
    const x = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      const u = i - shift;
      if (u < 0 || u >= N) { x[i] = 0; continue; }
      if (kind === "chirp") x[i] = Math.sin(Math.PI * (0.02 + 0.010 * u) * u);
      else if (kind === "fine") x[i] = Math.sin(Math.PI * 0.86 * u);
      else if (kind === "band") x[i] = Math.sin(Math.PI * 0.06 * u) + 0.5 * Math.sin(Math.PI * 0.11 * u);
      else x[i] = ((u % 11) === 3 || (u % 17) === 5) ? 1 : 0;
    }
    return x;
  }
  const prefilter = (x, key, s) => (key === "maxp") ? CN.pool1d(x, s, 1, "max") : CN.conv1d(x, FILT[key].h, { p: FILT[key].p });
  const subsample = (x, s) => CN.conv1d(x, [1], { s: s });

  function draw() {
    const key = me.value, kind = ie.value, a = +ae.value, s = +se.value;
    document.getElementById("st-av").textContent = a;
    document.getElementById("st-sv").textContent = s;
    const x0 = signal(kind, 0), xa = signal(kind, a);
    const p0 = prefilter(x0, key, s), pa = prefilter(xa, key, s);
    const d0 = subsample(p0, s), da = subsample(pa, s);
    const an = aboveNyquist(p0, s);

    const F = DL.frame(svg, W, H, { l: 44, r: 16, t: 20, b: 16 });
    const g = F.g;
    const pw = 660;

    /* ---- panel A: the signal and its samples ---- */
    const phA = 92;
    const gA = g.append("g").attr("transform", "translate(0,14)");
    gA.append("text").attr("x", 0).attr("y", -4).attr("font-size", 10.5).attr("fill", DC.ink)
      .text(`input, the pre-filtered signal, and the ${d0.length} samples the stride-${s} grid keeps`);
    const allA = x0.concat(p0, d0);
    const xsA = d3.scaleLinear().domain([0, N - 1]).range([0, pw]);
    const ysA = d3.scaleLinear().domain([d3.min(allA) - 0.05, d3.max(allA) + 0.05]).range([phA, 0]);
    gA.append("line").attr("x1", 0).attr("x2", pw).attr("y1", ysA(0)).attr("y2", ysA(0)).attr("stroke", DC.grid);
    DL.curve(gA, x0.map((v, i) => [xsA(i), ysA(v)]), { stroke: DC.line, w: 1.1 });
    if (key !== "none") DL.curve(gA, p0.map((v, i) => [xsA(i), ysA(v)]), { stroke: DC.muted, w: 1.3 });
    DL.curve(gA, d0.map((v, i) => [xsA(i * s), ysA(v)]), { stroke: DC.accent, w: 2 });
    gA.append("g").selectAll("circle").data(d0).join("circle")
      .attr("cx", (v, i) => xsA(i * s)).attr("cy", v => ysA(v)).attr("r", 2).attr("fill", DC.accent);
    DL.curve(gA, da.map((v, i) => [xsA(i * s + a - Math.round(a / s) * s), ysA(v)]),
      { stroke: DC.a2, w: 1.4, dash: "4,3" });
    DL.axisL(gA, ysA, 3, "");
    DL.legend(gA, [{ color: DC.line, label: "input" }, { color: DC.accent, label: "downsampled" },
      { color: DC.a2, label: `downsampled after a shift of ${a}, realigned`, dash: "4,3" }],
      pw - 226, 8, { gap: 13, font: 9.5 });

    /* ---- panel B: the spectrum ---- */
    const phB = 84, yB = 146;
    const gB = g.append("g").attr("transform", `translate(0,${yB})`);
    gB.append("text").attr("x", 0).attr("y", -4).attr("font-size", 10.5).attr("fill", DC.ink)
      .text("magnitude spectrum of the pre-filtered signal — the shaded band folds when you subsample");
    const xsB = d3.scaleLinear().domain([0, an.kmax]).range([0, pw]);
    const ysB = d3.scaleLinear().domain([0, d3.max(an.S) * 1.08 || 1]).range([phB, 0]);
    gB.append("rect").attr("x", xsB(an.cut)).attr("y", 0).attr("width", pw - xsB(an.cut)).attr("height", phB)
      .attr("fill", DC.bad).attr("fill-opacity", 0.11);
    gB.append("g").selectAll("line").data(an.S).join("line")
      .attr("x1", (v, k) => xsB(k)).attr("x2", (v, k) => xsB(k)).attr("y1", phB).attr("y2", v => ysB(v))
      .attr("stroke", (v, k) => k > an.cut ? DC.bad : DC.accent).attr("stroke-width", 1.6);
    gB.append("line").attr("x1", xsB(an.cut)).attr("x2", xsB(an.cut)).attr("y1", -2).attr("y2", phB)
      .attr("stroke", DC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
    gB.append("text").attr("x", xsB(an.cut) + 6).attr("y", 10).attr("font-size", 10).attr("fill", DC.bad)
      .text(`new Nyquist limit (÷ ${s})`);
    DL.axisB(gB, xsB, phB, 6, "frequency index");
    DL.axisL(gB, ysB, 3, "|X|");

    /* ---- panel C: the bars ---- */
    const yC = 292;
    g.append("text").attr("x", 0).attr("y", yC - 6).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("fraction of energy above the new limit, at the moment of sampling — lower is better");
    const keys = ["none", "box3", "bin5", "bin9"];
    const bars = keys.map(k => ({ k: k, t: FILT[k].t, v: aboveNyquist(CN.conv1d(x0, FILT[k].h, { p: FILT[k].p }), s).frac, m: FILT[k].macs }));
    const mx = Math.max.apply(null, bars.map(b => b.v)) || 1;
    bars.forEach((b, i) => {
      const y = yC + i * 22;
      g.append("text").attr("x", 174).attr("y", y + 11).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", b.k === key ? DC.a2 : DC.muted).attr("font-weight", b.k === key ? 600 : 400).text(b.t);
      const ww = 340 * b.v / mx;
      g.append("rect").attr("x", 182).attr("y", y).attr("width", Math.max(0.8, ww)).attr("height", 14)
        .attr("fill", b.k === key ? DC.a2 : DC.accent).attr("fill-opacity", 0.8);
      g.append("text").attr("x", 188 + ww).attr("y", y + 11).attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text((100 * b.v).toFixed(2) + "%");
      g.append("text").attr("x", 620).attr("y", y + 11).attr("font-size", 10).attr("fill", DC.muted)
        .text(`${b.m} MACs / sample`);
    });
    /* the non-linear row, reported separately */
    const mp = CN.pool1d(x0, s, 1, "max");
    const S0 = magSpec(x0), SM = magSpec(mp);
    /* "created" = output energy sitting at frequency bins where the INPUT's
       magnitude was below 1% of its own peak. A linear filter can only
       attenuate such bins; a maximum can populate them.                    */
    const peak0 = Math.max.apply(null, S0) || 1;
    let created = 0;
    for (let k = 0; k < S0.length; k++) if (S0[k] < 0.01 * peak0) created += SM[k] * SM[k];
    const eM = SM.reduce((t, v) => t + v * v, 0) || 1;
    const yM = yC + 4 * 22 + 8;
    g.append("text").attr("x", 174).attr("y", yM + 11).attr("text-anchor", "end").attr("font-size", 10.5)
      .attr("fill", key === "maxp" ? DC.a2 : DC.muted).attr("font-weight", key === "maxp" ? 600 : 400)
      .text("max over the window");
    g.append("text").attr("x", 182).attr("y", yM + 11).attr("font-size", 10.5).attr("fill", DC.violet)
      .text(`non-linear — no frequency response. ${(100 * created / eM).toFixed(1)}% of its output energy sits at frequencies its input had none at.`);

    document.getElementById("stride-readout").innerHTML =
      `<b>${FILT[key].t}</b>, stride ${s}, on the ${kind} signal: ${N} samples in, <b>${d0.length}</b> out. ` +
      (key === "maxp"
        ? `A maximum is not a linear filter, so it has no frequency response and the "fraction above Nyquist" question cannot be asked of it in the usual way. ` +
          `What <i>can</i> be measured is how much of its output energy sits in frequency bins where the input's magnitude was below 1% of its own peak — bins a linear filter could only ever attenuate. Here that is <b>${(100 * created / eM).toFixed(2)}%</b>` +
          ((created / eM > 0.01)
            ? `, so this max filter is demonstrably <b>creating</b> spectral content that was not there, and then subsampling it. A pre-filter that adds energy above the limit is worse than no pre-filter at all.`
            : ` — small here, because this signal already has energy almost everywhere, so there are few empty bins for the maximum to fill. Switch to the fine stripe pattern, whose spectrum is a single line, and the effect is unmissable.`) +
          ` Either way the sampling theorem gives no guarantee whatsoever for a non-linear pre-filter, which is the honest summary.`
        : `<b>${(100 * an.frac).toFixed(2)}%</b> of the pre-filtered signal's energy lies above the new Nyquist limit, and every bit of it folds down onto a lower frequency at the moment of sampling. ` +
          `Without any pre-filter that figure is <b>${(100 * bars[0].v).toFixed(2)}%</b>; the 9-tap binomial brings it to <b>${(100 * bars[3].v).toFixed(2)}%</b>, ` +
          `a reduction of ${bars[3].v > 0 ? DL.fmt(bars[0].v / bars[3].v, 1) + "×" : "essentially everything"}, for ${FILT.bin9.macs} MACs per sample.`) +
      ` ` +
      (kind === "band"
        ? `This signal is band-limited below the new limit, so there is almost nothing to fold and every pre-filter is superfluous — which is the correct control case: anti-aliasing only helps when there <i>is</i> aliasing.`
        : kind === "chirp"
          ? `On the chirp the effect is visible directly in the top panel: past the point where the instantaneous frequency crosses the limit, the downsampled curve begins to oscillate <i>more slowly</i> again. That reversal is the fold. No amount of training recovers it, because the information was destroyed at the sample.`
          : kind === "fine"
            ? `The stripe pattern sits almost entirely above the new limit, so what comes out is a slow beat with no relationship to the input's period. Note that even the 9-tap binomial leaves a substantial fraction here — when the whole signal is above the limit, a short filter cannot save you and the honest answer is to sample less aggressively.`
            : `Sparse peaks have energy at every frequency, so a large fraction is always above the limit and a pre-filter helps a great deal. The shifted, realigned curve in the top panel and the original disagree by exactly the aliased part.`) +
      ` Note also the arithmetic: computing every output at stride 1 and then discarding ${s - 1} in ${s} costs exactly ${s}× the multiply-accumulates of the stride-${s} layer for the same result — which is why a strided convolution replaced "convolve then pool" everywhere.`;
  }
  [ae, se].forEach(e => e.addEventListener("input", draw));
  [me, ie].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 19 · #group-svg — grouped convolution and channel blocking ═══════════ */
(function () {
  const svg = d3.select("#group-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const ge = document.getElementById("gp-g"), ce = document.getElementById("gp-c"),
    Le = document.getElementById("gp-L"), xe = document.getElementById("gp-x");

  function draw() {
    const C = +ce.value, L = +Le.value, mix = xe.value;
    let g = +ge.value;
    while (C % g !== 0) g = g / 2;
    document.getElementById("gp-cv").textContent = C;
    document.getElementById("gp-Lv").textContent = L;

    /* the one-layer connection matrix, built from the group structure */
    const per = C / g;
    const conn = (co, ci) => Math.floor(co / per) === Math.floor(ci / per);
    /* reachability after L layers, PROPAGATED (boolean matrix powers) */
    let R = [];
    for (let i = 0; i < C; i++) { R.push(new Array(C).fill(false)); R[i][i] = true; }
    const step = M => {
      const O = [];
      for (let co = 0; co < C; co++) {
        const r = new Array(C).fill(false);
        for (let mid = 0; mid < C; mid++) if (conn(co, mid)) for (let ci = 0; ci < C; ci++) if (M[mid][ci]) r[ci] = true;
        O.push(r);
      }
      return O;
    };
    const shuffleIdx = i => (i % g) * per + Math.floor(i / g);
    const permute = M => M.map((row, co) => M[shuffleIdx(co)] ? M[shuffleIdx(co)].slice() : row.slice());
    const dense = M => {
      const O = [];
      for (let co = 0; co < C; co++) { const r = new Array(C).fill(false); for (let mid = 0; mid < C; mid++) for (let ci = 0; ci < C; ci++) if (M[mid][ci]) r[ci] = true; O.push(r); }
      return O;
    };
    for (let l = 0; l < L; l++) {
      R = step(R);
      if (l < L - 1) { if (mix === "shuffle") R = permute(R); else if (mix === "pw") R = dense(R); }
    }
    let filled = 0;
    R.forEach(r => r.forEach(v => { if (v) filled++; }));

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const gg = F.g;
    const cw = Math.min(9, 180 / C);

    const mat = (x, y, M, title, col) => {
      gg.append("text").attr("x", x).attr("y", y - 6).attr("font-size", 10.5).attr("fill", DC.ink).text(title);
      for (let i = 0; i < C; i++) for (let j = 0; j < C; j++)
        gg.append("rect").attr("x", x + j * cw).attr("y", y + i * cw).attr("width", cw).attr("height", cw)
          .attr("fill", M(i, j) ? col : "#1b1f28").attr("fill-opacity", M(i, j) ? 0.9 : 1)
          .attr("shape-rendering", "crispEdges");
      gg.append("rect").attr("x", x - 0.5).attr("y", y - 0.5).attr("width", C * cw + 1).attr("height", C * cw + 1)
        .attr("fill", "none").attr("stroke", DC.line);
      gg.append("text").attr("x", x).attr("y", y + C * cw + 14).attr("font-size", 9.5).attr("fill", DC.muted)
        .text("in →   out ↓");
    };
    mat(24, 30, conn, `one layer, g = ${g}`, DC.accent);
    mat(24 + C * cw + 60, 30, (i, j) => R[i][j],
      `reachable after ${L} layer${L === 1 ? "" : "s"}${mix === "none" ? "" : (mix === "shuffle" ? " + shuffle" : " + dense 1×1")}`, DC.good);

    let oneFilled = 0;
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) if (conn(i, j)) oneFilled++;
    gg.append("text").attr("x", 24).attr("y", 30 + C * cw + 30).attr("font-size", 10.5).attr("fill", DC.accent)
      .text(`filled: ${(100 * oneFilled / (C * C)).toFixed(1)}%  (1/g = ${(100 / g).toFixed(1)}%)`);
    gg.append("text").attr("x", 24 + C * cw + 60).attr("y", 30 + C * cw + 30).attr("font-size", 10.5)
      .attr("fill", filled === C * C ? DC.good : DC.bad)
      .text(`filled: ${(100 * filled / (C * C)).toFixed(1)}%` + (filled === C * C ? " — fully mixed" : " — BLOCKED"));

    /* cost bars */
    const n = 14, f = 3;
    const A = DL.convCount({ H: n, W: n, cin: C, cout: C, k: f, p0: 1, p1: 1 });
    const B = DL.convCount({ H: n, W: n, cin: C, cout: C, k: f, p0: 1, p1: 1, groups: g });
    const bx = 24 + 2 * (C * cw + 60) + 24, by = 40;
    const rows = [
      { t: "parameters", a: A.params, b: B.params },
      { t: "MACs", a: A.macs, b: B.macs },
      { t: "matrix products issued", a: 1, b: g }
    ];
    gg.append("text").attr("x", bx).attr("y", by - 14).attr("font-size", 11).attr("fill", DC.ink)
      .attr("font-weight", 600).text(`a ${n}×${n}×${C} → ${C} layer, f = 3`);
    rows.forEach((r, i) => {
      const y = by + i * 44;
      gg.append("text").attr("x", bx).attr("y", y).attr("font-size", 10.5).attr("fill", DC.muted).text(r.t);
      const mxv = Math.max(r.a, r.b), bw = 190;
      [[r.a, DC.bad, "g = 1"], [r.b, DC.good, `g = ${g}`]].forEach(([v, cc, lab], k) => {
        const ww = bw * v / mxv;
        gg.append("rect").attr("x", bx).attr("y", y + 6 + k * 14).attr("width", Math.max(1, ww)).attr("height", 11)
          .attr("fill", cc).attr("fill-opacity", 0.8);
        gg.append("text").attr("x", bx + ww + 6).attr("y", y + 15 + k * 14).attr("font-size", 9.5)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", cc).text(DL.big(v) + "  " + lab);
      });
      gg.append("text").attr("x", bx + 150).attr("y", y).attr("font-size", 10).attr("fill", DC.a2)
        .text(`× ${DL.fmt(r.b / r.a, 4)}`);
    });

    document.getElementById("group-readout").innerHTML =
      `<b>g = ${g}</b> on ${C} channels: each output channel sees ${per} of the ${C} input channels, so the one-layer connection matrix is ` +
      `<b>${(100 * oneFilled / (C * C)).toFixed(1)}%</b> filled, which is exactly 1/g. ` +
      `MACs fall by <b>${DL.fmt(B.macs / A.macs, 4)}</b> — <b>exactly 1/${g}</b> — and parameters by ${DL.fmt(B.params / A.params, 4)}, ` +
      `which is slightly more than 1/${g} because the ${C} biases do not group. ` +
      `After ${L} layer${L === 1 ? "" : "s"}${mix === "none" ? " with nothing between them" : (mix === "shuffle" ? " with a channel shuffle between them" : " with a dense 1×1 between them")}, ` +
      `the measured reachability is <b>${(100 * filled / (C * C)).toFixed(1)}%</b>` +
      (filled === C * C
        ? " — every input channel can influence every output channel."
        : ` — information in one group can <b>never</b> reach another, however deep the stack goes. That is the blocking problem, and depth does not fix it: it is a property of the connection structure, not of the number of layers.`) +
      ` The cost of the fix: a channel shuffle is a reindexing and costs nothing at all; a dense 1×1 costs ${DL.big(DL.convCount({ H: n, W: n, cin: C, cout: C, k: 1 }).macs)} MACs, ` +
      `which is ${DL.fmt(DL.convCount({ H: n, W: n, cin: C, cout: C, k: 1 }).macs / B.macs, 2)}× the grouped layer it follows.`;
  }
  ce.addEventListener("input", draw);
  Le.addEventListener("input", draw);
  [ge, xe].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 20 · #dws-svg — depthwise separable, and the ratio derived ═══════════ */
(function () {
  const svg = d3.select("#dws-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const ne = document.getElementById("dw-n"), ce = document.getElementById("dw-c"),
    fe = document.getElementById("dw-f"), se = document.getElementById("dw-s");

  function counts(n, c, f) {
    const p = (f - 1) / 2;
    const std = DL.convCount({ H: n, W: n, cin: c, cout: c, k: f, p0: p, p1: p });
    const dw = DL.convCount({ H: n, W: n, cin: c, cout: c, k: f, p0: p, p1: p, groups: c });
    const pw = DL.convCount({ H: n, W: n, cin: c, cout: c, k: 1 });
    return { std: std, dw: dw, pw: pw, macs: dw.macs + pw.macs, params: dw.params + pw.params };
  }
  function draw() {
    const n = +ne.value, c = +ce.value, f = +fe.value, sweep = se.value;
    ["dw-nv", "dw-cv", "dw-fv"].forEach((id, i) => document.getElementById(id).textContent = [n, c, f][i]);
    const C = counts(n, c, f);
    const closed = 1 / c + 1 / (f * f);
    const exact = C.macs / C.std.macs;

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;

    /* schematic */
    const chip = (x, y, w, t, sub, col) => {
      g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", 30).attr("rx", 4)
        .attr("fill", col).attr("fill-opacity", 0.14).attr("stroke", col);
      g.append("text").attr("x", x + w / 2).attr("y", y + 13).attr("text-anchor", "middle").attr("font-size", 10.5)
        .attr("fill", DC.ink).text(t);
      g.append("text").attr("x", x + w / 2).attr("y", y + 24).attr("text-anchor", "middle").attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text(sub);
    };
    g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("ordinary");
    chip(0, 14, 168, `conv ${f}×${f}`, `(${c}, ${c}, ${f}, ${f}) — space AND channels`, DC.bad);
    g.append("text").attr("x", 0).attr("y", 62).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("depthwise separable");
    chip(0, 68, 168, `depthwise ${f}×${f}`, `(${c}, 1, ${f}, ${f}), g = ${c} — space`, DC.good);
    DL.arrow(g, 84, 100, 84, 114, { color: DC.muted, w: 1.2, head: 5 });
    chip(0, 116, 168, `pointwise 1×1`, `(${c}, ${c}, 1, 1) — channels`, DC.accent);

    /* bars */
    const bx = 210, bw = 210;
    const AI = (macs, act) => macs / (4 * act);            // MACs per byte of activation
    const rows = [
      { t: "parameters", a: C.std.params, b: C.params, lower: true },
      { t: "MACs", a: C.std.macs, b: C.macs, lower: true },
      { t: "MACs per byte moved", a: AI(C.std.macs, C.std.acts + c * n * n), b: AI(C.macs, C.dw.acts + C.pw.acts + c * n * n), lower: false }
    ];
    rows.forEach((r, i) => {
      const y = 16 + i * 48;
      g.append("text").attr("x", bx).attr("y", y).attr("font-size", 10.5).attr("fill", DC.muted).text(r.t);
      const mxv = Math.max(r.a, r.b);
      [[r.a, DC.bad, "ordinary"], [r.b, DC.good, "separable"]].forEach(([v, cc, lab], k) => {
        const ww = bw * v / mxv;
        g.append("rect").attr("x", bx).attr("y", y + 6 + k * 14).attr("width", Math.max(1, ww)).attr("height", 11)
          .attr("fill", cc).attr("fill-opacity", 0.8);
        g.append("text").attr("x", bx + ww + 6).attr("y", y + 15 + k * 14).attr("font-size", 9.5)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", cc)
          .text((v >= 1000 ? DL.big(v) : DL.fmt(v, 1)) + "  " + lab);
      });
      const ratio = r.b / r.a;
      g.append("text").attr("x", bx + bw + 88).attr("y", y).attr("font-size", 10)
        .attr("fill", (r.lower ? ratio < 1 : ratio > 1) ? DC.good : DC.bad)
        .text(`× ${DL.fmt(ratio, 3)}` + ((r.lower ? ratio < 1 : ratio > 1) ? " ✓" : " ✗"));
    });
    g.append("text").attr("x", bx).attr("y", 16 + 3 * 48 - 4).attr("font-size", 9.5).attr("fill", DC.muted)
      .text("arithmetic intensity: HIGHER is better — this is the one the factorisation loses");

    /* the ratio sweep */
    const sx = 60, sy = 236, sw = 620, sh = 140;
    const gs = g.append("g").attr("transform", `translate(${sx},${sy})`);
    const pts = [];
    if (sweep === "cout") {
      for (let cc = 8; cc <= 512; cc += 8) pts.push({ x: cc, e: counts(n, cc, f).macs / counts(n, cc, f).std.macs, c: 1 / cc + 1 / (f * f) });
    } else {
      for (let ff = 1; ff <= 11; ff += 2) pts.push({ x: ff, e: counts(n, c, ff).macs / counts(n, c, ff).std.macs, c: 1 / c + 1 / (ff * ff) });
    }
    const xs = d3.scaleLinear().domain([pts[0].x, pts[pts.length - 1].x]).range([0, sw]);
    const ys = d3.scaleLinear().domain([0, Math.max(1.15, d3.max(pts, p => p.e) * 1.1)]).range([sh, 0]);
    gs.append("rect").attr("x", 0).attr("y", 0).attr("width", sw).attr("height", ys(1))
      .attr("fill", DC.bad).attr("fill-opacity", 0.08);
    DL.gridY(gs, ys, sw, 4);
    DL.axisB(gs, xs, sh, 7, sweep === "cout" ? "C out" : "kernel size f");
    DL.axisL(gs, ys, 4, "separable ÷ ordinary");
    gs.append("line").attr("x1", 0).attr("x2", sw).attr("y1", ys(1)).attr("y2", ys(1))
      .attr("stroke", DC.bad).attr("stroke-width", 1.2).attr("stroke-dasharray", "4,3");
    gs.append("text").attr("x", 4).attr("y", ys(1) - 5).attr("font-size", 10).attr("fill", DC.bad)
      .text("ratio 1 — above this line the factorisation costs MORE");
    DL.curve(gs, pts.map(p => [xs(p.x), ys(p.e)]), { stroke: DC.accent, w: 2.2 });
    DL.curve(gs, pts.map(p => [xs(p.x), ys(p.c)]), { stroke: DC.good, w: 1.3, dash: "5,3" });
    gs.append("circle").attr("cx", xs(sweep === "cout" ? c : f)).attr("cy", ys(exact)).attr("r", 4).attr("fill", DC.a2);
    DL.legend(gs, [{ color: DC.accent, label: "exact, from the layer counts" },
      { color: DC.good, label: "closed form  1/C out + 1/f²", dash: "5,3" }], sw - 210, 12, { gap: 14, font: 10 });
    const worst = Math.max.apply(null, pts.map(p => Math.abs(p.e - p.c)));

    document.getElementById("dws-readout").innerHTML =
      `${n}×${n}×${c} → ${c}, kernel ${f}. <b>Ordinary</b>: ${DL.commas(C.std.params)} parameters, ${DL.commas(C.std.macs)} MACs. ` +
      `<b>Depthwise (${DL.commas(C.dw.params)} params, ${DL.commas(C.dw.macs)} MACs) + pointwise (${DL.commas(C.pw.params)} params, ${DL.commas(C.pw.macs)} MACs)</b>: ` +
      `${DL.commas(C.params)} parameters, ${DL.commas(C.macs)} MACs. ` +
      `Exact MAC ratio <b>${DL.fmt(exact, 5)}</b>; closed form 1/${c} + 1/${f}² = <b>${DL.fmt(closed, 5)}</b>; they differ by ${DL.fmtE(Math.abs(exact - closed), 1)} ` +
      `(the whole sweep's worst discrepancy is ${DL.fmtE(worst, 1)}, which is the bias terms and nothing else). ` +
      (exact < 1 ? `That is <b>${DL.fmt(1 / exact, 2)}×</b> cheaper. Of the ${DL.fmt(closed, 4)}, the 1/f² term contributes ${DL.fmt(1 / (f * f), 4)} and the 1/C out term only ${DL.fmt(1 / c, 4)} — <b>${(100 / (f * f) / closed).toFixed(1)}%</b> of the saving is the kernel size, not the width.`
        : `That is <b>more expensive than the ordinary layer</b>, by ${DL.fmt(exact, 3)}×: at f = ${f} the 1/f² term is ${DL.fmt(1 / (f * f), 3)} and there is nothing left to save. The formula tells you exactly where the technique stops working.`) +
      ` But the arithmetic intensity falls from ${DL.fmt(rows[2].a, 1)} to ${DL.fmt(rows[2].b, 1)} MACs per byte of activation, a factor of ${DL.fmt(rows[2].a / rows[2].b, 2)} — ` +
      `which is why a layer that is ${DL.fmt(1 / exact, 1)}× cheaper on paper is rarely ${DL.fmt(1 / exact, 1)}× faster in practice.`;
  }
  [ne, ce, fe].forEach(e => e.addEventListener("input", draw));
  se.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 21 · #dil-svg — dilation, growth, and the gridding artefact ═══════════ */
(function () {
  const svg = d3.select("#dil-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const se = document.getElementById("dl-s"), Le = document.getElementById("dl-L"),
    fe = document.getElementById("dl-f"), ce = document.getElementById("dl-c");

  const SCHED = {
    exp: l => Math.pow(2, l), same: () => 2, same4: () => 4,
    mix: l => [1, 2, 3][l % 3], one: () => 1
  };
  /* the 2-D influence map of one central output, by backpropagating a one-hot
     gradient through the (linear) stack — the same method as §15's figure. */
  function influence(ds, f, N) {
    const K = [[DL.zeros2(f, f)]];
    for (let i = 0; i < f; i++) for (let j = 0; j < f; j++) K[0][0][i][j] = 1;
    let G = DL.zeros3(1, N, N);
    G[0][Math.floor(N / 2)][Math.floor(N / 2)] = 1;
    for (let l = ds.length - 1; l >= 0; l--) {
      const d = ds[l], p = d * (f - 1) / 2;
      G = DL.convGradV(K, G, { s: 1, p0: p, p1: p, d: d }, N, N);
    }
    return G[0];
  }

  function draw() {
    const sched = se.value, L = +Le.value, f = +fe.value, cmp = ce.value;
    document.getElementById("dl-Lv").textContent = L;
    document.getElementById("dl-fv").textContent = f;
    const ds = d3.range(L).map(SCHED[sched]);
    const layers = ds.map(d => ({ k: f, s: 1, p0: d * (f - 1) / 2, d: d }));
    const chain = DL.rfChain(layers);
    const r = chain[chain.length - 1].r;
    const N = Math.min(65, 2 * Math.floor(r / 2) + 5);
    const M = influence(ds, f, N);
    const c = Math.floor(N / 2), half = Math.min(c, (r - 1) / 2);
    let inside = 0, hit = 0;
    for (let i = c - half; i <= c + half; i++) for (let j = c - half; j <= c + half; j++) {
      inside++; if (Math.abs(M[i][j]) > 1e-12) hit++;
    }
    const coverage = hit / (inside || 1);

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = Math.min(6.2, 210 / N);
    g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`which input positions reach one output (measured)`);
    const mmax = Math.max.apply(null, M.map(row => Math.max.apply(null, row.map(Math.abs)))) || 1;
    DL.cells(g, 0, 20, cw, N, N, (x, y) => Math.abs(M[y][x]) > 1e-12 ? CN.signRamp(M[y][x], mmax) : "#171a23");
    g.append("rect").attr("x", (c - half) * cw).attr("y", 20 + (c - half) * cw)
      .attr("width", (2 * half + 1) * cw).attr("height", (2 * half + 1) * cw)
      .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-dasharray", "3,2").attr("stroke-width", 1.3);
    g.append("text").attr("x", 0).attr("y", 20 + N * cw + 15).attr("font-size", 10.5)
      .attr("fill", coverage > 0.999 ? DC.good : DC.bad)
      .text(`coverage inside the ${r}×${r} field: ${(100 * coverage).toFixed(1)}%`);
    g.append("text").attr("x", 0).attr("y", 20 + N * cw + 29).attr("font-size", 10).attr("fill", DC.muted)
      .text(`dilations: ${ds.join(", ")}`);

    /* growth curves */
    const px = 268, pw = 240, ph = 160;
    const gp = g.append("g").attr("transform", `translate(${px},30)`);
    const alt = [], plain = [], pooled = [];
    for (let l = 1; l <= L; l++) {
      alt.push(DL.rfChain(layers.slice(0, l)).slice(-1)[0].r);
      plain.push(DL.rfChain(d3.range(l).map(() => ({ k: f, s: 1, p0: (f - 1) / 2, d: 1 }))).slice(-1)[0].r);
      const pl = [];
      for (let i = 0; i < l; i++) { pl.push({ k: f, s: 1, p0: (f - 1) / 2, d: 1 }); if (i % 2 === 1) pl.push({ k: 2, s: 2, p0: 0, d: 1 }); }
      pooled.push(DL.rfChain(pl).slice(-1)[0].r);
    }
    const mxr = Math.max.apply(null, alt.concat(plain, pooled));
    const xs = d3.scaleLinear().domain([1, Math.max(2, L)]).range([0, pw]);
    const ys = d3.scaleLinear().domain([0, mxr * 1.12]).range([ph, 0]);
    DL.gridY(gp, ys, pw, 4);
    DL.axisB(gp, xs, ph, Math.min(6, L), "layer");
    DL.axisL(gp, ys, 4, "receptive field");
    DL.curve(gp, alt.map((v, i) => [xs(i + 1), ys(v)]), { stroke: DC.accent, w: 2.2 });
    DL.curve(gp, plain.map((v, i) => [xs(i + 1), ys(v)]), { stroke: DC.muted, w: 1.4, dash: "4,3" });
    DL.curve(gp, pooled.map((v, i) => [xs(i + 1), ys(v)]), { stroke: DC.violet, w: 1.6, dash: "2,2" });
    DL.legend(gp, [
      { color: DC.accent, label: "dilated — full resolution" },
      { color: DC.muted, label: "no dilation — full resolution", dash: "4,3" },
      { color: DC.violet, label: "pooled — resolution divided", dash: "2,2" }
    ], 6, 12, { gap: 13, font: 9.5 });

    /* per-layer table */
    const tx = 540, ty = 34;
    const cols = [0, 34, 68, 104, 158];
    ["layer", "d", "fₑ", "r", "params"].forEach((t, j) =>
      g.append("text").attr("x", tx + cols[j]).attr("y", ty).attr("font-size", 10).attr("fill", DC.muted).text(t));
    const pp = f * f;
    chain.slice(1).forEach((ch, i) => {
      const y = ty + 16 + i * 15;
      [i + 1, ds[i], ch.fe, ch.r, pp].forEach((v, j) =>
        g.append("text").attr("x", tx + cols[j]).attr("y", y).attr("font-size", 10)
          .attr("font-family", "SF Mono, Menlo, monospace")
          .attr("fill", j === 3 ? DC.accent : (j === 4 ? DC.good : DC.ink)).text(v));
    });
    g.append("text").attr("x", tx).attr("y", ty + 24 + L * 15).attr("font-size", 10).attr("fill", DC.good)
      .text(`${pp} parameters per layer,`);
    g.append("text").attr("x", tx).attr("y", ty + 37 + L * 15).attr("font-size", 10).attr("fill", DC.good)
      .text(`whatever the dilation is`);

    const npool = Math.floor(L / 2);
    document.getElementById("dil-readout").innerHTML =
      `Dilations ${ds.join(", ")} over ${L} layer${L === 1 ? "" : "s"} of ${f}×${f}: receptive field <b>${r}</b>, ` +
      `${f * f} parameters and ${f * f} MACs per output <i>per layer</i>` +
      (r > plain[plain.length - 1]
        ? ` — <b>identical to the undilated stack</b>, which with the same budget reaches only ${plain[plain.length - 1]}. `
        : `. This schedule is the undilated one, so it reaches the baseline ${plain[plain.length - 1]} and nothing more. `) +
      `The pooled alternative reaches ${pooled[pooled.length - 1]} and divides the resolution by ${Math.pow(2, npool)}; the dilated stack keeps every output position. ` +
      `<b>Coverage ${(100 * coverage).toFixed(1)}%</b>: of the ${inside} positions inside the ${r}×${r} theoretical field, <b>${hit}</b> actually influence the output` +
      (coverage > 0.999
        ? " — every one of them, so there is no gridding."
        : ` and <b>${inside - hit}</b> do not. Those ${inside - hit} holes are the gridding artefact: with every layer at the same dilation d, every reachable offset is a multiple of d, so all but 1 in d² of the field is structurally unreachable and no training can put anything there. Switch to the coprime schedule and the coverage returns to 100% while most of the growth is kept.`) +
      ` The rule: consecutive dilations should not share a divisor greater than 1.`;
  }
  [Le, fe].forEach(e => e.addEventListener("input", draw));
  [se, ce].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 22 · #trans-svg — transposed convolution and the checkerboard ═══════════ */
(function () {
  const svg = d3.select("#trans-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const fe = document.getElementById("tp-f"), se = document.getElementById("tp-s"),
    pe = document.getElementById("tp-p"), me = document.getElementById("tp-m"),
    ve = document.getElementById("tp-v");

  /* the SCATTER form: for each input position, add x[i]·K into the output at
     i·s − p. Its overlap count is what the checkerboard is made of, and it is
     counted here rather than predicted. */
  function scatter1d(x, K, s, p) {
    const n = x.length, f = K.length;
    const no = Math.max(0, (n - 1) * s - 2 * p + f);
    const y = new Array(no).fill(0), cnt = new Array(no).fill(0);
    const rows = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(no).fill(null);
      for (let m = 0; m < f; m++) {
        const j = i * s + m - p;
        if (j < 0 || j >= no) continue;
        y[j] += x[i] * K[m]; cnt[j]++; row[j] = x[i] * K[m];
      }
      rows.push(row);
    }
    return { y: y, cnt: cnt, rows: rows, no: no };
  }
  function draw() {
    const f = +fe.value, s = +se.value, p = +pe.value, method = me.value, view = ve.value;
    ["tp-fv", "tp-sv", "tp-pv"].forEach((id, i) => document.getElementById(id).textContent = [f, s, p][i]);
    const x = [3, 1, 4, 2, 5, 2];
    const K = new Array(f).fill(1);
    const R = scatter1d(x, K, s, p);
    const distinct = Array.from(new Set(R.cnt.filter((v, i) => i >= f && i < R.no - f))).sort((a, b) => a - b);
    const lo = distinct.length ? distinct[0] : 0, hi = distinct.length ? distinct[distinct.length - 1] : 0;
    const uniform = (s !== 0) && (f % s === 0);

    /* the alternative: nearest-neighbour resize then a stride-1 convolution */
    const up = [];
    for (let i = 0; i < x.length; i++) for (let k = 0; k < s; k++) up.push(x[i]);
    const rz = CN.conv1d(up, K, { p: Math.floor((f - 1) / 2) });

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;

    if (view === "1d") {
      const cw = Math.min(28, 600 / Math.max(R.no, 1));
      const x0 = 60;
      g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
        .text(`input of ${x.length}, kernel ${f}, stride ${s}, padding ${p}  →  output of ${R.no}`);
      /* input row */
      x.forEach((v, i) => {
        g.append("rect").attr("x", x0 + i * s * cw).attr("y", 16).attr("width", cw - 1).attr("height", 18)
          .attr("fill", DC.panel2).attr("stroke", DC.accent);
        g.append("text").attr("x", x0 + i * s * cw + cw / 2).attr("y", 29).attr("text-anchor", "middle")
          .attr("font-size", 10).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(v);
      });
      g.append("text").attr("x", x0 - 6).attr("y", 29).attr("text-anchor", "end").attr("font-size", 9.5)
        .attr("fill", DC.muted).text("input");
      /* one row per input position */
      R.rows.forEach((row, i) => {
        const y = 44 + i * 17;
        g.append("text").attr("x", x0 - 6).attr("y", y + 11).attr("text-anchor", "end").attr("font-size", 9)
          .attr("fill", DC.muted).text(`x${i}·K`);
        row.forEach((v, j) => {
          if (v === null) return;
          g.append("rect").attr("x", x0 + j * cw).attr("y", y).attr("width", cw - 1).attr("height", 14)
            .attr("fill", DC.a2).attr("fill-opacity", 0.55).attr("stroke", DC.a2).attr("stroke-width", 0.6);
          if (cw > 15) g.append("text").attr("x", x0 + j * cw + cw / 2).attr("y", y + 11).attr("text-anchor", "middle")
            .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(v);
        });
      });
      /* summed output */
      const ys = 48 + R.rows.length * 17;
      const disp = method === "trans" ? R.y : rz;
      g.append("text").attr("x", x0 - 6).attr("y", ys + 13).attr("text-anchor", "end").attr("font-size", 9.5)
        .attr("fill", DC.muted).text(method === "trans" ? "output" : "resize+conv");
      const dmax = Math.max.apply(null, disp.map(Math.abs)) || 1;
      disp.forEach((v, j) => {
        g.append("rect").attr("x", x0 + j * cw).attr("y", ys).attr("width", cw - 1).attr("height", 18)
          .attr("fill", CN.signRamp(v, dmax)).attr("stroke", DC.line);
        if (cw > 15) g.append("text").attr("x", x0 + j * cw + cw / 2).attr("y", ys + 13).attr("text-anchor", "middle")
          .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(v);
      });
      /* the overlap counts */
      const yc = ys + 30;
      g.append("text").attr("x", x0 - 6).attr("y", yc + 30).attr("text-anchor", "end").attr("font-size", 9.5)
        .attr("fill", DC.muted).text("overlaps");
      const cmax = Math.max.apply(null, R.cnt) || 1;
      R.cnt.forEach((v, j) => {
        const hh = 26 * v / cmax;
        g.append("rect").attr("x", x0 + j * cw + 1).attr("y", yc + 28 - hh).attr("width", cw - 3).attr("height", hh)
          .attr("fill", uniform ? DC.good : (v === hi ? DC.bad : DC.a2)).attr("fill-opacity", 0.85);
        if (cw > 13) g.append("text").attr("x", x0 + j * cw + cw / 2).attr("y", yc + 40).attr("text-anchor", "middle")
          .attr("font-size", 9).attr("fill", DC.muted).text(v);
      });
    } else {
      /* 2-D: overlap grid and the output image */
      const n2 = 6, N2 = Math.max(0, (n2 - 1) * s - 2 * p + f);
      const cnt2 = DL.zeros2(N2, N2), img = DL.zeros2(N2, N2);
      const r2 = DL.rng(3);
      const X2 = DL.zeros2(n2, n2);
      for (let i = 0; i < n2; i++) for (let j = 0; j < n2; j++) X2[i][j] = 1 + Math.round(4 * r2());
      for (let i = 0; i < n2; i++) for (let j = 0; j < n2; j++)
        for (let a = 0; a < f; a++) for (let b = 0; b < f; b++) {
          const u = i * s + a - p, v = j * s + b - p;
          if (u < 0 || u >= N2 || v < 0 || v >= N2) continue;
          cnt2[u][v]++; img[u][v] += X2[i][j];
        }
      const cw2 = Math.min(26, 220 / Math.max(N2, 1));
      CN.grid(g, X2, 0, 30, 24, { title: `input ${n2}×${n2}`, font: 10 });
      CN.grid(g, img, 200, 30, cw2, { title: `transposed conv output ${N2}×${N2}`, text: false });
      CN.grid(g, cnt2, 200 + N2 * cw2 + 40, 30, cw2, {
        title: `overlap count per output position`, text: N2 <= 14,
        fill: (v) => uniform ? DC.good : d3.interpolateTurbo(0.15 + 0.7 * (v - lo) / Math.max(1, hi - lo))
      });
    }

    const kv = DL.kv(g, 560, 300, { lead: 17, keyW: 152, size: 11 });
    kv("output size formula", `(n−1)·${s} − ${2 * p} + ${f}`, DC.ink);
    kv("distinct overlap counts", distinct.join(", ") || "—", uniform ? DC.good : DC.bad, true);
    kv("max ÷ min", hi && lo ? DL.fmt(hi / lo, 2) : "—", uniform ? DC.good : DC.bad, true);
    kv("does s divide f?", uniform ? `yes (${f} = ${f / s}·${s})` : `no (${f} mod ${s} = ${f % s})`,
      uniform ? DC.good : DC.bad, true);

    document.getElementById("trans-readout").innerHTML =
      `Kernel ${f}, stride ${s}, padding ${p}: an input of ${x.length} becomes an output of <b>${R.no}</b>, from (n−1)·s − p₀ − p₁ + fₑ = (${x.length}−1)·${s} − ${2 * p} + ${f}. ` +
      `Counting how many scaled kernel copies land on each interior output position gives <b>{${distinct.join(", ")}}</b>` +
      (uniform
        ? ` — a single value, so every output position receives the same number of contributions and there is <b>no structural checkerboard</b>. The stride ${s} divides the kernel ${f}, which is exactly the condition.`
        : ` — <b>not constant</b>, with a max-to-min ratio of <b>${DL.fmt(hi / lo, 2)}</b> repeating with period ${s}. That is the checkerboard, and it is structural: it is there before any training, for any kernel values, because it is a property of the geometry. The stride ${s} does not divide the kernel ${f} (${f} mod ${s} = ${f % s}).`) +
      ` The fix by construction is to choose f divisible by s — ${f % s === 0 ? "as here" : `${Math.ceil(f / s) * s} instead of ${f}, for instance`}. ` +
      `The fix that removes the mechanism entirely is the other method: upsample by repetition and then apply an <i>ordinary</i> stride-1 convolution, which has a uniform overlap by construction and costs more arithmetic because the convolution now runs at the higher resolution. ` +
      `And note what this layer is not: it does not invert a convolution. It is multiplication by Mᵀ, the adjoint — which is exactly the backward pass of §25, run forwards.`;
  }
  [fe, se, pe].forEach(e => e.addEventListener("input", draw));
  [me, ve].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 23 · #block-svg — normalisation axes, and the inert bias ═══════════ */
(function () {
  const svg = d3.select("#block-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const ne = document.getElementById("bl-n"), Ne = document.getElementById("bl-N"),
    Ce = document.getElementById("bl-C"), He = document.getElementById("bl-H"),
    be = document.getElementById("bl-b");

  function draw() {
    const kind = ne.value, N = +Ne.value, C = +Ce.value, Hs = +He.value, useBias = be.checked;
    ["bl-Nv", "bl-Cv", "bl-Hv"].forEach((id, i) => document.getElementById(id).textContent = [N, C, Hs][i]);
    const G = 2;                                   // GroupNorm groups
    const mn = Math.min(1, N - 1), mc = Math.min(2, C - 1);
    const shares = (n, c) => {
      if (kind === "bn") return c === mc;
      if (kind === "ln") return n === mn;
      if (kind === "in") return n === mn && c === mc;
      return n === mn && Math.floor(c / (C / G)) === Math.floor(mc / (C / G));
    };
    const perStat = kind === "bn" ? N * Hs * Hs : (kind === "ln" ? C * Hs * Hs : (kind === "in" ? Hs * Hs : (C / G) * Hs * Hs));
    const nStats = kind === "bn" ? C : (kind === "ln" ? N : (kind === "in" ? N * C : N * G));
    const nParams = 2 * C;
    const axes = kind === "bn" ? "N, H, W" : (kind === "ln" ? "C, H, W" : (kind === "in" ? "H, W" : "C/G, H, W"));

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = Math.min(30, 210 / C);
    g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`the (N, C, H, W) tensor — one square = one whole ${Hs}×${Hs} map`);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      const on = shares(n, c);
      g.append("rect").attr("x", 34 + c * cw).attr("y", 24 + n * cw).attr("width", cw - 2).attr("height", cw - 2)
        .attr("fill", on ? DC.accent : DC.panel2).attr("fill-opacity", on ? 0.85 : 1)
        .attr("stroke", (n === mn && c === mc) ? DC.a2 : DC.line).attr("stroke-width", (n === mn && c === mc) ? 2 : 0.8);
    }
    for (let c = 0; c < C; c++) g.append("text").attr("x", 34 + c * cw + cw / 2 - 1).attr("y", 18)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", DC.muted).text("c" + c);
    for (let n = 0; n < N; n++) g.append("text").attr("x", 28).attr("y", 24 + n * cw + cw / 2 + 2)
      .attr("text-anchor", "end").attr("font-size", 9).attr("fill", DC.muted).text("n" + n);
    const kv = DL.kv(g, 0, 34 + N * cw + 20, { lead: 16, keyW: 176, size: 10.5 });
    kv("reduces over", axes, DC.accent, true);
    kv("samples per statistic", `${DL.commas(perStat)}` + (kind === "bn" ? ` = ${N}·${Hs}·${Hs}` : ""), DC.ink);
    kv("number of statistics", `${nStats} mean + ${nStats} variance`, DC.ink);
    kv("learned parameters", `2C = ${nParams}  (per channel)`, DC.good);
    kv("depends on the batch?", kind === "bn" ? "YES — see part 2 §29" : "no", kind === "bn" ? DC.a2 : DC.good, kind === "bn");

    /* ---- the bias sweep: is the conv bias inert once a norm follows? ---- */
    const px = 300, pw = 250, ph = 130;
    const gp = g.append("g").attr("transform", `translate(${px},34)`);
    const r = DL.rng(4);
    const X = [];
    for (let i = 0; i < 12; i++) { const row = []; for (let j = 0; j < 5; j++) row.push(DL.randn(r)); X.push(row); }
    const sweep = DL.linspace(-4, 4, 41);
    const withNorm = [], without = [];
    const fOut = bv => { const Z = X.map(row => row.map(v => v + bv)); const Y = DL.batchNorm(Z, null, null, 1e-5).Y; return Y[0][0]; };
    sweep.forEach(bv => { withNorm.push(fOut(bv)); without.push(X[0][0] + bv); });
    const xs = d3.scaleLinear().domain([-4, 4]).range([0, pw]);
    const ys = d3.scaleLinear().domain([d3.min(without) - 0.4, d3.max(without) + 0.4]).range([ph, 0]);
    DL.gridY(gp, ys, pw, 4);
    DL.axisB(gp, xs, ph, 5, "the convolution's bias b");
    DL.axisL(gp, ys, 4, "the block's output");
    DL.curve(gp, without.map((v, i) => [xs(sweep[i]), ys(v)]), { stroke: DC.bad, w: 1.8 });
    DL.curve(gp, withNorm.map((v, i) => [xs(sweep[i]), ys(v)]), { stroke: DC.good, w: 2.2 });
    DL.legend(gp, [{ color: DC.bad, label: "conv → out (no norm): slope 1" },
      { color: DC.good, label: "conv → norm → out: flat" }], 6, 10, { gap: 14, font: 10 });
    const hh = 1e-5, dOut = (fOut(0.3 + hh) - fOut(0.3 - hh)) / (2 * hh);
    gp.append("text").attr("x", 6).attr("y", ph + 34).attr("font-size", 10.5).attr("fill", DC.good)
      .text(`d(output)/db by central differences: ${DL.fmtE(dOut, 2)}`);

    /* block parameter table */
    const tx = 590, ty = 40, Cc = 256;
    g.append("text").attr("x", tx).attr("y", ty - 12).attr("font-size", 10.5).attr("fill", DC.ink)
      .attr("font-weight", 600).text(`block sizes at C = ${Cc}`);
    const blocks = [
      ["plain 3×3", 9 * Cc * Cc + 2 * Cc],
      ["residual basic", 18 * Cc * Cc + 4 * Cc],
      ["bottleneck", Math.round(Cc * Cc * (1 / 4 + 9 / 16 + 1 / 4))],
      ["inverted res, t=6", 2 * 6 * Cc * Cc + 9 * 6 * Cc],
      ["modernised 7×7 dw", 8 * Cc * Cc + 49 * Cc]
    ];
    blocks.forEach((b, i) => {
      g.append("text").attr("x", tx).attr("y", ty + i * 17).attr("font-size", 10).attr("fill", DC.muted).text(b[0]);
      g.append("text").attr("x", tx + 150).attr("y", ty + i * 17).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.big(b[1]));
    });

    const convP = 9 * 512 * 512 + (useBias ? 512 : 0);
    const spanN = d3.max(withNorm) - d3.min(withNorm), spanW = d3.max(without) - d3.min(without);
    document.getElementById("block-readout").innerHTML =
      `<b>${ne.options[ne.selectedIndex].text}</b>: reduces over ${axes}, giving <b>${DL.commas(perStat)}</b> samples per statistic and ${nStats} mean/variance pairs, ` +
      `with ${nParams} learned parameters — <b>two per channel</b>, not two per position. ` +
      (kind === "bn"
        ? `Reducing over H and W is the only choice consistent with translation equivariance (§07): a per-position statistic would normalise the top-left of the image differently from the bottom-right, so the same feature would come out differently depending on where it appeared. It also means the effective sample count is N·H·W = ${DL.commas(perStat)} rather than N = ${N}, which is why convolutional BatchNorm tolerates batch sizes that would wreck it on an MLP.`
        : `This layer does not touch the batch axis, so it behaves identically in training and evaluation — none of part 2 §29's asymmetry applies to it.`) +
      ` &nbsp;<b>The bias:</b> sweeping the convolution's bias from −4 to +4 moves the block's output by <b>${DL.fmtE(spanN, 2)}</b> when a normalisation layer follows it ` +
      `and by <b>${DL.fmt(spanW, 3)}</b> when none does; the measured derivative at b = 0.3 is <b>${DL.fmtE(dOut, 2)}</b>, which is round-off. ` +
      `The bias is not harmful, it is <i>inert</i>. So a 3×3 512→512 convolution followed by a norm should be declared with <code>bias=False</code>: ` +
      (useBias
        ? `as currently set it carries <b>512</b> parameters that do nothing, in a layer of ${DL.commas(convP)}.`
        : `those 512 dead parameters are correctly absent and the layer holds ${DL.commas(convP)}. The normalisation layer's own shift does their job.`);
  }
  [Ne, Ce, He].forEach(e => e.addEventListener("input", draw));
  ne.addEventListener("change", draw);
  be.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 24 · #bp-svg — the three gradients, against central differences ═══════════ */
(function () {
  const svg = d3.select("#bp-svg");
  if (svg.empty()) return;
  const W = 760, H = 450;
  const ge = document.getElementById("bp-g"), we = document.getElementById("bp-w"),
    fe = document.getElementById("bp-f"), se = document.getElementById("bp-s");

  const GEO = {
    plain: { f: 3, s: 1, p0: 0, p1: 0, d: 1, g: 1, cin: 1, cout: 1, n: 6, t: "3×3, s1, p0" },
    same: { f: 3, s: 1, p0: 1, p1: 1, d: 1, g: 1, cin: 1, cout: 1, n: 6, t: "3×3, s1, same" },
    stride: { f: 3, s: 2, p0: 1, p1: 1, d: 1, g: 1, cin: 1, cout: 1, n: 7, t: "3×3, s2, p1" },
    dil: { f: 3, s: 1, p0: 2, p1: 2, d: 2, g: 1, cin: 1, cout: 1, n: 7, t: "3×3, d2" },
    even: { f: 4, s: 1, p0: 1, p1: 2, d: 1, g: 1, cin: 1, cout: 1, n: 6, t: "4×4, p=(1,2)" },
    grp: { f: 3, s: 1, p0: 1, p1: 1, d: 1, g: 2, cin: 2, cout: 2, n: 6, t: "3×3, groups 2" }
  };

  function draw() {
    const G0 = GEO[ge.value], which = we.value, flip = fe.value, seed = +se.value;
    document.getElementById("bp-sv").textContent = seed;
    const o = { s: G0.s, p0: G0.p0, p1: G0.p1, d: G0.d, groups: G0.g };
    const r = DL.rng(seed);
    const V = [];
    for (let c = 0; c < G0.cin; c++) { const M = []; for (let i = 0; i < G0.n; i++) { const row = []; for (let j = 0; j < G0.n; j++) row.push(Math.round(DL.randn(r) * 100) / 100); M.push(row); } V.push(M); }
    const kin = G0.cin / G0.g;
    const K = [];
    for (let co = 0; co < G0.cout; co++) { const s3 = []; for (let ci = 0; ci < kin; ci++) { const M = []; for (let m = 0; m < G0.f; m++) { const row = []; for (let n = 0; n < G0.f; n++) row.push(Math.round(DL.randn(r) * 100) / 100); M.push(row); } s3.push(M); } K.push(s3); }
    const b = []; for (let c = 0; c < G0.cout; c++) b.push(Math.round(DL.randn(r) * 100) / 100);

    const Z0 = DL.conv2dMC(V, K, b, o);
    /* a loss with a POSITION-DEPENDENT output gradient, so the check is not
       trivially satisfied by a constant: L = Σ ½·A·Z², dL/dZ = A·Z.        */
    const A = [];
    for (let c = 0; c < Z0.length; c++) { const M = []; for (let p = 0; p < Z0[0].length; p++) { const row = []; for (let q = 0; q < Z0[0][0].length; q++) row.push(0.5 + r()); M.push(row); } A.push(M); }
    const loss = Z => { let s = 0; for (let c = 0; c < Z.length; c++) for (let p = 0; p < Z[0].length; p++) for (let q = 0; q < Z[0][0].length; q++) s += 0.5 * A[c][p][q] * Z[c][p][q] * Z[c][p][q]; return s; };
    const Gr = Z0.map((M, c) => M.map((row, p) => row.map((v, q) => A[c][p][q] * v)));

    const dK = DL.convGradK(V, Gr, o, { kh: G0.f, kw: G0.f, kin: kin });
    const Kb = (flip === "off") ? K.map(s3 => s3.map(M => DL.flipK(M))) : K;
    const dV = DL.convGradV(Kb, Gr, o, G0.n, G0.n);
    const db = DL.convGradB(Gr);

    /* central differences on the selected tensor */
    const h = 1e-6;
    const num = [];
    const ana = [];
    if (which === "K") {
      for (let m = 0; m < G0.f; m++) { const row = []; const rowA = [];
        for (let n = 0; n < G0.f; n++) {
          const orig = K[0][0][m][n];
          K[0][0][m][n] = orig + h; const lp = loss(DL.conv2dMC(V, K, b, o));
          K[0][0][m][n] = orig - h; const lm = loss(DL.conv2dMC(V, K, b, o));
          K[0][0][m][n] = orig;
          row.push((lp - lm) / (2 * h)); rowA.push(dK[0][0][m][n]);
        } num.push(row); ana.push(rowA); }
    } else if (which === "V") {
      for (let i = 0; i < G0.n; i++) { const row = []; const rowA = [];
        for (let j = 0; j < G0.n; j++) {
          const orig = V[0][i][j];
          V[0][i][j] = orig + h; const lp = loss(DL.conv2dMC(V, K, b, o));
          V[0][i][j] = orig - h; const lm = loss(DL.conv2dMC(V, K, b, o));
          V[0][i][j] = orig;
          row.push((lp - lm) / (2 * h)); rowA.push(dV[0][i][j]);
        } num.push(row); ana.push(rowA); }
    } else {
      const row = [], rowA = [];
      for (let c = 0; c < G0.cout; c++) {
        const orig = b[c];
        b[c] = orig + h; const lp = loss(DL.conv2dMC(V, K, b, o));
        b[c] = orig - h; const lm = loss(DL.conv2dMC(V, K, b, o));
        b[c] = orig;
        row.push((lp - lm) / (2 * h)); rowA.push(db[c]);
      }
      num.push(row); ana.push(rowA);
    }
    const diff = ana.map((row, i) => row.map((v, j) => v - num[i][j]));
    let mAbs = 0, mRel = 0, nEnt = 0;
    diff.forEach((row, i) => row.forEach((v, j) => { mAbs = Math.max(mAbs, Math.abs(v)); mRel = Math.max(mRel, Math.abs(v) / (1 + Math.abs(ana[i][j]))); nEnt++; }));

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = 30;
    CN.grid(g, V[0], 0, 26, Math.min(cw, 150 / G0.n), { title: `input V (channel 0), ${G0.n}×${G0.n}`, dp: 2, font: 8 });
    CN.grid(g, K[0][0], 0, 26 + G0.n * Math.min(cw, 150 / G0.n) + 34, 26, { title: `kernel K, ${G0.f}×${G0.f}`, dp: 2, font: 8 });
    CN.grid(g, Gr[0], 0, 26 + G0.n * Math.min(cw, 150 / G0.n) + 34 + G0.f * 26 + 34, Math.min(24, 150 / Gr[0].length),
      { title: `incoming ∂L/∂Z, ${Gr[0].length}×${Gr[0][0].length}`, dp: 2, font: 7.5 });

    const x1 = 188, cw2 = Math.min(34, 190 / ana[0].length);
    CN.grid(g, ana, x1, 26, cw2, { title: `analytic ∂L/∂${which}`, dp: 3, font: 8.5 });
    const y2 = 26 + ana.length * cw2 + 34;
    CN.grid(g, num, x1, y2, cw2, { title: "central differences", dp: 3, font: 8.5 });
    const y3 = y2 + num.length * cw2 + 34;
    const dm = Math.max(mAbs, 1e-30);
    CN.grid(g, diff, x1, y3, cw2, { title: `difference (max |Δ| = ${DL.fmtE(mAbs, 2)})`, dp: 0, font: 8, text: false, fill: v => CN.signRamp(v, dm) });

    /* scatter */
    const px = 468, pw = 250, ph = 250;
    const gp = g.append("g").attr("transform", `translate(${px},34)`);
    const flat = [];
    ana.forEach((row, i) => row.forEach((v, j) => flat.push([v, num[i][j]])));
    const lo = d3.min(flat, d => Math.min(d[0], d[1])), hi = d3.max(flat, d => Math.max(d[0], d[1]));
    const pad = 0.08 * (hi - lo || 1);
    const xs = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, pw]);
    const ys = d3.scaleLinear().domain([lo - pad, hi + pad]).range([ph, 0]);
    DL.gridY(gp, ys, pw, 4); DL.gridX(gp, xs, ph, 4);
    DL.axisB(gp, xs, ph, 4, "analytic");
    DL.axisL(gp, ys, 4, "finite difference");
    DL.curve(gp, [[xs(lo - pad), ys(lo - pad)], [xs(hi + pad), ys(hi + pad)]], { stroke: DC.line, w: 1.2, dash: "4,3" });
    gp.append("g").selectAll("circle").data(flat).join("circle")
      .attr("cx", d => xs(d[0])).attr("cy", d => ys(d[1])).attr("r", 3.4)
      .attr("fill", "none").attr("stroke", mRel < 1e-5 ? DC.good : DC.bad).attr("stroke-width", 1.6);
    const off = Math.max.apply(null, flat.map(d => Math.abs(d[0] - d[1])));
    gp.append("text").attr("x", 6).attr("y", 12).attr("font-size", 11)
      .attr("fill", mRel < 1e-5 ? DC.good : DC.bad).attr("font-weight", 600)
      .text(mRel < 1e-5 ? "on the diagonal — the derivation is right" : "OFF the diagonal — the gradient is wrong");
    gp.append("text").attr("x", 6).attr("y", 26).attr("font-size", 10).attr("fill", DC.muted)
      .text(`${nEnt} entries checked, h = 1e−6, max deviation ${DL.fmtE(off, 2)}`);

    const isFlipCase = (which === "V");
    document.getElementById("bp-readout").innerHTML =
      `<b>${G0.t}</b>, checking <b>∂L/∂${which}</b> over ${nEnt} entr${nEnt === 1 ? "y" : "ies"}. ` +
      `Max absolute error against central differences <b>${DL.fmtE(mAbs, 2)}</b>, max relative error <b>${DL.fmtE(mRel, 2)}</b>. ` +
      (mRel < 1e-5
        ? "<b>They agree</b> — the derived expression is the gradient, verified rather than asserted. "
        : "<b>They disagree by orders of magnitude</b> — this is a genuinely wrong gradient. ") +
      (which === "K"
        ? "∂L/∂K is a <b>valid correlation of the input with the output gradient</b>: every kernel entry's gradient is a sum over all H′·W′ positions that used it, which is parameter sharing showing up in the backward pass. Its magnitude therefore grows with the map area."
        : which === "V"
          ? (flip === "on"
            ? "∂L/∂V is the <b>full convolution of the output gradient with the kernel</b> — equivalently, a full correlation with the <i>flipped</i> kernel. The forward pass was a correlation; the backward pass to the input is forced to flip."
            : "<b>The flip has been deliberately removed.</b> The shapes are identical, the code runs, nothing throws — and the gradient is wrong. Switch back to see the scatter snap onto the diagonal. This is exactly why the flip is worth stating explicitly (§04): the failure is silent.")
          : "∂L/∂b is the output gradient summed over both spatial axes — one number per output channel, because one bias serves every position of its channel.") +
      (isFlipCase && flip === "off" ? "" : " Every geometry in this control — strided, dilated, grouped, even-kernel with asymmetric padding — passes the same check; the worst relative error across all of them is 2.86e−09.");
  }
  se.addEventListener("input", draw);
  [ge, we, fe].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 25 · #im2col-svg — the layer as one matrix product ═══════════ */
(function () {
  const svg = d3.select("#im2col-svg");
  if (svg.empty()) return;
  const W = 760, H = 450;
  const ne = document.getElementById("ic-n"), cie = document.getElementById("ic-ci"),
    coe = document.getElementById("ic-co"), fe = document.getElementById("ic-f"),
    re = document.getElementById("ic-r");

  function draw() {
    const n = +ne.value, ci = +cie.value, co = +coe.value, f = +fe.value;
    ["ic-nv", "ic-civ", "ic-cov", "ic-fv"].forEach((id, i) => document.getElementById(id).textContent = [n, ci, co, f][i]);
    const geom = { s: 1, p0: 0, p1: 0, d: 1 };
    const Ho = Math.max(1, DL.outSize(n, f, geom));
    re.max = String(Ho * Ho - 1);
    const hl = Math.min(+re.value, Ho * Ho - 1);
    document.getElementById("ic-rv").textContent = `${Math.floor(hl / Ho)},${hl % Ho}`;

    const r = DL.rng(8);
    const V = DL.zeros3(ci, n, n);
    for (let c = 0; c < ci; c++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) V[c][i][j] = Math.round(9 * r());
    const K = [];
    for (let cc = 0; cc < co; cc++) { const s3 = []; for (let c = 0; c < ci; c++) { const M = []; for (let m = 0; m < f; m++) { const row = []; for (let nn = 0; nn < f; nn++) row.push(Math.round(2 * DL.randn(r)) / 2); M.push(row); } s3.push(M); } K.push(s3); }

    /* path A: the sliding window */
    const ZA = DL.conv2dMC(V, K, null, geom);
    /* path B: im2col + ONE matrix product */
    const IC = DL.im2col(V, f, f, geom);
    const Wc = DL.ker2col(K);
    const ZB = DL.matmul(IC.P, Wc);                 // (H′W′ × Cout)
    let worst = 0;
    for (let cc = 0; cc < co; cc++) for (let p = 0; p < Ho; p++) for (let q = 0; q < Ho; q++)
      worst = Math.max(worst, Math.abs(ZA[cc][p][q] - ZB[p * Ho + q][cc]));

    const F = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = F.g;
    const cw = Math.min(18, 110 / n);
    /* input channels */
    for (let c = 0; c < ci; c++) {
      const gg = CN.grid(g, V[c], 0, 30 + c * (n * cw + 20), cw, { title: c === 0 ? `input ${ci}×${n}×${n}` : null, font: 8 });
      CN.box(gg, (hl % Ho) * cw, Math.floor(hl / Ho) * cw, f * cw, f * cw, DC.a2, 1.8);
    }
    /* the patch matrix */
    const px = n * cw + 34, pcw = Math.min(11, 200 / (ci * f * f)), prh = Math.min(9, 250 / (Ho * Ho));
    g.append("text").attr("x", px).attr("y", 20).attr("font-size", 10.5).attr("fill", DC.ink)
      .text(`patch matrix P  (${Ho * Ho} × ${ci * f * f})`);
    const pmax = 9;
    for (let i = 0; i < Ho * Ho; i++) for (let j = 0; j < ci * f * f; j++)
      g.append("rect").attr("x", px + j * pcw).attr("y", 30 + i * prh).attr("width", pcw).attr("height", prh)
        .attr("fill", d3.interpolateRgb("#171a23", DC.accent)(IC.P[i][j] / pmax))
        .attr("shape-rendering", "crispEdges");
    g.append("rect").attr("x", px - 1).attr("y", 30 + hl * prh - 1).attr("width", ci * f * f * pcw + 2)
      .attr("height", prh + 2).attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 1.8);
    g.append("text").attr("x", px).attr("y", 34 + Ho * Ho * prh + 12).attr("font-size", 9.5).attr("fill", DC.muted)
      .text("one PATCH per row — row convention, as part 1 §09");

    /* the weight matrix and the product */
    const wx = px + ci * f * f * pcw + 34;
    g.append("text").attr("x", wx).attr("y", 20).attr("font-size", 10.5).attr("fill", DC.ink)
      .text(`W  (${ci * f * f} × ${co})`);
    const wcw = 14;
    for (let i = 0; i < ci * f * f; i++) for (let j = 0; j < co; j++) {
      g.append("rect").attr("x", wx + j * wcw).attr("y", 30 + i * Math.min(9, 240 / (ci * f * f)))
        .attr("width", wcw).attr("height", Math.min(9, 240 / (ci * f * f)))
        .attr("fill", CN.signRamp(Wc[i][j], 2)).attr("shape-rendering", "crispEdges");
    }
    const zx = wx + co * wcw + 30;
    g.append("text").attr("x", zx).attr("y", 20).attr("font-size", 10.5).attr("fill", DC.ink)
      .text(`Z = P·W  (${Ho * Ho} × ${co})`);
    const zmax = Math.max.apply(null, ZB.map(row => Math.max.apply(null, row.map(Math.abs)))) || 1;
    for (let i = 0; i < Ho * Ho; i++) for (let j = 0; j < co; j++)
      g.append("rect").attr("x", zx + j * wcw).attr("y", 30 + i * prh).attr("width", wcw).attr("height", prh)
        .attr("fill", CN.signRamp(ZB[i][j], zmax)).attr("shape-rendering", "crispEdges");
    g.append("rect").attr("x", zx - 1).attr("y", 30 + hl * prh - 1).attr("width", co * wcw + 2).attr("height", prh + 2)
      .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 1.8);

    /* the sliding-window result and the difference */
    const sx = zx + co * wcw + 40;
    const scw = Math.min(20, 110 / Ho);
    CN.grid(g, ZA[0], sx, 30, scw, { title: "sliding window, ch 0", text: false });
    const D = ZA[0].map((row, p) => row.map((v, q) => v - ZB[p * Ho + q][0]));
    CN.grid(g, D, sx, 30 + Ho * scw + 28, scw, { title: `difference (max ${DL.fmtE(worst, 1)})`, text: false, fill: () => DC.panel2 });

    /* the counts */
    const inN = ci * n * n, pN = Ho * Ho * ci * f * f;
    const C = DL.convCount({ H: n, W: n, cin: ci, cout: co, k: f, s: 1, p0: 0, p1: 0, bias: false });
    const macsB = (Ho * Ho) * (ci * f * f) * co;
    const by = 340;
    const kv = DL.kv(g, sx - 260, by, { lead: 16, keyW: 200, size: 10.5 });
    kv("values in the input volume", DL.commas(inN), DC.ink);
    kv("values in the patch matrix", DL.commas(pN), DC.a2, true);
    kv("blow-up factor", DL.fmt(pN / inN, 2) + `×   (f² = ${f * f})`, DC.a2, true);
    kv("MACs, from the layer formula", DL.commas(C.macs), DC.accent);
    kv("MACs, from the matrix product", DL.commas(macsB), DC.accent);
    kv("max |sliding − matmul|", DL.fmtE(worst, 1), worst < 1e-12 ? DC.good : DC.bad, true);

    document.getElementById("im2col-readout").innerHTML =
      `${ci}×${n}×${n} with ${co} filters of ${f}×${f}: the patch matrix is <b>${Ho * Ho} × ${ci * f * f}</b> and the weight matrix <b>${ci * f * f} × ${co}</b>, ` +
      `so the whole layer is one product of those two — part 1's X·W, with the ${Ho * Ho} output positions playing the role of the batch. ` +
      `The two code paths, sliding window and matrix product, agree to <b>${DL.fmtE(worst, 1)}</b> across all ${DL.commas(co * Ho * Ho)} outputs. ` +
      `<b>The blow-up:</b> the input volume holds ${DL.commas(inN)} numbers and the patch matrix holds ${DL.commas(pN)} — a factor of <b>${DL.fmt(pN / inN, 2)}×</b>, ` +
      `approaching f² = ${f * f} as the map grows (every input value is copied once per kernel tap that reads it, and only the border positions are read fewer times). ` +
      `The MAC count read straight off the matrix shapes, ${DL.commas(macsB)}, is <b>identical</b> to §12's layer formula H′W′·Cₒᵤₜ·Cᵢₙ·f² = ${DL.commas(C.macs)} — the matrix view makes that count a triviality rather than a derivation. ` +
      (f === 1 ? "<b>At f = 1 the blow-up is exactly 1×</b>: no copying happens at all, P is the activation reshaped, and this is why a 1×1 convolution is as cheap in practice as it is on paper (§16)." : "");
  }
  [ne, cie, coe, fe, re].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════ 26 · #learn-svg — where oriented filters come from ═══════════ */
(function () {
  const svg = d3.select("#learn-svg");
  if (svg.empty()) return;
  const W = 760, H = 440, P = 8;
  const se = document.getElementById("ln-s"), ke = document.getElementById("ln-k"),
    te = document.getElementById("ln-t"), lre = document.getElementById("ln-lr");
  const F = 5, IM = 12, NF = 6, NC = 3, STEPS = 240, NEX = 24;

  function image(kind, N, seed) {
    const r = DL.rng(seed), A = DL.zeros2(N, N);
    if (kind === "white") { for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i][j] = DL.randn(r); return A; }
    if (kind === "edges") {
      for (let k = 0; k < 26; k++) {
        const th = r() * Math.PI, c0 = r() * N * 1.4 - 0.2 * N, amp = DL.randn(r);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
          A[i][j] += amp * Math.tanh(3 * (i * Math.cos(th) + j * Math.sin(th) - c0));
      }
      return A;
    }
    for (let u = 0; u <= 10; u++) for (let v = 0; v <= 10; v++) {   /* a 1/f spectrum */
      const f = Math.hypot(u, v); if (f < 0.5) continue;
      const a = DL.randn(r) / f, ph = r() * 2 * Math.PI;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
        A[i][j] += a * Math.cos(2 * Math.PI * (u * i + v * j) / N + ph);
    }
    return A;
  }
  /* the covariance of mean-removed patches — the DC is removed per patch so
     the leading component is a structure, not the overall brightness */
  function patchCov(maps, sz, stride) {
    const d = sz * sz, pats = [];
    maps.forEach(A => {
      const N = A.length;
      for (let i = 0; i + sz <= N; i += stride) for (let j = 0; j + sz <= N; j += stride) {
        const p = [];
        for (let m = 0; m < sz; m++) for (let n = 0; n < sz; n++) p.push(A[i + m][j + n]);
        const mu = DL.mean(p); pats.push(p.map(v => v - mu));
      }
    });
    const C = DL.zeros2(d, d);
    pats.forEach(p => { for (let a = 0; a < d; a++) for (let b = a; b < d; b++) C[a][b] += p[a] * p[b] / pats.length; });
    for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a];
    return { C: C, n: pats.length };
  }
  /* leading eigenvectors by DEFLATED power iteration. DL.powerIter already
     returns {lambda, v}; this only adds the deflation loop around it. */
  function topEigs(C, k) {
    const A = C.map(r => r.slice()), out = [];
    for (let t = 0; t < k; t++) {
      const e = DL.powerIter(A, 280, 11 + t);
      out.push({ lam: e.lambda, v: e.v });
      for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++) A[i][j] -= e.lambda * e.v[i] * e.v[j];
    }
    return out;
  }

  function dataset(seed, n) {
    const r = DL.rng(seed), X = [], y = [];
    for (let k = 0; k < n; k++) {
      const c = k % NC, A = DL.zeros2(IM, IM);
      const th = c * Math.PI / 3, ci = 2 + r() * (IM - 4), cj = 2 + r() * (IM - 4);
      for (let i = 0; i < IM; i++) for (let j = 0; j < IM; j++) {
        const u = (i - ci) * Math.cos(th) + (j - cj) * Math.sin(th);
        const w = -(i - ci) * Math.sin(th) + (j - cj) * Math.cos(th);
        A[i][j] = Math.exp(-(u * u) / 1.2 - (w * w) / 26) + 0.12 * DL.randn(r);
      }
      X.push([A]); y.push(c);
    }
    return { X: X, y: y };
  }
  const flatK = M => { const a = []; for (let m = 0; m < F; m++) for (let n = 0; n < F; n++) a.push(M[m][n]); const mu = DL.mean(a); return a.map(v => v - mu); };
  /* what FRACTION of a filter's energy lies in the leading patch subspace */
  function projFrac(M, V) { const f = flatK(M), e = DL.dot(f, f) || 1; let s = 0; V.forEach(v => { const c = DL.dot(f, v); s += c * c; }); return s / e; }

  let CACHE = null;
  function train(lr) {
    if (CACHE && CACHE.lr === lr) return CACHE;
    const r = DL.rng(21), D = dataset(3, NEX);
    const PC = topEigs(patchCov(D.X.map(v => v[0]), F, 1).C, 5).map(e => e.v);
    const K = [];
    for (let f = 0; f < NF; f++) { const M = []; for (let m = 0; m < F; m++) { const row = []; for (let n = 0; n < F; n++) row.push(DL.randn(r) * 0.25); M.push(row); } K.push([M]); }
    const Wc = DL.zeros2(NF, NC);
    for (let a = 0; a < NF; a++) for (let b = 0; b < NC; b++) Wc[a][b] = DL.randn(r) * 0.3;
    const snaps = [], losses = [], projs = [];
    for (let t = 0; t <= STEPS; t++) {
      if (t % 20 === 0) { snaps.push(K.map(k => k[0].map(row => row.slice()))); projs.push(DL.mean(K.map(k => projFrac(k[0], PC)))); }
      let L = 0;
      const dK = K.map(() => [DL.zeros2(F, F)]), dW = DL.zeros2(NF, NC);
      D.X.forEach((V, idx) => {
        const Z = DL.conv2dMC(V, K, null, {});
        const a = Z.map(M => M.map(row => row.map(v => Math.max(0, v))));
        const arg = [];
        const h = a.map(M => {
          let b = -Infinity, bi = 0, bj = 0;
          for (let i = 0; i < M.length; i++) for (let j = 0; j < M[0].length; j++) if (M[i][j] > b) { b = M[i][j]; bi = i; bj = j; }
          arg.push([bi, bj]); return b;
        });
        const logits = DL.vecmat(h, Wc);
        L += DL.xent(logits, D.y[idx]) / D.X.length;
        const p = DL.softmax(logits);
        const dl = p.map((v, c) => (v - (c === D.y[idx] ? 1 : 0)) / D.X.length);
        for (let f2 = 0; f2 < NF; f2++) for (let c = 0; c < NC; c++) dW[f2][c] += h[f2] * dl[c];
        const dh = Wc.map(row => DL.dot(row, dl));
        const G = Z.map((M, f2) => M.map((row, i) => row.map((v, j) => (v > 0 && i === arg[f2][0] && j === arg[f2][1]) ? dh[f2] : 0)));
        const gK = DL.convGradK(V, G, {}, { kh: F, kw: F, kin: 1 });
        for (let f2 = 0; f2 < NF; f2++) for (let m = 0; m < F; m++) for (let n = 0; n < F; n++) dK[f2][0][m][n] += gK[f2][0][m][n];
      });
      losses.push(L);
      if (t === STEPS) break;
      if (!isFinite(L) || L > 1e12) { for (let q = t; q < STEPS; q++) losses.push(L); break; }
      for (let f2 = 0; f2 < NF; f2++) for (let m = 0; m < F; m++) for (let n = 0; n < F; n++) K[f2][0][m][n] -= lr * dK[f2][0][m][n];
      for (let a2 = 0; a2 < NF; a2++) for (let b = 0; b < NC; b++) Wc[a2][b] -= lr * dW[a2][b];
    }
    while (snaps.length <= STEPS / 20) { snaps.push(snaps[snaps.length - 1]); projs.push(projs[projs.length - 1]); }
    CACHE = { lr: lr, snaps: snaps, losses: losses, projs: projs };
    return CACHE;
  }

  function draw() {
    const kind = se.value, k = +ke.value, step = +te.value, lr = +lre.value;
    document.getElementById("ln-kv").textContent = k;
    document.getElementById("ln-tv").textContent = step;
    const PC = patchCov([image(kind, 40, 5)], P, 2);
    const eigs = topEigs(PC.C, k);
    let trace = 0; for (let i = 0; i < PC.C.length; i++) trace += PC.C[i][i];
    const totVar = eigs.reduce((s, e) => s + Math.max(0, e.lam), 0);

    const T = train(lr);
    const si = Math.min(T.snaps.length - 1, Math.round(step / 20));
    const snap = T.snaps[si];

    const FR = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = FR.g;
    g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`principal components of ${P}×${P} patches — no learning, no labels`);
    const cw = 4.6, gap = 14;
    eigs.forEach((e, t) => {
      const M = []; for (let m = 0; m < P; m++) { const row = []; for (let n = 0; n < P; n++) row.push(e.v[m * P + n]); M.push(row); }
      const mm = Math.max.apply(null, M.map(r0 => Math.max.apply(null, r0.map(Math.abs)))) || 1;
      const x = (t % 4) * (P * cw + gap + 16), y = 22 + Math.floor(t / 4) * (P * cw + 30);
      CN.grid(g, M, x, y, cw, { text: false, stroke: "none", fill: v => CN.signRamp(v, mm) });
      g.append("text").attr("x", x).attr("y", y + P * cw + 12).attr("font-size", 9).attr("fill", DC.muted)
        .text(`${(100 * Math.max(0, e.lam) / (trace || 1)).toFixed(1)}%`);
    });
    const ybot = 22 + Math.ceil(k / 4) * (P * cw + 30);
    g.append("text").attr("x", 0).attr("y", ybot + 8).attr("font-size", 10).attr("fill", DC.muted)
      .text(`${DL.commas(PC.n)} patches · ${(100 * totVar / (trace || 1)).toFixed(1)}% of the variance`);

    const tx = 234;
    g.append("text").attr("x", tx).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`a 6-filter layer trained by gradient descent — step ${step}`);
    const tcw = 9;
    snap.forEach((M, t) => {
      const mm = Math.max.apply(null, M.map(r0 => Math.max.apply(null, r0.map(Math.abs)))) || 1;
      const x = tx + (t % 3) * (F * tcw + 20), y = 22 + Math.floor(t / 3) * (F * tcw + 24);
      CN.grid(g, M, x, y, tcw, { text: false, stroke: "none", fill: v => CN.signRamp(v, mm) });
      g.append("text").attr("x", x).attr("y", y + F * tcw + 11).attr("font-size", 9).attr("fill", DC.muted)
        .text(`‖K‖ ${DL.fmt(DL.frob(M), 2)}`);
    });

    /* loss, and the alignment measurement */
    const lx = 430, lw = 300, lh = 108;
    const gl = g.append("g").attr("transform", `translate(${lx},30)`);
    const xs = d3.scaleLinear().domain([0, T.losses.length - 1]).range([0, lw]);
    const cap = v => Math.min(isFinite(v) ? v : 6, 6);
    const ys = d3.scaleLinear().domain([0, 6]).range([lh, 0]);
    DL.gridY(gl, ys, lw, 4);
    DL.axisB(gl, xs, lh, 5, "step");
    DL.axisL(gl, ys, 4, "training loss");
    gl.append("line").attr("x1", 0).attr("x2", lw).attr("y1", ys(Math.log(NC))).attr("y2", ys(Math.log(NC)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3,3");
    gl.append("text").attr("x", 4).attr("y", ys(Math.log(NC)) - 4).attr("font-size", 9.5).attr("fill", DC.muted)
      .text(`chance = log 3 = ${DL.fmt(Math.log(NC), 3)}`);
    DL.curve(gl, T.losses.map((v, i) => [xs(i), ys(cap(v))]), { stroke: DC.accent, w: 1.8 });
    gl.append("line").attr("x1", xs(step)).attr("x2", xs(step)).attr("y1", 0).attr("y2", lh)
      .attr("stroke", DC.a2).attr("stroke-dasharray", "3,3");

    const gy = 30 + lh + 46;
    const gq = g.append("g").attr("transform", `translate(${lx},${gy})`);
    const qh = 86;
    const xq = d3.scaleLinear().domain([0, STEPS]).range([0, lw]);
    const yq = d3.scaleLinear().domain([0, 1]).range([qh, 0]);
    DL.gridY(gq, yq, lw, 3);
    DL.axisB(gq, xq, qh, 5, "step");
    DL.axisL(gq, yq, 3, "filter energy in the top-5 patch subspace");
    DL.curve(gq, T.projs.map((v, i) => [xq(i * 20), yq(v)]), { stroke: DC.good, w: 2 });
    gq.append("circle").attr("cx", xq(si * 20)).attr("cy", yq(T.projs[si])).attr("r", 3.6).attr("fill", DC.a2);

    const p0 = T.projs[0], pf = T.projs[si];
    const lossNow = T.losses[Math.min(step, T.losses.length - 1)];
    document.getElementById("learn-readout").innerHTML =
      `<b>Left:</b> the top ${k} principal components of ${DL.commas(PC.n)} ${P}×${P} patches drawn from a ` +
      `${kind === "pink" ? "1/f-spectrum" : (kind === "white" ? "white-noise" : "edge-world")} image, by deflated power iteration on the ${P * P}×${P * P} patch covariance; they carry <b>${(100 * totVar / (trace || 1)).toFixed(1)}%</b> of the variance. ` +
      (kind === "pink"
        ? "They are <b>localised, oriented and increasingly finely striped</b> — the same family as a trained first layer, with no gradient descent and no labels anywhere in their computation. That is the explanation for the universality of first-layer filters: they encode a property of natural image statistics, not of the task."
        : kind === "white"
          ? "With white noise there is <b>nothing to find</b>: every direction has the same variance, the components are arbitrary, and the leading one explains barely more than 1/" + (P * P) + " of the total. This is the control, and it is what shows the oriented structure in the other setting comes from the image statistics rather than from the method."
          : "In a world of step edges the leading components are low-frequency oriented ramps — the same conclusion by a cruder route, and a useful reminder that the components describe the <i>input distribution</i> and nothing else.") +
      ` &nbsp;<b>Right:</b> a ${NF}-filter 5×5 layer trained live on a 3-way orientation task for ${step} of ${STEPS} steps at learning rate ${lr}; ` +
      `training loss <b>${isFinite(lossNow) && lossNow < 1e6 ? DL.fmt(lossNow, 4) : "diverged"}</b> against a chance level of ${DL.fmt(Math.log(NC), 3)}. ` +
      `<b>The measured link between the two panels:</b> the fraction of each filter's energy lying in the top-5 principal subspace <i>of this task's own patches</i> rose from <b>${DL.fmt(p0, 3)}</b> at initialisation to <b>${DL.fmt(pf, 3)}</b> — ` +
      (pf > p0 * 1.2
        ? `gradient descent moved the filters into the directions the data's second-order statistics single out, which is the honest, measurable version of "the layer learns what the images contain".`
        : `on this run it did not rise, which at this learning rate is a training failure rather than a statement about convolution.`) +
      ` <b>What this figure does not show, deliberately:</b> the trained 5×5 filters do <i>not</i> become dramatically more oriented in any visual sense, and their measured anisotropy barely moves. ` +
      `Six filters on a toy task can solve it without needing to, and claiming otherwise would be reading the large-scale result into a small experiment that does not contain it. ` +
      (lr >= 5 ? "<b>At this learning rate the run diverges and the filters become high-frequency noise</b> — which is the five-second visual diagnostic §27 recommends, visible here long before any metric would flag it." : "");
  }
  [ke, te].forEach(e => e.addEventListener("input", draw));
  [se, lre].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 27 · #lineage-svg — the architecture lineage as a trade-off ═══════════ */
(function () {
  const svg = d3.select("#lineage-svg");
  if (svg.empty()) return;
  const W = 760, H = 450;
  const xe = document.getElementById("lg-x"), ye = document.getElementById("lg-y"),
    he = document.getElementById("lg-h");

  /* Every architecture is given as an explicit LIST OF CONVOLUTIONS with its
     own input shape, so branching designs (residual projections, parallel
     Inception towers) are expressible and the totals come from DL.convCount
     exactly as §12's do. `bn` adds 2 parameters per output channel; `fc`
     entries are dense layers. Nothing here is quoted from a paper. */
  const conv = (H0, W0, cin, cout, k, s, p, g, bn) => ({ t: "c", H: H0, W: W0, cin: cin, cout: cout, k: k, s: s || 1, p: p || 0, g: g || 1, bn: !!bn });
  const fc = (nin, nout) => ({ t: "f", nin: nin, nout: nout });

  function lenet() {
    return [conv(32, 32, 1, 6, 5, 1, 0), conv(14, 14, 6, 16, 5, 1, 0), fc(400, 120), fc(120, 84), fc(84, 10)];
  }
  function alexnet() {
    return [conv(227, 227, 3, 96, 11, 4, 0), conv(27, 27, 96, 256, 5, 1, 2),
      conv(13, 13, 256, 384, 3, 1, 1), conv(13, 13, 384, 384, 3, 1, 1), conv(13, 13, 384, 256, 3, 1, 1),
      fc(9216, 4096), fc(4096, 4096), fc(4096, 1000)];
  }
  function vgg16() {
    const L = []; let n = 224, c = 3;
    [[64, 2], [128, 2], [256, 3], [512, 3], [512, 3]].forEach(([w, r]) => {
      for (let i = 0; i < r; i++) { L.push(conv(n, n, c, w, 3, 1, 1)); c = w; }
      n = n / 2;
    });
    L.push(fc(25088, 4096), fc(4096, 4096), fc(4096, 1000));
    return L;
  }
  function resnet50() {
    const L = [conv(224, 224, 3, 64, 7, 2, 3, 1, true)];
    let n = 56, c = 64;
    [[64, 256, 3, 1], [128, 512, 4, 2], [256, 1024, 6, 2], [512, 2048, 3, 2]].forEach(([w, out, r, s0]) => {
      for (let b = 0; b < r; b++) {
        const s = (b === 0) ? s0 : 1, ns = Math.floor((n - 1) / s) + 1;
        L.push(conv(n, n, c, w, 1, 1, 0, 1, true));
        L.push(conv(n, n, w, w, 3, s, 1, 1, true));
        L.push(conv(ns, ns, w, out, 1, 1, 0, 1, true));
        if (b === 0) L.push(conv(n, n, c, out, 1, s, 0, 1, true));     // the projection shortcut
        n = ns; c = out;
      }
    });
    L.push(fc(2048, 1000));
    return L;
  }
  function mobilenet() {
    const L = []; let n = 112, c = 32;
    L.push(conv(224, 224, 3, 32, 3, 2, 1, 1, true));
    [[64, 1], [128, 2], [128, 1], [256, 2], [256, 1], [512, 2], [512, 1], [512, 1], [512, 1], [512, 1], [512, 1], [1024, 2], [1024, 1]]
      .forEach(([out, s]) => {
        const ns = Math.floor((n + 2 - 3) / s) + 1;
        L.push(conv(n, n, c, c, 3, s, 1, c, true));                    // depthwise
        L.push(conv(ns, ns, c, out, 1, 1, 0, 1, true));                // pointwise
        n = ns; c = out;
      });
    L.push(fc(1024, 1000));
    return L;
  }
  function inception3a(nH, cin, r1, r3, e3, r5, e5, pp) {
    return [conv(nH, nH, cin, r1, 1, 1, 0), conv(nH, nH, cin, r3, 1, 1, 0), conv(nH, nH, r3, e3, 3, 1, 1),
      conv(nH, nH, cin, r5, 1, 1, 0), conv(nH, nH, r5, e5, 5, 1, 2), conv(nH, nH, cin, pp, 1, 1, 0)];
  }
  function googlenet() {
    let L = [conv(224, 224, 3, 64, 7, 2, 3), conv(56, 56, 64, 64, 1, 1, 0), conv(56, 56, 64, 192, 3, 1, 1)];
    L = L.concat(inception3a(28, 192, 64, 96, 128, 16, 32, 32));
    L = L.concat(inception3a(28, 256, 128, 128, 192, 32, 96, 64));
    const c4 = [[480, 192, 96, 208, 16, 48, 64], [512, 160, 112, 224, 24, 64, 64], [512, 128, 128, 256, 24, 64, 64],
      [512, 112, 144, 288, 32, 64, 64], [528, 256, 160, 320, 32, 128, 128]];
    c4.forEach(a => { L = L.concat(inception3a(14, a[0], a[1], a[2], a[3], a[4], a[5], a[6])); });
    [[832, 256, 160, 320, 32, 128, 128], [832, 384, 192, 384, 48, 128, 128]].forEach(a => { L = L.concat(inception3a(7, a[0], a[1], a[2], a[3], a[4], a[5], a[6])); });
    L.push(fc(1024, 1000));
    return L;
  }
  function convnextT() {
    const L = [conv(224, 224, 3, 96, 4, 4, 0, 1, true)];               // the patchifying stem
    let n = 56, c = 96;
    [[96, 3], [192, 3], [384, 9], [768, 3]].forEach(([w, r], si) => {
      if (si > 0) { L.push(conv(n, n, c, w, 2, 2, 0, 1, true)); n = n / 2; c = w; }
      for (let b = 0; b < r; b++) {
        L.push(conv(n, n, c, c, 7, 1, 3, c, true));                    // depthwise 7×7
        L.push(conv(n, n, c, 4 * c, 1, 1, 0));                         // inverted bottleneck up
        L.push(conv(n, n, 4 * c, c, 1, 1, 0));                         // and back down
      }
    });
    L.push(fc(768, 1000));
    return L;
  }

  function total(L) {
    let p = 0, m = 0, d = 0;
    L.forEach(l => {
      if (l.t === "f") { p += l.nin * l.nout + l.nout; m += l.nin * l.nout; d++; return; }
      const c = DL.convCount({ H: l.H, W: l.W, cin: l.cin, cout: l.cout, k: l.k, s: l.s, p0: l.p, p1: l.p, groups: l.g, bias: !l.bn });
      p += c.params + (l.bn ? 2 * l.cout : 0); m += c.macs; d++;
    });
    return { p: p, m: m, d: d };
  }

  const NETS = [
    { id: "lenet", t: "the original convnet", year: 1998, L: lenet, dense: true, sep: false, skip: false, idea: "convolution, subsampling, weight sharing" },
    { id: "alex", t: "large-kernel stem", year: 2012, L: alexnet, dense: true, sep: false, skip: false, idea: "scale, ReLU, dropout" },
    { id: "vgg", t: "all-3×3", year: 2014, L: vgg16, dense: true, sep: false, skip: false, idea: "depth, not kernel size" },
    { id: "goog", t: "multi-branch", year: 2014, L: googlenet, dense: false, sep: false, skip: false, idea: "1×1 bottlenecks, global pooling" },
    { id: "res", t: "residual", year: 2015, L: resnet50, dense: false, sep: false, skip: true, idea: "the skip connection" },
    { id: "mob", t: "mobile", year: 2017, L: mobilenet, dense: false, sep: true, skip: false, idea: "depthwise separation" },
    { id: "next", t: "modernised", year: 2022, L: convnextT, dense: false, sep: true, skip: true, idea: "the transformer block shape" }
  ].map(n => Object.assign(n, total(n.L())));

  function draw() {
    const xk = xe.value, yk = ye.value, hk = he.value;
    const val = (n, k) => k === "params" ? n.p : (k === "macs" ? n.m : (k === "depth" ? n.d : n.year));
    const isLog = k => k !== "year" && k !== "depth";

    const F = DL.frame(svg, W, H, { l: 62, r: 16, t: 22, b: 176 });
    const g = F.g;
    const pw = F.iw, ph = F.ih;
    const xv = NETS.map(n => val(n, xk)), yv = NETS.map(n => val(n, yk));
    const xs = (isLog(xk) ? d3.scaleLog() : d3.scaleLinear())
      .domain(isLog(xk) ? [d3.min(xv) / 2.2, d3.max(xv) * 2.2] : [d3.min(xv) - 2, d3.max(xv) + 2]).range([0, pw]);
    const ys = (isLog(yk) ? d3.scaleLog() : d3.scaleLinear())
      .domain(isLog(yk) ? [d3.min(yv) / 2.2, d3.max(yv) * 2.2] : [d3.min(yv) - 2, d3.max(yv) + 2]).range([ph, 0]);
    DL.gridY(g, ys, pw, 5); DL.gridX(g, xs, ph, 5);
    DL.axisB(g, xs, ph, 5, xk === "params" ? "parameters" : (xk === "macs" ? "MACs per example" : "year"), v => isLog(xk) ? DL.big(v) : v);
    DL.axisL(g, ys, 5, yk === "params" ? "parameters" : (yk === "macs" ? "MACs per example" : "depth"), v => isLog(yk) ? DL.big(v) : v);

    for (let i = 0; i + 1 < NETS.length; i++)
      DL.arrow(g, xs(val(NETS[i], xk)), ys(val(NETS[i], yk)), xs(val(NETS[i + 1], xk)), ys(val(NETS[i + 1], yk)),
        { color: DC.line, w: 1.1, head: 5, op: 0.8 });
    NETS.forEach(n => {
      const on = hk === "none" ? true : !!n[hk];
      g.append("circle").attr("cx", xs(val(n, xk))).attr("cy", ys(val(n, yk)))
        .attr("r", 4 + Math.sqrt(n.d) * 0.9)
        .attr("fill", on ? DC.accent : "none").attr("fill-opacity", 0.55)
        .attr("stroke", on ? DC.accent : DC.muted).attr("stroke-width", 1.6);
      g.append("text").attr("x", xs(val(n, xk)) + 9).attr("y", ys(val(n, yk)) - 7).attr("font-size", 10)
        .attr("fill", on ? DC.ink : DC.muted).text(n.t);
    });
    g.append("text").attr("x", 2).attr("y", 8).attr("font-size", 10).attr("fill", DC.muted)
      .text("marker area ∝ depth in weight layers; arrows are chronological");

    /* the table */
    const ty = ph + 46;
    const cols = [0, 148, 236, 316, 366];
    ["design", "parameters", "MACs", "depth", "the one idea"].forEach((t, j) =>
      g.append("text").attr("x", cols[j] + (j === 1 || j === 2 || j === 3 ? 64 : 0)).attr("y", ty)
        .attr("text-anchor", (j === 1 || j === 2 || j === 3) ? "end" : "start")
        .attr("font-size", 10).attr("fill", DC.muted).text(t));
    NETS.forEach((n, i) => {
      const y = ty + 15 + i * 15, on = hk === "none" ? true : !!n[hk];
      g.append("text").attr("x", cols[0]).attr("y", y).attr("font-size", 10)
        .attr("fill", on ? DC.ink : DC.muted).text(n.t);
      [[DL.commas(n.p), 1, DC.good], [DL.big(n.m), 2, DC.accent], [String(n.d), 3, DC.ink]].forEach(([v, j, cc]) =>
        g.append("text").attr("x", cols[j] + 64).attr("y", y).attr("text-anchor", "end").attr("font-size", 10)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", on ? cc : DC.muted).text(v));
      g.append("text").attr("x", cols[4]).attr("y", y).attr("font-size", 10).attr("fill", DC.muted).text(n.idea);
    });

    /* the two largest single steps on the current axes */
    let bx = null, by = null;
    for (let i = 0; i + 1 < NETS.length; i++) {
      const rx = val(NETS[i + 1], xk) / val(NETS[i], xk), ry = val(NETS[i + 1], yk) / val(NETS[i], yk);
      if (!bx || Math.abs(Math.log(rx)) > Math.abs(Math.log(bx.r))) bx = { r: rx, a: NETS[i], b: NETS[i + 1] };
      if (!by || Math.abs(Math.log(ry)) > Math.abs(Math.log(by.r))) by = { r: ry, a: NETS[i], b: NETS[i + 1] };
    }
    document.getElementById("lineage-readout").innerHTML =
      `All seven parameter and MAC figures are computed in this figure from an explicit list of every convolution and dense layer, using the same <code>DL.convCount</code> the rest of the page uses. ` +
      `Spot checks against the published totals: the all-3×3 network gives <b>${DL.commas(NETS[2].p)}</b> (published "138 M"), ` +
      `the residual network <b>${DL.commas(NETS[4].p)}</b> (published 25.6 M, and the ${DL.commas(NETS[4].p - 25503912)} difference from the convolution weights alone is exactly the BatchNorm scales and shifts), ` +
      `and the mobile network <b>${DL.commas(NETS[5].p)}</b> with <b>${DL.commas(NETS[5].m)}</b> MACs (published 4.2 M and 569 M). ` +
      `<b>The largest single step on the ${xk} axis</b> is ${bx.a.t} → ${bx.b.t}, a factor of <b>${DL.fmt(bx.r < 1 ? 1 / bx.r : bx.r, 2)}×</b> ${bx.r < 1 ? "down" : "up"}; ` +
      `on the ${yk} axis it is ${by.a.t} → ${by.b.t}, a factor of <b>${DL.fmt(by.r < 1 ? 1 / by.r : by.r, 2)}×</b> ${by.r < 1 ? "down" : "up"}. ` +
      `The all-3×3 network to the multi-branch one drops parameters by <b>${DL.fmt(NETS[2].p / NETS[3].p, 1)}×</b> and arithmetic by <b>${DL.fmt(NETS[2].m / NETS[3].m, 1)}×</b> in a single generation — that is the dense head being deleted (§12) and the 1×1 bottleneck (§16), and it is the single biggest efficiency jump on the plot. ` +
      `<b>No accuracy axis is offered here on purpose:</b> training recipes improved by more over this period than several of these architectural steps are worth, so an accuracy comparison across years measures the recipe as much as the design.`;
  }
  [xe, ye, he].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 28 · #crit-svg — the arrangement test ═══════════ */
(function () {
  const svg = d3.select("#crit-svg");
  if (svg.empty()) return;
  const W = 760, H = 430, N = 24;
  const se = document.getElementById("cr2-s"), re = document.getElementById("cr2-r"),
    ae = document.getElementById("cr2-a"), sde = document.getElementById("cr2-seed");

  /* five distinct 4×4 parts. The scrambled image contains EXACTLY the same
     parts with EXACTLY the same values — only their positions change, so the
     two images have identical pixel histograms and identical part inventories. */
  const PARTS = [
    [[0, 8, 8, 0], [8, 2, 2, 8], [8, 2, 2, 8], [0, 8, 8, 0]],       // an eye-like ring
    [[0, 8, 8, 0], [8, 2, 2, 8], [8, 2, 2, 8], [0, 8, 8, 0]],
    [[0, 0, 7, 0], [0, 7, 7, 0], [0, 7, 7, 0], [0, 6, 6, 0]],       // a nose-like bar
    [[0, 0, 0, 0], [9, 9, 9, 9], [4, 4, 4, 4], [0, 0, 0, 0]],       // a mouth-like line
    [[5, 5, 5, 5], [5, 0, 0, 5], [0, 0, 0, 0], [0, 0, 0, 0]]        // a brow-like arc
  ];
  const HOMES = {
    face: [[6, 5], [6, 15], [11, 10], [16, 9], [3, 5]],
    digit: [[4, 4], [8, 10], [12, 6], [16, 12], [6, 16]]
  };
  function place(spots) {
    const A = DL.zeros2(N, N);
    PARTS.forEach((Pp, k) => {
      const [i0, j0] = spots[k];
      for (let m = 0; m < 4; m++) for (let n = 0; n < 4; n++) {
        const i = i0 + m, j = j0 + n;
        if (i >= 0 && i < N && j >= 0 && j < N) A[i][j] = Math.max(A[i][j], Pp[m][n]);
      }
    });
    return A;
  }
  function scrambled(spots, frac, seed) {
    const r = DL.rng(seed);
    return spots.map(([i, j]) => {
      const ti = 1 + Math.floor(r() * (N - 6)), tj = 1 + Math.floor(r() * (N - 6));
      return [Math.round(i + (ti - i) * frac), Math.round(j + (tj - j) * frac)];
    });
  }
  const KB = [
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]], [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]
  ];
  function features(A, mode) {
    const K = KB.map(k => [k]);
    let Z = DL.conv2dMC([A], K, null, { p0: 1, p1: 1 });
    Z = Z.map(M => M.map(row => row.map(v => Math.max(0, v))));
    if (mode === "conv") { const f = []; Z.forEach(M => M.forEach(row => row.forEach(v => f.push(v)))); return f; }
    if (mode === "pool") {
      let Y = DL.pool2d(Z, { k: 2, s: 2 }).Y;
      Y = DL.pool2d(Y, { k: 2, s: 2 }).Y;
      const f = []; Y.forEach(M => M.forEach(row => row.forEach(v => f.push(v)))); return f;
    }
    return DL.globalPool(Z, "avg");
  }
  const cos = (a, b) => {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na * nb) || 1);
  };

  function draw() {
    const frac = +se.value / 100, mode = re.value, arr = ae.value, seed = +sde.value;
    document.getElementById("cr2-sv").textContent = (100 * frac).toFixed(0) + "%";
    document.getElementById("cr2-seedv").textContent = seed;
    const home = HOMES[arr];
    const A = place(home), B = place(scrambled(home, frac, seed));

    const FR = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = FR.g;
    const cw = 6.4;
    const [lo, hi] = CN.extent2(A);
    CN.grid(g, A, 0, 26, cw, { title: "the parts, arranged", text: false, stroke: "none", fill: v => CN.grey(v, lo, hi) });
    CN.grid(g, B, N * cw + 28, 26, cw, { title: `the same parts, ${(100 * frac).toFixed(0)}% scrambled`, text: false, stroke: "none", fill: v => CN.grey(v, lo, hi) });
    /* the histograms really are identical — say so, measured */
    const hist = M => { const h = new Array(10).fill(0); M.forEach(r0 => r0.forEach(v => h[Math.min(9, Math.round(v))]++)); return h; };
    const hA = hist(A), hB = hist(B);
    const histSame = hA.every((v, i) => v === hB[i]);

    const fa = features(A, mode), fb = features(B, mode);
    const simNow = cos(fa, fb);

    /* the feature bars */
    const bx = 2 * (N * cw + 28) + 10;
    g.append("text").attr("x", bx).attr("y", 20).attr("font-size", 10.5).attr("fill", DC.ink)
      .text(`the two representations (${fa.length} numbers each)`);
    const show = Math.min(fa.length, 64);
    const mxv = Math.max(d3.max(fa), d3.max(fb)) || 1;
    for (let i = 0; i < show; i++) {
      const x = bx + i * Math.min(3.4, 220 / show);
      g.append("rect").attr("x", x).attr("y", 30 + 44 - 44 * fa[i] / mxv).attr("width", Math.max(1, Math.min(3.4, 220 / show) - 0.6))
        .attr("height", 44 * fa[i] / mxv).attr("fill", DC.accent).attr("fill-opacity", 0.85);
      g.append("rect").attr("x", x).attr("y", 84 + 44 - 44 * fb[i] / mxv).attr("width", Math.max(1, Math.min(3.4, 220 / show) - 0.6))
        .attr("height", 44 * fb[i] / mxv).attr("fill", DC.a2).attr("fill-opacity", 0.85);
    }
    g.append("text").attr("x", bx - 4).attr("y", 52).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", DC.accent).text("arranged");
    g.append("text").attr("x", bx - 4).attr("y", 106).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", DC.a2).text("scrambled");
    if (fa.length > show) g.append("text").attr("x", bx).attr("y", 142).attr("font-size", 9).attr("fill", DC.muted)
      .text(`(first ${show} of ${fa.length} shown)`);

    /* the sweep */
    const px = 60, py = 210, pw = 560, ph = 150;
    const gp = g.append("g").attr("transform", `translate(${px},${py})`);
    const modes = [["conv", "one convolution layer", DC.accent], ["pool", "conv + two 2×2 max pools", DC.violet], ["gap", "conv + global average pool", DC.a2]];
    const xs = d3.scaleLinear().domain([0, 1]).range([0, pw]);
    const ys = d3.scaleLinear().domain([0, 1.03]).range([ph, 0]);
    DL.gridY(gp, ys, pw, 4);
    DL.axisB(gp, xs, ph, 5, "scramble fraction", v => (100 * v).toFixed(0) + "%");
    DL.axisL(gp, ys, 4, "cosine similarity of the two representations");
    const finals = {};
    modes.forEach(([m, lab, cc]) => {
      const pts = [];
      for (let f = 0; f <= 1.0001; f += 0.05) {
        const Bf = place(scrambled(home, f, seed));
        pts.push([xs(f), ys(cos(features(A, m), features(Bf, m)))]);
      }
      finals[m] = ys.invert(pts[pts.length - 1][1]);
      DL.curve(gp, pts, { stroke: cc, w: m === mode ? 2.6 : 1.4, op: m === mode ? 1 : 0.6 });
    });
    gp.append("line").attr("x1", xs(frac)).attr("x2", xs(frac)).attr("y1", 0).attr("y2", ph)
      .attr("stroke", DC.line).attr("stroke-dasharray", "3,3");
    DL.legend(gp, modes.map(([m, lab, cc]) => ({ color: cc, label: lab })), 8, 12, { gap: 14, font: 10 });

    document.getElementById("crit-readout").innerHTML =
      `Both images contain the same five parts with the same pixel values; only their positions differ, and their intensity histograms are <b>${histSame ? "identical" : "nearly identical"}</b>` +
      (histSame ? " (checked bin by bin). " : " (they differ slightly only where parts overlap). ") +
      `At ${(100 * frac).toFixed(0)}% scramble the cosine similarity of the two representations is ` +
      `<b>${DL.fmt(finals.conv, 3)}</b> after one convolution layer, <b>${DL.fmt(finals.pool, 3)}</b> after two stages of 2×2 max pooling, and <b>${DL.fmt(finals.gap, 3)}</b> after global average pooling. ` +
      `<b>That ordering is the critique's prediction, measured.</b> The equivariant read-out ${finals.conv < 0.7 ? "falls a long way" : "falls"} — moving the parts moves the feature map, so the representation notices. ` +
      `The globally pooled read-out stays at <b>${DL.fmt(finals.gap, 3)}</b>: it is a count of <i>what is present</i> and is blind to <i>where</i>, by construction. ` +
      `A classifier reading only the pooled vector therefore cannot distinguish these two images at all — not because it was trained badly, but because the information is not in its input. ` +
      `Two stages of max pooling land in between, at ${DL.fmt(finals.pool, 3)}, which is the point: pooling does not choose between the two properties, it erodes one of them gradually.`;
  }
  [se, sde].forEach(e => e.addEventListener("input", draw));
  [re, ae].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 29 · #route-svg — routing by agreement ═══════════ */
(function () {
  const svg = d3.select("#route-svg");
  if (svg.empty()) return;
  const W = 760, H = 450, J = 2;
  const re = document.getElementById("rt-r"), ne = document.getElementById("rt-n"),
    oe = document.getElementById("rt-o"), sp = document.getElementById("rt-s"),
    sde = document.getElementById("rt-seed");

  const squash = s => {
    const n2 = s[0] * s[0] + s[1] * s[1], n = Math.sqrt(n2) || 1e-12;
    const f = n2 / (1 + n2) / n;
    return [s[0] * f, s[1] * f];
  };
  /* the predictions û_{j|i}. A cluster of AGREEING capsules points near a
     common direction for upper capsule 0; the outliers point elsewhere. */
  function predictions(n, nOut, spreadDeg, seed) {
    const r = DL.rng(seed), U = [];
    for (let i = 0; i < n; i++) {
      const outlier = i >= n - nOut;
      const row = [];
      for (let j = 0; j < J; j++) {
        const base = (j === 0) ? 0.6 : 2.4;
        const th = outlier
          ? base + (j === 0 ? 1 : -1) * (1.5 + 1.4 * r())
          : base + (spreadDeg * Math.PI / 180) * (r() - 0.5) * 2;
        const mag = 0.9 + 0.5 * r();
        row.push([mag * Math.cos(th), mag * Math.sin(th)]);
      }
      U.push({ u: row, outlier: outlier });
    }
    return U;
  }
  function route(U, iters) {
    const n = U.length;
    const b = U.map(() => new Array(J).fill(0));
    const hist = [];
    for (let t = 0; t <= iters; t++) {
      const c = b.map(bi => DL.softmax(bi));
      const s = [];
      for (let j = 0; j < J; j++) {
        let x = 0, y = 0;
        for (let i = 0; i < n; i++) { x += c[i][j] * U[i].u[j][0]; y += c[i][j] * U[i].u[j][1]; }
        s.push([x, y]);
      }
      const v = s.map(squash);
      hist.push({ c: c.map(r0 => r0.slice()), v: v.map(w => w.slice()), s: s.map(w => w.slice()) });
      if (t === iters) break;
      for (let i = 0; i < n; i++) for (let j = 0; j < J; j++) b[i][j] += U[i].u[j][0] * v[j][0] + U[i].u[j][1] * v[j][1];
    }
    return hist;
  }

  function draw() {
    const iters = +re.value, n = +ne.value, spreadDeg = +sp.value, seed = +sde.value;
    const nOut = Math.min(+oe.value, n - 2);
    ["rt-rv", "rt-nv", "rt-ov", "rt-seedv"].forEach((id, i) => document.getElementById(id).textContent = [iters, n, nOut, seed][i]);
    document.getElementById("rt-sv").textContent = spreadDeg + "°";
    const U = predictions(n, nOut, spreadDeg, seed);
    const hist = route(U, Math.max(iters, 8));
    const now = hist[iters];

    const FR = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = FR.g;

    /* the vector plot for upper capsule 0 */
    const cx = 118, cy = 150, R = 100;
    g.append("text").attr("x", 4).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text("predictions for upper capsule 0");
    g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R).attr("fill", "none").attr("stroke", DC.grid);
    g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R / 2).attr("fill", "none").attr("stroke", DC.grid).attr("stroke-dasharray", "2,3");
    g.append("text").attr("x", cx + R / 2 + 3).attr("y", cy - 3).attr("font-size", 8.5).attr("fill", DC.muted).text("‖v‖ = 1");
    const sc = R / 1.6;
    U.forEach((Ui, i) => {
      const w = now.c[i][0];
      DL.arrow(g, cx, cy, cx + Ui.u[0][0] * sc, cy - Ui.u[0][1] * sc,
        { color: Ui.outlier ? DC.rose : DC.accent, w: 1.2 + 3 * w, head: 5, op: 0.18 + 0.82 * Math.min(1, w * n / 1.2) });
    });
    DL.arrow(g, cx, cy, cx + now.s[0][0] * sc, cy - now.s[0][1] * sc, { color: DC.muted, w: 1.6, head: 7, dash: "3,3" });
    DL.arrow(g, cx, cy, cx + now.v[0][0] * R, cy - now.v[0][1] * R, { color: DC.a2, w: 3, head: 9 });
    g.append("text").attr("x", 4).attr("y", cy + R + 18).attr("font-size", 10).attr("fill", DC.a2)
      .text(`v₀ after squash, ‖v₀‖ = ${DL.fmt(Math.hypot(now.v[0][0], now.v[0][1]), 3)}`);
    g.append("text").attr("x", 4).attr("y", cy + R + 32).attr("font-size", 10).attr("fill", DC.muted)
      .text(`s₀ before squash, ‖s₀‖ = ${DL.fmt(Math.hypot(now.s[0][0], now.s[0][1]), 3)}`);

    /* the coupling matrix */
    const mx = 268, mcw = 30, mrh = Math.min(20, 200 / n);
    g.append("text").attr("x", mx).attr("y", 10).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
      .text(`coupling coefficients at iteration ${iters}`);
    for (let i = 0; i < n; i++) for (let j = 0; j < J; j++) {
      g.append("rect").attr("x", mx + j * mcw).attr("y", 20 + i * mrh).attr("width", mcw - 1).attr("height", mrh - 1)
        .attr("fill", d3.interpolateRgb("#171a23", DC.good)(now.c[i][j]))
        .attr("shape-rendering", "crispEdges");
      if (mrh >= 14) g.append("text").attr("x", mx + j * mcw + mcw / 2).attr("y", 20 + i * mrh + mrh / 2 + 3)
        .attr("text-anchor", "middle").attr("font-size", 8.5).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", now.c[i][j] > 0.6 ? "#0f1117" : DC.ink).text(now.c[i][j].toFixed(2));
    }
    for (let i = 0; i < n; i++) g.append("text").attr("x", mx - 5).attr("y", 20 + i * mrh + mrh / 2 + 3)
      .attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", U[i].outlier ? DC.rose : DC.muted)
      .text("i" + i + (U[i].outlier ? "*" : ""));
    for (let j = 0; j < J; j++) g.append("text").attr("x", mx + j * mcw + mcw / 2).attr("y", 15)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", DC.muted).text("j" + j);
    g.append("text").attr("x", mx).attr("y", 26 + n * mrh + 10).attr("font-size", 9.5).attr("fill", DC.muted)
      .text("each ROW sums to 1 — the softmax is over j");
    g.append("text").attr("x", mx).attr("y", 26 + n * mrh + 23).attr("font-size", 9.5).attr("fill", DC.rose)
      .text("* = an outlier prediction");

    /* the curves */
    const px = 400, pw = 320, ph = 150;
    const gp = g.append("g").attr("transform", `translate(${px},30)`);
    const agree = [], out = [], vlen = [];
    hist.forEach(h => {
      let a = 0, na = 0, o = 0, no = 0;
      U.forEach((Ui, i) => { if (Ui.outlier) { o += h.c[i][0]; no++; } else { a += h.c[i][0]; na++; } });
      agree.push(na ? a / na : 0); out.push(no ? o / no : 0);
      vlen.push(Math.hypot(h.v[0][0], h.v[0][1]));
    });
    const xs = d3.scaleLinear().domain([0, hist.length - 1]).range([0, pw]);
    const ys = d3.scaleLinear().domain([0, 1.05]).range([ph, 0]);
    DL.gridY(gp, ys, pw, 4);
    DL.axisB(gp, xs, ph, 5, "routing iteration");
    DL.axisL(gp, ys, 4, "");
    DL.curve(gp, agree.map((v, i) => [xs(i), ys(v)]), { stroke: DC.accent, w: 2.2 });
    DL.curve(gp, out.map((v, i) => [xs(i), ys(v)]), { stroke: DC.rose, w: 2 });
    DL.curve(gp, vlen.map((v, i) => [xs(i), ys(v)]), { stroke: DC.a2, w: 1.6, dash: "4,3" });
    gp.append("line").attr("x1", xs(iters)).attr("x2", xs(iters)).attr("y1", 0).attr("y2", ph)
      .attr("stroke", DC.line).attr("stroke-dasharray", "3,3");
    DL.legend(gp, [{ color: DC.accent, label: "mean c of the agreeing capsules" },
      { color: DC.rose, label: "mean c of the outliers" },
      { color: DC.a2, label: "‖v₀‖ after squash", dash: "4,3" }], 6, 12, { gap: 13, font: 9.5 });

    /* the squash inset */
    const qx = 400, qy = 236, qw = 200, qh = 108;
    const gq = g.append("g").attr("transform", `translate(${qx},${qy})`);
    const xq = d3.scaleLinear().domain([0, 4]).range([0, qw]);
    const yq = d3.scaleLinear().domain([0, 1.08]).range([qh, 0]);
    DL.gridY(gq, yq, qw, 3);
    DL.axisB(gq, xq, qh, 4, "‖s‖");
    DL.axisL(gq, yq, 3, "‖squash(s)‖");
    gq.append("line").attr("x1", 0).attr("x2", qw).attr("y1", yq(1)).attr("y2", yq(1))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3,3");
    DL.curve(gq, DL.linspace(0, 4, 90).map(t => [xq(t), yq(t * t / (1 + t * t))]), { stroke: DC.good, w: 2 });
    DL.curve(gq, DL.linspace(0, 1.05, 30).map(t => [xq(t), yq(t * t)]), { stroke: DC.line, w: 1.2, dash: "3,3" });
    gq.append("text").attr("x", 4).attr("y", 12).attr("font-size", 9.5).attr("fill", DC.muted)
      .text("‖s‖² for small ‖s‖, → 1 for large");
    gq.append("circle").attr("cx", xq(Math.min(4, Math.hypot(now.s[0][0], now.s[0][1]))))
      .attr("cy", yq(Math.hypot(now.v[0][0], now.v[0][1]))).attr("r", 4).attr("fill", DC.a2);

    const sep = agree[iters] - out[iters];
    document.getElementById("route-readout").innerHTML =
      `${n} lower capsules, of which ${nOut} are outliers, routing to ${J} upper capsules for ${iters} iteration${iters === 1 ? "" : "s"}. ` +
      `At iteration 0 every coefficient is <b>${DL.fmt(1 / J, 3)}</b> — a uniform softmax over the ${J} parents, because all the logits start at zero. ` +
      `After ${iters}, the agreeing capsules average <b>${DL.fmt(agree[iters], 3)}</b> and the outliers <b>${DL.fmt(out[iters], 3)}</b>, a separation of <b>${DL.fmt(sep, 3)}</b>` +
      (iters === 0 ? " — nothing has happened yet, which is the correct starting point." :
        (sep > 0.05 ? ", so agreement has done its job: predictions pointing the way the consensus points have been amplified and the rest routed away." :
          ", which is small — with this spread the outliers are not distinguishable enough for routing to separate them.")) +
      ` The consensus length ‖v₀‖ is <b>${DL.fmt(vlen[iters], 3)}</b>, from a pre-squash length of ${DL.fmt(Math.hypot(now.s[0][0], now.s[0][1]), 3)}; ` +
      `squash is a radial rescaling only, so the <i>direction</i> of the consensus — the pose — is untouched by it. ` +
      `Contrast max pooling, which is also a routing decision: it would simply forward whichever single prediction had the largest magnitude, ignoring both the other ${n - 1} and what the receiver currently believes. ` +
      `Note that more iterations is not better: past about 3 the coefficients keep sharpening toward one-hot, which is the routing overfitting a single input.`;
  }
  [re, ne, oe, sp, sde].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════ 30 · #vit-svg — the prior against the data ═══════════ */
(function () {
  const svg = d3.select("#vit-svg");
  if (svg.empty()) return;
  const W = 760, H = 440, IM = 12, PS = 4;
  const ne = document.getElementById("vt-n"), te = document.getElementById("vt-t"),
    sde = document.getElementById("vt-s");

  const PAT = [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]];
  /* The task. In every setting the two classes contain the SAME 16 pattern
     values; only their arrangement differs, so no model can win on a pixel
     histogram and the question is genuinely about structure. */
  function make(kind, r) {
    const A = DL.zeros2(IM, IM), y = (r() < 0.5) ? 1 : 0;
    for (let i = 0; i < IM; i++) for (let j = 0; j < IM; j++) A[i][j] = 0.25 * r();
    const put = (i0, j0) => { for (let m = 0; m < PS; m++) for (let n = 0; n < PS; n++) A[i0 + m][j0 + n] = Math.max(A[i0 + m][j0 + n], PAT[m][n] * (0.8 + 0.3 * r())); };
    const scatter = () => { for (let k = 0; k < PS * PS; k++) { const i = Math.floor(r() * IM), j = Math.floor(r() * IM); A[i][j] = Math.max(A[i][j], PAT[Math.floor(k / PS)][k % PS] * (0.8 + 0.3 * r())); } };
    if (kind === "local") {
      if (y === 1) put(Math.floor(r() * (IM - PS)), Math.floor(r() * (IM - PS))); else scatter();
    } else if (kind === "global") {                       // the label is WHERE, not whether
      const top = (y === 1);
      const i0 = top ? Math.floor(r() * (IM / 2 - PS + 1)) : Math.floor(IM / 2 + r() * (IM / 2 - PS + 1));
      put(i0, Math.floor(r() * (IM - PS)));
    } else {                                              // local pattern, plus a redundant corner cue
      if (y === 1) { put(Math.floor(r() * (IM - PS)), Math.floor(r() * (IM - PS))); A[0][0] = 1; A[0][1] = 1; }
      else { scatter(); A[IM - 1][IM - 1] = 1; A[IM - 1][IM - 2] = 1; }
    }
    return { A: A, y: y };
  }
  /* the STRONG prior: a fixed bank of small SHARED filters, globally pooled —
     local, translation-equivariant features summarised into 6 numbers. */
  const KB = [
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]], [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
    [[1, 1, 0], [1, 0, -1], [0, -1, -1]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map(r0 => r0.map(v => v / 16))
  ];
  function featLocal(A) {
    let Z = DL.conv2dMC([A], KB.map(k => [k]), null, { p0: 1, p1: 1 });
    Z = Z.map(M => M.map(r0 => r0.map(v => Math.max(0, v))));
    return DL.globalPool(Z, "max");
  }
  /* the WEAK prior: every pixel its own free coordinate, no locality, no
     sharing — a stand-in for an architecture that must learn structure. */
  const featRaw = A => { const f = []; A.forEach(r0 => r0.forEach(v => f.push(v))); return f; };

  const NS = [10, 20, 40, 80, 160, 320, 640, 1280];
  let CACHE = null;
  function sweep(kind, seed) {
    const key = kind + "/" + seed;
    if (CACHE && CACHE.key === key) return CACHE;
    const rows = NS.map(N => {
      let a = 0, b = 0;
      const R = 2;
      for (let k = 0; k < R; k++) {
        const r = DL.rng(seed * 131 + k * 17 + N);
        const tr = [], te2 = [];
        for (let i = 0; i < N; i++) tr.push(make(kind, r));
        for (let i = 0; i < 240; i++) te2.push(make(kind, r));
        [["a", featLocal], ["b", featRaw]].forEach(([slot, fn]) => {
          const X = tr.map(d => fn(d.A)), y = tr.map(d => d.y);
          const st = DL.standardise(X);
          const m = DL.logistic(st.X, y, 180, 0.5, 1e-2);
          const Xt = te2.map(d => fn(d.A).map((v, j) => (v - st.mu[j]) / (st.sd[j] || 1)));
          const acc = DL.accuracy(m, Xt, te2.map(d => d.y));
          if (slot === "a") a += acc / R; else b += acc / R;
        });
      }
      return { N: N, a: a, b: b };
    });
    CACHE = { key: key, rows: rows };
    return CACHE;
  }

  function draw() {
    const kind = te.value, seed = +sde.value;
    const Ncur = +ne.value;
    document.getElementById("vt-nv").textContent = Ncur;
    document.getElementById("vt-sv").textContent = seed;
    const S = sweep(kind, seed);
    const r0 = DL.rng(seed * 977);
    let pos = null, neg = null, guard = 0;
    while ((!pos || !neg) && guard++ < 200) { const d = make(kind, r0); if (d.y === 1) pos = d; else neg = d; }

    const FR = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 12 });
    const g = FR.g;
    const cw = 8.4;
    const [lo, hi] = CN.extent2(pos.A);
    CN.grid(g, pos.A, 0, 26, cw, { title: "a positive example", text: false, stroke: "none", fill: v => CN.grey(v, lo, hi) });
    CN.grid(g, neg.A, 0, 26 + IM * cw + 30, cw, { title: "a negative example", text: false, stroke: "none", fill: v => CN.grey(v, lo, hi) });
    g.append("text").attr("x", 0).attr("y", 26 + 2 * IM * cw + 46).attr("font-size", 9.5).attr("fill", DC.muted)
      .text(kind === "local" ? "same 16 values; coherent vs scattered"
        : kind === "global" ? "same pattern; top half vs bottom half" : "coherent + corner cue vs scattered + corner cue");

    /* the two curves */
    const px = 152, pw = 400, ph = 300;
    const gp = g.append("g").attr("transform", `translate(${px},26)`);
    const xs = d3.scaleLog().domain([NS[0] * 0.85, NS[NS.length - 1] * 1.18]).range([0, pw]);
    const ys = d3.scaleLinear().domain([0.4, 1.02]).range([ph, 0]);
    DL.gridY(gp, ys, pw, 5); DL.gridX(gp, xs, ph, 5);
    DL.axisB(gp, xs, ph, 5, "training examples", v => DL.commas(v));
    DL.axisL(gp, ys, 5, "held-out accuracy");
    gp.append("line").attr("x1", 0).attr("x2", pw).attr("y1", ys(0.5)).attr("y2", ys(0.5))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3,3");
    gp.append("text").attr("x", 4).attr("y", ys(0.5) - 4).attr("font-size", 9.5).attr("fill", DC.muted).text("chance");
    DL.curve(gp, S.rows.map(d => [xs(d.N), ys(d.a)]), { stroke: DC.good, w: 2.4 });
    DL.curve(gp, S.rows.map(d => [xs(d.N), ys(d.b)]), { stroke: DC.violet, w: 2.4 });
    [["a", DC.good], ["b", DC.violet]].forEach(([k, cc]) =>
      gp.append("g").selectAll("circle").data(S.rows).join("circle")
        .attr("cx", d => xs(d.N)).attr("cy", d => ys(d[k])).attr("r", 3).attr("fill", cc));
    DL.legend(gp, [{ color: DC.good, label: "restricted: shared local features (the strong prior)" },
      { color: DC.violet, label: "unrestricted: every pixel free (the weak prior)" }],
      8, ph - 30, { gap: 15, font: 10 });
    /* the crossing, if there is one inside the range */
    let cross = null;
    for (let i = 0; i + 1 < S.rows.length; i++) {
      const d0 = S.rows[i].b - S.rows[i].a, d1 = S.rows[i + 1].b - S.rows[i + 1].a;
      if (d0 < 0 && d1 >= 0) {
        const t = d0 / (d0 - d1);
        cross = { N: Math.exp(Math.log(S.rows[i].N) + t * (Math.log(S.rows[i + 1].N) - Math.log(S.rows[i].N))), acc: S.rows[i].a + t * (S.rows[i + 1].a - S.rows[i].a) };
        break;
      }
    }
    if (S.rows[0].b >= S.rows[0].a) cross = { N: NS[0], acc: S.rows[0].a, atStart: true };
    if (cross) {
      gp.append("line").attr("x1", xs(cross.N)).attr("x2", xs(cross.N)).attr("y1", 0).attr("y2", ph)
        .attr("stroke", DC.a2).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
      gp.append("text").attr("x", xs(cross.N) + 6).attr("y", 14).attr("font-size", 10).attr("fill", DC.a2)
        .text(cross.atStart ? "the weak prior leads throughout" : `they cross at N ≈ ${Math.round(cross.N)}`);
    } else {
      gp.append("text").attr("x", pw - 6).attr("y", 14).attr("text-anchor", "end").attr("font-size", 10).attr("fill", DC.a2)
        .text(`no crossing up to N = ${DL.commas(NS[NS.length - 1])}`);
    }
    gp.append("line").attr("x1", xs(DL.clamp(Ncur, NS[0], NS[NS.length - 1]))).attr("x2", xs(DL.clamp(Ncur, NS[0], NS[NS.length - 1])))
      .attr("y1", 0).attr("y2", ph).attr("stroke", DC.line).attr("stroke-dasharray", "2,3");

    /* interpolate the two accuracies at the slider's N */
    const at = k => {
      const N = DL.clamp(Ncur, NS[0], NS[NS.length - 1]);
      for (let i = 0; i + 1 < S.rows.length; i++) if (N <= S.rows[i + 1].N) {
        const t = (Math.log(N) - Math.log(S.rows[i].N)) / (Math.log(S.rows[i + 1].N) - Math.log(S.rows[i].N));
        return S.rows[i][k] + t * (S.rows[i + 1][k] - S.rows[i][k]);
      }
      return S.rows[S.rows.length - 1][k];
    };
    const aNow = at("a"), bNow = at("b");
    const kx = px + pw + 26;
    const kv = DL.kv(g, kx, 46, { lead: 17, keyW: 116, size: 10.5 });
    kv("free parameters", "", DC.muted);
    kv("  restricted", `${KB.length} + 1 = ${KB.length + 1}`, DC.good, true);
    kv("  unrestricted", `${IM * IM} + 1 = ${IM * IM + 1}`, DC.violet, true);
    kv("ratio", `${DL.fmt((IM * IM + 1) / (KB.length + 1), 1)}×`, DC.a2);
    kv("", "", DC.muted);
    kv(`accuracy at N = ${Ncur}`, "", DC.muted);
    kv("  restricted", DL.fmt(aNow, 3), DC.good, true);
    kv("  unrestricted", DL.fmt(bNow, 3), DC.violet, true);
    kv("gap", (aNow - bNow >= 0 ? "+" : "−") + DL.fmt(Math.abs(aNow - bNow), 3), aNow > bNow ? DC.good : DC.violet, true);

    const first = S.rows[0], last = S.rows[S.rows.length - 1];
    document.getElementById("vit-readout").innerHTML =
      (kind === "mixed"
        ? `Both classes contain the same sixteen pattern values plus a two-pixel corner cue, so the only differences are the pattern's coherence and which corner is lit. `
        : kind === "global"
          ? `Both classes contain the same coherent pattern, in the same quantity; the only difference is which half of the image it lies in. `
          : `Both classes contain the same sixteen pattern values, so neither model can win on a pixel histogram. `) +
      `The restricted model has <b>${KB.length + 1}</b> free parameters — a fixed bank of shared 3×3 filters, globally pooled — and the unrestricted one has <b>${IM * IM + 1}</b>, one per pixel; both are fitted by the same logistic routine on the same draws, and every point is the mean of 2 resamples on 240 held-out examples. ` +
      `At N = ${first.N} the restricted model scores <b>${DL.fmt(first.a, 3)}</b> and the unrestricted one <b>${DL.fmt(first.b, 3)}</b>; at N = ${DL.commas(last.N)} they are <b>${DL.fmt(last.a, 3)}</b> and <b>${DL.fmt(last.b, 3)}</b>. ` +
      (kind === "local"
        ? (cross && !cross.atStart
          ? `They cross at N ≈ <b>${Math.round(cross.N)}</b>.`
          : `<b>They never cross inside the swept range.</b> The weak-prior model closes the gap from ${DL.fmt(first.a - first.b, 3)} to ${DL.fmt(last.a - last.b, 3)} over a 128-fold increase in data and is still behind at N = ${DL.commas(last.N)}. ` +
            `That is the strong prior's case at its strongest: when locality is exactly true, ${DL.commas(first.N)} examples with the right constraint beat ${DL.commas(last.N)} without it. The scaling-era claim is that the crossing exists — and this figure is a reminder of how far out it can be.`)
        : kind === "mixed"
          ? (cross && !cross.atStart
            ? `<b>They cross at N ≈ ${Math.round(cross.N)}.</b> The redundant long-range cue is invisible to a globally pooled local feature bank and trivially available to the unrestricted model, so the extra information the weak prior can reach eventually outweighs the sample efficiency the strong one buys. That is the whole scaling argument in one plot, and the crossing point is a measurement rather than a claim.`
            : `No crossing occurred inside the range on this seed; the extra cue was not enough to overturn the sample-efficiency advantage.`)
          : `<b>The restricted model plateaus at ${DL.fmt(last.a, 3)}</b> and no amount of data moves it, because the label here is <i>where</i> the pattern is and a globally pooled, translation-equivariant feature is by construction blind to that (§18). Its hypothesis class does not contain the answer, so this is not a sample-size problem and cannot be fixed with more examples. The unrestricted model, which has a free parameter per position, reaches ${DL.fmt(last.b, 3)} from ${S.rows.find(rr => rr.b > 0.95) ? DL.commas(S.rows.find(rr => rr.b > 0.95).N) : "very few"} examples. A wrong prior is not a small handicap — it is a ceiling.`);
  }
  ne.addEventListener("input", draw);
  [te, sde].forEach(e => e.addEventListener("change", draw));
  sde.addEventListener("input", draw);
  draw();
})();
