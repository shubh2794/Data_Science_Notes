/* image-processing.viz.js — the twenty-two visualizations on vision/image-processing.html.
   Loaded after ../data.js → ../notes.js → vision-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #tone-svg    a tone curve, the patch it produces, and the histogram it leaves
     2  #hist-svg    histogram → CDF → equalisation, with the flatness actually measured
     3  #clahe-svg   global vs per-block vs bilinearly-interpolated (CLAHE) equalisation
     4  #conv-svg    a kernel editor: type a 3×3 and see correlation AND convolution
     5  #pad-svg     the five border modes, on a signal and on a 2D corner
     6  #smooth-svg  box, tent, binomial and sampled Gaussian: shape, moments, truncation
     7  #sep-svg     separability as two 1D passes, with the multiply count kept live
     8  #moment-svg  the moment conditions, measured for a menu of kernels
     9  #deriv-svg   central difference, Sobel, Laplacian, LoG and DoG on one edge
    10  #steer-svg   steer an oriented derivative from three fixed basis filters
    11  #bilat-svg   the bilateral filter's two weight maps, separately, at one pixel
    12  #shoot-svg   box vs Gaussian vs median vs bilateral on one noisy edge, measured
    13  #morph-svg   a structuring element sliding over a binary shape, as a count map
    14  #dt-svg      the two-pass city-block distance transform, sweep by sweep
    15  #integ-svg   an integral image, its four-corner query, and the direct sum
    16  #dec-svg     decimation with and without pre-blurring — the aliasing is visible
    17  #pyr-svg     Gaussian and Laplacian pyramids, with the reconstruction error
    18  #blend-svg   splice vs feather vs multi-resolution blend
    19  #wave-svg    a lifted (5/3) wavelet: subbands, sparsity, perfect reconstruction
    20  #warp-svg    forward warping leaves holes; inverse warping cannot
    21  #interp-svg  nearest, bilinear, bicubic and windowed sinc on a hard edge
    22  #mip-svg     the resampling Jacobian, the MIP-map level, and the anisotropic fix

   Every number these print is recomputed from the arrays they draw, so the pictures and
   the prose cannot drift apart.                                                       */

/* ══════════ page-local helpers (deliberately NOT in vision-viz.js yet) ══════════ */
const IP = (function () {

  /* ── 1-D and 2-D array plumbing ────────────────────────────────────────── */
  const zeros = n => new Float64Array(n);
  function zeros2(h, w) { const A = []; for (let i = 0; i < h; i++) A.push(new Float64Array(w)); return A; }
  function clone2(A) { return A.map(r => Float64Array.from(r)); }
  function map2(A, fn) { return A.map((r, i) => Float64Array.from(r, (v, j) => fn(v, i, j))); }
  function minmax2(A) {
    let lo = Infinity, hi = -Infinity;
    for (const r of A) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
  }
  function sum2(A) { let s = 0; for (const r of A) for (const v of r) s += v; return s; }
  /* Array.prototype.flat does NOT flatten typed-array rows, and most rows here are
     Float64Arrays. Every figure uses this instead. */
  function flat(A) { const o = []; for (const r of A) for (const v of r) o.push(v); return o; }
  const absMax = A => { let m = 0; for (const r of A) for (const v of r) { const a = Math.abs(v); if (a > m) m = a; } return m; };

  /* ── border handling. ONE function, used by every filter on the page, so the
        page cannot claim one convention and compute another.
        Modes: zero, const, clamp, wrap, mirror (whole-sample: …cb|abcd|cb…),
               symm  (half-sample: …ba|abcd|dc…)                                 */
  function bidx(i, n, mode) {
    if (i >= 0 && i < n) return i;
    if (n === 1) return 0;
    switch (mode) {
      case "zero": case "const": return -1;                  // caller substitutes cval
      case "clamp": return i < 0 ? 0 : n - 1;
      case "wrap": return ((i % n) + n) % n;
      case "mirror": {                                        // period 2n − 2
        const p = 2 * n - 2; let k = ((i % p) + p) % p;
        return k < n ? k : p - k;
      }
      case "symm": {                                          // period 2n
        const p = 2 * n; let k = ((i % p) + p) % p;
        return k < n ? k : p - 1 - k;
      }
      default: return i < 0 ? 0 : n - 1;
    }
  }
  function at1(f, i, mode, cval) {
    const k = bidx(i, f.length, mode);
    return k < 0 ? (cval === undefined ? 0 : cval) : f[k];
  }
  function at2(A, i, j, mode, cval) {
    const ii = bidx(i, A.length, mode), jj = bidx(j, A[0].length, mode);
    return (ii < 0 || jj < 0) ? (cval === undefined ? 0 : cval) : A[ii][jj];
  }

  /* ── correlation and convolution. The ONLY difference is the sign of the offset,
        and both are written out longhand so the page can point at the line.       */
  function corr1(f, h, mode, cval) {                 // g(i) = Σ_k f(i+k) h(k)
    const R = (h.length - 1) >> 1, g = zeros(f.length);
    for (let i = 0; i < f.length; i++) { let s = 0; for (let k = -R; k <= R; k++) s += at1(f, i + k, mode, cval) * h[k + R]; g[i] = s; }
    return g;
  }
  function conv1(f, h, mode, cval) {                 // g(i) = Σ_k f(i−k) h(k)
    const R = (h.length - 1) >> 1, g = zeros(f.length);
    for (let i = 0; i < f.length; i++) { let s = 0; for (let k = -R; k <= R; k++) s += at1(f, i - k, mode, cval) * h[k + R]; g[i] = s; }
    return g;
  }
  function corr2(A, H, mode, cval) {
    const R = (H.length - 1) >> 1, C = (H[0].length - 1) >> 1;
    const G = zeros2(A.length, A[0].length);
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
      let s = 0;
      for (let k = -R; k <= R; k++) for (let l = -C; l <= C; l++) s += at2(A, i + k, j + l, mode, cval) * H[k + R][l + C];
      G[i][j] = s;
    }
    return G;
  }
  const flip2 = H => H.map(r => r.slice()).reverse().map(r => r.reverse());
  function conv2(A, H, mode, cval) { return corr2(A, flip2(H), mode, cval); }

  /* separable: one horizontal pass then one vertical pass, kept distinct so the
     multiply count in §11 is the count of what actually ran */
  function sepH(A, h, mode, cval) {
    const R = (h.length - 1) >> 1, G = zeros2(A.length, A[0].length);
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
      let s = 0; for (let l = -R; l <= R; l++) s += at2(A, i, j + l, mode, cval) * h[l + R];
      G[i][j] = s;
    }
    return G;
  }
  function sepV(A, h, mode, cval) {
    const R = (h.length - 1) >> 1, G = zeros2(A.length, A[0].length);
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
      let s = 0; for (let k = -R; k <= R; k++) s += at2(A, i + k, j, mode, cval) * h[k + R];
      G[i][j] = s;
    }
    return G;
  }
  const sep2 = (A, h, mode, cval) => sepV(sepH(A, h, mode, cval), h, mode, cval);
  const outer = (v, h) => v.map ? Array.from(v, a => Array.from(h, b => a * b))
    : Array.from(v).map(a => Array.from(h).map(b => a * b));

  /* ── kernels ───────────────────────────────────────────────────────────── */
  function box(K) { const h = zeros(K); h.fill(1 / K); return h; }
  function tent(K) {                                   // triangular, unit sum, odd K
    const R = (K - 1) >> 1, h = zeros(K); let s = 0;
    for (let k = -R; k <= R; k++) { h[k + R] = (R + 1) - Math.abs(k); s += h[k + R]; }
    for (let k = 0; k < K; k++) h[k] /= s;
    return h;
  }
  function binomial(n) {                               // row n of Pascal, normalised
    const h = zeros(n + 1); let c = 1;
    for (let k = 0; k <= n; k++) { h[k] = c; c = c * (n - k) / (k + 1); }
    const s = h.reduce((a, b) => a + b, 0);
    for (let k = 0; k <= n; k++) h[k] /= s;
    return h;
  }
  function gauss1(sigma, trunc) {                      // sampled, truncated, renormalised
    const R = Math.max(1, Math.ceil((trunc === undefined ? 3 : trunc) * sigma));
    const h = zeros(2 * R + 1); let s = 0;
    for (let k = -R; k <= R; k++) { const v = Math.exp(-(k * k) / (2 * sigma * sigma)); h[k + R] = v; s += v; }
    for (let k = 0; k < h.length; k++) h[k] /= s;
    return h;
  }
  /* the moment conditions of §12, measured rather than asserted */
  function moments(h) {
    const R = (h.length - 1) >> 1;
    let m0 = 0, m1 = 0, m2 = 0, ev = 0, od = 0, nyq = 0;
    for (let k = -R; k <= R; k++) {
      const v = h[k + R];
      m0 += v; m1 += k * v; m2 += k * k * v; nyq += v * ((k & 1) ? -1 : 1);
      if (((k % 2) + 2) % 2 === 0) ev += v; else od += v;
    }
    return { m0, m1, m2, even: ev, odd: od, nyq, centroid: m0 !== 0 ? m1 / m0 : NaN };
  }
  /* the discrete-time frequency response H(ω) = Σ h(k) e^{−jωk}; symmetric kernels
     give a real response, which is why every low-pass kernel here is symmetric. */
  function freqResp(h, w) {
    const R = (h.length - 1) >> 1; let re = 0, im = 0;
    for (let k = -R; k <= R; k++) { re += h[k + R] * Math.cos(w * k); im -= h[k + R] * Math.sin(w * k); }
    return { re, im, mag: Math.hypot(re, im) };
  }

  /* ── the running 8×8 patch of §01, verbatim ────────────────────────────── */
  const PATCH = [
    [45, 60, 98, 127, 132, 133, 137, 133],
    [46, 65, 98, 123, 126, 128, 131, 133],
    [47, 65, 96, 115, 119, 123, 135, 137],
    [47, 63, 91, 107, 113, 122, 138, 134],
    [50, 59, 80, 97, 110, 123, 133, 134],
    [49, 53, 68, 83, 97, 113, 128, 133],
    [50, 50, 58, 70, 84, 102, 116, 126],
    [50, 50, 52, 58, 69, 86, 101, 120]
  ].map(r => Float64Array.from(r));

  /* ── a larger synthetic scene, shared by the filtering figures. Deterministic.
        It contains: a smooth luminance ramp, a hard disc, a bar grating whose pitch
        falls with x (so §16's aliasing has something to alias), a soft edge, and a
        few isolated specks for the median filter to eat.                          */
  function scene(W, H, opt) {
    const o = Object.assign({ noise: 0, shot: 0, seed: 7 }, opt || {});
    const r = VZ.rng(o.seed), A = zeros2(H, W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const x = j / (W - 1), y = i / (H - 1);
      let v = 0.28 + 0.20 * x;                                  // a gentle ramp
      if (x > 0.52) v += 0.34;                                  // a hard vertical edge
      const dx = j - W * 0.24, dy = i - H * 0.34, rr = Math.hypot(dx, dy);
      if (rr < Math.min(W, H) * 0.17) v = 0.80;                 // a bright disc
      if (y > 0.66) {                                           // a grating, pitch falling with x
        const per = 14 - 11 * x;
        v = 0.34 + 0.30 * (0.5 + 0.5 * Math.cos(2 * Math.PI * j / per));
      }
      A[i][j] = v;
    }
    if (o.noise > 0) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) A[i][j] += o.noise * VZ.randn(r);
    if (o.shot > 0) {
      const n = Math.round(o.shot * W * H);
      for (let k = 0; k < n; k++) A[Math.floor(r() * H)][Math.floor(r() * W)] = r() < 0.5 ? 0 : 1;
    }
    return map2(A, v => VZ.clamp(v, 0, 1));
  }

  /* ── drawing: a raster of a 2-D array as filled rects, grey or a signed ramp ── */
  function raster(g, A, x, y, cell, opt) {
    const o = Object.assign({ lo: null, hi: null, signed: false, stroke: null, grid: false, gamma: 1 }, opt || {});
    let lo = o.lo, hi = o.hi;
    if (lo === null || hi === null) { const m = minmax2(A); lo = (lo === null ? m[0] : lo); hi = (hi === null ? m[1] : hi); }
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    /* horizontal run-length merging: adjacent cells of identical colour become one
       rect. Purely a DOM-size optimisation — the picture is bit-identical. */
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    for (let i = 0; i < A.length; i++) {
      let j = 0;
      while (j < A[0].length) {
        const t = hi > lo ? (A[i][j] - lo) / (hi - lo) : 0.5;
        const col = o.signed ? signedColor(A[i][j], m) : greyColor(t, o.gamma);
        let k = j + 1;
        while (k < A[0].length) {
          const t2 = hi > lo ? (A[i][k] - lo) / (hi - lo) : 0.5;
          if ((o.signed ? signedColor(A[i][k], m) : greyColor(t2, o.gamma)) !== col) break;
          k++;
        }
        gg.append("rect").attr("x", j * cell).attr("y", i * cell)
          .attr("width", (k - j) * cell + 0.6).attr("height", cell + 0.6)
          .attr("fill", col).attr("shape-rendering", "crispEdges");
        j = k;
      }
    }
    if (o.grid && cell >= 9) {
      for (let i = 0; i <= A.length; i++) gg.append("line").attr("x1", 0).attr("x2", A[0].length * cell)
        .attr("y1", i * cell).attr("y2", i * cell).attr("stroke", VC.bg).attr("stroke-opacity", 0.35).attr("stroke-width", 0.5);
      for (let j = 0; j <= A[0].length; j++) gg.append("line").attr("y1", 0).attr("y2", A.length * cell)
        .attr("x1", j * cell).attr("x2", j * cell).attr("stroke", VC.bg).attr("stroke-opacity", 0.35).attr("stroke-width", 0.5);
    }
    gg.append("rect").attr("x", 0).attr("y", 0).attr("width", A[0].length * cell).attr("height", A.length * cell)
      .attr("fill", "none").attr("stroke", o.stroke || VC.line);
    return gg;
  }
  function greyColor(t, gamma) {
    const u = VZ.clamp(gamma && gamma !== 1 ? Math.pow(VZ.clamp(t, 0, 1), gamma) : t, 0, 1);
    const c = Math.round(255 * u);
    return `rgb(${c},${c},${c})`;
  }
  /* a diverging blue↔orange ramp for signed data (gradients, Laplacians, residuals) */
  function signedColor(v, m) {
    const t = VZ.clamp(m > 0 ? v / m : 0, -1, 1);
    if (t >= 0) return d3.interpolateRgb("#1b2130", VC.a2)(t);
    return d3.interpolateRgb("#1b2130", VC.accent)(-t);
  }

  /* a small caption above a sub-panel */
  function cap(g, x, y, text, color) {
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", 10.5)
      .attr("fill", color || VC.muted).text(text);
  }
  /* a monospace number block, aligned, for printing kernels inside an svg */
  function numText(g, rows, x, y, opt) {
    const o = Object.assign({ size: 10.5, lead: 13, fill: VC.ink }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    rows.forEach((r, i) => gg.append("text").attr("x", 0).attr("y", i * o.lead)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("font-size", o.size)
      .attr("xml:space", "preserve").attr("fill", o.fill).text(r));
    return gg;
  }

  /* ── histogram helpers, used by §04–§07 ────────────────────────────────── */
  function hist256(A) {
    const h = new Float64Array(256);
    for (const r of A) for (const v of r) h[VZ.clamp(Math.round(v), 0, 255)]++;
    return h;
  }
  function cdf(h) {
    const N = h.reduce((a, b) => a + b, 0), c = new Float64Array(h.length);
    let s = 0; for (let i = 0; i < h.length; i++) { s += h[i]; c[i] = s / N; }
    return c;
  }
  function entropy(h) {
    const N = h.reduce((a, b) => a + b, 0); let e = 0;
    for (const v of h) if (v > 0) { const p = v / N; e -= p * Math.log2(p); }
    return e;
  }

  return {
    zeros, zeros2, clone2, map2, minmax2, sum2, flat, absMax,
    bidx, at1, at2, corr1, conv1, corr2, conv2, flip2, sepH, sepV, sep2, outer,
    box, tent, binomial, gauss1, moments, freqResp,
    PATCH, scene, raster, greyColor, signedColor, cap, numText,
    hist256, cdf, entropy
  };
})();

/* ───────── 1 · a tone curve, the patch it makes, and the histogram left behind ───────── */
(function () {
  const svg = d3.select("#tone-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 420;
  const eOp = document.getElementById("tn-op");
  const eA = document.getElementById("tn-a"), eAv = document.getElementById("tn-av");
  const eB = document.getElementById("tn-b"), eBv = document.getElementById("tn-bv");
  const eG = document.getElementById("tn-g"), eGv = document.getElementById("tn-gv");
  const eK = document.getElementById("tn-k"), eKv = document.getElementById("tn-kv");
  const eC = document.getElementById("tn-clip");
  const out = document.getElementById("tone-readout");

  /* the operator, as a raw (unclipped) map on [0, 255] */
  function raw(v, op, a, b, gm, K) {
    const u = v / 255;
    switch (op) {
      case "affine": return a * v + b;
      case "gamma": return 255 * Math.pow(u, gm);
      case "scurve": {                                  /* a smoothstep re-centred by b */
        const c = VZ.clamp(0.5 + b / 255, 0.02, 0.98);
        const t = VZ.clamp((u - c) * a + 0.5, 0, 1);
        return 255 * t * t * (3 - 2 * t);
      }
      case "shoulder": return 255 * (1 - Math.exp(-a * u * (1 + K / 8)));
      case "post": return Math.round(u * (K - 1)) / (K - 1) * 255;
      case "invert": return 255 - (a * v + b);
      case "thresh": return (a * v + b) >= 128 ? 255 : 0;
    }
    return v;
  }

  function draw() {
    const op = eOp.value, a = +eA.value, b = +eB.value, gm = +eG.value, K = +eK.value, doClip = eC.checked;
    const T = new Float64Array(256), Traw = new Float64Array(256);
    for (let v = 0; v < 256; v++) {
      const y = raw(v, op, a, b, gm, K);
      Traw[v] = y;
      T[v] = doClip ? VZ.clamp(y, 0, 255) : y;
    }

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── left: the transfer curve ── */
    const cw = 250, ch = 250, cx0 = 34, cy0 = 26;
    const sx = d3.scaleLinear().domain([0, 255]).range([cx0, cx0 + cw]);
    const sy = d3.scaleLinear().domain([0, 255]).range([cy0 + ch, cy0]);
    g.append("rect").attr("x", cx0).attr("y", cy0).attr("width", cw).attr("height", ch)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    for (let t = 0; t <= 255; t += 64) {
      g.append("line").attr("x1", sx(t)).attr("x2", sx(t)).attr("y1", cy0).attr("y2", cy0 + ch)
        .attr("stroke", VC.grid);
      g.append("line").attr("y1", sy(t)).attr("y2", sy(t)).attr("x1", cx0).attr("x2", cx0 + cw)
        .attr("stroke", VC.grid);
    }
    g.append("line").attr("x1", sx(0)).attr("y1", sy(0)).attr("x2", sx(255)).attr("y2", sy(255))
      .attr("stroke", VC.muted).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
    /* the curve, split into the part inside the box and the clipped part */
    let dIn = "", dHi = "", dLo = "";
    for (let v = 0; v < 256; v++) {
      const yv = VZ.clamp(Traw[v], -60, 315);
      const pt = `${sx(v).toFixed(2)},${sy(VZ.clamp(yv, 0, 255)).toFixed(2)}`;
      if (Traw[v] > 255) dHi += (dHi ? "L" : "M") + pt;
      else if (Traw[v] < 0) dLo += (dLo ? "L" : "M") + pt;
      else dIn += (dIn ? "L" : "M") + pt;
    }
    if (dIn) g.append("path").attr("d", dIn).attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2.2);
    [dHi, dLo].forEach(d => { if (d) g.append("path").attr("d", d).attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 3); });
    IP.cap(g, cx0, cy0 - 8, "transfer curve  T(v)   —   input code → output code", VC.ink);
    g.append("text").attr("x", cx0 + cw).attr("y", cy0 + ch + 15).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("input v");
    g.append("text").attr("x", cx0 - 8).attr("y", cy0 + 4).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("255");
    g.append("text").attr("x", cx0 - 8).attr("y", cy0 + ch).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("0");

    /* ── middle: the patch, before and after ── */
    const cell = 15, px0 = 318, py0 = 26;
    const Gp = IP.map2(IP.PATCH, v => VZ.clamp(T[VZ.clamp(Math.round(v), 0, 255)], 0, 255));
    IP.cap(g, px0, py0 - 8, "the running 8×8 patch", VC.ink);
    IP.raster(g, IP.PATCH, px0, py0, cell, { lo: 0, hi: 255, grid: true });
    IP.cap(g, px0, py0 + 8 * cell + 14, "input", VC.muted);
    IP.cap(g, px0, py0 + 8 * cell + 40, "output", VC.muted);
    IP.raster(g, Gp, px0, py0 + 8 * cell + 48, cell, { lo: 0, hi: 255, grid: true });

    /* ── right: the two histograms, back to back ── */
    const hx0 = 470, hy0 = 26, hw = 268, hh = 250;
    const hin = IP.hist256(IP.PATCH), hou = IP.hist256(Gp);
    const hxs = d3.scaleLinear().domain([0, 255]).range([hx0, hx0 + hw]);
    const top = d3.max([d3.max(hin), d3.max(hou)]);
    const hys = d3.scaleLinear().domain([0, top]).range([0, hh / 2 - 12]);
    g.append("rect").attr("x", hx0).attr("y", hy0).attr("width", hw).attr("height", hh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    const mid = hy0 + hh / 2;
    g.append("line").attr("x1", hx0).attr("x2", hx0 + hw).attr("y1", mid).attr("y2", mid).attr("stroke", VC.line);
    for (let v = 0; v < 256; v++) {
      if (hin[v]) g.append("rect").attr("x", hxs(v)).attr("width", Math.max(1, hw / 256))
        .attr("y", mid - 4 - hys(hin[v])).attr("height", hys(hin[v])).attr("fill", VC.accent).attr("fill-opacity", 0.85);
      if (hou[v]) g.append("rect").attr("x", hxs(v)).attr("width", Math.max(1, hw / 256))
        .attr("y", mid + 4).attr("height", hys(hou[v])).attr("fill", VC.a2).attr("fill-opacity", 0.85);
    }
    IP.cap(g, hx0, hy0 - 8, "histograms — input above, output below", VC.ink);
    IP.cap(g, hx0 + 4, mid - 8, "input", VC.accent);
    IP.cap(g, hx0 + 4, mid + 16, "output", VC.a2);
    [0, 128, 255].forEach(t => g.append("text").attr("x", hxs(t)).attr("y", hy0 + hh + 13)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", VC.muted).text(t));

    /* ── measured, not asserted ── */
    let n = 0, s = 0, s2 = 0, hiC = 0, loC = 0;
    for (const r of IP.PATCH) for (const v of r) {
      const y = raw(v, op, a, b, gm, K);
      if (y > 255) hiC++; if (y < 0) loC++;
      const z = VZ.clamp(T[VZ.clamp(Math.round(v), 0, 255)], 0, 255);
      n++; s += z; s2 += z * z;
    }
    const mean = s / n, sd = Math.sqrt(Math.max(0, s2 / n - mean * mean));
    const distinctOut = new Set(Array.from({ length: 256 }, (_, v) => Math.round(VZ.clamp(T[v], 0, 255)))).size;
    const survivors = new Set(); for (const r of Gp) for (const v of r) survivors.add(Math.round(v));

    out.innerHTML =
      `patch after the operator: mean <b>${VZ.fmt(mean, 3)}</b> (was 95.609), sd <b>${VZ.fmt(sd, 3)}</b> (was 32.432) · `
      + `distinct grey levels surviving in the patch: <b>${survivors.size}</b> of 44 · `
      + `the table T itself takes <b>${distinctOut}</b> distinct values out of 256<br>`
      + `clipped: <b>${hiC}</b> pixel${hiC === 1 ? "" : "s"} above 255, <b>${loC}</b> below 0`
      + (hiC + loC === 0 ? " — nothing is being destroyed at the ends." : " — those pixels are now indistinguishable and no later operator can separate them.")
      + (op === "affine" ? `<br>affine check: a·mean + b = ${VZ.fmt(a, 2)} × 95.609375 + ${b} = <b>${VZ.fmt(a * 95.609375 + b, 4)}</b>, `
        + `a·sd = <b>${VZ.fmt(a * 32.4324, 4)}</b> — equal to the measured values above whenever nothing clips.` : "");
  }

  eOp.onchange = draw;
  eA.oninput = () => { eAv.textContent = (+eA.value).toFixed(2); draw(); };
  eB.oninput = () => { eBv.textContent = eB.value; draw(); };
  eG.oninput = () => { eGv.textContent = (+eG.value).toFixed(2); draw(); };
  eK.oninput = () => { eKv.textContent = eK.value; draw(); };
  eC.onchange = draw;
  draw();
})();

/* ───────── 2 · histogram → CDF → equalisation, with the flatness measured ───────── */
(function () {
  const svg = d3.select("#hist-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470;
  const eSrc = document.getElementById("hq-src");
  const eAl = document.getElementById("hq-al"), eAlv = document.getElementById("hq-alv");
  const eBits = document.getElementById("hq-bits"), eBitsv = document.getElementById("hq-bitsv");
  const eCdf = document.getElementById("hq-cdf");
  const out = document.getElementById("hist-readout");

  /* the five test images, all deterministic, all quantised to 0…255 */
  function source(name) {
    if (name === "patch") return IP.clone2(IP.PATCH);
    const w = 64, h = 48, r = VZ.rng(19), A = IP.zeros2(h, w);
    for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
      const x = j / (w - 1), y = i / (h - 1);
      /* a little scene: sky gradient, a dark building block, a bright window */
      let v = 0.55 + 0.25 * (1 - y);
      if (y > 0.42 && x > 0.18 && x < 0.72) v = 0.16 + 0.05 * Math.cos(9 * x) * Math.cos(7 * y);
      if (y > 0.52 && y < 0.66 && x > 0.30 && x < 0.44) v = 0.30;
      if (y > 0.80) v = 0.22 + 0.10 * x;
      A[i][j] = v;
    }
    let f;
    if (name === "dark") f = v => 0.22 * v;
    else if (name === "lowc") f = v => 0.38 + 0.24 * v;
    else if (name === "bimod") f = v => (v > 0.4 ? 0.78 + 0.12 * (v - 0.4) : 0.10 + 0.15 * v);
    else f = v => 0.40 + 0.20 * v;
    const noisy = (name === "noisy");
    return IP.map2(A, v => {
      let z = f(v);
      if (noisy) z += 0.045 * VZ.randn(r);
      return VZ.clamp(Math.round(255 * z), 0, 255);
    });
  }

  function draw() {
    const bits = +eBits.value, L = 1 << bits, al = +eAl.value;
    const A0 = source(eSrc.value);
    /* quantise to the chosen bit depth, then work in 0…255 for display */
    const q = 256 / L;
    const A = IP.map2(A0, v => VZ.clamp(Math.floor(v / q), 0, L - 1));

    const N = A.length * A[0].length;
    const hin = new Float64Array(L);
    for (const row of A) for (const v of row) hin[v]++;
    const cin = new Float64Array(L); { let s = 0; for (let v = 0; v < L; v++) { s += hin[v]; cin[v] = s / N; } }
    const T = new Float64Array(L);
    for (let v = 0; v < L; v++) T[v] = VZ.clamp(Math.round(al * (L - 1) * cin[v] + (1 - al) * v), 0, L - 1);
    const B = IP.map2(A, v => T[v]);
    const hou = new Float64Array(L);
    for (const row of B) for (const v of row) hou[v]++;
    const cou = new Float64Array(L); { let s = 0; for (let v = 0; v < L; v++) { s += hou[v]; cou[v] = s / N; } }

    const dev = c => { let m = 0; for (let v = 0; v < L; v++) m = Math.max(m, Math.abs(c[v] - (v + 1) / L)); return m; };
    const ent = hh => { let e = 0; for (const v of hh) if (v > 0) { const p = v / N; e -= p * Math.log2(p); } return e; };
    const occ = hh => { let n = 0; for (const v of hh) if (v > 0) n++; return n; };
    const stats = M => {
      let s = 0, s2 = 0, n = 0; for (const row of M) for (const v of row) { s += v; s2 += v * v; n++; }
      const m = s / n; return [m, Math.sqrt(Math.max(0, s2 / n - m * m))];
    };

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── top: the two images ── */
    const cell = A[0].length <= 8 ? 18 : 4.6;
    const iw = A[0].length * cell, ih = A.length * cell;
    const ix0 = 60, ix1 = 60 + iw + 90;
    IP.cap(g, ix0, 14, "input", VC.accent);
    IP.raster(g, A, ix0, 20, cell, { lo: 0, hi: L - 1, grid: A[0].length <= 8 });
    IP.cap(g, ix1, 14, al >= 0.999 ? "equalised" : "partially equalised, α = " + al.toFixed(2), VC.a2);
    IP.raster(g, B, ix1, 20, cell, { lo: 0, hi: L - 1, grid: A[0].length <= 8 });

    /* the transfer curve, in the gap between the two images */
    {
      const bx = ix0 + iw + 14, by = 20, bw = 62, bh = Math.min(ih, 150);
      g.append("rect").attr("x", bx).attr("y", by).attr("width", bw).attr("height", bh)
        .attr("fill", VC.panel2).attr("stroke", VC.line);
      const tx = d3.scaleLinear().domain([0, L - 1]).range([bx, bx + bw]);
      const ty = d3.scaleLinear().domain([0, L - 1]).range([by + bh, by]);
      g.append("line").attr("x1", tx(0)).attr("y1", ty(0)).attr("x2", tx(L - 1)).attr("y2", ty(L - 1))
        .attr("stroke", VC.muted).attr("stroke-dasharray", "2 3");
      let d = "";
      for (let v = 0; v < L; v++) d += (v ? "L" : "M") + tx(v).toFixed(1) + "," + ty(T[v]).toFixed(1);
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
      IP.cap(g, bx, by - 6, "T(v)", VC.a2);
    }

    /* ── bottom: the two histogram panels ── */
    const py = 200, ph = 200, pw = 350;
    function panel(x, hh, cc, colour, title) {
      g.append("rect").attr("x", x).attr("y", py).attr("width", pw).attr("height", ph)
        .attr("fill", VC.panel2).attr("stroke", VC.line);
      const hx = d3.scaleLinear().domain([0, L - 1]).range([x + 4, x + pw - 4]);
      const top = d3.max(hh) || 1;
      const hy = d3.scaleLinear().domain([0, top]).range([py + ph - 18, py + 10]);
      const cy = d3.scaleLinear().domain([0, 1]).range([py + ph - 18, py + 10]);
      const bw = Math.max(1, (pw - 8) / L);
      for (let v = 0; v < L; v++) if (hh[v] > 0)
        g.append("rect").attr("x", hx(v)).attr("width", bw).attr("y", hy(hh[v]))
          .attr("height", py + ph - 18 - hy(hh[v])).attr("fill", colour).attr("fill-opacity", 0.85);
      if (eCdf.checked) {
        g.append("line").attr("x1", hx(0)).attr("y1", cy(0)).attr("x2", hx(L - 1)).attr("y2", cy(1))
          .attr("stroke", VC.muted).attr("stroke-dasharray", "3 3");
        let d = "";
        for (let v = 0; v < L; v++) d += (v ? "L" : "M") + hx(v).toFixed(1) + "," + cy(cc[v]).toFixed(1);
        g.append("path").attr("d", d).attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
      }
      g.append("line").attr("x1", x + 4).attr("x2", x + pw - 4).attr("y1", py + ph - 18).attr("y2", py + ph - 18)
        .attr("stroke", VC.line);
      IP.cap(g, x + 4, py - 6, title, VC.ink);
      [0, (L - 1) >> 1, L - 1].forEach(t => g.append("text").attr("x", hx(t)).attr("y", py + ph - 4)
        .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", VC.muted).text(t));
    }
    panel(10, hin, cin, VC.accent, "input histogram   ·   its CDF in green   ·   uniform CDF dashed");
    panel(400, hou, cou, VC.a2, "output histogram   ·   its CDF   ·   the same reference diagonal");

    const [m0, s0] = stats(A), [m1, s1] = stats(B);
    out.innerHTML =
      `N = ${N} pixels, ${bits}-bit (${L} levels) · mean <b>${VZ.fmt(m0, 3)} → ${VZ.fmt(m1, 3)}</b>, `
      + `sd <b>${VZ.fmt(s0, 3)} → ${VZ.fmt(s1, 3)}</b><br>`
      + `max |CDF − uniform|: <b>${VZ.fmt(dev(cin), 4)} → ${VZ.fmt(dev(cou), 4)}</b> — this is what equalisation actually flattens · `
      + `occupied levels <b>${occ(hin)} → ${occ(hou)}</b> · entropy <b>${VZ.fmt(ent(hin), 4)} → ${VZ.fmt(ent(hou), 4)} bits</b>`
      + (ent(hou) < ent(hin) - 1e-9
        ? ` — <b>lower</b>, because ${occ(hin) - occ(hou)} level${occ(hin) - occ(hou) === 1 ? " was" : "s were"} merged. Information was destroyed, not created.`
        : ` — unchanged, because T is injective on the levels present here. It can never rise.`)
      + `<br>tallest single output bin holds ${VZ.fmt(100 * d3.max(hou) / N, 2)}% of the pixels; a genuinely flat histogram would put ${VZ.fmt(100 / L, 3)}% in every bin.`;
  }

  eSrc.onchange = draw;
  eAl.oninput = () => { eAlv.textContent = (+eAl.value).toFixed(2); draw(); };
  eBits.oninput = () => { eBitsv.textContent = eBits.value; draw(); };
  eCdf.onchange = draw;
  draw();
})();

/* ───────── 3 · global vs block vs interpolated vs clipped equalisation ───────── */
(function () {
  const svg = d3.select("#clahe-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430, IW = 112, IH = 84, L = 256;
  const eM = document.getElementById("cl-m");
  const eB = document.getElementById("cl-M"), eBv = document.getElementById("cl-Mv");
  const eC = document.getElementById("cl-c"), eCv = document.getElementById("cl-cv");
  const eN = document.getElementById("cl-n"), eNv = document.getElementById("cl-nv");
  const eS = document.getElementById("cl-seam");
  const out = document.getElementById("clahe-readout");

  /* a scene with a bright sky and a much darker foreground — the case global
     equalisation is supposed to help with and does not */
  function scene(noise) {
    const r = VZ.rng(23), A = IP.zeros2(IH, IW);
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      const x = j / (IW - 1), y = i / (IH - 1);
      let v = 0.55 + 0.25 * (1 - y);                                     // sky
      if (y > 0.42 && x > 0.18 && x < 0.72) v = 0.16 + 0.05 * Math.cos(9 * x) * Math.cos(7 * y);
      if (y > 0.52 && y < 0.66 && x > 0.30 && x < 0.44) v = 0.30;        // a window
      if (y > 0.80) v = 0.22 + 0.10 * x;                                 // shadowed ground
      A[i][j] = VZ.clamp(v + noise * VZ.randn(r), 0, 1);
    }
    return IP.map2(A, v => Math.round(255 * v));
  }

  const lutOf = (hist, n) => {
    const T = new Float64Array(L); let s = 0;
    for (let v = 0; v < L; v++) { s += hist[v]; T[v] = Math.round((L - 1) * s / n); }
    return T;
  };
  function clipHist(hist, n, c) {
    const beta = c * n / L, out_ = new Float64Array(L); let E = 0;
    for (let v = 0; v < L; v++) { const x = Math.min(hist[v], beta); E += hist[v] - x; out_[v] = x; }
    for (let v = 0; v < L; v++) out_[v] += E / L;
    return { h: out_, beta, E };
  }

  function draw() {
    const method = eM.value, M = +eB.value, cf = +eC.value, nz = +eN.value;
    const A = scene(nz);
    const nbi = Math.max(1, Math.floor(IH / M)), nbj = Math.max(1, Math.floor(IW / M));
    const N = IW * IH;

    /* per-block tables */
    const LUT = [], BETA = [], EXC = [];
    for (let bi = 0; bi < nbi; bi++) {
      LUT.push([]); BETA.push([]); EXC.push([]);
      for (let bj = 0; bj < nbj; bj++) {
        const i0 = bi * M, j0 = bj * M;
        const i1 = (bi === nbi - 1) ? IH : i0 + M, j1 = (bj === nbj - 1) ? IW : j0 + M;
        const hh = new Float64Array(L); let n = 0;
        for (let i = i0; i < i1; i++) for (let j = j0; j < j1; j++) { hh[A[i][j]]++; n++; }
        let use = hh, beta = Infinity, E = 0;
        if (method === "clahe") { const r = clipHist(hh, n, cf); use = r.h; beta = r.beta; E = r.E; }
        LUT[bi].push(lutOf(use, n)); BETA[bi].push(beta); EXC[bi].push(E);
      }
    }
    const gh = new Float64Array(L); for (const row of A) for (const v of row) gh[v]++;
    const GT = lutOf(gh, N);

    const lut = (p, q, v) => LUT[VZ.clamp(p, 0, nbi - 1)][VZ.clamp(q, 0, nbj - 1)][v];
    const B = IP.zeros2(IH, IW);
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      const v = A[i][j];
      if (method === "none") B[i][j] = v;
      else if (method === "global") B[i][j] = GT[v];
      else if (method === "block") B[i][j] = lut(Math.min(Math.floor(i / M), nbi - 1), Math.min(Math.floor(j / M), nbj - 1), v);
      else {
        const s = (i - M / 2) / M, t = (j - M / 2) / M;
        const bi = Math.floor(s), bj = Math.floor(t), a = s - bi, b = t - bj;
        B[i][j] = (1 - a) * (1 - b) * lut(bi, bj, v) + (1 - a) * b * lut(bi, bj + 1, v)
          + a * (1 - b) * lut(bi + 1, bj, v) + a * b * lut(bi + 1, bj + 1, v);
      }
    }

    /* The measurement. A blocking artefact is a systematic offset applied to a whole
       column of pixels, so averaging DOWN the column reinforces it while averaging the
       block's own texture away. Comparing the jump at a seam against the jump at a
       block centre then isolates the artefact from the image's real content. */
    const colMean = new Float64Array(IW);
    for (let j = 0; j < IW; j++) { let s = 0; for (let i = 0; i < IH; i++) s += B[i][j]; colMean[j] = s / IH; }
    function meanJump(cols) {
      let s = 0, n = 0;
      for (const c of cols) { if (c <= 0 || c >= IW) continue; s += Math.abs(colMean[c] - colMean[c - 1]); n++; }
      return n ? s / n : 0;
    }
    const seamCols = [], insideCols = [];
    for (let bj = 1; bj < nbj; bj++) seamCols.push(bj * M);
    for (let bj = 0; bj < nbj; bj++) insideCols.push(bj * M + Math.floor(M / 2));
    const jSeam = meanJump(seamCols), jIn = meanJump(insideCols);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell = 3.3;

    IP.cap(g, 8, 14, "input scene", VC.accent);
    IP.raster(g, A, 8, 20, cell, { lo: 0, hi: 255 });
    IP.cap(g, 8 + IW * cell + 22, 14, ({
      none: "unchanged", global: "global equalisation", block: "per-block, no blending",
      ahe: "interpolated (AHE)", clahe: "CLAHE, c = " + cf.toFixed(2)
    })[method], VC.a2);
    IP.raster(g, B, 8 + IW * cell + 22, 20, cell, { lo: 0, hi: 255 });
    if (eS.checked && method !== "none" && method !== "global") {
      [8, 8 + IW * cell + 22].forEach(x0 => {
        for (let bj = 1; bj < nbj; bj++) g.append("line").attr("x1", x0 + bj * M * cell).attr("x2", x0 + bj * M * cell)
          .attr("y1", 20).attr("y2", 20 + IH * cell).attr("stroke", VC.violet).attr("stroke-opacity", 0.55).attr("stroke-dasharray", "3 3");
        for (let bi = 1; bi < nbi; bi++) g.append("line").attr("y1", 20 + bi * M * cell).attr("y2", 20 + bi * M * cell)
          .attr("x1", x0).attr("x2", x0 + IW * cell).attr("stroke", VC.violet).attr("stroke-opacity", 0.55).attr("stroke-dasharray", "3 3");
      });
    }

    /* ── right column, upper: the two jump measurements as bars ── */
    const rx = 8 + 2 * (IW * cell + 22) + 8, rw = W - 20 - rx;
    const bh = 120;
    g.append("rect").attr("x", rx).attr("y", 20).attr("width", rw).attr("height", bh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, rx, 14, "mean |Δ| of the column average", VC.ink);
    const mx = Math.max(jSeam, jIn, 1) * 1.15;
    const bs = d3.scaleLinear().domain([0, mx]).range([0, rw - 96]);
    [["across a seam", jSeam, VC.bad], ["inside a block", jIn, VC.good]].forEach((d, i) => {
      g.append("rect").attr("x", rx + 90).attr("y", 40 + i * 34).attr("width", Math.max(1, bs(d[1]))).attr("height", 18)
        .attr("fill", d[2]).attr("fill-opacity", 0.75);
      g.append("text").attr("x", rx + 86).attr("y", 53 + i * 34).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", VC.muted).text(d[0]);
      g.append("text").attr("x", rx + 94 + Math.max(1, bs(d[1]))).attr("y", 53 + i * 34)
        .attr("font-size", 10.5).attr("fill", VC.ink).text(VZ.fmt(d[1], 2));
    });
    g.append("text").attr("x", rx + 8).attr("y", 112).attr("font-size", 10)
      .attr("fill", jSeam > 3 * jIn ? VC.bad : VC.good)
      .text("ratio = " + VZ.fmt(jIn > 0 ? jSeam / jIn : 0, 2) + (jSeam > 3 * jIn ? "  ← a visible seam" : "  ← no seam"));

    /* ── right column, lower: four neighbouring transfer curves ── */
    const cy0 = 168, chh = H - cy0 - 30;
    g.append("rect").attr("x", rx).attr("y", cy0).attr("width", rw).attr("height", chh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, rx, cy0 - 6, "four neighbouring block curves T(v)", VC.ink);
    const tx = d3.scaleLinear().domain([0, 255]).range([rx + 8, rx + rw - 8]);
    const ty = d3.scaleLinear().domain([0, 255]).range([cy0 + chh - 18, cy0 + 8]);
    g.append("line").attr("x1", tx(0)).attr("y1", ty(0)).attr("x2", tx(255)).attr("y2", ty(255))
      .attr("stroke", VC.muted).attr("stroke-dasharray", "2 3");
    const pick = [[0, 0], [0, Math.min(1, nbj - 1)], [Math.min(1, nbi - 1), 0], [Math.min(1, nbi - 1), Math.min(1, nbj - 1)]];
    const cols = [VC.accent, VC.a2, VC.good, VC.violet];
    pick.forEach((p, k) => {
      const T = LUT[p[0]][p[1]];
      let d = ""; for (let v = 0; v < 256; v++) d += (v ? "L" : "M") + tx(v).toFixed(1) + "," + ty(T[v]).toFixed(1);
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", cols[k]).attr("stroke-width", 1.6).attr("stroke-opacity", 0.9);
    });
    /* the measured steepest step of the four, against the clip bound */
    let maxStep = 0;
    pick.forEach(p => { const T = LUT[p[0]][p[1]]; for (let v = 1; v < 256; v++) maxStep = Math.max(maxStep, T[v] - T[v - 1]); });
    const nblk = M * M;
    const bound = method === "clahe" ? (L - 1) * (BETA[0][0] + EXC[0][0] / L) / nblk : null;
    g.append("text").attr("x", rx + 8).attr("y", cy0 + chh - 4).attr("font-size", 10).attr("fill", VC.muted)
      .text("steepest step " + maxStep + (bound !== null ? "   ·   clip bound " + VZ.fmt(bound, 3) : "   ·   unclipped"));

    out.innerHTML =
      `${IW} × ${IH} scene, ${nbi} × ${nbj} blocks of ${M} px, noise sd ${VZ.fmt(255 * nz, 1)} code values · `
      + `mean |Δ| of the column average across a block seam <b>${VZ.fmt(jSeam, 3)}</b> vs at a block centre <b>${VZ.fmt(jIn, 3)}</b> `
      + `(ratio <b>${VZ.fmt(jIn > 0 ? jSeam / jIn : 0, 2)}</b>)<br>`
      + (method === "block"
        ? `Every pixel on one side of a seam used a different lookup table from its neighbour on the other side. That is the whole artefact — nothing about the image changed at the seam, only the curve did.`
        : method === "ahe" || method === "clahe"
          ? `Bilinear blending of the four nearest block curves has removed the discontinuity: the seam ratio is now close to 1, meaning a seam column looks like any other column.`
          : method === "global"
            ? `One curve for the whole image, so there are no seams by construction — and no local adaptation either: the dark foreground and the bright sky are still fighting over the same 256 levels.`
            : `No operator applied; these are the scene's own column-to-column differences, the baseline every other method must be compared against.`)
      + (method === "clahe" ? `<br>clip factor c = ${cf.toFixed(2)} → <span class="keep">β</span> = ${VZ.fmt(BETA[0][0], 1)} counts, `
        + `excess E = ${VZ.fmt(EXC[0][0], 0)} of ${nblk} pixels in the corner block (${VZ.fmt(100 * EXC[0][0] / nblk, 1)}% redistributed), `
        + `so the real-valued slope of T is bounded by <b>${VZ.fmt(bound, 3)}</b>.` : "");
  }

  eM.onchange = draw;
  eB.oninput = () => { eBv.textContent = eB.value; draw(); };
  eC.oninput = () => { eCv.textContent = (+eC.value).toFixed(2); draw(); };
  eN.oninput = () => { eNv.textContent = (+eN.value).toFixed(3); draw(); };
  eS.onchange = draw;
  draw();
})();

/* ───────── 4 · a kernel editor: correlation and convolution, side by side ───────── */
(function () {
  const svg = d3.select("#conv-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440;
  const ePre = document.getElementById("cv-pre");
  const eNorm = document.getElementById("cv-norm");
  const eShow = document.getElementById("cv-show");
  const cells = document.getElementById("cv-cells");
  const out = document.getElementById("conv-readout");

  const PRESETS = {
    impulse: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    box: [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    gauss: [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    sharp: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    sobelx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    lap: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    emboss: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    shift: [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
  };

  /* build the nine number inputs once */
  const inputs = [];
  cells.innerHTML = "";
  const lab = document.createElement("span");
  lab.style.cssText = "font-size:12.5px;color:var(--muted);margin-right:4px";
  lab.textContent = "kernel h(k, l) —";
  cells.appendChild(lab);
  for (let i = 0; i < 3; i++) {
    const row = document.createElement("span");
    row.style.cssText = "display:inline-flex;gap:4px;margin-right:10px";
    for (let j = 0; j < 3; j++) {
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "any"; inp.value = "0";
      inp.setAttribute("aria-label", `kernel entry row ${i + 1} column ${j + 1}`);
      inp.oninput = draw;
      inputs.push(inp); row.appendChild(inp);
    }
    cells.appendChild(row);
  }
  function setKernel(K) { for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) inputs[3 * i + j].value = K[i][j]; }
  function getKernel() {
    const K = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const v = parseFloat(inputs[3 * i + j].value);
      K[i][j] = isFinite(v) ? v : 0;
    }
    return K;
  }
  ePre.onchange = () => { setKernel(PRESETS[ePre.value]); draw(); };
  eNorm.onchange = draw; eShow.onchange = draw;

  function draw() {
    let K = getKernel();
    const s = K.flat().reduce((a, b) => a + b, 0);
    if (eNorm.checked && Math.abs(s) > 1e-12) K = K.map(r => r.map(v => v / s));
    const sum = K.flat().reduce((a, b) => a + b, 0);
    const sym = K.every((r, i) => r.every((v, j) => Math.abs(v - K[2 - i][2 - j]) < 1e-12));

    /* an impulse image and the running patch */
    const IMP = IP.zeros2(7, 7); IMP[3][3] = 1;
    const impC = IP.corr2(IMP, K, "zero"), impV = IP.conv2(IMP, K, "zero");
    const patC = IP.corr2(IP.PATCH, K, "clamp"), patV = IP.conv2(IP.PATCH, K, "clamp");

    let maxd = 0;
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) maxd = Math.max(maxd, Math.abs(patC[i][j] - patV[i][j]));

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const show = eShow.value;
    const cols = [{ t: "input", C: null }];
    if (show !== "conv") cols.push({ t: "correlation   f ⊗ h", C: "corr" });
    if (show !== "corr") cols.push({ t: "convolution   f ∗ h", C: "conv" });
    const colW = 236, x0 = 8;

    /* the kernel, printed once */
    const kx = 8 + cols.length * colW + 8;
    if (kx < W - 120) {
      IP.cap(g, kx, 14, "h as used", VC.ink);
      IP.numText(g, K.map(r => r.map(v => (Math.abs(v) < 1e-9 ? "0" : (Math.abs(v) >= 100 || (v % 1 === 0) ? v.toFixed(0) : v.toFixed(3)))).map(t => t.padStart(7)).join("")),
        kx, 32, { size: 10.5, lead: 14 });
      IP.numText(g, ["", "sum = " + VZ.fmt(sum, 4), sym ? "180°-symmetric" : "NOT symmetric"], kx, 78,
        { size: 10.5, lead: 14, fill: sym ? VC.good : VC.a2 });
      IP.numText(g, ["so ⊗ and ∗", sym ? "agree exactly" : "differ by a 180° turn"], kx, 122,
        { size: 10.5, lead: 14, fill: VC.muted });
    }

    /* top band: the impulse test */
    const impCell = 15, imgY = 34;
    cols.forEach((c, ci) => {
      const x = x0 + ci * colW;
      const A = c.C === null ? IMP : (c.C === "corr" ? impC : impV);
      const m = Math.max(IP.absMax(A), 1e-9);
      IP.raster(g, A, x, imgY, impCell, { signed: !sym || sum < 0.999, lo: -m, hi: m, grid: true });
      g.append("text").attr("x", x).attr("y", imgY - 8).attr("font-size", 10.5).attr("fill", ci === 0 ? VC.muted : VC.ink)
        .text(ci === 0 ? "an impulse" : c.t);
    });
    /* bottom band: the running patch */
    const patCell = 20, patY = 34 + 7 * impCell + 46;
    cols.forEach((c, ci) => {
      const x = x0 + ci * colW;
      const A = c.C === null ? IP.PATCH : (c.C === "corr" ? patC : patV);
      const signed = Math.abs(sum) < 1e-9;
      const m = Math.max(IP.absMax(A), 1e-9);
      g.append("text").attr("x", x).attr("y", patY - 8).attr("font-size", 10.5).attr("fill", VC.muted)
        .text(ci === 0 ? "the running 8×8 patch" : "the same, filtered");
      IP.raster(g, A, x, patY, patCell, signed ? { signed: true, lo: -m, hi: m, grid: true } : { lo: 0, hi: 255, grid: true });
      g.append("text").attr("x", x).attr("y", patY + 8 * patCell + 15).attr("font-size", 10)
        .attr("fill", VC.a2).attr("font-family", "SF Mono, Menlo, monospace")
        .text("centre pixel (3,3): " + VZ.fmt(A[3][3], 3));
    });

    out.innerHTML =
      `kernel sum <b>${VZ.fmt(sum, 4)}</b> — ${Math.abs(sum - 1) < 1e-9 ? "unit DC gain, so a flat region keeps its value" :
        Math.abs(sum) < 1e-9 ? "zero DC gain, so a flat region maps to zero: this is a <b>derivative-like</b> kernel" :
          "DC gain " + VZ.fmt(sum, 3) + ", so a flat region is scaled by that factor"} · `
      + `180°-symmetric: <b>${sym ? "yes" : "no"}</b><br>`
      + (sym
        ? `Because h equals its own 180° rotation, <b>f ⊗ h = f ∗ h exactly</b> — the largest disagreement over the patch is ${VZ.fmt(maxd, 12)}. The distinction is invisible for this kernel, and for every box, tent, binomial and Gaussian on this page.`
        : `Because h is asymmetric, the two operators disagree: the largest difference over the patch is <b>${VZ.fmt(maxd, 3)}</b> code values. Look at the impulse row — convolution has put your kernel back exactly, correlation has put it back rotated by 180°.`);
  }

  setKernel(PRESETS[ePre.value]);
  draw();
})();

/* ───────── 5 · the six border modes, on a signal and on a corner ───────── */
(function () {
  const svg = d3.select("#pad-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440;
  const eM = document.getElementById("pd-m"), eK = document.getElementById("pd-k");
  const eOnes = document.getElementById("pd-ones");
  const out = document.getElementById("pad-readout");

  const SIG = [0.82, 0.66, 0.30, 0.44, 0.71, 0.58, 0.22, 0.35, 0.62, 0.90];

  function kern() {
    switch (eK.value) {
      case "box3": return { h: IP.box(3), name: "box 3", dc: 1 };
      case "box7": return { h: IP.box(7), name: "box 7", dc: 1 };
      case "gauss2": return { h: IP.gauss1(2, 3), name: "Gaussian σ = 2 (13 taps)", dc: 1 };
      case "gauss4": return { h: IP.gauss1(4, 3), name: "Gaussian σ = 4 (25 taps)", dc: 1 };
      case "dx": return { h: Float64Array.from([-0.5, 0, 0.5]), name: "central difference", dc: 0 };
    }
  }
  /* the extension actually used, for drawing */
  function extend(f, i, mode, cval) {
    if (mode === "norm") return i >= 0 && i < f.length ? f[i] : 0;
    return IP.at1(f, i, mode === "const" ? "const" : mode, cval);
  }

  function draw() {
    const mode = eM.value, K = kern(), R = (K.h.length - 1) >> 1;
    const useOnes = eOnes.checked;
    const sig = useOnes ? SIG.map(() => 1) : SIG.slice();
    const mean = sig.reduce((a, b) => a + b, 0) / sig.length;

    /* filter the signal in the chosen mode, plus the normalised-zero variant */
    const n = sig.length, g = new Float64Array(n), alpha = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0, a = 0;
      for (let k = -R; k <= R; k++) {
        const idx = i + k, inside = idx >= 0 && idx < n;
        const w = K.h[k + R];
        if (mode === "norm") { if (inside) { s += sig[idx] * w; a += w; } }
        else { s += extend(sig, idx, mode, mean) * w; a += w; }
      }
      alpha[i] = a;
      g[i] = mode === "norm" ? (a > 1e-12 ? s / a : 0) : s;
    }

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const gg = f.g;

    /* ── top: the signal and its extension ── */
    const pad = Math.min(R, 8);
    const sx = d3.scaleLinear().domain([-pad - 0.5, n - 0.5 + pad]).range([40, W - 30]);
    const sy = d3.scaleLinear().domain([-0.35, 1.35]).range([150, 30]);
    gg.append("rect").attr("x", 34).attr("y", 24).attr("width", W - 58).attr("height", 132)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    [0, 0.5, 1].forEach(v => gg.append("line").attr("x1", 36).attr("x2", W - 26).attr("y1", sy(v)).attr("y2", sy(v))
      .attr("stroke", VC.grid));
    /* the real samples */
    for (let i = -pad; i < n + pad; i++) {
      const inside = i >= 0 && i < n;
      const v = mode === "norm" && !inside ? null : extend(sig, i, mode, mean);
      if (v === null) continue;
      gg.append("circle").attr("cx", sx(i)).attr("cy", sy(v)).attr("r", inside ? 4 : 3)
        .attr("fill", inside ? VC.accent : "none")
        .attr("stroke", inside ? VC.bg : VC.a2).attr("stroke-width", inside ? 1 : 1.4);
      gg.append("line").attr("x1", sx(i)).attr("x2", sx(i)).attr("y1", sy(0)).attr("y2", sy(v))
        .attr("stroke", inside ? VC.accent : VC.a2).attr("stroke-opacity", inside ? 0.55 : 0.3);
    }
    gg.append("line").attr("x1", sx(-0.5)).attr("x2", sx(-0.5)).attr("y1", 28).attr("y2", 152)
      .attr("stroke", VC.violet).attr("stroke-dasharray", "3 3");
    gg.append("line").attr("x1", sx(n - 0.5)).attr("x2", sx(n - 0.5)).attr("y1", 28).attr("y2", 152)
      .attr("stroke", VC.violet).attr("stroke-dasharray", "3 3");
    IP.cap(gg, 36, 20, "the signal (filled) and the continuation the border mode invents (hollow) — mode: "
      + eM.options[eM.selectedIndex].text, VC.ink);
    /* the filtered result */
    let d = "";
    for (let i = 0; i < n; i++) d += (i ? "L" : "M") + sx(i).toFixed(1) + "," + sy(g[i]).toFixed(1);
    gg.append("path").attr("d", d).attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    for (let i = 0; i < n; i++) {
      const touched = i < R || i >= n - R;
      gg.append("circle").attr("cx", sx(i)).attr("cy", sy(g[i])).attr("r", 3.4)
        .attr("fill", touched ? VC.bad : VC.good);
    }
    VZ.legend(gg, [{ color: VC.accent, label: "input" }, { color: VC.a2, label: "invented by padding" },
    { color: VC.good, label: "filtered" }, { color: VC.bad, label: "filtered, but used padding" }],
      W - 200, 40, { vertical: true, gap: 14 });

    /* ── bottom left: a 2D image filtered in this mode ── */
    const IMG = useOnes ? IP.map2(IP.scene(56, 40, {}), () => 1) : IP.scene(56, 40, {});
    const mode2 = mode === "norm" ? "zero" : (mode === "const" ? "const" : mode);
    const cval = mode === "const" ? IP.sum2(IMG) / (IMG.length * IMG[0].length) : 0;
    let F = IP.sepH(IMG, K.h, mode2, cval); F = IP.sepV(F, K.h, mode2, cval);
    if (mode === "norm") {
      const M = IP.map2(IMG, () => 1);
      let A = IP.sepH(M, K.h, "zero", 0); A = IP.sepV(A, K.h, "zero", 0);
      F = F.map((r, i) => Float64Array.from(r, (v, j) => A[i][j] > 1e-12 ? v / A[i][j] : 0));
    }
    const cell = 4.2;
    IP.cap(gg, 34, 188, "a 56 × 40 image, filtered with " + K.name + " in this mode", VC.ink);
    IP.raster(gg, F, 34, 196, cell, K.dc === 0 ? { signed: true } : { lo: 0, hi: 1 });
    /* outline the band that used padding */
    gg.append("rect").attr("x", 34 + R * cell).attr("y", 196 + R * cell)
      .attr("width", (56 - 2 * R) * cell).attr("height", (40 - 2 * R) * cell)
      .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-dasharray", "3 2");
    IP.cap(gg, 34, 196 + 40 * cell + 16, "dashed box = the region that needed no padding at all", VC.violet);

    /* ── bottom right: the constant test along the top row ── */
    const px = 320, py = 200, pw = W - px - 24, ph = 180;
    gg.append("rect").attr("x", px).attr("y", py).attr("width", pw).attr("height", ph)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(gg, px, py - 6, "the constant test: an all-ones image, filtered, along the top row", VC.ink);
    const ONE = IP.map2(IP.scene(56, 40, {}), () => 1);
    const results = {};
    ["zero", "const", "clamp", "wrap", "mirror", "symm", "norm"].forEach(m => {
      const m2 = m === "norm" ? "zero" : m;
      let A = IP.sepH(ONE, K.h, m2, m === "const" ? 1 : 0); A = IP.sepV(A, K.h, m2, m === "const" ? 1 : 0);
      if (m === "norm") {
        let B = IP.sepH(IP.map2(ONE, () => 1), K.h, "zero", 0); B = IP.sepV(B, K.h, "zero", 0);
        A = A.map((r, i) => Float64Array.from(r, (v, j) => B[i][j] > 1e-12 ? v / B[i][j] : 0));
      }
      results[m] = A;
    });
    const cur = results[mode];
    const lox = d3.scaleLinear().domain([0, 55]).range([px + 8, px + pw - 8]);
    const loy = d3.scaleLinear().domain([-0.05, 1.15]).range([py + ph - 22, py + 10]);
    gg.append("line").attr("x1", lox(0)).attr("x2", lox(55)).attr("y1", loy(1)).attr("y2", loy(1))
      .attr("stroke", VC.good).attr("stroke-dasharray", "3 3");
    gg.append("text").attr("x", px + pw - 10).attr("y", loy(1) - 4).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", VC.good).text("the correct answer, 1");
    ["zero", "clamp", "mirror", "norm"].forEach((m, i) => {
      const A = results[m];
      let d2 = "";
      for (let j = 0; j < 56; j++) d2 += (j ? "L" : "M") + lox(j).toFixed(1) + "," + loy(A[0][j]).toFixed(1);
      gg.append("path").attr("d", d2).attr("fill", "none")
        .attr("stroke", [VC.bad, VC.a2, VC.accent, VC.violet][i])
        .attr("stroke-width", m === mode ? 2.6 : 1.2).attr("stroke-opacity", m === mode ? 1 : 0.55);
    });
    VZ.legend(gg, [{ color: VC.bad, label: "zero" }, { color: VC.a2, label: "clamp" },
    { color: VC.accent, label: "mirror" }, { color: VC.violet, label: "normalised zero" }],
      px + 10, py + 20, { vertical: true, gap: 13 });

    const corner = m => VZ.fmt(results[m][0][0], 4);
    out.innerHTML =
      `kernel: ${K.name}, radius ${R}. On an all-ones image a unit-gain filter must return exactly 1 everywhere.<br>`
      + `corner value by mode — zero <b>${corner("zero")}</b> · constant <b>${corner("const")}</b> · `
      + `clamp <b>${corner("clamp")}</b> · wrap <b>${corner("wrap")}</b> · mirror <b>${corner("mirror")}</b> · `
      + `symmetric <b>${corner("symm")}</b> · normalised zero <b>${corner("norm")}</b>`
      + (K.dc === 0 ? `<br>(this kernel sums to zero, so the correct answer on a constant image is 0, not 1 — every mode except zero-padding returns it.)`
        : `<br>Only <b>zero</b> padding gets it wrong here: it returns <b>${corner("zero")}</b> at the corner instead of 1, because ${VZ.fmt(100 * (1 - results.zero[0][0]), 1)}% of the kernel's weight fell outside the image and contributed zeros. `
        + `(For the smallest case, a 3-tap box, the corner keeps 4 of its 9 taps and returns exactly 4/9 = 0.4444; the wider the kernel, the deeper the dark band.) `
        + `Constant padding looks exact only because the padding constant is the image mean, and on a <i>constant</i> image that is the right answer — on any real image it is not.`);
  }

  eM.onchange = draw; eK.onchange = draw; eOnes.onchange = draw;
  draw();
})();

/* ───────── 6 · box, tent, binomial, Gaussian: shape, moments, response ───────── */
(function () {
  const svg = d3.select("#smooth-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440;
  const eS = document.getElementById("sm-s"), eSv = document.getElementById("sm-sv");
  const eT = document.getElementById("sm-t"), eTv = document.getElementById("sm-tv");
  const eP = document.getElementById("sm-p"), ePv = document.getElementById("sm-pv");
  const eL = document.getElementById("sm-log");
  const out = document.getElementById("smooth-readout");

  function conv(a, b) {                       // full 1D convolution of two odd kernels
    const n = a.length + b.length - 1, c = new Float64Array(n);
    for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) c[i + j] += a[i] * b[j];
    return c;
  }

  function draw() {
    const sig = +eS.value, tr = +eT.value, passes = +eP.value;
    const R = Math.max(1, Math.ceil(tr * sig));
    const K = 2 * R + 1;
    /* four kernels, all with the same support so the comparison is fair */
    const gaussFull = (() => {              /* untruncated reference, for the mass loss */
      let s = 0; for (let k = -400; k <= 400; k++) s += Math.exp(-(k * k) / (2 * sig * sig)); return s;
    })();
    const gaussKept = (() => { let s = 0; for (let k = -R; k <= R; k++) s += Math.exp(-(k * k) / (2 * sig * sig)); return s; })();

    let boxK = IP.box(K);
    for (let p = 1; p < passes; p++) boxK = conv(boxK, IP.box(K));
    const list = [
      { name: "box " + K + (passes > 1 ? " × " + passes + " passes" : ""), h: boxK, col: VC.bad, target: passes * (K * K - 1) / 12 },
      { name: "tent (width " + K + ")", h: IP.tent(K), col: VC.a2, target: null },
      { name: "binomial n = " + (K - 1), h: IP.binomial(K - 1), col: VC.violet, target: (K - 1) / 4 },
      { name: "Gaussian σ = " + sig.toFixed(2) + ", R = " + R, h: IP.gauss1(sig, tr), col: VC.good, target: sig * sig }
    ];

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── left: stem plots ── */
    const px = 34, pw = 230, py = 26, phh = 84;
    const maxR = Math.max(...list.map(d => (d.h.length - 1) >> 1));
    list.forEach((d, i) => {
      const y0 = py + i * (phh + 12);
      g.append("rect").attr("x", px).attr("y", y0).attr("width", pw).attr("height", phh)
        .attr("fill", VC.panel2).attr("stroke", VC.line);
      const rr = (d.h.length - 1) >> 1;
      const xs = d3.scaleLinear().domain([-maxR - 0.6, maxR + 0.6]).range([px + 6, px + pw - 6]);
      const top = Math.max(...d.h);
      const ys = d3.scaleLinear().domain([0, top * 1.12]).range([y0 + phh - 12, y0 + 8]);
      g.append("line").attr("x1", px + 6).attr("x2", px + pw - 6).attr("y1", ys(0)).attr("y2", ys(0)).attr("stroke", VC.line);
      for (let k = -rr; k <= rr; k++) {
        g.append("line").attr("x1", xs(k)).attr("x2", xs(k)).attr("y1", ys(0)).attr("y2", ys(d.h[k + rr]))
          .attr("stroke", d.col).attr("stroke-width", 1.6);
        g.append("circle").attr("cx", xs(k)).attr("cy", ys(d.h[k + rr])).attr("r", 2.4).attr("fill", d.col);
      }
      g.append("text").attr("x", px + 6).attr("y", y0 + 12).attr("font-size", 10)
        .attr("fill", d.col).text(d.name + "  ·  " + d.h.length + " taps");
    });
    IP.cap(g, px, 20, "the kernels, same support", VC.ink);

    /* ── middle: the measured table ── */
    const tx = px + pw + 22;
    const rows = ["kernel               ∑h      ∑k·h     ∑k²·h    target   even    odd"];
    list.forEach(d => {
      const m = IP.moments(d.h);
      rows.push(
        (d.name.split(" ×")[0].split(",")[0]).padEnd(20).slice(0, 20)
        + VZ.fmt(m.m0, 4).padStart(7) + VZ.fmt(m.m1, 3).padStart(9)
        + VZ.fmt(m.m2, 3).padStart(9) + (d.target === null ? "    —  " : VZ.fmt(d.target, 3).padStart(9))
        + VZ.fmt(m.even, 3).padStart(8) + VZ.fmt(m.odd, 3).padStart(8));
    });
    g.append("rect").attr("x", tx - 6).attr("y", 26).attr("width", W - tx - 4).attr("height", 108)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.numText(g, rows, tx, 42, { size: 9.2, lead: 13 });
    IP.cap(g, tx - 6, 20, "measured moments (§12 reads these)", VC.ink);

    /* ── right: frequency responses ── */
    const fy = 150, fh = H - fy - 26, fx = tx - 6, fw = W - tx - 4;
    g.append("rect").attr("x", fx).attr("y", fy).attr("width", fw).attr("height", fh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, fx, fy - 6, "frequency response H(ω), 0 → the Nyquist frequency π", VC.ink);
    const logY = eL.checked;
    const wx = d3.scaleLinear().domain([0, Math.PI]).range([fx + 30, fx + fw - 10]);
    const wy = logY ? d3.scaleLog().domain([1e-4, 1.2]).range([fy + fh - 22, fy + 10]).clamp(true)
      : d3.scaleLinear().domain([-0.4, 1.1]).range([fy + fh - 22, fy + 10]);
    if (!logY) {
      g.append("line").attr("x1", wx(0)).attr("x2", wx(Math.PI)).attr("y1", wy(0)).attr("y2", wy(0))
        .attr("stroke", VC.muted).attr("stroke-opacity", 0.8);
      g.append("text").attr("x", fx + 26).attr("y", wy(0) - 3).attr("text-anchor", "end")
        .attr("font-size", 9).attr("fill", VC.muted).text("0");
    }
    g.append("line").attr("x1", wx(0)).attr("x2", wx(Math.PI)).attr("y1", wy(logY ? 1 : 1)).attr("y2", wy(1))
      .attr("stroke", VC.grid);
    g.append("text").attr("x", fx + 26).attr("y", wy(1) + 3).attr("text-anchor", "end")
      .attr("font-size", 9).attr("fill", VC.muted).text("1");
    const halfPts = [];
    list.forEach(d => {
      let dd = "", prev = null, half = null;
      for (let i = 0; i <= 300; i++) {
        const w = Math.PI * i / 300;
        const re = IP.freqResp(d.h, w).re;
        const y = logY ? Math.max(1e-4, Math.abs(re)) : re;
        dd += (i ? "L" : "M") + wx(w).toFixed(1) + "," + wy(y).toFixed(1);
        if (prev !== null && half === null && prev >= 0.5 && re < 0.5) half = w;
        prev = re;
      }
      g.append("path").attr("d", dd).attr("fill", "none").attr("stroke", d.col).attr("stroke-width", 1.8);
      if (half !== null) {
        g.append("circle").attr("cx", wx(half)).attr("cy", wy(logY ? 0.5 : 0.5)).attr("r", 3).attr("fill", d.col);
        halfPts.push(d.name.split(" ")[0] + " " + VZ.fmt(half / Math.PI, 3) + "π");
      }
    });
    [0, 0.25, 0.5, 0.75, 1].forEach(t => g.append("text").attr("x", wx(t * Math.PI)).attr("y", fy + fh - 6)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", VC.muted).text(t === 0 ? "0" : t + "π"));

    /* the honest numbers */
    const gm = IP.moments(list[3].h);
    const bm = IP.moments(list[0].h);
    let boxMin = Infinity, boxMinW = 0;
    for (let i = 0; i <= 600; i++) { const w = Math.PI * i / 600; const v = IP.freqResp(list[0].h, w).re; if (v < boxMin) { boxMin = v; boxMinW = w; } }
    out.innerHTML =
      `Gaussian <span class="keep">σ</span> = ${sig.toFixed(2)}, truncated at ${tr.toFixed(2)}<span class="keep">σ</span> → R = ${R}, ${2 * R + 1} taps · `
      + `mass kept <b>${VZ.fmt(100 * gaussKept / gaussFull, 4)}%</b>, discarded ${(1 - gaussKept / gaussFull).toExponential(2)}<br>`
      + `its measured variance is <b>${VZ.fmt(gm.m2, 5)}</b> against the nominal <span class="keep">σ</span>² = ${VZ.fmt(sig * sig, 4)} — `
      + `${Math.abs(gm.m2 - sig * sig) / (sig * sig) < 0.01 ? "within 1%, so the label is honest" : VZ.fmt(100 * (1 - gm.m2 / (sig * sig)), 2) + "% short: the kernel is <b>narrower than its label</b>"}<br>`
      + `the box kernel's response reaches <b>${VZ.fmt(boxMin, 5)}</b> at <span class="keep">ω</span> = ${VZ.fmt(boxMinW / Math.PI, 3)}π`
      + (boxMin < -1e-6 ? " — <b>negative</b>, so that band of the image comes out inverted. The Gaussian's never does."
        : " — non-negative over the whole band.")
      + (halfPts.length ? `<br>half-response frequency: ${halfPts.join("  ·  ")}` : "");
  }

  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(2); draw(); };
  eT.oninput = () => { eTv.textContent = (+eT.value).toFixed(1); draw(); };
  eP.oninput = () => { ePv.textContent = eP.value; draw(); };
  eL.onchange = draw;
  draw();
})();

/* ───────── 7 · separability: two 1D passes, and the multiply count ───────── */
(function () {
  const svg = d3.select("#sep-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430, IW = 72, IH = 56;
  const eK = document.getElementById("sp-k");
  const eS = document.getElementById("sp-s"), eSv = document.getElementById("sp-sv");
  const eSt = document.getElementById("sp-st");
  const out = document.getElementById("sep-readout");

  /* a tiny SVD for small square matrices, via the eigenvalues of KᵀK by Jacobi */
  function singularValues(K) {
    const n = K[0].length;
    /* A = KᵀK, symmetric n×n; cyclic Jacobi */
    const A = [];
    for (let i = 0; i < n; i++) { A.push(new Float64Array(n)); for (let j = 0; j < n; j++) { let s = 0; for (let r = 0; r < K.length; r++) s += K[r][i] * K[r][j]; A[i][j] = s; } }
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-30) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
        const c = Math.cos(th), s = Math.sin(th);
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp + s * akq; A[k][q] = -s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk + s * aqk; A[q][k] = -s * apk + c * aqk; }
      }
    }
    const ev = []; for (let i = 0; i < n; i++) ev.push(Math.sqrt(Math.max(0, A[i][i])));
    return ev.sort((a, b) => b - a);
  }

  function build(kind, s) {
    if (kind === "gauss") { const h = IP.gauss1(s, 3); return { K: IP.outer(h, h), h, v: h, sep: true, name: "Gaussian σ = " + s.toFixed(2) }; }
    if (kind === "box") { const K1 = 2 * Math.round(s) + 1, h = IP.box(K1); return { K: IP.outer(h, h), h, v: h, sep: true, name: "box " + K1 }; }
    if (kind === "binom") { const n = 2 * Math.round(s), h = IP.binomial(n); return { K: IP.outer(h, h), h, v: h, sep: true, name: "binomial n = " + n }; }
    if (kind === "sobel") {
      const d = Float64Array.from([-1, 0, 1]), t = Float64Array.from([1, 2, 1]);
      return { K: IP.outer(t, d), h: d, v: t, sep: true, name: "Sobel x = [1,2,1]ᵀ · [−1,0,1]" };
    }
    if (kind === "log") {                                /* rank 2: G''⊗G + G⊗G'' */
      const g = IP.gauss1(s, 3), R = (g.length - 1) >> 1;
      const g2 = Float64Array.from(g, (v, i) => { const k = i - R; return v * (k * k - s * s) / (s * s * s * s); });
      const A = IP.outer(g2, g), B = IP.outer(g, g2);
      const K = A.map((r, i) => r.map((v, j) => v + B[i][j]));
      return { K, h: g, v: g, sep: false, rank2: { a: [g2, g], b: [g, g2] }, name: "Laplacian of Gaussian, σ = " + s.toFixed(2) };
    }
    /* a rotated anisotropic Gaussian — genuinely not separable in x and y */
    const R = Math.max(2, Math.ceil(3 * s)), th = Math.PI / 5, sa = s, sb = s / 2.6;
    const K = [];
    let tot = 0;
    for (let i = -R; i <= R; i++) {
      const row = new Float64Array(2 * R + 1);
      for (let j = -R; j <= R; j++) {
        const u = j * Math.cos(th) + i * Math.sin(th), v = -j * Math.sin(th) + i * Math.cos(th);
        row[j + R] = Math.exp(-(u * u) / (2 * sa * sa) - (v * v) / (2 * sb * sb)); tot += row[j + R];
      }
      K.push(row);
    }
    return { K: K.map(r => Float64Array.from(r, v => v / tot)), sep: false, name: "rotated anisotropic Gaussian" };
  }

  function draw() {
    const s = +eS.value, kind = eK.value, stage = eSt.value;
    const B = build(kind, s);
    const A0 = IP.scene(IW, IH, {});
    const K2d = B.K, Kn = K2d.length;
    const twoD = IP.corr2(A0, K2d, "mirror");

    let mid = null, sepR = null;
    if (B.sep) { mid = IP.sepH(A0, B.h, "mirror"); sepR = IP.sepV(mid, B.v, "mirror"); }
    else if (B.rank2) {
      const a = IP.sepV(IP.sepH(A0, B.rank2.a[1], "mirror"), B.rank2.a[0], "mirror");
      const b = IP.sepV(IP.sepH(A0, B.rank2.b[1], "mirror"), B.rank2.b[0], "mirror");
      mid = a; sepR = a.map((r, i) => Float64Array.from(r, (v, j) => v + b[i][j]));
    } else {
      /* not separable: use the BEST rank-1 approximation (one step of power iteration
         to convergence), so the "difference" stage shows a real, non-zero error */
      const n = Kn; let v = new Float64Array(n).fill(1 / Math.sqrt(n)), u = new Float64Array(n);
      for (let it = 0; it < 200; it++) {
        for (let i = 0; i < n; i++) { let t = 0; for (let j = 0; j < n; j++) t += K2d[i][j] * v[j]; u[i] = t; }
        let nu = Math.hypot(...u) || 1; for (let i = 0; i < n; i++) u[i] /= nu;
        for (let j = 0; j < n; j++) { let t = 0; for (let i = 0; i < n; i++) t += K2d[i][j] * u[i]; v[j] = t; }
        let nv = Math.hypot(...v) || 1; for (let j = 0; j < n; j++) v[j] /= nv;
      }
      let s1 = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s1 += u[i] * K2d[i][j] * v[j];
      const hh = Float64Array.from(v, x => x * s1), vv = Float64Array.from(u);
      mid = IP.sepH(A0, hh, "mirror"); sepR = IP.sepV(mid, vv, "mirror");
    }

    let maxd = 0;
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) maxd = Math.max(maxd, Math.abs(sepR[i][j] - twoD[i][j]));

    const sv = singularValues(K2d.map(r => Array.from(r)));
    /* the Jacobi solver is only good to about 1e-8 relative, so the rank tolerance has
       to be looser than that or a genuinely rank-1 kernel reads as rank 5 */
    const rank = B.sep ? 1 : (B.rank2 ? 2 : sv.filter(v => v > sv[0] * 1e-6).length);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── left: the 2D kernel and its singular values ── */
    const kc = Math.min(13, 150 / Kn);
    IP.cap(g, 14, 16, "the 2D kernel  (" + Kn + " × " + Kn + ")", VC.ink);
    const km = IP.absMax(K2d);
    IP.raster(g, K2d, 14, 24, kc, { signed: true, lo: -km, hi: km });
    IP.cap(g, 14, 24 + Kn * kc + 16, B.name, VC.muted);
    IP.numText(g, ["singular values:"].concat(sv.slice(0, 4).map((v, i) =>
      "  σ" + i + " = " + (v > sv[0] * 1e-6 ? v.toExponential(3) : "0 (below solver noise)"))),
      14, 24 + Kn * kc + 34, { size: 10, lead: 13 });
    g.append("text").attr("x", 14).attr("y", 24 + Kn * kc + 106).attr("font-size", 10.5)
      .attr("fill", rank === 1 ? VC.good : VC.a2)
      .text("numerical rank " + rank + (rank === 1 ? " → separable" : rank === 2 ? " → a sum of 2 separable filters" : " → not separable"));

    /* ── middle: the image at the chosen stage ── */
    const cell = 3.6, mx = 210;
    const showA = stage === "0" ? A0 : stage === "1" ? mid : stage === "2" ? sepR
      : sepR.map((r, i) => Float64Array.from(r, (v, j) => v - twoD[i][j]));
    IP.cap(g, mx, 16, ({
      "0": "input", "1": "after the horizontal 1D pass ONLY",
      "2": "after both passes", "d": "separable result − direct 2D result"
    })[stage], VC.ink);
    if (stage === "d") {
      const m = Math.max(1e-18, IP.absMax(showA));
      IP.raster(g, showA, mx, 24, cell, { signed: true, lo: -m, hi: m });
      IP.cap(g, mx, 24 + IH * cell + 16, "displayed at full contrast: the peak is " + m.toExponential(2), VC.muted);
    } else {
      const signed = Math.abs(IP.sum2(K2d)) < 1e-9;
      const m = IP.absMax(showA);
      IP.raster(g, showA, mx, 24, cell, signed ? { signed: true, lo: -m, hi: m } : { lo: 0, hi: 1 });
    }
    if (stage === "1") IP.cap(g, mx, 24 + IH * cell + 16,
      "blurred horizontally, still sharp vertically — the vertical pass has not run yet", VC.a2);

    /* ── right: the cost bars ── */
    const rx = mx + IW * cell + 26, rw = W - rx - 16;
    const N = IW * IH;
    const c2d = Kn * Kn, csep = B.rank2 ? 4 * Kn : 2 * Kn;
    g.append("rect").attr("x", rx).attr("y", 24).attr("width", rw).attr("height", 150)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, rx, 16, "multiply–accumulates per pixel", VC.ink);
    const cs = d3.scaleLinear().domain([0, Math.max(c2d, csep) * 1.1]).range([0, rw - 110]);
    [["direct 2D", c2d, VC.bad], [B.rank2 ? "2 × separable" : "separable", csep, VC.good]].forEach((d, i) => {
      g.append("rect").attr("x", rx + 92).attr("y", 46 + i * 40).attr("width", Math.max(2, cs(d[1]))).attr("height", 22)
        .attr("fill", d[2]).attr("fill-opacity", 0.75);
      g.append("text").attr("x", rx + 88).attr("y", 61 + i * 40).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", VC.muted).text(d[0]);
      g.append("text").attr("x", rx + 98 + Math.max(2, cs(d[1]))).attr("y", 61 + i * 40)
        .attr("font-size", 11).attr("fill", VC.ink).text(d[1]);
    });
    IP.numText(g, [
      "speed-up          " + VZ.fmt(c2d / csep, 3) + "×",
      "on this " + IW + " × " + IH + " image:",
      "  2D        " + (c2d * N).toLocaleString() + " mults",
      "  separable " + (csep * N).toLocaleString() + " mults"
    ], rx + 10, 116, { size: 10, lead: 13 });

    out.innerHTML =
      `kernel ${Kn} × ${Kn}, numerical rank <b>${rank}</b> · `
      + `singular values ${sv.slice(0, 3).map(v => v > sv[0] * 1e-6 ? v.toExponential(2) : "0").join(", ")}<br>`
      + (rank === 1
        ? `Separable. Two 1D passes cost <b>${csep}</b> multiplies per pixel against <b>${c2d}</b> for the direct 2D convolution, a <b>${VZ.fmt(c2d / csep, 2)}×</b> saving, and the maximum absolute disagreement between the two results over the whole image is <b>${maxd.toExponential(3)}</b> — floating-point noise, not approximation.`
        : rank === 2
          ? `Rank 2: not separable, but expressible as a <b>sum of two</b> separable filters. That costs ${csep} multiplies per pixel instead of ${c2d}, still a ${VZ.fmt(c2d / csep, 2)}× saving, and the result agrees with the direct 2D convolution to <b>${maxd.toExponential(3)}</b>.`
          : `Rank ${rank}. This kernel is <b>not</b> separable — a rotated anisotropic Gaussian carries a cross term <code>xy</code> in its exponent and no outer product produces one. What the two passes above actually apply is the <b>best rank-1 approximation</b>, and the "difference" stage measures the price: a maximum absolute error of <b>${maxd.toExponential(3)}</b> over the image, which is a real approximation error and not floating-point noise. The direct 2D convolution, at ${c2d} multiplies per pixel, is the only exact option.`);
  }

  eK.onchange = draw;
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(2); draw(); };
  eSt.onchange = draw;
  draw();
})();

/* ───────── 8 · the moment conditions, measured ───────── */
(function () {
  const svg = d3.select("#moment-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430;
  const eK = document.getElementById("mo-k");
  const eS = document.getElementById("mo-s"), eSv = document.getElementById("mo-sv");
  const eC = document.getElementById("mo-c");
  const out = document.getElementById("moment-readout");

  const F = a => Float64Array.from(a);
  function kern() {
    const s = +eS.value;
    switch (eK.value) {
      case "box3": return { h: IP.box(3), n: "box 3", kind: "low" };
      case "box5": return { h: IP.box(5), n: "box 5", kind: "low" };
      case "tent": return { h: F([.25, .5, .25]), n: "tent [1,2,1]/4", kind: "low" };
      case "binom4": return { h: F([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]), n: "binomial [1,4,6,4,1]/16", kind: "low" };
      case "gauss": return { h: IP.gauss1(s, 3), n: "Gaussian σ = " + s.toFixed(1), kind: "low" };
      case "cub1": return { h: F([-0.0625, 0, 0.3125, 0.5, 0.3125, 0, -0.0625]), n: "cubic a = −1", kind: "low" };
      case "cub05": return { h: F([-0.03125, 0, 0.28125, 0.5, 0.28125, 0, -0.03125]), n: "cubic a = −0.5", kind: "low" };
      /* the published coefficients are given to four decimals, so ∑h is 0.9998 rather
         than 1 and H(π) is −6 × 10⁻⁴ rather than 0 — rounding in the source table, not
         a property of the filters. The figure reports what the numbers actually do. */
      case "qmf": return { h: F([0.0198, -0.0431, -0.0519, 0.2932, 0.5638, 0.2932, -0.0519, -0.0431, 0.0198]), n: "QMF-9", kind: "low" };
      case "jp2": return { h: F([0.0267, -0.0169, -0.0782, 0.2669, 0.6029, 0.2669, -0.0782, -0.0169, 0.0267]), n: "JPEG-2000 9/7 analysis", kind: "low" };
      case "shift": return { h: F([0, 0, 1]), n: "[0, 0, 1] — an off-centre kernel", kind: "low" };
      case "scaled": return { h: F([1 / 3, 2 / 3, 1 / 3]), n: "[1,2,1]/3 — sums to 4/3", kind: "low" };
      case "dx": return { h: F([-0.5, 0, 0.5]), n: "central difference [−1,0,1]/2", kind: "d1" };
      case "lap": return { h: F([1, -2, 1]), n: "1D Laplacian [1,−2,1]", kind: "d2" };
    }
  }

  function draw() {
    const K = kern(), h = K.h, R = (h.length - 1) >> 1;
    const m = IP.moments(h);
    const want = K.kind === "low" ? { m0: 1, m1: 0, nyq: 0 } : K.kind === "d1" ? { m0: 0, m1: 1, nyq: null } : { m0: 0, m1: 0, nyq: null };

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── left: stem plot + the measured table ── */
    const px = 14, pw = 216;
    g.append("rect").attr("x", px).attr("y", 24).attr("width", pw).attr("height", 104)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, 18, K.n + "  ·  " + h.length + " taps", VC.ink);
    {
      const xs = d3.scaleLinear().domain([-R - 0.6, R + 0.6]).range([px + 10, px + pw - 10]);
      const lo = Math.min(0, ...h), hi = Math.max(...h);
      const ys = d3.scaleLinear().domain([lo - 0.06, hi * 1.14]).range([118, 32]);
      g.append("line").attr("x1", px + 6).attr("x2", px + pw - 6).attr("y1", ys(0)).attr("y2", ys(0)).attr("stroke", VC.line);
      for (let k = -R; k <= R; k++) {
        g.append("line").attr("x1", xs(k)).attr("x2", xs(k)).attr("y1", ys(0)).attr("y2", ys(h[k + R]))
          .attr("stroke", h[k + R] < 0 ? VC.bad : VC.accent).attr("stroke-width", 2);
        g.append("circle").attr("cx", xs(k)).attr("cy", ys(h[k + R])).attr("r", 2.6)
          .attr("fill", h[k + R] < 0 ? VC.bad : VC.accent);
      }
    }
    const rows = [
      ["∑ h(k)", m.m0, want.m0, "DC gain"],
      ["∑ k·h(k)", m.m1, want.m1, "centroid / shift"],
      ["∑ k²·h(k)", m.m2, null, "scale σ²"],
      ["H(π)", m.nyq, want.nyq, "Nyquist"],
      ["∑ h(even k)", m.even, K.kind === "low" ? 0.5 : null, "coset A"],
      ["∑ h(odd k)", m.odd, K.kind === "low" ? 0.5 : null, "coset B"]
    ];
    let ty = 152;
    IP.cap(g, px, 144, "measured", VC.ink);
    rows.forEach(r => {
      const ok = r[2] === null ? null : Math.abs(r[1] - r[2]) < 1e-3;
      g.append("text").attr("x", px).attr("y", ty).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("xml:space", "preserve")
        .attr("fill", VC.muted).text(r[0].padEnd(13));
      g.append("text").attr("x", px + 96).attr("y", ty).attr("font-size", 10.5).attr("text-anchor", "end")
        .attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", ok === null ? VC.ink : (ok ? VC.good : VC.bad)).text(VZ.fmt(r[1], 5));
      g.append("text").attr("x", px + 104).attr("y", ty).attr("font-size", 9.5).attr("fill", VC.muted)
        .text(r[2] === null ? r[3] : (ok ? "= " + r[2] + " ✓" : "≠ " + r[2] + " ✗  " + r[3]));
      ty += 17;
    });

    /* ── middle: frequency response ── */
    const fx = px + pw + 18, fw = 236, fy = 24, fh = 200;
    g.append("rect").attr("x", fx).attr("y", fy).attr("width", fw).attr("height", fh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, fx, 18, "H(ω), 0 → π", VC.ink);
    const lo2 = Math.min(-0.4, m.nyq - 0.2), hi2 = Math.max(1.15, ...[m.m0]);
    const wx = d3.scaleLinear().domain([0, Math.PI]).range([fx + 24, fx + fw - 10]);
    const wy = d3.scaleLinear().domain([lo2, hi2]).range([fy + fh - 20, fy + 10]);
    g.append("line").attr("x1", wx(0)).attr("x2", wx(Math.PI)).attr("y1", wy(0)).attr("y2", wy(0)).attr("stroke", VC.muted);
    g.append("line").attr("x1", wx(0)).attr("x2", wx(Math.PI)).attr("y1", wy(1)).attr("y2", wy(1)).attr("stroke", VC.grid);
    let d = "";
    for (let i = 0; i <= 300; i++) { const w = Math.PI * i / 300; d += (i ? "L" : "M") + wx(w).toFixed(1) + "," + wy(IP.freqResp(h, w).re).toFixed(1); }
    g.append("path").attr("d", d).attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    g.append("circle").attr("cx", wx(Math.PI)).attr("cy", wy(m.nyq)).attr("r", 4)
      .attr("fill", Math.abs(m.nyq) < 1e-9 ? VC.good : VC.bad);
    g.append("text").attr("x", wx(Math.PI) - 6).attr("y", wy(m.nyq) - 8).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", Math.abs(m.nyq) < 1e-9 ? VC.good : VC.bad)
      .text("H(π) = " + VZ.fmt(m.nyq, 4));
    [0, 0.5, 1].forEach(t => g.append("text").attr("x", wx(t * Math.PI)).attr("y", fy + fh - 5)
      .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", VC.muted).text(t === 0 ? "0" : t + "π"));

    /* ── right: the consequence ── */
    const cx = fx + fw + 18, cw = W - cx - 14;
    g.append("rect").attr("x", cx).attr("y", 24).attr("width", cw).attr("height", H - 60)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    const mode = eC.value;
    IP.cap(g, cx, 18, "the consequence", VC.ink);
    let note = "";
    if (mode === "const") {
      const n = 40, sig = new Float64Array(n).fill(1);
      const gsig = IP.corr1(sig, h, "clamp");
      const sy = d3.scaleLinear().domain([-0.35, 1.5]).range([200, 50]);
      const sx = d3.scaleLinear().domain([0, n - 1]).range([cx + 12, cx + cw - 12]);
      g.append("line").attr("x1", sx(0)).attr("x2", sx(n - 1)).attr("y1", sy(1)).attr("y2", sy(1))
        .attr("stroke", VC.muted).attr("stroke-dasharray", "3 3");
      g.append("text").attr("x", cx + cw - 12).attr("y", sy(1) - 4).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", VC.muted).text("input = 1");
      let dd = ""; for (let i = 0; i < n; i++) dd += (i ? "L" : "M") + sx(i).toFixed(1) + "," + sy(gsig[i]).toFixed(1);
      g.append("path").attr("d", dd).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
      note = Math.abs(m.m0 - 1) < 1e-9
        ? "∑h = 1, so a constant image comes out unchanged: the interior value is exactly " + VZ.fmt(gsig[20], 6) + "."
        : "∑h = " + VZ.fmt(m.m0, 5) + " ≠ 1, so the constant is rescaled to " + VZ.fmt(gsig[20], 5) + ". Every flat region in the image changes brightness by that factor.";
    } else if (mode === "shift") {
      const n = 41, sig = new Float64Array(n);
      for (let i = 0; i < n; i++) sig[i] = Math.exp(-((i - 20) ** 2) / 8);
      const gsig = IP.corr1(sig, h, "zero");
      let s0 = 0, s1 = 0, t0 = 0, t1 = 0;
      for (let i = 0; i < n; i++) { s0 += sig[i]; s1 += i * sig[i]; t0 += Math.abs(gsig[i]); t1 += i * Math.abs(gsig[i]); }
      const cIn = s1 / s0, cOut = t1 / t0;
      const sx = d3.scaleLinear().domain([8, 32]).range([cx + 12, cx + cw - 12]);
      const mx = Math.max(...gsig.map(Math.abs), 1);
      const sy = d3.scaleLinear().domain([-mx * 1.1, mx * 1.15]).range([230, 50]);
      g.append("line").attr("x1", sx(8)).attr("x2", sx(32)).attr("y1", sy(0)).attr("y2", sy(0)).attr("stroke", VC.line);
      [["input", sig, VC.accent], ["filtered", gsig, VC.a2]].forEach(([nm, arr, col]) => {
        let dd = ""; for (let i = 8; i <= 32; i++) dd += (i === 8 ? "M" : "L") + sx(i).toFixed(1) + "," + sy(arr[i]).toFixed(1);
        g.append("path").attr("d", dd).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
      });
      g.append("line").attr("x1", sx(cIn)).attr("x2", sx(cIn)).attr("y1", 50).attr("y2", 236).attr("stroke", VC.accent).attr("stroke-dasharray", "3 3");
      g.append("line").attr("x1", sx(cOut)).attr("x2", sx(cOut)).attr("y1", 50).attr("y2", 236).attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
      note = "centroid " + VZ.fmt(cIn, 4) + " → " + VZ.fmt(cOut, 4) + ", a shift of <b>" + VZ.fmt(cOut - cIn, 4)
        + " px</b>. The first moment of the kernel is " + VZ.fmt(m.m1, 4)
        + (Math.abs(m.m1) < 1e-9 ? " — zero, so the feature has not moved."
          : " — non-zero. Under the CORRELATION convention of §08 the predicted shift is −∑k·h/∑h = "
          + VZ.fmt(m.m0 !== 0 ? -m.m1 / m.m0 : NaN, 4) + " px, which is what was measured. Convolution would move it the other way.");
    } else {
      /* filter a constant, then keep every other sample */
      const n = 32, sig = new Float64Array(n).fill(1);
      const gsig = IP.corr1(sig, h, "wrap");
      const evenS = [], oddS = [];
      for (let i = 8; i < 24; i++) (i % 2 === 0 ? evenS : oddS).push(gsig[i]);
      const cellW = (cw - 24) / 16;
      for (let i = 0; i < 16; i++) {
        const v = gsig[8 + i];
        g.append("rect").attr("x", cx + 12 + i * cellW).attr("y", 60).attr("width", cellW + 0.5).attr("height", 40)
          .attr("fill", IP.greyColor(VZ.clamp(v, 0, 1.4) / 1.4, 1)).attr("shape-rendering", "crispEdges");
      }
      IP.cap(g, cx + 12, 54, "filtered constant field", VC.muted);
      for (let i = 0; i < 8; i++) {
        const v = gsig[8 + 2 * i];
        g.append("rect").attr("x", cx + 12 + i * cellW * 2).attr("y", 126).attr("width", cellW * 2 + 0.5).attr("height", 40)
          .attr("fill", IP.greyColor(VZ.clamp(v, 0, 1.4) / 1.4, 1)).attr("shape-rendering", "crispEdges");
      }
      IP.cap(g, cx + 12, 120, "after keeping every other sample (phase 0)", VC.muted);
      for (let i = 0; i < 8; i++) {
        const v = gsig[9 + 2 * i];
        g.append("rect").attr("x", cx + 12 + i * cellW * 2).attr("y", 192).attr("width", cellW * 2 + 0.5).attr("height", 40)
          .attr("fill", IP.greyColor(VZ.clamp(v, 0, 1.4) / 1.4, 1)).attr("shape-rendering", "crispEdges");
      }
      IP.cap(g, cx + 12, 186, "the OTHER phase", VC.muted);
      const a0 = evenS.reduce((x, y) => x + y, 0) / evenS.length, a1 = oddS.reduce((x, y) => x + y, 0) / oddS.length;
      note = "phase 0 mean <b>" + VZ.fmt(a0, 6) + "</b>, phase 1 mean <b>" + VZ.fmt(a1, 6) + "</b>, difference <b>"
        + VZ.fmt(Math.abs(a0 - a1), 6) + "</b>"
        + (Math.abs(a0 - a1) < 1e-9
          ? " — identical, because H(π) = 0. Either phase can be kept."
          : " — the two phases of a constant field are NOT equal, because H(π) = " + VZ.fmt(m.nyq, 4) + " ≠ 0. Subsample this and a checkerboard appears out of nothing.");
    }
    g.append("foreignObject").attr("x", cx + 8).attr("y", H - 92).attr("width", cw - 16).attr("height", 60)
      .append("xhtml:div").attr("style", "font-size:10.5px;color:var(--muted);line-height:1.4;font-family:'SF Mono',Menlo,monospace")
      .html(note);

    out.innerHTML =
      `${K.n} · ∑h = <b>${VZ.fmt(m.m0, 6)}</b> · ∑k·h = <b>${VZ.fmt(m.m1, 6)}</b> · ∑k²·h = <b>${VZ.fmt(m.m2, 6)}</b> · `
      + `H(π) = <b>${VZ.fmt(m.nyq, 6)}</b> · cosets <b>${VZ.fmt(m.even, 6)}</b> / <b>${VZ.fmt(m.odd, 6)}</b> `
      + `(their sum is H(0) = ${VZ.fmt(m.even + m.odd, 6)} and their difference is H(π) = ${VZ.fmt(m.even - m.odd, 6)}, exactly as the derivation says)<br>`
      + (K.kind === "low"
        ? (Math.abs(m.m0 - 1) < 1e-9 && Math.abs(m.m1) < 1e-9 && Math.abs(m.nyq) < 1e-9
          ? `<b>Passes every condition exactly.</b> Usable as a smoother and as a decimation pre-filter.`
          : (Math.abs(m.m0 - 1) < 1e-3 && Math.abs(m.m1) < 1e-3 && Math.abs(m.nyq) < 1e-3
            ? `<b>Passes every condition to the precision the coefficients are published in</b> (four decimals): ∑h = ${VZ.fmt(m.m0, 6)} and H(π) = ${VZ.fmt(m.nyq, 6)} are rounding, not design.`
            : `<b>Violated:</b> ` + [Math.abs(m.m0 - 1) > 1e-3 ? "DC gain (changes brightness)" : null,
            Math.abs(m.m1) > 1e-3 ? "first moment (shifts the image)" : null,
            Math.abs(m.nyq) > 1e-3 ? "Nyquist / coset (unsafe before subsampling)" : null].filter(Boolean).join(", ") + "."))
        : `A derivative filter: the conditions are different. ${K.kind === "d1"
          ? "M0 = 0 kills constants and M1 = 1 makes it return the exact slope of a ramp."
          : "M0 = M1 = 0 kills constants and ramps; M2 = 2 is what makes it return 2 on f = k², whose true second derivative is 2."}`);
  }

  eK.onchange = draw;
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(1); draw(); };
  eC.onchange = draw;
  draw();
})();

/* ───────── 9 · derivative filters on one noisy edge ───────── */
(function () {
  const svg = d3.select("#deriv-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 450, IW = 72, IH = 56;
  const eOp = document.getElementById("dv-op");
  const eS = document.getElementById("dv-s"), eSv = document.getElementById("dv-sv");
  const eK = document.getElementById("dv-k"), eKv = document.getElementById("dv-kv");
  const eG = document.getElementById("dv-g"), eGv = document.getElementById("dv-gv");
  const eN = document.getElementById("dv-n"), eNv = document.getElementById("dv-nv");
  const out = document.getElementById("deriv-readout");

  function gaussTaps(sig) { return IP.gauss1(sig, 3); }
  function dgaussTaps(sig) {                      /* −k/σ² · G(k), then scaled so ∑k·h = −1... */
    const g = gaussTaps(sig), R = (g.length - 1) >> 1;
    const h = new Float64Array(g.length);
    for (let k = -R; k <= R; k++) h[k + R] = -(k / (sig * sig)) * g[k + R];
    /* normalise so that it returns the exact slope of a ramp under CORRELATION */
    let m1 = 0; for (let k = -R; k <= R; k++) m1 += k * h[k + R];
    for (let i = 0; i < h.length; i++) h[i] /= m1;
    return h;
  }
  function d2gaussTaps(sig) {
    const g = gaussTaps(sig), R = (g.length - 1) >> 1;
    const h = new Float64Array(g.length);
    for (let k = -R; k <= R; k++) h[k + R] = ((k * k - sig * sig) / (sig ** 4)) * g[k + R];
    let s = 0; for (const v of h) s += v; for (let i = 0; i < h.length; i++) h[i] -= s / h.length;
    return h;
  }
  function build() {
    const sig = +eS.value, k = +eK.value, gam = +eG.value;
    const op = eOp.value;
    if (op === "cd") return { K: [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], order: 1, n: "central difference / 2", scale: 1 };
    if (op === "sobel") return { K: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]].map(r => r.map(v => v / 8)), order: 1, n: "Sobel x / 8", scale: 8 };
    if (op === "scharr") return { K: [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]].map(r => r.map(v => v / 32)), order: 1, n: "Scharr x / 32", scale: 32 };
    if (op === "dog") { const d = dgaussTaps(sig), g = gaussTaps(sig); return { K: IP.outer(g, d), order: 1, n: "∂G/∂x, σ = " + sig.toFixed(1), sepA: g, sepB: d }; }
    if (op === "lap") return { K: [[0, 1, 0], [1, -4, 1], [0, 1, 0]], order: 2, n: "5-point Laplacian" };
    if (op === "log") {
      const g = gaussTaps(sig), g2 = d2gaussTaps(sig);
      const A = IP.outer(g2, g), B = IP.outer(g, g2);
      return { K: A.map((r, i) => r.map((v, j) => v + B[i][j])), order: 2, n: "LoG, σ = " + sig.toFixed(1) };
    }
    if (op === "dogdiff") {
      const g1 = gaussTaps(sig), g2 = gaussTaps(sig * k);
      const R1 = (g1.length - 1) >> 1, R2 = (g2.length - 1) >> 1, R = R2;
      const A = [], B = [];
      for (let i = -R; i <= R; i++) {
        const ra = new Float64Array(2 * R + 1), rb = new Float64Array(2 * R + 1);
        for (let j = -R; j <= R; j++) {
          ra[j + R] = (Math.abs(i) <= R1 && Math.abs(j) <= R1) ? g1[i + R1] * g1[j + R1] : 0;
          rb[j + R] = g2[i + R2] * g2[j + R2];
        }
        A.push(ra); B.push(rb);
      }
      return { K: A.map((r, i) => Array.from(r, (v, j) => v - B[i][j])), order: 2, n: "DoG, σ = " + sig.toFixed(1) + ", k = " + k.toFixed(2) };
    }
    /* unsharp */
    const g = gaussTaps(sig), R = (g.length - 1) >> 1;
    const K = [];
    for (let i = -R; i <= R; i++) {
      const row = new Float64Array(2 * R + 1);
      for (let j = -R; j <= R; j++) row[j + R] = -gam * g[i + R] * g[j + R] + ((i === 0 && j === 0) ? (1 + gam) : 0);
      K.push(row);
    }
    return { K, order: 0, n: "unsharp mask, σ = " + sig.toFixed(1) + ", γ = " + gam.toFixed(2) };
  }

  function draw() {
    const B = build(), K = B.K, Kn = K.length, R = (Kn - 1) >> 1;
    const nz = +eN.value;
    /* the flat window must actually be flat: the scene has a bright disc near the top
       left (centre row 19, radius 9.5) and a grating below row 36, so rows 30–35 /
       columns 3–29 is the only genuinely featureless patch. Measuring "noise" on the
       disc's rim instead reports an edge as noise — a metric bug, not a filter property. */
    const FLAT = { i0: 30, i1: 36, j0: 3, j1: 30 };
    const EDGE = { i0: 30, i1: 36, j0: 30, j1: 46 };

    /* a 1-D cross-section: a soft edge plus noise */
    const n = 96, r = VZ.rng(31);
    const prof = new Float64Array(n), clean = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = (i - 48) / 3.0;
      clean[i] = 0.25 + 0.5 / (1 + Math.exp(-t));
      prof[i] = clean[i] + nz * VZ.randn(r);
    }
    /* the 1D response: use the centre row of the 2D kernel summed over rows */
    const h1 = new Float64Array(Kn);
    for (let j = 0; j < Kn; j++) { let s = 0; for (let i = 0; i < Kn; i++) s += K[i][j]; h1[j] = s; }
    const resp = IP.corr1(prof, h1, "clamp");
    const respClean = IP.corr1(clean, h1, "clamp");

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── left: the profile and the response ── */
    const px = 30, pw = 300;
    g.append("rect").attr("x", px).attr("y", 26).attr("width", pw).attr("height", 110)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, 20, "a noisy soft edge, cross-section", VC.ink);
    const sx = d3.scaleLinear().domain([24, 72]).range([px + 8, px + pw - 8]);
    const sy = d3.scaleLinear().domain([0.1, 0.9]).range([128, 34]);
    let d1 = "", d0 = "";
    for (let i = 24; i <= 72; i++) { d1 += (i === 24 ? "M" : "L") + sx(i).toFixed(1) + "," + sy(prof[i]).toFixed(1); d0 += (i === 24 ? "M" : "L") + sx(i).toFixed(1) + "," + sy(clean[i]).toFixed(1); }
    g.append("path").attr("d", d0).attr("fill", "none").attr("stroke", VC.muted).attr("stroke-dasharray", "3 3");
    g.append("path").attr("d", d1).attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);

    const ry0 = 150, rh = 120;
    g.append("rect").attr("x", px).attr("y", ry0).attr("width", pw).attr("height", rh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, ry0 - 6, "the operator's response along the same line", VC.ink);
    let m = 1e-9; for (let i = 24; i <= 72; i++) m = Math.max(m, Math.abs(resp[i]));
    const ry = d3.scaleLinear().domain([-m * 1.15, m * 1.15]).range([ry0 + rh - 8, ry0 + 8]);
    g.append("line").attr("x1", sx(24)).attr("x2", sx(72)).attr("y1", ry(0)).attr("y2", ry(0)).attr("stroke", VC.muted);
    let d2 = "", d3s = "";
    for (let i = 24; i <= 72; i++) { d2 += (i === 24 ? "M" : "L") + sx(i).toFixed(1) + "," + ry(resp[i]).toFixed(1); d3s += (i === 24 ? "M" : "L") + sx(i).toFixed(1) + "," + ry(respClean[i]).toFixed(1); }
    g.append("path").attr("d", d3s).attr("fill", "none").attr("stroke", VC.muted).attr("stroke-dasharray", "3 3");
    g.append("path").attr("d", d2).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.8);
    /* mark the peak or zero crossing */
    let mark = null, markLbl = "";
    if (B.order === 2) {
      for (let i = 30; i < 66; i++) if (respClean[i] * respClean[i + 1] < 0) {
        const t = respClean[i] / (respClean[i] - respClean[i + 1]); mark = i + t; markLbl = "zero crossing at " + VZ.fmt(mark, 3); break;
      }
    } else if (B.order === 1) {
      let best = 30; for (let i = 30; i < 66; i++) if (Math.abs(respClean[i]) > Math.abs(respClean[best])) best = i;
      /* parabolic refinement */
      const a = Math.abs(respClean[best - 1]), b = Math.abs(respClean[best]), c = Math.abs(respClean[best + 1]);
      mark = best + 0.5 * (a - c) / (a - 2 * b + c); markLbl = "peak at " + VZ.fmt(mark, 3);
    }
    if (mark !== null) {
      g.append("line").attr("x1", sx(mark)).attr("x2", sx(mark)).attr("y1", ry0 + 6).attr("y2", ry0 + rh - 6)
        .attr("stroke", VC.good).attr("stroke-dasharray", "3 3");
      g.append("line").attr("x1", sx(48)).attr("x2", sx(48)).attr("y1", 30).attr("y2", 132)
        .attr("stroke", VC.violet).attr("stroke-dasharray", "2 3");
      IP.cap(g, px + 8, ry0 + 14, markLbl + "   (the true edge is at 48)", VC.good);
    }

    /* ── middle: the kernel ── */
    const kx = px + pw + 22;
    const kc = Math.min(11, 130 / Kn);
    IP.cap(g, kx, 20, B.n, VC.ink);
    IP.raster(g, K, kx, 26, kc, { signed: true, lo: -IP.absMax(K), hi: IP.absMax(K) });
    const m1D = IP.moments(h1);
    IP.numText(g, [
      "the 1-D kernel it induces:",
      "  ∑h      = " + VZ.fmt(m1D.m0, 6),
      "  ∑k·h    = " + VZ.fmt(m1D.m1, 6),
      "  ∑k²·h   = " + VZ.fmt(m1D.m2, 6),
      "  H(π)    = " + VZ.fmt(m1D.nyq, 4),
      "  taps    = " + Kn + " × " + Kn
    ], kx, 40 + Kn * kc, { size: 10, lead: 13 });

    /* ── right: the 2D result and the noise/response trade ── */
    const ix = kx + 150, cell = 3.0;
    const A = IP.scene(IW, IH, { noise: nz, seed: 5 });
    const F2 = IP.corr2(A, K, "mirror");
    IP.cap(g, ix, 20, "applied to the test image", VC.ink);
    IP.raster(g, F2, ix, 26, cell, B.order === 0 ? { lo: 0, hi: 1 } : { signed: true, lo: -IP.absMax(F2), hi: IP.absMax(F2) });
    [[FLAT, VC.good, "flat"], [EDGE, VC.bad, "edge"]].forEach(([r, col, lbl]) => {
      g.append("rect").attr("x", ix + r.j0 * cell).attr("y", 26 + r.i0 * cell)
        .attr("width", (r.j1 - r.j0) * cell).attr("height", (r.i1 - r.i0) * cell)
        .attr("fill", "none").attr("stroke", col).attr("stroke-dasharray", "3 2");
      g.append("text").attr("x", ix + r.j0 * cell).attr("y", 26 + r.i0 * cell - 2)
        .attr("font-size", 8.5).attr("fill", col).text(lbl);
    });

    /* flat-region noise vs edge response, measured */
    let s0 = 0, s2 = 0, cnt = 0;
    for (let i = FLAT.i0; i < FLAT.i1; i++) for (let j = FLAT.j0; j < FLAT.j1; j++) { s0 += F2[i][j]; s2 += F2[i][j] * F2[i][j]; cnt++; }
    const mu = s0 / cnt, sd = Math.sqrt(Math.max(0, s2 / cnt - mu * mu));
    let edgeResp = 0;
    for (let i = EDGE.i0; i < EDGE.i1; i++) for (let j = EDGE.j0; j < EDGE.j1; j++) edgeResp = Math.max(edgeResp, Math.abs(F2[i][j]));

    const ty0 = 26 + IH * cell + 26, tw = W - ix - 16, th = H - ty0 - 22;
    g.append("rect").attr("x", ix).attr("y", ty0).attr("width", tw).attr("height", th)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, ix, ty0 - 6, "the Gaussian-derivative family: noise vs edge response, against σ", VC.ink);
    const sigs = VZ.linspace(0.6, 4, 18);
    const pts = sigs.map(sg => {
      const gg = gaussTaps(sg);
      const dd = B.order === 2 ? d2gaussTaps(sg) : dgaussTaps(sg);
      const Kx = B.order === 2
        ? (() => { const Aa = IP.outer(dd, gg), Bb = IP.outer(gg, dd); return Aa.map((r, i) => r.map((v, j) => v + Bb[i][j])); })()
        : IP.outer(gg, dd);
      const Ff = IP.corr2(A, Kx, "mirror");
      let a0 = 0, a2 = 0, c = 0;
      for (let i = FLAT.i0; i < FLAT.i1; i++) for (let j = FLAT.j0; j < FLAT.j1; j++) { a0 += Ff[i][j]; a2 += Ff[i][j] * Ff[i][j]; c++; }
      const mm = a0 / c, ss = Math.sqrt(Math.max(0, a2 / c - mm * mm));
      let er = 0; for (let i = EDGE.i0; i < EDGE.i1; i++) for (let j = EDGE.j0; j < EDGE.j1; j++) er = Math.max(er, Math.abs(Ff[i][j]));
      return { s: sg, noise: ss, edge: er };
    });
    const nmax = Math.max(...pts.map(p => p.noise)), emax = Math.max(...pts.map(p => p.edge));
    const tx = d3.scaleLinear().domain([0.6, 4]).range([ix + 26, ix + tw - 10]);
    const tyn = d3.scaleLinear().domain([0, 1.05]).range([ty0 + th - 20, ty0 + 10]);
    [["noise sd", "noise", nmax, VC.bad], ["edge response", "edge", emax, VC.good]].forEach(([lbl, key, mx, col]) => {
      let dd = "";
      pts.forEach((p, i) => dd += (i ? "L" : "M") + tx(p.s).toFixed(1) + "," + tyn(p[key] / mx).toFixed(1));
      g.append("path").attr("d", dd).attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.8);
    });
    g.append("line").attr("x1", tx(+eS.value)).attr("x2", tx(+eS.value)).attr("y1", ty0 + 8).attr("y2", ty0 + th - 18)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    [1, 2, 3, 4].forEach(t => g.append("text").attr("x", tx(t)).attr("y", ty0 + th - 5).attr("text-anchor", "middle")
      .attr("font-size", 9).attr("fill", VC.muted).text("σ=" + t));
    VZ.legend(g, [{ color: VC.bad, label: "noise let through" }, { color: VC.good, label: "response at the edge" }],
      ix + 34, ty0 + 18, { vertical: true, gap: 13, font: 9.5 });

    out.innerHTML =
      `${B.n} · ${Kn} × ${Kn} · induced 1-D moments: ∑h = <b>${VZ.fmt(m1D.m0, 6)}</b>, ∑k·h = <b>${VZ.fmt(m1D.m1, 6)}</b>, ∑k²·h = <b>${VZ.fmt(m1D.m2, 6)}</b><br>`
      + `measured on the image: noise standard deviation in a flat region <b>${VZ.fmt(sd, 5)}</b>, peak response at the edge <b>${VZ.fmt(edgeResp, 5)}</b>, `
      + `ratio <b>${VZ.fmt(sd > 0 ? edgeResp / sd : Infinity, 2)}</b><br>`
      + (B.order === 1
        ? `A first-derivative operator: ∑h = 0 kills the constant and ∑k·h = ${VZ.fmt(m1D.m1, 4)} sets the units. Its response peaks at the edge, and the peak was located at ${mark !== null ? VZ.fmt(mark, 3) : "—"} against a true edge at 48.`
        : B.order === 2
          ? `A second-derivative operator: ∑h and ∑k·h are both zero, so it kills constants and ramps. The edge is where the response crosses zero — measured at ${mark !== null ? VZ.fmt(mark, 3) : "—"}, true value 48 — which is why second-derivative operators localise better and are noisier.`
          : `A sharpening filter: ∑h = ${VZ.fmt(m1D.m0, 6)} exactly 1, so flat regions are untouched, but the negative surround produces the overshoot visible on either side of the edge above. Raise γ and the halo grows linearly.`);
  }

  eOp.onchange = draw;
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(1); draw(); };
  eK.oninput = () => { eKv.textContent = (+eK.value).toFixed(2); draw(); };
  eG.oninput = () => { eGv.textContent = (+eG.value).toFixed(2); draw(); };
  eN.oninput = () => { eNv.textContent = (+eN.value).toFixed(3); draw(); };
  draw();
})();

/* ───────── 10 · steerable filters ───────── */
(function () {
  const svg = d3.select("#steer-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 450, IW = 72, IH = 56;
  const eO = document.getElementById("st-o");
  const eT = document.getElementById("st-th"), eTv = document.getElementById("st-thv");
  const eS = document.getElementById("st-s"), eSv = document.getElementById("st-sv");
  const eB = document.getElementById("st-b");
  const eE = document.getElementById("st-en");
  const out = document.getElementById("steer-readout");

  /* analytic Gaussian derivatives, sampled on a (2R+1)² grid */
  function kernel(sig, dx, dy) {
    const R = Math.max(2, Math.ceil(3 * sig)), K = [];
    const s2 = sig * sig;
    for (let i = -R; i <= R; i++) {
      const row = new Float64Array(2 * R + 1);
      for (let j = -R; j <= R; j++) {
        const x = j, y = i;
        const G = Math.exp(-(x * x + y * y) / (2 * s2)) / (2 * Math.PI * s2);
        let v;
        if (dx === 1 && dy === 0) v = -x / s2 * G;
        else if (dx === 0 && dy === 1) v = -y / s2 * G;
        else if (dx === 2 && dy === 0) v = (x * x - s2) / (s2 * s2) * G;
        else if (dx === 0 && dy === 2) v = (y * y - s2) / (s2 * s2) * G;
        else if (dx === 1 && dy === 1) v = (x * y) / (s2 * s2) * G;
        else v = G;
        row[j + R] = v;
      }
      K.push(row);
    }
    return K;
  }
  /* the directional derivative filter built DIRECTLY, by rotating coordinates */
  function directional(sig, th, order) {
    const c = Math.cos(th), s = Math.sin(th);
    const R = Math.max(2, Math.ceil(3 * sig)), K = [], s2 = sig * sig;
    for (let i = -R; i <= R; i++) {
      const row = new Float64Array(2 * R + 1);
      for (let j = -R; j <= R; j++) {
        const x = j, y = i;
        const u = c * x + s * y;
        const G = Math.exp(-(x * x + y * y) / (2 * s2)) / (2 * Math.PI * s2);
        row[j + R] = order === 1 ? (-u / s2) * G : ((u * u - s2) / (s2 * s2)) * G;
      }
      K.push(row);
    }
    return K;
  }
  const lincomb = (Ks, w) => Ks[0].map((r, i) => Float64Array.from(r, (v, j) => Ks.reduce((s, K, m) => s + w[m] * K[i][j], 0)));

  function draw() {
    const order = +eO.value, th = VZ.rad(+eT.value), sig = +eS.value, useRot = eB.value === "rot";
    let basis, weights, labels;
    if (order === 1) {
      if (useRot) {
        const ang = [0, Math.PI / 2];
        basis = ang.map(a => directional(sig, a, 1));
        weights = [Math.cos(th), Math.sin(th)];
        labels = ["G at 0°", "G at 90°"];
      } else {
        basis = [kernel(sig, 1, 0), kernel(sig, 0, 1)];
        weights = [Math.cos(th), Math.sin(th)];
        labels = ["G_x", "G_y"];
      }
    } else {
      if (useRot) {
        const ang = [0, Math.PI / 3, 2 * Math.PI / 3];
        basis = ang.map(a => directional(sig, a, 2));
        /* solve Mᵀ w = target, where row m of M is (cos²a, 2 cos a sin a, sin²a) */
        const M = ang.map(a => [Math.cos(a) ** 2, 2 * Math.cos(a) * Math.sin(a), Math.sin(a) ** 2]);
        const t = [Math.cos(th) ** 2, 2 * Math.cos(th) * Math.sin(th), Math.sin(th) ** 2];
        weights = VZ.mv(VZ.invN(VZ.T(M)), t);
        labels = ["G_θθ at 0°", "G_θθ at 60°", "G_θθ at 120°"];
      } else {
        basis = [kernel(sig, 2, 0), kernel(sig, 1, 1), kernel(sig, 0, 2)];
        weights = [Math.cos(th) ** 2, 2 * Math.cos(th) * Math.sin(th), Math.sin(th) ** 2];
        labels = ["G_xx", "G_xy", "G_yy"];
      }
    }
    const synth = lincomb(basis, weights);
    const direct = directional(sig, th, order);
    let maxd = 0;
    for (let i = 0; i < synth.length; i++) for (let j = 0; j < synth[0].length; j++)
      maxd = Math.max(maxd, Math.abs(synth[i][j] - direct[i][j]));

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const Kn = synth.length, kc = Math.min(9, 76 / Kn);

    /* ── left: the basis, the synthesis, and the direct construction ── */
    IP.cap(g, 12, 18, "basis filters", VC.ink);
    basis.forEach((K, m) => {
      const x = 12 + m * 86;
      IP.raster(g, K, x, 26, kc, { signed: true, lo: -IP.absMax(K), hi: IP.absMax(K) });
      IP.cap(g, x, 34 + Kn * kc, labels[m], VC.muted);
      IP.cap(g, x, 48 + Kn * kc, "× " + VZ.fmt(weights[m], 4), VC.a2);
    });
    const y2 = 68 + Kn * kc;
    IP.cap(g, 12, y2 - 6, "synthesised: Σ wₘ · basisₘ", VC.good);
    IP.raster(g, synth, 12, y2, kc, { signed: true, lo: -IP.absMax(synth), hi: IP.absMax(synth) });
    IP.cap(g, 12 + 86, y2 - 6, "built directly, by rotating", VC.accent);
    IP.raster(g, direct, 12 + 86, y2, kc, { signed: true, lo: -IP.absMax(direct), hi: IP.absMax(direct) });
    g.append("text").attr("x", 12).attr("y", y2 + Kn * kc + 18).attr("font-size", 10.5)
      .attr("fill", maxd < 1e-12 ? VC.good : VC.bad)
      .text("max |difference| = " + maxd.toExponential(2));

    /* ── middle: the filtered image ── */
    const ix = 12 + 3 * 86 + 10, cell = 3.2;
    const A = IP.scene(IW, IH, { seed: 5 });
    const Rsp = IP.corr2(A, synth, "mirror");
    let SHOW = Rsp, ttl = "response of the steered filter, θ = " + eT.value + "°";
    if (eE.checked) {
      /* oriented energy from the basis: E(θ) computed per pixel, at the current θ */
      const comps = basis.map(K => IP.corr2(A, K, "mirror"));
      SHOW = comps[0].map((r, i) => Float64Array.from(r, (v, j) => {
        let s = 0; for (let m = 0; m < comps.length; m++) s += weights[m] * comps[m][i][j];
        return s * s;
      }));
      ttl = "oriented energy E(θ) at θ = " + eT.value + "°";
    }
    IP.cap(g, ix, 18, ttl, VC.ink);
    IP.raster(g, SHOW, ix, 26, cell, eE.checked ? { lo: 0, hi: IP.absMax(SHOW) } : { signed: true, lo: -IP.absMax(SHOW), hi: IP.absMax(SHOW) });
    /* the probe pixel */
    const pi = 20, pj = 40;
    g.append("circle").attr("cx", ix + pj * cell + cell / 2).attr("cy", 26 + pi * cell + cell / 2).attr("r", 4)
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.6);
    IP.cap(g, ix, 26 + IH * cell + 16, "the circled pixel is the one plotted at the right", VC.a2);

    /* ── right: the polar response at the probe pixel ── */
    const rx = ix, ry = 26 + IH * cell + 34, rr = Math.min(96, (H - ry - 26) / 2);
    const cx0 = W - rr - 30, cy0 = ry + rr;
    const comps = basis.map(K => IP.corr2(A, K, "mirror")[pi][pj]);
    const resp = a => {
      let w;
      if (order === 1) w = useRot ? [Math.cos(a), Math.sin(a)] : [Math.cos(a), Math.sin(a)];
      else {
        const t = [Math.cos(a) ** 2, 2 * Math.cos(a) * Math.sin(a), Math.sin(a) ** 2];
        if (useRot) {
          const ang = [0, Math.PI / 3, 2 * Math.PI / 3];
          const M = ang.map(q => [Math.cos(q) ** 2, 2 * Math.cos(q) * Math.sin(q), Math.sin(q) ** 2]);
          w = VZ.mv(VZ.invN(VZ.T(M)), t);
        } else w = t;
      }
      let s = 0; for (let m = 0; m < comps.length; m++) s += w[m] * comps[m];
      return eE.checked ? s * s : s;
    };
    let rmax = 1e-12;
    for (let i = 0; i < 360; i++) rmax = Math.max(rmax, Math.abs(resp(VZ.rad(i))));
    g.append("circle").attr("cx", cx0).attr("cy", cy0).attr("r", rr).attr("fill", VC.panel2).attr("stroke", VC.line);
    g.append("line").attr("x1", cx0 - rr).attr("x2", cx0 + rr).attr("y1", cy0).attr("y2", cy0).attr("stroke", VC.grid);
    g.append("line").attr("y1", cy0 - rr).attr("y2", cy0 + rr).attr("x1", cx0).attr("x2", cx0).attr("stroke", VC.grid);
    let dpos = "", dneg = "";
    for (let i = 0; i <= 360; i++) {
      const a = VZ.rad(i), v = resp(a), rad = Math.abs(v) / rmax * rr;
      const px2 = cx0 + rad * Math.cos(a), py2 = cy0 - rad * Math.sin(a);
      if (v >= 0) dpos += (dpos ? "L" : "M") + px2.toFixed(1) + "," + py2.toFixed(1);
      else dneg += (dneg ? "L" : "M") + px2.toFixed(1) + "," + py2.toFixed(1);
    }
    if (dpos) g.append("path").attr("d", dpos).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.8);
    if (dneg) g.append("path").attr("d", dneg).attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);
    /* the closed-form maximum */
    let bestA = 0, bestV = -Infinity;
    for (let i = 0; i < 720; i++) { const a = VZ.rad(i / 2), v = Math.abs(resp(a)); if (v > bestV) { bestV = v; bestA = a; } }
    g.append("line").attr("x1", cx0).attr("y1", cy0)
      .attr("x2", cx0 + rr * Math.cos(bestA)).attr("y2", cy0 - rr * Math.sin(bestA))
      .attr("stroke", VC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "3 3");
    g.append("line").attr("x1", cx0).attr("y1", cy0)
      .attr("x2", cx0 + rr * Math.cos(th)).attr("y2", cy0 - rr * Math.sin(th))
      .attr("stroke", VC.violet).attr("stroke-width", 2);
    IP.cap(g, cx0 - rr, ry - 6, "response at the probe, as θ sweeps a full turn", VC.ink);
    IP.numText(g, [
      "dominant orientation " + VZ.fmt(VZ.deg(bestA), 2) + "°",
      "current θ           " + VZ.fmt(VZ.deg(th), 2) + "°",
      "response there      " + VZ.fmt(resp(th), 5)
    ], rx, ry + 16, { size: 10, lead: 13 });

    out.innerHTML =
      `order ${order}, <span class="keep">σ</span> = ${sig.toFixed(1)}, ${basis.length} basis filters, ${eB.value === "rot" ? "rotated copies at equal angles" : "Cartesian derivatives"} · `
      + `weights (${weights.map(w => VZ.fmt(w, 5)).join(", ")})`
      + (order === 2 && eB.value === "rot"
        ? `, and for the second-order rotated basis they sum to <b>${VZ.fmt(weights.reduce((a, b) => a + b, 0), 6)}</b> — exactly 1, for every angle`
        : ` (nothing constrains their sum; only the second-order rotated basis has that property)`)
      + `<br>`
      + `The steered kernel built from the basis and the kernel built directly by rotating coordinates differ by at most <b>${maxd.toExponential(3)}</b> — `
      + `steering is <b>exact</b>, not an approximation. Cost: ${basis.length} convolutions once, then every one of the infinitely many orientations is a `
      + `${basis.length}-term weighted sum of stored responses.`;
  }

  eO.onchange = draw;
  eT.oninput = () => { eTv.textContent = eT.value + "°"; draw(); };
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(1); draw(); };
  eB.onchange = draw; eE.onchange = draw;
  draw();
})();

/* ───────── 11 · the bilateral filter's two weight maps ───────── */
(function () {
  const svg = d3.select("#bilat-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470, IW = 72, IH = 56;
  const eSd = document.getElementById("bl-sd"), eSdv = document.getElementById("bl-sdv");
  const eSr = document.getElementById("bl-sr"), eSrv = document.getElementById("bl-srv");
  const eX = document.getElementById("bl-x"), eXv = document.getElementById("bl-xv");
  const eY = document.getElementById("bl-y"), eYv = document.getElementById("bl-yv");
  const eN = document.getElementById("bl-n"), eNv = document.getElementById("bl-nv");
  const eIt = document.getElementById("bl-it"), eItv = document.getElementById("bl-itv");
  const out = document.getElementById("bilat-readout");

  function bilateral(A, sd, sr) {
    const R = Math.max(1, Math.ceil(2.5 * sd)), Hh = A.length, Ww = A[0].length;
    const G = IP.zeros2(Hh, Ww);
    for (let i = 0; i < Hh; i++) for (let j = 0; j < Ww; j++) {
      let s = 0, wsum = 0;
      for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
        const ii = i + k, jj = j + l;
        if (ii < 0 || ii >= Hh || jj < 0 || jj >= Ww) continue;
        const dv = A[ii][jj] - A[i][j];
        const w = Math.exp(-(k * k + l * l) / (2 * sd * sd) - (dv * dv) / (2 * sr * sr));
        s += w * A[ii][jj]; wsum += w;
      }
      G[i][j] = wsum > 0 ? s / wsum : A[i][j];
    }
    return G;
  }

  function draw() {
    const sd = +eSd.value, sr = +eSr.value, nz = +eN.value, iters = +eIt.value;
    const pi = Math.round(VZ.clamp(+eY.value, 0, IH - 1)), pj = Math.round(VZ.clamp(+eX.value, 0, IW - 1));
    const A = IP.scene(IW, IH, { noise: nz, seed: 11 });
    let B = A;
    for (let t = 0; t < iters; t++) B = bilateral(B, sd, sr);
    const gk = IP.gauss1(sd, 3);
    const Gs = IP.sepV(IP.sepH(A, gk, "mirror"), gk, "mirror");

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell = 3.0;

    IP.cap(g, 12, 18, "input (noise sd " + VZ.fmt(nz, 3) + ")", VC.accent);
    IP.raster(g, A, 12, 24, cell, { lo: 0, hi: 1 });
    IP.cap(g, 12 + IW * cell + 22, 18, "bilateral × " + iters, VC.a2);
    IP.raster(g, B, 12 + IW * cell + 22, 24, cell, { lo: 0, hi: 1 });
    IP.cap(g, 12 + 2 * (IW * cell + 22), 18, "plain Gaussian, same σ_d", VC.muted);
    IP.raster(g, Gs, 12 + 2 * (IW * cell + 22), 24, cell, { lo: 0, hi: 1 });
    /* the probe */
    const R = Math.max(1, Math.ceil(2.5 * sd));
    [0, 1, 2].forEach(m => {
      const x0 = 12 + m * (IW * cell + 22);
      g.append("rect").attr("x", x0 + (pj - R) * cell).attr("y", 24 + (pi - R) * cell)
        .attr("width", (2 * R + 1) * cell).attr("height", (2 * R + 1) * cell)
        .attr("fill", "none").attr("stroke", VC.violet);
      g.append("circle").attr("cx", x0 + pj * cell + cell / 2).attr("cy", 24 + pi * cell + cell / 2)
        .attr("r", 3).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.5);
    });

    /* ── the three weight maps at the probe ── */
    const wy0 = 24 + IH * cell + 30;
    const n = 2 * R + 1;
    const D = IP.zeros2(n, n), Rr = IP.zeros2(n, n), P = IP.zeros2(n, n), patch = IP.zeros2(n, n);
    let wsum = 0, dsum = 0, num = 0;
    for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
      const ii = VZ.clamp(pi + k, 0, IH - 1), jj = VZ.clamp(pj + l, 0, IW - 1);
      const dv = A[ii][jj] - A[pi][pj];
      const dw = Math.exp(-(k * k + l * l) / (2 * sd * sd));
      const rw = Math.exp(-(dv * dv) / (2 * sr * sr));
      D[k + R][l + R] = dw; Rr[k + R][l + R] = rw; P[k + R][l + R] = dw * rw;
      patch[k + R][l + R] = A[ii][jj];
      wsum += dw * rw; dsum += dw; num += dw * rw * A[ii][jj];
    }
    const wc = Math.min(9, 84 / n);
    const maps = [
      ["the image patch", patch, { lo: 0, hi: 1 }],
      ["domain weights d", D, { lo: 0, hi: 1 }],
      ["range weights r", Rr, { lo: 0, hi: 1 }],
      ["their product w = d·r", P, { lo: 0, hi: 1 }]
    ];
    maps.forEach((m, k) => {
      const x0 = 12 + k * (n * wc + 26);
      IP.cap(g, x0, wy0 - 6, m[0], k === 3 ? VC.good : VC.muted);
      IP.raster(g, m[1], x0, wy0, wc, m[2]);
    });
    IP.cap(g, 12, wy0 + n * wc + 16,
      "the domain map never changes; the range map is entirely the image's doing", VC.muted);

    /* ── the cross-section ── */
    const cx0 = 12 + 4 * (n * wc + 26) + 6, cw = W - cx0 - 14;
    const cy0 = wy0 - 14, chh = H - cy0 - 34;
    g.append("rect").attr("x", cx0).attr("y", cy0).attr("width", cw).attr("height", chh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, cx0, cy0 - 6, "row " + pi + ": input, bilateral, Gaussian", VC.ink);
    const lx = d3.scaleLinear().domain([0, IW - 1]).range([cx0 + 8, cx0 + cw - 8]);
    const ly = d3.scaleLinear().domain([0, 1.05]).range([cy0 + chh - 18, cy0 + 10]);
    [[A, VC.accent, 1], [Gs, VC.muted, 1.4], [B, VC.a2, 2]].forEach(([M, col, wd]) => {
      let d = ""; for (let j = 0; j < IW; j++) d += (j ? "L" : "M") + lx(j).toFixed(1) + "," + ly(M[pi][j]).toFixed(1);
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", col).attr("stroke-width", wd);
    });
    g.append("line").attr("x1", lx(pj)).attr("x2", lx(pj)).attr("y1", cy0 + 8).attr("y2", cy0 + chh - 16)
      .attr("stroke", VC.violet).attr("stroke-dasharray", "3 3");
    VZ.legend(g, [{ color: VC.accent, label: "input" }, { color: VC.muted, label: "Gaussian" }, { color: VC.a2, label: "bilateral" }],
      cx0 + 12, cy0 + 20, { vertical: true, gap: 13, font: 9.5 });

    const kept = wsum / dsum;
    out.innerHTML =
      `probe (${pi}, ${pj}), value ${VZ.fmt(A[pi][pj], 4)} · <span class="keep">σ</span>_d = ${sd.toFixed(1)} (window ${n} × ${n}), <span class="keep">σ</span>_r = ${sr.toFixed(2)}<br>`
      + `the domain kernel's total mass is <b>${VZ.fmt(dsum, 4)}</b>; after multiplying by the range kernel only <b>${VZ.fmt(wsum, 4)}</b> survives — `
      + `the range term discarded <b>${VZ.fmt(100 * (1 - kept), 1)}%</b> of the spatial kernel's weight.<br>`
      + `filtered value here: bilateral <b>${VZ.fmt(num / wsum, 5)}</b> · plain Gaussian <b>${VZ.fmt(Gs[pi][pj], 5)}</b> · input ${VZ.fmt(A[pi][pj], 5)}`
      + (kept > 0.97
        ? ` — the probe is in a flat region, so almost nothing was discarded and the bilateral filter is behaving like a Gaussian. Move it onto an edge.`
        : kept < 0.55
          ? ` — the probe is on a strong edge: more than half the kernel has been switched off, and the bilateral filter is averaging over one side only.`
          : ` — the probe is near an edge, and part of the kernel has been switched off.`);
  }

  [[eSd, eSdv, 1], [eSr, eSrv, 2], [eX, eXv, 0], [eY, eYv, 0], [eN, eNv, 3], [eIt, eItv, 0]].forEach(([el, lbl, dp]) => {
    el.oninput = () => { lbl.textContent = (+el.value).toFixed(dp); draw(); };
  });
  draw();
})();

/* ───────── 12 · box vs Gaussian vs median vs bilateral, measured ───────── */
(function () {
  const svg = d3.select("#shoot-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 480, N = 201;
  const eNt = document.getElementById("sh-nt");
  const eN = document.getElementById("sh-n"), eNv = document.getElementById("sh-nv");
  const eE = document.getElementById("sh-e"), eEv = document.getElementById("sh-ev");
  const eW = document.getElementById("sh-w"), eWv = document.getElementById("sh-wv");
  const eSr = document.getElementById("sh-sr"), eSrv = document.getElementById("sh-srv");
  const out = document.getElementById("shoot-readout");

  const LO = 0.25, HI = 0.75;

  function medianFilter(f, K) {
    const R = (K - 1) >> 1, g = new Float64Array(f.length), buf = [];
    for (let i = 0; i < f.length; i++) {
      buf.length = 0;
      for (let k = -R; k <= R; k++) buf.push(IP.at1(f, i + k, "clamp"));
      buf.sort((a, b) => a - b);
      g[i] = buf[R];
    }
    return g;
  }
  function bilat1(f, sd, sr) {
    const R = Math.max(1, Math.ceil(2.5 * sd)), g = new Float64Array(f.length);
    for (let i = 0; i < f.length; i++) {
      let s = 0, w = 0;
      for (let k = -R; k <= R; k++) {
        const v = IP.at1(f, i + k, "clamp");
        const ww = Math.exp(-(k * k) / (2 * sd * sd) - ((v - f[i]) ** 2) / (2 * sr * sr));
        s += ww * v; w += ww;
      }
      g[i] = w > 0 ? s / w : f[i];
    }
    return g;
  }

  function draw() {
    const nz = +eN.value, soft = +eE.value, wid = +eW.value, sr = +eSr.value, nt = eNt.value;
    const K = 2 * Math.round(wid * 2) + 1;
    const r = VZ.rng(4);
    const clean = new Float64Array(N), noisy = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      clean[i] = LO + (HI - LO) / (1 + Math.exp(-(i - 100) / soft));
      noisy[i] = clean[i] + ((nt === "gauss" || nt === "both") ? nz * VZ.randn(r) : 0);
    }
    if (nt === "shot" || nt === "both") {
      const cnt = Math.max(1, Math.round(0.05 * N));
      for (let k = 0; k < cnt; k++) noisy[Math.floor(r() * N)] = r() < 0.5 ? 0 : 1;
    }

    const gk = IP.gauss1(wid, 3);
    const res = [
      { n: "unfiltered", y: noisy, c: VC.muted },
      { n: "box K = " + K, y: IP.corr1(noisy, IP.box(K), "clamp"), c: VC.bad },
      { n: "Gaussian σ = " + wid.toFixed(2), y: IP.corr1(noisy, gk, "clamp"), c: VC.accent },
      { n: "median K = " + K, y: medianFilter(noisy, K), c: VC.violet },
      { n: "bilateral σ_r = " + sr.toFixed(2), y: bilat1(noisy, wid, sr), c: VC.a2 }
    ];

    /* metrics */
    /* the flat region: everything more than 8 softness-lengths from the transition.
       The bounds must be integers or the array lookups return undefined and every
       measurement below silently becomes NaN. */
    const flat = [], gap = Math.ceil(8 * soft) + 2;
    for (let i = 12; i < 100 - gap; i++) flat.push(i);
    for (let i = 100 + gap; i < N - 12; i++) flat.push(i);
    const a = LO + 0.1 * (HI - LO), b = LO + 0.9 * (HI - LO);
    res.forEach(d => {
      let s = 0, s2 = 0;
      flat.forEach(i => { const e = d.y[i] - clean[i]; s += e; s2 += e * e; });
      const mu = s / flat.length;
      d.noise = Math.sqrt(Math.max(0, s2 / flat.length - mu * mu));
      let se = 0; for (let i = 0; i < N; i++) se += (d.y[i] - clean[i]) ** 2;
      d.rmse = Math.sqrt(se / N);
      /* the 10–90 rise implied by the STEEPEST slope. Reading the two crossings
         directly off a noisy trace is not robust — one early excursion past the 10%
         level inflates the answer — whereas the steepest slope is. For an ideal box
         of width K the output ramps linearly, so this returns 0.8·K exactly. */
      let best = 0;
      for (let i = 80; i < 121; i++) best = Math.max(best, Math.abs(d.y[i + 1] - d.y[i - 1]) / 2);
      d.width = best > 1e-9 ? 0.8 * (HI - LO) / best : NaN;
    });

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── top: five stacked cross-sections ── */
    const px = 34, pw = W - 60, ph = 46;
    const sx = d3.scaleLinear().domain([60, 140]).range([px, px + pw]);
    res.forEach((d, k) => {
      const y0 = 24 + k * (ph + 6);
      g.append("rect").attr("x", px).attr("y", y0).attr("width", pw).attr("height", ph)
        .attr("fill", VC.panel2).attr("stroke", VC.line);
      const sy = d3.scaleLinear().domain([-0.05, 1.05]).range([y0 + ph - 4, y0 + 4]);
      let dc = "", dd = "";
      for (let i = 60; i <= 140; i++) { dc += (i === 60 ? "M" : "L") + sx(i).toFixed(1) + "," + sy(clean[i]).toFixed(1); dd += (i === 60 ? "M" : "L") + sx(i).toFixed(1) + "," + sy(d.y[i]).toFixed(1); }
      g.append("path").attr("d", dc).attr("fill", "none").attr("stroke", VC.good).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.8);
      g.append("path").attr("d", dd).attr("fill", "none").attr("stroke", d.c).attr("stroke-width", 1.7);
      g.append("text").attr("x", px + 6).attr("y", y0 + 12).attr("font-size", 10).attr("fill", d.c).text(d.n);
      g.append("text").attr("x", px + pw - 6).attr("y", y0 + 12).attr("text-anchor", "end").attr("font-size", 9.5)
        .attr("fill", VC.muted).attr("font-family", "SF Mono, Menlo, monospace")
        .text("sd " + VZ.fmt(d.noise, 4) + "   width " + (isNaN(d.width) ? "—" : VZ.fmt(d.width, 2)) + "   rmse " + VZ.fmt(d.rmse, 4));
    });

    /* ── bottom: grouped bars ── */
    const by0 = 24 + 5 * (ph + 6) + 12, bh = H - by0 - 26;
    const groups = [["residual noise sd", "noise", true], ["10–90 edge width", "width", true], ["RMSE", "rmse", true]];
    const gw = (W - 70) / 3;
    groups.forEach((grp, gi) => {
      const x0 = 34 + gi * gw;
      g.append("rect").attr("x", x0).attr("y", by0).attr("width", gw - 12).attr("height", bh)
        .attr("fill", VC.panel2).attr("stroke", VC.line);
      IP.cap(g, x0 + 4, by0 - 6, grp[0], VC.ink);
      const vals = res.map(d => d[grp[1]]).map(v => isNaN(v) ? 0 : v);
      const mx = Math.max(...vals, 1e-9);
      let best = 1; for (let k = 1; k < res.length; k++) if (vals[k] < vals[best]) best = k;
      const bw = (gw - 30) / res.length;
      res.forEach((d, k) => {
        const hgt = (vals[k] / mx) * (bh - 34);
        g.append("rect").attr("x", x0 + 8 + k * bw).attr("y", by0 + bh - 18 - hgt)
          .attr("width", bw - 4).attr("height", Math.max(1, hgt))
          .attr("fill", d.c).attr("fill-opacity", k === best ? 1 : 0.45)
          .attr("stroke", k === best ? VC.good : "none").attr("stroke-width", 1.4);
        g.append("text").attr("x", x0 + 8 + k * bw + (bw - 4) / 2).attr("y", by0 + bh - 6)
          .attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", VC.muted)
          .text(isNaN(vals[k]) ? "—" : VZ.fmt(vals[k], grp[1] === "width" ? 2 : 3));
      });
    });

    const winner = k => { let b = 1; for (let m = 1; m < res.length; m++) if (res[m][k] < res[b][k]) b = m; return res[b].n; };
    out.innerHTML =
      `${nt === "gauss" ? "Gaussian noise, sd " + VZ.fmt(nz, 3) : nt === "shot" ? "shot noise, 5% of samples replaced by 0 or 1" : "Gaussian noise sd " + VZ.fmt(nz, 3) + " plus 5% shot noise"} · `
      + `edge softness ${soft.toFixed(1)} px · window K = ${K} / <span class="keep">σ</span> = ${wid.toFixed(2)}<br>`
      + `best residual noise: <b>${winner("noise")}</b> · sharpest edge: <b>${winner("width")}</b> · best RMSE: <b>${winner("rmse")}</b><br>`
      + (nt !== "gauss"
        ? `With impulsive noise present the bilateral filter's RMSE is <b>${VZ.fmt(res[4].rmse, 5)}</b> against the unfiltered <b>${VZ.fmt(res[0].rmse, 5)}</b> — it removes only ${VZ.fmt(100 * (1 - res[4].rmse / res[0].rmse), 1)}% of the error, because an outlier is exactly the kind of pixel its range kernel refuses to average with. The median's RMSE is <b>${VZ.fmt(res[3].rmse, 5)}</b>, i.e. ${VZ.fmt(100 * (1 - res[3].rmse / res[0].rmse), 1)}% removed.`
        : `The <b>median's edge width equals the input's</b> (${VZ.fmt(res[3].width, 2)} against ${VZ.fmt(res[0].width, 2)}), which is why its RMSE is hard to beat here — it removed noise without touching the transition at all. The bilateral filter is useful only while <span class="keep">σ</span>_r = ${sr.toFixed(2)} sits above the noise level ${VZ.fmt(nz, 3)} and below the step height ${VZ.fmt(HI - LO, 2)}: drag it below the noise and it becomes the identity, above the step and it becomes the Gaussian.`);
  }

  [[eN, eNv, 3], [eE, eEv, 1], [eW, eWv, 2], [eSr, eSrv, 2]].forEach(([el, lbl, dp]) => {
    el.oninput = () => { lbl.textContent = (+el.value).toFixed(dp); draw(); };
  });
  eNt.onchange = draw;
  draw();
})();

/* ───────── 13 · morphology as one count map plus a threshold ───────── */
(function () {
  const svg = d3.select("#morph-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440, IW = 48, IH = 36;
  const eSE = document.getElementById("mp-se"), eOp = document.getElementById("mp-op");
  const eT = document.getElementById("mp-t"), eTv = document.getElementById("mp-tv");
  const eX = document.getElementById("mp-x"), eY = document.getElementById("mp-y");
  const out = document.getElementById("morph-readout");

  /* a shape with: a solid blob, a thin bridge, a hole, a hairline crack, three specks */
  function shape() {
    const A = IP.zeros2(IH, IW);
    const set = (i0, i1, j0, j1, v) => { for (let i = i0; i < i1; i++) for (let j = j0; j < j1; j++) if (i >= 0 && i < IH && j >= 0 && j < IW) A[i][j] = v; };
    set(6, 22, 6, 20, 1);                       // the blob
    set(12, 14, 20, 30, 1);                     // a thin bridge, 2 px tall
    set(6, 22, 30, 42, 1);                      // a second blob
    set(12, 17, 10, 15, 0);                     // a hole
    set(9, 19, 35, 36, 0);                      // a hairline crack, 1 px wide
    A[27][10] = 1; A[28][10] = 1; A[27][11] = 1;// a 3-pixel speck
    A[30][22] = 1;                              // a 1-pixel speck
    set(26, 32, 32, 38, 1);                     // a small square that survives
    return A;
  }
  function elem(name) {
    const mk = (n, fn) => { const S = []; const R = (n - 1) >> 1; for (let i = -R; i <= R; i++) { const r = new Float64Array(n); for (let j = -R; j <= R; j++) r[j + R] = fn(i, j) ? 1 : 0; S.push(r); } return S; };
    switch (name) {
      case "sq3": return { S: mk(3, () => true), n: "3 × 3 square" };
      case "sq5": return { S: mk(5, () => true), n: "5 × 5 square" };
      case "cross": return { S: mk(3, (i, j) => Math.abs(i) + Math.abs(j) <= 1), n: "3 × 3 cross" };
      case "disc": return { S: mk(5, (i, j) => i * i + j * j <= 4.5), n: "disc, radius 2" };
      case "hline": return { S: mk(5, (i) => i === 0), n: "horizontal line 1 × 5" };
      case "vline": return { S: mk(5, (i, j) => j === 0), n: "vertical line 5 × 1" };
    }
  }
  const count = (A, S) => IP.corr2(A, S, "zero");
  const thr = (C, t) => IP.map2(C, v => v >= t ? 1 : 0);
  const inv = A => IP.map2(A, v => 1 - v);

  function draw() {
    const A = shape(), E = elem(eSE.value), S = E.S, sz = IP.sum2(S);
    eT.max = Math.max(1, sz);
    let t = VZ.clamp(+eT.value, 1, sz);
    const C = count(A, S);
    const dil = thr(C, 1), ero = thr(C, sz);
    const opn = thr(count(ero, S), 1);
    const cls = thr(count(dil, S), sz);                    // erode(dilate(f, S), S)
    let B, opName;
    switch (eOp.value) {
      case "count": B = thr(C, t); opName = "θ(c, " + t + ")"; break;
      case "dilate": B = dil; opName = "dilation"; t = 1; break;
      case "erode": B = ero; opName = "erosion"; t = sz; break;
      case "maj": B = thr(C, Math.ceil(sz / 2)); opName = "majority"; t = Math.ceil(sz / 2); break;
      case "open": B = opn; opName = "opening"; break;
      case "close": B = cls; opName = "closing"; break;
      case "grad": B = IP.map2(dil, (v, i, j) => v - ero[i][j]); opName = "morphological gradient"; break;
      case "tophat": B = IP.map2(A, (v, i, j) => v - opn[i][j]); opName = "top-hat"; break;
      default: {
        /* hit-or-miss with a convex-corner template: FG at (0,0),(0,1),(1,0); BG at (−1,·),(·,−1) */
        const S1 = [[0, 0, 0], [0, 1, 1], [0, 1, 1]].map(r => Float64Array.from(r));
        const S2 = [[1, 1, 1], [1, 0, 0], [1, 0, 0]].map(r => Float64Array.from(r));
        const e1 = thr(count(A, S1), IP.sum2(S1)), e2 = thr(count(inv(A), S2), IP.sum2(S2));
        B = IP.map2(e1, (v, i, j) => v * e2[i][j]); opName = "hit-or-miss (top-left convex corners)";
      }
    }

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell = 4.6, gap = 18;
    const px = 12, mx = px + IW * cell + gap, rx = mx + IW * cell + gap;
    const pi = Math.round(VZ.clamp(+eY.value, 0, IH - 1)), pj = Math.round(VZ.clamp(+eX.value, 0, IW - 1));

    /* input */
    IP.cap(g, px, 18, "input, |f| = " + IP.sum2(A) + " pixels", VC.ink);
    IP.raster(g, A, px, 24, cell, { lo: 0, hi: 1 });
    /* the element at the probe */
    const R = (S.length - 1) >> 1;
    for (let i = -R; i <= R; i++) for (let j = -R; j <= R; j++) if (S[i + R][j + R]) {
      g.append("rect").attr("x", px + (pj + j) * cell).attr("y", 24 + (pi + i) * cell)
        .attr("width", cell).attr("height", cell).attr("fill", VC.a2).attr("fill-opacity", 0.42);
    }
    g.append("rect").attr("x", px + (pj - R) * cell).attr("y", 24 + (pi - R) * cell)
      .attr("width", (2 * R + 1) * cell).attr("height", (2 * R + 1) * cell)
      .attr("fill", "none").attr("stroke", VC.a2);
    IP.cap(g, px, 24 + IH * cell + 15, E.n + ", |S| = " + sz, VC.a2);

    /* the count map */
    IP.cap(g, mx, 18, "the count map c = f ⊗ S", VC.ink);
    const gg = g.append("g").attr("transform", `translate(${mx},24)`);
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      const u = C[i][j] / sz;
      gg.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
        .attr("fill", d3.interpolateViridis(u)).attr("shape-rendering", "crispEdges");
    }
    gg.append("rect").attr("x", 0).attr("y", 0).attr("width", IW * cell).attr("height", IH * cell)
      .attr("fill", "none").attr("stroke", VC.line);
    /* the colour bar with the threshold marked */
    const cbw = IW * cell;
    for (let k = 0; k <= 60; k++)
      g.append("rect").attr("x", mx + k * cbw / 61).attr("y", 24 + IH * cell + 8)
        .attr("width", cbw / 61 + 0.6).attr("height", 9).attr("fill", d3.interpolateViridis(k / 60));
    const tp = mx + (t / sz) * cbw;
    g.append("line").attr("x1", tp).attr("x2", tp).attr("y1", 24 + IH * cell + 4).attr("y2", 24 + IH * cell + 21)
      .attr("stroke", VC.bad).attr("stroke-width", 2);
    IP.cap(g, mx, 24 + IH * cell + 32, "c = 0 … " + sz + "   ·   threshold t = " + t + "   ·   c at the probe = " + C[pi][pj], VC.muted);

    /* the result, with added / removed marked */
    IP.cap(g, rx, 18, opName, VC.ink);
    const gr = g.append("g").attr("transform", `translate(${rx},24)`);
    let added = 0, removed = 0;
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      const v = B[i][j], a = A[i][j];
      let col = v > 0.5 ? "#ffffff" : "#000000";
      if (v > 0.5 && a < 0.5) { col = VC.good; added++; }
      else if (v < 0.5 && a > 0.5) { col = VC.bad; removed++; }
      gr.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
        .attr("fill", col).attr("shape-rendering", "crispEdges");
    }
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", IW * cell).attr("height", IH * cell)
      .attr("fill", "none").attr("stroke", VC.line);
    VZ.legend(g, [{ color: VC.good, label: "added" }, { color: VC.bad, label: "removed" }],
      rx, 24 + IH * cell + 16, { vertical: false, step: 76, font: 10 });

    /* did the specks survive? did the holes fill? */
    const speck1 = B[27][10] > 0.5, speck2 = B[30][22] > 0.5, sq = B[28][34] > 0.5;
    const holeFilled = B[14][12] > 0.5, crack = B[14][35] > 0.5;

    out.innerHTML =
      `${E.n}, |S| = ${sz} · operator ${opName} · at the probe (${pi}, ${pj}) the count is <b>${C[pi][pj]}</b> of ${sz}`
      + (["count", "dilate", "erode", "maj"].indexOf(eOp.value) >= 0
        ? `, so θ(c, ${t}) = <b>${C[pi][pj] >= t ? 1 : 0}</b>`
        : ` — and the selected operator is a <i>composition</i> of two thresholded counts, so no single t describes it`)
      + `<br>`
      + `pixels added <b>${added}</b>, removed <b>${removed}</b>; |f| = ${IP.sum2(A)} → |g| = ${IP.sum2(IP.map2(B, v => v > 0.5 ? 1 : 0))}<br>`
      + `the 3-pixel speck ${speck1 ? "<b>survived</b>" : "was <b>removed</b>"} · the 1-pixel speck ${speck2 ? "<b>survived</b>" : "was <b>removed</b>"} · `
      + `the 6 × 6 square ${sq ? "<b>survived</b>" : "was <b>removed</b>"} · the hole ${holeFilled ? "was <b>filled</b>" : "is still <b>open</b>"} · `
      + `the 1-pixel crack ${crack ? "was <b>sealed</b>" : "is still <b>open</b>"}<br>`
      + `An object survives an opening exactly when the structuring element fits entirely inside it — which is why the ${sz > 9 ? "5 × 5" : "small"} element ${sq ? "leaves" : "erases"} the square and ${speck1 ? "leaves" : "erases"} the speck.`;
  }

  eSE.onchange = () => { draw(); };
  eOp.onchange = draw;
  eT.oninput = () => { eTv.textContent = eT.value; draw(); };
  eX.oninput = draw; eY.oninput = draw;
  draw();
})();

/* ───────── 14 · the two-pass distance transform, and connected components ───────── */
(function () {
  const svg = d3.select("#dt-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440, IW = 26, IH = 20;
  const eMode = document.getElementById("dt-mode"), eM = document.getElementById("dt-m");
  const eP = document.getElementById("dt-p"), ePv = document.getElementById("dt-pv");
  const eC = document.getElementById("dt-c"), eNum = document.getElementById("dt-num");
  const out = document.getElementById("dt-readout");

  function shape() {
    const A = IP.zeros2(IH, IW);
    const set = (i0, i1, j0, j1) => { for (let i = i0; i < i1; i++) for (let j = j0; j < j1; j++) A[i][j] = 1; };
    set(2, 9, 2, 11); set(5, 7, 11, 16); set(3, 10, 16, 22);
    set(12, 17, 3, 8);
    A[13][10] = 1; A[14][11] = 1; A[15][12] = 1; A[16][13] = 1;   // a diagonal chain
    set(13, 17, 16, 22);
    return A;
  }

  /* the two raster sweeps, exactly as the prose writes them */
  function dt1(b, pass) {
    const D = IP.map2(b, v => v > 0.5 ? 1e9 : 0);
    if (pass >= 1) for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) if (b[i][j] > 0.5) {
      let m = D[i][j];
      if (i > 0) m = Math.min(m, D[i - 1][j] + 1);
      if (j > 0) m = Math.min(m, D[i][j - 1] + 1);
      D[i][j] = m;
    }
    if (pass >= 2) for (let i = IH - 1; i >= 0; i--) for (let j = IW - 1; j >= 0; j--) if (b[i][j] > 0.5) {
      let m = D[i][j];
      if (i < IH - 1) m = Math.min(m, D[i + 1][j] + 1);
      if (j < IW - 1) m = Math.min(m, D[i][j + 1] + 1);
      D[i][j] = m;
    }
    return D;
  }
  function dtCham(b, pass) {                      /* 3–4 chamfer, scaled back to pixels by /3 */
    const D = IP.map2(b, v => v > 0.5 ? 1e9 : 0);
    const fwd = [[-1, -1, 4], [-1, 0, 3], [-1, 1, 4], [0, -1, 3]];
    const bwd = [[1, 1, 4], [1, 0, 3], [1, -1, 4], [0, 1, 3]];
    const sweep = (nb, order) => {
      const is = order > 0 ? [...Array(IH).keys()] : [...Array(IH).keys()].reverse();
      const js = order > 0 ? [...Array(IW).keys()] : [...Array(IW).keys()].reverse();
      for (const i of is) for (const j of js) if (b[i][j] > 0.5) {
        let m = D[i][j];
        for (const [di, dj, w] of nb) { const ii = i + di, jj = j + dj; if (ii >= 0 && ii < IH && jj >= 0 && jj < IW) m = Math.min(m, D[ii][jj] + w); }
        D[i][j] = m;
      }
    };
    if (pass >= 1) sweep(fwd, 1);
    if (pass >= 2) sweep(bwd, -1);
    return IP.map2(D, v => v >= 1e8 ? 1e9 : v / 3);
  }
  function dtExact(b) {                            /* brute force, so it is unarguably exact */
    const bg = [];
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) if (b[i][j] < 0.5) bg.push([i, j]);
    return IP.map2(b, (v, i, j) => {
      if (v < 0.5) return 0;
      let m = Infinity;
      for (const [k, l] of bg) m = Math.min(m, (i - k) ** 2 + (j - l) ** 2);
      return Math.sqrt(m);
    });
  }
  function label(b, conn) {
    const L = IP.map2(b, () => 0), nb4 = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    const nb8 = nb4.concat([[-1, -1], [-1, 1], [1, -1], [1, 1]]);
    const nb = conn === 8 ? nb8 : nb4;
    let n = 0;
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      if (b[i][j] < 0.5 || L[i][j] > 0) continue;
      n++; const st = [[i, j]]; L[i][j] = n;
      while (st.length) {
        const [a, c] = st.pop();
        for (const [di, dj] of nb) {
          const ii = a + di, jj = c + dj;
          if (ii < 0 || ii >= IH || jj < 0 || jj >= IW) continue;
          if (b[ii][jj] > 0.5 && L[ii][jj] === 0) { L[ii][jj] = n; st.push([ii, jj]); }
        }
      }
    }
    return { L, n };
  }

  function draw() {
    const b = shape(), pass = +eP.value, conn = +eC.value, mode = eMode.value;
    ePv.textContent = ["initial only", "forward only", "both passes"][pass];
    const Dc = eM.value === "cham" ? dtCham(b, pass) : dt1(b, pass);
    const De = dtExact(b);
    const D = eM.value === "d2" ? (pass === 2 ? De : Dc) : Dc;
    const { L, n } = label(b, conn);
    const { n: n4 } = label(b, 4), { n: n8 } = label(b, 8);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell = 13, gap = 16;
    const x0 = 12, x1 = x0 + IW * cell + gap, x2 = x1 + IW * cell + gap;

    IP.cap(g, x0, 18, "binary input b", VC.ink);
    IP.raster(g, b, x0, 24, cell, { lo: 0, hi: 1, grid: true });

    /* middle: the distance transform */
    IP.cap(g, x1, 18, "D, after " + ["no sweep", "the forward sweep", "both sweeps"][pass], VC.ink);
    const finite = [];
    for (const r of D) for (const v of r) if (v < 1e8) finite.push(v);
    const dmax = Math.max(...finite, 1);
    const gm = g.append("g").attr("transform", `translate(${x1},24)`);
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      const v = D[i][j];
      gm.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
        .attr("fill", v >= 1e8 ? VC.bad : d3.interpolateMagma(0.12 + 0.8 * v / dmax))
        .attr("shape-rendering", "crispEdges");
      if (eNum.checked && b[i][j] > 0.5)
        gm.append("text").attr("x", j * cell + cell / 2).attr("y", i * cell + cell / 2 + 3.2)
          .attr("text-anchor", "middle").attr("font-size", 7.6)
          .attr("fill", v >= 1e8 ? "#fff" : (v / dmax > 0.55 ? "#000" : "#fff"))
          .text(v >= 1e8 ? "∞" : (eM.value === "d1" ? v : VZ.fmt(v, 1)));
    }
    gm.append("rect").attr("width", IW * cell).attr("height", IH * cell).attr("fill", "none").attr("stroke", VC.line);

    /* right: comparison or components */
    if (mode === "cc") {
      IP.cap(g, x2, 18, conn + "-connected components: " + n, VC.ink);
      const gr = g.append("g").attr("transform", `translate(${x2},24)`);
      for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
        const l = L[i][j];
        gr.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
          .attr("fill", l === 0 ? "#000" : d3.schemeTableau10[(l - 1) % 10]).attr("shape-rendering", "crispEdges");
      }
      gr.append("rect").attr("width", IW * cell).attr("height", IH * cell).attr("fill", "none").attr("stroke", VC.line);
      /* per-component statistics */
      const rows = ["  #   area    centroid        elong   compact"];
      for (let k = 1; k <= Math.min(n, 6); k++) {
        let a = 0, sx = 0, sy = 0;
        for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) if (L[i][j] === k) { a++; sx += j; sy += i; }
        const cxm = sx / a, cym = sy / a;
        let mxx = 0, myy = 0, mxy = 0, per = 0;
        for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) if (L[i][j] === k) {
          mxx += (j - cxm) ** 2; myy += (i - cym) ** 2; mxy += (j - cxm) * (i - cym);
          let boundary = false;
          for (const [di, dj] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            const ii = i + di, jj = j + dj;
            if (ii < 0 || ii >= IH || jj < 0 || jj >= IW || L[ii][jj] !== k) boundary = true;
          }
          if (boundary) per++;
        }
        mxx /= a; myy /= a; mxy /= a;
        const tr = mxx + myy, dd = Math.sqrt(Math.max(0, (mxx - myy) ** 2 + 4 * mxy * mxy));
        const l1 = (tr + dd) / 2, l2 = (tr - dd) / 2;
        rows.push(("  " + k).padEnd(5) + String(a).padStart(4)
          + ("  (" + VZ.fmt(cxm, 1) + ", " + VZ.fmt(cym, 1) + ")").padEnd(16)
          + VZ.fmt(l2 > 1e-9 ? Math.sqrt(l1 / l2) : 0, 2).padStart(6)
          + VZ.fmt(per > 0 ? 4 * Math.PI * a / (per * per) : 0, 3).padStart(9));
      }
      IP.numText(g, rows, x2, 34 + IH * cell, { size: 9.2, lead: 12 });
      out.innerHTML =
        `4-connected components: <b>${n4}</b> · 8-connected: <b>${n8}</b>. `
        + `The diagonal chain in the lower half is <b>${n8 < n4 ? "one object under 8-connectivity and " + (n4 - n8 + 1) + " under 4" : "unbroken"}</b>. `
        + `Neither is wrong — but if the foreground uses 8-connectivity the background must use 4, or a closed curve stops separating inside from outside.<br>`
        + `Compactness is 4π·area/perimeter², which is 1 for a perfect disc and falls for anything elongated or ragged; elongation is √(<span class="keep">λ</span>₁/<span class="keep">λ</span>₂) from the second-moment matrix.`;
    } else if (mode === "skel") {
      IP.cap(g, x2, 18, "ridges of D — the medial axis", VC.ink);
      const gr = g.append("g").attr("transform", `translate(${x2},24)`);
      const Df = dt1(b, 2);
      for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
        let ridge = false;
        if (b[i][j] > 0.5) {
          let ge = 0;
          for (const [di, dj] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            const ii = i + di, jj = j + dj;
            const v = (ii < 0 || ii >= IH || jj < 0 || jj >= IW) ? 0 : Df[ii][jj];
            if (Df[i][j] >= v) ge++;
          }
          ridge = ge === 4 && Df[i][j] >= 1;
        }
        gr.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
          .attr("fill", ridge ? VC.a2 : (b[i][j] > 0.5 ? "#2a3140" : "#000")).attr("shape-rendering", "crispEdges");
      }
      gr.append("rect").attr("width", IW * cell).attr("height", IH * cell).attr("fill", "none").attr("stroke", VC.line);
      out.innerHTML = `A ridge pixel is a local maximum of D — equidistant from two or more boundaries. `
        + `That set is the <b>medial axis</b>, and D restricted to it records the radius of the largest disc that fits: `
        + `the pair (skeleton, radius) reconstructs the shape exactly. This crude local-maximum test is enough to show the ridges; `
        + `a usable skeletonisation needs the hit-or-miss thinning of §18 to guarantee connectivity.`;
    } else {
      IP.cap(g, x2, 18, "exact Euclidean D, and the error of d₁", VC.ink);
      const gr = g.append("g").attr("transform", `translate(${x2},24)`);
      const Dfull = dt1(b, 2);
      let worst = 0, wi = 0, wj = 0;
      for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
        const e = b[i][j] > 0.5 ? Dfull[i][j] - De[i][j] : 0;
        if (e > worst) { worst = e; wi = i; wj = j; }
        gr.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
          .attr("fill", b[i][j] < 0.5 ? "#000" : d3.interpolateOranges(VZ.clamp(e / 2.2, 0, 1)))
          .attr("shape-rendering", "crispEdges");
        if (eNum.checked && b[i][j] > 0.5)
          gr.append("text").attr("x", j * cell + cell / 2).attr("y", i * cell + cell / 2 + 3.2)
            .attr("text-anchor", "middle").attr("font-size", 7.4).attr("fill", e / 2.2 > 0.6 ? "#000" : "#fff")
            .text(VZ.fmt(e, 1));
      }
      gr.append("rect").attr("width", IW * cell).attr("height", IH * cell).attr("fill", "none").attr("stroke", VC.line);
      gr.append("rect").attr("x", wj * cell).attr("y", wi * cell).attr("width", cell).attr("height", cell)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
      /* chamfer error too */
      const Dch = dtCham(b, 2);
      let wc = 0; for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) if (b[i][j] > 0.5) wc = Math.max(wc, Math.abs(Dch[i][j] - De[i][j]));
      out.innerHTML =
        `the two-pass city-block transform is exact <b>for d₁</b> but over-estimates the Euclidean distance: `
        + `the worst over-estimate on this shape is <b>${VZ.fmt(worst, 4)}</b> pixels, at the ringed cell. `
        + `The bound is a factor of √2 = 1.4142, attained on a pure diagonal.<br>`
        + `The 3–4 chamfer approximation, still two passes and still integer arithmetic, cuts the worst error to <b>${VZ.fmt(wc, 4)}</b>. `
        + `The exact Euclidean transform needs a <b>vector</b> propagated per pixel, not a scalar — it is a different algorithm, not a change of weights.`;
    }
  }

  eMode.onchange = draw; eM.onchange = draw; eC.onchange = draw; eNum.onchange = draw;
  eP.oninput = draw;
  draw();
})();

/* ───────── 15 · the integral image ───────── */
(function () {
  const svg = d3.select("#integ-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430;
  const eSrc = document.getElementById("ig-src");
  const eI0 = document.getElementById("ig-i0"), eHh = document.getElementById("ig-h");
  const eJ0 = document.getElementById("ig-j0"), eWw = document.getElementById("ig-w");
  const eBox = document.getElementById("ig-box");
  const out = document.getElementById("integ-readout");

  const DEMO = [[3, 2, 7, 2, 3], [1, 5, 1, 3, 4], [5, 1, 3, 5, 1], [4, 3, 2, 1, 6], [2, 4, 1, 4, 8]]
    .map(r => Float64Array.from(r));

  function source() {
    if (eSrc.value === "demo") return DEMO.map(r => Float64Array.from(r));
    if (eSrc.value === "patch") return IP.clone2(IP.PATCH);
    return IP.map2(IP.scene(40, 30, { noise: 0.06, seed: 3 }), v => Math.round(255 * v));
  }
  function integral(A) {
    const h = A.length, w = A[0].length, s = IP.zeros2(h, w);
    for (let i = 0; i < h; i++) for (let j = 0; j < w; j++)
      s[i][j] = A[i][j] + (i ? s[i - 1][j] : 0) + (j ? s[i][j - 1] : 0) - (i && j ? s[i - 1][j - 1] : 0);
    return s;
  }
  const query = (s, i0, i1, j0, j1) =>
    s[i1][j1] - (i0 ? s[i0 - 1][j1] : 0) - (j0 ? s[i1][j0 - 1] : 0) + (i0 && j0 ? s[i0 - 1][j0 - 1] : 0);

  function draw() {
    const A = source(), h = A.length, w = A[0].length;
    eI0.max = h - 1; eJ0.max = w - 1; eHh.max = h; eWw.max = w;
    const i0 = Math.round(VZ.clamp(+eI0.value, 0, h - 1)), j0 = Math.round(VZ.clamp(+eJ0.value, 0, w - 1));
    const i1 = Math.round(VZ.clamp(i0 + (+eHh.value) - 1, i0, h - 1)), j1 = Math.round(VZ.clamp(j0 + (+eWw.value) - 1, j0, w - 1));
    const s = integral(A);
    const S = query(s, i0, i1, j0, j1);
    let direct = 0, cells = 0;
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) { direct += A[i][j]; cells++; }

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    if (eBox.checked) {
      /* the whole image box-filtered at the current rectangle size, straight from the table */
      const rh = i1 - i0, rw = j1 - j0;
      const B = IP.zeros2(h, w);
      let maxErr = 0;
      for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
        const a0 = Math.max(0, i - (rh >> 1)), a1 = Math.min(h - 1, i + (rh >> 1));
        const b0 = Math.max(0, j - (rw >> 1)), b1 = Math.min(w - 1, j + (rw >> 1));
        const n = (a1 - a0 + 1) * (b1 - b0 + 1);
        B[i][j] = query(s, a0, a1, b0, b1) / n;
        let dsum = 0; for (let p = a0; p <= a1; p++) for (let q = b0; q <= b1; q++) dsum += A[p][q];
        maxErr = Math.max(maxErr, Math.abs(B[i][j] - dsum / n));
      }
      const cell = Math.min(11, 300 / Math.max(h, w));
      IP.cap(g, 20, 18, "input", VC.accent);
      IP.raster(g, A, 20, 24, cell, {});
      IP.cap(g, 40 + w * cell, 18, "box filtered " + (rh + 1) + " × " + (rw + 1) + ", entirely from the table", VC.a2);
      IP.raster(g, B, 40 + w * cell, 24, cell, {});
      out.innerHTML = `Every output pixel above is <b>three additions and one division</b>, whatever the window size. `
        + `Checked against the direct sum at all ${h * w} pixels: maximum absolute difference <b>${maxErr.toExponential(2)}</b> — exact.<br>`
        + `A direct ${rh + 1} × ${rw + 1} box would be ${(rh + 1) * (rw + 1)} additions per pixel; a separable one ${(rh + 1) + (rw + 1)}; the table needs 3 to build and 3 to query.`;
      return;
    }

    const cell = Math.min(26, 250 / Math.max(w, h));
    const fs = Math.min(10, cell * 0.42);
    function grid(x, M, title, hl) {
      IP.cap(g, x, 18, title, VC.ink);
      const gg = g.append("g").attr("transform", `translate(${x},24)`);
      for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
        const m = hl(i, j);
        gg.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell).attr("height", cell)
          .attr("fill", m ? m.fill : VC.panel2).attr("stroke", VC.line).attr("stroke-width", 0.5);
        if (cell >= 14)
          gg.append("text").attr("x", j * cell + cell / 2).attr("y", i * cell + cell / 2 + fs * 0.36)
            .attr("text-anchor", "middle").attr("font-size", fs)
            .attr("font-family", "SF Mono, Menlo, monospace")
            .attr("fill", m && m.ink ? m.ink : VC.muted).text(Math.round(M[i][j]));
      }
      return gg;
    }
    grid(14, A, "image f  ·  rectangle rows " + i0 + "–" + i1 + ", cols " + j0 + "–" + j1,
      (i, j) => (i >= i0 && i <= i1 && j >= j0 && j <= j1)
        ? { fill: "rgba(91,156,255,0.28)", ink: VC.ink } : null);
    const cornerX = 14 + w * cell + 30;
    grid(cornerX, s, "summed-area table s", (i, j) => {
      if (i === i1 && j === j1) return { fill: "rgba(74,222,128,0.42)", ink: VC.ink };
      if ((i === i0 - 1 && j === j1) || (i === i1 && j === j0 - 1)) return { fill: "rgba(248,113,113,0.42)", ink: VC.ink };
      if (i === i0 - 1 && j === j0 - 1) return { fill: "rgba(74,222,128,0.42)", ink: VC.ink };
      return null;
    });

    const tx = cornerX + w * cell + 26;
    const a = s[i1][j1], b2 = i0 ? s[i0 - 1][j1] : 0, c = j0 ? s[i1][j0 - 1] : 0, d = (i0 && j0) ? s[i0 - 1][j0 - 1] : 0;
    IP.numText(g, [
      "the four-corner query",
      "",
      "  s(" + i1 + "," + j1 + ")      = " + String(Math.round(a)).padStart(7),
      "− s(" + (i0 - 1) + "," + j1 + ")      = " + String(Math.round(b2)).padStart(7),
      "− s(" + i1 + "," + (j0 - 1) + ")      = " + String(Math.round(c)).padStart(7),
      "+ s(" + (i0 - 1) + "," + (j0 - 1) + ")      = " + String(Math.round(d)).padStart(7),
      "                    ───────",
      "  S               = " + String(Math.round(S)).padStart(7),
      "",
      "  direct sum      = " + String(Math.round(direct)).padStart(7),
      "  difference      = " + String(Math.round(S - direct)).padStart(7),
      "",
      "  cells in rect   = " + String(cells).padStart(7),
      "  additions, direct " + String(cells - 1).padStart(5),
      "  additions, table  " + String(3).padStart(5),
      "  mean            = " + VZ.fmt(S / cells, 4).padStart(7)
    ], tx, 34, { size: 10.5, lead: 14 });

    /* check every rectangle, once */
    let bad = 0, tested = 0;
    for (let p0 = 0; p0 < h; p0++) for (let p1 = p0; p1 < h; p1++) for (let q0 = 0; q0 < w; q0++) for (let q1 = q0; q1 < w; q1++) {
      if (tested > 6000) break;
      let ds = 0; for (let i = p0; i <= p1; i++) for (let j = q0; j <= q1; j++) ds += A[i][j];
      if (Math.abs(query(s, p0, p1, q0, q1) - ds) > 1e-9) bad++;
      tested++;
    }
    out.innerHTML =
      `rectangle ${i1 - i0 + 1} × ${j1 - j0 + 1} = ${cells} cells · four-corner query <b>${Math.round(S)}</b> · `
      + `direct sum <b>${Math.round(direct)}</b> · difference <b>${Math.round(S - direct)}</b><br>`
      + `${tested.toLocaleString()} rectangles checked against their direct sums: <b>${bad}</b> disagreed. `
      + `The query cost is <b>3 additions</b> regardless of the ${cells} cells inside it; the direct sum needed ${cells - 1}.<br>`
      + `The largest table entry here is ${Math.round(s[h - 1][w - 1])}; on an 8-bit 1920 × 1080 image it would be 255 × 1920 × 1080 = 528 768 000, needing 29 bits — `
      + `safe in a 32-bit integer, <b>not</b> safe in a 32-bit float, whose mantissa is only 24 bits.`;
  }

  eSrc.onchange = draw; eBox.onchange = draw;
  [eI0, eHh, eJ0, eWw].forEach(el => el.oninput = draw);
  draw();
})();

/* ───────── 16 · decimation with and without pre-blurring ───────── */
(function () {
  const svg = d3.select("#dec-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470, N = 512;
  const eR = document.getElementById("dc-r"), eRv = document.getElementById("dc-rv");
  const eF = document.getElementById("dc-f");
  const eC = document.getElementById("dc-cmp");
  const out = document.getElementById("dec-readout");

  /* a linear chirp: local frequency rises from 0 to fmax cycles/sample */
  const FMAX = 0.45;
  function chirp1(n) {
    const s = new Float64Array(n);
    for (let i = 0; i < n; i++) s[i] = 0.5 + 0.45 * Math.cos(2 * Math.PI * FMAX * i * i / (2 * n));
    return s;
  }
  const localFreq = i => FMAX * i / N;

  function prefilter(r) {
    switch (eF.value) {
      case "none": return null;
      case "box": return IP.box(2 * Math.floor(r / 2) + 1);
      case "lin": return Float64Array.from([0.25, 0.5, 0.25]);
      case "binom": return Float64Array.from([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]);
      case "gauss": return IP.gauss1(0.5 * r, 3);
      case "sinc": {
        const R = 4 * r, h = new Float64Array(2 * R + 1); let s = 0;
        for (let k = -R; k <= R; k++) {
          const x = k / r;
          const sinc = (x === 0) ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
          const win = 0.5 * (1 + Math.cos(Math.PI * k / R));      // Hann
          h[k + R] = sinc * win; s += h[k + R];
        }
        for (let i = 0; i < h.length; i++) h[i] /= s;
        return h;
      }
    }
  }
  /* an ideal band-limited decimation, done by zeroing the DFT above the new Nyquist */
  function idealDecimate(sig, r) {
    const n = sig.length, half = Math.floor(n / (2 * r));
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let k = 0; k <= n / 2; k++) {
      if (k > half) continue;
      let a = 0, b = 0;
      for (let i = 0; i < n; i++) { const th = -2 * Math.PI * k * i / n; a += sig[i] * Math.cos(th); b += sig[i] * Math.sin(th); }
      re[k] = a; im[k] = b;
    }
    const outp = new Float64Array(Math.floor(n / r));
    for (let m = 0; m < outp.length; m++) {
      const i = m * r; let s = re[0] / n;
      for (let k = 1; k <= half; k++) {
        const th = 2 * Math.PI * k * i / n;
        s += 2 * (re[k] * Math.cos(th) - im[k] * Math.sin(th)) / n;
      }
      outp[m] = s;
    }
    return outp;
  }

  function draw() {
    const r = +eR.value, h = prefilter(r);
    const sig = chirp1(N);
    const filt = h ? IP.corr1(sig, h, "mirror") : sig;
    const naive = [], pre = [];
    for (let i = 0; i < N; i += r) { naive.push(sig[i]); pre.push(filt[i]); }
    const ideal = idealDecimate(sig, r);
    const m = Math.min(naive.length, pre.length, ideal.length);
    const rms = arr => { let s = 0; for (let i = 0; i < m; i++) s += (arr[i] - ideal[i]) ** 2; return Math.sqrt(s / m); };
    const eN2 = rms(naive), eP2 = rms(pre);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const px = 14, pw = W - 28;

    /* draw a 1D signal as a grey strip, one cell per sample, at a chosen zoom */
    function strip(y, arr, zoom, label, col) {
      IP.cap(g, px, y - 5, label, col || VC.ink);
      const gg = g.append("g").attr("transform", `translate(${px},${y})`);
      const n = arr.length, cw = pw / (N / 1);
      for (let i = 0; i < n; i++) {
        gg.append("rect").attr("x", i * cw * zoom).attr("y", 0)
          .attr("width", cw * zoom + 0.4).attr("height", 44)
          .attr("fill", IP.greyColor(VZ.clamp(arr[i], 0, 1), 1)).attr("shape-rendering", "crispEdges");
      }
      gg.append("rect").attr("x", 0).attr("y", 0).attr("width", pw).attr("height", 44)
        .attr("fill", "none").attr("stroke", VC.line);
      return gg;
    }
    strip(24, Array.from(sig), 1, "full resolution — local frequency rises to " + FMAX + " cycles/sample on the right", VC.accent);
    strip(96, Array.from(naive), r, "decimated by " + r + " with NO pre-filter", VC.bad);
    strip(168, Array.from(pre), r, "decimated by " + r + (h ? " after the " + eF.options[eF.selectedIndex].text.split(",")[0] + " pre-filter" : " — no filter selected, so this row is identical"), VC.good);

    /* the Nyquist crossing marker */
    const iCross = Math.round(N * (0.5 / r) / FMAX);
    if (iCross < N) [24, 96, 168].forEach(y => {
      g.append("line").attr("x1", px + pw * iCross / N).attr("x2", px + pw * iCross / N)
        .attr("y1", y - 2).attr("y2", y + 46).attr("stroke", VC.violet).attr("stroke-width", 1.4);
    });
    g.append("text").attr("x", px + pw * iCross / N + 4).attr("y", 20).attr("font-size", 10)
      .attr("fill", VC.violet).text("the new Nyquist limit, " + VZ.fmt(0.5 / r, 4) + " c/sample — everything right of here must go");

    /* the trace comparison */
    const ty = 244, th2 = H - ty - 46;
    g.append("rect").attr("x", px).attr("y", ty).attr("width", pw).attr("height", th2)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, ty - 5, "the decimated samples, against an ideally band-limited reference", VC.ink);
    const sx = d3.scaleLinear().domain([0, m - 1]).range([px + 8, px + pw - 8]);
    const sy = d3.scaleLinear().domain([-0.05, 1.05]).range([ty + th2 - 16, ty + 10]);
    const series = [];
    if (eC.checked) series.push([ideal, VC.muted, 2.2, "ideal band-limited"]);
    series.push([naive, VC.bad, 1.5, "no pre-filter"]);
    if (h) series.push([pre, VC.good, 1.5, "pre-filtered"]);
    series.forEach(([arr, col, wd]) => {
      let d = ""; for (let i = 0; i < m; i++) d += (i ? "L" : "M") + sx(i).toFixed(1) + "," + sy(arr[i]).toFixed(1);
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", col).attr("stroke-width", wd);
    });
    VZ.legend(g, series.map(s => ({ color: s[1], label: s[3] })), px + 12, ty + 16, { vertical: true, gap: 13, font: 9.5 });

    out.innerHTML =
      `rate r = ${r}, ${m} output samples · pre-filter: <b>${eF.options[eF.selectedIndex].text}</b><br>`
      + `RMS error against an ideally band-limited decimation — no pre-filter <b>${VZ.fmt(eN2, 5)}</b>`
      + (h ? `, pre-filtered <b>${VZ.fmt(eP2, 5)}</b>, a factor of <b>${VZ.fmt(eP2 > 0 ? eN2 / eP2 : Infinity, 2)}×</b>` : ` (no pre-filter selected, so there is nothing to compare it against)`)
      + `<br>`
      + (h
        ? `The error the naive version makes is not noise. It is <b>coherent structure at the wrong frequency</b>: content above ${VZ.fmt(0.5 / r, 4)} cycles/sample has folded back into the band and now looks exactly like real low-frequency detail. No later filter can remove it, because there is nothing left to distinguish it from.`
        : `Select a pre-filter to see the comparison. As it stands, everything to the right of the violet line in the middle strip is fabricated: the fine end of the chirp has folded down and reappears as coarse stripes running the wrong way.`);
  }

  eR.oninput = () => { eRv.textContent = eR.value; draw(); };
  eF.onchange = draw; eC.onchange = draw;
  draw();
})();

/* ───────── shared pyramid machinery for figures 17–19 ───────── */
const PYR = (function () {
  const K5 = Float64Array.from([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]);
  function kernelOf(name) {
    if (name === "tent") return Float64Array.from([0, 0.25, 0.5, 0.25, 0]);
    if (name === "a06") return Float64Array.from([-0.05, 0.25, 0.6, 0.25, -0.05]);
    if (name === "none") return Float64Array.from([0, 0, 1, 0, 0]);
    return K5;
  }
  function reduce(A, h) {
    const B = IP.sepV(IP.sepH(A, h, "mirror"), h, "mirror");
    const H2 = Math.ceil(A.length / 2), W2 = Math.ceil(A[0].length / 2);
    const C = IP.zeros2(H2, W2);
    for (let i = 0; i < H2; i++) for (let j = 0; j < W2; j++) C[i][j] = B[2 * i][2 * j];
    return C;
  }
  function expand(A, h, hh, ww) {
    const U = IP.zeros2(hh, ww);
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
      if (2 * i < hh && 2 * j < ww) U[2 * i][2 * j] = A[i][j];
    }
    const h2 = Float64Array.from(h, v => 2 * v);
    return IP.sepV(IP.sepH(U, h2, "mirror"), h2, "mirror");
  }
  function gaussPyr(A, n, h) {
    const G = [A];
    for (let l = 0; l < n; l++) G.push(reduce(G[G.length - 1], h));
    return G;
  }
  function lapPyr(G, h) {
    const L = [];
    for (let l = 0; l < G.length - 1; l++) {
      const E = expand(G[l + 1], h, G[l].length, G[l][0].length);
      L.push(G[l].map((r, i) => Float64Array.from(r, (v, j) => v - E[i][j])));
    }
    L.push(IP.clone2(G[G.length - 1]));
    return L;
  }
  function collapse(L, h) {
    let R = IP.clone2(L[L.length - 1]);
    for (let l = L.length - 2; l >= 0; l--) {
      const E = expand(R, h, L[l].length, L[l][0].length);
      R = L[l].map((r, i) => Float64Array.from(r, (v, j) => v + E[i][j]));
    }
    return R;
  }
  return { kernelOf, reduce, expand, gaussPyr, lapPyr, collapse };
})();

/* ───────── 17 · Gaussian and Laplacian pyramids, with the reconstruction error ───────── */
(function () {
  const svg = d3.select("#pyr-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 450, IW = 128, IH = 96;
  const eL = document.getElementById("py-l"), eLv = document.getElementById("py-lv");
  const eK = document.getElementById("py-k"), eS = document.getElementById("py-s");
  const eQ = document.getElementById("py-q"), eQv = document.getElementById("py-qv");
  const out = document.getElementById("pyr-readout");

  function draw() {
    const n = +eL.value, h = PYR.kernelOf(eK.value), q = +eQ.value;
    eQv.textContent = q === 0 ? "off" : "step " + VZ.fmt(1 / (1 << (7 - q)), 4);
    const A = IP.scene(IW, IH, { seed: 7 });
    const G = PYR.gaussPyr(A, n, h);
    let L = PYR.lapPyr(G, h);
    let quantErr = 0;
    if (q > 0) {
      const step = 1 / (1 << (7 - q));
      L = L.map(M => IP.map2(M, v => Math.round(v / step) * step));
    }
    const R = PYR.collapse(L, h);
    let maxe = 0, sse = 0;
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) { const e = Math.abs(R[i][j] - A[i][j]); maxe = Math.max(maxe, e); sse += e * e; }
    const rmse = Math.sqrt(sse / (IW * IH));

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell0 = 2.1;

    if (eS.value === "recon") {
      IP.cap(g, 12, 18, "original", VC.accent);
      IP.raster(g, A, 12, 24, cell0, { lo: 0, hi: 1 });
      const x2 = 12 + IW * cell0 + 20;
      IP.cap(g, x2, 18, "reconstructed from the pyramid", VC.good);
      IP.raster(g, R, x2, 24, cell0, { lo: 0, hi: 1 });
      const x3 = x2 + IW * cell0 + 20;
      const D = R.map((r, i) => Float64Array.from(r, (v, j) => v - A[i][j]));
      const m = Math.max(IP.absMax(D), 1e-18);
      IP.cap(g, x3, 18, "difference, at full contrast (peak " + m.toExponential(1) + ")", VC.bad);
      IP.raster(g, D, x3, 24, cell0, { signed: true, lo: -m, hi: m });
    } else {
      const stack = eS.value === "gauss" ? G : L;
      let x = 12;
      stack.forEach((M, l) => {
        const c = cell0 * Math.pow(2, 0);
        IP.cap(g, x, 18, (eS.value === "gauss" ? "G" : "L") + l + "  " + M[0].length + "×" + M.length, VC.muted);
        if (eS.value === "gauss") IP.raster(g, M, x, 24, c, { lo: 0, hi: 1 });
        else {
          const m = Math.max(IP.absMax(M), 1e-9);
          IP.raster(g, M, x, 24, c, l === stack.length - 1 ? { lo: 0, hi: 1 } : { signed: true, lo: -m, hi: m });
        }
        x += M[0].length * c + 12;
      });
    }

    /* right / bottom: coefficient counts and the Laplacian histogram */
    const by = 24 + IH * cell0 + 26, bh = H - by - 24;
    const total = L.reduce((s, M) => s + M.length * M[0].length, 0);
    g.append("rect").attr("x", 12).attr("y", by).attr("width", 350).attr("height", bh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, 12, by - 6, "coefficients per level, and the running total", VC.ink);
    {
      const bx = d3.scaleBand().domain(L.map((_, i) => i)).range([24, 340]).padding(0.3);
      const byS = d3.scaleLinear().domain([0, IW * IH]).range([by + bh - 22, by + 12]);
      L.forEach((M, l) => {
        const c = M.length * M[0].length;
        g.append("rect").attr("x", bx(l)).attr("width", bx.bandwidth())
          .attr("y", byS(c)).attr("height", by + bh - 22 - byS(c))
          .attr("fill", VC.accent).attr("fill-opacity", 0.75);
        g.append("text").attr("x", bx(l) + bx.bandwidth() / 2).attr("y", by + bh - 8)
          .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", VC.muted).text("L" + l);
        g.append("text").attr("x", bx(l) + bx.bandwidth() / 2).attr("y", byS(c) - 4)
          .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", VC.ink).text(c);
      });
      g.append("line").attr("x1", 24).attr("x2", 340).attr("y1", byS(IW * IH)).attr("y2", byS(IW * IH))
        .attr("stroke", VC.good).attr("stroke-dasharray", "3 3");
      g.append("text").attr("x", 340).attr("y", byS(IW * IH) - 4).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", VC.good).text("original pixel count " + IW * IH);
    }
    /* the histogram of Laplacian values */
    g.append("rect").attr("x", 378).attr("y", by).attr("width", W - 392).attr("height", bh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, 378, by - 6, "Laplacian coefficient values, all bands but the coarsest", VC.ink);
    {
      const NB = 61, hist = new Float64Array(NB);
      let m = 1e-9, cnt = 0, near0 = 0;
      for (let l = 0; l < L.length - 1; l++) for (const r of L[l]) for (const v of r) m = Math.max(m, Math.abs(v));
      for (let l = 0; l < L.length - 1; l++) for (const r of L[l]) for (const v of r) {
        const b = VZ.clamp(Math.round((v / m + 1) / 2 * (NB - 1)), 0, NB - 1);
        hist[b]++; cnt++; if (Math.abs(v) < 0.02) near0++;
      }
      const hx = d3.scaleLinear().domain([0, NB - 1]).range([388, W - 22]);
      const hy = d3.scaleLinear().domain([0, d3.max(hist)]).range([by + bh - 24, by + 12]);
      for (let b = 0; b < NB; b++) if (hist[b] > 0)
        g.append("rect").attr("x", hx(b)).attr("width", (W - 410) / NB).attr("y", hy(hist[b]))
          .attr("height", by + bh - 24 - hy(hist[b])).attr("fill", VC.a2).attr("fill-opacity", 0.85);
      g.append("text").attr("x", 388).attr("y", by + bh - 8).attr("font-size", 9).attr("fill", VC.muted).text("−" + VZ.fmt(m, 2));
      g.append("text").attr("x", W - 22).attr("y", by + bh - 8).attr("text-anchor", "end").attr("font-size", 9).attr("fill", VC.muted).text("+" + VZ.fmt(m, 2));
      g.append("text").attr("x", (388 + W - 22) / 2).attr("y", by + bh - 8).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", VC.good).text(VZ.fmt(100 * near0 / cnt, 1) + "% within ±0.02 of zero");
    }

    out.innerHTML =
      `${n} reduction${n === 1 ? "" : "s"}, kernel <b>${eK.options[eK.selectedIndex].text}</b> · `
      + `total pyramid coefficients <b>${total.toLocaleString()}</b> against <b>${(IW * IH).toLocaleString()}</b> pixels, ratio <b>${VZ.fmt(total / (IW * IH), 5)}</b> `
      + `(the infinite-pyramid limit is 4/3 = 1.33333)<br>`
      + `reconstruction: max |error| = <b>${maxe.toExponential(3)}</b>, RMSE = <b>${rmse.toExponential(3)}</b>`
      + (q === 0
        ? ` — machine precision. The Laplacian pyramid is <b>exactly</b> invertible for any of these kernels, including the one with negative lobes and including no blur at all: perfect reconstruction is a property of the <i>construction</i>, not of the filter.`
        : ` — the bands were quantised to a step of ${VZ.fmt(1 / (1 << (7 - q)), 4)}, and that quantisation is the entire error. This is the compression experiment: most Laplacian coefficients are near zero, so most of them survive coarse quantisation.`)
      + (eK.value === "none"
        ? `<br><b>With no pre-blur the reconstruction is still exact, and the pyramid levels are still badly aliased.</b> Perfect reconstruction says nothing about whether the coarse levels are a faithful picture of the image at that scale — §22 is a separate requirement.`
        : "");
  }

  eL.oninput = () => { eLv.textContent = eL.value; draw(); };
  eQ.oninput = draw;
  eK.onchange = draw; eS.onchange = draw;
  draw();
})();

/* ───────── 18 · splice vs feather vs multi-resolution blend ───────── */
(function () {
  const svg = d3.select("#blend-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 450, IW = 96, IH = 64;
  const eM = document.getElementById("bd-m");
  const eW = document.getElementById("bd-w"), eWv = document.getElementById("bd-wv");
  const eL = document.getElementById("bd-l"), eLv = document.getElementById("bd-lv");
  const eE = document.getElementById("bd-e"), eEv = document.getElementById("bd-ev");
  const eK = document.getElementById("bd-k");
  const out = document.getElementById("blend-readout");

  /* Two textures of the SAME pitch running along OPPOSITE diagonals, plus a low-frequency
     exposure difference. Two properties make the metrics below exact rather than
     approximate: each texture averages to zero down a column (over a whole number of
     periods), so the column mean is a clean probe of the low-frequency component; and
     the two are orthogonal, so each one's projection onto the other's basis is exactly
     zero unless it has actually leaked there.                                          */
  const PER = 6, R0 = 2, R1 = 62;            // rows 2…61: 60 rows = exactly 10 periods

  function sources(exp) {
    const A = IP.zeros2(IH, IW), B = IP.zeros2(IH, IW);
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      A[i][j] = VZ.clamp(0.50 + exp / 2 + 0.22 * Math.cos(2 * Math.PI * (j + i) / PER), 0, 1);
      B[i][j] = VZ.clamp(0.50 - exp / 2 + 0.22 * Math.cos(2 * Math.PI * (j - i) / PER), 0, 1);
    }
    return [A, B];
  }
  function mask(kind, width) {
    const M = IP.zeros2(IH, IW);
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      let edge;
      if (kind === "line") edge = IW / 2;
      else edge = IW / 2 + 16 * Math.sin(2 * Math.PI * i / 40) + 8 * Math.sin(2 * Math.PI * i / 13);
      M[i][j] = VZ.clamp(0.5 - (j - edge) / Math.max(1e-6, width), 0, 1);
    }
    return M;
  }

  function draw() {
    const method = eM.value, wdt = +eW.value, lv = +eL.value, exp = +eE.value;
    const [A, B] = sources(exp);
    const hard = mask(eK.value, 1e-6);
    const soft = mask(eK.value, wdt);
    const h = PYR.kernelOf("binom");

    let C;
    if (method === "splice") C = A.map((r, i) => Float64Array.from(r, (v, j) => hard[i][j] * v + (1 - hard[i][j]) * B[i][j]));
    else if (method === "feather") C = A.map((r, i) => Float64Array.from(r, (v, j) => soft[i][j] * v + (1 - soft[i][j]) * B[i][j]));
    else {
      const LA = PYR.lapPyr(PYR.gaussPyr(A, lv, h), h);
      const LB = PYR.lapPyr(PYR.gaussPyr(B, lv, h), h);
      const GM = PYR.gaussPyr(hard, lv, h);
      const LO = LA.map((M, l) => M.map((r, i) => Float64Array.from(r, (v, j) => GM[l][i][j] * v + (1 - GM[l][i][j]) * LB[l][i][j])));
      C = PYR.collapse(LO, h);
    }

    /* ── the two metrics, both defined so that a perfect composite scores zero ──
       SEAM: both textures average to (near) zero down a column, so the column mean is a
       clean probe of the LOW-frequency component. The seam is the jump in that.
       GHOST: the energy at the FOREIGN texture's frequency, minus the same measurement
       on the pure source — so a composite with no leakage scores 0 rather than the
       spectral-leakage floor of the source itself.                                      */
    const seam = Math.round(IW / 2);
    /* SEAM: the column mean over a whole number of texture periods kills both textures
       exactly, leaving only the low-frequency component. The seam is the jump in it. */
    function seamStep(M) {
      const m = new Float64Array(IW);
      for (let j = 0; j < IW; j++) { let s2 = 0; for (let i = R0; i < R1; i++) s2 += M[i][j]; m[j] = s2 / (R1 - R0); }
      let st = 0;
      for (let j = seam - 1; j <= seam + 1; j++) st = Math.max(st, Math.abs(m[j] - m[j - 1]));
      return st;
    }
    /* GHOSTING: the two textures are orthogonal, so each one's projection onto the
       other's basis is exactly zero — unless it has leaked across the seam.          */
    const half = PER * Math.max(1, Math.round(Math.max(1, wdt) / PER));
    const jR0 = seam, jR1 = Math.min(IW, seam + half);
    const jL0 = Math.max(0, seam - half), jL1 = seam;
    function proj(M, sign, j0, j1) {
      let re = 0, im = 0, n = 0;
      for (let i = R0; i < R1; i++) for (let j = j0; j < j1; j++) {
        const th = 2 * Math.PI * (j + sign * i) / PER;
        re += M[i][j] * Math.cos(th); im += M[i][j] * Math.sin(th); n++;
      }
      return n ? 2 * Math.hypot(re, im) / n : 0;
    }
    const ghosting = M => Math.max(
      proj(M, +1, jR0, jR1),      // A's diagonal, measured on B's side — 0 for pure B
      proj(M, -1, jL0, jL1));     // B's diagonal, measured on A's side — 0 for pure A
    const step = seamStep(C), ghost = ghosting(C);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const c1 = 2.1;

    IP.cap(g, 12, 16, "source A — ╱ stripes, +" + VZ.fmt(exp / 2, 2), VC.accent);
    IP.raster(g, A, 12, 22, c1, { lo: 0, hi: 1 });
    IP.cap(g, 12 + IW * c1 + 16, 16, "source B — ╲ stripes, −" + VZ.fmt(exp / 2, 2), VC.a2);
    IP.raster(g, B, 12 + IW * c1 + 16, 22, c1, { lo: 0, hi: 1 });
    IP.cap(g, 12 + 2 * (IW * c1 + 16), 16, method === "feather" ? "mask, width " + wdt : "mask", VC.violet);
    IP.raster(g, method === "feather" ? soft : hard, 12 + 2 * (IW * c1 + 16), 22, c1, { lo: 0, hi: 1 });

    const c2 = 4.6, cy = 22 + IH * c1 + 26;
    IP.cap(g, 12, cy - 6, "composite — " + eM.options[eM.selectedIndex].text, VC.ink);
    IP.raster(g, C, 12, cy, c2, { lo: 0, hi: 1 });
    g.append("line").attr("x1", 12 + seam * c2).attr("x2", 12 + seam * c2).attr("y1", cy).attr("y2", cy + IH * c2)
      .attr("stroke", VC.violet).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "4 3");

    /* the cross-section */
    const px = 12 + IW * c2 + 22, pw = W - px - 14, py = cy, ph = 130;
    g.append("rect").attr("x", px).attr("y", py).attr("width", pw).attr("height", ph)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, py - 6, "row " + (IH >> 1) + " through the composite", VC.ink);
    const lx = d3.scaleLinear().domain([0, IW - 1]).range([px + 8, px + pw - 8]);
    const ly = d3.scaleLinear().domain([0.1, 0.95]).range([py + ph - 12, py + 10]);
    let d = "";
    for (let j = 0; j < IW; j++) d += (j ? "L" : "M") + lx(j).toFixed(1) + "," + ly(C[IH >> 1][j]).toFixed(1);
    g.append("path").attr("d", d).attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.7);
    g.append("line").attr("x1", lx(seam)).attr("x2", lx(seam)).attr("y1", py + 6).attr("y2", py + ph - 8)
      .attr("stroke", VC.violet).attr("stroke-dasharray", "3 3");

    /* the two metrics, for all three methods */
    const my = py + ph + 26, mh = H - my - 22;
    g.append("rect").attr("x", px).attr("y", my).attr("width", pw).attr("height", mh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, my - 6, "measured, for all three methods", VC.ink);
    const all = {};
    ["splice", "feather", "pyr"].forEach(mm => {
      let CC;
      if (mm === "splice") CC = A.map((r, i) => Float64Array.from(r, (v, j) => hard[i][j] * v + (1 - hard[i][j]) * B[i][j]));
      else if (mm === "feather") CC = A.map((r, i) => Float64Array.from(r, (v, j) => soft[i][j] * v + (1 - soft[i][j]) * B[i][j]));
      else {
        const LA = PYR.lapPyr(PYR.gaussPyr(A, lv, h), h), LB = PYR.lapPyr(PYR.gaussPyr(B, lv, h), h);
        const GM = PYR.gaussPyr(hard, lv, h);
        CC = PYR.collapse(LA.map((M, l) => M.map((r, i) => Float64Array.from(r, (v, j) => GM[l][i][j] * v + (1 - GM[l][i][j]) * LB[l][i][j]))), h);
      }
      all[mm] = { step: seamStep(CC), ghost: ghosting(CC) };
    });
    const names = { splice: "splice", feather: "feather", pyr: "pyramid" };
    [["seam step", "step"], ["ghosting", "ghost"]].forEach(([lbl, key], gi) => {
      const y0 = my + 16 + gi * (mh / 2 - 6);
      IP.cap(g, px + 8, y0, lbl, VC.muted);
      const mx = Math.max(...Object.values(all).map(v => v[key]), 1e-9);
      Object.keys(all).forEach((mm, k) => {
        const y = y0 + 10 + k * 15;
        g.append("rect").attr("x", px + 66).attr("y", y - 8).attr("width", Math.max(1, (all[mm][key] / mx) * (pw - 140))).attr("height", 10)
          .attr("fill", mm === method ? VC.good : VC.muted).attr("fill-opacity", mm === method ? 0.9 : 0.4);
        g.append("text").attr("x", px + 62).attr("y", y).attr("text-anchor", "end").attr("font-size", 9).attr("fill", VC.muted).text(names[mm]);
        g.append("text").attr("x", px + 72 + Math.max(1, (all[mm][key] / mx) * (pw - 140))).attr("y", y)
          .attr("font-size", 9).attr("fill", VC.ink).text(VZ.fmt(all[mm][key], 4));
      });
    });

    out.innerHTML =
      `exposure difference ${VZ.fmt(exp, 2)}, both textures at pitch ${PER} px on opposite diagonals · `
      + `largest intensity step across the seam <b>${VZ.fmt(step, 4)}</b> · cross-texture energy in the transition band <b>${VZ.fmt(ghost, 5)}</b><br>`
      + `all three, measured: splice ${VZ.fmt(all.splice.step, 4)} / ${VZ.fmt(all.splice.ghost, 5)} · `
      + `feather ${VZ.fmt(all.feather.step, 4)} / ${VZ.fmt(all.feather.ghost, 5)} · `
      + `pyramid ${VZ.fmt(all.pyr.step, 4)} / ${VZ.fmt(all.pyr.ghost, 5)}  (seam / ghosting)<br>`
      + (method === "splice"
        ? `The hard splice leaves the entire exposure difference as a step in the column mean, and essentially <b>no</b> ghosting — each pixel came from exactly one source. It fails on one metric and is perfect on the other.`
        : method === "feather"
          ? `Feathering has cut the seam step to ${VZ.fmt(all.feather.step, 4)} and raised the ghosting to ${VZ.fmt(all.feather.ghost, 5)} — inside the ${wdt}-pixel band <b>both</b> textures are present at once, at partial strength. It has traded one metric for the other, and widening the band trades further in the same direction.`
          : `The pyramid blend is the only one of the three that is <b>good on both at once</b>: seam ${VZ.fmt(all.pyr.step, 4)} against the splice's ${VZ.fmt(all.splice.step, 4)}, ghosting ${VZ.fmt(all.pyr.ghost, 5)} against the feather's ${VZ.fmt(all.feather.ghost, 5)}. The low-frequency exposure difference was blended over a wide band and the high-frequency texture over a narrow one, with no width chosen by hand: the mask's Gaussian pyramid supplies it.`);
  }

  eM.onchange = draw; eK.onchange = draw;
  eW.oninput = () => { eWv.textContent = eW.value; draw(); };
  eL.oninput = () => { eLv.textContent = eL.value; draw(); };
  eE.oninput = () => { eEv.textContent = (+eE.value).toFixed(2); draw(); };
  draw();
})();

/* ───────── 19 · a lifted 5/3 wavelet ───────── */
(function () {
  const svg = d3.select("#wave-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 450, IW = 128, IH = 128;
  const eT = document.getElementById("wv-t");
  const eL = document.getElementById("wv-l"), eLv = document.getElementById("wv-lv");
  const eK = document.getElementById("wv-k"), eKv = document.getElementById("wv-kv");
  const eS = document.getElementById("wv-s");
  const out = document.getElementById("wave-readout");

  /* one level of a 1-D transform, in place over the first n entries of a row */
  function fwd1(x, n, kind) {
    const half = n >> 1, s = new Float64Array(half), d = new Float64Array(n - half);
    const ev = i => x[2 * VZ.clamp(i, 0, half - 1)];
    if (kind === "haar") {
      for (let i = 0; i < half; i++) { s[i] = (x[2 * i] + x[2 * i + 1]) / Math.SQRT2; d[i] = (x[2 * i] - x[2 * i + 1]) / Math.SQRT2; }
    } else {
      for (let i = 0; i < n - half; i++) {
        const o = x[2 * i + 1] === undefined ? x[n - 1] : x[2 * i + 1];
        d[i] = o - 0.5 * (ev(i) + ev(i + 1));
      }
      for (let i = 0; i < half; i++) s[i] = x[2 * i] + 0.25 * ((i > 0 ? d[i - 1] : d[0]) + (i < d.length ? d[i] : d[d.length - 1]));
    }
    for (let i = 0; i < half; i++) x[i] = s[i];
    for (let i = 0; i < d.length; i++) x[half + i] = d[i];
  }
  function inv1(x, n, kind) {
    const half = n >> 1, s = new Float64Array(half), d = new Float64Array(n - half);
    for (let i = 0; i < half; i++) s[i] = x[i];
    for (let i = 0; i < d.length; i++) d[i] = x[half + i];
    const outp = new Float64Array(n);
    if (kind === "haar") {
      for (let i = 0; i < half; i++) { outp[2 * i] = (s[i] + d[i]) / Math.SQRT2; outp[2 * i + 1] = (s[i] - d[i]) / Math.SQRT2; }
    } else {
      const ev = new Float64Array(half);
      for (let i = 0; i < half; i++) ev[i] = s[i] - 0.25 * ((i > 0 ? d[i - 1] : d[0]) + (i < d.length ? d[i] : d[d.length - 1]));
      const e = i => ev[VZ.clamp(i, 0, half - 1)];
      for (let i = 0; i < half; i++) outp[2 * i] = ev[i];
      for (let i = 0; i < d.length; i++) if (2 * i + 1 < n) outp[2 * i + 1] = d[i] + 0.5 * (e(i) + e(i + 1));
    }
    for (let i = 0; i < n; i++) x[i] = outp[i];
  }
  function dwt2(A, levels, kind, inverse) {
    const B = IP.clone2(A);
    const sizes = [];
    let n = A.length, m = A[0].length;
    for (let l = 0; l < levels; l++) { sizes.push([n, m]); n >>= 1; m >>= 1; }
    const order = inverse ? sizes.slice().reverse() : sizes;
    for (const [nn, mm] of order) {
      const row = new Float64Array(Math.max(nn, mm));
      if (!inverse) {
        for (let i = 0; i < nn; i++) { for (let j = 0; j < mm; j++) row[j] = B[i][j]; fwd1(row, mm, kind); for (let j = 0; j < mm; j++) B[i][j] = row[j]; }
        for (let j = 0; j < mm; j++) { for (let i = 0; i < nn; i++) row[i] = B[i][j]; fwd1(row, nn, kind); for (let i = 0; i < nn; i++) B[i][j] = row[i]; }
      } else {
        for (let j = 0; j < mm; j++) { for (let i = 0; i < nn; i++) row[i] = B[i][j]; inv1(row, nn, kind); for (let i = 0; i < nn; i++) B[i][j] = row[i]; }
        for (let i = 0; i < nn; i++) { for (let j = 0; j < mm; j++) row[j] = B[i][j]; inv1(row, mm, kind); for (let j = 0; j < mm; j++) B[i][j] = row[j]; }
      }
    }
    return B;
  }

  function draw() {
    const kind = eT.value, lv = +eL.value, keep = +eK.value / 100;
    eKv.textContent = (+eK.value) + "%";
    const A = IP.scene(IW, IH, { seed: 9 });
    const Cf = dwt2(A, lv, kind, false);

    /* threshold: keep the largest `keep` fraction of DETAIL coefficients */
    const lowN = IW >> lv, lowM = IH >> lv;
    const mags = [];
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) if (i >= lowM || j >= lowN) mags.push(Math.abs(Cf[i][j]));
    mags.sort((a, b) => b - a);
    const thr = keep >= 1 ? -1 : (mags[Math.max(0, Math.floor(keep * mags.length) - 1)] || 0);
    let kept = 0;
    const Ct = IP.map2(Cf, (v, i, j) => {
      if (i < lowM && j < lowN) return v;
      if (Math.abs(v) >= thr) { kept++; return v; }
      return 0;
    });
    const R = dwt2(Ct, lv, kind, true);
    const Rfull = dwt2(Cf, lv, kind, true);
    let maxeFull = 0, sse = 0;
    for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
      maxeFull = Math.max(maxeFull, Math.abs(Rfull[i][j] - A[i][j]));
      sse += (R[i][j] - A[i][j]) ** 2;
    }
    const rmse = Math.sqrt(sse / (IW * IH));

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell = 2.4;

    if (eS.value === "recon") {
      IP.cap(g, 12, 18, "original", VC.accent);
      IP.raster(g, A, 12, 24, cell, { lo: 0, hi: 1 });
      const x2 = 12 + IW * cell + 18;
      IP.cap(g, x2, 18, "reconstructed, keeping " + eK.value + "% of the detail", VC.good);
      IP.raster(g, R, x2, 24, cell, { lo: 0, hi: 1 });
      const x3 = x2 + IW * cell + 18;
      const D = R.map((r, i) => Float64Array.from(r, (v, j) => v - A[i][j]));
      const m = Math.max(IP.absMax(D), 1e-18);
      IP.cap(g, x3, 18, "error, full contrast (peak " + m.toExponential(1) + ")", VC.bad);
      IP.raster(g, D, x3, 24, cell, { signed: true, lo: -m, hi: m });
    } else {
      IP.cap(g, 12, 18, "the sub-band layout, " + lv + " level" + (lv === 1 ? "" : "s"), VC.ink);
      const gg = g.append("g").attr("transform", `translate(12,24)`);
      /* per-band contrast so the sparse bands are visible at all */
      const bandMax = IP.zeros2(1, 1);
      for (let i = 0; i < IH; i++) for (let j = 0; j < IW; j++) {
        let v = Ct[i][j], col;
        if (i < lowM && j < lowN) col = IP.greyColor(VZ.clamp(v, 0, 1), 1);
        else {
          /* find which level this coefficient belongs to, for a per-level scale */
          let lvl = 0, n = IW >> 1, m2 = IH >> 1;
          for (let l = 0; l < lv; l++) { if (i < (IH >> l) && j < (IW >> l)) lvl = l; }
          const sc = 0.10 * Math.pow(1.6, lvl);
          col = IP.signedColor(VZ.clamp(v, -sc, sc), sc);
        }
        gg.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
          .attr("fill", col).attr("shape-rendering", "crispEdges");
      }
      for (let l = 1; l <= lv; l++) {
        const s = cell * (IW >> (l - 1)) / 2;
        gg.append("line").attr("x1", s).attr("x2", s).attr("y1", 0).attr("y2", 2 * s).attr("stroke", VC.violet).attr("stroke-opacity", 0.8);
        gg.append("line").attr("y1", s).attr("y2", s).attr("x1", 0).attr("x2", 2 * s).attr("stroke", VC.violet).attr("stroke-opacity", 0.8);
      }
      gg.append("rect").attr("width", IW * cell).attr("height", IH * cell).attr("fill", "none").attr("stroke", VC.line);
      IP.cap(g, 12, 24 + IH * cell + 16, "LL top-left · HL right (vertical edges) · LH below (horizontal) · HH diagonal", VC.muted);
    }

    /* energy concentration curve */
    const px = eS.value === "recon" ? 12 : 12 + IW * cell + 26, pw = W - px - 14;
    const py = eS.value === "recon" ? 24 + IH * cell + 30 : 24, ph = eS.value === "recon" ? H - py - 24 : IH * cell;
    g.append("rect").attr("x", px).attr("y", py).attr("width", pw).attr("height", ph)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, py - 6, "energy carried by the largest x% of detail coefficients", VC.ink);
    let tot = 0; for (const v of mags) tot += v * v;
    const cx = d3.scaleLinear().domain([0, 100]).range([px + 30, px + pw - 12]);
    const cy = d3.scaleLinear().domain([0, 1]).range([py + ph - 22, py + 10]);
    let acc = 0, d2 = "";
    for (let k = 0; k < mags.length; k++) {
      acc += mags[k] * mags[k];
      if (k % Math.max(1, Math.floor(mags.length / 260)) === 0)
        d2 += (d2 ? "L" : "M") + cx(100 * k / mags.length).toFixed(1) + "," + cy(acc / tot).toFixed(1);
    }
    g.append("path").attr("d", d2).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
    g.append("line").attr("x1", cx(0)).attr("y1", cy(0)).attr("x2", cx(100)).attr("y2", cy(1))
      .attr("stroke", VC.muted).attr("stroke-dasharray", "3 3");
    g.append("line").attr("x1", cx(+eK.value)).attr("x2", cx(+eK.value)).attr("y1", py + 8).attr("y2", py + ph - 20)
      .attr("stroke", VC.good).attr("stroke-dasharray", "3 3");
    [0, 25, 50, 75, 100].forEach(t => g.append("text").attr("x", cx(t)).attr("y", py + ph - 6)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", VC.muted).text(t + "%"));
    /* what fraction of energy the top 5% carries */
    let a5 = 0; const n5 = Math.max(1, Math.floor(0.05 * mags.length));
    for (let k = 0; k < n5; k++) a5 += mags[k] * mags[k];

    out.innerHTML =
      `${kind === "haar" ? "Haar" : "lifted 5/3 (CDF)"}, ${lv} level${lv === 1 ? "" : "s"} · `
      + `coefficients <b>${(IW * IH).toLocaleString()}</b> against <b>${(IW * IH).toLocaleString()}</b> pixels — a tight frame, exactly critically sampled, unlike the pyramid's 4/3<br>`
      + `with nothing discarded the reconstruction error is <b>${maxeFull.toExponential(3)}</b> — exact to machine precision.<br>`
      + `the largest <b>5%</b> of the detail coefficients carry <b>${VZ.fmt(100 * a5 / tot, 2)}%</b> of the detail energy. `
      + (keep >= 1
        ? `Keeping all of them, the RMSE is ${rmse.toExponential(2)}.`
        : `Keeping the largest ${eK.value}% — ${kept.toLocaleString()} of ${mags.length.toLocaleString()} coefficients — leaves an RMSE of <b>${VZ.fmt(rmse, 5)}</b> against a signal that spans [0, 1].`);
  }

  eT.onchange = draw; eS.onchange = draw;
  eL.oninput = () => { eLv.textContent = eL.value; draw(); };
  eK.oninput = draw;
  draw();
})();

/* ───────── 20 · forward vs inverse warping ───────── */
(function () {
  const svg = d3.select("#warp-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440, N = 80;
  const eD = document.getElementById("wp-d");
  const eTh = document.getElementById("wp-th"), eThv = document.getElementById("wp-thv");
  const eS = document.getElementById("wp-s"), eSv = document.getElementById("wp-sv");
  const eSh = document.getElementById("wp-sh"), eShv = document.getElementById("wp-shv");
  const eP = document.getElementById("wp-p"), ePv = document.getElementById("wp-pv");
  const eR = document.getElementById("wp-r");
  const out = document.getElementById("warp-readout");

  /* a source with a grid, a disc and fine detail, so both holes and blur show */
  function source() {
    const A = IP.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let v = 0.22;
      if (i % 12 === 0 || j % 12 === 0) v = 0.62;                        // a grid
      const dx = j - N * 0.32, dy = i - N * 0.34;
      if (dx * dx + dy * dy < 220) v = 0.92;                             // a disc
      if (i > N * 0.62 && i < N * 0.80 && j > N * 0.18 && j < N * 0.82)
        v = 0.30 + 0.55 * (j % 4 < 2 ? 1 : 0);                           // a 2-px stripe block
      A[i][j] = v;
    }
    return A;
  }
  function Hmat(th, s, sh, p) {
    const c = Math.cos(VZ.rad(th)) * s, d = Math.sin(VZ.rad(th)) * s;
    const A = [[c, -d + sh * s, 0], [d, c, 0], [p, p * 0.4, 1]];
    /* keep the centre fixed: T(+c) · A · T(−c) */
    const h = N / 2;
    const T1 = [[1, 0, h], [0, 1, h], [0, 0, 1]], T2 = [[1, 0, -h], [0, 1, -h], [0, 0, 1]];
    return VZ.mul(VZ.mul(T1, A), T2);
  }
  const applyH = (Hm, x, y) => {
    const w = Hm[2][0] * x + Hm[2][1] * y + Hm[2][2];
    return [(Hm[0][0] * x + Hm[0][1] * y + Hm[0][2]) / w, (Hm[1][0] * x + Hm[1][1] * y + Hm[1][2]) / w];
  };
  function bilin(A, y, x) {
    const i0 = Math.floor(y), j0 = Math.floor(x), a = y - i0, b = x - j0;
    const at = (i, j) => (i < 0 || i >= N || j < 0 || j >= N) ? 0 : A[i][j];
    return (1 - a) * (1 - b) * at(i0, j0) + (1 - a) * b * at(i0, j0 + 1)
      + a * (1 - b) * at(i0 + 1, j0) + a * b * at(i0 + 1, j0 + 1);
  }

  function draw() {
    const A = source();
    const Hm = Hmat(+eTh.value, +eS.value, +eSh.value, +eP.value);
    const Hi = VZ.inv3(Hm);
    const dir = eD.value;

    /* which destination pixels legitimately need a value? */
    const need = IP.zeros2(N, N);
    let needN = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const [sx, sy] = applyH(Hi, j, i);
      if (sx >= 0 && sx <= N - 1 && sy >= 0 && sy <= N - 1) { need[i][j] = 1; needN++; }
    }

    const B = IP.zeros2(N, N), Wt = IP.zeros2(N, N);
    if (dir === "inv") {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        if (!need[i][j]) continue;
        const [sx, sy] = applyH(Hi, j, i);
        B[i][j] = eR.value === "nn" ? A[VZ.clamp(Math.round(sy), 0, N - 1)][VZ.clamp(Math.round(sx), 0, N - 1)] : bilin(A, sy, sx);
        Wt[i][j] = 1;
      }
    } else {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const [dx, dy] = applyH(Hm, j, i);
        if (dir === "fwd") {
          const ii = Math.round(dy), jj = Math.round(dx);
          if (ii >= 0 && ii < N && jj >= 0 && jj < N) { B[ii][jj] = A[i][j]; Wt[ii][jj] = 1; }
        } else {
          const i0 = Math.floor(dy), j0 = Math.floor(dx), a = dy - i0, b = dx - j0;
          [[i0, j0, (1 - a) * (1 - b)], [i0, j0 + 1, (1 - a) * b], [i0 + 1, j0, a * (1 - b)], [i0 + 1, j0 + 1, a * b]]
            .forEach(([ii, jj, w]) => { if (ii >= 0 && ii < N && jj >= 0 && jj < N) { B[ii][jj] += w * A[i][j]; Wt[ii][jj] += w; } });
        }
      }
      if (dir === "fwdsplat") for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (Wt[i][j] > 1e-6) B[i][j] /= Wt[i][j];
    }

    let holes = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
      if (need[i][j] && Wt[i][j] < (dir === "fwdsplat" ? 1e-6 : 0.5)) holes++;

    /* a crude sharpness measure: mean |Laplacian| over the covered region */
    function sharp(M) {
      let s = 0, n = 0;
      for (let i = 1; i < N - 1; i++) for (let j = 1; j < N - 1; j++) if (need[i][j]) {
        s += Math.abs(4 * M[i][j] - M[i - 1][j] - M[i + 1][j] - M[i][j - 1] - M[i][j + 1]); n++;
      }
      return n ? s / n : 0;
    }
    /* reference: the inverse-warped bilinear result */
    const REF = IP.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (need[i][j]) {
      const [sx, sy] = applyH(Hi, j, i); REF[i][j] = bilin(A, sy, sx);
    }

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell = 3.1;

    IP.cap(g, 12, 18, "source", VC.accent);
    IP.raster(g, A, 12, 24, cell, { lo: 0, hi: 1 });
    /* the outline of where it lands */
    const corners = [[0, 0], [N - 1, 0], [N - 1, N - 1], [0, N - 1]].map(([x, y]) => applyH(Hm, x, y));
    VZ.poly(g.append("g").attr("transform", `translate(12,24)`),
      corners.map(([x, y]) => [x * cell, y * cell]), { stroke: VC.violet, w: 1.4, dash: "4 3" });

    const x2 = 12 + N * cell + 20;
    IP.cap(g, x2, 18, eD.options[eD.selectedIndex].text, VC.a2);
    const gg = g.append("g").attr("transform", `translate(${x2},24)`);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const hole = need[i][j] && Wt[i][j] < (dir === "fwdsplat" ? 1e-6 : 0.5);
      gg.append("rect").attr("x", j * cell).attr("y", i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
        .attr("fill", hole ? VC.bad : IP.greyColor(VZ.clamp(B[i][j], 0, 1), 1))
        .attr("shape-rendering", "crispEdges");
    }
    gg.append("rect").attr("width", N * cell).attr("height", N * cell).attr("fill", "none").attr("stroke", VC.line);
    IP.cap(g, x2, 24 + N * cell + 16, holes > 0 ? holes + " unwritten pixels, in red" : "no unwritten pixels", holes > 0 ? VC.bad : VC.good);

    /* right: the matrices and the counts */
    const tx = x2 + N * cell + 22;
    const fmtRow = r => r.map(v => (Math.abs(v) < 1e-9 ? "0" : v.toPrecision(4)).padStart(10)).join("");
    IP.numText(g, [
      "H  (source → destination)",
      "  " + fmtRow(Hm[0]),
      "  " + fmtRow(Hm[1]),
      "  " + fmtRow(Hm[2]),
      "",
      "H⁻¹  (destination → source)",
      "  " + fmtRow(Hi[0]),
      "  " + fmtRow(Hi[1]),
      "  " + fmtRow(Hi[2]),
      "",
      "destinations needing a value   " + String(needN).padStart(6),
      "destinations written           " + String(needN - holes).padStart(6),
      "HOLES                          " + String(holes).padStart(6),
      "                               " + (VZ.fmt(100 * holes / Math.max(1, needN), 1) + "%").padStart(6),
      "",
      "mean |Laplacian| (sharpness)",
      "  this method                  " + VZ.fmt(sharp(B), 5).padStart(6),
      "  inverse + bilinear reference " + VZ.fmt(sharp(REF), 5).padStart(6)
    ], tx, 32, { size: 9.6, lead: 13 });

    const detA = Hm[0][0] * Hm[1][1] - Hm[0][1] * Hm[1][0];
    out.innerHTML =
      `|det| of the linear part of H (source → destination) = <b>${VZ.fmt(Math.abs(detA), 4)}</b>, so one source pixel covers ${VZ.fmt(Math.abs(detA), 3)} destination pixels; §28 uses the Jacobian of the <i>inverse</i> map, whose determinant is the reciprocal · `
      + `holes <b>${holes}</b> of ${needN} (<b>${VZ.fmt(100 * holes / Math.max(1, needN), 1)}%</b>)<br>`
      + (dir === "fwd"
        ? `Forward warping with nearest-integer writes. The predicted hole fraction under a pure magnification by r is 1 − 1/r²; at scale ${(+eS.value).toFixed(2)} that is ${VZ.fmt(100 * Math.max(0, 1 - 1 / Math.max(1e-9, (+eS.value) ** 2)), 1)}%. A pure rotation still leaves holes, because a rotated integer lattice does not land on an integer lattice.`
        : dir === "fwdsplat"
          ? `Splatting distributes each source pixel over its four nearest destinations with bilinear weights and normalises by the accumulated weight — §09's normalised-zero construction again. The holes are gone, and the price is blur: mean |Laplacian| is <b>${VZ.fmt(sharp(B), 5)}</b> against <b>${VZ.fmt(sharp(REF), 5)}</b> for inverse warping with the same interpolant, a loss of ${VZ.fmt(100 * (1 - sharp(B) / Math.max(1e-9, sharp(REF))), 1)}% of the measured high-frequency detail.`
          : `Inverse warping. The loop is over the destination, so every destination pixel is visited exactly once: <b>holes are impossible</b>, and the count above is 0 at every setting. What remains is the interpolation question of §27 — and, when this determinant falls below 1 so that many source pixels crowd into one destination pixel, the anti-aliasing question of §28.`);
  }

  eD.onchange = draw; eR.onchange = draw;
  eTh.oninput = () => { eThv.textContent = eTh.value + "°"; draw(); };
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(2); draw(); };
  eSh.oninput = () => { eShv.textContent = (+eSh.value).toFixed(2); draw(); };
  eP.oninput = () => { ePv.textContent = (+eP.value).toFixed(4); draw(); };
  draw();
})();

/* ───────── 21 · interpolation kernels ───────── */
(function () {
  const svg = d3.select("#interp-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470;
  const eK = document.getElementById("in-k");
  const eA = document.getElementById("in-a"), eAv = document.getElementById("in-av");
  const eWl = document.getElementById("in-w"), eWv = document.getElementById("in-wv");
  const eZ = document.getElementById("in-z"), eZv = document.getElementById("in-zv");
  const eAll = document.getElementById("in-all");
  const out = document.getElementById("interp-readout");

  const nn = x => (Math.abs(x) < 0.5 ? 1 : 0);
  const bil = x => (Math.abs(x) < 1 ? 1 - Math.abs(x) : 0);
  function cub(x, a) {
    const t = Math.abs(x);
    if (t < 1) return 1 - (a + 3) * t * t + (a + 2) * t * t * t;
    if (t < 2) return a * (t - 1) * (t - 2) * (t - 2);
    return 0;
  }
  function wsinc(x, Wl) {
    const t = Math.abs(x);
    if (t >= Wl) return 0;
    const s = t === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    return s * 0.5 * (1 + Math.cos(Math.PI * x / Wl));       // Hann window
  }
  function kernels() {
    const a = +eA.value, Wl = +eWl.value;
    return [
      { n: "nearest", h: nn, R: 1, c: VC.muted },
      { n: "bilinear", h: bil, R: 1, c: VC.accent },
      { n: "bicubic a = " + a.toFixed(2), h: x => cub(x, a), R: 2, c: VC.a2 },
      { n: "windowed sinc W = " + Wl, h: x => wsinc(x, Wl), R: Wl, c: VC.violet }
    ];
  }
  const CUR = () => ({ nn: 0, bil: 1, cub: 2, sinc: 3 })[eK.value];

  /* 1-D resample of a sampled function */
  function resample(fn, t, k, dx) {
    const u = t / dx, k0 = Math.floor(u);
    let s = 0;
    for (let m = -k.R; m <= k.R; m++) s += fn((k0 + m) * dx) * k.h(u - (k0 + m));
    return s;
  }

  function draw() {
    eAv.textContent = (+eA.value).toFixed(2).replace("-", "−");
    eWv.textContent = eWl.value; eZv.textContent = eZ.value;
    const K = kernels(), cur = CUR(), show = eAll.checked ? K.map((_, i) => i) : [cur];
    const zoom = +eZ.value;

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;

    /* ── top left: the kernels ── */
    const px = 34, pw = 330, py = 26, ph = 170;
    g.append("rect").attr("x", px).attr("y", py).attr("width", pw).attr("height", ph)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, py - 6, "the kernels h(x)", VC.ink);
    const xs = d3.scaleLinear().domain([-4.2, 4.2]).range([px + 8, px + pw - 8]);
    const ys = d3.scaleLinear().domain([-0.34, 1.12]).range([py + ph - 18, py + 10]);
    g.append("line").attr("x1", xs(-4.2)).attr("x2", xs(4.2)).attr("y1", ys(0)).attr("y2", ys(0)).attr("stroke", VC.muted);
    for (let t = -4; t <= 4; t++) g.append("line").attr("x1", xs(t)).attr("x2", xs(t)).attr("y1", py + 8).attr("y2", py + ph - 16)
      .attr("stroke", VC.grid);
    show.forEach(i => {
      const k = K[i]; let d = "";
      for (let s = 0; s <= 420; s++) { const x = -4.2 + s * 8.4 / 420; d += (s ? "L" : "M") + xs(x).toFixed(1) + "," + ys(k.h(x)).toFixed(1); }
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", k.c).attr("stroke-width", i === cur ? 2.4 : 1.2)
        .attr("stroke-opacity", i === cur ? 1 : 0.6);
    });
    VZ.legend(g, show.map(i => ({ color: K[i].c, label: K[i].n })), px + 10, py + 18, { vertical: true, gap: 13, font: 9.5 });

    /* ── top right: log frequency responses ── */
    const qx = px + pw + 24, qw = W - qx - 14;
    g.append("rect").attr("x", qx).attr("y", py).attr("width", qw).attr("height", ph)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, qx, py - 6, "|H(ω)| in dB — everything right of π is aliasing", VC.ink);
    const wx = d3.scaleLinear().domain([0, 3 * Math.PI]).range([qx + 30, qx + qw - 8]);
    const wy = d3.scaleLinear().domain([-90, 6]).range([py + ph - 18, py + 10]);
    g.append("line").attr("x1", wx(Math.PI)).attr("x2", wx(Math.PI)).attr("y1", py + 8).attr("y2", py + ph - 16)
      .attr("stroke", VC.violet).attr("stroke-dasharray", "3 3");
    g.append("text").attr("x", wx(Math.PI) + 4).attr("y", py + 20).attr("font-size", 9.5).attr("fill", VC.violet).text("π");
    show.forEach(i => {
      const k = K[i]; let d = "";
      for (let s = 0; s <= 240; s++) {
        const w = 3 * Math.PI * s / 240;
        /* continuous-kernel transform, by numerical integration */
        let re = 0; const dx = 0.01;
        for (let x = -k.R; x <= k.R; x += dx) re += k.h(x) * Math.cos(w * x) * dx;
        const db = 20 * Math.log10(Math.max(1e-6, Math.abs(re)));
        d += (s ? "L" : "M") + wx(w).toFixed(1) + "," + wy(VZ.clamp(db, -90, 6)).toFixed(1);
      }
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", k.c).attr("stroke-width", i === cur ? 2.2 : 1.1)
        .attr("stroke-opacity", i === cur ? 1 : 0.55);
    });
    [-80, -60, -40, -20, 0].forEach(t => {
      g.append("line").attr("x1", wx(0)).attr("x2", wx(3 * Math.PI)).attr("y1", wy(t)).attr("y2", wy(t)).attr("stroke", VC.grid);
      g.append("text").attr("x", qx + 26).attr("y", wy(t) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(t);
    });

    /* ── bottom left: a hard step, zoomed ── */
    const sy0 = py + ph + 34, sh = H - sy0 - 26;
    g.append("rect").attr("x", px).attr("y", sy0).attr("width", pw).attr("height", sh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, px, sy0 - 6, "a hard step, magnified " + zoom + "×", VC.ink);
    const step = x => (x >= 0 ? 1 : 0);
    const ex = d3.scaleLinear().domain([-3, 3]).range([px + 8, px + pw - 8]);
    const ey = d3.scaleLinear().domain([-0.28, 1.28]).range([sy0 + sh - 16, sy0 + 10]);
    [0, 1].forEach(v => g.append("line").attr("x1", ex(-3)).attr("x2", ex(3)).attr("y1", ey(v)).attr("y2", ey(v))
      .attr("stroke", VC.grid));
    const over = {};
    show.forEach(i => {
      const k = K[i]; let d = "", mx = -Infinity, mn = Infinity;
      /* the polyline is drawn at exactly the chosen zoom, so the output samples the
         reader would actually get are the points on it */
      const NS = 6 * zoom;
      for (let s = 0; s <= NS; s++) {
        const x = -3 + 6 * s / NS;
        d += (s ? "L" : "M") + ex(x).toFixed(1) + "," + ey(resample(step, x, k, 1)).toFixed(1);
      }
      /* but the over/undershoot is measured on a fine grid, so the number is honest */
      for (let s = 0; s <= 1200; s++) {
        const v = resample(step, -3 + 6 * s / 1200, k, 1);
        mx = Math.max(mx, v); mn = Math.min(mn, v);
      }
      over[i] = [mx, mn];
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", k.c).attr("stroke-width", i === cur ? 2.2 : 1.1)
        .attr("stroke-opacity", i === cur ? 1 : 0.55);
    });
    for (let x = -3; x <= 3; x++) g.append("circle").attr("cx", ex(x)).attr("cy", ey(step(x))).attr("r", 2.6).attr("fill", VC.ink);

    /* ── bottom right: the error order, log-log ── */
    const ry = sy0, rh = sh;
    g.append("rect").attr("x", qx).attr("y", ry).attr("width", qw).attr("height", rh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, qx, ry - 6, "max error on sin x against sample spacing — log–log", VC.ink);
    const dxs = [0.4, 0.2, 0.1, 0.05, 0.025];
    const lx = d3.scaleLog().domain([0.02, 0.5]).range([qx + 34, qx + qw - 10]);
    const ly = d3.scaleLog().domain([1e-8, 1e-1]).range([ry + rh - 18, ry + 10]);
    const orders = {};
    show.forEach(i => {
      const k = K[i], pts = [];
      dxs.forEach(dx => {
        let e = 0;
        for (let s = 0; s <= 200; s++) { const t = 1 + s / 200; e = Math.max(e, Math.abs(resample(Math.sin, t, k, dx) - Math.sin(t))); }
        pts.push([dx, Math.max(e, 1e-9)]);
      });
      let d = ""; pts.forEach((p, s) => d += (s ? "L" : "M") + lx(p[0]).toFixed(1) + "," + ly(p[1]).toFixed(1));
      g.append("path").attr("d", d).attr("fill", "none").attr("stroke", k.c).attr("stroke-width", i === cur ? 2.2 : 1.1)
        .attr("stroke-opacity", i === cur ? 1 : 0.55);
      pts.forEach(p => g.append("circle").attr("cx", lx(p[0])).attr("cy", ly(p[1])).attr("r", 2.2).attr("fill", k.c));
      const n = pts.length;
      orders[i] = Math.log2(pts[n - 2][1] / pts[n - 1][1]);
    });
    [1e-8, 1e-6, 1e-4, 1e-2].forEach(t => {
      g.append("line").attr("x1", lx(0.02)).attr("x2", lx(0.5)).attr("y1", ly(t)).attr("y2", ly(t)).attr("stroke", VC.grid);
      g.append("text").attr("x", qx + 30).attr("y", ly(t) + 3).attr("text-anchor", "end").attr("font-size", 8.5)
        .attr("fill", VC.muted).text(t.toExponential(0));
    });
    [0.025, 0.1, 0.4].forEach(t => g.append("text").attr("x", lx(t)).attr("y", ry + rh - 5)
      .attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", VC.muted).text("Δ=" + t));

    const k = K[cur], [mx, mn] = over[cur] || [1, 0];
    out.innerHTML =
      `<b>${k.n}</b> · ${2 * k.R} taps per axis, ${(2 * k.R) ** 2} in 2-D · `
      + `measured order of accuracy <b>${VZ.fmt(orders[cur] || 0, 3)}</b><br>`
      + `on a hard step: overshoot <b>${VZ.fmt(100 * (mx - 1), 3)}%</b>, undershoot <b>${VZ.fmt(100 * Math.abs(mn), 3)}%</b> — `
      + (mx <= 1 + 1e-9 && mn >= -1e-9
        ? `none, because this kernel is non-negative everywhere. It cannot ring, and it cannot sharpen either.`
        : `the kernel has negative lobes, so it must overshoot at a step. That is the halo, and it is not a bug.`)
      + (eK.value === "cub"
        ? `<br>at <code>a</code> = ${(+eA.value).toFixed(2).replace("-", "−")}: h′(1) = ${(+eA.value).toFixed(2).replace("-", "−")} exactly, and the measured order is ${VZ.fmt(orders[cur] || 0, 2)}. `
        + (Math.abs(+eA.value + 0.5) < 1e-9
          ? `<b>This is the only value at which the family reaches third order.</b> It reproduces constants, ramps and parabolas exactly.`
          : `Set <code>a</code> to exactly −0.50 and the order jumps to 3: every other value fails to reproduce even a linear ramp, and is therefore first order.`)
        : "");
  }

  eK.onchange = draw; eAll.onchange = draw;
  eA.oninput = draw; eWl.oninput = draw; eZ.oninput = draw;
  draw();
})();

/* ───────── 22 · the resampling Jacobian, MIP levels and anisotropic filtering ───────── */
(function () {
  const svg = d3.select("#mip-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470, TW = 256, SW = 140, SH = 105;
  const eT = document.getElementById("mi-t"), eTv = document.getElementById("mi-tv");
  const eR = document.getElementById("mi-r"), eRv = document.getElementById("mi-rv");
  const eM = document.getElementById("mi-m");
  const eU = document.getElementById("mi-u"), eV = document.getElementById("mi-v");
  const out = document.getElementById("mip-readout");

  /* the texture: a checkerboard with a finer grid on top, so both moiré and blur show */
  function texture() {
    const A = IP.zeros2(TW, TW);
    for (let i = 0; i < TW; i++) for (let j = 0; j < TW; j++) {
      const c = ((i >> 5) + (j >> 5)) & 1;
      let v = c ? 0.78 : 0.22;
      if ((i & 7) === 0 || (j & 7) === 0) v = c ? 0.94 : 0.06;
      A[i][j] = v;
    }
    return A;
  }
  const TEX = texture();
  /* the MIP pyramid, built with the binomial kernel of §23 */
  const MIP = (() => {
    const h = PYR.kernelOf("binom"), L = [TEX];
    while (L[L.length - 1].length > 4) L.push(PYR.reduce(L[L.length - 1], h));
    return L;
  })();
  function sampleLevel(l, y, x) {
    const M = MIP[VZ.clamp(l, 0, MIP.length - 1)], n = M.length;
    const s = n / TW, yy = y * s, xx = x * s;
    const i0 = Math.floor(yy), j0 = Math.floor(xx), a = yy - i0, b = xx - j0;
    const at = (i, j) => M[((i % n) + n) % n][((j % n) + n) % n];
    return (1 - a) * (1 - b) * at(i0, j0) + (1 - a) * b * at(i0, j0 + 1)
      + a * (1 - b) * at(i0 + 1, j0) + a * b * at(i0 + 1, j0 + 1);
  }
  function sampleTri(l, y, x) {
    const l0 = Math.floor(VZ.clamp(l, 0, MIP.length - 1)), fr = VZ.clamp(l, 0, MIP.length - 1) - l0;
    return (1 - fr) * sampleLevel(l0, y, x) + fr * sampleLevel(l0 + 1, y, x);
  }

  /* the ground-plane map: screen (sx, sy) → texture (u, v), a perspective warp */
  function mapping(tilt, rot) {
    const t = VZ.rad(Math.max(1, tilt)), c = Math.cos(VZ.rad(rot)), s = Math.sin(VZ.rad(rot));
    /* screen y measured downward from the horizon; a standard ground-plane homography */
    return function (sx, sy) {
      const yy = sy + 0.10;                       // keep away from the horizon
      const d = 1 / (yy * Math.tan(t) + 1e-6);
      const X = (sx - 0.5) * d * 2.2, Y = d * 2.2;
      return [(c * X - s * Y) * 26 + 128, (s * X + c * Y) * 26 + 128];
    };
  }
  function jacobian(mp, sx, sy) {
    const e = 1e-3;
    const [x0, y0] = mp(sx, sy);
    const [xu, yu] = mp(sx + e, sy), [xv, yv] = mp(sx, sy + e);
    return [[(xu - x0) / e / SW, (xv - x0) / e / SH], [(yu - y0) / e / SW, (yv - y0) / e / SH]];
  }
  function svd2(A) {
    const a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1];
    const E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, Hh = (c - b) / 2;
    const q = Math.hypot(E, Hh), r = Math.hypot(F, G);
    return { s1: q + r, s2: Math.max(0, Math.abs(q - r)), theta: Math.atan2(G, F) / 2 + Math.atan2(Hh, E) };
  }

  function draw() {
    const tilt = +eT.value, rot = +eR.value, method = eM.value;
    const mp = mapping(tilt, rot);
    const img = IP.zeros2(SH, SW), lvl = IP.zeros2(SH, SW);
    const REF = IP.zeros2(SH, SW);
    for (let i = 0; i < SH; i++) for (let j = 0; j < SW; j++) {
      const sx = j / SW, sy = i / SH;
      const [tx, ty] = mp(sx, sy);
      const A = jacobian(mp, sx, sy);
      const { s1, s2, theta } = svd2(A);
      let v;
      if (method === "point") v = sampleLevel(0, ty, tx);
      else if (method === "mipmax") v = sampleLevel(Math.round(Math.log2(Math.max(1, s1))), ty, tx);
      else if (method === "mipmin") v = sampleLevel(Math.round(Math.log2(Math.max(1, s2))), ty, tx);
      else if (method === "tri") v = sampleTri(Math.log2(Math.max(1, s1)), ty, tx);
      else {
        const l = Math.log2(Math.max(1, s2)), ns = 8;
        const dx = Math.cos(theta) * s1 / ns, dy = Math.sin(theta) * s1 / ns;
        let s = 0;
        for (let k = 0; k < ns; k++) { const o = k - (ns - 1) / 2; s += sampleTri(l, ty + o * dy, tx + o * dx); }
        v = s / ns;
      }
      img[i][j] = v;
      lvl[i][j] = Math.log2(Math.max(1, s1));
      /* the reference: a dense average over the ACTUAL parallelogram footprint, obtained
         by walking the Jacobian rather than by taking a fixed screen-space box. This is
         the answer every method below is trying to approximate. */
      let acc = 0, n = 0;
      for (let p = -2; p <= 2; p++) for (let q = -2; q <= 2; q++) {
        const a = p / 5, b = q / 5;
        const rx = tx + A[0][0] * a + A[0][1] * b, ry = ty + A[1][0] * a + A[1][1] * b;
        acc += sampleLevel(0, ry, rx); n++;
      }
      REF[i][j] = acc / n;
    }

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const cell = 1.8;

    IP.cap(g, 12, 18, eM.options[eM.selectedIndex].text, VC.ink);
    IP.raster(g, img, 12, 24, cell, { lo: 0, hi: 1 });
    const pu = +eU.value, pv = +eV.value;
    g.append("circle").attr("cx", 12 + pu * SW * cell).attr("cy", 24 + pv * SH * cell).attr("r", 4)
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.6);

    /* the footprint at the probe */
    const A = jacobian(mp, pu, pv), { s1, s2, theta } = svd2(A);
    const [tx0, ty0] = mp(pu, pv);
    const fx = 12 + SW * cell + 24, fw = 210, fy = 24, fh = 210;
    g.append("rect").attr("x", fx).attr("y", fy).attr("width", fw).attr("height", fh)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    IP.cap(g, fx, 18, "the footprint in texture space, at the probe", VC.ink);
    const span = Math.max(8, 2.6 * s1);
    const tsx = d3.scaleLinear().domain([tx0 - span, tx0 + span]).range([fx + 6, fx + fw - 6]);
    const tsy = d3.scaleLinear().domain([ty0 - span, ty0 + span]).range([fy + 6, fy + fh - 6]);
    /* the texture behind it */
    const stepP = Math.max(1, Math.round(2 * span / 60));
    for (let y = Math.floor(ty0 - span); y < ty0 + span; y += stepP)
      for (let x = Math.floor(tx0 - span); x < tx0 + span; x += stepP) {
        const v = sampleLevel(0, y, x);
        g.append("rect").attr("x", tsx(x)).attr("y", tsy(y))
          .attr("width", Math.abs(tsx(x + stepP) - tsx(x)) + 0.5).attr("height", Math.abs(tsy(y + stepP) - tsy(y)) + 0.5)
          .attr("fill", IP.greyColor(VZ.clamp(v, 0, 1), 1)).attr("shape-rendering", "crispEdges");
      }
    /* the parallelogram the destination pixel pulls back to */
    const corners = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([a, b]) => {
      const dx = A[0][0] * a + A[0][1] * b, dy = A[1][0] * a + A[1][1] * b;
      return [tsx(tx0 + dx), tsy(ty0 + dy)];
    });
    VZ.poly(g, corners, { stroke: VC.a2, w: 2, fill: VC.a2, fillOp: 0.16 });
    /* the axes of the ellipse */
    [[s1, theta, VC.bad, "major " + VZ.fmt(s1, 2)], [s2, theta + Math.PI / 2, VC.good, "minor " + VZ.fmt(s2, 2)]].forEach(([len, th, col, lbl]) => {
      VZ.arrow(g, tsx(tx0), tsy(ty0), tsx(tx0 + len / 2 * Math.cos(th)), tsy(ty0 + len / 2 * Math.sin(th)), { color: col, w: 1.6, head: 6 });
    });
    IP.numText(g, [
      "major axis  " + VZ.fmt(s1, 4) + " texels",
      "minor axis  " + VZ.fmt(s2, 4) + " texels",
      "anisotropy  " + VZ.fmt(s2 > 1e-9 ? s1 / s2 : Infinity, 2) + " : 1",
      "|det A|     " + VZ.fmt(Math.abs(A[0][0] * A[1][1] - A[0][1] * A[1][0]), 4),
      "l = log2(max) = " + VZ.fmt(Math.log2(Math.max(1, s1)), 3),
      "l = log2(min) = " + VZ.fmt(Math.log2(Math.max(1, s2)), 3)
    ], fx + 6, fy + fh + 18, { size: 10, lead: 13 });

    /* the pyramid stack */
    const sx0 = fx + fw + 22, sy0 = 24;
    IP.cap(g, sx0, 18, "the MIP pyramid", VC.ink);
    let yy = sy0;
    const lSel = Math.log2(Math.max(1, s1)), lMin = Math.log2(Math.max(1, s2));
    MIP.forEach((M, l) => {
      /* drawn schematically: a bar whose width is the level's resolution. Rendering
         256² rects seven times over would cost more than the whole rest of the page. */
      const bw = 62 * M.length / TW + 6;
      const sel = Math.abs(l - lSel) < 0.5;
      g.append("rect").attr("x", sx0).attr("y", yy).attr("width", bw).attr("height", 16)
        .attr("fill", sel ? VC.a2 : VC.panel2).attr("fill-opacity", sel ? 0.55 : 1)
        .attr("stroke", sel ? VC.a2 : VC.line);
      g.append("text").attr("x", sx0 + 72).attr("y", yy + 12).attr("font-size", 9.5)
        .attr("fill", sel ? VC.a2 : VC.muted).text("l = " + l + "   " + M.length + "²");
      yy += 20;
    });
    /* the two candidate fractional levels, marked on the same axis */
    [[lSel, VC.bad, "log₂(major)"], [lMin, VC.good, "log₂(minor)"]].forEach(([lv, col, lbl]) => {
      const y = sy0 + VZ.clamp(lv, 0, MIP.length - 1) * 20 + 8;
      g.append("line").attr("x1", sx0 - 8).attr("x2", sx0 - 2).attr("y1", y).attr("y2", y)
        .attr("stroke", col).attr("stroke-width", 2);
      g.append("text").attr("x", sx0 + 72).attr("y", y + 3).attr("font-size", 8.6).attr("fill", col)
        .attr("text-anchor", "start").attr("dx", 46).text(lbl + " = " + VZ.fmt(lv, 2));
    });

    /* error against the densely area-averaged reference */
    let sse = 0, n = 0;
    for (let i = Math.floor(SH * 0.45); i < SH; i++) for (let j = 0; j < SW; j++) { sse += (img[i][j] - REF[i][j]) ** 2; n++; }
    const rmse = Math.sqrt(sse / n);
    /* local contrast far from the camera — high means either detail or moiré */
    let con = 0, cn = 0;
    for (let i = Math.floor(SH * 0.05); i < Math.floor(SH * 0.25); i++) for (let j = 1; j < SW; j++) { con += Math.abs(img[i][j] - img[i][j - 1]); cn++; }
    let conRef = 0;
    for (let i = Math.floor(SH * 0.05); i < Math.floor(SH * 0.25); i++) for (let j = 1; j < SW; j++) conRef += Math.abs(REF[i][j] - REF[i][j - 1]);

    out.innerHTML =
      `probe footprint: major axis <b>${VZ.fmt(s1, 3)}</b> texels, minor <b>${VZ.fmt(s2, 3)}</b>, anisotropy <b>${VZ.fmt(s2 > 1e-9 ? s1 / s2 : Infinity, 2)} : 1</b> · `
      + `|det A| = <b>${VZ.fmt(Math.abs(A[0][0] * A[1][1] - A[0][1] * A[1][0]), 3)}</b> source texels per destination pixel<br>`
      + `RMSE against a densely area-averaged reference, over the far half of the image: <b>${VZ.fmt(rmse, 5)}</b> · `
      + `mean |column difference| in the distance: <b>${VZ.fmt(con / cn, 5)}</b> against the reference's <b>${VZ.fmt(conRef / cn, 5)}</b><br>`
      + (method === "point"
        ? `No filtering at all: the far half is <b>moiré</b>, and the column-difference figure is far <i>above</i> the reference because the aliasing has invented contrast that is not in the texture.`
        : method === "mipmax"
          ? `One scalar rate from the <b>major</b> axis. Aliasing is suppressed, and the column-difference figure falls <i>below</i> the reference: the sharp direction has been over-blurred, because a single number cannot describe an ellipse ${VZ.fmt(s2 > 1e-9 ? s1 / s2 : 0, 1)} times longer than it is wide.`
          : method === "mipmin"
            ? `One scalar rate from the <b>minor</b> axis. Detail survives along the sharp direction and the squashed direction aliases — the opposite failure, and the reason there is no correct scalar choice.`
            : method === "tri"
              ? `Trilinear blending between the two bracketing levels removes the abrupt "MIP band" that appears wherever the rate crosses a power of two — but it is still isotropic, and still over-blurs by the anisotropy ratio.`
              : `Eight samples spread along the <b>major</b> axis at a level chosen from the <b>minor</b> axis. This is the right shape of answer, and the metric that shows it is the distant contrast: <b>${VZ.fmt(con / cn, 5)}</b> against the reference's <b>${VZ.fmt(conRef / cn, 5)}</b>, where point sampling comes out far above (invented detail) and isotropic MIP-mapping far below (destroyed detail). The RMSE separates the methods much less, because a squared error over a smooth reference is dominated by the low frequencies all of them get right.`);
  }

  eM.onchange = draw;
  eT.oninput = () => { eTv.textContent = eT.value + "°"; draw(); };
  eR.oninput = () => { eRv.textContent = eR.value + "°"; draw(); };
  eU.oninput = draw; eV.oninput = draw;
  draw();
})();
