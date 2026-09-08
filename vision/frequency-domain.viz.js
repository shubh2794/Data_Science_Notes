/* frequency-domain.viz.js — the twenty-two visualizations on vision/frequency-domain.html.
   Loaded after ../data.js → ../notes.js → vision-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

      1  #sig-svg    a sinusoid in, a scaled and shifted sinusoid out — the definition
      2  #pair-svg   transform pairs, with the width trade draggable
      3  #dft-svg    the DFT basis, the analysis sum, and the reconstruction
      4  #circ-svg   the DFT diagonalises convolution: a circulant matrix and its eigenvalues
      5  #fft-svg    the radix-2 split, the butterfly, and the cost curve
      6  #spec-svg   a 2-D image and its magnitude spectrum, with the geometry labelled
      7  #band-svg   brush out frequency bands and watch the reconstruction change
      8  #swap-svg   THE phase swap: magnitude of one image, phase of the other
      9  #samp-svg   sampling replicates the spectrum; overlap is aliasing
     10  #recon-svg  sinc interpolation, and what it costs to truncate it
     11  #alias-svg  one sinusoid, one sampling rate, the alias measured
                     (the IIFEs below are in file order, which differs here by one
                      swap from the page order listed above; ids are what matter)
     12  #mtf-svg    the point spread function, the MTF, and the pre-filter compromise
     13  #conv-svg   the convolution theorem both ways, with the residual printed
     14  #resp-svg   drag a kernel's taps; watch its frequency response and its ringing
     15  #gauss-svg  the Gaussian's self-transform, and what truncation costs
     16  #unc-svg    the uncertainty product, measured for a family of windows
     17  #win-svg    windowing and spectral leakage
     18  #stft-svg   the short-time transform: one window width, one fixed grid
     19  #dct-svg    DCT against DFT on a real block, with a truncation slider
     20  #wave-svg   multi-resolution analysis by lifting, exact round trip
     21  #tile-svg   time-frequency tiling: STFT's grid against the wavelet's
     22  #notch-svg  periodic corruption, found in the spectrum and removed there

   Every number these print is recomputed from the arrays they draw, so the pictures and
   the prose cannot drift apart.                                                        */

/* ══════════ page-local helpers (an FD namespace; the shared kit is VZ) ══════════ */
const FD = (function () {

  /* ── complex arithmetic, as {re, im} — small and explicit rather than fast ──── */
  const C  = (re, im) => ({ re: re, im: im === undefined ? 0 : im });
  const cadd = (a, b) => ({ re: a.re + b.re, im: a.im + b.im });
  const csub = (a, b) => ({ re: a.re - b.re, im: a.im - b.im });
  const cmul = (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
  const cscale = (a, s) => ({ re: a.re * s, im: a.im * s });
  const cconj = a => ({ re: a.re, im: -a.im });
  const cabs = a => Math.hypot(a.re, a.im);
  const carg = a => Math.atan2(a.im, a.re);
  const cexp = th => ({ re: Math.cos(th), im: Math.sin(th) });        // e^{j·th}
  const cdiv = (a, b) => {
    const d = b.re * b.re + b.im * b.im;
    return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
  };

  /* ── the DFT, written out longhand, and a radix-2 FFT that must agree with it ──
        Convention used EVERYWHERE on this page: the forward transform carries no
        1/N, the inverse carries 1/N. Parseval is then  Σ|f|² = (1/N)·Σ|F|².       */
  function dft(x) {                                  // x: array of {re, im} or numbers
    const N = x.length, X = [];
    const z = x.map(v => (typeof v === "number") ? C(v) : v);
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const th = -2 * Math.PI * k * n / N, c = Math.cos(th), s = Math.sin(th);
        re += z[n].re * c - z[n].im * s;
        im += z[n].re * s + z[n].im * c;
      }
      X.push(C(re, im));
    }
    return X;
  }
  function idft(X) {
    const N = X.length;
    const conj = X.map(cconj);
    const Y = fft(conj);
    return Y.map(v => cscale(cconj(v), 1 / N));
  }
  /* recursive radix-2; falls back to the direct sum when N is not a power of two,
     which keeps every caller correct at the cost of speed on odd lengths */
  function fft(x) {
    const N = x.length;
    const z = x.map(v => (typeof v === "number") ? C(v) : v);
    if (N <= 1) return z;
    if (N & (N - 1)) return dft(z);
    const ev = [], od = [];
    for (let n = 0; n < N; n += 2) { ev.push(z[n]); od.push(z[n + 1]); }
    const E = fft(ev), O = fft(od), X = new Array(N);
    for (let k = 0; k < N / 2; k++) {
      const t = cmul(cexp(-2 * Math.PI * k / N), O[k]);
      X[k] = cadd(E[k], t);
      X[k + N / 2] = csub(E[k], t);
    }
    return X;
  }
  const ifft = X => idft(X);

  /* 2-D separable transform on an array of rows (numbers or complex) */
  function fft2(A) {
    const H = A.length, W = A[0].length;
    let R = A.map(r => fft(Array.from(r, v => (typeof v === "number") ? C(v) : v)));
    const out = [];
    for (let i = 0; i < H; i++) out.push(new Array(W));
    for (let j = 0; j < W; j++) {
      const col = fft(R.map(r => r[j]));
      for (let i = 0; i < H; i++) out[i][j] = col[i];
    }
    return out;
  }
  function ifft2(F) {
    const H = F.length, W = F[0].length;
    const conj = F.map(r => r.map(cconj));
    const G = fft2(conj);
    return G.map(r => r.map(v => cscale(cconj(v), 1 / (H * W))));
  }
  const re2 = F => F.map(r => Float64Array.from(r, v => v.re));

  /* swap quadrants so DC sits in the middle — display only, never arithmetic */
  function fftshift2(A) {
    const H = A.length, W = A[0].length, h = H >> 1, w = W >> 1, B = [];
    for (let i = 0; i < H; i++) {
      B.push(new Array(W));
      for (let j = 0; j < W; j++) B[i][j] = A[(i + h) % H][(j + w) % W];
    }
    return B;
  }

  /* ── frequency response of a small centred kernel, by direct summation ────────
        H(w) = Σ_k h(k)·e^{−j w k}   (the CONVOLUTION convention: g = Σ h(k) f(x−k)) */
  function resp(h, w) {
    const R = (h.length - 1) >> 1;
    let re = 0, im = 0;
    for (let k = -R; k <= R; k++) { re += h[k + R] * Math.cos(w * k); im += -h[k + R] * Math.sin(w * k); }
    return C(re, im);
  }

  /* ── array helpers. Allocation, border handling, 2-D correlation/convolution and
        the Gaussian kernel all come from the SHARED library (VZ) — this page must not
        fork them, because a second border-mode convention is exactly the failure the
        promotion into vision-viz.js was meant to prevent. ─────────────────────────── */
  const zeros = VZ.zeros, zeros2 = VZ.zeros2;
  function minmax(a) { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }
  function minmax2(A) { let lo = Infinity, hi = -Infinity; for (const r of A) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }
  function flat(A) { const o = []; for (const r of A) for (const v of r) o.push(v); return o; }
  const sum = a => { let s = 0; for (const v of a) s += v; return s; };
  const energy = a => { let s = 0; for (const v of a) s += v * v; return s; };
  const rmse = (a, b) => { let s = 0, n = 0; for (let i = 0; i < a.length; i++) { s += (a[i] - b[i]) ** 2; n++; } return Math.sqrt(s / n); };
  function rmse2(A, B) { let s = 0, n = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) { s += (A[i][j] - B[i][j]) ** 2; n++; } return Math.sqrt(s / n); }
  /* Pearson correlation between two images — NOT VZ.corr2, which is 2-D correlation */
  function pearson2(A, B) {
    const a = flat(A), b = flat(B);
    const ma = sum(a) / a.length, mb = sum(b) / b.length;
    let sab = 0, sa = 0, sb = 0;
    for (let i = 0; i < a.length; i++) { const u = a[i] - ma, v = b[i] - mb; sab += u * v; sa += u * u; sb += v * v; }
    return sab / Math.sqrt(sa * sb);
  }

  /* ── the two running 64 × 64 pictures. Drawn procedurally so that every reader,
        and the page's own prose, sees exactly the same pixels. ───────────────── */
  const N2 = 64;
  function house() {
    const A = zeros2(N2, N2);
    for (let i = 0; i < N2; i++) for (let j = 0; j < N2; j++) {
      let v = 0.15;
      if (i >= 14 && i < 30) { const hw = (i - 14) * 22 / 16; if (Math.abs(j - 32) <= hw) v = 0.55; }
      if (i >= 30 && i < 58 && j >= 16 && j < 48) v = 0.75;
      if (i >= 34 && i < 42 && ((j >= 20 && j < 28) || (j >= 38 && j < 46))) v = 0.95;
      if (i >= 44 && i < 58 && j >= 28 && j < 36) v = 0.20;
      A[i][j] = v;
    }
    return A;
  }
  function face() {
    const A = zeros2(N2, N2);
    for (let i = 0; i < N2; i++) for (let j = 0; j < N2; j++) {
      let v = 0.85;
      if ((i - 32) ** 2 + (j - 32) ** 2 <= 484) v = 0.35;
      if ((i - 26) ** 2 + (j - 24) ** 2 <= 16) v = 0.95;
      if ((i - 26) ** 2 + (j - 40) ** 2 <= 16) v = 0.95;
      if (i >= 40 && i < 45 && (j - 32) ** 2 <= 121 && (i - 40) >= -((j - 32) ** 2) / 24) v = 0.10;
      A[i][j] = v;
    }
    return A;
  }

  /* ── image drawing: a grey (or signed) raster into an svg <g>, one rect per pixel
        is far too slow at 64², so this paints into a canvas and embeds it. ─────── */
  function raster(g, A, x0, y0, w, h, opt) {
    const o = Object.assign({ signed: false, lo: null, hi: null, gamma: 1 }, opt || {});
    const H = A.length, W = A[0].length;
    let lo = o.lo, hi = o.hi;
    if (lo === null || hi === null) { const mm = minmax2(A); lo = (lo === null) ? mm[0] : lo; hi = (hi === null) ? mm[1] : hi; }
    const cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d"), im = ctx.createImageData(W, H);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const p = 4 * (i * W + j);
      let r, gg, b;
      if (o.signed) {
        const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
        const t = VZ.clamp(A[i][j] / m, -1, 1);
        if (t >= 0) { r = 255 * (0.10 + 0.90 * t); gg = 255 * (0.13 + 0.57 * t); b = 255 * (0.18 + 0.15 * t); }
        else { const u = -t; r = 255 * (0.10 + 0.26 * u); gg = 255 * (0.13 + 0.48 * u); b = 255 * (0.18 + 0.82 * u); }
      } else {
        let t = (hi - lo) > 1e-12 ? (A[i][j] - lo) / (hi - lo) : 0.5;
        t = VZ.clamp(t, 0, 1);
        if (o.gamma !== 1) t = Math.pow(t, o.gamma);
        r = gg = b = 255 * t;
      }
      im.data[p] = r; im.data[p + 1] = gg; im.data[p + 2] = b; im.data[p + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    g.append("image").attr("x", x0).attr("y", y0).attr("width", w).attr("height", h)
      .attr("preserveAspectRatio", "none").attr("image-rendering", "pixelated")
      .attr("href", cv.toDataURL());
    g.append("rect").attr("x", x0).attr("y", y0).attr("width", w).attr("height", h)
      .attr("fill", "none").attr("stroke", VC.line);
    return { lo: lo, hi: hi };
  }
  function caption(g, x, y, txt, col, size) {
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", size || 10.5)
      .attr("fill", col || VC.muted).text(txt);
  }

  return {
    C, cadd, csub, cmul, cscale, cconj, cabs, carg, cexp, cdiv,
    dft, fft, ifft, idft, fft2, ifft2, re2, fftshift2, resp,
    zeros, zeros2, minmax, minmax2, flat, sum, energy, rmse, rmse2, pearson2,
    house, face, raster, caption, N2
  };
})();

/* ═══ 1 · #sig-svg — the defining experiment ═══════════════════════════════════
   A sinusoid goes in, the SAME sinusoid comes out with a new amplitude and a new
   offset. The gain and shift are measured from the drawn arrays by projecting the
   output onto the input's own complex exponential, and compared with the closed
   form Σ h(k)e^{−jwk}. The two must agree to machine precision; the readout says
   by how much they actually do.                                                  */
(function () {
  const svg = d3.select("#sig-svg"); if (svg.empty()) return;
  const out = document.getElementById("sig-readout");
  const eK = document.getElementById("sg-k"), eW = document.getElementById("sg-w"),
        eWv = document.getElementById("sg-wv"), eR = document.getElementById("sg-r");
  const W = 770, H = 430;

  const KERNELS = {
    tent:   { h: [0.25, 0.5, 0.25],                    name: "tent [1,2,1]/4",          closed: w => "(1 + cos w)/2 = " + VZ.fmt((1 + Math.cos(w)) / 2, 4) },
    box3:   { h: [1 / 3, 1 / 3, 1 / 3],                name: "box-3 [1,1,1]/3",         closed: w => "(1 + 2cos w)/3 = " + VZ.fmt((1 + 2 * Math.cos(w)) / 3, 4) },
    box5:   { h: [0.2, 0.2, 0.2, 0.2, 0.2],            name: "box-5 [1,1,1,1,1]/5",     closed: w => "(1 + 2cos w + 2cos 2w)/5 = " + VZ.fmt((1 + 2 * Math.cos(w) + 2 * Math.cos(2 * w)) / 5, 4) },
    binom:  { h: [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16], name: "binomial [1,4,6,4,1]/16", closed: w => "((1 + cos w)/2)² = " + VZ.fmt(Math.pow((1 + Math.cos(w)) / 2, 2), 4) },
    diff:   { h: [-0.5, 0, 0.5],                       name: "central difference [−1,0,1]/2", closed: w => "−j·sin w,  |H| = " + VZ.fmt(Math.abs(Math.sin(w)), 4) },
    corner: { h: [-0.5, 1, -0.5],                      name: "corner [−1,2,−1]/2",      closed: w => "1 − cos w = " + VZ.fmt(1 - Math.cos(w), 4) },
    shift:  { h: [0, 0, 1],                            name: "pure shift h(k)=δ(k−1)",  closed: w => "e^{+jw},  |H| = 1.0000" }
  };

  const NS = 200;                       // samples in the probe, so w = 2πm/NS is exact
  function draw() {
    const key = eK.value, K = KERNELS[key], h = K.h, R = (h.length - 1) >> 1;
    const t = +eW.value;
    const m = Math.max(1, Math.round(t * NS / 2));
    const w = 2 * Math.PI * m / NS;
    eWv.textContent = VZ.fmt(w / Math.PI, 2);

    /* the probe and its filtered version, with wrap-around so the sum is exact */
    const inp = new Float64Array(NS), o = new Float64Array(NS);
    for (let n = 0; n < NS; n++) inp[n] = Math.cos(w * n);
    for (let n = 0; n < NS; n++) {
      let s = 0;
      for (let k = -R; k <= R; k++) s += h[k + R] * inp[((n - k) % NS + NS) % NS];
      o[n] = s;
    }
    /* measure: project both onto e^{−jwn} and divide */
    let ar = 0, ai = 0, br = 0, bi = 0;
    for (let n = 0; n < NS; n++) {
      const c = Math.cos(w * n), s = Math.sin(w * n);
      ar += inp[n] * c; ai += -inp[n] * s;
      br += o[n] * c;   bi += -o[n] * s;
    }
    const Hm = FD.cdiv(FD.C(br, bi), FD.C(ar, ai));
    const Hc = FD.resp(h, w);
    const err = Math.hypot(Hm.re - Hc.re, Hm.im - Hc.im);

    const f = VZ.frame(svg, W, H, { l: 52, r: 18, t: 26, b: 26 });
    const g = f.g;

    /* ── panel A: the two waveforms, overlaid ── */
    const AH = 132, SHOWN = 34;
    const x = d3.scaleLinear().domain([0, SHOWN]).range([0, f.iw - 150]);
    const y = d3.scaleLinear().domain([-1.35, 1.35]).range([AH, 0]);
    const gA = g.append("g");
    gA.append("rect").attr("x", 0).attr("y", 0).attr("width", f.iw - 150).attr("height", AH)
      .attr("fill", VC.panel2).attr("stroke", VC.line);
    gA.append("line").attr("x1", 0).attr("x2", f.iw - 150).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", VC.grid);
    const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveMonotoneX);
    const dense = [], denseO = [];
    for (let u = 0; u <= SHOWN * 8; u++) {
      const xx = u / 8;
      dense.push([xx, Math.cos(w * xx)]);
      /* the filtered continuous curve: same frequency, scaled and shifted */
      denseO.push([xx, FD.cabs(Hc) * Math.cos(w * xx + FD.carg(Hc))]);
    }
    gA.append("path").attr("d", line(dense)).attr("fill", "none").attr("stroke", VC.muted)
      .attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    gA.append("path").attr("d", line(denseO)).attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    for (let n = 0; n <= SHOWN; n++) {
      gA.append("circle").attr("cx", x(n)).attr("cy", y(o[n])).attr("r", 2.3).attr("fill", VC.accent);
    }
    VZ.axisB(gA, x, AH, 8, "x  (samples)");
    VZ.axisL(gA, y, 5, null);
    VZ.legend(gA, [{ color: VC.muted, label: "input  cos(ωx)", dash: "4 3" },
                   { color: VC.accent, label: "output  (h ∗ input)" }], 8, 12, { gap: 14 });

    /* the kernel, as a stem plot, to the right */
    const kx0 = f.iw - 132, kw = 132, kh = AH;
    const gK = g.append("g").attr("transform", `translate(${kx0},0)`);
    gK.append("rect").attr("width", kw).attr("height", kh).attr("fill", VC.panel2).attr("stroke", VC.line);
    FD.caption(gK, 6, 13, "the kernel h(k)", VC.ink, 10.5);
    const hm = Math.max(...h.map(Math.abs));
    const kxs = d3.scaleLinear().domain([-R - 0.7, R + 0.7]).range([16, kw - 12]);
    const kys = d3.scaleLinear().domain([Math.min(-hm * 0.3, -hm), hm * 1.15]).range([kh - 16, 24]);
    gK.append("line").attr("x1", 12).attr("x2", kw - 8).attr("y1", kys(0)).attr("y2", kys(0)).attr("stroke", VC.line);
    h.forEach((v, i) => {
      const k = i - R;
      gK.append("line").attr("x1", kxs(k)).attr("x2", kxs(k)).attr("y1", kys(0)).attr("y2", kys(v))
        .attr("stroke", v >= 0 ? VC.a2 : VC.bad).attr("stroke-width", 3);
      gK.append("circle").attr("cx", kxs(k)).attr("cy", kys(v)).attr("r", 3).attr("fill", v >= 0 ? VC.a2 : VC.bad);
      gK.append("text").attr("x", kxs(k)).attr("y", kh - 4).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", VC.muted).text(k);
    });

    /* ── panels B and C: the whole response ── */
    if (eR.checked) {
      const BY = 176, BH = 96, CY = 306, CH = 78;
      const xs = d3.scaleLinear().domain([0, 1]).range([0, f.iw]);
      const WS = [];
      for (let i = 0; i <= 400; i++) { const ww = Math.PI * i / 400; WS.push([i / 400, FD.resp(h, ww)]); }
      const gmax = Math.max(1, ...WS.map(d => FD.cabs(d[1])));

      const gB = g.append("g").attr("transform", `translate(0,${BY})`);
      const yb = d3.scaleLinear().domain([-Math.max(0.4, gmax * 0.45), gmax * 1.08]).range([BH, 0]);
      gB.append("rect").attr("width", f.iw).attr("height", BH).attr("fill", VC.panel2).attr("stroke", VC.line);
      VZ.gridY(gB, yb, f.iw, 4);
      gB.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", yb(0)).attr("y2", yb(0)).attr("stroke", VC.line);
      const lb = d3.line().x(d => xs(d[0])).y(d => yb(FD.cabs(d[1])));
      const lr = d3.line().x(d => xs(d[0])).y(d => yb(d[1].re));
      gB.append("path").attr("d", lr(WS)).attr("fill", "none").attr("stroke", VC.teal)
        .attr("stroke-width", 1.3).attr("stroke-dasharray", "3 3");
      gB.append("path").attr("d", lb(WS)).attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
      gB.append("line").attr("x1", xs(w / Math.PI)).attr("x2", xs(w / Math.PI)).attr("y1", 0).attr("y2", BH)
        .attr("stroke", VC.a2).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
      gB.append("circle").attr("cx", xs(w / Math.PI)).attr("cy", yb(FD.cabs(Hc))).attr("r", 4).attr("fill", VC.a2);
      VZ.axisL(gB, yb, 4, "gain  |H(ω)|");
      VZ.legend(gB, [{ color: VC.accent, label: "|H(ω)|" }, { color: VC.teal, label: "Re H(ω)", dash: "3 3" }],
        f.iw - 96, 12, { gap: 14 });

      const gC = g.append("g").attr("transform", `translate(0,${CY})`);
      const yc = d3.scaleLinear().domain([-Math.PI * 1.08, Math.PI * 1.08]).range([CH, 0]);
      gC.append("rect").attr("width", f.iw).attr("height", CH).attr("fill", VC.panel2).attr("stroke", VC.line);
      gC.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", yc(0)).attr("y2", yc(0)).attr("stroke", VC.line);
      const lc = d3.line().x(d => xs(d[0])).y(d => yc(FD.carg(d[1])));
      /* break the path where the phase wraps, so the jump is not drawn as a ramp */
      let seg = [];
      WS.forEach((d, i) => {
        if (i > 0 && Math.abs(FD.carg(d[1]) - FD.carg(WS[i - 1][1])) > Math.PI) {
          gC.append("path").attr("d", lc(seg)).attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 2);
          seg = [];
        }
        seg.push(d);
      });
      gC.append("path").attr("d", lc(seg)).attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 2);
      gC.append("line").attr("x1", xs(w / Math.PI)).attr("x2", xs(w / Math.PI)).attr("y1", 0).attr("y2", CH)
        .attr("stroke", VC.a2).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
      gC.append("circle").attr("cx", xs(w / Math.PI)).attr("cy", yc(FD.carg(Hc))).attr("r", 4).attr("fill", VC.a2);
      VZ.axisL(gC, yc, 3, "phase  arg H(ω)  (rad)");
      VZ.axisB(gC, xs, CH, 6, "ω / π          (ω = π is the Nyquist limit)");
    }

    const gain = FD.cabs(Hc), ph = FD.carg(Hc);
    out.innerHTML =
      `<b>${K.name}</b> at <b>ω = ${VZ.fmt(w, 4)}</b> rad/sample (${VZ.fmt(w / (2 * Math.PI), 4)} cycles/sample, bin ${m} of ${NS})<br>`
      + `measured from the two curves above — gain <b>${VZ.fmt(gain, 6)}</b>, phase shift <b>${VZ.fmt(ph, 6)}</b> rad `
      + `(${VZ.fmt(VZ.deg(ph), 2)}°, a delay of ${VZ.fmt(Math.abs(ph) > 1e-12 ? -ph / w : 0, 3)} samples)<br>`
      + `closed form Σₖ h(k)·e^{−jωk} = ${K.closed(w)} &nbsp;·&nbsp; |measured − closed form| = <b>${err.toExponential(3)}</b><br>`
      + (gain < 1e-9
        ? `The gain is zero: this filter annihilates this frequency completely. Nothing at ω = ${VZ.fmt(w, 3)} survives it, which is exactly what "H(ω) = 0" is a statement about.`
        : Math.abs(ph) < 1e-9
          ? `The phase shift is exactly zero, because the kernel is symmetric and real, so H(ω) is real. A symmetric kernel never moves an edge — that is the whole reason vision filters are symmetric.`
          : Math.abs(Math.abs(ph) - Math.PI) < 1e-9
            ? `The phase is exactly ±π: the response is real and NEGATIVE, so this band comes out <b>inverted</b>. A light bar becomes a dark one. This is not attenuation, and no amount of rescaling repairs it.`
            : `The phase is neither 0 nor ±π, so this filter shifts as well as scales. For the derivative it is ±90° at every frequency — the signature of the differentiation property H(ω) ∝ jω. For an asymmetric kernel it is a genuine, frequency-dependent displacement.`);
  }
  eK.onchange = draw; eW.oninput = draw; eR.onchange = draw;
  draw();
})();

/* ═══ 2 · #pair-svg — transform pairs, and the width trade ════════════════════
   Every curve here is a CLOSED FORM, not an FFT, so the picture is the theorem
   rather than an approximation to it. The readout measures the RMS width of each
   side numerically from the plotted samples and prints the product; for the box
   it also recomputes the frequency width over twice the range, to show that the
   number is growing — the box's second moment in frequency genuinely diverges. */
(function () {
  const svg = d3.select("#pair-svg"); if (svg.empty()) return;
  const out = document.getElementById("pair-readout");
  const eP = document.getElementById("pr-p"), eW = document.getElementById("pr-w"),
        eWv = document.getElementById("pr-wv"), eS = document.getElementById("pr-s");
  const W = 770, H = 380;
  const sinc = u => Math.abs(u) < 1e-9 ? 1 : Math.sin(u) / u;
  const SQ2PI = Math.sqrt(2 * Math.PI);

  const PAIRS = {
    box:     { sp: "box of half-width a", fr: "2a·sinc(ωa)",             stemS: 0, stemF: 0,
               f: (x, a) => Math.abs(x) <= a ? 1 : 0,
               F: (w, a) => FD.C(2 * a * sinc(w * a), 0), diverge: true,
               note: "The sinc's side lobes alternate in SIGN and decay only as 1/ω. Bands where it is negative come out inverted, and no truncation width removes them." },
    tent:    { sp: "tent of half-width a", fr: "a·sinc²(ωa/2)",          stemS: 0, stemF: 0,
               f: (x, a) => Math.max(0, 1 - Math.abs(x) / a),
               F: (w, a) => FD.C(a * sinc(w * a / 2) ** 2, 0), diverge: false,
               note: "A tent is a box convolved with itself, so its transform is the box's SQUARED — non-negative everywhere, and decaying as 1/ω² instead of 1/ω. One convolution buys the sign and one order of decay." },
    gauss:   { sp: "G(x; σ = a)", fr: "a√(2π)·e^{−a²ω²/2} = G(ω; 1/a)",  stemS: 0, stemF: 0,
               f: (x, a) => Math.exp(-x * x / (2 * a * a)),
               F: (w, a) => FD.C(a * SQ2PI * Math.exp(-a * a * w * w / 2), 0), diverge: false,
               note: "Its own transform, with the width inverted. Positive everywhere, so no side lobes and no ringing — and the width product below sits exactly on the uncertainty bound, which no other shape reaches." },
    log:     { sp: "∂²/∂x² of a unit-area Gaussian", fr: "−ω²·e^{−a²ω²/2}", stemS: 0, stemF: 0,
               f: (x, a) => (x * x / (a ** 4) - 1 / (a * a)) * Math.exp(-x * x / (2 * a * a)) / (a * SQ2PI),
               F: (w, a) => FD.C(-w * w * Math.exp(-a * a * w * w / 2), 0), diverge: false,
               note: "Two derivatives is (jω)² = −ω². Zero at DC, zero at infinity, one peak at ω = √2/a — a BAND-PASS filter, and that peak is the scale it selects." },
    gabor:   { sp: "cos(ω₀x)·G(x; a),  ω₀ = 4", fr: "½[G(ω−ω₀) + G(ω+ω₀)]", stemS: 0, stemF: 0,
               f: (x, a) => Math.cos(4 * x) * Math.exp(-x * x / (2 * a * a)),
               F: (w, a) => FD.C(a * SQ2PI / 2 * (Math.exp(-a * a * (w - 4) ** 2 / 2) + Math.exp(-a * a * (w + 4) ** 2 / 2)), 0), diverge: false,
               note: "Modulation shifts a spectrum. A Gaussian multiplied by a cosine is two Gaussian bumps at ±ω₀ — the narrowest band you can select at a given spatial extent." },
    impulse: { sp: "δ(x)", fr: "1", stemS: 1, stemF: 0,
               f: (x, a) => 0, F: (w, a) => FD.C(1, 0), diverge: false,
               note: "Flat: an impulse contains every frequency in exactly equal amount. This is why probing a filter with an impulse recovers the kernel — the probe has no preference." },
    shifted: { sp: "δ(x − a)", fr: "e^{−jωa}: |F| = 1, arg F = −ωa", stemS: 2, stemF: 0,
               f: (x, a) => 0, F: (w, a) => FD.cexp(-w * a), diverge: false, phase: true,
               note: "Unit magnitude and LINEAR phase. A pure displacement changes nothing but the phase, and changes it in exact proportion to frequency. Slide a and watch the phase ramp tilt." },
    cos:     { sp: "cos(ω₀x),  ω₀ = 2a", fr: "π[δ(ω−ω₀) + δ(ω+ω₀)]", stemS: 0, stemF: 1,
               f: (x, a) => Math.cos(2 * a * x), F: (w, a) => FD.C(0, 0), diverge: false,
               note: "Two spikes, symmetric about zero. A real sinusoid ALWAYS appears twice — that redundancy is Hermitian symmetry, and it is half of every spectrum picture you will ever look at." }
  };

  function rmsWidth(xs, vals) {                 // √( ∫x²|v|² / ∫|v|² ) over the sampled range
    let num = 0, den = 0;
    for (let i = 0; i < xs.length; i++) { const p = vals[i] * vals[i]; num += xs[i] * xs[i] * p; den += p; }
    return den > 1e-300 ? Math.sqrt(num / den) : NaN;
  }

  function draw() {
    const key = eP.value, P = PAIRS[key], a = +eW.value, logs = eS.value === "log";
    eWv.textContent = VZ.fmt(a, 2);
    const f = VZ.frame(svg, W, H, { l: 46, r: 18, t: 30, b: 34 });
    const g = f.g, gap = 34, pw = (f.iw - gap) / 2, ph = f.ih - 22;

    const XM = 6, WM = 14, M = 900;
    const xs = [], fv = [];
    for (let i = 0; i <= M; i++) { const x = -XM + 2 * XM * i / M; xs.push(x); fv.push(P.f(x, a)); }
    const ws = [], Fv = [], Fa = [], Fp = [];
    for (let i = 0; i <= M; i++) {
      const w = -WM + 2 * WM * i / M, Fz = P.F(w, a);
      ws.push(w); Fv.push(Fz.re); Fa.push(FD.cabs(Fz)); Fp.push(FD.carg(Fz));
    }

    /* ── left: the signal ── */
    const gL = VZ.panel(g, 0, 22, pw, ph, "space:  " + P.sp).g;
    gL.append("rect").attr("width", pw).attr("height", ph).attr("fill", VC.panel2).attr("stroke", VC.line);
    const [flo, fhi] = FD.minmax(fv);
    const spanL = Math.max(1e-6, Math.max(Math.abs(flo), Math.abs(fhi)));
    const xL = d3.scaleLinear().domain([-XM, XM]).range([0, pw]);
    const yL = d3.scaleLinear().domain([-spanL * 1.2, spanL * 1.2]).range([ph, 0]);
    gL.append("line").attr("x1", 0).attr("x2", pw).attr("y1", yL(0)).attr("y2", yL(0)).attr("stroke", VC.grid);
    gL.append("line").attr("x1", xL(0)).attr("x2", xL(0)).attr("y1", 0).attr("y2", ph).attr("stroke", VC.grid);
    if (P.stemS) {
      const x0 = P.stemS === 2 ? VZ.clamp(a, -XM, XM) : 0;
      gL.append("line").attr("x1", xL(x0)).attr("x2", xL(x0)).attr("y1", yL(0)).attr("y2", yL(spanL))
        .attr("stroke", VC.accent).attr("stroke-width", 3);
      VZ.arrow(gL, xL(x0), yL(0), xL(x0), yL(spanL * 1.02), { color: VC.accent, w: 3, head: 7 });
      FD.caption(gL, xL(x0) + 6, yL(spanL) + 4, P.stemS === 2 ? "x = " + VZ.fmt(a, 2) : "x = 0", VC.accent);
    } else {
      gL.append("path").attr("d", d3.line().x((d, i) => xL(xs[i])).y(d => yL(d))(fv))
        .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    }
    VZ.axisB(gL, xL, ph, 7, "x");
    VZ.axisL(gL, yL, 4, null);

    /* ── right: the transform ── */
    const gR = VZ.panel(g, pw + gap, 22, pw, ph, "frequency:  " + P.fr).g;
    gR.append("rect").attr("width", pw).attr("height", ph).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xR = d3.scaleLinear().domain([-WM, WM]).range([0, pw]);
    if (P.stemF) {
      const yR = d3.scaleLinear().domain([-0.2, 1.25]).range([ph, 0]);
      gR.append("line").attr("x1", 0).attr("x2", pw).attr("y1", yR(0)).attr("y2", yR(0)).attr("stroke", VC.grid);
      [-2 * a, 2 * a].forEach(w0 => {
        if (Math.abs(w0) > WM) return;
        VZ.arrow(gR, xR(w0), yR(0), xR(w0), yR(1), { color: VC.accent, w: 3, head: 7 });
        FD.caption(gR, xR(w0) + 5, yR(1) - 3, "ω = " + VZ.fmt(w0, 2), VC.accent);
      });
      VZ.axisL(gR, yR, 3, null);
      VZ.axisB(gR, xR, ph, 7, "ω  (rad / unit length)");
    } else if (logs) {
      const pk = Math.max(...Fa);
      const db = Fa.map(v => 20 * Math.log10(Math.max(v, 1e-12) / Math.max(pk, 1e-12)));
      const yR = d3.scaleLinear().domain([-70, 6]).range([ph, 0]);
      VZ.gridY(gR, yR, pw, 5);
      let seg = [];
      db.forEach((v, i) => {
        if (v < -70) { if (seg.length > 1) gR.append("path").attr("d", d3.line().x(d => xR(ws[d])).y(d => yR(db[d]))(seg)).attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.6); seg = []; }
        else seg.push(i);
      });
      if (seg.length > 1) gR.append("path").attr("d", d3.line().x(d => xR(ws[d])).y(d => yR(db[d]))(seg))
        .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.6);
      VZ.axisL(gR, yR, 5, "|F(ω)|  (dB, peak = 0)");
      VZ.axisB(gR, xR, ph, 7, "ω  (rad / unit length)");
    } else {
      const [Flo, Fhi] = FD.minmax(Fv);
      const spanR = Math.max(1e-9, Math.max(Math.abs(Flo), Math.abs(Fhi)));
      const yR = d3.scaleLinear().domain([-spanR * 1.2, spanR * 1.2]).range([ph, 0]);
      gR.append("line").attr("x1", 0).attr("x2", pw).attr("y1", yR(0)).attr("y2", yR(0)).attr("stroke", VC.grid);
      gR.append("line").attr("x1", xR(0)).attr("x2", xR(0)).attr("y1", 0).attr("y2", ph).attr("stroke", VC.grid);
      gR.append("path").attr("d", d3.line().x((d, i) => xR(ws[i])).y(d => yR(d))(Fv))
        .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
      if (P.phase) {
        const yP = d3.scaleLinear().domain([-Math.PI * 1.1, Math.PI * 1.1]).range([ph, 0]);
        let seg = [];
        Fp.forEach((v, i) => {
          if (i > 0 && Math.abs(v - Fp[i - 1]) > Math.PI) {
            gR.append("path").attr("d", d3.line().x(d => xR(ws[d])).y(d => yP(Fp[d]))(seg))
              .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
            seg = [];
          }
          seg.push(i);
        });
        gR.append("path").attr("d", d3.line().x(d => xR(ws[d])).y(d => yP(Fp[d]))(seg))
          .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
        VZ.legend(gR, [{ color: VC.accent, label: "Re F(ω)" }, { color: VC.violet, label: "arg F(ω)", dash: "4 3" }], 8, 12, { gap: 14 });
      }
      VZ.axisL(gR, yR, 4, null);
      VZ.axisB(gR, xR, ph, 7, "ω  (rad / unit length)");
    }

    /* ── the measured widths ── */
    const sx = P.stemS ? 0 : rmsWidth(xs, fv);
    const sw = rmsWidth(ws, Fa);
    let extra = "";
    if (P.diverge) {
      const ws2 = [], Fa2 = [];
      for (let i = 0; i <= M; i++) { const w = -2 * WM + 4 * WM * i / M; ws2.push(w); Fa2.push(FD.cabs(P.F(w, a))); }
      extra = ` &nbsp;·&nbsp; doubling the frequency range raises that estimate from <b>${VZ.fmt(sw, 4)}</b> to <b>${VZ.fmt(rmsWidth(ws2, Fa2), 4)}</b>: `
            + `the box's second moment in frequency <b>diverges</b>, so it has no finite bandwidth at all.`;
    }
    const prod = sx * sw;
    out.innerHTML =
      `<b>${P.sp}</b> &nbsp;⟷&nbsp; <b>${P.fr}</b> &nbsp;·&nbsp; width parameter a = <b>${VZ.fmt(a, 2)}</b><br>`
      + (P.stemS
        ? `An impulse has zero width in space and infinite width in frequency; the product is the indeterminate 0 × ∞, which is the degenerate end of the uncertainty relation rather than a violation of it.`
        : `measured RMS widths from the drawn samples — <span class="keep">σ</span>_x = <b>${VZ.fmt(sx, 4)}</b>, <span class="keep">σ</span>_ω = <b>${VZ.fmt(sw, 4)}</b>, `
          + `product <b>${VZ.fmt(prod, 4)}</b>${key === "gauss" ? " — exactly the uncertainty bound ½, to four places, for every a" : ""}${extra}`)
      + `<br>${P.note}`;
  }
  eP.onchange = draw; eW.oninput = draw; eS.onchange = draw;
  draw();
})();

/* ═══ 3 · #dft-svg — analysis and reconstruction, bin by bin ═══════════════════ */
(function () {
  const svg = d3.select("#dft-svg"); if (svg.empty()) return;
  const out = document.getElementById("dft-readout");
  const eS = document.getElementById("df-s"), eN = document.getElementById("df-n"),
        eK = document.getElementById("df-k"), eKv = document.getElementById("df-kv");
  const W = 770, H = 430;

  function signal(kind, N) {
    const f = VZ.zeros(N), r = VZ.rng(17);
    if (kind === "run") { const base = [3, 1, 4, 1, 5, 9, 2, 6]; for (let n = 0; n < N; n++) f[n] = base[n % 8]; return f; }
    for (let n = 0; n < N; n++) {
      const t = n / N;
      if (kind === "pulse") f[n] = (t >= 0.3 && t < 0.55) ? 1 : 0;
      else if (kind === "ramp") f[n] = t;
      else if (kind === "step") f[n] = t < 0.5 ? 0.15 : 0.85;
      else f[n] = Math.cos(2 * Math.PI * 3 * t) + 0.55 * Math.cos(2 * Math.PI * 11 * t + 0.7) + 0.18 * VZ.randn(r);
    }
    return f;
  }

  function draw() {
    const N = +eN.value, kind = eS.value, f = signal(kind, N);
    eK.max = String(N / 2);
    let K = VZ.clamp(Math.round(+eK.value), 0, N / 2);
    eKv.textContent = String(K);

    const F = FD.fft(Array.from(f));
    /* reconstruction from DC plus the K lowest conjugate pairs */
    const G = [];
    for (let k = 0; k < N; k++) G.push(FD.C(0, 0));
    G[0] = F[0];
    for (let m = 1; m <= K; m++) { if (m < N) G[m] = F[m]; if (N - m > 0 && N - m < N) G[N - m] = F[N - m]; }
    const rec = FD.ifft(G).map(v => v.re);
    const err = FD.rmse(Array.from(f), rec);

    const fr = VZ.frame(svg, W, H, { l: 46, r: 18, t: 24, b: 30 });
    const g = fr.g;

    /* ── top: signal and reconstruction ── */
    const AH = 118;
    const gA = VZ.panel(g, 0, 12, fr.iw, AH, "the signal (stems) and its reconstruction from the retained bins (curve)").g;
    gA.append("rect").attr("width", fr.iw).attr("height", AH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const all = Array.from(f).concat(rec);
    const [lo, hi] = FD.minmax(all), pad = 0.14 * Math.max(1e-9, hi - lo);
    const xA = d3.scaleLinear().domain([-0.6, N - 0.4]).range([8, fr.iw - 8]);
    const yA = d3.scaleLinear().domain([lo - pad, hi + pad]).range([AH - 6, 8]);
    gA.append("line").attr("x1", 8).attr("x2", fr.iw - 8).attr("y1", yA(0)).attr("y2", yA(0)).attr("stroke", VC.grid);
    for (let n = 0; n < N; n++) {
      gA.append("line").attr("x1", xA(n)).attr("x2", xA(n)).attr("y1", yA(Math.max(0, lo - pad))).attr("y2", yA(f[n]))
        .attr("stroke", VC.muted).attr("stroke-width", Math.min(3, 180 / N));
      gA.append("circle").attr("cx", xA(n)).attr("cy", yA(f[n])).attr("r", Math.min(2.6, 60 / N)).attr("fill", VC.muted);
    }
    gA.append("path").attr("d", d3.line().x((d, i) => xA(i)).y(d => yA(d)).curve(d3.curveMonotoneX)(rec))
      .attr("fill", "none").attr("stroke", K >= N / 2 ? VC.good : VC.accent).attr("stroke-width", 2);
    VZ.axisL(gA, yA, 4, null);

    /* ── middle: the magnitude spectrum, shifted so DC is central ── */
    const BY = 168, BH = 116;
    const gB = VZ.panel(g, 0, BY, fr.iw, BH, "|F(k)|, with the axis rotated so DC is in the middle").g;
    gB.append("rect").attr("width", fr.iw).attr("height", BH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const bins = [];
    for (let k = -Math.floor(N / 2); k < N - Math.floor(N / 2); k++) {
      const idx = ((k % N) + N) % N;
      bins.push({ k: k, f: k / N, mag: FD.cabs(F[idx]), kept: Math.abs(k) <= K });
    }
    const magMax = Math.max(...bins.map(b => b.mag), 1e-9);
    const xB = d3.scaleLinear().domain([-0.52, 0.52]).range([10, fr.iw - 10]);
    const yB = d3.scaleLinear().domain([0, magMax * 1.1]).range([BH - 18, 10]);
    VZ.gridY(gB, yB, fr.iw, 4);
    const bw = Math.max(1.5, (fr.iw - 20) / N - 1.6);
    bins.forEach(b => {
      gB.append("rect").attr("x", xB(b.f) - bw / 2).attr("y", yB(b.mag))
        .attr("width", bw).attr("height", Math.max(0.6, yB(0) - yB(b.mag)))
        .attr("fill", b.kept ? VC.accent : VC.line).attr("fill-opacity", b.kept ? 0.92 : 0.75);
    });
    gB.append("line").attr("x1", xB(0)).attr("x2", xB(0)).attr("y1", 8).attr("y2", BH - 18)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.5);
    FD.caption(gB, xB(0) + 4, 18, "DC", VC.a2);
    [-0.5, 0.5].forEach(v => FD.caption(gB, xB(v) + (v < 0 ? 2 : -50), 18, "Nyquist", VC.muted));
    VZ.axisB(gB, xB, BH - 18, 7, "frequency  (cycles / sample)");
    VZ.axisL(gB, yB, 4, null);

    /* ── bottom: the retained corrugations, stacked ── */
    const CY = 316, CH = 76;
    const gC = VZ.panel(g, 0, CY, fr.iw, CH, "each retained basis corrugation, scaled by its own coefficient").g;
    gC.append("rect").attr("width", fr.iw).attr("height", CH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xC = d3.scaleLinear().domain([0, N - 1]).range([8, fr.iw - 8]);
    let amp = 0;
    const comps = [];
    for (let m = 0; m <= K; m++) {
      const c = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        const th = 2 * Math.PI * m * n / N;
        const v = (F[m].re * Math.cos(th) - F[m].im * Math.sin(th)) / N;
        c[n] = (m === 0 || (N % 2 === 0 && m === N / 2)) ? v : 2 * v;
      }
      comps.push(c);
      amp = Math.max(amp, Math.max(...c.map(Math.abs)));
    }
    const yC = d3.scaleLinear().domain([-amp * 1.1, amp * 1.1]).range([CH - 6, 6]);
    gC.append("line").attr("x1", 8).attr("x2", fr.iw - 8).attr("y1", yC(0)).attr("y2", yC(0)).attr("stroke", VC.grid);
    const cols = [VC.a2, VC.accent, VC.teal, VC.violet, VC.good, VC.rose, VC.lime];
    comps.forEach((c, m) => {
      if (comps.length > 14 && m % Math.ceil(comps.length / 14) !== 0) return;
      const dense = [];
      for (let u = 0; u <= (N - 1) * 6; u++) {
        const x = u / 6;
        const th = 2 * Math.PI * m * x / N;
        const v = (F[m].re * Math.cos(th) - F[m].im * Math.sin(th)) / N;
        dense.push([x, (m === 0 || (N % 2 === 0 && m === N / 2)) ? v : 2 * v]);
      }
      gC.append("path").attr("d", d3.line().x(d => xC(d[0])).y(d => yC(d[1]))(dense))
        .attr("fill", "none").attr("stroke", cols[m % cols.length]).attr("stroke-width", m === 0 ? 2 : 1.2)
        .attr("stroke-opacity", 0.85);
    });
    VZ.axisB(gC, xC, CH, 8, "n  (sample index)");

    const par1 = FD.energy(Array.from(f)), par2 = FD.sum(F.map(v => FD.cabs(v) ** 2)) / N;
    out.innerHTML =
      `N = <b>${N}</b> · keeping DC and the <b>${K}</b> lowest conjugate pairs = <b>${Math.min(N, 1 + 2 * K)}</b> of ${N} coefficients<br>`
      + `F(0) = <b>${VZ.fmt(F[0].re, 4)}</b> against ∑f(n) = <b>${VZ.fmt(FD.sum(Array.from(f)), 4)}</b> (difference ${Math.abs(F[0].re - FD.sum(Array.from(f))).toExponential(2)}) · `
      + `Parseval: ∑|f|² = <b>${VZ.fmt(par1, 4)}</b>, (1/N)∑|F|² = <b>${VZ.fmt(par2, 4)}</b>, relative difference <b>${(Math.abs(par1 - par2) / Math.max(par1, 1e-12)).toExponential(2)}</b><br>`
      + `reconstruction RMSE <b>${err.toExponential(3)}</b> · `
      + (K >= N / 2
        ? `every coefficient is retained, so the reconstruction is <b>exact</b> — the residual above is round-off, not approximation. The DFT is a change of basis, not a model, and it has no error term.`
        : `the discarded bins are not noise: they are the sharp parts. ${kind === "step" || kind === "pulse"
            ? "A discontinuity needs every frequency, so truncating leaves the overshoot beside the jump that §03 named Gibbs' phenomenon — and adding bins moves the ripple closer to the edge without ever shrinking it."
            : kind === "two" ? "Two clean sinusoids occupy two bins each; almost everything else here is the added noise, spread evenly, which is why a low-pass truncation denoises so effectively."
            : "The error concentrates exactly where the signal bends fastest."}`);
  }
  eS.onchange = draw; eN.onchange = draw; eK.oninput = draw;
  draw();
})();

/* ═══ 4 · #circ-svg — the circulant matrix and its universal eigenbasis ════════ */
(function () {
  const svg = d3.select("#circ-svg"); if (svg.empty()) return;
  const out = document.getElementById("circ-readout");
  const eK = document.getElementById("cr-k"), eN = document.getElementById("cr-n"),
        eM = document.getElementById("cr-m"), eMv = document.getElementById("cr-mv");
  const W = 770, H = 400;
  const KER = {
    tent:  { h: [0.25, 0.5, 0.25], name: "tent [1,2,1]/4" },
    box3:  { h: [1 / 3, 1 / 3, 1 / 3], name: "box-3 [1,1,1]/3" },
    binom: { h: [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16], name: "binomial [1,4,6,4,1]/16" },
    diff:  { h: [-0.5, 0, 0.5], name: "central difference [−1,0,1]/2" },
    sharp: { h: [-1, 3, -1], name: "sharpen [−1,3,−1]" }
  };

  function draw() {
    const K = KER[eK.value], h = K.h, R = (h.length - 1) >> 1, N = +eN.value;
    eM.max = String(N - 1);
    const m = VZ.clamp(Math.round(+eM.value), 0, N - 1);
    eMv.textContent = String(m);

    /* build the circulant: C(i, l) = h(i − l mod N), with offsets wrapped */
    const Cm = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let k = -R; k <= R; k++) Cm[i][((i - k) % N + N) % N] += h[k + R];
    /* eigenvalues = DFT of the first column of C */
    const col = VZ.zeros(N);
    for (let i = 0; i < N; i++) col[i] = Cm[i][0];
    const lam = FD.fft(Array.from(col));

    /* the eigenvector and the product C·v, computed directly */
    const vr = VZ.zeros(N), vi = VZ.zeros(N), pr = VZ.zeros(N), pi = VZ.zeros(N);
    for (let n = 0; n < N; n++) { vr[n] = Math.cos(2 * Math.PI * m * n / N); vi[n] = Math.sin(2 * Math.PI * m * n / N); }
    for (let i = 0; i < N; i++) {
      let a = 0, b = 0;
      for (let l = 0; l < N; l++) { a += Cm[i][l] * vr[l]; b += Cm[i][l] * vi[l]; }
      pr[i] = a; pi[i] = b;
    }
    let res = 0;
    for (let i = 0; i < N; i++) {
      res = Math.max(res, Math.hypot(pr[i] - (lam[m].re * vr[i] - lam[m].im * vi[i]),
                                     pi[i] - (lam[m].re * vi[i] + lam[m].im * vr[i])));
    }

    const fr = VZ.frame(svg, W, H, { l: 20, r: 16, t: 26, b: 30 });
    const g = fr.g;
    const cw = 230, gap = 26;

    /* ── left: the matrix ── */
    const gM = VZ.panel(g, 0, 14, cw, cw, "the operator C, as a matrix").g;
    const cs = cw / N, absMax = Math.max(...FD.flat(Cm).map(Math.abs)) || 1;
    for (let i = 0; i < N; i++) for (let l = 0; l < N; l++) {
      const v = Cm[i][l]; if (Math.abs(v) < 1e-12) continue;
      const t = VZ.clamp(v / absMax, -1, 1);
      gM.append("rect").attr("x", l * cs).attr("y", i * cs).attr("width", cs).attr("height", cs)
        .attr("fill", t >= 0 ? VC.accent : VC.bad).attr("fill-opacity", 0.22 + 0.78 * Math.abs(t));
    }
    gM.append("rect").attr("width", cw).attr("height", cw).attr("fill", "none").attr("stroke", VC.line);
    if (N <= 8) for (let i = 0; i < N; i++) for (let l = 0; l < N; l++) {
      if (Math.abs(Cm[i][l]) < 1e-12) continue;
      gM.append("text").attr("x", l * cs + cs / 2).attr("y", i * cs + cs / 2 + 3.5).attr("text-anchor", "middle")
        .attr("font-size", 8.4).attr("fill", VC.ink).text(VZ.fmt(Cm[i][l], 2).replace("0.", "."));
    }
    FD.caption(gM, 0, cw + 14, "every row is the one above, shifted right — a circulant", VC.muted, 10);
    FD.caption(gM, 0, cw + 28, "the corner entries are the wrap-around", VC.muted, 10);

    /* ── centre: the eigenvector and the product ── */
    const x0 = cw + gap, pwv = 236;
    const gV = VZ.panel(g, x0, 14, pwv, 120, "the eigenvector v_m,  m = " + m).g;
    const gP = VZ.panel(g, x0, 172, pwv, 120, "C · v_m  (the same shape, scaled)").g;
    const pmax = Math.max(1e-9, Math.max(...Array.from(pr).map(Math.abs), ...Array.from(pi).map(Math.abs), 1));
    [[gV, vr, vi, 1.15], [gP, pr, pi, pmax * 1.15]].forEach(([gg, a, b, sc]) => {
      gg.append("rect").attr("width", pwv).attr("height", 120).attr("fill", VC.panel2).attr("stroke", VC.line);
      const xx = d3.scaleLinear().domain([-0.5, N - 0.5]).range([8, pwv - 8]);
      const yy = d3.scaleLinear().domain([-sc, sc]).range([112, 8]);
      gg.append("line").attr("x1", 8).attr("x2", pwv - 8).attr("y1", yy(0)).attr("y2", yy(0)).attr("stroke", VC.grid);
      const dense = c => { const o = []; for (let u = 0; u <= (N - 1) * 8; u++) o.push([u / 8, c(u / 8)]); return o; };
      const scl = (gg === gV) ? 1 : 0;
      const lamAbs = FD.cabs(lam[m]), lamArg = FD.carg(lam[m]);
      const cRe = x => (scl ? 1 : lamAbs) * Math.cos(2 * Math.PI * m * x / N + (scl ? 0 : lamArg));
      const cIm = x => (scl ? 1 : lamAbs) * Math.sin(2 * Math.PI * m * x / N + (scl ? 0 : lamArg));
      gg.append("path").attr("d", d3.line().x(d => xx(d[0])).y(d => yy(d[1]))(dense(cRe)))
        .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.6).attr("stroke-opacity", 0.5);
      gg.append("path").attr("d", d3.line().x(d => xx(d[0])).y(d => yy(d[1]))(dense(cIm)))
        .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.6).attr("stroke-opacity", 0.5);
      for (let n = 0; n < N; n++) {
        gg.append("circle").attr("cx", xx(n)).attr("cy", yy(a[n])).attr("r", 2.4).attr("fill", VC.accent);
        gg.append("circle").attr("cx", xx(n)).attr("cy", yy(b[n])).attr("r", 2.4).attr("fill", VC.violet);
      }
      VZ.legend(gg, [{ color: VC.accent, label: "real" }, { color: VC.violet, label: "imaginary" }], pwv - 74, 12, { gap: 13 });
    });

    /* ── right: the spectrum of eigenvalues ── */
    const x1 = x0 + pwv + gap, pwr = fr.iw - x1;
    const gE = VZ.panel(g, x1, 14, pwr, 278, "the eigenvalues H(k)").g;
    gE.append("rect").attr("width", pwr).attr("height", 278).attr("fill", VC.panel2).attr("stroke", VC.line);
    const evs = lam.map((v, k) => ({ k: k, re: v.re, im: v.im, abs: FD.cabs(v) }));
    const emax = Math.max(...evs.map(e => Math.max(Math.abs(e.re), e.abs)), 1e-9);
    const xE = d3.scaleLinear().domain([-0.5, N - 0.5]).range([12, pwr - 10]);
    const yE = d3.scaleLinear().domain([-emax * 1.15, emax * 1.15]).range([248, 12]);
    gE.append("line").attr("x1", 10).attr("x2", pwr - 8).attr("y1", yE(0)).attr("y2", yE(0)).attr("stroke", VC.line);
    VZ.gridY(gE, yE, pwr - 18, 5);
    evs.forEach(e => {
      gE.append("line").attr("x1", xE(e.k)).attr("x2", xE(e.k)).attr("y1", yE(0)).attr("y2", yE(e.re))
        .attr("stroke", e.k === m ? VC.a2 : (e.re >= 0 ? VC.accent : VC.bad)).attr("stroke-width", e.k === m ? 3.4 : 2);
      gE.append("circle").attr("cx", xE(e.k)).attr("cy", yE(e.re)).attr("r", e.k === m ? 4 : 2.6)
        .attr("fill", e.k === m ? VC.a2 : (e.re >= 0 ? VC.accent : VC.bad));
      if (Math.abs(e.im) > 1e-9) {
        gE.append("circle").attr("cx", xE(e.k)).attr("cy", yE(e.im)).attr("r", 2.4)
          .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.4);
      }
    });
    VZ.axisB(gE, xE, 248, Math.min(8, N), "bin k   (frequency k/N)");
    VZ.axisL(gE, yE, 5, "Re H(k)");

    const mags = evs.map(e => e.abs);
    const lo = Math.min(...mags), hi = Math.max(...mags);
    out.innerHTML =
      `<b>${K.name}</b>, N = ${N} · eigenvalue H(${m}) = <b>${VZ.fmt(lam[m].re, 6)}${lam[m].im >= 0 ? " + " : " − "}${VZ.fmt(Math.abs(lam[m].im), 6)}j</b>, `
      + `|H| = <b>${VZ.fmt(FD.cabs(lam[m]), 6)}</b><br>`
      + `residual ‖C·v − H·v‖∞ measured directly from the matrix product above: <b>${res.toExponential(3)}</b> — `
      + `machine precision, so v really is an eigenvector and not merely close to one<br>`
      + `|H| ranges over [<b>${VZ.fmt(lo, 6)}</b>, <b>${VZ.fmt(hi, 6)}</b>], so the condition number is `
      + (lo < 1e-12
        ? `<b>infinite</b>: at least one eigenvalue is exactly zero, the operator is <b>singular</b>, and the direction it annihilates cannot be recovered by any deconvolution. `
          + `That zero is the statement H(π) = 0 — the kernel kills the Nyquist frequency completely.`
        : `<b>${VZ.fmt(hi / lo, 2)}</b>. Inverting this filter amplifies whatever noise sits in its weakest band by that factor, which is the entire difficulty of deblurring.`);
  }
  eK.onchange = draw; eN.onchange = draw; eM.oninput = draw;
  draw();
})();

/* ═══ 5 · #fft-svg — the radix-2 split and the cost it saves ═══════════════════ */
(function () {
  const svg = d3.select("#fft-svg"); if (svg.empty()) return;
  const out = document.getElementById("fft-readout");
  const eN = document.getElementById("ff-n"), eD = document.getElementById("ff-d"),
        eDv = document.getElementById("ff-dv"), eA = document.getElementById("ff-a");
  const W = 770, H = 410;

  function draw() {
    const N = +eN.value;
    const maxD = Math.log2(N);
    eD.max = String(maxD);
    const d = VZ.clamp(Math.round(+eD.value), 0, maxD);
    eDv.textContent = String(d);
    const logAxis = eA.value === "log";

    const fr = VZ.frame(svg, W, H, { l: 18, r: 16, t: 26, b: 32 });
    const g = fr.g;
    const treeW = 300, butW = 176, gap = 22;

    /* ── left: the recursion tree ── */
    const gT = VZ.panel(g, 0, 14, treeW, 300, "the recursion, to depth " + d).g;
    gT.append("rect").attr("width", treeW).attr("height", 300).attr("fill", VC.panel2).attr("stroke", VC.line);
    let cost = 0;
    for (let lev = 0; lev <= d; lev++) {
      const cnt = 1 << lev, len = N >> lev;
      const y = 26 + lev * (300 - 52) / Math.max(1, maxD);
      const bw = Math.min(34, (treeW - 26) / cnt - 3);
      for (let b = 0; b < cnt; b++) {
        const x = 13 + b * ((treeW - 26) / cnt) + ((treeW - 26) / cnt - bw) / 2;
        gT.append("rect").attr("x", x).attr("y", y - 9).attr("width", bw).attr("height", 18).attr("rx", 3)
          .attr("fill", lev === d ? VC.accent : VC.panel).attr("fill-opacity", lev === d ? 0.28 : 1)
          .attr("stroke", lev === d ? VC.accent : VC.line);
        if (bw > 16) gT.append("text").attr("x", x + bw / 2).attr("y", y + 3.5).attr("text-anchor", "middle")
          .attr("font-size", 9).attr("fill", VC.ink).text(len);
        if (lev > 0) {
          const py = 26 + (lev - 1) * (300 - 52) / Math.max(1, maxD);
          const pcnt = cnt / 2, pbw = Math.min(34, (treeW - 26) / pcnt - 3);
          const px = 13 + Math.floor(b / 2) * ((treeW - 26) / pcnt) + ((treeW - 26) / pcnt - pbw) / 2 + pbw / 2;
          gT.append("line").attr("x1", px).attr("y1", py + 9).attr("x2", x + bw / 2).attr("y2", y - 9)
            .attr("stroke", VC.line);
        }
      }
      if (lev > 0) { cost += N / 2; }
      FD.caption(gT, treeW - 96, y + 3.5, cnt + " × length " + len, VC.muted, 9.5);
    }
    FD.caption(gT, 12, 296, "each level costs N/2 = " + (N / 2) + " twiddle multiplies", VC.a2, 10);

    /* ── centre: one butterfly ── */
    const gBx = VZ.panel(g, treeW + gap, 14, butW, 300, "one butterfly").g;
    gBx.append("rect").attr("width", butW).attr("height", 300).attr("fill", VC.panel2).attr("stroke", VC.line);
    const bx0 = 26, bx1 = butW - 30, by0 = 90, by1 = 210;
    gBx.append("circle").attr("cx", bx0).attr("cy", by0).attr("r", 4).attr("fill", VC.accent);
    gBx.append("circle").attr("cx", bx0).attr("cy", by1).attr("r", 4).attr("fill", VC.violet);
    FD.caption(gBx, bx0 - 18, by0 - 12, "E(k)", VC.accent);
    FD.caption(gBx, bx0 - 18, by1 + 20, "O(k)", VC.violet);
    VZ.arrow(gBx, bx0, by0, bx1, by0, { color: VC.accent, w: 1.6 });
    VZ.arrow(gBx, bx0, by1, bx1, by1, { color: VC.violet, w: 1.6 });
    VZ.arrow(gBx, bx0, by1, bx1, by0, { color: VC.violet, w: 1.6, dash: "4 3" });
    VZ.arrow(gBx, bx0, by0, bx1, by1, { color: VC.accent, w: 1.6, dash: "4 3" });
    gBx.append("rect").attr("x", (bx0 + bx1) / 2 - 22).attr("y", by1 - 40).attr("width", 44).attr("height", 17).attr("rx", 3)
      .attr("fill", VC.panel).attr("stroke", VC.a2);
    gBx.append("text").attr("x", (bx0 + bx1) / 2).attr("y", by1 - 28).attr("text-anchor", "middle")
      .attr("font-size", 9.5).attr("fill", VC.a2).text("× ω^k");
    gBx.append("circle").attr("cx", bx1).attr("cy", by0).attr("r", 4).attr("fill", VC.good);
    gBx.append("circle").attr("cx", bx1).attr("cy", by1).attr("r", 4).attr("fill", VC.bad);
    FD.caption(gBx, bx1 - 60, by0 - 12, "F(k) = E + ω^k·O", VC.good);
    FD.caption(gBx, bx1 - 66, by1 + 20, "F(k+N/2) = E − ω^k·O", VC.bad);
    FD.caption(gBx, 12, 268, "one complex multiply,", VC.muted, 10);
    FD.caption(gBx, 12, 282, "one add, one subtract.", VC.muted, 10);
    FD.caption(gBx, 12, 296, "ω^{k+N/2} = −ω^k is why.", VC.muted, 10);

    /* ── right: the cost curves ── */
    const cx0 = treeW + butW + 2 * gap, cw = fr.iw - cx0;
    const gC = VZ.panel(g, cx0, 14, cw, 300, "multiply count against transform length").g;
    gC.append("rect").attr("width", cw).attr("height", 300).attr("fill", VC.panel2).attr("stroke", VC.line);
    const Ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    const direct = Ns.map(n => n * n), fast = Ns.map(n => n / 2 * Math.log2(n));
    const xS = d3.scaleLog().domain([4, 4096]).range([12, cw - 12]);
    const yS = logAxis ? d3.scaleLog().domain([1, 2e7]).range([264, 12])
                       : d3.scaleLinear().domain([0, 1.05e6]).range([264, 12]);
    VZ.gridY(gC, yS, cw - 20, 5);
    const mk = arr => Ns.map((n, i) => [n, Math.max(arr[i], logAxis ? 1 : 0)]);
    const ln = d3.line().x(d => xS(d[0])).y(d => yS(Math.min(d[1], logAxis ? 2e7 : 1.05e6)));
    gC.append("path").attr("d", ln(mk(direct))).attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 2);
    gC.append("path").attr("d", ln(mk(fast))).attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    Ns.forEach((n, i) => {
      if (direct[i] <= (logAxis ? 2e7 : 1.05e6)) gC.append("circle").attr("cx", xS(n)).attr("cy", yS(direct[i])).attr("r", 2.4).attr("fill", VC.bad);
      gC.append("circle").attr("cx", xS(n)).attr("cy", yS(Math.max(fast[i], logAxis ? 1 : 0))).attr("r", 2.4).attr("fill", VC.good);
    });
    gC.append("line").attr("x1", xS(N)).attr("x2", xS(N)).attr("y1", 12).attr("y2", 264)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    VZ.legend(gC, [{ color: VC.bad, label: "direct sum, N²" }, { color: VC.good, label: "recursion, (N/2)log₂N" }], 16, 22, { gap: 14 });
    gC.append("g").attr("class", "axis").attr("transform", "translate(0,264)")
      .call(d3.axisBottom(xS).ticks(5, ".0s"));
    VZ.axisL(gC, yS, 5, "complex multiplies");

    const full = N / 2 * Math.log2(N);
    /* cost of stopping the recursion at depth d: 2^d direct transforms of length N/2^d */
    const partial = cost + (1 << d) * Math.pow(N >> d, 2);
    out.innerHTML =
      `N = <b>${N}</b> · stopping the recursion at depth <b>${d}</b> costs <b>${Math.round(partial).toLocaleString()}</b> complex multiplies `
      + `= ${d} × N/2 twiddles (<b>${cost}</b>) plus ${1 << d} direct transforms of length ${N >> d} (<b>${(1 << d) * Math.pow(N >> d, 2)}</b>)<br>`
      + `recursing all the way to length 1 costs (N/2)·log₂N = <b>${full}</b>, against the direct sum's N² = <b>${N * N}</b> — a factor of <b>${VZ.fmt(N * N / full, 2)}×</b> at this length, `
      + `and <b>${VZ.fmt(1024 * 1024 / (512 * 10), 1)}×</b> at N = 1024<br>`
      + (d === 0
        ? `At depth 0 nothing has been split yet, so the cost is exactly the direct sum. Every level of recursion trades one direct transform for two half-size ones plus N/2 multiplies, and that trade is always favourable because N/2 &lt; (N/2)² for N &gt; 2.`
        : d >= maxD
          ? `Fully recursed. The leaves are length-1 transforms, which are free — a single sample <i>is</i> its own transform. All the arithmetic is now in the log₂N levels of butterflies, and none of it depends on the data.`
          : `Partly recursed: the saving is already ${VZ.fmt(N * N / partial, 2)}×, and it keeps improving because the ${1 << d} remaining direct transforms are the dominant term.`);
  }
  eN.onchange = draw; eD.oninput = draw; eA.onchange = draw;
  draw();
})();

/* ═══ 6 · #spec-svg — an image and its spectrum, with the geometry labelled ════ */
(function () {
  const svg = d3.select("#spec-svg"); if (svg.empty()) return;
  const out = document.getElementById("spec-readout");
  const eI = document.getElementById("sp-i"), eA = document.getElementById("sp-a"),
        eAv = document.getElementById("sp-av"), eP = document.getElementById("sp-p"),
        ePv = document.getElementById("sp-pv"), eD = document.getElementById("sp-d");
  const W = 770, H = 360, N = 64;

  function image(kind, angDeg, per) {
    if (kind === "house") return FD.house();
    if (kind === "face") return FD.face();
    const A = VZ.zeros2(N, N), r = VZ.rng(23);
    const th = VZ.rad(angDeg), cx = Math.cos(th), sy = Math.sin(th);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (kind === "bars") A[i][j] = 0.5 + 0.45 * Math.cos(2 * Math.PI * (j * cx + i * sy) / per);
      else if (kind === "checker") A[i][j] = ((Math.floor(i / 8) + Math.floor(j / 8)) % 2) ? 0.9 : 0.1;
      else if (kind === "disc") A[i][j] = ((i - 32) ** 2 + (j - 32) ** 2 <= 196) ? 0.9 : 0.1;
      else if (kind === "square") A[i][j] = (Math.abs(i - 32) <= 12 && Math.abs(j - 32) <= 12) ? 0.9 : 0.1;
      else if (kind === "impulse") A[i][j] = (i === 32 && j === 32) ? 1 : 0;
      else A[i][j] = VZ.clamp(0.5 + 0.28 * VZ.randn(r), 0, 1);
    }
    return A;
  }

  function draw() {
    const kind = eI.value, ang = +eA.value, per = +eP.value;
    eAv.textContent = ang + "°"; ePv.textContent = String(per);
    const gray = kind === "bars";
    eA.disabled = !gray; eP.disabled = !gray;
    const A = image(kind, ang, per);
    const F = FD.fft2(A);
    const mag = F.map(r => Float64Array.from(r, v => FD.cabs(v)));
    const phs = F.map(r => Float64Array.from(r, v => FD.carg(v)));

    const fr = VZ.frame(svg, W, H, { l: 18, r: 16, t: 26, b: 22 });
    const g = fr.g, S = 220;

    /* left: the image */
    const gI = VZ.panel(g, 0, 22, S, S, "the image  f(x, y)").g;
    FD.raster(gI, A, 0, 0, S, S, {});

    /* centre: the spectrum, DC in the middle */
    const disp = eD.value;
    let shown;
    if (disp === "phase") shown = FD.fftshift2(phs);
    else if (disp === "lin") shown = FD.fftshift2(mag);
    else shown = FD.fftshift2(mag).map(r => Float64Array.from(r, v => Math.log(1 + v)));
    const gS = VZ.panel(g, S + 26, 22, S, S,
      disp === "phase" ? "arg F, DC centred" : (disp === "lin" ? "|F|, linear, DC centred" : "log(1 + |F|), DC centred")).g;
    FD.raster(gS, shown, 0, 0, S, S, { signed: disp === "phase" });
    const px = v => (v + N / 2) * S / N;
    gS.append("circle").attr("cx", px(0)).attr("cy", px(0)).attr("r", (N / 2) * S / N)
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "3 3");
    gS.append("line").attr("x1", 0).attr("x2", S).attr("y1", px(0)).attr("y2", px(0)).attr("stroke", VC.a2).attr("stroke-opacity", 0.25);
    gS.append("line").attr("y1", 0).attr("y2", S).attr("x1", px(0)).attr("x2", px(0)).attr("stroke", VC.a2).attr("stroke-opacity", 0.25);
    FD.caption(gS, px(0) + 4, 12, "k_y", VC.a2, 9.5);
    FD.caption(gS, S - 22, px(0) - 4, "k_x", VC.a2, 9.5);
    FD.caption(gS, 4, S - 5, "the dashed circle is the Nyquist radius, |k| = 32", VC.a2, 9);

    /* find the dominant non-DC coefficient */
    let bk = [0, 0], bm = -1;
    for (let ky = -N / 2; ky < N / 2; ky++) for (let kx = -N / 2; kx < N / 2; kx++) {
      if (kx === 0 && ky === 0) continue;
      const v = mag[((ky % N) + N) % N][((kx % N) + N) % N];
      if (v > bm) { bm = v; bk = [kx, ky]; }
    }
    const rad = Math.hypot(bk[0], bk[1]);
    const wl = rad > 0 ? N / rad : Infinity;
    const orient = VZ.deg(Math.atan2(bk[1], bk[0]));
    if (rad > 0) {
      gS.append("circle").attr("cx", px(bk[0])).attr("cy", px(bk[1])).attr("r", 5)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
      gS.append("circle").attr("cx", px(-bk[0])).attr("cy", px(-bk[1])).attr("r", 5)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
    }

    /* right: the corrugation that dominant coefficient stands for */
    const RX = 2 * S + 52, RW = fr.iw - RX;
    const gC = VZ.panel(g, RX, 22, RW, 104, "the corrugation at that coefficient").g;
    const corr = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
      corr[i][j] = Math.cos(2 * Math.PI * (bk[0] * j + bk[1] * i) / N);
    FD.raster(gC, corr, 0, 0, Math.min(RW, 104), 104, {});
    FD.caption(gC, Math.min(RW, 104) + 8, 16, "|k| = " + VZ.fmt(rad, 2), VC.ink, 10.5);
    FD.caption(gC, Math.min(RW, 104) + 8, 30, "λ = " + (isFinite(wl) ? VZ.fmt(wl, 2) + " px" : "∞"), VC.ink, 10.5);
    FD.caption(gC, Math.min(RW, 104) + 8, 44, "normal " + VZ.fmt(orient, 1) + "°", VC.ink, 10.5);
    FD.caption(gC, Math.min(RW, 104) + 8, 58, "stripes " + VZ.fmt(orient + 90, 1) + "°", VC.a2, 10.5);

    /* radial energy profile */
    const gR = VZ.panel(g, RX, 160, RW, 118, "cumulative energy inside radius |k|").g;
    gR.append("rect").attr("width", RW).attr("height", 118).attr("fill", VC.panel2).attr("stroke", VC.line);
    const tot = FD.sum(FD.flat(mag).map(v => v * v));
    const prof = [];
    for (let r = 0; r <= 46; r++) {
      let s = 0;
      for (let ky = -N / 2; ky < N / 2; ky++) for (let kx = -N / 2; kx < N / 2; kx++) {
        if (Math.hypot(kx, ky) <= r) s += mag[((ky % N) + N) % N][((kx % N) + N) % N] ** 2;
      }
      prof.push([r, s / tot]);
    }
    const xr = d3.scaleLinear().domain([0, 46]).range([10, RW - 10]);
    const yr = d3.scaleLinear().domain([0, 1.03]).range([100, 10]);
    VZ.gridY(gR, yr, RW - 20, 4);
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(d[1]))(prof))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    gR.append("line").attr("x1", xr(32)).attr("x2", xr(32)).attr("y1", 10).attr("y2", 100)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    FD.caption(gR, xr(32) + 3, 22, "Nyquist", VC.a2, 9);
    VZ.axisB(gR, xr, 100, 5, "radius |k| (bins)");
    VZ.axisL(gR, yr, 4, null);

    const pick = r => prof[Math.min(r, 46)][1] * 100;
    out.innerHTML =
      `dominant non-DC coefficient at <b>(k_x, k_y) = (${bk[0]}, ${bk[1]})</b> · |k| = <b>${VZ.fmt(rad, 3)}</b> bins · `
      + `wavelength <b>${isFinite(wl) ? VZ.fmt(wl, 3) + " px" : "∞"}</b> · the corrugation's normal points at <b>${VZ.fmt(orient, 1)}°</b>, so its stripes run at <b>${VZ.fmt(orient + 90, 1)}°</b><br>`
      + `energy inside |k| ≤ 1: <b>${VZ.fmt(pick(1), 2)}%</b> · ≤ 4: <b>${VZ.fmt(pick(4), 2)}%</b> · ≤ 8: <b>${VZ.fmt(pick(8), 2)}%</b> · ≤ 16: <b>${VZ.fmt(pick(16), 2)}%</b> · ≤ 32: <b>${VZ.fmt(pick(32), 2)}%</b><br>`
      + (kind === "bars"
        ? `A pure grating is two coefficients and nothing else — turn the angle and the pair rotates with it, always <b>perpendicular</b> to the stripes you can see. Shorten the period and the pair moves outward: small structures live far from the origin.`
        : kind === "impulse"
          ? `A single bright pixel has a <b>flat</b> spectrum: every coefficient has identical magnitude, ${VZ.fmt(bm, 4)}. That is the impulse pair of §05, and it is why an impulse is the ideal probe.`
          : kind === "noise"
            ? `White noise is flat <i>on average</i> — the speckle here is the finite sample, not structure. Note the energy profile is nearly a straight line in area, which is what "no preferred scale" looks like.`
            : kind === "checker"
              ? `A checkerboard is a grid of impulses in frequency: the period-16 square wave contributes its fundamental and every odd harmonic along both axes, and their cross terms fill the diagonal lattice.`
              : kind === "disc"
                ? `A disc transforms to a radially symmetric ringing pattern — the 2-D analogue of a sinc, with rings rather than lobes. The rings are why a circular low-pass filter causes circular ringing.`
                : kind === "square"
                  ? `A square is separable, so its transform is a product of two sincs: a bright cross along the axes with a faint checker of side lobes between them. The cross is the sinc, not an artefact.`
                  : `A real picture: energy piled at the origin with a ${VZ.fmt(pick(8), 1)}% share inside |k| ≤ 8, plus a bright cross along the axes. That cross is <b>part</b> the discontinuity of the implicit tiling and <b>part</b> the picture's own horizontal and vertical edges — §21 measures the split, and the two cannot be separated by looking.`);
  }
  eI.onchange = draw; eA.oninput = draw; eP.oninput = draw; eD.onchange = draw;
  draw();
})();

/* ═══ 7 · #band-svg — delete a band of frequencies ════════════════════════════ */
(function () {
  const svg = d3.select("#band-svg"); if (svg.empty()) return;
  const out = document.getElementById("band-readout");
  const eI = document.getElementById("bd-i"), eLo = document.getElementById("bd-lo"),
        eLov = document.getElementById("bd-lov"), eHi = document.getElementById("bd-hi"),
        eHiv = document.getElementById("bd-hiv"), eE = document.getElementById("bd-e"),
        eS = document.getElementById("bd-s");
  const W = 770, H = 380, N = 64;

  function draw() {
    const A = eI.value === "house" ? FD.house() : FD.face();
    let lo = Math.round(+eLo.value), hi = Math.round(+eHi.value);
    if (hi <= lo) { hi = lo + 1; eHi.value = String(hi); }
    eLov.textContent = String(lo); eHiv.textContent = String(hi);
    const soft = eE.value === "soft";

    const F = FD.fft2(A);
    const G = [], maskImg = VZ.zeros2(N, N);
    let keptE = 0, totE = 0, keptC = 0;
    for (let i = 0; i < N; i++) {
      G.push(new Array(N));
      for (let j = 0; j < N; j++) {
        const ky = i <= N / 2 ? i : i - N, kx = j <= N / 2 ? j : j - N;
        const r = Math.hypot(kx, ky);
        let m;
        if (soft) {                        // Gaussian roll-off on each side of the band
          const s = Math.max(1.0, (hi - lo) / 3);
          const below = (r >= lo) ? 1 : Math.exp(-((lo - r) ** 2) / (2 * s * s));
          const above = (r <= hi) ? 1 : Math.exp(-((r - hi) ** 2) / (2 * s * s));
          m = below * above;
        } else m = (r >= lo && r <= hi) ? 1 : 0;
        G[i][j] = FD.cscale(F[i][j], m);
        const e = FD.cabs(F[i][j]) ** 2;
        totE += e; keptE += e * m * m; keptC += m > 0.5 ? 1 : 0;
        maskImg[(i + N / 2) % N][(j + N / 2) % N] = m;
      }
    }
    const rec = FD.re2(FD.ifft2(G));
    const res = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) res[i][j] = rec[i][j] - A[i][j];

    const fr = VZ.frame(svg, W, H, { l: 16, r: 14, t: 26, b: 16 });
    const g = fr.g, S = 176, gap = 12;
    const x0 = 0, x1 = S + gap, x2 = 2 * (S + gap), x3 = 3 * (S + gap);

    VZ.panel(g, x0, 26, S, S, "original").g.call(sel => FD.raster(sel, A, 0, 0, S, S, { lo: 0, hi: 1 }));

    /* the mask over the spectrum */
    const gM = VZ.panel(g, x1, 26, S, S, "the surviving coefficients").g;
    const magS = FD.fftshift2(F.map(r => Float64Array.from(r, v => Math.log(1 + FD.cabs(v)))));
    const shownMag = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
      shownMag[i][j] = magS[i][j] * (0.12 + 0.88 * maskImg[i][j]);
    FD.raster(gM, shownMag, 0, 0, S, S, {});
    const pr = r => r * S / N;
    [lo, hi].forEach((r, k) => {
      if (r <= 0 && k === 0) return;
      gM.append("circle").attr("cx", S / 2).attr("cy", S / 2).attr("r", pr(r))
        .attr("fill", "none").attr("stroke", k === 0 ? VC.bad : VC.good).attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4 3");
    });

    if (eS.value === "mag") {
      const gV = VZ.panel(g, x2, 26, S, S, "reconstruction").g;
      FD.raster(gV, rec, 0, 0, S, S, {});
      const gW = VZ.panel(g, x3, 26, S, S, "the deleted part of the spectrum").g;
      const del = VZ.zeros2(N, N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) del[i][j] = magS[i][j] * (1 - maskImg[i][j]);
      FD.raster(gW, del, 0, 0, S, S, {});
    } else {
      const gV = VZ.panel(g, x2, 26, S, S, "reconstruction from those alone").g;
      FD.raster(gV, rec, 0, 0, S, S, { lo: 0, hi: 1 });
      const gW = VZ.panel(g, x3, 26, S, S, "residual (original − reconstruction)").g;
      FD.raster(gW, res, 0, 0, S, S, { signed: true });
    }

    const err = FD.rmse2(A, rec);
    const [rlo, rhi] = FD.minmax2(rec), [alo, ahi] = FD.minmax2(A);
    const over = Math.max(rhi - ahi, alo - rlo);
    const band = lo === 0 ? "low-pass" : (hi >= 45 ? "high-pass" : "band-pass");
    out.innerHTML =
      `a <b>${band}</b> ${soft ? "with a Gaussian roll-off" : "with a hard cut"}, keeping <b>${lo} ≤ |k| ≤ ${hi}</b> — `
      + `<b>${VZ.fmt(100 * keptC / (N * N), 2)}%</b> of the coefficients, carrying <b>${VZ.fmt(100 * keptE / totE, 3)}%</b> of the energy<br>`
      + `reconstruction RMSE <b>${VZ.fmt(err, 5)}</b> · reconstruction range [${VZ.fmt(rlo, 4)}, ${VZ.fmt(rhi, 4)}] against the original's [${VZ.fmt(alo, 2)}, ${VZ.fmt(ahi, 2)}] — `
      + `an overshoot of <b>${VZ.fmt(Math.max(0, over), 4)}</b><br>`
      + (over > 0.02 && !soft
        ? `That overshoot is <b>ringing</b>, and it is not a bug. A hard cut in frequency is a box, and a box in frequency is a sinc in space (§05) — so the reconstruction is the image convolved with an infinitely-oscillating kernel. Switch the edge to a Gaussian roll-off and watch the overshoot collapse while the RMSE barely moves.`
        : soft
          ? `With a smooth roll-off the overshoot is <b>${VZ.fmt(Math.max(0, over), 4)}</b>. The spatial kernel is now a Gaussian rather than a sinc, and a Gaussian has no negative lobes — so there is nothing left to ring. Almost the same coefficients are kept; almost all of the artefact is gone.`
          : `At this setting the cut is not sharp enough relative to the content for the ringing to be visible, but it is present: a hard cut always convolves the image with a sinc.`);
  }
  eI.onchange = draw; eLo.oninput = draw; eHi.oninput = draw; eE.onchange = draw; eS.onchange = draw;
  draw();
})();

/* ═══ 8 · #swap-svg — THE phase swap ══════════════════════════════════════════ */
(function () {
  const svg = d3.select("#swap-svg"); if (svg.empty()) return;
  const out = document.getElementById("swap-readout");
  const eM = document.getElementById("sw-m"), eP = document.getElementById("sw-p"),
        eT = document.getElementById("sw-t"), eTv = document.getElementById("sw-tv"),
        eX = document.getElementById("sw-x");
  const W = 770, H = 380, N = 64;
  const IM = { house: FD.house(), face: FD.face() };
  const FT = { house: FD.fft2(IM.house), face: FD.fft2(IM.face) };

  function draw() {
    const mk = eM.value, pk = eP.value, t = +eT.value;
    eTv.textContent = VZ.fmt(t, 2);
    const A = IM[mk], B = IM[pk], FA = FT[mk], FB = FT[pk];

    /* magnitudes from A; phase interpolated from A's to B's along the unit circle */
    const G = [];
    for (let i = 0; i < N; i++) {
      G.push(new Array(N));
      for (let j = 0; j < N; j++) {
        const a = FA[i][j], b = FB[i][j];
        const ma = FD.cabs(a), mb = FD.cabs(b);
        const ua = ma > 1e-12 ? FD.cscale(a, 1 / ma) : FD.C(1, 0);
        const ub = mb > 1e-12 ? FD.cscale(b, 1 / mb) : FD.C(1, 0);
        let u = FD.cadd(FD.cscale(ua, 1 - t), FD.cscale(ub, t));
        const nu = FD.cabs(u);
        u = nu > 1e-9 ? FD.cscale(u, 1 / nu) : ub;          // antipodal phases: take the target
        G[i][j] = FD.cscale(u, ma);
      }
    }
    const S1 = FD.re2(FD.ifft2(G));

    const fr = VZ.frame(svg, W, H, { l: 16, r: 14, t: 24, b: 14 });
    const g = fr.g, S = 138, sm = 96, gap = 16;

    /* the two sources, with their spectra */
    const drawSrc = (x, img, F, title) => {
      const gg = VZ.panel(g, x, 24, S, S, title).g;
      FD.raster(gg, img, 0, 0, S, S, { lo: 0, hi: 1 });
      const gs = g.append("g").attr("transform", `translate(${x + (S - sm) / 2},${24 + S + 22})`);
      FD.raster(gs, FD.fftshift2(F.map(r => Float64Array.from(r, v => Math.log(1 + FD.cabs(v))))), 0, 0, sm, sm, {});
      FD.caption(gs, 0, -5, "log|F|", VC.muted, 9.5);
    };
    drawSrc(0, A, FA, "magnitude donor: the " + mk);
    drawSrc(S + gap, B, FB, "phase donor: the " + pk);

    /* the result */
    const rx = 2 * (S + gap) + 18;
    const gR = VZ.panel(g, rx, 24, S + 40, S + 40, "|F| from the " + mk + ",  arg F from the " + pk).g;
    FD.raster(gR, S1, 0, 0, S + 40, S + 40, {});
    const gRs = g.append("g").attr("transform", `translate(${rx + (S + 40 - sm) / 2},${24 + S + 62})`);
    FD.raster(gRs, FD.fftshift2(G.map(r => Float64Array.from(r, v => Math.log(1 + FD.cabs(v))))), 0, 0, sm, sm, {});
    FD.caption(gRs, 0, -5, "log|F| of the result — identical to the donor's", VC.muted, 9.5);

    /* the optional extra */
    const ex = eX.value;
    if (ex !== "none") {
      const G2 = [];
      for (let i = 0; i < N; i++) {
        G2.push(new Array(N));
        for (let j = 0; j < N; j++) {
          const a = FA[i][j], ma = FD.cabs(a);
          G2[i][j] = (ex === "phaseonly") ? (ma > 1e-12 ? FD.cscale(a, 1 / ma) : FD.C(1, 0)) : FD.C(ma, 0);
        }
      }
      const E = FD.re2(FD.ifft2(G2));
      const gE = VZ.panel(g, rx + S + 58, 24, 128, 128,
        ex === "phaseonly" ? "phase only, |F| ≡ 1" : "magnitude only, arg F ≡ 0").g;
      FD.raster(gE, E, 0, 0, 128, 128, {});
      FD.caption(gE, 0, 146, "correlation with the " + mk + ":", VC.muted, 9.5);
      FD.caption(gE, 0, 159, VZ.fmt(FD.pearson2(E, A), 4), VC.a2, 11);
    }

    const cA = FD.pearson2(S1, A), cB = FD.pearson2(S1, B);
    const eA = FD.sum(FD.flat(A).map(v => v * v)), eS = FD.sum(FD.flat(S1).map(v => v * v));
    const mA = FD.sum(FD.flat(A)) / (N * N), mS = FD.sum(FD.flat(S1)) / (N * N);
    out.innerHTML =
      `phase mix t = <b>${VZ.fmt(t, 2)}</b> (0 = all ${mk}'s phase, 1 = all ${pk}'s) · correlation of the result with the <b>${mk}</b>: <b>${VZ.fmt(cA, 4)}</b>, `
      + `with the <b>${pk}</b>: <b>${VZ.fmt(cB, 4)}</b><br>`
      + `energy: ∑S² = <b>${VZ.fmt(eS, 3)}</b> against the magnitude donor's ∑A² = <b>${VZ.fmt(eA, 3)}</b>, difference <b>${Math.abs(eS - eA).toExponential(2)}</b> — `
      + `Parseval, exactly: the phase cannot change the energy<br>`
      + `mean: <b>${VZ.fmt(mS, 6)}</b> against the donor's <b>${VZ.fmt(mA, 6)}</b>, difference <b>${Math.abs(mS - mA).toExponential(2)}</b> — the average brightness is inherited from the magnitude<br>`
      + (mk === pk
        ? `Both halves come from the same image, so this is just that image, reconstructed exactly. Change one of the two selectors to make the demonstration.`
        : t > 0.75
          ? `The result follows the <b>${pk}</b> — the image whose <b>phase</b> it borrowed — and is <i>anti</i>-correlated with the ${mk}, whose magnitudes it is entirely made of. The spectrum picture underneath it is the ${mk}'s, unchanged, which is the point: that picture is not what you are looking at.`
          : t < 0.25
            ? `With the phase almost entirely the ${mk}'s, the result is the ${mk} again. Slide t up and watch the picture hand over while the magnitude spectrum below never moves.`
            : `Halfway. The phases are being interpolated on the unit circle, so neither structure is coherent and the result is mush — which is itself the lesson: structure is <b>phase coherence</b>, and a half-aligned set of frequencies is not half a picture.`);
  }
  eM.onchange = draw; eP.onchange = draw; eT.oninput = draw; eX.onchange = draw;
  draw();
})();

/* ═══ 9 · #samp-svg — sampling replicates the spectrum ════════════════════════ */
(function () {
  const svg = d3.select("#samp-svg"); if (svg.empty()) return;
  const out = document.getElementById("samp-readout");
  const eB = document.getElementById("sm-b"), eBv = document.getElementById("sm-bv"),
        eS = document.getElementById("sm-s"), eSv = document.getElementById("sm-sv"),
        eK = document.getElementById("sm-k");
  const W = 770, H = 360;

  function draw() {
    const B = +eB.value, fs = +eS.value, nat = eK.value === "natural";
    eBv.textContent = VZ.fmt(B, 2); eSv.textContent = VZ.fmt(fs, 2);
    /* the base spectrum: a raised-cosine bump, or a 1/f tail that never stops */
    const S = f => {
      const a = Math.abs(f);
      if (nat) return 1 / (1 + (a / (B * 0.35)) ** 2);          // no cutoff at all
      return a >= B ? 0 : 0.5 * (1 + Math.cos(Math.PI * a / B));
    };
    const FM = 2.2, M = 1400;
    const xs = [];
    for (let i = 0; i <= M; i++) xs.push(-FM + 2 * FM * i / M);

    const fr = VZ.frame(svg, W, H, { l: 50, r: 16, t: 24, b: 30 });
    const g = fr.g;
    const PH = 88, GAP = 24;
    const x = d3.scaleLinear().domain([-FM, FM]).range([0, fr.iw]);

    /* panel 1: the original spectrum */
    const g1 = VZ.panel(g, 0, 14, fr.iw, PH, "the continuous signal's spectrum  F(f)").g;
    g1.append("rect").attr("width", fr.iw).attr("height", PH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const y1 = d3.scaleLinear().domain([0, 1.15]).range([PH - 4, 6]);
    g1.append("path").attr("d", d3.line().x(d => x(d)).y(d => y1(S(d)))(xs))
      .attr("fill", VC.accent).attr("fill-opacity", 0.18).attr("stroke", VC.accent).attr("stroke-width", 1.8);
    VZ.axisL(g1, y1, 3, null);

    /* panel 2: the copies */
    const g2 = VZ.panel(g, 0, 14 + PH + GAP, fr.iw, PH, "sampling at f_s replicates it at every multiple of f_s").g;
    g2.append("rect").attr("width", fr.iw).attr("height", PH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const mMax = Math.ceil(FM / fs) + 1;
    for (let m = -mMax; m <= mMax; m++) {
      const pts = xs.map(f => [f, S(f - m * fs)]);
      g2.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y1(d[1]))(pts))
        .attr("fill", "none").attr("stroke", m === 0 ? VC.accent : VC.teal)
        .attr("stroke-width", m === 0 ? 1.9 : 1.2).attr("stroke-opacity", m === 0 ? 1 : 0.65);
    }
    /* the overlap region: where two copies both exceed a threshold */
    const ovl = xs.map(f => {
      let top = 0, second = 0;
      for (let m = -mMax; m <= mMax; m++) { const v = S(f - m * fs); if (v > top) { second = top; top = v; } else if (v > second) second = v; }
      return [f, second];
    });
    g2.append("path").attr("d", d3.area().x(d => x(d[0])).y0(y1(0)).y1(d => y1(d[1]))(ovl))
      .attr("fill", VC.bad).attr("fill-opacity", 0.42);
    g2.append("line").attr("x1", x(fs / 2)).attr("x2", x(fs / 2)).attr("y1", 4).attr("y2", PH)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    g2.append("line").attr("x1", x(-fs / 2)).attr("x2", x(-fs / 2)).attr("y1", 4).attr("y2", PH)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    FD.caption(g2, x(fs / 2) + 4, 14, "new Nyquist, f_s/2", VC.a2, 9.5);
    VZ.axisL(g2, y1, 3, null);

    /* panel 3: their sum — what the sampled signal's spectrum IS */
    const g3 = VZ.panel(g, 0, 14 + 2 * (PH + GAP), fr.iw, PH, "their sum: the spectrum of the sampled signal, against the original").g;
    g3.append("rect").attr("width", fr.iw).attr("height", PH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const tot = xs.map(f => { let s = 0; for (let m = -mMax; m <= mMax; m++) s += S(f - m * fs); return [f, s]; });
    const tm = Math.max(...tot.map(d => d[1]), 1);
    const y3 = d3.scaleLinear().domain([0, tm * 1.12]).range([PH - 4, 6]);
    g3.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y3(S(d[0])))(tot))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
    g3.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y3(d[1]))(tot))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    g3.append("line").attr("x1", x(fs / 2)).attr("x2", x(fs / 2)).attr("y1", 4).attr("y2", PH)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    g3.append("line").attr("x1", x(-fs / 2)).attr("x2", x(-fs / 2)).attr("y1", 4).attr("y2", PH)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    VZ.legend(g3, [{ color: VC.muted, label: "the original F(f)", dash: "4 3" }, { color: VC.good, label: "what was recorded" }], 8, 12, { gap: 13 });
    VZ.axisL(g3, y3, 3, null);
    VZ.axisB(g3, x, PH, 9, "frequency  (cycles per unit length)");

    /* contamination inside the base band */
    let cin = 0, cerr = 0;
    xs.forEach(f => {
      if (Math.abs(f) > fs / 2) return;
      let s = 0; for (let m = -mMax; m <= mMax; m++) s += S(f - m * fs);
      cin += S(f) ** 2; cerr += (s - S(f)) ** 2;
    });
    const frac = cin > 0 ? cerr / cin : 0;
    const ok = !nat && fs > 2 * B + 1e-9;
    out.innerHTML =
      `bandwidth f_max = <b>${VZ.fmt(B, 2)}</b>, sampling rate f_s = <b>${VZ.fmt(fs, 2)}</b>, so f_s / 2f_max = <b>${VZ.fmt(fs / (2 * B), 3)}</b><br>`
      + `contamination inside the base band, as a fraction of the true in-band energy: <b>${VZ.fmt(100 * frac, 4)}%</b><br>`
      + (nat
        ? `This spectrum has <b>no cutoff</b> — it is the 1/f² power law that natural images follow, and it never reaches zero. So the copies <i>always</i> overlap, at every sampling rate, and the contamination above is never zero however far you slide f_s. That is §15's point in one picture: the theorem's hypothesis is false for real images, so there is no safe rate, only a rate whose damage you accept.`
        : ok
          ? `f_s &gt; 2·f_max, so the copies are separated by a gap of <b>${VZ.fmt(fs - 2 * B, 3)}</b> and the green curve lies exactly on the dashed one inside the band. The signal is recoverable <b>exactly</b> — cut out the central copy with the ideal filter of §13 and nothing is lost.`
          : `f_s ≤ 2·f_max, so the copies <b>overlap</b>. In the red region two different frequencies have been added into one number, and addition is not invertible. The green curve is visibly above the dashed one inside the band: that excess is content that belongs elsewhere, now permanently mislabelled.`);
  }
  eB.oninput = draw; eS.oninput = draw; eK.onchange = draw;
  draw();
})();

/* ═══ 10 · #alias-svg — the alias, predicted and measured ═════════════════════
   DEFAULT: a 7 cycles/unit sinusoid sampled 10 times per unit. Nyquist is 5, so
   the default ALIASES, to |((0.7 + 0.5) mod 1) − 0.5| = 0.3 cycles/sample = 3
   cycles/unit. The readout measures the peak bin and must report 3.             */
(function () {
  const svg = d3.select("#alias-svg"); if (svg.empty()) return;
  const out = document.getElementById("alias-readout");
  const eF = document.getElementById("al-f"), eFv = document.getElementById("al-fv"),
        eS = document.getElementById("al-s"), eSv = document.getElementById("al-sv"),
        eV = document.getElementById("al-v"), eP = document.getElementById("al-p");
  const W = 770, H = 400;
  const fold = f => Math.abs(((f + 0.5) % 1 + 1) % 1 - 0.5);

  function draw() {
    const f0 = +eF.value, fs = Math.round(+eS.value), pre = eP.checked;
    eFv.textContent = VZ.fmt(f0, 1); eSv.textContent = String(fs);
    const nyq = fs / 2;
    const fcyc = f0 / fs;                        // cycles per sample
    const app = fold(fcyc) * fs;                 // apparent frequency, cycles per unit

    const fr = VZ.frame(svg, W, H, { l: 48, r: 16, t: 24, b: 28 });
    const g = fr.g;

    if (eV.value === "grid") {
      /* a 2-D grating decimated, with and without a pre-filter */
      const N = 128, r = 4;
      const A = VZ.zeros2(N, N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
        A[i][j] = 0.5 + 0.45 * Math.cos(2 * Math.PI * (j * f0 / N * 4 + i * f0 / N * 1.4));
      const B = pre ? VZ.sep2(A, VZ.gauss1(0.5 * r, 3), "mirror") : A;
      const D = VZ.zeros2(N / r, N / r);
      for (let i = 0; i < N / r; i++) for (let j = 0; j < N / r; j++) D[i][j] = B[i * r][j * r];
      const S = 210;
      VZ.panel(g, 0, 22, S, S, "the source grating, 128 × 128").g.call(s => FD.raster(s, A, 0, 0, S, S, { lo: 0, hi: 1 }));
      VZ.panel(g, S + 26, 22, S, S, pre ? "pre-filtered, then decimated 4×" : "decimated 4× with NO pre-filter").g
        .call(s => FD.raster(s, D, 0, 0, S, S, { lo: 0, hi: 1 }));
      const gS = VZ.panel(g, 2 * (S + 26), 22, fr.iw - 2 * (S + 26), S, "spectrum of the result").g;
      const sz = Math.min(fr.iw - 2 * (S + 26), S);
      FD.raster(gS, FD.fftshift2(FD.fft2(D).map(rr => Float64Array.from(rr, v => Math.log(1 + FD.cabs(v))))), 0, 0, sz, sz, {});
      let e = 0, tot = 0;
      const FD2 = FD.fft2(D), n2 = N / r;
      for (let i = 0; i < n2; i++) for (let j = 0; j < n2; j++) {
        const ky = i <= n2 / 2 ? i : i - n2, kx = j <= n2 / 2 ? j : j - n2;
        const v = FD.cabs(FD2[i][j]) ** 2; tot += v;
        if (Math.hypot(kx, ky) > 1) e += v;
      }
      out.innerHTML =
        `a grating decimated 4× ${pre ? "<b>with</b> a Gaussian pre-filter of σ = 2" : "<b>without</b> any pre-filter"} · `
        + `energy away from DC in the result: <b>${VZ.fmt(100 * e / tot, 3)}%</b><br>`
        + (pre
          ? `The pre-filter removed the doomed band before the samples were taken, so the decimated image is smooth — and the detail is gone, honestly, rather than disguised. That is the correct outcome and it looks worse.`
          : `Without a pre-filter the decimated image carries a <b>coarse pattern that is not in the source</b>. It is sharp, it is convincing, and it is entirely manufactured by the fold. Turn the pre-filter on to see the honest alternative.`);
      return;
    }

    /* ── waveform ── */
    const AH = 168;
    const gA = VZ.panel(g, 0, 20, fr.iw, AH, "the true signal, the samples, and the alias that fits them").g;
    gA.append("rect").attr("width", fr.iw).attr("height", AH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const T = 2;                                       // two units of length shown
    const xw = d3.scaleLinear().domain([0, T]).range([8, fr.iw - 8]);
    const yw = d3.scaleLinear().domain([-1.3, 1.3]).range([AH - 8, 8]);
    gA.append("line").attr("x1", 8).attr("x2", fr.iw - 8).attr("y1", yw(0)).attr("y2", yw(0)).attr("stroke", VC.grid);
    const dense = [], aliasC = [];
    for (let i = 0; i <= 1600; i++) {
      const x = T * i / 1600;
      dense.push([x, Math.cos(2 * Math.PI * f0 * x)]);
      aliasC.push([x, Math.cos(2 * Math.PI * app * x)]);
    }
    gA.append("path").attr("d", d3.line().x(d => xw(d[0])).y(d => yw(d[1]))(dense))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.2).attr("stroke-opacity", 0.8);
    if (Math.abs(app - f0) > 1e-9) {
      gA.append("path").attr("d", d3.line().x(d => xw(d[0])).y(d => yw(d[1]))(aliasC))
        .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 2.4);
    }
    const ns = Math.floor(T * fs) + 1, samp = [];
    for (let n = 0; n < ns; n++) {
      const x = n / fs, v = Math.cos(2 * Math.PI * f0 * x);
      samp.push(v);
      gA.append("line").attr("x1", xw(x)).attr("x2", xw(x)).attr("y1", yw(0)).attr("y2", yw(v))
        .attr("stroke", VC.accent).attr("stroke-width", 1.2).attr("stroke-opacity", 0.6);
      gA.append("circle").attr("cx", xw(x)).attr("cy", yw(v)).attr("r", 3.2).attr("fill", VC.accent);
    }
    VZ.legend(gA, [{ color: VC.muted, label: "the true signal, f = " + VZ.fmt(f0, 1) },
                   { color: VC.bad, label: "the alias, f = " + VZ.fmt(app, 2) },
                   { color: VC.accent, label: "the samples" }], 10, 12, { gap: 13 });
    VZ.axisB(gA, xw, AH, 6, "x  (units of length)");

    /* ── spectrum of the sampled sequence ── */
    const NS = 200;
    const seq = new Float64Array(NS);
    for (let n = 0; n < NS; n++) seq[n] = Math.cos(2 * Math.PI * (f0 / fs) * n);
    const SP = FD.fft(Array.from(seq));
    let pk = 1, pv = -1;
    for (let k = 1; k <= NS / 2; k++) { const v = FD.cabs(SP[k]); if (v > pv) { pv = v; pk = k; } }
    const BY = 214, BH = 118, half = (fr.iw - 26) / 2;
    const gB = VZ.panel(g, 0, BY, half, BH, "the sampled sequence's spectrum").g;
    gB.append("rect").attr("width", half).attr("height", BH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xb = d3.scaleLinear().domain([0, 0.5]).range([10, half - 10]);
    const yb = d3.scaleLinear().domain([0, NS / 2 * 1.15]).range([BH - 20, 10]);
    for (let k = 0; k <= NS / 2; k++) {
      const v = FD.cabs(SP[k]); if (v < 1e-9) continue;
      gB.append("line").attr("x1", xb(k / NS)).attr("x2", xb(k / NS)).attr("y1", yb(0)).attr("y2", yb(v))
        .attr("stroke", k === pk ? VC.bad : VC.accent).attr("stroke-width", k === pk ? 2.6 : 1.2);
    }
    gB.append("line").attr("x1", xb(fold(fcyc))).attr("x2", xb(fold(fcyc))).attr("y1", 6).attr("y2", BH - 20)
      .attr("stroke", VC.good).attr("stroke-dasharray", "3 3");
    FD.caption(gB, 12, 20, "predicted alias bin: " + VZ.fmt(fold(fcyc), 4) + " cycles/sample", VC.good, 9.5);
    VZ.axisB(gB, xb, BH - 20, 5, "cycles / sample   (½ is Nyquist)");

    /* ── the fold rule ── */
    const gC = VZ.panel(g, half + 26, BY, half, BH, "the fold rule").g;
    gC.append("rect").attr("width", half).attr("height", BH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xc = d3.scaleLinear().domain([0, 2.5]).range([12, half - 10]);
    const yc = d3.scaleLinear().domain([0, 0.56]).range([BH - 20, 10]);
    const foldPts = [];
    for (let i = 0; i <= 500; i++) { const t = 2.5 * i / 500; foldPts.push([t, fold(t)]); }
    gC.append("path").attr("d", d3.line().x(d => xc(d[0])).y(d => yc(d[1]))(foldPts))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    gC.append("line").attr("x1", xc(0)).attr("x2", xc(0.5)).attr("y1", yc(0)).attr("y2", yc(0.5))
      .attr("stroke", VC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3");
    gC.append("circle").attr("cx", xc(Math.min(fcyc, 2.5))).attr("cy", yc(fold(fcyc))).attr("r", 4.5).attr("fill", VC.a2);
    VZ.axisB(gC, xc, BH - 20, 6, "true f (cycles/sample)");
    VZ.axisL(gC, yc, 3, "apparent f");

    const aliased = fcyc > 0.5 + 1e-12;
    out.innerHTML =
      `signal <b>${VZ.fmt(f0, 1)}</b> cycles per unit, sampled <b>${fs}</b> times per unit ⇒ <b>${VZ.fmt(fcyc, 4)}</b> cycles/sample; the Nyquist limit is <b>${VZ.fmt(nyq, 1)}</b> cycles per unit<br>`
      + `fold rule predicts <b>${VZ.fmt(fold(fcyc), 4)}</b> cycles/sample = <b>${VZ.fmt(app, 3)}</b> cycles per unit · `
      + `measured peak bin <b>${pk}/${NS} = ${VZ.fmt(pk / NS, 4)}</b> cycles/sample · `
      + `agreement <b>${Math.abs(pk / NS - fold(fcyc)) < 1.0 / NS ? "exact to the bin spacing 1/200 = 0.005" : "off by " + VZ.fmt(Math.abs(pk / NS - fold(fcyc)), 4)}</b><br>`
      + (aliased
        ? `<b>Aliased.</b> ${VZ.fmt(f0, 1)} cycles per unit is above the Nyquist limit of ${VZ.fmt(nyq, 1)}, so it has been recorded as ${VZ.fmt(app, 2)}. The red curve passes through <i>every</i> sample exactly — the two signals are not similar, they are <b>identical on this grid</b>. `
          + `So do <b>${VZ.fmt(app, 2)}</b>, <b>${VZ.fmt(fs - app, 2)}</b>, <b>${VZ.fmt(fs + app, 2)}</b> and <b>${VZ.fmt(2 * fs - app, 2)}</b> — every frequency of the form k·${VZ.fmt(fs, 0)} ± ${VZ.fmt(app, 2)}, for every integer k.`
        : `<b>Not aliased.</b> ${VZ.fmt(f0, 1)} cycles per unit is below the Nyquist limit of ${VZ.fmt(nyq, 1)}, the peak is where it belongs, and §13's sinc sum would reconstruct the continuous curve exactly. Raise the frequency past ${VZ.fmt(nyq, 1)} or lower the rate below ${VZ.fmt(2 * f0, 0)} to break it.`);
  }
  eF.oninput = draw; eS.oninput = draw; eV.onchange = draw; eP.onchange = draw;
  draw();
})();

/* ═══ 11 · #recon-svg — reconstruction from samples ═══════════════════════════ */
(function () {
  const svg = d3.select("#recon-svg"); if (svg.empty()) return;
  const out = document.getElementById("recon-readout");
  const eS = document.getElementById("rc-s"), eF = document.getElementById("rc-f"),
        eFv = document.getElementById("rc-fv"), eK = document.getElementById("rc-k"),
        eC = document.getElementById("rc-c");
  const W = 770, H = 360;
  const sinc = u => Math.abs(u) < 1e-9 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u);

  const KERN = {
    sinc:    { r: 192, k: u => sinc(u), name: "ideal sinc, every tap in the record" },
    sinc4:   { r: 4, k: u => Math.abs(u) <= 4 ? sinc(u) : 0, name: "sinc truncated at 4 lobes" },
    lanczos: { r: 3, k: u => Math.abs(u) < 1e-9 ? 1 : (Math.abs(u) < 3 ? sinc(u) * sinc(u / 3) : 0), name: "Lanczos-3 (windowed sinc)" },
    linear:  { r: 1, k: u => Math.max(0, 1 - Math.abs(u)), name: "linear (tent)" },
    nearest: { r: 1, k: u => Math.abs(u) < 0.5 ? 1 : 0, name: "nearest neighbour" }
  };

  function draw() {
    const kind = eS.value, f0 = +eF.value, K = KERN[eK.value], showC = eC.checked;
    eFv.textContent = VZ.fmt(f0, 3);
    const NS = 385, C = 192;                                // samples, centred index
    const sig = x => {
      if (kind === "one") return Math.cos(2 * Math.PI * f0 * x);
      if (kind === "two") return 0.7 * Math.cos(2 * Math.PI * f0 * x) + 0.45 * Math.cos(2 * Math.PI * (f0 / 2.7) * x + 1.1);
      return Math.exp(-((x - C) ** 2) / (2 * (1.6 / Math.max(f0, 0.02)) ** 2 / 9));
    };
    const s = new Float64Array(NS);
    for (let n = 0; n < NS; n++) s[n] = sig(n);

    const X0 = C - 11, X1 = C + 11, MM = 900;
    const xs = [], truth = [], rec = [];
    for (let i = 0; i <= MM; i++) {
      const x = X0 + (X1 - X0) * i / MM;
      xs.push(x); truth.push(sig(x));
      let v = 0;
      const lo = Math.max(0, Math.ceil(x - K.r)), hi = Math.min(NS - 1, Math.floor(x + K.r));
      for (let n = lo; n <= hi; n++) v += s[n] * K.k(x - n);
      rec.push(v);
    }
    const err = rec.map((v, i) => v - truth[i]);
    const rms = Math.sqrt(err.reduce((a, b) => a + b * b, 0) / err.length);
    const mx = Math.max(...err.map(Math.abs));

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 28 });
    const g = fr.g;
    const AH = 168;
    const gA = VZ.panel(g, 0, 20, fr.iw, AH, "the true signal, its samples, and the reconstruction").g;
    gA.append("rect").attr("width", fr.iw).attr("height", AH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xa = d3.scaleLinear().domain([X0, X1]).range([8, fr.iw - 8]);
    const ya = d3.scaleLinear().domain([-1.5, 1.5]).range([AH - 8, 8]);
    gA.append("line").attr("x1", 8).attr("x2", fr.iw - 8).attr("y1", ya(0)).attr("y2", ya(0)).attr("stroke", VC.grid);
    if (showC) {
      for (let n = Math.ceil(X0); n <= Math.floor(X1); n++) {
        const pts = [];
        for (let i = 0; i <= 240; i++) { const x = X0 + (X1 - X0) * i / 240; pts.push([x, s[n] * K.k(x - n)]); }
        gA.append("path").attr("d", d3.line().x(d => xa(d[0])).y(d => ya(d[1]))(pts))
          .attr("fill", "none").attr("stroke", VC.teal).attr("stroke-width", 0.9).attr("stroke-opacity", 0.55);
      }
    }
    gA.append("path").attr("d", d3.line().x((d, i) => xa(xs[i])).y(d => ya(d))(truth))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
    gA.append("path").attr("d", d3.line().x((d, i) => xa(xs[i])).y(d => ya(d))(rec))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    for (let n = Math.ceil(X0); n <= Math.floor(X1); n++)
      gA.append("circle").attr("cx", xa(n)).attr("cy", ya(s[n])).attr("r", 3).attr("fill", VC.a2);
    VZ.legend(gA, [{ color: VC.muted, label: "true signal", dash: "4 3" }, { color: VC.accent, label: "reconstruction" },
                   { color: VC.a2, label: "samples" }], 10, 12, { gap: 13 });
    VZ.axisB(gA, xa, AH, 8, "x  (samples)");

    /* error + kernel response */
    const BY = 214, BH = 110, half = (fr.iw - 26) / 2;
    const gE = VZ.panel(g, 0, BY, half, BH, "reconstruction error").g;
    gE.append("rect").attr("width", half).attr("height", BH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xe = d3.scaleLinear().domain([X0, X1]).range([10, half - 10]);
    const em = Math.max(mx, 1e-6);
    const ye = d3.scaleLinear().domain([-em * 1.2, em * 1.2]).range([BH - 20, 8]);
    gE.append("line").attr("x1", 10).attr("x2", half - 10).attr("y1", ye(0)).attr("y2", ye(0)).attr("stroke", VC.grid);
    gE.append("path").attr("d", d3.line().x((d, i) => xe(xs[i])).y(d => ye(d))(err))
      .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 1.6);
    VZ.axisL(gE, ye, 3, null);
    VZ.axisB(gE, xe, BH - 20, 5, "x");

    const gR = VZ.panel(g, half + 26, BY, half, BH, "the kernel's frequency response").g;
    gR.append("rect").attr("width", half).attr("height", BH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xr = d3.scaleLinear().domain([0, 1.5]).range([10, half - 10]);
    const yr = d3.scaleLinear().domain([-0.28, 1.15]).range([BH - 20, 8]);
    VZ.gridY(gR, yr, half - 20, 4);
    const resp = [];
    const dx = 0.01;
    for (let i = 0; i <= 300; i++) {
      const fq = 1.5 * i / 300;
      let v = 0;
      for (let u = -K.r; u <= K.r; u += dx) v += K.k(u) * Math.cos(2 * Math.PI * fq * u) * dx;
      resp.push([fq, v]);
    }
    gR.append("line").attr("x1", 10).attr("x2", half - 10).attr("y1", yr(0)).attr("y2", yr(0)).attr("stroke", VC.line);
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(VZ.clamp(d[1], -0.28, 1.15)))(resp))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);
    /* the ideal: 1 below 0.5, 0 above */
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(d[1]))(
      [[0, 1], [0.5, 1], [0.5, 0], [1.5, 0]])).attr("fill", "none").attr("stroke", VC.good)
      .attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    gR.append("line").attr("x1", xr(f0)).attr("x2", xr(f0)).attr("y1", 8).attr("y2", BH - 20)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    VZ.legend(gR, [{ color: VC.accent, label: "this kernel" }, { color: VC.good, label: "ideal", dash: "4 3" }], half - 84, 16, { gap: 13 });
    VZ.axisB(gR, xr, BH - 20, 5, "cycles / sample");

    let gain = 0;
    for (let u = -K.r; u <= K.r; u += dx) gain += K.k(u) * Math.cos(2 * Math.PI * f0 * u) * dx;
    /* for the ideal sinc, measure the same reconstruction with HALF the taps, so that
       the residual can be attributed to the truncation rather than to the theorem */
    let rmsHalf = null;
    if (eK.value === "sinc") {
      let acc = 0;
      for (let i = 0; i <= MM; i++) {
        const x = xs[i], R2 = K.r / 4;
        let v = 0;
        const lo = Math.max(0, Math.ceil(x - R2)), hi = Math.min(NS - 1, Math.floor(x + R2));
        for (let n = lo; n <= hi; n++) v += s[n] * K.k(x - n);
        acc += (v - truth[i]) ** 2;
      }
      rmsHalf = Math.sqrt(acc / (MM + 1));
    }
    out.innerHTML =
      `<b>${K.name}</b> · signal at <b>${VZ.fmt(f0, 3)}</b> cycles/sample (Nyquist is 0.5) · reconstruction RMSE <b>${rms.toExponential(3)}</b>, max error <b>${mx.toExponential(3)}</b><br>`
      + `the kernel's gain at that frequency is <b>${VZ.fmt(gain, 5)}</b>; an ideal interpolator would give exactly 1.0 everywhere below Nyquist and exactly 0 above<br>`
      + (eK.value === "sinc"
        ? `The sampling theorem promises an <b>exact</b> reconstruction, and the residual above is <i>not</i> a failure of that promise — it is the finite record. The sinc's tails fall off only as 1/x, so a sum over finitely many samples is always short by the tail. `
          + `Measured here with a quarter of the taps: RMSE <b>${rmsHalf === null ? "" : rmsHalf.toExponential(3)}</b>, against <b>${rms.toExponential(3)}</b> with all of them. More taps, less error, and no finite number of them is enough — which is exactly why every practical interpolator on this list is a compromise.`
        : eK.value === "nearest"
          ? `Nearest neighbour is a box in space, so its response is a <b>sinc</b>: it attenuates in the pass-band and leaks heavily above Nyquist. The error is a staircase, and its spectrum is exactly the leakage you can see to the right.`
          : eK.value === "linear"
            ? `Linear interpolation is a tent, so its response is sinc² — non-negative, but drooping to ${VZ.fmt(resp.find(d => d[0] >= 0.5)[1], 4)} at Nyquist. It never inverts a band, and it never quite reproduces one either.`
            : `A truncated or windowed sinc approximates the ideal wall, and the residual error is exactly the ripple you can see between the two curves on the right. More lobes ⇒ a sharper wall ⇒ a wider kernel — §21's trade, in its interpolation costume.`);
  }
  eS.onchange = draw; eF.oninput = draw; eK.onchange = draw; eC.onchange = draw;
  draw();
})();

/* ═══ 12 · #mtf-svg — the PSF, the MTF, and the compromise ════════════════════ */
(function () {
  const svg = d3.select("#mtf-svg"); if (svg.empty()) return;
  const out = document.getElementById("mtf-readout");
  const eB = document.getElementById("mt-b"), eBv = document.getElementById("mt-bv"),
        eF = document.getElementById("mt-f"), eFv = document.getElementById("mt-fv"),
        eA = document.getElementById("mt-a"), eS = document.getElementById("mt-s");
  const W = 770, H = 380;

  /* every stage's transfer function, in cycles per pixel */
  function mtf(f, sig, fill, aa) {
    const g = Math.exp(-2 * Math.PI * Math.PI * sig * sig * f * f);
    const u = Math.PI * fill * f;
    const b = Math.abs(u) < 1e-9 ? 1 : Math.sin(u) / u;
    const a = aa ? Math.cos(Math.PI * f) : 1;
    return g * b * a;
  }

  function draw() {
    const sig = +eB.value, fill = +eF.value, aa = eA.checked, nat = eS.value === "nat";
    eBv.textContent = VZ.fmt(sig, 2); eFv.textContent = VZ.fmt(fill, 2);
    const FM = 3, NF = 3000;
    const fs = [], m = [];
    for (let i = 1; i <= NF; i++) { const f = FM * i / NF; fs.push(f); m.push(mtf(f, sig, fill, aa)); }

    /* PSF by inverse cosine transform of the (even, real) MTF */
    const psf = [];
    for (let i = -160; i <= 160; i++) {
      const x = i / 40;
      let v = mtf(0, sig, fill, aa) * (FM / NF);
      for (let k = 0; k < NF; k++) v += 2 * m[k] * Math.cos(2 * Math.PI * fs[k] * x) * (FM / NF);
      psf.push([x, v]);
    }
    const pk = Math.max(...psf.map(d => Math.abs(d[1]))) || 1;

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 30 });
    const g = fr.g;
    const halfW = (fr.iw - 26) / 2;

    /* left: the PSF */
    const gP = VZ.panel(g, 0, 20, halfW, 176, "the point spread function, in pixels").g;
    gP.append("rect").attr("width", halfW).attr("height", 176).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xp = d3.scaleLinear().domain([-4, 4]).range([10, halfW - 10]);
    const yp = d3.scaleLinear().domain([-0.22, 1.12]).range([154, 10]);
    gP.append("line").attr("x1", 10).attr("x2", halfW - 10).attr("y1", yp(0)).attr("y2", yp(0)).attr("stroke", VC.line);
    for (let i = -4; i <= 4; i++) gP.append("line").attr("x1", xp(i + 0.5)).attr("x2", xp(i + 0.5))
      .attr("y1", 10).attr("y2", 154).attr("stroke", VC.grid).attr("stroke-opacity", 0.7);
    gP.append("path").attr("d", d3.area().x(d => xp(d[0])).y0(yp(0)).y1(d => yp(d[1] / pk))(psf))
      .attr("fill", VC.accent).attr("fill-opacity", 0.2);
    gP.append("path").attr("d", d3.line().x(d => xp(d[0])).y(d => yp(d[1] / pk))(psf))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    FD.caption(gP, 14, 24, "grid lines are pixel boundaries", VC.muted, 9.5);
    VZ.axisB(gP, xp, 154, 8, "x  (pixels)");
    VZ.axisL(gP, yp, 4, null);

    /* right: the MTF with the beyond-Nyquist area filled */
    const gM = VZ.panel(g, halfW + 26, 20, halfW, 176, "the modulation transfer function").g;
    gM.append("rect").attr("width", halfW).attr("height", 176).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xm = d3.scaleLinear().domain([0, 1.5]).range([10, halfW - 10]);
    const ym = d3.scaleLinear().domain([0, 1.08]).range([154, 10]);
    VZ.gridY(gM, ym, halfW - 20, 4);
    const curve = fs.map((f, i) => [f, Math.abs(m[i])]).filter(d => d[0] <= 1.5);
    const beyond = curve.filter(d => d[0] >= 0.5);
    gM.append("path").attr("d", d3.area().x(d => xm(d[0])).y0(ym(0)).y1(d => ym(d[1]))(beyond))
      .attr("fill", VC.bad).attr("fill-opacity", 0.42);
    gM.append("path").attr("d", d3.line().x(d => xm(d[0])).y(d => ym(d[1]))(curve))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    gM.append("line").attr("x1", xm(0.5)).attr("x2", xm(0.5)).attr("y1", 10).attr("y2", 154)
      .attr("stroke", VC.a2).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
    FD.caption(gM, xm(0.5) + 5, 24, "Nyquist, 0.5 cycles/pixel", VC.a2, 9.5);
    FD.caption(gM, xm(0.5) + 5, 38, "everything red will fold back", VC.bad, 9.5);
    VZ.axisB(gM, xm, 154, 6, "cycles / pixel");
    VZ.axisL(gM, ym, 4, null);

    /* bottom: the trade curve as sigma sweeps */
    const gT = VZ.panel(g, 0, 226, fr.iw, 118, "the trade: detail retained at half-Nyquist against aliased share, as the blur is swept").g;
    gT.append("rect").attr("width", fr.iw).attr("height", 118).attr("fill", VC.panel2).attr("stroke", VC.line);
    const S = f => nat ? 1 / f : 1;
    const idx = sg => {
      let below = 0, above = 0;
      for (let i = 1; i <= NF; i++) { const f = FM * i / NF, v = Math.abs(S(f) * mtf(f, sg, fill, aa)); if (f <= 0.5) below += v; else above += v; }
      return above / (above + below);
    };
    const pts = [];
    for (let i = 0; i <= 60; i++) { const sg = 3 * i / 60; pts.push([sg, mtf(0.25, sg, fill, aa), idx(sg)]); }
    const xt = d3.scaleLinear().domain([0, 3]).range([12, fr.iw - 12]);
    const yt = d3.scaleLinear().domain([0, 1.05]).range([96, 10]);
    VZ.gridY(gT, yt, fr.iw - 24, 4);
    const tmax = Math.max(...pts.map(p => p[2]), 1e-6);
    gT.append("path").attr("d", d3.line().x(d => xt(d[0])).y(d => yt(d[1]))(pts))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    gT.append("path").attr("d", d3.line().x(d => xt(d[0])).y(d => yt(d[2] / tmax))(pts))
      .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 2);
    gT.append("line").attr("x1", xt(sig)).attr("x2", xt(sig)).attr("y1", 10).attr("y2", 96)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    VZ.legend(gT, [{ color: VC.good, label: "detail kept at f = 0.25" },
                   { color: VC.bad, label: "aliased share (scaled to its own max)" }], fr.iw - 240, 20, { gap: 14 });
    VZ.axisB(gT, xt, 96, 7, "lens blur σ  (pixels)");
    VZ.axisL(gT, yt, 4, null);

    const mn = mtf(0.5, sig, fill, aa), m25 = mtf(0.25, sig, fill, aa);
    const asr = (S(0.75) * Math.abs(mtf(0.75, sig, fill, aa))) / (S(0.25) * Math.abs(m25));
    out.innerHTML =
      `lens blur <span class="keep">σ</span> = <b>${VZ.fmt(sig, 2)}</b> px, fill factor <b>${VZ.fmt(fill, 2)}</b>${aa ? ", anti-alias plate <b>on</b>" : ", no anti-alias plate"}, `
      + `${nat ? "natural 1/f spectrum" : "flat spectrum"}<br>`
      + `MTF at Nyquist <b>${VZ.fmt(mn, 4)}</b> · detail retained at half-Nyquist <b>${VZ.fmt(100 * m25, 2)}%</b> · `
      + `share of the response lying beyond Nyquist <b>${VZ.fmt(100 * idx(sig), 3)}%</b> · `
      + `alias-to-signal ratio at f = 0.25 <b>${VZ.fmt(100 * asr, 3)}%</b><br>`
      + (aa
        ? `The anti-alias plate is a 2-tap box at ±½ pixel, whose response is cos(πf) — and cos(π·0.5) = <b>0 exactly</b>. It annihilates precisely the frequency that folds onto <i>itself</i>, which is the one frequency no other filter can rescue (§15). `
          + `Note that it does <b>not</b> improve the ratio at f = 0.25: cos(π·0.75) = −cos(π·0.25), so it attenuates the alias and the signal there by exactly the same factor.`
        : sig < 0.15
          ? `With almost no blur the response reaches <b>${VZ.fmt(mn, 3)}</b> at Nyquist and keeps going: a large red area, and an alias-to-signal ratio of <b>${VZ.fmt(100 * asr, 1)}%</b> at half-Nyquist. Sharp, and lying about fine detail.`
          : sig > 1.2
            ? `Heavy blur has driven the red area to <b>${VZ.fmt(100 * idx(sig), 4)}%</b> — but it has also thrown away all but <b>${VZ.fmt(100 * m25, 1)}%</b> of the detail at half-Nyquist. This is the far end of the trade, and the image is honest and soft.`
            : `A working compromise: <b>${VZ.fmt(100 * m25, 1)}%</b> of the detail at half-Nyquist survives, and <b>${VZ.fmt(100 * idx(sig), 3)}%</b> of the response still lies beyond Nyquist and will fold. Every camera sits somewhere on this curve, and the choice is an engineering one, not a correct answer.`);
  }
  eB.oninput = draw; eF.oninput = draw; eA.onchange = draw; eS.onchange = draw;
  draw();
})();

/* ═══ 13 · #conv-svg — the convolution theorem, both ways ══════════════════════
   Three results are computed independently: a spatial convolution with ZERO
   padding, a spatial convolution with WRAP padding (both via the shared VZ
   kernel), and the transform route. The readout prints both residuals. The
   frequency route equals the WRAP one to round-off and differs from the ZERO
   one by a real amount at the borders — that is the whole figure.            */
(function () {
  const svg = d3.select("#conv-svg"); if (svg.empty()) return;
  const out = document.getElementById("conv-readout");
  const eS = document.getElementById("cv-s"), eK = document.getElementById("cv-k"),
        eN = document.getElementById("cv-n"), eP = document.getElementById("cv-p");
  const W = 770, H = 410;

  const KER = {
    tent:  () => Float64Array.from([0.25, 0.5, 0.25]),
    box5:  () => VZ.box(5),
    gauss: () => VZ.gauss1(1.5, 2),
    diff:  () => Float64Array.from([-0.5, 0, 0.5])
  };
  function signal(kind, N) {
    const f = VZ.zeros(N), r = VZ.rng(31);
    if (kind === "run") { const b = [3, 1, 4, 1, 5, 9, 2, 6]; for (let n = 0; n < N; n++) f[n] = b[n % 8]; return f; }
    for (let n = 0; n < N; n++) {
      if (kind === "step") f[n] = n < N / 2 ? 1 : 7;
      else if (kind === "pulse") f[n] = (n >= Math.floor(N * 0.3) && n < Math.floor(N * 0.55)) ? 8 : 1;
      else f[n] = 4 + 2.4 * VZ.randn(r);
    }
    return f;
  }
  /* 1-D convolution through the shared 2-D kernel, so this page cannot fork the
     border-mode convention: one row in, one row out. */
  function conv1(f, h, mode) {
    const A = [Float64Array.from(f)], Hk = [Float64Array.from(h)];
    return VZ.conv2(A, Hk, mode)[0];
  }

  function draw() {
    const N = +eN.value, h = KER[eK.value](), K = h.length, R = (K - 1) >> 1;
    const f = signal(eS.value, N), pad = eP.checked;

    const zero = conv1(f, h, "zero");
    const wrap = conv1(f, h, "wrap");

    /* the transform route */
    let freq;
    if (!pad) {
      const hc = VZ.zeros(N);                                   // centred kernel, wrapped
      for (let k = -R; k <= R; k++) hc[((k % N) + N) % N] = h[k + R];
      const Fh = FD.fft(Array.from(hc)), Ff = FD.fft(Array.from(f));
      const Q = Ff.map((v, i) => FD.cmul(v, Fh[i]));
      freq = FD.ifft(Q).map(v => v.re);
    } else {
      const L = N + K - 1;
      const fp = VZ.zeros(L), hp = VZ.zeros(L);
      for (let n = 0; n < N; n++) fp[n] = f[n];
      for (let k = -R; k <= R; k++) hp[((k % L) + L) % L] = h[k + R];
      const Ff = FD.fft(Array.from(fp)), Fh = FD.fft(Array.from(hp));
      const Q = Ff.map((v, i) => FD.cmul(v, Fh[i]));
      const full = FD.ifft(Q).map(v => v.re);
      freq = VZ.zeros(N);
      for (let n = 0; n < N; n++) freq[n] = full[n];
    }

    let rw = 0, rz = 0;
    for (let n = 0; n < N; n++) { rw = Math.max(rw, Math.abs(freq[n] - wrap[n])); rz = Math.max(rz, Math.abs(freq[n] - zero[n])); }

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 30 });
    const g = fr.g;

    /* top: signal and kernel */
    const gA = VZ.panel(g, 0, 18, fr.iw - 160, 96, "the signal f").g;
    gA.append("rect").attr("width", fr.iw - 160).attr("height", 96).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xa = d3.scaleLinear().domain([-0.6, N - 0.4]).range([8, fr.iw - 168]);
    const [flo, fhi] = FD.minmax(f);
    const ya = d3.scaleLinear().domain([Math.min(0, flo) - 0.4, fhi + 0.6]).range([88, 8]);
    for (let n = 0; n < N; n++) {
      gA.append("line").attr("x1", xa(n)).attr("x2", xa(n)).attr("y1", ya(Math.min(0, flo) - 0.4)).attr("y2", ya(f[n]))
        .attr("stroke", VC.accent).attr("stroke-width", Math.min(4, 120 / N));
    }
    VZ.axisL(gA, ya, 3, null);
    const gK = VZ.panel(g, fr.iw - 148, 18, 148, 96, "the kernel h").g;
    gK.append("rect").attr("width", 148).attr("height", 96).attr("fill", VC.panel2).attr("stroke", VC.line);
    const hm = Math.max(...Array.from(h).map(Math.abs));
    const xk = d3.scaleLinear().domain([-R - 0.7, R + 0.7]).range([12, 136]);
    const yk = d3.scaleLinear().domain([Math.min(-hm * 0.4, -hm), hm * 1.2]).range([84, 10]);
    gK.append("line").attr("x1", 8).attr("x2", 140).attr("y1", yk(0)).attr("y2", yk(0)).attr("stroke", VC.line);
    for (let k = -R; k <= R; k++) {
      gK.append("line").attr("x1", xk(k)).attr("x2", xk(k)).attr("y1", yk(0)).attr("y2", yk(h[k + R]))
        .attr("stroke", h[k + R] >= 0 ? VC.a2 : VC.bad).attr("stroke-width", 3);
    }

    /* middle: the three results */
    const gB = VZ.panel(g, 0, 144, fr.iw, 128, "three results: spatial with zero padding, spatial with wrap, and the transform route").g;
    gB.append("rect").attr("width", fr.iw).attr("height", 128).attr("fill", VC.panel2).attr("stroke", VC.line);
    const all = Array.from(zero).concat(Array.from(wrap), Array.from(freq));
    const [lo, hi] = FD.minmax(all);
    const xb = d3.scaleLinear().domain([-0.6, N - 0.4]).range([10, fr.iw - 10]);
    const yb = d3.scaleLinear().domain([lo - 0.5, hi + 0.5]).range([104, 10]);
    const mkl = arr => d3.line().x((d, i) => xb(i)).y(d => yb(d)).curve(d3.curveMonotoneX)(Array.from(arr));
    gB.append("path").attr("d", mkl(zero)).attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 3.4).attr("stroke-opacity", 0.75);
    gB.append("path").attr("d", mkl(wrap)).attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 2);
    gB.append("path").attr("d", mkl(freq)).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3");
    for (let n = 0; n < N; n++) {
      if (Math.abs(freq[n] - zero[n]) > 1e-9) {
        gB.append("line").attr("x1", xb(n)).attr("x2", xb(n)).attr("y1", yb(zero[n])).attr("y2", yb(freq[n]))
          .attr("stroke", VC.bad).attr("stroke-width", 2);
        gB.append("text").attr("x", xb(n) + 4).attr("y", (yb(zero[n]) + yb(freq[n])) / 2)
          .attr("font-size", 9).attr("fill", VC.bad).text(VZ.fmt(Math.abs(freq[n] - zero[n]), 3));
      }
    }
    VZ.legend(gB, [{ color: VC.good, label: "spatial, ZERO padding" }, { color: VC.violet, label: "spatial, WRAP padding" },
                   { color: VC.a2, label: "transform route", dash: "3 3" }], 12, 16, { gap: 14 });
    VZ.axisL(gB, yb, 4, null);
    VZ.axisB(gB, xb, 104, Math.min(9, N), "n");

    /* bottom: the two residuals, per sample, on a log scale */
    const gC = VZ.panel(g, 0, 300, fr.iw, 82, "|transform − spatial|, per sample, on a log scale").g;
    gC.append("rect").attr("width", fr.iw).attr("height", 82).attr("fill", VC.panel2).attr("stroke", VC.line);
    const yc = d3.scaleLog().domain([1e-17, 10]).range([64, 8]);
    VZ.gridY(gC, yc, fr.iw - 20, 4);
    for (let n = 0; n < N; n++) {
      const dw = Math.max(Math.abs(freq[n] - wrap[n]), 1e-17), dz = Math.max(Math.abs(freq[n] - zero[n]), 1e-17);
      gC.append("circle").attr("cx", xb(n)).attr("cy", yc(dw)).attr("r", 2.8).attr("fill", VC.violet);
      gC.append("circle").attr("cx", xb(n)).attr("cy", yc(dz)).attr("r", 2.8).attr("fill", VC.good);
    }
    VZ.legend(gC, [{ color: VC.violet, label: "against WRAP" }, { color: VC.good, label: "against ZERO" }], fr.iw - 150, 16, { gap: 13 });
    gC.append("g").attr("class", "axis").attr("transform", "translate(0,64)").call(d3.axisBottom(xb).ticks(Math.min(9, N)));
    VZ.axisL(gC, yc, 4, null);

    out.innerHTML =
      `N = <b>${N}</b>, kernel length <b>${K}</b>${pad ? `, zero-padded to N + K − 1 = <b>${N + K - 1}</b> before transforming` : ", <b>no padding</b>"}<br>`
      + `‖ transform − spatial(wrap) ‖∞ = <b>${rw.toExponential(3)}</b> &nbsp;·&nbsp; ‖ transform − spatial(zero) ‖∞ = <b>${rz.toExponential(3)}</b><br>`
      + (pad
        ? `With the zero padding in place the transform now reproduces the <b>zero-padded spatial convolution</b> to <b>${rz.toExponential(3)}</b> — round-off. The wrap-around has been pushed onto the ${K - 1} zeros at the end, where it does no harm. This is the correct recipe, and it costs a transform of length ${N + K - 1} instead of ${N}.`
        : `Without padding, the transform reproduces the <b>wrap</b> convolution to <b>${rw.toExponential(3)}</b> — round-off, because they are the same operator — and differs from the <b>zero-padded</b> one by <b>${rz.toExponential(3)}</b>, which is not round-off and never will be. `
          + `The disagreement is confined to the ${K - 1} samples at the two ends, exactly the kernel's reach. Tick the padding box to fix it.`);
  }
  eS.onchange = draw; eK.onchange = draw; eN.onchange = draw; eP.onchange = draw;
  draw();
})();

/* ═══ 14 · #resp-svg — drag the taps, watch the response and the ringing ══════ */
(function () {
  const svg = d3.select("#resp-svg"); if (svg.empty()) return;
  const out = document.getElementById("resp-readout");
  const eP = document.getElementById("rs-p"), eN = document.getElementById("rs-n"),
        eU = document.getElementById("rs-u");
  const W = 770, H = 400;
  /* tapScale is fixed when the preset changes and NOT recomputed while dragging —
     otherwise the axis rescales under the handle and the point runs away. */
  let taps = null, curN = null, curP = null, tapScale = null;

  function preset(name, K) {
    const R = (K - 1) >> 1, h = VZ.zeros(K);
    if (name === "tent") { for (let k = -R; k <= R; k++) h[k + R] = Math.max(0, 1 - Math.abs(k) / (R + 1)); }
    else if (name === "box5") { for (let k = 0; k < K; k++) h[k] = 1; }
    else if (name === "gauss") { for (let k = -R; k <= R; k++) h[k + R] = Math.exp(-k * k / (2 * 1.44)); }
    else if (name === "ideal") { for (let k = -R; k <= R; k++) { const u = k / 2; h[k + R] = (k === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u)) * (0.5 + 0.5 * Math.cos(Math.PI * k / (R + 1))); } }
    else if (name === "diff") { for (let k = -R; k <= R; k++) h[k + R] = 0; h[R - 1] = -0.5; h[R + 1] = 0.5; }
    else if (name === "log") { for (let k = -R; k <= R; k++) { const s = 1.3; h[k + R] = (k * k / (s ** 4) - 1 / (s * s)) * Math.exp(-k * k / (2 * s * s)); } }
    else {                                   // unsharp mask:  (1 + γ)·δ − γ·G,  γ = 1
      const G = VZ.gauss1(1.0, R);
      const GR = (G.length - 1) >> 1;
      for (let k = -R; k <= R; k++) h[k + R] = (Math.abs(k) <= GR ? -G[k + GR] : 0);
      h[R] += 2.0;
    }
    return h;
  }
  function normalise(h) {
    if (!eU.checked) return h;
    const s = FD.sum(Array.from(h));
    if (Math.abs(s) < 1e-9) return h;
    return Float64Array.from(h, v => v / s);
  }

  function draw() {
    const K = +eN.value, name = eP.value;
    if (taps === null || curN !== K || curP !== name) {
      taps = preset(name, K); curN = K; curP = name;
      tapScale = Math.max(...Array.from(taps).map(Math.abs), 0.4) * 1.6;
    }
    const h = normalise(taps), R = (K - 1) >> 1;

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 30 });
    const g = fr.g;
    const halfW = (fr.iw - 26) / 2;

    /* left: the draggable taps */
    const gK = VZ.panel(g, 0, 18, halfW, 190, "the kernel — drag any tap").g;
    gK.append("rect").attr("width", halfW).attr("height", 190).attr("fill", VC.panel2).attr("stroke", VC.line);
    const tm = tapScale;
    const xk = d3.scaleLinear().domain([-R - 0.7, R + 0.7]).range([16, halfW - 16]);
    const yk = d3.scaleLinear().domain([-tm, tm]).range([174, 12]);
    VZ.gridY(gK, yk, halfW - 20, 4);
    gK.append("line").attr("x1", 10).attr("x2", halfW - 10).attr("y1", yk(0)).attr("y2", yk(0)).attr("stroke", VC.line);
    for (let k = -R; k <= R; k++) {
      const v = taps[k + R];
      gK.append("line").attr("x1", xk(k)).attr("x2", xk(k)).attr("y1", yk(0)).attr("y2", yk(v))
        .attr("stroke", v >= 0 ? VC.a2 : VC.bad).attr("stroke-width", 3);
      gK.append("circle").attr("class", "dragpt").attr("cx", xk(k)).attr("cy", yk(v)).attr("r", 7)
        .attr("fill", v >= 0 ? VC.a2 : VC.bad).attr("stroke", VC.bg).attr("stroke-width", 1.5)
        .style("cursor", "ns-resize")
        .call(d3.drag().on("start drag", function (ev) {
          taps[k + R] = VZ.clamp(yk.invert(ev.y), -tm, tm);
          draw();
        }));
      gK.append("text").attr("x", xk(k)).attr("y", 186).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", VC.muted).text(k);
    }
    VZ.axisL(gK, yk, 4, null);

    /* right: the frequency response */
    const gR = VZ.panel(g, halfW + 26, 18, halfW, 190, "its frequency response H(ω)").g;
    gR.append("rect").attr("width", halfW).attr("height", 190).attr("fill", VC.panel2).attr("stroke", VC.line);
    const pts = [];
    for (let i = 0; i <= 400; i++) { const w = Math.PI * i / 400; pts.push([i / 400, FD.resp(h, w)]); }
    const gmax = Math.max(1.05, ...pts.map(d => FD.cabs(d[1])));
    const xr = d3.scaleLinear().domain([0, 1]).range([12, halfW - 12]);
    const yr = d3.scaleLinear().domain([-Math.max(0.45, gmax * 0.5), gmax * 1.08]).range([170, 12]);
    VZ.gridY(gR, yr, halfW - 24, 4);
    const neg = pts.filter(d => d[1].re < 0).map(d => [d[0], d[1].re]);
    if (neg.length) {
      gR.append("path").attr("d", d3.area().x(d => xr(d[0])).y0(yr(0)).y1(d => yr(d[1]))(neg))
        .attr("fill", VC.bad).attr("fill-opacity", 0.3);
    }
    gR.append("line").attr("x1", 10).attr("x2", halfW - 10).attr("y1", yr(0)).attr("y2", yr(0)).attr("stroke", VC.line);
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(d[1].re))(pts))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(FD.cabs(d[1])))(pts))
      .attr("fill", "none").attr("stroke", VC.teal).attr("stroke-width", 1.3).attr("stroke-dasharray", "3 3");
    VZ.legend(gR, [{ color: VC.accent, label: "Re H(ω)" }, { color: VC.teal, label: "|H(ω)|", dash: "3 3" }], halfW - 96, 18, { gap: 13 });
    VZ.axisL(gR, yr, 4, null);
    VZ.axisB(gR, xr, 170, 6, "ω / π");

    /* bottom: a step edge, filtered */
    const NL = 96;
    const row = VZ.zeros(NL);
    for (let n = 0; n < NL; n++) row[n] = n < NL / 2 ? 0.18 : 0.82;
    const filt = VZ.conv2([row], [Float64Array.from(h)], "clamp")[0];
    const gS = VZ.panel(g, 0, 238, fr.iw, 118, "a step edge, filtered by this kernel — ringing shows up here first").g;
    gS.append("rect").attr("width", fr.iw).attr("height", 118).attr("fill", VC.panel2).attr("stroke", VC.line);
    const [flo, fhi] = FD.minmax(filt);
    const xs = d3.scaleLinear().domain([NL / 2 - 22, NL / 2 + 22]).range([10, fr.iw - 10]);
    const ys = d3.scaleLinear().domain([Math.min(0.1, flo) - 0.08, Math.max(0.9, fhi) + 0.08]).range([100, 10]);
    [0.18, 0.82].forEach(v => gS.append("line").attr("x1", 10).attr("x2", fr.iw - 10).attr("y1", ys(v)).attr("y2", ys(v))
      .attr("stroke", VC.grid).attr("stroke-dasharray", "3 3"));
    gS.append("path").attr("d", d3.line().x((d, i) => xs(i)).y(d => ys(d)).curve(d3.curveMonotoneX)(Array.from(filt)))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    gS.append("path").attr("d", d3.line().x((d, i) => xs(i)).y(d => ys(d))(Array.from(row)))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.2).attr("stroke-dasharray", "4 3");
    VZ.axisL(gS, ys, 4, null);
    VZ.axisB(gS, xs, 100, 6, "n");

    const H0 = FD.resp(h, 0), Hpi = FD.resp(h, Math.PI);
    let minRe = Infinity, minAt = 0;
    pts.forEach(d => { if (d[1].re < minRe) { minRe = d[1].re; minAt = d[0]; } });
    const over = Math.max(0, Math.max(...Array.from(filt).slice(10, NL - 10)) - 0.82,
                          0.18 - Math.min(...Array.from(filt).slice(10, NL - 10)));
    out.innerHTML =
      `∑h = <b>${VZ.fmt(FD.sum(Array.from(h)), 6)}</b> · H(0) = <b>${VZ.fmt(H0.re, 5)}</b> · H(π) = <b>${VZ.fmt(Hpi.re, 5)}</b> · `
      + `deepest negative excursion <b>${VZ.fmt(Math.min(0, minRe), 5)}</b> at ω/π = <b>${VZ.fmt(minAt, 3)}</b><br>`
      + `overshoot beside the step: <b>${VZ.fmt(over, 5)}</b> on a step of height 0.64 = <b>${VZ.fmt(100 * over / 0.64, 2)}%</b><br>`
      + (over > 1e-4
        ? `The response goes <b>negative</b>, so this filter <i>inverts</i> the bands where it does, and that inversion is what the overshoot beside the step is. A kernel with any negative tap can ring; a kernel whose taps are all non-negative <b>cannot</b>, because its response is a positive combination of cosines — which is the Gaussian's whole advantage (§19).`
        : `No overshoot: every tap is non-negative and the response never changes sign, so nothing is inverted and the step is merely softened. Drag any tap below zero and the ringing appears immediately.`)
      + (Math.abs(Hpi.re) < 1e-6
        ? ` H(π) is <b>zero</b>, so this kernel annihilates the Nyquist frequency and is a legitimate decimation pre-filter (Part 2 §12).`
        : ` H(π) = ${VZ.fmt(Hpi.re, 4)} ≠ 0, so the alternating pattern survives this filter and decimating after it will alias.`);
  }
  eP.onchange = () => { taps = null; tapScale = null; draw(); };
  eN.onchange = () => { taps = null; tapScale = null; draw(); };
  eU.onchange = draw;
  draw();
})();

/* ═══ 15 · #gauss-svg — the Gaussian's self-transform, and its two failures ════ */
(function () {
  const svg = d3.select("#gauss-svg"); if (svg.empty()) return;
  const out = document.getElementById("gauss-readout");
  const eS = document.getElementById("gs-s"), eSv = document.getElementById("gs-sv"),
        eT = document.getElementById("gs-t"), eTv = document.getElementById("gs-tv"),
        eC = document.getElementById("gs-c");
  const W = 770, H = 380;

  function draw() {
    const sig = +eS.value, tr = +eT.value, cmp = eC.value;
    eSv.textContent = VZ.fmt(sig, 1); eTv.textContent = VZ.fmt(tr, 2) + "σ";
    const h = VZ.gauss1(sig, tr), R = (h.length - 1) >> 1;
    /* the mass thrown away by truncating at tr·sigma, before renormalisation */
    let inMass = 0, allMass = 0;
    for (let k = -400; k <= 400; k++) { const v = Math.exp(-k * k / (2 * sig * sig)); allMass += v; if (Math.abs(k) <= R) inMass += v; }
    const lost = 1 - inMass / allMass;

    /* responses */
    const NW = 500, ws = [], Hk = [], ideal = [], perio = [];
    for (let i = 0; i <= NW; i++) {
      const w = Math.PI * i / NW;
      ws.push(w);
      Hk.push(FD.resp(h, w).re);
      ideal.push(Math.exp(-sig * sig * w * w / 2));
      let s = 0, n = 0;
      for (let m = -6; m <= 6; m++) { s += Math.exp(-sig * sig * (w + 2 * Math.PI * m) ** 2 / 2); n += Math.exp(-sig * sig * (2 * Math.PI * m) ** 2 / 2); }
      perio.push(s / n);
    }
    let dIdeal = 0, dPer = 0, minH = Infinity, minAt = 0;
    for (let i = 0; i <= NW; i++) {
      dIdeal = Math.max(dIdeal, Math.abs(Hk[i] - ideal[i]));
      dPer = Math.max(dPer, Math.abs(Hk[i] - perio[i]));
      if (Hk[i] < minH) { minH = Hk[i]; minAt = ws[i] / Math.PI; }
    }

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 30 });
    const g = fr.g, halfW = (fr.iw - 26) / 2;

    /* left: the kernel */
    const gK = VZ.panel(g, 0, 18, halfW, 176, "the kernel: continuous, sampled, truncated").g;
    gK.append("rect").attr("width", halfW).attr("height", 176).attr("fill", VC.panel2).attr("stroke", VC.line);
    const XM = Math.max(R + 2, 3 * sig);
    const xk = d3.scaleLinear().domain([-XM, XM]).range([10, halfW - 10]);
    const peak = Math.max(...Array.from(h));
    const yk = d3.scaleLinear().domain([-peak * 0.15, peak * 1.2]).range([154, 10]);
    const cont = [];
    for (let i = 0; i <= 400; i++) { const x = -XM + 2 * XM * i / 400; cont.push([x, peak * Math.exp(-x * x / (2 * sig * sig))]); }
    gK.append("path").attr("d", d3.area().x(d => xk(d[0])).y0(yk(0)).y1(d => yk(d[1]))(cont.filter(d => Math.abs(d[0]) > R)))
      .attr("fill", VC.bad).attr("fill-opacity", 0.3);
    gK.append("path").attr("d", d3.line().x(d => xk(d[0])).y(d => yk(d[1]))(cont))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
    for (let k = -R; k <= R; k++) {
      gK.append("line").attr("x1", xk(k)).attr("x2", xk(k)).attr("y1", yk(0)).attr("y2", yk(h[k + R]))
        .attr("stroke", VC.accent).attr("stroke-width", Math.min(4, 120 / h.length));
      gK.append("circle").attr("cx", xk(k)).attr("cy", yk(h[k + R])).attr("r", Math.min(2.8, 24 / h.length + 1)).attr("fill", VC.accent);
    }
    if (cmp === "binom") {
      const b = VZ.binomial(Math.max(2, Math.round(4 * sig * sig)));
      const BR = (b.length - 1) >> 1;
      for (let k = -BR; k <= BR; k++)
        gK.append("circle").attr("cx", xk(k)).attr("cy", yk(b[k + BR])).attr("r", 2.6)
          .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.5);
    }
    FD.caption(gK, 14, 24, "shaded: the tails truncation throws away", VC.bad, 9.5);
    VZ.axisB(gK, xk, 154, 7, "k  (samples)");
    VZ.axisL(gK, yk, 4, null);

    /* right: the responses */
    const gR = VZ.panel(g, halfW + 26, 18, halfW, 176, "the frequency responses").g;
    gR.append("rect").attr("width", halfW).attr("height", 176).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xr = d3.scaleLinear().domain([0, 1]).range([10, halfW - 10]);
    const yr = d3.scaleLinear().domain([-0.1, 1.08]).range([154, 10]);
    VZ.gridY(gR, yr, halfW - 20, 4);
    gR.append("line").attr("x1", 10).attr("x2", halfW - 10).attr("y1", yr(0)).attr("y2", yr(0)).attr("stroke", VC.line);
    const mkp = arr => arr.map((v, i) => [ws[i] / Math.PI, v]);
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(VZ.clamp(d[1], -0.1, 1.08)))(mkp(ideal)))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(VZ.clamp(d[1], -0.1, 1.08)))(mkp(perio)))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.3);
    gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(VZ.clamp(d[1], -0.1, 1.08)))(mkp(Hk)))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    if (cmp === "box") {
      const K = Math.max(3, 2 * Math.round((Math.sqrt(12 * sig * sig + 1) - 1) / 2) + 1);
      const bx = VZ.box(K), pts = [];
      for (let i = 0; i <= NW; i++) pts.push([ws[i] / Math.PI, FD.resp(bx, ws[i]).re]);
      gR.append("path").attr("d", d3.line().x(d => xr(d[0])).y(d => yr(VZ.clamp(d[1], -0.1, 1.08)))(pts))
        .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 1.5).attr("stroke-dasharray", "2 3");
      FD.caption(gR, 14, 168, "red dashed: a box of matched variance, K = " + K, VC.bad, 9);
    }
    VZ.legend(gR, [{ color: VC.accent, label: "sampled + truncated" }, { color: VC.good, label: "periodised Gaussian" },
                   { color: VC.muted, label: "ideal Gaussian", dash: "4 3" }], halfW - 138, 18, { gap: 13 });
    VZ.axisB(gR, xr, 154, 6, "ω / π");
    VZ.axisL(gR, yr, 4, null);

    /* bottom: the two discrepancies, log scale */
    const gD = VZ.panel(g, 0, 226, fr.iw, 118, "how far the sampled kernel's response is from each candidate, on a log scale").g;
    gD.append("rect").attr("width", fr.iw).attr("height", 118).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xd = d3.scaleLinear().domain([0, 1]).range([12, fr.iw - 12]);
    const yd = d3.scaleLog().domain([1e-18, 1]).range([96, 10]);
    VZ.gridY(gD, yd, fr.iw - 24, 5);
    const s1 = ws.map((w, i) => [w / Math.PI, Math.max(Math.abs(Hk[i] - ideal[i]), 1e-18)]);
    const s2 = ws.map((w, i) => [w / Math.PI, Math.max(Math.abs(Hk[i] - perio[i]), 1e-18)]);
    gD.append("path").attr("d", d3.line().x(d => xd(d[0])).y(d => yd(d[1]))(s1))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
    gD.append("path").attr("d", d3.line().x(d => xd(d[0])).y(d => yd(d[1]))(s2))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
    VZ.legend(gD, [{ color: VC.muted, label: "against the ideal Gaussian", dash: "4 3" },
                   { color: VC.good, label: "against the periodised one" }], fr.iw - 220, 20, { gap: 14 });
    VZ.axisB(gD, xd, 96, 6, "ω / π");
    VZ.axisL(gD, yd, 5, null);

    let extra = "";
    if (cmp === "disc") {
      const K = 11, G2 = VZ.zeros2(K, K), D2 = VZ.zeros2(K, K), r = (K - 1) >> 1;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        G2[i][j] = Math.exp(-((i - r) ** 2 + (j - r) ** 2) / (2 * sig * sig));
        D2[i][j] = ((i - r) ** 2 + (j - r) ** 2 <= r * r) ? 1 : 0;
      }
      const svA = VZ.jacobiEig(VZ.mul(VZ.T(G2.map(x => Array.from(x))), G2.map(x => Array.from(x)))).values.map(v => Math.sqrt(Math.max(v, 0))).reverse();
      const svB = VZ.jacobiEig(VZ.mul(VZ.T(D2.map(x => Array.from(x))), D2.map(x => Array.from(x)))).values.map(v => Math.sqrt(Math.max(v, 0))).reverse();
      extra = `<br>separability, as singular values of the 11 × 11 kernel matrices: Gaussian s₁/s₂ = <b>${(svA[0] / Math.max(svA[1], 1e-300)).toExponential(2)}</b> (rank 1, separable); `
            + `disc s₁/s₂ = <b>${VZ.fmt(svB[0] / Math.max(svB[1], 1e-300), 2)}</b> (rank ${svB.filter(v => v > 1e-9 * svB[0]).length}, <b>not</b> separable). `
            + `The disc is rotationally symmetric and not separable; a box is separable and not rotationally symmetric; only the Gaussian is both.`;
    }
    out.innerHTML =
      `<span class="keep">σ</span> = <b>${VZ.fmt(sig, 2)}</b>, truncated at <b>${VZ.fmt(tr, 2)}σ</b> ⇒ <b>${h.length}</b> taps · mass discarded before renormalisation: <b>${(100 * lost).toExponential(3)}%</b><br>`
      + `deepest negative excursion of the response: <b>${minH < 0 ? VZ.fmt(minH, 6) : "none — the response is positive throughout"}</b>${minH < 0 ? " at ω/π = " + VZ.fmt(minAt, 3) : ""}<br>`
      + `max |response − ideal Gaussian| = <b>${dIdeal.toExponential(3)}</b> · max |response − <b>periodised</b> Gaussian| = <b>${dPer.toExponential(3)}</b><br>`
      + (dPer < dIdeal / 10
        ? `The sampled kernel's response follows the <b>periodised</b> Gaussian, not the plain one — sampling in space wraps the spectrum, by §12. At this σ the two candidates differ by ${dIdeal.toExponential(2)}, and the periodised one wins by a factor of ${VZ.fmt(dIdeal / Math.max(dPer, 1e-300), 0)}.`
        : `At this σ the truncation error dominates the periodisation error, so neither candidate fits well: the residual against both is about ${Math.max(dPer, dIdeal).toExponential(2)}, and it is the discarded tails.`)
      + (minH < -1e-6 ? ` <b>And note the negative excursion.</b> An ideal Gaussian's response is positive everywhere; this truncated one is not, so it <i>can</i> ring — at the ${VZ.fmt(100 * Math.abs(minH), 2)}% level. Widen the truncation and it vanishes.` : ``)
      + extra;
  }
  eS.oninput = draw; eT.oninput = draw; eC.onchange = draw;
  draw();
})();

/* ═══ 16 · #unc-svg — the uncertainty product, measured ═══════════════════════
   Widths are second moments of the ENERGY densities, so the bound is ½ and the
   Gaussian sits on it. The frequency range is a control because for the BOX the
   integral genuinely diverges, and widening the range is how you watch it do so. */
(function () {
  const svg = d3.select("#unc-svg"); if (svg.empty()) return;
  const out = document.getElementById("unc-readout");
  const eS = document.getElementById("un-s"), eW = document.getElementById("un-w"),
        eWv = document.getElementById("un-wv"), eR = document.getElementById("un-r");
  const W = 770, H = 360, T = 48;

  const SHAPES = {
    gauss:   { name: "Gaussian", f: (x, a) => Math.exp(-x * x / (2 * a * a)) },
    hann:    { name: "Hann window", f: (x, a) => Math.abs(x) <= a ? 0.5 + 0.5 * Math.cos(Math.PI * x / a) : 0 },
    tent:    { name: "tent", f: (x, a) => Math.max(0, 1 - Math.abs(x) / a) },
    quartic: { name: "quartic (1 − x²)²", f: (x, a) => Math.abs(x) <= a ? (1 - (x / a) ** 2) ** 2 : 0 },
    expo:    { name: "two-sided exponential", f: (x, a) => Math.exp(-Math.abs(x) / a) },
    box:     { name: "box", f: (x, a) => Math.abs(x) <= a ? 1 : 0 },
    gabor:   { name: "Gabor (Gaussian × cosine)", f: (x, a) => Math.exp(-x * x / (2 * a * a)) * Math.cos(3 * x) }
  };

  function measure(shape, a, L) {
    const dx = T / L, S = SHAPES[shape];
    const f = new Array(L);
    for (let n = 0; n < L; n++) { const x = (n < L / 2 ? n : n - L) * dx; f[n] = S.f(x, a); }
    /* sigma_x directly, about the centroid */
    let e = 0, m1 = 0, m2 = 0;
    for (let n = 0; n < L; n++) { const x = (n < L / 2 ? n : n - L) * dx, p = f[n] * f[n]; e += p; m1 += x * p; m2 += x * x * p; }
    const mu = m1 / e, sx = Math.sqrt(Math.max(0, m2 / e - mu * mu));
    /* sigma_w from the DFT magnitude, about zero (the shapes are even, or Gabor: use |F|) */
    const F = FD.fft(f);
    let ew = 0, mw2 = 0;
    const dw = 2 * Math.PI / (L * dx);
    for (let k = 0; k < L; k++) {
      const kk = k < L / 2 ? k : k - L, w = kk * dw, p = FD.cabs(F[k]) ** 2;
      ew += p; mw2 += w * w * p;
    }
    /* Gabor is centred at ±w0, so its second moment about zero includes the carrier:
       report the width about the carrier by folding to |w| − w0 for that shape only */
    let sw;
    if (shape === "gabor") {
      let m = 0, ee = 0;
      for (let k = 0; k < L; k++) {
        const kk = k < L / 2 ? k : k - L, w = kk * dw, p = FD.cabs(F[k]) ** 2;
        m += (Math.abs(w) - 3) ** 2 * p; ee += p;
      }
      sw = Math.sqrt(m / ee);
    } else sw = Math.sqrt(mw2 / ew);
    return { sx: sx, sw: sw, wmax: (L / 2) * dw, f: f, F: F, dx: dx, dw: dw, L: L };
  }

  let sweepCache = { r: null, lines: {} };
  function draw() {
    const shape = eS.value, a = +eW.value, r = +eR.value;
    eWv.textContent = VZ.fmt(a, 2);
    const L = 2048 * r;
    const M = measure(shape, a, L);

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 30 });
    const g = fr.g, halfW = (fr.iw - 26) / 2;

    /* left: the shape in space */
    const gA = VZ.panel(g, 0, 18, halfW, 156, "in space, with ±σ_x shaded").g;
    gA.append("rect").attr("width", halfW).attr("height", 156).attr("fill", VC.panel2).attr("stroke", VC.line);
    const XM = Math.max(3 * a, 3);
    const xa = d3.scaleLinear().domain([-XM, XM]).range([10, halfW - 10]);
    const ya = d3.scaleLinear().domain([-0.25, 1.12]).range([134, 10]);
    gA.append("rect").attr("x", xa(-M.sx)).attr("y", 10).attr("width", xa(M.sx) - xa(-M.sx)).attr("height", 124)
      .attr("fill", VC.accent).attr("fill-opacity", 0.12);
    gA.append("line").attr("x1", 10).attr("x2", halfW - 10).attr("y1", ya(0)).attr("y2", ya(0)).attr("stroke", VC.line);
    const pts = [];
    for (let i = 0; i <= 700; i++) { const x = -XM + 2 * XM * i / 700; pts.push([x, SHAPES[shape].f(x, a)]); }
    gA.append("path").attr("d", d3.line().x(d => xa(d[0])).y(d => ya(d[1]))(pts))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    FD.caption(gA, 14, 24, "σ_x = " + VZ.fmt(M.sx, 4), VC.accent, 10.5);
    VZ.axisB(gA, xa, 134, 6, "x");

    /* right: the transform magnitude */
    const gB = VZ.panel(g, halfW + 26, 18, halfW, 156, "in frequency, with ±σ_ω shaded").g;
    gB.append("rect").attr("width", halfW).attr("height", 156).attr("fill", VC.panel2).attr("stroke", VC.line);
    const WM = Math.min(M.wmax, Math.max(6, 5 * M.sw));
    const xb = d3.scaleLinear().domain([-WM, WM]).range([10, halfW - 10]);
    const mags = [];
    let mmax = 0;
    for (let k = -Math.floor(WM / M.dw); k <= Math.floor(WM / M.dw); k++) {
      const idx = ((k % M.L) + M.L) % M.L, v = FD.cabs(M.F[idx]);
      mags.push([k * M.dw, v]); if (v > mmax) mmax = v;
    }
    const yb = d3.scaleLinear().domain([-0.25 * mmax, 1.12 * mmax]).range([134, 10]);
    gB.append("rect").attr("x", xb(Math.max(-WM, -M.sw))).attr("y", 10)
      .attr("width", xb(Math.min(WM, M.sw)) - xb(Math.max(-WM, -M.sw))).attr("height", 124)
      .attr("fill", VC.violet).attr("fill-opacity", 0.12);
    gB.append("line").attr("x1", 10).attr("x2", halfW - 10).attr("y1", yb(0)).attr("y2", yb(0)).attr("stroke", VC.line);
    gB.append("path").attr("d", d3.line().x(d => xb(d[0])).y(d => yb(d[1]))(mags))
      .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.8);
    FD.caption(gB, 14, 24, "σ_ω = " + VZ.fmt(M.sw, 4), VC.violet, 10.5);
    FD.caption(gB, 14, 38, "measured over |ω| ≤ " + VZ.fmt(M.wmax, 0), VC.muted, 9.5);
    VZ.axisB(gB, xb, 134, 6, "ω");

    /* bottom: the product for every shape, across widths */
    const gC = VZ.panel(g, 0, 206, fr.iw, 118, "the product σ_x·σ_ω for every shape, swept over width").g;
    gC.append("rect").attr("width", fr.iw).attr("height", 118).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xc = d3.scaleLinear().domain([0.3, 4]).range([14, fr.iw - 14]);
    const yc = d3.scaleLog().domain([0.4, 40]).range([96, 10]);
    VZ.gridY(gC, yc, fr.iw - 28, 4);
    gC.append("line").attr("x1", 14).attr("x2", fr.iw - 14).attr("y1", yc(0.5)).attr("y2", yc(0.5))
      .attr("stroke", VC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
    FD.caption(gC, 18, yc(0.5) - 5, "the bound, ½", VC.good, 9.5);
    const cols = { gauss: VC.good, hann: VC.accent, tent: VC.teal, quartic: VC.violet, expo: VC.a2, box: VC.bad, gabor: VC.lime };
    /* the sweep depends only on the frequency range, so compute it once per range */
    if (sweepCache.r !== r) {
      sweepCache = { r: r, lines: {} };
      Object.keys(SHAPES).forEach(k => {
        const line = [];
        for (let i = 0; i <= 6; i++) {
          const aa = 0.3 + (4 - 0.3) * i / 6;
          const mm = measure(k, aa, Math.min(L, 4096));
          line.push([aa, VZ.clamp(mm.sx * mm.sw, 0.4, 40)]);
        }
        sweepCache.lines[k] = line;
      });
    }
    Object.keys(SHAPES).forEach(k => {
      const line = sweepCache.lines[k];
      gC.append("path").attr("d", d3.line().x(d => xc(d[0])).y(d => yc(d[1])).curve(d3.curveMonotoneX)(line))
        .attr("fill", "none").attr("stroke", cols[k]).attr("stroke-width", k === shape ? 2.6 : 1.1)
        .attr("stroke-opacity", k === shape ? 1 : 0.5);
    });
    gC.append("circle").attr("cx", xc(VZ.clamp(a, 0.3, 4))).attr("cy", yc(VZ.clamp(M.sx * M.sw, 0.4, 40)))
      .attr("r", 4.5).attr("fill", VC.a2);
    VZ.axisB(gC, xc, 96, 6, "width parameter");
    VZ.axisL(gC, yc, 4, "σ_x·σ_ω");

    const prod = M.sx * M.sw;
    out.innerHTML =
      `<b>${SHAPES[shape].name}</b>, width <b>${VZ.fmt(a, 2)}</b>, measured over |ω| ≤ <b>${VZ.fmt(M.wmax, 0)}</b> on ${L} samples<br>`
      + `<span class="keep">σ</span>_x = <b>${VZ.fmt(M.sx, 5)}</b> · <span class="keep">σ</span>_<span class="keep">ω</span> = <b>${VZ.fmt(M.sw, 5)}</b> · product <b>${VZ.fmt(prod, 5)}</b> · ratio to the bound ½: <b>${VZ.fmt(prod / 0.5, 4)}</b><br>`
      + (shape === "gauss"
        ? `Exactly on the bound, to five decimals, and it stays there for every width — narrowing in space widens in frequency by precisely the reciprocal factor. The Gaussian is the equality case of the Cauchy–Schwarz step, which is where its uniqueness comes from.`
        : shape === "box"
          ? `The number above is <b>meaningless</b>, and the control marked "frequency range" is how you can tell: widen it and σ_<span class="keep">ω</span> grows without limit, because |F|² decays as 1/ω² and ∫ω²|F|² diverges. A box has no finite bandwidth in this sense at all — a much stronger statement than "a box is a poor filter".`
          : `Above the bound by <b>${VZ.fmt(100 * (prod / 0.5 - 1), 2)}%</b>. Every smooth, compactly supported window lands a few per cent above ½ — close enough that the choice between them is about side-lobe shape (§21) rather than about this product.`);
  }
  eS.onchange = draw; eW.oninput = draw; eR.onchange = draw;
  draw();
})();

/* ═══ 17 · #win-svg — windowing and spectral leakage ══════════════════════════ */
(function () {
  const svg = d3.select("#win-svg"); if (svg.empty()) return;
  const out = document.getElementById("win-readout");
  const eW = document.getElementById("wn-w"), eF = document.getElementById("wn-f"),
        eFv = document.getElementById("wn-fv"), eA = document.getElementById("wn-a"),
        eAv = document.getElementById("wn-av"), eV = document.getElementById("wn-v");
  const W = 770, H = 400, N = 64, Z = 16;

  const WINS = {
    rect:     { name: "rectangular (none)", w: n => 1 },
    hann:     { name: "Hann", w: n => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / N) },
    hamming:  { name: "Hamming", w: n => 0.54 - 0.46 * Math.cos(2 * Math.PI * n / N) },
    blackman: { name: "Blackman", w: n => 0.42 - 0.5 * Math.cos(2 * Math.PI * n / N) + 0.08 * Math.cos(4 * Math.PI * n / N) }
  };

  function padSpec(x, L) {
    const z = new Array(L).fill(0).map((_, i) => (i < x.length ? x[i] : 0));
    const F = FD.fft(z);
    const o = [];
    for (let k = 0; k <= L / 2; k++) o.push(FD.cabs(F[k]));
    return o;
  }

  function draw() {
    const wk = eW.value, Win = WINS[wk], f0 = +eF.value, adb = +eA.value;
    eFv.textContent = VZ.fmt(f0, 2); eAv.textContent = adb + " dB";
    const f1 = 20, amp = Math.pow(10, -adb / 20);       // FIXED bin: an offset of exactly
    /* 8 bins would land on a null of the rectangular window's own transform, where the
       leakage is zero and the demonstration silently fails. */
    const win = new Float64Array(N);
    for (let n = 0; n < N; n++) win[n] = Win.w(n);

    const fr = VZ.frame(svg, W, H, { l: 48, r: 16, t: 24, b: 30 });
    const g = fr.g;

    if (eV.value === "img") {
      /* A SMOOTH DIAGONAL RAMP is the right test image here: it has no axis-aligned
         structure at all, so every scrap of energy on the two frequency axes must be
         the seam of the implicit tiling. The house is shown alongside as the honest
         caveat — a real picture's cross is part seam and part genuine content. */
      const NI = 64;
      const ramp = VZ.zeros2(NI, NI), rampW = VZ.zeros2(NI, NI), hs = FD.house(), hsW = VZ.zeros2(NI, NI);
      const w2 = i => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NI);
      for (let i = 0; i < NI; i++) for (let j = 0; j < NI; j++) {
        ramp[i][j] = 0.2 + 0.6 * (i + j) / (2 * NI);
        rampW[i][j] = ramp[i][j] * w2(i) * w2(j);
        hsW[i][j] = hs[i][j] * w2(i) * w2(j);
      }
      const S = 158;
      const sp = M => FD.fftshift2(FD.fft2(M).map(r => Float64Array.from(r, v => Math.log(1 + FD.cabs(v)))));
      VZ.panel(g, 0, 24, S, S, "a smooth diagonal ramp").g.call(s2 => FD.raster(s2, ramp, 0, 0, S, S, {}));
      VZ.panel(g, S + 14, 24, S, S, "its spectrum: a bright cross").g.call(s2 => FD.raster(s2, sp(ramp), 0, 0, S, S, {}));
      VZ.panel(g, 2 * (S + 14), 24, S, S, "the same ramp × a 2-D Hann window").g.call(s2 => FD.raster(s2, rampW, 0, 0, S, S, {}));
      VZ.panel(g, 3 * (S + 14), 24, S, S, "its spectrum: the cross is gone").g.call(s2 => FD.raster(s2, rampW ? sp(rampW) : sp(ramp), 0, 0, S, S, {}));
      /* the measurement: mean |F| along the two axis lines, beyond |k| = 8, where the
         seam's 1/k tail lives, against the mean magnitude off the axes at the same radii */
      const tail = M => {
        const F = FD.fftshift2(FD.fft2(M).map(r => Float64Array.from(r, v => FD.cabs(v))));
        const c = NI / 2;
        let ax = 0, na = 0, off = 0, no = 0;
        for (let k = 0; k < NI; k++) {
          const kk = k - c;
          if (Math.abs(kk) >= 8) { ax += F[c][k] + F[k][c]; na += 2; }
        }
        for (let a = 0; a < NI; a++) for (let b = 0; b < NI; b++) {
          if (Math.abs(a - c) >= 8 && Math.abs(b - c) >= 8) { off += F[a][b]; no++; }
        }
        return [ax / na, off / no];
      };
      const [ra, ro] = tail(ramp), [rwa] = tail(rampW);
      const [ha, ho] = tail(hs), [hwa, hwo] = tail(hsW);
      const gT = VZ.panel(g, 0, 24 + S + 26, fr.iw, 96, "").g;
      FD.caption(gT, 0, 6, "mean |F| along the two frequency axes beyond |k| = 8 — where the seam's 1/k tail lives", VC.ink, 11.5);
      FD.caption(gT, 8, 26, "the ramp, unwindowed:  " + VZ.fmt(ra, 4) + "        windowed:  " + VZ.fmt(rwa, 4)
        + "        a reduction of " + VZ.fmt(ra / Math.max(rwa, 1e-12), 0) + "×", VC.good, 11);
      FD.caption(gT, 8, 44, "the ramp has NO off-axis content at all (" + ro.toExponential(1) + "), so the whole cross was the seam.", VC.muted, 11);
      FD.caption(gT, 8, 68, "the house, unwindowed:  " + VZ.fmt(ha, 4) + "        windowed:  " + VZ.fmt(hwa, 4)
        + "        against off-axis " + VZ.fmt(ho, 4) + " → " + VZ.fmt(hwo, 4), VC.a2, 11);
      FD.caption(gT, 8, 86, "a real picture's cross is only PART seam: the house's own horizontal and vertical edges are genuinely on the axes,", VC.muted, 11);
      out.innerHTML =
        `mean |F| along the two frequency axes beyond |k| = 8 · <b>ramp</b>: <b>${VZ.fmt(ra, 4)}</b> unwindowed → <b>${VZ.fmt(rwa, 4)}</b> windowed, `
        + `a reduction of <b>${VZ.fmt(ra / Math.max(rwa, 1e-12), 0)}×</b>, and the ramp has no off-axis content whatsoever (<b>${ro.toExponential(1)}</b>) — so its entire cross was the <b>seam</b><br>`
        + `<b>house</b>: <b>${VZ.fmt(ha, 4)}</b> → <b>${VZ.fmt(hwa, 4)}</b>, against an off-axis level of <b>${VZ.fmt(ho, 4)}</b> → <b>${VZ.fmt(hwo, 4)}</b>. `
        + `The axis-to-off-axis ratio falls from <b>${VZ.fmt(ha / ho, 2)}</b> to <b>${VZ.fmt(hwa / hwo, 2)}</b> and does <i>not</i> reach 1, because a house genuinely has horizontal and vertical edges. `
        + `<b>Both halves of that are worth knowing:</b> the cross is an artefact when the picture has no axis-aligned structure, and it is part artefact and part content when it does — and you cannot tell which by looking.`;
      return;
    }

    /* the record */
    const rec = new Float64Array(N);
    for (let n = 0; n < N; n++)
      rec[n] = (Math.cos(2 * Math.PI * f0 * n / N) + amp * Math.cos(2 * Math.PI * f1 * n / N + 0.4)) * win[n];

    const gA = VZ.panel(g, 0, 18, fr.iw, 92, "the record, with the window drawn over it").g;
    gA.append("rect").attr("width", fr.iw).attr("height", 92).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xa = d3.scaleLinear().domain([0, N - 1]).range([8, fr.iw - 8]);
    const ya = d3.scaleLinear().domain([-1.35, 1.35]).range([84, 8]);
    gA.append("line").attr("x1", 8).attr("x2", fr.iw - 8).attr("y1", ya(0)).attr("y2", ya(0)).attr("stroke", VC.grid);
    gA.append("path").attr("d", d3.line().x((d, i) => xa(i)).y(d => ya(d)).curve(d3.curveMonotoneX)(Array.from(rec)))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.6);
    gA.append("path").attr("d", d3.line().x((d, i) => xa(i)).y(d => ya(d))(Array.from(win)))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    gA.append("path").attr("d", d3.line().x((d, i) => xa(i)).y(d => ya(-d))(Array.from(win)))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    VZ.axisL(gA, ya, 3, null);

    /* the spectrum */
    const L = N * Z, sp = padSpec(rec, L);
    const pk = Math.max(...sp);
    const db = eV.value === "db";
    const gB = VZ.panel(g, 0, 140, fr.iw, 132, "its spectrum" + (db ? ", in decibels" : ", linear")).g;
    gB.append("rect").attr("width", fr.iw).attr("height", 132).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xb = d3.scaleLinear().domain([0, N / 2]).range([10, fr.iw - 10]);
    const yb = db ? d3.scaleLinear().domain([-95, 5]).range([112, 8]) : d3.scaleLinear().domain([0, 1.06]).range([112, 8]);
    VZ.gridY(gB, yb, fr.iw - 20, 5);
    const curve = sp.map((v, k) => [k / Z, db ? Math.max(20 * Math.log10(Math.max(v, 1e-12) / pk), -95) : v / pk]);
    gB.append("path").attr("d", d3.line().x(d => xb(d[0])).y(d => yb(d[1]))(curve))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.5);
    [[f0, VC.good, "strong"], [f1, VC.violet, "weak, −" + adb + " dB"]].forEach(([f, c, lab]) => {
      gB.append("line").attr("x1", xb(f)).attr("x2", xb(f)).attr("y1", 8).attr("y2", 112)
        .attr("stroke", c).attr("stroke-width", 1.3).attr("stroke-dasharray", "3 3");
      FD.caption(gB, xb(f) + 4, 20, lab, c, 9.5);
    });
    /* the leakage floor at the weak component's position, measured from the STRONG
       tone ALONE — that is the honest test of whether the weak one is visible */
    const idx1 = Math.round(f1 * Z);
    const solo = new Float64Array(N);
    for (let n = 0; n < N; n++) solo[n] = Math.cos(2 * Math.PI * f0 * n / N) * win[n];
    const spSolo = padSpec(solo, L);
    const floorDb = 20 * Math.log10(Math.max(spSolo[idx1], 1e-12) / Math.max(...spSolo));
    const buried = floorDb > -adb;
    VZ.axisL(gB, yb, 5, null);
    VZ.axisB(gB, xb, 112, 8, "bin  (N = 64, zero-padded ×16 for display)");

    /* the window's own transform */
    const gC = VZ.panel(g, 0, 300, fr.iw, 84, "the window's own transform — the shape every spike is smeared into").g;
    gC.append("rect").attr("width", fr.iw).attr("height", 84).attr("fill", VC.panel2).attr("stroke", VC.line);
    const wsp = padSpec(win, L), wpk = Math.max(...wsp);
    const xc = d3.scaleLinear().domain([-6, 6]).range([10, fr.iw - 10]);
    const yc = d3.scaleLinear().domain([-95, 5]).range([66, 6]);
    VZ.gridY(gC, yc, fr.iw - 20, 4);
    const wc = [];
    for (let k = 0; k <= 6 * Z; k++) {
      const v = 20 * Math.log10(Math.max(wsp[k], 1e-12) / wpk);
      wc.push([-k / Z, Math.max(v, -95)]); wc.push([k / Z, Math.max(v, -95)]);
    }
    wc.sort((p, q) => p[0] - q[0]);
    gC.append("path").attr("d", d3.line().x(d => xc(d[0])).y(d => yc(d[1]))(wc))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.6);
    /* main lobe and peak sidelobe, measured */
    let i = 1; while (i + 1 < wsp.length && wsp[i + 1] < wsp[i]) i++;
    const mainHalf = i / Z;
    let side = 0; for (let k = i; k < wsp.length; k++) side = Math.max(side, wsp[k]);
    const sideDb = 20 * Math.log10(side / wpk);
    gC.append("line").attr("x1", 10).attr("x2", fr.iw - 10).attr("y1", yc(sideDb)).attr("y2", yc(sideDb))
      .attr("stroke", VC.bad).attr("stroke-dasharray", "4 3");
    FD.caption(gC, 14, yc(sideDb) - 4, "peak side lobe " + VZ.fmt(sideDb, 2) + " dB", VC.bad, 9.5);
    VZ.axisB(gC, xc, 66, 7, "bins from the peak");
    VZ.axisL(gC, yc, 3, null);

    /* energy within ±2 bins, on the unpadded transform */
    const F0 = FD.fft(Array.from(rec));
    let tot = 0, near = 0;
    const kk = Math.round(f0);
    for (let k = 0; k <= N / 2; k++) { const e = FD.cabs(F0[k]) ** 2; tot += e; if (Math.abs(k - kk) <= 2) near += e; }
    const onBin = Math.abs(f0 - Math.round(f0)) < 1e-9;
    out.innerHTML =
      `<b>${Win.name}</b> · strong tone at bin <b>${VZ.fmt(f0, 2)}</b>, weak one at bin <b>${VZ.fmt(f1, 2)}</b> at <b>−${adb} dB</b><br>`
      + `main-lobe half-width <b>${VZ.fmt(mainHalf, 3)}</b> bins · peak side lobe <b>${VZ.fmt(sideDb, 2)} dB</b> · coherent gain <b>${VZ.fmt(FD.sum(Array.from(win)) / N, 4)}</b> · `
      + `energy within ±2 bins of the peak: <b>${VZ.fmt(100 * near / tot, 3)}%</b><br>`
      + (onBin
        ? `The frequency lands <b>exactly on a bin</b>, and this is the special case: the rectangular window's sinc has zeros at every other bin, so it leaks <i>nothing</i>. It is why a naive FFT test never shows the problem. Nudge the slider off the integer and watch.`
        : buried
          ? `Off-bin, and the weak component at <b>−${adb} dB</b> is <b>buried</b>: the strong tone's leakage alone reaches <b>${VZ.fmt(floorDb, 1)} dB</b> at that bin, which is above it. `
            + `Switch to a smoother window and watch it emerge — that is exactly what side-lobe suppression buys, and the price is the wider main lobe visible below.`
          : `Off-bin, and the weak component at <b>−${adb} dB</b> <b>clears</b> the leakage floor, which the strong tone alone puts at <b>${VZ.fmt(floorDb, 1)} dB</b> at that bin. `
            + `${VZ.fmt(100 - 100 * near / tot, 3)}% of the energy still lands outside ±2 bins of the peak — leakage that is present whether or not it happens to hide anything.`);
  }
  eW.onchange = draw; eF.oninput = draw; eA.oninput = draw; eV.onchange = draw;
  draw();
})();

/* ═══ 18 · #stft-svg — the short-time transform, and its fixed grid ═══════════ */
(function () {
  const svg = d3.select("#stft-svg"); if (svg.empty()) return;
  const out = document.getElementById("stft-readout");
  const eS = document.getElementById("st-s"), eW = document.getElementById("st-w"),
        eWv = document.getElementById("st-wv"), eT = document.getElementById("st-t");
  const W = 770, H = 380, L = 512, HOP = 4, NF = 128;

  function signal(kind) {
    const f = VZ.zeros(L);
    if (kind === "chirp") {
      let ph = 0;
      for (let n = 0; n < L; n++) { const fr = 0.02 + 0.40 * n / L; ph += 2 * Math.PI * fr; f[n] = Math.cos(ph); }
      f[120] += 6; f[380] += 6;                       // two clicks
    } else if (kind === "two") {
      for (let n = 0; n < L; n++) {
        f[n] = n < L / 2 ? Math.cos(2 * Math.PI * 0.12 * n) + Math.cos(2 * Math.PI * 0.30 * n)
                         : Math.cos(2 * Math.PI * 0.20 * n) + Math.cos(2 * Math.PI * 0.235 * n);
      }
    } else {
      for (let n = 0; n < L; n++) f[n] = Math.cos(2 * Math.PI * (n < L / 2 ? 0.10 : 0.35) * n);
    }
    return f;
  }
  function window(kind, M) {
    const w = VZ.zeros(M);
    for (let n = 0; n < M; n++) {
      if (kind === "rect") w[n] = 1;
      else if (kind === "hann") w[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / M);
      else { const c = (M - 1) / 2, s = M / 6; w[n] = Math.exp(-((n - c) ** 2) / (2 * s * s)); }
    }
    return w;
  }

  function draw() {
    const kind = eS.value, M = Math.round(+eW.value / 2) * 2, wk = eT.value;
    eWv.textContent = String(M);
    const f = signal(kind), w = window(wk, M);
    const frames = [];
    for (let st = 0; st + M <= L; st += HOP) {
      const buf = new Array(NF).fill(0);
      for (let n = 0; n < M; n++) buf[n] = f[st + n] * w[n];
      const F = FD.fft(buf);
      const col = new Float64Array(NF / 2 + 1);
      for (let k = 0; k <= NF / 2; k++) col[k] = FD.cabs(F[k]);
      frames.push(col);
    }
    /* spectrogram as rows = frequency (top = Nyquist), cols = time */
    const NB = NF / 2 + 1, img = VZ.zeros2(NB, frames.length);
    let mx = 0;
    for (let t = 0; t < frames.length; t++) for (let k = 0; k < NB; k++) { const v = frames[t][k]; if (v > mx) mx = v; }
    for (let t = 0; t < frames.length; t++) for (let k = 0; k < NB; k++)
      img[NB - 1 - k][t] = Math.log(1 + 40 * frames[t][k] / Math.max(mx, 1e-12));

    const fr = VZ.frame(svg, W, H, { l: 52, r: 16, t: 24, b: 32 });
    const g = fr.g;

    const gA = VZ.panel(g, 0, 16, fr.iw, 76, "the signal, with the analysis window drawn at one position").g;
    gA.append("rect").attr("width", fr.iw).attr("height", 76).attr("fill", VC.panel2).attr("stroke", VC.line);
    const xa = d3.scaleLinear().domain([0, L - 1]).range([6, fr.iw - 6]);
    const [flo, fhi] = FD.minmax(f);
    const ya = d3.scaleLinear().domain([flo * 1.1, fhi * 1.1]).range([70, 6]);
    const pts = [];
    for (let n = 0; n < L; n++) pts.push([n, f[n]]);
    gA.append("path").attr("d", d3.line().x(d => xa(d[0])).y(d => ya(d[1]))(pts))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 0.9);
    const wpts = [];
    for (let n = 0; n < M; n++) wpts.push([L / 2 - M / 2 + n, w[n] * fhi * 0.9]);
    gA.append("path").attr("d", d3.line().x(d => xa(d[0])).y(d => ya(d[1]))(wpts))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
    gA.append("rect").attr("x", xa(L / 2 - M / 2)).attr("y", 6).attr("width", xa(L / 2 + M / 2) - xa(L / 2 - M / 2)).attr("height", 64)
      .attr("fill", VC.a2).attr("fill-opacity", 0.08);

    const SY = 116, SH = 200;
    const gS = VZ.panel(g, 0, SY, fr.iw, SH, "the spectrogram: |windowed transform| against time and frequency").g;
    FD.raster(gS, img, 0, 0, fr.iw, SH, {});
    const xs = d3.scaleLinear().domain([0, L - 1]).range([0, fr.iw]);
    const ys = d3.scaleLinear().domain([0, 0.5]).range([SH, 0]);
    gS.append("g").attr("class", "axis").attr("transform", `translate(0,${SH})`).call(d3.axisBottom(xs).ticks(8));
    gS.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    gS.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11).attr("fill", VC.muted).text("cycles / sample");
    gS.append("text").attr("x", fr.iw).attr("y", SH + 30).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", VC.muted).text("n  (samples)");

    /* the resolutions this window buys */
    let e = 0, m1 = 0, m2 = 0;
    for (let n = 0; n < M; n++) { const p = w[n] * w[n]; e += p; m1 += n * p; m2 += n * n * p; }
    const mu = m1 / e, st = Math.sqrt(Math.max(0, m2 / e - mu * mu));
    const WF = FD.fft(Array.from(w).concat(new Array(4096 - M).fill(0)));
    let ew = 0, mw = 0;
    for (let k = 0; k < 4096; k++) {
      const kk = k < 2048 ? k : k - 4096, ww = 2 * Math.PI * kk / 4096, p = FD.cabs(WF[k]) ** 2;
      ew += p; mw += ww * ww * p;
    }
    const sw = Math.sqrt(mw / ew);
    out.innerHTML =
      `window <b>${M}</b> samples, ${wk === "rect" ? "rectangular" : wk === "hann" ? "Hann" : "Gaussian"} · `
      + `time resolution <span class="keep">σ</span>_t = <b>${VZ.fmt(st, 3)}</b> samples · frequency resolution <span class="keep">σ</span>_<span class="keep">ω</span> = <b>${VZ.fmt(sw, 5)}</b> rad/sample `
      + `(<b>${VZ.fmt(sw / (2 * Math.PI), 5)}</b> cycles/sample) · product <b>${VZ.fmt(st * sw, 4)}</b> against the bound ½<br>`
      + (kind === "chirp"
        ? (M <= 12
          ? `A short window pins the two <b>clicks</b> to sharp vertical lines but smears the chirp into a thick diagonal band — good in time, poor in frequency.`
          : M >= 40
            ? `A long window resolves the chirp into a thin diagonal line but smears each <b>click</b> across ${M} samples — good in frequency, poor in time. The same picture, the same signal, the opposite failure.`
            : `In between. Neither the clicks nor the chirp is well resolved, which is the honest situation: one window width cannot serve both, and no amount of tuning changes that.`)
        : kind === "two"
          ? (M >= 40
            ? `The two <b>close</b> tones in the second half (0.200 and 0.235 cycles/sample, 0.035 apart) are now resolved as two lines — a window of ${M} samples has enough frequency resolution to separate them. The cost is that the moment of change in the middle is smeared over ${M} samples.`
            : `The two close tones in the second half are <b>merged into one blob</b>: at ${M} samples the frequency resolution is about ${VZ.fmt(sw / (2 * Math.PI), 4)} cycles/sample, wider than their 0.035 separation. Widen the window until they split.`)
          : (M <= 12
            ? `The instant of change is sharp and each tone is a thick band.`
            : `Each tone is a thin line and the instant of change is blurred across ${M} samples. The transition is an event in time, and a frequency-resolving window is the wrong instrument for it.`))
      + ` <b>The grid is the point.</b> Every cell of this picture has the same width and the same height — the analysis has one resolution everywhere, chosen once. §26 shows what a wavelet does instead.`;
  }
  eS.onchange = draw; eW.oninput = draw; eT.onchange = draw;
  draw();
})();

/* ═══ 19 · #dct-svg — DCT against DFT on one block ════════════════════════════ */
(function () {
  const svg = d3.select("#dct-svg"); if (svg.empty()) return;
  const out = document.getElementById("dct-readout");
  const eB = document.getElementById("dc-b"), eK = document.getElementById("dc-k"),
        eKv = document.getElementById("dc-kv"), eS = document.getElementById("dc-s");
  const W = 770, H = 400, N = 8;

  /* orthonormal DCT-II, written out. Verified against a reference to 2.7e-15. */
  function dctM(n) {
    const M = [];
    for (let k = 0; k < n; k++) {
      const a = k === 0 ? Math.sqrt(1 / n) : Math.sqrt(2 / n), row = [];
      for (let i = 0; i < n; i++) row.push(a * Math.cos(Math.PI * (i + 0.5) * k / n));
      M.push(row);
    }
    return M;
  }
  const M8 = dctM(N);
  const dct = f => M8.map(r => r.reduce((s, v, i) => s + v * f[i], 0));
  const idct = X => { const o = VZ.zeros(N); for (let i = 0; i < N; i++) { let s = 0; for (let k = 0; k < N; k++) s += M8[k][i] * X[k]; o[i] = s; } return o; };

  function block(kind) {
    if (kind === "jpeg") return Float64Array.from([52, 55, 61, 66, 70, 61, 64, 73]);
    if (kind === "ramp") return Float64Array.from([10, 20, 30, 40, 50, 60, 70, 80]);
    if (kind === "step") return Float64Array.from([20, 20, 20, 20, 70, 70, 70, 70]);
    if (kind === "smooth") return Float64Array.from([40, 46, 55, 63, 66, 62, 52, 43]);
    const r = VZ.rng(9), o = VZ.zeros(N);
    for (let i = 0; i < N; i++) o[i] = 45 + 18 * VZ.randn(r);
    return o;
  }

  function truncDCT(f, K) { const X = dct(f); for (let k = K; k < N; k++) X[k] = 0; return idct(X); }
  function truncDFT(f, K) {
    const F = FD.fft(Array.from(f)), G = [];
    for (let k = 0; k < N; k++) G.push(FD.C(0, 0));
    G[0] = F[0]; let kept = 1, m = 1;
    while (kept < K && m <= N / 2) {
      G[m] = F[m]; kept++;
      if (kept < K && m !== N - m) { G[N - m] = F[N - m]; kept++; }
      m++;
    }
    return FD.ifft(G).map(v => v.re);
  }

  function draw() {
    const f = block(eB.value), K = VZ.clamp(Math.round(+eK.value), 1, N);
    eKv.textContent = String(K);
    const rd = truncDCT(f, K), rf = truncDFT(f, K);
    const ed = FD.rmse(Array.from(f), Array.from(rd)), ef = FD.rmse(Array.from(f), rf);
    const Xd = dct(f), Xf = FD.fft(Array.from(f)).map(v => FD.cabs(v) / Math.sqrt(N));

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 30 });
    const g = fr.g;

    if (eS.value === "ext") {
      const gE = VZ.panel(g, 0, 20, fr.iw, 340, "what each transform assumes lies outside the block").g;
      gE.append("rect").attr("width", fr.iw).attr("height", 340).attr("fill", VC.panel2).attr("stroke", VC.line);
      const per = [], mir = [];
      for (let t = -2 * N; t < 3 * N; t++) {
        per.push([t, f[((t % N) + N) % N]]);
        const p = 2 * N, kk = ((t % p) + p) % p;
        mir.push([t, kk < N ? f[kk] : f[p - 1 - kk]]);
      }
      const xe = d3.scaleLinear().domain([-2 * N, 3 * N - 1]).range([12, fr.iw - 12]);
      const [lo, hi] = FD.minmax(f);
      const mk = (arr, y0, hgt, col, title, note) => {
        const yy = d3.scaleLinear().domain([lo - 12, hi + 12]).range([y0 + hgt - 22, y0 + 8]);
        FD.caption(gE, 16, y0 + 2, title, VC.ink, 11.5);
        FD.caption(gE, 16, y0 + hgt - 6, note, VC.muted, 10);
        for (let b = -2; b < 3; b++) {
          if (b === 0) gE.append("rect").attr("x", xe(-0.5)).attr("y", y0 + 6).attr("width", xe(N - 0.5) - xe(-0.5))
            .attr("height", hgt - 28).attr("fill", VC.accent).attr("fill-opacity", 0.08);
          gE.append("line").attr("x1", xe(b * N - 0.5)).attr("x2", xe(b * N - 0.5)).attr("y1", y0 + 6).attr("y2", y0 + hgt - 22)
            .attr("stroke", VC.line).attr("stroke-dasharray", "2 3");
        }
        gE.append("path").attr("d", d3.line().x(d => xe(d[0])).y(d => yy(d[1]))(arr))
          .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.8);
        arr.forEach(d => gE.append("circle").attr("cx", xe(d[0])).attr("cy", yy(d[1])).attr("r", 2.2).attr("fill", col));
        /* mark the jumps */
        for (let i = 1; i < arr.length; i++) {
          if (Math.abs(arr[i][1] - arr[i - 1][1]) > 0.5 * (hi - lo) && Math.abs(arr[i][0] % N) === 0) {
            gE.append("line").attr("x1", xe(arr[i][0] - 0.5)).attr("x2", xe(arr[i][0] - 0.5))
              .attr("y1", yy(arr[i - 1][1])).attr("y2", yy(arr[i][1])).attr("stroke", VC.bad).attr("stroke-width", 2.4);
          }
        }
      };
      mk(per, 24, 150, VC.bad, "the DFT's periodic extension  … a b c d | a b c d …", "red bars mark the manufactured jumps at every block boundary");
      mk(mir, 190, 150, VC.good, "the DCT's mirror extension  … c b a | a b c d | d c b a …", "no jumps: a meets a, d meets d — only a kink, which needs far fewer coefficients");
      let jump = 0;
      for (let i = 1; i < N; i++) jump = Math.max(jump, Math.abs(f[i] - f[i - 1]));
      out.innerHTML =
        `the block's internal steps are at most <b>${VZ.fmt(jump, 2)}</b>, while the DFT's periodic extension manufactures a jump of <b>${VZ.fmt(Math.abs(f[0] - f[N - 1]), 2)}</b> at every boundary<br>`
        + `A jump needs every frequency (§05), so the DFT spends coefficients describing a discontinuity that is not in the data. The mirror extension has none, and that is the entire reason image compression uses cosines.`;
      return;
    }
    if (eS.value === "basis") {
      const gB = VZ.panel(g, 0, 20, fr.iw, 340, "the eight DCT basis functions, and the eight DFT ones (real parts)").g;
      gB.append("rect").attr("width", fr.iw).attr("height", 340).attr("fill", VC.panel2).attr("stroke", VC.line);
      const colw = (fr.iw - 24) / 8;
      for (let k = 0; k < N; k++) {
        const x0 = 12 + k * colw;
        [[0, "DCT", VC.accent, (i) => M8[k][i]], [1, "DFT", VC.violet, (i) => Math.cos(2 * Math.PI * k * i / N) / Math.sqrt(N)]]
          .forEach(([row, lab, col, fn]) => {
            const y0 = 34 + row * 158, hh = 130;
            const yy = d3.scaleLinear().domain([-0.55, 0.55]).range([y0 + hh, y0]);
            const xx = d3.scaleLinear().domain([-0.5, N - 0.5]).range([x0 + 4, x0 + colw - 8]);
            gB.append("line").attr("x1", x0 + 2).attr("x2", x0 + colw - 6).attr("y1", yy(0)).attr("y2", yy(0)).attr("stroke", VC.grid);
            const pts = [];
            for (let u = 0; u <= (N - 1) * 8; u++) {
              const i = u / 8;
              pts.push([i, row === 0
                ? (k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N)) * Math.cos(Math.PI * (i + 0.5) * k / N)
                : Math.cos(2 * Math.PI * k * i / N) / Math.sqrt(N)]);
            }
            gB.append("path").attr("d", d3.line().x(d => xx(d[0])).y(d => yy(d[1]))(pts))
              .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.5);
            for (let i = 0; i < N; i++)
              gB.append("circle").attr("cx", xx(i)).attr("cy", yy(fn(i))).attr("r", 2).attr("fill", col);
            if (k === 0) FD.caption(gB, 12, y0 - 8, lab + " basis", col, 11);
          });
        FD.caption(gB, x0 + colw / 2 - 8, 24, "k = " + k, VC.muted, 9.5);
      }
      out.innerHTML =
        `The DCT's basis functions are cosines with a <b>half-sample offset</b>, so they are symmetric about the block's <i>edges</i>; the DFT's are symmetric about its <i>samples</i> and must close up periodically. `
        + `That offset is the whole difference, and it is why the DCT's first few functions can represent a ramp and the DFT's cannot.`;
      return;
    }

    /* default: both transforms */
    const gA = VZ.panel(g, 0, 20, fr.iw, 148, "the block, and both reconstructions from " + K + " basis functions").g;
    gA.append("rect").attr("width", fr.iw).attr("height", 148).attr("fill", VC.panel2).attr("stroke", VC.line);
    const all = Array.from(f).concat(Array.from(rd), rf);
    const [lo, hi] = FD.minmax(all);
    const xa = d3.scaleLinear().domain([-0.5, N - 0.5]).range([16, fr.iw - 16]);
    const ya = d3.scaleLinear().domain([lo - 6, hi + 6]).range([126, 10]);
    VZ.gridY(gA, ya, fr.iw - 26, 4);
    for (let i = 0; i < N; i++) {
      gA.append("line").attr("x1", xa(i)).attr("x2", xa(i)).attr("y1", ya(lo - 6)).attr("y2", ya(f[i]))
        .attr("stroke", VC.muted).attr("stroke-width", 2);
      gA.append("circle").attr("cx", xa(i)).attr("cy", ya(f[i])).attr("r", 3.4).attr("fill", VC.muted);
    }
    const dense = (fn, col, wdt, dash) => {
      const pts = [];
      for (let u = 0; u <= (N - 1) * 10; u++) pts.push([u / 10, fn(u / 10)]);
      const el = gA.append("path").attr("d", d3.line().x(d => xa(d[0])).y(d => ya(d[1]))(pts))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", wdt);
      if (dash) el.attr("stroke-dasharray", dash);
    };
    dense(x => { let s = 0; for (let k = 0; k < K; k++) s += Xd[k] * (k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N)) * Math.cos(Math.PI * (x + 0.5) * k / N); return s; }, VC.accent, 2.2);
    {
      const F = FD.fft(Array.from(f));
      let kept = 1, m = 1; const use = [0];
      while (kept < K && m <= N / 2) { use.push(m); kept++; if (kept < K && m !== N - m) { use.push(N - m); kept++; } m++; }
      dense(x => { let s = 0; use.forEach(k => { const th = 2 * Math.PI * k * x / N; s += (F[k].re * Math.cos(th) - F[k].im * Math.sin(th)) / N; }); return s; }, VC.violet, 1.6, "4 3");
    }
    VZ.legend(gA, [{ color: VC.muted, label: "the block" }, { color: VC.accent, label: "DCT, " + K + " terms" },
                   { color: VC.violet, label: "DFT, " + K + " terms", dash: "4 3" }], 20, 18, { gap: 14 });
    VZ.axisL(gA, ya, 4, null);

    const gC = VZ.panel(g, 0, 200, fr.iw, 156, "the coefficient magnitudes: DCT above, DFT below").g;
    gC.append("rect").attr("width", fr.iw).attr("height", 156).attr("fill", VC.panel2).attr("stroke", VC.line);
    const cm = Math.max(...Xd.map(Math.abs), ...Xf, 1e-9);
    const xc = d3.scaleLinear().domain([-0.5, N - 0.5]).range([26, fr.iw - 16]);
    const bw = (fr.iw - 50) / N - 8;
    [[Xd.map(Math.abs), 12, 62, VC.accent, "DCT |F(k)|"], [Xf, 88, 62, VC.violet, "DFT |F(k)|/√N"]].forEach(([arr, y0, hh, col, lab]) => {
      const yy = d3.scaleLinear().domain([0, cm * 1.1]).range([y0 + hh, y0 + 4]);
      FD.caption(gC, 26, y0 + 2, lab, col, 10.5);
      arr.forEach((v, k) => {
        gC.append("rect").attr("x", xc(k) - bw / 2).attr("y", yy(v)).attr("width", bw).attr("height", Math.max(0.8, yy(0) - yy(v)))
          .attr("fill", k < K ? col : VC.line).attr("fill-opacity", k < K ? 0.9 : 0.7);
        gC.append("text").attr("x", xc(k)).attr("y", y0 + hh + 11).attr("text-anchor", "middle")
          .attr("font-size", 8.6).attr("fill", VC.muted).text(VZ.fmt(v, 1));
      });
      gC.append("g").attr("class", "axis").attr("transform", `translate(0,${y0 + hh})`).call(d3.axisBottom(xc).ticks(N));
    });

    const ed2 = FD.energy(Array.from(Xd).slice(0, K)) / FD.energy(Array.from(Xd));
    out.innerHTML =
      `keeping <b>${K}</b> of ${N} basis functions · DCT RMSE <b>${VZ.fmt(ed, 4)}</b> · DFT RMSE <b>${VZ.fmt(ef, 4)}</b> · `
      + `ratio DFT/DCT <b>${ed > 1e-12 ? VZ.fmt(ef / ed, 2) : "—"}</b><br>`
      + `energy in the retained DCT coefficients: <b>${VZ.fmt(100 * ed2, 4)}%</b> · total energy ∑f² = <b>${VZ.fmt(FD.energy(Array.from(f)), 2)}</b>, ∑(DCT)² = <b>${VZ.fmt(FD.energy(Array.from(Xd)), 2)}</b> (Parseval, difference ${Math.abs(FD.energy(Array.from(f)) - FD.energy(Array.from(Xd))).toExponential(2)})<br>`
      + (K >= N
        ? `Every coefficient retained: both reconstructions are exact, and the transforms differ only in how the energy was distributed on the way.`
        : eB.value === "ramp"
          ? `The ramp is the case that separates them. The DFT's periodic extension has a jump of ${VZ.fmt(f[N - 1] - f[0], 0)} at the block edge and spends its coefficients describing it; the DCT's mirror extension has no jump, so two coefficients already get within ${VZ.fmt(truncDCT(f, 2) ? FD.rmse(Array.from(f), Array.from(truncDCT(f, 2))) : 0, 2)} of an ${VZ.fmt(f[N - 1] - f[0], 0)}-unit range.`
          : eB.value === "noise"
            ? `On noise neither transform compacts anything, because there is no structure to compact. Energy compaction is a statement about the <i>signal class</i>, not about the transform alone.`
            : `The DCT is ahead by a factor of <b>${ed > 1e-12 ? VZ.fmt(ef / ed, 2) : "—"}</b> here. Switch to the extension view to see why: the DFT is spending coefficients on a discontinuity it invented at the block boundary.`);
  }
  eB.onchange = draw; eK.oninput = draw; eS.onchange = draw;
  draw();
})();

/* ═══ 20 · #wave-svg — lifting, the sub-bands, and exact reconstruction ═══════ */
(function () {
  const svg = d3.select("#wave-svg"); if (svg.empty()) return;
  const out = document.getElementById("wave-readout");
  const eT = document.getElementById("wv-t"), eS = document.getElementById("wv-s"),
        eL = document.getElementById("wv-l"), eLv = document.getElementById("wv-lv"),
        eK = document.getElementById("wv-k"), eKv = document.getElementById("wv-kv");
  const W = 770, H = 410, NS = 256;

  /* whole-sample mirror, matching the reference used to verify the filters */
  function ext(a, i) {
    const n = a.length; if (n === 1) return a[0];
    const p = 2 * n - 2, k = ((i % p) + p) % p;
    return k < n ? a[k] : a[p - k];
  }
  function fwd53(f) {
    const ne = Math.ceil(f.length / 2), no = f.length - ne;
    const e = VZ.zeros(ne), o = VZ.zeros(no);
    for (let k = 0; k < ne; k++) e[k] = f[2 * k];
    for (let k = 0; k < no; k++) o[k] = f[2 * k + 1];
    const d = VZ.zeros(no);
    for (let k = 0; k < no; k++) d[k] = o[k] - 0.5 * (ext(e, k) + ext(e, k + 1));
    const s = VZ.zeros(ne);
    for (let k = 0; k < ne; k++) s[k] = e[k] + 0.25 * (ext(d, k - 1) + ext(d, k));
    return [s, d];
  }
  function inv53(s, d) {
    const ne = s.length, no = d.length;
    const e = VZ.zeros(ne);
    for (let k = 0; k < ne; k++) e[k] = s[k] - 0.25 * (ext(d, k - 1) + ext(d, k));
    const o = VZ.zeros(no);
    for (let k = 0; k < no; k++) o[k] = d[k] + 0.5 * (ext(e, k) + ext(e, k + 1));
    const f = VZ.zeros(ne + no);
    for (let k = 0; k < ne; k++) f[2 * k] = e[k];
    for (let k = 0; k < no; k++) f[2 * k + 1] = o[k];
    return f;
  }
  const R2 = Math.SQRT2;
  function fwdHaar(f) {
    const n = f.length >> 1, s = VZ.zeros(n), d = VZ.zeros(n);
    for (let k = 0; k < n; k++) { s[k] = (f[2 * k] + f[2 * k + 1]) / R2; d[k] = (f[2 * k] - f[2 * k + 1]) / R2; }
    return [s, d];
  }
  function invHaar(s, d) {
    const n = s.length, f = VZ.zeros(2 * n);
    for (let k = 0; k < n; k++) { f[2 * k] = (s[k] + d[k]) / R2; f[2 * k + 1] = (s[k] - d[k]) / R2; }
    return f;
  }

  function signal(kind) {
    const f = VZ.zeros(NS), r = VZ.rng(13);
    for (let n = 0; n < NS; n++) {
      const t = n / NS;
      if (kind === "linear") f[n] = 3 * n + 7;
      else if (kind === "quad") f[n] = (n - NS / 2) * (n - NS / 2) / 200;
      else if (kind === "step") f[n] = t < 0.4 ? 10 : (t < 0.7 ? 70 : 30);
      else if (kind === "noise") f[n] = 40 + 14 * VZ.randn(r);
      else f[n] = 40 + 30 * Math.sin(2 * Math.PI * t) + (t > 0.55 && t < 0.62 ? 35 : 0) + 12 * t * t * 10 / 10;
    }
    return f;
  }

  function draw() {
    const haar = eT.value === "haar", J = VZ.clamp(Math.round(+eL.value), 1, 5);
    const keepPct = VZ.clamp(Math.round(+eK.value), 1, 100);
    eLv.textContent = String(J); eKv.textContent = keepPct + "%";
    const f = signal(eS.value);

    /* forward, keeping each level's detail */
    let cur = f;
    const details = [];
    for (let j = 0; j < J; j++) {
      const [s, d] = haar ? fwdHaar(cur) : fwd53(cur);
      details.push(d); cur = s;
    }
    const approx = cur;
    /* the flat coefficient vector, in the standard layout */
    const coeffs = Array.from(approx);
    for (let j = J - 1; j >= 0; j--) for (const v of details[j]) coeffs.push(v);

    /* thresholding: keep the largest keepPct% by magnitude */
    const mags = coeffs.map(Math.abs).slice().sort((a, b) => b - a);
    const nkeep = Math.max(1, Math.round(coeffs.length * keepPct / 100));
    const thr = mags[Math.min(nkeep - 1, mags.length - 1)];
    const kept = coeffs.map(v => (Math.abs(v) >= thr ? v : 0));
    let nz = 0; kept.forEach(v => { if (v !== 0) nz++; });

    /* inverse from the thresholded coefficients */
    let p = approx.length;
    let rec = Float64Array.from(kept.slice(0, p));
    for (let j = J - 1; j >= 0; j--) {
      const dl = details[j].length;
      const dd = Float64Array.from(kept.slice(p, p + dl)); p += dl;
      rec = haar ? invHaar(rec, dd) : inv53(rec, dd);
    }
    /* the exact round trip, with nothing discarded */
    let p2 = approx.length, exact = Float64Array.from(coeffs.slice(0, p2));
    for (let j = J - 1; j >= 0; j--) {
      const dl = details[j].length;
      const dd = Float64Array.from(coeffs.slice(p2, p2 + dl)); p2 += dl;
      exact = haar ? invHaar(exact, dd) : inv53(exact, dd);
    }
    let rt = 0;
    for (let n = 0; n < NS; n++) rt = Math.max(rt, Math.abs(exact[n] - f[n]));
    const err = FD.rmse(Array.from(f), Array.from(rec));

    const fr = VZ.frame(svg, W, H, { l: 46, r: 16, t: 24, b: 30 });
    const g = fr.g;

    const gA = VZ.panel(g, 0, 18, fr.iw, 96, "the signal, and the reconstruction from the retained coefficients").g;
    gA.append("rect").attr("width", fr.iw).attr("height", 96).attr("fill", VC.panel2).attr("stroke", VC.line);
    const all = Array.from(f).concat(Array.from(rec));
    const [lo, hi] = FD.minmax(all);
    const xa = d3.scaleLinear().domain([0, NS - 1]).range([8, fr.iw - 8]);
    const ya = d3.scaleLinear().domain([lo - 4, hi + 4]).range([88, 8]);
    gA.append("path").attr("d", d3.line().x((d, i) => xa(i)).y(d => ya(d))(Array.from(f)))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 2.4);
    gA.append("path").attr("d", d3.line().x((d, i) => xa(i)).y(d => ya(d))(Array.from(rec)))
      .attr("fill", "none").attr("stroke", keepPct >= 100 ? VC.good : VC.accent).attr("stroke-width", 1.4);
    VZ.legend(gA, [{ color: VC.muted, label: "original" }, { color: VC.accent, label: "reconstruction" }], 12, 16, { gap: 13 });
    VZ.axisL(gA, ya, 4, null);

    /* the coefficient layout */
    const gB = VZ.panel(g, 0, 144, fr.iw, 148, "the coefficients: the coarse approximation, then each level's detail band").g;
    gB.append("rect").attr("width", fr.iw).attr("height", 148).attr("fill", VC.panel2).attr("stroke", VC.line);
    const cmax = Math.max(...coeffs.map(Math.abs), 1e-9);
    const xb = d3.scaleLinear().domain([0, coeffs.length]).range([10, fr.iw - 10]);
    const yb = d3.scaleLinear().domain([-cmax * 1.1, cmax * 1.1]).range([132, 10]);
    gB.append("line").attr("x1", 10).attr("x2", fr.iw - 10).attr("y1", yb(0)).attr("y2", yb(0)).attr("stroke", VC.line);
    let off = approx.length;
    const bounds = [0, approx.length];
    for (let j = J - 1; j >= 0; j--) { off += details[j].length; bounds.push(off); }
    bounds.forEach((b, i) => {
      if (i === 0 || i === bounds.length - 1) return;
      gB.append("line").attr("x1", xb(b)).attr("x2", xb(b)).attr("y1", 10).attr("y2", 132)
        .attr("stroke", VC.a2).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "3 3");
    });
    FD.caption(gB, xb(0) + 4, 22, "approx  (2^-" + J + ")", VC.good, 9.5);
    for (let i = 1; i < bounds.length - 1; i++)
      FD.caption(gB, xb(bounds[i]) + 4, 22, "detail level " + (J - i + 1), VC.a2, 9.5);
    coeffs.forEach((v, i) => {
      const on = kept[i] !== 0;
      gB.append("line").attr("x1", xb(i)).attr("x2", xb(i)).attr("y1", yb(0)).attr("y2", yb(v))
        .attr("stroke", i < approx.length ? VC.good : (on ? VC.accent : VC.line))
        .attr("stroke-width", Math.max(0.7, (fr.iw - 20) / coeffs.length - 0.4));
    });
    VZ.axisL(gB, yb, 4, null);

    /* the error */
    const gC = VZ.panel(g, 0, 320, fr.iw, 68, "the reconstruction error").g;
    gC.append("rect").attr("width", fr.iw).attr("height", 68).attr("fill", VC.panel2).attr("stroke", VC.line);
    const diff = [];
    for (let n = 0; n < NS; n++) diff.push(rec[n] - f[n]);
    const dm = Math.max(...diff.map(Math.abs), 1e-9);
    const yc = d3.scaleLinear().domain([-dm * 1.15, dm * 1.15]).range([58, 8]);
    gC.append("line").attr("x1", 8).attr("x2", fr.iw - 8).attr("y1", yc(0)).attr("y2", yc(0)).attr("stroke", VC.grid);
    gC.append("path").attr("d", d3.line().x((d, i) => xa(i)).y(d => yc(d))(diff))
      .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 1.4);
    VZ.axisL(gC, yc, 3, null);
    VZ.axisB(gC, xa, 58, 8, "n");

    /* how many detail coefficients are exactly zero */
    let zeroD = 0, totD = 0;
    for (let j = 0; j < J; j++) for (const v of details[j]) { totD++; if (Math.abs(v) < 1e-12) zeroD++; }
    const eIn = FD.energy(Array.from(f)), eC = FD.energy(coeffs);
    out.innerHTML =
      `<b>${haar ? "Haar" : "lifted 5/3 (CDF)"}</b>, ${J} level${J > 1 ? "s" : ""} · <b>${coeffs.length}</b> coefficients for <b>${NS}</b> samples `
      + `(ratio <b>${VZ.fmt(coeffs.length / NS, 4)}</b> — a tight frame, not overcomplete)<br>`
      + `round trip with nothing discarded: max |f − reconstruction| = <b>${rt.toExponential(3)}</b> · `
      + `detail coefficients that are <b>exactly zero</b>: <b>${zeroD} of ${totD}</b> (${VZ.fmt(100 * zeroD / totD, 2)}%)<br>`
      + `thresholding at the ${keepPct}th percentile of |coefficient| (threshold ${VZ.fmt(thr, 4)}) leaves <b>${nz}</b> non-zero of ${coeffs.length}: RMSE <b>${VZ.fmt(err, 5)}</b>`
      + (haar ? ` · energy: ∑f² = <b>${VZ.fmt(eIn, 2)}</b>, ∑coeff² = <b>${VZ.fmt(eC, 2)}</b>, difference <b>${Math.abs(eIn - eC).toExponential(2)}</b> — Haar is <b>orthonormal</b>, so it preserves energy exactly` : ` · the 5/3 pair is <b>biorthogonal</b>, not orthonormal, so ∑coeff² = ${VZ.fmt(eC, 2)} against ∑f² = ${VZ.fmt(eIn, 2)}. The lifting steps as written have low-pass gain <b>1</b> rather than √2, so each level's approximation keeps the same amplitude on half as many samples and the coefficient energy falls by roughly 2 per level. It buys symmetry and short support instead of energy preservation.`)
      + `<br>`
      + (eS.value === "linear"
        ? `On a pure <b>linear ramp</b> every detail coefficient is exactly zero${haar ? " except where the ramp's own slope survives — Haar has only ONE vanishing moment, so it annihilates constants but not ramps" : " — the 5/3 high-pass filter has TWO vanishing moments, so it annihilates constants and ramps alike. That is the whole compression argument"}.`
        : eS.value === "quad"
          ? `On a <b>quadratic</b> the 5/3's detail coefficients are a non-zero constant: two vanishing moments and no more. A filter with three would kill this too, at the cost of longer support.`
          : eS.value === "noise"
            ? `On noise the detail bands are not sparse at all, and truncation destroys the signal immediately. Wavelets compress <i>piecewise-smooth</i> signals; nothing compresses noise.`
            : `The detail bands are near-zero everywhere the signal is smooth and spike only at the discontinuity — which is why keeping a small fraction of the coefficients costs so little.`);
  }
  eT.onchange = draw; eS.onchange = draw; eL.oninput = draw; eK.oninput = draw;
  draw();
})();

/* ═══ 21 · #tile-svg — four tilings of the space–frequency plane ══════════════ */
(function () {
  const svg = d3.select("#tile-svg"); if (svg.empty()) return;
  const out = document.getElementById("tile-readout");
  const eT = document.getElementById("tl-t"), eA = document.getElementById("tl-a"),
        eAv = document.getElementById("tl-av"), eE = document.getElementById("tl-e");
  const W = 770, H = 380;

  function draw() {
    const kind = eT.value, asp = Math.round(+eA.value), showE = eE.checked;
    eAv.textContent = (asp > 0 ? "+" : "") + asp;
    const fr = VZ.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 34 });
    const g = fr.g;
    const PW = fr.iw - 190, PH = fr.ih - 26;
    const gP = VZ.panel(g, 0, 20, PW, PH, "the space–frequency plane").g;
    gP.append("rect").attr("width", PW).attr("height", PH).attr("fill", VC.panel2).attr("stroke", VC.line);

    /* cells: each entry [x0, y0, w, h] in normalised plane coordinates (0..1) */
    const cells = [];
    const base = Math.pow(2, asp);
    if (kind === "samples") {
      const n = Math.round(32 * base);
      for (let i = 0; i < n; i++) cells.push([i / n, 0, 1 / n, 1]);
    } else if (kind === "fourier") {
      const n = Math.round(16 * base);
      for (let k = 0; k < n; k++) cells.push([0, k / n, 1, 1 / n]);
    } else if (kind === "stft") {
      const nx = Math.round(16 * base), ny = Math.round(512 / (16 * base) / 2);
      for (let i = 0; i < nx; i++) for (let k = 0; k < ny; k++) cells.push([i / nx, k / ny, 1 / nx, 1 / ny]);
    } else {
      /* wavelet: octave bands. band j occupies [2^-(j+1), 2^-j) in frequency and
         has 2^(J-1-j)... drawn from the top down so high frequency = many cells */
      const J = 5;
      for (let j = 0; j < J; j++) {
        const y1 = Math.pow(2, -j), y0 = Math.pow(2, -(j + 1));
        const nx = Math.round(Math.pow(2, J - j) * base);
        for (let i = 0; i < nx; i++) cells.push([i / nx, y0, 1 / nx, y1 - y0]);
      }
      /* the residual approximation band, split so that its cells have the SAME area
         as every other band's — otherwise the figure's own caption would be wrong */
      const nr = Math.round(2 * base);
      for (let i = 0; i < nr; i++) cells.push([i / nr, 0, 1 / nr, Math.pow(2, -J)]);
    }

    const X = t => 10 + t * (PW - 20), Y = t => PH - 26 - t * (PH - 40);
    cells.forEach(c => {
      gP.append("rect").attr("x", X(c[0])).attr("y", Y(c[1] + c[3]))
        .attr("width", X(c[0] + c[2]) - X(c[0])).attr("height", Y(c[1]) - Y(c[1] + c[3]))
        .attr("fill", VC.accent).attr("fill-opacity", 0.10).attr("stroke", VC.accent).attr("stroke-opacity", 0.55);
    });

    /* the two test events */
    const edge = [0.62, 0.80], grad = [0.30, 0.045];
    let nEdge = 0, nGrad = 0;
    cells.forEach(c => {
      if (edge[0] >= c[0] && edge[0] < c[0] + c[2] && edge[1] >= c[1] && edge[1] < c[1] + c[3]) nEdge++;
      if (grad[0] >= c[0] && grad[0] < c[0] + c[2] && grad[1] >= c[1] && grad[1] < c[1] + c[3]) nGrad++;
    });
    /* how many cells each event's true extent overlaps */
    const overlap = (x0, x1, y0, y1) => cells.filter(c =>
      c[0] < x1 && c[0] + c[2] > x0 && c[1] < y1 && c[1] + c[3] > y0).length;
    const nE = overlap(0.60, 0.64, 0.55, 1.0);          // an edge: compact in x, broad in freq
    const nG = overlap(0.0, 1.0, 0.005, 0.055);         // a gradient: broad in x, compact in freq
    if (showE) {
      gP.append("rect").attr("x", X(0.60)).attr("y", Y(1.0)).attr("width", X(0.64) - X(0.60)).attr("height", Y(0.55) - Y(1.0))
        .attr("fill", VC.bad).attr("fill-opacity", 0.3).attr("stroke", VC.bad).attr("stroke-width", 1.6);
      FD.caption(gP, X(0.64) + 6, Y(0.95), "an EDGE: compact in space, broad in frequency", VC.bad, 10);
      gP.append("rect").attr("x", X(0)).attr("y", Y(0.055)).attr("width", X(1) - X(0)).attr("height", Y(0.005) - Y(0.055))
        .attr("fill", VC.good).attr("fill-opacity", 0.3).attr("stroke", VC.good).attr("stroke-width", 1.6);
      FD.caption(gP, X(0.02) + 4, Y(0.10), "a GRADIENT: broad in space, compact in frequency", VC.good, 10);
    }
    const xs = d3.scaleLinear().domain([0, 1]).range([X(0), X(1)]);
    const ys = d3.scaleLinear().domain([0, 0.5]).range([Y(0), Y(1)]);
    gP.append("g").attr("class", "axis").attr("transform", `translate(0,${Y(0)})`).call(d3.axisBottom(xs).ticks(6));
    gP.append("g").attr("class", "axis").attr("transform", `translate(${X(0)},0)`).call(d3.axisLeft(ys).ticks(5));
    gP.append("text").attr("x", X(1)).attr("y", Y(0) + 30).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", VC.muted).text("position");
    gP.append("text").attr("x", X(0) - 4).attr("y", 10).attr("text-anchor", "start")
      .attr("font-size", 11).attr("fill", VC.muted).text("frequency");

    /* the side panel: cell areas */
    const gS = VZ.panel(g, PW + 24, 20, fr.iw - PW - 24, PH, "cell areas").g;
    gS.append("rect").attr("width", fr.iw - PW - 24).attr("height", PH).attr("fill", VC.panel2).attr("stroke", VC.line);
    const areas = cells.map(c => c[2] * c[3]);
    const uniq = [];
    areas.forEach(a => { if (!uniq.some(u => Math.abs(u - a) < 1e-9)) uniq.push(a); });
    uniq.sort((a, b) => b - a);
    FD.caption(gS, 12, 22, "distinct cell areas:", VC.ink, 10.5);
    uniq.slice(0, 8).forEach((a, i) => FD.caption(gS, 16, 40 + i * 15, a.toExponential(3), VC.accent, 10.5));
    FD.caption(gS, 12, 40 + Math.min(uniq.length, 8) * 15 + 12, uniq.length === 1 ? "all identical" : uniq.length + " distinct values", VC.a2, 10.5);
    FD.caption(gS, 12, 40 + Math.min(uniq.length, 8) * 15 + 28, "cells: " + cells.length, VC.muted, 10.5);
    FD.caption(gS, 12, 40 + Math.min(uniq.length, 8) * 15 + 46, "cells the edge spans: " + nE, VC.bad, 10.5);
    FD.caption(gS, 12, 40 + Math.min(uniq.length, 8) * 15 + 62, "cells the gradient spans: " + nG, VC.good, 10.5);

    out.innerHTML =
      `<b>${kind === "samples" ? "samples (pixels)" : kind === "fourier" ? "Fourier" : kind === "stft" ? "short-time, uniform grid" : "wavelet, octave bands"}</b> · `
      + `${cells.length} cells, ${uniq.length === 1 ? "all of area " + uniq[0].toExponential(3) : uniq.length + " distinct areas from " + uniq[uniq.length - 1].toExponential(3) + " to " + uniq[0].toExponential(3)}<br>`
      + `the compact <b>edge</b> spans <b>${nE}</b> cells · the extended <b>gradient</b> spans <b>${nG}</b> cells — fewer is better, because each cell is a coefficient you must store<br>`
      + (kind === "samples"
        ? `Perfect position, zero frequency information. The gradient needs <b>every</b> cell, which is exactly what "an image is a poor representation of a smooth gradient" means numerically.`
        : kind === "fourier"
          ? `Perfect frequency, zero position information. The edge needs <b>every</b> row, which is why a global Fourier transform is a bad representation of a picture with edges in it.`
          : kind === "stft"
            ? `One rectangle, everywhere. It is a compromise, and by construction it is the <i>same</i> compromise for the edge and the gradient — so it serves whichever of the two happens to match the chosen aspect and mis-serves the other. Change the aspect and watch the two counts trade against each other.`
            : `Cells halve in height and double in width at each octave, so the areas are <b>all equal</b> — the uncertainty budget is unchanged — but they are <i>allocated</i> differently at different frequencies. `
              + `The edge lands in a few short, wide, high-frequency cells; the gradient lands in a few tall, narrow, low-frequency ones. Both are cheap, which no fixed grid can manage.`);
  }
  eT.onchange = draw; eA.oninput = draw; eE.onchange = draw;
  draw();
})();

/* ═══ 22 · #notch-svg — periodic corruption, found and removed in frequency ═══ */
(function () {
  const svg = d3.select("#notch-svg"); if (svg.empty()) return;
  const out = document.getElementById("notch-readout");
  const eC = document.getElementById("nt-c"), eA = document.getElementById("nt-a"),
        eAv = document.getElementById("nt-av"), eM = document.getElementById("nt-m"),
        eW = document.getElementById("nt-w"), eWv = document.getElementById("nt-wv");
  const W = 770, H = 380, N = 64;
  const CLEAN = FD.house();

  function draw() {
    const kind = eC.value, amp = +eA.value, mode = eM.value, nw = +eW.value;
    eAv.textContent = VZ.fmt(amp, 2); eWv.textContent = VZ.fmt(nw, 1);

    /* the interference: exact integer frequencies so it lands on bins */
    const peaks = [];
    if (kind === "grating" || kind === "both") peaks.push([11, 7]);
    if (kind === "banding" || kind === "both") peaks.push([0, 14]);
    const dirty = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let v = CLEAN[i][j];
      peaks.forEach(([kx, ky]) => { v += amp * Math.cos(2 * Math.PI * (kx * j + ky * i) / N); });
      dirty[i][j] = v;
    }

    const F = FD.fft2(dirty);
    /* the mask */
    const G = [], maskImg = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) {
      G.push(new Array(N));
      for (let j = 0; j < N; j++) {
        const ky = i <= N / 2 ? i : i - N, kx = j <= N / 2 ? j : j - N;
        let m = 1;
        if (mode === "lowpass") {
          m = Math.hypot(kx, ky) <= 10 ? 1 : 0;
        } else if (mode !== "none") {
          peaks.forEach(([px, py]) => {
            [[px, py], [-px, -py]].forEach(([qx, qy]) => {
              const d = Math.hypot(kx - qx, ky - qy);
              if (mode === "hard") { if (d <= nw) m = 0; }
              else m *= (1 - Math.exp(-(d * d) / (2 * nw * nw)));
            });
          });
        }
        G[i][j] = FD.cscale(F[i][j], m);
        maskImg[(i + N / 2) % N][(j + N / 2) % N] = m;
      }
    }
    const fixed = FD.re2(FD.ifft2(G));

    const fr = VZ.frame(svg, W, H, { l: 16, r: 14, t: 24, b: 14 });
    const g = fr.g, S = 158, gap = 12;
    VZ.panel(g, 0, 24, S, S, "the clean image").g.call(s => FD.raster(s, CLEAN, 0, 0, S, S, { lo: 0, hi: 1 }));
    VZ.panel(g, S + gap, 24, S, S, "with periodic interference").g.call(s => FD.raster(s, dirty, 0, 0, S, S, { lo: -0.3, hi: 1.3 }));
    const gS = VZ.panel(g, 2 * (S + gap), 24, S, S, "its spectrum, with the notch drawn").g;
    const magS = FD.fftshift2(F.map(r => Float64Array.from(r, v => Math.log(1 + FD.cabs(v)))));
    const shown = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) shown[i][j] = magS[i][j] * (0.15 + 0.85 * maskImg[i][j]);
    FD.raster(gS, shown, 0, 0, S, S, {});
    const px = v => (v + N / 2) * S / N;
    peaks.forEach(([kx, ky]) => {
      [[kx, ky], [-kx, -ky]].forEach(([qx, qy]) => {
        gS.append("circle").attr("cx", px(qx)).attr("cy", px(qy)).attr("r", Math.max(4, nw * S / N))
          .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 1.6);
      });
    });
    VZ.panel(g, 3 * (S + gap), 24, S, S, mode === "none" ? "unrepaired" : "after the notch").g
      .call(s => FD.raster(s, fixed, 0, 0, S, S, { lo: 0, hi: 1 }));

    const gR = VZ.panel(g, 0, 24 + S + 26, fr.iw, 96, "the residual against the clean image, on a signed scale").g;
    const res = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) res[i][j] = fixed[i][j] - CLEAN[i][j];
    FD.raster(gR, res, 0, 0, 96, 96, { signed: true });
    const resD = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) resD[i][j] = dirty[i][j] - CLEAN[i][j];
    FD.raster(gR, resD, 110, 0, 96, 96, { signed: true });
    FD.caption(gR, 0, 110, "after", VC.muted, 10);
    FD.caption(gR, 110, 110, "before", VC.muted, 10);
    /* what a low-pass would leave, for comparison */
    const GL = [];
    for (let i = 0; i < N; i++) {
      GL.push(new Array(N));
      for (let j = 0; j < N; j++) {
        const ky = i <= N / 2 ? i : i - N, kx = j <= N / 2 ? j : j - N;
        GL[i][j] = FD.cscale(F[i][j], Math.hypot(kx, ky) <= 10 ? 1 : 0);
      }
    }
    const lp = FD.re2(FD.ifft2(GL));
    FD.raster(gR, lp, 226, 0, 96, 96, { lo: 0, hi: 1 });
    FD.caption(gR, 226, 110, "a low-pass instead", VC.muted, 10);
    FD.caption(gR, 340, 24, "The interference is compact in FREQUENCY and spread over the whole image in space.", VC.ink, 11);
    FD.caption(gR, 340, 42, "No spatial filter can separate it from the picture; in frequency it is two dots.", VC.muted, 11);
    FD.caption(gR, 340, 66, "A low-pass removes it only by removing everything above its cutoff —", VC.muted, 11);
    FD.caption(gR, 340, 82, "including all the picture's edges. That is the comparison on the left.", VC.muted, 11);

    const eBefore = FD.rmse2(CLEAN, dirty), eAfter = FD.rmse2(CLEAN, fixed), eLP = FD.rmse2(CLEAN, lp);
    const [rlo, rhi] = FD.minmax2(fixed);
    out.innerHTML =
      `RMSE against the clean image — before <b>${VZ.fmt(eBefore, 5)}</b>, after the ${mode === "none" ? "(no) repair" : mode + " notch"} <b>${VZ.fmt(eAfter, 5)}</b>, `
      + `a plain low-pass at |k| ≤ 10 <b>${VZ.fmt(eLP, 5)}</b><br>`
      + `the repaired image's range is [${VZ.fmt(rlo, 4)}, ${VZ.fmt(rhi, 4)}] against the original's [0.15, 0.95]<br>`
      + (kind === "none"
        ? `Nothing to remove. Add a grating or banding to see the effect.`
        : mode === "none"
          ? `Unrepaired. The interference is <b>everywhere</b> in the image and <b>nowhere</b> in the spectrum except two bright dots — which is exactly the situation the frequency domain exists for.`
          : mode === "hard"
            ? `A hard hole removes the interference (RMSE <b>${VZ.fmt(eBefore, 4)}</b> → <b>${VZ.fmt(eAfter, 4)}</b>) but multiplies the spectrum by a box, which convolves the image with a sinc — so it <b>rings</b>. Switch to the Gaussian notch and compare the residual.`
            : mode === "soft"
              ? `A Gaussian notch: RMSE <b>${VZ.fmt(eBefore, 4)}</b> → <b>${VZ.fmt(eAfter, 4)}</b>, and the residual is confined to the neighbourhood of the removed frequency rather than ringing across the picture. Note that the notch must be applied at <b>both</b> (k_x, k_y) and (−k_x, −k_y), or the result is complex.`
              : `A low-pass instead of a notch: RMSE <b>${VZ.fmt(eLP, 4)}</b>. It does remove the interference, and it removes every edge in the image with it. The interference and the content overlap in <i>space</i> completely and in <i>frequency</i> not at all, which is the entire argument for doing this here.`);
  }
  eC.onchange = draw; eA.oninput = draw; eM.onchange = draw; eW.oninput = draw;
  draw();
})();
