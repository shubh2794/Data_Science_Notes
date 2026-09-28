/* neural-hardware.viz.js — figures for deep-learning/classical-nn/neural-hardware.html
   (part 10 of the Classical Neural Networks series).

   Loaded after ../../data.js and ../../notes.js, so the palette C is available.
   Each figure is an IIFE that returns quietly when its <svg> is absent.
   Every number a readout prints is computed from the thing drawn — quantized
   weights, rounded accumulators, approximated sigmoids, simulated PE cycles,
   sampled bitstreams — never typed into a caption.

     1  #q-svg     fixed-point word length: weight histogram, network output, error vs bits
     2  #sr-svg    tiny weight updates: round-to-nearest vs truncation vs stochastic rounding
     3  #sig-svg   sigmoid in hardware: LUT, PWL (chord / balanced), PLAN, odd polynomial
     4  #sys-svg   a systolic array multiplying two small matrices, output- or weight-stationary
     5  #roof-svg  the roofline, with a draggable arithmetic intensity
     6  #sc-svg    stochastic computing: multiplication by AND of two bitstreams              */

/* ══════════ page-local helpers ══════════ */
const NH = (function () {
  const P = { A: C.A, B: C.B, good: C.good, bad: C.bad, ink: C.ink, muted: C.muted, line: C.line,
              violet: "#c084fc", teal: "#2dd4bf", rose: "#fb7185" };
  function rng(seed) {                         // mulberry32 — deterministic, so figures are reproducible
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const sig = x => 1 / (1 + Math.exp(-x));
  function frame(id, W, H) {
    const svg = d3.select(id);
    if (svg.empty()) return null;
    svg.attr("viewBox", "0 0 " + W + " " + H);
    svg.selectAll("*").remove();
    return svg;
  }
  function axis(g, gen, x, y) { return g.append("g").attr("class", "axis").attr("transform", "translate(" + x + "," + y + ")").call(gen); }
  function label(g, x, y, t, o) {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("fill", o.fill || P.muted)
      .attr("font-size", o.size || 11).attr("text-anchor", o.anchor || "start")
      .attr("font-weight", o.bold ? 600 : null).text(t);
  }
  const $ = id => document.getElementById(id);
  const sci = v => (v === 0 ? "0" : Math.abs(v) >= 0.01 && Math.abs(v) < 1e4 ? (+v).toPrecision(3) : (+v).toExponential(2));
  return { P, rng, gauss, sig, frame, axis, label, $, sci };
})();

/* ══════════ 1 · #q-svg — fixed-point word length ══════════ */
(function () {
  const svg0 = d3.select("#q-svg");
  if (svg0.empty()) return;
  const { P, rng, gauss, frame, axis, label, $, sci } = NH;
  const W = 760, H = 470;

  /* A 1-24-24-1 tanh network whose 673 parameters ARE the plotted weights. The
     distribution is a scale mixture — most weights small, a tenth of them wide —
     which is what trained layers tend to look like, and what makes clipping bite. */
  const H1 = 24, H2 = 24;
  const NPAR = H1 + H1 + H1 * H2 + H2 + H2 + 1;
  const r = rng(20240610);
  const w = [];
  for (let i = 0; i < NPAR; i++) w.push(gauss(r) * (r() < 0.9 ? 0.3 : 0.9));

  function net(p, x) {
    let o = 0;
    const W1 = 0, B1 = H1, W2 = 2 * H1, B2 = 2 * H1 + H1 * H2, W3 = B2 + H2, B3 = W3 + H2;
    const h1 = new Array(H1), h2 = new Array(H2);
    for (let i = 0; i < H1; i++) h1[i] = Math.tanh(p[W1 + i] * 2 * x + p[B1 + i]);
    for (let j = 0; j < H2; j++) {
      let s = p[B2 + j];
      for (let i = 0; i < H1; i++) s += p[W2 + j * H1 + i] * h1[i];
      h2[j] = Math.tanh(s);
    }
    o = p[B3];
    for (let j = 0; j < H2; j++) o += p[W3 + j] * h2[j];
    return o;
  }
  const XS = d3.range(161).map(i => -2 + i * 4 / 160);
  const yRef = XS.map(x => net(w, x));
  const yStd = Math.sqrt(d3.mean(yRef.map(v => (v - d3.mean(yRef)) ** 2)));

  /* signed Qm.n in b bits: 1 sign bit, m integer bits, n = b − 1 − m fraction bits */
  function quant(v, b, m, ovf, rnd) {
    const n = b - 1 - m, d = Math.pow(2, -n);
    let q = rnd === "near" ? Math.round(v / d) : Math.floor(v / d);
    const qmin = -Math.pow(2, b - 1), qmax = Math.pow(2, b - 1) - 1, span = Math.pow(2, b);
    if (ovf === "sat") q = Math.max(qmin, Math.min(qmax, q));
    else q = ((((q - qmin) % span) + span) % span) + qmin;
    return q * d;
  }
  function measure(b, m, ovf, rnd) {
    const wq = w.map(v => quant(v, b, m, ovf, rnd));
    let se = 0, sw = 0, mx = 0, clip = 0;
    const lo = -Math.pow(2, m), hi = Math.pow(2, m) - Math.pow(2, -(b - 1 - m));
    for (let i = 0; i < w.length; i++) {
      const e = wq[i] - w[i]; se += e * e; sw += w[i] * w[i]; mx = Math.max(mx, Math.abs(e));
      if (w[i] < lo || w[i] > hi) clip++;
    }
    const yq = XS.map(x => net(wq, x));
    const oe = Math.sqrt(d3.mean(yq.map((v, i) => (v - yRef[i]) ** 2)));
    return { wq, yq, wr: Math.sqrt(se / w.length), mx, clip, sqnr: 10 * Math.log10(sw / Math.max(se, 1e-300)), oe };
  }

  const bits = $("q-bits"), ib = $("q-int"), ovf = $("q-ovf"), rnd = $("q-rnd");
  function draw() {
    let b = +bits.value, m = +ib.value;
    if (m > b - 1) { m = b - 1; ib.value = m; }
    const n = b - 1 - m;
    $("q-bits-out").textContent = b; $("q-int-out").textContent = m;
    const R = measure(b, m, ovf.value, rnd.value);
    const svg = frame("#q-svg", W, H);

    /* panel A — the weights and where the quantizer puts them */
    const xa = d3.scaleLinear().domain([-2.5, 2.5]).range([50, 360]);
    const g = svg.append("g");
    const lo = -Math.pow(2, m), hi = Math.pow(2, m) - Math.pow(2, -n);
    g.append("rect").attr("x", xa(Math.max(lo, -2.5))).attr("y", 26).attr("width", xa(Math.min(hi, 2.5)) - xa(Math.max(lo, -2.5)))
      .attr("height", 174).attr("fill", P.A).attr("opacity", 0.06);
    const bins = d3.bin().domain([-2.5, 2.5]).thresholds(d3.range(-2.5, 2.5001, 0.1))(w);
    const ya = d3.scaleLinear().domain([0, d3.max(bins, d => d.length)]).range([200, 34]);
    g.selectAll("rect.h").data(bins).join("rect").attr("x", d => xa(d.x0) + 0.5).attr("width", d => Math.max(0, xa(d.x1) - xa(d.x0) - 1))
      .attr("y", d => ya(d.length)).attr("height", d => 200 - ya(d.length)).attr("fill", P.A).attr("opacity", 0.45);
    const qb = d3.bin().domain([-2.5, 2.5]).thresholds(d3.range(-2.5, 2.5001, 0.1))(R.wq.filter(v => v >= -2.5 && v <= 2.5));
    g.append("path").attr("d", d3.line().curve(d3.curveStepAfter).x(d => xa(d.x0)).y(d => ya(Math.min(d.length, ya.domain()[1] * 1.08)))(qb.concat([{ x0: 2.5, length: 0 }])))
      .attr("fill", "none").attr("stroke", P.B).attr("stroke-width", 1.6);
    const nLev = Math.pow(2, b);                          // the grid itself, drawn when it is sparse enough to see
    if (Math.min(hi, 2.5) - Math.max(lo, -2.5) < 90 * Math.pow(2, -n)) {
      for (let k = -nLev / 2; k < nLev / 2; k++) {
        const v = k * Math.pow(2, -n);
        if (v >= -2.5 && v <= 2.5) g.append("line").attr("x1", xa(v)).attr("x2", xa(v)).attr("y1", 200).attr("y2", 206).attr("stroke", P.B);
      }
    }
    axis(g, d3.axisBottom(xa).ticks(6), 0, 200);
    label(g, 50, 16, "weight histogram: float (blue), " + b + "-bit (amber); ticks = grid", { fill: P.ink });
    label(g, 50, 228, "representable range Q" + m + "." + n + " = [" + lo + ", " + sci(hi) + "]   ·   LSB = 2^−" + n + " = " + sci(Math.pow(2, -n)), { size: 10.5 });

    /* panel B — the network's output, float vs quantized weights */
    const xb = d3.scaleLinear().domain([-2, 2]).range([430, 740]);
    const ext = d3.extent(yRef.concat(R.yq.filter(v => isFinite(v))));
    const pad = (ext[1] - ext[0]) * 0.08 || 1;
    const yb = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([200, 34]).clamp(true);
    axis(g, d3.axisBottom(xb).ticks(5), 0, 200);
    axis(g, d3.axisLeft(yb).ticks(5), 430, 0);
    const ln = d3.line().x((d, i) => xb(XS[i])).y(d => yb(d));
    g.append("path").attr("d", ln(yRef)).attr("fill", "none").attr("stroke", P.A).attr("stroke-width", 2.2);
    g.append("path").attr("d", ln(R.yq)).attr("fill", "none").attr("stroke", P.B).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 3");
    label(g, 430, 16, "network output f(x): float (blue), " + b + "-bit (amber)", { fill: P.ink });

    /* panel C — error against word length, for the current m, overflow and rounding modes */
    const sweep = d3.range(2, 17).map(bb => {
      const mm = Math.min(m, bb - 1);
      const s = measure(bb, mm, ovf.value, rnd.value);
      return { b: bb, wr: Math.max(s.wr, 1e-7), oe: Math.max(s.oe, 1e-7), th: Math.pow(2, -(bb - 1 - mm)) / Math.sqrt(12) };
    });
    const xc = d3.scaleLinear().domain([2, 16]).range([60, 740]);
    const yc = d3.scaleLog().domain([1e-6, 10]).range([440, 270]).clamp(true);
    axis(g, d3.axisBottom(xc).ticks(14).tickFormat(d3.format("d")), 0, 440);
    axis(g, d3.axisLeft(yc).ticks(4, "~e"), 60, 0);
    const lc = k => d3.line().x(d => xc(d.b)).y(d => yc(d[k]));
    g.append("path").attr("d", lc("th")(sweep)).attr("fill", "none").attr("stroke", P.muted).attr("stroke-dasharray", "3 3");
    g.append("path").attr("d", lc("wr")(sweep)).attr("fill", "none").attr("stroke", P.A).attr("stroke-width", 2);
    g.append("path").attr("d", lc("oe")(sweep)).attr("fill", "none").attr("stroke", P.B).attr("stroke-width", 2);
    g.append("line").attr("x1", xc(b)).attr("x2", xc(b)).attr("y1", 270).attr("y2", 440).attr("stroke", P.ink).attr("opacity", 0.35);
    label(g, 60, 258, "RMS error vs word length (Q" + m + ".n): weights (blue), network output (amber), LSB/√12 for rounding (dashed)", { fill: P.ink });
    label(g, 400, 468, "total bits b", { anchor: "middle" });

    $("q-readout").innerHTML =
      "Q" + m + "." + n + " (" + b + " bits) · weight RMS error <b>" + sci(R.wr) + "</b> (LSB/√12 = " + sci(Math.pow(2, -n) / Math.sqrt(12)) +
      ") · max |error| <b>" + sci(R.mx) + "</b> · out of range: <b>" + R.clip + "</b> of " + w.length +
      " · weight SQNR <b>" + R.sqnr.toFixed(1) + " dB</b><br>network output RMS error <b>" + sci(R.oe) + "</b> = <b>" +
      (100 * R.oe / yStd).toFixed(2) + "%</b> of the output's own spread" +
      (R.clip && ovf.value === "wrap" ? " — <b style='color:" + P.bad + "'>wrap-around turns each overflow into a sign flip</b>" : "");
  }
  [bits, ib, ovf, rnd].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ══════════ 2 · #sr-svg — updates smaller than one LSB ══════════ */
(function () {
  const svg0 = d3.select("#sr-svg");
  if (svg0.empty()) return;
  const { P, rng, frame, axis, label, $ } = NH;
  const W = 760, H = 330, T = 400;
  const us = $("sr-u");
  let seed = 7;
  function draw() {
    const u = +us.value;
    $("sr-u-out").textContent = u.toFixed(2);
    const r = rng(seed);
    /* all quantities in units of one LSB, weight starting at 0 */
    const exact = [0], near = [0], trunc = [0], master = [0], sr = [[0], [0], [0], [0], [0]];
    let e = 0, a = 0, t = 0, mst = 0;
    const s = [0, 0, 0, 0, 0];
    for (let k = 1; k <= T; k++) {
      e += u; exact.push(e);
      a = Math.round(a + u); near.push(a);          // round-to-nearest after every update
      t = Math.floor(t + u); trunc.push(t);         // truncation (two's complement: toward −∞)
      mst += u; master.push(Math.round(mst));       // a wide master copy, rounded only when read
      for (let j = 0; j < 5; j++) {                 // stochastic rounding: up with probability = fraction
        const v = s[j] + u, f = Math.floor(v);
        s[j] = f + (r() < v - f ? 1 : 0); sr[j].push(s[j]);
      }
    }
    const svg = frame("#sr-svg", W, H);
    const x = d3.scaleLinear().domain([0, T]).range([60, 540]);
    const top = Math.max(e, d3.max(sr, p => p[T]), 1) * 1.08;
    const y = d3.scaleLinear().domain([0, top]).range([290, 24]);
    axis(svg, d3.axisBottom(x).ticks(8), 0, 290);
    axis(svg, d3.axisLeft(y).ticks(6), 60, 0);
    const ln = d3.line().x((d, i) => x(i)).y(d => y(d));
    sr.forEach((p, j) => svg.append("path").attr("d", ln(p)).attr("fill", "none").attr("stroke", P.good).attr("opacity", j ? 0.35 : 0.95).attr("stroke-width", j ? 1 : 1.8));
    svg.append("path").attr("d", ln(exact)).attr("fill", "none").attr("stroke", P.A).attr("stroke-width", 2);
    svg.append("path").attr("d", ln(master)).attr("fill", "none").attr("stroke", P.violet).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
    svg.append("path").attr("d", ln(near)).attr("fill", "none").attr("stroke", P.B).attr("stroke-width", 2.4);
    svg.append("path").attr("d", ln(trunc)).attr("fill", "none").attr("stroke", P.bad).attr("stroke-width", 1.6);
    label(svg, 300, 318, "update number", { anchor: "middle" });
    label(svg, 60, 14, "weight value, in LSBs", { fill: P.ink });
    const L = [["exact (real arithmetic)", P.A], ["round-to-nearest", P.B], ["truncate", P.bad], ["stochastic rounding (5 runs)", P.good], ["wide master copy, rounded on read", P.violet]];
    L.forEach((d, i) => {
      svg.append("rect").attr("x", 556).attr("y", 40 + i * 22).attr("width", 12).attr("height", 3).attr("fill", d[1]);
      label(svg, 574, 45 + i * 22, d[0], { size: 10.5 });
    });
    const srMean = d3.mean(sr, p => p[T]);
    const f = u - Math.floor(u);
    $("sr-readout").innerHTML = "after " + T + " updates of " + u.toFixed(2) + " LSB: exact <b>" + e.toFixed(1) +
      "</b> · round-to-nearest <b>" + a + "</b> · truncate <b>" + t + "</b> · master copy <b>" + Math.round(mst) +
      "</b> · stochastic (mean of 5) <b>" + srMean.toFixed(1) + "</b>, spread predicted √(T·f(1−f)) = <b>" +
      Math.sqrt(T * f * (1 - f)).toFixed(1) + "</b> LSB";
  }
  us.addEventListener("input", draw);
  $("sr-new").addEventListener("click", () => { seed = (seed * 2654435761 + 1) >>> 0; draw(); });
  draw();
})();

/* ══════════ 3 · #sig-svg — the sigmoid, five hardware ways ══════════ */
(function () {
  const svg0 = d3.select("#sig-svg");
  if (svg0.empty()) return;
  const { P, sig, frame, axis, label, $, sci } = NH;
  const W = 760, H = 440;
  const nS = $("sig-n"), dS = $("sig-deg"), RS = $("sig-R"), qS = $("sig-q"), hS = $("sig-show");

  /* PLAN (Amin, Curtis & Hayes-Gill 1997): four segments with power-of-two slopes, so the
     "multiply" is a shift and the whole thing is shifts and adds. */
  function plan(x) {
    const a = Math.abs(x);
    const y = a >= 5 ? 1 : a >= 2.375 ? 0.03125 * a + 0.84375 : a >= 1 ? 0.125 * a + 0.625 : 0.25 * a + 0.5;
    return x >= 0 ? y : 1 - y;
  }
  /* least-squares odd polynomial for σ(x) − ½ in t = x/R on [0, R] (odd symmetry does the rest) */
  function fitOdd(R, deg) {
    const K = (deg + 1) / 2, M = 400;
    const A = d3.range(K).map(() => new Array(K + 1).fill(0));
    for (let s = 0; s < M; s++) {
      const t = 0.5 - 0.5 * Math.cos(Math.PI * (s + 0.5) / M);          // Chebyshev nodes on [0,1]
      const phi = d3.range(K).map(k => Math.pow(t, 2 * k + 1)), yv = sig(t * R) - 0.5;
      for (let i = 0; i < K; i++) { for (let j = 0; j < K; j++) A[i][j] += phi[i] * phi[j]; A[i][K] += phi[i] * yv; }
    }
    for (let c = 0; c < K; c++) {                                        // Gauss–Jordan with partial pivoting
      let p = c; for (let i = c + 1; i < K; i++) if (Math.abs(A[i][c]) > Math.abs(A[p][c])) p = i;
      [A[c], A[p]] = [A[p], A[c]];
      for (let i = 0; i < K; i++) if (i !== c) { const f = A[i][c] / A[c][c]; for (let j = c; j <= K; j++) A[i][j] -= f * A[c][j]; }
    }
    return d3.range(K).map(i => A[i][K] / A[i][i]);
  }

  function build(N, deg, R, qb) {
    const cell = 2 * R / N;
    const Q = v => (qb ? Math.round(v * Math.pow(2, qb)) / Math.pow(2, qb) : v);
    const sat = (x, f) => (x >= R ? 1 : x < -R ? 0 : f(x));
    /* LUT: N words, address = the top log2(N) bits of x over [−R, R), value at the cell centre */
    const lut = d3.range(N).map(k => Q(sig(-R + (k + 0.5) * cell)));
    const fLUT = x => sat(x, x => lut[Math.min(N - 1, Math.floor((x + R) / cell))]);
    /* PWL: N segments, a (slope, intercept) pair per segment — two words, one multiply, one add */
    const chord = [], bal = [];
    for (let k = 0; k < N; k++) {
      const x0 = -R + k * cell, x1 = x0 + cell, y0 = sig(x0), y1 = sig(x1);
      const s = (y1 - y0) / cell, c = y0 - s * x0;
      let dev = 0;                                       // largest signed gap between σ and the chord
      for (let i = 1; i < 64; i++) { const xx = x0 + i * cell / 64, dv = sig(xx) - (s * xx + c); if (Math.abs(dv) > Math.abs(dev)) dev = dv; }
      chord.push([Q(s), Q(c)]); bal.push([Q(s), Q(c + dev / 2)]);  // shift by half the gap: error equioscillates
    }
    const seg = (tab) => x => sat(x, x => { const k = Math.min(N - 1, Math.floor((x + R) / cell)); return tab[k][0] * x + tab[k][1]; });
    const co = fitOdd(R, deg).map(Q);
    const fPoly = x => sat(x, x => { const t = x / R, t2 = t * t; let acc = 0; for (let k = co.length - 1; k >= 0; k--) acc = acc * t2 + co[k]; return Math.min(1, Math.max(0, 0.5 + acc * t)); });
    return [
      { key: "LUT", name: "LUT, " + N + " entries", f: fLUT, col: P.B, words: N, mul: 0 },
      { key: "PWLc", name: "PWL chord, " + N + " seg", f: seg(chord), col: P.violet, words: 2 * N, mul: 1 },
      { key: "PWLb", name: "PWL balanced, " + N + " seg", f: seg(bal), col: P.good, words: 2 * N, mul: 1 },
      { key: "PLAN", name: "PLAN (shifts + adds)", f: x => Q(plan(x)), col: P.teal, words: 0, mul: 0 },
      { key: "poly", name: "odd poly, degree " + deg, f: fPoly, col: P.rose, words: co.length, mul: co.length + 1 }
    ];
  }

  function draw() {
    const N = Math.pow(2, +nS.value), deg = +dS.value, R = +RS.value, qb = +qS.value;
    $("sig-n-out").textContent = N; $("sig-deg-out").textContent = deg;
    const M = build(N, deg, R, qb);
    const xs = d3.range(1601).map(i => -10 + i * 20 / 1600);
    M.forEach(m => { m.err = xs.map(x => m.f(x) - sig(x)); m.max = d3.max(m.err, Math.abs); });

    const svg = frame("#sig-svg", W, H);
    const x = d3.scaleLinear().domain([-10, 10]).range([60, 740]);
    const y1 = d3.scaleLinear().domain([-0.05, 1.05]).range([150, 24]);
    axis(svg, d3.axisBottom(x).ticks(10), 0, 150);
    axis(svg, d3.axisLeft(y1).ticks(3), 60, 0);
    svg.append("path").attr("d", d3.line().x(d => x(d)).y(d => y1(sig(d)))(xs)).attr("fill", "none").attr("stroke", P.A).attr("stroke-width", 4).attr("opacity", 0.5);
    M.forEach(m => svg.append("path").attr("d", d3.line().x(d => x(d)).y(d => y1(m.f(d)))(xs)).attr("fill", "none").attr("stroke", m.col)
      .attr("stroke-width", hS.value === m.key ? 2 : 1.1).attr("opacity", hS.value === "all" || hS.value === m.key ? 1 : 0.12));
    label(svg, 60, 14, "σ(x) (thick blue) and the five approximations", { fill: P.ink });

    const y2 = d3.scaleLog().domain([1e-7, 0.3]).range([410, 190]).clamp(true);
    axis(svg, d3.axisBottom(x).ticks(10), 0, 410);
    axis(svg, d3.axisLeft(y2).ticks(6, "~e"), 60, 0);
    svg.append("line").attr("x1", 60).attr("x2", 740).attr("y1", y2(Math.pow(2, -9))).attr("y2", y2(Math.pow(2, -9)))
      .attr("stroke", P.muted).attr("stroke-dasharray", "2 3").attr("opacity", 0.6);
    label(svg, 738, y2(Math.pow(2, -9)) - 3, "½ LSB of 8 bits", { anchor: "end", size: 9.5 });
    M.forEach(m => svg.append("path").attr("d", d3.line().x((d, i) => x(xs[i])).y(d => y2(Math.max(Math.abs(d), 1e-7)))(m.err))
      .attr("fill", "none").attr("stroke", m.col).attr("stroke-width", hS.value === m.key ? 2 : 1.1)
      .attr("opacity", hS.value === "all" || hS.value === m.key ? 0.95 : 0.12));
    label(svg, 60, 182, "|approximation − σ(x)|, log scale  (outside [−" + R + ", " + R + "] every method saturates to 0 or 1)", { fill: P.ink });
    label(svg, 400, 434, "x", { anchor: "middle" });

    $("sig-readout").innerHTML = '<table class="cmp" style="margin:6px 0"><tr><th>method</th><th>max |error| on [−10, 10]</th><th>≈ bits of accuracy</th><th>stored words</th><th>multiplies</th></tr>' +
      M.map(m => '<tr><td><span style="color:' + m.col + '">■</span> ' + m.name + "</td><td><b>" + sci(m.max) + "</b></td><td>" +
        (-Math.log2(m.max)).toFixed(1) + "</td><td>" + (m.words || "0 (constants are shifts)") + "</td><td>" + m.mul + "</td></tr>").join("") +
      "</table>saturation floor from the input range alone: σ(−" + R + ") = <b>" + sci(sig(-R)) + "</b>" +
      (qb ? " · coefficients and table words rounded to " + qb + " fraction bits" : " · coefficients in full precision");
  }
  [nS, dS, RS, qS, hS].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ══════════ 4 · #sys-svg — systolic matrix multiply ══════════ */
(function () {
  const svg0 = d3.select("#sys-svg");
  if (svg0.empty()) return;
  const { P, rng, frame, label, $ } = NH;
  const W = 760, H = 420;
  const nS = $("sys-n"), mS = $("sys-mode");
  let n = 3, t = 0, A, B, timer = null;

  function reset() {
    n = +nS.value; t = 0;
    const r = rng(31 + n);
    A = d3.range(n).map(() => d3.range(n).map(() => Math.floor(r() * 5)));
    B = d3.range(n).map(() => d3.range(n).map(() => Math.floor(r() * 5) - 1));
    stop(); draw();
  }
  const T = () => 3 * n - 2;
  /* In both dataflows, PE(r, c) is busy in cycle τ exactly when idx = τ − r − c lies in [0, n):
       output-stationary: PE(i, j) holds C[i][j]; idx is the reduction index k
       weight-stationary: PE(k, j) holds B[k][j];  idx is the row i of A whose partial sum passes through */
  function partial(i, j, kmax) { let s = 0; for (let k = 0; k <= kmax && k < n; k++) s += A[i][k] * B[k][j]; return s; }

  function draw() {
    const ws = mS.value === "ws";
    const svg = frame("#sys-svg", W, H);
    const cs = n === 4 ? 54 : 64, gx = 250, gy = 150;
    const last = t - 1;                                  // the cycle most recently executed
    let busy = 0, macs = 0;
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) for (let tau = 0; tau < t; tau++) { const k = tau - r - c; if (k >= 0 && k < n) macs++; }

    // wires
    for (let r = 0; r < n; r++) svg.append("line").attr("x1", gx - 8).attr("x2", gx + n * cs).attr("y1", gy + r * cs + cs / 2).attr("y2", gy + r * cs + cs / 2).attr("stroke", P.line);
    for (let c = 0; c < n; c++) svg.append("line").attr("y1", gy - 8).attr("y2", gy + n * cs + 8).attr("x1", gx + c * cs + cs / 2).attr("x2", gx + c * cs + cs / 2).attr("stroke", P.line);

    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const idx = last - r - c, act = t > 0 && idx >= 0 && idx < n;
      if (act) busy++;
      const x0 = gx + c * cs + 3, y0 = gy + r * cs + 3;
      svg.append("rect").attr("x", x0).attr("y", y0).attr("width", cs - 6).attr("height", cs - 6).attr("rx", 6)
        .attr("fill", act ? "rgba(74,222,128,.16)" : "#1e222d").attr("stroke", act ? P.good : P.line);
      if (!ws) {
        const done = Math.min(n - 1, last - r - c);
        const acc = done >= 0 ? partial(r, c, done) : 0;
        label(svg, x0 + (cs - 6) / 2, y0 + cs / 2 + 2, String(acc), { anchor: "middle", fill: P.ink, size: 15, bold: true });
        if (act) label(svg, x0 + 4, y0 + 11, "a" + r + (idx) + "·b" + idx + c, { size: 9, fill: P.good });
      } else {
        label(svg, x0 + (cs - 6) - 3, y0 + 11, "w=" + B[r][c], { anchor: "end", size: 9.5, fill: P.B });
        if (act) {
          label(svg, x0 + (cs - 6) / 2, y0 + cs / 2 + 4, String(partial(idx, c, r)), { anchor: "middle", fill: P.ink, size: 14, bold: true });
          label(svg, x0 + 4, y0 + cs - 10, "row " + idx, { size: 9, fill: P.good });
        }
      }
    }
    // input queues (skewed): row r of the left queue carries A[i][k] entering at cycle i + k
    for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
      const enter = ws ? q + r : r + q;          // os: element k=q of row i=r;  ws: element i=q of column k=r
      const slot = enter - t + 1;
      if (slot < 1) continue;
      const v = ws ? A[q][r] : A[r][q];
      const cx = gx - slot * 26 - 4, cy = gy + r * cs + cs / 2;
      if (cx < 10) continue;
      svg.append("rect").attr("x", cx - 11).attr("y", cy - 10).attr("width", 22).attr("height", 20).attr("rx", 4).attr("fill", "rgba(91,156,255,.18)").attr("stroke", P.A);
      label(svg, cx, cy + 4, String(v), { anchor: "middle", fill: P.ink, size: 11 });
    }
    if (!ws) {
      for (let c = 0; c < n; c++) for (let k = 0; k < n; k++) {
        const slot = k + c - t + 1;
        if (slot < 1) continue;
        const sp = n === 4 ? 17 : 20, cx = gx + c * cs + cs / 2, cy = gy - slot * sp - 2;
        if (cy < 26) continue;
        svg.append("rect").attr("x", cx - 11).attr("y", cy - 9).attr("width", 22).attr("height", 15).attr("rx", 4).attr("fill", "rgba(255,180,84,.18)").attr("stroke", P.B);
        label(svg, cx, cy + 3, String(B[k][c]), { anchor: "middle", fill: P.ink, size: 11 });
      }
      label(svg, gx, gy + n * cs + 22, "A streams in from the left (row i delayed i cycles), B from the top; each PE keeps its C[i][j]", { size: 10.5 });
    } else {
      label(svg, gx, gy - 14, "weights B preloaded, one per PE; partial sums flow down, A flows right", { size: 10.5 });
      label(svg, gx, gy + n * cs + 22, "column j emits C[i][j] at its bottom edge", { size: 10.5 });
    }
    label(svg, 20, 20, (ws ? "weight-stationary" : "output-stationary") + " · cycle " + t + " of " + T(), { fill: P.ink, size: 13, bold: true });

    // result matrix
    const rx = 590, ry = 150, rc = 36;
    label(svg, rx, ry - 12, "C = A·B", { fill: P.ink });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const ready = t >= i + j + n;
      svg.append("rect").attr("x", rx + j * rc).attr("y", ry + i * rc).attr("width", rc - 3).attr("height", rc - 3).attr("rx", 4)
        .attr("fill", ready ? "rgba(74,222,128,.14)" : "none").attr("stroke", ready ? P.good : P.line);
      if (ready) label(svg, rx + j * rc + (rc - 3) / 2, ry + i * rc + rc / 2 + 2, String(partial(i, j, n - 1)), { anchor: "middle", fill: P.ink, size: 12 });
    }
    const mx = (M, x0, y0, nm, col) => {
      label(svg, x0, y0 - 6, nm, { fill: col, size: 10.5 });
      M.forEach((row, i) => label(svg, x0, y0 + 12 + i * 14, row.map(v => (v < 0 ? "" : " ") + v).join(" "), { fill: P.muted, size: 10.5 }));
    };
    const my = ry + n * rc + 36;
    mx(A, 590, my, "A", P.A); mx(B, 680, my, "B", P.B);

    $("sys-readout").innerHTML = "cycle <b>" + t + "</b>/" + T() + " · PEs busy this cycle <b>" + (t ? busy : 0) + "</b> of " + n * n +
      " · MACs done <b>" + macs + "</b> of n³ = " + n * n * n + " · whole-run utilisation n³/(n²·(3n−2)) = <b>" +
      (100 * n / T()).toFixed(0) + "%</b>" + (t >= T() ? " · <b style='color:" + P.good + "'>done — every A and B element was read from memory once and used n = " + n + " times</b>" : "");
  }
  function stepF() { if (t < T()) { t++; draw(); } else stop(); }
  function stop() { if (timer) { timer.stop(); timer = null; } $("sys-play").textContent = "▶ play"; }
  $("sys-step").addEventListener("click", () => { stop(); stepF(); });
  $("sys-play").addEventListener("click", () => {
    if (timer) { stop(); return; }
    if (t >= T()) { t = 0; draw(); }
    $("sys-play").textContent = "❚❚ pause";
    timer = d3.interval(stepF, 750);
  });
  $("sys-reset").addEventListener("click", () => { stop(); t = 0; draw(); });
  nS.addEventListener("input", reset);
  mS.addEventListener("input", () => { stop(); t = 0; draw(); });
  reset();
})();

/* ══════════ 5 · #roof-svg — roofline with draggable intensity ══════════ */
(function () {
  const svg0 = d3.select("#roof-svg");
  if (svg0.empty()) return;
  const { P, frame, axis, label, $, sci } = NH;
  const W = 760, H = 400;
  const DEV = {
    h100: { name: "datacenter GPU — H100 SXM, dense BF16", P: 989e12, B: 3.35e12 },
    tpu1: { name: "TPU v1 (2015), 8-bit", P: 92e12, B: 34e9 },
    cpu: { name: "server CPU, FP32 (illustrative)", P: 3e12, B: 300e9 },
    fpga: { name: "mid-size FPGA, int8 (illustrative)", P: 4e12, B: 19.2e9 },
    npu: { name: "phone / edge NPU, int8 (illustrative)", P: 4e12, B: 17e9 }
  };
  /* workload markers, intensities for 2-byte operands */
  const WL = [
    { n: "elementwise add", I: 1 / 6 },
    { n: "batch-1 dense layer / LLM decode", I: 1 },
    { n: "3×3 conv, 64→64 ch, 56×56", I: 2 * 56 * 56 * 64 * 64 * 9 / (2 * (56 * 56 * 64 * 2 + 64 * 64 * 9)) },
    { n: "4096³ GEMM", I: 2 * Math.pow(4096, 3) / (2 * 3 * 4096 * 4096) }
  ];
  const dS = $("roof-dev"), bS = $("roof-batch");
  let I = 1, xNow = null;
  const d = 4096;
  svg0.style("cursor", "ew-resize").call(d3.drag().on("start drag", ev => {
    if (!xNow) return;
    I = xNow.invert(Math.max(70, Math.min(560, ev.x))); bS.dataset.custom = "1"; draw();
  }));
  const fromBatch = b => 2 * b * d * d / (2 * (d * d + 2 * b * d));   // dense d×d layer, batch b, 2-byte values

  function draw() {
    const D = DEV[dS.value];
    const svg = frame("#roof-svg", W, H);
    const x = d3.scaleLog().domain([0.05, 1e4]).range([70, 560]).clamp(true);
    xNow = x;
    const y = d3.scaleLog().domain([D.P / 1e5, D.P * 3]).range([350, 30]);
    axis(svg, d3.axisBottom(x).ticks(6, "~g"), 0, 350);
    axis(svg, d3.axisLeft(y).ticks(5, "~s"), 70, 0);
    const ridge = D.P / D.B;
    const pts = d3.range(0, 201).map(i => 0.05 * Math.pow(1e4 / 0.05, i / 200));
    svg.append("path").attr("d", d3.line().x(v => x(v)).y(v => y(Math.min(D.P, D.B * v)))(pts)).attr("fill", "none").attr("stroke", P.A).attr("stroke-width", 2.6);
    svg.append("line").attr("x1", x(ridge)).attr("x2", x(ridge)).attr("y1", y(D.P)).attr("y2", 350).attr("stroke", P.muted).attr("stroke-dasharray", "3 3");
    label(svg, x(ridge) + 4, 344, "ridge " + sci(ridge) + " FLOP/B", { size: 10 });
    label(svg, 76, y(D.P) - 6, "peak compute " + sci(D.P / 1e12) + " T/s", { size: 10, fill: P.A });
    WL.forEach((w, i) => {
      const yy = y(Math.min(D.P, D.B * w.I));
      svg.append("circle").attr("cx", x(w.I)).attr("cy", yy).attr("r", 3.5).attr("fill", P.muted);
      const onRoof = D.B * w.I >= D.P;
      if (onRoof) label(svg, x(w.I), yy + 16 + (i % 2) * 12, w.n, { size: 9.5, anchor: "middle" });
      else if (i >= 2) label(svg, x(w.I) - 6, yy - 8, w.n, { size: 9.5, anchor: "end" });
      else label(svg, x(w.I) + 6, yy + 14, w.n, { size: 9.5 });
    });
    const perf = Math.min(D.P, D.B * I);
    const g = svg.append("g").attr("class", "dragpt");
    g.append("line").attr("x1", x(I)).attr("x2", x(I)).attr("y1", 350).attr("y2", y(perf)).attr("stroke", P.B).attr("stroke-dasharray", "2 2");
    g.append("circle").attr("cx", x(I)).attr("cy", y(perf)).attr("r", 9).attr("fill", P.B).attr("stroke", "#0f1117").attr("stroke-width", 2);
    label(svg, 315, 384, "arithmetic intensity I (FLOP per byte moved from DRAM / HBM)", { anchor: "middle" });
    label(svg, 70, 18, "attainable FLOP/s = min(peak, bandwidth × I)", { fill: P.ink });

    // side panel
    const bound = I < ridge ? "memory-bound" : "compute-bound";
    const bw = D.B >= 1e12 ? sci(D.B / 1e12) + " TB/s" : sci(D.B / 1e9) + " GB/s";
    const side = [["peak", sci(D.P / 1e12) + " TFLOP/s"], ["bandwidth", bw],
      ["ridge point", sci(ridge) + " FLOP/B"], ["your I", sci(I) + " FLOP/B"], ["attainable", sci(perf / 1e12) + " TFLOP/s"],
      ["of peak", (100 * perf / D.P).toFixed(1) + "%"], ["verdict", bound]];
    side.forEach((s, i) => { label(svg, 590, 56 + i * 38, s[0], { size: 10 }); label(svg, 590, 71 + i * 38, s[1], { size: 12.5, fill: i === 6 ? (I < ridge ? P.B : P.good) : P.ink, bold: i > 3 }); });
    const bTxt = bS.dataset.custom ? "custom (dragged)" : String(Math.pow(2, +bS.value));
    $("roof-batch-out").textContent = bTxt;
    $("roof-readout").innerHTML = "I = <b>" + sci(I) + "</b> FLOP/byte → <b>" + sci(perf / 1e12) + "</b> TFLOP/s, <b>" + (100 * perf / D.P).toFixed(1) +
      "%</b> of peak (" + bound + "). " + (ridge < d / 2
      ? "A " + d + "×" + d + " BF16 layer reaches this device's ridge at batch ≈ <b>" + Math.ceil(ridge * d / (d - 2 * ridge)) + "</b>."
      : "A " + d + "×" + d + " BF16 layer can never reach this ridge: its intensity tops out at d/2 = " + d / 2 + " FLOP/byte.");
  }
  dS.addEventListener("input", draw);
  bS.addEventListener("input", () => { delete bS.dataset.custom; I = fromBatch(Math.pow(2, +bS.value)); draw(); });
  I = fromBatch(Math.pow(2, +bS.value));
  draw();
})();

/* ══════════ 6 · #sc-svg — stochastic computing ══════════ */
(function () {
  const svg0 = d3.select("#sc-svg");
  if (svg0.empty()) return;
  const { P, rng, frame, axis, label, $, sci } = NH;
  const W = 760, H = 380;
  const p1S = $("sc-p1"), p2S = $("sc-p2"), nS = $("sc-n"), cS = $("sc-corr");
  let seed = 11;

  /* one AND-gate multiply: two N-bit streams, bit = 1 with probability p; shared source = correlated streams */
  function once(p1, p2, N, shared, r) {
    let ones = 0; const a = [], b = [], z = [];
    for (let i = 0; i < N; i++) {
      const u = r(), v = shared ? u : r();
      const x = u < p1 ? 1 : 0, y = v < p2 ? 1 : 0;
      if (i < 64) { a.push(x); b.push(y); z.push(x & y); }
      ones += x & y;
    }
    return { est: ones / N, a, b, z };
  }
  function draw() {
    const p1 = +p1S.value, p2 = +p2S.value, N = Math.pow(2, +nS.value), sh = cS.value === "shared";
    $("sc-p1-out").textContent = p1.toFixed(2); $("sc-p2-out").textContent = p2.toFixed(2); $("sc-n-out").textContent = N;
    const r = rng(seed);
    const run = once(p1, p2, N, sh, r);
    const svg = frame("#sc-svg", W, H);
    const rows = [["A  (p = " + p1.toFixed(2) + ")", run.a, P.A], ["B  (p = " + p2.toFixed(2) + ")", run.b, P.B], ["A AND B", run.z, P.good]];
    rows.forEach((rw, k) => {
      label(svg, 10, 34 + k * 30, rw[0], { size: 10.5, fill: rw[2] });
      rw[1].forEach((bit, i) => svg.append("rect").attr("x", 120 + i * 9.6).attr("y", 22 + k * 30).attr("width", 8).attr("height", 16)
        .attr("rx", 1.5).attr("fill", bit ? rw[2] : "#1e222d").attr("stroke", P.line));
    });
    label(svg, 120, 14, "first " + Math.min(64, N) + " bits of each " + N + "-bit stream (a 1 is a filled cell)", { fill: P.ink });

    /* error vs stream length, averaged over 150 trials per length */
    const Ns = d3.range(3, 15).map(k => Math.pow(2, k));
    const r2 = rng(seed + 99);
    const exact = p1 * p2;
    const curve = Ns.map(n => {
      let s = 0; for (let t = 0; t < 150; t++) { const e = once(p1, p2, n, sh, r2).est - exact; s += e * e; }
      return { n, rms: Math.max(Math.sqrt(s / 150), 1e-5), th: Math.max(Math.sqrt(exact * (1 - exact) / n), 1e-5) };
    });
    const x = d3.scaleLog().base(2).domain([8, 16384]).range([70, 560]);
    const y = d3.scaleLog().domain([1e-3, 0.7]).range([350, 130]).clamp(true);
    axis(svg, d3.axisBottom(x).tickValues(d3.range(3, 15).map(k => Math.pow(2, k))).tickFormat(d3.format("d")), 0, 350);
    axis(svg, d3.axisLeft(y).ticks(4, "~g"), 70, 0);
    svg.append("path").attr("d", d3.line().x(d => x(d.n)).y(d => y(d.th))(curve)).attr("fill", "none").attr("stroke", P.muted).attr("stroke-dasharray", "4 3");
    svg.append("path").attr("d", d3.line().x(d => x(d.n)).y(d => y(d.rms))(curve)).attr("fill", "none").attr("stroke", P.good).attr("stroke-width", 2);
    svg.append("line").attr("x1", x(N)).attr("x2", x(N)).attr("y1", 130).attr("y2", 350).attr("stroke", P.ink).attr("opacity", 0.3);
    [4, 6, 8].forEach(b => {                               // the error a b-bit binary number would have
      const e = Math.pow(2, -b) / Math.sqrt(12);
      svg.append("line").attr("x1", 70).attr("x2", 560).attr("y1", y(e)).attr("y2", y(e)).attr("stroke", P.line).attr("stroke-dasharray", "2 4");
      label(svg, 556, y(e) - 3, b + "-bit binary", { anchor: "end", size: 9.5 });
    });
    label(svg, 70, 122, "RMS error of p̂ vs stream length (green); √(p(1−p)/N) for independent streams (dashed)", { fill: P.ink });
    label(svg, 315, 376, "stream length N (clock cycles)", { anchor: "middle" });

    const err = run.est - exact;
    const side = [["exact p₁·p₂", exact.toFixed(4)], ["AND-gate estimate", run.est.toFixed(4)], ["error", (err >= 0 ? "+" : "") + err.toFixed(4)],
      ["min(p₁, p₂)", Math.min(p1, p2).toFixed(4)], ["hardware", "1 AND gate + counter"]];
    side.forEach((s, i) => { label(svg, 590, 150 + i * 38, s[0], { size: 10 }); label(svg, 590, 165 + i * 38, s[1], { size: 12.5, fill: P.ink, bold: i < 3 }); });
    $("sc-readout").innerHTML = "N = " + N + " cycles · estimate <b>" + run.est.toFixed(4) + "</b> vs exact <b>" + exact.toFixed(4) + "</b>" +
      (sh ? " · <b style='color:" + P.bad + "'>correlated streams: AND computes min(p₁, p₂) = " + Math.min(p1, p2).toFixed(2) + ", not the product</b>"
          : " · halving the error costs 4× the cycles; matching 8-bit binary accuracy needs N ≈ " + sci(Math.ceil(exact * (1 - exact) * 12 * 65536)) + " cycles");
  }
  [p1S, p2S, nS, cS].forEach(el => el.addEventListener("input", draw));
  $("sc-new").addEventListener("click", () => { seed = (seed * 2654435761 + 7) >>> 0; draw(); });
  draw();
})();
