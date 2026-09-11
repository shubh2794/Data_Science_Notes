/* attention.viz.js — the visualizations on deep-learning/attention.html
   (Deep Learning · part 5). Loaded after ../data.js → ../notes.js → dl-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the
   page, so a section can be reordered or removed without breaking the rest.

   Every numeric primitive — the row-wise softmax, the softmax Jacobian, the
   head split and merge, the masks, scaled dot-product attention, the whole
   multi-head block and its exact backward pass, the central-difference
   audit, the parameter and MAC counts, the KV-cache arithmetic and the
   decode arithmetic intensity — comes from dl-viz.js (DL.*). NOTHING here
   forks those. Read the ATTENTION LAYOUT LAW above DL.sdpa before touching
   any of it: one TOKEN per ROW, X is (n × d_model), heads split the LAST
   axis into ADJACENT blocks, masks are ADDITIVE and −Infinity.
   Page-specific numerics live in AT below.

     1  #bd-svg   part 4's curve, and the same measurement with every state kept
     2  #al-svg   one decode step: score, normalise, blend
     3  #hs-svg   soft against hard: the exact gradient against the sampled one
     4  #sf-svg   four score functions, cost and expressiveness
     5  #vr-svg   Var[qᵀk] = dₖ, measured
     6  #st-svg   the softmax without the scale
     7  #lk-svg   the differentiable dictionary lookup
     8  #sh-svg   every shape and every parameter of one block
     9  #sc-svg   self-attention against cross-attention
    10  #dw-svg   the mixing matrix: fixed, tied, or computed from the input
    11  #pe-svg   permutation equivariance, measured
    12  #hd-svg   heads specialise: three relations, trained live
    13  #ht-svg   the head-dimension trade
    14  #wo-svg   concat-and-project as a sum of per-head terms
    15  #ms-svg   the masks, drawn and combined
    16  #mc-svg   the mask constant: leaked mass, and the all-masked row
    17  #fd-svg   the backward pass, analytic against central differences
    18  #co-svg   the three MAC terms of one layer against n
    19  #cr-svg   the crossover, per configuration
    20  #mm-svg   activation memory: the n×n matrix
    21  #tm-svg   attention alone: token mixing with no per-token nonlinearity
    22  #rc-svg   rank collapse with depth, four wirings
    23  #ec-svg   entropy: dilution with length, collapse with logit scale
    24  #kv-svg   the KV cache: recompute against cache
    25  #km-svg   KV-cache memory arithmetic
    26  #ai-svg   arithmetic intensity and the roofline
    27  #gq-svg   MHA → GQA → MQA, and the cache computed
    28  #ml-svg   MLA: low-rank compression of the cache
    29  #sw-svg   sliding windows, dilation and sinks
    30  #la-svg   linear attention: the associativity reordering
    31  #fa-svg   FlashAttention: HBM traffic against block size
    32  #lc-svg   long context: p(right key) against n, and the ln n margin
    33  #ih-svg   the induction circuit, wired by hand
    34  #tk-svg   three token-mixing strategies

   Every number these print is recomputed from the data they draw.          */

/* ══════════ page-local helpers (deliberately NOT in dl-viz.js) ══════════ */
const AT = (function () {

  /* a square matrix with a prescribed spectral radius, seeded */
  function mat(n, seed, rho) {
    const r = DL.rng(seed || 1), A = DL.zeros2(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = DL.randn(r) / Math.sqrt(n);
    return rho ? DL.scaleToRho(A, rho) : A;
  }
  /* a (m × n) seeded Gaussian, scaled */
  function gauss(m, n, seed, s) {
    const r = DL.rng(seed || 2), A = DL.zeros2(m, n), sc = s === undefined ? 1 : s;
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) A[i][j] = DL.randn(r) * sc;
    return A;
  }
  function rowNorm(v) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); }
  function unit(v) { const n = rowNorm(v) || 1; return v.map(x => x / n); }

  /* small titles and notes, matching the other parts of the series */
  function title(g, x, y, t, col) {
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", 11.5)
      .attr("font-weight", 600).attr("fill", col || DC.ink).text(t);
  }
  function note(g, x, y, t, col, size) {
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", size || 10)
      .attr("fill", col || DC.muted).text(t);
  }
  /* a heat-map of a matrix, returning the group so the caller can annotate */
  function heat(g, M, x, y, cw, scale, opt) {
    const o = Object.assign({ stroke: null, maxCell: 26 }, opt || {});
    const n = M.length, m = M[0].length;
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    const data = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) data.push([j, i, M[i][j]]);
    gg.selectAll("rect").data(data).join("rect")
      .attr("x", d => d[0] * cw).attr("y", d => d[1] * cw)
      .attr("width", cw + 0.5).attr("height", cw + 0.5)
      .attr("shape-rendering", "crispEdges")
      .attr("fill", d => scale(d[2]))
      .attr("stroke", o.stroke).attr("stroke-width", o.stroke ? 0.4 : 0);
    return gg;
  }
  /* a vector drawn as a strip of cells */
  function strip(g, v, x, y, cw, scale, opt) {
    const o = Object.assign({ vertical: false, op: 1, stroke: null }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.selectAll("rect").data(v.map((val, i) => [i, val])).join("rect")
      .attr("x", d => o.vertical ? 0 : d[0] * cw).attr("y", d => o.vertical ? d[0] * cw : 0)
      .attr("width", cw + 0.4).attr("height", cw + 0.4)
      .attr("shape-rendering", "crispEdges")
      .attr("fill", d => scale(d[1])).attr("fill-opacity", o.op)
      .attr("stroke", o.stroke).attr("stroke-width", o.stroke ? 0.5 : 0);
    return gg;
  }
  /* a histogram: returns {bins, lo, hi, w} */
  function hist(vals, nb) {
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const w = (hi - lo) / nb || 1, bins = new Array(nb).fill(0);
    vals.forEach(v => { let k = Math.floor((v - lo) / w); if (k >= nb) k = nb - 1; if (k < 0) k = 0; bins[k]++; });
    return { bins: bins, lo: lo, hi: hi, w: w };
  }
  const CO = { soft: DC.accent, hard: DC.a2, exact: DC.good, bad: DC.bad, alt: DC.violet, third: DC.teal };
  return { mat: mat, gauss: gauss, unit: unit, rowNorm: rowNorm, title: title, note: note,
           heat: heat, strip: strip, hist: hist, CO: CO };
})();

/* ═════════ 1 · #bd-svg — the bottleneck, and the same measurement with
   every encoder state kept. Both curves are EXACT closed forms computed the
   same way: R²(k) = tr(Mₖ Σ⁺ Mₖᵀ)/dₓ for the linear read-out available to
   the reader. The final-state reader has Mₖ = U W⁽ᴸ⁻¹⁻ᵏ⁾ and Σ = ∑ⱼ MⱼᵀMⱼ.
   The all-states reader may use hₖ and hₖ₋₁, and hₖ − hₖ₋₁W = xₖU exactly,
   so its Mₖ is U itself with Σ = UᵀU — which is why it does not depend on
   the position, on L, or on W.  ═══════════════════════════════════════════ */
(function () {
  const svg = d3.select("#bd-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);

  function r2FromBlocks(M, dx) {          /* M: array of (dx × dh) blocks */
    const dh = M[0][0].length, S = DL.zeros2(dh, dh);
    M.forEach(Mk => {
      const P = DL.matmul(DL.transpose(Mk), Mk);
      for (let i = 0; i < dh; i++) for (let j = 0; j < dh; j++) S[i][j] += P[i][j];
    });
    const Si = DL.pinvSym(S, 1e-12);
    return M.map(Mk => {
      const P = DL.matmul(DL.matmul(Mk, Si), DL.transpose(Mk));
      let t = 0; for (let i = 0; i < dx; i++) t += P[i][i];
      return DL.clamp(t / dx, 0, 1);
    });
  }
  function curves(dx, dh, rho, L, seed) {
    const U = AT.gauss(dx, dh, seed, 1 / Math.sqrt(dx));
    const Wm = AT.mat(dh, (seed || 12) + 1, rho);
    const Mfin = [];
    for (let k = 0; k < L; k++) Mfin.push(DL.matmul(U, DL.matPow(Wm, L - 1 - k)));
    const fin = r2FromBlocks(Mfin, dx);
    /* the all-states reader: the only block that matters is U itself. */
    const all = r2FromBlocks([U], dx);
    return { fin: fin, all: new Array(L).fill(all[0]) };
  }

  function draw() {
    const L = +El("bd-L").value, dh = +El("bd-dh").value, dx = +El("bd-dx").value,
      rho = +El("bd-rho").value, both = El("bd-both").checked;
    El("bd-Lv").textContent = L; El("bd-dhv").textContent = dh;
    El("bd-dxv").textContent = dx; El("bd-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 52, r: 14, t: 26, b: 46 }), g = f.g;
    const c = curves(dx, dh, rho, L, 12);
    const mFin = d3.mean(c.fin), mAll = d3.mean(c.all);
    const bFin = Math.min(1, dh / (L * dx)), bAll = Math.min(1, dh / dx);

    /* left — per-position */
    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "what fraction of source position k survives");
    const lo = Math.max(1e-8, Math.min(...c.fin.filter(v => v > 0).concat([bAll])) / 3);
    const x = d3.scaleLinear().domain([1, Math.max(2, L)]).range([0, gw]);
    const y = d3.scaleLog().domain([lo, 1.5]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 6, "source position k");
    DL.axisL(g, y, 5, "R²", v => DL.fmtE(v, 0));
    if (both) DL.curve(g, c.fin.map((v, i) => [x(i + 1), y(Math.max(v, 1e-9))]), { stroke: DC.a2, w: 2 });
    DL.curve(g, c.all.map((v, i) => [x(i + 1), y(Math.max(v, 1e-9))]), { stroke: DC.accent, w: 2.4 });
    const items = [{ label: "every state kept — attention's reader", color: DC.accent }];
    if (both) items.push({ label: "the final state only — part 4's reader", color: DC.a2 });
    DL.legend(g, items, 2, gh + 34, { vertical: true, gap: 13, font: 9.5 });
    if (dx > dh) AT.note(g, 4, 12, "dₓ > dₕ: even keeping every state loses dₕ/dₓ", DC.bad, 9.5);

    /* right — mean against L */
    const rx = gw + 66, rw = f.iw - rx;
    AT.title(g, rx, -10, "mean fraction against source length");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const Ls = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
    const pts = Ls.map(l => { const cc = curves(dx, dh, rho, l, 12); return { l: l, f: d3.mean(cc.fin), a: cc.all[0] }; });
    const ylo = Math.max(1e-4, Math.min(...pts.map(p => p.f).filter(v => v > 0)) / 2);
    const xl = d3.scaleLog().domain([2, 64]).range([0, rw - 12]);
    const yl = d3.scaleLog().domain([ylo, 1.8]).range([gh, 0]).clamp(true);
    DL.gridY(gg, yl, rw - 12, 4);
    DL.axisB(gg, xl, gh, 4, "source length L", d3.format("d"));
    DL.axisL(gg, yl, 4, "mean R²", v => DL.fmtE(v, 0));
    DL.curve(gg, Ls.map(l => [xl(l), yl(Math.min(1, dh / (l * dx)))]), { stroke: DC.a2, w: 1.3, dash: "4 3" });
    DL.curve(gg, Ls.map(l => [xl(l), yl(bAll)]), { stroke: DC.accent, w: 1.3, dash: "4 3" });
    if (both) DL.curve(gg, pts.map(p => [xl(p.l), yl(Math.max(p.f, 1e-9))]), { stroke: DC.a2, w: 2 });
    DL.curve(gg, pts.map(p => [xl(p.l), yl(p.a)]), { stroke: DC.accent, w: 2.4 });
    pts.forEach(p => {
      gg.append("circle").attr("cx", xl(p.l)).attr("cy", yl(p.a)).attr("r", 2.3).attr("fill", DC.accent);
      if (both) gg.append("circle").attr("cx", xl(p.l)).attr("cy", yl(Math.max(p.f, 1e-9))).attr("r", 2.3).attr("fill", DC.a2);
    });
    gg.append("line").attr("x1", xl(DL.clamp(L, 2, 64))).attr("x2", xl(DL.clamp(L, 2, 64)))
      .attr("y1", 0).attr("y2", gh).attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.55);
    DL.legend(gg, [{ label: "bound dₕ/(L·dₓ) — falls with L", color: DC.a2, dash: "4 3" },
                   { label: "bound dₕ/dₓ — no L in it at all", color: DC.accent, dash: "4 3" }],
      2, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    const kv = DL.kv(g, 0, gh + 74, { keyW: 268, size: 10.5, lead: 14.5 });
    kv("mean R², every state kept", DL.fmt(mAll, 6), DC.accent);
    kv("   its exact bound  min(1, dₕ/dₓ)", DL.fmt(bAll, 6), DC.accent);
    kv("mean R², final state only", DL.fmt(mFin, 6), DC.a2);
    kv("   its exact bound  min(1, dₕ/(L·dₓ))", DL.fmt(bFin, 6), DC.a2);
    kv("ratio between the two readers", DL.fmt(mAll / Math.max(1e-12, mFin), 2) + "×", DC.good, true);
    kv("R² at the FIRST position, final state only", DL.fmtE(c.fin[0], 3), c.fin[0] < 0.01 ? DC.bad : DC.ink);
    kv("R² at the FIRST position, every state kept", DL.fmt(c.all[0], 6), DC.good);

    El("bd-readout").innerHTML =
      `dₓ = ${dx}, dₕ = ${dh}, L = ${L}, ρ(W) = ${DL.fmt(rho, 2)} · keeping every encoder state, the mean recoverable ` +
      `fraction is <b>${DL.fmt(mAll, 6)}</b> against its bound min(1, dₕ/dₓ) = <b>${DL.fmt(bAll, 6)}</b>; ` +
      `keeping only the final state it is <b>${DL.fmt(mFin, 6)}</b> against min(1, dₕ/(L·dₓ)) = <b>${DL.fmt(bFin, 6)}</b> — ` +
      `a factor of <b>${DL.fmt(mAll / Math.max(1e-12, mFin), 2)}×</b>. At the FIRST source position the two readers differ by ` +
      `<b>${c.fin[0] > 0 ? DL.fmtE(c.all[0] / c.fin[0], 2) : "∞"}</b>. ` +
      (dx > dh ? `<b>dₓ > dₕ</b>, so even the all-states bound is below 1: the encoder is too narrow to be injective in one step.`
               : `The length L does not appear in the all-states bound at all — that cancellation is what attention buys.`) +
      (mFin < 0.98 * bFin ? ` <b>The final-state point sits below its rank bound here</b>: with ρ(W) = ${DL.fmt(rho, 2)} < 1 the blocks ` +
        `U·W⁽ᴸ⁻¹⁻ᵏ⁾ for early positions have decayed below the pseudo-inverse's tolerance, so the linear reader cannot amplify them ` +
        `back — the bound counts rank in exact arithmetic, and this is the vanishing-signal version of the same story.` : ``);
  }
  ["bd-L", "bd-dh", "bd-dx", "bd-rho"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#bd-both").on("change", draw);
  draw();
})();

/* ═════════ 2 · #al-svg — one decode step: score, normalise, blend.
   The source states carry latent concept vectors; the decoder query at step
   i is a noisy mixture centred on the position it is currently emitting, so
   there IS a latent correspondence for a scoring function to find. The dot
   score finds it directly. The additive score is vᵀtanh(sW_a + hU_a + b)
   with W_a, U_a, b random and the OUTPUT LAYER v fitted by least squares to
   a target that is high on the corresponding position and zero elsewhere —
   which is what training does to it, and the fit residual is printed so the
   figure is not pretending the fit is perfect. Whether an additive net can
   represent a DOT PRODUCT is a different question, measured in figure 4.
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#al-svg"); if (svg.empty()) return;
  const W = 760, H = 430, D = 6, DA = 24, T = 8;
  const El = id => document.getElementById(id);

  function build(L) {
    const r = DL.rng(31), Hs = [], Q = [], conc = [];
    for (let j = 0; j < L; j++) { const v = []; for (let c = 0; c < D; c++) v.push(DL.randn(r)); conc.push(AT.unit(v)); }
    for (let j = 0; j < L; j++) Hs.push(conc[j].slice());
    /* the latent alignment: mostly monotone, with two deliberate swaps so the
       band is not a straight line — which is what real word order does. */
    const map = [];
    for (let i = 0; i < T; i++) map.push(Math.min(L - 1, Math.round(i * (L - 1) / (T - 1))));
    if (L >= 6) { const a = map[2]; map[2] = map[3]; map[3] = a; }
    if (L >= 9) { const a = map[5]; map[5] = map[6]; map[6] = a; }
    for (let i = 0; i < T; i++) {
      const v = conc[map[i]].map(x => 0.9 * x);
      for (let c = 0; c < D; c++) v[c] += 0.30 * DL.randn(r);
      Q.push(AT.unit(v));
    }
    return { Hs: Hs, Q: Q, map: map };
  }
  function scores(B, L, kind, t) {
    if (kind === "dot") {
      const S = DL.zeros2(T, L);
      for (let i = 0; i < T; i++) for (let j = 0; j < L; j++) {
        let s = 0; for (let c = 0; c < D; c++) s += B.Q[i][c] * B.Hs[j][c];
        S[i][j] = s * t * 4;
      }
      return { S: S, resid: null };
    }
    const Wa = AT.gauss(D, DA, 77, 2 / Math.sqrt(D)), Ua = AT.gauss(D, DA, 78, 2 / Math.sqrt(D));
    const rb = DL.rng(79), bias = []; for (let a = 0; a < DA; a++) bias.push(DL.randn(rb) * 2);
    const Phi = [], y = [];
    for (let i = 0; i < T; i++) for (let j = 0; j < L; j++) {
      const row = [];
      for (let a = 0; a < DA; a++) {
        let z = bias[a];
        for (let c = 0; c < D; c++) z += B.Q[i][c] * Wa[c][a] + B.Hs[j][c] * Ua[c][a];
        row.push(Math.tanh(z));
      }
      row.push(1);                                  /* an output bias column */
      Phi.push(row); y.push([j === B.map[i] ? 4 : 0]);
    }
    const v = DL.ridge(Phi, y, 1e-4);
    const P = DL.matmul(Phi, v);
    let mu = 0; for (let k = 0; k < y.length; k++) mu += y[k][0] / y.length;
    let sse = 0, sst = 0;
    for (let k = 0; k < y.length; k++) { sse += (P[k][0] - y[k][0]) ** 2; sst += (y[k][0] - mu) ** 2; }
    const S = DL.zeros2(T, L);
    for (let i = 0, k = 0; i < T; i++) for (let j = 0; j < L; j++, k++) S[i][j] = P[k][0] * t;
    return { S: S, resid: Math.sqrt(sse / Math.max(1e-12, sst)) };
  }

  function draw() {
    const i0 = +El("al-i").value - 1, L = +El("al-L").value, t = +El("al-t").value, kind = El("al-s").value;
    El("al-iv").textContent = i0 + 1; El("al-Lv").textContent = L; El("al-tv").textContent = DL.fmt(t, 1);
    const f = DL.frame(svg, W, H, { l: 44, r: 14, t: 24, b: 30 }), g = f.g;
    const B = build(L), sc = scores(B, L, kind, t);
    const A = DL.softmaxRows(sc.S);
    const a = A[i0];
    const cell = d3.scaleSequential(d3.interpolateRdBu).domain([1.1, -1.1]);
    const heatS = d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...A.map(r => Math.max(...r)))]);

    /* -- upper left: the source states and their weights -- */
    AT.title(g, 0, -8, "the L source states, and the weight step " + (i0 + 1) + " gives each");
    const cw = 13, y0 = 10, rowH = 17;
    for (let j = 0; j < L; j++) {
      const yy = y0 + j * rowH;
      AT.note(g, -14, yy + 10, "h" + (j + 1), DC.muted, 9);
      AT.strip(g, B.Hs[j], 8, yy, cw, cell, { stroke: "#0f1117" });
      const bw = 150 * a[j];
      g.append("rect").attr("x", 8 + D * cw + 12).attr("y", yy + 1).attr("width", Math.max(0.6, bw))
        .attr("height", cw - 1).attr("fill", DC.accent).attr("fill-opacity", 0.85).attr("rx", 1.5);
      g.append("text").attr("x", 8 + D * cw + 18 + Math.max(0.6, bw)).attr("y", yy + 11)
        .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", a[j] > 0.2 ? DC.accent : DC.muted).text(DL.fmt(a[j], 3));
      if (j === B.map[i0]) g.append("text").attr("x", 8 + D * cw + 200).attr("y", yy + 11)
        .attr("font-size", 9).attr("fill", DC.good).text("← latent match");
    }
    /* -- upper right: the query and the context vector -- */
    const rx = 340;
    AT.title(g, rx, -8, "query, and the blend it produces");
    AT.note(g, rx, y0 + 11, "s" + (i0 + 1), DC.muted, 9);
    AT.strip(g, B.Q[i0], rx + 24, y0, cw, cell, { stroke: "#0f1117" });
    let yy = y0 + 30;
    AT.note(g, rx, yy + 11, "αⱼ·hⱼ", DC.muted, 9);
    for (let j = 0; j < L; j++) {
      AT.strip(g, B.Hs[j].map(v => v * a[j] * L * 0.5), rx + 24, yy, cw, cell, { op: 0.10 + 0.9 * a[j] });
    }
    const c = new Array(D).fill(0);
    for (let j = 0; j < L; j++) for (let cc = 0; cc < D; cc++) c[cc] += a[j] * B.Hs[j][cc];
    yy += 26;
    AT.note(g, rx, yy + 11, "c" + (i0 + 1), DC.a2, 9);
    AT.strip(g, c, rx + 24, yy, cw, cell, { stroke: DC.a2 });
    DL.arrow(g, rx + 24 + D * cw / 2, yy - 8, rx + 24 + D * cw / 2, yy - 1, { color: DC.a2, head: 5 });
    AT.note(g, rx + 24 + D * cw + 12, yy + 11, "‖c‖ = " + DL.fmt(AT.rowNorm(c), 3) +
      "   (mean ‖h‖ = " + DL.fmt(d3.mean(B.Hs.map(AT.rowNorm)), 3) + ")", DC.muted, 9.5);
    AT.note(g, rx, yy + 34, "a convex combination stays inside the states' hull:", DC.muted, 9.5);
    AT.note(g, rx, yy + 46, "its norm cannot grow with L, whatever L is.", DC.muted, 9.5);

    /* -- lower: the whole alignment matrix -- */
    const my = 200, mcw = Math.min(22, 300 / L);
    AT.title(g, 0, my - 8, "the alignment matrix α for the whole decode");
    AT.heat(g, A, 30, my, mcw, heatS, { stroke: "#0f1117" });
    g.append("rect").attr("x", 30 - 1).attr("y", my + i0 * mcw - 1).attr("width", L * mcw + 2)
      .attr("height", mcw + 2).attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 1.8);
    for (let i = 0; i < T; i++) AT.note(g, 4, my + i * mcw + mcw * 0.72, "y" + (i + 1), i === i0 ? DC.a2 : DC.muted, 9);
    AT.note(g, 30, my + T * mcw + 13, "source position →", DC.muted, 9.5);
    for (let i = 0; i < T; i++) g.append("circle").attr("cx", 30 + (B.map[i] + 0.5) * mcw)
      .attr("cy", my + (i + 0.5) * mcw).attr("r", 2).attr("fill", "none")
      .attr("stroke", DC.good).attr("stroke-width", 1.2);
    AT.note(g, 30, my + T * mcw + 26, "small circles: the latent correspondence the states were built with", DC.good, 9);

    const Hn = DL.entropyOf(a), Hmax = Math.log(L);
    const kv = DL.kv(g, 360, my + 6, { keyW: 232, size: 10.5, lead: 15 });
    kv("∑ⱼ αᵢⱼ", DL.fmt(a.reduce((x, y2) => x + y2, 0), 10), DC.good);
    kv("largest weight", DL.fmt(Math.max(...a), 4), DC.accent);
    kv("entropy H(α) in nats", DL.fmt(Hn, 4));
    kv("its maximum, ln L", DL.fmt(Hmax, 4), DC.muted);
    kv("effective #positions read, eᴴ", DL.fmt(Math.exp(Hn), 3), DC.a2, true);
    kv("argmax position", String(a.indexOf(Math.max(...a)) + 1) + " (latent " + (B.map[i0] + 1) + ")",
      a.indexOf(Math.max(...a)) === B.map[i0] ? DC.good : DC.bad);
    let nOK = 0; for (let i = 0; i < T; i++) if (A[i].indexOf(Math.max(...A[i])) === B.map[i]) nOK++;
    kv("steps whose argmax is the latent position", nOK + " of " + T, nOK === T ? DC.good : DC.a2, true);
    if (kind === "add") kv("output-layer fit residual (0/4 target)", DL.fmt(sc.resid, 4), sc.resid < 0.6 ? DC.good : DC.a2);

    El("al-readout").innerHTML =
      `step ${i0 + 1} of ${T}, L = ${L}, ${kind === "add" ? "additive" : "dot-product"} score · the weights sum to ` +
      `<b>${DL.fmt(a.reduce((x, y2) => x + y2, 0), 10)}</b>, the largest is <b>${DL.fmt(Math.max(...a), 4)}</b>, ` +
      `the entropy is <b>${DL.fmt(Hn, 3)}</b> nats against a maximum of ln ${L} = <b>${DL.fmt(Hmax, 3)}</b>, ` +
      `so the step is effectively reading <b>${DL.fmt(Math.exp(Hn), 2)}</b> of the ${L} source positions. ` +
      `The context vector has norm <b>${DL.fmt(AT.rowNorm(c), 3)}</b> against a mean state norm of ` +
      `<b>${DL.fmt(d3.mean(B.Hs.map(AT.rowNorm)), 3)}</b> — a convex combination cannot exceed the largest, at any L. ` +
      `The argmax lands on the latent correspondence for <b>${nOK} of ${T}</b> decode steps. ` +
      (kind === "add" ? `The additive net's output layer — ${DA} units, fitted by least squares against a deliberately hard 0/4 target, ` +
        `relative residual <b>${DL.fmt(sc.resid, 4)}</b> — only has to get the ORDER of the scores right, because the softmax is ` +
        `invariant to anything the fit gets wrong uniformly across a row.` : ``);
  }
  ["al-i", "al-L", "al-t"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#al-s").on("change", draw);
  draw();
})();

/* ═════════ 3 · #hs-svg — soft against hard.
   The objective is J = ∑ⱼ αⱼ vⱼ, identical under both readings. The SOFT
   gradient ∂J/∂eₖ = αₖ(vₖ − J) is exact. The HARD reading samples j ∼ α and
   uses the score-function estimator vⱼ(δⱼₖ − αₖ), optionally with the
   baseline b = J. Both estimate the SAME number; the figure measures the
   spread of the second.  ═══════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#hs-svg"); if (svg.empty()) return;
  const W = 760, H = 400, TRIALS = 2000;
  const El = id => document.getElementById(id);

  function setup(L, sd, seed) {
    const r = DL.rng(seed || 5), e = [], v = [];
    for (let j = 0; j < L; j++) { e.push(DL.randn(r) * sd); v.push(DL.randn(r)); }
    const a = DL.softmax(e);
    let J = 0; for (let j = 0; j < L; j++) J += a[j] * v[j];
    return { e: e, v: v, a: a, J: J };
  }
  function sampleIdx(a, u) { let s = 0; for (let j = 0; j < a.length; j++) { s += a[j]; if (u < s) return j; } return a.length - 1; }
  function estimates(S, k, nSamp, base, seed) {
    const r = DL.rng(seed || 99), out = new Array(TRIALS);
    const b = base ? S.J : 0;
    for (let t = 0; t < TRIALS; t++) {
      let acc = 0;
      for (let s = 0; s < nSamp; s++) {
        const j = sampleIdx(S.a, r());
        acc += (S.v[j] - b) * ((j === k ? 1 : 0) - S.a[k]);
      }
      out[t] = acc / nSamp;
    }
    return out;
  }
  function sd(arr) { const m = d3.mean(arr); let s = 0; arr.forEach(x => s += (x - m) * (x - m)); return Math.sqrt(s / (arr.length - 1)); }

  function draw() {
    const L = +El("hs-L").value, nS = +El("hs-S").value, spread = +El("hs-sd").value, base = El("hs-base").checked;
    El("hs-Lv").textContent = L; El("hs-Sv").textContent = nS; El("hs-sdv").textContent = DL.fmt(spread, 1);
    const f = DL.frame(svg, W, H, { l: 52, r: 16, t: 26, b: 46 }), g = f.g;
    const S = setup(L, spread, 5);
    const k = S.a.indexOf(Math.max(...S.a));               /* the heaviest coordinate */
    const gExact = S.a[k] * (S.v[k] - S.J);
    const est = estimates(S, k, nS, base, 99);
    const mEst = d3.mean(est), sEst = sd(est);

    /* -- left: the histogram -- */
    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "2000 independent estimates of ∂J/∂e" + (k + 1));
    const hh = AT.hist(est, 42);
    const x = d3.scaleLinear().domain([Math.min(hh.lo, gExact) - hh.w, Math.max(hh.hi, gExact) + hh.w]).range([0, gw]);
    const y = d3.scaleLinear().domain([0, Math.max(...hh.bins) * 1.12]).range([gh, 0]);
    DL.gridY(g, y, gw, 4);
    DL.axisB(g, x, gh, 5, "estimated gradient");
    DL.axisL(g, y, 4, "count");
    g.selectAll("rect.hb").data(hh.bins.map((c, i) => [i, c])).join("rect").attr("class", "hb")
      .attr("x", d => x(hh.lo + d[0] * hh.w)).attr("y", d => y(d[1]))
      .attr("width", Math.max(1, x(hh.lo + hh.w) - x(hh.lo) - 0.6)).attr("height", d => gh - y(d[1]))
      .attr("fill", DC.a2).attr("fill-opacity", 0.55);
    g.append("line").attr("x1", x(gExact)).attr("x2", x(gExact)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.accent).attr("stroke-width", 2);
    AT.note(g, x(gExact) + 5, 12, "the exact soft gradient", DC.accent, 9.5);
    g.append("line").attr("x1", x(mEst)).attr("x2", x(mEst)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.good).attr("stroke-dasharray", "3 3");
    AT.note(g, x(mEst) + 5, 26, "mean of the sampled estimator", DC.good, 9.5);
    DL.legend(g, [{ label: "sampled (score-function) estimator", color: DC.a2 },
                  { label: "exact soft gradient — a single point", color: DC.accent }],
      2, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    /* -- right: relative spread of the WHOLE gradient vector against L --
       For one sample j ~ α the estimator of the full L-vector is
       ĝₖ = (vⱼ − b)(δⱼₖ − αₖ), so its total variance is computed exactly:
       E‖ĝ‖² − ‖g‖² = ∑ⱼ αⱼ (vⱼ − b)² ∑ₖ(δⱼₖ − αₖ)² − ‖g‖², divided by the
       number of samples averaged. It is reported RELATIVE to ‖g‖, and the
       geometric mean over 24 random score/value draws is plotted, so the
       curve is a law and not one draw's luck. ‖g‖ shrinks like 1/√L while
       the total sd stays O(1), which is where the √L comes from.        */
    const rx = gw + 66, rw = f.iw - rx;
    AT.title(g, rx, -10, "relative spread of the sampled gradient VECTOR, against L");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const Ls = [2, 4, 8, 16, 32, 64, 128, 256];
    const relSpread = (l, seed) => {
      const s2 = setup(l, spread, seed), b = base ? s2.J : 0;
      let g2 = 0, sa2 = 0, tot = 0;
      s2.a.forEach(x => sa2 += x * x);
      for (let k = 0; k < l; k++) { const gk = s2.a[k] * (s2.v[k] - s2.J); g2 += gk * gk; }
      for (let j = 0; j < l; j++) tot += s2.a[j] * (s2.v[j] - b) ** 2 * (1 - 2 * s2.a[j] + sa2);
      return Math.sqrt(Math.max(0, tot - g2) / nS) / Math.sqrt(Math.max(1e-300, g2));
    };
    const pts = Ls.map(l => {
      let acc = 0; const SEEDS = 24;
      for (let s = 0; s < SEEDS; s++) acc += Math.log(Math.max(1e-12, relSpread(l, 5 + s)));
      return { l: l, s: Math.exp(acc / SEEDS) };
    });
    const ref8 = pts[2].s;                                   /* √L reference through the L = 8 point */
    const xl = d3.scaleLog().domain([2, 256]).range([0, rw - 12]);
    const ylo = Math.max(1e-3, Math.min(...pts.map(p => p.s)) / 2), yhi = Math.max(...pts.map(p => p.s), ref8 * Math.sqrt(32)) * 2;
    const yl = d3.scaleLog().domain([ylo, yhi]).range([gh, 0]).clamp(true);
    DL.gridY(gg, yl, rw - 12, 4);
    DL.axisB(gg, xl, gh, 4, "number of candidates L", d3.format("d"));
    DL.axisL(gg, yl, 4, "sd‖ĝ − g‖ / ‖g‖", v => DL.fmtE(v, 0));
    DL.curve(gg, Ls.map(l => [xl(l), yl(ref8 * Math.sqrt(l / 8))]), { stroke: DC.muted, w: 1.2, dash: "4 3" });
    DL.curve(gg, pts.map(p => [xl(p.l), yl(p.s)]), { stroke: DC.a2, w: 2.2 });
    pts.forEach(p => gg.append("circle").attr("cx", xl(p.l)).attr("cy", yl(p.s)).attr("r", 2.6).attr("fill", DC.a2));
    gg.append("line").attr("x1", 0).attr("x2", rw - 12).attr("y1", gh).attr("y2", gh)
      .attr("stroke", DC.accent).attr("stroke-width", 2.4);
    AT.note(gg, 4, gh - 6, "the exact gradient has sd = 0, at every L", DC.accent, 9.5);
    AT.note(gg, 4, 12, "dashed: slope ½ — spread ∝ √L", DC.muted, 9);
    gg.append("line").attr("x1", xl(DL.clamp(L, 2, 256))).attr("x2", xl(DL.clamp(L, 2, 256)))
      .attr("y1", 0).attr("y2", gh).attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.55);
    /* the fitted slope of log(spread) against log L over L ≥ 8 */
    const fitPts = pts.filter(p => p.l >= 8);
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    fitPts.forEach(p => { const X = Math.log(p.l), Y = Math.log(p.s); sx += X; sy += Y; sxx += X * X; sxy += X * Y; });
    const nf = fitPts.length, slope = (nf * sxy - sx * sy) / (nf * sxx - sx * sx);

    const need = Math.ceil(Math.pow(sEst * Math.sqrt(nS) / (0.1 * Math.abs(gExact)), 2));
    const kv = DL.kv(g, 0, gh + 74, { keyW: 268, size: 10.5, lead: 14.5 });
    kv("exact soft gradient ∂J/∂e" + (k + 1), DL.fmt(gExact, 6), DC.accent, true);
    kv("mean of the sampled estimator", DL.fmt(mEst, 6), DC.good);
    kv("   bias  (mean − exact)", DL.fmtE(mEst - gExact, 2), Math.abs(mEst - gExact) < 3 * sEst / Math.sqrt(TRIALS) ? DC.good : DC.a2);
    kv("sd of the sampled estimator", DL.fmt(sEst, 6), DC.a2);
    kv("sd / |exact gradient|", DL.fmt(sEst / Math.max(1e-12, Math.abs(gExact)), 2) + "×", DC.bad, true);
    kv("samples needed for sd < 0.1·|g|", DL.commas(need), DC.bad);
    kv("baseline subtracted", base ? "yes, b = J" : "no", base ? DC.good : DC.bad);
    kv("relative spread of the whole gradient, L = 8 → 256", DL.fmt(pts[2].s, 2) + " → " + DL.fmt(pts[7].s, 2), DC.a2);
    kv("   fitted slope against L (½ = √L growth)", DL.fmt(slope, 2), Math.abs(slope - 0.5) < 0.15 ? DC.good : DC.a2);

    El("hs-readout").innerHTML =
      `L = ${L} candidates, ${nS} sample${nS > 1 ? "s" : ""} per estimate, ${base ? "with" : "without"} a baseline · ` +
      `the exact soft gradient is <b>${DL.fmt(gExact, 6)}</b>; the sampled estimator averages <b>${DL.fmt(mEst, 6)}</b> ` +
      `(bias <b>${DL.fmtE(mEst - gExact, 2)}</b>, within sampling error of zero — it is unbiased) with a standard deviation of ` +
      `<b>${DL.fmt(sEst, 6)}</b>, which is <b>${DL.fmt(sEst / Math.max(1e-12, Math.abs(gExact)), 2)}×</b> the quantity being ` +
      `estimated. Bringing that below a tenth would take <b>${DL.commas(need)}</b> samples per step. Over the whole gradient ` +
      `vector, the sampled estimator's spread relative to the exact gradient grows from <b>${DL.fmt(pts[2].s, 2)}×</b> at L = 8 to ` +
      `<b>${DL.fmt(pts[7].s, 2)}×</b> at L = 256 — a fitted slope of <b>${DL.fmt(slope, 2)}</b> against L on log axes` +
      (Math.abs(slope - 0.5) < 0.15 ? `, the √L law` : spread >= 2 ? `; with scores this spread the softmax is nearly one-hot, only a few candidates are ever sampled, and the √L growth is suppressed` : ``) +
      `. The soft gradient's spread is exactly zero because nothing is sampled.`;
  }
  ["hs-L", "hs-S", "hs-sd"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#hs-base").on("change", draw);
  draw();
})();

/* ═════════ 4 · #sf-svg — four score functions: what each can represent, and
   what each costs. The left panel fits ONLY the output layer v of an
   additive scorer vᵀtanh(qW_a + kU_a + b) by ridge regression, and reports
   the residual on HELD-OUT pairs. With b = 0 the scorer is an ODD function
   of (q,k) and the two even targets lie entirely outside its span, so the
   residual sits at 1.0 and rises with width. ═══════════════════════════════ */
(function () {
  const svg = d3.select("#sf-svg"); if (svg.empty()) return;
  const W = 760, H = 410, DAS = [2, 4, 8, 16, 32, 64];
  const El = id => document.getElementById(id);

  function pairs(N, d, seed) {
    const r = DL.rng(seed), Q = [], K = [];
    for (let i = 0; i < N; i++) {
      const a = [], b = [];
      for (let c = 0; c < d; c++) { a.push(DL.randn(r)); b.push(DL.randn(r)); }
      Q.push(AT.unit(a)); K.push(AT.unit(b));
    }
    return { Q: Q, K: K };
  }
  function target(P, d, kind, G) {
    return P.Q.map((q, i) => {
      const k = P.K[i];
      if (kind === "dot") { let t = 0; for (let c = 0; c < d; c++) t += q[c] * k[c]; return [t]; }
      if (kind === "bil") { let t = 0; for (let a = 0; a < d; a++) for (let c = 0; c < d; c++) t += q[a] * G[a][c] * k[c]; return [t]; }
      /* a deliberately ODD target: f(−q,−k) = −f(q,k) */
      let t = 0; for (let c = 0; c < d; c++) t += Math.tanh(2 * (q[c] + k[c])) * (c % 2 ? 1 : -1);
      return [t];
    });
  }
  function feats(P, d, da, Wa, Ua, b) {
    return P.Q.map((q, i) => {
      const k = P.K[i], row = [];
      for (let a = 0; a < da; a++) {
        let z = b ? b[a] : 0;
        for (let c = 0; c < d; c++) z += q[c] * Wa[c][a] + k[c] * Ua[c][a];
        row.push(Math.tanh(z));
      }
      row.push(1);
      return row;
    });
  }
  function residual(d, da, bs, kind, N, G) {
    const Wa = AT.gauss(d, da, 77, 2 / Math.sqrt(d)), Ua = AT.gauss(d, da, 78, 2 / Math.sqrt(d));
    let b = null;
    if (bs > 0) { const r = DL.rng(555); b = []; for (let a = 0; a < da; a++) b.push(DL.randn(r) * bs); }
    const Ptr = pairs(N, d, 101), Pte = pairs(1200, d, 202);
    const Phi = feats(Ptr, d, da, Wa, Ua, b), y = target(Ptr, d, kind, G);
    const v = DL.ridge(Phi, y, 1e-6);
    const P2 = feats(Pte, d, da, Wa, Ua, b), y2 = target(Pte, d, kind, G);
    const pr = DL.matmul(P2, v);
    let mu = 0; for (let i = 0; i < y2.length; i++) mu += y2[i][0] / y2.length;
    let sse = 0, sst = 0;
    for (let i = 0; i < y2.length; i++) { sse += (pr[i][0] - y2[i][0]) ** 2; sst += (y2[i][0] - mu) ** 2; }
    return Math.sqrt(sse / Math.max(1e-12, sst));
  }

  function draw() {
    const d = +El("sf-d").value, kind = El("sf-t").value, bs = +El("sf-b").value, N = +El("sf-n").value;
    El("sf-dv").textContent = d; El("sf-bv").textContent = DL.fmt(bs, 2);
    const f = DL.frame(svg, W, H, { l: 52, r: 16, t: 26, b: 46 }), g = f.g;
    const G = AT.gauss(d, d, 313, 1 / Math.sqrt(d));
    const noB = DAS.map(da => residual(d, da, 0, kind, N, G));
    const wiB = DAS.map(da => residual(d, da, bs, kind, N, G));

    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "held-out residual of an additive scorer fitting the target");
    const x = d3.scaleLog().domain([2, 64]).range([0, gw]);
    const y = d3.scaleLinear().domain([0, Math.max(1.35, ...noB, ...wiB) * 1.05]).range([gh, 0]);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "hidden units d_a", d3.format("d"));
    DL.axisL(g, y, 5, "‖a − target‖ / ‖target − mean‖");
    g.append("line").attr("x1", 0).attr("x2", gw).attr("y1", y(1)).attr("y2", y(1))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    AT.note(g, 4, y(1) - 5, "1.0 = explains nothing", DC.muted, 9);
    DL.curve(g, DAS.map((da, i) => [x(da), y(noB[i])]), { stroke: DC.bad, w: 2.2 });
    DL.curve(g, DAS.map((da, i) => [x(da), y(wiB[i])]), { stroke: DC.good, w: 2.2 });
    DAS.forEach((da, i) => {
      g.append("circle").attr("cx", x(da)).attr("cy", y(noB[i])).attr("r", 2.6).attr("fill", DC.bad);
      g.append("circle").attr("cx", x(da)).attr("cy", y(wiB[i])).attr("r", 2.6).attr("fill", DC.good);
    });
    DL.legend(g, [{ label: "no bias inside the tanh — an ODD function", color: DC.bad },
                  { label: "with a bias of scale " + DL.fmt(bs, 2), color: DC.good }],
      2, gh + 34, { vertical: true, gap: 13, font: 9.5 });
    AT.note(g, 2, gh + 66, kind === "odd" ? "the target is ODD: now the bias-free scorer is the one that fits"
      : "the target is EVEN: f(−q,−k) = +f(q,k), and no odd model can touch it", kind === "odd" ? DC.bad : DC.a2, 9.5);

    /* right — cost */
    const rx = gw + 66, rw = f.iw - rx;
    AT.title(g, rx, -10, "operations for one full n×n score matrix, d = d_a = 64");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const ns = [64, 128, 256, 512, 1024, 2048, 4096];
    const da = 64, dd = 64;
    const lines = [
      { k: "additive: n²·d_a MACs + n²·d_a tanh", col: DC.bad, f: n => n * n * da + n * n * da },
      { k: "general (bilinear)", col: DC.violet, f: n => n * dd * dd + n * n * dd },
      { k: "dot / scaled dot", col: DC.accent, f: n => n * n * dd }
    ];
    const xl = d3.scaleLog().domain([64, 4096]).range([0, rw - 12]);
    const yl = d3.scaleLog().domain([lines[2].f(64) / 2, lines[0].f(4096) * 2]).range([gh, 0]);
    DL.gridY(gg, yl, rw - 12, 4);
    DL.axisB(gg, xl, gh, 4, "sequence length n", d3.format("d"));
    DL.axisL(gg, yl, 4, "MACs", v => DL.big(v));
    lines.forEach(L => DL.curve(gg, ns.map(n => [xl(n), yl(L.f(n))]), { stroke: L.col, w: 2 }));
    DL.legend(gg, lines.map(L => ({ label: L.k, color: L.col })), 4, gh + 34, { vertical: true, gap: 13, font: 9.5 });
    AT.note(gg, 4, gh + 78, "and " + DL.big(4096 * 4096 * da) + " separate tanh evaluations for the additive form", DC.bad, 9.5);
    AT.note(gg, 4, gh + 90, "at n = 4096, d_a = 64 — none of which is a matrix product", DC.bad, 9.5);

    const i64 = DAS.length - 1;
    const kvv = DL.kv(g, 0, gh + 92, { keyW: 250, size: 10.5, lead: 14.5 });
    kvv("residual at d_a = 64, no bias", DL.fmt(noB[i64], 4), noB[i64] >= 0.98 ? DC.bad : DC.ink, true);
    kvv("residual at d_a = 64, with bias", DL.fmt(wiB[i64], 4), wiB[i64] < 0.9 ? DC.good : DC.a2, true);
    kvv("no-bias residual, d_a = 2 → 64", DL.fmt(noB[0], 4) + " → " + DL.fmt(noB[i64], 4), noB[i64] > noB[0] ? DC.bad : DC.ink);
    kvv("additive parameters (d = " + d + ", d_a = 64)", DL.commas(2 * d * 64 + 2 * 64));
    kvv("dot-product parameters", "0", DC.good);
    kvv("general (bilinear) parameters (d = " + d + ")", DL.commas(d * d));

    El("sf-readout").innerHTML =
      `d = ${d}, target = ${kind === "dot" ? "q·kᵀ (even)" : kind === "bil" ? "q·G·kᵀ (even)" : "a deliberately odd function"}, ` +
      `${N} fitting pairs, 1200 held out · at d_a = 64 the bias-free additive scorer reaches a residual of ` +
      `<b>${DL.fmt(noB[i64], 4)}</b> and the bias-carrying one <b>${DL.fmt(wiB[i64], 4)}</b> — a factor of ` +
      `<b>${DL.fmt(noB[i64] / Math.max(1e-9, wiB[i64]), 2)}×</b>. Going from d_a = 2 to d_a = 64 moves the bias-free residual ` +
      `from <b>${DL.fmt(noB[0], 4)}</b> to <b>${DL.fmt(noB[i64], 4)}</b>` +
      (noB[i64] >= noB[0] ? ` — <b>the wrong way</b>, which is what happens when the extra capacity can only overfit a target it cannot reach.`
        : `.`) +
      ` The dot product needs <b>0</b> extra parameters and one GEMM; the additive form needs ` +
      `<b>${DL.commas(2 * d * 64 + 2 * 64)}</b> and cannot be written as one.`;
  }
  ["sf-d", "sf-b"].forEach(id => d3.select("#" + id).on("input", draw));
  ["sf-t", "sf-n"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 5 · #vr-svg — Var[q·kᵀ] = dₖ, measured. The derivation needs only
   uncorrelated components with equal second moments, so the law survives the
   uniform, the Rademacher and the heavy-tailed draws; the correlation slider
   is what breaks it, because then the terms of the sum are dependent and the
   variances no longer add. ═════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#vr-svg"); if (svg.empty()) return;
  const W = 760, H = 400, DKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
  const El = id => document.getElementById(id);

  function comp(r, kind) {
    if (kind === "unif") return (r() - 0.5) * Math.sqrt(12);
    if (kind === "rade") return r() < 0.5 ? -1 : 1;
    if (kind === "heavy") {                       /* t with 5 dof, scaled to var 1 */
      let u = 0; for (let i = 0; i < 5; i++) { const z = DL.randn(r); u += z * z; }
      return DL.randn(r) / Math.sqrt(u / 5) / Math.sqrt(5 / 3);
    }
    return DL.randn(r);
  }
  /* correlated components: xᵢ = √(1−ρ)·zᵢ + √ρ·z₀, giving pairwise corr ρ
     and unit variance for every component. */
  function draws(dk, n, kind, rho, seed, scaled) {
    const r = DL.rng(seed), out = new Array(n);
    const a = Math.sqrt(1 - rho), b = Math.sqrt(rho);
    const sc = scaled ? 1 / Math.sqrt(dk) : 1;
    for (let t = 0; t < n; t++) {
      const q0 = comp(r, kind), k0 = comp(r, kind);
      let s = 0;
      for (let c = 0; c < dk; c++) s += (a * comp(r, kind) + b * q0) * (a * comp(r, kind) + b * k0);
      out[t] = s * sc;
    }
    return out;
  }
  function mv(a) { const m = d3.mean(a); let s = 0; a.forEach(x => s += (x - m) * (x - m)); return { m: m, v: s / (a.length - 1) }; }

  function draw() {
    const n = +El("vr-n").value, kind = El("vr-d").value, rho = +El("vr-r").value, scaled = El("vr-sc").checked;
    El("vr-rv").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 46 }), g = f.g;
    const nn = Math.min(n, 20000);
    const pts = DKS.map(dk => {
      const s = mv(draws(dk, Math.max(400, Math.round(nn * 64 / Math.max(64, dk))), kind, rho, 40 + dk, scaled));
      return { dk: dk, m: s.m, v: s.v };
    });
    /* fitted exponent of v ∝ dkᵖ, by least squares on the logs */
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    pts.forEach(p => { const X = Math.log(p.dk), Y = Math.log(Math.max(1e-12, p.v)); sx += X; sy += Y; sxx += X * X; sxy += X * Y; });
    const k = pts.length, slope = (k * sxy - sx * sy) / (k * sxx - sx * sx);

    const gw = 340, gh = 250;
    AT.title(g, 0, -10, "measured Var[ q·kᵀ" + (scaled ? " / √dₖ" : "") + " ] against dₖ");
    const x = d3.scaleLog().domain([1, 512]).range([0, gw]);
    const ylo = Math.min(0.3, ...pts.map(p => p.v)) / 2, yhi = Math.max(2, ...pts.map(p => p.v)) * 2;
    const y = d3.scaleLog().domain([ylo, yhi]).range([gh, 0]);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "dₖ", d3.format("d"));
    DL.axisL(g, y, 5, "variance", v => DL.fmtE(v, 0));
    DL.curve(g, DKS.map(dk => [x(dk), y(DL.clamp(scaled ? 1 : dk, ylo, yhi))]), { stroke: DC.a2, w: 1.4, dash: "4 3" });
    DL.curve(g, pts.map(p => [x(p.dk), y(DL.clamp(p.v, ylo, yhi))]), { stroke: DC.accent, w: 2.2 });
    pts.forEach(p => g.append("circle").attr("cx", x(p.dk)).attr("cy", y(DL.clamp(p.v, ylo, yhi))).attr("r", 2.6).attr("fill", DC.accent));
    DL.legend(g, [{ label: "measured", color: DC.accent },
                  { label: scaled ? "the prediction: 1, flat" : "the prediction: variance = dₖ", color: DC.a2, dash: "4 3" }],
      2, gh + 34, { vertical: true, gap: 13, font: 9.5 });
    AT.note(g, 2, gh + 66, "fitted exponent of variance ∝ dₖᵖ :  p = " + DL.fmt(slope, 4) +
      (rho > 0 ? "   (independence predicts " + (scaled ? "0" : "1") + ")" : "   — the prediction is " + (scaled ? "0" : "1")),
      Math.abs(slope - (scaled ? 0 : 1)) < 0.06 ? DC.good : DC.bad, 9.5);

    /* right — histograms at three dₖ */
    const rx = gw + 62, rw = f.iw - rx;
    AT.title(g, rx, -10, "the score itself, at three dₖ");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const show = [4, 64, 512], cols = [DC.teal, DC.accent, DC.violet];
    const all = show.map((dk, i) => draws(dk, 3000, kind, rho, 40 + dk, scaled));
    const lim = Math.max(...all.map(a => Math.max(...a.map(Math.abs)))) * 1.02;
    const xh = d3.scaleLinear().domain([-lim, lim]).range([0, rw - 12]);
    const hs = all.map(a => AT.hist(a, 46));
    const ymax = Math.max(...hs.map(h => Math.max(...h.bins)));
    const yh = d3.scaleLinear().domain([0, ymax * 1.1]).range([gh, 0]);
    DL.axisB(gg, xh, gh, 5, "score");
    DL.axisL(gg, yh, 4, "count");
    hs.forEach((h, i) => {
      DL.curve(gg, h.bins.map((c, b) => [xh(h.lo + (b + 0.5) * h.w), yh(c)]), { stroke: cols[i], w: 1.8 });
    });
    DL.legend(gg, show.map((dk, i) => ({ label: "dₖ = " + dk, color: cols[i] })), 4, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    const cur = pts[6];                                  /* dₖ = 64 */
    const kvv = DL.kv(g, 0, gh + 92, { keyW: 258, size: 10.5, lead: 14.5 });
    kvv("mean at dₖ = 64", DL.fmt(cur.m, 5));
    kvv("variance at dₖ = 64", DL.fmt(cur.v, 4), DC.accent, true);
    kvv("   / dₖ", DL.fmt(cur.v / 64, 5), Math.abs(cur.v / 64 - (scaled ? 1 / 64 : 1)) < 0.06 ? DC.good : DC.a2);
    kvv("sd at dₖ = 64", DL.fmt(Math.sqrt(cur.v), 4));
    kvv("   the prediction √dₖ" + (scaled ? " / √dₖ = 1" : ""), DL.fmt(scaled ? 1 : 8, 4), DC.a2);
    kvv("fitted exponent p in Var ∝ dₖᵖ", DL.fmt(slope, 4), Math.abs(slope - (scaled ? 0 : 1)) < 0.06 ? DC.good : DC.bad, true);

    El("vr-readout").innerHTML =
      `${{ norm: "standard normal", unif: "uniform", rade: "Rademacher ±1", heavy: "heavy-tailed t₅" }[kind]} components, ` +
      `pairwise correlation ρ = ${DL.fmt(rho, 2)}${scaled ? ", divided by √dₖ" : ", unscaled"} · at dₖ = 64 the measured variance is ` +
      `<b>${DL.fmt(cur.v, 4)}</b> against a prediction of <b>${scaled ? "1" : "64"}</b>. Fitting Var ∝ dₖᵖ over the whole sweep gives ` +
      `p = <b>${DL.fmt(slope, 4)}</b> where independence predicts <b>${scaled ? "0" : "1"}</b>. ` +
      (rho > 0
        ? `With ρ = ${DL.fmt(rho, 2)} the components are no longer independent, the variances stop simply adding, and the ` +
          `√dₖ scale is no longer the right standardisation — which is exactly what training does to a real head.`
        : `The distribution of the components does not enter: only their second moment does, which is why the bounded, ` +
          `the two-point and the heavy-tailed draws all land on the same line.`);
  }
  ["vr-r"].forEach(id => d3.select("#" + id).on("input", draw));
  ["vr-n", "vr-d"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#vr-sc").on("change", draw);
  draw();
})();

/* ═════════ 6 · #st-svg — the attention row without the scale, as a whole
   distribution. The mean of ‖J‖_F is NOT monotone in dₖ: it peaks at moderate
   concentration, because J = diag(α) − ααᵀ vanishes at BOTH the uniform and
   the one-hot end. The median and the 5th percentile are monotone, and the
   gap between them and the mean is the whole point. ═════════════════════════ */
(function () {
  const svg = d3.select("#st-svg"); if (svg.empty()) return;
  const W = 760, H = 420;
  const El = id => document.getElementById(id);
  const DKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

  function stats(n, dk, rows, which, scaled, seed) {
    const r = DL.rng(seed), sd = scaled ? 1 : Math.sqrt(dk), out = new Array(rows);
    const row = new Array(n);
    for (let t = 0; t < rows; t++) {
      for (let j = 0; j < n; j++) row[j] = DL.randn(r) * sd;
      const a = DL.softmax(row);
      out[t] = which === "jac" ? DL.softmaxJacFrob(a)
        : which === "max" ? Math.max(...a) : DL.entropyOf(a);
    }
    return out;
  }
  function quant(a, p) { const s = a.slice().sort((x, y) => x - y); return s[DL.clamp(Math.floor(p * (s.length - 1)), 0, s.length - 1)]; }

  function draw() {
    const e = +El("st-dk").value, dk = Math.pow(2, e), n = +El("st-n").value,
      which = El("st-w").value, rows = +El("st-r").value;
    El("st-dkv").textContent = dk;
    const f = DL.frame(svg, W, H, { l: 54, r: 16, t: 26, b: 46 }), g = f.g;
    const R = Math.max(400, Math.min(rows, Math.round(1200000 / n)));
    const U = stats(n, dk, R, which, false, 7), S = stats(n, dk, R, which, true, 7);
    const logx = which === "jac";
    const gw = 330, gh = 244;

    AT.title(g, 0, -10, DL.commas(R) + " random attention rows of " + n + " keys, at dₖ = " + dk);
    const lo = Math.max(1e-18, Math.min(...U.concat(S)) * 0.7), hi = Math.max(...U.concat(S)) * 1.05;
    const x = logx ? d3.scaleLog().domain([lo, hi]).range([0, gw]) : d3.scaleLinear().domain([0, hi]).range([0, gw]);
    const tf = v => logx ? Math.log(Math.max(v, lo)) : v;
    const HU = AT.hist(U.map(tf), 44), HS = AT.hist(S.map(tf), 44);
    const ymax = Math.max(...HU.bins, ...HS.bins);
    const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([gh, 0]);
    DL.axisB(g, x, gh, 5, which === "jac" ? "‖J‖_F" : which === "max" ? "max α" : "H(α), nats",
      logx ? (v => DL.fmtE(v, 0)) : null);
    DL.axisL(g, y, 4, "count");
    const back = h => h.bins.map((c, b) => [x(logx ? Math.exp(h.lo + (b + 0.5) * h.w) : (h.lo + (b + 0.5) * h.w)), y(c)]);
    DL.curve(g, back(HS), { stroke: DC.accent, w: 1.9, fill: DC.accent, fillOp: 0.14 });
    DL.curve(g, back(HU), { stroke: DC.bad, w: 1.9, fill: DC.bad, fillOp: 0.14 });
    const marks = [["mean", d3.mean(U), DC.bad, "4 2"], ["median", quant(U, 0.5), DC.a2, "2 3"], ["5th pct", quant(U, 0.05), DC.violet, "1 3"]];
    marks.forEach((m, i) => {
      const xv = DL.clamp(m[1], lo, hi);
      g.append("line").attr("x1", x(xv)).attr("x2", x(xv)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", m[2]).attr("stroke-dasharray", m[3]).attr("stroke-width", 1.3);
      AT.note(g, x(xv) + 3, 11 + i * 11, "unscaled " + m[0], m[2], 8.5);
    });
    DL.legend(g, [{ label: "scaled — q·kᵀ/√dₖ", color: DC.accent }, { label: "unscaled — q·kᵀ", color: DC.bad }],
      2, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    /* right — the three summaries against dₖ */
    const rx = gw + 68, rw = f.iw - rx;
    AT.title(g, rx, -10, "mean, median and 5th percentile against dₖ");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const RS = Math.max(300, Math.round(200000 / n));
    const sw = DKS.map(d => {
      const a = stats(n, d, RS, which, false, 9);
      return { dk: d, m: d3.mean(a), q5: quant(a, 0.05), md: quant(a, 0.5) };
    });
    const sref = stats(n, 1, RS, which, true, 9);
    const refM = d3.mean(sref);
    const vals = sw.flatMap(p => [p.m, p.md, Math.max(p.q5, 1e-18)]).concat([refM]);
    const yl = d3.scaleLog().domain([Math.max(1e-18, Math.min(...vals) * 0.5), Math.max(...vals) * 2]).range([gh, 0]).clamp(true);
    const xl = d3.scaleLog().domain([1, 1024]).range([0, rw - 12]);
    DL.gridY(gg, yl, rw - 12, 5);
    DL.axisB(gg, xl, gh, 5, "dₖ", d3.format("d"));
    DL.axisL(gg, yl, 5, which === "jac" ? "‖J‖_F" : which === "max" ? "max α" : "H(α)", v => DL.fmtE(v, 0));
    gg.append("line").attr("x1", 0).attr("x2", rw - 12).attr("y1", yl(refM)).attr("y2", yl(refM))
      .attr("stroke", DC.accent).attr("stroke-width", 2);
    AT.note(gg, 4, yl(refM) - 5, "scaled: flat, at every dₖ", DC.accent, 9);
    DL.curve(gg, sw.map(p => [xl(p.dk), yl(p.m)]), { stroke: DC.bad, w: 2.2 });
    DL.curve(gg, sw.map(p => [xl(p.dk), yl(p.md)]), { stroke: DC.a2, w: 1.8, dash: "4 3" });
    DL.curve(gg, sw.map(p => [xl(p.dk), yl(Math.max(p.q5, 1e-18))]), { stroke: DC.violet, w: 1.6, dash: "2 3" });
    gg.append("line").attr("x1", xl(dk)).attr("x2", xl(dk)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.55);
    DL.legend(gg, [{ label: "unscaled, mean", color: DC.bad },
                   { label: "unscaled, median", color: DC.a2, dash: "4 3" },
                   { label: "unscaled, 5th percentile", color: DC.violet, dash: "2 3" }],
      4, gh + 34, { vertical: true, gap: 13, font: 9.5 });
    /* where does the unscaled MEAN first drop below the scaled reference? */
    let cross = null;
    for (let i = 0; i < sw.length; i++) if (sw[i].m < refM) { cross = sw[i].dk; break; }
    if (which === "jac") AT.note(gg, 4, gh + 78, cross
      ? "the unscaled MEAN first falls below the scaled value only at dₖ = " + cross
      : "the unscaled MEAN never falls below the scaled value over this range", cross ? DC.a2 : DC.bad, 9.5);

    const sat = stats(n, dk, R, "max", false, 7);
    const kvv = DL.kv(g, 0, gh + 92, { keyW: 268, size: 10.5, lead: 14.5 });
    const nm = which === "jac" ? "‖J‖_F" : which === "max" ? "max α" : "H(α)";
    kvv(nm + " — scaled, mean", DL.fmtE(d3.mean(S), 4), DC.accent);
    kvv(nm + " — unscaled, mean", DL.fmtE(d3.mean(U), 4), d3.mean(U) < d3.mean(S) ? DC.bad : DC.a2, true);
    kvv(nm + " — unscaled, median", DL.fmtE(quant(U, 0.5), 4), DC.a2);
    kvv(nm + " — unscaled, 5th percentile", DL.fmtE(quant(U, 0.05), 3), DC.violet);
    kvv("median ratio, scaled / unscaled", DL.fmt(quant(S, 0.5) / Math.max(1e-300, quant(U, 0.5)), 1) + "×", DC.bad, true);
    kvv("unscaled rows with max α > 0.99", DL.fmt(100 * sat.filter(v => v > 0.99).length / sat.length, 2) + "%", DC.bad);
    kvv("largest entropy possible, ln n", DL.fmt(Math.log(n), 4), DC.muted);

    El("st-readout").innerHTML =
      `n = ${n} keys, dₖ = ${dk}, ${DL.commas(R)} rows · ${nm}: scaled mean <b>${DL.fmtE(d3.mean(S), 4)}</b>, ` +
      `unscaled mean <b>${DL.fmtE(d3.mean(U), 4)}</b>, unscaled median <b>${DL.fmtE(quant(U, 0.5), 4)}</b>, ` +
      `unscaled 5th percentile <b>${DL.fmtE(quant(U, 0.05), 3)}</b>. The <b>median</b> ratio is ` +
      `<b>${DL.fmt(quant(S, 0.5) / Math.max(1e-300, quant(U, 0.5)), 1)}×</b> while the <b>mean</b> ratio is only ` +
      `<b>${DL.fmt(d3.mean(S) / Math.max(1e-300, d3.mean(U)), 2)}×</b> — the mean is held up by the minority of rows that ` +
      `did not saturate. <b>${DL.fmt(100 * sat.filter(v => v > 0.99).length / sat.length, 2)}%</b> of unscaled rows put more ` +
      `than 99% of their weight on a single, arbitrary key before any training has happened.`;
  }
  d3.select("#st-dk").on("input", draw);
  ["st-n", "st-w", "st-r"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 7 · #lk-svg — from an exact lookup to attention.
   The table is built so that the raw embedding is DOMINATED by a nuisance
   block: dims 0–3 identify an entry and dims 4–7 are three times larger and
   shared. A raw dot product retrieves on the nuisance; the projections
   W_Q = W_K = the selector of dims 0–3 retrieve on the identity. That is
   exactly what W_Q and W_K are for. ══════════════════════════════════════ */
(function () {
  const svg = d3.select("#lk-svg"); if (svg.empty()) return;
  const W = 760, H = 410, N = 8, DID = 8, DNU = 4, D = DID + DNU, DV = 6;
  const El = id => document.getElementById(id);

  const TB = (function () {
    const r = DL.rng(5), K = [], V = [];
    for (let j = 0; j < N; j++) {
      const k = new Array(D).fill(0);
      /* the identity and nuisance blocks are each normalised, so that no key
         is found by everything merely for being long — see §07 on magnitude. */
      const id = [], nu = [];
      for (let c = 0; c < DID; c++) id.push(DL.randn(r));
      for (let c = 0; c < DNU; c++) nu.push(DL.randn(r));
      const idu = AT.unit(id), nuu = AT.unit(nu);
      for (let c = 0; c < DID; c++) k[c] = idu[c];
      for (let c = 0; c < DNU; c++) k[DID + c] = 3 * nuu[c];
      K.push(k);
      const v = []; for (let c = 0; c < DV; c++) v.push(DL.randn(r));
      V.push(v);
    }
    return { K: K, V: V };
  })();
  /* the query carries the IDENTITY block of the entry it wants, plus noise,
     and its OWN unrelated nuisance block — which is the whole difficulty. */
  function query(qi, nz) {
    const r = DL.rng(900 + qi), q = new Array(D).fill(0);
    const id = [], nu = [];
    for (let c = 0; c < DID; c++) id.push(TB.K[qi][c] + nz * DL.randn(r));
    for (let c = 0; c < DNU; c++) nu.push(DL.randn(r));
    const idu = AT.unit(id), nuu = AT.unit(nu);
    for (let c = 0; c < DID; c++) q[c] = idu[c];
    for (let c = 0; c < DNU; c++) q[DID + c] = 3 * nuu[c];
    return q;
  }
  function weights(q, mode, tau) {
    const s = TB.K.map(k => {
      let t = 0;
      const hi = mode === "proj" ? DID : D;
      for (let c = 0; c < hi; c++) t += q[c] * k[c];
      return t / tau;
    });
    return DL.softmax(s);
  }
  function blend(a) { const o = new Array(DV).fill(0); for (let j = 0; j < N; j++) for (let c = 0; c < DV; c++) o[c] += a[j] * TB.V[j][c]; return o; }
  function relErr(o, t) { let a = 0, b = 0; for (let c = 0; c < DV; c++) { a += (o[c] - t[c]) ** 2; b += t[c] * t[c]; } return Math.sqrt(a / b); }

  function draw() {
    const te = +El("lk-t").value, tau = Math.pow(10, te), qi = +El("lk-q").value,
      nz = +El("lk-nz").value, mode = El("lk-m").value;
    El("lk-tv").textContent = tau < 0.01 ? DL.fmtE(tau, 1) : DL.fmt(tau, 2);
    El("lk-qv").textContent = qi; El("lk-nzv").textContent = DL.fmt(nz, 2);
    const f = DL.frame(svg, W, H, { l: 44, r: 14, t: 26, b: 40 }), g = f.g;
    const q = query(qi, nz), a = weights(q, mode, tau), out = blend(a);
    const cell = d3.scaleSequential(d3.interpolateRdBu).domain([2.2, -2.2]);
    const mean = new Array(DV).fill(0); TB.V.forEach(v => v.forEach((x, c) => mean[c] += x / N));

    const cw = 12, y0 = 12, rh = 17;
    AT.title(g, 0, -10, "the table: a key strip and a value strip per entry");
    AT.note(g, -16, y0 + 10, "q", DC.a2, 9.5);
    AT.strip(g, q, 6, y0, cw, cell, { stroke: DC.a2 });
    AT.note(g, 6 + DID * cw - 3, y0 - 2, "│", DC.good, 11);
    AT.note(g, 6, y0 - 3, "identity", DC.good, 8);
    AT.note(g, 6 + DID * cw + 4, y0 - 3, "nuisance (×3, unrelated)", DC.bad, 8);
    for (let j = 0; j < N; j++) {
      const yy = y0 + 26 + j * rh;
      AT.note(g, -16, yy + 10, "k" + j, j === qi ? DC.good : DC.muted, 9);
      AT.strip(g, TB.K[j], 6, yy, cw, cell, { stroke: j === qi ? DC.good : "#0f1117" });
      AT.strip(g, TB.V[j], 6 + D * cw + 14, yy, cw, cell, { stroke: "#0f1117" });
      const bx = 6 + D * cw + 14 + DV * cw + 12, bw = 96 * a[j];
      g.append("rect").attr("x", bx).attr("y", yy + 1).attr("width", Math.max(0.5, bw)).attr("height", cw - 1)
        .attr("fill", j === qi ? DC.good : DC.accent).attr("fill-opacity", 0.85).attr("rx", 1.5);
      g.append("text").attr("x", bx + Math.max(0.5, bw) + 5).attr("y", yy + 11).attr("font-size", 8.6)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", a[j] > 0.15 ? DC.ink : DC.muted).text(DL.fmt(a[j], 3));
    }
    /* the retrieved output */
    const oy = y0 + 26 + N * rh + 14;
    AT.note(g, -16, oy + 10, "out", DC.a2, 9);
    AT.strip(g, out, 6 + D * cw + 14, oy, cw, cell, { stroke: DC.a2 });
    AT.note(g, -16, oy + rh + 10, "v" + qi, DC.good, 9);
    AT.strip(g, TB.V[qi], 6 + D * cw + 14, oy + rh, cw, cell, { stroke: DC.good });
    AT.note(g, -16, oy + 2 * rh + 10, "mean", DC.muted, 8.5);
    AT.strip(g, mean, 6 + D * cw + 14, oy + 2 * rh, cw, cell, { op: 0.45 });

    /* right — the three regimes against τ */
    const rx = 400, gw = f.iw - rx - 6, gh = 250;
    AT.title(g, rx, -10, "the three regimes, against temperature");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const ts = DL.linspace(-3, 3, 61).map(v => Math.pow(10, v));
    const rowsD = ts.map(t => {
      const aa = weights(q, mode, t), oo = blend(aa);
      return { t: t, w: aa[qi], h: DL.entropyOf(aa) / Math.log(N), e: relErr(oo, TB.V[qi]) };
    });
    const xl = d3.scaleLog().domain([1e-3, 1e3]).range([0, gw]);
    const yl = d3.scaleLinear().domain([0, 1.45]).range([gh, 0]);
    DL.gridY(gg, yl, gw, 5);
    DL.axisB(gg, xl, gh, 5, "temperature τ", v => DL.fmtE(v, 0));
    DL.axisL(gg, yl, 5, "");
    DL.curve(gg, rowsD.map(p => [xl(p.t), yl(p.w)]), { stroke: DC.good, w: 2 });
    DL.curve(gg, rowsD.map(p => [xl(p.t), yl(p.h)]), { stroke: DC.accent, w: 1.8, dash: "4 3" });
    DL.curve(gg, rowsD.map(p => [xl(p.t), yl(DL.clamp(p.e, 0, 1.45))]), { stroke: DC.bad, w: 2 });
    gg.append("line").attr("x1", xl(tau)).attr("x2", xl(tau)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    DL.legend(gg, [{ label: "weight on the intended entry", color: DC.good },
                   { label: "entropy of the weights / ln n", color: DC.accent, dash: "4 3" },
                   { label: "relative error against the stored value", color: DC.bad }],
      2, gh + 30, { vertical: true, gap: 13, font: 9.5 });

    const eProj = relErr(blend(weights(q, "proj", tau)), TB.V[qi]);
    const eRaw = relErr(blend(weights(q, "raw", tau)), TB.V[qi]);
    const kvv = DL.kv(g, rx, gh + 78, { keyW: 236, size: 10.5, lead: 14.5 });
    kvv("weight on the intended entry", DL.fmt(a[qi], 5), a[qi] > 0.9 ? DC.good : DC.a2, true);
    kvv("argmax entry", String(a.indexOf(Math.max(...a))) + " (wanted " + qi + ")",
      a.indexOf(Math.max(...a)) === qi ? DC.good : DC.bad);
    kvv("relative error, learned projections", DL.fmt(eProj, 5), DC.good);
    kvv("relative error, raw dot product", DL.fmt(eRaw, 5), DC.bad, true);
    kvv("entropy H(α) / ln n", DL.fmt(DL.entropyOf(a) / Math.log(N), 4));
    kvv("effective entries read, eᴴ", DL.fmt(Math.exp(DL.entropyOf(a)), 3), DC.a2);

    El("lk-readout").innerHTML =
      `τ = ${tau < 0.01 ? DL.fmtE(tau, 1) : DL.fmt(tau, 2)}, query built from entry ${qi} with noise ${DL.fmt(nz, 2)}, ` +
      `scoring on ${mode === "proj" ? "the learned projections" : "the raw vectors"} · the weight on the intended entry is ` +
      `<b>${DL.fmt(a[qi], 5)}</b> and the row is effectively reading <b>${DL.fmt(Math.exp(DL.entropyOf(a)), 2)}</b> of the ${N} ` +
      `entries. The retrieved vector differs from the stored one by <b>${DL.fmt(eProj, 4)}</b> with the projections and ` +
      `<b>${DL.fmt(eRaw, 4)}</b> without them. The raw score is dominated by the nuisance block, which is three times larger ` +
      `than the identity block and unrelated to it, so the raw lookup retrieves whichever entry happens to have a large ` +
      `nuisance overlap with the query — entry <b>${weights(q, "raw", tau).indexOf(Math.max(...weights(q, "raw", tau)))}</b> ` +
      `here, not entry ${qi}. The projections are what let the model decide which part of a representation counts as similar.`;
  }
  ["lk-t", "lk-q", "lk-nz"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#lk-m").on("change", draw);
  draw();
})();

/* ═════════ 8 · #sh-svg — every tensor in the block, with its size and cost.
   All counts come from DL.mhaParams and DL.attnMacs; nothing is recomputed
   here. MACs, not FLOPs — see the ATTENTION LAYOUT LAW. ══════════════════ */
(function () {
  const svg = d3.select("#sh-svg"); if (svg.empty()) return;
  const W = 760, H = 440;
  const El = id => document.getElementById(id);

  function draw() {
    const d = +El("sh-d").value, tie = El("sh-tie").checked;
    let h = +El("sh-h").value, g = +El("sh-g").value;
    const n = Math.pow(2, +El("sh-n").value);
    if (g > h) { g = h; El("sh-g").value = h; }
    const dk = tie ? Math.max(1, Math.round(d / h)) : +El("sh-dk").value;
    El("sh-hv").textContent = h; El("sh-gv").textContent = g;
    El("sh-nv").textContent = DL.commas(n); El("sh-dkv").textContent = dk;
    El("sh-dk").disabled = tie;
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 24, b: 34 }), g2 = f.g;

    const P = DL.mhaParams(d, dk, dk, h, g);
    const M = DL.attnMacs({ n: n, d: d, h: h, g: g, dh: dk, dff: 4 * d, nmat: 2 });
    const actScore = h * n * n, actQKV = n * (h + 2 * g) * dk, actOut = n * d;
    const peak = actScore + actQKV + actOut;

    /* left — the chain of boxes */
    const boxes = [
      ["X", n + " × " + d, n * d, DC.muted],
      ["X·W_Q", n + " × " + h * dk, n * h * dk, DC.accent],
      ["X·W_K", n + " × " + g * dk, n * g * dk, DC.accent],
      ["X·W_V", n + " × " + g * dk, n * g * dk, DC.accent],
      ["S = QKᵀ/√dₖ", h + " × " + n + " × " + n, h * n * n, DC.bad],
      ["A = softmax(S+M)", h + " × " + n + " × " + n, h * n * n, DC.bad],
      ["Z = A·V", n + " × " + h * dk, n * h * dk, DC.a2],
      ["Y = [Z]·W_O", n + " × " + d, n * d, DC.good]
    ];
    AT.title(g2, 0, -8, "every tensor in the block, and how many elements it holds");
    const mx = Math.max(...boxes.map(b => b[2]));
    const bh = 30, bw0 = 160;
    boxes.forEach((b, i) => {
      const yy = 8 + i * (bh + 8);
      const wd = bw0 * (0.22 + 0.78 * Math.sqrt(b[2] / mx));
      g2.append("rect").attr("x", 0).attr("y", yy).attr("width", wd).attr("height", bh - 6)
        .attr("rx", 4).attr("fill", b[3]).attr("fill-opacity", 0.22).attr("stroke", b[3]);
      AT.note(g2, 6, yy + 15, b[0], DC.ink, 10.5);
      AT.note(g2, bw0 + 12, yy + 11, b[1], DC.muted, 9.5);
      AT.note(g2, bw0 + 12, yy + 22, DL.big(b[2]) + " elements", b[3], 9.5);
      if (i) DL.arrow(g2, wd / 2 > 8 ? 8 : 4, yy - 8, 8, yy - 1, { color: DC.line, head: 4, w: 1 });
    });

    /* right — three bars */
    const rx = 330, rw = f.iw - rx;
    AT.title(g2, rx, -8, "parameters · arithmetic · activations");
    const gg = g2.append("g").attr("transform", `translate(${rx},0)`);
    const bars = [
      ["parameters", P.total, DC.accent, "d·dₖ·(h+g) + d·d_v·(g+h)  —  no n in it"],
      ["MACs, attention only", M.attn, DC.a2, "4nd² type term + 2n²·h·dₖ"],
      ["MACs, + a 4d FFN", M.total, DC.violet, "the FFN adds 8nd²"],
      ["peak activations", peak, DC.bad, "h·n² score elements dominate"]
    ];
    const bmax = Math.max(...bars.map(b => b[1]));
    const bmin = Math.min(...bars.map(b => b[1]));
    const xb = d3.scaleLog().domain([Math.min(bmin, bmax / 1e5), bmax * 1.6]).range([0, rw - 20]);
    bars.forEach((b, i) => {
      const yy = 14 + i * 56;
      AT.note(gg, 0, yy, b[0], DC.ink, 10.5);
      gg.append("rect").attr("x", 0).attr("y", yy + 5).attr("width", Math.max(2, xb(b[1]))).attr("height", 15)
        .attr("rx", 3).attr("fill", b[2]).attr("fill-opacity", 0.55).attr("stroke", b[2]);
      AT.note(gg, Math.max(2, xb(b[1])) + 6, yy + 17, DL.big(b[1]), b[2], 10);
      AT.note(gg, 0, yy + 34, b[3], DC.muted, 9);
    });
    /* sensitivity: how each scales when n doubles */
    const M2 = DL.attnMacs({ n: 2 * n, d: d, h: h, g: g, dh: dk, dff: 4 * d, nmat: 2 });
    const kvv = DL.kv(gg, 0, 250, { keyW: 248, size: 10.5, lead: 14.5 });
    kvv("parameters, per layer", DL.commas(P.total), DC.accent, true);
    kvv("   4d² would be", DL.commas(4 * d * d), h * dk === d && g === h ? DC.good : DC.a2);
    kvv("   W_Q / W_K / W_V / W_O", DL.big(P.Wq) + " / " + DL.big(P.Wk) + " / " + DL.big(P.Wv) + " / " + DL.big(P.Wo));
    kvv("saving from g < h", g < h ? DL.fmt(100 * (1 - P.total / (4 * d * d)), 1) + "%  (" + DL.fmt(4 * d * d / P.total, 3) + "×)" : "none, g = h", g < h ? DC.good : DC.muted);
    kvv("attention share of a block with a 4d FFN", DL.fmt(100 * P.total / (P.total + 8 * d * d), 1) + "%", DC.a2, true);
    kvv("MACs at n = " + DL.commas(n), DL.big(M.attn));
    kvv("   quadratic part", DL.big(M.quad) + "  (" + DL.fmt(100 * M.quad / M.attn, 1) + "% of attention)", DC.bad);
    kvv("if n doubles: params ×", "1.000", DC.accent);
    kvv("               attention MACs ×", DL.fmt(M2.attn / M.attn, 3), DC.a2);
    kvv("               activations ×", DL.fmt((h * 4 * n * n + 2 * n * (h + 2 * g) * dk + 2 * n * d) / peak, 3), DC.bad);

    El("sh-readout").innerHTML =
      `d = ${d}, h = ${h}, g = ${g}, dₖ = ${dk}, n = ${DL.commas(n)} · the block holds <b>${DL.commas(P.total)}</b> parameters ` +
      (h * dk === d && g === h ? `— exactly 4d² = <b>${DL.commas(4 * d * d)}</b>` :
        `against 4d² = <b>${DL.commas(4 * d * d)}</b>, a factor of <b>${DL.fmt(P.total / (4 * d * d), 3)}</b>`) +
      `, none of which depends on n. It does <b>${DL.big(M.attn)}</b> MACs, of which <b>${DL.fmt(100 * M.quad / M.attn, 1)}%</b> ` +
      `is the quadratic term, and holds <b>${DL.big(peak)}</b> activation elements at peak, of which the score matrices alone are ` +
      `<b>${DL.big(actScore)}</b>. Double n and the parameters are unchanged, the arithmetic goes up ` +
      `<b>${DL.fmt(M2.attn / M.attn, 3)}×</b> and the activations <b>${DL.fmt((h * 4 * n * n + 2 * n * (h + 2 * g) * dk + 2 * n * d) / peak, 3)}×</b>.`;
  }
  ["sh-h", "sh-g", "sh-n", "sh-dk"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#sh-d").on("change", draw);
  d3.select("#sh-tie").on("change", draw);
  draw();
})();

/* ═════════ 9 · #sc-svg — self against cross. One operator, two wirings; the
   only difference is where Q and where K,V are read from. ════════════════ */
(function () {
  const svg = d3.select("#sc-svg"); if (svg.empty()) return;
  const W = 760, H = 400, D = 8;
  const El = id => document.getElementById(id);

  function draw() {
    const mode = El("sc-m").value, mk = El("sc-mask").value;
    let nq = +El("sc-nq").value, nk = +El("sc-nk").value;
    if (mode === "self") nk = nq;                 /* locked to n_q, but the slider keeps its own value for cross mode */
    El("sc-nqv").textContent = nq; El("sc-nkv").textContent = nk;
    El("sc-nk").disabled = (mode === "self");
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 26, b: 34 }), g = f.g;

    const Xt = AT.gauss(nq, D, 21, 1), Xs = mode === "self" ? Xt : AT.gauss(nk, D, 22, 1);
    const P = DL.mhaInit(D, 4, 4, 1, { seed: 3 });
    const Q = DL.matmul(Xt, P.Wq), K = DL.matmul(Xs, P.Wk), V = DL.matmul(Xs, P.Wv);
    let M = null;
    if (mk === "causal") M = DL.causalMask(nq, nk);
    else if (mk === "pad") { const ok = []; for (let j = 0; j < nk; j++) ok.push(j < nk - 3); M = DL.padMask(nq, ok); }
    const o = DL.sdpa(Q, K, V, { mask: M });
    let blocked = 0;
    if (M) for (let i = 0; i < nq; i++) for (let j = 0; j < nk; j++) if (!isFinite(M[i][j])) blocked++;

    /* left — the wiring */
    AT.title(g, 0, -10, mode === "self" ? "self-attention: one sequence, three arrows" : "cross-attention: two sequences");
    const cw = 11;
    const tY = 20, sY = mode === "self" ? 20 : 190;
    AT.note(g, 0, tY - 5, mode === "self" ? "X   (n × d)" : "X_tgt   (n_q × d)", DC.accent, 9.5);
    AT.strip(g, [], 0, 0, 1, () => "none");
    for (let i = 0; i < nq; i++) AT.strip(g, Xt[i], 0, tY + i * (cw + 1), cw, d3.scaleSequential(d3.interpolateRdBu).domain([2.2, -2.2]), { stroke: "#0f1117" });
    if (mode !== "self") {
      AT.note(g, 0, sY - 5, "X_src   (n_k × d)", DC.a2, 9.5);
      for (let j = 0; j < nk; j++) AT.strip(g, Xs[j], 0, sY + j * (cw + 1), cw, d3.scaleSequential(d3.interpolateRdBu).domain([2.2, -2.2]), { stroke: "#0f1117" });
    }
    const ax = D * cw + 16;
    const arrows = [["W_Q → Q", tY + nq * (cw + 1) / 2, DC.accent], ["W_K → K", sY + nk * (cw + 1) / 2 - 12, DC.a2], ["W_V → V", sY + nk * (cw + 1) / 2 + 12, DC.violet]];
    arrows.forEach(a => {
      DL.arrow(g, ax, a[1], ax + 56, a[1], { color: a[2], head: 5, w: 1.4 });
      AT.note(g, ax + 2, a[1] - 4, a[0], a[2], 9);
    });

    /* right — the score matrix */
    const rx = 300;
    AT.title(g, rx, -10, "S  =  " + nq + " × " + nk + (mode === "self" ? "   (square)" : "   (rectangular)"));
    const mcw = Math.min(18, 380 / Math.max(nq, nk));
    const sc2 = d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...o.A.map(r => Math.max(...r)))]);
    AT.heat(g, o.A, rx, 8, mcw, sc2, { stroke: "#0f1117" });
    if (M) for (let i = 0; i < nq; i++) for (let j = 0; j < nk; j++) if (!isFinite(M[i][j]))
      g.append("path").attr("d", `M${rx + j * mcw},${8 + i * mcw} l${mcw},${mcw} M${rx + (j + 1) * mcw},${8 + i * mcw} l${-mcw},${mcw}`)
        .attr("stroke", DC.bad).attr("stroke-width", 0.7).attr("stroke-opacity", 0.85);
    AT.note(g, rx, 8 + nq * mcw + 14, "keys  j = 1 … " + nk + "  →", DC.muted, 9.5);
    AT.note(g, rx, 8 + nq * mcw + 26, "queries i = 1 … " + nq + "  ↓", DC.muted, 9.5);
    if (mode === "cross" && mk === "causal")
      AT.note(g, rx, 8 + nq * mcw + 44, "a causal mask on CROSS-attention hides " + blocked + " query–key pairs for no reason —", DC.bad, 9.5);
    if (mode === "cross" && mk === "causal")
      AT.note(g, rx, 8 + nq * mcw + 56, "the source is fully available; there is nothing to hide from.", DC.bad, 9.5);

    const kvv = DL.kv(g, 0, 290, { keyW: 244, size: 10.5, lead: 14.5 });
    kvv("score matrix shape", nq + " × " + nk, DC.ink, true);
    kvv("   elements", DL.commas(nq * nk));
    kvv("entries the mask forbids", DL.commas(blocked) + (blocked ? "  (" + DL.fmt(100 * blocked / (nq * nk), 1) + "%)" : ""), blocked ? DC.bad : DC.muted);
    kvv("K,V recomputed per generated token", mode === "self" ? "1 key + 1 value" : "0 — computed once", mode === "self" ? DC.a2 : DC.good, true);
    kvv("cache growth per generated token", mode === "self" ? "2·g·dₖ elements" : "0", mode === "self" ? DC.a2 : DC.good);
    kvv("all rows sum to 1", (() => { let w = 0; o.A.forEach(r => { let s = 0; r.forEach(v => s += v); w = Math.max(w, Math.abs(s - 1)); }); return "max |∑α − 1| = " + DL.fmtE(w, 1); })(), DC.good);

    El("sc-readout").innerHTML =
      `${mode === "self" ? "self" : "cross"}-attention, n_q = ${nq}, n_k = ${nk} · the score matrix is <b>${nq} × ${nk}</b> = ` +
      `<b>${DL.commas(nq * nk)}</b> entries, of which the ${mk === "none" ? "(no) mask forbids 0" : "mask forbids <b>" + blocked + "</b>"}. ` +
      (mode === "self"
        ? `In self-attention the two lengths are the same object, so generating one more token appends <b>one</b> key and <b>one</b> value and the cache grows for ever (§24).`
        : `In cross-attention the two lengths are independent, and the source keys and values are computed <b>once</b> for the whole generation — a cross-attention cache never grows.`) +
      (mode === "cross" && mk === "causal" ? ` A causal mask here is a bug: it hides <b>${blocked}</b> query–key pairs whose source positions are already known.` : ``);
  }
  ["sc-nq", "sc-nk"].forEach(id => d3.select("#" + id).on("input", draw));
  ["sc-m", "sc-mask"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 10 · #dw-svg — three mixing matrices. The dense one is a learned
   parameter, the convolutional one is a learned parameter that is banded and
   tied, and the attention one is COMPUTED from the input. Pressing "change
   input" is the experiment: two of the three do not move at all. ═════════ */
(function () {
  const svg = d3.select("#dw-svg"); if (svg.empty()) return;
  const W = 760, H = 400, D = 8;
  const El = id => document.getElementById(id);
  let prev = null;

  function dense(n) { return AT.gauss(n, n, 71, 1 / Math.sqrt(n)); }
  function conv(n, fw) {
    const r = DL.rng(72), k = [];
    for (let i = 0; i < fw; i++) k.push(DL.randn(r) / Math.sqrt(fw));
    const A = DL.zeros2(n, n), half = (fw - 1) / 2;
    for (let i = 0; i < n; i++) for (let t = -half; t <= half; t++) {
      const j = i + t; if (j >= 0 && j < n) A[i][j] = k[t + half];
    }
    return A;
  }
  function attn(n, xi) {
    const X = AT.gauss(n, D, 100 + xi, 1);
    const P = DL.mhaInit(D, 4, 4, 1, { seed: 9 });
    const Q = DL.matmul(X, P.Wq), K = DL.matmul(X, P.Wk);
    return DL.sdpa(Q, K, DL.matmul(X, P.Wv), {}).A;
  }
  function maxAbsDiff(A, B) {
    let m = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
    return m;
  }

  function draw() {
    const n = +El("dw-n").value, fw = +El("dw-f").value, xi = +El("dw-x").value, diff = El("dw-diff").checked;
    El("dw-nv").textContent = n; El("dw-fv").textContent = fw; El("dw-xv").textContent = xi;
    const f = DL.frame(svg, W, H, { l: 28, r: 16, t: 26, b: 34 }), g = f.g;
    const cur = { dense: dense(n), conv: conv(n, fw), attn: attn(n, xi) };
    const pv = prev && prev.n === n && prev.fw === fw ? prev : null;

    const cw = Math.min(15, 190 / n);
    const panels = [
      ["dense over positions", cur.dense, n * n, "a learned parameter", DC.violet, d3.scaleSequential(d3.interpolateRdBu).domain([0.6, -0.6])],
      ["convolution, width " + fw, cur.conv, fw, "learned, banded AND tied", DC.a2, d3.scaleSequential(d3.interpolateRdBu).domain([0.9, -0.9])],
      ["attention", cur.attn, 0, "computed from the input", DC.accent, d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...cur.attn.map(r => Math.max(...r)))])]
    ];
    panels.forEach((p, i) => {
      const x0 = i * 240;
      AT.title(g, x0, -10, p[0], p[4]);
      let Mshow = p[1], sc = p[5];
      if (diff) {
        const base = pv ? pv[["dense", "conv", "attn"][i]] : p[1];
        Mshow = p[1].map((r, a) => r.map((v, b) => v - base[a][b]));
        const mm = Math.max(1e-12, Math.max(...Mshow.map(r => Math.max(...r.map(Math.abs)))));
        sc = d3.scaleSequential(d3.interpolateRdBu).domain([mm, -mm]);
      }
      AT.heat(g, Mshow, x0, 6, cw, sc, { stroke: n <= 18 ? "#0f1117" : null });
      AT.note(g, x0, 6 + n * cw + 16, "free parameters in this matrix: " + (p[2] === 0 ? "0" : DL.commas(p[2])), p[4], 10);
      AT.note(g, x0, 6 + n * cw + 29, p[3], DC.muted, 9.5);
      const chg = pv ? maxAbsDiff(p[1], pv[["dense", "conv", "attn"][i]]) : 0;
      AT.note(g, x0, 6 + n * cw + 45, "change since the last input:", DC.muted, 9.5);
      AT.note(g, x0, 6 + n * cw + 57, chg === 0 ? "exactly 0.000e+00" : DL.fmtE(chg, 3), chg === 0 ? DC.good : DC.bad, 10.5);
      if (diff && !pv) AT.note(g, x0, 6 + n * cw + 73, "(press “change input” first)", DC.muted, 9);
    });

    const dD = pv ? maxAbsDiff(cur.dense, pv.dense) : 0;
    const dC = pv ? maxAbsDiff(cur.conv, pv.conv) : 0;
    const dA = pv ? maxAbsDiff(cur.attn, pv.attn) : 0;
    const kvv = DL.kv(g, 0, 6 + n * cw + 92, { keyW: 300, size: 10.5, lead: 14.5 });
    kvv("dense: free parameters, and does it depend on n?", DL.commas(n * n) + " — yes, quadratically", DC.violet);
    kvv("convolution: free parameters, depends on n?", DL.commas(fw) + " — no", DC.a2);
    kvv("attention: free parameters IN THE MATRIX", "0 — the matrix is a function", DC.accent, true);
    kvv("attention: parameters in the FUNCTION (d = " + D + ", dₖ = 4)", DL.commas(2 * D * 4) + " — and no n in it either", DC.accent);
    kvv("max change of the dense matrix", DL.fmtE(dD, 3), dD === 0 ? DC.good : DC.bad);
    kvv("max change of the convolution matrix", DL.fmtE(dC, 3), dC === 0 ? DC.good : DC.bad);
    kvv("max change of the ATTENTION matrix", DL.fmtE(dA, 3), dA > 1e-6 ? DC.bad : DC.muted, true);

    El("dw-readout").innerHTML =
      `n = ${n}, input ${xi} · the dense mixing matrix has <b>${DL.commas(n * n)}</b> free parameters and can only ever run at n = ${n}; ` +
      `the convolution has <b>${fw}</b>, repeated down the diagonals; attention's matrix has <b>0</b> — it is produced by ` +
      `<b>${DL.commas(2 * D * 4)}</b> parameters in W_Q and W_K, neither of which knows what n is. ` +
      (pv
        ? `Since the last input the dense matrix changed by <b>${DL.fmtE(dD, 3)}</b>, the convolution by <b>${DL.fmtE(dC, 3)}</b> ` +
          `and attention's by <b>${DL.fmtE(dA, 3)}</b>. The first two are exactly zero; that is the whole distinction.`
        : `Press <b>change input</b> to redraw with a fresh sequence and compare the three.`);
    prev = { dense: cur.dense, conv: cur.conv, attn: cur.attn, n: n, fw: fw };
  }
  ["dw-n", "dw-f", "dw-x"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#dw-diff").on("change", draw);
  d3.select("#dw-next").on("click", function () {
    const e = document.getElementById("dw-x");
    e.value = (+e.value + 1) % 10; draw();
  });
  draw();
})();

/* ═════════ 11 · #pe-svg — permutation equivariance, measured.
   Attention(PX) = P·Attention(X) exactly, to floating-point round-off. A
   causal mask, an added position embedding or a convolution each break it,
   and the figure shows the residual jumping fifteen orders of magnitude. */
(function () {
  const svg = d3.select("#pe-svg"); if (svg.empty()) return;
  const W = 760, H = 410, D = 8;
  const El = id => document.getElementById(id);
  const MODES = [["none", "plain self-attention"], ["causal", "a causal mask"],
                 ["pos", "added position embeddings"], ["conv", "a convolution"]];

  function permOf(n, k) {
    const p = []; for (let i = 0; i < n; i++) p.push(i);
    if (k === 0) return p;
    DL.shuffle(p, DL.rng(200 + k));
    return p;
  }
  function apply(X, p) { return p.map(i => X[i].slice()); }
  function op(X, kind, h) {
    const n = X.length;
    if (kind === "conv") {                       /* a banded, tied mixing */
      const r = DL.rng(72), k = [];
      for (let i = 0; i < 3; i++) k.push(DL.randn(r));
      const Y = DL.zeros2(n, D);
      for (let i = 0; i < n; i++) for (let t = -1; t <= 1; t++) {
        const j = i + t; if (j < 0 || j >= n) continue;
        for (let c = 0; c < D; c++) Y[i][c] += k[t + 1] * X[j][c];
      }
      return Y;
    }
    let Xi = X;
    if (kind === "pos") {
      const pe = AT.gauss(64, D, 55, 0.8);
      Xi = X.map((r, i) => r.map((v, c) => v + pe[i][c]));
    }
    const dk = Math.max(1, Math.floor(D / h));
    const P = DL.mhaInit(D, dk, dk, h, { seed: 11 });
    const M = kind === "causal" ? DL.causalMask(n) : null;
    return DL.mhaForward(Xi, P, { mask: M }).Y;
  }
  function resid(n, pk, kind, h) {
    const X = AT.gauss(n, D, 300, 1), p = permOf(n, pk);
    const Y = op(X, kind, h), Yp = op(apply(X, p), kind, h), PY = apply(Y, p);
    let m = 0, s = 0;
    for (let i = 0; i < n; i++) for (let c = 0; c < D; c++) { m = Math.max(m, Math.abs(Yp[i][c] - PY[i][c])); s = Math.max(s, Math.abs(PY[i][c])); }
    return { m: m, s: s, X: X, p: p, Y: Y, Yp: Yp, PY: PY };
  }

  function draw() {
    const n = +El("pe-n").value, pk = +El("pe-p").value, kind = El("pe-b").value, h = +El("pe-h").value;
    El("pe-nv").textContent = n; El("pe-pv").textContent = pk; El("pe-hv").textContent = h;
    const f = DL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 34 }), g = f.g;
    const R = resid(n, pk, kind, h);
    const cell = d3.scaleSequential(d3.interpolateRdBu).domain([2.4, -2.4]);
    const cw = Math.min(13, 110 / D);

    const cols = [["X", R.X, DC.muted], ["PX", apply(R.X, R.p), DC.muted],
                  ["f(PX)", R.Yp, DC.a2], ["P·f(X)", R.PY, DC.accent],
                  ["difference", R.Yp.map((r, i) => r.map((v, c) => v - R.PY[i][c])), DC.bad]];
    AT.title(g, 0, -10, "the operator applied to the permuted input, against the permuted output");
    cols.forEach((c, i) => {
      const x0 = i * 135;
      AT.note(g, x0, 6, c[0], c[2], 10.5);
      const sc = i === 4 ? d3.scaleSequential(d3.interpolateRdBu).domain([Math.max(1e-16, R.m), -Math.max(1e-16, R.m)]) : cell;
      for (let r2 = 0; r2 < n; r2++) AT.strip(g, c[1][r2], x0, 14 + r2 * (cw + 1), cw, sc, { stroke: "#0f1117" });
      if (i === 4) AT.note(g, x0, 14 + n * (cw + 1) + 14, "scaled to ±" + DL.fmtE(R.m, 1), DC.bad, 9);
    });

    /* the residual bar chart over all four settings */
    const by = 14 + n * (cw + 1) + 44;
    AT.title(g, 0, by - 8, "max |f(PX) − P f(X)|, for each of the four settings");
    const rs = MODES.map(m => ({ k: m[0], lab: m[1], v: resid(n, pk, m[0], h).m }));
    const xb = d3.scaleLog().domain([1e-17, Math.max(1e-15, ...rs.map(r => r.v)) * 4]).range([0, f.iw - 200]);
    rs.forEach((r, i) => {
      const yy = by + 8 + i * 22;
      AT.note(g, 0, yy + 10, r.lab, r.k === kind ? DC.ink : DC.muted, 10);
      g.append("rect").attr("x", 168).attr("y", yy).attr("width", Math.max(1.5, xb(Math.max(r.v, 1e-17))))
        .attr("height", 13).attr("rx", 2)
        .attr("fill", r.v < 1e-12 ? DC.good : DC.bad).attr("fill-opacity", r.k === kind ? 0.85 : 0.35);
      AT.note(g, 168 + Math.max(1.5, xb(Math.max(r.v, 1e-17))) + 6, yy + 11, DL.fmtE(r.v, 2), r.v < 1e-12 ? DC.good : DC.bad, 9.5);
    });

    const kvv = DL.kv(g, 470, by + 8, { keyW: 178, size: 10.5, lead: 14.5 });
    kvv("residual", DL.fmtE(R.m, 3), R.m < 1e-12 ? DC.good : DC.bad, true);
    kvv("scale of the output", DL.fmt(R.s, 4));
    kvv("residual / scale", DL.fmtE(R.m / Math.max(1e-300, R.s), 2), R.m < 1e-12 ? DC.good : DC.bad);
    kvv("verdict", R.m < 1e-12 ? "EQUIVARIANT" : "broken", R.m < 1e-12 ? DC.good : DC.bad, true);

    El("pe-readout").innerHTML =
      `n = ${n}, ${h} head${h > 1 ? "s" : ""}, permutation ${pk}, ${MODES.find(m => m[0] === kind)[1]} · ` +
      `max |f(PX) − P·f(X)| = <b>${DL.fmtE(R.m, 3)}</b> against an output scale of <b>${DL.fmt(R.s, 4)}</b>, a relative ` +
      `<b>${DL.fmtE(R.m / Math.max(1e-300, R.s), 2)}</b>. ` +
      (R.m < 1e-12
        ? `That is floating-point round-off: attention is <b>exactly</b> permutation-equivariant, and therefore cannot tell ` +
          `"dog bites man" from "man bites dog" at any depth. Something positional has to be added, and part 6 owns which.`
        : `The symmetry is broken — by ${MODES.find(m => m[0] === kind)[1]}. Note in particular that the <b>causal mask alone</b> ` +
          `breaks it, which is why decoder-only models are less position-blind than the proof suggests.`);
  }
  ["pe-n", "pe-p", "pe-h"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#pe-b").on("change", draw);
  draw();
})();

/* ═════════ 12 · #hd-svg — heads specialise, trained live.
   A causal multi-head block over a random symbol sequence with THREE
   simultaneous read-outs: the token at t−1, the token at position 0, and the
   token at t. One head produces one convex combination per position and so
   cannot serve three; three heads can. Trained with Adam here rather than
   through DL.OPT because the parameter set is a page-local pile of matrices
   rather than a DL net; the update rule is the same one DL.OPT.adam
   implements and was checked against it. ═══════════════════════════════════ */
(function () {
  const svg = d3.select("#hd-svg"); if (svg.empty()) return;
  const W = 760, H = 430;
  const V = 5, T = 6, DM = 10, DH = 4, B = 16, LR = 0.04;
  const El = id => document.getElementById(id);
  const cache = {};

  function data(n, r) {
    const idx = [], X = [], tg = [];
    for (let b = 0; b < n; b++) {
      const s = []; for (let t = 0; t < T; t++) s.push(Math.floor(r() * V));
      idx.push(s);
      const x = DL.zeros2(T, V);
      for (let t = 0; t < T; t++) x[t][s[t]] = 1;
      X.push(x);
      const tt = [];
      for (let t = 0; t < T; t++) tt.push([t === 0 ? s[0] : s[t - 1], s[0], s[t]]);
      tg.push(tt);
    }
    return { X: X, tg: tg };
  }
  function initP(h, seed) {
    const r = DL.rng(seed * 97 + h);
    const gen = (a, b, s) => { const M = DL.zeros2(a, b); for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) M[i][j] = DL.randn(r) * s; return M; };
    return { E: gen(V, DM, 0.5), Pos: gen(T, DM, 0.5),
             Wq: gen(DM, h * DH, 1 / Math.sqrt(DM)), Wk: gen(DM, h * DH, 1 / Math.sqrt(DM)),
             Wv: gen(DM, h * DH, 1 / Math.sqrt(DM)), Wo: gen(h * DH, DM, 1 / Math.sqrt(h * DH)),
             Wr: gen(DM, 3 * V, 1 / Math.sqrt(DM)), h: h };
  }
  const MSK = DL.causalMask(T);
  function fwd(P, x) {                                   /* one sequence */
    const Z = DL.zeros2(T, DM);
    for (let t = 0; t < T; t++) for (let c = 0; c < DM; c++) {
      let s = P.Pos[t][c];
      for (let v = 0; v < V; v++) s += x[t][v] * P.E[v][c];
      Z[t][c] = s;
    }
    const st = DL.mhaForward(Z, P, { mask: MSK });
    const L = DL.matmul(st.Y, P.Wr);
    return { Z: Z, st: st.st, Y: st.Y, L: L };
  }
  function lossOf(P, D) {
    let tot = 0;
    for (let b = 0; b < D.X.length; b++) {
      const o = fwd(P, D.X[b]);
      for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) {
        const lg = o.L[t].slice(k * V, (k + 1) * V);
        tot += DL.xent(lg, D.tg[b][t][k]);
      }
    }
    return tot / (D.X.length * T * 3);
  }
  function train(h, seed, steps) {
    const key = h + "|" + seed + "|" + steps;
    if (cache[key]) return cache[key];
    const P = initP(h, seed), r = DL.rng(seed * 31 + 5);
    const keys = ["E", "Pos", "Wq", "Wk", "Wv", "Wo", "Wr"];
    const m = {}, v2 = {};
    keys.forEach(k => { m[k] = DL.zeros2(P[k].length, P[k][0].length); v2[k] = DL.zeros2(P[k].length, P[k][0].length); });
    const test = data(240, DL.rng(999));
    const curve = [];
    for (let s = 0; s < steps; s++) {
      const D = data(B, r);
      const G = {}; keys.forEach(k => G[k] = DL.zeros2(P[k].length, P[k][0].length));
      const scale = 1 / (B * T * 3);
      for (let b = 0; b < B; b++) {
        const o = fwd(P, D.X[b]);
        const dL = DL.zeros2(T, 3 * V);
        for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) {
          const p = DL.softmax(o.L[t].slice(k * V, (k + 1) * V));
          for (let j = 0; j < V; j++) dL[t][k * V + j] = (p[j] - (j === D.tg[b][t][k] ? 1 : 0)) * scale;
        }
        const gWr = DL.matmul(DL.transpose(o.Y), dL);
        for (let i = 0; i < DM; i++) for (let j = 0; j < 3 * V; j++) G.Wr[i][j] += gWr[i][j];
        const dY = DL.matmul(dL, DL.transpose(P.Wr));
        const gA = DL.mhaBackward(dY, o.st, P);
        ["Wq", "Wk", "Wv", "Wo"].forEach(k => {
          for (let i = 0; i < G[k].length; i++) for (let j = 0; j < G[k][0].length; j++) G[k][i][j] += gA[k][i][j];
        });
        for (let t = 0; t < T; t++) {
          for (let c = 0; c < DM; c++) {
            G.Pos[t][c] += gA.X[t][c];
            for (let vv = 0; vv < V; vv++) G.E[vv][c] += D.X[b][t][vv] * gA.X[t][c];
          }
        }
      }
      const t1 = s + 1;
      keys.forEach(k => {
        for (let i = 0; i < P[k].length; i++) for (let j = 0; j < P[k][0].length; j++) {
          m[k][i][j] = 0.9 * m[k][i][j] + 0.1 * G[k][i][j];
          v2[k][i][j] = 0.999 * v2[k][i][j] + 0.001 * G[k][i][j] * G[k][i][j];
          P[k][i][j] -= LR * (m[k][i][j] / (1 - Math.pow(0.9, t1))) / (Math.sqrt(v2[k][i][j] / (1 - Math.pow(0.999, t1))) + 1e-8);
        }
      });
      if (s % Math.max(1, Math.floor(steps / 30)) === 0 || s === steps - 1) curve.push([s, lossOf(P, { X: test.X.slice(0, 60), tg: test.tg.slice(0, 60) })]);
    }
    /* mean attention pattern per head */
    const acc = [];
    for (let j = 0; j < h; j++) acc.push(DL.zeros2(T, T));
    let nS = 0;
    for (let b = 0; b < 120; b++) {
      const o = fwd(P, test.X[b]);
      for (let j = 0; j < h; j++) for (let t = 0; t < T; t++) for (let k = 0; k <= t; k++) acc[j][t][k] += o.st.per[j].A[t][k];
      nS++;
    }
    acc.forEach(Ah => { for (let t = 0; t < T; t++) for (let k = 0; k < T; k++) Ah[t][k] /= nS; });
    const res = { P: P, curve: curve, loss: lossOf(P, test), A: acc };
    cache[key] = res;
    return res;
  }
  function roles(Ah) {
    let a = 0, b = 0, c = 0, m = 0;
    for (let t = 1; t < T; t++) { a += Ah[t][t - 1]; b += Ah[t][0]; c += Ah[t][t]; m++; }
    return [a / m, b / m, c / m];
  }

  function draw() {
    const hShow = +El("hd-h").value, seed = +El("hd-s").value, steps = +El("hd-n").value;
    El("hd-hv").textContent = hShow; El("hd-sv").textContent = seed;
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 26, b: 34 }), g = f.g;
    const runs = [1, 2, 3, 4].map(h => train(h, seed, steps));
    const cur = runs[hShow - 1];
    const names = ["the previous token", "the first token", "myself"];

    AT.title(g, 0, -10, "the " + hShow + " head" + (hShow > 1 ? "s" : "") + " of the trained model, mean causal attention map");
    const cw = 15;
    for (let j = 0; j < hShow; j++) {
      const x0 = j * 168;
      const sc = d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...cur.A[j].map(r => Math.max(...r)))]);
      AT.heat(g, cur.A[j], x0, 6, cw, sc, { stroke: "#0f1117" });
      const rr = roles(cur.A[j]);
      const win = rr.indexOf(Math.max(...rr));
      AT.note(g, x0, 6 + T * cw + 14, "head " + j + " → " + names[win], DC.accent, 10);
      ["on t−1", "on position 0", "on t"].forEach((lab, k) => {
        AT.note(g, x0, 6 + T * cw + 27 + k * 11, lab + ": " + DL.fmt(rr[k], 4), k === win ? DC.good : DC.muted, 9);
      });
    }

    /* lower left — loss curves */
    const ly = 190, lw = 320, lh = 170;
    AT.title(g, 0, ly - 8, "training loss against step");
    const gg = g.append("g").attr("transform", `translate(0,${ly})`);
    const allv = runs.flatMap(r => r.curve.map(c => c[1]));
    const x = d3.scaleLinear().domain([0, steps]).range([0, lw]);
    const y = d3.scaleLog().domain([Math.max(1e-6, Math.min(...allv) / 2), Math.max(...allv) * 1.4]).range([lh, 0]).clamp(true);
    DL.gridY(gg, y, lw, 4);
    DL.axisB(gg, x, lh, 5, "step");
    DL.axisL(gg, y, 4, "loss", v => DL.fmtE(v, 0));
    const cols = [DC.bad, DC.a2, DC.good, DC.accent];
    runs.forEach((r, i) => DL.curve(gg, r.curve.map(c => [x(c[0]), y(Math.max(c[1], 1e-6))]),
      { stroke: cols[i], w: i === hShow - 1 ? 2.4 : 1.3, op: i === hShow - 1 ? 1 : 0.5 }));
    gg.append("line").attr("x1", 0).attr("x2", lw).attr("y1", y(Math.log(V))).attr("y2", y(Math.log(V)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    AT.note(gg, 4, y(Math.log(V)) - 5, "chance, ln " + V + " = " + DL.fmt(Math.log(V), 4), DC.muted, 9);
    DL.legend(gg, runs.map((r, i) => ({ label: (i + 1) + " head" + (i ? "s" : ""), color: cols[i] })), lw - 66, 6, { vertical: true, gap: 13, font: 9.5 });

    /* lower right — final loss bars */
    const rx = 400;
    AT.title(g, rx, ly - 8, "final loss against head count");
    const g3 = g.append("g").attr("transform", `translate(${rx},${ly})`);
    const bw = 300;
    const yb = d3.scaleLog().domain([Math.max(1e-6, Math.min(...runs.map(r => r.loss)) / 3), Math.max(...runs.map(r => r.loss), Math.log(V)) * 1.4]).range([lh, 0]).clamp(true);
    DL.gridY(g3, yb, bw, 4);
    DL.axisL(g3, yb, 4, "loss", v => DL.fmtE(v, 0));
    runs.forEach((r, i) => {
      const xx = 20 + i * 72;
      g3.append("rect").attr("x", xx).attr("y", yb(Math.max(r.loss, 1e-6))).attr("width", 46)
        .attr("height", lh - yb(Math.max(r.loss, 1e-6))).attr("fill", cols[i]).attr("fill-opacity", i === hShow - 1 ? 0.85 : 0.4);
      AT.note(g3, xx + 4, lh + 13, (i + 1) + " head" + (i ? "s" : ""), i === hShow - 1 ? DC.ink : DC.muted, 9.5);
      AT.note(g3, xx, yb(Math.max(r.loss, 1e-6)) - 5, DL.fmtE(r.loss, 2), cols[i], 9);
    });
    g3.append("line").attr("x1", 0).attr("x2", bw).attr("y1", yb(Math.log(V))).attr("y2", yb(Math.log(V)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    AT.note(g3, bw - 90, yb(Math.log(V)) - 5, "chance", DC.muted, 9);

    const rr3 = cur.A.map(roles).map(v => v.indexOf(Math.max(...v)));
    const dup = new Set(rr3).size < rr3.length;
    El("hd-readout").innerHTML =
      `seed ${seed}, ${steps} Adam steps, vocabulary ${V}, length ${T}, width ${DM}, dₖ ${DH} · final loss ` +
      runs.map((r, i) => `<b>${DL.fmtE(r.loss, 2)}</b> at ${i + 1} head${i ? "s" : ""}`).join(", ") +
      ` against a chance level of <b>${DL.fmt(Math.log(V), 4)}</b>. Going from one head to ${hShow} improves the loss by a factor of ` +
      `<b>${DL.big(runs[0].loss / Math.max(1e-12, cur.loss))}</b>. ` +
      (hShow >= 3
        ? (dup ? `On this seed two of the shown heads took the <b>same</b> relation and the task is solved by another route — heads are ` +
                 `only defined up to permutation and the loss does not care which does what.`
               : `Each shown head has taken a different one of the three relations: ` +
                 rr3.map((k, j) => `head ${j} → ${names[k]}`).join(", ") + `.`)
        : `With ${hShow} head${hShow > 1 ? "s" : ""} there are fewer weight rows than relations, which is the bottleneck: one head ` +
          `produces exactly one convex combination per position and the three facts arrive averaged.`);
  }
  ["hd-h", "hd-s"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#hd-n").on("change", draw);
  d3.select("#hd-go").on("click", draw);
  draw();
})();

/* ═════════ 13 · #ht-svg — the head-dimension trade. The budgets (parameters
   and quadratic MACs) are exactly flat in h; the rank of ONE head's bilinear
   form is d/h. The right panel measures both halves of that: what a single
   rank-dₖ head loses against a full-rank target, and that h such heads
   together lose nothing, because h·(d/h) = d. ═════════════════════════════ */
(function () {
  const svg = d3.select("#ht-svg"); if (svg.empty()) return;
  const W = 760, H = 410, DT = 32;
  const El = id => document.getElementById(id);
  const HS = [1, 2, 4, 8, 16, 32, 64, 128];

  const SPEC = (function () {                       /* a fixed symmetric target */
    const A = AT.gauss(DT, DT, 404, 1 / Math.sqrt(DT));
    const G = DL.zeros2(DT, DT);
    for (let i = 0; i < DT; i++) for (let j = 0; j < DT; j++) G[i][j] = (A[i][j] + A[j][i]) / 2;
    const e = DL.eigSym(G, 60);
    return { G: G, w: e.w, Q: e.Q };
  })();
  function tailResid(k) {                            /* best rank-k approximation */
    let tot = 0, tail = 0;
    for (let i = 0; i < DT; i++) { tot += SPEC.w[i] * SPEC.w[i]; if (i >= k) tail += SPEC.w[i] * SPEC.w[i]; }
    return Math.sqrt(tail / tot);
  }
  function sumResid(h) {                             /* h heads of rank ⌈DT/h⌉, summed */
    const k = Math.max(1, Math.floor(DT / h));
    const S = DL.zeros2(DT, DT);
    for (let j = 0; j < h; j++) for (let t = j * k; t < Math.min(DT, (j + 1) * k); t++)
      for (let a = 0; a < DT; a++) for (let b = 0; b < DT; b++) S[a][b] += SPEC.w[t] * SPEC.Q[a][t] * SPEC.Q[b][t];
    let num = 0, den = 0;
    for (let a = 0; a < DT; a++) for (let b = 0; b < DT; b++) { num += (S[a][b] - SPEC.G[a][b]) ** 2; den += SPEC.G[a][b] ** 2; }
    return Math.sqrt(num / den);
  }

  function draw() {
    const d = +El("ht-d").value, n = Math.pow(2, +El("ht-n").value),
      h = HS[+El("ht-h").value], meas = El("ht-m").value;
    El("ht-nv").textContent = DL.commas(n); El("ht-hv").textContent = h;
    const f = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 46 }), g = f.g;
    const hs = HS.filter(x => x <= d / 4);
    const rows = hs.map(x => {
      const dk = Math.max(1, Math.round(d / x));
      return { h: x, dk: dk,
        p: DL.mhaParams(d, dk, dk, x, x).total,
        q: DL.attnMacs({ n: n, d: d, h: x, g: x, dh: dk, dff: 4 * d, nmat: 2 }).quad };
    });
    const cur = rows.find(r => r.h === h) || rows[rows.length - 1];

    const gw = 320, gh = 250;
    AT.title(g, 0, -10, "what changes with h, at fixed d = " + d);
    const x = d3.scaleLog().domain([1, Math.max(2, hs[hs.length - 1])]).range([0, gw]);
    const lines = [
      { lab: "parameters, one block", col: DC.accent, v: r => r.p },
      { lab: "quadratic MACs at n = " + DL.commas(n), col: DC.violet, v: r => r.q },
      { lab: "head dimension dₖ = d/h", col: DC.a2, v: r => r.dk },
      { lab: "rank bound of one head's relation", col: DC.bad, v: r => r.dk }
    ];
    const all = rows.flatMap(r => lines.map(L => L.v(r)));
    const y = d3.scaleLog().domain([Math.min(...all) / 3, Math.max(...all) * 3]).range([gh, 0]);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "heads h", d3.format("d"));
    DL.axisL(g, y, 5, "", v => DL.big(v));
    lines.forEach((L, i) => DL.curve(g, rows.map(r => [x(r.h), y(L.v(r))]),
      { stroke: L.col, w: i === 3 ? 1.4 : 2, dash: i === 3 ? "3 3" : null }));
    g.append("line").attr("x1", x(h)).attr("x2", x(h)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    DL.legend(g, lines.slice(0, 3).map(L => ({ label: L.lab, color: L.col })), 2, gh + 34, { vertical: true, gap: 13, font: 9.5 });
    AT.note(g, 2, gh + 76, "the first two are exactly flat: h is free in both budgets", DC.good, 9.5);

    /* right — the rank measurement */
    const rx = gw + 62, rw = f.iw - rx;
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    if (meas === "rank") {
      AT.title(g, rx, -10, "eigenvalue spectrum of a full-rank relation, truncated at dₖ");
      const k = Math.max(1, Math.min(DT, Math.round(DT / h)));
      const xs = d3.scaleLinear().domain([0, DT]).range([0, rw - 12]);
      const ys = d3.scaleLinear().domain([0, Math.max(...SPEC.w.map(Math.abs)) * 1.1]).range([gh, 0]);
      DL.gridY(gg, ys, rw - 12, 4);
      DL.axisB(gg, xs, gh, 5, "eigenvalue index");
      DL.axisL(gg, ys, 4, "|λ|");
      gg.selectAll("rect.sp").data(SPEC.w.map((v, i) => [i, Math.abs(v)])).join("rect").attr("class", "sp")
        .attr("x", d => xs(d[0])).attr("y", d => ys(d[1])).attr("width", Math.max(1.5, (rw - 12) / DT - 1))
        .attr("height", d => gh - ys(d[1]))
        .attr("fill", d => d[0] < k ? DC.good : DC.bad).attr("fill-opacity", 0.75);
      gg.append("line").attr("x1", xs(k)).attr("x2", xs(k)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3");
      AT.note(gg, xs(k) + 4, 12, "one head keeps " + k + " of " + DT, DC.ink, 9.5);
      AT.note(gg, 4, gh + 34, "green: kept by one head    red: discarded by one head", DC.muted, 9.5);
    } else {
      AT.title(g, rx, -10, "residual: one head, against h heads summed");
      const xs = d3.scaleLog().domain([1, Math.max(2, hs[hs.length - 1])]).range([0, rw - 12]);
      const pts = hs.map(x2 => ({ h: x2, one: tailResid(Math.max(1, Math.floor(DT / x2))), sum: sumResid(x2) }));
      const ys = d3.scaleLog().domain([1e-16, 2]).range([gh, 0]).clamp(true);
      DL.gridY(gg, ys, rw - 12, 5);
      DL.axisB(gg, xs, gh, 5, "heads h", d3.format("d"));
      DL.axisL(gg, ys, 5, "relative residual", v => DL.fmtE(v, 0));
      DL.curve(gg, pts.map(p => [xs(p.h), ys(Math.max(p.one, 1e-16))]), { stroke: DC.bad, w: 2.2 });
      DL.curve(gg, pts.map(p => [xs(p.h), ys(Math.max(p.sum, 1e-16))]), { stroke: DC.good, w: 2.2 });
      DL.legend(gg, [{ label: "one head alone", color: DC.bad }, { label: "all h heads, summed", color: DC.good }],
        4, gh + 34, { vertical: true, gap: 13, font: 9.5 });
      AT.note(gg, 4, gh + 76, "the total rank h·(d/h) = d is conserved, so the SUM loses nothing", DC.good, 9.5);
    }

    const k = Math.max(1, Math.min(DT, Math.round(DT / h)));
    const one = tailResid(k), sm = sumResid(h);
    const base = rows[0];
    const kvv = DL.kv(g, 0, gh + 96, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("head dimension dₖ = d/h", DL.commas(cur.dk), DC.a2, true);
    kvv("rank bound of one head's bilinear form", DL.commas(cur.dk) + " of " + DL.commas(d), DC.bad, true);
    kvv("parameters, one block", DL.commas(cur.p) + "   (h = 1 gives " + DL.commas(base.p) + ")", cur.p === base.p ? DC.good : DC.a2);
    kvv("quadratic MACs", DL.big(cur.q) + "   (h = 1 gives " + DL.big(base.q) + ")", cur.q === base.q ? DC.good : DC.a2);
    kvv("residual of ONE rank-" + k + " head against a rank-" + DT + " target", DL.fmt(one, 4), DC.bad);
    kvv("residual of " + h + " such heads SUMMED", DL.fmtE(sm, 2), DC.good, true);

    El("ht-readout").innerHTML =
      `d = ${d}, h = ${h}, dₖ = ${cur.dk}, n = ${DL.commas(n)} · the block has <b>${DL.commas(cur.p)}</b> parameters and does ` +
      `<b>${DL.big(cur.q)}</b> quadratic MACs — <b>identical</b> to the single-head configuration, so the head count is free in ` +
      `both budgets. What is not free is the rank: one head's bilinear form has rank at most <b>${DL.commas(cur.dk)}</b> of ` +
      `<b>${DL.commas(d)}</b>. Measured on a scaled-down stand-in of width ${DT} (small enough to draw its spectrum), a single ` +
      `rank-${k} head leaves a relative residual of ` +
      `<b>${DL.fmt(one, 4)}</b>, while <b>${h}</b> such heads summed leave <b>${DL.fmtE(sm, 2)}</b> — round-off. The total rank is ` +
      `conserved; only the partition changes, and §12 showed that it is the partition that binds.`;
  }
  ["ht-n", "ht-h"].forEach(id => d3.select("#" + id).on("input", draw));
  ["ht-d", "ht-m"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 14 · #wo-svg — the block output as a running sum of per-head
   terms. Y = ∑ⱼ Zⱼ·W_Oʲ with W_Oʲ the j-th ROW BLOCK of W_O; the identity is
   exact and the figure prints the residual to prove it. ═══════════════════ */
(function () {
  const svg = d3.select("#wo-svg"); if (svg.empty()) return;
  const W = 760, H = 410, D = 12, DK = 4;
  const El = id => document.getElementById(id);

  function build(n, h) {
    const X = AT.gauss(n, D, 55, 1);
    const P = DL.mhaInit(D, DK, DK, h, { seed: 17 });
    const o = DL.mhaForward(X, P, {});
    const terms = [];
    for (let j = 0; j < h; j++) {
      const Wj = [];
      for (let r = 0; r < DK; r++) Wj.push(P.Wo[j * DK + r].slice());
      terms.push(DL.matmul(o.st.per[j].Z, Wj));
    }
    return { X: X, P: P, Y: o.Y, terms: terms };
  }
  function nrm(M) { let s = 0; M.forEach(r => r.forEach(v => s += v * v)); return Math.sqrt(s); }

  function draw() {
    const h = +El("wo-h").value, n = +El("wo-n").value, ord = El("wo-o").value;
    let k = +El("wo-k").value;
    El("wo-k").max = h; if (k > h) { k = h; El("wo-k").value = h; }
    El("wo-kv").textContent = k; El("wo-hv").textContent = h; El("wo-nv").textContent = n;
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 26, b: 34 }), g = f.g;
    const B = build(n, h);
    let order = B.terms.map((t, i) => i);
    if (ord === "norm") order.sort((a, b) => nrm(B.terms[b]) - nrm(B.terms[a]));
    const cell = d3.scaleSequential(d3.interpolateRdBu).domain([1.2, -1.2]);
    const cw = 11;

    AT.title(g, 0, -10, "the h per-head terms  Aⱼ·X·(W_Vʲ W_Oʲ)");
    order.forEach((j, i) => {
      const x0 = i * 72;
      const inc = i < k;
      AT.note(g, x0, 6, "head " + j, inc ? DC.accent : DC.muted, 9.5);
      for (let r = 0; r < n; r++) AT.strip(g, B.terms[j][r], x0, 12 + r * (cw + 1), cw, cell, { op: inc ? 1 : 0.22, stroke: inc ? "#0f1117" : null });
      AT.note(g, x0, 12 + n * (cw + 1) + 12, "‖·‖ " + DL.fmt(nrm(B.terms[j]), 2), inc ? DC.accent : DC.muted, 9);
    });

    /* running sum, true output, difference */
    const S = DL.zeros2(n, D);
    for (let i = 0; i < k; i++) { const t = B.terms[order[i]]; for (let r = 0; r < n; r++) for (let c = 0; c < D; c++) S[r][c] += t[r][c]; }
    const rx = 460;
    const cols = [["running sum of " + k, S, DC.accent], ["true block output Y", B.Y, DC.good],
                  ["difference", S.map((r, i) => r.map((v, c) => v - B.Y[i][c])), DC.bad]];
    cols.forEach((c, i) => {
      const x0 = rx + i * 100;
      AT.note(g, x0, 6, c[0], c[2], 9.5);
      for (let r = 0; r < n; r++) AT.strip(g, c[1][r], x0, 12 + r * (cw + 1), cw, cell, { stroke: "#0f1117" });
    });
    let mx = 0;
    for (let r = 0; r < n; r++) for (let c = 0; c < D; c++) mx = Math.max(mx, Math.abs(S[r][c] - B.Y[r][c]));

    /* the error curve */
    const cy = 12 + n * (cw + 1) + 40, cw2 = 300, ch = 100;
    AT.title(g, 0, cy - 8, "relative reconstruction error against heads included");
    const gg = g.append("g").attr("transform", `translate(0,${cy})`);
    const errs = [];
    const acc = DL.zeros2(n, D);
    errs.push(1);
    for (let i = 0; i < h; i++) {
      const t = B.terms[order[i]];
      for (let r = 0; r < n; r++) for (let c = 0; c < D; c++) acc[r][c] += t[r][c];
      errs.push(nrm(acc.map((r, a) => r.map((v, b2) => v - B.Y[a][b2]))) / nrm(B.Y));
    }
    const x = d3.scaleLinear().domain([0, h]).range([0, cw2]);
    const y = d3.scaleLog().domain([1e-17, 2]).range([ch, 0]).clamp(true);
    DL.gridY(gg, y, cw2, 4);
    DL.axisB(gg, x, ch, h, "heads included", d3.format("d"));
    DL.axisL(gg, y, 4, "rel. error", v => DL.fmtE(v, 0));
    DL.curve(gg, errs.map((e, i) => [x(i), y(Math.max(e, 1e-17))]), { stroke: DC.accent, w: 2.2 });
    errs.forEach((e, i) => gg.append("circle").attr("cx", x(i)).attr("cy", y(Math.max(e, 1e-17))).attr("r", 2.4)
      .attr("fill", i === k ? DC.a2 : DC.accent));

    const norms = B.terms.map(nrm), tot = nrm(B.Y);
    const kvv = DL.kv(g, 350, cy + 4, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("relative error with " + k + " of " + h + " heads", DL.fmtE(errs[k], 3), errs[k] < 1e-12 ? DC.good : DC.a2, true);
    kvv("max |running sum − true output| at h heads", DL.fmtE(errs[h] * tot, 2), DC.good);
    kvv("largest head's share of ‖Y‖", DL.fmt(100 * Math.max(...norms) / tot, 1) + "%", DC.a2);
    kvv("smallest head's share of ‖Y‖", DL.fmt(100 * Math.min(...norms) / tot, 1) + "%", DC.muted);
    kvv("rank bound of each W_Vʲ W_Oʲ", DK + " of " + D, DC.bad);
    kvv("heads interacting inside this block", "0 — the sum is the only coupling", DC.good, true);

    El("wo-readout").innerHTML =
      `h = ${h}, n = ${n}, ${k} head${k === 1 ? "" : "s"} included, ${ord === "norm" ? "largest first" : "in index order"} · ` +
      `the running sum differs from the true block output by a relative <b>${DL.fmtE(errs[k], 3)}</b>, and at the full ` +
      `${h} heads by <b>${DL.fmtE(errs[h], 2)}</b> — the identity Y = ∑ⱼ Zⱼ·W_Oʲ is <b>exact</b>, not an approximation. ` +
      `The largest head carries <b>${DL.fmt(100 * Math.max(...norms) / tot, 1)}%</b> of the output norm and the smallest ` +
      `<b>${DL.fmt(100 * Math.min(...norms) / tot, 1)}%</b>, so the work is shared unevenly even at identical initialisation scale. ` +
      `Each head's value–output matrix W_Vʲ W_Oʲ has rank at most <b>${DK}</b> of <b>${D}</b>, and no head can see another's ` +
      `output before W_O.`;
  }
  ["wo-k", "wo-h", "wo-n"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#wo-o").on("change", draw);
  draw();
})();

/* ═════════ 15 · #ms-svg — the two masks, combined by addition, and the
   multiply-after-the-softmax bug. Masks come from DL.causalMask / DL.padMask
   / DL.addMasks; nothing is rebuilt here. ═════════════════════════════════ */
(function () {
  const svg = d3.select("#ms-svg"); if (svg.empty()) return;
  const W = 760, H = 410, D = 8;
  const El = id => document.getElementById(id);

  function draw() {
    const n = +El("ms-n").value, kind = El("ms-m").value, how = El("ms-h").value;
    let nv = Math.min(+El("ms-v").value, n);
    El("ms-v").max = n; El("ms-nv").textContent = n; El("ms-vv").textContent = nv;
    const f = DL.frame(svg, W, H, { l: 34, r: 16, t: 26, b: 34 }), g = f.g;

    const X = AT.gauss(n, D, 61, 1), P = DL.mhaInit(D, 4, 4, 1, { seed: 8 });
    const Q = DL.matmul(X, P.Wq), K = DL.matmul(X, P.Wk), V = DL.matmul(X, P.Wv);
    const ok = []; for (let j = 0; j < n; j++) ok.push(j < nv);
    const Mc = DL.causalMask(n), Mp = DL.padMask(n, ok);
    let M = null;
    if (kind === "causal") M = Mc; else if (kind === "pad") M = Mp;
    else if (kind === "both") M = DL.addMasks(Mc, Mp);
    const A0 = DL.sdpa(Q, K, V, {}).A;                 /* unmasked weights */
    let A;
    if (how === "add") A = DL.sdpa(Q, K, V, { mask: M }).A;
    else {                                             /* the bug: zero AFTER */
      A = A0.map((r, i) => r.map((v, j) => (M && !isFinite(M[i][j])) ? 0 : v));
    }
    const sums = A.map(r => r.reduce((a, b) => a + b, 0));

    const cw = Math.min(15, 150 / n);
    const cellS = d3.scaleSequential(d3.interpolateRdBu).domain([3, -3]);
    const cellA = d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...A.map(r => Math.max(...r)))]);
    const panels = [
      ["scores S", DL.sdpa(Q, K, V, {}).S, cellS, null],
      ["causal mask", Mc, null, Mc],
      ["padding mask", Mp, null, Mp],
      ["S + M", M ? DL.addMasks(DL.sdpa(Q, K, V, {}).S, M) : DL.sdpa(Q, K, V, {}).S, cellS, M]
    ];
    panels.forEach((p, i) => {
      const x0 = i * (n * cw + 26);
      AT.note(g, x0, 4, p[0], DC.muted, 9.5);
      if (p[2]) AT.heat(g, p[1].map(r => r.map(v => isFinite(v) ? v : 0)), x0, 10, cw, p[2], { stroke: n <= 14 ? "#0f1117" : null });
      else AT.heat(g, p[1].map(r => r.map(v => isFinite(v) ? 0 : 1)), x0, 10, cw, v => v ? "#2a1a1a" : "#1e3a2a", { stroke: n <= 14 ? "#0f1117" : null });
      if (p[3]) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) if (!isFinite(p[3][a][b]))
        g.append("path").attr("d", `M${x0 + b * cw},${10 + a * cw} l${cw},${cw}`).attr("stroke", DC.bad).attr("stroke-width", 0.6).attr("stroke-opacity", 0.8);
    });

    /* the weights and the row sums */
    const ay = 10 + n * cw + 34;
    AT.note(g, 0, ay - 6, "attention weights A" + (how === "add" ? "" : "  (zeroed AFTER the softmax)"), how === "add" ? DC.accent : DC.bad, 10);
    AT.heat(g, A, 0, ay, cw, cellA, { stroke: n <= 14 ? "#0f1117" : null });
    const bx = n * cw + 16;
    AT.note(g, bx, ay - 6, "row total", DC.muted, 9.5);
    for (let i = 0; i < n; i++) {
      g.append("rect").attr("x", bx).attr("y", ay + i * cw + 1).attr("width", Math.max(0.5, 110 * sums[i])).attr("height", cw - 1)
        .attr("fill", Math.abs(sums[i] - 1) < 1e-9 ? DC.good : DC.bad).attr("fill-opacity", 0.8);
    }
    g.append("line").attr("x1", bx + 110).attr("x2", bx + 110).attr("y1", ay).attr("y2", ay + n * cw)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 2");
    AT.note(g, bx + 114, ay - 6, "1.0", DC.ink, 9);

    /* row total against query index, both applications */
    const rx = 380, gw = f.iw - rx, gh = 150;
    AT.title(g, rx, ay - 6, "row total against query position");
    const gg = g.append("g").attr("transform", `translate(${rx},${ay})`);
    const sAdd = DL.sdpa(Q, K, V, { mask: M }).A.map(r => r.reduce((a, b) => a + b, 0));
    const sMul = A0.map((r, i) => r.reduce((a, v, j) => a + ((M && !isFinite(M[i][j])) ? 0 : v), 0));
    const x = d3.scaleLinear().domain([1, Math.max(2, n)]).range([0, gw - 10]);
    const y = d3.scaleLinear().domain([0, 1.15]).range([gh, 0]);
    DL.gridY(gg, y, gw - 10, 4);
    DL.axisB(gg, x, gh, 5, "query position i");
    DL.axisL(gg, y, 4, "∑ⱼ Aᵢⱼ");
    DL.curve(gg, sAdd.map((v, i) => [x(i + 1), y(isFinite(v) ? v : 0)]), { stroke: DC.good, w: 2.2 });
    DL.curve(gg, sMul.map((v, i) => [x(i + 1), y(v)]), { stroke: DC.bad, w: 2 , dash: "4 3" });
    DL.legend(gg, [{ label: "added before the softmax — always 1", color: DC.good },
                   { label: "multiplied after — worst at small i", color: DC.bad, dash: "4 3" }],
      4, gh + 30, { vertical: true, gap: 13, font: 9.5 });

    const dev = Math.max(...sums.map(v => Math.abs(v - 1)));
    const kvv = DL.kv(g, rx, ay + gh + 66, { keyW: 250, size: 10.5, lead: 14.5 });
    kvv("max |row total − 1|", DL.fmtE(dev, 3), dev < 1e-9 ? DC.good : DC.bad, true);
    kvv("smallest row total", DL.fmt(Math.min(...sums), 6), Math.min(...sums) > 0.999999 ? DC.good : DC.bad);
    kvv("entries the mask forbids", (() => { let c = 0; if (M) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) if (!isFinite(M[a][b])) c++; return DL.commas(c) + " of " + DL.commas(n * n); })());
    kvv("padding keys silenced", DL.commas(n - nv));
    kvv("‖output‖ relative to the correct one", DL.fmt(d3.mean(sums), 6) + "  (should be 1)", dev < 1e-9 ? DC.good : DC.bad);

    El("ms-readout").innerHTML =
      `n = ${n}, ${nv} real tokens, ${kind === "none" ? "no mask" : kind + " mask"}, ${how === "add" ? "added before" : "multiplied after"} the softmax · ` +
      `the largest deviation of a row total from 1 is <b>${DL.fmtE(dev, 3)}</b>` +
      (how === "add"
        ? ` — round-off. Adding <b>−∞</b> before the softmax removes the forbidden entries AND renormalises what is left, in one operation.`
        : ` and the smallest row total is <b>${DL.fmt(Math.min(...sums), 6)}</b>. Zeroing after the softmax removes the entries ` +
          `but does NOT renormalise, so every output row is shrunk by exactly the mass the mask took. Under a causal mask the ` +
          `shrinkage is worst at the START of the sequence — position 1 may read one key and loses everything else, while the ` +
          `last position may read them all and loses almost nothing — so the bug is a monotone, position-dependent rescaling ` +
          `of the output that is severe exactly where the model has least context. Nothing throws.`);
  }
  ["ms-n", "ms-v"].forEach(id => d3.select("#" + id).on("input", draw));
  ["ms-m", "ms-h"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 16 · #mc-svg — the mask constant.
   The leaked mass is computed in double precision and then CLAMPED TO ZERO
   below each format's exponential-underflow gap (745.2 / 104.0 / 17.4, the
   measured values quoted in §16), which is exactly what the hardware does;
   the figure does not emulate float16 arithmetic and does not need to,
   because the only precision-dependent event here is that underflow.  ════ */
(function () {
  const svg = d3.select("#mc-svg"); if (svg.empty()) return;
  const W = 760, H = 410;
  const El = id => document.getElementById(id);
  const GAP = { "64": 745.2, "32": 104.0, "16": 17.4 };
  const FMAX = { "64": 1.7976931348623157e308, "32": 3.4028234663852886e38, "16": 65504 };

  function row(n, seed) { const r = DL.rng(seed), s = []; for (let j = 0; j < n; j++) s.push(DL.randn(r)); return s; }
  function leak(s, half, C, gap) {
    /* half = number of FORBIDDEN keys (the tail of the row) */
    const n = s.length, sm = s.slice();
    for (let j = n - half; j < n; j++) sm[j] += C;
    const mx = Math.max(...sm);
    let tot = 0; const e = new Array(n);
    for (let j = 0; j < n; j++) { const d = sm[j] - mx; e[j] = d < -gap ? 0 : Math.exp(d); tot += e[j]; }
    let lk = 0, mxl = 0;
    for (let j = n - half; j < n; j++) { const v = e[j] / tot; lk += v; mxl = Math.max(mxl, v); }
    return { lk: lk, mxl: mxl, a: e.map(v => v / tot) };
  }

  function draw() {
    const k = +El("mc-c").value, useInf = El("mc-inf").checked, prec = El("mc-p").value,
      n = +El("mc-n").value, all = El("mc-all").checked;
    const C = useInf ? -Infinity : -Math.pow(10, k);
    El("mc-cv").textContent = useInf ? "−∞" : "−" + DL.fmtE(Math.pow(10, k), 1).replace("−", "");
    El("mc-nv").textContent = n; El("mc-c").disabled = useInf;
    const f = DL.frame(svg, W, H, { l: 54, r: 16, t: 26, b: 46 }), g = f.g;
    const s = row(n, 12), forb = all ? n : Math.floor(n / 2);
    const gap = GAP[prec];

    /* left — leaked mass against |C|, per precision */
    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "leaked mass onto forbidden keys, against |mask constant|");
    const ks = DL.linspace(0, 9, 55);
    const x = d3.scaleLog().domain([1, 1e9]).range([0, gw]);
    const y = d3.scaleLog().domain([1e-22, 2]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "|mask constant|", v => DL.fmtE(v, 0));
    DL.axisL(g, y, 5, "leaked mass", v => DL.fmtE(v, 0));
    const cols = { "64": DC.accent, "32": DC.a2, "16": DC.violet };
    ["64", "32", "16"].forEach(p => {
      const pts = ks.map(kk => [x(Math.pow(10, kk)), y(Math.max(leak(s, Math.floor(n / 2), -Math.pow(10, kk), GAP[p]).lk, 1e-22))]);
      DL.curve(g, pts, { stroke: cols[p], w: p === prec ? 2.4 : 1.3, op: p === prec ? 1 : 0.55 });
      if (GAP[p] < 1e9) {
        g.append("line").attr("x1", x(GAP[p])).attr("x2", x(GAP[p])).attr("y1", 0).attr("y2", gh)
          .attr("stroke", cols[p]).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.7);
        AT.note(g, x(GAP[p]) + 3, 12 + (p === "16" ? 0 : p === "32" ? 11 : 22), "float" + p + " underflows at " + GAP[p], cols[p], 8.5);
      }
    });
    if (!useInf) {
      g.append("line").attr("x1", x(DL.clamp(Math.pow(10, k), 1, 1e9))).attr("x2", x(DL.clamp(Math.pow(10, k), 1, 1e9)))
        .attr("y1", 0).attr("y2", gh).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
    }
    AT.note(g, 2, gh + 34, "float16 cannot even hold −1e9: it becomes −∞ on the cast (max " + DL.commas(FMAX["16"]) + ")", DC.violet, 9.5);

    /* right — the row, masked and unmasked */
    const rx = gw + 62, rw = f.iw - rx;
    AT.title(g, rx, -10, all ? "EVERY key in this row is masked" : "half the keys masked");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const un = leak(s, 0, 0, gap).a;
    const isNaNrow = useInf && all;
    const mk = isNaNrow ? null : leak(s, forb, C, gap).a;
    const xb = d3.scaleLinear().domain([0, n]).range([0, rw - 12]);
    const yb = d3.scaleLinear().domain([0, Math.max(...un) * 1.2]).range([gh, 0]);
    DL.gridY(gg, yb, rw - 12, 4);
    DL.axisB(gg, xb, gh, 5, "key j");
    DL.axisL(gg, yb, 4, "weight");
    gg.selectAll("rect.u").data(un.map((v, i) => [i, v])).join("rect").attr("class", "u")
      .attr("x", d => xb(d[0])).attr("y", d => yb(d[1])).attr("width", Math.max(1.2, (rw - 12) / n - 1.5))
      .attr("height", d => gh - yb(d[1])).attr("fill", DC.muted).attr("fill-opacity", 0.5);
    if (mk) gg.selectAll("rect.m").data(mk.map((v, i) => [i, v])).join("rect").attr("class", "m")
      .attr("x", d => xb(d[0]) + 1).attr("y", d => yb(d[1])).attr("width", Math.max(1.2, (rw - 12) / n - 3.5))
      .attr("height", d => gh - yb(d[1])).attr("fill", d => d[0] >= n - forb ? DC.bad : DC.accent).attr("fill-opacity", 0.9);
    else {
      gg.append("text").attr("x", (rw - 12) / 2).attr("y", gh / 2).attr("text-anchor", "middle")
        .attr("font-size", 26).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.bad).text("NaN");
      AT.note(gg, 10, gh / 2 + 24, "every entry is −∞, so the row maximum is −∞ and", DC.bad, 9.5);
      AT.note(gg, 10, gh / 2 + 36, "s − max is undefined. Training stops here, loudly.", DC.bad, 9.5);
    }
    DL.legend(gg, [{ label: "unmasked row", color: DC.muted }, { label: "masked, allowed keys", color: DC.accent },
                   { label: "masked, forbidden keys", color: DC.bad }], 4, gh + 32, { vertical: true, gap: 13, font: 9.5 });

    const L = isNaNrow ? null : leak(s, forb, C, gap);
    let maxDiff = 0;
    if (mk) for (let j = 0; j < n; j++) maxDiff = Math.max(maxDiff, Math.abs(mk[j] - un[j]));
    const kvv = DL.kv(g, 0, gh + 66, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("mask constant", useInf ? "−∞" : "−" + DL.fmtE(Math.pow(10, k), 2), DC.ink, true);
    kvv("keys forbidden", forb + " of " + n, forb === n ? DC.bad : DC.muted);
    kvv("total leaked mass", isNaNrow ? "NaN" : DL.fmtE(L.lk, 3), isNaNrow ? DC.bad : (L.lk === 0 ? DC.good : DC.a2), true);
    kvv("largest single leaked weight", isNaNrow ? "NaN" : DL.fmtE(L.mxl, 3), isNaNrow ? DC.bad : (L.mxl === 0 ? DC.good : DC.a2));
    kvv("row sum", isNaNrow ? "NaN" : DL.fmt(mk.reduce((a, b) => a + b, 0), 8), isNaNrow ? DC.bad : DC.good);
    kvv("max |masked row − unmasked row|", isNaNrow ? "NaN" : DL.fmtE(maxDiff, 3),
      (!isNaNrow && all && maxDiff < 1e-12) ? DC.bad : DC.ink, all);
    kvv("verdict", isNaNrow ? "NaN — the bug is reported" : (all ? "the mask did NOTHING" : "masked correctly"),
      isNaNrow ? DC.a2 : (all ? DC.bad : DC.good), true);

    El("mc-readout").innerHTML =
      `${n} keys, ${forb} forbidden, mask constant ${useInf ? "−∞" : "−" + DL.fmtE(Math.pow(10, k), 2)}, float${prec} ` +
      `(exponential underflows ${gap} below the row maximum) · ` +
      (isNaNrow
        ? `every key is masked, so the row maximum is −∞ and s − max is undefined: the row is <b>NaN</b>, the loss is NaN, and ` +
          `training stops at the exact line that built the mask. That is the desirable outcome.`
        : all
          ? `every key is masked with a FINITE constant, and the resulting row differs from the <b>unmasked</b> row by ` +
            `<b>${DL.fmtE(maxDiff, 3)}</b>. The softmax is shift-invariant, so adding the same constant to every entry of a row ` +
            `is exactly a no-op: the layer attends at full strength to positions that were all supposed to be forbidden, and ` +
            `reports a perfectly normal distribution summing to <b>${DL.fmt(mk.reduce((a, b) => a + b, 0), 6)}</b> while doing it.`
          : `the total leaked mass is <b>${DL.fmtE(L.lk, 3)}</b> and the largest single leaked weight is <b>${DL.fmtE(L.mxl, 3)}</b>. ` +
            (L.lk === 0 ? `That is exactly zero: the constant is far enough below the row maximum for the exponential to underflow in this precision, so it is indistinguishable from −∞.`
                        : `Anything more negative than about −${Math.ceil(gap)} is indistinguishable from −∞ in float${prec}.`));
  }
  ["mc-c", "mc-n"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#mc-p").on("change", draw);
  ["mc-inf", "mc-all"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 17 · #fd-svg — the backward pass, analytic against central
   differences, with the three faults of §17 injectable.
   The CLEAN path calls DL.mhaBackward. The faulty path is a local copy of
   the same code with one line changed; with fault = "none" the two agree to
   0.000e+00 and the readout prints that check, so the faults are known to be
   the ONLY difference. Numeric gradients come from DL.numGradMats. ═══════ */
(function () {
  const svg = d3.select("#fd-svg"); if (svg.empty()) return;
  const W = 760, H = 410, D = 8, DK = 3;
  const El = id => document.getElementById(id);
  const NAMES = ["Wq", "Wk", "Wv", "Wo", "X"];
  const COLS = { Wq: DC.accent, Wk: DC.a2, Wv: DC.violet, Wo: DC.good, X: DC.teal };

  /* a local copy of DL.mhaBackward with one line switchable */
  function backLocal(dY, st, P, fault) {
    const h = st.h, g = st.g, X = st.X, C = st.C;
    const gWo = DL.matmul(DL.transpose(C), dY);
    const dC = DL.matmul(dY, DL.transpose(P.Wo));
    const dZ = DL.splitHeads(dC, h);
    const dQ = [], dKa = [], dVa = [];
    for (let j = 0; j < g; j++) { dKa.push(DL.zeros2(st.K[j].length, st.K[j][0].length)); dVa.push(DL.zeros2(st.V[j].length, st.V[j][0].length)); }
    for (let j = 0; j < h; j++) {
      const kv = DL.kvHeadOf(j, h, g), A = st.per[j].A, sc = st.per[j].scale;
      const V = st.V[kv], K = st.K[kv], Q = st.Q[j];
      const nq = A.length, nk = A[0].length, dv = V[0].length, dk = Q[0].length;
      const dA = DL.zeros2(nq, nk);
      for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
        let s = 0; for (let c = 0; c < dv; c++) s += dZ[j][i][c] * V[k][c];
        dA[i][k] = s;
      }
      for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
        const a = A[i][k]; if (!a) continue;
        for (let c = 0; c < dv; c++) dVa[kv][k][c] += a * dZ[j][i][c];
      }
      const dS = DL.zeros2(nq, nk);
      for (let i = 0; i < nq; i++) {
        let dot = 0; for (let k = 0; k < nk; k++) dot += dA[i][k] * A[i][k];
        if (fault === "norow") dot = 0;                          /* error 1 */
        for (let k = 0; k < nk; k++) dS[i][k] = A[i][k] * (dA[i][k] - dot) * sc;
      }
      if (fault === "remask" && st.mask)                          /* error 3 */
        for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++)
          if (!isFinite(st.mask[i][k])) dS[i][k] = dS[i][k] * 1e-9;
      const dQj = DL.zeros2(nq, dk);
      for (let i = 0; i < nq; i++) for (let k = 0; k < nk; k++) {
        const s = dS[i][k]; if (!s) continue;
        const sK = (fault === "onescale") ? s / sc : s;           /* error 2 */
        for (let c = 0; c < dk; c++) { dQj[i][c] += s * K[k][c]; dKa[kv][k][c] += sK * Q[i][c]; }
      }
      dQ.push(dQj);
    }
    const mQ = DL.mergeHeads(dQ), mK = DL.mergeHeads(dKa), mV = DL.mergeHeads(dVa);
    const gWq = DL.matmul(DL.transpose(X), mQ), gWk = DL.matmul(DL.transpose(X), mK), gWv = DL.matmul(DL.transpose(X), mV);
    const dX = DL.matmul(mQ, DL.transpose(P.Wq));
    const b1 = DL.matmul(mK, DL.transpose(P.Wk)), b2 = DL.matmul(mV, DL.transpose(P.Wv));
    for (let i = 0; i < dX.length; i++) for (let c = 0; c < dX[0].length; c++) dX[i][c] += b1[i][c] + b2[i][c];
    return { Wq: gWq, Wk: gWk, Wv: gWv, Wo: gWo, X: dX };
  }
  function maxAbs(A, B) { let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j])); return m; }
  function maxAbsOf(A) { let m = 0; A.forEach(r => r.forEach(v => m = Math.max(m, Math.abs(v)))); return m; }

  function setup(n, h, g, mask) {
    const r = DL.rng(41);
    const X = DL.zeros2(n, D).map(row => row.map(() => DL.randn(r)));
    const P = DL.mhaInit(D, DK, DK, h, { seed: 42, g: g });
    const T = DL.zeros2(n, D).map(row => row.map(() => DL.randn(r)));
    const M = mask === "causal" ? DL.causalMask(n) : null;
    const loss = () => {
      const o = DL.mhaForward(X, P, { mask: M });
      let L = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) L += 0.5 * (o.Y[i][j] - T[i][j]) ** 2;
      return L / n;
    };
    return { X: X, P: P, T: T, M: M, loss: loss, n: n };
  }

  function draw() {
    const n = +El("fd-n").value; let h = +El("fd-h").value, g = +El("fd-g").value;
    if (g > h) { g = h; El("fd-g").value = h; }
    const mask = El("fd-m").value, eps = +El("fd-e").value, fault = El("fd-f").value;
    El("fd-nv").textContent = n; El("fd-hv").textContent = h; El("fd-gv").textContent = g;
    const f = DL.frame(svg, W, H, { l: 54, r: 16, t: 26, b: 46 }), g2 = f.g;

    const S = setup(n, h, g, mask);
    const fw = DL.mhaForward(S.X, S.P, { mask: S.M });
    const dY = DL.zeros2(n, D).map((row, i) => row.map((_, j) => (fw.Y[i][j] - S.T[i][j]) / n));
    const clean = DL.mhaBackward(dY, fw.st, S.P);
    const local = backLocal(dY, fw.st, S.P, "none");
    let selfChk = 0; NAMES.forEach(k => selfChk = Math.max(selfChk, maxAbs(clean[k], local[k])));
    const ana = fault === "none" ? clean : backLocal(dY, fw.st, S.P, fault);
    const num = DL.numGradMats(S.loss, { Wq: S.P.Wq, Wk: S.P.Wk, Wv: S.P.Wv, Wo: S.P.Wo, X: S.X }, eps);

    /* left — scatter */
    const gw = 300, gh = 250;
    AT.title(g2, 0, -10, "analytic against central differences, every entry");
    const pts = [];
    NAMES.forEach(k => { for (let i = 0; i < num[k].length; i++) for (let j = 0; j < num[k][0].length; j++) pts.push([num[k][i][j], ana[k][i][j], k]); });
    const lim = Math.max(...pts.map(p => Math.max(Math.abs(p[0]), Math.abs(p[1])))) * 1.08;
    const x = d3.scaleLinear().domain([-lim, lim]).range([0, gw]);
    const y = d3.scaleLinear().domain([-lim, lim]).range([gh, 0]);
    DL.gridY(g2, y, gw, 4); DL.gridX(g2, x, gh, 4);
    DL.axisB(g2, x, gh, 4, "numeric", v => DL.fmtE(v, 0));
    DL.axisL(g2, y, 4, "analytic", v => DL.fmtE(v, 0));
    DL.curve(g2, [[x(-lim), y(-lim)], [x(lim), y(lim)]], { stroke: DC.muted, w: 1, dash: "3 3" });
    g2.selectAll("circle.p").data(pts).join("circle").attr("class", "p")
      .attr("cx", d => x(d[0])).attr("cy", d => y(d[1])).attr("r", 2).attr("fill", d => COLS[d[2]]).attr("fill-opacity", 0.75);
    DL.legend(g2, NAMES.map(k => ({ label: k, color: COLS[k] })), 4, gh + 32, { vertical: false, step: 52, font: 9.5 });

    /* right — per-tensor worst relative error, and the ε sweep */
    const rx = gw + 60, rw = f.iw - rx;
    AT.title(g2, rx, -10, "worst relative error per tensor");
    const gg = g2.append("g").attr("transform", `translate(${rx},0)`);
    const rels = NAMES.map(k => ({ k: k, v: maxAbs(ana[k], num[k]) / Math.max(1e-300, maxAbsOf(num[k])) }));
    const worst = rels.reduce((a, b) => a.v > b.v ? a : b);
    const xb = d3.scaleLog().domain([1e-12, Math.max(1e-6, ...rels.map(r => r.v)) * 6]).range([0, rw - 20]);
    rels.forEach((r, i) => {
      const yy = 8 + i * 24;
      AT.note(gg, 0, yy + 11, r.k, COLS[r.k], 10.5);
      gg.append("rect").attr("x", 30).attr("y", yy).attr("width", Math.max(1.5, xb(Math.max(r.v, 1e-12)))).attr("height", 14)
        .attr("rx", 2).attr("fill", r.v < 1e-6 ? DC.good : DC.bad).attr("fill-opacity", 0.8);
      AT.note(gg, 30 + Math.max(1.5, xb(Math.max(r.v, 1e-12))) + 6, yy + 11, DL.fmtE(r.v, 2), r.v < 1e-6 ? DC.good : DC.bad, 9.5);
    });
    gg.append("rect").attr("x", 30).attr("y", 0).attr("width", xb(1e-6) - 0).attr("height", 5 * 24 + 4)
      .attr("fill", DC.good).attr("fill-opacity", 0.06);
    AT.note(gg, 34, 5 * 24 + 18, "a correct implementation lives in the shaded band", DC.good, 9.5);

    /* ε sweep */
    const sy = 150, sh = 100;
    AT.title(gg, 0, sy - 8, "worst relative error against the step size ε");
    const g3 = gg.append("g").attr("transform", `translate(0,${sy})`);
    const es = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9];
    const sw = es.map(e => {
      const nn = DL.numGradMats(S.loss, { Wq: S.P.Wq, Wk: S.P.Wk }, e);
      let m = 0; ["Wq", "Wk"].forEach(k => m = Math.max(m, maxAbs(ana[k], nn[k]) / Math.max(1e-300, maxAbsOf(nn[k]))));
      return { e: e, v: m };
    });
    const xs = d3.scaleLog().domain([1e-9, 1e-2]).range([0, rw - 20]);
    const ys = d3.scaleLog().domain([Math.min(...sw.map(s2 => s2.v)) / 3, Math.max(...sw.map(s2 => s2.v)) * 3]).range([sh, 0]);
    DL.gridY(g3, ys, rw - 20, 3);
    DL.axisB(g3, xs, sh, 4, "ε", v => DL.fmtE(v, 0));
    DL.axisL(g3, ys, 3, "rel. err", v => DL.fmtE(v, 0));
    DL.curve(g3, sw.map(s2 => [xs(s2.e), ys(s2.v)]), { stroke: DC.accent, w: 2 });
    sw.forEach(s2 => g3.append("circle").attr("cx", xs(s2.e)).attr("cy", ys(s2.v)).attr("r", 2.4)
      .attr("fill", s2.e === eps ? DC.a2 : DC.accent));
    AT.note(g3, 4, sh + 30, "left arm: round-off.  right arm: O(ε²) truncation.", DC.muted, 9);

    const kvv = DL.kv(g2, 0, gh + 66, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("worst relative error", DL.fmtE(worst.v, 3) + "  on " + worst.k, worst.v < 1e-6 ? DC.good : DC.bad, true);
    kvv("verdict", worst.v < 1e-6 ? "PASS" : "FAIL — the gradient is wrong", worst.v < 1e-6 ? DC.good : DC.bad, true);
    kvv("local copy vs DL.mhaBackward (fault off)", DL.fmtE(selfChk, 1), selfChk === 0 ? DC.good : DC.bad);
    kvv("parameters checked", DL.commas(pts.length) + " entries");
    kvv("step size ε", DL.fmtE(eps, 0));

    El("fd-readout").innerHTML =
      `n = ${n}, h = ${h}, g = ${g}, ${mask === "causal" ? "causal" : "unmasked"}, ε = ${DL.fmtE(eps, 0)}, ` +
      `${fault === "none" ? "no fault" : "fault: " + El("fd-f").selectedOptions[0].textContent} · worst relative error ` +
      `<b>${DL.fmtE(worst.v, 3)}</b> on <b>${worst.k}</b> over <b>${DL.commas(pts.length)}</b> checked entries — ` +
      `<b>${worst.v < 1e-6 ? "PASS" : "FAIL"}</b>. ` +
      (fault === "none"
        ? `The local fault-injectable copy of the backward pass agrees with DL.mhaBackward to <b>${DL.fmtE(selfChk, 1)}</b>, so ` +
          `any difference below is the injected fault and nothing else.`
        : (rels.some(r => r.v > 1e-6)
          ? `Note which tensors moved: ` + rels.filter(r => r.v > 1e-6).map(r => `<b>${r.k}</b> (${DL.fmtE(r.v, 1)})`).join(", ") +
            `, and which did not — the structure of the failure identifies the error.`
          : `<b>Nothing moved.</b> With an −∞ mask the forbidden weights are exactly zero, so ∂L/∂S is exactly zero there and ` +
            `re-masking it is a no-op. That is error 3's whole character: it is invisible until someone replaces the −∞ with a ` +
            `finite constant, at which point it starts multiplying a non-zero leaked gradient by a second copy of the mask.`));
  }
  ["fd-n", "fd-h", "fd-g"].forEach(id => d3.select("#" + id).on("input", draw));
  ["fd-m", "fd-e", "fd-f"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ page-local: the published configurations used by figures 18–20,
   25 and 27. Every field is taken from the model's own published shape and
   is quoted in the prose of §08, §19 and §25. ════════════════════════════ */
const CFG = {
  gpt2s:  { lab: "GPT-2 small",  d: 768,   h: 12,  g: 12, dh: 64,  dff: 3072,  nmat: 2, L: 12, ctx: 1024,   P: 124e6,  kind: "gqa" },
  gpt2xl: { lab: "GPT-2 XL",     d: 1600,  h: 25,  g: 25, dh: 64,  dff: 6400,  nmat: 2, L: 48, ctx: 1024,   P: 1.558e9, kind: "gqa" },
  l27b:   { lab: "Llama-2 7B",   d: 4096,  h: 32,  g: 32, dh: 128, dff: 11008, nmat: 3, L: 32, ctx: 4096,   P: 6.74e9, kind: "gqa" },
  l38b:   { lab: "Llama-3 8B",   d: 4096,  h: 32,  g: 8,  dh: 128, dff: 14336, nmat: 3, L: 32, ctx: 8192,   P: 8.03e9, kind: "gqa" },
  mis7b:  { lab: "Mistral-7B",   d: 4096,  h: 32,  g: 8,  dh: 128, dff: 14336, nmat: 3, L: 32, ctx: 32768,  P: 7.24e9, kind: "gqa" },
  l370b:  { lab: "Llama-3 70B",  d: 8192,  h: 64,  g: 8,  dh: 128, dff: 28672, nmat: 3, L: 80, ctx: 8192,   P: 70.6e9, kind: "gqa" },
  dsv3:   { lab: "DeepSeek-V3",  d: 7168,  h: 128, g: 128, dh: 128, dff: 18432, nmat: 3, L: 61, ctx: 163840, P: 671e9, kind: "mla", dc: 512, dr: 64 }
};
/* cache elements per token for a CFG entry, via DL.kvCacheElems */
function cfgCache(c) {
  return c.kind === "mla" ? DL.kvCacheElems({ kind: "mla", dc: c.dc, dr: c.dr, L: c.L })
                          : DL.kvCacheElems({ h: c.h, g: c.g, dh: c.dh, L: c.L, kind: "gqa" });
}
const HW = {
  a100: { lab: "A100 80GB SXM", flops: 312e12, bw: 2039e9, mem: 80 },
  h100: { lab: "H100 SXM5",     flops: 989.4e12, bw: 3350e9, mem: 80 },
  h200: { lab: "H200 SXM",      flops: 989.4e12, bw: 4800e9, mem: 141 },
  rtx:  { lab: "RTX 4090",      flops: 165.2e12, bw: 1008e9, mem: 24 }
};

/* ═════════ 18 · #co-svg — the three MAC terms of one layer against n.
   Every count is DL.attnMacs; nothing is recomputed here. MACs, not FLOPs. */
(function () {
  const svg = d3.select("#co-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);

  function draw() {
    const c = CFG[El("co-p").value], causal = El("co-c").checked,
      n = Math.pow(2, +El("co-n").value), view = El("co-v").value;
    El("co-nv").textContent = DL.commas(n);
    const f = DL.frame(svg, W, H, { l: 58, r: 16, t: 26, b: 46 }), g = f.g;
    const opt = k => ({ n: k, d: c.d, h: c.h, g: c.g, dh: c.dh, dff: c.dff, nmat: c.nmat, causal: causal });
    const ns = []; for (let e = 7; e <= 18; e += 0.25) ns.push(Math.round(Math.pow(2, e)));
    const rows = ns.map(k => Object.assign({ n: k }, DL.attnMacs(opt(k))));
    const cur = DL.attnMacs(opt(n));
    const xp = DL.macCrossover(opt(1), "proj"), xf = DL.macCrossover(opt(1), "ffn"), xr = DL.macCrossover(opt(1), "rest");

    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "MACs per layer, one sequence");
    const x = d3.scaleLog().domain([128, 262144]).range([0, gw]);
    const y = d3.scaleLog().domain([rows[0].quad / 3, rows[rows.length - 1].total * 2]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "sequence length n", v => DL.big(v));
    DL.axisL(g, y, 5, "MACs", v => DL.big(v));
    const series = [["projections  (∝ n)", r => r.proj, DC.accent], ["quadratic  (∝ n²)", r => r.quad, DC.bad],
                    ["FFN  (∝ n)", r => r.ffn, DC.violet], ["total", r => r.total, DC.ink]];
    series.forEach((s, i) => DL.curve(g, rows.map(r => [x(r.n), y(DL.clamp(s[1](r), 1, 1e30))]),
      { stroke: s[2], w: i === 3 ? 1.2 : 2, dash: i === 3 ? "4 3" : null }));
    [[xp, "vs projections", DC.accent], [xf, "vs FFN", DC.violet], [xr, "vs BOTH", DC.a2]].forEach((m, i) => {
      if (m[0] < 128 || m[0] > 262144) return;
      g.append("line").attr("x1", x(m[0])).attr("x2", x(m[0])).attr("y1", 0).attr("y2", gh)
        .attr("stroke", m[2]).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.8);
      AT.note(g, x(m[0]) + 3, 12 + i * 11, DL.commas(m[0]) + "  " + m[1], m[2], 8.5);
    });
    g.append("line").attr("x1", x(DL.clamp(n, 128, 262144))).attr("x2", x(DL.clamp(n, 128, 262144)))
      .attr("y1", 0).attr("y2", gh).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.55);
    DL.legend(g, series.map(s => ({ label: s[0], color: s[2] })), 2, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    /* right — shares */
    const rx = gw + 62, rw = f.iw - rx;
    AT.title(g, rx, -10, view === "share" ? "share of the layer" : "share of the layer, stacked");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const xs = d3.scaleLog().domain([128, 262144]).range([0, rw - 12]);
    const ys = d3.scaleLinear().domain([0, 1]).range([gh, 0]);
    DL.gridY(gg, ys, rw - 12, 5);
    DL.axisB(gg, xs, gh, 5, "sequence length n", v => DL.big(v));
    DL.axisL(gg, ys, 5, "share", d3.format(".0%"));
    const stack = [["quadratic", r => r.quad / r.total, DC.bad], ["projections", r => (r.quad + r.proj) / r.total, DC.accent],
                   ["FFN", () => 1, DC.violet]];
    stack.slice().reverse().forEach(s => {
      const pts = rows.map(r => [xs(r.n), ys(s[1](r))]).concat([[xs(262144), ys(0)], [xs(128), ys(0)]]);
      DL.curve(gg, pts, { stroke: s[2], w: 1.2, fill: s[2], fillOp: 0.35 });
    });
    gg.append("line").attr("x1", 0).attr("x2", rw - 12).attr("y1", ys(0.5)).attr("y2", ys(0.5))
      .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    AT.note(gg, 4, ys(0.5) - 5, "half the layer", DC.ink, 9);
    if (c.ctx >= 128 && c.ctx <= 262144) {
      gg.append("line").attr("x1", xs(c.ctx)).attr("x2", xs(c.ctx)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", DC.good).attr("stroke-width", 1.4);
      AT.note(gg, xs(c.ctx) + 4, gh - 8, "its trained context, " + DL.commas(c.ctx), DC.good, 9);
    }
    DL.legend(gg, stack.map(s => ({ label: s[0], color: s[2], op: 0.5 })), 4, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    const at = DL.attnMacs(opt(c.ctx));
    const kvv = DL.kv(g, 0, gh + 92, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("projections at n = " + DL.commas(n), DL.big(cur.proj), DC.accent);
    kvv("quadratic at n = " + DL.commas(n), DL.big(cur.quad), DC.bad, true);
    kvv("FFN at n = " + DL.commas(n), DL.big(cur.ffn), DC.violet);
    kvv("quadratic share of the layer", DL.fmt(100 * cur.quad / cur.total, 2) + "%", cur.quad / cur.total > 0.5 ? DC.bad : DC.good, true);
    kvv("quadratic overtakes the projections at n =", DL.commas(xp) + (c.g < c.h ? "   (2d would be " + DL.commas(2 * c.d) + ")" : ""), DC.accent);
    kvv("quadratic overtakes the FFN at n =", DL.commas(xf) + "   (nmat·d_ff/2 = " + DL.commas(c.nmat * c.dff / 2) + ")", DC.violet);
    kvv("quadratic overtakes BOTH at n =", DL.commas(xr), DC.a2, true);
    kvv("quadratic share at its trained context " + DL.commas(c.ctx), DL.fmt(100 * at.quad / at.total, 2) + "%", DC.good, true);

    El("co-readout").innerHTML =
      `${c.lab}${causal ? ", causal" : ""} at n = ${DL.commas(n)} · projections <b>${DL.big(cur.proj)}</b>, quadratic ` +
      `<b>${DL.big(cur.quad)}</b>, FFN <b>${DL.big(cur.ffn)}</b> MACs — the quadratic term is <b>${DL.fmt(100 * cur.quad / cur.total, 2)}%</b> ` +
      `of the layer. It overtakes the projections at n = <b>${DL.commas(xp)}</b>, the FFN at <b>${DL.commas(xf)}</b> and both ` +
      `together at <b>${DL.commas(xr)}</b>. At this model's own trained context of ${DL.commas(c.ctx)} tokens it is ` +
      `<b>${DL.fmt(100 * at.quad / at.total, 2)}%</b>` +
      (at.quad / at.total < 0.5 ? ` — a minority of the arithmetic, in the architecture everyone calls quadratic.` : `.`);
  }
  d3.select("#co-n").on("input", draw);
  ["co-p", "co-v"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#co-c").on("change", draw);
  draw();
})();

/* ═════════ 19 · #cr-svg — the crossover, for an arbitrary configuration.
   Crossings from DL.macCrossover, which bisects DL.attnMacs, so the GQA and
   h·dₖ ≠ d cases are handled without a special case. ═════════════════════ */
(function () {
  const svg = d3.select("#cr-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);

  function draw() {
    const d = +El("cr-d").value, dff = +El("cr-ff").value, nmat = +El("cr-nm").value, causal = El("cr-c").checked;
    let h = +El("cr-h").value, g = +El("cr-g").value;
    if (g > h) { g = h; El("cr-g").value = h; }
    const dh = Math.max(1, Math.round(d / h));
    El("cr-dv").textContent = d; El("cr-ffv").textContent = dff;
    El("cr-hv").textContent = h; El("cr-gv").textContent = g;
    const f = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 46 }), g2 = f.g;
    const opt = k => ({ n: k, d: d, h: h, g: g, dh: dh, dff: dff, nmat: nmat, causal: causal });
    const xp = DL.macCrossover(opt(1), "proj"), xf = DL.macCrossover(opt(1), "ffn"), xr = DL.macCrossover(opt(1), "rest");

    const gw = 320, gh = 250;
    AT.title(g2, 0, -10, "quadratic share of one layer, against n");
    const ns = []; for (let e = 7; e <= 20; e += 0.2) ns.push(Math.round(Math.pow(2, e)));
    const x = d3.scaleLog().domain([128, 1048576]).range([0, gw]);
    const y = d3.scaleLinear().domain([0, 1]).range([gh, 0]);
    DL.gridY(g2, y, gw, 5);
    DL.axisB(g2, x, gh, 5, "sequence length n", v => DL.big(v));
    DL.axisL(g2, y, 5, "quadratic share", d3.format(".0%"));
    DL.curve(g2, ns.map(k => { const m = DL.attnMacs(opt(k)); return [x(k), y(m.quad / m.total)]; }), { stroke: DC.bad, w: 2.4 });
    g2.append("line").attr("x1", 0).attr("x2", gw).attr("y1", y(0.5)).attr("y2", y(0.5))
      .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    [[xp, "overtakes the projections", DC.accent, 0], [xf, "overtakes the FFN", DC.violet, 1], [xr, "overtakes BOTH", DC.a2, 2]].forEach(m => {
      if (m[0] < 128 || m[0] > 1048576) return;
      g2.append("line").attr("x1", x(m[0])).attr("x2", x(m[0])).attr("y1", 0).attr("y2", gh)
        .attr("stroke", m[2]).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.85);
      AT.note(g2, x(m[0]) + 3, 12 + m[3] * 11, DL.commas(m[0]) + " — " + m[1], m[2], 8.5);
    });

    /* right — published configurations against the closed form */
    const rx = gw + 66, rw = f.iw - rx;
    AT.title(g2, rx, -10, "published configurations: crossover against trained context");
    const gg = g2.append("g").attr("transform", `translate(${rx},0)`);
    const pts = Object.keys(CFG).map(k => {
      const c = CFG[k], o = { n: 1, d: c.d, h: c.h, g: c.g, dh: c.dh, dff: c.dff, nmat: c.nmat, causal: causal };
      return { lab: c.lab, d: c.d, cross: DL.macCrossover(o, "rest"), ctx: c.ctx };
    });
    const xs = d3.scaleLog().domain([512, 20000]).range([0, rw - 14]);
    const ys = d3.scaleLog().domain([512, 200000]).range([gh, 0]);
    DL.gridY(gg, ys, rw - 14, 4);
    DL.axisB(gg, xs, gh, 4, "d_model", v => DL.commas(v));
    DL.axisL(gg, ys, 4, "tokens", v => DL.big(v));
    pts.forEach(p => {
      gg.append("line").attr("x1", xs(p.d)).attr("x2", xs(p.d)).attr("y1", ys(p.ctx)).attr("y2", ys(p.cross))
        .attr("stroke", DC.line).attr("stroke-width", 1);
      gg.append("circle").attr("cx", xs(p.d)).attr("cy", ys(p.cross)).attr("r", 3.4).attr("fill", DC.bad);
      gg.append("circle").attr("cx", xs(p.d)).attr("cy", ys(p.ctx)).attr("r", 3.4).attr("fill", DC.good);
      AT.note(gg, xs(p.d) + 5, ys(p.cross) - 3, p.lab, DC.muted, 8.5);
    });
    DL.legend(gg, [{ label: "crossover: quadratic = everything else", color: DC.bad },
                   { label: "the context it was trained at", color: DC.good }],
      4, gh + 34, { vertical: true, gap: 13, font: 9.5 });
    const below = pts.filter(p => p.ctx < p.cross), above = pts.filter(p => p.ctx >= p.cross);
    const rat = below.map(p => p.cross / p.ctx);
    AT.note(gg, 4, gh + 76, below.length + " of " + pts.length + " trained contexts sit BELOW their crossover, by " +
      DL.fmt(Math.min(...rat), 1) + "× to " + DL.fmt(Math.max(...rat), 1) + "×", DC.good, 9.5);
    if (above.length) AT.note(gg, 4, gh + 88, "above it: " + above.map(p => p.lab + " (" + DL.fmt(p.ctx / p.cross, 1) + "× its crossover)").join(", "), DC.bad, 9);

    const ref = DL.attnMacs(opt(8192));
    const kvv = DL.kv(g2, 0, gh + 92, { keyW: 288, size: 10.5, lead: 14.5 });
    kvv("closed form  n* = 2d  (quadratic = projections, g = h)", DL.commas(2 * d), DC.accent);
    kvv("   measured, at g = " + g, DL.commas(xp), g === h ? DC.good : DC.a2, true);
    kvv("closed form  n* = nmat·d_ff/2  (quadratic = FFN)", DL.commas(nmat * dff / 2), DC.violet);
    kvv("   measured", DL.commas(xf), Math.abs(xf - nmat * dff / 2) <= 2 ? DC.good : DC.a2, true);
    kvv("quadratic = EVERYTHING ELSE at n =", DL.commas(xr), DC.a2, true);
    kvv("   that is this many times d", DL.fmt(xr / d, 2) + "×");
    kvv("quadratic share at n = 8192", DL.fmt(100 * ref.quad / ref.total, 2) + "%", ref.quad / ref.total > 0.5 ? DC.bad : DC.good);

    El("cr-readout").innerHTML =
      `d = ${DL.commas(d)}, d_ff = ${DL.commas(dff)} × ${nmat}, h = ${h}, g = ${g}${causal ? ", causal" : ""} · the quadratic term ` +
      `overtakes the projections at n = <b>${DL.commas(xp)}</b>, the FFN at <b>${DL.commas(xf)}</b> ` +
      `(closed form nmat·d_ff/2 = <b>${DL.commas(nmat * dff / 2)}</b>) and both together at <b>${DL.commas(xr)}</b> — ` +
      `<b>${DL.fmt(xr / d, 2)}×</b> the model width. At n = 8192 the quadratic term is <b>${DL.fmt(100 * ref.quad / ref.total, 2)}%</b> ` +
      `of a layer. Note that the FFN crossing does not contain d at all: <b>the part of the model that decides when attention's ` +
      `quadratic cost matters is the feedforward width</b>, which has nothing to do with attention.`;
  }
  ["cr-d", "cr-ff", "cr-h", "cr-g"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#cr-nm").on("change", draw);
  d3.select("#cr-c").on("change", draw);
  draw();
})();

/* ═════════ 20 · #mm-svg — activation memory. The quadratic term here has no
   crossover trick to save it: h·n² beats 4nd at n = 4dₖ·(…) and then never
   looks back. ═════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#mm-svg"); if (svg.empty()) return;
  const W = 760, H = 400, GiB = 1073741824;
  const El = id => document.getElementById(id);

  function draw() {
    const c = CFG[El("mm-p").value], B = +El("mm-b").value, bpe = +El("mm-t").value,
      budget = +El("mm-g").value * GiB, scope = El("mm-s").value;
    El("mm-bv").textContent = B; El("mm-gv").textContent = El("mm-g").value;
    const f = DL.frame(svg, W, H, { l: 60, r: 16, t: 26, b: 46 }), g = f.g;
    const mult = B * bpe * (scope === "all" ? c.L : 1);
    const lin = n => 4 * n * c.d * mult;
    const quad = n => 2 * c.h * n * n * mult;          /* S and A both held */
    const tot = n => lin(n) + quad(n);

    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "activation bytes, " + (scope === "all" ? "all " + c.L + " layers" : "one layer") + ", batch " + B);
    const ns = []; for (let e = 8; e <= 18; e += 0.2) ns.push(Math.round(Math.pow(2, e)));
    const x = d3.scaleLog().domain([256, 262144]).range([0, gw]);
    const y = d3.scaleLog().domain([lin(256) / 3, tot(262144) * 2]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "sequence length n", v => DL.big(v));
    DL.axisL(g, y, 5, "bytes", v => DL.big(v));
    DL.curve(g, ns.map(n => [x(n), y(lin(n))]), { stroke: DC.accent, w: 2 });
    DL.curve(g, ns.map(n => [x(n), y(quad(n))]), { stroke: DC.bad, w: 2 });
    DL.curve(g, ns.map(n => [x(n), y(tot(n))]), { stroke: DC.ink, w: 1.2, dash: "4 3" });
    g.append("line").attr("x1", 0).attr("x2", gw).attr("y1", y(budget)).attr("y2", y(budget))
      .attr("stroke", DC.good).attr("stroke-width", 1.6);
    AT.note(g, 4, y(budget) - 5, "budget " + El("mm-g").value + " GiB", DC.good, 9.5);
    /* longest n that fits, with and without materialising */
    const nMat = Math.floor((-4 * c.d + Math.sqrt(16 * c.d * c.d + 8 * c.h * budget / mult)) / (4 * c.h));
    const nFla = Math.floor(budget / (4 * c.d * mult));
    [[nMat, "materialised", DC.bad], [nFla, "not materialised", DC.accent]].forEach((m, i) => {
      if (m[0] < 256 || m[0] > 262144) return;
      g.append("line").attr("x1", x(m[0])).attr("x2", x(m[0])).attr("y1", 0).attr("y2", gh)
        .attr("stroke", m[2]).attr("stroke-dasharray", "2 3");
      AT.note(g, x(m[0]) + 3, 12 + i * 11, DL.commas(m[0]) + "  " + m[1], m[2], 8.5);
    });
    DL.legend(g, [{ label: "Q, K, V, output  (∝ n)", color: DC.accent },
                  { label: "score + weight matrices  (∝ n²)", color: DC.bad },
                  { label: "total", color: DC.ink, dash: "4 3" }],
      2, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    /* right — decomposition at a few lengths */
    const rx = gw + 66, rw = f.iw - rx;
    AT.title(g, rx, -10, "the split, at four lengths");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const shows = [1024, 4096, 16384, 65536];
    const yb = d3.scaleLinear().domain([0, 1]).range([gh, 0]);
    DL.axisL(gg, yb, 5, "share", d3.format(".0%"));
    shows.forEach((n, i) => {
      const xx = 18 + i * 76, l = lin(n), q = quad(n), t = l + q;
      gg.append("rect").attr("x", xx).attr("y", yb(1)).attr("width", 50).attr("height", gh - yb(q / t))
        .attr("fill", DC.bad).attr("fill-opacity", 0.6);
      gg.append("rect").attr("x", xx).attr("y", yb(l / t)).attr("width", 50).attr("height", gh - yb(l / t))
        .attr("fill", DC.accent).attr("fill-opacity", 0.6);
      AT.note(gg, xx, gh + 14, DL.big(n), DC.muted, 9.5);
      AT.note(gg, xx, gh + 26, DL.fmt(100 * q / t, 1) + "% n²", DC.bad, 9);
      AT.note(gg, xx, yb(1) - 5, DL.fmt(t / GiB, t / GiB >= 100 ? 0 : 1) + " GiB", DC.ink, 9);
    });

    const nHere = Math.pow(2, 13);
    const kvv = DL.kv(g, 0, gh + 92, { keyW: 300, size: 10.5, lead: 14.5 });
    kvv("longest n that fits, score matrix materialised", DL.commas(nMat), DC.bad, true);
    kvv("longest n that fits, not materialised", DL.commas(nFla), DC.accent, true);
    kvv("   ratio", DL.fmt(nFla / Math.max(1, nMat), 1) + "×", DC.good, true);
    kvv("at n = 8192: quadratic part", DL.big(quad(8192)) + " bytes  =  " + DL.fmt(quad(8192) / GiB, 3) + " GiB", DC.bad);
    kvv("at n = 8192: linear part", DL.big(lin(8192)) + " bytes  =  " + DL.fmt(lin(8192) / GiB, 4) + " GiB", DC.accent);
    kvv("   the saving from not materialising", DL.fmt(tot(8192) / lin(8192), 1) + "×", DC.good);
    kvv("saving formula  (2h·n + 4d) / 4d  at n = 8192", DL.fmt((2 * c.h * 8192 + 4 * c.d) / (4 * c.d), 1) + "×", DC.muted);

    El("mm-readout").innerHTML =
      `${c.lab}, batch ${B}, ${bpe} bytes per element, ${scope === "all" ? "all " + c.L + " layers" : "one layer"} · under a ` +
      `<b>${El("mm-g").value} GiB</b> budget the longest sequence that fits is <b>${DL.commas(nMat)}</b> tokens with the score ` +
      `matrix materialised and <b>${DL.commas(nFla)}</b> without it — a factor of <b>${DL.fmt(nFla / Math.max(1, nMat), 1)}×</b>. ` +
      `At n = 8192 the quadratic part alone is <b>${DL.fmt(quad(8192) / GiB, 3)} GiB</b> against <b>${DL.fmt(lin(8192) / GiB, 4)} GiB</b> ` +
      `for everything else, so not materialising it saves <b>${DL.fmt(tot(8192) / lin(8192), 1)}×</b>. The saving is ` +
      `(2h·n + 4d)/4d, which <b>grows with n</b> — which is why the fix had to be a kernel and not a bigger machine.`;
  }
  ["mm-b", "mm-g"].forEach(id => d3.select("#" + id).on("input", draw));
  ["mm-p", "mm-t", "mm-s"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 21 · #tm-svg — attention mixes tokens and leaves features alone.
   The additivity probe is exact: with n = 1 the softmax is the identity on a
   one-element row, so a stack of attention layers is a product of matrices.
   The "frozen weights" number isolates the softmax's contribution when
   n > 1. ═════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#tm-svg"); if (svg.empty()) return;
  const W = 760, H = 410, D = 8;
  const El = id => document.getElementById(id);

  function layers(L) {
    const P = [], F = [];
    for (let l = 0; l < L; l++) {
      P.push(DL.mhaInit(D, 4, 4, 2, { seed: 300 + l }));
      F.push({ W1: AT.gauss(D, 4 * D, 400 + l, Math.sqrt(2 / D)), W2: AT.gauss(4 * D, D, 500 + l, Math.sqrt(2 / (4 * D))) });
    }
    return { P: P, F: F };
  }
  function run(X, S, L, ffn, frozen) {
    let Z = X.map(r => r.slice());
    const As = [];
    for (let l = 0; l < L; l++) {
      let Y;
      if (frozen && frozen[l]) {
        /* reuse the stored weights: a purely linear value path */
        const V = DL.splitHeads(DL.matmul(Z, S.P[l].Wv), 2);
        const Zs = frozen[l].map((A, j) => DL.matmul(A, V[j]));
        Y = DL.matmul(DL.mergeHeads(Zs), S.P[l].Wo);
      } else {
        const o = DL.mhaForward(Z, S.P[l], {});
        As.push(o.st.per.map(p => p.A));
        Y = o.Y;
      }
      Z = Y;
      if (ffn) {
        const Hh = DL.matmul(Z, S.F[l].W1).map(r => r.map(v => Math.max(0, v)));
        Z = DL.matmul(Hh, S.F[l].W2);
      }
    }
    return { Z: Z, As: As };
  }
  function relDiff(A, B) {
    let n = 0, dd = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) { dd += (A[i][j] - B[i][j]) ** 2; n += B[i][j] * B[i][j]; }
    return Math.sqrt(dd / Math.max(1e-300, n));
  }

  function draw() {
    const L = +El("tm-L").value, n = +El("tm-n").value, a = +El("tm-a").value,
      ffn = El("tm-ffn").checked, probe = El("tm-p").value;
    const b = 1 - a;
    El("tm-Lv").textContent = L; El("tm-nv").textContent = n; El("tm-av").textContent = DL.fmt(a, 2);
    const f = DL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 40 }), g = f.g;
    const S = layers(L);
    const X1 = AT.gauss(n, D, 601, 1), X2 = AT.gauss(n, D, 602, 1);
    const Xm = X1.map((r, i) => r.map((v, c) => a * v + b * X2[i][c]));
    const f1 = run(X1, S, L, ffn), f2 = run(X2, S, L, ffn), fm = run(Xm, S, L, ffn);
    const comb = f1.Z.map((r, i) => r.map((v, c) => a * v + b * f2.Z[i][c]));
    const res = relDiff(fm.Z, comb);
    /* the same test with the attention weights FROZEN at the X1 pattern */
    const fz1 = run(X1, S, L, ffn, f1.As), fz2 = run(X2, S, L, ffn, f1.As), fzm = run(Xm, S, L, ffn, f1.As);
    const combF = fz1.Z.map((r, i) => r.map((v, c) => a * v + b * fz2.Z[i][c]));
    const resF = relDiff(fzm.Z, combF);

    const cell = d3.scaleSequential(d3.interpolateRdBu).domain([2.5, -2.5]);
    const cw = 13;
    if (probe === "lin") {
      AT.title(g, 0, -10, "f(αx + βy)   against   α f(x) + β f(y)");
      const cols = [["f(αx+βy)", fm.Z, DC.accent], ["αf(x)+βf(y)", comb, DC.good],
                    ["difference", fm.Z.map((r, i) => r.map((v, c) => v - comb[i][c])), DC.bad]];
      cols.forEach((c, i) => {
        const x0 = i * 135;
        AT.note(g, x0, 6, c[0], c[2], 10);
        for (let r = 0; r < n; r++) AT.strip(g, c[1][r], x0, 14 + r * (cw + 1), cw, cell, { stroke: "#0f1117" });
      });
      AT.note(g, 0, 14 + n * (cw + 1) + 18, "difference scaled to ±" + DL.fmtE(Math.max(1e-18, res), 1), DC.bad, 9.5);
    } else {
      AT.title(g, 0, -10, "∂Y/∂X, block by block (‖·‖_F of each token-pair block)");
      const eps = 1e-5, Jb = DL.zeros2(n, n);
      for (let jT = 0; jT < n; jT++) for (let c = 0; c < D; c++) {
        const o = Xm[jT][c];
        Xm[jT][c] = o + eps; const p = run(Xm, S, L, ffn).Z;
        Xm[jT][c] = o - eps; const m = run(Xm, S, L, ffn).Z;
        Xm[jT][c] = o;
        for (let iT = 0; iT < n; iT++) for (let e = 0; e < D; e++) {
          const dv = (p[iT][e] - m[iT][e]) / (2 * eps); Jb[iT][jT] += dv * dv;
        }
      }
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Jb[i][j] = Math.sqrt(Jb[i][j]);
      const sc = d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...Jb.map(r => Math.max(...r)))]);
      AT.heat(g, Jb, 0, 14, Math.min(20, 200 / n), sc, { stroke: "#0f1117" });
      AT.note(g, 0, 14 + n * Math.min(20, 200 / n) + 16, "off-diagonal blocks non-zero ⇒ information DOES move between tokens", DC.accent, 9.5);
    }

    /* right — residual against depth */
    const rx = 420, gw = f.iw - rx, gh = 230;
    AT.title(g, rx, -10, "additivity residual against depth");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const Ls = []; for (let l = 1; l <= 12; l++) Ls.push(l);
    const curves = [["attention only", false, DC.accent], ["attention + FFN", true, DC.bad]];
    const data = curves.map(c => Ls.map(l => {
      const S2 = layers(l);
      const q1 = run(X1, S2, l, c[1]).Z, q2 = run(X2, S2, l, c[1]).Z, qm = run(Xm, S2, l, c[1]).Z;
      return [l, relDiff(qm, q1.map((r, i) => r.map((v, cc) => a * v + b * q2[i][cc])))];
    }));
    const x = d3.scaleLinear().domain([1, 12]).range([0, gw - 10]);
    const y = d3.scaleLog().domain([1e-17, 4]).range([gh, 0]).clamp(true);
    DL.gridY(gg, y, gw - 10, 5);
    DL.axisB(gg, x, gh, 6, "layers stacked", d3.format("d"));
    DL.axisL(gg, y, 5, "relative residual", v => DL.fmtE(v, 0));
    data.forEach((dd, i) => DL.curve(gg, dd.map(p => [x(p[0]), y(Math.max(p[1], 1e-17))]), { stroke: curves[i][2], w: 2.2 }));
    gg.append("line").attr("x1", x(L)).attr("x2", x(L)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.55);
    DL.legend(gg, curves.map(c => ({ label: c[0], color: c[2] })), 4, gh + 30, { vertical: true, gap: 13, font: 9.5 });
    AT.note(gg, 4, gh + 62, n === 1 ? "n = 1: the softmax is the identity, so attention alone is one matrix"
      : "n > 1: the softmax makes A depend on X, which is the only source of curvature", n === 1 ? DC.good : DC.a2, 9.5);

    const kvv = DL.kv(g, rx, gh + 84, { keyW: 250, size: 10.5, lead: 14.5 });
    kvv("additivity residual", DL.fmtE(res, 3), res < 1e-12 ? DC.good : DC.bad, true);
    kvv("   with the attention weights FROZEN", DL.fmtE(resF, 3), resF < 1e-12 ? DC.good : DC.bad, true);
    kvv("   ratio", DL.fmtE(res / Math.max(1e-300, resF), 1));
    kvv("verdict on the FEATURE map", (resF < 1e-12 && !ffn) ? "exactly LINEAR" : "non-linear", (resF < 1e-12 && !ffn) ? DC.good : DC.bad, true);
    kvv("layers stacked", String(L));
    kvv("FFN inserted", ffn ? "yes" : "no", ffn ? DC.bad : DC.good);

    El("tm-readout").innerHTML =
      `${L} layer${L > 1 ? "s" : ""}, n = ${n} token${n > 1 ? "s" : ""}, ${ffn ? "with" : "without"} a position-wise FFN · ` +
      `the additivity residual is <b>${DL.fmtE(res, 3)}</b>, and with the attention weights frozen at their original values it is ` +
      `<b>${DL.fmtE(resF, 3)}</b>. ` +
      (ffn
        ? `The FFN makes the feature map non-linear, which is the whole reason it is there.`
        : (n === 1
          ? `With one token the softmax is the identity, so ${L} stacked attention layers are exactly one matrix multiply — the ` +
            `residual is round-off at any depth. Attention supplies <b>no</b> per-token non-linearity.`
          : `Freezing the weights removes the residual entirely: every departure from linearity comes from A depending on X, ` +
            `which is a non-linearity <b>across positions</b>. Along the feature axis the map is still a product of matrices.`));
  }
  ["tm-L", "tm-n", "tm-a"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#tm-p").on("change", draw);
  d3.select("#tm-ffn").on("change", draw);
  draw();
})();

/* ═════════ 22 · #rc-svg — rank collapse with depth, four wirings.
   Relative residual ‖X − 1x̄ᵀ‖/‖X‖ so that scale is divided out and only the
   SHAPE is measured. Random weights on purpose: the point is what the
   operation does before training opposes it. ═════════════════════════════ */
(function () {
  const svg = d3.select("#rc-svg"); if (svg.empty()) return;
  const W = 760, H = 420, D = 32;
  const El = id => document.getElementById(id);
  const WIR = [["pure", "pure attention", DC.bad], ["skip", "+ skip", DC.a2],
               ["mlp", "+ skip + MLP", DC.violet], ["ln", "+ skip + MLP + pre-LN", DC.accent]];

  function relres(X) {
    const n = X.length, d = X[0].length, mu = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += X[i][j] / n;
    let a = 0, b = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { a += (X[i][j] - mu[j]) ** 2; b += X[i][j] * X[i][j]; }
    return Math.sqrt(a / Math.max(1e-300, b));
  }
  function meanCos(X) {
    const n = X.length, U = X.map(r => AT.unit(r));
    let s = 0, c = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { let t = 0; for (let k = 0; k < X[0].length; k++) t += U[i][k] * U[j][k]; s += t; c++; }
    return s / c;
  }
  function LNrow(X) {
    return X.map(r => {
      let m = 0; r.forEach(v => m += v / r.length);
      let s = 0; r.forEach(v => s += (v - m) ** 2);
      s = Math.sqrt(s / r.length);
      return r.map(v => (v - m) / (s + 1e-5));
    });
  }
  function run(kind, depth, n, seed) {
    const r = DL.rng(seed);
    let X = DL.zeros2(n, D).map(row => row.map(() => DL.randn(r)));
    const res = [relres(X)], cos = [meanCos(X)];
    for (let l = 0; l < depth; l++) {
      const P = DL.mhaInit(D, D, D, 1, { seed: seed * 1000 + l });
      const W1 = AT.gauss(D, 4 * D, seed * 2000 + l, Math.sqrt(2 / D)), W2 = AT.gauss(4 * D, D, seed * 3000 + l, Math.sqrt(2 / (4 * D)));
      const U = (kind === "ln") ? LNrow(X) : X;
      const Y = DL.mhaForward(U, P, {}).Y;
      if (kind === "pure") X = Y;
      else X = X.map((row, i) => row.map((v, c) => v + Y[i][c]));
      if (kind === "mlp" || kind === "ln") {
        const V = (kind === "ln") ? LNrow(X) : X;
        const Hh = DL.matmul(V, W1).map(row => row.map(v => Math.max(0, v)));
        const O = DL.matmul(Hh, W2);
        X = X.map((row, i) => row.map((v, c) => v + O[i][c]));
      }
      res.push(relres(X)); cos.push(meanCos(X));
    }
    return { res: res, cos: cos, X: X };
  }
  function avg(kind, depth, n, seeds) {
    const R = [], C = [];
    for (let s = 1; s <= seeds; s++) {
      const o = run(kind, depth, n, s);
      o.res.forEach((v, i) => R[i] = (R[i] || 0) + v / seeds);
      o.cos.forEach((v, i) => C[i] = (C[i] || 0) + v / seeds);
    }
    return { res: R, cos: C };
  }

  function draw() {
    const L = +El("rc-L").value, n = +El("rc-n").value, seeds = +El("rc-s").value, kind = El("rc-w").value;
    El("rc-Lv").textContent = L; El("rc-nv").textContent = n;
    const f = DL.frame(svg, W, H, { l: 58, r: 16, t: 26, b: 46 }), g = f.g;
    const DEP = 20;
    const all = {};
    WIR.forEach(w => all[w[0]] = avg(w[0], DEP, n, seeds));

    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "relative residual ‖X − 1x̄ᵀ‖ / ‖X‖ against depth");
    const x = d3.scaleLinear().domain([0, DEP]).range([0, gw]);
    const y = d3.scaleLog().domain([1e-17, 1.4]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 6, "layers", d3.format("d"));
    DL.axisL(g, y, 5, "relative residual", v => DL.fmtE(v, 0));
    WIR.forEach(w => DL.curve(g, all[w[0]].res.map((v, i) => [x(i), y(Math.max(v, 1e-17))]),
      { stroke: w[2], w: w[0] === kind ? 2.6 : 1.3, op: w[0] === kind ? 1 : 0.55 }));
    g.append("line").attr("x1", x(L)).attr("x2", x(L)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.55);
    DL.legend(g, WIR.map(w => ({ label: w[1], color: w[2] })), 2, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    /* right — the cosine matrix at the selected depth */
    const rx = gw + 66, rw = f.iw - rx;
    AT.title(g, rx, -10, "pairwise cosine between tokens, depth 0 and depth " + L);
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const o0 = run(kind, 0, n, 1), oL = run(kind, L, n, 1);
    const cwc = Math.min(9, 130 / n);
    [["depth 0", o0.X, 0], ["depth " + L, oL.X, 150].slice(0, 3)].forEach((p, i) => {
      const U = p[1].map(r => AT.unit(r)), M = DL.zeros2(n, n);
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { let t = 0; for (let c = 0; c < D; c++) t += U[a][c] * U[b][c]; M[a][b] = t; }
      const sc = d3.scaleSequential(d3.interpolateMagma).domain([-1, 1]);
      AT.note(gg, i * 150, 4, p[0], DC.muted, 9.5);
      AT.heat(gg, M, i * 150, 10, cwc, sc, {});
    });
    const cy = 10 + n * cwc + 30;
    AT.title(gg, 0, cy - 8, "mean off-diagonal cosine against depth");
    const g3 = gg.append("g").attr("transform", `translate(0,${cy})`);
    const chh = gh - cy + 10;
    const xc = d3.scaleLinear().domain([0, DEP]).range([0, rw - 14]);
    const yc = d3.scaleLinear().domain([-0.2, 1.05]).range([chh, 0]);
    DL.gridY(g3, yc, rw - 14, 3);
    DL.axisB(g3, xc, chh, 5, "layers", d3.format("d"));
    DL.axisL(g3, yc, 3, "cos");
    WIR.forEach(w => DL.curve(g3, all[w[0]].cos.map((v, i) => [xc(i), yc(DL.clamp(v, -0.2, 1.05))]),
      { stroke: w[2], w: w[0] === kind ? 2.2 : 1.2, op: w[0] === kind ? 1 : 0.5 }));
    g3.append("line").attr("x1", 0).attr("x2", rw - 14).attr("y1", yc(0.99)).attr("y2", yc(0.99))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");

    const kvv = DL.kv(g, 0, gh + 96, { keyW: 280, size: 10.5, lead: 14.5 });
    WIR.forEach(w => {
      const c = all[w[0]].cos, idx = c.findIndex(v => v > 0.99);
      kvv(w[1] + ": residual at depth " + L,
        DL.fmtE(all[w[0]].res[L], 3) + "    cos > 0.99 at depth " + (idx > 0 ? idx : "> " + DEP),
        w[2], w[0] === kind);
    });

    const c = all[kind].cos, idx = c.findIndex(v => v > 0.99);
    El("rc-readout").innerHTML =
      `${n} tokens, width ${D}, ${seeds} seed${seeds > 1 ? "s" : ""} averaged, ${WIR.find(w => w[0] === kind)[1]} · at depth ` +
      `<b>${L}</b> the relative residual is <b>${DL.fmtE(all[kind].res[L], 3)}</b> and the mean pairwise cosine between tokens is ` +
      `<b>${DL.fmt(all[kind].cos[L], 4)}</b>. Across the four wirings at this depth: ` +
      WIR.map(w => `${w[1]} <b>${DL.fmtE(all[w[0]].res[L], 2)}</b>`).join(", ") + `. ` +
      `Pure attention reaches a cosine above 0.99 at depth <b>${all.pure.cos.findIndex(v => v > 0.99)}</b>; ` +
      `with the pre-norm it ${all.ln.cos.findIndex(v => v > 0.99) > 0 ? "reaches it at depth <b>" + all.ln.cos.findIndex(v => v > 0.99) + "</b>" : "does <b>not</b> reach it within " + DEP + " layers"}. ` +
      `Note the ordering: adding the MLP without a normalisation makes the collapse <b>faster</b>, not slower.`;
  }
  ["rc-L", "rc-n"].forEach(id => d3.select("#" + id).on("input", draw));
  ["rc-s", "rc-w"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 23 · #ec-svg — row entropy, the two failure ends and the healthy
   band between them. Entropy and the Jacobian norm come from DL.entropyOf and
   DL.softmaxJacFrob; nothing is recomputed here. ═════════════════════════ */
(function () {
  const svg = d3.select("#ec-svg"); if (svg.empty()) return;
  const W = 760, H = 410;
  const El = id => document.getElementById(id);

  function rowOf(n, sigma, m, bounded, seed) {
    const r = DL.rng(seed), s = new Array(n);
    for (let j = 0; j < n; j++) s[j] = DL.randn(r);
    if (bounded) { for (let j = 0; j < n; j++) s[j] = Math.tanh(s[j] / 2); }   /* |score| ≤ 1 */
    for (let j = 0; j < n; j++) s[j] *= sigma;
    s[0] += m;
    return DL.softmax(s);
  }
  function stats(n, sigma, m, bounded, trials) {
    let H = 0, mx = 0, J = 0;
    for (let t = 0; t < trials; t++) {
      const a = rowOf(n, sigma, m, bounded, 900 + t);
      H += DL.entropyOf(a) / trials; mx += Math.max(...a) / trials; J += DL.softmaxJacFrob(a) / trials;
    }
    return { H: H, mx: mx, J: J };
  }

  function draw() {
    const n = +El("ec-n").value, sigma = +El("ec-s").value, m = +El("ec-m").value, bounded = El("ec-b").value === "norm";
    El("ec-sv").textContent = DL.fmt(sigma, 2); El("ec-mv").textContent = DL.fmt(m, 1);
    const f = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 46 }), g = f.g;
    const T = n <= 512 ? 220 : 60;
    const sig = DL.linspace(-1.3, 1.2, 34).map(v => Math.pow(10, v));
    const sw = sig.map(s => Object.assign({ s: s }, stats(n, s, m, bounded, T)));
    const cur = stats(n, sigma, m, bounded, T * 3);
    const Jmax = Math.max(...sw.map(p => p.J));

    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "entropy and effective keys, against the logit scale");
    const x = d3.scaleLog().domain([0.05, 16]).range([0, gw]);
    const y = d3.scaleLinear().domain([0, Math.log(n) * 1.1]).range([gh, 0]);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "logit scale σ", v => DL.fmtE(v, 0));
    DL.axisL(g, y, 5, "H(α), nats");
    g.append("line").attr("x1", 0).attr("x2", gw).attr("y1", y(Math.log(n))).attr("y2", y(Math.log(n)))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    AT.note(g, 4, y(Math.log(n)) - 5, "maximum, ln " + n + " = " + DL.fmt(Math.log(n), 3), DC.muted, 9);
    /* healthy band: ‖J‖_F within a factor of 2 of its peak over this sweep */
    const band = sw.filter(p => p.J > Jmax / 2);
    if (band.length) {
      g.append("rect").attr("x", x(band[0].s)).attr("y", 0).attr("width", Math.max(1, x(band[band.length - 1].s) - x(band[0].s)))
        .attr("height", gh).attr("fill", DC.good).attr("fill-opacity", 0.09);
      AT.note(g, x(band[0].s) + 4, gh - 8, "‖J‖_F within 2× of its peak", DC.good, 9);
    }
    DL.curve(g, sw.map(p => [x(p.s), y(p.H)]), { stroke: DC.accent, w: 2.4 });
    g.append("line").attr("x1", x(DL.clamp(sigma, 0.05, 16))).attr("x2", x(DL.clamp(sigma, 0.05, 16)))
      .attr("y1", 0).attr("y2", gh).attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    AT.note(g, 2, gh + 34, bounded ? "bounded scores: |q·kᵀ| ≤ 1, so σ IS the inverse temperature and the entropy has a FLOOR"
      : "unbounded scores: nothing prevents σ from growing during training", bounded ? DC.good : DC.a2, 9.5);

    /* right — the row, and the Jacobian curve */
    const rx = gw + 62, rw = f.iw - rx;
    AT.title(g, rx, -10, "the row now, and ‖J‖_F across the sweep");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const a = rowOf(n, sigma, m, bounded, 901);
    const bh = 96;
    const xb = d3.scaleLinear().domain([0, Math.min(n, 128)]).range([0, rw - 12]);
    const yb = d3.scaleLinear().domain([0, Math.max(...a) * 1.15]).range([bh, 0]);
    DL.axisL(gg, yb, 3, "α");
    gg.selectAll("rect.a").data(a.slice(0, Math.min(n, 128)).map((v, i) => [i, v])).join("rect").attr("class", "a")
      .attr("x", d => xb(d[0])).attr("y", d => yb(d[1])).attr("width", Math.max(1, (rw - 12) / Math.min(n, 128) - 0.6))
      .attr("height", d => bh - yb(d[1])).attr("fill", d => d[0] === 0 && m > 0 ? DC.good : DC.accent).attr("fill-opacity", 0.85);
    AT.note(gg, 0, bh + 14, n > 128 ? "first 128 of " + n + " keys" : n + " keys", DC.muted, 9);
    const jy = bh + 40, jh = gh - jy;
    const xj = d3.scaleLog().domain([0.05, 16]).range([0, rw - 12]);
    const yj = d3.scaleLinear().domain([0, Jmax * 1.15]).range([jh, 0]);
    const gj = gg.append("g").attr("transform", `translate(0,${jy})`);
    DL.gridY(gj, yj, rw - 12, 3);
    DL.axisB(gj, xj, jh, 5, "logit scale σ", v => DL.fmtE(v, 0));
    DL.axisL(gj, yj, 3, "‖J‖_F");
    DL.curve(gj, sw.map(p => [xj(p.s), yj(p.J)]), { stroke: DC.a2, w: 2.2 });
    gj.append("line").attr("x1", xj(DL.clamp(sigma, 0.05, 16))).attr("x2", xj(DL.clamp(sigma, 0.05, 16)))
      .attr("y1", 0).attr("y2", jh).attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    AT.note(gj, 4, 12, "zero at BOTH ends: uniform and one-hot", DC.a2, 9);

    const eff = Math.exp(cur.H);
    const kvv = DL.kv(g, 0, gh + 66, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("entropy H(α), nats", DL.fmt(cur.H, 4), DC.accent, true);
    kvv("   its maximum, ln n", DL.fmt(Math.log(n), 4), DC.muted);
    kvv("effective number of keys read, eᴴ", DL.fmt(eff, 2) + " of " + DL.commas(n), DC.a2, true);
    kvv("largest weight", DL.fmt(cur.mx, 5), cur.mx > 0.99 ? DC.bad : DC.ink);
    kvv("softmax Jacobian ‖J‖_F", DL.fmtE(cur.J, 3), cur.J > Jmax / 2 ? DC.good : DC.bad, true);
    kvv("   as a fraction of its peak over the sweep", DL.fmt(100 * cur.J / Jmax, 1) + "%", cur.J > Jmax / 2 ? DC.good : DC.bad);
    kvv("regime", eff > n / 3 ? "DILUTED — reading nearly everything" : (cur.mx > 0.98 ? "COLLAPSED — one-hot" : "healthy"),
      eff > n / 3 ? DC.a2 : (cur.mx > 0.98 ? DC.bad : DC.good), true);

    /* bounded scores: |s| ≤ σ, so the sharpest row puts one key at +σ and the rest at −σ */
    const pTop = Math.exp(2 * sigma) / (Math.exp(2 * sigma) + n - 1);
    const Hfloor = -(pTop * Math.log(pTop) + (1 - pTop) * Math.log((1 - pTop) / (n - 1)));
    if (bounded) kvv("entropy FLOOR at this σ (one key at +σ, the rest at −σ)", DL.fmt(Hfloor, 4) + " nats  →  eᴴ ≥ " + DL.fmt(Math.exp(Hfloor), 2), DC.good);
    El("ec-readout").innerHTML =
      `n = ${DL.commas(n)} keys, logit scale σ = ${DL.fmt(sigma, 2)}, signal margin ${DL.fmt(m, 1)}, ` +
      `${bounded ? "bounded" : "unbounded"} scores · H(α) = <b>${DL.fmt(cur.H, 4)}</b> nats of a possible ` +
      `<b>${DL.fmt(Math.log(n), 4)}</b>, so the row is effectively reading <b>${DL.fmt(eff, 2)}</b> of its ${DL.commas(n)} keys. ` +
      `The largest weight is <b>${DL.fmt(cur.mx, 5)}</b> and the softmax Jacobian norm is <b>${DL.fmtE(cur.J, 3)}</b> — ` +
      `<b>${DL.fmt(100 * cur.J / Jmax, 1)}%</b> of its peak over this sweep. ` +
      (eff > n / 3 ? `This row is <b>diluted</b>: it is averaging almost everything, and its output carries little more than the mean of the values.`
        : cur.mx > 0.98 ? `This row has <b>collapsed</b>: it is one-hot, its Jacobian is near zero, and it will not move again.`
          : `This is the healthy regime: a handful of keys, and a Jacobian near its largest achievable value.`) +
      (bounded ? ` With every score bounded by ±σ the row cannot be sharper than one key at +σ against the rest at −σ, so its entropy ` +
        `has a floor of <b>${DL.fmt(Hfloor, 4)}</b> nats (eᴴ ≥ ${DL.fmt(Math.exp(Hfloor), 2)}) at this σ — collapse needs σ itself to grow, ` +
        `which is what the learned temperature is for.` : ``);
  }
  ["ec-s", "ec-m"].forEach(id => d3.select("#" + id).on("input", draw));
  ["ec-n", "ec-b"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 24 · #kv-svg — generation with and without the cache.
   MAC counts, batch 1, one layer, using the same decomposition as
   DL.attnMacs: 2·d·(h·dₖ) + 2·d·(g·dₖ) of projection work per NEW token
   (W_Q and W_O, then W_K and W_V) plus 2·n·(h·dₖ) against the cache,
   against a full O(n²) recompute. ══════════════════════════════════════ */
(function () {
  const svg = d3.select("#kv-svg"); if (svg.empty()) return;
  const W = 760, H = 410;
  const El = id => document.getElementById(id);
  const d = 4096, h = 32, g = 8, dh = 128;

  function stepMacs(n, cached) {
    if (cached) return { proj: 2 * d * (h * dh) + 2 * d * (g * dh), cache: 2 * n * h * dh };
    const m = DL.attnMacs({ n: n, d: d, h: h, g: g, dh: dh, dff: 0, nmat: 0, causal: false });
    return { proj: m.proj, cache: m.quad };
  }

  function draw() {
    const T = +El("kv-t").value, pre = +El("kv-p").value, mode = El("kv-m").value, view = El("kv-v").value;
    El("kv-tv").textContent = T; El("kv-pv").textContent = pre;
    const f = DL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 46 }), g2 = f.g;
    const n = pre + T, cached = mode === "cache";

    /* left — the causal matrix, with what this step touches outlined */
    const cw = Math.min(17, 210 / n);
    AT.title(g2, 0, -10, "the causal score matrix after " + T + " generated token" + (T > 1 ? "s" : ""));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      const isNew = cached ? (i === n - 1 || j === n - 1) : true;
      g2.append("rect").attr("x", j * cw).attr("y", 6 + i * cw).attr("width", cw - 0.6).attr("height", cw - 0.6)
        .attr("fill", j < pre ? DC.violet : DC.accent).attr("fill-opacity", isNew ? 0.85 : 0.16)
        .attr("stroke", isNew ? DC.a2 : "none").attr("stroke-width", isNew ? 0.9 : 0);
    }
    AT.note(g2, 0, 6 + n * cw + 15, cached ? "outlined: recomputed this step — one row and one column"
      : "outlined: recomputed this step — the ENTIRE triangle", cached ? DC.good : DC.bad, 9.5);
    AT.note(g2, 0, 6 + n * cw + 28, "purple: prompt positions    blue: generated positions", DC.muted, 9);

    /* the cache itself */
    const cx = n * cw + 34;
    AT.note(g2, cx, 0, "the cache", DC.a2, 10);
    for (let i = 0; i < n; i++) {
      g2.append("rect").attr("x", cx).attr("y", 6 + i * cw).attr("width", 28).attr("height", cw - 0.6)
        .attr("fill", DC.a2).attr("fill-opacity", i === n - 1 ? 0.9 : 0.35).attr("stroke", i === n - 1 ? DC.a2 : "none");
      g2.append("rect").attr("x", cx + 32).attr("y", 6 + i * cw).attr("width", 28).attr("height", cw - 0.6)
        .attr("fill", DC.violet).attr("fill-opacity", i === n - 1 ? 0.9 : 0.35).attr("stroke", i === n - 1 ? DC.violet : "none");
    }
    AT.note(g2, cx, 6 + n * cw + 15, "K            V", DC.muted, 9);
    const elems = 2 * g * dh * n;
    AT.note(g2, cx, 6 + n * cw + 28, DL.commas(elems) + " elements, one layer", DC.a2, 9);

    /* right — work per step */
    const rx = 380, gw = f.iw - rx, gh = 250;
    AT.title(g2, rx, -10, view === "cum" ? "cumulative MACs, one layer" : "MACs per decode step, one layer");
    const gg = g2.append("g").attr("transform", `translate(${rx},0)`);
    const steps = []; for (let t = 1; t <= 16; t++) steps.push(t);
    function series(c) {
      let cum = 0;
      return steps.map(t => {
        const m = stepMacs(pre + t, c), v = m.proj + m.cache;
        cum += v; return [t, view === "cum" ? cum : v];
      });
    }
    const sc = series(true), su = series(false);
    const x = d3.scaleLinear().domain([1, 16]).range([0, gw - 10]);
    const y = d3.scaleLog().domain([Math.min(sc[0][1], su[0][1]) / 2, Math.max(sc[15][1], su[15][1]) * 2]).range([gh, 0]);
    DL.gridY(gg, y, gw - 10, 5);
    DL.axisB(gg, x, gh, 6, "generated token", d3.format("d"));
    DL.axisL(gg, y, 5, "MACs", v => DL.big(v));
    DL.curve(gg, su.map(p => [x(p[0]), y(p[1])]), { stroke: DC.bad, w: cached ? 1.4 : 2.4, op: cached ? 0.6 : 1 });
    DL.curve(gg, sc.map(p => [x(p[0]), y(p[1])]), { stroke: DC.good, w: cached ? 2.4 : 1.4, op: cached ? 1 : 0.6 });
    gg.append("line").attr("x1", x(T)).attr("x2", x(T)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    DL.legend(gg, [{ label: "with the cache", color: DC.good }, { label: "recomputing everything", color: DC.bad }],
      4, gh + 32, { vertical: true, gap: 13, font: 9.5 });

    const mc = stepMacs(n, true), mu = stepMacs(n, false);
    let totC = 0, totU = 0;
    for (let t = 1; t <= T; t++) {
      const a = stepMacs(pre + t, true), b = stepMacs(pre + t, false);
      totC += a.proj + a.cache; totU += b.proj + b.cache;
    }
    const kvv = DL.kv(g2, 0, 6 + n * cw + 52, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("this step, with the cache", DL.big(mc.proj + mc.cache) + "   (" + DL.big(mc.cache) + " against the cache)", DC.good, true);
    kvv("this step, recomputing", DL.big(mu.proj + mu.cache), DC.bad, true);
    kvv("   ratio", DL.fmt((mu.proj + mu.cache) / (mc.proj + mc.cache), 1) + "×", DC.a2);
    kvv("cache, one layer", DL.commas(elems) + " elements  =  " + DL.big(elems * 2) + " B in bf16", DC.a2);
    kvv("cache, all 32 layers", DL.big(elems * 2 * 32) + " B", DC.a2);
    kvv("growth per generated token", DL.commas(2 * g * dh) + " elements per layer", DC.muted);

    El("kv-readout").innerHTML =
      `d = ${DL.commas(d)}, h = ${h}, g = ${g}, prompt ${pre} + ${T} generated = ${n} positions · this decode step costs ` +
      `<b>${DL.big(mc.proj + mc.cache)}</b> MACs with the cache and <b>${DL.big(mu.proj + mu.cache)}</b> without it, a factor of ` +
      `<b>${DL.fmt((mu.proj + mu.cache) / (mc.proj + mc.cache), 1)}×</b>. Cumulatively over the ${T} generated tokens: ` +
      `<b>${DL.big(totC)}</b> against <b>${DL.big(totU)}</b>. The cache now holds <b>${DL.commas(elems)}</b> elements per layer — ` +
      `<b>${DL.big(elems * 2 * 32)}</b> bytes across all 32 layers in bf16 — and grows by <b>${DL.commas(2 * g * dh)}</b> elements ` +
      `per layer for every further token, for ever.`;
  }
  ["kv-t", "kv-p"].forEach(id => d3.select("#" + id).on("input", draw));
  ["kv-m", "kv-v"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 25 · #km-svg — cache bytes against context and batch, with the
   weights drawn for scale. Cache sizes from DL.kvCacheElems. ═════════════ */
(function () {
  const svg = d3.select("#km-svg"); if (svg.empty()) return;
  const W = 760, H = 400, GiB = 1073741824;
  const El = id => document.getElementById(id);

  function draw() {
    const c = CFG[El("km-p").value], B = +El("km-b").value, bpe = +El("km-t").value, dev = +El("km-g").value * GiB;
    El("km-bv").textContent = B; El("km-gv").textContent = El("km-g").value;
    const f = DL.frame(svg, W, H, { l: 58, r: 16, t: 26, b: 46 }), g = f.g;
    const perTok = cfgCache(c) * bpe;
    const wts = c.P * 2;

    const gw = 340, gh = 250;
    AT.title(g, 0, -10, "KV-cache bytes against context length");
    const x = d3.scaleLog().domain([256, 1048576]).range([0, gw]);
    const y = d3.scaleLog().domain([perTok * 256 / 4, Math.max(dev, wts) * 8]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "context length n", v => DL.big(v));
    DL.axisL(g, y, 5, "bytes", v => DL.big(v));
    [1, 4, 16, 64, 128].forEach(b => {
      DL.curve(g, [[x(256), y(perTok * 256 * b)], [x(1048576), y(perTok * 1048576 * b)]],
        { stroke: DC.accent, w: b === B ? 2.6 : 1, op: b === B ? 1 : 0.35 });
      AT.note(g, gw - 34, y(DL.clamp(perTok * 1048576 * b, perTok * 64, Math.max(dev, wts) * 8)) + 3, "B=" + b, DC.accent, 8.5);
    });
    DL.curve(g, [[x(256), y(perTok * 256 * B)], [x(1048576), y(perTok * 1048576 * B)]], { stroke: DC.accent, w: 2.6 });
    [[wts, "weights, bf16  " + DL.fmt(wts / GiB, 1) + " GiB", DC.violet], [dev, "device memory  " + El("km-g").value + " GiB", DC.good]].forEach(r => {
      g.append("line").attr("x1", 0).attr("x2", gw).attr("y1", y(r[0])).attr("y2", y(r[0]))
        .attr("stroke", r[2]).attr("stroke-width", 1.6);
      AT.note(g, 4, y(r[0]) - 5, r[1], r[2], 9.5);
    });
    const nW = wts / (perTok * B), nD = (dev - wts) / (perTok * B);
    [[nW, "cache = weights", DC.violet, 0], [nD, "device full", DC.bad, 1]].forEach(r => {
      if (!(r[0] > 256 && r[0] < 1048576)) return;
      g.append("line").attr("x1", x(r[0])).attr("x2", x(r[0])).attr("y1", 0).attr("y2", gh)
        .attr("stroke", r[2]).attr("stroke-dasharray", "2 3");
      AT.note(g, x(r[0]) + 3, 12 + r[3] * 11, DL.commas(Math.round(r[0])) + " — " + r[1], r[2], 8.5);
    });

    /* right — the memory budget as a stacked bar */
    const rx = gw + 62, rw = f.iw - rx;
    AT.title(g, rx, -10, "the device, at n = " + DL.commas(c.ctx));
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const used = perTok * c.ctx * B, free = Math.max(0, dev - wts - used);
    const segs = [["weights", wts, DC.violet], ["cache, batch " + B, used, DC.accent], ["free", free, DC.line]];
    const tot = wts + used + free;
    let yy = 10;
    segs.forEach(s2 => {
      const hgt = Math.max(1, 190 * s2[1] / tot);
      gg.append("rect").attr("x", 0).attr("y", yy).attr("width", 90).attr("height", hgt)
        .attr("fill", s2[2]).attr("fill-opacity", 0.65).attr("stroke", s2[2]);
      AT.note(gg, 98, yy + Math.min(hgt / 2 + 4, 14), s2[0] + ": " + DL.fmt(s2[1] / GiB, 2) + " GiB", s2[2], 9.5);
      yy += hgt;
    });
    const maxB = Math.max(0, Math.floor((dev - wts) / (perTok * c.ctx)));
    AT.note(gg, 0, 220, "largest batch of " + DL.commas(c.ctx) + "-token sequences that fits:", DC.muted, 10);
    gg.append("text").attr("x", 0).attr("y", 244).attr("font-size", 22)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", maxB > 0 ? DC.good : DC.bad).text(DL.commas(maxB));

    const kvv = DL.kv(g, 0, gh + 66, { keyW: 288, size: 10.5, lead: 14.5 });
    kvv("cache elements per token, whole stack", DL.commas(cfgCache(c)), DC.accent, true);
    kvv("cache bytes per token", DL.commas(perTok) + "  =  " + DL.fmt(perTok / 1024, 1) + " KiB", DC.accent, true);
    kvv("one sequence at n = " + DL.commas(c.ctx), DL.fmt(perTok * c.ctx / GiB, 3) + " GiB");
    kvv("at batch " + B, DL.fmt(used / GiB, 2) + " GiB", used > wts ? DC.bad : DC.ink, true);
    kvv("weights, bf16", DL.fmt(wts / GiB, 2) + " GiB", DC.violet);
    kvv("context at which cache = weights (batch " + B + ")", DL.commas(Math.round(nW)), DC.violet, true);
    kvv("largest batch that fits at n = " + DL.commas(c.ctx), DL.commas(maxB), maxB > 0 ? DC.good : DC.bad, true);

    El("km-readout").innerHTML =
      `${c.lab}, batch ${B}, ${bpe} bytes per cache element, ${El("km-g").value} GiB device · the cache is ` +
      `<b>${DL.commas(cfgCache(c))}</b> elements = <b>${DL.fmt(perTok / 1024, 1)} KiB</b> per token, so one ` +
      `${DL.commas(c.ctx)}-token sequence costs <b>${DL.fmt(perTok * c.ctx / GiB, 3)} GiB</b> and batch ${B} costs ` +
      `<b>${DL.fmt(used / GiB, 2)} GiB</b> against <b>${DL.fmt(wts / GiB, 2)} GiB</b> of weights. At batch ${B} the cache ` +
      `overtakes the weights at a context of <b>${DL.commas(Math.round(nW))}</b> tokens, and the device holds at most ` +
      `<b>${DL.commas(maxB)}</b> concurrent ${DL.commas(c.ctx)}-token sequences. The weights are shared across the batch; ` +
      `the cache is not.`;
  }
  ["km-b", "km-g"].forEach(id => d3.select("#" + id).on("input", draw));
  ["km-p", "km-t"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 26 · #ai-svg — the roofline for a decode step.
   Attention's arithmetic intensity is DL.decodeIntensity(h, g, b) = h/(g·b)
   MAC per byte, and it does NOT move with the batch or with n. The weight
   matmuls' intensity is proportional to the batch, and that contrast is the
   figure. ════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#ai-svg"); if (svg.empty()) return;
  const W = 760, H = 410, GiB = 1073741824;
  const El = id => document.getElementById(id);

  function draw() {
    const hw = HW[El("ai-hw").value], B = +El("ai-b").value, n = Math.pow(2, +El("ai-n").value), bpe = +El("ai-t").value;
    El("ai-bv").textContent = B; El("ai-nv").textContent = DL.commas(n);
    const f = DL.frame(svg, W, H, { l: 58, r: 16, t: 26, b: 46 }), g = f.g;
    const ridge = hw.flops / hw.bw;

    const VAR = [
      { lab: "MHA  h=32 g=32", h: 32, g: 32, col: DC.bad },
      { lab: "GQA-8  h=32 g=8", h: 32, g: 8, col: DC.a2 },
      { lab: "GQA-8  h=64 g=8", h: 64, g: 8, col: DC.violet },
      { lab: "MQA  h=32 g=1", h: 32, g: 1, col: DC.good },
      { lab: "MQA  h=64 g=1", h: 64, g: 1, col: DC.teal }
    ].map(v => Object.assign(v, { I: DL.decodeIntensity(v.h, v.g, bpe).flop }));
    /* weight matmuls: 2·P FLOPs against 2·P bytes read (bf16), shared by B */
    const Iw = 2 * B / 2;

    const gw = 340, gh = 250;
    AT.title(g, 0, -10, "roofline · " + hw.lab + ", ridge point " + DL.fmt(ridge, 1) + " FLOP/byte");
    const x = d3.scaleLog().domain([0.5, 4096]).range([0, gw]);
    const y = d3.scaleLog().domain([hw.flops / 2000, hw.flops * 2]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "arithmetic intensity, FLOP/byte", v => DL.big(v));
    DL.axisL(g, y, 5, "attainable FLOP/s", v => DL.big(v));
    DL.curve(g, [[x(0.5), y(0.5 * hw.bw)], [x(ridge), y(hw.flops)]], { stroke: DC.muted, w: 2 });
    DL.curve(g, [[x(ridge), y(hw.flops)], [x(4096), y(hw.flops)]], { stroke: DC.muted, w: 2 });
    g.append("line").attr("x1", x(ridge)).attr("x2", x(ridge)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.5);
    AT.note(g, x(ridge) + 4, 12, "ridge " + DL.fmt(ridge, 1), DC.ink, 9.5);
    AT.note(g, 6, gh - 10, "memory bound", DC.muted, 9.5);
    AT.note(g, gw - 90, 26, "compute bound", DC.muted, 9.5);
    VAR.forEach((v, i) => {
      const att = Math.min(hw.flops, v.I * hw.bw);
      g.append("circle").attr("cx", x(v.I)).attr("cy", y(att)).attr("r", 4).attr("fill", v.col);
      AT.note(g, x(v.I) + 6, y(att) - 4 + (i % 2 ? 10 : 0), v.lab, v.col, 8.5);
    });
    const attW = Math.min(hw.flops, Iw * hw.bw);
    g.append("rect").attr("x", x(Iw) - 4).attr("y", y(attW) - 4).attr("width", 8).attr("height", 8)
      .attr("fill", DC.accent).attr("transform", `rotate(45,${x(Iw)},${y(attW)})`);
    AT.note(g, x(Iw) + 8, y(attW) + 4, "the weight matmuls, batch " + B, DC.accent, 9.5);

    /* right — time per token, split */
    const rx = gw + 62, rw = f.iw - rx;
    AT.title(g, rx, -10, "ms per generated token, by source");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const c = CFG.l370b;
    const cacheB = cfgCache(c) * bpe * n, wB = c.P * 2;
    const Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    const tCache = b => b * cacheB / hw.bw * 1000, tW = () => wB / hw.bw * 1000;
    const xs = d3.scaleLog().domain([1, 256]).range([0, rw - 12]);
    const ys = d3.scaleLog().domain([tW() / 40, Math.max(tCache(256), tW()) * 2]).range([gh, 0]).clamp(true);
    DL.gridY(gg, ys, rw - 12, 5);
    DL.axisB(gg, xs, gh, 5, "batch", d3.format("d"));
    DL.axisL(gg, ys, 5, "ms / token", v => DL.fmt(v, 1));
    DL.curve(gg, Bs.map(b => [xs(b), ys(tCache(b))]), { stroke: DC.bad, w: 2.2 });
    DL.curve(gg, Bs.map(b => [xs(b), ys(tW())]), { stroke: DC.violet, w: 2.2 });
    const bCross = wB / cacheB;
    if (bCross >= 1 && bCross <= 256) {
      gg.append("line").attr("x1", xs(bCross)).attr("x2", xs(bCross)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", DC.a2).attr("stroke-dasharray", "2 3");
      AT.note(gg, xs(bCross) + 3, 12, "cache overtakes the weights at batch " + DL.fmt(bCross, 1), DC.a2, 8.5);
    }
    gg.append("line").attr("x1", xs(DL.clamp(B, 1, 256))).attr("x2", xs(DL.clamp(B, 1, 256))).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.55);
    DL.legend(gg, [{ label: "KV-cache traffic (scales with batch)", color: DC.bad },
                   { label: "weight traffic (shared across batch)", color: DC.violet }],
      4, gh + 32, { vertical: true, gap: 13, font: 9.5 });
    AT.note(gg, 4, gh + 64, "Llama-3 70B at n = " + DL.commas(n) + ", " + hw.lab, DC.muted, 9.5);

    const kvv = DL.kv(g, 0, gh + 66, { keyW: 288, size: 10.5, lead: 14.5 });
    kvv("ridge point of " + hw.lab, DL.fmt(ridge, 1) + " FLOP/byte", DC.ink, true);
    VAR.slice(0, 4).forEach(v => kvv(v.lab + " intensity", DL.fmt(v.I, 2) + " FLOP/byte  →  " +
      DL.fmt(100 * v.I / ridge, 2) + "% of peak", v.col));
    kvv("weight matmuls at batch " + B, DL.fmt(Iw, 1) + " FLOP/byte  →  " + DL.fmt(100 * Math.min(1, Iw / ridge), 1) + "% of peak", DC.accent, true);
    kvv("Llama-3 70B cache at n = " + DL.commas(n), DL.fmt(cacheB / GiB, 2) + " GiB  →  " + DL.fmt(tCache(1), 2) + " ms/token/sequence", DC.bad);
    kvv("   ceiling from cache traffic alone", DL.fmt(1000 / tCache(1), 1) + " tokens/s", DC.bad, true);
    kvv("batch at which cache traffic = weight traffic", DL.fmt(bCross, 1), DC.a2, true);

    El("ai-readout").innerHTML =
      `${hw.lab}, batch ${B}, n = ${DL.commas(n)}, ${bpe === 2 ? "bf16" : "fp8"} cache · the ridge point is ` +
      `<b>${DL.fmt(ridge, 1)}</b> FLOP/byte. Decode attention sits at <b>${DL.fmt(VAR[0].I, 2)}</b> for MHA, ` +
      `<b>${DL.fmt(VAR[1].I, 2)}</b> for GQA-8 and <b>${DL.fmt(VAR[3].I, 2)}</b> for MQA — ` +
      `<b>${DL.fmt(100 * VAR[0].I / ridge, 2)}%</b>, <b>${DL.fmt(100 * VAR[1].I / ridge, 2)}%</b> and ` +
      `<b>${DL.fmt(100 * VAR[3].I / ridge, 2)}%</b> of peak arithmetic. None of those moves when the batch changes, because ` +
      `each sequence owns its cache. The weight matmuls, by contrast, sit at <b>${DL.fmt(Iw, 1)}</b> FLOP/byte at batch ${B} ` +
      `and rise in direct proportion to it — they cross the ridge at batch <b>${DL.commas(Math.ceil(ridge))}</b>, and ` +
      `attention never does. ` +
      `For Llama-3 70B at n = ${DL.commas(n)} the cache is <b>${DL.fmt(cacheB / GiB, 2)} GiB</b>, which takes ` +
      `<b>${DL.fmt(tCache(1), 2)} ms</b> to read once — a ceiling of <b>${DL.fmt(1000 / tCache(1), 1)}</b> tokens per second per ` +
      `sequence before a single weight has been touched, and cache traffic overtakes weight traffic at batch <b>${DL.fmt(bCross, 1)}</b>.`;
  }
  ["ai-b", "ai-n"].forEach(id => d3.select("#" + id).on("input", draw));
  ["ai-hw", "ai-t"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 27 · #gq-svg — MHA → GQA → MQA. Cache from DL.kvCacheElems,
   intensity from DL.decodeIntensity, parameters from DL.mhaParams, MACs from
   DL.attnMacs. The flat quadratic line is the point of the right panel. ══ */
(function () {
  const svg = d3.select("#gq-svg"); if (svg.empty()) return;
  const W = 760, H = 410, GiB = 1073741824;
  const El = id => document.getElementById(id);
  const GS = [1, 2, 4, 8, 16, 32, 64];

  function draw() {
    const h = +El("gq-h").value, dk = +El("gq-dk").value, L = +El("gq-L").value, n = Math.pow(2, +El("gq-n").value);
    let g = GS[+El("gq-g").value]; if (g > h) g = h;
    El("gq-hv").textContent = h; El("gq-gv").textContent = g; El("gq-Lv").textContent = L;
    El("gq-nv").textContent = DL.commas(n);
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 26, b: 46 }), g2 = f.g;
    const d = h * dk;

    /* left — the bipartite map */
    const gw = 300;
    AT.title(g2, 0, -10, h + " query heads  →  " + g + " key/value head" + (g > 1 ? "s" : ""));
    const showH = Math.min(h, 32), step = h / showH;
    const qx = i => 12 + i * (gw - 24) / Math.max(1, showH - 1);
    const kx = j => g === 1 ? 12 + (gw - 24) / 2 : 12 + j * (gw - 24) / (g - 1);
    const col = d3.scaleSequential(d3.interpolateTurbo).domain([-0.6, g - 0.4]);
    const topY = 30, botY = 150;
    for (let i = 0; i < showH; i++) {
      const j = DL.kvHeadOf(Math.floor(i * step), h, g);
      g2.append("line").attr("x1", qx(i)).attr("y1", topY).attr("x2", kx(j)).attr("y2", botY)
        .attr("stroke", col(j)).attr("stroke-width", 1.2).attr("stroke-opacity", 0.45);
    }
    for (let j = 0; j < g; j++) {
      g2.append("rect").attr("x", kx(j) - 12).attr("y", botY - 9).attr("width", 24).attr("height", 18).attr("rx", 4)
        .attr("fill", col(j)).attr("stroke", "#0f1117");
    }
    for (let i = 0; i < showH; i++) {
      const j = DL.kvHeadOf(Math.floor(i * step), h, g);
      g2.append("circle").attr("cx", qx(i)).attr("cy", topY).attr("r", 5).attr("fill", col(j)).attr("stroke", "#0f1117");
    }
    AT.note(g2, 0, topY - 12, "Q", DC.muted, 10);
    AT.note(g2, 0, botY + 26, "K, V — the only thing cached", DC.a2, 10);
    if (showH < h) AT.note(g2, 0, topY + 20, "(showing " + showH + " of " + h + " query heads)", DC.muted, 9);
    const lab = g === 1 ? "MQA · multi-query" : g === h ? "MHA · multi-head" : "GQA · grouped-query, " + g + " groups of " + (h / g);
    AT.note(g2, 0, botY + 46, lab, DC.ink, 12);

    /* right — the four quantities against g */
    const rx = gw + 56, rw = f.iw - rx, gh = 250;
    AT.title(g2, rx, -10, "what moves with g, and what does not");
    const gg = g2.append("g").attr("transform", `translate(${rx},0)`);
    const gs = GS.filter(x => x <= h);
    const rows = gs.map(x => ({
      g: x,
      cache: DL.kvCacheElems({ h: h, g: x, dh: dk, L: L, kind: "gqa" }) * 2,
      par: DL.mhaParams(d, dk, dk, h, x).total,
      I: DL.decodeIntensity(h, x, 2).flop,
      quad: DL.attnMacs({ n: n, d: d, h: h, g: x, dh: dk, dff: 0, nmat: 0 }).quad
    }));
    const cur = rows.find(r => r.g === g);
    const lines = [["KV-cache bytes / token", r => r.cache, DC.bad], ["block parameters", r => r.par, DC.violet],
                   ["decode intensity, FLOP/byte", r => r.I, DC.good], ["quadratic MACs at n = " + DL.commas(n), r => r.quad, DC.accent]];
    const all = rows.flatMap(r => lines.map(l => l[1](r)));
    const x = d3.scaleLog().domain([1, Math.max(2, gs[gs.length - 1])]).range([0, rw - 12]);
    const y = d3.scaleLog().domain([Math.min(...all) / 4, Math.max(...all) * 4]).range([gh, 0]);
    DL.gridY(gg, y, rw - 12, 5);
    DL.axisB(gg, x, gh, 5, "KV heads g", d3.format("d"));
    DL.axisL(gg, y, 5, "", v => DL.big(v));
    lines.forEach(l => DL.curve(gg, rows.map(r => [x(r.g), y(l[1](r))]), { stroke: l[2], w: 2 }));
    gg.append("line").attr("x1", x(g)).attr("x2", x(g)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    DL.legend(gg, lines.map(l => ({ label: l[0], color: l[2] })), 4, gh + 32, { vertical: true, gap: 13, font: 9.5 });
    AT.note(gg, 4, gh + 84, "the blue line is exactly flat — GQA does not touch the quadratic term", DC.accent, 9.5);

    const mha = rows.find(r => r.g === h) || rows[rows.length - 1];
    const kvv = DL.kv(g2, 0, 220, { keyW: 268, size: 10.5, lead: 14.5 });
    kvv("cache elements per token", DL.commas(cur.cache / 2), DC.bad, true);
    kvv("cache KiB per token, bf16", DL.fmt(cur.cache / 1024, 1), DC.bad);
    kvv("at n = " + DL.commas(n), DL.fmt(cur.cache * n / GiB, 3) + " GiB", DC.bad, true);
    kvv("against MHA (g = " + h + ")", DL.fmt(mha.cache / cur.cache, 1) + "× smaller   " +
      DL.fmt(100 * (1 - cur.cache / mha.cache), 2) + "% reduction", DC.good, true);
    kvv("block parameters", DL.commas(cur.par) + "   (" + DL.fmt(100 * (1 - cur.par / mha.par), 2) + "% fewer)", DC.violet);
    kvv("decode intensity h/(g·b), bf16", DL.fmt(cur.I, 2) + " FLOP/byte   =  " + DL.fmt(100 * cur.I / (989.4e12 / 3350e9), 2) + "% of an H100's ridge", DC.good, true);
    kvv("quadratic MACs at n = " + DL.commas(n), DL.big(cur.quad) + "   (MHA: " + DL.big(mha.quad) + ")",
      cur.quad === mha.quad ? DC.accent : DC.bad);

    El("gq-readout").innerHTML =
      `${lab} · h = ${h}, g = ${g}, dₖ = ${dk}, L = ${L} · the cache is <b>${DL.commas(cur.cache / 2)}</b> elements = ` +
      `<b>${DL.fmt(cur.cache / 1024, 1)} KiB</b> per token, which is <b>${DL.fmt(mha.cache / cur.cache, 1)}×</b> smaller than ` +
      `multi-head — a <b>${DL.fmt(100 * (1 - cur.cache / mha.cache), 2)}%</b> reduction — and <b>${DL.fmt(cur.cache * n / GiB, 3)} GiB</b> ` +
      `at ${DL.commas(n)} tokens. The block also loses <b>${DL.fmt(100 * (1 - cur.par / mha.par), 2)}%</b> of its parameters, and the ` +
      `decode arithmetic intensity rises to <b>${DL.fmt(cur.I, 2)}</b> FLOP/byte. The quadratic term is <b>${DL.big(cur.quad)}</b> MACs — ` +
      `<b>identical</b> to multi-head attention's <b>${DL.big(mha.quad)}</b>, because every query head still scores against every key.`;
  }
  ["gq-h", "gq-g", "gq-L", "gq-n"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#gq-dk").on("change", draw);
  draw();
})();

/* ═════════ 28 · #ml-svg — MLA against the GQA staircase. All cache counts
   come from DL.kvCacheElems, including the MLA branch. ═══════════════════ */
(function () {
  const svg = d3.select("#ml-svg"); if (svg.empty()) return;
  const W = 760, H = 410;
  const El = id => document.getElementById(id);

  function draw() {
    const h = +El("ml-h").value, dk = +El("ml-dk").value, L = +El("ml-L").value,
      dc = +El("ml-dc").value, dr = +El("ml-dr").value;
    El("ml-hv").textContent = h; El("ml-Lv").textContent = L;
    El("ml-dcv").textContent = dc; El("ml-drv").textContent = dr;
    const f = DL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 46 }), g = f.g;
    const d = h * dk;
    const mla = DL.kvCacheElems({ kind: "mla", dc: dc, dr: dr, L: L });
    const mha = DL.kvCacheElems({ h: h, dh: dk, L: L, kind: "mha" });
    const perLayer = dc + dr, mhaPer = 2 * h * dk;
    const equivG = perLayer / (2 * dk);

    /* left — the widths, to scale */
    AT.title(g, 0, -10, "one token, one layer — box widths to scale");
    const px = v => Math.min(300, 300 * v / mhaPer);
    const bars = [
      ["hidden state x_t", d, DC.muted, false],
      ["latent c_t = x·W_DKV", dc, DC.accent, true],
      ["decoupled rotary k_tᴿ", dr, DC.a2, true],
      ["reconstructed keys (never stored)", h * dk, DC.line, false],
      ["reconstructed values (never stored)", h * dk, DC.line, false],
      ["MHA would cache", mhaPer, DC.bad, true]
    ];
    bars.forEach((b, i) => {
      const yy = 6 + i * 34;
      g.append("rect").attr("x", 0).attr("y", yy).attr("width", Math.max(2, px(b[1]))).attr("height", 18).attr("rx", 3)
        .attr("fill", b[2]).attr("fill-opacity", b[3] ? 0.6 : 0.16)
        .attr("stroke", b[2]).attr("stroke-dasharray", b[3] ? null : "3 3");
      AT.note(g, 0, yy - 3, b[0], b[3] ? b[2] : DC.muted, 9.5);
      AT.note(g, Math.max(2, px(b[1])) + 6, yy + 13, DL.commas(b[1]), b[3] ? b[2] : DC.muted, 9.5);
    });
    AT.note(g, 0, 6 + 6 * 34 + 8, "shaded = cached.  dashed = computed on demand and thrown away —", DC.muted, 9.5);
    AT.note(g, 0, 6 + 6 * 34 + 20, "and at decode not even computed, because W_UK is absorbed into W_Q.", DC.good, 9.5);

    /* right — the GQA staircase with MLA read off it */
    const rx = 380, rw = f.iw - rx, gh = 250;
    AT.title(g, rx, -10, "cache elements per token, whole stack");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const gs = []; for (let x2 = 1; x2 <= h; x2 *= 2) gs.push(x2);
    if (gs[gs.length - 1] !== h) gs.push(h);
    const pts = gs.map(x2 => ({ g: x2, v: DL.kvCacheElems({ h: h, g: x2, dh: dk, L: L, kind: "gqa" }) }));
    const x = d3.scaleLog().domain([0.8, h]).range([0, rw - 12]);
    const y = d3.scaleLog().domain([Math.min(mla, pts[0].v) / 2.5, mha * 2]).range([gh, 0]).clamp(true);
    DL.gridY(gg, y, rw - 12, 5);
    DL.axisB(gg, x, gh, 5, "KV heads g (grouped-query family)", d3.format("d"));
    DL.axisL(gg, y, 5, "elements / token", v => DL.big(v));
    DL.curve(gg, pts.map(p => [x(p.g), y(p.v)]), { stroke: DC.a2, w: 2.2 });
    pts.forEach(p => gg.append("circle").attr("cx", x(p.g)).attr("cy", y(p.v)).attr("r", 2.6).attr("fill", DC.a2));
    [[mha, "MHA", DC.bad], [DL.kvCacheElems({ h: h, dh: dk, L: L, kind: "mqa" }), "MQA", DC.violet], [mla, "MLA", DC.accent]].forEach((r, i) => {
      gg.append("line").attr("x1", 0).attr("x2", rw - 12).attr("y1", y(r[0])).attr("y2", y(r[0]))
        .attr("stroke", r[2]).attr("stroke-width", i === 2 ? 2.2 : 1.3).attr("stroke-dasharray", i === 2 ? null : "4 3");
      AT.note(gg, 4, y(r[0]) - 5, r[1] + "  " + DL.big(r[0]), r[2], 9.5);
    });
    if (equivG >= 0.8 && equivG <= h) {
      gg.append("line").attr("x1", x(equivG)).attr("x2", x(equivG)).attr("y1", y(mla)).attr("y2", gh)
        .attr("stroke", DC.accent).attr("stroke-dasharray", "2 3");
      AT.note(gg, x(equivG) + 4, gh - 8, "equivalent to GQA with " + DL.fmt(equivG, 2) + " groups", DC.accent, 9.5);
    }

    const extra = 2 * d * dc + 2 * dc * h * dk + d * dr;   /* W_DKV, W_UK, W_UV, W_KR, per layer */
    const kvv = DL.kv(g, 0, 6 + 6 * 34 + 44, { keyW: 288, size: 10.5, lead: 14.5 });
    kvv("cached per token per layer", DL.commas(perLayer) + " = d_c + dₕᴿ", DC.accent, true);
    kvv("   as a multiple of dₖ", DL.fmt(perLayer / dk, 3) + " × dₖ", DC.accent);
    kvv("whole stack", DL.commas(mla) + " elements  =  " + DL.fmt(mla * 2 / 1024, 1) + " KiB in bf16", DC.accent, true);
    kvv("MHA of the same shape", DL.commas(mha) + "   →  " + DL.fmt(mha / mla, 2) + "× smaller  (" + DL.fmt(100 * (1 - mla / mha), 2) + "%)", DC.bad, true);
    kvv("equivalent GQA group count", DL.fmt(equivG, 3), DC.a2, true);
    kvv("extra parameters per layer", DL.commas(extra) + "   (a block is " + DL.commas(4 * d * d) + ")", DC.violet);

    El("ml-readout").innerHTML =
      `h = ${h}, dₖ = ${dk}, L = ${L}, d_c = ${dc}, dₕᴿ = ${dr} · MLA caches <b>${DL.commas(perLayer)}</b> elements per token per ` +
      `layer — <b>${DL.fmt(perLayer / dk, 3)}×</b> the head dimension — for <b>${DL.commas(mla)}</b> over the stack, against ` +
      `<b>${DL.commas(mha)}</b> for multi-head attention of the same shape: <b>${DL.fmt(mha / mla, 2)}×</b> smaller, a ` +
      `<b>${DL.fmt(100 * (1 - mla / mha), 2)}%</b> reduction. Read off the grouped-query staircase, that is the cache of ` +
      `<b>GQA with ${DL.fmt(equivG, 2)} groups</b> — a group count that does not have to be an integer, which is the whole ` +
      `freedom the low-rank design buys. It costs <b>${DL.commas(extra)}</b> extra parameters per layer, and nothing at decode, ` +
      `because W_UK folds into W_Q and W_UV folds into W_O.`;
  }
  ["ml-h", "ml-L", "ml-dc", "ml-dr"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#ml-dk").on("change", draw);
  draw();
})();

/* ═════════ 29 · #sw-svg — windows, dilation and sinks. The mask comes from
   DL.windowMask; the receptive field is MEASURED by propagating reachability
   through the stack, not by evaluating the closed form, so the two can be
   compared. ═════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#sw-svg"); if (svg.empty()) return;
  const W = 760, H = 410;
  const El = id => document.getElementById(id);

  function reach(M, L, q) {
    const n = M.length;
    let cur = new Array(n).fill(false); cur[q] = true;
    const hist = [cur.slice()];
    for (let l = 0; l < L; l++) {
      const nxt = new Array(n).fill(false);
      for (let i = 0; i < n; i++) if (cur[i]) for (let j = 0; j < n; j++) if (isFinite(M[i][j])) nxt[j] = true;
      cur = nxt; hist.push(cur.slice());
    }
    return hist;
  }

  const PRE = { full: null, slide: [6, 1, 0], dil: [5, 2, 0], sink: [5, 1, 2] };
  function draw() {
    const n = +El("sw-n").value;
    let w = +El("sw-w").value, r = +El("sw-r").value, s = +El("sw-s").value;
    const pre = El("sw-p").value;
    if (pre === "full") { w = n; r = 1; s = 0; }
    else if (PRE[pre]) { w = PRE[pre][0]; r = PRE[pre][1]; s = PRE[pre][2]; }
    if (pre !== "custom") { El("sw-w").value = Math.min(16, w); El("sw-r").value = r; El("sw-s").value = s; }
    const L = +El("sw-L").value;
    El("sw-nv").textContent = n; El("sw-wv").textContent = w;
    El("sw-rv").textContent = r; El("sw-sv").textContent = s; El("sw-Lv").textContent = L;
    const f = DL.frame(svg, W, H, { l: 34, r: 16, t: 26, b: 46 }), g = f.g;
    const M = DL.windowMask(n, w, { causal: true, dil: r, sinks: s });
    let kept = 0, full = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { full++; if (isFinite(M[i][j])) kept++; }

    const cw = Math.min(6, 230 / n);
    AT.title(g, 0, -10, "the mask");
    AT.heat(g, M.map(row => row.map(v => isFinite(v) ? 1 : 0)), 0, 6, cw,
      v => v ? DC.accent : "#141821", {});
    AT.note(g, 0, 6 + n * cw + 15, DL.commas(kept) + " of " + DL.commas(full) + " causal entries  (" +
      DL.fmt(100 * kept / full, 1) + "%)", DC.accent, 9.5);

    /* middle — measured reachability */
    const mx = 260;
    AT.title(g, mx, -10, "reachability from the last position, layer by layer");
    const hist = reach(M, L, n - 1);
    const rw2 = Math.min(6, 230 / n);
    hist.forEach((row, l) => {
      g.selectAll(null).data(row.map((v, i) => [i, v])).enter().append("rect")
        .attr("x", d => mx + d[0] * rw2).attr("y", 6 + l * 11).attr("width", rw2 + 0.4).attr("height", 9)
        .attr("shape-rendering", "crispEdges")
        .attr("fill", d => d[1] ? DC.good : "#141821");
      AT.note(g, mx - 16, 6 + l * 11 + 8, String(l), DC.muted, 8);
    });
    const span = (() => { const row = hist[L]; let lo = n; for (let i = 0; i < n; i++) if (row[i]) { lo = i; break; } return n - lo; })();
    const pred = Math.min(n, L * (w - 1) * r + 1);
    AT.note(g, mx, 6 + (L + 1) * 11 + 14, "measured span after " + L + " layers: " + span + " positions", DC.good, 9.5);
    AT.note(g, mx, 6 + (L + 1) * 11 + 26, "closed form L·(w − 1)·r + 1 = " + pred, s > 0 ? DC.a2 : (span === pred ? DC.good : DC.a2), 9.5);
    if (s > 0) AT.note(g, mx, 6 + (L + 1) * 11 + 38, "(sinks make the measured span reach position 0 at once)", DC.a2, 9);

    /* right — cost */
    const rx = 520, rw3 = f.iw - rx, gh = 190;
    AT.title(g, rx, -10, "scores per layer against n");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const ns = [64, 256, 1024, 4096, 16384, 65536, 262144];
    const x = d3.scaleLog().domain([64, 262144]).range([0, rw3 - 10]);
    const yv = d3.scaleLog().domain([64 * 4, 262144 * 262144 / 2]).range([gh, 0]).clamp(true);
    DL.gridY(gg, yv, rw3 - 10, 4);
    DL.axisB(gg, x, gh, 4, "n", v => DL.big(v));
    DL.axisL(gg, yv, 4, "scores", v => DL.big(v));
    DL.curve(gg, ns.map(k => [x(k), yv(k * (k + 1) / 2)]), { stroke: DC.bad, w: 2 });
    DL.curve(gg, ns.map(k => [x(k), yv(k * Math.min(k, w * r + s))]), { stroke: DC.accent, w: 2 });
    DL.legend(gg, [{ label: "full causal, ∝ n²", color: DC.bad }, { label: "windowed, ∝ n·w", color: DC.accent }],
      4, gh + 30, { vertical: true, gap: 13, font: 9.5 });

    const kvv = DL.kv(g, 0, 6 + n * cw + 34, { keyW: 250, size: 10.5, lead: 14.5 });
    kvv("permitted entries", DL.commas(kept) + " of " + DL.commas(full), DC.accent, true);
    kvv("   cost ratio against full causal", DL.fmt(kept / full, 4) + "×", DC.good);
    kvv("measured span after " + L + " layers", span + " of " + n + " positions", DC.good, true);
    kvv("closed form L·(w − 1)·r + 1", String(pred), s > 0 ? DC.a2 : DC.muted);
    kvv("rolling-buffer cache size", DL.commas(w * r + s) + " tokens (fixed)", DC.a2, true);
    kvv("full-attention cache at n = 131 072", DL.commas(131072) + " tokens", DC.bad);
    kvv("a 4096-window, 32-layer stack spans", DL.commas(32 * 4095 + 1) + " tokens", DC.muted);

    El("sw-readout").innerHTML =
      `n = ${n}, window ${w}, dilation ${r}, ${s} sink token${s === 1 ? "" : "s"}, ${L} layers · the mask permits ` +
      `<b>${DL.commas(kept)}</b> of <b>${DL.commas(full)}</b> causal entries, a factor of <b>${DL.fmt(full / kept, 2)}×</b> fewer. ` +
      `Propagating reachability through the stack, the last position can see <b>${span}</b> positions back after ${L} layers, ` +
      `against the closed form L·(w − 1)·r + 1 = <b>${pred}</b>` +
      (s > 0 ? ` — the sinks make position 0 reachable in one layer, so the measured span is the whole sequence and the closed form no longer applies.`
             : (span === pred ? ` — exactly.` : `.`)) +
      ` The rolling buffer holds a fixed <b>${DL.commas(w * r + s)}</b> tokens however long the input is, which is the property ` +
      `that lets a windowed model stream indefinitely.`;
  }
  ["sw-n", "sw-w", "sw-r", "sw-s", "sw-L"].forEach(id => d3.select("#" + id).on("input", function () {
    El("sw-p").value = "custom"; draw();
  }));
  d3.select("#sw-p").on("change", draw);
  draw();
})();

/* ═════════ 30 · #la-svg — the two bracketings, and the rank the linear one
   costs. The exactness check on the left is the whole point of the
   re-association; the spectrum on the right is what it gives back. ═══════ */
(function () {
  const svg = d3.select("#la-svg"); if (svg.empty()) return;
  const W = 760, H = 410, DV = 8;
  const El = id => document.getElementById(id);

  function phi(x, kind) {
    if (kind === "relu") return Math.max(0, x);
    if (kind === "exp") return Math.exp(DL.clamp(x / 2, -30, 30));
    return x > 0 ? x + 1 : Math.exp(DL.clamp(x, -30, 30));      /* elu + 1 */
  }
  function build(n, dp, kind) {
    const Q = AT.gauss(n, dp, 71, 1), K = AT.gauss(n, dp, 72, 1), V = AT.gauss(n, DV, 73, 1);
    const PQ = Q.map(r => r.map(v => phi(v, kind))), PK = K.map(r => r.map(v => phi(v, kind)));
    return { Q: Q, K: K, V: V, PQ: PQ, PK: PK };
  }
  function quadWay(B, n) {
    const S = DL.matmul(B.PQ, DL.transpose(B.PK));
    const Z = DL.zeros2(n, DV);
    for (let i = 0; i < n; i++) {
      let z = 0; for (let j = 0; j < n; j++) z += S[i][j];
      for (let j = 0; j < n; j++) for (let c = 0; c < DV; c++) Z[i][c] += S[i][j] * B.V[j][c] / z;
    }
    return { Z: Z, A: S.map(r => { let s = 0; r.forEach(v => s += v); return r.map(v => v / s); }) };
  }
  function linWay(B, n, dp) {
    const KV = DL.matmul(DL.transpose(B.PK), B.V);                 /* (d′ × d_v) */
    const u = new Array(dp).fill(0);
    for (let j = 0; j < n; j++) for (let c = 0; c < dp; c++) u[c] += B.PK[j][c];
    const Z = DL.zeros2(n, DV);
    for (let i = 0; i < n; i++) {
      let z = 0; for (let c = 0; c < dp; c++) z += B.PQ[i][c] * u[c];
      for (let c = 0; c < DV; c++) { let t = 0; for (let a = 0; a < dp; a++) t += B.PQ[i][a] * KV[a][c]; Z[i][c] = t / z; }
    }
    return Z;
  }

  function draw() {
    const n = +El("la-n").value, dp = +El("la-d").value, kind = El("la-f").value, ord = El("la-o").value;
    El("la-nv").textContent = n; El("la-dv").textContent = dp;
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 26, b: 46 }), g = f.g;
    const B = build(n, dp, kind);
    const qw = quadWay(B, n), lw = linWay(B, n, dp);
    let mx = 0, sc = 0;
    for (let i = 0; i < n; i++) for (let c = 0; c < DV; c++) { mx = Math.max(mx, Math.abs(qw.Z[i][c] - lw[i][c])); sc = Math.max(sc, Math.abs(qw.Z[i][c])); }
    const macQ = n * n * dp + n * n * DV, macL = n * dp * DV + n * dp * DV;

    /* left — the two bracketings, sized by their intermediates */
    AT.title(g, 0, -10, "the two bracketings, intermediates drawn to scale");
    const scale = 150 / Math.sqrt(n * n);
    const rows = [
      ["(Q Kᵀ) V  — quadratic", [["QKᵀ", n * n, DC.bad], ["·V", n * DV, DC.accent]], macQ, n * n],
      ["Q (Kᵀ V)  — linear", [["KᵀV", dp * DV, DC.good], ["Q·", n * DV, DC.accent]], macL, dp * DV]
    ];
    rows.forEach((r, i) => {
      const yy = 8 + i * 96;
      AT.note(g, 0, yy, r[0], ord === (i ? "kv" : "qk") ? DC.ink : DC.muted, 11);
      let x0 = 0;
      r[1].forEach(b => {
        const sq = Math.max(6, Math.sqrt(b[1]) * scale);
        g.append("rect").attr("x", x0).attr("y", yy + 8).attr("width", sq).attr("height", Math.min(sq, 56))
          .attr("fill", b[2]).attr("fill-opacity", 0.35).attr("stroke", b[2]);
        AT.note(g, x0, yy + 8 + Math.min(sq, 56) + 12, b[0] + "  " + DL.commas(b[1]), b[2], 9);
        x0 += sq + 40;
      });
      AT.note(g, 230, yy + 20, DL.big(r[2]) + " MACs", DC.muted, 10);
      AT.note(g, 230, yy + 34, "peak intermediate " + DL.commas(r[3]), DC.muted, 9.5);
    });
    AT.note(g, 0, 210, "max |quadratic − linear| = " + DL.fmtE(mx, 3) + "  (output scale " + DL.fmt(sc, 3) + ")",
      mx / sc < 1e-10 ? DC.good : DC.bad, 10.5);
    AT.note(g, 0, 224, "the re-association is EXACT — this is round-off, not an approximation", DC.good, 9.5);

    /* right — the spectrum */
    const rx = 400, rw = f.iw - rx, gh = 250;
    AT.title(g, rx, -10, "singular values of the mixing matrix");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const SM = DL.sdpa(B.Q, B.K, B.V, {}).A;                       /* softmax mixing */
    function spec(A) {
      const G = DL.matmul(DL.transpose(A), A), e = DL.eigSym(G, 40);
      return e.w.map(v => Math.sqrt(Math.max(0, v)));
    }
    const sL = spec(qw.A), sS = spec(SM);
    const K = Math.min(n, 40);
    const x = d3.scaleLinear().domain([0, K]).range([0, rw - 12]);
    const y = d3.scaleLog().domain([Math.max(1e-18, Math.min(...sS.slice(0, K).filter(v => v > 0)) / 3), Math.max(...sS) * 2]).range([gh, 0]).clamp(true);
    DL.gridY(gg, y, rw - 12, 5);
    DL.axisB(gg, x, gh, 5, "index");
    DL.axisL(gg, y, 5, "σ", v => DL.fmtE(v, 0));
    DL.curve(gg, sS.slice(0, K).map((v, i) => [x(i), y(Math.max(v, 1e-18))]), { stroke: DC.accent, w: 2.2 });
    DL.curve(gg, sL.slice(0, K).map((v, i) => [x(i), y(Math.max(v, 1e-18))]), { stroke: DC.bad, w: 2.2 });
    gg.append("line").attr("x1", x(dp)).attr("x2", x(dp)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
    AT.note(gg, x(dp) + 4, 12, "rank ≤ d′ = " + dp, DC.bad, 9.5);
    DL.legend(gg, [{ label: "softmax mixing", color: DC.accent }, { label: "kernel mixing φ(Q)φ(K)ᵀ", color: DC.bad }],
      4, gh + 32, { vertical: true, gap: 13, font: 9.5 });

    /* the best rank-d′ approximation residual of the softmax matrix */
    let tot = 0, tail = 0;
    sS.forEach((v, i) => { tot += v * v; if (i >= dp) tail += v * v; });
    const resid = Math.sqrt(tail / Math.max(1e-300, tot));
    /* a singular value is the square root of an eigenvalue of AᵀA, so the
       numerical floor is ~√(machine ε) relative, not machine ε relative. */
    const rankL = sL.filter(v => v > sL[0] * 1e-7).length;
    const rankS = sS.filter(v => v > sS[0] * 1e-7).length;

    const kvv = DL.kv(g, 0, 246, { keyW: 288, size: 10.5, lead: 14.5 });
    kvv("MACs, (QKᵀ)V", DL.big(macQ), DC.bad, true);
    kvv("MACs, Q(KᵀV)", DL.big(macL) + "   →  " + DL.fmt(macQ / macL, 1) + "× fewer", DC.good, true);
    kvv("agreement between the two", DL.fmtE(mx / Math.max(1e-300, sc), 2) + " relative", mx / sc < 1e-10 ? DC.good : DC.bad);
    kvv("numerical rank, kernel mixing", DL.commas(rankL) + " (bound d′ = " + dp + ")", DC.bad, true);
    kvv("numerical rank, softmax mixing", DL.commas(rankS) + " of n = " + n, DC.accent, true);
    kvv("residual of the best rank-" + dp + " fit to the softmax mixing", DL.fmt(resid, 4), resid > 0.2 ? DC.bad : DC.a2);
    kvv("recurrent state capacity d′·d_v / (n·d_v) = d′/n", DL.fmt(dp / n, 4), dp / n < 0.3 ? DC.bad : DC.a2, true);

    El("la-readout").innerHTML =
      `n = ${n}, d′ = ${dp}, φ = ${kind} · the quadratic bracketing costs <b>${DL.big(macQ)}</b> MACs and forms an ` +
      `<b>${DL.commas(n * n)}</b>-element intermediate; the linear one costs <b>${DL.big(macL)}</b> and forms ` +
      `<b>${DL.commas(dp * DV)}</b>. They agree to <b>${DL.fmtE(mx / Math.max(1e-300, sc), 2)}</b> relative — the re-association ` +
      `is exact. What it costs is rank: the kernel mixing matrix has numerical rank <b>${rankL}</b> against the softmax ` +
      `matrix's <b>${rankS}</b> of ${n}, and the best rank-${dp} approximation to the softmax mixing still leaves a relative ` +
      `residual of <b>${DL.fmt(resid, 4)}</b>. In recurrent form the state holds d′·d_v numbers against an input of n·d_v, ` +
      `so by §01's counting argument it retains <b>${DL.fmt(dp / n, 4)}</b> of the input — part 4's bottleneck, returned.`;
  }
  ["la-n", "la-d"].forEach(id => d3.select("#" + id).on("input", draw));
  ["la-f", "la-o"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 31 · #fa-svg — tiled attention with an online softmax.
   The running (m, ℓ, O) recurrence is the exact incremental form of the
   max-subtracting softmax DL.softmax uses; the figure checks the tiled
   result against a direct DL.sdpa call and prints the difference. ════════ */
(function () {
  const svg = d3.select("#fa-svg"); if (svg.empty()) return;
  const W = 760, H = 420, D = 6;
  const El = id => document.getElementById(id);

  function draw() {
    const n = +El("fa-n").value, Bc = +El("fa-b").value, Mkb = +El("fa-m").value, d = +El("fa-d").value;
    const nb = Math.ceil(n / Bc);
    let k = Math.min(+El("fa-k").value, nb);
    El("fa-k").max = nb;
    El("fa-nv").textContent = n; El("fa-bv").textContent = Bc;
    El("fa-kv").textContent = k; El("fa-mv").textContent = Mkb;
    const f = DL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 46 }), g = f.g;

    const Q = AT.gauss(1, D, 81, 1), K = AT.gauss(n, D, 82, 1), V = AT.gauss(n, D, 83, 1);
    const direct = DL.sdpa(Q, K, V, {});
    const s = direct.S[0];

    /* the online recurrence, block by block */
    let m = -Infinity, l = 0, O = new Array(D).fill(0);
    const trace = [];
    for (let b = 0; b < k; b++) {
      const lo = b * Bc, hi = Math.min(n, lo + Bc);
      let mt = -Infinity; for (let j = lo; j < hi; j++) mt = Math.max(mt, s[j]);
      let lt = 0; const Ot = new Array(D).fill(0);
      for (let j = lo; j < hi; j++) { const e = Math.exp(s[j] - mt); lt += e; for (let c = 0; c < D; c++) Ot[c] += e * V[j][c]; }
      for (let c = 0; c < D; c++) Ot[c] /= lt;
      const mn = Math.max(m, mt);
      const a = isFinite(m) ? Math.exp(m - mn) : 0, bb = Math.exp(mt - mn);
      const ln = a * l + bb * lt;
      const On = new Array(D);
      for (let c = 0; c < D; c++) On[c] = (a * l * O[c] + bb * lt * Ot[c]) / ln;
      trace.push({ b: b, lo: lo, hi: hi, mt: mt, m: mn, l: ln, resc: a, O: On.slice() });
      m = mn; l = ln; O = On;
    }
    let err = 0, scl = 0;
    for (let c = 0; c < D; c++) { err = Math.max(err, Math.abs(O[c] - direct.Z[0][c])); scl = Math.max(scl, Math.abs(direct.Z[0][c])); }

    /* left — the score row, split into blocks */
    AT.title(g, 0, -10, "one query row's scores, in blocks of " + Bc);
    const bw = Math.min(9, 300 / n);
    const smin = Math.min(...s), smax = Math.max(...s);
    const y0 = 10, bh = 70;
    const ys = v => y0 + bh - bh * (v - smin) / Math.max(1e-9, smax - smin);
    for (let j = 0; j < n; j++) {
      const done = j < k * Bc;
      g.append("rect").attr("x", j * bw).attr("y", ys(s[j])).attr("width", bw - 0.6).attr("height", y0 + bh - ys(s[j]))
        .attr("fill", done ? DC.accent : DC.line).attr("fill-opacity", done ? 0.85 : 0.4);
    }
    for (let b = 0; b <= nb; b++) g.append("line").attr("x1", b * Bc * bw).attr("x2", b * Bc * bw)
      .attr("y1", y0 - 4).attr("y2", y0 + bh + 4).attr("stroke", DC.muted).attr("stroke-dasharray", "2 2");
    AT.note(g, 0, y0 + bh + 16, k + " of " + nb + " key blocks processed", DC.accent, 9.5);

    /* the running state */
    const ty = y0 + bh + 34;
    AT.title(g, 0, ty - 6, "the running state, block by block");
    const kvv = DL.kv(g, 0, ty + 12, { keyW: 230, size: 10.5, lead: 14 });
    kvv("running max m", isFinite(m) ? DL.fmt(m, 6) : "−∞", DC.a2, true);
    kvv("running sum ℓ", DL.fmt(l, 6), DC.a2);
    kvv("last rescaling factor e⁽ᵐ⁻ᵐ′⁾", trace.length ? DL.fmt(trace[trace.length - 1].resc, 6) : "—", DC.violet);
    kvv("blocks processed", k + " of " + nb);
    kvv("‖running output‖", DL.fmt(AT.rowNorm(O), 6), DC.accent);
    kvv("direct one-shot output ‖·‖", DL.fmt(AT.rowNorm(direct.Z[0]), 6), DC.good);
    kvv(k === nb ? "max |tiled − direct|" : "max |partial − direct| (not yet complete)",
      DL.fmtE(err, 3), k === nb ? (err / scl < 1e-12 ? DC.good : DC.bad) : DC.muted, k === nb);
    if (k === nb) kvv("   relative", DL.fmtE(err / Math.max(1e-300, scl), 2), err / scl < 1e-12 ? DC.good : DC.bad, true);

    /* right — HBM traffic */
    const rx = 400, rw = f.iw - rx, gh = 230;
    AT.title(g, rx, -10, "HBM accesses against n");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const Mel = Mkb * 1024 / 4;                        /* elements, fp32 */
    const ns = [256, 1024, 4096, 16384, 65536, 262144];
    const naive = q => q * d + 3 * q * q, tiled = q => q * q * d * d / Mel;
    const x = d3.scaleLog().domain([256, 262144]).range([0, rw - 12]);
    const y = d3.scaleLog().domain([Math.min(tiled(256), naive(256)) / 3, naive(262144) * 3]).range([gh, 0]).clamp(true);
    DL.gridY(gg, y, rw - 12, 5);
    DL.axisB(gg, x, gh, 5, "n", v => DL.big(v));
    DL.axisL(gg, y, 5, "elements moved", v => DL.big(v));
    DL.curve(gg, ns.map(q => [x(q), y(naive(q))]), { stroke: DC.bad, w: 2.2 });
    DL.curve(gg, ns.map(q => [x(q), y(tiled(q))]), { stroke: DC.good, w: 2.2 });
    DL.curve(gg, ns.map(q => [x(q), y(q * q * d * d / (Mel / 4))]), { stroke: DC.good, w: 1.2, dash: "3 3" });
    DL.legend(gg, [{ label: "naive: Θ(n·d + n²)", color: DC.bad },
                   { label: "tiled: Θ(n²d²/M), M = " + Mkb + " KB", color: DC.good },
                   { label: "tiled at M/4", color: DC.good, dash: "3 3" }],
      4, gh + 32, { vertical: true, gap: 13, font: 9.5 });
    const BcR = Math.ceil(Mel / (4 * d)), BrR = Math.min(BcR, d);
    AT.note(gg, 4, gh + 76, "the standard rule at M = " + Mkb + " KB, d = " + d + ":  B_c = " + DL.commas(BcR) + ",  B_r = " + DL.commas(BrR), DC.muted, 9.5);

    El("fa-readout").innerHTML =
      `n = ${n}, block size ${Bc}, ${k} of ${nb} blocks processed, M = ${Mkb} KB, d = ${d} · the running state is ` +
      `m = <b>${isFinite(m) ? DL.fmt(m, 4) : "−∞"}</b>, ℓ = <b>${DL.fmt(l, 4)}</b>` +
      (k === nb
        ? `, and the completed tiled output differs from the direct one-shot computation by <b>${DL.fmtE(err, 3)}</b> ` +
          `(<b>${DL.fmtE(err / Math.max(1e-300, scl), 2)}</b> relative) — round-off. The tiling is <b>exact</b>.`
        : `. Advance the block slider to the last block to compare against the direct computation.`) +
      ` At n = 8192 the naive kernel moves <b>${DL.big(naive(8192))}</b> elements through HBM and the tiled one ` +
      `<b>${DL.big(tiled(8192))}</b> — a factor of <b>${DL.fmt(naive(8192) / tiled(8192), 1)}×</b>. That ratio is 3M/d² = ` +
      `<b>${DL.fmt(3 * Mel / (d * d), 1)}</b>, the three coming from the three round trips the naive kernel makes through the ` +
      `n² object. The arithmetic is <b>identical</b> in both.`;
  }
  ["fa-n", "fa-b", "fa-k", "fa-m"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#fa-d").on("change", draw);
  draw();
})();

/* ═════════ 32 · #lc-svg — finding one key among n. The closed form
   p ≈ 1/(1 + (n−1)·exp(σ²/2 − m}) is drawn against sampled measurements, and
   the margin law m*(n) = ln(n−1) + σ²/2 is FITTED to the measurements so the
   intercept can be checked against σ²/2. ═════════════════════════════════ */
(function () {
  const svg = d3.select("#lc-svg"); if (svg.empty()) return;
  const W = 760, H = 410;
  const El = id => document.getElementById(id);
  const NS = [8, 32, 128, 512, 2048, 8192, 32768, 131072];

  function pTarget(n, m, sd, trials, seed) {
    const r = DL.rng(seed);
    const cap = Math.min(n - 1, 4096);                 /* sample the tail, scale the sum */
    const scl = (n - 1) / cap;
    let tot = 0;
    for (let t = 0; t < trials; t++) {
      let S = 0;
      for (let j = 0; j < cap; j++) S += Math.exp(DL.randn(r) * sd - m);
      tot += 1 / (1 + S * scl);
    }
    return tot / trials;
  }
  function mStar(n, sd, p, trials, seed) {
    let lo = 0, hi = 40;
    for (let i = 0; i < 26; i++) { const mid = (lo + hi) / 2; if (pTarget(n, mid, sd, trials, seed) < p) lo = mid; else hi = mid; }
    return (lo + hi) / 2;
  }

  function draw() {
    const m = +El("lc-m").value, sd = +El("lc-s").value, T = +El("lc-t").value, P = +El("lc-p").value;
    El("lc-mv").textContent = DL.fmt(m, 1); El("lc-sv").textContent = DL.fmt(sd, 1);
    const f = DL.frame(svg, W, H, { l: 56, r: 16, t: 26, b: 46 }), g = f.g;
    const MS = [2, 4, 6, 8, 10, 12];
    const closed = (n, mm) => 1 / (1 + (n - 1) * Math.exp(sd * sd / 2 - mm));

    const gw = 330, gh = 250;
    AT.title(g, 0, -10, "probability on the one correct key");
    const x = d3.scaleLog().domain([8, 131072]).range([0, gw]);
    const y = d3.scaleLog().domain([1e-5, 1.4]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 5, "candidate keys n", v => DL.big(v));
    DL.axisL(g, y, 5, "p(right key)", v => DL.fmtE(v, 0));
    MS.forEach(mm => {
      const sel = Math.abs(mm - m) < 0.75;
      DL.curve(g, NS.map(n => [x(n), y(DL.clamp(closed(n, mm), 1e-5, 1.4))]),
        { stroke: DC.muted, w: 1, dash: "3 3", op: sel ? 0.9 : 0.35 });
      const pts = NS.map(n => [x(n), y(DL.clamp(pTarget(n, mm, sd, sel ? T : Math.min(T, 300), 7 + n), 1e-5, 1.4))]);
      DL.curve(g, pts, { stroke: sel ? DC.accent : DC.a2, w: sel ? 2.6 : 1.1, op: sel ? 1 : 0.45 });
      AT.note(g, gw - 40, y(DL.clamp(closed(131072, mm), 1e-5, 1.4)) + 3, "m=" + mm, sel ? DC.accent : DC.muted, 8.5);
    });
    DL.legend(g, [{ label: "measured", color: DC.accent }, { label: "closed form 1/(1+(n−1)·exp(σ²/2−m))", color: DC.muted, dash: "3 3" }],
      2, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    /* right — the margin law */
    const rx = gw + 62, rw = f.iw - rx;
    AT.title(g, rx, -10, "margin needed to hold p = " + P);
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const use = NS.filter(n => n >= 32);
    const ms = use.map(n => ({ n: n, v: mStar(n, sd, P, Math.max(200, Math.min(T, 1200)), 1234) }));
    /* fit v = a·ln n + b */
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    ms.forEach(p => { const X = Math.log(p.n); sx += X; sy += p.v; sxx += X * X; sxy += X * p.v; });
    const K = ms.length, slope = (K * sxy - sx * sy) / (K * sxx - sx * sx), inter = (sy - slope * sx) / K;
    const xl = d3.scaleLog().domain([32, 131072]).range([0, rw - 12]);
    const yl = d3.scaleLinear().domain([0, Math.max(...ms.map(p => p.v)) * 1.2]).range([gh, 0]);
    DL.gridY(gg, yl, rw - 12, 5);
    DL.axisB(gg, xl, gh, 5, "candidate keys n", v => DL.big(v));
    DL.axisL(gg, yl, 5, "margin m*, nats");
    DL.curve(gg, use.map(n => [xl(n), yl(Math.log(n - 1) + sd * sd / 2 + Math.log(P / (1 - P)))]),
      { stroke: DC.muted, w: 1.4, dash: "4 3" });
    DL.curve(gg, ms.map(p => [xl(p.n), yl(p.v)]), { stroke: DC.accent, w: 2.4 });
    ms.forEach(p => gg.append("circle").attr("cx", xl(p.n)).attr("cy", yl(p.v)).attr("r", 2.6).attr("fill", DC.accent));
    DL.legend(gg, [{ label: "measured", color: DC.accent },
                   { label: "ln(n−1) + σ²/2 + ln(p/(1−p))", color: DC.muted, dash: "4 3" }],
      4, gh + 34, { vertical: true, gap: 13, font: 9.5 });

    const pm = pTarget(4096, m, sd, T, 99), pc = closed(4096, m);
    const kvv = DL.kv(g, 0, gh + 74, { keyW: 288, size: 10.5, lead: 14.5 });
    kvv("p(right key) at n = 4096, m = " + DL.fmt(m, 1), DL.fmt(pm, 6), DC.accent, true);
    kvv("   the closed form predicts", DL.fmt(pc, 6), Math.abs(pm - pc) < 0.02 ? DC.good : DC.a2);
    kvv("p at n = 131072, same margin", DL.fmt(pTarget(131072, m, sd, Math.min(T, 400), 99), 6), DC.bad, true);
    kvv("fitted slope of m* against ln n", DL.fmt(slope, 4) + "   (the derivation says 1)", Math.abs(slope - 1) < 0.03 ? DC.good : DC.a2, true);
    kvv("fitted intercept", DL.fmt(inter, 4), DC.a2);
    kvv("   σ²/2 + ln(p/(1−p)) =", DL.fmt(sd * sd / 2 + Math.log(P / (1 - P)), 4),
      Math.abs(inter - (sd * sd / 2 + Math.log(P / (1 - P)))) < 0.15 ? DC.good : DC.a2, true);
    kvv("extra margin, 4096 → 131072 keys", DL.fmt(Math.log(131072 / 4096), 4) + " nats", DC.good, true);

    El("lc-readout").innerHTML =
      `margin m = ${DL.fmt(m, 1)}, distractor spread σ = ${DL.fmt(sd, 1)}, ${DL.commas(T)} draws per point · at n = 4096 the ` +
      `softmax puts <b>${DL.fmt(pm, 5)}</b> on the correct key against a closed-form prediction of <b>${DL.fmt(pc, 5)}</b>; ` +
      `at n = 131 072, with the <b>same</b> margin, it puts <b>${DL.fmt(pTarget(131072, m, sd, Math.min(T, 400), 99), 5)}</b>. ` +
      `Fitting the margin needed to hold p = ${P} gives a slope of <b>${DL.fmt(slope, 4)}</b> against ln n — the derivation says ` +
      `<b>1</b> — and an intercept of <b>${DL.fmt(inter, 4)}</b> against the predicted σ²/2 + ln(p/(1−p)) = ` +
      `<b>${DL.fmt(sd * sd / 2 + Math.log(P / (1 - P)), 4)}</b>. Extending from 4 096 to 131 072 keys therefore costs exactly ` +
      `<b>${DL.fmt(Math.log(32), 3)}</b> nats of extra margin, which is a small number — and it still has to be learned.`;
  }
  ["lc-m", "lc-s"].forEach(id => d3.select("#" + id).on("input", draw));
  ["lc-t", "lc-p"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 33 · #ih-svg — a hand-wired induction circuit.
   Layer 1 is a previous-token head (its attention map IS the first
   sub-diagonal, set exactly, not learned). Layer 2 scores a query's own
   token identity against each key's PREVIOUS-token slot, so it lands on the
   position AFTER the current token's earlier occurrence, and copies that
   position's token. Nothing here is trained; the point is that the circuit
   is writable in closed form and that removing either half breaks it. ════ */
(function () {
  const svg = d3.select("#ih-svg"); if (svg.empty()) return;
  const W = 760, H = 430;
  const El = id => document.getElementById(id);

  function draw() {
    const V = +El("ih-v").value, seg = +El("ih-n").value, beta = +El("ih-b").value,
      seed = +El("ih-s").value, abl = El("ih-a").value;
    El("ih-vv").textContent = V; El("ih-nv").textContent = seg;
    El("ih-bv").textContent = beta; El("ih-sv").textContent = seed;
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 26, b: 46 }), g = f.g;
    const n = 2 * seg;

    /* the sequence: a random segment, then an exact repeat. The segment is
       drawn WITHOUT replacement when the vocabulary allows it, because a
       symbol that occurs twice inside the segment gives the circuit two
       legitimate continuations to split its weight between — a real property
       of the algorithm, and one that would otherwise be read as an error. */
    const r = DL.rng(seed * 17 + 3), x = [];
    const distinct = V >= seg;
    if (distinct) {
      const pool = []; for (let i = 0; i < V; i++) pool.push(i);
      DL.shuffle(pool, r);
      for (let i = 0; i < seg; i++) x.push(pool[i]);
    } else {
      for (let i = 0; i < seg; i++) x.push(Math.floor(r() * V));
    }
    for (let i = 0; i < seg; i++) x.push(x[i]);

    /* LAYER 1 — the previous-token head, written down exactly */
    const A1 = DL.zeros2(n, n);
    for (let t = 0; t < n; t++) A1[t][Math.max(0, t - 1)] = 1;
    const prev = DL.zeros2(n, V);
    if (abl !== "prev") for (let t = 0; t < n; t++) for (let s = 0; s < n; s++)
      if (A1[t][s]) prev[t][x[s]] += A1[t][s];

    /* LAYER 2 — the induction head */
    const S2 = DL.zeros2(n, n);
    for (let t = 0; t < n; t++) for (let s = 0; s < n; s++)
      S2[t][s] = s <= t ? beta * prev[s][x[t]] : -Infinity;
    const A2 = abl === "ind" ? DL.zeros2(n, n).map(row => row.map(() => 1 / n)) : DL.softmaxRows(S2);
    const pred = DL.zeros2(n, V);
    for (let t = 0; t < n; t++) for (let s = 0; s <= t; s++) pred[t][x[s]] += A2[t][s];

    const pTrue = [];
    for (let t = 0; t < n - 1; t++) pTrue.push(abl === "ind" ? 1 / V : pred[t][x[t + 1]]);
    const before = pTrue.slice(0, seg), after = pTrue.slice(seg);
    const mB = d3.mean(before), mA = d3.mean(after);
    const accB = before.filter((v, i) => pred[i][x[i + 1]] >= Math.max(...pred[i]) - 1e-12).length / before.length;
    const accA = after.filter((v, i) => pred[seg + i][x[seg + i + 1]] >= Math.max(...pred[seg + i]) - 1e-12).length / after.length;

    /* the two maps */
    const cw = Math.min(13, 170 / n);
    AT.title(g, 0, -10, "layer 1 — the previous-token head");
    AT.heat(g, A1, 0, 6, cw, d3.scaleSequential(d3.interpolateViridis).domain([0, 1]), { stroke: n <= 24 ? "#0f1117" : null });
    AT.note(g, 0, 6 + n * cw + 14, abl === "prev" ? "ABLATED — writes nothing" : "the first sub-diagonal, exactly",
      abl === "prev" ? DC.bad : DC.good, 9.5);
    const rx2 = 230;
    AT.title(g, rx2, -10, "layer 2 — the induction head");
    AT.heat(g, A2.map(row => row.map(v => isFinite(v) ? v : 0)), rx2, 6, cw,
      d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...A2.map(row => Math.max(...row.map(v => isFinite(v) ? v : 0))))]),
      { stroke: n <= 24 ? "#0f1117" : null });
    g.append("line").attr("x1", rx2).attr("x2", rx2 + n * cw).attr("y1", 6 + seg * cw).attr("y2", 6 + seg * cw)
      .attr("stroke", DC.a2).attr("stroke-dasharray", "3 2");
    AT.note(g, rx2, 6 + n * cw + 14, abl === "none" ? "the stripe is displaced by exactly the segment length, " + seg
      : abl === "prev" ? "no stripe: the keys no longer describe the preceding token" : "ABLATED — uniform",
      abl === "none" ? DC.good : DC.bad, 9.5);

    /* the sequence itself */
    const sy = 6 + n * cw + 34;
    AT.note(g, 0, sy, "the sequence — a random segment, then an exact repeat", DC.muted, 9.5);
    for (let t = 0; t < n; t++) {
      g.append("rect").attr("x", t * 15).attr("y", sy + 6).attr("width", 13).attr("height", 15).attr("rx", 2)
        .attr("fill", t < seg ? DC.violet : DC.accent).attr("fill-opacity", 0.45);
      g.append("text").attr("x", t * 15 + 6.5).attr("y", sy + 17).attr("text-anchor", "middle")
        .attr("font-size", 8.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(x[t]);
    }

    /* probability on the true next token */
    const py = sy + 42, pw = 420, ph = 110;
    AT.title(g, 0, py - 6, "probability the circuit gives the TRUE next token");
    const gg = g.append("g").attr("transform", `translate(0,${py + 4})`);
    const xp = d3.scaleLinear().domain([0, n - 1]).range([0, pw]);
    const yp = d3.scaleLinear().domain([0, 1.05]).range([ph, 0]);
    DL.gridY(gg, yp, pw, 4);
    DL.axisB(gg, xp, ph, 6, "position t");
    DL.axisL(gg, yp, 4, "p");
    gg.selectAll("rect.p").data(pTrue.map((v, i) => [i, v])).join("rect").attr("class", "p")
      .attr("x", d => xp(d[0])).attr("y", d => yp(d[1])).attr("width", Math.max(1.5, pw / n - 2))
      .attr("height", d => ph - yp(d[1])).attr("fill", d => d[0] < seg ? DC.violet : DC.accent).attr("fill-opacity", 0.8);
    gg.append("line").attr("x1", xp(seg)).attr("x2", xp(seg)).attr("y1", 0).attr("y2", ph)
      .attr("stroke", DC.a2).attr("stroke-width", 1.6);
    AT.note(gg, xp(seg) + 4, 12, "the repeat starts here", DC.a2, 9.5);
    gg.append("line").attr("x1", 0).attr("x2", pw).attr("y1", yp(1 / V)).attr("y2", yp(1 / V))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    AT.note(gg, 4, yp(1 / V) - 4, "chance, 1/" + V + " = " + DL.fmt(1 / V, 3), DC.muted, 9);

    const kvv = DL.kv(g, 460, py + 8, { keyW: 216, size: 10.5, lead: 14.5 });
    kvv("mean p before the repeat", DL.fmt(mB, 5), DC.violet, true);
    kvv("mean p after the repeat", DL.fmt(mA, 5), mA > 0.8 ? DC.good : DC.bad, true);
    kvv("   ratio", DL.fmt(mA / Math.max(1e-12, mB), 1) + "×", mA > 0.8 ? DC.good : DC.bad);
    kvv("accuracy before / after", DL.fmt(100 * accB, 0) + "% / " + DL.fmt(100 * accA, 0) + "%", DC.ink);
    kvv("chance", DL.fmt(1 / V, 4), DC.muted);
    kvv("ablation", abl === "none" ? "none" : (abl === "prev" ? "previous-token head" : "induction head"),
      abl === "none" ? DC.good : DC.bad, true);

    El("ih-readout").innerHTML =
      `vocabulary ${V}, segment ${seg} (sequence ${n}), sharpness β = ${beta}, ` +
      `${abl === "none" ? "full circuit" : "ablating the " + (abl === "prev" ? "previous-token head" : "induction head")}, ` +
      `${distinct ? "segment symbols all distinct" : "segment symbols drawn with replacement"} · ` +
      `the mean probability on the true next token is <b>${DL.fmt(mB, 5)}</b> before the repeat begins and ` +
      `<b>${DL.fmt(mA, 5)}</b> after it — a factor of <b>${DL.fmt(mA / Math.max(1e-12, mB), 1)}×</b>, against a chance level of ` +
      `<b>${DL.fmt(1 / V, 4)}</b>. Accuracy goes from <b>${DL.fmt(100 * accB, 0)}%</b> to <b>${DL.fmt(100 * accA, 0)}%</b>. ` +
      (abl === "none"
        ? `Before the repeat, the true next symbol has not yet occurred anywhere in the context, so a circuit that can only ` +
          `COPY assigns it almost nothing — below chance, correctly. After the boundary every position finds its match. ` +
          `That step is in-context learning, with no weights specific to any of these symbols.` +
          (distinct ? ` The one position that is still ambiguous is the first of the repeat: position 0's previous-token slot ` +
            `holds its own symbol, so it matches too, and the weight splits.`
            : ` Lower the vocabulary below the segment length and symbols repeat inside the segment, which gives the circuit ` +
              `two legitimate continuations to split between — visible as the shorter bars.`)
        : abl === "prev"
          ? `With the previous-token head removed, layer 2's keys no longer say what preceded them, every score is equal, and ` +
            `the second half is no better than the first — the circuit needs BOTH layers and they compose across, not within, a layer.`
          : `With the induction head removed there is no lookup at all and the output is uniform over the vocabulary.`);
  }
  ["ih-v", "ih-n", "ih-b", "ih-s"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#ih-a").on("change", draw);
  draw();
})();

/* ═════════ 34 · #tk-svg — convolution, recurrence, attention on the same
   job. The gradient-at-distance panel is MEASURED by central differences
   through small stacks of each operator, not evaluated from a formula. ═══ */
(function () {
  const svg = d3.select("#tk-svg"); if (svg.empty()) return;
  const W = 760, H = 410, DM = 8, NM = 28, LM = 4;
  const El = id => document.getElementById(id);
  const OPS = [["convolution", DC.a2], ["recurrence", DC.violet], ["attention", DC.accent]];

  /* small stacks, for the measured gradient */
  function stackConv(X, fw, seed) {
    const n = X.length, r = DL.rng(91 + (seed || 0)), k = [];
    for (let i = 0; i < fw; i++) k.push(DL.randn(r) / Math.sqrt(fw));
    let Z = X.map(row => row.slice());
    for (let l = 0; l < LM; l++) {
      const M = AT.gauss(DM, DM, 700 + l + 10 * (seed || 0), 1 / Math.sqrt(DM)), Y = DL.zeros2(n, DM), half = (fw - 1) / 2;
      const P = DL.matmul(Z, M);
      for (let i = 0; i < n; i++) for (let t = -half; t <= half; t++) {
        const j = i + t; if (j < 0 || j >= n) continue;
        for (let c = 0; c < DM; c++) Y[i][c] += k[t + half] * P[j][c];
      }
      Z = Y.map(row => row.map(v => Math.tanh(v)));
    }
    return Z;
  }
  function stackRec(X, seed) {
    const n = X.length;
    let Z = X.map(row => row.slice());
    for (let l = 0; l < LM; l++) {
      const Wm = AT.mat(DM, 800 + l + 10 * (seed || 0), 0.95), U = AT.gauss(DM, DM, 900 + l + 10 * (seed || 0), 1 / Math.sqrt(DM));
      const Hh = DL.zeros2(n, DM);
      let h = new Array(DM).fill(0);
      for (let t = 0; t < n; t++) {
        const z = DL.vecmat(h, Wm), u = DL.vecmat(Z[t], U);
        h = z.map((v, c) => Math.tanh(v + u[c]));
        Hh[t] = h.slice();
      }
      Z = Hh;
    }
    return Z;
  }
  function stackAttn(X, seed) {
    let Z = X.map(row => row.slice());
    const M = DL.causalMask(X.length);
    for (let l = 0; l < LM; l++) {
      const P = DL.mhaInit(DM, 4, 4, 2, { seed: 1000 + l + 10 * (seed || 0) });
      Z = DL.mhaForward(Z, P, { mask: M }).Y.map(row => row.map(v => Math.tanh(v)));
    }
    return Z;
  }
  /* ‖∂y_{n−1}/∂x_j‖ by central differences, GEOMETRIC MEAN over SEEDS
     independent draws of weights and input, so the curve is the operator's
     law and not one draw's luck (a single draw of the attention stack swings
     by orders of magnitude from position to position).                  */
  const SEEDS = 6;
  function gradAtDistance(fn) {
    const eps = 1e-4, acc = new Array(NM).fill(0), zero = new Array(NM).fill(false);
    for (let sd = 0; sd < SEEDS; sd++) {
      const X = AT.gauss(NM, DM, 55 + sd, 1);
      for (let j = 0; j < NM; j++) {
        let a2 = 0;
        for (let c = 0; c < DM; c++) {
          const o = X[j][c];
          X[j][c] = o + eps; const a = fn(X, sd)[NM - 1];
          X[j][c] = o - eps; const b = fn(X, sd)[NM - 1];
          X[j][c] = o;
          for (let e = 0; e < DM; e++) { const dv = (a[e] - b[e]) / (2 * eps); a2 += dv * dv; }
        }
        const gj = Math.sqrt(a2);
        if (gj < 1e-14) zero[j] = true; else acc[j] += Math.log(gj) / SEEDS;
      }
    }
    const out = [];
    for (let j = 0; j < NM; j++) out.push({ dist: NM - 1 - j, g: zero[j] ? 0 : Math.exp(acc[j]) });
    return out.reverse();
  }

  function draw() {
    const n = Math.pow(2, +El("tk-n").value), fw = +El("tk-f").value, d = +El("tk-d").value, cmp = El("tk-c").value;
    const hQ = Math.max(1, Math.round(d / 128)), gKV = Math.max(1, Math.round(hQ / 4));
    /* the n at which one layer's cache (2·g·dₖ·n) equals one layer's attention weights (2d² + 2·d·g·dₖ) */
    const nCacheEqW = Math.round((2 * d * d + 2 * d * gKV * 128) / (2 * gKV * 128));
    El("tk-nv").textContent = DL.commas(n); El("tk-fv").textContent = fw;
    const f = DL.frame(svg, W, H, { l: 60, r: 16, t: 26, b: 46 }), g = f.g;

    /* left — the three reading patterns */
    AT.title(g, 0, -10, "which positions each operator reads, for one output");
    const cw = 7, N = 26, mid = 20;
    const pats = [
      ["convolution", i => Math.abs(i - mid) <= (fw - 1) / 2, DC.a2],
      ["recurrence", i => i === mid || i === mid - 1, DC.violet],
      ["attention", i => i <= mid, DC.accent]
    ];
    pats.forEach((p, k) => {
      const yy = 8 + k * 44;
      AT.note(g, 0, yy, p[0], p[2], 10);
      for (let i = 0; i < N; i++) {
        g.append("rect").attr("x", i * cw).attr("y", yy + 6).attr("width", cw - 0.8).attr("height", 12)
          .attr("fill", i === mid ? DC.ink : p[2]).attr("fill-opacity", p[1](i) ? 0.85 : 0.12);
      }
      /* the convolution drawn above (and measured below) is CENTRED: one layer reaches (f−1)/2 each side,
         so connecting two positions Δ apart takes ⌈2Δ/(f−1)⌉ layers (a causal kernel would take ⌈Δ/(f−1)⌉) */
      const need = k === 0 ? Math.ceil(2 * (n - 1) / Math.max(1, fw - 1)) : k === 1 ? n - 1 : 1;
      AT.note(g, 0, yy + 32, "layers/steps to connect the two ends of an n = " + DL.commas(n) + " sequence: " +
        DL.commas(need), DC.muted, 9);
    });

    /* right — the selected comparison */
    const rx = 250, rw = f.iw - rx, gh = 250;
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const ns = []; for (let e = 6; e <= 18; e++) ns.push(Math.pow(2, e));
    let lines, ylab, title, xlab = "sequence length n", logx = true, xs, msg = "";
    if (cmp === "path") {
      title = "layers needed to connect two ends"; ylab = "layers";
      lines = [[q => Math.ceil(2 * (q - 1) / Math.max(1, fw - 1)), DC.a2], [q => q - 1, DC.violet], [() => 1, DC.accent]];
    } else if (cmp === "cost") {
      title = "MACs per layer"; ylab = "MACs";
      lines = [[q => q * fw * d * d, DC.a2], [q => q * d * d, DC.violet],
               [q => DL.attnMacs({ n: q, d: d, h: 32, g: 32, dh: d / 32, dff: 0, nmat: 0 }).attn, DC.accent]];
    } else if (cmp === "gen") {
      /* per layer, per generated token, what has to be READ besides the shared weights:
         the convolution's window (f·d), the recurrence's state (d), attention's cache
         (2·g·dₖ·n at dₖ = 128, h = d/128 query heads and g = h/4 KV heads, Llama-3's ratio) */
      title = "state or cache read per generated token, per layer, elements"; ylab = "elements";
      lines = [[() => fw * d, DC.a2], [() => d, DC.violet], [q => 2 * gKV * 128 * q, DC.accent]];
    }
    if (cmp !== "grad") {
      xs = d3.scaleLog().domain([64, 262144]).range([0, rw - 12]);
      const all = ns.flatMap(q => lines.map(l => l[0](q)));
      const y = d3.scaleLog().domain([Math.max(0.5, Math.min(...all) / 3), Math.max(...all) * 3]).range([gh, 0]).clamp(true);
      AT.title(g, rx, -10, title);
      DL.gridY(gg, y, rw - 12, 5);
      DL.axisB(gg, xs, gh, 5, xlab, v => DL.big(v));
      DL.axisL(gg, y, 5, ylab, v => DL.big(v));
      lines.forEach((l, i) => DL.curve(gg, ns.map(q => [xs(q), y(Math.max(0.5, l[0](q)))]), { stroke: l[1], w: 2.2 }));
      gg.append("line").attr("x1", xs(DL.clamp(n, 64, 262144))).attr("x2", xs(DL.clamp(n, 64, 262144)))
        .attr("y1", 0).attr("y2", gh).attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.55);
      if (cmp === "gen" && nCacheEqW >= 64 && nCacheEqW <= 262144) {
        gg.append("line").attr("x1", xs(nCacheEqW)).attr("x2", xs(nCacheEqW)).attr("y1", 0).attr("y2", gh)
          .attr("stroke", DC.bad).attr("stroke-dasharray", "4 3").attr("stroke-opacity", 0.8);
        AT.note(gg, xs(nCacheEqW) + 3, 12, "n = " + DL.commas(nCacheEqW) + ": the cache read equals the attention weights read", DC.bad, 8.5);
      }
      DL.legend(gg, OPS.map(o => ({ label: o[0], color: o[1] })), 4, gh + 32, { vertical: true, gap: 13, font: 9.5 });
      const vals = lines.map(l => l[0](n));
      const kvv = DL.kv(g, rx, gh + 78, { keyW: 236, size: 10.5, lead: 14.5 });
      OPS.forEach((o, i) => kvv(o[0], DL.big(vals[i]), o[1], i === 2));
      kvv("attention / recurrence", DL.fmtE(vals[2] / Math.max(1e-12, vals[1]), 2) + "×", DC.a2, true);
      if (cmp === "gen") kvv("cache = attention weights (one layer) at n =", DL.commas(nCacheEqW) + "   (h = " + hQ + ", g = " + gKV + ", dₖ = 128)", DC.bad);
      msg = OPS.map((o, i) => `${o[0]} <b>${DL.big(vals[i])}</b>`).join(", ");
    } else {
      AT.title(g, rx, -10, "measured ‖∂ output(n−1) / ∂ input(j)‖ against distance");
      const gc = gradAtDistance((X, sd) => stackConv(X, fw, sd)), gr = gradAtDistance(stackRec), ga = gradAtDistance(stackAttn);
      const rfConv = LM * (fw - 1) / 2;                   /* the CENTRED stack's one-sided receptive field: exactly zero beyond it */
      const all = [gc, gr, ga].flatMap(a => a.map(p => p.g)).filter(v => v > 0);
      xs = d3.scaleLinear().domain([0, NM - 1]).range([0, rw - 12]);
      const y = d3.scaleLog().domain([Math.max(1e-18, Math.min(...all) / 3), Math.max(...all) * 3]).range([gh, 0]).clamp(true);
      DL.gridY(gg, y, rw - 12, 5);
      DL.axisB(gg, xs, gh, 6, "distance in positions");
      DL.axisL(gg, y, 5, "‖∂y/∂x‖", v => DL.fmtE(v, 0));
      [[gc, DC.a2], [gr, DC.violet], [ga, DC.accent]].forEach(s2 =>
        DL.curve(gg, s2[0].map(p => [xs(p.dist), y(Math.max(p.g, 1e-18))]), { stroke: s2[1], w: 2.2 }));
      DL.legend(gg, OPS.map(o => ({ label: o[0], color: o[1] })), 4, gh + 32, { vertical: true, gap: 13, font: 9.5 });
      AT.note(gg, 4, gh + 76, "measured through " + LM + "-layer stacks by central differences, width " + DM + ", geometric mean of " + SEEDS + " draws", DC.muted, 9.5);
      AT.note(gg, 4, gh + 88, "convolution: exactly 0 beyond its receptive field of " + rfConv + " positions", DC.a2, 9);
      const at = (a, k) => (a.find(p => p.dist === k) || { g: 0 }).g;
      const kvv = DL.kv(g, rx, gh + 92, { keyW: 236, size: 10.5, lead: 14.5 });
      [[gc, "convolution", DC.a2], [gr, "recurrence", DC.violet], [ga, "attention", DC.accent]].forEach(s2 =>
        kvv(s2[1] + ": distance 1 → " + (NM - 1),
          DL.fmtE(at(s2[0], 1), 2) + " → " + DL.fmtE(at(s2[0], NM - 1), 2) +
          "   (×" + DL.fmtE(at(s2[0], NM - 1) / Math.max(1e-300, at(s2[0], 1)), 1) + ")", s2[2], s2[1] === "attention"));
      msg = `from distance 1 to ${NM - 1} the gradient norm changes by a factor of <b>${DL.fmtE(at(gr, NM - 1) / Math.max(1e-300, at(gr, 1)), 1)}</b> ` +
        `for the recurrence, is <b>exactly zero</b> for the convolution beyond its ${rfConv}-position receptive field, and changes by ` +
        `<b>${DL.fmtE(at(ga, NM - 1) / Math.max(1e-300, at(ga, 1)), 1)}</b> for attention — it does not decay at all; through a causal stack it ` +
        `<b>grows</b> towards the start of the sequence, because the number of paths from an early position to the last one grows with the ` +
        `distance (every intermediate position can read it), the same mechanism that makes the first token a natural sink (§29)`;
    }

    El("tk-readout").innerHTML =
      `n = ${DL.commas(n)}, kernel width ${fw}, d = ${DL.commas(d)} · ` +
      (cmp === "path" ? `layers needed to connect the two ends: ${msg}. Attention is <b>1</b> at every length, which is the ` +
        `row that changed the field.`
        : cmp === "cost" ? `MACs per layer: ${msg}. Attention's line bends upward past §19's crossover; the other two do not bend.`
          : cmp === "gen" ? `state or cache per generated token: ${msg}. The first two are flat in n and the third is not — ` +
            `that is the price attention pays for its constant path length. Beyond n = <b>${DL.commas(nCacheEqW)}</b> one layer's cache ` +
            `read is larger than its attention-weight read (h = ${hQ}, g = ${gKV}, dₖ = 128), and unlike the weights it is not shared across a batch (§26).`
            : `${msg}. The recurrence's gradient decays because the path is a matrix POWER; attention's does not decay because ` +
              `the path is a single weight, with no product to contract.`);
  }
  ["tk-n", "tk-f"].forEach(id => d3.select("#" + id).on("input", draw));
  ["tk-d", "tk-c"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();
