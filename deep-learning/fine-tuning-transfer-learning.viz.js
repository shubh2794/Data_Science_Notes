/* fine-tuning-transfer-learning.viz.js — the visualizations on
   deep-learning/fine-tuning-transfer-learning.html.
   Loaded after ../data.js → ../notes.js → dl-viz.js (DC palette, DL.rng).
   Each block is an IIFE that exits quietly if its container is absent.

     1  #ac-svg   stylised accuracy vs number of unfrozen blocks (small / large N)
     2  #gd-svg   2×2 strategy chooser (data size × domain similarity)
     3  #lr-svg   per-layer-group LR: discriminative factor × schedule × unfreezing
     4  #fg-svg   forgetting toy: GD on L_T + {none, weight decay, L2-SP, EWC}      */

/* ---------- page-local helpers and the stylised transfer model ---------- */
const FT = (function () {
  function axisStyle(g) {
    g.selectAll("path,line").attr("stroke", DC.line);
    g.selectAll("text").attr("fill", DC.muted).attr("font-size", 10);
    return g;
  }
  function label(svg, x, y, txt, o) {
    o = o || {};
    return svg.append("text").attr("x", x).attr("y", y)
      .attr("fill", o.fill || DC.muted).attr("font-size", o.size || 10)
      .attr("text-anchor", o.anchor || "start").attr("font-weight", o.weight || null).text(txt);
  }
  function bind(id, out, fmt) {
    const el = document.getElementById(id), o = document.getElementById(out);
    const f = () => { if (o) o.textContent = fmt ? fmt(+el.value) : el.value; };
    f(); return f;
  }
  const fmtN = v => Math.round(Math.pow(10, v)).toLocaleString("en-US");
  const fmtP = p => p >= 1e6 ? (p / 1e6).toFixed(1) + "M" : p >= 1e3 ? (p / 1e3).toFixed(0) + "k" : String(p);

  /* The stylised model printed on the page:
       acc(k) = a_max − gap(s) + gap(s)·∑_{top k} wᵢ − λ·√(P_trainable(k)/N)
       wᵢ ∝ exp(−(K − i)/τ(s)),  τ(s) = 1 + 4(1 − s)   (diminishing returns
       going down the network; a dissimilar domain needs changes deeper down)
       P_trainable = head + k equal-sized blocks.                              */
  const A_MAX = 0.95, LAMBDA = 0.004, P_TOTAL = 15.3e6, P_HEAD = 2e4;
  function curve(K, s, N) {
    const Pb = P_TOTAL / K, gap = 0.06 + 0.40 * (1 - s), tau = 1 + 4 * (1 - s);
    const w = d3.range(1, K + 1).map(i => Math.exp(-(K - i) / tau)), ws = d3.sum(w);
    const out = [];
    for (let k = 0; k <= K; k++) {
      let gain = 0;
      for (let i = K - k; i < K; i++) gain += w[i] / ws;
      const p = P_HEAD + k * Pb;
      out.push({ k: k, acc: A_MAX - gap + gap * gain - LAMBDA * Math.sqrt(p / N), p: p });
    }
    return out;
  }
  function best(c) { return c.reduce((a, b) => (b.acc > a.acc + 1e-9 ? b : a), c[0]); }
  return { axisStyle: axisStyle, label: label, bind: bind, fmtN: fmtN, fmtP: fmtP, curve: curve, best: best };
})();

/* ══════════ 1 · accuracy vs unfrozen blocks ══════════ */
(function () {
  const svg = d3.select("#ac-svg"); if (svg.empty()) return;
  const W = 680, H = 330, m = { l: 54, r: 130, t: 16, b: 110 }, K = 8;
  function draw() {
    const lN = +d3.select("#ac-N").property("value"), s = +d3.select("#ac-s").property("value");
    const sets = [
      { name: "N = 1,000", N: 1e3, col: DC.bad, w: 1.6 },
      { name: "N = 100,000", N: 1e5, col: DC.good, w: 1.6 },
      { name: "your N = " + FT.fmtN(lN), N: Math.pow(10, lN), col: DC.a2, w: 2.8 }
    ].map(d => Object.assign(d, { c: FT.curve(K, s, d.N) }));
    svg.selectAll("*").remove();
    const all = [].concat.apply([], sets.map(d => d.c.map(e => e.acc)));
    const x = d3.scaleLinear().domain([0, K]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([Math.max(0.3, d3.min(all) - 0.02), Math.min(1, d3.max(all) + 0.02)]).nice().range([H - m.b, m.t]);
    FT.axisStyle(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(K)));
    FT.axisStyle(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".0%"))));
    FT.label(svg, (m.l + W - m.r) / 2, H - m.b + 30, "k = number of top backbone blocks unfrozen  (0 = feature extraction, 8 = full fine-tuning)", { anchor: "middle" });
    svg.append("text").attr("transform", `translate(14,${(H - m.b + m.t) / 2}) rotate(-90)`).attr("text-anchor", "middle")
      .attr("fill", DC.muted).attr("font-size", 10).text("target accuracy (stylised)");
    const ln = d3.line().x(d => x(d.k)).y(d => y(d.acc));
    sets.forEach((d, i) => {
      svg.append("path").datum(d.c).attr("d", ln).attr("fill", "none").attr("stroke", d.col).attr("stroke-width", d.w);
      svg.selectAll(null).data(d.c).enter().append("circle").attr("cx", e => x(e.k)).attr("cy", e => y(e.acc)).attr("r", 2.5).attr("fill", d.col);
      const b = FT.best(d.c);
      svg.append("path").attr("d", d3.symbol().type(d3.symbolTriangle).size(70)()).attr("transform", `translate(${x(b.k)},${y(b.acc) - 10}) rotate(180)`).attr("fill", d.col);
      svg.append("rect").attr("x", W - m.r + 12).attr("y", 26 + i * 20).attr("width", 14).attr("height", 3).attr("fill", d.col);
      FT.label(svg, W - m.r + 30, 30 + i * 20, d.name, { fill: DC.ink });
      d.best = b;
    });
    /* network strip for your N's best k */
    const bk = sets[2].best.k, sy = H - 58, bw = (W - m.r - m.l) / (K + 1.4);
    FT.label(svg, m.l, sy - 8, `your N: best k = ${bk} → the network`, { fill: DC.a2 });
    for (let i = 0; i < K; i++) {
      const frozen = i < K - bk;
      svg.append("rect").attr("x", m.l + i * bw).attr("y", sy).attr("width", bw - 6).attr("height", 30).attr("rx", 5)
        .attr("fill", frozen ? DC.panel2 : "rgba(255,180,84,.18)").attr("stroke", frozen ? DC.line : DC.a2);
      FT.label(svg, m.l + i * bw + (bw - 6) / 2, sy + 13, "block " + (i + 1), { anchor: "middle", fill: frozen ? DC.muted : DC.ink, size: 9.5 });
      FT.label(svg, m.l + i * bw + (bw - 6) / 2, sy + 25, frozen ? "frozen" : "train", { anchor: "middle", fill: frozen ? DC.muted : DC.a2, size: 9 });
    }
    const hx = m.l + K * bw;
    svg.append("rect").attr("x", hx).attr("y", sy).attr("width", bw * 1.2).attr("height", 30).attr("rx", 5).attr("fill", "rgba(91,156,255,.18)").attr("stroke", DC.accent);
    FT.label(svg, hx + bw * 0.6, sy + 19, "new head", { anchor: "middle", fill: DC.accent, size: 10 });
    FT.label(svg, m.l, sy + 46, "← input (general features)", { size: 9.5 });
    FT.label(svg, hx + bw * 1.2, sy + 46, "task-specific →", { anchor: "end", size: 9.5 });
    const [a, b, c] = sets.map(d => d.best);
    d3.select("#ac-read").html(
      `s = ${s.toFixed(2)} · best k: <b style="color:${DC.bad}">N=1k → ${a.k}</b> (${(100 * a.acc).toFixed(1)}%), ` +
      `<b style="color:${DC.good}">N=100k → ${b.k}</b> (${(100 * b.acc).toFixed(1)}%), ` +
      `<b style="color:${DC.a2}">your N → ${c.k}</b> (${(100 * c.acc).toFixed(1)}%, ${FT.fmtP(c.p)} trainable parameters). ` +
      (c.k === 0 ? "Feature extraction wins: the target set is too small to pay for moving any backbone weights." :
        c.k === 8 ? "Full fine-tuning wins: enough data to adapt every layer." : "Partial fine-tuning: adapt the top, keep the general lower layers frozen."));
  }
  const u = [FT.bind("ac-N", "ac-Nv", FT.fmtN), FT.bind("ac-s", "ac-sv", v => v.toFixed(2))];
  ["#ac-N", "#ac-s"].forEach(id => d3.select(id).on("input", () => { u.forEach(f => f()); draw(); }));
  draw();
})();

/* ══════════ 2 · 2×2 strategy chooser ══════════ */
(function () {
  const svg = d3.select("#gd-svg"); if (svg.empty()) return;
  const W = 680, H = 360, P = { l: 56, r: 380, t: 18, b: 310 };
  const x = d3.scaleLinear().domain([2, 6]).range([P.l, P.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([P.b, P.t]);
  const Q = {
    ss: { t: "Small data · similar", c: DC.good, rec: "Feature extraction: freeze the backbone, train a new head. Optionally unfreeze the top block with a tiny LR." },
    sd: { t: "Small data · different", c: DC.bad, rec: "Hardest case: use earlier-layer features (drop the top), fine-tune a few blocks with strong regularisation and augmentation — or continue self-supervised pretraining on unlabelled in-domain data first." },
    ls: { t: "Large data · similar", c: DC.accent, rec: "Fine-tune the top several blocks — or all of them — with a low LR and layer-wise LRs; low overfitting risk. The closer the domain, the sooner the gain from unfreezing more saturates." },
    ld: { t: "Large data · different", c: DC.a2, rec: "Fine-tune the whole network (pretrained init still speeds convergence); with very large data, training from scratch becomes competitive." }
  };
  function draw() {
    const lN = +d3.select("#gd-N").property("value"), s = +d3.select("#gd-s").property("value"), K = +d3.select("#gd-K").property("value");
    svg.selectAll("*").remove();
    const quads = [
      { k: "sd", x0: 2, x1: 4, y0: 0, y1: 0.5 }, { k: "ss", x0: 2, x1: 4, y0: 0.5, y1: 1 },
      { k: "ld", x0: 4, x1: 6, y0: 0, y1: 0.5 }, { k: "ls", x0: 4, x1: 6, y0: 0.5, y1: 1 }
    ];
    const cur = (lN >= 4 ? "l" : "s") + (s >= 0.5 ? "s" : "d");
    quads.forEach(q => {
      const on = q.k === cur;
      svg.append("rect").attr("x", x(q.x0)).attr("y", y(q.y1)).attr("width", x(q.x1) - x(q.x0)).attr("height", y(q.y0) - y(q.y1))
        .attr("fill", Q[q.k].c).attr("fill-opacity", on ? 0.2 : 0.06).attr("stroke", on ? Q[q.k].c : DC.line).attr("stroke-width", on ? 2 : 1);
      FT.label(svg, (x(q.x0) + x(q.x1)) / 2, y(q.y1) + 18, Q[q.k].t, { anchor: "middle", fill: on ? DC.ink : DC.muted, size: 11, weight: on ? 600 : null });
    });
    FT.axisStyle(svg.append("g").attr("transform", `translate(0,${P.b})`).call(d3.axisBottom(x).ticks(4).tickFormat(v => "10" + "⁰¹²³⁴⁵⁶"[v])));
    FT.axisStyle(svg.append("g").attr("transform", `translate(${P.l},0)`).call(d3.axisLeft(y).ticks(5)));
    FT.label(svg, (P.l + P.r) / 2, P.b + 34, "labelled target examples (log scale)", { anchor: "middle" });
    svg.append("text").attr("transform", `translate(16,${(P.t + P.b) / 2}) rotate(-90)`).attr("text-anchor", "middle")
      .attr("fill", DC.muted).attr("font-size", 10).text("domain similarity to the pretraining data");
    /* hit area for clicks */
    svg.append("rect").attr("x", P.l).attr("y", P.t).attr("width", P.r - P.l).attr("height", P.b - P.t).attr("fill", "transparent")
      .style("cursor", "crosshair").on("click", function (ev) {
        const pt = d3.pointer(ev, svg.node());
        const nN = Math.max(2, Math.min(6, x.invert(pt[0]))), ns = Math.max(0, Math.min(1, y.invert(pt[1])));
        d3.select("#gd-N").property("value", nN.toFixed(1)); d3.select("#gd-s").property("value", (Math.round(ns * 20) / 20).toFixed(2));
        u.forEach(f => f()); draw();
      });
    svg.append("circle").attr("cx", x(lN)).attr("cy", y(s)).attr("r", 7).attr("fill", Q[cur].c).attr("stroke", DC.ink).attr("stroke-width", 1.5).style("pointer-events", "none");

    /* right panel: blocks, frozen/unfrozen, per-block LR */
    const c = FT.curve(K, s, Math.pow(10, lN)), bk = FT.best(c).k;
    const rx = 430, top = 22, rowH = Math.min(26, 290 / (K + 1)), bw = 120;
    FT.label(svg, rx, top - 6, `recommended: unfreeze top ${bk} of ${K}`, { fill: DC.ink, size: 11, weight: 600 });
    const headLR = 1e-3, topLR = 1e-4;
    const rows = [{ name: "new head", lr: headLR, head: true }];
    for (let i = K; i >= 1; i--) {
      const trainable = i > K - bk, dist = K - i;
      rows.push({ name: "block " + i, lr: trainable ? topLR / Math.pow(2.6, dist) : 0 });
    }
    rows.forEach((r, j) => {
      const yy = top + 8 + j * rowH;
      const col = r.head ? DC.accent : r.lr > 0 ? DC.a2 : DC.muted;
      svg.append("rect").attr("x", rx).attr("y", yy).attr("width", bw).attr("height", rowH - 4).attr("rx", 4)
        .attr("fill", r.head ? "rgba(91,156,255,.18)" : r.lr > 0 ? "rgba(255,180,84,.18)" : DC.panel2).attr("stroke", r.lr > 0 || r.head ? col : DC.line);
      FT.label(svg, rx + 8, yy + rowH / 2 + 1, r.name, { fill: r.lr > 0 || r.head ? DC.ink : DC.muted, size: 10 });
      FT.label(svg, rx + bw + 10, yy + rowH / 2 + 1, r.lr > 0 ? "LR " + r.lr.toExponential(1) : "frozen (BN eval)", { fill: col, size: 10 });
    });
    d3.select("#gd-read").html(`<b style="color:${Q[cur].c}">${Q[cur].t}</b> (N ≈ ${FT.fmtN(lN)}, similarity ${s.toFixed(2)}): ${Q[cur].rec} ` +
      `<br>The stylised model of §03 puts the optimum at <b>${bk}</b> unfrozen block${bk === 1 ? "" : "s"}; LRs shown use head 1e-3, top block 1e-4, ÷2.6 per block below.`);
  }
  const u = [FT.bind("gd-N", "gd-Nv", FT.fmtN), FT.bind("gd-s", "gd-sv", v => v.toFixed(2))];
  ["#gd-N", "#gd-s"].forEach(id => d3.select(id).on("input", () => { u.forEach(f => f()); draw(); }));
  d3.select("#gd-K").on("change", draw);
  draw();
})();

/* ══════════ 3 · per-layer LR heatmap ══════════ */
(function () {
  const svg = d3.select("#lr-svg"); if (svg.empty()) return;
  const W = 680, H = 320, m = { l: 70, r: 130, t: 14, b: 40 }, S = 24, ETA = 1e-3;
  function sched(kind, t, T) {        // multiplier in (0, 1]
    if (kind === "const") return 1;
    if (kind === "stlr") {
      const cf = 0.1, ratio = 32, cut = Math.max(1, Math.floor(T * cf));
      const p = t < cut ? t / cut : 1 - (t - cut) / (cut * (1 / cf - 1));
      return (1 + Math.max(0, p) * (ratio - 1)) / ratio;
    }
    const wu = Math.max(1, Math.floor(0.1 * T));
    if (t < wu) return (t + 1) / wu;
    return Math.max(1e-3, 0.5 * (1 + Math.cos(Math.PI * (t - wu) / Math.max(1, T - wu))));
  }
  function draw() {
    const G = +d3.select("#lr-G").property("value"), f = +d3.select("#lr-f").property("value");
    const mode = d3.select("#lr-u").property("value"), kind = d3.select("#lr-sch").property("value");
    const E = G, T = E * S;
    svg.selectAll("*").remove();
    const cw = (W - m.l - m.r) / T, rh = (H - m.t - m.b) / G;
    const color = d3.scaleSequential(d3.interpolateViridis).domain([-7, -3]);
    const peak = new Array(G).fill(0);
    for (let g = 0; g < G; g++) {               // g = 0 is the head (drawn at the top)
      const base = ETA / Math.pow(f, g);
      for (let t = 0; t < T; t++) {
        const e = Math.floor(t / S);
        let on;
        if (mode === "gradual") on = g <= e;
        else if (mode === "head") on = g === 0 || e >= 1;
        else on = true;
        let mult;
        if (mode === "all") mult = sched(kind, t, T);
        else mult = sched(kind, t % S, S);
        const lr = on ? base * mult : 0;
        if (lr > peak[g]) peak[g] = lr;
        svg.append("rect").attr("x", m.l + t * cw).attr("y", m.t + g * rh).attr("width", cw + 0.3).attr("height", rh - 1.5)
          .attr("fill", on ? color(Math.log10(lr)) : DC.panel2);
      }
      FT.label(svg, m.l - 6, m.t + g * rh + rh / 2 + 3, g === 0 ? "head" : "group " + (G - g), { anchor: "end", fill: g === 0 ? DC.accent : DC.muted, size: 10 });
    }
    for (let e = 1; e < E; e++) svg.append("line").attr("x1", m.l + e * S * cw).attr("x2", m.l + e * S * cw).attr("y1", m.t).attr("y2", H - m.b)
      .attr("stroke", DC.bg).attr("stroke-width", 1.5);
    for (let e = 0; e < E; e++) FT.label(svg, m.l + (e + 0.5) * S * cw, H - m.b + 14, "epoch " + (e + 1), { anchor: "middle", size: 9.5 });
    FT.label(svg, (m.l + W - m.r) / 2, H - 6, "training steps →   (" + S + " steps per epoch)", { anchor: "middle" });
    /* peak bars */
    const bx = W - m.r + 10, bwMax = 70;
    const bs = d3.scaleLinear().domain([-7, -3]).range([0, bwMax]);
    peak.forEach((p, g) => {
      const v = p > 0 ? Math.log10(p) : -7;
      svg.append("rect").attr("x", bx).attr("y", m.t + g * rh + 2).attr("width", Math.max(1, bs(Math.max(-7, v)))).attr("height", rh - 5)
        .attr("fill", p > 0 ? color(v) : DC.panel2);
      FT.label(svg, bx + bwMax + 4, m.t + g * rh + rh / 2 + 3, p > 0 ? p.toExponential(1) : "—", { size: 9 });
    });
    FT.label(svg, bx, m.t - 2 + 0, "", {});
    const bottom = ETA / Math.pow(f, G - 1);
    const frozenFrac = (() => { let fr = 0; for (let g = 0; g < G; g++) for (let e = 0; e < E; e++) { const on = mode === "gradual" ? g <= e : mode === "head" ? (g === 0 || e >= 1) : true; if (!on) fr++; } return fr / (G * E); })();
    d3.select("#lr-read").html(`head LR <b>${ETA.toExponential(0)}</b>, bottom group <b>${bottom.toExponential(2)}</b> — ratio <b>${Math.pow(f, G - 1).toFixed(1)}×</b> (ξ<sup>${G - 1}</sup>). ` +
      `${(100 * frozenFrac).toFixed(0)}% of group-epochs are frozen. ` +
      (mode === "gradual" ? "Gradual unfreezing: each epoch brings one more group into training, each stage with its own " + (kind === "stlr" ? "slanted-triangular" : kind === "cos" ? "warmup + cosine" : "constant") + " cycle." :
        mode === "head" ? "Head-first: one epoch on the head alone, then everything, restarting the schedule each epoch." : "All at once: one schedule over the whole run — the random head's gradients reach every layer from step one."));
  }
  const u = [FT.bind("lr-G", "lr-Gv"), FT.bind("lr-f", "lr-fv", v => v.toFixed(1))];
  ["#lr-G", "#lr-f"].forEach(id => d3.select(id).on("input", () => { u.forEach(f => f()); draw(); }));
  ["#lr-u", "#lr-sch"].forEach(id => d3.select(id).on("change", draw));
  draw();
})();

/* ══════════ 4 · forgetting toy ══════════ */
(function () {
  const svg = d3.select("#fg-svg"); if (svg.empty()) return;
  const W = 680, H = 360, pl = 20, pw = 410, pt = 14, ph = 330;
  /* source: narrow valley at θS, flat along u, steep along n */
  const ang = -20 * Math.PI / 180, u = [Math.cos(ang), Math.sin(ang)], nv = [-Math.sin(ang), Math.cos(ang)];
  const kSteep = 12, kFlat = 0.3;
  const A = [[kSteep * nv[0] * nv[0] + kFlat * u[0] * u[0], kSteep * nv[0] * nv[1] + kFlat * u[0] * u[1]],
             [kSteep * nv[1] * nv[0] + kFlat * u[1] * u[0], kSteep * nv[1] * nv[1] + kFlat * u[1] * u[1]]];
  const meanEig = (kSteep + kFlat) / 2;
  const F = A.map(r => r.map(v => v / meanEig));          // Fisher ≈ source curvature, trace-matched to the identity
  const thS = [-1.4, -0.6];
  const thT = [thS[0] + 2.6 * u[0] + 0.9 * nv[0], thS[1] + 2.6 * u[1] + 0.9 * nv[1]];
  const LS = th => { const d = [th[0] - thS[0], th[1] - thS[1]]; return 0.5 * (d[0] * (A[0][0] * d[0] + A[0][1] * d[1]) + d[1] * (A[1][0] * d[0] + A[1][1] * d[1])); };
  const LT = th => 0.5 * ((th[0] - thT[0]) ** 2 + (th[1] - thT[1]) ** 2);
  const METHODS = [
    { key: "none", name: "no penalty", col: DC.bad },
    { key: "wd", name: "weight decay → 0", col: DC.muted },
    { key: "l2sp", name: "L2-SP → θ⁰", col: DC.accent },
    { key: "ewc", name: "EWC (Fisher-weighted)", col: DC.good }
  ];
  function grad(th, key, a) {
    const g = [th[0] - thT[0], th[1] - thT[1]];
    if (key === "wd") { g[0] += a * th[0]; g[1] += a * th[1]; }
    if (key === "l2sp") { g[0] += a * (th[0] - thS[0]); g[1] += a * (th[1] - thS[1]); }
    if (key === "ewc") { const d = [th[0] - thS[0], th[1] - thS[1]]; g[0] += a * (F[0][0] * d[0] + F[0][1] * d[1]); g[1] += a * (F[1][0] * d[0] + F[1][1] * d[1]); }
    return g;
  }
  const xs = d3.scaleLinear().domain([-2.8, 2.6]).range([pl, pl + pw]);
  const kpx = pw / 5.4;
  const ys = d3.scaleLinear().domain([-2.2, -2.2 + ph / kpx]).range([pt + ph, pt]);
  function contourSet(fn, thresholds, col, nx) {
    const ny = Math.round(nx * ph / pw), vals = new Array(nx * ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      vals[j * nx + i] = fn([xs.invert(pl + (i + 0.5) * pw / nx), ys.invert(pt + (j + 0.5) * ph / ny)]);
    }
    const cs = d3.contours().size([nx, ny]).thresholds(thresholds)(vals);
    const proj = d3.geoPath(d3.geoTransform({ point: function (px, py) { this.stream.point(pl + px * pw / nx, pt + py * ph / ny); } }));
    svg.append("g").selectAll("path").data(cs).enter().append("path").attr("d", proj)
      .attr("fill", "none").attr("stroke", col).attr("stroke-opacity", 0.55).attr("stroke-width", 1);
  }
  function draw() {
    const a = +d3.select("#fg-a").property("value"), T = +d3.select("#fg-T").property("value"), lr = +d3.select("#fg-lr").property("value");
    svg.selectAll("*").remove();
    svg.append("rect").attr("x", pl).attr("y", pt).attr("width", pw).attr("height", ph).attr("fill", DC.panel).attr("stroke", DC.line);
    contourSet(LS, [0.05, 0.2, 0.5, 1, 2, 4, 8], DC.accent, 90);
    contourSet(LT, [0.1, 0.4, 0.9, 1.6, 2.5, 3.6], DC.a2, 90);
    /* hide the contour segments d3.contours closes along the grid border */
    svg.append("rect").attr("x", pl).attr("y", pt).attr("width", pw).attr("height", ph).attr("fill", "none").attr("stroke", DC.panel).attr("stroke-width", 4);
    svg.append("rect").attr("x", pl - 2).attr("y", pt - 2).attr("width", pw + 4).attr("height", ph + 4).attr("fill", "none").attr("stroke", DC.line);
    /* origin marker */
    svg.append("path").attr("d", `M${xs(0) - 5},${ys(0)}H${xs(0) + 5}M${xs(0)},${ys(0) - 5}V${ys(0) + 5}`).attr("stroke", DC.muted);
    FT.label(svg, xs(0) + 6, ys(0) + 12, "0", { size: 9 });
    const res = METHODS.map(M => {
      let th = thS.slice(); const path = [th.slice()];
      for (let t = 0; t < T; t++) { const g = grad(th, M.key, a); th = [th[0] - lr * g[0], th[1] - lr * g[1]]; path.push(th.slice()); }
      svg.append("path").datum(path).attr("d", d3.line().x(p => xs(p[0])).y(p => ys(p[1])))
        .attr("fill", "none").attr("stroke", M.col).attr("stroke-width", 2.2);
      svg.append("circle").attr("cx", xs(th[0])).attr("cy", ys(th[1])).attr("r", 4.5).attr("fill", M.col).attr("stroke", DC.bg);
      return { M: M, th: th, ls: LS(th), lt: LT(th) };
    });
    svg.append("circle").attr("cx", xs(thS[0])).attr("cy", ys(thS[1])).attr("r", 5).attr("fill", DC.accent).attr("stroke", DC.ink);
    FT.label(svg, xs(thS[0]) - 8, ys(thS[1]) - 8, "θ*_S (pretrained)", { anchor: "end", fill: DC.accent, size: 10 });
    svg.append("circle").attr("cx", xs(thT[0])).attr("cy", ys(thT[1])).attr("r", 5).attr("fill", DC.a2).attr("stroke", DC.ink);
    FT.label(svg, xs(thT[0]) + 8, ys(thT[1]) - 8, "θ*_T", { fill: DC.a2, size: 10 });
    /* right-hand table */
    const tx = pl + pw + 18;
    FT.label(svg, tx, 26, "endpoint after " + T + " steps", { fill: DC.ink, size: 11, weight: 600 });
    FT.label(svg, tx + 128, 46, "L_source", { anchor: "end", size: 9.5 });
    FT.label(svg, tx + 190, 46, "L_target", { anchor: "end", size: 9.5 });
    res.forEach((r, i) => {
      const yy = 70 + i * 34;
      svg.append("rect").attr("x", tx).attr("y", yy - 8).attr("width", 12).attr("height", 3).attr("fill", r.M.col);
      FT.label(svg, tx + 16, yy - 4, r.M.name, { fill: DC.ink, size: 10 });
      FT.label(svg, tx + 128, yy + 11, r.ls.toFixed(2), { anchor: "end", fill: DC.accent, size: 10.5 });
      FT.label(svg, tx + 190, yy + 11, r.lt.toFixed(2), { anchor: "end", fill: DC.a2, size: 10.5 });
    });
    FT.label(svg, tx, 225, "blue contours: source loss", { fill: DC.accent, size: 9.5 });
    FT.label(svg, tx, 240, "(narrow valley: flat along one", { size: 9.5 });
    FT.label(svg, tx, 254, " direction, steep across it)", { size: 9.5 });
    FT.label(svg, tx, 272, "orange contours: target loss", { fill: DC.a2, size: 9.5 });
    FT.label(svg, tx, 290, "EWC uses F = source curvature,", { size: 9.5 });
    FT.label(svg, tx, 304, "trace-matched to L2-SP's identity", { size: 9.5 });
    const byKey = {}; res.forEach(r => byKey[r.M.key] = r);
    const total = r => r.ls + r.lt;
    d3.select("#fg-read").html(
      `sum of both losses at the endpoint: ` + res.map(r => `<b style="color:${r.M.col}">${r.M.name} ${total(r).toFixed(2)}</b>`).join(" · ") + ". " +
      (a === 0 ? "With zero penalty all four paths coincide — pure fine-tuning." :
        `EWC keeps the source loss at ${byKey.ewc.ls.toFixed(2)} while L2-SP keeps it at ${byKey.l2sp.ls.toFixed(2)}, at target losses ${byKey.ewc.lt.toFixed(2)} vs ${byKey.l2sp.lt.toFixed(2)}: moving along the flat direction is almost free for the source task.`));
  }
  const us = [FT.bind("fg-a", "fg-av", v => v.toFixed(2)), FT.bind("fg-T", "fg-Tv"), FT.bind("fg-lr", "fg-lrv", v => v.toFixed(3))];
  ["#fg-a", "#fg-T", "#fg-lr"].forEach(id => d3.select(id).on("input", () => { us.forEach(f => f()); draw(); }));
  draw();
})();
