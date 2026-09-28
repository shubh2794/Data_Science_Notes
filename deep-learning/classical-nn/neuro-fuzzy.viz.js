/* neuro-fuzzy.viz.js — figures for deep-learning/classical-nn/neuro-fuzzy.html (part 6 of 10).
   Loaded after ../../data.js and ../../notes.js, so the palette C and d3 are global.
   Each figure is an IIFE that returns quietly when its <svg> is absent.

     1  #mf-svg     membership-function explorer (tri / trap / Gaussian / bell, probe, α-cuts)
     2  #tn-svg     t-norm / s-norm pairs on two fuzzy sets + the operator surface
     3  #mm-svg     two-rule Mamdani inference with all five defuzzifiers
     4  #tsk-svg    four-rule first-order TSK surface, slice, local models, normalized weights
     5  #fam-svg    fuzzy associative memory: encode one pair, recall both ways
     6  #an-svg     ANFIS five-layer network with live node values
     7  #af-svg     ANFIS fitting a 1-D function: hybrid vs gradient-only vs LSE-only
     8  #ex-svg     rule explosion: parameter counts vs number of inputs                     */

/* ══════════ page-local helpers ══════════ */
const NF = (function () {
  const $ = id => document.getElementById(id);
  const val = id => +$(id).value;
  /* wire sliders/selects: update the <output id="…-o"> and call cb */
  function on(ids, cb) {
    ids.forEach(id => {
      const el = $(id); if (!el) return;
      const out = $(id + "-o");
      const h = () => { if (out) out.textContent = el.value; cb(); };
      el.addEventListener("input", h); el.addEventListener("change", h);
    });
  }
  const trap = (x, a, b, c, d) => {
    if (x < a || x > d) return 0;
    if (x >= b && x <= c) return 1;
    if (x < b) return b === a ? 1 : (x - a) / (b - a);
    return d === c ? 1 : (d - x) / (d - c);
  };
  const tri = (x, a, b, c) => trap(x, a, b, b, c);
  const gauss = (x, c, s) => Math.exp(-(x - c) * (x - c) / (2 * s * s));
  const bell = (x, a, b, c) => 1 / (1 + Math.pow(Math.abs((x - c) / a), 2 * b));
  const f2 = v => (+v).toFixed(2), f3 = v => (+v).toFixed(3);
  const rng = seed => { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; };
  const axisB = (g, sc, y, n) => g.append("g").attr("class", "axis").attr("transform", `translate(0,${y})`).call(d3.axisBottom(sc).ticks(n || 5));
  const axisL = (g, sc, x, n, fmt) => g.append("g").attr("class", "axis").attr("transform", `translate(${x},0)`).call(d3.axisLeft(sc).ticks(n || 4, fmt));
  const txt = (g, x, y, s, col, size, anchor) => g.append("text").attr("x", x).attr("y", y).attr("fill", col || C.muted)
    .attr("font-size", size || 11).attr("text-anchor", anchor || "start").text(s);
  const RULE = ["#5b9cff", "#ffb454", "#4ade80", "#c084fc", "#22d3ee", "#f87171", "#facc15", "#f472b6", "#a3e635"];
  return { $, val, on, trap, tri, gauss, bell, f2, f3, rng, axisB, axisL, txt, RULE };
})();

/* ══════════ 1. membership-function explorer ══════════ */
(function () {
  const svg = d3.select("#mf-svg"); if (svg.empty()) return;
  const x = d3.scaleLinear([0, 10], [90, 740]), y = d3.scaleLinear([0, 1.05], [220, 20]);
  NF.axisB(svg, x, 220, 10); NF.axisL(svg, y, 90, 5);
  NF.txt(svg, 96, 16, "μ(x)", C.muted, 11);
  const fam = [
    { k: "triangular", col: "#5b9cff" }, { k: "trapezoidal", col: "#ffb454" },
    { k: "Gaussian", col: "#4ade80" }, { k: "gen. bell", col: "#c084fc" }];
  const aLine = svg.append("line").attr("stroke", C.muted).attr("stroke-dasharray", "4 3");
  const paths = fam.map(f => svg.append("path").attr("fill", "none").attr("stroke", f.col).attr("stroke-width", 2));
  const probe = svg.append("line").attr("stroke", C.ink).attr("stroke-opacity", .5).attr("y1", 20).attr("y2", 220);
  const dots = fam.map(f => svg.append("circle").attr("r", 4.5).attr("fill", f.col).attr("stroke", "#0b0e14"));
  const bars = fam.map((f, i) => {
    NF.txt(svg, 84, 250 + i * 18, f.k, f.col, 10.5, "end");
    return svg.append("rect").attr("y", 241 + i * 18).attr("height", 11).attr("rx", 3).attr("fill", f.col).attr("fill-opacity", .75);
  });
  NF.txt(svg, 740, 318, "α-cut intervals  {x : μ(x) ≥ α}", C.muted, 10.5, "end");
  const legend = svg.append("g");
  fam.forEach((f, i) => { legend.append("rect").attr("x", 560).attr("y", 24 + i * 16).attr("width", 12).attr("height", 3).attr("fill", f.col); NF.txt(legend, 578, 29 + i * 16, f.k, C.muted, 11); });

  function draw() {
    const c = NF.val("mf-c"), w = NF.val("mf-w"), b = NF.val("mf-b"), px = NF.val("mf-x"), al = NF.val("mf-a");
    const sg = w / 2, ba = w / 2;
    const F = [
      t => NF.tri(t, c - w, c, c + w),
      t => NF.trap(t, c - w, c - w / 2, c + w / 2, c + w),
      t => NF.gauss(t, c, sg),
      t => NF.bell(t, ba, b, c)];
    const cuts = [
      [c - w * (1 - al), c + w * (1 - al)],
      [c - w + al * w / 2, c + w - al * w / 2],
      [c - sg * Math.sqrt(-2 * Math.log(al)), c + sg * Math.sqrt(-2 * Math.log(al))],
      [c - ba * Math.pow(1 / al - 1, 1 / (2 * b)), c + ba * Math.pow(1 / al - 1, 1 / (2 * b))]];
    const xs = d3.range(0, 10.0001, 0.02);
    paths.forEach((p, i) => p.attr("d", d3.line().x(t => x(t)).y(t => y(F[i](t)))(xs)));
    probe.attr("x1", x(px)).attr("x2", x(px));
    dots.forEach((d, i) => d.attr("cx", x(px)).attr("cy", y(F[i](px))));
    aLine.attr("x1", 90).attr("x2", 740).attr("y1", y(al)).attr("y2", y(al));
    bars.forEach((r, i) => {
      const lo = Math.max(0, cuts[i][0]), hi = Math.min(10, cuts[i][1]);
      r.attr("x", x(lo)).attr("width", Math.max(1, x(hi) - x(lo)));
    });
    NF.$("mf-readout").innerHTML =
      `μ at x = ${px.toFixed(2)}: ` + fam.map((f, i) => `${f.k} <b>${NF.f3(F[i](px))}</b>`).join(" · ") +
      `<br>α = ${al.toFixed(2)} cut widths: ` + fam.map((f, i) => `${f.k} ${NF.f2(cuts[i][1] - cuts[i][0])}`).join(" · ") +
      `<br>parameters: tri [${NF.f2(c - w)}, ${NF.f2(c)}, ${NF.f2(c + w)}] · trap [${NF.f2(c - w)}, ${NF.f2(c - w / 2)}, ${NF.f2(c + w / 2)}, ${NF.f2(c + w)}] · Gaussian σ = w/2 = ${NF.f2(sg)} · bell a = w/2 = ${NF.f2(ba)}, b = ${b}`;
  }
  NF.on(["mf-c", "mf-w", "mf-b", "mf-x", "mf-a"], draw); draw();
})();

/* ══════════ 2. t-norms and s-norms ══════════ */
(function () {
  const svg = d3.select("#tn-svg"); if (svg.empty()) return;
  const OPS = {
    min: { T: (a, b) => Math.min(a, b), S: (a, b) => Math.max(a, b), n: ["min", "max"] },
    prod: { T: (a, b) => a * b, S: (a, b) => a + b - a * b, n: ["product", "prob. sum"] },
    luk: { T: (a, b) => Math.max(0, a + b - 1), S: (a, b) => Math.min(1, a + b), n: ["Łukasiewicz", "bounded sum"] },
    drastic: { T: (a, b) => (a === 1 ? b : b === 1 ? a : 0), S: (a, b) => (a === 0 ? b : b === 0 ? a : 1), n: ["drastic", "drastic"] }
  };
  const x = d3.scaleLinear([0, 10], [40, 470]), y = d3.scaleLinear([0, 1.05], [260, 20]);
  NF.axisB(svg, x, 260, 10); NF.axisL(svg, y, 40, 5);
  const areaT = svg.append("path").attr("fill", C.A).attr("fill-opacity", .35);
  const lineS = svg.append("path").attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2.2);
  const lA = svg.append("path").attr("fill", "none").attr("stroke", C.ink).attr("stroke-dasharray", "5 3").attr("stroke-opacity", .8);
  const lB = svg.append("path").attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "2 3");
  const probe = svg.append("line").attr("y1", 20).attr("y2", 260).attr("stroke", C.ink).attr("stroke-opacity", .45);
  const lg = svg.append("g");
  [["A", C.ink, "5 3"], ["B", C.muted, "2 3"], ["A ∧ B (t-norm)", C.A, null], ["A ∨ B (s-norm)", C.B, null]].forEach((d, i) => {
    lg.append("line").attr("x1", 50 + i * 104).attr("x2", 66 + i * 104).attr("y1", 290).attr("y2", 290).attr("stroke", d[1]).attr("stroke-width", d[2] ? 1.5 : 4).attr("stroke-dasharray", d[2]);
    NF.txt(lg, 70 + i * 104, 294, d[0], C.muted, 10.5);
  });
  // heatmap
  const hx = d3.scaleLinear([0, 1], [530, 730]), hy = d3.scaleLinear([0, 1], [230, 30]);
  const n = 40, cells = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cells.push({ a: (j + .5) / n, b: (i + .5) / n });
  const cellR = svg.append("g").selectAll("rect").data(cells).join("rect")
    .attr("x", d => hx(d.a - .5 / n)).attr("y", d => hy(d.b + .5 / n)).attr("width", 200 / n + .4).attr("height", 200 / n + .4);
  NF.axisB(svg, hx, 230, 5); NF.axisL(svg, hy, 530, 5);
  NF.txt(svg, 630, 262, "a = μ_A(x)", C.muted, 11, "middle");
  svg.append("text").attr("transform", "translate(500,130) rotate(-90)").attr("fill", C.muted).attr("font-size", 11).attr("text-anchor", "middle").text("b = μ_B(x)");
  const surfTitle = NF.txt(svg, 630, 20, "", C.ink, 11.5, "middle");
  const pt = svg.append("circle").attr("r", 5.5).attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2);
  const col = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

  function draw() {
    const op = OPS[NF.$("tn-op").value], sep = NF.val("tn-sep"), px = NF.val("tn-x"), which = NF.$("tn-surf").value;
    const A = t => NF.gauss(t, 5 - sep / 2, 1.3), B = t => NF.gauss(t, 5 + sep / 2, 1.3);
    const xs = d3.range(0, 10.0001, 0.02);
    lA.attr("d", d3.line().x(t => x(t)).y(t => y(A(t)))(xs));
    lB.attr("d", d3.line().x(t => x(t)).y(t => y(B(t)))(xs));
    areaT.attr("d", d3.area().x(t => x(t)).y0(y(0)).y1(t => y(op.T(A(t), B(t))))(xs));
    lineS.attr("d", d3.line().x(t => x(t)).y(t => y(op.S(A(t), B(t))))(xs));
    probe.attr("x1", x(px)).attr("x2", x(px));
    const F = which === "t" ? op.T : op.S;
    cellR.attr("fill", d => col(F(d.a, d.b)));
    surfTitle.text((which === "t" ? "t-norm: " + op.n[0] : "s-norm: " + op.n[1]) + " over (a, b)");
    const a = A(px), b = B(px);
    pt.attr("cx", hx(a)).attr("cy", hy(b));
    const all = Object.values(OPS);
    NF.$("tn-readout").innerHTML = `at x = ${px.toFixed(2)}: a = ${NF.f3(a)}, b = ${NF.f3(b)} → ${op.n[0]} <b>${NF.f3(op.T(a, b))}</b>, ${op.n[1]} <b>${NF.f3(op.S(a, b))}</b>` +
      `<br>all t-norms here: ` + all.map(o => `${o.n[0]} ${NF.f3(o.T(a, b))}`).join(" ≥ ") +
      `<br>all s-norms here: ` + all.map(o => `${o.n[1]} ${NF.f3(o.S(a, b))}`).join(" ≤ ");
  }
  NF.on(["tn-op", "tn-sep", "tn-x", "tn-surf"], draw); draw();
})();

/* ══════════ 3. two-rule Mamdani inference ══════════ */
(function () {
  const svg = d3.select("#mm-svg"); if (svg.empty()) return;
  const MF = {
    cool: t => NF.trap(t, -1, 0, 12, 26), hot: t => NF.trap(t, 14, 28, 40, 41),
    dry: h => NF.trap(h, -1, 0, 30, 70), humid: h => NF.trap(h, 30, 70, 100, 101),
    slow: z => NF.trap(z, -1, 0, 20, 50), fast: z => NF.trap(z, 50, 80, 100, 101)
  };
  const cols = [[40, 240], [285, 485], [535, 740]];
  const rows = [[30, 110], [155, 235]];
  const sx = [d3.scaleLinear([0, 40], cols[0]), d3.scaleLinear([0, 100], cols[1]), d3.scaleLinear([0, 100], cols[2])];
  const RU = [
    { title: "R1:  temperature is cool  AND  humidity is dry  →  fan is slow", a: "cool", b: "dry", out: "slow", col: C.A },
    { title: "R2:  temperature is hot  OR  humidity is humid  →  fan is fast", a: "hot", b: "humid", out: "fast", col: C.B }];
  const cell = RU.map((r, ri) => {
    const [y0, y1] = rows[ri], sy = d3.scaleLinear([0, 1], [y1, y0 + 6]);
    NF.txt(svg, 40, y0 - 6, r.title, r.col, 11.5);
    const o = { sy };
    [r.a, r.b, r.out].forEach((name, ci) => {
      svg.append("rect").attr("x", cols[ci][0]).attr("y", y0).attr("width", cols[ci][1] - cols[ci][0]).attr("height", y1 - y0).attr("fill", "none").attr("stroke", C.line);
      NF.axisB(svg, sx[ci], y1, 4);
      NF.txt(svg, cols[ci][1] - 4, y0 + 14, name, C.muted, 10.5, "end");
    });
    const dom = [[0, 40], [0, 100], [0, 100]];
    o.curves = [r.a, r.b, r.out].map((name, ci) => {
      const xs = d3.range(dom[ci][0], dom[ci][1] + 1e-9, (dom[ci][1] - dom[ci][0]) / 200);
      return svg.append("path").attr("d", d3.line().x(t => sx[ci](t)).y(t => sy(MF[name](t)))(xs))
        .attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 1.3).attr("stroke-dasharray", ci === 2 ? "4 3" : null);
    });
    o.fillA = svg.append("path").attr("fill", r.col).attr("fill-opacity", .3);
    o.fillB = svg.append("path").attr("fill", r.col).attr("fill-opacity", .3);
    o.impl = svg.append("path").attr("fill", r.col).attr("fill-opacity", .55).attr("stroke", r.col);
    o.vA = svg.append("line").attr("y1", y0).attr("y2", y1).attr("stroke", C.ink).attr("stroke-opacity", .6);
    o.vB = svg.append("line").attr("y1", y0).attr("y2", y1).attr("stroke", C.ink).attr("stroke-opacity", .6);
    o.hA = svg.append("line").attr("stroke", r.col).attr("stroke-dasharray", "3 2");
    o.hB = svg.append("line").attr("stroke", r.col).attr("stroke-dasharray", "3 2");
    o.tA = NF.txt(svg, 0, 0, "", C.ink, 10.5); o.tB = NF.txt(svg, 0, 0, "", C.ink, 10.5);
    o.wt = NF.txt(svg, cols[2][0] + 4, y0 + 14, "", r.col, 11);
    o.op = NF.txt(svg, (cols[1][0] + cols[0][1]) / 2 + 3, (y0 + y1) / 2 + 4, ri === 0 ? "AND" : "OR", C.muted, 10, "middle");
    return o;
  });
  // aggregate row
  const ax = d3.scaleLinear([0, 100], [40, 740]), ay = d3.scaleLinear([0, 1], [380, 290]);
  NF.txt(svg, 40, 280, "aggregated output  μ_out(z) = max over rules  →  defuzzify to one fan speed", C.ink, 11.5);
  svg.append("rect").attr("x", 40).attr("y", 286).attr("width", 700).attr("height", 94).attr("fill", "none").attr("stroke", C.line);
  NF.axisB(svg, ax, 380, 10); NF.txt(svg, 740, 412, "fan speed z (%)", C.muted, 10.5, "end");
  const agg = svg.append("path").attr("fill", C.good).attr("fill-opacity", .35).attr("stroke", C.good);
  const DEF = [["cen", "centroid"], ["bis", "bisector"], ["mom", "MOM"], ["som", "SOM"], ["lom", "LOM"]];
  const marks = DEF.map((d, i) => ({
    l: svg.append("line").attr("y1", 290).attr("y2", 380),
    t: NF.txt(svg, 0, 0, d[1], C.ink, 10, "middle")
  }));

  function draw() {
    const T = NF.val("mm-t"), H = NF.val("mm-h"), imp = NF.$("mm-imp").value, sel = NF.$("mm-def").value;
    const I = imp === "min" ? (w, m) => Math.min(w, m) : (w, m) => w * m;
    const ws = [];
    RU.forEach((r, ri) => {
      const o = cell[ri], mA = MF[r.a](T), mB = MF[r.b](H);
      const w = ri === 0 ? Math.min(mA, mB) : Math.max(mA, mB); ws.push(w);
      const xsA = d3.range(0, 40.0001, .2), xsB = d3.range(0, 100.0001, .5);
      o.fillA.attr("d", d3.area().x(t => sx[0](t)).y0(o.sy(0)).y1(t => o.sy(Math.min(MF[r.a](t), mA)))(xsA));
      o.fillB.attr("d", d3.area().x(t => sx[1](t)).y0(o.sy(0)).y1(t => o.sy(Math.min(MF[r.b](t), mB)))(xsB));
      o.impl.attr("d", d3.area().x(z => sx[2](z)).y0(o.sy(0)).y1(z => o.sy(I(w, MF[r.out](z))))(xsB));
      o.vA.attr("x1", sx[0](T)).attr("x2", sx[0](T)); o.vB.attr("x1", sx[1](H)).attr("x2", sx[1](H));
      o.hA.attr("x1", cols[0][0]).attr("x2", cols[0][1]).attr("y1", o.sy(mA)).attr("y2", o.sy(mA));
      o.hB.attr("x1", cols[1][0]).attr("x2", cols[1][1]).attr("y1", o.sy(mB)).attr("y2", o.sy(mB));
      o.tA.attr("x", sx[0](T) + 4).attr("y", o.sy(mA) - 4).text(NF.f2(mA));
      o.tB.attr("x", sx[1](H) + 4).attr("y", o.sy(mB) - 4).text(NF.f2(mB));
      o.wt.text("w = " + NF.f3(w));
    });
    const N = 1001, zs = d3.range(N).map(i => 100 * i / (N - 1));
    const mu = zs.map(z => Math.max(I(ws[0], MF.slow(z)), I(ws[1], MF.fast(z))));
    agg.attr("d", d3.area().x((z, i) => ax(z)).y0(ay(0)).y1((z, i) => ay(mu[i]))(zs));
    const area = d3.sum(mu);
    const D = {};
    if (area < 1e-9) { DEF.forEach(d => D[d[0]] = NaN); }
    else {
      D.cen = d3.sum(zs, (z, i) => z * mu[i]) / area;
      let acc = 0; D.bis = zs[N - 1];
      for (let i = 0; i < N; i++) { acc += mu[i]; if (acc >= area / 2) { D.bis = zs[i]; break; } }
      const mx = d3.max(mu); const idx = []; mu.forEach((m, i) => { if (m >= mx - 1e-9) idx.push(i); });
      D.som = zs[idx[0]]; D.lom = zs[idx[idx.length - 1]]; D.mom = (D.som + D.lom) / 2;
    }
    const order = DEF.map((d, i) => i).filter(i => isFinite(D[DEF[i][0]])).sort((a, b) => D[DEF[a][0]] - D[DEF[b][0]]);
    const rowOf = {}; let lastX = [-1e9, -1e9, -1e9, -1e9, -1e9];
    order.forEach(i => { const px = ax(D[DEF[i][0]]); let r = 0; while (r < 4 && px - lastX[r] < 62) r++; rowOf[i] = r; lastX[r] = px; });
    DEF.forEach((d, i) => {
      const v = D[d[0]], on = d[0] === sel, m = marks[i];
      const ok = isFinite(v);
      m.l.attr("x1", ok ? ax(v) : -99).attr("x2", ok ? ax(v) : -99).attr("stroke", on ? C.bad : C.ink)
        .attr("stroke-width", on ? 3 : 1).attr("stroke-opacity", on ? 1 : .45).attr("stroke-dasharray", on ? null : "3 3");
      m.t.attr("x", ok ? ax(v) : -99).attr("y", 299 + (rowOf[i] || 0) * 11).attr("fill", on ? C.bad : C.muted)
        .attr("font-weight", on ? 700 : 400).text(on ? d[1] + " " + v.toFixed(1) : d[1]);
    });
    NF.$("mm-readout").innerHTML = `firing strengths: w₁ = <b>${NF.f3(ws[0])}</b> (min of cool ${NF.f3(MF.cool(T))}, dry ${NF.f3(MF.dry(H))}) · w₂ = <b>${NF.f3(ws[1])}</b> (max of hot ${NF.f3(MF.hot(T))}, humid ${NF.f3(MF.humid(H))})` +
      `<br>crisp fan speed — centroid <b>${NF.f2(D.cen)}</b> · bisector ${NF.f2(D.bis)} · MOM ${NF.f2(D.mom)} · SOM ${NF.f2(D.som)} · LOM ${NF.f2(D.lom)} %` +
      `<br>zeroth-order Sugeno shortcut (slow ≡ 25, fast ≡ 80): ${(ws[0] + ws[1]) > 0 ? NF.f2((25 * ws[0] + 80 * ws[1]) / (ws[0] + ws[1])) : "—"} %`;
  }
  NF.on(["mm-t", "mm-h", "mm-imp", "mm-def"], draw); draw();
})();

/* ══════════ shared rule set for figures 4 and 6 (four first-order TSK rules) ══════════ */
const TSK4 = {
  centres: [[0, 0], [0, 10], [10, 0], [10, 10]],             // R1 LL, R2 LH, R3 HL, R4 HH
  lin: [[0.6, 0.2, 1], [-0.4, -0.2, 9], [0.1, 0.5, 2], [-0.5, -0.3, 14]],  // f = p x + q y + r
  f(r, x, y, ord) {
    const L = this.lin[r];
    if (+ord === 0) { const c = this.centres[r]; return L[0] * c[0] + L[1] * c[1] + L[2]; }
    return L[0] * x + L[1] * y + L[2];
  },
  eval(x, y, s, ord) {
    const w = this.centres.map(c => NF.gauss(x, c[0], s) * NF.gauss(y, c[1], s));
    const S = d3.sum(w) || 1e-300, wb = w.map(v => v / S);
    const fs = [0, 1, 2, 3].map(r => this.f(r, x, y, ord));
    return { w, wb, fs, out: d3.sum(wb, (v, r) => v * fs[r]) };
  }
};

/* ══════════ 4. TSK surface ══════════ */
(function () {
  const svg = d3.select("#tsk-svg"); if (svg.empty()) return;
  const n = 50, cs = 300 / n, X0 = 40, Y0 = 30;
  const px = d3.scaleLinear([0, 10], [X0, X0 + 300]), py = d3.scaleLinear([0, 10], [Y0 + 300, Y0]);
  const col = d3.scaleSequential(d3.interpolateViridis).domain([0, 12]);
  const cells = []; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cells.push({ i, j, x: (j + .5) / n * 10, y: 10 - (i + .5) / n * 10 });
  const rects = svg.append("g").selectAll("rect").data(cells).join("rect")
    .attr("x", d => X0 + d.j * cs).attr("y", d => Y0 + d.i * cs).attr("width", cs + .3).attr("height", cs + .3);
  const cont = svg.append("g");
  NF.axisB(svg, px, Y0 + 300, 5); NF.axisL(svg, py, X0, 5);
  NF.txt(svg, X0 + 150, 358, "x", C.muted, 11, "middle"); NF.txt(svg, 14, Y0 + 150, "y", C.muted, 11);
  const sl = svg.append("line").attr("x1", X0).attr("x2", X0 + 300).attr("stroke", "#fff").attr("stroke-dasharray", "5 3");
  TSK4.centres.forEach((c, r) => {
    const cx = px(c[0]) + (c[0] ? -12 : 12), cy = py(c[1]) + (c[1] ? 14 : -8);
    NF.txt(svg, cx, cy, "R" + (r + 1), NF.RULE[r], 11, "middle").attr("font-weight", 700);
  });
  // right panels
  const rx = d3.scaleLinear([0, 10], [390, 740]), ry = d3.scaleLinear([-2, 15], [200, 30]), wy = d3.scaleLinear([0, 1], [330, 235]);
  NF.axisB(svg, rx, 200, 5); NF.axisL(svg, ry, 390, 5); NF.axisB(svg, rx, 330, 5); NF.axisL(svg, wy, 390, 2);
  NF.txt(svg, 396, 24, "slice at y₀: output (white) and local models fᵣ (dashed)", C.muted, 10.5);
  NF.txt(svg, 396, 229, "normalized firing strengths w̄ᵣ(x, y₀)", C.muted, 10.5);
  const locs = [0, 1, 2, 3].map(r => svg.append("path").attr("fill", "none").attr("stroke", NF.RULE[r]).attr("stroke-dasharray", "5 3").attr("stroke-width", 1.4));
  const outL = svg.append("path").attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2.4);
  const wls = [0, 1, 2, 3].map(r => svg.append("path").attr("fill", "none").attr("stroke", NF.RULE[r]).attr("stroke-width", 2));
  const clip = svg.append("defs").append("clipPath").attr("id", "tsk-clip").append("rect").attr("x", 390).attr("y", 30).attr("width", 350).attr("height", 170);
  locs.forEach(l => l.attr("clip-path", "url(#tsk-clip)"));

  function draw() {
    const s = NF.val("tsk-s"), y0 = NF.val("tsk-y"), ord = NF.$("tsk-ord").value;
    const vals = new Array(n * n);
    cells.forEach(d => { const v = TSK4.eval(d.x, d.y, s, ord).out; d.v = v; vals[d.i * n + d.j] = v; });
    rects.attr("fill", d => col(d.v));
    const th = d3.range(1, 14, 1);
    const cc = d3.contours().size([n, n]).thresholds(th)(vals);
    const path = d3.geoPath(d3.geoIdentity().scale(cs).translate([X0, Y0]));
    cont.selectAll("path").data(cc).join("path").attr("d", path).attr("fill", "none").attr("stroke", "#0b0e14").attr("stroke-opacity", .45).attr("stroke-width", .8);
    sl.attr("y1", py(y0)).attr("y2", py(y0));
    const xs = d3.range(0, 10.0001, .05);
    const ev = xs.map(x => TSK4.eval(x, y0, s, ord));
    outL.attr("d", d3.line().x((x, i) => rx(x)).y((x, i) => ry(ev[i].out))(xs));
    locs.forEach((l, r) => l.attr("d", d3.line().x(x => rx(x)).y(x => ry(TSK4.f(r, x, y0, ord)))(xs)));
    wls.forEach((l, r) => l.attr("d", d3.line().x((x, i) => rx(x)).y((x, i) => wy(ev[i].wb[r]))(xs)));
    const mid = TSK4.eval(5, y0, s, ord);
    const range = d3.extent(cells, d => d.v);
    NF.$("tsk-readout").innerHTML = `σ = ${s.toFixed(1)}, ${+ord ? "first" : "zeroth"}-order · output range over the square [${NF.f2(range[0])}, ${NF.f2(range[1])}]` +
      `<br>at (5, ${y0.toFixed(1)}): w̄ = (${mid.wb.map(NF.f3).join(", ")}) — sum ${NF.f3(d3.sum(mid.wb))} · local fᵣ = (${mid.fs.map(NF.f2).join(", ")}) · blend <b>${NF.f3(mid.out)}</b>`;
  }
  NF.on(["tsk-s", "tsk-y", "tsk-ord"], draw); draw();
})();

/* ══════════ 5. fuzzy associative memory ══════════ */
(function () {
  const svg = d3.select("#fam-svg"); if (svg.empty()) return;
  const A0 = [0.2, 0.8, 0.5, 0.1], B0 = [0.6, 0.3, 1.0];
  let A = A0.slice(), B = B0.slice();
  const bw = 30, gap = 10;
  const panels = {
    A: { x: 40, y0: 40, h: 100, n: 4, col: C.A, name: "A (input fit vector)" },
    B: { x: 40, y0: 215, h: 90, n: 3, col: C.B, name: "B (output fit vector)" },
    F: { x: 520, y0: 40, h: 100, n: 3, col: C.B, name: "A ∘ W  vs B" },
    K: { x: 520, y0: 215, h: 90, n: 4, col: C.A, name: "B ∘ Wᵀ  vs A" }
  };
  Object.values(panels).forEach(p => {
    p.sy = d3.scaleLinear([0, 1], [p.y0 + p.h, p.y0]);
    svg.append("line").attr("x1", p.x - 4).attr("x2", p.x + p.n * (bw + gap)).attr("y1", p.y0 + p.h).attr("y2", p.y0 + p.h).attr("stroke", C.line);
    svg.append("line").attr("x1", p.x - 4).attr("x2", p.x + p.n * (bw + gap)).attr("y1", p.y0).attr("y2", p.y0).attr("stroke", C.line).attr("stroke-dasharray", "2 3");
    p.title = NF.txt(svg, p.x - 4, p.y0 - 12, p.name, C.ink, 11.5);
    p.g = svg.append("g");
  });
  const matG = svg.append("g");
  NF.txt(svg, 250, 30, "W (4 × 3)", C.ink, 11.5);
  const verdictF = NF.txt(svg, 520, 160, "", C.muted, 11), verdictK = NF.txt(svg, 520, 324, "", C.muted, 11);
  const colW = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

  function editable(p, arr) {
    const sel = p.g.selectAll("g.bar").data(arr.map((v, i) => i)).join(enter => {
      const g = enter.append("g").attr("class", "bar").style("cursor", "ns-resize");
      g.append("rect").attr("class", "hit").attr("fill", "transparent");
      g.append("rect").attr("class", "v").attr("rx", 3);
      g.append("text").attr("class", "t").attr("font-size", 10.5).attr("text-anchor", "middle").attr("fill", C.ink);
      return g;
    });
    const set = (yy, i) => { arr[i] = Math.round(Math.max(0, Math.min(1, p.sy.invert(yy))) * 20) / 20; draw(); };
    sel.on("click", function (event, i) { set(d3.pointer(event, svg.node())[1], i); })
      .call(d3.drag().on("start drag", function (event, i) { set(event.y, i); }));
    return sel;
  }
  const selA = editable(panels.A, A), selB = editable(panels.B, B);

  function renderBars(p, vals, ghost, isEdit) {
    const g = p.g;
    if (isEdit) {
      g.selectAll("g.bar").each(function (i) {
        const gg = d3.select(this), x = p.x + i * (bw + gap);
        gg.select("rect.hit").attr("x", x - 3).attr("y", p.y0 - 4).attr("width", bw + 6).attr("height", p.h + 8);
        gg.select("rect.v").attr("x", x).attr("width", bw).attr("y", p.sy(vals[i])).attr("height", p.sy(0) - p.sy(vals[i])).attr("fill", p.col).attr("fill-opacity", .8);
        gg.select("text.t").attr("x", x + bw / 2).attr("y", p.sy(vals[i]) - 3).text(vals[i].toFixed(2));
      });
      return;
    }
    const data = vals.map((v, i) => ({ v, g: ghost[i], i }));
    g.selectAll("rect.gh").data(data).join("rect").attr("class", "gh")
      .attr("x", d => p.x + d.i * (bw + gap) - 2).attr("width", bw + 4).attr("y", d => p.sy(d.g)).attr("height", d => p.sy(0) - p.sy(d.g))
      .attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "3 2");
    g.selectAll("rect.v").data(data).join("rect").attr("class", "v").attr("rx", 3)
      .attr("x", d => p.x + d.i * (bw + gap)).attr("width", bw).attr("y", d => p.sy(d.v)).attr("height", d => p.sy(0) - p.sy(d.v))
      .attr("fill", d => Math.abs(d.v - d.g) < 1e-9 ? C.good : C.bad).attr("fill-opacity", .75);
    g.selectAll("text.t").data(data).join("text").attr("class", "t").attr("font-size", 10.5).attr("text-anchor", "middle").attr("fill", C.ink)
      .attr("x", d => p.x + d.i * (bw + gap) + bw / 2).attr("y", d => p.sy(Math.max(d.v, d.g)) - 3).text(d => d.v.toFixed(2));
  }

  function draw() {
    const enc = NF.$("fam-enc").value;
    const W = A.map(a => B.map(b => enc === "min" ? Math.min(a, b) : a * b));
    const fwd = B.map((_, j) => d3.max(A, (a, i) => Math.min(a, W[i][j])));
    const bwd = A.map((_, i) => d3.max(B, (b, j) => Math.min(b, W[i][j])));
    renderBars(panels.A, A, null, true); renderBars(panels.B, B, null, true);
    renderBars(panels.F, fwd, B); renderBars(panels.K, bwd, A);
    const cw = 58, ch = 44, mx = 250, my = 44;
    const cellsD = []; W.forEach((row, i) => row.forEach((v, j) => cellsD.push({ i, j, v })));
    matG.selectAll("rect").data(cellsD).join("rect").attr("x", d => mx + d.j * cw).attr("y", d => my + d.i * ch).attr("width", cw - 3).attr("height", ch - 3).attr("rx", 4).attr("fill", d => colW(d.v));
    matG.selectAll("text.c").data(cellsD).join("text").attr("class", "c").attr("x", d => mx + d.j * cw + cw / 2 - 1).attr("y", d => my + d.i * ch + ch / 2 + 3)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", d => d.v > .55 ? "#0b0e14" : C.ink).text(d => d.v.toFixed(2));
    matG.selectAll("text.ri").data(A).join("text").attr("class", "ri").attr("x", mx - 8).attr("y", (d, i) => my + i * ch + ch / 2 + 3).attr("text-anchor", "end").attr("font-size", 10).attr("fill", C.A).text((d, i) => "a" + (i + 1) + "=" + d.toFixed(2));
    matG.selectAll("text.ci").data(B).join("text").attr("class", "ci").attr("x", (d, j) => mx + j * cw + cw / 2).attr("y", my + 4 * ch + 14).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", C.B).text((d, j) => "b" + (j + 1) + "=" + d.toFixed(2));
    const hA = d3.max(A), hB = d3.max(B);
    const okF = fwd.every((v, j) => Math.abs(v - B[j]) < 1e-9), okK = bwd.every((v, i) => Math.abs(v - A[i]) < 1e-9);
    verdictF.text(okF ? "perfect recall of B" : "imperfect: B " + (enc === "min" ? "clipped" : "scaled") + " by h(A) = " + hA.toFixed(2)).attr("fill", okF ? C.good : C.bad);
    verdictK.text(okK ? "perfect recall of A" : "imperfect: A " + (enc === "min" ? "clipped" : "scaled") + " by h(B) = " + hB.toFixed(2)).attr("fill", okK ? C.good : C.bad);
    NF.$("fam-readout").innerHTML = `h(A) = <b>${hA.toFixed(2)}</b>, h(B) = <b>${hB.toFixed(2)}</b> · ${enc === "min" ? "max–min" : "correlation-product"} encoding` +
      `<br>A ∘ W = (${fwd.map(NF.f2).join(", ")}) — predicted ${enc === "min" ? "min(h(A), B)" : "h(A)·B"} = (${B.map(b => NF.f2(enc === "min" ? Math.min(hA, b) : hA * b)).join(", ")})` +
      `<br>B ∘ Wᵀ = (${bwd.map(NF.f2).join(", ")}) — predicted ${enc === "min" ? "min(h(B), A)" : "h(B)·A"} = (${A.map(a => NF.f2(enc === "min" ? Math.min(hB, a) : hB * a)).join(", ")})`;
  }
  NF.on(["fam-enc"], draw);
  NF.$("fam-reset").addEventListener("click", () => { A0.forEach((v, i) => A[i] = v); B0.forEach((v, i) => B[i] = v); draw(); });
  NF.$("fam-norm").addEventListener("click", () => { const hA = d3.max(A) || 1, hB = d3.max(B) || 1; A.forEach((v, i) => A[i] = Math.round(v / hA * 100) / 100); B.forEach((v, i) => B[i] = Math.round(v / hB * 100) / 100); draw(); });
  draw();
})();

/* ══════════ 6. ANFIS network with live values ══════════ */
(function () {
  const svg = d3.select("#an-svg"); if (svg.empty()) return;
  const S = 3;
  const colX = [50, 175, 320, 460, 600, 715];
  const ruleY = [60, 145, 230, 315];
  const N = {
    x: { x: colX[0], y: 110, lab: "x" }, y: { x: colX[0], y: 265, lab: "y" },
    A1: { x: colX[1], y: 55, lab: "A₁ low" }, A2: { x: colX[1], y: 145, lab: "A₂ high" },
    B1: { x: colX[1], y: 235, lab: "B₁ low" }, B2: { x: colX[1], y: 325, lab: "B₂ high" },
    out: { x: colX[5], y: 190, lab: "Σ" }
  };
  const rulePrem = [["A1", "B1"], ["A1", "B2"], ["A2", "B1"], ["A2", "B2"]];
  ["layer 1 · MF", "layer 2 · Π", "layer 3 · N", "layer 4 · wᵣfᵣ", "layer 5 · Σ"].forEach((t, i) => NF.txt(svg, colX[i + 1], 18, t, C.muted, 10.5, "middle"));
  NF.txt(svg, colX[0], 18, "inputs", C.muted, 10.5, "middle");
  const eg = svg.append("g"), ng = svg.append("g");
  const edges = [];
  const E = (a, b, key) => { const e = { a, b, key, el: eg.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", b.x).attr("y2", b.y).attr("stroke", C.muted) }; edges.push(e); return e; };
  E(N.x, N.A1, "A1"); E(N.x, N.A2, "A2"); E(N.y, N.B1, "B1"); E(N.y, N.B2, "B2");
  const L2 = ruleY.map((y, r) => ({ x: colX[2], y })), L3 = ruleY.map(y => ({ x: colX[3], y })), L4 = ruleY.map(y => ({ x: colX[4], y }));
  rulePrem.forEach((p, r) => { E(N[p[0]], L2[r], p[0]); E(N[p[1]], L2[r], p[1]); });
  L2.forEach((a, r) => L3.forEach((b, s) => { const e = E(a, b, "w" + r); if (r !== s) e.el.attr("stroke-dasharray", "2 4"); e.cross = r !== s; e.r = r; }));
  L3.forEach((a, r) => { const e = E(a, L4[r], "wb" + r); e.r = r; });
  L4.forEach((a, r) => { const e = E(a, N.out, "o" + r); e.r = r; });
  function node(p, shape, col, lab) {
    const g = ng.append("g").attr("transform", `translate(${p.x},${p.y})`);
    if (shape === "sq") g.append("rect").attr("x", -22).attr("y", -15).attr("width", 44).attr("height", 30).attr("rx", 4);
    else g.append("circle").attr("r", 19);
    g.select(":first-child").attr("fill", "#11141b").attr("stroke", col).attr("stroke-width", 1.6);
    if (lab) g.append("text").attr("y", -22).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", col).text(lab);
    p.v = g.append("text").attr("y", 4).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", C.ink);
    return g;
  }
  node(N.x, "c", C.ink); node(N.y, "c", C.ink);
  ["A1", "A2", "B1", "B2"].forEach(k => node(N[k], "sq", k[0] === "A" ? C.A : C.B, N[k].lab));
  L2.forEach((p, r) => node(p, "c", NF.RULE[r], "Π  R" + (r + 1)));
  L3.forEach((p, r) => node(p, "c", NF.RULE[r], "N"));
  L4.forEach((p, r) => node(p, "sq", NF.RULE[r], "f" + (r + 1)));
  node(N.out, "c", C.good, "output");
  function draw() {
    const x = NF.val("an-x"), y = NF.val("an-y");
    const mu = { A1: NF.gauss(x, 0, S), A2: NF.gauss(x, 10, S), B1: NF.gauss(y, 0, S), B2: NF.gauss(y, 10, S) };
    const ev = TSK4.eval(x, y, S, 1);
    const o4 = ev.wb.map((w, r) => w * ev.fs[r]);
    N.x.v.text(x.toFixed(1)); N.y.v.text(y.toFixed(1));
    ["A1", "A2", "B1", "B2"].forEach(k => N[k].v.text(mu[k].toFixed(3)));
    L2.forEach((p, r) => p.v.text(ev.w[r].toFixed(3)));
    L3.forEach((p, r) => p.v.text(ev.wb[r].toFixed(3)));
    L4.forEach((p, r) => p.v.text(o4[r].toFixed(2)));
    N.out.v.text(ev.out.toFixed(2));
    edges.forEach(e => {
      let v = 0.5;
      if (mu[e.key] !== undefined) v = mu[e.key];
      else if (e.key[0] === "w" && e.key[1] !== "b") v = e.cross ? .15 : ev.w[e.r];
      else if (e.key.startsWith("wb")) v = ev.wb[e.r];
      else if (e.key[0] === "o") v = Math.min(1, Math.abs(o4[e.r]) / 8);
      e.el.attr("stroke-width", .6 + 4 * v).attr("stroke-opacity", .25 + .6 * v);
    });
    NF.$("an-readout").innerHTML = `layer 1: μ_A = (${NF.f3(mu.A1)}, ${NF.f3(mu.A2)}), μ_B = (${NF.f3(mu.B1)}, ${NF.f3(mu.B2)}) · layer 2: w = (${ev.w.map(NF.f3).join(", ")}), ∑w = ${NF.f3(d3.sum(ev.w))}` +
      `<br>layer 3: w̄ = (${ev.wb.map(NF.f3).join(", ")}) · local fᵣ = (${ev.fs.map(NF.f2).join(", ")}) · layer 4: w̄ᵣfᵣ = (${o4.map(NF.f2).join(", ")}) · layer 5: f = <b>${NF.f3(ev.out)}</b>`;
  }
  NF.on(["an-x", "an-y"], draw); draw();
})();

/* ══════════ 7. ANFIS fitting a 1-D function ══════════ */
(function () {
  const svg = d3.select("#af-svg"); if (svg.empty()) return;
  const TG = {
    bump: x => Math.sin(0.8 * x) + 0.8 * Math.exp(-2 * (x - 6.5) * (x - 6.5)),
    sinc: x => { const u = 1.8 * (x - 5); return Math.abs(u) < 1e-9 ? 1 : Math.sin(u) / u; },
    step: x => Math.tanh(2.5 * (x - 5)),
    kink: x => Math.abs(x - 5) / 5
  };
  const NS = 101, XS = d3.range(NS).map(i => 10 * i / (NS - 1));
  let T = [], models = {}, target = "", mCount = 0, timer = null;

  function newModel(m) {
    const sig = 10 / (m - 1) * 0.6;
    const P = { m, c: d3.range(m).map(i => 10 * i / (m - 1)), ls: d3.range(m).map(() => Math.log(sig)), p: new Array(m).fill(0), r: new Array(m).fill(0), loss: [], ep: 0 };
    P.adam = { t: 0, m: new Array(4 * m).fill(0), v: new Array(4 * m).fill(0) };
    return P;
  }
  function fwd(P, x) {
    const m = P.m, mu = new Array(m); let S = 1e-12;
    for (let i = 0; i < m; i++) { const s = Math.exp(P.ls[i]); mu[i] = Math.exp(-(x - P.c[i]) * (x - P.c[i]) / (2 * s * s)); S += mu[i]; }
    let f = 0; const fr = new Array(m);
    for (let i = 0; i < m; i++) { fr[i] = P.p[i] * x + P.r[i]; f += mu[i] / S * fr[i]; }
    return { mu, S, fr, f };
  }
  const rmse = P => Math.sqrt(d3.mean(XS, (x, k) => { const e = fwd(P, x).f - T[k]; return e * e; }));
  function solve(A, b) {                       // Gaussian elimination with partial pivoting
    const n = b.length;
    for (let c = 0; c < n; c++) {
      let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
      const d = A[c][c] || 1e-12;
      for (let r = c + 1; r < n; r++) { const f = A[r][c] / d; if (!f) continue; for (let k = c; k < n; k++) A[r][k] -= f * A[c][k]; b[r] -= f * b[c]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k]; x[r] = s / (A[r][r] || 1e-12); }
    return x;
  }
  function lse(P) {
    const m = P.m, n = 2 * m, A = d3.range(n).map(() => new Array(n).fill(0)), b = new Array(n).fill(0);
    XS.forEach((x, k) => {
      const F = fwd(P, x), phi = [];
      for (let i = 0; i < m; i++) { const w = F.mu[i] / F.S; phi.push(w * x, w); }
      for (let a = 0; a < n; a++) { b[a] += phi[a] * T[k]; for (let c = 0; c < n; c++) A[a][c] += phi[a] * phi[c]; }
    });
    for (let a = 0; a < n; a++) A[a][a] += 1e-8 * (1 + A[a][a]);
    const th = solve(A, b);
    for (let i = 0; i < m; i++) { P.p[i] = th[2 * i]; P.r[i] = th[2 * i + 1]; }
  }
  function gradStep(P, all) {
    const m = P.m, g = new Array(4 * m).fill(0);
    XS.forEach((x, k) => {
      const F = fwd(P, x), e = F.f - T[k];
      for (let i = 0; i < m; i++) {
        const s2 = Math.exp(2 * P.ls[i]), d = x - P.c[i], dfdmu = (F.fr[i] - F.f) / F.S;
        g[i] += e * dfdmu * F.mu[i] * d / s2;
        g[m + i] += e * dfdmu * F.mu[i] * d * d / s2;
        const wb = F.mu[i] / F.S;
        g[2 * m + i] += e * wb * x; g[3 * m + i] += e * wb;
      }
    });
    const lr = 0.05, b1 = .9, b2 = .999, ad = P.adam; ad.t++;
    const lim = all ? 4 * m : 2 * m;
    for (let j = 0; j < lim; j++) {
      const gj = g[j] / NS;
      ad.m[j] = b1 * ad.m[j] + (1 - b1) * gj; ad.v[j] = b2 * ad.v[j] + (1 - b2) * gj * gj;
      const st = lr * (ad.m[j] / (1 - Math.pow(b1, ad.t))) / (Math.sqrt(ad.v[j] / (1 - Math.pow(b2, ad.t))) + 1e-8);
      const i = j % m, grp = Math.floor(j / m);
      if (grp === 0) P.c[i] -= st; else if (grp === 1) P.ls[i] = Math.max(Math.log(0.08), Math.min(Math.log(20), P.ls[i] - st));
      else if (grp === 2) P.p[i] -= st; else P.r[i] -= st;
    }
  }
  function epoch(mode) {
    const P = models[mode];
    if (P.ep === 0) P.loss.push(rmse(P));
    if (mode === "hybrid") { lse(P); gradStep(P, false); }
    else if (mode === "lse") lse(P);
    else gradStep(P, true);
    P.ep++; P.loss.push(rmse(P));
  }

  // layout
  const x = d3.scaleLinear([0, 10], [50, 480]);
  const y = d3.scaleLinear([-1.4, 1.6], [250, 20]), my = d3.scaleLinear([0, 1], [390, 290]);
  NF.axisB(svg, x, 250, 10); NF.axisL(svg, y, 50, 5); NF.axisB(svg, x, 390, 10); NF.axisL(svg, my, 50, 2);
  NF.txt(svg, 56, 284, "membership functions of the current model", C.muted, 10.5);
  const lx = d3.scaleLinear([0, 100], [560, 740]), ly = d3.scaleLog([1e-4, 1], [390, 20]);
  NF.axisL(svg, ly, 560, 4, "~e");
  NF.txt(svg, 740, 380, "epoch →", C.muted, 10.5, "end"); NF.txt(svg, 566, 14, "RMSE (log)", C.muted, 10.5);
  const MODES = { hybrid: C.good, grad: C.bad, lse: C.B };
  const MN = { hybrid: "hybrid", grad: "gradient only", lse: "LSE only" };
  Object.keys(MODES).forEach((k, i) => { svg.append("rect").attr("x", 640).attr("y", 26 + i * 14).attr("width", 12).attr("height", 3).attr("fill", MODES[k]); NF.txt(svg, 656, 31 + i * 14, MN[k], C.muted, 10); });
  const gTarget = svg.append("path").attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "3 3");
  const gPts = svg.append("g"), gLoc = svg.append("g"), gMF = svg.append("g");
  const gFit = svg.append("path").attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2.4);
  const gLoss = svg.append("g");
  const clip = svg.append("defs").append("clipPath").attr("id", "af-clip").append("rect").attr("x", 50).attr("y", 20).attr("width", 430).attr("height", 230);
  gLoc.attr("clip-path", "url(#af-clip)");

  function reset() {
    if (timer) { timer.stop(); timer = null; }
    target = NF.$("af-target").value; mCount = NF.val("af-m");
    T = XS.map(TG[target]);
    models = { hybrid: newModel(mCount), grad: newModel(mCount), lse: newModel(mCount) };
    gPts.selectAll("circle").data(XS).join("circle").attr("r", 2).attr("fill", C.ink).attr("fill-opacity", .55).attr("cx", d => x(d)).attr("cy", (d, k) => y(T[k]));
    const fine = d3.range(0, 10.0001, .02);
    gTarget.attr("d", d3.line().x(t => x(t)).y(t => y(TG[target](t)))(fine));
    draw();
  }
  function draw() {
    const mode = NF.$("af-mode").value, P = models[mode];
    const fine = d3.range(0, 10.0001, .04);
    gFit.attr("d", d3.line().x(t => x(t)).y(t => y(Math.max(-3, Math.min(3, fwd(P, t).f))))(fine));
    const rules = d3.range(P.m);
    gMF.selectAll("path").data(rules).join("path").attr("fill", "none").attr("stroke-width", 1.6).attr("stroke", i => NF.RULE[i % 9])
      .attr("d", i => d3.line().x(t => x(t)).y(t => my(NF.gauss(t, P.c[i], Math.exp(P.ls[i]))))(fine));
    gLoc.selectAll("path").data(P.ep > 0 || mode !== "grad" ? rules : []).join("path").attr("fill", "none").attr("stroke-width", 1.4).attr("stroke-dasharray", "4 2")
      .attr("stroke", i => NF.RULE[i % 9]).attr("d", i => {
        const s = Math.exp(P.ls[i]), a = Math.max(0, P.c[i] - s), b = Math.min(10, P.c[i] + s);
        return a < b ? d3.line().x(t => x(t)).y(t => y(P.p[i] * t + P.r[i]))([a, b]) : null;
      });
    const maxEp = Math.max(100, ...Object.values(models).map(M => M.ep));
    lx.domain([0, maxEp]);
    gLoss.selectAll("*").remove();
    gLoss.append("g").attr("class", "axis").attr("transform", "translate(0,390)").call(d3.axisBottom(lx).ticks(4));
    Object.keys(models).forEach(k => {
      const L = models[k].loss; if (L.length < 2) return;
      gLoss.append("path").attr("fill", "none").attr("stroke", MODES[k]).attr("stroke-width", k === mode ? 2.4 : 1.4)
        .attr("d", d3.line().x((v, i) => lx(i)).y(v => ly(Math.max(1e-4, Math.min(1, v))))(L));
    });
    const cur = rmse(P);
    NF.$("af-readout").innerHTML = `showing <b>${MN[mode]}</b> · epoch ${P.ep} · RMSE <b>${cur.toExponential(2)}</b> · ${P.m} rules, ${4 * P.m} parameters (${2 * P.m} premise + ${2 * P.m} consequent)` +
      `<br>other modes: ` + Object.keys(models).filter(k => k !== mode).map(k => `${MN[k]} epoch ${models[k].ep}, RMSE ${rmse(models[k]).toExponential(2)}`).join(" · ") +
      `<br>centres: ${P.c.map(NF.f2).join(", ")} · widths σ: ${P.ls.map(v => NF.f2(Math.exp(v))).join(", ")}`;
  }
  function run(n) {
    if (timer) { timer.stop(); timer = null; return; }
    const mode = NF.$("af-mode").value; let left = n;
    timer = d3.interval(() => {
      for (let k = 0; k < 4 && left > 0; k++, left--) epoch(mode);
      draw();
      if (left <= 0 && timer) { timer.stop(); timer = null; }
    }, 30);
  }
  NF.$("af-run").addEventListener("click", () => run(100));
  NF.$("af-step").addEventListener("click", () => { epoch(NF.$("af-mode").value); draw(); });
  NF.$("af-reset").addEventListener("click", reset);
  NF.on(["af-target", "af-m"], reset);
  NF.on(["af-mode"], draw);
  reset();
})();

/* ══════════ 8. rule explosion ══════════ */
(function () {
  const svg = d3.select("#ex-svg"); if (svg.empty()) return;
  const x = d3.scaleLinear([1, 12], [60, 600]), y = d3.scaleLog([1, 1e10], [290, 20]);
  NF.axisB(svg, x, 290, 12); NF.axisL(svg, y, 60, 5, "~s");
  NF.txt(svg, 330, 318, "number of inputs n", C.muted, 11, "middle");
  svg.append("line").attr("x1", 60).attr("x2", 600).attr("y1", y(1e4)).attr("y2", y(1e4)).attr("stroke", C.muted).attr("stroke-dasharray", "6 4");
  NF.txt(svg, 596, y(1e4) - 5, "10,000 samples", C.muted, 10, "end");
  const S = [
    { k: "grid", col: C.bad, name: "grid ANFIS parameters", f: (n, m) => 2 * n * m + Math.pow(m, n) * (n + 1) },
    { k: "rules", col: C.bad, name: "grid ANFIS rules mⁿ", f: (n, m) => Math.pow(m, n), dash: "3 3" },
    { k: "scat", col: C.good, name: "scatter ANFIS (R rules)", f: (n, m, R) => 2 * n * R + R * (n + 1) },
    { k: "mlp", col: C.A, name: "MLP (H hidden)", f: (n, m, R, H) => H * (n + 1) + H + 1 }];
  const lines = S.map(s => svg.append("path").attr("fill", "none").attr("stroke", s.col).attr("stroke-width", 2.2).attr("stroke-dasharray", s.dash || null));
  const dots = S.map(s => svg.append("g"));
  S.forEach((s, i) => { svg.append("line").attr("x1", 618).attr("x2", 640).attr("y1", 40 + i * 18).attr("y2", 40 + i * 18).attr("stroke", s.col).attr("stroke-width", 2.2).attr("stroke-dasharray", s.dash || null); NF.txt(svg, 646, 44 + i * 18, s.name, C.muted, 10.5); });
  function draw() {
    const m = NF.val("ex-m"), R = NF.val("ex-r"), H = NF.val("ex-h"), ns = d3.range(1, 13);
    S.forEach((s, i) => {
      const pts = ns.map(n => [n, Math.min(1e10, s.f(n, m, R, H))]);
      lines[i].attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1])).defined(d => d[1] >= 1)(pts));
      dots[i].selectAll("circle").data(pts).join("circle").attr("r", 2.6).attr("fill", s.col).attr("cx", d => x(d[0])).attr("cy", d => y(d[1]));
    });
    const fmt = d3.format(",");
    NF.$("ex-readout").innerHTML = [2, 5, 10].map(n => `n = ${n}: grid ${fmt(Math.pow(m, n))} rules / <b>${fmt(S[0].f(n, m))}</b> params · scatter ${fmt(S[2].f(n, m, R))} · MLP ${fmt(S[3].f(n, m, R, H))}`).join("<br>");
  }
  NF.on(["ex-m", "ex-r", "ex-h"], draw); draw();
})();
