/* model-interpretability.viz.js — interactive figures for model-interpretability.html.
   Loaded after data.js / notes.js.
   1) PDP vs ICE vs ALE for f(x1,x2) = 0.6 x1 + 0.3 x1^2 + beta x1 x2 + 0.5 x2
   2) Exact Shapley values on the 3-feature coalition lattice + waterfall */

/* ---------------- 1) PDP vs ICE vs ALE ---------------- */
(function(){
  const svgEl = document.getElementById("pd-svg");
  if (!svgEl) return;
  const C = {accent:"#5b9cff", a2:"#ffb454", good:"#4ade80", bad:"#f87171", muted:"#9aa3b2", ink:"#e6e9ef", line:"#2a2f3a"};
  const W = 640, H = 300, m = {t:16, r:120, b:34, l:46};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const svg = d3.select(svgEl);
  const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);

  // deterministic RNG + standard normal (Box-Muller) so the figure is stable between reloads
  function mulberry32(a){ return function(){ a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function normals(n, seed){ const r = mulberry32(seed), out = [];
    for (let i = 0; i < n; i++){ const u = Math.max(r(), 1e-9), v = r(); out.push(Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)); } return out; }

  const LO = -2, HI = 2, NG = 41;
  const grid = d3.range(NG).map(k => LO + (HI - LO) * k / (NG - 1));
  const NI = 24;                                   // rows shown as ICE curves
  const x2s = d3.range(NI).map(i => -1 + 2 * (i + 0.5) / NI);
  const zs = normals(NI, 7);
  const NB = 2000;                                 // hidden larger sample used for ALE
  const bigU = (function(){ const r = mulberry32(11); return d3.range(NB).map(() => -1 + 2*r()); })();
  const bigZ = normals(NB, 13);
  const A = 1.6, S = 0.9;                          // x1 = rho*A*x2 + sqrt(1-rho^2)*S*z

  const f = (x1, x2, beta) => 0.6*x1 + 0.3*x1*x1 + beta*x1*x2 + 0.5*x2;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const color = d3.scaleSequential(t => d3.interpolateRdBu(1 - t)).domain([-1, 1]);

  const x = d3.scaleLinear().domain([LO, HI]).range([0, iw]);
  const y = d3.scaleLinear().range([ih, 0]);
  const xAxis = g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`);
  const yAxis = g.append("g").attr("class", "axis");
  g.append("text").attr("x", iw/2).attr("y", ih + 28).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10).text("x₁ (feature of interest)");
  const yLab = g.append("text").attr("transform", "rotate(-90)").attr("x", -ih/2).attr("y", -34).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10);
  const gIce = g.append("g"), gPts = g.append("g");
  const pdpPath = g.append("path").attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 3.2);
  const alePath = g.append("path").attr("fill", "none").attr("stroke", C.good).attr("stroke-width", 2.4).attr("stroke-dasharray", "7 4");

  // legend
  const leg = svg.append("g").attr("transform", `translate(${W - m.r + 14},${m.t + 6})`);
  const legRows = [["PDP (mean of ICE)", C.ink, 3.2, null], ["ALE", C.good, 2.4, "7 4"], ["ICE on data", C.muted, 1.2, null], ["ICE off data", C.muted, 1.2, "3 3"]];
  legRows.forEach((r, i) => {
    leg.append("line").attr("x1", 0).attr("x2", 20).attr("y1", i*18).attr("y2", i*18).attr("stroke", r[1]).attr("stroke-width", r[2]).attr("stroke-dasharray", r[3]);
    leg.append("text").attr("x", 26).attr("y", i*18 + 4).attr("fill", C.muted).attr("font-size", 10).text(r[0]);
  });
  leg.append("text").attr("x", 0).attr("y", 4*18 + 10).attr("fill", C.muted).attr("font-size", 10).text("ICE colour = x₂");
  const cb = leg.append("g").attr("transform", `translate(0,${4*18 + 18})`);
  d3.range(20).forEach(k => cb.append("rect").attr("x", k*4.5).attr("y", 0).attr("width", 4.6).attr("height", 8).attr("fill", color(-1 + 2*k/19)));
  cb.append("text").attr("x", 0).attr("y", 20).attr("fill", C.muted).attr("font-size", 9).text("−1");
  cb.append("text").attr("x", 90).attr("y", 20).attr("text-anchor", "end").attr("fill", C.muted).attr("font-size", 9).text("+1");

  const line = d3.line().x(d => x(d[0])).y(d => y(d[1]));

  function draw(){
    const beta = +d3.select("#pd-beta").property("value");
    const rho = +d3.select("#pd-rho").property("value");
    const centred = d3.select("#pd-center").property("checked");
    const showAle = d3.select("#pd-ale").property("checked");
    d3.select("#pd-beta-v").text(beta.toFixed(2));
    d3.select("#pd-rho-v").text(rho.toFixed(2));
    yLab.text(centred ? "f(x₁, x₂) − f(−2, x₂)" : "prediction f(x₁, x₂)");

    const sCond = Math.sqrt(1 - rho*rho) * S;           // conditional sd of x1 given x2
    const half = Math.max(2 * sCond, 0.25);             // plausible half-width around E[x1 | x2]
    const rows = x2s.map((x2, i) => {
      const x1 = clamp(rho*A*x2 + sCond*zs[i], LO, HI);
      const ctr = rho*A*x2;
      const anchor = centred ? f(LO, x2, beta) : 0;
      const curve = grid.map(t => [t, f(t, x2, beta) - anchor]);
      return {x1, x2, ctr, curve, anchor, on: t => Math.abs(t - ctr) <= half};
    });
    const pdp = grid.map((t, k) => [t, d3.mean(rows, r => r.curve[k][1])]);

    // ALE on the hidden big sample: quantile bins of x1, local differences, accumulate
    const big = bigU.map((u, i) => ({x2: u, x1: clamp(rho*A*u + sCond*bigZ[i], LO, HI)}));
    const sorted = big.map(d => d.x1).sort(d3.ascending);
    const K = 20, edges = d3.range(K + 1).map(k => d3.quantileSorted(sorted, k / K));
    const eff = new Array(K).fill(0), cnt = new Array(K).fill(0);
    big.forEach(d => {
      let k = d3.bisectRight(edges, d.x1) - 1; k = clamp(k, 0, K - 1);
      eff[k] += f(edges[k+1], d.x2, beta) - f(edges[k], d.x2, beta); cnt[k]++;
    });
    const acc = [0]; for (let k = 0; k < K; k++) acc.push(acc[k] + (cnt[k] ? eff[k]/cnt[k] : 0));
    const aleAt = t => { if (t <= edges[0]) return acc[0]; if (t >= edges[K]) return acc[K];
      const k = clamp(d3.bisectRight(edges, t) - 1, 0, K - 1), w = (t - edges[k]) / Math.max(edges[k+1] - edges[k], 1e-9);
      return acc[k] + w * (acc[k+1] - acc[k]); };
    let aleCurve = grid.map(t => [t, aleAt(t)]);
    // align ALE with the PDP: share the anchor at x1 = -2 in centred mode, else share the mean level on the grid
    const shift = centred ? (pdp[0][1] - aleCurve[0][1]) : (d3.mean(pdp, d => d[1]) - d3.mean(aleCurve, d => d[1]));
    aleCurve = aleCurve.map(d => [d[0], d[1] + shift]);

    const all = rows.flatMap(r => r.curve.map(d => d[1])).concat(pdp.map(d => d[1]), showAle ? aleCurve.map(d => d[1]) : []);
    const ext = d3.extent(all), pad = (ext[1] - ext[0]) * 0.06 || 1;
    y.domain([ext[0] - pad, ext[1] + pad]);
    xAxis.call(d3.axisBottom(x).ticks(8));
    yAxis.call(d3.axisLeft(y).ticks(6));

    // each ICE curve drawn as two layers: dashed full curve (off data) + solid on-support part
    const sel = gIce.selectAll("g.ice").data(rows);
    const ent = sel.enter().append("g").attr("class", "ice");
    ent.append("path").attr("class", "off").attr("fill", "none").attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    ent.append("path").attr("class", "on").attr("fill", "none").attr("stroke-width", 1.4);
    const merged = ent.merge(sel);
    merged.select("path.off").attr("d", r => line(r.curve)).attr("stroke", r => color(r.x2)).attr("stroke-opacity", 0.35);
    merged.select("path.on").attr("d", r => line(r.curve.filter(d => r.on(d[0])))).attr("stroke", r => color(r.x2)).attr("stroke-opacity", 0.9);

    const pts = gPts.selectAll("circle").data(rows);
    pts.enter().append("circle").attr("r", 3.2).attr("stroke", "#0f1117").attr("stroke-width", 1).merge(pts)
      .attr("cx", r => x(r.x1)).attr("cy", r => y(f(r.x1, r.x2, beta) - r.anchor)).attr("fill", r => color(r.x2));

    pdpPath.attr("d", line(pdp)).raise();
    alePath.attr("d", line(aleCurve)).style("display", showAle ? null : "none").raise();
    gPts.raise();

    // readouts: ICE slope heterogeneity (slope of each curve at x1 = 0 is 0.6 + beta*x2) and off-support share
    const slopes = rows.map(r => 0.6 + beta*r.x2);
    const sdSlope = d3.deviation(slopes) || 0;
    const pdpSlope = (pdp[NG-1][1] - pdp[0][1]) / (HI - LO);
    let off = 0; rows.forEach(r => grid.forEach(t => { if (!r.on(t)) off++; }));
    const offPct = 100 * off / (rows.length * NG);
    const aleSlope = (aleCurve[NG-1][1] - aleCurve[0][1]) / (HI - LO);
    d3.select("#pd-read").html(
      `PDP average slope <b>${pdpSlope.toFixed(2)}</b> · ICE slopes at x₁=0 range <b>${d3.min(slopes).toFixed(2)}</b> to <b>${d3.max(slopes).toFixed(2)}</b> (sd ${sdSlope.toFixed(2)}) · ` +
      `ICE grid points off the data <b style="color:${offPct > 40 ? C.bad : offPct > 15 ? C.a2 : C.good}">${offPct.toFixed(0)}%</b>` +
      (showAle ? ` · ALE average slope <b style="color:${C.good}">${aleSlope.toFixed(2)}</b>` : "") +
      `<br><span class="lbl">${beta < 0.05 ? "Additive model: ICE curves are parallel; the PDP is a faithful summary." :
        sdSlope > 0.4 ? "Strong interaction: the PDP averages opposite slopes into a gentle curve that describes few rows." :
        "Mild interaction: the ICE curves begin to fan out around the PDP."}${rho > 0.6 ? " High correlation: much of each ICE curve, and so much of the PDP, is computed on combinations the data never contains." : ""}</span>`
    );
  }
  ["#pd-beta", "#pd-rho", "#pd-center", "#pd-ale"].forEach(id => d3.select(id).on("input", draw).on("change", draw));
  draw();
})();

/* ---------------- 2) Exact Shapley values ---------------- */
(function(){
  const svgEl = document.getElementById("sh-svg");
  if (!svgEl) return;
  const C = {accent:"#5b9cff", a2:"#ffb454", good:"#4ade80", bad:"#f87171", muted:"#9aa3b2", ink:"#e6e9ef", line:"#2a2f3a", panel:"#1e222d"};
  const FC = ["#5b9cff", "#ffb454", "#c084fc"];     // one colour per feature
  const SUB = ["₁", "₂", "₃"];
  const W = 640, H = 330;
  const svg = d3.select(svgEl);
  const M = 3, fact = n => n <= 1 ? 1 : n * fact(n - 1);
  const weight = s => fact(s) * fact(M - s - 1) / fact(M);
  const WFRAC = {0: "⅓", 1: "⅙", 2: "⅓"};
  const bits = S => [0, 1, 2].filter(i => S & (1 << i));
  const setLabel = S => S === 0 ? "∅" : "{" + bits(S).map(i => "x" + SUB[i]).join(",") + "}";
  let focus = 0;

  // lattice layout (left 400px), empty set on top
  const LX = [80, 200, 320], nodeW = 96, nodeH = 36;
  const pos = {0: [200, 30], 1: [LX[0], 115], 2: [LX[1], 115], 4: [LX[2], 115], 3: [LX[0], 205], 5: [LX[1], 205], 6: [LX[2], 205], 7: [200, 290]};
  const edges = [];
  for (let S = 0; S < 8; S++) for (let i = 0; i < M; i++) if (!(S & (1 << i))) edges.push({S, T: S | (1 << i), i});

  svg.append("text").attr("x", 200).attr("y", 10).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10).text("coalition lattice: v(S) = f(x on S, b elsewhere)");
  const gE = svg.append("g"), gEL = svg.append("g"), gN = svg.append("g");
  const eSel = gE.selectAll("line").data(edges).enter().append("line")
    .attr("x1", d => pos[d.S][0]).attr("y1", d => pos[d.S][1] + nodeH/2)
    .attr("x2", d => pos[d.T][0]).attr("y2", d => pos[d.T][1] - nodeH/2);
  const nSel = gN.selectAll("g").data(d3.range(8)).enter().append("g").attr("transform", S => `translate(${pos[S][0] - nodeW/2},${pos[S][1] - nodeH/2})`);
  nSel.append("rect").attr("width", nodeW).attr("height", nodeH).attr("rx", 7).attr("fill", C.panel).attr("stroke", C.line);
  nSel.append("text").attr("class", "nl").attr("x", nodeW/2).attr("y", 14).attr("text-anchor", "middle").attr("fill", C.ink).attr("font-size", 11).text(setLabel);
  const nVal = nSel.append("text").attr("x", nodeW/2).attr("y", 29).attr("text-anchor", "middle").attr("fill", C.a2).attr("font-size", 11).attr("font-family", "monospace");

  // waterfall (right)
  const wx0 = 430, wx1 = 628, wy0 = 40, rowH = 50;
  svg.append("text").attr("x", (wx0 + wx1)/2).attr("y", 10).attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10).text("waterfall: v(∅) + φ₁ + φ₂ + φ₃ = f(x)");
  svg.append("line").attr("x1", 410).attr("x2", 410).attr("y1", 20).attr("y2", 318).attr("stroke", C.line);
  const wScale = d3.scaleLinear().range([wx0, wx1]);
  const wAxis = svg.append("g").attr("class", "axis").attr("transform", `translate(0,${wy0 + 5*rowH - 10})`);
  const wG = svg.append("g");
  const zeroLine = svg.append("line").attr("stroke", C.muted).attr("stroke-dasharray", "2 3").attr("y1", wy0 - 8).attr("y2", wy0 + 5*rowH - 10);

  function read(id){ return +d3.select(id).property("value"); }

  function draw(){
    const w = [read("#sh-w1"), read("#sh-w2"), read("#sh-w3")], w12 = read("#sh-w12");
    const xv = [read("#sh-x1"), read("#sh-x2"), read("#sh-x3")];
    const bv = +d3.select("#sh-base").property("value");
    const b = [bv, bv, bv];
    ["w1","w2","w3","w12","x1","x2","x3"].forEach(k => d3.select(`#sh-${k}-v`).text((+d3.select(`#sh-${k}`).property("value")).toFixed(1)));

    const f = z => w[0]*z[0] + w[1]*z[1] + w[2]*z[2] + w12*z[0]*z[1];
    const v = d3.range(8).map(S => f([0, 1, 2].map(i => (S & (1 << i)) ? xv[i] : b[i])));
    const phi = [0, 1, 2].map(i => d3.sum(edges.filter(e => e.i === i), e => weight(bits(e.S).length) * (v[e.T] - v[e.S])));

    nVal.text(S => "v = " + v[S].toFixed(2));
    eSel.attr("stroke", d => FC[d.i]).attr("stroke-width", d => d.i === focus ? 3 : 1.2).attr("stroke-opacity", d => d.i === focus ? 1 : 0.35);
    nSel.select("rect").attr("stroke", S => (S & (1 << focus)) ? FC[focus] : C.line);

    // labels on focused edges: weight x delta
    const fe = edges.filter(e => e.i === focus);
    const lab = gEL.selectAll("g").data(fe);
    const le = lab.enter().append("g");
    le.append("rect").attr("rx", 4).attr("height", 16).attr("fill", "#0f1117").attr("stroke", FC[0]);
    le.append("text").attr("font-size", 10).attr("font-family", "monospace").attr("text-anchor", "middle").attr("fill", C.ink);
    const lm = le.merge(lab);
    lm.each(function(e){
      const mx = (pos[e.S][0] + pos[e.T][0]) / 2, my = (pos[e.S][1] + pos[e.T][1]) / 2;
      const txt = `${WFRAC[bits(e.S).length]}·${(v[e.T] - v[e.S]).toFixed(2)}`;
      const gg = d3.select(this);
      gg.select("text").attr("x", mx).attr("y", my + 4).text(txt);
      const tw = txt.length * 6 + 8;
      gg.select("rect").attr("x", mx - tw/2).attr("y", my - 8).attr("width", tw).attr("stroke", FC[focus]);
    });
    lab.exit().remove();

    // waterfall
    const steps = [{lab: "v(∅) = f(b)", from: 0, to: v[0], col: C.muted}];
    let run = v[0];
    phi.forEach((p, i) => { steps.push({lab: "φ" + SUB[i], from: run, to: run + p, col: FC[i]}); run += p; });
    steps.push({lab: "f(x)", from: 0, to: v[7], col: C.ink});
    const vals = steps.flatMap(s => [s.from, s.to]).concat([0]);
    const ext = d3.extent(vals), pad = (ext[1] - ext[0]) * 0.08 || 1;
    wScale.domain([ext[0] - pad, ext[1] + pad]).nice();
    wAxis.call(d3.axisBottom(wScale).ticks(5));
    zeroLine.attr("x1", wScale(0)).attr("x2", wScale(0));
    const rs = wG.selectAll("g").data(steps);
    const re = rs.enter().append("g");
    re.append("rect").attr("height", 20).attr("rx", 3);
    re.append("text").attr("class", "lab").attr("font-size", 11).attr("fill", C.ink);
    re.append("text").attr("class", "val").attr("font-size", 10).attr("font-family", "monospace").attr("fill", C.a2);
    const rm = re.merge(rs);
    rm.attr("transform", (s, k) => `translate(0,${wy0 + k*rowH})`);
    rm.select("rect").attr("x", s => wScale(Math.min(s.from, s.to))).attr("width", s => Math.max(1.5, Math.abs(wScale(s.to) - wScale(s.from))))
      .attr("fill", s => s.col).attr("fill-opacity", (s, k) => k === focus + 1 ? 0.95 : 0.6);
    rm.select("text.lab").attr("x", wx0 - 4).attr("y", -4).text(s => s.lab);
    rm.select("text.val").attr("x", wx1).attr("y", -4).attr("text-anchor", "end")
      .text((s, k) => (k >= 1 && k <= 3 ? (s.to - s.from >= 0 ? "+" : "") + (s.to - s.from).toFixed(2) : s.to.toFixed(2)));

    // axiom checks
    const sum = phi[0] + phi[1] + phi[2];
    const notes = [];
    [0, 1, 2].forEach(i => {
      if (edges.filter(e => e.i === i).every(e => Math.abs(v[e.T] - v[e.S]) < 1e-9)) notes.push(`x${SUB[i]} is a null player (every marginal contribution is 0), so φ${SUB[i]} = 0`);
    });
    [[0,1],[0,2],[1,2]].forEach(([i, j]) => {
      const others = d3.range(8).filter(S => !(S & (1 << i)) && !(S & (1 << j)));
      if (others.every(S => Math.abs(v[S | (1 << i)] - v[S | (1 << j)]) < 1e-9))
        notes.push(`x${SUB[i]} and x${SUB[j]} are symmetric players, so φ${SUB[i]} = φ${SUB[j]} = ${phi[i].toFixed(2)}`);
    });
    const vl = S => S === 0 ? "v(∅)" : "v" + setLabel(S);
    const deriv = fe.map(e => `${WFRAC[bits(e.S).length]}·[${vl(e.T)} − ${vl(e.S)}]`).join(" + ");
    d3.select("#sh-read").html(
      `φ₁ = <b>${phi[0].toFixed(3)}</b> · φ₂ = <b>${phi[1].toFixed(3)}</b> · φ₃ = <b>${phi[2].toFixed(3)}</b> · base v(∅) = <b>${v[0].toFixed(3)}</b><br>` +
      `efficiency: φ₁ + φ₂ + φ₃ = <b>${sum.toFixed(3)}</b> = f(x) − v(∅) = <b>${(v[7] - v[0]).toFixed(3)}</b> <b style="color:${C.good}">✓</b><br>` +
      `φ${SUB[focus]} = ${deriv}` +
      (notes.length ? `<br><span style="color:${C.good}">${notes.join(" · ")}</span>` : "")
    );
  }

  [0, 1, 2].forEach(i => d3.select("#sh-f" + i).on("click", () => { focus = i; draw(); }));
  ["#sh-w1","#sh-w2","#sh-w3","#sh-w12","#sh-x1","#sh-x2","#sh-x3","#sh-base"].forEach(id => d3.select(id).on("input", draw).on("change", draw));
  draw();
})();
