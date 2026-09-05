/* descriptive.viz.js — the seven visualizations on math/statistics/descriptive.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #oring-svg   selecting on the outcome (the launch record)
     2  #ans-svg     Anscombe's quartet, draggable
     3  #hist-svg    bin width / bin origin / kernel bandwidth
     4  #centre-svg  mean vs median vs trimmed mean, tug-of-war
     5  #box-svg     boxplot from the five-number summary
     6  #ecdf-svg    empirical distribution function + DKW band + plug-in estimates
     7  #corr-svg    what correlation sees and what it misses                        */

/* ─────────────────── 1 · selecting on the outcome ─────────────────── */
(function () {
  const svg = d3.select("#oring-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 360;
  const TEMP = [66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58];
  const DMG = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 1];
  const LAUNCH = 31;

  // deterministic dodge so coincident flights stay countable
  const seen = {}, dodge = TEMP.map((t, i) => {
    const k = t + "|" + DMG[i];
    seen[k] = (seen[k] || 0) + 1;
    return seen[k] - 1;
  });
  const mult = {};
  TEMP.forEach((t, i) => { const k = t + "|" + DMG[i]; mult[k] = (mult[k] || 0) + 1; });
  const pts = TEMP.map((t, i) => {
    const k = t + "|" + DMG[i];
    return { t: t, d: DMG[i], off: (dodge[i] - (mult[k] - 1) / 2) * 0.115 };
  });

  const F = ST.frame(svg, W, H, { l: 54, r: 22, t: 20, b: 50 });
  const x = d3.scaleLinear().domain([26, 86]).range([0, F.iw]);
  const y = d3.scaleLinear().domain([-0.45, 3.4]).range([F.ih, 0]);

  const gCold = F.g.append("g").attr("opacity", 0);
  gCold.append("rect").attr("x", x(26)).attr("y", 0).attr("width", x(52.5) - x(26))
    .attr("height", F.ih).attr("fill", SC.bad).attr("opacity", 0.09);
  gCold.append("text").attr("x", x(26) + 6).attr("y", 14).attr("font-size", 10).attr("fill", SC.bad)
    .text("no flight was ever made in here");
  ST.gridY(F.g, y, F.iw, 4);
  ST.axisB(F.g, x, F.ih, 7, "launch temperature (°F)");
  F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickValues([0, 1, 2, 3]).tickFormat(d3.format("d")));
  F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11).attr("fill", SC.muted)
    .text("sealing rings showing thermal distress");

  const gFit = F.g.append("g");
  const gLaunch = F.g.append("g").attr("opacity", 0);
  const gPts = F.g.append("g");

  gLaunch.append("line").attr("x1", x(LAUNCH)).attr("x2", x(LAUNCH)).attr("y1", 0).attr("y2", F.ih)
    .attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
  gLaunch.append("text").attr("x", x(LAUNCH) + 6).attr("y", 30).attr("font-size", 11).attr("fill", SC.a2)
    .text("24th launch · 31 °F");

  const cbZ = document.getElementById("or-zeros");
  const cbL = document.getElementById("or-launch");
  const cbF = document.getElementById("or-fit");
  const out = document.getElementById("oring-readout");

  function update() {
    const showZeros = cbZ.checked, showFit = cbF.checked;
    const shown = pts.filter(p => showZeros || p.d > 0);

    const sel = gPts.selectAll("circle").data(shown, (d, i) => d.t + "|" + d.d + "|" + d.off);
    sel.exit().remove();
    sel.enter().append("circle")
      .attr("r", 6).attr("stroke", "#0f1117").attr("stroke-width", 1.4)
      .merge(sel)
      .attr("cx", d => x(d.t)).attr("cy", d => y(d.d + d.off))
      .attr("fill", d => d.d > 0 ? SC.bad : SC.accent)
      .attr("fill-opacity", d => d.d > 0 ? 0.95 : 0.75);

    gLaunch.attr("opacity", cbL.checked ? 1 : 0);
    gCold.attr("opacity", cbL.checked ? 1 : 0);

    const tx = shown.map(p => p.t), ty = shown.map(p => p.d);
    const fit = ST.lsLine(tx, ty), r = ST.corr(tx, ty);
    gFit.selectAll("*").remove();
    if (showFit && isFinite(fit.b)) {
      gFit.append("line")
        .attr("x1", x(52)).attr("y1", y(fit.a + fit.b * 52))
        .attr("x2", x(84)).attr("y2", y(fit.a + fit.b * 84))
        .attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "7 5").attr("opacity", 0.9);
    }

    let msg = `showing <b>${shown.length}</b> flights · r = <b>${ST.fmt(r, 3)}</b>` +
      ` · fitted slope = <b>${ST.fmt(fit.b, 4)}</b> damaged rings per °F`;
    if (showZeros) {
      const c = pts.filter(p => p.t < 65), w = pts.filter(p => p.t >= 65);
      msg += `<br>below 65 °F: <b>${c.filter(p => p.d > 0).length} of ${c.length}</b> flights damaged` +
        ` · at 65 °F and above: <b>${w.filter(p => p.d > 0).length} of ${w.length}</b>`;
    } else {
      msg += `<br>conditioned on damage: the comparison group is missing, so temperature looks irrelevant`;
    }
    out.innerHTML = msg;
  }

  [cbZ, cbL, cbF].forEach(c => c.addEventListener("change", update));
  update();
})();

/* ─────────────────── 2 · Anscombe's quartet ─────────────────── */
(function () {
  const svg = d3.select("#ans-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 700, H = 540;
  const AX = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5];
  const ORIG = [
    { name: "I · a noisy linear relation", x: AX.slice(), y: [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68] },
    { name: "II · an exact parabola", x: AX.slice(), y: [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74] },
    { name: "III · a line plus one bad point", x: AX.slice(), y: [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73] },
    { name: "IV · one leverage point", x: [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], y: [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89] }
  ];
  const D = ORIG.map(p => ({ name: p.name, x: p.x.slice(), y: p.y.slice() }));

  const F = ST.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
  const CW = 350, CH = 268, PL = 48, PT = 26, PW = 278, PH = 158;
  const x = d3.scaleLinear().domain([2, 21]).range([0, PW]);
  const y = d3.scaleLinear().domain([2, 14]).range([PH, 0]);
  const out = document.getElementById("ans-readout");
  const cbLine = document.getElementById("ans-line");

  const panels = D.map((p, i) => {
    const cx = (i % 2) * CW, cy = Math.floor(i / 2) * CH;
    const g = F.g.append("g").attr("transform", `translate(${cx + PL},${cy + PT})`);
    g.append("rect").attr("x", -PL + 10).attr("y", -PT + 8).attr("width", CW - 22).attr("height", CH - 20)
      .attr("rx", 8).attr("fill", SC.panel2).attr("stroke", SC.line);
    ST.gridY(g, y, PW, 4);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${PH})`).call(d3.axisBottom(x).ticks(5));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11.5).attr("font-weight", 600)
      .attr("fill", SC.ink).text(p.name);
    const line = g.append("line").attr("stroke", SC.a2).attr("stroke-width", 2).attr("opacity", 0.9);
    const dots = g.append("g");
    const t1 = g.append("text").attr("x", -PL + 20).attr("y", PH + 40).attr("font-size", 10)
      .attr("font-family", "SF Mono,Menlo,monospace").attr("fill", SC.muted);
    const t2 = g.append("text").attr("x", -PL + 20).attr("y", PH + 54).attr("font-size", 10)
      .attr("font-family", "SF Mono,Menlo,monospace").attr("fill", SC.muted);
    const t3 = g.append("text").attr("x", -PL + 20).attr("y", PH + 68).attr("font-size", 10)
      .attr("font-family", "SF Mono,Menlo,monospace").attr("fill", SC.a2);
    return { g, line, dots, t1, t2, t3, i };
  });

  function stats(p) {
    const r = ST.corr(p.x, p.y), fit = ST.lsLine(p.x, p.y);
    return {
      mx: ST.mean(p.x), vx: ST.variance(p.x), my: ST.mean(p.y), vy: ST.variance(p.y),
      r: r, a: fit.a, b: fit.b
    };
  }

  function draw() {
    const S = D.map(stats);
    panels.forEach((P, i) => {
      const p = D[i], s = S[i];
      const sel = P.dots.selectAll("circle").data(p.x.map((v, j) => j));
      sel.enter().append("circle")
        .attr("class", "dragpt").attr("r", 5.2).attr("fill", SC.accent)
        .attr("stroke", "#0f1117").attr("stroke-width", 1.2).style("cursor", "grab")
        .call(d3.drag().on("drag", function (e, j) {
          p.x[j] = Math.max(2.2, Math.min(20.8, x.invert(e.x)));
          p.y[j] = Math.max(2.2, Math.min(13.8, y.invert(e.y)));
          draw();
        }))
        .merge(sel)
        .attr("cx", j => x(p.x[j])).attr("cy", j => y(p.y[j]));

      P.line.attr("opacity", cbLine.checked ? 0.9 : 0)
        .attr("x1", x(2)).attr("y1", y(s.a + s.b * 2))
        .attr("x2", x(21)).attr("y2", y(s.a + s.b * 21));

      P.t1.text(`x̄ ${ST.fmt(s.mx)}   s²x ${ST.fmt(s.vx)}   ȳ ${ST.fmt(s.my)}   s²y ${ST.fmt(s.vy)}`);
      P.t2.text(`r ${ST.fmt(s.r, 3)}   r² ${ST.fmt(s.r * s.r, 3)}`);
      P.t3.text(`ŷ = ${ST.fmt(s.a)} + ${ST.fmt(s.b, 3)}·x`);
    });

    // "do the four panels still agree?" — the widest spread of any one summary across the four
    const cols = ["mx", "vx", "my", "vy", "r", "a", "b"];
    let worst = 0, worstName = "";
    const NICE = { mx: "x̄", vx: "s²x", my: "ȳ", vy: "s²y", r: "r", a: "intercept", b: "slope" };
    cols.forEach(c => {
      const v = S.map(s => s[c]), d = Math.max(...v) - Math.min(...v);
      if (d > worst) { worst = d; worstName = NICE[c]; }
    });
    out.innerHTML = worst < 0.01
      ? `All four panels still agree: <b>x̄ = ${ST.fmt(S[0].mx)}</b>, <b>s²x = ${ST.fmt(S[0].vx)}</b>, ` +
      `<b>ȳ = ${ST.fmt(S[0].my)}</b>, <b>s²y = ${ST.fmt(S[0].vy)}</b>, <b>r = ${ST.fmt(S[0].r, 3)}</b>, ` +
      `<b>ŷ = ${ST.fmt(S[0].a)} + ${ST.fmt(S[0].b, 3)}x</b> — every summary matching to within ` +
      `<b>${ST.fmt(worst, 4)}</b> (worst case: ${worstName}), and they still look nothing alike.`
      : `The four panels no longer match: <b>${worstName}</b> now differs by <b>${ST.fmt(worst, 3)}</b> across them. ` +
      `Move a point back and the shared summary reappears — the point is how much freedom a fixed set of statistics leaves you.`;
  }

  cbLine.addEventListener("change", draw);
  document.getElementById("ans-reset").addEventListener("click", () => {
    // mutate in place so the drag handlers keep pointing at the live arrays
    D.forEach((p, i) => { ORIG[i].x.forEach((v, j) => { p.x[j] = v; p.y[j] = ORIG[i].y[j]; }); });
    draw();
  });

  draw();
})();

/* ─────────────────── 3 · bin width, bin origin, bandwidth ─────────────────── */
(function () {
  const svg = d3.select("#hist-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 360;
  const r = ST.rng(20260905), data = [];
  while (data.length < 220) {
    const v = (r() < 0.38) ? 52 + 5.5 * ST.randn(r) : 80 + 6.8 * ST.randn(r);
    if (v > 31 && v < 104) data.push(Math.round(v * 10) / 10);
  }
  const LO = 38, HI = 106;

  const F = ST.frame(svg, W, H, { l: 56, r: 20, t: 18, b: 54 });
  const x = d3.scaleLinear().domain([LO, HI]).range([0, F.iw]);
  const y = d3.scaleLinear().range([F.ih - 16, 0]);
  const gBars = F.g.append("g");
  const gKde = F.g.append("g");
  const gRug = F.g.append("g").attr("transform", `translate(0,${F.ih - 13})`);
  const gY = F.g.append("g").attr("class", "axis");
  ST.axisB(F.g, x, F.ih, 8, "measurement");
  F.g.append("text").attr("x", 0).attr("y", -4).attr("font-size", 11).attr("fill", SC.muted)
    .text("density (proportion per unit)");

  gRug.selectAll("line").data(data).join("line")
    .attr("x1", d => x(d)).attr("x2", d => x(d)).attr("y1", 0).attr("y2", 9)
    .attr("stroke", SC.muted).attr("stroke-opacity", 0.4);

  const sW = document.getElementById("h-w"), sO = document.getElementById("h-o"), sH = document.getElementById("h-h");
  const cK = document.getElementById("h-kde"), out = document.getElementById("hist-readout");
  const rules = ST.binRules(data), silver = ST.silverman(data);

  function update() {
    const w = +sW.value, off = +sO.value, h = +sH.value;
    document.getElementById("h-wv").textContent = w;
    document.getElementById("h-ov").textContent = ST.fmt(off, 2);
    document.getElementById("h-hv").textContent = h;

    const x0 = LO - w + off * w, k = Math.ceil((HI - x0) / w);
    const bins = ST.histBins(data, x0, w, k);
    const grid = d3.range(LO, HI, (HI - LO) / 260);
    const dens = ST.kde(data, h, grid);
    const top = Math.max(d3.max(bins, b => b.density) || 0, cK.checked ? d3.max(dens, d => d.y) : 0);
    y.domain([0, top * 1.14]);
    gY.call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".3f")));

    const sel = gBars.selectAll("rect").data(bins);
    sel.exit().remove();
    sel.enter().append("rect").merge(sel)
      .attr("x", b => x(b.x0)).attr("width", b => Math.max(0.6, x(b.x1) - x(b.x0) - 1))
      .attr("y", b => y(b.density)).attr("height", b => Math.max(0, y(0) - y(b.density)))
      .attr("fill", SC.accent).attr("fill-opacity", 0.42).attr("stroke", SC.accent).attr("stroke-opacity", 0.75);

    gKde.selectAll("*").remove();
    if (cK.checked) {
      gKde.append("path").datum(dens)
        .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2)
        .attr("d", d3.line().x(d => x(d.x)).y(d => y(d.y)).curve(d3.curveBasis));
    }

    // count the clearly separated peaks of the density estimate
    let peaks = 0;
    for (let i = 2; i < dens.length - 2; i++) {
      if (dens[i].y > dens[i - 1].y && dens[i].y > dens[i + 1].y && dens[i].y > 0.15 * d3.max(dens, d => d.y)) peaks++;
    }
    const nonEmpty = bins.filter(b => b.n > 0).length;
    out.innerHTML =
      `bin width <b>${w}</b> · origin offset <b>${ST.fmt(off, 2)}</b>·w · <b>${k}</b> bins, <b>${nonEmpty}</b> non-empty · ` +
      `tallest bin holds <b>${d3.max(bins, b => b.n)}</b> of ${data.length}` +
      `<br>suggested widths — Freedman–Diaconis <b>${ST.fmt(rules.fd, 2)}</b> · Scott <b>${ST.fmt(rules.scott, 2)}</b> · ` +
      `Sturges asks for <b>${rules.sturges}</b> bins (width ${ST.fmt((HI - LO) / rules.sturges, 2)})` +
      (cK.checked
        ? `<br>bandwidth <b>${h}</b> (a resistant default would be <b>${ST.fmt(silver, 2)}</b>) — the estimate shows <b>${peaks}</b> peak${peaks === 1 ? "" : "s"}`
        : "");
  }

  [sW, sO, sH].forEach(s => s.addEventListener("input", update));
  cK.addEventListener("change", update);
  document.getElementById("h-fd").addEventListener("click", () => {
    sW.value = Math.max(1, Math.round(rules.fd * 2) / 2); update();
  });
  update();
})();

/* ─────────────────── 4 · centre: tug-of-war ─────────────────── */
(function () {
  const svg = d3.select("#centre-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 300;
  const START = [22, 28, 31, 35, 38, 40, 42, 45, 48, 52, 55, 60, 66, 78, 110];
  let v = START.slice();

  const F = ST.frame(svg, W, H, { l: 34, r: 30, t: 60, b: 56 });
  const x = d3.scaleLinear().domain([0, 130]).range([0, F.iw]);
  const ROW = F.ih - 66;
  const gAxis = F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`);
  F.g.append("text").attr("x", F.iw).attr("y", F.ih + 31).attr("text-anchor", "end")
    .attr("font-size", 11).attr("fill", SC.muted).text("value");

  const gMark = F.g.append("g");
  const gPts = F.g.append("g");
  const out = document.getElementById("centre-readout");
  const sTrim = document.getElementById("c-trim");

  // a fixed legend row keeps the three values readable even when the lines sit on top of one another
  const marks = [
    { key: "mean", col: SC.accent, lab: "mean", lx: 0 },
    { key: "med", col: SC.a2, lab: "median", lx: 190 },
    { key: "trim", col: SC.good, lab: "trimmed mean", lx: 380 }
  ].map(m => {
    const g = gMark.append("g");
    m.line = g.append("line").attr("y1", -18).attr("y2", ROW + 46)
      .attr("stroke", m.col).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
    m.cap = g.append("path").attr("fill", m.col).attr("d", "M0,-18 L-6,-28 L6,-28 Z");
    F.g.append("circle").attr("cx", m.lx + 5).attr("cy", -38).attr("r", 5).attr("fill", m.col);
    m.txt = F.g.append("text").attr("x", m.lx + 16).attr("y", -34).attr("font-size", 12)
      .attr("font-weight", 600).attr("fill", m.col);
    return m;
  });
  const fulcrum = gMark.append("path").attr("fill", SC.accent).attr("opacity", 0.9)
    .attr("d", "M0,0 L-9,15 L9,15 Z");

  function update() {
    const a = +sTrim.value;
    document.getElementById("c-trimv").textContent = ST.fmt(a, 2);
    const m = ST.mean(v), md = ST.median(v), tm = ST.trimmedMean(v, a);
    const k = Math.floor(v.length * a);

    x.domain([0, Math.max(130, Math.max(...v) * 1.06)]);
    gAxis.call(d3.axisBottom(x).ticks(8));

    // dodge coincident points into up to three rows so every dot stays grabbable
    const order = v.map((_, i) => i).sort((p, q) => v[p] - v[q]);
    const row = new Array(v.length).fill(0);
    let lastPx = -1e9, lane = 0;
    order.forEach(i => {
      const px = x(v[i]);
      if (px - lastPx < 17) lane = (lane + 1) % 3;
      else { lane = 0; lastPx = px; }
      row[i] = lane;
    });

    const sel = gPts.selectAll("circle").data(v.map((_, i) => i));
    sel.enter().append("circle")
      .attr("class", "dragpt").attr("r", 7.5)
      .attr("stroke", "#0f1117").attr("stroke-width", 1.5).style("cursor", "grab")
      .call(d3.drag().on("drag", function (e, i) {
        v[i] = Math.max(0, Math.min(600, x.invert(e.x)));
        update();
      }))
      .merge(sel)
      .attr("cx", i => x(v[i]))
      .attr("cy", i => ROW + (row[i] - 1) * 17)
      .attr("fill", i => {
        const s = ST.asc(v), lo = s[k], hi = s[v.length - 1 - k];
        return (v[i] < lo || v[i] > hi) ? SC.bad : SC.muted;
      })
      .attr("fill-opacity", 0.9);

    const vals = { mean: m, med: md, trim: tm };
    marks.forEach(mk => {
      const px = x(vals[mk.key]);
      mk.line.attr("x1", px).attr("x2", px);
      mk.cap.attr("transform", `translate(${px},0)`);
      mk.txt.text(`${mk.lab} ${ST.fmt(vals[mk.key], 2)}`);
    });
    fulcrum.attr("transform", `translate(${x(m)},${ROW + 30})`);

    out.innerHTML =
      `n = <b>${v.length}</b> · mean <b>${ST.fmt(m, 2)}</b> · median <b>${ST.fmt(md, 2)}</b> · ` +
      `${ST.fmt(a * 100, 0)}% trimmed mean <b>${ST.fmt(tm, 2)}</b> (drops ${k} from each end) · ` +
      `mean − median = <b>${ST.fmt(m - md, 2)}</b>` +
      `<br>breakdown points — mean <b>${ST.fmt(1 / v.length, 3)}</b> (one point) · ` +
      `trimmed <b>${ST.fmt(a, 2)}</b> · median <b>${ST.fmt(Math.floor((v.length - 1) / 2) / v.length, 3)}</b> ` +
      `· red points are the ones the trim discards`;
  }

  sTrim.addEventListener("input", update);
  document.getElementById("c-reset").addEventListener("click", () => { v = START.slice(); update(); });
  document.getElementById("c-far").addEventListener("click", () => {
    let j = 0; v.forEach((val, i) => { if (val > v[j]) j = i; });
    v[j] = 300; update();
  });
  update();
})();

/* ─────────────────── 5 · boxplot from the five-number summary ─────────────────── */
(function () {
  const svg = d3.select("#box-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 330;
  function build(kind) {
    const cfg = { skew: [1234, 44], sym: [313, 45], bimodal: [1717, 44] }[kind];
    const r = ST.rng(cfg[0]), n = cfg[1], a = [];
    for (let i = 0; i < n; i++) {
      let v;
      if (kind === "skew") v = 24 * Math.exp(0.78 * ST.randn(r));
      else if (kind === "sym") v = 100 + 18 * ST.randn(r);
      else v = (i < n / 2 ? 62 : 142) + 8 * ST.randn(r);
      a.push(Math.round(v * 10) / 10);
    }
    return a;
  }
  const SETS = { skew: build("skew"), sym: build("sym"), bimodal: build("bimodal") };

  const F = ST.frame(svg, W, H, { l: 30, r: 26, t: 26, b: 52 });
  const x = d3.scaleLinear().range([0, F.iw]);
  const ROWPTS = 54, ROWBOX = 150, BOXH = 54;
  const gAx = F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`);
  F.g.append("text").attr("x", 0).attr("y", 6).attr("font-size", 11).attr("fill", SC.muted)
    .text("the raw values");
  F.g.append("text").attr("x", 0).attr("y", ROWBOX - 22).attr("font-size", 11).attr("fill", SC.muted)
    .text("the five-number summary drawn as a box");
  const gPts = F.g.append("g"), gBox = F.g.append("g");
  const out = document.getElementById("box-readout");
  const sK = document.getElementById("b-k"), sC = document.getElementById("b-conv"), sD = document.getElementById("b-data");

  function update() {
    const data = SETS[sD.value], k = +sK.value, conv = sC.value;
    document.getElementById("b-kv").textContent = ST.fmt(k, 1);
    const b = ST.boxStats(data, k, conv);
    // show the fences when they are near the data, but never let a far-off fence squash the plot
    const lo = Math.min(...data), hi = Math.max(...data), span = hi - lo;
    const dLo = Math.min(lo, Math.max(b.loFence, lo - 0.5 * span)) - 0.06 * span;
    const dHi = Math.max(hi, Math.min(b.hiFence, hi + 0.5 * span)) + 0.06 * span;
    x.domain([dLo, dHi]);
    gAx.call(d3.axisBottom(x).ticks(8));

    const jit = ST.jitterY(data, 31, 15);
    const sel = gPts.selectAll("circle").data(data);
    sel.exit().remove();
    sel.enter().append("circle").attr("r", 4.2).attr("stroke", "#0f1117").attr("stroke-width", 1)
      .merge(sel)
      .attr("cx", d => x(d)).attr("cy", (d, i) => ROWPTS + jit[i])
      .attr("fill", d => (d < b.loFence || d > b.hiFence) ? SC.bad : SC.accent)
      .attr("fill-opacity", 0.8);

    gBox.selectAll("*").remove();
    const yb = ROWBOX, mid = yb + BOXH / 2;
    // fences
    let offScale = 0;
    [[b.loFence, "Q₁ − k·IQR", -1], [b.hiFence, "Q₃ + k·IQR", 1]].forEach(([fx, lab, dir]) => {
      if (fx < dLo || fx > dHi) {                       // fence lies beyond the plotted range
        offScale++;
        const ex = dir < 0 ? 4 : F.iw - 4;
        gBox.append("text").attr("x", ex).attr("y", yb + BOXH / 2 + 4).attr("text-anchor", dir < 0 ? "start" : "end")
          .attr("font-size", 10).attr("fill", SC.bad).attr("opacity", 0.75)
          .text(dir < 0 ? "◀ fence off scale" : "fence off scale ▶");
        return;
      }
      gBox.append("line").attr("x1", x(fx)).attr("x2", x(fx)).attr("y1", yb - 14).attr("y2", yb + BOXH + 14)
        .attr("stroke", SC.bad).attr("stroke-width", 1).attr("stroke-dasharray", "3 4").attr("opacity", 0.75);
      gBox.append("text").attr("x", x(fx)).attr("y", yb + BOXH + 27).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", SC.bad).attr("opacity", 0.8).text(lab);
    });
    // whiskers
    gBox.append("line").attr("x1", x(b.whiskLo)).attr("x2", x(b.q1)).attr("y1", mid).attr("y2", mid)
      .attr("stroke", SC.muted).attr("stroke-width", 1.6);
    gBox.append("line").attr("x1", x(b.q3)).attr("x2", x(b.whiskHi)).attr("y1", mid).attr("y2", mid)
      .attr("stroke", SC.muted).attr("stroke-width", 1.6);
    [b.whiskLo, b.whiskHi].forEach(wv => gBox.append("line")
      .attr("x1", x(wv)).attr("x2", x(wv)).attr("y1", mid - 12).attr("y2", mid + 12)
      .attr("stroke", SC.muted).attr("stroke-width", 1.6));
    // box + median
    gBox.append("rect").attr("x", x(b.q1)).attr("y", yb).attr("width", Math.max(1, x(b.q3) - x(b.q1)))
      .attr("height", BOXH).attr("fill", SC.accent).attr("fill-opacity", 0.16)
      .attr("stroke", SC.accent).attr("stroke-width", 1.6);
    gBox.append("line").attr("x1", x(b.med)).attr("x2", x(b.med)).attr("y1", yb).attr("y2", yb + BOXH)
      .attr("stroke", SC.a2).attr("stroke-width", 2.6);
    // out-of-fence points on the box row
    gBox.selectAll("circle.o").data(b.out).join("circle").attr("class", "o")
      .attr("cx", d => x(d)).attr("cy", mid).attr("r", 4.2)
      .attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 1.6);
    // IQR bracket
    gBox.append("text").attr("x", (x(b.q1) + x(b.q3)) / 2).attr("y", yb - 6).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.accent).text(`IQR = ${ST.fmt(b.iqr, 1)}`);

    const names = { linear: "linear, h = (n−1)p", np1: "linear, h = (n+1)p", nearest: "nearest rank", hinge: "Tukey hinges" };
    out.innerHTML =
      `n = <b>${data.length}</b> · min <b>${ST.fmt(b.min, 1)}</b> · Q₁ <b>${ST.fmt(b.q1, 2)}</b> · ` +
      `median <b>${ST.fmt(b.med, 2)}</b> · Q₃ <b>${ST.fmt(b.q3, 2)}</b> · max <b>${ST.fmt(b.max, 1)}</b> · ` +
      `IQR <b>${ST.fmt(b.iqr, 2)}</b>  <span style="opacity:.7">[${names[conv]}]</span>` +
      `<br>fences at <b>${ST.fmt(b.loFence, 1)}</b> and <b>${ST.fmt(b.hiFence, 1)}</b> (k = ${ST.fmt(k, 1)}) · ` +
      `whiskers reach <b>${ST.fmt(b.whiskLo, 1)}</b> and <b>${ST.fmt(b.whiskHi, 1)}</b> · ` +
      `<b>${b.out.length}</b> point${b.out.length === 1 ? "" : "s"} drawn individually` +
      (offScale ? ` <span style="opacity:.7">(${offScale === 2 ? "both fences lie" : "one fence lies"} beyond the plotted range)</span>` : "");

    if (sD.value === "bimodal") {
      const s = ST.asc(data);
      const below = s.filter(v => v < b.med).pop(), above = s.filter(v => v > b.med)[0];
      out.innerHTML += `<br><span style="color:${SC.a2}">the box looks perfectly ordinary — yet the median ` +
        `${ST.fmt(b.med, 1)} sits in a gap of width ${ST.fmt(above - below, 1)} (nearest observations ` +
        `${below} and ${above}) where there is no data at all. Five numbers cannot show you two groups.</span>`;
    }
  }

  [sK].forEach(s => s.addEventListener("input", update));
  [sC, sD].forEach(s => s.addEventListener("change", update));
  update();
})();

/* ─────────────────── 6 · ECDF, DKW band, plug-in estimates ─────────────────── */
(function () {
  const svg = d3.select("#ecdf-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 380, MU = 50, SIG = 12, LO = 8, HI = 92;
  let seed = 424242, data = [], sorted = [];
  let readerX = 56;

  const F = ST.frame(svg, W, H, { l: 52, r: 22, t: 20, b: 52 });
  const x = d3.scaleLinear().domain([LO, HI]).range([0, F.iw]);
  const y = d3.scaleLinear().domain([0, 1]).range([F.ih, 0]);
  ST.gridY(F.g, y, F.iw, 5);
  ST.axisB(F.g, x, F.ih, 8, "x");
  ST.axisL(F.g, y, 5, "F(x)");

  const gBand = F.g.append("g");
  const gTrue = F.g.append("path").attr("fill", "none").attr("stroke", SC.good)
    .attr("stroke-width", 2).attr("stroke-dasharray", "6 4").attr("opacity", 0.9);
  const gStep = F.g.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
  const gRug = F.g.append("g");
  const gRead = F.g.append("g");
  const readLine = gRead.append("line").attr("y1", 0).attr("y2", F.ih)
    .attr("stroke", SC.a2).attr("stroke-width", 1.6);
  const dotFn = gRead.append("circle").attr("r", 5).attr("fill", SC.accent).attr("stroke", "#0f1117").attr("stroke-width", 1.4);
  const dotF = gRead.append("circle").attr("r", 5).attr("fill", SC.good).attr("stroke", "#0f1117").attr("stroke-width", 1.4);
  const handle = gRead.append("circle").attr("class", "dragpt").attr("r", 9).attr("cy", -1)
    .attr("fill", SC.a2).attr("stroke", "#0f1117").attr("stroke-width", 2).style("cursor", "grab");

  const out = document.getElementById("ecdf-readout");
  const sN = document.getElementById("e-n"), cB = document.getElementById("e-band"), cT = document.getElementById("e-true");

  const trueF = v => ST.normCdf((v - MU) / SIG);
  const grid = d3.range(LO, HI + 0.01, (HI - LO) / 200);
  gTrue.datum(grid.map(v => ({ x: v, y: trueF(v) })))
    .attr("d", d3.line().x(d => x(d.x)).y(d => y(d.y)));

  function resample() {
    const r = ST.rng(seed), n = +sN.value;
    data = [];
    for (let i = 0; i < n; i++) data.push(MU + SIG * ST.randn(r));
    sorted = ST.asc(data);
  }

  function stepPts() {
    const s = ST.ecdfSteps(sorted), p = [{ x: LO, F: 0 }];
    s.forEach(d => p.push({ x: Math.max(LO, Math.min(HI, d.x)), F: d.F }));
    p.push({ x: HI, F: 1 });
    return p;
  }

  function update() {
    const n = sorted.length, eps = ST.dkwEps(n, 0.05);
    document.getElementById("e-nv").textContent = n;
    const pts = stepPts();

    gStep.attr("d", d3.line().x(d => x(d.x)).y(d => y(d.F)).curve(d3.curveStepAfter)(pts));
    gTrue.attr("opacity", cT.checked ? 0.9 : 0);

    gBand.selectAll("*").remove();
    if (cB.checked) {
      gBand.append("path").datum(pts)
        .attr("fill", SC.accent).attr("fill-opacity", 0.13)
        .attr("d", d3.area()
          .x(d => x(d.x))
          .y0(d => y(Math.max(0, d.F - eps)))
          .y1(d => y(Math.min(1, d.F + eps)))
          .curve(d3.curveStepAfter));
    }

    gRug.selectAll("line").data(sorted).join("line")
      .attr("x1", d => x(d)).attr("x2", d => x(d))
      .attr("y1", F.ih).attr("y2", F.ih - 8)
      .attr("stroke", SC.muted).attr("stroke-opacity", 0.5);

    const fn = ST.ecdfAt(sorted, readerX), ft = trueF(readerX);
    readLine.attr("x1", x(readerX)).attr("x2", x(readerX));
    dotFn.attr("cx", x(readerX)).attr("cy", y(fn));
    dotF.attr("cx", x(readerX)).attr("cy", y(ft)).attr("opacity", cT.checked ? 1 : 0);
    handle.attr("cx", x(readerX));

    // worst gap between the staircase and the truth, checked either side of every jump
    let sup = 0;
    sorted.forEach((v, i) => {
      sup = Math.max(sup, Math.abs((i + 1) / n - trueF(v)), Math.abs(i / n - trueF(v)));
    });

    out.innerHTML =
      `n = <b>${n}</b> · at x = <b>${ST.fmt(readerX, 1)}</b>:  Fₙ(x) = <b>${ST.fmt(fn, 3)}</b>, ` +
      `true F(x) = <b>${ST.fmt(ft, 3)}</b>, gap <b>${ST.fmt(Math.abs(fn - ft), 3)}</b>` +
      `<br>worst gap anywhere: sup|Fₙ − F| = <b>${ST.fmt(sup, 3)}</b> · 95% DKW half-width εₙ = ` +
      `<b>${ST.fmt(eps, 3)}</b> — the band covers the whole curve ${sup <= eps ? "✓" : "✗ (this is the ~5% of samples where it fails)"}` +
      `<br>plug-in estimates: x̄ = <b>${ST.fmt(ST.mean(sorted), 2)}</b> (truth 50) · ` +
      `median = <b>${ST.fmt(ST.median(sorted), 2)}</b> (truth 50) · ` +
      `Fₙ⁻¹(0.9) = <b>${ST.fmt(ST.quantile(sorted, 0.9, "nearest"), 2)}</b> (truth ${ST.fmt(MU + SIG * 1.2816, 2)})`;
  }

  handle.call(d3.drag().on("drag", e => {
    readerX = Math.max(LO, Math.min(HI, x.invert(e.x)));
    update();
  }));
  svg.on("click", function (e) {
    const p = d3.pointer(e, F.g.node());
    if (p[1] < 0 || p[1] > F.ih) return;
    readerX = Math.max(LO, Math.min(HI, x.invert(p[0])));
    update();
  });
  sN.addEventListener("input", () => { resample(); update(); });
  [cB, cT].forEach(c => c.addEventListener("change", update));
  document.getElementById("e-new").addEventListener("click", () => { seed = (seed * 1103515245 + 12345) >>> 0; resample(); update(); });

  resample(); update();
})();

/* ─────────────────── 7 · what correlation sees ─────────────────── */
(function () {
  const svg = d3.select("#corr-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 380;
  const SHAPES = {
    linear: { seed: 11, home: [5, 6.4], make: (r, i) => { const xv = 1 + 8 * r(); return [xv, 1.0 + 0.95 * xv + 1.1 * ST.randn(r)]; } },
    curve: { seed: 22, home: [5, 2.0], make: (r, i) => { const xv = 1 + 8 * (i / 33); return [xv, 1.6 + 0.34 * (xv - 5) * (xv - 5) + 0.18 * ST.randn(r)]; } },
    lever: { seed: 33, home: [3.2, 3.2], make: (r, i) => [2.4 + 1.6 * r(), 2.4 + 1.6 * r()] },
    funnel: { seed: 44, home: [9.2, 5.5], make: (r, i) => { const xv = 1 + 8 * r(); return [xv, 5.5 + (0.15 + 0.42 * xv) * ST.randn(r)]; } },
    clusters: { seed: 55, home: [5.5, 5.5], make: (r, i) => (i < 17 ? [2 + 1.4 * r(), 2 + 1.4 * r()] : [7.2 + 1.4 * r(), 7.4 + 1.4 * r()]) }
  };
  let shape = "linear", base = [], special = [5, 6.4];

  const F = ST.frame(svg, W, H, { l: 48, r: 22, t: 20, b: 50 });
  const x = d3.scaleLinear().domain([0, 11]).range([0, F.iw]);
  const y = d3.scaleLinear().domain([0, 11]).range([F.ih, 0]);
  ST.gridY(F.g, y, F.iw, 6);
  ST.axisB(F.g, x, F.ih, 6, "x");
  ST.axisL(F.g, y, 6, "y");
  const gMeans = F.g.append("g").attr("opacity", 0.5);
  const vMean = gMeans.append("line").attr("y1", 0).attr("y2", F.ih).attr("stroke", SC.muted).attr("stroke-dasharray", "2 4");
  const hMean = gMeans.append("line").attr("x1", 0).attr("x2", F.iw).attr("stroke", SC.muted).attr("stroke-dasharray", "2 4");
  const gFit = F.g.append("line").attr("stroke", SC.a2).attr("stroke-width", 2).attr("opacity", 0.9);
  const gPts = F.g.append("g");
  const hSpecial = F.g.append("circle").attr("class", "dragpt").attr("r", 8)
    .attr("fill", SC.a2).attr("stroke", "#0f1117").attr("stroke-width", 2).style("cursor", "grab");

  const out = document.getElementById("corr-readout");
  const selShape = document.getElementById("k-shape"), cbFit = document.getElementById("k-fit");

  function load(name) {
    shape = name;
    const cfg = SHAPES[name], r = ST.rng(cfg.seed);
    base = [];
    for (let i = 0; i < 34; i++) {
      const p = cfg.make(r, i);
      base.push([Math.max(0.3, Math.min(10.7, p[0])), Math.max(0.3, Math.min(10.7, p[1]))]);
    }
    special = cfg.home.slice();
  }

  function update() {
    const px = base.map(p => p[0]).concat([special[0]]);
    const py = base.map(p => p[1]).concat([special[1]]);
    const r = ST.corr(px, py), rho = ST.spearman(px, py), fit = ST.lsLine(px, py);

    const sel = gPts.selectAll("circle").data(base);
    sel.exit().remove();
    sel.enter().append("circle").attr("r", 4.6).attr("fill", SC.accent).attr("fill-opacity", 0.8)
      .attr("stroke", "#0f1117").attr("stroke-width", 1)
      .merge(sel)
      .attr("cx", d => x(d[0])).attr("cy", d => y(d[1]));
    hSpecial.attr("cx", x(special[0])).attr("cy", y(special[1]));

    vMean.attr("x1", x(ST.mean(px))).attr("x2", x(ST.mean(px)));
    hMean.attr("y1", y(ST.mean(py))).attr("y2", y(ST.mean(py)));

    gFit.attr("opacity", cbFit.checked ? 0.9 : 0)
      .attr("x1", x(0)).attr("y1", y(fit.a))
      .attr("x2", x(11)).attr("y2", y(fit.a + fit.b * 11));

    const ang = Math.acos(Math.max(-1, Math.min(1, r))) * 180 / Math.PI;
    const notes = {
      linear: "a genuine linear relation — r earns its keep here, and one point cannot move it much",
      curve: "an almost deterministic curve. r stays near zero wherever you put the point — and so does Spearman's ρ, because the relationship is not monotone either. Only the picture shows it",
      lever: "the cloud on its own has almost no correlation. Drag the orange point up to the top-right corner and watch r climb past 0.85 out of nothing — that is leverage. Notice that ρ barely moves: rank correlation resists it",
      funnel: "the spread of y grows with x while the trend stays flat. r reports one number and says nothing about the fanning, which is what breaks every assumption downstream",
      clusters: "two separate groups. r is measuring the gap between the clusters, not any relationship inside either of them — and ρ disagrees, which is the tell"
    };
    out.innerHTML =
      `n = <b>${px.length}</b> · Pearson r = <b>${ST.fmt(r, 3)}</b> (angle between the centred columns = <b>${ST.fmt(ang, 1)}°</b>) · ` +
      `Spearman ρ = <b>${ST.fmt(rho, 3)}</b> · r² = <b>${ST.fmt(r * r, 3)}</b>` +
      `<br>cov = <b>${ST.fmt(ST.cov(px, py), 3)}</b> · sₓ = <b>${ST.fmt(ST.sd(px), 3)}</b> · s_y = <b>${ST.fmt(ST.sd(py), 3)}</b> · ` +
      `slope b = r·s_y/sₓ = <b>${ST.fmt(fit.b, 3)}</b>` +
      `<br><span style="color:${SC.a2}">${notes[shape]}</span>`;
  }

  hSpecial.call(d3.drag().on("drag", e => {
    special = [Math.max(0.3, Math.min(10.7, x.invert(e.x))), Math.max(0.3, Math.min(10.7, y.invert(e.y)))];
    update();
  }));
  selShape.addEventListener("change", () => { load(selShape.value); update(); });
  cbFit.addEventListener("change", update);
  document.getElementById("k-reset").addEventListener("click", () => { special = SHAPES[shape].home.slice(); update(); });

  load("linear"); update();
})();
