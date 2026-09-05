/* dimensionality-reduction-pca.viz.js — extracted from dimensionality-reduction-pca.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* Interactive D3 PCA Widget */
(function(){
  const svg = d3.select("#pca-svg"), W = 640, H = 380, m = { t: 30, r: 30, b: 30, l: 30 };
  const x = d3.scaleLinear().domain([-5, 5]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([-5, 5]).range([H - m.b, m.t]);

  // Center of the coordinate system in pixel space
  const cx = x(0);
  const cy = y(0);

  const SEED = [
    {x: -3.2, y: -2.1}, {x: -2.8, y: -1.9}, {x: -2.4, y: -1.2}, {x: -2.1, y: -2.0},
    {x: -1.9, y: -0.8}, {x: -1.5, y: -1.1}, {x: -1.2, y: -0.4}, {x: -1.0, y: -1.2},
    {x: -0.8, y: -0.2}, {x: -0.5, y: -0.6}, {x: -0.3, y: 0.1},  {x: -0.1, y: -0.5},
    {x: 0.1, y: 0.5},   {x: 0.3, y: -0.2},  {x: 0.5, y: 0.8},   {x: 0.8, y: 0.2},
    {x: 1.1, y: 1.2},   {x: 1.4, y: 0.4},   {x: 1.6, y: 1.5},   {x: 1.9, y: 0.9},
    {x: 2.2, y: 1.9},   {x: 2.5, y: 1.3},   {x: 2.9, y: 2.4},   {x: 3.3, y: 2.1},
    // some outliers
    {x: -1.8, y: 0.9},  {x: 1.5, y: -1.2},  {x: -0.5, y: -1.8}, {x: 2.5, y: -0.2},
    {x: -2.5, y: 0.5},  {x: 0.8, y: 1.9},   {x: -3.0, y: -1.0}, {x: 3.0, y: 0.8},
    {x: -1.0, y: -2.2}, {x: 1.0, y: 2.4},   {x: -0.2, y: 1.2},  {x: 0.2, y: -1.4},
    {x: -3.5, y: -2.8}, {x: 3.5, y: 2.9},   {x: -2.0, y: -2.5}, {x: 2.0, y: 2.7}
  ];
  let pts = SEED.map(p => ({ x: p.x, y: p.y }));

  // Draw grid axes
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${cy})`).call(d3.axisBottom(x).ticks(8).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${cx},0)`).call(d3.axisLeft(y).ticks(8).tickFormat(""));

  // Container groups
  const gProjLines = svg.append("g");
  const gLine = svg.append("g");
  const gPts = svg.append("g");
  const gProjPts = svg.append("g");
  const gPCArrows = svg.append("g");

  // Add arrowhead definitions
  const defs = svg.append("defs");
  defs.append("marker")
    .attr("id", "arrow-pc1").attr("viewBox", "0 0 10 10")
    .attr("refX", 5).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", C.A);
  defs.append("marker")
    .attr("id", "arrow-pc2").attr("viewBox", "0 0 10 10")
    .attr("refX", 5).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", C.B);

  // Main projection line
  const projLine = gLine.append("line")
    .attr("stroke", "#4b5563")
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "4 4");

  const readout = d3.select("#pca-readout");
  const clamp = v => Math.max(-4.8, Math.min(4.8, v));

  // Analytical eigendecomposition of the 2x2 sample covariance matrix
  function calculatePCA() {
    const n = pts.length;
    let mx = 0, my = 0;
    pts.forEach(p => { mx += p.x; my += p.y; });
    mx /= n; my /= n;

    let covXX = 0, covXY = 0, covYY = 0;
    pts.forEach(p => {
      const dx = p.x - mx, dy = p.y - my;
      covXX += dx * dx; covXY += dx * dy; covYY += dy * dy;
    });
    covXX /= (n - 1); covXY /= (n - 1); covYY /= (n - 1);

    // eigenvalues of a symmetric 2x2 from its trace and determinant
    const tr = covXX + covYY;
    const det = covXX * covYY - covXY * covXY;
    const diff = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const L1 = tr / 2 + diff;
    const L2 = tr / 2 - diff;

    // leading eigenvector
    let vx = 1, vy = 0;
    if (covXY !== 0) {
      vx = L1 - covYY;
      vy = covXY;
      const mag = Math.hypot(vx, vy);
      vx /= mag; vy /= mag;
    } else if (covYY > covXX) { vx = 0; vy = 1; }

    let angle = Math.atan2(vy, vx) * 180 / Math.PI;
    if (angle < 0) angle += 180;

    return { angle, L1, L2, ev1: [vx, vy], ev2: [-vy, vx], mean: [mx, my], sdX: Math.sqrt(covXX), sdY: Math.sqrt(covYY) };
  }

  function draw() {
    const angleDeg = parseFloat(d3.select("#pca-angle").property("value"));
    const rad = angleDeg * Math.PI / 180;
    const ux = Math.cos(rad), uy = Math.sin(rad);

    const n = pts.length;
    let mx = 0, my = 0;
    pts.forEach(p => { mx += p.x; my += p.y; });
    mx /= n; my /= n;

    // the candidate line, drawn through the centre of mass
    const len = 7;
    projLine
      .attr("x1", x(mx - len * ux)).attr("y1", y(my - len * uy))
      .attr("x2", x(mx + len * ux)).attr("y2", y(my + len * uy));

    // project every point onto it
    const projected = pts.map(p => {
      const dx = p.x - mx, dy = p.y - my;
      const dot = dx * ux + dy * uy;              // the score z = uᵀ(x − μ)
      return { orig: p, px: mx + dot * ux, py: my + dot * uy, val: dot,
               res: (dx - dot * ux) ** 2 + (dy - dot * uy) ** 2 };
    });

    let projVar = 0, sse = 0;
    projected.forEach(p => { projVar += p.val * p.val; sse += p.res; });
    projVar /= (n - 1);
    const mse = sse / n;                           // mean squared reconstruction error at k = 1

    // residual drop-lines
    const plines = gProjLines.selectAll("line").data(projected);
    plines.enter().append("line")
      .attr("stroke", "#374151").attr("stroke-width", 1).attr("stroke-dasharray", "2 2")
      .merge(plines)
      .attr("x1", d => x(d.orig.x)).attr("y1", d => y(d.orig.y))
      .attr("x2", d => x(d.px)).attr("y2", d => y(d.py));
    plines.exit().remove();

    // original points
    const circles = gPts.selectAll("circle").data(pts);
    circles.enter().append("circle").attr("r", 5.5).attr("fill", C.B).attr("stroke", "#0f1117").attr("stroke-width", 1.5).attr("cursor", "grab")
      .merge(circles)
      .attr("cx", d => x(d.x)).attr("cy", d => y(d.y));
    circles.exit().remove();

    gPts.selectAll("circle").call(d3.drag()
      .on("drag", function(ev, d) {
        d.x = clamp(x.invert(ev.x));
        d.y = clamp(y.invert(ev.y));
        draw();
      }));

    // shadows on the line
    const pcircles = gProjPts.selectAll("circle").data(projected);
    pcircles.enter().append("circle").attr("r", 4.5).attr("fill", "#9ca3af").attr("stroke", "#0f1117").attr("stroke-width", 1).style("pointer-events", "none")
      .merge(pcircles)
      .attr("cx", d => x(d.px)).attr("cy", d => y(d.py));
    pcircles.exit().remove();

    const pcaData = calculatePCA();
    const totalVar = pcaData.L1 + pcaData.L2;
    const currentPct = (projVar / totalVar * 100).toFixed(1);

    readout.html(`
      line angle <b>${angleDeg.toFixed(0)}°</b> · projected variance <b style="color:${C.A}">${projVar.toFixed(3)}</b> (${currentPct}% of total)
      · reconstruction error <b style="color:${C.bad}">${mse.toFixed(3)}</b><br>
      total variance <b>${totalVar.toFixed(3)}</b> = λ₁ <b>${pcaData.L1.toFixed(3)}</b> + λ₂ <b>${pcaData.L2.toFixed(3)}</b>
      · best angle <b style="color:${C.good}">${pcaData.angle.toFixed(0)}°</b> giving EVR₁ = <b>${(pcaData.L1 / totalVar * 100).toFixed(1)}%</b><br>
      feature sd: x <b>${pcaData.sdX.toFixed(2)}</b>, y <b>${pcaData.sdY.toFixed(2)}</b>
      <span style="color:${C.muted}">— when these differ, PC1 chases the larger one</span>
    `);
  }

  // Click empty space to add a point
  svg.on("click", function(ev) {
    if (ev.target.tagName === "circle" || ev.target.tagName === "line") return;
    const [mxp, myp] = d3.pointer(ev);
    if (pts.length >= 60) return;
    pts.push({ x: clamp(x.invert(mxp)), y: clamp(y.invert(myp)) });
    draw();
  });

  // Double click a point to remove it
  svg.on("dblclick", function(ev) {
    if (ev.target.tagName !== "circle") return;
    const nodeData = d3.select(ev.target).datum();
    if (pts.includes(nodeData)) {
      pts = pts.filter(p => p !== nodeData);
      draw();
    }
  });

  // Draw the fitted principal axes
  function drawFit() {
    const res = calculatePCA();
    d3.select("#pca-angle").property("value", res.angle);
    draw();

    gPCArrows.selectAll("*").remove();
    const scale2 = 2 * Math.sqrt(res.L2 / Math.max(res.L1, 1e-9));   // PC2 arrow length ∝ √λ₂/√λ₁

    gPCArrows.append("line")
      .attr("x1", x(res.mean[0])).attr("y1", y(res.mean[1]))
      .attr("x2", x(res.mean[0] + res.ev1[0] * 2)).attr("y2", y(res.mean[1] + res.ev1[1] * 2))
      .attr("stroke", C.A).attr("stroke-width", 3).attr("marker-end", "url(#arrow-pc1)");

    gPCArrows.append("line")
      .attr("x1", x(res.mean[0])).attr("y1", y(res.mean[1]))
      .attr("x2", x(res.mean[0] + res.ev2[0] * scale2)).attr("y2", y(res.mean[1] + res.ev2[1] * scale2))
      .attr("stroke", C.B).attr("stroke-width", 2.5).attr("marker-end", "url(#arrow-pc2)");

    gPCArrows.append("text")
      .attr("x", x(res.mean[0] + res.ev1[0] * 2.3)).attr("y", y(res.mean[1] + res.ev1[1] * 2.3))
      .attr("fill", C.A).attr("font-size", 11).attr("font-weight", "600").attr("text-anchor", "middle")
      .text("PC1");

    gPCArrows.append("text")
      .attr("x", x(res.mean[0] + res.ev2[0] * (scale2 + 0.4))).attr("y", y(res.mean[1] + res.ev2[1] * (scale2 + 0.4)))
      .attr("fill", C.B).attr("font-size", 11).attr("font-weight", "600").attr("text-anchor", "middle")
      .text("PC2");
  }

  d3.select("#pca-fit").on("click", drawFit);

  // Change the units of feature 1: variance along x grows, PC1 swings toward the x-axis
  d3.select("#pca-stretch").on("click", () => {
    const mx = d3.mean(pts, p => p.x);
    pts.forEach(p => { p.x = clamp(mx + (p.x - mx) * 1.4); });
    drawFit();
  });

  // Standardize both features to unit variance — the fix for the scaling trap
  d3.select("#pca-standardize").on("click", () => {
    const mx = d3.mean(pts, p => p.x), my = d3.mean(pts, p => p.y);
    const sx = d3.deviation(pts, p => p.x) || 1, sy = d3.deviation(pts, p => p.y) || 1;
    pts.forEach(p => { p.x = clamp((p.x - mx) / sx * 1.5); p.y = clamp((p.y - my) / sy * 1.5); });
    drawFit();
  });

  d3.select("#pca-reset").on("click", () => {
    pts = SEED.map(p => ({ x: p.x, y: p.y }));
    gPCArrows.selectAll("*").remove();
    draw();
  });

  d3.select("#pca-angle").on("input", () => {
    gPCArrows.selectAll("*").remove();   // drop the fitted axes when the reader takes over
    draw();
  });

  draw();
})();

/* PCA axis vs LDA axis, with both 1-D projections underneath */
(function(){
  const svg = d3.select("#ld-svg");
  if (svg.empty()) return;
  const W = 660, H = 600;

  // ---- scatter panel: 440 x 340 px at exactly 40 px per data unit (equal aspect) ----
  const SL = 34, ST = 22, SW = 440, SH = 340, UPX = 40;
  const x = d3.scaleLinear().domain([-SW / (2 * UPX), SW / (2 * UPX)]).range([SL, SL + SW]);
  const y = d3.scaleLinear().domain([-SH / (2 * UPX), SH / (2 * UPX)]).range([ST + SH, ST]);

  // ---- histogram panels ----
  const HL = 34, HR = 646, NBINS = 41, BARMAX = 56;
  const PANEL = [ { title: 386, base: 462, thr: 400 }, { title: 484, base: 560, thr: 498 } ];
  const hx = d3.scaleLinear().range([HL, HR]);

  const MEAN_DIR = 135 * Math.PI / 180;   // the class-offset direction is fixed
  const NPC = 90;                          // points per class
  const SMIN = 0.42;                       // within-class sd along the minor axis
  const DEF = { sep: 3, tilt: 45, elong: 5 };
  let seed = 20260904;

  // deterministic PRNG so the picture is reproducible
  function mulberry32(a){ return function(){ a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

  function sample(sd, tiltDeg, elong, sep){
    const rnd = mulberry32(sd);
    const gauss = () => { let u = 0, v = 0;
      while (u === 0) u = rnd(); while (v === 0) v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const th = tiltDeg * Math.PI / 180, ca = Math.cos(th), sa = Math.sin(th);
    const sMaj = SMIN * elong;
    const dx = Math.cos(MEAN_DIR) * sep / 2, dy = Math.sin(MEAN_DIR) * sep / 2;
    const pts = [];
    for (let c = 0; c < 2; c++){
      const s = c === 0 ? 1 : -1;
      for (let i = 0; i < NPC; i++){
        const a = gauss() * sMaj, b = gauss() * SMIN;
        pts.push({ c: c, x: ca * a - sa * b + s * dx, y: sa * a + ca * b + s * dy });
      }
    }
    return pts;
  }

  // eigen-pair of a symmetric 2x2 [[a,b],[b,c]] — same closed form as the PCA widget above
  function eig2(a, b, c){
    const tr = a + c, det = a * c - b * b;
    const g = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + g, l2 = tr / 2 - g;
    let vx = 1, vy = 0;
    if (Math.abs(b) > 1e-12){ vx = l1 - c; vy = b; const m = Math.hypot(vx, vy); vx /= m; vy /= m; }
    else if (c > a){ vx = 0; vy = 1; }
    return { l1: l1, l2: l2, v: [vx, vy] };
  }

  function analyse(pts){
    const n = [0, 0], mu = [[0, 0], [0, 0]];
    pts.forEach(p => { mu[p.c][0] += p.x; mu[p.c][1] += p.y; n[p.c]++; });
    for (let c = 0; c < 2; c++){ mu[c][0] /= n[c]; mu[c][1] /= n[c]; }
    const N = n[0] + n[1];
    const gm = [(mu[0][0] * n[0] + mu[1][0] * n[1]) / N, (mu[0][1] * n[0] + mu[1][1] * n[1]) / N];

    // pooled within-class scatter, and total scatter about the grand mean
    let wxx = 0, wxy = 0, wyy = 0, txx = 0, txy = 0, tyy = 0;
    pts.forEach(p => {
      const ax = p.x - mu[p.c][0], ay = p.y - mu[p.c][1];
      wxx += ax * ax; wxy += ax * ay; wyy += ay * ay;
      const bx = p.x - gm[0], by = p.y - gm[1];
      txx += bx * bx; txy += bx * by; tyy += by * by;
    });
    const dw = N - 2, dt = N - 1;
    const Wc = [wxx / dw, wxy / dw, wyy / dw];        // pooled within-class covariance
    const Tc = [txx / dt, txy / dt, tyy / dt];        // total covariance

    const pcaE = eig2(Tc[0], Tc[1], Tc[2]);
    const D = [mu[0][0] - mu[1][0], mu[0][1] - mu[1][1]];

    // LDA direction: w ∝ Sw⁻¹Δ, with a fallback to the plain centroid difference
    const det = Wc[0] * Wc[2] - Wc[1] * Wc[1];
    let w;
    if (Math.abs(det) > 1e-10){
      w = [( Wc[2] * D[0] - Wc[1] * D[1]) / det, (-Wc[1] * D[0] + Wc[0] * D[1]) / det];
    } else { w = D.slice(); }
    let wn = Math.hypot(w[0], w[1]);
    if (wn < 1e-9){ w = pcaE.v.slice(); wn = 1; }     // Δ ≈ 0: nothing to find
    w = [w[0] / wn, w[1] / wn];

    const qform = u => u[0] * u[0] * Wc[0] + 2 * u[0] * u[1] * Wc[1] + u[1] * u[1] * Wc[2];
    const tform = u => u[0] * u[0] * Tc[0] + 2 * u[0] * u[1] * Tc[1] + u[1] * u[1] * Tc[2];
    const fisher = u => { const num = Math.pow(u[0] * D[0] + u[1] * D[1], 2); const den = qform(u);
                          return den > 1e-12 ? num / den : 0; };

    return { pts: pts, mu: mu, gm: gm, Wc: Wc, Tc: Tc, D: D, N: N,
             pcaDir: pcaE.v, l1: pcaE.l1, l2: pcaE.l2,
             ldaDir: w, fisher: fisher, tform: tform, totVar: pcaE.l1 + pcaE.l2 };
  }

  // project onto a unit direction, centred at the grand mean
  function project(S, u){
    const t = S.pts.map(p => ({ c: p.c, t: (p.x - S.gm[0]) * u[0] + (p.y - S.gm[1]) * u[1] }));
    const m = [0, 0], n = [0, 0];
    t.forEach(o => { m[o.c] += o.t; n[o.c]++; });
    m[0] /= n[0]; m[1] /= n[1];
    const thr = (m[0] + m[1]) / 2, hi = m[0] >= m[1] ? 0 : 1;
    let wrong = 0;
    t.forEach(o => { const pred = (o.t >= thr) ? hi : 1 - hi; if (pred !== o.c) wrong++; });
    return { t: t, thr: thr, err: wrong / t.length, mA: m[0], mB: m[1] };
  }

  // ---- static chrome ----
  const defs = svg.append("defs");
  defs.append("clipPath").attr("id", "ld-clip").append("rect")
    .attr("x", SL).attr("y", ST).attr("width", SW).attr("height", SH);
  ["ld-pca", "ld-lda"].forEach((id, i) => {
    defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10")
      .attr("refX", 8).attr("refY", 5).attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", i ? C.good : C.bad);
  });

  svg.append("rect").attr("x", SL).attr("y", ST).attr("width", SW).attr("height", SH)
     .attr("fill", "none").attr("stroke", C.line);
  const gClip = svg.append("g").attr("clip-path", "url(#ld-clip)");
  const gEll  = gClip.append("g");
  const gPts  = gClip.append("g");
  const gAx   = gClip.append("g");
  const gLab  = svg.append("g");
  const gHist = svg.append("g");

  // legend column to the right of the scatter
  const leg = svg.append("g").attr("font-size", 11).attr("fill", C.muted);
  const legRow = (dy, colour, text, isLine, dash) => {
    const g = leg.append("g").attr("transform", "translate(492," + dy + ")");
    if (isLine) g.append("line").attr("x1", 0).attr("y1", -4).attr("x2", 20).attr("y2", -4)
      .attr("stroke", colour).attr("stroke-width", 2.5).attr("stroke-dasharray", dash || null);
    else g.append("circle").attr("cx", 6).attr("cy", -4).attr("r", 5).attr("fill", colour);
    g.append("text").attr("x", 27).attr("y", 0).attr("fill", C.ink).text(text);
  };
  leg.append("text").attr("x", 492).attr("y", 34).attr("fill", C.muted)
     .attr("font-size", 10.5).attr("letter-spacing", 1).text("LEGEND");
  legRow(58,  C.A, "class A", false);
  legRow(80,  C.B, "class B", false);
  legRow(108, C.bad,  "PC1 — max variance", true, "7 4");
  legRow(130, C.good, "LDA — max separation", true, null);
  legRow(158, C.muted, "1σ within-class", true, "2 3");
  leg.append("text").attr("x", 492).attr("y", 190).attr("fill", C.muted).attr("font-size", 10.5)
     .text("crosses = class means,");
  leg.append("text").attr("x", 492).attr("y", 205).attr("fill", C.muted).attr("font-size", 10.5)
     .text("both axes pass through");
  leg.append("text").attr("x", 492).attr("y", 220).attr("fill", C.muted).attr("font-size", 10.5)
     .text("the grand mean.");
  leg.append("text").attr("x", 492).attr("y", 254).attr("fill", C.muted).attr("font-size", 10.5)
     .text("Histograms below share");
  leg.append("text").attr("x", 492).attr("y", 269).attr("fill", C.muted).attr("font-size", 10.5)
     .text("one horizontal scale, so");
  leg.append("text").attr("x", 492).attr("y", 284).attr("fill", C.muted).attr("font-size", 10.5)
     .text("width is comparable.");

  const readout = d3.select("#ld-readout");
  const val = id => parseFloat(d3.select(id).property("value"));

  function drawHist(panel, proj, tmax, label, colour){
    const g = gHist.append("g");
    hx.domain([-tmax, tmax]);
    const bw = 2 * tmax / NBINS;
    const counts = [[], []];
    for (let c = 0; c < 2; c++){ counts[c] = new Array(NBINS).fill(0); }
    proj.t.forEach(o => {
      let k = Math.floor((o.t + tmax) / bw);
      if (k < 0) k = 0; if (k >= NBINS) k = NBINS - 1;
      counts[o.c][k]++;
    });
    const peak = Math.max(1, d3.max(counts[0].concat(counts[1])));
    const pxw = (HR - HL) / NBINS;

    g.append("line").attr("x1", HL).attr("y1", panel.base).attr("x2", HR).attr("y2", panel.base)
      .attr("stroke", "#3a4150");
    g.append("text").attr("x", HL).attr("y", panel.title).attr("fill", colour)
      .attr("font-size", 12).attr("font-weight", 600).text(label);

    for (let c = 0; c < 2; c++){
      g.selectAll("rect.h" + c).data(counts[c]).enter().append("rect")
        .attr("x", (d, i) => HL + i * pxw + 0.5)
        .attr("width", Math.max(1, pxw - 1))
        .attr("y", d => panel.base - d / peak * BARMAX)
        .attr("height", d => d / peak * BARMAX)
        .attr("fill", c === 0 ? C.A : C.B).attr("fill-opacity", 0.58);
    }
    // decision threshold on this projection
    g.append("line").attr("x1", hx(proj.thr)).attr("x2", hx(proj.thr))
      .attr("y1", panel.thr - 8).attr("y2", panel.base + 5)
      .attr("stroke", colour).attr("stroke-width", 1.5).attr("stroke-dasharray", "3 3");
    g.append("text").attr("x", hx(proj.thr) + 5).attr("y", panel.thr)
      .attr("fill", colour).attr("font-size", 10).text("threshold");
    // projected class means
    [proj.mA, proj.mB].forEach((mv, i) => {
      g.append("path").attr("d", "M" + hx(mv) + "," + (panel.base + 3) + " l-4,8 l8,0 z")
        .attr("fill", i === 0 ? C.A : C.B);
    });
    g.append("text").attr("x", HR).attr("y", panel.title).attr("text-anchor", "end")
      .attr("fill", C.muted).attr("font-size", 11)
      .text("1-D error at the threshold: " + (proj.err * 100).toFixed(1) + "%");
  }

  function draw(){
    const sep = val("#ld-sep"), tilt = val("#ld-tilt"), elong = val("#ld-elong");
    const S = analyse(sample(seed, tilt, elong, sep));

    // --- 1σ within-class ellipses ---
    const we = eig2(S.Wc[0], S.Wc[1], S.Wc[2]);
    const wang = -Math.atan2(we.v[1], we.v[0]) * 180 / Math.PI;   // SVG y points down
    gEll.selectAll("*").remove();
    [0, 1].forEach(c => {
      gEll.append("ellipse")
        .attr("rx", Math.sqrt(Math.max(we.l1, 1e-9)) * UPX)
        .attr("ry", Math.sqrt(Math.max(we.l2, 1e-9)) * UPX)
        .attr("transform", "translate(" + x(S.mu[c][0]) + "," + y(S.mu[c][1]) + ") rotate(" + wang + ")")
        .attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 1)
        .attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.85);
    });

    // --- points ---
    const sel = gPts.selectAll("circle").data(S.pts);
    sel.enter().append("circle").attr("r", 3.1).attr("stroke", "#0f1117").attr("stroke-width", 0.6)
      .merge(sel)
      .attr("cx", d => x(d.x)).attr("cy", d => y(d.y))
      .attr("fill", d => d.c === 0 ? C.A : C.B).attr("fill-opacity", 0.85);
    sel.exit().remove();

    // --- axes through the grand mean, plus class-mean crosses ---
    gAx.selectAll("*").remove();
    const ray = (u, colour, dash, marker, name) => {
      const L = 5.4;
      gAx.append("line")
        .attr("x1", x(S.gm[0] - L * u[0])).attr("y1", y(S.gm[1] - L * u[1]))
        .attr("x2", x(S.gm[0] + L * u[0])).attr("y2", y(S.gm[1] + L * u[1]))
        .attr("stroke", colour).attr("stroke-width", 2.4)
        .attr("stroke-dasharray", dash || null).attr("marker-end", "url(#" + marker + ")");
      gAx.append("text")
        .attr("x", x(S.gm[0] + 3.1 * u[0])).attr("y", y(S.gm[1] + 3.1 * u[1]) - 7)
        .attr("fill", colour).attr("font-size", 11.5).attr("font-weight", 700)
        .attr("text-anchor", "middle").attr("paint-order", "stroke")
        .attr("stroke", "#0f1117").attr("stroke-width", 3).text(name);
    };
    // orient each axis so its arrow points into the upper half-plane (cosmetic only)
    const orient = u => (u[1] < 0 || (u[1] === 0 && u[0] < 0)) ? [-u[0], -u[1]] : u;
    ray(orient(S.pcaDir), C.bad,  "7 4", "ld-pca", "PC1");
    ray(orient(S.ldaDir), C.good, null,  "ld-lda", "LDA");
    [0, 1].forEach(c => {
      const px = x(S.mu[c][0]), py = y(S.mu[c][1]), col = c === 0 ? C.A : C.B;
      gAx.append("path").attr("d", "M" + (px - 7) + "," + py + " H" + (px + 7) +
                                   " M" + px + "," + (py - 7) + " V" + (py + 7))
        .attr("stroke", col).attr("stroke-width", 2.4).attr("stroke-linecap", "round")
        .attr("paint-order", "stroke");
    });

    // --- projections + histograms on one shared scale ---
    const pPCA = project(S, S.pcaDir), pLDA = project(S, S.ldaDir);
    const tmax = Math.max(
      d3.max(pPCA.t, o => Math.abs(o.t)) || 1,
      d3.max(pLDA.t, o => Math.abs(o.t)) || 1) * 1.04;

    gHist.selectAll("*").remove();
    drawHist(PANEL[0], pPCA, tmax, "Projection onto PC1  (unsupervised: maximum variance)", C.bad);
    drawHist(PANEL[1], pLDA, tmax, "Projection onto the LDA axis  (supervised: maximum separation)", C.good);

    // --- numbers ---
    const ang = u => { let a = Math.atan2(u[1], u[0]) * 180 / Math.PI; a = ((a % 180) + 180) % 180; return a; };
    const evrP = S.tform(S.pcaDir) / S.totVar * 100;
    const evrL = S.tform(S.ldaDir) / S.totVar * 100;
    const jP = S.fisher(S.pcaDir), jL = S.fisher(S.ldaDir);
    const between = Math.hypot(S.D[0], S.D[1]);
    let gap = Math.abs(ang(S.pcaDir) - ang(S.ldaDir)); if (gap > 90) gap = 180 - gap;

    readout.html(
      "Δ = <b>" + between.toFixed(2) + "</b> · within-class tilt θ = <b>" + tilt.toFixed(0) +
      "°</b> · elongation <b>" + elong.toFixed(1) + "×</b> · angle between the two axes <b>" +
      gap.toFixed(0) + "°</b><br>" +
      "<b style=\"color:" + C.bad + "\">PC1</b> at <b>" + ang(S.pcaDir).toFixed(0) +
      "°</b> — carries <b>" + evrP.toFixed(0) + "%</b> of total variance, Fisher J = <b>" +
      jP.toFixed(2) + "</b>, 1-D error <b>" + (pPCA.err * 100).toFixed(1) + "%</b><br>" +
      "<b style=\"color:" + C.good + "\">LDA</b> at <b>" + ang(S.ldaDir).toFixed(0) +
      "°</b> — carries <b>" + evrL.toFixed(0) + "%</b> of total variance, Fisher J = <b>" +
      jL.toFixed(2) + "</b>, 1-D error <b>" + (pLDA.err * 100).toFixed(1) + "%</b>" +
      "<span style=\"color:" + C.muted + "\"> — J = Δᵀ Σ_W⁻¹ Δ at the optimum</span>"
    );
  }

  ["#ld-sep", "#ld-tilt", "#ld-elong"].forEach(id => d3.select(id).on("input", draw));
  d3.select("#ld-align").on("click",    () => { d3.select("#ld-tilt").property("value", 135); draw(); });
  d3.select("#ld-orth").on("click",     () => { d3.select("#ld-tilt").property("value", 45);  draw(); });
  d3.select("#ld-resample").on("click", () => { seed = (seed * 1103515245 + 12345) >>> 0; draw(); });
  d3.select("#ld-reset").on("click",    () => {
    seed = 20260904;
    d3.select("#ld-sep").property("value", DEF.sep);
    d3.select("#ld-tilt").property("value", DEF.tilt);
    d3.select("#ld-elong").property("value", DEF.elong);
    draw();
  });

  draw();
})();
