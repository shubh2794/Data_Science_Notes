/* regression.viz.js — the ten visualizations on math/statistics/regression.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #scat-svg   set the correlation, watch the cloud, and see r as the mean of
                    the SIGNED z-score products in four shaded quadrants; a side
                    panel re-centres/re-scales the same points and r does not move
     2  #trap-svg   six clouds whose Pearson r is a bad summary, with the fitted
                    line and (optionally) Spearman's rho printed on each
     3  #lie-svg    measurement error / range restriction / aggregation, each with
                    its exact theoretical curve drawn behind the simulated point
     4  #ls-svg     DRAG the line (translate handle + rotate handle) on the twelve
                    running points; the squares are literal, and an inset traces
                    SSE as a function of the slope
     5  #rtm-svg    test–retest: a draggable midterm band whose two group means
                    demonstrate z_final = r · z_midterm, plus a graph of averages
     6  #two-svg    y-on-x, x-on-y and the SD line on one cloud; the two slopes
                    multiply to r², and the round trip does not come back
     7  #slope-svg  two thousand data sets from one line: the histogram of b̂
                    against Normal(β₁, σ²/S_xx), or of t against t(n−2)
     8  #band-svg   the confidence and prediction bands, with a LIVE coverage
                    check that counts true means and new observations separately
     9  #resid-svg  a broken relationship, its residual plot, and the transform
                    that flattens it — with curvature and fanning scored
    10  #lev-svg    one DRAGGABLE point against a fixed cloud: the solid line is
                    the fit with it, the dashed line the fit without it

   Every number these draw is recomputed from the data they generate, so the
   pictures and the prose cannot drift apart.                                   */

/* ══════════ page-local helpers (deliberately NOT in stats-viz.js) ══════════ */
const RG = (function () {

  /* Full simple-linear fit. ST.lsLine gives a, b, r; this adds everything the
     inference sections need, and computes b from the sums rather than from r so
     that a degenerate s_y cannot produce a NaN slope.                          */
  function fit(x, y) {
    const n = x.length;
    const xb = ST.mean(x), yb = ST.mean(y);
    let Sxx = 0, Syy = 0, Sxy = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - xb, dy = y[i] - yb;
      Sxx += dx * dx; Syy += dy * dy; Sxy += dx * dy;
    }
    const b = Sxx > 0 ? Sxy / Sxx : 0;
    const a = yb - b * xb;
    const ssr = b * b * Sxx, sse = Math.max(0, Syy - ssr);
    return {
      n: n, xb: xb, yb: yb, Sxx: Sxx, Syy: Syy, Sxy: Sxy, a: a, b: b,
      r: (Sxx > 0 && Syy > 0) ? Sxy / Math.sqrt(Sxx * Syy) : NaN,
      sst: Syy, ssr: ssr, sse: sse,
      se: n > 2 ? Math.sqrt(sse / (n - 2)) : NaN,
      seb: (n > 2 && Sxx > 0) ? Math.sqrt(sse / (n - 2)) / Math.sqrt(Sxx) : NaN,
      r2: Syy > 0 ? ssr / Syy : NaN,
      sx: n > 1 ? Math.sqrt(Sxx / (n - 1)) : NaN,
      sy: n > 1 ? Math.sqrt(Syy / (n - 1)) : NaN,
      pred: function (v) { return this.a + this.b * v; }
    };
  }

  /* leverage h_i = 1/n + (x_i − x̄)²/S_xx */
  function lev(x) {
    const n = x.length, xb = ST.mean(x);
    let Sxx = 0;
    for (let i = 0; i < n; i++) Sxx += (x[i] - xb) * (x[i] - xb);
    return x.map(v => 1 / n + (v - xb) * (v - xb) / Sxx);
  }

  /* one draw from a standard bivariate normal with correlation rho */
  function bvn(rho, r) {
    const z1 = ST.randn(r), z2 = ST.randn(r);
    return [z1, rho * z1 + Math.sqrt(Math.max(0, 1 - rho * rho)) * z2];
  }
  /* n such draws, returned as two parallel arrays */
  function bvnSample(n, rho, r) {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) { const p = bvn(rho, r); X.push(p[0]); Y.push(p[1]); }
    return { x: X, y: Y };
  }

  /* the running example, used by figures 4 and 8 */
  const RUNX = [2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
  const RUNY = [10, 14, 26, 22, 26, 37, 31, 48, 52, 57, 53, 56];

  /* the fixed cloud figure 10 drags a thirteenth point against */
  const LEVX = [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9];
  const LEVY = [7, 7, 10, 11, 11, 14, 15, 17, 15, 19, 20, 21];

  /* two-sided p for a t statistic */
  const tp2 = (t, df) => 2 * (1 - ST.tCdf(Math.abs(t), df));

  /* a compact p-value string */
  function pstr(p) {
    if (!isFinite(p)) return "—";
    return p < 1e-4 ? p.toExponential(2) : ST.fmt(p, 4);
  }

  /* draw a scatter of points into an existing <g> */
  function pts(g, x, y, sx, sy, opt) {
    const o = Object.assign({ r: 3, fill: SC.accent, op: 0.62, stroke: null }, opt || {});
    const sel = g.append("g").selectAll("circle").data(x.map((v, i) => [v, y[i]])).join("circle")
      .attr("cx", d => sx(d[0])).attr("cy", d => sy(d[1])).attr("r", o.r)
      .attr("fill", o.fill).attr("fill-opacity", o.op);
    if (o.stroke) sel.attr("stroke", o.stroke).attr("stroke-width", 0.8);
    return sel;
  }

  /* A straight line y = a + b·x clipped to the plotted RECTANGLE — both the x-domain
     and the y-domain. Clipping in y matters: a steep line (the x-on-y line of figure 6,
     or a dragged line in figure 4) would otherwise be drawn straight over the axes and
     the legend.                                                                        */
  function lineSeg(g, a, b, sx, sy, opt) {
    const o = Object.assign({ color: SC.a2, w: 2, dash: null, op: 1 }, opt || {});
    const dx = sx.domain(), dyr = sy.domain();
    const y0 = Math.min(dyr[0], dyr[1]), y1 = Math.max(dyr[0], dyr[1]);
    let xa = dx[0], xb = dx[1];
    if (Math.abs(b) > 1e-12) {
      const u = (y0 - a) / b, v = (y1 - a) / b;
      xa = Math.max(xa, Math.min(u, v));
      xb = Math.min(xb, Math.max(u, v));
    } else if (a < y0 || a > y1) {
      return g.append("line");                    // entirely outside — draw nothing
    }
    if (!(xb > xa)) return g.append("line");
    const el = g.append("line")
      .attr("x1", sx(xa)).attr("y1", sy(a + b * xa))
      .attr("x2", sx(xb)).attr("y2", sy(a + b * xb))
      .attr("stroke", o.color).attr("stroke-width", o.w).attr("stroke-opacity", o.op);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    return el;
  }

  /* A rectangular clip region, so a scatter drawn on a FIXED axis range cannot
     spill over the axes when a stray point falls outside it. ST.frame wipes the
     svg on every redraw, so the id may safely be reused.                        */
  function clip(svg, id, x, y, w, h) {
    /* the rect is in the USER SPACE OF THE ELEMENT that references the clip, i.e.
       inside a margined <g> it starts at (0, 0), not at the margin                */
    svg.append("defs").append("clipPath").attr("id", id)
      .append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h);
    return "url(#" + id + ")";
  }

  /* a small titled sub-frame inside a bigger <g> */
  function sub(g, x, y, w, h, title) {
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    if (title) gg.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5)
      .attr("fill", SC.ink).attr("font-weight", 600).text(title);
    return { g: gg, w: w, h: h };
  }

  return { fit, lev, bvn, bvnSample, RUNX, RUNY, LEVX, LEVY, tp2, pstr, pts, lineSeg, sub, clip };
})();

/* ─────────────── 1 · the cloud, and r as a mean of signed products ─────────────── */
(function () {
  const svg = d3.select("#scat-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 440;
  const eR = document.getElementById("sc-r"), eRv = document.getElementById("sc-rv");
  const eN = document.getElementById("sc-n"), eNv = document.getElementById("sc-nv");
  const eSh = document.getElementById("sc-shape");
  const eQ = document.getElementById("sc-quad"), eSD = document.getElementById("sc-sd");
  const out = document.getElementById("scat-readout");
  let seed = 1607;

  function draw() {
    const rho = +eR.value, n = +eN.value, shape = eSh.value;
    const rng = ST.rng(seed);
    const s = RG.bvnSample(n, rho, rng);
    /* put the raw cloud on a readable, non-standardised scale */
    const X = s.x.map(v => 50 + 10 * v), Y = s.y.map(v => 70 + 12 * v);

    /* the same points after the chosen change of centre or scale */
    let X2 = X.slice(), Y2 = Y.slice(), lbl = "identical copy";
    if (shape === "stretch") { Y2 = Y.map(v => 4 * v); lbl = "y × 4"; }
    else if (shape === "shift") { X2 = X.map(v => v + 100); lbl = "x + 100"; }
    else if (shape === "flip") { Y2 = Y.map(v => -v); lbl = "y × (−1)"; }

    const f = ST.frame(svg, W, H, { l: 46, r: 14, t: 26, b: 44 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gapx = 26, mainW = Math.round(iw * 0.60), sideW = iw - mainW - gapx;

    /* ── main panel ── */
    const fitA = RG.fit(X, Y);
    const xd = d3.extent(X), yd = d3.extent(Y);
    const padx = (xd[1] - xd[0]) * 0.07 || 1, pady = (yd[1] - yd[0]) * 0.07 || 1;
    const x = d3.scaleLinear().domain([xd[0] - padx, xd[1] + padx]).range([0, mainW]);
    const y = d3.scaleLinear().domain([yd[0] - pady, yd[1] + pady]).range([ih, 0]);

    const gm = g.append("g");
    /* quadrant shading: the sign of (x−x̄)(y−ȳ) */
    if (eQ.checked) {
      const cx = x(fitA.xb), cy = y(fitA.yb);
      const q = (x0, y0, w, h, col) => gm.append("rect")
        .attr("x", x0).attr("y", y0).attr("width", Math.max(0, w)).attr("height", Math.max(0, h))
        .attr("fill", col).attr("fill-opacity", 0.075);
      q(cx, 0, mainW - cx, cy, SC.good);          // upper right: both above → +
      q(0, cy, cx, ih - cy, SC.good);             // lower left:  both below → +
      q(0, 0, cx, cy, SC.bad);                    // upper left:  x below, y above → −
      q(cx, cy, mainW - cx, ih - cy, SC.bad);     // lower right
      gm.append("text").attr("x", mainW - 4).attr("y", 12).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", SC.good).text("z_x z_y > 0");
      gm.append("text").attr("x", 4).attr("y", 12).attr("font-size", 10)
        .attr("fill", SC.bad).text("z_x z_y < 0");
    }
    ST.axisB(gm, x, ih, 6, "x");
    ST.axisL(gm, y, 5, "y");
    gm.append("line").attr("x1", 0).attr("x2", mainW).attr("y1", y(fitA.yb)).attr("y2", y(fitA.yb))
      .attr("stroke", SC.muted).attr("stroke-width", 1).attr("stroke-dasharray", "4 4");
    gm.append("line").attr("y1", 0).attr("y2", ih).attr("x1", x(fitA.xb)).attr("x2", x(fitA.xb))
      .attr("stroke", SC.muted).attr("stroke-width", 1).attr("stroke-dasharray", "4 4");

    RG.pts(gm, X, Y, x, y, { r: 2.8, op: 0.55 });
    RG.lineSeg(gm, fitA.a, fitA.b, x, y, { color: SC.a2, w: 2.2 });
    if (eSD.checked) {
      const bsd = Math.sign(fitA.r || 1) * fitA.sy / fitA.sx;
      RG.lineSeg(gm, fitA.yb - bsd * fitA.xb, bsd, x, y, { color: SC.violet, w: 1.6, dash: "6 4" });
    }
    gm.append("text").attr("x", 4).attr("y", -10).attr("font-size", 11).attr("fill", SC.ink)
      .attr("font-weight", 600).text("the sample");

    /* ── side panel: same points, transformed ── */
    const fitB = RG.fit(X2, Y2);
    const gs = g.append("g").attr("transform", `translate(${mainW + gapx},0)`);
    const xd2 = d3.extent(X2), yd2 = d3.extent(Y2);
    const px2 = (xd2[1] - xd2[0]) * 0.08 || 1, py2 = (yd2[1] - yd2[0]) * 0.08 || 1;
    const x2 = d3.scaleLinear().domain([xd2[0] - px2, xd2[1] + px2]).range([0, sideW]);
    const y2 = d3.scaleLinear().domain([yd2[0] - py2, yd2[1] + py2]).range([ih, 0]);
    gs.append("rect").attr("x", -6).attr("y", -20).attr("width", sideW + 12).attr("height", ih + 26)
      .attr("rx", 8).attr("fill", SC.panel2).attr("fill-opacity", 0.55)
      .attr("stroke", SC.line);
    ST.axisB(gs, x2, ih, 4);
    ST.axisL(gs, y2, 4);
    RG.pts(gs, X2, Y2, x2, y2, { r: 2.4, op: 0.5, fill: SC.teal });
    RG.lineSeg(gs, fitB.a, fitB.b, x2, y2, { color: SC.a2, w: 1.8 });
    gs.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.ink)
      .attr("font-weight", 600).text(lbl);
    gs.append("text").attr("x", sideW).attr("y", ih + 34).attr("text-anchor", "end")
      .attr("font-size", 11.5).attr("fill", Math.abs(Math.abs(fitB.r) - Math.abs(fitA.r)) < 1e-9 ? SC.good : SC.a2)
      .text("r = " + ST.fmt(fitB.r, 5));
    gm.append("text").attr("x", mainW).attr("y", ih + 34).attr("text-anchor", "end")
      .attr("font-size", 11.5).attr("fill", SC.ink).text("r = " + ST.fmt(fitA.r, 5));

    /* the mean-of-products check, computed the long way */
    const zx = ST.zScores(X), zy = ST.zScores(Y);
    let acc = 0;
    for (let i = 0; i < n; i++) acc += zx[i] * zy[i];
    const rz = acc / (n - 1);
    let plus = 0;
    for (let i = 0; i < n; i++) if (zx[i] * zy[i] > 0) plus++;

    out.innerHTML =
      `target <span class="keep">ρ</span> = <b>${ST.fmt(rho, 2)}</b> · n = <b>${n}</b> · `
      + `sample r = <b>${ST.fmt(fitA.r, 5)}</b>, and the long way — (1/(n−1))·Σ z_x z_y = <b>${ST.fmt(rz, 5)}</b>`
      + ` — is the same number.<br>`
      + `<b>${ST.pct(plus / n, 1)}</b> of the points sit in a green quadrant, contributing a positive product; `
      + `slope b = r·s_y/s_x = <b>${ST.fmt(fitA.b, 4)}</b>, s_x = ${ST.fmt(fitA.sx, 3)}, s_y = ${ST.fmt(fitA.sy, 3)}.<br>`
      + `The right-hand panel is the SAME sample after "${lbl}": `
      + (Math.abs(Math.abs(fitB.r) - Math.abs(fitA.r)) < 1e-9
        ? `|r| is unchanged to every digit${fitB.r * fitA.r < 0 ? ", only the sign flipped" : ""} — r is invariant to centre and scale.`
        : `r = ${ST.fmt(fitB.r, 5)}.`);
  }

  eR.oninput = () => { eRv.textContent = (+eR.value).toFixed(2); draw(); };
  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  eSh.onchange = draw; eQ.onchange = draw; eSD.onchange = draw;
  document.getElementById("sc-new").onclick = () => { seed = (seed * 31 + 7) % 100000; draw(); };
  draw();
})();

/* ─────────────── 2 · six clouds r cannot tell apart ─────────────── */
(function () {
  const svg = d3.select("#trap-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 500;
  const eShow = document.getElementById("tp-show");
  const eN = document.getElementById("tp-n"), eNv = document.getElementById("tp-nv");
  const out = document.getElementById("trap-readout");
  let seed = 8801;

  const makers = [
    { t: "1 · an honest linear cloud", note: "r describes it correctly",
      f: (n, r) => { const s = RG.bvnSample(n, 0.85, r); return { x: s.x, y: s.y }; } },
    { t: "2 · an exact parabola", note: "y is a FUNCTION of x, and r ≈ 0",
      f: (n) => { const x = ST.linspace(-2.2, 2.2, n); return { x: x, y: x.map(v => v * v - 1.6) }; } },
    { t: "3 · a blob plus one point", note: "one observation makes the correlation",
      f: (n, r) => {
        const x = [], y = [];
        for (let i = 0; i < n - 1; i++) { x.push(ST.randn(r) * 0.55 - 0.6); y.push(ST.randn(r) * 0.55 - 0.6); }
        /* the spike is placed at 1.3√n SDs out, because one point's share of r falls
           like 1/n — this keeps the demonstration honest at every sample size */
        const d = 1.3 * Math.sqrt(n);
        x.push(d); y.push(d); return { x: x, y: y };
      } },
    { t: "4 · two clusters", note: "no association inside either group",
      f: (n, r) => {
        const x = [], y = [];
        for (let i = 0; i < n; i++) {
          const g = i % 2 ? 1.6 : -1.6;
          x.push(g + ST.randn(r) * 0.45); y.push(g + ST.randn(r) * 0.45);
        }
        return { x: x, y: y };
      } },
    { t: "5 · monotone but curved", note: "Pearson understates; Spearman does not",
      f: (n, r) => {
        const x = [], y = [];
        for (let i = 0; i < n; i++) { const v = -2 + 4 * (i + 0.5) / n; x.push(v); y.push(Math.exp(1.15 * v) + 0.10 * ST.randn(r)); }
        return { x: x, y: y };
      } },
    { t: "6 · a fan", note: "the line is fine; every standard error is not",
      f: (n, r) => {
        const x = [], y = [];
        for (let i = 0; i < n; i++) { const v = -2 + 4 * (i + 0.5) / n; x.push(v); y.push(0.8 * v + (0.12 + 0.62 * (v + 2)) * ST.randn(r)); }
        return { x: x, y: y };
      } }
  ];

  function draw() {
    const n = +eN.value, show = eShow.value;
    const rng = ST.rng(seed);
    const f = ST.frame(svg, W, H, { l: 8, r: 8, t: 10, b: 8 });
    const cs = ST.cells(f.iw, f.ih, 3, 2, { l: 30, r: 10, t: 30, b: 26 }, { x: 16, y: 30 });
    const rows = [];
    cs.forEach((c, k) => {
      const m = makers[k];
      const d = m.f(n, rng);
      const fitD = RG.fit(d.x, d.y);
      const rs = ST.spearman(d.x, d.y);
      rows.push({ t: m.t, r: fitD.r, rs: rs, note: m.note });

      const gg = f.g.append("g").attr("transform", `translate(${c.x + c.m.l},${c.y + c.m.t})`);
      const xd = d3.extent(d.x), yd = d3.extent(d.y);
      const px = (xd[1] - xd[0]) * 0.08 || 1, py = (yd[1] - yd[0]) * 0.10 || 1;
      const x = d3.scaleLinear().domain([xd[0] - px, xd[1] + px]).range([0, c.iw]);
      const y = d3.scaleLinear().domain([yd[0] - py, yd[1] + py]).range([c.ih, 0]);
      gg.append("rect").attr("x", -6).attr("y", -6).attr("width", c.iw + 12).attr("height", c.ih + 12)
        .attr("rx", 7).attr("fill", SC.panel2).attr("fill-opacity", 0.45).attr("stroke", SC.line);
      ST.axisB(gg, x, c.ih, 3);
      ST.axisL(gg, y, 3);
      RG.pts(gg, d.x, d.y, x, y, { r: 2.2, op: 0.6 });
      if (show !== "none") RG.lineSeg(gg, fitD.a, fitD.b, x, y, { color: SC.a2, w: 1.8 });
      gg.append("text").attr("x", -6).attr("y", -14).attr("font-size", 11).attr("font-weight", 600)
        .attr("fill", SC.ink).text(m.t);
      const txt = gg.append("text").attr("x", c.iw + 4).attr("y", c.ih + 22).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.a2);
      txt.text("r = " + ST.fmt(fitD.r, 3) + (show === "both" ? "   ρ_S = " + ST.fmt(rs, 3) : ""));
      gg.append("text").attr("x", -6).attr("y", c.ih + 22).attr("font-size", 10)
        .attr("fill", SC.muted).text(m.note);
    });

    out.innerHTML = rows.map(r =>
      `<b>${r.t.split(" · ")[0]}</b> r = ${ST.fmt(r.r, 3)}${show === "both" ? " / ρ_S = " + ST.fmt(r.rs, 3) : ""}`
    ).join(" &nbsp;·&nbsp; ")
      + `<br>Panels 1 and 3 can be pushed to almost the same r by one point. Panel 2 has r ≈ 0 with a `
      + `deterministic relationship. Panel 5 shows the gap between Pearson and Spearman when the form is `
      + `monotone but curved. Panel 6's r and line are unobjectionable — it is the standard errors that are wrong.`;
  }

  eShow.onchange = draw;
  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  document.getElementById("tp-new").onclick = () => { seed = (seed * 37 + 11) % 100000; draw(); };
  draw();
})();

/* ─────────────── 3 · three ways r lies, each against its exact formula ─────────────── */
(function () {
  const svg = d3.select("#lie-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 450;
  const eM = document.getElementById("li-mech");
  const eRho = document.getElementById("li-rho"), eRhov = document.getElementById("li-rhov");
  const eK = document.getElementById("li-k"), eKv = document.getElementById("li-kv");
  const eKl = document.getElementById("li-klbl");
  const eN = document.getElementById("li-n"), eNv = document.getElementById("li-nv");
  const out = document.getElementById("lie-readout");
  let seed = 4242;

  /* the three exact results, verified by simulation in the prose:
       error      r' = ρ√λ,  λ = 1/(1+σu²)          and slope b' = ρ·λ  (σx = σy = 1)
       range      r' = ρu/√(ρ²u² + 1 − ρ²)          slope unchanged
       agg        r' = ρ_B/(1 + σW²/(m σB²))  with all four SDs equal, ρ_W = 0
                  → individual r = ρ_B/2, group-mean r = ρ_B/(1 + 1/m)              */
  const CFG = {
    err: { lab: "noise SD σ_u", min: 0, max: 3, step: 0.02, def: 1,
      kname: k => "σ_u = " + ST.fmt(k, 2),
      theory: (rho, k) => { const lam = 1 / (1 + k * k); return rho * Math.sqrt(lam); },
      xlab: "measurement-noise SD  σ_u" },
    range: { lab: "keep |x| < c·σ", min: 0.25, max: 3, step: 0.02, def: 1,
      kname: k => "c = " + ST.fmt(k, 2),
      theory: (rho, k) => {
        const u = truncSD(k);
        return rho * u / Math.sqrt(rho * rho * u * u + 1 - rho * rho);
      },
      xlab: "half-width of the kept range, c (in SDs)" },
    agg: { lab: "group size m", min: 1, max: 40, step: 1, def: 10,
      kname: k => "m = " + Math.round(k),
      theory: (rho, k) => rho / (1 + 1 / Math.max(1, Math.round(k))),
      xlab: "units averaged per group, m" }
  };

  /* SD of a standard normal truncated to |z| < c, relative to 1:
       Var = 1 − 2c·φ(c)/(2Φ(c) − 1)                                              */
  function truncSD(c) {
    const den = 2 * ST.normCdf(c) - 1;
    if (den <= 1e-12) return 0;
    return Math.sqrt(Math.max(0, 1 - 2 * c * ST.normPdf(c) / den));
  }

  function simulate(mech, rho, k, n, rng) {
    if (mech === "err") {
      const s = RG.bvnSample(n, rho, rng);
      const X = s.x.map(v => v + k * ST.randn(rng));
      return { x: X, y: s.y, trueA: 0, trueB: rho };
    }
    if (mech === "range") {
      const X = [], Y = [];
      let guard = 0;
      while (X.length < n && guard++ < n * 400) {
        const p = RG.bvn(rho, rng);
        if (Math.abs(p[0]) < k) { X.push(p[0]); Y.push(p[1]); }
      }
      return { x: X, y: Y, trueA: 0, trueB: rho };
    }
    /* aggregation: ρ_B = rho, ρ_W = 0, every SD equal to 1 */
    const m = Math.max(1, Math.round(k));
    const G = Math.max(8, Math.round(n / m));
    const X = [], Y = [];
    for (let j = 0; j < G; j++) {
      const gj = RG.bvn(rho, rng);
      let ax = 0, ay = 0;
      for (let i = 0; i < m; i++) { ax += gj[0] + ST.randn(rng); ay += gj[1] + ST.randn(rng); }
      X.push(ax / m); Y.push(ay / m);
    }
    return { x: X, y: Y, trueA: 0, trueB: rho, groups: G, m: m };
  }

  function draw() {
    const mech = eM.value, rho = +eRho.value, k = +eK.value, n = +eN.value;
    const cfg = CFG[mech];
    const rng = ST.rng(seed);
    const d = simulate(mech, rho, k, n, rng);
    const fitD = RG.fit(d.x, d.y);

    const f = ST.frame(svg, W, H, { l: 46, r: 16, t: 26, b: 44 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gap = 40, leftW = Math.round(iw * 0.50), rightW = iw - leftW - gap;

    /* ── left: the distorted cloud ── */
    const gl = g.append("g");
    const xd = d3.extent(d.x), yd = d3.extent(d.y);
    const px = (xd[1] - xd[0]) * 0.08 || 1, py = (yd[1] - yd[0]) * 0.08 || 1;
    const x = d3.scaleLinear().domain([xd[0] - px, xd[1] + px]).range([0, leftW]);
    const y = d3.scaleLinear().domain([yd[0] - py, yd[1] + py]).range([ih, 0]);
    ST.gridY(gl, y, leftW, 4);
    ST.axisB(gl, x, ih, 5, mech === "err" ? "x*  (measured with noise)" : "x");
    ST.axisL(gl, y, 4, "y");
    RG.pts(gl, d.x, d.y, x, y, { r: 2.4, op: 0.5 });
    RG.lineSeg(gl, 0, rho, x, y, { color: SC.good, w: 1.8, dash: "6 4" });   // the true line
    RG.lineSeg(gl, fitD.a, fitD.b, x, y, { color: SC.a2, w: 2.2 });
    gl.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink).text("the sample you actually observe");
    ST.legend(gl, [
      { label: "true population line, slope " + ST.fmt(rho, 2), color: SC.good, dash: "5 3" },
      { label: "fitted line, slope " + ST.fmt(fitD.b, 3), color: SC.a2 }
    ], 8, 14);

    /* ── right: observed r against the distortion, with the exact curve ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + gap},0)`);
    const ks = ST.linspace(cfg.min, cfg.max, 120);
    const curve = ks.map(v => ({ k: v, r: cfg.theory(rho, v) }));
    const x2 = d3.scaleLinear().domain([cfg.min, cfg.max]).range([0, rightW]);
    const y2 = d3.scaleLinear().domain([0, Math.max(1, rho * 1.15)]).range([ih, 0]);
    ST.gridY(gr, y2, rightW, 5);
    ST.axisB(gr, x2, ih, 5, cfg.xlab);
    ST.axisL(gr, y2, 5, "observed r");
    gr.append("line").attr("x1", 0).attr("x2", rightW).attr("y1", y2(rho)).attr("y2", y2(rho))
      .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 4");
    gr.append("text").attr("x", rightW - 2).attr("y", y2(rho) - 5).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.good).text("true ρ = " + ST.fmt(rho, 2));
    gr.append("path").attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 2)
      .attr("stroke-dasharray", "7 4")
      .attr("d", d3.line().x(p => x2(p.k)).y(p => y2(ST.clamp(p.r, 0, y2.domain()[1])))(curve));
    const pred = cfg.theory(rho, k);
    gr.append("line").attr("x1", x2(k)).attr("x2", x2(k)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.line).attr("stroke-width", 1);
    gr.append("circle").attr("cx", x2(k)).attr("cy", y2(ST.clamp(Math.abs(fitD.r), 0, y2.domain()[1])))
      .attr("r", 5.5).attr("fill", SC.a2).attr("stroke", SC.bg).attr("stroke-width", 1.5);
    gr.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink).text("observed r vs the exact formula");
    ST.legend(gr, [
      { label: "theory", color: SC.violet, dash: "5 3" },
      { label: "this sample", color: SC.a2 }
    ], rightW - 92, 14);

    const names = {
      err: "Measurement error in x shrinks BOTH r and the slope: r → ρ√λ and b → ρλ with λ = 1/(1 + σ_u²).",
      range: "Restricting the range of x shrinks r but leaves the slope alone.",
      agg: "Averaging into groups of m inflates r towards the between-group correlation."
    };
    const extra = mech === "err"
      ? ` λ = ${ST.fmt(1 / (1 + k * k), 3)}, so the slope should be ≈ ${ST.fmt(rho / (1 + k * k), 3)} and is <b>${ST.fmt(fitD.b, 3)}</b>.`
      : mech === "range"
        ? ` the kept SD is u = ${ST.fmt(truncSD(k), 3)} of the full one, and the slope should still be ≈ ${ST.fmt(rho, 2)} — it is <b>${ST.fmt(fitD.b, 3)}</b>.`
        : ` ${d.groups} groups of ${d.m}; the individual-level correlation for these settings is ρ/2 = ${ST.fmt(rho / 2, 3)}.`;

    out.innerHTML = `${names[mech]}<br>`
      + `${cfg.kname(k)} · n = ${d.x.length} · observed r = <b>${ST.fmt(fitD.r, 4)}</b> · `
      + `formula predicts <b>${ST.fmt(pred, 4)}</b> · true ρ = ${ST.fmt(rho, 2)}.` + extra;
  }

  function syncK() {
    const cfg = CFG[eM.value];
    eK.min = cfg.min; eK.max = cfg.max; eK.step = cfg.step; eK.value = cfg.def;
    eKl.textContent = cfg.lab;
    eKv.textContent = cfg.kname(cfg.def).split("= ")[1];
  }
  eM.onchange = () => { syncK(); draw(); };
  eRho.oninput = () => { eRhov.textContent = (+eRho.value).toFixed(2); draw(); };
  eK.oninput = () => { eKv.textContent = CFG[eM.value].kname(+eK.value).split("= ")[1]; draw(); };
  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  document.getElementById("li-new").onclick = () => { seed = (seed * 41 + 13) % 100000; draw(); };
  syncK(); draw();
})();

/* ─────────────── 4 · drag the line, watch the sum of squares ─────────────── */
(function () {
  const svg = d3.select("#ls-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const eA = document.getElementById("ls-a"), eAv = document.getElementById("ls-av");
  const eB = document.getElementById("ls-b"), eBv = document.getElementById("ls-bv");
  const eMode = document.getElementById("ls-mode");
  const out = document.getElementById("ls-readout");

  const X = RG.RUNX, Y = RG.RUNY;
  const best = RG.fit(X, Y);                 // a = 6, b = 3, SSE = 250 exactly
  const state = { a: 14, b: 1.8 };

  const sseOf = (a, b) => { let s = 0; for (let i = 0; i < X.length; i++) { const e = Y[i] - a - b * X[i]; s += e * e; } return s; };

  function draw() {
    const a = state.a, b = state.b, mode = eMode.value;
    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 22, b: 118 });
    const g = f.g, iw = f.iw, ih = f.ih;

    const x = d3.scaleLinear().domain([0, 20]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 78]).range([ih, 0]);
    /* one unit of y must be drawable as a square, so the square's side is taken in
       screen pixels from the VERTICAL scale and mirrored horizontally.            */
    ST.gridY(g, y, iw, 5);
    ST.axisB(g, x, ih, 8, "components replaced  x");
    ST.axisL(g, y, 5, "minutes on site  y");

    /* the squares / gaps */
    const gsq = g.append("g");
    for (let i = 0; i < X.length; i++) {
      const yi = Y[i], yh = a + b * X[i], e = yi - yh;
      const py = y(yi), ph = y(yh);
      if (mode === "sq") {
        const side = Math.abs(py - ph);
        const left = e >= 0 ? x(X[i]) : x(X[i]) - side;
        gsq.append("rect")
          .attr("x", Math.min(x(X[i]), left)).attr("y", Math.min(py, ph))
          .attr("width", side).attr("height", side)
          .attr("fill", e >= 0 ? SC.accent : SC.bad).attr("fill-opacity", 0.16)
          .attr("stroke", e >= 0 ? SC.accent : SC.bad).attr("stroke-opacity", 0.55);
      } else if (mode === "vert") {
        gsq.append("line").attr("x1", x(X[i])).attr("x2", x(X[i])).attr("y1", py).attr("y2", ph)
          .attr("stroke", e >= 0 ? SC.accent : SC.bad).attr("stroke-width", 1.8).attr("stroke-opacity", 0.8);
      } else {
        /* perpendicular foot, in DATA units, to make the point that this is a
           different criterion and is not what least squares minimises */
        const t = ST.clamp((X[i] + b * (yi - a)) / (1 + b * b), 0, 20);
        gsq.append("line").attr("x1", x(X[i])).attr("y1", y(yi))
          .attr("x2", x(t)).attr("y2", y(ST.clamp(a + b * t, 0, 78)))
          .attr("stroke", SC.violet).attr("stroke-width", 1.8).attr("stroke-opacity", 0.8);
      }
    }

    /* the least-squares line, faint, always present as the target */
    RG.lineSeg(g, best.a, best.b, x, y, { color: SC.good, w: 1.5, dash: "6 4", op: 0.85 });
    /* the draggable line */
    RG.lineSeg(g, a, b, x, y, { color: SC.a2, w: 2.4 });
    RG.pts(g, X, Y, x, y, { r: 4, op: 0.95, stroke: SC.bg });

    /* two handles: translate at x̄, rotate near the right edge */
    const hx1 = best.xb, hx2 = 18;
    /* the handles stay grabbable at the edge of the frame even when the line has
       been pushed off it, so the reader can always drag back                     */
    const clampY = v => ST.clamp(v, y.domain()[0], y.domain()[1]);
    const mk = (cx, cy, col) => g.append("circle").attr("class", "dragpt")
      .attr("cx", x(cx)).attr("cy", y(clampY(cy))).attr("r", 7)
      .attr("fill", col).attr("stroke", SC.bg).attr("stroke-width", 2)
      .style("cursor", "grab");
    const hT = mk(hx1, a + b * hx1, SC.a2);
    const hR = mk(hx2, a + b * hx2, SC.violet);
    hT.call(d3.drag().on("drag", ev => {
      const yv = y.invert(ST.clamp(ev.y, 0, ih));
      state.a = ST.clamp(yv - state.b * hx1, -14, 26);
      sync(); draw();
    }));
    hR.call(d3.drag().on("drag", ev => {
      const yv = y.invert(ST.clamp(ev.y, 0, ih));
      const pivot = state.a + state.b * hx1;                 // rotate about the translate handle
      const nb = ST.clamp((yv - pivot) / (hx2 - hx1), 0, 6);
      state.b = nb; state.a = pivot - nb * hx1;
      state.a = ST.clamp(state.a, -14, 26);
      sync(); draw();
    }));
    g.append("text").attr("x", x(hx1) + 10).attr("y", Math.max(12, y(clampY(a + b * hx1)) - 10))
      .attr("font-size", 10).attr("fill", SC.a2).text("drag: move");
    g.append("text").attr("x", x(hx2) - 8).attr("y", Math.max(12, y(clampY(a + b * hx2)) - 10)).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.violet).text("drag: tilt");

    /* ── the comparison bar and the SSE-vs-slope inset ── */
    const sse = sseOf(a, b), sseMin = best.sse;
    const strip = g.append("g").attr("transform", `translate(0,${ih + 44})`);
    const barW = Math.round(iw * 0.52);
    const bx = d3.scaleLinear().domain([0, Math.max(sse, sseMin * 6)]).range([0, barW]);
    strip.append("text").attr("x", -6).attr("y", 4).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("yours");
    strip.append("rect").attr("x", 0).attr("y", -8).attr("width", bx(sse)).attr("height", 14)
      .attr("rx", 2).attr("fill", SC.a2).attr("fill-opacity", 0.8);
    strip.append("text").attr("x", -6).attr("y", 26).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("best");
    strip.append("rect").attr("x", 0).attr("y", 14).attr("width", bx(sseMin)).attr("height", 14)
      .attr("rx", 2).attr("fill", SC.good).attr("fill-opacity", 0.8);
    strip.append("text").attr("x", bx(sse) + 8).attr("y", 3).attr("font-size", 11)
      .attr("fill", SC.a2).text("Σe² = " + ST.fmt(sse, 1));
    strip.append("text").attr("x", bx(sseMin) + 8).attr("y", 25).attr("font-size", 11)
      .attr("fill", SC.good).text("Σe² = 250 exactly");

    const insW = iw - barW - 70, insH = 62;
    const gi = g.append("g").attr("transform", `translate(${barW + 70},${ih + 30})`);
    const bs = ST.linspace(0, 6, 121);
    const curve = bs.map(bb => {
      /* profile: at each slope, use the BEST intercept, so the picture is the
         one-dimensional slice through the minimum rather than an arbitrary cut */
      const aa = best.yb - bb * best.xb;
      return { b: bb, s: sseOf(aa, bb) };
    });
    const xi = d3.scaleLinear().domain([0, 6]).range([0, insW]);
    const yi = d3.scaleLinear().domain([0, d3.max(curve, d => d.s)]).range([insH, 0]);
    gi.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.8)
      .attr("d", d3.line().x(d => xi(d.b)).y(d => yi(d.s))(curve));
    gi.append("line").attr("x1", xi(best.b)).attr("x2", xi(best.b)).attr("y1", 0).attr("y2", insH)
      .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3");
    gi.append("circle").attr("cx", xi(ST.clamp(b, 0, 6)))
      .attr("cy", yi(ST.clamp(sseOf(best.yb - b * best.xb, b), 0, yi.domain()[1])))
      .attr("r", 4).attr("fill", SC.a2);
    ST.axisB(gi, xi, insH, 4, "slope b");
    gi.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10).attr("fill", SC.muted)
      .text("Σe² against slope (intercept optimised)");

    const excess = sse - sseMin;
    out.innerHTML = `line: ŷ = ${ST.fmt(a, 2)} + ${ST.fmt(b, 3)}x · `
      + `Σe² = <b>${ST.fmt(sse, 2)}</b> vs the minimum <b>250</b> — `
      + (excess < 1e-6 ? `<span style="color:${SC.good}">you are at the least-squares line</span>`
        : `${ST.fmt(excess, 2)} worse (${ST.fmt(100 * excess / sseMin, 1)}% above)`) + `.<br>`
      + `The minimum is at a = ȳ − bx̄ = <b>6</b>, b = r·s_y/s_x = <b>3</b>, and there the residuals satisfy `
      + `Σeᵢ = 0 and Σxᵢeᵢ = 0 — currently Σeᵢ = ${ST.fmt(ST.sum(X.map((v, i) => Y[i] - a - b * v)), 2)}, `
      + `Σxᵢeᵢ = ${ST.fmt(ST.sum(X.map((v, i) => v * (Y[i] - a - b * v))), 2)}.`
      + (eMode.value === "perp" ? `<br>The violet segments are PERPENDICULAR distances — a different criterion, whose minimiser is the major axis of the cloud, not this line.` : "");
  }

  function sync() {
    eA.value = state.a.toFixed(1); eAv.textContent = state.a.toFixed(1);
    eB.value = state.b.toFixed(2); eBv.textContent = state.b.toFixed(2);
  }
  eA.oninput = () => { state.a = +eA.value; eAv.textContent = state.a.toFixed(1); draw(); };
  eB.oninput = () => { state.b = +eB.value; eBv.textContent = state.b.toFixed(2); draw(); };
  eMode.onchange = draw;
  document.getElementById("ls-fit").onclick = () => { state.a = best.a; state.b = best.b; sync(); draw(); };
  document.getElementById("ls-reset").onclick = () => { state.a = 14; state.b = 1.8; sync(); draw(); };
  sync(); draw();
})();

/* ─────────────── 5 · test and retest: the regression effect, measured ─────────────── */
(function () {
  const svg = d3.select("#rtm-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 480;
  const eR = document.getElementById("rt-r"), eRv = document.getElementById("rt-rv");
  const eN = document.getElementById("rt-n"), eNv = document.getElementById("rt-nv");
  const eC = document.getElementById("rt-c"), eCv = document.getElementById("rt-cv");
  const eAvg = document.getElementById("rt-avg"), eSD = document.getElementById("rt-sd");
  const out = document.getElementById("rtm-readout");

  /* the five numbers the prose uses */
  const MX = 49.5, SX = 10.2, MY = 69.1, SY = 11.8;
  const HALF = 4.5;                       // half-width of the selection band, in midterm points
  let seed = 606;
  const state = { c: 66 };

  function draw() {
    const rho = +eR.value, n = +eN.value;
    const rng = ST.rng(seed);
    const s = RG.bvnSample(n, rho, rng);
    const X = s.x.map(v => MX + SX * v), Y = s.y.map(v => MY + SY * v);
    const fitD = RG.fit(X, Y);

    const f = ST.frame(svg, W, H, { l: 52, r: 16, t: 24, b: 96 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLinear().domain([15, 85]).range([0, iw]);
    const y = d3.scaleLinear().domain([25, 110]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);
    ST.axisB(g, x, ih, 8, "midterm score  x");
    ST.axisL(g, y, 5, "final score  y");

    /* the selection band */
    const c = state.c;
    const gband = g.append("g").attr("class", "dragpt").style("cursor", "ew-resize");
    gband.append("rect")
      .attr("x", x(c - HALF)).attr("y", 0)
      .attr("width", Math.max(2, x(c + HALF) - x(c - HALF))).attr("height", ih)
      .attr("fill", SC.a2).attr("fill-opacity", 0.11)
      .attr("stroke", SC.a2).attr("stroke-opacity", 0.6);
    gband.call(d3.drag().on("drag", ev => {
      state.c = ST.clamp(x.invert(ST.clamp(ev.x, 0, iw)), 25, 75);
      eC.value = state.c.toFixed(1); eCv.textContent = state.c.toFixed(1);
      draw();
    }));

    /* points, split by whether they are in the band; clipped, because the axes are
       fixed and an occasional student sits outside them */
    const inb = X.map(v => Math.abs(v - c) <= HALF);
    const gp = g.append("g").attr("clip-path", RG.clip(f.svg, "rtm-clip", 0, 0, iw, ih));
    RG.pts(gp, X.filter((_, i) => !inb[i]), Y.filter((_, i) => !inb[i]), x, y, { r: 2.6, op: 0.35 });
    RG.pts(gp, X.filter((_, i) => inb[i]), Y.filter((_, i) => inb[i]), x, y,
      { r: 3.4, op: 0.95, fill: SC.a2, stroke: SC.bg });

    /* the two lines */
    if (eSD.checked) {
      const bsd = fitD.sy / fitD.sx;
      RG.lineSeg(g, fitD.yb - bsd * fitD.xb, bsd, x, y, { color: SC.violet, w: 1.6, dash: "6 4" });
    }
    RG.lineSeg(g, fitD.a, fitD.b, x, y, { color: SC.good, w: 2.2 });

    /* the graph of averages: strip means */
    if (eAvg.checked) {
      const K = 12, lo = 20, hi = 80, w = (hi - lo) / K;
      const gs = g.append("g");
      for (let k = 0; k < K; k++) {
        const a0 = lo + k * w, a1 = a0 + w, mid = (a0 + a1) / 2;
        const ys = [];
        for (let i = 0; i < n; i++) if (X[i] >= a0 && X[i] < a1) ys.push(Y[i]);
        if (ys.length < 3) continue;
        const m = ST.mean(ys);
        gs.append("rect").attr("x", x(mid) - 4).attr("y", y(m) - 4).attr("width", 8).attr("height", 8)
          .attr("fill", "none").attr("stroke", SC.teal).attr("stroke-width", 1.8);
      }
      gs.append("text").attr("x", iw).attr("y", 12).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", SC.teal).text("□ = mean y in this strip");
    }

    /* the means, in and out */
    const bx = [], by = [];
    for (let i = 0; i < n; i++) if (inb[i]) { bx.push(X[i]); by.push(Y[i]); }
    const mbx = bx.length ? ST.mean(bx) : NaN, mby = by.length ? ST.mean(by) : NaN;
    if (bx.length && mbx >= x.domain()[0] && mbx <= x.domain()[1]
        && mby >= y.domain()[0] && mby <= y.domain()[1]) {
      g.append("circle").attr("cx", x(mbx)).attr("cy", y(mby)).attr("r", 6.5)
        .attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 2.4);
      g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(mby)).attr("y2", y(mby))
        .attr("stroke", SC.bad).attr("stroke-width", 1).attr("stroke-dasharray", "3 4");
    }
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(fitD.yb)).attr("y2", y(fitD.yb))
      .attr("stroke", SC.muted).attr("stroke-width", 1).attr("stroke-dasharray", "2 5");
    g.append("text").attr("x", 2).attr("y", y(fitD.yb) - 5).attr("font-size", 10)
      .attr("fill", SC.muted).text("ȳ = " + ST.fmt(fitD.yb, 1));

    /* the z-score bar comparison beneath */
    const zx = (mbx - fitD.xb) / fitD.sx, zy = (mby - fitD.yb) / fitD.sy;
    const strip = g.append("g").attr("transform", `translate(0,${ih + 46})`);
    const zs = d3.scaleLinear().domain([-3, 3]).range([0, iw]);
    strip.append("line").attr("x1", 0).attr("x2", iw).attr("y1", 16).attr("y2", 16)
      .attr("stroke", SC.line);
    strip.append("line").attr("x1", zs(0)).attr("x2", zs(0)).attr("y1", -8).attr("y2", 40)
      .attr("stroke", SC.muted).attr("stroke-dasharray", "3 3");
    strip.append("text").attr("x", zs(0)).attr("y", 52).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.muted).text("the mean");
    const bar = (z, yy, col, lab) => {
      if (!isFinite(z)) return;
      strip.append("rect").attr("x", Math.min(zs(0), zs(z))).attr("y", yy)
        .attr("width", Math.abs(zs(z) - zs(0))).attr("height", 11).attr("rx", 2)
        .attr("fill", col).attr("fill-opacity", 0.8);
      strip.append("text").attr("x", zs(z) + (z >= 0 ? 6 : -6)).attr("y", yy + 9.5)
        .attr("text-anchor", z >= 0 ? "start" : "end")
        .attr("font-size", 10.5).attr("fill", col).text(lab + " z = " + ST.fmt(z, 3));
    };
    bar(zx, -2, SC.a2, "midterm");
    bar(zy, 22, SC.bad, "final");

    const ratio = (isFinite(zx) && Math.abs(zx) > 1e-9) ? zy / zx : NaN;
    out.innerHTML =
      `class of ${n}, sample r = <b>${ST.fmt(fitD.r, 3)}</b>, fitted line ŷ = ${ST.fmt(fitD.a, 2)} + ${ST.fmt(fitD.b, 3)}x.<br>`
      + (bx.length < 3
        ? `Fewer than three students in the band — widen the class or move the band towards the centre.`
        : `The <b>${bx.length}</b> students with a midterm near ${ST.fmt(c, 1)} averaged `
        + `<b>${ST.fmt(mbx, 2)}</b> on the midterm (z = ${ST.fmt(zx, 3)}) and `
        + `<b>${ST.fmt(mby, 2)}</b> on the final (z = ${ST.fmt(zy, 3)}). `
        + `Ratio z_final / z_midterm = <b>${ST.fmt(ratio, 3)}</b>, against r = ${ST.fmt(fitD.r, 3)}.<br>`
        + `The final z-score is closer to zero than the midterm one whenever |r| &lt; 1 — that is the whole `
        + `regression effect, and it happens in both directions: drag the band below the mean and the group improves.`);
  }

  eR.oninput = () => { eRv.textContent = (+eR.value).toFixed(2); draw(); };
  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  eC.oninput = () => { state.c = +eC.value; eCv.textContent = state.c.toFixed(1); draw(); };
  eAvg.onchange = draw; eSD.onchange = draw;
  document.getElementById("rt-new").onclick = () => { seed = (seed * 29 + 5) % 100000; draw(); };
  draw();
})();

/* ─────────────── 6 · the two regression lines and the SD line ─────────────── */
(function () {
  const svg = d3.select("#two-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 460;
  const eR = document.getElementById("tw-r"), eRv = document.getElementById("tw-rv");
  const eK = document.getElementById("tw-k"), eKv = document.getElementById("tw-kv");
  const eQ = document.getElementById("tw-q"), eQv = document.getElementById("tw-qv");
  const eSD = document.getElementById("tw-sd");
  const out = document.getElementById("two-readout");
  const seed = 3131, N = 420;

  function draw() {
    const rho = +eR.value, k = +eK.value, q = +eQ.value;
    const rng = ST.rng(seed);
    const s = RG.bvnSample(N, rho, rng);
    const X = s.x.map(v => 50 + 10 * v), Y = s.y.map(v => 50 + 10 * k * v);
    const F = RG.fit(X, Y);

    const bYX = F.b;                            // y on x
    const bXY = F.r * F.sx / F.sy;              // x on y, in the (y, x) plot
    const aXY = F.xb - bXY * F.yb;
    const sdSlope = Math.sign(F.r || 1) * F.sy / F.sx;

    const f = ST.frame(svg, W, H, { l: 50, r: 16, t: 24, b: 78 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLinear().domain([50 - 42, 50 + 42]).range([0, iw]);
    const y = d3.scaleLinear().domain([50 - 42 * k, 50 + 42 * k]).range([ih, 0]);
    ST.gridY(g, y, iw, 5); ST.gridX(g, x, ih, 5);
    ST.axisB(g, x, ih, 7, "x");
    ST.axisL(g, y, 5, "y");
    RG.pts(g, X, Y, x, y, { r: 2.3, op: 0.4 });

    /* the x-on-y line, drawn on these axes: x = aXY + bXY·y  ⟺  y = (x − aXY)/bXY */
    if (Math.abs(bXY) > 1e-9) {
      RG.lineSeg(g, -aXY / bXY, 1 / bXY, x, y, { color: SC.violet, w: 2.2 });
    } else {
      g.append("line").attr("x1", x(F.xb)).attr("x2", x(F.xb)).attr("y1", 0).attr("y2", ih)
        .attr("stroke", SC.violet).attr("stroke-width", 2.2);
    }
    if (eSD.checked) RG.lineSeg(g, F.yb - sdSlope * F.xb, sdSlope, x, y, { color: SC.teal, w: 1.6, dash: "7 4" });
    RG.lineSeg(g, F.a, bYX, x, y, { color: SC.a2, w: 2.4 });

    /* the round trip, from a chosen x */
    const xq = F.xb + ST.normQuant(q) * F.sx;
    const yq = F.a + bYX * xq;                      // predict y from x
    const xback = aXY + bXY * yq;                   // then predict x from that y
    const dot = (cx, cy, col) => g.append("circle").attr("cx", x(cx)).attr("cy", y(cy)).attr("r", 5)
      .attr("fill", col).attr("stroke", SC.bg).attr("stroke-width", 1.5);
    g.append("line").attr("x1", x(xq)).attr("x2", x(xq)).attr("y1", ih).attr("y2", y(yq))
      .attr("stroke", SC.a2).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
    g.append("line").attr("x1", x(xq)).attr("x2", 0).attr("y1", y(yq)).attr("y2", y(yq))
      .attr("stroke", SC.a2).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
    g.append("line").attr("x1", 0).attr("x2", x(xback)).attr("y1", y(yq)).attr("y2", y(yq))
      .attr("stroke", SC.violet).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
    g.append("line").attr("x1", x(xback)).attr("x2", x(xback)).attr("y1", y(yq)).attr("y2", ih)
      .attr("stroke", SC.violet).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
    dot(xq, yq, SC.a2);
    dot(xback, yq, SC.violet);
    g.append("circle").attr("cx", x(F.xb)).attr("cy", y(F.yb)).attr("r", 4.5)
      .attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 1.6);
    g.append("text").attr("x", x(F.xb) + 8).attr("y", y(F.yb) - 8).attr("font-size", 10)
      .attr("fill", SC.ink).text("(x̄, ȳ)");

    ST.legend(g, [
      { label: "y on x  — use when x is given", color: SC.a2 },
      { label: "x on y  — use when y is given", color: SC.violet },
      { label: "SD line — neither", color: SC.teal, dash: "5 3" }
    ], 8, 14);

    const prod = bYX * bXY;
    out.innerHTML =
      `r = <b>${ST.fmt(F.r, 3)}</b> · s_y/s_x = ${ST.fmt(F.sy / F.sx, 3)}<br>`
      + `slope of y on x: b = r·s_y/s_x = <b>${ST.fmt(bYX, 4)}</b> &nbsp;·&nbsp; `
      + `slope of x on y: b′ = r·s_x/s_y = <b>${ST.fmt(bXY, 4)}</b> &nbsp;·&nbsp; `
      + `b·b′ = <b>${ST.fmt(prod, 4)}</b> and r² = <b>${ST.fmt(F.r * F.r, 4)}</b> — the same number.<br>`
      + `Start at x = ${ST.fmt(xq, 2)} (z = ${ST.fmt((xq - F.xb) / F.sx, 3)}). Predicting y gives `
      + `${ST.fmt(yq, 2)} (z = ${ST.fmt((yq - F.yb) / F.sy, 3)}). Predicting x back from THAT gives `
      + `<b>${ST.fmt(xback, 2)}</b> (z = ${ST.fmt((xback - F.xb) / F.sx, 3)}) — not ${ST.fmt(xq, 2)}. `
      + `The round trip shrank the z-score by r² = ${ST.fmt(F.r * F.r, 3)}.`;
  }

  eR.oninput = () => { eRv.textContent = (+eR.value).toFixed(2); draw(); };
  eK.oninput = () => { eKv.textContent = (+eK.value).toFixed(2); draw(); };
  eQ.oninput = () => { eQv.textContent = (+eQ.value).toFixed(2); draw(); };
  eSD.onchange = draw;
  draw();
})();

/* ─────────────── 7 · where SE(b) comes from ─────────────── */
(function () {
  const svg = d3.select("#slope-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const eN = document.getElementById("sl-n"), eNv = document.getElementById("sl-nv");
  const eS = document.getElementById("sl-s"), eSv = document.getElementById("sl-sv");
  const eW = document.getElementById("sl-w"), eWv = document.getElementById("sl-wv");
  const eB = document.getElementById("sl-b"), eBv = document.getElementById("sl-bv");
  const eP = document.getElementById("sl-panel");
  const out = document.getElementById("slope-readout");
  const B = 2000, A0 = 6;
  let seed = 9091;

  /* The design. At n = 12 it is exactly the running example's x-values, so w = 1
     reproduces S_xx = 318 and SE(b) = 5/√318 = 0.280386 — the number in the prose.
     Any other n is evenly spaced over the same range. Stretching by w about x̄ = 10
     multiplies S_xx by w².                                                        */
  function design(n, w) {
    const base = (n === 12) ? RG.RUNX : ST.linspace(2, 18, n);
    return base.map(v => 10 + w * (v - 10));
  }

  function draw() {
    const n = +eN.value, sig = +eS.value, w = +eW.value, b1 = +eB.value, panel = eP.value;
    const xs = design(n, w);
    const xb = ST.mean(xs);
    let Sxx = 0;
    for (const v of xs) Sxx += (v - xb) * (v - xb);
    const seTheory = sig / Math.sqrt(Sxx);

    const rng = ST.rng(seed);
    const bs = [], ts = [], keep = [];
    for (let r = 0; r < B; r++) {
      const ys = xs.map(v => A0 + b1 * v + sig * ST.randn(rng));
      const F = RG.fit(xs, ys);
      bs.push(F.b);
      ts.push((F.b - b1) / F.seb);
      if (r < 24) keep.push(F);
    }
    const mb = ST.mean(bs), sdb = ST.sd(bs);

    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 26, b: 46 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gap = 44, leftW = Math.round(iw * 0.43), rightW = iw - leftW - gap;

    /* ── left: 24 of the fitted lines over the truth ── */
    const gl = g.append("g");
    const xlo = 10 - 8 * w - 1, xhi = 10 + 8 * w + 1;
    const ymid = A0 + b1 * 10, yspan = Math.max(8 * w * Math.max(b1, 1) * 1.6, 3 * sig) + 6;
    const x = d3.scaleLinear().domain([xlo, xhi]).range([0, leftW]);
    const y = d3.scaleLinear().domain([ymid - yspan, ymid + yspan]).range([ih, 0]);
    ST.gridY(gl, y, leftW, 4);
    ST.axisB(gl, x, ih, 4, "x");
    ST.axisL(gl, y, 4, "y");
    keep.forEach(F => RG.lineSeg(gl, F.a, F.b, x, y, { color: SC.accent, w: 1, op: 0.30 }));
    RG.lineSeg(gl, A0, b1, x, y, { color: SC.good, w: 2.2 });
    /* one sample's points, so the reader can see what a single data set looks like */
    const rng2 = ST.rng(seed + 1);
    const ys1 = xs.map(v => A0 + b1 * v + sig * ST.randn(rng2));
    RG.pts(gl, xs, ys1, x, y, { r: 2.6, op: 0.75, fill: SC.a2 });
    gl.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink).text("24 of the " + B + " data sets");

    /* ── right: the sampling distribution ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + gap},0)`);
    const vals = panel === "b" ? bs : ts;
    const lo = panel === "b" ? b1 - 4.2 * seTheory : -5;
    const hi = panel === "b" ? b1 + 4.2 * seTheory : 5;
    const K = 34, wbin = (hi - lo) / K;
    const bins = ST.histBins(vals, lo, wbin, K);
    const dens = panel === "b"
      ? v => ST.normPdf((v - b1) / seTheory) / seTheory
      : v => ST.tPdf(v, n - 2);
    const xg = ST.linspace(lo, hi, 200).map(v => ({ x: v, y: dens(v) }));
    const x2 = d3.scaleLinear().domain([lo, hi]).range([0, rightW]);
    const y2 = d3.scaleLinear()
      .domain([0, Math.max(d3.max(bins, d => d.density), d3.max(xg, d => d.y)) * 1.12])
      .range([ih, 0]);
    ST.gridY(gr, y2, rightW, 4);
    gr.append("g").selectAll("rect").data(bins).join("rect")
      .attr("x", d => x2(d.x0) + 0.6).attr("y", d => y2(d.density))
      .attr("width", Math.max(1, x2(lo + wbin) - x2(lo) - 1.2))
      .attr("height", d => ih - y2(d.density))
      .attr("fill", SC.accent).attr("fill-opacity", 0.42);
    gr.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(d => x2(d.x)).y(d => y2(d.y))(xg));
    ST.axisB(gr, x2, ih, 5, panel === "b" ? "estimated slope  b̂" : "t = (b̂ − β₁)/SE(b)");
    ST.axisL(gr, y2, 4, "density");
    gr.append("line").attr("x1", x2(panel === "b" ? b1 : 0)).attr("x2", x2(panel === "b" ? b1 : 0))
      .attr("y1", 0).attr("y2", ih).attr("stroke", SC.good).attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "4 4");
    gr.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink)
      .text(panel === "b" ? "b̂ against Normal(β₁, σ²/S_xx)" : "t against Student t on n − 2 df");

    const err = Math.abs(sdb - seTheory) / seTheory;
    out.innerHTML =
      `n = ${n} · <span class="keep">σ</span> = ${ST.fmt(sig, 1)} · S_xx = <b>${ST.fmt(Sxx, 1)}</b> · true <span class="keep">β</span>₁ = ${ST.fmt(b1, 1)}<br>`
      + `formula: SE(b) = <span class="keep">σ</span>/√S_xx = ${ST.fmt(sig, 1)}/${ST.fmt(Math.sqrt(Sxx), 3)} = <b>${ST.fmt(seTheory, 5)}</b> &nbsp;·&nbsp; `
      + `${B} simulated slopes: mean <b>${ST.fmt(mb, 4)}</b> (unbiased), SD <b>${ST.fmt(sdb, 5)}</b> `
      + `— ${ST.pct(err, 1)} from the formula, against a Monte-Carlo error of about ${ST.pct(1 / Math.sqrt(2 * (B - 1)), 1)}.<br>`
      + `Widening the spread of x multiplies S_xx by the square of the factor, so SE(b) falls in direct proportion — `
      + `doubling the spread of x is worth as much as quadrupling n.`;
  }

  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(1); draw(); };
  eW.oninput = () => { eWv.textContent = (+eW.value).toFixed(2); draw(); };
  eB.oninput = () => { eBv.textContent = (+eB.value).toFixed(1); draw(); };
  eP.onchange = draw;
  document.getElementById("sl-run").onclick = () => { seed = (seed * 43 + 17) % 100000; draw(); };
  draw();
})();

/* ─────────────── 8 · the two bands, with a live coverage check ─────────────── */
(function () {
  const svg = d3.select("#band-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 480;
  const eN = document.getElementById("bd-n"), eNv = document.getElementById("bd-nv");
  const eS = document.getElementById("bd-s"), eSv = document.getElementById("bd-sv");
  const eL = document.getElementById("bd-lvl");
  const eT = document.getElementById("bd-true");
  const out = document.getElementById("band-readout");
  const A0 = 6, B1 = 3, XLO = 2, XHI = 18;
  let seed = 1212, cov = null;
  const state = { xp: 14 };

  /* the design: for n = 12 this is exactly the running example's x-values */
  function design(n) {
    if (n === 12) return RG.RUNX.slice();
    return ST.linspace(XLO, XHI, n);
  }
  function sample(n, sig, sd) {
    const xs = design(n), r = ST.rng(sd);
    const ys = (n === 12 && sig === 5 && sd === 1212)
      ? RG.RUNY.slice()                       // the exact twelve points the prose uses
      : xs.map(v => A0 + B1 * v + sig * ST.randn(r));
    return { x: xs, y: ys };
  }

  function draw() {
    const n = +eN.value, sig = +eS.value, lvl = +eL.value;
    const d = sample(n, sig, seed);
    const F = RG.fit(d.x, d.y);
    const tc = ST.tQuant(1 - (1 - lvl) / 2, n - 2);
    const vAt = xp => 1 / n + (xp - F.xb) * (xp - F.xb) / F.Sxx;

    const f = ST.frame(svg, W, H, { l: 52, r: 16, t: 24, b: 104 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLinear().domain([XLO - 2, XHI + 9]).range([0, iw]);
    const gridx = ST.linspace(XLO - 2, XHI + 9, 160);
    /* size the vertical axis from the widest band, so no path is drawn off-frame */
    let ylo = Math.min(0, d3.min(d.y)), yhi = d3.max(d.y);
    gridx.forEach(v => {
      const half = tc * F.se * Math.sqrt(1 + 1 / n + (v - F.xb) * (v - F.xb) / F.Sxx);
      ylo = Math.min(ylo, F.pred(v) - half);
      yhi = Math.max(yhi, F.pred(v) + half);
    });
    const padY = (yhi - ylo) * 0.04;
    const y = d3.scaleLinear().domain([ylo - padY, yhi + padY]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);
    ST.axisB(g, x, ih, 8, "x");
    ST.axisL(g, y, 5, "y");

    const band = (mult, col, op) => {
      const up = gridx.map(v => ({ x: v, y: F.pred(v) + tc * F.se * Math.sqrt(mult + vAt(v)) }));
      const dn = gridx.map(v => ({ x: v, y: F.pred(v) - tc * F.se * Math.sqrt(mult + vAt(v)) }));
      g.append("path").attr("fill", col).attr("fill-opacity", op)
        .attr("d", d3.area().x(p => x(p.x)).y0((p, i) => y(dn[i].y)).y1(p => y(p.y))(up));
      [up, dn].forEach(arr => g.append("path").attr("fill", "none").attr("stroke", col)
        .attr("stroke-width", 1.2).attr("stroke-opacity", 0.8)
        .attr("d", d3.line().x(p => x(p.x)).y(p => y(p.y))(arr)));
    };
    band(1, SC.a2, 0.10);          // prediction band, drawn first (wider)
    band(0, SC.accent, 0.22);      // confidence band

    if (eT.checked) RG.lineSeg(g, A0, B1, x, y, { color: SC.good, w: 1.6, dash: "6 4" });
    RG.lineSeg(g, F.a, F.b, x, y, { color: SC.ink, w: 2.2 });
    RG.pts(g, d.x, d.y, x, y, { r: 3.2, op: 0.9, stroke: SC.bg });
    g.append("line").attr("x1", x(F.xb)).attr("x2", x(F.xb)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.muted).attr("stroke-width", 1).attr("stroke-dasharray", "2 5");
    g.append("text").attr("x", x(F.xb) + 5).attr("y", 12).attr("font-size", 10)
      .attr("fill", SC.muted).text("x̄");
    g.append("rect").attr("x", x(XHI)).attr("y", 0).attr("width", Math.max(0, iw - x(XHI)))
      .attr("height", ih).attr("fill", SC.bad).attr("fill-opacity", 0.05);
    g.append("text").attr("x", x(XHI) + 6).attr("y", ih - 8).attr("font-size", 10)
      .attr("fill", SC.bad).text("beyond the data → extrapolation");

    /* the draggable cut */
    const xp = ST.clamp(state.xp, XLO - 2, XHI + 9);
    const yh = F.pred(xp), v = vAt(xp);
    const hc = tc * F.se * Math.sqrt(v), hp = tc * F.se * Math.sqrt(1 + v);
    const gcut = g.append("g").attr("class", "dragpt").style("cursor", "ew-resize");
    gcut.append("rect").attr("x", x(xp) - 11).attr("y", 0).attr("width", 22).attr("height", ih)
      .attr("fill", "transparent");
    gcut.append("line").attr("x1", x(xp)).attr("x2", x(xp)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.violet).attr("stroke-width", 1.6);
    const brack = (half, off, col) => {
      gcut.append("line").attr("x1", x(xp) + off).attr("x2", x(xp) + off)
        .attr("y1", y(yh - half)).attr("y2", y(yh + half))
        .attr("stroke", col).attr("stroke-width", 3.2).attr("stroke-linecap", "round");
    };
    brack(hp, 9, SC.a2);
    brack(hc, -9, SC.accent);
    gcut.append("circle").attr("cx", x(xp)).attr("cy", y(yh)).attr("r", 5)
      .attr("fill", SC.violet).attr("stroke", SC.bg).attr("stroke-width", 1.5);
    gcut.call(d3.drag().on("drag", ev => {
      state.xp = ST.clamp(x.invert(ST.clamp(ev.x, 0, iw)), XLO - 2, XHI + 9);
      draw();
    }));

    ST.legend(g, [
      { label: "confidence band — where the LINE is", color: SC.accent, op: 0.45 },
      { label: "prediction band — where the NEXT POINT is", color: SC.a2, op: 0.35 },
      { label: "the true line", color: SC.good, dash: "5 3" }
    ], 8, 14);

    let covTxt = "";
    if (cov) {
      const sw = iw, gs = g.append("g").attr("transform", `translate(0,${ih + 58})`);
      const bxs = d3.scaleLinear().domain([0.85, 1.0]).range([0, Math.round(sw * 0.55)]);
      const row = (lab, val, yy, col) => {
        gs.append("text").attr("x", -6).attr("y", yy + 4).attr("text-anchor", "end")
          .attr("font-size", 10.5).attr("fill", SC.muted).text(lab);
        gs.append("rect").attr("x", 0).attr("y", yy - 6).attr("width", Math.max(0, bxs(ST.clamp(val, 0.85, 1))))
          .attr("height", 12).attr("rx", 2).attr("fill", col).attr("fill-opacity", 0.8);
        gs.append("text").attr("x", bxs(ST.clamp(val, 0.85, 1)) + 8).attr("y", yy + 4)
          .attr("font-size", 10.5).attr("fill", col).text(ST.pct(val, 2));
      };
      row("CI covers the true mean", cov.ci, 0, SC.accent);
      row("PI covers a new y", cov.pi, 22, SC.a2);
      gs.append("line").attr("x1", bxs(lvl)).attr("x2", bxs(lvl)).attr("y1", -10).attr("y2", 32)
        .attr("stroke", SC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3");
      gs.append("text").attr("x", bxs(lvl)).attr("y", 46).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.good).text("nominal " + ST.pct(lvl, 0));
      covTxt = `<br>Coverage over ${cov.B} fresh data sets at x = ${ST.fmt(cov.xp, 1)}: `
        + `the confidence interval contained the true mean <b>${ST.pct(cov.ci, 2)}</b> of the time and the `
        + `prediction interval contained a new observation <b>${ST.pct(cov.pi, 2)}</b> of the time, against a nominal ${ST.pct(lvl, 0)}.`;
    }

    out.innerHTML =
      `n = ${n} · s_e = <b>${ST.fmt(F.se, 3)}</b> · S_xx = ${ST.fmt(F.Sxx, 1)} · t* = ${ST.fmt(tc, 4)} on ${n - 2} df<br>`
      + `at x = <b>${ST.fmt(xp, 2)}</b>: ŷ = <b>${ST.fmt(yh, 2)}</b> · `
      + `confidence (${ST.pct(lvl, 0)}) <b>(${ST.fmt(yh - hc, 2)}, ${ST.fmt(yh + hc, 2)})</b> half-width ${ST.fmt(hc, 3)} · `
      + `prediction <b>(${ST.fmt(yh - hp, 2)}, ${ST.fmt(yh + hp, 2)})</b> half-width ${ST.fmt(hp, 3)} · `
      + `ratio <b>${ST.fmt(hp / hc, 3)}</b> (at x̄ it would be √(n+1) = ${ST.fmt(Math.sqrt(n + 1), 3)}).`
      + covTxt;
  }

  function runCoverage() {
    const n = +eN.value, sig = +eS.value, lvl = +eL.value, xp = state.xp;
    const tc = ST.tQuant(1 - (1 - lvl) / 2, n - 2);
    const xs = design(n), r = ST.rng(77777);
    const REP = 4000;
    let ci = 0, pi = 0;
    const truth = A0 + B1 * xp;
    for (let k = 0; k < REP; k++) {
      const ys = xs.map(v => A0 + B1 * v + sig * ST.randn(r));
      const F = RG.fit(xs, ys);
      const v = 1 / n + (xp - F.xb) * (xp - F.xb) / F.Sxx;
      const yh = F.pred(xp);
      if (Math.abs(yh - truth) < tc * F.se * Math.sqrt(v)) ci++;
      const ynew = truth + sig * ST.randn(r);
      if (Math.abs(yh - ynew) < tc * F.se * Math.sqrt(1 + v)) pi++;
    }
    cov = { ci: ci / REP, pi: pi / REP, B: REP, xp: xp };
    draw();
  }

  eN.oninput = () => { eNv.textContent = eN.value; cov = null; draw(); };
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(1); cov = null; draw(); };
  eL.onchange = () => { cov = null; draw(); };
  eT.onchange = draw;
  document.getElementById("bd-cov").onclick = runCoverage;
  document.getElementById("bd-new").onclick = () => { seed = (seed * 47 + 19) % 100000; cov = null; draw(); };
  draw();
})();

/* ─────────────── 9 · residual plots and the transform that flattens them ─────────────── */
(function () {
  const svg = d3.select("#resid-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const eK = document.getElementById("rs-kind"), eT = document.getElementById("rs-tr");
  const eN = document.getElementById("rs-n"), eNv = document.getElementById("rs-nv");
  const out = document.getElementById("resid-readout");
  let seed = 5150;

  /* every generator returns strictly positive x and y so that logs are always legal */
  const GEN = {
    ok: (n, r) => {
      const x = [], y = [];
      for (let i = 0; i < n; i++) { const v = 1 + 19 * r(); x.push(v); y.push(8 + 2.2 * v + 3.2 * ST.randn(r)); }
      return { x, y, why: "an honest linear cloud with constant spread — the control case" };
    },
    curve: (n, r) => {
      /* √y is linear in x, so y is a quadratic in x: fitting y on x bends the residuals */
      const x = [], y = [];
      for (let i = 0; i < n; i++) {
        const v = 1 + 19 * r();
        const u = 3 + 0.9 * v + 1.0 * ST.randn(r);
        x.push(v); y.push(Math.max(0.2, u * u));
      }
      return { x, y, why: "√y is linear in x, so fitting y on x bends the residuals into an arc" };
    },
    exp: (n, r) => {
      const x = [], y = [];
      for (let i = 0; i < n; i++) { const v = 0.4 + 9.6 * r(); x.push(v); y.push(3 * Math.exp(0.42 * v) * Math.exp(0.20 * ST.randn(r))); }
      return { x, y, why: "exponential growth with proportional error — log y on x is exactly linear" };
    },
    fan: (n, r) => {
      const x = [], y = [];
      for (let i = 0; i < n; i++) { const v = 1 + 19 * r(); x.push(v); y.push(10 + 2.0 * v + (0.6 + 0.55 * v) * ST.randn(r)); }
      return { x, y, why: "the line is right and the spread is not — every standard error is wrong" };
    },
    power: (n, r) => {
      const x = [], y = [];
      for (let i = 0; i < n; i++) {
        const v = Math.exp(Math.log(20) + (Math.log(60000) - Math.log(20)) * r());
        x.push(v); y.push(0.9 * Math.pow(v, 0.82) * Math.exp(0.33 * ST.randn(r)));
      }
      return { x, y, why: "a power law over three orders of magnitude — log–log fixes shape AND spread at once" };
    }
  };

  const TR = {
    none: { fx: v => v, fy: v => v, lx: "x", ly: "y" },
    sqrty: { fx: v => v, fy: v => Math.sqrt(Math.max(v, 1e-9)), lx: "x", ly: "√y" },
    logy: { fx: v => v, fy: v => Math.log(Math.max(v, 1e-9)), lx: "x", ly: "log y" },
    loglog: { fx: v => Math.log(Math.max(v, 1e-9)), fy: v => Math.log(Math.max(v, 1e-9)), lx: "log x", ly: "log y" }
  };

  /* curvature score: fit a quadratic in x to the residuals and report how much of
     their variance it removes. Heteroscedasticity score: correlation between x and
     |residual|. Both are descriptive numbers, not tests.                          */
  function scores(x, e) {
    const n = x.length, xb = ST.mean(x);
    const z = x.map(v => v - xb), z2 = z.map(v => v * v);
    const c2 = ST.corr(z2, e);
    const curvature = isFinite(c2) ? c2 * c2 : 0;
    const het = ST.corr(x, e.map(Math.abs));
    return { curv: curvature, het: isFinite(het) ? het : 0 };
  }

  function draw() {
    const kind = eK.value, tr = TR[eT.value], n = +eN.value;
    const raw = GEN[kind](n, ST.rng(seed));
    const X = raw.x.map(tr.fx), Y = raw.y.map(tr.fy);
    const F = RG.fit(X, Y);
    const E = X.map((v, i) => Y[i] - F.pred(v));
    const sc = scores(X, E);

    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 26, b: 46 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gap = 46, leftW = Math.round((iw - gap) / 2), rightW = iw - leftW - gap;

    /* ── left: the fit on the current scale ── */
    const gl = g.append("g");
    const xd = d3.extent(X), yd = d3.extent(Y);
    const px = (xd[1] - xd[0]) * 0.06 || 1, py = (yd[1] - yd[0]) * 0.08 || 1;
    const x = d3.scaleLinear().domain([xd[0] - px, xd[1] + px]).range([0, leftW]);
    const y = d3.scaleLinear().domain([yd[0] - py, yd[1] + py]).range([ih, 0]);
    ST.gridY(gl, y, leftW, 4);
    ST.axisB(gl, x, ih, 5, tr.lx);
    ST.axisL(gl, y, 4, tr.ly);
    RG.pts(gl, X, Y, x, y, { r: 2.4, op: 0.5 });
    RG.lineSeg(gl, F.a, F.b, x, y, { color: SC.a2, w: 2.2 });
    gl.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink).text("the fit, on this scale");

    /* ── right: the residual plot, with a local mean trace and a spread envelope ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + gap},0)`);
    const em = d3.max(E, Math.abs) * 1.12 || 1;
    const x2 = d3.scaleLinear().domain(x.domain()).range([0, rightW]);
    const y2 = d3.scaleLinear().domain([-em, em]).range([ih, 0]);
    /* the envelope: local mean ± local SD in 10 equal-count slices */
    const idx = X.map((v, i) => i).sort((p, q) => X[p] - X[q]);
    const K = Math.max(4, Math.min(12, Math.floor(n / 12)));
    const per = Math.floor(n / K), env = [];
    for (let k = 0; k < K; k++) {
      const sl = idx.slice(k * per, k === K - 1 ? n : (k + 1) * per);
      if (sl.length < 2) continue;
      const xs = sl.map(i => X[i]), es = sl.map(i => E[i]);
      env.push({ x: ST.mean(xs), m: ST.mean(es), s: ST.sd(es) });
    }
    if (env.length > 1) {
      gr.append("path").attr("fill", SC.muted).attr("fill-opacity", 0.13)
        .attr("d", d3.area().x(p => x2(p.x)).y0(p => y2(p.m - p.s)).y1(p => y2(p.m + p.s))
          .curve(d3.curveMonotoneX)(env));
      gr.append("path").attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 2)
        .attr("d", d3.line().x(p => x2(p.x)).y(p => y2(p.m)).curve(d3.curveMonotoneX)(env));
    }
    gr.append("line").attr("x1", 0).attr("x2", rightW).attr("y1", y2(0)).attr("y2", y2(0))
      .attr("stroke", SC.good).attr("stroke-width", 1.6);
    RG.pts(gr, X, E, x2, y2, { r: 2.4, op: 0.55, fill: SC.a2 });
    ST.axisB(gr, x2, ih, 5, tr.lx);
    ST.axisL(gr, y2, 4, "residual  e");
    gr.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", SC.ink).text("residual plot — want a flat band of constant width");
    ST.legend(gr, [
      { label: "local mean residual", color: SC.violet },
      { label: "± local SD", color: SC.muted, op: 0.35 }
    ], rightW - 118, 14);

    const verdictC = sc.curv < 0.02 ? ["straight", SC.good] : sc.curv < 0.10 ? ["a little bent", SC.a2] : ["clearly curved", SC.bad];
    const verdictH = Math.abs(sc.het) < 0.12 ? ["even", SC.good] : Math.abs(sc.het) < 0.30 ? ["slightly uneven", SC.a2] : ["fanning", SC.bad];

    out.innerHTML =
      `<b>${raw.why}</b><br>`
      + `on the <b>${tr.ly} vs ${tr.lx}</b> scale: R² = <b>${ST.fmt(F.r2, 4)}</b> · s_e = ${ST.fmt(F.se, 4)} · `
      + `slope = ${ST.fmt(F.b, 4)} ± ${ST.fmt(ST.tQuant(0.975, n - 2) * F.seb, 4)}<br>`
      + `curvature score (variance of the residuals explained by a quadratic in x): `
      + `<b style="color:${verdictC[1]}">${ST.fmt(sc.curv, 3)} — ${verdictC[0]}</b> &nbsp;·&nbsp; `
      + `spread score (correlation between x and |e|): `
      + `<b style="color:${verdictH[1]}">${ST.fmt(sc.het, 3)} — ${verdictH[0]}</b><br>`
      + (eT.value === "none"
        ? `Try each transform and watch the two scores. R² alone will not tell you which scale is right — `
        + `on a badly curved fit it can still be above 0.95.`
        : eT.value === "loglog"
          ? `On the log–log scale the slope is an ELASTICITY: a 1% rise in x goes with about ${ST.fmt(F.b, 3)}% in y.`
          : eT.value === "logy"
            ? `On the log scale the slope is a growth rate: each unit of x multiplies y by e^${ST.fmt(F.b, 3)} = ${ST.fmt(Math.exp(F.b), 3)}.`
            : `The square root is the gentlest useful rung of the ladder — the right one for counts.`);
  }

  eK.onchange = draw; eT.onchange = draw;
  eN.oninput = () => { eNv.textContent = eN.value; draw(); };
  document.getElementById("rs-new").onclick = () => { seed = (seed * 53 + 23) % 100000; draw(); };
  draw();
})();

/* ─────────────── 10 · leverage against influence, one draggable point ─────────────── */
(function () {
  const svg = d3.select("#lev-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 500;
  const ePre = document.getElementById("lv-preset");
  const eSq = document.getElementById("lv-sq");
  const out = document.getElementById("lev-readout");

  const BX = RG.LEVX, BY = RG.LEVY;
  const base = RG.fit(BX, BY);                    // b = 1.8470, a = 5.1435, s_e = 1.1520
  const PRESETS = {
    hihi: [24, 35.5],
    hilo: [28, 49.9],
    onl: [24, 49.5],
    out: [5, 30]
  };
  const state = { p: PRESETS.hihi.slice() };

  function draw() {
    const px = state.p[0], py = state.p[1];
    const X = BX.concat([px]), Y = BY.concat([py]);
    const F = RG.fit(X, Y);
    const h = RG.lev(X);
    const hp = h[h.length - 1];
    const e = Y.map((v, i) => v - F.pred(X[i]));
    const ep = e[e.length - 1];
    const std = ep / (F.se * Math.sqrt(Math.max(1e-9, 1 - hp)));
    const cook = (std * std / 2) * hp / Math.max(1e-9, 1 - hp);
    const dslope = F.b - base.b;

    const f = ST.frame(svg, W, H, { l: 48, r: 16, t: 24, b: 118 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLinear().domain([0, 31]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 66]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);
    ST.axisB(g, x, ih, 8, "x");
    ST.axisL(g, y, 5, "y");

    /* the region of the twelve fixed points, for the leverage intuition */
    g.append("rect").attr("x", x(Math.min(...BX))).attr("y", 0)
      .attr("width", x(Math.max(...BX)) - x(Math.min(...BX))).attr("height", ih)
      .attr("fill", SC.accent).attr("fill-opacity", 0.05);
    g.append("line").attr("x1", x(base.xb)).attr("x2", x(base.xb)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.muted).attr("stroke-width", 1).attr("stroke-dasharray", "2 5");
    g.append("text").attr("x", x(base.xb) + 5).attr("y", 12).attr("font-size", 10)
      .attr("fill", SC.muted).text("x̄ of the twelve");

    if (eSq.checked) {
      for (let i = 0; i < X.length; i++) {
        const side = Math.abs(y(Y[i]) - y(F.pred(X[i])));
        const up = Y[i] >= F.pred(X[i]);
        g.append("rect").attr("x", x(X[i])).attr("y", Math.min(y(Y[i]), y(F.pred(X[i]))))
          .attr("width", side).attr("height", side)
          .attr("fill", up ? SC.accent : SC.bad).attr("fill-opacity", 0.10)
          .attr("stroke", up ? SC.accent : SC.bad).attr("stroke-opacity", 0.4);
      }
    }

    RG.lineSeg(g, base.a, base.b, x, y, { color: SC.good, w: 2, dash: "7 4" });
    RG.lineSeg(g, F.a, F.b, x, y, { color: SC.a2, w: 2.4 });
    RG.pts(g, BX, BY, x, y, { r: 3.6, op: 0.9, stroke: SC.bg });

    /* the draggable point */
    const gp = g.append("g").attr("class", "dragpt").style("cursor", "grab");
    gp.append("circle").attr("cx", x(px)).attr("cy", y(py)).attr("r", 9)
      .attr("fill", SC.bad).attr("fill-opacity", 0.85)
      .attr("stroke", SC.bg).attr("stroke-width", 2);
    gp.append("text").attr("x", x(px)).attr("y", y(py) - 16).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.bad).text("drag me");
    gp.call(d3.drag().on("drag", ev => {
      state.p = [ST.clamp(x.invert(ST.clamp(ev.x, 0, iw)), 0.5, 30.5),
                 ST.clamp(y.invert(ST.clamp(ev.y, 0, ih)), 0.5, 65)];
      draw();
    }));

    ST.legend(g, [
      { label: "fit WITHOUT the red point — b = " + ST.fmt(base.b, 3), color: SC.good, dash: "5 3" },
      { label: "fit WITH it — b = " + ST.fmt(F.b, 3), color: SC.a2 }
    ], 8, 14);

    /* the slope bar */
    const strip = g.append("g").attr("transform", `translate(0,${ih + 48})`);
    const barW = Math.round(iw * 0.5);
    const bs = d3.scaleLinear().domain([0, Math.max(3, base.b * 1.6, F.b * 1.25)]).range([0, barW]);
    const row = (lab, val, yy, col) => {
      strip.append("text").attr("x", -6).attr("y", yy + 4).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.muted).text(lab);
      strip.append("rect").attr("x", 0).attr("y", yy - 6).attr("width", Math.max(0, bs(Math.max(0, val))))
        .attr("height", 12).attr("rx", 2).attr("fill", col).attr("fill-opacity", 0.8);
      strip.append("text").attr("x", bs(Math.max(0, val)) + 8).attr("y", yy + 4)
        .attr("font-size", 10.5).attr("fill", col).text(ST.fmt(val, 3));
    };
    row("slope without", base.b, 0, SC.good);
    row("slope with", F.b, 22, SC.a2);

    /* leverage / residual / Cook readout panel */
    const gp2 = g.append("g").attr("transform", `translate(${barW + 66},${ih + 34})`);
    const items = [
      ["leverage h", ST.fmt(hp, 3) + (hp > 4 / X.length ? "   > 4/n = " + ST.fmt(4 / X.length, 3) + " ✱" : "")],
      ["raw residual", ST.fmt(ep, 2)],
      ["standardised residual", ST.fmt(std, 2) + (Math.abs(std) > 2 ? "  ✱" : "")],
      ["Cook's D", ST.fmt(cook, 2) + (cook > 1 ? "  ✱" : "")]
    ];
    items.forEach((it, i) => {
      gp2.append("text").attr("x", 0).attr("y", i * 15).attr("font-size", 10.5)
        .attr("fill", SC.muted).text(it[0]);
      gp2.append("text").attr("x", 150).attr("y", i * 15).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.ink).text(it[1]);
    });

    const pct = 100 * dslope / base.b;
    const verdict = Math.abs(pct) < 2
      ? ["not influential", SC.good]
      : (Math.abs(std) > 2 ? ["influential, and the residual plot can see it", SC.a2]
        : ["influential, and the residual plot CANNOT see it", SC.bad]);

    out.innerHTML =
      `the point at (${ST.fmt(px, 1)}, ${ST.fmt(py, 1)}): leverage h = <b>${ST.fmt(hp, 3)}</b> `
      + `(average is 2/n = ${ST.fmt(2 / X.length, 3)}), raw residual <b>${ST.fmt(ep, 2)}</b>, `
      + `standardised <b>${ST.fmt(std, 2)}</b>, Cook's D <b>${ST.fmt(cook, 2)}</b>.<br>`
      + `Including it moves the slope from <b>${ST.fmt(base.b, 3)}</b> to <b>${ST.fmt(F.b, 3)}</b> — `
      + `a change of <b style="color:${verdict[1]}">${ST.fmt(pct, 1)}%</b> — so it is `
      + `<b style="color:${verdict[1]}">${verdict[0]}</b>.<br>`
      + `Drag it to the far right and slide it up and down: the residual stays small while the line follows it. `
      + `That is the case no residual plot can flag, and the only way to find it is to refit without the point.`;
  }

  ePre.onchange = () => { state.p = PRESETS[ePre.value].slice(); draw(); };
  eSq.onchange = draw;
  document.getElementById("lv-reset").onclick = () => {
    ePre.value = "hihi"; state.p = PRESETS.hihi.slice(); draw();
  };
  draw();
})();
