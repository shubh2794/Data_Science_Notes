/* categorical.viz.js — the fourteen visualizations on math/statistics/categorical.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

      1  #tab-svg   a two-way table as a mosaic, with the percentaging switchable between
                    column, row, joint and raw counts — independence is a flat dividing line
      2  #chid-svg  the chi-square density with a live df, the observed statistic and its
                    right tail, the 5% critical value, and the two normal approximations
      3  #mos-svg   the ship's table mosaic with the expected split ghosted behind it, and
                    a slider that pulls the observed table continuously toward the null
      4  #lab-svg   the laboratory: an editable contingency table driving chi-square, G²,
                    Yates, Fisher, df, p, residuals and effect sizes, all live
      5  #res-svg   residuals as a cell map against their N(0,1) reference, with the
                    measured probability that SOME cell exceeds the threshold by table size
      6  #lvl-svg   the EXACT type I error of four tests against n, computed by enumeration
                    over two independent binomials — the sawtooth is real
      7  #fis-svg   Fisher's reference set: every table with the observed margins, its
                    hypergeometric probability, and the tail that is literally the p-value
      8  #gvc-svg   chi-square, G² and Fisher on one table as n shrinks at fixed proportions
      9  #mcn-svg   paired against unpaired: the discordant pairs, their exact binomial
                    null, and the p-value the wrong test would have reported
     10  #eff-svg   the unit square of (p₀, p₁) with contours of constant OR, RR and RD,
                    and the OR/RR overstatement shaded behind them
     11  #orci-svg  the sampling distribution of ln OR̂, its Wald interval, measured
                    coverage, and the fraction of studies with no computable estimate
     12  #cc-svg    one population, two designs: the odds ratio pinned to its true value
                    while the naive risk ratio slides with the control-to-case ratio
     13  #simp-svg  Simpson's paradox: both strata fixed, the allocation sliding, and the
                    aggregate conclusion flipping at a computed threshold
     14  #big-svg   p-value against n at a fixed association — significance bought with data

   Every number these draw is recomputed from the data they generate, so the pictures
   and the prose cannot drift apart.                                                  */

/* ══════════ page-local helpers (deliberately NOT in stats-viz.js) ══════════ */
const CT = (function () {

  /* ── the tables the page keeps returning to ──────────────────────────── */

  /* the running 2 × 4: margins 325/285/706/913 and 713/1516, n = 2229.
     χ² = 192.339864 on 3 df, G² = 182.417978, Cramér V = 0.293750. */
  const TITANIC = {
    name: "survival by ticket class",
    rows: ["Survived", "Died"], cols: ["First", "Second", "Third", "Crew"],
    O: [[202, 118, 178, 215], [123, 167, 528, 698]]
  };
  /* §07's survey: χ² = 9.307568 on 1 df, p = 0.002282, OR = 1.735040 */
  const IND = {
    name: "a survey, two variables",
    rows: ["Group 1", "Group 2"], cols: ["Pref A", "Pref B"],
    O: [[132, 118], [98, 152]]
  };
  /* the small table where the tests disagree: min(E) = 5.5 and yet
     χ² p = 0.040520 against Fisher p = 0.099533 */
  const SMALL = {
    name: "a small 2 × 2",
    rows: ["Group 1", "Group 2"], cols: ["Success", "Failure"],
    O: [[9, 3], [4, 8]]
  };
  const SPARSE = {
    name: "a sparse 3 × 3",
    rows: ["A", "B", "C"], cols: ["x", "y", "z"],
    O: [[6, 2, 1], [2, 5, 2], [1, 2, 6]]
  };
  /* an exactly independent table: every cell is R·C/n on the nose */
  const NOASSOC = {
    name: "no association at all",
    rows: ["Yes", "No"], cols: ["P", "Q", "R"],
    O: [[60, 90, 150], [40, 60, 100]]
  };
  const SKEW = {
    name: "strong association, unequal margins",
    rows: ["Yes", "No"], cols: ["P", "Q", "R", "S"],
    O: [[90, 40, 30, 12], [10, 60, 170, 588]]
  };
  /* the goodness-of-fit example: n = 410, χ² = 8.566984 on 5 df, p = 0.127633 */
  const MM = {
    name: "six sweet colours",
    cats: ["Blue", "Orange", "Green", "Yellow", "Red", "Brown"],
    p: [0.24, 0.20, 0.16, 0.14, 0.13, 0.13],
    O: [85, 79, 56, 64, 58, 68]
  };
  /* §13's paired table: McNemar χ² = 4.8, exact 0.042772, wrong unpaired 0.0833 */
  const PAIRED = { a: 45, b: 21, c: 9, d: 25 };
  /* §17's reversal: +10 and +5 points within strata, −3.75 pooled */
  const SIMPSON = {
    strata: ["Severe", "Mild"],
    tr: [[210, 300], [90, 100]],       // [recovered, total] for treatment
    ct: [[60, 100], [255, 300]]        // ditto for control
  };

  /* ── table arithmetic; everything below takes a plain r × c array ────── */

  const rowSums = O => O.map(r => r.reduce((s, x) => s + x, 0));
  const colSums = O => O[0].map((_, j) => O.reduce((s, r) => s + r[j], 0));
  const total = O => O.reduce((s, r) => s + r.reduce((t, x) => t + x, 0), 0);

  /* The chi-square SURVIVAL function, computed as the upper incomplete gamma
     directly. `1 - ST.chi2Cdf(x, k)` cancels catastrophically once the cdf is within
     1e-16 of 1 and returns exactly 0 - which would print the ship's p-value, a
     genuine 1.9e-41, as "0". Below the peak the series form is stable and the
     subtraction is harmless; above it, Lentz's continued fraction for Q(a, x) is
     evaluated on its own and underflows only near 1e-308.                        */
  function chi2Sf(x, k) {
    if (!(x > 0)) return 1;
    const a = k / 2, z = x / 2;
    if (z < a + 1) return 1 - ST.gammaP(a, z);
    let b = z + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (let i = 1; i < 2000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return Math.exp(-z + a * Math.log(z) - ST.lnGamma(a)) * h;
  }

  function expected(O) {
    const R = rowSums(O), C = colSums(O), n = total(O);
    return O.map((r, i) => r.map((_, j) => n > 0 ? R[i] * C[j] / n : 0));
  }

  /* Everything a table can tell you, in one pass. Residual conventions:
       std = (O − E)/√E                       — squares sum to χ² exactly
       adj = (O − E)/√(E(1−Rᵢ/n)(1−Cⱼ/n))     — approximately N(0, 1)          */
  function stats(O) {
    const R = rowSums(O), C = colSums(O), n = total(O);
    const r = O.length, c = O[0].length;
    const E = expected(O);
    let chi2 = 0, g2 = 0, yates = 0, minE = Infinity;
    const contrib = [], std = [], adj = [];
    for (let i = 0; i < r; i++) {
      contrib.push([]); std.push([]); adj.push([]);
      for (let j = 0; j < c; j++) {
        const e = E[i][j], o = O[i][j], d = o - e;
        if (e > 0 && e < minE) minE = e;
        if (e > 0) {
          const ct = d * d / e;
          chi2 += ct; contrib[i].push(ct);
          std[i].push(d / Math.sqrt(e));
          const v = e * (1 - R[i] / n) * (1 - C[j] / n);
          adj[i].push(v > 1e-12 ? d / Math.sqrt(v) : 0);
          const y = Math.max(0, Math.abs(d) - 0.5);
          yates += y * y / e;
          if (o > 0) g2 += 2 * o * Math.log(o / e);
        } else { contrib[i].push(0); std[i].push(0); adj[i].push(0); }
      }
    }
    const df = (r - 1) * (c - 1);
    const ok = isFinite(minE) && minE > 0 && df > 0;
    const V = (n > 0 && ok) ? Math.sqrt(chi2 / (n * Math.min(r - 1, c - 1))) : NaN;
    return {
      O: O, E: E, R: R, C: C, n: n, r: r, c: c, df: df,
      chi2: chi2, p: ok ? chi2Sf(chi2, df) : NaN,
      g2: g2, gp: ok ? chi2Sf(g2, df) : NaN,
      yates: yates, yp: (ok && r === 2 && c === 2) ? chi2Sf(yates, 1) : NaN,
      minE: isFinite(minE) ? minE : 0, contrib: contrib, std: std, adj: adj,
      phi: (n > 0 && ok) ? Math.sqrt(chi2 / n) : NaN, V: V,
      crit: ok ? ST.chi2Quant(0.95, df) : NaN
    };
  }

  /* goodness-of-fit against a fully specified p, with the matching residuals */
  function gof(O, p) {
    const n = O.reduce((s, x) => s + x, 0), k = O.length;
    const E = p.map(q => q * n);
    let chi2 = 0, g2 = 0;
    const contrib = [], std = [], adj = [];
    for (let j = 0; j < k; j++) {
      const d = O[j] - E[j];
      contrib.push(d * d / E[j]); chi2 += d * d / E[j];
      std.push(d / Math.sqrt(E[j]));
      adj.push(d / Math.sqrt(E[j] * (1 - p[j])));
      if (O[j] > 0) g2 += 2 * O[j] * Math.log(O[j] / E[j]);
    }
    const df = k - 1;
    return {
      O: O, E: E, n: n, df: df, chi2: chi2, p: chi2Sf(chi2, df),
      g2: g2, gp: chi2Sf(g2, df), contrib: contrib, std: std, adj: adj,
      minE: Math.min.apply(null, E), crit: ST.chi2Quant(0.95, df)
    };
  }

  /* ── the hypergeometric, built in LOG space so large margins are safe ─
     A direct factorial form overflows a double past 170!, so every term is a
     log-binomial from ST.lnChoose and the exponential is taken once at the end.
     Checked against published hypergeometric values for margins up to 100 000,
     where the naive form returns NaN and this returns 11 correct digits.      */
  function hyperPmf(k, N, K, nDraw) {
    if (k < 0 || k > K || nDraw - k < 0 || nDraw - k > N - K) return 0;
    return Math.exp(ST.lnChoose(K, k) + ST.lnChoose(N - K, nDraw - k) - ST.lnChoose(N, nDraw));
  }

  /* the whole reference set for a 2 × 2 with the observed margins, plus every
     tail convention §11 lists. side: "two" | "dbl" | "right" | "left" | "midp" */
  function fisher(a, b, c, d, side) {
    const R1 = a + b, R2 = c + d, C1 = a + c, N = R1 + R2;
    const lo = Math.max(0, C1 - R2), hi = Math.min(R1, C1);
    const ks = [], pm = [];
    for (let k = lo; k <= hi; k++) { ks.push(k); pm.push(hyperPmf(k, N, C1, R1)); }
    const tot = pm.reduce((s, x) => s + x, 0) || 1;
    for (let i = 0; i < pm.length; i++) pm[i] /= tot;
    const idx = a - lo;
    const obs = (idx >= 0 && idx < pm.length) ? pm[idx] : 0;
    let left = 0, right = 0, pts = 0, strict = 0, ties = 0;
    for (let i = 0; i < ks.length; i++) {
      if (ks[i] <= a) left += pm[i];
      if (ks[i] >= a) right += pm[i];
      if (pm[i] <= obs * (1 + 1e-9)) pts += pm[i];
      if (pm[i] < obs * (1 - 1e-9)) strict += pm[i]; else if (pm[i] <= obs * (1 + 1e-9)) ties += pm[i];
    }
    const dbl = Math.min(1, 2 * Math.min(left, right));
    const midp = strict + 0.5 * ties;
    let p, keep;
    switch (side || "two") {
      case "right": p = right; keep = i => ks[i] >= a; break;
      case "left": p = left; keep = i => ks[i] <= a; break;
      case "dbl": {
        p = dbl;
        const rightSide = right <= left;
        keep = i => rightSide ? ks[i] >= a : ks[i] <= a;
        break;
      }
      case "midp": p = midp; keep = i => pm[i] <= obs * (1 + 1e-9); break;
      default: p = pts; keep = i => pm[i] <= obs * (1 + 1e-9);
    }
    return { ks: ks, pm: pm, lo: lo, hi: hi, idx: idx, obs: obs, p: p, keep: keep,
             left: left, right: right, pts: pts, dbl: dbl, midp: midp,
             mean: R1 * C1 / N, N: N, R1: R1, R2: R2, C1: C1, C2: N - C1 };
  }

  /* the three summaries of a 2 × 2, with the log-OR interval of §15 */
  function measures(a, b, c, d, corr) {
    const k = corr ? 0.5 : 0;
    const A = a + k, B = b + k, C = c + k, D = d + k;
    const p1 = (a + b) > 0 ? a / (a + b) : NaN, p0 = (c + d) > 0 ? c / (c + d) : NaN;
    const or = (B * C) > 0 ? (A * D) / (B * C) : (A * D > 0 ? Infinity : NaN);
    const se = Math.sqrt(1 / A + 1 / B + 1 / C + 1 / D);
    const l = Math.log(or), z = 1.959964;
    const seRR = Math.sqrt(b / (a * (a + b)) + d / (c * (c + d)));
    return {
      p1: p1, p0: p0, rd: p1 - p0, rr: p0 > 0 ? p1 / p0 : NaN, or: or,
      lnor: l, se: se, lo: Math.exp(l - z * se), hi: Math.exp(l + z * se),
      wald: l / se, nnt: Math.abs(1 / (p1 - p0)),
      rrLo: Math.exp(Math.log(p1 / p0) - z * seRR), rrHi: Math.exp(Math.log(p1 / p0) + z * seRR)
    };
  }

  /* McNemar, both approximations and the exact conditional binomial */
  function mcnemar(a, b, c, d) {
    const m = b + c, n = a + b + c + d;
    const chi = m > 0 ? (b - c) * (b - c) / m : 0;
    const cc = m > 0 ? Math.pow(Math.max(0, Math.abs(b - c) - 1), 2) / m : 0;
    const ex = m > 0 ? Math.min(1, 2 * ST.binomCdf(Math.min(b, c), m, 0.5)) : 1;
    const m1 = (a + b) / n, m2 = (a + c) / n;
    /* the WRONG analysis: an independence test on the two marginal proportions */
    const wrongTab = [[a + b, n - (a + b)], [a + c, n - (a + c)]];
    const w = stats(wrongTab);
    return { m: m, chi: chi, p: m > 0 ? chi2Sf(chi, 1) : 1,
             cc: cc, ccp: m > 0 ? chi2Sf(cc, 1) : 1, exact: ex,
             m1: m1, m2: m2, wrong: w.chi2, wrongP: w.p, n: n };
  }

  /* the pooled two-proportion z of §12, for the χ² = z² demonstration */
  function twoPropZ(a, b, c, d) {
    const n1 = a + b, n2 = c + d, pp = (a + c) / (n1 + n2);
    const den = Math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2));
    return den > 0 ? (a / n1 - c / n2) / den : 0;
  }

  /* ── drawing helpers ─────────────────────────────────────────────────── */

  /* mosaic geometry: column j gets width ∝ Cⱼ (or equal widths), split
     vertically by the conditional distribution within that column. */
  function mosaic(O, W, H, opt) {
    const o = Object.assign({ widths: true, gapX: 4, gapY: 2 }, opt || {});
    const C = colSums(O), n = total(O), c = O[0].length, r = O.length;
    const avail = W - (c - 1) * o.gapX;
    const tiles = [];
    let x = 0;
    for (let j = 0; j < c; j++) {
      const w = o.widths ? (n > 0 ? avail * C[j] / n : avail / c) : avail / c;
      let y = 0;
      const inner = H - (r - 1) * o.gapY;
      for (let i = 0; i < r; i++) {
        const h = C[j] > 0 ? inner * O[i][j] / C[j] : inner / r;
        tiles.push({ i: i, j: j, x: x, y: y, w: w, h: h, o: O[i][j],
                     frac: C[j] > 0 ? O[i][j] / C[j] : 0 });
        y += h + o.gapY;
      }
      x += w + o.gapX;
    }
    return { tiles: tiles, C: C, n: n, W: avail + (c - 1) * o.gapX };
  }

  /* a diverging colour for a residual: red below, green above, grey near 0 */
  function residColour(v, cap) {
    const t = ST.clamp(Math.abs(v) / (cap || 4), 0, 1);
    return d3.interpolateRgb(SC.muted, v >= 0 ? SC.good : SC.bad)(0.25 + 0.75 * t);
  }
  const residOp = (v, cap) => 0.22 + 0.62 * ST.clamp(Math.abs(v) / (cap || 4), 0, 1);

  /* a compact p-value string, identical in every figure on the page */
  const pstr = p => !isFinite(p) ? "—"
    : (p <= 0 ? "&lt; 1e-300" : (p < 1e-3 ? p.toExponential(2) : ST.fmt(p, 4)));
  /* an integer with thin spaces every three digits, for readable margins */
  const grp = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  /* a count that may be astronomically large: exact below 2^53, otherwise scientific */
  const bigN = x => (isFinite(x) && x < 1e15) ? grp(x) : x.toExponential(3);

  /* a labelled value chip inside an svg — used by half the figures */
  function chip(g, x, y, label, value, colour, w) {
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.append("rect").attr("x", 0).attr("y", 0).attr("width", w || 118).attr("height", 34)
      .attr("rx", 6).attr("fill", SC.panel2).attr("stroke", SC.line);
    gg.append("text").attr("x", 8).attr("y", 13).attr("font-size", 9)
      .attr("fill", SC.muted).attr("letter-spacing", 0.4).text(label);
    gg.append("text").attr("x", 8).attr("y", 27).attr("font-size", 13)
      .attr("font-weight", 600).attr("fill", colour || SC.ink).text(value);
    return gg;
  }

  /* a titled sub-frame inside a bigger <g> */
  function sub(g, x, y, title) {
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    if (title) gg.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5)
      .attr("fill", SC.ink).attr("font-weight", 600).text(title);
    return gg;
  }

  /* a vertical rule with an optional label at the top */
  function rule(g, xp, ih, col, lbl, dash, dy) {
    g.append("line").attr("x1", xp).attr("x2", xp).attr("y1", 0).attr("y2", ih)
      .attr("stroke", col).attr("stroke-width", 1.6).attr("stroke-dasharray", dash || null);
    if (lbl) g.append("text").attr("x", xp).attr("y", (dy === undefined ? -3 : dy))
      .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", col).text(lbl);
  }

  /* draw an r × c grid of numbers with per-cell fill; returns the cell selection
     so a caller can attach a drag behaviour to it (figure 4 does).            */
  function gridTable(g, O, x0, y0, cw, ch, opt) {
    const o = Object.assign({ rows: null, cols: null, fill: null, text: null,
                              sub: null, head: true }, opt || {});
    const r = O.length, c = O[0].length;
    const gg = g.append("g").attr("transform", `translate(${x0},${y0})`);
    if (o.head && o.cols) o.cols.forEach((lb, j) => gg.append("text")
      .attr("x", (j + 1) * cw + cw / 2).attr("y", ch * 0.62).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", SC.accent).text(lb));
    if (o.rows) o.rows.forEach((lb, i) => gg.append("text")
      .attr("x", cw - 8).attr("y", (i + 1) * ch + ch / 2 + 4).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.accent).text(lb));
    const data = [];
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) data.push({ i: i, j: j, v: O[i][j] });
    const cell = gg.selectAll("g.cell").data(data).join("g").attr("class", "cell")
      .attr("transform", d => `translate(${(d.j + 1) * cw},${(d.i + 1) * ch})`);
    cell.append("rect").attr("width", cw - 3).attr("height", ch - 3).attr("rx", 4)
      .attr("fill", d => o.fill ? o.fill(d.i, d.j) : SC.panel2)
      .attr("stroke", SC.line);
    cell.append("text").attr("x", (cw - 3) / 2).attr("y", o.sub ? ch / 2 - 1 : ch / 2 + 4)
      .attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 600)
      .attr("fill", SC.ink).text(d => o.text ? o.text(d.i, d.j) : d.v);
    if (o.sub) cell.append("text").attr("x", (cw - 3) / 2).attr("y", ch / 2 + 12)
      .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", SC.muted)
      .text(d => o.sub(d.i, d.j));
    return { g: gg, cell: cell };
  }

  return {
    TITANIC, IND, SMALL, SPARSE, NOASSOC, SKEW, MM, PAIRED, SIMPSON,
    rowSums, colSums, total, expected, stats, gof,
    chi2Sf, hyperPmf, fisher, measures, mcnemar, twoPropZ,
    mosaic, residColour, residOp, pstr, grp, bigN, chip, sub, rule, gridTable
  };
})();

/* ─────────────── 1 · the mosaic, and three ways to percentage ─────────────── */
(function () {
  const svg = d3.select("#tab-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 440;
  const eD = document.getElementById("tb-data"), eM = document.getElementById("tb-mode");
  const eMg = document.getElementById("tb-marg"), eW = document.getElementById("tb-width");
  const out = document.getElementById("tab-readout");
  const SETS = { titanic: CT.TITANIC, ind: CT.IND, null: CT.NOASSOC, skew: CT.SKEW };

  function draw() {
    const T = SETS[eD.value], O = T.O, mode = eM.value;
    const s = CT.stats(O), n = s.n;
    const f = ST.frame(svg, W, H, { l: 12, r: 12, t: 14, b: 12 });
    const g = f.g;

    /* ── left: the mosaic ───────────────────────────────────────────── */
    const MW = 400, MH = 320;
    const gm = CT.sub(g, 62, 34, "the table as a mosaic");
    const mo = CT.mosaic(O, MW, MH, { widths: eW.checked });
    const pal = [SC.accent, SC.a2, SC.violet, SC.teal, SC.good];

    mo.tiles.forEach(t => {
      gm.append("rect").attr("x", t.x).attr("y", t.y).attr("width", Math.max(0, t.w))
        .attr("height", Math.max(0, t.h)).attr("rx", 2)
        .attr("fill", pal[t.i % pal.length]).attr("fill-opacity", 0.55)
        .attr("stroke", SC.bg).attr("stroke-width", 0.7);
      if (t.w > 34 && t.h > 16) gm.append("text").attr("x", t.x + t.w / 2).attr("y", t.y + t.h / 2 + 4)
        .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", SC.bg)
        .attr("font-weight", 600).text(ST.pct(t.frac, 1));
    });
    /* the overall marginal, which independence says every column should sit at */
    if (eMg.checked) {
      const q = s.R[0] / n, yq = MH * q;
      gm.append("line").attr("x1", -4).attr("x2", mo.W + 4).attr("y1", yq).attr("y2", yq)
        .attr("stroke", SC.a2).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 4");
      gm.append("text").attr("x", mo.W + 6).attr("y", yq + 4).attr("font-size", 10)
        .attr("fill", SC.a2).text(ST.pct(q, 1));
    }
    /* column labels with their totals underneath */
    let cx = 0;
    for (let j = 0; j < s.c; j++) {
      const w = eW.checked ? (MW - (s.c - 1) * 4) * s.C[j] / n : (MW - (s.c - 1) * 4) / s.c;
      gm.append("text").attr("x", cx + w / 2).attr("y", MH + 15).attr("text-anchor", "middle")
        .attr("font-size", w > 46 ? 10.5 : 8.5).attr("fill", SC.ink).text(T.cols[j]);
      gm.append("text").attr("x", cx + w / 2).attr("y", MH + 27).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", SC.muted).text(CT.grp(s.C[j]));
      cx += w + 4;
    }
    /* row labels down the left */
    T.rows.forEach((lb, i) => {
      const t0 = mo.tiles.find(t => t.i === i && t.j === 0);
      gm.append("text").attr("x", -8).attr("y", t0.y + t0.h / 2 + 4).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", pal[i % pal.length]).text(lb);
    });

    /* ── right: the same table as numbers under the chosen percentaging ─ */
    const gt = CT.sub(g, 500, 34, mode === "count" ? "raw counts"
      : mode === "col" ? "column % — within each column"
        : mode === "row" ? "row % — within each row" : "joint % — of the grand total");
    const cw = 52, ch = 30;
    const val = (i, j) => mode === "count" ? O[i][j]
      : mode === "col" ? O[i][j] / s.C[j]
        : mode === "row" ? O[i][j] / s.R[i] : O[i][j] / n;
    const ref = (i, j) => mode === "col" ? s.R[i] / n : mode === "row" ? s.C[j] / n : NaN;
    for (let j = 0; j < s.c; j++) gt.append("text").attr("x", j * cw + cw / 2).attr("y", 0)
      .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", SC.accent)
      .text(T.cols[j].slice(0, 6));
    for (let i = 0; i < s.r; i++) {
      gt.append("text").attr("x", -6).attr("y", 14 + i * ch + ch / 2 + 4).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", SC.accent).text(T.rows[i].slice(0, 8));
      for (let j = 0; j < s.c; j++) {
        const v = val(i, j), rf = ref(i, j);
        const dev = isFinite(rf) ? (v - rf) : 0;
        gt.append("rect").attr("x", j * cw).attr("y", 14 + i * ch).attr("width", cw - 3)
          .attr("height", ch - 3).attr("rx", 4).attr("stroke", SC.line)
          .attr("fill", isFinite(rf) ? CT.residColour(dev, 0.25) : SC.panel2)
          .attr("fill-opacity", isFinite(rf) ? CT.residOp(dev, 0.25) : 1);
        gt.append("text").attr("x", j * cw + (cw - 3) / 2).attr("y", 14 + i * ch + ch / 2 + 4)
          .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", SC.ink)
          .text(mode === "count" ? CT.grp(v) : ST.pct(v, 1));
      }
    }
    /* the marginal row/column, which is what the conditionals are compared against */
    const my = 14 + s.r * ch + 6;
    gt.append("text").attr("x", -6).attr("y", my + 13).attr("text-anchor", "end")
      .attr("font-size", 9).attr("fill", SC.a2).text("marginal");
    for (let j = 0; j < s.c; j++) gt.append("text").attr("x", j * cw + (cw - 3) / 2).attr("y", my + 13)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.a2)
      .text(mode === "col" ? "" : ST.pct(s.C[j] / n, 1));
    if (mode === "col") for (let i = 0; i < s.r; i++)
      gt.append("text").attr("x", s.c * cw + 8).attr("y", 14 + i * ch + ch / 2 + 4)
        .attr("font-size", 10).attr("fill", SC.a2).text(ST.pct(s.R[i] / n, 1));

    /* the sentence each percentaging licenses */
    const say = { col: "read DOWN a column: the rate within that group",
      row: "read ACROSS a row: the composition of that group",
      joint: "every cell as a share of the whole table — the eight sum to 100%",
      count: "the raw counts, from which all three percentagings are derived" };
    CT.sub(g, 500, my + 78, "").append("text").attr("x", -66).attr("y", 0)
      .attr("font-size", 10.5).attr("fill", SC.muted).text(say[mode]);

    const q0 = s.R[0] / n;
    const spread = s.C.map((_, j) => O[0][j] / s.C[j]);
    out.innerHTML =
      `<b>${T.name}</b> · n = ${CT.grp(n)}. Conditional distribution of <b>${T.rows[0]}</b> by column: ` +
      spread.map((v, j) => `${T.cols[j]} <b>${ST.pct(v, 2)}</b>`).join(" · ") +
      ` — against the overall marginal <b>${ST.pct(q0, 2)}</b>. ` +
      (s.chi2 < 1e-9
        ? `Every column sits at exactly the marginal, so the mosaic's dividing line is flat and <code>χ² = 0</code>: this is what independence looks like.`
        : `The spread of those percentages is what <code>χ² = ${ST.fmt(s.chi2, 3)}</code> on ${s.df} df measures (p = ${CT.pstr(s.p)}, φ = ${ST.fmt(s.phi, 4)}). ` +
          `Independence would draw the dividing line flat across every column.`);
  }

  [eD, eM, eMg, eW].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ─────────────── 2 · the chi-square density, its tail, its approximations ─────────────── */
(function () {
  const svg = d3.select("#chid-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 420;
  const eDf = document.getElementById("cd-df"), eDfv = document.getElementById("cd-dfv");
  const eX = document.getElementById("cd-x"), eXv = document.getElementById("cd-xv");
  const eC = document.getElementById("cd-crit"), eN = document.getElementById("cd-norm");
  const eF = document.getElementById("cd-fam");
  const out = document.getElementById("chid-readout");

  function draw() {
    const k = +eDf.value, x0 = +eX.value;
    eDfv.textContent = k; eXv.textContent = ST.fmt(x0, 3);

    const hi = Math.max(x0 * 1.25, k + 5 * Math.sqrt(2 * k), 14);
    const f = ST.frame(svg, W, H, { l: 58, r: 18, t: 26, b: 46 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const xs = ST.linspace(1e-4, hi, 700);
    const dens = xs.map(x => ST.chi2Pdf(x, k));
    /* cap the y axis: at df 1 and 2 the density is unbounded at the origin */
    const yMax = Math.max(0.02, Math.min(0.5, d3.max(dens.filter((_, i) => xs[i] > hi * 0.012)) * 1.35));
    const x = d3.scaleLinear().domain([0, hi]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, yMax]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);
    ST.axisB(g, x, ih, 8, "the statistic");
    ST.axisL(g, y, 5, "density");

    if (eF.checked) {                       // the whole family, faintly
      [1, 2, 3, 5, 8, 12, 20, 30].forEach((kk, i) => {
        if (kk === k) return;
        const pts = xs.map(v => ({ x: v, y: ST.chi2Pdf(v, kk) }));
        g.append("path").attr("fill", "none").attr("stroke", SC.muted).attr("stroke-opacity", 0.35)
          .attr("stroke-width", 1)
          .attr("d", d3.line().x(d => x(d.x)).y(d => y(Math.min(d.y, yMax)))(pts));
      });
    }

    /* the right tail — the p-value, drawn as area */
    const tail = xs.filter(v => v >= x0).map(v => ({ x: v, y: Math.min(ST.chi2Pdf(v, k), yMax) }));
    if (tail.length > 1) g.append("path").attr("fill", SC.bad).attr("fill-opacity", 0.34)
      .attr("d", d3.area().x(d => x(d.x)).y0(ih).y1(d => y(d.y))(tail));

    const pts = xs.map(v => ({ x: v, y: ST.chi2Pdf(v, k) }));
    g.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(d => x(d.x)).y(d => y(Math.min(d.y, yMax)))(pts));

    if (eN.checked) {                       // the two classical tail approximations, as curves
      const wh = xs.map(v => {              // Wilson–Hilferty, differentiated numerically
        const h = Math.max(1e-4, v * 1e-4);
        const F = u => ST.normCdf((Math.pow(u / k, 1 / 3) - (1 - 2 / (9 * k))) / Math.sqrt(2 / (9 * k)));
        return { x: v, y: Math.max(0, (F(v + h) - F(v - h)) / (2 * h)) };
      });
      g.append("path").attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "5 3")
        .attr("d", d3.line().x(d => x(d.x)).y(d => y(Math.min(d.y, yMax)))(wh));
      ST.legend(g, [{ label: "χ² density", color: SC.accent },
        { label: "cube-root normal approximation", color: SC.good, dash: "5 3" }], iw - 210, 8);
    }

    const mean = k, sd = Math.sqrt(2 * k);
    CT.rule(g, x(mean), ih, SC.muted, "mean = df", "3 3");
    if (eC.checked) {
      const cv = ST.chi2Quant(0.95, k);
      if (cv <= hi) CT.rule(g, x(cv), ih, SC.a2, "5% critical " + ST.fmt(cv, 3), "4 3", 22);
    }
    if (x0 <= hi) {
      g.append("line").attr("x1", x(x0)).attr("x2", x(x0)).attr("y1", 0).attr("y2", ih)
        .attr("stroke", SC.bad).attr("stroke-width", 2.4);
      g.append("text").attr("x", x(x0)).attr("y", -8).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("font-weight", 600).attr("fill", SC.bad)
        .text("observed " + ST.fmt(x0, 2));
    } else {
      g.append("text").attr("x", iw - 4).attr("y", 14).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", SC.bad)
        .text("observed " + ST.fmt(x0, 2) + " — off the right of this axis");
    }

    const p = CT.chi2Sf(x0, k);
    const z1 = Math.sqrt(2 * x0) - Math.sqrt(2 * k - 1);
    const z2 = (Math.pow(x0 / k, 1 / 3) - (1 - 2 / (9 * k))) / Math.sqrt(2 / (9 * k));
    out.innerHTML =
      `<code>χ²</code> on <b>${k}</b> df — mean <b>${mean}</b>, SD <b>${ST.fmt(sd, 3)}</b>, mode <b>${Math.max(0, k - 2)}</b>. ` +
      `Observed <b>${ST.fmt(x0, 3)}</b> gives <code>p = ${CT.pstr(p)}</code>; the 5% critical value is <b>${ST.fmt(ST.chi2Quant(0.95, k), 4)}</b>, ` +
      `so this ${x0 > ST.chi2Quant(0.95, k) ? "<b>rejects</b>" : "does <b>not</b> reject"} at 5%. ` +
      `The statistic is ${ST.fmt(x0 / k, 2)}× its mean. ` +
      `Normal approximations: square-root form <code>z = ${ST.fmt(z1, 4)}</code> → <code>p = ${CT.pstr(1 - ST.normCdf(z1))}</code>; ` +
      `cube-root form <code>z = ${ST.fmt(z2, 4)}</code> → <code>p = ${CT.pstr(1 - ST.normCdf(z2))}</code>.`;
  }

  [eDf, eX, eC, eN, eF].forEach(el => el.addEventListener("input", draw));
  document.getElementById("cd-mm").addEventListener("click", () => {
    eDf.value = 5; eX.value = 8.567; draw();
  });
  document.getElementById("cd-tit").addEventListener("click", () => {
    eDf.value = 3; eX.value = 192.34; draw();
  });
  draw();
})();

/* ─────────────── 3 · the ship's mosaic with the null ghosted behind it ─────────────── */
(function () {
  const svg = d3.select("#mos-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const eV = document.getElementById("mo-view"), eMix = document.getElementById("mo-mix");
  const eMixv = document.getElementById("mo-mixv"), eL = document.getElementById("mo-lbl");
  const out = document.getElementById("mos-readout");
  const T = CT.TITANIC;

  /* interpolate the observed table toward its own expected table. The margins are
     preserved exactly at every t, because E has the same margins as O (§03), so the
     mosaic's column widths never move and only the splits do.                     */
  function blend(t) {
    const E = CT.expected(T.O);
    return T.O.map((r, i) => r.map((o, j) => o + t * (E[i][j] - o)));
  }

  function draw() {
    const t = +eMix.value / 100;
    eMixv.textContent = Math.round(t * 100) + "%";
    const O = blend(t), s = CT.stats(O), n = s.n;
    const view = eV.value;

    const f = ST.frame(svg, W, H, { l: 12, r: 12, t: 14, b: 12 });
    const g = f.g;
    const MW = 560, MH = 300;
    const gm = CT.sub(g, 96, 40, "columns are ticket classes, widths ∝ how many people were in them");
    const mo = CT.mosaic(O, MW, MH, { widths: true, gapX: 5 });
    const q = s.R[0] / n;

    mo.tiles.forEach(tl => {
      let fill, op = 0.6, lab = "";
      if (view === "ghost") { fill = tl.i === 0 ? SC.good : SC.bad; op = 0.5; lab = ST.pct(tl.frac, 1); }
      else {
        const v = view === "contrib" ? s.contrib[tl.i][tl.j]
          : view === "resid" ? s.std[tl.i][tl.j] : s.adj[tl.i][tl.j];
        const cap = view === "contrib" ? 95 : view === "resid" ? 10 : 13;
        fill = view === "contrib" ? SC.a2 : CT.residColour(v, cap);
        op = view === "contrib" ? 0.20 + 0.7 * ST.clamp(v / cap, 0, 1) : CT.residOp(v, cap);
        lab = view === "contrib" ? ST.fmt(v, 1) : ST.fmt(v, 2);
      }
      gm.append("rect").attr("x", tl.x).attr("y", tl.y).attr("width", Math.max(0, tl.w))
        .attr("height", Math.max(0, tl.h)).attr("rx", 3).attr("fill", fill)
        .attr("fill-opacity", op).attr("stroke", SC.bg).attr("stroke-width", 0.8);
      if (eL.checked && tl.w > 40 && tl.h > 18) {
        gm.append("text").attr("x", tl.x + tl.w / 2).attr("y", tl.y + tl.h / 2 - 2)
          .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
          .attr("fill", SC.ink).text(lab);
        gm.append("text").attr("x", tl.x + tl.w / 2).attr("y", tl.y + tl.h / 2 + 11)
          .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", SC.muted)
          .text("O " + Math.round(tl.o) + " · E " + ST.fmt(s.E[tl.i][tl.j], 0));
      }
    });

    /* the null's split height in every column: the pooled rate — a straight line */
    const yq = MH * q;
    gm.append("line").attr("x1", -6).attr("x2", mo.W + 6).attr("y1", yq).attr("y2", yq)
      .attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
    gm.append("text").attr("x", mo.W + 10).attr("y", yq + 4).attr("font-size", 10)
      .attr("fill", SC.a2).text("null: " + ST.pct(q, 1));
    /* and each column's own expected split as a short ghost bar, so the gap is visible */
    let cx = 0;
    for (let j = 0; j < s.c; j++) {
      const w = (MW - (s.c - 1) * 5) * s.C[j] / n;
      gm.append("rect").attr("x", cx).attr("y", 0).attr("width", w).attr("height", yq)
        .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.4)
        .attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.85);
      gm.append("text").attr("x", cx + w / 2).attr("y", MH + 16).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", SC.ink).text(T.cols[j]);
      gm.append("text").attr("x", cx + w / 2).attr("y", MH + 28).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", SC.muted)
        .text(CT.grp(s.C[j]) + " · " + ST.pct(O[0][j] / s.C[j], 1));
      cx += w + 5;
    }
    T.rows.forEach((lb, i) => {
      const t0 = mo.tiles.find(u => u.i === i && u.j === 0);
      gm.append("text").attr("x", -10).attr("y", t0.y + t0.h / 2 + 4).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", i === 0 ? SC.good : SC.bad).text(lb);
    });

    /* the readouts */
    const gy = 40 + MH + 52;
    CT.chip(g, 96, gy, "CHI-SQUARE", ST.fmt(s.chi2, 3), s.chi2 > s.crit ? SC.bad : SC.good, 122);
    CT.chip(g, 226, gy, "DF", String(s.df), SC.ink, 60);
    CT.chip(g, 294, gy, "P-VALUE", CT.pstr(s.p), s.p < 0.05 ? SC.bad : SC.good, 126);
    CT.chip(g, 428, gy, "SMALLEST E", ST.fmt(s.minE, 1), SC.ink, 118);
    CT.chip(g, 554, gy, "CRAMÉR V", ST.fmt(s.V, 4), SC.a2, 102);

    const worst = [];
    for (let i = 0; i < s.r; i++) for (let j = 0; j < s.c; j++)
      worst.push({ i: i, j: j, a: s.adj[i][j], c: s.contrib[i][j] });
    worst.sort((p, q2) => q2.c - p.c);
    const top = worst[0];
    out.innerHTML =
      (t === 0
        ? `The real table. `
        : `The table has been pulled <b>${Math.round(t * 100)}%</b> of the way toward its own expected counts — the margins are untouched, only the splits move. `) +
      `<code>χ² = ${ST.fmt(s.chi2, 4)}</code> on ${s.df} df, <code>p = ${CT.pstr(s.p)}</code>, ` +
      `Cramér <code>V = ${ST.fmt(s.V, 4)}</code>. The largest single contribution is ` +
      `<b>${T.rows[top.i]} / ${T.cols[top.j]}</b> at ${ST.fmt(top.c, 2)} — ` +
      `<b>${ST.pct(top.c / s.chi2, 1)}</b> of the total, with an adjusted residual of ${ST.fmt(top.a, 2)}. ` +
      (t === 0 ? `Drag the slider and watch the statistic fall to zero as the dividing line flattens onto the dashed null line.`
        : t >= 0.999 ? `At 100% the table <i>is</i> its expected table: every split sits on the dashed line and <code>χ²</code> is zero.`
          : `Note that <code>χ²</code> falls like <b>(1 − t)²</b>, not linearly — halving every deviation quarters the statistic.`);
  }

  [eV, eMix, eL].forEach(el => el.addEventListener("input", draw));
  document.getElementById("mo-reset").addEventListener("click", () => { eMix.value = 0; draw(); });
  draw();
})();

/* ─────────────── 4 · the laboratory: an editable table ─────────────── */
(function () {
  const svg = d3.select("#lab-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 480;
  const eP = document.getElementById("lb-preset"), eV = document.getElementById("lb-view");
  const eS = document.getElementById("lb-scale"), eSv = document.getElementById("lb-scalev");
  const out = document.getElementById("lab-readout");

  const PRESETS = {
    ind: CT.IND, titanic: CT.TITANIC, fisher: CT.SMALL, sparse: CT.SPARSE,
    null: CT.NOASSOC,
    big: { name: "a huge table with a trivial association", rows: ["Yes", "No"],
           cols: ["A", "B"], O: [[10200, 9800], [9800, 10200]] }
  };
  let cur = null, base = null, sel = null;

  function load(key) {
    const T = PRESETS[key];
    base = T.O.map(r => r.slice());
    cur = { rows: T.rows.slice(), cols: T.cols.slice(), name: T.name,
            O: base.map(r => r.slice()) };
    eS.value = 10; eSv.textContent = "1.0"; sel = null;
  }

  function applyScale() {
    const k = +eS.value / 10;
    eSv.textContent = ST.fmt(k, 1);
    cur.O = base.map(r => r.map(v => Math.max(0, Math.round(v * k))));
  }

  function draw() {
    const O = cur.O, s = CT.stats(O), view = eV.value;
    const is2 = (s.r === 2 && s.c === 2);
    const fi = is2 ? CT.fisher(O[0][0], O[0][1], O[1][0], O[1][1], "two") : null;
    const me = is2 ? CT.measures(O[0][0], O[0][1], O[1][0], O[1][1], false) : null;

    const f = ST.frame(svg, W, H, { l: 12, r: 12, t: 14, b: 12 });
    const g = f.g;

    /* ── the editable grid ─────────────────────────────────────────── */
    const cw = Math.min(84, 340 / (s.c + 1)), ch = Math.min(58, 250 / (s.r + 1));
    const cap = view === "contrib" ? Math.max(1, d3.max(s.contrib.flat())) : 4;
    const tbl = CT.gridTable(g, O, 30, 34, cw, ch, {
      rows: cur.rows, cols: cur.cols,
      fill: (i, j) => {
        if (view === "contrib") return SC.a2;
        if (view === "std") return CT.residColour(s.std[i][j], 4);
        if (view === "adj") return CT.residColour(s.adj[i][j], 4);
        if (view === "colpct") return CT.residColour(O[i][j] / s.C[j] - s.R[i] / s.n, 0.25);
        return SC.panel2;
      },
      text: (i, j) => view === "contrib" ? ST.fmt(s.contrib[i][j], 2)
        : view === "std" ? ST.fmt(s.std[i][j], 2)
          : view === "adj" ? ST.fmt(s.adj[i][j], 2)
            : view === "colpct" ? ST.pct(O[i][j] / s.C[j], 1) : CT.grp(O[i][j]),
      sub: (i, j) => view === "oe" ? "E " + ST.fmt(s.E[i][j], 1) : "O " + CT.grp(O[i][j])
    });
    tbl.cell.selectAll("rect").attr("fill-opacity", function (d) {
      if (view === "contrib") return 0.18 + 0.7 * ST.clamp(s.contrib[d.i][d.j] / cap, 0, 1);
      if (view === "std") return CT.residOp(s.std[d.i][d.j], 4);
      if (view === "adj") return CT.residOp(s.adj[d.i][d.j], 4);
      if (view === "colpct") return CT.residOp(O[d.i][d.j] / s.C[d.j] - s.R[d.i] / s.n, 0.25);
      return 1;
    });
    tbl.cell.style("cursor", "ns-resize")
      .call(d3.drag()
        .on("start", function (ev, d) { sel = d; d.start = cur.O[d.i][d.j]; })
        .on("drag", function (ev, d) {
          const step = ev.sourceEvent && ev.sourceEvent.shiftKey ? 5 : 1;
          const unit = Math.max(1, Math.round(Math.max(1, d.start) / 40));
          const nv = Math.max(0, Math.round(d.start - ev.y * step * unit / 6));
          if (nv !== cur.O[d.i][d.j]) {
            cur.O[d.i][d.j] = nv;
            base[d.i][d.j] = nv / Math.max(0.1, +eS.value / 10);
            draw();
          }
        }));
    if (sel) tbl.cell.filter(d => d.i === sel.i && d.j === sel.j).selectAll("rect")
      .attr("stroke", SC.accent).attr("stroke-width", 2);
    /* the margins, printed outside the grid */
    for (let i = 0; i < s.r; i++) g.append("text")
      .attr("x", 30 + (s.c + 1) * cw + 6).attr("y", 34 + (i + 1) * ch + ch / 2 + 4)
      .attr("font-size", 10).attr("fill", SC.muted).text(CT.grp(s.R[i]));
    for (let j = 0; j < s.c; j++) g.append("text")
      .attr("x", 30 + (j + 1) * cw + (cw - 3) / 2).attr("y", 34 + (s.r + 1) * ch + 12)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.muted).text(CT.grp(s.C[j]));
    g.append("text").attr("x", 30).attr("y", 34 + (s.r + 1) * ch + 12)
      .attr("font-size", 10).attr("fill", SC.accent).text("n = " + CT.grp(s.n));
    g.append("text").attr("x", 30).attr("y", 22).attr("font-size", 11).attr("fill", SC.muted)
      .text("drag a cell up or down to change its count");

    /* ── the panel of results ─────────────────────────────────────── */
    const px = 400;
    const rows = [
      ["Pearson χ²", ST.fmt(s.chi2, 4), CT.pstr(s.p), s.p < 0.05],
      ["likelihood ratio G²", ST.fmt(s.g2, 4), CT.pstr(s.gp), s.gp < 0.05]
    ];
    if (is2) {
      rows.push(["Yates corrected", ST.fmt(s.yates, 4), CT.pstr(s.yp), s.yp < 0.05]);
      rows.push(["Fisher exact (two-sided)", "—", CT.pstr(fi.p), fi.p < 0.05]);
    }
    g.append("text").attr("x", px).attr("y", 22).attr("font-size", 11.5)
      .attr("font-weight", 600).attr("fill", SC.ink).text("four tests on the same table");
    g.append("text").attr("x", px).attr("y", 40).attr("font-size", 9.5).attr("fill", SC.muted)
      .text("test");
    g.append("text").attr("x", px + 176).attr("y", 40).attr("font-size", 9.5).attr("fill", SC.muted)
      .text("statistic");
    g.append("text").attr("x", px + 250).attr("y", 40).attr("font-size", 9.5).attr("fill", SC.muted)
      .text("p");
    rows.forEach((r, i) => {
      const yy = 58 + i * 24;
      g.append("rect").attr("x", px - 6).attr("y", yy - 13).attr("width", 322).attr("height", 21)
        .attr("rx", 4).attr("fill", i % 2 ? SC.panel2 : "none");
      g.append("text").attr("x", px).attr("y", yy + 2).attr("font-size", 11).attr("fill", SC.ink).text(r[0]);
      g.append("text").attr("x", px + 176).attr("y", yy + 2).attr("font-size", 11)
        .attr("fill", SC.muted).text(r[1]);
      g.append("text").attr("x", px + 250).attr("y", yy + 2).attr("font-size", 11)
        .attr("font-weight", 600).attr("fill", r[3] ? SC.bad : SC.good).text(r[2]);
    });

    const by = 58 + rows.length * 24 + 14;
    CT.chip(g, px - 6, by, "DEGREES OF FREEDOM", String(s.df), SC.ink, 150);
    CT.chip(g, px + 152, by, "SMALLEST EXPECTED", ST.fmt(s.minE, 2),
      s.minE < 5 ? SC.a2 : SC.good, 164);
    CT.chip(g, px - 6, by + 42, "PHI  √(χ²/n)", ST.fmt(s.phi, 4), SC.a2, 150);
    CT.chip(g, px + 152, by + 42, "CRAMÉR V", ST.fmt(s.V, 4), SC.a2, 164);
    if (is2) {
      CT.chip(g, px - 6, by + 84, "ODDS RATIO", isFinite(me.or) ? ST.fmt(me.or, 4) : "undefined",
        SC.violet, 150);
      CT.chip(g, px + 152, by + 84, "95% CI FOR OR",
        isFinite(me.lo) ? `${ST.fmt(me.lo, 3)} – ${ST.fmt(me.hi, 3)}` : "—", SC.violet, 164);
    }

    /* ── the reference density with the statistic on it ─────────────── */
    const dy = 330, dh = 108, dw = 320;
    const gd = CT.sub(g, 30, dy, "the χ² reference density on " + s.df + " df, with this table's statistic");
    const hiX = Math.max(s.chi2 * 1.25, s.df + 4.5 * Math.sqrt(2 * s.df), 12);
    const xs = ST.linspace(1e-4, hiX, 320);
    const yM = Math.max(1e-3, d3.max(xs.filter(v => v > hiX * 0.02).map(v => ST.chi2Pdf(v, s.df))) * 1.25);
    const xx = d3.scaleLinear().domain([0, hiX]).range([0, dw]);
    const yy2 = d3.scaleLinear().domain([0, yM]).range([dh, 0]);
    gd.append("g").attr("class", "axis").attr("transform", `translate(0,${dh})`)
      .call(d3.axisBottom(xx).ticks(6));
    const tail = xs.filter(v => v >= s.chi2).map(v => ({ x: v, y: Math.min(ST.chi2Pdf(v, s.df), yM) }));
    if (tail.length > 1) gd.append("path").attr("fill", SC.bad).attr("fill-opacity", 0.32)
      .attr("d", d3.area().x(d => xx(d.x)).y0(dh).y1(d => yy2(d.y))(tail));
    gd.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 1.8)
      .attr("d", d3.line().x(d => xx(d.x)).y(d => yy2(Math.min(ST.chi2Pdf(d.x, s.df), yM)))
        (xs.map(v => ({ x: v }))));
    if (s.chi2 <= hiX) {
      gd.append("line").attr("x1", xx(s.chi2)).attr("x2", xx(s.chi2)).attr("y1", 0).attr("y2", dh)
        .attr("stroke", SC.bad).attr("stroke-width", 2);
      gd.append("text").attr("x", xx(s.chi2)).attr("y", -2).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.bad).text(ST.fmt(s.chi2, 2));
    }
    const cv = s.crit;
    if (cv <= hiX) gd.append("line").attr("x1", xx(cv)).attr("x2", xx(cv)).attr("y1", 0).attr("y2", dh)
      .attr("stroke", SC.a2).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");

    /* ── the readout ───────────────────────────────────────────────── */
    let msg = `<b>${cur.name}</b>, ${s.r} × ${s.c}, n = ${CT.grp(s.n)}. ` +
      `<code>χ²(${s.df}) = ${ST.fmt(s.chi2, 4)}</code>, p = ${CT.pstr(s.p)}, φ = ${ST.fmt(s.phi, 4)}. `;
    if (s.minE < 5) msg += `<b>Smallest expected count is ${ST.fmt(s.minE, 2)}</b> — below the conventional guideline, so ` +
      (is2 ? `compare against Fisher's exact p of <b>${CT.pstr(fi.p)}</b>. ` : `treat the p-value with caution and consider a permutation test (§10). `);
    if (is2) {
      const disagree = (s.p < 0.05) !== (fi.p < 0.05);
      msg += disagree
        ? `<b>The approximation and the exact test disagree at 5%</b>: <code>χ²</code> gives ${CT.pstr(s.p)} and Fisher gives ${CT.pstr(fi.p)}. Report the exact one. `
        : `<code>χ²</code> and Fisher agree (${CT.pstr(s.p)} against ${CT.pstr(fi.p)}). `;
      msg += `Odds ratio ${isFinite(me.or) ? ST.fmt(me.or, 3) : "undefined — a cell is empty"}` +
        (isFinite(me.lo) ? `, 95% CI (${ST.fmt(me.lo, 3)}, ${ST.fmt(me.hi, 3)}). ` : ". ");
      msg += `Two-proportion <code>z = ${ST.fmt(CT.twoPropZ(O[0][0], O[0][1], O[1][0], O[1][1]), 5)}</code>, ` +
        `and <code>z² = ${ST.fmt(Math.pow(CT.twoPropZ(O[0][0], O[0][1], O[1][0], O[1][1]), 2), 5)}</code> — the same number as <code>χ²</code>.`;
    } else {
      msg += `<code>G² = ${ST.fmt(s.g2, 4)}</code> (p = ${CT.pstr(s.gp)}). Largest |adjusted residual| = ` +
        ST.fmt(Math.max.apply(null, s.adj.flat().map(Math.abs)), 2) + `, against a Bonferroni threshold of ` +
        ST.fmt(ST.normQuant(1 - 0.025 / (s.r * s.c)), 3) + ` for ${s.r * s.c} cells.`;
    }
    out.innerHTML = msg;
  }

  eP.addEventListener("change", () => { load(eP.value); draw(); });
  eV.addEventListener("change", draw);
  eS.addEventListener("input", () => { applyScale(); draw(); });
  document.getElementById("lb-null").addEventListener("click", () => {
    const E = CT.expected(cur.O);
    cur.O = E.map(r => r.map(v => Math.round(v)));
    base = cur.O.map(r => r.map(v => v / Math.max(0.1, +eS.value / 10)));
    draw();
  });
  document.getElementById("lb-reset").addEventListener("click", () => { load(eP.value); draw(); });
  load(eP.value); draw();
})();

/* ─────────────── 5 · residuals, their reference, and the multiplicity ─────────────── */
(function () {
  const svg = d3.select("#res-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 440;
  const eD = document.getElementById("rs-data"), eK = document.getElementById("rs-kind");
  const eT = document.getElementById("rs-thr");
  const out = document.getElementById("res-readout");
  let seed = 20260908;

  /* the measured curve on the right: P(some |adjusted residual| > 1.96) under
     independence, against the number of cells. Simulated once, here, rather than
     quoted — 4000 tables per shape is enough for two significant figures.       */
  /* The measured curve on the right: P(some |adjusted residual| > 1.96) under
     independence, against the number of cells — simulated here rather than quoted.
     Two engineering notes. It is written flat and allocation-free because the
     general CT.stats call evaluates an incomplete gamma per table, which dominates
     everything else; and it is computed LAZILY, one frame after the first paint,
     so the page is never blocked waiting for it. */
  const SHAPES = [[2, 2], [2, 3], [2, 4], [3, 3], [3, 4], [4, 4], [4, 5], [5, 5], [6, 6]];
  let MULT = null, partial = [], simRng = null;

  /* one shape's worth of work — about a quarter of a second — so the loop can be
     spread across animation ticks instead of freezing the page for two seconds. */
  function simShape(R, C) {
    const N = 700, B = 2000;
    if (!simRng) simRng = ST.rng(4242);
    const r = simRng;
    const crit = ST.chi2Quant(0.95, (R - 1) * (C - 1));
    const O = new Float64Array(R * C), rs = new Float64Array(R), cs = new Float64Array(C);
    let any = 0, rej = 0;
    for (let b = 0; b < B; b++) {
      O.fill(0); rs.fill(0); cs.fill(0);
      for (let t = 0; t < N; t++) {
        const i = (r() * R) | 0, j = (r() * C) | 0;
        O[i * C + j]++; rs[i]++; cs[j]++;
      }
      let chi2 = 0, mx = 0;
      for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
        const e = rs[i] * cs[j] / N;
        if (e <= 0) continue;
        const d = O[i * C + j] - e;
        chi2 += d * d / e;
        const v = e * (1 - rs[i] / N) * (1 - cs[j] / N);
        if (v > 1e-12) { const a2 = Math.abs(d) / Math.sqrt(v); if (a2 > mx) mx = a2; }
      }
      if (chi2 > crit) rej++;
      if (mx > 1.959964) any++;
    }
    return { cells: R * C, R: R, C: C, any: any / B, rej: rej / B, B: B };
  }
  let simQueued = false;
  function stepSim() {
    simQueued = false;
    if (partial.length >= SHAPES.length) { MULT = partial; draw(); return; }
    partial.push(simShape(SHAPES[partial.length][0], SHAPES[partial.length][1]));
    if (partial.length >= SHAPES.length) MULT = partial;
    draw();
  }
  function queueSim() {
    if (simQueued || MULT) return;
    simQueued = true;
    setTimeout(stepSim, 30);
  }

  function tableFor(kind) {
    if (kind === "titanic") return { kind: "tab", T: CT.TITANIC };
    if (kind === "ind") return { kind: "tab", T: CT.IND };
    if (kind === "mm") return { kind: "gof" };
    /* a random 4 × 4 generated under exact independence — no association at all */
    const r = ST.rng(seed), O = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    for (let t = 0; t < 1200; t++) O[Math.floor(r() * 4)][Math.floor(r() * 4)]++;
    return { kind: "tab", T: { name: "a random null table", rows: ["w", "x", "y", "z"],
      cols: ["1", "2", "3", "4"], O: O } };
  }

  function draw() {
    const spec = tableFor(eD.value), useAdj = eK.value === "adj";
    let vals = [], labels = [], nCells, s, title;
    if (spec.kind === "gof") {
      s = CT.gof(CT.MM.O, CT.MM.p);
      vals = (useAdj ? s.adj : s.std).map(v => v);
      labels = CT.MM.cats.slice();
      nCells = vals.length; title = "six sweet colours, goodness-of-fit";
    } else {
      s = CT.stats(spec.T.O);
      for (let i = 0; i < s.r; i++) for (let j = 0; j < s.c; j++) {
        vals.push(useAdj ? s.adj[i][j] : s.std[i][j]);
        labels.push(spec.T.rows[i].slice(0, 8) + " / " + spec.T.cols[j].slice(0, 8));
      }
      nCells = s.r * s.c; title = spec.T.name;
    }
    const thr = eT.value === "196" ? 1.959964
      : eT.value === "bonf" ? ST.normQuant(1 - 0.025 / nCells) : Infinity;

    const f = ST.frame(svg, W, H, { l: 12, r: 12, t: 14, b: 12 });
    const g = f.g;

    /* ── left: the cells as a residual map ─────────────────────────── */
    const gl = CT.sub(g, 26, 36, title + " — " + (useAdj ? "adjusted" : "standardised") + " residuals");
    const cols = spec.kind === "gof" ? 3 : (s.c || 4);
    const rowsN = Math.ceil(vals.length / cols);
    const cw = Math.min(96, 300 / cols), ch = 52;
    vals.forEach((v, idx) => {
      const i = Math.floor(idx / cols), j = idx % cols;
      const x = j * cw, y = i * ch;
      const mag = ST.clamp(Math.abs(v) / 6, 0.10, 1);
      gl.append("rect").attr("x", x + (cw - 3) * (1 - mag) / 2).attr("y", y + (ch - 6) * (1 - mag) / 2)
        .attr("width", (cw - 3) * mag).attr("height", (ch - 6) * mag).attr("rx", 3)
        .attr("fill", CT.residColour(v, 6)).attr("fill-opacity", 0.8)
        .attr("stroke", Math.abs(v) > thr ? SC.ink : SC.line)
        .attr("stroke-width", Math.abs(v) > thr ? 2 : 0.7);
      gl.append("text").attr("x", x + (cw - 3) / 2).attr("y", y + ch / 2 + 2)
        .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
        .attr("fill", SC.ink).text(ST.fmt(v, 2));
      gl.append("text").attr("x", x + (cw - 3) / 2).attr("y", y + ch / 2 + 14)
        .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", SC.muted)
        .text(labels[idx].length > 16 ? labels[idx].slice(0, 15) + "…" : labels[idx]);
    });

    /* the residuals as ticks on a standard normal, directly underneath */
    const ny = 36 + rowsN * ch + 34, nw = cols * cw - 6, nh = 76;
    const gn = CT.sub(g, 26, ny, "the same residuals against N(0, 1)");
    const xn = d3.scaleLinear().domain([-Math.max(4.5, d3.max(vals.map(Math.abs)) * 1.15),
      Math.max(4.5, d3.max(vals.map(Math.abs)) * 1.15)]).range([0, nw]);
    const zs = ST.linspace(xn.domain()[0], xn.domain()[1], 240);
    const yn = d3.scaleLinear().domain([0, 0.42]).range([nh, 0]);
    gn.append("path").attr("fill", SC.accent).attr("fill-opacity", 0.16)
      .attr("d", d3.area().x(d => xn(d)).y0(nh).y1(d => yn(ST.normPdf(d)))(zs));
    gn.append("g").attr("class", "axis").attr("transform", `translate(0,${nh})`)
      .call(d3.axisBottom(xn).ticks(7));
    [-thr, thr].forEach(t => { if (isFinite(t) && t < xn.domain()[1])
      gn.append("line").attr("x1", xn(t)).attr("x2", xn(t)).attr("y1", 0).attr("y2", nh)
        .attr("stroke", SC.a2).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3"); });
    vals.forEach(v => gn.append("line")
      .attr("x1", xn(ST.clamp(v, xn.domain()[0], xn.domain()[1])))
      .attr("x2", xn(ST.clamp(v, xn.domain()[0], xn.domain()[1])))
      .attr("y1", nh).attr("y2", nh - 26).attr("stroke", CT.residColour(v, 6))
      .attr("stroke-width", 2.2));

    /* ── right: the measured multiplicity curve ────────────────────── */
    const rw = 288, rh = 250;
    const gr = CT.sub(g, 424, 36, "P(some cell past ±1.96) under independence");
    const near = MULT
      ? MULT.reduce((b2, d) => Math.abs(d.cells - nCells) < Math.abs(b2.cells - nCells) ? d : b2, MULT[0])
      : null;
    if (!MULT) {
      gr.append("text").attr("x", rw / 2).attr("y", rh / 2).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", SC.muted)
        .text("simulating… " + partial.length + " of " + SHAPES.length + " table shapes");
      queueSim();
    } else {
    const xm = d3.scaleLinear().domain([0, 38]).range([0, rw]);
    const ym = d3.scaleLinear().domain([0, 1]).range([rh, 0]);
    ST.gridY(gr, ym, rw, 5);
    gr.append("g").attr("class", "axis").attr("transform", `translate(0,${rh})`)
      .call(d3.axisBottom(xm).ticks(6));
    gr.append("g").attr("class", "axis").call(d3.axisLeft(ym).ticks(5).tickFormat(d3.format(".0%")));
    gr.append("text").attr("x", rw).attr("y", rh + 31).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("number of cells");
    gr.append("path").attr("fill", "none").attr("stroke", SC.bad).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(d => xm(d.cells)).y(d => ym(d.any))(MULT));
    gr.selectAll("circle.a").data(MULT).join("circle").attr("class", "a")
      .attr("cx", d => xm(d.cells)).attr("cy", d => ym(d.any)).attr("r", 3).attr("fill", SC.bad);
    gr.append("path").attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2)
      .attr("d", d3.line().x(d => xm(d.cells)).y(d => ym(d.rej))(MULT));
    gr.append("line").attr("x1", 0).attr("x2", rw).attr("y1", ym(0.05)).attr("y2", ym(0.05))
      .attr("stroke", SC.a2).attr("stroke-dasharray", "4 3");
    ST.legend(gr, [{ label: "some cell exceeds 1.96", color: SC.bad },
      { label: "the omnibus χ² rejects", color: SC.good },
      { label: "nominal 5%", color: SC.a2, dash: "4 3" }], 12, 16);
    /* where this table sits on that curve */
    gr.append("circle").attr("cx", xm(ST.clamp(nCells, 0, 38))).attr("cy", ym(near.any))
      .attr("r", 6).attr("fill", "none").attr("stroke", SC.ink).attr("stroke-width", 2);
    }

    const big = vals.map((v, i) => ({ v: v, l: labels[i] })).filter(d => Math.abs(d.v) > thr);
    const mx = vals.reduce((b, v) => Math.abs(v) > Math.abs(b) ? v : b, 0);
    out.innerHTML =
      `<b>${title}</b> · ${nCells} cells · ${useAdj ? "adjusted" : "standardised (Pearson)"} residuals. ` +
      (useAdj
        ? `These are approximately standard normal, so ±1.96 is the uncorrected 5% cut. `
        : `These sum in square to <code>χ² = ${ST.fmt(spec.kind === "gof" ? s.chi2 : s.chi2, 3)}</code> — check: Σr² = ` +
          ST.fmt(vals.reduce((t, v) => t + v * v, 0), 3) + `. They are systematically <b>too small</b> to read against ±1.96. `) +
      `Threshold in use: <b>${isFinite(thr) ? "±" + ST.fmt(thr, 3) : "none"}</b>` +
      (eT.value === "bonf" ? ` (Bonferroni over ${nCells} cells)` : "") + `. ` +
      (big.length
        ? `Past it: ${big.map(d => `<b>${d.l}</b> ${ST.fmt(d.v, 2)}`).join(", ")}. `
        : `No cell passes it; the largest is ${ST.fmt(mx, 2)}. `) +
      (near
        ? `Under complete independence, a table of about ${near.cells} cells has a chance of <b>${ST.pct(near.any, 1)}</b> ` +
          `that some cell exceeds 1.96, while the omnibus test holds at ${ST.pct(near.rej, 1)} — measured here from ` +
          `${CT.grp(near.B)} simulated null tables per shape, not quoted.`
        : `The multiplicity curve on the right is still simulating.`);
  }

  [eD, eK, eT].forEach(el => el.addEventListener("change", draw));
  document.getElementById("rs-new").addEventListener("click", () => {
    seed = (seed * 1103515245 + 12345) >>> 0; eD.value = "rand"; draw();
  });
  draw();
})();

/* ─────────────── 6 · the EXACT type I error of four tests ─────────────── */
(function () {
  const svg = d3.select("#lvl-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 430;
  const eP = document.getElementById("lv-p"), ePv = document.getElementById("lv-pv");
  const eN = document.getElementById("lv-n"), eNv = document.getElementById("lv-nv");
  const eA = document.getElementById("lv-a");
  const eY = document.getElementById("lv-y"), eG = document.getElementById("lv-g");
  const eF = document.getElementById("lv-f");
  const out = document.getElementById("lvl-readout");

  /* The exact level, by enumeration. Rows are two independent Binomial(n, p)
     samples, so the null distribution of the whole table is the outer product of
     two binomial pmfs — (n+1)² outcomes, each with a known weight. Summing the
     weights of the rejections IS the type I error: no simulation, no Monte-Carlo
     error, just a finite sum. Fisher's p is cached by (a, c) because it is by far
     the most expensive term.                                                     */
  /* Fisher's rejection decision, tabulated once per column total. Every table with
     the same C₁ shares one reference set, so the hypergeometric pmf is built 2n times
     instead of (n+1)² times, and each table's point-probability p-value is then a
     binary search into the sorted pmf rather than a fresh O(n) scan. That is the
     difference between this figure taking four seconds and taking four milliseconds. */
  function fisherTable(n, alpha) {
    const N = 2 * n, out2 = new Array(N + 1);
    for (let C1 = 1; C1 <= N - 1; C1++) {
      const lo = Math.max(0, C1 - n), hi = Math.min(n, C1), m = hi - lo + 1;
      const pm = new Float64Array(m);
      let tot = 0;
      for (let k = lo; k <= hi; k++) { const v = CT.hyperPmf(k, N, C1, n); pm[k - lo] = v; tot += v; }
      if (!(tot > 0)) { out2[C1] = null; continue; }
      for (let i = 0; i < m; i++) pm[i] /= tot;
      const idx = new Array(m);
      for (let i = 0; i < m; i++) idx[i] = i;
      idx.sort((u, v) => pm[u] - pm[v]);
      const sv = new Float64Array(m), pre = new Float64Array(m + 1);
      for (let i = 0; i < m; i++) { sv[i] = pm[idx[i]]; pre[i + 1] = pre[i] + sv[i]; }
      const rej = new Uint8Array(m);
      for (let i = 0; i < m; i++) {
        const t = pm[i] * (1 + 1e-9);
        let a2 = 0, b2 = m;
        while (a2 < b2) { const mid = (a2 + b2) >> 1; if (sv[mid] <= t) a2 = mid + 1; else b2 = mid; }
        rej[i] = pre[a2] < alpha ? 1 : 0;
      }
      out2[C1] = { lo: lo, rej: rej };
    }
    return out2;
  }

  function exactLevel(n, p, alpha, want) {
    const N = 2 * n, crit = ST.chi2Quant(1 - alpha, 1);
    const fis = want.f ? fisherTable(n, alpha) : null;
    const pb = [];
    for (let k = 0; k <= n; k++) pb.push(ST.binomPmf(k, n, p));
    let lx = 0, ly = 0, lg = 0, lf = 0, minE = 0;
    for (let a = 0; a <= n; a++) {
      const wa = pb[a];
      if (wa < 1e-14) continue;
      for (let c = 0; c <= n; c++) {
        const w = wa * pb[c];
        if (w < 1e-15) continue;
        const b = n - a, d = n - c, C1 = a + c, C2 = b + d;
        if (C1 === 0 || C2 === 0) continue;             // a dead column: no rejection
        const E11 = n * C1 / N, E12 = n * C2 / N, E21 = E11, E22 = E12;
        const dd = a - E11;                              // all four deviations are ±dd
        const inv = 1 / E11 + 1 / E12 + 1 / E21 + 1 / E22;
        const chi = dd * dd * inv;
        if (chi > crit) lx += w;
        minE += w * Math.min(E11, E12);
        if (want.y) {
          const yv = Math.max(0, Math.abs(dd) - 0.5);
          if (yv * yv * inv > crit) ly += w;
        }
        if (want.g) {
          let g2 = 0;
          if (a > 0) g2 += 2 * a * Math.log(a / E11);
          if (b > 0) g2 += 2 * b * Math.log(b / E12);
          if (c > 0) g2 += 2 * c * Math.log(c / E21);
          if (d > 0) g2 += 2 * d * Math.log(d / E22);
          if (g2 > crit) lg += w;
        }
        if (fis) {
          const t = fis[C1];
          if (t && t.rej[a - t.lo]) lf += w;
        }
      }
    }
    return { n: n, x: lx, y: ly, g: lg, f: lf, minE: minE };
  }

  function draw() {
    const p = +eP.value, nMax = +eN.value, alpha = +eA.value;
    ePv.textContent = ST.fmt(p, 2); eNv.textContent = nMax;
    const want = { y: eY.checked, g: eG.checked, f: eF.checked };

    const ns = [];
    const step = nMax > 45 ? 2 : 1;
    for (let n = 4; n <= nMax; n += step) ns.push(n);
    const rows = ns.map(n => exactLevel(n, p, alpha, want));

    const f = ST.frame(svg, W, H, { l: 56, r: 130, t: 22, b: 74 });
    const g = f.g, iw = f.iw, ih = f.ih - 54;
    const x = d3.scaleLinear().domain([4, nMax]).range([0, iw]);
    const top = Math.max(alpha * 2.6, d3.max(rows.map(r => Math.max(r.x, want.g ? r.g : 0))) * 1.15);
    const y = d3.scaleLinear().domain([0, top]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".3f")));
    g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 10.5).attr("fill", SC.muted)
      .text("true probability of a false rejection");
    g.append("text").attr("x", iw).attr("y", ih + 30).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("observations per group");

    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(alpha)).attr("y2", y(alpha))
      .attr("stroke", SC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 4");
    g.append("text").attr("x", iw + 4).attr("y", y(alpha) + 4).attr("font-size", 10)
      .attr("fill", SC.a2).text("nominal " + alpha);

    const line = key => d3.line().x(d => x(d.n)).y(d => y(ST.clamp(d[key], 0, top)));
    const series = [{ k: "x", c: SC.accent, l: "Pearson χ²", on: true },
      { k: "g", c: SC.bad, l: "likelihood ratio G²", on: want.g },
      { k: "y", c: SC.violet, l: "Yates corrected", on: want.y },
      { k: "f", c: SC.good, l: "Fisher exact", on: want.f }];
    series.forEach(s => {
      if (!s.on) return;
      g.append("path").attr("fill", "none").attr("stroke", s.c)
        .attr("stroke-width", s.k === "x" ? 2.3 : 1.7).attr("d", line(s.k)(rows));
    });
    ST.legend(g, series.filter(s => s.on).map(s => ({ label: s.l, color: s.c })), iw + 6, 14);

    /* the smallest expected count on its own strip underneath */
    const gs = CT.sub(g, 0, ih + 46, "");
    const ys = d3.scaleLinear().domain([0, d3.max(rows.map(r => r.minE)) * 1.1]).range([40, 0]);
    gs.append("path").attr("fill", SC.a2).attr("fill-opacity", 0.18)
      .attr("d", d3.area().x(d => x(d.n)).y0(40).y1(d => ys(d.minE))(rows));
    gs.append("line").attr("x1", 0).attr("x2", iw).attr("y1", ys(5)).attr("y2", ys(5))
      .attr("stroke", SC.a2).attr("stroke-dasharray", "3 3");
    gs.append("text").attr("x", iw + 4).attr("y", ys(5) + 4).attr("font-size", 9.5)
      .attr("fill", SC.a2).text("E = 5");
    gs.append("text").attr("x", 0).attr("y", 52).attr("font-size", 9.5).attr("fill", SC.muted)
      .text("mean smallest expected count — the quantity the conventional rule of thumb is about");

    const last = rows[rows.length - 1];
    const xs = rows.map(r => r.x);
    const lo = Math.min.apply(null, xs), hiV = Math.max.apply(null, xs);
    const argLo = rows[xs.indexOf(lo)].n, argHi = rows[xs.indexOf(hiV)].n;
    out.innerHTML =
      `Exact levels, by enumeration over all <b>${(last.n + 1) * (last.n + 1)}</b> outcomes at the largest size — no simulation error at all. ` +
      `Nominal <b>${alpha}</b>, common success probability <b>${ST.fmt(p, 2)}</b>. ` +
      `Pearson's true level ranges from <b>${ST.fmt(lo, 4)}</b> (at n = ${argLo}) to <b>${ST.fmt(hiV, 4)}</b> (at n = ${argHi}); ` +
      `at the largest size shown it is <b>${ST.fmt(last.x, 4)}</b>. ` +
      (want.g ? `<code>G²</code> is at ${ST.fmt(rows[0].g, 4)} for the smallest table and ${ST.fmt(last.g, 4)} for the largest — <b>above</b> nominal where the counts are thin. ` : "") +
      (want.y && want.f
        ? `Yates and Fisher track each other closely (${ST.fmt(last.y, 4)} against ${ST.fmt(last.f, 4)} at n = ${last.n}) because Yates' correction <i>is</i> an approximation to Fisher's exact test — and both stay well under the nominal level. `
        : "") +
      `The sawtooth is not noise: the statistic takes finitely many values, so the tail probability jumps as the achievable values shuffle past the critical value, and the true level oscillates rather than settling.`;
  }

  [eP, eN, eA, eY, eG, eF].forEach(el => el.addEventListener("change", draw));
  [eP, eN].forEach(el => el.addEventListener("input", draw));
  draw();
})();

/* ─────────────── 7 · Fisher's reference set, enumerated ─────────────── */
(function () {
  const svg = d3.select("#fis-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const E = { a: document.getElementById("fs-a"), b: document.getElementById("fs-b"),
    c: document.getElementById("fs-c"), d: document.getElementById("fs-d") };
  const V = { a: document.getElementById("fs-av"), b: document.getElementById("fs-bv"),
    c: document.getElementById("fs-cv"), d: document.getElementById("fs-dv") };
  const eS = document.getElementById("fs-side"), eP = document.getElementById("fs-perm");
  const out = document.getElementById("fis-readout");

  function draw() {
    const a = +E.a.value, b = +E.b.value, c = +E.c.value, d = +E.d.value;
    ["a", "b", "c", "d"].forEach(k => { V[k].textContent = E[k].value; });
    const side = eS.value;
    const fi = CT.fisher(a, b, c, d, side);
    const s = CT.stats([[a, b], [c, d]]);
    const me = CT.measures(a, b, c, d, false);

    const f = ST.frame(svg, W, H, { l: 14, r: 14, t: 14, b: 14 });
    const g = f.g;

    /* ── the observed table, top left ──────────────────────────────── */
    const gt = CT.sub(g, 26, 34, "the observed table, and its fixed margins");
    const cw = 62, ch = 40;
    [["", "col 1", "col 2", "total"], ["row 1", a, b, fi.R1],
      ["row 2", c, d, fi.R2], ["total", fi.C1, fi.C2, fi.N]].forEach((row, i) => {
      row.forEach((v, j) => {
        const isM = (i === 3 || j === 3), isCell = (i > 0 && i < 3 && j > 0 && j < 3);
        if (i > 0 && j > 0) gt.append("rect").attr("x", j * cw - cw + 34).attr("y", i * ch - ch + 6)
          .attr("width", cw - 6).attr("height", ch - 6).attr("rx", 4)
          .attr("fill", isM ? SC.panel2 : (isCell ? SC.accent : SC.panel2))
          .attr("fill-opacity", isCell ? 0.22 : 1).attr("stroke", isM ? SC.a2 : SC.line);
        gt.append("text").attr("x", j * cw - cw + 34 + (j === 0 ? -8 : (cw - 6) / 2))
          .attr("y", i * ch - ch + 6 + ch / 2)
          .attr("text-anchor", j === 0 ? "end" : "middle").attr("font-size", j === 0 || i === 0 ? 10 : 13)
          .attr("font-weight", isCell ? 600 : 400)
          .attr("fill", isM ? SC.a2 : (i === 0 || j === 0 ? SC.accent : SC.ink)).text(v);
      });
    });
    gt.append("text").attr("x", 0).attr("y", 4 * ch + 4).attr("font-size", 10).attr("fill", SC.muted)
      .text("the margins are FIXED — no relabelling can move them");

    /* ── the reference set as a bar chart ──────────────────────────── */
    const bw = 420, bh = 236;
    const gb = CT.sub(g, 268, 34, "every table with those margins, and its probability");
    const x = d3.scaleBand().domain(fi.ks).range([0, bw]).padding(0.14);
    const yMax = Math.max(1e-4, d3.max(fi.pm)) * 1.16;
    const y = d3.scaleLinear().domain([0, yMax]).range([bh, 0]);
    ST.gridY(gb, y, bw, 4);
    gb.append("g").attr("class", "axis").attr("transform", `translate(0,${bh})`)
      .call(d3.axisBottom(x).tickValues(fi.ks.filter((_, i) => fi.ks.length < 16 || i % Math.ceil(fi.ks.length / 14) === 0)));
    gb.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".2f")));
    gb.append("text").attr("x", bw).attr("y", bh + 32).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("the top-left count, which determines the whole table");

    fi.ks.forEach((k, i) => {
      const inTail = fi.keep(i), isObs = (k === a);
      gb.append("rect").attr("x", x(k)).attr("y", y(fi.pm[i]))
        .attr("width", x.bandwidth()).attr("height", Math.max(0, bh - y(fi.pm[i]))).attr("rx", 2)
        .attr("fill", inTail ? SC.bad : SC.accent).attr("fill-opacity", inTail ? 0.72 : 0.34)
        .attr("stroke", isObs ? SC.ink : "none").attr("stroke-width", isObs ? 2 : 0);
    });
    /* the hypergeometric mean, which is exactly the χ² expected count */
    gb.append("line").attr("x1", x.range()[0] + (fi.mean - fi.lo + 0.5) * (bw / fi.ks.length))
      .attr("x2", x.range()[0] + (fi.mean - fi.lo + 0.5) * (bw / fi.ks.length))
      .attr("y1", 0).attr("y2", bh).attr("stroke", SC.a2).attr("stroke-dasharray", "4 3");
    gb.append("text").attr("x", x.range()[0] + (fi.mean - fi.lo + 0.5) * (bw / fi.ks.length))
      .attr("y", -4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.a2)
      .text("E₁₁ = " + ST.fmt(fi.mean, 2));
    if (x(a) !== undefined) gb.append("text").attr("x", x(a) + x.bandwidth() / 2)
      .attr("y", y(fi.obs) - 6).attr("text-anchor", "middle").attr("font-size", 10)
      .attr("font-weight", 600).attr("fill", SC.ink).text("observed");

    /* ── the permutation view, if asked for ────────────────────────── */
    if (eP.checked) {
      const gp = CT.sub(g, 26, 330, "the same reference set as a relabelling of fixed units");
      const N = fi.N, per = Math.min(N, 46), sz = Math.min(13, 420 / per);
      const r = ST.rng(97 + a * 13 + b * 7 + c * 3 + d);
      /* the units: C₁ successes and C₂ failures, in a fixed strip */
      for (let i = 0; i < Math.min(N, per); i++) {
        gp.append("rect").attr("x", i * (sz + 2)).attr("y", 0).attr("width", sz).attr("height", sz)
          .attr("rx", 2).attr("fill", i < Math.round(fi.C1 * per / N) ? SC.good : SC.muted)
          .attr("fill-opacity", 0.8);
      }
      gp.append("text").attr("x", 0).attr("y", sz + 14).attr("font-size", 10).attr("fill", SC.muted)
        .text(`the ${N} units and their outcomes — ${fi.C1} of one kind, ${fi.C2} of the other. These never change.`);
      /* one random relabelling, and where it lands */
      const lab = ST.shuffle(d3.range(Math.min(N, per)).map(i => i < Math.round(fi.R1 * per / N) ? 1 : 0), r);
      for (let i = 0; i < Math.min(N, per); i++) {
        gp.append("rect").attr("x", i * (sz + 2)).attr("y", sz + 24).attr("width", sz).attr("height", sz)
          .attr("rx", 2).attr("fill", lab[i] ? SC.accent : SC.violet).attr("fill-opacity", 0.75);
      }
      gp.append("text").attr("x", 0).attr("y", 2 * sz + 38).attr("font-size", 10).attr("fill", SC.muted)
        .text(`one of the C(${N}, ${fi.R1}) ways of dealing the group labels. Each one lands on exactly one bar above.`);
    }

    /* ── the comparison chips ──────────────────────────────────────── */
    const cy = eP.checked ? 424 : 336;
    CT.chip(g, 268, cy - 6, "FISHER, THIS CONVENTION", CT.pstr(fi.p), fi.p < 0.05 ? SC.bad : SC.good, 168);
    CT.chip(g, 444, cy - 6, "PEARSON χ² APPROXIMATION", CT.pstr(s.p), s.p < 0.05 ? SC.bad : SC.good, 176);
    CT.chip(g, 628, cy - 6, "SMALLEST E", ST.fmt(s.minE, 2), s.minE < 5 ? SC.a2 : SC.good, 96);
    if (!eP.checked) {
      CT.chip(g, 26, cy - 6, "ODDS RATIO", isFinite(me.or) ? ST.fmt(me.or, 3) : "undefined", SC.violet, 118);
      CT.chip(g, 26, cy + 36, "TABLES IN THE SET", String(fi.ks.length), SC.ink, 118);
      CT.chip(g, 152, cy + 36, "LABELLINGS", CT.bigN(Math.exp(ST.lnChoose(fi.N, fi.R1))), SC.muted, 108);
    }

    const disagree = (s.p < 0.05) !== (fi.p < 0.05);
    out.innerHTML =
      `Margins ${fi.R1}, ${fi.R2} by ${fi.C1}, ${fi.C2} on n = ${fi.N} — <b>${fi.ks.length}</b> tables in the reference set, ` +
      `reached by <code>C(${fi.N}, ${fi.R1}) = ${CT.bigN(Math.exp(ST.lnChoose(fi.N, fi.R1)))}</code> relabellings. ` +
      `The observed table has probability <b>${ST.fmt(fi.obs, 6)}</b>. ` +
      `Tails: point-probability <b>${ST.fmt(fi.pts, 6)}</b> · doubling <b>${ST.fmt(fi.dbl, 6)}</b> · ` +
      `mid-p <b>${ST.fmt(fi.midp, 6)}</b> · one-sided right <b>${ST.fmt(fi.right, 6)}</b> · left <b>${ST.fmt(fi.left, 6)}</b>. ` +
      `Selected: <b>${CT.pstr(fi.p)}</b>. Pearson's approximation gives <b>${CT.pstr(s.p)}</b> ` +
      `(smallest expected count ${ST.fmt(s.minE, 2)}). ` +
      (disagree
        ? `<b>They disagree across the 5% line</b>${s.minE >= 5 ? " — and every expected count clears 5, so the rule of thumb gave no warning" : ""}. Report the exact one.`
        : `They agree at 5%.`) +
      ` The shaded bars <i>are</i> the p-value: nothing has been approximated, only counted.`;
  }

  ["a", "b", "c", "d"].forEach(k => E[k].addEventListener("input", draw));
  eS.addEventListener("change", draw);
  eP.addEventListener("change", draw);
  draw();
})();

/* ─────────────── 8 · χ², G² and Fisher on one shrinking table ─────────────── */
(function () {
  const svg = d3.select("#gvc-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 420;
  const e1 = document.getElementById("gv-p1"), e1v = document.getElementById("gv-p1v");
  const e0 = document.getElementById("gv-p2"), e0v = document.getElementById("gv-p2v");
  const eN = document.getElementById("gv-n"), eNv = document.getElementById("gv-nv");
  const eY = document.getElementById("gv-y"), eL = document.getElementById("gv-log");
  const out = document.getElementById("gvc-readout");

  function tableAt(n, p1, p0) {
    const a = Math.round(n * p1), c = Math.round(n * p0);
    return [[a, n - a], [c, n - c]];
  }
  function row(n, p1, p0) {
    const O = tableAt(n, p1, p0), s = CT.stats(O);
    const fi = CT.fisher(O[0][0], O[0][1], O[1][0], O[1][1], "two");
    return { n: n, chi: s.p, g: s.gp, y: s.yp, f: fi.p, minE: s.minE, O: O, s: s };
  }

  function draw() {
    const p1 = +e1.value, p0 = +e0.value, nSel = +eN.value;
    e1v.textContent = ST.fmt(p1, 2); e0v.textContent = ST.fmt(p0, 2); eNv.textContent = nSel;
    const rows = [];
    for (let n = 6; n <= 130; n++) rows.push(row(n, p1, p0));
    const here = row(nSel, p1, p0);

    const f = ST.frame(svg, W, H, { l: 62, r: 152, t: 22, b: 84 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLinear().domain([6, 130]).range([0, iw]);
    const logY = eL.checked;
    const all = rows.flatMap(r => [r.chi, r.g, r.f].filter(v => v > 0));
    const lowest = Math.max(1e-12, Math.min.apply(null, all));
    const y = logY
      ? d3.scaleLog().domain([Math.max(1e-10, lowest * 0.7), 1]).range([ih, 0]).clamp(true)
      : d3.scaleLinear().domain([0, 1]).range([ih, 0]);
    ST.gridY(g, y, iw, 5);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").call(logY
      ? d3.axisLeft(y).ticks(6, ".0e") : d3.axisLeft(y).ticks(5, ".2f"));
    g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 10.5).attr("fill", SC.muted)
      .text("p-value" + (logY ? " (log scale)" : ""));
    g.append("text").attr("x", iw).attr("y", ih + 32).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("observations per group");

    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(0.05)).attr("y2", y(0.05))
      .attr("stroke", SC.a2).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 4");
    g.append("text").attr("x", iw + 4).attr("y", y(0.05) + 4).attr("font-size", 10)
      .attr("fill", SC.a2).text("p = 0.05");

    const ser = [{ k: "chi", c: SC.accent, l: "Pearson χ²" },
      { k: "g", c: SC.bad, l: "likelihood ratio G²" },
      { k: "f", c: SC.good, l: "Fisher exact" }];
    if (eY.checked) ser.push({ k: "y", c: SC.violet, l: "Yates corrected" });
    ser.forEach(s => g.append("path").attr("fill", "none").attr("stroke", s.c)
      .attr("stroke-width", 1.9)
      .attr("d", d3.line().x(d => x(d.n)).y(d => y(ST.clamp(d[s.k], y.domain()[0], 1)))(rows)));
    ST.legend(g, ser.map(s => ({ label: s.l, color: s.c })), iw + 6, 14);

    g.append("line").attr("x1", x(nSel)).attr("x2", x(nSel)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.ink).attr("stroke-width", 1.2).attr("stroke-opacity", 0.7);
    ser.forEach(s => g.append("circle").attr("cx", x(nSel))
      .attr("cy", y(ST.clamp(here[s.k], y.domain()[0], 1))).attr("r", 3.4).attr("fill", s.c));

    /* the crossing points: the smallest n at which each test first calls it */
    const cross = {};
    ser.forEach(s => {
      for (let i = 0; i < rows.length; i++) if (rows[i][s.k] < 0.05) { cross[s.k] = rows[i].n; break; }
    });

    /* the current table, printed under the axis */
    const gt = CT.sub(g, 0, ih + 48, "");
    const O = here.O;
    [["", "yes", "no"], ["group 1", O[0][0], O[0][1]], ["group 2", O[1][0], O[1][1]]]
      .forEach((r2, i) => r2.forEach((v, j) => {
        gt.append("text").attr("x", 4 + j * 62).attr("y", i * 14)
          .attr("font-size", 10.5).attr("fill", i === 0 || j === 0 ? SC.accent : SC.ink).text(v);
      }));
    gt.append("text").attr("x", 210).attr("y", 0).attr("font-size", 10.5).attr("fill", SC.muted)
      .text(`n = ${nSel} per group · smallest expected count ${ST.fmt(here.minE, 2)} · χ² = ${ST.fmt(here.s.chi2, 3)}`);
    gt.append("text").attr("x", 210).attr("y", 14).attr("font-size", 10.5).attr("fill", SC.muted)
      .text(`the proportions ${ST.pct(p1, 0)} against ${ST.pct(p0, 0)} are held fixed at every n on this axis`);

    out.innerHTML =
      `Rates <b>${ST.pct(p1, 1)}</b> against <b>${ST.pct(p0, 1)}</b>, held fixed while <code>n</code> varies. ` +
      `At <code>n = ${nSel}</code> per group: <code>χ²</code> p = <b>${CT.pstr(here.chi)}</b>, ` +
      `<code>G²</code> p = <b>${CT.pstr(here.g)}</b>, Fisher p = <b>${CT.pstr(here.f)}</b>` +
      (eY.checked ? `, Yates p = <b>${CT.pstr(here.y)}</b>` : "") +
      ` (smallest expected count ${ST.fmt(here.minE, 2)}). ` +
      `First group size at which each test crosses 0.05: ` +
      `<b>χ² at n = ${cross.chi || "—"}</b>, <b>G² at n = ${cross.g || "—"}</b>, <b>Fisher at n = ${cross.f || "—"}</b>` +
      (eY.checked ? `, <b>Yates at n = ${cross.y || "—"}</b>` : "") + `. ` +
      `The approximations declare significance on a smaller study than the exact test does; the gap is the conservatism §10 measured, ` +
      `and it closes as the counts grow.`;
  }

  [e1, e0, eN].forEach(el => el.addEventListener("input", draw));
  [eY, eL].forEach(el => el.addEventListener("change", draw));
  draw();
})();

/* ─────────────── 9 · paired against unpaired ─────────────── */
(function () {
  const svg = d3.select("#mcn-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 440;
  const E = { a: document.getElementById("mc-a"), b: document.getElementById("mc-b"),
    c: document.getElementById("mc-c"), d: document.getElementById("mc-d") };
  const V = { a: document.getElementById("mc-av"), b: document.getElementById("mc-bv"),
    c: document.getElementById("mc-cv"), d: document.getElementById("mc-dv") };
  const eX = document.getElementById("mc-exact");
  const out = document.getElementById("mcn-readout");

  function draw() {
    const a = +E.a.value, b = +E.b.value, c = +E.c.value, d = +E.d.value;
    ["a", "b", "c", "d"].forEach(k => { V[k].textContent = E[k].value; });
    const m = CT.mcnemar(a, b, c, d), n = a + b + c + d;

    const f = ST.frame(svg, W, H, { l: 14, r: 14, t: 14, b: 14 });
    const g = f.g;

    /* ── left: the paired table, concordant cells greyed ───────────── */
    const gt = CT.sub(g, 40, 36, "the paired table — each cell counts PAIRS");
    const cw = 96, ch = 62;
    const cells = [{ i: 0, j: 0, v: a, lab: "yes → yes", disc: false },
      { i: 0, j: 1, v: b, lab: "yes → no", disc: true },
      { i: 1, j: 0, v: c, lab: "no → yes", disc: true },
      { i: 1, j: 1, v: d, lab: "no → no", disc: false }];
    cells.forEach(cl => {
      gt.append("rect").attr("x", cl.j * cw).attr("y", cl.i * ch).attr("width", cw - 6)
        .attr("height", ch - 6).attr("rx", 5)
        .attr("fill", cl.disc ? (cl.j === 1 ? SC.bad : SC.good) : SC.muted)
        .attr("fill-opacity", cl.disc ? 0.35 : 0.10)
        .attr("stroke", cl.disc ? (cl.j === 1 ? SC.bad : SC.good) : SC.line)
        .attr("stroke-width", cl.disc ? 1.8 : 0.8);
      gt.append("text").attr("x", cl.j * cw + (cw - 6) / 2).attr("y", cl.i * ch + ch / 2 - 2)
        .attr("text-anchor", "middle").attr("font-size", 17).attr("font-weight", 600)
        .attr("fill", cl.disc ? SC.ink : SC.muted).text(cl.v);
      gt.append("text").attr("x", cl.j * cw + (cw - 6) / 2).attr("y", cl.i * ch + ch / 2 + 14)
        .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", SC.muted).text(cl.lab);
    });
    gt.append("text").attr("x", 0).attr("y", 2 * ch + 14).attr("font-size", 10.5).attr("fill", SC.muted)
      .text(`the ${a + d} concordant pairs are IGNORED — they carry no information about a change`);
    gt.append("text").attr("x", 0).attr("y", 2 * ch + 28).attr("font-size", 10.5).attr("fill", SC.ink)
      .text(`only the ${m.m} discordant pairs are used`);

    /* the two marginal proportions as bars */
    const gm = CT.sub(g, 40, 36 + 2 * ch + 62, "the two marginal rates being compared");
    const bw = 168;
    [["before", m.m1, SC.accent], ["after", m.m2, SC.a2]].forEach((r, i) => {
      gm.append("text").attr("x", 0).attr("y", i * 28 + 12).attr("font-size", 10.5)
        .attr("fill", SC.muted).text(r[0]);
      gm.append("rect").attr("x", 44).attr("y", i * 28).attr("width", bw).attr("height", 16)
        .attr("rx", 3).attr("fill", SC.panel2).attr("stroke", SC.line);
      gm.append("rect").attr("x", 44).attr("y", i * 28).attr("width", bw * r[1]).attr("height", 16)
        .attr("rx", 3).attr("fill", r[2]).attr("fill-opacity", 0.7);
      gm.append("text").attr("x", 44 + bw + 8).attr("y", i * 28 + 12).attr("font-size", 10.5)
        .attr("fill", SC.ink).text(ST.pct(r[1], 1));
    });

    /* ── right: the exact binomial null over the discordant pairs ──── */
    const rw = 360, rh = 208;
    const gr = CT.sub(g, 344, 36, `given ${m.m} discordant pairs, how many go one way?`);
    if (m.m > 0 && m.m <= 400) {
      const ks = d3.range(0, m.m + 1);
      const pm = ks.map(k => ST.binomPmf(k, m.m, 0.5));
      const x = d3.scaleLinear().domain([0, m.m]).range([0, rw]);
      const y = d3.scaleLinear().domain([0, Math.max(1e-6, d3.max(pm)) * 1.15]).range([rh, 0]);
      gr.append("g").attr("class", "axis").attr("transform", `translate(0,${rh})`)
        .call(d3.axisBottom(x).ticks(7));
      const bwid = Math.max(1, rw / (m.m + 1) - 1);
      const lo = Math.min(b, c), hiK = Math.max(b, c);
      ks.forEach((k, i) => {
        const tail = eX.checked && (k <= lo || k >= hiK);
        gr.append("rect").attr("x", x(k) - bwid / 2).attr("y", y(pm[i]))
          .attr("width", bwid).attr("height", Math.max(0, rh - y(pm[i])))
          .attr("fill", tail ? SC.bad : SC.accent).attr("fill-opacity", tail ? 0.75 : 0.34);
      });
      gr.append("line").attr("x1", x(m.m / 2)).attr("x2", x(m.m / 2)).attr("y1", 0).attr("y2", rh)
        .attr("stroke", SC.a2).attr("stroke-dasharray", "4 3");
      gr.append("text").attr("x", x(m.m / 2)).attr("y", -4).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.a2).text("null centre " + ST.fmt(m.m / 2, 1));
      gr.append("line").attr("x1", x(b)).attr("x2", x(b)).attr("y1", 0).attr("y2", rh)
        .attr("stroke", SC.ink).attr("stroke-width", 2);
      gr.append("text").attr("x", x(b)).attr("y", rh + 30).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.ink).text("observed b = " + b);
      gr.append("text").attr("x", 0).attr("y", rh + 46).attr("font-size", 10).attr("fill", SC.muted)
        .text("under the null this is exactly Binomial(" + m.m + ", ½) — no approximation");
    } else {
      gr.append("text").attr("x", rw / 2).attr("y", rh / 2).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", SC.muted)
        .text(m.m === 0 ? "no discordant pairs — nothing to test" : "too many pairs to draw");
    }

    const cy = 36 + rh + 74;
    CT.chip(g, 344, cy, "McNEMAR χ²", ST.fmt(m.chi, 4), SC.ink, 112);
    CT.chip(g, 464, cy, "ITS P-VALUE", CT.pstr(m.p), m.p < 0.05 ? SC.bad : SC.good, 112);
    CT.chip(g, 584, cy, "EXACT BINOMIAL", CT.pstr(m.exact), m.exact < 0.05 ? SC.bad : SC.good, 122);
    CT.chip(g, 344, cy + 42, "CONTINUITY CORRECTED", CT.pstr(m.ccp), m.ccp < 0.05 ? SC.bad : SC.good, 172);
    CT.chip(g, 524, cy + 42, "WRONG UNPAIRED TEST", CT.pstr(m.wrongP), SC.a2, 182);

    out.innerHTML =
      `n = ${n} pairs, of which <b>${m.m}</b> are discordant (${b} one way, ${c} the other) and ${a + d} are concordant. ` +
      `Marginal rates <b>${ST.pct(m.m1, 1)}</b> and <b>${ST.pct(m.m2, 1)}</b>, a change of ${ST.fmt(100 * (m.m2 - m.m1), 1)} points. ` +
      `McNemar <code>χ² = ${ST.fmt(m.chi, 4)}</code>, p = <b>${CT.pstr(m.p)}</b>; continuity corrected p = ${CT.pstr(m.ccp)}; ` +
      `exact binomial p = <b>${CT.pstr(m.exact)}</b>. ` +
      `The <b>incorrect</b> unpaired independence test on the two marginal proportions would report ` +
      `<code>χ² = ${ST.fmt(m.wrong, 4)}</code>, p = <b>${CT.pstr(m.wrongP)}</b>` +
      ((m.p < 0.05) !== (m.wrongP < 0.05)
        ? ` — <b>a different verdict at 5%</b>, and the pairing is the whole reason.`
        : `.`) + ` ` +
      `Move the two concordant sliders: McNemar's p does not budge, because those pairs contain no information about a change, ` +
      `while the unpaired test's p moves freely — which is the clearest possible sign that it is answering a different question.`;
  }

  ["a", "b", "c", "d"].forEach(k => E[k].addEventListener("input", draw));
  eX.addEventListener("change", draw);
  draw();
})();

/* ─────────────── 10 · three measures on the unit square ─────────────── */
(function () {
  const svg = d3.select("#eff-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 450;
  const eC = document.getElementById("ef-c"), eN = document.getElementById("ef-n");
  const eNv = document.getElementById("ef-nv"), eR = document.getElementById("ef-ratio");
  const out = document.getElementById("eff-readout");
  let p1 = 0.528, p0 = 0.392;                       // §07's survey, exactly

  const odds = p => p / (1 - p);
  const orOf = (u, v) => odds(u) / odds(v);

  function draw() {
    const n = +eN.value;
    eNv.textContent = CT.grp(n);
    const a = Math.round(n * p1), b = n - Math.round(n * p1);
    const c = Math.round(n * p0), d = n - Math.round(n * p0);
    const me = CT.measures(a, b, c, d, false);
    const s = CT.stats([[a, b], [c, d]]);

    const f = ST.frame(svg, W, H, { l: 58, r: 14, t: 22, b: 48 });
    const g = f.g;
    const SQ = 356;
    const x = d3.scaleLinear().domain([0, 1]).range([0, SQ]);
    const y = d3.scaleLinear().domain([0, 1]).range([SQ, 0]);

    /* the OR/RR overstatement, as a background field */
    if (eR.checked) {
      const K = 30, cell = SQ / K;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        const u = (i + 0.5) / K, v = (j + 0.5) / K;      // u = p₀, v = p₁
        const ratio = (1 - u) / (1 - v);                  // OR / RR
        const t = ST.clamp((Math.log(ratio)) / Math.log(6), -1, 1);
        g.append("rect").attr("x", i * cell).attr("y", SQ - (j + 1) * cell)
          .attr("width", cell + 0.5).attr("height", cell + 0.5)
          .attr("fill", t >= 0 ? SC.a2 : SC.teal).attr("fill-opacity", 0.30 * Math.abs(t));
      }
    }
    g.append("rect").attr("x", 0).attr("y", 0).attr("width", SQ).attr("height", SQ)
      .attr("fill", "none").attr("stroke", SC.line);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${SQ})`)
      .call(d3.axisBottom(x).ticks(6, ".1f"));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6, ".1f"));
    g.append("text").attr("x", SQ).attr("y", SQ + 34).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("risk in the unexposed group, p₀");
    g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 10.5).attr("fill", SC.muted)
      .text("risk in the exposed group, p₁");
    /* the no-association diagonal */
    g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1))
      .attr("stroke", SC.muted).attr("stroke-dasharray", "4 4");

    /* the contours */
    const grid = ST.linspace(0.002, 0.998, 300);
    function contour(kind, val, col, dash) {
      const pts = [];
      grid.forEach(u => {
        let v;
        if (kind === "or") { const o = val * odds(u); v = o / (1 + o); }
        else if (kind === "rr") v = val * u;
        else v = u + val;
        if (v > 0.001 && v < 0.999) pts.push({ u: u, v: v });
      });
      if (pts.length < 2) return;
      g.append("path").attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.85).attr("stroke-dasharray", dash || null)
        .attr("d", d3.line().x(p => x(p.u)).y(p => y(p.v))(pts));
    }
    const want = eC.value;
    const curOR = orOf(p1, p0), curRR = p1 / p0, curRD = p1 - p0;
    if (want === "or" || want === "all") {
      [0.25, 0.5, 1, 2, 4, 8].forEach(k => contour("or", k, SC.violet));
      contour("or", curOR, SC.violet, "5 3");
    }
    if (want === "rr" || want === "all") {
      [0.25, 0.5, 1, 1.5, 2, 3, 5].forEach(k => contour("rr", k, SC.teal));
      contour("rr", curRR, SC.teal, "5 3");
    }
    if (want === "rd" || want === "all") {
      [-0.4, -0.2, 0, 0.2, 0.4, 0.6].forEach(k => contour("rd", k, SC.good));
      contour("rd", curRD, SC.good, "5 3");
    }
    ST.legend(g, [{ label: "constant odds ratio", color: SC.violet },
      { label: "constant relative risk", color: SC.teal },
      { label: "constant risk difference", color: SC.good }].filter((_, i) =>
        want === "all" || (want === "or" && i === 0) || (want === "rr" && i === 1) || (want === "rd" && i === 2)),
      10, 14);

    /* the draggable point */
    const pt = g.append("circle").attr("class", "dragpt").attr("cx", x(p0)).attr("cy", y(p1))
      .attr("r", 8).attr("fill", SC.accent).attr("stroke", SC.ink).attr("stroke-width", 2)
      .style("cursor", "grab");
    pt.call(d3.drag().on("drag", ev => {
      p0 = ST.clamp(x.invert(ev.x), 0.01, 0.99);
      p1 = ST.clamp(y.invert(ev.y), 0.01, 0.99);
      draw();
    }));
    g.append("line").attr("x1", x(p0)).attr("x2", x(p0)).attr("y1", SQ).attr("y2", y(p1))
      .attr("stroke", SC.accent).attr("stroke-opacity", 0.35).attr("stroke-dasharray", "2 3");
    g.append("line").attr("x1", 0).attr("x2", x(p0)).attr("y1", y(p1)).attr("y2", y(p1))
      .attr("stroke", SC.accent).attr("stroke-opacity", 0.35).attr("stroke-dasharray", "2 3");

    /* the implied table and its summaries */
    const px = 428;
    g.append("text").attr("x", px).attr("y", 8).attr("font-size", 11.5).attr("font-weight", 600)
      .attr("fill", SC.ink).text(`the implied 2 × 2 at n = ${n} per arm`);
    const cw = 66, ch = 34;
    [["", "outcome", "not"], ["exposed", a, b], ["unexposed", c, d]].forEach((r, i) =>
      r.forEach((v, j) => {
        if (i > 0 && j > 0) g.append("rect").attr("x", px + 62 + (j - 1) * cw).attr("y", 18 + (i - 1) * ch)
          .attr("width", cw - 5).attr("height", ch - 5).attr("rx", 4)
          .attr("fill", SC.panel2).attr("stroke", SC.line);
        g.append("text").attr("x", px + (j === 0 ? 56 : 62 + (j - 1) * cw + (cw - 5) / 2))
          .attr("y", i === 0 ? 12 : 18 + (i - 1) * ch + ch / 2)
          .attr("text-anchor", j === 0 ? "end" : "middle")
          .attr("font-size", i === 0 || j === 0 ? 10 : 12.5)
          .attr("fill", i === 0 || j === 0 ? SC.accent : SC.ink).text(v);
      }));
    const gy = 18 + 2 * ch + 16;
    CT.chip(g, px, gy, "RISK DIFFERENCE", ST.fmt(me.rd, 4), SC.good, 138);
    CT.chip(g, px + 146, gy, "NUMBER NEEDED", ST.fmt(me.nnt, 2), SC.good, 118);
    CT.chip(g, px, gy + 42, "RELATIVE RISK", ST.fmt(me.rr, 4), SC.teal, 138);
    CT.chip(g, px + 146, gy + 42, "RR 95% CI",
      `${ST.fmt(me.rrLo, 3)} – ${ST.fmt(me.rrHi, 3)}`, SC.teal, 118);
    CT.chip(g, px, gy + 84, "ODDS RATIO", ST.fmt(me.or, 4), SC.violet, 138);
    CT.chip(g, px + 146, gy + 84, "OR 95% CI",
      `${ST.fmt(me.lo, 3)} – ${ST.fmt(me.hi, 3)}`, SC.violet, 118);
    CT.chip(g, px, gy + 126, "CHI-SQUARE p", CT.pstr(s.p), s.p < 0.05 ? SC.bad : SC.ink, 138);
    CT.chip(g, px + 146, gy + 126, "PHI", ST.fmt(s.phi, 4), SC.a2, 118);
    CT.chip(g, px, gy + 168, "OR ÷ RR", ST.fmt(me.or / me.rr, 4),
      me.or / me.rr > 1.15 ? SC.bad : SC.good, 138);
    CT.chip(g, px + 146, gy + 168, "(1 − p₀)/(1 − p₁)", ST.fmt((1 - p0) / (1 - p1), 4), SC.muted, 118);

    const over = me.or / me.rr;
    out.innerHTML =
      `<code>p₁ = ${ST.fmt(p1, 4)}</code>, <code>p₀ = ${ST.fmt(p0, 4)}</code>. ` +
      `Risk difference <b>${ST.fmt(me.rd, 4)}</b>, relative risk <b>${ST.fmt(me.rr, 4)}</b>, odds ratio <b>${ST.fmt(me.or, 4)}</b>. ` +
      `The identity <code>OR = RR·(1 − p₀)/(1 − p₁)</code> checks: ` +
      `${ST.fmt(me.rr, 4)} × ${ST.fmt((1 - p0) / (1 - p1), 4)} = ${ST.fmt(me.rr * (1 - p0) / (1 - p1), 4)}. ` +
      (over > 1.5 ? `The odds ratio is <b>${ST.pct(over - 1, 0)} larger</b> than the relative risk — the outcome is common, and calling this "${ST.fmt(me.or, 2)} times more likely" would be badly wrong. `
        : over > 1.12 ? `The odds ratio overstates the relative risk by <b>${ST.pct(over - 1, 0)}</b>. `
          : `The outcome is rare enough that the two ratios nearly coincide (a gap of ${ST.pct(over - 1, 1)}), which is the rare-disease approximation §16 relies on. `) +
      `Drag the point along a dashed contour and the corresponding measure stays fixed while the other two move — ` +
      `which is exactly why the three cannot be substituted for one another.`;
  }

  [eC, eR].forEach(el => el.addEventListener("change", draw));
  eN.addEventListener("input", draw);
  document.getElementById("ef-survey").addEventListener("click", () => { p1 = 0.528; p0 = 0.392; draw(); });
  document.getElementById("ef-rare").addEventListener("click", () => { p1 = 0.04; p0 = 0.02; draw(); });
  document.getElementById("ef-common").addEventListener("click", () => { p1 = 0.80; p0 = 0.40; draw(); });
  draw();
})();

/* ─────────────── 11 · the log-odds-ratio interval and its real coverage ─────────────── */
(function () {
  const svg = d3.select("#orci-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 450;
  const eN = document.getElementById("oc-n"), eNv = document.getElementById("oc-nv");
  const e1 = document.getElementById("oc-p1"), e1v = document.getElementById("oc-p1v");
  const e0 = document.getElementById("oc-p0"), e0v = document.getElementById("oc-p0v");
  const eH = document.getElementById("oc-ha");
  const out = document.getElementById("orci-readout");
  let seed = 20260908;

  function draw() {
    const n = +eN.value, p1 = +e1.value, p0 = +e0.value, ha = eH.checked;
    eNv.textContent = n; e1v.textContent = ST.fmt(p1, 2); e0v.textContent = ST.fmt(p0, 2);
    const truth = Math.log((p1 / (1 - p1)) / (p0 / (1 - p0)));
    const r = ST.rng(seed);
    const B = 6000;
    const reps = [], keep = [];
    let finite = 0, cover = 0, coverFinite = 0;
    for (let t = 0; t < B; t++) {
      const a = ST.binom(n, p1, r), c = ST.binom(n, p0, r);
      const b = n - a, d = n - c;
      const me = CT.measures(a, b, c, d, ha);
      const good = isFinite(me.lnor) && isFinite(me.se);
      if (good) {
        finite++;
        const lo = me.lnor - 1.959964 * me.se, hi = me.lnor + 1.959964 * me.se;
        const in_ = (lo <= truth && truth <= hi);
        if (in_) { cover++; coverFinite++; }
        reps.push(me.lnor);
        if (keep.length < 44) keep.push({ lo: lo, hi: hi, est: me.lnor, in: in_ });
      } else {
        reps.push(a * d > b * c ? Infinity : -Infinity);
        if (keep.length < 44) keep.push({ lo: NaN, hi: NaN, est: NaN, in: false });
      }
    }
    const covAll = cover / B, covFin = finite ? coverFinite / finite : NaN;

    const f = ST.frame(svg, W, H, { l: 54, r: 14, t: 22, b: 46 });
    const g = f.g;

    /* ── left: the histogram of ln OR̂ ─────────────────────────────── */
    const hw = 386, hh = 300;
    const gh = CT.sub(g, 0, 30, "the sampling distribution of ln OR̂ over " + CT.grp(B) + " simulated studies");
    const fin = reps.filter(isFinite);
    const spread = fin.length ? Math.max(1.2, 3.4 * ST.sd(fin)) : 3;
    const dom = [truth - spread, truth + spread];
    const x = d3.scaleLinear().domain(dom).range([0, hw]);
    const K = 42, wbin = (dom[1] - dom[0]) / K;
    const bins = ST.histBins(fin.filter(v => v >= dom[0] && v <= dom[1]), dom[0], wbin, K);
    const yMax = Math.max(1e-9, d3.max(bins.map(b2 => b2.n))) * 1.2;
    const y = d3.scaleLinear().domain([0, yMax]).range([hh, 0]);
    gh.append("g").attr("class", "axis").attr("transform", `translate(0,${hh})`)
      .call(d3.axisBottom(x).ticks(7, ".1f"));
    gh.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    bins.forEach(b2 => gh.append("rect").attr("x", x(b2.x0)).attr("y", y(b2.n))
      .attr("width", Math.max(1, x(b2.x1) - x(b2.x0) - 1)).attr("height", Math.max(0, hh - y(b2.n)))
      .attr("fill", SC.accent).attr("fill-opacity", 0.42));
    /* the theoretical normal, using the expected-cell standard error */
    const seTh = Math.sqrt(1 / (n * p1) + 1 / (n * (1 - p1)) + 1 / (n * p0) + 1 / (n * (1 - p0)));
    const curve = ST.linspace(dom[0], dom[1], 240).map(v => ({
      v: v, y: fin.length * wbin * ST.normPdf((v - truth) / seTh) / seTh
    }));
    gh.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2)
      .attr("d", d3.line().x(d => x(d.v)).y(d => y(Math.min(d.y, yMax)))(curve));
    CT.rule(gh, x(truth), hh, SC.good, "true ln OR = " + ST.fmt(truth, 3), null, -4);
    /* the infinite spikes, drawn at the edges so they cannot be missed */
    const nInf = reps.filter(v => v === Infinity).length, nNeg = reps.filter(v => v === -Infinity).length;
    [[nNeg, 0, "−∞"], [nInf, hw, "+∞"]].forEach(sp => {
      if (!sp[0]) return;
      const hgt = Math.min(hh, hh * sp[0] / (yMax));
      gh.append("rect").attr("x", sp[1] === 0 ? 0 : hw - 12).attr("y", hh - hgt)
        .attr("width", 12).attr("height", hgt).attr("fill", SC.bad).attr("fill-opacity", 0.8);
      gh.append("text").attr("x", sp[1] === 0 ? 6 : hw - 6).attr("y", hh - hgt - 5)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.bad)
        .text(sp[2] + " ×" + sp[0]);
    });
    gh.append("text").attr("x", hw).attr("y", hh + 32).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("estimated log odds ratio");

    /* ── right: the last 44 intervals ──────────────────────────────── */
    const iw2 = 234;
    const gi = CT.sub(g, 428, 30, "the first 44 intervals");
    const xi = d3.scaleLinear().domain(dom).range([0, iw2]);
    gi.append("line").attr("x1", xi(truth)).attr("x2", xi(truth)).attr("y1", 0).attr("y2", hh)
      .attr("stroke", SC.good).attr("stroke-width", 1.6);
    keep.forEach((k, i) => {
      const yy = 4 + i * (hh - 8) / keep.length;
      if (!isFinite(k.lo)) {
        gi.append("text").attr("x", 0).attr("y", yy + 3).attr("font-size", 8)
          .attr("fill", SC.bad).text("no estimate — a cell was empty");
        return;
      }
      gi.append("line").attr("x1", xi(ST.clamp(k.lo, dom[0], dom[1])))
        .attr("x2", xi(ST.clamp(k.hi, dom[0], dom[1]))).attr("y1", yy).attr("y2", yy)
        .attr("stroke", k.in ? SC.accent : SC.bad).attr("stroke-width", 2)
        .attr("stroke-opacity", k.in ? 0.65 : 1);
      gi.append("circle").attr("cx", xi(ST.clamp(k.est, dom[0], dom[1]))).attr("cy", yy)
        .attr("r", 1.6).attr("fill", k.in ? SC.accent : SC.bad);
    });
    gi.append("g").attr("class", "axis").attr("transform", `translate(0,${hh})`)
      .call(d3.axisBottom(xi).ticks(4, ".1f"));

    const cy = 30 + hh + 44;
    CT.chip(g, 0, cy, "COVERAGE, ALL SAMPLES", ST.pct(covAll, 2),
      covAll < 0.90 ? SC.bad : SC.good, 168);
    CT.chip(g, 176, cy, "COVERAGE, WHERE DEFINED", isFinite(covFin) ? ST.pct(covFin, 2) : "—", SC.ink, 184);
    CT.chip(g, 368, cy, "FRACTION WITH AN ESTIMATE", ST.pct(finite / B, 2),
      finite / B < 0.99 ? SC.bad : SC.good, 190);
    CT.chip(g, 566, cy, "TRUE ln OR", ST.fmt(truth, 4), SC.good, 110);

    /* exact: a row contributes an empty cell iff it is all successes or all failures,
       and the two rows are independent — so P(some empty cell) = 1 − (1−q₁)(1−q₀). */
    const q1 = Math.pow(1 - p1, n) + Math.pow(p1, n);
    const q0 = Math.pow(1 - p0, n) + Math.pow(p0, n);
    const pZero = 1 - (1 - q1) * (1 - q0);
    out.innerHTML =
      `n = <b>${n}</b> per arm, true risks ${ST.pct(p1, 1)} and ${ST.pct(p0, 1)}, true <code>ln OR = ${ST.fmt(truth, 4)}</code> ` +
      `(<code>OR = ${ST.fmt(Math.exp(truth), 4)}</code>). ${ha ? "<b>Haldane–Anscombe +½ applied.</b> " : "No correction. "}` +
      `Over ${CT.grp(B)} simulated studies: <b>${ST.pct(finite / B, 2)}</b> produced a finite estimate, ` +
      `and the nominal 95% interval covered the truth <b>${ST.pct(covAll, 2)}</b> of the time overall` +
      (finite < B ? ` — but <b>${ST.pct(covFin, 2)}</b> among the studies that produced an estimate at all, which is a selection artefact, not a success. `
        : `. `) +
      `Theoretical SE from the expected cells: <b>${ST.fmt(seTh, 4)}</b>; measured spread of the finite estimates: ` +
      `<b>${ST.fmt(fin.length > 1 ? ST.sd(fin) : NaN, 4)}</b>. ` +
      (finite / B < 0.995
        ? `The probability that some cell comes out empty is about <b>${ST.pct(Math.min(1, pZero), 1)}</b> at these settings — tick the correction and every study yields an interval, at the price of intervals that are too wide.`
        : `With counts this size the interval is honest; shrink <code>n</code> or push a risk toward 0 and the empty-cell spikes appear at the edges of the histogram.`);
  }

  [eN, e1, e0].forEach(el => el.addEventListener("input", draw));
  eH.addEventListener("change", draw);
  document.getElementById("oc-run").addEventListener("click", () => {
    seed = (seed * 1103515245 + 12345) >>> 0; draw();
  });
  draw();
})();

/* ─────────────── 12 · one population, two designs ─────────────── */
(function () {
  const svg = d3.select("#cc-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 430;
  const ePe = document.getElementById("cc-pe"), ePev = document.getElementById("cc-pev");
  const eR1 = document.getElementById("cc-r1"), eR1v = document.getElementById("cc-r1v");
  const eR0 = document.getElementById("cc-r0"), eR0v = document.getElementById("cc-r0v");
  const eK = document.getElementById("cc-k"), eKv = document.getElementById("cc-kv");
  const out = document.getElementById("cc-readout");
  const NPOP = 20000;

  function draw() {
    const pe = +ePe.value, r1 = +eR1.value, r0 = +eR0.value, k = +eK.value;
    ePev.textContent = ST.fmt(pe, 2); eR1v.textContent = ST.fmt(r1, 3);
    eR0v.textContent = ST.fmt(r0, 3); eKv.textContent = ST.fmt(k, 2);

    /* the population, as counts */
    const nE = Math.round(NPOP * pe), nU = NPOP - nE;
    const a = Math.round(nE * r1), b = nE - a;
    const c = Math.round(nU * r0), d = nU - c;
    const cases = a + c, ctrls = b + d;
    const trueRD = a / nE - c / nU, trueRR = (a / nE) / (c / nU), trueOR = (a * d) / (b * c);

    /* the case-control sample: all cases, k controls per case, sampled at random
       from the controls — so the exposed/unexposed split among the sampled controls
       is in expectation proportional to b : d. */
    const nCtrl = Math.round(k * cases);
    const sb = nCtrl * b / ctrls, sd = nCtrl * d / ctrls;
    const ccOR = (a * sd) / (sb * c);
    const ccRR = (a / (a + sb)) / (c / (c + sd));
    const ccRD = a / (a + sb) - c / (c + sd);

    const f = ST.frame(svg, W, H, { l: 14, r: 14, t: 14, b: 14 });
    const g = f.g;

    function panel(gx, title, A, B, C, D, labelB, note) {
      const gp = CT.sub(g, gx, 38, title);
      const tot = A + B + C + D, PW = 250, PH = 150;
      const wCase = PW * (A + C) / tot, wCtl = PW - wCase;
      const rows = [[A, B, "exposed"], [C, D, "unexposed"]];
      let yy = 0;
      rows.forEach((r, i) => {
        const h = PH * (r[0] + r[1]) / tot;
        gp.append("rect").attr("x", 0).attr("y", yy).attr("width", PW * r[0] / (r[0] + r[1]))
          .attr("height", h - 2).attr("fill", SC.bad).attr("fill-opacity", 0.55);
        gp.append("rect").attr("x", PW * r[0] / (r[0] + r[1])).attr("y", yy)
          .attr("width", PW * r[1] / (r[0] + r[1])).attr("height", h - 2)
          .attr("fill", SC.accent).attr("fill-opacity", 0.30);
        gp.append("text").attr("x", -6).attr("y", yy + h / 2).attr("text-anchor", "end")
          .attr("font-size", 10).attr("fill", SC.accent).text(r[2]);
        gp.append("text").attr("x", 4).attr("y", yy + h / 2 + 4).attr("font-size", 10.5)
          .attr("fill", SC.ink).text(CT.grp(r[0]));
        gp.append("text").attr("x", PW - 4).attr("y", yy + h / 2 + 4).attr("text-anchor", "end")
          .attr("font-size", 10.5).attr("fill", SC.muted).text(CT.grp(r[1]));
        yy += h;
      });
      gp.append("text").attr("x", 0).attr("y", PH + 14).attr("font-size", 9.5).attr("fill", SC.bad)
        .text("cases " + CT.grp(A + C));
      gp.append("text").attr("x", PW).attr("y", PH + 14).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", SC.accent).text(labelB + " " + CT.grp(Math.round(B + D)));
      if (note) gp.append("text").attr("x", 0).attr("y", PH + 30).attr("font-size", 9.5)
        .attr("fill", SC.muted).text(note);
      return PH;
    }

    panel(78, "the whole population — every measure is estimable", a, b, c, d, "healthy",
      `exposure prevalence ${ST.pct(pe, 0)}, risks ${ST.pct(r1, 1)} and ${ST.pct(r0, 1)}`);
    panel(432, `the case-control study — ${k === 1 ? "one control" : ST.fmt(k, 2) + " controls"} per case`, a, sb, c, sd, "controls",
      `all ${CT.grp(cases)} cases plus ${CT.grp(nCtrl)} controls drawn from the ${CT.grp(ctrls)}`);

    /* the three measures, side by side */
    const gy = 38 + 150 + 56;
    const items = [
      ["ODDS RATIO", trueOR, ccOR, SC.violet],
      ["RELATIVE RISK", trueRR, ccRR, SC.teal],
      ["RISK DIFFERENCE", trueRD, ccRD, SC.good]
    ];
    items.forEach((it, i) => {
      const gx = 78 + i * 218;
      const same = Math.abs(it[1] - it[2]) < 1e-9 * Math.max(1, Math.abs(it[1]));
      g.append("text").attr("x", gx).attr("y", gy).attr("font-size", 10)
        .attr("letter-spacing", 0.4).attr("fill", it[3]).text(it[0]);
      g.append("text").attr("x", gx).attr("y", gy + 20).attr("font-size", 13)
        .attr("fill", SC.ink).text("population  " + ST.fmt(it[1], 4));
      g.append("text").attr("x", gx).attr("y", gy + 38).attr("font-size", 13)
        .attr("font-weight", 600).attr("fill", same ? SC.good : SC.bad)
        .text("case-control  " + ST.fmt(it[2], 4));
      g.append("text").attr("x", gx).attr("y", gy + 54).attr("font-size", 9.5)
        .attr("fill", same ? SC.good : SC.bad)
        .text(same ? "identical — the sampling fractions cancel"
          : "different, and it moves with the control ratio");
    });

    out.innerHTML =
      `A population of ${CT.grp(NPOP)} with exposure prevalence ${ST.pct(pe, 0)} and risks ` +
      `${ST.pct(r1, 2)} (exposed) against ${ST.pct(r0, 2)} (unexposed). True <b>RR = ${ST.fmt(trueRR, 4)}</b>, ` +
      `<b>OR = ${ST.fmt(trueOR, 4)}</b>, <b>RD = ${ST.fmt(trueRD, 4)}</b>. ` +
      `A case-control study taking all ${CT.grp(cases)} cases and ${k === 1 ? "one control" : ST.fmt(k, 2) + " controls"} per case recovers ` +
      `<b>OR = ${ST.fmt(ccOR, 4)}</b> — the population value, exactly, because the sampling fractions cancel out of <code>ad/bc</code>. ` +
      `The same table's naive "relative risk" is <b>${ST.fmt(ccRR, 4)}</b> against a truth of ${ST.fmt(trueRR, 4)}, ` +
      `and its "risk difference" is <b>${ST.fmt(ccRD, 4)}</b> against ${ST.fmt(trueRD, 4)}. ` +
      `Move the control-ratio slider: those two numbers slide with a design choice, so they are not estimates of anything. ` +
      `The odds ratio does not move at all. ` +
      (trueOR / trueRR < 1.02
        ? `The outcome is rare enough here that <code>OR ≈ RR</code> (${ST.fmt(trueOR, 4)} against ${ST.fmt(trueRR, 4)}, a gap of ${ST.pct(trueOR / trueRR - 1, 1)}) — the rare-disease approximation that licenses reading a case-control odds ratio as a risk ratio.`
        : `Note that even the <i>correct</i> odds ratio of ${ST.fmt(trueOR, 4)} overstates the true risk ratio of ${ST.fmt(trueRR, 4)} by <b>${ST.pct(trueOR / trueRR - 1, 1)}</b> — the outcome is not rare here, so the rare-disease approximation does not apply. Press "make the outcome rare" and the gap closes.`);
  }

  [ePe, eR1, eR0, eK].forEach(el => el.addEventListener("input", draw));
  document.getElementById("cc-rare").addEventListener("click", () => {
    eR1.value = 0.02; eR0.value = 0.005; draw();
  });
  draw();
})();

/* ─────────────── 13 · Simpson's paradox ─────────────── */
(function () {
  const svg = d3.select("#simp-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const eW = document.getElementById("sp-w"), eWv = document.getElementById("sp-wv");
  const eA = document.getElementById("sp-a"), eAv = document.getElementById("sp-av");
  const eB = document.getElementById("sp-b"), eBv = document.getElementById("sp-bv");
  const eC = document.getElementById("sp-c"), eCv = document.getElementById("sp-cv");
  const eD = document.getElementById("sp-d"), eDv = document.getElementById("sp-dv");
  const out = document.getElementById("simp-readout");
  const ARM = 400;                       // 400 in each arm, always

  /* rates are held fixed; only WHO is in which arm moves. w = share of the treated
     arm that is severe; the control arm gets the complementary share, which is what
     makes the two arms' weight vectors differ.                                    */
  function build(w, a, b, c, d) {
    const tSev = Math.round(ARM * w), tMild = ARM - tSev;
    const cSev = ARM - tSev, cMild = ARM - cSev;
    return {
      sev: { tN: tSev, tR: tSev * a, cN: cSev, cR: cSev * b, tp: a, cp: b },
      mil: { tN: tMild, tR: tMild * c, cN: cMild, cR: cMild * d, tp: c, cp: d },
      tp: (tSev * a + tMild * c) / ARM, cp: (cSev * b + cMild * d) / ARM,
      tSev: tSev, cSev: cSev
    };
  }
  /* the allocation share at which the pooled difference changes sign, solved exactly.
     pooled(t) − pooled(c) = w·a + (1−w)·c − [(1−w)·b + w·d] = w(a − c − d + b) + (c − b).
     Setting that to zero gives w* = (b − c)/(a + b − c − d).                          */
  function flipPoint(a, b, c, d) {
    const den = a + b - c - d;
    if (Math.abs(den) < 1e-12) return null;
    const w = (b - c) / den;
    return (w > 0 && w < 1) ? w : null;
  }

  function draw() {
    const w = +eW.value / 100, a = +eA.value, b = +eB.value, c = +eC.value, d = +eD.value;
    eWv.textContent = Math.round(w * 100) + "%";
    eAv.textContent = ST.fmt(a, 2); eBv.textContent = ST.fmt(b, 2);
    eCv.textContent = ST.fmt(c, 2); eDv.textContent = ST.fmt(d, 2);
    const S = build(w, a, b, c, d);
    const dSev = a - b, dMil = c - d, dPool = S.tp - S.cp;
    const consistent = (dSev > 0 && dMil > 0) || (dSev < 0 && dMil < 0);
    const reversed = consistent && (Math.sign(dPool) !== Math.sign(dSev)) && Math.abs(dPool) > 1e-12;
    const flip = flipPoint(a, b, c, d);

    const f = ST.frame(svg, W, H, { l: 14, r: 14, t: 14, b: 14 });
    const g = f.g;

    function bars(gx, title, tp, cp, tN, cN, wide) {
      const gp = CT.sub(g, gx, 44, title);
      const BW = wide ? 190 : 150, BH = 168;
      [["treatment", tp, tN, SC.accent], ["control", cp, cN, SC.a2]].forEach((r, i) => {
        const bx = i * (BW / 2 + 14);
        gp.append("rect").attr("x", bx).attr("y", 0).attr("width", BW / 2 - 10).attr("height", BH)
          .attr("rx", 4).attr("fill", SC.panel2).attr("stroke", SC.line);
        gp.append("rect").attr("x", bx).attr("y", BH * (1 - r[1])).attr("width", BW / 2 - 10)
          .attr("height", BH * r[1]).attr("rx", 4).attr("fill", r[3]).attr("fill-opacity", 0.62);
        gp.append("text").attr("x", bx + (BW / 2 - 10) / 2).attr("y", BH * (1 - r[1]) - 6)
          .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 600)
          .attr("fill", SC.ink).text(ST.pct(r[1], 1));
        gp.append("text").attr("x", bx + (BW / 2 - 10) / 2).attr("y", BH + 14)
          .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.muted).text(r[0]);
        gp.append("text").attr("x", bx + (BW / 2 - 10) / 2).attr("y", BH + 26)
          .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", SC.muted)
          .text("n = " + CT.grp(r[2]));
      });
      const diff = tp - cp;
      gp.append("text").attr("x", 0).attr("y", BH + 46).attr("font-size", 11.5)
        .attr("font-weight", 600).attr("fill", diff > 0 ? SC.good : SC.bad)
        .text((diff > 0 ? "+" : "") + ST.fmt(100 * diff, 2) + " points for treatment");
      return BH;
    }
    bars(56, "severe cases", a, b, S.sev.tN, S.sev.cN);
    bars(246, "mild cases", c, d, S.mil.tN, S.mil.cN);
    bars(452, "POOLED — everyone together", S.tp, S.cp, ARM, ARM, true);

    /* the verdict banner */
    const bannerY = 44 + 168 + 62;
    g.append("rect").attr("x", 40).attr("y", bannerY - 18).attr("width", 660).attr("height", 34)
      .attr("rx", 7).attr("fill", reversed ? SC.bad : SC.good).attr("fill-opacity", 0.16)
      .attr("stroke", reversed ? SC.bad : SC.good);
    g.append("text").attr("x", 370).attr("y", bannerY + 4).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", reversed ? SC.bad : SC.good)
      .text(reversed
        ? "REVERSED — the treatment wins in both strata and loses overall"
        : consistent
          ? "consistent — the pooled comparison agrees with both strata"
          : "the two strata disagree with each other, so there is nothing to reverse");

    /* the track: pooled difference against allocation share */
    const tw = 620, th = 86;
    const gt = CT.sub(g, 56, bannerY + 46, "the pooled difference, as the allocation share moves");
    const xs = d3.scaleLinear().domain([0, 1]).range([0, tw]);
    const pts = ST.linspace(0, 1, 201).map(u => {
      const S2 = build(u, a, b, c, d);
      return { u: u, v: S2.tp - S2.cp };
    });
    const ext = Math.max(0.02, d3.max(pts.map(p => Math.abs(p.v))) * 1.2);
    const ys = d3.scaleLinear().domain([-ext, ext]).range([th, 0]);
    gt.append("rect").attr("x", 0).attr("y", 0).attr("width", tw).attr("height", ys(0))
      .attr("fill", SC.good).attr("fill-opacity", 0.07);
    gt.append("rect").attr("x", 0).attr("y", ys(0)).attr("width", tw).attr("height", th - ys(0))
      .attr("fill", SC.bad).attr("fill-opacity", 0.07);
    gt.append("line").attr("x1", 0).attr("x2", tw).attr("y1", ys(0)).attr("y2", ys(0))
      .attr("stroke", SC.muted);
    gt.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(p => xs(p.u)).y(p => ys(ST.clamp(p.v, -ext, ext)))(pts));
    gt.append("g").attr("class", "axis").attr("transform", `translate(0,${th})`)
      .call(d3.axisBottom(xs).ticks(6, ".0%"));
    gt.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(3, "+.2f"));
    if (flip !== null) {
      gt.append("line").attr("x1", xs(flip)).attr("x2", xs(flip)).attr("y1", 0).attr("y2", th)
        .attr("stroke", SC.bad).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
      gt.append("text").attr("x", xs(flip)).attr("y", -4).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.bad).text("flips at " + ST.pct(flip, 1));
    }
    gt.append("circle").attr("cx", xs(w)).attr("cy", ys(ST.clamp(dPool, -ext, ext)))
      .attr("r", 5).attr("fill", SC.ink);
    gt.append("line").attr("x1", xs(0.5)).attr("x2", xs(0.5)).attr("y1", 0).attr("y2", th)
      .attr("stroke", SC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "2 3");
    gt.append("text").attr("x", xs(0.5)).attr("y", th + 30).attr("text-anchor", "middle")
      .attr("font-size", 9.5).attr("fill", SC.good).text("balanced — no reversal is possible here");

    /* the odds ratios, which reverse too */
    const orS = (a / (1 - a)) / (b / (1 - b)), orM = (c / (1 - c)) / (d / (1 - d));
    const orP = (S.tp / (1 - S.tp)) / (S.cp / (1 - S.cp));
    /* the standardised comparison: reweight the control arm to the treated arm's mix */
    const stdC = w * b + (1 - w) * d;

    out.innerHTML =
      `Severe: treatment <b>${ST.pct(a, 1)}</b> against <b>${ST.pct(b, 1)}</b> — ${dSev >= 0 ? "+" : ""}${ST.fmt(100 * dSev, 2)} points. ` +
      `Mild: <b>${ST.pct(c, 1)}</b> against <b>${ST.pct(d, 1)}</b> — ${dMil >= 0 ? "+" : ""}${ST.fmt(100 * dMil, 2)} points. ` +
      `Pooled: <b>${ST.pct(S.tp, 2)}</b> against <b>${ST.pct(S.cp, 2)}</b> — <b>${dPool >= 0 ? "+" : ""}${ST.fmt(100 * dPool, 2)} points</b>. ` +
      `Odds ratios ${ST.fmt(orS, 3)} and ${ST.fmt(orM, 3)} within strata, <b>${ST.fmt(orP, 3)}</b> pooled. ` +
      `The treated arm is ${ST.pct(w, 0)} severe and the control arm ${ST.pct(1 - w, 0)} severe: <b>different weights on the same rates</b>. ` +
      `Standardise the control arm to the treated arm's mix and its rate becomes ${ST.pct(stdC, 2)}, ` +
      `so the standardised difference is ${(S.tp - stdC) >= 0 ? "+" : ""}${ST.fmt(100 * (S.tp - stdC), 2)} points — ` +
      `back between the two stratum-specific gains, where a weighted average must lie. ` +
      (flip !== null
        ? `The pooled conclusion flips at an allocation share of <b>${ST.pct(flip, 1)}</b>.`
        : `At these rates no allocation reverses the conclusion.`) +
      ` At 50% the arms have identical mixes and reversal is impossible — which is exactly what randomisation buys.`;
  }

  [eW, eA, eB, eC, eD].forEach(el => el.addEventListener("input", draw));
  document.getElementById("sp-bal").addEventListener("click", () => { eW.value = 50; draw(); });
  document.getElementById("sp-reset").addEventListener("click", () => {
    eW.value = 75; eA.value = 0.70; eB.value = 0.60; eC.value = 0.90; eD.value = 0.85; draw();
  });
  draw();
})();

/* ─────────────── 14 · significance bought with sample size ─────────────── */
(function () {
  const svg = d3.select("#big-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 410;
  const eP = document.getElementById("bg-phi"), ePv = document.getElementById("bg-phiv");
  const eN = document.getElementById("bg-n"), eNv = document.getElementById("bg-nv");
  const eV = document.getElementById("bg-view");
  const out = document.getElementById("big-readout");

  /* A balanced 2 × 2 with a given φ: put p₁₁ = p₂₂ = (1 + φ)/4 and the off-diagonals
     at (1 − φ)/4. Then both margins are ½, and χ² = n·φ² exactly. */
  const joint = phi => [[(1 + phi) / 4, (1 - phi) / 4], [(1 - phi) / 4, (1 + phi) / 4]];

  function draw() {
    const phi = +eP.value, nSel = Math.round(Math.pow(10, +eN.value));
    ePv.textContent = ST.fmt(phi, 3); eNv.textContent = CT.grp(nSel);
    const P = joint(phi);

    const f = ST.frame(svg, W, H, { l: 60, r: 200, t: 22, b: 48 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLog().domain([10, 2e6]).range([0, iw]);
    const y = d3.scaleLog().domain([1e-20, 1]).range([ih, 0]).clamp(true);
    ST.gridY(g, y, iw, 6);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(7, ".0s"));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(7, ".0e"));
    g.append("text").attr("x", iw).attr("y", ih + 34).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("sample size");
    g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 10.5).attr("fill", SC.muted)
      .text("p-value (log scale)");

    const ns = ST.linspace(1, 6.301, 200).map(u => Math.pow(10, u));
    const pv = ns.map(n => ({ n: n, p: CT.chi2Sf(n * phi * phi, 1) }));
    g.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.4)
      .attr("d", d3.line().x(d => x(d.n)).y(d => y(Math.max(1e-20, d.p)))(pv));
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(0.05)).attr("y2", y(0.05))
      .attr("stroke", SC.a2).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 4");
    g.append("text").attr("x", iw + 4).attr("y", y(0.05) + 4).attr("font-size", 10)
      .attr("fill", SC.a2).text("p = 0.05");

    if (eV.value === "both") {      // the effect size, flat, on the same axes for contrast
      g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(Math.max(1e-20, phi)))
        .attr("y2", y(Math.max(1e-20, phi))).attr("stroke", SC.good).attr("stroke-width", 2.2);
      g.append("text").attr("x", iw + 4).attr("y", y(Math.max(1e-20, phi)) + 4)
        .attr("font-size", 10).attr("fill", SC.good).text("φ = " + ST.fmt(phi, 3) + ", constant");
    }

    /* where 0.05 is crossed: χ² = 3.841459 ⟹ n = 3.841459/φ² */
    const nStar = 3.841459 / (phi * phi);
    if (nStar >= 10 && nStar <= 2e6) {
      g.append("line").attr("x1", x(nStar)).attr("x2", x(nStar)).attr("y1", 0).attr("y2", ih)
        .attr("stroke", SC.bad).attr("stroke-width", 1.3).attr("stroke-dasharray", "3 3");
      g.append("text").attr("x", x(nStar)).attr("y", -6).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", SC.bad).text("n = " + CT.grp(nStar));
    }
    const chiSel = nSel * phi * phi, pSel = CT.chi2Sf(chiSel, 1);
    g.append("circle").attr("cx", x(ST.clamp(nSel, 10, 2e6)))
      .attr("cy", y(Math.max(1e-20, pSel))).attr("r", 5).attr("fill", SC.ink);

    /* the table, which never changes */
    const gt = CT.sub(g, iw + 12, 44, "the table, in percentages");
    const cw = 62, chh = 30;
    [["", "yes", "no"], ["A", P[0][0], P[0][1]], ["B", P[1][0], P[1][1]]].forEach((r, i) =>
      r.forEach((v, j) => {
        if (i > 0 && j > 0) gt.append("rect").attr("x", 26 + (j - 1) * cw).attr("y", (i - 1) * chh)
          .attr("width", cw - 5).attr("height", chh - 5).attr("rx", 4)
          .attr("fill", SC.panel2).attr("stroke", SC.line);
        gt.append("text").attr("x", j === 0 ? 20 : 26 + (j - 1) * cw + (cw - 5) / 2)
          .attr("y", i === 0 ? -4 : (i - 1) * chh + chh / 2)
          .attr("text-anchor", j === 0 ? "end" : "middle").attr("font-size", i === 0 || j === 0 ? 10 : 11.5)
          .attr("fill", i === 0 || j === 0 ? SC.accent : SC.ink)
          .text(i === 0 || j === 0 ? v : ST.pct(v, 2));
      }));
    gt.append("text").attr("x", 0).attr("y", 2 * chh + 14).attr("font-size", 9.5)
      .attr("fill", SC.muted).text("rates " + ST.pct((1 + phi) / 2, 2) + " vs " + ST.pct((1 - phi) / 2, 2));
    CT.chip(g, iw + 12, 44 + 2 * chh + 30, "CHI-SQUARE", ST.fmt(chiSel, 3), SC.ink, 172);
    CT.chip(g, iw + 12, 44 + 2 * chh + 74, "P-VALUE", CT.pstr(pSel),
      pSel < 0.05 ? SC.bad : SC.good, 172);
    CT.chip(g, iw + 12, 44 + 2 * chh + 118, "EFFECT SIZE φ", ST.fmt(phi, 4), SC.good, 172);

    const or = ((1 + phi) / (1 - phi)) * ((1 + phi) / (1 - phi));
    out.innerHTML =
      `Association held at <code>φ = ${ST.fmt(phi, 4)}</code> — a ${ST.pct((1 + phi) / 2, 2)} against ${ST.pct((1 - phi) / 2, 2)} split, ` +
      `an odds ratio of ${ST.fmt(or, 4)}. That never changes on this axis. ` +
      `At <code>n = ${CT.grp(nSel)}</code>: <code>χ² = n·φ² = ${ST.fmt(chiSel, 4)}</code>, <code>p = ${CT.pstr(pSel)}</code>. ` +
      `The 5% line is crossed at <b>n = ${CT.grp(nStar)}</b> — and since <code>χ² = nφ²</code> exactly, that threshold is simply ` +
      `<code>3.8415/φ² = ${CT.grp(nStar)}</code>. ` +
      `Ten times more data multiplies the statistic by ten and drives the p-value down by orders of magnitude ` +
      `<i>without changing the association by a hair</i>. A p-value reported without an effect size beside it says nothing about the world; ` +
      `it says how much data you bought.`;
  }

  [eP, eN].forEach(el => el.addEventListener("input", draw));
  eV.addEventListener("change", draw);
  draw();
})();
