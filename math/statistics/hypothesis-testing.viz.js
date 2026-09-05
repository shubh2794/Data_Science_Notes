/* hypothesis-testing.viz.js — the seven visualizations on
   math/statistics/hypothesis-testing.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #reg-svg    the null distribution with a DRAGGABLE critical value and a
                    draggable observed statistic; alpha and the p-value shaded
                    separately, and compared as two bars on one scale
     2  #punif-svg  a thousand studies at a time: the p-value histogram is FLAT
                    under a true null and piles at zero as the effect grows
     3  #base-svg   1000 hypotheses as a waffle, sorted into the four outcomes,
                    with the discoveries column split into true and false
     4  #pow-svg    the two overlapping sampling distributions with a draggable
                    threshold: alpha, beta and power as three live areas, plus
                    the whole power function underneath
     5  #plan-svg   power against n on a log axis, one curve per effect size,
                    with the target-power crossing solved exactly
     6  #dual-svg   the interval–test duality on ONE axis, with the p-value
                    function above it crossing alpha exactly at the endpoints
     7  #hack-svg   best-of-k and optional stopping simulated under a TRUE null:
                    the nominal level against the rate the procedure really has

   Every number these draw is computed from the same ST helpers the prose was
   checked against, so the pictures and the text cannot drift apart.            */

/* ══════════ page-local numeric helpers (deliberately NOT in stats-viz.js) ══════════ */
const HT = (function () {
  const Z975 = ST.normQuant(0.975);

  /* Two-sided power of the z test at noncentrality D = delta*sqrt(n)/sigma. */
  function zPower(D, alpha, oneSided) {
    if (oneSided) return 1 - ST.normCdf(ST.normQuant(1 - alpha) - D);
    const c = ST.normQuant(1 - alpha / 2);
    return ST.normCdf(D - c) + ST.normCdf(-D - c);
  }

  /* EXACT power of the two-sided one-sample t test, by integrating the chi-square
     density of s/sigma against the conditional normal probability:
        T = (Z + λ)/√(V/ν),   λ = δ√n/σ,   V ~ χ²ν,   ν = n − 1
        power = E_s[ Φ(−t*·s − λ) + 1 − Φ(t*·s − λ) ],   s = √(V/ν)
     Simpson on 900 panels; agrees with a 80 000-run Monte-Carlo to ~1.5e-3
     (which is the Monte-Carlo's own standard error) and returns exactly alpha
     at δ = 0. `pairs` = 1 for one-sample/paired, 2 for two independent groups
     with n per group (then ν = 2n − 2 and λ = δ√(n/2)).                        */
  function tPower(delta, n, alpha, pairs) {
    const two = (pairs === 2);
    const nu = two ? 2 * n - 2 : n - 1;
    if (nu < 1) return NaN;
    const lam = two ? delta * Math.sqrt(n / 2) : delta * Math.sqrt(n);
    const tc = ST.tQuant(1 - alpha / 2, nu);
    const smax = 1 + 10 / Math.sqrt(nu) + (nu < 4 ? 6 : 0);
    const N = 900, h = smax / N;
    const g = s => {
      if (s <= 0) return 0;
      const dens = 2 * s * nu * ST.chi2Pdf(nu * s * s, nu);
      return dens * (ST.normCdf(-tc * s - lam) + 1 - ST.normCdf(tc * s - lam));
    };
    let acc = 0;
    for (let i = 0; i < N; i++) {
      const a = i * h, b = a + h;
      acc += (h / 6) * (g(a) + 4 * g((a + b) / 2) + g(b));
    }
    return ST.clamp(acc, 0, 1);
  }

  /* smallest n reaching the target power — bisection on a monotone function */
  function nForPower(delta, target, alpha, pairs, exact) {
    const f = n => exact ? tPower(delta, n, alpha, pairs)
                         : zPower(delta * Math.sqrt(pairs === 2 ? n / 2 : n), alpha, false);
    let lo = pairs === 2 ? 2 : 3, hi = 4;
    if (f(lo) >= target) return lo;
    let guard = 0;
    while (f(hi) < target && guard++ < 40) hi *= 2;
    if (f(hi) < target) return Infinity;
    while (hi - lo > 1) { const m = Math.floor((lo + hi) / 2); if (f(m) < target) lo = m; else hi = m; }
    return hi;
  }

  /* density / quantile of the chosen reference: df = 0 means the standard normal */
  const refPdf = (x, df) => df ? ST.tPdf(x, df) : ST.normPdf(x);
  const refCdf = (x, df) => df ? ST.tCdf(x, df) : ST.normCdf(x);
  const refQnt = (p, df) => df ? ST.tQuant(p, df) : ST.normQuant(p);

  /* one simulated one-sample t test: returns the two-sided p-value */
  function oneT(n, delta, r) {
    let s = 0, ss = 0;
    for (let i = 0; i < n; i++) { const x = delta + ST.randn(r); s += x; ss += x * x; }
    const m = s / n, v = (ss - n * m * m) / (n - 1);
    if (!(v > 0)) return 1;
    const t = m / Math.sqrt(v / n);
    return 2 * (1 - ST.tCdf(Math.abs(t), n - 1));
  }

  return { Z975, zPower, tPower, nForPower, refPdf, refCdf, refQnt, oneT };
})();

/* ─────────────── 1 · the null distribution and the rejection region ─────────────── */
(function () {
  const svg = d3.select("#reg-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 430;
  const elDist = document.getElementById("rg-dist"), elSide = document.getElementById("rg-side");
  const elA = document.getElementById("rg-alpha"), elAv = document.getElementById("rg-alphav");
  const elT = document.getElementById("rg-t"), elTv = document.getElementById("rg-tv");
  const out = document.getElementById("reg-readout");

  const state = { df: 0, side: 2, alpha: 0.05, t: 2.31 };

  function draw() {
    const df = state.df, side = state.side, al = state.alpha, tob = state.t;
    const f = ST.frame(svg, W, H, { l: 52, r: 18, t: 16, b: 108 });
    const g = f.g, iw = f.iw, ih = f.ih - 46;     // reserve the bottom strip for the bars

    const XMAX = 4.6;
    const x = d3.scaleLinear().domain([-XMAX, XMAX]).range([0, iw]);
    const xs = ST.linspace(-XMAX, XMAX, 481);
    const ys = xs.map(v => HT.refPdf(v, df));
    const y = d3.scaleLinear().domain([0, d3.max(ys) * 1.14]).range([ih, 0]);

    ST.gridY(g, y, iw, 4);
    ST.axisB(g, x, ih, 9, "test statistic  t");
    ST.axisL(g, y, 4, "density under H₀");

    const area = d3.area().x(d => x(d)).y0(ih).y1(d => y(HT.refPdf(d, df))).curve(d3.curveLinear);
    const crit = side === 2 ? HT.refQnt(1 - al / 2, df) : HT.refQnt(1 - al, df);

    /* ── the rejection region ── */
    const upper = ST.linspace(crit, XMAX, 140);
    g.append("path").attr("d", area(upper)).attr("fill", SC.bad).attr("fill-opacity", 0.30);
    if (side === 2) {
      const lower = ST.linspace(-XMAX, -crit, 140);
      g.append("path").attr("d", area(lower)).attr("fill", SC.bad).attr("fill-opacity", 0.30);
    }

    /* ── the p-value area, drawn on top and outlined so it survives the overlap ── */
    const at = Math.abs(tob);
    const pUp = ST.linspace(at, XMAX, 140);
    g.append("path").attr("d", area(pUp)).attr("fill", SC.a2).attr("fill-opacity", 0.50)
      .attr("stroke", SC.a2).attr("stroke-width", 1.1).attr("stroke-opacity", 0.9);
    if (side === 2) {
      const pLo = ST.linspace(-XMAX, -at, 140);
      g.append("path").attr("d", area(pLo)).attr("fill", SC.a2).attr("fill-opacity", 0.50)
        .attr("stroke", SC.a2).attr("stroke-width", 1.1).attr("stroke-opacity", 0.9);
    }

    /* ── the curve ── */
    const line = d3.line().x(d => x(d)).y(d => y(HT.refPdf(d, df)));
    g.append("path").attr("d", line(xs)).attr("fill", "none")
      .attr("stroke", SC.accent).attr("stroke-width", 2);

    /* ── the critical lines ── */
    const critG = g.append("g");
    const critLine = c => {
      const gg = critG.append("g").attr("class", "dragpt").style("cursor", "ew-resize");
      gg.append("line").attr("x1", x(c)).attr("x2", x(c)).attr("y1", 0).attr("y2", ih)
        .attr("stroke", SC.bad).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
      gg.append("rect").attr("x", x(c) - 9).attr("y", 0).attr("width", 18).attr("height", ih)
        .attr("fill", "transparent");
      gg.append("circle").attr("cx", x(c)).attr("cy", 8).attr("r", 5.5)
        .attr("fill", SC.bad).attr("stroke", SC.bg).attr("stroke-width", 1.5);
      return gg;
    };
    const gU = critLine(crit);
    if (side === 2) critLine(-crit);
    const lblSide = x(crit) > iw - 130 ? -6 : 6;
    g.append("text").attr("x", x(crit) + lblSide).attr("y", 24)
      .attr("text-anchor", lblSide < 0 ? "end" : "start")
      .attr("font-size", 10.5).attr("fill", SC.bad)
      .text("critical value " + ST.fmt(crit, 3));

    /* dragging the critical line re-solves for alpha */
    const dragC = d3.drag().on("drag", ev => {
      const c = Math.abs(x.invert(Math.max(0, Math.min(iw, ev.x))));
      const cc = Math.max(0.05, Math.min(XMAX, c));
      const tail = 1 - HT.refCdf(cc, df);
      state.alpha = ST.clamp(side === 2 ? 2 * tail : tail, 0.001, 0.30);
      elA.value = state.alpha.toFixed(3); elAv.textContent = state.alpha.toFixed(3);
      draw();
    });
    gU.call(dragC);

    /* ── the observed statistic ── */
    const obsG = g.append("g").attr("class", "dragpt").style("cursor", "ew-resize");
    obsG.append("line").attr("x1", x(tob)).attr("x2", x(tob)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.a2).attr("stroke-width", 2.5);
    obsG.append("rect").attr("x", x(tob) - 10).attr("y", 0).attr("width", 20).attr("height", ih)
      .attr("fill", "transparent");
    obsG.append("path").attr("d", "M0,-7 L7,4 L-7,4 Z")
      .attr("transform", `translate(${x(tob)},${ih})`)
      .attr("fill", SC.a2).attr("stroke", SC.bg).attr("stroke-width", 1);
    const oSide = x(tob) > iw - 110 ? -7 : 7;
    obsG.append("text").attr("x", x(tob) + oSide).attr("y", 48)
      .attr("text-anchor", oSide < 0 ? "end" : "start")
      .attr("font-size", 10.5).attr("fill", SC.a2)
      .text("observed  t = " + ST.fmt(tob, 2));
    obsG.call(d3.drag().on("drag", ev => {
      state.t = ST.clamp(x.invert(Math.max(0, Math.min(iw, ev.x))), -4.5, 4.5);
      elT.value = state.t.toFixed(2); elTv.textContent = state.t.toFixed(2);
      draw();
    }));

    ST.legend(g, [
      { label: "rejection region — area " + ST.fmt(al, 3), color: SC.bad, op: 0.35 },
      { label: "p-value — the observed tail", color: SC.a2, op: 0.45 }
    ], 8, 14);

    /* ── the two-bar comparison strip ── */
    const tail = 1 - HT.refCdf(at, df);
    const pv = side === 2 ? Math.min(1, 2 * tail) : (tob >= 0 ? tail : 1 - tail);
    const bx = d3.scaleLinear().domain([0, Math.max(0.32, al * 1.3, pv * 1.15)]).range([0, iw]);
    const strip = g.append("g").attr("transform", `translate(0,${ih + 56})`);
    strip.append("text").attr("x", -6).attr("y", 4).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("α");
    strip.append("rect").attr("x", 0).attr("y", -7).attr("width", bx(al)).attr("height", 13)
      .attr("rx", 2).attr("fill", SC.bad).attr("fill-opacity", 0.75);
    strip.append("text").attr("x", -6).attr("y", 26).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("p");
    strip.append("rect").attr("x", 0).attr("y", 15).attr("width", bx(Math.min(pv, bx.domain()[1])))
      .attr("height", 13).attr("rx", 2)
      .attr("fill", pv <= al ? SC.good : SC.muted).attr("fill-opacity", 0.8);
    strip.append("line").attr("x1", bx(al)).attr("x2", bx(al)).attr("y1", -11).attr("y2", 32)
      .attr("stroke", SC.bad).attr("stroke-width", 1.5).attr("stroke-dasharray", "3 3");
    strip.append("text").attr("x", iw).attr("y", 44).attr("text-anchor", "end")
      .attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", pv <= al ? SC.good : SC.muted)
      .text(pv <= al ? "p ≤ α → REJECT H₀" : "p > α → do not reject");

    const dname = df ? ("t with " + df + " df") : "Normal(0, 1)";
    out.innerHTML = `<b>${dname}</b> · ${side === 2 ? "two-sided" : "one-sided (upper)"} · `
      + `α = <b>${ST.fmt(al, 3)}</b> · critical value <b>${side === 2 ? "±" : ""}${ST.fmt(crit, 4)}</b>`
      + ` · observed t = <b>${ST.fmt(tob, 2)}</b> · p = <b>${pv < 1e-4 ? pv.toExponential(2) : ST.fmt(pv, 4)}</b>`
      + ` — ${pv <= al ? "reject H₀ at this level" : "do not reject at this level"}.`
      + `<br>The two bars are the same two numbers on one scale: the test rejects exactly when the lower bar (p) is shorter than the upper one (α).`;
  }

  elDist.onchange = () => { state.df = +elDist.value; draw(); };
  elSide.onchange = () => { state.side = +elSide.value; draw(); };
  elA.oninput = () => { state.alpha = +elA.value; elAv.textContent = state.alpha.toFixed(3); draw(); };
  elT.oninput = () => { state.t = +elT.value; elTv.textContent = state.t.toFixed(2); draw(); };
  document.getElementById("rg-reset").onclick = () => {
    state.df = 0; state.side = 2; state.alpha = 0.05; state.t = 2.31;
    elDist.value = "0"; elSide.value = "2"; elA.value = "0.05"; elT.value = "2.31";
    elAv.textContent = "0.050"; elTv.textContent = "2.31";
    draw();
  };
  draw();
})();

/* ─────────────── 2 · the p-value distribution ─────────────── */
(function () {
  const svg = d3.select("#punif-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 420;
  const eD = document.getElementById("pu-d"), eDv = document.getElementById("pu-dv");
  const eN = document.getElementById("pu-n"), eNv = document.getElementById("pu-nv");
  const eB = document.getElementById("pu-B"), eBv = document.getElementById("pu-Bv");
  const eCum = document.getElementById("pu-cum");
  const out = document.getElementById("punif-readout");

  let seed = 41;

  function run() {
    const delta = +eD.value, n = +eN.value, B = +eB.value, cum = eCum.checked;
    const r = ST.rng(seed);
    const K = 20, counts = new Array(K).fill(0);
    let below05 = 0, below01 = 0;
    for (let b = 0; b < B; b++) {
      const p = HT.oneT(n, delta, r);
      counts[Math.min(K - 1, Math.floor(p * K))]++;
      if (p < 0.05) below05++;
      if (p < 0.01) below01++;
    }

    const f = ST.frame(svg, W, H, { l: 56, r: 18, t: 18, b: 44 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
    const props = counts.map(c => c / B);
    const cums = []; let acc = 0;
    props.forEach(v => { acc += v; cums.push(acc); });
    const shown = cum ? cums : props;
    const ymax = cum ? 1.05 : Math.max(0.10, d3.max(props) * 1.18);
    const y = d3.scaleLinear().domain([0, ymax]).range([ih, 0]);

    ST.gridY(g, y, iw, 5);
    ST.axisB(g, x, ih, 11, "p-value");
    ST.axisL(g, y, 5, cum ? "cumulative share of studies" : "share of studies", d3.format(".0%"));

    /* the 0.05 zone */
    g.append("rect").attr("x", 0).attr("y", 0).attr("width", x(0.05)).attr("height", ih)
      .attr("fill", SC.bad).attr("fill-opacity", 0.10);
    g.append("line").attr("x1", x(0.05)).attr("x2", x(0.05)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.bad).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 4");
    g.append("text").attr("x", x(0.05) + 5).attr("y", 12).attr("font-size", 10.5)
      .attr("fill", SC.bad).text("p < 0.05");

    const bw = iw / K;
    g.selectAll("rect.bar").data(shown).join("rect").attr("class", "bar")
      .attr("x", (d, i) => i * bw + 1).attr("width", bw - 2)
      .attr("y", d => y(d)).attr("height", d => ih - y(d))
      .attr("rx", 2)
      .attr("fill", (d, i) => (i === 0 ? SC.a2 : SC.accent))
      .attr("fill-opacity", 0.82);

    /* the uniform reference */
    if (!cum) {
      g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(1 / K)).attr("y2", y(1 / K))
        .attr("stroke", SC.good).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
      g.append("text").attr("x", iw - 4).attr("y", y(1 / K) - 6).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.good)
        .text("uniform: every bar 1/20 = 5%");
    } else {
      const dl = d3.line().x(d => x(d)).y(d => y(d));
      g.append("path").attr("d", dl([0, 1])).attr("fill", "none")
        .attr("stroke", SC.good).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
      g.append("text").attr("x", iw - 4).attr("y", y(0.88)).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.good).text("uniform cdf: the 45° line");
    }

    const emp = below05 / B;
    const theory = delta === 0 ? 0.05 : HT.tPower(delta, n, 0.05, 1);
    const se = Math.sqrt(theory * (1 - theory) / B);
    out.innerHTML = `<b>${B.toLocaleString()}</b> simulated one-sample t tests, n = <b>${n}</b>, true effect δ = <b>${ST.fmt(delta, 2)}σ</b>.`
      + `<br>Fraction with p &lt; 0.05: <b>${ST.pct(emp, 2)}</b>`
      + ` — exact ${delta === 0 ? "type I rate" : "power"} for this design is <b>${ST.pct(theory, 2)}</b>`
      + ` (simulation SE ≈ ${ST.pct(se, 2)}). Fraction with p &lt; 0.01: <b>${ST.pct(below01 / B, 2)}</b>.`
      + (delta === 0
        ? `<br>With no effect the histogram is <b>flat</b>: p = 0.03 is exactly as likely as p = 0.83. One study in twenty is "significant", by construction, and nothing has gone wrong.`
        : `<br>With a real effect the mass slides left. The amber bar is the p &lt; 0.05 bin — the power of the test is how much of the distribution it and its neighbours have captured.`);
  }

  eD.oninput = () => { eDv.textContent = (+eD.value).toFixed(2); run(); };
  eN.oninput = () => { eNv.textContent = eN.value; run(); };
  eB.oninput = () => { eBv.textContent = (+eB.value).toLocaleString(); run(); };
  eCum.onchange = run;
  document.getElementById("pu-run").onclick = () => { seed = (seed * 1103515245 + 12345) >>> 0; run(); };
  run();
})();

/* ─────────────── 3 · the base rate, as a thousand squares ─────────────── */
(function () {
  const svg = d3.select("#base-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 460, M = 1000;
  const ePi = document.getElementById("br-pi"), ePiv = document.getElementById("br-piv");
  const ePw = document.getElementById("br-pow"), ePwv = document.getElementById("br-powv");
  const eAl = document.getElementById("br-al"), eAlv = document.getElementById("br-alv");
  const ePub = document.getElementById("br-pub");
  const out = document.getElementById("base-readout");

  const CL = { tp: SC.good, fp: SC.bad, fn: SC.a2, tn: "#39445a" };

  function draw() {
    const pi = +ePi.value, pow = +ePw.value, al = +eAl.value, pub = ePub.checked;

    /* integer counts that always sum to exactly 1000 */
    const nAlt = Math.round(M * pi), nNul = M - nAlt;
    const TP = Math.round(nAlt * pow), FN = nAlt - TP;
    const FP = Math.round(nNul * al), TN = nNul - FP;
    const disc = TP + FP;
    const ppv = disc ? TP / disc : NaN;

    const f = ST.frame(svg, W, H, { l: 14, r: 14, t: 46, b: 74 });
    const g = f.g, iw = f.iw, ih = f.ih;

    /* two columns: discoveries on the left, the rest on the right */
    const gap = 34;
    const leftW = Math.max(90, Math.round((iw - gap) * 0.42));
    const rightW = iw - gap - leftW;

    /* ONE square = ONE hypothesis, at the SAME size in both columns, so the areas
       are directly comparable. Pick the largest cell that fits both grids.        */
    const colsL = 20, colsR = 34;
    const rowsL = Math.max(1, Math.ceil(disc / colsL));
    const rowsR = Math.max(1, Math.ceil((FN + TN) / colsR));
    const cell = Math.min(leftW / colsL, rightW / colsR, ih / rowsL, pub ? Infinity : ih / rowsR);
    const sq = Math.max(1.2, cell - 0.8);

    function waffle(gx, gy, groups, cols) {
      const gg = g.append("g").attr("transform", `translate(${gx},${gy})`);
      let k = 0;
      groups.forEach(q => {
        const data = d3.range(q.n).map(() => k++);
        gg.selectAll("rect.c" + q.key).data(data).join("rect").attr("class", "c" + q.key)
          .attr("x", d => (d % cols) * cell).attr("y", d => Math.floor(d / cols) * cell)
          .attr("width", sq).attr("height", sq).attr("fill", q.color).attr("fill-opacity", q.op || 0.92);
      });
    }

    waffle(0, 0, [
      { key: "tp", n: TP, color: CL.tp },
      { key: "fp", n: FP, color: CL.fp }
    ], colsL);
    if (!pub) {
      waffle(leftW + gap, 0, [
        { key: "fn", n: FN, color: CL.fn, op: 0.85 },
        { key: "tn", n: TN, color: CL.tn, op: 0.9 }
      ], colsR);
    }

    /* headings */
    const head = (cx, txt, sub, col) => {
      g.append("text").attr("x", cx).attr("y", -26).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", col).text(txt);
      g.append("text").attr("x", cx).attr("y", -11).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", SC.muted).text(sub);
    };
    head(leftW / 2, "DECLARED SIGNIFICANT", disc + " of 1 000 hypotheses", SC.ink);
    if (!pub) head(leftW + gap + rightW / 2, "not significant", (FN + TN) + " hypotheses", SC.muted);
    else g.append("text").attr("x", leftW + gap + rightW / 2).attr("y", ih / 2)
      .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", SC.muted)
      .text("the other " + (FN + TN) + " were never published");

    /* the PPV split bar */
    const by = ih + 26, bh = 20;
    const bx = d3.scaleLinear().domain([0, Math.max(1, disc)]).range([0, iw]);
    g.append("rect").attr("x", 0).attr("y", by).attr("width", bx(TP)).attr("height", bh)
      .attr("fill", CL.tp).attr("fill-opacity", 0.85).attr("rx", 3);
    g.append("rect").attr("x", bx(TP)).attr("y", by).attr("width", bx(FP)).attr("height", bh)
      .attr("fill", CL.fp).attr("fill-opacity", 0.85).attr("rx", 3);
    if (bx(TP) > 54) g.append("text").attr("x", 8).attr("y", by + 14).attr("font-size", 11)
      .attr("fill", "#06101f").attr("font-weight", 600).text(TP + " real");
    if (bx(FP) > 54) g.append("text").attr("x", bx(TP) + 8).attr("y", by + 14).attr("font-size", 11)
      .attr("fill", "#1a0808").attr("font-weight", 600).text(FP + " false");
    g.append("text").attr("x", 0).attr("y", by + 38).attr("font-size", 11).attr("fill", SC.muted)
      .text("every square in this bar carries a p-value below α, and every one of them is correct");

    /* the legend lives in the empty space under the (short) discoveries column */
    ST.legend(g, [
      { label: "true positive — real effect, found", color: CL.tp },
      { label: "FALSE positive — no effect, declared significant", color: CL.fp },
      { label: "false negative — real effect, missed", color: CL.fn },
      { label: "true negative — no effect, correctly retained", color: CL.tn }
    ], 2, Math.min(ih - 52, rowsL * cell + 22), { gap: 15, font: 10.5 });

    out.innerHTML = `π = <b>${ST.fmt(pi, 3)}</b> · power = <b>${ST.fmt(pow, 2)}</b> · α = <b>${ST.fmt(al, 3)}</b>`
      + ` → <b>${disc}</b> discoveries: ${TP} true, <b>${FP} false</b>.`
      + `<br>Positive predictive value = ${TP}/${disc} = <b>${ST.pct(ppv, 1)}</b>`
      + ` · false discovery rate = <b>${ST.pct(1 - ppv, 1)}</b>.`
      + (ppv < 0.5
        ? ` <b>More than half the "discoveries" are false</b>, with a perfectly honest ${ST.fmt(al, 3)} test.`
        : ``)
      + (pub ? `<br>With publication bias on, this bar <b>is the literature</b> — a reader sees nothing else.` : ``);
  }

  ePi.oninput = () => { ePiv.textContent = (+ePi.value).toFixed(3); draw(); };
  ePw.oninput = () => { ePwv.textContent = (+ePw.value).toFixed(2); draw(); };
  eAl.oninput = () => { eAlv.textContent = (+eAl.value).toFixed(3); draw(); };
  ePub.onchange = draw;
  draw();
})();

/* ─────────────── 4 · alpha, beta and power as three areas ─────────────── */
(function () {
  const svg = d3.select("#pow-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 510;
  const eD = document.getElementById("pw-d"), eDv = document.getElementById("pw-dv");
  const eN = document.getElementById("pw-n"), eNv = document.getElementById("pw-nv");
  const eA = document.getElementById("pw-al"), eAv = document.getElementById("pw-alv");
  const eS = document.getElementById("pw-side");
  const out = document.getElementById("pow-readout");

  const st = { d: 0.6, n: 25, al: 0.05, side: 1 };

  function draw() {
    const { d, n, al, side } = st;
    const se = 1 / Math.sqrt(n);                 // sigma = 1 throughout
    const f = ST.frame(svg, W, H, { l: 54, r: 18, t: 16, b: 200 });
    const g = f.g, iw = f.iw, ih = f.ih;

    const lo = Math.min(-4.2 * se, d - 4.2 * se), hi = Math.max(4.2 * se, d + 4.2 * se);
    const x = d3.scaleLinear().domain([lo, hi]).range([0, iw]);
    const pdf0 = v => ST.normPdf(v / se) / se;
    const pdf1 = v => ST.normPdf((v - d) / se) / se;
    const xs = ST.linspace(lo, hi, 481);
    const y = d3.scaleLinear().domain([0, pdf0(0) * 1.16]).range([ih, 0]);

    ST.gridY(g, y, iw, 4);
    ST.axisB(g, x, ih, 8, "the estimate  x̄ − μ₀   (σ = 1)");
    ST.axisL(g, y, 4, "sampling density");

    const zc = side === 2 ? ST.normQuant(1 - al / 2) : ST.normQuant(1 - al);
    let crit = zc * se;
    crit = ST.clamp(crit, lo + 1e-9, hi - 1e-9);

    const a0 = d3.area().x(v => x(v)).y0(ih).y1(v => y(pdf0(v)));
    const a1 = d3.area().x(v => x(v)).y0(ih).y1(v => y(pdf1(v)));

    /* beta: alternative curve left of the threshold */
    g.append("path").attr("d", a1(ST.linspace(lo, crit, 200)))
      .attr("fill", SC.a2).attr("fill-opacity", 0.34);
    /* power: alternative curve right of the threshold */
    g.append("path").attr("d", a1(ST.linspace(crit, hi, 200)))
      .attr("fill", SC.good).attr("fill-opacity", 0.34);
    /* alpha sits ON TOP of beta and they genuinely overlap in x, so it gets a
       diagonal hatch as well as a fill — otherwise the overlap reads as one colour */
    const defs = f.svg.append("defs");
    defs.append("pattern").attr("id", "pw-hatch").attr("width", 6).attr("height", 6)
      .attr("patternUnits", "userSpaceOnUse").attr("patternTransform", "rotate(45)")
      .append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 6)
      .attr("stroke", SC.bad).attr("stroke-width", 2.6);
    const alphaArea = seg => {
      g.append("path").attr("d", a0(seg)).attr("fill", SC.bad).attr("fill-opacity", 0.40);
      g.append("path").attr("d", a0(seg)).attr("fill", "url(#pw-hatch)").attr("fill-opacity", 0.55)
        .attr("stroke", SC.bad).attr("stroke-width", 1.2);
    };
    alphaArea(ST.linspace(crit, hi, 200));
    if (side === 2) {
      alphaArea(ST.linspace(lo, -crit, 200));
      g.append("path").attr("d", a1(ST.linspace(lo, -crit, 200)))
        .attr("fill", SC.good).attr("fill-opacity", 0.34);
    }

    const l0 = d3.line().x(v => x(v)).y(v => y(pdf0(v)));
    const l1 = d3.line().x(v => x(v)).y(v => y(pdf1(v)));
    g.append("path").attr("d", l0(xs)).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
    g.append("path").attr("d", l1(xs)).attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 2);
    g.append("text").attr("x", x(0)).attr("y", y(pdf0(0)) - 8).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", SC.accent).text("H₀ : δ = 0");
    g.append("text").attr("x", x(d)).attr("y", y(pdf1(d)) - 8).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", SC.violet).text("truth : δ = " + ST.fmt(d, 2));

    /* draggable threshold */
    const dg = g.append("g").attr("class", "dragpt").style("cursor", "ew-resize");
    dg.append("line").attr("x1", x(crit)).attr("x2", x(crit)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", SC.ink).attr("stroke-width", 2);
    dg.append("rect").attr("x", x(crit) - 10).attr("y", 0).attr("width", 20).attr("height", ih)
      .attr("fill", "transparent");
    dg.append("circle").attr("cx", x(crit)).attr("cy", 9).attr("r", 5.5)
      .attr("fill", SC.ink).attr("stroke", SC.bg).attr("stroke-width", 1.5);
    dg.call(d3.drag().on("drag", ev => {
      const v = Math.max(1e-4, x.invert(ST.clamp(ev.x, 0, iw)));
      const z = v / se;
      st.al = ST.clamp(side === 2 ? 2 * (1 - ST.normCdf(z)) : 1 - ST.normCdf(z), 0.001, 0.30);
      eA.value = st.al.toFixed(3); eAv.textContent = st.al.toFixed(3);
      draw();
    }));

    const alpha = side === 2 ? 2 * (1 - ST.normCdf(zc)) : 1 - ST.normCdf(zc);
    const D = d * Math.sqrt(n);
    const power = HT.zPower(D, alpha, side === 1);
    const beta = 1 - power;

    ST.legend(g, [
      { label: "α — reject though H₀ is true", color: SC.bad, op: 0.5 },
      { label: "β — fail to reject though H₁ is true", color: SC.a2, op: 0.36 },
      { label: "power = 1 − β", color: SC.good, op: 0.36 }
    ], 8, 14, { gap: 14, font: 10.5 });

    /* the power function underneath */
    const ph = 104;
    const gp = g.append("g").attr("transform", `translate(0,${ih + 52})`);
    const px = d3.scaleLinear().domain([0, 1.6]).range([0, iw]);
    const py = d3.scaleLinear().domain([0, 1]).range([ph, 0]);
    gp.append("g").attr("class", "gridlines").selectAll("line").data(py.ticks(3)).join("line")
      .attr("x1", 0).attr("x2", iw).attr("y1", v => py(v)).attr("y2", v => py(v)).attr("stroke", SC.grid);
    gp.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`)
      .call(d3.axisBottom(px).ticks(8));
    gp.append("g").attr("class", "axis").call(d3.axisLeft(py).ticks(3).tickFormat(d3.format(".0%")));
    gp.append("text").attr("x", iw).attr("y", ph + 30).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("true effect δ (in σ)  →  the power function");

    const grid = ST.linspace(0, 1.6, 161);
    const pl = d3.line().x(v => px(v)).y(v => py(HT.zPower(v * Math.sqrt(n), alpha, side === 1)));
    gp.append("path").attr("d", pl(grid)).attr("fill", "none")
      .attr("stroke", SC.good).attr("stroke-width", 2);
    gp.append("line").attr("x1", 0).attr("x2", iw).attr("y1", py(alpha)).attr("y2", py(alpha))
      .attr("stroke", SC.bad).attr("stroke-width", 1.2).attr("stroke-dasharray", "4 3");
    gp.append("text").attr("x", iw - 3).attr("y", py(alpha) - 7).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.bad).text("the power function's floor is α, reached at δ = 0");
    gp.append("circle").attr("cx", px(Math.min(d, 1.6))).attr("cy", py(power)).attr("r", 4.5)
      .attr("fill", SC.a2).attr("stroke", SC.bg).attr("stroke-width", 1.4);

    out.innerHTML = `n = <b>${n}</b> · SE = σ/√n = <b>${ST.fmt(se, 4)}</b> · true effect δ = <b>${ST.fmt(d, 2)}σ</b>`
      + ` · noncentrality δ√n/σ = <b>${ST.fmt(D, 3)}</b>`
      + `<br>α = <b>${ST.fmt(alpha, 4)}</b> · β = <b>${ST.fmt(beta, 4)}</b> · power = <b>${ST.pct(power, 1)}</b>`
      + ` · critical value = <b>${ST.fmt(crit, 4)}</b>`
      + `<br>α + β = <b>${ST.fmt(alpha + beta, 3)}</b> — not 1, and no reason it should be: they are probabilities in two different worlds.`
      + ` Push the line right and α falls while β rises; only more data shrinks both.`;
  }

  eD.oninput = () => { st.d = +eD.value; eDv.textContent = st.d.toFixed(2); draw(); };
  eN.oninput = () => { st.n = +eN.value; eNv.textContent = st.n; draw(); };
  eA.oninput = () => { st.al = +eA.value; eAv.textContent = st.al.toFixed(3); draw(); };
  eS.onchange = () => { st.side = +eS.value; draw(); };
  document.getElementById("pw-reset").onclick = () => {
    st.d = 0.6; st.n = 25; st.al = 0.05; st.side = 1;
    eD.value = "0.6"; eN.value = "25"; eA.value = "0.05"; eS.value = "1";
    eDv.textContent = "0.60"; eNv.textContent = "25"; eAv.textContent = "0.050";
    draw();
  };
  draw();
})();

/* ─────────────── 5 · the sample-size planner ─────────────── */
(function () {
  const svg = d3.select("#plan-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 460;
  const eDes = document.getElementById("pl-design");
  const eTgt = document.getElementById("pl-tgt"), eTgtv = document.getElementById("pl-tgtv");
  const eAl = document.getElementById("pl-al");
  const eD = document.getElementById("pl-d"), eDv = document.getElementById("pl-dv");
  const eEx = document.getElementById("pl-exact");
  const out = document.getElementById("plan-readout");

  const DELTAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2];
  const COLS = [SC.bad, SC.a2, SC.violet, SC.accent, SC.teal, SC.good];

  function draw() {
    const pairs = +eDes.value, target = +eTgt.value, al = +eAl.value;
    const dHi = +eD.value, exact = eEx.checked;

    const f = ST.frame(svg, W, H, { l: 56, r: 96, t: 18, b: 46 });
    const g = f.g, iw = f.iw, ih = f.ih;

    const nMin = pairs === 2 ? 2 : 3, nMax = 20000;
    const x = d3.scaleLog().domain([nMin, nMax]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(5)).join("line")
      .attr("x1", 0).attr("x2", iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", SC.grid);
    g.append("g").attr("class", "gridlines").selectAll("line.v")
      .data([10, 100, 1000, 10000]).join("line")
      .attr("y1", 0).attr("y2", ih).attr("x1", d => x(d)).attr("x2", d => x(d)).attr("stroke", SC.grid);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6, "~s"));
    ST.axisL(g, y, 5, "power", d3.format(".0%"));
    g.append("text").attr("x", iw).attr("y", ih + 36).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted)
      .text(pairs === 2 ? "n per group  (log scale)" : "n  (log scale)");

    /* target line */
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(target)).attr("y2", y(target))
      .attr("stroke", SC.ink).attr("stroke-width", 1.6).attr("stroke-dasharray", "6 4");
    g.append("text").attr("x", 3).attr("y", y(target) - 6).attr("font-size", 10.5)
      .attr("fill", SC.ink).text("target power " + ST.pct(target, 0));
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(al)).attr("y2", y(al))
      .attr("stroke", SC.bad).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");

    const ns = [];
    for (let e = Math.log10(nMin); e <= Math.log10(nMax) + 1e-9; e += 0.02) ns.push(Math.max(nMin, Math.round(Math.pow(10, e))));
    const uniq = [...new Set(ns)];

    const pw = (delta, n) => exact ? HT.tPower(delta, n, al, pairs)
      : HT.zPower(delta * Math.sqrt(pairs === 2 ? n / 2 : n), al, false);

    const rows = [];
    DELTAS.forEach((delta, i) => {
      const pts = uniq.map(n => [n, pw(delta, n)]).filter(q => isFinite(q[1]));
      const isHi = Math.abs(delta - dHi) < 0.026;
      const ln = d3.line().x(q => x(q[0])).y(q => y(q[1])).curve(d3.curveMonotoneX);
      g.append("path").attr("d", ln(pts)).attr("fill", "none")
        .attr("stroke", COLS[i]).attr("stroke-width", isHi ? 3.2 : 1.7)
        .attr("stroke-opacity", isHi ? 1 : 0.62);
      const need = HT.nForPower(delta, target, al, pairs, exact);
      rows.push({ delta, need, col: COLS[i], isHi });
      if (isFinite(need) && need <= nMax) {
        g.append("line").attr("x1", x(need)).attr("x2", x(need)).attr("y1", y(target)).attr("y2", ih)
          .attr("stroke", COLS[i]).attr("stroke-width", isHi ? 2 : 1)
          .attr("stroke-dasharray", "3 3").attr("stroke-opacity", isHi ? 0.95 : 0.45);
        g.append("circle").attr("cx", x(need)).attr("cy", y(target)).attr("r", isHi ? 5 : 3)
          .attr("fill", COLS[i]).attr("stroke", SC.bg).attr("stroke-width", 1.2);
      }
    });

    /* labels, stacked at even spacing — at n = 20 000 every curve is at power 1,
       so labelling them at the curve's right end would pile them all up */
    const lg = g.append("g").attr("transform", `translate(${iw + 10},18)`);
    rows.forEach((rw, i) => {
      lg.append("line").attr("x1", 0).attr("x2", 14).attr("y1", i * 23).attr("y2", i * 23)
        .attr("stroke", rw.col).attr("stroke-width", rw.isHi ? 3.2 : 1.8);
      lg.append("text").attr("x", 18).attr("y", i * 23 + 3.5).attr("font-size", 10.5)
        .attr("fill", rw.col).attr("font-weight", rw.isHi ? 700 : 400)
        .text("δ = " + rw.delta + "σ");
      lg.append("text").attr("x", 18).attr("y", i * 23 + 3.5 + 10).attr("font-size", 9)
        .attr("fill", SC.muted)
        .text(isFinite(rw.need) ? "n = " + rw.need.toLocaleString() : "n > 20 000");
    });

    /* the highlighted row, spelled out */
    const hi = rows.find(r => r.isHi) || rows[0];
    const zn = HT.nForPower(hi.delta, target, al, pairs, false);
    const tn = HT.nForPower(hi.delta, target, al, pairs, true);
    out.innerHTML =
      `${pairs === 2 ? "Two independent groups" : "One sample / paired"} · α = <b>${al}</b> · target power <b>${ST.pct(target, 0)}</b>`
      + ` · using the <b>${exact ? "exact t" : "normal formula"}</b>.`
      + `<br>To detect δ = <b>${ST.fmt(hi.delta, 2)}σ</b> you need <b>${isFinite(hi.need) ? hi.need.toLocaleString() : "more than 20 000"}</b>`
      + `${pairs === 2 ? " per group (" + (isFinite(hi.need) ? (2 * hi.need).toLocaleString() : "—") + " in total)" : ""}.`
      + ` Normal formula says ${isFinite(zn) ? zn.toLocaleString() : "—"}; exact t says ${isFinite(tn) ? tn.toLocaleString() : "—"}`
      + ` — the formula always <b>under</b>-counts, and by more the smaller n is.`
      + `<br>Required n scales like 1/δ²: halving the effect you want to catch <b>quadruples</b> the study.`;
  }

  eDes.onchange = draw;
  eAl.onchange = draw;
  eEx.onchange = draw;
  eTgt.oninput = () => { eTgtv.textContent = (+eTgt.value).toFixed(2); draw(); };
  eD.oninput = () => { eDv.textContent = (+eD.value).toFixed(2); draw(); };
  draw();
})();

/* ─────────────── 6 · the interval–test duality ─────────────── */
(function () {
  const svg = d3.select("#dual-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 440;
  const eT = document.getElementById("du-th0"), eTv = document.getElementById("du-th0v");
  const eL = document.getElementById("du-lvl");
  const eN = document.getElementById("du-n"), eNv = document.getElementById("du-nv");
  const eC = document.getElementById("du-curve");
  const out = document.getElementById("dual-readout");

  let seed = 9021, samp = null;

  /* one sample from Normal(TRUE_MU, 1); the reader only ever sees x̄ and s */
  const TRUE_MU = 1.7;
  function newSample(n) {
    const r = ST.rng(seed), a = [];
    for (let i = 0; i < n; i++) a.push(TRUE_MU + ST.randn(r));
    samp = { xbar: ST.mean(a), s: ST.sd(a), n: n };
  }

  function draw() {
    const n = +eN.value, lvl = +eL.value, th0 = +eT.value;
    if (!samp || samp.n !== n) newSample(n);
    const { xbar, s } = samp;
    const se = s / Math.sqrt(n), df = n - 1, al = 1 - lvl;
    const tc = ST.tQuant(1 - al / 2, df);
    const lo = xbar - tc * se, hi = xbar + tc * se;
    const tobs = (xbar - th0) / se;
    const pv = 2 * (1 - ST.tCdf(Math.abs(tobs), df));
    const inside = th0 >= lo && th0 <= hi;

    const showCurve = eC.checked;
    const f = ST.frame(svg, W, H, { l: 56, r: 20, t: 18, b: 54 });
    const g = f.g, iw = f.iw, ih = f.ih;
    /* one fixed vertical budget so nothing can collide: curve, caption, mean
       label, the interval bar, its endpoint numbers, then the axis, then the
       verdict. Every y below is derived from curveH.                          */
    const curveH = showCurve ? Math.round(ih * 0.55) : 0;
    const meanY = curveH + (showCurve ? 48 : 44);
    const barY = curveH + (showCurve ? 62 : 58);
    const endY = curveH + (showCurve ? 80 : 76);
    const axisY = curveH + (showCurve ? 96 : 92);

    const XLO = -1.2, XHI = 4.6;
    const x = d3.scaleLinear().domain([XLO, XHI]).range([0, iw]);

    /* ── the p-value function ── */
    if (showCurve) {
      const y = d3.scaleLinear().domain([0, 1]).range([curveH, 0]);
      g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(4)).join("line")
        .attr("x1", 0).attr("x2", iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", SC.grid);
      g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".1f")));
      g.append("text").attr("x", 0).attr("y", -4).attr("font-size", 11).attr("fill", SC.muted)
        .text("p-value of H₀: μ = the value below");

      const grid = ST.linspace(XLO, XHI, 421);
      const pAt = v => 2 * (1 - ST.tCdf(Math.abs((xbar - v) / se), df));

      /* shade the region where p > alpha — this IS the interval */
      const ar = d3.area().x(v => x(v)).y0(y(al)).y1(v => y(pAt(v)));
      g.append("path").attr("d", ar(ST.linspace(lo, hi, 200)))
        .attr("fill", SC.accent).attr("fill-opacity", 0.22);

      const ln = d3.line().x(v => x(v)).y(v => y(pAt(v)));
      g.append("path").attr("d", ln(grid)).attr("fill", "none")
        .attr("stroke", SC.accent).attr("stroke-width", 2.2);

      g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(al)).attr("y2", y(al))
        .attr("stroke", SC.bad).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 4");
      g.append("text").attr("x", iw - 3).attr("y", y(al) - 6).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.bad).text("α = " + ST.fmt(al, 2));

      /* drop-lines from the two crossings all the way down to the interval bar */
      [lo, hi].forEach(v => {
        g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", y(al)).attr("y2", barY)
          .attr("stroke", SC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
        g.append("circle").attr("cx", x(v)).attr("cy", y(al)).attr("r", 4)
          .attr("fill", SC.good).attr("stroke", SC.bg).attr("stroke-width", 1.2);
      });
      g.append("text").attr("x", 4).attr("y", 30).attr("font-size", 10.5).attr("fill", SC.good)
        .text("the curve sits above α on exactly the interval drawn below — that is the whole theorem");

      /* the marker on the curve */
      g.append("line").attr("x1", x(th0)).attr("x2", x(th0)).attr("y1", y(pv)).attr("y2", curveH)
        .attr("stroke", SC.a2).attr("stroke-width", 1.4);
      g.append("circle").attr("cx", x(th0)).attr("cy", y(pv)).attr("r", 5)
        .attr("fill", SC.a2).attr("stroke", SC.bg).attr("stroke-width", 1.4);
    }

    /* ── the single parameter axis ── */
    const ax = g.append("g").attr("transform", `translate(0,${axisY})`);
    ax.append("g").attr("class", "axis").call(d3.axisBottom(x).ticks(9));
    ax.append("text").attr("x", iw).attr("y", 32).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("μ  — one axis, every candidate null value on it");

    /* the interval, on its own row */
    g.append("line").attr("x1", x(lo)).attr("x2", x(hi)).attr("y1", barY).attr("y2", barY)
      .attr("stroke", SC.good).attr("stroke-width", 5).attr("stroke-linecap", "round");
    [lo, hi].forEach(v => g.append("line").attr("x1", x(v)).attr("x2", x(v))
      .attr("y1", barY - 7).attr("y2", barY + 7).attr("stroke", SC.good).attr("stroke-width", 2.4));
    g.append("circle").attr("cx", x(xbar)).attr("cy", barY).attr("r", 5)
      .attr("fill", SC.ink).attr("stroke", SC.bg).attr("stroke-width", 1.5);
    g.append("text").attr("x", x(xbar)).attr("y", meanY).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", SC.ink).text("x̄ = " + ST.fmt(xbar, 3));
    g.append("text").attr("x", x(lo)).attr("y", endY).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.good).text(ST.fmt(lo, 2));
    g.append("text").attr("x", x(hi)).attr("y", endY).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.good).text(ST.fmt(hi, 2));
    g.append("text").attr("x", iw).attr("y", barY - 12).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.good)
      .text(ST.pct(lvl, 0) + " confidence interval");

    /* the null under test, draggable — its label goes to whichever side has room */
    const dg = g.append("g").attr("class", "dragpt").style("cursor", "ew-resize");
    const topY = showCurve ? curveH + 6 : barY - 44;
    dg.append("line").attr("x1", x(th0)).attr("x2", x(th0)).attr("y1", topY).attr("y2", axisY + 8)
      .attr("stroke", inside ? SC.a2 : SC.bad).attr("stroke-width", 2.5);
    dg.append("rect").attr("x", x(th0) - 12).attr("y", topY).attr("width", 24)
      .attr("height", axisY + 8 - topY).attr("fill", "transparent");
    dg.append("circle").attr("cx", x(th0)).attr("cy", topY).attr("r", 6)
      .attr("fill", inside ? SC.a2 : SC.bad).attr("stroke", SC.bg).attr("stroke-width", 1.5);
    const tSide = x(th0) > iw - 90 ? -10 : 10;
    dg.append("text").attr("x", x(th0) + tSide).attr("y", topY + 4)
      .attr("text-anchor", tSide < 0 ? "end" : "start")
      .attr("font-size", 10.5).attr("font-weight", 600)
      .attr("fill", inside ? SC.a2 : SC.bad)
      .text("θ₀ = " + ST.fmt(th0, 2));
    dg.call(d3.drag().on("drag", ev => {
      const v = ST.clamp(x.invert(ST.clamp(ev.x, 0, iw)), -1.2, 4.6);
      eT.value = v.toFixed(2); eTv.textContent = v.toFixed(2);
      draw();
    }));

    /* verdict banner */
    g.append("text").attr("x", iw).attr("y", axisY + 52).attr("text-anchor", "end")
      .attr("font-size", 12.5).attr("font-weight", 600)
      .attr("fill", inside ? SC.good : SC.bad)
      .text(inside ? "θ₀ inside the interval  ⟺  p > α  ⟹  do not reject"
                   : "θ₀ outside the interval  ⟺  p ≤ α  ⟹  REJECT");

    out.innerHTML = `n = <b>${n}</b> · x̄ = <b>${ST.fmt(xbar, 4)}</b> · s = <b>${ST.fmt(s, 4)}</b>`
      + ` · SE = <b>${ST.fmt(se, 4)}</b> · t*(${df}) = <b>${ST.fmt(tc, 4)}</b>`
      + `<br>${ST.pct(lvl, 0)} interval = <b>(${ST.fmt(lo, 3)}, ${ST.fmt(hi, 3)})</b>`
      + ` · testing H₀: μ = ${ST.fmt(th0, 2)} gives t = <b>${ST.fmt(tobs, 4)}</b>, p = <b>${pv < 1e-4 ? pv.toExponential(2) : ST.fmt(pv, 4)}</b>.`
      + `<br>${inside
        ? `p = ${ST.fmt(pv, 3)} &gt; α = ${ST.fmt(al, 2)} and θ₀ is inside the bar. The two facts are the same fact.`
        : `p = ${pv < 1e-4 ? pv.toExponential(2) : ST.fmt(pv, 4)} ≤ α = ${ST.fmt(al, 2)} and θ₀ is outside the bar. The two facts are the same fact.`}`
      + ` Slide θ₀ to an endpoint and p lands exactly on α.`;
  }

  eT.oninput = () => { eTv.textContent = (+eT.value).toFixed(2); draw(); };
  eL.onchange = draw;
  eC.onchange = draw;
  eN.oninput = () => { eNv.textContent = eN.value; samp = null; draw(); };
  document.getElementById("du-new").onclick = () => {
    seed = (seed * 1103515245 + 12345) >>> 0; samp = null; draw();
  };
  draw();
})();

/* ─────────────── 7 · how 5% becomes 60% ─────────────── */
(function () {
  const svg = d3.select("#hack-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 440;
  const eM = document.getElementById("hk-mode");
  const eK = document.getElementById("hk-k"), eKv = document.getElementById("hk-kv");
  const eKK = document.getElementById("hk-K"), eKKv = document.getElementById("hk-Kv");
  const eA = document.getElementById("hk-al");
  const out = document.getElementById("hack-readout");

  let seed = 5150;

  /* ONE hacked "study", under a TRUE null. Returns the p-value the researcher reports.
     - k outcomes: k independent measurements on the same subjects, best (smallest p) kept
     - K looks: data arrive in K equal batches up to nMax, stop at the first significant look
     Both are honest simulations: the data contain no effect whatsoever.               */
  function study(k, K, al, r) {
    const nMax = 240, per = Math.max(4, Math.round(nMax / K));
    let best = 1;
    for (let j = 0; j < k; j++) {
      let s = 0, ss = 0, m = 0, pj = 1;
      for (let look = 1; look <= K; look++) {
        const upto = Math.min(nMax, look * per);
        while (m < upto) { const x = ST.randn(r); s += x; ss += x * x; m++; }
        const mn = s / m, v = (ss - m * mn * mn) / (m - 1);
        if (v > 0 && m > 2) {
          const t = mn / Math.sqrt(v / m);
          const p = 2 * (1 - ST.tCdf(Math.abs(t), m - 1));
          if (p < pj) pj = p;
          if (p <= al) break;                      // the researcher stops here
        }
      }
      if (pj < best) best = pj;
    }
    return best;
  }

  function run() {
    const mode = eM.value, al = +eA.value;
    const k = (mode === "peek") ? 1 : +eK.value;
    const K = (mode === "best") ? 1 : +eKK.value;
    const B = 4000;
    const r = ST.rng(seed);

    const ps = [];
    let hits = 0;
    for (let b = 0; b < B; b++) { const p = study(k, K, al, r); ps.push(p); if (p <= al) hits++; }
    const actual = hits / B;
    const bestOnly = 1 - Math.pow(1 - al, k);          // exact for the k-outcomes half

    const f = ST.frame(svg, W, H, { l: 52, r: 16, t: 20, b: 52 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const gapx = 40, leftW = Math.round(iw * 0.36), rightW = iw - leftW - gapx;

    /* ── left: nominal vs actual ── */
    const gl = g.append("g");
    const ymax = Math.max(0.12, actual * 1.28);
    const y = d3.scaleLinear().domain([0, ymax]).range([ih, 0]);
    const bx = d3.scaleBand().domain(["nominal", "actual"]).range([0, leftW]).padding(0.34);
    gl.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(5)).join("line")
      .attr("x1", 0).attr("x2", leftW).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", SC.grid);
    gl.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
    gl.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(bx));
    gl.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11).attr("fill", SC.muted)
      .text("false-positive rate, H₀ true");

    [["nominal", al, SC.accent], ["actual", actual, actual > al * 1.25 ? SC.bad : SC.good]]
      .forEach(([kk, v, c]) => {
        gl.append("rect").attr("x", bx(kk)).attr("y", y(v)).attr("width", bx.bandwidth())
          .attr("height", ih - y(v)).attr("rx", 3).attr("fill", c).attr("fill-opacity", 0.82);
        gl.append("text").attr("x", bx(kk) + bx.bandwidth() / 2).attr("y", y(v) - 7)
          .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 600)
          .attr("fill", c).text(ST.pct(v, 1));
      });
    if (mode === "best" && k > 1) {
      gl.append("line").attr("x1", bx("actual") - 6).attr("x2", bx("actual") + bx.bandwidth() + 6)
        .attr("y1", y(bestOnly)).attr("y2", y(bestOnly))
        .attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
      gl.append("text").attr("x", bx("actual") + bx.bandwidth() + 8).attr("y", y(bestOnly) + 3.5)
        .attr("font-size", 10).attr("fill", SC.a2).text("1−(1−α)ᵏ");
    }

    /* ── right: the reported p-values ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + gapx},0)`);
    const NB = 25, counts = new Array(NB).fill(0);
    ps.forEach(p => counts[Math.min(NB - 1, Math.floor(p * NB))]++);
    const props = counts.map(c => c / B);
    const x2 = d3.scaleLinear().domain([0, 1]).range([0, rightW]);
    const y2 = d3.scaleLinear().domain([0, Math.max(0.06, d3.max(props) * 1.15)]).range([ih, 0]);
    gr.append("g").attr("class", "gridlines").selectAll("line").data(y2.ticks(5)).join("line")
      .attr("x1", 0).attr("x2", rightW).attr("y1", d => y2(d)).attr("y2", d => y2(d)).attr("stroke", SC.grid);
    gr.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5).tickFormat(d3.format(".0%")));
    gr.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x2).ticks(5));
    gr.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11).attr("fill", SC.muted)
      .text("the p-values the researcher reports");
    gr.append("text").attr("x", rightW).attr("y", ih + 38).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("reported p-value");

    const bw2 = rightW / NB;
    gr.selectAll("rect.pb").data(props).join("rect").attr("class", "pb")
      .attr("x", (d, i) => i * bw2 + 0.8).attr("width", bw2 - 1.6)
      .attr("y", d => y2(d)).attr("height", d => ih - y2(d)).attr("rx", 1.5)
      .attr("fill", (d, i) => (i * (1 / NB) < al ? SC.bad : SC.accent)).attr("fill-opacity", 0.85);
    gr.append("line").attr("x1", 0).attr("x2", rightW).attr("y1", y2(1 / NB)).attr("y2", y2(1 / NB))
      .attr("stroke", SC.good).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 4");
    /* the reference height sits right among the bars, so its label goes to the
       clear space at the top of the panel with its own dashed swatch */
    gr.append("line").attr("x1", rightW - 176).attr("x2", rightW - 162).attr("y1", 12).attr("y2", 12)
      .attr("stroke", SC.good).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
    gr.append("text").attr("x", rightW - 156).attr("y", 15.5)
      .attr("font-size", 10).attr("fill", SC.good).text("what an honest test would give");

    const mult = actual / al;
    out.innerHTML = `<b>4 000</b> simulated studies, each with <b>no effect at all</b> in the data.`
      + ` Strategy: ${mode === "best" ? `try <b>${k}</b> outcome${k > 1 ? "s" : ""} and report the best`
        : mode === "peek" ? `test after each of <b>${K}</b> equally spaced looks and stop at the first significant one`
          : `<b>${k}</b> outcome${k > 1 ? "s" : ""} × <b>${K}</b> look${K > 1 ? "s" : ""}`}.`
      + `<br>Nominal α = <b>${al}</b>. Actual false-positive rate = <b>${ST.pct(actual, 1)}</b>`
      + ` — <b>${ST.fmt(mult, 2)}×</b> the advertised rate.`
      + (mode === "best" && K === 1
        ? ` Theory for k independent looks: 1 − (1 − α)ᵏ = <b>${ST.pct(bestOnly, 1)}</b>.` : ``)
      + `<br>Nothing dishonest happened in any single study: every p-value on the right is correctly computed for the test that produced it. The error rate broke because of the analyses that were <i>available</i>, not the one that was run.`;
  }

  eM.onchange = run;
  eA.onchange = run;
  eK.oninput = () => { eKv.textContent = eK.value; run(); };
  eKK.oninput = () => { eKKv.textContent = eKK.value; run(); };
  document.getElementById("hk-run").onclick = () => { seed = (seed * 1103515245 + 12345) >>> 0; run(); };
  run();
})();
