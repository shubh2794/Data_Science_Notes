/* estimation.viz.js — the seven visualizations on
   math/statistics/estimation.html.
   Loaded after ../../data.js → ../../notes.js → stats-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #bvm-svg    bias, variance and MSE traded against each other by a shrinkage
                    factor λ — the MSE minimum sits at a NON-zero amount of bias
     2  #lik-svg    the likelihood machine: drag λ along the log-likelihood, watch the
                    fitted density move, read the curvature off as a standard error
     3  #cover-svg  a hundred intervals against one fixed truth, with a running
                    coverage counter that converges on the nominal level
     4  #t-svg      t against normal: the tails, the critical value, and t*(ν) → 1.96
     5  #wald-svg   EXACT coverage of a nominal 95% interval as p varies — the Wald
                    saw teeth, with Wilson / Agresti–Coull / Clopper–Pearson overlaid
     6  #plan-svg   the sample-size planner: margin of error against n on log–log
     7  #four-svg   confidence, prediction, tolerance and credible intervals, one axis  */

/* ══════════ shared local helpers (deliberately NOT added to stats-viz.js) ══════════ */
const EST = (function () {
  /* χ² quantile, by bisection on the regularised incomplete gamma that stats-viz
     already provides. Only this page needs it, so it stays local.               */
  /* chi-square quantile now lives in stats-viz.js as ST.chi2Quant (part 5 added it,
     tested against 60 published critical values); the local bisection that used to
     sit here agreed with it to 4e-14 and was removed as a duplicate. */
  /* beta quantile, for the Clopper–Pearson endpoints */
  function betaQuant(q, a, b) {
    let lo = 0, hi = 1;
    for (let i = 0; i < 80; i++) {
      const m = 0.5 * (lo + hi);
      if (ST.betaI(a, b, m) < q) lo = m; else hi = m;
    }
    return 0.5 * (lo + hi);
  }
  /* the four interval recipes for a proportion, all as [lo, hi] given a count k */
  const CI = {
    wald: (k, n, z) => { const p = k / n, h = z * Math.sqrt(p * (1 - p) / n); return [p - h, p + h]; },
    wilson: (k, n, z) => {
      const c = (k + z * z / 2) / (n + z * z);
      const h = (z / (n + z * z)) * Math.sqrt(k * (n - k) / n + z * z / 4);
      return [Math.max(0, c - h), Math.min(1, c + h)];
    },
    ac: (k, n, z) => {
      const nt = n + z * z, pt = (k + z * z / 2) / nt, h = z * Math.sqrt(pt * (1 - pt) / nt);
      return [Math.max(0, pt - h), Math.min(1, pt + h)];
    },
    cp: (k, n, alpha) => [
      k === 0 ? 0 : betaQuant(alpha / 2, k, n - k + 1),
      k === n ? 1 : betaQuant(1 - alpha / 2, k + 1, n - k)
    ]
  };
  /* EXACT coverage of an interval recipe at (n, p): sum the binomial mass of the
     counts whose interval happens to contain p. Uses the pmf recurrence so the
     whole curve costs O(#p · n) cheap multiplications instead of O(#p · n) lgammas. */
  function coverage(n, p, lohi) {
    if (p <= 0 || p >= 1) return NaN;
    let pmf = Math.exp(n * Math.log1p(-p)), s = 0;          // pmf at k = 0
    const r = p / (1 - p);
    for (let k = 0; k <= n; k++) {
      if (lohi[k][0] <= p && p <= lohi[k][1]) s += pmf;
      pmf *= r * (n - k) / (k + 1);
    }
    return Math.min(1, s);
  }
  return { betaQuant, CI, coverage };
})();

/* ─────────────────── 1 · bias, variance and MSE ─────────────────── */
(function () {
  const svg = d3.select("#bvm-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 430;
  const sLam = document.getElementById("bv-lam"), oLam = document.getElementById("bv-lamv");
  const sMu = document.getElementById("bv-mu"), oMu = document.getElementById("bv-muv");
  const sSd = document.getElementById("bv-sd"), oSd = document.getElementById("bv-sdv");
  const sN = document.getElementById("bv-n"), oN = document.getElementById("bv-nv");
  const bOpt = document.getElementById("bv-opt");
  const out = document.getElementById("bvm-readout");

  const F = ST.frame(svg, W, H, { l: 44, r: 14, t: 26, b: 42 });
  const GAPX = 56, PW = (F.iw - GAPX) / 2;
  const gL = F.g.append("g"), gR = F.g.append("g").attr("transform", `translate(${PW + GAPX},0)`);
  const PH = F.ih - 26;

  function render() {
    const lam = +sLam.value, mu = +sMu.value, sd = +sSd.value, n = +sN.value;
    oLam.textContent = ST.fmt(lam, 3); oMu.textContent = ST.fmt(mu, 2);
    oSd.textContent = ST.fmt(sd, 1); oN.textContent = n;

    const v = sd * sd / n;                        // variance of the unbiased estimator
    const lamStar = mu * mu / (mu * mu + v);
    const bias = (lam - 1) * mu, varL = lam * lam * v, mse = bias * bias + varL;
    const mseStar = mu * mu * v / (mu * mu + v);

    gL.selectAll("*").remove(); gR.selectAll("*").remove();

    /* ── left: the two sampling distributions ─────────────────────────── */
    const sdU = Math.sqrt(v);
    const half = Math.max(3.6 * sdU, 1.25 * Math.abs(mu) + 0.6);
    const x = d3.scaleLinear().domain([mu - half, mu + half]).range([0, PW]);
    const peak = ST.normPdf(0) / Math.max(1e-9, Math.min(sdU, lam * sdU || sdU));
    const y = d3.scaleLinear().domain([0, peak * 1.1]).range([PH, 8]);
    ST.gridY(gL, y, PW, 4);

    const grid = ST.linspace(x.domain()[0], x.domain()[1], 260);
    const curve = (c, s, col, op, dash) => {
      if (!(s > 1e-9)) return;
      const pts = grid.map(g => ({ g: g, d: ST.normPdf((g - c) / s) / s }));
      gL.append("path").datum(pts).attr("fill", col).attr("fill-opacity", op)
        .attr("d", d3.area().x(q => x(q.g)).y0(PH).y1(q => y(Math.min(q.d, y.domain()[1]))));
      gL.append("path").datum(pts).attr("fill", "none").attr("stroke", col)
        .attr("stroke-width", 2).attr("stroke-dasharray", dash || null)
        .attr("d", d3.line().x(q => x(q.g)).y(q => y(Math.min(q.d, y.domain()[1]))));
    };
    curve(mu, sdU, SC.muted, 0.14, "4 3");
    curve(lam * mu, lam * sdU, SC.accent, 0.24);

    gL.append("line").attr("x1", x(mu)).attr("x2", x(mu)).attr("y1", 4).attr("y2", PH)
      .attr("stroke", SC.good).attr("stroke-width", 2);
    gL.append("text").attr("x", x(mu) + 5).attr("y", 14).attr("font-size", 10.5)
      .attr("fill", SC.good).text("true θ");

    if (Math.abs(bias) > 1e-9) {                 // the bias arrow
      const yb = PH - 12;
      gL.append("line").attr("x1", x(mu)).attr("x2", x(lam * mu)).attr("y1", yb).attr("y2", yb)
        .attr("stroke", SC.bad).attr("stroke-width", 2.4);
      gL.append("text").attr("x", x((mu + lam * mu) / 2)).attr("y", yb - 6)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.bad)
        .text(`bias ${ST.fmt(bias, 3)}`);
    }
    gL.append("g").attr("class", "axis").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(x).ticks(5));
    gL.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.accent)
      .text("sampling distribution of θ̂ = λ·x̄");
    gL.append("text").attr("x", PW).attr("y", F.ih + 6).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted)
      .text("dashed grey = the unbiased estimator (λ = 1)");

    /* ── right: bias², variance and MSE against λ ─────────────────────── */
    const LMAX = 1.3;
    const xl = d3.scaleLinear().domain([0, LMAX]).range([0, PW]);
    const top = Math.max(mu * mu, v * LMAX * LMAX, mse) * 1.12;
    const yl = d3.scaleLinear().domain([0, top]).range([PH, 8]);
    ST.gridY(gR, yl, PW, 4);
    const ls = ST.linspace(0, LMAX, 200);
    const path = (f, col, dash, wdt) =>
      gR.append("path").datum(ls.map(L => ({ L: L, y: f(L) })))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", wdt || 2)
        .attr("stroke-dasharray", dash || null)
        .attr("d", d3.line().x(q => xl(q.L)).y(q => yl(Math.min(q.y, top))));
    path(L => (L - 1) * (L - 1) * mu * mu, SC.bad, "5 3");
    path(L => L * L * v, SC.a2, "5 3");
    path(L => (L - 1) * (L - 1) * mu * mu + L * L * v, SC.accent, null, 2.6);

    gR.append("line").attr("x1", xl(1)).attr("x2", xl(1)).attr("y1", 8).attr("y2", PH)
      .attr("stroke", SC.muted).attr("stroke-width", 1).attr("stroke-dasharray", "2 4");
    gR.append("text").attr("x", xl(1) - 5).attr("y", PH - 6).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.muted).text("unbiased");

    gR.append("circle").attr("cx", xl(lamStar)).attr("cy", yl(mseStar)).attr("r", 5)
      .attr("fill", "none").attr("stroke", SC.good).attr("stroke-width", 2);
    gR.append("text").attr("x", xl(lamStar)).attr("y", yl(mseStar) - 10).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.good).text("MSE minimum");

    gR.append("line").attr("x1", xl(lam)).attr("x2", xl(lam)).attr("y1", 8).attr("y2", PH)
      .attr("stroke", SC.ink).attr("stroke-width", 1.2);
    gR.append("circle").attr("class", "dragpt").attr("cx", xl(lam)).attr("cy", yl(Math.min(mse, top)))
      .attr("r", 6.5).attr("fill", SC.accent).attr("stroke", SC.bg).attr("stroke-width", 2)
      .style("cursor", "ew-resize");

    gR.append("g").attr("class", "axis").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(xl).ticks(6));
    gR.append("g").attr("class", "axis").call(d3.axisLeft(yl).ticks(4).tickFormat(d3.format(".3f")));
    gR.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.accent)
      .text("error against the shrinkage factor λ");
    gR.append("text").attr("x", PW).attr("y", F.ih + 6).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("λ  (1 = the unbiased estimator)");
    ST.legend(gR, [
      { label: "bias²", color: SC.bad, dash: "5 3" },
      { label: "variance", color: SC.a2, dash: "5 3" },
      { label: "MSE = bias² + variance", color: SC.accent }
    ], 8, 14);

    /* transparent drag surface over the right panel */
    gR.append("rect").attr("class", "dragpt").attr("x", 0).attr("y", 0)
      .attr("width", PW).attr("height", PH).attr("fill", "transparent")
      .style("cursor", "ew-resize")
      .call(d3.drag().on("start drag", ev => {
        const L = ST.clamp(xl.invert(d3.pointer(ev, gR.node())[0]), 0, LMAX);
        sLam.value = L.toFixed(3); render();
      }));

    const verdict = (Math.abs(lam - 1) < 1e-9)
      ? { col: SC.muted, msg: "λ = 1 IS the unbiased estimator — its MSE is exactly its variance, v" }
      : (mse < v
        ? { col: SC.good, msg: "this λ beats the unbiased estimator on mean squared error" }
        : { col: SC.bad, msg: "this λ is worse than the unbiased estimator — you have over- or under-shrunk" });
    out.innerHTML =
      `θ = <b>${ST.fmt(mu, 2)}</b> · σ = ${ST.fmt(sd, 1)} · n = ${n} → the unbiased estimator has ` +
      `variance v = σ²/n = <b>${ST.fmt(v, 5)}</b>` +
      `<br>at λ = <b>${ST.fmt(lam, 3)}</b>: bias = ${ST.fmt(bias, 4)} · bias² = ${ST.fmt(bias * bias, 5)} · ` +
      `variance = ${ST.fmt(varL, 5)} · <b>MSE = ${ST.fmt(mse, 5)}</b>` +
      `<br>MSE-optimal λ* = θ²/(θ² + v) = <b>${ST.fmt(lamStar, 4)}</b>, giving MSE = <b>${ST.fmt(mseStar, 5)}</b> — ` +
      `<b>${ST.fmt(100 * (1 - mseStar / v), 1)}%</b> below the unbiased estimator's ${ST.fmt(v, 5)}` +
      `<br><span style="color:${verdict.col}">${verdict.msg}</span>` +
      `<span style="color:${SC.muted}"> · λ* is strictly below 1 for every positive variance, so the ` +
      `unbiased estimator is never the MSE optimum.</span>`;
  }
  [sLam, sMu, sSd, sN].forEach(s => s.addEventListener("input", render));
  bOpt.addEventListener("click", () => {
    const mu = +sMu.value, v = (+sSd.value) * (+sSd.value) / (+sN.value);
    sLam.value = (mu * mu / (mu * mu + v)).toFixed(3); render();
  });
  render();
})();

/* ─────────────────── 2 · the likelihood machine ─────────────────── */
(function () {
  const svg = d3.select("#lik-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 450;
  const sN = document.getElementById("lk-n"), oN = document.getElementById("lk-nv");
  const sT = document.getElementById("lk-true"), oT = document.getElementById("lk-tv");
  const cQ = document.getElementById("lk-quad");
  const bNew = document.getElementById("lk-new"), bSnap = document.getElementById("lk-snap");
  const out = document.getElementById("lik-readout");

  /* the twelve lifetimes from the worked example on the page, used verbatim at n = 12
     so every printed number in §05 can be checked against the picture             */
  const SEED12 = [3.1, 0.4, 8.2, 1.7, 12.6, 2.9, 0.8, 5.5, 1.2, 6.7, 4.3, 0.6];
  let seed = 20250904, data = SEED12.slice(), handle = null;

  function resample() {
    const n = +sN.value, lam = +sT.value;
    if (n === 12 && Math.abs(lam - 0.25) < 1e-9 && seed === 20250904) { data = SEED12.slice(); return; }
    const r = ST.rng(seed);
    data = [];
    for (let i = 0; i < n; i++) data.push(-Math.log(1 - r()) / lam);
  }

  function render() {
    const n = data.length, S = ST.sum(data), lamHat = n / S;
    oN.textContent = +sN.value; oT.textContent = ST.fmt(+sT.value, 2);
    if (handle === null) handle = lamHat;
    const lam = ST.clamp(handle, lamHat / 6, lamHat * 4);

    const F = ST.frame(svg, W, H, { l: 50, r: 16, t: 22, b: 40 });
    const TOPH = 138, GAP = 56;
    const gT = F.g.append("g"), gB = F.g.append("g").attr("transform", `translate(0,${TOPH + GAP})`);
    const BOTH = F.ih - TOPH - GAP;

    /* ── top: the data and the fitted density ─────────────────────────── */
    const xmax = Math.max(d3.max(data) * 1.08, 3 / lamHat);
    const x = d3.scaleLinear().domain([0, xmax]).range([0, F.iw]);
    const dmax = Math.max(lam, lamHat) * 1.15;
    const y = d3.scaleLinear().domain([0, dmax]).range([TOPH, 6]);
    ST.gridY(gT, y, F.iw, 3);
    const gridx = ST.linspace(0, xmax, 240);
    const dens = (L, col, wdt, dash, op) => {
      const pts = gridx.map(g => ({ g: g, d: L * Math.exp(-L * g) }));
      if (op) gT.append("path").datum(pts).attr("fill", col).attr("fill-opacity", op)
        .attr("d", d3.area().x(q => x(q.g)).y0(TOPH).y1(q => y(q.d)));
      gT.append("path").datum(pts).attr("fill", "none").attr("stroke", col)
        .attr("stroke-width", wdt).attr("stroke-dasharray", dash || null)
        .attr("d", d3.line().x(q => x(q.g)).y(q => y(q.d)));
    };
    dens(lamHat, SC.good, 1.6, "5 3");
    dens(lam, SC.accent, 2.4, null, 0.16);
    gT.selectAll("line.rug").data(data).join("line").attr("class", "rug")
      .attr("x1", d => x(d)).attr("x2", d => x(d))
      .attr("y1", TOPH).attr("y2", TOPH - 13)
      .attr("stroke", SC.a2).attr("stroke-width", 1.4).attr("stroke-opacity", 0.75);
    gT.append("g").attr("class", "axis").attr("transform", `translate(0,${TOPH})`)
      .call(d3.axisBottom(x).ticks(7));
    gT.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3));
    gT.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.accent)
      .text(`the ${n} observed lifetimes (orange ticks) and the exponential density at the handle`);
    gT.append("text").attr("x", F.iw).attr("y", TOPH + 30).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("lifetime, hours");
    gT.append("text").attr("x", F.iw).attr("y", 10).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.good).text("dashed green = the maximum-likelihood fit");

    /* ── bottom: the log-likelihood ───────────────────────────────────── */
    const lo = lamHat / 6, hi = lamHat * 4;
    const xl = d3.scaleLinear().domain([lo, hi]).range([0, F.iw]);
    const ll = L => n * Math.log(L) - L * S;
    const peak = ll(lamHat), J = n / (lamHat * lamHat), se = 1 / Math.sqrt(J);
    const span = Math.max(6, 3.2 * n * 0.5);
    const ymin = Math.min(ll(lo), ll(hi), peak - span);
    const yl = d3.scaleLinear().domain([ymin, peak + (peak - ymin) * 0.10]).range([BOTH, 8]);
    ST.gridY(gB, yl, F.iw, 4);

    const lg = ST.linspace(lo, hi, 300);
    gB.append("path").datum(lg.map(L => ({ L: L, y: ll(L) })))
      .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.4)
      .attr("d", d3.line().x(q => xl(q.L)).y(q => yl(Math.max(q.y, ymin))));
    if (cQ.checked) {
      gB.append("path").datum(lg.map(L => ({ L: L, y: peak - 0.5 * J * (L - lamHat) * (L - lamHat) })))
        .attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 1.8)
        .attr("stroke-dasharray", "6 4")
        .attr("d", d3.line().x(q => xl(q.L)).y(q => yl(Math.max(q.y, ymin))));
    }
    gB.append("line").attr("x1", xl(lamHat)).attr("x2", xl(lamHat))
      .attr("y1", yl(peak)).attr("y2", BOTH).attr("stroke", SC.good).attr("stroke-width", 1.6);
    gB.append("text").attr("x", xl(lamHat) + 5).attr("y", yl(peak) - 6)
      .attr("font-size", 10.5).attr("fill", SC.good).text(`λ̂ = ${ST.fmt(lamHat, 4)}`);

    /* the 95% interval, drawn as a band on the λ axis */
    const zc = ST.normQuant(0.975), ciLo = lamHat - zc * se, ciHi = lamHat + zc * se;
    const yb = BOTH - 10;
    gB.append("rect").attr("x", xl(Math.max(lo, ciLo))).attr("y", yb - 5)
      .attr("width", Math.max(0, xl(Math.min(hi, ciHi)) - xl(Math.max(lo, ciLo)))).attr("height", 10)
      .attr("fill", SC.good).attr("fill-opacity", 0.18).attr("stroke", SC.good).attr("stroke-width", 1);
    gB.append("text").attr("x", xl(lamHat)).attr("y", yb - 17).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", SC.good)
      .text(`95% interval  λ̂ ± 1.96/√J  =  (${ST.fmt(ciLo, 4)}, ${ST.fmt(ciHi, 4)})`);

    gB.append("line").attr("x1", xl(lam)).attr("x2", xl(lam)).attr("y1", 8).attr("y2", BOTH)
      .attr("stroke", SC.ink).attr("stroke-width", 1.2);
    gB.append("circle").attr("class", "dragpt").attr("cx", xl(lam))
      .attr("cy", yl(Math.max(ll(lam), ymin))).attr("r", 7)
      .attr("fill", SC.accent).attr("stroke", SC.bg).attr("stroke-width", 2)
      .style("cursor", "ew-resize");
    gB.append("g").attr("class", "axis").attr("transform", `translate(0,${BOTH})`)
      .call(d3.axisBottom(xl).ticks(7));
    gB.append("g").attr("class", "axis").call(d3.axisLeft(yl).ticks(4));
    gB.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.accent)
      .text("log-likelihood  ℓ(λ) = n·log λ − λ·∑xᵢ");
    gB.append("text").attr("x", F.iw).attr("y", BOTH + 32).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("rate λ, per hour");
    if (cQ.checked) ST.legend(gB, [
      { label: "log-likelihood", color: SC.accent },
      { label: "quadratic from the observed information", color: SC.violet, dash: "6 4" }
    ], F.iw - 238, 16);

    gB.append("rect").attr("class", "dragpt").attr("x", 0).attr("y", 0)
      .attr("width", F.iw).attr("height", BOTH).attr("fill", "transparent")
      .style("cursor", "ew-resize")
      .call(d3.drag().on("start drag", ev => {
        handle = ST.clamp(xl.invert(d3.pointer(ev, gB.node())[0]), lo, hi); render();
      }));

    const drop = peak - ll(lam);
    out.innerHTML =
      `n = <b>${n}</b> · ∑xᵢ = <b>${ST.fmt(S, 2)}</b> · x̄ = ${ST.fmt(S / n, 4)} → ` +
      `λ̂ = n/∑xᵢ = <b>${ST.fmt(lamHat, 5)}</b> per hour` +
      `<br>observed information J = −ℓ″(λ̂) = n/λ̂² = <b>${ST.fmt(J, 2)}</b> → ` +
      `ŜE = 1/√J = <b>${ST.fmt(se, 5)}</b> (and λ̂/√n = ${ST.fmt(lamHat / Math.sqrt(n), 5)}, the same number)` +
      `<br>handle at λ = <b>${ST.fmt(lam, 4)}</b>: ℓ(λ) = ${ST.fmt(ll(lam), 3)}, which is ` +
      `<b>${ST.fmt(drop, 3)}</b> below the peak of ${ST.fmt(peak, 3)} — the likelihood ratio is ` +
      `e^(−${ST.fmt(drop, 2)}) = ${ST.sig(Math.exp(-drop), 3)}` +
      `<br>by invariance: mean = 1/λ̂ = <b>${ST.fmt(1 / lamHat, 3)}</b> h · ` +
      `median = log2/λ̂ = <b>${ST.fmt(Math.LN2 / lamHat, 3)}</b> h · P(X &gt; 10 h) = ` +
      `<b>${ST.pct(Math.exp(-10 * lamHat), 2)}</b>` +
      `<br><span style="color:${SC.muted}">The sharper the peak, the larger J, the smaller the standard ` +
      `error. Fisher information IS curvature — raise n and watch the parabola tighten.</span>`;
  }
  sN.addEventListener("input", () => { seed = (seed * 1103515245 + 12345) >>> 0; resample(); handle = null; render(); });
  sT.addEventListener("input", () => { resample(); handle = null; render(); });
  cQ.addEventListener("change", render);
  bNew.addEventListener("click", () => { seed = (seed * 1103515245 + 12345) >>> 0; resample(); handle = null; render(); });
  bSnap.addEventListener("click", () => { handle = null; render(); });
  render();
})();

/* ─────────────────── 3 · a hundred intervals, one fixed truth ─────────────────── */
(function () {
  const svg = d3.select("#cover-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 470;
  const sLvl = document.getElementById("cv-lvl"), sMeth = document.getElementById("cv-method");
  const sN = document.getElementById("cv-n"), oN = document.getElementById("cv-nv");
  const bMore = document.getElementById("cv-more"), bReset = document.getElementById("cv-reset");
  const out = document.getElementById("cover-readout");

  const MU = 100, SD = 15;                       // a fixed, known-to-us population
  let r = ST.rng(770425), shown = [], hits = 0, total = 0, trace = [];

  function draw(n, lvl, meth) {
    let s = 0, s2 = 0;
    for (let i = 0; i < n; i++) { const v = MU + SD * ST.randn(r); s += v; s2 += v * v; }
    const m = s / n, sd = Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1)));
    const a = 1 - lvl;
    let half;
    if (meth === "z") half = ST.normQuant(1 - a / 2) * SD / Math.sqrt(n);
    else if (meth === "t") half = ST.tQuant(1 - a / 2, n - 1) * sd / Math.sqrt(n);
    else half = ST.normQuant(1 - a / 2) * sd / Math.sqrt(n);
    return { lo: m - half, hi: m + half, m: m, ok: (m - half <= MU && MU <= m + half) };
  }
  function batch(k) {
    const n = +sN.value, lvl = +sLvl.value, meth = sMeth.value;
    for (let i = 0; i < k; i++) {
      const c = draw(n, lvl, meth);
      shown.push(c); total++; if (c.ok) hits++;
      if (total % 5 === 0) trace.push({ t: total, p: hits / total });
    }
    if (shown.length > 100) shown = shown.slice(-100);
  }
  function reset() { r = ST.rng(770425); shown = []; hits = 0; total = 0; trace = []; batch(100); }

  function render() {
    oN.textContent = +sN.value;
    const lvl = +sLvl.value, meth = sMeth.value, n = +sN.value;
    const F = ST.frame(svg, W, H, { l: 46, r: 16, t: 22, b: 38 });
    const TOPH = 272, GAP = 42;
    const gT = F.g.append("g"), gB = F.g.append("g").attr("transform", `translate(0,${TOPH + GAP})`);
    const BOTH = F.ih - TOPH - GAP;

    const wid = d3.max(shown, c => c.hi - c.lo) || 1;
    const half = Math.max(wid * 0.9, d3.max(shown, c => Math.abs(c.m - MU)) * 1.15 || 1);
    const x = d3.scaleLinear().domain([MU - half, MU + half]).range([0, F.iw]);
    const rows = shown.length;
    const yr = i => 6 + (TOPH - 34) * (i + 0.5) / Math.max(rows, 1);   // leave a clear strip for the caption

    ST.gridX(gT, x, TOPH, 6);
    gT.selectAll("line.iv").data(shown).join("line").attr("class", "iv")
      .attr("x1", c => x(Math.max(c.lo, x.domain()[0]))).attr("x2", c => x(Math.min(c.hi, x.domain()[1])))
      .attr("y1", (c, i) => yr(i)).attr("y2", (c, i) => yr(i))
      .attr("stroke", c => c.ok ? SC.accent : SC.bad)
      .attr("stroke-width", c => c.ok ? 1.5 : 2.2)
      .attr("stroke-opacity", c => c.ok ? 0.62 : 1);
    gT.selectAll("circle.pt").data(shown).join("circle").attr("class", "pt")
      .attr("cx", c => x(ST.clamp(c.m, x.domain()[0], x.domain()[1]))).attr("cy", (c, i) => yr(i))
      .attr("r", 1.5).attr("fill", c => c.ok ? SC.accent : SC.bad);
    gT.append("line").attr("x1", x(MU)).attr("x2", x(MU)).attr("y1", 0).attr("y2", TOPH)
      .attr("stroke", SC.good).attr("stroke-width", 2.4);
    gT.append("text").attr("x", x(MU) + 6).attr("y", TOPH - 6).attr("font-size", 11).attr("fill", SC.good)
      .text("the true μ = 100 — fixed, and it never moves");
    gT.append("g").attr("class", "axis").attr("transform", `translate(0,${TOPH})`)
      .call(d3.axisBottom(x).ticks(7));
    gT.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.accent)
      .text("the last 100 intervals — each from its own fresh sample of n");
    ST.legend(gT, [{ label: "covers μ", color: SC.accent }, { label: "misses μ", color: SC.bad }],
      F.iw - 148, -12, { vertical: false, gap: 10 });

    /* the running coverage trace */
    const xt = d3.scaleLinear().domain([0, Math.max(100, total)]).range([0, F.iw]);
    const yt = d3.scaleLinear().domain([Math.min(0.75, lvl - 0.14), 1]).range([BOTH, 4]);
    ST.gridY(gB, yt, F.iw, 3);
    gB.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", yt(lvl)).attr("y2", yt(lvl))
      .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "6 4");
    gB.append("text").attr("x", 2).attr("y", yt(lvl) - 5).attr("font-size", 10).attr("fill", SC.good)
      .text(`nominal ${ST.pct(lvl, 0)}`);
    if (trace.length > 1) {
      gB.append("path").datum(trace).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2)
        .attr("d", d3.line().x(q => xt(q.t)).y(q => yt(ST.clamp(q.p, yt.domain()[0], 1))));
    }
    gB.append("g").attr("class", "axis").attr("transform", `translate(0,${BOTH})`)
      .call(d3.axisBottom(xt).ticks(6));
    gB.append("g").attr("class", "axis").call(d3.axisLeft(yt).ticks(3).tickFormat(d3.format(".0%")));
    gB.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.a2)
      .text("running coverage: the fraction of ALL intervals so far that contain μ");
    gB.append("text").attr("x", F.iw).attr("y", BOTH + 30).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("intervals drawn");

    const miss = shown.filter(c => !c.ok).length;
    const label = meth === "z" ? "z interval with the true σ (exact)"
      : meth === "t" ? "t interval with s and t*(n−1) (exact for normal data)"
        : "z interval with s — the common mistake";
    const expect = meth === "zs" ? 2 * ST.tCdf(ST.normQuant(1 - (1 - lvl) / 2), n - 1) - 1 : lvl;
    out.innerHTML =
      `${label} · n = <b>${n}</b> · nominal <b>${ST.pct(lvl, 0)}</b>` +
      `<br>on screen: <b>${shown.length - miss}</b> of ${shown.length} cover μ, <b>${miss}</b> miss` +
      `<br>cumulative over <b>${total.toLocaleString("en-US")}</b> intervals: <b>${ST.pct(hits / total, 2)}</b> ` +
      `coverage (expected ${ST.pct(expect, 2)})` +
      (meth === "zs"
        ? `<br><span style="color:${SC.bad}">Using 1.96 with an estimated σ under-covers: the true coverage ` +
        `here is P(|T${n - 1}| &lt; ${ST.fmt(ST.normQuant(1 - (1 - lvl) / 2), 3)}) = ${ST.pct(expect, 2)}, not ${ST.pct(lvl, 0)}.</span>`
        : `<br><span style="color:${SC.muted}">Each red interval was produced by exactly the same, correct ` +
        `procedure as every blue one. Nothing was done wrong in those samples — a ${ST.pct(1 - lvl, 0)} miss rate is what ` +
        `${ST.pct(lvl, 0)} confidence means.</span>`);
  }
  [sLvl, sMeth, sN].forEach(s => s.addEventListener("input", () => { reset(); render(); }));
  bMore.addEventListener("click", () => { batch(100); render(); });
  bReset.addEventListener("click", () => { reset(); render(); });
  reset(); render();
})();

/* ─────────────────── 4 · t against normal ─────────────────── */
(function () {
  const svg = d3.select("#t-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 440;
  const sDf = document.getElementById("td-df"), oDf = document.getElementById("td-dfv");
  const sLvl = document.getElementById("td-lvl"), cLog = document.getElementById("td-log");
  const out = document.getElementById("t-readout");

  function render() {
    const df = +sDf.value, lvl = +sLvl.value, a = 1 - lvl, logy = cLog.checked;
    oDf.textContent = df;
    const tStar = ST.tQuant(1 - a / 2, df), zStar = ST.normQuant(1 - a / 2);

    const F = ST.frame(svg, W, H, { l: 46, r: 16, t: 26, b: 44 });
    const GAPX = 60, PW = (F.iw - GAPX) / 2, PH = F.ih - 26;
    const gL = F.g.append("g"), gR = F.g.append("g").attr("transform", `translate(${PW + GAPX},0)`);

    /* ── left: the two densities ───────────────────────────────────────── */
    const XL = 5.2;
    const x = d3.scaleLinear().domain([-XL, XL]).range([0, PW]);
    const FLOOR = 2e-4;
    const y = logy
      ? d3.scaleLog().domain([FLOOR, 0.45]).range([PH, 8]).clamp(true)
      : d3.scaleLinear().domain([0, 0.45]).range([PH, 8]);
    ST.gridY(gL, y, PW, 4);
    const gx = ST.linspace(-XL, XL, 420);
    const clip = v => logy ? Math.max(v, FLOOR) : v;

    const tail = (fn, col, op) => {                       // shade both tails beyond ±t*
      const left = gx.filter(v => v <= -tStar).concat([-tStar]);
      const right = [tStar].concat(gx.filter(v => v >= tStar));
      [left, right].forEach(seg => {
        if (seg.length < 2) return;
        gL.append("path").datum(seg.map(v => ({ v: v, d: clip(fn(v)) })))
          .attr("fill", col).attr("fill-opacity", op)
          .attr("d", d3.area().x(q => x(q.v)).y0(PH).y1(q => y(q.d)));
      });
    };
    tail(v => ST.tPdf(v, df), SC.accent, 0.4);
    const lineOf = (fn, col, wdt, dash) =>
      gL.append("path").datum(gx.map(v => ({ v: v, d: clip(fn(v)) })))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", wdt)
        .attr("stroke-dasharray", dash || null)
        .attr("d", d3.line().x(q => x(q.v)).y(q => y(q.d)));
    lineOf(v => ST.normPdf(v), SC.muted, 1.8, "5 3");
    lineOf(v => ST.tPdf(v, df), SC.accent, 2.4);

    [-1, 1].forEach(s => {
      gL.append("line").attr("x1", x(s * tStar)).attr("x2", x(s * tStar))
        .attr("y1", 8).attr("y2", PH).attr("stroke", SC.a2).attr("stroke-width", 1.4);
      gL.append("line").attr("x1", x(s * zStar)).attr("x2", x(s * zStar))
        .attr("y1", 8).attr("y2", PH).attr("stroke", SC.muted).attr("stroke-width", 1)
        .attr("stroke-dasharray", "3 3");
    });
    gL.append("text").attr("x", x(tStar) + 4).attr("y", 20).attr("font-size", 10.5)
      .attr("fill", SC.a2).text(`t* = ${ST.fmt(tStar, 3)}`);
    gL.append("text").attr("x", x(-zStar) - 4).attr("y", 34).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text(`z* = ${ST.fmt(zStar, 3)}`);
    gL.append("g").attr("class", "axis").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(x).ticks(7));
    gL.append("g").attr("class", "axis")
      .call(logy ? d3.axisLeft(y).ticks(4, ".0e") : d3.axisLeft(y).ticks(4));
    gL.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.accent)
      .text(`t(${df}) against the standard normal`);
    gL.append("text").attr("x", PW).attr("y", F.ih + 8).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted)
      .text(logy ? "standardised value (density on a log scale)" : "standardised value");
    ST.legend(gL, [{ label: `t(${df})`, color: SC.accent }, { label: "Normal(0,1)", color: SC.muted, dash: "5 3" }],
      6, PH - 30);

    /* ── right: t* against df ──────────────────────────────────────────── */
    const xd = d3.scaleLog().domain([1, 400]).range([0, PW]);
    const dfs = ST.linspace(0, 1, 160).map(u => Math.pow(400, u));
    const tops = dfs.map(d => ST.tQuant(1 - a / 2, d));
    const yd = d3.scaleLinear().domain([zStar * 0.94, Math.min(d3.max(tops), zStar * 3.4)]).range([PH, 8]);
    ST.gridY(gR, yd, PW, 5);
    gR.append("line").attr("x1", 0).attr("x2", PW).attr("y1", yd(zStar)).attr("y2", yd(zStar))
      .attr("stroke", SC.muted).attr("stroke-width", 1.6).attr("stroke-dasharray", "6 4");
    gR.append("text").attr("x", PW - 2).attr("y", Math.max(16, yd(zStar) - 8)).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text(`z* = ${ST.fmt(zStar, 4)}, the ν → ∞ limit`);
    gR.append("path").datum(dfs.map((d, i) => ({ d: d, t: tops[i] })))
      .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.4)
      .attr("d", d3.line().x(q => xd(q.d)).y(q => yd(ST.clamp(q.t, yd.domain()[0], yd.domain()[1]))));
    gR.append("circle").attr("cx", xd(ST.clamp(df, 1, 400)))
      .attr("cy", yd(ST.clamp(tStar, yd.domain()[0], yd.domain()[1])))
      .attr("r", 6).attr("fill", SC.a2).attr("stroke", SC.bg).attr("stroke-width", 2);
    gR.append("g").attr("class", "axis").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(xd).ticks(5, "~s"));
    gR.append("g").attr("class", "axis").call(d3.axisLeft(yd).ticks(5));
    gR.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", SC.accent)
      .text("the critical value t*(ν), falling to z*");
    gR.append("text").attr("x", PW).attr("y", F.ih + 8).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("degrees of freedom ν (log scale)");

    const naive = 2 * ST.tCdf(zStar, df) - 1;
    const varT = df > 2 ? df / (df - 2) : Infinity;
    out.innerHTML =
      `ν = <b>${df}</b> (so n = ${df + 1}) at <b>${ST.pct(lvl, 0)}</b>: ` +
      `t* = <b>${ST.fmt(tStar, 5)}</b> against z* = ${ST.fmt(zStar, 5)} → the interval is ` +
      `<b>${ST.fmt(100 * (tStar / zStar - 1), 1)}%</b> wider` +
      `<br>tail weight: P(|T| &gt; 3) = <b>${ST.pct(2 * (1 - ST.tCdf(3, df)), 2)}</b> against the normal's ` +
      `${ST.pct(2 * (1 - ST.normCdf(3)), 2)} · Var(T) = ${isFinite(varT) ? ST.fmt(varT, 4) : "infinite"}` +
      `<br>if you used z* = ${ST.fmt(zStar, 3)} with an estimated σ, the actual coverage would be ` +
      `<b style="color:${naive < lvl - 0.005 ? SC.bad : SC.a2}">${ST.pct(naive, 2)}</b>, not ${ST.pct(lvl, 0)}` +
      `<br><span style="color:${SC.muted}">The whole correction is the price of not knowing σ, and it decays ` +
      `like 1/ν: by ν = 30 it is 4.4%, by ν = 100 it is 1.2%, and it never quite reaches zero.</span>`;
  }
  [sDf, sLvl].forEach(s => s.addEventListener("input", render));
  cLog.addEventListener("change", render);
  render();
})();

/* ─────────────────── 5 · exact coverage of a proportion interval ─────────────────── */
(function () {
  const svg = d3.select("#wald-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 430;
  const sN = document.getElementById("wd-n"), oN = document.getElementById("wd-nv");
  const sLvl = document.getElementById("wd-lvl"), sZoom = document.getElementById("wd-zoom");
  const cWil = document.getElementById("wd-wil"), cAc = document.getElementById("wd-ac"),
    cCp = document.getElementById("wd-cp");
  const out = document.getElementById("wald-readout");
  let cursor = 0.10;

  const cpCache = new Map();
  function endpoints(n, z, alpha) {
    const out = { wald: [], wilson: [], ac: [], cp: null };
    for (let k = 0; k <= n; k++) {
      out.wald.push(EST.CI.wald(k, n, z));
      out.wilson.push(EST.CI.wilson(k, n, z));
      out.ac.push(EST.CI.ac(k, n, z));
    }
    if (cCp.checked) {
      const key = n + "|" + alpha;
      if (!cpCache.has(key)) {
        const arr = [];
        for (let k = 0; k <= n; k++) arr.push(EST.CI.cp(k, n, alpha));
        cpCache.set(key, arr);
      }
      out.cp = cpCache.get(key);
    }
    return out;
  }

  function render() {
    const n = +sN.value, lvl = +sLvl.value, alpha = 1 - lvl, z = ST.normQuant(1 - alpha / 2);
    oN.textContent = n;
    const zoom = sZoom.value === "low";
    const P0 = zoom ? 0.002 : 0.004, P1 = zoom ? 0.20 : 0.996;
    const E = endpoints(n, z, alpha);
    const ps = ST.linspace(P0, P1, 720);
    const series = [{ key: "wald", label: "Wald", col: SC.bad, on: true, ep: E.wald }];
    if (cWil.checked) series.push({ key: "wilson", label: "Wilson", col: SC.accent, on: true, ep: E.wilson });
    if (cAc.checked) series.push({ key: "ac", label: "Agresti–Coull", col: SC.a2, on: true, ep: E.ac });
    if (cCp.checked) series.push({ key: "cp", label: "Clopper–Pearson", col: SC.violet, on: true, ep: E.cp });
    series.forEach(s => { s.cov = ps.map(p => EST.coverage(n, p, s.ep)); });

    const F = ST.frame(svg, W, H, { l: 52, r: 16, t: 24, b: 44 });
    const x = d3.scaleLinear().domain([P0, P1]).range([0, F.iw]);
    const rawMin = d3.min(series, s => d3.min(s.cov));
    const FLOORY = Math.max(0.55, Math.min(rawMin - 0.03, lvl - 0.12));   // keep the 90-100% band readable
    const y = d3.scaleLinear().domain([FLOORY, 1.035]).range([F.ih, 8]);
    ST.gridY(F.g, y, F.iw, 5);

    F.g.append("rect").attr("x", 0).attr("y", y(1)).attr("width", F.iw)
      .attr("height", Math.max(0, y(lvl) - y(1))).attr("fill", SC.good).attr("fill-opacity", 0.055);
    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(lvl)).attr("y2", y(lvl))
      .attr("stroke", SC.good).attr("stroke-width", 1.8).attr("stroke-dasharray", "7 4");
    F.g.append("text").attr("x", F.iw - 3).attr("y", y(1.021)).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.good)
      .text(`nominal ${ST.pct(lvl, 0)} — below the dashed line the interval under-covers`);
    if (rawMin < FLOORY) F.g.append("text").attr("x", F.iw / 2).attr("y", F.ih - 7)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", SC.bad)
      .text(`clipped at ${ST.pct(FLOORY, 0)} — the Wald coverage dives to ${ST.pct(rawMin, 1)} near the edges`);

    series.forEach(s => {
      F.g.append("path").datum(ps.map((p, i) => ({ p: p, c: s.cov[i] })))
        .attr("fill", "none").attr("stroke", s.col).attr("stroke-width", s.key === "wald" ? 1.7 : 1.5)
        .attr("stroke-opacity", 0.95)
        .attr("d", d3.line().x(q => x(q.p)).y(q => y(ST.clamp(q.c, y.domain()[0], y.domain()[1]))));
    });

    const cu = ST.clamp(cursor, P0, P1);
    F.g.append("line").attr("x1", x(cu)).attr("x2", x(cu)).attr("y1", 8).attr("y2", F.ih)
      .attr("stroke", SC.ink).attr("stroke-width", 1.2);
    series.forEach(s => {
      const c = EST.coverage(n, cu, s.ep);
      F.g.append("circle").attr("cx", x(cu)).attr("cy", y(ST.clamp(c, y.domain()[0], y.domain()[1])))
        .attr("r", 4).attr("fill", s.col).attr("stroke", SC.bg).attr("stroke-width", 1.5);
    });

    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).ticks(8).tickFormat(d3.format(".2f")));
    F.g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).tickValues(d3.ticks(FLOORY, 1, 6)).tickFormat(d3.format(".0%")));
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.accent)
      .text("exact coverage probability — computed by summing the binomial, not simulated");
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 34).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("the true proportion p");
    ST.legend(F.g, series.map(s => ({ label: s.label, color: s.col })), 8, y(1.021),
      { vertical: false, gap: 12 });

    F.g.append("rect").attr("class", "dragpt").attr("x", 0).attr("y", 0)
      .attr("width", F.iw).attr("height", F.ih).attr("fill", "transparent").style("cursor", "ew-resize")
      .call(d3.drag().on("start drag", ev => {
        cursor = ST.clamp(x.invert(d3.pointer(ev, F.g.node())[0]), P0, P1); render();
      }));

    /* summary statistics over the "sane" band, and at the cursor */
    const band = ST.linspace(0.05, 0.95, 400);
    const stat = s => {
      const c = band.map(p => EST.coverage(n, p, s.ep));
      return { mean: ST.mean(c), min: d3.min(c) };
    };
    const wStat = stat(series[0]);
    const at = series.map(s => `${s.label} <b>${ST.pct(EST.coverage(n, cu, s.ep), 1)}</b>`).join(" · ");
    const p0 = Math.pow(1 - cu, n);
    out.innerHTML =
      `n = <b>${n}</b> · nominal <b>${ST.pct(lvl, 0)}</b> · at p = <b>${ST.fmt(cu, 4)}</b>: ${at}` +
      `<br>Wald over p ∈ [0.05, 0.95]: mean coverage <b>${ST.pct(wStat.mean, 1)}</b>, worst <b>${ST.pct(wStat.min, 1)}</b>` +
      (cWil.checked ? ` · Wilson: mean <b>${ST.pct(stat(series.find(s => s.key === "wilson")).mean, 1)}</b>, ` +
        `worst <b>${ST.pct(stat(series.find(s => s.key === "wilson")).min, 1)}</b>` : "") +
      `<br>P(no successes at all) = (1 − p)ⁿ = <b>${ST.pct(p0, 2)}</b> — and when that happens the Wald interval ` +
      `is the single point [0, 0], which covers nothing` +
      `<br><span style="color:${SC.muted}">The saw teeth are exact arithmetic, not noise: as p creeps up, one ` +
      `value of the count abruptly stops covering and its whole binomial probability drops out of the sum.</span>`;
  }
  [sN, sLvl, sZoom].forEach(s => s.addEventListener("input", render));
  [cWil, cAc, cCp].forEach(c => c.addEventListener("change", render));
  render();
})();

/* ─────────────────── 6 · the sample-size planner ─────────────────── */
(function () {
  const svg = d3.select("#plan-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 410;
  const sMode = document.getElementById("pl-mode");
  const sM = document.getElementById("pl-m"), oM = document.getElementById("pl-mv");
  const sLvl = document.getElementById("pl-lvl");
  const lP = document.getElementById("pl-plab"), sP = document.getElementById("pl-p"), oP = document.getElementById("pl-pv");
  const lS = document.getElementById("pl-slab"), sSd = document.getElementById("pl-sd"), oSd = document.getElementById("pl-sdv");
  const sN = document.getElementById("pl-N"), oNv = document.getElementById("pl-Nv");
  const out = document.getElementById("plan-readout");

  function render() {
    const prop = sMode.value === "prop";
    lP.style.display = prop ? "" : "none";
    lS.style.display = prop ? "none" : "";
    const lvl = +sLvl.value, z = ST.normQuant(1 - (1 - lvl) / 2);
    const p = +sP.value, sd = +sSd.value;
    const Nexp = +sN.value, N = Nexp >= 7.95 ? Infinity : Math.round(Math.pow(10, Nexp));
    oP.textContent = ST.fmt(p, 2); oSd.textContent = sd;
    oNv.textContent = isFinite(N) ? N.toLocaleString("en-US") : "∞ (ignore)";

    /* the margin is in POINTS for a proportion, in raw units for a mean */
    const m = prop ? (+sM.value) / 100 : (+sM.value) * sd / 20;
    oM.textContent = prop ? ST.fmt(+sM.value, 1) + " pts" : ST.fmt(m, 2) + " units";

    const unitSd = prop ? Math.sqrt(p * (1 - p)) : sd;
    const marginAt = nn => {
      const base = z * unitSd / Math.sqrt(nn);
      return isFinite(N) ? base * Math.sqrt(Math.max(0, (N - nn) / (N - 1))) : base;
    };
    const nInf = Math.pow(z * unitSd / m, 2);
    const nNeed = isFinite(N) ? nInf / (1 + (nInf - 1) / N) : nInf;

    const F = ST.frame(svg, W, H, { l: 60, r: 16, t: 24, b: 46 });
    const NMAX = isFinite(N) ? Math.min(N, 200000) : 200000;
    const x = d3.scaleLog().domain([5, NMAX]).range([0, F.iw]);
    const ns = ST.linspace(Math.log10(5), Math.log10(NMAX), 240).map(u => Math.pow(10, u));
    const ms = ns.map(marginAt).filter(v => v > 0);
    const y = d3.scaleLog().domain([Math.max(1e-5, Math.min(m, d3.min(ms)) * 0.6), z * unitSd / Math.sqrt(5) * 1.2])
      .range([F.ih, 8]);
    ST.gridY(F.g, y, F.iw, 5); ST.gridX(F.g, x, F.ih, 6);

    const plain = ns.map(nn => ({ n: nn, m: z * unitSd / Math.sqrt(nn) }));
    F.g.append("path").datum(plain).attr("fill", "none")
      .attr("stroke", isFinite(N) ? SC.muted : SC.accent)
      .attr("stroke-width", isFinite(N) ? 1.6 : 2.4).attr("stroke-dasharray", isFinite(N) ? "5 3" : null)
      .attr("d", d3.line().x(q => x(q.n)).y(q => y(ST.clamp(q.m, y.domain()[0], y.domain()[1]))));
    if (isFinite(N)) {
      const corr = ns.filter(nn => nn <= N).map(nn => ({ n: nn, m: marginAt(nn) })).filter(q => q.m > y.domain()[0]);
      F.g.append("path").datum(corr).attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2.4)
        .attr("d", d3.line().x(q => x(q.n)).y(q => y(ST.clamp(q.m, y.domain()[0], y.domain()[1]))));
    }

    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(m)).attr("y2", y(m))
      .attr("stroke", SC.a2).attr("stroke-width", 1.8).attr("stroke-dasharray", "7 4");
    F.g.append("text").attr("x", 3).attr("y", y(m) - 5).attr("font-size", 10.5).attr("fill", SC.a2)
      .text(`target margin ${prop ? ST.fmt(100 * m, 2) + " points" : "±" + ST.fmt(m, 3)}`);

    if (nNeed >= 5 && nNeed <= NMAX) {
      F.g.append("line").attr("x1", x(nNeed)).attr("x2", x(nNeed)).attr("y1", y(m)).attr("y2", F.ih)
        .attr("stroke", SC.good).attr("stroke-width", 1.8);
      F.g.append("circle").attr("cx", x(nNeed)).attr("cy", y(m)).attr("r", 6)
        .attr("fill", SC.good).attr("stroke", SC.bg).attr("stroke-width", 2);
      F.g.append("text").attr("x", x(nNeed) + 8).attr("y", F.ih - 8).attr("font-size", 11).attr("fill", SC.good)
        .text(`n = ${Math.ceil(nNeed).toLocaleString("en-US")}`);
    }
    /* the fourfold-cost markers */
    [1, 2, 4].forEach(k => {
      const mm = m * Math.pow(2, k - 1), nn = Math.pow(z * unitSd / mm, 2);
      if (nn < 5 || nn > NMAX) return;
      F.g.append("circle").attr("cx", x(nn)).attr("cy", y(mm)).attr("r", 3).attr("fill", SC.muted);
    });

    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).ticks(6, "~s"));
    F.g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).ticks(6, prop ? ".1%" : "~g"));
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.accent)
      .text("margin of error against sample size — a straight line of slope −1/2 on log–log");
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 36).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("sample size n (log scale)");
    if (isFinite(N)) ST.legend(F.g, [
      { label: "with the finite-population correction", color: SC.accent },
      { label: "ignoring it", color: SC.muted, dash: "5 3" }
    ], 8, 16);

    const worst = prop ? Math.pow(z / (2 * m), 2) : null;
    out.innerHTML = prop
      ? `a proportion at <b>${ST.pct(lvl, 0)}</b>, target margin <b>±${ST.fmt(100 * m, 2)}</b> points, ` +
      `assumed p = <b>${ST.fmt(p, 2)}</b> → n = z*²p(1−p)/m² = <b>${Math.ceil(nInf).toLocaleString("en-US")}</b>` +
      (Math.abs(p - 0.5) < 1e-9
        ? `<br>that already IS the conservative choice: p(1−p) ≤ ¼, so n = (z*/2m)² = ` +
          `<b>${Math.ceil(worst).toLocaleString("en-US")}</b> is an upper bound valid for EVERY p`
        : `<br>with no pilot at all, use the worst case p = ½: n = (z*/2m)² = ` +
          `<b>${Math.ceil(worst).toLocaleString("en-US")}</b> — ` +
          `<b>${ST.fmt(worst / nInf, 2)}×</b> more than the pilot-informed answer`) +
      (isFinite(N) ? `<br>sampling from a finite population of ${N.toLocaleString("en-US")}: the requirement drops ` +
        `to <b>${Math.ceil(nNeed).toLocaleString("en-US")}</b>, a saving of ` +
        `${ST.pct(1 - nNeed / nInf, 1)}` : "") +
      `<br>halving this margin to ±${ST.fmt(50 * m, 2)} points would need ` +
      `<b>${Math.ceil(4 * nInf).toLocaleString("en-US")}</b> — four times the data for twice the precision` +
      `<br><span style="color:${SC.muted}">Every number here prices the SAMPLING gap only. Non-response, ` +
      `coverage error and measurement bias are not in the formula and are frequently larger.</span>`
      : `a mean at <b>${ST.pct(lvl, 0)}</b>, target margin <b>±${ST.fmt(m, 3)}</b>, assumed σ = <b>${sd}</b> → ` +
      `n = (z*σ/m)² = <b>${Math.ceil(nInf).toLocaleString("en-US")}</b>` +
      `<br>the analysis will use t*, not z*, so at this n add one iteration: ` +
      `t*(${Math.max(1, Math.ceil(nInf) - 1)}) = ${ST.fmt(ST.tQuant(1 - (1 - lvl) / 2, Math.max(1, Math.ceil(nInf) - 1)), 4)} → ` +
      `n = <b>${Math.ceil(Math.pow(ST.tQuant(1 - (1 - lvl) / 2, Math.max(1, Math.ceil(nInf) - 1)) * sd / m, 2)).toLocaleString("en-US")}</b>` +
      (isFinite(N) ? `<br>with a finite population of ${N.toLocaleString("en-US")}: <b>${Math.ceil(nNeed).toLocaleString("en-US")}</b>` : "") +
      `<br>halving the margin needs <b>${Math.ceil(4 * nInf).toLocaleString("en-US")}</b>` +
      `<br><span style="color:${SC.muted}">σ is the ingredient you do not have. Use a pilot, a previous study, ` +
      `or range/4 — and inflate it, because a pilot's own s is noisy.</span>`;
  }
  [sMode, sM, sLvl, sP, sSd, sN].forEach(s => s.addEventListener("input", render));
  render();
})();

/* ─────────────────── 7 · the four intervals on one axis ─────────────────── */
(function () {
  const svg = d3.select("#four-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 740, H = 420;
  const sN = document.getElementById("fr-n"), oN = document.getElementById("fr-nv");
  const sM0 = document.getElementById("fr-m0"), oM0 = document.getElementById("fr-m0v");
  const sTau = document.getElementById("fr-tau"), oTau = document.getElementById("fr-tauv");
  const sP = document.getElementById("fr-P"), oP = document.getElementById("fr-Pv");
  const bNew = document.getElementById("fr-new");
  const out = document.getElementById("four-readout");

  const MU = 0, SD = 1;                          // standardised population, so the axis reads in σ
  let seed = 60313, stats = null;

  function resample() {
    const n = +sN.value, r = ST.rng(seed);
    let s = 0, s2 = 0;
    for (let i = 0; i < n; i++) { const v = MU + SD * ST.randn(r); s += v; s2 += v * v; }
    const m = s / n;
    stats = { n: n, m: m, s: Math.sqrt(Math.max(1e-9, (s2 - n * m * m) / (n - 1))) };
  }

  function render() {
    if (!stats || stats.n !== +sN.value) resample();
    const n = stats.n, xb = stats.m, s = stats.s;
    const m0 = +sM0.value, tau = Math.pow(10, +sTau.value), P = +sP.value;
    oN.textContent = n; oM0.textContent = ST.fmt(m0, 1);
    oTau.textContent = tau >= 15 ? "flat" : ST.fmt(tau, 2);
    oP.textContent = ST.pct(P, 0);

    const z = ST.normQuant(0.975), t = ST.tQuant(0.975, n - 1);
    const ci = t * s / Math.sqrt(n);
    const pi = t * s * Math.sqrt(1 + 1 / n);
    const kTol = ST.normQuant((1 + P) / 2) * Math.sqrt((n - 1) * (1 + 1 / n) / ST.chi2Quant(0.05, n - 1));
    const ti = kTol * s;
    const prec0 = 1 / (tau * tau), precN = n / (SD * SD);       // σ treated as known for the posterior
    const postM = (m0 * prec0 + xb * precN) / (prec0 + precN);
    const postS = 1 / Math.sqrt(prec0 + precN);
    const cr = z * postS;

    const F = ST.frame(svg, W, H, { l: 34, r: 16, t: 22, b: 44 });
    const DENSH = 118, ROWH = 46, ROW0 = DENSH + 34;
    const half = Math.max(ti, pi, Math.abs(m0 - xb) + cr, 3.4) * 1.10;
    const x = d3.scaleLinear().domain([xb - half, xb + half]).range([0, F.iw]);

    /* the population density, for scale */
    const gx = ST.linspace(x.domain()[0], x.domain()[1], 300);
    const yd = d3.scaleLinear().domain([0, ST.normPdf(0) / SD * 1.12]).range([DENSH, 6]);
    F.g.append("path").datum(gx.map(v => ({ v: v, d: ST.normPdf((v - MU) / SD) / SD })))
      .attr("fill", SC.muted).attr("fill-opacity", 0.13)
      .attr("d", d3.area().x(q => x(q.v)).y0(DENSH).y1(q => yd(q.d)));
    F.g.append("path").datum(gx.map(v => ({ v: v, d: ST.normPdf((v - MU) / SD) / SD })))
      .attr("fill", "none").attr("stroke", SC.muted).attr("stroke-width", 1.6)
      .attr("d", d3.line().x(q => x(q.v)).y(q => yd(q.d)));
    F.g.append("line").attr("x1", x(MU)).attr("x2", x(MU)).attr("y1", 4).attr("y2", F.ih)
      .attr("stroke", SC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 4");
    F.g.append("text").attr("x", x(MU) + 5).attr("y", 14).attr("font-size", 10.5).attr("fill", SC.good)
      .text("true μ");
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", SC.muted)
      .text("the population itself — the thing three of these four intervals are NOT about");

    const bars = [
      { c: xb, h: ci, col: SC.accent, t: "① confidence — the mean", g: `x̄ ± t*·s/√n = ±${ST.fmt(ci, 3)}` },
      { c: xb, h: pi, col: SC.a2, t: "② prediction — one future value", g: `x̄ ± t*·s·√(1+1/n) = ±${ST.fmt(pi, 3)}` },
      { c: xb, h: ti, col: SC.bad, t: `③ tolerance — ${ST.pct(P, 0)} of the population`, g: `x̄ ± k·s = ±${ST.fmt(ti, 3)}` },
      { c: postM, h: cr, col: SC.violet, t: "④ credible — belief about μ", g: `posterior ${ST.fmt(postM, 3)} ± ${ST.fmt(cr, 3)}` }
    ];
    bars.forEach((b, i) => {
      const yy = ROW0 + i * ROWH;
      F.g.append("line").attr("x1", x(ST.clamp(b.c - b.h, x.domain()[0], x.domain()[1])))
        .attr("x2", x(ST.clamp(b.c + b.h, x.domain()[0], x.domain()[1])))
        .attr("y1", yy).attr("y2", yy).attr("stroke", b.col).attr("stroke-width", 7)
        .attr("stroke-opacity", 0.55).attr("stroke-linecap", "round");
      F.g.append("circle").attr("cx", x(ST.clamp(b.c, x.domain()[0], x.domain()[1]))).attr("cy", yy)
        .attr("r", 3.2).attr("fill", b.col);
      F.g.append("text").attr("x", 2).attr("y", yy - 11).attr("font-size", 11).attr("fill", b.col).text(b.t);
      F.g.append("text").attr("x", F.iw).attr("y", yy - 11).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", SC.muted).text(b.g);
    });
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).ticks(8));
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 36).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", SC.muted).text("value, in units of the population σ");

    out.innerHTML =
      `n = <b>${n}</b> · this sample: x̄ = ${ST.fmt(xb, 4)}, s = ${ST.fmt(s, 4)} · all four at 95%` +
      `<br>① confidence half-width <b>${ST.fmt(ci, 4)}</b> · ② prediction <b>${ST.fmt(pi, 4)}</b> · ` +
      `③ tolerance (${ST.pct(P, 0)} of units, 95% conf., k = ${ST.fmt(kTol, 3)}) <b>${ST.fmt(ti, 4)}</b> · ` +
      `④ credible <b>${ST.fmt(cr, 4)}</b>` +
      `<br>prediction ÷ confidence = <b>${ST.fmt(pi / ci, 2)}×</b> — and this ratio grows like √n without limit, ` +
      `because only ① shrinks to a point` +
      `<br>prior Normal(${ST.fmt(m0, 2)}, ${tau >= 15 ? "flat" : ST.fmt(tau, 2) + "²"}) → posterior mean ` +
      `<b>${ST.fmt(postM, 4)}</b>: the prior pulls the credible interval ` +
      `${Math.abs(postM - xb) < 0.005 ? "essentially not at all" : ST.fmt(Math.abs(postM - xb), 3) + " away from x̄"}` +
      `<br><span style="color:${SC.muted}">With a flat prior ④ coincides numerically with ①, which is exactly ` +
      `why the two are confused — identical numbers, different claims. Tighten τ and watch them separate.</span>`;
  }
  sN.addEventListener("input", () => { resample(); render(); });
  [sM0, sTau, sP].forEach(s => s.addEventListener("input", render));
  bNew.addEventListener("click", () => { seed = (seed * 1103515245 + 12345) >>> 0; resample(); render(); });
  render();
})();
